#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""带宽利用率配置模块。

数据量从 32B 到 128MB (按 2 的倍数递增), 每个数据量对应一个带宽利用率 (0~1)。
数据传输任务的时延 = 数据量 / (端口带宽 × 对应数据量的利用率)

多端口场景 (eid 查出 >1 个端口) 使用两个配置:
  multi_port_utilization_low:  共用端口传输数 < 7 时使用
  multi_port_utilization_high: 共用端口传输数 >= 7 时使用

单端口场景 (eid 查出仅 1 个端口) 使用固定利用率:
  single_port_utilization: 常量利用率 (0~1), 默认 1.0

配置文件格式 (JSON):
{
  "multi_port_utilization_low":  { "32": 0.50, "64": 0.55, ... },
  "multi_port_utilization_high": { "524288": 0.217, ... },
  "single_port_utilization": 1.0,
  "jetty_utilization_step": 0.02745
}
key 为数据量 (bytes, str), value 为利用率 (0~1)。

多 jetty 折算 (jetty_utilization_step):
  常规配置是一个 channel 一个 jetty, multi_port_utilization_low 即按该前提标定;
  HCCL_UB_MULTI_CHANNEL_NUM=n 时一个 channel 挂 n 个 jetty, 并发搬运能力近似
  线性提升。轻量仿真不建 jetty 级模型, 由 for_jetty_num() 折算为 profile_low
  表的整体抬升: util' = min(1.0, util + (n - 1) × step), n<=1 时原样用表值。
  step 取值范围 [0.0109, 0.044], 缺省取中值 0.02745, 越界告警后夹取。
"""

from __future__ import annotations

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# 默认数据量序列: 32B, 64B, 128B, ..., 128MB
_DEFAULT_SIZES = [32 * (2 ** i) for i in range(23)]  # 32 到 134217728

# 多 jetty 折算步长: 每多一个 jetty 为 multi_port_utilization_low 各档带来的
# 利用率增量, 取值范围 [0.0109, 0.044], 未配置时取中值
JETTY_STEP_MIN = 0.0109
JETTY_STEP_MAX = 0.044
JETTY_STEP_DEFAULT = 0.02745


def _clamp_jetty_step(step: float) -> float:
    """把 jetty 折算步长夹取到 [JETTY_STEP_MIN, JETTY_STEP_MAX], 越界告警。"""
    value = float(step)
    if value < JETTY_STEP_MIN or value > JETTY_STEP_MAX:
        logger.warning(
            "jetty_utilization_step=%s out of range [%s, %s], clamped before use",
            value, JETTY_STEP_MIN, JETTY_STEP_MAX)
        return min(max(value, JETTY_STEP_MIN), JETTY_STEP_MAX)
    return value


class BandwidthProfile:
    """带宽利用率表: 数据量 → 利用率。

    查询时按最接近的数据量档位匹配 (不要求精确匹配)。
    """

    __slots__ = ("_table", "_sorted_sizes")

    def __init__(self, table: Optional[dict[int, float]] = None):
        self._table: dict[int, float] = {}
        self._sorted_sizes: list[int] = []
        if table:
            for k, v in table.items():
                self._table[int(k)] = float(v)
            self._sorted_sizes = sorted(self._table.keys())

    def get_utilization(self, data_size: float) -> float:
        """查询数据量对应的带宽利用率。

        按最接近的档位匹配 (向下取整到最近的配置档位)。
        """
        if not self._sorted_sizes:
            return 1.0  # 无配置, 利用率为 100%

        ds = int(data_size)
        # 找到 <= ds 的最大档位
        sizes = self._sorted_sizes
        if ds <= sizes[0]:
            return self._table[sizes[0]]
        if ds >= sizes[-1]:
            return self._table[sizes[-1]]

        # 二分查找
        lo, hi = 0, len(sizes) - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if sizes[mid] <= ds:
                lo = mid
            else:
                hi = mid
        return self._table[sizes[lo]]

    def shifted(self, delta: float) -> "BandwidthProfile":
        """返回每档利用率整体抬升 delta 的新表 (逐档夹取到 [0, 1])。

        delta 为 0 时返回自身 (多 jetty 折算 n=1 的快路径)。
        """
        if not delta:
            return self
        return BandwidthProfile({
            size: min(1.0, max(0.0, util + delta))
            for size, util in self._table.items()})

    def to_dict(self) -> dict:
        return {str(k): v for k, v in sorted(self._table.items())}

    @classmethod
    def from_dict(cls, d: dict) -> "BandwidthProfile":
        return cls({int(k): float(v) for k, v in d.items()})

    @classmethod
    def from_json(cls, filepath: str) -> "BandwidthProfile":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def default(cls) -> "BandwidthProfile":
        """默认利用率: 全部为 1.0 (无降效)。"""
        return cls({s: 1.0 for s in _DEFAULT_SIZES})

    def __len__(self) -> int:
        return len(self._table)

    def __repr__(self) -> str:
        return f"BandwidthProfile(entries={len(self._table)})"


class BandwidthProfileSet:
    """带宽利用率配置集。

    多端口场景: 按 concurrency 阈值选择 profile_low / profile_high 查表。
    单端口场景: single_port_utilization 为固定常量, 通过 for_single_port() 创建
        常量模式实例 (constant_utilization 非 None 时 get_utilization 直接返回它)。
    多 jetty 场景: for_jetty_num() 按 HCCL_UB_MULTI_CHANNEL_NUM 抬升 profile_low
        (每多一个 jetty 加一个 jetty_utilization_step, 详见模块 docstring)。
    """

    THRESHOLD = 7  # 共用端口的传输数阈值

    __slots__ = ("profile_low", "profile_high",
                 "single_port_utilization", "constant_utilization",
                 "jetty_utilization_step")

    def __init__(self, profile_low: Optional[BandwidthProfile] = None,
                 profile_high: Optional[BandwidthProfile] = None,
                 single_port_utilization: float = 1.0,
                 constant_utilization: Optional[float] = None,
                 jetty_utilization_step: float = JETTY_STEP_DEFAULT):
        self.profile_low = profile_low or BandwidthProfile.default()
        self.profile_high = profile_high or BandwidthProfile.default()
        self.single_port_utilization = single_port_utilization
        self.constant_utilization = constant_utilization
        self.jetty_utilization_step = _clamp_jetty_step(jetty_utilization_step)

    def get_utilization(self, concurrency: int, data_size: float) -> float:
        """根据并发数和数据量查询利用率。

        常量模式 (constant_utilization 非 None, 由 for_single_port 创建) 直接返回常量。
        否则按多端口逻辑: concurrency < THRESHOLD 用 profile_low, 否则 profile_high。
        """
        if self.constant_utilization is not None:
            return self.constant_utilization
        if concurrency < self.THRESHOLD:
            return self.profile_low.get_utilization(data_size)
        else:
            return self.profile_high.get_utilization(data_size)

    def for_jetty_num(self, jetty_num: int) -> "BandwidthProfileSet":
        """按 jetty 数 (HCCL_UB_MULTI_CHANNEL_NUM) 折算出本次仿真用的配置集。

        util' = min(1.0, util + (jetty_num - 1) × jetty_utilization_step),
        仅作用于 multi_port_utilization_low 表 (profile_high 为高并发档, 端口已被
        多路传输占满, 多 jetty 不再额外增益; 单端口/d2h 走常量模式, 无表可抬升)。

        jetty_num <= 1 (env 未配置/非法回落 1) 或常量模式时原样返回 self。
        """
        if self.constant_utilization is not None:
            return self
        num = int(jetty_num or 1)
        if num <= 1:
            return self
        delta = (num - 1) * self.jetty_utilization_step
        logger.info("multi-jetty scaling: jetty_num=%d, step=%s, "
                    "multi_port_utilization_low each tier +%.6f (upper bound 1.0)",
                    num, self.jetty_utilization_step, delta)
        return BandwidthProfileSet(
            profile_low=self.profile_low.shifted(delta),
            profile_high=self.profile_high,
            single_port_utilization=self.single_port_utilization,
            jetty_utilization_step=self.jetty_utilization_step)

    @classmethod
    def for_single_port(cls, utilization: float) -> "BandwidthProfileSet":
        """创建单端口常量利用率配置集。

        Args:
            utilization: 固定带宽利用率 (0~1), 忽略并发数和数据量
        """
        return cls(profile_low=None, profile_high=None,
                   constant_utilization=utilization)

    @classmethod
    def from_dict(cls, d: dict) -> "BandwidthProfileSet":
        return cls(
            profile_low=BandwidthProfile.from_dict(
                d.get("multi_port_utilization_low", d.get("profile_low", {}))),
            profile_high=BandwidthProfile.from_dict(
                d.get("multi_port_utilization_high", d.get("profile_high", {}))),
            single_port_utilization=float(d.get("single_port_utilization", 1.0)),
            jetty_utilization_step=float(
                d.get("jetty_utilization_step", JETTY_STEP_DEFAULT)),
        )

    def to_dict(self) -> dict:
        d = {
            "multi_port_utilization_low": self.profile_low.to_dict(),
            "multi_port_utilization_high": self.profile_high.to_dict(),
            "single_port_utilization": self.single_port_utilization,
            "jetty_utilization_step": self.jetty_utilization_step,
        }
        return d

    @classmethod
    def from_json(cls, filepath: str) -> "BandwidthProfileSet":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def __repr__(self) -> str:
        return (f"BandwidthProfileSet(low={self.profile_low}, "
                f"high={self.profile_high}, "
                f"single_port={self.single_port_utilization}, "
                f"jetty_step={self.jetty_utilization_step})")
