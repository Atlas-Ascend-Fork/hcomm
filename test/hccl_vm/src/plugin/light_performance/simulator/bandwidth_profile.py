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
  multi_port_utilization_low:  共用端口传输数 < 8 时使用
  multi_port_utilization_high: 共用端口传输数 >= 8 时使用

单端口场景 (eid 查出仅 1 个端口) 使用固定利用率:
  single_port_utilization: 常量利用率 (0~1), 默认 1.0

配置文件格式 (JSON):
{
  "multi_port_utilization_low":  { "32": 0.50, "64": 0.55, ... },
  "multi_port_utilization_high": { "524288": 0.217, ... },
  "single_port_utilization": 1.0
}
key 为数据量 (bytes, str), value 为利用率 (0~1)。
"""

from __future__ import annotations

import json
from typing import Optional


# 默认数据量序列: 32B, 64B, 128B, ..., 128MB
_DEFAULT_SIZES = [32 * (2 ** i) for i in range(23)]  # 32 到 134217728


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
    """

    THRESHOLD = 7  # 共用端口的传输数阈值

    __slots__ = ("profile_low", "profile_high",
                 "single_port_utilization", "constant_utilization")

    def __init__(self, profile_low: Optional[BandwidthProfile] = None,
                 profile_high: Optional[BandwidthProfile] = None,
                 single_port_utilization: float = 1.0,
                 constant_utilization: Optional[float] = None):
        self.profile_low = profile_low or BandwidthProfile.default()
        self.profile_high = profile_high or BandwidthProfile.default()
        self.single_port_utilization = single_port_utilization
        self.constant_utilization = constant_utilization

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
        )

    def to_dict(self) -> dict:
        d = {
            "multi_port_utilization_low": self.profile_low.to_dict(),
            "multi_port_utilization_high": self.profile_high.to_dict(),
            "single_port_utilization": self.single_port_utilization,
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
                f"single_port={self.single_port_utilization})")
