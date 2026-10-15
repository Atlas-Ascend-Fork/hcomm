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

"""配置模块: 固定时延、带宽、带宽利用率等仿真参数。

JSON 配置单位 (在 JSON 文件中用注释标注):
  时延: us (微秒)   — 1 us = 1e-6 s
  带宽: GBps (字节) — 1 GBps = 1e9 B/s

端口标识格式 (参考 ranktable.json):
  "die_id/port_number", 如 "1/0", "0/7"
  一个 eid 可对应多个端口, ports 为列表: ["0/0", "0/1", ..., "0/7"]
  配置按端口编号逐个配带宽, 多端口时总带宽 = 各端口带宽之和。
"""

from __future__ import annotations

import logging

from dataclasses import dataclass, field
from typing import Optional, Union

from .bandwidth_profile import BandwidthProfileSet

_logger = logging.getLogger(__name__)

_GBPS_TO_BPS = 1e9
_US_TO_S = 1e-6


@dataclass
class SimConfig:
    """仿真配置参数。

    内部单位: 时延=秒, 带宽=bytes/s。
    JSON 配置单位: 时延=us, 带宽=GBps (from_dict 自动转换)。

    Attributes:
        topo_type: 拓扑类型, 决定使用哪组端口带宽配置 ("8p_server" 或 "1d_pod")
        port_bandwidth: {"die/port": 带宽 (bytes/s)}, 当前 topo_type 选中的端口带宽
        all_port_bandwidths: {"topo_type": {"die/port": 带宽}}, 全部拓扑的端口带宽配置
        default_port_bandwidth: 未配置端口的默认带宽 (bytes/s)
        hbm_bandwidth: HBM 总带宽 (bytes/s), 用于 Reduce_Inline / Memcpy
        hbm_utilization: HBM 带宽利用率, 有效带宽 = hbm_bandwidth × hbm_utilization
        hbm_latency: HBM 固定时延 (秒), 默认 50ns, 用于 Reduce_Inline / Memcpy
        bw_profile_set: 带宽利用率配置集 (含多端口 profile_low/high + 单端口 single_port_utilization)
        d2h_bandwidth: DPU d2h (Device-to-Host) 端口带宽 (bytes/s), 用于 DPU 数据传输任务
        d2h_utilization: d2h 带宽利用率, 有效带宽 = d2h_bandwidth × d2h_utilization;
            d2h 默认代表一个端口, 带宽和利用率需配置为单卡可使用的 DPU 带宽
    """
    # 时延 (JSON 配置单位: us)
    local_record_delay: float = 30e-9         # local_record 固定耗时 (秒), 默认 30ns, JSON 值单位: us
    remote_record_delay: float = 1e-6         # remote_record 固定耗时 (秒), JSON 值单位: us
    local_wait_delay: float = 30e-9           # local_wait 固定耗时 (秒), 默认 30ns, JSON 值单位: us
    remote_wait_delay: float = 1e-6           # remote_wait 固定耗时 (秒), JSON 值单位: us
    hbm_latency: float = 50e-9               # HBM 固定时延 (秒), 默认 50ns, JSON 值单位: us
    # 带宽 (JSON 配置单位: GBps)
    topo_type: str = "8p_server"              # 拓扑类型, 决定使用哪组端口带宽
    port_bandwidth: dict[str, float] = field(default_factory=dict)  # 当前生效的端口带宽
    all_port_bandwidths: dict[str, dict[str, float]] = field(default_factory=dict)  # 全部拓扑的端口带宽
    default_port_bandwidth: float = 53.125e9  # 53.125 GB/s, JSON 值单位: GBps
    hbm_bandwidth: float = 4e12             # 4000 GB/s, JSON 值单位: GBps
    hbm_utilization: float = 0.7            # HBM 带宽利用率, 有效带宽 = hbm_bandwidth × hbm_utilization
    bw_profile_set: Optional[BandwidthProfileSet] = None
    # DPU d2h (Device-to-Host) 端口带宽与利用率, 用于 DPU 数据传输任务;
    # d2h 默认代表一个端口, 带宽和利用率需配置为单卡可使用的 DPU 带宽
    d2h_bandwidth: float = 32e9            # 32 GB/s, JSON 值单位: GBps
    d2h_utilization: float = 0.8          # d2h 带宽利用率

    @classmethod
    def from_dict(cls, d: dict) -> "SimConfig":
        """从 JSON 字典加载, 时延按 us→s 转换, 带宽按 GBps→B/s 转换。

        端口带宽按 topo_type 选择:
          - 读取 "topo_type" (默认 "8p_server")
          - 读取 "<topo_type>_port_bandwidth" 作为当前生效配置
          - 全部 "<xxx>_port_bandwidth" 存入 all_port_bandwidths
          - 兼容旧格式: 若无 topo_type 且有 "port_bandwidth", 直接使用
        """
        cfg = cls()
        if "local_record_delay" in d:
            cfg.local_record_delay = float(d["local_record_delay"]) * _US_TO_S
        if "remote_record_delay" in d:
            cfg.remote_record_delay = float(d["remote_record_delay"]) * _US_TO_S
        if "local_wait_delay" in d:
            cfg.local_wait_delay = float(d["local_wait_delay"]) * _US_TO_S
        if "remote_wait_delay" in d:
            cfg.remote_wait_delay = float(d["remote_wait_delay"]) * _US_TO_S
        # topo_type
        cfg.topo_type = d.get("topo_type", "8p_server")
        # 收集全部 <xxx>_port_bandwidth 配置
        for key, val in d.items():
            if key.endswith("_port_bandwidth") and isinstance(val, dict):
                topo_name = key[:-len("_port_bandwidth")]
                cfg.all_port_bandwidths[topo_name] = {
                    k: float(v) * _GBPS_TO_BPS for k, v in val.items()}
        # 兼容旧格式: 无 topo_type 但有 port_bandwidth
        if not cfg.all_port_bandwidths and "port_bandwidth" in d:
            cfg.port_bandwidth = {k: float(v) * _GBPS_TO_BPS
                                  for k, v in d["port_bandwidth"].items()}
        else:
            # 按 topo_type 选择当前生效的端口带宽
            if cfg.topo_type not in cfg.all_port_bandwidths:
                available = ", ".join(cfg.all_port_bandwidths.keys()) or "(无)"
                raise ValueError(
                    f"topo_type='{cfg.topo_type}' 在配置中未找到对应的 "
                    f"'{cfg.topo_type}_port_bandwidth', 可用: {available}")
            cfg.port_bandwidth = cfg.all_port_bandwidths[cfg.topo_type]
        if "default_port_bandwidth" in d:
            cfg.default_port_bandwidth = float(d["default_port_bandwidth"]) * _GBPS_TO_BPS
        if "hbm_bandwidth" in d:
            cfg.hbm_bandwidth = float(d["hbm_bandwidth"]) * _GBPS_TO_BPS
        if "hbm_utilization" in d:
            cfg.hbm_utilization = float(d["hbm_utilization"])
        if "hbm_latency" in d:
            cfg.hbm_latency = float(d["hbm_latency"]) * _US_TO_S
        if "bw_profile_set" in d:
            cfg.bw_profile_set = BandwidthProfileSet.from_dict(d["bw_profile_set"])
        if "d2h_bandwidth" in d:
            cfg.d2h_bandwidth = float(d["d2h_bandwidth"]) * _GBPS_TO_BPS
        if "d2h_utilization" in d:
            cfg.d2h_utilization = float(d["d2h_utilization"])
        return cfg

    def to_dict(self) -> dict:
        """序列化为 JSON 字典, 时延按 s→us, 带宽按 B/s→GBps。"""
        d = {
            "local_record_delay": self.local_record_delay / _US_TO_S,
            "remote_record_delay": self.remote_record_delay / _US_TO_S,
            "local_wait_delay": self.local_wait_delay / _US_TO_S,
            "remote_wait_delay": self.remote_wait_delay / _US_TO_S,
            "topo_type": self.topo_type,
        }
        # 输出全部拓扑的端口带宽
        for topo_name, bw_dict in self.all_port_bandwidths.items():
            d[f"{topo_name}_port_bandwidth"] = {
                k: v / _GBPS_TO_BPS for k, v in bw_dict.items()}
        # 若无分组配置, 回退为 port_bandwidth
        if not self.all_port_bandwidths:
            d["port_bandwidth"] = {k: v / _GBPS_TO_BPS
                                   for k, v in self.port_bandwidth.items()}
        d["default_port_bandwidth"] = self.default_port_bandwidth / _GBPS_TO_BPS
        d["hbm_bandwidth"] = self.hbm_bandwidth / _GBPS_TO_BPS
        d["hbm_utilization"] = self.hbm_utilization
        d["hbm_latency"] = self.hbm_latency / _US_TO_S
        if self.bw_profile_set is not None:
            d["bw_profile_set"] = self.bw_profile_set.to_dict()
        d["d2h_bandwidth"] = self.d2h_bandwidth / _GBPS_TO_BPS
        d["d2h_utilization"] = self.d2h_utilization
        return d

    def get_port_bandwidth(self, port_id: str) -> float:
        """查单个端口带宽 (bytes/s), 未配置则回退 default_port_bandwidth。

        d2h 端口使用 d2h_bandwidth, 不受 topo_type 端口配置影响。
        """
        if port_id == "d2h":
            return self.d2h_bandwidth
        if port_id not in self.port_bandwidth:
            _logger.warning(
                "端口 '%s' 未在 %s_port_bandwidth 中配置, 回退 "
                "default_port_bandwidth (%.3f GBps)",
                port_id, self.topo_type, self.default_port_bandwidth / _GBPS_TO_BPS)
            return self.default_port_bandwidth
        return self.port_bandwidth[port_id]

    def get_ports_total_bandwidth(self, port_ids: Union[str, list[str]]) -> float:
        """查一个或多个端口的总带宽 (bytes/s), 多端口求和。"""
        if isinstance(port_ids, str):
            port_ids = [port_ids]
        return sum(self.get_port_bandwidth(pid) for pid in port_ids)

    @classmethod
    def default_for_task_txt(cls) -> "SimConfig":
        """task.txt 解析模式下的默认配置。"""
        return cls(
            local_record_delay=30e-9,
            remote_record_delay=1e-6,
            local_wait_delay=30e-9,
            remote_wait_delay=1e-6,
            topo_type="8p_server",
            port_bandwidth={},
            all_port_bandwidths={},
            default_port_bandwidth=53.125e9,
            hbm_bandwidth=4e12,
            hbm_utilization=0.7,
            hbm_latency=50e-9,
            d2h_bandwidth=32e9,
            d2h_utilization=0.8,
        )
