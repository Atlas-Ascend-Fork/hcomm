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

from dataclasses import dataclass, field, replace
from typing import Optional, Union

from .bandwidth_profile import BandwidthProfileSet

_logger = logging.getLogger(__name__)

_GBPS_TO_BPS = 1e9
_US_TO_S = 1e-6

# 端口类型合法值与缺省值: 来自 server/pod 拓扑配置 port_type 字段
# (servers_info.json 透传, C++ 集群解析器入库 Server.port_type); 配置带宽基准为 x4
_PORT_TYPES = ("x2", "x4")
_PORT_TYPE_DEFAULT = "x4"
# x2 端口带宽缩放系数: x2 端口带宽是 x4 端口带宽的一半
_PORT_TYPE_SCALES = {"x2": 0.5, "x4": 1.0}


def _normalize_port_type(value) -> str:
    """端口类型归一化(小写): 仅接受 x2/x4(不区分大小写), 非法值报错。"""
    normalized = str(value if value is not None else _PORT_TYPE_DEFAULT).strip().lower()
    if normalized not in _PORT_TYPES:
        raise ValueError(
            f"invalid port_type '{value}', only {list(_PORT_TYPES)} supported "
            "(case-insensitive)")
    return normalized


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
        port_type: 端口类型 ("x2"/"x4", 默认 "x4"), 来自 db Server.port_type 字段
            (server/pod 拓扑配置 port_type 字段经 servers_info.json 入库); 配置的端口带宽
            基准为 x4, x2 时端口带宽 (含 default_port_bandwidth) 减半, d2h 不受影响
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
    # 端口类型 ("x2"/"x4"): 配置带宽基准为 x4; x2 时端口带宽减半 (db Server.port_type)
    port_type: str = "x4"
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
                available = ", ".join(cfg.all_port_bandwidths.keys()) or "(none)"
                raise ValueError(
                    f"topo_type='{cfg.topo_type}' has no corresponding "
                    f"'{cfg.topo_type}_port_bandwidth' in config, available: {available}")
            cfg.port_bandwidth = cfg.all_port_bandwidths[cfg.topo_type]
        if "default_port_bandwidth" in d:
            cfg.default_port_bandwidth = float(d["default_port_bandwidth"]) * _GBPS_TO_BPS
        # port_type: 端口类型(x2/x4), 配置带宽基准为 x4, x2 时端口带宽减半。
        # 权威来源为 db Server.port_type (pipeline 覆盖); 配置文件可给缺省值。
        if "port_type" in d:
            cfg.port_type = _normalize_port_type(d["port_type"])
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
        d["port_type"] = self.port_type
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

        d2h 端口使用 d2h_bandwidth, 不受 topo_type 端口配置与 port_type 缩放影响。
        其余端口在配置带宽(x4 基准)上按 port_type 缩放: x2 减半 (port_type_scale)。
        """
        if port_id == "d2h":
            return self.d2h_bandwidth
        scale = self.port_type_scale
        if port_id not in self.port_bandwidth:
            _logger.warning(
                "port '%s' not configured in %s_port_bandwidth, "
                "falling back to default_port_bandwidth (%.3f GBps, port_type=%s scale=%.2f)",
                port_id, self.topo_type, self.default_port_bandwidth / _GBPS_TO_BPS,
                self.port_type, scale)
            return self.default_port_bandwidth * scale
        return self.port_bandwidth[port_id] * scale

    @property
    def port_type_scale(self) -> float:
        """端口带宽缩放系数: x4=1.0 (配置基准), x2=0.5 (端口带宽减半)。"""
        return _PORT_TYPE_SCALES.get(self.port_type, 1.0)

    def with_port_type(self, port_type: str) -> "SimConfig":
        """返回按 port_type(x2/x4) 设置端口类型后的 config 副本 (其余字段共享)。

        port_type 权威来源为 db Server.port_type (server/pod 拓扑配置 port_type
        字段经 servers_info.json 入库); 配置带宽基准为 x4, x2 时 get_port_bandwidth
        对非 d2h 端口减半。值不变时原样返回 self。
        """
        normalized = _normalize_port_type(port_type)
        if normalized == self.port_type:
            return self
        return replace(self, port_type=normalized)

    def get_ports_total_bandwidth(self, port_ids: Union[str, list[str]]) -> float:
        """查一个或多个端口的总带宽 (bytes/s), 多端口求和。"""
        if isinstance(port_ids, str):
            port_ids = [port_ids]
        return sum(self.get_port_bandwidth(pid) for pid in port_ids)

    def for_jetty_num(self, jetty_num: int) -> "SimConfig":
        """返回按 jetty 数折算 bw_profile_set 后的 config 副本 (其余字段共享)。

        jetty_num 即 HCCL_UB_MULTI_CHANNEL_NUM (proxy 侧读取并落库到
        OpDetailTab.ubMultiChannelNum, 每 channel 的 jetty 数): >1 时
        multi_port_utilization_low 各档利用率抬升
        (jetty_num - 1) × jetty_utilization_step (见 bandwidth_profile)。
        未配置 bw_profile_set (利用率恒 1.0) 或 jetty_num <= 1 时原样返回 self。
        """
        if self.bw_profile_set is None or int(jetty_num or 1) <= 1:
            return self
        return replace(self, bw_profile_set=
                       self.bw_profile_set.for_jetty_num(jetty_num))

    def describe_jetty_num(self, jetty_num: int) -> str:
        """多 jetty 折算的一行说明 (供各入口打印, 未折算时说明原因)。"""
        num = int(jetty_num or 1)
        if num <= 1:
            return f"jetty_num={num}, using original multi_port_utilization_low table"
        if self.bw_profile_set is None:
            return (f"jetty_num={num}, but bw_profile not configured "
                    "(utilization constant 1.0), not scaled")
        step = self.bw_profile_set.jetty_utilization_step
        return (f"jetty_num={num}, multi_port_utilization_low each tier "
                f"+{(num - 1) * step:.6f} (step={step})")

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
