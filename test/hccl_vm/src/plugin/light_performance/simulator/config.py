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
# (servers_info.json 透传, C++ 集群解析器入库 Server.port_type);
# ascend960 按 port_type 选择端口带宽表 (ascend960_x2/ascend960_x4),
# ascend950 只有 x4 端口 (port_type 配置为 x2 报错)
_PORT_TYPES = ("x2", "x4")
_PORT_TYPE_DEFAULT = "x4"

# soc_version 家族: 端口带宽表按芯片代际划分 (<family>_xxx_port_bandwidth),
# 家族由 soc_version 前缀匹配 (大小写不敏感, 如 Ascend950DT_95A1 → ascend950);
# ascend950 按 topo_type 选表 (ascend950_<topo_type>), ascend960 按 port_type
# 选表 (ascend960_<port_type>); 无匹配表报错
_SOC_FAMILIES = ("ascend950", "ascend960")
# soc_version 缺失时的默认家族: 与旧版行为一致 (原 8p_server/1d_pod 表即 ascend950)
_SOC_FAMILY_DEFAULT = "ascend950"
_SOC_VERSION_DEFAULT = "Ascend950"


def _normalize_port_type(value) -> str:
    """端口类型归一化(小写): 仅接受 x2/x4(不区分大小写), 非法值报错。"""
    normalized = str(value if value is not None else _PORT_TYPE_DEFAULT).strip().lower()
    if normalized not in _PORT_TYPES:
        raise ValueError(
            f"invalid port_type '{value}', only {list(_PORT_TYPES)} supported "
            "(case-insensitive)")
    return normalized


def _normalize_soc_family(value) -> str:
    """soc_version 归一化为家族名: 大小写不敏感前缀匹配 ascend950/ascend960。

    空值回退默认家族 ascend950 (旧库无 Device.soc_version 列);
    非空但不匹配任何家族报错 (端口带宽表按家族划分, 无法选表)。
    """
    text = str(value if value is not None else "").strip()
    if not text:
        return _SOC_FAMILY_DEFAULT
    lowered = text.lower()
    for family in _SOC_FAMILIES:
        if lowered.startswith(family):
            return family
    raise ValueError(
        f"invalid soc_version '{value}', only {list(_SOC_FAMILIES)} prefixes "
        "supported (case-insensitive)")


@dataclass
class SimConfig:
    """仿真配置参数。

    内部单位: 时延=秒, 带宽=bytes/s。
    JSON 配置单位: 时延=us, 带宽=GBps (from_dict 自动转换)。

    Attributes:
        topo_type: 拓扑类型, ascend950 按其选择端口带宽表
            ("ascend950_8p_server" 或 "ascend950_1d_pod")
        port_bandwidth: {"die/port": 带宽 (bytes/s)}, 当前选中的端口带宽表
        all_port_bandwidths: {"表名": {"die/port": 带宽}}, 全部端口带宽表配置
            (表名即配置键 "<表名>_port_bandwidth" 去掉后缀)
        default_port_bandwidth: 未配置端口的默认带宽 (bytes/s)
        soc_version: 芯片版本 (默认 "Ascend950"), 来自 db Device.soc_version 字段
            (集群配置 soc_version 字段经 servers_info.json 入库); 前缀匹配
            ascend950/ascend960 家族决定端口带宽表: ascend950 按 topo_type 选
            ascend950_<topo_type> 表 (仅支持 x4 端口, port_type=x2 报错),
            ascend960 按 port_type 选 ascend960_<port_type> 表; 无匹配表报错
        port_type: 端口类型 ("x2"/"x4", 默认 "x4"), 来自 db Server.port_type 字段
            (server/pod 拓扑配置 port_type 字段经 servers_info.json 入库);
            ascend960 按其选择端口带宽表 (ascend960_x2/ascend960_x4),
            ascend950 只有 x4 端口 (配置 x2 报错); d2h 带宽不受影响
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
    topo_type: str = "8p_server"              # 拓扑类型, ascend950 按其选择端口带宽表
    port_bandwidth: dict[str, float] = field(default_factory=dict)  # 当前生效的端口带宽
    all_port_bandwidths: dict[str, dict[str, float]] = field(default_factory=dict)  # 全部端口带宽表
    default_port_bandwidth: float = 53.125e9  # 53.125 GB/s, JSON 值单位: GBps
    # 芯片版本 (db Device.soc_version): 前缀匹配 ascend950/ascend960 家族,
    # 决定端口带宽表选择 (ascend950 按 topo_type, ascend960 按 port_type)
    soc_version: str = _SOC_VERSION_DEFAULT
    # 端口类型 ("x2"/"x4", db Server.port_type): ascend960 按其选端口带宽表;
    # ascend950 只有 x4 端口 (x2 报错)
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

        端口带宽按 soc_version + topo_type/port_type 选择:
          - 读取 "topo_type" (默认 "8p_server")、"soc_version" (默认 Ascend950)、
            "port_type" (默认 x4)
          - 全部 "<xxx>_port_bandwidth" 存入 all_port_bandwidths
          - ascend950 → "ascend950_<topo_type>" 表 (仅支持 x4, x2 报错);
            ascend960 → "ascend960_<port_type>" 表; 无匹配表报错
          - 兼容旧格式: 若无任何 "<xxx>_port_bandwidth" 且有 "port_bandwidth",
            直接使用
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
        # 兼容旧格式: 无任何分组表但有 port_bandwidth
        if not cfg.all_port_bandwidths and "port_bandwidth" in d:
            cfg.port_bandwidth = {k: float(v) * _GBPS_TO_BPS
                                  for k, v in d["port_bandwidth"].items()}
        if "default_port_bandwidth" in d:
            cfg.default_port_bandwidth = float(d["default_port_bandwidth"]) * _GBPS_TO_BPS
        # soc_version / port_type: 共同决定端口带宽表选择。权威来源为 db
        # Device.soc_version / Server.port_type (pipeline 覆盖); 配置文件可给缺省值。
        if "soc_version" in d:
            cfg.soc_version = str(d["soc_version"] or "").strip() \
                or _SOC_VERSION_DEFAULT
            _normalize_soc_family(cfg.soc_version)
        if "port_type" in d:
            cfg.port_type = _normalize_port_type(d["port_type"])
        cfg._select_port_bandwidth()
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
        # 输出全部端口带宽表
        for topo_name, bw_dict in self.all_port_bandwidths.items():
            d[f"{topo_name}_port_bandwidth"] = {
                k: v / _GBPS_TO_BPS for k, v in bw_dict.items()}
        # 若无分组配置, 回退为 port_bandwidth
        if not self.all_port_bandwidths:
            d["port_bandwidth"] = {k: v / _GBPS_TO_BPS
                                   for k, v in self.port_bandwidth.items()}
        d["default_port_bandwidth"] = self.default_port_bandwidth / _GBPS_TO_BPS
        d["soc_version"] = self.soc_version
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

        d2h 端口使用 d2h_bandwidth, 不受端口带宽表选择影响。
        其余端口取当前选中的端口带宽表 (按 soc_version + topo_type/port_type
        选择, 见 _select_port_bandwidth)。
        """
        if port_id == "d2h":
            return self.d2h_bandwidth
        if port_id not in self.port_bandwidth:
            _logger.warning(
                "port '%s' not configured in port bandwidth table "
                "(soc_version=%s, topo_type=%s, port_type=%s), "
                "falling back to default_port_bandwidth (%.3f GBps)",
                port_id, self.soc_version, self.topo_type, self.port_type,
                self.default_port_bandwidth / _GBPS_TO_BPS)
            return self.default_port_bandwidth
        return self.port_bandwidth[port_id]

    @property
    def soc_family(self) -> str:
        """soc_version 归一化家族 (ascend950/ascend960), 非法值报错。"""
        return _normalize_soc_family(self.soc_version)

    @property
    def bandwidth_table_key(self) -> str:
        """当前生效端口带宽表名 (配置键 '<表名>_port_bandwidth' 的 <表名>)。

        ascend950 按 topo_type 选表 (仅支持 x4 端口, port_type=x2 报错);
        ascend960 按 port_type 选表 (ascend960_x2/ascend960_x4)。
        """
        family = self.soc_family
        if family == "ascend950":
            if self.port_type != "x4":
                raise ValueError(
                    f"soc_version '{self.soc_version}' (ascend950) only supports "
                    f"port_type 'x4' (no x2 ports), got '{self.port_type}'")
            return f"ascend950_{self.topo_type}"
        return f"ascend960_{self.port_type}"

    def _select_port_bandwidth(self) -> None:
        """按 soc_version + topo_type/port_type 选择当前生效端口带宽表。

        无分组表 (旧格式直接给 port_bandwidth / 缺省构造) 时不选择;
        选出的表不在配置中 (soc_version 与带宽表不匹配) 报错。
        """
        if not self.all_port_bandwidths:
            return
        key = self.bandwidth_table_key
        if key not in self.all_port_bandwidths:
            available = ", ".join(sorted(self.all_port_bandwidths)) or "(none)"
            raise ValueError(
                f"soc_version='{self.soc_version}' topo_type='{self.topo_type}' "
                f"port_type='{self.port_type}' has no corresponding "
                f"'{key}_port_bandwidth' in config, available: {available}")
        self.port_bandwidth = self.all_port_bandwidths[key]

    def with_soc_and_port_type(self, soc_version: str = None,
                               port_type: str = None) -> "SimConfig":
        """返回设置 soc_version/port_type 并重选端口带宽表后的 config 副本。

        soc_version 权威来源为 db Device.soc_version (集群配置 soc_version 字段),
        port_type 权威来源为 db Server.port_type (server/pod 拓扑配置 port_type
        字段); 二者共同决定端口带宽表 (ascend950_<topo_type> /
        ascend960_<port_type>), 同时设置后再选表, 避免逐项生效时按旧值选表
        产生中间态误报 (如目标为 ascend960+x2 时, 先按 ascend950 校验 x2 报错)。
        值均不变时原样返回 self; 组合非法 (ascend950+x2 / 无匹配表) 报错。
        """
        new_soc = (self.soc_version if soc_version is None
                   else str(soc_version).strip() or _SOC_VERSION_DEFAULT)
        new_port = (self.port_type if port_type is None
                    else _normalize_port_type(port_type))
        _normalize_soc_family(new_soc)
        if new_soc == self.soc_version and new_port == self.port_type:
            return self
        cfg = replace(self, soc_version=new_soc, port_type=new_port)
        cfg._select_port_bandwidth()
        return cfg

    def with_soc_version(self, soc_version: str) -> "SimConfig":
        """返回按 soc_version 重选端口带宽表后的 config 副本 (其余字段共享)。"""
        return self.with_soc_and_port_type(soc_version=soc_version)

    def with_port_type(self, port_type: str) -> "SimConfig":
        """返回按 port_type 重选端口带宽表后的 config 副本 (其余字段共享)。

        port_type 权威来源为 db Server.port_type (server/pod 拓扑配置 port_type
        字段经 servers_info.json 入库); ascend960 按其选择端口带宽表
        (ascend960_x2/ascend960_x4), ascend950 只有 x4 端口 (x2 报错)。
        """
        return self.with_soc_and_port_type(port_type=port_type)

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
            soc_version=_SOC_VERSION_DEFAULT,
            hbm_bandwidth=4e12,
            hbm_utilization=0.7,
            hbm_latency=50e-9,
            d2h_bandwidth=32e9,
            d2h_utilization=0.8,
        )
