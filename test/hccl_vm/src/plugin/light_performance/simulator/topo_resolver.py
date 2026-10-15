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

"""拓扑解析模块: 基于 ranktable.json 查找端口。

ranktable.json 结构:
  rank_list[].level_list[].rank_addr_list[]:
    addr (EID值) → ports (端口列表如 ["1/0"] 或 ["0/0","0/1",...,"0/7"])

查询方式:
  write 任务 → task.src_eid → ranktable[eid] → ports
  read 任务  → task.dst_eid → ranktable[eid] → ports
"""

from __future__ import annotations

import json


class TopoResolver:
    """拓扑解析器: eid → port 查询。"""

    def __init__(self, ranktable_path: str):
        if not ranktable_path:
            raise ValueError("ranktable_path 不能为空")
        self._eid_to_ports: dict[str, list[str]] = {}
        self.load_ranktable(ranktable_path)

    def load_ranktable(self, filepath: str) -> None:
        """加载 ranktable.json, 构建 eid → ports 映射。"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        for rank_entry in data.get("rank_list", []):
            for level in rank_entry.get("level_list", []):
                for addr in level.get("rank_addr_list", []):
                    if addr.get("addr_type") == "EID":
                        eid = addr.get("addr", "")
                        ports = addr.get("ports", [])
                        if eid and ports:
                            self._eid_to_ports[eid] = ports

    def get_ports_by_eid(self, eid: str) -> list[str]:
        """通过 eid 查找端口列表, 未找到则报错。"""
        ports = self._eid_to_ports.get(eid)
        if ports is None:
            raise ValueError(
                f"eid='{eid}' 在 ranktable.json 中未找到对应端口, "
                f"请检查 ranktable 与集群拓扑配置是否匹配")
        return ports

    @property
    def eid_count(self) -> int:
        return len(self._eid_to_ports)
