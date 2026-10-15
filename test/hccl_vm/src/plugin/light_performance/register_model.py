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

"""CCU 寄存器与 CKE 执行模型 (设计文档 6.4 节)。

对 checker AllRankParamRecorder 的 Python 简化: 仅保留 xn / gsa / cke 三类寄存器
与 seen_post 记账, LOAD/算逻指令在该模型上"执行"; 单 rank 成图下只登记本 rank 的
post, 依赖远端置位的 wait 由多 queue 联合推进的停滞检测发现后按远端 wait 节点
仿真 (6.6 节)。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PostRecord:
    task_id: str
    cke_id: int
    cke_mask: int
    remaining_mask: int
    record_device_id: int
    die_id: int
    is_local: bool


class CcuRegisterModel:
    """xn/gsa/cke 寄存器模型 + CKE post 记账。"""

    def __init__(self, device_id: int):
        self.device_id = device_id
        self.xn: dict = {}
        self.gsa: dict = {}
        self.cke: dict = {}
        self.seen_post: dict = {}
        self.used_post_ids: set = set()

    def _key(self, die_id: int, reg_id: int) -> tuple:
        return (self.device_id, die_id, reg_id)

    def get_xn(self, die_id: int, xn_id: int) -> int:
        key = self._key(die_id, xn_id)
        if key not in self.xn:
            logger.warning("Xn 寄存器未初始化按 0 处理: dev=%s die=%s xn=%s",
                           self.device_id, die_id, xn_id)
            return 0
        return self.xn[key]

    def set_xn(self, die_id: int, xn_id: int, value: int) -> None:
        self.xn[self._key(die_id, xn_id)] = value & 0xFFFFFFFFFFFFFFFF

    def get_gsa(self, die_id: int, gsa_id: int) -> int:
        key = self._key(die_id, gsa_id)
        if key not in self.gsa:
            logger.warning("GSA 寄存器未初始化按 0 处理: dev=%s die=%s gsa=%s",
                           self.device_id, die_id, gsa_id)
            return 0
        return self.gsa[key]

    def set_gsa(self, die_id: int, gsa_id: int, value: int) -> None:
        self.gsa[self._key(die_id, gsa_id)] = value & 0xFFFFFFFFFFFFFFFF

    def get_cke(self, die_id: int, cke_id: int) -> int:
        return self.cke.get(self._key(die_id, cke_id), 0)

    def set_cke(self, die_id: int, cke_id: int, value: int) -> None:
        self.cke[self._key(die_id, cke_id)] = value & 0xFFFF

    def or_cke(self, die_id: int, cke_id: int, mask: int) -> None:
        key = self._key(die_id, cke_id)
        self.cke[key] = (self.cke.get(key, 0) | mask) & 0xFFFF

    def clear_cke(self, die_id: int, cke_id: int, mask: int) -> None:
        key = self._key(die_id, cke_id)
        self.cke[key] = self.cke.get(key, 0) & (~mask & 0xFFFF)

    def register_post(self, task_id: str, die_id: int, cke_id: int, cke_mask: int,
                      record_device_id: int, is_local: bool) -> None:
        if cke_mask == 0:
            return
        key = (self.device_id, die_id, cke_id)
        self.seen_post.setdefault(key, []).append(
            PostRecord(task_id, cke_id, cke_mask, cke_mask,
                       record_device_id, die_id, is_local))

    def wait_satisfied(self, die_id: int, cke_id: int, wait_mask: int) -> bool:
        return (self.get_cke(die_id, cke_id) & wait_mask) == wait_mask

    def try_match_wait(self, die_id: int, cke_id: int, wait_mask: int) -> list:
        """按 6.4 节规则将 wait 与已登记的本地 post 配对。

        返回有位交集的本地 PostRecord 列表 (并扣减其 remaining_mask): 调用方
        按 checker GenWaitNode 语义将全部贡献 post 聚合为单个 wait 节点
        (mask 取位或), 节点与各贡献 post 建依赖边。
        """
        matched = []
        if wait_mask == 0:
            return matched
        key = (self.device_id, die_id, cke_id)
        for post in self.seen_post.get(key, []):
            if post.remaining_mask & wait_mask == 0:
                continue
            matched.append(post)
            post.remaining_mask &= ~wait_mask & 0xFFFF
        self.seen_post[key] = [p for p in self.seen_post.get(key, [])
                               if p.remaining_mask != 0]
        return matched

    @property
    def residual_post_count(self) -> int:
        return sum(len(v) for v in self.seen_post.values())
