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

"""输入用例构造模块。

提供 _streams_to_dep_graph 函数, 将 streams dict 列表转换为 DependencyGraph。
调用方 (如 test_verify.py) 自行构造 streams dict, 调用此函数构建依赖图。
"""

from __future__ import annotations

import json

from .config import SimConfig
from .task_model import Task, TaskType, INVALID
from .dependency_graph import DependencyGraph

_TASK_TYPE_MAP = {
    "Notify_Record": TaskType.NOTIFY_RECORD,
    "Notify_Wait": TaskType.NOTIFY_WAIT,
    "Reduce_Inline": TaskType.REDUCE_INLINE,
    "Memcpy": TaskType.MEMCPY,
    "Ub_Inline_Write": TaskType.UB_INLINE_WRITE,
    "Ub_Write_Or_Read": TaskType.UB_WRITE_OR_READ,
    "Write_With_Notify": TaskType.WRITE_WITH_NOTIFY,
}


def _streams_to_dep_graph(streams: list[dict]) -> DependencyGraph:
    """从 streams dict 列表构建 DependencyGraph。

    每个 stream:
      {"stream_id": "0", "tasks": [{task_id, type, ...}, ...]}

    自动构建:
      - 任务节点 (带全部参数)
      - 串行链 (stream 内按顺序)
      - notify 配对 (Notify_Record + Notify_Wait, src_rank==dst_rank 即 local)
    """
    g = DependencyGraph()
    chains: dict[int, list[str]] = {}
    records: dict[int, list[str]] = {}
    waits: dict[int, list[str]] = {}

    for stream in streams:
        sid = int(stream["stream_id"])
        task_ids = []
        for raw in stream["tasks"]:
            ttype = _TASK_TYPE_MAP.get(raw["type"])
            if ttype is None:
                continue

            task = Task(
                task_id=raw["task_id"],
                task_type=ttype,
                stream_id=sid,
                notify_id=int(raw["notify_id"]) if raw.get("notify_id") is not None else INVALID,
                src_rank=int(raw.get("src_rank", INVALID)),
                dst_rank=int(raw.get("dst_rank", INVALID)),
                size=int(raw.get("size", 0)),
                port=raw.get("port"),
                src_eid=raw.get("src_eid", ""),
                dst_eid=raw.get("dst_eid", ""),
                transport_type=raw.get("transport_type", ""),
                data_type=raw.get("data_type", ""),
                link_type=raw.get("link_type", ""),
                protocol=raw.get("protocol", ""),
                jetty_id=int(raw.get("jetty_id", INVALID)),
                src_buf_type=raw.get("src_buf_type", ""),
                dst_buf_type=raw.get("dst_buf_type", ""),
                loop_cnt=int(raw.get("loop_cnt", 0)),
                loop_iter_cnt=int(raw.get("loop_iter_cnt", 0)),
                loop_tag=str(raw.get("loop_tag", "") or ""),
            )
            g.add_task(task)
            task_ids.append(task.task_id)

            if ttype == TaskType.NOTIFY_RECORD and task.is_local_notify:
                records.setdefault(task.notify_id, []).append(task.task_id)
            elif ttype == TaskType.NOTIFY_WAIT and task.is_local_notify:
                waits.setdefault(task.notify_id, []).append(task.task_id)

        g.add_stream_chain(sid, task_ids)

    for nid, wait_ids in waits.items():
        rec_ids = records.get(nid, [])
        if rec_ids:
            for wid in wait_ids:
                g.add_match(wid, rec_ids[0])

    return g


def load_config(filepath: str) -> SimConfig:
    """从 JSON 文件加载 SimConfig。"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return SimConfig.from_dict(data)
