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

"""依赖图: 仿真器的唯一输入, 包含任务节点 (带全部参数) + 依赖边。

四类内容:
  1. 任务节点: dict[task_id → Task], 每个 Task 携带全部参数
  2. 串行依赖 (stream chain): dict[stream_id → [task_id_0, ...]]
  3. notify 依赖: dict[wait_task_id → record_task_id] (1:1 显式配对)
  4. 额外依赖边 (extra edge): dict[child_task_id → [parent_task_id, ...]],
     child 执行前须等待全部 parent 完成。用于 CCU 队列并行化的 fork/join
     边 (7.4 节); 与 notify 边不同, 仅时序约束, 不承载事件语义。

notify 配对规则:
  Notify_Record (src_rank == dst_rank, 即 local) 与 Notify_Wait (local) 配对。
  远端 notify (src_rank != dst_rank) 不参与配对, 走固定时延。
"""

from __future__ import annotations

from .task_model import Task, TaskType, NOTIFY_TYPES, INVALID


class DependencyGraph:
    """完整的任务依赖图 (纯内存数据结构)。"""

    __slots__ = ("_tasks", "_stream_chains", "_notify_matches", "_extra_edges")

    def __init__(self):
        self._tasks: dict[str, Task] = {}
        self._stream_chains: dict[int, list[str]] = {}
        self._notify_matches: dict[str, str] = {}
        self._extra_edges: dict[str, list[str]] = {}

    def add_task(self, task: Task) -> None:
        self._tasks[task.task_id] = task

    def remove_task(self, task_id: str) -> None:
        """删除任务节点及其全部关联边 (6.5.1 节特征链消解)。

        同步清理: 以该任务为 child 的 extra 边、其余任务 extra 边中指向
        该任务的 parent 引用、该任务参与的 notify 配对 (作为 wait 或
        record 一侧)。流串行链不在此清理 (调用方保证仅在组装前使用)。
        """
        self._tasks.pop(task_id, None)
        self._extra_edges.pop(task_id, None)
        for parents in self._extra_edges.values():
            if task_id in parents:
                parents.remove(task_id)
        self._notify_matches.pop(task_id, None)
        for wid, rid in list(self._notify_matches.items()):
            if rid == task_id:
                del self._notify_matches[wid]

    def get_task(self, task_id: str) -> Task:
        return self._tasks[task_id]

    @property
    def tasks(self) -> dict[str, Task]:
        return self._tasks

    @property
    def task_count(self) -> int:
        return len(self._tasks)

    def add_stream_chain(self, stream_id: int, task_ids: list[str]) -> None:
        self._stream_chains[stream_id] = list(task_ids)

    def remove_stream_chain(self, stream_id: int) -> None:
        """移除串行链 (如全部任务被 CCU 占位替换的空 SQE 流)。"""
        self._stream_chains.pop(stream_id, None)

    def get_chain(self, stream_id: int) -> list[str]:
        return self._stream_chains.get(stream_id, [])

    @property
    def stream_ids(self):
        return self._stream_chains.keys()

    @property
    def chain_count(self) -> int:
        return len(self._stream_chains)

    def add_match(self, wait_task_id: str, record_task_id: str) -> None:
        self._notify_matches[wait_task_id] = record_task_id

    def remove_match(self, wait_task_id: str) -> None:
        self._notify_matches.pop(wait_task_id, None)

    def get_match(self, wait_task_id: str):
        return self._notify_matches.get(wait_task_id)

    def is_matched_wait(self, wait_task_id: str) -> bool:
        return wait_task_id in self._notify_matches

    def is_matched_record(self, record_task_id: str) -> bool:
        return record_task_id in self._notify_matches.values()

    @property
    def matched_wait_ids(self):
        return self._notify_matches.keys()

    @property
    def match_count(self) -> int:
        return len(self._notify_matches)

    def add_edge(self, child_task_id: str, parent_task_id: str) -> None:
        """添加额外依赖边: child 须等待 parent 完成后才能开始。"""
        parents = self._extra_edges.setdefault(child_task_id, [])
        if parent_task_id not in parents:
            parents.append(parent_task_id)

    def get_extra_parents(self, task_id: str) -> list:
        return self._extra_edges.get(task_id, [])

    @property
    def extra_edges(self) -> dict:
        return self._extra_edges

    @property
    def extra_parent_ids(self):
        """全部 extra 边的父任务 id 集合 (仿真器预建完成事件用)。"""
        return {p for parents in self._extra_edges.values() for p in parents}

    @property
    def extra_edge_count(self) -> int:
        return sum(len(v) for v in self._extra_edges.values())

    def __len__(self) -> int:
        return self.task_count

    def __repr__(self) -> str:
        return (f"DependencyGraph(tasks={self.task_count}, "
                f"streams={self.chain_count}, "
                f"notify_matches={self.match_count}, "
                f"extra_edges={self.extra_edge_count})")
