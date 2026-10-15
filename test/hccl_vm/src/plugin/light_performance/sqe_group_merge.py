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

"""SQE 批任务合并 (性能仿真入口预处理)。

采集侧为每个 UDMA SQE(一次 doorbell)内的全部 WQE 任务打上相同 sqeGroupId
(环形递增门铃批号 1..255, 255 回绕, 永不取 0; 0 = 构造默认 "未打标"), 任务仍逐
WQE 落库。本模块按 (streamId, sqeGroupId) 切批, 把批内全部数据任务
(MEM_CPY/REDUCE/WRITE_WITH_NOTIFY) 合并为单条, 对齐 hcomm 批量搬运 profiling 的
ExecProfilingAll 口径: 描述信息(类型/地址/端点/notifyId/规约参数)取批内最后一个
数据任务, 长度取全批累加。

按流分组的隔离保证: 切批键显式含 streamId, 跨流任务永不并入同批; 批号在 255
窗口内唯一 (其他算子的门铃虽不可见但消耗批号), 同流同号必为同一门铃, 不跨门铃
错并。

不参与合并: notify/SDMA/同步等本地任务 (原样保留, 不影响批边界); 未打标/无组号
任务 (sqeGroupId <= 0, v3 及更早 BLOB 无此字段) 原样保留并切断当前批; UDMA 非
数据任务 (inline 写 notify) 仅参与切批。
"""

from __future__ import annotations

import logging

from .db_reader import (JETTY_INVALID, RawTask, ReduceTaskData, TransMemTaskData,
                        WriteWithNotifyTaskData)

logger = logging.getLogger(__name__)


def _is_udma_task(task: RawTask) -> bool:
    """UDMA(jetty路径)来源任务; SDMA/notify/同步任务 jettyId 为无效值"""
    return task.meta.jetty_id != JETTY_INVALID


def _is_mergeable(task: RawTask) -> bool:
    if not _is_udma_task(task):
        return False
    return isinstance(task.meta.data, (TransMemTaskData, ReduceTaskData,
                                       WriteWithNotifyTaskData))


def _task_len(task: RawTask) -> int:
    data = task.meta.data
    if isinstance(data, ReduceTaskData):
        return data.data_size
    return data.length


def _set_len(task: RawTask, length: int) -> None:
    data = task.meta.data
    if isinstance(data, ReduceTaskData):
        data.data_size = length
    else:
        data.length = length


def merge_sqe_group_tasks(tasks: list) -> tuple:
    """按 (streamId, sqeGroupId) 切批, 批内数据任务无条件合并为单条。

    切批键含 streamId: 跨流任务永不并入同批 (一个门铃本就只属一条流, 单门铃
    批不会被键拆开); 组号 (门铃批号) 在 255 窗口内唯一, 同流同号必为同一门铃。
    合并任务占批内第一个数据任务的槽位(批首), 描述信息随批内每个数据任务逐个
    替换(最终为最后一个数据任务的描述), 长度取全批累加——与 hcomm
    ExecProfilingAll(loc[insNum-1], rmt[insNum-1], totalSize, transferOp[insNum-1])
    逐字段一致。批内夹杂的 inline notify 排在合并任务之后(notify 因果)。

    Returns:
        (合并后任务列表, 被并入的任务数)。非 UDMA 与 sqeGroupId <= 0 的任务原样保留。
    """
    result = []
    merged_count = 0
    acc = None  # 当前批的合并任务(占位于 result, meta 持续替换为批内最后数据任务)
    total_len = 0
    curr_key = None  # (stream_id, sqe_group_id): 流或组号任一变化即切新批
    for task in tasks:
        if not _is_udma_task(task):
            result.append(task)  # 本地任务: 原样保留, 不影响批边界
            continue
        if task.sqe_group_id <= 0:
            # 未打标(0)/无组号(-1): 原样保留, 并保守切断当前批
            result.append(task)
            curr_key = None
            continue
        key = (task.stream_id, task.sqe_group_id)
        if key != curr_key:
            acc = None
            curr_key = key
        if not _is_mergeable(task):
            result.append(task)
            continue
        if acc is None:
            acc = task
            total_len = _task_len(task)
            result.append(task)
        else:
            total_len += _task_len(task)
            acc.meta = task.meta
            _set_len(acc, total_len)
            merged_count += 1
    if merged_count:
        logger.info("SQE 批合并: %d 个 UDMA 数据任务并入同组任务", merged_count)
    return result, merged_count


def merge_dpu_write_sequences(tasks: list) -> tuple:
    """DPU 大消息分片合并: 双地址连续 MEM_CPY 链由 WRITE_WITH_NOTIFY 收尾并成单条
    (对齐 hcomm 整操作单条上报; 无收尾不合并; 仅 src_type==1 参与, 其他任务不切断链)。

    Returns:
        (合并后任务列表, 被并入的任务数)。
    """
    result = []
    merged_count = 0
    chain_idx = []      # 当前链任务在 result 中的占位下标 (合并时置 None)
    chain_src_end = -1  # 链尾 src_offset + length
    chain_dst_end = -1  # 链尾 dst_offset + length
    chain_src_rank = -1
    chain_dst_rank = -1

    for task in tasks:
        data = task.meta.data
        if (task.meta.src_type != 1
                or not isinstance(data, (TransMemTaskData, WriteWithNotifyTaskData))):
            result.append(task)
            continue
        # 双地址连续判定 (WRITE_WITH_NOTIFY 收尾与 MEM_CPY 串链共用)
        contiguous = (chain_idx
                      and data.src_rank_id == chain_src_rank
                      and data.dst_rank_id == chain_dst_rank
                      and data.src_offset == chain_src_end
                      and data.dst_offset == chain_dst_end)
        if isinstance(data, WriteWithNotifyTaskData):
            if contiguous:
                total_len = data.length
                for idx in chain_idx:
                    total_len += result[idx].meta.data.length
                    result[idx] = None
                data.length = total_len
                merged_count += len(chain_idx)
            chain_idx = []
            result.append(task)
            continue
        # DPU MEM_CPY: 双地址连续则串入链, 否则旧链就地保留(不合并)、重开新链
        if not contiguous:
            chain_idx = []
            chain_src_rank = data.src_rank_id
            chain_dst_rank = data.dst_rank_id
        result.append(task)
        chain_idx.append(len(result) - 1)
        chain_src_end = data.src_offset + data.length
        chain_dst_end = data.dst_offset + data.length

    if merged_count:
        logger.info("DPU 序列合并: %d 个前块 MEM_CPY 并入 WRITE_WITH_NOTIFY", merged_count)
    return [t for t in result if t is not None], merged_count
