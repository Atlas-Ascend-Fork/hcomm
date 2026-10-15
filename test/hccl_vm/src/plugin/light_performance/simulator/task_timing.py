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

"""任务时延计算: 各任务类型的独立时延计算函数。

每种任务类型一个函数, 方便查看和修改。

时延模型:
  Notify_Record (local):  固定时延 local_record_delay (30ns)
  Notify_Record (remote): 固定时延 remote_record_delay
  Notify_Wait (local):    等待 Event + 固定时延 local_wait_delay
  Notify_Wait (remote):   固定时延 remote_wait_delay
  Ub_Inline_Write:        固定时延 (同步通知任务, 同 Notify_Record)
  Reduce_Inline:          hbm_latency + (size × 3) / (hbm_bandwidth × hbm_utilization)
  Memcpy:                 hbm_latency + (size × 2) / (hbm_bandwidth × hbm_utilization)
  Ub_Write_Or_Read:       端口带宽共享 (动态)
  Write_With_Notify:      端口带宽共享 (动态, write 方向) + 写落地后
                          notify 固定 record 时延 (对端 wait 不在本 db)
"""

from __future__ import annotations

from .config import SimConfig
from .task_model import Task, TaskType, HBM_TYPES, TRANSFER_TYPES


def calc_notify_record_delay(task: Task, config: SimConfig) -> float:
    """Notify_Record 时延: local 固定时延, remote 固定时延。"""
    if task.is_local_notify:
        return config.local_record_delay
    return config.remote_record_delay


def calc_ub_inline_write_delay(task: Task, config: SimConfig) -> float:
    """Ub_Inline_Write 时延: 同步通知任务, 固定时延 (同 Notify_Record)。

    notify inline 写本质是向远端发通知, 不走数据传输带宽, 按 notify 固定时延计费:
    跨 rank 用 remote_record_delay, 自写边缘场景 (src==dst) 用 local_record_delay。
    """
    if task.is_local_notify:
        return config.local_record_delay
    return config.remote_record_delay


def calc_notify_wait_delay(task: Task, config: SimConfig) -> float:
    """Notify_Wait 时延: local 固定时延 (local_wait_delay), remote 固定时延 (remote_wait_delay)。

    local matched wait 由 simulator 的 Event 机制阻塞, 本函数返回的固定时延
    作为 Event 触发后的处理开销; local unmatched wait 直接用此时延。
    """
    if task.is_local_notify:
        return config.local_wait_delay
    return config.remote_wait_delay


def calc_write_with_notify_delay(task: Task, config: SimConfig) -> float:
    """Write_With_Notify 写后 notify 开销: 写数据落地后向对端发通知的固定时延。

    写传输本身走端口 write 方向带宽共享 (同 Ub_Inline_Write, 由仿真器端口
    管理器计费), 本函数只计 notify 部分: 对端 wait 不在本 db (单 rank 仿真
    前提, 5.4 节), 按 remote record 固定时延; 自写边缘场景 (src==dst) 按
    local record 固定时延。
    """
    if task.is_local_notify:
        return config.local_record_delay
    return config.remote_record_delay


def calc_hbm_delay(task: Task, config: SimConfig) -> float:
    """Reduce_Inline / Memcpy 时延: hbm_latency + (size × HBM 访问次数) / 有效 HBM 带宽。

    HBM 访问次数: Memcpy = 2 (1读+1写), Reduce_Inline = 3 (2读+1写, RMW)。
    有效 HBM 带宽 = hbm_bandwidth × hbm_utilization。
    """
    hbm = config.hbm_bandwidth * config.hbm_utilization
    latency = config.hbm_latency
    hbm_access = 3 if task.task_type == TaskType.REDUCE_INLINE else 2
    if hbm > 0 and task.size > 0:
        return latency + task.size * hbm_access / hbm
    return latency


def is_remote_notify_wait(task: Task, config: SimConfig) -> bool:
    """Notify_Wait 是否为远端 (走固定时延而非 Event 等待)。"""
    return not task.is_local_notify


def is_remote_notify_record(task: Task, config: SimConfig) -> bool:
    """Notify_Record 是否为远端 (仅固定时延, 不触发 Event)。"""
    return not task.is_local_notify


def get_task_timing_info(task: Task, config: SimConfig) -> dict:
    """返回任务的时延计算信息 (用于调试和 trace)。"""
    tt = task.task_type
    if tt == TaskType.NOTIFY_RECORD:
        if task.is_local_notify:
            return {"model": "fixed", "delay_us": calc_notify_record_delay(task, config) * 1e6}
        return {"model": "fixed_remote", "delay_us": calc_notify_record_delay(task, config) * 1e6}
    elif tt == TaskType.NOTIFY_WAIT:
        if task.is_local_notify:
            return {"model": "event_wait+fixed", "delay_us": calc_notify_wait_delay(task, config) * 1e6}
        return {"model": "fixed_remote", "delay_us": calc_notify_wait_delay(task, config) * 1e6}
    elif tt == TaskType.UB_INLINE_WRITE:
        if task.is_local_notify:
            return {"model": "fixed", "delay_us": calc_ub_inline_write_delay(task, config) * 1e6}
        return {"model": "fixed_remote", "delay_us": calc_ub_inline_write_delay(task, config) * 1e6}
    elif tt == TaskType.WRITE_WITH_NOTIFY:
        return {"model": "port_shared_dynamic+fixed_notify",
                "delay_us": calc_write_with_notify_delay(task, config) * 1e6}
    elif tt in HBM_TYPES:
        return {"model": "hbm", "delay_us": calc_hbm_delay(task, config) * 1e6}
    elif tt in TRANSFER_TYPES:
        return {"model": "port_shared_dynamic", "delay_us": "dynamic"}
    return {"model": "unknown"}
