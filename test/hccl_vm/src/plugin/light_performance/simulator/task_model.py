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

"""任务模型: 任务类型枚举 + Task 数据结构。

任务类型 (时延模型分类, 内部使用):
  Notify_Record       — notify 通知 (src==dst 为 local, src!=dst 为远端直接时延)
  Notify_Wait         — notify 等待 (src==dst 为 local, src!=dst 为远端直接时延)
  Reduce_Inline       — HBM 规约 (时延 = size / hbm_bandwidth)
  Memcpy              — HBM 拷贝 (时延 = size / hbm_bandwidth)
  Ub_Inline_Write     — notify inline 写 (固定时延, 同 Notify_Record; SQE 成图
                        路径不产出, 仅旧输入路径 task.txt 使用)
  Ub_Write_Or_Read    — 网络数据传输 write/read 双方向 (端口带宽共享,
                        is_write 区分计费方向)
  Write_With_Notify   — 网络写 + 附带 notify (端口 write 方向带宽共享 +
                        写落地后 notify 固定 record 时延; 与 db 的
                        WRITE_WITH_NOTIFY (taskType=8) 任务一一对应)

trace/dump 显示名 (11.4 节): SQE 路径任务由 task_name 覆盖, 取采集侧
profInfo.profTaskType 的 op 名 (db_reader.PROF_TASK_TYPE_NAMES, 与
sim_profiling_defs.h PROF_TASK_NAMES 逐字符一致, 11.4.5 节对账约定);
无 profInfo 时 (CCU 微码子任务/CLI 输入) 回退内部 TaskType 名。
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Union

INVALID = 0xFFFFFFFF


class TaskType(Enum):
    NOTIFY_RECORD = "Notify_Record"
    NOTIFY_WAIT = "Notify_Wait"
    REDUCE_INLINE = "Reduce_Inline"
    MEMCPY = "Memcpy"
    UB_INLINE_WRITE = "Ub_Inline_Write"
    UB_WRITE_OR_READ = "Ub_Write_Or_Read"
    WRITE_WITH_NOTIFY = "Write_With_Notify"


# notify 类型集合 (用于依赖图配对)
NOTIFY_TYPES = {TaskType.NOTIFY_RECORD, TaskType.NOTIFY_WAIT}

# 数据传输类型集合 (走端口带宽共享)
TRANSFER_TYPES = {TaskType.UB_WRITE_OR_READ, TaskType.WRITE_WITH_NOTIFY}

# 固定时延同步任务 (notify inline 写, 通知远端, 同 Notify_Record)
SYNC_NOTIFY_TYPES = {TaskType.UB_INLINE_WRITE}

# HBM 访问类型集合 (时延 = size / hbm_bandwidth)
HBM_TYPES = {TaskType.REDUCE_INLINE, TaskType.MEMCPY}


class Task:
    """单个仿真任务, 包含全部属性。

    时延计算相关: task_type, src_rank, dst_rank, size, src_eid
    trace 显示相关: 全部属性
    依赖图相关: task_id, stream_id, notify_id, task_type
    notify 语义: notify_id + mask 唯一标识一个 notify 动作 (CCU 成图路径
    填 cke id 与位掩码; SQE 路径无掩码, 保持 0)
    缓冲区类型: src/dst_buf_type 仅 Memcpy / Ub_Write_Or_Read 任务携带
    (CCU 微码转换路径区分源/目的缓冲区 mem 或 ms; SQE/CLI 等其余来源
    无该信息, 保持空串, trace 输出 N/A)
    循环合并信息: loop_cnt / loop_iter_cnt 表示循环合并细节, loop_cnt 为
    展开后的 loop 数 (同一 loop 模板扩展 N 次后的 N+1), loop_iter_cnt 为
    单个 loop 自循环次数, 总迭代数 = loop_cnt × loop_iter_cnt (CCU loop
    合并发射的任务填写, 含数据任务与折叠后的 notify 任务;
    直行指令/SQE/CLI 等非合并来源保持 0 (无效), trace 输出 N/A)
    loop 合并标签: loop_tag = "loopgroup_{i}_loop_{j}" 标识任务所属的合并
    loop 发射窗口 (i 为派发上下文内 LoopGroup 序号, j 为 LoopGroup 内 loop
    序号, 6.5.1 节); 窗口内任务 (数据 + notify) 均携带, 供流水线特征链
    消解识别同一折叠发射; 非合并来源保持空串, trace 输出 N/A
    """

    __slots__ = (
        "task_id", "task_type", "task_name", "stream_id",
        "notify_id", "mask", "src_rank", "dst_rank",
        "size", "port", "src_eid", "dst_eid",
        "transport_type", "data_type", "link_type", "protocol", "jetty_id",
        "is_write", "src_buf_type", "dst_buf_type",
        "loop_cnt", "loop_iter_cnt", "loop_tag",
        "start_time", "end_time",
    )

    def __init__(
        self,
        task_id: str,
        task_type: TaskType,
        stream_id: int,
        notify_id: int = INVALID,
        mask: int = 0,
        src_rank: int = INVALID,
        dst_rank: int = INVALID,
        size: int = 0,
        port: Optional[Union[str, list[str]]] = None,
        src_eid: str = "",
        dst_eid: str = "",
        transport_type: str = "",
        data_type: str = "",
        link_type: str = "",
        protocol: str = "",
        jetty_id: int = INVALID,
        is_write: Optional[bool] = None,
        task_name: str = "",
        src_buf_type: str = "",
        dst_buf_type: str = "",
        loop_cnt: int = 0,
        loop_iter_cnt: int = 0,
        loop_tag: str = "",
        start_time: float = -1.0,
        end_time: float = -1.0,
    ):
        self.task_id = task_id
        self.task_type = task_type
        self.task_name = task_name
        self.stream_id = stream_id
        self.notify_id = notify_id
        self.mask = mask
        self.src_rank = src_rank
        self.dst_rank = dst_rank
        self.size = size
        self.port = port
        self.src_eid = src_eid
        self.dst_eid = dst_eid
        self.transport_type = transport_type
        self.data_type = data_type
        self.link_type = link_type
        self.protocol = protocol
        self.jetty_id = jetty_id
        self.is_write = is_write
        self.src_buf_type = src_buf_type
        self.dst_buf_type = dst_buf_type
        self.loop_cnt = loop_cnt
        self.loop_iter_cnt = loop_iter_cnt
        self.loop_tag = loop_tag
        self.start_time = start_time
        self.end_time = end_time

    @property
    def duration(self) -> float:
        if self.start_time < 0 or self.end_time < 0:
            return -1.0
        return self.end_time - self.start_time

    @property
    def duration_us(self) -> float:
        d = self.duration
        return d * 1e6 if d >= 0 else -1.0

    @property
    def bandwidth_gbs(self) -> float:
        """bandwidth(GB/s) = size(Byte) / duration(s) / 1e9 = size / duration_us / 1000。"""
        d = self.duration_us
        if d > 0 and self.size > 0:
            return self.size / d / 1000.0  # Byte/us → GB/s
        return 0.0

    @property
    def display_name(self) -> str:
        """trace/dump 显示名: profInfo op 名优先, 无则回退内部类型名 (11.4 节)。"""
        return self.task_name or self.task_type.value

    @property
    def is_local_notify(self) -> bool:
        """Notify_Record/Notify_Wait 是否为本地 (src_rank == dst_rank)。"""
        return self.src_rank == self.dst_rank

    def to_trace_args(self) -> dict:
        """生成 trace args 字典。"""
        args = {
            "task_type": self.display_name,
            "stream_id": self.stream_id,
            "notify_id": self.notify_id if self.notify_id != INVALID else "invalid",
            "mask": f"0x{self.mask:x}" if self.mask else "N/A",
            "src_rank": self.src_rank if self.src_rank != INVALID else "invalid",
            "dst_rank": self.dst_rank if self.dst_rank != INVALID else "invalid",
            "duration_estimated_us": round(self.duration_us, 3) if self.duration_us >= 0 else "invalid",
            "transport_type": self.transport_type or "N/A",
            "size_Byte": self.size if self.size > 0 else "N/A",
            "data_type": self.data_type or "N/A",
            "link_type": self.link_type or "N/A",
            "protocol": self.protocol or "N/A",
            "bandwidth_GBs": round(self.bandwidth_gbs, 3) if self.bandwidth_gbs > 0 else "N/A",
            "jetty_id": self.jetty_id if self.jetty_id != INVALID else "N/A",
            "src_eid": self.src_eid or "N/A",
            "dst_eid": self.dst_eid or "N/A",
            "direction": "write" if self.is_write else "read"
                         if self.is_write is not None else "N/A",
        }
        # 缓冲区类型 (mem/ms): 仅 Memcpy / Ub_Write_Or_Read 任务携带;
        # CCU 微码转换路径填有效值, 其余来源 (SQE/CLI) 无效值 N/A
        if self.task_type in (TaskType.MEMCPY, TaskType.UB_WRITE_OR_READ):
            args["src_buf_type"] = self.src_buf_type or "N/A"
            args["dst_buf_type"] = self.dst_buf_type or "N/A"
        # 循环合并信息: 全部任务统一携带; 合并折叠任务填 loop 细节 (>0),
        # loop_cnt = 展开后的 loop 数, loop_iter_cnt = 单个 loop 自循环次数;
        # 非合并来源 (直行指令/SQE/CLI) 为 0 (无效), trace 输出 N/A
        args["loop_cnt"] = self.loop_cnt if self.loop_cnt > 0 else "N/A"
        args["loop_iter_cnt"] = (self.loop_iter_cnt
                                 if self.loop_iter_cnt > 0 else "N/A")
        # loop 合并标签: 仅合并发射窗口内任务携带, 其余来源空串 (N/A)
        args["loop_tag"] = self.loop_tag or "N/A"
        # 端口列表: 数据传输任务基于 eid 查 ranktable 得到
        if self.port:
            args["ports"] = list(self.port) if isinstance(self.port, (list, tuple)) else [self.port]
        return args

    def __repr__(self) -> str:
        return f"Task({self.task_id}, {self.display_name}, stream={self.stream_id})"
