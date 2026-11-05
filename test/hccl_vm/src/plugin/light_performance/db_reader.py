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

"""db 读取与 BLOB 解码模块 (设计文档第 4 章)。

从 hccl_vm_data.db 只读加载 SQE 任务序列、算子参数、内存布局、CCU 通道与微码,
产出 graph_builder 可消费的 OpExecutionGroup 列表 (单 rank 视角, 4.5/4.6 节)。
另提供 eid→rank 索引构建 (build_rank_index, 4.7 节): 对端 rank 解析用。
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

logger = logging.getLogger(__name__)

JETTY_INVALID = 0xFFFFFFFF
RANK_INVALID = 0xFFFFFFFF

# 与 task_meta_defs.h HcclTaskMetaData (pack(1)) 一致 (4.4.1 节):
#   v4: 头部 69B (含 profInfo(3B) + sqeGroupId(1B)) + union 120B
#   v3 (77643105): 头部 68B; v2 (旧版): 头部 65B (按 BLOB 长度自动识别)
# taskData 布局: srcRankId/dstRankId 字段已回退删除, estimate 采集约定下
# srcDeviceId/dstDeviceId 字段承载通信域 rank (硬切换, 旧版 BLOB 不再支持)
TASK_META_LEN_V5 = 190
TASK_META_LEN = 189
TASK_META_LEN_V3 = 188
TASK_META_LEN_LEGACY = 185
TASK_DATA_OFFSET_V5 = 70
TASK_DATA_OFFSET = 69
TASK_DATA_OFFSET_V3 = 68
TASK_DATA_OFFSET_LEGACY = 65
OP_DETAIL_LEN = 28
CCU_INSTRUCTION_SIZE = 32

# OpDetailTab 列分组 (见 _load_op_detail_rows): 基线列各版本库都有;
# 后加列旧库可能尚未迁移, 缺失时按 None 处理
_OP_DETAIL_BASE_COLS = ("id", "deviceId", "rankId", "commId", "opIter",
                        "streamId", "rankSize", "devType", "opDetail")
_OP_DETAIL_LATE_COLS = ("algorithmName", "ubMultiChannelNum")


class HccLTaskMetaType(IntEnum):
    # 与 include/sim_common_defs.h HccLTaskMetaType 定义一致 (以该定义为准):
    # 7=MODEL_EXEC (模型重放边界标记, 无设备侧负载), 8=WRITE_WITH_NOTIFY,
    # 9=INVALID
    NOTIFY_WAIT = 0
    NOTIFY_RECORD = 1
    REDUCE = 2
    MEM_CPY = 3
    CCU_GRAPH = 4
    AIV_GRAPH = 5
    SYNC_STREAM = 6
    MODEL_EXEC = 7
    WRITE_WITH_NOTIFY = 8
    INVALID = 9


class DevType(IntEnum):
    # 与 HCCL dtype_common.h 中 DevType 枚举保持一致
    DEV_TYPE_950 = 6
    DEV_TYPE_960 = 8


# ---------------------------------------------------------------------------
# HcclTaskProfInfo 值域显示名 (11.4 节): 与 sim_profiling_defs.h 三个名称函数
# (ProfTaskParamTypeName/ProfLinkTypeName/ProfTransportTypeName) 逐字符一致
# (11.4.5 节对账约定)。唯一例外: 9=Reduce_With_Notify 为工具侧修正拼写
# (hcomm 源码/msprof 为错拼 Reuce_With_Notify)。
# ---------------------------------------------------------------------------
PROF_TASK_TYPE_NAMES = {
    0: "Memcpy", 1: "RDMASend", 2: "Reduce_Inline", 3: "Reduce_TBE",
    4: "Notify_Record", 5: "Notify_Wait", 6: "Send_Notify", 7: "Send_Payload",
    8: "Write_With_Notify", 9: "Reduce_With_Notify", 10: "Ccu",
    11: "AicpuKernel", 12: "Aicpu_Reduce", 13: "AivKernel",
    14: "Ub_Inline_Write", 15: "Ub_Reduce_Inline", 16: "Ub_Write_Or_Read",
    17: "DpuKernel", 18: "Dpu_ThreadFence", 19: "Dpu_ChannelFence",
    20: "Dpu_Notify_Record", 21: "Dpu_Notify_Wait", 22: "Dpu_Write_With_Notify",
    0xFF: "Unknown",
}
PROF_LINK_TYPE_NAMES = {
    0: "ONCHIP", 1: "HCCS", 2: "PCIE", 3: "ROCE", 4: "SIO", 5: "HCCS_SW",
    6: "STANDARD_ROCE", 7: "UB", 8: "UBoE", 9: "RESERVED", 0xFF: "Unknown",
}
PROF_TRANSPORT_TYPE_NAMES = {
    0: "SDMA", 1: "RDMA", 2: "LOCAL", 3: "UB", 4: "ROCE", 0xFF: "Unknown",
}


def prof_task_type_name(value: int) -> str:
    return PROF_TASK_TYPE_NAMES.get(value, "")


def prof_link_type_name(value: int) -> str:
    return PROF_LINK_TYPE_NAMES.get(value, "")


def prof_transport_type_name(value: int) -> str:
    return PROF_TRANSPORT_TYPE_NAMES.get(value, "")


_TASK_META_HEAD_V5 = struct.Struct("<BBBBQIQQIBB16s16s")   # v5: 含 profInfo(3B) + sqeGroupId(1B) + taskSrc(1B)
_TASK_META_HEAD = struct.Struct("<BBBBQIQQIB16s16s")      # v4: 含 profInfo(3B) + sqeGroupId(1B)
_TASK_META_HEAD_V3 = struct.Struct("<BBBBQIQQI16s16s")    # v3: 含 profInfo(3B)
_TASK_META_HEAD_LEGACY = struct.Struct("<BQIQQI16s16s")   # v2: 无 profInfo
_NOTIFY_TASK = struct.Struct("<IQIBB")
_TRANS_MEM_TASK = struct.Struct("<IQIQQB")
_REDUCE_TASK = struct.Struct("<IQIQQBBBB")
_WRITE_WITH_NOTIFY_TASK = struct.Struct("<IQIQQQBBBBB")
_CCU_TASK_HEAD = struct.Struct("<BBHHHII")
_SYNC_STREAM_TASK = struct.Struct("<Q")
# ModelExecTask (include/sim_common_defs.h, pack(1)):
# execSeq(uint32) + role(uint8) + modelId(uint64) + peerStreamId(uint64)
_MODEL_EXEC_TASK = struct.Struct("<IBQQ")
_OP_DETAIL_HEAD = struct.Struct("<HHH")
_OP_DETAIL_V1 = struct.Struct("<QHI")
# OpInfoV2 (include/sim_common_defs.h, pack(1), union 起始偏移 6):
#   sendCount(u64) + sendDataType(u16) + recvCount(u64) + recvDataType(u16)
#   + extInfo(u16), 共 22B → opDetail BLOB 总长 28B
_OP_DETAIL_V2 = struct.Struct("<QHQHH")
# 使用 V2 布局的算子 (level0 桩 BuildOpDetailsV2): AllToAllV(8)/
# AllToAllVC(9)/AllToAll(10); 其余算子 (含 AllGatherV/ReduceScatterV,
# V1 + opExtInfo 计数/偏移数组) 均为 V1
V2_OP_TYPES = {8, 9, 10}


# taskData dataclass: 字段名沿用 rank 语义 —— estimate 采集约定下 BLOB 中的
# srcDeviceId/dstDeviceId 字段实际承载通信域 rank (对端未解析为 0xFFFFFFFF),
# 解包后直接以 src_rank_id/dst_rank_id 命名消费 (读 srcDeviceId 当 srcRankId)。
@dataclass
class NotifyTaskData:
    src_rank_id: int
    notify_id: int
    dst_rank_id: int
    notify_count: int
    protocol: int


@dataclass
class TransMemTaskData:
    src_rank_id: int
    src_offset: int
    dst_rank_id: int
    dst_offset: int
    length: int
    protocol: int


@dataclass
class ReduceTaskData:
    src_rank_id: int
    src_offset: int
    dst_rank_id: int
    dst_offset: int
    data_size: int
    data_type: int
    reduce_op: int
    protocol: int


@dataclass
class CcuTaskData:
    die_id: int
    mission_id: int
    timeout: int
    inst_start_id: int
    inst_cnt: int
    key: int
    arg_size: int
    args: list


@dataclass
class SyncStreamTaskData:
    sync_idx: int


@dataclass
class ModelExecTaskData:
    """ModelExecTask (include/sim_common_defs.h): 模型重放触发任务,
    同一次 rtModelExecute 的 START/START_SUB/END_SUB/END 四任务共享
    execSeq (checker 据此配对建边)。无设备侧负载, light_performance
    处理时跳过 (仅告警)。"""

    exec_seq: int
    role: int
    model_id: int
    peer_stream_id: int


@dataclass
class WriteWithNotifyTaskData:
    """WriteWithNotifyTask (task_meta_defs.h): 本端 src → 远端 dst 写入后发 notify。

    len: 传输长度; isReduce=1 时为规约写载荷 (data_type/reduce_op 有效)。
    WRITE_WITH_NOTIFY 为 db 原生任务类型, 解码后单任务透传成图/仿真
    (不拆分为写 + NOTIFY_RECORD 两个子任务, 与 db 任务一一对应)。
    """
    src_rank_id: int
    src_offset: int
    dst_rank_id: int
    dst_offset: int
    length: int
    notify_id: int
    is_reduce: int
    data_type: int
    reduce_op: int
    protocol: int
    notify_count: int


@dataclass
class HcclTaskMetaData:
    task_type: int
    comm_id: int
    rank_id: int
    device_id: int
    stream_id: int
    jetty_id: int
    rm_eid: bytes
    lc_eid: bytes
    data: object
    # HcclTaskProfInfo 三字段 (4.4.1 节, v3 采集新增); 旧版 BLOB 无此字段填 0xFF=UNKNOWN
    prof_task_type: int = 0xFF
    link_type: int = 0xFF
    transport_type: int = 0xFF
    # UDMA SQE批分组标记(v4新增, 门铃批号环形递增1~255); 旧版 BLOB 无此字段填 -1(不参与合并)
    sqe_group_id: int = -1
    # 任务来源 (v5新增): 0=默认(SQE/aclrt), 1=DPU(host ibv, HcclTaskSrc); 旧版缺省 0
    src_type: int = 0


@dataclass
class MemInfo:
    input_addr: int = 0
    input_size: int = 0
    output_addr: int = 0
    output_size: int = 0
    ccl_addr: int = 0
    ccl_size: int = 0

    def _contains(self, addr: int, base: int, size: int) -> bool:
        return size > 0 and base <= addr < base + size

    def in_output(self, addr: int) -> bool:
        return self._contains(addr, self.output_addr, self.output_size)

    def in_input(self, addr: int) -> bool:
        return self._contains(addr, self.input_addr, self.input_size)

    def in_ccl(self, addr: int) -> bool:
        return self._contains(addr, self.ccl_addr, self.ccl_size)

    def region_of(self, addr: int) -> Optional[str]:
        if self.in_output(addr):
            return "output"
        if self.in_input(addr):
            return "input"
        if self.in_ccl(addr):
            return "ccl"
        return None


@dataclass
class ChannelInfo:
    """CcuChannelTab 表行 (4.4.4 节)。

    注意: 表内 srcDeviceId/srcRankId 等列均为 endpoint 设备键 (通道建立早于
    通信域, srcRankId 列实际填 device_id), 与 taskData 的 deviceId 字段
    (estimate 约定承载 rank) 是两套语义, 图代码仅消费 leid/reid/jetty。
    """
    channel_id: int
    src_device_id: int
    src_die_id: int
    dst_device_id: int
    dst_die_id: int
    src_rank_id: int
    dst_rank_id: int
    leid: bytes
    reid: bytes
    protocol: int
    jetty_ids: list


@dataclass
class RawTask:
    global_id: int
    op_detail_id: int
    device_id: int
    stream_id: int
    meta: HcclTaskMetaData
    # UDMA SQE批分组标记(取自meta.sqe_group_id); -1=无组号, 不参与合并
    sqe_group_id: int = -1


@dataclass
class OpExecutionGroup:
    comm_id: int
    op_iter: int
    op_detail_id: int
    device_id: int
    rank_id: int
    rank_size: int
    dev_type: int
    main_stream_id: int
    op_type: int
    data_size: int
    data_type: int
    reduce_op: int
    tasks: list = field(default_factory=list)
    mem_info: Optional[MemInfo] = None
    channels: dict = field(default_factory=dict)
    instr_spaces: dict = field(default_factory=dict)
    # 算法选择结果 (OpDetailTab.algorithmName, 日志截取回填; 旧库无该列时空串)
    algorithm_name: str = ""
    # UB 多 channel (jetty) 数 (OpDetailTab.ubMultiChannelNum, proxy 侧取
    # HCCL_UB_MULTI_CHANNEL_NUM): 1 = 不使能多 jetty; 旧库无该列时按 1 处理。
    # 仿真侧据此抬升 multi_port_utilization_low 表 (bandwidth_profile)
    ub_multi_channel_num: int = 1
    # V2 布局算子 (AllToAll/AllToAllV/AllToAllVC) 的收发两侧元素数与
    # 数据类型; V1 算子与 data_size/data_type 一致
    send_count: int = 0
    recv_count: int = 0
    send_data_type: int = 0
    recv_data_type: int = 0


def decode_op_task_meta(blob: bytes, row_ref: str = "") -> HcclTaskMetaData:
    if len(blob) >= TASK_META_LEN_V5:
        # v5: +taskSrc(1B); 分支须在 v4 前, 否则 190B 按 v4 解析会偏移错位
        task_type, prof_task, prof_link, prof_transport, comm_id, rank_id, \
            device_id, stream_id, jetty_id, sqe_group_id, task_src, rm_eid, lc_eid = \
            _TASK_META_HEAD_V5.unpack_from(blob, 0)
        data_offset = TASK_DATA_OFFSET_V5
    elif len(blob) >= TASK_META_LEN:
        # v4: 含 profInfo(3B) + sqeGroupId(1B)
        task_type, prof_task, prof_link, prof_transport, comm_id, rank_id, \
            device_id, stream_id, jetty_id, sqe_group_id, rm_eid, lc_eid = \
            _TASK_META_HEAD.unpack_from(blob, 0)
        task_src = 0
        data_offset = TASK_DATA_OFFSET
    elif len(blob) >= TASK_META_LEN_V3:
        # v3: 含 profInfo(3B), 无 sqeGroupId
        task_type, prof_task, prof_link, prof_transport, comm_id, rank_id, \
            device_id, stream_id, jetty_id, rm_eid, lc_eid = \
            _TASK_META_HEAD_V3.unpack_from(blob, 0)
        sqe_group_id = -1
        task_src = 0
        data_offset = TASK_DATA_OFFSET_V3
    elif len(blob) >= TASK_META_LEN_LEGACY:
        prof_task = prof_link = prof_transport = 0xFF
        task_type, comm_id, rank_id, device_id, stream_id, jetty_id, \
            rm_eid, lc_eid = _TASK_META_HEAD_LEGACY.unpack_from(blob, 0)
        sqe_group_id = -1
        task_src = 0
        data_offset = TASK_DATA_OFFSET_LEGACY
    else:
        raise ValueError(f"opTaskMeta too short: {len(blob)} < {TASK_META_LEN_LEGACY} ({row_ref})")
    if task_type in (HccLTaskMetaType.NOTIFY_WAIT, HccLTaskMetaType.NOTIFY_RECORD):
        # estimate 采集约定: srcDeviceId/dstDeviceId 字段承载通信域 rank,
        # 解包后以 rank 命名消费 (对端未解析为 0xFFFFFFFF)
        src_rank_id, notify_id, dst_rank_id, notify_count, protocol = \
            _NOTIFY_TASK.unpack_from(blob, data_offset)
        data = NotifyTaskData(src_rank_id, notify_id, dst_rank_id,
                              notify_count, protocol)
    elif task_type == HccLTaskMetaType.REDUCE:
        src_rank_id, src_offset, dst_rank_id, dst_offset, data_size, \
            data_type, reduce_op, protocol, _resv = \
            _REDUCE_TASK.unpack_from(blob, data_offset)
        data = ReduceTaskData(src_rank_id, src_offset, dst_rank_id,
                              dst_offset, data_size, data_type, reduce_op,
                              protocol)
    elif task_type == HccLTaskMetaType.MEM_CPY:
        src_rank_id, src_offset, dst_rank_id, dst_offset, length, protocol = \
            _TRANS_MEM_TASK.unpack_from(blob, data_offset)
        data = TransMemTaskData(src_rank_id, src_offset, dst_rank_id,
                                dst_offset, length, protocol)
    elif task_type == HccLTaskMetaType.CCU_GRAPH:
        die_id, mission_id, timeout, inst_start_id, inst_cnt, key, arg_size = \
            _CCU_TASK_HEAD.unpack_from(blob, data_offset)
        args = list(struct.unpack_from(
            "<13Q", blob, data_offset + _CCU_TASK_HEAD.size))
        data = CcuTaskData(die_id, mission_id, timeout, inst_start_id,
                           inst_cnt, key, arg_size, args)
    elif task_type == HccLTaskMetaType.SYNC_STREAM:
        (sync_idx,) = _SYNC_STREAM_TASK.unpack_from(blob, data_offset)
        data = SyncStreamTaskData(sync_idx)
    elif task_type == HccLTaskMetaType.MODEL_EXEC:
        exec_seq, role, model_id, peer_stream_id = \
            _MODEL_EXEC_TASK.unpack_from(blob, data_offset)
        data = ModelExecTaskData(exec_seq, role, model_id, peer_stream_id)
    elif task_type == HccLTaskMetaType.WRITE_WITH_NOTIFY:
        src_rank_id, src_offset, dst_rank_id, dst_offset, length, notify_id, \
            is_reduce, data_type, reduce_op, protocol, notify_count = \
            _WRITE_WITH_NOTIFY_TASK.unpack_from(blob, data_offset)
        data = WriteWithNotifyTaskData(src_rank_id, src_offset, dst_rank_id,
                                       dst_offset, length, notify_id,
                                       is_reduce, data_type, reduce_op,
                                       protocol, notify_count)
    else:
        raise ValueError(f"unsupported taskType: {task_type} ({row_ref})")
    return HcclTaskMetaData(task_type, comm_id, rank_id, device_id, stream_id,
                            jetty_id, rm_eid, lc_eid, data,
                            prof_task_type=prof_task, link_type=prof_link,
                            transport_type=prof_transport,
                            sqe_group_id=sqe_group_id,
                            src_type=task_src)


def _decode_op_detail(blob: bytes) -> tuple:
    """解包 opDetail BLOB → (op_type, count, data_type, reduce_op,
    send_count, recv_count, send_data_type, recv_data_type)。

    V1 算子: count/data_type 即收发两侧 (send/recv 同值);
    V2 算子 (AllToAll/AllToAllV/AllToAllVC): count/data_type 沿用发送侧
    (保持 data_size/data_type 旧消费方兼容), reduce_op 无语义置 0,
    收发两侧元素数与类型由 V2 布局解出。
    """
    op_type, data_type, reduce_type = _OP_DETAIL_HEAD.unpack_from(blob, 0)
    if op_type in V2_OP_TYPES:
        send_count, send_data_type, recv_count, recv_data_type, _ = \
            _OP_DETAIL_V2.unpack_from(blob, 6)
        return (op_type, send_count, send_data_type, 0,
                send_count, recv_count, send_data_type, recv_data_type)
    count, v1_data_type, reduce_op = _OP_DETAIL_V1.unpack_from(blob, 6)
    return (op_type, count, v1_data_type, reduce_op,
            count, count, v1_data_type, v1_data_type)


def _to_int(value) -> int:
    if isinstance(value, int):
        return value
    return int(str(value))


def _load_mem_info(cur: sqlite3.Cursor, op_detail_id: int) -> Optional[MemInfo]:
    cur.execute(
        "SELECT inputAddr, inputSize, outputAddr, outputSize, cclAddr, cclSize "
        "FROM OpMemInfoTab WHERE opDetailId=?", (op_detail_id,))
    row = cur.fetchone()
    if row is None:
        return None
    return MemInfo(_to_int(row[0]), int(row[1]), _to_int(row[2]), int(row[3]),
                   _to_int(row[4]), int(row[5]))


def _load_channels(cur: sqlite3.Cursor, device_id: int) -> dict:
    channels = {}
    cur.execute(
        "SELECT channelId, srcDieId, dstDieId, srcDeviceId, dstDeviceId, "
        "srcRankId, dstRankId, leid, reid, protocol, jettyNum, jettyId "
        "FROM CcuChannelTab WHERE srcDeviceId=?", (device_id,))
    for row in cur.fetchall():
        (channel_id, src_die, dst_die, src_dev, dst_dev, src_rank, dst_rank,
         leid, reid, protocol, jetty_num, jetty_blob) = row
        jetty_ids = []
        if jetty_blob is not None and jetty_num > 0:
            jetty_ids = list(struct.unpack_from(f"<{jetty_num}I", jetty_blob, 0))
        info = ChannelInfo(channel_id, src_dev, src_die, dst_dev, dst_die,
                           src_rank, dst_rank, bytes(leid or b""), bytes(reid or b""),
                           protocol, jetty_ids)
        channels[(src_dev, src_die, channel_id)] = info
    return channels


def _load_instr_spaces(cur: sqlite3.Cursor, device_id: int) -> dict:
    spaces = {}
    cur.execute(
        "SELECT dieId, instrCount, instrSpace FROM CcuInstrResTab WHERE deviceId=?",
        (device_id,))
    for die_id, instr_count, blob in cur.fetchall():
        if instr_count is None or instr_count <= 0 or blob is None:
            continue
        if len(blob) < instr_count * CCU_INSTRUCTION_SIZE:
            logger.warning("instrSpace too short, deviceId=%s dieId=%s "
                           "instrCount=%s blobLen=%s", device_id, die_id,
                           instr_count, len(blob))
            instr_count = len(blob) // CCU_INSTRUCTION_SIZE
        instrs = [bytes(blob[i * CCU_INSTRUCTION_SIZE:(i + 1) * CCU_INSTRUCTION_SIZE])
                  for i in range(instr_count)]
        spaces[(device_id, die_id)] = instrs
    return spaces


def _load_group_tasks(cur: sqlite3.Cursor, task_tables: list,
                      op_detail_id: int) -> list:
    rows = []
    for table_pid, table_name in task_tables:
        cur.execute(
            f"SELECT id, deviceId, streamId, taskType, opTaskMeta FROM {table_name} "
            "WHERE opDetailId=? ORDER BY id", (op_detail_id,))
        for row_id, device_id, stream_id, task_type, blob in cur.fetchall():
            rows.append((table_pid, row_id, device_id, stream_id, task_type, blob))
    rows.sort(key=lambda r: (r[0], r[1]))
    tasks = []
    global_id = 0
    for table_pid, row_id, device_id, stream_id, task_type, blob in rows:
        row_ref = f"OpTaskTab_P_{table_pid}#{row_id}"
        try:
            meta = decode_op_task_meta(bytes(blob or b""), row_ref)
        except ValueError as exc:
            logger.warning("Dropping task row %s: %s", row_ref, exc)
            continue
        # WRITE_WITH_NOTIFY 为 db 原生任务类型: 单任务透传 (不拆分),
        # 由 graph_builder 映射为 Write_With_Notify (与 db 任务一一对应)
        tasks.append(RawTask(global_id, op_detail_id, device_id,
                             stream_id, meta, meta.sqe_group_id))
        global_id += 1
    return tasks


def _load_op_detail_rows(cur: sqlite3.Cursor) -> list:
    """读取 OpDetailTab 全部行, 列序归一为 基线列 + 后加列 (缺失补 None)。

    后加列 (旧库可能尚未迁移):
      algorithmName     算法选择结果 (日志截取回填)
      ubMultiChannelNum HCCL_UB_MULTI_CHANNEL_NUM 落库值 (每 channel 的 jetty 数,
                        proxy level0 桩写入; 轻量仿真据此折算端口带宽利用率)
    先 PRAGMA table_info 探测再按名取值, 不受列顺序影响 (原实现按整条 SELECT
    失败回退, 后加列增多后无法逐列降级)。
    """
    existing = {row[1] for row in
                cur.execute("PRAGMA table_info(OpDetailTab)").fetchall()}
    cols = [c for c in _OP_DETAIL_BASE_COLS + _OP_DETAIL_LATE_COLS
            if c in existing]
    if not cols:
        # PRAGMA 对不存在的表返回空 (不报错), 此处显式给出可读原因
        raise ValueError("no OpDetailTab table in db (or none of its columns recognized)")
    missing = [c for c in _OP_DETAIL_LATE_COLS if c not in existing]
    if missing:
        logger.info("OpDetailTab missing late-added column(s) %s (old db), using defaults", missing)
    index = {c: i for i, c in enumerate(cols)}
    cur.execute(f"SELECT {', '.join(cols)} FROM OpDetailTab ORDER BY id")
    return [tuple(row[index[c]] if c in index else None
                  for c in _OP_DETAIL_BASE_COLS + _OP_DETAIL_LATE_COLS)
            for row in cur.fetchall()]


def _load_rank_eid_map(sim_db_path: str) -> dict:
    """读取 runnerdb (hccl_sim.db) RankEidMap 表: eid(hex) → rank_id (4.7 节)。

    estimate 模式 Communicator 表只有本 rank 一行, 对端 rank 以该表为准
    (sim_models.h 注释; 宿主先例 GetRmtRankIdByEid); 行 BLOB 布局与
    sim::RankEidMap 一致: id(u64) + rankId(u32) + eid(16B) + pad。
    """
    index: dict = {}
    if not sim_db_path or not os.path.isfile(sim_db_path):
        return index
    try:
        conn = sqlite3.connect(f"file:{sim_db_path}?mode=ro", uri=True)
        try:
            cur = conn.cursor()
            cur.execute("SELECT data FROM RankEidMap")
            for (blob,) in cur.fetchall():
                if blob is None or len(blob) < 28:
                    continue
                eid = bytes(blob[12:28]).hex()
                if eid:
                    index[eid] = struct.unpack_from("<I", blob, 8)[0]
        finally:
            conn.close()
    except sqlite3.Error as exc:
        logger.warning("failed to read RankEidMap, falling back to ranktable.json: %s (%s)",
                       sim_db_path, exc)
    return index


# Server.port_type 缺省值: server/pod 拓扑配置未配置 port_type 字段时为 x4
# (配置端口带宽基准); x2 时端口带宽减半, 仅影响性能仿真端口带宽计算
PORT_TYPE_DEFAULT = "x4"


def _normalize_port_type(raw) -> str:
    """端口类型归一化: 仅接受 x2/x4 (不区分大小写), 缺失/非法回退 x4。"""
    value = str(raw if raw is not None else "").strip().lower()
    if value in ("x2", "x4"):
        return value
    if value:
        logger.warning("invalid Server.port_type value %r, falling back to %s",
                       raw, PORT_TYPE_DEFAULT)
    return PORT_TYPE_DEFAULT


def load_port_type(db_path: str, device_id: Optional[int] = None) -> str:
    """读取 runnerdb (db 同目录 hccl_sim.db) Server 表的端口类型 (x2/x4)。

    port_type 来自 server/pod 拓扑配置文件的 port_type 字段 (generate_cluster_topo.sh
    写入 servers_info.json, C++ 集群解析器入库 Server.port_type)。device_id 非空时按
    Device.id → Device.server_id → Server.id 定位该设备所属 server 的 port_type;
    未定位到时回退全表唯一值。文件/表/列缺失 (旧库) 或值非法时回退默认 x4。
    """
    sim_db_path = os.path.join(
        os.path.dirname(os.path.abspath(db_path)), "hccl_sim.db")
    if not os.path.isfile(sim_db_path):
        return PORT_TYPE_DEFAULT
    raw = None
    try:
        conn = sqlite3.connect(f"file:{sim_db_path}?mode=ro", uri=True)
        try:
            cur = conn.cursor()
            server_cols = {r[1] for r in cur.execute("PRAGMA table_info(Server)")}
            if "port_type" not in server_cols:
                logger.info("Server table has no port_type column (old db), "
                            "using default %s", PORT_TYPE_DEFAULT)
                return PORT_TYPE_DEFAULT
            if device_id is not None:
                row = cur.execute(
                    "SELECT s.port_type FROM Device d "
                    "JOIN Server s ON d.server_id = s.id WHERE d.id = ? LIMIT 1",
                    (device_id,)).fetchone()
                if row is not None:
                    raw = row[0]
            if raw is None:
                rows = cur.execute(
                    "SELECT DISTINCT port_type FROM Server").fetchall()
                if len(rows) == 1:
                    raw = rows[0][0]
                elif len(rows) > 1:
                    logger.warning("Server.port_type has multiple distinct values %s, "
                                   "using default %s", [r[0] for r in rows],
                                   PORT_TYPE_DEFAULT)
        finally:
            conn.close()
    except sqlite3.Error as exc:
        logger.warning("failed to read Server.port_type from %s, using default %s: %s",
                       sim_db_path, PORT_TYPE_DEFAULT, exc)
        return PORT_TYPE_DEFAULT
    port_type = _normalize_port_type(raw)
    logger.info("server port_type: %s (device_id=%s)", port_type, device_id)
    return port_type


def build_rank_index(db_path: str, ranktable_path: str = None) -> dict:
    """eid(hex) → rank_id 索引 (4.7 节, 对端 rank 解析)。

    优先 runnerdb (db 同目录 hccl_sim.db) 的 RankEidMap 表 (ranktable 解析
    产物, mock-comm 随通信域初始化整体重写, 权威); 表空/文件缺失时回退
    ranktable.json 展开 (rank_list[].rank_id ↔ rank_addr_list[].addr)。
    """
    index = _load_rank_eid_map(os.path.join(
        os.path.dirname(os.path.abspath(db_path)), "hccl_sim.db"))
    if index:
        logger.info("eid→rank index: RankEidMap table %s entries", len(index))
        return index
    if ranktable_path and os.path.isfile(ranktable_path):
        try:
            with open(ranktable_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for rank_entry in data.get("rank_list", []):
                rank_id = rank_entry.get("rank_id")
                for level in rank_entry.get("level_list", []):
                    for addr in level.get("rank_addr_list", []):
                        if addr.get("addr_type") == "EID":
                            eid = str(addr.get("addr", "")).lower()
                            if eid and rank_id is not None:
                                index[eid] = rank_id
            if index:
                logger.info("eid→rank index: ranktable.json fallback %s entries",
                            len(index))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("failed to parse ranktable: %s (%s)", ranktable_path, exc)
    return index


def load_op_executions(db_path: str,
                       target_device: Optional[int] = None) -> list:
    """读取 db, 返回按 (commId, opIter, deviceId) 组织的单 rank 算子执行分组。"""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'OpTaskTab_P_%'")
        task_tables = sorted(
            ((int(name[len("OpTaskTab_P_"):]), name) for (name,) in cur.fetchall()),
            key=lambda t: t[0])

        detail_rows = _load_op_detail_rows(cur)
        if not detail_rows:
            logger.warning("OpDetailTab is empty: %s", db_path)
            return []

        devices = []
        for row in detail_rows:
            if row[1] not in devices:
                devices.append(row[1])
        chosen = target_device if target_device is not None else devices[0]
        if target_device is not None and chosen not in devices:
            logger.warning("target device %s not in OpDetailTab, using first device %s",
                           target_device, devices[0])
            chosen = devices[0]
        other = [d for d in devices if d != chosen]
        if other:
            logger.info("db has multi-device data %s, single-rank graph "
                        "building keeps only device %s (section 4.6)",
                        devices, chosen)

        channels = _load_channels(cur, chosen)
        instr_spaces = _load_instr_spaces(cur, chosen)

        groups = []
        for (op_detail_id, device_id, rank_id, comm_id, op_iter, stream_id,
             rank_size, dev_type, op_detail_blob, algorithm_name,
             ub_multi_channel_num) in detail_rows:
            if device_id != chosen:
                continue
            op_type = count = v1_data_type = reduce_op = 0
            send_count = recv_count = send_dt = recv_dt = 0
            if op_detail_blob is not None and len(op_detail_blob) >= OP_DETAIL_LEN:
                (op_type, count, v1_data_type, reduce_op,
                 send_count, recv_count, send_dt, recv_dt) = \
                    _decode_op_detail(bytes(op_detail_blob))
            else:
                logger.warning("invalid opDetail BLOB, opDetailId=%s", op_detail_id)
            group = OpExecutionGroup(
                comm_id=comm_id, op_iter=op_iter, op_detail_id=op_detail_id,
                device_id=device_id, rank_id=rank_id, rank_size=rank_size,
                dev_type=dev_type if dev_type is not None else 0,
                main_stream_id=stream_id, op_type=op_type, data_size=count,
                data_type=v1_data_type, reduce_op=reduce_op,
                mem_info=_load_mem_info(cur, op_detail_id),
                channels=channels, instr_spaces=instr_spaces,
                algorithm_name=(algorithm_name or ""),
                # 旧库无该列 (None) 或历史零值一律按 1 (不使能多 jetty) 处理
                ub_multi_channel_num=int(ub_multi_channel_num or 1),
                send_count=send_count, recv_count=recv_count,
                send_data_type=send_dt, recv_data_type=recv_dt)
            group.tasks = _load_group_tasks(cur, task_tables, op_detail_id)
            groups.append(group)
        return groups
    finally:
        conn.close()
