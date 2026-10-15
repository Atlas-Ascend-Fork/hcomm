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

"""CCU 微码 → 任务图生成模块 (设计文档第 6 章)。

仅实现 A5 (V1, Ascend950) 指令集; A6 (Ascend960) 暂不支持, 告警跳过。
指令为 32B 定长: 头部 2B (code:11bit | type:4bit | reserved:1bit, 小端) +
30B union 载荷 (pack(1)); 载荷偏移已对 hcomm ccu_microcode_v1.h 实测核对。

产出任务类型与 hccl_sim 新版任务模型对齐:
  SetCKE/本端 post → Notify_Record; wait → Notify_Wait
  本地搬运 (Loc↔Loc) → Memcpy (HBM)
  走网搬运 (跨 die/卡, Loc↔Rmt 读写两向) → Ub_Write_Or_Read
  规约 (Add/Max/Min) → Reduce_Inline (MS 为 rank 本地 buffer, 恒为本地 HBM 操作)
搬运任务携带 src/dst_buf_type (mem/ms) 字段: 微码指令可区分 Mem 与 MS 缓冲区,
转换后以该二字段保留区分; SQE 路径同名任务无此信息, 填无效值 N/A。
loop 合并任务携带 loop_cnt / loop_iter_cnt 字段: 表示循环合并细节,
loop_cnt = 展开后的 loop 数 (= times, 即扩展 N 次后的 N+1), loop_iter_cnt
= 单个 loop 自循环次数 (= inner_cnt), 总迭代数 = loop_cnt × loop_iter_cnt;
数据任务与折叠后的 notify 任务均携带; 直行指令/回退全展开/SQE/CLI 路径的
任务保持 0 (无效, trace 输出 N/A)。
notify 任务携带 mask 字段: notify_id (cke id) + mask (位掩码) 唯一标识一个
notify 动作; 一条 wait 指令按 checker GenWaitNode 语义聚合为至多一个本地
wait 节点 (mask 取全部贡献 post 的位或) + 一个远端 wait 节点 (未满足位)。

rank 语义 (6.7 节): 本地搬运/本地规约的 src/dst 均填本端 rank; 通道类走网任务
经 rank_map (RankEidMap 表, 缺省 ranktable.json 回退, db_reader.build_rank_index)
由对端 eid 查得, 查不到填 INVALID 并计入 rank_unresolved。UB 任务均带显式
is_write (read 按本端 eid 查端口), 不依赖 simulator 的 rank 启发式。

SQE 归并为派发上下文 (6.1 节, checker CcuMissionKey + IsContiguousCcuSqe
语义): 同 (die, mission, queue) 的 SQE 仅当指令窗口连续 (前条 instCnt==13
且首尾相接) 时并入同一上下文; 指令窗口回绕重发 (同 mission 多次 dispatch)
则新建上下文独立展开 —— 每个上下文完整重放自己的指令窗口 (含 Load
SqeArgs 头部, 用本 dispatch 的 args 重新初始化循环计数等寄存器), args
按派发序逐槽顺序消费 (读满 13 个推进到下一 SQE, checker GetSqe 语义)。
同 queue 的派发上下文严格按派发序串行推进 (单 queue 严格按指令顺序,
checker queue 语义); 寄存器/CKE 位图跨上下文全局共享 (checker
AllRankParamRecorder::Global 语义)。

单 rank 成图 (6.6 节): 跨 rank post (SyncCKE/SyncGSA/SyncXn 的远端 CKE 置位
与 TransLocMSToRmtMS 的 setRmtCKE) 生成远端 Notify_Record 节点 —— 不建跨
rank 配对边 (src≠dst → 固定 remote_record_delay); 依赖远端置位的 wait 同样
生成远端 Notify_Wait 节点 (固定 remote_wait_delay), 顶层停滞的指令在生成
远端 wait 后重放其余部分 (set/数据搬运), 循环体内的 wait 直接生成节点继续。
任务图中不凭空产生任何节点 (仅微码转换结果); CKE 位图按本 rank 寄存器模型
记账, 跨 queue 联合推进消解本地依赖。

LoopGroup 按 checker ccu_loop_merge_v3 语义做合并展开 (6.5 节): 循环体
满足前置校验 (可合并指令集 / wait-set 掩码配平 / 跨展开 CKE-MS 无冲突) 时,
折叠为单次发射, 数据任务 size × 迭代数; 校验失败回退按迭代全展开。
合并发射窗口内任务携带 loop_tag (loopgroup_{i}_loop_{j}), 供特征链消解
识别同一折叠发射 (6.5.1 节)。

流水线并发恢复 (6.5.1 节): 合并折叠把逐迭代 CKE 握手坍缩为单对
wait/record, 循环体中段的流水线并发被伪串行化 (数据任务全量等待前序
全量完成)。组装前按 loop_tag 做特征链消解 (_break_pipeline_chains):
识别"同标签数据任务 → 数据完成 record → 中段 wait"链并删除伪屏障,
后续单元 fork 自然重接前一骨架节点, 恢复流水线并发; 循环末 wait 为
真实汇合点保留。

队列并行化 (7.4 节, checker ApplyCcuQueueParallelizationV3 简化移植):
组装阶段将队列串行链切分为骨架与异步单元 —— 单元 = 数据任务 + 紧随的
NOTIFY_RECORD 链; 单元任务改挂虚拟流 (ccu_d{die}_q{queue}#v{n}) 并发执行,
fork 边 (前一骨架节点 → 单元头) 经依赖图 extra 边表达; 虚拟流按 queue 跨
mission 复用 (复用安全性由虚拟流链自身串行保证), mission 内不复用 (同
mission 单元彼此并发)。子图与 SQE 流的挂接同样经 extra 边 (7.4.1 节):
入口边 (SQE 前驱任务 → 队列骨架头) 与出口边 (mission 完成集 → SQE 后继
任务), 不引入任何占位同步节点。与 checker 的差异: 复用粒度按 mission 而非
afterNode, GRAPH_UNMATCHED 严格性降级为容忍 (性能工具)。
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass, field

from .db_reader import DevType
from .register_model import CcuRegisterModel
from .simulator.dependency_graph import DependencyGraph
from .simulator.task_model import INVALID, Task, TaskType

logger = logging.getLogger(__name__)

# CommProtocol 枚举 → protocol 字段显示名 (11.4.2 节: 通道建链协议; CCU 微码
# 子任务为 simulator 语义产物, 无 profInfo, link_type/transport_type 不填)
_PROTOCOL_NAMES = {0: "HCCS", 1: "TCP", 2: "ROCE", 3: "UB_CTP",
                   4: "UB_TP", 5: "PCIE", 6: "SIO"}

_U16 = struct.Struct("<H")
_U64 = struct.Struct("<Q")

LOAD_TYPE = 0x0
CTRL_TYPE = 0x1
TRANS_TYPE = 0x2
REDUCE_TYPE = 0x3

# 每 SQE 固定 13 个 arg 槽位 (checker CCU_SQE_ARGS_LEN); args 按派发序
# 逐槽顺序消费, 读满 13 个推进到下一 SQE (checker GetSqe 语义, 6.1 节)
CCU_SQE_ARGS_LEN = 13
# legacy 连续段合并阈值 (checker LEGACY_CCU_CONTINUATION_INST_CNT): 前一 SQE
# instCnt==13 且与后一 SQE 首尾相接时并入同一派发上下文 (IsContiguousCcuSqe)
CCU_SQE_CONTINUATION_INST_CNT = 13

LOADSQEARGSTOGSA_CODE = 0x0
LOADSQEARGSTOXN_CODE = 0x1
LOADIMDTOGSA_CODE = 0x2
LOADIMDTOXN_CODE = 0x3
LOADGSAXN_CODE = 0x4
LOADGSAGSA_CODE = 0x5
LOADXX_CODE = 0x6

LOOP_CODE = 0x0
LOOPGROUP_CODE = 0x1
SETCKE_CODE = 0x2
CLEARCKE_CODE = 0x4
JMP_CODE = 0x5

TRANSLOCMEMTOLOCMS_CODE = 0x0
TRANSRMTMEMTOLOCMS_CODE = 0x1
TRANSLOCMSTOLOCMEM_CODE = 0x2
TRANSLOCMSTORMTMEM_CODE = 0x3
TRANSRMTMSTOLOCMEM_CODE = 0x4
TRANSLOCMSTOLOCMS_CODE = 0x5
TRANSRMTMSTOLOCMS_CODE = 0x6
TRANSLOCMSTORMTMS_CODE = 0x7
TRANSRMTMEMTOLOCMEM_CODE = 0x8
TRANSLOCMEMTORMTMEM_CODE = 0x9
TRANSLOCMEMTOLOCMEM_CODE = 0xA
SYNCCKE_CODE = 0xB
SYNCGSA_CODE = 0xC
SYNCXN_CODE = 0xD

ADD_CODE = 0x0
MAX_CODE = 0x1
MIN_CODE = 0x2

# 异步单元数据任务类型 (7.4 节): 从队列串行链剥离、改挂虚拟流并发执行的
# 搬运/规约任务 (对应 checker CollectAsyncBoundRecordNodesV3 的异步角色)
_QUEUE_DATA_TYPES = {TaskType.MEMCPY, TaskType.UB_INLINE_WRITE,
                     TaskType.UB_WRITE_OR_READ, TaskType.REDUCE_INLINE}


@dataclass
class CcuPlaceholder:
    """SQE 流链中的 CCU 占位 (携带任务头部与 taskData.ccu 全部字段)。"""

    marker: str
    die_id: int
    mission_id: int
    inst_start_id: int
    inst_cnt: int
    key: int
    arg_size: int
    args: list
    queue: int
    order: int
    lc_eid: bytes = b""


@dataclass
class CcuBuildStats:
    missions: int = 0
    queues: int = 0
    instructions: int = 0
    wait_matched: int = 0
    local_posts: int = 0
    remote_posts: int = 0
    remote_waits: int = 0
    skipped_unsupported_version: int = 0
    residual_posts: int = 0
    merged_loops: int = 0
    merged_loop_iters: int = 0
    merged_loop_fallbacks: int = 0
    eid_unresolved: int = 0
    rank_unresolved: int = 0
    parallel_units: int = 0
    virtual_streams: int = 0
    fork_edges: int = 0
    join_edges: int = 0
    pipeline_chains_broken: int = 0


@dataclass
class CcuExpansionResult:
    queue_chains: dict = field(default_factory=dict)   # int 流 id → 任务链
    virtual_chains: dict = field(default_factory=dict)  # 虚拟流 id → 单元链
    marker_replacements: dict = field(default_factory=dict)
    # marker → SQE 流挂接信息 (7.4.1 节 extra 边挂接):
    #   head: 队列骨架头任务 id (骨架空时为首个单元头, 无任务为 None)
    #   entry_units: 骨架前单元的头任务 id 列表 (入口边目标)
    #   completion: mission 完成集 (骨架尾 + 全部单元尾)
    #   queue_prev_completion: 同 queue 前一 mission 完成集 (队列串行)
    attach: dict = field(default_factory=dict)
    stats: CcuBuildStats = field(default_factory=CcuBuildStats)


class _Instr:
    """解码后的单条 A5 微码指令 (载荷偏移相对 32B 指令的第 2 字节)。"""

    __slots__ = ("index", "itype", "code", "header", "_blob")

    def __init__(self, blob: bytes, index: int):
        self._blob = blob
        self.index = index
        (self.header,) = _U16.unpack_from(blob, 0)
        self.itype = (self.header >> 11) & 0xF
        self.code = self.header & 0x7FF

    def u16(self, payload_off: int) -> int:
        return _U16.unpack_from(self._blob, payload_off + 2)[0]

    def u64(self, payload_off: int) -> int:
        return _U64.unpack_from(self._blob, payload_off + 2)[0]


@dataclass
class _LoopCtx:
    """A5 循环上下文 (checker LoopGroupParam 对应物)。"""

    in_loop: bool = False
    cur_loop_idx: int = 0
    cur_expand_cnt: int = 0
    cur_loop_cnt: int = 0
    expand_offset: int = 0
    expand_cnt: int = 0
    ck_offset: int = 0
    ms_offset: int = 0
    gsa_offset: int = 0
    gsa_stride: int = 0

    def update_cke_id(self, cke_id: int) -> int:
        if not self.in_loop or self.cur_loop_idx < self.expand_offset:
            return cke_id
        return cke_id + self.cur_expand_cnt * self.ck_offset

    def update_ms_id(self, ms_id: int) -> int:
        if not self.in_loop or self.cur_loop_idx < self.expand_offset:
            return ms_id
        return ms_id + self.cur_expand_cnt * self.ms_offset

    def update_gsa(self, gsa_value: int) -> int:
        if not self.in_loop:
            return gsa_value
        value = gsa_value + self.cur_loop_cnt * self.gsa_stride
        if self.cur_loop_idx >= self.expand_offset:
            value += self.cur_expand_cnt * self.gsa_offset
        return value


# TRANS 简单搬运码 → (方向, 源缓冲区类型, 目的缓冲区类型, length 字段偏移,
# channel 字段偏移)。缓冲区类型 mem/ms 经 src/dst_buf_type 保留到任务;
# 跨 die/卡访问 (read/write) 统一映射 Ub_Write_Or_Read (is_write 区分方向),
# 本地访问 (local) 映射 Memcpy (6.3.2 节)
_TRANS_SIMPLE_FIELDS = {
    TRANSLOCMEMTOLOCMS_CODE: ("local", "mem", "ms", 6, 8),
    TRANSRMTMEMTOLOCMS_CODE: ("read", "mem", "ms", 6, 8),
    TRANSLOCMSTOLOCMEM_CODE: ("local", "ms", "mem", 6, 8),
    TRANSLOCMSTORMTMEM_CODE: ("write", "ms", "mem", 6, 8),
    TRANSRMTMSTOLOCMEM_CODE: ("read", "ms", "mem", 6, 8),
    TRANSLOCMSTOLOCMS_CODE: ("local", "ms", "ms", 4, 6),
    TRANSRMTMSTOLOCMS_CODE: ("read", "ms", "ms", 4, 6),
    TRANSLOCMSTORMTMS_CODE: ("write", "ms", "ms", 4, 6),
    TRANSRMTMEMTOLOCMEM_CODE: ("read", "mem", "mem", 8, 10),
    TRANSLOCMEMTORMTMEM_CODE: ("write", "mem", "mem", 8, 10),
    TRANSLOCMEMTOLOCMEM_CODE: ("local", "mem", "mem", 8, 10),
}


@dataclass
class _MergedSlot:
    """合并循环的单个槽位: 循环体一条指令折叠 total_iters 次后的表示。

    对应 checker CcuLoopInstrV3 系 (TransMem/Reduce/ClearCke); wait/set 掩码
    取 sample 展开 (curExpandCnt=0) 的值, 各展开经 cke/ms 偏移互不冲突
    (6.5 节合并前置校验), 故只保留一份。
    """

    kind: str                       # trans / reduce / sync / clearcke
    direction: str = ""             # trans: local/write/read
    length: int = 0                 # trans/reduce: 单次迭代字节数
    channel_id: int = 0             # trans
    src_buf_type: str = ""          # trans: 源缓冲区类型 (mem/ms)
    dst_buf_type: str = ""          # trans: 目的缓冲区类型 (mem/ms)
    ms_ids: list = field(default_factory=list)   # reduce: 8 个 ms 基址
    wait_id: int = 0
    wait_mask: int = 0
    set_id: int = 0
    set_mask: int = 0
    rmt_id: int = 0                 # sync: 远端 post cke id (0 = 无)
    rmt_mask: int = 0               # sync: 远端 post 位掩码
    clear_wait: bool = False        # trans/reduce: 消费后清 wait 位标志


@dataclass
class _MergedLoopPlan:
    """单个 Loop 的合并计划 (checker CcuLoopV3::LoopExpandMerge 结果)。"""

    total_iters: int                # loop_cnt × loop_iter_cnt (折叠迭代数)
    times: int = 1                  # 展开后的 loop 数 (>= expand_offset 后为 expand_cnt+1)
    inner_cnt: int = 1              # 单个 loop 自循环次数
    slots: list = field(default_factory=list)   # _MergedSlot 序列 (body 顺序)


class _MissionContext:
    """单个 (die, mission, queue) 的解释执行上下文。"""

    def __init__(self, device_id: int, die_id: int, mission_id: int,
                 placeholders: list, instr_space: list, channels: dict,
                 reg_model: CcuRegisterModel, dep: DependencyGraph,
                 prefix: str, stats: CcuBuildStats, stream_id: int,
                 local_rank: int, die_eids: dict, rank_map: dict = None):
        self.device_id = device_id
        self.die_id = die_id
        self.mission_id = mission_id
        self.placeholders = sorted(placeholders, key=lambda p: p.order)
        self.instr_space = instr_space
        self.channels = channels
        self.reg_model = reg_model
        self.dep = dep
        self.prefix = prefix
        self.stats = stats
        self.stream_id = stream_id
        self.local_rank = local_rank
        self.die_eids = die_eids
        self.rank_map = rank_map or {}
        self.nodes: list = []
        self._seq = 0
        # loop 合并发射窗口内的 loop_cnt / loop_iter_cnt (6.5 节):
        # _emit_merged_loop 进入时置为 (times, inner_cnt), 发射的各类任务
        # (数据/notify) 经 add_task 缺省继承; 窗口外 (直行指令) 保持 0
        # (无效, trace N/A)
        self._loop_cnt = 0
        self._loop_iter_cnt = 0
        # loop 合并发射窗口标签 (6.5.1 节): "loopgroup_{i}_loop_{j}",
        # i 为派发上下文内 LoopGroup 序号 (0 起), j 为 LoopGroup 内 loop
        # 序号 (cur_loop_idx, 0 起); 窗口内任务经 add_task 缺省继承,
        # 窗口外保持空串
        self._loop_tag = ""
        # 派发上下文内 LoopGroup 序号计数器 (标签 i 分量)
        self._lg_seq = 0
        # 已配对 wait → 消费的 post 任务 id 列表 (6.5.1 节): 发射期记录,
        # 供组装前特征链消解判定 record 的 wait 消费者
        self.wait_posts: dict = {}
        # 停滞重放标志 (6.6 节): skip_blocked 重放当前指令时置位,
        # _gen_waits 对未满足位生成远端 wait 节点而非返回停滞
        self._force_wait = False
        # JMP 求值结果 (checker TransformJmpInstrA5 语义): Xn[cond] !=
        # expectData 时记录目标指令号, 由 advance 消费移动 pc
        self._jmp_target = None
        # 推进步数预算: 防御后向跳转构成的死循环 (正常路径受窗口约束)
        self._steps = None
        # SQE 参数游标 (checker CcuGraphStateV3::GetSqe 语义): 当前派发上下文
        # 的 args 按派发序逐槽消费, 每读满 13 个 arg 推进到下一 SQE (6.1 节)
        self._sqe_idx = 0
        self.runs = self._build_runs()
        self.run_idx = 0
        self.ptr = self.runs[0][0] if self.runs else 0
        self.finished = not self.runs

    def _build_runs(self) -> list:
        frags = sorted(self.placeholders, key=lambda p: p.inst_start_id)
        runs = []
        for frag in frags:
            if runs and frag.inst_start_id == runs[-1][1]:
                runs[-1][1] = frag.inst_start_id + frag.inst_cnt
            else:
                runs.append([frag.inst_start_id,
                             frag.inst_start_id + frag.inst_cnt])
        return runs

    @property
    def first_placeholder(self) -> CcuPlaceholder:
        return self.placeholders[0]

    def _next_id(self, tag: str) -> str:
        self._seq += 1
        return f"{self.prefix}_{tag}{self._seq}"

    def _cur_index(self) -> int:
        return self.ptr

    def sqe_arg(self, sqe_args_id: int) -> int:
        """顺序消费本派发上下文 SQE 的 args (checker GetSqe 语义, 6.1 节)。

        取当前游标 SQE 的 args[sqe_args_id]; 读到第 13 个 (末槽) 后游标推进
        到下一 SQE (同一上下文内指令连续段拆批发射时, 后段 Load 指令消费
        后一 SQE 的 args)。游标越界 (全部 SQE 的 args 已耗尽) 返回 0。
        """
        if sqe_args_id >= CCU_SQE_ARGS_LEN:
            return 0
        if self._sqe_idx >= len(self.placeholders):
            return 0
        ph = self.placeholders[self._sqe_idx]
        arg_val = ph.args[sqe_args_id] if sqe_args_id < len(ph.args) else 0
        if sqe_args_id == CCU_SQE_ARGS_LEN - 1:
            self._sqe_idx += 1
        return arg_val

    def add_task(self, task_type: TaskType, tag: str,
                 loop_cnt: int = None, loop_iter_cnt: int = None,
                 loop_tag: str = None, **kwargs) -> Task:
        if loop_cnt is None:
            loop_cnt = self._loop_cnt
        if loop_iter_cnt is None:
            loop_iter_cnt = self._loop_iter_cnt
        if loop_tag is None:
            loop_tag = self._loop_tag
        task = Task(self._next_id(tag), task_type, stream_id=self.stream_id,
                    loop_cnt=loop_cnt, loop_iter_cnt=loop_iter_cnt,
                    loop_tag=loop_tag, **kwargs)
        self.dep.add_task(task)
        self.nodes.append(task.task_id)
        return task

    def fetch(self, instr_id: int) -> _Instr:
        return _Instr(self.instr_space[instr_id], instr_id)

    def _local_eid(self) -> str:
        """本端 eid: 优先 SQE 显式 lcEid, 缺省取同 die 通道 leid 近似 (10.1 节)。"""
        return self.die_eids.get(self.die_id, "")

    def _channel_eids(self, channel_id: int) -> tuple:
        """(本端 eid, 对端 eid, jetty_id, protocol), 查不到处填无效值。"""
        channel = self.channels.get((self.device_id, self.die_id, channel_id))
        if channel is None:
            return self._local_eid(), "", INVALID, ""
        local_eid = channel.leid.hex() if channel.leid else self._local_eid()
        peer_eid = channel.reid.hex() if channel.reid else ""
        jetty = channel.jetty_ids[0] if channel.jetty_ids else INVALID
        protocol = _PROTOCOL_NAMES.get(channel.protocol, "")
        return local_eid, peer_eid, jetty, protocol

    def _eid_rank(self, eid: str) -> int:
        """对端 eid → rank (6.7 节): rank_map 查得; 查不到填 INVALID 并计数。"""
        rank = self.rank_map.get(eid, INVALID) if eid else INVALID
        if rank == INVALID:
            self.stats.rank_unresolved += 1
        return rank

    def _post_local(self, cke_id: int, cke_mask: int, ctx: _LoopCtx) -> None:
        cke_id = ctx.update_cke_id(cke_id)
        if cke_mask == 0:
            return
        task = self.add_task(TaskType.NOTIFY_RECORD, "set",
                             notify_id=cke_id, mask=cke_mask,
                             src_rank=self.local_rank,
                             dst_rank=self.local_rank)
        self.reg_model.or_cke(self.die_id, cke_id, cke_mask)
        self.reg_model.register_post(task.task_id, self.die_id, cke_id,
                                     cke_mask, self.device_id, is_local=True)
        self.stats.local_posts += 1

    def _post_remote(self, cke_id: int, cke_mask: int) -> None:
        """跨 rank post (6.6 节): 生成远端 Notify_Record 节点。

        不建跨 rank 配对边 (src≠dst → 固定 remote_record_delay, 不触发
        Event), 也不登记本地 seen_post (置位动作发生在对端 rank 的 CKE)。
        cke id 不做循环偏移修正 (checker AddPost 对远端 id 不 Update)。
        """
        if cke_mask == 0:
            return
        self.add_task(TaskType.NOTIFY_RECORD, "rset",
                      notify_id=cke_id, mask=cke_mask,
                      src_rank=self.local_rank, dst_rank=INVALID)
        self.stats.remote_posts += 1

    def _gen_remote_wait(self, cke_id: int, wait_mask: int) -> None:
        """依赖远端置位的 wait 位 (6.6 节): 生成远端 Notify_Wait 节点。

        单 rank 图中对端的 record 不可见, 按固定 remote_wait_delay 仿真
        (src≠dst, 不参与本地配对)。
        """
        self.add_task(TaskType.NOTIFY_WAIT, "rwt",
                      notify_id=cke_id, mask=wait_mask,
                      src_rank=self.local_rank, dst_rank=INVALID)
        self.stats.remote_waits += 1

    def _gen_waits(self, cke_id: int, wait_mask: int, ctx: _LoopCtx) -> bool:
        cke_id = ctx.update_cke_id(cke_id)
        if wait_mask == 0:
            return True
        cur = self.reg_model.get_cke(self.die_id, cke_id)
        satisfied = cur & wait_mask
        unsat = wait_mask & ~cur
        if unsat and not (ctx.in_loop or self._force_wait):
            # 顶层 wait 位不足: 交由多 queue 联合推进的停滞检测消解
            # (6.6 节) —— 其它 queue 的本 rank post 可能尚未执行
            return False
        if satisfied:
            # 本地可满足位: 聚合为单个 wait 节点 (checker GenWaitNode 语义:
            # 一条 wait 指令最多产出 1 个本地节点, mask 取全部贡献 post 的
            # 位或, 节点与全部贡献 post 建依赖) —— 首个贡献 post 走 notify
            # 配对 (Event), 其余经 extra 边表达多父等待 (7.4 节)
            matched = self.reg_model.try_match_wait(self.die_id, cke_id,
                                                    satisfied)
            posts = [p for p in matched
                     if p.task_id not in self.reg_model.used_post_ids]
            if posts:
                local_mask = 0
                for post in posts:
                    self.reg_model.used_post_ids.add(post.task_id)
                    local_mask |= post.cke_mask & satisfied
                wait_task = self.add_task(TaskType.NOTIFY_WAIT, "wt",
                                          notify_id=cke_id, mask=local_mask,
                                          src_rank=self.local_rank,
                                          dst_rank=self.local_rank)
                self.dep.add_match(wait_task.task_id, posts[0].task_id)
                for post in posts[1:]:
                    self.dep.add_edge(wait_task.task_id, post.task_id)
                # 发射期记录 wait 消费的 post 集 (6.5.1 节): 供组装前特征链
                # 消解判定各 record 的 wait 消费者
                self.wait_posts[wait_task.task_id] = [p.task_id for p in posts]
                self.stats.wait_matched += 1
        if unsat:
            # 依赖远端置位的位: 远端 wait 节点 (固定时延, 无跨 rank 边)
            self._gen_remote_wait(cke_id, unsat)
        return True

    def _add_transfer(self, direction: str, length: int,
                      channel_id: int, tag: str,
                      src_buf_type: str = "", dst_buf_type: str = "") -> None:
        """按方向生成搬运任务 (跨 die/卡统一 Ub_Write_Or_Read, 本地 Memcpy)。

        src/dst_buf_type (mem/ms) 保留微码指令的源/目的缓冲区类型;
        方向仅决定端口计费方向 (is_write) 与 rank 端点, 不再区分任务类型。
        """
        if length <= 0:
            logger.warning("CCU 搬运长度无效, 跳过: dev=%s die=%s mission=%s "
                            "instr=%s len=%s", self.device_id, self.die_id,
                            self.mission_id, self._cur_index(), length)
            return
        if direction == "local":
            self.add_task(TaskType.MEMCPY, tag, size=length,
                          src_rank=self.local_rank, dst_rank=self.local_rank,
                          src_buf_type=src_buf_type, dst_buf_type=dst_buf_type)
        else:
            local_eid, peer_eid, jetty, protocol = self._channel_eids(channel_id)
            if not local_eid:
                self.stats.eid_unresolved += 1
            if direction == "write":
                self.add_task(TaskType.UB_WRITE_OR_READ, tag, size=length,
                              src_rank=self.local_rank,
                              dst_rank=self._eid_rank(peer_eid),
                              src_eid=local_eid, dst_eid=peer_eid,
                              jetty_id=jetty, protocol=protocol, is_write=True,
                              src_buf_type=src_buf_type,
                              dst_buf_type=dst_buf_type)
            else:
                self.add_task(TaskType.UB_WRITE_OR_READ, tag, size=length,
                              src_rank=self._eid_rank(peer_eid),
                              dst_rank=self.local_rank,
                              src_eid=peer_eid, dst_eid=local_eid,
                              jetty_id=jetty, protocol=protocol, is_write=False,
                              src_buf_type=src_buf_type,
                              dst_buf_type=dst_buf_type)

    def transform(self, instr: _Instr, ctx: _LoopCtx, count: bool = True) -> bool:
        """执行一条指令; 返回 False = 顶层 wait 未满足 (停滞)。

        instructions 统计按"成功执行"计: 停滞指令的多次尝试不计数,
        由 skip_blocked 重放成功时计一次 (避免重试双计)。
        """
        ok = self._transform(instr, ctx)
        if ok and count:
            self.stats.instructions += 1
        return ok

    def _transform(self, instr: _Instr, ctx: _LoopCtx) -> bool:
        itype, code = instr.itype, instr.code

        if itype == LOAD_TYPE:
            self._transform_load(instr, ctx)
            return True

        if itype == CTRL_TYPE:
            return self._transform_ctrl(instr, ctx)

        if itype == TRANS_TYPE:
            return self._transform_trans(instr, ctx)

        if itype == REDUCE_TYPE:
            return self._transform_reduce(instr, ctx)

        logger.warning("未支持的 A5 指令大类: type=%s code=%s header=0x%04x",
                       itype, code, instr.header)
        return True

    def _transform_load(self, instr: _Instr, ctx: _LoopCtx) -> None:
        code = instr.code
        if code in (LOADSQEARGSTOGSA_CODE, LOADSQEARGSTOXN_CODE):
            reg_id = instr.u16(0)
            arg_val = self.sqe_arg(instr.u16(2))
            if code == LOADSQEARGSTOGSA_CODE:
                self.reg_model.set_gsa(self.die_id, reg_id, arg_val)
            else:
                self.reg_model.set_xn(self.die_id, reg_id, arg_val)
        elif code == LOADIMDTOGSA_CODE:
            self.reg_model.set_gsa(self.die_id, instr.u16(0), instr.u64(2))
        elif code == LOADIMDTOXN_CODE:
            self.reg_model.set_xn(self.die_id, instr.u16(0), instr.u64(2))
        elif code == LOADGSAXN_CODE:
            base = ctx.update_gsa(self.reg_model.get_gsa(self.die_id, instr.u16(2)))
            offset = self.reg_model.get_xn(self.die_id, instr.u16(4))
            self.reg_model.set_gsa(self.die_id, instr.u16(0), base + offset)
        elif code == LOADGSAGSA_CODE:
            v1 = ctx.update_gsa(self.reg_model.get_gsa(self.die_id, instr.u16(2)))
            v2 = self.reg_model.get_gsa(self.die_id, instr.u16(4))
            self.reg_model.set_gsa(self.die_id, instr.u16(0), v1 + v2)
        elif code == LOADXX_CODE:
            v1 = self.reg_model.get_xn(self.die_id, instr.u16(2))
            v2 = self.reg_model.get_xn(self.die_id, instr.u16(4))
            self.reg_model.set_xn(self.die_id, instr.u16(0), v1 + v2)
        else:
            logger.warning("未支持的 A5 LOAD 指令: header=0x%04x", instr.header)

    def _transform_ctrl(self, instr: _Instr, ctx: _LoopCtx) -> bool:
        code = instr.code
        if code == SETCKE_CODE:
            wait_id, wait_mask = instr.u16(6), instr.u16(8)
            if not self._gen_waits(wait_id, wait_mask, ctx):
                return False
            self._post_local(instr.u16(2), instr.u16(4), ctx)
            if instr.u16(0) & 0x1:
                self.reg_model.clear_cke(self.die_id, ctx.update_cke_id(wait_id),
                                         wait_mask)
            return True
        if code == CLEARCKE_CODE:
            wait_id, wait_mask = instr.u16(6), instr.u16(8)
            if not self._gen_waits(wait_id, wait_mask, ctx):
                return False
            self.reg_model.clear_cke(self.die_id,
                                     ctx.update_cke_id(instr.u16(2)),
                                     instr.u16(4))
            return True
        if code == LOOPGROUP_CODE:
            return self._exec_loop_group(instr)
        if code == LOOP_CODE:
            logger.warning("A5 Loop 指令须由 LoopGroup 触发, 跳过: instr=%s",
                           instr.index)
            return True
        if code == JMP_CODE:
            if ctx.in_loop:
                # 循环体内 JMP: 合并前置校验已拒绝含 JMP 的循环体, 全展开
                # 回退路径按直通处理 (保守)
                logger.warning("循环体内 JMP 指令按直通处理 (不跳转): instr=%s",
                               instr.index)
                return True
            # A5 Jmp: Xn[conditionXnId] != expectData 时跳至 Xn[dstInstrXnId]
            # 所指指令 (绝对指令号); 目标经 advance 消费
            cond = self.reg_model.get_xn(self.die_id, instr.u16(2))
            if cond != instr.u64(4):
                self._jmp_target = self.reg_model.get_xn(self.die_id,
                                                         instr.u16(0))
            return True
        logger.warning("未支持的 A5 CTRL 指令: header=0x%04x", instr.header)
        return True

    def _transform_trans(self, instr: _Instr, ctx: _LoopCtx) -> bool:
        code = instr.code

        if code in (SYNCCKE_CODE, SYNCGSA_CODE, SYNCXN_CODE):
            wait_id, wait_mask = instr.u16(26), instr.u16(28)
            if not self._gen_waits(wait_id, wait_mask, ctx):
                return False
            if code == SYNCCKE_CODE:
                # SyncCKE: LocCKE 位同步到远端 (checker AddPost(rmtCKEId,
                # locCKEMask)); Xn/GSA 值同步仅为远端寄存器写, 不建任务
                self._post_remote(instr.u16(0), instr.u16(4))
            else:
                # SyncGSA/SyncXn: Set rmtCKE (AddPost(setRmtCKEId,
                # setRmtCKEMask))
                self._post_remote(instr.u16(8), instr.u16(10))
            self._post_local(instr.u16(22), instr.u16(24), ctx)
            self.reg_model.clear_cke(self.die_id, ctx.update_cke_id(wait_id),
                                     wait_mask)
            return True

        simple = _TRANS_SIMPLE_FIELDS
        if code in simple:
            direction, src_buf, dst_buf, length_at, channel_at = simple[code]
            wait_id, wait_mask = instr.u16(26), instr.u16(28)
            if not self._gen_waits(wait_id, wait_mask, ctx):
                return False
            length = self.reg_model.get_xn(self.die_id, instr.u16(length_at))
            self._add_transfer(direction, length,
                               instr.u16(channel_at), "mv",
                               src_buf_type=src_buf, dst_buf_type=dst_buf)
            self._post_local(instr.u16(22), instr.u16(24), ctx)
            if instr.u16(20) & 0x1:
                self.reg_model.clear_cke(self.die_id, ctx.update_cke_id(wait_id),
                                         wait_mask)
            if code == TRANSLOCMSTORMTMS_CODE:
                # TransLocMSToRmtMS: setRmtCKE 伴随的跨 rank post
                self._post_remote(instr.u16(8), instr.u16(10))
            return True

        logger.warning("未支持的 A5 TRANS 指令: header=0x%04x", instr.header)
        return True

    def _transform_reduce(self, instr: _Instr, ctx: _LoopCtx) -> bool:
        wait_id, wait_mask = instr.u16(26), instr.u16(28)
        if not self._gen_waits(wait_id, wait_mask, ctx):
            return False
        length = self.reg_model.get_xn(self.die_id, instr.u16(16))
        # MS 为 rank 本地 buffer (checker GenSliceFromMs/AddLocalBatchReduce
        # 对全部 msId 恒本地), 规约恒为本地 HBM 操作, 不按 msId 高位判远端
        if length > 0:
            self.add_task(TaskType.REDUCE_INLINE, "rc", size=length,
                          src_rank=self.local_rank,
                          dst_rank=self.local_rank)
        self._post_local(instr.u16(22), instr.u16(24), ctx)
        if instr.u16(20) & 0x1:
            self.reg_model.clear_cke(self.die_id, ctx.update_cke_id(wait_id),
                                     wait_mask)
        return True

    def _exec_loop_group(self, instr: _Instr) -> bool:
        start_loop_instr_id = instr.u16(0)
        group_xn = self.reg_model.get_xn(self.die_id, instr.u16(2))
        group_xm = self.reg_model.get_xn(self.die_id, instr.u16(4))
        loop_cnt = (group_xn >> 41) & 0x7F
        expand_offset = (group_xn >> 48) & 0x7F
        expand_cnt = (group_xn >> 55) & 0x7F
        ctx = _LoopCtx(in_loop=True, expand_offset=expand_offset,
                       expand_cnt=expand_cnt, ck_offset=group_xm & 0x3FF,
                       ms_offset=(group_xm >> 10) & 0x7FF,
                       gsa_offset=(group_xm >> 21) & 0xFFFFFFFF)
        # LoopGroup 序号 (6.5.1 节标签 i 分量): 派发上下文内 0 起
        lg_seq = self._lg_seq
        self._lg_seq += 1
        for cur_loop_idx in range(loop_cnt):
            ctx.cur_loop_idx = cur_loop_idx
            loop_instr_id = start_loop_instr_id + cur_loop_idx
            if loop_instr_id >= len(self.instr_space):
                logger.warning("LoopGroup 指针越界: instr=%s", loop_instr_id)
                continue
            loop_instr = self.fetch(loop_instr_id)
            if loop_instr.itype != CTRL_TYPE or loop_instr.code != LOOP_CODE:
                logger.warning("LoopGroup 指向非 Loop 指令: instr=%s "
                                "header=0x%04x", loop_instr_id, loop_instr.header)
                continue
            loop_xm = self.reg_model.get_xn(self.die_id, loop_instr.u16(4))
            inner_cnt = loop_xm & 0x1FFF
            ctx.gsa_stride = (loop_xm >> 13) & 0xFFFFFFFF
            times = 1 if cur_loop_idx < expand_offset else expand_cnt + 1
            plan = self._plan_merged_loop(loop_instr, ctx, inner_cnt, times)
            if plan is not None:
                if not self._emit_merged_loop(
                        plan, ctx,
                        loop_tag=f"loopgroup_{lg_seq}_loop_{cur_loop_idx}"):
                    return False
                continue
            if inner_cnt > 0:
                self.stats.merged_loop_fallbacks += 1
            for expand_no in range(times):
                ctx.cur_expand_cnt = expand_no
                for cur_loop_cnt in range(inner_cnt):
                    ctx.cur_loop_cnt = cur_loop_cnt
                    if not self._exec_range(loop_instr.u16(0), loop_instr.u16(2),
                                            ctx):
                        return False
        return True

    def _plan_merged_loop(self, loop_instr: _Instr, ctx: _LoopCtx,
                          inner_cnt: int, times: int) -> _MergedLoopPlan | None:
        """按 checker ccu_loop_merge_v3 语义收集并校验循环体 (6.5 节)。

        校验不过返回 None (调用方回退全展开):
          - 循环体仅含可合并指令: 简单 TRANS / SYNC TRANS / REDUCE / ClearCKE
            (SetCKE/LOAD/LOOP/JMP 等按 checker CollectLoopCkeOpsA5 规则拒绝)
          - wait/set 掩码收支配平 (ConsumeProducedMask/ProduceMask, 末尾全零)
          - 跨展开 CKE/MS 资源无冲突 (CheckResourceConflict)
          - 搬运长度 > 0 (ValidateLoopLen), ClearCKE clearMask == 0
        """
        def bail(reason: str):
            logger.warning("循环体不满足合并条件 (%s), 回退全展开: dev=%s "
                            "die=%s mission=%s instr=%s", reason, self.device_id,
                            self.die_id, self.mission_id, loop_instr.index)
            return None

        if inner_cnt <= 0 or times <= 0:
            return None
        start_id, end_id = loop_instr.u16(0), loop_instr.u16(2)
        if end_id >= len(self.instr_space) or start_id > end_id:
            return None
        sample = _LoopCtx(in_loop=True, cur_loop_idx=ctx.cur_loop_idx,
                          expand_offset=ctx.expand_offset, expand_cnt=ctx.expand_cnt,
                          ck_offset=ctx.ck_offset, ms_offset=ctx.ms_offset,
                          gsa_offset=ctx.gsa_offset, gsa_stride=ctx.gsa_stride)
        slots = []
        produced: dict = {}
        for instr_id in range(start_id, end_id + 1):
            ins = self.fetch(instr_id)
            itype, code = ins.itype, ins.code
            if itype == TRANS_TYPE and code in _TRANS_SIMPLE_FIELDS:
                direction, src_buf, dst_buf, length_at, channel_at = \
                    _TRANS_SIMPLE_FIELDS[code]
                if code == TRANSLOCMSTORMTMS_CODE and ins.u16(10):
                    # setRmtCKEMask != 0: 远端 post 每迭代触发, 回退全展开
                    return bail("TransMem setRmtCKEMask != 0")
                length = self.reg_model.get_xn(self.die_id, ins.u16(length_at))
                if length <= 0:
                    return bail(f"搬运长度无效 len={length}")
                slots.append(_MergedSlot(
                    kind="trans", direction=direction, length=length,
                    channel_id=ins.u16(channel_at),
                    src_buf_type=src_buf, dst_buf_type=dst_buf,
                    wait_id=ins.u16(26), wait_mask=ins.u16(28),
                    set_id=ins.u16(22), set_mask=ins.u16(24),
                    clear_wait=bool(ins.u16(20) & 0x1)))
            elif itype == TRANS_TYPE and code in (SYNCCKE_CODE, SYNCGSA_CODE,
                                                  SYNCXN_CODE):
                if code == SYNCCKE_CODE:
                    rmt_id, rmt_mask = ins.u16(0), ins.u16(4)
                else:
                    rmt_id, rmt_mask = ins.u16(8), ins.u16(10)
                slots.append(_MergedSlot(
                    kind="sync", wait_id=ins.u16(26), wait_mask=ins.u16(28),
                    set_id=ins.u16(22), set_mask=ins.u16(24),
                    rmt_id=rmt_id, rmt_mask=rmt_mask))
            elif itype == REDUCE_TYPE:
                length = self.reg_model.get_xn(self.die_id, ins.u16(16))
                if length <= 0:
                    return bail(f"规约长度无效 len={length}")
                slots.append(_MergedSlot(
                    kind="reduce", length=length,
                    ms_ids=[ins.u16(2 * i) for i in range(8)],
                    wait_id=ins.u16(26), wait_mask=ins.u16(28),
                    set_id=ins.u16(22), set_mask=ins.u16(24),
                    clear_wait=bool(ins.u16(20) & 0x1)))
            elif itype == CTRL_TYPE and code == CLEARCKE_CODE:
                if ins.u16(4):
                    return bail("ClearCKE clearMask != 0")
                slots.append(_MergedSlot(
                    kind="clearcke", wait_id=ins.u16(6), wait_mask=ins.u16(8)))
            else:
                return bail(f"不可合并指令 type={itype} code={code}")
            slot = slots[-1]
            if slot.wait_mask:
                wait_key = sample.update_cke_id(slot.wait_id)
                if (produced.get(wait_key, 0) & slot.wait_mask) != slot.wait_mask:
                    return bail(f"wait 掩码超出循环体内产出 cke={wait_key} "
                                f"mask=0x{slot.wait_mask:x}")
                produced[wait_key] = produced.get(wait_key, 0) & ~slot.wait_mask
            if slot.set_mask:
                set_key = sample.update_cke_id(slot.set_id)
                produced[set_key] = produced.get(set_key, 0) | slot.set_mask
        if any(produced.values()):
            return bail(f"set 掩码未在循环体内耗尽 {produced}")

        total_iters = inner_cnt * times
        if times > 1:
            seen_cke: dict = {}
            seen_ms: dict = {}
            for expand_no in range(times):
                sample.cur_expand_cnt = expand_no
                for slot in slots:
                    for cke_id, mask in ((slot.wait_id, slot.wait_mask),
                                         (slot.set_id, slot.set_mask)):
                        if not mask:
                            continue
                        key = sample.update_cke_id(cke_id)
                        if key in seen_cke and seen_cke[key] != expand_no:
                            return bail(f"跨展开 CKE 冲突 cke={key}")
                        seen_cke[key] = expand_no
                for slot in slots:
                    if slot.kind != "reduce":
                        continue
                    for ms_id in slot.ms_ids:
                        key = sample.update_ms_id(ms_id)
                        if key in seen_ms and seen_ms[key] != expand_no:
                            return bail(f"跨展开 MS 冲突 ms={key}")
                        seen_ms[key] = expand_no
        return _MergedLoopPlan(total_iters=total_iters, times=times,
                               inner_cnt=inner_cnt, slots=slots)

    def _emit_merged_loop(self, plan: _MergedLoopPlan, ctx: _LoopCtx,
                          loop_tag: str = "") -> bool:
        """发射合并循环: 循环体执行一次 (sample 展开) + 数据量 × total_iters。

        checker EmitMergedLoopInstrsV3 语义: 逐槽位 wait → 数据任务 → set →
        清 wait 位。数据任务 size = 单次长度 × total_iters (时序等价折叠);
        其余展开的 CKE 位图按 or/clear 幂等性重放, 保证循环后寄存器状态与
        全展开一致 (仅 PostRecord 记账不重复建立)。发射窗口内全部任务
        (数据任务 + 折叠后的 notify 任务) 携带 loop_cnt = times (展开后的
        loop 数) / loop_iter_cnt = inner_cnt (单个 loop 自循环次数) 与
        loop_tag (合并标签, 6.5.1 节, 供特征链消解与后续性能仿真识别同一
        折叠发射); 窗口外保持 0 / 空串 (trace N/A)。
        """
        total = plan.total_iters
        ctx.cur_expand_cnt = 0
        ctx.cur_loop_cnt = 0
        self._loop_cnt = plan.times
        self._loop_iter_cnt = plan.inner_cnt
        self._loop_tag = loop_tag
        try:
            for slot in plan.slots:
                self.stats.instructions += 1
                if slot.wait_mask:
                    if not self._gen_waits(slot.wait_id, slot.wait_mask, ctx):
                        return False
                if slot.kind == "trans":
                    self._add_transfer(slot.direction, slot.length * total,
                                       slot.channel_id, "mv",
                                       src_buf_type=slot.src_buf_type,
                                       dst_buf_type=slot.dst_buf_type)
                elif slot.kind == "reduce":
                    # MS 为 rank 本地 buffer, 恒为本地 HBM 规约 (6.3.2 节)
                    self.add_task(TaskType.REDUCE_INLINE, "rc",
                                  size=slot.length * total,
                                  src_rank=self.local_rank,
                                  dst_rank=self.local_rank)
                elif slot.kind == "sync":
                    # 远端 post 折叠为单节点 (固定时延, 与数据任务折叠一致)
                    self._post_remote(slot.rmt_id, slot.rmt_mask)
                if slot.set_mask:
                    self._post_local(slot.set_id, slot.set_mask, ctx)
                if slot.wait_mask and (slot.clear_wait or slot.kind == "sync"):
                    self.reg_model.clear_cke(
                        self.die_id, ctx.update_cke_id(slot.wait_id), slot.wait_mask)
        finally:
            self._loop_cnt = 0
            self._loop_iter_cnt = 0
            self._loop_tag = ""
        # 其余展开的寄存器位图重放 (or/clear 幂等 → 各展开折叠为一次)
        for expand_no in range(1, plan.times):
            ctx.cur_expand_cnt = expand_no
            for slot in plan.slots:
                if slot.set_mask:
                    self.reg_model.or_cke(
                        self.die_id, ctx.update_cke_id(slot.set_id),
                        slot.set_mask)
                if slot.wait_mask and (slot.clear_wait or slot.kind == "sync"):
                    self.reg_model.clear_cke(
                        self.die_id, ctx.update_cke_id(slot.wait_id),
                        slot.wait_mask)
        ctx.cur_expand_cnt = 0
        self.stats.merged_loops += 1
        self.stats.merged_loop_iters += total
        return True

    def _exec_range(self, start_id: int, end_id: int, ctx: _LoopCtx) -> bool:
        for instr_id in range(start_id, end_id + 1):
            if instr_id >= len(self.instr_space):
                logger.warning("循环体指令指针越界: instr=%s", instr_id)
                break
            if not self.transform(self.fetch(instr_id), ctx):
                return False
        return True

    def advance(self) -> bool:
        """推进到窗口尾或被 wait 阻塞; 返回是否发生推进。"""
        before = (self.run_idx, self.ptr)
        if self._steps is None:
            self._steps = 64 * sum(end - start
                                   for start, end in self.runs) + 4096
        while self.run_idx < len(self.runs):
            start, end = self.runs[self.run_idx]
            if self.ptr < start:
                self.ptr = start
            while self.ptr < end:
                if self._steps <= 0:
                    logger.warning("CCU 指令步数预算耗尽, 强制结束 (疑似跳转"
                                   "死循环): dev=%s die=%s mission=%s",
                                   self.device_id, self.die_id,
                                   self.mission_id)
                    self.finished = True
                    return True
                self._steps -= 1
                if self.ptr >= len(self.instr_space):
                    logger.warning("指令指针越界: dev=%s die=%s mission=%s "
                                   "ptr=%s", self.device_id, self.die_id,
                                   self.mission_id, self.ptr)
                    self.ptr = end
                    break
                instr = self.fetch(self.ptr)
                if not self.transform(instr, _LoopCtx()):
                    return (self.run_idx, self.ptr) != before
                if self._jmp_target is not None:
                    # 条件成立: pc 跳至目标 (限当前 run 窗口内, 目标==end
                    # 等价跳到窗口尾自然结束)
                    target = self._jmp_target
                    self._jmp_target = None
                    if start <= target <= end \
                            and target < len(self.instr_space):
                        self.ptr = target
                    else:
                        logger.warning("JMP 目标越界 (target=%s window=[%s,"
                                       "%s)), 按直通处理: dev=%s die=%s "
                                       "mission=%s instr=%s", target, start,
                                       end, self.device_id, self.die_id,
                                       self.mission_id, instr.index)
                        self.ptr += 1
                else:
                    self.ptr += 1
            self.run_idx += 1
            if self.run_idx < len(self.runs):
                self.ptr = self.runs[self.run_idx][0]
        self.finished = True
        return (self.run_idx, self.ptr) != before

    def skip_blocked(self) -> None:
        """停滞消解 (6.6 节): 当前指令的 wait 依赖远端 post (单 rank 图不可见)。

        生成远端 Notify_Wait 节点 (固定时延) 后重放本条指令的其余部分
        (本地 set / 数据搬运); 部分满足的位在重放中与本地 post 正常配对。
        """
        logger.warning("wait 依赖远端 post, 生成远端 wait 节点并继续: "
                        "dev=%s die=%s mission=%s instr=%s",
                        self.device_id, self.die_id, self.mission_id, self.ptr)
        self._force_wait = True
        try:
            self.transform(self.fetch(self.ptr), _LoopCtx())
        finally:
            self._force_wait = False
        self.ptr += 1
        if self.run_idx < len(self.runs) and self.ptr >= self.runs[self.run_idx][1]:
            self.run_idx += 1
            if self.run_idx < len(self.runs):
                self.ptr = self.runs[self.run_idx][0]
            else:
                self.finished = True


def _break_pipeline_chains(ctx: _MissionContext, dep: DependencyGraph) -> int:
    """合并 loop 特征链消解 (6.5.1 节): 删除折叠产生的伪串行屏障。

    合并折叠把逐迭代 CKE 握手坍缩为单对 wait/record, 循环体中段的流水线
    并发被伪串行化 (后段数据任务全量等待前段全量完成)。特征链 = 同一
    loop_tag 发射窗口内的 [数据任务 M* → 数据完成 record R*] 与消费 R*
    的中段 wait W。识别条件 (全部满足才删, 任一不满足保守跳过):
      1. W 为本地已配对 Notify_Wait 且携带 loop_tag T
      2. W 的全部 post (发射期 wait_posts 记录) 均为携带 T 的
         Notify_Record, 且在节点序中紧随携带 T 的数据任务 M*
         (数据完成通知; 排除游离 set / 跨阶段 post / sync 产物)
      3. W 之后仍存在携带 T 的数据任务 (循环体中段判定; 循环末 wait
         为真实汇合点, 其等待语义在全量折叠下近似成立, 保留)
    删除: W 恒删 (伪屏障); R* 仅当无其它 wait 消费者时删 (被其它 wait
    消费的 record 保留, 保守分支)。删除后骨架/单元切分自然把后续单元
    fork 重接至前一骨架节点, 恢复流水线并发。逐链消解后重扫 (R* 的
    其它消费者可能随其 wait 删除而清空, 多级链逐级放行)。

    返回消解的特征链条数。
    """
    broken = 0
    while True:
        target = None
        nodes = ctx.nodes
        for w_idx, wait_id in enumerate(nodes):
            wait_task = dep.get_task(wait_id)
            if (wait_task.task_type != TaskType.NOTIFY_WAIT
                    or not wait_task.loop_tag
                    or not wait_task.is_local_notify):
                continue
            posts = ctx.wait_posts.get(wait_id)
            if not posts:
                continue
            tag = wait_task.loop_tag
            # 条件 3: W 之后存在同标签数据任务 (循环体中段)
            if not any(dep.get_task(nid).task_type in _QUEUE_DATA_TYPES
                       and dep.get_task(nid).loop_tag == tag
                       for nid in nodes[w_idx + 1:]):
                continue
            # 条件 2: 全部 post 为同标签数据完成 record
            chain_ok = True
            for rid in posts:
                try:
                    r_idx = nodes.index(rid)
                except ValueError:
                    chain_ok = False
                    break
                if r_idx == 0:
                    chain_ok = False
                    break
                rec = dep.get_task(rid)
                prev_task = dep.get_task(nodes[r_idx - 1])
                if (rec.task_type != TaskType.NOTIFY_RECORD
                        or rec.loop_tag != tag
                        or prev_task.task_type not in _QUEUE_DATA_TYPES
                        or prev_task.loop_tag != tag):
                    chain_ok = False
                    break
            if not chain_ok:
                continue
            target = (wait_id, posts)
            break
        if target is None:
            return broken
        wait_id, posts = target
        # 其它 wait 消费者集合 (保留分支判定); 已删除 wait 的记录已弹出,
        # 不再视为消费者
        other_consumers = {rid for wid, rids in ctx.wait_posts.items()
                           if wid != wait_id for rid in rids}
        dep.remove_task(wait_id)
        ctx.wait_posts.pop(wait_id, None)
        for rid in posts:
            if rid in other_consumers:
                continue
            dep.remove_task(rid)
        nodes.remove(wait_id)
        for rid in posts:
            if rid not in other_consumers:
                nodes.remove(rid)
        broken += 1


def expand_ccu_graph(device_id: int, dev_type: int, placeholders: list,
                     instr_spaces: dict, channels: dict,
                     dep: DependencyGraph, local_rank: int = 0,
                     alloc_stream=None, rank_map: dict = None) -> CcuExpansionResult:
    """展开全部 CCU 占位 (设计文档 6.6 节)。

    placeholders 为 CcuPlaceholder 列表 (派发顺序); instr_spaces 为
    (dev, die) → 32B 指令 bytes 列表; channels 为 (srcDev, srcDie,
    channelId) → ChannelInfo。local_rank 为本端 rank (notify 任务
    src==dst 判本地); rank_map 为 eid(hex) → rank 索引 (6.7 节对端
    rank 解析); alloc_stream(key)→int 为流 id 分配器 (同名复用,
    CCU 队列链据此合并)。返回 queue 链 (int 流 id) → 任务 id 列表、
    虚拟链 (虚拟流 id → 异步单元任务 id 列表, 7.4 节)、占位 → SQE 流
    替换对 (占位被移除, 替换列表恒空)、SQE 挂接信息 (attach, 7.4.1
    节)、统计信息。
    """
    result = CcuExpansionResult()
    stats = result.stats
    if not placeholders:
        return result

    if alloc_stream is None:
        _next = [1 << 20]
        _memo: dict = {}

        def alloc_stream(key):
            # "s{n}" (SQE 流) 直用原始流号, 与 GraphBuilder._sqe_stream 一致
            if key.startswith("s") and key[1:].isdigit() \
                    and int(key[1:]) < (1 << 20):
                return int(key[1:])
            if key not in _memo:
                _memo[key] = _next[0]
                _next[0] += 1
            return _memo[key]

    if dev_type == int(DevType.DEV_TYPE_960):
        stats.skipped_unsupported_version = len(placeholders)
        logger.warning("devType=%s (Ascend960/A6) 指令集暂不支持, 跳过全部 "
                       "%s 个 CCU 任务", dev_type, len(placeholders))
        for ph in placeholders:
            result.marker_replacements[ph.marker] = []
        return result

    # die → 本端 eid 表: 通道 leid 近似打底, SQE 显式 lcEid (v2 新增)
    # 覆盖同 die 的近似值 (10.1 节缺口)
    die_eids: dict = {}
    for (src_dev, die_id, _chid), info in channels.items():
        if src_dev == device_id and info.leid:
            die_eids.setdefault(die_id, info.leid.hex())
    for ph in placeholders:
        if any(ph.lc_eid or b""):
            die_eids[ph.die_id] = ph.lc_eid.hex()

    # 本端 rank 自校验 (6.7 节): 通道 leid ↔ rank 索引应与 opDetails.rankId
    # 一致, 不一致以 opDetails 为准并告警 (eid 编号系错位防护)
    rank_map = rank_map or {}
    warned: set = set()
    for (src_dev, _die, _chid), info in channels.items():
        if src_dev != device_id or not info.leid:
            continue
        eid = info.leid.hex()
        if eid in warned:
            continue
        eid_rank = rank_map.get(eid)
        if eid_rank is not None and eid_rank != local_rank:
            warned.add(eid)
            logger.warning("通道本端 eid 的 rank 映射 (%s) 与 opDetails.rankId "
                           "(%s) 不一致, 以 opDetails 为准: leid=%s",
                           eid_rank, local_rank, eid)

    # SQE 归并为派发上下文 (checker CcuMissionKey + IsContiguousCcuSqe 语义,
    # 6.1 节): 同 (die, mission, queue) 的 SQE 仅当与前一条连续 (前条
    # instCnt==13 且 instStartId 首尾相接) 时并入同一上下文, 否则新建 ——
    # 同一 mission 的多次 dispatch (指令窗口回绕重发) 各自独立展开: 每个上下文
    # 完整重放自己的指令窗口 (含 Load SqeArgs 头部, 用本 dispatch 的 args
    # 重新初始化寄存器), 避免后一 dispatch 沿用前次残留的循环计数等寄存器
    # 状态导致数据搬运丢失; 寄存器/CKE 位图仍全局共享 (checker
    # AllRankParamRecorder::Global 语义)。
    groups: dict = {}
    for ph in sorted(placeholders, key=lambda p: p.order):
        segs = groups.setdefault((ph.die_id, ph.mission_id, ph.queue), [])
        if segs:
            last = segs[-1][-1]
            if (last.inst_cnt == CCU_SQE_CONTINUATION_INST_CNT
                    and ph.inst_start_id == last.inst_start_id + last.inst_cnt):
                segs[-1].append(ph)
                continue
        segs.append([ph])

    reg_model = CcuRegisterModel(device_id)
    contexts = []
    for (die_id, mission_id, queue), segs in groups.items():
        instr_space = instr_spaces.get((device_id, die_id))
        if not instr_space:
            logger.warning("无微码指令空间, 跳过: dev=%s die=%s mission=%s",
                           device_id, die_id, mission_id)
            stats.skipped_unsupported_version += sum(len(seg) for seg in segs)
            continue
        base_prefix = f"ccu_d{die_id}_q{queue}_m{mission_id}"
        stream_id = alloc_stream(f"ccu_d{die_id}_q{queue}")
        for seg_idx, seg in enumerate(segs):
            # 多 dispatch 上下文共用 mission 前缀时任务 id 会冲突, 追加段序号
            prefix = base_prefix if len(segs) == 1 \
                else f"{base_prefix}_s{seg_idx}"
            ctx = _MissionContext(device_id, die_id, mission_id, seg,
                                  instr_space, channels, reg_model, dep,
                                  prefix, stats, stream_id, local_rank,
                                  die_eids, rank_map=rank_map)
            contexts.append(ctx)
            stats.missions += 1
    # 全局派发序: 组装阶段 (骨架/单元切分、虚拟流复用、队列串行链) 与
    # checker 的 TaskCcuGraph 流序对齐, 同 queue 上下文按此序串行
    contexts.sort(key=lambda c: c.first_placeholder.order)

    # 同 queue 的派发上下文严格按派发序串行推进 (checker queue 语义:
    # 单个 queue 严格按指令顺序推进, 后一 TaskCcuGraph 须等前一完成);
    # 跨 queue 轮询联合推进, 本地 post/wait 配对经共享寄存器模型消解
    queues: dict = {}
    for ctx in contexts:
        queues.setdefault((ctx.die_id, ctx.first_placeholder.queue), []).append(ctx)
    for ctxs in queues.values():
        ctxs.sort(key=lambda c: c.first_placeholder.order)
    qpos = {key: 0 for key in queues}
    active = []
    for key, ctxs in queues.items():
        if ctxs:
            active.append(ctxs[0])
            qpos[key] = 1
    while active:
        progress = False
        for ctx in active:
            if ctx.advance():
                progress = True
        nxt = []
        for ctx in active:
            if not ctx.finished:
                nxt.append(ctx)
                continue
            key = (ctx.die_id, ctx.first_placeholder.queue)
            ctxs = queues[key]
            if qpos[key] < len(ctxs):
                nxt.append(ctxs[qpos[key]])
                qpos[key] += 1
        active = nxt
        if active and not progress:
            active[0].skip_blocked()
    stats.residual_posts = reg_model.residual_post_count
    if stats.residual_posts:
        logger.warning("CCU post 记账残留 %s 项 (本地配对异常或仅被远端消费)",
                       stats.residual_posts)

    # 虚拟流分配 (7.4 节): 按 queue 维护计数器与复用池; 同 mission 内单元
    # 彼此并发不复用, 跨 mission 复用安全 (复用虚拟流的链内串行保证
    # 后一 mission 单元晚于前一 mission 单元)
    vstream_seq: dict = {}     # queue_key → 已创建 #v 序号
    vstream_free: dict = {}    # queue_key → 前一 mission 释放的可复用 key
    vstream_busy: dict = {}    # queue_key → 本 mission 占用中的 key
    prev_completion: dict = {}  # queue 流 id → 前一 mission 完成集

    for ctx in contexts:
        # 合并 loop 特征链消解 (6.5.1 节): 骨架/单元切分前删除折叠握手
        # 产生的伪串行屏障, 后续单元 fork 自然重接前一骨架节点
        stats.pipeline_chains_broken += _break_pipeline_chains(ctx, dep)
        # 骨架/异步单元切分 (7.4 节): 单元 = 数据任务 + 紧随的
        # NOTIFY_RECORD 链; 其余节点 (wt / 游离 set / 远端同步节点) 留在
        # 队列串行链。单元 fork 父为前一骨架节点; 骨架前的单元 (无骨架
        # 前驱) 的入口依赖由 SQE 流挂接边补 (graph_builder 消费 attach)
        nodes = ctx.nodes
        skeleton: list = []
        units: list = []       # (单元任务 id 列表, fork 父节点 id | None)
        last_skel = None
        i = 0
        while i < len(nodes):
            if dep.get_task(nodes[i]).task_type in _QUEUE_DATA_TYPES:
                head = i
                i += 1
                while (i < len(nodes)
                        and dep.get_task(nodes[i]).task_type
                        == TaskType.NOTIFY_RECORD):
                    i += 1
                units.append((nodes[head:i], last_skel))
            else:
                skeleton.append(nodes[i])
                last_skel = nodes[i]
                i += 1

        # 进入本 mission: 前一 mission 的虚拟流全部释放 (其单元链上先于
        # 本 mission 任意单元, 时间轴无重叠)
        qkey = f"ccu_d{ctx.die_id}_q{ctx.first_placeholder.queue}"
        free = vstream_free.setdefault(qkey, [])
        free.extend(vstream_busy.get(qkey, ()))
        vstream_busy[qkey] = []

        unit_tails: list = []
        for unit_ids, parent_id in units:
            vkey = free.pop(0) if free else None
            if vkey is None:
                vstream_seq[qkey] = vstream_seq.get(qkey, 0) + 1
                vkey = f"{qkey}#v{vstream_seq[qkey]}"
            vstream_busy[qkey].append(vkey)
            vsid = alloc_stream(vkey)
            for tid in unit_ids:
                dep.get_task(tid).stream_id = vsid
            if parent_id is not None:
                dep.add_edge(unit_ids[0], parent_id)      # fork
                stats.fork_edges += 1
            unit_tails.append(unit_ids[-1])
            result.virtual_chains.setdefault(vsid, []).extend(unit_ids)
            stats.parallel_units += 1

        # mission 完成集 = 骨架尾 + 全部单元尾 (原 comp_rec 汇合语义)
        completion = ([skeleton[-1]] if skeleton else []) + unit_tails
        head = skeleton[0] if skeleton else \
            (units[0][0][0] if units else None)
        entry_units = [ids[0] for ids, parent in units if parent is None]

        # 队列串行语义: 同 queue 后一 mission 的入口 (骨架头 + 骨架前单元
        # 头) 须等前一 mission 全部单元尾 (骨架尾已由队列链串行覆盖)
        prev_comp = prev_completion.get(ctx.stream_id, [])
        for target in ([head] if head else []) + entry_units:
            for p in prev_comp:
                dep.add_edge(target, p)
                stats.join_edges += 1

        # SQE 流挂接信息 (graph_builder 消费, 7.4.1 节)
        att = {"head": head, "entry_units": entry_units,
               "completion": completion,
               "queue_prev_completion": prev_comp}
        for ph in ctx.placeholders:
            result.marker_replacements[ph.marker] = []
            result.attach[ph.marker] = att

        if skeleton:
            result.queue_chains.setdefault(ctx.stream_id, []).extend(skeleton)
        prev_completion[ctx.stream_id] = completion

    stats.queues = len(result.queue_chains)
    stats.virtual_streams = len(result.virtual_chains)

    for ph in placeholders:
        if ph.marker not in result.marker_replacements:
            result.marker_replacements[ph.marker] = []
    return result
