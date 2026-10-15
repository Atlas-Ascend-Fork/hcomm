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

"""SQE 任务图生成模块 (设计文档第 5 章)。

将 OpExecutionGroup (单 rank 任务数据) 转换为 (SimConfig, DependencyGraph)。
产出任务类型与 hccl_sim 新版任务模型对齐 (docs/light_performance/HCCLSIM):
  NOTIFY_RECORD / NOTIFY_WAIT   — notify 同步 (src==dst 即本地)
  MEMCPY / REDUCE_INLINE        — HBM 拷贝 / 规约 (动态共享 HBM 带宽)
  UB_WRITE_OR_READ              — 网络数据传输 (端口带宽共享; SQE 走网任务
                                  write/read 两方向统一映射此类型, is_write
                                  区分端口计费方向与端点 rank)
  WRITE_WITH_NOTIFY             — 网络写 + 附带 notify 单任务 (与 db 的
                                  WRITE_WITH_NOTIFY 任务一一对应, 不拆分)
注: 上述 TaskType 为内部时延模型分类; trace/dump 显示名 (task_name) 由
  _add_task 统一取任务 profInfo.profTaskType 的 op 名 (11.4 节, 与 msprof
  timeline 一一对应), 二者经 11.4.1 节对照表映射。
成图流水线: ① SQE → Task 映射 (5.2 节) → ② stream 串行链 → ③ 本地 notify
按流内执行序配对 (5.4 节) → ④ 网络任务方向/eid/rank 补全 (5.5 节) → ⑤ CCU 子图
展开 (第 6 章) → ⑥ 输出组装。跨 rank post-wait 边全部忽略 (5.1 节设计前提):
跨 rank 的 notify 任务节点保留 (按 stream 顺序成图, src≠dst 走固定远端时延),
仅不建立其 record-wait 配对边。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from .ccu_graph_builder import CcuBuildStats, CcuPlaceholder, expand_ccu_graph
from .db_reader import (HccLTaskMetaType, JETTY_INVALID, RANK_INVALID,
                        OpExecutionGroup, RawTask, prof_link_type_name,
                        prof_task_type_name, prof_transport_type_name)
from .simulator.config import SimConfig
from .simulator.dependency_graph import DependencyGraph
from .simulator.task_model import INVALID, Task, TaskType

logger = logging.getLogger(__name__)

# CCU 队列合成流 id 基址 (6.5 节): SQE 流 id 直用原始 streamId,
# CCU queue 链从该基址顺序分配, 避免与 SQE 流冲突。
CCU_STREAM_BASE = 1 << 20

# CommProtocol 枚举 → protocol 字段显示名 (db_reader 4.4.4 节; 11.4.2 节:
# 通道建链协议, 与 DfxLinkType 维度的 link_type 为两个不同概念, 方案 a 迁移至此)
_PROTOCOL_NAMES = {0: "HCCS", 1: "TCP", 2: "ROCE", 3: "UB_CTP",
                   4: "UB_TP", 5: "PCIE", 6: "SIO"}

# HcclDataType (hccl_types.h) → 显示名 (v2 采集在 REDUCE 任务级填 dataType)
_DATA_TYPE_NAMES = {0: "int8", 1: "int16", 2: "int32", 3: "fp16",
                    4: "fp32", 5: "int64", 6: "uint64", 7: "uint8",
                    8: "uint16", 9: "uint32", 10: "fp64", 11: "bfp16"}


@dataclass
class GraphBuildStats:
    op_desc: str = ""
    total_raw: int = 0
    tasks: int = 0
    notify_records: int = 0
    notify_waits: int = 0
    ub_writes: int = 0
    ub_reads: int = 0
    write_with_notify: int = 0
    hbm_copies: int = 0
    hbm_reduces: int = 0
    skipped_sync: int = 0
    skipped_model_exec: int = 0
    cross_rank_notify: int = 0
    skipped_aiv: int = 0
    notify_matches: int = 0
    cyclic_matches: int = 0
    unmatched_records: int = 0
    unmatched_waits: int = 0
    direction_inferred: int = 0
    eid_unresolved: int = 0
    rank_unresolved: int = 0
    ccu: CcuBuildStats = field(default_factory=CcuBuildStats)

    def summary(self) -> str:
        return (f"{self.op_desc}: raw={self.total_raw} tasks={self.tasks} "
                f"(rec={self.notify_records} wait={self.notify_waits} "
                f"ubw={self.ub_writes} ubr={self.ub_reads} "
                f"wn={self.write_with_notify} "
                f"hbm={self.hbm_copies + self.hbm_reduces} "
                f"sync_skip={self.skipped_sync} "
                f"me_skip={self.skipped_model_exec}) matches={self.notify_matches} "
                f"cyclic_drop={self.cyclic_matches} "
                f"xrank_notify={self.cross_rank_notify} "
                f"aiv={self.skipped_aiv} "
                f"inferred_dir={self.direction_inferred} "
                f"eid_unresolved={self.eid_unresolved} "
                f"rank_unresolved={self.rank_unresolved} "
                f"ccu(m={self.ccu.missions}/q={self.ccu.queues}/"
                f"i={self.ccu.instructions}/"
                f"pu={self.ccu.parallel_units}/vs={self.ccu.virtual_streams}/"
                f"ml={self.ccu.merged_loops}/mi={self.ccu.merged_loop_iters}/"
                f"mf={self.ccu.merged_loop_fallbacks}/"
                f"pb={self.ccu.pipeline_chains_broken}/"
                f"lp={self.ccu.local_posts}/rp={self.ccu.remote_posts}/"
                f"wm={self.ccu.wait_matched}/rw={self.ccu.remote_waits}/"
                f"eu={self.ccu.eid_unresolved}/ru={self.ccu.rank_unresolved})")


class GraphBuilder:
    """单算子 (单 rank) SQE → DependencyGraph 构建器 (hccl_sim 新版任务模型)。"""

    def __init__(self, group: OpExecutionGroup,
                  sim_config: Optional[SimConfig] = None,
                  rank_map: dict = None):
        self.group = group
        self.config = sim_config
        # eid(hex) → rank 索引 (4.7 节): 对端 rank 解析 (SQE 回退 + CCU 主路径)
        self._rank_map = rank_map or {}
        self.stats = GraphBuildStats(
            op_desc=f"comm{group.comm_id}_iter{group.op_iter}"
                    f"_dev{group.device_id}")
        self.dep = DependencyGraph()
        # 本端 rank: opDetails.rankId 优先, 缺省按 deviceId-1 回退
        # (task.txt 路径的 deviceId-1=rankId 约定)
        self.local_rank = group.rank_id if group.rank_id is not None \
            else max(group.device_id - 1, 0)
        self._stream_tasks: dict[int, list] = {}
        self._stream_labels: dict[int, str] = {}
        self._ccu_streams: dict[str, int] = {}
        self._next_ccu_stream = CCU_STREAM_BASE
        self._notify_tasks: list = []
        self._placeholders: list = []
        self._jetty_channels = self._build_jetty_channels()
        self._local_eid = self._first_local_eid()

    # ------------------------------------------------------------------
    # 流 id 分配 (新版任务模型 stream_id 为 int, 6.5 节)
    # ------------------------------------------------------------------

    def _sqe_stream(self, stream_id: int) -> int:
        return self._alloc_stream(f"s{int(stream_id)}")

    def _alloc_stream(self, key: str) -> int:
        """按逻辑流名分配 (或复用) int 流 id。

        "s{n}" (SQE 流) 直用原始 streamId (小于基址时); CCU 队列链
        ("ccu_d{die}_q{queue}") 从基址顺序分配, 同名复用以合并多 mission。
        """
        sid = self._ccu_streams.get(key)
        if sid is not None:
            return sid
        if key.startswith("s") and key[1:].isdigit() \
                and int(key[1:]) < CCU_STREAM_BASE:
            sid = int(key[1:])
        else:
            sid = self._next_ccu_stream
            self._next_ccu_stream += 1
        self._ccu_streams[key] = sid
        self._stream_labels.setdefault(sid, key)
        return sid

    @property
    def stream_labels(self) -> dict:
        """int 流 id → 逻辑流名 (dump/报告显示用)。"""
        return self._stream_labels

    @property
    def stream_meta(self) -> dict:
        """int 流 id → 流元信息 (虚拟流标记, 7.4 节; 报告/trace/dump 用)。

        kind 分类 (按逻辑流名): "s{n}" → real (SQE 流); 含 "#" → virtual
        (CCU 队列并行化虚拟流, parent_sid 为其所属 queue 流 id); 其余 →
        ccu (queue 合成流)。
        """
        meta = {}
        for sid, label in self._stream_labels.items():
            if label.startswith("s") and label[1:].isdigit():
                meta[sid] = {"kind": "real", "label": label}
            elif "#" in label:
                meta[sid] = {"kind": "virtual", "label": label,
                             "parent_sid": self._ccu_streams.get(
                                 label.split("#", 1)[0])}
            else:
                meta[sid] = {"kind": "ccu", "label": label}
        return meta

    # ------------------------------------------------------------------
    # eid 解析 (5.5 节): jettyId → ccuChannels → 本端/对端 eid
    # ------------------------------------------------------------------

    def _build_jetty_channels(self) -> dict:
        """jettyId → 本设备出向通道 (用于 SQE 走网任务查本端 eid)。"""
        jetty_map = {}
        for (src_dev, _die, _chid), info in self.group.channels.items():
            if src_dev != self.group.device_id:
                continue
            for jid in info.jetty_ids:
                jetty_map.setdefault(jid, info)
        return jetty_map

    def _first_local_eid(self) -> str:
        for (src_dev, _die, _chid), info in self.group.channels.items():
            if src_dev == self.group.device_id and info.leid:
                return info.leid.hex()
        return ""

    def _resolve_eids(self, jetty_id: int, rm_eid: bytes,
                      lc_eid: bytes = b"") -> tuple:
        """返回 (本端 eid, 对端 eid, jetty_id, protocol)。

        本端优先取 SQE 显式 lcEid (v2 采集新增, 见 10.1 节); 为空时经 jettyId
        命中的出向通道 leid 推导; 对端优先取通道 reid, 退回 SQE rmEid;
        均无则为空串。protocol 为通道 CommProtocol 名 (11.4.2 节, 仅显示用);
        任务的 link_type/transport_type 由 _add_task 统一取 profInfo (11.4 节)。
        """
        if any(lc_eid or b""):
            local_eid = lc_eid.hex()
        else:
            local_eid = ""
        channel = self._jetty_channels.get(jetty_id)
        if channel is not None:
            if not local_eid:
                local_eid = channel.leid.hex() if channel.leid else ""
            peer_eid = channel.reid.hex() if channel.reid else ""
            link = _PROTOCOL_NAMES.get(channel.protocol, "")
        else:
            peer_eid = rm_eid.hex() if any(rm_eid or b"") else ""
            link = ""
        if not local_eid:
            local_eid = self._local_eid
        return local_eid, peer_eid, jetty_id, link

    # ------------------------------------------------------------------
    # ① SQE → Task 映射 (5.2 节)
    # ------------------------------------------------------------------

    def build(self) -> tuple:
        self._map_tasks()
        self._build_stream_chains()
        self._pair_notify()
        self._expand_ccu()
        self._drop_cyclic_matches()
        self.stats.tasks = self.dep.task_count
        if self.config is None:
            from .simulator.config import SimConfig as _SC
            self.config = _SC()
        return self.config, self.dep

    def _map_tasks(self) -> None:
        group = self.group
        self.stats.total_raw = len(group.tasks)
        for task in group.tasks:
            meta = task.meta
            data = meta.data
            if meta.task_type == HccLTaskMetaType.NOTIFY_RECORD:
                # estimate 采集约定: deviceId 字段承载 rank, 跨 rank 判定按 rank 维度
                if data.dst_rank_id != self.local_rank:
                    # 跨 rank post (5.1 节): 节点保留, 按 stream 顺序成图;
                    # 只是不建跨 rank 配对边 (src≠dst → 固定 remote_record_delay,
                    # simulator 不触发 Event)
                    self.stats.cross_rank_notify += 1
                    self._add_task(task, TaskType.NOTIFY_RECORD,
                                   notify_id=data.notify_id,
                                   src_rank=self.local_rank,
                                   dst_rank=self._remote_rank(data.dst_rank_id))
                else:
                    self._add_task(task, TaskType.NOTIFY_RECORD,
                                   notify_id=data.notify_id,
                                   src_rank=self.local_rank,
                                   dst_rank=self.local_rank)
            elif meta.task_type == HccLTaskMetaType.NOTIFY_WAIT:
                if data.src_rank_id != self.local_rank:
                    # 跨 rank wait (record 侧在对端, 不在本 db): 节点同样保留,
                    # 不参与本地配对 (src≠dst → 固定 remote_wait_delay)
                    self.stats.cross_rank_notify += 1
                    self._add_task(task, TaskType.NOTIFY_WAIT,
                                   notify_id=data.notify_id,
                                   src_rank=self._remote_rank(data.src_rank_id),
                                   dst_rank=self.local_rank)
                else:
                    self._add_task(task, TaskType.NOTIFY_WAIT,
                                   notify_id=data.notify_id,
                                   src_rank=self.local_rank,
                                   dst_rank=self.local_rank)
            elif meta.task_type in (HccLTaskMetaType.MEM_CPY,
                                    HccLTaskMetaType.REDUCE):
                self._map_transfer(task)
            elif meta.task_type == HccLTaskMetaType.WRITE_WITH_NOTIFY:
                self._map_write_with_notify(task)
            elif meta.task_type == HccLTaskMetaType.SYNC_STREAM:
                # host 侧流同步点 (等待本流完成), 不代表设备侧任务,
                # 跳过不生成节点 (5.6 节); makespan 即流链末任务结束时间
                self.stats.skipped_sync += 1
            elif meta.task_type == HccLTaskMetaType.MODEL_EXEC:
                # 模型重放触发任务 (rtModelExecute 边界标记, checker 按
                # execSeq 配对建边): 无设备侧负载与时延模型, 跳过不生成
                # 节点, 仅告警
                self.stats.skipped_model_exec += 1
                logger.warning("MODEL_EXEC 任务跳过 (模型重放边界标记, 无设备"
                               "侧负载): %s stream=%s role=%s execSeq=%s "
                               "peerStream=%s", self.stats.op_desc,
                               task.stream_id, data.role, data.exec_seq,
                               data.peer_stream_id)
            elif meta.task_type == HccLTaskMetaType.CCU_GRAPH:
                marker = f"ccu{task.global_id}"
                sid = self._sqe_stream(task.stream_id)
                self._stream_tasks.setdefault(sid, []).append(marker)
                self._placeholders.append(CcuPlaceholder(
                    marker=marker, die_id=data.die_id,
                    mission_id=data.mission_id,
                    inst_start_id=data.inst_start_id,
                    inst_cnt=data.inst_cnt, key=data.key,
                    arg_size=data.arg_size, args=data.args,
                    queue=task.stream_id, order=task.global_id,
                    lc_eid=meta.lc_eid))
            else:
                self.stats.skipped_aiv += 1
                logger.warning("AIV 任务暂不支持, 跳过: %s", self.stats.op_desc)

    def _map_transfer(self, raw: RawTask) -> None:
        data = raw.meta.data
        is_reduce = raw.meta.task_type == HccLTaskMetaType.REDUCE
        size = data.data_size if is_reduce else data.length
        if size < 0:
            size = 0
        data_type = _DATA_TYPE_NAMES.get(data.data_type, "") if is_reduce else ""
        if raw.meta.jetty_id == JETTY_INVALID or raw.meta.jetty_id is None:
            # 本地任务: rank 取 taskData 显式值 (deviceId 字段承载 rank, 实测为本端),
            # 未填退本端
            src_rank = self._local_task_rank(data.src_rank_id)
            dst_rank = self._local_task_rank(data.dst_rank_id)
            if is_reduce:
                self._add_task(raw, TaskType.REDUCE_INLINE, size=size,
                               data_type=data_type,
                               src_rank=src_rank, dst_rank=dst_rank)
                self.stats.hbm_reduces += 1
            else:
                self._add_task(raw, TaskType.MEMCPY, size=size,
                               src_rank=src_rank, dst_rank=dst_rank)
                self.stats.hbm_copies += 1
            return
        # 走网: 方向按 taskData rank 维度判定 (deviceId 字段承载 rank),
        # 退回区间推断 (5.5 节)
        direction, inferred = self._infer_direction(data)
        if inferred:
            self.stats.direction_inferred += 1
        local_eid, peer_eid, jetty_id, protocol = self._resolve_eids(
            raw.meta.jetty_id, raw.meta.rm_eid, raw.meta.lc_eid)
        if not local_eid:
            self.stats.eid_unresolved += 1
            logger.warning("SQE 走网任务本端 eid 无法解析 (jettyId=%s): %s",
                           raw.meta.jetty_id, self.stats.op_desc)
        if direction == "write":
            # 采集约定: write = src 本端 / dst 对端 (rank 未解析时经 eid 回退);
            # 内部类型与 read 分支统一为 UB_WRITE_OR_READ (is_write 区分计费
            # 方向), UB_INLINE_WRITE 仅保留 notify inline 写语义 (固定时延)
            self._add_task(raw, TaskType.UB_WRITE_OR_READ, size=size,
                           src_rank=self.local_rank,
                           dst_rank=self._peer_rank(data.dst_rank_id, peer_eid),
                           src_eid=local_eid, dst_eid=peer_eid,
                           jetty_id=jetty_id, protocol=protocol,
                           data_type=data_type, is_write=True)
            self.stats.ub_writes += 1
        else:
            # 采集约定: read = src 对端 / dst 本端
            self._add_task(raw, TaskType.UB_WRITE_OR_READ, size=size,
                           src_rank=self._peer_rank(data.src_rank_id, peer_eid),
                           dst_rank=self.local_rank,
                           src_eid=peer_eid, dst_eid=local_eid,
                           jetty_id=jetty_id, protocol=protocol,
                           data_type=data_type, is_write=False)
            self.stats.ub_reads += 1

    def _map_write_with_notify(self, raw: RawTask) -> None:
        """WRITE_WITH_NOTIFY (taskType=8) 单任务映射 (5.2 节)。

        本端 src → 远端 dst 网络写 + 写后附带 notify, 与 db 任务一一对应
        (不拆分为写 + NOTIFY_RECORD 两个子任务): 写部分恒为 write 方向
        (本端 → 对端), 走端口 write 方向带宽共享; notifyId 随任务保留
        (trace/dump 显示)。对端 wait 不在本 db (单 rank 仿真前提, 5.4 节),
        不参与本地 notify 配对; isReduce=1 时写载荷为规约数据, 仅影响
        data_type 显示。
        """
        data = raw.meta.data
        size = data.length if data.length > 0 else 0
        data_type = _DATA_TYPE_NAMES.get(data.data_type, "") \
            if data.is_reduce == 1 else ""
        local_eid, peer_eid, jetty_id, protocol = self._resolve_eids(
            raw.meta.jetty_id, raw.meta.rm_eid, raw.meta.lc_eid)
        if not local_eid:
            self.stats.eid_unresolved += 1
            logger.warning("WRITE_WITH_NOTIFY 本端 eid 无法解析 (jettyId=%s): %s",
                           raw.meta.jetty_id, self.stats.op_desc)
        self._add_task(raw, TaskType.WRITE_WITH_NOTIFY, size=size,
                       notify_id=data.notify_id,
                       src_rank=self.local_rank,
                       dst_rank=self._peer_rank(data.dst_rank_id, peer_eid),
                       src_eid=local_eid, dst_eid=peer_eid,
                       jetty_id=jetty_id, protocol=protocol,
                       data_type=data_type, is_write=True)
        self.stats.write_with_notify += 1

    def _infer_direction(self, data) -> tuple:
        """走网方向判定 (5.5 节)。

        estimate 采集约定: taskData 的 deviceId 字段承载 rank (原 srcRankId/
        dstRankId 字段已回退合并), 发射端约定 (device_sqe_parse_stub.cc):
        write = src 本端/dst 对端, read = src 对端/dst 本端。按 rank 判定,
        退回 opMemInfo 区间推断 (旧路径), 均无法判定时缺省 write。
        """
        if data.src_rank_id != data.dst_rank_id:
            if data.src_rank_id == self.local_rank:
                return "write", False
            if data.dst_rank_id == self.local_rank:
                return "read", False
        mem_info = self.group.mem_info
        if mem_info is not None:
            if mem_info.in_output(data.dst_offset):
                return "read", False
            if mem_info.in_input(data.src_offset):
                return "write", False
        return "write", True

    def _remote_rank(self, rank_id: int) -> int:
        """对端 rank: taskData 显式值有效时采信 (对端 rank 必不等于本端;
        发射端解析失败填 UINT32_MAX), 否则退 INVALID。"""
        if rank_id in (RANK_INVALID, self.local_rank):
            return INVALID
        return rank_id

    def _peer_rank(self, rank_id: int, peer_eid: str) -> int:
        """对端 rank (5.5 节): taskData 显式值 (deviceId 字段承载 rank,
        发射端已按 GetRmtRankIdByEid 解析, _remote_rank 语义) 优先;
        未解析时经 eid→rank 索引 (4.7 节) 回退, 仍无则 INVALID 并计入
        rank_unresolved。"""
        rank = self._remote_rank(rank_id)
        if rank != INVALID:
            return rank
        rank = self._rank_map.get(peer_eid, INVALID)
        if rank == INVALID:
            self.stats.rank_unresolved += 1
        return rank

    def _local_task_rank(self, rank_id: int) -> int:
        """本地任务 (不走网) 的端点 rank: 采信 taskData 显式值 (deviceId
        字段承载 rank), 未填 (UINT32_MAX) 时退本端 rank。"""
        if rank_id == RANK_INVALID:
            return self.local_rank
        return rank_id

    def _add_task(self, raw: RawTask, task_type: TaskType,
                  **kwargs) -> Task:
        sid = self._sqe_stream(raw.stream_id)
        # task_name/link_type/transport_type 统一取任务 profInfo (11.4 节, 与
        # msprof timeline op 名/L1 明细对账): task_name 为显示名覆盖, 内部
        # task_type 仍为时延模型分类; v2 旧版 BLOB 无 profInfo 时为 Unknown
        kwargs.setdefault("task_name", prof_task_type_name(raw.meta.prof_task_type))
        kwargs.setdefault("link_type", prof_link_type_name(raw.meta.link_type))
        kwargs.setdefault("transport_type",
                          prof_transport_type_name(raw.meta.transport_type))
        task = Task(f"t{raw.global_id}", task_type, stream_id=sid, **kwargs)
        self.dep.add_task(task)
        self._stream_tasks.setdefault(sid, []).append(task.task_id)
        if task_type == TaskType.NOTIFY_RECORD:
            self.stats.notify_records += 1
            # 仅本地 record 进配对池 (跨 rank 的 wait 侧在对端, 5.4 节)
            if task.notify_id != INVALID and task.is_local_notify:
                self._notify_tasks.append(("R", raw.global_id, task.task_id,
                                           task.notify_id))
        elif task_type == TaskType.NOTIFY_WAIT:
            self.stats.notify_waits += 1
            # 仅本地 wait 进配对池 (跨 rank 的 record 侧在对端, 5.4 节)
            if task.is_local_notify:
                self._notify_tasks.append(("W", raw.global_id, task.task_id,
                                           task.notify_id))
        return task

    # ------------------------------------------------------------------
    # ② stream 串行链 (5.3 节)
    # ------------------------------------------------------------------

    def _build_stream_chains(self) -> None:
        for sid, task_ids in self._stream_tasks.items():
            self.dep.add_stream_chain(sid, task_ids)

    # ------------------------------------------------------------------
    # ③ notify 配对 (5.4 节, 仅本地, 按流内执行序)
    # ------------------------------------------------------------------

    def _pair_notify(self) -> None:
        """按流内执行序配对 (5.4 节): 同一 notify 的第 i 个本地 WAIT 配
        第 i 个本地 RECORD。

        notify 为计数信号量 (record +1 / wait 等 >0 后 -1), 各流按链序
        执行时与静态 i↔i 配对等价; 而 db 提交序不等于执行序, 按提交序
        FCFS 会交叉配对、虚增 wait 耗时。多余 wait 按固定时延仿真
        (等远端), 多余 record 不配对 (对端 wait 不在本 db)。
        """
        pos: dict[str, int] = {}
        for task_ids in self._stream_tasks.values():
            for idx, tid in enumerate(task_ids):
                pos[tid] = idx
        waits: dict = {}
        records: dict = {}
        for kind, _, task_id, notify_id in self._notify_tasks:
            task = self.dep.get_task(task_id)
            entry = (task.stream_id, pos.get(task_id, 0), task_id)
            (waits if kind == "W" else records) \
                .setdefault(notify_id, []).append(entry)
        for notify_id in waits.keys() | records.keys():
            wait_list = sorted(waits.get(notify_id, ()))
            record_list = sorted(records.get(notify_id, ()))
            for (_ws, _wp, wait_id), (_rs, _rp, record_id) \
                    in zip(wait_list, record_list):
                self.dep.add_match(wait_id, record_id)
                self.stats.notify_matches += 1
            self.stats.unmatched_waits += max(0, len(wait_list) - len(record_list))
            self.stats.unmatched_records += max(0, len(record_list) - len(wait_list))
        if self.stats.unmatched_waits:
            logger.warning("本地 WAIT 未配对 %s 个 (等远端, 按固定时延仿真): %s",
                           self.stats.unmatched_waits, self.stats.op_desc)
        if self.stats.unmatched_records:
            logger.warning("本地 RECORD 未配对 %s 个: %s",
                           self.stats.unmatched_records, self.stats.op_desc)

    def _drop_cyclic_matches(self) -> None:
        """删除与串行链方向冲突的配对边, 保证依赖图无环 (5.4 节)。

        流内序配对正常不与 stream 链成环; 本检查保留作安全网 (多流
        交叉等待等场景仍可能出环): record 已传递依赖 wait 时删除该
        配对, wait 转 "未配对"语义按固定时延仿真; 删除边只会破环
        不会生环。
        """
        if self.dep.match_count == 0:
            return

        deps: dict[str, list] = {}
        for sid in self.dep.stream_ids:
            chain = self.dep.get_chain(sid)
            for prev, tid in zip(chain, chain[1:]):
                deps.setdefault(tid, []).append(prev)
        for wait_id in self.dep.matched_wait_ids:
            deps.setdefault(wait_id, []).append(self.dep.get_match(wait_id))
        for child_id, parents in self.dep.extra_edges.items():
            deps.setdefault(child_id, []).extend(parents)

        def reaches(src: str, target: str) -> bool:
            stack, seen = [src], set()
            while stack:
                cur = stack.pop()
                if cur == target:
                    return True
                if cur in seen:
                    continue
                seen.add(cur)
                stack.extend(deps.get(cur, ()))
            return False

        dropped = 0
        for wait_id in list(self.dep.matched_wait_ids):
            record_id = self.dep.get_match(wait_id)
            if reaches(record_id, wait_id):
                self.dep.remove_match(wait_id)
                edges = deps.get(wait_id)
                if edges and record_id in edges:
                    edges.remove(record_id)
                dropped += 1
        if dropped:
            self.stats.cyclic_matches = dropped
            self.stats.notify_matches = self.dep.match_count
            self.stats.unmatched_waits += dropped
            self.stats.unmatched_records += dropped
            logger.warning("配对边与 stream 链成环, 删除 %d 个配对 "
                           "(wait 按固定时延仿真): %s",
                           dropped, self.stats.op_desc)

    # ------------------------------------------------------------------
    # ⑤ CCU 子图展开 (第 6 章)
    # ------------------------------------------------------------------

    def _expand_ccu(self) -> None:
        if not self._placeholders:
            return
        result = expand_ccu_graph(
            self.group.device_id, self.group.dev_type,
            self._placeholders, self.group.instr_spaces,
            self.group.channels, self.dep,
            local_rank=self.local_rank, alloc_stream=self._alloc_stream,
            rank_map=self._rank_map)
        self.stats.ccu = result.stats
        # SQE 流链重建: CCU 占位移除 (任务图中不引入任何占位节点),
        # 子图经 extra 边挂接 (7.4.1 节) ——
        #   入口边: SQE 前驱任务 (及前续 CCU mission 完成集) → 骨架头
        #           与骨架前单元头 (mission 发射须等 SQE 前序完成)
        #   出口边: mission 完成集 (骨架尾 + 全部单元尾) → SQE 后继任务
        attached: set = set()  # 已挂接的 mission attach (同 mission 去重)
        for sid, task_ids in self._stream_tasks.items():
            resolved: list = []
            prev_real = None
            pending: list = []   # 已过 CCU 占位、待挂到下一 SQE 任务的完成集
            for tid in task_ids:
                if tid not in result.marker_replacements:
                    if pending:
                        for p in pending:
                            self.dep.add_edge(tid, p)      # 出口边
                            result.stats.join_edges += 1
                        pending = []
                    resolved.append(tid)
                    prev_real = tid
                    continue
                att = result.attach.get(tid)
                if att is None:
                    continue    # 空展开 mission (无指令空间), 无任务可挂
                if id(att) in attached:
                    # 同一 mission 的后续 SQE 占位 (指令区间拆批发射):
                    # 子图只挂接一次, 重复挂接会使 mission 头依赖自身完成集成环
                    continue
                attached.add(id(att))
                parents = ([prev_real] if prev_real else []) + list(pending)
                for target in ([att["head"]] if att["head"] else []) \
                        + list(att["entry_units"]):
                    for p in parents:
                        self.dep.add_edge(target, p)       # 入口边
                        result.stats.join_edges += 1
                pending.extend(att["completion"])
            # 覆盖写回 (空链移除, 避免残留含占位 marker 的旧链)
            if resolved:
                self.dep.add_stream_chain(sid, resolved)
            else:
                self.dep.remove_stream_chain(sid)
        for sid, chain in result.queue_chains.items():
            self.dep.add_stream_chain(sid, chain)
        for sid, chain in result.virtual_chains.items():
            self.dep.add_stream_chain(sid, chain)
        return


def build(group: OpExecutionGroup, sim_config: Optional[SimConfig] = None) -> tuple:
    """5.9 节输出组装: 返回 (SimConfig, DependencyGraph)。"""
    if sim_config is None:
        sim_config = SimConfig()
    builder = GraphBuilder(group, sim_config)
    return builder.build()
