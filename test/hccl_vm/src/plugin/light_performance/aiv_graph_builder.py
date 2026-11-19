#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.
# -----------------------------------------------------------------------------------------------------------

"""AIV 任务图生成模块 (设计文档第 12 章)。

移植自 checker 插件 C++ 实现 (aiv_snapshot_json_loader_v3 / aiv_task_transform_v3),
按 light_performance 单 rank 成图约束 (1.3/5.1/5.4 节) 裁剪:

  - 输入: SQE 流中的 AIV_GRAPH 占位 (taskType=5, taskData=launchIdx) +
    采集侧任务快照 JSON (data/hcclvm_aiv_device{dev}_launch{idx}_task.json)。
  - 每 launch 展开为并行泳道子图: 每 (block, pipe) 一条虚拟流串行链
    (对应 checker queueId = launch*3072 + block*3 + pipe 的泳道语义);
    pipe 0=SCALAR / 1=MTE2 / 2=MTE3。
  - 任务映射 (对齐 checker TranslateAivRuntimeTask, 时延复用现有模型):
      MEM_COPY   → Memcpy        (size=payload 字节数, HBM 带宽模型)
      REDUCE     → Reduce_Inline (size, data_type; HBM ×3 RMW)
      SET_FLAG   → Notify_Record (local, 与 WAIT_FLAG 按事件键 FIFO 配对)
      WAIT_FLAG  → Notify_Wait   (local, add_match 显式配对)
      SEND_FLAG  → Notify_Record (targetRank≠本端 → remote 固定时延;
                                   跨 rank 配对边丢弃 —— 对端节点不在本 db,
                                   5.1/5.4 节单 rank 前提)
      RECV_FLAG  → Notify_Wait   (targetRank≠本端 → remote 固定时延, 不配对)
      PIPE_BARRIER → 并查集按 barrierGroupTaskIds 合并为单 join 节点
                     (成员从各 pipe 链移除; join 等全部成员前驱, 成员后继等 join)
      SYNC_ALL   → 按 syncRound 跨 core 合并为单 join 节点 (同上)
  - 与 checker 的差异 (性能仿真裁剪):
      * 不建 TaskStart/TaskEnd 边界节点 (CCU 展开同一先例, 占位不产节点);
      * 不做 CpGM2GM 双缓冲循环合并 (checker 的图规模优化, 仿真侧非必需);
      * 跨 rank SEND/RECV_FLAG 配对 (checker MatchSendRecvFlagEdges) 不移植:
        单 rank 成图丢弃对端节点, 远端 flag 按固定时延;
      * estimate 快照的 slice.deviceId 为回环值 (12.4/12.5 节), 跨卡判定
        一律用 targetRank, 不用 slice.deviceId。
  - 挂接 (对齐 checker ReconnectPlaceholder + 7.4.1 节 extra 边):
      入口边: SQE 前驱 → 各泳道首任务 (及无前驱的 join);
      出口边: 完成集 (子图内无后继的节点) → SQE 后继。
"""

from __future__ import annotations

import glob
import json
import logging
import os
from dataclasses import dataclass, field

from .db_reader import _DATA_TYPE_NAMES
from .simulator.dependency_graph import DependencyGraph
from .simulator.task_model import Task, TaskType

logger = logging.getLogger(__name__)

# 快照文件命名 (采集侧 aiv_task_json.cc MakeTaskFileName 同一口径)
AIV_TASK_FILE_PREFIX = "hcclvm_aiv_device"
AIV_TASK_FILE_LAUNCH_MARKER = "_launch"
AIV_TASK_FILE_SUFFIX = "_task.json"

# AivRuntimeTaskType (aiv_task.h AivTaskType, 与 checker AivRuntimeTaskTypeV3 一致)
AIV_MEM_COPY = 0
AIV_REDUCE = 1
AIV_SET_FLAG = 2
AIV_WAIT_FLAG = 3
AIV_PIPE_BARRIER = 4
AIV_SYNC_ALL = 5
AIV_SEND_FLAG = 6
AIV_RECV_FLAG = 7

_AIV_TASK_NAMES = {
    AIV_MEM_COPY: "AivMemCopy",
    AIV_REDUCE: "AivReduce",
    AIV_SET_FLAG: "AivSetFlag",
    AIV_WAIT_FLAG: "AivWaitFlag",
    AIV_PIPE_BARRIER: "AivPipeBarrier",
    AIV_SYNC_ALL: "AivSyncAll",
    AIV_SEND_FLAG: "AivSendFlag",
    AIV_RECV_FLAG: "AivRecvFlag",
}

# pipe (curPipe) → 快照数组字段
_PIPE_FIELDS = {0: "scalarTasks", 1: "mte2Tasks", 2: "mte3Tasks"}


@dataclass
class AivPlaceholder:
    """SQE 流链中的 AIV launch 占位 (taskType=5 行)。"""

    marker: str
    device_id: int
    launch_idx: int
    order: int
    origin_sid: int


@dataclass
class AivBuildStats:
    launches: int = 0
    missing_snapshots: int = 0
    tasks: int = 0            # 发射的数据/flag 节点 (MEM_COPY/REDUCE/SET/WAIT/SEND/RECV)
    lanes: int = 0            # (block, pipe) 泳道数
    barrier_joins: int = 0
    syncall_joins: int = 0
    set_wait_matches: int = 0
    unmatched_waits: int = 0
    remote_flags: int = 0     # targetRank≠本端 的 SEND/RECV_FLAG (固定远端时延)
    local_flags: int = 0
    entry_edges: int = 0
    exit_edges: int = 0

    def summary(self) -> str:
        return (f"aiv(l={self.launches}/t={self.tasks}/lane={self.lanes}/"
                f"pb={self.barrier_joins}/sa={self.syncall_joins}/"
                f"sw={self.set_wait_matches}/uw={self.unmatched_waits}/"
                f"rf={self.remote_flags}/lf={self.local_flags}/"
                f"miss={self.missing_snapshots})")


@dataclass
class AivExpansionResult:
    attach: dict = field(default_factory=dict)        # marker → {"heads","completion"}
    lane_chains: dict = field(default_factory=dict)   # 虚拟流 id → 任务链
    stats: AivBuildStats = field(default_factory=AivBuildStats)


# ---------------------------------------------------------------------------
# 快照定位与加载 (移植 aiv_snapshot_json_loader_v3)
# ---------------------------------------------------------------------------

def _snapshot_filename(device_id: int, launch_idx: int) -> str:
    return (f"{AIV_TASK_FILE_PREFIX}{device_id}"
            f"{AIV_TASK_FILE_LAUNCH_MARKER}{launch_idx}{AIV_TASK_FILE_SUFFIX}")


def find_aiv_snapshot(data_dirs: list, device_id: int,
                      launch_idx: int) -> str:
    """在 data_dirs (及其一级时间戳子目录) 中定位快照文件。

    采集侧运行期写 <install>/data/ 根, runner 归档后移至 data/<timestamp>/;
    两处都查, 多命中取 mtime 最新 (与当前 db 同轮次)。
    """
    name = _snapshot_filename(device_id, launch_idx)
    candidates = []
    for base in data_dirs or ():
        if not base:
            continue
        direct = os.path.join(base, name)
        if os.path.isfile(direct):
            candidates.append(direct)
        for sub in glob.glob(os.path.join(base, "*")):
            if os.path.isdir(sub):
                nested = os.path.join(sub, name)
                if os.path.isfile(nested):
                    candidates.append(nested)
    if not candidates:
        return ""
    return max(candidates, key=os.path.getmtime)


def load_aiv_snapshot(data_dirs: list, device_id: int, launch_idx: int):
    """加载并校验快照 (checker LoadByDeviceAndLaunch 同口径)。

    校验: 可解析、aivCores 为 array、deviceId/launchIndex 与请求一致。
    失败返回 None (调用方告警并跳过该 launch)。
    """
    path = find_aiv_snapshot(data_dirs, device_id, launch_idx)
    if not path:
        logger.warning("AIV snapshot not found: %s (searched %s)",
                       _snapshot_filename(device_id, launch_idx), data_dirs)
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            snap = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("AIV snapshot unreadable: %s (%s)", path, exc)
        return None
    if not isinstance(snap, dict) or not isinstance(snap.get("aivCores"), list):
        logger.warning("AIV snapshot malformed (no aivCores array): %s", path)
        return None
    if snap.get("deviceId") != device_id or snap.get("launchIndex") != launch_idx:
        logger.warning("AIV snapshot identity mismatch: %s (deviceId=%s "
                       "launchIndex=%s, expected %s/%s)", path,
                       snap.get("deviceId"), snap.get("launchIndex"),
                       device_id, launch_idx)
        return None
    snap["_path"] = path
    return snap


# ---------------------------------------------------------------------------
# 单 launch 子图构建 (移植 aiv_task_transform_v3 ExpandOneAivGraph)
# ---------------------------------------------------------------------------

class _LaunchBuilder:
    """一个 AIV launch 的子图构建器。

    节点 id 方案 (图内唯一):
      运行时任务  a{launch}_t{taskId}
      barrier 组  a{launch}_pb{组内最小 taskId}
      syncAll 轮  a{launch}_sa{syncRound}
    泳道流标签 "s{origin}#a{launch}b{block}p{pipe}", join 流标签
    "s{origin}#a{launch}j{n}" —— 含 "#" 被 stream_meta 归类 virtual,
    parent 解析回占位所在 SQE 原流 (graph_builder._alloc_stream 约定)。
    """

    def __init__(self, snap: dict, ph: AivPlaceholder, dep: DependencyGraph,
                 local_rank: int, alloc_stream, stats: AivBuildStats):
        self.snap = snap
        self.ph = ph
        self.dep = dep
        self.local_rank = local_rank
        self.alloc_stream = alloc_stream
        self.stats = stats
        self.launch = ph.launch_idx
        self.lane_chains: dict = {}       # sid → [task_id]
        self.heads: list = []
        self.entry_joins: list = []       # 无成员前驱的 join (挂接时补 SQE 前驱边)
        self._tasks_by_id: dict = {}      # JSON taskId → 解析上下文
        self._emitted: dict = {}          # JSON taskId → 发射节点 task_id
        self._lane_of: dict = {}          # JSON taskId → (block, pipe)
        self._lane_order: dict = {}       # (block, pipe) → [JSON taskId 程序序]
        self._successors: dict = {}       # 节点 task_id → set(后继 task_id)
        self._join_seq = 0

    # -- 解析快照 -----------------------------------------------------------

    def _index_snapshot(self) -> None:
        for core in self.snap["aivCores"]:
            block = core.get("blockIdx", 0)
            for pipe, field_name in _PIPE_FIELDS.items():
                order = []
                for raw in core.get(field_name) or ():
                    tid = raw.get("taskId")
                    if tid is None:
                        continue
                    self._tasks_by_id[tid] = raw
                    self._lane_of[tid] = (block, pipe)
                    order.append(tid)
                if order:
                    self._lane_order[(block, pipe)] = order

    # -- 节点发射 -----------------------------------------------------------

    def _add_node(self, node_id: str, task_type: TaskType, stream_id: int,
                  task_name: str, **kwargs) -> Task:
        task = Task(node_id, task_type, stream_id=stream_id,
                    task_name=task_name, **kwargs)
        self.dep.add_task(task)
        self.stats.tasks += 1
        return task

    def _emit_runtime_task(self, tid: int, raw: dict, block: int, pipe: int,
                           sid: int) -> str:
        """发射一个运行时任务节点, 返回节点 id; 不可发射类型返回 ""。"""
        ttype = raw.get("taskType")
        payload = raw.get("payload") or {}
        node_id = f"a{self.launch}_t{tid}"
        if ttype == AIV_MEM_COPY:
            src = payload.get("src") or {}
            dst = payload.get("dst") or {}
            size = int(src.get("size") or dst.get("size") or 0)
            self._add_node(node_id, TaskType.MEMCPY, sid,
                           _AIV_TASK_NAMES[AIV_MEM_COPY],
                           size=size,
                           src_rank=self.local_rank, dst_rank=self.local_rank,
                           src_buf_type=src.get("bufferTypeName", ""),
                           dst_buf_type=dst.get("bufferTypeName", ""))
        elif ttype == AIV_REDUCE:
            src = payload.get("src") or {}
            dst = payload.get("dst") or {}
            size = int(src.get("size") or dst.get("size") or 0)
            data_type = _DATA_TYPE_NAMES.get(payload.get("dataType", -1), "")
            self._add_node(node_id, TaskType.REDUCE_INLINE, sid,
                           _AIV_TASK_NAMES[AIV_REDUCE],
                           size=size, data_type=data_type,
                           src_rank=self.local_rank, dst_rank=self.local_rank,
                           src_buf_type=src.get("bufferTypeName", ""),
                           dst_buf_type=dst.get("bufferTypeName", ""))
        elif ttype in (AIV_SET_FLAG, AIV_WAIT_FLAG):
            # pipe 内事件同步: local notify 节点, SET/WAIT 由 _match_set_wait
            # 按 (block, srcPipe, dstPipe, eventId) FIFO 显式配对
            tt = TaskType.NOTIFY_RECORD if ttype == AIV_SET_FLAG \
                else TaskType.NOTIFY_WAIT
            self._add_node(node_id, tt, sid, _AIV_TASK_NAMES[ttype],
                           notify_id=int(payload.get("eventId", 0)),
                           src_rank=self.local_rank, dst_rank=self.local_rank)
        elif ttype in (AIV_SEND_FLAG, AIV_RECV_FLAG):
            # 跨 rank flag: targetRank≠本端 → remote 固定时延, 不配对
            # (对端节点不在本 db, 单 rank 前提 5.1/5.4 节; estimate 快照
            # slice.deviceId 为回环值, 跨卡判定一律用 targetRank, 12.5 节)
            target_rank = payload.get("targetRank", self.local_rank)
            flag_buf = payload.get("flagBuffer") or {}
            notify_id = int(flag_buf.get("offset", 0))
            remote = target_rank != self.local_rank
            if remote:
                self.stats.remote_flags += 1
            else:
                self.stats.local_flags += 1
            if ttype == AIV_SEND_FLAG:
                self._add_node(
                    node_id, TaskType.NOTIFY_RECORD, sid,
                    _AIV_TASK_NAMES[AIV_SEND_FLAG], notify_id=notify_id,
                    src_rank=self.local_rank,
                    dst_rank=target_rank if remote else self.local_rank)
            else:
                self._add_node(
                    node_id, TaskType.NOTIFY_WAIT, sid,
                    _AIV_TASK_NAMES[AIV_RECV_FLAG], notify_id=notify_id,
                    src_rank=target_rank if remote else self.local_rank,
                    dst_rank=self.local_rank)
        else:
            logger.warning("AIV launch %s: unsupported runtime taskType=%s "
                           "(taskId=%s), node dropped", self.launch, ttype, tid)
            return ""
        self._emitted[tid] = node_id
        return node_id

    # -- barrier / syncAll 分组合并 (移植 MergePipeBarrierGroups/MergeSyncAllGroups)

    def _barrier_groups(self) -> list:
        """并查集按 barrierGroupTaskIds 聚合 PIPE_BARRIER 成员。"""
        members = {tid: raw for tid, raw in self._tasks_by_id.items()
                   if raw.get("taskType") == AIV_PIPE_BARRIER}
        parent = {tid: tid for tid in members}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for tid, raw in members.items():
            for other in (raw.get("payload") or {}).get("barrierGroupTaskIds") or ():
                if other in members:
                    union(tid, other)
        groups: dict = {}
        for tid in members:
            groups.setdefault(find(tid), []).append(tid)
        result = []
        for root, group in groups.items():
            group.sort()
            blocks = {self._lane_of[t][0] for t in group if t in self._lane_of}
            pipes = [self._lane_of[t][1] for t in group if t in self._lane_of]
            if len(blocks) > 1:
                logger.warning("AIV launch %s: barrier group %s spans blocks "
                               "%s (checker 要求同 block), 按同组处理",
                               self.launch, group, sorted(blocks))
            if len(set(pipes)) != len(pipes):
                logger.warning("AIV launch %s: barrier group %s has duplicate "
                               "pipes %s", self.launch, group, pipes)
            result.append(group)
        result.sort(key=lambda g: min(g))
        return result

    def _syncall_groups(self) -> list:
        """按 syncRound 跨 core 聚合 SYNC_ALL 成员。"""
        rounds: dict = {}
        for tid, raw in self._tasks_by_id.items():
            if raw.get("taskType") != AIV_SYNC_ALL:
                continue
            rnd = (raw.get("payload") or {}).get("syncRound", 0)
            rounds.setdefault(rnd, []).append(tid)
        expected = len(self._lane_order)  # 每 (block, pipe) 恰一员
        result = []
        for rnd in sorted(rounds):
            group = sorted(rounds[rnd])
            if expected and len(group) != expected:
                logger.warning("AIV launch %s: syncAll round %s has %d members, "
                               "expected %d (blocks×pipes)", self.launch, rnd,
                               len(group), expected)
            result.append((rnd, group))
        return result

    # -- 主流程 --------------------------------------------------------------

    def build(self) -> None:
        self._index_snapshot()
        origin = self.ph.origin_sid
        barrier_members: set = set()
        barrier_group_of: dict = {}
        groups = self._barrier_groups()
        for group in groups:
            for tid in group:
                barrier_members.add(tid)
                barrier_group_of[tid] = group
        syncall_members: set = set()
        syncall_round_of: dict = {}
        for rnd, group in self._syncall_groups():
            for tid in group:
                syncall_members.add(tid)
                syncall_round_of[tid] = rnd

        # ① 泳道链: 每 (block, pipe) 一条虚拟流, 成员 barrier/syncAll 不入链
        #    (合并为 join 节点, checker RewireGroupMember 同一语义);
        #    记录每个 join 成员的链上前驱/后继用于挂边。
        join_pred: dict = {}    # join_key → [前驱节点 id]
        join_succ: dict = {}    # join_key → [后继节点 id]

        def join_key_of(tid: int):
            if tid in barrier_group_of:
                return ("pb", min(barrier_group_of[tid]))
            if tid in syncall_round_of:
                return ("sa", syncall_round_of[tid])
            return None

        for (block, pipe), order in sorted(self._lane_order.items()):
            label = f"s{origin}#a{self.launch}b{block}p{pipe}"
            sid = self.alloc_stream(label)
            chain: list = []
            prev_node = None
            pending_joins: list = []   # 最近一个 join 成员之后待挂 join 的后继
            for tid in order:
                raw = self._tasks_by_id[tid]
                jk = join_key_of(tid)
                if jk is not None:
                    # join 成员: 记录前驱 (当前链尾); 其后继在下一节点处补
                    if prev_node is not None:
                        join_pred.setdefault(jk, [])
                        if prev_node not in join_pred[jk]:
                            join_pred[jk].append(prev_node)
                    pending_joins.append(jk)
                    continue
                node_id = self._emit_runtime_task(tid, raw, block, pipe, sid)
                if not node_id:
                    continue
                for pjk in pending_joins:
                    join_succ.setdefault(pjk, [])
                    if node_id not in join_succ[pjk]:
                        join_succ[pjk].append(node_id)
                pending_joins = []
                chain.append(node_id)
                prev_node = node_id
            if chain:
                self.lane_chains[sid] = chain
                self.stats.lanes += 1
                self.heads.append(chain[0])

        # ② join 节点: 每 barrier 组 / syncAll 轮一个 (自身独占虚拟流,
        #    避免不同 barrier 串行化); NOTIFY_RECORD local 未配对 = 30ns 固定
        #    同步开销; 边: join ← 各成员前驱; 成员后继 ← join。
        for jk in sorted(set(list(join_pred) + list(join_succ))):
            kind, key = jk
            if kind == "pb":
                node_id = f"a{self.launch}_pb{key}"
                name = _AIV_TASK_NAMES[AIV_PIPE_BARRIER]
                self.stats.barrier_joins += 1
            else:
                node_id = f"a{self.launch}_sa{key}"
                name = _AIV_TASK_NAMES[AIV_SYNC_ALL]
                self.stats.syncall_joins += 1
            self._join_seq += 1
            sid = self.alloc_stream(f"s{origin}#a{self.launch}j{self._join_seq}")
            self._add_node(node_id, TaskType.NOTIFY_RECORD, sid, name,
                           src_rank=self.local_rank, dst_rank=self.local_rank)
            self.lane_chains[sid] = [node_id]
            parents = join_pred.get(jk, [])
            for p in parents:
                self.dep.add_edge(node_id, p)          # join 等各 pipe 前驱
                self._successors.setdefault(p, set()).add(node_id)
            for s in join_succ.get(jk, ()):
                self.dep.add_edge(s, node_id)          # 各 pipe 后继等 join
                self._successors.setdefault(node_id, set()).add(s)
            if not parents:
                # 成员均位于链首: join 由 SQE 前驱驱动 (挂接阶段补边)
                self.entry_joins.append(node_id)
                self.heads.append(node_id)

        # ③ SET/WAIT 事件配对: 键 (block, srcPipe, dstPipe, eventId),
        #    程序序 FIFO (checker Kahn 模拟的等价简化, 良构 kernel 下一致)
        set_keys: dict = {}
        wait_keys: dict = {}
        for (block, pipe), order in sorted(self._lane_order.items()):
            for tid in order:
                raw = self._tasks_by_id[tid]
                ttype = raw.get("taskType")
                if ttype not in (AIV_SET_FLAG, AIV_WAIT_FLAG):
                    continue
                if tid not in self._emitted:
                    continue
                payload = raw.get("payload") or {}
                key = (block, payload.get("srcPipe"), payload.get("dstPipe"),
                       payload.get("eventId"))
                bucket = set_keys if ttype == AIV_SET_FLAG else wait_keys
                bucket.setdefault(key, []).append(self._emitted[tid])
        for key, waits in wait_keys.items():
            sets = set_keys.get(key, [])
            for wait_id, set_id in zip(waits, sets):
                self.dep.add_match(wait_id, set_id)
                self.stats.set_wait_matches += 1
                self._successors.setdefault(set_id, set()).add(wait_id)
            if len(waits) > len(sets):
                self.stats.unmatched_waits += len(waits) - len(sets)
                logger.warning("AIV launch %s: %d WAIT_FLAG unmatched for key "
                               "%s (fixed-delay simulation)", self.launch,
                               len(waits) - len(sets), key)
        for key, sets in set_keys.items():
            paired = len(wait_keys.get(key, ()))
            if len(sets) > paired:
                logger.info("AIV launch %s: %d SET_FLAG unconsumed for key %s",
                            self.launch, len(sets) - paired, key)

        # ④ 泳道链内串行后继 (完成集计算用)
        for chain in self.lane_chains.values():
            for prev, nxt in zip(chain, chain[1:]):
                self._successors.setdefault(prev, set()).add(nxt)

    def completion_set(self) -> list:
        """子图完成集: 全部发射节点中无内部后继者 (checker
        ConnectCurrentTailsToEnd 的 tails 语义)。"""
        all_nodes = [t for chain in self.lane_chains.values() for t in chain]
        return [n for n in all_nodes if not self._successors.get(n)]


# ---------------------------------------------------------------------------
# 展开入口 (移植 ExpandAivGraphsV3, 单 rank 裁剪版)
# ---------------------------------------------------------------------------

def expand_aiv_graph(placeholders: list, dep: DependencyGraph,
                     local_rank: int, alloc_stream,
                     data_dirs: list) -> AivExpansionResult:
    """展开全部 AIV 占位。返回 attach/lane_chains/stats, SQE 流链重建由
    graph_builder._expand_aiv 完成 (与 _expand_ccu 同一挂接模式)。"""
    result = AivExpansionResult()
    if not placeholders:
        return result
    cache: dict = {}
    for ph in sorted(placeholders, key=lambda p: p.order):
        result.stats.launches += 1
        key = (ph.device_id, ph.launch_idx)
        if key not in cache:
            cache[key] = load_aiv_snapshot(data_dirs, ph.device_id, ph.launch_idx)
        snap = cache[key]
        if snap is None:
            result.stats.missing_snapshots += 1
            result.attach[ph.marker] = None
            continue
        builder = _LaunchBuilder(snap, ph, dep, local_rank, alloc_stream,
                                 result.stats)
        builder.build()
        result.lane_chains.update(builder.lane_chains)
        result.attach[ph.marker] = {
            "heads": list(builder.heads),
            "entry_joins": list(builder.entry_joins),
            "completion": builder.completion_set(),
        }
    return result
