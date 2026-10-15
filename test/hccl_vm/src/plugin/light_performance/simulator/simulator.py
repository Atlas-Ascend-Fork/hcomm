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

"""仿真模块: 带宽共享管理器 + HcclSimulator 主仿真器。

带宽共享管理器 (_BandwidthManager):
  - 通用的动态带宽共享管理, 用于端口传输和 HBM 访问。
  - 单管理进程模型: 新任务加入时 interrupt, 管理进程重算均分带宽。
  - 端口传输: write 和 read 方向各自独立管理。
  - HBM: Reduce_Inline 和 Memcpy 共享 HBM 带宽。

任务时延模型 (详见 task_timing.py):
  Notify_Record (local):  固定时延 + 触发 Event
  Notify_Record (remote): 固定时延
  Notify_Wait (local):    等待 Event
  Notify_Wait (remote):   固定时延
  Reduce_Inline / Memcpy: HBM 带宽共享 (动态)
  Ub_Inline_Write / Ub_Write_Or_Read: 端口带宽共享 (动态)
  Write_With_Notify:      端口 write 方向带宽共享 (动态) + 写落地后
                          notify 固定 record 时延 (对端 wait 不在本 db)

依赖边 (dependency_graph):
  stream 链内串行 + notify 配对 (record→wait Event) + extra 边
  (CCU 队列并行化 fork/join, 7.4 节): extra 边父任务完成时触发完成事件,
  子任务等待全部父事件后才置 start_time 并执行。
"""

from __future__ import annotations

import logging

import simpy

from .config import SimConfig
from .task_model import Task, TaskType, NOTIFY_TYPES, HBM_TYPES, TRANSFER_TYPES, INVALID
from .dependency_graph import DependencyGraph
from .bandwidth_profile import BandwidthProfileSet
from .topo_resolver import TopoResolver
from . import task_timing

logger = logging.getLogger(__name__)


class _Transfer:
    __slots__ = ("tid", "data_size", "weight", "done",
                 "transferred", "slice_start", "slice_bw")

    def __init__(self, tid: str, data_size: float, done: simpy.Event,
                 weight: float = 1.0):
        self.tid = tid
        self.data_size = data_size
        self.weight = weight
        self.done = done
        self.transferred = 0.0
        self.slice_start = 0.0
        self.slice_bw = 0.0


class _BandwidthManager:
    """通用带宽共享管理器 (单管理进程模型)。

    用于:
      - 端口的 write/read 方向 (带带宽利用率, 均分模式)
      - HBM 带宽 (利用率固定为 1.0, 按权重比例分配模式)

    均分模式 (proportional=False): 有效带宽 = (总带宽 / 并发数) × 利用率
    权重模式 (proportional=True): 有效带宽 = 总带宽 × (weight / sum_weights) × 利用率
      HBM 控制器按请求仲裁, Memcpy 产生 2 路 HBM 请求 (1读+1写),
      Reduce_Inline 产生 3 路 (2读+1写, RMW), 带宽按 2:3 固定比例分配。
      小任务先完成, 大任务随后独占全部带宽。
    """

    __slots__ = ("env", "bandwidth", "bw_profile_set", "proportional",
                 "_active", "_peak_concurrency", "_manager_proc")

    def __init__(self, env: simpy.Environment, bandwidth: float,
                 bw_profile_set: BandwidthProfileSet,
                 proportional: bool = False):
        self.env = env
        self.bandwidth = bandwidth
        self.bw_profile_set = bw_profile_set
        self.proportional = proportional
        self._active: dict[str, _Transfer] = {}
        self._peak_concurrency = 0
        self._manager_proc = None

    def send(self, tid: str, data_size: float,
             weight: float = 1.0) -> simpy.Event:
        done = self.env.event()
        transfer = _Transfer(tid, data_size, done, weight=weight)
        self._active[tid] = transfer
        n = len(self._active)
        if n > self._peak_concurrency:
            self._peak_concurrency = n

        mp = self._manager_proc
        if mp is None or not mp.is_alive:
            self._manager_proc = self.env.process(self._manage())
        else:
            try:
                mp.interrupt()
            except RuntimeError:
                pass
        return done

    def _manage(self):
        active = self._active
        bandwidth = self.bandwidth
        env = self.env
        bps = self.bw_profile_set

        while active:
            n = len(active)
            now = env.now
            min_time = float("inf")
            finished = None

            # 找出当前活跃任务中最小的剩余数据量, 作为利用率查询基准
            # 同一组并发的任务用相同基准查利用率
            min_remaining = float("inf")
            for t in active.values():
                remaining = t.data_size - t.transferred
                if remaining <= 1.0:
                    if finished is None:
                        finished = []
                    finished.append(t)
                elif remaining < min_remaining:
                    min_remaining = remaining

            if finished:
                for t in finished:
                    active.pop(t.tid, None)
                    if not t.done.triggered:
                        t.done.succeed(env.now)
                continue

            # 剩余数据量 < 8MB 时用原始数据量查利用率, 不再按 min_remaining 拆分
            if min_remaining >= 8388608:
                lookup_size = min_remaining
            else:
                lookup_size = min(t.data_size for t in active.values())
            util = bps.get_utilization(n, lookup_size)

            if self.proportional:
                total_weight = sum(t.weight for t in active.values())
            else:
                total_weight = None

            for t in active.values():
                t.slice_start = now
                remaining = t.data_size - t.transferred
                if self.proportional and total_weight > 0:
                    effective_bw = bandwidth * (t.weight / total_weight) * util
                else:
                    effective_bw = (bandwidth / n) * util
                t.slice_bw = effective_bw
                t_time = remaining / effective_bw if effective_bw > 0 else float("inf")
                if t_time < min_time:
                    min_time = t_time

            try:
                yield env.timeout(min_time)
                elapsed = env.now - now
                finished = None
                for t in active.values():
                    t.transferred += elapsed * t.slice_bw
                    if t.data_size - t.transferred <= 1.0:
                        if finished is None:
                            finished = []
                        finished.append(t)
                if finished:
                    for t in finished:
                        active.pop(t.tid, None)
                        if not t.done.triggered:
                            t.done.succeed(env.now)
            except simpy.Interrupt:
                elapsed = env.now - now
                for t in active.values():
                    t.transferred += elapsed * t.slice_bw

    @property
    def peak_concurrency(self) -> int:
        return self._peak_concurrency


class NetworkPort:
    """逻辑端口, 含 write 和 read 两个独立方向的带宽管理。"""

    __slots__ = ("port_ids", "_write", "_read", "_total_bandwidth")

    def __init__(self, env: simpy.Environment, port_ids: list[str],
                 total_bandwidth: float, bw_profile_set: BandwidthProfileSet):
        self.port_ids = port_ids
        self._total_bandwidth = total_bandwidth
        self._write = _BandwidthManager(env, total_bandwidth, bw_profile_set)
        self._read = _BandwidthManager(env, total_bandwidth, bw_profile_set)

    def send(self, tid: str, data_size: float, is_write: bool) -> simpy.Event:
        if is_write:
            return self._write.send(tid, data_size)
        else:
            return self._read.send(tid, data_size)

    @property
    def bandwidth(self) -> float:
        return self._total_bandwidth

    @property
    def write_peak_concurrency(self) -> int:
        return self._write.peak_concurrency

    @property
    def read_peak_concurrency(self) -> int:
        return self._read.peak_concurrency

    @property
    def peak_concurrency(self) -> int:
        return max(self._write.peak_concurrency, self._read.peak_concurrency)


class HcclSimulator:
    """HCCL 算法性能仿真器。

    接收 (config, dep_graph, topo_resolver)。
    数据传输任务通过 eid 查 ranktable 找端口:
      write → task.src_eid → ports
      read  → task.dst_eid → ports

    模块入口:
      sim = HcclSimulator(config, dep_graph, topo_resolver)
      result = sim.run()  # 返回 dict, 含 makespan, trace 路径, 运行耗时等
    """

    def __init__(self, config: SimConfig, dep_graph: DependencyGraph,
                 topo_resolver: TopoResolver):
        self.config = config
        self.dep_graph = dep_graph
        self.topo_resolver = topo_resolver
        self.tasks: list[Task] = []
        self._notify_events: dict[str, simpy.Event] = {}
        self._task_done_events: dict[str, simpy.Event] = {}
        self._ports: dict[tuple, NetworkPort] = {}
        self._hbm_mgr: _BandwidthManager = None
        self._makespan: float = 0.0

    def _build_sync_primitives(self, env: simpy.Environment) -> None:
        for wait_task_id in self.dep_graph.matched_wait_ids:
            self._notify_events[wait_task_id] = env.event()
        # extra 边父任务的完成事件 (7.4 节 fork/join): 须在流进程启动前预建,
        # 否则父任务先完成时子任务后取事件会永久等待
        for parent_id in self.dep_graph.extra_parent_ids:
            self._task_done_events[parent_id] = env.event()

    def run(self, trace_filepath: str = None, max_trace_events: int = 0) -> dict:
        """运行仿真, 返回结果 dict。

        Args:
            trace_filepath: trace 输出路径, None 则不输出
            max_trace_events: trace 事件上限, 0=全部
        Returns:
            {
                "makespan": float (秒),                 # 任务执行耗时
                "makespan_us": float (微秒),
                "trace_file": str or None,
                "trace_events": int,
                "sim_time_s": float (软件运行耗时秒),
                "tasks": list[Task],
                "ports": dict,
            }
        """
        import time
        from .report import generate_report, export_trace

        t0 = time.perf_counter()

        env = simpy.Environment()
        self.tasks = list(self.dep_graph.tasks.values())
        self._build_sync_primitives(env)

        hbm_profile = BandwidthProfileSet(profile_low=None, profile_high=None)
        self._hbm_mgr = _BandwidthManager(
            env, self.config.hbm_bandwidth * self.config.hbm_utilization,
            hbm_profile, proportional=True)

        procs = [env.process(self._stream_process(env, sid))
                 for sid in self.dep_graph.stream_ids]
        env.run()

        unfinished = sum(1 for t in self.tasks if t.end_time < 0)
        if unfinished:
            logger.error("仿真结束但 %d/%d 个任务未完成 (依赖图存在环, 死锁)",
                         unfinished, len(self.tasks))

        self._makespan = max(
            (t.end_time for t in self.tasks if t.end_time >= 0),
            default=0.0,
        )
        sim_time = time.perf_counter() - t0

        # 构建 tasks_by_stream 供 report/trace 使用
        tasks_by_stream = {}
        for sid in self.dep_graph.stream_ids:
            tasks_by_stream[sid] = [self.dep_graph.get_task(tid)
                                    for tid in self.dep_graph.get_chain(sid)]

        # 输出 trace
        trace_file = None
        trace_events = 0
        if trace_filepath:
            trace_events = len(export_trace(
                self.tasks, tasks_by_stream, trace_filepath,
                max_events=max_trace_events,
            ))
            trace_file = trace_filepath

        return {
            "makespan": self._makespan,
            "makespan_us": self._makespan * 1e6,
            "trace_file": trace_file,
            "trace_events": trace_events,
            "sim_time_s": sim_time,
            "tasks": self.tasks,
            "ports": self._ports,
            "tasks_by_stream": tasks_by_stream,
        }

    def _stream_process(self, env: simpy.Environment, stream_id: int):
        config = self.config
        notify_events = self._notify_events
        task_done = self._task_done_events
        ports = self._ports
        bw_profile_set = config.bw_profile_set or BandwidthProfileSet()
        hbm_mgr = self._hbm_mgr
        get_task = self.dep_graph.get_task
        dg = self.dep_graph

        for task_id in dg.get_chain(stream_id):
            task = get_task(task_id)
            # extra 边 (fork/join): 等待全部 extra 父任务完成后再开始,
            # start_time 取等待结束时刻 (不含等待时间)
            for pid in dg.get_extra_parents(task_id):
                ev = task_done.get(pid)
                if ev is not None:
                    yield ev
            task.start_time = env.now
            tt = task.task_type

            if tt == TaskType.NOTIFY_RECORD:
                delay = task_timing.calc_notify_record_delay(task, config)
                yield env.timeout(delay)
                if task.is_local_notify and dg.is_matched_record(task_id):
                    for wait_tid, rec_tid in dg._notify_matches.items():
                        if rec_tid == task_id:
                            ev = notify_events.get(wait_tid)
                            if ev is not None and not ev.triggered:
                                ev.succeed(env.now)
                            break

            elif tt == TaskType.NOTIFY_WAIT:
                if task.is_local_notify and dg.is_matched_wait(task_id):
                    ev = notify_events.get(task_id)
                    if ev is not None:
                        yield ev
                    yield env.timeout(task_timing.calc_notify_wait_delay(task, config))
                else:
                    # 未配对本地 wait: 等远端 record (nid 复用, 对端任务不在
                    # 本 db, 5.4 节), local 用 local_wait_delay, remote 用 remote_wait_delay
                    delay = task_timing.calc_notify_wait_delay(task, config)
                    yield env.timeout(delay)

            elif tt in HBM_TYPES:
                # Reduce_Inline / Memcpy: 固定时延 + 共享 HBM 带宽 (动态)
                # HBM 访问次数 (权重): Memcpy = 2 (1读+1写),
                #   Reduce_Inline = 3 (2读+1写, RMW)
                # HBM 流量 = size × 权重; 带宽按权重比例分配
                hbm_access = 3 if tt == TaskType.REDUCE_INLINE else 2
                yield env.timeout(config.hbm_latency)
                yield hbm_mgr.send(task.task_id, float(task.size * hbm_access),
                                   weight=hbm_access)

            elif tt == TaskType.UB_INLINE_WRITE:
                # 同步通知任务 (notify inline 写): 固定时延, 不走端口带宽
                yield env.timeout(task_timing.calc_ub_inline_write_delay(task, config))

            elif tt in TRANSFER_TYPES:
                # 通过 eid 查 ranktable 找端口
                # write → src_eid, read → dst_eid
                # 方向: db 成图路径在 Task 上带显式 is_write (10.3 节 #1 修复);
                # 旧输入路径 (task.txt) 无该字段, 退回 rank 启发式
                if task.is_write is not None:
                    is_write = task.is_write
                else:
                    is_write = (tt == TaskType.WRITE_WITH_NOTIFY or
                                task.src_rank == task.dst_rank or
                                (task.src_rank != INVALID and task.src_rank == 0))
                eid = task.src_eid if is_write else task.dst_eid

                port_list = self.topo_resolver.get_ports_by_eid(eid)
                if not port_list:
                    raise ValueError(f"任务 {task_id} 的 eid={eid} 在 ranktable 中未找到端口")

                task.port = port_list

                port_key = tuple(port_list)
                port = ports.get(port_key)
                if port is None:
                    total_bw = config.get_ports_total_bandwidth(port_list)
                    if len(port_list) > 1:
                        port_profile = bw_profile_set
                    elif port_list == ["d2h"]:
                        port_profile = BandwidthProfileSet.for_single_port(
                            config.d2h_utilization)
                    else:
                        port_profile = BandwidthProfileSet.for_single_port(
                            bw_profile_set.single_port_utilization)
                    port = NetworkPort(env, port_list, total_bw, port_profile)
                    ports[port_key] = port
                yield port.send(task.task_id, float(task.size), is_write)
                if tt == TaskType.WRITE_WITH_NOTIFY:
                    # 写数据落地后向对端发 notify: 固定 record 时延
                    # (对端 wait 不在本 db, 单 rank 仿真前提, 5.4 节;
                    #  自写 src==dst 边缘场景按 local record 时延)
                    yield env.timeout(
                        task_timing.calc_write_with_notify_delay(task, config))

            else:
                raise ValueError(f"未知任务类型: {tt}")

            task.end_time = env.now
            # 通知 extra 边子任务: 本任务已完成 (事件可能无人等待, 无害)
            done_ev = task_done.get(task_id)
            if done_ev is not None and not done_ev.triggered:
                done_ev.succeed(env.now)
