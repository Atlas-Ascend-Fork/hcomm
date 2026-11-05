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

"""light_performance db → 任务图 → 仿真 CLI 入口 (hccl_sim 新版任务模型)。

用法:
    python -m light_performance --db /path/to/hccl_vm_data.db [--device 1]
        [--iter 0] [--config sim_config.json] [--bw-profile bw_profile.json]
        [--ranktable ranktable.json] [-v] [-s] [--trace trace.json]
"""

from __future__ import annotations

import argparse
import os
import time

from .db_reader import build_rank_index, load_op_executions, load_port_type
from .graph_builder import GraphBuilder
from .graph_dumper import dump_group
from .sqe_group_merge import merge_sqe_group_tasks, merge_dpu_write_sequences
from .simulator.bandwidth_profile import BandwidthProfileSet
from .simulator.config import SimConfig
from .simulator.input_builder import load_config
from .simulator.report import export_trace, generate_report
from .simulator.simulator import HcclSimulator
from .simulator.task_model import TRANSFER_TYPES


def _find_ranktable(db_path: str, explicit: str = None):
    """ranktable 定位: 显式指定优先, 缺省取 db 同目录 ranktable.json。"""
    if explicit:
        if os.path.isfile(explicit):
            return explicit
        print(f"WARNING: specified ranktable does not exist: {explicit}")
        return None
    candidate = os.path.join(os.path.dirname(os.path.abspath(db_path)),
                            "ranktable.json")
    return candidate if os.path.isfile(candidate) else None


def main():
    parser = argparse.ArgumentParser(
        description="light_performance: hccl db → 任务图 → 仿真 (基于 simpy)")
    parser.add_argument("--db", required=True, help="hccl_vm_data.db 路径")
    parser.add_argument("--device", type=int, default=None,
                        help="目标 deviceId (默认取 opDetails 首个设备)")
    parser.add_argument("--iter", type=int, default=None,
                        help="仅仿真指定 opIter (默认全部)")
    parser.add_argument("--jetty-num", type=int, default=None,
                        help="覆盖 db 落库的 UB 多 channel(jetty)数 "
                             "(HCCL_UB_MULTI_CHANNEL_NUM), 用于折算带宽利用率")
    parser.add_argument("--port-type", choices=["x2", "x4"], default=None,
                        help="覆盖 db 落库的端口类型 (Server.port_type); "
                             "配置带宽基准为 x4, x2 时端口带宽减半 (缺省读 db 值)")
    parser.add_argument("--config", help="时延/带宽配置 JSON (可选, 单位 us/GBps)")
    parser.add_argument("--bw-profile", help="带宽利用率配置 JSON (可选)")
    parser.add_argument("--ranktable",
                        help="ranktable.json (eid→端口解析; 默认取 db 同目录)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="输出任务级明细")
    parser.add_argument("--summary", "-s", action="store_true",
                        help="仅输出汇总")
    parser.add_argument("--trace", metavar="FILE",
                        help="导出 Chrome tracing JSON")
    parser.add_argument("--dump-dir", metavar="DIR",
                        help="任务图 dump 目录 (每算子输出 .json + .html)")
    parser.add_argument("--max-trace-events", type=int, default=0,
                        help="trace 事件上限 (0=全部)")
    args = parser.parse_args()

    groups = load_op_executions(args.db, target_device=args.device)
    if not groups:
        print("ERROR: no operator execution data read from db")
        return
    if args.iter is not None:
        groups = [g for g in groups if g.op_iter == args.iter]
        if not groups:
            print("ERROR: specified opIter not found")
            return

    config = load_config(args.config) if args.config else SimConfig()
    if args.bw_profile:
        config.bw_profile_set = BandwidthProfileSet.from_json(args.bw_profile)
    # 端口类型: db Server.port_type 为准 (server/pod 拓扑配置 port_type 字段),
    # --port-type 非空时整体覆盖 (调试/旧库无该列); x2 时端口带宽减半
    config = config.with_port_type(
        args.port_type or load_port_type(args.db, groups[0].device_id))
    print(f"[PORT_TYPE] port_type={config.port_type}, port bandwidth "
          f"scale={config.port_type_scale:g} (config baseline is x4)")

    ranktable = _find_ranktable(args.db, args.ranktable)
    topo = None
    if ranktable:
        from .simulator.topo_resolver import TopoResolver
        topo = TopoResolver(ranktable_path=ranktable)
    else:
        print("WARNING: ranktable.json not found, network tasks cannot resolve "
              "ports (simulation skipped when network tasks exist)")

    # eid→rank 索引 (4.7 节): RankEidMap 表优先, ranktable.json 回退
    rank_map = build_rank_index(args.db, ranktable)

    for group in groups:
        t0 = time.perf_counter()
        # SQE 批合并: 同批同键 UDMA 任务并成 1 个 (见 sqe_group_merge), 与 pipeline 入口口径一致
        group.tasks, merged = merge_sqe_group_tasks(group.tasks)
        if merged:
            print(f"[MERGE] comm{group.comm_id}/iter{group.op_iter}: "
                  f"{merged} UDMA data tasks merged into one group "
                  f"({len(group.tasks)} remaining)")
        # DPU 序列合并: 大消息前块并入 WRITE_WITH_NOTIFY, 与 pipeline 入口口径一致
        group.tasks, merged = merge_dpu_write_sequences(group.tasks)
        if merged:
            print(f"[MERGE-DPU] comm{group.comm_id}/iter{group.op_iter}: "
                  f"{merged} leading MEM_CPY merged into WRITE_WITH_NOTIFY "
                  f"({len(group.tasks)} remaining)")
        builder = GraphBuilder(group, config, rank_map=rank_map)
        _, dep = builder.build()
        if dep.task_count == 0:
            print(f"comm{group.comm_id}/iter{group.op_iter}: dependency graph is empty, skip")
            continue
        print(f"[OK] comm{group.comm_id}/iter{group.op_iter}: "
              f"{builder.stats.summary()}")

        # 多 jetty 折算: 以 db 落库值为准 (proxy 读 HCCL_UB_MULTI_CHANNEL_NUM),
        # --jetty-num 非空时整体覆盖 (调试/旧库无该列)
        group_jetty_num = args.jetty_num or group.ub_multi_channel_num
        sim_config = config.for_jetty_num(group_jetty_num)
        if group_jetty_num > 1:
            print(f"[JETTY] comm{group.comm_id}/iter{group.op_iter}: "
                  f"{sim_config.describe_jetty_num(group_jetty_num)}")

        has_transfer = any(t.task_type in TRANSFER_TYPES
                           for t in dep.tasks.values())
        makespan = None
        if has_transfer and topo is None:
            print(f"comm{group.comm_id}/iter{group.op_iter}: network tasks exist but "
                  "no ranktable (eid→port unresolvable), skip simulation")
        else:
            sim = HcclSimulator(sim_config, dep, topo)
            result = sim.run()
            makespan = result["makespan"]
            tasks_by_stream = result["tasks_by_stream"]
            print(generate_report(
                result["tasks"], tasks_by_stream, result["ports"], sim_config,
                makespan, stream_meta=builder.stream_meta,
                verbose=args.verbose, summary_only=args.summary))
            print(f"Software runtime: {result['sim_time_s']:.3f} s "
                  f"(build {time.perf_counter() - t0:.3f} s)")

            if args.trace:
                base, ext = os.path.splitext(args.trace)
                path = (f"{base}_comm{group.comm_id}_iter{group.op_iter}"
                        f"{ext or '.json'}")
                count = len(export_trace(result["tasks"], tasks_by_stream,
                                          path, max_events=args.max_trace_events,
                                          stream_meta=builder.stream_meta))
                print(f"Trace exported: {path} ({count} events)")

        if args.dump_dir:
            os.makedirs(args.dump_dir, exist_ok=True)
            prefix = os.path.join(
                args.dump_dir,
                f"graph_comm{group.comm_id}_iter{group.op_iter}"
                f"_dev{group.device_id}")
            meta = {"op_desc": f"comm{group.comm_id}/iter{group.op_iter}"
                               f"/dev{group.device_id}",
                    "makespan_us": makespan * 1e6 if makespan else None}
            for path in dump_group(dep, prefix, meta, stats=builder.stats,
                                   stream_labels=builder.stream_labels,
                                   stream_meta=builder.stream_meta):
                print(f"Graph dump: {path}")


if __name__ == "__main__":
    main()
