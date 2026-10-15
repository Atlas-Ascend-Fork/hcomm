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

from .db_reader import build_rank_index, load_op_executions
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
        print(f"警告: 指定的 ranktable 不存在: {explicit}")
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
        print("错误: db 中未读取到算子执行数据")
        return
    if args.iter is not None:
        groups = [g for g in groups if g.op_iter == args.iter]
        if not groups:
            print("错误: 未找到指定 opIter")
            return

    config = load_config(args.config) if args.config else SimConfig()
    if args.bw_profile:
        config.bw_profile_set = BandwidthProfileSet.from_json(args.bw_profile)

    ranktable = _find_ranktable(args.db, args.ranktable)
    topo = None
    if ranktable:
        from .simulator.topo_resolver import TopoResolver
        topo = TopoResolver(ranktable_path=ranktable)
    else:
        print("警告: 未找到 ranktable.json, 走网任务将无法解析端口 "
              "(存在走网任务时跳过仿真)")

    # eid→rank 索引 (4.7 节): RankEidMap 表优先, ranktable.json 回退
    rank_map = build_rank_index(args.db, ranktable)

    for group in groups:
        t0 = time.perf_counter()
        # SQE 批合并: 同批同键 UDMA 任务并成 1 个 (见 sqe_group_merge), 与 pipeline 入口口径一致
        group.tasks, merged = merge_sqe_group_tasks(group.tasks)
        if merged:
            print(f"[MERGE] comm{group.comm_id}/iter{group.op_iter}: "
                  f"{merged} 个 UDMA 数据任务并入同组 (剩余 {len(group.tasks)})")
        # DPU 序列合并: 大消息前块并入 WRITE_WITH_NOTIFY, 与 pipeline 入口口径一致
        group.tasks, merged = merge_dpu_write_sequences(group.tasks)
        if merged:
            print(f"[MERGE-DPU] comm{group.comm_id}/iter{group.op_iter}: "
                  f"{merged} 个前块 MEM_CPY 并入 WRITE_WITH_NOTIFY "
                  f"(剩余 {len(group.tasks)})")
        builder = GraphBuilder(group, config, rank_map=rank_map)
        _, dep = builder.build()
        if dep.task_count == 0:
            print(f"comm{group.comm_id}/iter{group.op_iter}: 依赖图为空, 跳过")
            continue
        print(f"[OK] comm{group.comm_id}/iter{group.op_iter}: "
              f"{builder.stats.summary()}")

        has_transfer = any(t.task_type in TRANSFER_TYPES
                           for t in dep.tasks.values())
        makespan = None
        if has_transfer and topo is None:
            print(f"comm{group.comm_id}/iter{group.op_iter}: 存在走网任务但"
                  "无 ranktable (eid→端口不可解析), 跳过仿真")
        else:
            sim = HcclSimulator(config, dep, topo)
            result = sim.run()
            makespan = result["makespan"]
            tasks_by_stream = result["tasks_by_stream"]
            print(generate_report(
                result["tasks"], tasks_by_stream, result["ports"], config,
                makespan, stream_meta=builder.stream_meta,
                verbose=args.verbose, summary_only=args.summary))
            print(f"软件运行耗时: {result['sim_time_s']:.3f} s "
                  f"(构建 {time.perf_counter() - t0:.3f} s)")

            if args.trace:
                base, ext = os.path.splitext(args.trace)
                path = (f"{base}_comm{group.comm_id}_iter{group.op_iter}"
                        f"{ext or '.json'}")
                count = len(export_trace(result["tasks"], tasks_by_stream,
                                          path, max_events=args.max_trace_events,
                                          stream_meta=builder.stream_meta))
                print(f"Trace 导出: {path} (事件数 {count})")

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
                print(f"任务图 dump: {path}")


if __name__ == "__main__":
    main()
