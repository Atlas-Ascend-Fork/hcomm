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

"""light_performance 一键 dump 入口: db → 任务图 → DOT + JSON。

用法 (仓库根目录执行):
    python3 dump_graph.py hccl_vm_install/data/hccl_vm_data.db
    python3 dump_graph.py <db> --device 1 --iter 0 --out graph_dump --no-sim
    python3 dump_graph.py <db> --html        # 额外输出单文件 HTML

输出 (默认 graph_dump/ 目录):
    graph_comm{X}_iter{Y}_dev{Z}.dot    graphviz 任务图 (VSCode Graphviz 插件直接预览)
    graph_comm{X}_iter{Y}_dev{Z}.json   固定格式任务图 (nodes/streams/matches)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "src", "plugin"))

from light_performance.__main__ import _find_ranktable
from light_performance.db_reader import load_op_executions
from light_performance.graph_builder import GraphBuilder
from light_performance.graph_dumper import dump_group
from light_performance.sqe_group_merge import merge_sqe_group_tasks, merge_dpu_write_sequences
from light_performance.simulator.config import SimConfig
from light_performance.simulator.simulator import HcclSimulator
from light_performance.simulator.task_model import TRANSFER_TYPES
from light_performance.simulator.topo_resolver import TopoResolver


def main():
    parser = argparse.ArgumentParser(
        description="读取 hccl_vm_data.db, 生成任务依赖图 dump (JSON + HTML)")
    parser.add_argument("db", help="hccl_vm_data.db 路径")
    parser.add_argument("--device", type=int, default=None,
                        help="目标 deviceId (默认取 opDetails 首个设备)")
    parser.add_argument("--iter", type=int, default=None,
                        help="仅处理指定 opIter (默认全部)")
    parser.add_argument("--ranktable",
                        help="ranktable.json (eid→端口解析; 默认取 db 同目录)")
    parser.add_argument("--out", default="graph_dump",
                        help="dump 输出目录 (默认 graph_dump)")
    parser.add_argument("--no-sim", action="store_true",
                        help="不执行仿真 (dump 中不含任务时刻信息)")
    parser.add_argument("--html", action="store_true",
                        help="额外输出单文件 HTML 视图")
    args = parser.parse_args()

    if not os.path.isfile(args.db):
        print(f"错误: db 文件不存在: {args.db}")
        return 1

    groups = load_op_executions(args.db, target_device=args.device)
    if not groups:
        print("错误: db 中未读取到算子执行数据")
        return 1
    if args.iter is not None:
        groups = [g for g in groups if g.op_iter == args.iter]
        if not groups:
            print(f"错误: 未找到 opIter={args.iter}")
            return 1

    os.makedirs(args.out, exist_ok=True)
    config = SimConfig()
    topo = None
    ranktable = _find_ranktable(args.db, args.ranktable)
    if ranktable:
        topo = TopoResolver(ranktable_path=ranktable)
    outputs = []

    for group in groups:
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
        builder = GraphBuilder(group, config)
        _, dep = builder.build()
        if dep.task_count == 0:
            print(f"comm{group.comm_id}/iter{group.op_iter}: 依赖图为空, 跳过")
            continue

        meta = {"op_desc": f"comm{group.comm_id}/iter{group.op_iter}"
                           f"/dev{group.device_id}",
                "op_type": group.op_type, "rank_size": group.rank_size}
        if not args.no_sim:
            has_transfer = any(t.task_type in TRANSFER_TYPES
                               for t in dep.tasks.values())
            if has_transfer and topo is None:
                print(f"comm{group.comm_id}/iter{group.op_iter}: 存在走网任务"
                      "但无 ranktable, 跳过仿真")
            else:
                sim = HcclSimulator(config, dep, topo)
                result = sim.run()
                meta["makespan_us"] = result["makespan"] * 1e6

        prefix = os.path.join(
            args.out,
            f"graph_comm{group.comm_id}_iter{group.op_iter}"
            f"_dev{group.device_id}")
        paths = dump_group(dep, prefix, meta, stats=builder.stats,
                           stream_labels=builder.stream_labels,
                           stream_meta=builder.stream_meta,
                           html=args.html)
        outputs.extend(paths)
        print(f"[OK] comm{group.comm_id}/iter{group.op_iter}: "
              f"{builder.stats.summary()}")

    print("\n输出文件:")
    for path in outputs:
        tag = "  <- VSCode Graphviz 插件预览" if path.endswith(".dot") else ""
        print(f"  {path}{tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
