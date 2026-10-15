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

"""HCCL 算子性能仿真工具 CLI 入口。

CLI 包装层, 不含功能逻辑。所有功能在 HcclSimulator.run() 中实现。

用法:
    python -m hccl_sim --demo --ranktable ranktable.json
    python -m hccl_sim --tasks tasks.json --config sim_config.json --ranktable ranktable.json
"""

from __future__ import annotations

import argparse
import os

from .config import SimConfig
from .simulator import HcclSimulator
from .topo_resolver import TopoResolver
from .bandwidth_profile import BandwidthProfileSet
from .input_builder import _streams_to_dep_graph, load_config
from .report import generate_report
from .task_model import Task, TaskType, INVALID


def build_demo_input():
    """构造内置演示场景。"""
    config = SimConfig(
        local_record_delay=1e-6, remote_record_delay=1e-6,
        remote_wait_delay=1e-6,
        port_bandwidth={"1/0": 53.125e9},
        all_port_bandwidths={},
        default_port_bandwidth=53.125e9,
        hbm_bandwidth=1.4e12,
    )
    _EID_SDMA = "000000000040000000000000c0010001"
    streams = [
        {"stream_id": "0", "tasks": [
            {"task_id": "t0", "type": "Notify_Record", "notify_id": "1",
             "src_rank": 0, "dst_rank": 0},
            {"task_id": "t1", "type": "Ub_Write_Or_Read",
             "size": 2 * 1024 * 1024,
             "src_eid": _EID_SDMA, "dst_eid": _EID_SDMA,
             "src_rank": 0, "dst_rank": 1},
            {"task_id": "t2", "type": "Notify_Wait", "notify_id": "2",
             "src_rank": 0, "dst_rank": 0},
        ]},
        {"stream_id": "1", "tasks": [
            {"task_id": "t3", "type": "Notify_Wait", "notify_id": "1",
             "src_rank": 0, "dst_rank": 0},
            {"task_id": "t4", "type": "Ub_Write_Or_Read",
             "size": 2 * 1024 * 1024,
             "src_eid": _EID_SDMA, "dst_eid": _EID_SDMA,
             "src_rank": 0, "dst_rank": 2},
            {"task_id": "t5", "type": "Notify_Record", "notify_id": "2",
             "src_rank": 0, "dst_rank": 0},
        ]},
        {"stream_id": "2", "tasks": [
            {"task_id": "t6", "type": "Memcpy",
             "size": 4 * 1024 * 1024,
             "src_rank": 0, "dst_rank": 0},
        ]},
    ]
    dep_graph = _streams_to_dep_graph(streams)
    return config, dep_graph


def main():
    parser = argparse.ArgumentParser(
        description="HCCL 算子性能仿真工具 (基于 simpy)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--config", help="时延/带宽配置 JSON 文件")
    parser.add_argument("--bw-profile", help="带宽利用率配置 JSON 文件")
    parser.add_argument("--ranktable", required=True,
                        help="ranktable.json 文件路径 (必选, 用于 eid→端口查询)")
    parser.add_argument("--tasks", help="JSON 格式任务序列文件")
    parser.add_argument("--demo", action="store_true", help="运行内置演示场景")
    parser.add_argument("--verbose", "-v", action="store_true", help="输出任务级明细")
    parser.add_argument("--summary", "-s", action="store_true", help="仅输出汇总")
    parser.add_argument("--trace", metavar="FILE", default="trace.json",
                        help="Chrome tracing JSON 输出路径 (默认: trace.json)")
    parser.add_argument("--no-trace", action="store_true", help="不输出 trace 文件")
    parser.add_argument("--max-trace-events", type=int, default=0, help="trace 事件上限")
    args = parser.parse_args()

    # --- 构建拓扑解析器 ---
    topo_resolver = TopoResolver(ranktable_path=args.ranktable)

    # --- 加载输入 ---
    if args.tasks:
        import json
        with open(args.tasks, "r", encoding="utf-8") as f:
            data = json.load(f)
        streams = data if isinstance(data, list) else data.get("streams", [])
        dep_graph = _streams_to_dep_graph(streams)
        config = SimConfig()
    elif args.demo:
        config, dep_graph = build_demo_input()
    else:
        parser.print_help()
        print("\n请提供任务序列输入: --tasks / --demo")
        return

    if dep_graph.task_count == 0:
        print("错误: 依赖图中无任务")
        return

    if args.config:
        config = load_config(args.config)

    if args.bw_profile:
        config.bw_profile_set = BandwidthProfileSet.from_json(args.bw_profile)

    # --- 仿真 ---
    trace_path = None if args.no_trace else args.trace
    sim = HcclSimulator(config, dep_graph, topo_resolver)
    result = sim.run(
        trace_filepath=trace_path,
        max_trace_events=args.max_trace_events,
    )

    # --- 输出报告 ---
    print(generate_report(
        result["tasks"], result["tasks_by_stream"], result["ports"],
        config, result["makespan"],
        verbose=args.verbose, summary_only=args.summary,
    ))
    print(f"软件运行耗时: {result['sim_time_s']:.3f} s")

    if result["trace_file"]:
        fsize = os.path.getsize(result["trace_file"]) / 1e6
        print(f"Trace: {result['trace_file']}")
        print(f"  事件数: {result['trace_events']}, 文件大小: {fsize:.2f} MB")

    return result


if __name__ == "__main__":
    main()
