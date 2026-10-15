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

"""性能仿真共享 pipeline: db → 任务图 → 仿真 → perf_profiling.json。

供两个入口复用 (设计文档 3.1 节):
    - 仓库根目录 run_perf_sim.py              源码树手动入口 (argparse)
    - light_performance/main.py               安装树 @estimate 插件入口 (manifest setting)

完整串联端到端流程 (设计文档 1.1 节):
    ① db_reader 读取 hccl_vm_data.db (第 4 章)
    ② graph_builder 成图 (SQE 映射 + CCU 微码展开, 第 5、6 章)
    ③ HcclSimulator 性能仿真 (第 7 章)
    ④ 调用 export_trace() 生成 perf_profiling.json (Chrome tracing 格式,
       多算子按 pid 分进程组)
"""

from __future__ import annotations

import json
import os

from .__main__ import _find_ranktable
from .db_reader import build_rank_index, load_op_executions
from .graph_builder import GraphBuilder, _DATA_TYPE_NAMES
from .graph_dumper import dump_group
from .simulator.bandwidth_profile import BandwidthProfileSet
from .simulator.config import SimConfig
from .simulator.input_builder import load_config
from .simulator.report import export_trace, generate_report
from .simulator.simulator import HcclSimulator
from .simulator.task_model import TRANSFER_TYPES
from .simulator.topo_resolver import TopoResolver
from .sqe_group_merge import merge_sqe_group_tasks, merge_dpu_write_sequences

# HcclCMDType (checker/header/external/hccl_types.h) 常见值 → 名称
OP_NAMES = {0: "INVALID", 1: "BROADCAST", 2: "ALLREDUCE", 3: "REDUCE",
            4: "SEND", 5: "RECEIVE", 6: "ALLGATHER", 7: "REDUCE_SCATTER",
            8: "ALLTOALLV", 9: "ALLTOALLVC", 10: "ALLTOALL", 11: "GATHER",
            12: "SCATTER", 13: "BATCH_SEND_RECV", 16: "ALLGATHER_V",
            17: "REDUCE_SCATTER_V", 20: "HALF_ALLTOALLV", 100: "FINALIZE",
            102: "INIT", 103: "BARRIER"}


def _dtype_name(data_type: int) -> str:
    return _DATA_TYPE_NAMES.get(data_type, f"type{data_type}")


def build_op_summary(group, result, op_name: str, task_count: int = 0) -> dict:
    """组装算子综合信息 (每轮 comm_iter 的 trace 首行)。

    args 字段: 算法选择结果 (opDetails.algorithmName, 日志截取回填)、
    数据类型、数据量 (元素数, V2 布局的 AllToAll/AllToAllV/AllToAllVC
    为收发两侧, 其余算子为 count)、task_count (成图后任务总数; CCU 模式
    即微码指令翻译成 task 图后的数量) 及 op 上下文; 算子起始时间与整体
    持续时间由汇总切片自身的 ts/dur 承载 (取任务最早 start 到最晚 end)。
    """
    tasks = [t for t in result["tasks"]
             if t.start_time >= 0 and t.end_time >= 0]
    if tasks:
        start_s = min(t.start_time for t in tasks)
        wall_us = round((max(t.end_time for t in tasks) - start_s) * 1e6, 3)
        start_us = round(start_s * 1e6, 3)
    else:
        start_us = 0.0
        wall_us = 0.0

    algorithm_name = group.algorithm_name or "unknown"
    if group.op_type in (8, 9, 10):  # AllToAllV/AllToAllVC/AllToAll (V2 布局)
        data_type = (f"send={_dtype_name(group.send_data_type)}, "
                     f"recv={_dtype_name(group.recv_data_type)}")
        data_count = (f"send={group.send_count}, "
                      f"recv={group.recv_count}")
    else:
        data_type = _dtype_name(group.data_type)
        data_count = f"{group.data_size}"

    args = {
        "algorithm_name": algorithm_name,
        "op_type": op_name,
        "data_type": data_type,
        "data_count": data_count,
        "task_count": task_count,
        "comm_id": group.comm_id,
        "op_iter": group.op_iter,
        "rank": group.rank_id,
        "rank_size": group.rank_size,
    }
    if group.mem_info is not None:
        args["input_bytes"] = group.mem_info.input_size
        args["output_bytes"] = group.mem_info.output_size
        # 算法带宽(GB/s): 取 input/output 较大值作为算法数据量 (gather 型取
        # output, scatter 型取 input, 对称型二者相等), 除以算子 Wall Duration。
        # 与 task_model.bandwidth_gbs 同公式: bytes / us / 1000 → GB/s。
        data_bytes = max(group.mem_info.input_size, group.mem_info.output_size)
        if wall_us > 0 and data_bytes > 0:
            args["alg_bw(GB/s)"] = round(data_bytes / wall_us / 1000.0, 4)
        else:
            args["alg_bw(GB/s)"] = 0.0
    return {
        "name": f"{op_name} | {algorithm_name}",
        "start_us": start_us,
        "wall_duration_us": wall_us,
        "args": args,
    }


def run_perf_pipeline(
    db_path: str,
    out_path: str,
    *,
    config_path: str = None,
    bw_profile_path: str = None,
    ranktable_path: str = None,
    device: int = None,
    iter_filter: int = None,
    trace_dir: str = None,
    dump_dir: str = None,
    max_trace_events: int = 0,
    verbose: bool = False,
    summary_only: bool = False,
) -> int:
    """端到端仿真: 逐算子成图 + 仿真, 汇总输出 perf_profiling.json。

    Args:
        db_path: hccl_vm_data.db 路径 (必填)
        out_path: perf_profiling.json 输出路径 (必填)
        config_path / bw_profile_path: 时延/带宽与利用率配置 JSON
            (缺省用内置默认值, 第 7.2 节格式, 单位 us/GBps)
        ranktable_path: 拓扑文件 (缺省取 db 同目录 ranktable.json;
            存在走网任务且无拓扑时该算子跳过仿真)
        device: 单 rank 成图的目标 deviceId (缺省取 opDetails 首个设备)
        iter_filter: 仅仿真指定 opIter 的算子 (缺省全部)
        trace_dir: 额外输出单算子 Chrome tracing (每算子一个文件)
        dump_dir: 额外输出任务图 dump (DOT/JSON, 每算子)
        max_trace_events: trace 事件上限 (0=全部)
        verbose / summary_only: 任务级明细 / 仅汇总

    Returns:
        进程退出码 (0=成功)。
    """
    if not os.path.isfile(db_path):
        print(f"错误: db 文件不存在: {db_path}")
        return 1

    groups = load_op_executions(db_path, target_device=device)
    if not groups:
        print("错误: db 中未读取到算子执行数据")
        return 1
    if iter_filter is not None:
        groups = [g for g in groups if g.op_iter == iter_filter]
        if not groups:
            print(f"错误: 未找到 opIter={iter_filter}")
            return 1

    # ① 配置与拓扑
    config = load_config(config_path) if config_path else SimConfig()
    if bw_profile_path:
        config.bw_profile_set = BandwidthProfileSet.from_json(bw_profile_path)
    ranktable = _find_ranktable(db_path, ranktable_path)
    topo = None
    if ranktable:
        topo = TopoResolver(ranktable_path=ranktable)
    else:
        print("警告: 未找到 ranktable.json, 走网任务将跳过仿真")

    # eid→rank 索引 (4.7 节): RankEidMap 表优先, ranktable.json 回退
    rank_map = build_rank_index(db_path, ranktable)

    if trace_dir:
        os.makedirs(trace_dir, exist_ok=True)
    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)

    # ②③ 逐算子: SQE 批合并 + DPU 序列合并 + 成图 + 仿真
    op_count = 0
    sim_results = []  # (group, result, builder): 已仿真的算子
    for group in groups:
        # SQE 批合并: 同批同键 UDMA 任务并成 1 个 (见 sqe_group_merge)
        group.tasks, merged = merge_sqe_group_tasks(group.tasks)
        if merged:
            print(f"[MERGE] comm{group.comm_id}/iter{group.op_iter}: "
                  f"{merged} 个 UDMA 数据任务并入同组 (剩余 {len(group.tasks)})")
        # DPU 序列合并: 大消息前块并入 WRITE_WITH_NOTIFY, 与其他入口口径一致
        group.tasks, merged = merge_dpu_write_sequences(group.tasks)
        if merged:
            print(f"[MERGE-DPU] comm{group.comm_id}/iter{group.op_iter}: "
                  f"{merged} 个前块 MEM_CPY 并入 WRITE_WITH_NOTIFY "
                  f"(剩余 {len(group.tasks)})")
        builder = GraphBuilder(group, config, rank_map=rank_map)
        _, dep = builder.build()
        if dep.task_count == 0:
            print(f"[SKIP] comm{group.comm_id}/iter{group.op_iter}: "
                  "依赖图为空")
            continue
        op_count += 1
        op_name = OP_NAMES.get(group.op_type, f"OP_{group.op_type}")

        has_transfer = any(t.task_type in TRANSFER_TYPES
                           for t in dep.tasks.values())
        result = None
        if has_transfer and topo is None:
            print(f"[SKIP] comm{group.comm_id}/iter{group.op_iter} "
                  f"({op_name}): 存在走网任务但无 ranktable "
                  "(eid→端口不可解析), 跳过仿真")
        else:
            sim = HcclSimulator(config, dep, topo)
            result = sim.run()
            sim_results.append((group, result, builder, dep))
            print(f"[OK] comm{group.comm_id}/iter{group.op_iter} "
                  f"({op_name}): {builder.stats.summary()} → "
                  f"makespan {result['makespan_us']:.3f} us")
            print(generate_report(result["tasks"],
                                  result["tasks_by_stream"],
                                  result["ports"], config,
                                  result["makespan"],
                                  stream_meta=builder.stream_meta,
                                  verbose=verbose,
                                  summary_only=summary_only))
            print(f"软件运行耗时: {result['sim_time_s']:.4f} s")

        if result is not None and trace_dir:
            path = os.path.join(trace_dir,
                                f"trace_comm{group.comm_id}"
                                f"_iter{group.op_iter}.json")
            count = len(export_trace(result["tasks"],
                                      result["tasks_by_stream"], path,
                                      max_events=max_trace_events,
                                      stream_meta=builder.stream_meta,
                                      op_summary=build_op_summary(
                                          group, result, op_name,
                                          dep.task_count)))
            print(f"      trace: {path} (事件数 {count})")
        if dump_dir:
            prefix = os.path.join(dump_dir,
                                  f"graph_comm{group.comm_id}"
                                  f"_iter{group.op_iter}_dev{group.device_id}")
            meta = {"op_desc": op_name,
                    "makespan_us": (result["makespan_us"]
                                    if result is not None else None)}
            for path in dump_group(dep, prefix, meta, stats=builder.stats,
                                   stream_labels=builder.stream_labels,
                                   stream_meta=builder.stream_meta):
                print(f"      dump: {path}")

    # ④ 汇总输出 perf_profiling.json: 调用 export_trace() 获取 Chrome
    #    tracing 事件; 多算子时按 pid 分进程组合并到同一文件
    multi_op = len(sim_results) > 1
    events = []
    for pid, (group, result, builder, dep) in enumerate(sim_results):
        op_name = OP_NAMES.get(group.op_type, f"OP_{group.op_type}")
        events.extend(export_trace(
            result["tasks"], result["tasks_by_stream"],
            pid=pid,
            process_name=(f"comm{group.comm_id}/iter{group.op_iter} "
                           f"{op_name}") if multi_op else None,
            stream_meta=builder.stream_meta,
            op_summary=build_op_summary(group, result, op_name,
                                        dep.task_count),
        ))

    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        # Chrome Trace Event 格式: ts/dur 值单位为 us (规范约定);
        # displayTimeUnit 指定查看器显示单位为 ns (缺省 ms)
        json.dump({"displayTimeUnit": "ns", "traceEvents": events}, f,
                  separators=(",", ":"))

    print(f"\n共 {op_count} 个算子 (仿真 {len(sim_results)} 个), "
          f"输出: {os.path.abspath(out_path)} (事件数 {len(events)})")
    return 0
