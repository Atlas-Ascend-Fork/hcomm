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

"""报告模块: 文本报告 + Chrome tracing JSON 导出。

虚拟流标记 (7.4 节): CCU 队列并行化产生的虚拟流 (kind == "virtual")
行名改用 "vir_stream {n} ({逻辑流名})" (n 按 sid 升序即分配顺序从 0 编号,
替代 2^20 基址起的大数值流号), 任务级 args 加 virtual_stream 标志;
stream_meta 缺省 None (simulator 独立 CLI 旧输入无该信息, 不标注)。

loop 从流呈现 (6.5.1 节, 仅呈现层): 虚拟从流上携带 loop_tag 的任务,
其显示行改挂本流的 loop 标签行 (行名即标签, 如 loopgroup_0_loop_0),
表明其为 loop 内指令转换而来; 同一流其余任务保持原显示行 (跨 mission
复用的混合流因此拆为原行 + 标签行)。显示行按 (stream_id, loop_tag)
划分, 不同虚拟流的任务不合行 (保留从流间任务划分), 主流与其余从流
显示不变, 不影响任何仿真时序。
"""

from __future__ import annotations

import json
import logging

from .task_model import Task, INVALID, TRANSFER_TYPES

logger = logging.getLogger(__name__)

# stream 明细行数上限: 超出后省略逐行列表, 避免大规模并行图刷屏
_STREAM_LIST_LIMIT = 64


def _virtual_stream_ids(stream_meta: dict) -> set:
    """stream_meta → 虚拟流 id 集合。"""
    return {sid for sid, m in (stream_meta or {}).items()
            if m.get("kind") == "virtual"}


def _vir_display_ids(virtual_sids: set, tasks_by_stream: dict) -> dict:
    """虚拟流显示编号: 有任务的虚拟流按 sid 升序 (即分配顺序) 从 0 编号。

    行名 "vir_stream {n} ({逻辑流名})" 替代 "stream {sid} (...)", 摆脱
    2^20 基址起的大数值流号; 真实流号保留在任务 args.stream_id 中可溯源。
    """
    return {sid: n for n, sid in enumerate(sorted(
        virtual_sids & tasks_by_stream.keys()))}


def generate_report(
    tasks: list[Task],
    tasks_by_stream: dict,
    ports: dict,
    config,
    makespan: float,
    stream_meta: dict = None,
    verbose: bool = False,
    summary_only: bool = False,
) -> str:
    virtual_sids = _virtual_stream_ids(stream_meta)
    lines = []
    lines.append("=" * 72)
    lines.append("HCCL 算子性能仿真报告")
    lines.append("=" * 72)
    lines.append(f"任务总数: {len(tasks)}")
    lines.append(f"Stream 数: {len(tasks_by_stream)}"
                 + (f" (含虚拟 {len(virtual_sids)})"
                    if virtual_sids else ""))
    lines.append(f"任务执行耗时: {makespan * 1e6:.2f} us")
    unfinished = sum(1 for t in tasks if t.end_time < 0)
    if unfinished:
        lines.append(f"警告: {unfinished} 个任务未完成 (依赖图存在环, 仿真死锁)")
    lines.append("")

    if summary_only:
        lines.append("-" * 72)
        lines.append("网卡端口汇总:")
        if not ports:
            lines.append("  (无数据传输任务)")
        max_conc = 0
        total_bytes = 0.0
        for pkey, p in ports.items():
            max_conc = max(max_conc, p.peak_concurrency)
            total_bytes += sum(t.size for t in tasks
                               if t.task_type in TRANSFER_TYPES
                               and tuple(t.port if isinstance(t.port, list) else [t.port]) == pkey)
        lines.append(f"  端口数: {len(ports)}")
        lines.append(f"  峰值并发: {max_conc}")
        lines.append(f"  总传输量: {total_bytes / 1e9:.4f} GB")
        lines.append("")
        lines.append("=" * 72)
        return "\n".join(lines)

    lines.append("-" * 72)
    if len(tasks_by_stream) > _STREAM_LIST_LIMIT:
        lines.append(f"Stream 完成时间: 共 {len(tasks_by_stream)} 条 "
                     f"(超过 {_STREAM_LIST_LIMIT}, 略去明细)")
    else:
        lines.append("Stream 完成时间:")
        vir_id = _vir_display_ids(virtual_sids, tasks_by_stream)
        for sid in sorted(tasks_by_stream):
            stasks = tasks_by_stream[sid]
            s0 = stasks[0].start_time
            s1 = stasks[-1].end_time
            m = (stream_meta or {}).get(sid)
            if m is not None and m.get("kind") == "virtual":
                head = f"vir_stream {vir_id[sid]} ({m.get('label', '')})"
            elif m is not None and m.get("kind") == "ccu":
                head = f"stream {sid} ({m.get('label', '')})"
            else:
                head = f"stream {sid}"
            lines.append(f"  {head}: {s0:.9f} -> {s1:.9f} s  "
                         f"(耗时 {(s1 - s0) * 1e6:.2f} us, {len(stasks)} 任务)")
    lines.append("")

    lines.append("-" * 72)
    lines.append("网卡端口统计:")
    if not ports:
        lines.append("  (无数据传输任务)")
    for pkey in sorted(ports, key=lambda x: str(x)):
        p = ports[pkey]
        bw_gbs = p.bandwidth / 1e9
        lines.append(f"  {pkey}: 带宽 {bw_gbs:.2f} GB/s/方向 (write+read 独立), "
                     f"write峰值 {p.write_peak_concurrency}, "
                     f"read峰值 {p.read_peak_concurrency}")
    lines.append("")

    if verbose:
        lines.append("-" * 72)
        lines.append("任务明细 (按开始时间排序):")
        for t in sorted(tasks, key=lambda x: x.start_time):
            args = t.to_trace_args()
            lines.append(f"  {t.task_id:<12} {t.display_name:<18} stream={t.stream_id}  "
                         f"dur={t.duration_us:.3f}us  bw={t.bandwidth_gbs:.2f}GB/s")
    lines.append("")

    lines.append("=" * 72)
    return "\n".join(lines)


def export_trace(
    tasks: list[Task],
    tasks_by_stream: dict,
    filepath: str = None,
    max_events: int = 0,
    pid: int = 0,
    process_name: str = None,
    stream_meta: dict = None,
    op_summary: dict = None,
) -> list:
    """构建 Chrome tracing 事件列表 (最后 json 数据), filepath 非空时写入文件。

    Args:
        tasks: 仿真后的任务列表 (start_time/end_time 已填充)
        tasks_by_stream: stream_id → 任务链
        filepath: 输出文件路径; None 则只构建事件不写文件
        max_events: 任务事件上限 (0=全部, metadata 不计入)
        pid: 进程标识; 多算子合并导出时用于区分不同算子的进程组
        process_name: 进程名 (chrome://tracing 进程组标签); None 则不添加
        stream_meta: 流元信息 (graph_builder.stream_meta); 虚拟流行名改用
            "vir_stream {n} ({逻辑流名})" (n 从 0 顺序编号), 任务 args 加
            virtual_stream
        op_summary: 算子综合信息 (每轮 comm_iter 的首行); None 则不加。
            结构: {"name": 行标签, "start_us": 起始时间, "wall_duration_us":
            持续时间, "args": 汇总字段}; 非空时以 tid 1 + thread_sort_index
            0 保留首行 (任务行 tid 与 sort_index 顺延), 事件置于数组最前,
            强制 op_summary 恒为每个算子的第一行; 汇总事件不计入 max_events

    Returns:
        事件列表 (含 metadata 事件), 调用方可直接 json.dump 或跨算子合并。
    """
    events = []
    virtual_sids = _virtual_stream_ids(stream_meta)

    # 算子综合信息首行: tid 避开 0 (部分查看器如 Perfetto 将 tid 0 视作
    # idle 线程特殊处理), 取 tid 1 + thread_sort_index 0 (Chrome Trace
    # 标准行序控制, chrome://tracing 与 Perfetto 均支持), 任务行 tid 与
    # sort_index 顺延 —— 强制 op_summary 恒为每个算子 (pid 进程组) 的
    # 第一行; 事件置于数组最前, 兼容按首次出现顺序建 track 的查看器
    tid_base = 1
    if op_summary is not None:
        tid_base = 2
        events.append({
            "name": "thread_name", "ph": "M",
            "pid": pid, "tid": 1,
            "args": {"name": "op_summary"},
        })
        events.append({
            "name": "thread_sort_index", "ph": "M",
            "pid": pid, "tid": 1,
            "args": {"sort_index": 0},
        })
        start_us = float(op_summary.get("start_us", 0.0))
        dur_us = max(0.0, float(op_summary.get("wall_duration_us", 0.0)))
        events.append({
            "name": op_summary.get("name", "op_summary"),
            "cat": "op_summary",
            "ph": "X", "ts": start_us, "dur": dur_us,
            "pid": pid, "tid": 1,
            "args": op_summary.get("args", {}),
        })

    # loop 显示行划分 (6.5.1 节, 仅呈现层): 虚拟流上携带 loop_tag 的任务
    # 改挂本流的 loop 标签行, 同一流其余任务保持原行; 行键 (stream_id,
    # loop_tag) 保证不同虚拟流的任务不合行。主流任务 (含 loop 末 wait)
    # 一律保持原行
    task_row: dict = {}
    for t in tasks:
        tag = t.loop_tag if t.stream_id in virtual_sids else ""
        task_row[t.task_id] = (t.stream_id, tag)

    # 进程组标签 (可选), 与 thread_name 同为 Chrome tracing metadata 事件
    if process_name is not None:
        events.append({
            "name": "process_name", "ph": "M",
            "pid": pid, "tid": 0,
            "args": {"name": process_name},
        })

    # tid 映射: 按 stream id 升序逐流编号, 流内先原行 (无标签任务) 后各
    # loop 标签行。直接用原始 streamId 作 tid 时, 部分查看器按字符串排序
    # 会把 10..13 排到 2 之前; 顺序编号在数值/字典序下均稳定。真实
    # stream id 保留在 thread_name 标签与 args.stream_id 中
    rows: list = []
    for sid in sorted(tasks_by_stream):
        stasks = tasks_by_stream[sid]
        tags = sorted({task_row.get(t.task_id, (sid, ""))[1]
                       for t in stasks} - {""})
        if not tags or any(task_row.get(t.task_id, (sid, ""))[1] == ""
                           for t in stasks):
            rows.append((sid, ""))
        rows.extend((sid, tag) for tag in tags)
    row_map = {row: idx + tid_base for idx, row in enumerate(rows)}

    # 为每个显示行添加 metadata event, 设置 Chrome tracing 左侧的行标签
    # metadata 不计入 max_events 限制; thread_sort_index 与 tid 序一致
    # (op_summary 行 0 最前, 任务行 1..N), 免疫查看器排序策略差异
    vir_id = _vir_display_ids(virtual_sids, tasks_by_stream)
    for (sid, tag), tid in row_map.items():
        if tag:
            # loop 标签行: 行名即 loop 标签 (任务仍属本虚拟流, 真实流号
            # 见 args.stream_id)
            name = tag
        else:
            m = (stream_meta or {}).get(sid)
            if m is not None and m.get("kind") == "virtual":
                # 虚拟流: vir_stream + 顺序编号 (0 起, 按 sid 即分配顺序),
                # 括号内为逻辑流名; 真实流号见行内任务 args.stream_id
                name = f"vir_stream {vir_id[sid]} ({m.get('label', '')})"
            else:
                name = f"stream {sid}"
        events.append({
            "name": "thread_name", "ph": "M",
            "pid": pid, "tid": tid,
            "args": {"name": name},
        })
        events.append({
            "name": "thread_sort_index", "ph": "M",
            "pid": pid, "tid": tid,
            "args": {"sort_index": tid - tid_base + 1},
        })

    task_event_count = 0
    skipped_tasks = 0
    for t in sorted(tasks, key=lambda x: x.start_time):
        if t.start_time < 0 or t.end_time < 0:
            skipped_tasks += 1
            continue

        tid = row_map.get(task_row.get(t.task_id, (t.stream_id, "")),
                          t.stream_id + tid_base)
        # 端点分别取整: ts 与 dur 各自独立取整会产生伪重叠 (后继 ts 早于
        # 前驱 end 0.001us), 同 tid 重叠切片会被查看器嵌套成子行
        ts_us = round(t.start_time * 1e6, 3)
        end_us = round(t.end_time * 1e6, 3)
        dur_us = max(0.0, round(end_us - ts_us, 3))

        args = t.to_trace_args()
        if t.stream_id in virtual_sids:
            args["virtual_stream"] = True

        events.append({
            "name": t.display_name,
            "cat": t.display_name,
            "ph": "X", "ts": ts_us, "dur": dur_us,
            "pid": pid, "tid": tid, "args": args,
        })

        task_event_count += 1
        if max_events > 0 and task_event_count >= max_events:
            break

    if skipped_tasks:
        logger.warning("trace 跳过 %d 个未完成任务 (start/end 时间无效, 疑似死锁)",
                       skipped_tasks)

    if filepath:
        with open(filepath, "w", encoding="utf-8") as f:
            # Chrome Trace Event 格式: ts/dur 值单位为 us (规范约定);
            # displayTimeUnit 声明查看器 (chrome://tracing 等) 的显示单位,
            # 缺省 ms, 此处指定 ns
            json.dump({"displayTimeUnit": "ns", "traceEvents": events}, f,
                      separators=(",", ":"))

    return events
