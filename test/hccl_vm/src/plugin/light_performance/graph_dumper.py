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

"""任务图 dump 模块: DependencyGraph → 固定格式 JSON + DOT (graphviz) + HTML。

JSON 固定格式 (light_perf.graph_dump/2, 对齐 hccl_sim 新版任务模型):
    meta:     算子元信息 (comm/iter/device/op_type/统计)
    nodes:    任务节点数组 (id/type/stream(逻辑名)/size/jetty_id/eid/... 及仿真时刻)
    streams:  [{id(逻辑名), sid(int 流 id), tasks: [task_id...],
               kind: real/ccu/virtual, virtual: bool, parent_sid(仅虚拟)}]
              串行链 (kind/virtual/parent_sid 为 7.4 节新增附加字段)
    matches:  [[wait_id, record_id], ...]  notify 1:1 配对边
    extra_edges: [[child_id, parent_id], ...]  额外依赖边 (CCU fork/汇接/挂接)

DOT: 每个 stream 一个 cluster 泳道, 链内串行依赖为实线箭头, notify 配对为
record→wait 橙色虚线 (constraint=false), extra 边为 parent→child 蓝色点线
(constraint=false); 虚拟流 cluster 紫色虚线边框 + "(虚拟)" 标注; 节点按
任务类型着色, 可直接用 VSCode Graphviz 插件预览。

HTML 为单文件离线页面 (无外部依赖): 泳道图渲染, 每个 stream 一条泳道,
链内串行依赖画实线箭头, notify 配对画橙色虚线弧 (record → wait), extra
边画蓝色点线弧 (parent → child); 虚拟流泳道带 "虚拟" 徽标 (基于 streams[].kind,
旧数据退回 #v 后缀识别), 鼠标悬停显示节点详情, 支持拖拽/滚轮缩放/自适应。
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass

from .graph_builder import GraphBuildStats
from .simulator.dependency_graph import DependencyGraph
from .simulator.task_model import INVALID, Task

logger = logging.getLogger(__name__)

DUMP_FORMAT = "light_perf.graph_dump/2"

_NODE_COLORS = {
    "Notify_Record": "#43a047",
    "Notify_Wait": "#fb8c00",
    "Reduce_Inline": "#8e24aa",
    "Memcpy": "#78909c",
    "Ub_Inline_Write": "#1e88e5",
    "Ub_Write_Or_Read": "#00acc1",
    "Write_With_Notify": "#5e35b1",
    "Ub_Reduce_Inline": "#3949ab",
    "Reduce_With_Notify": "#8d6e63",
    "AicpuKernel": "#f4511e",
    "Aicpu_Reduce": "#6d4c41",
    "Unknown": "#90a4ae",
}


def _node_dict(task: Task, stream_labels: dict = None) -> dict:
    labels = stream_labels or {}
    d = {
        "id": task.task_id,
        "type": task.display_name,
        "stream": labels.get(task.stream_id, task.stream_id),
    }
    if task.notify_id != INVALID:
        d["notify_id"] = task.notify_id
    if task.mask:
        d["mask"] = task.mask
    if task.size:
        d["size"] = task.size
    if task.jetty_id != INVALID:
        d["jetty_id"] = task.jetty_id
    if task.src_eid:
        d["src_eid"] = task.src_eid
    if task.dst_eid:
        d["dst_eid"] = task.dst_eid
    if task.link_type:
        d["link_type"] = task.link_type
    if task.protocol:
        d["protocol"] = task.protocol
    # 缓冲区类型 (mem/ms): 仅 CCU 转换的 Memcpy/Ub_Write_Or_Read 任务携带
    if task.src_buf_type:
        d["src_buf_type"] = task.src_buf_type
    if task.dst_buf_type:
        d["dst_buf_type"] = task.dst_buf_type
    # 循环合并信息: 仅合并折叠任务 (loop_cnt > 0) 携带, 其余不占字段;
    # loop_cnt = 展开后的 loop 数, loop_iter_cnt = 单个 loop 自循环次数
    if task.loop_cnt > 0:
        d["loop_cnt"] = task.loop_cnt
        d["loop_iter_cnt"] = task.loop_iter_cnt
    # loop 合并标签 (6.5.1 节): 仅合并发射窗口内任务携带, 供特征链识别
    # 与后续性能仿真使用; 其余来源不占字段
    if task.loop_tag:
        d["loop_tag"] = task.loop_tag
    if task.src_rank != INVALID:
        d["src_rank"] = task.src_rank
    if task.dst_rank != INVALID:
        d["dst_rank"] = task.dst_rank
    if task.start_time >= 0:
        d["start_us"] = task.start_time * 1e6
        d["end_us"] = task.end_time * 1e6
    return d


def _stats_dict(stats) -> dict:
    if stats is None:
        return {}
    if is_dataclass(stats):
        return asdict(stats)
    return dict(stats)


def build_dump_dict(dep: DependencyGraph, meta: dict,
                    stats: GraphBuildStats = None,
                    stream_labels: dict = None,
                    stream_meta: dict = None) -> dict:
    """DependencyGraph → 固定格式 dict (可直接 json.dumps)。"""
    labels = stream_labels or {}
    smeta = stream_meta or {}
    streams = []
    for sid in dep.stream_ids:
        m = smeta.get(sid)
        entry = {"id": labels.get(sid, str(sid)), "sid": sid,
                 "tasks": list(dep.get_chain(sid))}
        if m is not None:
            entry["kind"] = m.get("kind")
            entry["virtual"] = m.get("kind") == "virtual"
            if m.get("kind") == "virtual":
                entry["parent_sid"] = m.get("parent_sid")
        streams.append(entry)
    return {
        "format": DUMP_FORMAT,
        "meta": meta,
        "stats": _stats_dict(stats),
        "nodes": [_node_dict(t, labels) for t in dep.tasks.values()],
        "streams": streams,
        "matches": [[w, dep.get_match(w)] for w in dep.matched_wait_ids],
        "extra_edges": [[c, p] for c, ps in dep.extra_edges.items()
                        for p in ps],
    }


def dump_group_json(dep: DependencyGraph, path: str, meta: dict,
                    stats: GraphBuildStats = None,
                    stream_labels: dict = None,
                    stream_meta: dict = None) -> str:
    data = build_dump_dict(dep, meta, stats, stream_labels, stream_meta)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def dump_group_html(dep: DependencyGraph, path: str, meta: dict,
                    stats: GraphBuildStats = None,
                    stream_labels: dict = None,
                    stream_meta: dict = None) -> str:
    data = build_dump_dict(dep, meta, stats, stream_labels, stream_meta)
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_HTML_TEMPLATE.replace("__GRAPH_DATA_JSON__", payload))
    return path


# ----------------------------------------------------------------------
# DOT (graphviz) 输出
# ----------------------------------------------------------------------

def _dot_escape(value) -> str:
    return (str(value).replace("\\", "\\\\").replace('"', '\\"')
            .replace("\n", "\\n").replace("\r", ""))


def _dot_label(parts) -> str:
    return "\\n".join(_dot_escape(p) for p in parts if p not in ("", None))


def _fmt_size(size) -> str:
    if size >= 1e9:
        return f"{size / 1e9:.2f} GB"
    if size >= 1e6:
        return f"{size / 1e6:.2f} MB"
    if size >= 1e3:
        return f"{size / 1e3:.1f} KB"
    return f"{size} B"


def build_dot_source(dep: DependencyGraph, meta: dict = None,
                     stats: GraphBuildStats = None,
                     stream_labels: dict = None,
                     stream_meta: dict = None) -> str:
    """DependencyGraph → DOT 源码字符串。"""
    meta = meta or {}
    labels = stream_labels or {}
    smeta = stream_meta or {}
    virtual_sids = {sid for sid, m in smeta.items()
                    if m.get("kind") == "virtual"}
    lines = [
        "digraph task_graph {",
        "  rankdir=LR;",
        "  newrank=true;",
        '  bgcolor="#f5f6f8";',
        '  graph [fontname="Helvetica", labelloc="t", fontsize=13];',
        "  node [style=filled, shape=box, height=0.42, fontsize=10, "
        'fontcolor=white, fontname="Helvetica", margin="0.14,0.07"];',
        "  edge [arrowsize=0.7, fontsize=9];",
    ]

    chips = [f"任务 {dep.task_count}", f"stream {dep.chain_count}",
             f"notify 配对 {dep.match_count}"]
    if dep.extra_edge_count:
        chips.append(f"fork/join {dep.extra_edge_count}")
    if virtual_sids:
        chips.append(f"虚拟流 {len(virtual_sids)}")
    if meta.get("makespan_us") is not None:
        chips.append(f"makespan {meta['makespan_us']:.1f} us")
    if stats is not None:
        chips.append(f"raw {stats.total_raw}")
    lines.append(f'  label="{_dot_label([meta.get("op_desc", "任务图")] + chips)}";')

    for idx, sid in enumerate(dep.stream_ids):
        chain = dep.get_chain(sid)
        vname = " (虚拟)" if sid in virtual_sids else ""
        lines.append(f"  subgraph cluster_{idx} {{")
        lines.append(f'    label="{_dot_label([labels.get(sid, str(sid)) + vname, f"{len(chain)} 任务"])}";')
        if sid in virtual_sids:
            lines.append('    color="#7e57c2"; style="rounded,dashed"; '
                         'fillcolor="#f3eefb"; fontcolor="#4527a0"; fontsize=11;')
        else:
            lines.append('    color="#cfd8dc"; style="rounded,filled"; '
                         'fillcolor="#eef1f4"; fontcolor="#37474f"; fontsize=11;')
        for tid in chain:
            task = dep.get_task(tid)
            parts = [tid, task.display_name]
            if task.size:
                parts.append(_fmt_size(task.size))
            if task.jetty_id != INVALID:
                parts.append(f"jet{task.jetty_id}")
            color = _NODE_COLORS.get(task.display_name, "#546e7a")
            lines.append(f'    "{_dot_escape(tid)}" '
                         f'[label="{_dot_label(parts)}", fillcolor="{color}"];')
        lines.append("  }")

    for sid in dep.stream_ids:
        chain = dep.get_chain(sid)
        for a, b in zip(chain, chain[1:]):
            lines.append(f'  "{_dot_escape(a)}" -> "{_dot_escape(b)}" '
                         '[color="#90a4ae", penwidth=1.2];')

    for wait_id in dep.matched_wait_ids:
        record_id = dep.get_match(wait_id)
        lines.append(f'  "{_dot_escape(record_id)}" -> "{_dot_escape(wait_id)}" '
                     '[style=dashed, color="#fb8c00", penwidth=1.4, '
                     "constraint=false];")

    for child_id, parents in dep.extra_edges.items():
        for parent_id in parents:
            lines.append(f'  "{_dot_escape(parent_id)}" -> '
                         f'"{_dot_escape(child_id)}" '
                         '[style=dotted, color="#1e88e5", penwidth=1.3, '
                         "constraint=false];")

    seen_types = sorted({t.display_name for t in dep.tasks.values()})
    if seen_types:
        lines.append("  subgraph cluster_legend {")
        lines.append('    label="图例"; color="#cfd8dc"; '
                      'style="rounded,filled"; fillcolor="#fafafa"; '
                      'fontcolor="#37474f"; fontsize=11;')
        for t in seen_types:
            lines.append(f'    "legend:{t}" [label="{t}", '
                         f'fillcolor="{_NODE_COLORS.get(t, "#546e7a")}"];')
        lines.append("  }")
        lines.append('  "legend:串行依赖" [label="串行依赖 (链内顺序)", '
                     'fillcolor="#90a4ae"];')
        lines.append('  "legend:notify配对" [label="notify 配对 '
                     '(record → wait)", fillcolor="#fb8c00"];')
        if dep.extra_edge_count:
            lines.append('  "legend:forkjoin" [label="fork/join 边 '
                         '(parent → child, 虚拟流)", fillcolor="#1e88e5"];')

    lines.append("}")
    return "\n".join(lines) + "\n"


def dump_group_dot(dep: DependencyGraph, path: str, meta: dict = None,
                   stats: GraphBuildStats = None,
                   stream_labels: dict = None,
                   stream_meta: dict = None) -> str:
    with open(path, "w", encoding="utf-8") as f:
        f.write(build_dot_source(dep, meta, stats, stream_labels, stream_meta))
    return path


def dump_group(dep: DependencyGraph, out_prefix: str, meta: dict,
               stats: GraphBuildStats = None, stream_labels: dict = None,
               stream_meta: dict = None, html: bool = False) -> list:
    """输出 <prefix>.dot + <prefix>.json (可选 <prefix>.html), 返回路径列表。"""
    paths = [dump_group_dot(dep, f"{out_prefix}.dot", meta, stats,
                            stream_labels, stream_meta),
             dump_group_json(dep, f"{out_prefix}.json", meta, stats,
                             stream_labels, stream_meta)]
    if html:
        paths.append(dump_group_html(dep, f"{out_prefix}.html", meta, stats,
                                     stream_labels, stream_meta))
    return paths


# ----------------------------------------------------------------------
# 离线 HTML 模板 (vanilla JS/SVG, 泳道式 DAG 渲染)
# ----------------------------------------------------------------------

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>HCCL 任务依赖图</title>
<style>
  html, body { margin: 0; padding: 0; height: 100%; background: #f5f6f8;
               font-family: "Helvetica Neue", Arial, "PingFang SC", sans-serif; }
  #topbar { position: fixed; top: 0; left: 0; right: 0; z-index: 10;
            background: #ffffff; border-bottom: 1px solid #e0e0e0;
            padding: 8px 16px; display: flex; gap: 18px; align-items: center;
            flex-wrap: wrap; box-shadow: 0 1px 4px rgba(0,0,0,.08); }
  #topbar .title { font-size: 15px; font-weight: 600; color: #263238; }
  #topbar .chip { font-size: 12px; color: #546e7a; background: #eceff1;
                  border-radius: 10px; padding: 2px 10px; }
  #topbar button { font-size: 12px; border: 1px solid #b0bec5; background: #fff;
                   border-radius: 4px; padding: 3px 12px; cursor: pointer; }
  #topbar button:hover { background: #eceff1; }
  #legend { display: flex; gap: 12px; align-items: center; font-size: 11px;
            color: #455a64; }
  #legend .sw { display: inline-block; width: 10px; height: 10px;
                border-radius: 2px; margin-right: 4px; vertical-align: -1px; }
  #viewport { position: absolute; top: 52px; bottom: 0; left: 0; right: 0;
              overflow: hidden; cursor: grab; }
  #viewport.dragging { cursor: grabbing; }
  svg text { user-select: none; }
  .node rect { stroke: rgba(0,0,0,.35); stroke-width: 1; cursor: pointer; }
  .node:hover rect { stroke: #000; stroke-width: 2; }
  .edge-serial { stroke: #78909c; stroke-width: 1.4; fill: none; }
  .edge-match { stroke: #fb8c00; stroke-width: 1.6; fill: none;
                stroke-dasharray: 6 4; }
  .edge-extra { stroke: #1e88e5; stroke-width: 1.4; fill: none;
                stroke-dasharray: 2 3; }
  .edge.hl { stroke: #d81b60; stroke-width: 3; }
  #tooltip { position: fixed; z-index: 20; display: none; pointer-events: none;
             background: #263238; color: #fff; font-size: 12px; line-height: 1.6;
             border-radius: 6px; padding: 8px 12px; max-width: 380px;
             white-space: pre; box-shadow: 0 2px 10px rgba(0,0,0,.35); }
  .lane-label { font-size: 12px; font-weight: 600; fill: #37474f; }
  .lane-badge { fill: #8e24aa; }
  .lane-badge-v { fill: #1e88e5; }
  .lane-badge-text { font-size: 9px; fill: #fff; font-weight: 600; }
  .node-label { font-size: 10px; fill: #fff; font-weight: 600;
                text-anchor: middle; pointer-events: none; }
  .node-sub { font-size: 8.5px; fill: #eceff1; text-anchor: middle;
              pointer-events: none; }
</style>
</head>
<body>
<div id="topbar">
  <span class="title">HCCL 任务依赖图</span>
  <span id="chips"></span>
  <div id="legend"></div>
  <button onclick="fitView()">自适应</button>
  <button onclick="resetView()">重置</button>
</div>
<div id="viewport"><svg id="canvas" width="10" height="10"></svg></div>
<div id="tooltip"></div>
<script type="application/json" id="graph-data">__GRAPH_DATA_JSON__</script>
<script>
const DATA = JSON.parse(document.getElementById('graph-data').textContent);
const COLOR = __NODE_COLOR_MAP__;

const NODE_W = 148, NODE_H = 40, GAP_X = 54, LANE_H = 96, MARGIN_L = 150,
      MARGIN_T = 30;

function laneSort(a, b) {
  const ca = a.id.startsWith('s') && !a.id.startsWith('ccu_');
  const cb = b.id.startsWith('s') && !b.id.startsWith('ccu_');
  if (ca !== cb) return ca ? -1 : 1;
  if (ca) return (parseInt(a.id.slice(1)) || 0) - (parseInt(b.id.slice(1)) || 0)
             || a.id.localeCompare(b.id);
  return a.id.localeCompare(b.id);
}

const streams = DATA.streams.slice().sort(laneSort);
const nodesById = {};
DATA.nodes.forEach(n => nodesById[n.id] = n);

const laneOf = {}, posOf = {};
streams.forEach((s, li) => {
  laneOf[s.id] = li;
  s.tasks.forEach((tid, ci) => {
    posOf[tid] = { x: MARGIN_L + ci * (NODE_W + GAP_X),
                   y: MARGIN_T + li * LANE_H + (LANE_H - NODE_H) / 2 };
  });
});

const maxCols = Math.max(1, ...streams.map(s => s.tasks.length));
const W = MARGIN_L + maxCols * (NODE_W + GAP_X) + 40;
const H = MARGIN_T + streams.length * LANE_H + 30;

const svg = document.getElementById('canvas');
const NS = 'http://www.w3.org/2000/svg';
svg.setAttribute('width', W); svg.setAttribute('height', H);
const root = document.createElementNS(NS, 'g');
svg.appendChild(root);

function el(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  (parent || root).appendChild(e);
  return e;
}

function fmtSize(b) {
  if (b == null) return '';
  if (b >= 1e9) return (b / 1e9).toFixed(2) + ' GB';
  if (b >= 1e6) return (b / 1e6).toFixed(2) + ' MB';
  if (b >= 1e3) return (b / 1e3).toFixed(1) + ' KB';
  return b + ' B';
}

// defs: 箭头 marker
const defs = el('defs', {});
['serial', 'match', 'extra'].forEach(kind => {
  const m = el('marker', { id: 'arrow-' + kind, viewBox: '0 0 10 10',
    refX: 9, refY: 5, markerWidth: 7, markerHeight: 7,
    orient: 'auto-start-reverse' }, defs);
  const fill = kind === 'match' ? '#fb8c00'
             : kind === 'extra' ? '#1e88e5' : '#78909c';
  el('path', { d: 'M 0 0 L 10 5 L 0 10 z', fill }, m);
});

// 泳道背景 + 标签
streams.forEach((s, li) => {
  const y = MARGIN_T + li * LANE_H;
  el('rect', { x: 0, y: y, width: W, height: LANE_H,
    fill: li % 2 ? '#eceff1' : '#e3e7ea' });
  el('text', { x: 10, y: y + LANE_H / 2 - 6, class: 'lane-label' })
    .textContent = s.id;
  el('text', { x: 10, y: y + LANE_H / 2 + 12, 'font-size': 10, fill: '#78909c' })
    .textContent = s.tasks.length + ' 任务';
  // 虚拟流徽标 (kind 字段优先, 旧数据退回 #v 后缀识别)
  if (s.kind === 'virtual' || /#v\d+$/.test(s.id)) {
    el('rect', { x: 10, y: y + LANE_H / 2 + 20, width: 32, height: 14,
      rx: 3, class: 'lane-badge-v' });
    el('text', { x: 26, y: y + LANE_H / 2 + 30, class: 'lane-badge-text',
      'text-anchor': 'middle' }).textContent = '虚拟';
  } else if (s.id.startsWith('ccu_')) {
    el('rect', { x: 10, y: y + LANE_H / 2 + 20, width: 36, height: 14,
      rx: 3, class: 'lane-badge' });
    el('text', { x: 28, y: y + LANE_H / 2 + 30, class: 'lane-badge-text',
      'text-anchor': 'middle' }).textContent = 'CCU';
  }
});

// 串行边 (链内)
streams.forEach(s => {
  for (let i = 1; i < s.tasks.length; i++) {
    const a = posOf[s.tasks[i - 1]], b = posOf[s.tasks[i]];
    if (!a || !b) continue;
    const e = el('path', { class: 'edge edge-serial',
      d: `M ${a.x + NODE_W} ${a.y + NODE_H / 2} L ${b.x - 4} ${b.y + NODE_H / 2}`,
      'marker-end': 'url(#arrow-serial)',
      'data-from': s.tasks[i - 1], 'data-to': s.tasks[i] });
  }
});

// 弧线边绘制 (match: record→wait / extra: parent→child)
function arc(fromId, toId, cls, marker) {
  const a = posOf[fromId], b = posOf[toId];
  if (!a || !b) return;
  const x1 = a.x + NODE_W / 2, y1 = a.y + (a.y < b.y ? NODE_H : 0);
  const x2 = b.x + NODE_W / 2, y2 = b.y + (a.y < b.y ? 0 : NODE_H);
  const lift = Math.min(70, Math.abs(y1 - y2) * 0.6 + 24);
  const c1 = y1 + (a.y < b.y ? lift : -lift), c2 = y2 + (a.y < b.y ? -lift : lift);
  el('path', { class: 'edge ' + cls,
    d: `M ${x1} ${y1} C ${x1} ${c1}, ${x2} ${c2}, ${x2} ${y2 - (a.y < b.y ? 4 : -4)}`,
    'marker-end': 'url(#arrow-' + marker + ')',
    'data-from': fromId, 'data-to': toId });
}

// notify 配对边 (record → wait, 弧线)
(DATA.matches || []).forEach(([waitId, recId]) =>
  arc(recId, waitId, 'edge-match', 'match'));

// extra 边 (parent → child, CCU fork/join, 蓝色点线弧)
(DATA.extra_edges || []).forEach(([childId, parentId]) =>
  arc(parentId, childId, 'edge-extra', 'extra'));

// 节点
const tooltip = document.getElementById('tooltip');
DATA.nodes.forEach(n => {
  const p = posOf[n.id];
  if (!p) return;
  const g = el('g', { class: 'node', transform: `translate(${p.x},${p.y})` });
  el('rect', { width: NODE_W, height: NODE_H, rx: 6,
    fill: COLOR[n.type] || '#546e7a' }, g);
  const shortId = n.id.length > 20 ? n.id.slice(0, 19) + '…' : n.id;
  el('text', { x: NODE_W / 2, y: 16, class: 'node-label' }, g)
    .textContent = shortId;
  let sub = n.type;
  if (n.size) sub += ' · ' + fmtSize(n.size);
  if (n.jetty_id != null) sub += ' · jet' + n.jetty_id;
  el('text', { x: NODE_W / 2, y: 30, class: 'node-sub' }, g).textContent = sub;

  g.addEventListener('mousemove', ev => {
    const lines = [`id:      ${n.id}`, `type:    ${n.type}`,
                   `stream:  ${n.stream}`];
    if (n.notify_id != null) lines.push(`notify:  ${n.notify_id}`);
    if (n.mask != null) lines.push(`mask:    0x${n.mask.toString(16)}`);
    if (n.size != null) lines.push(`size:    ${fmtSize(n.size)}`);
    if (n.jetty_id != null) lines.push(`jetty:   ${n.jetty_id}`);
    if (n.link_type) lines.push(`link:    ${n.link_type}`);
    if (n.protocol) lines.push(`proto:   ${n.protocol}`);
    if (n.src_buf_type) lines.push(`src_buf: ${n.src_buf_type}`);
    if (n.dst_buf_type) lines.push(`dst_buf: ${n.dst_buf_type}`);
    if (n.loop_cnt != null) lines.push(`loop_cnt: ${n.loop_cnt}`);
    if (n.loop_iter_cnt != null)
      lines.push(`loop_iter: ${n.loop_iter_cnt}`);
    if (n.loop_tag) lines.push(`loop:    ${n.loop_tag}`);
    if (n.src_eid) lines.push(`src_eid: ${n.src_eid}`);
    if (n.dst_eid) lines.push(`dst_eid: ${n.dst_eid}`);
    if (n.src_rank != null) lines.push(`src:     rank ${n.src_rank}`);
    if (n.dst_rank != null) lines.push(`dst:     rank ${n.dst_rank}`);
    if (n.start_us != null)
      lines.push(`time:    ${n.start_us.toFixed(1)} ~ ${n.end_us.toFixed(1)} us`
                 + `  (dur ${(n.end_us - n.start_us).toFixed(1)} us)`);
    tooltip.textContent = lines.join('\n');
    tooltip.style.display = 'block';
    tooltip.style.left = Math.min(ev.clientX + 16,
      window.innerWidth - 400) + 'px';
    tooltip.style.top = (ev.clientY + 14) + 'px';
    highlight(n.id, true);
  });
  g.addEventListener('mouseleave', () => {
    tooltip.style.display = 'none';
    highlight(n.id, false);
  });
});

function highlight(id, on) {
  root.querySelectorAll('.edge').forEach(e => {
    if (e.getAttribute('data-from') === id || e.getAttribute('data-to') === id)
      e.classList.toggle('hl', on);
  });
}

// ---- 平移 / 缩放 ----
let scale = 1, tx = 0, ty = 0;
function apply() {
  root.setAttribute('transform', `translate(${tx},${ty}) scale(${scale})`);
}
const viewport = document.getElementById('viewport');
let drag = null;
viewport.addEventListener('mousedown', ev => {
  drag = { x: ev.clientX, y: ev.clientY, tx, ty };
  viewport.classList.add('dragging');
});
window.addEventListener('mousemove', ev => {
  if (!drag) return;
  tx = drag.tx + ev.clientX - drag.x;
  ty = drag.ty + ev.clientY - drag.y;
  apply();
});
window.addEventListener('mouseup', () => {
  drag = null;
  viewport.classList.remove('dragging');
});
viewport.addEventListener('wheel', ev => {
  ev.preventDefault();
  const k = ev.deltaY < 0 ? 1.15 : 1 / 1.15;
  const ns = Math.min(4, Math.max(0.08, scale * k));
  const rect = viewport.getBoundingClientRect();
  const cx = ev.clientX - rect.left, cy = ev.clientY - rect.top;
  tx = cx - (cx - tx) * (ns / scale);
  ty = cy - (cy - ty) * (ns / scale);
  scale = ns;
  apply();
}, { passive: false });

function fitView() {
  const vw = viewport.clientWidth, vh = viewport.clientHeight;
  scale = Math.min(vw / W, vh / H) * 0.96;
  scale = Math.min(1.5, Math.max(0.05, scale));
  tx = (vw - W * scale) / 2;
  ty = (vh - H * scale) / 2;
  apply();
}
function resetView() { scale = 1; tx = 0; ty = 0; apply(); }

// ---- 顶栏 ----
const chips = document.getElementById('chips');
const m = DATA.meta || {};
[[ 'op', m.op_desc || '' ], [ '任务', DATA.nodes.length ],
 [ 'stream', streams.length ],
 [ '配对', (DATA.matches || []).length ],
 [ 'fork/join', (DATA.extra_edges || []).length ]].forEach(([k, v]) => {
   if (v === '' || v == null || v === 0) return;
   const s = document.createElement('span');
   s.className = 'chip';
   s.textContent = `${k}: ${v}`;
   chips.appendChild(s);
});
if (m.makespan_us != null) {
  const s = document.createElement('span');
  s.className = 'chip';
  s.textContent = `makespan: ${Number(m.makespan_us).toFixed(1)} us`;
  chips.appendChild(s);
}
const legend = document.getElementById('legend');
const seen = new Set(DATA.nodes.map(n => n.type));
Object.entries(COLOR).forEach(([t, c]) => {
  if (!seen.has(t)) return;
  const item = document.createElement('span');
  item.innerHTML = `<span class="sw" style="background:${c}"></span>${t}`;
  legend.appendChild(item);
});
const lgSerial = document.createElement('span');
lgSerial.innerHTML = '<svg width="22" height="8"><line x1="0" y1="4" x2="22" ' +
  'y2="4" stroke="#78909c" stroke-width="1.6"/></svg>串行依赖';
legend.appendChild(lgSerial);
const lgMatch = document.createElement('span');
lgMatch.innerHTML = '<svg width="22" height="8"><line x1="0" y1="4" x2="22" ' +
  'y2="4" stroke="#fb8c00" stroke-width="1.6" stroke-dasharray="5 3"/></svg>' +
  'notify 配对 (record→wait)';
legend.appendChild(lgMatch);
if ((DATA.extra_edges || []).length) {
  const lgExtra = document.createElement('span');
  lgExtra.innerHTML = '<svg width="22" height="8"><line x1="0" y1="4" x2="22" ' +
    'y2="4" stroke="#1e88e5" stroke-width="1.6" stroke-dasharray="2 3"/></svg>' +
    'fork/join (虚拟流)';
  legend.appendChild(lgExtra);
}

fitView();
</script>
</body>
</html>
"""


def _render_template() -> str:
    return _HTML_TEMPLATE.replace("__NODE_COLOR_MAP__",
                                   json.dumps(_NODE_COLORS))


_HTML_TEMPLATE = _render_template()
