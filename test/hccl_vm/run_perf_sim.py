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

"""light_performance 一键性能仿真入口: db → 任务图 → 仿真 → perf_profiling.json。

pipeline 实现收编于 light_performance/pipeline.py, 与安装树 @estimate 插件入口
(light_performance/main.py, 由 `hccl-vm plugin run @estimate` 触发) 共用同一
实现 (设计文档 3.1 节); 本脚本仅为源码树手动调试的 CLI 包装。

用法 (仓库根目录执行):
    python3 run_perf_sim.py hccl_vm_install/data/hccl_vm_data.db
    python3 run_perf_sim.py <db> -o perf.json --config src/plugin/light_performance/sim_config.json
    python3 run_perf_sim.py <db> --device 1 --iter 0 --trace-dir trace/ --dump-dir graph_dump/

说明:
    - ranktable.json 缺省取 db 同目录 (eid→端口解析, 走网任务必需);
      缺失时该算子跳过仿真 (不产生 trace 事件), 仅输出任务图 dump。
    - --dump-dir 额外输出任务图 (DOT/JSON/HTML); --trace-dir 额外输出单算子
      Chrome tracing 文件 (每算子一个, 格式同 perf_profiling.json)。
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "src", "plugin"))

from light_performance.pipeline import run_perf_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="light_performance 一键仿真: db → 任务图 → 仿真 → "
                    "perf_profiling.json (Chrome tracing 格式, 设计文档 1.1 节)")
    parser.add_argument("db", help="hccl_vm_data.db 路径")
    parser.add_argument("-o", "--out", default="perf_profiling.json",
                        help="输出 json 路径 (默认 perf_profiling.json)")
    parser.add_argument("--config", help="时延/带宽配置 JSON (单位 us/GBps)")
    parser.add_argument("--bw-profile", help="带宽利用率配置 JSON")
    parser.add_argument("--ranktable",
                        help="ranktable.json (默认取 db 同目录)")
    parser.add_argument("--device", type=int, default=None,
                        help="目标 deviceId (默认取 opDetails 首个设备)")
    parser.add_argument("--iter", type=int, default=None,
                        help="仅仿真指定 opIter (默认全部)")
    parser.add_argument("--trace-dir", metavar="DIR",
                        help="额外输出单算子 Chrome tracing (每算子一个文件)")
    parser.add_argument("--dump-dir", metavar="DIR",
                        help="额外输出任务图 dump (DOT/JSON, 每算子)")
    args = parser.parse_args()

    return run_perf_pipeline(
        args.db, args.out,
        config_path=args.config,
        bw_profile_path=args.bw_profile,
        ranktable_path=args.ranktable,
        device=args.device,
        iter_filter=args.iter,
        trace_dir=args.trace_dir,
        dump_dir=args.dump_dir,
    )


if __name__ == "__main__":
    sys.exit(main())
