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

"""@estimate 性能仿真插件一次性执行入口 (设计文档 3.1/3.2 节)。

由宿主 `hccl-vm plugin run @estimate` 触发 (cmd_base_utils.cc 的 RunUserPlugin
对 @estimate 特判, 仅 estimate 模式可用):
    cd <installRoot>/plugin/light_performance && python3 main.py

流程 (pipeline 与源码树 run_perf_sim.py 共用, light_performance/pipeline.py):
    manifest setting 解析 → db 读取 (第 4 章) → 任务图 (第 5、6 章)
    → 仿真 (第 7 章) → data/<时间戳>/perf_profiling.json (第 8 章)

路径约定 (setting 内相对路径均以插件目录 plugin/light_performance 为基准解析;
插件目录即 light_performance 包目录, 包经其父目录 plugin/ 导入):
    db          缺省 <installRoot>/data/hccl_vm_data.db
    ranktable   缺省 db 同目录 ranktable.json (与宿主 InitHvmCommEnv 写入位置一致)
    sim_config / bw_profile   与 manifest 同目录 (随包一并 CMake 安装)
    输出        out_dir/<运行时间戳>/ (8.1 节; 一次性通道不传递 data_id)
"""

from __future__ import annotations

import json
import os
import sys
import time

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(PLUGIN_DIR))

from light_performance.pipeline import run_perf_pipeline

MANIFEST_PATH = os.path.join(PLUGIN_DIR, "manifest.json")


def _resolve(path):
    """setting 内相对路径以插件目录为基准解析; 空值返回 None。"""
    if not path:
        return None
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(PLUGIN_DIR, path))


def _load_setting():
    """读取 manifest.json 的 setting 段; 异常时打印错误并返回 None。"""
    try:
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"错误: 解析 manifest.json 失败: {MANIFEST_PATH} ({e})")
        return None
    setting = manifest.get("setting") or {}
    return setting


class _Tee:
    """stdout 双写: 宿主控制台 + 运行日志 (8.1 节 light_performance.log)。"""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()


def main():
    setting = _load_setting()
    if setting is None:
        return 1

    # 输出目录: <插件目录>/<out_dir>/<时间戳>/ (8.1 节)
    out_root = _resolve(setting.get("out_dir") or "data")
    out_dir = os.path.join(out_root, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    # 运行日志与 pipeline 输入 (stdout 与 logging 所在的 stderr 一并落盘)
    log_path = os.path.join(out_dir, "light_performance.log")
    with open(log_path, "w", encoding="utf-8") as log_file:
        stdout_backup = sys.stdout
        stderr_backup = sys.stderr
        sys.stdout = _Tee(stdout_backup, log_file)
        sys.stderr = _Tee(stderr_backup, log_file)
        try:
            if not setting.get("per_op_mode", True):
                print("警告: 暂不支持全量(all-in-one)模式, 按逐算子模式执行")

            db_path = (_resolve(setting.get("db"))
                       or os.path.join(os.path.dirname(os.path.dirname(
                           PLUGIN_DIR)), "data", "hccl_vm_data.db"))
            config_path = _resolve(setting.get("sim_config"))
            bw_profile_path = _resolve(setting.get("bw_profile"))
            ranktable_path = _resolve(setting.get("ranktable"))
            device = setting.get("device")
            iter_filter = setting.get("iter")

            print(f"@estimate 插件启动: db={db_path}")
            print(f"输出目录: {out_dir}")
            ret = run_perf_pipeline(
                db_path,
                os.path.join(out_dir, "perf_profiling.json"),
                config_path=config_path,
                bw_profile_path=bw_profile_path,
                ranktable_path=ranktable_path,
                device=device,
                iter_filter=iter_filter,
                trace_dir=out_dir if setting.get("enable_trace", True)
                         else None,
                dump_dir=out_dir if setting.get("dump_graph", False) else None,
                max_trace_events=int(setting.get("max_trace_events", 0) or 0),
            )
        except Exception:
            # 异常须在 Tee 还原前落盘, 否则 traceback 只进控制台,
            # light_performance.log 无失败原因 (宿主日志亦不捕获子进程输出)
            import traceback
            traceback.print_exc()
            ret = 1
        finally:
            sys.stdout = stdout_backup
            sys.stderr = stderr_backup
    return ret


if __name__ == "__main__":
    sys.exit(main())
