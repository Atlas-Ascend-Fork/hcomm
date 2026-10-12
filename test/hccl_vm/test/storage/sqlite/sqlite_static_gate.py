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

# See the License in the specific file packaged with this Software for details.

"""SQLite 后端模块专项静态门禁（随模块目录裁剪；模块被移除时本门禁随之消失）。

检查本模块自身完整性：引擎/工厂/装配源码、Host 与 Device 依赖配方、模块
CMakeLists 的自描述（注册名、默认候选、入口声明）与基线路径/profile 知识归属。
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # 仓库根
MODULE = ROOT / "src/storage_backends/sqlite"


def main():
    failures = []

    required_files = [
        "src/storage_backends/sqlite/CMakeLists.txt",
        "src/storage_backends/sqlite/engine/sqlite_engine.h",
        "src/storage_backends/sqlite/engine/sqlite_engine.cc",
        "src/storage_backends/sqlite/engine/sqlite_backend_factory.h",
        "src/storage_backends/sqlite/composition/sqlite_backend_module.h",
        "src/storage_backends/sqlite/composition/sqlite_backend_module.cc",
        "src/storage_backends/sqlite/cmake/sqlite3_host.cmake",
        "src/storage_backends/sqlite/cmake/sqlite3_device.cmake",
    ]
    for relative in required_files:
        if not (ROOT / relative).is_file():
            failures.append("缺少模块文件: " + relative)

    contracts = {
        "src/storage_backends/sqlite/CMakeLists.txt": [
            r'set\(HCCLVM_STORAGE_BACKEND_NAME "sqlite"\)',
            r"set\(HCCLVM_STORAGE_BACKEND_DEFAULT_CANDIDATE ON\)",
            r"composition/sqlite_backend_module\.h",
            r"sim::backends::SqliteBackendModule",
        ],
        "src/storage_backends/sqlite/composition/sqlite_backend_module.cc": [
            r"data/hccl_sim\.db",
            r"data/hccl_vm_data\.db",
            r"SqliteConnectionProfile\{60000, true, 1000\}",
            r"SqliteConnectionProfile\{10000, false, 200\}",
            r"StorageOpenMode::BACKUP_ONLY",
        ],
        "src/storage_backends/sqlite/cmake/sqlite3_host.cmake": [
            r"sqlite-amalgamation-3510300",
            r"SQLITE_THREADSAFE=1",
        ],
    }
    for relative, patterns in contracts.items():
        path = ROOT / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in patterns:
            if re.search(pattern, text) is None:
                failures.append(relative + ": 缺少模块接线/部署知识 " + pattern)

    if failures:
        print("sqlite backend module gate FAIL:")
        for failure in failures:
            print(" -", failure)
        return 1
    print("sqlite backend module gate PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
