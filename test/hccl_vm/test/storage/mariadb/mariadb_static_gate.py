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

"""MariaDB 后端模块专项静态门禁（随模块目录裁剪；模块被移除时本门禁随之消失）。

检查本模块自身完整性：引擎/工厂/profile 源码、verified-source 版本与双哈希、
Host/Device 依赖配方、LGPL 许可材料与部署配置样例。
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # 仓库根
MODULE = ROOT / "src/storage_backends/mariadb"


def main():
    failures = []

    required_files = [
        "src/storage_backends/mariadb/CMakeLists.txt",
        "src/storage_backends/mariadb/engine/mariadb_engine.h",
        "src/storage_backends/mariadb/engine/mariadb_engine.cc",
        "src/storage_backends/mariadb/engine/mariadb_backend_factory.h",
        "src/storage_backends/mariadb/engine/mariadb_connection_profile.h",
        "src/storage_backends/mariadb/engine/mariadb_statement_cache.h",
        "src/storage_backends/mariadb/composition/mariadb_backend_module.h",
        "src/storage_backends/mariadb/composition/mariadb_backend_module.cc",
        "src/storage_backends/mariadb/composition/mariadb_deploy_section.h",
        "src/storage_backends/mariadb/composition/mariadb_deploy_section.cc",
        "src/storage_backends/mariadb/cmake/mariadb_host.cmake",
        "src/storage_backends/mariadb/cmake/mariadb_device.cmake",
        "src/storage_backends/mariadb/cmake/verified_source.cmake",
        "src/storage_backends/mariadb/config/storage_mariadb.yaml.example",
        "src/storage_backends/mariadb/config/z-hccl-vm.cnf.example",
        "src/storage_backends/mariadb/licenses/LGPL-2.1.txt",
    ]
    for relative in required_files:
        if not (ROOT / relative).is_file():
            failures.append("缺少模块文件: " + relative)

    contracts = {
        "src/storage_backends/mariadb/CMakeLists.txt": [
            r'set\(HCCLVM_STORAGE_BACKEND_NAME "mariadb"\)',
            r"set\(HCCLVM_STORAGE_BACKEND_DEFAULT_CANDIDATE OFF\)",
            r"composition/mariadb_backend_module\.h",
            r"sim::backends::MariadbBackendModule",
        ],
        "src/storage_backends/mariadb/cmake/mariadb_host.cmake": [
            r'set\(MARIADB_VERSION "3\.3\.5"\)',
            r"c0fda1fa6e52dc85de27156cd847088a72d40d9de6514f7efa57c8d93134a54c",
            r"f7db6810ea82bcdcd11babafae2b7d9c02ca8fec0c3200bcb33c3ce48196a9d1",
            r"hcclvm_prepare_verified_source",
            r"CLIENT_PLUGIN_CACHING_SHA2_PASSWORD=OFF",
        ],
        "src/storage_backends/mariadb/cmake/mariadb_device.cmake": [
            r"MARIADB_SOURCE_MANIFEST_SHA256",
            r"mariadb_connector_device",
            r"BUILD_SHARED_LIBS=OFF",
        ],
        "src/storage_backends/mariadb/composition/mariadb_deploy_section.cc": [
            r"storage\.mariadb\.uri is required",
            r"connect_timeout_ms",
            r"wire_protocol_compression",
        ],
    }
    for relative, patterns in contracts.items():
        path = ROOT / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in patterns:
            if re.search(pattern, text) is None:
                failures.append(relative + ": 缺少模块接线/校验规则 " + pattern)

    if failures:
        print("mariadb backend module gate FAIL:")
        for failure in failures:
            print(" -", failure)
        return 1
    print("mariadb backend module gate PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
