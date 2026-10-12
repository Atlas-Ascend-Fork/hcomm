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

"""公共存储层静态门禁（后端中立部分）。

只检查公共文件与通用模块化接线：具体后端的专属断言由各模块自己的门禁
（src/storage_backends/<模块名>/tests/）承担，本脚本不无条件要求两个后端
目录存在——裁剪掉任一模块的源码树同样必须通过本门禁。

当前合同：
- 公共存储实现（会话/事务/关系辅助）与 composition（装配/部署配置/模块合同）
  只依赖后端中立接口；
- 后端接入只经 src/storage_backends/ 通用收集器，公共构建文件不出现具体后端名；
- 已删除的 Redis 生产路径不得回流。
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 后端专属标识（include/原生 API/工厂类型）：公共存储实现与公共构建不得出现。
BACKEND_INCLUDE = re.compile(r'#include\s*[<"](sqlite3\.h|mysql\.h|mariadb/|sqlite/)')
BACKEND_HEADER_PATH = re.compile(
    r'#include\s*[<"](storage/engine/(sqlite|mariadb)|storage_backends/)')
BACKEND_API_CALL = re.compile(r'\b(sqlite3|mysql|mariadb)_[A-Za-z0-9_]+\s*\(')
BACKEND_FACTORY = re.compile(r'\b(Create(Sqlite|Mariadb)Engine|SQLiteEngine|MariaDbEngine)\b')
COMMON_CXX_SUFFIXES = (".cc", ".h")


def read(relative):
    return (ROOT / relative).read_text(encoding="utf-8")


def rglob_files(base, suffixes):
    root = ROOT / base
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in suffixes)


def main():
    failures = []

    # ---- 1. 公共存储/composition 实现文件存在 ----
    required_files = [
        "src/storage/engine/i_storage_engine.h",
        "src/storage/engine/i_relational_engine.h",
        "src/storage/engine/relational/relational_sql_helpers.cc",
        "src/storage/internal/storage_runtime_impl.cc",
        "src/storage/internal/session_control.cc",
        "src/storage/internal/process_storage_context.cc",
        "src/storage/session_coordinator.cc",
        "src/storage/storage_session.cc",
        "src/storage/transaction.cc",
        "src/storage_composition/storage_composition.h",
        "src/storage_composition/storage_composition.cc",
        "src/storage_composition/storage_deploy_config.h",
        "src/storage_composition/storage_deploy_config.cc",
        "src/storage_composition/backend_module.h",
        "include/storage/query_spec.h",
    ]
    for relative in required_files:
        if not (ROOT / relative).is_file():
            failures.append("缺少公共存储实现文件: " + relative)

    # ---- 2. 通用模块化接线（不出现具体后端名） ----
    build_contracts = {
        "CMakeLists.txt": [r"include\(\$\{CMAKE_SOURCE_DIR\}/src/storage_backends/CMakeLists\.txt\)",
                           r"hcclvm_storage_backends_prepare\(\)"],
        "src/CMakeLists.txt": [r"add_subdirectory\(storage\)", r"add_subdirectory\(storage_composition\)"],
        "src/storage/CMakeLists.txt": [r"engine/relational/relational_sql_helpers\.cc"],
        "src/storage_composition/CMakeLists.txt": [r"storage_composition\.cc",
                                                   r"storage_deploy_config\.cc",
                                                   r"hcclvm_storage_backends_attach\("],
        "src/device_arm/CMakeLists.txt": [r"storage_composition/storage_composition\.cc",
                                          r"storage_composition/storage_deploy_config\.cc",
                                          r"yaml-cpp-aarch64",
                                          r"hcclvm_storage_backends_attach_device\("],
        "test/storage/CMakeLists.txt": [r"test_storage_contract", r"test_backend_conformance_fake",
                                        r"hcclvm_storage_backends_add_test_dir\("],
        "test/simulation/CMakeLists.txt": [r"hcclvm_storage_backends_add_test_dir\("],
        "test/proxy/CMakeLists.txt": [r"hcclvm_storage_backends_add_test_dir\("],
        "test/store/CMakeLists.txt": [r"hcclvm_storage_backends_add_test_dir\("],
        "test/topo/CMakeLists.txt": [r"hcclvm_storage_backends_add_test_dir\("],
        "test/device_arm/CMakeLists.txt": [r"hcclvm_storage_backends_add_test_dir\("],
        "test/device_arm/proxy/CMakeLists.txt": [r"hcclvm_storage_backends_add_test_dir\("],
        "test/cmd/subcmd/CMakeLists.txt": [r"hcclvm_storage_backends_add_test_dir\("],
    }
    for relative, patterns in build_contracts.items():
        if not (ROOT / relative).is_file():
            failures.append("缺少公共构建文件: " + relative)
            continue
        text = read(relative)
        for pattern in patterns:
            if re.search(pattern, text) is None:
                failures.append(relative + ": 缺少通用模块化接线 " + pattern)

    # ---- 3. 公共层后端中立：实现代码不得包含后端专属 include/原生 API/工厂 ----
    common_sources = []
    for base in ("src/storage", "src/storage_composition", "include/storage"):
        common_sources += rglob_files(base, COMMON_CXX_SUFFIXES)
    for source in common_sources:
        text = source.read_text(encoding="utf-8")
        relative = str(source.relative_to(ROOT))
        for label, pattern in (("后端原生头", BACKEND_INCLUDE),
                               ("后端专属头路径", BACKEND_HEADER_PATH),
                               ("后端原生 API", BACKEND_API_CALL),
                               ("具体引擎/工厂类型", BACKEND_FACTORY)):
            if pattern.search(text) is not None:
                failures.append(relative + ": 公共层出现" + label + "依赖 " + pattern.pattern)
                break

    # composition 是纯装配层：额外要求正文不含具体后端字样（含注释，中性化措辞）
    for base in ("src/storage_composition",):
        for source in rglob_files(base, COMMON_CXX_SUFFIXES):
            text = source.read_text(encoding="utf-8")
            if re.search(r'(?i)\b(sqlite|mariadb|mysql)\b', text) is not None:
                failures.append(
                    str(source.relative_to(ROOT)) + ": composition 公共文件含具体后端字样（应使用模块化措辞）")

    # ---- 4. 公共构建文件不出现具体后端名/客户库直链 ----
    common_build_files = [
        "CMakeLists.txt",
        "src/CMakeLists.txt",
        "src/storage/CMakeLists.txt",
        "src/storage_composition/CMakeLists.txt",
        "src/device_arm/CMakeLists.txt",
        "src/device_arm/level1/CMakeLists.txt",
        "src/device_arm/level2/CMakeLists.txt",
        "src/device_arm/level2/proxy/CMakeLists.txt",
        "src/proxy/CMakeLists.txt",
        "src/proxy/level2/CMakeLists.txt",
        "src/proxy/level2/aiv_kernel/CMakeLists.txt",
        "src/plugin/runner/CMakeLists.txt",
        "src/plugin/checker/CMakeLists.txt",
    ]
    backend_name_pattern = re.compile(r'(?i)\b(sqlite3?|mariadb|mysql)\b')
    for relative in common_build_files:
        path = ROOT / relative
        if not path.is_file():
            continue
        for line_no, line in enumerate(read(relative).splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue  # 注释行不判
            if backend_name_pattern.search(line) is not None:
                failures.append(relative + ":" + str(line_no)
                                + ": 公共构建文件含具体后端名（应经模块收集器接入）")
                break

    # ---- 4b. 公共测试目录不出现后端注册名字面量 ----
    # 引擎专属测试只在 test/<领域>/<模块名>/ 子目录（随模块裁剪）；公共测试里的
    # Fake/内存引擎用测试自有注册名（test_fake）。字面量判定只针对带引号的
    # 注册名（注释里的裸词不算依赖）。
    backend_literal = re.compile(r'"(sqlite|mariadb)"')
    for source in rglob_files("test", (".cc", ".h")):
        relative_path = str(source.relative_to(ROOT)).replace("\\", "/")
        parts = relative_path.split("/")
        if any(part in ("sqlite", "mariadb") for part in parts[:-1]):
            continue  # 引擎专属子目录内允许（随模块整体裁剪）
        text = source.read_text(encoding="utf-8")
        if backend_literal.search(text) is not None:
            failures.append(relative_path + ': 公共测试目录出现后端注册名字面量 '
                            + backend_literal.search(text).group(0)
                            + '（真实 fixture 应迁引擎子目录；Fake 应改测试自有注册名）')

    # ---- 5. 模块目录结构（至少一个模块；不要求具体是哪个） ----
    backends_root = ROOT / "src/storage_backends"
    if not (backends_root / "CMakeLists.txt").is_file():
        failures.append("缺少模块收集器: src/storage_backends/CMakeLists.txt")
    module_dirs = [p for p in sorted(backends_root.iterdir())
                   if p.is_dir() and not p.name.startswith(".")]
    if not module_dirs:
        failures.append("src/storage_backends/ 下没有任何后端模块目录")
    for module_dir in module_dirs:
        if not (module_dir / "CMakeLists.txt").is_file():
            failures.append("后端模块缺少 CMakeLists.txt: " + str(module_dir.relative_to(ROOT)))

    # ---- 6. 两个模块互不依赖（include/链接面；源码树同时保留两个模块时） ----
    for module_dir in module_dirs:
        other = [p for p in module_dirs if p != module_dir]
        for source in rglob_files("src/storage_backends/" + module_dir.name, COMMON_CXX_SUFFIXES):
            text = source.read_text(encoding="utf-8")
            for other_dir in other:
                if re.search(r'#include\s*[<"]storage_backends/' + other_dir.name + r'/', text):
                    failures.append(str(source.relative_to(ROOT))
                                    + ": 模块间依赖（include 了 " + other_dir.name + "）")
        for build_file in module_dir.rglob("CMakeLists.txt"):
            text = build_file.read_text(encoding="utf-8")
            for other_dir in other:
                if re.search(r'(?i)\b' + other_dir.name + r'\b', text) is not None:
                    failures.append(str(build_file.relative_to(ROOT))
                                    + ": 模块构建引用了另一模块 " + other_dir.name)
        for recipe in module_dir.glob("cmake/*.cmake"):
            text = recipe.read_text(encoding="utf-8")
            for other_dir in other:
                if re.search(r'(?i)\b' + other_dir.name + r'\b', text) is not None:
                    failures.append(str(recipe.relative_to(ROOT))
                                    + ": 模块依赖配方引用了另一模块 " + other_dir.name)

    # ---- 7. Redis 生产路径不得回流 ----
    removed_paths = [
        "src/storage/placement/physical_mapping.h",
        "src/storage/engine/redis",
        "config/redis",
        "tools/redis_diagnose.sh",
        "cmake/third_party/hiredis.cmake",
        "cmake/third_party/zlib.cmake",
        "third_party/zlib-aarch64",
        "test/storage/backend_conformance_redis_test.cc",
        "test/storage/fake_redis_server.h",
        "test/storage/redis_engine_config_test.cc",
        "test/storage/redis_transaction_test.cc",
    ]
    for relative in removed_paths:
        if (ROOT / relative).exists():
            failures.append("过期路径仍存在: " + relative)

    forbidden_symbols = [
        r"RedisEngineConfig",
        r"CreateRedisEngine",
        r"StorageBackend::REDIS",
        r"StorageBackend::SQLITE",
        r"StorageBackend::MARIADB",
        r"StorageDeployBackend",
        r"HCCL_VM_DDL_INTERVAL_INDEX",
        r"HCCL_VM_DDL_PARTITION_BY",
        r"LogicalRecord",
        r"hiredis",
    ]
    for source in common_sources:
        text = source.read_text(encoding="utf-8")
        for symbol in forbidden_symbols:
            if re.search(symbol, text) is not None:
                failures.append(
                    str(source.relative_to(ROOT)) + ": 公共源码含已删除/已字符串化的后端枚举符号 " + symbol)

    verifier_files = sorted((ROOT / "test").glob("verify_*.py"))
    if verifier_files:
        failures.append("过期 verifier 未清空: "
                        + ", ".join(str(p.relative_to(ROOT)) for p in verifier_files))

    if failures:
        print("current storage static gate FAIL:")
        for failure in failures:
            print(" -", failure)
        return 1
    print("current storage static gate PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
