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

# SQLite 后端模块的客户库依赖配方（Device 交叉架构）。
# 与宿主同一份源码缓存（Host prepare 经 HCCLVM_SQLITE3_SOURCE_DIR 交接；缺省
# 回退到模块源码缓存路径），在本工程内按交叉工具链重新编译为独立静态库——
# 不链接宿主架构产物，二进制输出分开。源码缺失时明确失败，不从其他工作区或
# 系统库补回。
if(NOT DEFINED HCCLVM_SQLITE3_SOURCE_DIR OR HCCLVM_SQLITE3_SOURCE_DIR STREQUAL "")
    set(HCCLVM_SQLITE3_SOURCE_DIR "${HCCL_VM_SRC}/storage_backends/sqlite/third_party/sqlite")
endif()
if(NOT (EXISTS "${HCCLVM_SQLITE3_SOURCE_DIR}/sqlite3.c" AND EXISTS "${HCCLVM_SQLITE3_SOURCE_DIR}/sqlite3.h"))
    message(FATAL_ERROR
        "[StorageBackends:sqlite] device build requires the Host-prepared SQLite source cache, "
        "but it is missing: ${HCCLVM_SQLITE3_SOURCE_DIR}. Build with the top-level configure "
        "(BUILD_DEVICE_ARM=ON) which prepares the module source cache first.")
endif()
message(STATUS "[StorageBackends:sqlite] device source: ${HCCLVM_SQLITE3_SOURCE_DIR}")

if(NOT TARGET sqlite3)
    # sqlite3 的 SQLITE_THREADSAFE=1 引入 pthread 依赖，用 Threads::Threads 适配
    # glibc 2.34 前后的差异（新版 pthread 已并入 libc，无需显式 -lpthread）。
    find_package(Threads REQUIRED)
    add_library(sqlite3 STATIC "")
    target_sources(sqlite3 PRIVATE
        ${HCCLVM_SQLITE3_SOURCE_DIR}/sqlite3.c
    )
    target_include_directories(sqlite3 PUBLIC
        ${HCCLVM_SQLITE3_SOURCE_DIR}
    )
    set_target_properties(sqlite3 PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )
    target_compile_options(sqlite3 PRIVATE
        -O2
        -DSQLITE_THREADSAFE=1
        -DSQLITE_OMIT_LOAD_EXTENSION
        -DSQLITE_DEFAULT_MEMSTATUS=0
        -DSQLITE_DEFAULT_WAL_SYNCHRONOUS=0
        -DSQLITE_LIKE_DOESNT_MATCH_BLOBS
        -DSQLITE_OMIT_DEPRECATED
    )
    # PUBLIC：pthread 依赖随 sqlite3 传播给 device 等链接者
    target_link_libraries(sqlite3 PUBLIC Threads::Threads)
endif()
