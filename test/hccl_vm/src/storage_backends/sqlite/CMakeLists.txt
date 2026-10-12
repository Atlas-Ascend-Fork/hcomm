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

# 文件职责：SQLite 后端模块自描述（由 src/storage_backends/CMakeLists.txt 收集器
# 按阶段 include；本文件不直接 add_subdirectory，也不被公共构建按名引用）。
# 模块内容：engine/（引擎与通用工厂）、composition/（默认路由/连接 profile/
# 备份等部署知识）、cmake/（Host 与 Device 的 sqlite3 依赖配方）、tests/。
set(HCCLVM_STORAGE_BACKEND_NAME "sqlite")
set(HCCLVM_STORAGE_BACKEND_DEFAULT_CANDIDATE ON)
set(HCCLVM_STORAGE_BACKEND_ENTRY_HEADER "storage_backends/sqlite/composition/sqlite_backend_module.h")
set(HCCLVM_STORAGE_BACKEND_ENTRY_SYMBOL "sim::backends::SqliteBackendModule")

if(HCCLVM_STORAGE_BACKEND_PHASE STREQUAL "PREPARE")
    include("${CMAKE_CURRENT_LIST_DIR}/cmake/sqlite3_host.cmake")
    # device 子构建经交接变量取得同一份源码缓存（Host prepare 已下载/解压）
    list(APPEND HCCLVM_STORAGE_BACKEND_DEVICE_ARGS
        "-DHCCLVM_SQLITE3_SOURCE_DIR=${HCCLVM_SQLITE3_SOURCE_DIR}")
elseif(HCCLVM_STORAGE_BACKEND_PHASE STREQUAL "ATTACH")
    # 引擎与依赖接入公共 storage 静态库；composition 源码接入 simulator_storage
    # （引擎头/客户库 include 经 sqlite3 PUBLIC 传递，链接方不感知模块路径）。
    target_sources(${HCCLVM_STORAGE_BACKEND_ATTACH_TARGET} PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/engine/sqlite_engine.cc")
    target_link_libraries(${HCCLVM_STORAGE_BACKEND_ATTACH_TARGET} PUBLIC sqlite3)
    target_sources(${HCCLVM_STORAGE_BACKEND_COMPOSITION_TARGET} PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/composition/sqlite_backend_module.cc")
elseif(HCCLVM_STORAGE_BACKEND_PHASE STREQUAL "DEVICE")
    # device 子工程：依赖配方（交叉架构独立编译）+ 引擎与 composition 源码接入
    # device 本地 storage 静态库（与宿主同一份源码，二进制输出分开）。
    include("${CMAKE_CURRENT_LIST_DIR}/cmake/sqlite3_device.cmake")
    target_sources(${HCCLVM_STORAGE_BACKEND_DEVICE_TARGET} PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/engine/sqlite_engine.cc"
        "${CMAKE_CURRENT_LIST_DIR}/composition/sqlite_backend_module.cc")
    target_link_libraries(${HCCLVM_STORAGE_BACKEND_DEVICE_TARGET} PUBLIC sqlite3)
endif()
