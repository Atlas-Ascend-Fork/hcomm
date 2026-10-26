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

# 文件职责：MariaDB 后端模块自描述（由 src/storage_backends/CMakeLists.txt 收集器
# 按阶段 include；本文件不直接 add_subdirectory，也不被公共构建按名引用）。
# 模块内容：engine/（引擎/工厂/连接 profile/语句缓存）、composition/（部署配置
# 段解析与五操作回调）、cmake/（Connector 获取、校验、Host/Device 构建）、
# config/（部署配置与服务端调优样例）、licenses/（LGPL-2.1 全文）、tests/。
# 本模块不提供无配置默认值：必须显式配置（backend=mariadb）才成为选中后端。
set(HCCLVM_STORAGE_BACKEND_NAME "mariadb")
set(HCCLVM_STORAGE_BACKEND_DEFAULT_CANDIDATE OFF)
set(HCCLVM_STORAGE_BACKEND_ENTRY_HEADER "storage_backends/mariadb/composition/mariadb_backend_module.h")
set(HCCLVM_STORAGE_BACKEND_ENTRY_SYMBOL "sim::backends::MariadbBackendModule")

if(HCCLVM_STORAGE_BACKEND_PHASE STREQUAL "PREPARE")
    include("${CMAKE_CURRENT_LIST_DIR}/cmake/mariadb_host.cmake")
    # device 子构建交接：已验证源码树位置与校验指纹（device 侧独立复核 stamp 与
    # manifest 后按交叉架构独立子构建，不链接宿主产物）。
    list(APPEND HCCLVM_STORAGE_BACKEND_DEVICE_ARGS
        "-DMARIADB_SOURCE_DIR=${MARIADB_SOURCE_DIR}"
        "-DMARIADB_VERSION=${MARIADB_VERSION}"
        "-DMARIADB_SHA256=${MARIADB_SHA256}"
        "-DMARIADB_SOURCE_MANIFEST_SHA256=${MARIADB_SOURCE_MANIFEST_SHA256}")
elseif(HCCLVM_STORAGE_BACKEND_PHASE STREQUAL "ATTACH")
    # 引擎与 connector 依赖接入公共 storage 静态库（构建序：connector 子构建先行，
    # 最终链接产物需要 libmariadb）；composition 源码接入 simulator_storage。
    target_sources(${HCCLVM_STORAGE_BACKEND_ATTACH_TARGET} PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/engine/mariadb_engine.cc")
    target_link_libraries(${HCCLVM_STORAGE_BACKEND_ATTACH_TARGET} PUBLIC mariadb_client)
    add_dependencies(${HCCLVM_STORAGE_BACKEND_ATTACH_TARGET} mariadb_connector)
    target_sources(${HCCLVM_STORAGE_BACKEND_COMPOSITION_TARGET} PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/composition/mariadb_backend_module.cc"
        "${CMAKE_CURRENT_LIST_DIR}/composition/mariadb_deploy_section.cc")
elseif(HCCLVM_STORAGE_BACKEND_PHASE STREQUAL "DEVICE")
    # device 子工程：handoff 复核 + 交叉子构建 + 引擎与 composition 源码接入
    # device 本地 storage 静态库（恒静态，不受 HVM_MARIADB_LINKAGE 影响）。
    include("${CMAKE_CURRENT_LIST_DIR}/cmake/mariadb_device.cmake")
    target_sources(${HCCLVM_STORAGE_BACKEND_DEVICE_TARGET} PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/engine/mariadb_engine.cc"
        "${CMAKE_CURRENT_LIST_DIR}/composition/mariadb_backend_module.cc"
        "${CMAKE_CURRENT_LIST_DIR}/composition/mariadb_deploy_section.cc")
    target_link_libraries(${HCCLVM_STORAGE_BACKEND_DEVICE_TARGET} PUBLIC mariadb_client)
    add_dependencies(${HCCLVM_STORAGE_BACKEND_DEVICE_TARGET} mariadb_connector_device)
endif()
