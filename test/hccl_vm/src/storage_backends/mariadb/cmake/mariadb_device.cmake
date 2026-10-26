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

# MariaDB 后端模块的客户库依赖配方（Device 交叉架构，静态集成）。
# 父 configure 已验证归档并重建源码缓存；此处复查 identity stamp 与 canonical
# manifest（handoff 复核协议），随后以独立 cmake 子构建按 device 交叉工具链
# 编译静态 libmariadbclient（开关面与宿主一致，见 cmake/mariadb_host.cmake
# 头注释）。device 二进制自包含，恒静态，不向 lib/aarch64 分发预编译
# libmariadb.so.3，也不受 HVM_MARIADB_LINKAGE 影响。
# 交接材料（MARIADB_SOURCE_DIR/VERSION/SHA256/SOURCE_MANIFEST_SHA256）缺失或
# 内容漂移时明确失败，不取消校验、不偷用宿主产物。

foreach(_mariadb_required_variable IN ITEMS
    MARIADB_SOURCE_DIR
    MARIADB_VERSION
    MARIADB_SHA256
    MARIADB_SOURCE_MANIFEST_SHA256
)
    if(NOT DEFINED ${_mariadb_required_variable} OR
        "${${_mariadb_required_variable}}" STREQUAL "")
        message(FATAL_ERROR
            "Verified mariadb handoff is missing ${_mariadb_required_variable}")
    endif()
endforeach()

set(_mariadb_source_stamp "${MARIADB_SOURCE_DIR}/.hcclvm-source.stamp")
set(_mariadb_expected_stamp
    "dependency=mariadb\nversion=${MARIADB_VERSION}\narchive_sha256=${MARIADB_SHA256}\nsource_manifest_sha256=${MARIADB_SOURCE_MANIFEST_SHA256}\n")
if(NOT EXISTS "${_mariadb_source_stamp}")
    message(FATAL_ERROR "Verified mariadb source stamp is missing: ${_mariadb_source_stamp}")
endif()
file(READ "${_mariadb_source_stamp}" _mariadb_actual_stamp)
if(NOT "${_mariadb_actual_stamp}" STREQUAL "${_mariadb_expected_stamp}")
    message(FATAL_ERROR "Verified mariadb source stamp changed before Device configure")
endif()

function(hcclvm_compute_device_mariadb_manifest output_variable)
    file(GLOB_RECURSE _mariadb_manifest_entries
        LIST_DIRECTORIES FALSE
        RELATIVE "${MARIADB_SOURCE_DIR}"
        "${MARIADB_SOURCE_DIR}/*"
    )
    list(SORT _mariadb_manifest_entries)
    set(_mariadb_manifest_payload "")
    foreach(_relative_path IN LISTS _mariadb_manifest_entries)
        if("${_relative_path}" STREQUAL ".hcclvm-source.stamp")
            continue()
        endif()
        set(_absolute_path "${MARIADB_SOURCE_DIR}/${_relative_path}")
        if(IS_SYMLINK "${_absolute_path}")
            file(READ_SYMLINK "${_absolute_path}" _link_target)
            string(APPEND _mariadb_manifest_payload "L\t${_relative_path}\t${_link_target}\n")
        else()
            file(SHA256 "${_absolute_path}" _entry_sha256)
            string(APPEND _mariadb_manifest_payload "F\t${_relative_path}\t${_entry_sha256}\n")
        endif()
    endforeach()
    string(SHA256 _manifest_sha256 "${_mariadb_manifest_payload}")
    set(${output_variable} "${_manifest_sha256}" PARENT_SCOPE)
endfunction()

hcclvm_compute_device_mariadb_manifest(_mariadb_actual_manifest_sha256)
if(NOT "${_mariadb_actual_manifest_sha256}" STREQUAL "${MARIADB_SOURCE_MANIFEST_SHA256}")
    message(FATAL_ERROR "Verified mariadb source manifest changed before Device configure")
endif()

# 交叉子构建：hcc 工具链 + 禁 OpenSSL 发现路径（与已验证的交叉配方一致）
include(ExternalProject)
set(MARIADB_DEVICE_CBUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/mariadb-cbuild")
# my_config.h 等由子构建生成；configure 期预建 include 目录（同宿主）
file(MAKE_DIRECTORY "${MARIADB_DEVICE_CBUILD_DIR}/include")
ExternalProject_Add(mariadb_connector_device
    SOURCE_DIR "${MARIADB_SOURCE_DIR}"
    BINARY_DIR "${MARIADB_DEVICE_CBUILD_DIR}"
    CMAKE_ARGS
        -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
        -DCANN_HCC_ROOT=${CANN_HCC_ROOT}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_SYSTEM_NAME=${CMAKE_SYSTEM_NAME}
        -DCMAKE_SYSTEM_PROCESSOR=aarch64
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_DISABLE_FIND_PACKAGE_OpenSSL=TRUE
        -DWITH_UNIT_TESTS=OFF
        -DWITH_MYSQLCOMPAT=OFF
        -DWITH_REMOTEIO=OFF
        # 变量名必须 CLIENT_PLUGIN_<大写目标名>（见 mariadb_host.cmake 注释）
        -DCLIENT_PLUGIN_CACHING_SHA2_PASSWORD=OFF
        -DCLIENT_PLUGIN_SHA256_PASSWORD=OFF
        -DCLIENT_PLUGIN_CLIENT_ED25519=OFF
        # GSSAPI 探测短路与 CMake 4.x 兼容，机制见 mariadb_host.cmake；
        # /bin/true 于 configure 期在本机执行，交叉工具链下同样成立
        -DKRB5_CONFIG=/bin/true
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5
        -DBUILD_SHARED_LIBS=OFF
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS "${MARIADB_DEVICE_CBUILD_DIR}/libmariadb/libmariadb.a"
    BUILD_ALWAYS ON
)
add_library(mariadb_client STATIC IMPORTED GLOBAL)
set_target_properties(mariadb_client PROPERTIES
    IMPORTED_LOCATION "${MARIADB_DEVICE_CBUILD_DIR}/libmariadb/libmariadb.a"
    INTERFACE_INCLUDE_DIRECTORIES "${MARIADB_SOURCE_DIR}/include;${MARIADB_DEVICE_CBUILD_DIR}/include"
    # 归档成员的 dl/pthread 符号依赖与宿主侧相同（见 mariadb_host.cmake），
    # CMAKE_DL_LIBS 按交叉工具链检测结果取值
    INTERFACE_LINK_LIBRARIES "${CMAKE_DL_LIBS};m;pthread"
)
