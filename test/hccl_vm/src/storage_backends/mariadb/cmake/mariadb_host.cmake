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

# 文件职责：MariaDB 后端模块的客户库依赖配方（Host 架构，源码 vendor 化）。
# MariaDB Connector/C（libmariadb）源码经 verified-source 流程下载/校验/解包，
# 由本模块独立 configure+build 为静态库（默认）或共享库（发布变体），不要求
# 部署环境安装 libmariadb-dev。MariaDB Server 本体仍由部署人员独立安装运行
# （本仓不构建/不集成/不分发 Server）。
# 源码缓存归属模块自身（src/storage_backends/mariadb/third_party/，git 忽略）；
# device 子构建经交接变量（MARIADB_*，见本模块 CMakeLists 的 device 透传）
# 复用同一份已验证源码并独立交叉构建（见 cmake/mariadb_device.cmake）。
# 链接形态由 HVM_MARIADB_LINKAGE（static|shared，默认 static）控制，仅供发布
# 流水线使用：二进制分发形态必须用 shared + 随包 libmariadb.so.3 + LGPL 四件套
# （源码形态客户自编，静态链接对源码接收者天然合规）。
include_guard(GLOBAL)

# verified-source 公共函数随本模块提供（归档下载/SHA-256 校验/内容指纹复核/
# 解包重建）；缺失时明确失败。
if(NOT COMMAND hcclvm_prepare_verified_source)
    include("${CMAKE_CURRENT_LIST_DIR}/verified_source.cmake")
endif()
if(NOT COMMAND hcclvm_prepare_verified_source)
    message(FATAL_ERROR
        "[StorageBackends:mariadb] verified_source.cmake did not define hcclvm_prepare_verified_source")
endif()

# 链接形态：命令行 -D 优先；环境变量 HVM_MARIADB_LINKAGE 作为缺省（沿用
# 迁移前 build.sh 的取值序），最终缺省 static。取值只允许 static|shared。
if(DEFINED ENV{HVM_MARIADB_LINKAGE})
    set(HVM_MARIADB_LINKAGE_DEFAULT "$ENV{HVM_MARIADB_LINKAGE}")
else()
    set(HVM_MARIADB_LINKAGE_DEFAULT "static")
endif()
set(HVM_MARIADB_LINKAGE "${HVM_MARIADB_LINKAGE_DEFAULT}"
    CACHE STRING "MariaDB client linkage for release packaging (static|shared)")
if(NOT HVM_MARIADB_LINKAGE MATCHES "^(static|shared)$")
    message(FATAL_ERROR
        "[StorageBackends:mariadb] HVM_MARIADB_LINKAGE must be static or shared (got: ${HVM_MARIADB_LINKAGE})")
endif()

set(MARIADB_VERSION "3.3.8")
set(MARIADB_PROVENANCE_URL "https://gitee.com/src-openeuler/mariadb-connector-c/tree/openEuler-24.03-LTS-SP2")
set(MARIADB_URL "https://gitee.com/src-openeuler/mariadb-connector-c/raw/openEuler-24.03-LTS-SP2/mariadb-connector-c-3.3.8-src.tar.gz")
set(MARIADB_SHA256 "f9f076b4aa9fb22cc94b24f82c80f9ef063805ecd6533a2eb5d5060cf93833e8")
set(MARIADB_SOURCE_MANIFEST_SHA256 "58efcf2b5a3440432b54d5d96ad915ef2c32ca8d5bf69dc49e5beaf6057b04f9")
set(MARIADB_MODULE_THIRD_PARTY_DIR "${CMAKE_CURRENT_LIST_DIR}/../third_party")
set(MARIADB_ARCHIVE "${MARIADB_MODULE_THIRD_PARTY_DIR}/mariadb-connector-c-${MARIADB_VERSION}-src.tar.gz")
set(MARIADB_SOURCE_DIR "${MARIADB_MODULE_THIRD_PARTY_DIR}/mariadb-connector-c-${MARIADB_VERSION}")
set(MARIADB_SOURCE_STAMP "${MARIADB_SOURCE_DIR}/.hcclvm-source.stamp")

hcclvm_prepare_verified_source(
    "mariadb" "${MARIADB_VERSION}" "${MARIADB_PROVENANCE_URL}" "${MARIADB_URL}"
    "${MARIADB_SHA256}" "${MARIADB_SOURCE_MANIFEST_SHA256}"
    "${MARIADB_ARCHIVE}" "${MARIADB_SOURCE_DIR}" ".hcclvm-source.stamp"
)

# 上游许可原文（LGPL-2.1）随 verified 源码树自带（COPYING.LIB），安装落位
# share/licenses/hccl-vm/，保证许可文本与实际链接的源码严格对应（LGPL 四件套
# 之二）。许可全文另随模块分发：src/storage_backends/mariadb/licenses/。
# 模块部署配置样例（部署 YAML / 服务端调优 cnf）随模块安装到安装包 config/
#（裁剪掉本模块的源码树不安装这些样例）。
if(NOT BUILD_DEVICE_ARM)
    install(FILES "${MARIADB_SOURCE_DIR}/COPYING.LIB"
        DESTINATION "share/licenses/hccl-vm"
        RENAME "mariadb-connector-c-${MARIADB_VERSION}-COPYING.LIB"
    )
    install(FILES "${CMAKE_CURRENT_LIST_DIR}/../config/storage_mariadb.yaml.example"
        DESTINATION "config"
    )
    install(FILES "${CMAKE_CURRENT_LIST_DIR}/../config/z-hccl-vm.cnf.example"
        DESTINATION "config/mariadb"
    )
endif()
set(MARIADB_CBUILD_DIR "${CMAKE_BINARY_DIR}/mariadb-connector-c-build")
set(MARIADB_STATIC_LIBRARY "${MARIADB_CBUILD_DIR}/libmariadb/libmariadb.a")
set(MARIADB_SHARED_LIBRARY "${MARIADB_CBUILD_DIR}/libmariadb/libmariadb.so.3")

# ---- 上游固定开关面（与大规模验证期间的交叉构建配方一致）----
#   不显式传 WITH_SSL：保持上游默认值 + CMAKE_DISABLE_FIND_PACKAGE_OpenSSL，
#     构建机无 OpenSSL 时自动无 TLS（显式 WITH_SSL=OFF 会触发 3.3.5 的坏
#     分支——openssl_crypt.c 被编入且缺 MA_HASH_CTX 类型，实测失败）
#   WITH_UNIT_TESTS/WITH_MYSQLCOMPAT/WITH_REMOTEIO/OFF + 两个 auth 插件 OFF：
#     固定最小客户端面
#   zlib：connector-c 3.3.5 无 WITH_ZLIB 开关，其 bundled zlib 对象进入
#     mariadbclient 归档；线路协议压缩（MYSQL_OPT_COMPRESS）即由该内嵌 zlib
#     承载——本仓不另有 zlib 实体，最终产物按归档符号拉取语义使用 connector
#     自带 zlib。
set(_mariadb_common_cmake_args
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_DISABLE_FIND_PACKAGE_OpenSSL=TRUE
    -DWITH_UNIT_TESTS=OFF
    -DWITH_MYSQLCOMPAT=OFF
    -DWITH_REMOTEIO=OFF
    # 插件 OFF 的变量名必须为 CLIENT_PLUGIN_<大写目标名>（REGISTER_PLUGIN 按
    # string(TOUPPER target) 读取，CMake 变量名大小写敏感；小写拼写会被静默
    # 忽略并按默认 DYNAMIC 构建——无 OpenSSL 配置下 client_ed25519/
    # caching_sha2_password/sha256_password 编译必炸）
    -DCLIENT_PLUGIN_CACHING_SHA2_PASSWORD=OFF
    -DCLIENT_PLUGIN_SHA256_PASSWORD=OFF
    -DCLIENT_PLUGIN_CLIENT_ED25519=OFF
    # FindGSSAPI（plugins/auth 与根 CMakeLists 两处无门控 include）经
    # find_program(KRB5_CONFIG) 从 PATH 找 krb5-config；预置缓存有效值使
    # find_program 跳过搜索，/bin/true 对 --libs/--cflags/--vendor 输出为空
    # 且返回 0，GSSAPI_LIBS 为空、GSSAPI_FOUND 保持未置位，GSSAPI_INCS/
    # GSSAPI_LIBS 不落缓存。本工具不使用 Kerberos 认证；不短路时构建环境
    # 的 PATH 前缀（如 conda base 自带 krb5）会经 GSSAPI_INCS 把该前缀的
    # include/lib 注入全部子构建目标。
    -DKRB5_CONFIG=/bin/true
    # connector-c 根 CMakeLists 的 cmake_minimum_required 为 2.8.12
    # （external/zlib 为 2.4.4），CMake 4.x 拒绝配置低于 3.5 的工程；
    # 显式声明按 3.5 策略层配置（CMake <3.31 无此变量，作为普通缓存项
    # 忽略，无副作用）。
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
)

# libmariadb 静态归档成员引用 dlopen/dlsym/dlclose/dlerror 与
# pthread_mutex_*（上游独立构建的 SYSTEM_LIBS 为 dl;m，静态归档消费方需
# 自行补齐）：glibc <2.34（如 Ubuntu 20.04）的 dl/pthread 符号不在 libc，
# glibc >=2.34 链接桩库同样无害。共享库形态下该依赖集合同样成立。
set(_mariadb_client_link_libs "${CMAKE_DL_LIBS}")
list(APPEND _mariadb_client_link_libs m pthread)

if(NOT BUILD_DEVICE_ARM)
    # ---- 宿主构建：独立 cmake 子构建（ExternalProject 隔离，不上溯其
    # install/export 规则污染 hccl_vm_install）----
    include(ExternalProject)
    # my_config.h/mariadb_version.h 由子构建生成于 binary dir；configure 期
    # 预建 include 目录，避免 IMPORTED target 生成期存在性校验失败
    file(MAKE_DIRECTORY "${MARIADB_CBUILD_DIR}/include")
    if(HVM_MARIADB_LINKAGE STREQUAL "shared")
        set(_mariadb_shared_flag -DBUILD_SHARED_LIBS=ON)
        set(_mariadb_byproduct "${MARIADB_SHARED_LIBRARY}")
    else()
        set(_mariadb_shared_flag -DBUILD_SHARED_LIBS=OFF)
        set(_mariadb_byproduct "${MARIADB_STATIC_LIBRARY}")
    endif()
    ExternalProject_Add(mariadb_connector
        SOURCE_DIR "${MARIADB_SOURCE_DIR}"
        BINARY_DIR "${MARIADB_CBUILD_DIR}"
        CMAKE_ARGS ${_mariadb_common_cmake_args} ${_mariadb_shared_flag}
        INSTALL_COMMAND ""
        BUILD_BYPRODUCTS "${_mariadb_byproduct}"
        BUILD_ALWAYS ON
    )
    if(HVM_MARIADB_LINKAGE STREQUAL "shared")
        add_library(mariadb_client SHARED IMPORTED GLOBAL)
        set_target_properties(mariadb_client PROPERTIES
            IMPORTED_LOCATION "${MARIADB_SHARED_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${MARIADB_SOURCE_DIR}/include;${MARIADB_CBUILD_DIR}/include"
            INTERFACE_LINK_LIBRARIES "${_mariadb_client_link_libs}")
        # 发布变体：libmariadb.so.3 随包分发（LGPL 动态链接合规形态）
        install(FILES "${MARIADB_SHARED_LIBRARY}" DESTINATION "lib")
    else()
        add_library(mariadb_client STATIC IMPORTED GLOBAL)
        set_target_properties(mariadb_client PROPERTIES
            IMPORTED_LOCATION "${MARIADB_STATIC_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${MARIADB_SOURCE_DIR}/include;${MARIADB_CBUILD_DIR}/include"
            INTERFACE_LINK_LIBRARIES "${_mariadb_client_link_libs}")
    endif()
endif()
