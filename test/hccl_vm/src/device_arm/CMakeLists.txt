# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.14)
project(device LANGUAGES C CXX)

if(NOT DEFINED CANN_HCC_ROOT OR CANN_HCC_ROOT STREQUAL "")
    message(FATAL_ERROR "device_arm must be configured with the CANN HCC toolchain")
endif()

get_filename_component(_expected_c_compiler
    "${CANN_HCC_ROOT}/bin/aarch64-target-linux-gnu-gcc" REALPATH)
get_filename_component(_expected_cxx_compiler
    "${CANN_HCC_ROOT}/bin/aarch64-target-linux-gnu-g++" REALPATH)
get_filename_component(_expected_sysroot "${CANN_HCC_ROOT}/sysroot" REALPATH)
get_filename_component(_actual_c_compiler "${CMAKE_C_COMPILER}" REALPATH)
get_filename_component(_actual_cxx_compiler "${CMAKE_CXX_COMPILER}" REALPATH)
get_filename_component(_actual_sysroot "${CMAKE_SYSROOT}" REALPATH)

if(NOT CMAKE_CROSSCOMPILING OR
   NOT _actual_c_compiler STREQUAL _expected_c_compiler OR
   NOT _actual_cxx_compiler STREQUAL _expected_cxx_compiler OR
   NOT _actual_sysroot STREQUAL _expected_sysroot)
    message(FATAL_ERROR
        "device_arm compiler/sysroot does not match CANN HCC. "
        "Remove the device build directory and configure it with "
        "cmake/toolchains/aarch64-cann-hcc.cmake.")
endif()

message(STATUS "Cross C compiler:   ${CMAKE_C_COMPILER}")
message(STATUS "Cross CXX compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Cross sysroot:      ${CMAKE_SYSROOT}")

if(NOT DEFINED HCCL_VM_SRC)
    set(HCCL_VM_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()
if(NOT DEFINED THRID_PARTY_DIR)
    set(THRID_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party)
endif()
if(NOT DEFINED HCCL_VM_INCLUDE)
    set(HCCL_VM_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
endif()
if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
    set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_TOOLKIT_HOME})
endif()
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/device-install" CACHE PATH "..." FORCE)
endif()
if(NOT DEFINED HCCL_CODE_ROOT)
    set(HCCL_CODE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# device_arm模块按劫持层级组织:
#   level1/ : 设备侧HCCP通信域/控制面接口劫持 (hccl_device_proxy_level1)
#   level2/ : 设备侧驱动/Runtime层接口劫持 (hccl_device_proxy / slog / aicpu_sharder)
#   common/ : 各层级共用的源码与头文件 (预留)
# 顶层为device可执行程序的入口(main.cc)与框架文件
add_subdirectory(level2)
add_subdirectory(level1)

# ---- yaml-cpp（Device 交叉架构，复用主工程交叉编译产物）----
# aarch64 版 yaml-cpp 静态库 (由顶层 cmake/third_party/yaml-cpp-aarch64.cmake 交叉编译产出，
# C++17/PIC/_GLIBCXX_USE_CXX11_ABI=0 由该配方负责)。存储部署配置解析与 bin/device 的
# HAVE_YAML_CPP 均复用该 IMPORTED 目标；不重复解压源码编译，不链接宿主架构的 libyaml-cpp。
if(NOT DEFINED YAMLCPP_AARCH64_INSTALL_PATH OR YAMLCPP_AARCH64_INSTALL_PATH STREQUAL "")
    set(YAMLCPP_AARCH64_INSTALL_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/yaml-cpp-aarch64)
endif()
set(YAMLCPP_AARCH64_INCLUDE_DIR ${YAMLCPP_AARCH64_INSTALL_PATH}/include)
set(YAMLCPP_AARCH64_LIB_DIR "")
if(EXISTS "${YAMLCPP_AARCH64_INSTALL_PATH}/lib64/libyaml-cpp.a")
    set(YAMLCPP_AARCH64_LIB_DIR "${YAMLCPP_AARCH64_INSTALL_PATH}/lib64")
elseif(EXISTS "${YAMLCPP_AARCH64_INSTALL_PATH}/lib/libyaml-cpp.a")
    set(YAMLCPP_AARCH64_LIB_DIR "${YAMLCPP_AARCH64_INSTALL_PATH}/lib")
endif()
if(YAMLCPP_AARCH64_LIB_DIR STREQUAL "" OR NOT EXISTS "${YAMLCPP_AARCH64_INCLUDE_DIR}/yaml-cpp/yaml.h")
    message(FATAL_ERROR
        "aarch64 yaml-cpp not found under ${YAMLCPP_AARCH64_INSTALL_PATH}. "
        "Build with -DBUILD_DEVICE_ARM=ON (top-level), which cross-compiles yaml-cpp via "
        "cmake/third_party/yaml-cpp-aarch64.cmake before building device.")
endif()
add_library(yaml-cpp-aarch64 STATIC IMPORTED GLOBAL)
set_target_properties(yaml-cpp-aarch64 PROPERTIES
    IMPORTED_LOCATION ${YAMLCPP_AARCH64_LIB_DIR}/libyaml-cpp.a
    INTERFACE_INCLUDE_DIRECTORIES ${YAMLCPP_AARCH64_INCLUDE_DIR}
)

# storage 后端无关实现与 composition 装配：与主工程同一份源码，按 device 交叉
# 架构在本工程内重新编译为独立静态库——不链接宿主架构 libstorage，不复制
# bootstrap 实现。storage_composition.cc 的 registration-only 静态注册器
# 经 device 内 operation_data_ops.cc 对 sim::EnsureSimulatorStorageSession 的既有
# 引用必然进入实际链接产物（链接器按未解析符号收入该翻译单元及其静态注册器）。
# 后端模块（引擎/composition 源码、客户库交叉构建依赖、生成的模块注册清单）
# 经通用收集器接入本目标，本文件不出现具体后端名；部署配置解析链接
# yaml-cpp-aarch64。
add_library(storage STATIC "")
target_sources(storage PRIVATE
    ${HCCL_VM_SRC}/storage/engine/relational/relational_sql_helpers.cc
    ${HCCL_VM_SRC}/storage/internal/session_control.cc
    ${HCCL_VM_SRC}/storage/internal/process_storage_context.cc
    ${HCCL_VM_SRC}/storage/internal/storage_runtime_impl.cc
    ${HCCL_VM_SRC}/storage/session_coordinator.cc
    ${HCCL_VM_SRC}/storage/storage_session.cc
    ${HCCL_VM_SRC}/storage/transaction.cc
    ${HCCL_VM_SRC}/storage_composition/storage_composition.cc
    ${HCCL_VM_SRC}/storage_composition/storage_deploy_config.cc
)
target_include_directories(storage PUBLIC
    ${HCCL_VM_INCLUDE}
    ${HCCL_VM_SRC}
    ${THRID_PARTY_DIR}
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/pkg_inc/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/include
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/include/hccl
    ${HCOMM_CODE_ROOT}/src/base_comm/resources/ccu/ccu_microcode
)
target_link_libraries(storage PUBLIC yaml-cpp-aarch64)

# 后端模块接入（device 交叉架构：各模块依赖配方 + 引擎/composition 源码 +
# 生成注册清单；交接材料缺失/校验失败由模块配方明确失败）
include(${HCCL_VM_SRC}/storage_backends/CMakeLists.txt)
hcclvm_storage_backends_attach_device(TARGET storage)

set_target_properties(storage PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
target_compile_options(storage PRIVATE
    -Werror
    -fno-common
    -fno-strict-aliasing
    -pipe
    -O0
    -g
)
target_compile_definitions(storage PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
)

# aarch64 版 yaml-cpp 静态库已在文件前部定义为 IMPORTED 目标（storage 链接同一份）；
# bin/device 启用 HAVE_YAML_CPP 以直接读取 config/log_config.yaml 的 proxy 段

add_executable(device
    main.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/hccl_device_pub.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/hccl_task_collection.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/sim_aicpu_pipe_handler.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/sim_kernel_lib_mgr.cc
    ${HCCL_VM_SRC}/store/store_sim_memory_manager.cc
    ${HCCL_VM_SRC}/store/store_sim_shm_ops.cc
    ${HCCL_VM_SRC}/store/store_sim_resource_root.cc

    ${HCCL_VM_SRC}/utils/sim_log.cc
    ${HCCL_VM_SRC}/utils/sim_yaml_config.cc
    ${HCCL_VM_SRC}/utils/sim_common_api.cc
    ${HCCL_VM_SRC}/proxy/level2/log_stub.cc
    ${HCCL_VM_SRC}/operation_data/operation_data_ops.cc
    # runtime_state 业务单行函数与全局量（g_cur_server_key / GetEndPointByEid 等）：
    # device 侧打桩库（libhccl_device_proxy.so）在子进程加载时经 --export-dynamic
    # 从本可执行文件解析，迁移前对应 runnerdb/tracedb/db_sim_runner_common.cc。
    ${HCCL_VM_SRC}/runtime_state/db_sim_runner_common.cc
    ${HCCL_VM_SRC}/common/sim_pipe_io.cc
)

target_link_libraries(device PRIVATE
    slog
    aicpu_sharder
    storage
    yaml-cpp-aarch64
    -ldl
    -lrt
    -lpthread
)

target_include_directories(device PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/pkg_inc/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/include
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/include/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/pkg_inc/base
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/include/hcomm/ccu

    ${HCOMM_CODE_ROOT}/src/base_comm/resources/ccu/ccu_microcode

    ${HCCL_VM_INCLUDE}
    ${HCCL_VM_INCLUDE}/include
    ${HCCL_VM_INCLUDE}/store

    ${HCCL_VM_SRC}/utils
    ${HCCL_VM_SRC}/device_arm
    ${HCCL_VM_SRC}/device_arm/common
    ${HCCL_VM_SRC}/common
    ${THRID_PARTY_DIR}
    ${THRID_PARTY_DIR}/nlohmann_json
    ${THRID_PARTY_DIR}/spdlog
    ${YAMLCPP_AARCH64_INCLUDE_DIR}
)

# 自适应CANN版本号变更，从CANN包version头解析CANN_VERSION_NUM
set(HCCL_CANN_VERSION_NUM 0)
set(_hccl_cann_version_header "${ASCEND_CANN_PACKAGE_PATH}/include/version/cann_version.h")
message(STATUS "Checking HCCL scoped CANN version header: ${_hccl_cann_version_header}")
if(EXISTS "${_hccl_cann_version_header}")
    file(STRINGS "${_hccl_cann_version_header}" _hccl_cann_ver_line
        REGEX "^#define[ \t]+CANN_VERSION_NUM[ \t]+")
    if(_hccl_cann_ver_line)
        string(REGEX REPLACE "^#define[ \t]+CANN_VERSION_NUM[ \t]+(.+)$" "\\1"
            CANN_EXPR "${_hccl_cann_ver_line}")
        math(EXPR HCCL_CANN_VERSION_NUM "${CANN_EXPR}" OUTPUT_FORMAT DECIMAL)
    endif()
endif()

message(STATUS "HCCL_CANN_VERSION_NUM: ${HCCL_CANN_VERSION_NUM}")
if(HCCL_CANN_VERSION_NUM EQUAL 0)
    message(WARNING "HCCL failed to resolve CANN_VERSION_NUM (got 0). "
        "Compatibility branches (e.g. HCCL_CANN_COMPAT_850) will fall back to the 9.0.0+ path.")
endif()

target_compile_options(device PRIVATE
    -Werror
    -fno-common
    -fno-strict-aliasing
    -pipe
    -O0
    -g
)
target_compile_definitions(device PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
    CANN_VERSION_NUM=${HCCL_CANN_VERSION_NUM}
    HAVE_YAML_CPP
)

target_link_options(device PUBLIC "-Wl,--export-dynamic")

install(TARGETS device
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib/aarch64
    ARCHIVE DESTINATION lib/aarch64
)

# 日志染色: 模块 tag
target_compile_definitions(device PRIVATE HCCL_VM_MODULE="DEVICE")
