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
project(hccl_vm VERSION 1.0.0)
#set(CMAKE_VERBOSE_MAKEFILE true)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 主机侧各 target 的优化级别：默认 -O2（原为硬编码 -O0，checker/可达性等计算密集代码
# 在 -O0 下慢一个数量级）。Debug/覆盖/asan 场景可用 -DHCCLVM_OPT_LEVEL=-O0 回退。
set(HCCLVM_OPT_LEVEL "-O2" CACHE STRING "Optimization level for host targets (use -O0 for debug)")

function(hcclvm_apply_common_flags target)
    target_compile_options(${target} PRIVATE
        -Werror
        -fno-common
        -fno-strict-aliasing
        -pipe
        ${HCCLVM_OPT_LEVEL}
        -g
        # GCC 13 在 -O2 下内联 STL(std::copy/__builtin_memmove) 处理空区间时会误报
        # -Wnonnull（-O0 不内联故不触发）；这里仅把该类告警降级为 warning，其它仍按 -Werror。
        -Wno-error=nonnull
    )
    target_compile_definitions(${target} PRIVATE
        _GLIBCXX_USE_CXX11_ABI=0
    )
endfunction()

set(OUTPUT_DIR ${CMAKE_BINARY_DIR}/output)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${OUTPUT_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${OUTPUT_DIR})

option(ENABLE_COVERAGE "Enable gcov/lcov coverage instrumentation for all targets" OFF)
if(ENABLE_COVERAGE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage -fprofile-arcs -ftest-coverage")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} --coverage")
    message(STATUS "Coverage instrumentation enabled: --coverage (all targets)")
endif()

option(BUILD_DEVICE_ARM "Build device_arm (ARM cross-compile)" OFF)
set(HCOMM_VM_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
    if(NOT DEFINED ENV{ASCEND_TOOLKIT_HOME})
    message(FATAL_ERROR "Error: ASCEND_TOOLKIT_HOME environment variable is not set. Please set it to the Ascend Toolkit installation path.")
    endif()
    set(ASCEND_CANN_PACKAGE_PATH  $ENV{ASCEND_TOOLKIT_HOME})
endif()
message(STATUS "ASCEND_CANN_PACKAGE_PATH: ${ASCEND_CANN_PACKAGE_PATH}")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)")
    set(CANN_ARCH_DIR "aarch64-linux")
else()
    set(CANN_ARCH_DIR "x86_64-linux")
endif()
message(STATUS "CANN_ARCH_DIR: ${CANN_ARCH_DIR} (processor: ${CMAKE_SYSTEM_PROCESSOR})")

if(BUILD_DEVICE_ARM)
    if(NOT DEFINED HCCL_CODE_ROOT)
        if(NOT DEFINED ENV{HCCL_CODE_HOME})
            message(FATAL_ERROR "Error: BUILD_DEVICE_ARM is ON, but HCCL_CODE_HOME environment variable is not set. Please set it to the HCCL source path.")
        endif()
        set(HCCL_CODE_ROOT $ENV{HCCL_CODE_HOME})
    endif()
    message(STATUS "HCCL_CODE_ROOT: ${HCCL_CODE_ROOT}")
endif()

if(NOT DEFINED HCOMM_CODE_ROOT)
    if(NOT DEFINED ENV{HCOMM_CODE_HOME})
    message(FATAL_ERROR "Error: HCOMM_CODE_HOME environment variable is not set. Please set it to the Ascend Toolkit installation path.")
    endif()
    set(HCOMM_CODE_ROOT  $ENV{HCOMM_CODE_HOME})
endif()
message(STATUS "HCOMM_CODE_ROOT: ${HCOMM_CODE_ROOT}")

set(HCCL_VM_INCLUDE      ${CMAKE_SOURCE_DIR}/include)
set(HCCL_VM_SRC          ${CMAKE_SOURCE_DIR}/src)
set(HCCL_VM_INSTALL      ${CMAKE_SOURCE_DIR}/hccl_vm_install)
set(THRID_PARTY_DIR      ${CMAKE_SOURCE_DIR}/third_party)
message(STATUS "HCOMM_CODE_ROOT: ${HCOMM_CODE_ROOT}")

set(HCOMM_CODE_PLATFORM  ${HCOMM_CODE_ROOT}/src/platform)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/hccl_vm_install" CACHE PATH "Default install prefix" FORCE)
    message(STATUS "Install prefix not set by user, defaulting to: ${CMAKE_INSTALL_PREFIX}")
endif()

set(CMAKE_BUILD_RPATH "$ORIGIN")
set(CMAKE_INSTALL_RPATH "$ORIGIN")
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)

include(cmake/third_party/yaml-cpp.cmake)
include(cmake/third_party/nlohmann_json.cmake)
include(cmake/third_party/cli11.cmake)
include(cmake/third_party/spdlog.cmake)
# 存储后端模块（src/storage_backends/<模块名>/，随目录整体保留或裁剪）：
# 通用收集器负责发现与接入——此处先发现模块并准备各自依赖（源码缓存、客户库
# target）与 device 子构建交接参数；engine/composition 源码在 src 各目标建立后
# 经 hcclvm_storage_backends_attach 接入。本层不出现具体后端名。
include(${CMAKE_SOURCE_DIR}/src/storage_backends/CMakeLists.txt)
hcclvm_storage_backends_prepare()
include(cmake/third_party/cann_cmake.cmake)

include_directories(
    ${CMAKE_SOURCE_DIR}/third_party/nlohmann_json
    ${CMAKE_SOURCE_DIR}/third_party
    ${CMAKE_SOURCE_DIR}/third_party/yaml-cpp/include
)

function(config_hcclvm_install)
    install(DIRECTORY DESTINATION "data")
    install(DIRECTORY DESTINATION "archive")
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/config/"
        DESTINATION "config")
    install(FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/asset/allocate_eid.sh"
        "${CMAKE_CURRENT_SOURCE_DIR}/asset/generate_cluster_topo.sh"
        "${CMAKE_CURRENT_SOURCE_DIR}/asset/generate_server_topo.sh"
        "${CMAKE_CURRENT_SOURCE_DIR}/asset/hccl_config.sh"
        "${CMAKE_CURRENT_SOURCE_DIR}/asset/archive.sh"
        DESTINATION "script"
        PERMISSIONS
            OWNER_READ OWNER_WRITE OWNER_EXECUTE
            GROUP_READ GROUP_EXECUTE
            WORLD_READ WORLD_EXECUTE
    )
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/asset/cluster_model/config/cluster"
        DESTINATION "config"
        FILE_PERMISSIONS
            OWNER_READ OWNER_WRITE
            GROUP_READ
            WORLD_READ
        DIRECTORY_PERMISSIONS
            OWNER_READ OWNER_WRITE OWNER_EXECUTE
            GROUP_READ GROUP_EXECUTE
            WORLD_READ WORLD_EXECUTE
    )
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/asset/cluster_model/config/server_or_pod"
        DESTINATION "config"
        FILE_PERMISSIONS
            OWNER_READ OWNER_WRITE
            GROUP_READ
            WORLD_READ
        DIRECTORY_PERMISSIONS
            OWNER_READ OWNER_WRITE OWNER_EXECUTE
            GROUP_READ GROUP_EXECUTE
            WORLD_READ WORLD_EXECUTE
    )
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/asset/cluster_model/topo_meta"
        DESTINATION "config"
        FILE_PERMISSIONS
            OWNER_READ OWNER_WRITE
            GROUP_READ
            WORLD_READ
        DIRECTORY_PERMISSIONS
            OWNER_READ OWNER_WRITE OWNER_EXECUTE
            GROUP_READ GROUP_EXECUTE
            WORLD_READ WORLD_EXECUTE
    )
endfunction()

# 编译HCCL-VM
if (NOT BUILD_DEVICE_ARM)
    config_hcclvm_install()
    add_subdirectory(src)
endif()

set(CPACK_PACKAGE_NAME "hccl_vm")
set(CPACK_GENERATOR "TGZ")
include(CPack)

# 编译DEVICE-ARM-交叉编译版本
if(BUILD_DEVICE_ARM)
    message(STATUS "BUILD_DEVICE_ARM start")
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/cann_hcc_validation.cmake")

    if(NOT DEFINED CANN_HCC_ROOT OR CANN_HCC_ROOT STREQUAL "")
        set(_cann_hcc_candidates
            "${ASCEND_CANN_PACKAGE_PATH}/tools/hcc"
            "${ASCEND_CANN_PACKAGE_PATH}/toolkit/toolchain/hcc"
        )
        set(_cann_hcc_candidate_errors "")
        foreach(_candidate IN LISTS _cann_hcc_candidates)
            hcclvm_validate_cann_hcc("${_candidate}" _candidate_is_complete _candidate_missing_path)
            if(_candidate_is_complete)
                set(CANN_HCC_ROOT "${_candidate}")
                break()
            endif()
            list(APPEND _cann_hcc_candidate_errors
                "${_candidate}: missing ${_candidate_missing_path}"
            )
        endforeach()
    endif()

    if(NOT DEFINED CANN_HCC_ROOT OR CANN_HCC_ROOT STREQUAL "")
        list(JOIN _cann_hcc_candidate_errors "\n  " _cann_hcc_error_details)
        message(FATAL_ERROR
            "Complete CANN HCC toolchain was not found under ${ASCEND_CANN_PACKAGE_PATH}.\n"
            "Tried:\n  ${_cann_hcc_error_details}\n"
            "Expected tools/hcc or toolkit/toolchain/hcc, or set CANN_HCC_ROOT explicitly.")
    endif()

    hcclvm_validate_cann_hcc("${CANN_HCC_ROOT}" _cann_hcc_is_complete _cann_hcc_missing_path)
    if(NOT _cann_hcc_is_complete)
        message(FATAL_ERROR "Incomplete CANN HCC toolchain: ${_cann_hcc_missing_path} was not found")
    endif()

    message(STATUS "CANN HCC root: ${CANN_HCC_ROOT}")
    message(STATUS "CANN HCC sysroot: ${CANN_HCC_ROOT}/sysroot")

    # 交叉编译 aarch64 版 yaml-cpp 静态库, 供 bin/device 直接读取 log_config.yaml
    include(cmake/third_party/yaml-cpp-aarch64.cmake)
    if(TARGET third_party_yaml_cpp_aarch64)
        set(_yaml_cpp_aarch64_dep third_party_yaml_cpp_aarch64)
    else()
        set(_yaml_cpp_aarch64_dep "")
    endif()

    # 与旧系统交叉 GCC 的 CMake cache 隔离，避免 CMake 忽略首次配置后传入的 toolchain file。
    set(DEVICE_ARM_BUILD_DIR ${CMAKE_SOURCE_DIR}/build_device/hcc)
    set(DEVICE_ARM_INSTALL_DIR ${CMAKE_SOURCE_DIR}/hccl_vm_install)
    # 存储后端模块的 device 交接参数（各模块自述的源码缓存/校验指纹；本层只
    # 传递列表，不出现具体后端名）
    hcclvm_storage_backends_device_args(HCCLVM_STORAGE_BACKEND_DEVICE_ARGS)
    include(ExternalProject)
    ExternalProject_Add(device_arm_external
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/src/device_arm
        CMAKE_ARGS
            -DCMAKE_TOOLCHAIN_FILE=${CMAKE_SOURCE_DIR}/cmake/toolchains/aarch64-cann-hcc.cmake
            -DCANN_HCC_ROOT=${CANN_HCC_ROOT}
            -DCMAKE_INSTALL_PREFIX=${DEVICE_ARM_INSTALL_DIR}
            -DHCCL_VM_SRC=${HCCL_VM_SRC}
            -DTHRID_PARTY_DIR=${THRID_PARTY_DIR}
            -DHCCL_VM_INCLUDE=${HCCL_VM_INCLUDE}
            -DASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}
            -DCANN_ARCH_DIR=${CANN_ARCH_DIR}
            -DHCOMM_CODE_ROOT=${HCOMM_CODE_ROOT}
            -DHCCL_CODE_ROOT=${HCCL_CODE_ROOT}
            -DYAMLCPP_AARCH64_INSTALL_PATH=${YAMLCPP_AARCH64_INSTALL_PATH}
            ${HCCLVM_STORAGE_BACKEND_DEVICE_ARGS}
        DEPENDS ${_yaml_cpp_aarch64_dep}
        BINARY_DIR ${DEVICE_ARM_BUILD_DIR}
        INSTALL_DIR ${DEVICE_ARM_INSTALL_DIR}
        BUILD_ALWAYS ON
    )

    message(STATUS "device_arm cross-compilation enabled")
    message(STATUS "  Build dir: ${DEVICE_ARM_BUILD_DIR}")
    message(STATUS "  Install dir: ${DEVICE_ARM_INSTALL_DIR}")
endif()

option(BUILD_TESTS "Build unit tests" OFF)
if(BUILD_TESTS)
    include(cmake/third_party/gtest.cmake)
    add_subdirectory(test)
endif()
