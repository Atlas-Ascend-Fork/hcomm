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

add_subdirectory(proxy)

set(COMMON_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/device_arm
    ${CMAKE_SOURCE_DIR}/src/device_arm/common
    ${CMAKE_SOURCE_DIR}/src/device_arm/level2/proxy
    ${CMAKE_SOURCE_DIR}/src/common
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/store
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/utils
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/header/external
    ${CMAKE_SOURCE_DIR}/third_party/CLI11
    ${YAMLCPP_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/third_party
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/include/driver
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/pkg_inc/driver
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/pkg_inc/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/pkg_inc/toolchain
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/pkg_inc/profiling
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/pkg_inc/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/pkg_inc/base
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/include
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/include/hccl

    ${HCOMM_CODE_ROOT}/src/base_comm/resources/ccu/ccu_microcode
)

set(COMMON_SOURCES
    ${CMAKE_SOURCE_DIR}/src/utils/sim_log.cc
    ${CMAKE_SOURCE_DIR}/src/proxy/level2/log_stub.cc
)

set(COMMON_LIBS
    gtest_main
    gtest
    log
    dl
    rt
    -Wl,--start-group
    store
    runtime_state
    operation_data
    -Wl,--end-group
)

# 1. test_hccl_task_collection (10 TEST_F)
add_executable(test_hccl_task_collection hccl_task_collection_test.cc)

target_sources(test_hccl_task_collection PRIVATE
    ${CMAKE_SOURCE_DIR}/src/device_arm/common/hccl_task_collection.cc
    ${COMMON_SOURCES}
)

target_include_directories(test_hccl_task_collection PRIVATE ${COMMON_INCLUDE_DIRS})

target_link_libraries(test_hccl_task_collection PRIVATE ${COMMON_LIBS})

set_target_properties(test_hccl_task_collection PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

# test_data_op_channel_context_fake —— GetChannelTaskContext 查询次数回归的 Fake
# 引擎用例（自 data_op_device_stub_test.cc 拆分：ConformanceMemoryEngine 观察点，
# 不依赖真实后端，全部源码形态构建执行；真实 SQLite fixture 部分在 sqlite/ 子目录）
add_executable(test_data_op_channel_context_fake data_op_channel_context_fake_test.cc)

target_sources(test_data_op_channel_context_fake PRIVATE
    ${CMAKE_SOURCE_DIR}/src/device_arm/common/hccl_task_collection.cc
    ${CMAKE_SOURCE_DIR}/src/device_arm/common/hccl_device_pub.cc
    ${CMAKE_SOURCE_DIR}/src/common/sim_pipe_io.cc
    ${CMAKE_SOURCE_DIR}/src/store/store_sim_memory_manager.cc
    ${COMMON_SOURCES}
)

target_include_directories(test_data_op_channel_context_fake PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_libraries(test_data_op_channel_context_fake PRIVATE ${COMMON_LIBS})
target_compile_options(test_data_op_channel_context_fake PRIVATE -DCANN_VERSION_NUM=90100000)

set_target_properties(test_data_op_channel_context_fake PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

# 真实共享内存和 pipe 响应：独立 device VA 共享复用池，及 CPU 访问/释放回归。
add_executable(test_sim_aicpu_memory sim_aicpu_memory_test.cc)
target_sources(test_sim_aicpu_memory PRIVATE
    ${CMAKE_SOURCE_DIR}/src/device_arm/common/sim_aicpu_pipe_handler.cc
    # sim_aicpu_pipe_handler 的 kernel 执行窗口调用 sim::SetBarrierGateContext（方案二真等共享模块）；
    # 生产形态该符号由 device 可执行 --export-dynamic 运行时解析，测试可执行须直接编入（同 test_sim_aicpu_comminit）。
    ${CMAKE_SOURCE_DIR}/src/device_arm/common/barrier_real_wait.cc
    ${CMAKE_SOURCE_DIR}/src/device_arm/common/hccl_device_pub.cc
    ${CMAKE_SOURCE_DIR}/src/device_arm/common/hccl_task_collection.cc
    ${CMAKE_SOURCE_DIR}/src/device_arm/common/sim_kernel_lib_mgr.cc
    ${CMAKE_SOURCE_DIR}/src/common/sim_pipe_io.cc
    ${CMAKE_SOURCE_DIR}/src/store/store_sim_memory_manager.cc
    ${COMMON_SOURCES}
)
target_include_directories(test_sim_aicpu_memory PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_libraries(test_sim_aicpu_memory PRIVATE ${COMMON_LIBS})
set_target_properties(test_sim_aicpu_memory PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
add_test(NAME sim_aicpu_memory COMMAND test_sim_aicpu_memory)
set_tests_properties(sim_aicpu_memory PROPERTIES TIMEOUT 30)

# 引擎专属测试子目录（test/device_arm/<模块名>/）：模块存在才接入
#（真实 SQLite fixture/惰性默认会话依赖的 device_arm 用例随 sqlite 模块裁剪）。
include(${CMAKE_SOURCE_DIR}/src/storage_backends/CMakeLists.txt)
hcclvm_storage_backends_add_test_dir(${CMAKE_CURRENT_SOURCE_DIR})
