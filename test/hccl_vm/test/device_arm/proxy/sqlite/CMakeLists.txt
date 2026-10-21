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

# 文件职责：device_arm proxy 域的 SQLite 引擎专属测试（真实 SQLite fixture/惰性默认
# 会话；随 sqlite 后端模块一起保留或裁剪）。COMMON_* 变量继承自
# test/device_arm/proxy/CMakeLists.txt；目标名与断言保持迁移前不变。

# test_device_sqe_parse_stub (90 TEST_F)
add_executable(test_device_sqe_parse_stub device_sqe_parse_stub_test.cc)

target_sources(test_device_sqe_parse_stub PRIVATE
    ${CMAKE_SOURCE_DIR}/src/device_arm/level2/proxy/device_sqe_parse_stub.cc
    ${CMAKE_SOURCE_DIR}/src/device_arm/common/hccl_device_pub.cc
    ${CMAKE_SOURCE_DIR}/src/device_arm/common/hccl_task_collection.cc
    ${CMAKE_SOURCE_DIR}/src/common/sim_pipe_io.cc
    ${CMAKE_SOURCE_DIR}/src/store/store_sim_memory_manager.cc
    ${COMMON_SOURCES}
)

target_include_directories(test_device_sqe_parse_stub PRIVATE ${COMMON_INCLUDE_DIRS})

target_link_libraries(test_device_sqe_parse_stub PRIVATE ${COMMON_LIBS})

set_target_properties(test_device_sqe_parse_stub PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

target_compile_options(test_device_sqe_parse_stub PRIVATE
    -DCANN_VERSION_NUM=90000000
)

# test_device_sqe_task_resolution —— SQE 任务构造地址解析回归（任务字段语义/失败关闭）
add_executable(test_device_sqe_task_resolution device_sqe_task_resolution_test.cc)

target_sources(test_device_sqe_task_resolution PRIVATE
    ${CMAKE_SOURCE_DIR}/src/device_arm/level2/proxy/device_sqe_parse_stub.cc
    ${CMAKE_SOURCE_DIR}/src/device_arm/common/hccl_device_pub.cc
    ${CMAKE_SOURCE_DIR}/src/device_arm/common/hccl_task_collection.cc
    ${CMAKE_SOURCE_DIR}/src/common/sim_pipe_io.cc
    ${CMAKE_SOURCE_DIR}/src/store/store_sim_memory_manager.cc
    ${COMMON_SOURCES}
)

target_include_directories(test_device_sqe_task_resolution PRIVATE ${COMMON_INCLUDE_DIRS})

target_link_libraries(test_device_sqe_task_resolution PRIVATE ${COMMON_LIBS})

set_target_properties(test_device_sqe_task_resolution PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

target_compile_options(test_device_sqe_task_resolution PRIVATE
    -DCANN_VERSION_NUM=90000000
)

# test_ascend_hal_stub (36 TEST_F)
add_executable(test_ascend_hal_stub ascend_hal_stub_test.cc)

target_sources(test_ascend_hal_stub PRIVATE
    ${CMAKE_SOURCE_DIR}/src/device_arm/level2/proxy/ascend_hal_stub.cc
    ${CMAKE_SOURCE_DIR}/src/device_arm/level2/proxy/device_sqe_parse_stub.cc
    ${CMAKE_SOURCE_DIR}/src/device_arm/common/hccl_device_pub.cc
    ${CMAKE_SOURCE_DIR}/src/device_arm/common/hccl_task_collection.cc
    ${CMAKE_SOURCE_DIR}/src/common/sim_pipe_io.cc
    ${CMAKE_SOURCE_DIR}/src/store/store_sim_memory_manager.cc
    ${CMAKE_SOURCE_DIR}/src/runtime_state/db_sim_runner_common.cc
    ${COMMON_SOURCES}
)

target_include_directories(test_ascend_hal_stub PRIVATE ${COMMON_INCLUDE_DIRS})

target_link_libraries(test_ascend_hal_stub PRIVATE ${COMMON_LIBS})

set_target_properties(test_ascend_hal_stub PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
