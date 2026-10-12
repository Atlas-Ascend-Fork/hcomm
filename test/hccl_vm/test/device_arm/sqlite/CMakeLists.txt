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

# 文件职责：device_arm 域的 SQLite 引擎专属测试（真实 SQLite 文件 fixture/惰性默认
# 会话；随 sqlite 后端模块一起保留或裁剪）。COMMON_* 变量继承自
# test/device_arm/CMakeLists.txt；目标名与断言保持迁移前不变。

# AICPU数据面桩的x86纯转换与参数校验测试。
add_executable(test_data_op_device_stub data_op_device_stub_test.cc)

target_sources(test_data_op_device_stub PRIVATE
    ${CMAKE_SOURCE_DIR}/src/device_arm/common/hccl_task_collection.cc
    ${CMAKE_SOURCE_DIR}/src/device_arm/common/hccl_device_pub.cc
    ${CMAKE_SOURCE_DIR}/src/common/sim_pipe_io.cc
    ${CMAKE_SOURCE_DIR}/src/store/store_sim_memory_manager.cc
    ${COMMON_SOURCES}
)

target_include_directories(test_data_op_device_stub PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_libraries(test_data_op_device_stub PRIVATE ${COMMON_LIBS})
target_compile_options(test_data_op_device_stub PRIVATE -DCANN_VERSION_NUM=90100000)

set_target_properties(test_data_op_device_stub PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

# 2. test_hccl_device_pub (12 TEST_F)
add_executable(test_hccl_device_pub hccl_device_pub_test.cc)

target_sources(test_hccl_device_pub PRIVATE
    ${CMAKE_SOURCE_DIR}/src/device_arm/common/hccl_device_pub.cc
    ${CMAKE_SOURCE_DIR}/src/common/sim_pipe_io.cc
    ${CMAKE_SOURCE_DIR}/src/store/store_sim_memory_manager.cc
    ${COMMON_SOURCES}
)

target_include_directories(test_hccl_device_pub PRIVATE ${COMMON_INCLUDE_DIRS})

target_link_libraries(test_hccl_device_pub PRIVATE ${COMMON_LIBS})

set_target_properties(test_hccl_device_pub PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

target_compile_options(test_hccl_device_pub PRIVATE
    -DCANN_VERSION_NUM=90000000
)

# 3. CommInit 入口检查专项：受控 kernel .so（记录调用计数与收到的 deviceAddr；
# 桩源文件留在 test/device_arm/ 根，仅本目录目标引用）
add_library(test_kern_comminit SHARED ${CMAKE_SOURCE_DIR}/test/device_arm/test_kern_comminit.cc)
set_target_properties(test_kern_comminit PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)
target_include_directories(test_kern_comminit PRIVATE
    ${CMAKE_SOURCE_DIR}/src/device_arm/common
)

# test_sim_aicpu_comminit：必需地址映射失败 kernel 不被调用 / 可选块保留 /
# is_dev_access 直通（消费合同已与冻结 HCOMM 48b54e99 逐字段核对）
add_executable(test_sim_aicpu_comminit sim_aicpu_comminit_test.cc)
target_sources(test_sim_aicpu_comminit PRIVATE
    ${CMAKE_SOURCE_DIR}/src/device_arm/common/sim_aicpu_pipe_handler.cc
    ${CMAKE_SOURCE_DIR}/src/device_arm/common/hccl_device_pub.cc
    ${CMAKE_SOURCE_DIR}/src/device_arm/common/hccl_task_collection.cc
    ${CMAKE_SOURCE_DIR}/src/device_arm/common/sim_kernel_lib_mgr.cc
    ${CMAKE_SOURCE_DIR}/src/common/sim_pipe_io.cc
    ${CMAKE_SOURCE_DIR}/src/store/store_sim_memory_manager.cc
    ${COMMON_SOURCES}
)
target_include_directories(test_sim_aicpu_comminit PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_libraries(test_sim_aicpu_comminit PRIVATE ${COMMON_LIBS})
add_dependencies(test_sim_aicpu_comminit test_kern_comminit)
target_compile_definitions(test_sim_aicpu_comminit PRIVATE
    HCCL_VM_TEST_KERNEL_SO="${CMAKE_CURRENT_BINARY_DIR}/$<TARGET_FILE_NAME:test_kern_comminit>"
)
set_target_properties(test_sim_aicpu_comminit PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
