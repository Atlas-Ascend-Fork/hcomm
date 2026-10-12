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

# 文件职责：store 域的 SQLite 引擎专属测试（经 simulation sqlite fixture/惰性默认
# 会话依赖真实 SQLite；随 sqlite 后端模块一起保留或裁剪）。目录级 include/link
# 设置继承自 test/store/CMakeLists.txt；目标名与断言保持迁移前不变。

add_executable(test_dump_shm_data dump_shm_data_test.cc)
target_link_libraries(test_dump_shm_data PRIVATE store runtime_state operation_data log)
set_target_properties(test_dump_shm_data PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_executable(test_sim_device_memory_manager sim_device_memory_manager_test.cc)
target_link_libraries(test_sim_device_memory_manager PRIVATE pthread rt store runtime_state operation_data log)
set_target_properties(test_sim_device_memory_manager PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_executable(test_hccl_shm_pub hccl_shm_pub_test.cc
    ${CMAKE_SOURCE_DIR}/src/store/store_sim_store_pub.cc
)
target_include_directories(test_hccl_shm_pub PRIVATE
    ${CMAKE_SOURCE_DIR}/src/store
)
target_link_libraries(test_hccl_shm_pub PRIVATE pthread rt store runtime_state operation_data log)
set_target_properties(test_hccl_shm_pub PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_executable(test_sim_run_mode sim_run_mode_test.cc)
target_link_libraries(test_sim_run_mode PRIVATE pthread rt store runtime_state operation_data log)
set_target_properties(test_sim_run_mode PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_executable(test_sim_check_only_off sim_check_only_off_test.cc)
target_link_libraries(test_sim_check_only_off PRIVATE pthread rt store runtime_state operation_data log)
set_target_properties(test_sim_check_only_off PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
