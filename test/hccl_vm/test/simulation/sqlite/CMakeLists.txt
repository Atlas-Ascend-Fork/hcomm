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

# 文件职责：simulation 域的 SQLite 引擎专属测试（随 sqlite 后端模块一起保留
# 或裁剪）。目标名与断言保持迁移前不变。

function(add_sqlite_simulation_test target source)
    add_executable(${target} ${source})
    target_include_directories(${target} PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/store
        ${CMAKE_SOURCE_DIR}/test/simulation
    )
    target_link_libraries(${target} PRIVATE
        pthread
        gmock_main
        gmock
        gtest
        simulator_storage
        storage
        sqlite3
    )
    set_target_properties(${target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
    )
endfunction()

add_sqlite_simulation_test(test_sqlite_runner_table sqlite_runner_table_test.cc)

add_sqlite_simulation_test(test_sqlite_opdb_ops sqlite_opdb_ops_test.cc)
target_link_libraries(test_sqlite_opdb_ops PRIVATE runtime_state operation_data)

# ---- RunnerDB 业务回归（自 test/simulation 根迁入；目标名与断言不变）----
add_sqlite_simulation_test(test_sim_runner_common sim_runner_common_test.cc)
target_link_libraries(test_sim_runner_common PRIVATE runtime_state operation_data gmock_main gmock)

add_sqlite_simulation_test(test_sim_runner_db_test sim_runner_db_test.cc)
target_link_libraries(test_sim_runner_db_test PRIVATE runtime_state operation_data gmock_main gmock)

add_sqlite_simulation_test(test_sim_runner_ops sim_runner_ops_test.cc)
target_link_libraries(test_sim_runner_ops PRIVATE runtime_state operation_data gmock_main gmock)
