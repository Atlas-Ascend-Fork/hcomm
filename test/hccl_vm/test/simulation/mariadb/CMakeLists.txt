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

# 文件职责：simulation 域的 MariaDB 引擎专属测试（随 mariadb 后端模块一起
# 保留或裁剪）。目标名与断言保持迁移前不变。

# OpDetail 标量查询字段投影的 MariaDB SQL 生成路径回归（业务函数级；
# endpoint 由 HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 提供，未设置 GTEST_SKIP，
# 与 test/storage/mariadb 的 MariaDB 目标同一 env 供给协议）
add_executable(test_mariadb_opdb_projection mariadb_opdb_projection_test.cc)
target_include_directories(test_mariadb_opdb_projection PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/store
    ${CMAKE_SOURCE_DIR}/test/simulation
)
target_link_libraries(test_mariadb_opdb_projection PRIVATE
    pthread
    gmock_main
    gmock
    gtest
    runtime_state
    operation_data
    simulator_storage
    storage
    mariadb_client
)
add_dependencies(test_mariadb_opdb_projection mariadb_connector)
set_target_properties(test_mariadb_opdb_projection PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
