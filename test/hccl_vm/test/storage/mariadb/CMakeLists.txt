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

# 文件职责：MariaDB 引擎专属测试（test/storage/mariadb/，随 mariadb 后端模块
# 一起保留或裁剪；由收集器按模块存在性接入）。endpoint 由
# HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 提供，未设置 GTEST_SKIP，设置但不可用
# 必须失败。目标名与断言保持迁移前不变。

# 引擎级单元测试：直连真实 mysqld。六类型往返/自增回填/大 BLOB/定长数组/
# 语句缓存命中计数/外键 DDL/重开预检等；mariadb 段部署配置解析合同用例
# （模块段解析入口 + 公共中立解析器外层行为）。
add_executable(test_mariadb_engine mariadb_engine_test.cc)
target_include_directories(test_mariadb_engine PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/test/storage
)
target_link_libraries(test_mariadb_engine PRIVATE
    simulator_storage
    storage
    gtest_main
    gtest
    mariadb_client
    yaml-cpp::yaml-cpp
)
add_dependencies(test_mariadb_engine mariadb_connector)
if(TARGET third_party_yaml_cpp)
    add_dependencies(test_mariadb_engine third_party_yaml_cpp)
endif()
set_target_properties(test_mariadb_engine PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

# 公共 backend conformance suite 的 MariaDB fixture：实例化 ALL 清单（含事务
# case——InnoDB 原生事务直达模型；含投影清单——MariaDB 实现字段投影）。
add_executable(test_backend_conformance_mariadb backend_conformance_mariadb_test.cc)
target_include_directories(test_backend_conformance_mariadb PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/test/storage
)
target_link_libraries(test_backend_conformance_mariadb PRIVATE
    storage
    gtest_main
    gtest
    mariadb_client
)
add_dependencies(test_backend_conformance_mariadb mariadb_connector)
set_target_properties(test_backend_conformance_mariadb PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

# 事务重叠合同专项：同线程立即拒绝（回归防线）/跨线程即时 CRUD 与第二笔
# Begin 有界等待/等待超时 RESOURCE_EXHAUSTED/关闭唤醒（等待上限经测试 peer
# 缩短；旧实现无等待成员时编译期退化为无操作——超时用例即 RED 证据）。
add_executable(test_mariadb_transaction_overlap mariadb_transaction_overlap_test.cc)
target_include_directories(test_mariadb_transaction_overlap PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/test/storage
)
target_link_libraries(test_mariadb_transaction_overlap PRIVATE
    simulator_storage
    storage
    gtest_main
    gtest
    mariadb_client
)
add_dependencies(test_mariadb_transaction_overlap mariadb_connector)
set_target_properties(test_mariadb_transaction_overlap PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

# 批量插入 override 专项：入参合同/快路径真实主键与类型保真/显式混合与
# Unique/FK/旧服务端回退/中途失败行号与回退/batch statement 计数观察点
# （合格批次一次批命令；旧实现计数恒 0 即 RED 证据）。
add_executable(test_mariadb_insert_many mariadb_insert_many_test.cc)
target_include_directories(test_mariadb_insert_many PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/test/storage
)
target_link_libraries(test_mariadb_insert_many PRIVATE
    simulator_storage
    storage
    gtest_main
    gtest
    mariadb_client
)
add_dependencies(test_mariadb_insert_many mariadb_connector)
set_target_properties(test_mariadb_insert_many PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

# MariaDB 后端模块专项静态门禁（verified-source 版本/双哈希/许可/配置样例）
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(
    NAME mariadb_backend_static_gate
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/mariadb_static_gate.py
)
