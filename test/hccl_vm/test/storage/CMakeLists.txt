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

# 文件职责：公共 storage 合同测试（后端无关，Fake/内存引擎）。引擎专属测试在
# 按引擎命名的子目录（sqlite/、mariadb/）——由收集器按模块存在性接入，裁剪掉
# 对应后端模块的源码树不生成其测试目标。
add_executable(test_storage_contract storage_contract_test.cc)
target_include_directories(test_storage_contract PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(test_storage_contract PRIVATE
    storage
    gtest_main
    gtest
)
set_target_properties(test_storage_contract PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

# 统一逻辑 DDL 目录（StorageCatalog）机制合同测试。
add_executable(test_storage_catalog storage_catalog_test.cc)
target_include_directories(test_storage_catalog PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(test_storage_catalog PRIVATE
    storage
    gtest_main
    gtest
)
set_target_properties(test_storage_catalog PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

# 公共 backend conformance suite 的 Fake 后端 fixture（内存语义引擎：验证 fixture 协议与全部公共 case）。
add_executable(test_backend_conformance_fake backend_conformance_fake_test.cc)
target_include_directories(test_backend_conformance_fake PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(test_backend_conformance_fake PRIVATE
    storage
    gtest_main
    gtest
)
set_target_properties(test_backend_conformance_fake PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

# 【集成点】test_relational_engine 与 test_storage_schema_manifest 验证的公共
# 关系层/schema manifest 已在公共接口冻结中删除，目标随之移除。

# 引擎专属测试子目录（test/storage/<模块名>/）：模块存在才接入。
include(${CMAKE_SOURCE_DIR}/src/storage_backends/CMakeLists.txt)
hcclvm_storage_backends_add_test_dir(${CMAKE_CURRENT_SOURCE_DIR})
