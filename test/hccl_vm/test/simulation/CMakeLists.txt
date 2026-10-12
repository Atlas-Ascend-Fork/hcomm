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
# See the License in the specific file packaged with the Software for details.

cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -D_GLIBCXX_USE_CXX11_ABI=0")

include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/store
    ${YAMLCPP_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/third_party
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/include
)

# test_sim_models
add_executable(test_sim_models sim_models_test.cc)
target_link_libraries(test_sim_models PRIVATE
    gtest_main
    gtest
)
set_target_properties(test_sim_models PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

# 引擎专属测试子目录（test/simulation/<模块名>/）：模块存在才接入
#（sqlite_runner_table / sqlite_opdb_ops / mariadb_opdb_projection 与 RunnerDB
# 业务回归 sim_runner_* 自此处迁入 sqlite/ 子目录）。
include(${CMAKE_SOURCE_DIR}/src/storage_backends/CMakeLists.txt)
hcclvm_storage_backends_add_test_dir(${CMAKE_CURRENT_SOURCE_DIR})
