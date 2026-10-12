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

# 文件职责：proxy 域的 MariaDB 引擎专属测试（随 mariadb 后端模块一起保留或
# 裁剪）。endpoint 由 HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 提供，未设置
# GTEST_SKIP，设置但不可用必须失败。

# RaRmem endpoint_id 在 MariaDB 后端的序列化回环（链接真实 hccp_stub.cc；
# 自 test/proxy/hccp_rmem_stub_test.cc 拆分，断言不变）
add_executable(test_hccp_rmem_mariadb hccp_rmem_mariadb_test.cc)
target_sources(test_hccp_rmem_mariadb PRIVATE
    ${CMAKE_SOURCE_DIR}/src/proxy/level2/hccp_stub.cc
    ${CMAKE_SOURCE_DIR}/src/proxy/level2/hccp_async_request.cc
    ${CMAKE_SOURCE_DIR}/src/proxy/level2/ibv_dpu_stub.cc
    ${CMAKE_SOURCE_DIR}/src/proxy/common/hccl_proxy_common.cc
)
target_include_directories(test_hccp_rmem_mariadb PRIVATE
    ${CMAKE_SOURCE_DIR}/src/proxy/level2
    ${CMAKE_SOURCE_DIR}/src/proxy/common
    ${CMAKE_SOURCE_DIR}/src/store
    ${HCOMM_CODE_ROOT}/src/base_comm/resources/hccp
    ${HCOMM_CODE_ROOT}/src/base_comm/resources/hccp/inc
    ${HCOMM_CODE_ROOT}/src/base_comm/resources/hccp/inc/network
    ${HCOMM_CODE_ROOT}/src/base_comm/resources/hccp/external_depends/ccu
    ${HCOMM_CODE_ROOT}/src/platform/hccp
    ${HCOMM_CODE_ROOT}/src/platform/hccp/inc
    ${HCOMM_CODE_ROOT}/src/platform/hccp/inc/network
)
set_target_properties(test_hccp_rmem_mariadb PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
target_link_libraries(test_hccp_rmem_mariadb PRIVATE
    gtest_main
    common
    runtime_state
    operation_data
    simulator_storage
    storage
    mariadb_client
    log
)
add_dependencies(test_hccp_rmem_mariadb mariadb_connector)
