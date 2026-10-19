# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_library(mock SHARED
    mock.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../stub/hccp_vm_net.c
    )

target_include_directories(mock PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../stub
    ${HCCP_DIR}/rdma_agent/inc
    ${HCCP_DIR}/rdma_agent/hdc
    ${HCCP_DIR}/rdma_agent/hdc/async
    ${HCCP_DIR}/inc/network
    ${HCCP_DIR}/inc/private/network
    ${HCCP_DIR}/rdma_agent/comm
    ${HCCP_DIR}/rdma_agent/peer
    ${HCCP_DIR}/rdma_agent/client/
    ${HCCP_DIR}/common/
    ${HCCP_DIR}/external_depends/host_lite
    ${HCCP_DIR}/rdma_agent/client/async
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base/
    ${ASCEND_CANN_PACKAGE_PATH}/include/
    ${ASCEND_CANN_PACKAGE_PATH}/include/toolchain/
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/include/
    ${ASCEND_CANN_PACKAGE_PATH}/include/driver/
    ${RDMA_CORE_INCLUDE_DIR}
)

install(
    TARGETS mock
    LIBRARY DESTINATION ${HCCP_TEST_DIR}/common
    COMPONENT hccp_test
)