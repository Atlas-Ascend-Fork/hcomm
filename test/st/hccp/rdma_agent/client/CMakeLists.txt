# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(test_ra_list
    ${RDMA_AGENT_SRC_LIST}
    ${RDMA_COMMON_SRC_LIST}
)

add_library(test_ra  SHARED ${test_ra_list})

target_include_directories(test_ra PRIVATE
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


target_compile_options(test_ra PRIVATE
    -Werror -fvisibility=hidden -fno-strict-aliasing -fsigned-char -fno-common -std=gnu11 -Wextra
    -fstack-protector-strong
    -g -O0
)

target_link_options(test_ra PRIVATE
    -Wl,-z,now
)

target_link_libraries(test_ra PRIVATE
    test_ra_hdc
    test_ra_peer
    unified_dlog_stub
    -ldl
    -lrt
)

install(
    TARGETS test_ra
    LIBRARY DESTINATION ${HCCP_TEST_DIR}/host
    COMPONENT hccp_test
)