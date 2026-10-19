# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(test_ra_hdc_list
    ${RDMA_HDC_SRC_LIST}
    ${HCCP_COMMON_SRC_LIST}
    ${RDMA_COMMON_SRC_LIST}
)

add_library(test_ra_hdc  SHARED ${test_ra_hdc_list})

target_include_directories(test_ra_hdc PRIVATE
    ${HCCP_DIR}/rdma_agent/inc
    ${HCCP_DIR}/inc/network
    ${HCCP_DIR}/inc/private
    ${HCCP_DIR}/rdma_service
    ${HCCP_DIR}/inc/private/network
    ${HCCP_DIR}/common/
    ${HCCP_DIR}/external_depends/rdma-core/providers/hns
    ${HCCP_DIR}/external_depends/host_lite
    ${HCCP_DIR}/rdma_agent/comm/ctx
    ${HCCP_DIR}/rdma_agent/hdc
    ${HCCP_DIR}/rdma_agent/hdc/async
    ${HCCP_DIR}/rdma_agent/hdc/async
    ${HCCP_DIR}/rdma_agent/comm
    ${RDMA_CORE_INCLUDE_DIR}
    ${ASCEND_CANN_PACKAGE_PATH}/include/driver/
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base/
    ${ASCEND_CANN_PACKAGE_PATH}/include/
    ${ASCEND_CANN_PACKAGE_PATH}/include/toolchain/
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/include/
)

target_compile_options(test_ra_hdc PRIVATE
    -Werror -fno-common -fsigned-char -fno-strict-aliasing -std=gnu11 -Wextra
    -fstack-protector-strong
    -g -O0
)

target_link_options(test_ra_hdc PRIVATE
    -Wl,-z,now
)

target_link_libraries(test_ra_hdc PRIVATE
    unified_dlog_stub
    -lrt
    -ldl
)

install(
    TARGETS test_ra_hdc
    LIBRARY DESTINATION ${HCCP_TEST_DIR}/host
    COMPONENT hccp_test
)