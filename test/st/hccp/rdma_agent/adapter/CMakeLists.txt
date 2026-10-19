# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set(test_ra_adp_list
    ${RDMA_ADP_SRC_LIST}
    ${HCCP_COMMON_SRC_LIST}
)

add_library(test_ra_adp SHARED
    ${test_ra_adp_list}
)

target_include_directories(test_ra_adp PRIVATE
    ${HCCP_DIR}/external_depends/inc/log
    ${HCCP_DIR}/inc/network
    ${HCCP_DIR}/inc/private/network
    ${HCCP_DIR}/rdma_agent/inc
    ${HCCP_DIR}/rdma_agent/hdc/
    ${HCCP_DIR}/rdma_agent/hdc/async/
    ${HCCP_DIR}/rdma_agent/comm
    ${HCCP_DIR}/rdma_agent/adapter
    ${HCCP_DIR}/rdma_agent/adapter/async/
    ${HCCP_DIR}/rdma_service
    ${HCCP_DIR}/common
    ${HCCP_DIR}/rdma_agent/adapter/tlv
    ${HCCP_DIR}/rdma_service/tlv
    ${HCCP_DIR}/external_depends/host_lite
    ${HCCP_DIR}/rdma_agent/adapter/ctx
    ${HCCP_DIR}/rdma_service/ctx
    ${RDMA_CORE_INCLUDE_DIR}
    ${ASCEND_CANN_PACKAGE_PATH}/include/
    ${ASCEND_CANN_PACKAGE_PATH}/include/driver/
)


target_compile_options(test_ra_adp PRIVATE
    -Werror -fsigned-char -fno-common -fno-strict-aliasing -std=gnu11
    -fstack-protector-strong
    -g -O0
)

target_link_options(test_ra_adp PRIVATE
    -Wl,-Bsymbolic
    -rdynamic
    -Wl,--no-undefined
    -Wl,-z,now
)

target_link_libraries(test_ra_adp PRIVATE
    test_rs_device
    hccp_stub
    unified_dlog_stub
    -lrt
    -ldl
    -pthread
)

install(
    TARGETS test_ra_adp
    LIBRARY DESTINATION ${HCCP_TEST_DIR}/device
    COMPONENT hccp_test
)