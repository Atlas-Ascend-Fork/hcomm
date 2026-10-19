# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
add_executable(test_hccp_service.bin
    ${HCCP_SERVICE_SRC_LIST}
    ${HCCP_COMMON_SRC_LIST}
)

target_include_directories(test_hccp_service.bin PRIVATE
    ${HCCP_DIR}/hccp_service
    ${HCCP_DIR}/external_depends/inc/log
    ${HCCP_DIR}/external_depends/host_lite
    ${HCCP_DIR}/external_depends/rdma-core/providers/hns
    ${HCCP_DIR}/inc/network
    ${HCCP_DIR}/inc/private/network
    ${HCCP_DIR}/rdma_service
    ${HCCP_DIR}/rdma_agent/adapter
    ${HCCP_DIR}/rdma_agent/adapter/async
    ${HCCP_DIR}/rdma_agent/hdc
    ${HCCP_DIR}/rdma_agent/inc
    ${HCCP_DIR}/common
    ${RDMA_CORE_INCLUDE_DIR}
    ${ASCEND_CANN_PACKAGE_PATH}/include/driver/
    ${ASCEND_CANN_PACKAGE_PATH}/include/
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/aicpu
    ${TEST_DIR}/st/hccp/stub
)

set_target_properties(test_hccp_service.bin PROPERTIES OUTPUT_NAME "hccp_service.bin")

target_compile_options(test_hccp_service.bin PRIVATE
    -Werror -Wall -fPIC -fPIE -rdynamic -fvisibility=hidden -fno-common -fsigned-char -fno-strict-aliasing -std=gnu11
    -fstack-protector-strong
    -g -O0
)

target_link_options(test_hccp_service.bin PRIVATE
    -Wl,-Bsymbolic -Wl,--no-undefined -rdynamic
    -Wl,-z,now
    -pie
)

target_compile_definitions(test_hccp_service.bin PRIVATE
    google=ascend_private
    $<$<STREQUAL:${PRODUCT_SIDE},device>:CONFIG_CGROUP>
)

target_link_libraries(test_hccp_service.bin PRIVATE
    test_ra_adp
    test_rs_device
    hccp_stub
    unified_dlog_stub
    -ldl
)

install(
    TARGETS test_hccp_service.bin
    RUNTIME DESTINATION ${HCCP_TEST_DIR}/device
    COMPONENT hccp_test
)