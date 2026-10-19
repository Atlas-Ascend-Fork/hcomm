# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(test_rs_src_list
    ${RDMA_SERVICE_SRC_LIST}
    ${HCCP_COMMON_SRC_LIST}
)

function(create_rs_library PRODUCT_SIDE)
    set(TARGET_NAME "test_rs_${PRODUCT_SIDE}")
    add_library(${TARGET_NAME}  SHARED ${test_rs_src_list})

    target_include_directories(${TARGET_NAME} PRIVATE
        ${HCCP_DIR}/rdma_service/
        ${HCCP_DIR}/rdma_service/tlv
        ${HCCP_DIR}/rdma_service/ctx
        ${HCCP_DIR}/rdma_service/ctx/product
        ${HCCP_DIR}/inc
        ${HCCP_DIR}/inc/network
        ${HCCP_DIR}/inc/private/network
        ${HCCP_DIR}/common/
        ${HCCP_DIR}/external_depends/host_lite
        ${HCCP_DIR}/external_depends/rdma-core/providers/hns
        ${HCCP_DIR}/external_depends/ibv_extend
        ${HCCP_DIR}/external_depends/aubdfx
        ${HCCP_DIR}/external_depends/ccu
        ${HCCP_DIR}/external_depends/net_adapt
        ${HCCP_DIR}/external_depends/ubengine
        ${HCOMM_UTILS_INSTALL_PATH}/device/include/tls_adp
        ${HCOMM_UTILS_INSTALL_PATH}/device/include/kms
        ${OPENSSL_INCLUDE_DIR}
        ${RDMA_CORE_INCLUDE_DIR}
        ${URMA_INCLUDE_DIR}

        ${ASCEND_CANN_PACKAGE_PATH}/include/driver/
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base/
        ${ASCEND_CANN_PACKAGE_PATH}/include/
        ${ASCEND_CANN_PACKAGE_PATH}/include/toolchain/
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
        ${ASCEND_CANN_PACKAGE_PATH}/include/
    )

    target_compile_options(${TARGET_NAME} PRIVATE
        -Werror -fno-strict-aliasing -fsigned-char -fno-common -Wfloat-equal
        -Wextra -Wno-unused-parameter -std=gnu11
        -fstack-protector-strong
        -g -O0
    )

    target_link_options(${TARGET_NAME} PRIVATE
        -Wl,-Bsymbolic -Wl,--exclude-libs,ALL
        -Wl,-z,now
    )

    target_compile_definitions(${TARGET_NAME} PRIVATE
        $<$<STREQUAL:${PRODUCT_SIDE},host>:LOG_HOST>
        $<$<STREQUAL:${PRODUCT_SIDE},device>:CUSTOM_INTERFACE>
        $<$<STREQUAL:${PRODUCT_SIDE},device>:google=ascend_private>
        $<$<STREQUAL:${PRODUCT_SIDE},device>:CONFIG_TLV>
        CONFIG_V121
    )

    target_link_libraries(${TARGET_NAME} PRIVATE
        hccp_stub
        -lrt
        -ldl
    )

    install(
        TARGETS ${TARGET_NAME}
        LIBRARY DESTINATION ${HCCP_TEST_DIR}/${PRODUCT_SIDE}
        COMPONENT hccp_test
    )
endfunction()

create_rs_library(host)
create_rs_library(device)