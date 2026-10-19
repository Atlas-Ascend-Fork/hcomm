# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_library(hccp_stub SHARED
    hccp_stub.c
    sec_stub.c
    ssl_stub.c
    ${CMAKE_CURRENT_SOURCE_DIR}/hccp_shm_infra.c
)

target_include_directories(hccp_stub PRIVATE
    ${HCCP_DIR}/inc/network
    ${HCCP_DIR}/inc/private/network
    ${HCCP_DIR}/rdma_service/
    ${HCCP_DIR}/external_depends/ibv_extend
    ${HCCP_DIR}/external_depends/ubengine
    ${HCCP_DIR}/external_depends/ccu
    ${HCCP_DIR}/external_depends/net_adapt
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base/
    ${ASCEND_CANN_PACKAGE_PATH}/include/
    ${ASCEND_CANN_PACKAGE_PATH}/include/toolchain/
    ${ASCEND_CANN_PACKAGE_PATH}/include/driver/
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/include/
    ${HCOMM_UTILS_INSTALL_PATH}/device/include/tls_adp
    ${HCOMM_UTILS_INSTALL_PATH}/device/include/kms
    ${OPENSSL_INCLUDE_DIR}
    ${RDMA_CORE_INCLUDE_DIR}
)

install(
    TARGETS hccp_stub
    LIBRARY DESTINATION ${HCCP_TEST_DIR}/common
    COMPONENT hccp_test
)

add_library(unified_dlog_stub SHARED
    dlog_stub.c
)

target_include_directories(unified_dlog_stub PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base/
    ${ASCEND_CANN_PACKAGE_PATH}/include/
    ${ASCEND_CANN_PACKAGE_PATH}/include/toolchain/
    ${ASCEND_CANN_PACKAGE_PATH}/include/driver/
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/include/
)

install(
    TARGETS unified_dlog_stub
    LIBRARY DESTINATION ${HCCP_TEST_DIR}/common
    COMPONENT hccp_test
)

function(create_hal_library PRODUCT_SIDE)
    set(TARGET_NAME "ascend_hal_${PRODUCT_SIDE}_stub")
    add_library(${TARGET_NAME} SHARED
        ascend_hal_${PRODUCT_SIDE}_stub.c
        ascend_hal_common.c
        hccp_vm_net.c
        ${CMAKE_CURRENT_SOURCE_DIR}/hccp_shm_infra.c
    )

    string(TOUPPER ${PRODUCT_SIDE} PRODUCT_SIDE_UPPER)

    target_compile_definitions(${TARGET_NAME} PRIVATE
        LOG_SIDE="${PRODUCT_SIDE_UPPER}"
    )

    target_compile_options(${TARGET_NAME} PRIVATE
        -fno-strict-aliasing -fsigned-char -fno-common -std=gnu11 -Wextra
        -g -O0
    )

    target_link_libraries(${TARGET_NAME} PRIVATE
        unified_dlog_stub
    )

    set_target_properties(${TARGET_NAME} PROPERTIES
        OUTPUT_NAME "ascend_hal"
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${PRODUCT_SIDE}
    )

    target_include_directories(${TARGET_NAME} PRIVATE
        ${ASCEND_CANN_PACKAGE_PATH}/include/driver/
    )

    install(
        TARGETS ${TARGET_NAME}
        LIBRARY DESTINATION ${HCCP_TEST_DIR}/${PRODUCT_SIDE}
        COMPONENT hccp_test
    )
endfunction()

create_hal_library(host)
create_hal_library(device)

add_library(urma SHARED
    urma.c
)

set_target_properties(urma PROPERTIES
    OUTPUT_NAME          "urma"
    SOVERSION            "0"
    VERSION              "0.0.0"
    PREFIX               "lib"
)

target_include_directories(urma PRIVATE
    ${URMA_INCLUDE_DIR}
    ${TEST_DIR}/stub/depends/include
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base/
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(urma PRIVATE
    pthread
    unified_dlog_stub
)

target_sources(urma PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/hccp_shm_infra.c
)

install(
    TARGETS urma
    LIBRARY DESTINATION ${HCCP_TEST_DIR}/common
    COMPONENT hccp_test
)

function(create_ibverbs_library PRODUCT_SIDE)
    set(TARGET_NAME "ibverbs_${PRODUCT_SIDE}")
    add_library(${TARGET_NAME} SHARED
        ibverbs_stub.c
        ${CMAKE_CURRENT_SOURCE_DIR}/hccp_shm_infra.c
    )
    if(PRODUCT_SIDE STREQUAL "host")
        target_compile_definitions(${TARGET_NAME} PRIVATE IBV_STUB_HOST)
    endif()
    target_include_directories(${TARGET_NAME} PRIVATE
        ${RDMA_CORE_INCLUDE_DIR}
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base/
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_link_libraries(${TARGET_NAME} PRIVATE
        pthread
        unified_dlog_stub
    )
    set_target_properties(${TARGET_NAME} PROPERTIES
        OUTPUT_NAME "ibverbs"
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${PRODUCT_SIDE}
    )
    install(
        TARGETS ${TARGET_NAME}
        LIBRARY DESTINATION ${HCCP_TEST_DIR}/${PRODUCT_SIDE}
        COMPONENT hccp_test
    )
endfunction()

create_ibverbs_library(host)
create_ibverbs_library(device)

add_dependencies(ascend_hal_host_stub ibverbs_host ibverbs_device)

add_library(runtime_stub SHARED
    runtime_stub.c
    runtime_net_stub.c
    runtime_rt_stub.c
    ${CMAKE_CURRENT_SOURCE_DIR}/hccp_shm_infra.c
)
set_source_files_properties(runtime_net_stub.c PROPERTIES LANGUAGE CXX)
# runtime_rt_stub.c is C++ as well: tsd_client.h pulls in C++ standard headers
set_source_files_properties(runtime_rt_stub.c PROPERTIES LANGUAGE CXX)

set_target_properties(runtime_stub PROPERTIES
    OUTPUT_NAME "runtime"
    PREFIX "lib"
)

target_include_directories(runtime_stub PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/aicpu
    ${ASCEND_CANN_PACKAGE_PATH}/include/driver
    ${CMAKE_CURRENT_SOURCE_DIR}
)

install(
    TARGETS runtime_stub
    LIBRARY DESTINATION ${HCCP_TEST_DIR}/common
    COMPONENT hccp_test
)

add_library(ccu-user-drv SHARED
    ccu_u_stub.c
)

target_include_directories(ccu-user-drv PRIVATE
    ${HCCP_DIR}/external_depends/ccu
)

target_link_libraries(ccu-user-drv PRIVATE
    pthread
)

install(
    TARGETS ccu-user-drv
    LIBRARY DESTINATION ${HCCP_TEST_DIR}/device
    COMPONENT hccp_test
)


add_library(hns-rdmav17 SHARED
    hns-rdmav17_stub.c
)

target_include_directories(hns-rdmav17 PRIVATE
    ${RDMA_CORE_INCLUDE_DIR}
    ${HCCP_DIR}/external_depends/rdma-core/providers/hns
    ${HCCP_DIR}/inc/private/network
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base/
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ASCEND_CANN_PACKAGE_PATH}/include/driver/
)

target_link_libraries(hns-rdmav17 PRIVATE
    ascend_hal_device_stub
    unified_dlog_stub
)

install(
    TARGETS hns-rdmav17
    LIBRARY DESTINATION ${HCCP_TEST_DIR}/device
    COMPONENT hccp_test
)

add_library(hrn3-rdmav34 SHARED
    hrn3-rdmav34_stub.c
)

target_include_directories(hrn3-rdmav34 PRIVATE
    ${RDMA_CORE_INCLUDE_DIR}
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base/
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(hrn3-rdmav34 PRIVATE
    unified_dlog_stub
)

install(
    TARGETS hrn3-rdmav34
    LIBRARY DESTINATION ${HCCP_TEST_DIR}/device
    COMPONENT hccp_test
)

add_library(ascend_rdma_lite SHARED
    ascend_rdma_lite_stub.c
)

target_include_directories(ascend_rdma_lite PRIVATE
    ${RDMA_CORE_INCLUDE_DIR}
    ${HCCP_DIR}/external_depends/host_lite
    ${HCCP_DIR}/inc/private/network
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base/
    ${ASCEND_CANN_PACKAGE_PATH}/include/driver/
)

target_link_libraries(ascend_rdma_lite PRIVATE
    ibverbs_host
    pthread
    unified_dlog_stub
)

install(
    TARGETS ascend_rdma_lite
    LIBRARY DESTINATION ${HCCP_TEST_DIR}/common
    COMPONENT hccp_test
)

add_library(net_adapt SHARED
    net_adapt_stub.c
)

set_target_properties(net_adapt PROPERTIES
    OUTPUT_NAME          "net_adapt"
    SOVERSION            "0"
    VERSION              "0.0.0"
    PREFIX               "lib"
)

target_compile_options(net_adapt PRIVATE
    -fno-strict-aliasing -fsigned-char -fno-common -std=gnu11 -Wextra -g -O0
)

install(
    TARGETS net_adapt
    LIBRARY DESTINATION ${HCCP_TEST_DIR}/common
    COMPONENT hccp_test
)
