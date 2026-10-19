# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(hccp_test_list
    main.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../stub/hccp_vm_net.c
)

add_executable(hccp_test.bin ${hccp_test_list})

target_include_directories(hccp_test.bin PRIVATE
    ${GTEST_INCLUDE}
    ${CMAKE_CURRENT_SOURCE_DIR}/../stub
)

target_compile_options(hccp_test.bin PRIVATE
    -O0 -g
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++14>
    -Wall -Wextra -Werror -Wshadow
    -D_GLIBCXX_USE_CXX11_ABI=0
    # No -pedantic: the TEST_LOG_* macros rely on the GNU ", ## __VA_ARGS__" extension and
    # the CANN driver headers use C idioms (zero-size arrays, anonymous structs); both are
    # rejected by -pedantic. Deprecated runtime APIs are intentionally exercised by the
    # cases, so the deprecation attribute carries no meaning for this test target.
    -Wno-deprecated-declarations
)

target_link_options(hccp_test.bin PRIVATE
    -Wl,--no-undefined
    -rdynamic
)

target_link_libraries(hccp_test.bin PRIVATE
    GTest::gtest
    ascend_hal_host_stub
    -lpthread
    -lrt
    -ldl
)

set_target_properties(hccp_test.bin PROPERTIES
    INSTALL_RPATH "$ORIGIN/../host;$ORIGIN/../common"
    BUILD_WITH_INSTALL_RPATH TRUE
)

install(
    TARGETS hccp_test.bin
    RUNTIME DESTINATION ${HCCP_TEST_DIR}/host
    COMPONENT hccp_test
)

add_library(hccp_case SHARED
    cases_common.cc
    cases_socket.cc
    cases_socket_hlt.cc
    cases_pingmesh.cc
    cases_rdma.cc
    cases_udma.cc
    cases_ub_ctrl.cc
    cases_ub_dataplane.cc
    cases_ccu_smoke.cc
    cases_ccu.cc
    cases_ccu_driver.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../stub/hccp_vm_net.c
)

target_include_directories(hccp_case PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HCCP_DIR}/inc
    ${HCCP_DIR}/inc/network
    ${HCCP_DIR}/external_depends/ccu
    ${CMAKE_CURRENT_SOURCE_DIR}/../stub
    ${GTEST_INCLUDE}
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${RDMA_CORE_INCLUDE_DIR}
)

target_compile_options(hccp_case PRIVATE
    -U_FORTIFY_SOURCE
    -O0 -g
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++14>
    -Wall
    -Wextra
    -Werror
    -Wshadow
    -D_GLIBCXX_USE_CXX11_ABI=0
    # No -pedantic: the TEST_LOG_* macros rely on the GNU ", ## __VA_ARGS__" extension and
    # the CANN driver headers use C idioms (zero-size arrays, anonymous structs); both are
    # rejected by -pedantic. Deprecated runtime APIs are intentionally exercised by the
    # cases, so the deprecation attribute carries no meaning for this test target.
    -Wno-deprecated-declarations
)

# hccp_case only consumes gtest headers (${GTEST_INCLUDE}) without linking GTest::gtest,
# so the ExternalProject install of gtest has no order edge to this target: on a cold
# gtest cache the parallel build may compile cases_*.cc before gtest/gtest.h is installed.
# Declare the dependency explicitly (hccp_test.bin is already covered by linking GTest::gtest).
add_dependencies(hccp_case GTest::gtest)

target_link_libraries(hccp_case PRIVATE
    test_ra
    runtime_stub
    -lpthread
)

set_target_properties(hccp_case PROPERTIES
    INSTALL_RPATH "$ORIGIN/../host;$ORIGIN/../common"
    BUILD_WITH_INSTALL_RPATH TRUE
)

install(
    TARGETS hccp_case
    LIBRARY DESTINATION ${HCCP_TEST_DIR}/lib
    COMPONENT hccp_test
)

add_dependencies(hccp_test.bin test_ra hccp_case)
