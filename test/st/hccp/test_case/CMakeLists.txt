# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_library(hccp_verdict STATIC hccp_verdict.cc)
target_compile_options(hccp_verdict PRIVATE
    -U_FORTIFY_SOURCE -O2 -g -std=c++14 -Wall -Wextra -Werror -Wshadow -D_GLIBCXX_USE_CXX11_ABI=0
)
target_link_libraries(hccp_verdict PUBLIC -lpthread)

add_executable(hccp_verdict_test.bin hccp_verdict_test.cc)
target_include_directories(hccp_verdict_test.bin PRIVATE ${GTEST_INCLUDE} ${CMAKE_CURRENT_SOURCE_DIR}/../stub)
target_compile_options(hccp_verdict_test.bin PRIVATE
    -U_FORTIFY_SOURCE -O2 -g -std=c++14 -Wall -Wextra -Werror -Wshadow -D_GLIBCXX_USE_CXX11_ABI=0
)
target_link_libraries(hccp_verdict_test.bin PRIVATE hccp_verdict GTest::gtest)
# Inject system-call failures only into the framework self-test executable.
foreach(call pipe dup dup2 pthread_create pthread_join read write opendir readdir closedir lstat fopen fclose)
    target_link_options(hccp_verdict_test.bin PRIVATE "-Wl,--wrap=${call}")
endforeach()
add_test(NAME hccp_verdict COMMAND hccp_verdict_test.bin)
set_tests_properties(hccp_verdict PROPERTIES TIMEOUT 30)
install(TARGETS hccp_verdict_test.bin RUNTIME DESTINATION ${HCCP_TEST_DIR}/host COMPONENT hccp_test)

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
    -U_FORTIFY_SOURCE
    -O2 -g
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++14>
    -Wall -Wextra -Werror -Wshadow
    -D_GLIBCXX_USE_CXX11_ABI=0
    # No -pedantic: the TEST_LOG_* macros rely on the GNU ", ## __VA_ARGS__" extension and
    # the CANN driver headers use C idioms (zero-size arrays, anonymous structs); both are
    # rejected by -pedantic. The cases call the modern aclrt* runtime entries; the one
    # retained deprecated rt call (rtNotifyWait in cases_ub_dataplane.cc, kept because its
    # aclrt successor's wait-and-reset semantics break the sim's notify model) carries its
    # own scoped #pragma, so no target-wide deprecation suppression is needed.
)

target_link_options(hccp_test.bin PRIVATE
    -Wl,--no-undefined
    -rdynamic
)

target_link_libraries(hccp_test.bin PRIVATE
    hccp_verdict
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
    -O2 -g
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++14>
    -Wall
    -Wextra
    -Werror
    -Wshadow
    -D_GLIBCXX_USE_CXX11_ABI=0
    # No -pedantic: the TEST_LOG_* macros rely on the GNU ", ## __VA_ARGS__" extension and
    # the CANN driver headers use C idioms (zero-size arrays, anonymous structs); both are
    # rejected by -pedantic. The cases call the modern aclrt* runtime entries; the one
    # retained deprecated rt call (rtNotifyWait in cases_ub_dataplane.cc, kept because its
    # aclrt successor's wait-and-reset semantics break the sim's notify model) carries its
    # own scoped #pragma, so no target-wide deprecation suppression is needed.
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
