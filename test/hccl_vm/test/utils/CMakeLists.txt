# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -D_GLIBCXX_USE_CXX11_ABI=0")

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/utils
    ${CMAKE_SOURCE_DIR}/third_party
    ${CMAKE_SOURCE_DIR}/third_party/nlohmann_json
    ${CMAKE_SOURCE_DIR}/third_party/yaml-cpp/include
    ${CMAKE_SOURCE_DIR}/third_party/spdlog
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
)

link_libraries(
    gtest_main
    gtest
)

# 共用少量功能用例，按启动环境分别验证 stub、host cann 和 device cann。
foreach(target test_sim_log test_log_host test_log_device)
    add_executable(${target} sim_log_test.cc ${HCCLVM_LOG_SOURCES})
    target_link_libraries(${target} PRIVATE pthread yaml-cpp::yaml-cpp ${CMAKE_DL_LIBS})
    set_target_properties(${target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin")
endforeach()

add_library(test_log_dlog SHARED log_dlog_stub.cc)
set_target_properties(test_log_dlog PROPERTIES OUTPUT_NAME unified_dlog
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/cann")
add_dependencies(test_log_host test_log_dlog)
target_sources(test_log_host PRIVATE ${HCCL_VM_SRC}/proxy/level2/log_stub.cc)
target_compile_definitions(test_log_host PRIVATE HCCLVM_TEST_CANN HCCLVM_TEST_CANN_HOST
    HCCL_VM_MODULE="COMM_STUB")
target_compile_definitions(test_log_device PRIVATE HCCLVM_TEST_CANN HCCLVM_LOG_PROCESS_ROLE_DEVICE=1
    HCCL_VM_MODULE="COMM_STUB")

add_test(NAME log_stub COMMAND ${CMAKE_COMMAND} -E env
    "HCCL_VM_LOG_MODE=stub" "LD_PRELOAD=" "TEST_LOG_LEVEL=2" $<TARGET_FILE:test_sim_log>)
# 后端和等级在启动时确定，使用独立进程验证 INFO、默认 ERROR、模块覆盖及关闭。
foreach(role host device)
    foreach(mode info default module off)
        set(level_args "ASCEND_GLOBAL_LOG_LEVEL=1" "ASCEND_MODULE_LOG_LEVEL=" "TEST_LOG_LEVEL=2")
        if(mode STREQUAL "default")
            set(level_args --unset=ASCEND_GLOBAL_LOG_LEVEL "ASCEND_MODULE_LOG_LEVEL=" "TEST_LOG_LEVEL=4")
        elseif(mode STREQUAL "module")
            set(level_args "ASCEND_GLOBAL_LOG_LEVEL=3" "ASCEND_MODULE_LOG_LEVEL=HCCL=1" "TEST_LOG_LEVEL=2")
        elseif(mode STREQUAL "off")
            set(level_args "ASCEND_GLOBAL_LOG_LEVEL=4" "ASCEND_MODULE_LOG_LEVEL=" "TEST_LOG_LEVEL=6")
        endif()
        add_test(NAME log_${role}_${mode} COMMAND ${CMAKE_COMMAND} -E env
            "HCCL_VM_LOG_MODE=cann" "LD_PRELOAD="
            "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:test_log_dlog>:$ENV{LD_LIBRARY_PATH}"
            ${level_args} $<TARGET_FILE:test_log_${role}>)
        set_tests_properties(log_${role}_${mode} PROPERTIES TIMEOUT 20)
    endforeach()
endforeach()
set_tests_properties(log_stub PROPERTIES TIMEOUT 20)
