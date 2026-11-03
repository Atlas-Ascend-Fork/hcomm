# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(CHECKER_SRC ${CMAKE_SOURCE_DIR}/src/plugin/checker/src)
set(CHECKER_HDR ${CMAKE_SOURCE_DIR}/src/plugin/checker/header)

include_directories(BEFORE
    ${CHECKER_SRC}
    ${CHECKER_SRC}/utils/dump
    ${CHECKER_SRC}/utils
    ${CHECKER_SRC}/checker
    ${CHECKER_SRC}/log
    ${CHECKER_SRC}/framework/semantics_check
    ${CHECKER_SRC}/framework/task_graph_generator_v3
    ${CHECKER_SRC}/dfx
    ${CHECKER_HDR}/internal
    ${CHECKER_HDR}/external
    ${CHECKER_HDR}/external/nlohmann
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/runner_utils
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/include
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/pkg_inc/hccl

    ${HCOMM_CODE_ROOT}/src/base_comm/resources/ccu/ccu_microcode
)

add_executable(test_check_utils check_utils_test.cc main.cc)
target_sources(test_check_utils PRIVATE
    ${CHECKER_SRC}/utils/check_utils.cc
    ${CHECKER_SRC}/log/log_stub.cc
)
target_link_libraries(test_check_utils PRIVATE
    log
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/lib64/libc_sec.a
)
set_target_properties(test_check_utils PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_executable(test_ccu_resource_stats_dfx dfx/ccu_resource_stats_dfx_test.cc main.cc)
target_sources(test_ccu_resource_stats_dfx PRIVATE
    ${CHECKER_SRC}/dfx/ccu_resource_stats_dfx.cc
    ${CHECKER_SRC}/dfx/ccu_resource_stats_reporter.cc
    ${CHECKER_SRC}/log/log_stub.cc
)
target_link_libraries(test_ccu_resource_stats_dfx PRIVATE
    log
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/lib64/libc_sec.a
)
set_target_properties(test_ccu_resource_stats_dfx PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_executable(test_binary_data_operator_checker binary_data_operator_checker_test.cc main.cc)
target_sources(test_binary_data_operator_checker PRIVATE
    ${CHECKER_SRC}/utils/binary_data_operator.cc
    ${CHECKER_SRC}/log/log_stub.cc
)
target_link_libraries(test_binary_data_operator_checker PRIVATE
    log
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/lib64/libc_sec.a
)
set_target_properties(test_binary_data_operator_checker PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_executable(test_storage_manager_checker storage_manager_checker_test.cc main.cc)
target_sources(test_storage_manager_checker PRIVATE
    ${CHECKER_SRC}/utils/storage_manager.cc
    ${CHECKER_SRC}/utils/binary_data_operator.cc
    ${CHECKER_SRC}/log/log_stub.cc
)
target_include_directories(test_storage_manager_checker BEFORE PRIVATE
    ${CHECKER_SRC}/utils
    ${CHECKER_HDR}/internal
    ${CHECKER_HDR}/external
    ${CHECKER_HDR}/external/nlohmann
    ${THRID_PARTY_DIR}/nlohmann_json
    ${THRID_PARTY_DIR}
    ${THRID_PARTY_DIR}/yaml-cpp/include
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/include
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/pkg_inc/hccl

    ${HCOMM_CODE_ROOT}/src/base_comm/resources/ccu/ccu_microcode
)
get_target_property(_storage_mgr_includes test_storage_manager_checker INCLUDE_DIRECTORIES)
list(FILTER _storage_mgr_includes EXCLUDE REGEX "^${CMAKE_SOURCE_DIR}/include$")
set_target_properties(test_storage_manager_checker PROPERTIES INCLUDE_DIRECTORIES "${_storage_mgr_includes}")
target_include_directories(test_storage_manager_checker PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_storage_manager_checker PRIVATE
    log
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/lib64/libc_sec.a
)
set_target_properties(test_storage_manager_checker PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)

add_subdirectory(framework)
add_subdirectory(st)
