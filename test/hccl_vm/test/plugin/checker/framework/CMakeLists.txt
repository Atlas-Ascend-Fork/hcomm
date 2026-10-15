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
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/semantics_check
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/task_graph_generator_v3
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/header/external
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/header/external/nlohmann
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/header/internal
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/ccu_executor
    ${CMAKE_SOURCE_DIR}/src/plugin/runner/runner_utils
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/store
    ${CMAKE_SOURCE_DIR}/third_party/CLI11
    ${YAMLCPP_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/third_party/nlohmann_json
    ${CMAKE_SOURCE_DIR}/third_party/spdlog
    ${CMAKE_SOURCE_DIR}/third_party/spdlog/include
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base
    ${HCOMM_CODE_ROOT}/src/base_comm/resources/ccu/ccu_microcode
)

set(CHECKER_LOG_SRC ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/log/log_stub.cc)
set(CHECKER_LIBC_SEC ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/lib64/libc_sec.a)
set(CHECKER_OUTPUT_DIR ${CMAKE_BINARY_DIR}/output/bin)
set(CHECKER_DUMP_UTILS_SRC
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils/dump/dump_json_utils.cc
)
set(CHECKER_UTILS_SRC
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils/check_utils.cc
)

link_libraries(
    gtest
    log
    ${CHECKER_LIBC_SEC}
)

function(configure_checker_test TEST_NAME)
    set_target_properties(${TEST_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CHECKER_OUTPUT_DIR}"
    )
endfunction()

set(CHECKER_SEMANTICS_TESTS
    allgather_semantics_checker
    allgather_v_semantics_checker
    allreduce_semantics_checker
    broadcast_semantics_checker
    reduce_semantics_checker
    scatter_semantics_checker
    send_recv_semantics_checker
    all2all_semantics_checker
    reduce_scatter_semantics_checker
    reduce_scatter_v_semantics_checker
    batchsendrecv_semantics_checker
)
set(CHECKER_SEMANTICS_DUMP_TESTS
    allreduce_semantics_checker
    reduce_semantics_checker
    reduce_scatter_semantics_checker
    reduce_scatter_v_semantics_checker
)

foreach(TEST_NAME IN LISTS CHECKER_SEMANTICS_TESTS)
    add_executable(test_${TEST_NAME} semantics_check/${TEST_NAME}_test.cc main.cc)
    target_sources(test_${TEST_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/semantics_check/${TEST_NAME}.cc
        ${CHECKER_UTILS_SRC}
        ${CHECKER_LOG_SRC}
    )
    if(TEST_NAME IN_LIST CHECKER_SEMANTICS_DUMP_TESTS)
        target_sources(test_${TEST_NAME} PRIVATE ${CHECKER_DUMP_UTILS_SRC})
    endif()
    configure_checker_test(test_${TEST_NAME})
endforeach()

add_executable(test_task_graph_sync_conflict_v3
    task_graph_generator_v3/task_graph_sync_conflict_v3_test.cc
    main.cc
)
target_sources(test_task_graph_sync_conflict_v3 PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/task_graph_generator_v3/task_graph_sync_conflict_v3.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/task_graph_generator_v3/task_def_v3.cc
    ${CHECKER_LOG_SRC}
)
target_include_directories(test_task_graph_sync_conflict_v3 BEFORE PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/task_graph_generator_v3
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/log
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/header/internal
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/header/external
)
target_link_libraries(test_task_graph_sync_conflict_v3 PRIVATE ${CHECKER_LIBC_SEC})
configure_checker_test(test_task_graph_sync_conflict_v3)

add_executable(test_ccu_loop_parallel_boundary_v3
    task_graph_generator_v3/ccu_loop_parallel_boundary_v3_test.cc
    main.cc
)
target_sources(test_ccu_loop_parallel_boundary_v3 PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/task_graph_generator_v3/task_def_v3.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/task_graph_generator_v3/task_graph_generator_v3.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/task_graph_generator_v3/ccu_graph_generator_v3/ccu_task_common_v3.cc
    ${CHECKER_LOG_SRC}
)
target_include_directories(test_ccu_loop_parallel_boundary_v3 BEFORE PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/task_graph_generator_v3
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/log
)
target_compile_options(test_ccu_loop_parallel_boundary_v3 PRIVATE -ffunction-sections -fdata-sections)
target_link_options(test_ccu_loop_parallel_boundary_v3 PRIVATE -Wl,--gc-sections)
target_link_libraries(test_ccu_loop_parallel_boundary_v3 PRIVATE ${CHECKER_LIBC_SEC})
configure_checker_test(test_ccu_loop_parallel_boundary_v3)

add_executable(test_task_graph_mem_conflict_v3
    task_graph_generator_v3/task_graph_mem_conflict_v3_test.cc
    main.cc
)
target_sources(test_task_graph_mem_conflict_v3 PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/task_graph_generator_v3/task_graph_mem_conflict_v3.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/task_graph_generator_v3/task_graph_reachability_v3.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/task_graph_generator_v3/task_graph_reachability_shard_v3.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/task_graph_generator_v3/task_def_v3.cc
    ${CHECKER_LOG_SRC}
)
target_include_directories(test_task_graph_mem_conflict_v3 BEFORE PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/task_graph_generator_v3
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/log
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/header/internal
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/header/external
)
target_link_libraries(test_task_graph_mem_conflict_v3 PRIVATE ${CHECKER_LIBC_SEC})
configure_checker_test(test_task_graph_mem_conflict_v3)

add_executable(test_task_graph_reachability_shard_stress
    task_graph_generator_v3/task_graph_reachability_shard_stress_test.cc
    main.cc
)
target_sources(test_task_graph_reachability_shard_stress PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/task_graph_generator_v3/task_graph_reachability_v3.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/task_graph_generator_v3/task_graph_reachability_shard_v3.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/task_graph_generator_v3/task_def_v3.cc
    ${CHECKER_LOG_SRC}
)
target_include_directories(test_task_graph_reachability_shard_stress BEFORE PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/task_graph_generator_v3
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/log
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/header/internal
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/header/external
)
target_link_libraries(test_task_graph_reachability_shard_stress PRIVATE ${CHECKER_LIBC_SEC})
configure_checker_test(test_task_graph_reachability_shard_stress)

add_executable(test_composite_op_grouping
    composite_op_grouping_test.cc
    main.cc
)
target_sources(test_composite_op_grouping PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/composite_op_grouping.cc
    ${CHECKER_LOG_SRC}
)
target_link_libraries(test_composite_op_grouping PRIVATE ${CHECKER_LIBC_SEC})
configure_checker_test(test_composite_op_grouping)

add_executable(test_big_graph_checker_v3
    big_graph_check/big_graph_checker_v3_test.cc
    big_graph_check/real_task_orchestration_sync_test.cc
    big_graph_check/real_ccu_graph_sync_test.cc
    big_graph_check/real_aiv_graph_sync_test.cc
    main.cc
)
target_sources(test_big_graph_checker_v3 PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/task_graph_generator_v3/task_graph_sync_conflict_v3.cc
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/task_graph_generator_v3/task_def_v3.cc
    ${CHECKER_LOG_SRC}
)
target_include_directories(test_big_graph_checker_v3 BEFORE PRIVATE
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/framework/task_graph_generator_v3
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/utils
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/src/log
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/header/internal
    ${CMAKE_SOURCE_DIR}/src/plugin/checker/header/external
)
target_link_libraries(test_big_graph_checker_v3 PRIVATE ${CHECKER_LIBC_SEC})
configure_checker_test(test_big_graph_checker_v3)
