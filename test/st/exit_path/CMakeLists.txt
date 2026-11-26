#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -
#Copyright(c) 2026 Huawei Technologies Co., Ltd.
#This program is free software, you can redistribute it and / or modify it under the terms and conditions of
#CANN Open Software License Agreement Version 2.0(the "License").
#Please refer to the License for details.You may not use this file except in compliance with the License.
#THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
#INCLUDING BUT NOT LIMITED TO NON - INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
#See LICENSE in the root of the software repository for the full text of the License.
#-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -

#退出期专项用例：独立 target，不并入 gtest 主二进制，
#避免 "不清理直接退出"的行为干扰 gtest 用例进程。
#
#链接 hccl_llt（test / ut / stub 构建的全量源码 + 打桩动态库，含 HcommResMgrInit 与锚定层全部单例），
#因此当前随 `bash build.sh - u`（ENABLE_UT）构建并经 ctest 运行；ST / CI 接线为后续工作点。
#
#覆盖点（双 profile）：
#- exit_without_cleanup ：不清理直接退出（兜底析构链，T1 既有）；
#- exit_graceful_deinit ：显式 DeInit(EXIT_PROFILE_GRACEFUL) 后退出 + 幂等验证；
#- exit_reset_deinit    ：显式 DeInit(EXIT_PROFILE_RESET) 后退出 + 幂等验证。

if (NOT TARGET hccl_llt)
    message(STATUS "exit_path test skipped: target hccl_llt not available (build with -u/--ut)")
    return()
endif()

set(exit_path_tests exit_without_cleanup exit_graceful_deinit exit_reset_deinit)

foreach (test_name IN LISTS exit_path_tests)
    add_executable(${test_name} ${test_name}.cc)

    target_include_directories(${test_name} PRIVATE
        ${HCOMM_DIR}/include
        ${HCOMM_DIR}/pkg_inc
        ${HCOMM_DIR}/pkg_inc/hccl
        ${HCOMM_DIR}/pkg_inc/legacy
        ${HCOMM_DIR}/src/base_comm/primitives/api_c_adpt
        ${HCOMM_DIR}/src/base_comm
        ${HCOMM_DIR}/src/legacy/ascend910/pub_inc
        ${UT_COMMON_INCLUDE_LIST}
        ${HCOMM_RES_TREE_INCLUDES}
        ${HCOMM_DIR}/test/stub/depends/include
        ${HCOMM_DIR}/test/stub/depends/include/urma)

    target_compile_definitions(${test_name} PRIVATE _GLIBCXX_USE_CXX11_ABI=0 google=ascend_private)

    target_compile_options(${test_name} PRIVATE
        $<$<STREQUAL:${ENABLE_ASAN},true>:-fsanitize=address -fsanitize-recover=address,all -fno-omit-frame-pointer>)

    # intf_llt_pub / rdma_core_headers：与 test/ut/framework/* 同款接口目标。
    # 缺 rdma_core_headers 时 hccp_common.h -> <infiniband/verbs.h> 找不到。
    target_link_libraries(${test_name} PRIVATE
        $<BUILD_INTERFACE:intf_llt_pub>
        $<BUILD_INTERFACE:rdma_core_headers>
        $<$<STREQUAL:${ENABLE_ASAN},true>:-fsanitize=address>
        hccl_llt
        -pthread
        -lrt
        -ldl)

    add_dependencies(${test_name} hccl_llt)

    add_test(NAME ${test_name} COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/run_exit_path_test.sh $<TARGET_FILE:${test_name}>)

    # 退出路径回归若死锁不能无限阻塞 ctest / CI
    set_tests_properties(${test_name} PROPERTIES TIMEOUT 300)
endforeach()
