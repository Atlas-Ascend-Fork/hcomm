# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# ODR 整改回归测试：编译级测试，验证整改对的两个同名头文件可在同一 TU 同时包含而不重复定义。
# 编译通过即证明 namespace/guard 分离生效，ODR 隐患已消除。

add_executable(hccl_utest_odr_coexist
    odr_header_coexist_test.cc
)

target_include_directories(hccl_utest_odr_coexist PRIVATE
    ${UT_COMMON_INCLUDE_LIST}
    # coll_comm 侧整改头文件所在目录
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/topo_info
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/phy_topo_builder
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/rank_table_info
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/common
    ${HCCL_CODE_ROOT}/src/coll_communicator_mgr/rank_graph/phy_topo
    # legacy950 侧整改头文件所在目录（改名为 _legacy950.h）
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/framework/topo/new_topo_builder/topo_info
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/framework/topo/new_topo_builder/phy_topo_builder
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/framework/topo/new_topo_builder/rank_table_info
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/framework/topo/new_topo_builder/common
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/framework/topo/new_topo_builder/phy_topo
    ${HCCL_CODE_ROOT}/src/legacy/ascend950/common/types
    # legacy910 侧组8 两份 zero_copy_address_mgr
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/communicator/impl/zero_copy
    ${HCCL_CODE_ROOT}/src/legacy/ascend910/framework/device/framework
)

target_compile_definitions(hccl_utest_odr_coexist PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
    HALF_T=float
    google=ascend_private
    CCL_KERNEL
    CCL_LLT
    OPEN_HCCL_TEST
    OPEN_BUILD_PROJECT
)

target_compile_options(hccl_utest_odr_coexist PRIVATE
    -U_FORTIFY_SOURCE
    -fno-var-tracking-assignments
    -Os
    -O0 -g --coverage -fprofile-arcs -ftest-coverage -fno-access-control
    # offsetof 用于非 standard-layout 类（序列化类含 std::string/vector/map，C++17 条件支持），
    # 显式抑制 -Winvalid-offsetof 告警，不依赖全局 -w（避免后续加 -Werror 断编译）。
    -Wno-invalid-offsetof
)

target_link_libraries(hccl_utest_odr_coexist PRIVATE
    intf_llt_pub
    rdma_core_headers
    hccl_llt
    -pthread
    -lgcov
    -lrt
    -ldl
)
add_dependencies(hccl_utest_odr_coexist hccl_llt)

enable_testing()
add_test(NAME hccl_utest_odr_coexist COMMAND hccl_utest_odr_coexist)
