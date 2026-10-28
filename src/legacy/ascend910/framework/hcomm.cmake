# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 定义 hcomm 动态链接库，在 host 侧使用
add_library(hcomm SHARED)

# 编译选项
target_compile_options(hcomm PRIVATE
    -Werror
    -fno-common
    -fno-strict-aliasing
    $<$<CONFIG:Debug>:-Og -g>
    $<$<CONFIG:Release>:-O3>
)

target_include_directories(hcomm PRIVATE
    ${HCOMM_DIR}/include
    ${HCOMM_DIR}/include/hccl
    ${HCOMM_DIR}/pkg_inc
    ${HCOMM_DIR}/pkg_inc/hccl
    ${HCOMM_DIR}/pkg_inc/legacy
    ${HCOMM_DIR}/pkg_inc/legacy/hccl
    ${HCOMM_DIR}/src/legacy/ascend910/common/error_manager 
    ${HCOMM_DIR}/include/ccu

    # src/pub_inc 头文件 (legacy/ascend910)
    ${HCOMM_DIR}/src/legacy/ascend910/pub_inc
    ${HCOMM_DIR}/src/legacy/ascend910/pub_inc/inner
    ${HCOMM_DIR}/src/legacy/ascend910/pub_inc/new

    # src/framework 头文件 (legacy/ascend910)
    ${HCOMM_DIR}/src/legacy/ascend910/framework
    ${HCOMM_DIR}/src/legacy/ascend910/framework/inc
    ${HCOMM_DIR}/src/legacy/ascend910/framework/common/src
    ${HCOMM_DIR}/src/legacy/ascend910/framework/common/src/config
    ${HCOMM_DIR}/src/legacy/ascend910/framework/communicator/impl
    ${HCOMM_DIR}/src/legacy/ascend910/framework/communicator/impl/resource_manager
    ${HCOMM_DIR}/src/legacy/ascend910/framework/communicator/impl/one_sided_service
    ${HCOMM_DIR}/src/legacy/ascend910/framework/op_base/src

    # src/base_comm 头文件 (hcomm 对 base_comm 的依赖)
    ${HCOMM_DIR}/src/base_comm/common
    ${HCOMM_DIR}/src/base_comm/common/debug
    ${HCOMM_DIR}/src/base_comm/primitives/api_c_adpt
    ${HCOMM_DIR}/src/base_comm/primitives/api_c_adpt/nic_plugin
    ${HCOMM_DIR}/src/base_comm/resources/ccu/pub_inc
    ${HCOMM_DIR}/src/base_comm/resources/comm_engine_res/threads
    ${HCOMM_DIR}/src/base_comm/resources/endpoint_pairs
    ${HCOMM_DIR}/src/base_comm/resources/endpoint_pairs/channels/host
    ${HCOMM_DIR}/src/base_comm/resources/endpoint_pairs/sockets
    ${HCOMM_DIR}/src/base_comm/resources/hccp/inc
    ${HCOMM_DIR}/src/base_comm/resources/hccp/inc/network

    # src/common 头文件 (legacy/ascend910)
    ${HCOMM_DIR}/src/legacy/ascend910/common/debug/profiling/inc
    ${HCOMM_DIR}/src/legacy/ascend910/common/debug/profiling/inc/host
    ${HCOMM_DIR}/src/legacy/ascend910/common/debug/config
    ${HCOMM_DIR}/src/legacy/ascend910/common/stream
    ${HCOMM_DIR}/src/legacy/ascend910/common/launch_device
    ${HCOMM_DIR}/src/legacy/ascend910/common/launch_aicpu
    ${HCOMM_DIR}/src/legacy/ascend910/common/error_manager
    ${HCOMM_DIR}/src/legacy/ascend910/common

    # src/algorithm 头文件 (legacy/ascend910)
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/pub_inc
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/base/inc
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/base/alg_template
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/base/communicator
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/base/communicator/legacy
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/impl
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/impl/resource_manager
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/impl/task
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/impl/legacy
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/impl/coll_executor

    # src/platform 头文件 (legacy/ascend910)
    ${HCOMM_DIR}/src/legacy/ascend910/platform/inc
    ${HCOMM_DIR}/src/legacy/ascend910/platform/inc/adapter
    ${HCOMM_DIR}/src/legacy/ascend910/platform/common
    ${HCOMM_DIR}/src/legacy/ascend910/platform/resource/dispatcher_ctx

    # base_comm/resources 头文件 (原 platform/hccp)
    ${HCOMM_DIR}/src/base_comm/inc
    ${HCOMM_DIR}/src/base_comm/resources/hccp/inc
    ${HCOMM_DIR}/src/base_comm/resources/hccp/inc/network
    ${HCOMM_DIR}/src/base_comm/resources/southbound_adpt
    ${HCOMM_DIR}/src/base_comm/resources/hccp/external_depends/ubengine

    # src/legacy 头文件 (legacy/ascend950)
    ${HCOMM_DIR}/src/legacy/ascend950
    ${HCOMM_DIR}/src/legacy/ascend950/common
    ${HCOMM_DIR}/src/legacy/ascend950/common/exception
    ${HCOMM_DIR}/src/legacy/ascend950/common/types
    ${HCOMM_DIR}/src/legacy/ascend950/common/utils
    ${HCOMM_DIR}/src/legacy/ascend950/framework
    ${HCOMM_DIR}/src/legacy/ascend950/framework/ccu
    ${HCOMM_DIR}/src/legacy/ascend950/framework/communicator
    ${HCOMM_DIR}/src/legacy/ascend950/framework/communicator/aicpu
    ${HCOMM_DIR}/src/legacy/ascend950/framework/communicator/aicpu/one_sided_component
    ${HCOMM_DIR}/src/base_comm/resources/comm_engine_res/threads/cpu_thread/legacy_dpu_thread
    ${HCOMM_DIR}/src/legacy/ascend950/framework/communicator/hostdpu
    ${HCOMM_DIR}/src/legacy/ascend950/framework/dfx/aicpu/common
    ${HCOMM_DIR}/src/legacy/ascend950/framework/dfx/aicpu/profiling
    ${HCOMM_DIR}/src/legacy/ascend950/framework/dfx/common
    ${HCOMM_DIR}/src/legacy/ascend950/framework/entrance
    ${HCOMM_DIR}/src/legacy/ascend950/framework/env_config
    ${HCOMM_DIR}/src/legacy/ascend950/framework/resource_manager/buffer/aicpu
    ${HCOMM_DIR}/src/legacy/ascend950/framework/resource_manager/notify/aicpu
    ${HCOMM_DIR}/src/legacy/ascend950/framework/resource_manager/socket
    ${HCOMM_DIR}/src/legacy/ascend950/framework/resource_manager/stream
    ${HCOMM_DIR}/src/legacy/ascend950/framework/resource_manager/stream/aicpu
    ${HCOMM_DIR}/src/legacy/ascend950/framework/resource_manager/transport/aicpu
    ${HCOMM_DIR}/src/legacy/ascend950/framework/service/aicpu
    ${HCOMM_DIR}/src/legacy/ascend950/framework/topo
    ${HCOMM_DIR}/src/legacy/ascend950/framework/topo/new_topo_builder/common
    ${HCOMM_DIR}/src/legacy/ascend950/framework/topo/new_topo_builder/rank_table_info
    ${HCOMM_DIR}/src/legacy/ascend950/include
    ${HCOMM_DIR}/src/legacy/ascend950/interface
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective/alg/alg_registry
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective/alg/coll_alg_factory/alg_executor
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective/alg/coll_alg_factory/alg_executor/ins_alg_executor
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective/alg/coll_alg_factory/alg_executor/prim_alg_executor
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective/alg/coll_alg_factory/alg_template
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective/alg/coll_alg_factory/alg_template/ccu_alg_template
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective/alg/coll_alg_factory/alg_template/ins_alg_template
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective/alg/coll_alg_factory/alg_template/prim_alg_template
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective/alg/interface
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective/alg/interface/device
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective/alg/interface/host
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective/alg/selector
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective/instruction
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective/primitive
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/common
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/external_system
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/pub_inc
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/pub_inc/ccu
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/pub_inc/resource
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/pub_inc/resource/transport
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/buffer
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/buffer/aicpu
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/connection/aicpu
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/notify/aicpu
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/stream
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/stream/aicpu
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/transport
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/transport/aicpu

    # coll_communicator_mgr headers needed transitively by legacy compile units (centralized here;
    # coll_comm sources now compile in hcomm_coll_comm_obj with their own -I whitelist)
    ${HCOMM_DIR}/src/coll_communicator_mgr/api_c_adpt
    ${HCOMM_DIR}/src/coll_communicator_mgr/common
    ${HCOMM_DIR}/src/coll_communicator_mgr/communicator
    ${HCOMM_DIR}/src/coll_communicator_mgr/communicator/group_schedule_mgr
    ${HCOMM_DIR}/src/coll_communicator_mgr/communicator/op_group_queue_mgr
    ${HCOMM_DIR}/src/coll_communicator_mgr/config_mgr
    ${HCOMM_DIR}/src/coll_communicator_mgr/dfx
    ${HCOMM_DIR}/src/coll_communicator_mgr/dfx/cluster_monitor
    ${HCOMM_DIR}/src/coll_communicator_mgr/dfx/inconsistentCheck
    ${HCOMM_DIR}/src/coll_communicator_mgr/dfx/profiling/host
    ${HCOMM_DIR}/src/coll_communicator_mgr/rank_graph
    ${HCOMM_DIR}/src/coll_communicator_mgr/rank_graph/rank_graph
    ${HCOMM_DIR}/src/coll_communicator_mgr/resource_mgr/local/my_rank
    ${HCOMM_DIR}/src/coll_communicator_mgr/resource_mgr/local/my_rank/comm_engine
    ${HCOMM_DIR}/src/coll_communicator_mgr/resource_mgr/local/my_rank/comm_engine/engine_ctxs
    ${HCOMM_DIR}/src/coll_communicator_mgr/resource_mgr/local/my_rank/comm_engine/notify
    ${HCOMM_DIR}/src/coll_communicator_mgr/resource_mgr/local/my_rank/comm_engine/threads
    ${HCOMM_DIR}/src/coll_communicator_mgr/resource_mgr/local/my_rank/comm_engine/threads/order_launch_thread_mgr
    ${HCOMM_DIR}/src/coll_communicator_mgr/resource_mgr/local/my_rank/comm_mems
    ${HCOMM_DIR}/src/coll_communicator_mgr/resource_mgr/remote/rank_pairs

    # 三方件头文件
    ${URMA_INCLUDE_DIR}
)

if(BUILD_OPEN_PROJECT)
    target_compile_definitions(hcomm PRIVATE
        OPEN_BUILD_PROJECT
        LOG_CPP
    )

    target_link_libraries(hcomm
        $<BUILD_INTERFACE:intf_pub>
        $<BUILD_INTERFACE:error_manager_headers>
        $<BUILD_INTERFACE:acl_rt_headers>
        $<BUILD_INTERFACE:asc_host_headers>
        $<BUILD_INTERFACE:ascend_hal_headers>
        $<BUILD_INTERFACE:kernel_tiling_headers>
        $<BUILD_INTERFACE:atrace_headers>
        $<BUILD_INTERFACE:mmpa_headers>
        $<BUILD_INTERFACE:runtime_headers>
        $<BUILD_INTERFACE:rdma_core_headers>
        $<BUILD_INTERFACE:hccl_legacy_headers>
        $<BUILD_INTERFACE:json>
        -Wl,--no-as-needed
        c_sec
        unified_dlog
        acl_rt
        hccl_alg
        hccl_plf
        hccl_v2
        ccl_dpu
        -Wl,--as-needed
        -lrt
        -ldl
        -lpthread
    )
else()
    target_include_directories(hcomm PRIVATE
        ${TOP_DIR}/abl/msprof/inc
        ${TOP_DIR}/ace/npuruntime/inc
        ${TOP_DIR}/inc/driver
        ${TOP_DIR}/inc/
        ${TOP_DIR}/open_source/json/include
        ${TOP_DIR}/ace/npuruntime/inc/runtime
        ${TOP_DIR}/runtime/include/external
        ${TOP_DIR}/abl/msprof/inc/toolchain
        ${TOP_DIR}/inc/aicpu
    )
    target_link_libraries(hcomm
        $<BUILD_INTERFACE:intf_pub>
        $<BUILD_INTERFACE:kernel_tiling_headers>
        -Wl,--no-as-needed
        c_sec
        unified_dlog
        mmpa
        hccl_alg
        hccl_plf
        hccl_v2
        ccl_dpu
        -Wl,--as-needed
        -lrt
        -ldl
        -lpthread
    )
endif()

# 符号隐藏: 仅导出白名单符号; 非 release 模式依赖动态符号, 不挂载
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_link_options(hcomm PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_LIST_DIR}/hcomm.map")
    set_property(TARGET hcomm APPEND PROPERTY LINK_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/hcomm.map")
endif()

# 安装 hcomm 库
install(TARGETS hcomm
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} ${INSTALL_OPTIONAL}
    COMPONENT hcomm
)
