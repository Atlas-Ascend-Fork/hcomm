# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# hcomm_coll_comm_obj: standalone compile isolation layer for coll_communicator_mgr sources (linked into hcomm).
# Purpose: move coll_comm -I dependency on base_comm internal dirs out of hcomm.cmake into this whitelist;
# hcomm.cmake keeps only legacy-side needs; this whitelist is the coll_comm dependency ledger,
# governance goal is to shrink it line by line down to base_comm/inc.
#
# NOTE(inc/): the boundary dir base_comm/inc is NOT listed here; it is provided to all targets via the directory-level
# include_directories() in CMakeLists.txt / cmake/device/CMakeLists.txt (shared by base_comm/coll/legacy/test/device).
# So the boundary headers are currently repo-global and this ledger cannot measure boundary consumers. Exit plan:
# attach base_comm/inc via target_include_directories to explicit consumers (hcomm_coll_comm_obj, hcomm_base_obj,
# ccl_kernel/hccl_v2, related UT/ST) and register it here.

add_library(hcomm_coll_comm_obj OBJECT)

# compile options (aligned with hcomm; OBJECT lib merged into SHARED needs explicit -fPIC)
target_compile_options(hcomm_coll_comm_obj PRIVATE
    -Werror
    -fno-common
    -fno-strict-aliasing
    -fPIC
    $<$<CONFIG:Debug>:-Og -g>
    $<$<CONFIG:Release>:-O3>
)

target_compile_definitions(hcomm_coll_comm_obj PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:_GLIBCXX_USE_CXX11_ABI=0>
)

if(BUILD_OPEN_PROJECT)
    target_compile_definitions(hcomm_coll_comm_obj PRIVATE
        OPEN_BUILD_PROJECT
        LOG_CPP
    )
endif()

# include paths: actual current needs of coll_comm compile units (per dependency ledger analysis)

target_include_directories(hcomm_coll_comm_obj PRIVATE
    # public and inter-package headers
    ${HCOMM_DIR}/include
    ${HCOMM_DIR}/include/ccu
    ${HCOMM_DIR}/include/hccl
    ${HCOMM_DIR}/pkg_inc
    ${HCOMM_DIR}/pkg_inc/hccl
    ${HCOMM_DIR}/pkg_inc/hcomm
    ${HCOMM_DIR}/pkg_inc/legacy
    ${HCOMM_DIR}/pkg_inc/legacy/hccl
    
    # base_comm dependency ledger

    # hccp internal headers (coll_comm is allowed to depend on resources/hccp/inc)
    ${HCOMM_DIR}/src/base_comm/resources/hccp/inc
    ${HCOMM_DIR}/src/base_comm/resources/hccp/inc/network
    ${HCOMM_DIR}/src/base_comm/resources/hccp/external_depends/aubdfx

    # southbound adapter (hosts hcomm_adapter_hccp.h) — not hccp; coll depended on it in mainline
    ${HCOMM_DIR}/src/base_comm/resources/southbound_adpt

    # coll_communicator_mgr itself (source dir + all subdirs)
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/api_c_adpt
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/communicator
    ${CMAKE_CURRENT_SOURCE_DIR}/communicator/group_schedule_mgr
    ${CMAKE_CURRENT_SOURCE_DIR}/config_mgr
    ${CMAKE_CURRENT_SOURCE_DIR}/dfx
    ${CMAKE_CURRENT_SOURCE_DIR}/dfx/cluster_monitor
    ${CMAKE_CURRENT_SOURCE_DIR}/dfx/profiling/host
    ${CMAKE_CURRENT_SOURCE_DIR}/dfx/taskException/host
    ${CMAKE_CURRENT_SOURCE_DIR}/rank_graph
    ${CMAKE_CURRENT_SOURCE_DIR}/rank_graph/common
    ${CMAKE_CURRENT_SOURCE_DIR}/rank_graph/phy_topo
    ${CMAKE_CURRENT_SOURCE_DIR}/rank_graph/phy_topo_builder
    ${CMAKE_CURRENT_SOURCE_DIR}/rank_graph/rank_graph
    ${CMAKE_CURRENT_SOURCE_DIR}/rank_graph/rank_table_info
    ${CMAKE_CURRENT_SOURCE_DIR}/rank_graph/topo_info
    ${CMAKE_CURRENT_SOURCE_DIR}/resource_mgr/local/my_rank
    ${CMAKE_CURRENT_SOURCE_DIR}/resource_mgr/local/my_rank/comm_engine
    ${CMAKE_CURRENT_SOURCE_DIR}/resource_mgr/local/my_rank/comm_engine/engine_ctxs
    ${CMAKE_CURRENT_SOURCE_DIR}/resource_mgr/local/my_rank/comm_engine/notify
    ${CMAKE_CURRENT_SOURCE_DIR}/resource_mgr/local/my_rank/comm_engine/threads
    ${CMAKE_CURRENT_SOURCE_DIR}/resource_mgr/local/my_rank/comm_engine/threads/order_launch_thread_mgr
    ${CMAKE_CURRENT_SOURCE_DIR}/resource_mgr/local/my_rank/comm_mems
    ${CMAKE_CURRENT_SOURCE_DIR}/resource_mgr/remote/rank_pairs
    ${CMAKE_CURRENT_SOURCE_DIR}/team/hccl
    ${CMAKE_CURRENT_SOURCE_DIR}/team/hcomm

    # legacy/ascend910 (needed transitively via legacy headers)
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/base/alg_template
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/base/communicator
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/base/communicator/legacy
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/base/inc
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/impl
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/impl/coll_executor
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/impl/legacy
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/impl/resource_manager
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/impl/task
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/pub_inc
    ${HCOMM_DIR}/src/legacy/ascend910/common
    ${HCOMM_DIR}/src/legacy/ascend910/common/debug/config
    ${HCOMM_DIR}/src/legacy/ascend910/common/debug/profiling/inc
    ${HCOMM_DIR}/src/legacy/ascend910/common/debug/profiling/inc/host
    ${HCOMM_DIR}/src/legacy/ascend910/common/error_manager
    ${HCOMM_DIR}/src/legacy/ascend910/common/launch_aicpu
    ${HCOMM_DIR}/src/legacy/ascend910/common/launch_device
    ${HCOMM_DIR}/src/legacy/ascend910/framework
    ${HCOMM_DIR}/src/legacy/ascend910/framework/cluster_maintenance/health/heartbeat
    ${HCOMM_DIR}/src/legacy/ascend910/framework/cluster_maintenance/recovery/operator_retry
    ${HCOMM_DIR}/src/legacy/ascend910/framework/common/src
    ${HCOMM_DIR}/src/legacy/ascend910/framework/common/src/config
    ${HCOMM_DIR}/src/legacy/ascend910/framework/common/src/exception
    ${HCOMM_DIR}/src/legacy/ascend910/framework/common/src/host
    ${HCOMM_DIR}/src/legacy/ascend910/framework/common/src/task
    ${HCOMM_DIR}/src/legacy/ascend910/framework/common/src/topo
    ${HCOMM_DIR}/src/legacy/ascend910/framework/communicator/impl
    ${HCOMM_DIR}/src/legacy/ascend910/framework/communicator/impl/one_sided_service
    ${HCOMM_DIR}/src/legacy/ascend910/framework/communicator/impl/resource_manager
    ${HCOMM_DIR}/src/legacy/ascend910/framework/communicator/impl/symmetric_memory
    ${HCOMM_DIR}/src/legacy/ascend910/framework/device
    ${HCOMM_DIR}/src/legacy/ascend910/framework/group
    ${HCOMM_DIR}/src/legacy/ascend910/framework/hcom
    ${HCOMM_DIR}/src/legacy/ascend910/framework/inc
    ${HCOMM_DIR}/src/legacy/ascend910/framework/op_base/src
    ${HCOMM_DIR}/src/legacy/ascend910/platform/common
    ${HCOMM_DIR}/src/legacy/ascend910/platform/inc
    ${HCOMM_DIR}/src/legacy/ascend910/platform/inc/adapter
    ${HCOMM_DIR}/src/legacy/ascend910/pub_inc
    ${HCOMM_DIR}/src/legacy/ascend910/pub_inc/new

    # legacy/ascend950 (needed transitively via legacy headers)
    ${HCOMM_DIR}/src/legacy/ascend950
    ${HCOMM_DIR}/src/legacy/ascend950/common
    ${HCOMM_DIR}/src/legacy/ascend950/common/exception
    ${HCOMM_DIR}/src/legacy/ascend950/common/types
    ${HCOMM_DIR}/src/legacy/ascend950/common/utils
    ${HCOMM_DIR}/src/legacy/ascend950/framework
    ${HCOMM_DIR}/src/legacy/ascend950/framework/communicator
    ${HCOMM_DIR}/src/legacy/ascend950/framework/communicator/aicpu
    ${HCOMM_DIR}/src/legacy/ascend950/framework/communicator/hostdpu
    ${HCOMM_DIR}/src/legacy/ascend950/framework/dfx/common
    ${HCOMM_DIR}/src/legacy/ascend950/framework/dfx/task_exception
    ${HCOMM_DIR}/src/legacy/ascend950/framework/entrance
    ${HCOMM_DIR}/src/legacy/ascend950/framework/entrance/hcom_comm
    ${HCOMM_DIR}/src/legacy/ascend950/framework/env_config
    ${HCOMM_DIR}/src/legacy/ascend950/framework/resource_manager/socket
    ${HCOMM_DIR}/src/legacy/ascend950/framework/resource_manager/stream
    ${HCOMM_DIR}/src/legacy/ascend950/framework/topo
    ${HCOMM_DIR}/src/legacy/ascend950/framework/topo/rank_info_detect
    ${HCOMM_DIR}/src/legacy/ascend950/framework/topo/topo_addr_info/include
    ${HCOMM_DIR}/src/legacy/ascend950/interface
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective/primitive
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/ccu/ccu_device
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/ccu/ccu_representation/reps/common
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/ccu/dfx
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/common
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/external_system
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/pub_inc
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/pub_inc/ccu
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/pub_inc/resource/transport
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/buffer
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/buffer/aicpu
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/connection/aicpu
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/notify/aicpu
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/socket
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/stream
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/stream/aicpu
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/transport

    # third-party headers
    ${URMA_INCLUDE_DIR}
)

if(NOT BUILD_OPEN_PROJECT)
    target_include_directories(hcomm_coll_comm_obj PRIVATE
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
endif()

# link CANN header libs for third-party header paths
target_link_libraries(hcomm_coll_comm_obj
    $<BUILD_INTERFACE:c_sec_headers>
    $<BUILD_INTERFACE:slog_headers>
    $<BUILD_INTERFACE:json>
    $<BUILD_INTERFACE:error_manager_headers>
    $<BUILD_INTERFACE:acl_rt_headers>
    $<BUILD_INTERFACE:ascend_hal_headers>
    $<BUILD_INTERFACE:runtime_headers>
    $<BUILD_INTERFACE:mmpa_headers>
    $<BUILD_INTERFACE:atrace_headers>
    $<BUILD_INTERFACE:rdma_core_headers>
    $<BUILD_INTERFACE:hccl_legacy_headers>
    $<BUILD_INTERFACE:asc_host_headers>
    $<BUILD_INTERFACE:kernel_tiling_headers>
)
