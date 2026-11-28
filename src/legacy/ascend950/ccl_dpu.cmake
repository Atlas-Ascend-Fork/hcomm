# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 定义 ccl_dpu 动态链接库，在 host 侧使用
add_library(ccl_dpu SHARED
    ${HCOMM_DIR}/src/base_comm/resources/comm_engine_res/threads/cpu_thread/legacy_dpu_thread/dpu_kernel_entrance.cc
    ${HCOMM_DIR}/src/base_comm/resources/comm_engine_res/threads/cpu_thread/legacy_dpu_thread/task_service.cc
    # SQ/CQ 队列：任务管道与 kernel 入口（CpuTaskPipe 与宿主侧共用同一份源码，编两遍）
    ${HCOMM_DIR}/src/base_comm/resources/comm_engine_res/threads/cpu_thread/cpu_task_pipe.cc
    ${HCOMM_DIR}/src/base_comm/resources/comm_engine_res/threads/cpu_thread/cpu_queue_kernel_entrance.cc
    # SQE 执行段直调的 cpu adpt（与 hcomm_base_obj 共用同一份源码，编两遍）
    ${HCOMM_DIR}/src/base_comm/primitives/api_c_adpt/cpu_primitives_c_adpt.cc
    ${HCOMM_DIR}/src/base_comm/primitives/api_c_adpt/plf_debug_config_aicpu.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/framework/dfx/profiling/dlprof_function.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common/log.cc
)

# 宏定义
target_compile_definitions(ccl_dpu PRIVATE
    HCCD
    OPEN_BUILD_PROJECT
)

# 编译选项
target_compile_options(ccl_dpu PRIVATE
    -Werror
    -fno-common
    -fno-strict-aliasing
    $<$<CONFIG:Debug>:-Og -g>
    $<$<CONFIG:Release>:-O3>
)

# 链接选项
target_link_options(ccl_dpu PRIVATE
    -Wl,-Bsymbolic
    # 本库编入了宿主侧 cpu adpt 与任务管道，它们引用的宿主符号由 AICPU 进程内的库族在加载期解析，
    # 链接期无法闭环，故不启用 --no-undefined
    -Wl,--exclude-libs,ALL
)

# 链接库
target_link_libraries(ccl_dpu
    $<BUILD_INTERFACE:intf_pub>
    $<BUILD_INTERFACE:atrace_headers>
    $<BUILD_INTERFACE:runtime_headers>
    $<BUILD_INTERFACE:slog_headers>
    $<BUILD_INTERFACE:acl_rt_headers>
    $<BUILD_INTERFACE:ascend_hal_headers>
    $<BUILD_INTERFACE:rdma_core_headers>
    $<BUILD_INTERFACE:json>
    -Wl,--no-as-needed
    c_sec
    unified_dlog
    mmpa
    runtime
    acl_rt
    -Wl,--as-needed
    -lrt
    -ldl
    -lpthread
)

# 头文件搜索路径
target_include_directories(ccl_dpu PRIVATE
    ${HCOMM_DIR}/include
    ${HCOMM_DIR}/include/hccl
    ${HCOMM_DIR}/pkg_inc
    ${HCOMM_DIR}/pkg_inc/legacy
    ${HCOMM_DIR}/pkg_inc/hcomm
    ${HCOMM_DIR}/src/common
    ${HCOMM_DIR}/src/base_comm/inc
    ${HCOMM_DIR}/src/base_comm/common
    # 收编后 kernel 侧复用宿主侧 legacy_dpu_thread 的共享类型定义（DpuKernelLaunchParam/TASK_* 状态机）
    ${HCOMM_DIR}/src/base_comm/resources/comm_engine_res/threads/cpu_thread/legacy_dpu_thread
    # SQ/CQ 队列的布局契约与任务管道（dpu_sync_layout.h / cpu_task_pipe.h）
    ${HCOMM_DIR}/src/base_comm/resources/comm_engine_res/threads/cpu_thread

    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/common/types
    ${CMAKE_CURRENT_SOURCE_DIR}/common/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/common/exception
    ${CMAKE_CURRENT_SOURCE_DIR}/common/error_manager
    ${CMAKE_CURRENT_SOURCE_DIR}/base/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/unified_platform/pub_inc
    ${CMAKE_CURRENT_SOURCE_DIR}/unified_platform/external_system
    ${CMAKE_CURRENT_SOURCE_DIR}/framework/dfx
    ${CMAKE_CURRENT_SOURCE_DIR}/framework/dfx/profiling
    ${HCOMM_DIR}/src/coll_communicator_mgr/dfx/profiling/host

    # cpu_primitives_c_adpt.cc 的头文件闭包（SQE 执行段直调的 cpu adpt）
    ${HCOMM_DIR}/pkg_inc/hccl
    ${HCOMM_DIR}/pkg_inc/legacy/hccl
    ${HCOMM_DIR}/src/base_comm/common/debug
    ${HCOMM_DIR}/src/base_comm/dfx
    ${HCOMM_DIR}/src/base_comm/primitives
    ${HCOMM_DIR}/src/base_comm/primitives/api_c_adpt
    ${HCOMM_DIR}/src/base_comm/primitives/api_c_adpt/nic_plugin
    ${HCOMM_DIR}/src/base_comm/resources/comm_engine_res/threads
    ${HCOMM_DIR}/src/base_comm/resources/endpoint_pairs/channels
    ${HCOMM_DIR}/src/base_comm/resources/endpoint_pairs/channels/host
    # nic_plugin_holder.h 的闭包：endpoint.h 与 server_socket_context/ 头
    ${HCOMM_DIR}/src/base_comm/resources/endpoints
    ${HCOMM_DIR}/src/base_comm/resources/reged_mems
    ${HCOMM_DIR}/src/legacy/ascend950/common
    ${HCOMM_DIR}/src/base_comm/resources/flush
    ${HCOMM_DIR}/src/base_comm/resources/hccp/inc
    ${HCOMM_DIR}/src/base_comm/resources/hccp/inc/network
    ${HCOMM_DIR}/src/coll_communicator_mgr/rank_graph/rank_graph
    ${HCOMM_DIR}/src/legacy/ascend950/framework/topo/new_topo_builder/common
    ${HCOMM_DIR}/src/legacy/ascend950/framework/topo/new_topo_builder/rank_table_info
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/buffer/aicpu

    ${HCOMM_DIR}/src/legacy/ascend910/pub_inc
    ${HCOMM_DIR}/src/legacy/ascend910/pub_inc/new
    ${HCOMM_DIR}/src/legacy/ascend910/common
    ${HCOMM_DIR}/src/legacy/ascend910/common/debug/profiling/inc
    ${HCOMM_DIR}/src/legacy/ascend910/common/error_manager
    ${HCOMM_DIR}/src/legacy/ascend910/framework/inc
    ${HCOMM_DIR}/src/legacy/ascend910/framework/common/src/exception
    ${HCOMM_DIR}/src/legacy/ascend910/platform/inc
    ${HCOMM_DIR}/src/legacy/ascend910/platform/inc/adapter
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/pub_inc
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/base/alg_template
    ${HCOMM_DIR}/src/legacy/ascend910/algorithm/base/communicator

    ${HCOMM_DIR}/src/legacy/ascend950/framework/dfx/common
    ${HCOMM_DIR}/src/legacy/ascend950/framework/resource_manager/socket
    ${HCOMM_DIR}/src/legacy/ascend950/framework/topo
    ${HCOMM_DIR}/src/legacy/ascend950/service/collective
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/common
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/buffer
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/connection
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/connection/aicpu
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/notify/aicpu
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/stream/aicpu
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/transport
    # channel.h 引入 A5 transport 头后，其闭包需要下列目录；编进 ccl_dpu 的 cpu adpt 前端会经
    # host_cpu_roce_channel.h 走到这条链
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/transport/aicpu
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/task
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/mem
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/notify
    ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/socket
)

# 符号隐藏: 仅导出白名单符号; 非 release 模式依赖动态符号, 不挂载
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_link_options(ccl_dpu PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_LIST_DIR}/ccl_dpu.map")
    set_property(TARGET ccl_dpu APPEND PROPERTY LINK_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/ccl_dpu.map")
endif()

# 指定 ccl_dpu 构建完成后安装到指定的目标位置
install(TARGETS ccl_dpu
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} ${INSTALL_OPTIONAL}
    COMPONENT hcomm
)
install(TARGETS ccl_dpu
    LIBRARY DESTINATION ${INSTALL_DPU_KERNEL_JSON_DIR} ${INSTALL_OPTIONAL}
    COMPONENT hcomm
)

# 将 ccl_dpu.ini 转换为 json 格式
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libccl_dpu.json
    COMMAND ${HI_PYTHON} ${HCOMM_DIR}/cmake/scripts/parser_ini.py
                         ${HCOMM_DIR}/src/base_comm/resources/comm_engine_res/threads/cpu_thread/legacy_dpu_thread/ccl_dpu.ini
                         ${CMAKE_CURRENT_BINARY_DIR}/libccl_dpu.json
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    COMMENT "Generating libccl_dpu.json"
    VERBATIM
)
add_custom_target(ccl_dpu_json ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libccl_dpu.json
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libccl_dpu.json
    DESTINATION ${INSTALL_DPU_KERNEL_JSON_DIR} ${INSTALL_OPTIONAL}
    COMPONENT hcomm
)
