# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# Level2: Runtime/CANN底层接口劫持
add_library(hccl_proxy_level2 SHARED "")
target_include_directories(hccl_proxy_level2 PRIVATE ${PROXY_INC_DIRS})
target_sources(hccl_proxy_level2 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/aclrt_device_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/aclrt_model_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/aclrt_graph_capture_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/graph_capture.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/aclrt_context_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/aclrt_memory_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/aclrt_notify_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/aclrt_event_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/aclrt_stream_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/scatter_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/log_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/hccl_inner_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/hccp_ccu_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/hccp_ra_socket_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/hccp_async_request.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/aclrt_new_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/aclrt_exception_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/aclrt_exec_control.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/aclrt_kernel_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/hccp_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ascend_hal_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/adapter_rts_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/tsd_client_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/aprofiling_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/aclrt_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/aclrt_runtime_config.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/hcomm_api_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ibv_dpu_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/topo_addr_info_stub.cc
        # 各层级共用的公共源码
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/hccl_proxy_common.cc
)
target_compile_options(hccl_proxy_level2 PRIVATE ${PROXY_COMPILE_OPTS})
target_link_libraries(hccl_proxy_level2 PRIVATE ${PROXY_LINK_LIBS})
install(TARGETS hccl_proxy_level2
    DESTINATION "lib/${CMAKE_SYSTEM_PROCESSOR}"
    # 文件的权限 (755)
    PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE
)

# AIV构建
option(BUILD_AIV "Enable aiv kernel build" OFF)
MESSAGE(STATUS "BUILD_AIV: " ${BUILD_AIV})
if(BUILD_AIV)
    add_subdirectory(aiv_kernel)
endif()
