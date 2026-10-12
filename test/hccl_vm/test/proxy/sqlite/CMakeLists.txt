# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# See the License in the specific file packaged with this Software for details.

# 文件职责：proxy 域的 SQLite 引擎专属测试（真实 SQLite 文件 fixture；随 sqlite
# 后端模块一起保留或裁剪，由收集器按模块存在性接入）。目录级 include/link 设置
# 继承自 test/proxy/CMakeLists.txt；目标名与断言保持迁移前不变。

# test_aclrt_memory_stub —— 主机侧大块引流/上界/FreeHost 行为单测
# InsertTaskToCollection 由 link_libraries 的 store 库提供，无需额外编 device_arm 源。
add_executable(test_aclrt_memory_stub aclrt_memory_stub_test.cc)
target_sources(test_aclrt_memory_stub PRIVATE
    ${CMAKE_SOURCE_DIR}/src/proxy/level2/aclrt_memory_stub.cc
    ${CMAKE_SOURCE_DIR}/src/proxy/common/hccl_proxy_common.cc
    ${CMAKE_SOURCE_DIR}/src/common/sim_sub_process_manager.cc
    ${CMAKE_SOURCE_DIR}/src/common/sim_pipe_io.cc
)
set_target_properties(test_aclrt_memory_stub PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
target_link_libraries(test_aclrt_memory_stub gtest_main)

# test_ascend_hal_proxy_stub —— 主机侧 drvMemcpy 地址分类与错误传播单测
# （IsDeviceAddress 三态合同：命中/确定未命中/查询失败传播；拷贝失败不返回成功）
add_executable(test_ascend_hal_proxy_stub ascend_hal_proxy_stub_test.cc)
target_sources(test_ascend_hal_proxy_stub PRIVATE
    ${CMAKE_SOURCE_DIR}/src/proxy/level2/ascend_hal_stub.cc
    ${CMAKE_SOURCE_DIR}/src/proxy/common/hccl_proxy_common.cc
    ${CMAKE_SOURCE_DIR}/src/proxy/level2/aclrt_memory_stub.cc
    ${CMAKE_SOURCE_DIR}/src/runtime_state/db_sim_runner_common.cc
    ${CMAKE_SOURCE_DIR}/src/common/sim_sub_process_manager.cc
    ${CMAKE_SOURCE_DIR}/src/common/sim_pipe_io.cc
)
set_target_properties(test_ascend_hal_proxy_stub PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
target_link_libraries(test_ascend_hal_proxy_stub gtest_main)

# test_ctrl_base_resource_stub —— HComm基础Endpoint创建和南向绑定行为单测
add_executable(test_ctrl_base_resource_stub ctrl_base_resource_stub_test.cc)
target_sources(test_ctrl_base_resource_stub PRIVATE
    ${CMAKE_SOURCE_DIR}/src/proxy/level1/ctrl_base_resource_stub.cc
    ${CMAKE_SOURCE_DIR}/src/proxy/level2/aclrt_stream_stub.cc
)
set_target_properties(test_ctrl_base_resource_stub PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
target_link_libraries(test_ctrl_base_resource_stub PRIVATE
    gtest_main
    runtime_state
    operation_data
    simulator_storage
    storage
)

# test_aclrt_device_stub —— aclInit 冷启动进程 Session bootstrap 回归（链接真实 production 文件）
add_executable(test_aclrt_device_stub aclrt_device_stub_test.cc)
target_sources(test_aclrt_device_stub PRIVATE
    ${CMAKE_SOURCE_DIR}/src/proxy/level2/aclrt_device_stub.cc
    ${CMAKE_SOURCE_DIR}/src/proxy/common/hccl_proxy_common.cc
    ${CMAKE_SOURCE_DIR}/src/common/sim_sub_process_manager.cc
    ${CMAKE_SOURCE_DIR}/src/common/sim_dpu_kernel_lib_mgr.cc
    ${CMAKE_SOURCE_DIR}/src/common/sim_pipe_io.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/platform_info_stub.cc
)
set_target_properties(test_aclrt_device_stub PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
target_link_libraries(test_aclrt_device_stub PRIVATE
    gtest_main
    common
    runtime_state
    operation_data
    simulator_storage
    storage
    log
)

# test_hccp_rmem_stub —— RaRmem/RaLmem 注册注销句柄回归（链接真实 hccp_stub.cc）
add_executable(test_hccp_rmem_stub hccp_rmem_stub_test.cc)
target_sources(test_hccp_rmem_stub PRIVATE
    ${CMAKE_SOURCE_DIR}/src/proxy/level2/hccp_stub.cc
    ${CMAKE_SOURCE_DIR}/src/proxy/level2/hccp_async_request.cc
    ${CMAKE_SOURCE_DIR}/src/proxy/level2/ibv_dpu_stub.cc
    ${CMAKE_SOURCE_DIR}/src/proxy/common/hccl_proxy_common.cc
)
target_include_directories(test_hccp_rmem_stub PRIVATE
    ${CMAKE_SOURCE_DIR}/src/proxy/level2
    ${CMAKE_SOURCE_DIR}/src/proxy/common
    ${CMAKE_SOURCE_DIR}/src/store
    ${HCOMM_CODE_ROOT}/src/base_comm/resources/hccp
    ${HCOMM_CODE_ROOT}/src/base_comm/resources/hccp/inc
    ${HCOMM_CODE_ROOT}/src/base_comm/resources/hccp/inc/network
    ${HCOMM_CODE_ROOT}/src/base_comm/resources/hccp/external_depends/ccu
    ${HCOMM_CODE_ROOT}/src/platform/hccp
    ${HCOMM_CODE_ROOT}/src/platform/hccp/inc
    ${HCOMM_CODE_ROOT}/src/platform/hccp/inc/network
)
set_target_properties(test_hccp_rmem_stub PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
target_link_libraries(test_hccp_rmem_stub PRIVATE
    gtest_main
    common
    runtime_state
    operation_data
    simulator_storage
    storage
    log
)

# test_hccp_ra_socket_async —— RaSocket 异步请求合同回归（512p 忙等风暴修复：
# 空队列 EAGAIN、真实唯一句柄、查询即释放、满槽 EAGAIN、并发独立性；
# 含 QP/TP 族跨存根合同用例，故同时链接 hccp_stub.cc）
add_executable(test_hccp_ra_socket_async hccp_ra_socket_async_test.cc)
target_sources(test_hccp_ra_socket_async PRIVATE
    ${CMAKE_SOURCE_DIR}/src/proxy/level2/hccp_ra_socket_stub.cc
    ${CMAKE_SOURCE_DIR}/src/proxy/level2/hccp_stub.cc
    ${CMAKE_SOURCE_DIR}/src/proxy/level2/hccp_async_request.cc
    ${CMAKE_SOURCE_DIR}/src/proxy/level2/ibv_dpu_stub.cc
    ${CMAKE_SOURCE_DIR}/src/proxy/common/hccl_proxy_common.cc
)
target_include_directories(test_hccp_ra_socket_async PRIVATE
    ${CMAKE_SOURCE_DIR}/src/proxy/level2
    ${CMAKE_SOURCE_DIR}/src/proxy/common
    ${CMAKE_SOURCE_DIR}/src/store
    ${HCOMM_CODE_ROOT}/src/base_comm/resources/hccp
    ${HCOMM_CODE_ROOT}/src/base_comm/resources/hccp/inc
    ${HCOMM_CODE_ROOT}/src/base_comm/resources/hccp/inc/network
    ${HCOMM_CODE_ROOT}/src/base_comm/resources/hccp/external_depends/ccu
    ${HCOMM_CODE_ROOT}/src/platform/hccp
    ${HCOMM_CODE_ROOT}/src/platform/hccp/inc
    ${HCOMM_CODE_ROOT}/src/platform/hccp/inc/network
)
set_target_properties(test_hccp_ra_socket_async PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
target_link_libraries(test_hccp_ra_socket_async PRIVATE
    gtest_main
    common
    runtime_state
    operation_data
    simulator_storage
    storage
    log
)

# test_hccp_ra_socket_diag —— RaSocketPair 生产链首错诊断回归（Add 失败字段、
# 删除 1 行/0 行区分、轮询 NOT_FOUND→成功、后端失败→NOT_FOUND 超时首错保留；
# 含一个约 93 秒的有界慢用例）
add_executable(test_hccp_ra_socket_diag hccp_ra_socket_diag_test.cc)
target_sources(test_hccp_ra_socket_diag PRIVATE
    ${CMAKE_SOURCE_DIR}/src/proxy/level2/hccp_ra_socket_stub.cc
    ${CMAKE_SOURCE_DIR}/src/proxy/level2/hccp_stub.cc
    ${CMAKE_SOURCE_DIR}/src/proxy/level2/hccp_async_request.cc
    ${CMAKE_SOURCE_DIR}/src/proxy/level2/ibv_dpu_stub.cc
    ${CMAKE_SOURCE_DIR}/src/proxy/common/hccl_proxy_common.cc
)
target_include_directories(test_hccp_ra_socket_diag PRIVATE
    ${CMAKE_SOURCE_DIR}/src/proxy/level2
    ${CMAKE_SOURCE_DIR}/src/proxy/common
    ${CMAKE_SOURCE_DIR}/src/store
    ${HCOMM_CODE_ROOT}/src/base_comm/resources/hccp
    ${HCOMM_CODE_ROOT}/src/base_comm/resources/hccp/inc
    ${HCOMM_CODE_ROOT}/src/base_comm/resources/hccp/inc/network
    ${HCOMM_CODE_ROOT}/src/base_comm/resources/hccp/external_depends/ccu
    ${HCOMM_CODE_ROOT}/src/platform/hccp
    ${HCOMM_CODE_ROOT}/src/platform/hccp/inc
    ${HCOMM_CODE_ROOT}/src/platform/hccp/inc/network
)
set_target_properties(test_hccp_ra_socket_diag PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
target_link_libraries(test_hccp_ra_socket_diag PRIVATE
    gtest_main
    common
    runtime_state
    operation_data
    simulator_storage
    storage
    log
)
