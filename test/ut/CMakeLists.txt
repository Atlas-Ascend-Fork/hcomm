# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(HCCL_CODE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
message(STATUS "${HCCL_CODE_ROOT}/src/legacy/ascend910/platform/common/ *******************************************")
set(HCCL_LLT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)

add_subdirectory(stub)
add_subdirectory(framework/next)
add_subdirectory(framework/next/coll_comms/dfx)
add_subdirectory(framework/next/comms/ccu)
add_subdirectory(common/prepare_ut_env)
add_subdirectory(common/binary_package)
add_subdirectory(common/debug/profiling)
add_subdirectory(experimental/base_comm/nic_plugin)
add_subdirectory(misc)
add_subdirectory(impl)
add_subdirectory(device)
add_subdirectory(platform/hcom)
add_subdirectory(platform/resource/dispatcher)
add_subdirectory(platform/resource/notify)
add_subdirectory(platform/resource/transport)
add_subdirectory(platform/ping_mesh)
add_subdirectory(platform/common/unfold_cache)
add_subdirectory(platform/common/device_capacity)
add_subdirectory(platform/common/mr_manager)
add_subdirectory(platform/task)
add_subdirectory(platform/task/task_logic)
add_subdirectory(framework/communicator)
add_subdirectory(framework/cluster_maintenance)
add_subdirectory(framework/op_base_api)
add_subdirectory(framework/hcom)
add_subdirectory(framework/group)
add_subdirectory(framework/communicator/impl)
add_subdirectory(framework/resource)
add_subdirectory(framework/communicator/impl/independent_op/data_api)
add_subdirectory(framework/communicator/impl/independent_op/data_api_cpu)
add_subdirectory(aicpu_kfc/aicpu)
add_subdirectory(aicpu_kfc/misc)
add_subdirectory(inter/comm)
add_subdirectory(inter/hcom)
add_subdirectory(platform/hccp)
add_subdirectory(dfx)
add_subdirectory(base_comm/primitives/aicpu/task_cache)
add_subdirectory(framework/nslbdp)
add_subdirectory(base_comm/primitives/api_c_adpt/nic_plugin)
add_subdirectory(base_comm/common)
add_subdirectory(framework/dpu_npu_sync)
add_subdirectory(base_comm/common/orion_adpt_utils)
