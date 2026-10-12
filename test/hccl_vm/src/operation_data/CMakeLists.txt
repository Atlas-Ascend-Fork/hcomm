# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_library(operation_data STATIC)

target_sources(operation_data PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/operation_data_ops.cc
)

target_include_directories(operation_data PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/profiling
    ${PROJECT_SOURCE_DIR}/include
    ${HCOMM_VM_ROOT_PATH}/include
    ${HCOMM_VM_ROOT_PATH}/src
    ${HCOMM_VM_ROOT_PATH}/src/runtime_state
    ${HCOMM_VM_ROOT_PATH}/src/operation_data
)

target_link_libraries(operation_data PRIVATE log storage simulator_storage)
# operation_data_ops 调用 sim::runtime::GetCommunicatorIdentity（opIter 分配按通信域
# 身份自增）；静态库符号在最终链接期解析，依赖必须传递给消费者
target_link_libraries(operation_data PUBLIC runtime_state)

hcclvm_apply_common_flags(operation_data)

# 日志染色: 模块 tag
target_compile_definitions(operation_data PRIVATE HCCL_VM_MODULE="OPERATION_DATA")
