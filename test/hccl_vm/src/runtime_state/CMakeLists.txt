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

add_library(runtime_state STATIC)

target_sources(runtime_state PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/db_sim_runner_common.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/db_sim_runner_ops.cc
)

target_include_directories(runtime_state PRIVATE
    ${THRID_PARTY_DIR}/nlohmann_json
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/profiling
    ${PROJECT_SOURCE_DIR}/include
    ${HCOMM_VM_ROOT_PATH}/include
    ${HCOMM_VM_ROOT_PATH}/src
    ${HCOMM_VM_ROOT_PATH}/src/store
    ${HCOMM_VM_ROOT_PATH}/src/runtime_state
)

# db_sim_runner_common.cc 调 sim::IsEstimateMode()(在 store),store 又反向依赖本库 Db 接口,静态库互链
target_link_libraries(runtime_state PRIVATE log storage store)

hcclvm_apply_common_flags(runtime_state)

# 日志染色: 模块 tag
target_compile_definitions(runtime_state PRIVATE HCCL_VM_MODULE="RUNTIME_STATE")
