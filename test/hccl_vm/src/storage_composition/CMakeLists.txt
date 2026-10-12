# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# See the License for the specific file packaged with this Software for details.

# simulator_storage 是后端中立的存储装配（bootstrap/composition）：登记两个业务
# 目录、经生成的模块注册清单发现已编入后端并按当前路由分派维护操作。具体后端
# 的引擎/工厂/profile/配置段规则在 src/storage_backends/<模块名>/，经收集器接入。

add_library(simulator_storage STATIC)

target_sources(simulator_storage PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/storage_composition.cc
    # 存储部署配置解析（HCCL_VM_STORAGE_CONFIG，中立外层 + 模块段分发）：
    # 配置只由本 composition bootstrap 读取，storage 公共库不依赖 yaml。
    ${CMAKE_CURRENT_SOURCE_DIR}/storage_deploy_config.cc
)

target_include_directories(simulator_storage PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/profiling
    ${PROJECT_SOURCE_DIR}/include
    ${HCOMM_VM_ROOT_PATH}/include
    ${HCOMM_VM_ROOT_PATH}/src
)

target_link_libraries(simulator_storage PRIVATE log storage yaml-cpp::yaml-cpp)
if(TARGET third_party_yaml_cpp)
    add_dependencies(simulator_storage third_party_yaml_cpp)
endif()

hcclvm_apply_common_flags(simulator_storage)

# 日志染色: 模块 tag
target_compile_definitions(simulator_storage PRIVATE HCCL_VM_MODULE="SIMULATOR_STORAGE")

# ---- 后端模块接入（先发现和准备依赖[顶层 prepare]，再向已存在目标接入）----
# engine 源码与客户库依赖接入 storage（PUBLIC 传递给全部链接方）；模块
# composition 源码与生成的注册清单接入 simulator_storage。
include(${CMAKE_SOURCE_DIR}/src/storage_backends/CMakeLists.txt)
hcclvm_storage_backends_attach(TARGET storage COMPOSITION_TARGET simulator_storage)
