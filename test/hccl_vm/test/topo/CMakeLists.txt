# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.14)

# 集群拓扑生成 golden 测试:
#   1. generate_cluster_topo.sh 脚本链路(测试内部在临时目录按安装态布局软链接运行脚本)
#   2. mock-comm 链路(InitClusterTopo + InitCommunicationDomain 生成 ranktable.json)
# 测试从 output/bin 运行, 拷贝配置镜像安装目录布局, 供 InstallPath 解析 config/topo_meta
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin/config")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin/data")
file(COPY "${CMAKE_SOURCE_DIR}/config/log_config.yaml"
    DESTINATION "${CMAKE_BINARY_DIR}/output/bin/config")
file(COPY "${CMAKE_SOURCE_DIR}/asset/cluster_model/config/"
    DESTINATION "${CMAKE_BINARY_DIR}/output/bin/config")
file(COPY "${CMAKE_SOURCE_DIR}/asset/cluster_model/topo_meta"
    DESTINATION "${CMAKE_BINARY_DIR}/output/bin/config")

set(TOPO_GOLDEN_INCLUDES
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/common
    ${CMAKE_SOURCE_DIR}/src/plugin
    ${CMAKE_SOURCE_DIR}/src/cmd
    ${CMAKE_SOURCE_DIR}/src/utils
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/store
    ${CMAKE_SOURCE_DIR}/include/topo
    ${CMAKE_SOURCE_DIR}/third_party/CLI11
    ${YAMLCPP_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/third_party/nlohmann_json
)

set(TOPO_GOLDEN_LIBS
    gtest_main
    gtest
    hccl_plugin_objs
    cmd
    common
    store
    runtime_state
    operation_data
    simulator_storage
    storage
    topo
    log
    yaml-cpp
    z
    dl
    rt
)

# 引擎专属测试子目录（test/topo/<模块名>/）：模块存在才接入
#（topo_golden 依赖真实 SQLite 会话，随 sqlite 模块一起保留或裁剪）。
include(${CMAKE_SOURCE_DIR}/src/storage_backends/CMakeLists.txt)
hcclvm_storage_backends_add_test_dir(${CMAKE_CURRENT_SOURCE_DIR})
