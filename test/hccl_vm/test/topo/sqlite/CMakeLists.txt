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

# 文件职责：topo 域的 SQLite 引擎专属测试（集群拓扑 golden 生成走真实 SQLite 会话；
# 随 sqlite 后端模块一起保留或裁剪）。目标名与断言保持迁移前不变；TOPO_GOLDEN_*
# 变量与配置镜像拷贝由 test/topo/CMakeLists.txt 提供（变量向子目录传播）。

# test_topo_golden
add_executable(test_topo_golden topo_golden_test.cc)
add_dependencies(test_topo_golden hccl-vm)
target_include_directories(test_topo_golden PRIVATE ${TOPO_GOLDEN_INCLUDES})
target_link_libraries(test_topo_golden PRIVATE ${TOPO_GOLDEN_LIBS})
set_target_properties(test_topo_golden PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin"
)
