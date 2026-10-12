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

# 文件职责：cmd subcmd 域的 SQLite 引擎专属测试（start/run 子命令经惰性默认会话
# 依赖真实 SQLite；随 sqlite 后端模块一起保留或裁剪）。SUBCMD_* 变量继承自
# test/cmd/subcmd/CMakeLists.txt；目标名与断言保持迁移前不变。

# test_subcmd_start
add_executable(test_subcmd_start subcmd_start_test.cc)
if(TARGET hccl-vm)
    add_dependencies(test_subcmd_start hccl-vm)
endif()
target_include_directories(test_subcmd_start PRIVATE ${SUBCMD_INCLUDE_DIRS})
target_link_libraries(test_subcmd_start PRIVATE ${SUBCMD_LIBS})
set_target_properties(test_subcmd_start PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin")

# test_subcmd_run
add_executable(test_subcmd_run subcmd_run_test.cc)
if(TARGET hccl-vm)
    add_dependencies(test_subcmd_run hccl-vm)
endif()
target_include_directories(test_subcmd_run PRIVATE ${SUBCMD_INCLUDE_DIRS})
target_link_libraries(test_subcmd_run PRIVATE ${SUBCMD_LIBS})
set_target_properties(test_subcmd_run PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/output/bin")
