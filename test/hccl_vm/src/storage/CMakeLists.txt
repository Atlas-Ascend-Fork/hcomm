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

# 文件职责：后端无关 storage 公共实现（会话、路由、协调器、事务、类型化表达式
# 入口与两个 SQL 后端共用的关系处理辅助）。具体引擎与客户库依赖由
# src/storage_backends/ 的后端模块经收集器接入（engine 源码 + PUBLIC 链接），
# 本文件不出现任何具体后端名。业务库目录登记与默认路由装配在
# storage_composition，本库不包含业务表目录或具体数据库管理器。
add_library(storage STATIC
    engine/relational/relational_sql_helpers.cc
    internal/session_control.cc
    internal/process_storage_context.cc
    internal/storage_runtime_impl.cc
    session_coordinator.cc
    storage_session.cc
    transaction.cc
)

target_include_directories(storage PUBLIC
    ${HCOMM_VM_ROOT_PATH}/include
    ${HCOMM_VM_ROOT_PATH}/src
)

hcclvm_apply_common_flags(storage)
