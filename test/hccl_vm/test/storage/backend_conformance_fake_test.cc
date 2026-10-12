/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file backend_conformance_fake_test.cc
 * @brief 用内存语义引擎（Fake 后端）实例化全部公共 backend conformance case。
 *
 * fixture 协议：经 StorageRuntime 注册 fake 工厂与 conformance 目录，
 * 按后端无关配置安装进程会话；每个用例获得全新引擎实例（零数据耦合）。
 * 公共 case 的期望全部来自 backend_conformance.h 单一来源，不为 fake 后端修改。
 */
#include "backend_conformance.h"
#include "conformance_memory_engine.h"
#include "test_fake_backend.h"

namespace storage_conformance {

/// fixture 会话安装：注册 fake 工厂 + conformance 目录，安装进程唯一会话
bool OpenConformanceSession()
{
    auto& runtime = HcclSim::Storage::detail::StorageRuntimeImpl::Instance();
    runtime.ResetForTest();
    runtime.RegisterEngineFactory(
        HcclSim::Storage::kTestFakeBackendName, [](const HcclSim::Storage::LogicalDatabaseRuntimeConfig&) {
            return ConformanceMemoryEngine::Create(&kConformanceDatabase);
        });
    runtime.UseDatabase(kConformanceDatabase);
    const HcclSim::Storage::StorageRuntimeConfig config{
        {{"CONFORMANCE_DB", HcclSim::Storage::kTestFakeBackendName, "memory://conformance"}}};
    const auto ensured = HcclSim::Storage::StorageRuntime::EnsureProcessSession(config);
    return ensured.ok();
}

/// fixture 会话关闭：排空并关闭进程会话，清空装配登记
bool CloseConformanceSession()
{
    const auto closed = HcclSim::Storage::StorageRuntime::CloseProcessSession();
    HcclSim::Storage::detail::StorageRuntimeImpl::Instance().ResetForTest();
    return closed.ok();
}
} // namespace storage_conformance

// Fake 引擎实例化非事务 + 事务清单（真实后端合同）；投影清单不实例化——
// Fake 不实现字段投影，保持其既有能力合同（见 backend_conformance.h 注释）。
HCCL_VM_NON_TRANSACTIONAL_CONFORMANCE_CASES()
HCCL_VM_CONFORMANCE_CASE(Transaction)
