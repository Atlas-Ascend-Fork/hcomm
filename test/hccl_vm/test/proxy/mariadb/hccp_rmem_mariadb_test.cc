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
 * You may not use this file except in compliance with the License.
 * A copy of the License is located at ...
 * See the License for the specific language governing permissions and limitations under the License.
 */

// RaRmem endpoint_id 在 MariaDB 后端的序列化回环（自 test/proxy/hccp_rmem_stub_test.cc
// 拆分迁入，随 mariadb 后端模块一起保留或裁剪）：
//   endpoint_id 新字段在 MariaDB 后端的序列化回环不遗漏（MariaDB 用例由
//   HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 提供 per-run 私有实例，未设置跳过）。
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <gtest/gtest.h>
#include <memory>
#include <string>

#include "hccp_ctx.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "storage/storage_runtime.h"
#include "storage/table_access.h"
#include "storage/internal/storage_runtime_impl.h"
#include "storage_backends/mariadb/composition/mariadb_backend_module.h"
#include "storage_backends/mariadb/engine/mariadb_backend_factory.h"

using sim::backends::kMariadbBackendName;

// 链接占位：hccp_stub.cc 引用 hccp_ccu_stub.cc 的 SimRaCustomChannel（本用例
// 不触达该路径；生产中由 hccl_proxy_level2 一并链接 ccu_stub）。签名与
// hccp_ccu_stub.cc:531 一致。
extern "C" {
int SimRaCustomChannel(void* tlvHandle, struct TlvMsg* sendMsg, struct TlvMsg* recvMsg);
}

// 占位实现：仅满足链接，不参与任何用例路径。
int SimRaCustomChannel(void* tlvHandle, struct TlvMsg* sendMsg, struct TlvMsg* recvMsg)
{
    (void)tlvHandle;
    (void)sendMsg;
    (void)recvMsg;
    return -1;
}

namespace {

/// endpoint_id 序列化回环业务流：清表 → 建上下文/Token → 注册（endpoint_id
/// 随行写入）→ 读回 → Import → 读回。由本用例在已安装 MariaDB 会话的前提下
/// 调用；任何一个字段未跨线路序列化都会失败。
void RunEndpointIdRoundTripFlow()
{
    constexpr uint64_t kEndpointId = 0x7U;
    // 共享服务器隔离：清空本流程涉及的四张表（endpoint 按合同为 per-run 私有实例）
    ASSERT_TRUE(HcclSim::Storage::Clear<sim::runtime::RaContext>(HcclSim::Storage::ClearMode::RESET_SEQUENCE).ok());
    ASSERT_TRUE(HcclSim::Storage::Clear<sim::runtime::RaTokenId>(HcclSim::Storage::ClearMode::RESET_SEQUENCE).ok());
    ASSERT_TRUE(HcclSim::Storage::Clear<sim::runtime::RaLmem>(HcclSim::Storage::ClearMode::RESET_SEQUENCE).ok());
    ASSERT_TRUE(HcclSim::Storage::Clear<sim::runtime::RaRmem>(HcclSim::Storage::ClearMode::RESET_SEQUENCE).ok());

    sim::runtime::RaContext ctx{};
    ctx.device_id = 1U;
    ctx.mode = 0U;
    ctx.endpoint_id = kEndpointId;
    ctx.eidIndex = 0U;
    ctx.max_jetty_num = 8U;
    ctx.max_jfc_num = 4U;
    auto ctxAdded = sim::runtime::Db::Add<sim::runtime::RaContext>(ctx);
    ASSERT_TRUE(ctxAdded.ok() && ctxAdded.value.has_value());
    const uint64_t ctxId = ctxAdded.value->value;

    sim::runtime::RaTokenId token{};
    token.ctx_handle = ctxId;
    token.token_id = 0x5A5AU;
    token.ref_count = 1U;
    auto tokenAdded = sim::runtime::Db::Add<sim::runtime::RaTokenId>(token);
    ASSERT_TRUE(tokenAdded.ok() && tokenAdded.value.has_value());

    struct MrRegInfoT reg = {};
    reg.in.mem.addr = 0x20000U;
    reg.in.mem.size = 0x2000U;
    reg.in.ub.tokenIdHandle = reinterpret_cast<void*>(static_cast<uintptr_t>(tokenAdded.value->value));
    void* lmemHandle = nullptr;
    ASSERT_EQ(0, RaCtxLmemRegister(reinterpret_cast<void*>(static_cast<uintptr_t>(ctxId)), &reg, &lmemHandle));
    const uint64_t lmemId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(lmemHandle));

    // 注册行回读：endpoint_id 新字段跨线路往返不丢失
    auto lmem = sim::runtime::Db::GetById<sim::runtime::RaLmem>(lmemId);
    ASSERT_TRUE(lmem.ok() && lmem.value.has_value());
    EXPECT_EQ(lmem.value->endpoint_id, kEndpointId);
    EXPECT_EQ(lmem.value->ctx_handle, ctxId);
    EXPECT_EQ(lmem.value->token_id, 0x5A5AU);

    // Import：remote_eid 取自快照并再次跨线路往返
    struct MrImportInfoT info = {};
    std::memcpy(info.in.key.value, &lmemId, sizeof(lmemId));
    info.in.key.size = sizeof(lmemId);
    void* rmemHandle = nullptr;
    ASSERT_EQ(0, RaCtxRmemImport(reinterpret_cast<void*>(0x5555U), &info, &rmemHandle));
    const uint64_t rmemId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(rmemHandle));
    auto rmem = sim::runtime::Db::GetById<sim::runtime::RaRmem>(rmemId);
    ASSERT_TRUE(rmem.ok() && rmem.value.has_value());
    EXPECT_EQ(rmem.value->remote_eid, kEndpointId);
    EXPECT_EQ(rmem.value->remote_key, lmem.value->mem_key);
    EXPECT_EQ(rmem.value->target_seg_handle, lmemId);
}

/// 会话收尾：关闭进程会话并清理装配登记（镜像 conformance fixture 协议）
void CloseBackendSession()
{
    (void)HcclSim::Storage::StorageRuntime::CloseProcessSession();
    HcclSim::Storage::detail::StorageRuntimeImpl::Instance().ResetForTest();
}
} // namespace

/// MariaDB 后端 endpoint_id 序列化回环（endpoint 未配置时跳过）。
TEST(HccpRmemEndpointIdRoundTrip, MariadbBackendRetainsEndpointId)
{
    const char* endpointEnv = std::getenv("HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT");
    if (endpointEnv == nullptr || endpointEnv[0] == '\0') {
        GTEST_SKIP() << "HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 未设置";
    }
    const std::string endpointUri{endpointEnv};

    HcclSim::Storage::MariadbConnectionProfile profile;
    profile.uri = endpointUri;
    profile.connectionEstablishmentTimeoutS = 3;
    profile.packetReadTimeoutS = 30;
    profile.packetWriteTimeoutS = 30;
    profile.wireProtocolCompression = false;

    auto& runtime = HcclSim::Storage::detail::StorageRuntimeImpl::Instance();
    runtime.ResetForTest();
    runtime.RegisterEngineFactory(
        kMariadbBackendName,
        [profile](const HcclSim::Storage::LogicalDatabaseRuntimeConfig& databaseConfig)
            -> HcclSim::Storage::DbResult<std::unique_ptr<HcclSim::Storage::IStorageEngine>> {
            return HcclSim::Storage::CreateMariadbEngine(
                HcclSim::Storage::Ddl::kRunnerDatabase, databaseConfig, profile);
        });
    runtime.UseDatabase(HcclSim::Storage::Ddl::kRunnerDatabase);
    const HcclSim::Storage::StorageRuntimeConfig sessionConfig{
        {{std::string{HcclSim::Storage::Ddl::kRunnerDatabase.databaseId}, kMariadbBackendName, endpointUri}}};
    ASSERT_TRUE(HcclSim::Storage::StorageRuntime::EnsureProcessSession(sessionConfig).ok())
        << "MariaDB 会话装配失败（endpoint 已配置但不可用必须失败）";

    RunEndpointIdRoundTripFlow();
    CloseBackendSession();
}
