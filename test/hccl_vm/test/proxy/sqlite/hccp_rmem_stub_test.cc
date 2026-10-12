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

// RaRmem/RaLmem 注册注销回归（句柄错位修复）：
//   1) Import 返回【本 RaRmem 行自己的 id】且 Unimport 删除该行（旧实现返回远端
//      RaLmem 的 id → Unimport 永远删不中 → 注册行泄漏）；
//   2) RaLmem/RaRmem 双表自增 id 撞号时 Unimport 不得误删无关行（旧实现按错表
//      主键删除，撞号即跨行误删，checker 判据数据被破坏）；
//   3) 异步注销补删除（旧实现只回请求句柄不删 → 拆卸路径注册行无限累积）；
//   4) 注销不存在的 id 保持返回 0（拆卸健壮性，不因簿记失败中断）。
// 负对照：用例 1/2/3 在修复前的实现上必然失败。
//
// endpoint_id 快照回归（RaLmem 表 Version(3)）：
//   5) RaCtxLmemRegister 校验所属 RaContext 存在（缺失明确失败不写行）并把
//      endpoint_id 快照保存在 RaLmem 行；
//   6) RaCtxRmemImport 的 remote_eid 直接取该快照，不再按远端 ctx_handle 二次
//      查询 RaContext（1024p RaRmem 30 万+行的注册热点减量）；
//   7) endpoint_id 新字段的 MariaDB 序列化回环用例随 mariadb 引擎测试子目录
//      （test/proxy/mariadb/hccp_rmem_mariadb_test.cc）一起保留或裁剪。
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <utility>

#include "hccp_ctx.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "storage/storage_runtime.h"
#include "storage/table_access.h"
#include "storage_composition/storage_composition.h"

// hccp_ctx.h 未导出异步注销原型（hcomm 侧声明在另一头文件）；被测实现编译在本
// 目标内，此处按实现签名直接声明（extern "C" 与 hccp_stub.cc 定义一致）。
extern "C" {
int RaCtxLmemUnregisterAsync(void* ctxHandle, void* lmemHandle, void** reqHandle);
// 链接占位：hccp_stub.cc 引用 hccp_ccu_stub.cc 的 SimRaCustomChannel（本用例
// 不触达该路径；生产中由 hccl_proxy_level2 一并链接 ccu_stub）。签名与
// hccp_ccu_stub.cc:531 一致。
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
constexpr char kRunnerDb[] = "/tmp/hccp_rmem_stub_runner.db";
constexpr char kOpDataDb[] = "/tmp/hccp_rmem_stub_opdata.db";

void RemoveDatabaseFiles(const char* path)
{
    std::remove(path);
    std::remove((std::string(path) + "-wal").c_str());
    std::remove((std::string(path) + "-shm").c_str());
}
} // namespace

class HccpRmemStubTest : public testing::Test {
protected:
    void SetUp() override
    {
        (void)HcclSim::Storage::StorageRuntime::CloseProcessSession();
        RemoveDatabaseFiles(kRunnerDb);
        RemoveDatabaseFiles(kOpDataDb);
        HcclSim::Storage::StorageRuntimeConfig config;
        config.databases.push_back({std::string{"RUNNER_DB"}, "sqlite", kRunnerDb});
        config.databases.push_back({std::string{"OP_DATA_DB"}, "sqlite", kOpDataDb});
        ASSERT_TRUE(sim::EnsureSimulatorStorageSession(config).ok());
    }

    void TearDown() override
    {
        (void)HcclSim::Storage::StorageRuntime::CloseProcessSession();
        RemoveDatabaseFiles(kRunnerDb);
        RemoveDatabaseFiles(kOpDataDb);
    }

    /// 建远端 RaContext + RaLmem（lmemId=0 自增；非 0 显式主键用于撞号场景）。
    /// 返回 <remoteLmemId>（远端内存 id，即 Import 输入 key）。
    uint64_t SeedRemoteMemory(uint64_t lmemId = 0U)
    {
        sim::runtime::RaContext ctx{};
        ctx.device_id = 1U;
        ctx.mode = 0U;
        ctx.endpoint_id = 7U;
        ctx.eidIndex = 0U;
        ctx.max_jetty_num = 8U;
        ctx.max_jfc_num = 4U;
        auto ctxAdded = sim::runtime::Db::Add<sim::runtime::RaContext>(ctx);
        EXPECT_TRUE(ctxAdded.ok() && ctxAdded.value.has_value());

        sim::runtime::RaLmem lmem{};
        lmem.id = lmemId;
        lmem.ctx_handle = ctxAdded.value->value;
        lmem.addr = 0x10000U;
        lmem.size = 0x1000U;
        lmem.mem_key = 0xA5A5U;
        lmem.token_id = 1U;
        // 与 RaCtxLmemRegister 的快照语义一致：endpoint_id 随行保存
        lmem.endpoint_id = ctx.endpoint_id;
        auto lmemAdded = sim::runtime::Db::Add<sim::runtime::RaLmem>(lmem);
        EXPECT_TRUE(lmemAdded.ok() && lmemAdded.value.has_value());
        return lmemAdded.value->value;
    }

    /// 以远端内存 id 执行一次 Import，返回句柄。
    void* ImportRemote(uint64_t remoteLmemId, uint64_t localCtx)
    {
        struct MrImportInfoT info = {};
        std::memcpy(info.in.key.value, &remoteLmemId, sizeof(remoteLmemId));
        info.in.key.size = sizeof(remoteLmemId);
        void* handle = nullptr;
        EXPECT_EQ(0, RaCtxRmemImport(reinterpret_cast<void*>(localCtx), &info, &handle));
        EXPECT_NE(nullptr, handle);
        return handle;
    }

    /// 建本地 RaContext(endpoint_id) + RaTokenId，返回 <ctxId, tokenRowId>。
    std::pair<uint64_t, uint64_t> SeedLocalContextAndToken(uint64_t endpointId = 7U)
    {
        sim::runtime::RaContext ctx{};
        ctx.device_id = 1U;
        ctx.mode = 0U;
        ctx.endpoint_id = endpointId;
        ctx.eidIndex = 0U;
        ctx.max_jetty_num = 8U;
        ctx.max_jfc_num = 4U;
        auto ctxAdded = sim::runtime::Db::Add<sim::runtime::RaContext>(ctx);
        EXPECT_TRUE(ctxAdded.ok() && ctxAdded.value.has_value());

        sim::runtime::RaTokenId token{};
        token.ctx_handle = ctxAdded.value->value;
        token.token_id = 0x5A5AU;
        token.ref_count = 1U;
        auto tokenAdded = sim::runtime::Db::Add<sim::runtime::RaTokenId>(token);
        EXPECT_TRUE(tokenAdded.ok() && tokenAdded.value.has_value());
        return {ctxAdded.value->value, tokenAdded.value->value};
    }

    /// 走真实注册路径，返回注册行 id（lmemHandle 数值）。
    uint64_t RegisterLocalMemory(uint64_t ctxId, uint64_t tokenRowId)
    {
        struct MrRegInfoT reg = {};
        reg.in.mem.addr = 0x20000U;
        reg.in.mem.size = 0x2000U;
        reg.in.ub.tokenIdHandle = reinterpret_cast<void*>(static_cast<uintptr_t>(tokenRowId));
        void* lmemHandle = nullptr;
        EXPECT_EQ(0, RaCtxLmemRegister(reinterpret_cast<void*>(static_cast<uintptr_t>(ctxId)), &reg, &lmemHandle));
        EXPECT_NE(nullptr, lmemHandle);
        return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(lmemHandle));
    }

    /// 表行数（失败路径"不写不完整行"的计数断言用）。
    template <typename T>
    uint64_t TableCount()
    {
        const auto counted = sim::runtime::Db::Count<T>(HcclSim::Storage::All<T>());
        EXPECT_TRUE(counted.ok());
        return counted.ok() && counted.value.has_value() ? *counted.value : 0U;
    }
};

/// Import 返回本 RaRmem 行 id（非远端 RaLmem id），Unimport 恰好删除本行。
TEST_F(HccpRmemStubTest, ImportReturnsOwnRowIdAndUnimportDeletesIt)
{
    const uint64_t remoteLmemId = SeedRemoteMemory();
    const uint64_t localCtx = 0x9999U;
    void* handle = ImportRemote(remoteLmemId, localCtx);
    const uint64_t handleId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(handle));

    // 修复点 1：句柄是 RaRmem 自己的行 id——用“按句柄可取回该行且字段正确”验证
    // （不与 remoteLmemId 比大小：全新库两表自增序列同从 1 开始，数值可能相等）
    auto row = sim::runtime::Db::GetById<sim::runtime::RaRmem>(handleId);
    ASSERT_TRUE(row.ok() && row.value.has_value());
    EXPECT_EQ(row.value->target_seg_handle, remoteLmemId); // 远端 id 在字段里保留
    EXPECT_EQ(row.value->ctx_handle, localCtx);

    // 修复点 2：Unimport 删除本行
    EXPECT_EQ(0, RaCtxRmemUnimport(reinterpret_cast<void*>(localCtx), handle));
    auto gone = sim::runtime::Db::GetById<sim::runtime::RaRmem>(handleId);
    EXPECT_FALSE(gone.ok());
}

/// 双表自增 id 撞号：远端 RaLmem id=1000 且已存在无关 RaRmem 行 id=1000 时，
/// Import/Unimport 不得误删该无关行（旧实现按远端 id 删 RaRmem → 误删）。
TEST_F(HccpRmemStubTest, UnimportDoesNotDeleteForeignRowWithCollidingId)
{
    constexpr uint64_t kCollidingId = 1000U;
    const uint64_t remoteLmemId = SeedRemoteMemory(kCollidingId);
    ASSERT_EQ(remoteLmemId, kCollidingId);

    // 无关第三方注册行：显式主键 1000，与远端 RaLmem id 撞号
    sim::runtime::RaRmem victim{};
    victim.id = kCollidingId;
    victim.ctx_handle = 0x7777U;
    victim.remote_key = 0x1234U;
    victim.target_seg_handle = 0x5555U;
    victim.remote_eid = 9U;
    auto victimAdded = sim::runtime::Db::Add<sim::runtime::RaRmem>(victim);
    ASSERT_TRUE(victimAdded.ok() && victimAdded.value.has_value());

    const uint64_t localCtx = 0x8888U;
    void* handle = ImportRemote(remoteLmemId, localCtx);
    const uint64_t handleId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(handle));
    ASSERT_NE(handleId, kCollidingId); // 新行自增 id 不会取已被占用的 1000

    EXPECT_EQ(0, RaCtxRmemUnimport(reinterpret_cast<void*>(localCtx), handle));

    // 修复点：无关行（id=1000）必须还在；旧实现会把 handle 当 1000 删掉它
    auto survivor = sim::runtime::Db::GetById<sim::runtime::RaRmem>(kCollidingId);
    EXPECT_TRUE(survivor.ok() && survivor.value.has_value());
    EXPECT_EQ(survivor.value->ctx_handle, 0x7777U);
    // 本行已删
    auto own = sim::runtime::Db::GetById<sim::runtime::RaRmem>(handleId);
    EXPECT_FALSE(own.ok());
}

/// 异步注销补删除：旧实现只回请求句柄、不删行（拆卸路径泄漏）。
TEST_F(HccpRmemStubTest, LmemUnregisterAsyncDeletesRow)
{
    const uint64_t lmemId = SeedRemoteMemory();
    void* reqHandle = nullptr;
    EXPECT_EQ(
        0, RaCtxLmemUnregisterAsync(
               reinterpret_cast<void*>(0x1111U), reinterpret_cast<void*>(static_cast<uintptr_t>(lmemId)), &reqHandle));
    EXPECT_NE(nullptr, reqHandle);
    auto gone = sim::runtime::Db::GetById<sim::runtime::RaLmem>(lmemId);
    EXPECT_FALSE(gone.ok());
}

/// 注销不存在的 id：保持返回 0（拆卸健壮性），不中断调用方。
TEST_F(HccpRmemStubTest, UnregisterMissingRowStaysZero)
{
    void* reqHandle = nullptr;
    EXPECT_EQ(0, RaCtxLmemUnregister(reinterpret_cast<void*>(0x2222U), reinterpret_cast<void*>(0xDEADU)));
    EXPECT_EQ(
        0, RaCtxLmemUnregisterAsync(reinterpret_cast<void*>(0x2222U), reinterpret_cast<void*>(0xDEADU), &reqHandle));
    EXPECT_EQ(0, RaCtxRmemUnimport(reinterpret_cast<void*>(0x2222U), reinterpret_cast<void*>(0xDEADU)));
}

/// 注册保存 endpoint_id 快照；Import 的 remote_eid 直接取该快照（不再按远端
/// ctx_handle 二次查询 RaContext）。
TEST_F(HccpRmemStubTest, RegisterSavesEndpointIdAndImportUsesIt)
{
    constexpr uint64_t kEndpointId = 0x7U;
    const auto [ctxId, tokenRowId] = SeedLocalContextAndToken(kEndpointId);
    const uint64_t lmemId = RegisterLocalMemory(ctxId, tokenRowId);

    // 注册行：endpoint_id 快照正确保存，其余字段语义不变
    auto lmem = sim::runtime::Db::GetById<sim::runtime::RaLmem>(lmemId);
    ASSERT_TRUE(lmem.ok() && lmem.value.has_value());
    EXPECT_EQ(lmem.value->endpoint_id, kEndpointId);
    EXPECT_EQ(lmem.value->ctx_handle, ctxId);
    EXPECT_EQ(lmem.value->token_id, 0x5A5AU);
    EXPECT_EQ(lmem.value->addr, 0x20000U);
    EXPECT_EQ(lmem.value->size, 0x2000U);

    // Import：remote_eid 取自 RaLmem 快照；句柄/remote_key/target_seg_handle 不变
    const uint64_t localCtx = 0x3333U;
    void* handle = ImportRemote(lmemId, localCtx);
    const uint64_t rmemId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(handle));
    auto rmem = sim::runtime::Db::GetById<sim::runtime::RaRmem>(rmemId);
    ASSERT_TRUE(rmem.ok() && rmem.value.has_value());
    EXPECT_EQ(rmem.value->remote_eid, kEndpointId);
    EXPECT_EQ(rmem.value->remote_key, lmem.value->mem_key);
    EXPECT_EQ(rmem.value->target_seg_handle, lmemId);
    EXPECT_EQ(rmem.value->ctx_handle, localCtx);
}

/// 不存在的 RaContext：注册明确失败且不产生 RaLmem 行（不写不完整行）。
TEST_F(HccpRmemStubTest, RegisterWithMissingRaContextFailsAndWritesNothing)
{
    const auto [ctxId, tokenRowId] = SeedLocalContextAndToken();
    (void)ctxId; // 合法 ctx 留作对照：失败用例使用不存在的 ctx 句柄
    const uint64_t before = TableCount<sim::runtime::RaLmem>();

    struct MrRegInfoT reg = {};
    reg.in.mem.addr = 0x20000U;
    reg.in.mem.size = 0x2000U;
    reg.in.ub.tokenIdHandle = reinterpret_cast<void*>(static_cast<uintptr_t>(tokenRowId));
    void* lmemHandle = reinterpret_cast<void*>(0x1U);
    EXPECT_EQ(-1, RaCtxLmemRegister(reinterpret_cast<void*>(0xBEEFU), &reg, &lmemHandle));
    EXPECT_EQ(reinterpret_cast<void*>(0x1U), lmemHandle);
    EXPECT_EQ(before, TableCount<sim::runtime::RaLmem>());
}

/// 不存在的远端 RaLmem：Import 明确失败且不产生 RaRmem 行。
TEST_F(HccpRmemStubTest, ImportMissingRemoteLmemFailsAndWritesNothing)
{
    const uint64_t before = TableCount<sim::runtime::RaRmem>();
    struct MrImportInfoT info = {};
    const uint64_t missing = 0xDEADU;
    std::memcpy(info.in.key.value, &missing, sizeof(missing));
    info.in.key.size = sizeof(missing);
    void* handle = reinterpret_cast<void*>(0x1U);
    EXPECT_EQ(-1, RaCtxRmemImport(reinterpret_cast<void*>(0x4444U), &info, &handle));
    EXPECT_EQ(reinterpret_cast<void*>(0x1U), handle);
    EXPECT_EQ(before, TableCount<sim::runtime::RaRmem>());
}
