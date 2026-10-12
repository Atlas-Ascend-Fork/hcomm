/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 主机侧单测：覆盖大块引流、上界报错、aclrtFreeHost 身份判定。
// MallocHost 走真实 MemoryManager 和 HcclCommPool 共享内存，不用 mock。
// memcpy/memset 的设备侧短路判定由 CommPoolPolicy::ShouldRedirect 的纯函数用例覆盖边界。
// 本文件聚焦不依赖 RunnerDB/Runner 上下文的主机侧入口。

#include <cstdint>
#include <cstring>
#include <sys/mman.h> // shm_unlink
#include <gtest/gtest.h>

#include "acl/acl_base.h"
#include "acl/acl_rt.h"
#include "store_sim_comm_pool_policy.h"
#include "store_sim_device_memory_manager.h"
#include "store_sim_memory_manager.h"
#include "store_sim_run_mode.h"
#include "storage/internal/process_storage_context.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "storage/storage_session.h"
#include "runtime_state/sim_models.h"
#include "sqlite/simulation_storage_test_helper.h"

extern "C" {
aclError aclrtMallocHost(void** hostPtr, size_t size);
aclError aclrtFreeHost(void* hostPtr);
aclError aclrtMallocHostWithCfg(void** ptr, uint64_t size, aclrtMallocConfig* cfg);
}

// 复用区在整个套件期间保持存活，套件开头建池一次，结束再回收。
class AclrtMemStubTest : public testing::Test {
protected:
    static void SetUpTestSuite()
    {
        // 在套件初始化（已过 main、RunnerDB/sqlite 就绪）时把 mode=1 写进 DB，再主动触发一次
        // IsCheckOnlyMode() 的懒加载，使进程内 static 缓存 latch 成仅校验模式；不依赖任何加载期构造器。
        runnerdb_test::ClearRecords<sim::runtime::RunModeConfig>();
        sim::runtime::RunModeConfig cfg{};
        cfg.mode = 1;
        runnerdb_test::InsertRecord<sim::runtime::RunModeConfig>(cfg);
        ASSERT_TRUE(sim::IsCheckOnlyMode());
        // 清掉上次异常退出残留的 /dev/shm/HcclCommPool（ShmCreate 用 O_EXCL，残留会建池失败）。
        shm_unlink(sim::CommPoolPolicy::kPoolName);
        ASSERT_NE(
            sim::MemoryManager::GetInstance().AllocMemByName(
                sim::CommPoolPolicy::kPoolName, sim::CommPoolPolicy::kPoolSize),
            nullptr);
    }
    static void TearDownTestSuite()
    {
        // 关闭并 unlink，保证 /dev/shm 不泄漏，独立重跑不撞 O_EXCL。
        sim::MemoryManager::GetInstance().FreeMemByName(sim::CommPoolPolicy::kPoolName);
        // 清掉本套件写进共享 DB 的仅校验模式行，避免给其它测试二进制留下 check-only(mode=1)。
        runnerdb_test::ClearRecords<sim::runtime::RunModeConfig>();
    }
};

// 主机大块两次申请归同一池首址，且与设备侧大块同址，主机与设备共用 HcclCommPool。
TEST_F(AclrtMemStubTest, MallocHost_BigBlock_TwiceSameAddr_SharesDevicePool)
{
    EXPECT_TRUE(sim::IsCheckOnlyMode()); // SetUpTestSuite 已写入 mode=1 并预热缓存，须为 true。
    const size_t big = sim::CommPoolPolicy::kBigBlockThreshold; // 200MB
    void* h1 = nullptr;
    void* h2 = nullptr;
    ASSERT_EQ(aclrtMallocHost(&h1, big), ACL_SUCCESS);
    ASSERT_EQ(aclrtMallocHost(&h2, big), ACL_SUCCESS);
    ASSERT_NE(h1, nullptr);
    EXPECT_EQ(h1, h2); // 两次大块同址

    // 设备侧大块也归同一池基址
    auto& mgr = sim::DeviceMemoryManager::GetInstance();
    void* d = mgr.AllocPhyMem("host_share_probe", 0, big);
    EXPECT_EQ(h1, d); // 主机与设备共用同一池
    mgr.FreePhyMem("host_share_probe", 0);

    EXPECT_EQ(aclrtFreeHost(h1), ACL_SUCCESS);
    EXPECT_EQ(aclrtFreeHost(h2), ACL_SUCCESS);
}

// 主机小块走真实 malloc，独立于池，内容正确。
TEST_F(AclrtMemStubTest, MallocHost_SmallBlock_RealAlloc_NotPool)
{
    void* big = nullptr;
    ASSERT_EQ(aclrtMallocHost(&big, sim::CommPoolPolicy::kBigBlockThreshold), ACL_SUCCESS);

    void* s1 = nullptr;
    void* s2 = nullptr;
    ASSERT_EQ(aclrtMallocHost(&s1, 4096), ACL_SUCCESS);
    ASSERT_EQ(aclrtMallocHost(&s2, 4096), ACL_SUCCESS);
    ASSERT_NE(s1, nullptr);
    ASSERT_NE(s2, nullptr);
    EXPECT_NE(s1, s2);  // 两个小块各自独立分配
    EXPECT_NE(s1, big); // 小块不进池

    // 小块内容正确，互不覆盖
    memset(s1, 0xAA, 4096);
    memset(s2, 0xBB, 4096);
    EXPECT_EQ(static_cast<unsigned char*>(s1)[0], 0xAAu);
    EXPECT_EQ(static_cast<unsigned char*>(s2)[0], 0xBBu);

    EXPECT_EQ(aclrtFreeHost(s1), ACL_SUCCESS);
    EXPECT_EQ(aclrtFreeHost(s2), ACL_SUCCESS);
    EXPECT_EQ(aclrtFreeHost(big), ACL_SUCCESS);
}

// 主机大块 >4GB 报错，不引流也不真实分配。
TEST_F(AclrtMemStubTest, MallocHost_ExceedCeiling_Reject)
{
    void* p = reinterpret_cast<void*>(0xDEADBEEF); // 哨兵：失败时不应被改写
    aclError ret = aclrtMallocHost(&p, sim::CommPoolPolicy::kPoolSize + 1);
    EXPECT_NE(ret, ACL_SUCCESS);
    EXPECT_EQ(p, reinterpret_cast<void*>(0xDEADBEEF)); // 报错路径未写出指针
}

// aclrtFreeHost 身份判定：池内地址 noop，池外真实地址正常 free。
TEST_F(AclrtMemStubTest, FreeHost_PoolAddrNoop_RealAddrFree)
{
    const size_t big = sim::CommPoolPolicy::kBigBlockThreshold;
    void* poolPtr = nullptr;
    ASSERT_EQ(aclrtMallocHost(&poolPtr, big), ACL_SUCCESS);
    ASSERT_NE(poolPtr, nullptr);

    // 池内地址 free 为 noop：返回成功且池仍可写读
    EXPECT_EQ(aclrtFreeHost(poolPtr), ACL_SUCCESS);
    const char* sentinel = "pool-alive-after-noop-free";
    memcpy(poolPtr, sentinel, strlen(sentinel) + 1);
    EXPECT_STREQ(static_cast<char*>(poolPtr), sentinel);

    // 池外真实地址：正常 free，不抛异常
    void* real = nullptr;
    ASSERT_EQ(aclrtMallocHost(&real, 4096), ACL_SUCCESS);
    EXPECT_EQ(aclrtFreeHost(real), ACL_SUCCESS);
}

// MallocHostWithCfg 委托 MallocHost：大块同样引流到池。
TEST_F(AclrtMemStubTest, MallocHostWithCfg_BigBlock_DelegatesToPool)
{
    void* viaPlain = nullptr;
    void* viaCfg = nullptr;
    const size_t big = sim::CommPoolPolicy::kBigBlockThreshold;
    ASSERT_EQ(aclrtMallocHost(&viaPlain, big), ACL_SUCCESS);
    ASSERT_EQ(aclrtMallocHostWithCfg(&viaCfg, big, nullptr), ACL_SUCCESS);
    EXPECT_EQ(viaPlain, viaCfg); // 两条入口归同一池
    aclrtFreeHost(viaPlain);
    aclrtFreeHost(viaCfg);
}

// ==================== Reserve 设备归属回归（任务书 v3 §C/回归清单7）====================
// aclrtReserveMemAddress 曾遗漏 virMem.device_id（结构零初始化 → 0），而
// Release/MapMem/Unmap 均按 (start_ptr, src_type=DEV, device_id=当前设备) 查找，
// 预留块对本设备不可见；这也是 VMB 启用 PartitionBy(device_id) 前必须补齐的
// 写入归属。本组用例锁定：Reserve 行 device_id == Context.device_id，且既有
// 生命周期入口（Map/Unmap/Release）按原谓词能查到并更新/删除该记录。

extern "C" {
aclError aclrtReserveMemAddress(void** virPtr, size_t size, size_t alignment, void* expectPtr, uint64_t flags);
aclError aclrtReleaseMemAddress(void* virPtr);
aclError aclrtMapMem(void* virPtr, size_t size, size_t offset, aclrtDrvMemHandle handle, uint64_t flags);
aclError aclrtUnmapMem(void* virPtr);
}

// db_sim_runner_common.cc 的全局当前 server key（GetCurServerId 短路入口）
extern uint64_t g_cur_server_key;

namespace {
const std::string kReserveDbPath = "/tmp/test_aclrt_reserve_ownership.db";
}

class AclrtReserveOwnershipTest : public testing::Test {
protected:
    uint64_t serverId{0};
    uint64_t deviceId{0};
    uint64_t ctxId{0};

    void SetUp() override
    {
        ASSERT_TRUE(runnerdb_test::ResetTestSession(kReserveDbPath, kReserveDbPath));

        sim::runtime::Server server{};
        server.pod_id = 100;
        serverId = runnerdb_test::InsertRecord<sim::runtime::Server>(server);
        ASSERT_NE(serverId, 0U);
        sim::runtime::Host host{};
        host.server_id = serverId;
        host.arch = 1;
        ASSERT_NE(runnerdb_test::InsertRecord<sim::runtime::Host>(host), 0U);
        sim::runtime::Device device{};
        device.server_id = serverId;
        device.logic_id = 0;
        device.physical_id = 0;
        deviceId = runnerdb_test::InsertRecord<sim::runtime::Device>(device);
        ASSERT_NE(deviceId, 0U);

        // 当前 server + runner TLS（GetCurrRunnerTls 惰性插入 Runner 行并置 TLS）
        g_cur_server_key = serverId;
        sim::runtime::Runner runner{};
        ASSERT_TRUE(sim::runtime::GetCurrRunnerTls(serverId, runner));

        // 当前上下文绑定到本设备（Reserve 的 device_id 来源 = Context.device_id）
        sim::runtime::Context context{};
        context.device_id = deviceId;
        context.is_default = 1;
        context.ref_cnt = 1;
        ctxId = runnerdb_test::InsertRecord<sim::runtime::Context>(context);
        ASSERT_NE(ctxId, 0U);
        ASSERT_TRUE(sim::runtime::SetCurrCtxTls(ctxId));
    }

    void TearDown() override
    {
        g_cur_server_key = 0;
        runnerdb_test::CleanUpDatabases(kReserveDbPath, kReserveDbPath);
    }

    /// 按 Release/MapMem/Unmap 使用的原谓词查找行（不放宽任何条件）
    static std::optional<sim::runtime::VirtualMemBlock> FindByLifecyclePredicate(uint64_t startPtr)
    {
        auto found = runnerdb_test::SelectFirstRecord<sim::runtime::VirtualMemBlock>(HcclSim::Storage::And(
            HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::start_ptr, startPtr),
            HcclSim::Storage::Eq(
                &sim::runtime::VirtualMemBlock::src_type, static_cast<uint8_t>(sim::runtime::VIR_MEM_TYPE_DEV)),
            HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::device_id, sim::runtime::GetCurrDeviceKey())));
        if (!found.second) {
            return std::nullopt;
        }
        return found.first;
    }
};

TEST_F(AclrtReserveOwnershipTest, ReserveWritesContextDeviceOwnership)
{
    void* virPtr = nullptr;
    ASSERT_EQ(aclrtReserveMemAddress(&virPtr, 0x10000, 0, nullptr, 0), ACL_SUCCESS);
    ASSERT_NE(virPtr, nullptr);

    // 行归属 = Context.device_id（修复前恒为 0）
    const uint64_t startPtr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(virPtr));
    auto row = FindByLifecyclePredicate(startPtr);
    ASSERT_TRUE(row.has_value()) << "既有 (start_ptr, src_type, device_id=当前设备) 谓词必须能查到预留行";
    EXPECT_EQ(row->device_id, deviceId);
    EXPECT_EQ(row->ctx_id, ctxId);
    EXPECT_EQ(row->phy_mem_id, 0U);
    EXPECT_EQ(row->src_type, static_cast<uint8_t>(sim::runtime::VIR_MEM_TYPE_DEV));
    EXPECT_EQ(row->size, 0x10000U);

    ASSERT_EQ(aclrtReleaseMemAddress(virPtr), ACL_SUCCESS);
    // 新合同（2026-09-17 拆卸竞态修复）：Release 后行保留并标记 is_freed=1
    // （不再物理删除）——拆卸期慢 rank 设备进程的迟到解析（FR1/R3K4 取证
    // 43/48 次未命中晚于成功删除 16.7s+）仍可按原谓词命中。
    auto released = FindByLifecyclePredicate(startPtr);
    ASSERT_TRUE(released.has_value()) << "Release 后行必须保留（软删标记）";
    EXPECT_EQ(released->is_freed, 1U);
}

TEST_F(AclrtReserveOwnershipTest, ReserveMapUnmapReleaseLifecycleKeepsOriginalPredicates)
{
    void* virPtr = nullptr;
    ASSERT_EQ(aclrtReserveMemAddress(&virPtr, 0x10000, 0, nullptr, 0), ACL_SUCCESS);
    const uint64_t startPtr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(virPtr));

    // MapMem：物理句柄主键写入 phy_mem_id（原谓词按 device_id=当前设备查到预留行）
    sim::runtime::PhyMemBlock phyMem{};
    phyMem.device_id = deviceId;
    snprintf(phyMem.name, sizeof(phyMem.name), "%s", "reserve_lifecycle_phy");
    phyMem.size = 0x10000;
    phyMem.ref_count = 1;
    const uint64_t phyMemId = runnerdb_test::InsertRecord<sim::runtime::PhyMemBlock>(phyMem);
    ASSERT_NE(phyMemId, 0U);
    ASSERT_EQ(
        aclrtMapMem(virPtr, 0x10000, 0, reinterpret_cast<aclrtDrvMemHandle>(static_cast<uintptr_t>(phyMemId)), 0),
        ACL_SUCCESS);
    auto mapped = FindByLifecyclePredicate(startPtr);
    ASSERT_TRUE(mapped.has_value());
    EXPECT_EQ(mapped->phy_mem_id, phyMemId);

    // UnmapMem：phy_mem_id 归零，行仍可按原谓词查到
    ASSERT_EQ(aclrtUnmapMem(virPtr), ACL_SUCCESS);
    auto unmapped = FindByLifecyclePredicate(startPtr);
    ASSERT_TRUE(unmapped.has_value());
    EXPECT_EQ(unmapped->phy_mem_id, 0U);

    // Release：行保留并标记 is_freed=1（新合同，同 ReserveWritesContextDeviceOwnership）
    ASSERT_EQ(aclrtReleaseMemAddress(virPtr), ACL_SUCCESS);
    auto released = FindByLifecyclePredicate(startPtr);
    ASSERT_TRUE(released.has_value());
    EXPECT_EQ(released->is_freed, 1U);
}

TEST_F(AclrtReserveOwnershipTest, LegacyZeroOwnershipRowStaysInvisibleToLifecycle)
{
    // 负对照（修复前 Reserve 行的形状）：device_id=0 的行对当前设备的
    // Release/MapMem 谓词不可见——证明归属字段是生命周期查得的必要条件，
    // 也说明补齐写入不会"顺带"改变零归属脏行的既有可见性语义
    sim::runtime::VirtualMemBlock orphan{};
    orphan.start_ptr = 0x600000000ULL;
    orphan.size = 0x1000;
    orphan.ctx_id = ctxId;
    orphan.device_id = 0;
    orphan.phy_mem_id = 0;
    orphan.src_type = static_cast<uint8_t>(sim::runtime::VIR_MEM_TYPE_DEV);
    ASSERT_NE(runnerdb_test::InsertRecord<sim::runtime::VirtualMemBlock>(orphan), 0U);

    EXPECT_EQ(
        aclrtReleaseMemAddress(reinterpret_cast<void*>(static_cast<uintptr_t>(0x600000000ULL))),
        ACL_ERROR_INTERNAL_ERROR);
    EXPECT_EQ(aclrtUnmapMem(reinterpret_cast<void*>(static_cast<uintptr_t>(0x600000000ULL))), ACL_ERROR_INTERNAL_ERROR);
}

// ==================== is_freed 软删新合同（2026-09-17 拆卸竞态修复）====================
// 释放 = 标记 is_freed=1（行保留），插入前 purge 同设备标记行（物理删除）。
// 双释放幂等：旧合同靠"行已删 NOT_FOUND"提前失败；标记后行对等值读可见，
// 由 aclrtReleaseMemAddress 入口的 is_freed 守卫等价恢复。
TEST_F(AclrtReserveOwnershipTest, ReleaseTwice_FailsOnSecondRelease)
{
    void* virPtr = nullptr;
    ASSERT_EQ(aclrtReserveMemAddress(&virPtr, 0x10000, 0, nullptr, 0), ACL_SUCCESS);
    ASSERT_NE(virPtr, nullptr);

    // 第一次释放成功并标记
    ASSERT_EQ(aclrtReleaseMemAddress(virPtr), ACL_SUCCESS);
    const uint64_t startPtr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(virPtr));
    auto released = FindByLifecyclePredicate(startPtr);
    ASSERT_TRUE(released.has_value());
    EXPECT_EQ(released->is_freed, 1U);

    // 第二次释放：等值读命中标记行，守卫按旧合同提前失败
    EXPECT_EQ(aclrtReleaseMemAddress(virPtr), ACL_ERROR_INTERNAL_ERROR);
}

// 同设备再分配触发 purge：标记行物理消失、新行 is_freed=0——钉住
// "标记行与活跃行永不同址共存"不变量（GetOneByPred 单命中无歧义）。
TEST_F(AclrtReserveOwnershipTest, ReallocAfterRelease_PurgesMarkedRow_NewRowActive)
{
    void* first = nullptr;
    ASSERT_EQ(aclrtReserveMemAddress(&first, 0x10000, 0, nullptr, 0), ACL_SUCCESS);
    const uint64_t firstPtr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(first));
    auto firstRow = FindByLifecyclePredicate(firstPtr);
    ASSERT_TRUE(firstRow.has_value());
    const uint64_t firstRowId = firstRow->id;
    ASSERT_EQ(aclrtReleaseMemAddress(first), ACL_SUCCESS);

    // 同设备再分配：插入前 purge 物理清除本设备标记行
    void* second = nullptr;
    ASSERT_EQ(aclrtReserveMemAddress(&second, 0x10000, 0, nullptr, 0), ACL_SUCCESS);
    ASSERT_NE(second, nullptr);

    auto purged = runnerdb_test::SelectFirstRecord<sim::runtime::VirtualMemBlock>(
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::id, firstRowId));
    EXPECT_FALSE(purged.second) << "同设备再分配后旧标记行必须物理消失";

    const uint64_t secondPtr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(second));
    auto secondRow = FindByLifecyclePredicate(secondPtr);
    ASSERT_TRUE(secondRow.has_value()) << "新行必须按原生命周期谓词可见";
    EXPECT_EQ(secondRow->is_freed, 0U);
    EXPECT_EQ(secondRow->device_id, deviceId);

    // 新行生命周期完整：可再次释放（守卫只拦已标记行）
    ASSERT_EQ(aclrtReleaseMemAddress(second), ACL_SUCCESS);
}
