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

// ==================== 对称内存统一 reserve/map/unmap 模型回归 ====================
// 新合同：reserve 只发跨 rank 同址对称 VA、不写 DB；aclrtMapMem 是唯一写表点，
// 行内 start_ptr 为 map 时另发的按设备唯一本地址（与对称 VA 不同），dev_mapped_ptr
// 为入参对称 VA；aclrtUnmapMem 物理删除该行；aclrtReleaseMemAddress 为 no-op。
// 本组用例锁定 Map/Unmap/Release 在该模型下的可见性与幂等语义。

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

    /// 按 Map/Unmap 使用的谓词查找本设备窗口行：(dev_mapped_ptr==对称VA, src_type=DEV, device=当前设备)
    static std::optional<sim::runtime::VirtualMemBlock> FindWindowBySymVa(uint64_t symVa)
    {
        auto found = runnerdb_test::SelectFirstRecord<sim::runtime::VirtualMemBlock>(HcclSim::Storage::And(
            HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::dev_mapped_ptr, symVa),
            HcclSim::Storage::Eq(
                &sim::runtime::VirtualMemBlock::src_type, static_cast<uint8_t>(sim::runtime::VIR_MEM_TYPE_DEV)),
            HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::device_id, sim::runtime::GetCurrDeviceKey())));
        if (!found.second) {
            return std::nullopt;
        }
        return found.first;
    }

    /// 构造并插入一块本设备物理块，返回其主键 id（供 aclrtMapMem handle 使用）
    uint64_t InsertLocalPhy(const char* name, uint64_t size)
    {
        sim::runtime::PhyMemBlock phyMem{};
        phyMem.device_id = deviceId;
        snprintf(phyMem.name, sizeof(phyMem.name), "%s", name);
        phyMem.size = size;
        phyMem.ref_count = 1;
        return runnerdb_test::InsertRecord<sim::runtime::PhyMemBlock>(phyMem);
    }
};

constexpr uint64_t kBit63SymBase = 1ULL << 63;

// reserve 只发对称 VA、不落任何 DEV 行；release 为 no-op（含未 map 的失败清理路径）。
TEST_F(AclrtReserveOwnershipTest, ReserveOnlyIssuesSymVaWithoutWritingRow)
{
    void* virPtr = nullptr;
    ASSERT_EQ(aclrtReserveMemAddress(&virPtr, 0x10000, 0, nullptr, 1), ACL_SUCCESS);
    ASSERT_NE(virPtr, nullptr);
    const uint64_t symVa = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(virPtr));
    EXPECT_GE(symVa, kBit63SymBase) << "必须发 bit63 GLOBAL 对称 VA";
    EXPECT_FALSE(FindWindowBySymVa(symVa).has_value()) << "reserve 阶段不得写映射行";

    // 未 map 直接 release（对应 hcomm malloc/map 失败的清理路径）：no-op 成功，仍无行。
    ASSERT_EQ(aclrtReleaseMemAddress(virPtr), ACL_SUCCESS);
    EXPECT_FALSE(FindWindowBySymVa(symVa).has_value());
    // 重复 release 同样成功（no-op 幂等）。
    ASSERT_EQ(aclrtReleaseMemAddress(virPtr), ACL_SUCCESS);
}

// 完整 map/unmap/release：map 是唯一写表点，start_ptr 另发唯一本地址，unmap 物理删行。
TEST_F(AclrtReserveOwnershipTest, MapWritesWindowRowUnmapHardDeletesReleaseNoop)
{
    void* virPtr = nullptr;
    ASSERT_EQ(aclrtReserveMemAddress(&virPtr, 0x10000, 0, nullptr, 1), ACL_SUCCESS);
    const uint64_t symVa = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(virPtr));
    const uint64_t phyMemId = InsertLocalPhy("reserve_lifecycle_phy", 0x10000);
    ASSERT_NE(phyMemId, 0U);

    ASSERT_EQ(
        aclrtMapMem(virPtr, 0x10000, 0, reinterpret_cast<aclrtDrvMemHandle>(static_cast<uintptr_t>(phyMemId)), 0),
        ACL_SUCCESS);
    auto mapped = FindWindowBySymVa(symVa);
    ASSERT_TRUE(mapped.has_value());
    EXPECT_EQ(mapped->phy_mem_id, phyMemId);
    EXPECT_EQ(mapped->dev_mapped_ptr, symVa);
    EXPECT_NE(mapped->start_ptr, symVa) << "start_ptr 必须是另发的按设备唯一本地址，不等于对称 VA";
    EXPECT_NE(mapped->start_ptr, 0U);
    EXPECT_EQ(mapped->device_id, deviceId);
    EXPECT_EQ(mapped->ctx_id, ctxId);
    EXPECT_EQ(mapped->rank_id, deviceId);
    EXPECT_EQ(mapped->src_type, static_cast<uint8_t>(sim::runtime::VIR_MEM_TYPE_DEV));
    EXPECT_EQ(mapped->is_dev_access, 1U);
    EXPECT_EQ(mapped->size, 0x10000U);

    // Unmap：物理删行（不再软删保留）。
    ASSERT_EQ(aclrtUnmapMem(virPtr), ACL_SUCCESS);
    EXPECT_FALSE(FindWindowBySymVa(symVa).has_value());

    // Release：no-op，行保持已删除。
    ASSERT_EQ(aclrtReleaseMemAddress(virPtr), ACL_SUCCESS);
    EXPECT_FALSE(FindWindowBySymVa(symVa).has_value());
}

// 未 unmap 就对同一对称 VA 再次 map：区间重复占用，直接报错且原行不被改动。
TEST_F(AclrtReserveOwnershipTest, RemapSameSymVaWithoutUnmapIsRejected)
{
    void* virPtr = nullptr;
    ASSERT_EQ(aclrtReserveMemAddress(&virPtr, 0x10000, 0, nullptr, 1), ACL_SUCCESS);
    const uint64_t symVa = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(virPtr));
    const uint64_t phyA = InsertLocalPhy("remap_phy_a", 0x10000);
    const uint64_t phyB = InsertLocalPhy("remap_phy_b", 0x10000);
    ASSERT_NE(phyA, 0U);
    ASSERT_NE(phyB, 0U);

    auto handleOf = [](uint64_t id) {
        return reinterpret_cast<aclrtDrvMemHandle>(static_cast<uintptr_t>(id));
    };
    ASSERT_EQ(aclrtMapMem(virPtr, 0x10000, 0, handleOf(phyA), 0), ACL_SUCCESS);
    const uint64_t firstStart = FindWindowBySymVa(symVa)->start_ptr;

    // 重复占用必须报错（重映射须先 unmap），且不得 update、不得插第二行。
    EXPECT_EQ(aclrtMapMem(virPtr, 0x10000, 0, handleOf(phyB), 0), ACL_ERROR_INVALID_PARAM);

    auto rows = runnerdb_test::SelectRecords<sim::runtime::VirtualMemBlock>(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::dev_mapped_ptr, symVa),
        HcclSim::Storage::Eq(
            &sim::runtime::VirtualMemBlock::src_type, static_cast<uint8_t>(sim::runtime::VIR_MEM_TYPE_DEV)),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::device_id, deviceId)));
    ASSERT_EQ(rows.size(), 1U) << "重复 map 被拒绝后仍只有一行";
    EXPECT_EQ(rows[0].phy_mem_id, phyA) << "原行物理块不得被改动";
    EXPECT_EQ(rows[0].start_ptr, firstStart);

    // 清理：unmap 硬删本用例写入的窗口行，避免 DB 行跨用例残留。
    ASSERT_EQ(aclrtUnmapMem(virPtr), ACL_SUCCESS);
}

// map 入参校验：非法 VA（非 bit62/bit63 段）、窗口长度超过物理块长度，均拒绝。
TEST_F(AclrtReserveOwnershipTest, MapRejectsInvalidVaAndOversizeWindow)
{
    const uint64_t phyMemId = InsertLocalPhy("map_validate_phy", 0x10000);
    ASSERT_NE(phyMemId, 0U);
    auto handleOf = [](uint64_t id) {
        return reinterpret_cast<aclrtDrvMemHandle>(static_cast<uintptr_t>(id));
    };

    // 非对称段 VA（普通按卡编址 0x80_0000000000 段）不得走对称 map。
    void* lowVa = reinterpret_cast<void*>(static_cast<uintptr_t>(0x8000000000ULL));
    EXPECT_EQ(aclrtMapMem(lowVa, 0x10000, 0, handleOf(phyMemId), 0), ACL_ERROR_INVALID_PARAM);

    // 合法对称 VA，但窗口尺寸大于物理块。
    void* virPtr = nullptr;
    ASSERT_EQ(aclrtReserveMemAddress(&virPtr, 0x20000, 0, nullptr, 1), ACL_SUCCESS);
    const uint64_t bigSymVa = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(virPtr));
    EXPECT_EQ(aclrtMapMem(virPtr, 0x20000, 0, handleOf(phyMemId), 0), ACL_ERROR_INVALID_PARAM);
    EXPECT_FALSE(FindWindowBySymVa(bigSymVa).has_value()) << "拒绝后不得落行";
}

// 对从未映射的对称 VA 调 unmap/release：均幂等成功（不报错）。
TEST_F(AclrtReserveOwnershipTest, UnmapAndReleaseNeverMappedSymVaAreIdempotent)
{
    void* virPtr = nullptr;
    ASSERT_EQ(aclrtReserveMemAddress(&virPtr, 0x10000, 0, nullptr, 1), ACL_SUCCESS);
    const uint64_t symVa = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(virPtr));
    ASSERT_EQ(aclrtUnmapMem(virPtr), ACL_SUCCESS);
    EXPECT_FALSE(FindWindowBySymVa(symVa).has_value());
    ASSERT_EQ(aclrtReleaseMemAddress(virPtr), ACL_SUCCESS);
    EXPECT_FALSE(FindWindowBySymVa(symVa).has_value());
}

// unmap 后同对称 VA 再次 map：旧行已物理删除，新行重建并由 AllocVirMem 另发一个全新 start_ptr。
TEST_F(AclrtReserveOwnershipTest, RemapAfterUnmapAllocatesFreshUniqueStart)
{
    void* virPtr = nullptr;
    ASSERT_EQ(aclrtReserveMemAddress(&virPtr, 0x10000, 0, nullptr, 1), ACL_SUCCESS);
    const uint64_t symVa = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(virPtr));
    const uint64_t phyA = InsertLocalPhy("recycle_phy_a", 0x10000);
    const uint64_t phyB = InsertLocalPhy("recycle_phy_b", 0x10000);
    ASSERT_NE(phyA, 0U);
    ASSERT_NE(phyB, 0U);

    auto handleOf = [](uint64_t id) {
        return reinterpret_cast<aclrtDrvMemHandle>(static_cast<uintptr_t>(id));
    };
    ASSERT_EQ(aclrtMapMem(virPtr, 0x10000, 0, handleOf(phyA), 0), ACL_SUCCESS);
    const uint64_t firstStart = FindWindowBySymVa(symVa)->start_ptr;
    ASSERT_EQ(aclrtUnmapMem(virPtr), ACL_SUCCESS);
    EXPECT_FALSE(FindWindowBySymVa(symVa).has_value());

    ASSERT_EQ(aclrtMapMem(virPtr, 0x10000, 0, handleOf(phyB), 0), ACL_SUCCESS);
    auto second = FindWindowBySymVa(symVa);
    ASSERT_TRUE(second.has_value());
    EXPECT_NE(second->start_ptr, firstStart) << "AllocVirMem 游标单调递增、不复用，重映射必须拿到全新地址";
    EXPECT_EQ(second->phy_mem_id, phyB);
    EXPECT_EQ(second->dev_mapped_ptr, symVa);

    // 清理重建后仍存活的窗口。
    ASSERT_EQ(aclrtUnmapMem(virPtr), ACL_SUCCESS);
}
