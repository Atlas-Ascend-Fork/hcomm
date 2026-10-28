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
 * UT: 资源树窄访问器（src/base_comm/base_comm_node_accessors.h）
 * 约束：严禁调用 BaseCommMgr::DeInit（共享 UT 进程，会污染其他用例）；
 * 越界回落的 reserve 实例为静态对象，只做引用身份只读校验。
 */

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>

#define private public
#define protected public
#include "base_comm_node_accessors.h"
#include "thread_manager.h"
#undef protected
#undef private

#include "acl/acl_rt.h"
#include "adapter_rts_common.h"
#include "dfx/dfx_procs.h"
#include "hccl_common.h"
#include "launch_device.h"
#include "comm_configer.h"
#include "hccp_hdc_manager.h"
#include "hccp_peer_manager.h"
#include "hccp_tlv_hdc_manager.h"
#include "host_socket_handle_manager.h"
#include "inner_net_dev_manager.h"
#include "rdma_handle_manager.h"
#include "socket_handle_manager.h"

using namespace hcomm;
using hccl::LoadBinaryFromFile;

namespace {

// LoadBinaryFromFile 打桩：计数 + 填充伪句柄，观测 AICPU bin 加载次数
std::atomic<uint32_t> g_loadBinaryCallCount{0};

HcclResult StubLoadBinaryFromFile(
    const char* binPath, aclrtBinaryLoadOptionType loadType, uint32_t cpuKernelMode, aclrtBinHandle& binHandle)
{
    (void)binPath;
    (void)loadType;
    (void)cpuKernelMode;
    g_loadBinaryCallCount.fetch_add(1, std::memory_order_relaxed);
    binHandle = reinterpret_cast<aclrtBinHandle>(0x1);
    return HCCL_SUCCESS;
}

// phyId/logicId 恒等映射打桩：合法键落到同编号设备子树
HcclResult StubHrtGetDevicePhyIdByIndexIdentity(uint32_t deviceLogicId, uint32_t& devicePhyId, bool isRefresh)
{
    (void)isRefresh;
    devicePhyId = deviceLogicId;
    return HCCL_SUCCESS;
}

HcclResult StubHrtGetDeviceIndexByPhyIdIdentity(uint32_t devicePhyId, uint32_t& deviceLogicId)
{
    deviceLogicId = devicePhyId;
    return HCCL_SUCCESS;
}

} // namespace

class ResTreeAccessorsTest : public testing::Test {
protected:
    void SetUp() override { GlobalMockObject::reset(); }
    void TearDown() override
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
    }
};

// 用例 1：BaseCommMgr 为 Meyers 单例，多次 GetInstance 返回同一地址
TEST_F(ResTreeAccessorsTest, Ut_When_BaseCommMgrGetInstanceRepeated_Expect_SameAddress)
{
    BaseCommMgr& first = BaseCommMgr::GetInstance();
    BaseCommMgr& second = BaseCommMgr::GetInstance();
    EXPECT_EQ(&first, &second);
}

// 用例 2：锚定冒烟——树根构造后 8 个进程级锚定单例逐一 GetInstance 有效
TEST_F(ResTreeAccessorsTest, Ut_When_BaseCommMgrAlive_Expect_AnchorSingletonsValid)
{
    (void)BaseCommMgr::GetInstance(); // 触发 AnchorProcessSingletons
    EXPECT_EQ(&hccl::CommConfiger::GetInstance(), &hccl::CommConfiger::GetInstance());
    EXPECT_EQ(&Hccl::HccpHdcManager::GetInstance(), &Hccl::HccpHdcManager::GetInstance());
    EXPECT_EQ(&Hccl::HccpPeerManager::GetInstance(), &Hccl::HccpPeerManager::GetInstance());
    EXPECT_EQ(&Hccl::HccpTlvHdcManager::GetInstance(), &Hccl::HccpTlvHdcManager::GetInstance());
    EXPECT_EQ(&Hccl::RdmaHandleManager::GetInstance(), &Hccl::RdmaHandleManager::GetInstance());
    EXPECT_EQ(&Hccl::InnerNetDevManager::GetInstance(), &Hccl::InnerNetDevManager::GetInstance());
    EXPECT_EQ(&Hccl::SocketHandleManager::GetInstance(), &Hccl::SocketHandleManager::GetInstance());
    EXPECT_EQ(&Hccl::HostSocketHandleManager::GetInstance(), &Hccl::HostSocketHandleManager::GetInstance());
}

// 用例 3/5/8：devPhyId=0 正常键，EidInfoMgr/EndpointCtxMgr/HccpTlvHdcMgr 返回有效且稳定的引用
TEST_F(ResTreeAccessorsTest, Ut_When_ValidPhyIdZero_Expect_KeyedAccessorsValidRef)
{
    EXPECT_EQ(&GetEidInfoMgr(0), &GetEidInfoMgr(0));
    EXPECT_EQ(&GetEndpointCtxMgr(0), &GetEndpointCtxMgr(0));
    EXPECT_EQ(&GetHccpTlvHdcMgr(0), &GetHccpTlvHdcMgr(0));
}

// 用例 7：打桩 phyId→logicId 恒等映射后，同键两次访问 GetGlobalNetDevMgr 返回同一引用
TEST_F(ResTreeAccessorsTest, Ut_When_GlobalNetDevMgrSameKeyTwice_Expect_SameRef)
{
    MOCKER(hrtGetDeviceIndexByPhyId).stubs().will(invoke(StubHrtGetDeviceIndexByPhyIdIdentity));
    auto& first = GetGlobalNetDevMgr(0);
    auto& second = GetGlobalNetDevMgr(0);
    EXPECT_EQ(&first, &second);
}

// 用例 10：进程级访问器 GetDlUrmaFunction 返回有效引用
TEST_F(ResTreeAccessorsTest, Ut_When_GetDlUrmaFunction_Expect_ValidRef)
{
    EXPECT_EQ(&GetDlUrmaFunction(), &GetDlUrmaFunction());
}

// 用例 11：GetProcRegedMemMgrCacheHolder 为非 owning 别名（空 deleter），析构不释放实例
TEST_F(ResTreeAccessorsTest, Ut_When_ProcRegedMemMgrCacheHolder_Expect_NonOwningAlias)
{
    auto& cache = GetProcRegedMemMgrCache();
    auto holder = GetProcRegedMemMgrCacheHolder();
    EXPECT_EQ(holder.get(), &cache);
    EXPECT_EQ(holder.use_count(), 1);
    {
        auto holderCopy = holder;
        EXPECT_EQ(holder.use_count(), 2);
    }
    EXPECT_EQ(holder.use_count(), 1);
    holder.reset();
    EXPECT_EQ(&GetProcRegedMemMgrCache(), &cache); // 实例随树存活，不随 holder 析构
}

// 用例 14：GetThreadHandleRegistryProc 有效引用 + EngineThreadMap 登记/查询正常
TEST_F(ResTreeAccessorsTest, Ut_When_ThreadHandleRegistryProc_Expect_RegisterAndQueryWorks)
{
    auto& registry = GetThreadHandleRegistryProc();
    EXPECT_EQ(&registry, &GetThreadHandleRegistryProc());
    {
        std::lock_guard<std::shared_mutex> lock(registry.EngineThreadMapMtx());
        registry.EngineThreadMap().clear();
        registry.EngineThreadD2HMap().clear();
    }

    constexpr ThreadHandle kHostHandle = 0xABCDEF01;
    registry.EngineThreadMap().emplace(kHostHandle, std::shared_ptr<hccl::Thread>());
    EXPECT_EQ(registry.EngineThreadMap().count(kHostHandle), 1U);

    const DeviceThreadKey key{0, static_cast<ThreadHandle>(0x1234)};
    registry.EngineThreadD2HMap().emplace(key, kHostHandle);
    auto it = registry.EngineThreadD2HMap().find(key);
    ASSERT_TRUE(it != registry.EngineThreadD2HMap().end());
    EXPECT_EQ(it->second, kHostHandle);

    {
        std::lock_guard<std::shared_mutex> lock(registry.EngineThreadMapMtx());
        registry.EngineThreadMap().clear();
        registry.EngineThreadD2HMap().clear();
    }
    EXPECT_TRUE(registry.EngineThreadMap().empty());
    EXPECT_TRUE(registry.EngineThreadD2HMap().empty());
}

// 用例 16：ThreadMgr::ResetThreadLocalNotifies 对不在句柄注册表的 dedicated 句柄容错跳过
TEST_F(ResTreeAccessorsTest, Ut_When_ResetNotifiesWithStaleDedicatedHandle_Expect_SkipAndSuccess)
{
    static int32_t fakeLogicId = 0;
    fakeLogicId = 0;
    MOCKER(hrtGetDevice).stubs().with(outBoundP(&fakeLogicId)).will(returnValue(HCCL_SUCCESS));

    hccl::ManagerCallbacks callbacks{};
    hccl::ThreadMgr mgr(1, 1, "ut_base_comm_node_accessors", nullptr, callbacks);
    constexpr ThreadHandle kStaleHandle = 0xDEADBEEF;
    mgr.dedicatedThreadMap_[HCCL_DED_THREAD_TYPE_AICPU_LAUNCH_GE] = kStaleHandle;

    // LookupThreadByHandle 未命中（Thread/notify 已 Free）应跳过，不中断整个 reset 流程
    EXPECT_EQ(mgr.ResetThreadLocalNotifies(), HCCL_SUCCESS);
    EXPECT_EQ(mgr.dedicatedThreadMap_.count(HCCL_DED_THREAD_TYPE_AICPU_LAUNCH_GE), 1U);

    mgr.dedicatedThreadMap_.clear(); // 伪句柄不得进入析构的 HcommThreadFree 路径
}

// 用例 17：EidInfoMgr 按 devPhyId 区分实例，DpuNotifyManager 进程级唯一，二者互不串位
TEST_F(ResTreeAccessorsTest, Ut_When_EidInfoMgrPerDeviceVsDpuNotifyProcLevel_Expect_NoCross)
{
    auto& eid0 = GetEidInfoMgr(0);
    auto& eid1 = GetEidInfoMgr(1);
    EXPECT_NE(&eid0, &eid1);

    auto& dpuFirst = GetDpuNotifyManager();
    auto& dpuSecond = GetDpuNotifyManager();
    EXPECT_EQ(&dpuFirst, &dpuSecond);
    EXPECT_NE(static_cast<const void*>(&eid0), static_cast<const void*>(&dpuFirst));

    EXPECT_EQ(&GetEidInfoMgr(0), &eid0); // 访问 DpuNotifyManager 后 EidInfoMgr 实例身份不变
}

// 用例 19：同 devPhyId 多次访问设备级访问器返回同一引用
TEST_F(ResTreeAccessorsTest, Ut_When_SameDevPhyIdRepeated_Expect_SameDeviceAccessorRef)
{
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().will(invoke(StubHrtGetDevicePhyIdByIndexIdentity));
    constexpr uint32_t phyId = 2;
    EXPECT_EQ(&GetEidInfoMgr(phyId), &GetEidInfoMgr(phyId));
    EXPECT_EQ(&GetEndpointCtxMgr(phyId), &GetEndpointCtxMgr(phyId));
    EXPECT_EQ(&GetSocketMgr(static_cast<int32_t>(phyId)), &GetSocketMgr(static_cast<int32_t>(phyId)));
    EXPECT_EQ(&GetTpMgr(phyId), &GetTpMgr(phyId));
    EXPECT_EQ(&GetHccpTlvHdcMgr(phyId), &GetHccpTlvHdcMgr(phyId));
    const int32_t logicId = static_cast<int32_t>(phyId);
    EXPECT_EQ(&GetCcuKernelMgr(logicId), &GetCcuKernelMgr(logicId));
    EXPECT_EQ(&GetCcuInstanceMgr(logicId), &GetCcuInstanceMgr(logicId));
    EXPECT_EQ(&GetCcuVarEventResMgr(logicId), &GetCcuVarEventResMgr(logicId));
}

// 用例 22：AICPU 首调加载、AICPU_TS 命中缓存、其余引擎不触发加载
TEST_F(ResTreeAccessorsTest, Ut_When_EnsureKernelBinLoaded_Expect_AicpuLoadOnceAndOtherEnginesSkip)
{
    auto& binProc = GetAicpuCommEngineBinProc();
    binProc.binHandle_ = nullptr; // 进程级实例跨用例共享，先复位句柄保证用例独立
    g_loadBinaryCallCount.store(0, std::memory_order_relaxed);
    MOCKER(LoadBinaryFromFile).stubs().will(invoke(StubLoadBinaryFromFile));

    EXPECT_EQ(binProc.EnsureKernelBinLoaded(COMM_ENGINE_AICPU), HCCL_SUCCESS);
    EXPECT_EQ(g_loadBinaryCallCount.load(std::memory_order_relaxed), 1U);
    EXPECT_NE(binProc.GetBinHandle(), nullptr);

    EXPECT_EQ(binProc.EnsureKernelBinLoaded(COMM_ENGINE_AICPU_TS), HCCL_SUCCESS);
    EXPECT_EQ(g_loadBinaryCallCount.load(std::memory_order_relaxed), 1U); // 命中缓存，未再加载

    EXPECT_EQ(binProc.EnsureKernelBinLoaded(COMM_ENGINE_CPU), HCCL_SUCCESS);
    EXPECT_EQ(g_loadBinaryCallCount.load(std::memory_order_relaxed), 1U); // 其余引擎不加载

    binProc.binHandle_ = nullptr; // 伪句柄不得遗留到共享进程的其他用例
}

// 用例 23：多线程并发 EnsureKernelBinLoaded(COMM_ENGINE_AICPU)，互斥锁保护下仅加载一次
TEST_F(ResTreeAccessorsTest, Ut_When_ConcurrentEnsureKernelBinLoaded_Expect_LoadExactlyOnce)
{
    auto& binProc = GetAicpuCommEngineBinProc();
    binProc.binHandle_ = nullptr;
    g_loadBinaryCallCount.store(0, std::memory_order_relaxed);
    MOCKER(LoadBinaryFromFile).stubs().will(invoke(StubLoadBinaryFromFile));

    constexpr uint32_t kThreadNum = 8;
    constexpr uint32_t kCallPerThread = 4;
    std::atomic<uint32_t> failCount{0};
    std::vector<std::thread> threads;
    threads.reserve(kThreadNum);
    for (uint32_t i = 0; i < kThreadNum; ++i) {
        threads.emplace_back([&binProc, &failCount]() {
            for (uint32_t j = 0; j < kCallPerThread; ++j) {
                if (binProc.EnsureKernelBinLoaded(COMM_ENGINE_AICPU) != HCCL_SUCCESS) {
                    failCount.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }
    for (auto& worker : threads) {
        worker.join();
    }
    EXPECT_EQ(failCount.load(std::memory_order_relaxed), 0U);
    EXPECT_EQ(g_loadBinaryCallCount.load(std::memory_order_relaxed), 1U);

    binProc.binHandle_ = nullptr;
}

// 用例 25：SocketMgr 越界回落 device 0 子树实例
TEST_F(ResTreeAccessorsTest, Ut_When_SocketMgrsOutOfRange_Expect_FallbackNoCrash)
{
    auto& socketDev0 = GetSocketMgr(0);
    EXPECT_EQ(&GetSocketMgr(static_cast<int32_t>(MAX_MODULE_DEVICE_NUM)), &socketDev0);
    EXPECT_EQ(&GetSocketMgr(-1), &socketDev0); // 负值经 u32 转换判越界，回落 device 0
}

// 用例 26/33：hrt 双向转换失败：GlobalNetDevMgr 按设备隔离落未 Init 实例，CCU 三件套回落共享 reserve
TEST_F(ResTreeAccessorsTest, Ut_When_HrtConvertFails_Expect_ReserveFallbackNoCrash)
{
    MOCKER(hrtGetDeviceIndexByPhyId).stubs().will(returnValue(HCCL_E_INTERNAL));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().will(returnValue(HCCL_E_INTERNAL));

    EXPECT_NE(&GetGlobalNetDevMgr(0), &GetGlobalNetDevMgr(3));
    EXPECT_EQ(&GetGlobalNetDevMgr(0), &GetGlobalNetDevMgr(0));

    // CCU 三件套：不同入参回落同一 reserve 实例
    EXPECT_EQ(&GetCcuKernelMgr(0), &GetCcuKernelMgr(1));
    EXPECT_EQ(&GetCcuInstanceMgr(0), &GetCcuInstanceMgr(1));
    EXPECT_EQ(&GetCcuVarEventResMgr(0), &GetCcuVarEventResMgr(1));
}

// 用例 27/37/38/39/40：越界键回落 reserve，reserve 引用与合法键引用不是同一个
TEST_F(ResTreeAccessorsTest, Ut_When_DeviceAccessorOutOfRange_Expect_ReserveDistinctFromValidRef)
{
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().will(invoke(StubHrtGetDevicePhyIdByIndexIdentity));
    constexpr int32_t validLogicId = 0;
    const int32_t outOfRangeLogicId = static_cast<int32_t>(MAX_MODULE_DEVICE_NUM);

    // logicId 键：MAX_MODULE_DEVICE_NUM 短路落 reserve；负值钳位到 backup id 后同落 reserve
    EXPECT_NE(&GetCcuKernelMgr(outOfRangeLogicId), &GetCcuKernelMgr(validLogicId));
    EXPECT_NE(&GetCcuInstanceMgr(outOfRangeLogicId), &GetCcuInstanceMgr(validLogicId));
    EXPECT_NE(&GetCcuVarEventResMgr(outOfRangeLogicId), &GetCcuVarEventResMgr(validLogicId));
    EXPECT_EQ(&GetCcuKernelMgr(-1), &GetCcuKernelMgr(outOfRangeLogicId));
    EXPECT_EQ(&GetCcuInstanceMgr(-1), &GetCcuInstanceMgr(outOfRangeLogicId));
    EXPECT_EQ(&GetCcuVarEventResMgr(-1), &GetCcuVarEventResMgr(outOfRangeLogicId));

    // devPhyId 键：65 取真实实例（与 0 号设备不同）；>65 钳位到 65 同实例
    EXPECT_NE(&GetTpMgr(MAX_MODULE_DEVICE_NUM), &GetTpMgr(0));
    EXPECT_NE(&GetHccpTlvHdcMgr(MAX_MODULE_DEVICE_NUM), &GetHccpTlvHdcMgr(0));
    EXPECT_EQ(&GetTpMgr(static_cast<uint32_t>(-1)), &GetTpMgr(MAX_MODULE_DEVICE_NUM)); // 钳位到 65 同实例
    EXPECT_EQ(&GetHccpTlvHdcMgr(MAX_MODULE_DEVICE_NUM + 1), &GetHccpTlvHdcMgr(MAX_MODULE_DEVICE_NUM));
}

// 用例 28：DfxProc::GetProfilingProc 懒创建，两次调用返回同一实例
TEST_F(ResTreeAccessorsTest, Ut_When_DfxProcGetProfilingProc_Expect_LazySameInstance)
{
    auto& dfxProc = BaseCommMgr::GetInstance().GetDfxProc();
    auto& first = dfxProc.GetProfilingProc();
    auto& second = dfxProc.GetProfilingProc();
    EXPECT_EQ(&first, &second);
}

// 用例 32：边界遍历——全部设备级访问器逐一喂 MAX_MODULE_DEVICE_NUM / MAX_MODULE_DEVICE_NUM+1
TEST_F(ResTreeAccessorsTest, Ut_When_DeviceAccessorsBoundaryKeys_Expect_NoCrash)
{
    // GlobalNetDevMgr 无键值短路，必须经 hrt 转换；打桩失败使其安全落 reserve
    MOCKER(hrtGetDeviceIndexByPhyId).stubs().will(returnValue(HCCL_E_INTERNAL));
    const uint32_t boundaryKeys[] = {MAX_MODULE_DEVICE_NUM, MAX_MODULE_DEVICE_NUM + 1};
    for (uint32_t key : boundaryKeys) {
        (void)GetEidInfoMgr(key);
        (void)GetEndpointCtxMgr(key);
        (void)GetSocketMgr(static_cast<int32_t>(key));
        (void)GetGlobalNetDevMgr(key);
        (void)GetTpMgr(key);
        (void)GetHccpTlvHdcMgr(key);
        const int32_t logicKey = static_cast<int32_t>(key);
        (void)GetCcuKernelMgr(logicKey);
        (void)GetCcuInstanceMgr(logicKey);
        (void)GetCcuVarEventResMgr(logicKey);
    }
}

// 用例 34：同键同引用，且正常实例与 reserve 实例互不串位
TEST_F(ResTreeAccessorsTest, Ut_When_SameKeyAndReserveIsolation_Expect_NoCross)
{
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().will(invoke(StubHrtGetDevicePhyIdByIndexIdentity));
    constexpr uint32_t phyId = 3;
    const int32_t logicId = static_cast<int32_t>(phyId);
    const int32_t reserveLogicId = static_cast<int32_t>(MAX_MODULE_DEVICE_NUM);

    EXPECT_EQ(&GetTpMgr(phyId), &GetTpMgr(phyId));
    EXPECT_EQ(&GetHccpTlvHdcMgr(phyId), &GetHccpTlvHdcMgr(phyId));
    EXPECT_EQ(&GetCcuKernelMgr(logicId), &GetCcuKernelMgr(logicId));

    EXPECT_NE(&GetTpMgr(phyId), &GetTpMgr(MAX_MODULE_DEVICE_NUM));
    EXPECT_NE(&GetHccpTlvHdcMgr(phyId), &GetHccpTlvHdcMgr(MAX_MODULE_DEVICE_NUM));
    EXPECT_NE(&GetCcuKernelMgr(logicId), &GetCcuKernelMgr(reserveLogicId));
    EXPECT_NE(&GetCcuInstanceMgr(logicId), &GetCcuInstanceMgr(reserveLogicId));
    EXPECT_NE(&GetCcuVarEventResMgr(logicId), &GetCcuVarEventResMgr(reserveLogicId));

    // 访问 reserve 后正常键实例身份不变
    EXPECT_EQ(&GetTpMgr(phyId), &GetTpMgr(phyId));
    EXPECT_EQ(&GetCcuKernelMgr(logicId), &GetCcuKernelMgr(logicId));
}
