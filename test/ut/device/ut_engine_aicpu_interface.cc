/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include <memory>
#include <string>
#include <vector>

#ifndef private
#define private public
#define protected public
#endif
#include "stream_utils.h"
#include "adapter_rts.h"
#include "env_config.h"
#include "op_base.h"
#include "comm_engine/aicpu_launch_manager_pub.h"
#include "aicpu_thread_kernel.h"
#include "aicpu_communicator.h"
#include "coll_comm_aicpu_mgr.h"
#include "coll_comm_aicpu.h"
#include "coll_comm_aicpu_kernel.h"
#include "coll_comm_aicpu_mgr.h"
#include "coll_comm_aicpu_kernel_adpt.h"
#include "threads/thread_aicpu_mgr.h"
#include "comm_engine/thread/device/device_thread_pub.h"
#include "aicpu_thread_process.h"
#include "cpu_ts_thread.h"
#include "aicpu/aicpu_hccl_sqcq.h"
#undef private
#undef protected

using namespace std;
using namespace hccl;

// base_comm 的 HcommThreadCreateAicpu 以真实 hccl::AicpuTsThread* 作为句柄返回，
// 后续 AicpuThreadProcess::ResumeThread 会把该句柄 reinterpret 回 AicpuTsThread* 使用，
// 因此桩必须返回真实对象指针，不能返回伪造常量。
static std::vector<std::shared_ptr<hccl::AicpuTsThread>> g_utStubAicpuThreads;
static HcommResult StubUtHcommThreadCreateAicpu(const char* uniqueId, size_t uniqueIdLen, ThreadHandle* handle)
{
    auto thread = std::make_shared<hccl::AicpuTsThread>(std::string(uniqueId, uniqueIdLen));
    HcclResult ret = thread->Init();
    if (ret != HCCL_SUCCESS) {
        return static_cast<HcommResult>(ret);
    }
    *handle = reinterpret_cast<ThreadHandle>(thread.get());
    g_utStubAicpuThreads.push_back(std::move(thread));
    return HCOMM_SUCCESS;
}

static HcommResult StubUtHcommThreadFillEntity(ThreadHandle handle, HcommThreadEntity* entity)
{
    return static_cast<HcommResult>(reinterpret_cast<hccl::AicpuTsThread*>(handle)->FillHcommThreadEntity(*entity));
}

static HcommResult StubUtHcommThreadGetStreamLite(ThreadHandle handle, void** streamLite)
{
    (void)handle;
    *streamLite = reinterpret_cast<void*>(0x1);
    return HCOMM_SUCCESS;
}

// 与设备侧边界接口一致，按线程实际标志区分真实流和 GE 假流。
constexpr size_t uniqueIdMaxLen = 256;

static HcommResult StubUtHcommThreadIsFakeDeviceRes(ThreadHandle handle, bool* isFake)
{
    *isFake = reinterpret_cast<hccl::AicpuTsThread*>(handle)->IsFakeDeviceRes();
    return HCOMM_SUCCESS;
}

class Test_Engine_Aicpu_Interface : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "Test_Engine_Aicpu_Interface SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "Test_Engine_Aicpu_Interface TearDown" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp() { std::cout << "A Test SetUP" << std::endl; }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test TearDown" << std::endl;
    }
};

static drvError_t SetThreadLocalDeviceId(uint32_t, uint32_t* localDevId)
{
    *localDevId = 7;
    return DRV_ERROR_NONE;
}

constexpr uint64_t THREAD_TEST_SQ_BASE_ADDR = 0x123456789ABCULL;
constexpr uint64_t THREAD_TEST_LOCAL_SQ_ADDR = 0x987654321000ULL;

static std::unique_ptr<Hccl::RtsqBase> MakeInitializedThreadRtsq()
{
    // 公共 UT 桩的 RTSQ 构造函数为空，显式提供资源恢复后的状态。
    auto rtsq = std::make_unique<Hccl::RtsqBase>(0, 0, 0);
    rtsq->sqDepth_ = 2048;
    rtsq->sqBaseAddr_ = THREAD_TEST_SQ_BASE_ADDR;
    rtsq->sqHead_ = 13;
    rtsq->sqTail_ = 17;
    rtsq->taskId_ = 73;
    rtsq->taskIdEnd_ = 1024;
    return rtsq;
}

TEST_F(Test_Engine_Aicpu_Interface, test_RunAicpuThreadSupplementNotify)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    AicpuTsThread aicpuThread(StreamType::STREAM_TYPE_DEVICE, 2, NotifyLoadType::DEVICE_NOTIFY);
    HcclResult ret = aicpuThread.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::string mainStr = aicpuThread.GetUniqueId();

    ret = aicpuThread.SupplementNotify(4);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::string supplementStr = aicpuThread.GetUniqueId();
    GlobalMockObject::verify();

    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    isDeviceSide = true;
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(drvGetLocalDevIDByHostDevID).stubs().will(invoke(SetThreadLocalDeviceId));
    auto restoredRtsq = MakeInitializedThreadRtsq();
    MOCKER_CPP(&Hccl::StreamLite::GetRtsq).stubs().will(returnValue(restoredRtsq.get()));

    CommAicpuParam aicpuPara{};
    std::string hcomId = "abcd";
    EXPECT_EQ(memcpy_s(aicpuPara.hcomId, sizeof(aicpuPara.hcomId), hcomId.c_str(), hcomId.size()), 0);
    aicpuPara.deviceLogicId = 0;
    aicpuPara.devicePhyId = 0;
    aicpuPara.deviceType = (u32)DevType::DEV_TYPE_950;
    MOCKER_CPP(&CollCommAicpu::InitAicpuIndOp).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&ThreadAicpuMgr::RegisterThreadAddDfxTaskInfo).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(HcommThreadRegisterCacheCallback).stubs().will(returnValue(static_cast<HcommResult>(HCOMM_SUCCESS)));
    MOCKER(HcommThreadCreateAicpu).stubs().will(invoke(StubUtHcommThreadCreateAicpu));
    MOCKER(HcommThreadFillEntity).stubs().will(invoke(StubUtHcommThreadFillEntity));
    MOCKER(HcommThreadGetStreamLite).stubs().will(invoke(StubUtHcommThreadGetStreamLite));
    MOCKER(HcommThreadIsFakeDeviceRes).stubs().will(invoke(StubUtHcommThreadIsFakeDeviceRes));
    MOCKER(HcommThreadDestroy).stubs().will(returnValue(static_cast<HcommResult>(HCOMM_SUCCESS)));
    void* args = (void*)(&aicpuPara);
    EXPECT_EQ(RunAicpuCommInit(args), 0);

    // InitAicpuIndOp 被 mock，手动初始化 commEngineResMgr_
    std::vector<std::pair<std::string, CollCommAicpu*>> commInfo;
    CollCommAicpuMgr::GetInstance().GetAllComms(commInfo);
    for (auto& kv : commInfo) {
        if (kv.second->commEngineResMgr_ == nullptr) {
            kv.second->commEngineResMgr_ = std::make_unique<CommEngineResAicpuMgr>(kv.second->dfx_, [](bool) {
                return HCCL_SUCCESS;
            });
        }
    }

    ThreadMgrAicpuParam para{};
    para.threadNum = 1;
    HcommThreadEntity ctx{};
    ctx.abiHeader = {HCOMM_THREAD_ENTITY_VERSION, HCOMM_THREAD_ENTITY_MAGIC_WORD, sizeof(HcommThreadEntity), 0};
    RegedNotifyEntity notifyEntities[6]{};
    HcommThreadSqContext rtsq{};
    ctx.localNotifyAddr = notifyEntities;
    ctx.localNotifyNum = 2;
    ctx.sqContextAddr = &rtsq;
    HcommThreadEntity* deviceHandle[]{&ctx};
    para.deviceHandle = deviceHandle;
    EXPECT_EQ(memcpy_s(para.hcomId, sizeof(para.hcomId), hcomId.c_str(), hcomId.size()), 0);

    // 动态布局：entry 表 + 变长 uid 缓冲由调用方构造（UT 在 host 侧直接运行 device 代码，地址取 host 内存）
    ThreadParamEntry entry{};
    entry.uidOffset = 0;
    entry.uidLen = static_cast<uint32_t>(mainStr.size());
    std::vector<char> uidBuf(mainStr.begin(), mainStr.end());
    para.entryTableAddr = reinterpret_cast<uint64_t>(&entry);
    para.paramBufAddr = reinterpret_cast<uint64_t>(uidBuf.data());
    para.paramBufSize = uidBuf.size();

    u64 addr = (u64)(&para);
    args = (void*)(&addr);
    EXPECT_EQ(RunAicpuIndOpThreadInit(args), 0);
    ASSERT_NE(ctx.compatThreadHandle, 0U);
    auto* deviceThread = reinterpret_cast<AicpuTsThread*>(ctx.compatThreadHandle);
    ASSERT_EQ(deviceThread->GetNotifyNum(), 2);
    ASSERT_EQ(ctx.localNotifyNum, 2U);
    EXPECT_EQ(notifyEntities[0].type, REGED_NOTIFY_IPC_RT);
    EXPECT_EQ(notifyEntities[0].notifyInfo.ipcRt.size, 0U);
    EXPECT_EQ(notifyEntities[0].notifyInfo.ipcRt.addr, 0U);
    EXPECT_EQ(notifyEntities[0].notifyInfo.ipcRt.notifyId, deviceThread->GetNotify(0)->notifyId_);
    auto* streamLite = static_cast<Hccl::StreamLite*>(deviceThread->GetStreamLitePtr());
    ASSERT_NE(streamLite, nullptr);
    EXPECT_EQ(rtsq.contextInfo.RtsqContext.streamId, streamLite->GetId());
    EXPECT_EQ(rtsq.contextInfo.RtsqContext.sqId, streamLite->GetSqId());
    EXPECT_EQ(rtsq.type, 0U);
    EXPECT_EQ(rtsq.contextInfo.RtsqContext.sqBaseAddr, THREAD_TEST_SQ_BASE_ADDR);
    EXPECT_EQ(rtsq.contextInfo.RtsqContext.sqDepth, 2048U);
    EXPECT_EQ(rtsq.contextInfo.RtsqContext.sqHead, 13U);
    EXPECT_EQ(rtsq.contextInfo.RtsqContext.sqTail, 17U);
    EXPECT_EQ(rtsq.contextInfo.RtsqContext.taskId, 0U);
    EXPECT_EQ(rtsq.contextInfo.RtsqContext.taskIdEnd, 0U);
    EXPECT_EQ(rtsq.contextInfo.RtsqContext.pendingSqeCnt, 0U);
    rtsq.contextInfo.RtsqContext.sqHead = 19;
    rtsq.contextInfo.RtsqContext.sqTail = 21;
    rtsq.contextInfo.RtsqContext.taskId = 23;
    rtsq.contextInfo.RtsqContext.taskIdEnd = 1024;
    rtsq.contextInfo.RtsqContext.pendingSqeCnt = 2;
    const HcommThreadSqContext beforeSupplement = rtsq;
    entry.uidLen = static_cast<uint32_t>(supplementStr.size());
    uidBuf.assign(supplementStr.begin(), supplementStr.end());
    // assign 可能触发 vector 重分配，必须在赋值后重新取缓冲地址
    para.paramBufAddr = reinterpret_cast<uint64_t>(uidBuf.data());
    para.paramBufSize = uidBuf.size();
    ctx.localNotifyNum = 6; // 模拟 Host 完成扩容后发布新的数组元素数。
    EXPECT_EQ(RunAicpuThreadSupplementNotify(args), 0);
    EXPECT_EQ(ctx.localNotifyNum, 6U);
    EXPECT_EQ(notifyEntities[5].type, REGED_NOTIFY_IPC_RT);
    EXPECT_EQ(notifyEntities[5].notifyInfo.ipcRt.notifyId, deviceThread->GetNotify(5)->notifyId_);
    EXPECT_EQ(notifyEntities[5].notifyInfo.ipcRt.addr, 0U);
    EXPECT_EQ(notifyEntities[5].notifyInfo.ipcRt.size, 0U);
    EXPECT_EQ(memcmp(&rtsq, &beforeSupplement, sizeof(rtsq)), 0);
    EXPECT_EQ(deviceThread->GetNotifyNum(), 6);

    para.deviceHandle = nullptr;
    EXPECT_EQ(RunAicpuThreadSupplementNotify(args), HCCL_E_PTR);
    EXPECT_EQ(deviceThread->GetNotifyNum(), 6);

    para.deviceHandle = deviceHandle;
    deviceHandle[0] = nullptr;
    EXPECT_EQ(RunAicpuThreadSupplementNotify(args), HCCL_E_PTR);
    deviceHandle[0] = &ctx;

    MOCKER(CollCommAicpuKernelAdptInitNotify).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AicpuThreadProcess::AicpuThreadInit).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AicpuThreadProcess::AicpuThreadDestroy).stubs().will(returnValue(HCCL_SUCCESS));
    EXPECT_EQ(RunAicpuIndOpNotify(args), 0);
    EXPECT_EQ(RunAicpuThreadInit(args), 0);
    EXPECT_EQ(RunAicpuThreadDestroy(args), 0);
}

static HcclResult ResumeTestThreads(ThreadMgrAicpuParam* param, std::vector<std::shared_ptr<Thread>>& outThreads, bool)
{
    for (u32 i = 0; i < param->threadNum; ++i) {
        outThreads.emplace_back(std::make_shared<AicpuTsThread>(std::string("thread_ctx_ut")));
    }
    return HCCL_SUCCESS;
}

TEST_F(Test_Engine_Aicpu_Interface, BaseThreadInitFillsPodAfterDeviceRestore)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(bool{false})).will(returnValue(HCCL_SUCCESS));
    AicpuTsThread host(StreamType::STREAM_TYPE_DEVICE, 1, NotifyLoadType::HOST_NOTIFY);
    ASSERT_EQ(host.Init(), HCCL_SUCCESS);
    std::string uniqueId = host.GetUniqueId();
    HcclSignalInfo info{};
    ASSERT_EQ(host.GetNotify(0)->GetNotifyData(info), HCCL_SUCCESS);
    GlobalMockObject::verify();
    MOCKER(QuerySqBaseAddr).expects(never());
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(bool{true})).will(returnValue(HCCL_SUCCESS));
    MOCKER(drvGetLocalDevIDByHostDevID).stubs().will(invoke(SetThreadLocalDeviceId));
    auto restoredRtsq = MakeInitializedThreadRtsq();
    MOCKER_CPP(&Hccl::StreamLite::GetRtsq).stubs().will(returnValue(restoredRtsq.get()));
    MOCKER(HcommThreadRegisterDfx).stubs().will(returnValue(0));
    MOCKER_CPP(&AicpuThreadProcess::StopBackGroundThread).stubs();
    HcommThreadEntity entity{};
    entity.abiHeader = {HCOMM_THREAD_ENTITY_VERSION, HCOMM_THREAD_ENTITY_MAGIC_WORD, sizeof(HcommThreadEntity), 0};
    RegedNotifyEntity notify{};
    HcommThreadSqContext sq{};
    sq.contextInfo.RtsqContext.localSqAddr = THREAD_TEST_LOCAL_SQ_ADDR;
    sq.contextInfo.RtsqContext.localSqDepth = 128U;
    entity.localNotifyAddr = &notify;
    entity.localNotifyNum = 1;
    entity.sqContextAddr = &sq;
    HcommThreadEntity* entities[]{&entity};
    ThreadMgrAicpuParam param{};
    param.threadNum = 1;
    param.engine = COMM_ENGINE_AICPU_TS;
    param.deviceHandle = entities;
    ThreadParamEntry entry{};
    std::vector<char> uidBuf(uniqueId.begin(), uniqueId.end());
    entry.uidOffset = 0;
    entry.uidLen = static_cast<uint32_t>(uniqueId.size());
    param.entryTableAddr = reinterpret_cast<uint64_t>(&entry);
    param.paramBufAddr = reinterpret_cast<uint64_t>(uidBuf.data());
    param.paramBufSize = uidBuf.size();
    ASSERT_EQ(AicpuThreadProcess::InitThreads(&param), HCCL_SUCCESS);
    ASSERT_NE(entity.compatThreadHandle, 0U);
    EXPECT_EQ(entity.localNotifyNum, 1U);
    EXPECT_EQ(notify.notifyInfo.ipcRt.addr, 0U);
    EXPECT_EQ(notify.notifyInfo.ipcRt.size, 0U);
    EXPECT_EQ(notify.notifyInfo.ipcRt.notifyId, info.resId);
    EXPECT_EQ(sq.contextInfo.RtsqContext.streamId, host.GetStream()->id());
    EXPECT_EQ(sq.contextInfo.RtsqContext.sqId, host.GetStream()->sqId());
    EXPECT_EQ(sq.contextInfo.RtsqContext.localSqAddr, THREAD_TEST_LOCAL_SQ_ADDR);
    EXPECT_EQ(sq.contextInfo.RtsqContext.localSqDepth, 128U);
    EXPECT_EQ(sq.contextInfo.RtsqContext.sqDepth, 2048U);
    EXPECT_EQ(sq.contextInfo.RtsqContext.sqHead, 13U);
    EXPECT_EQ(sq.contextInfo.RtsqContext.sqTail, 17U);
    EXPECT_EQ(sq.contextInfo.RtsqContext.sqBaseAddr, THREAD_TEST_SQ_BASE_ADDR);
    EXPECT_EQ(sq.contextInfo.RtsqContext.taskId, 0U);
    EXPECT_EQ(sq.contextInfo.RtsqContext.taskIdEnd, 0U);
    EXPECT_EQ(sq.contextInfo.RtsqContext.pendingSqeCnt, 0U);
    EXPECT_EQ(AicpuThreadProcess::AicpuThreadDestroy(&param), HCCL_SUCCESS);
}

TEST_F(Test_Engine_Aicpu_Interface, A3HcommThreadEntityDoesNotQueryOrFillRtsq)
{
    AicpuTsThread thread(std::string("unused_serialization"));
    thread.devType_ = DevType::DEV_TYPE_910_93;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_910_93)).will(returnValue(HCCL_SUCCESS));
    MOCKER(QuerySqBaseAddr).expects(never());
    HcommThreadEntity entity{};
    EXPECT_EQ(thread.FillHcommThreadEntity(entity), HCCL_SUCCESS);
    EXPECT_EQ(entity.sqContextAddr, nullptr);
    EXPECT_EQ(entity.localNotifyAddr, nullptr);
}

TEST_F(Test_Engine_Aicpu_Interface, CollectiveMixedThreadsKeepEntityOrderAndNotifyOnlyExports)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(bool{false})).will(returnValue(HCCL_SUCCESS));
    constexpr u32 threadNum = 3;
    std::vector<std::shared_ptr<Thread>> hosts;
    HcommThreadEntity* entities[threadNum]{};
    HcclSignalInfo notifyInfo[threadNum]{};
    ThreadMgrAicpuParam param{};
    param.threadNum = threadNum;
    param.deviceHandle = entities;
    std::vector<ThreadParamEntry> entries(threadNum);
    std::vector<char> uidBuf;
    uidBuf.reserve(threadNum * uniqueIdMaxLen);
    param.entryTableAddr = reinterpret_cast<uint64_t>(entries.data());
    for (u32 i = 0; i < threadNum; ++i) {
        std::shared_ptr<Thread> host;
        if (i == 1) {
            host = std::make_shared<AicpuTsThread>(StreamType::STREAM_TYPE_DEVICE, 1, NotifyLoadType::HOST_NOTIFY);
        } else {
            host = std::make_shared<CpuTsThread>(StreamType::STREAM_TYPE_ONLINE, 1, NotifyLoadType::HOST_NOTIFY);
            host->SetFakeDeviceRes();
        }
        ASSERT_EQ(host->Init(), HCCL_SUCCESS);
        ThreadHandle handle{};
        ASSERT_EQ(host->PreAllocThreadEntity(handle, COMM_ENGINE_AICPU), HCCL_SUCCESS);
        entities[i] = reinterpret_cast<HcommThreadEntity*>(handle);
        const std::string uniqueId = host->GetUniqueId();
        entries[i].uidOffset = static_cast<uint32_t>(uidBuf.size());
        entries[i].uidLen = static_cast<uint32_t>(uniqueId.size());
        uidBuf.insert(uidBuf.end(), uniqueId.begin(), uniqueId.end());
        ASSERT_EQ(host->GetNotify(0)->GetNotifyData(notifyInfo[i]), HCCL_SUCCESS);
        hosts.emplace_back(std::move(host));
    }
    param.paramBufAddr = reinterpret_cast<uint64_t>(uidBuf.data());
    param.paramBufSize = uidBuf.size();
    GlobalMockObject::verify();

    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(bool{true})).will(returnValue(HCCL_SUCCESS));
    MOCKER(drvGetLocalDevIDByHostDevID).stubs().will(invoke(SetThreadLocalDeviceId));
    auto restoredRtsq = MakeInitializedThreadRtsq();
    MOCKER_CPP(&Hccl::StreamLite::GetRtsq).stubs().will(returnValue(restoredRtsq.get()));
    // 只有中间的真实线程注册流相关回调；首尾的 Notify 桩线程不能进入轮询列表。
    MOCKER_CPP(&ThreadAicpuMgr::RegisterThreadAddDfxTaskInfo).expects(once()).will(returnValue(HCCL_SUCCESS));
    MOCKER(HcommThreadRegisterCacheCallback).expects(once()).will(returnValue(static_cast<HcommResult>(HCOMM_SUCCESS)));
    MOCKER(HcommThreadCreateAicpu).stubs().will(invoke(StubUtHcommThreadCreateAicpu));
    MOCKER(HcommThreadFillEntity).stubs().will(invoke(StubUtHcommThreadFillEntity));
    MOCKER(HcommThreadIsFakeDeviceRes).stubs().will(invoke(StubUtHcommThreadIsFakeDeviceRes));
    MOCKER(HcommThreadDestroy).stubs().will(returnValue(static_cast<HcommResult>(HCOMM_SUCCESS)));
    MOCKER(HcommThreadRegisterCheckExecStatus).stubs().will(returnValue(0));
    HcclCommDfxLite dfx;
    ThreadAicpuMgr mgr(dfx, [](bool) {
        return HCCL_SUCCESS;
    });
    ASSERT_EQ(mgr.InitThreads(&param), HCCL_SUCCESS);
    ASSERT_EQ(mgr.GetAllThread().size(), 1U);
    ASSERT_EQ(mgr.cpuExportThread_.size(), 2U);
    EXPECT_EQ(entities[0]->compatThreadHandle, mgr.cpuExportThread_[0]);
    EXPECT_EQ(entities[1]->compatThreadHandle, mgr.GetAllThread()[0]);
    EXPECT_EQ(entities[2]->compatThreadHandle, mgr.cpuExportThread_[1]);
    for (u32 i = 0; i < threadNum; ++i) {
        ThreadHandle raw = reinterpret_cast<ThreadHandle>(entities[i]);
        ASSERT_EQ(UnwrapThreadHandle(raw), HCCL_SUCCESS);
        auto* thread = reinterpret_cast<AicpuTsThread*>(raw);
        EXPECT_EQ(thread->IsFakeDeviceRes(), i != 1);
        EXPECT_EQ(entities[i]->localNotifyNum, 1U);
        EXPECT_EQ(entities[i]->localNotifyAddr[0].type, REGED_NOTIFY_IPC_RT);
        EXPECT_EQ(entities[i]->localNotifyAddr[0].notifyInfo.ipcRt.notifyId, notifyInfo[i].resId);
        if (i == 1) {
            ASSERT_NE(entities[i]->sqContextAddr, nullptr);
            EXPECT_EQ(entities[i]->sqContextAddr->contextInfo.RtsqContext.sqBaseAddr, THREAD_TEST_SQ_BASE_ADDR);
        } else {
            EXPECT_EQ(entities[i]->sqContextAddr, nullptr);
            EXPECT_EQ(thread->GetStreamLitePtr(), nullptr);
        }
    }
}

// GetThreadParamEntry/GetThreadUid 越界防御：idx 越界、空 uid、uidOffset+uidLen 越过 paramBufSize 均返回 nullptr
TEST_F(Test_Engine_Aicpu_Interface, test_GetThreadParamEntry_And_GetThreadUid_When_InvalidLayout_Expect_Defense)
{
    std::vector<char> uidBuf = {'a', 'b', 'c'};
    ThreadParamEntry entry{};
    ThreadMgrAicpuParam para{};
    para.threadNum = 1;
    para.entryTableAddr = reinterpret_cast<uint64_t>(&entry);
    para.paramBufAddr = reinterpret_cast<uint64_t>(uidBuf.data());
    para.paramBufSize = uidBuf.size();

    // idx 越界 / 参数为空
    EXPECT_EQ(GetThreadParamEntry(&para, para.threadNum), nullptr);
    EXPECT_EQ(GetThreadParamEntry(nullptr, 0), nullptr);
    EXPECT_NE(GetThreadParamEntry(&para, 0), nullptr);

    uint32_t uidLen = 0;
    // 空 uid：uidLen == 0
    entry.uidOffset = 0;
    entry.uidLen = 0;
    EXPECT_EQ(GetThreadUid(&para, 0, uidLen), nullptr);

    // uidOffset + uidLen 越过 paramBufSize（缓冲仅 3 字节）
    entry.uidLen = 4;
    EXPECT_EQ(GetThreadUid(&para, 0, uidLen), nullptr);
    entry.uidOffset = 1;
    entry.uidLen = 3;
    EXPECT_EQ(GetThreadUid(&para, 0, uidLen), nullptr);

    // 合法布局不受影响
    entry.uidOffset = 0;
    entry.uidLen = 3;
    const char* uidData = GetThreadUid(&para, 0, uidLen);
    EXPECT_NE(uidData, nullptr);
    EXPECT_EQ(uidLen, 3U);

    // entry 表地址为空：GetThreadParamEntry/GetThreadUid 均返回 nullptr
    uint64_t entryTableBak = para.entryTableAddr;
    para.entryTableAddr = 0;
    EXPECT_EQ(GetThreadParamEntry(&para, 0), nullptr);
    EXPECT_EQ(GetThreadUid(&para, 0, uidLen), nullptr);
    para.entryTableAddr = entryTableBak;

    // uid 缓冲地址为空：GetThreadUid 返回 nullptr
    uint64_t paramBufBak = para.paramBufAddr;
    para.paramBufAddr = 0;
    EXPECT_EQ(GetThreadUid(&para, 0, uidLen), nullptr);
    para.paramBufAddr = paramBufBak;
}

// device 侧入口：uid 布局非法时 RunAicpuIndOpThreadInit 应返回 HCCL_E_PARA 而非越界访问
TEST_F(Test_Engine_Aicpu_Interface, test_RunAicpuIndOpThreadInit_When_UidLayoutInvalid_Expect_HCCL_E_PARA)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{true};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    CommAicpuParam aicpuPara{};
    std::string hcomId = "erruid";
    EXPECT_EQ(memcpy_s(aicpuPara.hcomId, sizeof(aicpuPara.hcomId), hcomId.c_str(), hcomId.size()), 0);
    aicpuPara.deviceLogicId = 0;
    aicpuPara.devicePhyId = 0;
    aicpuPara.deviceType = (u32)DevType::DEV_TYPE_950;
    MOCKER_CPP(&CollCommAicpu::InitAicpuIndOp).stubs().will(returnValue(HCCL_SUCCESS));
    void* args = (void*)(&aicpuPara);
    EXPECT_EQ(RunAicpuCommInit(args), 0);

    // InitAicpuIndOp 被 mock，手动初始化 commEngineResMgr_
    std::vector<std::pair<std::string, CollCommAicpu*>> commInfo;
    CollCommAicpuMgr::GetInstance().GetAllComms(commInfo);
    for (auto& kv : commInfo) {
        if (kv.second->commEngineResMgr_ == nullptr) {
            kv.second->commEngineResMgr_ = std::make_unique<CommEngineResAicpuMgr>(kv.second->dfx_, [](bool) {
                return HCCL_SUCCESS;
            });
        }
    }

    ThreadMgrAicpuParam para{};
    para.threadNum = 1;
    para.deviceHandle = malloc(sizeof(ThreadHandle));
    EXPECT_EQ(memcpy_s(para.hcomId, sizeof(para.hcomId), hcomId.c_str(), hcomId.size()), 0);
    std::vector<char> uidBuf = {'a', 'b', 'c'};
    ThreadParamEntry entry{};
    para.entryTableAddr = reinterpret_cast<uint64_t>(&entry);
    para.paramBufAddr = reinterpret_cast<uint64_t>(uidBuf.data());
    para.paramBufSize = uidBuf.size();

    u64 addr = (u64)(&para);
    args = (void*)(&addr);

    // 空 uid：uidLen == 0，device 侧解析失败返回 HCCL_E_PARA
    entry.uidOffset = 0;
    entry.uidLen = 0;
    EXPECT_EQ(RunAicpuIndOpThreadInit(args), (uint32_t)HCCL_E_PARA);

    // uidOffset + uidLen 越过 paramBufSize，device 侧解析失败返回 HCCL_E_PARA
    entry.uidOffset = 2;
    entry.uidLen = 2;
    EXPECT_EQ(RunAicpuIndOpThreadInit(args), (uint32_t)HCCL_E_PARA);

    // 清理：与首个用例保持一致，销毁线程并释放句柄，避免全局单例残留干扰其它用例
    MOCKER_CPP(&AicpuThreadProcess::AicpuThreadDestroy).stubs().will(returnValue(HCCL_SUCCESS));
    EXPECT_EQ(RunAicpuThreadDestroy(args), 0);
    free(para.deviceHandle);
}
