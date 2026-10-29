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

#ifndef private
#define private public
#define protected public
#endif
#include "stream_utils.h"
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
#undef private
#undef protected

using namespace std;
using namespace hccl;

// base_comm 的 HcommThreadCreateAicpu 以真实 hccl::AicpuTsThread* 作为句柄返回，
// 后续 AicpuThreadProcess::ResumeThread 会把该句柄 reinterpret 回 AicpuTsThread* 使用，
// 因此桩必须返回真实对象指针，不能返回伪造常量。
static std::shared_ptr<hccl::AicpuTsThread> g_utStubAicpuThread;
static HcommResult StubUtHcommThreadCreateAicpu(const char* uniqueId, size_t uniqueIdLen, ThreadHandle* handle)
{
    g_utStubAicpuThread = std::make_shared<hccl::AicpuTsThread>(std::string(uniqueId, uniqueIdLen));
    HcclResult ret = g_utStubAicpuThread->Init();
    if (ret != HCCL_SUCCESS) {
        return static_cast<HcommResult>(ret);
    }
    *handle = reinterpret_cast<ThreadHandle>(g_utStubAicpuThread.get());
    return HCOMM_SUCCESS;
}

static HcommResult StubUtHcommThreadGetStreamLite(ThreadHandle handle, void** streamLite)
{
    (void)handle;
    *streamLite = reinterpret_cast<void*>(0x1);
    return HCOMM_SUCCESS;
}

// 真/桩分类改用显式标志（IsFakeDeviceRes）后，使用例中的线程被判为真线程。
static HcommResult StubUtHcommThreadIsFakeDeviceRes(ThreadHandle handle, bool* isFake)
{
    (void)handle;
    *isFake = false;
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
    para.deviceHandle = malloc(sizeof(ThreadHandle));
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
    entry.uidLen = static_cast<uint32_t>(supplementStr.size());
    uidBuf.assign(supplementStr.begin(), supplementStr.end());
    // assign 可能触发 vector 重分配，必须在赋值后重新取缓冲地址
    para.paramBufAddr = reinterpret_cast<uint64_t>(uidBuf.data());
    para.paramBufSize = uidBuf.size();
    EXPECT_EQ(RunAicpuThreadSupplementNotify(args), 0);

    MOCKER(CollCommAicpuKernelAdptInitNotify).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AicpuThreadProcess::AicpuThreadInit).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AicpuThreadProcess::AicpuThreadDestroy).stubs().will(returnValue(HCCL_SUCCESS));
    EXPECT_EQ(RunAicpuIndOpNotify(args), 0);
    EXPECT_EQ(RunAicpuThreadInit(args), 0);
    EXPECT_EQ(RunAicpuThreadDestroy(args), 0);
    free(para.deviceHandle);
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
