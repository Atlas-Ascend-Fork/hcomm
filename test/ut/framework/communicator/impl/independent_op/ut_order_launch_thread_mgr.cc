/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "hcomm/hcomm_res_expt.h"
#include <mockcpp/mockcpp.hpp>
#include "../../hccl_api_base_test.h"
#include "order_launch_thread_mgr.h"
#include "hcomm_c_adpt.h"
#include "comm_engine/thread/thread_pub.h"
#include "adapter_rts_common.h"

using namespace hccl;

static u64 g_mockCtx = 0;
static ThreadHandle g_mockThreadHandle = 0;
static int64_t g_mockCoreNum = 0;
static HcommResult g_mockAllocRet = HCCL_SUCCESS;
static HcommResult g_mockAllocWithStreamRet = HCCL_SUCCESS;
static ThreadHandle g_nextMockHandle = 0x1000;

static aclError StubAclrtGetCurrentContext(aclrtContext* ctx)
{
    *ctx = reinterpret_cast<aclrtContext>(g_mockCtx);
    return ACL_SUCCESS;
}

static HcommResult
StubHcommThreadAlloc(CommEngine engine, uint32_t threadNum, const uint32_t* notifyNum, ThreadHandle* handle)
{
    if (g_mockAllocRet == HCCL_SUCCESS) {
        *handle = g_nextMockHandle++;
    }
    return g_mockAllocRet;
}

static HcommResult
StubHcommThreadAllocWithStream(CommEngine engine, void* stream, uint32_t notifyNum, ThreadHandle* handle)
{
    if (g_mockAllocWithStreamRet == HCCL_SUCCESS) {
        *handle = g_mockThreadHandle;
    }
    return g_mockAllocWithStreamRet;
}

static HcommResult g_mockAcquireByNotifyRet = HCCL_SUCCESS;

static HcommResult
StubHcommThreadAcquireByNotify(void* stream, void** notifys, uint32_t notifyNum, ThreadHandle* handle)
{
    (void)stream;
    (void)notifys;
    (void)notifyNum;
    if (g_mockAcquireByNotifyRet == HCCL_SUCCESS) {
        *handle = g_nextMockHandle++;
    }
    return g_mockAcquireByNotifyRet;
}

static aclError StubAclrtGetDeviceInfo(uint32_t deviceId, aclrtDevAttr attr, int64_t* val)
{
    *val = g_mockCoreNum;
    return ACL_SUCCESS;
}

static HcclResult StubHrtGetStreamId(void* stream, s32& streamId)
{
    // UT用假流指针（如0x1234），SetAttachedStream日志中的流ID真实查询会解引用假指针导致段错误，
    // 此处mock为固定值；返回值仅用于日志，不影响主流程
    (void)stream;
    streamId = 1;
    return HCCL_SUCCESS;
}

class OrderLaunchThreadMgrTest : public BaseInit {
public:
    void SetUp() override
    {
        BaseInit::SetUp();
        mgr_ = std::make_unique<OrderLaunchThreadMgr>();
        g_nextMockHandle = 0x1000;
        // SetAttachedStream日志以流ID真实查询（hrtGetStreamId→aclrtStreamGetId会解引用句柄读
        // magic校验），UT的假流指针未映射→段错误；fixture级mock保证全部用例安全
        MOCKER(hrtGetStreamId).stubs().will(invoke(StubHrtGetStreamId));
    }
    void TearDown() override
    {
        mgr_.reset();
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }

    void MockGetCurrentContext(u64 ctx)
    {
        g_mockCtx = ctx;
        MOCKER(aclrtGetCurrentContext).stubs().will(invoke(StubAclrtGetCurrentContext));
    }

    void MockHcommThreadAlloc(HcommResult ret)
    {
        g_mockThreadHandle = static_cast<ThreadHandle>(0x1000);
        g_mockAllocRet = ret;
        HcommResult (*allocFunc)(CommEngine, uint32_t, const uint32_t*, ThreadHandle*) = HcommThreadAlloc;
        MOCKER(allocFunc).stubs().will(invoke(StubHcommThreadAlloc));
    }

    void MockHcommThreadAllocWithStream(HcommResult ret)
    {
        g_mockThreadHandle = static_cast<ThreadHandle>(0x2000);
        g_mockAllocWithStreamRet = ret;
        MOCKER(HcommThreadAllocWithStream).stubs().will(invoke(StubHcommThreadAllocWithStream));
    }

    void MockHcommThreadAcquireByNotify(HcommResult ret)
    {
        g_mockAcquireByNotifyRet = ret;
        MOCKER(HcommThreadAcquireByNotify).stubs().will(invoke(StubHcommThreadAcquireByNotify));
        // 配套mock释放：换图/换流失效路径会对旧句柄调HcommThreadFree，mock句柄不在真实g_ThreadMap中
        MockHcommThreadFree(HCCL_SUCCESS);
    }

    void MockHcommThreadFree(HcommResult ret) { MOCKER(HcommThreadFree).stubs().will(returnValue(ret)); }

    void MockAclrtGetDeviceInfo(u32 blockNum)
    {
        g_mockCoreNum = static_cast<int64_t>(blockNum);
        MOCKER(aclrtGetDeviceInfo).stubs().will(invoke(StubAclrtGetDeviceInfo));
    }

    void SetupGeEnvironment()
    {
        MockGetCurrentContext(0x100);
        MockHcommThreadAllocWithStream(HCCL_SUCCESS);
        MockHcommThreadAcquireByNotify(HCCL_SUCCESS);
        MockHcommThreadFree(HCCL_SUCCESS);
        MockAclrtGetDeviceInfo(1);
        mgr_->RegisterOrderLaunch("group1");
        mgr_->RegisterOrderLaunch("group2");

        ThreadHandle warmup = 0;
        mgr_->EnsureOrderThread(OrderThreadMode::OPBASE, "warmup", 1, warmup);

        void* fakeStream = reinterpret_cast<void*>(0x1234);
        mgr_->SetAttachedStream("group1", 100, fakeStream);
    }

    std::unique_ptr<OrderLaunchThreadMgr> mgr_;
};

/* ======================== RegisterOrderLaunch ======================== */

TEST_F(OrderLaunchThreadMgrTest, Ut_RegisterOrderLaunch_When_NewGroup_Expect_Success)
{
    HcclResult ret = mgr_->RegisterOrderLaunch("group1");
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OrderLaunchThreadMgrTest, Ut_RegisterOrderLaunch_When_DuplicateGroup_Expect_Success)
{
    mgr_->RegisterOrderLaunch("group1");
    HcclResult ret = mgr_->RegisterOrderLaunch("group1");
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OrderLaunchThreadMgrTest, Ut_RegisterOrderLaunch_When_MultiGroup_Expect_Success)
{
    EXPECT_EQ(mgr_->RegisterOrderLaunch("group1"), HCCL_SUCCESS);
    EXPECT_EQ(mgr_->RegisterOrderLaunch("group2"), HCCL_SUCCESS);
    EXPECT_EQ(mgr_->RegisterOrderLaunch("group3"), HCCL_SUCCESS);
}

/* ======================== UnRegisterOrderLaunch ======================== */

TEST_F(OrderLaunchThreadMgrTest, Ut_UnRegisterOrderLaunch_When_RegisteredGroup_Expect_Success)
{
    mgr_->RegisterOrderLaunch("group1");
    HcclResult ret = mgr_->UnRegisterOrderLaunch("group1");
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OrderLaunchThreadMgrTest, Ut_UnRegisterOrderLaunch_When_UnregisteredGroup_Expect_Success)
{
    HcclResult ret = mgr_->UnRegisterOrderLaunch("group_not_exist");
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OrderLaunchThreadMgrTest, Ut_UnRegisterOrderLaunch_After_Destroy_Expect_Success)
{
    mgr_.reset();
    mgr_ = std::make_unique<OrderLaunchThreadMgr>();
    HcclResult ret = mgr_->UnRegisterOrderLaunch("group1");
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

/* ======================== EnsureContextRes (via EnsureOrderThread) ======================== */

TEST_F(OrderLaunchThreadMgrTest, Ut_EnsureContextRes_When_NewContext_Expect_Success)
{
    MockGetCurrentContext(0x100);
    MockHcommThreadAlloc(HCCL_SUCCESS);
    MockAclrtGetDeviceInfo(0);

    ThreadHandle thread = 0;
    HcclResult ret = mgr_->EnsureOrderThread(OrderThreadMode::OPBASE, "group1", 1, thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

/* ======================== EnsureOrderThread ======================== */

TEST_F(OrderLaunchThreadMgrTest, Ut_EnsureOrderThread_When_HcommThreadAllocFails_Expect_Error)
{
    MockGetCurrentContext(0x100);
    MockHcommThreadAlloc(static_cast<HcommResult>(HCCL_E_RUNTIME));
    MockAclrtGetDeviceInfo(0);

    ThreadHandle warmup = 0;
    mgr_->EnsureOrderThread(OrderThreadMode::OPBASE, "warmup", 1, warmup);

    ThreadHandle thread = 0;
    HcclResult ret = mgr_->EnsureOrderThread(OrderThreadMode::OPBASE, "group1", 1, thread);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

TEST_F(OrderLaunchThreadMgrTest, Ut_EnsureOrderThread_When_GetCurrentContextFails_Expect_Error)
{
    MOCKER(aclrtGetCurrentContext).stubs().will(returnValue(static_cast<aclError>(1)));

    ThreadHandle thread = 0;
    HcclResult ret = mgr_->EnsureOrderThread(OrderThreadMode::OPBASE, "group1", 1, thread);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

TEST_F(OrderLaunchThreadMgrTest, Ut_EnsureOrderThread_When_GroupCountLeBlockNum_Expect_ThreadZero)
{
    MockGetCurrentContext(0x100);
    MockAclrtGetDeviceInfo(10);
    mgr_->RegisterOrderLaunch("group1");

    ThreadHandle thread = 0;
    HcclResult ret = mgr_->EnsureOrderThread(OrderThreadMode::OPBASE, "group1", 1, thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(thread, static_cast<ThreadHandle>(0));
}

TEST_F(OrderLaunchThreadMgrTest, Ut_EnsureOrderThread_When_GroupCountGtBlockNum_Expect_ThreadCreated)
{
    MockGetCurrentContext(0x100);
    MockAclrtGetDeviceInfo(1);
    MockHcommThreadAlloc(HCCL_SUCCESS);
    mgr_->RegisterOrderLaunch("group1");
    mgr_->RegisterOrderLaunch("group2");
    mgr_->RegisterOrderLaunch("group3");

    ThreadHandle warmup = 0;
    mgr_->EnsureOrderThread(OrderThreadMode::OPBASE, "group1", 1, warmup);

    ThreadHandle thread = 0;
    HcclResult ret = mgr_->EnsureOrderThread(OrderThreadMode::OPBASE, "group2", 1, thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(thread, static_cast<ThreadHandle>(0));
}

TEST_F(OrderLaunchThreadMgrTest, Ut_EnsureOrderThread_When_ThreadExists_Expect_Reuse)
{
    MockGetCurrentContext(0x100);
    MockAclrtGetDeviceInfo(1);
    MockHcommThreadAlloc(HCCL_SUCCESS);
    mgr_->RegisterOrderLaunch("group1");
    mgr_->RegisterOrderLaunch("group2");

    ThreadHandle warmup = 0;
    mgr_->EnsureOrderThread(OrderThreadMode::OPBASE, "warmup", 1, warmup);

    ThreadHandle thread1 = 0;
    mgr_->EnsureOrderThread(OrderThreadMode::OPBASE, "group1", 1, thread1);

    ThreadHandle thread2 = 0;
    HcclResult ret = mgr_->EnsureOrderThread(OrderThreadMode::OPBASE, "group2", 1, thread2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(thread1, thread2);
}

TEST_F(OrderLaunchThreadMgrTest, Ut_EnsureOrderThread_When_AclgraphMode_Expect_Success)
{
    MockGetCurrentContext(0x100);
    MockAclrtGetDeviceInfo(1);
    MockHcommThreadAlloc(HCCL_SUCCESS);
    mgr_->RegisterOrderLaunch("group1");
    mgr_->RegisterOrderLaunch("group2");

    ThreadHandle warmup = 0;
    mgr_->EnsureOrderThread(OrderThreadMode::OPBASE, "warmup", 1, warmup);

    ThreadHandle thread = 0;
    HcclResult ret = mgr_->EnsureOrderThread(OrderThreadMode::ACLGRAPH, "group1", 1, thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(thread, static_cast<ThreadHandle>(0));
}

/* ======================== EnsureDeviceOrderThread ======================== */

TEST_F(OrderLaunchThreadMgrTest, Ut_EnsureDeviceOrderThread_When_CollCommNull_Expect_Error)
{
    ThreadHandle thread = 0;
    HcclResult ret = mgr_->EnsureDeviceOrderThread(nullptr, "group1", 1, thread);
    EXPECT_NE(ret, HCCL_SUCCESS);
    EXPECT_EQ(thread, static_cast<ThreadHandle>(0));
}

/* ======================== GetHcomAttachedThreadByGroup ======================== */

TEST_F(OrderLaunchThreadMgrTest, Ut_GetHcomAttachedThreadByGroup_When_NotSet_Expect_Zero)
{
    ThreadHandle ret = mgr_->GetHcomAttachedThreadByGroup("group1");
    EXPECT_EQ(ret, static_cast<ThreadHandle>(0));
}

TEST_F(OrderLaunchThreadMgrTest, Ut_GetHcomAttachedThreadByGroup_When_Set_Expect_Handle)
{
    MockHcommThreadAcquireByNotify(HCCL_SUCCESS);
    void* fakeStream = reinterpret_cast<void*>(0x1234);
    mgr_->SetAttachedStream("group1", 100, fakeStream);

    ThreadHandle handle = mgr_->GetHcomAttachedThreadByGroup("group1");
    EXPECT_NE(handle, static_cast<ThreadHandle>(0));
}

TEST_F(OrderLaunchThreadMgrTest, Ut_UnRegisterOrderLaunch_When_GroupOnGraph_Expect_UnregisteredGroupNoThread)
{
    mgr_->RegisterOrderLaunch("group1");

    void* fakeStream = reinterpret_cast<void*>(0x1234);
    mgr_->SetAttachedStream("group1", 100, fakeStream);

    // 注销后 group->graphId 映射应同步清除：已注销 group 不得再经
    // GetHcomAttachedThreadByGroup 凭残留映射复活（为其重建 notify/thread）
    mgr_->UnRegisterOrderLaunch("group1");

    ThreadHandle handle = mgr_->GetHcomAttachedThreadByGroup("group1");
    EXPECT_EQ(handle, static_cast<ThreadHandle>(0));
}

TEST_F(OrderLaunchThreadMgrTest, Ut_GetHcomAttachedThreadByGroup_When_SameGraphDifferentGroup_Expect_DifferentThreads)
{
    MockHcommThreadAcquireByNotify(HCCL_SUCCESS);
    MockAclrtGetDeviceInfo(1);

    void* fakeStream = reinterpret_cast<void*>(0x1234);
    mgr_->SetAttachedStream("group1", 100, fakeStream);
    mgr_->SetAttachedStream("group2", 100, fakeStream);

    ThreadHandle thread1 = mgr_->GetHcomAttachedThreadByGroup("group1");
    ThreadHandle thread2 = mgr_->GetHcomAttachedThreadByGroup("group2");
    EXPECT_NE(thread1, static_cast<ThreadHandle>(0));
    EXPECT_NE(thread2, static_cast<ThreadHandle>(0));
    EXPECT_NE(thread1, thread2);

    ThreadHandle reuse1 = mgr_->GetHcomAttachedThreadByGroup("group1");
    EXPECT_EQ(reuse1, thread1);
}

TEST_F(OrderLaunchThreadMgrTest, Ut_SetAttachedStream_When_GroupSwitchGraph_Expect_ThreadRebuilt)
{
    MockHcommThreadAcquireByNotify(HCCL_SUCCESS);
    MockAclrtGetDeviceInfo(1);

    void* streamA = reinterpret_cast<void*>(0x1234);
    void* streamB = reinterpret_cast<void*>(0x5678);
    mgr_->SetAttachedStream("group1", 100, streamA);

    ThreadHandle threadOnGraph1 = mgr_->GetHcomAttachedThreadByGroup("group1");
    EXPECT_NE(threadOnGraph1, static_cast<ThreadHandle>(0));

    mgr_->SetAttachedStream("group1", 200, streamB);
    ThreadHandle threadOnGraph2 = mgr_->GetHcomAttachedThreadByGroup("group1");
    EXPECT_NE(threadOnGraph2, static_cast<ThreadHandle>(0));
    EXPECT_NE(threadOnGraph2, threadOnGraph1);
}

TEST_F(OrderLaunchThreadMgrTest, Ut_SetAttachedStream_When_GroupSwitchGraphBackAndForth_Expect_RebuiltEachTime)
{
    MockHcommThreadAcquireByNotify(HCCL_SUCCESS);
    MockAclrtGetDeviceInfo(1);

    void* streamA = reinterpret_cast<void*>(0x1234);
    void* streamB = reinterpret_cast<void*>(0x5678);
    mgr_->SetAttachedStream("groupA", 100, streamA);
    ThreadHandle threadOnA1 = mgr_->GetHcomAttachedThreadByGroup("groupA");
    EXPECT_NE(threadOnA1, static_cast<ThreadHandle>(0));

    mgr_->SetAttachedStream("groupA", 200, streamB);
    ThreadHandle threadOnB = mgr_->GetHcomAttachedThreadByGroup("groupA");
    EXPECT_NE(threadOnB, static_cast<ThreadHandle>(0));
    EXPECT_NE(threadOnB, threadOnA1);

    mgr_->SetAttachedStream("groupA", 100, streamA);
    ThreadHandle threadOnA2 = mgr_->GetHcomAttachedThreadByGroup("groupA");
    EXPECT_NE(threadOnA2, static_cast<ThreadHandle>(0));
    EXPECT_NE(threadOnA2, threadOnB);
    EXPECT_NE(threadOnA2, threadOnA1);
}

TEST_F(OrderLaunchThreadMgrTest, Ut_SetAttachedStream_When_GraphStreamChanged_Expect_ThreadInvalidated)
{
    MockHcommThreadAcquireByNotify(HCCL_SUCCESS);
    MockAclrtGetDeviceInfo(1);

    void* streamA = reinterpret_cast<void*>(0x1234);
    void* streamB = reinterpret_cast<void*>(0x5678);
    mgr_->SetAttachedStream("group1", 100, streamA);
    mgr_->SetAttachedStream("group2", 100, streamA);

    ThreadHandle thread1 = mgr_->GetHcomAttachedThreadByGroup("group1");
    ThreadHandle thread2 = mgr_->GetHcomAttachedThreadByGroup("group2");
    EXPECT_NE(thread1, thread2);

    mgr_->SetAttachedStream("group1", 100, streamB);

    ThreadHandle newThread1 = mgr_->GetHcomAttachedThreadByGroup("group1");
    ThreadHandle newThread2 = mgr_->GetHcomAttachedThreadByGroup("group2");
    EXPECT_NE(newThread1, thread1);
    EXPECT_NE(newThread2, thread2);
}

TEST_F(OrderLaunchThreadMgrTest, Ut_GetHcomAttachedThreadByGroup_When_DifferentGroup_Expect_Zero)
{
    SetupGeEnvironment();
    void* fakeStream = reinterpret_cast<void*>(0x1234);
    mgr_->SetAttachedStream("group2", 200, fakeStream);

    ThreadHandle handle = mgr_->GetHcomAttachedThreadByGroup("group3");
    EXPECT_EQ(handle, static_cast<ThreadHandle>(0));
}

/* ======================== SetAttachedStream ======================== */

TEST_F(OrderLaunchThreadMgrTest, Ut_SetAttachedStream_When_NullStream_Expect_Error)
{
    HcclResult ret = mgr_->SetAttachedStream("group1", 100, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(OrderLaunchThreadMgrTest, Ut_SetAttachedStream_When_SameStream_Expect_Reuse)
{
    void* fakeStream = reinterpret_cast<void*>(0x1234);
    mgr_->SetAttachedStream("group1", 100, fakeStream);

    HcclResult ret = mgr_->SetAttachedStream("group1", 100, fakeStream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OrderLaunchThreadMgrTest, Ut_SetAttachedStream_When_DifferentStream_Expect_Overwrite)
{
    MockHcommThreadAllocWithStream(HCCL_SUCCESS);
    MockHcommThreadFree(HCCL_SUCCESS);

    void* fakeStream1 = reinterpret_cast<void*>(0x1234);
    mgr_->SetAttachedStream("group1", 100, fakeStream1);

    void* fakeStream2 = reinterpret_cast<void*>(0x5678);
    HcclResult ret = mgr_->SetAttachedStream("group1", 100, fakeStream2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

/* ======================== OrderLaunchThreadAcquire ======================== */

TEST_F(OrderLaunchThreadMgrTest, Ut_OrderLaunchThreadAcquire_When_InvalidUseType_Expect_Error)
{
    ThreadHandle thread = 0;
    HcclResult ret
        = mgr_->OrderLaunchThreadAcquire(static_cast<HcclDedicatedThreadType>(99), nullptr, "group1", 1, thread);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(OrderLaunchThreadMgrTest, Ut_OrderLaunchThreadAcquire_When_Opbase_Expect_Success)
{
    MockGetCurrentContext(0x100);
    MockAclrtGetDeviceInfo(0);
    MockHcommThreadAlloc(HCCL_SUCCESS);

    ThreadHandle warmup = 0;
    mgr_->EnsureOrderThread(OrderThreadMode::OPBASE, "warmup", 1, warmup);

    ThreadHandle thread = 0;
    HcclResult ret
        = mgr_->OrderLaunchThreadAcquire(HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_OPBASE, nullptr, "group1", 1, thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(thread, static_cast<ThreadHandle>(0));
}

TEST_F(OrderLaunchThreadMgrTest, Ut_OrderLaunchThreadAcquire_When_Aclgraph_Expect_Success)
{
    MockGetCurrentContext(0x100);
    MockAclrtGetDeviceInfo(0);
    MockHcommThreadAlloc(HCCL_SUCCESS);

    ThreadHandle warmup = 0;
    mgr_->EnsureOrderThread(OrderThreadMode::OPBASE, "warmup", 1, warmup);

    ThreadHandle thread = 0;
    HcclResult ret = mgr_->OrderLaunchThreadAcquire(
        HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_ACLGRAPH, nullptr, "group1", 1, thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(thread, static_cast<ThreadHandle>(0));
}

TEST_F(OrderLaunchThreadMgrTest, Ut_OrderLaunchThreadAcquire_When_Ge_Expect_Success)
{
    SetupGeEnvironment();
    MockAclrtGetDeviceInfo(1);

    void* fakeStream = reinterpret_cast<void*>(0x1234);
    mgr_->SetAttachedStream("group1", 100, fakeStream);
    mgr_->SetAttachedStream("group2", 100, fakeStream);

    ThreadHandle thread = 0;
    HcclResult ret
        = mgr_->OrderLaunchThreadAcquire(HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_GE, nullptr, "group1", 1, thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(thread, static_cast<ThreadHandle>(0));
}

TEST_F(OrderLaunchThreadMgrTest, Ut_OrderLaunchThreadAcquire_When_GeNotSet_Expect_Success_ThreadZero)
{
    ThreadHandle thread = 0;
    HcclResult ret
        = mgr_->OrderLaunchThreadAcquire(HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_GE, nullptr, "group1", 1, thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(thread, static_cast<ThreadHandle>(0));
}

TEST_F(OrderLaunchThreadMgrTest, Ut_OrderLaunchThreadAcquire_When_DeviceCollCommNull_Expect_Error)
{
    MockGetCurrentContext(0x100);
    MockHcommThreadAlloc(HCCL_SUCCESS);
    MockAclrtGetDeviceInfo(1);
    mgr_->RegisterOrderLaunch("group1");
    mgr_->RegisterOrderLaunch("group2");

    ThreadHandle warmup1 = 0;
    mgr_->EnsureOrderThread(OrderThreadMode::OPBASE, "warmup1", 1, warmup1);
    ThreadHandle warmup2 = 0;
    mgr_->EnsureOrderThread(OrderThreadMode::OPBASE, "warmup2", 1, warmup2);

    ThreadHandle thread = 0;
    HcclResult ret
        = mgr_->OrderLaunchThreadAcquire(HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE, nullptr, "group1", 1, thread);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

/* ======================== Destroy ======================== */

TEST_F(OrderLaunchThreadMgrTest, Ut_Destroy_When_NoResources_Expect_NoCrash)
{
    mgr_.reset();
    SUCCEED();
}

TEST_F(OrderLaunchThreadMgrTest, Ut_Destroy_When_HasResources_Expect_NoCrash)
{
    MockGetCurrentContext(0x100);
    MockAclrtGetDeviceInfo(0);
    MockHcommThreadAlloc(HCCL_SUCCESS);
    MockHcommThreadFree(HCCL_SUCCESS);

    ThreadHandle thread = 0;
    mgr_->EnsureOrderThread(OrderThreadMode::OPBASE, "group1", 1, thread);

    mgr_.reset();
    SUCCEED();
}

TEST_F(OrderLaunchThreadMgrTest, Ut_Destroy_When_HasDeviceOrderThread_Expect_NoCrash)
{
    MockGetCurrentContext(0x100);
    MockAclrtGetDeviceInfo(1);
    mgr_->RegisterOrderLaunch("group1");
    mgr_->RegisterOrderLaunch("group2");

    mgr_.reset();
    SUCCEED();
}

TEST_F(OrderLaunchThreadMgrTest, Ut_Destroy_When_HasAttachedStream_Expect_NoCrash)
{
    void* fakeStream = reinterpret_cast<void*>(0x1234);
    mgr_->SetAttachedStream("group1", 100, fakeStream);

    mgr_.reset();
    SUCCEED();
}
