/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>

#define private public
#define protected public
#include "flush_manager.h"
#include "flush_handle.h"
#include "rdma_handle_manager.h"
#include "orion_adapter_rts.h"
#include "orion_adapter_hccp.h"
#include "hccp.h"
#undef private
#undef protected

using namespace Hccl;

namespace {
// ibv_post_send 为 static inline，通过 qp->context->ops.post_send 函数指针调用；
// 这里提供桩函数控制其返回值，避免依赖真实 RDMA 驱动。
int StubPostSendOk(struct ibv_qp*, struct ibv_send_wr*, struct ibv_send_wr**) { return 0; }
int StubPostSendFail(struct ibv_qp*, struct ibv_send_wr*, struct ibv_send_wr**) { return -1; }
// ProbeAndCacheRdmaHandle 桩：MOCKER_CPP mock 成员函数，桩函数首参须为对象指针(this)
HcclResult StubProbeLinkDown(RdmaHandleManager*, u32, const IpAddress&, bool& linkDown)
{
    linkDown = true;
    return HCCL_SUCCESS;
}
HcclResult StubProbeOk(RdmaHandleManager*, u32, const IpAddress&, bool& linkDown)
{
    linkDown = false;
    return HCCL_SUCCESS;
}
HcclResult StubProbeFail(RdmaHandleManager*, u32, const IpAddress&, bool& linkDown)
{
    linkDown = false;
    return HCCL_E_NETWORK;
}
} // namespace

class FlushManagerTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override { GlobalMockObject::verify(); }
};

// 覆盖 flush_manager.cc:96,97 — Flush() 调用 ExecuteRdmaRead(...GetExecTimeOut())
TEST_F(FlushManagerTest, Ut_Flush_When_PostSendFails_Expect_NetworkError)
{
    FlushManager fm;

    // 构造 fake ibv 结构：qp->context->ops.post_send 返回失败，
    // 使 ExecuteRdmaRead 在 ibv_post_send 失败后立即返回 HCCL_E_NETWORK。
    struct ibv_context_ops ops = {};
    ops.post_send = &StubPostSendFail;
    struct ibv_context ctx = {};
    ctx.ops = ops;
    struct ibv_cq cq = {};
    cq.context = &ctx;
    struct ibv_qp qp = {};
    qp.context = &ctx;
    qp.send_cq = &cq;

    auto handle = std::make_shared<FlushHandle>();
    handle->loopBackQpParam.ibvQp0 = &qp;
    IpAddress ip("192.168.1.1");
    fm.flushHandleMap_.insert({ip, handle});

    // Flush() 遍历 map，FlushParamPrepare 成功后调用 ExecuteRdmaRead（line 96-97），
    // ibv_post_send 失败 -> ExecuteRdmaRead 返回 HCCL_E_NETWORK -> Flush 返回 HCCL_E_NETWORK。
    HcclResult ret = fm.Flush();
    EXPECT_EQ(ret, HCCL_E_NETWORK);

    // 清空 map，避免 FlushManager 析构时 DestroyAll 触发 fake handle 的 Destroy。
    fm.flushHandleMap_.clear();
}

// 覆盖 flush_manager.cc:124,145 — ExecuteRdmaRead 入口 + 超时分支
TEST_F(FlushManagerTest, Ut_ExecuteRdmaRead_When_Timeout_Expect_HcclETimeout)
{
    FlushManager fm;

    // ibv_post_send 成功，进入轮询循环。
    struct ibv_context_ops ops = {};
    ops.post_send = &StubPostSendOk;
    struct ibv_context ctx = {};
    ctx.ops = ops;
    struct ibv_cq cq = {};
    cq.context = &ctx;
    struct ibv_qp qp = {};
    qp.context = &ctx;
    qp.send_cq = &cq;

    ibv_send_wr swr = {};
    ibv_sge sg = {};
    swr.sg_list = &sg;

    // timeoutSec=0：首次循环 elapsedSec(0) >= 0 立即触发超时分支（line 144-145），返回 HCCL_E_TIMEOUT。
    HcclResult ret = fm.ExecuteRdmaRead(&qp, &cq, swr, 0);
    EXPECT_EQ(ret, HCCL_E_TIMEOUT);
}

// 覆盖 initFlushHandle 入口探测网口 down（linkDown=true）分支：
// ProbeAndCacheRdmaHandle 返回 HCCL_SUCCESS 且 linkDown=true，
// initFlushHandle 打印 WARNING、不建 FlushHandle，直接返回 HCCL_SUCCESS。
TEST_F(FlushManagerTest, Ut_InitFlushHandle_When_LinkDown_Expect_SuccessAndNoHandle)
{
    FlushManager fm;

    MOCKER_CPP(&RdmaHandleManager::ProbeAndCacheRdmaHandle).stubs().will(invoke(StubProbeLinkDown));

    IpAddress ip("192.168.1.1");
    HcclResult ret = fm.initFlushHandle(ip, 0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(fm.flushHandleMap_.empty());
}

// 覆盖 initFlushHandle 入口探测成功分支：
// ProbeAndCacheRdmaHandle 返回 HCCL_SUCCESS 且 linkDown=false，
// 随后正常创建 FlushHandle 并插入 map。
TEST_F(FlushManagerTest, Ut_InitFlushHandle_When_ProbeSucceeds_Expect_FlushHandleCreated)
{
    FlushManager fm;
    void* fakeRdmaHandle = reinterpret_cast<void*>(0xDEAD0001);
    void* fakeMrHandle = reinterpret_cast<void*>(0xDEAD0003);
    void* fakeLocalMem = reinterpret_cast<void*>(0xBEEF0001);

    // 探测成功，rdmaHandle 已由 ProbeAndCacheRdmaHandle 缓存
    MOCKER_CPP(&RdmaHandleManager::ProbeAndCacheRdmaHandle).stubs().will(invoke(StubProbeOk));

    // FlushHandle::Init 建链链路
    MOCKER_CPP(&RdmaHandleManager::GetByAddr)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(fakeRdmaHandle));
    int lbMax = 1;
    MOCKER(RaGetLbMax).stubs().with(mockcpp::any(), outBoundP(&lbMax, sizeof(lbMax))).will(returnValue(0));
    MOCKER(HrtMalloc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(fakeLocalMem));
    MOCKER(RaLoopbackQpCreate).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any()).will(returnValue(0));
    MOCKER(RaRegisterMr)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&fakeMrHandle, sizeof(fakeMrHandle)))
        .will(returnValue(0));

    // map 清理触发 FlushHandle::Destroy 的释放链路
    MOCKER(RaDeregisterMr).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(0));
    MOCKER(RaQpDestroy).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER(HrtFree).stubs();

    IpAddress ip("192.168.1.1");
    HcclResult ret = fm.initFlushHandle(ip, 0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(fm.flushHandleMap_.size(), 1u);
    auto it = fm.flushHandleMap_.find(ip);
    ASSERT_NE(it, fm.flushHandleMap_.end());
    EXPECT_TRUE(it->second->flushIsInitialized);

    // 清空 map，避免 FlushManager 析构时对 fake 资源二次 Destroy。
    fm.flushHandleMap_.clear();
}

// 覆盖 initFlushHandle 入口探测返回非 linkdown 错误分支：
// ProbeAndCacheRdmaHandle 返回 HCCL_E_NETWORK，initFlushHandle 直接返回失败，不创建 FlushHandle。
TEST_F(FlushManagerTest, Ut_InitFlushHandle_When_ProbeFails_Expect_FailureAndNoHandle)
{
    FlushManager fm;

    MOCKER_CPP(&RdmaHandleManager::ProbeAndCacheRdmaHandle).stubs().will(invoke(StubProbeFail));

    IpAddress ip("192.168.1.1");
    HcclResult ret = fm.initFlushHandle(ip, 0);
    EXPECT_EQ(ret, HCCL_E_NETWORK);
    EXPECT_TRUE(fm.flushHandleMap_.empty());
}
