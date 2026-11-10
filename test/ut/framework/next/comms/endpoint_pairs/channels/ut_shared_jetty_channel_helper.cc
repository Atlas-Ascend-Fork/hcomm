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
#include <functional>
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>

#define private public
#define protected public
#include "shared_jetty_channel_helper.h"
#include "endpoint.h"
#include "urma_endpoint.h"
#include "comm_queue_context/jetty_context.h"
#include "dev_ub_connection.h"
#include "shared_jetty_connection_adapter.h"
#include "ut_shared_jetty_test_helper.h"
#undef private
#undef protected

using namespace hcomm;
using namespace Hccl;

namespace {
class StubEndpointForHelper : public UrmaEndpoint {
public:
    StubEndpointForHelper() : UrmaEndpoint(MakeDesc())
    {
        // GetCommQueueContext 已上移为 Endpoint 基类虚函数（stub 经 UrmaEndpoint override 生效）；
        // 此处预创建供测试断言，与 GetCommQueueContext() 首调的延迟创建效果等价。
        jettyContext_ = std::make_unique<JettyContext>();
    }

    HcclResult Init() override { return HCCL_SUCCESS; }

private:
    static EndpointDesc MakeDesc()
    {
        EndpointDesc desc{};
        desc.protocol = COMM_PROTOCOL_UB_CTP;
        desc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
        Hccl::IpAddress localIp("127.0.0.1");
        desc.commAddr.addr = localIp.GetBinaryAddress().addr;
        desc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
        desc.loc.device.devPhyId = 3;
        return desc;
    }
};

inline std::function<std::unique_ptr<DevUbConnection>()> MakeCountingFactory(uint32_t& count)
{
    return [&count]() -> std::unique_ptr<DevUbConnection> {
        count++;
        return MakeTestJettyConnection();
    };
}

} // namespace

class SharedJettyChannelHelperTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
    }
    static void TearDownTestCase()
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
    }
    void SetUp() override
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
    }
    void TearDown() override
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
    }
};

TEST_F(SharedJettyChannelHelperTest, Ut_AcquireSharedJettyForChannel_When_NullEndpoint_Expect_HCCL_E_PARA)
{
    auto conn = MakeTestJettyConnection();
    auto factory = []() {
        return MakeTestJettyConnection();
    };
    JettyContext::Ctx outCtx;
    EXPECT_EQ(AcquireSharedJettyForChannel(nullptr, conn.get(), factory, outCtx), HCCL_E_PARA);
}

TEST_F(SharedJettyChannelHelperTest, Ut_AcquireSharedJettyForChannel_When_NullConnection_Expect_HCCL_E_PARA)
{
    StubEndpointForHelper endpoint;
    auto factory = []() {
        return MakeTestJettyConnection();
    };
    JettyContext::Ctx outCtx;
    EXPECT_EQ(AcquireSharedJettyForChannel(&endpoint, nullptr, factory, outCtx), HCCL_E_PARA);
}

TEST_F(SharedJettyChannelHelperTest, Ut_AcquireSharedJettyForChannel_When_FirstCreate_Expect_Success)
{
    StubEndpointForHelper endpoint;
    auto conn = MakeTestJettyConnection(DevUbConnection::JettyMode::EXTERNAL_INJECT);
    uint32_t factoryCallCount = 0;
    auto factory = MakeCountingFactory(factoryCallCount);

    JettyContext::Ctx outCtx;
    EXPECT_EQ(AcquireSharedJettyForChannel(&endpoint, conn.get(), factory, outCtx), HCCL_SUCCESS);
    EXPECT_EQ(factoryCallCount, 1U);
    EXPECT_EQ(conn->jettyMode_, DevUbConnection::JettyMode::EXTERNAL_INJECT);
}

TEST_F(SharedJettyChannelHelperTest, Ut_AcquireSharedJettyForChannel_When_CacheHit_Expect_Success)
{
    StubEndpointForHelper endpoint;
    auto conn1 = MakeTestJettyConnection(DevUbConnection::JettyMode::EXTERNAL_INJECT);
    auto conn2 = MakeTestJettyConnection(DevUbConnection::JettyMode::EXTERNAL_INJECT);
    uint32_t factoryCallCount = 0;
    auto factory = MakeCountingFactory(factoryCallCount);

    JettyContext::Ctx outCtx;
    ASSERT_EQ(AcquireSharedJettyForChannel(&endpoint, conn1.get(), factory, outCtx), HCCL_SUCCESS);
    EXPECT_EQ(factoryCallCount, 1U);

    EXPECT_EQ(AcquireSharedJettyForChannel(&endpoint, conn2.get(), factory, outCtx), HCCL_SUCCESS);
    EXPECT_EQ(factoryCallCount, 1U);
    EXPECT_EQ(conn2->jettyMode_, DevUbConnection::JettyMode::EXTERNAL_INJECT);
}

TEST_F(SharedJettyChannelHelperTest, Ut_AcquireSharedJettyForChannel_When_ReleaseDecrementsRef_Expect_Success)
{
    StubEndpointForHelper endpoint;
    auto conn = MakeTestJettyConnection(DevUbConnection::JettyMode::EXTERNAL_INJECT);
    auto factory = []() {
        return MakeTestJettyConnection();
    };

    JettyContext::Ctx outCtx;
    ASSERT_EQ(AcquireSharedJettyForChannel(&endpoint, conn.get(), factory, outCtx), HCCL_SUCCESS);

    EXPECT_EQ(endpoint.jettyContext_->Release(), HCCL_SUCCESS);
    EXPECT_EQ(endpoint.jettyContext_->Release(), HCCL_SUCCESS);
}

TEST_F(SharedJettyChannelHelperTest, Ut_AcquireSharedJettyForChannel_When_ReleaseAndReacquire_Expect_Success)
{
    StubEndpointForHelper endpoint;
    auto conn1 = MakeTestJettyConnection(DevUbConnection::JettyMode::EXTERNAL_INJECT);
    auto conn2 = MakeTestJettyConnection(DevUbConnection::JettyMode::EXTERNAL_INJECT);
    uint32_t factoryCallCount = 0;
    auto factory = MakeCountingFactory(factoryCallCount);

    JettyContext::Ctx outCtx;
    ASSERT_EQ(AcquireSharedJettyForChannel(&endpoint, conn1.get(), factory, outCtx), HCCL_SUCCESS);
    EXPECT_EQ(endpoint.jettyContext_->Release(), HCCL_SUCCESS);

    EXPECT_EQ(AcquireSharedJettyForChannel(&endpoint, conn2.get(), factory, outCtx), HCCL_SUCCESS);
    EXPECT_EQ(factoryCallCount, 2U);
    EXPECT_EQ(conn2->jettyMode_, DevUbConnection::JettyMode::EXTERNAL_INJECT);
}

TEST_F(SharedJettyChannelHelperTest, Ut_AcquireSharedJettyForChannel_When_TempConnHasTp_Expect_TpOwnershipTransferred)
{
    // 首建路径端到端：临时 connection 持有有效 tpHandle 时，TP 引用所有权移交 JettyContext 保管，
    // Release 归零（共享 jetty 销毁）时统一释放，修复"临时 connection 析构即释放"击穿 TpManager 缓存
    StubEndpointForHelper endpoint;
    auto conn = MakeTestJettyConnection(DevUbConnection::JettyMode::EXTERNAL_INJECT);
    uint32_t factoryCallCount = 0;
    auto factory = [&factoryCallCount]() -> std::unique_ptr<DevUbConnection> {
        factoryCallCount++;
        auto tempConn = MakeTestJettyConnection();
        // 模拟临时 connection GetTpInfo 已完成：持有有效 tpHandle
        tempConn->tpProtocol = Hccl::TpProtocol::CTP;
        tempConn->tpInfo.tpHandle = 0x87654321FULL;
        return tempConn;
    };

    JettyContext::Ctx outCtx;
    ASSERT_EQ(AcquireSharedJettyForChannel(&endpoint, conn.get(), factory, outCtx), HCCL_SUCCESS);
    EXPECT_EQ(factoryCallCount, 1U);
    // TP 释放回调已移交 JettyContext；channel 视图（outCtx）不携带
    EXPECT_TRUE(endpoint.jettyContext_->inner_.tpReleaseCb != nullptr);
    EXPECT_TRUE(outCtx.tpReleaseCb == nullptr);

    // Release 归零销毁共享资源并执行回调（stub 环境 ReleaseUbConnectionTp 恒成功），返回 SUCCESS
    EXPECT_EQ(endpoint.jettyContext_->Release(), HCCL_SUCCESS);
    EXPECT_TRUE(endpoint.jettyContext_->inner_.tpReleaseCb == nullptr);
}

TEST_F(SharedJettyChannelHelperTest, Ut_AcquireSharedJettyForChannel_When_MidStageFail_Expect_TpRefNotTransferred)
{
    // 失败注入（jetty 所有权分离失败，TP 移交前的最后一步）：临时 connection 已持有有效 tpHandle，
    // 但 TP 引用移交（DetachConnTpOwnership）尚未执行——回调未建立、Inner 不保管，
    // releaseTpOnDestroy_ 保持 true，TP 引用由临时 connection 析构 ReleaseTp 兜底（旧行为回退）。
    // 固化"所有可失败步骤完成后才移交"的顺序不变量：provider 失败路径不得残留已填充的回调。
    // 注：失败点仅 mock 非虚自由函数（MOCKER(DetachConnJetty)，与 memcpy_s 同机制）；
    // 不得用 MOCKER_CPP mock 虚函数（GetStatus 等）——会改写 vtable，临时 connection
    // 在 provider 内析构时经虚析构分发即段错误（无堆栈、无断言输出）。
    StubEndpointForHelper endpoint;
    auto conn = MakeTestJettyConnection(DevUbConnection::JettyMode::EXTERNAL_INJECT);
    uint32_t factoryCallCount = 0;
    auto factory = [&factoryCallCount]() -> std::unique_ptr<DevUbConnection> {
        factoryCallCount++;
        auto tempConn = MakeTestJettyConnection();
        // 模拟临时 connection GetTpInfo 已完成：持有有效 tpHandle；协议同步设置
        // （与 When_TempConnHasTp 用例一致，避免析构回退路径携带未初始化协议）
        tempConn->tpProtocol = Hccl::TpProtocol::CTP;
        tempConn->tpInfo.tpHandle = 0x87654321ULL;
        return tempConn;
    };

    MOCKER(DetachConnJetty).stubs().will(returnValue(HcclResult::HCCL_E_PARA));
    JettyContext::Ctx outCtx;
    EXPECT_EQ(AcquireSharedJettyForChannel(&endpoint, conn.get(), factory, outCtx), HCCL_E_PARA);
    EXPECT_EQ(factoryCallCount, 1U);
    // 回调未建立：移交未发生，TP 引用走临时 connection 析构兜底
    EXPECT_TRUE(endpoint.jettyContext_->inner_.tpReleaseCb == nullptr);
    EXPECT_FALSE(endpoint.jettyContext_->inner_.valid);
    EXPECT_FALSE(endpoint.jettyContext_->inner_.creating);
    GlobalMockObject::verify();
    // verify() 仅校验期望、不移除桩（mockcpp 语义：除桩须 reset()）：
    // 不 reset 则 DetachConnJetty 桩仍生效，后半段重建路径必然失败
    GlobalMockObject::reset();

    // 失败后同 endpoint 可重新创建（顺序不变量与失败路径状态清理不被破坏）
    JettyContext::Ctx outCtx2;
    EXPECT_EQ(AcquireSharedJettyForChannel(&endpoint, conn.get(), factory, outCtx2), HCCL_SUCCESS);
    EXPECT_EQ(factoryCallCount, 2U);
    // 重建的临时 connection 同样持有有效 tpHandle：移交正常发生，回调由 Inner 保管
    EXPECT_TRUE(endpoint.jettyContext_->inner_.tpReleaseCb != nullptr);
    // 全生命周期收尾：Release 归零销毁共享资源并执行回调（与 When_TempConnHasTp 用例一致）
    EXPECT_EQ(endpoint.jettyContext_->Release(), HCCL_SUCCESS);
}

TEST_F(SharedJettyChannelHelperTest, Ut_AcquireSharedJettyForChannel_When_TempConnNoTp_Expect_NoOwnershipTransfer)
{
    // 临时 connection 无有效 tpHandle（GetTpInfo 未完成/失败场景）：不移交，行为与旧行为一致
    StubEndpointForHelper endpoint;
    auto conn = MakeTestJettyConnection(DevUbConnection::JettyMode::EXTERNAL_INJECT);
    uint32_t factoryCallCount = 0;
    auto factory = MakeCountingFactory(factoryCallCount);

    JettyContext::Ctx outCtx;
    ASSERT_EQ(AcquireSharedJettyForChannel(&endpoint, conn.get(), factory, outCtx), HCCL_SUCCESS);
    EXPECT_EQ(factoryCallCount, 1U);
    EXPECT_TRUE(endpoint.jettyContext_->inner_.tpReleaseCb == nullptr);

    EXPECT_EQ(endpoint.jettyContext_->Release(), HCCL_SUCCESS);
}
