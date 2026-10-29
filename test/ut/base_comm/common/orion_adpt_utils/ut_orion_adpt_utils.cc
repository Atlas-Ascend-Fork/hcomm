/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

#include "orion_adpt_utils.h"
#include "hccl/hccl_types.h"
#include "hcomm_channel.h"
#include "adapter_rts_common.h"

using namespace hcomm;

namespace {

static constexpr u32 EXPECTED_DEFAULT_QOS = 4U; // Hccl::kRaUbGetTpInfoParamDefaultQos

// hrtGetDevice stub for mockcpp invoke
static HcclResult hrtGetDeviceTestStub(s32* deviceLogicId)
{
    if (deviceLogicId != nullptr) {
        *deviceLogicId = 0;
    }
    return HCCL_SUCCESS;
}

// ============================================================================
// PrepareUbConnBuildContext 测试用例集
// ============================================================================

class PrepareUbConnBuildContextTest : public testing::Test {
protected:
    void SetUp() override { MOCKER(hrtGetDevice).stubs().with(mockcpp::any()).will(invoke(hrtGetDeviceTestStub)); }

    void TearDown() override { GlobalMockObject::verify(); }

    static EndpointDesc MakeValidEndpointDesc()
    {
        EndpointDesc ep{};
        ep.protocol = COMM_PROTOCOL_UB_CTP;
        ep.commAddr.type = COMM_ADDR_TYPE_IP_V4;
        ep.commAddr.addr.s_addr = inet_addr("10.0.0.1");
        return ep;
    }

    static HcommChannelDesc MakeChannelDesc(uint32_t sqDepth, uint32_t scqDepth, uint32_t qos)
    {
        HcommChannelDesc desc{};
        desc.ubAttr.sqDepth = sqDepth;
        desc.ubAttr.scqDepth = scqDepth;
        desc.qos = qos;
        return desc;
    }
};

/**
 * 场景：指定 scqDepth 和 sqDepth，验证正确传播到 ctx
 */
TEST_F(PrepareUbConnBuildContextTest, SqAndScqDepth_PropagatedToContext)
{
    auto locEp = MakeValidEndpointDesc();
    auto rmtEp = MakeValidEndpointDesc();
    auto channelDesc = MakeChannelDesc(256U, 1024U, 3U);

    hcomm::UbConnBuildContext ctx{};
    EXPECT_EQ(hcomm::PrepareUbConnBuildContext(locEp, rmtEp, channelDesc, ctx), HCCL_SUCCESS);
    EXPECT_EQ(ctx.sqDepth, 256U);
    EXPECT_EQ(ctx.scqDepth, 1024U);
}

/**
 * 场景：scqDepth/sqDepth 为哨兵值(0xFFFFFFFF)，验证传播到 ctx
 */
TEST_F(PrepareUbConnBuildContextTest, SentinelDepth_PropagatedToContext)
{
    auto locEp = MakeValidEndpointDesc();
    auto rmtEp = MakeValidEndpointDesc();
    auto channelDesc = MakeChannelDesc(0xFFFFFFFFU, 0xFFFFFFFFU, 0U);

    hcomm::UbConnBuildContext ctx{};
    EXPECT_EQ(hcomm::PrepareUbConnBuildContext(locEp, rmtEp, channelDesc, ctx), HCCL_SUCCESS);
    EXPECT_EQ(ctx.scqDepth, 0xFFFFFFFFU);
    EXPECT_EQ(ctx.sqDepth, 0xFFFFFFFFU);
}

/**
 * 场景：scqDepth 为 0（使用默认值），验证传播到 ctx
 */
TEST_F(PrepareUbConnBuildContextTest, ZeroScqDepth_PropagatedToContext)
{
    auto locEp = MakeValidEndpointDesc();
    auto rmtEp = MakeValidEndpointDesc();
    auto channelDesc = MakeChannelDesc(0U, 0U, 0U);

    hcomm::UbConnBuildContext ctx{};
    EXPECT_EQ(hcomm::PrepareUbConnBuildContext(locEp, rmtEp, channelDesc, ctx), HCCL_SUCCESS);
    EXPECT_EQ(ctx.scqDepth, 0U);
}

/**
 * 场景：qos 为有效值（0-7），验证正确传播到 ctx.qosPre
 */
TEST_F(PrepareUbConnBuildContextTest, ValidQos_PropagatedToQosPre)
{
    auto locEp = MakeValidEndpointDesc();
    auto rmtEp = MakeValidEndpointDesc();
    auto channelDesc = MakeChannelDesc(256U, 1024U, 5U);

    hcomm::UbConnBuildContext ctx{};
    EXPECT_EQ(hcomm::PrepareUbConnBuildContext(locEp, rmtEp, channelDesc, ctx), HCCL_SUCCESS);
    EXPECT_EQ(ctx.qosPre, static_cast<u8>(5U));
}

/**
 * 场景：qos 超过 7，验证使用默认 qos（EXPECTED_DEFAULT_QOS=4）
 */
TEST_F(PrepareUbConnBuildContextTest, InvalidQos_UsesDefaultQos)
{
    auto locEp = MakeValidEndpointDesc();
    auto rmtEp = MakeValidEndpointDesc();
    auto channelDesc = MakeChannelDesc(256U, 1024U, 8U);

    hcomm::UbConnBuildContext ctx{};
    EXPECT_EQ(hcomm::PrepareUbConnBuildContext(locEp, rmtEp, channelDesc, ctx), HCCL_SUCCESS);
    EXPECT_EQ(ctx.qosPre, static_cast<u8>(EXPECTED_DEFAULT_QOS));
}

/**
 * 场景：无效的 commAddr 类型（COMM_ADDR_TYPE_RESERVED），应返回错误
 */
TEST_F(PrepareUbConnBuildContextTest, InvalidCommAddrType_ReturnsError)
{
    EndpointDesc locEp{};
    locEp.protocol = COMM_PROTOCOL_UB_CTP;
    locEp.commAddr.type = COMM_ADDR_TYPE_RESERVED;

    auto rmtEp = MakeValidEndpointDesc();
    auto channelDesc = MakeChannelDesc(256U, 1024U, 3U);

    hcomm::UbConnBuildContext ctx{};
    EXPECT_NE(hcomm::PrepareUbConnBuildContext(locEp, rmtEp, channelDesc, ctx), HCCL_SUCCESS);
}

/**
 * 场景：无效的协议（COMM_PROTOCOL_RESERVED），应返回错误
 */
TEST_F(PrepareUbConnBuildContextTest, InvalidProtocol_ReturnsError)
{
    EndpointDesc locEp{};
    locEp.protocol = COMM_PROTOCOL_RESERVED;
    locEp.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    locEp.commAddr.addr.s_addr = inet_addr("10.0.0.1");

    auto rmtEp = MakeValidEndpointDesc();
    auto channelDesc = MakeChannelDesc(256U, 1024U, 3U);

    hcomm::UbConnBuildContext ctx{};
    EXPECT_NE(hcomm::PrepareUbConnBuildContext(locEp, rmtEp, channelDesc, ctx), HCCL_SUCCESS);
}

// ============================================================================
// CheckUbScqDepth 测试用例集
// ============================================================================

/**
 * 场景：scqDepth为未设置哨兵值(0xFFFFFFFF)时，应返回SUCCESS（使用默认值）
 * 预期：返回 HCCL_SUCCESS
 */
TEST(CheckUbScqDepthTest, ScqDepth_NotSet_ReturnsSuccess)
{
    hcomm::UbConnBuildContext ctx{};
    ctx.scqDepth = 0xFFFFFFFFU; // UB_SQ_DEPTH_NOT_SET

    EXPECT_EQ(hcomm::CheckUbScqDepth(ctx, HrtUbJfcMode::NORMAL), HCCL_SUCCESS);
}

/**
 * 场景：scqDepth为0时，应返回SUCCESS（使用默认值）
 * 预期：返回 HCCL_SUCCESS
 */
TEST(CheckUbScqDepthTest, ScqDepth_Zero_ReturnsSuccess)
{
    hcomm::UbConnBuildContext ctx{};
    ctx.scqDepth = 0;

    EXPECT_EQ(hcomm::CheckUbScqDepth(ctx, HrtUbJfcMode::NORMAL), HCCL_SUCCESS);
}

/**
 * 场景：Host场景(NORMAL模式)，scqDepth取最小值64
 * 预期：返回 HCCL_SUCCESS
 */
TEST(CheckUbScqDepthTest, Host_MinValue_ReturnsSuccess)
{
    hcomm::UbConnBuildContext ctx{};
    ctx.scqDepth = UB_SCQ_DEPTH_MIN; // 64

    EXPECT_EQ(hcomm::CheckUbScqDepth(ctx, HrtUbJfcMode::NORMAL), HCCL_SUCCESS);
}

/**
 * 场景：Host场景(NORMAL模式)，scqDepth取最大值32768
 * 预期：返回 HCCL_SUCCESS
 */
TEST(CheckUbScqDepthTest, Host_MaxValue_ReturnsSuccess)
{
    hcomm::UbConnBuildContext ctx{};
    ctx.scqDepth = UB_SCQ_DEPTH_MAX_NORMAL; // 32768

    EXPECT_EQ(hcomm::CheckUbScqDepth(ctx, HrtUbJfcMode::NORMAL), HCCL_SUCCESS);
}

/**
 * 场景：Host场景(NORMAL模式)，scqDepth超出最大值32769
 * 预期：返回 HCCL_E_PARA（参数错误）
 */
TEST(CheckUbScqDepthTest, Host_ExceedMax_ReturnsError)
{
    hcomm::UbConnBuildContext ctx{};
    ctx.scqDepth = UB_SCQ_DEPTH_MAX_NORMAL + 1; // 32769

    EXPECT_EQ(hcomm::CheckUbScqDepth(ctx, HrtUbJfcMode::NORMAL), HCCL_E_PARA);
}

/**
 * 场景：Host场景(NORMAL模式)，scqDepth低于最小值63
 * 预期：返回 HCCL_E_PARA（参数错误）
 */
TEST(CheckUbScqDepthTest, Host_BelowMin_ReturnsError)
{
    hcomm::UbConnBuildContext ctx{};
    ctx.scqDepth = UB_SCQ_DEPTH_MIN - 1; // 63

    EXPECT_EQ(hcomm::CheckUbScqDepth(ctx, HrtUbJfcMode::NORMAL), HCCL_E_PARA);
}

/**
 * 场景：AICPU场景(STARS_POLL模式)，scqDepth取最大值16384
 * 预期：返回 HCCL_SUCCESS
 */
TEST(CheckUbScqDepthTest, Aicpu_MaxValue_ReturnsSuccess)
{
    hcomm::UbConnBuildContext ctx{};
    ctx.scqDepth = UB_SCQ_DEPTH_MAX_STARS_POLL; // 16384

    EXPECT_EQ(hcomm::CheckUbScqDepth(ctx, HrtUbJfcMode::STARS_POLL), HCCL_SUCCESS);
}

/**
 * 场景：AICPU场景(STARS_POLL模式)，scqDepth超出最大值16385
 * 预期：返回 HCCL_E_PARA（参数错误）
 */
TEST(CheckUbScqDepthTest, Aicpu_ExceedMax_ReturnsError)
{
    hcomm::UbConnBuildContext ctx{};
    ctx.scqDepth = UB_SCQ_DEPTH_MAX_STARS_POLL + 1; // 16385

    EXPECT_EQ(hcomm::CheckUbScqDepth(ctx, HrtUbJfcMode::STARS_POLL), HCCL_E_PARA);
}

/**
 * 场景：jfcMode为无效值(非NORMAL和STARS_POLL)
 * 预期：返回 HCCL_E_PARA（参数错误）
 */
TEST(CheckUbScqDepthTest, InvalidJfcMode_ReturnsError)
{
    hcomm::UbConnBuildContext ctx{};
    ctx.scqDepth = 1024;

    EXPECT_EQ(hcomm::CheckUbScqDepth(ctx, HrtUbJfcMode::CCU_POLL), HCCL_E_PARA);
}

// ============================================================================
// HcclIpAddressToIpAddress / IpAddressToHcclIpAddress 测试用例集
// ============================================================================

/**
 * 场景：V4 地址 V1->V2->V1 往返转换
 * 预期：family、二进制地址、scopeID 全程保持一致
 */
TEST(HcclIpAddressConvertTest, V4_RoundTrip_KeepsFamilyAddrAndScopeId)
{
    hccl::HcclInAddr bin{};
    bin.addr.s_addr = inet_addr("192.168.10.1");
    hccl::HcclIpAddress v1(AF_INET, bin);
    (void)v1.SetScopeID(3);

    const Hccl::IpAddress v2 = HcclIpAddressToIpAddress(v1);
    ASSERT_EQ(v2.GetFamily(), AF_INET);
    ASSERT_EQ(v2.GetBinaryAddress().addr.s_addr, inet_addr("192.168.10.1"));
    ASSERT_EQ(v2.GetScopeID(), 3);

    const hccl::HcclIpAddress back = IpAddressToHcclIpAddress(v2);
    ASSERT_EQ(back.GetFamily(), AF_INET);
    ASSERT_EQ(back.GetBinaryAddress().addr.s_addr, inet_addr("192.168.10.1"));
    ASSERT_EQ(back.GetScopeID(), 3);
}

/**
 * 场景：V6 地址 V1->V2->V1 往返转换
 * 预期：family、128 位地址（经字符串校验）、scopeID 全程保持一致
 */
TEST(HcclIpAddressConvertTest, V6_RoundTrip_KeepsFamilyAddrAndScopeId)
{
    struct in6_addr addr6 {};
    ASSERT_EQ(inet_pton(AF_INET6, "fe80::1", &addr6), 1);
    hccl::HcclInAddr bin{};
    bin.addr6 = addr6;
    hccl::HcclIpAddress v1(AF_INET6, bin);
    (void)v1.SetScopeID(5);

    const Hccl::IpAddress v2 = HcclIpAddressToIpAddress(v1);
    ASSERT_EQ(v2.GetFamily(), AF_INET6);
    ASSERT_EQ(v2.GetScopeID(), 5);
    ASSERT_EQ(v2.GetIpStr(), v1.GetIpStr());

    const hccl::HcclIpAddress back = IpAddressToHcclIpAddress(v2);
    ASSERT_EQ(back.GetFamily(), AF_INET6);
    ASSERT_EQ(back.GetScopeID(), 5);
    ASSERT_EQ(back.GetIpStr(), v1.GetIpStr());
}

} // namespace
