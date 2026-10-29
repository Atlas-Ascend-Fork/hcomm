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
#include <mockcpp/mockcpp.hpp>

#define private public
#define protected public
#include "aicpu/aicpu_ts_hccs_channel.h"
#include "endpoint.h"
#include "channel_param.h"
#include "hccs_reged_mem_mgr.h"
#undef private
#undef protected

#include "hcomm_c_adpt.h" // HcommEndpointStartListen/HcommEndpointListenConfig（StartListen 收敛基类钉住用例）

using namespace hcomm;

namespace {
class StubAicpuTsHccsEndpoint : public AicpuTsHccsEndpoint {
public:
    explicit StubAicpuTsHccsEndpoint(const EndpointDesc& desc) : AicpuTsHccsEndpoint(desc) {}
    HcclResult Init() override { return HCCL_SUCCESS; }
};
} // namespace

class AicpuTsHccsChannelTest : public testing::Test {
protected:
    void TearDown() override { GlobalMockObject::verify(); }
};

static HcclResult StubGetTransportAttr(hccl::Transport* /*self*/, hccl::TransportAttr& attr)
{
    attr = hccl::TransportAttr{};
    return HCCL_SUCCESS;
}

TEST_F(AicpuTsHccsChannelTest, UT_BuildHcclChannelHccsRes_WhenQosConfigured_ExpectQosPropagated)
{
    EndpointDesc epDesc{};
    epDesc.protocol = COMM_PROTOCOL_HCCS;
    epDesc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    epDesc.loc.device.devPhyId = 3;
    StubAicpuTsHccsEndpoint stubEp(epDesc);
    stubEp.regedMemMgr_ = std::make_shared<HccsRegedMemMgr>(nullptr);

    HcommChannelDesc desc{};
    desc.qos = 42;
    desc.role = HCOMM_SOCKET_ROLE_CLIENT;
    desc.remoteEndpoint.protocol = COMM_PROTOCOL_HCCS;
    desc.remoteEndpoint.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    desc.remoteEndpoint.loc.device.devPhyId = 4;

    AicpuTsHccsChannel ch(reinterpret_cast<EndpointHandle>(&stubEp), desc);
    ch.socketTag_ = "ut_test";
    ch.local_.isSocketServer = false;
    ch.local_.ep.protocol = COMM_PROTOCOL_HCCS;
    ch.local_.ep.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    ch.local_.ep.loc.device.devPhyId = 3;
    ch.remote_.ep.protocol = COMM_PROTOCOL_HCCS;
    ch.remote_.ep.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    ch.remote_.ep.loc.device.devPhyId = 4;
    ch.local_.epPtr = &stubEp;
    ch.transport_ = std::make_shared<hccl::Transport>();

    MOCKER_CPP(&hccl::Transport::GetNotifyNum).stubs().will(returnValue(8U));
    MOCKER_CPP(&hccl::Transport::GetTransportAttr).stubs().will(invoke(StubGetTransportAttr));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceIndexByPhyId).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HccsRegedMemMgr::GetRemoteIpcRmaBufferEx).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HccsRegedMemMgr::GetLocalIpcRmaBufferEx).stubs().will(returnValue(HCCL_SUCCESS));

    HcclChannelHccsRes res;
    HcclResult ret = ch.BuildHcclChannelHccsRes(res);

    ASSERT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(res.channelP2p.qos, static_cast<u32>(desc.qos));
}

// ===================== StartListen 钉住用例（hccs 自有实现，不调基类）=====================
// master 语义：HCCS 从不端点监听/自建监听——server 侧由 BuildConnection 的 GlobalNetDevMgr::ServerInit
// 延迟初始化，数据 socket 亦全程走 GlobalNetDevMgr。HCCS 的 ParseInputParam 按 IP 推导角色
// （GetDeviceVnicIP 为静态调用不可安全打桩），故沿用本文件既有模式手工置位成员，
// 分别钉住 server/client 两角色均不触发任何监听。

TEST_F(AicpuTsHccsChannelTest, Ut_StartListen_WhenServerSide_Expect_NoListen)
{
    EndpointDesc epDesc{};
    epDesc.protocol = COMM_PROTOCOL_HCCS;
    epDesc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    epDesc.loc.device.devPhyId = 3;
    StubAicpuTsHccsEndpoint stubEp(epDesc);
    HcommChannelDesc desc{};
    desc.role = HCOMM_SOCKET_ROLE_RESERVED; // 语义来源：RESERVED 经 IP 推导出 server 端
    AicpuTsHccsChannel ch(reinterpret_cast<EndpointHandle>(&stubEp), desc);
    ch.local_.isSocketServer = true;
    ch.local_.serverPort = 16666; // AICPU HCCS 侧默认监听端口（AICPU_CHANNEL_DEFAULT_PORT）

    MOCKER(HcommEndpointStartListen).expects(never());
    EXPECT_EQ(ch.StartListen(), HCCL_SUCCESS);
    EXPECT_EQ(ch.local_.serverSocket, nullptr); // server 侧不监听：server 由 BuildConnection 的 ServerInit 延迟初始化
}

TEST_F(AicpuTsHccsChannelTest, Ut_StartListen_WhenClientSide_Expect_NoListen)
{
    EndpointDesc epDesc{};
    epDesc.protocol = COMM_PROTOCOL_HCCS;
    epDesc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    epDesc.loc.device.devPhyId = 3;
    StubAicpuTsHccsEndpoint stubEp(epDesc);
    HcommChannelDesc desc{};
    desc.role = HCOMM_SOCKET_ROLE_RESERVED;
    AicpuTsHccsChannel ch(reinterpret_cast<EndpointHandle>(&stubEp), desc);
    ch.local_.isSocketServer = false;
    ch.local_.serverPort = 16666;

    MOCKER(HcommEndpointStartListen).expects(never());
    EXPECT_EQ(ch.StartListen(), HCCL_SUCCESS);
    EXPECT_EQ(ch.local_.serverSocket, nullptr); // client 侧不监听；亦不自建
}

// ===================== BuildSocket 收敛基类后的钉住用例 =====================
// HCCS 无 BuildSocket 流程：traits_.socketBuildMode 折算为 NONE（原基类 PCIE 监听原型对 HCCS 属误伤，收敛后跳过）

TEST_F(AicpuTsHccsChannelTest, Ut_GetSocketBuildMode_When_Hccs_Expect_None)
{
    EndpointDesc epDesc{};
    epDesc.protocol = COMM_PROTOCOL_HCCS;
    epDesc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    epDesc.loc.device.devPhyId = 3;
    StubAicpuTsHccsEndpoint stubEp(epDesc);
    HcommChannelDesc desc{};
    AicpuTsHccsChannel ch(reinterpret_cast<EndpointHandle>(&stubEp), desc);
    // HCCS ParseInputParam 依赖 GetDeviceVnicIP（静态不可打桩），此处直测 kind 分派结果
    EXPECT_EQ(ch.GetSocketBuildMode(), SocketBuildMode::NONE);
}

TEST_F(AicpuTsHccsChannelTest, Ut_BuildSocket_WhenModeNone_Expect_NoOpSuccess)
{
    EndpointDesc epDesc{};
    epDesc.protocol = COMM_PROTOCOL_HCCS;
    epDesc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    epDesc.loc.device.devPhyId = 3;
    StubAicpuTsHccsEndpoint stubEp(epDesc);
    HcommChannelDesc desc{};
    AicpuTsHccsChannel ch(reinterpret_cast<EndpointHandle>(&stubEp), desc);
    ch.traits_.socketBuildMode = SocketBuildMode::NONE; // 模拟 ParseInputParam 对 HCCS 的折算结果（VNIC 查询不可打桩）

    EXPECT_EQ(ch.BuildSocket(), HCCL_SUCCESS);
    EXPECT_EQ(ch.socket_, nullptr);             // 未注入不受影响
    EXPECT_EQ(ch.local_.serverSocket, nullptr); // 不建监听 socket
}
