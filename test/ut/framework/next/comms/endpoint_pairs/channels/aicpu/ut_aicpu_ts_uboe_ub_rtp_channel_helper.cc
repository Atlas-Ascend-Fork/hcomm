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
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>

#define private public
#define protected public
#include "aicpu/aicpu_ts_uboe_channel.h"
#include "aicpu/aicpu_ts_ub_rtp_channel.h"
#undef protected
#undef private
#include "endpoint.h"
#include "sockets/socket_mgr.h"

using namespace hcomm;

namespace {
// IpAddress 按地址字节比较：127.0.0.1 < 127.0.0.2，用于确定自动模式（RESERVED）推导角色
constexpr const char* SMALL_IP = "127.0.0.1";
constexpr const char* LARGE_IP = "127.0.0.2";
// 与被测实现中 SERVER_LISTEN_PORT / 旧构造默认端口（DEFAULT_LISTENING_PORT）一致
constexpr uint32_t EXPECTED_DEFAULT_PORT = 60001;
constexpr uint16_t CUSTOM_PORT = 61001;

EndpointDesc MakeEpDesc(CommProtocol protocol, const std::string& ip)
{
    EndpointDesc desc{};
    desc.protocol = protocol;
    desc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    desc.commAddr.addr = Hccl::IpAddress(ip).GetBinaryAddress().addr;
    desc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    return desc;
}

// BuildServerSocket 桩：记录调用次数与监听端口，避免触达真实 RA 层
uint32_t g_buildServerSocketCount = 0;
uint32_t g_lastListenPort = 0;
HcclResult StubBuildServerSocket(AicpuTsUboeUbRtpChannelHelper*, uint32_t listenPort)
{
    g_buildServerSocketCount++;
    g_lastListenPort = listenPort;
    return HCCL_SUCCESS;
}

// GetSocket 桩：按 SocketConfig 角色返回 fake socket（Socket 构造仅成员赋值，析构无副作用）
HcclResult StubGetSocket(SocketMgr*, const Hccl::SocketConfig& socketConfig, Hccl::Socket*& socket)
{
    static Hccl::Socket serverSocket(
        nullptr, Hccl::IpAddress(), 0, Hccl::IpAddress(), "ut_server", Hccl::SocketRole::SERVER,
        Hccl::NicType::DEVICE_NIC_TYPE);
    static Hccl::Socket clientSocket(
        nullptr, Hccl::IpAddress(), 0, Hccl::IpAddress(), "ut_client", Hccl::SocketRole::CLIENT,
        Hccl::NicType::DEVICE_NIC_TYPE);
    socket = (socketConfig.GetRole() == Hccl::SocketRole::CLIENT) ? &clientSocket : &serverSocket;
    return HCCL_SUCCESS;
}

// 测试输入：endpoint 对与通道描述
struct TestInput {
    CommProtocol protocol;
    std::string localIp;
    std::string remoteIp;
    HcommSocketRole role;
    uint16_t port;
    const char* channelName;
};

// 断言期望：监听行为（次数/端口）+ SocketConfig 字段（端口/角色/去重标记/hccpTag/先收方向）
struct TestExpect {
    uint32_t listenCount;
    uint32_t listenPort;
    uint32_t configPort;
    Hccl::SocketRole configRole;
    bool noRankId;
    std::string hccpTag;
    bool recvFirst;
};
} // namespace

class AicpuTsUboeUbRtpChannelHelperTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override
    {
        const char* dfsConfig = std::getenv("HCCL_DFS_CONFIG");
        if (dfsConfig != nullptr) {
            savedDfsConfig_ = dfsConfig;
            hadDfsConfig_ = true;
        }
        (void)setenv("HCCL_DFS_CONFIG", "task_exception:on", 1);
        g_buildServerSocketCount = 0;
        g_lastListenPort = 0;
        MOCKER_CPP(&AicpuTsUboeUbRtpChannelHelper::BuildServerSocket).stubs().will(invoke(StubBuildServerSocket));
        MOCKER_CPP(&SocketMgr::GetSocket).stubs().will(invoke(StubGetSocket));
    }

    void TearDown() override
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
        if (hadDfsConfig_) {
            (void)setenv("HCCL_DFS_CONFIG", savedDfsConfig_.c_str(), 1);
        } else {
            (void)unsetenv("HCCL_DFS_CONFIG");
        }
    }

    // 统一执行 BuildSocket 并按 TestExpect 断言。
    // UBOE 与 UB_RTP 共用 helper，两个子类各跑一遍保证协议路径一致。
    template <typename ChannelT>
    void RunBuildSocketAndExpect(const TestInput& in, const TestExpect& exp)
    {
        g_buildServerSocketCount = 0;
        g_lastListenPort = 0;

        HcommChannelDesc desc{};
        desc.role = in.role;
        desc.port = in.port;
        desc.channelName = in.channelName;

        ChannelT ch(reinterpret_cast<EndpointHandle>(0x1234), desc);
        ch.localEp_ = MakeEpDesc(in.protocol, in.localIp);
        ch.remoteEp_ = MakeEpDesc(in.protocol, in.remoteIp);
        ch.socket_ = nullptr;
        ch.devicePhyId_ = 0;

        EXPECT_EQ(ch.BuildSocket(), HCCL_SUCCESS);
        EXPECT_EQ(g_buildServerSocketCount, exp.listenCount);
        if (exp.listenCount > 0) {
            EXPECT_EQ(g_lastListenPort, exp.listenPort);
        }
        ASSERT_NE(ch.socketConfigHolder_, nullptr);
        EXPECT_EQ(ch.socketConfigHolder_->listeningPort, exp.configPort);
        EXPECT_EQ(ch.socketConfigHolder_->GetRole(), exp.configRole);
        EXPECT_EQ(ch.socketConfigHolder_->noRankId, exp.noRankId);
        EXPECT_EQ(ch.socketConfigHolder_->GetHccpTag(), exp.hccpTag);
        EXPECT_EQ(ch.isRecvFirst_, exp.recvFirst);
    }

private:
    std::string savedDfsConfig_;
    bool hadDfsConfig_{false};
};

// 自动模式 + port=0 + 本端地址较小（推导 SERVER）：默认端口监听一次；
// hccpTag 沿用旧构造格式（无端口后缀），保持存量 wire 兼容
TEST_F(AicpuTsUboeUbRtpChannelHelperTest, Ut_BuildSocket_When_ReservedPort0DerivedServer_Expect_ListenDefaultPort)
{
    RunBuildSocketAndExpect<AicpuTsUboeChannel>(
        {COMM_PROTOCOL_UBOE, SMALL_IP, LARGE_IP, HCOMM_SOCKET_ROLE_RESERVED, 0, nullptr},
        {1, EXPECTED_DEFAULT_PORT, EXPECTED_DEFAULT_PORT, Hccl::SocketRole::SERVER, true,
         "AUTOMATIC_SOCKET_TAG_127.0.0.1_127.0.0.2", false});
    RunBuildSocketAndExpect<AicpuTsUbRtpChannel>(
        {COMM_PROTOCOL_UB_RTP, SMALL_IP, LARGE_IP, HCOMM_SOCKET_ROLE_RESERVED, 0, nullptr},
        {1, EXPECTED_DEFAULT_PORT, EXPECTED_DEFAULT_PORT, Hccl::SocketRole::SERVER, true,
         "AUTOMATIC_SOCKET_TAG_127.0.0.1_127.0.0.2", false});
}

// 自动模式 + port=0 + 本端地址较大（推导 CLIENT）：不创建监听 socket，先收方向
TEST_F(AicpuTsUboeUbRtpChannelHelperTest, Ut_BuildSocket_When_ReservedPort0DerivedClient_Expect_NoListenRecvFirst)
{
    RunBuildSocketAndExpect<AicpuTsUboeChannel>(
        {COMM_PROTOCOL_UBOE, LARGE_IP, SMALL_IP, HCOMM_SOCKET_ROLE_RESERVED, 0, nullptr},
        {0, 0, EXPECTED_DEFAULT_PORT, Hccl::SocketRole::CLIENT, true, "AUTOMATIC_SOCKET_TAG_127.0.0.1_127.0.0.2",
         true});
    RunBuildSocketAndExpect<AicpuTsUbRtpChannel>(
        {COMM_PROTOCOL_UB_RTP, LARGE_IP, SMALL_IP, HCOMM_SOCKET_ROLE_RESERVED, 0, nullptr},
        {0, 0, EXPECTED_DEFAULT_PORT, Hccl::SocketRole::CLIENT, true, "AUTOMATIC_SOCKET_TAG_127.0.0.1_127.0.0.2",
         true});
}

// 自动模式 + 仅传 port：端口生效（监听/SocketConfig/hccpTag 均带端口），角色仍自动推导
TEST_F(AicpuTsUboeUbRtpChannelHelperTest, Ut_BuildSocket_When_ReservedCustomPort_Expect_PortEffective)
{
    RunBuildSocketAndExpect<AicpuTsUboeChannel>(
        {COMM_PROTOCOL_UBOE, SMALL_IP, LARGE_IP, HCOMM_SOCKET_ROLE_RESERVED, CUSTOM_PORT, nullptr},
        {1, CUSTOM_PORT, CUSTOM_PORT, Hccl::SocketRole::SERVER, true, "AUTOMATIC_SOCKET_TAG_127.0.0.1_127.0.0.2_61001",
         false});
    RunBuildSocketAndExpect<AicpuTsUbRtpChannel>(
        {COMM_PROTOCOL_UB_RTP, SMALL_IP, LARGE_IP, HCOMM_SOCKET_ROLE_RESERVED, CUSTOM_PORT, nullptr},
        {1, CUSTOM_PORT, CUSTOM_PORT, Hccl::SocketRole::SERVER, true, "AUTOMATIC_SOCKET_TAG_127.0.0.1_127.0.0.2_61001",
         false});
}

// 显式 SERVER：角色覆盖地址推导（本端地址较大仍为 SERVER），仅 SERVER 端监听
TEST_F(AicpuTsUboeUbRtpChannelHelperTest, Ut_BuildSocket_When_ExplicitServer_Expect_ServerListenOnly)
{
    RunBuildSocketAndExpect<AicpuTsUboeChannel>(
        {COMM_PROTOCOL_UBOE, LARGE_IP, SMALL_IP, HCOMM_SOCKET_ROLE_SERVER, CUSTOM_PORT, nullptr},
        {1, CUSTOM_PORT, CUSTOM_PORT, Hccl::SocketRole::SERVER, false, "AUTOMATIC_SOCKET_TAG_127.0.0.2_127.0.0.1_61001",
         false});
    RunBuildSocketAndExpect<AicpuTsUbRtpChannel>(
        {COMM_PROTOCOL_UB_RTP, LARGE_IP, SMALL_IP, HCOMM_SOCKET_ROLE_SERVER, CUSTOM_PORT, nullptr},
        {1, CUSTOM_PORT, CUSTOM_PORT, Hccl::SocketRole::SERVER, false, "AUTOMATIC_SOCKET_TAG_127.0.0.2_127.0.0.1_61001",
         false});
}

// 显式 CLIENT：角色覆盖地址推导（本端地址较小仍为 CLIENT），不创建监听，先收方向
TEST_F(AicpuTsUboeUbRtpChannelHelperTest, Ut_BuildSocket_When_ExplicitClient_Expect_NoListenRecvFirst)
{
    RunBuildSocketAndExpect<AicpuTsUboeChannel>(
        {COMM_PROTOCOL_UBOE, SMALL_IP, LARGE_IP, HCOMM_SOCKET_ROLE_CLIENT, CUSTOM_PORT, nullptr},
        {0, 0, CUSTOM_PORT, Hccl::SocketRole::CLIENT, false, "AUTOMATIC_SOCKET_TAG_127.0.0.2_127.0.0.1_61001", true});
    RunBuildSocketAndExpect<AicpuTsUbRtpChannel>(
        {COMM_PROTOCOL_UB_RTP, SMALL_IP, LARGE_IP, HCOMM_SOCKET_ROLE_CLIENT, CUSTOM_PORT, nullptr},
        {0, 0, CUSTOM_PORT, Hccl::SocketRole::CLIENT, false, "AUTOMATIC_SOCKET_TAG_127.0.0.2_127.0.0.1_61001", true});
}

// 配置 channelName：SetHccpTag 绕过后缀拼接（上游 hccpTag 超长修复 !5779 语义），tag 即 channelName
TEST_F(AicpuTsUboeUbRtpChannelHelperTest, Ut_BuildSocket_When_ChannelNameSet_Expect_HccpTagBypass)
{
    RunBuildSocketAndExpect<AicpuTsUboeChannel>(
        {COMM_PROTOCOL_UBOE, SMALL_IP, LARGE_IP, HCOMM_SOCKET_ROLE_RESERVED, 0, "ut_channel"},
        {1, EXPECTED_DEFAULT_PORT, EXPECTED_DEFAULT_PORT, Hccl::SocketRole::SERVER, true, "ut_channel", false});
    RunBuildSocketAndExpect<AicpuTsUbRtpChannel>(
        {COMM_PROTOCOL_UB_RTP, SMALL_IP, LARGE_IP, HCOMM_SOCKET_ROLE_RESERVED, 0, "ut_channel"},
        {1, EXPECTED_DEFAULT_PORT, EXPECTED_DEFAULT_PORT, Hccl::SocketRole::SERVER, true, "ut_channel", false});
}

// 显式 SERVER + port=0：端口回退默认 60001（显式模式下回退分支此前未覆盖；
// 注意 tag 带 _60001 后缀——显式模式走带端口构造，与自动模式 port=0 的旧构造（无后缀）为有意差异）
TEST_F(AicpuTsUboeUbRtpChannelHelperTest, Ut_BuildSocket_When_ExplicitServerPort0_Expect_FallbackDefaultPort)
{
    RunBuildSocketAndExpect<AicpuTsUboeChannel>(
        {COMM_PROTOCOL_UBOE, LARGE_IP, SMALL_IP, HCOMM_SOCKET_ROLE_SERVER, 0, nullptr},
        {1, EXPECTED_DEFAULT_PORT, EXPECTED_DEFAULT_PORT, Hccl::SocketRole::SERVER, false,
         "AUTOMATIC_SOCKET_TAG_127.0.0.2_127.0.0.1_60001", false});
    RunBuildSocketAndExpect<AicpuTsUbRtpChannel>(
        {COMM_PROTOCOL_UB_RTP, LARGE_IP, SMALL_IP, HCOMM_SOCKET_ROLE_SERVER, 0, nullptr},
        {1, EXPECTED_DEFAULT_PORT, EXPECTED_DEFAULT_PORT, Hccl::SocketRole::SERVER, false,
         "AUTOMATIC_SOCKET_TAG_127.0.0.2_127.0.0.1_60001", false});
}

// 显式 CLIENT + port=0：同样回退默认端口，不创建监听，先收方向（tag 为 CLIENT 格式，带 _60001 后缀）
TEST_F(AicpuTsUboeUbRtpChannelHelperTest, Ut_BuildSocket_When_ExplicitClientPort0_Expect_FallbackNoListen)
{
    RunBuildSocketAndExpect<AicpuTsUboeChannel>(
        {COMM_PROTOCOL_UBOE, SMALL_IP, LARGE_IP, HCOMM_SOCKET_ROLE_CLIENT, 0, nullptr},
        {0, 0, EXPECTED_DEFAULT_PORT, Hccl::SocketRole::CLIENT, false, "AUTOMATIC_SOCKET_TAG_127.0.0.2_127.0.0.1_60001",
         true});
    RunBuildSocketAndExpect<AicpuTsUbRtpChannel>(
        {COMM_PROTOCOL_UB_RTP, SMALL_IP, LARGE_IP, HCOMM_SOCKET_ROLE_CLIENT, 0, nullptr},
        {0, 0, EXPECTED_DEFAULT_PORT, Hccl::SocketRole::CLIENT, false, "AUTOMATIC_SOCKET_TAG_127.0.0.2_127.0.0.1_60001",
         true});
}

// channelName + 自定义端口：SetHccpTag 旁路与带端口后缀路径（BuildHccpTagWithAddrAndPort）组合
// ——旁路优先级更高，tag 即 channelName（不追加 IP/端口后缀），端口本身仍生效
TEST_F(AicpuTsUboeUbRtpChannelHelperTest, Ut_BuildSocket_When_ChannelNameWithCustomPort_Expect_TagBypassWithPort)
{
    RunBuildSocketAndExpect<AicpuTsUboeChannel>(
        {COMM_PROTOCOL_UBOE, SMALL_IP, LARGE_IP, HCOMM_SOCKET_ROLE_RESERVED, CUSTOM_PORT, "ut_channel"},
        {1, CUSTOM_PORT, CUSTOM_PORT, Hccl::SocketRole::SERVER, true, "ut_channel", false});
    RunBuildSocketAndExpect<AicpuTsUbRtpChannel>(
        {COMM_PROTOCOL_UB_RTP, SMALL_IP, LARGE_IP, HCOMM_SOCKET_ROLE_RESERVED, CUSTOM_PORT, "ut_channel"},
        {1, CUSTOM_PORT, CUSTOM_PORT, Hccl::SocketRole::SERVER, true, "ut_channel", false});
}

// 显式 SERVER + GetSocket 失败：返回错误码，socket_ 未被赋值（GetSocket 桩不写出参）
TEST_F(AicpuTsUboeUbRtpChannelHelperTest, Ut_BuildSocket_When_GetSocketFail_Expect_ErrorReturn)
{
    GlobalMockObject::reset();
    MOCKER_CPP(&AicpuTsUboeUbRtpChannelHelper::BuildServerSocket).stubs().will(invoke(StubBuildServerSocket));
    MOCKER_CPP(&SocketMgr::GetSocket).stubs().will(returnValue(HCCL_E_INTERNAL));

    g_buildServerSocketCount = 0;
    HcommChannelDesc desc{};
    desc.role = HCOMM_SOCKET_ROLE_SERVER;
    desc.port = CUSTOM_PORT;

    AicpuTsUboeChannel ch(reinterpret_cast<EndpointHandle>(0x1234), desc);
    ch.localEp_ = MakeEpDesc(COMM_PROTOCOL_UBOE, SMALL_IP);
    ch.remoteEp_ = MakeEpDesc(COMM_PROTOCOL_UBOE, LARGE_IP);
    ch.socket_ = nullptr;
    ch.devicePhyId_ = 0;

    EXPECT_EQ(ch.BuildSocket(), HCCL_E_INTERNAL);
    EXPECT_EQ(g_buildServerSocketCount, 1u); // 显式 SERVER：先建监听再取工作 socket
    EXPECT_EQ(ch.socket_, nullptr);
}
