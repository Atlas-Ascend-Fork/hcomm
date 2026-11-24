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
#include <mockcpp/mockcpp.hpp>
#include <cerrno>
#include <cstring>
#include <sys/socket.h>
#include <sys/epoll.h>
#include <unistd.h>
#include <functional>
#include "hccp.h"

#define private public
#include "base_config_legacy.h"
#include "env_config/env_config_v2.h"
#include "rank_info_dispatcher.h"
#include "socket.h"
#include "rank_info_detect_service.h"
#include "rank_info_detect_client.h"
#undef private

using namespace std;
using namespace Hccl;

extern "C" int RaCtlEventHandle(int eventHandle, const void* fdHandle, int opcode, enum RaEpollEvent event);

namespace {
std::shared_ptr<Socket> MakeDeviceSocket()
{
    IpAddress remoteIp;
    IpAddress localIp;
    SocketHandle socketHandle{};
    return std::make_shared<Socket>(
        socketHandle, localIp, 0, remoteIp, "test", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
}
} // namespace

class RankInfoDispatcherTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RankInfoDispatcherTest SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "RankInfoDispatcherTest TearDown" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp() { std::cout << "A Test SetUP" << std::endl; }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test TearDown" << std::endl;
    }
};

TEST_F(RankInfoDispatcherTest, Ut_ProcessOneSendEvent_When_Input_Right_Expect_Send_Ok)
{
    // when
    MOCKER_CPP(&RankInfoDispatcher::SendState::Send).stubs().with(mockcpp::any()).will(returnValue(true));
    MOCKER_CPP(&RankInfoDispatcher::SendState::IsOk).stubs().with(mockcpp::any()).will(returnValue(true));
    MOCKER(RaCtlEventHandle)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(0));

    // then
    auto socket = MakeDeviceSocket();
    RankInfoDetectService topoServer(0, socket, "test", {});
    RankInfoDispatcher workers(&topoServer);
    RankInfoDispatcher::SendState txS;
    RankInfoDispatcher::FdContext fdCtx;
    fdCtx.txState = txS;
    fdCtx.socket = socket;
    workers.fdHandleToFdContextMap_.emplace(socket->fdHandle, fdCtx);

    // check
    EXPECT_NO_THROW(workers.ProcessOneSendEvent(1, socket->fdHandle));
    EXPECT_EQ(workers.stop_, false);
}

TEST_F(RankInfoDispatcherTest, Ut_ProcessOneSendEvent_When_Input_Error_Expect_Stop_True)
{
    // when
    auto socket = MakeDeviceSocket();
    RankInfoDetectService topoServer(0, socket, "test", {});
    RankInfoDispatcher workers(&topoServer);

    // check
    EXPECT_NO_THROW(workers.ProcessOneSendEvent(1, socket->fdHandle));
    EXPECT_EQ(workers.stop_, true);

    // when
    MOCKER_CPP(&RankInfoDispatcher::SendState::Send)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(false))
        .then(returnValue(true));
    RankInfoDispatcher::SendState txS;
    RankInfoDispatcher::FdContext fdCtx;
    fdCtx.txState = txS;
    fdCtx.socket = socket;
    workers.fdHandleToFdContextMap_.emplace(socket->fdHandle, fdCtx);

    // check
    EXPECT_NO_THROW(workers.ProcessOneSendEvent(1, socket->fdHandle));
    EXPECT_EQ(workers.stop_, true);

    // when
    MOCKER_CPP(&RankInfoDispatcher::SendState::IsOk).stubs().with(mockcpp::any()).will(returnValue(false));

    // check
    EXPECT_NO_THROW(workers.ProcessOneSendEvent(1, socket->fdHandle));
    EXPECT_EQ(workers.stop_, true);
}

TEST_F(RankInfoDispatcherTest, Ut_SendHeader_When_ISend_Ok_Expect_Return_True)
{
    // when
    MOCKER_CPP(&Hccl::Socket::ISend)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(+[](Hccl::Socket*, const void*, u64 size, u64& sentSize) -> HcclResult {
            sentSize = size;
            return HCCL_SUCCESS;
        }));

    // then
    auto socket = MakeDeviceSocket();
    RankInfoDetectService topoServer(0, socket, "test", {});
    RankInfoDispatcher workers(&topoServer);

    // check
    RankInfoDispatcher::SendState txS;
    char* buf;
    size_t dataLen = 100;
    size_t sendedLen = 80;
    EXPECT_EQ(txS.SendHelper(socket, buf, dataLen, sendedLen), true);
}

TEST_F(RankInfoDispatcherTest, Ut_SendHeader_When_ISend_False_Expect_Return_False)
{
    // when
    MOCKER_CPP(&Hccl::Socket::ISend)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_NETWORK));

    // then
    auto socket = MakeDeviceSocket();
    RankInfoDetectService topoServer(0, socket, "test", {});
    RankInfoDispatcher workers(&topoServer);

    // check
    RankInfoDispatcher::SendState txS;
    char* buf;
    size_t dataLen = 100;
    size_t sendedLen = 80;
    EXPECT_EQ(txS.SendHelper(socket, buf, dataLen, sendedLen), false);
}

// SendHelper: ISend 返回 E_INTERNAL（SOCK_CLOSE，hccp 进程异常退出）时判定发送失败
// 快速失败——失败判定口径 `!=SUCCESS && !=E_AGAIN` 的第 3 态覆盖
TEST_F(RankInfoDispatcherTest, Ut_SendHelper_When_ISendReturnInternal_Expect_False)
{
    // when
    MOCKER_CPP(&Hccl::Socket::ISend)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_INTERNAL));

    // then
    auto socket = MakeDeviceSocket();
    RankInfoDetectService topoServer(0, socket, "test", {});
    RankInfoDispatcher workers(&topoServer);

    // check
    RankInfoDispatcher::SendState txS;
    char* buf;
    size_t dataLen = 100;
    size_t sendedLen = 80;
    EXPECT_EQ(txS.SendHelper(socket, buf, dataLen, sendedLen), false);
}

TEST_F(RankInfoDispatcherTest, Ut_SendState_Send_When_ISend_Ok_Expect_SendHeader)
{
    // when
    MOCKER_CPP(&Hccl::Socket::ISend)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(+[](Hccl::Socket*, const void*, u64 size, u64& sentSize) -> HcclResult {
            sentSize = size;
            return HCCL_SUCCESS;
        }));

    // then
    auto socket = MakeDeviceSocket();
    RankInfoDetectService topoServer(0, socket, "test", {});
    RankInfoDispatcher workers(&topoServer);

    // check
    RankInfoDispatcher::SendState txS;
    EXPECT_EQ(txS.Send(socket), true);
}

TEST_F(RankInfoDispatcherTest, Ut_SendState_Send_When_ISend_Ok_Expect_SendBody)
{
    // when
    MOCKER_CPP(&Hccl::Socket::ISend)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(+[](Hccl::Socket*, const void*, u64 size, u64& sentSize) -> HcclResult {
            sentSize = size;
            return HCCL_SUCCESS;
        }));

    // then
    auto socket = MakeDeviceSocket();
    RankInfoDetectService topoServer(0, socket, "test", {});
    RankInfoDispatcher workers(&topoServer);

    // check
    RankInfoDispatcher::SendState txS;
    txS.headerSent = 1;
    txS.headerLen = 1;
    txS.bodyLen = 1;
    txS.bodySent = 0;
    EXPECT_EQ(txS.Send(socket), true);
}

TEST_F(RankInfoDispatcherTest, Ut_SendState_Send_When_ISend_False_Expect_Return_False)
{
    // when
    MOCKER_CPP(&Hccl::Socket::ISend)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_NETWORK));

    // when
    auto socket = MakeDeviceSocket();
    RankInfoDetectService topoServer(0, socket, "test", {});
    RankInfoDispatcher workers(&topoServer);

    // check
    RankInfoDispatcher::SendState txS;
    EXPECT_EQ(txS.Send(socket), false);
}

TEST_F(RankInfoDispatcherTest, Ut_ProcessSend_When_Send_Again_Expect_Return_TimeOut)
{
    // when
    MOCKER_CPP(&Hccl::Socket::ISend)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_AGAIN));
    MOCKER_CPP(&RankInfoDispatcher::SendOnce).stubs().with().will(returnValue(HcclResult::HCCL_SUCCESS));
    u32 eventsNum1 = 1;
    MOCKER(HrtRaWaitEventHandle)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), outBound(eventsNum1))
        .will(returnValue(HCCL_SUCCESS));

    EnvSocketConfig envConfig;
    EnvSocketConfig& fakeEnvConfig = envConfig;
    fakeEnvConfig.linkTimeOut = CfgField<s32>{"HCCL_CONNECT_TIMEOUT", s32(1), Str2T<s32>};
    fakeEnvConfig.linkTimeOut.isParsed = true;
    MOCKER_CPP(&EnvConfig::GetSocketConfig).stubs().will(returnValue(fakeEnvConfig));

    // then
    auto socket = MakeDeviceSocket();
    RankInfoDetectService topoServer(0, socket, "test", {});
    RankInfoDispatcher workers(&topoServer);

    // check
    workers.rankNum_ = 1;
    EXPECT_EQ(workers.ProcessSend(), HcclResult::HCCL_E_TIMEOUT);
}

TEST_F(RankInfoDispatcherTest, Ut_ProcessSend_When_EventsNum_Error_Expect_E_PARA)
{
    // when
    MOCKER_CPP(&Hccl::Socket::ISend)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_NETWORK));
    MOCKER_CPP(&RankInfoDispatcher::SendOnce).stubs().with().will(returnValue(HcclResult::HCCL_SUCCESS));
    u32 eventsNum = 0;
    MOCKER(HrtRaWaitEventHandle)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), outBound(eventsNum))
        .will(returnValue(HCCL_E_NETWORK));

    // then
    auto socket = MakeDeviceSocket();
    RankInfoDetectService topoServer(0, socket, "test", {});
    RankInfoDispatcher workers(&topoServer);

    // check
    workers.rankNum_ = 1;
    EXPECT_EQ(workers.ProcessSend(), HcclResult::HCCL_E_PARA);
}

TEST_F(RankInfoDispatcherTest, Ut_SendOnce_When_InputValue_Expect_NO_THROW)
{
    // when
    MOCKER_CPP(&RankInfoDispatcher::SendState::Send).stubs().with(mockcpp::any()).will(returnValue(true));
    MOCKER_CPP(&RankInfoDispatcher::SendState::IsOk).stubs().with(mockcpp::any()).will(returnValue(true));

    // then
    auto socket = MakeDeviceSocket();
    RankInfoDetectService topoServer(0, socket, "test", {});
    RankInfoDispatcher workers(&topoServer);
    RankInfoDispatcher::SendState txS;
    RankInfoDispatcher::FdContext fdCtx;
    fdCtx.txState = txS;
    fdCtx.socket = socket;
    workers.fdHandleToFdContextMap_.emplace(socket->fdHandle, fdCtx);

    EXPECT_EQ(workers.SendOnce(), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(workers.sendDoneCount_, 1);
}

TEST_F(RankInfoDispatcherTest, Ut_SendOnce_When_Input_Error_Expect_E_PARA)
{
    // when
    MOCKER_CPP(&RankInfoDispatcher::SendState::Send)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(false))
        .then(returnValue(true));
    MOCKER_CPP(&RankInfoDispatcher::SendState::IsOk).stubs().with(mockcpp::any()).will(returnValue(false));
    MOCKER(RaCtlEventHandle)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(0));

    // then
    auto socket = MakeDeviceSocket();
    RankInfoDetectService topoServer(0, socket, "test", {});
    RankInfoDispatcher workers(&topoServer);
    RankInfoDispatcher::SendState txS;
    RankInfoDispatcher::FdContext fdCtx;
    fdCtx.txState = txS;
    fdCtx.socket = socket;
    workers.fdHandleToFdContextMap_.emplace(socket->fdHandle, fdCtx);

    EXPECT_EQ(workers.SendOnce(), HcclResult::HCCL_E_PARA);
    EXPECT_EQ(workers.SendOnce(), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(workers.sendDoneCount_, 0);
}

TEST_F(RankInfoDispatcherTest, Ut_CleanResource_When_Input_Expect_NO_THROW)
{
    // when
    MOCKER_CPP(&RankInfoDispatcher::WakeWoker).stubs().with().will(ignoreReturnValue());

    // then
    auto socket = MakeDeviceSocket();
    RankInfoDetectService topoServer(0, socket, "test", {});
    RankInfoDispatcher workers(&topoServer);

    // check
    EXPECT_NO_THROW(workers.CleanResource());
    EXPECT_EQ(workers.stop_, true);
}

TEST_F(RankInfoDispatcherTest, Ut_PrepareResource_When_Input_Expect_NO_THROW)
{
    // when
    MOCKER_CPP(&RankInfoDispatcher::InitWorkerThread).stubs().with().will(ignoreReturnValue());

    // then
    auto socket = MakeDeviceSocket();
    RankInfoDetectService topoServer(0, socket, "test", {});
    RankInfoDispatcher workers(&topoServer);

    // check
    std::unordered_map<std::string, std::shared_ptr<Socket>> connectSockets;
    connectSockets["0"] = socket;
    std::string failedAgentIdList;
    RankTableInfo clusterInfo;
    RankInfoDetectClient client(0, 1, 0, socket);
    client.ConstructSingleRank(clusterInfo);
    EXPECT_EQ(workers.PrepareResource(connectSockets, clusterInfo, failedAgentIdList, 0), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(workers.fdHandleToFdContextMap_.size(), 1);
}

TEST_F(RankInfoDispatcherTest, Ut_PrepareResource_BroadcastsNetLayerCheckResult)
{
    MOCKER_CPP(&RankInfoDispatcher::InitWorkerThread).stubs().will(ignoreReturnValue());
    MOCKER(RaCreateEventHandle).stubs().will(returnValue(0));
    MOCKER(RaDestroyEventHandle).stubs().will(returnValue(0));
    auto socket = MakeDeviceSocket();
    RankInfoDetectService server(0, socket, "test", {});
    RankTableInfo clusterInfo;
    RankInfoDetectClient client(0, 1, 0, socket);
    client.ConstructSingleRank(clusterInfo);
    auto secondRank = clusterInfo.ranks.front();
    secondRank.rankId = 1;
    clusterInfo.ranks.push_back(secondRank);
    clusterInfo.rankCount = 2;
    server.rankTable_ = clusterInfo;
    server.SetRootRankId(0);

    NetLayerCheckInfo selected;
    selected.version = NetLayerCheckInfo::VERSION;
    selected.explicitSelection = 1;
    selected.layers = 0x6;
    const struct {
        bool explicitSelection;
        HcclResult result;
    } cases[] = {
        {false, HCCL_SUCCESS}, {false, HCCL_E_PARA}, {false, HCCL_E_NOT_SUPPORT},
        {true, HCCL_SUCCESS},  {true, HCCL_E_PARA},  {true, HCCL_E_NOT_SUPPORT},
    };
    for (const auto& item : cases) {
        SCOPED_TRACE(testing::Message() << item.explicitSelection << ":" << item.result);
        // root 自身未配置时，也必须广播其他 rank 的显式选层结果或任一端失败。
        server.netLayerCheckInfos_[0] = selected;
        server.netLayerCheckInfos_[0].explicitSelection = 0;
        server.netLayerCheckInfos_[1] = selected;
        server.netLayerCheckInfos_[1].explicitSelection = item.explicitSelection;
        if (!item.explicitSelection) {
            server.netLayerCheckInfos_[1].result = item.result;
        } else if (item.result == HCCL_E_PARA) {
            server.netLayerCheckInfos_[1].layers = 0x3;
        } else if (item.result == HCCL_E_NOT_SUPPORT) {
            server.netLayerCheckInfos_[1].version = 0;
        }
        RankInfoDispatcher workers(&server);
        std::unordered_map<std::string, std::shared_ptr<Socket>> sockets{{"0", socket}};
        ASSERT_EQ(workers.PrepareResource(sockets, clusterInfo, "", 1), HCCL_SUCCESS);

        // 读取真实发送缓冲区，确认尾部结果没有破坏既有消息字段。
        BinaryStream stream(workers.rankTableMsg_);
        RankTableInfo receivedTable(stream);
        u32 step = 0;
        u32 receivedRootRankId = UINT32_MAX;
        std::string failedRanks;
        stream >> step >> receivedRootRankId >> failedRanks;
        NetLayerCheckInfo received;
        ASSERT_EQ(received.Deserialize(stream), HCCL_SUCCESS);
        EXPECT_EQ(receivedTable.rankCount, 2);
        EXPECT_EQ(step, 1);
        EXPECT_EQ(receivedRootRankId, 0U);
        EXPECT_TRUE(failedRanks.empty());
        const bool omitted = !item.explicitSelection && item.result == HCCL_SUCCESS;
        EXPECT_EQ(received.version, omitted ? 0U : static_cast<u32>(NetLayerCheckInfo::VERSION));
        EXPECT_EQ(received.explicitSelection, static_cast<u32>(item.explicitSelection));
        EXPECT_EQ(received.result, item.result);
        if (item.explicitSelection && item.result == HCCL_SUCCESS) {
            EXPECT_EQ(received.layers, 0x6);
        }
        const auto& tx = workers.fdHandleToFdContextMap_.begin()->second.txState;
        EXPECT_EQ(tx.bodyLen, workers.rankTableMsg_.size());
        EXPECT_EQ(tx.data, workers.rankTableMsg_.data());
    }
}

TEST_F(RankInfoDispatcherTest, Ut_PrepareResource_When_Input_Expect_E_PARA)
{
    // when
    MOCKER_CPP(&RankInfoDispatcher::InitWorkerThread).stubs().with().will(ignoreReturnValue());

    // then
    auto socket = MakeDeviceSocket();
    RankInfoDetectService topoServer(0, socket, "test", {});
    RankInfoDispatcher workers(&topoServer);

    // check
    std::unordered_map<std::string, std::shared_ptr<Socket>> connectSockets;
    connectSockets["abc"] = socket;
    RankTableInfo clusterInfo;
    RankInfoDetectClient client(0, 1, 0, socket);
    client.ConstructSingleRank(clusterInfo);
    std::string failedAgentIdList;
    EXPECT_EQ(workers.PrepareResource(connectSockets, clusterInfo, failedAgentIdList, 0), HcclResult::HCCL_E_PARA);
    EXPECT_EQ(workers.fdHandleToFdContextMap_.size(), 0);
}

TEST_F(RankInfoDispatcherTest, Ut_GetTask_When_Input_Null_Expect_Return_false)
{
    // then
    auto socket = MakeDeviceSocket();
    RankInfoDetectService topoServer(0, socket, "test", {});
    RankInfoDispatcher workers(&topoServer);

    // check
    RankInfoDispatcher::WorkerTask workTask;
    EXPECT_EQ(workers.GetTask(workTask), false);
}

TEST_F(RankInfoDispatcherTest, Ut_InitWorkerThread_When_Input_Expect_Return_NO_THROW)
{
    // then
    auto socket = MakeDeviceSocket();
    RankInfoDetectService topoServer(0, socket, "test", {});
    RankInfoDispatcher workers(&topoServer);

    // check
    workers.rankNum_ = 1;
    EXPECT_NO_THROW(workers.InitWorkerThread());
}

TEST_F(RankInfoDispatcherTest, Ut_WorkerWait_When_Input_Expect_Return_NO_THROW)
{
    // then
    auto socket = MakeDeviceSocket();
    RankInfoDetectService topoServer(0, socket, "test", {});
    RankInfoDispatcher workers(&topoServer);

    // check
    workers.ready_ = true;
    EXPECT_NO_THROW(workers.WorkerWait(0));
}

TEST_F(RankInfoDispatcherTest, Ut_BroadcastRankTable_When_Input_Expect_NO_THROW)
{
    // when
    MOCKER_CPP(&RankInfoDispatcher::PrepareResource).stubs().with().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&RankInfoDispatcher::ProcessSend).stubs().with().will(returnValue(HcclResult::HCCL_SUCCESS));

    // then
    auto socket = MakeDeviceSocket();
    RankInfoDetectService topoServer(0, socket, "test", {});
    RankInfoDispatcher workers(&topoServer);

    // check
    std::unordered_map<std::string, std::shared_ptr<Socket>> connectSockets;
    connectSockets["0"] = socket;
    std::string failedAgentIdList;
    RankTableInfo clusterInfo;
    RankInfoDetectClient client(0, 1, 0, socket);
    client.ConstructSingleRank(clusterInfo);
    EXPECT_EQ(workers.BroadcastRankTable(connectSockets, clusterInfo, failedAgentIdList, 0), HcclResult::HCCL_SUCCESS);
}
