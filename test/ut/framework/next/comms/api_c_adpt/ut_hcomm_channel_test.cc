/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../ut_hcomm_base.h"
#include <memory>
#include "channel_process.h"
#include "endpoint.h"
#include "hcomm_channel.h"
#include "mockcpp/mockcpp.hpp"
#include "shared_jetty_mgr.h"
#include "../../../../../../src/base_comm/hcomm_res_mgr.h"
#include "../../../../../../src/base_comm/resources/endpoints/mgr/endpoint_mgr.h"

namespace {

class ChannelStubEndpoint final : public hcomm::Endpoint {
public:
    explicit ChannelStubEndpoint(const EndpointDesc& desc) : Endpoint(desc) {}

    HcclResult Init() override { return HCCL_SUCCESS; }
    hcomm::LocalRegedMemMgr* GetLocalRegMemMgr() override { return nullptr; }
    void* GetRdmaHandle() override { return nullptr; }
    bool IsCtxHandleValid() const override { return false; }
};

class ScopedChannelStubEndpoint {
public:
    explicit ScopedChannelStubEndpoint(EndpointLocType locType = ENDPOINT_LOC_TYPE_DEVICE)
    {
        EndpointDesc desc{};
        desc.loc.locType = locType;
        auto ep = std::make_unique<ChannelStubEndpoint>(desc);
        handle_ = reinterpret_cast<EndpointHandle>(ep.get());
        hcomm::HcommResMgr::GetInstance().GetEndpointMgr().Add(handle_, std::move(ep));
    }

    ~ScopedChannelStubEndpoint()
    {
        if (handle_ != nullptr) {
            hcomm::HcommResMgr::GetInstance().GetEndpointMgr().Remove(handle_);
        }
    }

    EndpointHandle Get() const { return handle_; }

private:
    EndpointHandle handle_ = nullptr;
};

} // namespace

class TestHcommChannel : public TestHcommCAdptBase {
public:
    void SetUp() override { TestHcommCAdptBase::SetUp(); }
    void TearDown() override { TestHcommCAdptBase::TearDown(); }

protected:
    // 构造一个已初始化的 HcommChannelDesc，remoteEndpoint.protocol 设为指定协议
    static HcommChannelDesc MakeSharedQueueChannelDesc(CommProtocol protocol)
    {
        HcommChannelDesc desc{};
        HcommChannelDescInit(&desc, 1);
        desc.remoteEndpoint.protocol = protocol;
        return desc;
    }

    // 创建配置对象并设为共享队列模式
    static HcommChannelConfig MakeSharedQueueConfig()
    {
        HcommChannelConfig config = nullptr;
        HcommChannelConfigCreate(&config);
        HcommChannelConfigSetInt(config, HCOMM_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE, 1);
        return config;
    }
};

TEST_F(TestHcommChannel, Ut_TestHcommChannelCreate_When_DescsNullptr_Return_HCCL_E_PTR)
{
    ChannelHandle channels[1];
    HcommResult ret = HcommChannelCreate(nullptr, COMM_ENGINE_AICPU, nullptr, 1, channels);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcommChannel, Ut_TestHcommChannelCreate_When_ChannelsNullptr_Return_HCCL_E_PTR)
{
    HcommChannelDesc desc;
    desc.role = HCOMM_SOCKET_ROLE_SERVER;
    desc.port = 12345;
    HcommResult ret = HcommChannelCreate(nullptr, COMM_ENGINE_AICPU, &desc, 1, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcommChannel, Ut_TestHcommChannelCreate_When_NumZero_Return_HCCL_E_PARA)
{
    HcommChannelDesc desc;
    desc.role = HCOMM_SOCKET_ROLE_SERVER;
    desc.port = 12345;
    ChannelHandle channels[1];
    HcommResult ret = HcommChannelCreate(nullptr, COMM_ENGINE_AICPU, &desc, 0, channels);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcommChannel, Ut_TestHcommChannelCreate_When_CcuUbRtp_Return_NotSupport)
{
    ScopedChannelStubEndpoint stubEndpoint;
    HcommChannelDesc desc = MakeSharedQueueChannelDesc(COMM_PROTOCOL_UB_RTP);

    ChannelHandle channels[1] = {0};
    HcommResult ret = HcommChannelCreate(stubEndpoint.Get(), COMM_ENGINE_CCU, &desc, 1, channels);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(channels[0], static_cast<ChannelHandle>(0));
}

TEST_F(TestHcommChannel, Ut_TestHcommCollectiveChannelCreate_When_ParamsNull_Return_HCCL_E_PTR)
{
    ChannelHandle channels[1];
    HcommResult ret = HcommCollectiveChannelCreate(nullptr, COMM_ENGINE_AICPU, nullptr, 1, channels);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcommChannel, Ut_TestHcommCollectiveChannelCreate_When_NumZero_Return_HCCL_E_PARA)
{
    HcommChannelDesc desc;
    desc.role = HCOMM_SOCKET_ROLE_SERVER;
    desc.port = 12345;
    ChannelHandle channels[1];
    HcommResult ret = HcommCollectiveChannelCreate(nullptr, COMM_ENGINE_AICPU, &desc, 0, channels);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(TestHcommChannel, Ut_TestHcommChannelGet_When_ChannelNullptr_Return_HCCL_E_PTR)
{
    HcommResult ret = HcommChannelGet(0, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcommChannel, Ut_TestHcommChannelGetStatus_When_ListNullptr_Return_HCCL_E_PTR)
{
    int32_t statusList[1] = {-1};
    HcommResult ret = HcommChannelGetStatus(nullptr, 1, statusList);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcommChannel, Ut_TestHcommChannelGetStatus_When_StatusListNullptr_Return_HCCL_E_PTR)
{
    ChannelHandle channelList[1] = {0};
    HcommResult ret = HcommChannelGetStatus(channelList, 1, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcommChannel, Ut_TestHcommChannelGetStatus_When_NumZero_Return_HCCL_E_PARA)
{
    ChannelHandle channelList[1] = {0};
    int32_t statusList[1] = {-1};
    HcommResult ret = HcommChannelGetStatus(channelList, 0, statusList);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(TestHcommChannel, Ut_TestHcommChannelGetNotifyNum_When_NumNullptr_Return_HCCL_E_PTR)
{
    HcommResult ret = HcommChannelGetNotifyNum(0, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcommChannel, Ut_TestHcommChannelDestroy_When_ChannelsNullptr_Return_HCCL_E_PTR)
{
    HcommResult ret = HcommChannelDestroy(nullptr, 1);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcommChannel, Ut_TestHcommChannelDestroy_When_NumZero_Return_HCCL_E_PARA)
{
    ChannelHandle channels[1] = {0};
    HcommResult ret = HcommChannelDestroy(channels, 0);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(TestHcommChannel, Ut_TestHcommChannelGetRemoteMems_When_MemNullptr_Return_HCCL_E_PTR)
{
    uint32_t memNum = 0;
    char** memInfos = nullptr;
    HcommResult ret = HcommChannelGetRemoteMems(0, &memNum, nullptr, &memInfos);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcommChannel, Ut_TestHcommChannelGetRemoteMems_When_NumNullptr_Return_HCCL_E_PTR)
{
    CommMem* remoteMem = nullptr;
    char** memInfos = nullptr;
    HcommResult ret = HcommChannelGetRemoteMems(0, nullptr, &remoteMem, &memInfos);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// HcommChannelGetRemoteMems 当 memInfos 为 nullptr 返回 HCOMM_E_PTR
TEST_F(TestHcommChannel, Ut_TestHcommChannelGetRemoteMems_When_MemInfosNullptr_Return_HCCL_E_PTR)
{
    uint32_t memNum = 0;
    CommMem* remoteMem = nullptr;
    HcommResult ret = HcommChannelGetRemoteMems(0, &memNum, &remoteMem, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// HcommChannelGetRemoteMems 当 ChannelProcess 返回错误时透传错误码
TEST_F(TestHcommChannel, Ut_TestHcommChannelGetRemoteMems_When_ChannelProcessError_Expect_ErrorPropagated)
{
    ChannelHandle channelHandle = 0xFFFF;
    CommMem* remoteMem = nullptr;
    uint32_t memNum = 0;
    char* memInfosStorage[2] = {nullptr, nullptr};
    char** memInfos = memInfosStorage;
    MOCKER(&hcomm::ChannelProcess::ChannelGetRemoteMems)
        .stubs()
        .will(returnValue(static_cast<HcclResult>(HCCL_E_NOT_FOUND)));
    HcommResult ret = HcommChannelGetRemoteMems(channelHandle, &memNum, &remoteMem, &memInfos);
    EXPECT_EQ(ret, static_cast<HcommResult>(HCCL_E_NOT_FOUND));
}

// ===================== HcommChannelCreate AIV+UB_MEM API UT =====================

// HcommChannelCreate API 级 AIV+UB_MEM 创建成功（自创建 Socket 路径）
TEST_F(TestHcommChannel, Ut_HcommChannelCreate_When_AivUbMem_Expect_ReturnSuccess)
{
    ScopedChannelStubEndpoint stubEndpoint;
    EndpointHandle ep = stubEndpoint.Get();
    HcommChannelDesc desc{};
    ASSERT_EQ(HcommChannelDescInit(&desc, 1), HCCL_SUCCESS);
    desc.remoteEndpoint.protocol = COMM_PROTOCOL_UB_MEM;
    desc.ubMemAttr.pathMode = 0;
    desc.socket = nullptr;
    desc.role = HCOMM_SOCKET_ROLE_SERVER;
    desc.port = 60001;
    ChannelHandle channels[1] = {0};

    MOCKER(HcommResMgrInit).stubs().will(returnValue(static_cast<HcommResult>(HCCL_SUCCESS)));
    MOCKER(&hcomm::ChannelProcess::CreateChannelsLoop).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(&hcomm::ChannelProcess::PrepareUserChannels).stubs().will(returnValue(HCCL_SUCCESS));

    HcommResult ret = HcommChannelCreate(ep, COMM_ENGINE_AIV, &desc, 1, channels);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    hcomm::SharedJettyMgr::GetInstance().UnregisterEndpoint(ep);
}

// HcommChannelCreate AIV+UB_MEM + 非法 pathMode 在 Normalize 阶段被拦截
TEST_F(TestHcommChannel, Ut_HcommChannelCreate_When_AivUbMemInvalidPathMode_Expect_ReturnEPARA)
{
    ScopedChannelStubEndpoint stubEndpoint;
    EndpointHandle ep = stubEndpoint.Get();
    HcommChannelDesc desc{};
    ASSERT_EQ(HcommChannelDescInit(&desc, 1), HCCL_SUCCESS);
    desc.remoteEndpoint.protocol = COMM_PROTOCOL_UB_MEM;
    desc.ubMemAttr.pathMode = 5; // 非法值
    ChannelHandle channels[1] = {0};

    HcommResult ret = HcommChannelCreate(ep, COMM_ENGINE_AIV, &desc, 1, channels);
    EXPECT_EQ(ret, HCCL_E_PARA);
    hcomm::SharedJettyMgr::GetInstance().UnregisterEndpoint(ep);
}

// HcommChannelCreate 当 channelDescs 为 nullptr 返回 HCOMM_E_PTR
TEST_F(TestHcommChannel, Ut_HcommChannelCreate_When_ChannelDescsNullptr_Expect_ReturnEPtr)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x12345);
    ChannelHandle channels[1] = {0};
    HcommResult ret = HcommChannelCreate(ep, COMM_ENGINE_AIV, nullptr, 1, channels);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// base_comm 层 IsUbProtocol 门控覆盖：
// HcommChannelCreateWithConfig → ValidateSharedQueueConfig → IsUbProtocol

// 正向：UB_RTP 协议 + IS_SHARED_QUEUE=true，门控放行后由 CreateChannelsLoop mock 拦截
TEST_F(TestHcommChannel, Ut_HcommChannelCreateWithConfig_When_SharedQueue_UbRtp_Expect_Success)
{
    HcommChannelConfig config = MakeSharedQueueConfig();
    HcommChannelDesc desc = MakeSharedQueueChannelDesc(COMM_PROTOCOL_UB_RTP);
    ChannelHandle channels[1] = {0};
    ScopedChannelStubEndpoint stubEndpoint;
    EndpointHandle ep = stubEndpoint.Get();

    MOCKER(HcommResMgrInit).stubs().will(returnValue(static_cast<HcommResult>(HCCL_SUCCESS)));
    MOCKER(&hcomm::ChannelProcess::CreateChannelsLoop).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(&hcomm::ChannelProcess::PrepareUserChannels).stubs().will(returnValue(HCCL_SUCCESS));

    HcommResult ret = HcommChannelCreateWithConfig(ep, COMM_ENGINE_AIV, &desc, 1, config, channels);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    hcomm::SharedJettyMgr::GetInstance().UnregisterEndpoint(ep);
    HcommChannelConfigDestroy(config);
}

// 反向：RoCE 协议 + IS_SHARED_QUEUE=true，ValidateSharedQueueConfig 拦截
TEST_F(TestHcommChannel, Ut_HcommChannelCreateWithConfig_When_SharedQueue_Roce_Expect_NotSupport)
{
    HcommChannelConfig config = MakeSharedQueueConfig();
    HcommChannelDesc desc = MakeSharedQueueChannelDesc(COMM_PROTOCOL_ROCE);
    ChannelHandle channels[1] = {0};
    ScopedChannelStubEndpoint stubEndpoint;
    EndpointHandle ep = stubEndpoint.Get();

    MOCKER(HcommResMgrInit).stubs().will(returnValue(static_cast<HcommResult>(HCCL_SUCCESS)));

    HcommResult ret = HcommChannelCreateWithConfig(ep, COMM_ENGINE_AIV, &desc, 1, config, channels);
    EXPECT_EQ(ret, static_cast<HcommResult>(HCCL_E_NOT_SUPPORT));
    HcommChannelConfigDestroy(config);
}

// 反向：UBOE 协议 + IS_SHARED_QUEUE=true，ValidateSharedQueueConfig 拦截
TEST_F(TestHcommChannel, Ut_HcommChannelCreateWithConfig_When_SharedQueue_Uboe_Expect_NotSupport)
{
    HcommChannelConfig config = MakeSharedQueueConfig();
    HcommChannelDesc desc = MakeSharedQueueChannelDesc(COMM_PROTOCOL_UBOE);
    ChannelHandle channels[1] = {0};
    ScopedChannelStubEndpoint stubEndpoint;
    EndpointHandle ep = stubEndpoint.Get();

    MOCKER(HcommResMgrInit).stubs().will(returnValue(static_cast<HcommResult>(HCCL_SUCCESS)));

    HcommResult ret = HcommChannelCreateWithConfig(ep, COMM_ENGINE_AIV, &desc, 1, config, channels);
    EXPECT_EQ(ret, static_cast<HcommResult>(HCCL_E_NOT_SUPPORT));
    HcommChannelConfigDestroy(config);
}
