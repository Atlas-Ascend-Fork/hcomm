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
#include "ub_mem_endpoint.h"
#include "hcomm_res.h"
#include "hcomm_c_adpt.h"
#include "ip_address.h"
#include "hccp.h"
#include "buffer.h"
#include "network_api_exception.h"
#include "endpoint.h"
#include "adapter_rts.h"
#include "server_socket_manager.h"
#include "opbase_adpt.h"
#include "eid_info_mgr.h"
#include "hccp_ctx.h"
#include "hcomm_adapter_hccp.h"

using namespace hcomm;

class AicpuUbRtpEndpointTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AicpuUbRtpEndpointTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "AicpuUbRtpEndpointTest tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in AicpuUbRtpEndpointTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
        std::cout << "A Test case in AicpuUbRtpEndpointTest TearDown" << std::endl;
    }

    void CreateEndpointDesc(EndpointDesc& endpointDesc, const std::string& ip = "1.0.0.0")
    {
        Hccl::IpAddress localIp(ip);
        endpointDesc.protocol = COMM_PROTOCOL_UB_RTP;
        endpointDesc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
        endpointDesc.commAddr.addr = localIp.GetBinaryAddress().addr;
        endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    }

    HcommResult CreateEndpoint(EndpointHandle& endpointHandle, const std::string& ip = "1.0.0.0")
    {
        EndpointDesc endpointDesc;
        CreateEndpointDesc(endpointDesc, ip);
        return HcommEndpointCreate(&endpointDesc, &endpointHandle);
    }

    CommMem CreateCommMem(void* addr, size_t size, CommMemType type)
    {
        CommMem mem;
        mem.type = type;
        mem.size = size;
        mem.addr = addr;
        return mem;
    }
};

static HcclResult stub_hrtGetDeviceType_950(DevType& devType)
{
    devType = DevType::DEV_TYPE_950;
    return HCCL_SUCCESS;
}

static HcclResult stub_hrtGetDeviceType_910B(DevType& devType)
{
    devType = DevType::DEV_TYPE_910B;
    return HCCL_SUCCESS;
}

static int g_notifyCount = 0;

static int StubRaCtxNotifyEvent(void*, CtxNotifyEvent*)
{
    g_notifyCount++;
    return 0;
}

static HcclResult StubHrtGetDevicePhyIdByIndex3(uint32_t deviceLogicId, u32& devPhyId)
{
    (void)deviceLogicId;
    devPhyId = 3U;
    return HCCL_SUCCESS;
}

static int StubMatchEidInfoList(struct RaInfo, struct HccpDevEidInfo infoList[], unsigned int* num)
{
    if (*num < 1U) {
        return -1;
    }
    // 与用例中 endpoint 的 commAddr（EID raw[15]=1）保持一致，EID 转 IP 后归一化可命中
    infoList[0].eid.raw[15] = 1U;
    infoList[0].eidIndex = 0U;
    infoList[0].dieId = 0U;
    *num = 1U;
    return 0;
}

TEST_F(AicpuUbRtpEndpointTest, Ut_HcommEndpointCreate_When_UbRtp_Device_Expect_Return_SUCCESS)
{
    MOCKER(hrtGetDeviceType).stubs().will(invoke(stub_hrtGetDeviceType_950));

    EndpointDesc endpointDesc;
    CreateEndpointDesc(endpointDesc);
    EndpointHandle endpointHandle;

    HcommResult ret = HcommEndpointCreate(&endpointDesc, &endpointHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(AicpuUbRtpEndpointTest, Ut_CreateBuiltinEndpoint_When_DeviceIdInvalid_Expect_SkipNotifyAndCreateSuccess)
{
    MOCKER(hrtGetDeviceType).stubs().will(invoke(stub_hrtGetDeviceType_950));
    MOCKER(HcclGetThreadDeviceId).stubs().will(returnValue(-1));

    g_notifyCount = 0;
    MOCKER(RaHasCapability).stubs().will(returnValue(true));
    MOCKER(RaCtxNotifyEvent).stubs().will(invoke(StubRaCtxNotifyEvent));

    HccpResetDfxSnapshot(0);
    EndpointDesc endpointDesc;
    CreateEndpointDesc(endpointDesc);
    EndpointHandle endpointHandle = nullptr;
    EXPECT_EQ(HcommEndpointCreate(&endpointDesc, &endpointHandle), HCCL_SUCCESS);
    EXPECT_EQ(g_notifyCount, 0);
    HccpResetDfxSnapshot(0);
}

TEST_F(AicpuUbRtpEndpointTest, Ut_CreateBuiltinEndpoint_When_EidMatched_Expect_NotifyOnce)
{
    // 使用独立 devPhyId=3 的 EidInfoMgr 实例，避免其他用例空缓存污染
    MOCKER(hrtGetDeviceType).stubs().will(invoke(stub_hrtGetDeviceType_950));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().will(invoke(StubHrtGetDevicePhyIdByIndex3));
    uint32_t eidNum = 1U;
    MOCKER(RaGetDevEidInfoNum).stubs().with(mockcpp::any(), outBoundP(&eidNum)).will(returnValue(0));
    MOCKER(RaGetDevEidInfoList).stubs().will(invoke(StubMatchEidInfoList));

    g_notifyCount = 0;
    MOCKER(RaHasCapability).stubs().will(returnValue(true));
    MOCKER(RaCtxNotifyEvent).expects(exactly(1)).will(invoke(StubRaCtxNotifyEvent));

    HccpResetDfxSnapshot(3);
    EndpointDesc endpointDesc;
    CreateEndpointDesc(endpointDesc);
    endpointDesc.loc.device.devPhyId = 3U; // 与 hrtGetDevicePhyIdByIndex 桩一致
    // 真实场景本 endpoint 的 commAddr 为 EID 类型：与 StubMatchEidInfoList 的 EID（raw[15]=1）一致
    endpointDesc.commAddr.type = COMM_ADDR_TYPE_EID;
    memset(endpointDesc.commAddr.eid, 0, sizeof(endpointDesc.commAddr.eid));
    endpointDesc.commAddr.eid[15] = 1U;
    EndpointHandle endpointHandle = nullptr;
    EXPECT_EQ(HcommEndpointCreate(&endpointDesc, &endpointHandle), HCCL_SUCCESS);
    // 匹配成功：仅对本 endpoint 所属 die 下发一次 start
    EXPECT_EQ(g_notifyCount, 1);
    HccpResetDfxSnapshot(3);
}

TEST_F(AicpuUbRtpEndpointTest, Ut_HcommEndpointCreate_When_UbRtp_On_910B_Expect_Return_NOT_SUPPORT)
{
    MOCKER(hrtGetDeviceType).stubs().will(invoke(stub_hrtGetDeviceType_910B));

    EndpointDesc endpointDesc;
    CreateEndpointDesc(endpointDesc);
    EndpointHandle endpointHandle;

    HcommResult ret = HcommEndpointCreate(&endpointDesc, &endpointHandle);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(AicpuUbRtpEndpointTest, Ut_HcommEndpointCreate_When_UbCtp_On_910B_Expect_Return_NOT_SUPPORT)
{
    MOCKER(hrtGetDeviceType).stubs().will(invoke(stub_hrtGetDeviceType_910B));

    EndpointDesc endpointDesc;
    CreateEndpointDesc(endpointDesc);
    endpointDesc.protocol = COMM_PROTOCOL_UB_CTP;
    EndpointHandle endpointHandle;

    HcommResult ret = HcommEndpointCreate(&endpointDesc, &endpointHandle);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(AicpuUbRtpEndpointTest, Ut_HcommEndpointCreate_When_Hccs_On_950_Expect_Return_NOT_SUPPORT)
{
    MOCKER(hrtGetDeviceType).stubs().will(invoke(stub_hrtGetDeviceType_950));

    EndpointDesc endpointDesc;
    CreateEndpointDesc(endpointDesc);
    endpointDesc.protocol = COMM_PROTOCOL_HCCS;
    EndpointHandle endpointHandle;

    HcommResult ret = HcommEndpointCreate(&endpointDesc, &endpointHandle);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(AicpuUbRtpEndpointTest, Ut_HcommEndpointCreate_When_Roce_Device_On_950_Expect_Return_NOT_SUPPORT)
{
    MOCKER(hrtGetDeviceType).stubs().will(invoke(stub_hrtGetDeviceType_950));

    EndpointDesc endpointDesc;
    CreateEndpointDesc(endpointDesc);
    endpointDesc.protocol = COMM_PROTOCOL_ROCE;
    EndpointHandle endpointHandle;

    HcommResult ret = HcommEndpointCreate(&endpointDesc, &endpointHandle);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(AicpuUbRtpEndpointTest, Ut_HcommEndpointCreate_When_Roce_Host_On_910B_Expect_Return_NOT_SUPPORT)
{
    MOCKER(hrtGetDeviceType).stubs().will(invoke(stub_hrtGetDeviceType_910B));

    EndpointDesc endpointDesc;
    CreateEndpointDesc(endpointDesc);
    endpointDesc.protocol = COMM_PROTOCOL_ROCE;
    endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    EndpointHandle endpointHandle;

    HcommResult ret = HcommEndpointCreate(&endpointDesc, &endpointHandle);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(AicpuUbRtpEndpointTest, Ut_HcommEndpointCreate_When_UbCtp_Host_On_910B_Expect_Return_NOT_SUPPORT)
{
    MOCKER(hrtGetDeviceType).stubs().will(invoke(stub_hrtGetDeviceType_910B));

    EndpointDesc endpointDesc;
    CreateEndpointDesc(endpointDesc);
    endpointDesc.protocol = COMM_PROTOCOL_UB_CTP;
    endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    EndpointHandle endpointHandle;

    HcommResult ret = HcommEndpointCreate(&endpointDesc, &endpointHandle);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(AicpuUbRtpEndpointTest, Ut_HcommEndpointCreate_When_UbRtp_Host_Expect_Return_ERROR)
{
    EndpointDesc endpointDesc;
    CreateEndpointDesc(endpointDesc);
    endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    EndpointHandle endpointHandle;

    HcommResult ret = HcommEndpointCreate(&endpointDesc, &endpointHandle);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(AicpuUbRtpEndpointTest, Ut_When_Register_Memory_NORMAL_Expect_Return_SUCCESS)
{
    MOCKER(hrtGetDeviceType).stubs().will(invoke(stub_hrtGetDeviceType_950));

    EndpointHandle endpointHandle;
    HcommResult ret = CreateEndpoint(endpointHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    CommMem mem = CreateCommMem((void*)0x01, 10, COMM_MEM_TYPE_DEVICE);
    void* memHandle;

    ret = HcommMemReg(endpointHandle, "memTag", &mem, &memHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcommMemUnreg(endpointHandle, memHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// ============ UB_RTP endpoint 接入 DeviceServerSocketContext 后的监听行为 ============

static HcclResult stub_hrtGetDeviceRefresh_success(s32* deviceLogicId)
{
    *deviceLogicId = 0;
    return HCCL_SUCCESS;
}

// 指定端口建听必须调用底层一次，防止监听上下文退回 no-op 后仍返回成功。
TEST_F(AicpuUbRtpEndpointTest, Ut_HcommEndpointStartListen_When_UbRtp_Device_Expect_CallManagerOnce_And_SUCCESS)
{
    MOCKER(hrtGetDeviceType).stubs().will(invoke(stub_hrtGetDeviceType_950));
    MOCKER(hrtGetDeviceRefresh).stubs().will(invoke(stub_hrtGetDeviceRefresh_success));

    EndpointHandle endpointHandle;
    ASSERT_EQ(CreateEndpoint(endpointHandle), HCCL_SUCCESS);

    const uint32_t listenPort = 10000;
    MOCKER_CPP(&hcomm::ServerSocketManager::ServerSocketStartListen).expects(once()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hcomm::ServerSocketManager::ServerSocketStopListen)
        .expects(once())
        .with(any(), any(), eq(listenPort))
        .will(returnValue(HCCL_SUCCESS));

    EXPECT_EQ(HcommEndpointStartListen(endpointHandle, listenPort, nullptr), HCCL_SUCCESS);
    EXPECT_EQ(HcommEndpointStopListen(endpointHandle, listenPort), HCCL_SUCCESS);
    EXPECT_EQ(HcommEndpointDestroy(endpointHandle), HCCL_SUCCESS);
}

// 停止监听必须透传底层错误，不能以 no-op SUCCESS 掩盖失败。
TEST_F(AicpuUbRtpEndpointTest, Ut_HcommEndpointStopListen_When_ManagerFails_Expect_ReturnError)
{
    MOCKER(hrtGetDeviceType).stubs().will(invoke(stub_hrtGetDeviceType_950));
    MOCKER(hrtGetDeviceRefresh).stubs().will(invoke(stub_hrtGetDeviceRefresh_success));

    EndpointHandle endpointHandle;
    ASSERT_EQ(CreateEndpoint(endpointHandle), HCCL_SUCCESS);

    const uint32_t listenPort = 10000;
    MOCKER_CPP(&hcomm::ServerSocketManager::ServerSocketStopListen)
        .expects(once())
        .with(any(), any(), eq(listenPort))
        .will(returnValue(HCCL_E_INTERNAL));

    EXPECT_EQ(HcommEndpointStopListen(endpointHandle, listenPort), HCCL_E_INTERNAL);
    EXPECT_EQ(HcommEndpointDestroy(endpointHandle), HCCL_SUCCESS);
}

// TC-UbRtpEndpoint_GetListenPort-001: DEVICE 场景首次查询端口，惰性建听并返回有效端口。
// 修复前旧 context 未覆写 GetListenPort 而返回 NOT_SUPPORT（客户实测阻断点）。
TEST_F(AicpuUbRtpEndpointTest, Ut_HcommEndpointGetListenPort_When_UbRtp_Device_Expect_Return_SUCCESS_And_ValidPort)
{
    MOCKER(hrtGetDeviceType).stubs().will(invoke(stub_hrtGetDeviceType_950));
    MOCKER(hrtGetDeviceRefresh).stubs().will(invoke(stub_hrtGetDeviceRefresh_success));

    EndpointHandle endpointHandle;
    ASSERT_EQ(CreateEndpoint(endpointHandle), HCCL_SUCCESS);

    uint32_t listenPort = 60001;
    MOCKER_CPP(&hcomm::ServerSocketManager::ServerSocketStartListen)
        .expects(once())
        .with(any(), any(), any(), outBoundP(&listenPort, sizeof(listenPort)))
        .will(returnValue(HCCL_SUCCESS));

    uint32_t port = 0;
    HcommResult ret = HcommEndpointGetListenPort(endpointHandle, &port);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(port, 60001U);
}

// TC-UbRtpEndpoint_GetListenPort-002: 二次查询命中 dynamicPort_ 缓存，不再触发底层建听（幂等）。
// 顺序桩首次建听成功，后续建听失败：若缓存分支失效，第二次查询将返回失败。
TEST_F(AicpuUbRtpEndpointTest, Ut_HcommEndpointGetListenPort_When_RepeatedQuery_Expect_ReturnCachedPort)
{
    MOCKER(hrtGetDeviceType).stubs().will(invoke(stub_hrtGetDeviceType_950));
    MOCKER(hrtGetDeviceRefresh).stubs().will(invoke(stub_hrtGetDeviceRefresh_success));

    EndpointHandle endpointHandle;
    ASSERT_EQ(CreateEndpoint(endpointHandle), HCCL_SUCCESS);

    uint32_t listenPort = 50001;
    MOCKER_CPP(&hcomm::ServerSocketManager::ServerSocketStartListen)
        .stubs()
        .with(any(), any(), any(), outBoundP(&listenPort, sizeof(listenPort)))
        .will(returnValue(HCCL_SUCCESS))
        .then(returnValue(HCCL_E_INTERNAL));

    uint32_t firstPort = 0;
    EXPECT_EQ(HcommEndpointGetListenPort(endpointHandle, &firstPort), HCCL_SUCCESS);
    EXPECT_EQ(firstPort, 50001U);

    // 第二次查询应命中缓存；若再次下发建听，顺序桩返回失败，使成功断言失败。
    uint32_t secondPort = 0;
    EXPECT_EQ(HcommEndpointGetListenPort(endpointHandle, &secondPort), HCCL_SUCCESS);
    EXPECT_EQ(secondPort, firstPort);

    // 销毁 Endpoint 时按缓存端口释放监听引用，重复查询不应增加释放次数。
    MOCKER_CPP(&hcomm::ServerSocketManager::ServerSocketStopListen)
        .expects(once())
        .with(any(), any(), eq(listenPort))
        .will(returnValue(HCCL_SUCCESS));
    EXPECT_EQ(HcommEndpointDestroy(endpointHandle), HCCL_SUCCESS);
}

// TC-UbRtpEndpoint_GetListenPort-003: 底层建听返回端口 0 视为无效，返回 HCCL_E_NETWORK
TEST_F(AicpuUbRtpEndpointTest, Ut_HcommEndpointGetListenPort_When_StartListenReturnsPortZero_Expect_E_NETWORK)
{
    MOCKER(hrtGetDeviceType).stubs().will(invoke(stub_hrtGetDeviceType_950));
    MOCKER(hrtGetDeviceRefresh).stubs().will(invoke(stub_hrtGetDeviceRefresh_success));

    EndpointHandle endpointHandle;
    ASSERT_EQ(CreateEndpoint(endpointHandle), HCCL_SUCCESS);

    uint32_t listenPort = 0;
    MOCKER_CPP(&hcomm::ServerSocketManager::ServerSocketStartListen)
        .stubs()
        .with(any(), any(), any(), outBoundP(&listenPort, sizeof(listenPort)))
        .will(returnValue(HCCL_SUCCESS));

    uint32_t port = 0;
    HcommResult ret = HcommEndpointGetListenPort(endpointHandle, &port);
    EXPECT_EQ(ret, HCCL_E_NETWORK);
}

// TC-UbRtpEndpoint_GetListenPort-004: 底层建听失败，错误码透传
TEST_F(AicpuUbRtpEndpointTest, Ut_HcommEndpointGetListenPort_When_StartListenFailed_Expect_Error)
{
    MOCKER(hrtGetDeviceType).stubs().will(invoke(stub_hrtGetDeviceType_950));
    MOCKER(hrtGetDeviceRefresh).stubs().will(invoke(stub_hrtGetDeviceRefresh_success));

    EndpointHandle endpointHandle;
    ASSERT_EQ(CreateEndpoint(endpointHandle), HCCL_SUCCESS);

    MOCKER_CPP(&hcomm::ServerSocketManager::ServerSocketStartListen).stubs().will(returnValue(HCCL_E_INTERNAL));

    uint32_t port = 0;
    HcommResult ret = HcommEndpointGetListenPort(endpointHandle, &port);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}
