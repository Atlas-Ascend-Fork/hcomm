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
#include <tuple>
#include <vector>
#include "ub_mem_endpoint.h"
#include "hcomm_res.h"
#include "hcomm_c_adpt.h"
#include "ip_address.h"
#include "hccp.h"
#include "buffer.h"
#include "network_api_exception.h"
#include "endpoint.h"
#include "adapter_rts.h"
#include "ep_mem_table_launch.h"
#include "common/process_exit_flag.h"
#include "server_socket_mgr.h"
#include "opbase_adpt.h"
#include "eid_info_mgr.h"
#include "hccp_ctx.h"
#include "hcomm_adapter_hccp.h"

using namespace hcomm;

class AicpuUbRtpEndpointTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AicpuUbRtpEndpointTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "AicpuUbRtpEndpointTest tests tear down." << std::endl; }

    virtual void SetUp()
    {
        std::cout << "A Test case in AicpuUbRtpEndpointTest SetUP" << std::endl;
        // PR 新增 DevEpMemTable 下发链（InitDevEpMemTable）UT 桩：UT 无真实 AICPU——
        // init 置 0（"无表"哨兵：下游 ptr!=0 守卫全跳过、uniqueIdV2 携 0，无野指针风险）；
        // 表同步空转 SUCCESS（Reg/Unreg 钩子不因无表回滚主流程）。
        // 注意：勿在此加 hrtGetDeviceType 全局桩——各用例体内自设 950/910B 桩，
        // 且 mockcpp 先注册的 .stubs() 优先，SetUp 全局桩会遮蔽 On_910B 用例的 910B 桩
        MOCKER(hcomm::LaunchEpMemTableInit).stubs().will(invoke(+[](uint64_t& outTablePtr) -> HcclResult {
            outTablePtr = 0;
            return HCCL_SUCCESS;
        }));
        MOCKER(hcomm::SyncUbLocMemToEpTable).stubs().will(returnValue(HCCL_SUCCESS));
    }

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
    MOCKER_CPP(&hcomm::ServerSocketMgr::ServerSocketStartListen).expects(once()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hcomm::ServerSocketMgr::ServerSocketStopListen)
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
    MOCKER_CPP(&hcomm::ServerSocketMgr::ServerSocketStopListen)
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
    MOCKER_CPP(&hcomm::ServerSocketMgr::ServerSocketStartListen)
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
    MOCKER_CPP(&hcomm::ServerSocketMgr::ServerSocketStartListen)
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
    MOCKER_CPP(&hcomm::ServerSocketMgr::ServerSocketStopListen)
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
    MOCKER_CPP(&hcomm::ServerSocketMgr::ServerSocketStartListen)
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

    MOCKER_CPP(&hcomm::ServerSocketMgr::ServerSocketStartListen).stubs().will(returnValue(HCCL_E_INTERNAL));

    uint32_t port = 0;
    HcommResult ret = HcommEndpointGetListenPort(endpointHandle, &port);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

// ===================== HcommMemUnreg 纯值提取 + 无条件 REMOVE 同步链路 UT（检视整改） =====================
// 适配层链路（RemoveLocalMemHandle 守卫 → GetMemRange 基类虚接口纯值提取 → UnregisterMemory →
// OnLocalMemUnregistered 纯值 REMOVE 下发）此前零 UT 覆盖——既有用例只覆盖 DevEpMemTable 类本身。
// 本组用例以真实 UB_RTP endpoint 驱动全链路（真实账本/真实 mgr 注册与提取/真实钩子体），仅桩 kernel 边界。

// 捕获 SyncUbLocMemToEpTable 的 (op,addr,size,tokenId,tokenValue)（tablePtr 不参与断言）
static std::vector<std::tuple<EpMemTableOp, uint64_t, uint64_t, uint32_t, uint32_t>> g_unregSyncCalls;

class AicpuUbRtpMemUnregSyncTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AicpuUbRtpMemUnregSyncTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "AicpuUbRtpMemUnregSyncTest tests tear down." << std::endl; }

    virtual void SetUp()
    {
        std::cout << "A Test case in AicpuUbRtpMemUnregSyncTest SetUP" << std::endl;
        g_unregSyncCalls.clear();
        // 仅桩表地址 init（0=无表哨兵，下游 ptr!=0 守卫全跳过）。SyncUbLocMemToEpTable 特意不预桩：
        // mockcpp 先注册的 .stubs() 优先，预桩会遮蔽用例内的捕获/失败桩（同上方 910B 桩注释先例）
        MOCKER(hcomm::LaunchEpMemTableInit).stubs().will(invoke(+[](uint64_t& outTablePtr) -> HcclResult {
            outTablePtr = 0;
            return HCCL_SUCCESS;
        }));
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
        // 退出标志为进程级全局态，置位用例结束即复位，防污染同二进制其它用例
        hcomm::ProcessExitFlag::ResetForTest();
        std::cout << "A Test case in AicpuUbRtpMemUnregSyncTest TearDown" << std::endl;
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

// (a) UB mgr 提取路径：Reg 的 UPSERT 与 Unreg 的 REMOVE 均携带注册时 (addr,size)，
// 且 REMOVE 的 token 按钩子契约填 0（REMOVE 仅消费 addr/size，kernel 侧忽略 token）
TEST_F(AicpuUbRtpMemUnregSyncTest, Ut_When_UnregUbLocalMem_Expect_TableRemoveCarriesRegisteredRange)
{
    MOCKER(hrtGetDeviceType).stubs().will(invoke(stub_hrtGetDeviceType_950));

    // 捕获桩先于 Reg 注册：Reg 侧 OnLocalMemRegistered（UPSERT）与 Unreg 侧（REMOVE）共用同一桩
    MOCKER(hcomm::SyncUbLocMemToEpTable)
        .stubs()
        .will(invoke(
            +[](uint64_t, EpMemTableOp op, uint64_t addr, uint64_t size, uint32_t tokenId,
                uint32_t tokenValue) -> HcclResult {
                g_unregSyncCalls.emplace_back(op, addr, size, tokenId, tokenValue);
                return HCCL_SUCCESS;
            }));

    EndpointHandle endpointHandle;
    ASSERT_EQ(CreateEndpoint(endpointHandle), HCCL_SUCCESS);

    // 真实链路：真实 UbRegedMemMgr 注册/GetMemRange 提取 + 真实 OnLocalMem* 钩子体转发
    constexpr uint64_t memAddr = 0x12340000U;
    constexpr uint64_t memSize = 0x1000U;
    CommMem mem = CreateCommMem(reinterpret_cast<void*>(memAddr), memSize, COMM_MEM_TYPE_DEVICE);
    HcommMemHandle memHandle = nullptr;
    ASSERT_EQ(HcommMemReg(endpointHandle, "unregSync", &mem, &memHandle), HCCL_SUCCESS);
    ASSERT_EQ(HcommMemUnreg(endpointHandle, memHandle), HCCL_SUCCESS);

    // Reg→UPSERT、Unreg→REMOVE 恰各一次，(addr,size) 均为注册值（device 表键与 host 账本严格同步的契约锚点）
    ASSERT_EQ(g_unregSyncCalls.size(), 2U);
    EXPECT_EQ(std::get<0>(g_unregSyncCalls[0]), EpMemTableOp::UPSERT);
    EXPECT_EQ(std::get<1>(g_unregSyncCalls[0]), memAddr);
    EXPECT_EQ(std::get<2>(g_unregSyncCalls[0]), memSize);
    EXPECT_EQ(std::get<0>(g_unregSyncCalls[1]), EpMemTableOp::REMOVE);
    EXPECT_EQ(std::get<1>(g_unregSyncCalls[1]), memAddr);
    EXPECT_EQ(std::get<2>(g_unregSyncCalls[1]), memSize);
    EXPECT_EQ(std::get<3>(g_unregSyncCalls[1]), 0U);
    EXPECT_EQ(std::get<4>(g_unregSyncCalls[1]), 0U);

    EXPECT_EQ(HcommEndpointDestroy(endpointHandle), HCCL_SUCCESS);
}

// (c) 表同步失败语义：注销主流程（账本+硬件）已成功且不可逆，REMOVE 失败不回滚、返回失败并
// ERROR 打印残留四元组（hcomm_mem_c_adpt.cc 既有语义；检视意见文字"仍返回成功仅告警"为旧语义）
TEST_F(AicpuUbRtpMemUnregSyncTest, Ut_When_TableSyncRemoveFails_Expect_UnregReturnsFailure)
{
    MOCKER(hrtGetDeviceType).stubs().will(invoke(stub_hrtGetDeviceType_950));
    MOCKER(hcomm::SyncUbLocMemToEpTable)
        .stubs()
        .will(invoke(+[](uint64_t, EpMemTableOp op, uint64_t, uint64_t, uint32_t, uint32_t) -> HcclResult {
            return op == EpMemTableOp::REMOVE ? HCCL_E_RUNTIME : HCCL_SUCCESS;
        }));

    EndpointHandle endpointHandle;
    ASSERT_EQ(CreateEndpoint(endpointHandle), HCCL_SUCCESS);

    CommMem mem = CreateCommMem(reinterpret_cast<void*>(0x12340000U), 0x1000U, COMM_MEM_TYPE_DEVICE);
    HcommMemHandle memHandle = nullptr;
    // Reg 侧 UPSERT 成功（返回成功佐证），Unreg 侧 REMOVE 失败透传为 Unreg 返回值
    ASSERT_EQ(HcommMemReg(endpointHandle, "unregSyncFail", &mem, &memHandle), HCCL_SUCCESS);
    EXPECT_EQ(HcommMemUnreg(endpointHandle, memHandle), HCCL_E_RUNTIME);

    EXPECT_EQ(HcommEndpointDestroy(endpointHandle), HCCL_SUCCESS);
}

// (d) 退出期注销：进程退出标志置位后走 ~TaggedMemMap 兜底同款链路（HcommMemUnreg）——
// OnLocalMemUnregistered 退出期守卫命中，跳过 REMOVE kernel 下发（退出期 runtime 资源不稳定，
// 下发会 core）；host 账本/归属记录照常清理，Unreg 返回成功（~TaggedMemMap 兜底无 ERROR 刷屏）
TEST_F(AicpuUbRtpMemUnregSyncTest, Ut_When_UnregAtProcessExit_Expect_SkipRemoveKernelAndReturnSuccess)
{
    MOCKER(hrtGetDeviceType).stubs().will(invoke(stub_hrtGetDeviceType_950));
    MOCKER(hcomm::SyncUbLocMemToEpTable)
        .stubs()
        .will(invoke(
            +[](uint64_t, EpMemTableOp op, uint64_t addr, uint64_t size, uint32_t tokenId,
                uint32_t tokenValue) -> HcclResult {
                g_unregSyncCalls.emplace_back(op, addr, size, tokenId, tokenValue);
                return HCCL_SUCCESS;
            }));

    EndpointHandle endpointHandle;
    ASSERT_EQ(CreateEndpoint(endpointHandle), HCCL_SUCCESS);

    // 正常期 Reg：UPSERT 恰一次
    CommMem mem = CreateCommMem(reinterpret_cast<void*>(0x12340000U), 0x1000U, COMM_MEM_TYPE_DEVICE);
    HcommMemHandle memHandle = nullptr;
    ASSERT_EQ(HcommMemReg(endpointHandle, "unregAtExit", &mem, &memHandle), HCCL_SUCCESS);
    ASSERT_EQ(g_unregSyncCalls.size(), 1U);

    // 退出期 Unreg：守卫命中，REMOVE 不下发，调用序列无新增
    hcomm::ProcessExitFlag::SetExiting();
    EXPECT_EQ(HcommMemUnreg(endpointHandle, memHandle), HCCL_SUCCESS);
    EXPECT_EQ(g_unregSyncCalls.size(), 1U);
    EXPECT_EQ(std::get<0>(g_unregSyncCalls[0]), EpMemTableOp::UPSERT);

    EXPECT_EQ(HcommEndpointDestroy(endpointHandle), HCCL_SUCCESS);
}
