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
#include <thread>
#include <chrono>
#include "securec.h"
#include <stdio.h>

#include "ping_mesh.h"
#include "hccn_rping.h"
#include "hccl_ip_address.h"

#include "hccn_rping.h"
#include "ping_mesh.h"
#include "hccl_ip_address.h"
#include "adapter_hccp.h"
#include "network/hccp_common.h"
#include "network/hccp_ping.h"
#include "dlra_function.h"
#include "adapter_rts.h"

using namespace hccl;

HcclResult HccnRpingGetResultStub(PingMesh* obj, u32 deviceId, u32 targetNum, RpingInput* input, RpingOutput* output)
{
    for (int i = 0; i < targetNum; i++) {
        output[i].state = 2;
        output[i].txPkt = 10;
        output[i].rxPkt = 10;
        output[i].minRTT = 5;
        output[i].maxRTT = 15;
        output[i].avgRTT = 10;
    }
    return HCCL_SUCCESS;
}

inline HcclResult hrtGetDeviceRefreshStub(s32* deviceLogicId)
{
    *deviceLogicId = 1;
    return HCCL_SUCCESS;
}

class HccnRping_UT : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HccnRping_UT SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "HccnRping_UT TearDown" << std::endl; }

    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        s32 portNum = 7;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(
            (HcclResult(PingMesh::*)(u32, u32, HcclIpAddress, u32, u32, u32, u32, u32)) & PingMesh::HccnRpingInit)
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(HCCL_SUCCESS));

        MOCKER_CPP(&PingMesh::HccnRpingDeinit).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

        MOCKER_CPP(&PingMesh::HccnRpingAddTarget).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

        MOCKER_CPP(&PingMesh::HccnRpingRemoveTarget).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

        MOCKER_CPP(&PingMesh::HccnRpingGetTarget).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

        MOCKER_CPP(&PingMesh::HccnRpingBatchPingStart).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

        MOCKER_CPP(&PingMesh::HccnRpingBatchPingStop).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

        MOCKER_CPP(&PingMesh::HccnRpingGetResult).stubs().with(mockcpp::any()).will(invoke(HccnRpingGetResultStub));

        MOCKER_CPP(&PingMesh::HccnRpingGetPayload).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

        MOCKER(hrtGetDeviceRefresh).stubs().with(mockcpp::any()).will(invoke(hrtGetDeviceRefreshStub));

        std::cout << "HccnRping_UT Test SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "HccnRping_UT Test TearDown" << std::endl;
    }

public:
    char deviceIp_[8][11] = {"192.1.1.58", "192.1.2.58", "192.1.3.58", "192.1.4.58",
                             "192.1.1.59", "192.1.2.59", "192.1.3.59", "192.1.4.59"};

    int ipLen_ = 16;
};

TEST_F(HccnRping_UT, ut_HccnRpingInit)
{
    MOCKER_CPP(&PingMesh::GetDeviceLogicId).stubs().with(mockcpp::any()).will(returnValue(1));

    u32 devId = 1;
    u32 devserverId = 2;
    HccnRpingInitAttr* initAttr = new HccnRpingInitAttr();
    initAttr->mode = HCCN_RPING_MODE_ROCE;
    initAttr->port = 13886;
    initAttr->npuNum = 128;
    initAttr->bufferSize = 4096 * 10;
    initAttr->sl = 0;
    initAttr->tc = 0;
    initAttr->ipAddr = new char[ipLen_];
    strcpy(initAttr->ipAddr, deviceIp_[devId]);
    void* pingmesh = nullptr;

    // 初始化
    auto ret = HccnRpingInit(devId, initAttr, &pingmesh);
    EXPECT_EQ(ret, HCCN_SUCCESS);
    EXPECT_TRUE(pingmesh != nullptr);
    delete[] initAttr->ipAddr;
    delete initAttr;

    // 反初始化
    ret = HccnRpingDeinit(pingmesh);
    EXPECT_EQ(ret, HCCN_SUCCESS);
}

TEST_F(HccnRping_UT, ut_HccnRpingInit_error)
{
    MOCKER_CPP(&PingMesh::GetDeviceLogicId).stubs().with(mockcpp::any()).will(returnValue(1));

    u32 devId = 1;
    u32 devserverId = 2;
    HccnRpingInitAttr* initAttr = nullptr;
    void* pingmesh = nullptr;
    // 空initAttr初始化错误检查
    auto ret = HccnRpingInit(devId, initAttr, &pingmesh);
    EXPECT_EQ(ret, HCCN_E_PARA);

    initAttr = new HccnRpingInitAttr();
    initAttr->mode = HCCN_RPING_MODE_ROCE;
    initAttr->port = 13886;
    initAttr->npuNum = 128;
    initAttr->bufferSize = 4096 * 10;
    initAttr->sl = 0;
    initAttr->tc = 0;
    initAttr->ipAddr = new char[ipLen_];
    strcpy(initAttr->ipAddr, deviceIp_[devId]);

    ret = HccnRpingInit(devId, initAttr, &pingmesh);
    EXPECT_EQ(ret, HCCN_SUCCESS);
    EXPECT_TRUE(pingmesh != nullptr);
    delete[] initAttr->ipAddr;
    delete initAttr;
}

TEST_F(HccnRping_UT, ut_HccnRpingInit_DeprecatedMode3_Rejected)
{
    MOCKER_CPP(&PingMesh::GetDeviceLogicId).stubs().with(mockcpp::any()).will(returnValue(1));

    u32 devId = 1;
    HccnRpingInitAttr* initAttr = new HccnRpingInitAttr();
    initAttr->mode = static_cast<HccnRpingMode>(3); // old UB_RC_TP(value 3) removed, now falls into >= RESERVED reject
    initAttr->port = 13886;
    initAttr->npuNum = 128;
    initAttr->bufferSize = 4096 * 10;
    initAttr->sl = 0;
    initAttr->tc = 0;
    initAttr->ipAddr = new char[ipLen_];
    strcpy_s(initAttr->ipAddr, ipLen_, deviceIp_[devId]);
    void* pingmesh = nullptr;

    // old HccnRpingInit no longer supports RC+RTP(mode=3); should return HCCN_E_PARA
    auto ret = HccnRpingInit(devId, initAttr, &pingmesh);
    EXPECT_EQ(ret, HCCN_E_PARA);
    EXPECT_TRUE(pingmesh == nullptr);

    delete[] initAttr->ipAddr;
    delete initAttr;
}

TEST_F(HccnRping_UT, ut_HccnRping)
{
    MOCKER_CPP(&PingMesh::GetDeviceLogicId).stubs().with(mockcpp::any()).will(returnValue(1));

    u32 devId = 1;
    u32 devserverId = 2;
    HccnRpingInitAttr* initAttr = new HccnRpingInitAttr();
    initAttr->mode = HCCN_RPING_MODE_ROCE;
    initAttr->port = 13886;
    initAttr->npuNum = 128;
    initAttr->bufferSize = 4096 * 10;
    initAttr->sl = 0;
    initAttr->tc = 0;
    initAttr->ipAddr = new char[ipLen_];
    strcpy(initAttr->ipAddr, deviceIp_[devId]);
    void* pingmesh = nullptr;

    // 初始化
    auto ret = HccnRpingInit(devId, initAttr, &pingmesh);
    EXPECT_EQ(ret, HCCN_SUCCESS);
    EXPECT_TRUE(pingmesh != nullptr);
    delete[] initAttr->ipAddr;
    delete initAttr;

    // 添加节点
    int targetNum = 1;
    HccnRpingTargetInfo* target = new HccnRpingTargetInfo[1];
    target[0].srcPort = 0;
    target[0].sl = 0;
    target[0].tc = 0;
    target[0].port = 13886;
    target[0].payloadLen = 12;
    target[0].addrType = 0;
    target[0].srcIp = new char[ipLen_];
    target[0].dstIp = new char[ipLen_];
    char payload[12] = "hellotarget";
    strcpy(target[0].payload, payload);
    strcpy(target[0].srcIp, deviceIp_[devserverId]);
    strcpy(target[0].dstIp, deviceIp_[devId]);
    ret = HccnRpingAddTarget(pingmesh, targetNum, target);
    EXPECT_EQ(ret, HCCN_SUCCESS);

    // 获取节点
    HccnRpingAddTargetState targetState[1]{HCCN_RPING_ADDTARGET_STATE_DONE};
    ret = HccnRpingGetTarget(pingmesh, targetNum, target, targetState);
    EXPECT_EQ(ret, HCCN_SUCCESS);

    // 开始批量测试
    uint32_t pktNum = 10;
    uint32_t interval = 1;
    uint32_t timeout = 1000;
    ret = HccnRpingBatchPingStart(pingmesh, pktNum, interval, timeout);
    EXPECT_EQ(ret, HCCN_SUCCESS);

    // 停止批量测试
    ret = HccnRpingBatchPingStop(pingmesh);
    EXPECT_EQ(ret, HCCN_SUCCESS);

    // 获取结果
    HccnRpingResultInfo result[1]{0};
    ret = HccnRpingGetResult(pingmesh, targetNum, target, result);
    EXPECT_EQ(ret, HCCN_SUCCESS);

    // 获取payload
    void* payloadOutput = nullptr;
    u32 payloadLenOutput = 0;
    ret = HccnRpingGetPayload(pingmesh, &payloadOutput, &payloadLenOutput);
    EXPECT_EQ(ret, HCCN_SUCCESS);

    // 删除节点
    ret = HccnRpingRemoveTarget(pingmesh, targetNum, target);
    EXPECT_EQ(ret, HCCN_SUCCESS);

    // 反初始化
    ret = HccnRpingDeinit(pingmesh);
    EXPECT_EQ(ret, HCCN_SUCCESS);

    delete[] target[0].srcIp;
    delete[] target[0].dstIp;
    delete[] target;
}

TEST_F(HccnRping_UT, ut_HccnRping_RpingAddTargetV2)
{
    MOCKER_CPP(&PingMesh::GetDeviceLogicId).stubs().with(mockcpp::any()).will(returnValue(1));

    u32 devId = 1;
    u32 devserverId = 2;
    HccnRpingInitAttr* initAttr = new HccnRpingInitAttr();
    initAttr->mode = HCCN_RPING_MODE_ROCE;
    initAttr->port = 13886;
    initAttr->npuNum = 128;
    initAttr->bufferSize = 4096 * 10;
    initAttr->sl = 0;
    initAttr->tc = 0;
    initAttr->ipAddr = new char[ipLen_];
    strcpy(initAttr->ipAddr, deviceIp_[devId]);
    void* pingmesh = nullptr;

    // 初始化
    auto ret = HccnRpingInit(devId, initAttr, &pingmesh);
    EXPECT_EQ(ret, HCCN_SUCCESS);
    EXPECT_TRUE(pingmesh != nullptr);
    delete[] initAttr->ipAddr;
    delete initAttr;

    // 添加节点
    int targetNum = 1;
    HccnRpingTargetInfo* target = new HccnRpingTargetInfo[1];
    target[0].srcPort = 0;
    target[0].sl = 0;
    target[0].tc = 0;
    target[0].port = 13886;
    target[0].payloadLen = 12;
    target[0].addrType = 0;
    target[0].srcIp = new char[ipLen_];
    target[0].dstIp = new char[ipLen_];
    char payload[12] = "hellotarget";
    strcpy(target[0].payload, payload);
    strcpy(target[0].srcIp, deviceIp_[devserverId]);
    strcpy(target[0].dstIp, deviceIp_[devId]);

    HccnRpingAddTargetConfig config;
    config.connectTimeout = 1000;
    ret = HccnRpingAddTargetV2(pingmesh, targetNum, target, &config);
    EXPECT_EQ(ret, HCCN_SUCCESS);

    // 删除节点
    ret = HccnRpingRemoveTarget(pingmesh, targetNum, target);
    EXPECT_EQ(ret, HCCN_SUCCESS);

    // 反初始化
    ret = HccnRpingDeinit(pingmesh);
    EXPECT_EQ(ret, HCCN_SUCCESS);

    delete[] target[0].srcIp;
    delete[] target[0].dstIp;
    delete[] target;
}

TEST_F(HccnRping_UT, ut_HccnRping_RpingAddTargetWithCfg)
{
    MOCKER_CPP(&PingMesh::GetDeviceLogicId).stubs().with(mockcpp::any()).will(returnValue(1));

    u32 devId = 1;
    u32 devserverId = 2;
    HccnRpingInitAttr* initAttr = new HccnRpingInitAttr();
    initAttr->mode = HCCN_RPING_MODE_ROCE;
    initAttr->port = 13886;
    initAttr->npuNum = 128;
    initAttr->bufferSize = 4096 * 10;
    initAttr->sl = 0;
    initAttr->tc = 0;
    initAttr->ipAddr = new char[ipLen_];
    strcpy(initAttr->ipAddr, deviceIp_[devId]);
    void* pingmesh = nullptr;

    // 初始化
    auto ret = HccnRpingInit(devId, initAttr, &pingmesh);
    EXPECT_EQ(ret, HCCN_SUCCESS);
    EXPECT_TRUE(pingmesh != nullptr);
    delete[] initAttr->ipAddr;
    delete initAttr;

    // 添加节点
    int targetNum = 1;
    HccnRpingTargetInfo* target = new HccnRpingTargetInfo[1];
    target[0].srcPort = 0;
    target[0].sl = 0;
    target[0].tc = 0;
    target[0].port = 13886;
    target[0].payloadLen = 12;
    target[0].addrType = 0;
    target[0].srcIp = new char[ipLen_];
    target[0].dstIp = new char[ipLen_];
    char payload[12] = "hellotarget";
    strcpy(target[0].payload, payload);
    strcpy(target[0].srcIp, deviceIp_[devserverId]);
    strcpy(target[0].dstIp, deviceIp_[devId]);

    HccnRpingAddTargetConfig config;
    config.connectTimeout = 1000;
    ret = HccnRpingAddTargetWithCfg(pingmesh, targetNum, target, &config);
    EXPECT_EQ(ret, HCCN_SUCCESS);

    // 删除节点
    ret = HccnRpingRemoveTarget(pingmesh, targetNum, target);
    EXPECT_EQ(ret, HCCN_SUCCESS);

    // 反初始化
    ret = HccnRpingDeinit(pingmesh);
    EXPECT_EQ(ret, HCCN_SUCCESS);

    delete[] target[0].srcIp;
    delete[] target[0].dstIp;
    delete[] target;
}

TEST_F(HccnRping_UT, ut_HccnRping_RpingAddTargetV2_err)
{
    HccnRpingAddTargetConfig config;
    config.connectTimeout = 0;
    auto ret = HccnRpingAddTargetWithCfg(nullptr, 1, nullptr, &config);
    EXPECT_EQ(ret, HCCN_E_PARA);
}

TEST_F(HccnRping_UT, ut_HccnRpingHandleCheck)
{
    u32 devId = 3;
    u32 devserverId = 2;
    HccnRpingInitAttr* initAttr = new HccnRpingInitAttr();
    initAttr->mode = HCCN_RPING_MODE_ROCE;
    initAttr->port = 13886;
    initAttr->npuNum = 128;
    initAttr->bufferSize = 4096 * 10;
    initAttr->sl = 0;
    initAttr->tc = 0;
    initAttr->ipAddr = new char[ipLen_];
    strcpy(initAttr->ipAddr, deviceIp_[devId]);
    void* pingmesh = nullptr;

    // 初始化
    auto ret = HccnRpingInit(1, initAttr, &pingmesh);
    EXPECT_EQ(ret, HCCN_SUCCESS);
    EXPECT_TRUE(pingmesh != nullptr);
    delete[] initAttr->ipAddr;
    delete initAttr;

    // 添加节点
    int targetNum = 1;
    HccnRpingTargetInfo* target = new HccnRpingTargetInfo[1];
    target[0].srcPort = 0;
    target[0].sl = 0;
    target[0].tc = 0;
    target[0].port = 13886;
    target[0].payloadLen = 12;
    target[0].addrType = 0;
    target[0].srcIp = new char[ipLen_];
    target[0].dstIp = new char[ipLen_];
    char payload[12] = "hellotarget";
    strcpy(target[0].payload, payload);
    strcpy(target[0].srcIp, deviceIp_[devserverId]);
    strcpy(target[0].dstIp, deviceIp_[devId]);
    ret = HccnRpingAddTarget(pingmesh, targetNum, target);
    EXPECT_EQ(ret, HCCN_E_PARA);

    // 获取节点
    HccnRpingAddTargetState targetState[1]{HCCN_RPING_ADDTARGET_STATE_DONE};
    ret = HccnRpingGetTarget(pingmesh, targetNum, target, targetState);
    EXPECT_EQ(ret, HCCN_E_PARA);

    // 开始批量测试
    uint32_t pktNum = 10;
    uint32_t interval = 1;
    uint32_t timeout = 1000;
    ret = HccnRpingBatchPingStart(pingmesh, pktNum, interval, timeout);
    EXPECT_EQ(ret, HCCN_E_PARA);

    // 停止批量测试
    ret = HccnRpingBatchPingStop(pingmesh);
    EXPECT_EQ(ret, HCCN_E_PARA);

    // 获取结果
    HccnRpingResultInfo result[1]{0};
    ret = HccnRpingGetResult(pingmesh, targetNum, target, result);
    EXPECT_EQ(ret, HCCN_E_PARA);

    // 删除节点
    ret = HccnRpingRemoveTarget(pingmesh, targetNum, target);
    EXPECT_EQ(ret, HCCN_E_PARA);

    // 反初始化
    ret = HccnRpingDeinit(pingmesh);
    EXPECT_EQ(ret, HCCN_E_PARA);

    delete[] target[0].srcIp;
    delete[] target[0].dstIp;
    delete[] target;
    delete static_cast<PingMesh*>(pingmesh);
}

// ========== HccnRpingGetCtxInfo unit tests ==========

// Call GetCtxInfo right after UB-mode Init: after filling initInfo_.client.ub.key/size it should return jettyId
TEST_F(HccnRping_UT, ut_HccnRpingGetCtxInfo_UbSuccess)
{
    // Construct PingMesh directly, simulating the state after UB-mode Init
    PingMesh* rping = new PingMesh();
    HccnRpingInitAttr initAttr = {};
    initAttr.mode = HCCN_RPING_MODE_UB;
    rping->init(&initAttr);
    rping->rpingState_ = RpingState::INITED; // collaborator state machine: GetCtxInfo requires initialized

    // Simulate HccnRpingInit filling initInfo_.client.ub: size=24, key[20~23] is jettyId
    const u32 expectJettyId = 1234U;
    PingInitInfo& initInfo = rping->initInfo_;
    initInfo.client.ub.size = 24U; // sizeof(UbJettyId) = 24
    // write jettyId into key[20~23] (little-endian)
    initInfo.client.ub.key[20] = static_cast<uint8_t>(expectJettyId & 0xFF);
    initInfo.client.ub.key[21] = static_cast<uint8_t>((expectJettyId >> 8) & 0xFF);
    initInfo.client.ub.key[22] = static_cast<uint8_t>((expectJettyId >> 16) & 0xFF);
    initInfo.client.ub.key[23] = static_cast<uint8_t>((expectJettyId >> 24) & 0xFF);

    // Call PingMesh method (skipping C API layer devLogicId check)
    HccnRpingCtxJettyInfo jettyInfo = {};
    HccnRpingCtxInfo info = &jettyInfo;
    auto ret = rping->HccnRpingGetCtxInfo(HCCN_RPING_CTX_INFO_JETTY, &info, sizeof(jettyInfo));
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(jettyInfo.jettyId, expectJettyId);

    delete rping;
}

// GetCtxInfo without Init: mode/state check (mode=ROCE default, rpingState_=UNINIT) -> HCCL_E_INTERNAL
TEST_F(HccnRping_UT, ut_HccnRpingGetCtxInfo_NotInited)
{
    PingMesh* rping = new PingMesh();
    // default rpingState_=UNINIT, intercepted at the mode/state check (not the ub.size check)
    HccnRpingCtxJettyInfo jettyInfo = {};
    HccnRpingCtxInfo info = &jettyInfo;
    auto ret = rping->HccnRpingGetCtxInfo(HCCN_RPING_CTX_INFO_JETTY, &info, sizeof(jettyInfo));
    EXPECT_EQ(ret, HCCL_E_INTERNAL);

    delete rping;
}

// RoCE mode: intercepted at the mode/state check (mode != UB) -> HCCL_E_INTERNAL
TEST_F(HccnRping_UT, ut_HccnRpingGetCtxInfo_RoCeModeNotSupport)
{
    PingMesh* rping = new PingMesh();
    HccnRpingInitAttr initAttr = {};
    initAttr.mode = HCCN_RPING_MODE_ROCE;
    rping->init(&initAttr);
    HccnRpingCtxJettyInfo jettyInfo = {};
    HccnRpingCtxInfo info = &jettyInfo;
    auto ret = rping->HccnRpingGetCtxInfo(HCCN_RPING_CTX_INFO_JETTY, &info, sizeof(jettyInfo));
    EXPECT_EQ(ret, HCCL_E_INTERNAL);

    delete rping;
}

// Inited UB context with corrupted jetty key size: secondary ub.size guard (size < sizeof(UbJettyId)) ->
// HCCL_E_INTERNAL
TEST_F(HccnRping_UT, ut_HccnRpingGetCtxInfo_UbSizeTooSmall)
{
    PingMesh* rping = new PingMesh();
    HccnRpingInitAttr initAttr = {};
    initAttr.mode = HCCN_RPING_MODE_UB;
    rping->init(&initAttr);
    rping->rpingState_ = RpingState::INITED;
    // corrupted key size: 23 < sizeof(UbJettyId)=24, reaches the secondary guard after mode/state passes
    rping->initInfo_.client.ub.size = 23U;
    HccnRpingCtxJettyInfo jettyInfo = {};
    HccnRpingCtxInfo info = &jettyInfo;
    auto ret = rping->HccnRpingGetCtxInfo(HCCN_RPING_CTX_INFO_JETTY, &info, sizeof(jettyInfo));
    EXPECT_EQ(ret, HCCL_E_INTERNAL);

    delete rping;
}

// Inited-UB-only guard: mode=UB but rpingState_ not set to INITED, should fail at the mode/state check
TEST_F(HccnRping_UT, ut_HccnRpingGetCtxInfo_UbNotInited)
{
    PingMesh* rping = new PingMesh();
    HccnRpingInitAttr initAttr = {};
    initAttr.mode = HCCN_RPING_MODE_UB;
    rping->init(&initAttr);
    // leave rpingState_ = UNINIT (default); set a valid key size so only the state guard can reject
    // (without it, the ub.size<24 fallback would also return E_INTERNAL and mask the state guard)
    rping->initInfo_.client.ub.size = 24U;
    HccnRpingCtxJettyInfo jettyInfo = {};
    HccnRpingCtxInfo info = &jettyInfo;
    auto ret = rping->HccnRpingGetCtxInfo(HCCN_RPING_CTX_INFO_JETTY, &info, sizeof(jettyInfo));
    EXPECT_EQ(ret, HCCL_E_INTERNAL);

    delete rping;
}

// infoType not JETTY: should return HCCL_E_PARA
TEST_F(HccnRping_UT, ut_HccnRpingGetCtxInfo_InvalidInfoType)
{
    PingMesh* rping = new PingMesh();
    HccnRpingCtxJettyInfo jettyInfo = {};
    HccnRpingCtxInfo info = &jettyInfo;
    auto ret = rping->HccnRpingGetCtxInfo(HCCN_RPING_CTX_INFO_RESERVED, &info, sizeof(jettyInfo));
    EXPECT_EQ(ret, HCCL_E_PARA);

    delete rping;
}

// infoLen insufficient: should return HCCL_E_PARA
TEST_F(HccnRping_UT, ut_HccnRpingGetCtxInfo_InfoLenTooSmall)
{
    PingMesh* rping = new PingMesh();
    HccnRpingCtxJettyInfo jettyInfo = {};
    HccnRpingCtxInfo info = &jettyInfo;
    auto ret = rping->HccnRpingGetCtxInfo(HCCN_RPING_CTX_INFO_JETTY, &info, sizeof(uint32_t));
    EXPECT_EQ(ret, HCCL_E_PARA);

    delete rping;
}

// info is null: should return HCCL_E_PTR (CHK_PTR_NULL)
TEST_F(HccnRping_UT, ut_HccnRpingGetCtxInfo_NullInfo)
{
    PingMesh* rping = new PingMesh();
    auto ret = rping->HccnRpingGetCtxInfo(HCCN_RPING_CTX_INFO_JETTY, nullptr, sizeof(HccnRpingCtxJettyInfo));
    EXPECT_EQ(ret, HCCL_E_PTR);

    delete rping;
}

// C API: rpingCtx is null
TEST_F(HccnRping_UT, ut_HccnRpingGetCtxInfo_CApi_NullRpingCtx)
{
    HccnRpingCtxJettyInfo jettyInfo = {};
    HccnRpingCtxInfo info = &jettyInfo;
    HccnRpingCtx rpingCtx = nullptr;
    auto ret = HccnRpingGetCtxInfo(&rpingCtx, HCCN_RPING_CTX_INFO_JETTY, &info, sizeof(jettyInfo));
    EXPECT_EQ(ret, HCCN_E_PARA);
}

// C API: *rpingCtx is null
TEST_F(HccnRping_UT, ut_HccnRpingGetCtxInfo_CApi_NullDerefRpingCtx)
{
    HccnRpingCtxJettyInfo jettyInfo = {};
    HccnRpingCtxInfo info = &jettyInfo;
    HccnRpingCtx innerCtx = nullptr;
    HccnRpingCtx* rpingCtxPtr = &innerCtx;
    auto ret = HccnRpingGetCtxInfo(rpingCtxPtr, HCCN_RPING_CTX_INFO_JETTY, &info, sizeof(jettyInfo));
    EXPECT_EQ(ret, HCCN_E_PARA);
}

// C API: info is null
TEST_F(HccnRping_UT, ut_HccnRpingGetCtxInfo_CApi_NullInfo)
{
    PingMesh* rping = new PingMesh();
    HccnRpingCtx rpingCtx = rping;
    auto ret = HccnRpingGetCtxInfo(&rpingCtx, HCCN_RPING_CTX_INFO_JETTY, nullptr, sizeof(HccnRpingCtxJettyInfo));
    EXPECT_EQ(ret, HCCN_E_PARA);
    delete rping;
}

// C API: *info is null
TEST_F(HccnRping_UT, ut_HccnRpingGetCtxInfo_CApi_NullDerefInfo)
{
    PingMesh* rping = new PingMesh();
    HccnRpingCtx rpingCtx = rping;
    HccnRpingCtxInfo innerInfo = nullptr;
    HccnRpingCtxInfo* infoPtr = &innerInfo;
    auto ret = HccnRpingGetCtxInfo(&rpingCtx, HCCN_RPING_CTX_INFO_JETTY, infoPtr, sizeof(HccnRpingCtxJettyInfo));
    EXPECT_EQ(ret, HCCN_E_PARA);
    delete rping;
}

// C API: devLogicId mismatch
TEST_F(HccnRping_UT, ut_HccnRpingGetCtxInfo_CApi_DevIdMismatch)
{
    // GetDeviceLogicId mock returns 99, hrtGetDeviceRefreshStub returns 1, creating a mismatch
    MOCKER_CPP(&PingMesh::GetDeviceLogicId).stubs().with(mockcpp::any()).will(returnValue(99));

    PingMesh* rping = new PingMesh();
    HccnRpingCtx rpingCtx = rping;
    HccnRpingCtxJettyInfo jettyInfo = {};
    HccnRpingCtxInfo info = &jettyInfo;
    auto ret = HccnRpingGetCtxInfo(&rpingCtx, HCCN_RPING_CTX_INFO_JETTY, &info, sizeof(jettyInfo));
    EXPECT_EQ(ret, HCCN_E_PARA);
    delete rping;
}

// C API: full success path (UB mode + devLogicId match)
TEST_F(HccnRping_UT, ut_HccnRpingGetCtxInfo_CApi_Success)
{
    MOCKER_CPP(&PingMesh::GetDeviceLogicId).stubs().with(mockcpp::any()).will(returnValue(1));

    PingMesh* rping = new PingMesh();
    HccnRpingInitAttr initAttr = {};
    initAttr.mode = HCCN_RPING_MODE_UB;
    rping->init(&initAttr);
    rping->rpingState_ = RpingState::INITED; // collaborator state machine

    const u32 expectJettyId = 5678U;
    PingInitInfo& initInfo = rping->initInfo_;
    initInfo.client.ub.size = 24U; // sizeof(UbJettyId) = 24
    initInfo.client.ub.key[20] = static_cast<uint8_t>(expectJettyId & 0xFF);
    initInfo.client.ub.key[21] = static_cast<uint8_t>((expectJettyId >> 8) & 0xFF);
    initInfo.client.ub.key[22] = static_cast<uint8_t>((expectJettyId >> 16) & 0xFF);
    initInfo.client.ub.key[23] = static_cast<uint8_t>((expectJettyId >> 24) & 0xFF);

    HccnRpingCtx rpingCtx = rping;
    HccnRpingCtxJettyInfo jettyInfo = {};
    HccnRpingCtxInfo info = &jettyInfo;
    auto ret = HccnRpingGetCtxInfo(&rpingCtx, HCCN_RPING_CTX_INFO_JETTY, &info, sizeof(jettyInfo));
    EXPECT_EQ(ret, HCCN_SUCCESS);
    EXPECT_EQ(jettyInfo.jettyId, expectJettyId);
    delete rping;
}

// C API: illegal infoType should pass through HCCN_E_PARA (not HCCN_E_FAIL)
TEST_F(HccnRping_UT, ut_HccnRpingGetCtxInfo_CApi_InvalidInfoType)
{
    MOCKER_CPP(&PingMesh::GetDeviceLogicId).stubs().with(mockcpp::any()).will(returnValue(1));

    PingMesh* rping = new PingMesh();
    HccnRpingInitAttr initAttr = {};
    initAttr.mode = HCCN_RPING_MODE_UB;
    rping->init(&initAttr);
    PingInitInfo& initInfo = rping->initInfo_;
    initInfo.client.ub.size = 24U; // satisfy ub.size check so it won't fail on ub.size

    HccnRpingCtx rpingCtx = rping;
    HccnRpingCtxJettyInfo jettyInfo = {};
    HccnRpingCtxInfo info = &jettyInfo;
    auto ret = HccnRpingGetCtxInfo(&rpingCtx, HCCN_RPING_CTX_INFO_RESERVED, &info, sizeof(jettyInfo));
    EXPECT_EQ(ret, HCCN_E_PARA);
    delete rping;
}

// C API: insufficient infoLen should pass through HCCN_E_PARA
TEST_F(HccnRping_UT, ut_HccnRpingGetCtxInfo_CApi_InfoLenTooSmall)
{
    MOCKER_CPP(&PingMesh::GetDeviceLogicId).stubs().with(mockcpp::any()).will(returnValue(1));

    PingMesh* rping = new PingMesh();
    HccnRpingInitAttr initAttr = {};
    initAttr.mode = HCCN_RPING_MODE_UB;
    rping->init(&initAttr);
    PingInitInfo& initInfo = rping->initInfo_;
    initInfo.client.ub.size = 24U;

    HccnRpingCtx rpingCtx = rping;
    HccnRpingCtxJettyInfo jettyInfo = {};
    HccnRpingCtxInfo info = &jettyInfo;
    auto ret = HccnRpingGetCtxInfo(&rpingCtx, HCCN_RPING_CTX_INFO_JETTY, &info, sizeof(uint32_t));
    EXPECT_EQ(ret, HCCN_E_PARA);
    delete rping;
}

// C API: RoCE mode (ub.size=0) should return HCCN_E_FAIL (internal failure)
TEST_F(HccnRping_UT, ut_HccnRpingGetCtxInfo_CApi_RoCeModeFail)
{
    MOCKER_CPP(&PingMesh::GetDeviceLogicId).stubs().with(mockcpp::any()).will(returnValue(1));

    PingMesh* rping = new PingMesh();
    HccnRpingInitAttr initAttr = {};
    initAttr.mode = HCCN_RPING_MODE_ROCE;
    rping->init(&initAttr);
    // ub.size=0 in RoCE mode

    HccnRpingCtx rpingCtx = rping;
    HccnRpingCtxJettyInfo jettyInfo = {};
    HccnRpingCtxInfo info = &jettyInfo;
    auto ret = HccnRpingGetCtxInfo(&rpingCtx, HCCN_RPING_CTX_INFO_JETTY, &info, sizeof(jettyInfo));
    EXPECT_EQ(ret, HCCN_E_FAIL);
    delete rping;
}

TEST_F(HccnRping_UT, ut_HccnRpingAdapter)
{
    // 补充覆盖率
    auto ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 未初始化新增接口
    struct PingInitAttr initAttr {
        0
    };
    struct PingInitInfo initInfo {
        0
    };
    void* pingHandle = nullptr;
    ret = hrtRaPingInit(&initAttr, &initInfo, &pingHandle);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
    ret = hrtRaPingDeinit(pingHandle);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);

    struct PingTargetInfo target[2];
    struct PingTargetCommInfo targetRemoteInfo[2];
    struct PingTaskAttr attr;
    u32 num;
    struct PingTargetResult result[2];
    ret = hrtRaPingTargetAdd(pingHandle, target, num);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
    ret = hrtRaPingTargetDel(pingHandle, targetRemoteInfo, num);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
    ret = hrtRaPingTaskStart(pingHandle, &attr);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
    ret = hrtRaPingTaskStop(pingHandle);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
    ret = hrtRaPingGetResults(pingHandle, result, &num);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);

    // 手动初始化新增接口
    DlRaFunction::GetInstance().dlRaPingInit = [](struct PingInitAttr*, struct PingInitInfo*, void**) -> int {
        return 0;
    };
    DlRaFunction::GetInstance().dlRaPingDeinit = [](void*) -> int {
        return 0;
    };
    DlRaFunction::GetInstance().dlRaPingTargetAdd = [](void*, struct PingTargetInfo*, uint32_t) -> int {
        return 0;
    };
    DlRaFunction::GetInstance().dlRaPingTargetDel = [](void*, struct PingTargetCommInfo*, uint32_t) -> int {
        return 0;
    };
    DlRaFunction::GetInstance().dlRaPingTaskStart = [](void*, struct PingTaskAttr*) -> int {
        return 0;
    };
    DlRaFunction::GetInstance().dlRaPingTaskStop = [](void*) -> int {
        return 0;
    };
    DlRaFunction::GetInstance().dlRaPingGetResults = [](void*, struct PingTargetResult*, uint32_t*) -> int {
        return 0;
    };

    ret = hrtRaPingInit(&initAttr, &initInfo, &pingHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = hrtRaPingDeinit(pingHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = hrtRaPingTargetAdd(pingHandle, target, num);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = hrtRaPingTargetDel(pingHandle, targetRemoteInfo, num);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = hrtRaPingTaskStart(pingHandle, &attr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = hrtRaPingTaskStop(pingHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = hrtRaPingGetResults(pingHandle, result, &num);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // rts新增接口
    char dst[10]{0};
    uint64_t destMax = 10;
    char src[10]{1};
    uint64_t count = 10;
    HcclRtMemcpyKind kind = HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_HOST;
    ret = hrtMemSyncCopyEx(dst, destMax, src, count, kind);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// ========== HccnRpingInitWithCfg entry param checks ==========

TEST_F(HccnRping_UT, ut_HccnRpingInitWithCfg_NullInitAttr)
{
    u32 devId = 1;
    HccnRpingInitCfg* initCfg = new HccnRpingInitCfg();
    initCfg->validCfgLen = sizeof(HccnRpingInitCfg);
    initCfg->ub.taMode = HCCN_RPING_UB_RC;
    initCfg->ub.tpMode = HCCN_RPING_UB_CTP;
    void* pingmesh = nullptr;
    auto ret = HccnRpingInitWithCfg(devId, nullptr, initCfg, &pingmesh);
    EXPECT_EQ(ret, HCCN_E_PARA);
    delete initCfg;
}

TEST_F(HccnRping_UT, ut_HccnRpingInitWithCfg_NullInitCfg)
{
    u32 devId = 1;
    HccnRpingInitAttr* initAttr = new HccnRpingInitAttr();
    initAttr->mode = HCCN_RPING_MODE_UB;
    initAttr->port = 13886;
    initAttr->npuNum = 128;
    initAttr->bufferSize = 4096 * 10;
    initAttr->sl = 0;
    initAttr->tc = 0;
    initAttr->ipAddr = new char[ipLen_];
    strcpy_s(initAttr->ipAddr, ipLen_, deviceIp_[devId]);
    void* pingmesh = nullptr;
    auto ret = HccnRpingInitWithCfg(devId, initAttr, nullptr, &pingmesh);
    EXPECT_EQ(ret, HCCN_E_PARA);
    delete[] initAttr->ipAddr;
    delete initAttr;
}

TEST_F(HccnRping_UT, ut_HccnRpingInitWithCfg_NullRpingCtx)
{
    u32 devId = 1;
    HccnRpingInitAttr* initAttr = new HccnRpingInitAttr();
    initAttr->mode = HCCN_RPING_MODE_UB;
    initAttr->port = 13886;
    initAttr->npuNum = 128;
    initAttr->bufferSize = 4096 * 10;
    initAttr->sl = 0;
    initAttr->tc = 0;
    initAttr->ipAddr = new char[ipLen_];
    strcpy_s(initAttr->ipAddr, ipLen_, deviceIp_[devId]);
    HccnRpingInitCfg* initCfg = new HccnRpingInitCfg();
    initCfg->validCfgLen = sizeof(HccnRpingInitCfg);
    initCfg->ub.taMode = HCCN_RPING_UB_RC;
    initCfg->ub.tpMode = HCCN_RPING_UB_CTP;
    auto ret = HccnRpingInitWithCfg(devId, initAttr, initCfg, nullptr);
    EXPECT_EQ(ret, HCCN_E_PARA);
    delete[] initAttr->ipAddr;
    delete initAttr;
    delete initCfg;
}

TEST_F(HccnRping_UT, ut_HccnRpingInitWithCfg_InvalidMode)
{
    u32 devId = 1;
    HccnRpingInitAttr* initAttr = new HccnRpingInitAttr();
    initAttr->mode = PING_MODE_IPoUB;
    initAttr->port = 13886;
    initAttr->npuNum = 128;
    initAttr->bufferSize = 4096 * 10;
    initAttr->sl = 0;
    initAttr->tc = 0;
    initAttr->ipAddr = new char[ipLen_];
    strcpy_s(initAttr->ipAddr, ipLen_, deviceIp_[devId]);
    HccnRpingInitCfg* initCfg = new HccnRpingInitCfg();
    initCfg->validCfgLen = sizeof(HccnRpingInitCfg);
    initCfg->ub.taMode = HCCN_RPING_UB_RC;
    initCfg->ub.tpMode = HCCN_RPING_UB_CTP;
    void* pingmesh = nullptr;
    auto ret = HccnRpingInitWithCfg(devId, initAttr, initCfg, &pingmesh);
    EXPECT_EQ(ret, HCCN_E_PARA);
    delete[] initAttr->ipAddr;
    delete initAttr;
    delete initCfg;
}

TEST_F(HccnRping_UT, ut_HccnRpingInitWithCfg_InvalidTaMode)
{
    u32 devId = 1;
    HccnRpingInitAttr* initAttr = new HccnRpingInitAttr();
    initAttr->mode = HCCN_RPING_MODE_UB;
    initAttr->port = 13886;
    initAttr->npuNum = 128;
    initAttr->bufferSize = 4096 * 10;
    initAttr->sl = 0;
    initAttr->tc = 0;
    initAttr->ipAddr = new char[ipLen_];
    strcpy_s(initAttr->ipAddr, ipLen_, deviceIp_[devId]);
    HccnRpingInitCfg* initCfg = new HccnRpingInitCfg();
    initCfg->validCfgLen = sizeof(HccnRpingInitCfg);
    initCfg->ub.taMode = HCCN_RPING_UB_RM;
    initCfg->ub.tpMode = HCCN_RPING_UB_CTP;
    void* pingmesh = nullptr;
    auto ret = HccnRpingInitWithCfg(devId, initAttr, initCfg, &pingmesh);
    EXPECT_EQ(ret, HCCN_E_PARA);
    delete[] initAttr->ipAddr;
    delete initAttr;
    delete initCfg;
}

TEST_F(HccnRping_UT, ut_HccnRpingInitWithCfg_InvalidTpMode)
{
    u32 devId = 1;
    HccnRpingInitAttr* initAttr = new HccnRpingInitAttr();
    initAttr->mode = HCCN_RPING_MODE_UB;
    initAttr->port = 13886;
    initAttr->npuNum = 128;
    initAttr->bufferSize = 4096 * 10;
    initAttr->sl = 0;
    initAttr->tc = 0;
    initAttr->ipAddr = new char[ipLen_];
    strcpy_s(initAttr->ipAddr, ipLen_, deviceIp_[devId]);
    HccnRpingInitCfg* initCfg = new HccnRpingInitCfg();
    initCfg->validCfgLen = sizeof(HccnRpingInitCfg);
    initCfg->ub.taMode = HCCN_RPING_UB_RC;
    initCfg->ub.tpMode = static_cast<HccnUbTransportMode>(99);
    void* pingmesh = nullptr;
    auto ret = HccnRpingInitWithCfg(devId, initAttr, initCfg, &pingmesh);
    EXPECT_EQ(ret, HCCN_E_PARA);
    delete[] initAttr->ipAddr;
    delete initAttr;
    delete initCfg;
}

TEST_F(HccnRping_UT, ut_HccnRpingInitWithCfg_UM_NotUTP)
{
    u32 devId = 1;
    HccnRpingInitAttr* initAttr = new HccnRpingInitAttr();
    initAttr->mode = HCCN_RPING_MODE_UB;
    initAttr->port = 13886;
    initAttr->npuNum = 128;
    initAttr->bufferSize = 4096 * 10;
    initAttr->sl = 0;
    initAttr->tc = 0;
    initAttr->ipAddr = new char[ipLen_];
    strcpy_s(initAttr->ipAddr, ipLen_, deviceIp_[devId]);
    HccnRpingInitCfg* initCfg = new HccnRpingInitCfg();
    initCfg->validCfgLen = sizeof(HccnRpingInitCfg);
    initCfg->ub.taMode = HCCN_RPING_UB_UM;
    initCfg->ub.tpMode = HCCN_RPING_UB_CTP;
    void* pingmesh = nullptr;
    auto ret = HccnRpingInitWithCfg(devId, initAttr, initCfg, &pingmesh);
    EXPECT_EQ(ret, HCCN_E_PARA);
    delete[] initAttr->ipAddr;
    delete initAttr;
    delete initCfg;
}

TEST_F(HccnRping_UT, ut_HccnRpingInitWithCfg_DevIdMismatch)
{
    // hrtGetDeviceRefresh is mocked in SetUp to return 1; passing devId=2 triggers mismatch (HCCN_E_FAIL)
    u32 devId = 2;
    HccnRpingInitAttr* initAttr = new HccnRpingInitAttr();
    initAttr->mode = HCCN_RPING_MODE_UB;
    initAttr->port = 13886;
    initAttr->npuNum = 128;
    initAttr->bufferSize = 4096 * 10;
    initAttr->sl = 0;
    initAttr->tc = 0;
    initAttr->ipAddr = new char[ipLen_];
    strcpy_s(initAttr->ipAddr, ipLen_, deviceIp_[devId]);
    HccnRpingInitCfg* initCfg = new HccnRpingInitCfg();
    initCfg->validCfgLen = sizeof(HccnRpingInitCfg);
    initCfg->ub.taMode = HCCN_RPING_UB_RC;
    initCfg->ub.tpMode = HCCN_RPING_UB_CTP;
    void* pingmesh = nullptr;
    auto ret = HccnRpingInitWithCfg(devId, initAttr, initCfg, &pingmesh);
    EXPECT_EQ(ret, HCCN_E_FAIL);
    EXPECT_TRUE(pingmesh == nullptr);
    delete[] initAttr->ipAddr;
    delete initAttr;
    delete initCfg;
}

// ========== HccnRpingGetPayload RC-mode interception ==========

TEST_F(HccnRping_UT, ut_HccnRpingGetPayload_RC_Rejected)
{
    MOCKER_CPP(&PingMesh::GetDeviceLogicId).stubs().with(mockcpp::any()).will(returnValue(1));

    PingMesh* rping = new PingMesh();
    HccnRpingInitAttr initAttr = {};
    initAttr.mode = HCCN_RPING_MODE_UB;
    HccnRpingInitCfg initCfg = {};
    initCfg.validCfgLen = sizeof(HccnRpingInitCfg);
    initCfg.ub.taMode = HCCN_RPING_UB_RC;
    initCfg.ub.tpMode = HCCN_RPING_UB_CTP;
    rping->initWithCfg(&initAttr, &initCfg);

    HccnRpingCtx rpingCtx = rping;
    void* payloadOutput = nullptr;
    u32 payloadLenOutput = 0;
    auto ret = HccnRpingGetPayload(rpingCtx, &payloadOutput, &payloadLenOutput);
    EXPECT_EQ(ret, HCCN_E_FAIL);
    delete rping;
}

TEST_F(HccnRping_UT, ut_HccnRpingGetPayload_UM_NotRejected)
{
    MOCKER_CPP(&PingMesh::GetDeviceLogicId).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&PingMesh::HccnRpingGetPayload).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    PingMesh* rping = new PingMesh();
    HccnRpingInitAttr initAttr = {};
    initAttr.mode = HCCN_RPING_MODE_UB;
    HccnRpingInitCfg initCfg = {};
    initCfg.validCfgLen = sizeof(HccnRpingInitCfg);
    initCfg.ub.taMode = HCCN_RPING_UB_UM;
    initCfg.ub.tpMode = HCCN_RPING_UB_UTP;
    rping->initWithCfg(&initAttr, &initCfg);

    HccnRpingCtx rpingCtx = rping;
    void* payloadOutput = nullptr;
    u32 payloadLenOutput = 0;
    auto ret = HccnRpingGetPayload(rpingCtx, &payloadOutput, &payloadLenOutput);
    EXPECT_EQ(ret, HCCN_SUCCESS);
    delete rping;
}

// ========== HccnRpingInitWithCfg new param checks (validCfgLen / combo interception) ==========

// validCfgLen too small should return HCCN_E_PARA (forward-compat check, backup code lacked it)
TEST_F(HccnRping_UT, ut_HccnRpingInitWithCfg_ValidCfgLenTooSmall)
{
    u32 devId = 1;
    HccnRpingInitAttr* initAttr = new HccnRpingInitAttr();
    initAttr->mode = HCCN_RPING_MODE_UB;
    initAttr->port = 13886;
    initAttr->npuNum = 128;
    initAttr->bufferSize = 4096 * 10;
    initAttr->sl = 0;
    initAttr->tc = 0;
    initAttr->ipAddr = new char[ipLen_];
    strcpy_s(initAttr->ipAddr, ipLen_, deviceIp_[devId]);
    HccnRpingInitCfg* initCfg = new HccnRpingInitCfg();
    initCfg->validCfgLen = sizeof(uint32_t); // smaller than sizeof(HccnRpingInitCfg)
    initCfg->ub.taMode = HCCN_RPING_UB_RC;
    initCfg->ub.tpMode = HCCN_RPING_UB_CTP;
    void* pingmesh = nullptr;
    auto ret = HccnRpingInitWithCfg(devId, initAttr, initCfg, &pingmesh);
    EXPECT_EQ(ret, HCCN_E_PARA);
    EXPECT_TRUE(pingmesh == nullptr);
    delete[] initAttr->ipAddr;
    delete initAttr;
    delete initCfg;
}

// RC + UTP combo should return HCCN_E_PARA (RC only supports RTP/CTP)
TEST_F(HccnRping_UT, ut_HccnRpingInitWithCfg_Rc_Utp_Rejected)
{
    u32 devId = 1;
    HccnRpingInitAttr* initAttr = new HccnRpingInitAttr();
    initAttr->mode = HCCN_RPING_MODE_UB;
    initAttr->port = 13886;
    initAttr->npuNum = 128;
    initAttr->bufferSize = 4096 * 10;
    initAttr->sl = 0;
    initAttr->tc = 0;
    initAttr->ipAddr = new char[ipLen_];
    strcpy_s(initAttr->ipAddr, ipLen_, deviceIp_[devId]);
    HccnRpingInitCfg* initCfg = new HccnRpingInitCfg();
    initCfg->validCfgLen = sizeof(HccnRpingInitCfg);
    initCfg->ub.taMode = HCCN_RPING_UB_RC;
    initCfg->ub.tpMode = HCCN_RPING_UB_UTP;
    void* pingmesh = nullptr;
    auto ret = HccnRpingInitWithCfg(devId, initAttr, initCfg, &pingmesh);
    EXPECT_EQ(ret, HCCN_E_PARA);
    EXPECT_TRUE(pingmesh == nullptr);
    delete[] initAttr->ipAddr;
    delete initAttr;
    delete initCfg;
}

// old HccnRpingInit: mode=PING_MODE_IPoUB should return HCCN_E_PARA (g260915 adds rejection)
TEST_F(HccnRping_UT, ut_HccnRpingInit_IPoUB_Rejected)
{
    u32 devId = 1;
    HccnRpingInitAttr* initAttr = new HccnRpingInitAttr();
    initAttr->mode = PING_MODE_IPoUB;
    initAttr->port = 13886;
    initAttr->npuNum = 128;
    initAttr->bufferSize = 4096 * 10;
    initAttr->sl = 0;
    initAttr->tc = 0;
    initAttr->ipAddr = new char[ipLen_];
    strcpy_s(initAttr->ipAddr, ipLen_, deviceIp_[devId]);
    void* pingmesh = nullptr;
    auto ret = HccnRpingInit(devId, initAttr, &pingmesh);
    EXPECT_EQ(ret, HCCN_E_PARA);
    EXPECT_TRUE(pingmesh == nullptr);
    delete[] initAttr->ipAddr;
    delete initAttr;
}

// ROCE mode disallows custom taMode/tpMode: ROCE + taMode=RC -> HCCN_E_PARA (collaborator new)
TEST_F(HccnRping_UT, ut_HccnRpingInitWithCfg_RoCeNonDefaultTaMode_Rejected)
{
    u32 devId = 1;
    HccnRpingInitAttr* initAttr = new HccnRpingInitAttr();
    initAttr->mode = HCCN_RPING_MODE_ROCE;
    initAttr->port = 13886;
    initAttr->npuNum = 128;
    initAttr->bufferSize = 4096 * 10;
    initAttr->sl = 0;
    initAttr->tc = 0;
    initAttr->ipAddr = new char[ipLen_];
    strcpy_s(initAttr->ipAddr, ipLen_, deviceIp_[devId]);
    HccnRpingInitCfg* initCfg = new HccnRpingInitCfg();
    initCfg->validCfgLen = sizeof(HccnRpingInitCfg);
    initCfg->ub.taMode = HCCN_RPING_UB_RC; // ROCE mode should not set RC
    initCfg->ub.tpMode = HCCN_RPING_UB_CTP;
    void* pingmesh = nullptr;
    auto ret = HccnRpingInitWithCfg(devId, initAttr, initCfg, &pingmesh);
    EXPECT_EQ(ret, HCCN_E_PARA);
    EXPECT_TRUE(pingmesh == nullptr);
    delete[] initAttr->ipAddr;
    delete initAttr;
    delete initCfg;
}

// HccnRpingAddTargetWithCfg: RC mode allows only 1 target per add; targetNum!=1 -> HCCN_E_PARA
TEST_F(HccnRping_UT, ut_HccnRpingAddTargetWithCfg_RcMultiTarget_Rejected)
{
    MOCKER_CPP(&PingMesh::GetDeviceLogicId).stubs().with(mockcpp::any()).will(returnValue(1));
    PingMesh* rping = new PingMesh();
    HccnRpingInitAttr initAttr = {};
    initAttr.mode = HCCN_RPING_MODE_UB;
    HccnRpingInitCfg initCfg = {};
    initCfg.validCfgLen = sizeof(HccnRpingInitCfg);
    initCfg.ub.taMode = HCCN_RPING_UB_RC;
    initCfg.ub.tpMode = HCCN_RPING_UB_CTP;
    rping->initWithCfg(&initAttr, &initCfg);
    HccnRpingCtx rpingCtx = rping;

    HccnRpingTargetInfo target[2] = {0};
    HccnRpingAddTargetConfig config;
    config.connectTimeout = 1000;
    auto ret = HccnRpingAddTargetWithCfg(rpingCtx, 2, target, &config);
    EXPECT_EQ(ret, HCCN_E_PARA);
    delete rping;
}
