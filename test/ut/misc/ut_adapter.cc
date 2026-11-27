/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>

#include "network/hccp.h"
#include "dlra_function.h"
#include "adapter_rts_common.h"
#include "adapter_rts.h"

#define private public
#define protected public
#include "network_manager_pub.h"
#undef private

using namespace std;
using namespace hccl;

class HccpTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HccpTest SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "HccpTest TearDown" << std::endl; }
};

TEST_F(HccpTest, ut_hrtRdmaInitWithBackupAttr_notSupport)
{
    struct RdevInitInfo redvInitInfo;
    redvInitInfo.mode = 1;
    redvInitInfo.notifyType = 0;
    redvInitInfo.enabled910aLite = false;
    redvInitInfo.disabledLiteThread = false;
    redvInitInfo.enabled2mbLite = false;

    struct rdev rdevInfo;
    rdevInfo.phyId = 0;
    struct rdev backupRdevInfo;
    backupRdevInfo.phyId = 1;
    RdmaHandle rdmaHandle;

    MOCKER(hrtRaGetInterfaceVersion).expects(atMost(1)).will(returnValue(HCCL_E_NOT_SUPPORT));
    auto ret = HrtRdmaInitWithBackupAttr(redvInitInfo, rdevInfo, backupRdevInfo, rdmaHandle);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
    GlobalMockObject::verify();
}

TEST_F(HccpTest, ut_CheckAutoListenVersion)
{
    MOCKER(hrtRaGetInterfaceVersion).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));

    HcclResult ret = NetworkManager::GetInstance(0).CheckAutoListenVersion(true);
    EXPECT_EQ(ret, (ret, HCCL_E_NOT_SUPPORT));
    GlobalMockObject::verify();
}

TEST_F(HccpTest, Ut_CreateQp_SetQpAttrQosFail_Expect_DestoryQp)
{
    MOCKER(hrtRaGetInterfaceVersion).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    MOCKER(HrtRaQpCreate).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRaSetQpAttrQos).expects(atMost(1)).will(returnValue(HCCL_E_INTERNAL));
    MOCKER(hrtRaGetQpAttr).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    MOCKER(HrtRaQpDestroy).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));

    QpInfo qp;
    qp.qpHandle = reinterpret_cast<QpHandle>(0x1234);
    qp.trafficClass = 4;
    qp.serviceLevel = 0;
    qp.attr.maxWr = 128;
    qp.attr.maxSendSge = 1;
    qp.attr.maxRecvSge = 1;

    RdmaHandle rdmaHandle = reinterpret_cast<RdmaHandle>(0x1234);
    int flag = 0;
    s32 qpMode = 0;
    HcclResult ret = CreateQp(rdmaHandle, flag, qpMode, qp);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);

    GlobalMockObject::verify();
}

TEST_F(HccpTest, Ut_CreateQp_SetQpAttrTimeOutFail_Expect_DestoryQp)
{
    MOCKER(hrtRaGetInterfaceVersion).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    MOCKER(HrtRaQpCreate).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRaSetQpAttrQos).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRaSetQpAttrTimeOut).expects(atMost(1)).will(returnValue(HCCL_E_INTERNAL));
    MOCKER(hrtRaGetQpAttr).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    MOCKER(HrtRaQpDestroy).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));

    QpInfo qp;
    qp.qpHandle = reinterpret_cast<QpHandle>(0x1234);
    qp.trafficClass = 4;
    qp.serviceLevel = 0;
    qp.attr.maxWr = 128;
    qp.attr.maxSendSge = 1;
    qp.attr.maxRecvSge = 1;

    RdmaHandle rdmaHandle = reinterpret_cast<RdmaHandle>(0x1234);
    int flag = 0;
    s32 qpMode = 0;
    HcclResult ret = CreateQp(rdmaHandle, flag, qpMode, qp);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);

    GlobalMockObject::verify();
}

TEST_F(HccpTest, Ut_CreateQp_SetQpAttrRetryCntFail_Expect_DestoryQp)
{
    MOCKER(hrtRaGetInterfaceVersion).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    MOCKER(HrtRaQpCreate).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRaSetQpAttrQos).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRaSetQpAttrTimeOut).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRaSetQpAttrRetryCnt).expects(atMost(1)).will(returnValue(HCCL_E_INTERNAL));
    MOCKER(hrtRaGetQpAttr).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    MOCKER(HrtRaQpDestroy).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));

    QpInfo qp;
    qp.qpHandle = reinterpret_cast<QpHandle>(0x1234);
    qp.trafficClass = 4;
    qp.serviceLevel = 0;
    qp.attr.maxWr = 128;
    qp.attr.maxSendSge = 1;
    qp.attr.maxRecvSge = 1;

    RdmaHandle rdmaHandle = reinterpret_cast<RdmaHandle>(0x1234);
    int flag = 0;
    s32 qpMode = 0;
    HcclResult ret = CreateQp(rdmaHandle, flag, qpMode, qp);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);

    GlobalMockObject::verify();
}

TEST_F(HccpTest, Ut_CreateNormalQp_SetQpAttrTimeOutFail_Expect_DestoryQp)
{
    MOCKER(hrtRaGetInterfaceVersion).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRaNormalQpCreate).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRaSetQpAttrQos).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRaSetQpAttrTimeOut).expects(atMost(1)).will(returnValue(HCCL_E_INTERNAL));
    MOCKER(hrtRaGetQpAttr).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    MOCKER(HrtRaQpDestroy).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));

    QpInfo qp;
    qp.qpHandle = reinterpret_cast<QpHandle>(0x1234);
    qp.trafficClass = 4;
    qp.serviceLevel = 0;
    qp.attr.maxWr = 128;
    qp.attr.maxSendSge = 1;
    qp.attr.maxRecvSge = 1;

    RdmaHandle rdmaHandle = reinterpret_cast<RdmaHandle>(0x1234);
    HcclResult ret = CreateNormalQp(rdmaHandle, qp);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);

    GlobalMockObject::verify();
}

TEST_F(HccpTest, Ut_CreateNormalQp_SetQpAttrRetryCntFail_Expect_DestoryQp)
{
    MOCKER(hrtRaGetInterfaceVersion).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRaNormalQpCreate).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRaSetQpAttrQos).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRaSetQpAttrTimeOut).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRaSetQpAttrRetryCnt).expects(atMost(1)).will(returnValue(HCCL_E_INTERNAL));
    MOCKER(hrtRaGetQpAttr).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    MOCKER(HrtRaQpDestroy).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));

    QpInfo qp;
    qp.qpHandle = reinterpret_cast<QpHandle>(0x1234);
    qp.trafficClass = 4;
    qp.serviceLevel = 0;
    qp.attr.maxWr = 128;
    qp.attr.maxSendSge = 1;
    qp.attr.maxRecvSge = 1;

    RdmaHandle rdmaHandle = reinterpret_cast<RdmaHandle>(0x1234);
    HcclResult ret = CreateNormalQp(rdmaHandle, qp);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);

    GlobalMockObject::verify();
}

TEST_F(HccpTest, Ut_CreateCqAndQp_CreateNormalQpFail_Expect_DestoryCq)
{
    MOCKER(hrtRaGetInterfaceVersion).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRaNormalQpCreate).expects(atMost(1)).will(returnValue(HCCL_E_INTERNAL));
    MOCKER(hrtRaDestroyCq).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));

    RdmaHandle rdmaHandle = reinterpret_cast<RdmaHandle>(0x1234);
    string label = "test_label";
    HcclIpAddress selfIp("1.1.1.1");
    HcclIpAddress peerIp("2.2.2.2");
    QpConfig config(selfIp, peerIp, 128, 1, 1, 0, 0);
    QpInfo info;
    info.srq = nullptr;
    info.srqCq = nullptr;
    info.srqContext = nullptr;
    HcclResult ret = CreateCqAndQp(rdmaHandle, label, config, info);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);

    GlobalMockObject::verify();
}

TEST_F(HccpTest, Ut_CreateQpWithCq_CreateNormalQpFail_Expect_DestoryCq)
{
    MOCKER(hrtRaGetInterfaceVersion).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRaNormalQpCreate).expects(atMost(1)).will(returnValue(HCCL_E_INTERNAL));
    MOCKER(hrtRaDestroyCq).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));

    RdmaHandle rdmaHandle = reinterpret_cast<RdmaHandle>(0x1234);
    QpInfo info;
    info.srq = nullptr;
    info.srqCq = nullptr;
    info.srqContext = nullptr;
    info.trafficClass = 4;
    info.serviceLevel = 0;
    HcclResult ret = CreateQpWithCq(rdmaHandle, 0, 0, nullptr, nullptr, info, false, false);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);

    GlobalMockObject::verify();
}

TEST_F(HccpTest, Ut_CreateAiQp_SetQpAttrQosFail_Expect_DestoryQp)
{
    MOCKER(hrtGetDevice).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRaGetInterfaceVersion).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRaAiQpCreate).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRaSetQpAttrQos).expects(atMost(1)).will(returnValue(HCCL_E_INTERNAL));
    MOCKER(hrtRaGetQpAttr).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    MOCKER(HrtRaQpDestroy).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));

    RdmaHandle rdmaHandle = reinterpret_cast<RdmaHandle>(0x1234);
    struct AiQpInfo aiQpInfo;
    aiQpInfo.aiQpAddr = 0x1234;
    aiQpInfo.aiScqAddr = 0x5678;
    aiQpInfo.aiRcqAddr = 0x9abc;
    QpInfo info;
    info.qpHandle = nullptr;
    info.qpMode = 0;
    info.trafficClass = 4;
    info.serviceLevel = 0;
    u32 devicePhyId = 0;
    HcclResult ret = CreateAiQp(rdmaHandle, aiQpInfo, info, devicePhyId);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);

    GlobalMockObject::verify();
}

TEST_F(HccpTest, Ut_CreateAiQp_SetQpAttrTimeOutFail_Expect_DestoryQp)
{
    MOCKER(hrtGetDevice).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRaGetInterfaceVersion).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRaAiQpCreate).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRaSetQpAttrQos).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRaSetQpAttrTimeOut).expects(atMost(1)).will(returnValue(HCCL_E_INTERNAL));
    MOCKER(hrtRaGetQpAttr).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    MOCKER(HrtRaQpDestroy).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));

    RdmaHandle rdmaHandle = reinterpret_cast<RdmaHandle>(0x1234);
    struct AiQpInfo aiQpInfo;
    aiQpInfo.aiQpAddr = 0x1234;
    aiQpInfo.aiScqAddr = 0x5678;
    aiQpInfo.aiRcqAddr = 0x9abc;
    QpInfo info;
    info.qpHandle = nullptr;
    info.qpMode = 0;
    info.trafficClass = 4;
    info.serviceLevel = 0;
    u32 devicePhyId = 0;
    HcclResult ret = CreateAiQp(rdmaHandle, aiQpInfo, info, devicePhyId);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);

    GlobalMockObject::verify();
}

TEST_F(HccpTest, Ut_CreateAiQp_SetQpAttrRetoryCntFail_Expect_DestoryQp)
{
    MOCKER(hrtGetDevice).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRaGetInterfaceVersion).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRaAiQpCreate).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRaSetQpAttrQos).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRaSetQpAttrTimeOut).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRaSetQpAttrRetryCnt).expects(atMost(1)).will(returnValue(HCCL_E_INTERNAL));
    MOCKER(hrtRaGetQpAttr).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    MOCKER(HrtRaQpDestroy).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));

    RdmaHandle rdmaHandle = reinterpret_cast<RdmaHandle>(0x1234);
    struct AiQpInfo aiQpInfo;
    aiQpInfo.aiQpAddr = 0x1234;
    aiQpInfo.aiScqAddr = 0x5678;
    aiQpInfo.aiRcqAddr = 0x9abc;
    QpInfo info;
    info.qpHandle = nullptr;
    info.qpMode = 0;
    info.trafficClass = 4;
    info.serviceLevel = 0;
    u32 devicePhyId = 0;
    HcclResult ret = CreateAiQp(rdmaHandle, aiQpInfo, info, devicePhyId);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);

    GlobalMockObject::verify();
}

TEST_F(HccpTest, Ut_CreateAiQp_QpIsNullptr_Expect_DestoryQp)
{
    MOCKER(hrtGetDevice).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRaGetInterfaceVersion).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRaAiQpCreate).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRaSetQpAttrQos).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRaSetQpAttrTimeOut).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRaSetQpAttrRetryCnt).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    MOCKER(HrtRaQpDestroy).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));

    RdmaHandle rdmaHandle = reinterpret_cast<RdmaHandle>(0x1234);
    struct AiQpInfo aiQpInfo;
    aiQpInfo.aiQpAddr = 0;
    aiQpInfo.aiScqAddr = 0x5678;
    aiQpInfo.aiRcqAddr = 0x9abc;
    QpInfo info;
    info.qpHandle = nullptr;
    info.qpMode = 0;
    info.trafficClass = 4;
    info.serviceLevel = 0;
    u32 devicePhyId = 0;
    HcclResult ret = CreateAiQp(rdmaHandle, aiQpInfo, info, devicePhyId);
    EXPECT_EQ(ret, HCCL_E_PARA);

    GlobalMockObject::verify();
}

TEST_F(HccpTest, Ut_hrtStreamCreate_hrtStreamIdFail_Expect_DestoryStream)
{
    MOCKER(aclrtCreateStream).expects(atMost(1)).will(returnValue(ACL_SUCCESS));
    MOCKER(hrtGetStreamId).expects(atMost(1)).will(returnValue(HCCL_E_INTERNAL));
    MOCKER(aclrtDestroyStream).expects(atMost(1)).will(returnValue(ACL_SUCCESS));

    aclrtStream stream = nullptr;
    HcclResult ret = hrtStreamCreate(&stream);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);

    GlobalMockObject::verify();
}

TEST_F(HccpTest, Ut_hrtStreamCreateWithFlags_hrtGetStreamIdFail_Expect_DestoryStream)
{
    MOCKER(aclrtCreateStreamWithConfig).expects(atMost(1)).will(returnValue(ACL_SUCCESS));
    MOCKER(hrtGetStreamId).expects(atMost(1)).will(returnValue(HCCL_E_INTERNAL));
    MOCKER(aclrtDestroyStream).expects(atMost(1)).will(returnValue(ACL_SUCCESS));

    aclrtStream stream = nullptr;
    int32_t priority = 0;
    uint32_t flags = 0;
    HcclResult ret = hrtStreamCreateWithFlags(&stream, priority, flags);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);

    GlobalMockObject::verify();
}

TEST_F(HccpTest, Ut_hrtMemcpy_CountExceedsDestMax_Expect_ParaError)
{
    void* dst = malloc(100);
    void* src = malloc(100);
    uint64_t destMax = 50;
    uint64_t count = 100;
    HcclRtMemcpyKind kind = HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE;

    HcclResult ret = hrtMemcpy(dst, destMax, src, count, kind);
    EXPECT_EQ(ret, HCCL_E_PARA);

    free(dst);
    free(src);
}

// NetworkManager测试用例
TEST_F(HccpTest, Ut_NetworkManager_HeterogInit_SocketInitFail)
{
    s32 device_id = 0;
    u32 port = 16666;

    MOCKER(hrtRaSocketInit).stubs().will(returnValue(HCCL_E_TCP_CONNECT));

    HcclIpAddress ipAddr("127.0.0.1");
    HcclResult ret = NetworkManager::GetInstance(device_id).HeterogInit(device_id, ipAddr, port);
    EXPECT_NE(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HccpTest, Ut_NetworkManager_StopVnicSocketHandle_DeinitFail)
{
    s32 device_id = 0;
    u32 port = 16666;

    HcclResult ret = NetworkManager::GetInstance(device_id).Init(NICDeployment::NIC_DEPLOYMENT_DEVICE);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = NetworkManager::GetInstance(device_id).StartVnic(HcclIpAddress(device_id), port);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    MOCKER(hrtRaSocketInit).stubs().will(returnValue(HCCL_E_NETWORK));

    ret = NetworkManager::GetInstance(device_id).StopVnicSocketHandle(HcclIpAddress(device_id));
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = NetworkManager::GetInstance(device_id).Destroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HccpTest, Ut_NetworkManager_StopAllDeviceNicSockets_StopListenFail)
{
    s32 device_id = 0;
    u32 port = 16666;

    HcclResult ret = NetworkManager::GetInstance(device_id).Init(NICDeployment::NIC_DEPLOYMENT_DEVICE);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = NetworkManager::GetInstance(device_id).StartVnic(HcclIpAddress(device_id), port);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclIpAddress ipAddr("10.21.78.208");
    ret = NetworkManager::GetInstance(device_id).StartNic(ipAddr, port, false);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    MOCKER_CPP(&hccl::NetworkManager::StopNicsSocketListen).stubs().will(returnValue(HCCL_E_NETWORK));

    ret = NetworkManager::GetInstance(device_id).StopAllDeviceNicSockets();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = NetworkManager::GetInstance(device_id).Destroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HccpTest, Ut_NetworkManager_StopAllDeviceNicSockets_StopSocketFail)
{
    s32 device_id = 0;
    // 独占IP避免单例残留的in-use计数污染erase断言; 构造双map同handle状态
    HcclIpAddress ipAddr("10.77.77.77");
    auto& mgr = NetworkManager::GetInstance(device_id);
    SocketHandle fakeHandle = (SocketHandle)0x1234;
    mgr.raResourceInfo_.nicSocketMap[ipAddr].nicSocketHandle = fakeHandle;
    mgr.raResourceInfo_.hostNetSocketMap[ipAddr].nicSocketHandle = fakeHandle;

    // 内部调StopNicsSocketInternal注入失败; 失败分支兜底会真调hrtRaSocketDeInit, 一并mock
    MOCKER_CPP(&hccl::NetworkManager::StopNicsSocketInternal).stubs().will(returnValue(HCCL_E_NETWORK));
    MOCKER(hrtRaSocketDeInit).stubs().will(returnValue(HCCL_SUCCESS));

    EXPECT_EQ(mgr.StopAllDeviceNicSockets(), HCCL_SUCCESS);
    // 失败分支同步置空host侧副本, 条目留由host侧清理
    EXPECT_EQ(mgr.raResourceInfo_.hostNetSocketMap.count(ipAddr), 1U);
    EXPECT_EQ(mgr.raResourceInfo_.hostNetSocketMap[ipAddr].nicSocketHandle, nullptr);
    EXPECT_EQ(mgr.raResourceInfo_.nicSocketMap.count(ipAddr), 0U);

    mgr.raResourceInfo_.hostNetSocketMap.erase(ipAddr);
    GlobalMockObject::verify();
}

TEST_F(HccpTest, Ut_NetworkManager_StopAllDeviceVnicSockets_StopSocketFail)
{
    s32 device_id = 0;
    u32 port = 16666;

    HcclResult ret = NetworkManager::GetInstance(device_id).Init(NICDeployment::NIC_DEPLOYMENT_DEVICE);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = NetworkManager::GetInstance(device_id).StartVnic(HcclIpAddress(device_id), port);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    MOCKER_CPP(&hccl::NetworkManager::StopNicsSocket).stubs().will(returnValue(HCCL_E_NETWORK));

    ret = NetworkManager::GetInstance(device_id).StopAllDeviceVnicSockets();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = NetworkManager::GetInstance(device_id).Destroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HccpTest, Ut_NetworkManager_StopAllDeviceVnicSockets_DeinitFail)
{
    s32 device_id = 0;
    u32 port = 16666;

    HcclResult ret = NetworkManager::GetInstance(device_id).Init(NICDeployment::NIC_DEPLOYMENT_DEVICE);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = NetworkManager::GetInstance(device_id).StartVnic(HcclIpAddress(device_id), port);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    MOCKER(hrtRaSocketDeInit).stubs().will(returnValue(HCCL_E_NETWORK));

    ret = NetworkManager::GetInstance(device_id).StopAllDeviceVnicSockets();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = NetworkManager::GetInstance(device_id).Destroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HccpTest, Ut_NetworkManager_StopAllHostNicSockets_StopListenFail)
{
    s32 device_id = 0;
    u32 port = 16666;

    HcclResult ret = NetworkManager::GetInstance(device_id).Init(NICDeployment::NIC_DEPLOYMENT_HOST);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    HcclIpAddress ipAddr("127.0.0.1");
    ret = NetworkManager::GetInstance(device_id).StartHostNetAndListen(ipAddr, *(SocketHandle*)&ipAddr, port, false);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    MOCKER_CPP(&hccl::NetworkManager::StopListenSocket).stubs().will(returnValue(HCCL_E_NETWORK));

    ret = NetworkManager::GetInstance(device_id).StopAllHostNicSockets();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = NetworkManager::GetInstance(device_id).Destroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HccpTest, Ut_NetworkManager_StopAllHostNicSockets_DeinitFail)
{
    s32 device_id = 0;
    u32 port = 16666;

    HcclResult ret = NetworkManager::GetInstance(device_id).Init(NICDeployment::NIC_DEPLOYMENT_HOST);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    HcclIpAddress ipAddr("127.0.0.1");
    ret = NetworkManager::GetInstance(device_id).StartHostNetAndListen(ipAddr, *(SocketHandle*)&ipAddr, port, false);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    MOCKER(hrtRaSocketDeInit).stubs().will(returnValue(HCCL_E_NETWORK));

    ret = NetworkManager::GetInstance(device_id).StopAllHostNicSockets();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = NetworkManager::GetInstance(device_id).Destroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HccpTest, Ut_NetworkManager_PsWorkerRaInit_SocketInitFail)
{
    s32 device_id = 0;
    u32 port = 16666;

    MOCKER(hrtRaSocketInit).stubs().will(returnValue(HCCL_E_TCP_CONNECT));

    HcclIpAddress ipAddr("10.21.78.208");
    HcclResult ret
        = NetworkManager::GetInstance(device_id).PsWorkerRaInit(device_id, ipAddr, port, false, false, false);
    EXPECT_EQ(ret, HCCL_E_TCP_CONNECT);

    GlobalMockObject::verify();
}

TEST_F(HccpTest, Ut_NetworkManager_Destory_DeviceRaDeinitFail)
{
    s32 device_id = 0;
    u32 port = 16666;

    HcclResult ret = NetworkManager::GetInstance(device_id).Init(NICDeployment::NIC_DEPLOYMENT_DEVICE);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = NetworkManager::GetInstance(device_id).StartVnic(HcclIpAddress(device_id), port);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    NetworkManager::GetInstance(device_id).deviceNicInitRef_.Ref();
    NetworkManager::GetInstance(device_id).deviceNicInitRef_.Ref();

    MOCKER(HrtRaDeInit).stubs().will(returnValue(HCCL_E_NETWORK));

    ret = NetworkManager::GetInstance(device_id).Destroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HccpTest, Ut_NetworkManager_Destory_HostRaDeinitFail)
{
    s32 device_id = 0;
    u32 port = 16666;

    HcclResult ret = NetworkManager::GetInstance(device_id).Init(NICDeployment::NIC_DEPLOYMENT_HOST);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = NetworkManager::GetInstance(device_id).StartVnic(HcclIpAddress(device_id), port);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    NetworkManager::GetInstance(device_id).hostNicInitRef_.Ref();
    NetworkManager::GetInstance(device_id).hostNicInitRef_.Ref();

    MOCKER(HrtRaDeInit).stubs().will(returnValue(HCCL_E_NETWORK));

    ret = NetworkManager::GetInstance(device_id).Destroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}

// #1035: 直接构造内部状态, 每用例独立ip避免单例残留污染
TEST_F(HccpTest, Ut_NetworkManager_NicSocketUser_AddRemovePaired_EntryLifecycle)
{
    s32 device_id = 0;
    HcclIpAddress ipAddr("10.0.0.1");
    auto& mgr = NetworkManager::GetInstance(device_id);

    EXPECT_EQ(mgr.AddNicSocketUser(ipAddr), HCCL_SUCCESS);
    EXPECT_EQ(mgr.AddNicSocketUser(ipAddr), HCCL_SUCCESS);
    EXPECT_TRUE(mgr.MarkNicUserPendingStop(ipAddr, true, true));
    EXPECT_TRUE(mgr.nicSocketUserMap_[ipAddr].pendingStop);
    EXPECT_TRUE(mgr.nicSocketUserMap_[ipAddr].pendingRdmaStop);
    EXPECT_EQ(mgr.RemoveNicSocketUser(ipAddr), HCCL_SUCCESS);
    EXPECT_EQ(mgr.nicSocketUserMap_.count(ipAddr), 1U);
    EXPECT_EQ(mgr.RemoveNicSocketUser(ipAddr), HCCL_SUCCESS);
    EXPECT_EQ(mgr.nicSocketUserMap_.count(ipAddr), 0U);
}

TEST_F(HccpTest, Ut_NetworkManager_StopNicSocketHandle_InUse_DeferTeardown)
{
    s32 device_id = 0;
    HcclIpAddress ipAddr("10.0.0.2");
    auto& mgr = NetworkManager::GetInstance(device_id);
    SocketHandle fakeHandle = (SocketHandle)0x1234;
    mgr.raResourceInfo_.nicSocketMap[ipAddr].nicSocketHandle = fakeHandle;
    mgr.raResourceInfo_.hostNetSocketMap[ipAddr].nicSocketHandle = fakeHandle;

    EXPECT_EQ(mgr.AddNicSocketUser(ipAddr), HCCL_SUCCESS);
    MOCKER(hrtRaSocketDeInit).expects(once()).will(returnValue(HCCL_SUCCESS));
    EXPECT_EQ(mgr.StopNicSocketHandle(ipAddr), HCCL_SUCCESS);
    EXPECT_NE(mgr.raResourceInfo_.nicSocketMap[ipAddr].nicSocketHandle, nullptr);
    EXPECT_EQ(mgr.RemoveNicSocketUser(ipAddr), HCCL_SUCCESS);
    EXPECT_EQ(mgr.raResourceInfo_.nicSocketMap.count(ipAddr), 0U);
    EXPECT_EQ(mgr.raResourceInfo_.hostNetSocketMap[ipAddr].nicSocketHandle, nullptr);

    mgr.raResourceInfo_.hostNetSocketMap.erase(ipAddr);
    GlobalMockObject::verify();
}

TEST_F(HccpTest, Ut_NetworkManager_StopNicsSocket_DualMapSameHandle_SyncNullOut)
{
    s32 device_id = 0;
    HcclIpAddress ipAddr("10.0.0.9");
    auto& mgr = NetworkManager::GetInstance(device_id);
    SocketHandle fakeSocketHandle = (SocketHandle)0x1234;
    RdmaHandle fakeRdmaHandle = (RdmaHandle)0x5678;
    mgr.raResourceInfo_.nicSocketMap[ipAddr].nicSocketHandle = fakeSocketHandle;
    mgr.raResourceInfo_.nicSocketMap[ipAddr].nicRdmaHandle = fakeRdmaHandle;
    mgr.raResourceInfo_.hostNetSocketMap[ipAddr].nicSocketHandle = fakeSocketHandle;
    mgr.raResourceInfo_.hostNetSocketMap[ipAddr].nicRdmaHandle = fakeRdmaHandle;

    MOCKER(HrtRaRdmaDeInit).expects(once()).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRaSocketDeInit).expects(once()).will(returnValue(HCCL_SUCCESS));
    EXPECT_EQ(mgr.StopNicsSocket(ipAddr), HCCL_SUCCESS);
    EXPECT_EQ(mgr.raResourceInfo_.nicSocketMap[ipAddr].nicSocketHandle, nullptr);
    EXPECT_EQ(mgr.raResourceInfo_.nicSocketMap[ipAddr].nicRdmaHandle, nullptr);
    EXPECT_EQ(mgr.raResourceInfo_.hostNetSocketMap.count(ipAddr), 1U);
    EXPECT_EQ(mgr.raResourceInfo_.hostNetSocketMap[ipAddr].nicSocketHandle, nullptr);
    EXPECT_EQ(mgr.raResourceInfo_.hostNetSocketMap[ipAddr].nicRdmaHandle, nullptr);

    mgr.raResourceInfo_.nicSocketMap.erase(ipAddr);
    mgr.raResourceInfo_.hostNetSocketMap.erase(ipAddr);
    GlobalMockObject::verify();
}

TEST_F(HccpTest, Ut_NetworkManager_StopHostNetAndListen_KeepDeviceSideEntry)
{
    s32 device_id = 0;
    u32 port = 16666;
    HcclIpAddress ipAddr("10.0.0.7");
    auto& mgr = NetworkManager::GetInstance(device_id);
    mgr.hostNicInitRef_.Ref();
    SocketHandle fakeHostHandle = (SocketHandle)0x1234;
    mgr.raResourceInfo_.hostNetSocketMap[ipAddr].nicSocketHandle = fakeHostHandle;
    mgr.raResourceInfo_.hostNetSocketMap[ipAddr].listenedPort.insert(port);
    mgr.IPPortListenRefMapHost_[ipAddr][port].Ref();
    mgr.hostNicSocketClientRef_[ipAddr].Ref();
    SocketHandle devHandle = (SocketHandle)0x5678;
    mgr.raResourceInfo_.nicSocketMap[ipAddr].nicSocketHandle = devHandle;

    MOCKER_CPP(&hccl::NetworkManager::StopListenSocket).expects(once()).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRaSocketDeInit).expects(once()).will(returnValue(HCCL_SUCCESS));
    EXPECT_EQ(mgr.StopHostNetAndListen(fakeHostHandle, ipAddr, port), HCCL_SUCCESS);
    EXPECT_EQ(mgr.raResourceInfo_.hostNetSocketMap.count(ipAddr), 0U);
    EXPECT_EQ(mgr.raResourceInfo_.nicSocketMap[ipAddr].nicSocketHandle, devHandle);

    mgr.raResourceInfo_.nicSocketMap.erase(ipAddr);
    mgr.hostNicInitRef_.Clear();
    GlobalMockObject::verify();
}

TEST_F(HccpTest, Ut_NetworkManager_StopHostNetAndListen_AfterDeviceSideTeardown)
{
    s32 device_id = 0;
    u32 port = 16666;
    HcclIpAddress ipAddr("10.0.0.8");
    auto& mgr = NetworkManager::GetInstance(device_id);
    mgr.hostNicInitRef_.Ref();
    SocketHandle fakeHandle = (SocketHandle)0x1234;
    mgr.raResourceInfo_.nicSocketMap[ipAddr].nicSocketHandle = fakeHandle;
    mgr.raResourceInfo_.hostNetSocketMap[ipAddr].nicSocketHandle = fakeHandle;
    mgr.raResourceInfo_.hostNetSocketMap[ipAddr].listenedPort.insert(port);
    mgr.IPPortListenRefMapHost_[ipAddr][port].Ref();
    mgr.hostNicSocketClientRef_[ipAddr].Ref();

    MOCKER(hrtRaSocketDeInit).expects(once()).will(returnValue(HCCL_SUCCESS));
    EXPECT_EQ(mgr.StopNicsSocket(ipAddr), HCCL_SUCCESS);
    EXPECT_EQ(mgr.raResourceInfo_.hostNetSocketMap.count(ipAddr), 1U);
    EXPECT_EQ(mgr.raResourceInfo_.hostNetSocketMap[ipAddr].nicSocketHandle, nullptr);

    MOCKER_CPP(&hccl::NetworkManager::StopListenSocket).expects(never());
    EXPECT_EQ(mgr.StopHostNetAndListen(fakeHandle, ipAddr, port), HCCL_SUCCESS);
    EXPECT_EQ(mgr.raResourceInfo_.hostNetSocketMap.count(ipAddr), 0U);
    EXPECT_EQ(mgr.raResourceInfo_.nicSocketMap.count(ipAddr), 0U);

    mgr.hostNicInitRef_.Clear();
    GlobalMockObject::verify();
}

TEST_F(HccpTest, Ut_NetworkManager_StopHostNet_AfterDeviceSideTeardown)
{
    s32 device_id = 0;
    HcclIpAddress ipAddr("10.0.0.12");
    auto& mgr = NetworkManager::GetInstance(device_id);
    mgr.hostNicInitRef_.Ref();
    SocketHandle fakeHandle = (SocketHandle)0x1234;
    mgr.raResourceInfo_.nicSocketMap[ipAddr].nicSocketHandle = fakeHandle;
    mgr.raResourceInfo_.hostNetSocketMap[ipAddr].nicSocketHandle = fakeHandle;
    mgr.hostNicSocketClientRef_[ipAddr].Ref();

    MOCKER(hrtRaSocketDeInit).expects(once()).will(returnValue(HCCL_SUCCESS));
    EXPECT_EQ(mgr.StopNicsSocket(ipAddr), HCCL_SUCCESS);
    EXPECT_EQ(mgr.raResourceInfo_.hostNetSocketMap.count(ipAddr), 1U);
    EXPECT_EQ(mgr.raResourceInfo_.hostNetSocketMap[ipAddr].nicSocketHandle, nullptr);

    MOCKER(hrtRaSocketDeInit).expects(never());
    EXPECT_EQ(mgr.StopHostNet(fakeHandle, ipAddr), HCCL_SUCCESS);
    EXPECT_EQ(mgr.raResourceInfo_.hostNetSocketMap.count(ipAddr), 0U);
    EXPECT_EQ(mgr.raResourceInfo_.nicSocketMap.count(ipAddr), 0U);

    mgr.hostNicInitRef_.Clear();
    GlobalMockObject::verify();
}

TEST_F(HccpTest, Ut_NetworkManager_StopHostSocketHandle_AfterDeviceSideTeardown)
{
    s32 device_id = 0;
    u32 port = 16666;
    HcclIpAddress ipAddr("10.0.0.10");
    auto& mgr = NetworkManager::GetInstance(device_id);
    mgr.hostNicInitRef_.Ref();
    SocketHandle fakeHandle = (SocketHandle)0x1234;
    mgr.raResourceInfo_.nicSocketMap[ipAddr].nicSocketHandle = fakeHandle;
    mgr.raResourceInfo_.hostNetSocketMap[ipAddr].nicSocketHandle = fakeHandle;
    mgr.raResourceInfo_.hostNetSocketMap[ipAddr].listenedPort.insert(port);
    mgr.IPPortListenRefMapHost_[ipAddr][port].Ref();
    mgr.hostNicSocketClientRef_[ipAddr].Ref();

    MOCKER(hrtRaSocketDeInit).expects(once()).will(returnValue(HCCL_SUCCESS));
    EXPECT_EQ(mgr.StopNicsSocket(ipAddr), HCCL_SUCCESS);
    EXPECT_EQ(mgr.raResourceInfo_.hostNetSocketMap.count(ipAddr), 1U);
    EXPECT_EQ(mgr.raResourceInfo_.hostNetSocketMap[ipAddr].nicSocketHandle, nullptr);

    MOCKER_CPP(&hccl::NetworkManager::StopListenSocket).expects(never());
    EXPECT_EQ(mgr.StopHostSocketHandle(ipAddr), HCCL_SUCCESS);
    EXPECT_EQ(mgr.raResourceInfo_.hostNetSocketMap.count(ipAddr), 0U);
    EXPECT_EQ(mgr.raResourceInfo_.nicSocketMap.count(ipAddr), 0U);
    EXPECT_EQ(mgr.IPPortListenRefMapHost_[ipAddr][port].Count(), 0);

    mgr.IPPortListenRefMapHost_.erase(ipAddr);
    mgr.hostNicInitRef_.Clear();
    GlobalMockObject::verify();
}

TEST_F(HccpTest, Ut_NetworkManager_StopRdmaHandle_DualMapSameHandle_SyncNullOutDevice)
{
    s32 device_id = 0;
    HcclIpAddress ipAddr("10.0.0.13");
    auto& mgr = NetworkManager::GetInstance(device_id);
    RdmaHandle fakeRdmaHandle = (RdmaHandle)0x5678;
    mgr.raResourceInfo_.nicSocketMap[ipAddr].nicRdmaHandle = fakeRdmaHandle;
    mgr.raResourceInfo_.hostNetSocketMap[ipAddr].nicRdmaHandle = fakeRdmaHandle;

    MOCKER(HrtRaRdmaDeInit).expects(once()).will(returnValue(HCCL_SUCCESS));
    EXPECT_EQ(mgr.StopRdmaHandle(ipAddr, HcclNetDevDeployment::HCCL_NETDEV_DEPLOYMENT_DEVICE), HCCL_SUCCESS);
    EXPECT_EQ(mgr.raResourceInfo_.nicSocketMap.count(ipAddr), 0U);
    // 先断言count再取值: operator[]对缺失key会插入空条目, 掩盖误erase
    EXPECT_EQ(mgr.raResourceInfo_.hostNetSocketMap.count(ipAddr), 1U);
    EXPECT_EQ(mgr.raResourceInfo_.hostNetSocketMap[ipAddr].nicRdmaHandle, nullptr);

    mgr.raResourceInfo_.hostNetSocketMap.erase(ipAddr);
    GlobalMockObject::verify();
}

TEST_F(HccpTest, Ut_NetworkManager_StopRdmaHandle_DualMapSameHandle_SyncNullOutHost)
{
    s32 device_id = 0;
    HcclIpAddress ipAddr("10.0.0.14");
    auto& mgr = NetworkManager::GetInstance(device_id);
    RdmaHandle fakeRdmaHandle = (RdmaHandle)0x5678;
    mgr.raResourceInfo_.nicSocketMap[ipAddr].nicRdmaHandle = fakeRdmaHandle;
    mgr.raResourceInfo_.hostNetSocketMap[ipAddr].nicRdmaHandle = fakeRdmaHandle;

    MOCKER(HrtRaRdmaDeInit).expects(once()).will(returnValue(HCCL_SUCCESS));
    EXPECT_EQ(mgr.StopRdmaHandle(ipAddr, HcclNetDevDeployment::HCCL_NETDEV_DEPLOYMENT_HOST), HCCL_SUCCESS);
    EXPECT_EQ(mgr.raResourceInfo_.nicSocketMap.count(ipAddr), 0U);
    EXPECT_EQ(mgr.raResourceInfo_.hostNetSocketMap.count(ipAddr), 0U);

    GlobalMockObject::verify();
}

TEST_F(HccpTest, Ut_NetworkManager_StopRdmaHandle_KeepEntryWithAliveSocketHandle)
{
    s32 device_id = 0;
    HcclIpAddress ipAddr("10.0.0.15");
    auto& mgr = NetworkManager::GetInstance(device_id);
    RdmaHandle fakeRdmaHandle = (RdmaHandle)0x5678;
    SocketHandle fakeSocketHandle = (SocketHandle)0x1234;
    mgr.raResourceInfo_.nicSocketMap[ipAddr].nicRdmaHandle = fakeRdmaHandle;
    mgr.raResourceInfo_.hostNetSocketMap[ipAddr].nicRdmaHandle = fakeRdmaHandle;
    mgr.raResourceInfo_.hostNetSocketMap[ipAddr].nicSocketHandle = fakeSocketHandle;

    MOCKER(HrtRaRdmaDeInit).expects(once()).will(returnValue(HCCL_SUCCESS));
    EXPECT_EQ(mgr.StopRdmaHandle(ipAddr, HcclNetDevDeployment::HCCL_NETDEV_DEPLOYMENT_HOST), HCCL_SUCCESS);
    EXPECT_EQ(mgr.raResourceInfo_.nicSocketMap.count(ipAddr), 0U);
    EXPECT_EQ(mgr.raResourceInfo_.hostNetSocketMap.count(ipAddr), 1U);
    EXPECT_EQ(mgr.raResourceInfo_.hostNetSocketMap[ipAddr].nicRdmaHandle, nullptr);
    EXPECT_EQ(mgr.raResourceInfo_.hostNetSocketMap[ipAddr].nicSocketHandle, fakeSocketHandle);

    mgr.raResourceInfo_.hostNetSocketMap.erase(ipAddr);
    GlobalMockObject::verify();
}

TEST_F(HccpTest, Ut_NetworkManager_StopAllHostNicSockets_EmptyHandle_SkipPhysicalTeardown)
{
    s32 device_id = 0;
    HcclIpAddress ipAddr("10.0.0.11");
    auto& mgr = NetworkManager::GetInstance(device_id);
    mgr.raResourceInfo_.hostNetSocketMap[ipAddr].nicSocketHandle = nullptr;
    mgr.raResourceInfo_.hostNetSocketMap[ipAddr].nicRdmaHandle = nullptr;

    MOCKER_CPP(&hccl::NetworkManager::StopListenSocket).expects(never());
    MOCKER(hrtRaSocketDeInit).expects(never());
    EXPECT_EQ(mgr.StopAllHostNicSockets(), HCCL_SUCCESS);
    EXPECT_EQ(mgr.raResourceInfo_.hostNetSocketMap.count(ipAddr), 0U);

    GlobalMockObject::verify();
}

TEST_F(HccpTest, Ut_NetworkManager_RemoveNicSocketUser_Underflow_EntryStillCleared)
{
    s32 device_id = 0;
    HcclIpAddress ipAddr("10.0.0.3");
    auto& mgr = NetworkManager::GetInstance(device_id);
    EXPECT_EQ(mgr.AddNicSocketUser(ipAddr), HCCL_SUCCESS);
    mgr.nicSocketUserMap_[ipAddr].ref.Clear();
    EXPECT_EQ(mgr.RemoveNicSocketUser(ipAddr), HCCL_SUCCESS);
    EXPECT_EQ(mgr.nicSocketUserMap_.count(ipAddr), 0U);
}

TEST_F(HccpTest, Ut_NetworkManager_MarkNicUserPendingStop_FlagsByParameter)
{
    s32 device_id = 0;
    HcclIpAddress ipAddr("10.0.0.4");
    auto& mgr = NetworkManager::GetInstance(device_id);
    EXPECT_EQ(mgr.AddNicSocketUser(ipAddr), HCCL_SUCCESS);
    EXPECT_TRUE(mgr.MarkNicUserPendingStop(ipAddr, true, false));
    EXPECT_TRUE(mgr.nicSocketUserMap_[ipAddr].pendingStop);
    EXPECT_FALSE(mgr.nicSocketUserMap_[ipAddr].pendingRdmaStop);
    EXPECT_TRUE(mgr.MarkNicUserPendingStop(ipAddr, false, true));
    EXPECT_TRUE(mgr.nicSocketUserMap_[ipAddr].pendingStop);
    EXPECT_TRUE(mgr.nicSocketUserMap_[ipAddr].pendingRdmaStop);
    EXPECT_EQ(mgr.RemoveNicSocketUser(ipAddr), HCCL_SUCCESS);
    EXPECT_FALSE(mgr.MarkNicUserPendingStop(ipAddr, true, true));
}

TEST_F(HccpTest, Ut_NetworkManager_HeterogDeinit_InUse_SkipAndKeepRefcount)
{
    s32 device_id = 0;
    u32 port = 16666;
    HcclIpAddress ipAddr("10.0.0.5");
    auto& mgr = NetworkManager::GetInstance(device_id);
    EXPECT_EQ(mgr.AddNicSocketUser(ipAddr), HCCL_SUCCESS);
    s32 refBefore = mgr.hostNicInitRef_.Count();
    EXPECT_EQ(mgr.HeterogDeinit(device_id, ipAddr, port), HCCL_SUCCESS);
    EXPECT_EQ(mgr.hostNicInitRef_.Count(), refBefore);
    EXPECT_EQ(mgr.nicSocketUserMap_.count(ipAddr), 1U);
    EXPECT_EQ(mgr.RemoveNicSocketUser(ipAddr), HCCL_SUCCESS);
}

TEST_F(HccpTest, Ut_NetworkManager_PsWorkerRaDeinit_InUse_SkipAndKeepRefcount)
{
    s32 device_id = 0;
    u32 port = 16666;
    HcclIpAddress ipAddr("10.0.0.6");
    auto& mgr = NetworkManager::GetInstance(device_id);
    EXPECT_EQ(mgr.AddNicSocketUser(ipAddr), HCCL_SUCCESS);
    s32 hostRefBefore = mgr.hostNicInitRef_.Count();
    s32 devRefBefore = mgr.deviceNicInitRef_.Count();
    EXPECT_EQ(mgr.PsWorkerRaDeinit(device_id, ipAddr, port), HCCL_SUCCESS);
    EXPECT_EQ(mgr.hostNicInitRef_.Count(), hostRefBefore);
    EXPECT_EQ(mgr.deviceNicInitRef_.Count(), devRefBefore);
    EXPECT_EQ(mgr.nicSocketUserMap_.count(ipAddr), 1U);
    EXPECT_EQ(mgr.RemoveNicSocketUser(ipAddr), HCCL_SUCCESS);
}

TEST_F(HccpTest, Ut_NetworkManager_PsWorkerRaDeinit_EraseBothMaps_HostEntryAbsent)
{
    s32 device_id = 0;
    u32 port = 16666;
    HcclIpAddress ipAddr("10.88.88.88");
    auto& mgr = NetworkManager::GetInstance(device_id);
    mgr.raResourceInfo_.nicSocketMap[ipAddr].nicSocketHandle = (SocketHandle)0x1234;
    // 计数规整为1, Unref后归0走lastUsed=true完整拆除路径
    mgr.deviceNicInitRef_.Clear();
    mgr.deviceNicInitRef_.Ref();

    // mock真实副作用调用, 避免真实监听拆除与子进程关闭
    MOCKER(hrtRaIsLastUsed).stubs().with(mockcpp::any(), outBound(true)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hccl::NetworkManager::HeterogStopListenInternal).expects(once()).will(returnValue(HCCL_SUCCESS));
    MOCKER(HrtRaDeInit).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hccl::NetworkManager::CloseHccpSubProc).expects(once()).will(returnValue(HCCL_SUCCESS));

    EXPECT_EQ(mgr.PsWorkerRaDeinit(device_id, ipAddr, port), HCCL_SUCCESS);
    EXPECT_EQ(mgr.raResourceInfo_.nicSocketMap.count(ipAddr), 0U);
    EXPECT_EQ(mgr.raResourceInfo_.hostNetSocketMap.count(ipAddr), 0U);

    GlobalMockObject::verify();
}

TEST_F(HccpTest, ut_hrtMemcpyAddrAsync_When_FeatureNotSupport_Expect_NotSupport)
{
    MOCKER(aclrtMemcpyAsyncWithOffset).stubs().will(returnValue(ACL_ERROR_RT_FEATURE_NOT_SUPPORT));

    u32 dst = 0;
    u32 src = 0;
    HcclResult ret = hrtMemcpyAddrAsync(
        reinterpret_cast<void*>(&dst), sizeof(dst), 0, reinterpret_cast<const void*>(&src), sizeof(src), 0,
        reinterpret_cast<rtStream_t>(0x1));
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
    GlobalMockObject::verify();
}
