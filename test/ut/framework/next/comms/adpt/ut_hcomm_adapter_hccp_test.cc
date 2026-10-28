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
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>
#include "hcomm_adapter_hccp.h"
#include "adapter_error_manager_pub.h"
#include "hccp_async.h"
#include "hccp_common.h"
#include "hccp_tlv.h"

using namespace hcomm;
using namespace Hccl;

namespace {
constexpr uintptr_t FAKE_HCCP_CONTEXT = 0x12345678U;
constexpr uint32_t FAKE_UBOE_IPV4 = 0xC0A80367U;
constexpr uint8_t FAKE_UBOE_EID_LAST_BYTE = 0x67U;

std::string gCapturedErrorCode;
std::vector<std::string> gCapturedValues;

void StubRptInputErrCapture(std::string errorCode, std::vector<std::string> keys, std::vector<std::string> values)
{
    (void)keys;
    gCapturedErrorCode = errorCode;
    gCapturedValues = values;
}

int RaGetIpByEidSuccessStub(void* ctxHandle, union HccpEid eid[], struct IpInfo ip[], unsigned int* num)
{
    EXPECT_EQ(ctxHandle, reinterpret_cast<void*>(FAKE_HCCP_CONTEXT));
    EXPECT_NE(eid, nullptr);
    EXPECT_NE(ip, nullptr);
    EXPECT_NE(num, nullptr);
    EXPECT_EQ(*num, 1U);
    EXPECT_EQ(eid[0].raw[15], FAKE_UBOE_EID_LAST_BYTE);

    ip[0].family = AF_INET;
    ip[0].ip.addr.s_addr = htonl(FAKE_UBOE_IPV4);
    *num = 1U;
    return 0;
}
} // namespace

class HcommAdapterHccpTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HcommAdapterHccpTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "HcommAdapterHccpTest tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in HcommAdapterHccpTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
        std::cout << "A Test case in HcommAdapterHccpTest TearDown" << std::endl;
    }
};

TEST_F(HcommAdapterHccpTest, ut_HccpGetUboeFlagEnable_VersionEnough_Expect_Success)
{
    u32 devPhyId = 1;
    u32 mock_version = GET_UBOE_FLAG_ENABLE_VERSION;
    MOCKER(RaGetInterfaceVersion)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&mock_version, sizeof(s32)))
        .will(returnValue(0));

    HcclResult ret = HccpGetUboeFlagEnable(devPhyId);

    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcommAdapterHccpTest, ut_HccpGetUboeFlagEnable_When_VersionNotEnough_Expect_NotSupport)
{
    u32 devPhyId = 1;
    u32 mock_version = GET_UBOE_FLAG_ENABLE_VERSION - 1;
    MOCKER(RaGetInterfaceVersion)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&mock_version, sizeof(s32)))
        .will(returnValue(0));

    HcclResult ret = HccpGetUboeFlagEnable(devPhyId);

    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(HcommAdapterHccpTest, ut_HccpGetUboeFlagEnable_When_RaGetInterfaceFailed_Expect_E_INTERNAL)
{
    u32 devPhyId = 1;
    MOCKER(RaGetInterfaceVersion).stubs().will(returnValue(-1));
    HcclResult ret = HccpGetUboeFlagEnable(devPhyId);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(HcommAdapterHccpTest, ut_HccpCheckUboeSupported_When_DevFeatureBitSet_Expect_True)
{
    u32 devFeature = 1 << UBOE_DEV_FLAG_RIGHT_SHIFT;
    bool result = HccpCheckUboeSupported(devFeature);
    EXPECT_TRUE(result);

    // 测试其他位也有值的情况
    devFeature = (1 << UBOE_DEV_FLAG_RIGHT_SHIFT) | 0xFFFF;
    result = HccpCheckUboeSupported(devFeature);
    EXPECT_TRUE(result);
}

TEST_F(HcommAdapterHccpTest, ut_HccpCheckUboeSupported_When_DevFeatureBitNotSet_Expect_False)
{
    u32 devFeature = 0;
    bool result = HccpCheckUboeSupported(devFeature);
    EXPECT_FALSE(result);

    // 测试只有其他位被设置，但UBOE位未设置
    devFeature = 0xFFFFFFFF & ~(1 << UBOE_DEV_FLAG_RIGHT_SHIFT);
    result = HccpCheckUboeSupported(devFeature);
    EXPECT_FALSE(result);
}

TEST_F(HcommAdapterHccpTest, ut_HccpGetIpByEid_When_ValidEid_Expect_Ipv4)
{
    MOCKER(RaGetIpByEid).stubs().will(invoke(RaGetIpByEidSuccessStub));

    CommAddr eidAddr{};
    eidAddr.type = COMM_ADDR_TYPE_EID;
    eidAddr.eid[15] = FAKE_UBOE_EID_LAST_BYTE;
    CommAddr ipAddr{};

    HcclResult ret = HccpGetIpByEid(reinterpret_cast<void*>(FAKE_HCCP_CONTEXT), eidAddr, ipAddr);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(ipAddr.type, COMM_ADDR_TYPE_IP_V4);
    EXPECT_EQ(ipAddr.addr.s_addr, htonl(FAKE_UBOE_IPV4));
}

TEST_F(HcommAdapterHccpTest, ut_HccpGetIpByEid_When_RaQueryFails_Expect_E_NETWORK)
{
    MOCKER(RaGetIpByEid).stubs().will(returnValue(-1));

    CommAddr eidAddr{};
    eidAddr.type = COMM_ADDR_TYPE_EID;
    CommAddr ipAddr{};

    HcclResult ret = HccpGetIpByEid(reinterpret_cast<void*>(FAKE_HCCP_CONTEXT), eidAddr, ipAddr);

    EXPECT_EQ(ret, HCCL_E_NETWORK);
    EXPECT_EQ(ipAddr.type, COMM_ADDR_TYPE_RESERVED);
}

TEST_F(HcommAdapterHccpTest, ut_HccpGetCtpEnable_When_CtpExists_Expect_True)
{
    DevBaseAttr attr{};
    attr.ub.priorityInfo[0].tpType.bs.rtp = 1;
    attr.ub.priorityInfo[1].tpType.bs.ctp = 1;
    MOCKER(RaGetDevBaseAttr).stubs().with(mockcpp::any(), outBoundP(&attr, sizeof(attr))).will(returnValue(0));

    bool ctpEnable = false;
    EXPECT_EQ(HccpGetCtpEnable(reinterpret_cast<void*>(FAKE_HCCP_CONTEXT), ctpEnable), HCCL_SUCCESS);
    EXPECT_TRUE(ctpEnable);
}

TEST_F(HcommAdapterHccpTest, ut_HccpGetCtpEnable_When_AllRtp_Expect_False)
{
    DevBaseAttr attr{};
    for (uint32_t i = 0; i < MAX_PRIORITY_CNT; ++i) {
        attr.ub.priorityInfo[i].tpType.bs.rtp = 1;
    }
    MOCKER(RaGetDevBaseAttr).stubs().with(mockcpp::any(), outBoundP(&attr, sizeof(attr))).will(returnValue(0));

    bool ctpEnable = true;
    EXPECT_EQ(HccpGetCtpEnable(reinterpret_cast<void*>(FAKE_HCCP_CONTEXT), ctpEnable), HCCL_SUCCESS);
    EXPECT_FALSE(ctpEnable);
}

TEST_F(HcommAdapterHccpTest, ut_HccpGetCtpEnable_When_QueryFails_Expect_ENetworkAndFalse)
{
    MOCKER(RaGetDevBaseAttr).stubs().will(returnValue(-1));

    bool ctpEnable = true;
    EXPECT_EQ(HccpGetCtpEnable(reinterpret_cast<void*>(FAKE_HCCP_CONTEXT), ctpEnable), HCCL_E_NETWORK);
    EXPECT_FALSE(ctpEnable);
}

TEST_F(HcommAdapterHccpTest, ut_HccpGetCtpEnable_When_CtxHandleNull_Expect_EPtrAndFalse)
{
    bool ctpEnable = true;
    EXPECT_EQ(HccpGetCtpEnable(nullptr, ctpEnable), HCCL_E_PTR);
    EXPECT_FALSE(ctpEnable);
}

TEST_F(HcommAdapterHccpTest, Ut_HccpUbCreateJetty_When_ResourceExhausted_Expect_ReportEI0007)
{
    gCapturedErrorCode.clear();
    gCapturedValues.clear();
    MOCKER(RaCtxQpCreate).stubs().will(returnValue(ROCE_ENOSR));
    MOCKER(RptInputErr).stubs().will(invoke(StubRptInputErrCapture));
    HrtRaUbCreateJettyParam in;
    HrtRaUbJettyCreatedOutParam out;

    testing::internal::CaptureStdout();
    const HcclResult ret = HccpUbCreateJetty(reinterpret_cast<void*>(FAKE_HCCP_CONTEXT), in, out);
    const std::string log = testing::internal::GetCapturedStdout();

    EXPECT_EQ(ret, HCCL_E_NETWORK);
    EXPECT_EQ(gCapturedErrorCode, "EI0007");
    EXPECT_EQ(gCapturedValues, (std::vector<std::string>{"jetty", "CreateJetty"}));
    EXPECT_NE(log.find("[InitChannelStage][Resource]"), std::string::npos);
    EXPECT_NE(log.find("jetty resources are exhausted"), std::string::npos);
}

TEST_F(HcommAdapterHccpTest, Ut_HccpGetAsyncReqResult_When_JettyResourceExhausted_Expect_ReportEI0007)
{
    gCapturedErrorCode.clear();
    gCapturedValues.clear();
    AsyncReqResult asyncResult = {ROCE_ENOSR, HCCP_OP_CTX_QP_CREATE};
    MOCKER(RaGetAsyncReqResult)
        .stubs()
        .with(mockcpp::any(), outBoundP(&asyncResult, sizeof(asyncResult)))
        .will(returnValue(0));
    MOCKER(RptInputErr).stubs().will(invoke(StubRptInputErrCapture));
    RequestHandle reqHandle = 12;

    testing::internal::CaptureStdout();
    const RequestResult ret = HccpGetAsyncReqResult(reqHandle);
    const std::string log = testing::internal::GetCapturedStdout();

    EXPECT_EQ(ret, RequestResult::ASYNC_REQUEST_FAILED);
    EXPECT_EQ(gCapturedErrorCode, "EI0007");
    EXPECT_EQ(gCapturedValues, (std::vector<std::string>{"jetty", "CreateJettyAsync"}));
    EXPECT_NE(log.find("[InitChannelStage][Resource]"), std::string::npos);
    EXPECT_NE(log.find("jetty resources are exhausted"), std::string::npos);
}

TEST_F(HcommAdapterHccpTest, Ut_HccpGetAsyncReqResult_When_ReqHandleZero_Expect_ReturnInvalidPara)
{
    RequestHandle reqHandle = 0;

    const RequestResult ret = HccpGetAsyncReqResult(reqHandle);

    EXPECT_EQ(ret, RequestResult::INVALID_PARA);
}

TEST_F(HcommAdapterHccpTest, Ut_HccpGetAsyncReqResult_When_RaRetEagain_Expect_ReturnNotCompleted)
{
    MOCKER(RaGetAsyncReqResult).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(OTHERS_EAGAIN));
    RequestHandle reqHandle = 12;

    const RequestResult ret = HccpGetAsyncReqResult(reqHandle);

    EXPECT_EQ(ret, RequestResult::NOT_COMPLETED);
}

TEST_F(HcommAdapterHccpTest, Ut_HccpGetAsyncReqResult_When_RaRetError_Expect_ReturnGetReqResultFailed)
{
    AsyncReqResult asyncResult = {12345, 0x1234};
    MOCKER(RaGetAsyncReqResult)
        .stubs()
        .with(mockcpp::any(), outBoundP(&asyncResult, sizeof(asyncResult)))
        .will(returnValue(OTHERS_EAGAIN + 1));
    RequestHandle reqHandle = 12;

    testing::internal::CaptureStdout();
    const RequestResult ret = HccpGetAsyncReqResult(reqHandle);
    const std::string log = testing::internal::GetCapturedStdout();

    EXPECT_EQ(ret, RequestResult::GET_REQ_RESULT_FAILED);
    EXPECT_NE(log.find("interfaceOpCode[4660]"), std::string::npos);
}

TEST_F(HcommAdapterHccpTest, Ut_HccpGetAsyncReqResult_When_ReqResultSockEagain_Expect_ReturnSockEAgain)
{
    AsyncReqResult asyncResult = {SOCK_EAGAIN, 0};
    MOCKER(RaGetAsyncReqResult)
        .stubs()
        .with(mockcpp::any(), outBoundP(&asyncResult, sizeof(asyncResult)))
        .will(returnValue(0));
    RequestHandle reqHandle = 12;

    const RequestResult ret = HccpGetAsyncReqResult(reqHandle);

    EXPECT_EQ(ret, RequestResult::SOCK_E_AGAIN);
}

TEST_F(HcommAdapterHccpTest, Ut_HccpGetAsyncReqResult_When_ReqResultFailed_Expect_ReturnAsyncRequestFailed)
{
    AsyncReqResult asyncResult = {12345, 0x1234};
    MOCKER(RaGetAsyncReqResult)
        .stubs()
        .with(mockcpp::any(), outBoundP(&asyncResult, sizeof(asyncResult)))
        .will(returnValue(0));
    RequestHandle reqHandle = 12;

    testing::internal::CaptureStdout();
    const RequestResult ret = HccpGetAsyncReqResult(reqHandle);
    const std::string log = testing::internal::GetCapturedStdout();

    EXPECT_EQ(ret, RequestResult::ASYNC_REQUEST_FAILED);
    EXPECT_NE(log.find("interfaceOpCode[4660]"), std::string::npos);
}

TEST_F(HcommAdapterHccpTest, Ut_HccpGetAsyncReqResult_When_ReqResultOk_Expect_ReturnCompleted)
{
    AsyncReqResult asyncResult = {0, 0};
    MOCKER(RaGetAsyncReqResult)
        .stubs()
        .with(mockcpp::any(), outBoundP(&asyncResult, sizeof(asyncResult)))
        .will(returnValue(0));
    RequestHandle reqHandle = 12;

    const RequestResult ret = HccpGetAsyncReqResult(reqHandle);

    EXPECT_EQ(ret, RequestResult::COMPLETED);
}

TEST_F(HcommAdapterHccpTest, Ut_HccpUbCreateJetty_When_DevMemExhausted_Expect_ReportsEI0011)
{
    gCapturedErrorCode.clear();
    gCapturedValues.clear();
    MOCKER(RaCtxQpCreate).stubs().will(returnValue(ROCE_ENOMEM));
    MOCKER(RptInputErr).stubs().will(invoke(StubRptInputErrCapture));
    HrtRaUbCreateJettyParam in;
    HrtRaUbJettyCreatedOutParam out;

    testing::internal::CaptureStdout();
    const HcclResult ret = HccpUbCreateJetty(reinterpret_cast<void*>(FAKE_HCCP_CONTEXT), in, out);
    const std::string log = testing::internal::GetCapturedStdout();

    EXPECT_EQ(ret, HCCL_E_NETWORK);
    EXPECT_EQ(gCapturedErrorCode, "EI0011");
    EXPECT_EQ(gCapturedValues, (std::vector<std::string>{"16384~2097152"}));
    EXPECT_NE(log.find("insufficient memory"), std::string::npos);
}

TEST_F(HcommAdapterHccpTest, Ut_HccpGetAsyncReqResult_When_DevMemExhausted_Expect_ReportsEI0011)
{
    gCapturedErrorCode.clear();
    gCapturedValues.clear();
    AsyncReqResult asyncResult = {ROCE_ENOMEM, HCCP_OP_CTX_QP_CREATE};
    MOCKER(RaGetAsyncReqResult)
        .stubs()
        .with(mockcpp::any(), outBoundP(&asyncResult, sizeof(asyncResult)))
        .will(returnValue(0));
    MOCKER(RptInputErr).stubs().will(invoke(StubRptInputErrCapture));
    RequestHandle reqHandle = 12;

    testing::internal::CaptureStdout();
    const RequestResult ret = HccpGetAsyncReqResult(reqHandle);
    const std::string log = testing::internal::GetCapturedStdout();

    EXPECT_EQ(ret, RequestResult::ASYNC_REQUEST_FAILED);
    EXPECT_EQ(gCapturedErrorCode, "EI0011");
    EXPECT_EQ(gCapturedValues, (std::vector<std::string>{"16384~2097152"}));
    EXPECT_NE(log.find("insufficient memory"), std::string::npos);
}

TEST_F(HcommAdapterHccpTest, Ut_HccpTpImportJetty_When_ResourceExhausted_Expect_ReportsEI0007)
{
    gCapturedErrorCode.clear();
    gCapturedValues.clear();
    MOCKER(RaCtxQpImport).stubs().will(returnValue(ROCE_ENOSR));
    MOCKER(RptInputErr).stubs().will(invoke(StubRptInputErrCapture));
    u8 key[16] = {0};
    JettyImportCfg jettyImportCfg;
    jettyImportCfg.protocol = TpProtocol::RTP;
    HrtRaUbJettyImportedOutParam out;

    testing::internal::CaptureStdout();
    const HcclResult ret
        = HccpUbTpImportJetty(reinterpret_cast<void*>(FAKE_HCCP_CONTEXT), key, sizeof(key), 0, jettyImportCfg, out);
    const std::string log = testing::internal::GetCapturedStdout();

    EXPECT_EQ(ret, HCCL_E_NETWORK);
    EXPECT_EQ(gCapturedErrorCode, "EI0007");
    EXPECT_EQ(gCapturedValues, (std::vector<std::string>{"jetty", "ImportJetty"}));
    EXPECT_NE(log.find("jetty resources are exhausted"), std::string::npos);
}

TEST_F(HcommAdapterHccpTest, Ut_HccpGetAsyncReqResult_When_QpImportResourceExhausted_Expect_ReportsEI0007)
{
    gCapturedErrorCode.clear();
    gCapturedValues.clear();
    AsyncReqResult asyncResult = {ROCE_ENOSR, HCCP_OP_CTX_QP_IMPORT};
    MOCKER(RaGetAsyncReqResult)
        .stubs()
        .with(mockcpp::any(), outBoundP(&asyncResult, sizeof(asyncResult)))
        .will(returnValue(0));
    MOCKER(RptInputErr).stubs().will(invoke(StubRptInputErrCapture));
    RequestHandle reqHandle = 12;

    testing::internal::CaptureStdout();
    const RequestResult ret = HccpGetAsyncReqResult(reqHandle);
    const std::string log = testing::internal::GetCapturedStdout();

    EXPECT_EQ(ret, RequestResult::ASYNC_REQUEST_FAILED);
    EXPECT_EQ(gCapturedErrorCode, "EI0007");
    EXPECT_EQ(gCapturedValues, (std::vector<std::string>{"jetty", "ImportJettyAsync"}));
    EXPECT_NE(log.find("ImportJettyAsync failed"), std::string::npos);
    EXPECT_NE(log.find("jetty resources are exhausted"), std::string::npos);
}

TEST_F(HcommAdapterHccpTest, Ut_HccpGetAsyncReqResult_When_LmemRegResourceExhausted_Expect_ReportsEI0007)
{
    gCapturedErrorCode.clear();
    gCapturedValues.clear();
    AsyncReqResult asyncResult = {ROCE_ENOSR, HCCP_OP_LMEM_REG};
    MOCKER(RaGetAsyncReqResult)
        .stubs()
        .with(mockcpp::any(), outBoundP(&asyncResult, sizeof(asyncResult)))
        .will(returnValue(0));
    MOCKER(RptInputErr).stubs().will(invoke(StubRptInputErrCapture));
    RequestHandle reqHandle = 12;

    testing::internal::CaptureStdout();
    const RequestResult ret = HccpGetAsyncReqResult(reqHandle);
    const std::string log = testing::internal::GetCapturedStdout();

    EXPECT_EQ(ret, RequestResult::ASYNC_REQUEST_FAILED);
    EXPECT_EQ(gCapturedErrorCode, "EI0007");
    EXPECT_EQ(gCapturedValues, (std::vector<std::string>{"segment", "RegisterSegAsync"}));
    EXPECT_NE(log.find("RegisterSegAsync failed"), std::string::npos);
    EXPECT_NE(log.find("segment resources are exhausted"), std::string::npos);
}

TEST_F(HcommAdapterHccpTest, Ut_HccpGetAsyncReqResult_When_TpInfoListResourceExhausted_Expect_ReportsEI0007)
{
    gCapturedErrorCode.clear();
    gCapturedValues.clear();
    AsyncReqResult asyncResult = {ROCE_ENOSR, HCCP_OP_GET_TP_INFO_LIST};
    MOCKER(RaGetAsyncReqResult)
        .stubs()
        .with(mockcpp::any(), outBoundP(&asyncResult, sizeof(asyncResult)))
        .will(returnValue(0));
    MOCKER(RptInputErr).stubs().will(invoke(StubRptInputErrCapture));
    RequestHandle reqHandle = 12;

    testing::internal::CaptureStdout();
    const RequestResult ret = HccpGetAsyncReqResult(reqHandle);
    const std::string log = testing::internal::GetCapturedStdout();

    EXPECT_EQ(ret, RequestResult::ASYNC_REQUEST_FAILED);
    EXPECT_EQ(gCapturedErrorCode, "EI0007");
    EXPECT_EQ(gCapturedValues, (std::vector<std::string>{"tp", "GetTpListAsync"}));
    EXPECT_NE(log.find("GetTpListAsync failed"), std::string::npos);
    EXPECT_NE(log.find("tp resources are exhausted"), std::string::npos);
}

TEST_F(HcommAdapterHccpTest, Ut_HccpGetAsyncReqResult_When_LmemRegDevMemExhausted_Expect_ReportsEI0011)
{
    gCapturedErrorCode.clear();
    gCapturedValues.clear();
    AsyncReqResult asyncResult = {ROCE_ENOMEM, HCCP_OP_LMEM_REG};
    MOCKER(RaGetAsyncReqResult)
        .stubs()
        .with(mockcpp::any(), outBoundP(&asyncResult, sizeof(asyncResult)))
        .will(returnValue(0));
    MOCKER(RptInputErr).stubs().will(invoke(StubRptInputErrCapture));
    RequestHandle reqHandle = 12;

    testing::internal::CaptureStdout();
    const RequestResult ret = HccpGetAsyncReqResult(reqHandle);
    const std::string log = testing::internal::GetCapturedStdout();

    EXPECT_EQ(ret, RequestResult::ASYNC_REQUEST_FAILED);
    EXPECT_EQ(gCapturedErrorCode, "EI0011");
    EXPECT_EQ(gCapturedValues, (std::vector<std::string>{"N/A"}));
    EXPECT_NE(log.find("insufficient memory"), std::string::npos);
}

TEST_F(HcommAdapterHccpTest, Ut_HccpGetAsyncReqResult_When_NonExhaustedError_Expect_NoReport)
{
    gCapturedErrorCode.clear();
    gCapturedValues.clear();
    AsyncReqResult asyncResult = {ROCE_EINVAL, HCCP_OP_CTX_QP_IMPORT};
    MOCKER(RaGetAsyncReqResult)
        .stubs()
        .with(mockcpp::any(), outBoundP(&asyncResult, sizeof(asyncResult)))
        .will(returnValue(0));
    MOCKER(RptInputErr).stubs().will(invoke(StubRptInputErrCapture));
    RequestHandle reqHandle = 12;

    testing::internal::CaptureStdout();
    const RequestResult ret = HccpGetAsyncReqResult(reqHandle);
    testing::internal::GetCapturedStdout();

    EXPECT_EQ(ret, RequestResult::ASYNC_REQUEST_FAILED);
    EXPECT_TRUE(gCapturedErrorCode.empty());
}

// ========== HccpRaTlvRequestForCustomChannel 测试 ==========

TEST_F(HcommAdapterHccpTest, ut_HccpRaTlvRequestForCustomChannel_When_RaTlvRequestOk_Expect_ReturnHCCL_SUCCESS)
{
    void* tlvHandle = reinterpret_cast<void*>(0x1234);
    char inBuff[8] = {0};
    char outBuff[8] = {0};

    MOCKER(RaTlvRequest)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(0));

    HcclResult ret = HccpRaTlvRequestForCustomChannel(tlvHandle, MSG_TYPE_CCU_DISPATCH_CMD, inBuff, outBuff);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcommAdapterHccpTest, ut_HccpRaTlvRequestForCustomChannel_When_TlvHandleNull_Expect_ReturnHCCL_E_PTR)
{
    char inBuff[8] = {0};
    char outBuff[8] = {0};

    HcclResult ret = HccpRaTlvRequestForCustomChannel(nullptr, MSG_TYPE_CCU_DISPATCH_CMD, inBuff, outBuff);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcommAdapterHccpTest, ut_HccpRaTlvRequestForCustomChannel_When_CustomInNull_Expect_ReturnHCCL_E_PTR)
{
    void* tlvHandle = reinterpret_cast<void*>(0x1234);
    char outBuff[8] = {0};

    HcclResult ret = HccpRaTlvRequestForCustomChannel(tlvHandle, MSG_TYPE_CCU_DISPATCH_CMD, nullptr, outBuff);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcommAdapterHccpTest, ut_HccpRaTlvRequestForCustomChannel_When_CustomOutNull_Expect_ReturnHCCL_E_PTR)
{
    void* tlvHandle = reinterpret_cast<void*>(0x1234);
    char inBuff[8] = {0};

    HcclResult ret = HccpRaTlvRequestForCustomChannel(tlvHandle, MSG_TYPE_CCU_DISPATCH_CMD, inBuff, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcommAdapterHccpTest, ut_HccpRaTlvRequestForCustomChannel_When_RaTlvRequestFail_Expect_ReturnHCCL_E_NETWORK)
{
    void* tlvHandle = reinterpret_cast<void*>(0x1234);
    char inBuff[8] = {0};
    char outBuff[8] = {0};

    MOCKER(RaTlvRequest)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(-1));

    HcclResult ret = HccpRaTlvRequestForCustomChannel(tlvHandle, MSG_TYPE_CCU_DISPATCH_CMD, inBuff, outBuff);
    EXPECT_EQ(ret, HCCL_E_NETWORK);
}

// ========== HrtRaDumpJettyContext 测试 ==========

TEST_F(HcommAdapterHccpTest, Ut_HrtRaDumpJettyContext_When_JettyHandleIsNull_Expect_ReturnPtr)
{
    HcclResult ret = HrtRaDumpJettyContext(nullptr, 0);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcommAdapterHccpTest, Ut_HrtRaDumpJettyContext_When_RaCtxGetJettyContextFails_Expect_ReturnInternal)
{
    void* validHandle = reinterpret_cast<void*>(0x1234);
    MOCKER(RaCtxGetJettyContext).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any()).will(returnValue(-1));
    HcclResult ret = HrtRaDumpJettyContext(validHandle, 1);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(HcommAdapterHccpTest, Ut_HrtRaDumpJettyContext_When_ContextLenIsZero_Expect_ReturnInternal)
{
    void* validHandle = reinterpret_cast<void*>(0x1234);
    unsigned int mockLen = 0;
    MOCKER(RaCtxGetJettyContext)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&mockLen, sizeof(mockLen)))
        .will(returnValue(0));
    HcclResult ret = HrtRaDumpJettyContext(validHandle, 1);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(HcommAdapterHccpTest, Ut_HrtRaDumpJettyContext_When_ContextLenExceedsMax_Expect_ReturnInternal)
{
    void* validHandle = reinterpret_cast<void*>(0x1234);
    unsigned int mockLen = 600;
    MOCKER(RaCtxGetJettyContext)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&mockLen, sizeof(mockLen)))
        .will(returnValue(0));
    HcclResult ret = HrtRaDumpJettyContext(validHandle, 1);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(HcommAdapterHccpTest, Ut_HrtRaDumpJettyContext_When_Normal_Expect_ReturnSuccess)
{
    void* validHandle = reinterpret_cast<void*>(0x1234);
    unsigned int mockLen = 256;
    MOCKER(RaCtxGetJettyContext)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&mockLen, sizeof(mockLen)))
        .will(returnValue(0));
    HcclResult ret = HrtRaDumpJettyContext(validHandle, 7);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// ========== HccpNotifyDfxSnapshot 测试 ==========

static bool StubRaHasCapabilityFilter(RaInfo* raInfo, unsigned int capability, u32 allowPhyId)
{
    (void)capability;
    return raInfo != nullptr && raInfo->phyId == allowPhyId;
}

static bool StubRaHasCapabilityDev1(RaInfo* raInfo, unsigned int capability)
{
    return StubRaHasCapabilityFilter(raInfo, capability, 1);
}

static bool StubRaHasCapabilityDev1Or2(RaInfo* raInfo, unsigned int capability)
{
    (void)capability;
    return raInfo != nullptr && (raInfo->phyId == 1 || raInfo->phyId == 2);
}

void MockSnapshotNotify(bool capability, int retCode)
{
    MOCKER(RaHasCapability).stubs().will(returnValue(capability));
    MOCKER(RaCtxNotifyEvent).stubs().will(returnValue(retCode));
}

TEST_F(HcommAdapterHccpTest, Ut_HccpNotifyDfxSnapshot_When_StartAndEnd_Expect_Success)
{
    void* handle = reinterpret_cast<void*>(0x1234);
    // 仅用例设备(dev1)支持能力：残留设备(其他suite产生)的end遍历被能力检查拦截，不调RaCtxNotifyEvent
    MOCKER(RaHasCapability).stubs().will(invoke(StubRaHasCapabilityDev1));
    // 重复 start 去重契约：仅首次 start 与 end 各真实下发一次
    MOCKER(RaCtxNotifyEvent).expects(exactly(2)).will(returnValue(0));
    HccpResetDfxSnapshot(0);
    HccpResetDfxSnapshot(1);
    HccpResetDfxSnapshot(2);
    EXPECT_EQ(HccpNotifyDfxSnapshot(handle, 1, 0, true), HCCL_SUCCESS);
    // 重复 start 去重：仍成功但不重复发送
    EXPECT_EQ(HccpNotifyDfxSnapshot(handle, 1, 0, true), HCCL_SUCCESS);
    EXPECT_EQ(HccpNotifyDfxSnapshot(handle, 1, 0, false), HCCL_SUCCESS);
    HccpResetDfxSnapshot(0);
    HccpResetDfxSnapshot(1);
    HccpResetDfxSnapshot(2);
}

TEST_F(HcommAdapterHccpTest, Ut_HccpNotifyDfxSnapshot_When_EndWithoutStart_Expect_Success)
{
    void* handle = reinterpret_cast<void*>(0x1234);
    MockSnapshotNotify(true, 0);
    HccpResetDfxSnapshot(0);
    HccpResetDfxSnapshot(1);
    HccpResetDfxSnapshot(2);
    // 未开始直接结束：直接忽略
    EXPECT_EQ(HccpNotifyDfxSnapshot(handle, 1, 0, false), HCCL_SUCCESS);
    HccpResetDfxSnapshot(0);
    HccpResetDfxSnapshot(1);
    HccpResetDfxSnapshot(2);
}

TEST_F(HcommAdapterHccpTest, Ut_HccpNotifyDfxSnapshot_When_StartWithNullHandle_Expect_EPtr)
{
    MockSnapshotNotify(true, 0);
    HccpResetDfxSnapshot(0);
    HccpResetDfxSnapshot(1);
    HccpResetDfxSnapshot(2);
    EXPECT_EQ(HccpNotifyDfxSnapshot(nullptr, 1, 0, true), HCCL_E_PTR);
    HccpResetDfxSnapshot(0);
    HccpResetDfxSnapshot(1);
    HccpResetDfxSnapshot(2);
}

TEST_F(HcommAdapterHccpTest, Ut_HccpNotifyDfxSnapshot_When_NoCapability_Expect_SkipAndSuccess)
{
    void* handle = reinterpret_cast<void*>(0x1234);
    MOCKER(RaHasCapability).stubs().will(returnValue(false));
    // 能力不支持：未真正下发任何通知
    MOCKER(RaCtxNotifyEvent).expects(exactly(0));
    HccpResetDfxSnapshot(0);
    HccpResetDfxSnapshot(1);
    HccpResetDfxSnapshot(2);
    EXPECT_EQ(HccpNotifyDfxSnapshot(handle, 1, 0, true), HCCL_SUCCESS);
    // 能力不支持也视为已通知，重复 start 直接去重
    EXPECT_EQ(HccpNotifyDfxSnapshot(handle, 1, 0, true), HCCL_SUCCESS);
    // 能力不支持时未开始采集，结束直接忽略
    EXPECT_EQ(HccpNotifyDfxSnapshot(handle, 1, 0, false), HCCL_SUCCESS);
    HccpResetDfxSnapshot(0);
    HccpResetDfxSnapshot(1);
    HccpResetDfxSnapshot(2);
}

TEST_F(HcommAdapterHccpTest, Ut_HccpNotifyDfxSnapshot_When_NotifyEventFails_Expect_InternalAndNoRetry)
{
    void* handle = reinterpret_cast<void*>(0x1234);
    // mockcpp 同函数二次stubs不会覆盖首个桩，需用then串联首次失败与后续成功
    MOCKER(RaHasCapability).stubs().will(invoke(StubRaHasCapabilityDev1));
    MOCKER(RaCtxNotifyEvent).expects(exactly(1)).will(returnValue(-1));
    HccpResetDfxSnapshot(0);
    HccpResetDfxSnapshot(1);
    HccpResetDfxSnapshot(2);
    // 发送失败：仅通知一次，后续 start 直接去重不再发送
    EXPECT_EQ(HccpNotifyDfxSnapshot(handle, 1, 0, true), HCCL_E_INTERNAL);
    EXPECT_EQ(HccpNotifyDfxSnapshot(handle, 1, 0, true), HCCL_SUCCESS);
    // 未成功开始采集，结束直接忽略
    EXPECT_EQ(HccpNotifyDfxSnapshot(handle, 1, 0, false), HCCL_SUCCESS);
    HccpResetDfxSnapshot(0);
    HccpResetDfxSnapshot(1);
    HccpResetDfxSnapshot(2);
}

TEST_F(HcommAdapterHccpTest, Ut_HccpNotifyDfxSnapshot_When_EndNotifyFails_Expect_RetryOnNextEnd)
{
    void* handle = reinterpret_cast<void*>(0x1234);
    // start 成功（第 1 次）、end 失败（第 2 次，回滚去重登记并保留待结束账本）、end 重试成功（第 3 次）
    MOCKER(RaHasCapability).stubs().will(invoke(StubRaHasCapabilityDev1));
    MOCKER(RaCtxNotifyEvent).expects(exactly(3)).will(returnValue(0)).then(returnValue(-1)).then(returnValue(0));
    HccpResetDfxSnapshot(0);
    HccpResetDfxSnapshot(1);
    HccpResetDfxSnapshot(2);
    EXPECT_EQ(HccpNotifyDfxSnapshot(handle, 1, 0, true), HCCL_SUCCESS);
    // 结束发送失败：回滚去重登记，保留待结束账本，后续 end 可重试（不永久漏掉）
    EXPECT_EQ(HccpNotifyDfxSnapshot(handle, 1, 0, false), HCCL_E_INTERNAL);
    // 重试成功：再次 end 真实下发并收尾
    EXPECT_EQ(HccpNotifyDfxSnapshot(handle, 1, 0, false), HCCL_SUCCESS);
    // 已成功收尾，重复 end 直接忽略
    EXPECT_EQ(HccpNotifyDfxSnapshot(handle, 1, 0, false), HCCL_SUCCESS);
    HccpResetDfxSnapshot(0);
    HccpResetDfxSnapshot(1);
    HccpResetDfxSnapshot(2);
}

TEST_F(HcommAdapterHccpTest, Ut_HccpNotifyDfxSnapshot_When_MultiDieStart_Expect_NotifyEachDieOnce)
{
    void* handle = reinterpret_cast<void*>(0x1234);
    MOCKER(RaHasCapability).stubs().will(invoke(StubRaHasCapabilityDev1Or2));
    // 去重契约：两 die start + 两 die end + 其他设备 start/end 各一，共 6 次真实下发
    MOCKER(RaCtxNotifyEvent).expects(exactly(6)).will(returnValue(0));
    HccpResetDfxSnapshot(0);
    HccpResetDfxSnapshot(1);
    HccpResetDfxSnapshot(2);
    // 同一设备两个 die 各自通知一次
    EXPECT_EQ(HccpNotifyDfxSnapshot(handle, 1, 0, true), HCCL_SUCCESS);
    EXPECT_EQ(HccpNotifyDfxSnapshot(handle, 1, 1, true), HCCL_SUCCESS);
    // 同一 (设备, die) 重复 start 去重
    EXPECT_EQ(HccpNotifyDfxSnapshot(handle, 1, 0, true), HCCL_SUCCESS);
    // end 为进程级全局结束：覆盖本进程已开始的所有 die（此处含该设备两个 die）
    EXPECT_EQ(HccpNotifyDfxSnapshot(handle, 1, 0, false), HCCL_SUCCESS);
    // 全部结束后重复 end 直接忽略
    EXPECT_EQ(HccpNotifyDfxSnapshot(handle, 1, 0, false), HCCL_SUCCESS);
    // dev2 此前未开始采集，不受此前全局 end 影响，可独立开始
    EXPECT_EQ(HccpNotifyDfxSnapshot(handle, 2, 0, true), HCCL_SUCCESS);
    EXPECT_EQ(HccpNotifyDfxSnapshot(handle, 2, 0, false), HCCL_SUCCESS);
    HccpResetDfxSnapshot(0);
    HccpResetDfxSnapshot(1);
    HccpResetDfxSnapshot(2);
}

TEST_F(HcommAdapterHccpTest, Ut_HccpNotifyDfxSnapshot_When_CrossDeviceEnd_Expect_GlobalEndSemantics)
{
    void* handle = reinterpret_cast<void*>(0x1234);
    // dev1、dev2 均支持能力：end 为进程级全局结束，仅对 dev1 触发 end 时 dev2 一并被结束
    MOCKER(RaHasCapability).stubs().will(invoke(StubRaHasCapabilityDev1Or2));
    // 下发计数：dev1 start 1 + dev2 start 1 + dev1 end（全局遍历结束 dev1 与 dev2）2 + dev2 重复 end 0 = 4
    MOCKER(RaCtxNotifyEvent).expects(exactly(4)).will(returnValue(0));
    HccpResetDfxSnapshot(0);
    HccpResetDfxSnapshot(1);
    HccpResetDfxSnapshot(2);
    EXPECT_EQ(HccpNotifyDfxSnapshot(handle, 1, 0, true), HCCL_SUCCESS);
    EXPECT_EQ(HccpNotifyDfxSnapshot(handle, 2, 0, true), HCCL_SUCCESS);
    // 仅对 dev1 触发 end：全局语义下 dev2 的采集状态一并被结束（两笔 end 下发）
    EXPECT_EQ(HccpNotifyDfxSnapshot(handle, 1, 0, false), HCCL_SUCCESS);
    // dev2 已被全局 end 结束，重复 end 直接忽略
    EXPECT_EQ(HccpNotifyDfxSnapshot(handle, 2, 0, false), HCCL_SUCCESS);
    HccpResetDfxSnapshot(0);
    HccpResetDfxSnapshot(1);
    HccpResetDfxSnapshot(2);
}

TEST_F(HcommAdapterHccpTest, Ut_HccpNotifyDfxSnapshot_When_Reset_Expect_StateCleared)
{
    void* handle = reinterpret_cast<void*>(0x1234);
    MockSnapshotNotify(true, 0);
    HccpResetDfxSnapshot(0);
    HccpResetDfxSnapshot(1);
    HccpResetDfxSnapshot(2);
    EXPECT_EQ(HccpNotifyDfxSnapshot(handle, 1, 0, true), HCCL_SUCCESS);
    // reset 清空状态，可再次开始
    HccpResetDfxSnapshot(0);
    HccpResetDfxSnapshot(1);
    HccpResetDfxSnapshot(2);
    EXPECT_EQ(HccpNotifyDfxSnapshot(handle, 2, 0, true), HCCL_SUCCESS);
    EXPECT_EQ(HccpNotifyDfxSnapshot(handle, 2, 0, false), HCCL_SUCCESS);
    HccpResetDfxSnapshot(0);
    HccpResetDfxSnapshot(1);
    HccpResetDfxSnapshot(2);
}
