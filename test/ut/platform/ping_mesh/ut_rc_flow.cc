/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// PingMesh RC link setup/teardown UT for both server and client flows (PR #5346 UB_RC mode)
// Covers: RcHandleLinkReq / RcHandleGetLocalTp / RcHandleImportBind / RcHandleUnlink /
//         RcAttrGetLocalTp / RcAttrImportBind / HccnRcTargetAttrProcess /
//         RcClientUnlink / RcClientExchangeInfo
// Note: BuildPingPayLoadCmd / MapTpType are inline functions in ping_mesh.cc and cannot be
//       called directly; their normal path is covered indirectly via RcAttrGetLocalTp.

#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>
#include <thread>
#include <chrono>
#include "securec.h"
#define private public
#include "hccl_network_pub.h"
#include "network_manager_pub.h"
#include "ping_mesh.h"
#undef private
#include "adapter_hccp.h"
#include "adapter_rts.h"
#include "network/hccp_common.h"
#include "network/hccp_ping.h"

using namespace hccl;

// hrtRaPingTargetAdd controllable return value (per-case success/failure switch)
static HcclResult g_targetAddRet = HCCL_SUCCESS;
static HcclResult hrtRaPingTargetAddFlowStub(void* pingHandle, struct PingTargetInfo target[], uint32_t num)
{
    (void)pingHandle;
    (void)num;
    // Simulate device backfill of GET_LOCAL_TP: write tpHandle/tpPsn
    if (target != nullptr && target[0].version >= 1
        && target[0].targetAddAction == PING_TARGET_ADD_ACTION_GET_LOCAL_TP_INFO) {
        target[0].localInfo.ub.tpHandle = 0xAABBCCDD11223344ULL;
        target[0].localInfo.ub.tpPsn = 0x5678;
    }
    return g_targetAddRet;
}

static HcclResult hrtRaPingTargetDelFlowStub(void* pingHandle, struct PingTargetCommInfo target[], uint32_t num)
{
    (void)pingHandle;
    (void)target;
    (void)num;
    return HCCL_SUCCESS;
}

// Recv stub: fill protocol content by target length, simulating legal peer send
//  len==sizeof(RcMsgHeader)(6): fill msgType;  len==sizeof(RcExchangeInfoSendable): fill localTp/clientJetty
static u8 g_nextMsgType = static_cast<u8>(RcMsgType::LINK_REQ);
static u32 g_recvErrorCode = 0; // inject RcExchangeInfoSendable.clientJetty.errorCode
// Size of UbJettyId (ping_mesh.cc: eid[16]+uasid[4]+id[4]); RcClientExchangeInfo requires
// clientJetty.size >= this, so tests use size-1 to trigger the invalid-size branch.
static constexpr u32 g_ubJettyIdSize = 24;
static u32 g_recvJettySize = g_ubJettyIdSize; // inject RcExchangeInfoSendable.clientJetty.size
// header Recv sequence: return in order, fall back to g_nextMsgType when exhausted (LINK_REQ then UNLINK_REQ)
static u8 g_msgTypeSeq[8] = {0};
static int g_msgTypeSeqLen = 0;
static int g_msgTypeSeqIdx = 0;
static HcclResult HcclSocketRecvFlowStub(hccl::HcclSocket* self, void* recvBuf, uint32_t recvBufLen, uint32_t timeout)
{
    (void)self;
    (void)timeout;
    if (recvBuf == nullptr) {
        return HCCL_E_PTR;
    }
    (void)memset_s(recvBuf, recvBufLen, 0, recvBufLen);
    if (recvBufLen == sizeof(RcMsgHeader)) {
        RcMsgHeader* h = static_cast<RcMsgHeader*>(recvBuf);
        if (g_msgTypeSeqIdx < g_msgTypeSeqLen) {
            h->msgType = g_msgTypeSeq[g_msgTypeSeqIdx++];
        } else {
            h->msgType = g_nextMsgType;
        }
    } else if (recvBufLen == sizeof(RcExchangeInfoSendable)) {
        RcExchangeInfoSendable* b = static_cast<RcExchangeInfoSendable*>(recvBuf);
        b->localTp.tpHandle = 0x5566;
        b->localTp.tpPsn = 0x77;
        b->clientJetty.errorCode = g_recvErrorCode;
        b->clientJetty.size = g_recvJettySize;
    }
    return HCCL_SUCCESS;
}

class RcFlow_UT : public testing::Test {
protected:
    virtual void SetUp()
    {
        g_targetAddRet = HCCL_SUCCESS;
        g_recvErrorCode = 0;
        g_recvJettySize = 24;
        g_nextMsgType = static_cast<u8>(RcMsgType::LINK_REQ);
        g_msgTypeSeqLen = 0;
        g_msgTypeSeqIdx = 0;
        MOCKER(hrtRaPingTargetAdd).stubs().with(mockcpp::any()).will(invoke(hrtRaPingTargetAddFlowStub));
        MOCKER(hrtRaPingTargetDel).stubs().with(mockcpp::any()).will(invoke(hrtRaPingTargetDelFlowStub));
        MOCKER_CPP(&HcclSocket::Send, HcclResult(HcclSocket::*)(const void*, u64))
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(
            static_cast<HcclResult (hccl::HcclSocket::*)(void*, uint32_t, uint32_t)>(&hccl::HcclSocket::Recv),
            HcclResult(hccl::HcclSocket::*)(void*, uint32_t, uint32_t))
            .stubs()
            .with(mockcpp::any())
            .will(invoke(HcclSocketRecvFlowStub));
        MOCKER_CPP(&HcclSocket::GetStatus).stubs().with(mockcpp::any()).will(returnValue(HcclSocketStatus::SOCKET_OK));
        MOCKER_CPP(&HcclSocket::Init).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&HcclSocket::Connect).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    }

    virtual void TearDown() { GlobalMockObject::verify(); }
};

// RcHandleLinkReq: normal LINK_REQ, Recv ok -> HCCL_SUCCESS, abortSever unchanged
TEST_F(RcFlow_UT, ut_RcHandleLinkReq_Success)
{
    PingMesh rping;
    HcclSocket socket(nullptr, 0);
    struct RcServerLinkCtx ctx;
    (void)memset_s(&ctx, sizeof(ctx), 0, sizeof(ctx));
    bool abortSever = false;

    // SetUp already mocks Recv with HcclSocketRecvFlowStub, which writes msgType (default LINK_REQ)
    HcclResult ret = rping.RcHandleLinkReq(socket, ctx, abortSever);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_FALSE(abortSever);
}

// RcHandleGetLocalTp: TargetAdd ok -> backfill localTpHandle_/localTpPsn_
TEST_F(RcFlow_UT, ut_RcHandleGetLocalTp_Success)
{
    PingMesh rping;
    struct RcServerLinkCtx ctx;
    (void)memset_s(&ctx, sizeof(ctx), 0, sizeof(ctx));
    bool abortSever = false;

    HcclResult ret = rping.RcHandleGetLocalTp(ctx, abortSever);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_FALSE(abortSever);
    EXPECT_EQ(rping.localTpHandle_, 0xAABBCCDD11223344ULL);
    EXPECT_EQ(rping.localTpPsn_, 0x5678U);
}

// RcHandleGetLocalTp: TargetAdd fails -> return error, abortSever unchanged (recoverable)
TEST_F(RcFlow_UT, ut_RcHandleGetLocalTp_TargetAddFail)
{
    g_targetAddRet = HCCL_E_INTERNAL;
    PingMesh rping;
    struct RcServerLinkCtx ctx;
    (void)memset_s(&ctx, sizeof(ctx), 0, sizeof(ctx));
    bool abortSever = false;

    HcclResult ret = rping.RcHandleGetLocalTp(ctx, abortSever);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
    // Collaborator version: GET_TP query failure is recoverable, no abortSever (thread keeps serving)
    EXPECT_FALSE(abortSever);
}

// RcHandleImportBind: TargetAdd ok -> remoteTpHandle_ taken from recvBuf
TEST_F(RcFlow_UT, ut_RcHandleImportBind_Success)
{
    PingMesh rping;
    HcclSocket socket(nullptr, 0);
    struct RcServerLinkCtx ctx;
    (void)memset_s(&ctx, sizeof(ctx), 0, sizeof(ctx));
    ctx.recvBuf.localTp.tpHandle = 0x1234;
    ctx.recvBuf.localTp.tpPsn = 0x99;
    bool abortSever = false;

    HcclResult ret = rping.RcHandleImportBind(socket, ctx, abortSever);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(rping.remoteTpHandle_, 0x1234ULL);
    EXPECT_EQ(rping.remoteTpPsn_, 0x99U);
}

// RcHandleImportBind: TargetAdd fails -> abortSever=true
TEST_F(RcFlow_UT, ut_RcHandleImportBind_TargetAddFail)
{
    g_targetAddRet = HCCL_E_INTERNAL;
    PingMesh rping;
    HcclSocket socket(nullptr, 0);
    struct RcServerLinkCtx ctx;
    (void)memset_s(&ctx, sizeof(ctx), 0, sizeof(ctx));
    bool abortSever = false;

    HcclResult ret = rping.RcHandleImportBind(socket, ctx, abortSever);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
    EXPECT_TRUE(abortSever);
}

// RcHandleUnlink: normal teardown (TargetDel ok) -> HCCL_SUCCESS
TEST_F(RcFlow_UT, ut_RcHandleUnlink_Success)
{
    g_nextMsgType = static_cast<u8>(RcMsgType::UNLINK_REQ);
    PingMesh rping;
    HcclSocket socket(nullptr, 0);
    struct RcServerLinkCtx ctx;
    (void)memset_s(&ctx, sizeof(ctx), 0, sizeof(ctx));
    bool abortSever = false;

    HcclResult ret = rping.RcHandleUnlink(socket, ctx, abortSever);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_FALSE(abortSever);
    g_nextMsgType = static_cast<u8>(RcMsgType::LINK_REQ);
}

// RcHandleUnlink: peer illegal msgType (non-UNLINK_REQ) -> abortSever=true, return HCCL_E_PARA
TEST_F(RcFlow_UT, ut_RcHandleUnlink_InvalidMsgType)
{
    g_nextMsgType = static_cast<u8>(RcMsgType::LINK_REQ); // not UNLINK_REQ
    PingMesh rping;
    HcclSocket socket(nullptr, 0);
    struct RcServerLinkCtx ctx;
    (void)memset_s(&ctx, sizeof(ctx), 0, sizeof(ctx));
    bool abortSever = false;

    HcclResult ret = rping.RcHandleUnlink(socket, ctx, abortSever);
    // Collaborator version: illegal msgType uniformly returns HCCL_E_PARA + abortSever
    EXPECT_EQ(ret, HCCL_E_PARA);
    EXPECT_TRUE(abortSever);
    g_nextMsgType = static_cast<u8>(RcMsgType::LINK_REQ);
}

// RcClientUnlink: socket is null -> HCCL_E_PARA
TEST_F(RcFlow_UT, ut_RcClientUnlink_NullSocket)
{
    PingMesh rping;
    RpingInput input;
    std::shared_ptr<HcclSocket> nullSocket = nullptr;

    HcclResult ret = rping.RcClientUnlink(&input, nullSocket);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// RcClientUnlink: socket status not OK (TIMEOUT) and not CONNECTING -> HCCL_E_TCP_CONNECT
// Note: SetUp registers a SOCKET_OK stub for GetStatus; mockcpp matches in registration order, must reset first to
// override
TEST_F(RcFlow_UT, ut_RcClientUnlink_StatusNotOk)
{
    MOCKER_CPP(&HcclSocket::GetStatus).reset();
    MOCKER_CPP(&HcclSocket::GetStatus).stubs().with(mockcpp::any()).will(returnValue(HcclSocketStatus::SOCKET_TIMEOUT));
    PingMesh rping;
    RpingInput input;
    std::shared_ptr<HcclSocket> socket = std::make_shared<HcclSocket>(nullptr, 0);

    HcclResult ret = rping.RcClientUnlink(&input, socket);
    // First while-loop GetStatus=TIMEOUT (not CONNECTING/not OK) -> immediately return E_TCP_CONNECT(1653)
    EXPECT_EQ(ret, HCCL_E_TCP_CONNECT);
}

// RcAttrGetLocalTp: TargetAdd ok -> record localTpHandle_/localTpPsn_
TEST_F(RcFlow_UT, ut_RcAttrGetLocalTp_Success)
{
    PingMesh rping;
    RpingInput input;
    PingTargetInfo targetInter;
    (void)memset_s(&targetInter, sizeof(targetInter), 0, sizeof(targetInter));

    HcclResult ret = rping.RcAttrGetLocalTp(&input, &targetInter);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(rping.localTpHandle_, 0xAABBCCDD11223344ULL);
    EXPECT_EQ(targetInter.targetAddAction, PING_TARGET_ADD_ACTION_GET_LOCAL_TP_INFO);
}

// RcAttrGetLocalTp: TargetAdd fails -> return error
TEST_F(RcFlow_UT, ut_RcAttrGetLocalTp_TargetAddFail)
{
    g_targetAddRet = HCCL_E_INTERNAL;
    PingMesh rping;
    RpingInput input;
    PingTargetInfo targetInter;
    (void)memset_s(&targetInter, sizeof(targetInter), 0, sizeof(targetInter));

    HcclResult ret = rping.RcAttrGetLocalTp(&input, &targetInter);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

// RcAttrImportBind: TargetAdd ok -> rpingTargetNum_ incremented
TEST_F(RcFlow_UT, ut_RcAttrImportBind_Success)
{
    PingMesh rping;
    RpingInput input;
    PingTargetInfo targetInter;
    (void)memset_s(&targetInter, sizeof(targetInter), 0, sizeof(targetInter));
    PingInitInfo recvInfo;
    (void)memset_s(&recvInfo, sizeof(recvInfo), 0, sizeof(recvInfo));
    u32 before = rping.rpingTargetNum_;

    HcclResult ret = rping.RcAttrImportBind(&input, &targetInter, &recvInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(rping.rpingTargetNum_, before + 1);
    EXPECT_EQ(targetInter.targetAddAction, PING_TARGET_ADD_ACTION_DEFAULT);
}

// RcAttrImportBind: TargetAdd fails -> return error, count unchanged
TEST_F(RcFlow_UT, ut_RcAttrImportBind_TargetAddFail)
{
    g_targetAddRet = HCCL_E_INTERNAL;
    PingMesh rping;
    RpingInput input;
    PingTargetInfo targetInter;
    (void)memset_s(&targetInter, sizeof(targetInter), 0, sizeof(targetInter));
    PingInitInfo recvInfo;
    (void)memset_s(&recvInfo, sizeof(recvInfo), 0, sizeof(recvInfo));
    u32 before = rping.rpingTargetNum_;

    HcclResult ret = rping.RcAttrImportBind(&input, &targetInter, &recvInfo);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
    EXPECT_EQ(rping.rpingTargetNum_, before);
}

// RcClientExchangeInfo: server returns errorCode!=0 -> HCCL_E_PARA
TEST_F(RcFlow_UT, ut_RcClientExchangeInfo_ServerError)
{
    g_recvErrorCode = 1; // server errorCode non-zero
    PingMesh rping;
    std::shared_ptr<HcclSocket> socket = std::make_shared<HcclSocket>(nullptr, 0);
    HcclIpAddress ipAddr(0x7F000001);
    PingInitInfo recvInfo;
    (void)memset_s(&recvInfo, sizeof(recvInfo), 0, sizeof(recvInfo));

    HcclResult ret = rping.RcClientExchangeInfo(socket, ipAddr, recvInfo);
    EXPECT_EQ(ret, HCCL_E_PARA);
    g_recvErrorCode = 0;
}

// RcClientExchangeInfo: normal exchange -> remoteTpHandle_/remoteTpPsn_ from server reply
TEST_F(RcFlow_UT, ut_RcClientExchangeInfo_Success)
{
    PingMesh rping;
    std::shared_ptr<HcclSocket> socket = std::make_shared<HcclSocket>(nullptr, 0);
    HcclIpAddress ipAddr(0x7F000001);
    PingInitInfo recvInfo;
    (void)memset_s(&recvInfo, sizeof(recvInfo), 0, sizeof(recvInfo));

    HcclResult ret = rping.RcClientExchangeInfo(socket, ipAddr, recvInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(rping.remoteTpHandle_, 0x5566ULL);
    EXPECT_EQ(rping.remoteTpPsn_, 0x77U);
}

// HccnRcTargetAttrProcess: RcAttrGetLocalTp fails -> pass error through (orchestration short-circuit)
TEST_F(RcFlow_UT, ut_HccnRcTargetAttrProcess_GetLocalTpFail)
{
    g_targetAddRet = HCCL_E_INTERNAL;
    PingMesh rping;
    RpingInput input;
    PingTargetInfo targetInter;
    (void)memset_s(&targetInter, sizeof(targetInter), 0, sizeof(targetInter));

    HcclResult ret = rping.HccnRcTargetAttrProcess(&input, 1000, HCCN_RPING_ADDR_TYPE_IP, &targetInter);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

// ========== RcAttrRecvAndFill branches (collaborator version) ==========
// Precondition: socketMaps_ pre-populated IP -> socket, so RpingRecvTargetInfo first step succeeds directly (no link)
static HcclIpAddress MakeInputIp()
{
    return HcclIpAddress(0x7F000001); // 127.0.0.1
}
static void PrepareSocketMap(PingMesh& rping, const HcclIpAddress& ip)
{
    rping.socketMaps_.insert({std::string(ip.GetReadableIP()), std::make_shared<HcclSocket>(nullptr, 0)});
}

// Branch1: RpingRecvTargetInfo fails (no IP in socketMaps_, internal link fails) -> return error
TEST_F(RcFlow_UT, ut_RcAttrRecvAndFill_RecvFail)
{
    // Make RpingRecvTargetInfo link fail (Connect returns error), verify error pass-through
    MOCKER_CPP(&HcclSocket::Connect).reset();
    MOCKER_CPP(&HcclSocket::Connect).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_TCP_CONNECT));
    PingMesh rping;
    RpingInput input;
    // input.dip not pre-set in socketMaps_ => RpingRecvTargetInfo does internal link, Connect fails -> error
    PingTargetInfo targetInter;
    (void)memset_s(&targetInter, sizeof(targetInter), 0, sizeof(targetInter));
    PingInitInfo recvInfo;
    bool skipImportBind = false;

    HcclResult ret
        = rping.RcAttrRecvAndFill(&input, 1000, HCCN_RPING_ADDR_TYPE_IP, &targetInter, &recvInfo, &skipImportBind);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

// Branch2: rdmaInfoMaps_ already exists -> skipImportBind=true
TEST_F(RcFlow_UT, ut_RcAttrRecvAndFill_SkipOnRdmaInfoHit)
{
    PingMesh rping;
    HcclIpAddress ip = MakeInputIp();
    PrepareSocketMap(rping, ip);
    RpingInput input;
    input.dip = ip;
    PingTargetInfo targetInter;
    (void)memset_s(&targetInter, sizeof(targetInter), 0, sizeof(targetInter));
    PingInitInfo recvInfo;
    bool skipImportBind = false;
    // pre-populate rdmaInfoMaps_
    PingQpInfo qp{};
    rping.rdmaInfoMaps_.Emplace(std::pair<std::string, PingQpInfo>(std::string(ip.GetReadableIP()), qp));

    HcclResult ret
        = rping.RcAttrRecvAndFill(&input, 1000, HCCN_RPING_ADDR_TYPE_IP, &targetInter, &recvInfo, &skipImportBind);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(skipImportBind);
}

// Branch4: payloadLenMap_ already exists -> skipImportBind=true (rdmaInfo newly inserted)
TEST_F(RcFlow_UT, ut_RcAttrRecvAndFill_SkipOnPayloadLenHit)
{
    PingMesh rping;
    HcclIpAddress ip = MakeInputIp();
    PrepareSocketMap(rping, ip);
    RpingInput input;
    input.dip = ip;
    input.len = 100;
    PingTargetInfo targetInter;
    (void)memset_s(&targetInter, sizeof(targetInter), 0, sizeof(targetInter));
    PingInitInfo recvInfo;
    bool skipImportBind = false;
    // pre-populate payloadLenMap_
    rping.payloadLenMap_.Emplace(std::pair<std::string, u32>(std::string(ip.GetReadableIP()), 200));

    HcclResult ret
        = rping.RcAttrRecvAndFill(&input, 1000, HCCN_RPING_ADDR_TYPE_IP, &targetInter, &recvInfo, &skipImportBind);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(skipImportBind);
}

// Branch3: socName is null -> CHK_PTR_NULL early return, no map residue (collaborator pre-check)
TEST_F(RcFlow_UT, ut_RcAttrRecvAndFill_SocNameNullNoResidue)
{
    PingMesh rping;
    HcclIpAddress ip = MakeInputIp();
    PrepareSocketMap(rping, ip);
    RpingInput input;
    input.dip = ip;
    input.len = 100;
    PingTargetInfo targetInter;
    (void)memset_s(&targetInter, sizeof(targetInter), 0, sizeof(targetInter));
    PingInitInfo recvInfo;
    bool skipImportBind = false;

    MOCKER(aclrtGetSocName).stubs().with(mockcpp::any()).will(returnValue((const char*)nullptr));
    HcclResult ret
        = rping.RcAttrRecvAndFill(&input, 1000, HCCN_RPING_ADDR_TYPE_IP, &targetInter, &recvInfo, &skipImportBind);
    EXPECT_EQ(ret, HCCL_E_PTR);
    EXPECT_FALSE(skipImportBind);
    EXPECT_FALSE(rping.rdmaInfoMaps_.Find(std::string(ip.GetReadableIP())).second);
    EXPECT_FALSE(rping.payloadLenMap_.Find(std::string(ip.GetReadableIP())).second);
    GlobalMockObject::verify();
}

// HccnRcTargetAttrProcess: RecvAndFill fails -> pass through (non-short-circuit path)
TEST_F(RcFlow_UT, ut_HccnRcTargetAttrProcess_RecvFail)
{
    MOCKER_CPP(&HcclSocket::Connect).reset();
    MOCKER_CPP(&HcclSocket::Connect).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_TCP_CONNECT));
    PingMesh rping;
    RpingInput input; // dip not pre-set in socketMaps_ => RpingRecvTargetInfo link fails
    PingTargetInfo targetInter;
    (void)memset_s(&targetInter, sizeof(targetInter), 0, sizeof(targetInter));

    HcclResult ret = rping.HccnRcTargetAttrProcess(&input, 1000, HCCN_RPING_ADDR_TYPE_IP, &targetInter);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

// RcHandleLinkReq: peer illegal msgType (non-LINK_REQ) -> abortSever=true and return HCCL_E_PARA
TEST_F(RcFlow_UT, ut_RcHandleLinkReq_InvalidMsgType)
{
    g_nextMsgType = static_cast<u8>(RcMsgType::UNLINK_REQ); // not LINK_REQ
    PingMesh rping;
    HcclSocket socket(nullptr, 0);
    struct RcServerLinkCtx ctx;
    (void)memset_s(&ctx, sizeof(ctx), 0, sizeof(ctx));
    bool abortSever = false;

    HcclResult ret = rping.RcHandleLinkReq(socket, ctx, abortSever);
    EXPECT_EQ(ret, HCCL_E_PARA);
    EXPECT_TRUE(abortSever);
    g_nextMsgType = static_cast<u8>(RcMsgType::LINK_REQ);
}

// RcClientExchangeInfo: server reply jetty size < sizeof(UbJettyId) -> HCCL_E_PARA (anti-corruption)
TEST_F(RcFlow_UT, ut_RcClientExchangeInfo_InvalidSize)
{
    g_recvJettySize = g_ubJettyIdSize - 1; // one byte short of UbJettyId, trigger invalid-size
    PingMesh rping;
    std::shared_ptr<HcclSocket> socket = std::make_shared<HcclSocket>(nullptr, 0);
    HcclIpAddress ipAddr(0x7F000001);
    PingInitInfo recvInfo;
    (void)memset_s(&recvInfo, sizeof(recvInfo), 0, sizeof(recvInfo));

    HcclResult ret = rping.RcClientExchangeInfo(socket, ipAddr, recvInfo);
    EXPECT_EQ(ret, HCCL_E_PARA);
    g_recvJettySize = 24;
}

// HandleRcLinkRequest: RC server 3-step link orchestration all ok -> HCCL_SUCCESS, abortSever unchanged
TEST_F(RcFlow_UT, ut_HandleRcLinkRequest_Success)
{
    // Orchestration: RcHandleLinkReq reads LINK_REQ, RcHandleUnlink reads UNLINK_REQ -> feed header via sequence
    g_msgTypeSeq[0] = static_cast<u8>(RcMsgType::LINK_REQ);
    g_msgTypeSeq[1] = static_cast<u8>(RcMsgType::UNLINK_REQ);
    g_msgTypeSeqLen = 2;
    g_msgTypeSeqIdx = 0;
    PingMesh rping;
    HcclSocket socket(nullptr, 0);
    bool abortSever = false;

    HcclResult ret = rping.HandleRcLinkRequest(socket, 1, abortSever);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_FALSE(abortSever);
}

// HandleRcLinkRequest: first step RcHandleLinkReq fails -> short-circuit return, abortSever set
TEST_F(RcFlow_UT, ut_HandleRcLinkRequest_LinkReqFail)
{
    g_nextMsgType = static_cast<u8>(RcMsgType::UNLINK_REQ); // illegal LINK_REQ
    PingMesh rping;
    HcclSocket socket(nullptr, 0);
    bool abortSever = false;

    HcclResult ret = rping.HandleRcLinkRequest(socket, 1, abortSever);
    EXPECT_EQ(ret, HCCL_E_PARA);
    EXPECT_TRUE(abortSever);
    g_nextMsgType = static_cast<u8>(RcMsgType::LINK_REQ);
}

// HandleRcLinkRequest: second step RcHandleGetLocalTp fails -> short-circuit error (recoverable, no abortSever)
TEST_F(RcFlow_UT, ut_HandleRcLinkRequest_GetLocalTpFail)
{
    g_targetAddRet = HCCL_E_INTERNAL;
    PingMesh rping;
    HcclSocket socket(nullptr, 0);
    bool abortSever = false;

    HcclResult ret = rping.HandleRcLinkRequest(socket, 1, abortSever);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
    EXPECT_FALSE(abortSever);
    g_nextMsgType = static_cast<u8>(RcMsgType::LINK_REQ);
}

// RpingRecvTargetInfo: RC mode isRcMode=true goes to RcClientExchangeInfo (collaborator new branch)
// Precondition: socketMaps_ has no such IP -> do link (mock Init/Connect/GetStatus OK) -> RC branch -> exchange ok
TEST_F(RcFlow_UT, ut_RpingRecvTargetInfo_RcMode)
{
    PingMesh rping;
    rping.taMode_ = HCCN_RPING_UB_RC;
    rping.remoteEid_ = HcclIpAddress(0x7F000001);
    HcclIpAddress ipAddr(0x7F000001);
    PingInitInfo recvInfo;
    (void)memset_s(&recvInfo, sizeof(recvInfo), 0, sizeof(recvInfo));

    HcclResult ret = rping.RpingRecvTargetInfo(nullptr, 13886, ipAddr, recvInfo, 5000);
    // RC branch ok: RcClientExchangeInfo returns SUCCESS and remoteTpHandle backfilled
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(rping.remoteTpHandle_, 0x5566ULL);
}
