/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stddef.h>
#include <errno.h>
#include "ascend_hal.h"
#include "dl_hal_function.h"
#include "dl_urma_function.h"
#include "rs_socket.h"
#include "ut_dispatch.h"
#include "ra_rs_comm.h"
#include "rs.h"
#include "hccp_common.h"
#include "rs_inner.h"
#include "rs_ping_inner.h"
#include "hccp_ping.h"
#include "rs_epoll.h"
#include "rs_ping.h"
#include "rs_ping_urma.h"
#include "tc_ut_rs_ping_urma.h"

/* Compile-time ABI guard for cross-version RA/RS mixing (host/device pass-by-value).
 * These structs are shared with the peer firmware; sizes must stay unchanged. */
_Static_assert(sizeof(struct PingLocalCommInfo) == 52U, "PingLocalCommInfo ABI size changed");
_Static_assert(sizeof(struct PingInitAttr) == 360U, "PingInitAttr ABI size changed");
_Static_assert(sizeof(struct PingTargetInfo) == 1688U, "PingTargetInfo ABI size changed");

extern int RsGetPingCb(struct RaRsDevInfo* rdev, struct RsPingCtxCb** pingCb);
extern int RsPingUrmaPollRcq(struct RsPingCtxCb* pingCb, int* polledCnt, struct timeval* timestamp2);
extern void RsPongUrmaHandleSend(struct RsPingCtxCb* pingCb, int polledCnt, struct timeval* timestamp2);
extern void RsPongUrmaPollRcq(struct RsPingCtxCb* pingCb);
extern int RsPingUrmaPollScq(struct RsPingCtxCb* pingCb, struct RsPingTargetInfo* target);
extern struct RsPingPongOps* RsPingUrmaGetOps(void);
extern struct RsPingPongDfx* RsPingUrmaGetDfx(void);
extern int
RsPingCommonImportJetty(urma_context_t* urmaCtx, struct PingQpInfo* target, urma_target_jetty_t** importTjetty);
extern int
RsPingCommonInitJettyWithAttr(struct RsPingCtxCb* pingCb, union PingQpAttr* attr, struct RsPingLocalJettyCb* jettyCb);
extern int RsPingCommonInitJfce(struct rs_cb* rscb, struct RsPingCtxCb* pingCb, struct RsPingLocalJettyCb* jettyCb);
extern int RsPingCommonInitSendJfcWithAttr(
    struct rs_cb* rscb, struct RsPingCtxCb* pingCb, union PingQpAttr* attr, struct RsPingLocalJettyCb* jettyCb);
extern int
RsPingCommonInitRecvJfcWithAttr(struct RsPingCtxCb* pingCb, union PingQpAttr* attr, struct RsPingLocalJettyCb* jettyCb);
extern int RsPongJettyPostSend(struct RsPingCtxCb* pingCb, urma_cr_t* cr, struct timeval* timestamp2);
extern int RsPingCommonJfrPostRecv(struct RsPingLocalJettyCb* jettyCb);
extern int RsPongJettyResolveResponsePacket(struct RsPingCtxCb* pingCb, uint32_t sgeIdx, struct timeval* timestamp4);
extern int
RsPingUrmaFindTargetNode(struct RsPingCtxCb* pingCb, struct PingQpInfo* target, struct RsPingTargetInfo** node);
extern int RsPingCommonPollSendJfc(struct RsPingLocalJettyCb* jettyCb);
extern int
RsPongJettyFindAllocTargetNode(struct RsPingCtxCb* pingCb, struct PingQpInfo* target, struct RsPongTargetInfo** node);
extern int
RsPongJettyFindTargetNode(struct RsPingCtxCb* pingCb, struct PingQpInfo* target, struct RsPongTargetInfo** node);
extern int RsGetJettyInfo(struct PingQpInfo* qpInfo, urma_jetty_id_t* jettyId, urma_eid_t* eid);
extern int RsPingUrmaPostSend(struct RsPingCtxCb* pingCb, struct RsPingTargetInfo* target);
extern void RsPingUrmaResetRecvBuffer(struct RsPingCtxCb* pingCb);
extern int RsPingCommonInitSegCb(struct rs_cb* rscb, struct RsPingCtxCb* pingCb, struct RsPingSegCb* segCb);
extern int
RsPingUrmaGetLocalTpInfo(struct RsPingCtxCb* pingCb, struct PingTargetInfo* target, struct PingTargetInfo* result);
extern int RsPingCommonImportJettyEx(
    struct RsPingCtxCb* pingCb, struct PingTargetInfo* target, urma_target_jetty_t** importTjetty);
extern int
RsPingUrmaAllocTargetNode(struct RsPingCtxCb* pingCb, struct PingTargetInfo* target, struct RsPingTargetInfo** node);
extern void
RsPingCommonDeinitLocalJetty(struct rs_cb* rscb, struct RsPingCtxCb* pingCb, struct RsPingLocalJettyCb* jettyCb);
extern void RsPingCommonDeinitLocalJettyBuffer(struct RsPingCtxCb* pingCb);
extern int RsPingPongInitLocalUbResources(
    struct rs_cb* rscb, struct PingInitAttr* attr, struct PingInitInfo* info, struct RsPingCtxCb* pingCb);
extern int
RsPingCommonInitJettyWithAttr(struct RsPingCtxCb* pingCb, union PingQpAttr* attr, struct RsPingLocalJettyCb* jettyCb);
extern void RsPingUrmaFreeTargetNode(struct RsPingCtxCb* pingCb, struct RsPingTargetInfo* targetInfo);
extern int RsPingUrmaPingCbInit(
    unsigned int phyId, struct PingInitAttr* attr, struct PingInitInfo* info, unsigned int* devIndex,
    struct RsPingCtxCb* pingCb);
extern int RsUrmaQueryDevice(urma_device_t* dev, urma_device_attr_t* devAttr);
extern int RsPingCommonInitLocalJetty(
    struct rs_cb* rscb, struct RsPingCtxCb* pingCb, union PingQpAttr* attr, struct RsPingLocalJettyCb* jettyCb);
extern int RsPingPongInitLocalJettyBuffer(
    struct rs_cb* rscb, struct PingInitAttr* attr, struct PingInitInfo* info, struct RsPingCtxCb* pingCb);
extern int RsPingCommonInitJettyPostRecvAll(struct RsPingLocalJettyCb* jettyCb);

urma_jfc_t gTmpJfc;
static struct rs_cb gTmpRsCb;
static struct RsPingCtxCb gTmpPingCb;
static struct RsPingTargetInfo gTmpTarget;
static struct RsPingTargetInfo gTmpTarget1;

#define TEST_SGE_LIST_LEN 1024

int RsGetRsCbUrmaStub(unsigned int phyId, struct rs_cb** rsCb)
{
    *rsCb = &gTmpRsCb;
    (*rsCb)->pingCb.pingPongOps = RsPingUrmaGetOps();
    (*rsCb)->pingCb.pingPongDfx = RsPingUrmaGetDfx();
    return 0;
}

int RsGetPingCbUrmaStub(struct RaRsDevInfo* rdev, struct RsPingCtxCb** pingCb)
{
    *pingCb = &gTmpPingCb;
    (*pingCb)->pingPongOps = RsPingUrmaGetOps();
    (*pingCb)->pingPongDfx = RsPingUrmaGetDfx();
    return 0;
}

int RsGetPingCbUrmaStub1(struct RaRsDevInfo* rdev, struct RsPingCtxCb** pingCb)
{
    *pingCb = &gTmpRsCb.pingCb;
    (*pingCb)->pingPongOps = RsPingUrmaGetOps();
    (*pingCb)->pingPongDfx = RsPingUrmaGetDfx();
    return 0;
}

int RsUrmaPollJfcStubPing0(urma_jfc_t* jfc, int crCnt, urma_cr_t* cr)
{
    cr->status = URMA_CR_LOC_LEN_ERR;
    return 1;
}

int RsUrmaPollJfcStubPing1(urma_jfc_t* jfc, int crCnt, urma_cr_t* cr)
{
    cr->status = URMA_CR_SUCCESS;
    return 1;
}

int RsUrmaWaitJfcStub(urma_jfce_t* jfce, uint32_t jfcCnt, int timeOut, urma_jfc_t* jfc[])
{
    *jfc = &gTmpJfc;
    return 1;
}

int RsPingUrmaFindTargetNodeStub(struct RsPingCtxCb* pingCb, struct PingQpInfo* target, struct RsPingTargetInfo** node)
{
    *node = &gTmpTarget;

    return -ENODEV;
}

int RsPingUrmaFindTargetNodeStub1(struct RsPingCtxCb* pingCb, struct PingQpInfo* target, struct RsPingTargetInfo** node)
{
    *node = &gTmpTarget1;

    return 0;
}

int RsPongJettyFindAllocTargetNodeStub(
    struct RsPingCtxCb* pingCb, struct PingQpInfo* target, struct RsPongTargetInfo** node)
{
    static struct RsPongTargetInfo tmpNode = {0};

    *node = &tmpNode;

    return 0;
}

int RsPongJettyFindTargetNodeStub(struct RsPingCtxCb* pingCb, struct PingQpInfo* target, struct RsPongTargetInfo** node)
{
    *node = NULL;
    return -19;
}

int RsPongJettyFindTargetNodeStub2(
    struct RsPingCtxCb* pingCb, struct PingQpInfo* target, struct RsPongTargetInfo** node)
{
    *node = calloc(1, sizeof(struct RsPongTargetInfo));
    RS_INIT_LIST_HEAD(&(*node)->list);
    return 0;
}

int RsUrmaGetTpListStub(urma_context_t* ctx, urma_get_tp_cfg_t* cfg, uint32_t* tpCnt, urma_tp_info_t* tpList)
{
    tpList->tp_handle = 0x1234;
    *tpCnt = 1;
    return 0;
}

int RsUrmaGetTpListFailStub(urma_context_t* ctx, urma_get_tp_cfg_t* cfg, uint32_t* tpCnt, urma_tp_info_t* tpList)
{
    return -1;
}

int RsUrmaGetTpListEmptyStub(urma_context_t* ctx, urma_get_tp_cfg_t* cfg, uint32_t* tpCnt, urma_tp_info_t* tpList)
{
    *tpCnt = 0;
    return 0;
}

urma_target_jetty_t* RsUrmaImportJettyExStub(
    urma_context_t* ctx, urma_rjetty_t* rjetty, urma_token_t* tokenValue, urma_import_jetty_ex_cfg_t* cfg)
{
    static urma_target_jetty_t tjetty;
    return &tjetty;
}

urma_target_jetty_t* RsUrmaImportJettyExFailStub(
    urma_context_t* ctx, urma_rjetty_t* rjetty, urma_token_t* tokenValue, urma_import_jetty_ex_cfg_t* cfg)
{
    return NULL;
}

int RsUrmaQueryDeviceStub(urma_device_t* dev, urma_device_attr_t* devAttr)
{
    devAttr->dev_cap.priority_info[0].tp_type.bs.utp = 1;
    return 0;
}

urma_jfr_t* RsUrmaCreateJfrStub(urma_context_t* ctx, urma_jfr_cfg_t* cfg)
{
    static urma_jfr_t jfr;
    return &jfr;
}

urma_jetty_t* RsUrmaCreateJettyStub(urma_context_t* ctx, urma_jetty_cfg_t* cfg)
{
    static urma_jetty_t jetty;
    return &jetty;
}

void TcRsPingInitDeinitUrma()
{
    struct ibv_comp_channel clientChannel = {0};
    struct ibv_comp_channel serverChannel = {0};
    struct PingTargetInfo target = {0};
    struct PingInitAttr attr = {0};
    struct PingInitInfo info = {0};
    struct RaRsDevInfo rdev = {0};
    struct PingTargetInfo rxData = {0};
    unsigned int rdevIndex = 0;
    int ret = 0;

    mocker_invoke(RsGetRsCb, RsGetRsCbUrmaStub, 20);
    mocker_invoke(RsUrmaQueryDevice, RsUrmaQueryDeviceStub, 20);
    mocker(rsGetLocalDevIDByHostDevID, 20, 0);
    mocker(RsSetupSharemem, 20, 0);
    mocker(RsEpollCtl, 20, 0);
    mocker(DlHalBuffAllocAlignEx, 20, 0);
    attr.protocol = PROTOCOL_UDMA;
    ret = RsPingInit(&attr, &info, &rdevIndex);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_INT_EQ(info.result.headerSize, RS_PING_PAYLOAD_HEADER_RESV_CUSTOM);
    mocker_clean();

    gTmpRsCb.pingCb.initCnt = 1;
    RS_INIT_LIST_HEAD(&gTmpRsCb.pingCb.pingList);
    RS_INIT_LIST_HEAD(&gTmpRsCb.pingCb.pongList);
    target.payload.size = 1;
    mocker_invoke(RsGetPingCb, RsGetPingCbUrmaStub1, 10);
    mocker_invoke(RsGetRsCb, RsGetRsCbUrmaStub, 20);
    ret = RsPingTargetAdd(&rdev, &target, &rxData);
    EXPECT_INT_EQ(ret, 0);
    mocker(RsEpollCtl, 20, 0);
    ret = RsPingDeinit(&rdev);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRsPingTargetAddDelUrma()
{
    struct PingTargetInfo target = {0};
    struct PingTargetInfo rxData = {0};
    struct RaRsDevInfo rdev = {0};
    unsigned int num = 1;
    int ret = 0;

    gTmpPingCb.udevCb.transMode = URMA_TM_UM;
    gTmpPingCb.taskStatus = RS_PING_TASK_RESET;
    RS_INIT_LIST_HEAD(&gTmpPingCb.pingList);
    mocker_invoke(RsGetPingCb, RsGetPingCbUrmaStub, 2);
    ret = RsPingTargetAdd(&rdev, &target, &rxData);
    EXPECT_INT_EQ(ret, 0);
    ret = RsPingTargetDel(&rdev, (struct PingTargetCommInfo*)&target, &num);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    target.payload.size = 1;
    gTmpPingCb.taskStatus = RS_PING_TASK_RESET;
    RS_INIT_LIST_HEAD(&gTmpPingCb.pingList);
    mocker_invoke(RsGetPingCb, RsGetPingCbUrmaStub, 1);
    mocker(RsPingCommonImportJetty, 1, -1);
    ret = RsPingTargetAdd(&rdev, &target, &rxData);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    gTmpPingCb.taskStatus = RS_PING_TASK_RESET;
    RS_INIT_LIST_HEAD(&gTmpPingCb.pingList);
    mocker_invoke(RsGetPingCb, RsGetPingCbUrmaStub, 1);
    mocker(calloc, 10, 0);
    ret = RsPingTargetAdd(&rdev, &target, &rxData);
    EXPECT_INT_EQ(ret, -ENOMEM);
    mocker_clean();

    gTmpPingCb.taskStatus = RS_PING_TASK_RESET;
    mocker_invoke(RsGetPingCb, RsGetPingCbUrmaStub, 1);
    mocker(RsPingUrmaFindTargetNode, 1, -1);
    ret = RsPingTargetDel(&rdev, (struct PingTargetCommInfo*)&target, &num);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
}

void TcRsPingUrmaPostSend()
{
    struct RsPingTargetInfo target = {0};
    struct RsPingCtxCb pingCb = {0};
    urma_jetty_t serverJetty = {0};
    urma_jetty_t clientJetty = {0};
    void* addr = malloc(256);
    urma_sge_t sge = {0};
    int ret = 0;

    target.payloadBuffer = malloc(1);
    target.payloadSize = 1;
    sge.addr = (uintptr_t)addr;
    sge.len = 256;
    pingCb.pingJetty.sendSegCb.sgeNum = 1;
    pingCb.pingJetty.sendSegCb.sgeList = &sge;
    pingCb.pongJetty.jetty = &serverJetty;
    pingCb.pingJetty.jetty = &clientJetty;
    ret = RsPingUrmaPostSend(&pingCb, &target);
    EXPECT_INT_EQ(ret, 0);

    mocker(RsUrmaPostJettySendWr, 1, -1);
    ret = RsPingUrmaPostSend(&pingCb, &target);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    free(addr);
    addr = NULL;
    free(target.payloadBuffer);
    target.payloadBuffer = NULL;
}

void TcRsPingClientPollCqUrma()
{
    struct RsPingCtxCb pingCb = {0};
    struct timeval timestamp2 = {0};
    int polledCnt = 0;
    int ret = 0;

    mocker_clean();
    ret = RsPingUrmaPollRcq(&pingCb, &polledCnt, &timestamp2);
    EXPECT_INT_EQ(ret, -EAGAIN);
    mocker(RsUrmaWaitJfc, 1, -1);
    ret = RsPingUrmaPollRcq(&pingCb, &polledCnt, &timestamp2);
    EXPECT_INT_EQ(ret, -EOPENSRC);
    mocker_clean();

    mocker(RsUrmaWaitJfc, 1, 1);
    mocker(RsUrmaAckJfc, 1, 0);
    ret = RsPingUrmaPollRcq(&pingCb, &polledCnt, &timestamp2);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    pingCb.pingJetty.recvJfc.jfc = &gTmpJfc;
    mocker_invoke(RsUrmaWaitJfc, RsUrmaWaitJfcStub, 1);
    mocker(RsUrmaPollJfc, 1, -1);
    ret = RsPingUrmaPollRcq(&pingCb, &polledCnt, &timestamp2);
    EXPECT_INT_EQ(ret, -EOPENSRC);
    mocker_clean();

    pingCb.pingJetty.recvJfc.maxRecvWcNum = 16;
    mocker_invoke(RsUrmaWaitJfc, RsUrmaWaitJfcStub, 1);
    mocker(RsUrmaPollJfc, 1, 1);
    mocker(RsPongJettyPostSend, 1, -1);
    ret = RsPingUrmaPollRcq(&pingCb, &polledCnt, &timestamp2);
    EXPECT_INT_EQ(ret, 0);
    RsPongUrmaHandleSend(&pingCb, polledCnt, &timestamp2);
    mocker_clean();

    mocker_invoke(RsUrmaWaitJfc, RsUrmaWaitJfcStub, 1);
    mocker(RsUrmaPollJfc, 1, 1);
    mocker(RsPongJettyPostSend, 1, 0);
    mocker(RsPingCommonJfrPostRecv, 1, -1);
    ret = RsPingUrmaPollRcq(&pingCb, &polledCnt, &timestamp2);
    EXPECT_INT_EQ(ret, 0);
    RsPongUrmaHandleSend(&pingCb, polledCnt, &timestamp2);
    mocker_clean();

    mocker_invoke(RsUrmaWaitJfc, RsUrmaWaitJfcStub, 1);
    mocker(RsUrmaPollJfc, 1, 1);
    mocker(RsPongJettyPostSend, 1, 0);
    mocker(RsPingCommonJfrPostRecv, 1, -1);
    mocker(RsUrmaRearmJfc, 1, -1);
    ret = RsPingUrmaPollRcq(&pingCb, &polledCnt, &timestamp2);
    EXPECT_INT_EQ(ret, 0);
    RsPongUrmaHandleSend(&pingCb, polledCnt, &timestamp2);
    mocker_clean();
}

void TcRsPingUrmaPollScq()
{
    struct RsPingTargetInfo target = {0};
    struct RsPingCtxCb pingCb = {0};
    int ret = 0;

    ret = RsPingUrmaPollScq(&pingCb, &target);
    EXPECT_INT_EQ(ret, -ENODATA);

    mocker_invoke(RsUrmaPollJfc, RsUrmaPollJfcStubPing0, 10);
    ret = RsPingUrmaPollScq(&pingCb, &target);
    EXPECT_INT_EQ(ret, -EOPENSRC);
    mocker_clean();

    mocker_invoke(RsUrmaPollJfc, RsUrmaPollJfcStubPing1, 10);
    ret = RsPingUrmaPollScq(&pingCb, &target);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRsPingServerPollCqUrma()
{
    struct RsPingCtxCb pingCb = {0};

    mocker(RsUrmaWaitJfc, 1, -1);
    RsPongUrmaPollRcq(&pingCb);
    mocker_clean();

    mocker(RsUrmaWaitJfc, 1, 0);
    RsPongUrmaPollRcq(&pingCb);
    mocker_clean();

    pingCb.pongJetty.recvJfc.jfc = &gTmpJfc;
    pingCb.pongJetty.recvJfc.maxRecvWcNum = 16;
    mocker_invoke(RsUrmaWaitJfc, RsUrmaWaitJfcStub, 1);
    mocker(RsUrmaPollJfc, 1, -1);
    RsPongUrmaPollRcq(&pingCb);
    mocker_clean();

    mocker_invoke(RsUrmaWaitJfc, RsUrmaWaitJfcStub, 1);
    mocker(RsUrmaPollJfc, 1, 1);
    mocker(RsPongJettyResolveResponsePacket, 1, -1);
    RsPongUrmaPollRcq(&pingCb);
    mocker_clean();

    mocker_invoke(RsUrmaWaitJfc, RsUrmaWaitJfcStub, 1);
    mocker(RsUrmaPollJfc, 1, 1);
    mocker(RsPongJettyResolveResponsePacket, 1, 0);
    mocker(RsPingCommonJfrPostRecv, 1, -1);
    RsPongUrmaPollRcq(&pingCb);
    mocker_clean();

    mocker_invoke(RsUrmaWaitJfc, RsUrmaWaitJfcStub, 1);
    mocker(RsUrmaPollJfc, 1, 1);
    mocker(RsPongJettyResolveResponsePacket, 1, 0);
    mocker(RsPingCommonJfrPostRecv, 1, 0);
    mocker(RsUrmaRearmJfc, 1, -1);
    RsPongUrmaPollRcq(&pingCb);
    mocker_clean();
}

void TcRsEpollEventPingHandleUrma()
{
    urma_jfce_t pingJfce = {0};
    urma_jfce_t pongJfce = {0};
    struct rs_cb rscb = {0};
    int ret = 0;

    pongJfce.fd = 1;
    rscb.pingCb.initCnt = 1;
    rscb.pingCb.pingJetty.jfce = &pingJfce;
    rscb.pingCb.pongJetty.jfce = &pongJfce;
    rscb.pingCb.threadStatus = RS_PING_THREAD_RUNNING;
    rscb.pingCb.pingPongOps = RsPingUrmaGetOps();
    rscb.pingCb.pingPongDfx = RsPingUrmaGetDfx();

    mocker(RsPingUrmaPollRcq, 10, 0);
    mocker(RsPongUrmaHandleSend, 10, 0);
    mocker(RsPongUrmaPollRcq, 10, 0);

    ret = RsEpollEventPingHandle(&rscb, 0);
    EXPECT_INT_EQ(ret, 0);
    ret = RsEpollEventPingHandle(&rscb, 1);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
    return;
}

void TcRsPingGetResultsUrma()
{
    struct PingTargetCommInfo target = {0};
    struct PingResultInfo result = {0};
    struct RaRsDevInfo rdev = {0};
    unsigned int num = 1;
    int ret = 0;

    gTmpPingCb.taskStatus = RS_PING_TASK_RESET;
    mocker_invoke(RsGetPingCb, RsGetPingCbUrmaStub, 1);
    mocker_invoke(RsPingUrmaFindTargetNode, RsPingUrmaFindTargetNodeStub, 1);
    ret = RsPingGetResults(&rdev, &target, &num, &result);
    EXPECT_INT_EQ(ret, -ENODEV);
    mocker_clean();

    num = 1;
    gTmpPingCb.taskStatus = RS_PING_TASK_RESET;
    mocker_invoke(RsGetPingCb, RsGetPingCbUrmaStub, 1);
    mocker_invoke(RsPingUrmaFindTargetNode, RsPingUrmaFindTargetNodeStub1, 1);
    ret = RsPingGetResults(&rdev, &target, &num, &result);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRsPingServerPostSendUrma()
{
    struct RsPingCtxCb pingCb = {0};
    struct timeval timestamp2 = {0};
    void* sendAddr = malloc(TEST_SGE_LIST_LEN);
    void* recvAddr = malloc(TEST_SGE_LIST_LEN);
    urma_cr_t cr = {0};
    int ret = 0;

    cr.user_ctx = 1;
    mocker(RsPingCommonPollSendJfc, 1, 0);
    ret = RsPongJettyPostSend(&pingCb, &cr, &timestamp2);
    EXPECT_INT_EQ(ret, -EIO);
    mocker_clean();

    cr.user_ctx = 0;
    cr.completion_len = 16;
    pingCb.pingJetty.recvSegCb.sgeList = calloc(1, sizeof(urma_sge_t));
    pingCb.pingJetty.recvSegCb.sgeNum = 1;
    pingCb.pongJetty.sendSegCb.sgeList = calloc(1, sizeof(urma_sge_t));
    pingCb.pongJetty.sendSegCb.sgeNum = 1;
    pingCb.pongJetty.sendSegCb.sgeList->addr = (uintptr_t)sendAddr;
    pingCb.pongJetty.sendSegCb.sgeList->len = TEST_SGE_LIST_LEN;
    pingCb.pingJetty.recvSegCb.sgeList->addr = (uintptr_t)recvAddr;
    pingCb.pingJetty.recvSegCb.sgeList->len = TEST_SGE_LIST_LEN;
    mocker(RsPingCommonPollSendJfc, 1, 0);
    mocker(RsPongJettyFindAllocTargetNode, 1, -1);
    ret = RsPongJettyPostSend(&pingCb, &cr, &timestamp2);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker(RsPingCommonPollSendJfc, 1, 0);
    mocker_invoke(RsPongJettyFindAllocTargetNode, RsPongJettyFindAllocTargetNodeStub, 1);
    mocker(gettimeofday, 20, 1);
    mocker(RsUrmaPostJettySendWr, 1, -1);
    ret = RsPongJettyPostSend(&pingCb, &cr, &timestamp2);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker(RsPingCommonPollSendJfc, 1, 0);
    mocker_invoke(RsPongJettyFindAllocTargetNode, RsPongJettyFindAllocTargetNodeStub, 1);
    ret = RsPongJettyPostSend(&pingCb, &cr, &timestamp2);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    free(pingCb.pongJetty.sendSegCb.sgeList);
    pingCb.pongJetty.sendSegCb.sgeList = NULL;
    free(pingCb.pingJetty.recvSegCb.sgeList);
    pingCb.pingJetty.recvSegCb.sgeList = NULL;
    free(recvAddr);
    recvAddr = NULL;
    free(sendAddr);
    sendAddr = NULL;
}

void TcRsPongJettyFindAllocTargetNode()
{
    struct RsPongTargetInfo* node = NULL;
    struct RsPingCtxCb pingCb = {0};
    struct PingQpInfo target = {0};
    int ret = 0;

    mocker_invoke(RsPongJettyFindTargetNode, RsPongJettyFindTargetNodeStub2, 1);
    mocker(RsPingCommonImportJetty, 1, -1);
    ret = RsPongJettyFindAllocTargetNode(&pingCb, &target, &node);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    pthread_mutex_init(&pingCb.pongMutex, NULL);
    RS_INIT_LIST_HEAD(&pingCb.pongList);
    mocker_invoke(RsPongJettyFindTargetNode, RsPongJettyFindTargetNodeStub, 1);
    mocker(RsPingCommonImportJetty, 1, 0);
    ret = RsPongJettyFindAllocTargetNode(&pingCb, &target, &node);
    free(node);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRsPingCommonPollSendJfc()
{
    struct RsPingLocalQpCb qpCb = {0};
    int ret = 0;

    mocker(RsUrmaPollJfc, 1, -1);
    ret = RsPingCommonPollSendJfc((struct RsPingLocalJettyCb*)&qpCb);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    mocker(RsUrmaPollJfc, 1, 1);
    ret = RsPingCommonPollSendJfc((struct RsPingLocalJettyCb*)&qpCb);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRsPongJettyFindTargetNode()
{
    struct RsPongTargetInfo stubNode = {0};
    struct RsPongTargetInfo* node = NULL;
    struct RsPingCtxCb pingCb = {0};
    struct PingQpInfo target = {0};
    int ret = 0;

    RS_INIT_LIST_HEAD(&pingCb.pongList);
    ret = RsPongJettyFindTargetNode(&pingCb, &target, &node);
    EXPECT_INT_EQ(ret, -ENODEV);

    RsListAddTail(&stubNode.list, &pingCb.pongList);
    ret = RsPongJettyFindTargetNode(&pingCb, &target, &node);
    EXPECT_INT_EQ(ret, 0);
}

void TcRsPongJettyResolveResponsePacket()
{
    struct RsPingCtxCb pingCb = {0};
    struct timeval timestamp4 = {0};
    void* recvAddr = calloc(1, TEST_SGE_LIST_LEN);
    uint32_t sgeIdx = 0;
    int ret = 0;

    pingCb.pongJetty.recvSegCb.sgeList = calloc(1, sizeof(urma_sge_t));
    pingCb.pongJetty.recvSegCb.sgeList->addr = (uintptr_t)recvAddr;
    pingCb.pongJetty.recvSegCb.sgeList->len = TEST_SGE_LIST_LEN;
    pingCb.taskId = 1;

    ret = RsPongJettyResolveResponsePacket(&pingCb, sgeIdx, &timestamp4);
    EXPECT_INT_EQ(ret, 0);

    pingCb.taskId = 0;
    mocker_invoke(RsPingUrmaFindTargetNode, RsPingUrmaFindTargetNodeStub, 1);
    ret = RsPongJettyResolveResponsePacket(&pingCb, sgeIdx, &timestamp4);
    EXPECT_INT_EQ(ret, -ENODEV);
    mocker_clean();

    gTmpTarget1.resultSummary.rttMax = 10;
    gTmpTarget1.resultSummary.rttMin = 4;
    pthread_mutex_init(&gTmpTarget1.tripMutex, NULL);
    mocker(RsPingGetTripTime, 1, 11);
    mocker_invoke(RsPingUrmaFindTargetNode, RsPingUrmaFindTargetNodeStub1, 1);
    ret = RsPongJettyResolveResponsePacket(&pingCb, sgeIdx, &timestamp4);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    gTmpTarget1.resultSummary.rttMax = 10;
    gTmpTarget1.resultSummary.rttMin = 4;
    gTmpTarget1.resultSummary.taskAttr.timeoutInterval = 12;
    pthread_mutex_init(&gTmpTarget1.tripMutex, NULL);
    mocker(RsPingGetTripTime, 1, 11);
    mocker_invoke(RsPingUrmaFindTargetNode, RsPingUrmaFindTargetNodeStub1, 1);
    ret = RsPongJettyResolveResponsePacket(&pingCb, sgeIdx, &timestamp4);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    gTmpTarget1.resultSummary.rttMin = 20;
    gTmpTarget1.resultSummary.rttMax = 4;
    gTmpTarget1.resultSummary.rttAvg = 0;
    gTmpTarget1.resultSummary.recvCnt = 0;
    gTmpTarget1.resultSummary.timeoutCnt = 0;
    gTmpTarget1.resultSummary.taskId = 0;
    gTmpTarget1.resultSummary.taskAttr.timeoutInterval = 12;
    mocker(RsPingGetTripTime, 1, 11);
    mocker_invoke(RsPingUrmaFindTargetNode, RsPingUrmaFindTargetNodeStub1, 1);
    ret = RsPongJettyResolveResponsePacket(&pingCb, sgeIdx, &timestamp4);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_INT_EQ(gTmpTarget1.resultSummary.rttMin, 11);
    EXPECT_INT_EQ(gTmpTarget1.resultSummary.rttMax, 11);
    EXPECT_INT_EQ(gTmpTarget1.resultSummary.rttAvg, 11);
    EXPECT_INT_EQ(gTmpTarget1.resultSummary.recvCnt, 1);
    EXPECT_INT_EQ(gTmpTarget1.resultSummary.timeoutCnt, 0);
    mocker_clean();

    gTmpTarget1.resultSummary.rttMin = 11;
    gTmpTarget1.resultSummary.rttMax = 11;
    gTmpTarget1.resultSummary.rttAvg = 11;
    gTmpTarget1.resultSummary.recvCnt = 1;
    gTmpTarget1.resultSummary.timeoutCnt = 0;
    gTmpTarget1.resultSummary.taskId = 0;
    gTmpTarget1.resultSummary.taskAttr.timeoutInterval = 12;
    mocker(RsPingGetTripTime, 1, 5);
    mocker_invoke(RsPingUrmaFindTargetNode, RsPingUrmaFindTargetNodeStub1, 1);
    ret = RsPongJettyResolveResponsePacket(&pingCb, sgeIdx, &timestamp4);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_INT_EQ(gTmpTarget1.resultSummary.rttMin, 5);
    EXPECT_INT_EQ(gTmpTarget1.resultSummary.rttMax, 11);
    EXPECT_INT_EQ(gTmpTarget1.resultSummary.rttAvg, 8);
    EXPECT_INT_EQ(gTmpTarget1.resultSummary.recvCnt, 2);
    EXPECT_INT_EQ(gTmpTarget1.resultSummary.timeoutCnt, 0);
    mocker_clean();

    gTmpTarget1.resultSummary.rttMin = 20;
    gTmpTarget1.resultSummary.rttMax = 4;
    gTmpTarget1.resultSummary.rttAvg = 0;
    gTmpTarget1.resultSummary.recvCnt = 0;
    gTmpTarget1.resultSummary.timeoutCnt = 0;
    gTmpTarget1.resultSummary.taskId = 0;
    gTmpTarget1.resultSummary.taskAttr.timeoutInterval = 1;
    mocker(RsPingGetTripTime, 1, 2000);
    mocker_invoke(RsPingUrmaFindTargetNode, RsPingUrmaFindTargetNodeStub1, 1);
    ret = RsPongJettyResolveResponsePacket(&pingCb, sgeIdx, &timestamp4);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_INT_EQ(gTmpTarget1.resultSummary.rttMin, 20);
    EXPECT_INT_EQ(gTmpTarget1.resultSummary.rttMax, 4);
    EXPECT_INT_EQ(gTmpTarget1.resultSummary.rttAvg, 0);
    EXPECT_INT_EQ(gTmpTarget1.resultSummary.recvCnt, 1);
    EXPECT_INT_EQ(gTmpTarget1.resultSummary.timeoutCnt, 1);
    mocker_clean();

    pthread_mutex_destroy(&gTmpTarget1.tripMutex);
    free(pingCb.pongJetty.recvSegCb.sgeList);
    pingCb.pongJetty.recvSegCb.sgeList = NULL;
    free(recvAddr);
    recvAddr = NULL;
}

void TcRsPingCommonImportJetty()
{
    urma_target_jetty_t* importTjetty = NULL;
    struct PingQpInfo target = {0};
    urma_context_t urmaCtx = {0};
    int ret = 0;

    mocker(RsGetJettyInfo, 1, 0);
    ret = RsPingCommonImportJetty(&urmaCtx, &target, &importTjetty);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
    free(importTjetty);
    importTjetty = NULL;

    mocker(RsGetJettyInfo, 1, 0);
    mocker(RsUrmaImportJetty, 1, NULL);
    errno = 0;
    ret = RsPingCommonImportJetty(&urmaCtx, &target, &importTjetty);
    EXPECT_INT_EQ(-EOPENSRC, ret);
    mocker_clean();
}

void TcRsPingCommonInitJettyCreateJfrFail()
{
    struct RsPingCtxCb pingCb = {0};
    union PingQpAttr attr = {0};
    struct RsPingLocalJettyCb jettyCb = {0};
    urma_jfc_t jfc = {0};
    urma_context_t urmaCtx = {0};
    int ret = 0;

    pingCb.udevCb.urmaCtx = &urmaCtx;
    jettyCb.sendJfc.jfc = &jfc;
    jettyCb.recvJfc.jfc = &jfc;

    mocker(RsUrmaCreateJfr, 1, NULL);
    errno = 0;
    ret = RsPingCommonInitJettyWithAttr(&pingCb, &attr, &jettyCb);
    EXPECT_INT_EQ(-EOPENSRC, ret);
    mocker_clean();
}

void TcRsPingCommonInitJettyCreateJettyFail()
{
    struct RsPingCtxCb pingCb = {0};
    union PingQpAttr attr = {0};
    struct RsPingLocalJettyCb jettyCb = {0};
    urma_jfc_t jfc = {0};
    urma_context_t urmaCtx = {0};
    int ret = 0;

    pingCb.udevCb.urmaCtx = &urmaCtx;
    jettyCb.sendJfc.jfc = &jfc;
    jettyCb.recvJfc.jfc = &jfc;

    mocker(RsUrmaCreateJetty, 1, NULL);
    errno = 0;
    ret = RsPingCommonInitJettyWithAttr(&pingCb, &attr, &jettyCb);
    EXPECT_INT_EQ(-EOPENSRC, ret);
    mocker_clean();
}

void TcRsPingUrmaResetRecvBuffer()
{
    struct RsPingCtxCb pingCb = {0};

    RsPingUrmaResetRecvBuffer(&pingCb);
}

void TcRsPingCommonJfrPostRecv()
{
    struct RsPingLocalJettyCb jettyCb = {0};
    urma_sge_t sge = {0};
    int ret = 0;

    jettyCb.recvSegCb.sgeNum = 1;
    jettyCb.recvSegCb.sgeList = &sge;
    ret = RsPingCommonJfrPostRecv(&jettyCb);
    EXPECT_INT_EQ(ret, 0);

    mocker(RsUrmaPostJettyRecvWr, 1, -1);
    ret = RsPingCommonJfrPostRecv(&jettyCb);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
}

void TcRsPingCommonInitSegCbFail()
{
    struct rs_cb rscb = {0};
    struct RsPingCtxCb pingCb = {0};
    struct RsPingSegCb segCb = {0};
    int ret = 0;

    segCb.len = 1;
    rscb.grpId = 0;
    pingCb.logicDevid = 0;

    mocker(pthread_mutex_init, 1, 0);
    mocker(DlHalBuffAllocAlignEx, 1, 0);
    mocker(RsUrmaRegisterSeg, 1, NULL);
    mocker(DlHalBuffFree, 1, 0);
    mocker(pthread_mutex_destroy, 1, 0);
    ret = RsPingCommonInitSegCb(&rscb, &pingCb, &segCb);
    EXPECT_INT_NE(0, ret);
    mocker_clean();
}

void TcRsPingCommonInitLocalJettyFail()
{
    struct rs_cb rscb = {0};
    struct RsPingCtxCb pingCb = {0};
    union PingQpAttr attr = {0};
    struct RsPingLocalJettyCb jettyCb = {0};
    urma_context_t urmaCtx = {0};
    urma_jfce_t jfce = {0};
    int ret = 0;

    pingCb.udevCb.urmaCtx = &urmaCtx;

    mocker(RsUrmaCreateJfce, 1, NULL);
    ret = RsPingCommonInitJfce(&rscb, &pingCb, &jettyCb);
    EXPECT_INT_NE(0, ret);
    mocker_clean();

    // jfce created ok but RsEpollCtl ADD fails: should clean up jfce and return failure
    // Do not mock RsUrmaCreateJfce (go through real calloc full address, avoid 64-bit pointer truncation),
    // only mock RsEpollCtl failure to verify the failure cleanup path
    pingCb.udevCb.urmaCtx = &urmaCtx;
    mocker(RsEpollCtl, 1, -1);
    ret = RsPingCommonInitJfce(&rscb, &pingCb, &jettyCb);
    EXPECT_INT_NE(0, ret);
    EXPECT_ADDR_EQ(jettyCb.jfce, NULL);
    mocker_clean();
    jettyCb.jfce = NULL;

    jettyCb.jfce = &jfce;
    mocker(RsUrmaCreateJfc, 1, NULL);
    ret = RsPingCommonInitSendJfcWithAttr(NULL, &pingCb, &attr, &jettyCb);
    EXPECT_INT_NE(0, ret);
    mocker_clean();

    mocker(RsUrmaCreateJfc, 1, NULL);
    ret = RsPingCommonInitRecvJfcWithAttr(&pingCb, &attr, &jettyCb);
    EXPECT_INT_NE(0, ret);
    mocker_clean();
}

void TcRsPingUrmaGetLocalTpInfo()
{
    struct RsPingCtxCb pingCb = {0};
    struct PingTargetInfo target = {0};
    struct PingTargetInfo rxData = {0};
    urma_context_t urmaCtx = {0};
    urma_jetty_t jetty = {0};
    int ret = 0;

    pingCb.udevCb.urmaCtx = &urmaCtx;
    pingCb.pingJetty.jetty = &jetty;
    pingCb.udevCb.transMode = URMA_TM_RC;
    pingCb.udevCb.tpType = URMA_RTP;
    mocker_invoke(RsUrmaGetTpList, RsUrmaGetTpListStub, 1);
    ret = RsPingUrmaGetLocalTpInfo(&pingCb, &target, &rxData);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_INT_EQ((int)rxData.localInfo.ub.tpHandle, 0x1234);
    // tpPsn is randomly generated by RsDrvGetRandomNum, not checking the concrete value
    mocker_clean();

    mocker_invoke(RsUrmaGetTpList, RsUrmaGetTpListFailStub, 1);
    ret = RsPingUrmaGetLocalTpInfo(&pingCb, &target, &rxData);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker_invoke(RsUrmaGetTpList, RsUrmaGetTpListEmptyStub, 1);
    ret = RsPingUrmaGetLocalTpInfo(&pingCb, &target, &rxData);
    // Collaborator version: tpCnt==0 treated as tp not found, returns -ENODATA
    EXPECT_INT_EQ(ret, -ENODATA);
    mocker_clean();

    // UM mode entering GET_LOCAL_TP_INFO should be intercepted (-EINVAL, RC-only)
    pingCb.udevCb.transMode = URMA_TM_UM;
    pingCb.udevCb.tpType = URMA_UTP;
    mocker_invoke(RsUrmaGetTpList, RsUrmaGetTpListStub, 1);
    ret = RsPingUrmaGetLocalTpInfo(&pingCb, &target, &rxData);
    EXPECT_INT_EQ(ret, -EINVAL);
    mocker_clean();

    // RC+CTP: tpType=CTP goes to tpCfg.flag.bs.ctp=1 branch (equivalence class supplement, default only tested RTP)
    struct RsPingCtxCb pingCbCtp = {0};
    struct PingTargetInfo rxDataCtp = {0};
    pingCbCtp.udevCb.urmaCtx = &urmaCtx;
    pingCbCtp.pingJetty.jetty = &jetty;
    pingCbCtp.udevCb.transMode = URMA_TM_RC;
    pingCbCtp.udevCb.tpType = URMA_CTP;
    mocker_invoke(RsUrmaGetTpList, RsUrmaGetTpListStub, 1);
    ret = RsPingUrmaGetLocalTpInfo(&pingCbCtp, &target, &rxDataCtp);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_INT_EQ((int)rxDataCtp.localInfo.ub.tpHandle, 0x1234);
    mocker_clean();

    // RsDrvGetRandomNum fails: psn random fetch fails -> -EIO (collaborator new branch)
    struct RsPingCtxCb pingCbPsn = {0};
    struct PingTargetInfo rxDataPsn = {0};
    pingCbPsn.udevCb.urmaCtx = &urmaCtx;
    pingCbPsn.pingJetty.jetty = &jetty;
    pingCbPsn.udevCb.transMode = URMA_TM_RC;
    pingCbPsn.udevCb.tpType = URMA_RTP;
    mocker_invoke(RsUrmaGetTpList, RsUrmaGetTpListStub, 1);
    mocker(RsDrvGetRandomNum, 1, -1);
    ret = RsPingUrmaGetLocalTpInfo(&pingCbPsn, &target, &rxDataPsn);
    EXPECT_INT_EQ(ret, -EIO);
    mocker_clean();
}

void TcRsPingCommonImportJettyEx()
{
    struct RsPingCtxCb pingCb = {0};
    struct PingTargetInfo target = {0};
    urma_target_jetty_t* importTjetty = NULL;
    urma_context_t urmaCtx = {0};
    urma_jetty_t jetty = {0};
    int ret = 0;

    pingCb.udevCb.urmaCtx = &urmaCtx;
    pingCb.pingJetty.jetty = &jetty;
    pingCb.udevCb.transMode = URMA_TM_RC;
    pingCb.udevCb.tpType = URMA_RTP;
    target.remoteInfo.qpInfo.ub.tokenValue = 0x5678;
    target.localInfo.ub.tpHandle = 0x100;
    target.remoteExInfo.ub.tpHandle = 0x200;
    mocker(RsGetJettyInfo, 1, 0);
    mocker_invoke(RsUrmaImportJettyEx, RsUrmaImportJettyExStub, 1);
    mocker(RsUrmaBindJettyEx, 1, 0);
    ret = RsPingCommonImportJettyEx(&pingCb, &target, &importTjetty);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_ADDR_NE(importTjetty, NULL);
    mocker_clean();

    importTjetty = NULL;
    pingCb.udevCb.transMode = URMA_TM_UM;
    pingCb.udevCb.tpType = URMA_UTP;
    mocker(RsGetJettyInfo, 1, 0);
    mocker_invoke(RsUrmaImportJettyEx, RsUrmaImportJettyExStub, 1);
    ret = RsPingCommonImportJettyEx(&pingCb, &target, &importTjetty);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_ADDR_NE(importTjetty, NULL);
    mocker_clean();

    importTjetty = NULL;
    pingCb.udevCb.transMode = URMA_TM_RC;
    pingCb.udevCb.tpType = URMA_RTP;
    mocker(RsGetJettyInfo, 1, 0);
    mocker_invoke(RsUrmaImportJettyEx, RsUrmaImportJettyExFailStub, 1);
    ret = RsPingCommonImportJettyEx(&pingCb, &target, &importTjetty);
    EXPECT_INT_EQ(ret, -EOPENSRC);
    EXPECT_ADDR_EQ(importTjetty, NULL);
    mocker_clean();

    importTjetty = NULL;
    mocker(RsGetJettyInfo, 1, 0);
    mocker_invoke(RsUrmaImportJettyEx, RsUrmaImportJettyExStub, 1);
    mocker(RsUrmaBindJettyEx, 1, -1);
    mocker(RsUrmaUnimportJetty, 1, 0); // bind failure triggers unimport; intercept to avoid free(static)
    ret = RsPingCommonImportJettyEx(&pingCb, &target, &importTjetty);
    EXPECT_INT_EQ(ret, -1);
    EXPECT_ADDR_EQ(importTjetty, NULL);
    mocker_clean();
}

void TcRsPingUrmaAllocTargetNodeRc()
{
    struct RsPingCtxCb pingCb = {0};
    struct PingTargetInfo target = {0};
    struct RsPingTargetInfo* node = NULL;
    urma_context_t urmaCtx = {0};
    urma_jetty_t jetty = {0};
    int ret = 0;

    pingCb.udevCb.urmaCtx = &urmaCtx;
    pingCb.pingJetty.jetty = &jetty;
    pingCb.udevCb.transMode = URMA_TM_RC;
    pingCb.udevCb.tpType = URMA_RTP;
    target.remoteInfo.qpInfo.ub.tokenValue = 0x5678;
    target.localInfo.ub.tpHandle = 0x100;
    target.remoteExInfo.ub.tpHandle = 0x200;
    mocker(RsGetJettyInfo, 1, 0);
    mocker_invoke(RsUrmaImportJettyEx, RsUrmaImportJettyExStub, 1);
    mocker(RsUrmaBindJettyEx, 1, 0);
    ret = RsPingUrmaAllocTargetNode(&pingCb, &target, &node);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_ADDR_NE(node, NULL);
    EXPECT_INT_EQ((int)node->resultSummary.rttMin, (int)~0);
    EXPECT_INT_EQ(node->state, RS_PING_PONG_TARGET_READY);
    free(node->payloadBuffer);
    pthread_mutex_destroy(&node->tripMutex);
    free(node);
    node = NULL;
    mocker_clean();

    node = NULL;
    mocker(calloc, 1, 0);
    ret = RsPingUrmaAllocTargetNode(&pingCb, &target, &node);
    EXPECT_INT_EQ(ret, -ENOMEM);
    EXPECT_ADDR_EQ(node, NULL);
    mocker_clean();

    node = NULL;
    mocker(RsGetJettyInfo, 1, 0);
    mocker_invoke(RsUrmaImportJettyEx, RsUrmaImportJettyExFailStub, 1);
    ret = RsPingUrmaAllocTargetNode(&pingCb, &target, &node);
    EXPECT_INT_NE(ret, 0);
    EXPECT_ADDR_EQ(node, NULL);
    mocker_clean();

    node = NULL;
    mocker(RsGetJettyInfo, 1, 0);
    mocker_invoke(RsUrmaImportJettyEx, RsUrmaImportJettyExStub, 1);
    mocker(RsUrmaBindJettyEx, 1, -1);
    mocker(RsUrmaUnimportJetty, 1, 0);
    ret = RsPingUrmaAllocTargetNode(&pingCb, &target, &node);
    EXPECT_INT_NE(ret, 0);
    EXPECT_ADDR_EQ(node, NULL);
    mocker_clean();
}

void TcRsPingUrmaPollScqRc()
{
    struct RsPingCtxCb pingCb = {0};
    struct RsPingTargetInfo target = {0};
    int ret = 0;

    // g260915: maxRetries = taskAttr.packetInterval * RS_PING_MSEC_TO_USEC(1000),
    // packetInterval=1 guarantees the loop can enter (sufficient retry window)
    pingCb.taskAttr.packetInterval = 1;
    pingCb.udevCb.transMode = URMA_TM_RC;
    mocker_invoke(RsUrmaPollJfc, RsUrmaPollJfcStubPing1, 1);
    ret = RsPingUrmaPollScq(&pingCb, &target);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    pingCb.taskAttr.packetInterval = 1;
    pingCb.udevCb.transMode = URMA_TM_RC;
    // packetInterval=1 => maxRetries=1000; mock up to 1000 poll/usleep calls for the timeout path
    mocker(RsUrmaPollJfc, 1000, 0);
    mocker(usleep, 1000, 0);
    ret = RsPingUrmaPollScq(&pingCb, &target);
    EXPECT_INT_EQ(ret, -ENODATA);
    mocker_clean();

    pingCb.taskAttr.packetInterval = 1;
    pingCb.udevCb.transMode = URMA_TM_RC;
    mocker_invoke(RsUrmaPollJfc, RsUrmaPollJfcStubPing0, 1);
    ret = RsPingUrmaPollScq(&pingCb, &target);
    EXPECT_INT_EQ(ret, -EOPENSRC);
    mocker_clean();
}

void TcRsPingCommonInitJettyWithAttr()
{
    struct RsPingCtxCb pingCb = {0};
    union PingQpAttr attr = {0};
    struct RsPingLocalJettyCb jettyCb = {0};
    urma_jfc_t jfc = {0};
    urma_jfr_t jfr = {0};
    urma_jetty_t jetty = {0};
    int ret = 0;

    pingCb.udevCb.transMode = URMA_TM_RC;
    jettyCb.sendJfc.jfc = &jfc;
    jettyCb.recvJfc.jfc = &jfc;
    attr.ub.qpAttr.cap.maxSendWr = 1;
    attr.ub.qpAttr.cap.maxRecvWr = 1;
    attr.ub.qpAttr.cap.maxSendSge = 1;
    attr.ub.qpAttr.cap.maxRecvSge = 1;
    attr.ub.qpAttr.cap.maxInlineData = 0;
    attr.ub.qpAttr.tokenValue = 0x100;
    // Do not mock pointer-returning CreateJfr/Jetty: ut_dispatch stub_fn_t returns int, truncating 64-bit addresses,
    // go through real wrappers -> underlying calloc stub, getting full heap address for success-log dereference
    ret = RsPingCommonInitJettyWithAttr(&pingCb, &attr, &jettyCb);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_ADDR_NE(jettyCb.jetty, NULL);
    EXPECT_ADDR_NE(jettyCb.jfr, NULL);
    // underlying stub allocates via calloc; free at case end to avoid leak
    free(jettyCb.jetty);
    free(jettyCb.jfr);
    jettyCb.jetty = NULL;
    jettyCb.jfr = NULL;
    mocker_clean();

    struct RsPingLocalJettyCb jettyCb2 = {0};
    jettyCb2.sendJfc.jfc = &jfc;
    jettyCb2.recvJfc.jfc = &jfc;
    pingCb.udevCb.transMode = URMA_TM_RC;
    mocker(RsUrmaCreateJfr, 1, 0);
    ret = RsPingCommonInitJettyWithAttr(&pingCb, &attr, &jettyCb2);
    // RS_URMA_PTR_TO_ERRNO returns -EOPENSRC for NULL (unless errno is non-zero)
    EXPECT_INT_EQ(ret, -EOPENSRC);
    mocker_clean();

    struct RsPingLocalJettyCb jettyCb3 = {0};
    jettyCb3.sendJfc.jfc = &jfc;
    jettyCb3.recvJfc.jfc = &jfc;
    pingCb.udevCb.transMode = URMA_TM_RC;
    // jfr goes real calloc (full address); jetty mocked NULL triggers create_jetty_fail cleanup
    mocker(RsUrmaCreateJetty, 1, 0);
    ret = RsPingCommonInitJettyWithAttr(&pingCb, &attr, &jettyCb3);
    EXPECT_INT_EQ(ret, -EOPENSRC);
    mocker_clean();
}

void TcRsPingPongInitLocalUbResourcesRc()
{
    struct rs_cb rscb = {0};
    struct RsPingCtxCb pingCb = {0};
    struct PingInitAttr attr = {0};
    struct PingInitInfo info = {0};
    urma_jetty_t jetty = {0};
    urma_jetty_t pongJetty = {0};
    int ret = 0;

    pingCb.udevCb.transMode = URMA_TM_RC;
    pingCb.pingJetty.jetty = &jetty;
    pingCb.pongJetty.jetty = &pongJetty;
    attr.client.ub.qpAttr.tokenValue = 0x100;
    attr.client.ub.qpAttr.cap.maxSendWr = 1;
    attr.client.ub.qpAttr.cap.maxRecvWr = 1;
    mocker(RsPingCommonInitLocalJetty, 2, -1);
    ret = RsPingPongInitLocalUbResources(&rscb, &attr, &info, &pingCb);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    pingCb.udevCb.transMode = URMA_TM_RC;
    pingCb.pingJetty.jetty = &jetty;
    pingCb.pongJetty.jetty = &pongJetty;
    attr.client.ub.qpAttr.tokenValue = 0x100;
    mocker(RsPingCommonInitLocalJetty, 2, 0);
    mocker(RsPingPongInitLocalJettyBuffer, 1, -1);
    mocker(RsPingCommonDeinitLocalJetty, 5, 0);
    ret = RsPingPongInitLocalUbResources(&rscb, &attr, &info, &pingCb);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    pingCb.udevCb.transMode = URMA_TM_RC;
    pingCb.pingJetty.jetty = &jetty;
    pingCb.pongJetty.jetty = &pongJetty;
    attr.client.ub.qpAttr.tokenValue = 0x100;
    mocker(RsPingCommonInitLocalJetty, 2, 0);
    mocker(RsPingPongInitLocalJettyBuffer, 1, 0);
    mocker(RsPingCommonInitJettyPostRecvAll, 2, -1);
    mocker(RsPingCommonDeinitLocalJettyBuffer, 1, 0);
    mocker(RsPingCommonDeinitLocalJetty, 5, 0);
    ret = RsPingPongInitLocalUbResources(&rscb, &attr, &info, &pingCb);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();

    pingCb.udevCb.transMode = URMA_TM_RC;
    pingCb.pingJetty.jetty = &jetty;
    pingCb.pongJetty.jetty = &pongJetty;
    attr.client.ub.qpAttr.tokenValue = 0x100;
    mocker(RsPingCommonInitLocalJetty, 2, 0);
    mocker(RsPingPongInitLocalJettyBuffer, 1, 0);
    mocker(RsPingCommonInitJettyPostRecvAll, 2, 0);
    ret = RsPingPongInitLocalUbResources(&rscb, &attr, &info, &pingCb);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRsPingCommonDeinitLocalJettyRc()
{
    struct rs_cb rscb = {0};
    struct RsPingCtxCb pingCb = {0};
    struct RsPingLocalJettyCb jettyCb = {0};
    urma_jetty_t jetty = {0};
    urma_jfce_t jfce = {0};

    pingCb.udevCb.transMode = URMA_TM_RC;
    jettyCb.jetty = &jetty;
    jettyCb.jfce = &jfce;
    mocker(RsUrmaFlushJetty, 1, 0);
    mocker(RsUrmaDeleteJetty, 5, 0);
    mocker(RsUrmaDeleteJfr, 5, 0);
    mocker(RsUrmaDeleteJfc, 10, 0);
    mocker(RsUrmaDeleteJfce, 5, 0);
    RsPingCommonDeinitLocalJetty(&rscb, &pingCb, &jettyCb);
    mocker_clean();

    struct RsPingLocalJettyCb jettyCb2 = {0};
    pingCb.udevCb.transMode = URMA_TM_RC;
    jettyCb2.jetty = NULL;
    jettyCb2.jfce = &jfce;
    mocker(RsUrmaDeleteJetty, 5, 0);
    mocker(RsUrmaDeleteJfr, 5, 0);
    mocker(RsUrmaDeleteJfc, 10, 0);
    mocker(RsUrmaDeleteJfce, 5, 0);
    RsPingCommonDeinitLocalJetty(&rscb, &pingCb, &jettyCb2);
    mocker_clean();
}

void TcRsPingCommonDeinitLocalJettyBufferRc()
{
    struct RsPingCtxCb pingCb = {0};

    pingCb.udevCb.transMode = URMA_TM_RC;
    mocker(RsUrmaUnregisterSeg, 5, 0);
    mocker(DlHalBuffFree, 5, 0);
    mocker(pthread_mutex_destroy, 5, 0);
    RsPingCommonDeinitLocalJettyBuffer(&pingCb);
    mocker_clean();

    pingCb.udevCb.transMode = URMA_TM_UM;
    mocker(RsUrmaUnregisterSeg, 5, 0);
    mocker(DlHalBuffFree, 5, 0);
    mocker(pthread_mutex_destroy, 5, 0);
    RsPingCommonDeinitLocalJettyBuffer(&pingCb);
    mocker_clean();
}

void TcRsPingUrmaPostSendRc()
{
    struct RsPingCtxCb pingCb = {0};
    struct RsPingTargetInfo target = {0};
    urma_jetty_t jetty = {0};
    urma_jfc_t jfc = {0};
    void* addr = malloc(256);
    urma_sge_t sge = {0};
    int ret = 0;

    target.payloadBuffer = malloc(1);
    target.payloadSize = 1;
    sge.addr = (uintptr_t)addr;
    sge.len = 256;
    pingCb.udevCb.transMode = URMA_TM_RC;
    pingCb.pingJetty.sendSegCb.sgeNum = 1;
    pingCb.pingJetty.sendSegCb.sgeList = &sge;
    pingCb.pingJetty.jetty = &jetty;
    pingCb.pingJetty.sendJfc.jfc = &jfc;
    mocker(RsGetJettyInfo, 1, 0);
    mocker(gettimeofday, 1, 0);
    mocker(RsUrmaPostJettySendWr, 1, 0);
    ret = RsPingUrmaPostSend(&pingCb, &target);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
    free(addr);
    free(target.payloadBuffer);

    addr = malloc(256);
    target.payloadBuffer = malloc(1);
    target.payloadSize = 1;
    sge.addr = (uintptr_t)addr;
    sge.len = 256;
    pingCb.udevCb.transMode = URMA_TM_RC;
    pingCb.pingJetty.sendSegCb.sgeNum = 1;
    pingCb.pingJetty.sendSegCb.sgeList = &sge;
    pingCb.pingJetty.jetty = &jetty;
    pingCb.pingJetty.sendJfc.jfc = &jfc;
    mocker(RsGetJettyInfo, 1, 0);
    mocker(gettimeofday, 1, 0);
    mocker(RsUrmaPostJettySendWr, 1, -1);
    ret = RsPingUrmaPostSend(&pingCb, &target);
    EXPECT_INT_NE(ret, 0);
    mocker_clean();
    free(addr);
    free(target.payloadBuffer);
}

void TcRsPingUrmaFreeTargetNodeRc()
{
    struct RsPingCtxCb pingCb = {0};
    struct RsPingTargetInfo targetInfo = {0};
    urma_jetty_t jetty = {0};

    RS_INIT_LIST_HEAD(&pingCb.pingList);
    RS_INIT_LIST_HEAD(&targetInfo.list);
    RsListAddTail(&targetInfo.list, &pingCb.pingList);
    pthread_mutex_init(&pingCb.pingMutex, NULL);
    pingCb.udevCb.transMode = URMA_TM_RC;
    pingCb.pingJetty.jetty = &jetty;
    mocker(RsUrmaUnbindJetty, 1, 0);
    RsPingUrmaFreeTargetNode(&pingCb, &targetInfo);
    mocker_clean();
    pthread_mutex_destroy(&pingCb.pingMutex);

    RS_INIT_LIST_HEAD(&pingCb.pingList);
    RS_INIT_LIST_HEAD(&targetInfo.list);
    RsListAddTail(&targetInfo.list, &pingCb.pingList);
    pthread_mutex_init(&pingCb.pingMutex, NULL);
    pingCb.udevCb.transMode = URMA_TM_RC;
    pingCb.pingJetty.jetty = NULL;
    RsPingUrmaFreeTargetNode(&pingCb, &targetInfo);
    pthread_mutex_destroy(&pingCb.pingMutex);
}

void TcRsPingUrmaPingCbInitTransMode()
{
    struct PingInitAttr attr = {0};
    struct PingInitInfo info = {0};
    unsigned int devIndex = 0;
    struct RsPingCtxCb pingCb = {0};
    int ret = 0;

    attr.version = 1;
    attr.ub.transMode = URMA_TM_RC;
    attr.ub.tpType = URMA_RTP;
    mocker_invoke(RsGetRsCb, RsGetRsCbUrmaStub, 1);
    mocker(RsUrmaGetDeviceByEid, 1, 0);
    ret = RsPingUrmaPingCbInit(0, &attr, &info, &devIndex, &pingCb);
    EXPECT_INT_NE(ret, 0);
    EXPECT_INT_EQ((int)pingCb.udevCb.transMode, (int)URMA_TM_RC);
    mocker_clean();

    attr.ub.transMode = URMA_TM_UM;
    attr.ub.tpType = URMA_UTP;
    mocker_invoke(RsGetRsCb, RsGetRsCbUrmaStub, 1);
    mocker(RsUrmaGetDeviceByEid, 1, 0);
    ret = RsPingUrmaPingCbInit(0, &attr, &info, &devIndex, &pingCb);
    EXPECT_INT_NE(ret, 0);
    EXPECT_INT_EQ((int)pingCb.udevCb.transMode, (int)URMA_TM_UM);
    mocker_clean();

    // Illegal transMode: returns -EINVAL at the check, not entering RsUrmaGetDeviceByEid
    attr.version = 1;
    attr.ub.transMode = 0xFFFF;
    attr.ub.tpType = URMA_RTP;
    mocker_invoke(RsGetRsCb, RsGetRsCbUrmaStub, 1);
    ret = RsPingUrmaPingCbInit(0, &attr, &info, &devIndex, &pingCb);
    EXPECT_INT_EQ(ret, -EINVAL);
    mocker_clean();

    // Illegal tpType: likewise returns -EINVAL at the check
    attr.version = 1;
    attr.ub.transMode = URMA_TM_UM;
    attr.ub.tpType = 0xFFFF;
    mocker_invoke(RsGetRsCb, RsGetRsCbUrmaStub, 1);
    ret = RsPingUrmaPingCbInit(0, &attr, &info, &devIndex, &pingCb);
    EXPECT_INT_EQ(ret, -EINVAL);
    mocker_clean();

    // version=0 compat path: old callers do not set transMode/tpType, default to UM/UTP
    struct PingInitAttr attrV0 = {0};
    attrV0.version = 0;
    mocker_invoke(RsGetRsCb, RsGetRsCbUrmaStub, 1);
    mocker(RsUrmaGetDeviceByEid, 1, 0);
    ret = RsPingUrmaPingCbInit(0, &attrV0, &info, &devIndex, &pingCb);
    EXPECT_INT_NE(ret, 0);
    EXPECT_INT_EQ((int)pingCb.udevCb.transMode, (int)URMA_TM_UM);
    EXPECT_INT_EQ((int)pingCb.udevCb.tpType, (int)URMA_UTP);
    mocker_clean();

    // RsPingUrmaGetTpPriority fails: RC+RTP but device priority only offers UTP (UT stub default),
    // no rtp match -> -EOPNOTSUPP. Do not mock GetDeviceByEid (go real -> &g_urma_dev),
    // do not mock QueryDevice (UT stub only sets utp=1).
    struct PingInitAttr attrNoRtp = {0};
    attrNoRtp.version = 1;
    attrNoRtp.ub.transMode = URMA_TM_RC;
    attrNoRtp.ub.tpType = URMA_RTP;
    mocker_invoke(RsGetRsCb, RsGetRsCbUrmaStub, 1);
    ret = RsPingUrmaPingCbInit(0, &attrNoRtp, &info, &devIndex, &pingCb);
    EXPECT_INT_EQ(ret, -EOPNOTSUPP);
    mocker_clean();
}

// hccp struct size/offset assertions after RC+CTP change: host<->device cross-process param consistency (regression
// guard) vs baseline 3038ec444: PingTargetInfo/PingQpInfo/PingInitInfo sizes unchanged (RoCE cross-process safe),
// PingLocalCommInfo/PingInitAttr keep old sizes (52/360; option B: packed ub branch removes u64 alignment bloat, ABI
// unchanged) same PR compiles host+device consistently; cross-version mixing must mind these two.
void TcRsPingStructSizeCheck()
{
    EXPECT_INT_EQ((int)sizeof(struct PingTargetInfo), 1688);
    EXPECT_INT_EQ((int)sizeof(struct PingLocalCommInfo), 52);
    EXPECT_INT_EQ((int)sizeof(struct PingTargetCommInfo), 64);
    EXPECT_INT_EQ((int)sizeof(struct PingQpInfo), 48);
    EXPECT_INT_EQ((int)sizeof(struct PingInitAttr), 360);
    EXPECT_INT_EQ((int)sizeof(struct PingInitInfo), 256);
    // PingTargetInfo tail: targetAddAction(4) + union remoteExInfo(48), ensuring targetAddAction is consistently rw by
    // host/device
    EXPECT_INT_EQ((int)offsetof(struct PingTargetInfo, targetAddAction), 1624);
    EXPECT_INT_EQ((int)sizeof(((struct PingTargetInfo*)0)->remoteExInfo.ub), 48);
    // ub branch localInfo layout: tpHandle/tpPsn for RC link, size 40 (8-aligned)
    EXPECT_INT_EQ((int)sizeof(((struct PingLocalCommInfo*)0)->ub), 36);
}
