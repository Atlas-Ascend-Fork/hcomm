/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCP_TEST_HELPERS_H
#define HCCP_TEST_HELPERS_H

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "hccp_sys_test_common.h"
#include "hccp.h"
#include "hccp_vm_net.h"
#include "hccp_ctx.h"
#include "hccp_shm_infra.h"

static inline int RaSocketSendAll(const void* fd, const void* data, unsigned long long size)
{
    unsigned long long left = size, s = 0;
    const char* p = (const char*)data;
    for (int retry = 0; left > 0;) {
        int ret = RaSocketSend(fd, p, left, &s);
        if (ret == 0) {
            left -= s;
            p += s;
            retry = 0;
            continue;
        }
        if (++retry > 5000)
            return ret;
        usleep(1000);
    }
    return 0;
}

static inline int RaSocketRecvAll(const void* fd, void* data, unsigned long long size)
{
    unsigned long long left = size, s = 0;
    char* p = (char*)data;
    for (int retry = 0; left > 0;) {
        int ret = RaSocketRecv(fd, p, left, &s);
        if (ret == 0) {
            left -= s;
            p += s;
            retry = 0;
            continue;
        }
        if (++retry > 5000)
            return ret;
        usleep(1000);
    }
    return 0;
}

struct UbRendezvous {
    struct QpKey qpKey;
    struct MemKey memKey;
};

template <typename T>
static inline int ExchangePeer(const void* fd, const T& mine, T* peer)
{
    int ret = RaSocketSendAll(fd, &mine, sizeof(mine));
    if (ret == 0) {
        ret = RaSocketRecvAll(fd, peer, sizeof(*peer));
    }
    return ret;
}

static inline int SocketBarrier(const void* fd)
{
    uint8_t done = 1;
    uint8_t peerDone = 0;
    int ret = RaSocketSendAll(fd, &done, sizeof(done));
    if (ret == 0) {
        ret = RaSocketRecvAll(fd, &peerDone, sizeof(peerDone));
    }
    return ret;
}

/* Record value the ST host ibverbs stub writes into the peer notify unit when emulating the
 * WRITE_WITH_NOTIFY hardware side effect (ibverbs_stub.c ibv_stub_notify_side_effect). Same
 * convention as the business notify source buffer, which carries the value 1
 * ("notify value 1 means record", transport_ibverbs.cc CreateNotifyValueBuffer). */
#define NOTIFY_BARRIER_RECORD_VALUE 1ULL

/* rs QP state value RS_QP_STATUS_CONNECTED (rs.h is not exported to the cases). The lite
 * local/remote MR tables that the RaSendWr barrier path resolves addresses from are only
 * filled once the handshake stream (qpInfo plus the full mrList) has been received, i.e.
 * after the state leaves CONNECTING(3) and reaches CONNECTED(1), so the notify cases must
 * wait for this exact state instead of breaking on any non-(-1) sample. */
#define TEST_QP_STATUS_CONNECTED 1

/* Notify-based one-sided barrier for the verbs data plane (see the notify synchronization
 * note at the top of cases_rdma.cc). Business shape: when the transport supports RDMA notify,
 * the last WQE of a data batch is switched to RA_WR_RDMA_WRITE_WITH_NOTIFY with
 * aux.notifyOffset selecting the peer notify unit exchanged during link setup
 * (TransportIbverbs::TxSendDataAndNotifyWithSingleQP, transport_ibverbs.cc:1279-1287); the
 * hardware then writes a record value into that unit once the payload has landed, and the
 * peer observes the record instead of polling its data buffer.
 *
 * ST simulation: NotifySyncInit publishes the local notify page base into the shm registry
 * (hccp_notify_publish) before the QP handshake; the host ibverbs stub resolves the peer
 * notify unit from that registry plus the imm encoding (offset | record flag) the RA lite
 * layer builds for opcode 0xf2 (ra_hdc_lite.c RaHdcLitePostSend), and writes the record value
 * after the payload has landed. Observing the record therefore proves the peer's
 * WRITE_WITH_NOTIFY WR fully completed, so neither side frees a buffer the peer may still
 * reference, and the payload echo check (localTarget vs expectedMarker) keeps the wrong-write
 * detection of the former DONE-marker barrier.
 *
 * Constraints:
 * - NotifySyncInit must run before RaQpConnectAsync: it zeroes the local notify unit, and the
 *   peer can only post its record WR after the bilateral handshake completes, so the zeroing
 *   can never race a record.
 * - The record WR must go through RaSendWr: struct SendWr has no aux/ext union member, and
 *   RaSendWr's lite path (RaHdcSendWr -> RaHdcLiteSendWr -> RaHdcLitePostSend) is the only
 *   public entry that encodes the notify imm correctly. RaSendWrVerbs and the other typical
 *   entries pass through RaHdcLiteTypicalSendWr, whose unconditional liteWr.imm_data
 *   overwrite (ra_hdc_lite.c:1347) drops the record flag.
 * - notifyOffset is fixed at 0 (SendWr carries no aux), so both sides use the notify unit at
 *   offset 0, and one NotifySyncInit arms exactly one barrier: the stub writes a fixed record
 *   value, so re-arming the same unit for a second barrier would race the peer's record.
 * - The data payload rkey is not taken from the WR: the lite path resolves it from the
 *   handshake-exchanged remote MR table by dstAddr range hit (RaHdcLiteGetMr on
 *   qpHdc->remMr, refreshed by every RaGetQpStatus), so the real peer rkey is used. The
 *   notify record write is a hardware-side effect that carries no rkey at all; the ST data
 *   plane does not validate rkeys either (ibverbs stub process_vm path). */
struct NotifySyncCtx {
    int phyid;                    /* logging id */
    void* qpHandle;               /* RaQpCreate handle (lite data plane) */
    volatile uint64_t* localCell; /* local notify unit = notify page base + offset 0 */
    struct SendWr notifyWr;       /* pre-filled WRITE_WITH_NOTIFY record WR */
    struct SgList notifySg;       /* payload sge (lkey resolved by the RA lite path) */
    struct SendWrRsp notifyRsp;
    char* localTarget;          /* buffer the peer's barrier payload lands in */
    const char* expectedMarker; /* expected peer payload content in localTarget */
};

/* Query the local notify page (RaGetNotifyBaseAddr), zero the barrier unit, publish the page
 * base into the ST shm registry for the peer's stub, and pre-fill the WRITE_WITH_NOTIFY WR:
 * payloadAddr/payloadLen select the registered local buffer whose content is written into the
 * peer's dstAddr buffer (a registered remote MR), and the record lands in the peer's notify
 * unit at offset 0. */
static inline int NotifySyncInit(
    struct NotifySyncCtx* ctx, int phyid, void* qpHandle, void* rdevHandle, uint64_t payloadAddr,
    unsigned int payloadLen, uint64_t dstAddr, char* localTarget, const char* expectedMarker)
{
    unsigned long long notifyVa = 0;
    unsigned long long notifySize = 0;
    int ret = RaGetNotifyBaseAddr(rdevHandle, &notifyVa, &notifySize);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaGetNotifyBaseAddr failed ret=%d", ret);
        return ret;
    }

    memset(ctx, 0, sizeof(*ctx));
    ctx->phyid = phyid;
    ctx->qpHandle = qpHandle;
    ctx->localCell = (volatile uint64_t*)(uintptr_t)notifyVa;
    *ctx->localCell = 0;
    ctx->localTarget = localTarget;
    ctx->expectedMarker = expectedMarker;

    ret = hccp_notify_publish(getpid(), notifyVa);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "hccp_notify_publish failed ret=%d", ret);
        return ret;
    }

    ctx->notifySg.addr = payloadAddr;
    ctx->notifySg.len = payloadLen;
    ctx->notifyWr.bufList = &ctx->notifySg;
    ctx->notifyWr.bufNum = 1;
    ctx->notifyWr.dstAddr = dstAddr;
    ctx->notifyWr.op = RA_WR_RDMA_WRITE_WITH_NOTIFY;
    ctx->notifyWr.sendFlag = RA_SEND_SIGNALED;
    TEST_LOG_INFO(
        phyid, "notify sync init: notifyVa=0x%llx size=0x%llx dstAddr=0x%llx", notifyVa, notifySize,
        (unsigned long long)dstAddr);
    return 0;
}

/* Post the pre-filled WRITE_WITH_NOTIFY WR (single WQE: payload write into the peer's buffer
 * plus the hardware notify record into the peer's notify unit) and poll the send CQ for the
 * local completion. The caller prepares the payload content beforehand. */
static inline int NotifyBarrierRecord(struct NotifySyncCtx* ctx)
{
    unsigned char wc[64] = {0};
    int n = 0;
    int ret = RaSendWr(ctx->qpHandle, &ctx->notifyWr, &ctx->notifyRsp);
    if (ret != 0) {
        TEST_LOG_ERROR(ctx->phyid, "RaSendWr WRITE_WITH_NOTIFY failed ret=%d", ret);
        return ret;
    }
    for (int i = 0; i < 5000; i++) {
        n = RaPollCq(ctx->qpHandle, true, 1, wc);
        if (n > 0) {
            break;
        }
        usleep(1000);
    }
    if (n <= 0) {
        TEST_LOG_ERROR(ctx->phyid, "poll sendCq timeout, no WRITE_WITH_NOTIFY completion");
        return -1;
    }
    return 0;
}

/* Poll the local notify unit until the peer's record value shows up, then verify the peer's
 * barrier payload: the stub writes the record after the payload, so the payload must have
 * landed once the record is visible. */
static inline int NotifyBarrierWait(struct NotifySyncCtx* ctx)
{
    int done = 0;
    for (int i = 0; i < 5000; i++) {
        if (*ctx->localCell == NOTIFY_BARRIER_RECORD_VALUE) {
            done = 1;
            break;
        }
        usleep(1000);
    }
    if (done == 0) {
        TEST_LOG_ERROR(
            ctx->phyid, "peer notify record timeout, cell=%llu target=%.64s", (unsigned long long)*ctx->localCell,
            ctx->localTarget);
        return -1;
    }
    if (memcmp(ctx->localTarget, ctx->expectedMarker, strlen(ctx->expectedMarker) + 1) != 0) {
        TEST_LOG_ERROR(ctx->phyid, "notify barrier payload FAIL exp=%s act=%s", ctx->expectedMarker, ctx->localTarget);
        return -1;
    }
    TEST_LOG_INFO(
        ctx->phyid, "peer notify record received: cell=%llu, payload=%s", (unsigned long long)*ctx->localCell,
        ctx->localTarget);
    return 0;
}

/* Full barrier: record the local completion into the peer's notify unit, then wait for the
 * peer's record in the local notify unit. Returns 0 on success. */
static inline int NotifyBarrier(struct NotifySyncCtx* ctx)
{
    int ret = NotifyBarrierRecord(ctx);
    if (ret != 0) {
        return ret;
    }
    return NotifyBarrierWait(ctx);
}

/* Per-run case-port displacement (t5 dual-runner prerequisite): every case
 * meeting port shifts by RUN_ID*1000. N=0 keeps the historical 15666-15950
 * domain byte-for-byte (zero-change promise); N=1 -> 16666-16950 and
 * N=2 -> 17666-17950 stay clear of the HDC service blocks (9999 / 19999 /
 * 29999 + phyId) and of every other runner's rendezvous domain. One formula,
 * two application shapes: the EstablishRendezvous funnel below, and the
 * direct-bind families (pingmesh/socket/socket_hlt/ub_dataplane) which apply
 * it at their per-process port ORIGIN (constant or entry parameter) so the
 * listener and the connector derive the same displaced value. SocketBarrier
 * operates on the already-established fd and never binds a port. */
static inline unsigned short HccpStCasePort(unsigned short port)
{
    return (unsigned short)(port + hccp_st_run_id() * 1000);
}

static inline int EstablishRendezvous(
    int phyid, int nicPosition, const struct SocketInitInfoT& socketInit, unsigned short port, const char* tag,
    bool zeroInit, void** outSocketHandle, struct SocketInfoT* outSock)
{
    port = HccpStCasePort(port);

    void* socketHandle = NULL;
    struct SocketInitInfoT vmSocketInit = socketInit;
    vmSocketInit.rdevInfo.localIp.addr = HccpVmToLoopback(socketInit.rdevInfo.localIp.addr);
    int ret = RaSocketInitV1(nicPosition, vmSocketInit, &socketHandle);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaSocketInitV1 failed ret=%d", ret);
        return ret;
    }

    struct SocketListenInfoT serverSocket;
    struct SocketWlistInfoT whiteList;
    struct SocketConnectInfoT conn;
    struct SocketInfoT sock;
    if (zeroInit) {
        memset(&serverSocket, 0, sizeof(serverSocket));
        memset(&whiteList, 0, sizeof(whiteList));
        memset(&conn, 0, sizeof(conn));
        memset(&sock, 0, sizeof(sock));
    }

    serverSocket.socketHandle = socketHandle;
    serverSocket.port = port;
    if (phyid == 1) {
        ret = RaSocketListenStart(&serverSocket, 1);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaSocketListenStart failed ret=%d", ret);
            return ret;
        }
    }

    int peerPhyId = (phyid == 1) ? 5 : 1;
    uint32_t remoteIface = (nicPosition == NETWORK_PEER_ONLINE) ? HCCP_VM_IFACE_ROCE : HCCP_VM_IFACE_SOCKET;
    struct in_addr remoteLoopback = HccpVmToLoopback(in_addr{htonl(HccpVmNicIp((uint32_t)peerPhyId, remoteIface))});
    uint32_t remoteIpVal = remoteLoopback.s_addr;

    whiteList.remoteIp.addr.s_addr = remoteIpVal;
    whiteList.connLimit = 1;
    snprintf(whiteList.tag, sizeof(whiteList.tag), "%s", tag);
    ret = RaSocketWhiteListAdd(socketHandle, &whiteList, 1);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaSocketWhiteListAdd failed ret=%d", ret);
        return ret;
    }

    conn.socketHandle = socketHandle;
    conn.port = port;
    snprintf(conn.tag, sizeof(conn.tag), "%s", tag);
    conn.remoteIp.addr.s_addr = remoteIpVal;

    sock.socketHandle = socketHandle;
    sock.remoteIp.addr.s_addr = remoteIpVal;
    sock.fdHandle = NULL;
    snprintf(sock.tag, sizeof(sock.tag), "%s", tag);

    unsigned int num = 0;
    if (phyid == 5) {
        ret = RaSocketBatchConnect(&conn, 1);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaSocketBatchConnect failed ret=%d", ret);
            return ret;
        }
        while (num <= 0) {
            ret = RaGetSockets(1, &sock, 1, &num);
            if (ret != 0)
                break;
            usleep(100000);
        }
    } else {
        while (num <= 0) {
            ret = RaGetSockets(0, &sock, 1, &num);
            if (ret != 0)
                break;
            usleep(100000);
        }
    }
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaGetSockets failed ret=%d", ret);
        return ret;
    }

    *outSocketHandle = socketHandle;
    *outSock = sock;
    return 0;
}

#endif
