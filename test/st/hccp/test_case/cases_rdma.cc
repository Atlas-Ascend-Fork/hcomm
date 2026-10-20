/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cases_common.h"

/* Notify synchronization note (design for the one-sided verbs cases): the round-trip cases
 * end with a notify-based barrier (shared helper NotifyBarrier in hccp_test_helpers.h)
 * instead of the former DONE-marker polling: each side posts a single WRITE_WITH_NOTIFY
 * (opcode 0xf2) work request through RaSendWr whose payload is the final marker written into
 * the peer's target buffer, while the notify side effect records a value into the peer's
 * notify unit. This mirrors the business notify fast path, where the last WQE of a data
 * batch is switched to RA_WR_RDMA_WRITE_WITH_NOTIFY with aux.notifyOffset selecting the peer
 * notify address exchanged during link setup
 * (TransportIbverbs::TxSendDataAndNotifyWithSingleQP,
 * src/legacy/ascend910/platform/resource/transport/host/transport_ibverbs.cc:1279-1287), and
 * the notify source buffer carries the record value 1 ("notify value 1 means record",
 * transport_ibverbs.cc CreateNotifyValueBuffer). On real hardware that fallback is what runs
 * today (IsSupportRdmaNotify() returns false, transport_ibverbs.cc:1148, so an explicit
 * notify WRITE is appended instead), but the notify-page lifecycle is identical: the page is
 * allocated and configured by RaRdevInit, registered as an MR, and its address is exchanged
 * with the peer, so polling the notify unit is the production completion mechanism.
 *
 * ST simulation of the WRITE_WITH_NOTIFY side effect: the RA lite layer encodes the imm
 * (notifyOffset | record flag) for opcode 0xf2 (ra_hdc_lite.c RaHdcLitePostSend); the host
 * ibverbs stub decodes it after the payload has landed and writes the record value into the
 * peer notify unit, whose base address the case publishes into the ST shm registry before
 * the QP handshake (hccp_notify_publish / hccp_notify_query), mirroring the business MemMsg
 * exchange of notify addresses (transport_ibverbs.cc GetRemoteNotifyAddr). Observing the
 * record proves the peer's barrier WR fully landed (payload first, record second), so neither
 * side frees a buffer the peer may still reference, and the payload echo check keeps the
 * wrong-write detection of the former DONE-marker barrier. See the NotifySyncCtx comment in
 * hccp_test_helpers.h for the RaSendWr entry-point choice and the rkey notes. */

/* Two-node verbs RDMA SEND/RECV (device lite data plane): after socket rendezvous,
 * RaRdevInit + RaQpCreate, both MRs registered and the QP connected asynchronously; each
 * side posts one RECV and one SEND, polls both send/recv CQs for completions, and
 * memcmp-verifies the received message against the peer's content. */
TEST_F(HccpSystemTest, d2dRdmaSendRecv)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        int qpStatus = -1;
        char rdmaSend[64], rdmaRecv[64] = {0};
        struct MrInfoT sendMr = {}, recvMr = {};
        struct SgList recvSg = {}, sendSg = {};
        struct RecvWrVerbs recvWr = {};
        struct SendWrVerbs sendWr = {};
        struct SendWrRsp sendRsp = {};
        unsigned char swc[64] = {0}, rwc[64] = {0};
        int n = 0;
        unsigned long long notifyVa = 0, notifySize = 0;

        const char* expected = (phyid == 1) ? "RDMA from phyId 5" : "RDMA from phyId 1";
        CaseEnv env;
        int ret = CaseOfflineInit(env, phyid, HCCP_VM_IFACE_SOCKET, 15666, "HELLO WORD!");
        if (ret != 0) {
            return ret;
        }

        ret = CaseRdevInit(env, phyid, 1, 1, HCCP_VM_IFACE_SOCKET);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaRdevInit failed ret=%d", ret);
            goto err;
        }

        ret = RaQpCreate(env.rdmaHandle, 0, 2, &env.qpHandle);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaQpCreate failed ret=%d", ret);
            goto err;
        }

        ret = RaGetNotifyBaseAddr(env.rdmaHandle, &notifyVa, &notifySize);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaGetNotifyBaseAddr failed ret=%d", ret);
            goto err;
        }

        (void)snprintf(rdmaSend, sizeof(rdmaSend), "RDMA from phyId %d", phyid);
        sendMr.addr = rdmaSend;
        sendMr.size = sizeof(rdmaSend);
        recvMr.addr = rdmaRecv;
        recvMr.size = sizeof(rdmaRecv);
        ret = RaMrReg(env.qpHandle, &sendMr);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaMrReg send failed ret=%d", ret);
            goto err;
        }
        ret = RaMrReg(env.qpHandle, &recvMr);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaMrReg recv failed ret=%d", ret);
            goto err;
        }

        ret = RaQpConnectAsync(env.qpHandle, env.sock.fdHandle);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaQpConnectAsync failed ret=%d", ret);
            goto err;
        }

        for (int i = 0; i < 600; i++) {
            ret = RaGetQpStatus(env.qpHandle, &qpStatus);
            if (ret != 0 || qpStatus == TEST_QP_STATUS_CONNECTED)
                break;
            usleep(5000);
        }
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaGetQpStatus failed ret=%d", ret);
            goto err;
        }
        if (qpStatus != TEST_QP_STATUS_CONNECTED) {
            TEST_LOG_ERROR(
                phyid,
                "QP handshake not settled within 3s: qp_status=%d (2=TIMEOUT: handshake send "
                "failed; 3=CONNECTING: peer handshake bytes incomplete - check the peer process)",
                qpStatus);
            ret = -1;
            goto err;
        }
        TEST_LOG_INFO(phyid, "QP connected, qp_status=%d pid=%d", qpStatus, getpid());

        recvSg.addr = (uint64_t)(uintptr_t)rdmaRecv;
        recvSg.len = sizeof(rdmaRecv);
        recvSg.lkey = recvMr.lkey;
        recvWr.wrId = 0xA1;
        recvWr.sgList = &recvSg;
        recvWr.numSge = 1;
        ret = RaRecvWrVerbs(env.qpHandle, &recvWr);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaRecvWrVerbs failed ret=%d (lite data plane not enabled?)", ret);
            goto err;
        }

        sendSg.addr = (uint64_t)(uintptr_t)rdmaSend;
        sendSg.len = (uint32_t)(strlen(rdmaSend) + 1);
        sendSg.lkey = sendMr.lkey;
        sendWr.wrId = 0xB2;
        sendWr.sgList = &sendSg;
        sendWr.numSge = 1;
        sendWr.opcode = 2;
        sendWr.sendFlags = 2;
        ret = RaSendWrVerbs(env.qpHandle, &sendWr, &sendRsp);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaSendWrVerbs failed ret=%d (lite data plane not enabled?)", ret);
            goto err;
        }

        for (int i = 0; i < 5000; i++) {
            n = RaPollCq(env.qpHandle, true, 1, swc);
            if (n > 0)
                break;
            usleep(1000);
        }
        if (n <= 0) {
            TEST_LOG_ERROR(phyid, "poll sendCq timeout (no SEND completion)");
            ret = -1;
            goto err;
        }
        n = 0;
        for (int i = 0; i < 5000; i++) {
            n = RaPollCq(env.qpHandle, false, 1, rwc);
            if (n > 0)
                break;
            usleep(1000);
        }
        if (n <= 0) {
            TEST_LOG_ERROR(phyid, "poll recvCq timeout (no RECV completion)");
            ret = -1;
            goto err;
        }
        TEST_LOG_INFO(
            phyid, "poll CQ ok: send opcode=%u status=%u, recv opcode=%u status=%u byte_len=%u", *(uint32_t*)(swc + 12),
            *(uint32_t*)(swc + 8), *(uint32_t*)(rwc + 12), *(uint32_t*)(rwc + 8), *(uint32_t*)(rwc + 20));

        if (memcmp(rdmaRecv, expected, strlen(expected) + 1) != 0) {
            TEST_LOG_ERROR(phyid, "RDMA memcmp FAIL exp=%s act=%s", expected, rdmaRecv);
            ret = -1;
            goto err;
        }
        TEST_LOG_INFO(phyid, "RDMA memcmp PASS: recv=%s", rdmaRecv);
        ret = 0;
    err:
        CaseCleanup(env);
        return ret;
    });
}

/* Two-node verbs RDMA WRITE (OFFLINE/lite path, one-sided into the peer's registered
 * memory): setup mirrors d2dRdmaSendRecv (CaseOfflineInit + CaseRdevInit + QP/MR); each
 * side exchanges its recv-buffer address/rkey over the socket BEFORE the QP handshake
 * (the handshake records ride the same connection, so case data must be exchanged
 * first), then connects the QP, posts one RDMA WRITE pushing its 256B heap buffer into
 * the peer's recv buffer and polls its send CQ for the completion; finally each side
 * polls its own recv buffer until the peer's content shows up, proving the one-sided
 * write landed in the peer's memory. */
TEST_F(HccpSystemTest, d2dRdmaWriteRecv)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        struct MrInfoT sendMr = {}, recvMr = {};
        struct SgList sendSg = {};
        struct SendWrVerbs sendWr = {};
        struct SendWrRsp sendRsp = {};
        unsigned char swc[64] = {0};
        int n = 0;
        int matched = 0;
        int qpStatus = -1;
        unsigned int mLen = 256;
        uint64_t peerRecvAddr = 0;
        uint32_t peerRecvRkey = 0;
        uint64_t localRecvAddr = 0;
        uint32_t localRecvRkey = 0;

        const char* expected = (phyid == 1) ? "WRITE from phyId 5" : "WRITE from phyId 1";
        CaseEnv env;
        int ret = CaseOfflineInit(env, phyid, HCCP_VM_IFACE_SOCKET, 15674, "D2D RDMA WRITE");
        if (ret != 0) {
            return ret;
        }

        ret = CaseRdevInit(env, phyid, 1, 1, HCCP_VM_IFACE_SOCKET);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaRdevInit failed ret=%d", ret);
            goto err;
        }

        ret = RaQpCreate(env.rdmaHandle, 0, 2, &env.qpHandle);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaQpCreate failed ret=%d", ret);
            goto err;
        }

        env.heapBuf[0] = (char*)malloc(mLen);
        env.heapBuf[1] = (char*)calloc(1, mLen);
        if (env.heapBuf[0] == nullptr || env.heapBuf[1] == nullptr) {
            ret = -1;
            goto err;
        }
        (void)snprintf(env.heapBuf[0], mLen, "WRITE from phyId %d", phyid);
        sendMr.addr = env.heapBuf[0];
        sendMr.size = mLen;
        recvMr.addr = env.heapBuf[1];
        recvMr.size = mLen;
        ret = RaMrReg(env.qpHandle, &sendMr);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaMrReg send failed ret=%d", ret);
            goto err;
        }
        ret = RaMrReg(env.qpHandle, &recvMr);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaMrReg recv failed ret=%d", ret);
            goto err;
        }

        localRecvAddr = (uint64_t)(uintptr_t)env.heapBuf[1];
        localRecvRkey = recvMr.rkey;
        ret = RaSocketSendAll(env.sock.fdHandle, &localRecvAddr, sizeof(localRecvAddr));
        ret |= RaSocketSendAll(env.sock.fdHandle, &localRecvRkey, sizeof(localRecvRkey));
        ret |= RaSocketRecvAll(env.sock.fdHandle, &peerRecvAddr, sizeof(peerRecvAddr));
        ret |= RaSocketRecvAll(env.sock.fdHandle, &peerRecvRkey, sizeof(peerRecvRkey));
        if (ret != 0 || peerRecvAddr == 0) {
            TEST_LOG_ERROR(phyid, "addr/rkey exchange failed ret=%d", ret);
            ret = -1;
            goto err;
        }

        ret = RaQpConnectAsync(env.qpHandle, env.sock.fdHandle);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaQpConnectAsync failed ret=%d", ret);
            goto err;
        }

        for (int i = 0; i < 600; i++) {
            ret = RaGetQpStatus(env.qpHandle, &qpStatus);
            if (ret != 0 || qpStatus == TEST_QP_STATUS_CONNECTED)
                break;
            usleep(5000);
        }
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaGetQpStatus failed ret=%d", ret);
            goto err;
        }
        if (qpStatus != TEST_QP_STATUS_CONNECTED) {
            TEST_LOG_ERROR(
                phyid,
                "QP handshake not settled within 3s: qp_status=%d (2=TIMEOUT: handshake send "
                "failed; 3=CONNECTING: peer handshake bytes incomplete - check the peer process)",
                qpStatus);
            ret = -1;
            goto err;
        }
        TEST_LOG_INFO(phyid, "QP connected, qp_status=%d pid=%d", qpStatus, getpid());

        sendSg.addr = (uint64_t)(uintptr_t)env.heapBuf[0];
        sendSg.len = mLen;
        sendSg.lkey = sendMr.lkey;
        sendWr.wrId = 0xC3;
        sendWr.sgList = &sendSg;
        sendWr.numSge = 1;
        sendWr.opcode = 0;
        sendWr.sendFlags = 2;
        sendWr.remoteAddr = peerRecvAddr;
        sendWr.rkey = peerRecvRkey;
        ret = RaSendWrVerbs(env.qpHandle, &sendWr, &sendRsp);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaSendWrVerbs WRITE failed ret=%d", ret);
            goto err;
        }

        for (int i = 0; i < 5000; i++) {
            n = RaPollCq(env.qpHandle, true, 1, swc);
            if (n > 0)
                break;
            usleep(1000);
        }
        if (n <= 0) {
            TEST_LOG_ERROR(phyid, "poll sendCq timeout, no WRITE completion");
            ret = -1;
            goto err;
        }

        matched = 0;
        for (int i = 0; i < 5000; i++) {
            if (memcmp(env.heapBuf[1], expected, strlen(expected) + 1) == 0) {
                matched = 1;
                break;
            }
            usleep(1000);
        }
        if (matched == 0) {
            TEST_LOG_ERROR(phyid, "WRITE memcmp FAIL exp=%s act=%s", expected, env.heapBuf[1]);
            ret = -1;
            goto err;
        }
        TEST_LOG_INFO(phyid, "WRITE memcmp PASS: recv=%s", env.heapBuf[1]);
        ret = 0;
    err:
        CaseCleanup(env);
        return ret;
    });
}

/* Two-node verbs RDMA WRITE+READ round trip (OFFLINE/lite path, one-sided both ways):
 * setup as d2dRdmaWriteRecv (addr/rkey exchange first, then QP connect); each side posts
 * one RDMA WRITE pushing its own content into the peer's target buffer (send-CQ
 * completion), then one RDMA READ pulling that same peer buffer back into a local
 * readback buffer (send-CQ completion); memcmp verifies the readback equals the local
 * content, proving the write landed in the peer's memory and the read retrieved it.
 * Because the READ depends on the peer's target buffer still being mapped, each side then
 * runs the shared NotifyBarrier (WRITE_WITH_NOTIFY marker write into the peer's target plus
 * waiting for the record in its own notify unit), so neither side frees its buffers while
 * the peer may still read them. */
TEST_F(HccpSystemTest, d2dRdmaWriteRead)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        struct MrInfoT sendMr = {}, targetMr = {}, rdMr = {};
        struct SgList writeSg = {}, readSg = {};
        struct SendWrVerbs writeWr = {}, readWr = {};
        struct SendWrRsp writeRsp = {}, readRsp = {};
        struct NotifySyncCtx notifyCtx = {};
        unsigned char wc1[64] = {0}, wc2[64] = {0};
        int n = 0;
        int qpStatus = -1;
        unsigned int mLen = 256;
        uint64_t peerTargetAddr = 0;
        uint32_t peerTargetRkey = 0;
        uint64_t localTargetAddr = 0;
        uint32_t localTargetRkey = 0;
        char rdBuf[256] = {0};

        const char* mine = (phyid == 1) ? "WREAD from phyId 1" : "WREAD from phyId 5";
        const char* doneExp = (phyid == 1) ? "DONE from phyId 5" : "DONE from phyId 1";
        CaseEnv env;
        int ret = CaseOfflineInit(env, phyid, HCCP_VM_IFACE_SOCKET, 15675, "D2D RDMA WREAD");
        if (ret != 0) {
            return ret;
        }

        ret = CaseRdevInit(env, phyid, 1, 1, HCCP_VM_IFACE_SOCKET);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaRdevInit failed ret=%d", ret);
            goto err;
        }

        ret = RaQpCreate(env.rdmaHandle, 0, 2, &env.qpHandle);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaQpCreate failed ret=%d", ret);
            goto err;
        }

        env.heapBuf[0] = (char*)malloc(mLen);
        env.heapBuf[1] = (char*)calloc(1, mLen);
        if (env.heapBuf[0] == nullptr || env.heapBuf[1] == nullptr) {
            ret = -1;
            goto err;
        }
        (void)snprintf(env.heapBuf[0], mLen, "WREAD from phyId %d", phyid);
        sendMr.addr = env.heapBuf[0];
        sendMr.size = mLen;
        targetMr.addr = env.heapBuf[1];
        targetMr.size = mLen;
        rdMr.addr = rdBuf;
        rdMr.size = sizeof(rdBuf);
        ret = RaMrReg(env.qpHandle, &sendMr);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaMrReg send failed ret=%d", ret);
            goto err;
        }
        ret = RaMrReg(env.qpHandle, &targetMr);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaMrReg target failed ret=%d", ret);
            goto err;
        }
        ret = RaMrReg(env.qpHandle, &rdMr);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaMrReg readback failed ret=%d", ret);
            goto err;
        }

        localTargetAddr = (uint64_t)(uintptr_t)env.heapBuf[1];
        localTargetRkey = targetMr.rkey;
        ret = RaSocketSendAll(env.sock.fdHandle, &localTargetAddr, sizeof(localTargetAddr));
        ret |= RaSocketSendAll(env.sock.fdHandle, &localTargetRkey, sizeof(localTargetRkey));
        ret |= RaSocketRecvAll(env.sock.fdHandle, &peerTargetAddr, sizeof(peerTargetAddr));
        ret |= RaSocketRecvAll(env.sock.fdHandle, &peerTargetRkey, sizeof(peerTargetRkey));
        if (ret != 0 || peerTargetAddr == 0) {
            TEST_LOG_ERROR(phyid, "addr/rkey exchange failed ret=%d", ret);
            ret = -1;
            goto err;
        }

        /* Publish the notify page and pre-fill the WRITE_WITH_NOTIFY barrier WR before the
         * QP handshake (the local notify unit must be zeroed before any record can arrive). */
        ret = NotifySyncInit(
            &notifyCtx, phyid, env.qpHandle, env.rdmaHandle, (uint64_t)(uintptr_t)env.heapBuf[0], mLen, peerTargetAddr,
            env.heapBuf[1], doneExp);
        if (ret != 0) {
            goto err;
        }

        ret = RaQpConnectAsync(env.qpHandle, env.sock.fdHandle);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaQpConnectAsync failed ret=%d", ret);
            goto err;
        }

        /* Wait for the full CONNECTED state: the notify barrier WR goes through RaSendWr,
         * whose lite path resolves MRs from the tables that are only filled once the
         * handshake mrList exchange has completed. */
        for (int i = 0; i < 600; i++) {
            ret = RaGetQpStatus(env.qpHandle, &qpStatus);
            if (ret != 0 || qpStatus == TEST_QP_STATUS_CONNECTED)
                break;
            usleep(5000);
        }
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaGetQpStatus failed ret=%d", ret);
            goto err;
        }
        if (qpStatus != TEST_QP_STATUS_CONNECTED) {
            TEST_LOG_ERROR(
                phyid,
                "QP handshake not settled within 3s: qp_status=%d (2=TIMEOUT: handshake send "
                "failed; 3=CONNECTING: peer handshake bytes incomplete - check the peer process)",
                qpStatus);
            ret = -1;
            goto err;
        }
        TEST_LOG_INFO(phyid, "QP connected, qp_status=%d pid=%d", qpStatus, getpid());

        writeSg.addr = (uint64_t)(uintptr_t)env.heapBuf[0];
        writeSg.len = mLen;
        writeSg.lkey = sendMr.lkey;
        writeWr.wrId = 0xD4;
        writeWr.sgList = &writeSg;
        writeWr.numSge = 1;
        writeWr.opcode = 0;
        writeWr.sendFlags = 2;
        writeWr.remoteAddr = peerTargetAddr;
        writeWr.rkey = peerTargetRkey;
        ret = RaSendWrVerbs(env.qpHandle, &writeWr, &writeRsp);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaSendWrVerbs WRITE failed ret=%d", ret);
            goto err;
        }

        for (int i = 0; i < 5000; i++) {
            n = RaPollCq(env.qpHandle, true, 1, wc1);
            if (n > 0)
                break;
            usleep(1000);
        }
        if (n <= 0) {
            TEST_LOG_ERROR(phyid, "poll sendCq timeout, no WRITE completion");
            ret = -1;
            goto err;
        }

        readSg.addr = (uint64_t)(uintptr_t)rdBuf;
        readSg.len = sizeof(rdBuf);
        readSg.lkey = rdMr.lkey;
        readWr.wrId = 0xE5;
        readWr.sgList = &readSg;
        readWr.numSge = 1;
        readWr.opcode = 4;
        readWr.sendFlags = 2;
        readWr.remoteAddr = peerTargetAddr;
        readWr.rkey = peerTargetRkey;
        ret = RaSendWrVerbs(env.qpHandle, &readWr, &readRsp);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaSendWrVerbs READ failed ret=%d", ret);
            goto err;
        }

        for (int i = 0; i < 5000; i++) {
            n = RaPollCq(env.qpHandle, true, 1, wc2);
            if (n > 0)
                break;
            usleep(1000);
        }
        if (n <= 0) {
            TEST_LOG_ERROR(phyid, "poll sendCq timeout, no READ completion");
            ret = -1;
            goto err;
        }

        if (memcmp(rdBuf, mine, strlen(mine) + 1) != 0) {
            TEST_LOG_ERROR(phyid, "WREAD round-trip memcmp FAIL exp=%s act=%s", mine, rdBuf);
            ret = -1;
            goto err;
        }
        TEST_LOG_INFO(phyid, "WREAD round-trip memcmp PASS: result=%s", rdBuf);

        (void)snprintf(env.heapBuf[0], mLen, "DONE from phyId %d", phyid);
        ret = NotifyBarrier(&notifyCtx);
        if (ret != 0) {
            goto err;
        }
        ret = 0;
    err:
        CaseCleanup(env);
        return ret;
    });
}

/* Two-node h2h RoCE WRITE (PEER_ONLINE, NIC on the host side, process_vm writes straight
 * into the peer): after peer rendezvous + QP connection, the local recv buffer
 * address/rkey is exchanged over the socket; RaSendWrlist issues a WRITE that pushes the
 * local 256B buffer directly into the peer's recv buffer; after a barrier, memcmp
 * verifies the local side received the peer's content. */
TEST_F(HccpSystemTest, h2hRdmaWrite)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        struct MrInfoT sendMr = {}, recvMr = {};
        struct SendWrlistData wr = {};
        unsigned int mLen = 256;
        uint64_t peerRecvAddr = 0;
        uint32_t peerRecvRkey = 0;
        uint64_t localRecvAddr = 0;
        uint32_t localRecvRkey = 0;
        struct SendWrRsp rsp = {};
        unsigned int completeNum = 0;
        int qpStatus = -1;

        const char* expected = (phyid == 1) ? "RDMA from phyId 5" : "RDMA from phyId 1";
        CaseEnv env;
        int ret = CasePeerInit(env, phyid, HCCP_VM_IFACE_ROCE, 15671, "H2H RDMA WRITE!");
        if (ret != 0) {
            return ret;
        }

        /* Test-protocol side channel: the QP rendezvous socket is drained by the RA
         * layer while RaGetQpStatus polls (RsGetQpStatus consumes pending handshake
         * bytes on every uncertified call), so the case's own protocol bytes (addr/rkey
         * exchange, barrier) must travel on a separate rendezvous. Sharing one stream
         * lets the side that certifies first have its protocol bytes swallowed by the
         * peer's still-running drain, which poisons the handshake accounting - the
         * failure mode the removed 1s pre-sleep barrier used to paper over. */
        void* xchgSocketHandle = NULL;
        struct SocketInfoT xchgSock = {};
        ret = EstablishRendezvous(
            phyid, 0, env.sockInit, 15678, "H2H RDMA WRITE XCHG", true, &xchgSocketHandle, &xchgSock);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "exchange rendezvous failed ret=%d", ret);
            goto err;
        }

        ret = CaseRdevInit(env, phyid, 0, 0, HCCP_VM_IFACE_ROCE);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaRdevInit failed ret=%d", ret);
            goto err;
        }

        ret = RaQpCreate(env.rdmaHandle, 0, 0, &env.qpHandle);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaQpCreate failed ret=%d", ret);
            goto err;
        }

        env.heapBuf[0] = (char*)malloc(mLen);
        env.heapBuf[1] = (char*)calloc(1, mLen);
        if (env.heapBuf[0] == nullptr || env.heapBuf[1] == nullptr) {
            ret = -1;
            goto err;
        }
        (void)snprintf(env.heapBuf[0], mLen, "RDMA from phyId %d", phyid);
        sendMr.addr = env.heapBuf[0];
        sendMr.size = mLen;
        ret = RaMrReg(env.qpHandle, &sendMr);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaMrReg send failed ret=%d", ret);
            goto err;
        }
        recvMr.addr = env.heapBuf[1];
        recvMr.size = mLen;
        ret = RaMrReg(env.qpHandle, &recvMr);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaMrReg recv failed ret=%d", ret);
            goto err;
        }

        ret = RaQpConnectAsync(env.qpHandle, env.sock.fdHandle);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaQpConnectAsync failed ret=%d", ret);
            goto err;
        }

        /* Settled-handshake wait: CONNECTED is certified by the product state machine
         * only after the peer's full handshake stream (qpInfo + mrList + lenInfo) has
         * been received and parsed - the remote-MR table the WRITE resolves dstAddr
         * against is filled by that parse (RsSendWrlist -> RsGetMrcb on remMrList). */
        for (int i = 0; i < 600; i++) { /* 3s budget at 5ms granularity */
            ret = RaGetQpStatus(env.qpHandle, &qpStatus);
            if (ret != 0 || qpStatus == TEST_QP_STATUS_CONNECTED)
                break;
            usleep(5000);
        }
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaGetQpStatus failed ret=%d", ret);
            goto err;
        }
        if (qpStatus != TEST_QP_STATUS_CONNECTED) {
            TEST_LOG_ERROR(
                phyid,
                "QP handshake not settled within 3s: qp_status=%d (2=TIMEOUT: handshake send "
                "failed; 3=CONNECTING: peer handshake bytes incomplete - check the peer process)",
                qpStatus);
            ret = -1;
            goto err;
        }
        TEST_LOG_INFO(phyid, "QP connected qp_status=%d pid=%d", qpStatus, getpid());

        localRecvAddr = (uint64_t)(uintptr_t)env.heapBuf[1];
        localRecvRkey = recvMr.rkey;
        ret = RaSocketSendAll(xchgSock.fdHandle, &localRecvAddr, sizeof(localRecvAddr));
        ret |= RaSocketSendAll(xchgSock.fdHandle, &localRecvRkey, sizeof(localRecvRkey));
        ret |= RaSocketRecvAll(xchgSock.fdHandle, &peerRecvAddr, sizeof(peerRecvAddr));
        ret |= RaSocketRecvAll(xchgSock.fdHandle, &peerRecvRkey, sizeof(peerRecvRkey));
        if (ret != 0 || peerRecvAddr == 0) {
            TEST_LOG_ERROR(phyid, "recv addr exchange failed ret=%d", ret);
            ret = -1;
            goto err;
        }

        wr.dstAddr = peerRecvAddr;
        wr.op = 0;
        wr.sendFlags = 2;
        wr.memList.addr = (uint64_t)(uintptr_t)env.heapBuf[0];
        wr.memList.len = mLen;
        wr.memList.lkey = sendMr.lkey;
        ret = RaSendWrlist(env.qpHandle, &wr, &rsp, 1, &completeNum);
        if (ret != 0 || completeNum != 1) {
            TEST_LOG_ERROR(phyid, "RaSendWrlist WRITE failed ret=%d completeNum=%u", ret, completeNum);
            ret = -1;
            goto err;
        }

        (void)SocketBarrier(xchgSock.fdHandle);

        if (memcmp(env.heapBuf[1], expected, strlen(expected) + 1) != 0) {
            TEST_LOG_ERROR(phyid, "WRITE memcmp FAIL exp=%s act=%s", expected, env.heapBuf[1]);
            ret = -1;
            goto err;
        }
        TEST_LOG_INFO(phyid, "WRITE memcmp PASS recv=%s", env.heapBuf[1]);
        ret = 0;
    err:
        if (xchgSocketHandle != NULL) {
            (void)RaSocketDeinit(xchgSocketHandle);
        }
        CaseCleanup(env);
        return ret;
    });
}

/* Two-node h2h RoCE READ (PEER_ONLINE, NIC on the host side, process_vm reads straight
 * from the peer): symmetric to h2hRdmaWrite - the local send buffer address/rkey is
 * exchanged, RaSendWrlist issues a READ that pulls 256B from the peer's send buffer into
 * the local recv buffer, and memcmp verifies it after the barrier. */
TEST_F(HccpSystemTest, h2hRdmaRead)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        struct MrInfoT sendMr = {}, recvMr = {};
        struct SendWrlistData wr = {};
        unsigned int mLen = 256;
        uint64_t peerSendAddr = 0;
        uint32_t peerSendRkey = 0;
        uint64_t localSendAddr = 0;
        uint32_t localSendRkey = 0;
        struct SendWrRsp rsp = {};
        unsigned int completeNum = 0;
        int qpStatus = -1;

        const char* expected = (phyid == 1) ? "READ from phyId 5" : "READ from phyId 1";
        CaseEnv env;
        int ret = CasePeerInit(env, phyid, HCCP_VM_IFACE_ROCE, 15672, "H2H RDMA READ!");
        if (ret != 0) {
            return ret;
        }

        /* Test-protocol side channel, same rationale as h2hRdmaWrite: the QP rendezvous
         * socket is drained by the RA layer's RaGetQpStatus polling, so the addr/rkey
         * exchange and the barrier travel on a separate rendezvous to keep the drain
         * stream free of case protocol bytes. */
        void* xchgSocketHandle = NULL;
        struct SocketInfoT xchgSock = {};
        ret = EstablishRendezvous(
            phyid, 0, env.sockInit, 15679, "H2H RDMA READ XCHG", true, &xchgSocketHandle, &xchgSock);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "exchange rendezvous failed ret=%d", ret);
            goto err;
        }

        ret = CaseRdevInit(env, phyid, 0, 0, HCCP_VM_IFACE_ROCE);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaRdevInit failed ret=%d", ret);
            goto err;
        }

        ret = RaQpCreate(env.rdmaHandle, 0, 0, &env.qpHandle);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaQpCreate failed ret=%d", ret);
            goto err;
        }

        env.heapBuf[0] = (char*)malloc(mLen);
        env.heapBuf[1] = (char*)calloc(1, mLen);
        if (env.heapBuf[0] == nullptr || env.heapBuf[1] == nullptr) {
            ret = -1;
            goto err;
        }
        (void)snprintf(env.heapBuf[0], mLen, "READ from phyId %d", phyid);
        sendMr.addr = env.heapBuf[0];
        sendMr.size = mLen;
        ret = RaMrReg(env.qpHandle, &sendMr);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaMrReg send failed ret=%d", ret);
            goto err;
        }
        recvMr.addr = env.heapBuf[1];
        recvMr.size = mLen;
        ret = RaMrReg(env.qpHandle, &recvMr);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaMrReg recv failed ret=%d", ret);
            goto err;
        }

        ret = RaQpConnectAsync(env.qpHandle, env.sock.fdHandle);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaQpConnectAsync failed ret=%d", ret);
            goto err;
        }

        /* Settled-handshake wait, same criterion as h2hRdmaWrite: CONNECTED certifies
         * that the peer's mrList has landed in the local remote-MR table the READ
         * resolves peerSendAddr against. */
        for (int i = 0; i < 600; i++) { /* 3s budget at 5ms granularity */
            ret = RaGetQpStatus(env.qpHandle, &qpStatus);
            if (ret != 0 || qpStatus == TEST_QP_STATUS_CONNECTED)
                break;
            usleep(5000);
        }
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaGetQpStatus failed ret=%d", ret);
            goto err;
        }
        if (qpStatus != TEST_QP_STATUS_CONNECTED) {
            TEST_LOG_ERROR(
                phyid,
                "QP handshake not settled within 3s: qp_status=%d (2=TIMEOUT: handshake send "
                "failed; 3=CONNECTING: peer handshake bytes incomplete - check the peer process)",
                qpStatus);
            ret = -1;
            goto err;
        }
        TEST_LOG_INFO(phyid, "QP connected qp_status=%d pid=%d", qpStatus, getpid());

        localSendAddr = (uint64_t)(uintptr_t)env.heapBuf[0];
        localSendRkey = sendMr.rkey;
        ret = RaSocketSendAll(xchgSock.fdHandle, &localSendAddr, sizeof(localSendAddr));
        ret |= RaSocketSendAll(xchgSock.fdHandle, &localSendRkey, sizeof(localSendRkey));
        ret |= RaSocketRecvAll(xchgSock.fdHandle, &peerSendAddr, sizeof(peerSendAddr));
        ret |= RaSocketRecvAll(xchgSock.fdHandle, &peerSendRkey, sizeof(peerSendRkey));
        if (ret != 0 || peerSendAddr == 0) {
            TEST_LOG_ERROR(phyid, "send addr exchange failed ret=%d", ret);
            ret = -1;
            goto err;
        }

        wr.dstAddr = peerSendAddr;
        wr.op = 4;
        wr.sendFlags = 2;
        wr.memList.addr = (uint64_t)(uintptr_t)env.heapBuf[1];
        wr.memList.len = mLen;
        wr.memList.lkey = recvMr.lkey;
        ret = RaSendWrlist(env.qpHandle, &wr, &rsp, 1, &completeNum);
        if (ret != 0 || completeNum != 1) {
            TEST_LOG_ERROR(phyid, "RaSendWrlist READ failed ret=%d completeNum=%u", ret, completeNum);
            ret = -1;
            goto err;
        }

        (void)SocketBarrier(xchgSock.fdHandle);

        if (memcmp(env.heapBuf[1], expected, strlen(expected) + 1) != 0) {
            TEST_LOG_ERROR(phyid, "READ memcmp FAIL exp=%s act=%s", expected, env.heapBuf[1]);
            ret = -1;
            goto err;
        }
        TEST_LOG_INFO(phyid, "READ memcmp PASS result=%s", env.heapBuf[1]);
        ret = 0;
    err:
        if (xchgSocketHandle != NULL) {
            (void)RaSocketDeinit(xchgSocketHandle);
        }
        CaseCleanup(env);
        return ret;
    });
}

/* Two-node h2h RoCE SEND/RECV (PEER_ONLINE, NIC on the host side, two-sided verbs message
 * semantics): setup mirrors h2hRdmaWrite (peer rendezvous, NOR QP, heap buffers, two MRs,
 * asynchronous QP connection). The peer-mode verbs trio (RaSendWrVerbs/RaRecvWrVerbs/
 * RaPollCq) is not wired and RaSendWr needs the exp provider op, so the data plane uses
 * RaSendWrlist with op=RA_WR_SEND (needs no remote addr/rkey) plus RaRecvWrlist to post
 * the receive buffer. No socket traffic after the QP handshake: each side posts its
 * receive buffer, sends its 256B message, and polls its own recv buffer; because peer
 * mode exposes no CQ poll, the receive side re-posts the same buffer periodically (each
 * RaRecvWrlist drives the stub's recv drain), and the SEND payload itself lands fully in
 * the shared packet ring at post time, so once memcmp matches on both sides neither side
 * can still hold data the peer needs. */
TEST_F(HccpSystemTest, h2hRdmaSendRecv)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        struct MrInfoT sendMr = {}, recvMr = {};
        struct SendWrlistData wr = {};
        struct RecvWrlistData recvWr = {};
        struct SendWrRsp rsp = {};
        unsigned int mLen = 256;
        unsigned int completeNum = 0;
        int qpStatus = -1;
        int matched = 0;

        const char* expected = (phyid == 1) ? "SENDRECV from phyId 5" : "SENDRECV from phyId 1";
        CaseEnv env;
        int ret = CasePeerInit(env, phyid, HCCP_VM_IFACE_ROCE, 15690, "H2H RDMA SENDRECV");
        if (ret != 0) {
            return ret;
        }

        ret = CaseRdevInit(env, phyid, 0, 0, HCCP_VM_IFACE_ROCE);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaRdevInit failed ret=%d", ret);
            goto err;
        }

        ret = RaQpCreate(env.rdmaHandle, 0, 0, &env.qpHandle);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaQpCreate failed ret=%d", ret);
            goto err;
        }

        env.heapBuf[0] = (char*)malloc(mLen);
        env.heapBuf[1] = (char*)calloc(1, mLen);
        if (env.heapBuf[0] == nullptr || env.heapBuf[1] == nullptr) {
            ret = -1;
            goto err;
        }
        (void)snprintf(env.heapBuf[0], mLen, "SENDRECV from phyId %d", phyid);
        sendMr.addr = env.heapBuf[0];
        sendMr.size = mLen;
        ret = RaMrReg(env.qpHandle, &sendMr);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaMrReg send failed ret=%d", ret);
            goto err;
        }
        recvMr.addr = env.heapBuf[1];
        recvMr.size = mLen;
        ret = RaMrReg(env.qpHandle, &recvMr);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaMrReg recv failed ret=%d", ret);
            goto err;
        }

        ret = RaQpConnectAsync(env.qpHandle, env.sock.fdHandle);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaQpConnectAsync failed ret=%d", ret);
            goto err;
        }

        /* Settled-handshake wait, same criterion as the other h2h cases. SendRecv has
         * no post-poll socket protocol (the tail polls the receive buffer directly), so
         * no side channel is needed: the drain stream only ever carries handshake
         * bytes and the certification skew is harmless. */
        for (int i = 0; i < 600; i++) { /* 3s budget at 5ms granularity */
            ret = RaGetQpStatus(env.qpHandle, &qpStatus);
            if (ret != 0 || qpStatus == TEST_QP_STATUS_CONNECTED)
                break;
            usleep(5000);
        }
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaGetQpStatus failed ret=%d", ret);
            goto err;
        }
        if (qpStatus != TEST_QP_STATUS_CONNECTED) {
            TEST_LOG_ERROR(
                phyid,
                "QP handshake not settled within 3s: qp_status=%d (2=TIMEOUT: handshake send "
                "failed; 3=CONNECTING: peer handshake bytes incomplete - check the peer process)",
                qpStatus);
            ret = -1;
            goto err;
        }
        TEST_LOG_INFO(phyid, "QP connected qp_status=%d pid=%d", qpStatus, getpid());

        recvWr.wrId = 0xA1;
        recvWr.memList.addr = (uint64_t)(uintptr_t)env.heapBuf[1];
        recvWr.memList.len = mLen;
        recvWr.memList.lkey = recvMr.lkey;
        ret = RaRecvWrlist(env.qpHandle, &recvWr, 1, &completeNum);
        if (ret != 0 || completeNum != 1) {
            TEST_LOG_ERROR(phyid, "RaRecvWrlist post failed ret=%d completeNum=%u", ret, completeNum);
            ret = -1;
            goto err;
        }

        /* Two-sided SEND: no remote addr/rkey is needed, only the local MR. */
        wr.op = RA_WR_SEND;
        wr.sendFlags = RA_SEND_SIGNALED;
        wr.memList.addr = (uint64_t)(uintptr_t)env.heapBuf[0];
        wr.memList.len = mLen;
        wr.memList.lkey = sendMr.lkey;
        ret = RaSendWrlist(env.qpHandle, &wr, &rsp, 1, &completeNum);
        if (ret != 0 || completeNum != 1) {
            TEST_LOG_ERROR(phyid, "RaSendWrlist SEND failed ret=%d completeNum=%u", ret, completeNum);
            ret = -1;
            goto err;
        }

        /* Peer mode has no CQ poll: re-post the same receive buffer every 100ms to drive
         * the stub's recv drain (sparse cadence keeps the queued WR count under the stub
         * recv queue depth) until the peer's message shows up in the recv buffer. */
        for (int i = 0; i < 3000; i++) {
            if (memcmp(env.heapBuf[1], expected, strlen(expected) + 1) == 0) {
                matched = 1;
                break;
            }
            if (i > 0 && i % 100 == 0) {
                ret = RaRecvWrlist(env.qpHandle, &recvWr, 1, &completeNum);
                if (ret != 0) {
                    TEST_LOG_ERROR(phyid, "RaRecvWrlist re-post failed ret=%d", ret);
                    goto err;
                }
            }
            usleep(1000);
        }
        if (matched == 0) {
            TEST_LOG_ERROR(phyid, "SEND/RECV memcmp timeout exp=%s act=%s", expected, env.heapBuf[1]);
            ret = -1;
            goto err;
        }
        TEST_LOG_INFO(phyid, "SEND/RECV memcmp PASS: recv=%s", env.heapBuf[1]);
        ret = 0;
    err:
        CaseCleanup(env);
        return ret;
    });
}

/* Two-node verbs RDMA WRITE on device memory (OFFLINE/lite path, production allocation
 * model): the send and receive buffers are allocated through aclrtMalloc
 * (aclrtMalloc -> DevMalloc -> halMemAlloc MEM_DEV|MEM_TYPE_HBM -> per-phyId devmem pool),
 * exactly like a business RDMA buffer; the pool lives in the fixed-VA devmem window, so
 * each returned pointer is at once the device VA (MR registration, sge addresses) and the
 * host VA (direct snprintf/memcmp access). The device-memory pool the provider engine
 * allocates during RaQpCreate (roce_init_mem_pool) is the provider's own context memory
 * and is no longer carved by the case. The buffer addresses are registered as MRs and
 * exchanged over the socket before the QP handshake, then one RDMA WRITE pushes the local
 * buffer across into the peer's device-memory buffer; the send CQ completion is polled
 * and each side polls its own receive buffer (host VA) until the peer's content shows
 * up. */
TEST_F(HccpSystemTest, d2dRdmaDevMemWriteRecv)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        struct MrInfoT sendMr = {}, recvMr = {};
        struct SgList sendSg = {};
        struct SendWrVerbs sendWr = {};
        struct SendWrRsp sendRsp = {};
        unsigned char swc[64] = {0};
        int n = 0;
        int matched = 0;
        int qpStatus = -1;
        unsigned int mLen = 256;
        uint64_t peerRecvDva = 0;
        uint32_t peerRecvRkey = 0;
        uint64_t localRecvDva = 0;
        uint32_t localRecvRkey = 0;
        char* sendHva = nullptr;
        char* recvHva = nullptr;

        const char* expected = (phyid == 1) ? "DEVMEM WRITE from phyId 5" : "DEVMEM WRITE from phyId 1";
        CaseEnv env;
        int ret = CaseOfflineInit(env, phyid, HCCP_VM_IFACE_SOCKET, 15676, "D2D DEVMEM WRITE");
        if (ret != 0) {
            return ret;
        }

        ret = CaseRdevInit(env, phyid, 1, 1, HCCP_VM_IFACE_SOCKET);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaRdevInit failed ret=%d", ret);
            goto err;
        }

        ret = RaQpCreate(env.rdmaHandle, 0, 2, &env.qpHandle);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaQpCreate failed ret=%d", ret);
            goto err;
        }

        /* Production allocation chain: aclrtMalloc -> DevMalloc -> halMemAlloc
         * (MEM_DEV|MEM_TYPE_HBM) -> per-phyId devmem pool. The pool is mapped in the
         * fixed-VA devmem window, so each returned pointer is both the device VA (MR
         * registration, sge address) and the host VA (direct host access). */
        ret = aclrtMalloc(&env.rtBuf[0], mLen, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "aclrtMalloc send failed ret=%d (devmem pool exhausted?)", ret);
            ret = -1;
            goto err;
        }
        ret = aclrtMalloc(&env.rtBuf[1], mLen, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "aclrtMalloc recv failed ret=%d (devmem pool exhausted?)", ret);
            ret = -1;
            goto err;
        }
        sendHva = (char*)env.rtBuf[0];
        recvHva = (char*)env.rtBuf[1];
        TEST_LOG_INFO(
            phyid, "devmem buf (aclrtMalloc): send dva=0x%llx hva=%p, recv dva=0x%llx hva=%p",
            (unsigned long long)(uintptr_t)env.rtBuf[0], sendHva, (unsigned long long)(uintptr_t)env.rtBuf[1], recvHva);

        (void)snprintf(sendHva, mLen, "DEVMEM WRITE from phyId %d", phyid);

        sendMr.addr = env.rtBuf[0]; /* dva equals the returned pointer value */
        sendMr.size = mLen;
        recvMr.addr = env.rtBuf[1];
        recvMr.size = mLen;
        ret = RaMrReg(env.qpHandle, &sendMr);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaMrReg send failed ret=%d", ret);
            goto err;
        }
        ret = RaMrReg(env.qpHandle, &recvMr);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaMrReg recv failed ret=%d", ret);
            goto err;
        }

        localRecvDva = (uint64_t)(uintptr_t)env.rtBuf[1];
        localRecvRkey = recvMr.rkey;
        ret = RaSocketSendAll(env.sock.fdHandle, &localRecvDva, sizeof(localRecvDva));
        ret |= RaSocketSendAll(env.sock.fdHandle, &localRecvRkey, sizeof(localRecvRkey));
        ret |= RaSocketRecvAll(env.sock.fdHandle, &peerRecvDva, sizeof(peerRecvDva));
        ret |= RaSocketRecvAll(env.sock.fdHandle, &peerRecvRkey, sizeof(peerRecvRkey));
        if (ret != 0 || peerRecvDva == 0) {
            TEST_LOG_ERROR(phyid, "addr/rkey exchange failed ret=%d", ret);
            ret = -1;
            goto err;
        }

        ret = RaQpConnectAsync(env.qpHandle, env.sock.fdHandle);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaQpConnectAsync failed ret=%d", ret);
            goto err;
        }

        for (int i = 0; i < 600; i++) {
            ret = RaGetQpStatus(env.qpHandle, &qpStatus);
            if (ret != 0 || qpStatus == TEST_QP_STATUS_CONNECTED)
                break;
            usleep(5000);
        }
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaGetQpStatus failed ret=%d", ret);
            goto err;
        }
        if (qpStatus != TEST_QP_STATUS_CONNECTED) {
            TEST_LOG_ERROR(
                phyid,
                "QP handshake not settled within 3s: qp_status=%d (2=TIMEOUT: handshake send "
                "failed; 3=CONNECTING: peer handshake bytes incomplete - check the peer process)",
                qpStatus);
            ret = -1;
            goto err;
        }
        TEST_LOG_INFO(phyid, "QP connected, qp_status=%d pid=%d", qpStatus, getpid());

        sendSg.addr = (uint64_t)(uintptr_t)env.rtBuf[0];
        sendSg.len = mLen;
        sendSg.lkey = sendMr.lkey;
        sendWr.wrId = 0xC3;
        sendWr.sgList = &sendSg;
        sendWr.numSge = 1;
        sendWr.opcode = 0;
        sendWr.sendFlags = 2;
        sendWr.remoteAddr = peerRecvDva;
        sendWr.rkey = peerRecvRkey;
        ret = RaSendWrVerbs(env.qpHandle, &sendWr, &sendRsp);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaSendWrVerbs WRITE failed ret=%d", ret);
            goto err;
        }

        for (int i = 0; i < 5000; i++) {
            n = RaPollCq(env.qpHandle, true, 1, swc);
            if (n > 0)
                break;
            usleep(1000);
        }
        if (n <= 0) {
            TEST_LOG_ERROR(phyid, "poll sendCq timeout, no WRITE completion");
            ret = -1;
            goto err;
        }

        for (int i = 0; i < 5000; i++) {
            if (memcmp(recvHva, expected, strlen(expected) + 1) == 0) {
                matched = 1;
                break;
            }
            usleep(1000);
        }
        if (matched == 0) {
            TEST_LOG_ERROR(phyid, "DEVMEM WRITE memcmp FAIL exp=%s act=%s", expected, recvHva);
            ret = -1;
            goto err;
        }
        TEST_LOG_INFO(phyid, "DEVMEM WRITE memcmp PASS: recv=%s", recvHva);
        ret = 0;
    err:
        CaseCleanup(env);
        return ret;
    });
}

/* Two-node verbs RDMA WRITE+READ round trip on device memory (OFFLINE/lite path,
 * production allocation model): setup as d2dRdmaDevMemWriteRecv (send/target/readback
 * buffers each allocated via aclrtMalloc from the per-phyId devmem pool, the fixed-VA
 * window making every pointer both a device VA and a host VA, MRs and the peer target
 * dva/rkey exchanged before the QP handshake); each side posts one RDMA WRITE into the
 * peer's target device-memory buffer, then one RDMA READ pulling that same peer buffer
 * back into a local readback device-memory buffer; memcmp on the readback host VA
 * verifies the full dva round trip through the peer's device memory. As in
 * d2dRdmaWriteRead, each side finishes with the shared NotifyBarrier (WRITE_WITH_NOTIFY
 * marker write into the peer's target plus waiting for the record in its own notify
 * unit), so neither side frees its buffers while the peer may still read them. */
TEST_F(HccpSystemTest, d2dRdmaDevMemWriteRead)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        struct MrInfoT sendMr = {}, targetMr = {}, rdMr = {};
        struct SgList writeSg = {}, readSg = {};
        struct SendWrVerbs writeWr = {}, readWr = {};
        struct SendWrRsp writeRsp = {}, readRsp = {};
        struct NotifySyncCtx notifyCtx = {};
        unsigned char wc1[64] = {0}, wc2[64] = {0};
        int n = 0;
        int qpStatus = -1;
        unsigned int mLen = 256;
        uint64_t peerTargetDva = 0;
        uint32_t peerTargetRkey = 0;
        uint64_t localTargetDva = 0;
        uint32_t localTargetRkey = 0;
        char* sendHva = nullptr;
        char* targetHva = nullptr;
        char* rdHva = nullptr;

        const char* mine = (phyid == 1) ? "DEVMEM WREAD from phyId 1" : "DEVMEM WREAD from phyId 5";
        const char* doneExp = (phyid == 1) ? "DONE from phyId 5" : "DONE from phyId 1";
        CaseEnv env;
        int ret = CaseOfflineInit(env, phyid, HCCP_VM_IFACE_SOCKET, 15677, "D2D DEVMEM WREAD");
        if (ret != 0) {
            return ret;
        }

        ret = CaseRdevInit(env, phyid, 1, 1, HCCP_VM_IFACE_SOCKET);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaRdevInit failed ret=%d", ret);
            goto err;
        }

        ret = RaQpCreate(env.rdmaHandle, 0, 2, &env.qpHandle);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaQpCreate failed ret=%d", ret);
            goto err;
        }

        /* Production allocation chain: aclrtMalloc -> DevMalloc -> halMemAlloc
         * (MEM_DEV|MEM_TYPE_HBM) -> per-phyId devmem pool; each pointer is both the
         * device VA and the host VA (fixed-VA devmem window). */
        ret = aclrtMalloc(&env.rtBuf[0], mLen, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "aclrtMalloc send failed ret=%d (devmem pool exhausted?)", ret);
            ret = -1;
            goto err;
        }
        ret = aclrtMalloc(&env.rtBuf[1], mLen, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "aclrtMalloc target failed ret=%d (devmem pool exhausted?)", ret);
            ret = -1;
            goto err;
        }
        ret = aclrtMalloc(&env.rtBuf[2], mLen, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "aclrtMalloc readback failed ret=%d (devmem pool exhausted?)", ret);
            ret = -1;
            goto err;
        }
        sendHva = (char*)env.rtBuf[0];
        targetHva = (char*)env.rtBuf[1];
        rdHva = (char*)env.rtBuf[2];
        TEST_LOG_INFO(
            phyid, "devmem buf (aclrtMalloc): send dva=0x%llx, target dva=0x%llx, readback dva=0x%llx hva=%p",
            (unsigned long long)(uintptr_t)env.rtBuf[0], (unsigned long long)(uintptr_t)env.rtBuf[1],
            (unsigned long long)(uintptr_t)env.rtBuf[2], rdHva);

        (void)snprintf(sendHva, mLen, "DEVMEM WREAD from phyId %d", phyid);

        sendMr.addr = env.rtBuf[0]; /* dva equals the returned pointer value */
        sendMr.size = mLen;
        targetMr.addr = env.rtBuf[1];
        targetMr.size = mLen;
        rdMr.addr = env.rtBuf[2];
        rdMr.size = mLen;
        ret = RaMrReg(env.qpHandle, &sendMr);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaMrReg send failed ret=%d", ret);
            goto err;
        }
        ret = RaMrReg(env.qpHandle, &targetMr);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaMrReg target failed ret=%d", ret);
            goto err;
        }
        ret = RaMrReg(env.qpHandle, &rdMr);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaMrReg readback failed ret=%d", ret);
            goto err;
        }

        localTargetDva = (uint64_t)(uintptr_t)env.rtBuf[1];
        localTargetRkey = targetMr.rkey;
        ret = RaSocketSendAll(env.sock.fdHandle, &localTargetDva, sizeof(localTargetDva));
        ret |= RaSocketSendAll(env.sock.fdHandle, &localTargetRkey, sizeof(localTargetRkey));
        ret |= RaSocketRecvAll(env.sock.fdHandle, &peerTargetDva, sizeof(peerTargetDva));
        ret |= RaSocketRecvAll(env.sock.fdHandle, &peerTargetRkey, sizeof(peerTargetRkey));
        if (ret != 0 || peerTargetDva == 0) {
            TEST_LOG_ERROR(phyid, "addr/rkey exchange failed ret=%d", ret);
            ret = -1;
            goto err;
        }

        /* Publish the notify page and pre-fill the WRITE_WITH_NOTIFY barrier WR before the
         * QP handshake; the payload sge carries the device VA (MR-registered dva) while the
         * marker content is stamped through the host VA at the barrier. */
        ret = NotifySyncInit(
            &notifyCtx, phyid, env.qpHandle, env.rdmaHandle, (uint64_t)(uintptr_t)env.rtBuf[0], mLen, peerTargetDva,
            targetHva, doneExp);
        if (ret != 0) {
            goto err;
        }

        ret = RaQpConnectAsync(env.qpHandle, env.sock.fdHandle);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaQpConnectAsync failed ret=%d", ret);
            goto err;
        }

        /* Wait for the full CONNECTED state: the notify barrier WR goes through RaSendWr,
         * whose lite path resolves MRs from the tables that are only filled once the
         * handshake mrList exchange has completed. */
        for (int i = 0; i < 600; i++) {
            ret = RaGetQpStatus(env.qpHandle, &qpStatus);
            if (ret != 0 || qpStatus == TEST_QP_STATUS_CONNECTED)
                break;
            usleep(5000);
        }
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaGetQpStatus failed ret=%d", ret);
            goto err;
        }
        if (qpStatus != TEST_QP_STATUS_CONNECTED) {
            TEST_LOG_ERROR(
                phyid,
                "QP handshake not settled within 3s: qp_status=%d (2=TIMEOUT: handshake send "
                "failed; 3=CONNECTING: peer handshake bytes incomplete - check the peer process)",
                qpStatus);
            ret = -1;
            goto err;
        }
        TEST_LOG_INFO(phyid, "QP connected, qp_status=%d pid=%d", qpStatus, getpid());

        writeSg.addr = (uint64_t)(uintptr_t)env.rtBuf[0];
        writeSg.len = mLen;
        writeSg.lkey = sendMr.lkey;
        writeWr.wrId = 0xD4;
        writeWr.sgList = &writeSg;
        writeWr.numSge = 1;
        writeWr.opcode = 0;
        writeWr.sendFlags = 2;
        writeWr.remoteAddr = peerTargetDva;
        writeWr.rkey = peerTargetRkey;
        ret = RaSendWrVerbs(env.qpHandle, &writeWr, &writeRsp);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaSendWrVerbs WRITE failed ret=%d", ret);
            goto err;
        }

        for (int i = 0; i < 5000; i++) {
            n = RaPollCq(env.qpHandle, true, 1, wc1);
            if (n > 0)
                break;
            usleep(1000);
        }
        if (n <= 0) {
            TEST_LOG_ERROR(phyid, "poll sendCq timeout, no WRITE completion");
            ret = -1;
            goto err;
        }

        readSg.addr = (uint64_t)(uintptr_t)env.rtBuf[2];
        readSg.len = mLen;
        readSg.lkey = rdMr.lkey;
        readWr.wrId = 0xE5;
        readWr.sgList = &readSg;
        readWr.numSge = 1;
        readWr.opcode = 4;
        readWr.sendFlags = 2;
        readWr.remoteAddr = peerTargetDva;
        readWr.rkey = peerTargetRkey;
        ret = RaSendWrVerbs(env.qpHandle, &readWr, &readRsp);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaSendWrVerbs READ failed ret=%d", ret);
            goto err;
        }

        for (int i = 0; i < 5000; i++) {
            n = RaPollCq(env.qpHandle, true, 1, wc2);
            if (n > 0)
                break;
            usleep(1000);
        }
        if (n <= 0) {
            TEST_LOG_ERROR(phyid, "poll sendCq timeout, no READ completion");
            ret = -1;
            goto err;
        }

        if (memcmp(rdHva, mine, strlen(mine) + 1) != 0) {
            TEST_LOG_ERROR(phyid, "DEVMEM WREAD round-trip memcmp FAIL exp=%s act=%s", mine, rdHva);
            ret = -1;
            goto err;
        }
        TEST_LOG_INFO(phyid, "DEVMEM WREAD round-trip memcmp PASS: result=%s", rdHva);

        (void)snprintf(sendHva, mLen, "DONE from phyId %d", phyid);
        ret = NotifyBarrier(&notifyCtx);
        if (ret != 0) {
            goto err;
        }
        ret = 0;
    err:
        CaseCleanup(env);
        return ret;
    });
}

/* Two-node verbs RDMA WRITE_WITH_NOTIFY (OFFLINE/lite path): the data WRITE itself carries
 * the notify, business fast-path style - a single WQE both pushes the payload into the
 * peer's recv buffer and records a value in the peer's notify unit. Setup mirrors
 * d2dRdmaWriteRecv (CaseOfflineInit + CaseRdevInit + QP/MR, recv addr/rkey exchanged over
 * the socket before the handshake); NotifySyncInit publishes the local notify page and
 * pre-fills the 0xf2 WR before the connect. Each side then stamps its payload, posts the
 * single WRITE_WITH_NOTIFY WR (NotifyBarrierRecord) and waits for the record in its own
 * notify unit (NotifyBarrierWait) instead of polling its recv buffer; the payload is only
 * verified after the record arrives, which the stub guarantees by writing the record after
 * the payload. This proves the WRITE_WITH_NOTIFY semantics end to end: data lands and the
 * notify unit observes the completion. */
TEST_F(HccpSystemTest, d2dRdmaNotifyWriteRecv)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        struct MrInfoT sendMr = {}, recvMr = {};
        struct NotifySyncCtx notifyCtx = {};
        int qpStatus = -1;
        unsigned int mLen = 256;
        uint64_t peerRecvAddr = 0;
        uint32_t peerRecvRkey = 0;
        uint64_t localRecvAddr = 0;
        uint32_t localRecvRkey = 0;

        const char* expected = (phyid == 1) ? "NOTIFY from phyId 5" : "NOTIFY from phyId 1";
        CaseEnv env;
        int ret = CaseOfflineInit(env, phyid, HCCP_VM_IFACE_SOCKET, 15692, "D2D RDMA NOTIFY");
        if (ret != 0) {
            return ret;
        }

        ret = CaseRdevInit(env, phyid, 1, 1, HCCP_VM_IFACE_SOCKET);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaRdevInit failed ret=%d", ret);
            goto err;
        }

        ret = RaQpCreate(env.rdmaHandle, 0, 2, &env.qpHandle);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaQpCreate failed ret=%d", ret);
            goto err;
        }

        env.heapBuf[0] = (char*)malloc(mLen);
        env.heapBuf[1] = (char*)calloc(1, mLen);
        if (env.heapBuf[0] == nullptr || env.heapBuf[1] == nullptr) {
            ret = -1;
            goto err;
        }
        (void)snprintf(env.heapBuf[0], mLen, "NOTIFY from phyId %d", phyid);
        sendMr.addr = env.heapBuf[0];
        sendMr.size = mLen;
        recvMr.addr = env.heapBuf[1];
        recvMr.size = mLen;
        ret = RaMrReg(env.qpHandle, &sendMr);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaMrReg send failed ret=%d", ret);
            goto err;
        }
        ret = RaMrReg(env.qpHandle, &recvMr);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaMrReg recv failed ret=%d", ret);
            goto err;
        }

        localRecvAddr = (uint64_t)(uintptr_t)env.heapBuf[1];
        localRecvRkey = recvMr.rkey;
        ret = RaSocketSendAll(env.sock.fdHandle, &localRecvAddr, sizeof(localRecvAddr));
        ret |= RaSocketSendAll(env.sock.fdHandle, &localRecvRkey, sizeof(localRecvRkey));
        ret |= RaSocketRecvAll(env.sock.fdHandle, &peerRecvAddr, sizeof(peerRecvAddr));
        ret |= RaSocketRecvAll(env.sock.fdHandle, &peerRecvRkey, sizeof(peerRecvRkey));
        if (ret != 0 || peerRecvAddr == 0) {
            TEST_LOG_ERROR(phyid, "addr/rkey exchange failed ret=%d", ret);
            ret = -1;
            goto err;
        }

        /* Publish the notify page and pre-fill the WRITE_WITH_NOTIFY data WR before the QP
         * handshake (the local notify unit must be zeroed before any record can arrive). */
        ret = NotifySyncInit(
            &notifyCtx, phyid, env.qpHandle, env.rdmaHandle, (uint64_t)(uintptr_t)env.heapBuf[0], mLen, peerRecvAddr,
            env.heapBuf[1], expected);
        if (ret != 0) {
            goto err;
        }

        ret = RaQpConnectAsync(env.qpHandle, env.sock.fdHandle);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaQpConnectAsync failed ret=%d", ret);
            goto err;
        }

        /* Wait for the full CONNECTED state: the WRITE_WITH_NOTIFY WR goes through RaSendWr,
         * whose lite path resolves MRs from the tables that are only filled once the
         * handshake mrList exchange has completed. */
        for (int i = 0; i < 600; i++) {
            ret = RaGetQpStatus(env.qpHandle, &qpStatus);
            if (ret != 0 || qpStatus == TEST_QP_STATUS_CONNECTED)
                break;
            usleep(5000);
        }
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaGetQpStatus failed ret=%d", ret);
            goto err;
        }
        if (qpStatus != TEST_QP_STATUS_CONNECTED) {
            TEST_LOG_ERROR(
                phyid,
                "QP handshake not settled within 3s: qp_status=%d (2=TIMEOUT: handshake send "
                "failed; 3=CONNECTING: peer handshake bytes incomplete - check the peer process)",
                qpStatus);
            ret = -1;
            goto err;
        }
        TEST_LOG_INFO(phyid, "QP connected, qp_status=%d pid=%d", qpStatus, getpid());

        ret = NotifyBarrierRecord(&notifyCtx);
        if (ret != 0) {
            goto err;
        }
        ret = NotifyBarrierWait(&notifyCtx);
        if (ret != 0) {
            goto err;
        }
        TEST_LOG_INFO(
            phyid, "WRITE_WITH_NOTIFY PASS: record=%llu payload=%s", (unsigned long long)*notifyCtx.localCell,
            env.heapBuf[1]);
        ret = 0;
    err:
        CaseCleanup(env);
        return ret;
    });
}
