/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* Ported socket-domain cases from the business test repo hccp_test (gstool entries):
 *   hlt_hccp_socket_test_001        -> HltSocketDataExchange
 *   hlt_hccp_uboe_socket_test_001   -> HltUboeSocketDataExchange
 *   hlt_hccp_whitelist_test_001     -> HltWhitelist* (8 variants driven by the python CI)
 *   hlt_hccp_socket_loop_test_001   -> HltSocketLoopServicePath
 *   hlt_hccp_socket_loop_test_002   -> HltSocketLoopRaInitVariant
 * The gstool/cJSON/json/hlt_log wiring is not ported: configuration is inlined and
 * logging goes through TEST_LOG_*. The two-machine server/client model maps onto the
 * RunTests({1,5}) host children (phyId 1 = server/machine_type 0, phyId 5 =
 * client/machine_type 1); the loop entries stay single-process with a self-connect.
 * Port allocation for this batch: 15750-15759 on the SOCKET iface (127.0.1.x/127.0.5.x),
 * the two loop cases reuse 15750 on the H2D (127.0.1.1) and ROCE (127.0.1.2) ifaces. */

#include "cases_common.h"
#include <arpa/inet.h>
#include <vector>

static const int HLT_SERVER_PHYID = 1;            /* machine_type 0 in the original entries */
static const int HLT_CLIENT_PHYID = 5;            /* machine_type 1 in the original entries */
static const unsigned int HLT_EXCHANGE_LEN = 200; /* gstool len parameter used by the python CI */
static const int HLT_ESTABLISH_MS = 3500;         /* positive window: all sockets must establish */
/* Observation budget, NOT a synchronization sleep: the negative assertion is an absence
 * proof (no socket may establish while the whitelist cannot match), and absence has no
 * completion event to poll or signal - the window itself is the assertion. Everything
 * cross-side that does have an event (del RPC landing, phase-1 window closing) is
 * explicitized on the sync side channel instead of slept for. */
static const int HLT_REJECT_MS = 600;
static const int HLT_RECOVER_MS = 2500; /* negative phase 2: delayed whitelist recovery */

/* Loopback form of the VM NIC ip of (phyId, iface), network byte order, ready for
 * HccpIpAddr.addr.s_addr assignment (same conversion as EstablishRendezvous). */
static uint32_t HltLoopbackIp(uint32_t phyId, uint32_t iface)
{
    struct in_addr virt = {};
    virt.s_addr = htonl(HccpVmNicIp(phyId, iface));
    return HccpVmToLoopback(virt).s_addr;
}

/* Mirrors test_socket_data_exchange: fill a len-byte buffer with 0x01, send it, receive
 * len bytes back and verify every received byte is 0x01. Both peers run the same
 * send-then-recv sequence, so each side receives the peer's buffer. */
static int HltSocketExchange(int phyid, int idx, const void* fdHandle, unsigned int len)
{
    std::vector<char> sendBuf(len, '\x01');
    std::vector<char> recvBuf(len, '\0');
    int ret = RaSocketSendAll(fdHandle, sendBuf.data(), len);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "socket[%d] RaSocketSend failed ret=%d", idx, ret);
        return ret;
    }
    ret = RaSocketRecvAll(fdHandle, recvBuf.data(), len);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "socket[%d] RaSocketRecv failed ret=%d", idx, ret);
        return ret;
    }
    for (unsigned int i = 0; i < len; i++) {
        if ((unsigned char)recvBuf[i] != 0x01) {
            TEST_LOG_ERROR(phyid, "socket[%d] data check FAIL at byte %u: 0x%02x", idx, i, (unsigned char)recvBuf[i]);
            return -1;
        }
    }
    TEST_LOG_INFO(phyid, "socket[%d] data exchange PASS (len=%u)", idx, len);
    return 0;
}

/* Mirrors test_socket_data_exchange_loop (single-process dual-end): both endpoints send
 * first, then both receive and verify, so the two directions of one connection pair are
 * checked without a second process. */
static int HltSocketExchangeLoop(int phyid, const void* serverFd, const void* clientFd, unsigned int len)
{
    std::vector<char> serverSend(len, '\x01');
    std::vector<char> clientSend(len, '\x01');
    std::vector<char> serverRecv(len, '\0');
    std::vector<char> clientRecv(len, '\0');
    int ret = RaSocketSendAll(serverFd, serverSend.data(), len);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "server fd RaSocketSend failed ret=%d", ret);
        return ret;
    }
    ret = RaSocketSendAll(clientFd, clientSend.data(), len);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "client fd RaSocketSend failed ret=%d", ret);
        return ret;
    }
    ret = RaSocketRecvAll(serverFd, serverRecv.data(), len);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "server fd RaSocketRecv failed ret=%d", ret);
        return ret;
    }
    ret = RaSocketRecvAll(clientFd, clientRecv.data(), len);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "client fd RaSocketRecv failed ret=%d", ret);
        return ret;
    }
    for (unsigned int i = 0; i < len; i++) {
        if ((unsigned char)serverRecv[i] != 0x01 || (unsigned char)clientRecv[i] != 0x01) {
            TEST_LOG_ERROR(
                phyid, "loop data check FAIL at byte %u: server=0x%02x client=0x%02x", i, (unsigned char)serverRecv[i],
                (unsigned char)clientRecv[i]);
            return -1;
        }
    }
    TEST_LOG_INFO(phyid, "loop data exchange PASS (len=%u)", len);
    return 0;
}

/* Mirrors test_get_socket_handle, generalized to several sockets under one total window:
 * poll RaGetSockets(role) round-robin until every socket reports status==1 with a valid
 * fdHandle and the expected remote ip. Returns 0 when all sockets are established, 1 when
 * the window expired with some socket(s) still not established (the expected outcome for
 * whitelist-rejected connects), negative on a hard mismatch. */
static int HltWaitSockets(
    int phyid, unsigned int role, void* socketHandle, uint32_t peerIp, const char* const tags[], int sockCnt,
    int windowMs, struct SocketInfoT out[])
{
    bool done[2] = {false, false};
    int doneCnt = 0;
    const int spinUs = 5 * 1000;
    for (int spin = 0; doneCnt < sockCnt && spin * spinUs < windowMs * 1000; spin++) {
        for (int i = 0; i < sockCnt; i++) {
            if (done[i]) {
                continue;
            }
            struct SocketInfoT ent = {};
            ent.socketHandle = socketHandle;
            ent.remoteIp.addr.s_addr = peerIp;
            (void)snprintf(ent.tag, sizeof(ent.tag), "%s", tags[i]);
            unsigned int num = 0;
            int ret = RaGetSockets(role, &ent, 1, &num);
            if (ret == 0 && num == 1 && ent.status == 1 && ent.fdHandle != nullptr) {
                if (ent.remoteIp.addr.s_addr != peerIp) {
                    TEST_LOG_ERROR(
                        phyid, "socket[%d] tag=%s remote ip mismatch: got=0x%08x expect=0x%08x", i, tags[i],
                        ent.remoteIp.addr.s_addr, peerIp);
                    return -1;
                }
                out[i] = ent;
                done[i] = true;
                doneCnt++;
                TEST_LOG_INFO(phyid, "socket[%d] tag=%s established (role=%u fd=%p)", i, tags[i], role, ent.fdHandle);
            } else if (ret != 0) {
                /* Not established (yet); keep the evidence line at INFO so an expected
                 * rejection window does not trip the VERDICT error scan. */
                TEST_LOG_INFO(
                    phyid, "socket[%d] tag=%s RaGetSockets ret=%d (not established, role=%u)", i, tags[i], ret, role);
            }
        }
        if (doneCnt < sockCnt) {
            usleep((useconds_t)spinUs);
        }
    }
    if (doneCnt < sockCnt) {
        TEST_LOG_INFO(phyid, "%d/%d sockets established within %dms (role=%u)", doneCnt, sockCnt, windowMs, role);
        return 1;
    }
    return 0;
}

/* Per-host state of one ported case; teardown walks it in the same order as the original
 * test_hccp_socket_resource_deinit: batch close -> listen stop -> socket deinit ->
 * ctx deinit -> RaDeinit -> close net service. Every deinit failure is counted. */
struct HltSockEnv {
    int phyid = 0;
    unsigned short port = 0;
    bool isServer = false;
    bool listening = false;
    bool raInited = false;
    bool netOpened = false;
    struct RaInitConfig cfg = {};
    void* socketHandle = nullptr;
    void* ctxHandle = nullptr;
    void* syncHandle = nullptr; /* ordering-signal side channel (whitelist variants) */
    struct SocketInfoT socks[2] = {};
    int sockCnt = 0;
};

static int HltSockTeardown(HltSockEnv& e, int ret)
{
    for (int i = 0; i < e.sockCnt; i++) {
        if (e.socks[i].fdHandle == nullptr) {
            continue; /* mirrors test_normal_socket_close skipping NULL fds */
        }
        struct SocketCloseInfoT closeInfo = {};
        closeInfo.socketHandle = e.socketHandle;
        closeInfo.fdHandle = e.socks[i].fdHandle;
        closeInfo.disuseLinger = 0;
        int r = RaSocketBatchClose(&closeInfo, 1);
        if (r != 0) {
            TEST_LOG_ERROR(e.phyid, "sock[%d] RaSocketBatchClose failed ret=%d", i, r);
            ret = -1;
        } else {
            e.socks[i].fdHandle = nullptr;
        }
    }
    if (e.listening) {
        struct SocketListenInfoT listenInfo = {};
        listenInfo.socketHandle = e.socketHandle;
        listenInfo.port = e.port;
        int r = RaSocketListenStop(&listenInfo, 1);
        if (r != 0) {
            TEST_LOG_ERROR(e.phyid, "RaSocketListenStop failed ret=%d", r);
            ret = -1;
        }
        e.listening = false;
    }
    if (e.socketHandle != nullptr) {
        int r = RaSocketDeinit(e.socketHandle);
        if (r != 0) {
            TEST_LOG_ERROR(e.phyid, "RaSocketDeinit failed ret=%d", r);
            ret = -1;
        }
        e.socketHandle = nullptr;
    }
    if (e.syncHandle != nullptr) {
        int r = RaSocketDeinit(e.syncHandle);
        if (r != 0) {
            TEST_LOG_ERROR(e.phyid, "sync channel RaSocketDeinit failed ret=%d", r);
            ret = -1;
        }
        e.syncHandle = nullptr;
    }
    if (e.ctxHandle != nullptr) {
        int r = RaCtxDeinit(e.ctxHandle);
        if (r != 0) {
            TEST_LOG_ERROR(e.phyid, "RaCtxDeinit failed ret=%d", r);
            ret = -1;
        }
        e.ctxHandle = nullptr;
    }
    if (e.raInited) {
        int r = RaDeinit(&e.cfg);
        if (r != 0) {
            TEST_LOG_ERROR(e.phyid, "RaDeinit failed ret=%d", r);
            ret = -1;
        }
        e.raInited = false;
    }
    if (e.netOpened) {
        if (rtCloseNetService() != RT_ERROR_NONE) {
            TEST_LOG_ERROR(e.phyid, "rtCloseNetService failed");
            ret = -1;
        }
        e.netOpened = false;
    }
    return ret;
}

/* Offline base (mirrors test_open_hccp_by_tsd + test_ra_init_ub): spawn the per-device
 * service process, then RaInit(NETWORK_OFFLINE) with the CaseOfflineInitNoRdv retry loop. */
static int HltOfflineBase(HltSockEnv& e, int phyid)
{
    e.phyid = phyid;
    e.isServer = (phyid == HLT_SERVER_PHYID);

    char devIdParam[32];
    (void)snprintf(devIdParam, sizeof(devIdParam), "--deviceId=%d", phyid);
    char pidParam[32];
    (void)snprintf(pidParam, sizeof(pidParam), "--pid=%d", (int)getpid());
    rtProcExtParam extParams[2] = {
        {devIdParam, strlen(devIdParam)},
        {pidParam, strlen(pidParam)},
    };
    rtNetServiceOpenArgs openArgs = {extParams, 2};
    if (rtOpenNetService(&openArgs) != RT_ERROR_NONE) {
        TEST_LOG_ERROR(phyid, "rtOpenNetService failed");
        return 1;
    }
    e.netOpened = true;

    e.cfg = {};
    e.cfg.phyId = (unsigned int)phyid;
    e.cfg.nicPosition = NETWORK_OFFLINE;
    e.cfg.hdcType = 18;
    e.cfg.enableHdcAsync = false;
    /* rtOpenNetService returns only after the device stack reports ready
     * (stub waits on deviceReady, 30s budget), so RaInit must succeed on
     * the first call: a failure is a real defect, not a startup race. */
    int ret = RaInit(&e.cfg);
    if (ret != 0) {
        (void)RaDeinit(&e.cfg); /* release any half-initialized RA state */
        TEST_LOG_ERROR(phyid, "RaInit failed ret=%d", ret);
        (void)HltSockTeardown(e, ret);
        return ret;
    }
    e.raInited = true;
    return 0;
}

/* Socket handle + optional ctx on top of the offline base (mirrors test_ra_ctx_init +
 * test_ra_socket_init_v2). The whitelist entries are keyed by the handle local ip, so the
 * loopback form of the VM NIC ip is used, exactly like EstablishRendezvous. */
static int HltOfflineSocketSetup(HltSockEnv& e, int phyid, uint32_t iface, unsigned short port, bool withCtx)
{
    e.port = port;
    int ret = HltOfflineBase(e, phyid);
    if (ret != 0) {
        return ret;
    }
    if (withCtx) {
        struct CtxInitCfg ctxCfg = {};
        struct CtxInitAttr ctxAttr = {};
        ctxCfg.mode = NETWORK_OFFLINE;
        ctxAttr.phyId = (unsigned int)phyid;
        ret = RaCtxInit(&ctxCfg, &ctxAttr, &e.ctxHandle);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaCtxInit failed ret=%d", ret);
            return HltSockTeardown(e, ret);
        }
    }
    struct SocketInitInfoT sockInit = {};
    sockInit.rdevInfo.phyId = (unsigned int)phyid;
    sockInit.rdevInfo.family = AF_INET;
    sockInit.rdevInfo.localIp.addr.s_addr = HltLoopbackIp((uint32_t)phyid, iface);
    sockInit.scopeId = 4;
    ret = RaSocketInitV1(NETWORK_OFFLINE, sockInit, &e.socketHandle);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaSocketInitV1 failed ret=%d", ret);
        return HltSockTeardown(e, ret);
    }
    return 0;
}

/* Server-side listen (mirrors test_ra_socket_listen_start_v2, machine_type 0 only). */
static int HltListenStart(HltSockEnv& e)
{
    struct SocketListenInfoT listenInfo = {};
    listenInfo.socketHandle = e.socketHandle;
    listenInfo.port = e.port;
    int ret = RaSocketListenStart(&listenInfo, 1);
    if (ret != 0) {
        TEST_LOG_ERROR(e.phyid, "RaSocketListenStart failed ret=%d (port=%u)", ret, e.port);
        return ret;
    }
    e.listening = true;
    return 0;
}

/* Client-side batch connect (mirrors the 16-per-batch loop of test_normal_socket_connect). */
static int HltBatchConnect(HltSockEnv& e, uint32_t serverIp, const char* const tags[], int sockCnt)
{
    for (int base = 0; base < sockCnt; base += 16) {
        int num = (sockCnt - base < 16) ? (sockCnt - base) : 16;
        struct SocketConnectInfoT conn[16] = {};
        for (int i = 0; i < num; i++) {
            conn[i].socketHandle = e.socketHandle;
            conn[i].remoteIp.addr.s_addr = serverIp;
            conn[i].port = e.port;
            (void)snprintf(conn[i].tag, sizeof(conn[i].tag), "%s", tags[base + i]);
        }
        int ret = RaSocketBatchConnect(conn, (unsigned int)num);
        if (ret != 0) {
            TEST_LOG_ERROR(e.phyid, "RaSocketBatchConnect failed ret=%d (num=%d)", ret, num);
            return ret;
        }
    }
    return 0;
}

/* Common runner for hlt_hccp_socket_test_001 / hlt_hccp_uboe_socket_test_001: full
 * resource chain (offline base [+ ctx] + socket init), server listens and whitelists one
 * entry per socket (tag = decimal index, limit 1, mirroring the json defaults), client
 * batch connects, both sides wait for every socket to establish with the peer ip checked,
 * then run the 0x01-pattern exchange on each socket. */
static int RunHltSocketExchange(int phyid, unsigned short port, bool withCtx, int sockCnt)
{
    port = HccpStCasePort(port); /* direct-bind family: displace at entry (listener+connectors) */
    HltSockEnv e;
    const char* tags[2] = {"0", "1"};
    const uint32_t peerIp = HltLoopbackIp(
        (uint32_t)(phyid == HLT_SERVER_PHYID ? HLT_CLIENT_PHYID : HLT_SERVER_PHYID), HCCP_VM_IFACE_SOCKET);
    int ret = HltOfflineSocketSetup(e, phyid, HCCP_VM_IFACE_SOCKET, port, withCtx);
    if (ret != 0) {
        return ret;
    }

    if (e.isServer) {
        ret = HltListenStart(e);
        if (ret != 0) {
            return HltSockTeardown(e, ret);
        }
        struct SocketWlistInfoT wlist[2] = {};
        for (int i = 0; i < sockCnt; i++) {
            wlist[i].remoteIp.addr.s_addr = peerIp;
            wlist[i].connLimit = 1;
            (void)snprintf(wlist[i].tag, sizeof(wlist[i].tag), "%s", tags[i]);
        }
        ret = RaSocketWhiteListAdd(e.socketHandle, wlist, (unsigned int)sockCnt);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaSocketWhiteListAdd failed ret=%d", ret);
            return HltSockTeardown(e, ret);
        }
    } else {
        ret = HltBatchConnect(e, peerIp, tags, sockCnt);
        if (ret != 0) {
            return HltSockTeardown(e, ret);
        }
    }

    e.sockCnt = sockCnt;
    ret = HltWaitSockets(phyid, e.isServer ? 0 : 1, e.socketHandle, peerIp, tags, sockCnt, HLT_ESTABLISH_MS, e.socks);
    if (ret != 0) {
        return HltSockTeardown(e, -1);
    }
    for (int i = 0; i < sockCnt; i++) {
        ret = HltSocketExchange(phyid, i, e.socks[i].fdHandle, HLT_EXCHANGE_LEN);
        if (ret != 0) {
            return HltSockTeardown(e, ret);
        }
    }
    return HltSockTeardown(e, 0);
}

/* One whitelist variant of hlt_hccp_whitelist_test_001. The python CI drove the entry with
 * whitelist_type 0..6 plus limit 0/1/2; positive variants only assert the value=0 path,
 * negative variants asserted a specific product error line in the plog. The gtest port
 * asserts the same behavior directly:
 *   - positive: the whitelisted connect establishes and exchanges data;
 *   - negative: no socket establishes within the window while the whitelist cannot match
 *     (missing / deleted / mismatched tag / connLimit 0), and afterwards the server
 *     adds the correct entry late: the client's connect retry must then recover (this is
 *     the CI's delayed-whitelist variant, whiteList_test_019) and exchange data, proving
 *     the phase-1 absence was really caused by the whitelist.
 * A mismatched tag is always a mismatch; a mismatched ip alone is only a mismatch on
 * products where the whitelist enforces client-ip equality (see the ip-mismatch variant
 * for the product-conditional semantics). */
struct HltWlVariant {
    const char* name;
    int phase1Adds;          /* whitelist adds executed in phase 1 (0 = none, 2 = repeat add) */
    bool phase1CorrectEntry; /* phase-1 entries carry the correct client ip+tag */
    bool phase1ZeroIp;       /* phase-1 entries carry ip 0.0.0.0 (original: remoteIp = {}) */
    bool phase1WrongTag;     /* phase-1 entries carry tag "white_list_test" */
    unsigned int limit;      /* connLimit of the phase-1 entries */
    bool phase1Del;          /* delete the phase-1 entries again (original type 3) */
    bool negative;           /* phase 1 must reject, phase 2 must recover */
    int clientDelayMs;       /* client-side connect delay: >0 only for the add-del variant */
    int serverPhase2DelayMs; /* server delay between the phase-1 window and the phase-2 re-add */
};

static int RunHltWhitelistVariant(const HltWlVariant& v, int phyid, unsigned short port)
{
    /* Ordering-signal side channel port (raw, pre-displacement): +100 keeps it clear of
     * every case port while staying inside the per-run displaced domain. Only the
     * variants that need cross-side ordering establish it. */
    const unsigned short syncPort = (unsigned short)(port + 100);
    port = HccpStCasePort(port); /* direct-bind family: displace at entry (listener+connectors) */
    HltSockEnv e;
    const char* tags[1] = {"0"};
    const uint32_t peerIp = HltLoopbackIp(
        (uint32_t)(phyid == HLT_SERVER_PHYID ? HLT_CLIENT_PHYID : HLT_SERVER_PHYID), HCCP_VM_IFACE_SOCKET);
    int ret = HltOfflineSocketSetup(e, phyid, HCCP_VM_IFACE_SOCKET, port, true);
    if (ret != 0) {
        return ret;
    }

    /* Side channel for the two ordering guards: the whitelist socket under test is the
     * thing being verified, so the ordering signals (server's del-done, client's
     * phase-1-done) travel on a separate plain rendezvous instead of fixed sleeps. */
    struct SocketInfoT syncSock = {};
    if (v.clientDelayMs > 0 || v.serverPhase2DelayMs > 0) {
        struct SocketInitInfoT syncInit = {};
        syncInit.rdevInfo.phyId = (unsigned int)phyid;
        syncInit.rdevInfo.family = AF_INET;
        syncInit.rdevInfo.localIp.addr.s_addr = htonl(HccpVmNicIp((uint32_t)phyid, HCCP_VM_IFACE_SOCKET));
        syncInit.scopeId = 4;
        ret = EstablishRendezvous(
            phyid, NETWORK_OFFLINE, syncInit, syncPort, "WL SYNC", true, &e.syncHandle, &syncSock);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "variant %s: sync rendezvous failed ret=%d", v.name, ret);
            return HltSockTeardown(e, ret);
        }
    }

    if (e.isServer) {
        ret = HltListenStart(e);
        if (ret != 0) {
            return HltSockTeardown(e, ret);
        }
        for (int add = 0; add < v.phase1Adds; add++) {
            struct SocketWlistInfoT wlist[1] = {};
            wlist[0].remoteIp.addr.s_addr = v.phase1ZeroIp ? 0 : peerIp;
            wlist[0].connLimit = v.limit;
            (void)snprintf(wlist[0].tag, sizeof(wlist[0].tag), "%s", v.phase1WrongTag ? "white_list_test" : tags[0]);
            ret = RaSocketWhiteListAdd(e.socketHandle, wlist, 1);
            if (ret != 0) {
                TEST_LOG_ERROR(phyid, "variant %s: RaSocketWhiteListAdd #%d failed ret=%d", v.name, add, ret);
                return HltSockTeardown(e, ret);
            }
        }
        if (v.phase1Del) {
            struct SocketWlistInfoT wlist[1] = {};
            wlist[0].remoteIp.addr.s_addr = peerIp;
            wlist[0].connLimit = v.limit;
            (void)snprintf(wlist[0].tag, sizeof(wlist[0].tag), "%s", tags[0]);
            ret = RaSocketWhiteListDel(e.socketHandle, wlist, 1);
            if (ret != 0) {
                TEST_LOG_ERROR(phyid, "variant %s: RaSocketWhiteListDel failed ret=%d", v.name, ret);
                return HltSockTeardown(e, ret);
            }
            if (v.clientDelayMs > 0) {
                /* Signal the del RPC completion: the client waits for this instead of a
                 * fixed connect delay, so the connect lands right after the del. */
                uint8_t delDone = 1;
                ret = RaSocketSendAll(syncSock.fdHandle, &delDone, sizeof(delDone));
                if (ret != 0) {
                    TEST_LOG_ERROR(phyid, "variant %s: del-done signal failed ret=%d", v.name, ret);
                    return HltSockTeardown(e, ret);
                }
            }
        }
    } else {
        /* Ordering guard for the add-del variant (race R1): the original
         * whiteList_test_015 runs the server config (add + del) strictly before the
         * client config (connect), so the connect can never hit the rs while the deleted
         * entry is still live. The ported case forks both hosts concurrently, and the
         * client's tag-sync validation can win the race against the server's
         * WhiteListDel RPC (observed as a 1-in-6+ suite flake). The client therefore
         * waits for the server's del-done signal on the side channel (the old form was a
         * fixed 350ms connect delay) while staying well ahead of the server's delayed
         * phase-2 re-add (race R2 - see the phase-2 block). The other variants need no
         * delay (no-add never has an entry, mismatch entries can never match). */
        if (v.clientDelayMs > 0) {
            uint8_t delDone = 0;
            ret = RaSocketRecvAll(syncSock.fdHandle, &delDone, sizeof(delDone));
            if (ret != 0 || delDone != 1) {
                TEST_LOG_ERROR(
                    phyid, "variant %s: waiting for the server's del-done signal failed ret=%d", v.name, ret);
                return HltSockTeardown(e, ret);
            }
            TEST_LOG_INFO(phyid, "variant %s: server's whitelist del landed, connecting", v.name);
        }
        ret = HltBatchConnect(e, peerIp, tags, 1);
        if (ret != 0) {
            return HltSockTeardown(e, ret);
        }
    }
    e.sockCnt = 1;

    if (v.negative) {
        /* Phase 1: the connect attempt must stay rejected while the whitelist cannot
         * match. Both sides must observe "not established". */
        ret = HltWaitSockets(phyid, e.isServer ? 0 : 1, e.socketHandle, peerIp, tags, 1, HLT_REJECT_MS, e.socks);
        if (ret == 0) {
            TEST_LOG_ERROR(phyid, "variant %s: connection ESTABLISHED although the whitelist cannot match", v.name);
            return HltSockTeardown(e, -1);
        }
        if (ret < 0) {
            return HltSockTeardown(e, ret);
        }
        TEST_LOG_INFO(phyid, "variant %s: connect rejected as expected in phase 1", v.name);

        /* The client's reject window has closed: tell the server so its phase-2 re-add
         * can land immediately instead of after a fixed margin. */
        if (!e.isServer && v.serverPhase2DelayMs > 0) {
            uint8_t phase1Done = 1;
            ret = RaSocketSendAll(syncSock.fdHandle, &phase1Done, sizeof(phase1Done));
            if (ret != 0) {
                TEST_LOG_ERROR(phyid, "variant %s: phase-1-done signal failed ret=%d", v.name, ret);
                return HltSockTeardown(e, ret);
            }
        }

        /* Phase 2 (delayed whitelist, mirrors whiteList_test_019): the server adds the
         * correct entry late; the client's retry must establish and exchange data. The
         * add must not land while the peer is still inside its phase-1 reject window:
         * the connect retry validates ~instantly once the entry exists, so an early
         * phase-2 add makes the peer observe ESTABLISHED inside its own phase 1 (the
         * mirror race of the clientDelayMs one; both stem from the original running the
         * server and client configs sequentially). The server therefore waits for the
         * peer's phase-1-done signal on the side channel (the old form was a fixed
         * 800ms margin over the 600ms reject window). */
        if (e.isServer) {
            if (v.serverPhase2DelayMs > 0) {
                uint8_t phase1Done = 0;
                ret = RaSocketRecvAll(syncSock.fdHandle, &phase1Done, sizeof(phase1Done));
                if (ret != 0 || phase1Done != 1) {
                    TEST_LOG_ERROR(
                        phyid, "variant %s: waiting for the peer's phase-1-done signal failed ret=%d", v.name, ret);
                    return HltSockTeardown(e, ret);
                }
                TEST_LOG_INFO(phyid, "variant %s: peer's phase-1 window closed, re-adding", v.name);
            }
            struct SocketWlistInfoT wlist[1] = {};
            wlist[0].remoteIp.addr.s_addr = peerIp;
            wlist[0].connLimit = 1;
            (void)snprintf(wlist[0].tag, sizeof(wlist[0].tag), "%s", tags[0]);
            ret = RaSocketWhiteListAdd(e.socketHandle, wlist, 1);
            if (ret != 0) {
                TEST_LOG_ERROR(phyid, "variant %s: phase-2 RaSocketWhiteListAdd failed ret=%d", v.name, ret);
                return HltSockTeardown(e, ret);
            }
        }
        ret = HltWaitSockets(phyid, e.isServer ? 0 : 1, e.socketHandle, peerIp, tags, 1, HLT_RECOVER_MS, e.socks);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "variant %s: no recovery after the delayed whitelist add", v.name);
            return HltSockTeardown(e, -1);
        }
    } else {
        ret = HltWaitSockets(phyid, e.isServer ? 0 : 1, e.socketHandle, peerIp, tags, 1, HLT_ESTABLISH_MS, e.socks);
        if (ret != 0) {
            return HltSockTeardown(e, -1);
        }
    }

    ret = HltSocketExchange(phyid, 0, e.socks[0].fdHandle, HLT_EXCHANGE_LEN);
    if (ret != 0) {
        return HltSockTeardown(e, ret);
    }
    return HltSockTeardown(e, 0);
}

/* Common runner for the loop entries. The original runs one process with two configs
 * (server + client) sharing a single socket handle: listen on the own ip, whitelist the
 * own ip, batch connect to itself, then query both roles (server 0 / client 1) and run
 * the dual-end exchange on the two fds.
 *   - service path (001): TsdProcessOpen + RaInit offline + ctx  -> rtOpenNetService +
 *     RaInit(NETWORK_OFFLINE) + RaCtxInit, H2D iface loopback;
 *   - RaInit variant (002): RaInit directly, no service process, no ctx -> the sim's
 *     host-direct path RaInit(NETWORK_PEER_ONLINE) (no rtOpenNetService exists to start a
 *     service first), ROCE iface loopback, mirroring the CasePeerInit shape. */
static int RunHltSocketLoop(int phyid, unsigned short port, bool servicePath)
{
    port = HccpStCasePort(port); /* direct-bind family: displace at entry (listener+connectors) */
    HltSockEnv e;
    const char* tags[1] = {"0"};
    const uint32_t iface = servicePath ? HCCP_VM_IFACE_H2D : HCCP_VM_IFACE_ROCE;
    const uint32_t selfIp = HltLoopbackIp((uint32_t)phyid, iface);
    e.phyid = phyid;
    e.port = port;

    int ret;
    if (servicePath) {
        ret = HltOfflineSocketSetup(e, phyid, iface, port, true);
        if (ret != 0) {
            return ret;
        }
    } else {
        e.cfg = {};
        e.cfg.phyId = (unsigned int)phyid;
        e.cfg.nicPosition = NETWORK_PEER_ONLINE;
        e.cfg.hdcType = 18;
        e.cfg.enableHdcAsync = false;
        ret = RaInit(&e.cfg);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaInit(peer) failed ret=%d", ret);
            return ret;
        }
        e.raInited = true;
        struct SocketInitInfoT sockInit = {};
        sockInit.rdevInfo.phyId = (unsigned int)phyid;
        sockInit.rdevInfo.family = AF_INET;
        sockInit.rdevInfo.localIp.addr.s_addr = selfIp;
        sockInit.scopeId = 4;
        ret = RaSocketInitV1(NETWORK_PEER_ONLINE, sockInit, &e.socketHandle);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaSocketInitV1(peer) failed ret=%d", ret);
            return HltSockTeardown(e, ret);
        }
    }

    ret = HltListenStart(e);
    if (ret != 0) {
        return HltSockTeardown(e, ret);
    }
    struct SocketWlistInfoT wlist[1] = {};
    wlist[0].remoteIp.addr.s_addr = selfIp; /* self-connect: the peer is the own ip */
    wlist[0].connLimit = 1;
    (void)snprintf(wlist[0].tag, sizeof(wlist[0].tag), "%s", tags[0]);
    ret = RaSocketWhiteListAdd(e.socketHandle, wlist, 1);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaSocketWhiteListAdd failed ret=%d", ret);
        return HltSockTeardown(e, ret);
    }

    ret = HltBatchConnect(e, selfIp, tags, 1);
    if (ret != 0) {
        return HltSockTeardown(e, ret);
    }

    /* Client-role and server-role established sockets of the self-connect; each fd is
     * registered in the teardown state as soon as it is obtained. */
    struct SocketInfoT cliSock = {};
    struct SocketInfoT srvSock = {};
    ret = HltWaitSockets(phyid, 1, e.socketHandle, selfIp, tags, 1, HLT_ESTABLISH_MS, &cliSock);
    if (ret != 0) {
        return HltSockTeardown(e, -1);
    }
    e.socks[0] = cliSock;
    e.sockCnt = 1;
    ret = HltWaitSockets(phyid, 0, e.socketHandle, selfIp, tags, 1, HLT_ESTABLISH_MS, &srvSock);
    if (ret != 0) {
        return HltSockTeardown(e, -1);
    }
    e.socks[1] = srvSock;
    e.sockCnt = 2;

    ret = HltSocketExchangeLoop(phyid, srvSock.fdHandle, cliSock.fdHandle, HLT_EXCHANGE_LEN);
    if (ret != 0) {
        return HltSockTeardown(e, ret);
    }
    return HltSockTeardown(e, 0);
}

/* Ported from hlt_hccp_socket_test_001: two-node RaSocket full chain (offline service +
 * RaCtxInit + socket init, server listen + per-socket whitelist, client batch connect,
 * RaGetSockets convergence with remote-ip verification) and a 0x01-pattern data exchange
 * on every socket. Two sockets exercise the per-socket whitelist/tag/connect loop of the
 * original (socket_num from json, tags "0"/"1"). */
TEST_F(HccpSystemTest, HltSocketDataExchange)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        return RunHltSocketExchange(phyid, 15750, true, 2);
    });
}

/* Ported from hlt_hccp_uboe_socket_test_001: identical chain minus RaCtxInit (the uboe
 * variant drives the socket plane without a ctx), two sockets, port 15751. */
TEST_F(HccpSystemTest, HltUboeSocketDataExchange)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        return RunHltSocketExchange(phyid, 15751, false, 2);
    });
}

/* Ported from hlt_hccp_whitelist_test_001, whitelist_type=0 (whiteList_test_002, single
 * whitelist add): the server adds the correct client entry once; the connect must
 * establish and exchange data. */
TEST_F(HccpSystemTest, HltWhitelistSingleAdd)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        const HltWlVariant v = {"single-add", 1, true, false, false, 1, false, false, 0, 0};
        return RunHltWhitelistVariant(v, phyid, 15752);
    });
}

/* Ported from hlt_hccp_whitelist_test_001, whitelist_type=1 (whiteList_test_004, repeat
 * add): the same correct entry is added twice; the connect must still establish (the
 * product accumulates connLimit on an existing node) and exchange data. */
TEST_F(HccpSystemTest, HltWhitelistRepeatAdd)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        const HltWlVariant v = {"repeat-add", 2, true, false, false, 1, false, false, 0, 0};
        return RunHltWhitelistVariant(v, phyid, 15753);
    });
}

/* Ported from hlt_hccp_whitelist_test_001, whitelist_type=2 (whiteList_test_006, no
 * whitelist): the server never adds an entry; the connect must stay rejected, then
 * recover after the delayed correct add. */
TEST_F(HccpSystemTest, HltWhitelistNoAddRejected)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        const HltWlVariant v = {"no-add", 0, false, false, false, 1, false, true, 0, 800};
        return RunHltWhitelistVariant(v, phyid, 15754);
    });
}

/* Ported from hlt_hccp_whitelist_test_001, whitelist_type=3 (whiteList_test_015,
 * whitelist delete): the correct entry is added and deleted again; the connect must stay
 * rejected, then recover after the delayed re-add. */
TEST_F(HccpSystemTest, HltWhitelistAddDelRejected)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        const HltWlVariant v = {"add-del", 1, true, false, false, 1, true, true, 350, 800};
        return RunHltWhitelistVariant(v, phyid, 15755);
    });
}

/* Ported from hlt_hccp_whitelist_test_001, whitelist_type=4 (whiteList_test_014, ip and
 * tag both inconsistent): the entry carries ip 0.0.0.0 (original: remoteIp = {}) and a
 * foreign tag, so nothing can match; rejected, then delayed-correct-add recovery. */
TEST_F(HccpSystemTest, HltWhitelistIpTagMismatchRejected)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        const HltWlVariant v = {"ip-tag-mismatch", 1, false, true, true, 1, false, true, 0, 800};
        return RunHltWhitelistVariant(v, phyid, 15756);
    });
}

/* Ported from hlt_hccp_whitelist_test_001, whitelist_type=5 (whiteList_test_010, tag
 * inconsistent, ip consistent): the entry carries the correct client ip but a foreign
 * tag; rejected, then delayed-correct-add recovery. */
TEST_F(HccpSystemTest, HltWhitelistTagMismatchRejected)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        const HltWlVariant v = {"tag-mismatch", 1, false, false, true, 1, false, true, 0, 800};
        return RunHltWhitelistVariant(v, phyid, 15757);
    });
}

/* Ported from hlt_hccp_whitelist_test_001, whitelist_type=6 (whiteList_test_012, ip
 * inconsistent, tag consistent): the entry carries the correct tag but ip 0.0.0.0.
 * On the 910-family DUT of the original hardware CI this is a rejection case: the
 * whitelist used to compare the client ip strictly. Since the rs change of
 * cann/hcomm!5895, RsFindWhiteListNode enforces client-ip equality only when
 * RsIsClientIpCheckSupported() holds (310p/910/910B/910_93); on 950/960/350 the
 * whitelist matches by tag alone. The ST stub (ascend_hal_common.c halGetChipInfo)
 * simulates a 950DT chip, so this variant asserts the tag-only semantics: the
 * mismatched-ip entry with the correct tag and connLimit still admits the connect,
 * which then exchanges data. No phase-2 recovery applies (nothing is rejected). If
 * the stub ever simulates a chip with client-ip check support, flip this variant
 * back to negative=true with the delayed phase-2 re-add of the original. */
TEST_F(HccpSystemTest, HltWhitelistIpMismatchRejected)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        const HltWlVariant v = {"ip-mismatch", 1, false, true, false, 1, false, false, 0, 0};
        return RunHltWhitelistVariant(v, phyid, 15758);
    });
}

/* Ported from hlt_hccp_whitelist_test_001, whitelist_type=0 with limit=0
 * (whiteList_test_016, connection limit 0): the entry matches ip+tag but carries
 * connLimit 0, which the server rejects (connLimit < 1); recovery via a delayed add with
 * connLimit 1, which the product accumulates onto the existing node (0 + 1). */
TEST_F(HccpSystemTest, HltWhitelistLimitZeroRejected)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        const HltWlVariant v = {"limit-zero", 1, true, false, false, 0, false, true, 0, 800};
        return RunHltWhitelistVariant(v, phyid, 15759);
    });
}

/* Ported from hlt_hccp_socket_loop_test_001: single-process loopback chain on one socket
 * handle (service process + RaInit offline + RaCtxInit + listen + self-whitelist +
 * self-connect + both-role RaGetSockets + dual-end exchange), H2D iface loopback. */
TEST_F(HccpSystemTest, HltSocketLoopServicePath)
{
    RunTests({1}, [this](int phyid) -> int {
        return RunHltSocketLoop(phyid, 15750, true);
    });
}

/* Ported from hlt_hccp_socket_loop_test_002 (RaInit variant): no service process and no
 * ctx, RaInit alone drives the chain; maps onto the sim's host-direct peer path
 * (NETWORK_PEER_ONLINE), ROCE iface loopback. */
TEST_F(HccpSystemTest, HltSocketLoopRaInitVariant)
{
    RunTests({1}, [this](int phyid) -> int {
        return RunHltSocketLoop(phyid, 15750, false);
    });
}
