/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* Ported pingmesh case from the business test repo hccp_test (gstool entry):
 *   hlt_hccp_pingmesh_test_001 -> HltPingmeshBasic
 *
 * The original entry (hlt_hccp_ub_pingmesh_test.c + hlt_hccp_ub_pingmesh_common.c) is a
 * two-machine symmetric flow: each side opens the device service, runs a full RaSocket
 * sync-socket setup (listen + whitelist + connect to the peer), resolves its local eid
 * through RaGetDevEidInfoNum/List, calls RaPingInit, exchanges the ping comm info
 * (eid + client qpInfo) with the peer over the sync socket, then drives one ping round:
 * TargetAdd -> TaskStart -> GetResults (ROCE_EAGAIN retry) -> "sync" handshake ->
 * TaskStop -> TargetDel. The gstool/cJSON/json/hlt_log wiring is not ported:
 * configuration is inlined and logging goes through TEST_LOG_*.
 *
 * PoC notes (sim adaptations, all evidenced in the product code):
 * - The original starts the device service with "--whiteListStatus=0" and then connects
 *   with tags derived from the ip decimal strings while the whitelist entry carries the
 *   fixed tag "sync" (ub_pingmesh_common.c). In this product checkout that service param
 *   is dead state (parsed into rscb->connCb.wlistEnable in rs.c:242 but never read), so
 *   the port cannot rely on a disabled check. The whitelist must match the real conns
 *   instead: the peer's incoming conn carries the decimal tag of MY local ip (the peer
 *   derives its connect tag from its target list), so each side adds the entry
 *   {remoteIp=peer, tag=dec(local ip), connLimit=1}. Note the per-listener tagChkDis bit
 *   cannot be used either: RsServerSendWlistCheckResult (the "a5a5a" reply that moves
 *   the client conn out of CONNECTING) only runs on the validation path that tagChkDis
 *   skips, so a tagChkDis listener accepts the conn but leaves the peer's client socket
 *   permanently CONNECTING (verified in the first PoC run).
 * - Verification is strengthened per the t1 survey section 2.6: the original only
 *   printed the result summary; this port asserts result.state == PING_RESULT_STATE_VALID
 *   and recvCnt > 0 so the ping data plane is actually proven, and verifies the
 *   "sync" handshake content (section 6.3: every data transfer gets a content check).
 * - Constants mirror the original json defaults: packet_cnt=50, packet_interval=10,
 *   timeout_interval=100, payload_policy=0 (fixed size), size=100, bufferSize 100*1*2048,
 *   qos tc=(33&0x3f)<<2 sl=4, cq depth 128, qp caps {inline 32, sge 1, wr 128}.
 * - Port allocation for this batch: 15760 on the ROCE iface loopback (127.1.2/127.5.2).
 *
 * HltPingmeshMesh (below) ports the mesh-topology semantics of the same business entry:
 * the original is driven per node by ping_mesh.json (local_ip + target_ip[] of up to 64
 * entries; running it on every node of the cluster forms the full interconnect). The
 * json is inlined as the phyId list of RunTests: meshN nodes (phyId 0..meshN-1, the
 * d2dFullMesh32 form), each node's target list = all other nodes. Sync sockets, comm
 * info exchange, TargetAdd/GetResults/TargetDel run per pair in batch form.
 *   Sim degradations (documented per the t7 directive):
 *   - node count capped at 8 (default 4): the 5s host-child budget bounds the task time,
 *     which grows as packet_cnt * (meshN-1) * packet_interval (the rs ping thread walks
 *     every target once per packet, rs_ping.c RsPingHandle);
 *   - for meshN > 4 packet_cnt is scaled down from the json default 50 so the task stays
 *     inside ~2s (ping_mesh.json itself allows packet_cnt to be tuned per deployment);
 *   - per-target result verification keeps the strengthened t1 survey assertions
 *     (state == PING_RESULT_STATE_VALID, sendCnt > 0, recvCnt > 0 for every pair).
 *   The payload byte content cannot be verified end-to-end: the product result API only
 *   returns counters/rtt per target (hccp_ping.h PingResultInfo), so the comm-info
 *     exchange and the "sync" handshake carry the content checks of this port. */

#include "cases_common.h"
#include "hccp_ping.h"
#include <arpa/inet.h>

static const unsigned short PING_MESH_PORT = HccpStCasePort(15760); /* original listen_port 16667, remapped */
static const unsigned int PING_PACKET_CNT = 50;                     /* DEFAULT_PING_PACKET_CNT */
static const unsigned int PING_PACKET_INTERVAL = 10;                /* DEFAULT_PING_PACKET_INTERVAL */
static const unsigned int PING_TIMEOUT_INTERVAL = 100;              /* DEFAULT_PING_TIMEOUT_INTERVAL */
static const unsigned int PING_PAYLOAD_SIZE = 100;                  /* DEFAULT_PING_SIZE */
static const unsigned int PING_BUFFER_SIZE = 100 * 1 * 2048;        /* test_pingmesh_init_attr_default_init */
static const int PING_ESTABLISH_MS = 3500;                          /* sync socket establishment window */
static const int PING_RESULTS_MS = 4000;                            /* GetResults window (fits the 5s host budget) */

/* HltPingmeshMesh: inlined ping_mesh.json topology. Port per the t7 directive; node cap
 * and packet_cnt scaling are the documented sim degradations (see the file header). */
static const unsigned short PINGMESH_MESH_PORT = HccpStCasePort(15900); /* sync socket port of the mesh case */
static const int PINGMESH_MESH_DEFAULT_N = 4;                           /* default full-interconnect node count */
static const int PINGMESH_MESH_MAX_N = 8;                               /* hard cap: 5s host budget */
static const int PINGMESH_TASK_BUDGET_MS = 2000;                        /* keep packet_cnt*(N-1)*10ms inside this */

/* Loopback form of the VM NIC ip of (phyId, iface), network byte order (same conversion
 * as EstablishRendezvous). */
static uint32_t PingMeshLoopbackIp(uint32_t phyId, uint32_t iface)
{
    struct in_addr virt = {};
    virt.s_addr = htonl(HccpVmNicIp(phyId, iface));
    return HccpVmToLoopback(virt).s_addr;
}

/* Decimal-string tag of an ip value, mirroring the original
 * sprintf_s(tag, "%u", (unsigned int)ip.addr.s_addr). The exact value does not matter,
 * only that the connecting side and the accepting side derive the same string for the
 * same conn: client conns carry the decimal of the remote ip, server-side queries use
 * the decimal of the local ip. */
static void PingMeshIpTag(uint32_t ipNetOrder, char* tag, size_t tagLen)
{
    (void)snprintf(tag, tagLen, "%u", (unsigned int)ipNetOrder);
}

/* Per-host state of the pingmesh case; teardown walks it in the order of the original
 * test_pingmesh_hccp_resource_deinit: RaPingDeinit -> sync socket close -> listen stop ->
 * socket deinit -> RaDeinit -> close net service. Every deinit failure is counted. */
struct PingMeshEnv {
    int phyid = 0;
    bool raInited = false;
    bool netOpened = false;
    struct RaInitConfig cfg = {};
    void* socketHandle = nullptr;
    bool listening = false;
    struct SocketInfoT serverSock = {}; /* fd of the conn the peer initiated (role 0) */
    struct SocketInfoT clientSock = {}; /* fd of the conn this host initiated (role 1) */
    void* pingHandle = nullptr;
};

static int PingMeshTeardown(PingMeshEnv& e, int ret)
{
    if (e.pingHandle != nullptr) {
        int r = RaPingDeinit(e.pingHandle);
        if (r != 0) {
            TEST_LOG_ERROR(e.phyid, "RaPingDeinit failed ret=%d", r);
            ret = -1;
        }
        e.pingHandle = nullptr;
    }
    struct SocketCloseInfoT closeInfo = {};
    closeInfo.socketHandle = e.socketHandle;
    closeInfo.disuseLinger = 0;
    if (e.serverSock.fdHandle != nullptr) {
        closeInfo.fdHandle = e.serverSock.fdHandle;
        int r = RaSocketBatchClose(&closeInfo, 1);
        if (r != 0) {
            TEST_LOG_ERROR(e.phyid, "server fd RaSocketBatchClose failed ret=%d", r);
            ret = -1;
        }
        e.serverSock.fdHandle = nullptr;
    }
    if (e.clientSock.fdHandle != nullptr) {
        closeInfo.fdHandle = e.clientSock.fdHandle;
        int r = RaSocketBatchClose(&closeInfo, 1);
        if (r != 0) {
            TEST_LOG_ERROR(e.phyid, "client fd RaSocketBatchClose failed ret=%d", r);
            ret = -1;
        }
        e.clientSock.fdHandle = nullptr;
    }
    if (e.listening) {
        struct SocketListenInfoT listenInfo = {};
        listenInfo.socketHandle = e.socketHandle;
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

/* Mirrors test_pingmesh_open_hccp_by_tsd + test_pingmesh_RaInit: spawn the per-device
 * service process, then RaInit(NETWORK_OFFLINE) with the standard retry loop. Shared by
 * the basic and mesh envs; the env-specific teardown handles the RaInit failure path. */
template <typename EnvT>
static int PingMeshOfflineBaseT(EnvT& e, int phyid, int (*teardown)(EnvT&, int))
{
    e.phyid = phyid;
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
        return teardown(e, ret);
    }
    e.raInited = true;
    return 0;
}

static int PingMeshOfflineBase(PingMeshEnv& e, int phyid) { return PingMeshOfflineBaseT(e, phyid, PingMeshTeardown); }

/* Mirrors test_pingmesh_socket_resource_init: socket init on the ROCE iface loopback,
 * listen, the whitelist entry matching the peer's incoming conn (see the file header),
 * and the connect towards the peer. Returns after the connect request is posted;
 * establishment is polled by PingMeshWaitSyncSockets. */
static int PingMeshSyncSocketSetup(PingMeshEnv& e, int phyid, uint32_t localIp, uint32_t peerIp)
{
    struct SocketInitInfoT socketInit = {};
    socketInit.rdevInfo.phyId = (unsigned int)phyid;
    socketInit.rdevInfo.family = AF_INET;
    socketInit.rdevInfo.localIp.addr.s_addr = localIp;
    socketInit.scopeId = 4;
    int ret = RaSocketInitV1(NETWORK_OFFLINE, socketInit, &e.socketHandle);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaSocketInitV1 failed ret=%d", ret);
        return ret;
    }

    struct SocketListenInfoT listenInfo = {};
    listenInfo.socketHandle = e.socketHandle;
    listenInfo.port = PING_MESH_PORT;
    ret = RaSocketListenStart(&listenInfo, 1);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaSocketListenStart failed ret=%d", ret);
        return ret;
    }
    e.listening = true;

    /* The peer connects to me with the decimal tag of MY local ip (it derives the tag
     * from its target list), so the entry that validates the peer's conn carries that
     * tag. Replaces the original's dead "--whiteListStatus=0" disabled check. */
    struct SocketWlistInfoT wlist = {};
    wlist.remoteIp.addr.s_addr = peerIp;
    wlist.connLimit = 1;
    PingMeshIpTag(localIp, wlist.tag, sizeof(wlist.tag));
    ret = RaSocketWhiteListAdd(e.socketHandle, &wlist, 1);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaSocketWhiteListAdd failed ret=%d", ret);
        return ret;
    }

    char clientTag[SOCK_CONN_TAG_SIZE] = {0};
    PingMeshIpTag(peerIp, clientTag, sizeof(clientTag));
    struct SocketConnectInfoT conn = {};
    conn.socketHandle = e.socketHandle;
    conn.remoteIp.addr.s_addr = peerIp;
    conn.port = PING_MESH_PORT;
    (void)snprintf(conn.tag, sizeof(conn.tag), "%s", clientTag);
    ret = RaSocketBatchConnect(&conn, 1);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaSocketBatchConnect failed ret=%d", ret);
        return ret;
    }
    return 0;
}

/* Mirrors test_pingmesh_sync_socket_connect: poll for the server-role fd (the conn the
 * peer initiated, queried with the decimal tag of the local ip) and the client-role fd
 * (the conn this host initiated, queried with the decimal tag of the peer ip). */
static int PingMeshWaitSyncSockets(PingMeshEnv& e, int phyid, uint32_t localIp, uint32_t peerIp)
{
    char serverTag[SOCK_CONN_TAG_SIZE] = {0};
    char clientTag[SOCK_CONN_TAG_SIZE] = {0};
    PingMeshIpTag(localIp, serverTag, sizeof(serverTag));
    PingMeshIpTag(peerIp, clientTag, sizeof(clientTag));

    bool serverDone = false;
    bool clientDone = false;
    const int spinUs = 5 * 1000;
    for (int spin = 0; (!serverDone || !clientDone) && spin * spinUs < PING_ESTABLISH_MS * 1000; spin++) {
        if (!clientDone) {
            struct SocketInfoT ent = {};
            ent.socketHandle = e.socketHandle;
            ent.remoteIp.addr.s_addr = peerIp;
            (void)snprintf(ent.tag, sizeof(ent.tag), "%s", clientTag);
            unsigned int num = 0;
            int ret = RaGetSockets(1, &ent, 1, &num);
            if (ret == 0 && num == 1 && ent.status == 1 && ent.fdHandle != nullptr) {
                e.clientSock = ent;
                clientDone = true;
                TEST_LOG_INFO(phyid, "client sync socket established (tag=%s fd=%p)", clientTag, ent.fdHandle);
            }
        }
        if (!serverDone) {
            struct SocketInfoT ent = {};
            ent.socketHandle = e.socketHandle;
            ent.remoteIp.addr.s_addr = peerIp;
            (void)snprintf(ent.tag, sizeof(ent.tag), "%s", serverTag);
            unsigned int num = 0;
            int ret = RaGetSockets(0, &ent, 1, &num);
            if (ret == 0 && num == 1 && ent.status == 1 && ent.fdHandle != nullptr) {
                e.serverSock = ent;
                serverDone = true;
                TEST_LOG_INFO(phyid, "server sync socket established (tag=%s fd=%p)", serverTag, ent.fdHandle);
            }
        }
        if (!serverDone || !clientDone) {
            usleep((useconds_t)spinUs);
        }
    }
    if (!serverDone || !clientDone) {
        TEST_LOG_ERROR(
            phyid, "sync sockets not established within %dms (server=%d client=%d)", PING_ESTABLISH_MS, (int)serverDone,
            (int)clientDone);
        return 1;
    }
    return 0;
}

/* Mirrors test_pingmesh_ra_get_dev_eid_info_num + test_pingmesh_ra_get_dev_eid_info_list:
 * resolve the eid + eidIndex whose in4 addr equals the local nic ip. */
static int PingMeshResolveEid(int phyid, uint32_t localIp, union HccpEid* outEid, uint32_t* outEidIndex)
{
    struct RaInfo info = {};
    info.mode = NETWORK_OFFLINE;
    info.phyId = (unsigned int)phyid;
    unsigned int eidNum = 0;
    int ret = RaGetDevEidInfoNum(info, &eidNum);
    if (ret != 0 || eidNum == 0) {
        TEST_LOG_ERROR(phyid, "RaGetDevEidInfoNum failed ret=%d num=%u", ret, eidNum);
        return -1;
    }
    TEST_LOG_INFO(phyid, "RaGetDevEidInfoNum success num=%u", eidNum);

    const unsigned int maxList = 64;
    struct HccpDevEidInfo list[64] = {};
    unsigned int num = (eidNum < maxList) ? eidNum : maxList;
    unsigned int got = num;
    ret = RaGetDevEidInfoList(info, list, &got);
    if (ret != 0 || got != num) {
        TEST_LOG_ERROR(phyid, "RaGetDevEidInfoList failed ret=%d got=%u expect=%u", ret, got, num);
        return -1;
    }
    for (unsigned int i = 0; i < got; i++) {
        TEST_LOG_INFO(
            phyid, "eid[%u] index=%u in4.addr=0x%08x (local 0x%08x)", i, list[i].eidIndex, list[i].eid.in4.addr,
            localIp);
        if (list[i].eid.in4.addr == localIp) {
            *outEid = list[i].eid;
            *outEidIndex = list[i].eidIndex;
            TEST_LOG_INFO(
                phyid, "eid resolved: index=%u dieId=%u chipId=%u funcId=%u", list[i].eidIndex, list[i].dieId,
                list[i].chipId, list[i].funcId);
            return 0;
        }
    }
    TEST_LOG_ERROR(phyid, "local ip not present in the eid list");
    return -1;
}

/* Mirrors test_RaPingInit with test_pingmesh_init_attr_default_init values. */
static int PingMeshInit(
    int phyid, uint32_t phyIdU, const union HccpEid& eid, uint32_t eidIndex, struct PingInitInfo& initInfo,
    void** pingHandle)
{
    struct PingInitAttr initAttr = {};
    initAttr.version = 0;
    initAttr.mode = NETWORK_OFFLINE;
    initAttr.dev.ub.eid = eid;
    initAttr.dev.ub.eidIndex = eidIndex;
    initAttr.ub.phyId = phyIdU;
    initAttr.commInfo.version = 0;
    initAttr.commInfo.ub.qosAttr.tc = (unsigned char)((33 & 0x3f) << 2);
    initAttr.commInfo.ub.qosAttr.sl = 4;
    initAttr.protocol = PROTOCOL_UDMA;
    initAttr.client.ub.cqAttr.sendCqDepth = 128;
    initAttr.client.ub.cqAttr.recvCqDepth = 128;
    initAttr.client.ub.cqAttr.recvCqCompVector = 1;
    initAttr.client.ub.qpAttr.cap.maxInlineData = 32;
    initAttr.client.ub.qpAttr.cap.maxSendSge = 1;
    initAttr.client.ub.qpAttr.cap.maxSendWr = 128;
    initAttr.client.ub.qpAttr.cap.maxRecvSge = 1;
    initAttr.client.ub.qpAttr.cap.maxRecvWr = 128;
    initAttr.server = initAttr.client;
    initAttr.bufferSize = PING_BUFFER_SIZE;

    int ret = RaPingInit(&initAttr, &initInfo, pingHandle);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaPingInit failed ret=%d", ret);
        return ret;
    }
    TEST_LOG_INFO(phyid, "RaPingInit success handle=%p", *pingHandle);
    return 0;
}

/* One host of hlt_hccp_pingmesh_test_001: the symmetric full chain. */
static int PingMeshRun(int phyid)
{
    PingMeshEnv e;
    int ret = PingMeshOfflineBase(e, phyid);
    if (ret != 0) {
        return ret;
    }

    const uint32_t localIp = PingMeshLoopbackIp((uint32_t)phyid, HCCP_VM_IFACE_ROCE);
    const int peerPhyid = (phyid == 1) ? 5 : 1;
    const uint32_t peerIp = PingMeshLoopbackIp((uint32_t)peerPhyid, HCCP_VM_IFACE_ROCE);

    ret = PingMeshSyncSocketSetup(e, phyid, localIp, peerIp);
    if (ret == 0) {
        ret = PingMeshWaitSyncSockets(e, phyid, localIp, peerIp);
    }
    if (ret != 0) {
        return PingMeshTeardown(e, ret);
    }

    union HccpEid eid = {};
    uint32_t eidIndex = 0;
    ret = PingMeshResolveEid(phyid, localIp, &eid, &eidIndex);
    if (ret != 0) {
        return PingMeshTeardown(e, ret);
    }

    struct PingInitInfo initInfo = {};
    ret = PingMeshInit(phyid, (uint32_t)phyid, eid, eidIndex, initInfo, &e.pingHandle);
    if (ret != 0) {
        return PingMeshTeardown(e, ret);
    }

    /* test_pingmesh_info_exchange: my {eid, client qpInfo} goes to the peer via the
     * server-side fd; the peer's arrives on the client-side fd. */
    struct PingTargetCommInfo mine = {};
    mine.eid = eid;
    mine.qpInfo = initInfo.client;
    struct PingTargetCommInfo peerComm = {};
    ret = RaSocketSendAll(e.serverSock.fdHandle, &mine, sizeof(mine));
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "send comm info failed ret=%d", ret);
        return PingMeshTeardown(e, ret);
    }
    ret = RaSocketRecvAll(e.clientSock.fdHandle, &peerComm, sizeof(peerComm));
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "recv comm info failed ret=%d", ret);
        return PingMeshTeardown(e, ret);
    }
    TEST_LOG_INFO(
        phyid, "ping comm info exchanged: peer eid.in4.addr=0x%08x peer qp token=0x%x", peerComm.eid.in4.addr,
        peerComm.qpInfo.ub.tokenValue);

    /* One fixed-size ping round (payload_policy=0). */
    struct PingTargetInfo target = {};
    target.version = 0;
    target.localInfo.version = 0;
    target.localInfo.ub.qosAttr.tc = (unsigned char)((33 & 0x3f) << 2);
    target.localInfo.ub.qosAttr.sl = 4;
    target.remoteInfo = peerComm;
    (void)memset(target.payload.buffer, 0xff, sizeof(target.payload.buffer));
    target.payload.size = PING_PAYLOAD_SIZE;

    ret = RaPingTargetAdd(e.pingHandle, &target, 1);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaPingTargetAdd failed ret=%d", ret);
        return PingMeshTeardown(e, ret);
    }

    struct PingTaskAttr taskAttr = {};
    taskAttr.packetCnt = PING_PACKET_CNT;
    taskAttr.packetInterval = PING_PACKET_INTERVAL;
    taskAttr.timeoutInterval = PING_TIMEOUT_INTERVAL;
    ret = RaPingTaskStart(e.pingHandle, &taskAttr);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaPingTaskStart failed ret=%d", ret);
        return PingMeshTeardown(e, ret);
    }

    /* test_pingmesh_RaPingGetResults: retry on ROCE_EAGAIN (= ConverReturnCode(RDMA_OP,
     * -EAGAIN) = 128101) until the task completes. The original seeds each result's
     * remoteInfo from its target_info so the service can match the target node
     * (rs_ping_urma.c rs_ping_urma_find_target_node compares the qpInfo bytes). */
    struct PingTargetResult result = {};
    result.remoteInfo = peerComm;
    unsigned int resultNum = 1;
    const int spinUs = 1000;
    int spin = 0;
    do {
        resultNum = 1;
        ret = RaPingGetResults(e.pingHandle, &result, &resultNum);
        if (ret == ROCE_EAGAIN) {
            if (spin * spinUs >= PING_RESULTS_MS * 1000) {
                break;
            }
            usleep((useconds_t)spinUs);
            spin++;
        }
    } while (ret == ROCE_EAGAIN);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaPingGetResults failed ret=%d after %dms", ret, spin * spinUs / 1000);
        return PingMeshTeardown(e, ret);
    }
    if (resultNum != 1) {
        TEST_LOG_ERROR(phyid, "RaPingGetResults num=%u expect=1", resultNum);
        return PingMeshTeardown(e, -1);
    }

    /* Original only printed the summary; strengthened per the t1 survey: the data plane
     * is only proven when the state is VALID and packets were received. */
    const struct PingResultInfo& rr = result.result;
    TEST_LOG_INFO(
        phyid,
        "ping result: state=%d rttMin=%uus rttMax=%uus rttAvg=%uus sendCnt=%u recvCnt=%u timeoutCnt=%u "
        "taskId=%u packetCnt=%u packetInterval=%u timeoutInterval=%u",
        (int)rr.state, rr.summary.rttMin, rr.summary.rttMax, rr.summary.rttAvg, rr.summary.sendCnt, rr.summary.recvCnt,
        rr.summary.timeoutCnt, rr.summary.taskId, rr.summary.taskAttr.packetCnt, rr.summary.taskAttr.packetInterval,
        rr.summary.taskAttr.timeoutInterval);
    if (rr.state != PING_RESULT_STATE_VALID) {
        TEST_LOG_ERROR(phyid, "ping result state=%d is not PING_RESULT_STATE_VALID", (int)rr.state);
        return PingMeshTeardown(e, -1);
    }
    if (rr.summary.sendCnt == 0 || rr.summary.recvCnt == 0) {
        TEST_LOG_ERROR(phyid, "ping counters empty: sendCnt=%u recvCnt=%u", rr.summary.sendCnt, rr.summary.recvCnt);
        return PingMeshTeardown(e, -1);
    }

    /* test_pingmesh_shake_hand: send "sync" via the server fd, receive it on the client
     * fd, and verify the content (the original received without checking). */
    static const char syncMsg[] = "sync";
    char syncRecv[sizeof(syncMsg)] = {0};
    ret = RaSocketSendAll(e.serverSock.fdHandle, syncMsg, sizeof(syncMsg));
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "handshake send failed ret=%d", ret);
        return PingMeshTeardown(e, ret);
    }
    ret = RaSocketRecvAll(e.clientSock.fdHandle, syncRecv, sizeof(syncRecv));
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "handshake recv failed ret=%d", ret);
        return PingMeshTeardown(e, ret);
    }
    if (memcmp(syncRecv, syncMsg, sizeof(syncMsg)) != 0) {
        TEST_LOG_ERROR(phyid, "handshake content mismatch: got=%.5s expect=%s", syncRecv, syncMsg);
        return PingMeshTeardown(e, -1);
    }

    ret = RaPingTaskStop(e.pingHandle);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaPingTaskStop failed ret=%d", ret);
        return PingMeshTeardown(e, ret);
    }
    ret = RaPingTargetDel(e.pingHandle, &peerComm, 1);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaPingTargetDel failed ret=%d", ret);
        return PingMeshTeardown(e, ret);
    }

    return PingMeshTeardown(e, 0);
}

/* Ported from hlt_hccp_pingmesh_test_001 (payload_policy=0, one fixed-size round). */
TEST_F(HccpSystemTest, HltPingmeshBasic)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        int ret = PingMeshRun(phyid);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "HltPingmeshBasic failed ret=%d", ret);
        }
        return ret;
    });
}

/* ---- HltPingmeshMesh: the multi-target topology of ping_mesh.json, inlined ---- */

struct PingMeshMeshEnv {
    int phyid = 0;
    bool raInited = false;
    bool netOpened = false;
    struct RaInitConfig cfg = {};
    void* socketHandle = nullptr;
    bool listening = false;
    /* zero-initialized so untouched slots read as "no fd yet" (the duplicate check and
     * the teardown walk rely on fdHandle being null for unused slots) */
    struct SocketInfoT serverSock[PINGMESH_MESH_MAX_N - 1] = {};
    struct SocketInfoT clientSock[PINGMESH_MESH_MAX_N - 1] = {};
    int peerPhyid[PINGMESH_MESH_MAX_N - 1] = {0};
    int peerCnt = 0;
    void* pingHandle = nullptr;
};

static int PingMeshMeshTeardown(PingMeshMeshEnv& e, int ret)
{
    if (e.pingHandle != nullptr) {
        int r = RaPingDeinit(e.pingHandle);
        if (r != 0) {
            TEST_LOG_ERROR(e.phyid, "RaPingDeinit failed ret=%d", r);
            ret = -1;
        }
        e.pingHandle = nullptr;
    }
    /* one close entry per socket fd: up to (N-1) server + (N-1) client conns */
    struct SocketCloseInfoT closeInfo[2 * (PINGMESH_MESH_MAX_N - 1)];
    int closeCnt = 0;
    for (int i = 0; i < e.peerCnt; i++) {
        if (e.serverSock[i].fdHandle != nullptr) {
            closeInfo[closeCnt].socketHandle = e.socketHandle;
            closeInfo[closeCnt].fdHandle = e.serverSock[i].fdHandle;
            closeInfo[closeCnt].disuseLinger = 0;
            closeCnt++;
            e.serverSock[i].fdHandle = nullptr;
        }
    }
    for (int i = 0; i < e.peerCnt; i++) {
        if (e.clientSock[i].fdHandle != nullptr) {
            closeInfo[closeCnt].socketHandle = e.socketHandle;
            closeInfo[closeCnt].fdHandle = e.clientSock[i].fdHandle;
            closeInfo[closeCnt].disuseLinger = 0;
            closeCnt++;
            e.clientSock[i].fdHandle = nullptr;
        }
    }
    if (closeCnt > 0) {
        int r = RaSocketBatchClose(closeInfo, closeCnt);
        if (r != 0) {
            TEST_LOG_ERROR(e.phyid, "RaSocketBatchClose(%d fds) failed ret=%d", closeCnt, r);
            ret = -1;
        }
    }
    if (e.listening) {
        struct SocketListenInfoT listenInfo = {};
        listenInfo.socketHandle = e.socketHandle;
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

/* Mesh variant of the sync socket setup: one listener on PINGMESH_MESH_PORT, one
 * whitelist entry per peer (same tag derivation as the basic case), one connect per
 * peer. e.serverSock/e.clientSock are indexed by the peer's position in peerPhyid[]. */
static int PingMeshMeshSocketsSetup(PingMeshMeshEnv& e, int phyid, uint32_t localIp, const int* peerPhyid, int peerCnt)
{
    e.peerCnt = peerCnt;
    for (int i = 0; i < peerCnt; i++) {
        e.peerPhyid[i] = peerPhyid[i];
    }

    struct SocketInitInfoT socketInit = {};
    socketInit.rdevInfo.phyId = (unsigned int)phyid;
    socketInit.rdevInfo.family = AF_INET;
    socketInit.rdevInfo.localIp.addr.s_addr = localIp;
    socketInit.scopeId = 4;
    int ret = RaSocketInitV1(NETWORK_OFFLINE, socketInit, &e.socketHandle);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaSocketInitV1 failed ret=%d", ret);
        return ret;
    }

    struct SocketListenInfoT listenInfo = {};
    listenInfo.socketHandle = e.socketHandle;
    listenInfo.port = PINGMESH_MESH_PORT;
    ret = RaSocketListenStart(&listenInfo, 1);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaSocketListenStart failed ret=%d", ret);
        return ret;
    }
    e.listening = true;

    struct SocketWlistInfoT wlist[PINGMESH_MESH_MAX_N - 1] = {};
    for (int i = 0; i < peerCnt; i++) {
        wlist[i].remoteIp.addr.s_addr = PingMeshLoopbackIp((uint32_t)peerPhyid[i], HCCP_VM_IFACE_ROCE);
        wlist[i].connLimit = 1;
        PingMeshIpTag(localIp, wlist[i].tag, sizeof(wlist[i].tag));
    }
    ret = RaSocketWhiteListAdd(e.socketHandle, wlist, peerCnt);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaSocketWhiteListAdd(%d peers) failed ret=%d", peerCnt, ret);
        return ret;
    }

    struct SocketConnectInfoT conn[PINGMESH_MESH_MAX_N - 1] = {};
    for (int i = 0; i < peerCnt; i++) {
        const uint32_t peerIp = PingMeshLoopbackIp((uint32_t)peerPhyid[i], HCCP_VM_IFACE_ROCE);
        conn[i].socketHandle = e.socketHandle;
        conn[i].remoteIp.addr.s_addr = peerIp;
        conn[i].port = PINGMESH_MESH_PORT;
        PingMeshIpTag(peerIp, conn[i].tag, sizeof(conn[i].tag));
    }
    ret = RaSocketBatchConnect(conn, peerCnt);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaSocketBatchConnect(%d peers) failed ret=%d", peerCnt, ret);
        return ret;
    }
    return 0;
}

/* Mesh variant of the establishment poll. Client-role conns are queried per peer (the
 * rs matches the query remoteIp + tag, rs_socket.c RsFindSockets client path). The
 * server-role query ignores the query remoteIp: every peer connects with the same tag
 * (the decimal of MY local ip), so each one-shot query (connTmp->isGot) returns one
 * still-unclaimed conn whose result remoteIp is backfilled with the actual peer ip
 * (RsSocketsBackfill) - that identifies the peer slot. */
static int PingMeshMeshWaitSockets(PingMeshMeshEnv& e, int phyid, uint32_t localIp)
{
    char serverTag[SOCK_CONN_TAG_SIZE] = {0};
    PingMeshIpTag(localIp, serverTag, sizeof(serverTag));

    bool cliDone[PINGMESH_MESH_MAX_N - 1] = {};
    int srvGot = 0;
    const int spinUs = 5 * 1000;
    int spin = 0;
    for (;; spin++) {
        for (int i = 0; i < e.peerCnt; i++) {
            if (cliDone[i]) {
                continue;
            }
            const uint32_t peerIp = PingMeshLoopbackIp((uint32_t)e.peerPhyid[i], HCCP_VM_IFACE_ROCE);
            struct SocketInfoT ent = {};
            ent.socketHandle = e.socketHandle;
            ent.remoteIp.addr.s_addr = peerIp;
            PingMeshIpTag(peerIp, ent.tag, sizeof(ent.tag));
            unsigned int num = 0;
            int ret = RaGetSockets(1, &ent, 1, &num);
            if (ret == 0 && num == 1 && ent.status == 1 && ent.fdHandle != nullptr) {
                e.clientSock[i] = ent;
                cliDone[i] = true;
            }
        }
        while (srvGot < e.peerCnt) {
            struct SocketInfoT ent = {};
            ent.socketHandle = e.socketHandle;
            (void)snprintf(ent.tag, sizeof(ent.tag), "%s", serverTag);
            unsigned int num = 0;
            int ret = RaGetSockets(0, &ent, 1, &num);
            if (ret != 0 || num != 1 || ent.status != 1 || ent.fdHandle == nullptr) {
                break;
            }
            int slot = -1;
            for (int i = 0; i < e.peerCnt; i++) {
                if (PingMeshLoopbackIp((uint32_t)e.peerPhyid[i], HCCP_VM_IFACE_ROCE) == ent.remoteIp.addr.s_addr) {
                    slot = i;
                    break;
                }
            }
            if (slot < 0) {
                TEST_LOG_ERROR(phyid, "server conn from unknown ip 0x%08x", ent.remoteIp.addr.s_addr);
                return 1;
            }
            if (e.serverSock[slot].fdHandle != nullptr) {
                TEST_LOG_ERROR(phyid, "duplicate server conn for peer %d", e.peerPhyid[slot]);
                return 1;
            }
            e.serverSock[slot] = ent;
            srvGot++;
            TEST_LOG_INFO(phyid, "server conn from peer %d (fd=%p)", e.peerPhyid[slot], ent.fdHandle);
        }
        bool all = (srvGot == e.peerCnt);
        for (int i = 0; i < e.peerCnt; i++) {
            all = all && cliDone[i];
        }
        if (all) {
            TEST_LOG_INFO(phyid, "all %d sync socket pairs established", e.peerCnt);
            return 0;
        }
        if (spin * spinUs >= PING_ESTABLISH_MS * 1000) {
            TEST_LOG_ERROR(
                phyid, "sync sockets incomplete within %dms (server=%d/%d)", PING_ESTABLISH_MS, srvGot, e.peerCnt);
            return 1;
        }
        usleep((useconds_t)spinUs);
    }
}

/* One node of the full-interconnect mesh: ping every other node in a single task. */
static int PingMeshMeshRun(int phyid, int meshN)
{
    PingMeshMeshEnv e;
    int ret = PingMeshOfflineBaseT(e, phyid, PingMeshMeshTeardown);
    if (ret != 0) {
        return ret;
    }

    const uint32_t localIp = PingMeshLoopbackIp((uint32_t)phyid, HCCP_VM_IFACE_ROCE);
    int peerPhyid[PINGMESH_MESH_MAX_N - 1];
    int peerCnt = 0;
    for (int p = 0; p < meshN; p++) {
        if (p != phyid) {
            peerPhyid[peerCnt++] = p;
        }
    }

    ret = PingMeshMeshSocketsSetup(e, phyid, localIp, peerPhyid, peerCnt);
    if (ret == 0) {
        ret = PingMeshMeshWaitSockets(e, phyid, localIp);
    }
    if (ret != 0) {
        return PingMeshMeshTeardown(e, ret);
    }

    union HccpEid eid = {};
    uint32_t eidIndex = 0;
    ret = PingMeshResolveEid(phyid, localIp, &eid, &eidIndex);
    if (ret != 0) {
        return PingMeshMeshTeardown(e, ret);
    }

    struct PingInitInfo initInfo = {};
    ret = PingMeshInit(phyid, (uint32_t)phyid, eid, eidIndex, initInfo, &e.pingHandle);
    if (ret != 0) {
        return PingMeshMeshTeardown(e, ret);
    }

    /* Comm info fan-out: every node first sends its {eid, client qpInfo} on all
     * server-side fds, then receives on all client-side fds (the sends are small enough
     * to sit in the peer socket buffers, so the symmetric order cannot deadlock). The
     * peer identity comes from the exchanged eid, cross-checked against the expected
     * loopback ip of that peer slot. */
    struct PingTargetCommInfo mine = {};
    mine.eid = eid;
    mine.qpInfo = initInfo.client;
    struct PingTargetCommInfo peerComm[PINGMESH_MESH_MAX_N - 1] = {};
    for (int i = 0; i < peerCnt; i++) {
        ret = RaSocketSendAll(e.serverSock[i].fdHandle, &mine, sizeof(mine));
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "send comm info to peer %d failed ret=%d", e.peerPhyid[i], ret);
            return PingMeshMeshTeardown(e, ret);
        }
    }
    for (int i = 0; i < peerCnt; i++) {
        ret = RaSocketRecvAll(e.clientSock[i].fdHandle, &peerComm[i], sizeof(peerComm[i]));
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "recv comm info from peer %d failed ret=%d", e.peerPhyid[i], ret);
            return PingMeshMeshTeardown(e, ret);
        }
        const uint32_t expectIp = PingMeshLoopbackIp((uint32_t)e.peerPhyid[i], HCCP_VM_IFACE_ROCE);
        if (peerComm[i].eid.in4.addr != expectIp) {
            TEST_LOG_ERROR(
                phyid, "comm info of peer %d has eid 0x%08x, expect 0x%08x", e.peerPhyid[i], peerComm[i].eid.in4.addr,
                expectIp);
            return PingMeshMeshTeardown(e, -1);
        }
    }
    TEST_LOG_INFO(phyid, "comm info exchanged with %d peers", peerCnt);

    /* One batch TargetAdd of the whole target list (json target_ip[] -> targets[]). */
    struct PingTargetInfo targets[PINGMESH_MESH_MAX_N - 1] = {};
    for (int i = 0; i < peerCnt; i++) {
        targets[i].version = 0;
        targets[i].localInfo.version = 0;
        targets[i].localInfo.ub.qosAttr.tc = (unsigned char)((33 & 0x3f) << 2);
        targets[i].localInfo.ub.qosAttr.sl = 4;
        targets[i].remoteInfo = peerComm[i];
        (void)memset(targets[i].payload.buffer, 0xff, sizeof(targets[i].payload.buffer));
        targets[i].payload.size = PING_PAYLOAD_SIZE;
    }
    ret = RaPingTargetAdd(e.pingHandle, targets, peerCnt);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaPingTargetAdd(%d) failed ret=%d", peerCnt, ret);
        return PingMeshMeshTeardown(e, ret);
    }

    /* json task_attr with packet_cnt scaled to the sim time budget when the mesh grows
     * (see the file header: the rs ping thread walks all targets once per packet). */
    struct PingTaskAttr taskAttr = {};
    taskAttr.packetCnt = PING_PACKET_CNT;
    const unsigned int budgetCnt = (unsigned int)(PINGMESH_TASK_BUDGET_MS / (peerCnt * (int)PING_PACKET_INTERVAL));
    if (budgetCnt < taskAttr.packetCnt) {
        taskAttr.packetCnt = (budgetCnt < 10) ? 10 : budgetCnt;
    }
    taskAttr.packetInterval = PING_PACKET_INTERVAL;
    taskAttr.timeoutInterval = PING_TIMEOUT_INTERVAL;
    TEST_LOG_INFO(phyid, "mesh task: peers=%d packetCnt=%u", peerCnt, taskAttr.packetCnt);
    ret = RaPingTaskStart(e.pingHandle, &taskAttr);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaPingTaskStart failed ret=%d", ret);
        return PingMeshMeshTeardown(e, ret);
    }

    /* GetResults in batch form: every result slot carries its peer's remoteInfo (same
     * matching contract as the basic case), retried on ROCE_EAGAIN until the task ends. */
    struct PingTargetResult results[PINGMESH_MESH_MAX_N - 1] = {};
    for (int i = 0; i < peerCnt; i++) {
        results[i].remoteInfo = peerComm[i];
    }
    unsigned int resultNum = 0;
    const int spinUs = 1000;
    int spin = 0;
    do {
        resultNum = (unsigned int)peerCnt;
        ret = RaPingGetResults(e.pingHandle, results, &resultNum);
        if (ret == ROCE_EAGAIN) {
            if (spin * spinUs >= PING_RESULTS_MS * 1000) {
                break;
            }
            usleep((useconds_t)spinUs);
            spin++;
        }
    } while (ret == ROCE_EAGAIN);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaPingGetResults failed ret=%d after %dms", ret, spin * spinUs / 1000);
        return PingMeshMeshTeardown(e, ret);
    }
    if (resultNum != (unsigned int)peerCnt) {
        TEST_LOG_ERROR(phyid, "RaPingGetResults num=%u expect=%d", resultNum, peerCnt);
        return PingMeshMeshTeardown(e, -1);
    }

    /* Strengthened per-target verification: every directed pair must be VALID with
     * traffic in both counters. */
    for (int i = 0; i < peerCnt; i++) {
        const struct PingResultInfo& rr = results[i].result;
        TEST_LOG_INFO(
            phyid,
            "mesh result peer=%d: state=%d rttMin=%uus rttMax=%uus rttAvg=%uus sendCnt=%u recvCnt=%u "
            "timeoutCnt=%u",
            e.peerPhyid[i], (int)rr.state, rr.summary.rttMin, rr.summary.rttMax, rr.summary.rttAvg, rr.summary.sendCnt,
            rr.summary.recvCnt, rr.summary.timeoutCnt);
        if (rr.state != PING_RESULT_STATE_VALID) {
            TEST_LOG_ERROR(
                phyid, "peer %d result state=%d is not PING_RESULT_STATE_VALID", e.peerPhyid[i], (int)rr.state);
            return PingMeshMeshTeardown(e, -1);
        }
        if (rr.summary.sendCnt == 0 || rr.summary.recvCnt == 0) {
            TEST_LOG_ERROR(
                phyid, "peer %d counters empty: sendCnt=%u recvCnt=%u", e.peerPhyid[i], rr.summary.sendCnt,
                rr.summary.recvCnt);
            return PingMeshMeshTeardown(e, -1);
        }
    }

    /* "sync" handshake fan-out with content check on every pair. */
    static const char syncMsg[] = "sync";
    char syncRecv[sizeof(syncMsg)] = {0};
    for (int i = 0; i < peerCnt; i++) {
        ret = RaSocketSendAll(e.serverSock[i].fdHandle, syncMsg, sizeof(syncMsg));
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "handshake send to peer %d failed ret=%d", e.peerPhyid[i], ret);
            return PingMeshMeshTeardown(e, ret);
        }
    }
    for (int i = 0; i < peerCnt; i++) {
        ret = RaSocketRecvAll(e.clientSock[i].fdHandle, syncRecv, sizeof(syncRecv));
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "handshake recv from peer %d failed ret=%d", e.peerPhyid[i], ret);
            return PingMeshMeshTeardown(e, ret);
        }
        if (memcmp(syncRecv, syncMsg, sizeof(syncMsg)) != 0) {
            TEST_LOG_ERROR(phyid, "handshake content mismatch with peer %d", e.peerPhyid[i]);
            return PingMeshMeshTeardown(e, -1);
        }
    }

    ret = RaPingTaskStop(e.pingHandle);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaPingTaskStop failed ret=%d", ret);
        return PingMeshMeshTeardown(e, ret);
    }
    ret = RaPingTargetDel(e.pingHandle, peerComm, peerCnt);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaPingTargetDel(%d) failed ret=%d", peerCnt, ret);
        return PingMeshMeshTeardown(e, ret);
    }

    return PingMeshMeshTeardown(e, 0);
}

/* Ported from hlt_hccp_pingmesh_test_001 driven by ping_mesh.json target_ip lists: the
 * full-interconnect mesh form (each node pings all others in one task). */
TEST_F(HccpSystemTest, HltPingmeshMesh)
{
    int meshN = PINGMESH_MESH_DEFAULT_N;
    if (const char* env = getenv("HCCP_VM_PINGMESH_NPROC")) {
        long v = atol(env);
        if (v >= 2 && v <= PINGMESH_MESH_MAX_N) {
            meshN = (int)v;
        }
    }
    std::vector<int> phyids;
    for (int i = 0; i < meshN; i++) {
        phyids.push_back(i);
    }
    RunTests(phyids, [this, meshN](int phyid) -> int {
        int ret = PingMeshMeshRun(phyid, meshN);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "HltPingmeshMesh failed ret=%d", ret);
        }
        return ret;
    });
}
