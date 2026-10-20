/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* Ported UB data-plane entries of the hlt network test repo
 * (cann-hccl/hlt/network/hccp_test): hlt_hccp_ub_test.c entries 001/002/graph/
 * dwqe/writevalue/ccu_jetty 001+002/udma_write_value_002 plus the three
 * opbase/graph/dwqe loop entries of hlt_hccp_ub_test_loop.c.
 *
 * Port conventions (same as the other migrated files):
 * - Every original hlt_* entry becomes TEST_F(HccpSystemTest, <semantic name>)
 *   with a "Ported from hlt_xxx" comment on the line above.
 * - The original config json is not read; the CI default values are inlined as
 *   constants below (qp_num=1, wqe_num=2, tp_num=1 (ctp), mr size 4096 with
 *   read|write access, sqebb_num=64, transport CONN_RM, jetty mode 3
 *   (USER_CTL_NORMAL, share jfr, jfs flag 1) or 2 (CCU) for the ccu_jetty
 *   entries, jetty import mode EXP, token mode 1, async mode 0, rep_time 1).
 * - The original sync channel is the Ra socket layer (test_socket_resource_init:
 *   RaSocketInit + listen/whitelist on the server machine, batch connect on the
 *   client machine); the port keeps exactly that channel for every exchange and
 *   for the 5-byte "sync" shake hand.
 * - Two-machine entries (machine_type 0/1) run as RunTests({1, 5}) with the
 *   server role on phyId 1 and the client role on phyId 5. Loop entries are
 *   single-process dual-config in the original, so they run as one host child
 *   that builds both halves (server/client) against one shared ctx.
 *
 * Sim adaptations (documented per case where they apply):
 * - The last WQE of a data batch is an inline WRITE that records value 1 into
 *   the peer's notify cell; the sim lands inline data synchronously through the
 *   urma stub's segment write, so rtNotifyWait observes the record (same
 *   convention as the NotifyBarrier helper).
 * - The write_with_notify variant builds RA_UB_OPC_WRITE_NOTIFY WQEs: the
 *   WrNotifyInfo struct is documented as "required for opcode
 *   RA_UB_OPC_WRITE_NOTIFY", and the ra layer only ships notifyInfo for that
 *   opcode, so the original's builder (notify fields, opcode left at the json
 *   default) is ported with the opcode that makes the notify path engage.
 * - The graph entries use pi_type=1: the original builds sqebb_num WQEs and the
 *   batch send posts sqebb_num WQEs only in pi mode, so pi_type=1 is the only
 *   self-consistent reading of the two-machine graph entry.
 * - The original rnotify exchange ships the lnotify registration window
 *   (page-aligned base, offset+8 bytes) and the notify WQE writes at that
 *   address; on hardware every notify unit owns its page, so the window base is
 *   the cell itself. The sim sub-allocates 8-byte cells inside one pool page,
 *   so the second notify of a device (the loop halves) would miss its cell. The
 *   port exchanges the notify cell address directly (the intent of the original
 *   exchange) while keeping the page-window registration.
 * - udma_write_value_002 sleeps 8s twice as an observation window for async CQE
 *   errors; the framework kills a host child after ~5s, so the window is
 *   shortened to 1s (the sim has no async error source; the query itself is
 *   the tested object).
 * - The original ccu_jetty_test_002 deinit path (test_hccp_resource_deinit_batch)
 *   only calls RaDeinit and leaks ctx/qp/segments; the port runs the full
 *   teardown instead (same defect class as the t1 section 7 fixes).
 */

#include "cases_common.h"
#include "hccp_ccu_tlv_shim.h"

#include <arpa/inet.h>
#include <runtime/runtime/dev.h>
#include <runtime/runtime/event.h>
#include <runtime/runtime/mem.h>
#include <runtime/runtime/stream.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include "hccp_async.h"
#include "hccp_async_ctx.h"

/* rtGetDevResAddress lives in the rts_device.h family, which cannot be included
 * together with the runtime/runtime headers; the aclrt* entries come from the
 * real <acl/acl_rt.h> via cases_common.h, implemented by the runtime stub. */
extern "C" rtError_t rtGetDevResAddress(rtDevResInfo* const resInfo, rtDevResAddrInfo* const addrInfo);

namespace {

/* ---------------------------------------------------------------- constants */

constexpr unsigned int UB_DP_QP_NUM = 1U;        /* DEFAULT_QP_NUM */
constexpr unsigned int UB_DP_WQE_NUM = 2U;       /* DEFAULT_QP_WQE_NUM */
constexpr unsigned int UB_DP_TP_NUM = 1U;        /* DEFAULT_TP_NUM */
constexpr unsigned int UB_DP_SQ_DEPTH = 4096U;   /* DEFAULT_QP_SQ_DEPTH */
constexpr unsigned int UB_DP_RQ_DEPTH = 256U;    /* DEFAULT_QP_RQ_DEPTH */
constexpr unsigned int UB_DP_RNR_RETRY = 7U;     /* DEFAULT_QP_RNR_RETRY */
constexpr unsigned int UB_DP_SQEBB_NUM = 64U;    /* DEFAULT_QP_SQEBB_NUM */
constexpr unsigned int UB_DP_CQ_DEPTH = 16384U;  /* DEFAULT_CQ_DEPTH (non-CCU) */
constexpr unsigned int UB_DP_CQ_DEPTH_CCU = 64U; /* DEFAULT_CQ_DEPTH_CCU */
constexpr unsigned int UB_DP_SEG_SIZE = 4096U;   /* DEFAULT_MR_SIZE */
constexpr unsigned int UB_DP_NOTIFY_SIZE = 8U;   /* RT_NOTIFY_SIZE */
constexpr unsigned int UB_DP_PAGE_SIZE = 4096U;  /* UB_MEM_PAGE_SIZE */
constexpr unsigned int UB_DP_MAX_WQE = 64U;      /* sqebb_num bound of the graph build */
constexpr unsigned int UB_DP_BATCH_MAX = 32U;    /* RaBatchSendWr batch of test_ra_batch_send_wr_v1 */
constexpr unsigned int UB_DP_SEG_ACCESS = (MEM_SEG_ACCESS_READ | MEM_SEG_ACCESS_WRITE);
constexpr uint64_t UB_DP_NOTIFY_VALUE = 1U;     /* g_notify_value / SERVER record value */
constexpr unsigned int UB_DP_WQE_SIZE_64 = 64U; /* UB_WQE_SIZE_64 */
constexpr unsigned int UB_DP_WQE_SIZE_128 = 128U;
constexpr unsigned int UB_DP_TP_FLAG_CTP = 0U; /* DEFAULT_TP_FLAG: ctp */
constexpr int UB_DP_SOCK_ESTABLISH_MS = 10000;

/* rendezvous ports (one per case; t4 used 15700-15727, t3 15750+) */
constexpr unsigned short UB_PORT_DP_DBSEND = 15730;
constexpr unsigned short UB_PORT_DP_DBSEND_NOTIFY = 15731;
constexpr unsigned short UB_PORT_DP_GRAPH = 15732;
constexpr unsigned short UB_PORT_DP_DWQE = 15733;
constexpr unsigned short UB_PORT_DP_WRITEVALUE = 15734;
constexpr unsigned short UB_PORT_DP_CCU_JETTY = 15735;
constexpr unsigned short UB_PORT_DP_CCU_JETTY_002 = 15736;
constexpr unsigned short UB_PORT_DP_ERRINFO = 15737;
constexpr unsigned short UB_PORT_DP_LOOP_DBSEND = 15738;
constexpr unsigned short UB_PORT_DP_LOOP_DWQE = 15739;
constexpr unsigned short UB_PORT_DP_LOOP_GRAPH = 15740;

/* ----------------------------------------------------------------- env state */

/* One end (machine or loop half) of a data-plane case. The loop entries build
 * two of these against one shared ctx (only the server half owns the base init
 * fields: net service / RaInit / ctx / token). */
struct UbDpEnv {
    int phyid = 0;
    bool isServer = false;

    /* base init (server half owns them in the loop cases) */
    bool netOpened = false;
    bool rtDeviceSet = false;
    bool raInited = false;
    struct RaInitConfig raCfg = {};
    struct CtxInitAttr ctxAttr = {};
    void* ctxHandle = nullptr;
    void* tokenHandle = nullptr;

    /* sync socket channel */
    void* socketHandle = nullptr;
    bool listening = false;
    void* sockFd = nullptr;

    /* tp resources (ctp) */
    union HccpEid remoteEid = {};
    struct HccpTpInfo localTp[UB_DP_TP_NUM] = {};
    struct HccpTpInfo remoteTp[UB_DP_TP_NUM] = {};
    unsigned int localPsn[UB_DP_TP_NUM] = {0};
    unsigned int remotePsn[UB_DP_TP_NUM] = {0};
    bool psnSet = false; /* loop entries keep the psns at 0 (no rand step) */

    /* chan/cq/qp */
    void* chanHandle = nullptr;
    struct CqInfoT cqInfo = {};
    void* cqHandle = nullptr;
    struct QpCreateAttr qpAttr = {};
    struct QpCreateInfo qpInfo = {};
    void* qpHandle = nullptr;
    struct QpImportInfoT lqpExport = {};
    struct QpImportInfoT rqpImport = {};
    void* rqpHandle = nullptr;
    bool qpBound = false;

    /* data segment */
    struct MrRegInfoT segReg = {};
    void* segLmem = nullptr;
    struct MrImportInfoT segExport = {};
    struct HccpMemInfo segMem = {};
    struct MrImportInfoT peerSegImport = {};
    struct HccpMemInfo peerSegMem = {};
    void* segRmem = nullptr;

    /* notify segment (skipped when the variant has no notify) */
    bool hasNotify = true;
    aclrtNotify notify = nullptr;
    unsigned int notifyId = 0;
    unsigned long notifyAddr = 0;
    struct MrRegInfoT notifyReg = {};
    void* notifyLmem = nullptr;
    struct MrImportInfoT notifyExport = {};
    struct HccpMemInfo notifyMem = {};
    struct MrImportInfoT peerNotifyImport = {};
    struct HccpMemInfo peerNotifyMem = {};
    void* notifyRmem = nullptr;

    /* rts */
    aclrtStream stream = nullptr;

    /* WQE arrays (wr.sges points into sgeList, one slot per WQE) */
    struct SendWrData wr[UB_DP_MAX_WQE] = {};
    struct SendWrResp opResp[UB_DP_MAX_WQE] = {};
    struct WrSgeList sgeList[UB_DP_MAX_WQE] = {};

    /* ccu mode TLV channel (ccu_jetty entries) */
    void* tlvHandle = nullptr;
};

/* ------------------------------------------------------------------ helpers */

uint32_t UbDpLoopbackIp(uint32_t phyId, uint32_t iface)
{
    struct in_addr virt = {};
    virt.s_addr = htonl(HccpVmNicIp(phyId, iface));
    return HccpVmToLoopback(virt).s_addr;
}

/* Mirrors test_get_socket_handle for one socket: poll RaGetSockets(role) until
 * the socket reports established with the expected remote ip. */
int UbDpWaitSocket(
    int phyid, unsigned int role, void* socketHandle, uint32_t peerIp, const char* tag, struct SocketInfoT* out)
{
    const int spinUs = 5 * 1000;
    for (int spin = 0; spin * spinUs < UB_DP_SOCK_ESTABLISH_MS * 1000; spin++) {
        struct SocketInfoT ent = {};
        ent.socketHandle = socketHandle;
        ent.remoteIp.addr.s_addr = peerIp;
        (void)snprintf(ent.tag, sizeof(ent.tag), "%s", tag);
        unsigned int num = 0;
        int ret = RaGetSockets(role, &ent, 1, &num);
        if (ret == 0 && num == 1 && ent.status == 1 && ent.fdHandle != nullptr) {
            *out = ent;
            TEST_LOG_INFO(phyid, "sync socket established (role=%u fd=%p)", role, ent.fdHandle);
            return 0;
        }
        usleep((useconds_t)spinUs);
    }
    TEST_LOG_ERROR(
        phyid, "sync socket not established within %dms (role=%u tag=%s)", UB_DP_SOCK_ESTABLISH_MS, role, tag);
    return -1;
}

/* test_open_hccp_by_tsd + test_rts_resource_init + test_ra_init_ub + the
 * test_ra_ctx_init resource part (eid num/list + ctx init + token alloc). The
 * sync socket (RaSocketInitV1 .. connect) is set up by UbDpSocketSetup. */
int UbDpBegin(UbDpEnv& env, int phyid, unsigned int notifyNum)
{
    env.phyid = phyid;
    env.isServer = (phyid == 1); /* machine_type 0 = server side, phyId 1 in the sim */

    env.raCfg = {};
    env.raCfg.phyId = (unsigned int)phyid;
    env.raCfg.nicPosition = NETWORK_OFFLINE;
    env.raCfg.hdcType = 18;
    env.raCfg.enableHdcAsync = true; /* test_ra_init_ub sets enable_hdc_async=1 */

    /* test_open_hccp_by_tsd: TSD process pull-up -> rtOpenNetService */
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
        return -1;
    }
    env.netOpened = true;

    /* test_rtSetDevice + test_rtStreamCreate + test_rtNotifyCreate (via the
     * modern aclrt entries) */
    if (aclrtSetDevice(phyid) != 0) {
        TEST_LOG_ERROR(phyid, "aclrtSetDevice failed");
        return -1;
    }
    env.rtDeviceSet = true;
    if (aclrtCreateStream(&env.stream) != 0) {
        TEST_LOG_ERROR(phyid, "aclrtCreateStream failed");
        return -1;
    }
    for (unsigned int i = 0; i < notifyNum; i++) {
        aclrtNotify ntf = nullptr;
        if (aclrtCreateNotify(&ntf, 0) != 0) {
            TEST_LOG_ERROR(phyid, "aclrtCreateNotify %u failed", i);
            return -1;
        }
        uint32_t id = 0;
        if (aclrtGetNotifyId(ntf, &id) != 0) {
            TEST_LOG_ERROR(phyid, "aclrtGetNotifyId %u failed", i);
            return -1;
        }
        unsigned long notifyAddr = 0;
        unsigned int notifyLen = 0;
        rtDevResInfo resInfo = {};
        rtDevResAddrInfo addrInfo = {};
        addrInfo.resAddress = &notifyAddr;
        addrInfo.len = &notifyLen;
        resInfo.dieId = 0;
        resInfo.procType = RT_PROCESS_HCCP;
        resInfo.resType = RT_RES_TYPE_STARS_NOTIFY_RECORD;
        resInfo.flag = 0;
        resInfo.resId = id;
        if (rtGetDevResAddress(&resInfo, &addrInfo) != RT_ERROR_NONE) {
            TEST_LOG_ERROR(phyid, "rtGetDevResAddress %u failed", i);
            return -1;
        }
        env.notify = ntf;
        env.notifyId = id;
        env.notifyAddr = notifyAddr;
        TEST_LOG_INFO(phyid, "notify %u created id=%u addr=0x%lx", i, id, notifyAddr);
    }
    env.hasNotify = (notifyNum != 0);

    /* rtOpenNetService returns only after the device stack reports ready
     * (stub waits on deviceReady, 30s budget), so RaInit must succeed on
     * the first call: a failure is a real defect, not a startup race. */
    int ret = RaInit(&env.raCfg);
    if (ret != 0) {
        (void)RaDeinit(&env.raCfg); /* release any half-initialized RA state */
        TEST_LOG_ERROR(phyid, "RaInit failed ret=%d", ret);
        return -1;
    }
    env.raInited = true;

    /* test_ra_ctx_init = eid num + eid list + (ctx init + token alloc) */
    struct RaInfo raInfo = {};
    raInfo.mode = NETWORK_OFFLINE;
    raInfo.phyId = (unsigned int)phyid;
    unsigned int eidNum = 0;
    ret = RaGetDevEidInfoNum(raInfo, &eidNum);
    if (ret != 0 || eidNum == 0) {
        TEST_LOG_ERROR(phyid, "RaGetDevEidInfoNum failed ret=%d num=%u", ret, eidNum);
        return -1;
    }
    struct HccpDevEidInfo eidList[8] = {};
    unsigned int listNum = eidNum;
    if (listNum > 8) {
        listNum = 8;
    }
    unsigned int got = listNum;
    ret = RaGetDevEidInfoList(raInfo, eidList, &got);
    if (ret != 0 || got == 0) {
        TEST_LOG_ERROR(phyid, "RaGetDevEidInfoList failed ret=%d num=%u", ret, got);
        return -1;
    }
    struct in_addr nicIp;
    nicIp.s_addr = htonl(HccpVmNicIp((uint32_t)phyid, HCCP_VM_IFACE_SOCKET));
    struct in_addr localIp = HccpVmToLoopback(nicIp);
    bool matched = false;
    for (unsigned int i = 0; i < got; i++) {
        if (eidList[i].eid.in4.addr == localIp.s_addr) {
            env.ctxAttr.ub.eid = eidList[i].eid;
            matched = true;
            break;
        }
    }
    if (!matched) {
        TEST_LOG_ERROR(phyid, "no eid matches the socket nic loopback ip");
        return -1;
    }
    env.ctxAttr.phyId = (unsigned int)phyid;
    struct CtxInitCfg ctxCfg = {};
    ctxCfg.mode = NETWORK_OFFLINE;
    ret = RaCtxInit(&ctxCfg, &env.ctxAttr, &env.ctxHandle);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaCtxInit failed ret=%d", ret);
        return -1;
    }
    struct HccpTokenId tokenInfo = {};
    ret = RaCtxTokenIdAlloc(env.ctxHandle, &tokenInfo, &env.tokenHandle);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaCtxTokenIdAlloc failed ret=%d", ret);
        return -1;
    }
    return 0;
}

/* Loop-half rts resources: stream + notify only (the shared init belongs to the
 * server half, mirroring test_rts_resource_init_loop). */
int UbDpHalfRts(UbDpEnv& env, int phyid)
{
    env.phyid = phyid;
    env.isServer = false;
    if (aclrtCreateStream(&env.stream) != 0) {
        TEST_LOG_ERROR(phyid, "client-half aclrtCreateStream failed");
        return -1;
    }
    aclrtNotify ntf = nullptr;
    if (aclrtCreateNotify(&ntf, 0) != 0) {
        TEST_LOG_ERROR(phyid, "client-half aclrtCreateNotify failed");
        return -1;
    }
    uint32_t id = 0;
    if (aclrtGetNotifyId(ntf, &id) != 0) {
        TEST_LOG_ERROR(phyid, "client-half aclrtGetNotifyId failed");
        return -1;
    }
    unsigned long notifyAddr = 0;
    unsigned int notifyLen = 0;
    rtDevResInfo resInfo = {};
    rtDevResAddrInfo addrInfo = {};
    addrInfo.resAddress = &notifyAddr;
    addrInfo.len = &notifyLen;
    resInfo.dieId = 0;
    resInfo.procType = RT_PROCESS_HCCP;
    resInfo.resType = RT_RES_TYPE_STARS_NOTIFY_RECORD;
    resInfo.flag = 0;
    resInfo.resId = id;
    if (rtGetDevResAddress(&resInfo, &addrInfo) != RT_ERROR_NONE) {
        TEST_LOG_ERROR(phyid, "client-half rtGetDevResAddress failed");
        return -1;
    }
    env.notify = ntf;
    env.notifyId = id;
    env.notifyAddr = notifyAddr;
    env.hasNotify = true;
    return 0;
}

/* test_socket_resource_init over the Ra socket layer: init, then server
 * listen+whitelist / client batch connect, and the established wait. */
int UbDpSocketSetup(UbDpEnv& env, unsigned short port)
{
    int phyid = env.phyid;
    uint32_t selfIp = UbDpLoopbackIp((uint32_t)phyid, HCCP_VM_IFACE_SOCKET);
    uint32_t peerIp = UbDpLoopbackIp((uint32_t)(env.isServer ? 5 : 1), HCCP_VM_IFACE_SOCKET);

    struct SocketInitInfoT sockInit = {};
    sockInit.rdevInfo.phyId = (unsigned int)phyid;
    sockInit.rdevInfo.family = AF_INET;
    sockInit.rdevInfo.localIp.addr.s_addr = selfIp;
    sockInit.scopeId = 4;
    int ret = RaSocketInitV1(NETWORK_OFFLINE, sockInit, &env.socketHandle);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaSocketInitV1 failed ret=%d", ret);
        return -1;
    }

    const char* tag = "0";
    if (env.isServer) {
        struct SocketListenInfoT listenInfo = {};
        listenInfo.socketHandle = env.socketHandle;
        listenInfo.port = port;
        ret = RaSocketListenStart(&listenInfo, 1);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaSocketListenStart failed ret=%d (port=%u)", ret, port);
            return -1;
        }
        env.listening = true;
        struct SocketWlistInfoT wlist = {};
        wlist.remoteIp.addr.s_addr = peerIp;
        wlist.connLimit = 1; /* DEFAULT_SOCKET_LIMIT */
        (void)snprintf(wlist.tag, sizeof(wlist.tag), "%s", tag);
        ret = RaSocketWhiteListAdd(env.socketHandle, &wlist, 1);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaSocketWhiteListAdd failed ret=%d", ret);
            return -1;
        }
    } else {
        struct SocketConnectInfoT conn = {};
        conn.socketHandle = env.socketHandle;
        conn.remoteIp.addr.s_addr = peerIp;
        conn.port = port;
        (void)snprintf(conn.tag, sizeof(conn.tag), "%s", tag);
        ret = RaSocketBatchConnect(&conn, 1);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaSocketBatchConnect failed ret=%d", ret);
            return -1;
        }
    }

    struct SocketInfoT sock = {};
    ret = UbDpWaitSocket(phyid, env.isServer ? 0 : 1, env.socketHandle, peerIp, tag, &sock);
    if (ret != 0) {
        return -1;
    }
    env.sockFd = sock.fdHandle;
    return 0;
}

/* Socket exchange of one fixed-size object (test_message_exchange shape: send
 * mine, then receive the peer's). */
template <typename T>
int UbDpExchange(int phyid, void* fd, const T& mine, T* peer)
{
    int ret = RaSocketSendAll(fd, &mine, sizeof(T));
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "exchange send failed ret=%d", ret);
        return -1;
    }
    ret = RaSocketRecvAll(fd, peer, sizeof(T));
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "exchange recv failed ret=%d", ret);
        return -1;
    }
    return 0;
}

/* test_eid_info_exchange */
int UbDpEidExchange(UbDpEnv& env)
{
    int ret = UbDpExchange(env.phyid, env.sockFd, env.ctxAttr.ub.eid, &env.remoteEid);
    if (ret != 0) {
        TEST_LOG_ERROR(env.phyid, "test_eid_info_exchange failed ret=%d", ret);
        return -1;
    }
    TEST_LOG_INFO(env.phyid, "test_eid_info_exchange success");
    return 0;
}

/* test_ra_get_tp_info_list: async tp list with the ctp flag (tp_flag=0) and the
 * qp transport mode (CONN_RM). */
int UbDpTpList(UbDpEnv& env)
{
    struct GetTpCfg tpCfg[UB_DP_TP_NUM] = {};
    for (unsigned int i = 0; i < UB_DP_TP_NUM; i++) {
        tpCfg[i].flag.bs.ctp = 1; /* tp_flag 0 = ctp */
        tpCfg[i].transMode = CONN_RM;
        tpCfg[i].localEid = env.ctxAttr.ub.eid;
        tpCfg[i].peerEid = env.remoteEid;
    }
    struct HccpTpInfo tpList[UB_DP_TP_NUM] = {};
    unsigned int num = UB_DP_TP_NUM;
    void* reqHandle = nullptr;
    int ret = RaGetTpInfoListAsync(env.ctxHandle, tpCfg, tpList, &num, &reqHandle);
    if (ret != 0) {
        TEST_LOG_ERROR(env.phyid, "RaGetTpInfoListAsync failed ret=%d", ret);
        return -1;
    }
    /* test_ra_get_async_req_result poll */
    struct AsyncReqResult result = {};
    int pollRet = -1;
    for (int poll = 0; poll < 2000; poll++) {
        pollRet = RaGetAsyncReqResult(reqHandle, &result);
        if (pollRet == OTHERS_EAGAIN) {
            usleep(10000);
            continue;
        }
        break;
    }
    if (pollRet != 0 || result.reqResult != 0) {
        TEST_LOG_ERROR(env.phyid, "tp list async result failed ret=%d result=%d", pollRet, result.reqResult);
        return -1;
    }
    if (num == 0) {
        TEST_LOG_ERROR(env.phyid, "tp list empty");
        return -1;
    }
    for (unsigned int i = 0; i < num && i < UB_DP_TP_NUM; i++) {
        env.localTp[i] = tpList[i];
    }
    TEST_LOG_INFO(env.phyid, "get tp_info_list success tp=0x%llx", (unsigned long long)env.localTp[0].tpHandle);
    return 0;
}

/* test_tp_info_exchange: tp num + tp list, then a random psn per tp. */
int UbDpTpExchange(UbDpEnv& env)
{
    unsigned int ltpNum = UB_DP_TP_NUM;
    int ret = UbDpExchange(env.phyid, env.sockFd, ltpNum, &ltpNum);
    if (ret != 0) {
        return -1;
    }
    ret = UbDpExchange(env.phyid, env.sockFd, env.localTp, &env.remoteTp);
    if (ret != 0) {
        return -1;
    }
    for (unsigned int i = 0; i < UB_DP_TP_NUM; i++) {
        env.localPsn[i] = (unsigned int)(random() & 0xFFFFu);
    }
    ret = UbDpExchange(env.phyid, env.sockFd, env.localPsn, &env.remotePsn);
    if (ret != 0) {
        return -1;
    }
    env.psnSet = true;
    TEST_LOG_INFO(
        env.phyid, "test_tp_info_exchange success local_tp=0x%llx remote_tp=0x%llx",
        (unsigned long long)env.localTp[0].tpHandle, (unsigned long long)env.remoteTp[0].tpHandle);
    return 0;
}

/* test_ra_ctx_chan_create (zeroed struct; the original read an uninitialized
 * stack value for dataPlaneFlag, defect fixed in the t1 section 7 pass). */
int UbDpChanCreate(UbDpEnv& env)
{
    struct ChanInfoT chanInfo = {};
    int ret = RaCtxChanCreate(env.ctxHandle, &chanInfo, &env.chanHandle);
    if (ret != 0) {
        TEST_LOG_ERROR(env.phyid, "RaCtxChanCreate failed ret=%d", ret);
        return -1;
    }
    return 0;
}

/* test_ra_ctx_cq_create: depth/mode follow the jetty mode (CCU -> 64/CCU poll,
 * otherwise 16384/stars poll); the chan handle comes from the env. */
int UbDpCqCreate(UbDpEnv& env, bool ccuMode)
{
    struct CqInfoT cqInfo = {};
    cqInfo.in.depth = ccuMode ? UB_DP_CQ_DEPTH_CCU : UB_DP_CQ_DEPTH;
    cqInfo.in.ub.mode = ccuMode ? JFC_MODE_CCU_POLL : JFC_MODE_STARS_POLL;
    cqInfo.in.chanHandle = env.chanHandle;
    int ret = RaCtxCqCreate(env.ctxHandle, &cqInfo, &env.cqHandle);
    if (ret != 0) {
        TEST_LOG_ERROR(env.phyid, "RaCtxCqCreate failed ret=%d", ret);
        return -1;
    }
    env.cqInfo = cqInfo;
    return 0;
}

/* test_ra_ctx_qp_create: scq/rcq/srq all point at the cq; token mode 1 assigns
 * the ctx token; sq_cstm stays 0 (USER_CTL_NORMAL default) so no sq buffer
 * alloc; the tail fills the local qp import info. */
int UbDpQpCreate(UbDpEnv& env, enum JettyMode jettyMode, bool piType)
{
    struct QpCreateAttr qpAttr = {};
    qpAttr.scqHandle = env.cqHandle;
    qpAttr.rcqHandle = env.cqHandle;
    qpAttr.srqHandle = env.cqHandle;
    qpAttr.sqDepth = UB_DP_SQ_DEPTH;
    qpAttr.rqDepth = UB_DP_RQ_DEPTH;
    qpAttr.transportMode = CONN_RM;
    qpAttr.ub.mode = jettyMode;
    qpAttr.ub.jettyId = 0;
    qpAttr.ub.flag.bs.shareJfr = 1; /* DEFAULT_QP_JETTY_FLAG */
    qpAttr.ub.jfsFlag.value = 1;
    qpAttr.ub.tokenValue = 0;
    qpAttr.ub.priority = 0;
    qpAttr.ub.rnrRetry = UB_DP_RNR_RETRY;
    qpAttr.ub.errTimeout = 0;
    qpAttr.ub.tokenIdHandle = env.tokenHandle; /* token mode 1 */
    qpAttr.ub.extMode.piType = piType;
    qpAttr.ub.extMode.cstmFlag.bs.sqCstm = 0;
    qpAttr.ub.extMode.sqebbNum = UB_DP_SQEBB_NUM;
    int ret = RaCtxQpCreate(env.ctxHandle, &qpAttr, &env.qpInfo, &env.qpHandle);
    if (ret != 0) {
        TEST_LOG_ERROR(env.phyid, "RaCtxQpCreate failed ret=%d", ret);
        return -1;
    }
    env.qpAttr = qpAttr;
    TEST_LOG_INFO(env.phyid, "RaCtxQpCreate success jetty_id=%u", env.qpInfo.ub.id);

    env.lqpExport.in.key = env.qpInfo.key;
    env.lqpExport.in.ub.tokenValue = qpAttr.ub.tokenValue;
    env.lqpExport.in.ub.policy = JETTY_GRP_POLICY_RR;
    env.lqpExport.in.ub.type = TARGET_TYPE_JETTY;
    env.lqpExport.in.ub.flag.bs.tokenPolicy = MEM_SEG_TOKEN_PLAIN_TEXT; /* == URMA_TOKEN_PLAIN_TEXT */
    return 0;
}

/* test_qp_import_info_t_exchange */
int UbDpQpExchange(UbDpEnv& env)
{
    int ret = UbDpExchange(env.phyid, env.sockFd, env.lqpExport, &env.rqpImport);
    if (ret != 0) {
        TEST_LOG_ERROR(env.phyid, "test_qp_import_info_t_exchange failed ret=%d", ret);
        return -1;
    }
    return 0;
}

/* test_ra_ctx_qp_import: EXP mode with the exchanged tp/psn pair (ctp maps to
 * tp_type 1), matching test_get_exp_import_cfg. */
int UbDpQpImport(UbDpEnv& env)
{
    struct QpImportInfoT importInfo = env.rqpImport;
    importInfo.in.ub.mode = JETTY_IMPORT_MODE_EXP;
    importInfo.in.ub.expImportCfg.tpHandle = env.localTp[0].tpHandle;
    importInfo.in.ub.expImportCfg.peerTpHandle = env.remoteTp[0].tpHandle;
    importInfo.in.ub.expImportCfg.txPsn = env.localPsn[0];
    importInfo.in.ub.expImportCfg.rxPsn = env.remotePsn[0];
    importInfo.in.ub.tpType = 1; /* ctp */
    int ret = RaCtxQpImport(env.ctxHandle, &importInfo, &env.rqpHandle);
    if (ret != 0) {
        TEST_LOG_ERROR(env.phyid, "RaCtxQpImport failed ret=%d", ret);
        return -1;
    }
    return 0;
}

/* test_ra_ctx_qp_bind */
int UbDpQpBind(UbDpEnv& env)
{
    int ret = RaCtxQpBind(env.qpHandle, env.rqpHandle);
    if (ret != 0) {
        TEST_LOG_ERROR(env.phyid, "RaCtxQpBind failed ret=%d", ret);
        return -1;
    }
    env.qpBound = true;
    return 0;
}

/* test_segment_memory_init: one 4K HBM segment per qp, init value 1 on the
 * server machine and 2 on the client machine. */
int UbDpSegMemInit(UbDpEnv& env)
{
    void* devMem = nullptr;
    if (aclrtMalloc(&devMem, UB_DP_SEG_SIZE, ACL_MEM_MALLOC_HUGE_FIRST) != 0) {
        TEST_LOG_ERROR(env.phyid, "aclrtMalloc data segment failed");
        return -1;
    }
    unsigned int initValue = env.isServer ? 0x1U : 0x2U;
    if (aclrtMemset(devMem, UB_DP_SEG_SIZE, initValue, UB_DP_SEG_SIZE) != 0) {
        TEST_LOG_ERROR(env.phyid, "aclrtMemset data segment failed");
        return -1;
    }
    env.segReg.in.mem.addr = (uint64_t)(uintptr_t)devMem;
    env.segReg.in.mem.size = UB_DP_SEG_SIZE;
    env.segMem.addr = env.segReg.in.mem.addr;
    env.segMem.size = UB_DP_SEG_SIZE;
    TEST_LOG_INFO(env.phyid, "segment memory init success addr=0x%llx", (unsigned long long)env.segMem.addr);
    return 0;
}

/* test_ra_ctx_lmem_register: data segment with token mode 1 (token id valid +
 * handle) and read|write access, then the notify segment page-aligned (the
 * notify handle is passed with the valid flag left at the template default 0,
 * exactly like the original lnotify registration). */
int UbDpSegRegister(UbDpEnv& env)
{
    env.segReg.in.ub.flags.bs.access = UB_DP_SEG_ACCESS;
    env.segReg.in.ub.flags.bs.nonPin = 1; /* DEFAULT_MR_NON_PIN */
    env.segReg.in.ub.flags.bs.tokenIdValid = 1;
    env.segReg.in.ub.tokenIdHandle = env.tokenHandle;
    env.segReg.in.ub.tokenValue = 0;
    int ret = RaCtxLmemRegister(env.ctxHandle, &env.segReg, &env.segLmem);
    if (ret != 0) {
        TEST_LOG_ERROR(env.phyid, "lmem RaCtxLmemRegister failed ret=%d", ret);
        return -1;
    }
    env.segExport.in.key = env.segReg.out.key;
    env.segExport.in.ub.flags.bs.cacheable = env.segReg.in.ub.flags.bs.cacheable;
    env.segExport.in.ub.flags.bs.access = env.segReg.in.ub.flags.bs.access;
    env.segExport.in.ub.tokenValue = env.segReg.in.ub.tokenValue;

    if (!env.hasNotify) {
        return 0;
    }
    env.notifyReg = {};
    env.notifyReg.in.mem.addr = env.notifyAddr & ~(unsigned long)(UB_DP_PAGE_SIZE - 1);
    env.notifyReg.in.mem.size
        = UB_DP_NOTIFY_SIZE + (env.notifyAddr - (env.notifyAddr & ~(unsigned long)(UB_DP_PAGE_SIZE - 1)));
    env.notifyReg.in.ub.flags.bs.access = UB_DP_SEG_ACCESS;
    env.notifyReg.in.ub.flags.bs.nonPin = 1;
    env.notifyReg.in.ub.tokenIdHandle = env.tokenHandle;
    ret = RaCtxLmemRegister(env.ctxHandle, &env.notifyReg, &env.notifyLmem);
    if (ret != 0) {
        TEST_LOG_ERROR(env.phyid, "lnotify RaCtxLmemRegister failed ret=%d", ret);
        return -1;
    }
    env.notifyExport.in.key = env.notifyReg.out.key;
    env.notifyExport.in.ub.flags.bs.cacheable = env.notifyReg.in.ub.flags.bs.cacheable;
    env.notifyExport.in.ub.flags.bs.access = env.notifyReg.in.ub.flags.bs.access;
    env.notifyExport.in.ub.tokenValue = env.notifyReg.in.ub.tokenValue;
    return 0;
}

/* Exchange payload of test_rmem_context_exchange: the import info plus the raw
 * mem info of one segment. */
struct UbDpSegRdv {
    struct MrImportInfoT exportInfo;
    struct HccpMemInfo mem;
};

/* test_rmem_context_exchange: data segment + notify segment. */
int UbDpRmemExchange(UbDpEnv& env)
{
    struct UbDpSegRdv mine = {};
    struct UbDpSegRdv peer = {};
    mine.exportInfo = env.segExport;
    mine.mem = env.segMem;
    int ret = UbDpExchange(env.phyid, env.sockFd, mine, &peer);
    if (ret != 0) {
        TEST_LOG_ERROR(env.phyid, "segment exchange failed ret=%d", ret);
        return -1;
    }
    env.peerSegImport = peer.exportInfo;
    env.peerSegMem = peer.mem;

    if (!env.hasNotify) {
        return 0;
    }
    mine = {};
    peer = {};
    mine.exportInfo = env.notifyExport;
    mine.mem.addr = env.notifyAddr;
    mine.mem.size = UB_DP_NOTIFY_SIZE;
    ret = UbDpExchange(env.phyid, env.sockFd, mine, &peer);
    if (ret != 0) {
        TEST_LOG_ERROR(env.phyid, "notify segment exchange failed ret=%d", ret);
        return -1;
    }
    env.peerNotifyImport = peer.exportInfo;
    env.peerNotifyMem = peer.mem;
    TEST_LOG_INFO(env.phyid, "test_rmem_context_exchange success");
    return 0;
}

/* test_ra_ctx_rmem_import: data segment + notify segment. */
int UbDpRmemImport(UbDpEnv& env)
{
    struct MrImportInfoT segImport = env.peerSegImport;
    int ret = RaCtxRmemImport(env.ctxHandle, &segImport, &env.segRmem);
    if (ret != 0) {
        TEST_LOG_ERROR(env.phyid, "RaCtxRmemImport failed ret=%d", ret);
        return -1;
    }
    if (!env.hasNotify) {
        return 0;
    }
    struct MrImportInfoT notifyImport = env.peerNotifyImport;
    ret = RaCtxRmemImport(env.ctxHandle, &notifyImport, &env.notifyRmem);
    if (ret != 0) {
        TEST_LOG_ERROR(env.phyid, "RaCtxRmemImport notify failed ret=%d", ret);
        return -1;
    }
    return 0;
}

/* test_sync_socket_shake_hand: 5-byte "sync" round trip. */
int UbDpShakeHand(UbDpEnv& env)
{
    const char sync[5] = {'s', 'y', 'n', 'c', '\0'};
    char peer[5] = {0};
    int ret = RaSocketSendAll(env.sockFd, sync, sizeof(sync));
    if (ret != 0) {
        TEST_LOG_ERROR(env.phyid, "shake hand send failed ret=%d", ret);
        return -1;
    }
    ret = RaSocketRecvAll(env.sockFd, peer, sizeof(peer));
    if (ret != 0) {
        TEST_LOG_ERROR(env.phyid, "shake hand recv failed ret=%d", ret);
        return -1;
    }
    if (memcmp(peer, sync, sizeof(sync)) != 0) {
        TEST_LOG_ERROR(env.phyid, "shake hand payload mismatch");
        return -1;
    }
    TEST_LOG_INFO(env.phyid, "test_sync_socket_shake_hand success");
    return 0;
}

/* ---------------------------------------------------------------- WQE build */

/* Fill the WQE template defaults of test_get_send_wr_data: opcode WRITE (0x00),
 * completion/fence/solicited/complete flags set, place order 0, no reduce. */
void UbDpWrTemplate(struct SendWrData& wr)
{
    wr.ub.opcode = RA_UB_OPC_WRITE;
    wr.ub.flags.bs.placeOrder = 0;
    wr.ub.flags.bs.compOrder = 1;
    wr.ub.flags.bs.fence = 1;
    wr.ub.flags.bs.solicitedEnable = 1;
    wr.ub.flags.bs.completeEnable = 1;
    wr.ub.flags.bs.inlineFlag = 0;
    wr.ub.reduceInfo.reduceEn = 0;
}

/* test_inline_notify_wr_build: WRITE with 8 inline bytes of value 1 into the
 * peer's notify segment. */
void UbDpNotifyWrBuild(struct SendWrData& wr, UbDpEnv& env)
{
    static uint64_t notifyValue = UB_DP_NOTIFY_VALUE; /* g_notify_value = 1 */
    wr.ub.opcode = RA_UB_OPC_WRITE;
    wr.ub.reduceInfo.reduceEn = 0;
    wr.remoteAddr = env.peerNotifyMem.addr;
    wr.rmemHandle = env.notifyRmem;
    wr.ub.flags.bs.inlineFlag = 1;
    wr.inlineSize = UB_DP_NOTIFY_SIZE;
    wr.inlineData = (uint8_t*)&notifyValue;
    wr.ub.remQpHandle = env.rqpHandle;
}

/* test_op_base_send_wr_data_build: the first wqe_num-1 WQEs write the upper
 * half of the peer's data segment from the local segment, the last WQE is the
 * inline notify write. Every non-NOP WQE carries the remote qp handle (the ra
 * layer's RaCtxBatchSendWrCheck requires it). */
void UbDpWrBuildOpBase(UbDpEnv& env)
{
    uint64_t packetSize = env.segMem.size / 2;
    for (unsigned int j = 0; j + 1 < UB_DP_WQE_NUM; j++) {
        struct SendWrData& wr = env.wr[j];
        UbDpWrTemplate(wr);
        wr.remoteAddr = env.peerSegMem.addr + packetSize;
        wr.rmemHandle = env.segRmem;
        wr.numSge = 1;
        env.sgeList[j].addr = env.segMem.addr;
        env.sgeList[j].len = (uint32_t)packetSize;
        env.sgeList[j].lmemHandle = env.segLmem;
        wr.sges = &env.sgeList[j];
        wr.ub.remQpHandle = env.rqpHandle;
    }
    UbDpNotifyWrBuild(env.wr[UB_DP_WQE_NUM - 1], env);
}

/* test_op_base_send_wr_data_build_write_with_notify: every WQE is a data WRITE
 * carrying the notify descriptor (RA_UB_OPC_WRITE_NOTIFY, see the file header
 * note on the opcode). */
void UbDpWrBuildWriteWithNotify(UbDpEnv& env)
{
    uint64_t packetSize = env.segMem.size / 2;
    for (unsigned int j = 0; j < UB_DP_WQE_NUM; j++) {
        struct SendWrData& wr = env.wr[j];
        UbDpWrTemplate(wr);
        wr.ub.opcode = RA_UB_OPC_WRITE_NOTIFY;
        wr.remoteAddr = env.peerSegMem.addr + packetSize;
        wr.rmemHandle = env.segRmem;
        wr.numSge = 1;
        env.sgeList[j].addr = env.segMem.addr;
        env.sgeList[j].len = (uint32_t)packetSize;
        env.sgeList[j].lmemHandle = env.segLmem;
        wr.sges = &env.sgeList[j];
        wr.ub.remQpHandle = env.rqpHandle;
        wr.ub.notifyInfo.notifyData = UB_DP_NOTIFY_VALUE;
        wr.ub.notifyInfo.notifyAddr = env.peerNotifyMem.addr;
        wr.ub.notifyInfo.notifyHandle = env.notifyRmem;
    }
}

/* test_graph_send_wr_data_build: WRITE + inline notify + NOP padding up to
 * sqebb_num. */
void UbDpWrBuildGraph(UbDpEnv& env)
{
    uint64_t packetSize = env.segMem.size / 2;
    for (unsigned int j = 0; j + 1 < UB_DP_WQE_NUM; j++) {
        struct SendWrData& wr = env.wr[j];
        UbDpWrTemplate(wr);
        wr.remoteAddr = env.peerSegMem.addr + packetSize;
        wr.rmemHandle = env.segRmem;
        wr.numSge = 1;
        env.sgeList[j].addr = env.segMem.addr;
        env.sgeList[j].len = (uint32_t)packetSize;
        env.sgeList[j].lmemHandle = env.segLmem;
        wr.sges = &env.sgeList[j];
        wr.ub.remQpHandle = env.rqpHandle;
    }
    UbDpNotifyWrBuild(env.wr[UB_DP_WQE_NUM - 1], env);
    for (unsigned int j = UB_DP_WQE_NUM; j < env.qpAttr.ub.extMode.sqebbNum && j < UB_DP_MAX_WQE; j++) {
        env.wr[j].ub.opcode = RA_UB_OPC_NOP;
    }
}

/* ------------------------------------------------------- post / doorbell ops */

/* test_ra_batch_send_wr_v1: batches of at most 32 WQEs, complete count checked
 * per batch; the posted count follows pi_type (op mode: wqe_num, pi mode:
 * sqebb_num). */
int UbDpBatchSend(UbDpEnv& env)
{
    unsigned int wqeNum = (env.qpAttr.ub.extMode.piType == 0) ? UB_DP_WQE_NUM : env.qpAttr.ub.extMode.sqebbNum;
    for (unsigned int j = 0; j < wqeNum; j += UB_DP_BATCH_MAX) {
        unsigned int sendNum = (wqeNum - j >= UB_DP_BATCH_MAX) ? UB_DP_BATCH_MAX : (wqeNum - j);
        unsigned int completeNum = 0;
        int ret = RaBatchSendWr(env.qpHandle, &env.wr[j], &env.opResp[j], sendNum, &completeNum);
        if (ret != 0 || sendNum != completeNum) {
            TEST_LOG_ERROR(
                env.phyid, "RaBatchSendWr failed qp wqe=%u ret=%d send=%u complete=%u", j, ret, sendNum, completeNum);
            return -1;
        }
    }
    TEST_LOG_INFO(env.phyid, "test_ra_batch_send_wr_v1 success (wqe_num=%u)", wqeNum);
    return 0;
}

/* test_rtUbDbSend: one doorbell carrying the posted pi value. */
int UbDpDbSend(UbDpEnv& env, struct SendWrResp& opRsp)
{
    rtUbDbInfo_t dbInfo = {};
    dbInfo.dbNum = 1;
    dbInfo.wrCqe = 0;
    dbInfo.info[0].functionId = opRsp.doorbellInfo.funcId;
    dbInfo.info[0].dieId = opRsp.doorbellInfo.dieId;
    dbInfo.info[0].jettyId = opRsp.doorbellInfo.jettyId;
    dbInfo.info[0].piValue = opRsp.doorbellInfo.piVal;
    if (rtUbDbSend(&dbInfo, env.stream) != RT_ERROR_NONE) {
        TEST_LOG_ERROR(env.phyid, "rtUbDbSend failed");
        return -1;
    }
    return 0;
}

/* test_offload_rtUbDbSend: pi value 1 (64B) / 2 (128B) from the dwqe size. */
int UbDpOffloadDbSend(UbDpEnv& env, struct SendWrResp& opRsp)
{
    rtUbDbInfo_t dbInfo = {};
    dbInfo.dbNum = 1;
    dbInfo.wrCqe = 0;
    dbInfo.info[0].functionId = opRsp.doorbellInfo.funcId;
    dbInfo.info[0].dieId = opRsp.doorbellInfo.dieId;
    dbInfo.info[0].jettyId = opRsp.doorbellInfo.jettyId;
    if (opRsp.doorbellInfo.dwqeSize == UB_DP_WQE_SIZE_64) {
        dbInfo.info[0].piValue = 1;
    } else if (opRsp.doorbellInfo.dwqeSize == UB_DP_WQE_SIZE_128) {
        dbInfo.info[0].piValue = 2;
    } else {
        TEST_LOG_ERROR(env.phyid, "invalid UB dwqeSize=%u", opRsp.doorbellInfo.dwqeSize);
        return -1;
    }
    if (rtUbDbSend(&dbInfo, env.stream) != RT_ERROR_NONE) {
        TEST_LOG_ERROR(env.phyid, "rtUbDbSend failed");
        return -1;
    }
    return 0;
}

/* test_rtUbDirectSend: DWQE direct send with the wqe blob from the post resp. */
int UbDpDirectSend(UbDpEnv& env, struct SendWrResp& opRsp)
{
    rtUbWqeInfo_t wqeInfo = {};
    wqeInfo.wrCqe = 0;
    wqeInfo.functionId = opRsp.doorbellInfo.funcId;
    wqeInfo.dieId = opRsp.doorbellInfo.dieId;
    wqeInfo.jettyId = opRsp.doorbellInfo.jettyId;
    wqeInfo.wqe = opRsp.doorbellInfo.dwqe;
    wqeInfo.wqePtrLen = opRsp.doorbellInfo.dwqeSize;
    wqeInfo.wqeSize = (opRsp.doorbellInfo.dwqeSize == UB_DP_WQE_SIZE_64) ? 0 : 1;
    if (rtUbDirectSend(&wqeInfo, env.stream) != RT_ERROR_NONE) {
        TEST_LOG_ERROR(env.phyid, "rtUbDirectSend failed");
        return -1;
    }
    return 0;
}

/* test_rtNotifyWait + test_rtStreamSynchronize.
 * rtNotifyWait is the one deprecated rt entry retained on purpose: its ACL
 * successor aclrtWaitAndResetNotify additionally RESETS the record cell, and
 * under this ST's synchronous WRITE_WITH_NOTIFY landing model (all records of
 * a batch land during RaBatchSendWr, before the first wait) the reset would
 * starve the per-WQE waits that follow in the db-send loops. The non-resetting
 * wait keeps the ported behaviour; the suppression is scoped to this call. */
int UbDpNotifyWait(UbDpEnv& env)
{
    if (!env.hasNotify || env.notify == nullptr) {
        TEST_LOG_ERROR(env.phyid, "notify wait without a notify unit");
        return -1;
    }
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    if (rtNotifyWait(env.notify, env.stream) != RT_ERROR_NONE) {
#pragma GCC diagnostic pop
        TEST_LOG_ERROR(env.phyid, "rtNotifyWait failed");
        return -1;
    }
    if (aclrtSynchronizeStream(env.stream) != 0) {
        TEST_LOG_ERROR(env.phyid, "aclrtSynchronizeStream failed");
        return -1;
    }
    return 0;
}

int UbDpStreamSync(UbDpEnv& env)
{
    if (aclrtSynchronizeStream(env.stream) != 0) {
        TEST_LOG_ERROR(env.phyid, "aclrtSynchronizeStream failed");
        return -1;
    }
    return 0;
}

/* test_ra_ctx_update_ci: the ci passed to the API is pi_val - 1. */
int UbDpUpdateCi(UbDpEnv& env, struct SendWrResp& opRsp)
{
    int ret = RaCtxUpdateCi(env.qpHandle, (uint16_t)(opRsp.doorbellInfo.piVal - 1));
    if (ret != 0) {
        TEST_LOG_ERROR(env.phyid, "RaCtxUpdateCi failed ret=%d", ret);
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------ teardown */

/* test_hccp_resource_deinit + test_rts_resource_deinit + test_close_hccp_by_tsd
 * in the original order; every step is null-guarded so a partially built chain
 * also cleans up. sharedBase marks a loop half that borrows the other half's
 * ctx/socket/net service: the per-half resources still tear down against the
 * shared ctx, but the once-only steps (socket deinit, token free, ctx deinit,
 * RaDeinit, device reset, net close) belong to the owning half. */
void UbDpEnd(UbDpEnv& env, bool sharedBase = false)
{
    int phyid = env.phyid;
    if (env.segRmem != nullptr) {
        (void)RaCtxRmemUnimport(env.ctxHandle, env.segRmem);
        env.segRmem = nullptr;
    }
    if (env.notifyRmem != nullptr) {
        (void)RaCtxRmemUnimport(env.ctxHandle, env.notifyRmem);
        env.notifyRmem = nullptr;
    }
    if (env.segLmem != nullptr) {
        (void)RaCtxLmemUnregister(env.ctxHandle, env.segLmem);
        env.segLmem = nullptr;
    }
    if (env.notifyLmem != nullptr) {
        (void)RaCtxLmemUnregister(env.ctxHandle, env.notifyLmem);
        env.notifyLmem = nullptr;
    }
    if (env.segReg.in.mem.addr != 0) {
        (void)aclrtFree((void*)(uintptr_t)env.segReg.in.mem.addr);
        env.segReg.in.mem.addr = 0;
    }
    if (env.tlvHandle != nullptr) {
        (void)HccpTlvCcuRequestVoid(phyid, env.tlvHandle, MSG_TYPE_CCU_UNINIT);
        (void)HccpTlvCcuDeinit(phyid, env.tlvHandle);
        env.tlvHandle = nullptr;
    }
    if (env.qpBound) {
        (void)RaCtxQpUnbind(env.qpHandle);
        env.qpBound = false;
    }
    if (env.rqpHandle != nullptr) {
        (void)RaCtxQpUnimport(env.ctxHandle, env.rqpHandle);
        env.rqpHandle = nullptr;
    }
    if (env.qpHandle != nullptr) {
        (void)RaCtxQpDestroy(env.qpHandle);
        env.qpHandle = nullptr;
    }
    if (env.cqHandle != nullptr) {
        (void)RaCtxCqDestroy(env.ctxHandle, env.cqHandle);
        env.cqHandle = nullptr;
    }
    if (env.chanHandle != nullptr) {
        (void)RaCtxChanDestroy(env.ctxHandle, env.chanHandle);
        env.chanHandle = nullptr;
    }
    if (env.sockFd != nullptr) {
        struct SocketCloseInfoT closeInfo = {};
        closeInfo.socketHandle = env.socketHandle;
        closeInfo.fdHandle = env.sockFd;
        (void)RaSocketBatchClose(&closeInfo, 1);
        env.sockFd = nullptr;
    }
    if (!sharedBase) {
        if (env.listening) {
            struct SocketListenInfoT listenInfo = {};
            listenInfo.socketHandle = env.socketHandle;
            listenInfo.port = 0;
            (void)RaSocketListenStop(&listenInfo, 1);
            env.listening = false;
        }
        if (env.socketHandle != nullptr) {
            (void)RaSocketDeinit(env.socketHandle);
            env.socketHandle = nullptr;
        }
        if (env.tokenHandle != nullptr) {
            (void)RaCtxTokenIdFree(env.ctxHandle, env.tokenHandle);
            env.tokenHandle = nullptr;
        }
        if (env.ctxHandle != nullptr) {
            (void)RaCtxDeinit(env.ctxHandle);
            env.ctxHandle = nullptr;
        }
        if (env.raInited) {
            (void)RaDeinit(&env.raCfg);
            env.raInited = false;
        }
    }
    if (env.notify != nullptr) {
        (void)aclrtDestroyNotify(env.notify);
        env.notify = nullptr;
    }
    if (env.stream != nullptr) {
        (void)aclrtDestroyStream(env.stream);
        env.stream = nullptr;
    }
    if (env.rtDeviceSet) {
        (void)aclrtResetDevice(phyid);
        env.rtDeviceSet = false;
    }
    if (env.netOpened) {
        (void)rtCloseNetService();
        env.netOpened = false;
    }
}

/* ------------------------------------------------------- chain compositions */

/* The two-machine data-plane chain of test_hccp_resource_init: base init,
 * optional CCU TLV init, sync socket, eid/tp exchange, chan+cq+qp, qp import
 * exchange + import + bind, segments + rmem exchange + import. */
int UbDpFullChain(
    UbDpEnv& env, int phyid, unsigned short port, const char* tag, enum JettyMode jettyMode, bool piType,
    unsigned int notifyNum)
{
    port = HccpStCasePort(port); /* direct-bind family: displace at entry (listener+connectors) */
    int ret = UbDpBegin(env, phyid, notifyNum);
    if (ret != 0) {
        return ret;
    }
    if (jettyMode == JETTY_MODE_CCU) {
        ret = HccpTlvCcuInit(phyid, &env.tlvHandle);
        if (ret != 0) {
            return ret;
        }
        ret = HccpTlvCcuRequestVoid(phyid, env.tlvHandle, MSG_TYPE_CCU_INIT);
        if (ret != 0) {
            return ret;
        }
    }
    ret = UbDpSocketSetup(env, port);
    if (ret != 0) {
        return ret;
    }
    ret = UbDpEidExchange(env);
    if (ret != 0) {
        return ret;
    }
    ret = UbDpTpList(env);
    if (ret != 0) {
        return ret;
    }
    ret = UbDpTpExchange(env);
    if (ret != 0) {
        return ret;
    }
    ret = UbDpChanCreate(env);
    if (ret != 0) {
        return ret;
    }
    ret = UbDpCqCreate(env, jettyMode == JETTY_MODE_CCU);
    if (ret != 0) {
        return ret;
    }
    ret = UbDpQpCreate(env, jettyMode, piType);
    if (ret != 0) {
        return ret;
    }
    ret = UbDpQpExchange(env);
    if (ret != 0) {
        return ret;
    }
    ret = UbDpQpImport(env);
    if (ret != 0) {
        return ret;
    }
    ret = UbDpQpBind(env);
    if (ret != 0) {
        return ret;
    }
    ret = UbDpSegMemInit(env);
    if (ret != 0) {
        return ret;
    }
    ret = UbDpSegRegister(env);
    if (ret != 0) {
        return ret;
    }
    ret = UbDpRmemExchange(env);
    if (ret != 0) {
        return ret;
    }
    ret = UbDpRmemImport(env);
    if (ret != 0) {
        return ret;
    }
    ret = UbDpShakeHand(env);
    if (ret != 0) {
        return ret;
    }
    TEST_LOG_INFO(phyid, "%s chain success", tag);
    return 0;
}

} // namespace

/* ------------------------------------------------------------------- cases */

/* Ported from hlt_hccp_opbase_dbsend_test_001: full resource chain, WQE build
 * (first wqe_num-1 WRITEs into the peer segment's upper half + inline notify
 * WRITE), then per rep: RaBatchSendWr, a doorbell per WQE, notify wait + stream
 * sync + ci update per qp, and a shake hand per rep. */
TEST_F(HccpSystemTest, ubDpOpbaseDbSend)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        UbDpEnv env;
        int ret = UbDpFullChain(
            env, phyid, UB_PORT_DP_DBSEND, "opbase_dbsend_001", JETTY_MODE_USER_CTL_NORMAL, false, UB_DP_QP_NUM);
        if (ret != 0) {
            goto err;
        }
        UbDpWrBuildOpBase(env);
        if (ret != 0) {
            goto err;
        }
        for (unsigned int k = 0; k < 1; k++) { /* rep_time = 1 */
            ret = UbDpBatchSend(env);
            if (ret != 0) {
                goto err;
            }
            for (unsigned int i = 0; i < UB_DP_QP_NUM; i++) {
                for (unsigned int j = 0; j < UB_DP_WQE_NUM; j++) {
                    ret = UbDpDbSend(env, env.opResp[j]);
                    if (ret != 0) {
                        goto err;
                    }
                }
                ret = UbDpNotifyWait(env);
                if (ret != 0) {
                    goto err;
                }
                ret = UbDpUpdateCi(env, env.opResp[UB_DP_WQE_NUM - 1]);
                if (ret != 0) {
                    goto err;
                }
            }
            ret = UbDpShakeHand(env);
            if (ret != 0) {
                goto err;
            }
        }
        TEST_LOG_INFO(phyid, "opbase_dbsend_test_001 passed");
        ret = 0;
    err:
        UbDpEnd(env);
        return ret;
    });
}

/* Ported from hlt_hccp_opbase_dbsend_test_002: write_with_notify WQE build (see
 * the file header opcode note), a doorbell per WQE with the notify wait inside
 * the WQE loop, ci update and shake hand per rep. */
TEST_F(HccpSystemTest, ubDpOpbaseDbSendNotify)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        UbDpEnv env;
        int ret = UbDpFullChain(
            env, phyid, UB_PORT_DP_DBSEND_NOTIFY, "opbase_dbsend_002", JETTY_MODE_USER_CTL_NORMAL, false, UB_DP_QP_NUM);
        if (ret != 0) {
            goto err;
        }
        UbDpWrBuildWriteWithNotify(env);
        for (unsigned int k = 0; k < 1; k++) {
            ret = UbDpBatchSend(env);
            if (ret != 0) {
                goto err;
            }
            for (unsigned int i = 0; i < UB_DP_QP_NUM; i++) {
                for (unsigned int j = 0; j < UB_DP_WQE_NUM; j++) {
                    ret = UbDpDbSend(env, env.opResp[j]);
                    if (ret != 0) {
                        goto err;
                    }
                    ret = UbDpNotifyWait(env);
                    if (ret != 0) {
                        goto err;
                    }
                }
                ret = UbDpUpdateCi(env, env.opResp[UB_DP_WQE_NUM - 1]);
                if (ret != 0) {
                    goto err;
                }
            }
            ret = UbDpShakeHand(env);
            if (ret != 0) {
                goto err;
            }
        }
        TEST_LOG_INFO(phyid, "opbase_dbsend_test_002 passed");
        ret = 0;
    err:
        UbDpEnd(env);
        return ret;
    });
}

/* Ported from hlt_hccp_graph_dbsend_test_001: graph WQE build (WRITE + notify +
 * NOP padding to sqebb_num, pi mode), a single batch post before the rep loop,
 * then per rep an offload doorbell per sqebb and the notify wait per qp. The
 * original posts no ci update and no shake hand in this variant. */
TEST_F(HccpSystemTest, ubDpGraphDbSend)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        UbDpEnv env;
        int ret = UbDpFullChain(
            env, phyid, UB_PORT_DP_GRAPH, "graph_dbsend_001", JETTY_MODE_USER_CTL_NORMAL, true, UB_DP_QP_NUM);
        if (ret != 0) {
            goto err;
        }
        UbDpWrBuildGraph(env);
        ret = UbDpBatchSend(env); /* posted once, before the rep loop */
        if (ret != 0) {
            goto err;
        }
        for (unsigned int k = 0; k < 1; k++) {
            for (unsigned int i = 0; i < UB_DP_QP_NUM; i++) {
                for (unsigned int j = 0; j < env.qpAttr.ub.extMode.sqebbNum; j++) {
                    ret = UbDpOffloadDbSend(env, env.opResp[j]);
                    if (ret != 0) {
                        goto err;
                    }
                }
                ret = UbDpNotifyWait(env);
                if (ret != 0) {
                    goto err;
                }
                ret = UbDpStreamSync(env);
                if (ret != 0) {
                    goto err;
                }
            }
        }
        TEST_LOG_INFO(phyid, "graph_dbsend_test_001 passed");
        ret = 0;
    err:
        UbDpEnd(env);
        return ret;
    });
}

/* Ported from hlt_hccp_opbase_dwqe_test_001: op_base WQE build with a DWQE
 * direct send per WQE (rtUbDirectSend carrying the wqe blob of the post resp),
 * notify wait + ci update per qp; the original runs no shake hand in this
 * variant. */
TEST_F(HccpSystemTest, ubDpOpbaseDwqe)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        UbDpEnv env;
        int ret = UbDpFullChain(
            env, phyid, UB_PORT_DP_DWQE, "opbase_dwqe_001", JETTY_MODE_USER_CTL_NORMAL, false, UB_DP_QP_NUM);
        if (ret != 0) {
            goto err;
        }
        UbDpWrBuildOpBase(env);
        if (ret != 0) {
            goto err;
        }
        for (unsigned int k = 0; k < 1; k++) {
            ret = UbDpBatchSend(env);
            if (ret != 0) {
                goto err;
            }
            for (unsigned int i = 0; i < UB_DP_QP_NUM; i++) {
                for (unsigned int j = 0; j < UB_DP_WQE_NUM; j++) {
                    ret = UbDpDirectSend(env, env.opResp[j]);
                    if (ret != 0) {
                        goto err;
                    }
                }
                ret = UbDpNotifyWait(env);
                if (ret != 0) {
                    goto err;
                }
                ret = UbDpUpdateCi(env, env.opResp[UB_DP_WQE_NUM - 1]);
                if (ret != 0) {
                    goto err;
                }
            }
        }
        TEST_LOG_INFO(phyid, "opbase_dwqe_test_001 passed");
        ret = 0;
    err:
        UbDpEnd(env);
        return ret;
    });
}

/* Ported from hlt_hccp_opbase_writevalue_test_001: despite the name, the
 * original entry's body is the same helper sequence as the dwqe entry (op_base
 * build + test_rtUbDirectSend + notify wait + ci update, no rtWriteValue call
 * and no shake hand); the port keeps that body one-to-one. */
TEST_F(HccpSystemTest, ubDpOpbaseWriteValue)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        UbDpEnv env;
        int ret = UbDpFullChain(
            env, phyid, UB_PORT_DP_WRITEVALUE, "opbase_writevalue_001", JETTY_MODE_USER_CTL_NORMAL, false,
            UB_DP_QP_NUM);
        if (ret != 0) {
            goto err;
        }
        UbDpWrBuildOpBase(env);
        if (ret != 0) {
            goto err;
        }
        for (unsigned int k = 0; k < 1; k++) {
            ret = UbDpBatchSend(env);
            if (ret != 0) {
                goto err;
            }
            for (unsigned int i = 0; i < UB_DP_QP_NUM; i++) {
                for (unsigned int j = 0; j < UB_DP_WQE_NUM; j++) {
                    ret = UbDpDirectSend(env, env.opResp[j]);
                    if (ret != 0) {
                        goto err;
                    }
                }
                ret = UbDpNotifyWait(env);
                if (ret != 0) {
                    goto err;
                }
                ret = UbDpUpdateCi(env, env.opResp[UB_DP_WQE_NUM - 1]);
                if (ret != 0) {
                    goto err;
                }
            }
        }
        TEST_LOG_INFO(phyid, "opbase_writevalue_test_001 passed");
        ret = 0;
    err:
        UbDpEnd(env);
        return ret;
    });
}

/* Ported from hlt_hccp_ccu_jetty_test_001: resource chain with the CCU jetty
 * mode (TLV CCU init after ctx init, cq depth 64 with CCU poll, qp create in
 * JETTY_MODE_CCU) and the sync socket shake hand; the original drives no data
 * ops in this entry. Full teardown despite the original's leaky deinit. */
TEST_F(HccpSystemTest, ubDpCcuJetty)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        UbDpEnv env;
        int ret = UbDpFullChain(env, phyid, UB_PORT_DP_CCU_JETTY, "ccu_jetty_001", JETTY_MODE_CCU, false, UB_DP_QP_NUM);
        if (ret != 0) {
            goto err;
        }
        TEST_LOG_INFO(phyid, "ccu_jetty_test_001 passed");
        ret = 0;
    err:
        UbDpEnd(env);
        return ret;
    });
}

/* Ported from hlt_hccp_ccu_jetty_test_002: CCU jetty variant without the notify
 * segment (no rts notify, no lnotify registration/exchange/import). The
 * original's deinit path (test_hccp_resource_deinit_batch) only calls RaDeinit;
 * the port runs the full teardown (file header note). */
TEST_F(HccpSystemTest, ubDpCcuJettyNoNotify)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        UbDpEnv env;
        int ret = UbDpFullChain(env, phyid, UB_PORT_DP_CCU_JETTY_002, "ccu_jetty_002", JETTY_MODE_CCU, false, 0);
        if (ret != 0) {
            goto err;
        }
        TEST_LOG_INFO(phyid, "ccu_jetty_test_002 passed");
        ret = 0;
    err:
        UbDpEnd(env);
        return ret;
    });
}

/* Ported from hlt_hccp_udma_write_value_002: full chain + op_base WQE build,
 * per rep RaBatchSendWr + doorbell per WQE + notify wait per qp + shake hand,
 * then the CrErrInfoList query inside two observation windows. The original
 * sleeps 8s twice; the windows are shortened to 1s (framework host timeout ~5s,
 * the sim has no async error source). */
TEST_F(HccpSystemTest, ubDpUdmaWriteValueErrInfo)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        UbDpEnv env;
        struct CrErrInfo infoList[1] = {};
        unsigned int infoNum = 0;
        int ret = UbDpFullChain(
            env, phyid, UB_PORT_DP_ERRINFO, "udma_write_value_002", JETTY_MODE_USER_CTL_NORMAL, false, UB_DP_QP_NUM);
        if (ret != 0) {
            goto err;
        }
        UbDpWrBuildOpBase(env);
        if (ret != 0) {
            goto err;
        }
        for (unsigned int k = 0; k < 1; k++) {
            ret = UbDpBatchSend(env);
            if (ret != 0) {
                goto err;
            }
            for (unsigned int i = 0; i < UB_DP_QP_NUM; i++) {
                for (unsigned int j = 0; j < UB_DP_WQE_NUM; j++) {
                    ret = UbDpDbSend(env, env.opResp[j]);
                    if (ret != 0) {
                        goto err;
                    }
                }
                ret = UbDpNotifyWait(env);
                if (ret != 0) {
                    goto err;
                }
            }
            ret = UbDpShakeHand(env);
            if (ret != 0) {
                goto err;
            }
        }
        sleep(1); /* observation window (original: 8s) */
        infoNum = 1;
        ret = RaCtxGetCrErrInfoList(env.ctxHandle, infoList, &infoNum);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaCtxGetCrErrInfoList failed ret=%d", ret);
            goto err;
        }
        if (infoNum != 0) {
            TEST_LOG_INFO(
                phyid, "RaCtxGetCrErrInfoList entry status=%u jetty_id=%u tv_sec=%ld tv_usec=%ld", infoList[0].status,
                infoList[0].jettyId, infoList[0].time.tv_sec, infoList[0].time.tv_usec);
        } else {
            TEST_LOG_INFO(phyid, "RaCtxGetCrErrInfoList null");
        }
        sleep(1); /* second window (original: 8s) */
        TEST_LOG_INFO(phyid, "udma_write_value_002 passed");
        ret = 0;
    err:
        UbDpEnd(env);
        return ret;
    });
}

/* ------------------------------------------------------------ loop entries */

/* Loop chain of test_hccp_resource_init_loop in one host child: shared base
 * init (server half), client-half stream+notify, self-connect socket pair,
 * tp list on both halves (direct copy, psns stay 0), cq+qp per half (no chan in
 * the loop chain), cross import + bind on both halves, segments on both halves.
 * The server half posts; the client half's notify is the wait target. */
struct UbDpLoopPair {
    UbDpEnv srv;
    UbDpEnv cli;
};

int UbDpLoopChain(UbDpLoopPair& lp, int phyid, unsigned short port, bool piType)
{
    port = HccpStCasePort(port); /* direct-bind family: displace at entry (listener+connectors) */
    int ret = UbDpBegin(lp.srv, phyid, UB_DP_QP_NUM);
    if (ret != 0) {
        return ret;
    }
    /* the client half shares the ctx/token/eid of the server half (one
     * RaCtxInit in the loop chain, both configs point at it) */
    lp.cli.phyid = phyid;
    lp.cli.isServer = false;
    lp.cli.hasNotify = true;
    lp.cli.ctxAttr = lp.srv.ctxAttr;
    lp.cli.ctxHandle = lp.srv.ctxHandle;
    lp.cli.tokenHandle = lp.srv.tokenHandle;
    ret = UbDpHalfRts(lp.cli, phyid);
    if (ret != 0) {
        return ret;
    }

    /* self-connect socket pair on one socket handle */
    uint32_t selfIp = UbDpLoopbackIp((uint32_t)phyid, HCCP_VM_IFACE_SOCKET);
    struct SocketInitInfoT sockInit = {};
    sockInit.rdevInfo.phyId = (unsigned int)phyid;
    sockInit.rdevInfo.family = AF_INET;
    sockInit.rdevInfo.localIp.addr.s_addr = selfIp;
    sockInit.scopeId = 4;
    ret = RaSocketInitV1(NETWORK_OFFLINE, sockInit, &lp.srv.socketHandle);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaSocketInitV1 failed ret=%d", ret);
        return -1;
    }
    const char* tag = "0";
    struct SocketListenInfoT listenInfo = {};
    listenInfo.socketHandle = lp.srv.socketHandle;
    listenInfo.port = port;
    ret = RaSocketListenStart(&listenInfo, 1);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaSocketListenStart failed ret=%d", ret);
        return -1;
    }
    lp.srv.listening = true;
    struct SocketWlistInfoT wlist = {};
    wlist.remoteIp.addr.s_addr = selfIp; /* the peer is the own ip */
    wlist.connLimit = 1;
    (void)snprintf(wlist.tag, sizeof(wlist.tag), "%s", tag);
    ret = RaSocketWhiteListAdd(lp.srv.socketHandle, &wlist, 1);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaSocketWhiteListAdd failed ret=%d", ret);
        return -1;
    }
    struct SocketConnectInfoT conn = {};
    conn.socketHandle = lp.srv.socketHandle;
    conn.remoteIp.addr.s_addr = selfIp;
    conn.port = port;
    (void)snprintf(conn.tag, sizeof(conn.tag), "%s", tag);
    ret = RaSocketBatchConnect(&conn, 1);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaSocketBatchConnect failed ret=%d", ret);
        return -1;
    }
    struct SocketInfoT cliSock = {};
    struct SocketInfoT srvSock = {};
    ret = UbDpWaitSocket(phyid, 1, lp.srv.socketHandle, selfIp, tag, &cliSock);
    if (ret != 0) {
        return -1;
    }
    ret = UbDpWaitSocket(phyid, 0, lp.srv.socketHandle, selfIp, tag, &srvSock);
    if (ret != 0) {
        return -1;
    }
    lp.cli.socketHandle = lp.srv.socketHandle;
    lp.cli.sockFd = cliSock.fdHandle;
    lp.srv.sockFd = srvSock.fdHandle;

    /* tp on both halves: both query with local == peer eid, then direct-copy
     * exchange (no psn step in the loop chain) */
    lp.srv.remoteEid = lp.srv.ctxAttr.ub.eid;
    ret = UbDpTpList(lp.srv);
    if (ret != 0) {
        return ret;
    }
    lp.cli.remoteEid = lp.cli.ctxAttr.ub.eid;
    ret = UbDpTpList(lp.cli);
    if (ret != 0) {
        return ret;
    }
    /* both halves query with the same (local == peer) eid pair, so the tp table
     * hands both the same tp; the exchange is the direct-copy of the loop chain
     * and the psns stay at 0 (no random step there) */
    lp.cli.localTp[0] = lp.srv.localTp[0];
    lp.cli.remoteTp[0] = lp.srv.localTp[0];
    lp.srv.remoteTp[0] = lp.cli.localTp[0];

    /* cq + qp per half (the loop chain creates no chan) */
    ret = UbDpCqCreate(lp.srv, false);
    if (ret != 0) {
        return ret;
    }
    ret = UbDpCqCreate(lp.cli, false);
    if (ret != 0) {
        return ret;
    }
    ret = UbDpQpCreate(lp.srv, JETTY_MODE_USER_CTL_NORMAL, piType);
    if (ret != 0) {
        return ret;
    }
    ret = UbDpQpCreate(lp.cli, JETTY_MODE_USER_CTL_NORMAL, piType);
    if (ret != 0) {
        return ret;
    }
    /* import exchange by direct copy, import + bind on both halves */
    lp.srv.rqpImport = lp.cli.lqpExport;
    lp.cli.rqpImport = lp.srv.lqpExport;
    ret = UbDpQpImport(lp.srv);
    if (ret != 0) {
        return ret;
    }
    ret = UbDpQpImport(lp.cli);
    if (ret != 0) {
        return ret;
    }
    ret = UbDpQpBind(lp.srv);
    if (ret != 0) {
        return ret;
    }
    ret = UbDpQpBind(lp.cli);
    if (ret != 0) {
        return ret;
    }

    /* segments on both halves, direct-copy rmem exchange, import each other's */
    ret = UbDpSegMemInit(lp.srv);
    if (ret != 0) {
        return ret;
    }
    ret = UbDpSegMemInit(lp.cli);
    if (ret != 0) {
        return ret;
    }
    ret = UbDpSegRegister(lp.srv);
    if (ret != 0) {
        return ret;
    }
    ret = UbDpSegRegister(lp.cli);
    if (ret != 0) {
        return ret;
    }
    lp.srv.peerSegImport = lp.cli.segExport;
    lp.srv.peerSegMem = lp.cli.segMem;
    lp.srv.peerNotifyImport = lp.cli.notifyExport;
    lp.srv.peerNotifyMem.addr = lp.cli.notifyAddr;
    lp.srv.peerNotifyMem.size = UB_DP_NOTIFY_SIZE;
    lp.cli.peerSegImport = lp.srv.segExport;
    lp.cli.peerSegMem = lp.srv.segMem;
    lp.cli.peerNotifyImport = lp.srv.notifyExport;
    lp.cli.peerNotifyMem.addr = lp.srv.notifyAddr;
    lp.cli.peerNotifyMem.size = UB_DP_NOTIFY_SIZE;
    ret = UbDpRmemImport(lp.srv);
    if (ret != 0) {
        return ret;
    }
    ret = UbDpRmemImport(lp.cli);
    if (ret != 0) {
        return ret;
    }
    TEST_LOG_INFO(phyid, "loop chain success");
    return 0;
}

/* Loop teardown mirroring test_hccp_resource_deinit_loop + rts deinit loop:
 * segments both halves -> qp both halves -> socket pair -> ctx/RaDeinit once ->
 * notifies/streams both halves -> device reset once. The client half's shared
 * handles (ctx/token/socket/net) are detached first so only the server half
 * tears them down. */
void UbDpLoopEnd(UbDpLoopPair& lp)
{
    /* the client half's fd closes first (the socket handle belongs to the
     * server half), then its per-half resources tear down against the shared
     * ctx; the once-only steps stay with the server half */
    if (lp.cli.sockFd != nullptr) {
        struct SocketCloseInfoT closeInfo = {};
        closeInfo.socketHandle = lp.srv.socketHandle;
        closeInfo.fdHandle = lp.cli.sockFd;
        (void)RaSocketBatchClose(&closeInfo, 1);
        lp.cli.sockFd = nullptr;
    }
    lp.cli.socketHandle = nullptr;
    lp.cli.listening = false;
    lp.cli.tokenHandle = nullptr;
    lp.cli.netOpened = false;
    lp.cli.raInited = false;
    lp.cli.rtDeviceSet = false;
    UbDpEnd(lp.cli, true); /* destroys its qp/cq/segments/notify/stream */
    UbDpEnd(lp.srv);
}

/* Ported from hlt_hccp_opbase_dbsend_loop_test_001: single-process dual-config
 * loop of the op_base dbsend flow - the server half posts, the client half's
 * notify is waited on (client stream), both streams sync, ci updates the
 * server qp. The loop entries run no socket shake hand (both ends are local). */
TEST_F(HccpSystemTest, ubDpDbSendLoop)
{
    RunTests({1}, [this](int phyid) -> int {
        UbDpLoopPair lp;
        int ret = UbDpLoopChain(lp, phyid, UB_PORT_DP_LOOP_DBSEND, false);
        if (ret != 0) {
            goto err;
        }
        UbDpWrBuildOpBase(lp.srv);
        if (ret != 0) {
            goto err;
        }
        for (unsigned int k = 0; k < 1; k++) {
            ret = UbDpBatchSend(lp.srv);
            if (ret != 0) {
                goto err;
            }
            for (unsigned int i = 0; i < UB_DP_QP_NUM; i++) {
                for (unsigned int j = 0; j < UB_DP_WQE_NUM; j++) {
                    ret = UbDpDbSend(lp.srv, lp.srv.opResp[j]);
                    if (ret != 0) {
                        goto err;
                    }
                }
                ret = UbDpNotifyWait(lp.cli); /* client half's notify + stream */
                if (ret != 0) {
                    goto err;
                }
                ret = UbDpStreamSync(lp.srv);
                if (ret != 0) {
                    goto err;
                }
                ret = UbDpUpdateCi(lp.srv, lp.srv.opResp[UB_DP_WQE_NUM - 1]);
                if (ret != 0) {
                    goto err;
                }
            }
        }
        TEST_LOG_INFO(phyid, "opbase_dbsend_loop_test_001 passed");
        ret = 0;
    err:
        UbDpLoopEnd(lp);
        return ret;
    });
}

/* Ported from hlt_hccp_opbase_dwqe_loop_test_001: the loop variant of the DWQE
 * direct-send flow (server half posts, client half's notify, both streams
 * sync, ci on the server qp). */
TEST_F(HccpSystemTest, ubDpDwqeLoop)
{
    RunTests({1}, [this](int phyid) -> int {
        UbDpLoopPair lp;
        int ret = UbDpLoopChain(lp, phyid, UB_PORT_DP_LOOP_DWQE, false);
        if (ret != 0) {
            goto err;
        }
        UbDpWrBuildOpBase(lp.srv);
        if (ret != 0) {
            goto err;
        }
        for (unsigned int k = 0; k < 1; k++) {
            ret = UbDpBatchSend(lp.srv);
            if (ret != 0) {
                goto err;
            }
            for (unsigned int i = 0; i < UB_DP_QP_NUM; i++) {
                for (unsigned int j = 0; j < UB_DP_WQE_NUM; j++) {
                    ret = UbDpDirectSend(lp.srv, lp.srv.opResp[j]);
                    if (ret != 0) {
                        goto err;
                    }
                }
                ret = UbDpNotifyWait(lp.cli);
                if (ret != 0) {
                    goto err;
                }
                ret = UbDpStreamSync(lp.srv);
                if (ret != 0) {
                    goto err;
                }
                ret = UbDpUpdateCi(lp.srv, lp.srv.opResp[UB_DP_WQE_NUM - 1]);
                if (ret != 0) {
                    goto err;
                }
            }
        }
        TEST_LOG_INFO(phyid, "opbase_dwqe_loop_test_001 passed");
        ret = 0;
    err:
        UbDpLoopEnd(lp);
        return ret;
    });
}

/* Ported from hlt_hccp_opbase_graph_loop_test_001: the loop variant of the
 * graph flow - pi mode WQE build, one batch post, offload doorbells over the
 * sqebb range, client-half notify wait, both streams sync, ci on the server qp
 * (the loop graph entry does update ci, unlike the two-machine graph entry). */
TEST_F(HccpSystemTest, ubDpGraphLoop)
{
    RunTests({1}, [this](int phyid) -> int {
        UbDpLoopPair lp;
        int ret = UbDpLoopChain(lp, phyid, UB_PORT_DP_LOOP_GRAPH, true);
        if (ret != 0) {
            goto err;
        }
        UbDpWrBuildGraph(lp.srv);
        ret = UbDpBatchSend(lp.srv);
        if (ret != 0) {
            goto err;
        }
        for (unsigned int k = 0; k < 1; k++) {
            for (unsigned int i = 0; i < UB_DP_QP_NUM; i++) {
                for (unsigned int j = 0; j < lp.srv.qpAttr.ub.extMode.sqebbNum; j++) {
                    ret = UbDpOffloadDbSend(lp.srv, lp.srv.opResp[j]);
                    if (ret != 0) {
                        goto err;
                    }
                }
                ret = UbDpNotifyWait(lp.cli);
                if (ret != 0) {
                    goto err;
                }
                ret = UbDpStreamSync(lp.srv);
                if (ret != 0) {
                    goto err;
                }
                ret = UbDpUpdateCi(lp.srv, lp.srv.opResp[lp.srv.qpAttr.ub.extMode.sqebbNum - 1]);
                if (ret != 0) {
                    goto err;
                }
            }
        }
        TEST_LOG_INFO(phyid, "opbase_graph_loop_test_001 passed");
        ret = 0;
    err:
        UbDpLoopEnd(lp);
        return ret;
    });
}
