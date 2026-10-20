/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * Batch 2 of the test-dev hlt/network/hccp_test migration: the 28 UB control plane
 * cases (t1 survey section 6.3) ported from hlt_hccp_ub_control_plane.c,
 * hlt_hccp_ub_test.c and hlt_hccp_ub_test_loop.c.
 *
 * Port conventions (t1 survey section 7):
 * - Each original hlt_* entry becomes TEST_F(HccpSystemTest, <name>) with a
 *   "Ported from hlt_xxx" comment right above it.
 * - The original gstool two-machine model (machine_type 0/1 = server/client, two
 *   processes) maps onto the framework's per-phyId host children: RunTests({1, 5})
 *   forks one child per phyId and both run the same ported logic, so the original
 *   sync-socket exchanges become live exchanges between the two children.
 * - Business config defaults of hlt_hccp_conf_init.c/hccp_temp.json are inlined as
 *   the UB_CTRL_* constants; no gstool/cJSON/json parsing is carried over.
 * - Logging goes through TEST_LOG_* and keeps the original step names so business
 *   log greps stay meaningful.
 * - Original defects are fixed, not replicated; each fix is flagged in a comment.
 * - Ports 15700-15727 are this file's rendezvous ports (other cases hold
 *   15666-15692 and 15950).
 *
 * Sim adaptation notes that apply to the whole file:
 * - test_ra_init_ub enables the async HDC session (enableHdcAsync=1) because the
 *   tp/eid batch async APIs need it. The shared CaseOfflineInit keeps the session
 *   off, so this file carries its own UbCtrlBegin/UbCtrlEnd helpers.
 * - test_open_hccp_by_tsd (TSD pull-up of the device process) maps to
 *   rtOpenNetService, which forks the per-device hccp_service.bin. Every case needs
 *   it in the sim, including the original peer-online entries that skipped it
 *   (udma_peer_002, get_eid_by_ip): the sim always runs the offline service model.
 * - The TpAttr/tp-list coverage relies on the urma stub tp table (stub/urma.c
 *   urma_get_tp_list/urma_get_tp_attr/urma_set_tp_attr): a query for an unseen
 *   (ctx, eids, flag, trans_mode) tuple creates the tp handle and its attribute
 *   shadow, which the set/get pair then carries - the production semantics.
 * - The device eid table (stub urma_get_eid_list) returns the per-device vm nic
 *   eids 127.<phyId>.<iface>, so the "match the local nic ip against the eid list"
 *   step of test_ra_get_dev_eid_info_list resolves a real eid, as in production.
 */

#include "cases_common.h"
#include "hccp_ccu_tlv_shim.h"

#include <arpa/inet.h>
#include <runtime/runtime/dev.h>
#include <runtime/runtime/event.h>
#include <runtime/runtime/stream.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

/* the async request APIs (RaGetTpInfoListAsync/RaSetTpAttrAsync/RaGetAsyncReqResult
 * ...) are declared in the async headers, which the hccp.h chain does not pull in */
#include "hccp_async.h"
#include "hccp_async_ctx.h"

/* rtGetDevResAddress lives in the rts_device.h family, which cannot be included
 * together with the runtime/runtime headers; the aclrt* entries come from the
 * real <acl/acl_rt.h> via cases_common.h, implemented by the runtime stub. */
extern "C" rtError_t rtGetDevResAddress(rtDevResInfo* const resInfo, rtDevResAddrInfo* const addrInfo);

namespace {

/* ==== business defaults inlined from hlt_hccp_conf_init.c ==== */
constexpr unsigned int UB_CTRL_TP_NUM = 1;            /* tp_config.tp_num */
constexpr unsigned int UB_CTRL_QP_NUM = 1;            /* qp_config.qp_num */
constexpr unsigned int UB_CTRL_NOTIFY_NUM = 1;        /* rts_config.notify_num (defaults to qp_num) */
constexpr unsigned int UB_CTRL_SEG_SIZE = 4096;       /* DEFAULT_MR_SIZE */
constexpr unsigned int UB_CTRL_CQ_DEPTH = 16384;      /* DEFAULT_CQ_DEPTH */
constexpr unsigned int UB_CTRL_SQ_DEPTH = 4096;       /* DEFAULT_QP_SQ_DEPTH */
constexpr unsigned int UB_CTRL_RQ_DEPTH = 256;        /* DEFAULT_QP_RQ_DEPTH */
constexpr unsigned int UB_CTRL_RNR_RETRY = 7;         /* DEFAULT_QP_RNR_RETRY */
constexpr unsigned int UB_CTRL_SQEBB_NUM = 64;        /* DEFAULT_QP_SQEBB_NUM */
constexpr unsigned int UB_CTRL_NOTIFY_SIZE = 8;       /* RT_NOTIFY_SIZE */
constexpr unsigned int UB_CTRL_SEG_INIT_SERVER = 0x1; /* SERVER_MR_INIT_VALUE */
constexpr unsigned int UB_CTRL_SEG_INIT_CLIENT = 0x2; /* CLIENT_MR_INIT_VALUE */
constexpr unsigned int UB_CTRL_SEG_ACCESS = (MEM_SEG_ACCESS_READ | MEM_SEG_ACCESS_WRITE);
constexpr unsigned long long UB_CTRL_PAGE_SIZE = 4096ULL;
constexpr unsigned int UB_CTRL_CONTEXT_LEN = 512U; /* CONTEXT_MAX_LEN */
/* the original polls events[10], but this repo's RaCtxGetAsyncEvents caps the
 * batch at ASYNC_EVENT_MAX_NUM (4); the port uses the repo limit */
constexpr unsigned int UB_CTRL_ASYNC_EVENT_NUM = 4U;
/* RA_RS_GET_IFNUM of the private inc/network/ra_rs_opcode.h: a known-good
 * interface opcode the business ra_host.c itself queries. The gstool op_type
 * parameter of control_plane_013 is never driven by the CI, so this port pins a
 * valid opcode instead. */
constexpr unsigned int UB_CTRL_IF_VERSION_OPCODE = 33;
/* tp attr bitmaps: tp_attr_test uses 4035 (0xFC3, no sip/dip/smac/dmac),
 * uboe_tp_attr_001 uses 0x1FFFF/0xFFFF (full, uboe allows the mac/ip fields) */
constexpr uint32_t UB_CTRL_TP_BITMAP_V2 = 4035;
constexpr uint32_t UB_CTRL_TP_BITMAP_UBOE_GET = 0x1FFFF;
constexpr uint32_t UB_CTRL_TP_BITMAP_UBOE_SET = 0xFFFF;

/* Ports 15700-15727, one per case (both children rendezvous on the same port). */
constexpr unsigned short UB_PORT_RA_INIT = 15700;
constexpr unsigned short UB_PORT_EID_NUM = 15701;
constexpr unsigned short UB_PORT_EID_LIST = 15702;
constexpr unsigned short UB_PORT_CTX_INIT = 15703;
constexpr unsigned short UB_PORT_CHAN_CREATE = 15704;
constexpr unsigned short UB_PORT_SOCKET_RES = 15705;
constexpr unsigned short UB_PORT_CQ_CREATE = 15706;
constexpr unsigned short UB_PORT_QP_CREATE = 15707;
constexpr unsigned short UB_PORT_QP_IMPORT = 15708;
constexpr unsigned short UB_PORT_QP_BIND = 15709;
constexpr unsigned short UB_PORT_LMEM_REG = 15710;
constexpr unsigned short UB_PORT_RMEM_IMPORT = 15711;
constexpr unsigned short UB_PORT_IF_VERSION = 15712;
constexpr unsigned short UB_PORT_UDMA_PEER_CTX = 15713;
constexpr unsigned short UB_PORT_UDMA_PEER_FULL = 15714;
constexpr unsigned short UB_PORT_UBOE_TP_ATTR = 15715;
constexpr unsigned short UB_PORT_QP_QUERY_BATCH = 15716;
constexpr unsigned short UB_PORT_QP_DESTROY_BATCH = 15717;
constexpr unsigned short UB_PORT_QP_BATCH_V2 = 15718;
constexpr unsigned short UB_PORT_GET_EID_BY_IP = 15719;
constexpr unsigned short UB_PORT_GET_AUX_INFO = 15720;
constexpr unsigned short UB_PORT_TP_ATTR = 15721;
constexpr unsigned short UB_PORT_CHAN_CQ_QP = 15722;
constexpr unsigned short UB_PORT_AIV_URMA = 15723;
constexpr unsigned short UB_PORT_ASYNC_EVENT = 15724;
constexpr unsigned short UB_PORT_JETTY_CONTEXT = 15725;
constexpr unsigned short UB_PORT_JETTY_CREATE = 15726;
constexpr unsigned short UB_PORT_JETTY_QUERY = 15727;

/* Per-child state of the ported control plane chain: the union of every
 * test_config sub-struct the 28 original entries touch, at the default sizes
 * (qp_num/tp_num/notify_num/mem_num are all 1). */
struct UbCtrlEnv {
    CaseEnv base; /* ctx/cq/qp/lmem/rmem/socket handles + generic cleanup */

    /* rts resources (test_rts_resource_init) */
    bool rtDeviceSet = false;
    aclrtStream stream = nullptr;
    aclrtNotify notify[UB_CTRL_NOTIFY_NUM] = {nullptr};
    uint32_t notifyId[UB_CTRL_NOTIFY_NUM] = {0};
    uint64_t notifyAddr[UB_CTRL_NOTIFY_NUM] = {0};

    /* eid info (test_ra_get_dev_eid_info_num/list) */
    unsigned int eidNum = 0;
    struct HccpDevEidInfo eidList[8] = {};
    struct CtxInitAttr ctxAttr = {};
    void* tokenHandle = nullptr;
    struct DevBaseAttr devAttr = {};

    /* chan/cq/qp (qp_num == 1) */
    void* chanHandle = nullptr;
    struct CqInfoT cqInfo = {};
    struct QpCreateAttr qpAttr = {};
    struct QpCreateInfo qpInfo = {};
    struct QpImportInfoT lqpExport = {}; /* local qp export info (after create) */
    struct QpImportInfoT rqpImport = {}; /* peer qp import info (after exchange) */
    bool qpBindDone = false;

    /* tp (test_tp_resource_init) */
    union HccpEid remoteEid = {};
    unsigned int localTpNum = 0;
    struct HccpTpInfo localTp[UB_CTRL_TP_NUM] = {};
    struct HccpTpInfo remoteTp[UB_CTRL_TP_NUM] = {};
    unsigned int localPsn[UB_CTRL_TP_NUM] = {0};
    unsigned int remotePsn[UB_CTRL_TP_NUM] = {0};

    /* segments (test_segment_memory_init + test_ra_ctx_lmem_register) */
    struct MrRegInfoT segReg = {}; /* data segment register info */
    void* segLmem = nullptr;
    struct MrImportInfoT segExport = {}; /* data segment export info for the peer */
    void* segRmem = nullptr;
    struct MrImportInfoT peerSegImport = {};
    struct HccpMemInfo peerMem = {};
    struct MrRegInfoT notifyReg = {}; /* notify segment (rts notify addr) */
    void* notifyLmem = nullptr;
    struct MrImportInfoT notifyExport = {};
    void* notifyRmem = nullptr;
    struct MrImportInfoT peerNotifyImport = {};
    struct HccpMemInfo peerNotifyMem = {};
    void* hostSegMem = nullptr; /* udma_peer_002: posix_memalign host segment */
    bool notifySegSkip = false; /* host-segment case: no rts, no notify segment */
};

/* Segment rendezvous payload of test_rmem_context_exchange: the lmem export info
 * and the raw mem info travel together for each segment. */
struct UbCtrlSegRdv {
    struct MrImportInfoT exportInfo;
    struct HccpMemInfo mem;
};

inline unsigned int UbCtrlSegInitValue(int phyid)
{
    /* machine_type server(0)/client(1) of the original maps to phyId 1/5 */
    return (phyid == 1) ? UB_CTRL_SEG_INIT_SERVER : UB_CTRL_SEG_INIT_CLIENT;
}

/* test_ra_get_async_req_result: poll RaGetAsyncReqResult until it stops returning
 * OTHERS_EAGAIN (the original allows 20s). */
int UbCtrlAsyncReqResult(int phyid, void* reqHandle)
{
    struct AsyncReqResult result = {};
    for (int i = 0; i < 2000; i++) {
        int ret = RaGetAsyncReqResult(reqHandle, &result);
        if (ret == OTHERS_EAGAIN) {
            usleep(10000);
            continue;
        }
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaGetAsyncReqResult failed ret=%d", ret);
            return ret;
        }
        if (result.reqResult != 0) {
            TEST_LOG_ERROR(
                phyid, "async request failed reqResult=%d opcode=%u", result.reqResult, result.interfaceOpcode);
            return -1;
        }
        return 0;
    }
    TEST_LOG_ERROR(phyid, "RaGetAsyncReqResult timeout after 20s");
    return -1;
}

/* test_open_hccp_by_tsd (+ optional test_rtSetDevice / test_rts_resource_init) +
 * test_ra_init_ub (+ optional test_socket_resource_init): opens the per-device
 * service, optionally the rts stream/notify pair, RaInit with the async HDC
 * session the original test_ra_init_ub enables, and optionally the sync
 * rendezvous socket pair that stands in for the original sync socket. */
int UbCtrlBegin(UbCtrlEnv& env, int phyid, unsigned short port, const char* tag, bool rts, bool rdv)
{
    env.base.cfg = {};
    env.base.cfg.phyId = (unsigned int)phyid;
    env.base.cfg.nicPosition = NETWORK_OFFLINE; /* business default nic_position=1 */
    env.base.cfg.hdcType = 18;
    env.base.cfg.enableHdcAsync = true; /* test_ra_init_ub sets enable_hdc_async=1 */

    env.base.sockInit = {};
    env.base.sockInit.rdevInfo.phyId = (unsigned int)phyid;
    env.base.sockInit.rdevInfo.family = AF_INET;
    struct in_addr nicIp;
    nicIp.s_addr = htonl(HccpVmNicIp((uint32_t)phyid, HCCP_VM_IFACE_SOCKET));
    env.base.sockInit.rdevInfo.localIp.addr = nicIp;
    env.base.sockInit.scopeId = 4;

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
        return 1;
    }
    env.base.netServiceOpened = true;

    if (rts) {
        /* test_rts_resource_init = test_rtSetDevice + test_rtStreamCreate +
         * test_rtNotifyCreate (via the modern aclrt entries); the notify create
         * also resolves the notify unit's device address, which the lnotify
         * registration step consumes */
        if (aclrtSetDevice(phyid) != 0) {
            TEST_LOG_ERROR(phyid, "aclrtSetDevice failed");
            return -1;
        }
        env.rtDeviceSet = true;
        if (aclrtCreateStream(&env.stream) != 0) {
            TEST_LOG_ERROR(phyid, "aclrtCreateStream failed");
            return -1;
        }
        for (unsigned int i = 0; i < UB_CTRL_NOTIFY_NUM; i++) {
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
            TEST_LOG_INFO(phyid, "aclrtCreateNotify %u success notify_id=%u", i, id);
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
            env.notify[i] = ntf;
            env.notifyId[i] = id;
            env.notifyAddr[i] = notifyAddr;
            TEST_LOG_INFO(phyid, "rtGetDevResAddress %u success notify_addr=0x%lx", i, notifyAddr);
        }
    }

    /* rtOpenNetService returns only after the device stack reports ready
     * (stub waits on deviceReady, 30s budget), so RaInit must succeed on
     * the first call: a failure is a real defect, not a startup race. */
    int ret = RaInit(&env.base.cfg);
    if (ret != 0) {
        (void)RaDeinit(&env.base.cfg); /* release any half-initialized RA state */
        TEST_LOG_ERROR(phyid, "RaInit failed ret=%d", ret);
        return ret;
    }

    if (rdv) {
        /* test_socket_resource_init: the two-machine sync socket of the original
         * chain - the framework rendezvous provides the same connected pair */
        ret = EstablishRendezvous(phyid, 1, env.base.sockInit, port, tag, true, &env.base.socketHandle, &env.base.sock);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "EstablishRendezvous failed ret=%d", ret);
            return ret;
        }
    }
    return 0;
}

/* test_ra_get_dev_eid_info_num */
int UbCtrlEidNum(int phyid, UbCtrlEnv& env)
{
    struct RaInfo info = {};
    info.mode = NETWORK_OFFLINE;
    info.phyId = (unsigned int)phyid;
    int ret = RaGetDevEidInfoNum(info, &env.eidNum);
    if (ret != 0 || env.eidNum == 0) {
        TEST_LOG_ERROR(phyid, "RaGetDevEidInfoNum failed ret=%d num=%u", ret, env.eidNum);
        return -1;
    }
    TEST_LOG_INFO(phyid, "RaGetDevEidInfoNum success num=%u", env.eidNum);
    return 0;
}

/* test_ra_get_dev_eid_info_list: query the eid list and resolve the local nic's
 * eid + eidIndex into ctxAttr (the original matches testconfig->local_ip). */
int UbCtrlEidList(int phyid, UbCtrlEnv& env)
{
    struct RaInfo info = {};
    info.mode = NETWORK_OFFLINE;
    info.phyId = (unsigned int)phyid;
    unsigned int num = env.eidNum;
    if (num == 0 || num > (sizeof(env.eidList) / sizeof(env.eidList[0]))) {
        num = sizeof(env.eidList) / sizeof(env.eidList[0]);
    }
    unsigned int got = num;
    int ret = RaGetDevEidInfoList(info, env.eidList, &got);
    if (ret != 0 || got != num) {
        TEST_LOG_ERROR(phyid, "RaGetDevEidInfoList failed ret=%d num=%u expect=%u", ret, got, num);
        return -1;
    }
    TEST_LOG_INFO(phyid, "RaGetDevEidInfoList success num=%u", got);

    struct in_addr nicIp;
    nicIp.s_addr = htonl(HccpVmNicIp((uint32_t)phyid, HCCP_VM_IFACE_SOCKET));
    struct in_addr localIp = HccpVmToLoopback(nicIp);
    unsigned int match = 0xFFFFFFFFu;
    for (unsigned int i = 0; i < got; i++) {
        TEST_LOG_INFO(
            phyid, "eid[%u] index=%u in4.addr=0x%x (local 0x%x)", i, env.eidList[i].eidIndex,
            env.eidList[i].eid.in4.addr, localIp.s_addr);
        if (env.eidList[i].eid.in4.addr == localIp.s_addr) {
            match = i;
            break;
        }
    }
    if (match == 0xFFFFFFFFu) {
        TEST_LOG_ERROR(phyid, "get eid_index fail, local_ip not exist");
        return -1;
    }
    env.ctxAttr.phyId = (unsigned int)phyid;
    env.ctxAttr.ub.eid = env.eidList[match].eid;
    env.ctxAttr.ub.eidIndex = env.eidList[match].eidIndex;
    TEST_LOG_INFO(
        phyid, "get_dev_eid_info_list success die_id=%u chip_id=%u func_id=%u", env.eidList[match].dieId,
        env.eidList[match].chipId, env.eidList[match].funcId);
    return 0;
}

/* test_ra_ctx_ini_atomic: RaCtxInit(offline) + RaCtxTokenIdAlloc */
int UbCtrlCtxInitAtomic(int phyid, UbCtrlEnv& env)
{
    struct CtxInitCfg cfg = {};
    cfg.mode = NETWORK_OFFLINE;
    struct CtxInitAttr attr = env.ctxAttr;
    int ret = RaCtxInit(&cfg, &attr, &env.base.ctxHandle);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaCtxInit failed ret=%d", ret);
        return -1;
    }
    TEST_LOG_INFO(phyid, "RaCtxInit success ctx_handle=%p", env.base.ctxHandle);

    struct HccpTokenId tokenInfo = {};
    ret = RaCtxTokenIdAlloc(env.base.ctxHandle, &tokenInfo, &env.tokenHandle);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaCtxTokenIdAlloc failed ret=%d", ret);
        return -1;
    }
    TEST_LOG_INFO(phyid, "RaCtxTokenIdAlloc success token_id_handle=%p", env.tokenHandle);
    return 0;
}

/* test_ra_get_dev_base_attr */
int UbCtrlDevBaseAttr(int phyid, UbCtrlEnv& env)
{
    int ret = RaGetDevBaseAttr(env.base.ctxHandle, &env.devAttr);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaGetDevBaseAttr failed ret=%d", ret);
        return -1;
    }
    TEST_LOG_INFO(
        phyid, "RaGetDevBaseAttr success sqMaxDepth=%u rqMaxDepth=%u dieId=%u", env.devAttr.sqMaxDepth,
        env.devAttr.rqMaxDepth, env.devAttr.ub.dieId);
    return 0;
}

/* test_ra_ctx_chan_create. The original passes an UNINITIALISED stack struct here
 * (pollCqCstm garbage); fixed per t1 section 7 by zeroing it, matching the
 * explicit pollCqCstm=0 of hlt_hccp_udma_write_value_001. */
int UbCtrlChanCreate(int phyid, UbCtrlEnv& env)
{
    struct ChanInfoT chanInfo = {};
    chanInfo.in.dataPlaneFlag.bs.pollCqCstm = 0;
    int ret = RaCtxChanCreate(env.base.ctxHandle, &chanInfo, &env.chanHandle);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaCtxChanCreate failed ret=%d", ret);
        return -1;
    }
    TEST_LOG_INFO(phyid, "RaCtxChanCreate success chan_handle=%p", env.chanHandle);
    return 0;
}

/* test_ra_ctx_cq_create: depth 16384, JFC_MODE_STARS_POLL (chosen because the qp
 * jetty mode is USER_CTL_NORMAL), ceqn 0, cqeFlag 1 - the business cq_info_t
 * defaults. cqInfo.in.chanHandle follows whatever the chain created. */
int UbCtrlCqCreate(int phyid, UbCtrlEnv& env)
{
    struct CqInfoT cqInfo = {};
    cqInfo.in.chanHandle = env.chanHandle;
    cqInfo.in.depth = UB_CTRL_CQ_DEPTH;
    cqInfo.in.ub.userCtx = 0;
    cqInfo.in.ub.mode = JFC_MODE_STARS_POLL;
    cqInfo.in.ub.ceqn = 0;
    cqInfo.in.ub.ccuExCfg.valid = false;
    cqInfo.in.ub.ccuExCfg.cqeFlag = 1;
    int ret = RaCtxCqCreate(env.base.ctxHandle, &cqInfo, &env.base.sendCq);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaCtxCqCreate failed ret=%d", ret);
        return -1;
    }
    env.cqInfo = cqInfo;
    TEST_LOG_INFO(
        phyid, "RaCtxCqCreate success cq_handle=%p jfc_id=%u cqe_size=%u", env.base.sendCq, cqInfo.out.id,
        cqInfo.out.cqeSize);
    return 0;
}

/* test_ra_ctx_qp_create: qp_num=1 with the business defaults (CONN_RM,
 * JETTY_MODE_USER_CTL_NORMAL, jettyId 0=auto, shareJfr, jfs lockFree, sqDepth
 * 4096, rqDepth 256, rnrRetry 7, sqCstm 0); scq/rcq/srq share the single cq;
 * the local qp export info is prepared afterwards. */
int UbCtrlQpCreate(int phyid, UbCtrlEnv& env)
{
    struct QpCreateAttr qpAttr = {};
    qpAttr.scqHandle = env.base.sendCq;
    qpAttr.rcqHandle = env.base.sendCq;
    qpAttr.srqHandle = env.base.sendCq;
    qpAttr.sqDepth = UB_CTRL_SQ_DEPTH;
    qpAttr.rqDepth = UB_CTRL_RQ_DEPTH;
    qpAttr.transportMode = CONN_RM;
    qpAttr.ub.mode = JETTY_MODE_USER_CTL_NORMAL;
    qpAttr.ub.jettyId = 0;
    qpAttr.ub.flag.bs.shareJfr = 1;
    qpAttr.ub.jfsFlag.value = 1; /* DEFAULT_QP_JETTY_FLAG applied to jfs_flag */
    qpAttr.ub.tokenValue = 0;
    qpAttr.ub.priority = 0;
    qpAttr.ub.rnrRetry = UB_CTRL_RNR_RETRY;
    qpAttr.ub.errTimeout = 0;
    qpAttr.ub.extMode.piType = false; /* op mode */
    qpAttr.ub.extMode.cstmFlag.bs.sqCstm = 0;
    qpAttr.ub.extMode.sqebbNum = UB_CTRL_SQEBB_NUM;
    int ret = RaCtxQpCreate(env.base.ctxHandle, &qpAttr, &env.qpInfo, &env.base.qpHandle);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaCtxQpCreate failed ret=%d", ret);
        return -1;
    }
    env.qpAttr = qpAttr;
    TEST_LOG_INFO(phyid, "RaCtxQpCreate success jetty_id=%u", env.qpInfo.ub.id);

    /* local qp export info (test_ra_ctx_qp_create tail) */
    env.lqpExport.in.key = env.qpInfo.key;
    env.lqpExport.in.ub.tokenValue = qpAttr.ub.tokenValue;
    env.lqpExport.in.ub.policy = JETTY_GRP_POLICY_RR;
    env.lqpExport.in.ub.type = TARGET_TYPE_JETTY;
    env.lqpExport.in.ub.flag.bs.tokenPolicy = MEM_SEG_TOKEN_PLAIN_TEXT;
    return 0;
}

/* test_qp_import_info_t_exchange */
int UbCtrlQpExchange(int phyid, UbCtrlEnv& env)
{
    int ret = ExchangePeer(env.base.sock.fdHandle, env.lqpExport, &env.rqpImport);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "test_qp_import_info_t_exchange failed ret=%d", ret);
        return -1;
    }
    TEST_LOG_INFO(phyid, "test_qp_import_info_t_exchange success");
    return 0;
}

/* test_ra_ctx_qp_import: EXP mode with the current tp info (tp handles and psns
 * stay 0 for the plain CTP chain without a tp step - production accepts a zero
 * tp cfg, and so does the sim import path) */
int UbCtrlQpImport(int phyid, UbCtrlEnv& env)
{
    struct QpImportInfoT importInfo = env.rqpImport;
    importInfo.in.ub.mode = JETTY_IMPORT_MODE_EXP;
    importInfo.in.ub.expImportCfg.tpHandle = env.localTp[0].tpHandle;
    importInfo.in.ub.expImportCfg.peerTpHandle = env.remoteTp[0].tpHandle;
    importInfo.in.ub.expImportCfg.txPsn = env.localPsn[0];
    importInfo.in.ub.expImportCfg.rxPsn = env.remotePsn[0];
    importInfo.in.ub.tpType = 1; /* URMA_CTP: business default tp_flag=0 */
    int ret = RaCtxQpImport(env.base.ctxHandle, &importInfo, &env.base.remQp);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaCtxQpImport failed ret=%d", ret);
        return -1;
    }
    TEST_LOG_INFO(phyid, "RaCtxQpImport success rem_qp=%p", env.base.remQp);
    return 0;
}

/* test_ra_ctx_qp_bind. The original wrapper returns early for CONN_RM (the
 * business default), so control_plane_010 never exercised its own API; per t1
 * section 7 this port calls RaCtxQpBind for real (bind works in RM mode). */
int UbCtrlQpBind(int phyid, UbCtrlEnv& env)
{
    int ret = RaCtxQpBind(env.base.qpHandle, env.base.remQp);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaCtxQpBind failed ret=%d", ret);
        return -1;
    }
    env.qpBindDone = true;
    TEST_LOG_INFO(phyid, "RaCtxQpBind success");
    return 0;
}

/* test_eid_info_exchange */
int UbCtrlEidExchange(int phyid, UbCtrlEnv& env)
{
    union HccpEid local = env.ctxAttr.ub.eid;
    int ret = ExchangePeer(env.base.sock.fdHandle, local, &env.remoteEid);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "test_eid_info_exchange failed ret=%d", ret);
        return -1;
    }
    TEST_LOG_INFO(phyid, "test_eid_info_exchange success remote in4.addr=0x%x", env.remoteEid.in4.addr);
    return 0;
}

/* test_ra_get_tp_info_list (async form, offline mode): flag per the business
 * tp_flag (CTP by default, uboe for the uboe case), transMode CONN_RM, local/peer
 * eid from the ctx attr and the eid exchange. */
int UbCtrlTpListAsync(int phyid, UbCtrlEnv& env, bool uboe)
{
    struct GetTpCfg tpCfg[UB_CTRL_TP_NUM] = {};
    for (unsigned int i = 0; i < UB_CTRL_TP_NUM; i++) {
        if (uboe) {
            tpCfg[i].flag.bs.uboe = 1;
        } else {
            tpCfg[i].flag.bs.ctp = 1; /* tp_flag=0: JETTY_IMPORT_MODE_EXP CTP */
        }
        tpCfg[i].transMode = CONN_RM;
        tpCfg[i].localEid = env.ctxAttr.ub.eid;
        tpCfg[i].peerEid = env.remoteEid;
    }
    unsigned int num = UB_CTRL_TP_NUM;
    void* reqHandle = nullptr;
    int ret = RaGetTpInfoListAsync(env.base.ctxHandle, tpCfg, env.localTp, &num, &reqHandle);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaGetTpInfoListAsync failed ret=%d", ret);
        return -1;
    }
    ret = UbCtrlAsyncReqResult(phyid, reqHandle);
    if (ret != 0 || num == 0) {
        TEST_LOG_ERROR(phyid, "get tp_info_list failed ret=%d num=%u", ret, num);
        return -1;
    }
    env.localTpNum = num;
    TEST_LOG_INFO(
        phyid, "get tp_info_list success num=%u tp_handle=0x%llx", num, (unsigned long long)env.localTp[0].tpHandle);
    return 0;
}

/* test_tp_info_exchange: tp num + tp list + random psn */
int UbCtrlTpExchange(int phyid, UbCtrlEnv& env)
{
    unsigned int peerNum = 0;
    int ret = ExchangePeer(env.base.sock.fdHandle, env.localTpNum, &peerNum);
    if (ret != 0 || peerNum == 0 || peerNum > UB_CTRL_TP_NUM) {
        TEST_LOG_ERROR(phyid, "tp num exchange failed ret=%d peer_num=%u", ret, peerNum);
        return -1;
    }
    ret = ExchangePeer(env.base.sock.fdHandle, env.localTp, &env.remoteTp);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "tp list exchange failed ret=%d", ret);
        return -1;
    }
    for (unsigned int i = 0; i < env.localTpNum; i++) {
        env.localPsn[i] = (unsigned int)(random() & 0xFFFFu);
    }
    ret = ExchangePeer(env.base.sock.fdHandle, env.localPsn, &env.remotePsn);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "psn exchange failed ret=%d", ret);
        return -1;
    }
    TEST_LOG_INFO(
        phyid, "test_tp_info_exchange success local_tp=0x%llx remote_tp=0x%llx",
        (unsigned long long)env.localTp[0].tpHandle, (unsigned long long)env.remoteTp[0].tpHandle);
    return 0;
}

/* test_ra_get_tp_attr_uboe / test_get_tp_attr_async: async get with the case's
 * bitmap; the log keeps the original field dump shape. */
int UbCtrlTpAttrGetAsync(int phyid, UbCtrlEnv& env, uint32_t bitmap)
{
    struct TpAttr attr = {};
    uint32_t attrBitmap = bitmap;
    void* reqHandle = nullptr;
    int ret = RaGetTpAttrAsync(env.base.ctxHandle, env.localTp[0].tpHandle, &attrBitmap, &attr, &reqHandle);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaGetTpAttrAsync failed ret=%d", ret);
        return -1;
    }
    ret = UbCtrlAsyncReqResult(phyid, reqHandle);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "get tp_attr failed ret=%d bitmap=0x%x", ret, attrBitmap);
        return -1;
    }
    TEST_LOG_INFO(
        phyid,
        "get TpAttr success sizeof(TpAttr):%zu attr_bitmap:0x%x retry_times_init:%u at:%u vlan_id:%u "
        "vlan_en:%u dscp:%u at_times:%u sl:%u ttl:%u",
        sizeof(attr), attrBitmap, (unsigned)attr.retryTimesInit, (unsigned)attr.at, (unsigned)attr.vlanId,
        (unsigned)attr.vlanEn, (unsigned)attr.dscp, (unsigned)attr.atTimes, (unsigned)attr.sl, (unsigned)attr.ttl);
    return 0;
}

/* test_ra_set_tp_attr_uboe / test_set_tp_attr_async: async set with the original
 * fixed values (sip/dip bytes only in the uboe variant, which allows them). */
int UbCtrlTpAttrSetAsync(int phyid, UbCtrlEnv& env, uint32_t bitmap, bool withIp)
{
    struct TpAttr attr = {};
    attr.retryTimesInit = 4;
    attr.at = 3;
    attr.vlanId = 13;
    attr.vlanEn = 1;
    attr.dscp = 7;
    attr.atTimes = 6;
    attr.sl = 1;
    attr.ttl = 4;
    attr.ackUdpSrcport = 1;
    attr.dataUdpSrcport = 1;
    attr.udpSrcportRange = 2;
    attr.sprayEn = 1;
    if (withIp) {
        /* the uboe variant fills the v4-mapped sip/dip from the local/remote nics */
        struct in_addr myNic;
        myNic.s_addr = htonl(HccpVmNicIp((uint32_t)phyid, HCCP_VM_IFACE_SOCKET));
        struct in_addr localIp = HccpVmToLoopback(myNic);
        uint32_t ipv4Local = ntohl(localIp.s_addr);
        attr.sip[12] = (uint8_t)((ipv4Local >> 24) & 0xFF);
        attr.sip[13] = (uint8_t)((ipv4Local >> 16) & 0xFF);
        attr.sip[14] = (uint8_t)((ipv4Local >> 8) & 0xFF);
        attr.sip[15] = (uint8_t)(ipv4Local & 0xFF);
        uint32_t ipv4Remote = ntohl(env.remoteEid.in4.addr);
        attr.dip[12] = (uint8_t)((ipv4Remote >> 24) & 0xFF);
        attr.dip[13] = (uint8_t)((ipv4Remote >> 16) & 0xFF);
        attr.dip[14] = (uint8_t)((ipv4Remote >> 8) & 0xFF);
        attr.dip[15] = (uint8_t)(ipv4Remote & 0xFF);
    }
    void* reqHandle = nullptr;
    int ret = RaSetTpAttrAsync(env.base.ctxHandle, env.localTp[0].tpHandle, bitmap, &attr, &reqHandle);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaSetTpAttrAsync failed ret=%d", ret);
        return -1;
    }
    ret = UbCtrlAsyncReqResult(phyid, reqHandle);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "set tp_attr failed ret=%d bitmap=0x%x", ret, bitmap);
        return -1;
    }
    TEST_LOG_INFO(phyid, "set TpAttr success attr_bitmap:0x%x", bitmap);
    return 0;
}

/* test_segment_memory_init: aclrtMalloc + aclrtMemset(init value by machine role) */
int UbCtrlSegInit(int phyid, UbCtrlEnv& env)
{
    void* devMem = nullptr;
    if (aclrtMalloc(&devMem, UB_CTRL_SEG_SIZE, ACL_MEM_MALLOC_HUGE_FIRST) != 0) {
        TEST_LOG_ERROR(phyid, "aclrtMalloc data failed");
        return -1;
    }
    if (aclrtMemset(devMem, UB_CTRL_SEG_SIZE, UbCtrlSegInitValue(phyid), UB_CTRL_SEG_SIZE) != 0) {
        TEST_LOG_ERROR(phyid, "aclrtMemset data failed");
        return -1;
    }
    env.base.rtBuf[0] = devMem; /* CaseCleanup aclrtFrees it */
    env.segReg.in.mem.addr = (uint64_t)(uintptr_t)devMem;
    env.segReg.in.mem.size = UB_CTRL_SEG_SIZE;
    TEST_LOG_INFO(phyid, "aclrtMalloc & aclrtMemset success dev_mem=%p", devMem);
    return 0;
}

/* test_segment_memory_calloc_init: 4k-aligned host memory variant of udma_peer_002 */
int UbCtrlSegInitHost(int phyid, UbCtrlEnv& env)
{
    void* hostMem = nullptr;
    if (posix_memalign(&hostMem, UB_CTRL_PAGE_SIZE, UB_CTRL_SEG_SIZE) != 0 || hostMem == nullptr) {
        TEST_LOG_ERROR(phyid, "posix_memalign failed");
        return -1;
    }
    memset(hostMem, UbCtrlSegInitValue(phyid), UB_CTRL_SEG_SIZE);
    env.hostSegMem = hostMem;
    env.segReg.in.mem.addr = (uint64_t)(uintptr_t)hostMem;
    env.segReg.in.mem.size = UB_CTRL_SEG_SIZE;
    TEST_LOG_INFO(phyid, "host segment alloc success dev_mem=%p", hostMem);
    return 0;
}

/* test_ra_ctx_lmem_register = test_ra_ctx_lmem_register_common (data segment;
 * token_mode=1: tokenIdValid + the ctx token handle) +
 * test_ra_ctx_lnotify_register_common (notify segment, page-aligned rts notify
 * addr, token handle passed but tokenIdValid stays 0 like the original). */
int UbCtrlLmemRegister(int phyid, UbCtrlEnv& env)
{
    struct MrRegInfoT segReg = env.segReg;
    segReg.in.ub.flags.bs.access = UB_CTRL_SEG_ACCESS;
    segReg.in.ub.flags.bs.nonPin = 1;
    segReg.in.ub.flags.bs.tokenIdValid = 1;
    segReg.in.ub.tokenIdHandle = env.tokenHandle;
    int ret = RaCtxLmemRegister(env.base.ctxHandle, &segReg, &env.segLmem);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "lmem RaCtxLmemRegister failed ret=%d", ret);
        return -1;
    }
    env.segReg = segReg;
    env.base.sendLmem = env.segLmem; /* CaseCleanup unregisters it */
    TEST_LOG_INFO(phyid, "lmem RaCtxLmemRegister success");

    /* export info for the peer (lmem_register_common tail) */
    env.segExport.in.key = segReg.out.key;
    env.segExport.in.ub.flags.bs.cacheable = segReg.in.ub.flags.bs.cacheable;
    env.segExport.in.ub.flags.bs.access = segReg.in.ub.flags.bs.access;
    env.segExport.in.ub.tokenValue = segReg.in.ub.tokenValue;

    if (env.notifySegSkip) {
        /* udma_peer_002 never initialises the rts resources, so its notify
         * segment is inert; the port skips the notify leg instead of exchanging
         * a zeroed entry (t1 section 7 fix). */
        return 0;
    }

    struct MrRegInfoT notifyReg = {};
    notifyReg.in.ub.flags.bs.tokenPolicy = 0;
    notifyReg.in.ub.flags.bs.cacheable = 0;
    notifyReg.in.ub.flags.bs.dsva = 0;
    notifyReg.in.ub.flags.bs.access = UB_CTRL_SEG_ACCESS;
    notifyReg.in.ub.flags.bs.nonPin = 1;
    notifyReg.in.ub.flags.bs.tokenIdValid = 0;
    notifyReg.in.ub.tokenValue = 0;
    notifyReg.in.mem.addr = env.notifyAddr[0];
    notifyReg.in.mem.size = UB_CTRL_NOTIFY_SIZE;
    uint64_t pageBase = notifyReg.in.mem.addr & ~(UB_CTRL_PAGE_SIZE - 1);
    uint64_t offset = notifyReg.in.mem.addr - pageBase;
    notifyReg.in.mem.addr = pageBase;
    notifyReg.in.mem.size += offset;
    notifyReg.in.ub.tokenIdHandle = env.tokenHandle; /* token_mode != 0 passes the handle */
    ret = RaCtxLmemRegister(env.base.ctxHandle, &notifyReg, &env.notifyLmem);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "lnotify RaCtxLmemRegister failed ret=%d", ret);
        return -1;
    }
    env.notifyReg = notifyReg;
    TEST_LOG_INFO(
        phyid, "lnotify RaCtxLmemRegister success addr=0x%llx size=%llu", (unsigned long long)notifyReg.in.mem.addr,
        (unsigned long long)notifyReg.in.mem.size);
    env.notifyExport.in.key = notifyReg.out.key;
    env.notifyExport.in.ub.flags.bs.cacheable = notifyReg.in.ub.flags.bs.cacheable;
    env.notifyExport.in.ub.flags.bs.access = notifyReg.in.ub.flags.bs.access;
    env.notifyExport.in.ub.tokenValue = notifyReg.in.ub.tokenValue;
    return 0;
}

/* test_rmem_context_exchange: data segment + notify segment */
int UbCtrlRmemExchange(int phyid, UbCtrlEnv& env)
{
    struct UbCtrlSegRdv mine = {};
    struct UbCtrlSegRdv peer = {};
    mine.exportInfo = env.segExport;
    mine.mem = env.segReg.in.mem;
    int ret = ExchangePeer(env.base.sock.fdHandle, mine, &peer);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "segment exchange failed ret=%d", ret);
        return -1;
    }
    env.peerSegImport = peer.exportInfo;
    env.peerMem = peer.mem;

    if (env.notifySegSkip) {
        TEST_LOG_INFO(phyid, "test_rmem_context_exchange success (data segment only)");
        return 0;
    }
    mine = {};
    peer = {};
    mine.exportInfo = env.notifyExport;
    mine.mem = env.notifyReg.in.mem;
    ret = ExchangePeer(env.base.sock.fdHandle, mine, &peer);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "notify segment exchange failed ret=%d", ret);
        return -1;
    }
    env.peerNotifyImport = peer.exportInfo;
    env.peerNotifyMem = peer.mem;
    TEST_LOG_INFO(phyid, "test_rmem_context_exchange success");
    return 0;
}

/* test_ra_ctx_rmem_import: data segment + notify segment */
int UbCtrlRmemImport(int phyid, UbCtrlEnv& env)
{
    struct MrImportInfoT segImport = env.peerSegImport;
    int ret = RaCtxRmemImport(env.base.ctxHandle, &segImport, &env.segRmem);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaCtxRmemImport failed ret=%d", ret);
        return -1;
    }
    env.base.rmem = env.segRmem; /* CaseCleanup unimports it */
    TEST_LOG_INFO(phyid, "RaCtxRmemImport success rmem=%p", env.segRmem);

    if (env.notifySegSkip) {
        return 0;
    }
    struct MrImportInfoT notifyImport = env.peerNotifyImport;
    ret = RaCtxRmemImport(env.base.ctxHandle, &notifyImport, &env.notifyRmem);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaCtxRmemImport notify failed ret=%d", ret);
        return -1;
    }
    TEST_LOG_INFO(phyid, "RaCtxRmemImport notify success rmem=%p", env.notifyRmem);
    return 0;
}

/* test_qp_query_batch (handle-based: the API takes no ctx, so the same handle can
 * be queried regardless of which context created it) */
int UbCtrlQpQueryBatch(int phyid, void* qpHandle)
{
    void* qpHandles[1] = {qpHandle};
    struct JettyAttr attr[1] = {};
    unsigned int num = 1;
    int ret = RaCtxQpQueryBatch(qpHandles, attr, &num);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaCtxQpQueryBatch failed ret=%d num=%u", ret, num);
        return -1;
    }
    TEST_LOG_INFO(
        phyid, "RaCtxQpQueryBatch success num=%u mask=%u rx_threshold=%u state=%u", num, attr[0].mask,
        attr[0].rxThreshold, (unsigned)attr[0].state);
    return 0;
}

/* test_qp_destroy_batch_async: destroys the qp handles; afterwards the handle is
 * gone, so the caller must clear env.base.qpHandle (the original nulls the
 * lqp_context handles the same way). */
int UbCtrlQpDestroyBatchAsync(int phyid, UbCtrlEnv& env, void* qpHandle)
{
    void* qpHandles[1] = {qpHandle};
    unsigned int num = 1;
    void* reqHandle = nullptr;
    int ret = RaCtxQpDestroyBatchAsync(env.base.ctxHandle, qpHandles, &num, &reqHandle);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaCtxQpDestroyBatchAsync failed ret=%d", ret);
        return -1;
    }
    ret = UbCtrlAsyncReqResult(phyid, reqHandle);
    if (ret != 0 || num == 0) {
        TEST_LOG_ERROR(phyid, "qp destroy batch failed ret=%d num=%u", ret, num);
        return -1;
    }
    TEST_LOG_INFO(phyid, "RaCtxQpDestroyBatchAsync success num=%u", num);
    return 0;
}

/* test_sync_socket_shake_hand: 5-byte "sync" round trip (the original sends
 * sizeof("sync")=5 including the NUL) */
int UbCtrlShakeHand(int phyid, UbCtrlEnv& env)
{
    char sendMsg[5] = "sync";
    char recvMsg[5] = {0};
    int ret = RaSocketSendAll(env.base.sock.fdHandle, sendMsg, sizeof(sendMsg));
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "shake hand send failed ret=%d", ret);
        return -1;
    }
    ret = RaSocketRecvAll(env.base.sock.fdHandle, recvMsg, sizeof(recvMsg));
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "shake hand recv failed ret=%d", ret);
        return -1;
    }
    TEST_LOG_INFO(phyid, "test_sync_socket_shake_hand success");
    return 0;
}

/* The end blocks of the original entries, in the original order (rmem unimport ->
 * lmem unregister -> segment free -> qp chain -> cq -> chan -> token/ctx -> notify/
 * stream/device -> socket -> RaDeinit -> net service). Every step is null-guarded
 * so a partially built chain also cleans up; handles stored in base are nulled as
 * they are consumed so CaseCleanup only handles socket/RaDeinit/aclrtFree/close. */
void UbCtrlEnd(HccpSystemTest* test, UbCtrlEnv& env, int phyid)
{
    if (env.base.ctxHandle != nullptr) {
        if (env.notifyRmem != nullptr) {
            (void)RaCtxRmemUnimport(env.base.ctxHandle, env.notifyRmem);
            env.notifyRmem = nullptr;
        }
        if (env.base.rmem != nullptr) {
            (void)RaCtxRmemUnimport(env.base.ctxHandle, env.base.rmem);
            env.base.rmem = nullptr;
        }
        if (env.notifyLmem != nullptr) {
            (void)RaCtxLmemUnregister(env.base.ctxHandle, env.notifyLmem);
            env.notifyLmem = nullptr;
        }
        if (env.base.sendLmem != nullptr) {
            (void)RaCtxLmemUnregister(env.base.ctxHandle, env.base.sendLmem);
            env.base.sendLmem = nullptr;
        }
        if (env.qpBindDone && env.base.qpHandle != nullptr) {
            (void)RaCtxQpUnbind(env.base.qpHandle);
            env.qpBindDone = false;
        }
        if (env.base.remQp != nullptr) {
            (void)RaCtxQpUnimport(env.base.ctxHandle, env.base.remQp);
            env.base.remQp = nullptr;
        }
        if (env.base.qpHandle != nullptr) {
            (void)RaCtxQpDestroy(env.base.qpHandle);
            env.base.qpHandle = nullptr;
        }
        if (env.base.sendCq != nullptr) {
            (void)RaCtxCqDestroy(env.base.ctxHandle, env.base.sendCq);
            env.base.sendCq = nullptr;
        }
        if (env.chanHandle != nullptr) {
            (void)RaCtxChanDestroy(env.base.ctxHandle, env.chanHandle);
            env.chanHandle = nullptr;
        }
        if (env.tokenHandle != nullptr) {
            (void)RaCtxTokenIdFree(env.base.ctxHandle, env.tokenHandle);
            env.tokenHandle = nullptr;
        }
        (void)RaCtxDeinit(env.base.ctxHandle);
        env.base.ctxHandle = nullptr;
    }
    if (env.hostSegMem != nullptr) {
        free(env.hostSegMem);
        env.hostSegMem = nullptr;
    }
    for (unsigned int i = 0; i < UB_CTRL_NOTIFY_NUM; i++) {
        if (env.notify[i] != nullptr) {
            (void)aclrtDestroyNotify(env.notify[i]);
            env.notify[i] = nullptr;
        }
    }
    if (env.stream != nullptr) {
        (void)aclrtDestroyStream(env.stream);
        env.stream = nullptr;
    }
    if (env.rtDeviceSet) {
        (void)aclrtResetDevice(phyid);
        env.rtDeviceSet = false;
    }
    test->CaseCleanup(env.base); /* socket deinit + aclrtFree buffers + RaDeinit + close */
}

} // namespace

/* Ported from hlt_hccp_control_plane_test_001: rts resource init + RaInit/RaDeinit
 * (open tsd -> rts -> ra_init; end: ra_deinit + rts deinit + tsd close). */
TEST_F(HccpSystemTest, ubCtrlRaInitOffline)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        UbCtrlEnv env;
        int ret = UbCtrlBegin(env, phyid, UB_PORT_RA_INIT, "UB CTRL 001", true, false);
        if (ret != 0) {
            UbCtrlEnd(this, env, phyid);
            return ret;
        }
        TEST_LOG_INFO(phyid, "control_plane_001 passed");
        UbCtrlEnd(this, env, phyid);
        return 0;
    });
}

/* Ported from hlt_hccp_control_plane_test_002: 001 + RaGetDevEidInfoNum. */
TEST_F(HccpSystemTest, ubCtrlGetDevEidInfoNum)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        UbCtrlEnv env;
        int ret = UbCtrlBegin(env, phyid, UB_PORT_EID_NUM, "UB CTRL 002", true, false);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidNum(phyid, env);
        if (ret != 0) {
            goto err;
        }
        TEST_LOG_INFO(phyid, "control_plane_002 passed");
    err:
        UbCtrlEnd(this, env, phyid);
        return ret;
    });
}

/* Ported from hlt_hccp_control_plane_test_003: 002 + RaGetDevEidInfoList with the
 * local-ip match resolving the ctx eid. */
TEST_F(HccpSystemTest, ubCtrlGetDevEidInfoList)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        UbCtrlEnv env;
        int ret = UbCtrlBegin(env, phyid, UB_PORT_EID_LIST, "UB CTRL 003", true, false);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidNum(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidList(phyid, env);
        if (ret != 0) {
            goto err;
        }
        TEST_LOG_INFO(phyid, "control_plane_003 passed");
    err:
        UbCtrlEnd(this, env, phyid);
        return ret;
    });
}

/* Ported from hlt_hccp_control_plane_test_004: 003 + RaCtxInit + RaCtxTokenIdAlloc
 * (test_ra_ctx_ini_atomic). */
TEST_F(HccpSystemTest, ubCtrlCtxInitToken)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        UbCtrlEnv env;
        int ret = UbCtrlBegin(env, phyid, UB_PORT_CTX_INIT, "UB CTRL 004", true, false);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidNum(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidList(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlCtxInitAtomic(phyid, env);
        if (ret != 0) {
            goto err;
        }
        TEST_LOG_INFO(phyid, "control_plane_004 passed");
    err:
        UbCtrlEnd(this, env, phyid);
        return ret;
    });
}

/* Ported from hlt_hccp_control_plane_test_005: 004 + RaGetDevBaseAttr + sync
 * socket + RaCtxChanCreate + shake hand. */
TEST_F(HccpSystemTest, ubCtrlChanCreate)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        UbCtrlEnv env;
        int ret = UbCtrlBegin(env, phyid, UB_PORT_CHAN_CREATE, "UB CTRL 005", true, true);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidNum(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidList(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlCtxInitAtomic(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlDevBaseAttr(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlChanCreate(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlShakeHand(phyid, env);
        if (ret != 0) {
            goto err;
        }
        TEST_LOG_INFO(phyid, "control_plane_005 passed");
    err:
        UbCtrlEnd(this, env, phyid);
        return ret;
    });
}

/* Ported from hlt_hccp_control_plane_test_006 ("remove RaGetDevBaseAttr test" -
 * the comment is stale, the entry keeps dev_base_attr and only drops the chan):
 * 004 + dev base attr + sync socket + shake hand. */
TEST_F(HccpSystemTest, ubCtrlSocketResource)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        UbCtrlEnv env;
        int ret = UbCtrlBegin(env, phyid, UB_PORT_SOCKET_RES, "UB CTRL 006", true, true);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidNum(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidList(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlCtxInitAtomic(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlDevBaseAttr(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlShakeHand(phyid, env);
        if (ret != 0) {
            goto err;
        }
        TEST_LOG_INFO(phyid, "control_plane_006 passed");
    err:
        UbCtrlEnd(this, env, phyid);
        return ret;
    });
}

/* Ported from hlt_hccp_control_plane_test_007: 006 + RaCtxCqCreate (no chan in
 * this entry: cq_config.chan_handle stays 0) + shake hand. */
TEST_F(HccpSystemTest, ubCtrlCqCreate)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        UbCtrlEnv env;
        int ret = UbCtrlBegin(env, phyid, UB_PORT_CQ_CREATE, "UB CTRL 007", true, true);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidNum(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidList(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlCtxInitAtomic(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlDevBaseAttr(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlCqCreate(phyid, env); /* chanHandle == nullptr here */
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlShakeHand(phyid, env);
        if (ret != 0) {
            goto err;
        }
        TEST_LOG_INFO(phyid, "control_plane_007 passed");
    err:
        UbCtrlEnd(this, env, phyid);
        return ret;
    });
}

/* Ported from hlt_hccp_control_plane_test_008: 007 + RaCtxQpCreate + shake hand. */
TEST_F(HccpSystemTest, ubCtrlQpCreate)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        UbCtrlEnv env;
        int ret = UbCtrlBegin(env, phyid, UB_PORT_QP_CREATE, "UB CTRL 008", true, true);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidNum(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidList(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlCtxInitAtomic(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlDevBaseAttr(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlCqCreate(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlQpCreate(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlShakeHand(phyid, env);
        if (ret != 0) {
            goto err;
        }
        TEST_LOG_INFO(phyid, "control_plane_008 passed");
    err:
        UbCtrlEnd(this, env, phyid);
        return ret;
    });
}

/* Ported from hlt_hccp_control_plane_test_009: 008 + qp import info exchange +
 * RaCtxQpImport + shake hand (the original's bind step skips for CONN_RM). */
TEST_F(HccpSystemTest, ubCtrlQpImport)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        UbCtrlEnv env;
        int ret = UbCtrlBegin(env, phyid, UB_PORT_QP_IMPORT, "UB CTRL 009", true, true);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidNum(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidList(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlCtxInitAtomic(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlDevBaseAttr(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlCqCreate(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlQpCreate(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlQpExchange(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlQpImport(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlShakeHand(phyid, env);
        if (ret != 0) {
            goto err;
        }
        TEST_LOG_INFO(phyid, "control_plane_009 passed");
    err:
        UbCtrlEnd(this, env, phyid);
        return ret;
    });
}

/* Ported from hlt_hccp_control_plane_test_010 ("add RaCtxQpBind test"): 009 with
 * the bind actually invoked - the original wrapper skips RaCtxQpBind for CONN_RM,
 * so the entry never exercised its own API (fixed per t1 section 7). */
TEST_F(HccpSystemTest, ubCtrlQpBind)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        UbCtrlEnv env;
        int ret = UbCtrlBegin(env, phyid, UB_PORT_QP_BIND, "UB CTRL 010", true, true);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidNum(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidList(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlCtxInitAtomic(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlDevBaseAttr(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlCqCreate(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlQpCreate(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlQpExchange(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlQpImport(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlQpBind(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlShakeHand(phyid, env);
        if (ret != 0) {
            goto err;
        }
        TEST_LOG_INFO(phyid, "control_plane_010 passed");
    err:
        UbCtrlEnd(this, env, phyid);
        return ret;
    });
}

/* Ported from hlt_hccp_control_plane_test_011: 010 + segment memory init +
 * RaCtxLmemRegister (data + notify segments) + shake hand. */
TEST_F(HccpSystemTest, ubCtrlLmemRegister)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        UbCtrlEnv env;
        int ret = UbCtrlBegin(env, phyid, UB_PORT_LMEM_REG, "UB CTRL 011", true, true);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidNum(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidList(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlCtxInitAtomic(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlDevBaseAttr(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlCqCreate(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlQpCreate(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlQpExchange(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlQpImport(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlSegInit(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlLmemRegister(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlShakeHand(phyid, env);
        if (ret != 0) {
            goto err;
        }
        TEST_LOG_INFO(phyid, "control_plane_011 passed");
    err:
        UbCtrlEnd(this, env, phyid);
        return ret;
    });
}

/* Ported from hlt_hccp_control_plane_test_012: 011 + rmem context exchange +
 * RaCtxRmemImport + shake hand - the full control plane chain. */
TEST_F(HccpSystemTest, ubCtrlRmemImport)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        UbCtrlEnv env;
        int ret = UbCtrlBegin(env, phyid, UB_PORT_RMEM_IMPORT, "UB CTRL 012", true, true);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidNum(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidList(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlCtxInitAtomic(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlDevBaseAttr(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlCqCreate(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlQpCreate(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlQpExchange(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlQpImport(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlSegInit(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlLmemRegister(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlRmemExchange(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlRmemImport(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlShakeHand(phyid, env);
        if (ret != 0) {
            goto err;
        }
        TEST_LOG_INFO(phyid, "control_plane_012 passed");
    err:
        UbCtrlEnd(this, env, phyid);
        return ret;
    });
}

/* Ported from hlt_hccp_control_plane_test_013: rts + RaInit + RaGetInterfaceVersion.
 * The gstool op_type parameter is never driven by the CI; this port pins
 * RA_RS_GET_IFNUM (33), an interface opcode the business ra_host.c itself queries.
 * The entry never creates a ctx or a socket. */
TEST_F(HccpSystemTest, ubCtrlGetInterfaceVersion)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        UbCtrlEnv env;
        struct timeval start = {};
        struct timeval end = {};
        unsigned int interfaceVersion = 0;
        float costMs = 0.0f;
        int ret = UbCtrlBegin(env, phyid, UB_PORT_IF_VERSION, "UB CTRL 013", true, false);
        if (ret != 0) {
            goto err;
        }
        (void)gettimeofday(&start, nullptr);
        ret = RaGetInterfaceVersion((unsigned int)phyid, UB_CTRL_IF_VERSION_OPCODE, &interfaceVersion);
        (void)gettimeofday(&end, nullptr);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaGetInterfaceVersion failed ret=%d", ret);
            goto err;
        }
        costMs = (float)(end.tv_sec - start.tv_sec) * 1000.0f + (float)(end.tv_usec - start.tv_usec) / 1000.0f;
        TEST_LOG_INFO(phyid, "RaGetInterfaceVersion success version=%u time=%.3fms", interfaceVersion, costMs);
    err:
        UbCtrlEnd(this, env, phyid);
        return ret;
    });
}

/* Ported from hlt_hccp_udma_peer_001: service open + RaInit + eid num/list +
 * plain RaCtxInit (no token alloc, no rts, no socket) + RaCtxDeinit. */
TEST_F(HccpSystemTest, ubCtrlUdmaPeerCtxInit)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        UbCtrlEnv env;
        struct CtxInitCfg cfg = {};
        struct CtxInitAttr attr = {};
        int ret = UbCtrlBegin(env, phyid, UB_PORT_UDMA_PEER_CTX, "UB CTRL PEER1", false, false);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidNum(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidList(phyid, env);
        if (ret != 0) {
            goto err;
        }
        cfg.mode = NETWORK_OFFLINE;
        attr = env.ctxAttr;
        ret = RaCtxInit(&cfg, &attr, &env.base.ctxHandle);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaCtxInit failed ret=%d", ret);
            goto err;
        }
        TEST_LOG_INFO(phyid, "RaCtxInit success ctx_handle=%p", env.base.ctxHandle);
        TEST_LOG_INFO(phyid, "udma_peer_001 passed");
    err:
        UbCtrlEnd(this, env, phyid);
        return ret;
    });
}

/* Ported from hlt_hccp_udma_peer_002: the full host-memory chain - no rts
 * resources, posix_memalign data segment, lmem register without the notify leg
 * (the sim still opens the net service because every offline RaInit needs the
 * per-device service process; the original ran this entry peer-online). */
TEST_F(HccpSystemTest, ubCtrlUdmaPeerFullChain)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        UbCtrlEnv env;
        env.notifySegSkip = true;
        int ret = UbCtrlBegin(env, phyid, UB_PORT_UDMA_PEER_FULL, "UB CTRL PEER2", false, true);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidNum(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidList(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlCtxInitAtomic(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlDevBaseAttr(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlChanCreate(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlCqCreate(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlQpCreate(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlQpExchange(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlQpImport(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlSegInitHost(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlLmemRegister(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlRmemExchange(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlRmemImport(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlShakeHand(phyid, env);
        if (ret != 0) {
            goto err;
        }
        TEST_LOG_INFO(phyid, "udma_peer_002 passed");
    err:
        UbCtrlEnd(this, env, phyid);
        return ret;
    });
}

/* Ported from hlt_hccp_uboe_tp_attr_001: the full chain (with chan) plus the uboe
 * tp leg - tp list with flag.uboe, TpAttr get(0x1FFFF)/set(0xFFFF, with
 * sip/dip)/get again - before the qp exchange/import. */
TEST_F(HccpSystemTest, ubCtrlUboeTpAttr)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        UbCtrlEnv env;
        int ret = UbCtrlBegin(env, phyid, UB_PORT_UBOE_TP_ATTR, "UB CTRL UBOE", true, true);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidNum(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidList(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlCtxInitAtomic(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlDevBaseAttr(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlChanCreate(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlCqCreate(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlQpCreate(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidExchange(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlTpListAsync(phyid, env, true); /* uboe flag */
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlTpAttrGetAsync(phyid, env, UB_CTRL_TP_BITMAP_UBOE_GET);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlTpAttrSetAsync(phyid, env, UB_CTRL_TP_BITMAP_UBOE_SET, true);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlTpAttrGetAsync(phyid, env, UB_CTRL_TP_BITMAP_UBOE_GET);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlQpExchange(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlQpImport(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlSegInit(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlLmemRegister(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlRmemExchange(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlRmemImport(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlShakeHand(phyid, env);
        if (ret != 0) {
            goto err;
        }
        TEST_LOG_INFO(phyid, "uboe_tp_attr_001 passed");
    err:
        UbCtrlEnd(this, env, phyid);
        return ret;
    });
}

/* Ported from hlt_hccp_qp_batch_test: full chain (socket + tp resource + qp
 * resource + segments) + shake hand + RaCtxQpQueryBatch +
 * RaCtxQpDestroyBatchAsync. */
TEST_F(HccpSystemTest, ubCtrlQpQueryBatch)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        UbCtrlEnv env;
        int ret = UbCtrlBegin(env, phyid, UB_PORT_QP_QUERY_BATCH, "UB CTRL QBATCH", true, true);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidNum(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidList(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlCtxInitAtomic(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlDevBaseAttr(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidExchange(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlTpListAsync(phyid, env, false);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlTpExchange(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlChanCreate(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlCqCreate(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlQpCreate(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlQpExchange(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlQpImport(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlSegInit(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlLmemRegister(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlRmemExchange(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlRmemImport(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlShakeHand(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlQpQueryBatch(phyid, env.base.qpHandle);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlQpDestroyBatchAsync(phyid, env, env.base.qpHandle);
        if (ret != 0) {
            goto err;
        }
        env.base.qpHandle = nullptr; /* batch destroy consumed it */
        TEST_LOG_INFO(phyid, "qp_batch_test passed");
    err:
        UbCtrlEnd(this, env, phyid);
        return ret;
    });
}

/* Ported from hlt_hccp_qp_batch_test_001: full chain + shake hand +
 * RaCtxQpDestroyBatchAsync only. */
TEST_F(HccpSystemTest, ubCtrlQpDestroyBatchAsync)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        UbCtrlEnv env;
        int ret = UbCtrlBegin(env, phyid, UB_PORT_QP_DESTROY_BATCH, "UB CTRL QDSTR", true, true);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidNum(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidList(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlCtxInitAtomic(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlDevBaseAttr(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidExchange(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlTpListAsync(phyid, env, false);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlTpExchange(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlChanCreate(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlCqCreate(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlQpCreate(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlQpExchange(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlQpImport(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlSegInit(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlLmemRegister(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlRmemExchange(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlRmemImport(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlShakeHand(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlQpDestroyBatchAsync(phyid, env, env.base.qpHandle);
        if (ret != 0) {
            goto err;
        }
        env.base.qpHandle = nullptr; /* batch destroy consumed it */
        TEST_LOG_INFO(phyid, "qp_batch_test_001 passed");
    err:
        UbCtrlEnd(this, env, phyid);
        return ret;
    });
}

/* Ported from hlt_hccp_qp_batch_test_v2: the full chain + shake hand (no batch
 * operation - the entry exists to hold the chain-only regression). */
TEST_F(HccpSystemTest, ubCtrlQpBatchV2)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        UbCtrlEnv env;
        int ret = UbCtrlBegin(env, phyid, UB_PORT_QP_BATCH_V2, "UB CTRL QBV2", true, true);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidNum(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidList(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlCtxInitAtomic(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlDevBaseAttr(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidExchange(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlTpListAsync(phyid, env, false);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlTpExchange(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlChanCreate(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlCqCreate(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlQpCreate(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlQpExchange(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlQpImport(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlSegInit(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlLmemRegister(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlRmemExchange(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlRmemImport(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlShakeHand(phyid, env);
        if (ret != 0) {
            goto err;
        }
        TEST_LOG_INFO(phyid, "qp_batch_test_v2 passed");
    err:
        UbCtrlEnd(this, env, phyid);
        return ret;
    });
}

/* Ported from hlt_hccp_get_eid_by_ip_test_001: RaInit + ctx init (full wrapper
 * with dev base attr) + RaGetEidByIp over the local and the peer nic ip. The
 * original takes server_ip/client_ip from the peer machine's json; the port uses
 * the two children's own vm nic addresses. */
TEST_F(HccpSystemTest, ubCtrlGetEidByIp)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        UbCtrlEnv env;
        struct IpInfo ip[2] = {};
        union HccpEid eid[2] = {};
        unsigned int ipNum = 2;
        int peerPhyId = (phyid == 1) ? 5 : 1;
        struct in_addr myNic;
        struct in_addr peerNic;
        int ret = UbCtrlBegin(env, phyid, UB_PORT_GET_EID_BY_IP, "UB CTRL EIDIP", true, false);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidNum(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidList(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlCtxInitAtomic(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlDevBaseAttr(phyid, env);
        if (ret != 0) {
            goto err;
        }

        myNic.s_addr = htonl(HccpVmNicIp((uint32_t)phyid, HCCP_VM_IFACE_SOCKET));
        peerNic.s_addr = htonl(HccpVmNicIp((uint32_t)peerPhyId, HCCP_VM_IFACE_SOCKET));
        ip[0].family = AF_INET;
        ip[0].ip.addr = HccpVmToLoopback(myNic);
        ip[1].family = AF_INET;
        ip[1].ip.addr = HccpVmToLoopback(peerNic);
        ret = RaGetEidByIp(env.base.ctxHandle, ip, eid, &ipNum);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaGetEidByIp failed ret=%d", ret);
            goto err;
        }
        /* Per-EID dump is debug-only noise: default silent, HCCP_VM_DUMP_ELEM=1 enables. */
        if (getenv("HCCP_VM_DUMP_ELEM") != nullptr) {
            for (unsigned int i = 0; i < ipNum; i++) {
                TEST_LOG_INFO(phyid, "RaGetEidByIp eid(%u) in4.addr=0x%x", i, eid[i].in4.addr);
            }
        }
        TEST_LOG_INFO(phyid, "get_eid_by_ip_test_001 passed");
    err:
        UbCtrlEnd(this, env, phyid);
        return ret;
    });
}

/* Ported from hlt_hccp_get_aux_info: ctx init + RaCtxGetAuxInfo for a CQE
 * (status/sR) and an AE (eventType) decode. The CI drives the parameters; this
 * port pins documented-valid values (cqe status must not be 0/3/7/12, ae event
 * must be one of 0/1/4/11/13). */
TEST_F(HccpSystemTest, ubCtrlGetAuxInfo)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        UbCtrlEnv env;
        struct HccpAuxInfoIn in = {};
        struct HccpAuxInfoOut out = {};
        int ret = UbCtrlBegin(env, phyid, UB_PORT_GET_AUX_INFO, "UB CTRL AUX", true, false);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidNum(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidList(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlCtxInitAtomic(phyid, env);
        if (ret != 0) {
            goto err;
        }

        memset(&out, 1, sizeof(out));
        memset(&in, 1, sizeof(in)); /* the original memsets both with 1 first */
        in.type = AUX_INFO_IN_TYPE_CQE;
        in.cqe.sR = 0;
        in.cqe.status = 1; /* valid poll-cqe status (not 0/3/7/12) */
        ret = RaCtxGetAuxInfo(env.base.ctxHandle, &in, &out);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaCtxGetAuxInfo CQE failed ret=%d", ret);
            goto err;
        }
        /* Per-element aux dump is debug-only noise: silent unless HCCP_VM_DUMP_ELEM=1. */
        if (getenv("HCCP_VM_DUMP_ELEM") != nullptr) {
            for (unsigned int i = 0; i < out.auxInfoNum; i++) {
                TEST_LOG_INFO(phyid, "cqe aux[%u] type=%u value=%u", i, out.auxInfoType[i], out.auxInfoValue[i]);
            }
        }

        memset(&out, 1, sizeof(out));
        memset(&in, 1, sizeof(in));
        in.type = AUX_INFO_IN_TYPE_AE;
        in.ae.eventType = 1; /* valid ae event type (0/1/4/11/13) */
        ret = RaCtxGetAuxInfo(env.base.ctxHandle, &in, &out);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaCtxGetAuxInfo AE failed ret=%d", ret);
            goto err;
        }
        if (getenv("HCCP_VM_DUMP_ELEM") != nullptr) {
            for (unsigned int i = 0; i < out.auxInfoNum; i++) {
                TEST_LOG_INFO(phyid, "ae aux[%u] type=%u value=%u", i, out.auxInfoType[i], out.auxInfoValue[i]);
            }
        }
        TEST_LOG_INFO(phyid, "get_aux_info passed");
    err:
        UbCtrlEnd(this, env, phyid);
        return ret;
    });
}

/* Ported from hlt_hccp_tp_attr_test: the full v2 chain - tp resource init (eid
 * exchange + tp list + tp exchange), TpAttr async get/set(4035)/get, then the qp
 * and segment resource chain. */
TEST_F(HccpSystemTest, ubCtrlTpAttrAsync)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        UbCtrlEnv env;
        int ret = UbCtrlBegin(env, phyid, UB_PORT_TP_ATTR, "UB CTRL TPATTR", true, true);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidNum(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidList(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlCtxInitAtomic(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlDevBaseAttr(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidExchange(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlTpListAsync(phyid, env, false);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlTpExchange(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlTpAttrGetAsync(phyid, env, UB_CTRL_TP_BITMAP_V2);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlTpAttrSetAsync(phyid, env, UB_CTRL_TP_BITMAP_V2, false);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlTpAttrGetAsync(phyid, env, UB_CTRL_TP_BITMAP_V2);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlChanCreate(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlCqCreate(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlQpCreate(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlQpExchange(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlQpImport(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlSegInit(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlLmemRegister(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlRmemExchange(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlRmemImport(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlShakeHand(phyid, env);
        if (ret != 0) {
            goto err;
        }
        TEST_LOG_INFO(phyid, "tp_attr_test passed");
    err:
        UbCtrlEnd(this, env, phyid);
        return ret;
    });
}

/* Ported from hlt_hccp_udma_write_value_001 (name is misleading - despite
 * "write_value" the entry only builds the control plane): ctx init + chan
 * (pollCqCstm=0) + cq + qp create. No socket, no shake hand. */
TEST_F(HccpSystemTest, ubCtrlChanCqQpChain)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        UbCtrlEnv env;
        int ret = UbCtrlBegin(env, phyid, UB_PORT_CHAN_CQ_QP, "UB CTRL CCQ", true, false);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidNum(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidList(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlCtxInitAtomic(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlDevBaseAttr(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlChanCreate(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlCqCreate(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlQpCreate(phyid, env);
        if (ret != 0) {
            goto err;
        }
        TEST_LOG_INFO(phyid, "udma_write_value_001 passed");
    err:
        UbCtrlEnd(this, env, phyid);
        return ret;
    });
}

/* Ported from hlt_hccp_aiv_urma_test: ctx init + cq create + qp create, dumping
 * the jfc/jetty info the original prints for the aiv (vector core) bring-up. */
TEST_F(HccpSystemTest, ubCtrlAivUrma)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        UbCtrlEnv env;
        int ret = UbCtrlBegin(env, phyid, UB_PORT_AIV_URMA, "UB CTRL AIV", true, false);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidNum(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidList(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlCtxInitAtomic(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlDevBaseAttr(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlCqCreate(phyid, env); /* no chan in this entry */
        if (ret != 0) {
            goto err;
        }
        TEST_LOG_INFO(
            phyid, "jfc_id=%u cqe_size=%u buf_addr=0x%llx db_addr=0x%llx", env.cqInfo.out.id, env.cqInfo.out.cqeSize,
            (unsigned long long)env.cqInfo.out.bufAddr, (unsigned long long)env.cqInfo.out.swdbAddr);
        ret = UbCtrlQpCreate(phyid, env);
        if (ret != 0) {
            goto err;
        }
        TEST_LOG_INFO(
            phyid, "sq_buff_va=0x%llx wqebb_size=%llu", (unsigned long long)env.qpInfo.ub.sqBuffVa,
            (unsigned long long)env.qpInfo.ub.wqebbSize);
        TEST_LOG_INFO(phyid, "aiv_urma_test passed");
    err:
        UbCtrlEnd(this, env, phyid);
        return ret;
    });
}

/* Ported from hlt_hccp_ub_async_event_test: ctx init + cq create + sleep(10) +
 * RaCtxGetAsyncEvents (up to 10 events). */
TEST_F(HccpSystemTest, ubCtrlGetAsyncEvents)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        UbCtrlEnv env;
        struct AsyncEvent events[UB_CTRL_ASYNC_EVENT_NUM] = {};
        unsigned int num = UB_CTRL_ASYNC_EVENT_NUM;
        int ret = UbCtrlBegin(env, phyid, UB_PORT_ASYNC_EVENT, "UB CTRL AEVT", true, false);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidNum(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidList(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlCtxInitAtomic(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlDevBaseAttr(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlCqCreate(phyid, env);
        if (ret != 0) {
            goto err;
        }
        sleep(2); /* collection window, as the original (shortened from 10s: the
                     framework kills a host child after ~5s, and the sim has no
                     real async event source - the window only needs to elapse) */
        ret = RaCtxGetAsyncEvents(env.base.ctxHandle, events, &num);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaCtxGetAsyncEvents failed ret=%d", ret);
            goto err;
        }
        for (unsigned int i = 0; i < num; i++) {
            TEST_LOG_INFO(phyid, "async event[%u] res_id=%u event_type=%u", i, events[i].resId, events[i].eventType);
        }
        TEST_LOG_INFO(phyid, "ub_async_event_test passed");
    err:
        UbCtrlEnd(this, env, phyid);
        return ret;
    });
}

/* Ported from hlt_hccp_ub_get_jetty_context_test: full qp resource chain (chan +
 * cq + qp + exchange + import) + shake hand + RaCtxGetJettyContext on the local
 * qp. The original sleeps 60s afterwards to keep the ctx alive for manual
 * inspection; that hold is operational, not semantic, and is not carried over. */
TEST_F(HccpSystemTest, ubCtrlGetJettyContext)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        UbCtrlEnv env;
        uint8_t context[UB_CTRL_CONTEXT_LEN] = {0};
        /* memcpy the byte context into words like the original (no aliasing cast) */
        unsigned int words[UB_CTRL_CONTEXT_LEN / sizeof(unsigned int)] = {0};
        unsigned int len = UB_CTRL_CONTEXT_LEN;
        int ret = UbCtrlBegin(env, phyid, UB_PORT_JETTY_CONTEXT, "UB CTRL JCTX", true, true);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidNum(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidList(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlCtxInitAtomic(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlDevBaseAttr(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlChanCreate(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlCqCreate(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlQpCreate(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlQpExchange(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlQpImport(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlShakeHand(phyid, env);
        if (ret != 0) {
            goto err;
        }

        ret = RaCtxGetJettyContext(env.base.qpHandle, context, &len);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaCtxGetJettyContext failed ret=%d", ret);
            goto err;
        }
        (void)memcpy(words, context, sizeof(context));
        for (unsigned int i = 0; i < len / sizeof(unsigned int); i++) {
            TEST_LOG_INFO(phyid, "jetty context[%u] 0x%x", i, words[i]);
        }
        TEST_LOG_INFO(phyid, "ub_get_jetty_context_test passed");
    err:
        UbCtrlEnd(this, env, phyid);
        return ret;
    });
}

/* Ported from hlt_hccp_jetty_create_test_001: single-machine flow - rtSetDevice +
 * RaInit + ctx init + CCU TLV init request + self-peer tp list + cq + qp create.
 * The original runs one machine per dev; in the sim each child runs the same
 * single-device flow on its own phyId. */
TEST_F(HccpSystemTest, ubCtrlJettyCreate)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        UbCtrlEnv env;
        void* tlvHandle = nullptr;
        int ret = UbCtrlBegin(env, phyid, UB_PORT_JETTY_CREATE, "UB CTRL JCREATE", false, false);
        if (ret != 0) {
            goto err;
        }
        if (aclrtSetDevice(phyid) != 0) { /* test_rtSetDevice of the loop entry */
            TEST_LOG_ERROR(phyid, "aclrtSetDevice failed");
            ret = -1;
            goto err;
        }
        env.rtDeviceSet = true;
        ret = UbCtrlEidNum(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidList(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlCtxInitAtomic(phyid, env);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlDevBaseAttr(phyid, env);
        if (ret != 0) {
            goto err;
        }

        /* test_ra_tlv_init + test_ccu_init_request via the CCU TLV shim */
        ret = HccpTlvCcuInit(phyid, &tlvHandle);
        if (ret != 0) {
            goto err;
        }
        ret = HccpTlvCcuRequestVoid(phyid, tlvHandle, MSG_TYPE_CCU_INIT);
        if (ret != 0) {
            goto err_deinit_tlv;
        }

        /* single machine: the peer eid is the own eid (self tp) */
        env.remoteEid = env.ctxAttr.ub.eid;
        ret = UbCtrlTpListAsync(phyid, env, false);
        if (ret != 0) {
            goto err_deinit_tlv;
        }
        ret = UbCtrlCqCreate(phyid, env);
        if (ret != 0) {
            goto err_deinit_tlv;
        }
        ret = UbCtrlQpCreate(phyid, env);
        if (ret != 0) {
            goto err_deinit_tlv;
        }
        TEST_LOG_INFO(phyid, "jetty_create_test_001 passed");

    err_deinit_tlv:
        (void)HccpTlvCcuRequestVoid(phyid, tlvHandle, MSG_TYPE_CCU_UNINIT);
        (void)HccpTlvCcuDeinit(phyid, tlvHandle);
    err:
        UbCtrlEnd(this, env, phyid);
        return ret;
    });
}

/* Ported from hlt_hccp_jetty_query_test_001: two device contexts in one process,
 * each with ctx init + self-peer tp list + cq + qp; then RaCtxQpQueryBatch from
 * ctx A, the same qp queried again through ctx B (the original's cross-ctx query
 * intent), and RaCtxQpDestroyBatchAsync per ctx.
 *
 * Adaptations (t1 section 7): the sim has one device per child, so the original's
 * two machines (dev 0/dev 1) become two contexts on the same phyId; and the
 * original's handle aliasing (config[1].qp_handle = config[0].qp_handle before the
 * second query) makes the batch destroy run twice over ctx A's qp while ctx B's qp
 * leaks - the port queries ctx A's qp through ctx B (keeping the cross-ctx query
 * semantics) but destroys each context's own qp exactly once. */
TEST_F(HccpSystemTest, ubCtrlJettyQuery)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        UbCtrlEnv envA;
        void* cqB = nullptr;
        void* qpB = nullptr;
        void* ctxB = nullptr;
        void* tokenB = nullptr;
        struct QpCreateAttr qpAttrB = {};
        struct QpCreateInfo qpInfoB = {};
        struct CtxInitCfg cfgB = {};
        struct CtxInitAttr attrB = {};
        struct HccpTokenId tokenInfoB = {};
        struct CqInfoT cqInfoB = {};
        int ret = UbCtrlBegin(envA, phyid, UB_PORT_JETTY_QUERY, "UB CTRL JQUERY", false, false);
        if (ret != 0) {
            goto err;
        }
        if (aclrtSetDevice(phyid) != 0) {
            TEST_LOG_ERROR(phyid, "aclrtSetDevice failed");
            ret = -1;
            goto err;
        }
        envA.rtDeviceSet = true;

        /* context A */
        ret = UbCtrlEidNum(phyid, envA);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlEidList(phyid, envA);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlCtxInitAtomic(phyid, envA);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlDevBaseAttr(phyid, envA);
        if (ret != 0) {
            goto err;
        }
        envA.remoteEid = envA.ctxAttr.ub.eid;
        ret = UbCtrlTpListAsync(phyid, envA, false);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlCqCreate(phyid, envA);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlQpCreate(phyid, envA);
        if (ret != 0) {
            goto err;
        }

        /* context B on the same device (second machine in the original) */
        cfgB.mode = NETWORK_OFFLINE;
        attrB = envA.ctxAttr;
        ret = RaCtxInit(&cfgB, &attrB, &ctxB);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaCtxInit B failed ret=%d", ret);
            goto err;
        }
        ret = RaCtxTokenIdAlloc(ctxB, &tokenInfoB, &tokenB);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaCtxTokenIdAlloc B failed ret=%d", ret);
            goto err;
        }
        /* self-peer tp list on ctx B (test_ra_get_tp_info_list of the second loop) */
        {
            struct GetTpCfg tpCfgB[UB_CTRL_TP_NUM] = {};
            tpCfgB[0].flag.bs.ctp = 1;
            tpCfgB[0].transMode = CONN_RM;
            tpCfgB[0].localEid = envA.ctxAttr.ub.eid;
            tpCfgB[0].peerEid = envA.ctxAttr.ub.eid;
            struct HccpTpInfo tpInfoB[UB_CTRL_TP_NUM] = {};
            unsigned int tpNumB = UB_CTRL_TP_NUM;
            void* tpReqB = nullptr;
            ret = RaGetTpInfoListAsync(ctxB, tpCfgB, tpInfoB, &tpNumB, &tpReqB);
            if (ret == 0) {
                ret = UbCtrlAsyncReqResult(phyid, tpReqB);
            }
            if (ret != 0 || tpNumB == 0) {
                TEST_LOG_ERROR(phyid, "get tp_info_list B failed ret=%d num=%u", ret, tpNumB);
                goto err;
            }
        }
        cqInfoB.in.depth = UB_CTRL_CQ_DEPTH;
        cqInfoB.in.ub.mode = JFC_MODE_STARS_POLL;
        cqInfoB.in.ub.ceqn = 0;
        cqInfoB.in.ub.ccuExCfg.valid = false;
        cqInfoB.in.ub.ccuExCfg.cqeFlag = 1;
        ret = RaCtxCqCreate(ctxB, &cqInfoB, &cqB);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaCtxCqCreate B failed ret=%d", ret);
            goto err;
        }
        qpAttrB = envA.qpAttr;
        qpAttrB.scqHandle = cqB;
        qpAttrB.rcqHandle = cqB;
        qpAttrB.srqHandle = cqB;
        ret = RaCtxQpCreate(ctxB, &qpAttrB, &qpInfoB, &qpB);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaCtxQpCreate B failed ret=%d", ret);
            goto err;
        }

        /* the original queries config[0]'s qp, then repeats the same query after
         * aliasing the handle into config[1] - the entry's point is that a query
         * works regardless of which context created the qp; keep both queries and
         * add one for ctx B's own qp */
        ret = UbCtrlQpQueryBatch(phyid, envA.base.qpHandle);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlQpQueryBatch(phyid, envA.base.qpHandle);
        if (ret != 0) {
            goto err;
        }
        ret = UbCtrlQpQueryBatch(phyid, qpB);
        if (ret != 0) {
            goto err;
        }

        /* batch destroy each context's own qp exactly once */
        ret = UbCtrlQpDestroyBatchAsync(phyid, envA, envA.base.qpHandle);
        if (ret != 0) {
            goto err;
        }
        envA.base.qpHandle = nullptr;
        {
            void* qpHandlesB[1] = {qpB};
            unsigned int numB = 1;
            void* reqB = nullptr;
            ret = RaCtxQpDestroyBatchAsync(ctxB, qpHandlesB, &numB, &reqB);
            if (ret != 0) {
                TEST_LOG_ERROR(phyid, "RaCtxQpDestroyBatchAsync B failed ret=%d", ret);
                goto err;
            }
            ret = UbCtrlAsyncReqResult(phyid, reqB);
            if (ret != 0 || numB == 0) {
                TEST_LOG_ERROR(phyid, "qp destroy batch B failed ret=%d num=%u", ret, numB);
                goto err;
            }
            qpB = nullptr;
        }
        TEST_LOG_INFO(phyid, "jetty_query_test_001 passed");

    err:
        if (qpB != nullptr) {
            (void)RaCtxQpDestroy(qpB);
        }
        if (cqB != nullptr) {
            (void)RaCtxCqDestroy(ctxB, cqB);
        }
        if (tokenB != nullptr) {
            (void)RaCtxTokenIdFree(ctxB, tokenB);
        }
        if (ctxB != nullptr) {
            (void)RaCtxDeinit(ctxB);
        }
        UbCtrlEnd(this, envA, phyid);
        return ret;
    });
}
