/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * Batch 4 of the test-dev hccp_test migration: the CCU driver domain of
 * hlt_hccp_ccu_driver_test.c (32 gstool entries, one TEST_F per entry; the t1
 * survey counted 31, the file exports 32).
 *
 * The original domain runs every entry single-machine on host[0] (python CI
 * cann-drv-python/testcase/hccl/ccu.py), so every case here uses the
 * single-end form RunTests({1}). Two session shapes exist in the original:
 *   - test_ccu_custom_channel: a full open -> init -> ONE dispatch -> uninit
 *     chain per call (ported as CcuDrvOneShot);
 *   - test_ccu_init_via_hccp ... test_ccu_uninit_via_hccp: one session around
 *     several dispatches (the set/get pairs with the set->get comparison).
 * hlt_ccu_set_msid used two separate full chains for its SET and its GET; in
 * the simulation the device process is killed by rtCloseNetService and its
 * successor starts with a fresh CCU state (ccu_init wipes the shadow), so the
 * port keeps the assertion but runs both commands inside one session.
 *
 * The gstool/conf_name parameters (ccu.json payloads, start_entry, die,
 * mission ids, register values) become in-case constants taken from the
 * python CI invocations; the json loader itself is not ported - the SET
 * payloads are the defined 8-byte-block pattern the CI json carried
 * (0x1122334455667788 per block).
 */

#include "cases_common.h"
#include "hccp_ccu_tlv_shim.h"
#include <arpa/inet.h>

/* aclrt entry points come from the real <acl/acl_rt.h> via cases_common.h,
 * implemented by the runtime stub (runtime_rt_stub.c / runtime_stub.c). */

/* ------------------------------------------------------------------ helpers */

/* Port of test_get_opcode_entry_size (hlt_hccp_ccu_driver_common.c). */
static unsigned int CcuEntrySize(unsigned int op)
{
    switch (op) {
        case CCU_U_OP_GET_INSTRUCTION:
        case CCU_U_OP_SET_INSTRUCTION:
        case CCU_U_OP_GET_JETTY_CTX:
        case CCU_U_OP_SET_JETTY_CTX:
        case CCU_U_OP_GET_LOOP_CKE_CTX:
        case CCU_U_OP_SET_LOOP_CKE_CTX:
            return 32U;
        case CCU_U_OP_GET_GSA:
        case CCU_U_OP_SET_GSA:
        case CCU_U_OP_GET_XN:
        case CCU_U_OP_SET_XN:
        case CCU_U_OP_GET_CKE:
        case CCU_U_OP_SET_CKE:
        case CCU_U_OP_GET_PFE:
        case CCU_U_OP_SET_PFE:
            return 8U;
        case CCU_U_OP_GET_CHANNEL:
        case CCU_U_OP_SET_CHANNEL:
        case CCU_U_OP_GET_MISSION_CTX:
        case CCU_U_OP_GET_LOOP_CTX:
            return 64U;
        default:
            return 0U;
    }
}

/* One CCU command through the TLV dispatch shim; op_ret is part of the driver
 * contract, so a rejected operation fails the case. */
static int CcuDispatch(int phyid, void* tlvHandle, const struct channel_info_in* in, struct channel_info_out* out)
{
    int ret = HccpTlvCcuCustomChannel(phyid, tlvHandle, in, out);
    if (ret != 0) {
        return ret;
    }
    if (out->op_ret != 0) {
        TEST_LOG_ERROR(phyid, "CCU op %u rejected, op_ret=%d", (unsigned int)in->op, out->op_ret);
        return -1;
    }
    return 0;
}

/* Ported shape of test_ccu_init_via_hccp: TsdProcessOpen + aclrtSetDevice +
 * RaInit(offline) -> CaseOfflineInitNoRdv, then RaTlvInit + CCU_INIT. */
static int CcuDrvOpen(HccpSystemTest& t, int phyid, CaseEnv& env, void** tlvHandle)
{
    *tlvHandle = nullptr;
    int ret = t.CaseOfflineInitNoRdv(env, phyid, HCCP_VM_IFACE_SOCKET);
    if (ret != 0) {
        return ret; /* nothing left to clean up */
    }
    if (aclrtSetDevice(phyid) != 0) {
        TEST_LOG_ERROR(phyid, "aclrtSetDevice failed");
        (void)aclrtResetDevice(phyid);
        t.CaseCleanup(env);
        return -1;
    }
    ret = HccpTlvCcuInit(phyid, tlvHandle);
    if (ret != 0) {
        if (*tlvHandle != nullptr) {
            (void)HccpTlvCcuDeinit(phyid, *tlvHandle);
            *tlvHandle = nullptr;
        }
        (void)aclrtResetDevice(phyid);
        t.CaseCleanup(env);
        return ret;
    }
    ret = HccpTlvCcuRequestVoid(phyid, *tlvHandle, MSG_TYPE_CCU_INIT);
    if (ret != 0) {
        (void)HccpTlvCcuDeinit(phyid, *tlvHandle);
        *tlvHandle = nullptr;
        (void)aclrtResetDevice(phyid);
        t.CaseCleanup(env);
        return ret;
    }
    return 0;
}

/* Ported shape of test_ccu_uninit_via_hccp: CCU_UNINIT + RaTlvDeinit +
 * RaDeinit + aclrtResetDevice + TsdClose (RaDeinit/TsdClose live inside
 * CaseCleanup). Returns the first error so a failing command is not masked. */
static int CcuDrvClose(HccpSystemTest& t, int phyid, CaseEnv& env, void* tlvHandle, int firstErr)
{
    if (tlvHandle != nullptr) {
        (void)HccpTlvCcuRequestVoid(phyid, tlvHandle, MSG_TYPE_CCU_UNINIT);
        (void)HccpTlvCcuDeinit(phyid, tlvHandle);
    }
    if (aclrtResetDevice(phyid) != 0) {
        TEST_LOG_ERROR(phyid, "aclrtResetDevice failed");
        if (firstErr == 0) {
            firstErr = -1;
        }
    }
    t.CaseCleanup(env);
    return firstErr;
}

/* Ported shape of test_ccu_custom_channel: the full open -> init -> one
 * dispatch -> uninit chain around a single command. */
static int CcuDrvOneShot(HccpSystemTest& t, int phyid, const struct channel_info_in* in, struct channel_info_out* out)
{
    CaseEnv env;
    void* tlvHandle = nullptr;
    int ret = CcuDrvOpen(t, phyid, env, &tlvHandle);
    if (ret != 0) {
        return ret;
    }
    ret = CcuDispatch(phyid, tlvHandle, in, out);
    return CcuDrvClose(t, phyid, env, tlvHandle, ret);
}

/* Port of the ccu.json payload loader (test_get_ccu_data_byte): udie_idx,
 * data_array_size (the python CI value 2) and the 8-byte-block pattern the CI
 * json carried per entry stand in for the ccu_data_byteN arrays. */
static int CcuFillSetPayload(
    struct channel_info_in* in, unsigned int op, unsigned int die, unsigned int startEntry, unsigned int arraySize)
{
    unsigned int entrySize = CcuEntrySize(op);
    if (entrySize == 0U || arraySize == 0U || arraySize > CCU_DATA_TYPE_UNION_ARRAY_SIZE) {
        return -1;
    }
    in->op = (ccu_u_opcode_t)op;
    in->offset_start = startEntry;
    in->data.data_info.udie_idx = die;
    in->data.data_info.data_array_size = arraySize;
    in->data.data_info.data_array_len = entrySize * arraySize;
    for (unsigned int i = 0; i < arraySize; i++) {
        for (unsigned int blk = 0; blk < entrySize / 8U; blk++) {
            unsigned long long v = 0x1122334455667788ULL + (unsigned long long)(i * 16U + blk);
            (void)memcpy(&in->data.data_info.data_array[i].byte64.raw[blk * 8U], &v, sizeof(v));
        }
    }
    return 0;
}

/* Port of test_in_get_build: same offset/die/entry count as the SET request,
 * data_array_len derived from the GET entry size. */
static void CcuBuildGet(const struct channel_info_in& in, unsigned int getOp, struct channel_info_in* inGet)
{
    *inGet = {};
    inGet->op = (ccu_u_opcode_t)getOp;
    inGet->offset_start = in.offset_start;
    inGet->data.data_info.udie_idx = in.data.data_info.udie_idx;
    inGet->data.data_info.data_array_size = in.data.data_info.data_array_size;
    inGet->data.data_info.data_array_len = CcuEntrySize(getOp) * in.data.data_info.data_array_size;
}

/* Port of test_set_and_get_cmp: byte-wise comparison of the SET payload
 * against the GET reply over all union entries; PFE keeps its 3-byte compare
 * window (the rest of the entry is RO/reserved space). */
static int CcuSetGetCmp(int phyid, const struct channel_info_in& in, const struct channel_info_out& out)
{
    unsigned int size = CcuEntrySize(in.op);
    if (in.op == CCU_U_OP_SET_PFE) {
        size = 3U; /* reserved space does not need to be compared */
    }
    for (unsigned int i = 0; i < CCU_DATA_TYPE_UNION_ARRAY_SIZE; i++) {
        for (unsigned int j = 0; j < size; j++) {
            if (in.data.data_info.data_array[i].byte64.raw[j] != out.data.data_info.data_array[i].byte64.raw[j]) {
                TEST_LOG_ERROR(
                    phyid, "set/get mismatch: entry %u byte %u: set=%u get=%u", i, j,
                    (unsigned char)in.data.data_info.data_array[i].byte64.raw[j],
                    (unsigned char)out.data.data_info.data_array[i].byte64.raw[j]);
                return -1;
            }
        }
    }
    return 0;
}

/* Port of test_baseinfo_print: the basic-info fields the driver stamps. */
static void CcuBaseinfoPrint(int phyid, const struct channel_info_out& out)
{
    const struct ccu_baseinfo* info = &out.data.data_info.data_array[0].baseinfo;
    TEST_LOG_INFO(
        phyid, "baseinfo: ms_id=%u missionKey=0x%X resourceAddr=%p cap0=0x%X", info->ms_id, info->missionKey,
        info->resourceAddr, info->caps.lqc_ccu_cap0);
}

/* -------------------------------------------------------------------- cases */

/* Ported from hlt_ccu_get_basic_info: one full chain around GET_BASIC_INFO;
 * the driver-stamped fields are asserted non-empty (the original only
 * printed them). */
TEST_F(HccpSystemTest, ccuDriverGetBasicInfo)
{
    RunTests({1}, [this](int phyid) -> int {
        struct channel_info_in in = {};
        struct channel_info_out out = {};
        in.op = CCU_U_OP_GET_BASIC_INFO;
        in.data.data_info.udie_idx = 0;
        in.data.data_info.data_array_size = 1;
        in.data.data_info.data_array_len = sizeof(struct ccu_baseinfo);
        int ret = CcuDrvOneShot(*this, phyid, &in, &out);
        if (ret != 0) {
            return ret;
        }
        CcuBaseinfoPrint(phyid, out);
        const struct ccu_baseinfo* info = &out.data.data_info.data_array[0].baseinfo;
        if (info->token_valid == 0 || info->missionKey == 0 || info->resourceAddr == nullptr) {
            TEST_LOG_ERROR(
                phyid, "GET_BASIC_INFO fields not filled: token_valid=%u missionKey=0x%X res=%p", info->token_valid,
                info->missionKey, info->resourceAddr);
            return -1;
        }
        return 0;
    });
}

/* Ported from hlt_ccu_set_msid: SET_MSID_TOKEN followed by the GET_MSID
 * readback comparison (out_get.baseinfo.ms_id == ms_id). The original ran the
 * two commands in two separate full chains; in the simulation the device
 * process is killed between chains and its successor starts with a fresh CCU
 * state, so both commands share one session here - the readback assertion is
 * unchanged. */
TEST_F(HccpSystemTest, ccuDriverSetMsid)
{
    RunTests({1}, [this](int phyid) -> int {
        CaseEnv env;
        void* tlvHandle = nullptr;
        struct channel_info_in in = {};
        struct channel_info_in inGet = {};
        struct channel_info_out out = {};
        struct channel_info_out outGet = {};
        const unsigned int msid = 0xA6;
        int ret = CcuDrvOpen(*this, phyid, env, &tlvHandle);
        if (ret != 0) {
            return ret;
        }

        in.op = CCU_U_OP_SET_MSID_TOKEN;
        in.data.data_info.udie_idx = 0;
        in.data.data_info.data_array_size = 1;
        in.data.data_info.data_array_len = sizeof(struct ccu_baseinfo);
        in.data.data_info.data_array[0].baseinfo.ms_id = msid;
        ret = CcuDispatch(phyid, tlvHandle, &in, &out);
        if (ret != 0) {
            goto err;
        }

        inGet.op = CCU_U_OP_GET_MSID;
        inGet.data.data_info.udie_idx = 0;
        inGet.data.data_info.data_array_size = 1;
        inGet.data.data_info.data_array_len = sizeof(struct ccu_baseinfo);
        ret = CcuDispatch(phyid, tlvHandle, &inGet, &outGet);
        if (ret != 0) {
            goto err;
        }
        if (outGet.data.data_info.data_array[0].baseinfo.ms_id != msid) {
            TEST_LOG_ERROR(
                phyid, "set msid %u != get msid %u", msid, outGet.data.data_info.data_array[0].baseinfo.ms_id);
            ret = -1;
            goto err;
        }
        TEST_LOG_INFO(phyid, "msid readback ok: 0x%X", msid);
    err:
        return CcuDrvClose(*this, phyid, env, tlvHandle, ret);
    });
}

/* Ported from hlt_ccu_get_msid: one full chain around GET_MSID. */
TEST_F(HccpSystemTest, ccuDriverGetMsid)
{
    RunTests({1}, [this](int phyid) -> int {
        struct channel_info_in in = {};
        struct channel_info_out out = {};
        in.op = CCU_U_OP_GET_MSID;
        in.data.data_info.udie_idx = 0;
        in.data.data_info.data_array_size = 1;
        in.data.data_info.data_array_len = sizeof(struct ccu_baseinfo);
        int ret = CcuDrvOneShot(*this, phyid, &in, &out);
        if (ret != 0) {
            return ret;
        }
        CcuBaseinfoPrint(phyid, out);
        return 0;
    });
}

/* Ported from hlt_ccu_set_instruction: the full custom flow of the original -
 * open + aclrtSetDevice + RaInit, EID enumeration with a local-IP match,
 * RaCtxInit on the matched EID, TLV session, aclrtMalloc + aclrtMemset of the
 * instruction buffer, then SET_INSTRUCTION with the device address. gstool
 * parameters follow the python CI: start_entry=0, udie=0, data_array_size=8,
 * data_array_len=64, init=0xff. */
TEST_F(HccpSystemTest, ccuDriverSetInstruction)
{
    RunTests({1}, [this](int phyid) -> int {
        CaseEnv env;
        void* tlvHandle = nullptr;
        void* ctxHandle = nullptr;
        void* devMem = nullptr;
        struct HccpDevEidInfo* eidList = nullptr;
        struct channel_info_in in = {};
        struct channel_info_out out = {};
        const unsigned int startEntry = 0;
        const unsigned int udie = 0;
        const unsigned int arraySize = 8;
        const unsigned int arrayLen = 64;
        int ret = CaseOfflineInitNoRdv(env, phyid, HCCP_VM_IFACE_SOCKET);
        if (ret != 0) {
            return ret;
        }
        if (aclrtSetDevice(phyid) != 0) {
            TEST_LOG_ERROR(phyid, "aclrtSetDevice failed");
            ret = -1;
            goto err;
        }

        {
            struct RaInfo info = {};
            info.mode = 1; /* offline */
            info.phyId = (unsigned int)phyid;
            unsigned int eidNum = 0;
            ret = RaGetDevEidInfoNum(info, &eidNum);
            if (ret != 0 || eidNum == 0) {
                TEST_LOG_ERROR(phyid, "RaGetDevEidInfoNum failed ret=%d num=%u", ret, eidNum);
                ret = -1;
                goto err;
            }
            eidList = (struct HccpDevEidInfo*)calloc(eidNum, sizeof(struct HccpDevEidInfo));
            if (eidList == nullptr) {
                TEST_LOG_ERROR(phyid, "calloc eid list failed");
                ret = -1;
                goto err;
            }
            unsigned int listNum = eidNum;
            ret = RaGetDevEidInfoList(info, eidList, &listNum);
            if (ret != 0 || eidNum != listNum) {
                TEST_LOG_ERROR(phyid, "RaGetDevEidInfoList failed ret=%d num=%u/%u", ret, listNum, eidNum);
                ret = -1;
                goto err;
            }
            struct in_addr wantIp
                = HccpVmToLoopback(in_addr{htonl(HccpVmNicIp((uint32_t)phyid, HCCP_VM_IFACE_SOCKET))});
            int hit = -1;
            for (unsigned int i = 0; i < eidNum; i++) {
                TEST_LOG_INFO(
                    phyid, "eid[%u]: addr=0x%X eidIndex=%u dieId=%u", i, eidList[i].eid.in4.addr, eidList[i].eidIndex,
                    eidList[i].dieId);
                if (eidList[i].eid.in4.addr == wantIp.s_addr) {
                    hit = (int)i;
                    break;
                }
            }
            if (hit < 0) {
                TEST_LOG_ERROR(phyid, "no eid matches local ip %s", inet_ntoa(wantIp));
                ret = -1;
                goto err;
            }

            struct CtxInitCfg ctxCfg = {};
            struct CtxInitAttr ctxAttr = {};
            ctxCfg.mode = 1; /* offline, the original nic_position */
            ctxAttr.phyId = (unsigned int)phyid;
            ctxAttr.ub.eid = eidList[hit].eid;
            ctxAttr.ub.eidIndex = eidList[hit].eidIndex;
            ret = RaCtxInit(&ctxCfg, &ctxAttr, &ctxHandle);
            if (ret != 0) {
                TEST_LOG_ERROR(phyid, "RaCtxInit failed ret=%d", ret);
                goto err;
            }
        }

        ret = HccpTlvCcuInit(phyid, &tlvHandle);
        if (ret != 0) {
            goto err;
        }
        ret = HccpTlvCcuRequestVoid(phyid, tlvHandle, MSG_TYPE_CCU_INIT);
        if (ret != 0) {
            goto err;
        }

        if (aclrtMalloc(&devMem, arrayLen, ACL_MEM_MALLOC_HUGE_FIRST) != 0) {
            TEST_LOG_ERROR(phyid, "aclrtMalloc failed");
            ret = -1;
            goto err;
        }
        if (aclrtMemset(devMem, arrayLen, 0xff, arrayLen) != 0) {
            TEST_LOG_ERROR(phyid, "aclrtMemset failed");
            ret = -1;
            goto err;
        }

        in.op = CCU_U_OP_SET_INSTRUCTION;
        in.offset_start = startEntry;
        in.data.data_info.udie_idx = udie;
        in.data.data_info.data_array_len = arrayLen;
        in.data.data_info.data_array_size = arraySize;
        in.data.data_info.data_array[0].insinfo.resourceAddr = (unsigned long)(uintptr_t)devMem;
        TEST_LOG_INFO(
            phyid, "SET_INSTRUCTION: offset=%u udie=%u size=%u len=%u addr=0x%lx", startEntry, udie, arraySize,
            arrayLen, in.data.data_info.data_array[0].insinfo.resourceAddr);
        ret = CcuDispatch(phyid, tlvHandle, &in, &out);
        if (ret != 0) {
            goto err;
        }

        ret = 0;
    err:
        if (devMem != nullptr) {
            (void)aclrtFree(devMem); /* the original leaked the buffer */
        }
        if (ctxHandle != nullptr) {
            (void)RaCtxDeinit(ctxHandle);
        }
        if (tlvHandle != nullptr) {
            (void)HccpTlvCcuRequestVoid(phyid, tlvHandle, MSG_TYPE_CCU_UNINIT);
            (void)HccpTlvCcuDeinit(phyid, tlvHandle);
        }
        (void)aclrtResetDevice(phyid);
        CaseCleanup(env);
        free(eidList);
        return ret;
    });
}

/* Ported from hlt_ccu_get_instruction: one full chain around GET_INSTRUCTION
 * (python CI: udie=0, start_entry=5, data_array_size=5). */
TEST_F(HccpSystemTest, ccuDriverGetInstruction)
{
    RunTests({1}, [this](int phyid) -> int {
        struct channel_info_in in = {};
        struct channel_info_out out = {};
        in.op = CCU_U_OP_GET_INSTRUCTION;
        in.offset_start = 5;
        in.data.data_info.udie_idx = 0;
        in.data.data_info.data_array_size = 5;
        in.data.data_info.data_array_len = CcuEntrySize(CCU_U_OP_GET_INSTRUCTION) * 5U;
        int ret = CcuDrvOneShot(*this, phyid, &in, &out);
        if (ret != 0) {
            return ret;
        }
        TEST_LOG_INFO(phyid, "GET_INSTRUCTION ok: offset_next=%u", out.offset_next);
        return 0;
    });
}

/* Ported from hlt_ccu_set_gsa: SET_GSA (8B entries) -> GET_GSA with the
 * byte-wise set->get comparison (python CI: start_entry=2). */
TEST_F(HccpSystemTest, ccuDriverSetGsa)
{
    RunTests({1}, [this](int phyid) -> int {
        CaseEnv env;
        void* tlvHandle = nullptr;
        struct channel_info_in in = {};
        struct channel_info_in inGet = {};
        struct channel_info_out out = {};
        struct channel_info_out outGet = {};
        int ret = CcuFillSetPayload(&in, CCU_U_OP_SET_GSA, 0, 2, 2);
        if (ret != 0) {
            return ret;
        }
        ret = CcuDrvOpen(*this, phyid, env, &tlvHandle);
        if (ret != 0) {
            return ret;
        }
        ret = CcuDispatch(phyid, tlvHandle, &in, &out);
        if (ret != 0) {
            goto err;
        }
        CcuBuildGet(in, CCU_U_OP_GET_GSA, &inGet);
        ret = CcuDispatch(phyid, tlvHandle, &inGet, &outGet);
        if (ret != 0) {
            goto err;
        }
        ret = CcuSetGetCmp(phyid, in, outGet);
        if (ret == 0) {
            TEST_LOG_INFO(phyid, "GSA set/get compare ok");
        }
    err:
        return CcuDrvClose(*this, phyid, env, tlvHandle, ret);
    });
}

/* Ported from hlt_ccu_get_gsa: one full chain around GET_GSA per die (python
 * CI drove udie 0 and 1). */
TEST_F(HccpSystemTest, ccuDriverGetGsa)
{
    RunTests({1}, [this](int phyid) -> int {
        static const unsigned int dies[] = {0, 1};
        for (unsigned int d : dies) {
            struct channel_info_in in = {};
            struct channel_info_out out = {};
            in.op = CCU_U_OP_GET_GSA;
            in.offset_start = 8;
            in.data.data_info.udie_idx = d;
            in.data.data_info.data_array_size = 8;
            in.data.data_info.data_array_len = CcuEntrySize(CCU_U_OP_GET_GSA) * 8U;
            int ret = CcuDrvOneShot(*this, phyid, &in, &out);
            if (ret != 0) {
                return ret;
            }
            TEST_LOG_INFO(phyid, "GET_GSA ok: die=%u offset_next=%u", d, out.offset_next);
        }
        return 0;
    });
}

/* Ported from hlt_ccu_set_xn: SET_XN (8B entries) -> GET_XN with the
 * set->get comparison (python CI: start_entry=8). */
TEST_F(HccpSystemTest, ccuDriverSetXn)
{
    RunTests({1}, [this](int phyid) -> int {
        CaseEnv env;
        void* tlvHandle = nullptr;
        struct channel_info_in in = {};
        struct channel_info_in inGet = {};
        struct channel_info_out out = {};
        struct channel_info_out outGet = {};
        int ret = CcuFillSetPayload(&in, CCU_U_OP_SET_XN, 0, 8, 2);
        if (ret != 0) {
            return ret;
        }
        ret = CcuDrvOpen(*this, phyid, env, &tlvHandle);
        if (ret != 0) {
            return ret;
        }
        ret = CcuDispatch(phyid, tlvHandle, &in, &out);
        if (ret != 0) {
            goto err;
        }
        CcuBuildGet(in, CCU_U_OP_GET_XN, &inGet);
        ret = CcuDispatch(phyid, tlvHandle, &inGet, &outGet);
        if (ret != 0) {
            goto err;
        }
        ret = CcuSetGetCmp(phyid, in, outGet);
        if (ret == 0) {
            TEST_LOG_INFO(phyid, "XN set/get compare ok");
        }
    err:
        return CcuDrvClose(*this, phyid, env, tlvHandle, ret);
    });
}

/* Ported from hlt_ccu_get_xn: one full chain around GET_XN per die. */
TEST_F(HccpSystemTest, ccuDriverGetXn)
{
    RunTests({1}, [this](int phyid) -> int {
        static const unsigned int dies[] = {0, 1};
        for (unsigned int d : dies) {
            struct channel_info_in in = {};
            struct channel_info_out out = {};
            in.op = CCU_U_OP_GET_XN;
            in.offset_start = 8;
            in.data.data_info.udie_idx = d;
            in.data.data_info.data_array_size = 8;
            in.data.data_info.data_array_len = CcuEntrySize(CCU_U_OP_GET_XN) * 8U;
            int ret = CcuDrvOneShot(*this, phyid, &in, &out);
            if (ret != 0) {
                return ret;
            }
            TEST_LOG_INFO(phyid, "GET_XN ok: die=%u offset_next=%u", d, out.offset_next);
        }
        return 0;
    });
}

/* Ported from hlt_ccu_set_cke: SET_CKE (8B entries) -> GET_CKE without the
 * comparison - the original comment notes the entry has RO fields, so only
 * the channel round trip is verified (python CI: start_entry=8). */
TEST_F(HccpSystemTest, ccuDriverSetCke)
{
    RunTests({1}, [this](int phyid) -> int {
        CaseEnv env;
        void* tlvHandle = nullptr;
        struct channel_info_in in = {};
        struct channel_info_in inGet = {};
        struct channel_info_out out = {};
        struct channel_info_out outGet = {};
        int ret = CcuFillSetPayload(&in, CCU_U_OP_SET_CKE, 0, 8, 2);
        if (ret != 0) {
            return ret;
        }
        ret = CcuDrvOpen(*this, phyid, env, &tlvHandle);
        if (ret != 0) {
            return ret;
        }
        ret = CcuDispatch(phyid, tlvHandle, &in, &out);
        if (ret != 0) {
            goto err;
        }
        CcuBuildGet(in, CCU_U_OP_GET_CKE, &inGet);
        ret = CcuDispatch(phyid, tlvHandle, &inGet, &outGet);
        if (ret != 0) {
            goto err;
        }
        TEST_LOG_INFO(phyid, "CKE set/get round trip ok (RO fields not compared)");
    err:
        return CcuDrvClose(*this, phyid, env, tlvHandle, ret);
    });
}

/* Ported from hlt_ccu_get_cke: one full chain around GET_CKE per die. */
TEST_F(HccpSystemTest, ccuDriverGetCke)
{
    RunTests({1}, [this](int phyid) -> int {
        static const unsigned int dies[] = {0, 1};
        for (unsigned int d : dies) {
            struct channel_info_in in = {};
            struct channel_info_out out = {};
            in.op = CCU_U_OP_GET_CKE;
            in.offset_start = 8;
            in.data.data_info.udie_idx = d;
            in.data.data_info.data_array_size = 8;
            in.data.data_info.data_array_len = CcuEntrySize(CCU_U_OP_GET_CKE) * 8U;
            int ret = CcuDrvOneShot(*this, phyid, &in, &out);
            if (ret != 0) {
                return ret;
            }
            TEST_LOG_INFO(phyid, "GET_CKE ok: die=%u offset_next=%u", d, out.offset_next);
        }
        return 0;
    });
}

/* Ported from hlt_ccu_set_pfe: SET_PFE (8B entries) -> GET_PFE with the
 * 3-byte comparison window (the rest of the entry is reserved space). */
TEST_F(HccpSystemTest, ccuDriverSetPfe)
{
    RunTests({1}, [this](int phyid) -> int {
        CaseEnv env;
        void* tlvHandle = nullptr;
        struct channel_info_in in = {};
        struct channel_info_in inGet = {};
        struct channel_info_out out = {};
        struct channel_info_out outGet = {};
        int ret = CcuFillSetPayload(&in, CCU_U_OP_SET_PFE, 0, 8, 2);
        if (ret != 0) {
            return ret;
        }
        ret = CcuDrvOpen(*this, phyid, env, &tlvHandle);
        if (ret != 0) {
            return ret;
        }
        ret = CcuDispatch(phyid, tlvHandle, &in, &out);
        if (ret != 0) {
            goto err;
        }
        CcuBuildGet(in, CCU_U_OP_GET_PFE, &inGet);
        ret = CcuDispatch(phyid, tlvHandle, &inGet, &outGet);
        if (ret != 0) {
            goto err;
        }
        ret = CcuSetGetCmp(phyid, in, outGet);
        if (ret == 0) {
            TEST_LOG_INFO(phyid, "PFE set/get compare ok (3-byte window)");
        }
    err:
        return CcuDrvClose(*this, phyid, env, tlvHandle, ret);
    });
}

/* Ported from hlt_ccu_get_pfe: one full chain around GET_PFE per die. */
TEST_F(HccpSystemTest, ccuDriverGetPfe)
{
    RunTests({1}, [this](int phyid) -> int {
        static const unsigned int dies[] = {0, 1};
        for (unsigned int d : dies) {
            struct channel_info_in in = {};
            struct channel_info_out out = {};
            in.op = CCU_U_OP_GET_PFE;
            in.offset_start = 8;
            in.data.data_info.udie_idx = d;
            in.data.data_info.data_array_size = 8;
            in.data.data_info.data_array_len = CcuEntrySize(CCU_U_OP_GET_PFE) * 8U;
            int ret = CcuDrvOneShot(*this, phyid, &in, &out);
            if (ret != 0) {
                return ret;
            }
            TEST_LOG_INFO(phyid, "GET_PFE ok: die=%u offset_next=%u", d, out.offset_next);
        }
        return 0;
    });
}

/* Ported from hlt_ccu_set_channel: SET_CHANNEL (64B entries) -> GET_CHANNEL
 * without the comparison (RO/reserved fields; python CI: start_entry=6). */
TEST_F(HccpSystemTest, ccuDriverSetChannel)
{
    RunTests({1}, [this](int phyid) -> int {
        CaseEnv env;
        void* tlvHandle = nullptr;
        struct channel_info_in in = {};
        struct channel_info_in inGet = {};
        struct channel_info_out out = {};
        struct channel_info_out outGet = {};
        int ret = CcuFillSetPayload(&in, CCU_U_OP_SET_CHANNEL, 0, 6, 2);
        if (ret != 0) {
            return ret;
        }
        ret = CcuDrvOpen(*this, phyid, env, &tlvHandle);
        if (ret != 0) {
            return ret;
        }
        ret = CcuDispatch(phyid, tlvHandle, &in, &out);
        if (ret != 0) {
            goto err;
        }
        CcuBuildGet(in, CCU_U_OP_GET_CHANNEL, &inGet);
        ret = CcuDispatch(phyid, tlvHandle, &inGet, &outGet);
        if (ret != 0) {
            goto err;
        }
        TEST_LOG_INFO(phyid, "CHANNEL set/get round trip ok (RO fields not compared)");
    err:
        return CcuDrvClose(*this, phyid, env, tlvHandle, ret);
    });
}

/* Ported from hlt_ccu_get_channel: one full chain around GET_CHANNEL per die
 * (python CI: start_entry=8, data_array_size=8, udie 0 and 1). */
TEST_F(HccpSystemTest, ccuDriverGetChannel)
{
    RunTests({1}, [this](int phyid) -> int {
        static const unsigned int dies[] = {0, 1};
        for (unsigned int d : dies) {
            struct channel_info_in in = {};
            struct channel_info_out out = {};
            in.op = CCU_U_OP_GET_CHANNEL;
            in.offset_start = 8;
            in.data.data_info.udie_idx = d;
            in.data.data_info.data_array_size = 8;
            in.data.data_info.data_array_len = CcuEntrySize(CCU_U_OP_GET_CHANNEL) * 8U;
            int ret = CcuDrvOneShot(*this, phyid, &in, &out);
            if (ret != 0) {
                return ret;
            }
            TEST_LOG_INFO(phyid, "GET_CHANNEL ok: die=%u offset_next=%u", d, out.offset_next);
        }
        return 0;
    });
}

/* Ported from hlt_ccu_set_jetty_ctx: SET_JETTY_CTX (32B entries) -> GET_JETTY_CTX
 * without the comparison (RO/reserved fields; python CI: start_entry=8). */
TEST_F(HccpSystemTest, ccuDriverSetJettyCtx)
{
    RunTests({1}, [this](int phyid) -> int {
        CaseEnv env;
        void* tlvHandle = nullptr;
        struct channel_info_in in = {};
        struct channel_info_in inGet = {};
        struct channel_info_out out = {};
        struct channel_info_out outGet = {};
        int ret = CcuFillSetPayload(&in, CCU_U_OP_SET_JETTY_CTX, 0, 8, 2);
        if (ret != 0) {
            return ret;
        }
        ret = CcuDrvOpen(*this, phyid, env, &tlvHandle);
        if (ret != 0) {
            return ret;
        }
        ret = CcuDispatch(phyid, tlvHandle, &in, &out);
        if (ret != 0) {
            goto err;
        }
        CcuBuildGet(in, CCU_U_OP_GET_JETTY_CTX, &inGet);
        ret = CcuDispatch(phyid, tlvHandle, &inGet, &outGet);
        if (ret != 0) {
            goto err;
        }
        TEST_LOG_INFO(phyid, "JETTY_CTX set/get round trip ok (RO fields not compared)");
    err:
        return CcuDrvClose(*this, phyid, env, tlvHandle, ret);
    });
}

/* Ported from hlt_ccu_get_jetty_ctx: one full chain around GET_JETTY_CTX per
 * die. */
TEST_F(HccpSystemTest, ccuDriverGetJettyCtx)
{
    RunTests({1}, [this](int phyid) -> int {
        static const unsigned int dies[] = {0, 1};
        for (unsigned int d : dies) {
            struct channel_info_in in = {};
            struct channel_info_out out = {};
            in.op = CCU_U_OP_GET_JETTY_CTX;
            in.offset_start = 8;
            in.data.data_info.udie_idx = d;
            in.data.data_info.data_array_size = 8;
            in.data.data_info.data_array_len = CcuEntrySize(CCU_U_OP_GET_JETTY_CTX) * 8U;
            int ret = CcuDrvOneShot(*this, phyid, &in, &out);
            if (ret != 0) {
                return ret;
            }
            TEST_LOG_INFO(phyid, "GET_JETTY_CTX ok: die=%u offset_next=%u", d, out.offset_next);
        }
        return 0;
    });
}

/* Ported from hlt_ccu_get_mission_ctx: one full chain around GET_MISSION_CTX
 * per die (python CI: start_entry=8, data_array_size=8). */
TEST_F(HccpSystemTest, ccuDriverGetMissionCtx)
{
    RunTests({1}, [this](int phyid) -> int {
        static const unsigned int dies[] = {0, 1};
        for (unsigned int d : dies) {
            struct channel_info_in in = {};
            struct channel_info_out out = {};
            in.op = CCU_U_OP_GET_MISSION_CTX;
            in.offset_start = 8;
            in.data.data_info.udie_idx = d;
            in.data.data_info.data_array_size = 8;
            in.data.data_info.data_array_len = CcuEntrySize(CCU_U_OP_GET_MISSION_CTX) * 8U;
            int ret = CcuDrvOneShot(*this, phyid, &in, &out);
            if (ret != 0) {
                return ret;
            }
            TEST_LOG_INFO(phyid, "GET_MISSION_CTX ok: die=%u offset_next=%u", d, out.offset_next);
        }
        return 0;
    });
}

/* Ported from hlt_ccu_get_loop_ctx: one full chain around GET_LOOP_CTX per
 * die. */
TEST_F(HccpSystemTest, ccuDriverGetLoopCtx)
{
    RunTests({1}, [this](int phyid) -> int {
        static const unsigned int dies[] = {0, 1};
        for (unsigned int d : dies) {
            struct channel_info_in in = {};
            struct channel_info_out out = {};
            in.op = CCU_U_OP_GET_LOOP_CTX;
            in.offset_start = 8;
            in.data.data_info.udie_idx = d;
            in.data.data_info.data_array_size = 8;
            in.data.data_info.data_array_len = CcuEntrySize(CCU_U_OP_GET_LOOP_CTX) * 8U;
            int ret = CcuDrvOneShot(*this, phyid, &in, &out);
            if (ret != 0) {
                return ret;
            }
            TEST_LOG_INFO(phyid, "GET_LOOP_CTX ok: die=%u offset_next=%u", d, out.offset_next);
        }
        return 0;
    });
}

/* Ported from hlt_ccu_set_loop_cke_ctx: SET_LOOP_CKE_CTX (32B entries) ->
 * GET_LOOP_CKE_CTX without the comparison (python CI: start_entry=8). */
TEST_F(HccpSystemTest, ccuDriverSetLoopCkeCtx)
{
    RunTests({1}, [this](int phyid) -> int {
        CaseEnv env;
        void* tlvHandle = nullptr;
        struct channel_info_in in = {};
        struct channel_info_in inGet = {};
        struct channel_info_out out = {};
        struct channel_info_out outGet = {};
        int ret = CcuFillSetPayload(&in, CCU_U_OP_SET_LOOP_CKE_CTX, 0, 8, 2);
        if (ret != 0) {
            return ret;
        }
        ret = CcuDrvOpen(*this, phyid, env, &tlvHandle);
        if (ret != 0) {
            return ret;
        }
        ret = CcuDispatch(phyid, tlvHandle, &in, &out);
        if (ret != 0) {
            goto err;
        }
        CcuBuildGet(in, CCU_U_OP_GET_LOOP_CKE_CTX, &inGet);
        ret = CcuDispatch(phyid, tlvHandle, &inGet, &outGet);
        if (ret != 0) {
            goto err;
        }
        TEST_LOG_INFO(phyid, "LOOP_CKE_CTX set/get round trip ok (RO fields not compared)");
    err:
        return CcuDrvClose(*this, phyid, env, tlvHandle, ret);
    });
}

/* Ported from hlt_ccu_get_loop_cke_ctx: one full chain around
 * GET_LOOP_CKE_CTX per die. */
TEST_F(HccpSystemTest, ccuDriverGetLoopCkeCtx)
{
    RunTests({1}, [this](int phyid) -> int {
        static const unsigned int dies[] = {0, 1};
        for (unsigned int d : dies) {
            struct channel_info_in in = {};
            struct channel_info_out out = {};
            in.op = CCU_U_OP_GET_LOOP_CKE_CTX;
            in.offset_start = 8;
            in.data.data_info.udie_idx = d;
            in.data.data_info.data_array_size = 8;
            in.data.data_info.data_array_len = CcuEntrySize(CCU_U_OP_GET_LOOP_CKE_CTX) * 8U;
            int ret = CcuDrvOneShot(*this, phyid, &in, &out);
            if (ret != 0) {
                return ret;
            }
            TEST_LOG_INFO(phyid, "GET_LOOP_CKE_CTX ok: die=%u offset_next=%u", d, out.offset_next);
        }
        return 0;
    });
}

/* Ported from hlt_ccu_set_chn_jetty_map: one full chain around
 * SET_CHN_JETTY_MAP with the python CI mapping value 15. */
TEST_F(HccpSystemTest, ccuDriverSetChnJettyMap)
{
    RunTests({1}, [this](int phyid) -> int {
        struct channel_info_in in = {};
        struct channel_info_out out = {};
        in.op = CCU_U_OP_SET_CHN_JETTY_MAP;
        in.data.data_info.udie_idx = 0;
        in.data.data_info.data_array[0].baseinfo.chn_jetty_map = 15;
        int ret = CcuDrvOneShot(*this, phyid, &in, &out);
        if (ret != 0) {
            return ret;
        }
        TEST_LOG_INFO(phyid, "SET_CHN_JETTY_MAP ok: map=15");
        return 0;
    });
}

/* Ported from hlt_ccu_get_chn_jetty_map: one full chain around
 * GET_CHN_JETTY_MAP. */
TEST_F(HccpSystemTest, ccuDriverGetChnJettyMap)
{
    RunTests({1}, [this](int phyid) -> int {
        struct channel_info_in in = {};
        struct channel_info_out out = {};
        in.op = CCU_U_OP_GET_CHN_JETTY_MAP;
        in.data.data_info.udie_idx = 0;
        int ret = CcuDrvOneShot(*this, phyid, &in, &out);
        if (ret != 0) {
            return ret;
        }
        TEST_LOG_INFO(phyid, "GET_CHN_JETTY_MAP ok: map=%u", out.data.data_info.data_array[0].baseinfo.chn_jetty_map);
        return 0;
    });
}

/* Ported from hlt_ccu_clean_tif_table: one full chain around
 * CLEAN_TIF_TABLE. */
TEST_F(HccpSystemTest, ccuDriverCleanTifTable)
{
    RunTests({1}, [this](int phyid) -> int {
        struct channel_info_in in = {};
        struct channel_info_out out = {};
        in.op = CCU_U_OP_CLEAN_TIF_TABLE;
        in.data.data_info.udie_idx = 0;
        int ret = CcuDrvOneShot(*this, phyid, &in, &out);
        if (ret != 0) {
            return ret;
        }
        TEST_LOG_INFO(phyid, "CLEAN_TIF_TABLE ok");
        return 0;
    });
}

/* Ported from hlt_ccu_set_task_kill: one full chain around SET_TASKKILL per
 * die with the die-only request form (no mission_info), like the python CI
 * invocations udie 0 and 1. */
TEST_F(HccpSystemTest, ccuDriverSetTaskKill)
{
    RunTests({1}, [this](int phyid) -> int {
        static const unsigned int dies[] = {0, 1};
        for (unsigned int d : dies) {
            struct channel_info_in in = {};
            struct channel_info_out out = {};
            in.op = CCU_U_OP_SET_TASKKILL;
            in.data.data_info.udie_idx = d;
            int ret = CcuDrvOneShot(*this, phyid, &in, &out);
            if (ret != 0) {
                return ret;
            }
            TEST_LOG_INFO(phyid, "SET_TASKKILL ok: die=%u", d);
        }
        return 0;
    });
}

/* Ported from hlt_ccu_set_task_kill_01: one full chain around SET_TASKKILL
 * with the single-mission form (python CI: ms_type=1, ms_id=16). */
TEST_F(HccpSystemTest, ccuDriverSetTaskKill01)
{
    RunTests({1}, [this](int phyid) -> int {
        struct channel_info_in in = {};
        struct channel_info_out out = {};
        in.op = CCU_U_OP_SET_TASKKILL;
        in.data.data_info.udie_idx = 0;
        in.data.data_info.data_array[0].mission_info.is_single_mission = true;
        in.data.data_info.data_array[0].mission_info.mission_id = 16;
        int ret = CcuDrvOneShot(*this, phyid, &in, &out);
        if (ret != 0) {
            return ret;
        }
        TEST_LOG_INFO(phyid, "SET_TASKKILL ok: single mission, mission_id=16");
        return 0;
    });
}

/* Ported from hlt_ccu_clean_task_kill_state_01: one full chain around
 * CLEAN_TASKKILL_STATE with the single-mission form (python CI: ms_type=1). */
TEST_F(HccpSystemTest, ccuDriverCleanTaskKillState01)
{
    RunTests({1}, [this](int phyid) -> int {
        struct channel_info_in in = {};
        struct channel_info_out out = {};
        in.op = CCU_U_OP_CLEAN_TASKKILL_STATE;
        in.data.data_info.udie_idx = 0;
        in.data.data_info.data_array[0].mission_info.is_single_mission = true;
        in.data.data_info.data_array[0].mission_info.mission_id = 0;
        int ret = CcuDrvOneShot(*this, phyid, &in, &out);
        if (ret != 0) {
            return ret;
        }
        TEST_LOG_INFO(phyid, "CLEAN_TASKKILL_STATE ok: mission_id=0");
        return 0;
    });
}

/* Ported from hlt_ccu_clean_taskkill_state: one full chain around
 * CLEAN_TASKKILL_STATE with the die-only request form. */
TEST_F(HccpSystemTest, ccuDriverCleanTaskkillState)
{
    RunTests({1}, [this](int phyid) -> int {
        struct channel_info_in in = {};
        struct channel_info_out out = {};
        in.op = CCU_U_OP_CLEAN_TASKKILL_STATE;
        in.data.data_info.udie_idx = 0;
        int ret = CcuDrvOneShot(*this, phyid, &in, &out);
        if (ret != 0) {
            return ret;
        }
        TEST_LOG_INFO(phyid, "CLEAN_TASKKILL_STATE ok");
        return 0;
    });
}

/* Ported from hlt_ccu_set_tif_split_size: one full chain around
 * SET_TIF_SPLIT_SIZE with split_pkt_unit = 0 (unit: byte). */
TEST_F(HccpSystemTest, ccuDriverSetTifSplitSize)
{
    RunTests({1}, [this](int phyid) -> int {
        struct channel_info_in in = {};
        struct channel_info_out out = {};
        in.op = CCU_U_OP_SET_TIF_SPLIT_SIZE;
        in.data.data_info.udie_idx = 0;
        in.data.data_info.data_array[0].tif_split_info.split_pkt_unit = 0; /* 0: byte */
        int ret = CcuDrvOneShot(*this, phyid, &in, &out);
        if (ret != 0) {
            return ret;
        }
        TEST_LOG_INFO(phyid, "SET_TIF_SPLIT_SIZE ok: split_pkt_unit=0 (byte)");
        return 0;
    });
}

/* Ported from hlt_ccu_get_high_perf_xn: one full chain around
 * GET_HIGH_PERF_XN; the register range values are logged like the original. */
TEST_F(HccpSystemTest, ccuDriverGetHighPerfXn)
{
    RunTests({1}, [this](int phyid) -> int {
        struct channel_info_in in = {};
        struct channel_info_out out = {};
        in.op = CCU_U_OP_GET_HIGH_PERF_XN;
        in.data.data_info.udie_idx = 0;
        in.data.data_info.data_array_size = 1;
        in.data.data_info.data_array_len = sizeof(struct ccu_high_perf_xn);
        int ret = CcuDrvOneShot(*this, phyid, &in, &out);
        if (ret != 0) {
            return ret;
        }
        TEST_LOG_INFO(
            phyid, "GET_HIGH_PERF_XN ok: start_id=%u xn_size=%u",
            out.data.data_info.data_array[0].high_perf_xn.start_id,
            out.data.data_info.data_array[0].high_perf_xn.xn_size);
        return 0;
    });
}

/* Ported from hlt_ccu_set_tranm_ntf_hval: one full chain around
 * SET_TRANM_NTF_HVAL = 1. */
TEST_F(HccpSystemTest, ccuDriverSetTranmNtfHval)
{
    RunTests({1}, [this](int phyid) -> int {
        struct channel_info_in in = {};
        struct channel_info_out out = {};
        in.op = CCU_U_OP_SET_TRANM_NTF_HVAL;
        in.data.data_info.udie_idx = 0;
        in.data.data_info.data_array[0].tranm_ntf_hval = 1;
        int ret = CcuDrvOneShot(*this, phyid, &in, &out);
        if (ret != 0) {
            return ret;
        }
        TEST_LOG_INFO(phyid, "SET_TRANM_NTF_HVAL ok: hval=1");
        return 0;
    });
}

/* Ported from hlt_ccu_set_xn_total_cnt: one full chain per register write -
 * the python CI drove all four compare registers (cnt_index 0..3). */
TEST_F(HccpSystemTest, ccuDriverSetXnTotalCnt)
{
    RunTests({1}, [this](int phyid) -> int {
        static const struct {
            unsigned int cntIndex;
            unsigned int totalAddr;
            unsigned int flagFrom;
            unsigned int flagTo;
        } ci[4] = {
            {0, 1201, 1, 3},
            {1, 1202, 2, 4},
            {2, 1203, 5, 7},
            {3, 1204, 8, 10},
        };
        for (int i = 0; i < 4; i++) {
            struct channel_info_in in = {};
            struct channel_info_out out = {};
            in.op = CCU_U_OP_SET_XN_TOTAL_CNT;
            in.data.data_info.udie_idx = 0;
            in.data.data_info.data_array[0].xn_total_cnt.cnt_index = ci[i].cntIndex;
            in.data.data_info.data_array[0].xn_total_cnt.total_addr = ci[i].totalAddr;
            in.data.data_info.data_array[0].xn_total_cnt.flag_from_addr = ci[i].flagFrom;
            in.data.data_info.data_array[0].xn_total_cnt.flag_to_addr = ci[i].flagTo;
            int ret = CcuDrvOneShot(*this, phyid, &in, &out);
            if (ret != 0) {
                return ret;
            }
            TEST_LOG_INFO(
                phyid, "SET_XN_TOTAL_CNT ok: cnt_index=%u total_addr=%u flag=%u->%u", ci[i].cntIndex, ci[i].totalAddr,
                ci[i].flagFrom, ci[i].flagTo);
        }
        return 0;
    });
}
