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
 * Batch 4 of the test-dev hccp_test migration: the CCU domain of
 * hlt_hccp_ccu_test.c (18 gstool entries, one TEST_F per entry).
 *
 * Every original entry shares one skeleton: config -> TsdProcessOpen ->
 * rts resource init -> RaInit -> RaTlvInit -> MSG_TYPE_CCU_INIT -> one CCU
 * operation through RaCustomChannel -> MSG_TYPE_CCU_UNINIT -> teardown. The
 * migrated mapping of that skeleton is:
 *   - config + TsdProcessOpen + RaInit  -> CaseOfflineInitNoRdv (rtOpenNetService
 *     forks the device process, RaInit offline); the json/cJSON layer is not
 *     ported, the per-case gstool parameters become in-case constants taken
 *     from the original python CI invocations where they exist;
 *   - rts resource init                  -> CcuRtsResourceInit (aclrtSetDevice +
 *     one stream + one notify + notify id -> device resource address, the json
 *     defaults stream_num=1 and notify_num=qp_num=1);
 *   - RaTlvInit / CCU_INIT / op / UNINIT -> the CCU TLV shim
 *     (hccp_ccu_tlv_shim.h): RaCustomChannel does not exist in this repository,
 *     MSG_TYPE_CCU_DISPATCH_CMD carries the identical channel_info_in/out.
 *
 * Topology: the CCU operations are device-local, so the original "both machines
 * run the same entry" model maps to RunTests({1, 5}) with both device processes
 * exercised independently. The three entries the python CI only ever drove on
 * host[0] (init_and_deinit, repeat_init, multi_proc) keep the single-end form
 * RunTests({1}).
 *
 * Original-suite defects fixed here (t1 survey 2.6):
 *   - hlt_hccp_ccu_set_tranm_ntf_hval_test issued its TLV requests through a
 *     NULL tlv_handle (test_tlv_init was never called); the port drives a valid
 *     handle and one CCU_INIT.
 *   - the 32B/64B data builders copied 32/64 bytes out of an 8-byte value
 *     (memcpy_s source overrun); the port fills each 8-byte block of the raw
 *     payload with a defined pattern instead.
 *   - hlt_hccp_ccu_test_ins freed the rtMalloc'ed device buffer with
 *     rtFreeHost; the port frees it with aclrtFree.
 */

#include "cases_common.h"
#include "hccp_ccu_tlv_shim.h"
#include <runtime/runtime/stream.h>
#include <runtime/runtime/event.h>
#include <runtime/runtime/dev.h>
#include <arpa/inet.h>

/* rtGetDevResAddress lives in the rts_device.h / rt_external_device.h families,
 * which cannot be included together with the runtime/runtime headers (same
 * declaration shape the runtime stub uses). The aclrt* entries come from the
 * real <acl/acl_rt.h> via cases_common.h, implemented by the runtime stub. */
extern "C" rtError_t rtGetDevResAddress(rtDevResInfo* const resInfo, rtDevResAddrInfo* const addrInfo);

/* Client-side mirror of the CCU mem-info reply (layout identical to the
 * device-side struct ccu_mem_rsp of the ccu-user-drv stub and to the original
 * hlt_hccp_ccu_test.h definition; ccu_u_comm.h only exports ccu_mem_info). */
struct hccp_test_ccu_mem_rsp {
    unsigned int die_id;
    unsigned int num;
    struct ccu_mem_info list[64U];
};

/* ------------------------------------------------------------------ helpers */

/* Ported shape of test_rts_resource_init (hlt_hccp_ub_common.c): aclrtSetDevice,
 * one stream (json default stream_num = 1) and one notify (default
 * notify_num = qp_num = 1), each notify id resolved to its device resource
 * address through rtGetDevResAddress. */
struct CcuRtsRes {
    aclrtStream stream;
    aclrtNotify notify;
};

static int CcuRtsResourceInit(int phyid, struct CcuRtsRes& rts)
{
    rts.stream = nullptr;
    rts.notify = nullptr;

    if (aclrtSetDevice(phyid) != 0) {
        TEST_LOG_ERROR(phyid, "aclrtSetDevice failed");
        return -1;
    }
    if (aclrtCreateStream(&rts.stream) != 0) {
        TEST_LOG_ERROR(phyid, "aclrtCreateStream failed");
        return -1;
    }
    if (aclrtCreateNotify(&rts.notify, 0) != 0) {
        TEST_LOG_ERROR(phyid, "aclrtCreateNotify failed");
        return -1;
    }

    unsigned int notifyId = 0;
    if (aclrtGetNotifyId(rts.notify, &notifyId) != 0) {
        TEST_LOG_ERROR(phyid, "aclrtGetNotifyId failed");
        return -1;
    }

    rtDevResInfo resInfo = {};
    uint64_t notifyAddr = 0;
    unsigned int notifyLen = 0;
    rtDevResAddrInfo addrInfo = {};
    addrInfo.resAddress = &notifyAddr;
    addrInfo.len = &notifyLen;
    resInfo.dieId = 0;
    resInfo.procType = RT_PROCESS_HCCP;
    resInfo.resType = RT_RES_TYPE_STARS_NOTIFY_RECORD;
    resInfo.resId = notifyId;
    if (rtGetDevResAddress(&resInfo, &addrInfo) != RT_ERROR_NONE) {
        TEST_LOG_ERROR(phyid, "rtGetDevResAddress failed");
        return -1;
    }
    if (notifyAddr == 0) {
        TEST_LOG_ERROR(phyid, "rtGetDevResAddress returned a NULL notify address");
        return -1;
    }
    TEST_LOG_INFO(phyid, "rts ready: notifyId=%u notifyAddr=0x%llx", notifyId, (unsigned long long)notifyAddr);
    return 0;
}

/* Ported shape of test_rts_resource_deinit: notify destroy, stream destroy,
 * device reset. */
static void CcuRtsResourceDeinit(int phyid, struct CcuRtsRes& rts)
{
    if (rts.notify != nullptr) {
        (void)aclrtDestroyNotify(rts.notify);
    }
    if (rts.stream != nullptr) {
        (void)aclrtDestroyStream(rts.stream);
    }
    (void)aclrtResetDevice(phyid);
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

/* -------------------------------------------------------------------- cases */

/* Ported from hlt_hccp_ccu_get_baseinfo_test: GET_BASIC_INFO + GET_VERSION per
 * die. Beyond the original return-code checks the driver-stamped fields
 * (token_valid / missionKey / resourceAddr) are asserted non-empty, same as the
 * batch-0 ccuTlvSmoke. The GET_VERSION request carries data_array_size = 1 so
 * the version entry is actually served. */
TEST_F(HccpSystemTest, ccuGetBaseinfo)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        CaseEnv env;
        struct CcuRtsRes rts;
        void* tlvHandle = nullptr;
        struct channel_info_in in = {};
        struct channel_info_out out = {};
        int ret = CaseOfflineInitNoRdv(env, phyid, HCCP_VM_IFACE_SOCKET);
        if (ret != 0) {
            return ret;
        }
        ret = CcuRtsResourceInit(phyid, rts);
        if (ret != 0) {
            goto err;
        }
        ret = HccpTlvCcuInit(phyid, &tlvHandle);
        if (ret != 0) {
            goto err_rts;
        }
        ret = HccpTlvCcuRequestVoid(phyid, tlvHandle, MSG_TYPE_CCU_INIT);
        if (ret != 0) {
            goto err_deinit;
        }

        in.op = CCU_U_OP_GET_BASIC_INFO;
        in.data.data_info.udie_idx = 0;
        in.data.data_info.data_array_size = 1;
        in.data.data_info.data_array_len = sizeof(struct ccu_baseinfo);
        ret = CcuDispatch(phyid, tlvHandle, &in, &out);
        if (ret != 0) {
            goto err_deinit;
        }
        {
            const struct ccu_baseinfo* info = &out.data.data_info.data_array[0].baseinfo;
            TEST_LOG_INFO(
                phyid, "baseinfo: ms_id=%u missionKey=0x%X resourceAddr=%p", info->ms_id, info->missionKey,
                info->resourceAddr);
            if (info->token_valid == 0 || info->missionKey == 0 || info->resourceAddr == nullptr) {
                TEST_LOG_ERROR(
                    phyid, "GET_BASIC_INFO fields not filled: token_valid=%u missionKey=0x%X res=%p", info->token_valid,
                    info->missionKey, info->resourceAddr);
                ret = -1;
                goto err_deinit;
            }
        }

        in = {};
        in.op = CCU_U_OP_GET_VERSION;
        in.data.data_info.udie_idx = 0;
        in.data.data_info.data_array_size = 1;
        in.data.data_info.data_array_len = sizeof(unsigned int);
        ret = CcuDispatch(phyid, tlvHandle, &in, &out);
        if (ret != 0) {
            goto err_deinit;
        }
        if (out.data.data_info.data_array[0].version != CCU_V2) {
            TEST_LOG_ERROR(
                phyid, "GET_VERSION returned %d, expected CCU_V2", (int)out.data.data_info.data_array[0].version);
            ret = -1;
            goto err_deinit;
        }
        TEST_LOG_INFO(phyid, "GET_VERSION ok: CCU_V2");

        ret = 0;
    err_deinit:
        (void)HccpTlvCcuRequestVoid(phyid, tlvHandle, MSG_TYPE_CCU_UNINIT);
        (void)HccpTlvCcuDeinit(phyid, tlvHandle);
    err_rts:
        CcuRtsResourceDeinit(phyid, rts);
    err:
        CaseCleanup(env);
        return ret;
    });
}

/* Ported from hlt_hccp_ccu_set_task_tif_test: the task/TIF-table channel entry.
 * The original takes the opcode as a gstool parameter (rejecting only
 * SET_INSTRUCTION); the TIF-family opcode without a dedicated entry in this
 * file is CLEAN_TIF_TABLE, so that is the representative operation driven
 * here. */
TEST_F(HccpSystemTest, ccuSetTaskTif)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        CaseEnv env;
        struct CcuRtsRes rts;
        void* tlvHandle = nullptr;
        struct channel_info_in in = {};
        struct channel_info_out out = {};
        int ret = CaseOfflineInitNoRdv(env, phyid, HCCP_VM_IFACE_SOCKET);
        if (ret != 0) {
            return ret;
        }
        ret = CcuRtsResourceInit(phyid, rts);
        if (ret != 0) {
            goto err;
        }
        ret = HccpTlvCcuInit(phyid, &tlvHandle);
        if (ret != 0) {
            goto err_rts;
        }
        ret = HccpTlvCcuRequestVoid(phyid, tlvHandle, MSG_TYPE_CCU_INIT);
        if (ret != 0) {
            goto err_deinit;
        }

        in.op = CCU_U_OP_CLEAN_TIF_TABLE;
        in.data.data_info.udie_idx = 0;
        ret = CcuDispatch(phyid, tlvHandle, &in, &out);
        if (ret != 0) {
            goto err_deinit;
        }
        TEST_LOG_INFO(phyid, "CLEAN_TIF_TABLE ok");

        ret = 0;
    err_deinit:
        (void)HccpTlvCcuRequestVoid(phyid, tlvHandle, MSG_TYPE_CCU_UNINIT);
        (void)HccpTlvCcuDeinit(phyid, tlvHandle);
    err_rts:
        CcuRtsResourceDeinit(phyid, rts);
    err:
        CaseCleanup(env);
        return ret;
    });
}

/* Ported from hlt_hccp_ccu_test_8B: the 8-byte CCU resource path. The original
 * drives one gstool-selected 8B opcode per invocation; this port drives the
 * SET_GSA/GET_GSA pair (the 8B family the driver CI exercised) and, thanks to
 * the stateful ccu-user-drv stub, asserts the read-back payload, which the
 * original could only log. The payload is the defined 8-byte pattern instead
 * of the original's 8-byte-source memcpy. */
TEST_F(HccpSystemTest, ccuTest8B)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        CaseEnv env;
        struct CcuRtsRes rts;
        void* tlvHandle = nullptr;
        struct channel_info_in in = {};
        struct channel_info_out out = {};
        const unsigned long long payload = 0x1122334455667788ULL;
        const unsigned int offsetStart = 2; /* python CI start_entry for the 8B family */
        int ret = CaseOfflineInitNoRdv(env, phyid, HCCP_VM_IFACE_SOCKET);
        if (ret != 0) {
            return ret;
        }
        ret = CcuRtsResourceInit(phyid, rts);
        if (ret != 0) {
            goto err;
        }
        ret = HccpTlvCcuInit(phyid, &tlvHandle);
        if (ret != 0) {
            goto err_rts;
        }
        ret = HccpTlvCcuRequestVoid(phyid, tlvHandle, MSG_TYPE_CCU_INIT);
        if (ret != 0) {
            goto err_deinit;
        }

        in.op = CCU_U_OP_SET_GSA;
        in.offset_start = offsetStart;
        in.data.data_info.udie_idx = 0;
        in.data.data_info.data_array_size = 1;
        in.data.data_info.data_array_len = 8;
        (void)memcpy(in.data.data_info.data_array[0].byte8.raw, &payload, sizeof(payload));
        ret = CcuDispatch(phyid, tlvHandle, &in, &out);
        if (ret != 0) {
            goto err_deinit;
        }

        in = {};
        in.op = CCU_U_OP_GET_GSA;
        in.offset_start = offsetStart;
        in.data.data_info.udie_idx = 0;
        in.data.data_info.data_array_size = 1;
        in.data.data_info.data_array_len = 8;
        ret = CcuDispatch(phyid, tlvHandle, &in, &out);
        if (ret != 0) {
            goto err_deinit;
        }
        {
            unsigned long long got = 0;
            (void)memcpy(&got, out.data.data_info.data_array[0].byte8.raw, sizeof(got));
            if (got != payload) {
                TEST_LOG_ERROR(phyid, "GSA 8B readback mismatch: set=0x%llx get=0x%llx", payload, got);
                ret = -1;
                goto err_deinit;
            }
            if (out.offset_next != offsetStart + 1) {
                TEST_LOG_ERROR(phyid, "GSA 8B offset_next=%u, expected %u", out.offset_next, offsetStart + 1);
                ret = -1;
                goto err_deinit;
            }
        }
        TEST_LOG_INFO(phyid, "GSA 8B set/get readback ok: 0x%llx", payload);

        ret = 0;
    err_deinit:
        (void)HccpTlvCcuRequestVoid(phyid, tlvHandle, MSG_TYPE_CCU_UNINIT);
        (void)HccpTlvCcuDeinit(phyid, tlvHandle);
    err_rts:
        CcuRtsResourceDeinit(phyid, rts);
    err:
        CaseCleanup(env);
        return ret;
    });
}

/* Ported from hlt_hccp_ccu_test_ins: SET_INSTRUCTION with a device buffer.
 * The instruction address comes from aclrtMalloc + aclrtMemcpy(H2D) of the 8-byte
 * instruction value, exactly the original shape. Original defect fixed: the
 * buffer is released with aclrtFree, not rtFreeHost. */
TEST_F(HccpSystemTest, ccuTestIns)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        CaseEnv env;
        struct CcuRtsRes rts;
        void* tlvHandle = nullptr;
        void* devMem = nullptr;
        struct channel_info_in in = {};
        struct channel_info_out out = {};
        const unsigned long long insData = 0xA5A5A5A5UL;
        const unsigned int offsetStart = 0;
        int ret = CaseOfflineInitNoRdv(env, phyid, HCCP_VM_IFACE_SOCKET);
        if (ret != 0) {
            return ret;
        }
        ret = CcuRtsResourceInit(phyid, rts);
        if (ret != 0) {
            goto err;
        }
        ret = HccpTlvCcuInit(phyid, &tlvHandle);
        if (ret != 0) {
            goto err_rts;
        }
        ret = HccpTlvCcuRequestVoid(phyid, tlvHandle, MSG_TYPE_CCU_INIT);
        if (ret != 0) {
            goto err_deinit;
        }

        if (aclrtMalloc(&devMem, 32, ACL_MEM_MALLOC_HUGE_FIRST) != 0) {
            TEST_LOG_ERROR(phyid, "aclrtMalloc failed");
            ret = -1;
            goto err_deinit;
        }
        if (aclrtMemcpy(devMem, sizeof(insData), &insData, sizeof(insData), ACL_MEMCPY_HOST_TO_DEVICE) != 0) {
            TEST_LOG_ERROR(phyid, "aclrtMemcpy H2D failed");
            ret = -1;
            goto err_free;
        }

        in.op = CCU_U_OP_SET_INSTRUCTION;
        in.offset_start = offsetStart;
        in.data.data_info.udie_idx = 0;
        in.data.data_info.data_array_size = 1;
        in.data.data_info.data_array_len = 32;
        in.data.data_info.data_array[0].insinfo.resourceAddr = (unsigned long)(uintptr_t)devMem;
        TEST_LOG_INFO(
            phyid, "SET_INSTRUCTION: resourceAddr=0x%lx", in.data.data_info.data_array[0].insinfo.resourceAddr);
        ret = CcuDispatch(phyid, tlvHandle, &in, &out);
        if (ret != 0) {
            goto err_free;
        }

        ret = 0;
    err_free:
        if (devMem != nullptr) {
            (void)aclrtFree(devMem); /* original freed with rtFreeHost (wrong API) */
        }
    err_deinit:
        (void)HccpTlvCcuRequestVoid(phyid, tlvHandle, MSG_TYPE_CCU_UNINIT);
        (void)HccpTlvCcuDeinit(phyid, tlvHandle);
    err_rts:
        CcuRtsResourceDeinit(phyid, rts);
    err:
        CaseCleanup(env);
        return ret;
    });
}

/* Ported from hlt_hccp_ccu_test_32B: the 32-byte CCU resource path, driven as
 * the SET_JETTY_CTX/GET_JETTY_CTX pair with a set->get readback. Original
 * defect fixed: the payload fills all four 8-byte blocks with a defined
 * pattern instead of memcpy'ing 32 bytes out of an 8-byte value. */
TEST_F(HccpSystemTest, ccuTest32B)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        CaseEnv env;
        struct CcuRtsRes rts;
        void* tlvHandle = nullptr;
        struct channel_info_in in = {};
        struct channel_info_out out = {};
        const unsigned int offsetStart = 8; /* python CI start_entry for jetty ctx */
        int ret = CaseOfflineInitNoRdv(env, phyid, HCCP_VM_IFACE_SOCKET);
        if (ret != 0) {
            return ret;
        }
        ret = CcuRtsResourceInit(phyid, rts);
        if (ret != 0) {
            goto err;
        }
        ret = HccpTlvCcuInit(phyid, &tlvHandle);
        if (ret != 0) {
            goto err_rts;
        }
        ret = HccpTlvCcuRequestVoid(phyid, tlvHandle, MSG_TYPE_CCU_INIT);
        if (ret != 0) {
            goto err_deinit;
        }

        in.op = CCU_U_OP_SET_JETTY_CTX;
        in.offset_start = offsetStart;
        in.data.data_info.udie_idx = 0;
        in.data.data_info.data_array_size = 1;
        in.data.data_info.data_array_len = 32;
        for (int blk = 0; blk < 4; blk++) {
            unsigned long long v = 0x1122334455667788ULL + (unsigned long long)blk;
            (void)memcpy(&in.data.data_info.data_array[0].byte32.raw[blk * 8], &v, sizeof(v));
        }
        ret = CcuDispatch(phyid, tlvHandle, &in, &out);
        if (ret != 0) {
            goto err_deinit;
        }

        in = {};
        in.op = CCU_U_OP_GET_JETTY_CTX;
        in.offset_start = offsetStart;
        in.data.data_info.udie_idx = 0;
        in.data.data_info.data_array_size = 1;
        in.data.data_info.data_array_len = 32;
        ret = CcuDispatch(phyid, tlvHandle, &in, &out);
        if (ret != 0) {
            goto err_deinit;
        }
        for (int blk = 0; blk < 4; blk++) {
            unsigned long long got = 0;
            (void)memcpy(&got, &out.data.data_info.data_array[0].byte32.raw[blk * 8], sizeof(got));
            if (got != 0x1122334455667788ULL + (unsigned long long)blk) {
                TEST_LOG_ERROR(phyid, "JETTY_CTX 32B readback mismatch blk=%d: get=0x%llx", blk, got);
                ret = -1;
                goto err_deinit;
            }
        }
        TEST_LOG_INFO(phyid, "JETTY_CTX 32B set/get readback ok");

        ret = 0;
    err_deinit:
        (void)HccpTlvCcuRequestVoid(phyid, tlvHandle, MSG_TYPE_CCU_UNINIT);
        (void)HccpTlvCcuDeinit(phyid, tlvHandle);
    err_rts:
        CcuRtsResourceDeinit(phyid, rts);
    err:
        CaseCleanup(env);
        return ret;
    });
}

/* Ported from hlt_hccp_ccu_test_64B: the 64-byte CCU resource path, driven as
 * the SET_CHANNEL/GET_CHANNEL pair with a set->get readback. Original defect
 * fixed: the payload fills all eight 8-byte blocks with a defined pattern
 * instead of memcpy'ing 64 bytes out of an 8-byte value. */
TEST_F(HccpSystemTest, ccuTest64B)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        CaseEnv env;
        struct CcuRtsRes rts;
        void* tlvHandle = nullptr;
        struct channel_info_in in = {};
        struct channel_info_out out = {};
        const unsigned int offsetStart = 6; /* python CI start_entry for set_channel */
        int ret = CaseOfflineInitNoRdv(env, phyid, HCCP_VM_IFACE_SOCKET);
        if (ret != 0) {
            return ret;
        }
        ret = CcuRtsResourceInit(phyid, rts);
        if (ret != 0) {
            goto err;
        }
        ret = HccpTlvCcuInit(phyid, &tlvHandle);
        if (ret != 0) {
            goto err_rts;
        }
        ret = HccpTlvCcuRequestVoid(phyid, tlvHandle, MSG_TYPE_CCU_INIT);
        if (ret != 0) {
            goto err_deinit;
        }

        in.op = CCU_U_OP_SET_CHANNEL;
        in.offset_start = offsetStart;
        in.data.data_info.udie_idx = 0;
        in.data.data_info.data_array_size = 1;
        in.data.data_info.data_array_len = 64;
        for (int blk = 0; blk < 8; blk++) {
            unsigned long long v = 0x1122334455667788ULL + (unsigned long long)blk;
            (void)memcpy(&in.data.data_info.data_array[0].byte64.raw[blk * 8], &v, sizeof(v));
        }
        ret = CcuDispatch(phyid, tlvHandle, &in, &out);
        if (ret != 0) {
            goto err_deinit;
        }

        in = {};
        in.op = CCU_U_OP_GET_CHANNEL;
        in.offset_start = offsetStart;
        in.data.data_info.udie_idx = 0;
        in.data.data_info.data_array_size = 1;
        in.data.data_info.data_array_len = 64;
        ret = CcuDispatch(phyid, tlvHandle, &in, &out);
        if (ret != 0) {
            goto err_deinit;
        }
        for (int blk = 0; blk < 8; blk++) {
            unsigned long long got = 0;
            (void)memcpy(&got, &out.data.data_info.data_array[0].byte64.raw[blk * 8], sizeof(got));
            if (got != 0x1122334455667788ULL + (unsigned long long)blk) {
                TEST_LOG_ERROR(phyid, "CHANNEL 64B readback mismatch blk=%d: get=0x%llx", blk, got);
                ret = -1;
                goto err_deinit;
            }
        }
        TEST_LOG_INFO(phyid, "CHANNEL 64B set/get readback ok");

        ret = 0;
    err_deinit:
        (void)HccpTlvCcuRequestVoid(phyid, tlvHandle, MSG_TYPE_CCU_UNINIT);
        (void)HccpTlvCcuDeinit(phyid, tlvHandle);
    err_rts:
        CcuRtsResourceDeinit(phyid, rts);
    err:
        CaseCleanup(env);
        return ret;
    });
}

/* Ported from hlt_hccp_ccu_set_chn_jetty_map_test: SET_CHN_JETTY_MAP with the
 * python CI value 15, plus a GET readback so the mapping actually round-trips
 * through the stateful stub. */
TEST_F(HccpSystemTest, ccuSetChnJettyMap)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        CaseEnv env;
        struct CcuRtsRes rts;
        void* tlvHandle = nullptr;
        struct channel_info_in in = {};
        struct channel_info_out out = {};
        const unsigned int chnJettyMap = 15;
        int ret = CaseOfflineInitNoRdv(env, phyid, HCCP_VM_IFACE_SOCKET);
        if (ret != 0) {
            return ret;
        }
        ret = CcuRtsResourceInit(phyid, rts);
        if (ret != 0) {
            goto err;
        }
        ret = HccpTlvCcuInit(phyid, &tlvHandle);
        if (ret != 0) {
            goto err_rts;
        }
        ret = HccpTlvCcuRequestVoid(phyid, tlvHandle, MSG_TYPE_CCU_INIT);
        if (ret != 0) {
            goto err_deinit;
        }

        in.op = CCU_U_OP_SET_CHN_JETTY_MAP;
        in.data.data_info.udie_idx = 0;
        in.data.data_info.data_array[0].baseinfo.chn_jetty_map = chnJettyMap;
        ret = CcuDispatch(phyid, tlvHandle, &in, &out);
        if (ret != 0) {
            goto err_deinit;
        }
        TEST_LOG_INFO(phyid, "SET_CHN_JETTY_MAP ok: %u", chnJettyMap);

        in = {};
        in.op = CCU_U_OP_GET_CHN_JETTY_MAP;
        in.data.data_info.udie_idx = 0;
        ret = CcuDispatch(phyid, tlvHandle, &in, &out);
        if (ret != 0) {
            goto err_deinit;
        }
        if (out.data.data_info.data_array[0].baseinfo.chn_jetty_map != chnJettyMap) {
            TEST_LOG_ERROR(
                phyid, "chn_jetty_map readback mismatch: set=%u get=%u", chnJettyMap,
                out.data.data_info.data_array[0].baseinfo.chn_jetty_map);
            ret = -1;
            goto err_deinit;
        }

        ret = 0;
    err_deinit:
        (void)HccpTlvCcuRequestVoid(phyid, tlvHandle, MSG_TYPE_CCU_UNINIT);
        (void)HccpTlvCcuDeinit(phyid, tlvHandle);
    err_rts:
        CcuRtsResourceDeinit(phyid, rts);
    err:
        CaseCleanup(env);
        return ret;
    });
}

/* Ported from hlt_hccp_ccu_get_chn_jetty_map_test: standalone GET_CHN_JETTY_MAP
 * (no prior SET in this entry, so the value is only logged, like the
 * original). */
TEST_F(HccpSystemTest, ccuGetChnJettyMap)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        CaseEnv env;
        struct CcuRtsRes rts;
        void* tlvHandle = nullptr;
        struct channel_info_in in = {};
        struct channel_info_out out = {};
        int ret = CaseOfflineInitNoRdv(env, phyid, HCCP_VM_IFACE_SOCKET);
        if (ret != 0) {
            return ret;
        }
        ret = CcuRtsResourceInit(phyid, rts);
        if (ret != 0) {
            goto err;
        }
        ret = HccpTlvCcuInit(phyid, &tlvHandle);
        if (ret != 0) {
            goto err_rts;
        }
        ret = HccpTlvCcuRequestVoid(phyid, tlvHandle, MSG_TYPE_CCU_INIT);
        if (ret != 0) {
            goto err_deinit;
        }

        in.op = CCU_U_OP_GET_CHN_JETTY_MAP;
        in.data.data_info.udie_idx = 0;
        ret = CcuDispatch(phyid, tlvHandle, &in, &out);
        if (ret != 0) {
            goto err_deinit;
        }
        TEST_LOG_INFO(phyid, "GET_CHN_JETTY_MAP: %u", out.data.data_info.data_array[0].baseinfo.chn_jetty_map);

        ret = 0;
    err_deinit:
        (void)HccpTlvCcuRequestVoid(phyid, tlvHandle, MSG_TYPE_CCU_UNINIT);
        (void)HccpTlvCcuDeinit(phyid, tlvHandle);
    err_rts:
        CcuRtsResourceDeinit(phyid, rts);
    err:
        CaseCleanup(env);
        return ret;
    });
}

/* Ported from hlt_hccp_ccu_set_tif_split_size_test: SET_TIF_SPLIT_SIZE with
 * split_pkt_unit = 0 (unit: byte). */
TEST_F(HccpSystemTest, ccuSetTifSplitSize)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        CaseEnv env;
        struct CcuRtsRes rts;
        void* tlvHandle = nullptr;
        struct channel_info_in in = {};
        struct channel_info_out out = {};
        int ret = CaseOfflineInitNoRdv(env, phyid, HCCP_VM_IFACE_SOCKET);
        if (ret != 0) {
            return ret;
        }
        ret = CcuRtsResourceInit(phyid, rts);
        if (ret != 0) {
            goto err;
        }
        ret = HccpTlvCcuInit(phyid, &tlvHandle);
        if (ret != 0) {
            goto err_rts;
        }
        ret = HccpTlvCcuRequestVoid(phyid, tlvHandle, MSG_TYPE_CCU_INIT);
        if (ret != 0) {
            goto err_deinit;
        }

        in.op = CCU_U_OP_SET_TIF_SPLIT_SIZE;
        in.data.data_info.udie_idx = 0;
        in.data.data_info.data_array[0].tif_split_info.split_pkt_unit = 0; /* 0: byte */
        ret = CcuDispatch(phyid, tlvHandle, &in, &out);
        if (ret != 0) {
            goto err_deinit;
        }
        TEST_LOG_INFO(phyid, "SET_TIF_SPLIT_SIZE ok: split_pkt_unit=0 (byte)");

        ret = 0;
    err_deinit:
        (void)HccpTlvCcuRequestVoid(phyid, tlvHandle, MSG_TYPE_CCU_UNINIT);
        (void)HccpTlvCcuDeinit(phyid, tlvHandle);
    err_rts:
        CcuRtsResourceDeinit(phyid, rts);
    err:
        CaseCleanup(env);
        return ret;
    });
}

/* Ported from hlt_hccp_ccu_get_high_perf_xn_test: GET_HIGH_PERF_XN; the
 * register range values are logged like the original (no prior SET exists for
 * this read-only register range). */
TEST_F(HccpSystemTest, ccuGetHighPerfXn)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        CaseEnv env;
        struct CcuRtsRes rts;
        void* tlvHandle = nullptr;
        struct channel_info_in in = {};
        struct channel_info_out out = {};
        int ret = CaseOfflineInitNoRdv(env, phyid, HCCP_VM_IFACE_SOCKET);
        if (ret != 0) {
            return ret;
        }
        ret = CcuRtsResourceInit(phyid, rts);
        if (ret != 0) {
            goto err;
        }
        ret = HccpTlvCcuInit(phyid, &tlvHandle);
        if (ret != 0) {
            goto err_rts;
        }
        ret = HccpTlvCcuRequestVoid(phyid, tlvHandle, MSG_TYPE_CCU_INIT);
        if (ret != 0) {
            goto err_deinit;
        }

        in.op = CCU_U_OP_GET_HIGH_PERF_XN;
        in.data.data_info.udie_idx = 0;
        in.data.data_info.data_array_size = 1;
        in.data.data_info.data_array_len = sizeof(struct ccu_high_perf_xn);
        ret = CcuDispatch(phyid, tlvHandle, &in, &out);
        if (ret != 0) {
            goto err_deinit;
        }
        TEST_LOG_INFO(
            phyid, "GET_HIGH_PERF_XN: start_id=%u xn_size=%u", out.data.data_info.data_array[0].high_perf_xn.start_id,
            out.data.data_info.data_array[0].high_perf_xn.xn_size);

        ret = 0;
    err_deinit:
        (void)HccpTlvCcuRequestVoid(phyid, tlvHandle, MSG_TYPE_CCU_UNINIT);
        (void)HccpTlvCcuDeinit(phyid, tlvHandle);
    err_rts:
        CcuRtsResourceDeinit(phyid, rts);
    err:
        CaseCleanup(env);
        return ret;
    });
}

/* Ported from hlt_hccp_ccu_set_xn_total_cnt_test: four consecutive
 * SET_XN_TOTAL_CNT writes (cnt_index 0..3) with the python CI register
 * values. */
TEST_F(HccpSystemTest, ccuSetXnTotalCnt)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        CaseEnv env;
        struct CcuRtsRes rts;
        void* tlvHandle = nullptr;
        struct channel_info_in in = {};
        struct channel_info_out out = {};
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
        int ret = CaseOfflineInitNoRdv(env, phyid, HCCP_VM_IFACE_SOCKET);
        if (ret != 0) {
            return ret;
        }
        ret = CcuRtsResourceInit(phyid, rts);
        if (ret != 0) {
            goto err;
        }
        ret = HccpTlvCcuInit(phyid, &tlvHandle);
        if (ret != 0) {
            goto err_rts;
        }
        ret = HccpTlvCcuRequestVoid(phyid, tlvHandle, MSG_TYPE_CCU_INIT);
        if (ret != 0) {
            goto err_deinit;
        }

        for (int i = 0; i < 4; i++) {
            in = {};
            in.op = CCU_U_OP_SET_XN_TOTAL_CNT;
            in.data.data_info.udie_idx = 0;
            in.data.data_info.data_array[0].xn_total_cnt.cnt_index = ci[i].cntIndex;
            in.data.data_info.data_array[0].xn_total_cnt.total_addr = ci[i].totalAddr;
            in.data.data_info.data_array[0].xn_total_cnt.flag_from_addr = ci[i].flagFrom;
            in.data.data_info.data_array[0].xn_total_cnt.flag_to_addr = ci[i].flagTo;
            ret = CcuDispatch(phyid, tlvHandle, &in, &out);
            if (ret != 0) {
                goto err_deinit;
            }
            TEST_LOG_INFO(
                phyid, "SET_XN_TOTAL_CNT ok: cnt_index=%u total_addr=%u flag=%u->%u", ci[i].cntIndex, ci[i].totalAddr,
                ci[i].flagFrom, ci[i].flagTo);
        }

        ret = 0;
    err_deinit:
        (void)HccpTlvCcuRequestVoid(phyid, tlvHandle, MSG_TYPE_CCU_UNINIT);
        (void)HccpTlvCcuDeinit(phyid, tlvHandle);
    err_rts:
        CcuRtsResourceDeinit(phyid, rts);
    err:
        CaseCleanup(env);
        return ret;
    });
}

/* Ported from hlt_hccp_ccu_set_tranm_ntf_hval_test: SET_TRANM_NTF_HVAL = 1.
 * Original defect fixed: the original never called test_tlv_init and pushed its
 * two CCU_INIT requests through the NULL tlv_handle (guaranteed crash on real
 * hardware too); the port drives a valid handle with a single CCU_INIT. */
TEST_F(HccpSystemTest, ccuSetTranmNtfHval)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        CaseEnv env;
        struct CcuRtsRes rts;
        void* tlvHandle = nullptr;
        struct channel_info_in in = {};
        struct channel_info_out out = {};
        int ret = CaseOfflineInitNoRdv(env, phyid, HCCP_VM_IFACE_SOCKET);
        if (ret != 0) {
            return ret;
        }
        ret = CcuRtsResourceInit(phyid, rts);
        if (ret != 0) {
            goto err;
        }
        ret = HccpTlvCcuInit(phyid, &tlvHandle);
        if (ret != 0) {
            goto err_rts;
        }
        ret = HccpTlvCcuRequestVoid(phyid, tlvHandle, MSG_TYPE_CCU_INIT);
        if (ret != 0) {
            goto err_deinit;
        }

        in.op = CCU_U_OP_SET_TRANM_NTF_HVAL;
        in.data.data_info.udie_idx = 0;
        in.data.data_info.data_array[0].tranm_ntf_hval = 1;
        ret = CcuDispatch(phyid, tlvHandle, &in, &out);
        if (ret != 0) {
            goto err_deinit;
        }
        TEST_LOG_INFO(phyid, "SET_TRANM_NTF_HVAL ok: hval=1");

        ret = 0;
    err_deinit:
        (void)HccpTlvCcuRequestVoid(phyid, tlvHandle, MSG_TYPE_CCU_UNINIT);
        (void)HccpTlvCcuDeinit(phyid, tlvHandle);
    err_rts:
        CcuRtsResourceDeinit(phyid, rts);
    err:
        CaseCleanup(env);
        return ret;
    });
}

/* Ported from hlt_hccp_ccu_init_and_deinit_test: the TLV-only lifecycle
 * (RaTlvInit -> CCU_INIT -> CCU_UNINIT -> RaTlvDeinit). The python CI drove
 * this entry on host[0] only, so it keeps the single-end form. */
TEST_F(HccpSystemTest, ccuInitAndDeinit)
{
    RunTests({1}, [this](int phyid) -> int {
        CaseEnv env;
        struct CcuRtsRes rts;
        void* tlvHandle = nullptr;
        int ret = CaseOfflineInitNoRdv(env, phyid, HCCP_VM_IFACE_SOCKET);
        if (ret != 0) {
            return ret;
        }
        ret = CcuRtsResourceInit(phyid, rts);
        if (ret != 0) {
            goto err;
        }
        ret = HccpTlvCcuInit(phyid, &tlvHandle);
        if (ret != 0) {
            goto err_rts;
        }
        ret = HccpTlvCcuRequestVoid(phyid, tlvHandle, MSG_TYPE_CCU_INIT);
        if (ret != 0) {
            goto err_deinit;
        }
        TEST_LOG_INFO(phyid, "CCU init ok");

        ret = HccpTlvCcuRequestVoid(phyid, tlvHandle, MSG_TYPE_CCU_UNINIT);
        if (ret != 0) {
            goto err_deinit;
        }
        TEST_LOG_INFO(phyid, "CCU uninit ok");

        ret = HccpTlvCcuDeinit(phyid, tlvHandle);
        tlvHandle = nullptr;
        if (ret != 0) {
            goto err_rts;
        }
        CcuRtsResourceDeinit(phyid, rts);
        CaseCleanup(env);
        return 0;
    err_deinit:
        (void)HccpTlvCcuRequestVoid(phyid, tlvHandle, MSG_TYPE_CCU_UNINIT);
        (void)HccpTlvCcuDeinit(phyid, tlvHandle);
    err_rts:
        CcuRtsResourceDeinit(phyid, rts);
    err:
        CaseCleanup(env);
        return ret;
    });
}

/* Ported from hlt_hccp_ccu_repeat_init_test: CCU_INIT issued twice on the same
 * TLV session, then one CCU_UNINIT (the original end-block always sends exactly
 * one UNINIT). The python CI drove this entry on host[0] only. */
TEST_F(HccpSystemTest, ccuRepeatInit)
{
    RunTests({1}, [this](int phyid) -> int {
        CaseEnv env;
        struct CcuRtsRes rts;
        void* tlvHandle = nullptr;
        int ret = CaseOfflineInitNoRdv(env, phyid, HCCP_VM_IFACE_SOCKET);
        if (ret != 0) {
            return ret;
        }
        ret = CcuRtsResourceInit(phyid, rts);
        if (ret != 0) {
            goto err;
        }
        ret = HccpTlvCcuInit(phyid, &tlvHandle);
        if (ret != 0) {
            goto err_rts;
        }
        ret = HccpTlvCcuRequestVoid(phyid, tlvHandle, MSG_TYPE_CCU_INIT);
        if (ret != 0) {
            goto err_deinit;
        }
        TEST_LOG_INFO(phyid, "CCU init #1 ok");
        ret = HccpTlvCcuRequestVoid(phyid, tlvHandle, MSG_TYPE_CCU_INIT);
        if (ret != 0) {
            goto err_deinit;
        }
        TEST_LOG_INFO(phyid, "CCU init #2 ok (repeat)");

        ret = 0;
    err_deinit:
        (void)HccpTlvCcuRequestVoid(phyid, tlvHandle, MSG_TYPE_CCU_UNINIT);
        (void)HccpTlvCcuDeinit(phyid, tlvHandle);
    err_rts:
        CcuRtsResourceDeinit(phyid, rts);
    err:
        CaseCleanup(env);
        return ret;
    });
}

/* Ported from hlt_hccp_ccu_multi_proc_test: the multi-open flavor of the CCU
 * session (test_multi_open_hccp_by_tsd / test_multi_RaInit only add a
 * whitelist parameter and hdcType, which the sim TSD forward ignores). The
 * original slept 5 minutes so a tester could attach more processes manually on
 * real hardware; the sim has no such observation phase, so the sleep is
 * dropped (t1 survey R8) and replaced by what it stood for: a second full
 * service lifetime in the same host child. The second CCU_INIT must observe a
 * fresh device state (the stub wipes its shadow on init), which is asserted
 * through a SET_MSID/GET_MSID round trip in each lifetime. */
TEST_F(HccpSystemTest, ccuMultiProc)
{
    RunTests({1}, [this](int phyid) -> int {
        for (int lifetime = 0; lifetime < 2; lifetime++) {
            CaseEnv env;
            struct CcuRtsRes rts;
            void* tlvHandle = nullptr;
            struct channel_info_in in = {};
            struct channel_info_out out = {};
            const unsigned int msid = 0xB0u + (unsigned int)lifetime;
            int ret = CaseOfflineInitNoRdv(env, phyid, HCCP_VM_IFACE_SOCKET);
            if (ret != 0) {
                return ret;
            }
            ret = CcuRtsResourceInit(phyid, rts);
            if (ret != 0) {
                goto err;
            }
            ret = HccpTlvCcuInit(phyid, &tlvHandle);
            if (ret != 0) {
                goto err_rts;
            }
            ret = HccpTlvCcuRequestVoid(phyid, tlvHandle, MSG_TYPE_CCU_INIT);
            if (ret != 0) {
                goto err_deinit;
            }

            in.op = CCU_U_OP_SET_MSID_TOKEN;
            in.data.data_info.udie_idx = 0;
            in.data.data_info.data_array_size = 1;
            in.data.data_info.data_array_len = sizeof(struct ccu_baseinfo);
            in.data.data_info.data_array[0].baseinfo.ms_id = msid;
            ret = CcuDispatch(phyid, tlvHandle, &in, &out);
            if (ret != 0) {
                goto err_deinit;
            }

            in = {};
            in.op = CCU_U_OP_GET_MSID;
            in.data.data_info.udie_idx = 0;
            in.data.data_info.data_array_size = 1;
            in.data.data_info.data_array_len = sizeof(struct ccu_baseinfo);
            ret = CcuDispatch(phyid, tlvHandle, &in, &out);
            if (ret != 0) {
                goto err_deinit;
            }
            if (out.data.data_info.data_array[0].baseinfo.ms_id != msid) {
                TEST_LOG_ERROR(
                    phyid, "lifetime %d msid readback mismatch: set=0x%X get=0x%X", lifetime, msid,
                    out.data.data_info.data_array[0].baseinfo.ms_id);
                ret = -1;
                goto err_deinit;
            }
            TEST_LOG_INFO(phyid, "multi-open lifetime %d ok: msid=0x%X", lifetime, msid);

            ret = 0;
        err_deinit:
            (void)HccpTlvCcuRequestVoid(phyid, tlvHandle, MSG_TYPE_CCU_UNINIT);
            (void)HccpTlvCcuDeinit(phyid, tlvHandle);
        err_rts:
            CcuRtsResourceDeinit(phyid, rts);
        err:
            CaseCleanup(env);
            if (ret != 0) {
                return ret;
            }
        }
        return 0;
    });
}

/* Ported from hlt_hccp_ccu_set_task_kill: SET_TASKKILL with the single-mission
 * form the driver CI used (is_single_mission = 1, mission_id = 16). */
TEST_F(HccpSystemTest, ccuSetTaskKill)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        CaseEnv env;
        struct CcuRtsRes rts;
        void* tlvHandle = nullptr;
        struct channel_info_in in = {};
        struct channel_info_out out = {};
        int ret = CaseOfflineInitNoRdv(env, phyid, HCCP_VM_IFACE_SOCKET);
        if (ret != 0) {
            return ret;
        }
        ret = CcuRtsResourceInit(phyid, rts);
        if (ret != 0) {
            goto err;
        }
        ret = HccpTlvCcuInit(phyid, &tlvHandle);
        if (ret != 0) {
            goto err_rts;
        }
        ret = HccpTlvCcuRequestVoid(phyid, tlvHandle, MSG_TYPE_CCU_INIT);
        if (ret != 0) {
            goto err_deinit;
        }

        in.op = CCU_U_OP_SET_TASKKILL;
        in.data.data_info.udie_idx = 0;
        in.data.data_info.data_array[0].mission_info.is_single_mission = true;
        in.data.data_info.data_array[0].mission_info.mission_id = 16;
        ret = CcuDispatch(phyid, tlvHandle, &in, &out);
        if (ret != 0) {
            goto err_deinit;
        }
        TEST_LOG_INFO(phyid, "SET_TASKKILL ok: single mission, mission_id=16");

        ret = 0;
    err_deinit:
        (void)HccpTlvCcuRequestVoid(phyid, tlvHandle, MSG_TYPE_CCU_UNINIT);
        (void)HccpTlvCcuDeinit(phyid, tlvHandle);
    err_rts:
        CcuRtsResourceDeinit(phyid, rts);
    err:
        CaseCleanup(env);
        return ret;
    });
}

/* Ported from hlt_hccp_ccu_clean_task_kill_state: CLEAN_TASKKILL_STATE with
 * the single-mission form. */
TEST_F(HccpSystemTest, ccuCleanTaskKillState)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        CaseEnv env;
        struct CcuRtsRes rts;
        void* tlvHandle = nullptr;
        struct channel_info_in in = {};
        struct channel_info_out out = {};
        int ret = CaseOfflineInitNoRdv(env, phyid, HCCP_VM_IFACE_SOCKET);
        if (ret != 0) {
            return ret;
        }
        ret = CcuRtsResourceInit(phyid, rts);
        if (ret != 0) {
            goto err;
        }
        ret = HccpTlvCcuInit(phyid, &tlvHandle);
        if (ret != 0) {
            goto err_rts;
        }
        ret = HccpTlvCcuRequestVoid(phyid, tlvHandle, MSG_TYPE_CCU_INIT);
        if (ret != 0) {
            goto err_deinit;
        }

        in.op = CCU_U_OP_CLEAN_TASKKILL_STATE;
        in.data.data_info.udie_idx = 0;
        in.data.data_info.data_array[0].mission_info.is_single_mission = true;
        in.data.data_info.data_array[0].mission_info.mission_id = 16;
        ret = CcuDispatch(phyid, tlvHandle, &in, &out);
        if (ret != 0) {
            goto err_deinit;
        }
        TEST_LOG_INFO(phyid, "CLEAN_TASKKILL_STATE ok: mission_id=16");

        ret = 0;
    err_deinit:
        (void)HccpTlvCcuRequestVoid(phyid, tlvHandle, MSG_TYPE_CCU_UNINIT);
        (void)HccpTlvCcuDeinit(phyid, tlvHandle);
    err_rts:
        CcuRtsResourceDeinit(phyid, rts);
    err:
        CaseCleanup(env);
        return ret;
    });
}

/* Ported from hlt_hccp_ccu_get_mem_info: MSG_TYPE_CCU_GET_MEM_INFO with a
 * multi-type bitmap; the reply must list one mapped region per requested type
 * (asserted, where the original only printed). */
TEST_F(HccpSystemTest, ccuGetMemInfo)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        CaseEnv env;
        struct CcuRtsRes rts;
        void* tlvHandle = nullptr;
        struct CcuMemReq req = {};
        struct hccp_test_ccu_mem_rsp rsp = {};
        struct TlvMsg sendMsg = {};
        struct TlvMsg recvMsg = {};
        const unsigned long long bitmap
            = CCU_MEMTYPE_INS | CCU_MEMTYPE_GSA | CCU_MEMTYPE_XN | CCU_MEMTYPE_CKE | CCU_MEMTYPE_PFE | CCU_MEMTYPE_CHN;
        const unsigned int wantNum = 6;
        int ret = CaseOfflineInitNoRdv(env, phyid, HCCP_VM_IFACE_SOCKET);
        if (ret != 0) {
            return ret;
        }
        ret = CcuRtsResourceInit(phyid, rts);
        if (ret != 0) {
            goto err;
        }
        ret = HccpTlvCcuInit(phyid, &tlvHandle);
        if (ret != 0) {
            goto err_rts;
        }
        ret = HccpTlvCcuRequestVoid(phyid, tlvHandle, MSG_TYPE_CCU_INIT);
        if (ret != 0) {
            goto err_deinit;
        }

        req.udieIdx = 0;
        req.memTypeBitmap = bitmap;
        sendMsg.type = MSG_TYPE_CCU_GET_MEM_INFO;
        sendMsg.length = sizeof(struct CcuMemReq);
        sendMsg.data = (char*)&req;
        recvMsg.length = sizeof(struct hccp_test_ccu_mem_rsp);
        recvMsg.data = (char*)&rsp;
        ret = RaTlvRequest(tlvHandle, TLV_MODULE_TYPE_CCU, &sendMsg, &recvMsg);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaTlvRequest GET_MEM_INFO failed ret=%d", ret);
            goto err_deinit;
        }
        if (rsp.num != wantNum) {
            TEST_LOG_ERROR(phyid, "mem info num=%u, expected %u", rsp.num, wantNum);
            ret = -1;
            goto err_deinit;
        }
        for (unsigned int i = 0; i < rsp.num; i++) {
            if (rsp.list[i].mem_va == 0 || rsp.list[i].mem_size == 0) {
                TEST_LOG_ERROR(
                    phyid, "mem region %u not filled: va=0x%llx size=%u", i, rsp.list[i].mem_va, rsp.list[i].mem_size);
                ret = -1;
                goto err_deinit;
            }
            TEST_LOG_INFO(phyid, "mem region %u: va=0x%llx size=%u", i, rsp.list[i].mem_va, rsp.list[i].mem_size);
        }

        ret = 0;
    err_deinit:
        (void)HccpTlvCcuRequestVoid(phyid, tlvHandle, MSG_TYPE_CCU_UNINIT);
        (void)HccpTlvCcuDeinit(phyid, tlvHandle);
    err_rts:
        CcuRtsResourceDeinit(phyid, rts);
    err:
        CaseCleanup(env);
        return ret;
    });
}
