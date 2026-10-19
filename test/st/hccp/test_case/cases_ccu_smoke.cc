/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cases_common.h"
#include "hccp_ccu_tlv_shim.h"

/*
 * Batch-0 smoke of the CCU TLV channel (the base the migrated hlt_ccu_* cases
 * build on). Both host children run the same offline flow: open the per-device
 * service, RaInit, RaTlvInit, CCU_INIT, then a SET_MSID_TOKEN -> GET_MSID
 * round trip through the TLV dispatch shim, and a GET_BASIC_INFO sanity check.
 * The msid value is derived from the local phyId, so a shared or leaked device
 * state between the two ends cannot satisfy both assertions.
 */

TEST_F(HccpSystemTest, ccuTlvSmoke)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        (void)setvbuf(stdout, nullptr, _IONBF, 0);
        CaseEnv env;
        void* tlvHandle = nullptr;
        struct channel_info_in in = {};
        struct channel_info_out out = {};
        struct channel_info_in inGet = {};
        struct channel_info_out outGet = {};
        struct channel_info_in inInfo = {};
        struct channel_info_out outInfo = {};
        const unsigned int msid = 0xA5u + (unsigned int)phyid;
        const unsigned int tokenId = 0x10u + (unsigned int)phyid;
        const unsigned int tokenValue = 0x20u + (unsigned int)phyid;
        unsigned int gotMsid = 0;
        struct ccu_baseinfo* info = nullptr;
        int ret = 0;

        ret = CaseOfflineInit(env, phyid, HCCP_VM_IFACE_SOCKET, 15950, "CCU TLV SMOKE");
        if (ret != 0) {
            return ret;
        }

        ret = HccpTlvCcuInit(phyid, &tlvHandle);
        if (ret != 0) {
            goto err;
        }
        ret = HccpTlvCcuRequestVoid(phyid, tlvHandle, MSG_TYPE_CCU_INIT);
        if (ret != 0) {
            goto err_deinit;
        }

        /* SET_MSID_TOKEN: the hlt_ccu_set_msid write side */
        in.op = CCU_U_OP_SET_MSID_TOKEN;
        in.data.data_info.udie_idx = 0;
        in.data.data_info.data_array_size = 1;
        in.data.data_info.data_array_len = sizeof(struct ccu_baseinfo);
        in.data.data_info.data_array[0].baseinfo.ms_id = msid;
        in.data.data_info.data_array[0].baseinfo.token_id = tokenId;
        in.data.data_info.data_array[0].baseinfo.token_value = tokenValue;
        ret = HccpTlvCcuCustomChannel(phyid, tlvHandle, &in, &out);
        if (ret != 0 || out.op_ret != 0) {
            TEST_LOG_ERROR(phyid, "SET_MSID_TOKEN failed ret=%d op_ret=%d", ret, out.op_ret);
            ret = -1;
            goto err_deinit;
        }
        TEST_LOG_INFO(phyid, "SET_MSID_TOKEN ok msid=0x%X", msid);

        /* both ends set their own device state before either reads it back */
        ret = SocketBarrier(env.sock.fdHandle);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "barrier failed ret=%d", ret);
            goto err_deinit;
        }

        /* GET_MSID: the readback assertion of hlt_ccu_set_msid */
        inGet.op = CCU_U_OP_GET_MSID;
        inGet.data.data_info.udie_idx = 0;
        inGet.data.data_info.data_array_size = 1;
        inGet.data.data_info.data_array_len = sizeof(struct ccu_baseinfo);
        ret = HccpTlvCcuCustomChannel(phyid, tlvHandle, &inGet, &outGet);
        if (ret != 0 || outGet.op_ret != 0) {
            TEST_LOG_ERROR(phyid, "GET_MSID failed ret=%d op_ret=%d", ret, outGet.op_ret);
            ret = -1;
            goto err_deinit;
        }
        gotMsid = outGet.data.data_info.data_array[0].baseinfo.ms_id;
        if (gotMsid != msid) {
            TEST_LOG_ERROR(phyid, "msid readback mismatch: set=0x%X get=0x%X", msid, gotMsid);
            ret = -1;
            goto err_deinit;
        }
        TEST_LOG_INFO(phyid, "GET_MSID readback ok msid=0x%X", gotMsid);

        /* GET_BASIC_INFO: driver/chip-derived fields must come back filled */
        inInfo.op = CCU_U_OP_GET_BASIC_INFO;
        inInfo.data.data_info.udie_idx = 0;
        inInfo.data.data_info.data_array_size = 1;
        inInfo.data.data_info.data_array_len = sizeof(struct ccu_baseinfo);
        ret = HccpTlvCcuCustomChannel(phyid, tlvHandle, &inInfo, &outInfo);
        if (ret != 0 || outInfo.op_ret != 0) {
            TEST_LOG_ERROR(phyid, "GET_BASIC_INFO failed ret=%d op_ret=%d", ret, outInfo.op_ret);
            ret = -1;
            goto err_deinit;
        }
        info = &outInfo.data.data_info.data_array[0].baseinfo;
        if (info->ms_id != msid || info->token_valid == 0 || info->missionKey == 0 || info->resourceAddr == nullptr) {
            TEST_LOG_ERROR(
                phyid, "GET_BASIC_INFO fields not filled: msid=0x%X tokenValid=%u missionKey=0x%X res=%p", info->ms_id,
                info->token_valid, info->missionKey, info->resourceAddr);
            ret = -1;
            goto err_deinit;
        }
        TEST_LOG_INFO(phyid, "GET_BASIC_INFO ok missionKey=0x%X res=%p", info->missionKey, info->resourceAddr);

        (void)HccpTlvCcuRequestVoid(phyid, tlvHandle, MSG_TYPE_CCU_UNINIT);
        (void)HccpTlvCcuDeinit(phyid, tlvHandle);
        CaseCleanup(env);
        TEST_LOG_INFO(phyid, "ccuTlvSmoke passed");
        return 0;

    err_deinit:
        (void)HccpTlvCcuRequestVoid(phyid, tlvHandle, MSG_TYPE_CCU_UNINIT);
        (void)HccpTlvCcuDeinit(phyid, tlvHandle);
    err:
        CaseCleanup(env);
        return ret;
    });
}
