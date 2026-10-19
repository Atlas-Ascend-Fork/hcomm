/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCP_CCU_TLV_SHIM_H
#define HCCP_CCU_TLV_SHIM_H

#include <stdint.h>
#include "hccp_tlv.h"
#include "ccu_u_comm.h"
#include "hccp_sys_test_common.h"

/*
 * Client-side CCU custom-channel shim for the migrated HLT hccp cases.
 *
 * The original suite drives the CCU through the product entry RaCustomChannel
 * (test_ra_custom_channel in hlt_hccp_ccu_driver_common.c), which this
 * repository does not export. The supported equivalent is the CCU module of the
 * TLV service: RaTlvRequest(TLV_MODULE_TYPE_CCU, MSG_TYPE_CCU_DISPATCH_CMD)
 * carries the same 2056-byte channel_info_in/out structs to the device process,
 * where rs_tlv.c RsCcuRequest dispatches them to the dlsym'd ccu_custom_channel
 * of libccu-user-drv.so. The struct layouts are identical on both sides
 * (CustomChanInfoIn/out and channel_info_in/out only differ in field names).
 *
 * Call sequence (mirrors test_ccu_init_via_hccp / test_ccu_custom_channel /
 * test_ccu_uninit_via_hccp of the original driver common):
 *   CaseOfflineInit (RaInit offline) -> HccpTlvCcuInit -> HccpTlvCcuRequestInit
 *   -> HccpTlvCcuCustomChannel xN -> HccpTlvCcuRequestUninit -> HccpTlvCcuDeinit
 */

/* RaTlvInit wrapper: offline mode, same shape as test_ra_tlv_init */
static inline int HccpTlvCcuInit(int phyid, void** tlvHandle)
{
    struct TlvInitInfo initInfo = {};
    unsigned int bufferSize = 0;
    initInfo.version = 0;
    initInfo.phyId = (unsigned int)phyid;
    initInfo.nicPosition = NETWORK_OFFLINE;
    int ret = RaTlvInit(&initInfo, &bufferSize, tlvHandle);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaTlvInit failed ret=%d", ret);
        return ret;
    }
    return 0;
}

/* One TLV request to the CCU module with no payload (MSG_TYPE_CCU_INIT/_UNINIT) */
static inline int HccpTlvCcuRequestVoid(int phyid, void* tlvHandle, unsigned int msgType)
{
    struct TlvMsg sendMsg = {};
    struct TlvMsg recvMsg = {};
    sendMsg.type = msgType;
    sendMsg.length = 0;
    sendMsg.data = nullptr;
    recvMsg.length = 0;
    recvMsg.data = nullptr;
    int ret = RaTlvRequest(tlvHandle, TLV_MODULE_TYPE_CCU, &sendMsg, &recvMsg);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaTlvRequest type=%u failed ret=%d", msgType, ret);
        return ret;
    }
    return 0;
}

/* RaCustomChannel equivalent: round-trip one CCU command through the CCU TLV
 * module. The full channel_info_in travels as one V2 chunk (2056 <= 3072). */
static inline int
HccpTlvCcuCustomChannel(int phyid, void* tlvHandle, const struct channel_info_in* in, struct channel_info_out* out)
{
    struct TlvMsg sendMsg = {};
    struct TlvMsg recvMsg = {};
    sendMsg.type = MSG_TYPE_CCU_DISPATCH_CMD;
    sendMsg.length = sizeof(struct channel_info_in);
    sendMsg.data = (char*)in; /* TlvMsg carries an untyped buffer */
    recvMsg.length = sizeof(struct channel_info_out);
    recvMsg.data = (char*)out;
    int ret = RaTlvRequest(tlvHandle, TLV_MODULE_TYPE_CCU, &sendMsg, &recvMsg);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaTlvRequest dispatch op=%u failed ret=%d", (unsigned int)in->op, ret);
        return ret;
    }
    if (recvMsg.length < sizeof(struct channel_info_out)) {
        TEST_LOG_ERROR(phyid, "short dispatch reply len=%u", recvMsg.length);
        return -1;
    }
    return 0;
}

static inline int HccpTlvCcuDeinit(int phyid, void* tlvHandle)
{
    int ret = RaTlvDeinit(tlvHandle);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaTlvDeinit failed ret=%d", ret);
    }
    return ret;
}

#endif /* HCCP_CCU_TLV_SHIM_H */
