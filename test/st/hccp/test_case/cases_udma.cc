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

/* Two-node UB WRITE (one-sided, urma stub data plane; the Ra layer has no UB SEND opcode,
 * so this covers the WRITE direction only): RaCtxInit(NETWORK_OFFLINE) + a send/recv CQ
 * pair; rtMalloc two HBM buffers and LmemRegister them; create a jetty QP (CONN_RM/NORMAL,
 * jettyId=1000+phyId); exchange qpKey/recv memKey over the socket, then QpImport/QpBind/
 * RmemImport, and RaBatchSendWr issues the WRITE; RaCtxUpdateCi drives the receive-side
 * drain; after the barrier, memcmp verifies the recv buffer holds the peer's content. */
TEST_F(HccpSystemTest, d2dUbWriteRecv)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        (void)setvbuf(stdout, nullptr, _IONBF, 0);
        struct QpCreateAttr qpAttr = {};
        struct QpCreateInfo qpInfo = {};
        struct MrRegInfoT sendMr = {}, recvMr = {};
        struct MrImportInfoT rmemImport = {};
        struct QpImportInfoT qpImport = {};
        struct CqInfoT cqInfo = {};
        struct CtxInitCfg ctxCfg = {};
        struct CtxInitAttr ctxAttr = {};
        struct WrSgeList sge = {};
        struct SendWrData wrData = {};
        struct SendWrResp wrResp = {};
        struct UbRendezvous mine = {}, peer = {};
        unsigned long long sendLen = 0;
        unsigned int completeNum = 0;

        const char* expected = (phyid == 1) ? "UB from phyId 5" : "UB from phyId 1";
        CaseEnv env;
        int ret = CaseOfflineInit(env, phyid, HCCP_VM_IFACE_SOCKET, 15667, "UB RENDEZVOUS");
        if (ret != 0) {
            return ret;
        }

        ctxCfg.mode = NETWORK_OFFLINE;
        ctxAttr.phyId = (unsigned int)phyid;
        ret = RaCtxInit(&ctxCfg, &ctxAttr, &env.ctxHandle);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaCtxInit failed ret=%d", ret);
            goto err;
        }

        cqInfo.in.depth = 64;
        cqInfo.in.ub.mode = JFC_MODE_NORMAL;
        ret = RaCtxCqCreate(env.ctxHandle, &cqInfo, &env.sendCq);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaCtxCqCreate send failed ret=%d", ret);
            goto err;
        }
        ret = RaCtxCqCreate(env.ctxHandle, &cqInfo, &env.recvCq);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaCtxCqCreate recv failed ret=%d", ret);
            goto err;
        }

        if (rtMalloc(&env.rtBuf[0], 256, RT_MEMORY_RDMA_HBM, 0) != RT_ERROR_NONE) {
            TEST_LOG_ERROR(phyid, "rtMalloc send failed");
            ret = -1;
            goto err;
        }
        sendMr.in.mem.addr = (uint64_t)(uintptr_t)env.rtBuf[0];
        sendMr.in.mem.size = 256;
        ret = RaCtxLmemRegister(env.ctxHandle, &sendMr, &env.sendLmem);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaCtxLmemRegister send failed ret=%d", ret);
            goto err;
        }
        (void)snprintf((char*)env.rtBuf[0], 256, "UB from phyId %d", phyid);
        sendLen = strlen((char*)env.rtBuf[0]) + 1;

        if (rtMalloc(&env.rtBuf[1], 256, RT_MEMORY_RDMA_HBM, 0) != RT_ERROR_NONE) {
            TEST_LOG_ERROR(phyid, "rtMalloc recv failed");
            ret = -1;
            goto err;
        }
        recvMr.in.mem.addr = (uint64_t)(uintptr_t)env.rtBuf[1];
        recvMr.in.mem.size = 256;
        ret = RaCtxLmemRegister(env.ctxHandle, &recvMr, &env.recvLmem);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaCtxLmemRegister recv failed ret=%d", ret);
            goto err;
        }

        qpAttr.scqHandle = env.sendCq;
        qpAttr.rcqHandle = env.recvCq;
        qpAttr.sqDepth = 64;
        qpAttr.rqDepth = 64;
        qpAttr.transportMode = CONN_RM;
        qpAttr.ub.mode = JETTY_MODE_URMA_NORMAL;
        qpAttr.ub.jettyId = 1000u + (uint32_t)phyid;
        ret = RaCtxQpCreate(env.ctxHandle, &qpAttr, &qpInfo, &env.qpHandle);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaCtxQpCreate failed ret=%d", ret);
            goto err;
        }

        mine.qpKey = qpInfo.key;
        mine.memKey = recvMr.out.key;
        ret = ExchangePeer(env.sock.fdHandle, mine, &peer);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "exchange failed ret=%d", ret);
            goto err;
        }

        qpImport.in.key = peer.qpKey;
        qpImport.in.ub.mode = JETTY_IMPORT_MODE_NORMAL;
        qpImport.in.ub.type = TARGET_TYPE_JETTY;
        ret = RaCtxQpImport(env.ctxHandle, &qpImport, &env.remQp);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaCtxQpImport failed ret=%d", ret);
            goto err;
        }
        ret = RaCtxQpBind(env.qpHandle, env.remQp);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaCtxQpBind failed ret=%d", ret);
            goto err;
        }

        rmemImport.in.key = peer.memKey;
        ret = RaCtxRmemImport(env.ctxHandle, &rmemImport, &env.rmem);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaCtxRmemImport failed ret=%d", ret);
            goto err;
        }

        sge.addr = (uint64_t)(uintptr_t)env.rtBuf[0];
        sge.len = (uint32_t)sendLen;
        sge.lmemHandle = env.sendLmem;
        wrData.sges = &sge;
        wrData.numSge = 1;
        wrData.rmemHandle = env.rmem;
        wrData.ub.opcode = RA_UB_OPC_WRITE;
        wrData.ub.userCtx = 0xB2;
        wrData.ub.remQpHandle = env.remQp;
        ret = RaBatchSendWr(env.qpHandle, &wrData, &wrResp, 1, &completeNum);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaBatchSendWr failed ret=%d", ret);
            goto err;
        }

        ret = RaCtxUpdateCi(env.qpHandle, 1);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaCtxUpdateCi failed ret=%d", ret);
            goto err;
        }

        ret = SocketBarrier(env.sock.fdHandle);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "barrier failed ret=%d", ret);
            goto err;
        }

        ret = RaCtxUpdateCi(env.qpHandle, 2);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "drain failed ret=%d", ret);
            goto err;
        }
        for (int i = 0; i < 200 && ((char*)env.rtBuf[1])[0] == 0; i++) {
            usleep(1000);
            (void)RaCtxUpdateCi(env.qpHandle, (uint16_t)(2 + i));
        }

        if (memcmp(env.rtBuf[1], expected, strlen(expected) + 1) != 0) {
            TEST_LOG_ERROR(phyid, "UB memcmp FAIL exp=%s act=%s", expected, (char*)env.rtBuf[1]);
            ret = -1;
            goto err;
        }
        TEST_LOG_INFO(phyid, "UB memcmp PASS: recv=%s", (char*)env.rtBuf[1]);
        ret = 0;
    err:
        CaseCleanup(env);
        return ret;
    });
}

/* Two-node UB READ (one-sided): same structure as d2dUbWriteRecv but in the READ direction
 * - after exchanging the data buffer memKey, RaBatchSendWr issues a READ that pulls 256B
 * from the peer's data buffer into the local result buffer, memcmp-verified on completion. */
TEST_F(HccpSystemTest, d2dUbRead)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        (void)setvbuf(stdout, nullptr, _IONBF, 0);
        struct QpCreateAttr qpAttr = {};
        struct QpCreateInfo qpInfo = {};
        struct MrRegInfoT resultMr = {}, dataMr = {};
        struct MrImportInfoT rmemImport = {};
        struct QpImportInfoT qpImport = {};
        struct CqInfoT cqInfo = {};
        struct CtxInitCfg ctxCfg = {};
        struct CtxInitAttr ctxAttr = {};
        struct WrSgeList sge = {};
        struct SendWrData wrData = {};
        struct SendWrResp wrResp = {};
        struct UbRendezvous mine = {}, peer = {};
        unsigned int completeNum = 0;

        const char* expected = (phyid == 1) ? "READ from phyId 5" : "READ from phyId 1";
        CaseEnv env;
        int ret = CaseOfflineInit(env, phyid, HCCP_VM_IFACE_SOCKET, 15668, "UB READ RDV");
        if (ret != 0) {
            return ret;
        }

        ctxCfg.mode = NETWORK_OFFLINE;
        ctxAttr.phyId = (unsigned int)phyid;
        ret = RaCtxInit(&ctxCfg, &ctxAttr, &env.ctxHandle);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaCtxInit failed ret=%d", ret);
            goto err;
        }

        cqInfo.in.depth = 64;
        cqInfo.in.ub.mode = JFC_MODE_NORMAL;
        ret = RaCtxCqCreate(env.ctxHandle, &cqInfo, &env.sendCq);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaCtxCqCreate send failed ret=%d", ret);
            goto err;
        }
        ret = RaCtxCqCreate(env.ctxHandle, &cqInfo, &env.recvCq);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaCtxCqCreate recv failed ret=%d", ret);
            goto err;
        }

        if (rtMalloc(&env.rtBuf[1], 256, RT_MEMORY_RDMA_HBM, 0) != RT_ERROR_NONE) {
            TEST_LOG_ERROR(phyid, "rtMalloc result failed");
            ret = -1;
            goto err;
        }
        resultMr.in.mem.addr = (uint64_t)(uintptr_t)env.rtBuf[1];
        resultMr.in.mem.size = 256;
        ret = RaCtxLmemRegister(env.ctxHandle, &resultMr, &env.sendLmem);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaCtxLmemRegister result failed ret=%d", ret);
            goto err;
        }

        if (rtMalloc(&env.rtBuf[0], 256, RT_MEMORY_RDMA_HBM, 0) != RT_ERROR_NONE) {
            TEST_LOG_ERROR(phyid, "rtMalloc data failed");
            ret = -1;
            goto err;
        }
        dataMr.in.mem.addr = (uint64_t)(uintptr_t)env.rtBuf[0];
        dataMr.in.mem.size = 256;
        ret = RaCtxLmemRegister(env.ctxHandle, &dataMr, &env.recvLmem);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaCtxLmemRegister data failed ret=%d", ret);
            goto err;
        }
        (void)snprintf((char*)env.rtBuf[0], 256, "READ from phyId %d", phyid);

        qpAttr.scqHandle = env.sendCq;
        qpAttr.rcqHandle = env.recvCq;
        qpAttr.sqDepth = 64;
        qpAttr.rqDepth = 64;
        qpAttr.transportMode = CONN_RM;
        qpAttr.ub.mode = JETTY_MODE_URMA_NORMAL;
        qpAttr.ub.jettyId = 1000u + (uint32_t)phyid;
        ret = RaCtxQpCreate(env.ctxHandle, &qpAttr, &qpInfo, &env.qpHandle);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaCtxQpCreate failed ret=%d", ret);
            goto err;
        }

        mine.qpKey = qpInfo.key;
        mine.memKey = dataMr.out.key;
        ret = ExchangePeer(env.sock.fdHandle, mine, &peer);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "exchange failed ret=%d", ret);
            goto err;
        }

        qpImport.in.key = peer.qpKey;
        qpImport.in.ub.mode = JETTY_IMPORT_MODE_NORMAL;
        qpImport.in.ub.type = TARGET_TYPE_JETTY;
        ret = RaCtxQpImport(env.ctxHandle, &qpImport, &env.remQp);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaCtxQpImport failed ret=%d", ret);
            goto err;
        }
        ret = RaCtxQpBind(env.qpHandle, env.remQp);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaCtxQpBind failed ret=%d", ret);
            goto err;
        }

        rmemImport.in.key = peer.memKey;
        ret = RaCtxRmemImport(env.ctxHandle, &rmemImport, &env.rmem);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaCtxRmemImport failed ret=%d", ret);
            goto err;
        }

        sge.addr = (uint64_t)(uintptr_t)env.rtBuf[1];
        sge.len = 256;
        sge.lmemHandle = env.sendLmem;
        wrData.sges = &sge;
        wrData.numSge = 1;
        wrData.rmemHandle = env.rmem;
        wrData.ub.opcode = RA_UB_OPC_READ;
        wrData.ub.userCtx = 0xD3;
        wrData.ub.remQpHandle = env.remQp;
        ret = RaBatchSendWr(env.qpHandle, &wrData, &wrResp, 1, &completeNum);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaBatchSendWr READ failed ret=%d", ret);
            goto err;
        }

        if (memcmp(env.rtBuf[1], expected, strlen(expected) + 1) != 0) {
            TEST_LOG_ERROR(phyid, "UB READ memcmp FAIL exp=%s act=%s", expected, (char*)env.rtBuf[1]);
            ret = -1;
            goto err;
        }
        TEST_LOG_INFO(phyid, "UB READ memcmp PASS: result=%s", (char*)env.rtBuf[1]);
        ret = 0;
    err:
        CaseCleanup(env);
        return ret;
    });
}
