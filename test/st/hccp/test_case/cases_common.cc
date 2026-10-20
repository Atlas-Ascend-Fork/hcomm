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

void HccpSystemTest::CaseCleanup(CaseEnv& env)
{
    if (env.ctxHandle != nullptr) {
        if (env.rmem != nullptr) {
            (void)RaCtxRmemUnimport(env.ctxHandle, env.rmem);
        }
        if (env.remQp != nullptr && env.qpHandle != nullptr) {
            (void)RaCtxQpUnbind(env.qpHandle);
        }
        if (env.remQp != nullptr) {
            (void)RaCtxQpUnimport(env.ctxHandle, env.remQp);
        }
        if (env.qpHandle != nullptr) {
            (void)RaCtxQpDestroy(env.qpHandle);
        }
        if (env.recvLmem != nullptr) {
            (void)RaCtxLmemUnregister(env.ctxHandle, env.recvLmem);
        }
        if (env.sendLmem != nullptr) {
            (void)RaCtxLmemUnregister(env.ctxHandle, env.sendLmem);
        }
        if (env.recvCq != nullptr) {
            (void)RaCtxCqDestroy(env.ctxHandle, env.recvCq);
        }
        if (env.sendCq != nullptr) {
            (void)RaCtxCqDestroy(env.ctxHandle, env.sendCq);
        }
        (void)RaCtxDeinit(env.ctxHandle);
    } else if (env.qpHandle != nullptr) {
        (void)RaQpDestroy(env.qpHandle);
    }

    if (env.socketHandle != nullptr) {
        (void)RaSocketDeinit(env.socketHandle);
    }

    for (int i = 0; i < 4; i++) {
        if (env.rtBuf[i] != nullptr) {
            (void)aclrtFree(env.rtBuf[i]);
        }
    }
    for (int i = 0; i < 2; i++) {
        if (env.heapBuf[i] != nullptr) {
            free(env.heapBuf[i]);
        }
    }

    (void)RaDeinit(&env.cfg);
    if (env.netServiceOpened) {
        (void)rtCloseNetService();
    }
}

static void CaseConfig(CaseEnv& env, int phyid, int nicPosition, uint32_t iface)
{
    env.cfg = {};
    env.cfg.phyId = (unsigned int)phyid;
    env.cfg.nicPosition = (unsigned int)nicPosition;
    env.cfg.hdcType = 18;
    env.cfg.enableHdcAsync = false;

    env.sockInit = {};
    env.sockInit.rdevInfo.phyId = (unsigned int)phyid;
    env.sockInit.rdevInfo.family = AF_INET;
    env.sockInit.rdevInfo.localIp.addr.s_addr = htonl(HccpVmNicIp((uint32_t)phyid, iface));
    env.sockInit.scopeId = 4;
}

int HccpSystemTest::CaseOfflineInit(CaseEnv& env, int phyid, uint32_t iface, unsigned short port, const char* tag)
{
    CaseConfig(env, phyid, 1, iface);

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
    env.netServiceOpened = true;

    /* rtOpenNetService returns only after the device stack reports ready
     * (stub waits on deviceReady, 30s budget), so RaInit must succeed on
     * the first call: a failure is a real defect, not a startup race. */
    int ret = RaInit(&env.cfg);
    if (ret != 0) {
        (void)RaDeinit(&env.cfg); /* release any half-initialized RA state */
        TEST_LOG_ERROR(phyid, "RaInit failed ret=%d", ret);
        CaseCleanup(env);
        return ret;
    }

    ret = EstablishRendezvous(phyid, 1, env.sockInit, port, tag, true, &env.socketHandle, &env.sock);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "EstablishRendezvous failed ret=%d", ret);
        CaseCleanup(env);
        return ret;
    }
    return 0;
}

int HccpSystemTest::CaseOfflineInitNoRdv(CaseEnv& env, int phyid, uint32_t iface)
{
    CaseConfig(env, phyid, 1, iface);

    char devIdParam[32];
    (void)snprintf(devIdParam, sizeof(devIdParam), "--deviceId=%d", phyid);
    char pidParam[32];
    (void)snprintf(pidParam, sizeof(pidParam), "--pid=%d", (int)getpid());
    rtProcExtParam extParams[2] = {{devIdParam, strlen(devIdParam)}, {pidParam, strlen(pidParam)}};
    rtNetServiceOpenArgs openArgs = {extParams, 2};
    if (rtOpenNetService(&openArgs) != RT_ERROR_NONE) {
        TEST_LOG_ERROR(phyid, "rtOpenNetService failed");
        return 1;
    }
    env.netServiceOpened = true;

    /* rtOpenNetService returns only after the device stack reports ready
     * (stub waits on deviceReady, 30s budget), so RaInit must succeed on
     * the first call: a failure is a real defect, not a startup race. */
    int ret = RaInit(&env.cfg);
    if (ret != 0) {
        (void)RaDeinit(&env.cfg); /* release any half-initialized RA state */
        TEST_LOG_ERROR(phyid, "RaInit failed ret=%d", ret);
        CaseCleanup(env);
        return ret;
    }
    return 0;
}

int HccpSystemTest::CasePeerInit(CaseEnv& env, int phyid, uint32_t iface, unsigned short port, const char* tag)
{
    CaseConfig(env, phyid, 0, iface);

    int ret = RaInit(&env.cfg);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "RaInit failed ret=%d", ret);
        CaseCleanup(env);
        return ret;
    }

    ret = EstablishRendezvous(phyid, 0, env.sockInit, port, tag, true, &env.socketHandle, &env.sock);
    if (ret != 0) {
        TEST_LOG_ERROR(phyid, "EstablishRendezvous failed ret=%d", ret);
        CaseCleanup(env);
        return ret;
    }
    return 0;
}

int HccpSystemTest::CaseRdevInit(CaseEnv& env, int phyid, int mode, unsigned int notifyType, uint32_t iface)
{
    rdev rdevInfo = {};
    rdevInfo.family = AF_INET;
    rdevInfo.localIp.addr.s_addr = htonl(HccpVmNicIp((uint32_t)phyid, iface));
    rdevInfo.localIp.addr = HccpVmToLoopback(rdevInfo.localIp.addr);
    rdevInfo.phyId = (unsigned int)phyid;
    /* Use RaRdevInitV2 so the business lite path runs in the 2MB-align mode: QP create
     * allocates the device-memory pool (provider side) and maps it to host (lite lib). */
    struct RdevInitInfo initInfo = {};
    initInfo.mode = mode;
    initInfo.notifyType = notifyType;
    initInfo.enabled2mbLite = true;
    return RaRdevInitV2(initInfo, rdevInfo, &env.rdmaHandle);
}
