/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* g++ predefines _GNU_SOURCE for C++ (this TU is compiled as C++ for the
 * tsd_client.h include chain), so guard the define instead of redefining it */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <limits.h>
#include <signal.h>
#include <sys/wait.h>
#include <sys/prctl.h>
#include <sys/mman.h>
#include <dlfcn.h>
#include <libgen.h>
#include <pthread.h>
#include <runtime/rt_external_device.h>
#ifndef RT_ERROR_INVALID_VALUE
#define RT_ERROR_INVALID_VALUE ((rtError_t)0x07110001)
#endif
#include "ascend_hal.h"
#include "ascend_hal_stub.h"
#include "hccp_shm_infra.h"

static char g_vmHome[PATH_MAX] = {0};

static const char* VmHome(void)
{
    if (g_vmHome[0] != '\0') {
        return g_vmHome;
    }
    const char* env = getenv("HCCP_VM_HOME");
    if (env != NULL && env[0] != '\0') {
        (void)snprintf(g_vmHome, sizeof(g_vmHome), "%s", env);
        return g_vmHome;
    }
    Dl_info info = {0};
    if (dladdr((void*)&VmHome, &info) != 0 && info.dli_fname != NULL) {
        char path[PATH_MAX];
        (void)snprintf(path, sizeof(path), "%s", info.dli_fname);
        char* dir = dirname(path);
        (void)snprintf(g_vmHome, sizeof(g_vmHome), "%s/..", dir);
        return g_vmHome;
    }
    (void)snprintf(g_vmHome, sizeof(g_vmHome), ".");
    return g_vmHome;
}

static h2d_info_t (*H2dAttach(void))[MAX_DEV_ID]
{
    /* Attach to the precreated H2D registry (born-zero memfd table); pair
     * every attach with H2dDetach -- the mapping is private to this call. */
    return (h2d_info_t(*)[MAX_DEV_ID])hccp_shm_open(HCCP_SHM_KEY_H2D, sizeof(h2d_info_t) * MAX_DEV_ID * HOST_OS_MAX, 0);
}

static void H2dDetach(h2d_info_t (*info)[MAX_DEV_ID])
{
    if (info != NULL && info != (void*)-1) {
        (void)munmap(info, sizeof(h2d_info_t) * MAX_DEV_ID * HOST_OS_MAX);
    }
}

static unsigned int ParseExtParam(const struct rtProcExtParam* list, uint64_t cnt, const char* key, long defVal)
{
    char prefix[32];
    (void)snprintf(prefix, sizeof(prefix), "%s=", key);
    for (uint64_t i = 0; i < cnt && list != NULL; i++) {
        if (list[i].paramInfo == NULL) {
            continue;
        }
        if (strncmp(list[i].paramInfo, prefix, strlen(prefix)) == 0) {
            return (unsigned int)atol(list[i].paramInfo + strlen(prefix));
        }
    }
    return (unsigned int)defVal;
}

struct DevProcNode {
    pid_t pid;
    unsigned int phyId;
    struct DevProcNode* next;
};

static struct DevProcNode* g_devProcs = NULL;
static pthread_mutex_t g_devProcsMutex = PTHREAD_MUTEX_INITIALIZER;

static void DevProcsAdd(pid_t pid, unsigned int phyId)
{
    struct DevProcNode* node = (struct DevProcNode*)calloc(1, sizeof(*node));
    if (node == NULL) {
        return;
    }
    node->pid = pid;
    node->phyId = phyId;
    pthread_mutex_lock(&g_devProcsMutex);
    node->next = g_devProcs;
    g_devProcs = node;
    pthread_mutex_unlock(&g_devProcsMutex);
}

static pid_t DevProcsTake(void)
{
    pthread_mutex_lock(&g_devProcsMutex);
    struct DevProcNode* node = g_devProcs;
    pid_t pid = -1;
    if (node != NULL) {
        pid = node->pid;
        g_devProcs = node->next;
        free(node);
    }
    pthread_mutex_unlock(&g_devProcsMutex);
    return pid;
}

static int WaitDevReady(unsigned int phyId, unsigned int timeoutMs)
{
    for (unsigned int elapsed = 0; elapsed < timeoutMs; elapsed += 5) {
        h2d_info_t(*info)[MAX_DEV_ID] = H2dAttach();
        if (info != NULL && info != (void*)-1) {
            int ready = (phyId < MAX_DEV_ID) ? info[HOST_ID_0][phyId].deviceReady : 0;
            H2dDetach(info);
            if (ready == 1) {
                return 0;
            }
        }
        usleep(5000);
    }
    return -1;
}

rtError_t rtOpenNetService(const rtNetServiceOpenArgs* args)
{
    if (args == NULL || args->extParamList == NULL || args->extParamCnt == 0) {
        return RT_ERROR_INVALID_VALUE;
    }

    long devIdVal = (long)ParseExtParam(args->extParamList, args->extParamCnt, "--deviceId", -1L);
    if (devIdVal < 0 || (unsigned long)devIdVal >= MAX_DEV_ID) {
        return RT_ERROR_INVALID_VALUE;
    }
    unsigned int phyId = (unsigned int)devIdVal;
    long hostPid = (long)ParseExtParam(args->extParamList, args->extParamCnt, "--pid", (long)getpid());

    h2d_info_t(*info)[MAX_DEV_ID] = H2dAttach();
    if (info == NULL || info == (void*)-1) {
        return RT_ERROR_INVALID_VALUE;
    }
    info[HOST_ID_0][phyId].hostPid = (unsigned int)getpid();
    info[HOST_ID_0][phyId].devPid = 0;
    info[HOST_ID_0][phyId].deviceReady = 0;
    H2dDetach(info);

    char serviceBin[PATH_MAX];
    char deviceDir[PATH_MAX];
    char commonDir[PATH_MAX];
    (void)snprintf(serviceBin, sizeof(serviceBin), "%s/device/hccp_service.bin", VmHome());
    (void)snprintf(deviceDir, sizeof(deviceDir), "%s/device", VmHome());
    (void)snprintf(commonDir, sizeof(commonDir), "%s/common", VmHome());
    if (access(serviceBin, X_OK) != 0) {
        return RT_ERROR_INVALID_VALUE;
    }

    pid_t pid = fork();
    if (pid == -1) {
        return RT_ERROR_INVALID_VALUE;
    }
    if (pid == 0) {
        (void)prctl(PR_SET_PDEATHSIG, SIGKILL);
        h2d_info_t(*cinfo)[MAX_DEV_ID] = H2dAttach();
        if (cinfo != NULL && cinfo != (void*)-1) {
            cinfo[HOST_ID_0][phyId].devPid = (unsigned int)getpid();
            H2dDetach(cinfo);
        }
        char mockPath[PATH_MAX];
        (void)snprintf(mockPath, sizeof(mockPath), "%s/common/libmock.so", VmHome());
        char ldPath[PATH_MAX * 2];
        (void)snprintf(ldPath, sizeof(ldPath), "%s:%s/common", deviceDir, VmHome());
        setenv("LD_PRELOAD", mockPath, 1);
        setenv("LD_LIBRARY_PATH", ldPath, 1);
        char pidParam[32];
        (void)snprintf(pidParam, sizeof(pidParam), "--pid=%ld", hostPid);
        char devIdParam[32];
        (void)snprintf(devIdParam, sizeof(devIdParam), "--deviceId=%u", phyId);
        char* argv[RT_EXT_PARAM_CNT_MAX + 8];
        int argc = 0;
        argv[argc++] = (char*)"hccp_service.bin";
        for (uint64_t i = 0; i < args->extParamCnt && argc < RT_EXT_PARAM_CNT_MAX + 6; i++) {
            if (args->extParamList[i].paramInfo == NULL) {
                continue;
            }
            argv[argc++] = (char*)args->extParamList[i].paramInfo;
        }
        int hasHdcType = 0;
        int hasLogLevel = 0;
        for (int k = 1; k < argc; k++) {
            if (strncmp(argv[k], "--hdcType=", 10) == 0)
                hasHdcType = 1;
            if (strncmp(argv[k], "--logLevelInPid=", 16) == 0)
                hasLogLevel = 1;
        }
        if (!hasHdcType) {
            argv[argc++] = (char*)"--hdcType=18";
        }
        if (!hasLogLevel) {
            argv[argc++] = (char*)"--logLevelInPid=3";
        }
        int hasPid = 0;
        for (int k = 1; k < argc; k++) {
            if (strncmp(argv[k], "--pid=", 6) == 0)
                hasPid = 1;
        }
        if (!hasPid) {
            argv[argc++] = pidParam;
        }
        argv[argc] = NULL;
        (void)devIdParam;
        (void)execv(serviceBin, argv);
        perror("[rtOpenNetService] execv failed");
        _exit(127);
    }

    DevProcsAdd(pid, phyId);
    if (WaitDevReady(phyId, 30000) != 0) {
        (void)kill(pid, SIGKILL);
        (void)waitpid(pid, NULL, 0);
        return RT_ERROR_INVALID_VALUE;
    }
    return RT_ERROR_NONE;
}

/* runtime_rt_stub.c: returns this process's notify arena slots to the shared
 * devmem pool so later cases can reuse them (children fork per case) */
extern void RtStubNotifyRecycleSlots(void);

rtError_t rtCloseNetService(void)
{
    pid_t pid;
    int failed = 0;
    while ((pid = DevProcsTake()) != -1) {
        (void)kill(pid, SIGUSR1);
        if (waitpid(pid, NULL, 0) == -1) {
            failed = 1;
        }
    }
    RtStubNotifyRecycleSlots();
    return failed ? RT_ERROR_INVALID_VALUE : RT_ERROR_NONE;
}
