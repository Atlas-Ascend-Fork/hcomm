/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <unistd.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <errno.h>
#include <string.h>
#include <dlfcn.h>
#include "ascend_hal.h"
#include "ascend_hal_stub.h"
#include "hccp_shm_infra.h"

/* The rs layer pairs its HccpDlopen("liburma.so.0") with a balanced dlclose
 * (RsUbApiDeinit) but never clears gUrmaOps, so once the refcount drops to zero
 * the dlclose unmaps the ST urma stub while the ops table still points into
 * its text - the next indirect call jumps into unmapped memory (SIGSEGV in the
 * RA_RS_GET_DEV_EID_INFO_LIST handler). In production liburma stays resident
 * through link-time dependencies of the rs component, so the dlclose only drops
 * the dlopen refcount. Model the same invariant here: this library is a
 * DT_NEEDED dependency of hccp_service.bin, and pinning the urma stub from its
 * constructor (before any rs-side dlopen, outside the urma load lock) makes
 * dlclose never unmap it. If the stub is not resolvable the pin is a no-op and
 * behavior is unchanged. */
__attribute__((constructor)) static void HccpStubPinUrmaResident(void)
{
    (void)dlopen("liburma.so.0", RTLD_NOW | RTLD_NODELETE);
}

typedef enum { TSD_HCCP = 0, TSD_COMPUTE, TSD_CUSTOM_COMPUTE, TSD_QS, TSD_WAITTYPE_MAX } TsdWaitType;

int32_t
SendStartUpFinishMsg(const uint32_t deviceId, const TsdWaitType waitType, const uint32_t hostPid, const uint32_t vfId)
{
    (void)waitType;
    (void)hostPid;
    (void)vfId;
    /* One-shot attach to the precreated H2D registry (born-zero memfd table):
     * mark the device ready, drop the mapping, park until teardown. */
    h2d_info_t(*info)[MAX_DEV_ID]
        = (h2d_info_t(*)[MAX_DEV_ID])hccp_shm_open(HCCP_SHM_KEY_H2D, sizeof(h2d_info_t) * MAX_DEV_ID * HOST_OS_MAX, 0);
    if (info != NULL) {
        if (deviceId < MAX_DEV_ID) {
            info[HOST_ID_0][deviceId].deviceReady = 1;
        }
        (void)munmap(info, sizeof(h2d_info_t) * MAX_DEV_ID * HOST_OS_MAX);
    }

    sigset_t mask;
    sigemptyset(&mask);
    sigaddset(&mask, SIGUSR1);
    sigprocmask(SIG_BLOCK, &mask, NULL);
    int sig;
    sigwait(&mask, &sig);
    return 0;
}

int32_t ReportProcessStartUpErrorCode(
    const uint32_t deviceId, const TsdWaitType waitType, const uint32_t hostPid, const uint32_t vfId,
    const char* errCode, const uint32_t errLen)
{
    return 0;
}
