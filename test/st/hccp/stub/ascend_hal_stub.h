/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCEND_HAL_STUB_H
#define ASCEND_HAL_STUB_H

#include <sys/syscall.h>
#include <sys/types.h>
#include <dlog_pub.h>

/* The H2D registry lives behind hccp_shm_open(HCCP_SHM_KEY_H2D, ...) in
 * hccp_shm_infra.h now; the legacy SHM_KEY constant retired with it. */

#define DRV_LOG_INFO(fmt, ...)                                                    \
    do {                                                                          \
        if (CheckLogLevel(DRV, DLOG_INFO) == 1) {                                 \
            DlogRecord(DRV, DLOG_INFO, "[DRV-" LOG_SIDE "] " fmt, ##__VA_ARGS__); \
        }                                                                         \
    } while (0)
#define DRV_LOG_ERROR(fmt, ...)                                                                                      \
    do {                                                                                                             \
        DlogRecord(                                                                                                  \
            DRV, DLOG_ERROR, "[DRV-" LOG_SIDE "][%s:%d]pid:%d,tid:%ld,%s : " fmt, __FILE__, __LINE__, (int)getpid(), \
            (long)syscall(SYS_gettid), __func__, ##__VA_ARGS__);                                                     \
    } while (0)

#ifndef container_of
#define container_of(ptr, type, member)                   \
    ({                                                    \
        const typeof(((type*)0)->member)* __mptr = (ptr); \
        (type*)((char*)__mptr - offsetof(type, member));  \
    })
#endif

#define LISTEN_BACKLOG 5
#define MAX_DEV_ID 64
#define HOST_OS_MAX 4
#define HOST_ID_0 0

typedef struct {
    int sessionId;
    int conn_fd;
    unsigned int hostPhyId;
    unsigned int devPhyId;
    int lastRecvStatus;
    int isUsed;
    int peerNode;
    pid_t hostPid;
    pid_t devPid;
    pthread_mutex_t lock;
} HdcSessionT;

typedef struct ServerInfo {
    int listen_fd;
    unsigned int phyId;
    int port;
} ServerInfoT;

struct hdc_msg_head {
    bool freeBuf;
    struct drvHdcMsg msg;
};

#include "hccp_shm_infra.h"

/* HDC service ports live at 9999+phyId in the default run. HCCP_ST_RUN_ID=N
 * moves the whole block to 9999+phyId+N*10000 (N documented 0..2) so
 * concurrent runners keep disjoint port sets; parsing and range rules are
 * owned by hccp_st_run_id() in hccp_shm_infra.h. */
static inline int GetPortByServiceType(int phyId) { return 9999 + phyId + hccp_st_run_id() * 10000; }

static inline int GetPeerPortByDevid(int peerDevid) { return 9999 + peerDevid + hccp_st_run_id() * 10000; }

typedef struct h2d_info {
    pid_t hostPid;
    pid_t devPid;
    unsigned int hostLogicId;
    unsigned int devLogicId;
    unsigned int hostPhyId;
    unsigned int devPhyId;
    volatile unsigned int deviceReady;
} h2d_info_t;

#if defined(__cplusplus)
#define HCCP_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#else
#define HCCP_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#endif
HCCP_STATIC_ASSERT(sizeof(h2d_info_t) == 28, "h2d_info_t layout drifted (7*4B expected)");
HCCP_STATIC_ASSERT(
    sizeof(h2d_info_t) * MAX_DEV_ID * HOST_OS_MAX == HCCP_SHM_SIZE_H2D,
    "HCCP_SHM_SIZE_H2D drifted from h2d_info_t layout");

#if defined(__cplusplus)
extern "C" {
#endif

void setHostPid(pid_t pid, int hostId, int phyId);
void setDevPid(pid_t pid, int hostId, int phyId);
void InitHdcId();
h2d_info_t (*GetH2dInfo())[MAX_DEV_ID];
int AllocSessionId(void);
HdcSessionT** GetSession();
#if defined(__cplusplus)
}
#endif

#endif
