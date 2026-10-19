/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <errno.h>
#include <poll.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <sys/uio.h>
#include <stdint.h>
#include <sys/mman.h>
#include <arpa/inet.h>
#include "ascend_hal.h"
#include "hccp_vm_net.h"
#include "ascend_hal_stub.h"
#include "hccp_shm_infra.h"

static h2d_info_t (*g_h2dInfo)[MAX_DEV_ID] = NULL;

#define SHM_SIZE (sizeof(h2d_info_t) * MAX_DEV_ID * HOST_OS_MAX)

h2d_info_t (*GetH2dInfo())[MAX_DEV_ID]
{
    if (g_h2dInfo == NULL) {
        /* Attach to the precreated H2D registry (born-zero memfd table): a
         * NULL result is a visibly broken world, never a reason to create --
         * only the infra precreate constructor may do that. */
        g_h2dInfo = (h2d_info_t(*)[MAX_DEV_ID])hccp_shm_open(HCCP_SHM_KEY_H2D, SHM_SIZE, 0);
    }
    return g_h2dInfo;
}

/* Priority 102: strictly after the infra precreate constructor (101) that
 * creates and publishes the H2D segment, so this attach cannot lose the race
 * against publication (same-priority constructors have undefined order). */
__attribute__((constructor(102))) static int ShmInitHostPid(void)
{
    if (GetH2dInfo() == NULL) {
        DRV_LOG_INFO("h2d registry attach failed in constructor");
        return -1;
    }
    DRV_LOG_INFO("Shared memory initialized: hostPid=%d, devPid=%d", g_h2dInfo[0][0].hostPid, g_h2dInfo[0][0].devPid);
    return 0;
}

static int g_halPhyId = -1;

static int HalPhyId(void)
{
    if (g_halPhyId < 0) {
        const char* s = getenv("HCCP_VM_PHYID");
        if (s != NULL) {
            g_halPhyId = atoi(s);
        }
    }
    return g_halPhyId;
}

void setHostPid(pid_t pid, int hostId, int phyId)
{
    DRV_LOG_INFO("Setting Host PID in shared memory: %d", pid);
    g_halPhyId = phyId;
    if (GetH2dInfo() == NULL) {
        DRV_LOG_INFO("setHostPid init failed");
        return;
    }
    GetH2dInfo()[hostId][phyId].hostPid = pid;
}

void setDevPid(pid_t pid, int hostId, int phyId)
{
    DRV_LOG_INFO("Setting Dev PID in shared memory: %d", pid);
    if (GetH2dInfo() == NULL) {
        DRV_LOG_INFO("setDevPid init failed");
        return;
    }
    GetH2dInfo()[hostId][phyId].devPid = pid;
}

void InitHdcId()
{
    for (int i = 0; i < MAX_DEV_ID; i++) {
        GetH2dInfo()[HOST_ID_0][i].hostPhyId = i;
        GetH2dInfo()[HOST_ID_0][i].devPhyId = 0;
        GetH2dInfo()[HOST_ID_0][i].hostLogicId = i;
        GetH2dInfo()[HOST_ID_0][i].devLogicId = 0;
    }
}

drvError_t drvGetDevNum(unsigned int* numDev) { return DRV_ERROR_NONE; }

drvError_t halQueryDevPid(struct halQueryDevpidInfo info, pid_t* devPid)
{
    if (GetH2dInfo() == NULL) {
        DRV_LOG_ERROR("halQueryDevPid init failed");
        return DRV_ERROR_INVALID_VALUE;
    }

    *devPid = GetH2dInfo()[HOST_ID_0][info.devid].devPid;
    return DRV_ERROR_NONE;
}

drvError_t halMemBindSibling(int hostPid, int aicpuPid, unsigned int vfid, unsigned int devId, unsigned int flag)
{
    return DRV_ERROR_NONE;
}

drvError_t
drvQueryProcessHostPid(int pid, unsigned int* chipId, unsigned int* vfid, unsigned int* hostPid, unsigned int* cpType)
{
    if (GetH2dInfo() == NULL) {
        DRV_LOG_ERROR("drvQueryProcessHostPid init failed");
        return DRV_ERROR_INVALID_VALUE;
    }
    for (int i = 0; i < MAX_DEV_ID; i++) {
        if (GetH2dInfo()[HOST_ID_0][i].devPid == pid) {
            if (chipId != NULL) {
                *chipId = GetH2dInfo()[HOST_ID_0][i].devPhyId;
            }
            if (vfid != NULL) {
                *vfid = 0;
            }
            if (hostPid != NULL) {
                *hostPid = GetH2dInfo()[HOST_ID_0][i].hostPid;
            }
            if (cpType != NULL) {
                *cpType = 0;
            }
            return DRV_ERROR_NONE;
        }
    }
    DRV_LOG_ERROR("hal err");
    return DRV_ERROR_INVALID_VALUE;
}

drvError_t halMemGetInfoEx(unsigned int devId, unsigned int type, struct MemInfo* info) { return DRV_ERROR_NONE; }

int halGrpQuery(GroupQueryCmdType cmd, void* inBuff, unsigned int inLen, void* outBuff, unsigned int* outLen)
{
    return 0;
}

hdcError_t drvHdcGetCapacity(struct drvHdcCapacity* capacity) { return DRV_ERROR_NONE; }

hdcError_t drvHdcClientCreate(HDC_CLIENT* client, int maxSessionNum, int serviceType, int flag)
{
    *client = calloc(1, 1);
    return DRV_ERROR_NONE;
}

hdcError_t drvHdcClientDestroy(HDC_CLIENT client)
{
    free(client);
    return DRV_ERROR_NONE;
}

hdcError_t drvHdcSessionConnect(int peerNode, int peerDevid, HDC_CLIENT client, HDC_SESSION* session)
{
    *session = calloc(1, sizeof(HdcSessionT));
    HdcSessionT* pSession = (HdcSessionT*)(*session);
    pSession->hostPid = getpid();
    return DRV_ERROR_NONE;
}

hdcError_t halHdcSessionConnectEx(int peerNode, int peerDevid, int peerPid, HDC_CLIENT client, HDC_SESSION* pSession)
{
    if (pSession == NULL) {
        DRV_LOG_ERROR("[ERR][ConnectEx] output session pointer is null");
        return DRV_ERROR_INVALID_VALUE;
    }
    if (peerDevid < 0) {
        DRV_LOG_ERROR("[ERR][ConnectEx] peer device id is invalid");
        return DRV_ERROR_INVALID_VALUE;
    }

    HdcSessionT* pSessionNode = (HdcSessionT*)calloc(1, sizeof(HdcSessionT));
    if (pSessionNode == NULL) {
        DRV_LOG_ERROR("[ERR][ConnectEx] session memory alloc failed");
        return DRV_ERROR_INVALID_VALUE;
    }

    int sock_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (sock_fd < 0) {
        DRV_LOG_ERROR("Socket create failed, errno:%d, msg:%s", errno, strerror(errno));
        free(pSessionNode);
        *pSession = NULL;
        return DRV_ERROR_INVALID_VALUE;
    }

    int host_phy_id = -1;
    for (int i = 0; i < MAX_DEV_ID; i++) {
        if (GetH2dInfo()[HOST_ID_0][i].devPid == peerPid) {
            host_phy_id = (int)GetH2dInfo()[HOST_ID_0][i].hostPhyId;
            break;
        }
    }
    if (host_phy_id < 0) {
        DRV_LOG_ERROR("[ConnectEx] no h2d devPid match for peerPid=%d (peerDevid=%d)", peerPid, peerDevid);
        free(pSessionNode);
        *pSession = NULL;
        return DRV_ERROR_INVALID_VALUE;
    }
    int listen_port = GetPeerPortByDevid(host_phy_id);
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(listen_port);
    server_addr.sin_addr.s_addr
        = HccpVmToLoopback((struct in_addr){.s_addr = htonl(HccpVmNicIp((uint32_t)host_phy_id, HCCP_VM_IFACE_H2D))})
              .s_addr;

retry:
    int ret = connect(sock_fd, (struct sockaddr*)&server_addr, sizeof(server_addr));
    if (ret < 0) {
        static __thread int ceRetryCnt = 0;
        ceRetryCnt++;
        if (ceRetryCnt >= 1500) {
            DRV_LOG_ERROR(
                "[ConnectEx] connect phy=%d port=%d peerPid=%d retry exhausted", host_phy_id, listen_port, peerPid);
            free(pSessionNode);
            *pSession = NULL;
            close(sock_fd);
            return DRV_ERROR_INVALID_VALUE;
        }
        usleep(10000);
        goto retry;
    }

    pSessionNode->sessionId = AllocSessionId();
    pSessionNode->conn_fd = sock_fd;

    pSessionNode->lastRecvStatus = 0;
    pSessionNode->isUsed = 1;

    pSessionNode->peerNode = peerNode;

    *pSession = pSessionNode;

    DRV_LOG_INFO(
        "conn success sessionId=%d, conn_fd=%d, local PID=%d, remote devID=%d, peerPid=%d, peerNode=%d"
        "listen port %d",
        pSessionNode->sessionId, sock_fd, getpid(), peerDevid, peerPid, peerNode, listen_port);

    return DRV_ERROR_NONE;
}

hdcError_t drvHdcServerCreate(int devid, int serviceType, HDC_SERVER* pServer)
{
    ServerInfoT* server = NULL;
    if (devid < 0 || devid >= MAX_DEV_ID) {
        DRV_LOG_ERROR("[ERR] devid(%d) invalid 0~%d", devid, MAX_DEV_ID - 1);
        return DRV_ERROR_INVALID_VALUE;
    }
    if (pServer == NULL) {
        DRV_LOG_ERROR("hal err");
        return DRV_ERROR_INVALID_VALUE;
    }
    server = (ServerInfoT*)calloc(1, sizeof(ServerInfoT));
    if (server == NULL) {
        DRV_LOG_ERROR("[ERR] ServerInfo memory alloc failed");
        return DRV_ERROR_INVALID_VALUE;
    }

    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        DRV_LOG_ERROR("[ERR] socket create failed: %s", strerror(errno));
        free(server);
        server = NULL;
        return DRV_ERROR_INVALID_VALUE;
    }

    int opt = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    int host_phy_id;
    drvGetDevIDByLocalDevID(devid, &host_phy_id);
    int listen_port = GetPortByServiceType(host_phy_id);
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(listen_port);
    addr.sin_addr.s_addr
        = HccpVmToLoopback((struct in_addr){.s_addr = htonl(HccpVmNicIp((uint32_t)host_phy_id, HCCP_VM_IFACE_H2D))})
              .s_addr;

    if (bind(listen_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        DRV_LOG_ERROR("[ERR] bind failed: %s", strerror(errno));
        close(listen_fd);
        free(server);
        server = NULL;
        return DRV_ERROR_INVALID_VALUE;
    }
    DRV_LOG_INFO("[SUCCESS] Server created and bound to port %d", listen_port);
    if (listen(listen_fd, LISTEN_BACKLOG) < 0) {
        DRV_LOG_ERROR("[ERR] listen failed: %s", strerror(errno));
        close(listen_fd);
        free(server);
        server = NULL;
        return DRV_ERROR_INVALID_VALUE;
    }

    DRV_LOG_INFO("[SUCCESS] Server is listening on port %d", listen_port);
    server->listen_fd = listen_fd;
    server->phyId = host_phy_id;

    *pServer = (void*)server;
    return DRV_ERROR_NONE;
}

hdcError_t drvHdcServerDestroy(HDC_SERVER server)
{
    free(server);
    return DRV_ERROR_NONE;
}

hdcError_t drvHdcSessionAccept(HDC_SERVER server, HDC_SESSION* session)
{
    ServerInfoT* pServer = (ServerInfoT*)server;
    if (pServer == NULL || session == NULL) {
        DRV_LOG_ERROR("[ERR] null pointer");
        return DRV_ERROR_INVALID_VALUE;
    }

    HdcSessionT* pSession = (HdcSessionT*)calloc(1, sizeof(HdcSessionT));
    if (pSession == NULL) {
        DRV_LOG_ERROR("[ERR] session memory alloc failed");
        return DRV_ERROR_INVALID_VALUE;
    }

    struct sockaddr_in client_addr;
    socklen_t addr_len = sizeof(client_addr);
    int conn_fd = accept(pServer->listen_fd, (struct sockaddr*)&client_addr, &addr_len);
    if (conn_fd < 0) {
        DRV_LOG_ERROR("[ERR] accept failed: %s", strerror(errno));
        free(pSession);
        return DRV_ERROR_INVALID_VALUE;
    }

    pSession->sessionId = AllocSessionId();
    pSession->conn_fd = conn_fd;
    pSession->hostPhyId = pServer->phyId;
    pSession->lastRecvStatus = 0;
    pSession->isUsed = 1;
    pSession->hostPid = g_h2dInfo[HOST_ID_0][pServer->phyId].hostPid;
    pSession->devPid = g_h2dInfo[HOST_ID_0][pServer->phyId].devPid;
    *session = pSession;

    GetSession()[pServer->phyId] = pSession;
    DRV_LOG_INFO(
        "conn succ hostPhyId=%u, sessionId=%d, conn_fd=%d", pSession->hostPhyId, pSession->sessionId,
        pSession->conn_fd);

    return DRV_ERROR_NONE;
}

hdcError_t drvHdcSessionClose(HDC_SESSION session) { return DRV_ERROR_NONE; }

hdcError_t drvHdcFreeMsg(struct drvHdcMsg* msg)
{
    if (msg == NULL) {
        DRV_LOG_ERROR(" ReuseMsg error: msg is NULL");
        return DRV_ERROR_INVALID_VALUE;
    }

    struct hdc_msg_head* pHead = container_of(msg, struct hdc_msg_head, msg);

    if (!pHead->freeBuf) {
        free(pHead);
        return DRV_ERROR_NONE;
    }

    for (int i = 0; i < msg->count; i++) {
        free(msg->bufList[i].pBuf);
    }
    free(pHead);
    DRV_LOG_INFO(" FreeMsg count=%d", msg->count);
    return DRV_ERROR_NONE;
}

hdcError_t drvHdcReuseMsg(struct drvHdcMsg* msg)
{
    if (msg == NULL) {
        DRV_LOG_ERROR(" ReuseMsg error: msg is NULL");
        return DRV_ERROR_INVALID_VALUE;
    }

    if (msg->bufList == NULL) {
        DRV_LOG_ERROR(" ReuseMsg error: bufList is NULL");
        return DRV_ERROR_INVALID_VALUE;
    }

    if (msg->count < 0) {
        DRV_LOG_ERROR(" ReuseMsg error: invalid count=%d", msg->count);
        return DRV_ERROR_INVALID_VALUE;
    }

    struct hdc_msg_head* pHead = container_of(msg, struct hdc_msg_head, msg);

    for (int i = 0; i < msg->count; i++) {
        if (pHead->freeBuf) {
            free(msg->bufList[i].pBuf);
        }
        msg->bufList[i].pBuf = NULL;
        msg->bufList[i].len = 0;
    }

    DRV_LOG_INFO(" ReuseMsg count=%d", msg->count);
    return DRV_ERROR_NONE;
}

hdcError_t drvHdcAddMsgBuffer(struct drvHdcMsg* msg, char* pBuf, int len)
{
    for (int i = 0; i < msg->count; i++) {
        if (msg->bufList[i].pBuf == NULL) {
            msg->bufList[i].pBuf = pBuf;
            msg->bufList[i].len = len;
            DRV_LOG_INFO(" AddMsgBuffer count=%d, buf=%p, len=%d", msg->count, pBuf, len);
            return DRV_ERROR_NONE;
        }
    }
    DRV_LOG_ERROR("hal err");
    return DRV_ERROR_INVALID_VALUE;
}

hdcError_t drvHdcGetMsgBuffer(struct drvHdcMsg* msg, int index, char** pBuf, int* pLen)
{
    if (index < 0 || index >= msg->count) {
        DRV_LOG_ERROR("hal err");
        return DRV_ERROR_INVALID_VALUE;
    }
    DRV_LOG_INFO(
        " GetMsgBuffer count=%d, index=%d, buf=%p, len=%d", msg->count, index, msg->bufList[index].pBuf,
        msg->bufList[index].len);
    *pBuf = msg->bufList[index].pBuf;
    *pLen = msg->bufList[index].len;
    return DRV_ERROR_NONE;
}

static int gStubIsDeviceSide = -1;
static int StubIsDeviceSide(void)
{
    if (gStubIsDeviceSide < 0) {
        gStubIsDeviceSide = (strcmp(LOG_SIDE, "DEVICE") == 0) ? 1 : 0;
    }
    return gStubIsDeviceSide;
}

static int safe_read(int fd, void* buf, size_t len, int timeoutMs, const char* tag)
{
    size_t has_read = 0;
    while (has_read < len) {
        struct pollfd pfd = {fd, POLLIN, 0};
        int pret = poll(&pfd, 1, timeoutMs);
        if (pret == 0) {
            DRV_LOG_ERROR(
                "hal recv timeout: %s fd=%d want=%zu got=%zu timeout=%dms", tag, fd, len, has_read, timeoutMs);
            return -DRV_ERROR_WAIT_TIMEOUT;
        }
        if (pret < 0 && errno == EINTR) {
            continue;
        }
        if (pret < 0) {
            DRV_LOG_ERROR("hal recv poll err: %s fd=%d errno=%d", tag, fd, errno);
            return -DRV_ERROR_INVALID_VALUE;
        }
        if (pfd.revents & (POLLHUP | POLLERR)) {
        }
        ssize_t r = read(fd, (char*)buf + has_read, len - has_read);
        if (r == 0) {
            DRV_LOG_ERROR("hal recv closed: %s fd=%d want=%zu got=%zu", tag, fd, len, has_read);
            return -DRV_ERROR_INVALID_VALUE;
        }
        if (r < 0) {
            if (errno == EINTR) {
                continue;
            }
            DRV_LOG_ERROR(
                "hal recv err: %s fd=%d errno=%d(%s) want=%zu got=%zu", tag, fd, errno, strerror(errno), len, has_read);
            return -DRV_ERROR_INVALID_VALUE;
        }
        has_read += (size_t)r;
    }
    return DRV_ERROR_NONE;
}

hdcError_t
halHdcRecv(HDC_SESSION session, struct drvHdcMsg* pMsg, int bufLen, UINT64 flag, int* recvBufCount, UINT32 timeout)
{
    (void)bufLen;
    (void)flag;
    struct hdc_msg_head* pHead = container_of(pMsg, struct hdc_msg_head, msg);
    HdcSessionT* pSession = (HdcSessionT*)session;
    hdcError_t ret = DRV_ERROR_NONE;
    int expected_count = 0;
    int received_count = 0;
    int* tmp_lens = NULL;
    int allocated_count = 0;

    if (pMsg == NULL || recvBufCount == NULL || pSession == NULL) {
        DRV_LOG_ERROR("hal err");
        return DRV_ERROR_INVALID_VALUE;
    }
    if (pSession->conn_fd < 0) {
        DRV_LOG_ERROR("hal err");
        return DRV_ERROR_INVALID_VALUE;
    }

    expected_count = pMsg->count;
    if (expected_count <= 0 || expected_count > 128) {
        DRV_LOG_ERROR(" Invalid expected count=%d from caller", expected_count);
        return DRV_ERROR_INVALID_VALUE;
    }

    *recvBufCount = 0;

    int effTimeout = StubIsDeviceSide() ? -1 : (int)timeout;
    struct timeval old_tv, new_tv;
    socklen_t old_tv_len = sizeof(old_tv);
    if (timeout > 0) {
        getsockopt(pSession->conn_fd, SOL_SOCKET, SO_RCVTIMEO, &old_tv, &old_tv_len);
        new_tv.tv_sec = effTimeout / 1000;
        new_tv.tv_usec = (effTimeout % 1000) * 1000;
        setsockopt(pSession->conn_fd, SOL_SOCKET, SO_RCVTIMEO, &new_tv, sizeof(new_tv));
    }

    ret = safe_read(pSession->conn_fd, &received_count, sizeof(int), effTimeout, "RECV-HEAD");
    if (ret != DRV_ERROR_NONE) {
        goto EXIT_RESTORE_TV;
    }

    if (received_count != expected_count) {
        DRV_LOG_ERROR(" Count Mismatch! Expected=%d, Received=%d", expected_count, received_count);
        ret = DRV_ERROR_INVALID_VALUE;
        goto EXIT_RESTORE_TV;
    }

    tmp_lens = (int*)malloc(sizeof(int) * received_count);
    if (tmp_lens == NULL) {
        DRV_LOG_ERROR("hal err");
        ret = DRV_ERROR_INVALID_VALUE;
        goto EXIT_RESTORE_TV;
    }

    ret = safe_read(pSession->conn_fd, tmp_lens, sizeof(int) * received_count, effTimeout, "RECV-LENS");
    if (ret != DRV_ERROR_NONE) {
        goto EXIT_FREE_TMP;
    }

    for (int i = 0; i < received_count; i++) {
        if (tmp_lens[i] <= 0) {
            DRV_LOG_ERROR("hal err");
            ret = DRV_ERROR_INVALID_VALUE;
            goto EXIT_FREE_ALLOCATED;
        }

        pMsg->bufList[i].len = tmp_lens[i];
        pMsg->bufList[i].pBuf = (char*)malloc(tmp_lens[i]);
        if (pMsg->bufList[i].pBuf == NULL) {
            DRV_LOG_ERROR("hal err");
            ret = DRV_ERROR_INVALID_VALUE;
            goto EXIT_FREE_ALLOCATED;
        }
        allocated_count = i + 1;

        ret = safe_read(pSession->conn_fd, pMsg->bufList[i].pBuf, tmp_lens[i], effTimeout, "RECV-BODY");
        if (ret != DRV_ERROR_NONE) {
            goto EXIT_FREE_ALLOCATED;
        }

        *recvBufCount += tmp_lens[i];
    }
    pHead->freeBuf = true;
    ret = DRV_ERROR_NONE;
    DRV_LOG_INFO(" Recv success: total=%d", *recvBufCount);
    goto EXIT_FREE_TMP;

EXIT_FREE_ALLOCATED:
    for (int i = 0; i < allocated_count; i++) {
        if (pMsg->bufList[i].pBuf != NULL) {
            free(pMsg->bufList[i].pBuf);
            pMsg->bufList[i].pBuf = NULL;
        }
    }
    pMsg->count = 0;

EXIT_FREE_TMP:
    if (tmp_lens != NULL) {
        free(tmp_lens);
    }

EXIT_RESTORE_TV:
    if (timeout > 0) {
        setsockopt(pSession->conn_fd, SOL_SOCKET, SO_RCVTIMEO, &old_tv, sizeof(old_tv));
    }
    if (ret != DRV_ERROR_NONE) {
        DRV_LOG_ERROR(" Recv failed with error %d", ret);
    }
    return ret;
}

hdcError_t halHdcSend(HDC_SESSION session, struct drvHdcMsg* pMsg, UINT64 flag, UINT32 timeout)
{
    (void)flag;
    struct hdc_msg_head* pHead = container_of(pMsg, struct hdc_msg_head, msg);
    hdcError_t ret = DRV_ERROR_NONE;
    struct timeval old_tv, new_tv;
    socklen_t old_tv_len = sizeof(old_tv);
    int* tmp_lens = NULL;
    struct iovec* iov = NULL;
    int iov_cnt = 0;
    size_t total_len = 0;
    size_t sent = 0;

    if (session == NULL || pMsg == NULL) {
        DRV_LOG_ERROR("hal err");
        return DRV_ERROR_INVALID_VALUE;
    }
    HdcSessionT* pSession = (HdcSessionT*)session;
    if (pSession->conn_fd < 0 || !pSession->isUsed) {
        DRV_LOG_ERROR("hal err");
        return DRV_ERROR_INVALID_VALUE;
    }
    if (pMsg->count <= 0 || pMsg->count > 128) {
        DRV_LOG_ERROR("hal err");
        return DRV_ERROR_INVALID_VALUE;
    }

    pHead->freeBuf = false;
    if (timeout > 0) {
        getsockopt(pSession->conn_fd, SOL_SOCKET, SO_SNDTIMEO, &old_tv, &old_tv_len);
        new_tv.tv_sec = timeout / 1000;
        new_tv.tv_usec = (timeout % 1000) * 1000;
        setsockopt(pSession->conn_fd, SOL_SOCKET, SO_SNDTIMEO, &new_tv, sizeof(new_tv));
    }

    tmp_lens = (int*)malloc(sizeof(int) * pMsg->count);
    if (tmp_lens == NULL) {
        DRV_LOG_ERROR("hal err");
        ret = DRV_ERROR_INVALID_VALUE;
        goto EXIT_RESTORE_TV;
    }
    for (int i = 0; i < pMsg->count; i++) {
        if (pMsg->bufList[i].pBuf == NULL || pMsg->bufList[i].len <= 0) {
            DRV_LOG_ERROR("hal err");
            ret = DRV_ERROR_INVALID_VALUE;
            goto EXIT_FREE_TMP;
        }
        tmp_lens[i] = pMsg->bufList[i].len;
    }

    iov_cnt = 1 + 1 + pMsg->count;
    iov = (struct iovec*)malloc(sizeof(struct iovec) * iov_cnt);
    if (iov == NULL) {
        DRV_LOG_ERROR("hal err");
        ret = DRV_ERROR_INVALID_VALUE;
        goto EXIT_FREE_TMP;
    }

    iov[0].iov_base = &pMsg->count;
    iov[0].iov_len = sizeof(int);

    iov[1].iov_base = tmp_lens;
    iov[1].iov_len = sizeof(int) * pMsg->count;

    for (int i = 0; i < pMsg->count; i++) {
        iov[2 + i].iov_base = pMsg->bufList[i].pBuf;
        iov[2 + i].iov_len = pMsg->bufList[i].len;
    }

    total_len = 0;
    for (int i = 0; i < iov_cnt; i++)
        total_len += iov[i].iov_len;

    sent = 0;
    int iov_idx = 0;
    size_t iov_offset = 0;

    while (sent < total_len) {
        struct iovec* cur_iov = &iov[iov_idx];
        cur_iov->iov_base = (char*)cur_iov->iov_base + iov_offset;
        cur_iov->iov_len -= iov_offset;

        ssize_t w_ret = writev(pSession->conn_fd, cur_iov, iov_cnt - iov_idx);
        if (w_ret < 0) {
            if (errno == EINTR)
                continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                DRV_LOG_ERROR(
                    "hal send timeout: fd=%d total=%zu sent=%zu timeout=%ums", pSession->conn_fd, total_len, sent,
                    timeout);
                ret = -DRV_ERROR_WAIT_TIMEOUT;
                goto EXIT_FREE_IOV;
            }
            DRV_LOG_ERROR(
                "hal send err: fd=%d errno=%d(%s) total=%zu sent=%zu", pSession->conn_fd, errno, strerror(errno),
                total_len, sent);
            ret = DRV_ERROR_INVALID_VALUE;
            goto EXIT_FREE_IOV;
        }

        sent += w_ret;

        size_t remaining = w_ret;
        while (iov_idx < iov_cnt && remaining >= iov[iov_idx].iov_len) {
            remaining -= iov[iov_idx].iov_len;
            iov_idx++;
            iov_offset = 0;
        }
        if (iov_idx < iov_cnt) {
            iov_offset = remaining;
        }
    }

    DRV_LOG_INFO("Send success: total=%zd", sent);
    ret = DRV_ERROR_NONE;

EXIT_FREE_IOV:
    if (iov != NULL)
        free(iov);
EXIT_FREE_TMP:
    if (tmp_lens != NULL)
        free(tmp_lens);
EXIT_RESTORE_TV:
    if (timeout > 0) {
        setsockopt(pSession->conn_fd, SOL_SOCKET, SO_SNDTIMEO, &old_tv, sizeof(old_tv));
    }
    return ret;
}

hdcError_t drvHdcAllocMsg(HDC_SESSION session, struct drvHdcMsg** ppMsg, int count)
{
    struct hdc_msg_head* pHead = calloc(1, sizeof(struct hdc_msg_head) + count * sizeof(struct drvHdcMsgBuf));
    pHead->freeBuf = false;
    *ppMsg = &(pHead->msg);
    if (*ppMsg == NULL) {
        DRV_LOG_ERROR("hal err");
        return DRV_ERROR_INVALID_VALUE;
    }
    (*ppMsg)->count = count;
    DRV_LOG_INFO(" AllocMsg success: session=%d, count=%d", ((HdcSessionT*)session)->sessionId, count);
    return DRV_ERROR_NONE;
}

hdcError_t drvHdcSetSessionReference(HDC_SESSION session) { return DRV_ERROR_NONE; }

drvError_t drvGetProcessSign(struct process_sign* sign) { return DRV_ERROR_NONE; }

pid_t drvDeviceGetBareTgid(void) { return DRV_ERROR_NONE; }

drvError_t halNotifyGetInfo(uint32_t devId, uint32_t tsId, uint32_t type, uint32_t* val)
{
    *val = 1048576;
    return DRV_ERROR_NONE;
}

drvError_t halMemAlloc(void** pp, unsigned long long size, unsigned long long flag)
{
    if (pp == NULL) {
        return DRV_ERROR_PARA_ERROR;
    }
    unsigned long long virtMask = ((unsigned long long)1 << MEM_VIRT_WIDTH) - 1;
    unsigned long long virtType = (flag >> MEM_VIRT_BIT) & virtMask;
    if (virtType == MEM_DEV_VAL) {
        if (HalPhyId() < 0 || size == 0 || size > (unsigned long long)HCCP_DEVMEM_SIZE) {
            *pp = NULL;
            return DRV_ERROR_PARA_ERROR;
        }
        *pp = hccp_devmem_alloc(HalPhyId(), (size_t)size);
        if (*pp == NULL) {
            return DRV_ERROR_OUT_OF_MEMORY;
        }
        return DRV_ERROR_NONE;
    }
    *pp = malloc(size);
    return DRV_ERROR_NONE;
}

drvError_t halMemFree(void* pp)
{
    if (pp == NULL) {
        return DRV_ERROR_NONE;
    }
    /* Decode-first routing, two defense layers with distinct jobs: the
     * predicate answers address validity (a slot DVA anywhere in the segment
     * passes it; headers are rejected by construction), and the phy equality
     * answers ownership (a foreign zone's slot DVA is a VALID slot address --
     * only the ownership check stops it). Everything that fails either layer
     * falls to the heap path exactly as before. */
    int phy = 0;
    if (HalPhyId() >= 0 && hccp_devmem_dva_decode((uint64_t)(uintptr_t)pp, &phy, NULL) && phy == HalPhyId()) {
        hccp_devmem_free(phy, pp);
        return DRV_ERROR_NONE;
    }
    free(pp);
    return DRV_ERROR_NONE;
}

drvError_t halEschedSubmitEvent(uint32_t devId, struct event_summary* event) { return DRV_ERROR_NONE; }

drvError_t halMemCtl(int type, void* paramValue, size_t paramValueSize, void* outValue, size_t* outSizeRet)
{
    if (type == CTRL_TYPE_SUPPORT_FEATURE && outValue != NULL) {
        unsigned long long* feat = (unsigned long long*)outValue;
        *feat = (1ULL << CTRL_SUPPORT_PCIE_BAR_HUGE_MEM_BIT);
        if (outSizeRet != NULL) {
            *outSizeRet = sizeof(unsigned long long);
        }
        return DRV_ERROR_NONE;
    }
    if (type == CTRL_TYPE_ADDR_MAP && paramValue != NULL && outValue != NULL
        && paramValueSize >= sizeof(struct AddrMapInPara)) {
        struct AddrMapInPara* in = (struct AddrMapInPara*)paramValue;
        if (in->addr_type != DEV_MEM_MAP_HOST || HalPhyId() < 0) {
            return DRV_ERROR_OPER_NOT_PERMITTED;
        }
        size_t winLen = 0;
        void* hva = hccp_devmem_map_host(HalPhyId(), &winLen);
        if (hva == NULL) {
            return DRV_ERROR_OPER_NOT_PERMITTED;
        }
        struct AddrMapOutPara* out = (struct AddrMapOutPara*)outValue;
        out->ptr = (unsigned long long)(uintptr_t)hva;
        out->len = (unsigned long long)winLen;
        if (outSizeRet != NULL) {
            *outSizeRet = sizeof(struct AddrMapOutPara);
        }
        return DRV_ERROR_NONE;
    }
    if (type == CTRL_TYPE_ADDR_UNMAP && paramValue != NULL && paramValueSize >= sizeof(struct AddrUnmapInPara)) {
        struct AddrUnmapInPara* in = (struct AddrUnmapInPara*)paramValue;
        if (in->addr_type != DEV_MEM_MAP_HOST) {
            return DRV_ERROR_OPER_NOT_PERMITTED;
        }
        hccp_devmem_unmap_host((void*)(uintptr_t)in->ptr);
        return DRV_ERROR_NONE;
    }
    return DRV_ERROR_NONE;
}

/* halBuffAllocAlignEx backs the SVM share-pool allocations of the rs paths
 * (rs_ping_urma.c RsPingCommonInitSegCb ping jetty send/recv payload buffers,
 * rs_ub.c qp share-info buffer). The ST urma data plane consumes these buffers
 * locally: jetty SEND copies the payload from the sender's segment into the shm
 * ring and the peer's jfr drain copies into its own registered buffer, so
 * process-local aligned heap memory is a faithful stand-in for the device SVM
 * pool. The former no-op stub returned success without touching *buff, which
 * made RsPingInit fail with a NULL seg va (urma_register_seg addr:0x0). */
int halBuffAllocAlignEx(uint64_t size, unsigned int align, unsigned long flag, int grpId, void** buff)
{
    (void)flag;
    (void)grpId;
    if (buff == NULL || size == 0) {
        return DRV_ERROR_PARA_ERROR;
    }
    size_t effAlign = (align < sizeof(void*)) ? sizeof(void*) : (size_t)align;
    void* p = NULL;
    if (posix_memalign(&p, effAlign, (size_t)size) != 0) {
        *buff = NULL;
        return DRV_ERROR_OUT_OF_MEMORY;
    }
    *buff = p;
    return DRV_ERROR_NONE;
}

int halBuffFree(void* buff)
{
    free(buff); /* pairs with halBuffAllocAlignEx; free(NULL) is a no-op */
    return DRV_ERROR_NONE;
}

drvError_t halBindCgroup(BIND_CGROUP_TYPE bindType) { return DRV_ERROR_NONE; }

drvError_t drvGetPlatformInfo(uint32_t* info) { return DRV_ERROR_NONE; }

drvError_t halGetDeviceInfo(uint32_t devId, int32_t moduleType, int32_t infoType, int64_t* value)
{
    switch (moduleType) {
        case MODULE_TYPE_CCPU:
            *value = 1;
            break;
        case MODULE_TYPE_DCPU:
            *value = 1;
            break;
        case MODULE_TYPE_AICPU:
            *value = 1;
            break;
        case MODULE_TYPE_SYSTEM:
            if (infoType == INFO_TYPE_BOARD_ID) {
                *value = 0x10;
            } else if (infoType == INFO_TYPE_VERSION) {
                *value = 0;
            }
            break;
        default:
            DRV_LOG_ERROR("hal err");
            *value = 0;
            return DRV_ERROR_INVALID_VALUE;
    }
    return DRV_ERROR_NONE;
}

drvError_t halGetChipInfo(unsigned int devId, halChipInfo* chipInfo)
{
    char name[] = "950DT";
    strcpy(chipInfo->name, name);
    return DRV_ERROR_NONE;
}

drvError_t halSensorNodeRegister(uint32_t devid, struct halSensorNodeCfg* cfg, uint64_t* handle)
{
    return DRV_ERROR_NONE;
}

drvError_t halSensorNodeUnregister(uint32_t devid, uint64_t handle) { return DRV_ERROR_NONE; }

drvError_t halSensorNodeUpdateState(uint32_t devid, uint64_t handle, int val, halGeneralEventType_t assertion)
{
    return DRV_ERROR_NONE;
}

drvError_t halEschedAttachDevice(uint32_t devId) { return DRV_ERROR_NONE; }

drvError_t halEschedCreateGrp(uint32_t devId, uint32_t grpId, GROUP_TYPE type) { return DRV_ERROR_NONE; }

drvError_t
halEschedSubscribeEvent(unsigned int devId, unsigned int grpId, unsigned int threadId, unsigned long long eventBitmap)
{
    return DRV_ERROR_NONE;
}

drvError_t
halEschedWaitEvent(uint32_t devId, uint32_t grpId, uint32_t threadId, int32_t timeout, struct event_info* event)
{
    (void)devId;
    (void)grpId;
    (void)threadId;
    (void)timeout;
    (void)event;
    return DRV_ERROR_NO_EVENT;
}

drvError_t halQueryDevpid(struct halQueryDevpidInfo info, pid_t* dev_pid) { return halQueryDevPid(info, dev_pid); }

/* Mirror of rs_ctx_inner.h struct UdmaVaInfo, which the resource-mapping callers
 * (rs_ub_jetty.c / rs_ub_jfc.c) pass through res_map_info_in.priv; the hal header
 * chain only forward-declares the struct, so the layout is repeated here. */
struct HalStubUdmaVaInfo {
    int resType;
    int pid;
    unsigned long long va;
    unsigned long long len;
};

/* Production maps a SoC resource (jetty sq buffer, doorbell) into the calling
 * process VA space for the USER_CTL_NORMAL jetty create path (RsMmapJettyVa).
 * The sim hands back an anonymous mapping of the requested length; the migrated
 * control-plane cases never dereference the range, they only carry the va. */
drvError_t halResAddrMapV2(unsigned int devId, struct res_map_info_in* resInfoIn, struct res_map_info_out* resInfoOut)
{
    (void)devId;
    if (resInfoIn == NULL || resInfoOut == NULL) {
        return DRV_ERROR_INVALID_VALUE;
    }
    unsigned long long len = 0;
    if (resInfoIn->priv != NULL && resInfoIn->priv_len >= sizeof(struct HalStubUdmaVaInfo)) {
        struct HalStubUdmaVaInfo* vaInfo = (struct HalStubUdmaVaInfo*)resInfoIn->priv;
        len = vaInfo->len;
    }
    if (len == 0) {
        len = 4096;
    }
    void* va = mmap(NULL, (size_t)len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (va == MAP_FAILED) {
        DRV_LOG_ERROR("halResAddrMapV2 mmap failed errno:%d len:%llu", errno, len);
        return DRV_ERROR_OUT_OF_MEMORY;
    }
    resInfoOut->va = (unsigned long)(uintptr_t)va;
    resInfoOut->len = (unsigned int)len;
    DRV_LOG_INFO(
        "halResAddrMapV2 res_type:%u res_id:%u len:%llu va:0x%lx", resInfoIn->res_type, resInfoIn->res_id, len,
        resInfoOut->va);
    return DRV_ERROR_NONE;
}

/* Unmap counterpart of halResAddrMapV2; the sim unmaps the previously returned
 * anonymous range (unmap failures are ignored, matching the tolerant cleanup). */
drvError_t halResAddrUnmapV2(unsigned int devId, struct res_map_info_in* resInfoIn)
{
    (void)devId;
    if (resInfoIn == NULL || resInfoIn->priv == NULL || resInfoIn->priv_len < sizeof(struct HalStubUdmaVaInfo)) {
        return DRV_ERROR_NONE;
    }
    struct HalStubUdmaVaInfo* vaInfo = (struct HalStubUdmaVaInfo*)resInfoIn->priv;
    if (vaInfo->va != 0 && vaInfo->len != 0) {
        (void)munmap((void*)(uintptr_t)vaInfo->va, (size_t)vaInfo->len);
    }
    return DRV_ERROR_NONE;
}
