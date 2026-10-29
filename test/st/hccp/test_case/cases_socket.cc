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
#include <chrono>
#include <unordered_map>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <errno.h>

/* Smoke test: a single node (phyId 60) exercises only the framework's host child
 * spawn/wait/reclaim path and logging, without calling any Ra/rt interfaces. */
TEST_F(HccpSystemTest, HelloWorld)
{
    RunTests({60}, [](int phyid) -> int {
        TEST_LOG_INFO(phyid, "hello world!");
        return 0;
    });
}

/* Two-node TLS switch query: CaseOfflineInitNoRdv (rtOpenNetService spawns the device
 * process + RaInit, no rendezvous channel), then RaGetTlsEnable queries the TLS
 * enablement state, verifying the RPC query path and multi-host coexistence. */
TEST_F(HccpSystemTest, MultiHostTlsTest)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        CaseEnv env;
        int ret = CaseOfflineInitNoRdv(env, phyid, HCCP_VM_IFACE_SOCKET);
        if (ret != 0) {
            return ret;
        }

        bool tlsEnable = false;
        RaInfo info = {};
        info.mode = 1;
        info.phyId = (unsigned int)phyid;
        ret = RaGetTlsEnable(&info, &tlsEnable);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaGetTlsEnable failed ret=%d", ret);
            goto err;
        }

        TEST_LOG_INFO(phyid, "TLS enable result: %d", tlsEnable);
    err:
        CaseCleanup(env);
        return ret;
    });
}

/* Two-node socket send/recv (OFFLINE/HDC path, via the device process): CaseOfflineInit
 * establishes the rendezvous channel (listen/connect/hello exchange, port 15666); both
 * sides send one message and receive the peer's, memcmp-verifying the exchanged content,
 * which validates bidirectional RaSocket data-plane connectivity. */
TEST_F(HccpSystemTest, d2dSocketSendRecv)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        const char* expected = (phyid == 1) ? "Hello HCCP client!" : "Hello HCCP sever!";
        CaseEnv env;
        int ret = CaseOfflineInit(env, phyid, HCCP_VM_IFACE_SOCKET, 15666, "HELLO WORD!");
        if (ret != 0) {
            return ret;
        }

        char sendBuf[32], recvBuf[256] = {0};
        (void)snprintf(sendBuf, sizeof(sendBuf), "Hello HCCP %s!", (phyid == 1) ? "sever" : "client");

        ret = RaSocketSendAll(env.sock.fdHandle, sendBuf, strlen(sendBuf) + 1);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "send failed ret=%d", ret);
            goto err;
        }
        ret = RaSocketRecvAll(env.sock.fdHandle, recvBuf, strlen(expected) + 1);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "recv failed ret=%d", ret);
            goto err;
        }
        if (memcmp(recvBuf, expected, strlen(expected) + 1) != 0) {
            TEST_LOG_ERROR(phyid, "memcmp FAIL exp=%s act=%s", expected, recvBuf);
            ret = -1;
            goto err;
        }
        TEST_LOG_INFO(phyid, "socket memcmp PASS: recv=%s", recvBuf);
        ret = 0;
    err:
        CaseCleanup(env);
        return ret;
    });
}

/* Two-node socket send/recv (PEER_ONLINE host-direct path, NIC on the host side, no device
 * process): CasePeerInit performs peer-mode rendezvous over the RoCE interface (port 15673);
 * flow and verification are identical to d2dSocketSendRecv, forming an OFFLINE/PEER
 * network-mode behavior pair. */
TEST_F(HccpSystemTest, h2hSocketSendRecv)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        const char* expected = (phyid == 1) ? "Hello host HCCP client!" : "Hello host HCCP sever!";
        CaseEnv env;
        int ret = CasePeerInit(env, phyid, HCCP_VM_IFACE_ROCE, 15673, "H2H SOCKET!");
        if (ret != 0) {
            return ret;
        }

        char sendBuf[32], recvBuf[256] = {0};
        (void)snprintf(sendBuf, sizeof(sendBuf), "Hello host HCCP %s!", (phyid == 1) ? "sever" : "client");

        ret = RaSocketSendAll(env.sock.fdHandle, sendBuf, strlen(sendBuf) + 1);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "send failed ret=%d", ret);
            goto err;
        }
        ret = RaSocketRecvAll(env.sock.fdHandle, recvBuf, strlen(expected) + 1);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "recv failed ret=%d", ret);
            goto err;
        }
        if (memcmp(recvBuf, expected, strlen(expected) + 1) != 0) {
            TEST_LOG_ERROR(phyid, "memcmp FAIL exp=%s act=%s", expected, recvBuf);
            ret = -1;
            goto err;
        }
        TEST_LOG_INFO(phyid, "peer socket memcmp PASS: recv=%s", recvBuf);
        ret = 0;
    err:
        CaseCleanup(env);
        return ret;
    });
}

/* 32-node full mesh (HCCP_VM_MESH_NPROC tunable 2..32, default 32): each node spawns its
 * device process, whitelists all peers in batches, listens on port 15680 and batch-connects
 * to the other 31 nodes; RaGetSockets polling converges 31 client + 31 server connections;
 * hello messages self-identify peers and build an injective fd-to-peer map; ROOT (phyId 0)
 * gathers the ProcInfo{phyId,pid} membership table and broadcasts it in full. Every node
 * (including ROOT) then self-verifies: its own slot must hold its own {phyId,pid} and all
 * 32 entries must be valid. */
TEST_F(HccpSystemTest, d2dFullMesh32)
{
    std::vector<int> test_phyids;
    int meshN = 32;
    if (const char* e = getenv("HCCP_VM_MESH_NPROC")) {
        long v = atol(e);
        if (v >= 2 && v <= 32)
            meshN = (int)v;
    }
    for (int i = 0; i < meshN; i++) {
        test_phyids.push_back(i);
    }
    const int NPROC = meshN;
    const int ROOT = 0;
    const unsigned short meshPort = HccpStCasePort(15680); /* direct-bind family: displace at the
                                                            * per-process origin shared by listener
                                                            * and connectors (t5 dual-runner) */

    struct ProcInfo {
        int phyId;
        int pid;
    };

    RunTests(test_phyids, [&](int phyid) -> int {
        TEST_LOG_INFO(phyid, "d2dFullMesh32 start, pid=%d", getpid());

        int ret = 0;
        void* socketHandle = nullptr;
        struct HelloMsg {
            uint32_t magic;
            int32_t phyId;
        };
        const uint32_t HELLO_MAGIC = 0x4D455348u;
        struct SocketInfoT socks[64];
        struct SocketInfoT srvSocks[64];
        int nCli = 0;
        int nSrv = 0;
        int fd2peer[64];
        void* srvFdByPeer[65] = {nullptr};
        void* srvEntryFd[64] = {nullptr};
        int srvEntryCnt = 0;
        struct ProcInfo table[64] = {};
        struct ProcInfo mine = {};
        char nsDevId[32];
        (void)snprintf(nsDevId, sizeof(nsDevId), "--deviceId=%d", phyid);
        char nsPid[32];
        (void)snprintf(nsPid, sizeof(nsPid), "--pid=%d", (int)getpid());
        rtProcExtParam nsExt[2] = {{nsDevId, strlen(nsDevId)}, {nsPid, strlen(nsPid)}};
        rtNetServiceOpenArgs nsArgs = {nsExt, 2};
        int netOpened = 0;
        if (rtOpenNetService(&nsArgs) != RT_ERROR_NONE) {
            TEST_LOG_ERROR(phyid, "rtOpenNetService failed");
            return 1;
        }
        netOpened = 1;

        struct RaInitConfig config = {};
        config.phyId = (unsigned int)phyid;
        config.nicPosition = 1;
        config.hdcType = 18;
        config.enableHdcAsync = false;
        ret = RaInit(&config);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaInit failed ret=%d", ret);
            (void)rtCloseNetService();
            return ret;
        }

        struct SocketInitInfoT socketInit = {};
        socketInit.rdevInfo.phyId = (unsigned int)phyid;
        socketInit.rdevInfo.family = AF_INET;
        socketInit.rdevInfo.localIp.addr.s_addr = htonl(HccpVmNicIp((uint32_t)phyid, HCCP_VM_IFACE_SOCKET));
        struct SocketInitInfoT vmSocketInit = socketInit;
        vmSocketInit.rdevInfo.localIp.addr = HccpVmToLoopback(socketInit.rdevInfo.localIp.addr);

        ret = RaSocketInitV1((int)config.nicPosition, vmSocketInit, &socketHandle);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaSocketInitV1 failed ret=%d", ret);
            RaDeinit(&config);
            (void)rtCloseNetService();
            return ret;
        }

        struct SocketListenInfoT listenInfo = {};
        {
            struct SocketWlistInfoT wl[16];
            int wlCnt = 0;
            for (int peer = 0; peer < NPROC; peer++) {
                if (peer == phyid)
                    continue;
                memset(&wl[wlCnt], 0, sizeof(wl[wlCnt]));
                struct in_addr lp = HccpVmToLoopback(in_addr{htonl(HccpVmNicIp((uint32_t)peer, HCCP_VM_IFACE_SOCKET))});
                wl[wlCnt].remoteIp.addr.s_addr = lp.s_addr;
                wl[wlCnt].connLimit = 4;
                (void)snprintf(wl[wlCnt].tag, sizeof(wl[wlCnt].tag), "mesh%d", meshPort);
                wlCnt++;
                if (wlCnt == 16) {
                    ret = RaSocketWhiteListAdd(socketHandle, wl, wlCnt);
                    if (ret != 0) {
                        TEST_LOG_ERROR(phyid, "WhiteListAdd failed ret=%d", ret);
                        goto err_deinit;
                    }
                    wlCnt = 0;
                }
            }
            if (wlCnt > 0) {
                ret = RaSocketWhiteListAdd(socketHandle, wl, wlCnt);
                if (ret != 0) {
                    TEST_LOG_ERROR(phyid, "WhiteListAdd(tail) failed ret=%d", ret);
                    goto err_deinit;
                }
            }
        }

        listenInfo.socketHandle = socketHandle;
        listenInfo.port = meshPort;
        ret = RaSocketListenStart(&listenInfo, 1);
        if (ret != 0) {
            TEST_LOG_ERROR(phyid, "RaSocketListenStart failed ret=%d", ret);
            goto err_deinit;
        }

        {
            struct SocketConnectInfoT cn[16];
            int cnCnt = 0;
            for (int peer = 0; peer < NPROC; peer++) {
                if (peer == phyid)
                    continue;
                memset(&cn[cnCnt], 0, sizeof(cn[cnCnt]));
                cn[cnCnt].socketHandle = socketHandle;
                cn[cnCnt].port = meshPort;
                struct in_addr rp = HccpVmToLoopback(in_addr{htonl(HccpVmNicIp((uint32_t)peer, HCCP_VM_IFACE_SOCKET))});
                cn[cnCnt].remoteIp.addr.s_addr = rp.s_addr;
                (void)snprintf(cn[cnCnt].tag, sizeof(cn[cnCnt].tag), "mesh%d", meshPort);
                cnCnt++;
                if (cnCnt == 16) {
                    ret = RaSocketBatchConnect(cn, cnCnt);
                    if (ret != 0) {
                        TEST_LOG_ERROR(phyid, "BatchConnect failed ret=%d", ret);
                        goto err_deinit;
                    }
                    cnCnt = 0;
                }
            }
            if (cnCnt > 0) {
                ret = RaSocketBatchConnect(cn, cnCnt);
                if (ret != 0) {
                    TEST_LOG_ERROR(phyid, "BatchConnect(tail) failed ret=%d", ret);
                    goto err_deinit;
                }
            }
        }

        memset(socks, 0, sizeof(socks));
        memset(srvSocks, 0, sizeof(srvSocks));
        nCli = 0;
        nSrv = 0;
        {
            for (int peer = 0; peer < NPROC; peer++) {
                if (peer == phyid)
                    continue;
                struct SocketInfoT* ent = &socks[nCli];
                ent->socketHandle = socketHandle;
                ent->fdHandle = nullptr;
                ent->status = 0;
                struct in_addr rp = HccpVmToLoopback(in_addr{htonl(HccpVmNicIp((uint32_t)peer, HCCP_VM_IFACE_SOCKET))});
                ent->remoteIp.addr.s_addr = rp.s_addr;
                (void)snprintf(ent->tag, sizeof(ent->tag), "mesh%d", meshPort);
                nCli++;
            }
            for (int peer = 0; peer < NPROC; peer++) {
                if (peer == phyid)
                    continue;
                struct SocketInfoT* ent = &srvSocks[nSrv];
                ent->socketHandle = socketHandle;
                ent->fdHandle = nullptr;
                ent->status = 0;
                struct in_addr rp = HccpVmToLoopback(in_addr{htonl(HccpVmNicIp((uint32_t)peer, HCCP_VM_IFACE_SOCKET))});
                ent->remoteIp.addr.s_addr = rp.s_addr;
                (void)snprintf(ent->tag, sizeof(ent->tag), "mesh%d", meshPort);
                nSrv++;
            }
            int cliDone = 0, srvDone = 0;
            int cliGot[64] = {0};
            for (int spin = 0; spin < 2000 && (cliDone < nCli || srvDone < nSrv); spin++) {
                for (int b = 0; b < nCli && cliDone < nCli; b += 16) {
                    int cnt = (nCli - b < 16) ? (nCli - b) : 16;
                    struct SocketInfoT probe[16];
                    int idxMap[16];
                    int pcnt = 0;
                    for (int k = 0; k < cnt; k++) {
                        if (cliGot[b + k])
                            continue;
                        probe[pcnt] = socks[b + k];
                        probe[pcnt].fdHandle = nullptr;
                        probe[pcnt].status = 0;
                        idxMap[pcnt] = b + k;
                        pcnt++;
                    }
                    if (pcnt == 0)
                        continue;
                    unsigned int num = (unsigned int)pcnt;
                    ret = RaGetSockets(1, probe, pcnt, &num);
                    if (ret < 0) {
                        TEST_LOG_ERROR(phyid, "RaGetSockets(cli) failed ret=%d", ret);
                        ret = -1;
                        goto err_deinit;
                    }
                    for (int k = 0; k < pcnt; k++) {
                        if (probe[k].status == 1 && probe[k].fdHandle != nullptr) {
                            socks[idxMap[k]] = probe[k];
                            cliGot[idxMap[k]] = 1;
                            cliDone++;
                        }
                    }
                }
                for (int b = 0; b < nSrv && srvDone < nSrv; b += 16) {
                    int cnt = (nSrv - b < 16) ? (nSrv - b) : 16;
                    struct SocketInfoT probe[16];
                    for (int k = 0; k < cnt; k++) {
                        probe[k] = srvSocks[b + k];
                        probe[k].fdHandle = nullptr;
                        probe[k].status = 0;
                    }
                    unsigned int num = (unsigned int)cnt;
                    ret = RaGetSockets(0, probe, cnt, &num);
                    if (ret == 0 && num > 0) {
                        for (int k = 0; k < (int)num; k++) {
                            if (probe[k].fdHandle == nullptr)
                                continue;
                            srvEntryFd[srvEntryCnt++] = probe[k].fdHandle;
                            srvDone++;
                        }
                    }
                }
                if (spin % 10 == 0) {
                    TEST_LOG_INFO(
                        phyid, "collecting... spin=%d cli=%d/%d srv=%d/%d", spin, cliDone, nCli, srvDone, nSrv);
                }
                usleep(5000);
            }
            if (cliDone < nCli || srvDone < nSrv) {
                TEST_LOG_ERROR(phyid, "connections incomplete: cli=%d/%d srv=%d/%d", cliDone, nCli, srvDone, nSrv);
                ret = -1;
                goto err_deinit;
            }
        }
        TEST_LOG_INFO(phyid, "mesh established: %d client + %d server connections", nCli, nSrv);
        /* The fd table dump only aids debugging the fd2peer mapping below; keep the
         * success path silent and set HCCP_VM_DUMP_FD=1 to enable it. */
        if (getenv("HCCP_VM_DUMP_FD") != nullptr) {
            for (int i = 0; i < nCli; i++) {
                TEST_LOG_INFO(phyid, "cli[%d] fd=%p", i, socks[i].fdHandle);
            }
            for (int i = 0; i < nSrv; i++) {
                TEST_LOG_INFO(phyid, "srv[%d] fd=%p", i, srvSocks[i].fdHandle);
            }
        }

        memset(fd2peer, 0, sizeof(fd2peer));
        {
            {
                int sent[64] = {0};
                int sentCnt = 0;
                for (int round = 0; round < 2000 && sentCnt < nCli; round++) {
                    for (int i = 0; i < nCli && sentCnt < nCli; i++) {
                        if (sent[i])
                            continue;
                        struct HelloMsg out = {HELLO_MAGIC, phyid};
                        unsigned long long sz = 0;
                        ret = RaSocketSend(socks[i].fdHandle, &out, sizeof(out), &sz);
                        if (ret == 0 && sz == sizeof(out)) {
                            sent[i] = 1;
                            sentCnt++;
                        } else if (ret != 0 && ret != 128201 && ret != 128203 && ret != 128207 && ret != 228205) {
                            TEST_LOG_ERROR(phyid, "hello send[%d] failed ret=%d", i, ret);
                            goto err_deinit;
                        }
                    }
                    usleep(1000);
                }
                if (sentCnt < nCli) {
                    TEST_LOG_ERROR(phyid, "hello send incomplete: %d/%d", sentCnt, nCli);
                    ret = -1;
                    goto err_deinit;
                }
            }
            {
                int helloGot[64] = {0};
                int got = 0;
                int srvTotal = srvEntryCnt;
                for (int round = 0; round < 2000 && got < srvTotal; round++) {
                    for (int i = 0; i < srvTotal && got < srvTotal; i++) {
                        if (helloGot[i] || srvEntryFd[i] == nullptr)
                            continue;
                        struct HelloMsg in = {0, 0};
                        unsigned long long sz = 0;
                        ret = RaSocketRecv(srvEntryFd[i], &in, sizeof(in), &sz);
                        if (ret == 0 && sz == sizeof(in)) {
                            if (in.magic != HELLO_MAGIC || in.phyId < 0 || in.phyId >= NPROC || in.phyId == phyid
                                || srvFdByPeer[in.phyId] != nullptr) {
                                TEST_LOG_ERROR(phyid, "srv hello[%d] bad: magic=0x%x from=%d", i, in.magic, in.phyId);
                                ret = -1;
                                goto err_deinit;
                            }
                            helloGot[i] = 1;
                            got++;
                            srvFdByPeer[in.phyId] = srvEntryFd[i];
                            struct HelloMsg out = {HELLO_MAGIC, phyid};
                            ret = RaSocketSendAll(srvEntryFd[i], &out, sizeof(out));
                            if (ret != 0) {
                                TEST_LOG_ERROR(phyid, "srv hello reply[%d] failed ret=%d", i, ret);
                                goto err_deinit;
                            }
                        } else if (ret != 0 && ret != 128201 && ret != 128203 && ret != 128207 && ret != 228205) {
                            TEST_LOG_ERROR(phyid, "srv hello recv[%d] failed ret=%d", i, ret);
                            ret = -1;
                            goto err_deinit;
                        }
                    }
                    usleep(1000);
                }
                if (got < srvTotal) {
                    TEST_LOG_ERROR(phyid, "srv hello incomplete: %d/%d", got, srvTotal);
                    ret = -1;
                    goto err_deinit;
                }
            }
            {
                int replyGot[64] = {0};
                struct HelloMsg reply[64];
                int got = 0;
                for (int round = 0; round < 2000 && got < nCli; round++) {
                    for (int i = 0; i < nCli && got < nCli; i++) {
                        if (replyGot[i])
                            continue;
                        unsigned long long sz = 0;
                        ret = RaSocketRecv(socks[i].fdHandle, &reply[i], sizeof(struct HelloMsg), &sz);
                        if (ret == 0 && sz == sizeof(struct HelloMsg)) {
                            replyGot[i] = 1;
                            got++;
                        } else if (ret != 0 && ret != 128201 && ret != 128203 && ret != 128207 && ret != 228205) {
                            TEST_LOG_ERROR(phyid, "hello recv[%d] failed ret=%d", i, ret);
                            ret = -1;
                            goto err_deinit;
                        }
                    }
                    usleep(1000);
                }
                if (got < nCli) {
                    TEST_LOG_ERROR(phyid, "cli reply incomplete: %d/%d", got, nCli);
                    ret = -1;
                    goto err_deinit;
                }
                for (int i = 0; i < nCli; i++) {
                    if (reply[i].magic != HELLO_MAGIC || reply[i].phyId < 0 || reply[i].phyId >= NPROC
                        || reply[i].phyId == phyid) {
                        TEST_LOG_ERROR(
                            phyid, "cli reply[%d] bad: magic=0x%x peer=%d", i, reply[i].magic, reply[i].phyId);
                        ret = -1;
                        goto err_deinit;
                    }
                    fd2peer[i] = reply[i].phyId;
                }
            }
            for (int i = 0; i < nCli; i++) {
                for (int j = i + 1; j < nCli; j++) {
                    if (fd2peer[i] == fd2peer[j]) {
                        TEST_LOG_ERROR(phyid, "dup peer %d on fd[%d]/[%d]", fd2peer[i], i, j);
                        ret = -1;
                        goto err_deinit;
                    }
                }
            }
        }

        mine.phyId = phyid;
        mine.pid = getpid();
        memset(table, 0, sizeof(table));

        if (phyid != ROOT) {
            int rootIdx = -1;
            for (int i = 0; i < nCli; i++) {
                if (fd2peer[i] == ROOT) {
                    rootIdx = i;
                    break;
                }
            }
            if (rootIdx < 0) {
                TEST_LOG_ERROR(phyid, "no root fd");
                ret = -1;
                goto err_deinit;
            }
            ret = RaSocketSendAll(socks[rootIdx].fdHandle, &mine, sizeof(mine));
            if (ret != 0) {
                TEST_LOG_ERROR(phyid, "send mine→root failed ret=%d", ret);
                goto err_deinit;
            }
            if (srvFdByPeer[ROOT] == nullptr) {
                TEST_LOG_ERROR(phyid, "no root srv fd");
                ret = -1;
                goto err_deinit;
            }
            ret = RaSocketRecvAll(srvFdByPeer[ROOT], table, sizeof(table));
            if (ret != 0) {
                TEST_LOG_ERROR(phyid, "recv table←root failed ret=%d", ret);
                goto err_deinit;
            }
        } else {
            table[ROOT] = mine;
            int collected = 1;
            {
                int mineGot[65] = {0};
                for (int round = 0; round < 2000 && collected < NPROC; round++) {
                    for (int p = 0; p < NPROC && collected < NPROC; p++) {
                        if (p == phyid || mineGot[p] || srvFdByPeer[p] == nullptr)
                            continue;
                        struct ProcInfo info = {};
                        unsigned long long sz = 0;
                        ret = RaSocketRecv(srvFdByPeer[p], &info, sizeof(info), &sz);
                        if (ret == 0 && sz == sizeof(info)) {
                            mineGot[p] = 1;
                            if (info.phyId < 0 || info.phyId >= NPROC || table[info.phyId].pid != 0) {
                                TEST_LOG_ERROR(phyid, "bad/dup ProcInfo phyId=%d", info.phyId);
                                ret = -1;
                                goto err_deinit;
                            }
                            table[info.phyId] = info;
                            collected++;
                        } else if (ret != 0 && ret != 128201 && ret != 128203 && ret != 128207 && ret != 228205) {
                            TEST_LOG_ERROR(phyid, "root recv peer=%d failed ret=%d", p, ret);
                            goto err_deinit;
                        }
                    }
                    usleep(1000);
                }
            }
            if (collected != NPROC) {
                TEST_LOG_ERROR(phyid, "collected %d/%d", collected, NPROC);
                ret = -1;
                goto err_deinit;
            }
            for (int i = 0; i < nCli; i++) {
                ret = RaSocketSendAll(socks[i].fdHandle, table, sizeof(table));
                if (ret != 0) {
                    TEST_LOG_ERROR(phyid, "root bcast[%d] failed ret=%d", i, ret);
                    goto err_deinit;
                }
            }
        }

        if (table[phyid].phyId != phyid || table[phyid].pid != getpid()) {
            TEST_LOG_ERROR(
                phyid, "table self entry mismatch: slot={%d,%d} expect={%d,%d}", table[phyid].phyId, table[phyid].pid,
                phyid, (int)getpid());
            ret = -1;
            goto err_deinit;
        }
        {
            int valid = 0;
            for (int i = 0; i < NPROC; i++) {
                if (table[i].pid != 0 && table[i].phyId == i) {
                    valid++;
                }
            }
            if (valid != NPROC) {
                TEST_LOG_ERROR(phyid, "table incomplete: valid=%d/%d", valid, NPROC);
                ret = -1;
                goto err_deinit;
            }
        }
        TEST_LOG_INFO(phyid, "table verified: self slot OK, all %d entries valid", NPROC);

        ret = 0;
    err_deinit:
        RaDeinit(&config);
        if (netOpened) {
            (void)rtCloseNetService();
        }
        return ret;
    });
}

/* ==================== listen-reuse refcount guards (MR !3681) ==================== */

/* Budget note: TearDown reclaims each host child within 5 s of spawn (hccp_system_test_base.h
 * HCCP_VM_HOST_TIMEOUT_SEC), so the WHOLE case body — init, both link rounds and every
 * poll/connect/send/recv inside them — must fit in that window: keep the SUM under budget,
 * not each individual wait, otherwise the fixture SIGUSR1-kills the child and the attribution
 * log is lost. Both link rounds therefore share one deadline taken at body entry: whichever
 * round is slow converges on the deadline instead of burning its own full spin budget, so
 * slow-but-correct machines fail inside the case with an explicit timeout log rather than
 * being killed silently. The 4 s deadline leaves ~1 s for init plus the release check; the
 * per-round spin cap (3 s) alone bounds the regression path, where round 1 exhausts its
 * spins and exits well inside the window. Blocking RaSocketSendAll/RaSocketRecvAll retries
 * draw from the same window: they run only after a link appeared (polls returned quickly),
 * so a pathological retry there eats the window and the fixture kill is the backstop. */
#define LISTEN_REUSE_BUDGET_MS 4000 /* whole-case poll budget, shared by both link rounds */
#define LISTEN_REUSE_POLL_SPIN 600  /* per-round spin cap: 600 * 5ms = 3s */

/* Body-level invariants of the listen-reuse cases: everything fixed for the whole case body
 * (node identity, peer addressing, env, and the shared poll deadline) bundles here so the
 * per-round helpers stay small-signature; only what actually varies per link round (tag,
 * round) remains an explicit argument. Mirrors the CaseEnv& bundling idiom of this file. */
struct ListenReuseCtx {
    int phyid;
    uint32_t iface;
    CaseEnv& env;
    struct in_addr peerIp;
    unsigned short port;
    std::chrono::steady_clock::time_point deadline;
};

/* Shared body of the listen-reuse guard cases. Two users listen on the same IP+port
 * (user B reuses user A's listen node, mirroring HCCL's multi-comm-domain port reuse),
 * then user A stops first: the listen must survive for user B, and the client must
 * still establish the link on that port. Before MR !3681 the creator's own counter +1
 * was missing in RsListenNodeAlloc, so user A's stop tore down the still-shared listen
 * node (close listen fd + free) and the client link establishment timed out — exactly
 * the regression this body intercepts. The lifecycle is then closed with an independent
 * release check: user B stops and a plain system connect to the port must be refused
 * (ECONNREFUSED), proving the listen socket really went away — a mere re-listen
 * success could not tell a released node from a leaked one being reused. Only then is
 * the port re-listened and a second link (fresh tag) verified.
 *
 * API-path note: the two TEST_F cases below differ in the RUN path, not in the listen
 * interface version — d2d goes through the HDC device process, h2h through the peer
 * host-direct path. With the current capability tables both run paths select the V2
 * listen implementation (HDC: RaHdcSocketListenStart version table picks V2; peer:
 * RaPeerSocketListenStart calls RsSocketListenStartV2 directly). The refcount
 * lifecycle under guard (RsListenNodeAlloc create +1 / RsSocketInitListenV2 reuse +1 /
 * RsSocketListenStop last-reference teardown) is shared by both listen versions. */
/* Server-side setup: user A listens first (creates the listen node), user B listens again on
 * the same IP+port (must reuse the node), both link-round tags are whitelisted for the client,
 * and user A then stops — the step that closed the still-shared listen fd before MR !3681. */
static int ListenReuseSetupSharedListen(const ListenReuseCtx& ctx, const char* tagA, const char* tagB)
{
    struct SocketListenInfoT listenInfo = {};
    listenInfo.socketHandle = ctx.env.socketHandle;
    listenInfo.port = ctx.port;

    listenInfo.err = 0;
    int ret = RaSocketListenStart(&listenInfo, 1); /* user A: creates the listen node */
    if (ret != 0) {
        TEST_LOG_ERROR(ctx.phyid, "userA RaSocketListenStart failed ret=%d", ret);
        return ret;
    }
    listenInfo.err = 0;
    ret = RaSocketListenStart(&listenInfo, 1); /* user B: same IP+port must reuse the node */
    if (ret != 0) {
        TEST_LOG_ERROR(ctx.phyid, "userB RaSocketListenStart(reuse) failed ret=%d", ret);
        return ret;
    }

    /* whitelist the client for both link rounds before any connect happens */
    struct SocketWlistInfoT wl[2] = {};
    wl[0].remoteIp.addr.s_addr = ctx.peerIp.s_addr;
    wl[0].connLimit = 1;
    (void)snprintf(wl[0].tag, sizeof(wl[0].tag), "%s", tagA);
    wl[1] = wl[0];
    (void)snprintf(wl[1].tag, sizeof(wl[1].tag), "%s", tagB);
    ret = RaSocketWhiteListAdd(ctx.env.socketHandle, wl, 2);
    if (ret != 0) {
        TEST_LOG_ERROR(ctx.phyid, "RaSocketWhiteListAdd failed ret=%d", ret);
        return ret;
    }

    /* user A stops: with the refcount bug this closed the listen fd user B still relies on;
     * the round-1 link must catch that */
    ret = RaSocketListenStop(&listenInfo, 1);
    if (ret != 0) {
        TEST_LOG_ERROR(ctx.phyid, "userA RaSocketListenStop failed ret=%d", ret);
        return ret;
    }
    TEST_LOG_INFO(ctx.phyid, "userA stopped; userB still holds the listen on port %u", ctx.port);
    return 0;
}

/* Client-side half of one link round: batch-connect to the server's listen port on tag.
 * The server (phyid 1) does not connect; it accepts. */
static int ListenReuseClientConnect(const ListenReuseCtx& ctx, const char* tag)
{
    if (ctx.phyid == 1) {
        return 0;
    }
    struct SocketConnectInfoT conn = {};
    conn.socketHandle = ctx.env.socketHandle;
    conn.port = ctx.port;
    conn.remoteIp.addr.s_addr = ctx.peerIp.s_addr;
    (void)snprintf(conn.tag, sizeof(conn.tag), "%s", tag);
    int ret = RaSocketBatchConnect(&conn, 1);
    if (ret != 0) {
        TEST_LOG_ERROR(ctx.phyid, "RaSocketBatchConnect failed ret=%d tag=%s", ret, tag);
    }
    return ret;
}

/* One link round on tag: bounded-poll RaGetSockets until a link appears, then send this
 * side's "LR-<role>-<round>" message and verify the peer's counterpart comes back.
 * Poll exhaustion (fdHandle never appearing) is the exact — and, with the current
 * RaGetSockets, the only observable — symptom the refcount regression produces: no code
 * path in this repo reports a socket status other than RS_SOCK_STATUS_OK, so there is
 * no separate timeout branch to check here. The poll loop obeys both the per-round spin
 * cap and the whole-case deadline shared with the other round (see budget note above). */
static int ListenReuseLinkRound(const ListenReuseCtx& ctx, const char* tag, int round)
{
    const bool isServer = (ctx.phyid == 1);
    char myMsg[16] = {0};
    char peerMsg[16] = {0};
    (void)snprintf(myMsg, sizeof(myMsg), "LR-%s-%d", isServer ? "SRV" : "CLI", round);
    (void)snprintf(peerMsg, sizeof(peerMsg), "LR-%s-%d", isServer ? "CLI" : "SRV", round);

    struct SocketInfoT probe = {};
    probe.socketHandle = ctx.env.socketHandle;
    probe.remoteIp.addr.s_addr = ctx.peerIp.s_addr;
    (void)snprintf(probe.tag, sizeof(probe.tag), "%s", tag);

    for (int spin = 0; spin < LISTEN_REUSE_POLL_SPIN && std::chrono::steady_clock::now() < ctx.deadline; spin++) {
        probe.fdHandle = NULL;
        probe.status = 0;
        unsigned int num = 0;
        int r = RaGetSockets(isServer ? 0 : 1, &probe, 1, &num);
        if (r != 0) { /* RaGetSockets errors are positive 6-digit codes; abort with the real code */
            TEST_LOG_ERROR(ctx.phyid, "RaGetSockets failed ret=%d tag=%s", r, tag);
            return r;
        }
        if (num > 0 && probe.status == 1 && probe.fdHandle != NULL) {
            break;
        }
        usleep(5000);
    }
    if (probe.fdHandle == NULL) {
        TEST_LOG_ERROR(ctx.phyid, "link establish timeout on port %u tag=%s", ctx.port, tag);
        return -1;
    }
    TEST_LOG_INFO(ctx.phyid, "link established on port %u tag=%s fd=%p", ctx.port, tag, probe.fdHandle);

    int ret = RaSocketSendAll(probe.fdHandle, myMsg, strlen(myMsg) + 1);
    if (ret != 0) {
        TEST_LOG_ERROR(ctx.phyid, "send failed ret=%d tag=%s", ret, tag);
        return ret;
    }
    char recvBuf[64] = {0};
    ret = RaSocketRecvAll(probe.fdHandle, recvBuf, strlen(peerMsg) + 1);
    if (ret != 0 || memcmp(recvBuf, peerMsg, strlen(peerMsg) + 1) != 0) {
        TEST_LOG_ERROR(ctx.phyid, "recv FAIL ret=%d exp=%s act=%s tag=%s", ret, peerMsg, recvBuf, tag);
        return -1;
    }
    return 0;
}

/* One full link round with both sides in lockstep: barrier, then the client batch-connects
 * on tag, then both sides bounded-poll the link and exchange the round's message. */
static int ListenReuseRunLinkRound(const ListenReuseCtx& ctx, const char* tag, int round)
{
    int ret = SocketBarrier(ctx.env.sock.fdHandle);
    if (ret != 0) {
        TEST_LOG_ERROR(ctx.phyid, "round-%d barrier failed ret=%d", round, ret);
        return ret;
    }
    ret = ListenReuseClientConnect(ctx, tag);
    if (ret != 0) {
        return ret;
    }
    return ListenReuseLinkRound(ctx, tag, round);
}

/* Server-side teardown: user B stops (last reference — the listen node must be torn down),
 * a plain system connect to the business port must then be refused, and a re-listen on the
 * freed port must succeed. The release check is independent evidence: a successful connect
 * would mean the node leaked (a reference dropped without teardown), which a plain
 * re-listen cannot tell from reuse of a leaked node. */
static int ListenReuseVerifyFullRelease(const ListenReuseCtx& ctx)
{
    struct SocketListenInfoT listenInfo = {};
    listenInfo.socketHandle = ctx.env.socketHandle;
    listenInfo.port = ctx.port;
    int ret = RaSocketListenStop(&listenInfo, 1);
    if (ret != 0) {
        TEST_LOG_ERROR(ctx.phyid, "userB RaSocketListenStop failed ret=%d", ret);
        return ret;
    }

    struct in_addr selfIp = HccpVmToLoopback(in_addr{htonl(HccpVmNicIp((uint32_t)ctx.phyid, ctx.iface))});
    char selfIpText[INET_ADDRSTRLEN] = {0};
    (void)inet_ntop(AF_INET, &selfIp, selfIpText, sizeof(selfIpText));
    int probeFd = socket(AF_INET, SOCK_STREAM, 0);
    if (probeFd < 0) {
        /* Fail before connect() would replace the real cause (EMFILE/ENFILE) with EBADF. */
        TEST_LOG_ERROR(ctx.phyid, "probe socket failed errno=%d", errno);
        return -1;
    }
    struct sockaddr_in probeAddr;
    memset(&probeAddr, 0, sizeof(probeAddr));
    probeAddr.sin_family = AF_INET;
    probeAddr.sin_port = htons(ctx.port);
    probeAddr.sin_addr = selfIp;
    int cr = connect(probeFd, (struct sockaddr*)&probeAddr, sizeof(probeAddr));
    int crErr = errno;
    close(probeFd);
    if (cr == 0 || crErr != ECONNREFUSED) {
        TEST_LOG_ERROR(
            ctx.phyid, "release check FAIL: connect ret=%d errno=%d, expected ECONNREFUSED on %s:%u", cr, crErr,
            selfIpText[0] != '\0' ? selfIpText : "?", ctx.port);
        return -1;
    }
    TEST_LOG_INFO(
        ctx.phyid, "release check OK: %s:%u refused after last stop", selfIpText[0] != '\0' ? selfIpText : "?",
        ctx.port);

    listenInfo.err = 0;
    ret = RaSocketListenStart(&listenInfo, 1);
    if (ret != 0) {
        TEST_LOG_ERROR(ctx.phyid, "re-listen after full stop failed ret=%d", ret);
        return ret;
    }
    return 0;
}

static int
SocketListenReuseBody(int phyid, CaseEnv& env, unsigned short port, uint32_t iface, const char* tagA, const char* tagB)
{
    const int peerPhyId = (phyid == 1) ? 5 : 1;
    /* Whole-case session: peer addressing and the shared poll deadline (budget note above)
     * are fixed here; every helper below reads them through ctx. */
    ListenReuseCtx ctx = {phyid, iface,
                          env,   HccpVmToLoopback(in_addr{htonl(HccpVmNicIp((uint32_t)peerPhyId, iface))}),
                          port,  std::chrono::steady_clock::now() + std::chrono::milliseconds(LISTEN_REUSE_BUDGET_MS)};
    int ret = 0;

    if (phyid == 1) { /* server: two users share one listen node, then user A stops */
        ret = ListenReuseSetupSharedListen(ctx, tagA, tagB);
        if (ret != 0) {
            return ret;
        }
    }

    /* round 1: link must establish on the port user B still listens on */
    ret = ListenReuseRunLinkRound(ctx, tagA, 1);
    if (ret != 0) {
        return ret;
    }

    if (phyid == 1) { /* server: last reference gone — verify release, then re-listen */
        ret = ListenReuseVerifyFullRelease(ctx);
        if (ret != 0) {
            return ret;
        }
    }

    /* round 2: fresh tag must link on the re-created listen node */
    ret = ListenReuseRunLinkRound(ctx, tagB, 2);
    if (ret != 0) {
        return ret;
    }

    TEST_LOG_INFO(phyid, "listen-reuse lifecycle PASS on port %u", port);
    return 0;
}

/* Two-node listen-reuse guard over the OFFLINE/HDC run path (listen handled inside the
 * device process; capability table currently selects the V2 listen implementation):
 * user A and user B listen on the same socketHandle+port, user A stops, and the client
 * must still establish the link — the exact flow that timed out before MR !3681. */
TEST_F(HccpSystemTest, d2dSocketListenReuse)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        CaseEnv env;
        int ret = CaseOfflineInit(env, phyid, HCCP_VM_IFACE_SOCKET, 15666, "LR RDV 1");
        if (ret != 0) {
            return ret;
        }
        ret = SocketListenReuseBody(phyid, env, HccpStCasePort(15690), HCCP_VM_IFACE_SOCKET, "LR-TAG-A", "LR-TAG-B");
        CaseCleanup(env);
        return ret;
    });
}

/* Same listen-reuse flow over the PEER_ONLINE host-direct run path (NIC on the host, no
 * device process, RaPeerSocketListenStart calls RsSocketListenStartV2 in-process):
 * forms the HDC/peer run-path behavior pair with d2dSocketListenReuse and guards the
 * shared listen-node refcount lifecycle on both run paths. */
TEST_F(HccpSystemTest, h2hSocketListenReuse)
{
    RunTests({1, 5}, [this](int phyid) -> int {
        CaseEnv env;
        int ret = CasePeerInit(env, phyid, HCCP_VM_IFACE_ROCE, 15673, "LR RDV 2");
        if (ret != 0) {
            return ret;
        }
        ret = SocketListenReuseBody(phyid, env, HccpStCasePort(15691), HCCP_VM_IFACE_ROCE, "LR-PEER-A", "LR-PEER-B");
        CaseCleanup(env);
        return ret;
    });
}
