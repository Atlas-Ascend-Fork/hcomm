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
#include <unordered_map>

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
