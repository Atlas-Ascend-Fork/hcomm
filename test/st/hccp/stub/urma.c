/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * urma.c - URMA send/recv simulation stub (liburma.so)
 *
 * Simulates UB send/recv via shared-memory copies, mirroring ibverbs_stub.c
 * (the RDMA verbs simulation stub). URMA QP/CQ/MR equivalent abstractions:
 *   jfs (jetty for send, like a send QP) / jfr (jetty for recv, like a recv QP)
 *   jfc (jetty for completion, like a CQ) / seg (memory segment, like an MR)
 *   cr  (completion record, like a wc)
 *
 * Data plane: one PTHREAD_PROCESS_SHARED robust shared-memory segment (memfd,
 * precreated by the infra constructor and inherited through fork/exec) holding
 *   - a jetty-id allocator (cross-process unique, like rdma_alloc_unique_qp_num)
 *   - a ring-queue slot per jetty (like rdma_pkt_enqueue/dequeue)
 *   - a cross-process rendezvous area (jfr-id / rjfr / token exchange table)
 *   - a pthread_mutex_t (PROCESS_SHARED) for protection
 *
 * Completion semantics: a completed send produces a send-cr (on the sender's jfc);
 * once data lands in the peer's recv buf, a recv-cr is produced (on the receiver's
 * jfc, byte_len = the actually copied length).
 *
 * Design contract: ub_st/design_docs/architecture_design_v1.0.md section 3/4
 * Stub design: urma_stub/design_docs/urma_stub_design_v1.0.md
 */

#define _GNU_SOURCE

#include <urma/urma_api.h>
#include "udma_u_ctl.h"

#include <arpa/inet.h>
#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/eventfd.h>
#include <sys/ipc.h>
#include <sys/prctl.h>
#include <sys/shm.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>

#include "hccp_shm_infra.h"
#include "stub_log.h"

#define URMA_DEV_NUM_MAX 1
#define URMA_SHM_KEY 0x1234ABCF
#define URMA_SHM_MAGIC 0x55524D41u
#define URMA_RING_SLOTS 64

#define URMA_MAX_MSG (64 * 1024)
#define URMA_MAX_CQE 1024
/* must cover the jfr depth of the migrated cases: the pingmesh ping jetty posts
 * maxRecvWr (128) pre-posted recv buffers (rs_ping_urma.c
 * RsPingCommonInitJettyPostRecvAll), 64 rejected the 65th post with EAGAIN */
#define URMA_MAX_RECV_WR 256
#define URMA_MAX_RDV 64

#define URMA_MAX_JFCS 256
#define URMA_MAX_JFSS 256
#define URMA_MAX_JFRS 256
#define URMA_MAX_SEGS 256
#define URMA_MAX_TJFRS 256
#define URMA_MAX_JETTYS 256
#define URMA_JETTY_RECV_WR_NUM 16

typedef struct {
    int in_use;
    uint64_t buf;
    uint32_t len;
    uint64_t user_ctx;
} urma_recv_wr_priv_t;

typedef struct {
    urma_jfc_t* jfc;
    int in_use;
    urma_cr_t crq[URMA_MAX_CQE];
    int head;
    int tail;
    int count;
    urma_jfce_t* jfce; /* event channel bound at create (cfg.jfce); NULL = no async signal */
    int ext_mode;      /* 1 = created via the ext API (urma_alloc_jfc): STARS_POLL /
                        * CCU_POLL / USER_CTL_NORMAL cq. On real hardware these modes
                        * deliver completions by polling the cqe/ci region, never via
                        * jfc CR events, so the stub must not queue CRs or signal the
                        * jfce for them (the rs CR-event handler assumes a
                        * JFC_MODE_NORMAL jetty whose qpShareInfoAddr is allocated). */
} urma_jfc_priv_t;

typedef struct {
    urma_jfs_t* jfs;
    int in_use;
    urma_jetty_state_t state;
    uint32_t jetty_id;
    urma_jfc_t* jfc;
} urma_jfs_priv_t;

typedef struct {
    urma_jfr_t* jfr;
    int in_use;
    urma_jfr_state_t state;
    uint32_t jetty_id;
    urma_jfc_t* jfc;
    urma_recv_wr_priv_t recv_q[URMA_MAX_RECV_WR];
    int recv_head;
    int recv_tail;
    int recv_count;
} urma_jfr_priv_t;

typedef struct {
    urma_target_seg_t* tseg;
    int in_use;
    uint64_t addr;
    uint64_t len;
    uint32_t seg_id;
    uint32_t lkey;
    urma_token_t token;
    int owns_token;
} urma_seg_priv_t;

typedef struct {
    urma_target_jetty_t* tjfr;
    int in_use;
    uint32_t remote_jetty_id;
} urma_tjfr_priv_t;

typedef struct {
    urma_jetty_t* jetty;
    int in_use;
    urma_jfs_t* jfs;
    urma_jfr_t* jfr;
    urma_jfc_t* send_jfc;
    uint32_t jetty_id;
    uint32_t pi_counter;
} urma_jetty_priv_t;

static urma_jfc_priv_t g_jfc_table[URMA_MAX_JFCS];
static urma_jfs_priv_t g_jfs_table[URMA_MAX_JFSS];
static urma_jfr_priv_t g_jfr_table[URMA_MAX_JFRS];
static urma_seg_priv_t g_seg_table[URMA_MAX_SEGS];
static urma_tjfr_priv_t g_tjfr_table[URMA_MAX_TJFRS];
static urma_jetty_priv_t g_jetty_table[URMA_MAX_JETTYS];

/* forward: urma_post_jetty_send/recv_wr (below) resolve the jetty through jetty_find */
static urma_jetty_priv_t* jetty_find(urma_jetty_t* jetty);

/* Serializes the software data plane (jfc CQE rings, jfr recv queues, jfce eventfd
 * signals and the ring drain). Real hardware pushes completions asynchronously; here a
 * pump thread (g_pump_thread, see urma_create_jfr) converts ring packets into recv CQEs
 * and signals the bound jfce eventfd so the rs epoll loop (rs_ping_urma.c
 * RsPingCommonInitLocalJetty registers jfce->fd) wakes and handles them, mirroring
 * async completion delivery. Lock order: g_dp_mutex -> shm ring lock only; the ring
 * helpers acquire/release their own lock internally. */
static pthread_mutex_t g_dp_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_t g_pump_thread;
static int g_pump_started = 0;

static uint32_t g_uasid = 0;
static int g_inited = 0;

typedef struct {
    int in_use;
    uint32_t jetty_id;
    urma_transport_mode_t trans_mode;
    urma_token_t token;
    pid_t owner_pid;
} urma_rdv_entry_t;

typedef struct {
    uint32_t magic;
    pid_t creator_pid;
    pthread_mutex_t lock;
    uint64_t next_seq;
    uint32_t next_jetty_id;
    urma_rdv_entry_t rdv[URMA_MAX_RDV];
    hccp_pkt_slot_t slots[URMA_RING_SLOTS];
} urma_shm_t;

static urma_shm_t* g_urma_shm = NULL;

void* hccp_test_alloc_shm_buf(void) { return hccp_shm_alloc(HCCP_BUF_SIZE); }

/* process_vm_writev/readv need ptrace permission over the peer process; every stub
 * process allows it so sibling ST processes can act as pvm targets (same approach as
 * the ibverbs host stub). Runs at library load in every process that opens a context. */
static void urma_stub_set_ptracer_any(void) { (void)prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0); }

__attribute__((constructor)) static void urma_shm_ctor(void)
{
    urma_stub_set_ptracer_any();
    g_urma_shm = (urma_shm_t*)hccp_shm_open(URMA_SHM_KEY, sizeof(urma_shm_t), URMA_SHM_MAGIC);
    hccp_buf_pool_init();
    /* Map the consolidated devmem segment at its fixed DVA window: the engine
     * process must be able to act as the local endpoint of cross-node devmem transfers
     * (rtMalloc now allocates from this pool in the client process, and both endpoints
     * of a WRITE/READ need the window mapped when the copy is routed via pvm). */
    const char* phyIdEnv = getenv("HCCP_VM_PHYID");
    if (phyIdEnv != NULL) {
        hccp_devmem_pool_attach(atoi(phyIdEnv));
    }
}

static void urma_rdv_publish(uint32_t jetty_id, urma_transport_mode_t trans_mode, urma_token_t token)
{
    if (g_urma_shm == NULL)
        return;
    if (hccp_shm_lock_acquire(&g_urma_shm->lock) != 0)
        return; /* table not recoverable */
    int free_idx = -1;
    for (int i = 0; i < URMA_MAX_RDV; i++) {
        if (g_urma_shm->rdv[i].in_use && g_urma_shm->rdv[i].jetty_id == jetty_id) {
            g_urma_shm->rdv[i].trans_mode = trans_mode;
            g_urma_shm->rdv[i].token = token;
            g_urma_shm->rdv[i].owner_pid = getpid();
            (void)hccp_shm_lock_release(&g_urma_shm->lock);
            return;
        }
        if (!g_urma_shm->rdv[i].in_use && free_idx < 0) {
            free_idx = i;
        } else if (
            g_urma_shm->rdv[i].in_use && free_idx < 0 && kill(g_urma_shm->rdv[i].owner_pid, 0) != 0 && errno == ESRCH) {
            /* watchdog-killed processes never run the delete path; reclaim their
             * entries so one failed case cannot shrink the table for the rest
             * of the suite (a recycled pid keeps the entry, which is safe:
             * it just stays unclaimed) */
            free_idx = i;
        }
    }
    if (free_idx >= 0) {
        g_urma_shm->rdv[free_idx].in_use = 1;
        g_urma_shm->rdv[free_idx].jetty_id = jetty_id;
        g_urma_shm->rdv[free_idx].trans_mode = trans_mode;
        g_urma_shm->rdv[free_idx].token = token;
        g_urma_shm->rdv[free_idx].owner_pid = getpid();
    }
    (void)hccp_shm_lock_release(&g_urma_shm->lock);
}

static void urma_rdv_unpublish(uint32_t jetty_id)
{
    if (g_urma_shm == NULL)
        return;
    if (hccp_shm_lock_acquire(&g_urma_shm->lock) != 0)
        return; /* table not recoverable */
    for (int i = 0; i < URMA_MAX_RDV; i++) {
        if (g_urma_shm->rdv[i].in_use && g_urma_shm->rdv[i].jetty_id == jetty_id
            && g_urma_shm->rdv[i].owner_pid == getpid()) {
            memset(&g_urma_shm->rdv[i], 0, sizeof(g_urma_shm->rdv[i]));
            break;
        }
    }
    (void)hccp_shm_lock_release(&g_urma_shm->lock);
}

/* Owner pid of a jetty id, from the shared rendezvous table. The peer publishes its
 * jetty when creating it, so by the time a WRITE/READ targets that jetty the entry
 * exists and identifies the process that registered the remote devmem segment. */
static pid_t urma_rdv_owner_pid(uint32_t jetty_id)
{
    if (g_urma_shm == NULL || jetty_id == 0) {
        return 0;
    }
    pid_t owner = 0;
    if (hccp_shm_lock_acquire(&g_urma_shm->lock) != 0) {
        return 0; /* table not recoverable: no routing decision possible */
    }
    for (int i = 0; i < URMA_MAX_RDV; i++) {
        if (g_urma_shm->rdv[i].in_use && g_urma_shm->rdv[i].jetty_id == jetty_id) {
            owner = g_urma_shm->rdv[i].owner_pid;
            break;
        }
    }
    (void)hccp_shm_lock_release(&g_urma_shm->lock);
    return owner;
}

/* Numeric window of the consolidated devmem segment (rtMalloc -> halMemAlloc backing
 * store). The whole segment is mapped at the fixed DVA base in every process, so a
 * peer DVA cannot be told apart from a local one by value; routing is decided by the
 * owning pid. The layout predicate is the single slot-validity authority: it rejects
 * the master/zone headers by construction, which a plain range check never could. */
static int urma_va_in_devmem_window(uint64_t va) { return hccp_devmem_dva_decode(va, NULL, NULL); }

/* Cross-process transfer between a local VA and the same numeric VA in the peer process
 * (the peer maps its own devmem pool at that address), mirroring ibv_pvm_xfer. */
static ssize_t urma_pvm_xfer(int is_write, pid_t peer_pid, void* local_addr, void* remote_addr, size_t len)
{
    size_t done = 0;
    while (done < len) {
        struct iovec local = {
            .iov_base = (char*)local_addr + done,
            .iov_len = len - done,
        };
        struct iovec remote = {
            .iov_base = (char*)remote_addr + done,
            .iov_len = len - done,
        };
        ssize_t n = is_write ? process_vm_writev(peer_pid, &local, 1, &remote, 1, 0) :
                               process_vm_readv(peer_pid, &local, 1, &remote, 1, 0);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            URMA_LOG_ERROR(
                "[URMA-PVM-ERR] op=%s peer=%d local=%p remote=%p off=%zu len=%zu errno=%d(%s)",
                is_write ? "write" : "read", (int)peer_pid, local_addr, remote_addr, done, len, errno, strerror(errno));
            return -1;
        }
        if (n == 0) {
            return -1;
        }
        done += (size_t)n;
    }
    return (ssize_t)done;
}

static urma_jfc_priv_t* jfc_find(urma_jfc_t* jfc)
{
    for (int i = 0; i < URMA_MAX_JFCS; i++) {
        if (g_jfc_table[i].in_use && g_jfc_table[i].jfc == jfc) {
            return &g_jfc_table[i];
        }
    }
    return NULL;
}

static urma_jfc_priv_t* jfc_register(urma_jfc_t* jfc)
{
    if (jfc == NULL)
        return NULL;
    urma_jfc_priv_t* p = jfc_find(jfc);
    if (p != NULL)
        return p;
    for (int i = 0; i < URMA_MAX_JFCS; i++) {
        if (!g_jfc_table[i].in_use) {
            memset(&g_jfc_table[i], 0, sizeof(g_jfc_table[i]));
            g_jfc_table[i].in_use = 1;
            g_jfc_table[i].jfc = jfc;
            g_jfc_table[i].jfce = jfc->jfc_cfg.jfce;
            return &g_jfc_table[i];
        }
    }
    return NULL;
}

static void jfc_unregister(urma_jfc_t* jfc)
{
    urma_jfc_priv_t* p = jfc_find(jfc);
    if (p != NULL)
        memset(p, 0, sizeof(*p));
}

/* jfce eventfd level signal: wakes the epoll loop of the process owning this jfc.
 * Called with g_dp_mutex held (or from single-threaded init paths). */
static void jfc_signal_jfce(urma_jfc_priv_t* p)
{
    if (p == NULL || p->jfce == NULL || p->jfce->fd < 0)
        return;
    uint64_t one = 1;
    (void)write(p->jfce->fd, &one, sizeof(one));
}

/* clears stale eventfd signal after CQEs have been drained by poll; avoids spurious
 * epoll wakes that the rs would log as ping_poll_rcq errors. */
static void jfc_drain_jfce(urma_jfc_priv_t* p)
{
    if (p == NULL || p->jfce == NULL || p->jfce->fd < 0)
        return;
    uint64_t val;
    while (read(p->jfce->fd, &val, sizeof(val)) == (ssize_t)sizeof(val)) {
    }
}

/* caller must hold g_dp_mutex */
static void jfc_push_cr_locked(urma_jfc_priv_t* p, const urma_cr_t* cr)
{
    if (p == NULL || cr == NULL)
        return;
    if (p->ext_mode)
        return; /* ext-mode cqs consume completions by polling the cqe/ci
                 * region (hardware semantics); queuing CRs and signaling
                 * the jfce would drive the rs CR-event handler down a path
                 * that assumes a JFC_MODE_NORMAL jetty (qpShareInfoAddr) */
    if (p->count >= URMA_MAX_CQE)
        return;
    p->crq[p->tail] = *cr;
    p->tail = (p->tail + 1) % URMA_MAX_CQE;
    p->count++;
    jfc_signal_jfce(p);
}

static void jfc_push_cr(urma_jfc_t* jfc, const urma_cr_t* cr)
{
    pthread_mutex_lock(&g_dp_mutex);
    jfc_push_cr_locked(jfc_find(jfc), cr);
    pthread_mutex_unlock(&g_dp_mutex);
}

static urma_jfs_priv_t* jfs_find(urma_jfs_t* jfs)
{
    for (int i = 0; i < URMA_MAX_JFSS; i++) {
        if (g_jfs_table[i].in_use && g_jfs_table[i].jfs == jfs) {
            return &g_jfs_table[i];
        }
    }
    return NULL;
}

static urma_jfr_priv_t* jfr_find(urma_jfr_t* jfr)
{
    for (int i = 0; i < URMA_MAX_JFRS; i++) {
        if (g_jfr_table[i].in_use && g_jfr_table[i].jfr == jfr) {
            return &g_jfr_table[i];
        }
    }
    return NULL;
}

static urma_seg_priv_t* seg_find(urma_target_seg_t* tseg)
{
    for (int i = 0; i < URMA_MAX_SEGS; i++) {
        if (g_seg_table[i].in_use && g_seg_table[i].tseg == tseg) {
            return &g_seg_table[i];
        }
    }
    return NULL;
}

static urma_seg_priv_t* seg_find_by_lkey(uint32_t lkey)
{
    for (int i = 0; i < URMA_MAX_SEGS; i++) {
        if (g_seg_table[i].in_use && g_seg_table[i].lkey == lkey) {
            return &g_seg_table[i];
        }
    }
    return NULL;
}

static urma_cr_opcode_t hccp_seg_rw_to_cr_opcode(hccp_seg_rw_kind_t k)
{
    switch (k) {
        case HCCP_SEG_RW_WRITE:
            return URMA_CR_OPC_WRITE_WITH_IMM;
        case HCCP_SEG_RW_READ:
            return URMA_CR_OPC_SEND_WITH_INV;
        case HCCP_SEG_RW_SEND:
        default:
            return URMA_CR_OPC_SEND;
    }
}

__attribute__((unused)) static hccp_seg_rw_kind_t hccp_seg_rw_from_cr_opcode(urma_cr_opcode_t op)
{
    switch (op) {
        case URMA_CR_OPC_WRITE_WITH_IMM:
            return HCCP_SEG_RW_WRITE;
        case URMA_CR_OPC_SEND_WITH_INV:
            return HCCP_SEG_RW_READ;
        case URMA_CR_OPC_SEND:
        default:
            return HCCP_SEG_RW_SEND;
    }
}

uint32_t hccp_seg_register(uint64_t va, size_t len)
{
    if (va == 0)
        return 0;

    urma_seg_priv_t* p = NULL;
    for (int i = 0; i < URMA_MAX_SEGS; i++) {
        if (!g_seg_table[i].in_use) {
            p = &g_seg_table[i];
            break;
        }
    }
    if (p == NULL) {
        return 0;
    }
    memset(p, 0, sizeof(*p));
    p->in_use = 1;
    p->addr = va;
    p->len = (len > 0 && len <= HCCP_BUF_SIZE) ? (uint64_t)len : (uint64_t)HCCP_BUF_SIZE;
    p->seg_id = (g_urma_shm != NULL) ? hccp_alloc_unique_id(&g_urma_shm->next_jetty_id) : 0;
    if (p->seg_id == 0)
        p->seg_id = 1;
    p->lkey = p->seg_id;
    return p->lkey;
}

int hccp_seg_write(uint32_t lkey, uint64_t dst_va, const void* src, uint32_t len)
{
    if (dst_va == 0 || len == 0)
        return 0;
    urma_seg_priv_t* p = seg_find_by_lkey(lkey);
    uint64_t src_va = (src != NULL) ? (uint64_t)(uintptr_t)src : ((p != NULL) ? p->addr : 0);
    if (src_va == 0)
        return -1;
    if (p != NULL && len > p->len)
        len = (uint32_t)p->len;
    memcpy((void*)(uintptr_t)dst_va, (const void*)(uintptr_t)src_va, len);
    return 0;
}

int hccp_seg_read(uint32_t lkey, void* dst, uint64_t src_va, uint32_t len)
{
    if (src_va == 0 || len == 0)
        return 0;
    urma_seg_priv_t* p = seg_find_by_lkey(lkey);
    uint64_t dst_va = (dst != NULL) ? (uint64_t)(uintptr_t)dst : ((p != NULL) ? p->addr : 0);
    if (dst_va == 0)
        return -1;
    if (p != NULL && len > p->len)
        len = (uint32_t)p->len;
    memcpy((void*)(uintptr_t)dst_va, (const void*)(uintptr_t)src_va, len);
    return 0;
}

/* WRITE data plane with devmem routing: a dst VA inside the per-phyId devmem window
 * belongs to the peer process when the WR targets the peer's jetty, so the copy must
 * be routed cross-process (the local mapping of that numeric address is our own pool).
 * Everything else keeps the legacy local-memcpy semantics. */
static int urma_seg_write_routed(uint32_t lkey, uint64_t dst_va, const void* src, uint32_t len, pid_t peer_pid)
{
    if (peer_pid != 0 && peer_pid != getpid() && urma_va_in_devmem_window(dst_va)) {
        urma_seg_priv_t* p = seg_find_by_lkey(lkey);
        uint64_t src_va = (src != NULL) ? (uint64_t)(uintptr_t)src : ((p != NULL) ? p->addr : 0);
        if (src_va == 0)
            return -1;
        if (p != NULL && len > p->len)
            len = (uint32_t)p->len;
        if (urma_pvm_xfer(1, peer_pid, (void*)(uintptr_t)src_va, (void*)(uintptr_t)dst_va, len) < 0) {
            return -1;
        }
        return 0;
    }
    return hccp_seg_write(lkey, dst_va, src, len);
}

/* READ data plane with devmem routing, mirroring urma_seg_write_routed. */
static int urma_seg_read_routed(uint32_t lkey, void* dst, uint64_t src_va, uint32_t len, pid_t peer_pid)
{
    if (peer_pid != 0 && peer_pid != getpid() && urma_va_in_devmem_window(src_va)) {
        urma_seg_priv_t* p = seg_find_by_lkey(lkey);
        uint64_t dst_va = (dst != NULL) ? (uint64_t)(uintptr_t)dst : ((p != NULL) ? p->addr : 0);
        if (dst_va == 0)
            return -1;
        if (p != NULL && len > p->len)
            len = (uint32_t)p->len;
        if (urma_pvm_xfer(0, peer_pid, (void*)(uintptr_t)dst_va, (void*)(uintptr_t)src_va, len) < 0) {
            return -1;
        }
        return 0;
    }
    return hccp_seg_read(lkey, dst, src_va, len);
}

static urma_tjfr_priv_t* tjfr_find(urma_target_jetty_t* tjfr)
{
    for (int i = 0; i < URMA_MAX_TJFRS; i++) {
        if (g_tjfr_table[i].in_use && g_tjfr_table[i].tjfr == tjfr) {
            return &g_tjfr_table[i];
        }
    }
    return NULL;
}

/* caller must hold g_dp_mutex */
static void jfr_drain_recv_locked(urma_jfr_priv_t* p)
{
    if (p == NULL)
        return;

    hccp_pkt_slot_t pkt;
    while (p->recv_count > 0 && g_urma_shm != NULL
           && hccp_pkt_dequeue(&g_urma_shm->lock, g_urma_shm->slots, URMA_RING_SLOTS, p->jetty_id, &pkt)) {
        urma_recv_wr_priv_t* r = &p->recv_q[p->recv_head];
        uint32_t copy = pkt.byte_len;
        if (copy > r->len)
            copy = r->len;
        if (copy > 0 && r->buf != 0) {
            memcpy((void*)(uintptr_t)r->buf, pkt.data, copy);
        }

        urma_cr_t cr;
        memset(&cr, 0, sizeof(cr));
        cr.status = URMA_CR_SUCCESS;
        cr.user_ctx = r->user_ctx;
        cr.opcode = URMA_CR_OPC_SEND;
        cr.flag.bs.s_r = 1;
        cr.completion_len = copy;
        cr.local_id = p->jetty_id;
        cr.remote_id.id = pkt.src_id;
        jfc_push_cr_locked(jfc_find(p->jfc), &cr);

        p->recv_head = (p->recv_head + 1) % URMA_MAX_RECV_WR;
        p->recv_count--;
    }
}

/* pump thread body: converts ring packets destined to this process's jfrs into recv
 * CQEs and jfce signals, standing in for asynchronous hardware completion delivery.
 * Started once per process on first jfr creation (service processes only; host children
 * never create jfrs, so no pump runs there). */
static void* urma_completion_pump(void* arg)
{
    (void)arg;
    prctl(PR_SET_NAME, (uintptr_t) "urma_pump", 0, 0, 0);
    while (g_pump_started) {
        pthread_mutex_lock(&g_dp_mutex);
        for (int i = 0; i < URMA_MAX_JFRS; i++) {
            if (g_jfr_table[i].in_use) {
                jfr_drain_recv_locked(&g_jfr_table[i]);
            }
        }
        pthread_mutex_unlock(&g_dp_mutex);
        usleep(2000); /* 2ms cadence, well below the 10ms ping interval */
    }
    return NULL;
}

static void jfr_drain_recv(urma_jfr_priv_t* p)
{
    if (p == NULL)
        return;
    pthread_mutex_lock(&g_dp_mutex);
    jfr_drain_recv_locked(p);
    pthread_mutex_unlock(&g_dp_mutex);
}

urma_status_t urma_init(urma_init_attr_t* conf)
{
    if (g_urma_shm == NULL) {
        g_urma_shm = (urma_shm_t*)hccp_shm_open(URMA_SHM_KEY, sizeof(urma_shm_t), URMA_SHM_MAGIC);
    }
    if (conf != NULL && conf->uasid != 0) {
        g_uasid = conf->uasid;
    } else {
        g_uasid = (uint32_t)getpid();
        if (g_uasid == 0)
            g_uasid = 1;
    }
    g_inited = 1;
    return URMA_SUCCESS;
}

urma_status_t urma_uninit(void)
{
    g_inited = 0;
    return URMA_SUCCESS;
}

static urma_device_t g_urma_devs[URMA_DEV_NUM_MAX];
static int g_urma_dev_inited = 0;

static void urma_dev_init_once(void)
{
    if (g_urma_dev_inited)
        return;
    for (int i = 0; i < URMA_DEV_NUM_MAX; i++) {
        snprintf(g_urma_devs[i].name, sizeof(g_urma_devs[i].name), "ub%u", i);
        snprintf(g_urma_devs[i].path, sizeof(g_urma_devs[i].path), "/sys/class/ub/mock%u", i);
        g_urma_devs[i].type = URMA_TRANSPORT_UB;
        g_urma_devs[i].ops = NULL;
        g_urma_devs[i].sysfs_dev = NULL;
    }
    g_urma_dev_inited = 1;
}

urma_device_t** urma_get_device_list(int* num_devices)
{
    urma_dev_init_once();
    urma_device_t** list = (urma_device_t**)calloc(URMA_DEV_NUM_MAX + 1, sizeof(urma_device_t*));
    if (list == NULL) {
        if (num_devices)
            *num_devices = 0;
        return NULL;
    }
    for (int i = 0; i < URMA_DEV_NUM_MAX; i++) {
        list[i] = &g_urma_devs[i];
    }
    if (num_devices)
        *num_devices = URMA_DEV_NUM_MAX;
    return list;
}

void urma_free_device_list(urma_device_t** device_list)
{
    if (device_list == NULL)
        return;
    free(device_list);
}

/* Device eid table. Production exposes one eid per device nic (in the ST vm net the
 * nics are the loopback-mapped addresses 127.<phyId>.<iface>). The per-device process
 * inherits HCCP_VM_PHYID from the forking host child, so the stub builds the same
 * table: three eids (socket/h2d/roce iface), eid_index = iface id. The migrated HLT
 * control plane cases resolve CtxInitAttr.ub.eid by matching the local nic ip against
 * this list (test_ra_get_dev_eid_info_list semantics), so the table must carry the
 * actual nic addresses instead of a placeholder. */
#define URMA_DEV_EID_NUM 3
urma_eid_info_t* urma_get_eid_list(urma_device_t* dev, uint32_t* cnt)
{
    (void)dev;
    urma_eid_info_t* eids = (urma_eid_info_t*)calloc(URMA_DEV_EID_NUM, sizeof(urma_eid_info_t));
    if (eids == NULL) {
        if (cnt)
            *cnt = 0;
        return NULL;
    }
    uint32_t phyId = 0;
    const char* phyIdEnv = getenv("HCCP_VM_PHYID");
    if (phyIdEnv != NULL) {
        phyId = (uint32_t)atoi(phyIdEnv);
    }
    for (int iface = 0; iface < URMA_DEV_EID_NUM; iface++) {
        /* mirrors HccpVmToLoopback(HccpVmNicIp(phyId, iface)) of hccp_vm_net.c;
         * urma_u32_to_eid takes the host-order value and stores it network-order */
        uint32_t byte2 = (phyId == 0u) ? 100u : phyId;
        uint32_t addr = (127u << 24) | (byte2 << 8) | (uint32_t)iface;
        eids[iface].eid_index = (uint32_t)iface;
        urma_u32_to_eid(addr, &eids[iface].eid);
    }
    if (cnt)
        *cnt = URMA_DEV_EID_NUM;
    return eids;
}

void urma_free_eid_list(urma_eid_info_t* eid_list) { free(eid_list); }

urma_device_t* urma_get_device_by_name(char* dev_name)
{
    urma_dev_init_once();
    if (dev_name == NULL)
        return NULL;
    for (int i = 0; i < URMA_DEV_NUM_MAX; i++) {
        if (strcmp(g_urma_devs[i].name, dev_name) == 0) {
            return &g_urma_devs[i];
        }
    }
    return NULL;
}

urma_device_t* urma_get_device_by_eid(urma_eid_t eid, urma_transport_type_t type)
{
    (void)eid;
    (void)type;
    urma_dev_init_once();
    return &g_urma_devs[0];
}

urma_status_t urma_query_device(urma_device_t* dev, urma_device_attr_t* dev_attr)
{
    if (dev == NULL || dev_attr == NULL)
        return URMA_EINVAL;
    memset(dev_attr, 0, sizeof(*dev_attr));
    dev_attr->port_cnt = 1;
    dev_attr->dev_cap.max_jfc = 1024;
    dev_attr->dev_cap.max_jfs = 1024;
    dev_attr->dev_cap.max_jfr = 1024;
    dev_attr->dev_cap.max_jetty = 1024;
    dev_attr->dev_cap.max_jfc_depth = URMA_MAX_CQE;
    dev_attr->dev_cap.max_jfs_depth = 1024;
    dev_attr->dev_cap.max_jfr_depth = URMA_MAX_RECV_WR;
    dev_attr->dev_cap.max_jfs_sge = 8;
    dev_attr->dev_cap.max_jfr_sge = 8;
    dev_attr->dev_cap.max_msg_size = URMA_MAX_MSG;
    dev_attr->dev_cap.trans_mode = URMA_TM_RM | URMA_TM_RC | URMA_TM_UM;
    dev_attr->port_attr[0].state = URMA_PORT_ACTIVE;
    dev_attr->port_attr[0].max_mtu = URMA_MTU_4096;
    dev_attr->port_attr[0].active_mtu = URMA_MTU_4096;
    return URMA_SUCCESS;
}

urma_context_t* urma_create_context(urma_device_t* dev, uint32_t eid_index)
{
    if (dev == NULL)
        return NULL;
    urma_context_t* ctx = (urma_context_t*)calloc(1, sizeof(urma_context_t));
    if (ctx == NULL)
        return NULL;
    ctx->dev = dev;
    ctx->ops = NULL;
    ctx->dev_fd = -1;
    ctx->async_fd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    ctx->eid_index = eid_index;
    ctx->uasid = g_uasid;
    ctx->aggr_mode = URMA_AGGR_MODE_STANDALONE;
    return ctx;
}

urma_status_t urma_delete_context(urma_context_t* ctx)
{
    if (ctx == NULL)
        return URMA_EINVAL;
    if (ctx->async_fd >= 0) {
        (void)close(ctx->async_fd);
        ctx->async_fd = -1;
    }
    free(ctx);
    return URMA_SUCCESS;
}

urma_status_t urma_set_context_opt(urma_context_t* ctx, urma_opt_name_t opt_name, const void* opt_value, size_t opt_len)
{
    (void)opt_value;
    (void)opt_len;
    if (ctx == NULL)
        return URMA_EINVAL;
    if (opt_name == URMA_OPT_AGGR_MODE) {
    }
    return URMA_SUCCESS;
}

urma_jfc_t* urma_create_jfc(urma_context_t* ctx, urma_jfc_cfg_t* jfc_cfg)
{
    if (ctx == NULL || jfc_cfg == NULL)
        return NULL;
    urma_jfc_t* jfc = (urma_jfc_t*)calloc(1, sizeof(urma_jfc_t));
    if (jfc == NULL)
        return NULL;
    jfc->urma_ctx = ctx;
    jfc->jfc_cfg = *jfc_cfg;
    jfc->jfc_id.id = (g_urma_shm != NULL) ? hccp_alloc_unique_id(&g_urma_shm->next_jetty_id) : 0;
    if (jfc->jfc_id.id == 0)
        jfc->jfc_id.id = 1;
    jfc->jfc_id.uasid = ctx->uasid;
    if (jfc_register(jfc) == NULL) {
        free(jfc);
        return NULL;
    }
    return jfc;
}

urma_status_t urma_delete_jfc(urma_jfc_t* jfc)
{
    if (jfc == NULL)
        return URMA_EINVAL;
    pthread_mutex_lock(&g_dp_mutex);
    jfc_unregister(jfc);
    pthread_mutex_unlock(&g_dp_mutex);
    free(jfc);
    return URMA_SUCCESS;
}

urma_status_t urma_modify_jfc(urma_jfc_t* jfc, urma_jfc_attr_t* attr)
{
    (void)attr;
    if (jfc == NULL)
        return URMA_EINVAL;
    return URMA_SUCCESS;
}

urma_status_t urma_rearm_jfc(urma_jfc_t* jfc, bool solicited_only)
{
    (void)solicited_only;
    if (jfc == NULL)
        return URMA_EINVAL;
    return URMA_SUCCESS;
}

urma_jfs_t* urma_create_jfs(urma_context_t* ctx, urma_jfs_cfg_t* jfs_cfg)
{
    if (ctx == NULL || jfs_cfg == NULL)
        return NULL;
    urma_jfs_t* jfs = (urma_jfs_t*)calloc(1, sizeof(urma_jfs_t));
    if (jfs == NULL)
        return NULL;
    jfs->urma_ctx = ctx;
    jfs->jfs_cfg = *jfs_cfg;
    jfs->jfs_id.id = (g_urma_shm != NULL) ? hccp_alloc_unique_id(&g_urma_shm->next_jetty_id) : 0;
    if (jfs->jfs_id.id == 0)
        jfs->jfs_id.id = 1;
    jfs->jfs_id.uasid = ctx->uasid;
    jfs->jfs_id.eid = ctx->eid;

    urma_jfs_priv_t* p = NULL;
    for (int i = 0; i < URMA_MAX_JFSS; i++) {
        if (!g_jfs_table[i].in_use) {
            p = &g_jfs_table[i];
            break;
        }
    }
    if (p == NULL) {
        free(jfs);
        return NULL;
    }
    memset(p, 0, sizeof(*p));
    p->in_use = 1;
    p->jfs = jfs;
    p->state = URMA_JETTY_STATE_RESET;
    p->jetty_id = jfs->jfs_id.id;
    p->jfc = jfs_cfg->jfc;
    return jfs;
}

urma_status_t urma_delete_jfs(urma_jfs_t* jfs)
{
    if (jfs == NULL)
        return URMA_EINVAL;
    urma_jfs_priv_t* p = jfs_find(jfs);
    if (p != NULL)
        memset(p, 0, sizeof(*p));
    free(jfs);
    return URMA_SUCCESS;
}

urma_status_t urma_modify_jfs(urma_jfs_t* jfs, urma_jfs_attr_t* attr)
{
    if (jfs == NULL || attr == NULL)
        return URMA_EINVAL;
    urma_jfs_priv_t* p = jfs_find(jfs);
    if (p != NULL && (attr->mask & JFS_STATE)) {
        p->state = attr->state;
    }
    return URMA_SUCCESS;
}

urma_status_t urma_query_jfs(urma_jfs_t* jfs, urma_jfs_cfg_t* cfg, urma_jfs_attr_t* attr)
{
    if (jfs == NULL)
        return URMA_EINVAL;
    if (cfg)
        *cfg = jfs->jfs_cfg;
    if (attr) {
        memset(attr, 0, sizeof(*attr));
        urma_jfs_priv_t* p = jfs_find(jfs);
        attr->state = (p != NULL) ? p->state : URMA_JETTY_STATE_RESET;
    }
    return URMA_SUCCESS;
}

urma_jfr_t* urma_create_jfr(urma_context_t* ctx, urma_jfr_cfg_t* jfr_cfg)
{
    if (ctx == NULL || jfr_cfg == NULL)
        return NULL;
    urma_jfr_t* jfr = (urma_jfr_t*)calloc(1, sizeof(urma_jfr_t));
    if (jfr == NULL)
        return NULL;
    jfr->urma_ctx = ctx;
    jfr->jfr_cfg = *jfr_cfg;
    uint32_t jetty_id = (jfr_cfg->id != 0) ?
                            jfr_cfg->id :
                            ((g_urma_shm != NULL) ? hccp_alloc_unique_id(&g_urma_shm->next_jetty_id) : 0);
    if (jetty_id == 0)
        jetty_id = 1;
    jfr->jfr_id.id = jetty_id;
    jfr->jfr_id.uasid = ctx->uasid;
    jfr->jfr_id.eid = ctx->eid;

    urma_jfr_priv_t* p = NULL;
    for (int i = 0; i < URMA_MAX_JFRS; i++) {
        if (!g_jfr_table[i].in_use) {
            p = &g_jfr_table[i];
            break;
        }
    }
    if (p == NULL) {
        free(jfr);
        return NULL;
    }
    memset(p, 0, sizeof(*p));
    p->in_use = 1;
    p->jfr = jfr;
    p->state = URMA_JFR_STATE_RESET;
    p->jetty_id = jetty_id;
    p->jfc = jfr_cfg->jfc;

    urma_rdv_publish(jetty_id, jfr_cfg->trans_mode, jfr_cfg->token_value);

    /* first jfr in this process: start the completion pump that converts ring packets
     * into recv CQEs and jfce signals (async completion delivery stand-in) */
    if (!g_pump_started) {
        g_pump_started = 1;
        if (pthread_create(&g_pump_thread, NULL, urma_completion_pump, NULL) != 0) {
            g_pump_started = 0;
        } else {
            (void)pthread_detach(g_pump_thread);
        }
    }
    return jfr;
}

urma_status_t urma_delete_jfr(urma_jfr_t* jfr)
{
    if (jfr == NULL)
        return URMA_EINVAL;
    pthread_mutex_lock(&g_dp_mutex);
    urma_jfr_priv_t* p = jfr_find(jfr);
    if (p != NULL) {
        urma_rdv_unpublish(p->jetty_id);
        memset(p, 0, sizeof(*p));
    }
    pthread_mutex_unlock(&g_dp_mutex);
    free(jfr);
    return URMA_SUCCESS;
}

urma_status_t urma_modify_jfr(urma_jfr_t* jfr, urma_jfr_attr_t* attr)
{
    if (jfr == NULL || attr == NULL)
        return URMA_EINVAL;
    urma_jfr_priv_t* p = jfr_find(jfr);
    if (p != NULL && (attr->mask & JFR_STATE)) {
        p->state = attr->state;
    }
    return URMA_SUCCESS;
}

urma_status_t urma_query_jfr(urma_jfr_t* jfr, urma_jfr_cfg_t* cfg, urma_jfr_attr_t* attr)
{
    if (jfr == NULL)
        return URMA_EINVAL;
    if (cfg)
        *cfg = jfr->jfr_cfg;
    if (attr) {
        memset(attr, 0, sizeof(*attr));
        urma_jfr_priv_t* p = jfr_find(jfr);
        attr->state = (p != NULL) ? p->state : URMA_JFR_STATE_RESET;
    }
    return URMA_SUCCESS;
}

urma_token_id_t* urma_alloc_token_id(urma_context_t* ctx);
urma_status_t urma_free_token_id(urma_token_id_t* token_id);

urma_target_seg_t* urma_register_seg(urma_context_t* ctx, urma_seg_cfg_t* seg_cfg)
{
    if (ctx == NULL || seg_cfg == NULL)
        return NULL;
    urma_target_seg_t* tseg = (urma_target_seg_t*)calloc(1, sizeof(urma_target_seg_t));
    if (tseg == NULL)
        return NULL;
    tseg->urma_ctx = ctx;
    tseg->user_ctx = seg_cfg->user_ctx;
    tseg->seg.ubva.va = seg_cfg->va;
    uint32_t lkey = hccp_seg_register(seg_cfg->va, seg_cfg->len);
    if (lkey == 0) {
        free(tseg);
        return NULL;
    }
    tseg->seg.len = (seg_cfg->len > 0) ? seg_cfg->len : HCCP_BUF_SIZE;
    if (seg_cfg->token_id != NULL) {
        tseg->token_id = seg_cfg->token_id;
    } else {
        tseg->token_id = urma_alloc_token_id(ctx);
    }

    urma_seg_priv_t* p = seg_find_by_lkey(lkey);
    if (p == NULL) {
        if (seg_cfg->token_id == NULL)
            urma_free_token_id(tseg->token_id);
        free(tseg);
        return NULL;
    }
    p->tseg = tseg;
    p->token = seg_cfg->token_value;
    p->owns_token = (seg_cfg->token_id == NULL) ? 1 : 0;
    tseg->handle = p->lkey;
    return tseg;
}

urma_status_t urma_unregister_seg(urma_target_seg_t* target_seg)
{
    if (target_seg == NULL)
        return URMA_EINVAL;
    urma_seg_priv_t* p = seg_find(target_seg);
    int owns_token = (p != NULL) ? p->owns_token : 0;
    if (owns_token && target_seg->token_id != NULL) {
        urma_free_token_id(target_seg->token_id);
        target_seg->token_id = NULL;
    }
    if (p != NULL) {
        memset(p, 0, sizeof(*p));
    }
    free(target_seg);
    return URMA_SUCCESS;
}

urma_target_jetty_t* urma_import_jfr(urma_context_t* ctx, urma_rjfr_t* rjfr, urma_token_t* token_value)
{
    if (ctx == NULL || rjfr == NULL)
        return NULL;
    urma_target_jetty_t* tjfr = (urma_target_jetty_t*)calloc(1, sizeof(urma_target_jetty_t));
    if (tjfr == NULL)
        return NULL;
    tjfr->urma_ctx = ctx;
    tjfr->id = rjfr->jfr_id;
    tjfr->trans_mode = rjfr->trans_mode;
    tjfr->tp_type = rjfr->tp_type;
    tjfr->flag = rjfr->flag;
    if (token_value != NULL) {
        tjfr->flag.user_tag.stag = token_value->token;
    }

    urma_tjfr_priv_t* p = NULL;
    for (int i = 0; i < URMA_MAX_TJFRS; i++) {
        if (!g_tjfr_table[i].in_use) {
            p = &g_tjfr_table[i];
            break;
        }
    }
    if (p == NULL) {
        free(tjfr);
        return NULL;
    }
    memset(p, 0, sizeof(*p));
    p->in_use = 1;
    p->tjfr = tjfr;
    p->remote_jetty_id = rjfr->jfr_id.id;
    return tjfr;
}

urma_target_jetty_t*
urma_import_jfr_ex(urma_context_t* ctx, urma_rjfr_t* rjfr, urma_token_t* token_value, urma_import_jfr_ex_cfg_t* cfg)
{
    (void)cfg;
    return urma_import_jfr(ctx, rjfr, token_value);
}

urma_status_t urma_unimport_jfr(urma_target_jetty_t* target_jfr)
{
    if (target_jfr == NULL)
        return URMA_EINVAL;
    urma_tjfr_priv_t* p = tjfr_find(target_jfr);
    if (p != NULL)
        memset(p, 0, sizeof(*p));
    free(target_jfr);
    return URMA_SUCCESS;
}

urma_status_t urma_advise_jfr(urma_jfs_t* jfs, urma_target_jetty_t* tjfr)
{
    if (jfs == NULL || tjfr == NULL)
        return URMA_EINVAL;
    urma_jfs_priv_t* p = jfs_find(jfs);
    if (p != NULL) {
        p->state = URMA_JETTY_STATE_READY;
    }
    return URMA_SUCCESS;
}

urma_status_t
urma_advise_jfr_async(urma_jfs_t* jfs, urma_target_jetty_t* tjfr, urma_advise_async_cb_func cb_fun, void* cb_arg)
{
    urma_status_t ret = urma_advise_jfr(jfs, tjfr);
    if (cb_fun != NULL) {
        cb_fun(ret, cb_arg);
    }
    return ret;
}

urma_status_t urma_unadvise_jfr(urma_jfs_t* jfs, urma_target_jetty_t* tjfr)
{
    (void)jfs;
    (void)tjfr;
    return URMA_SUCCESS;
}

urma_status_t urma_send(
    urma_jfs_t* jfs, urma_target_jetty_t* target_jfr, urma_target_seg_t* src_tseg, uint64_t src, uint32_t len,
    urma_jfs_wr_flag_t flag, uint64_t user_ctx)
{
    (void)flag;
    if (jfs == NULL || target_jfr == NULL)
        return URMA_EINVAL;

    urma_jfs_priv_t* jp = jfs_find(jfs);
    urma_tjfr_priv_t* tp = tjfr_find(target_jfr);
    if (jp == NULL || tp == NULL)
        return URMA_EINVAL;

    if (src_tseg != NULL) {
        urma_seg_priv_t* sp = seg_find_by_lkey((uint32_t)src_tseg->handle);
        if (sp != NULL && src == 0) {
            src = sp->addr;
        }
    }

    int enq_rc = (g_urma_shm != NULL) ?
                     hccp_pkt_enqueue(
                         &g_urma_shm->lock, g_urma_shm->slots, URMA_RING_SLOTS, &g_urma_shm->next_seq,
                         tp->remote_jetty_id, jp->jetty_id, (const uint8_t*)(uintptr_t)src, len) :
                     -1;
    if (enq_rc != 0) {
        return URMA_EAGAIN;
    }

    urma_cr_t cr;
    memset(&cr, 0, sizeof(cr));
    cr.status = URMA_CR_SUCCESS;
    cr.user_ctx = user_ctx;
    cr.opcode = URMA_CR_OPC_SEND;
    cr.flag.bs.s_r = 0;
    cr.completion_len = len;
    cr.local_id = jp->jetty_id;
    jfc_push_cr(jp->jfc, &cr);
    return URMA_SUCCESS;
}

urma_status_t urma_recv(urma_jfr_t* jfr, urma_target_seg_t* recv_tseg, uint64_t buf, uint32_t len, uint64_t user_ctx)
{
    (void)recv_tseg;
    if (jfr == NULL)
        return URMA_EINVAL;
    urma_jfr_priv_t* p = jfr_find(jfr);
    if (p == NULL)
        return URMA_EINVAL;

    pthread_mutex_lock(&g_dp_mutex);
    if (p->recv_count >= URMA_MAX_RECV_WR) {
        pthread_mutex_unlock(&g_dp_mutex);
        return URMA_EAGAIN;
    }
    urma_recv_wr_priv_t* r = &p->recv_q[p->recv_tail];
    r->in_use = 1;
    r->buf = buf;
    r->len = len;
    r->user_ctx = user_ctx;
    p->recv_tail = (p->recv_tail + 1) % URMA_MAX_RECV_WR;
    p->recv_count++;

    jfr_drain_recv_locked(p);
    pthread_mutex_unlock(&g_dp_mutex);
    return URMA_SUCCESS;
}

int urma_poll_jfc(urma_jfc_t* jfc, int cr_cnt, urma_cr_t* cr)
{
    if (jfc == NULL || cr == NULL || cr_cnt <= 0)
        return 0;
    urma_jfc_priv_t* p = jfc_find(jfc);
    if (p == NULL)
        return 0;

    pthread_mutex_lock(&g_dp_mutex);
    for (int i = 0; i < URMA_MAX_JFRS; i++) {
        urma_jfr_priv_t* jp = &g_jfr_table[i];
        if (jp->in_use && jp->jfc == jfc) {
            jfr_drain_recv_locked(jp);
        }
    }

    int n = 0;
    while (n < cr_cnt && p->count > 0) {
        cr[n] = p->crq[p->head];
        p->head = (p->head + 1) % URMA_MAX_CQE;
        p->count--;
        n++;
    }
    /* clear the level signal so the epoll does not re-wake with zero deliverable CQEs;
     * a push racing inside this window self-heals on the next ring arrival (pump) */
    jfc_drain_jfce(p);
    pthread_mutex_unlock(&g_dp_mutex);
    return n;
}

urma_status_t urma_write(
    urma_jfs_t* jfs, urma_target_jetty_t* target_jfr, urma_target_seg_t* dst_tseg, urma_target_seg_t* src_tseg,
    uint64_t dst, uint64_t src, uint32_t len, urma_jfs_wr_flag_t flag, uint64_t user_ctx)
{
    (void)flag;
    urma_jfs_priv_t* jp = (jfs != NULL) ? jfs_find(jfs) : NULL;
    uint32_t lkey = (src_tseg != NULL) ? (uint32_t)src_tseg->handle : 0;
    if (dst == 0 && dst_tseg != NULL) {
        urma_seg_priv_t* dp = seg_find(dst_tseg);
        if (dp != NULL)
            dst = dp->addr;
    }
    pid_t peer_pid = urma_rdv_owner_pid((target_jfr != NULL) ? target_jfr->id.id : 0);
    (void)urma_seg_write_routed(lkey, dst, (const void*)(uintptr_t)src, len, peer_pid);

    if (jp != NULL && jp->jfc != NULL) {
        urma_cr_t cr;
        memset(&cr, 0, sizeof(cr));
        cr.status = URMA_CR_SUCCESS;
        cr.user_ctx = user_ctx;
        cr.opcode = hccp_seg_rw_to_cr_opcode(HCCP_SEG_RW_WRITE);
        cr.flag.bs.s_r = 0;
        cr.completion_len = len;
        cr.local_id = jp->jetty_id;
        jfc_push_cr(jp->jfc, &cr);
    }
    return URMA_SUCCESS;
}

urma_status_t urma_read(
    urma_jfs_t* jfs, urma_target_jetty_t* target_jfr, urma_target_seg_t* dst_tseg, urma_target_seg_t* src_tseg,
    uint64_t dst, uint64_t src, uint32_t len, urma_jfs_wr_flag_t flag, uint64_t user_ctx)
{
    (void)flag;
    urma_jfs_priv_t* jp = (jfs != NULL) ? jfs_find(jfs) : NULL;
    uint32_t lkey = (dst_tseg != NULL) ? (uint32_t)dst_tseg->handle : 0;
    if (src == 0 && src_tseg != NULL) {
        urma_seg_priv_t* sp = seg_find(src_tseg);
        if (sp != NULL)
            src = sp->addr;
    }
    pid_t peer_pid = urma_rdv_owner_pid((target_jfr != NULL) ? target_jfr->id.id : 0);
    (void)urma_seg_read_routed(lkey, (void*)(uintptr_t)dst, src, len, peer_pid);

    if (jp != NULL && jp->jfc != NULL) {
        urma_cr_t cr;
        memset(&cr, 0, sizeof(cr));
        cr.status = URMA_CR_SUCCESS;
        cr.user_ctx = user_ctx;
        cr.opcode = hccp_seg_rw_to_cr_opcode(HCCP_SEG_RW_READ);
        cr.flag.bs.s_r = 0;
        cr.completion_len = len;
        cr.local_id = jp->jetty_id;
        jfc_push_cr(jp->jfc, &cr);
    }
    (void)user_ctx;
    return URMA_SUCCESS;
}

urma_jfce_t* urma_create_jfce(urma_context_t* ctx)
{
    if (ctx == NULL)
        return NULL;
    urma_jfce_t* jfce = (urma_jfce_t*)calloc(1, sizeof(urma_jfce_t));
    if (jfce == NULL)
        return NULL;
    jfce->urma_ctx = ctx;
    jfce->fd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    return jfce;
}

urma_status_t urma_delete_jfce(urma_jfce_t* jfce)
{
    if (jfce == NULL)
        return URMA_EINVAL;
    if (jfce->fd > 0)
        close(jfce->fd);
    free(jfce);
    return URMA_SUCCESS;
}

int urma_wait_jfc(urma_jfce_t* jfce, uint32_t jfc_cnt, int time_out, urma_jfc_t* jfc[])
{
    if (jfce == NULL || jfc == NULL || jfc_cnt == 0)
        return -1;

    /* poll the jfcs bound to this event channel for pending completions; the rs ping
     * loop (rs_ping_urma.c RsPingUrmaPollRcq) calls this with time_out=0 from its
     * epoll handler, so the fast path must not sleep */
    for (;;) {
        pthread_mutex_lock(&g_dp_mutex);
        int found = 0;
        for (int i = 0; i < URMA_MAX_JFCS && found < (int)jfc_cnt; i++) {
            if (g_jfc_table[i].in_use && g_jfc_table[i].jfce == jfce && g_jfc_table[i].count > 0) {
                jfc[found] = g_jfc_table[i].jfc;
                found++;
            }
        }
        pthread_mutex_unlock(&g_dp_mutex);
        if (found > 0 || time_out <= 0) {
            return found;
        }
        usleep(1000);
        time_out--;
    }
}

void urma_ack_jfc(urma_jfc_t* jfc[], uint32_t nevents[], uint32_t jfc_cnt)
{
    /* completion ring level signals are cleared by urma_poll_jfc once the CQEs are
     * drained; ack has nothing left to consume */
    (void)jfc;
    (void)nevents;
    (void)jfc_cnt;
}

urma_status_t urma_get_uasid(uint32_t* uasid)
{
    if (uasid == NULL)
        return URMA_EINVAL;
    if (!g_inited) {
        (void)urma_init(NULL);
    }
    *uasid = g_uasid;
    return URMA_SUCCESS;
}

static urma_vlog_level_t g_log_level = URMA_VLOG_LEVEL_ERR;
static char g_log_thread_tag[32] = "urma";

urma_status_t urma_register_log_func(urma_log_cb_t func)
{
    (void)func;
    return URMA_SUCCESS;
}

urma_status_t urma_unregister_log_func(void) { return URMA_SUCCESS; }

urma_vlog_level_t urma_log_get_level(void) { return g_log_level; }

void urma_log_set_level(urma_vlog_level_t level)
{
    if (level >= URMA_VLOG_LEVEL_EMERG && level < URMA_VLOG_LEVEL_MAX) {
        g_log_level = level;
    }
}

const char* urma_log_get_thread_tag(void) { return g_log_thread_tag; }

void urma_log_set_thread_tag(const char* tag)
{
    if (tag == NULL)
        return;
    snprintf(g_log_thread_tag, sizeof(g_log_thread_tag), "%s", tag);
}

urma_status_t urma_post_jfs_wr(urma_jfs_t* jfs, urma_jfs_wr_t* wr, urma_jfs_wr_t** bad_wr)
{
    if (jfs == NULL || wr == NULL) {
        if (bad_wr)
            *bad_wr = wr;
        return URMA_EINVAL;
    }
    urma_jfs_wr_t* cur = wr;
    while (cur != NULL) {
        urma_jfs_wr_flag_t flag = cur->flag;
        uint64_t user_ctx = cur->user_ctx;
        urma_target_jetty_t* tjfr = cur->tjetty;
        urma_status_t ret = URMA_SUCCESS;
        if (cur->opcode == URMA_OPC_SEND || cur->opcode == URMA_OPC_SEND_IMM
            || cur->opcode == URMA_OPC_SEND_INVALIDATE) {
            uint64_t src = 0;
            uint32_t len = 0;
            if (cur->send.src.num_sge > 0 && cur->send.src.sge != NULL) {
                src = cur->send.src.sge[0].addr;
                len = cur->send.src.sge[0].len;
            }
            ret = urma_send(jfs, tjfr, NULL, src, len, flag, user_ctx);
        } else if (
            cur->opcode == URMA_OPC_WRITE || cur->opcode == URMA_OPC_WRITE_IMM
            || cur->opcode == URMA_OPC_WRITE_NOTIFY) {
            uint64_t src = 0;
            uint32_t len = 0;
            urma_target_seg_t* src_tseg = NULL;
            if (cur->rw.src.num_sge > 0 && cur->rw.src.sge != NULL) {
                src = cur->rw.src.sge[0].addr;
                len = cur->rw.src.sge[0].len;
                src_tseg = cur->rw.src.sge[0].tseg;
            }
            uint64_t dst = 0;
            urma_target_seg_t* dst_tseg = NULL;
            if (cur->rw.dst.num_sge > 0 && cur->rw.dst.sge != NULL) {
                dst = cur->rw.dst.sge[0].addr;
                dst_tseg = cur->rw.dst.sge[0].tseg;
            }
            ret = urma_write(jfs, tjfr, dst_tseg, src_tseg, dst, src, len, flag, user_ctx);
        } else if (cur->opcode == URMA_OPC_READ) {
            uint64_t dst = 0;
            uint32_t len = 0;
            urma_target_seg_t* dst_tseg = NULL;
            if (cur->rw.dst.num_sge > 0 && cur->rw.dst.sge != NULL) {
                dst = cur->rw.dst.sge[0].addr;
                len = cur->rw.dst.sge[0].len;
                dst_tseg = cur->rw.dst.sge[0].tseg;
            }
            uint64_t src = 0;
            urma_target_seg_t* src_tseg = NULL;
            if (cur->rw.src.num_sge > 0 && cur->rw.src.sge != NULL) {
                src = cur->rw.src.sge[0].addr;
                src_tseg = cur->rw.src.sge[0].tseg;
            }
            ret = urma_read(jfs, tjfr, dst_tseg, src_tseg, dst, src, len, flag, user_ctx);
        } else {
        }
        if (ret != URMA_SUCCESS) {
            if (bad_wr)
                *bad_wr = cur;
            return ret;
        }
        cur = cur->next;
    }
    if (bad_wr)
        *bad_wr = NULL;
    return URMA_SUCCESS;
}

urma_status_t urma_post_jfr_wr(urma_jfr_t* jfr, urma_jfr_wr_t* wr, urma_jfr_wr_t** bad_wr)
{
    if (jfr == NULL || wr == NULL) {
        if (bad_wr)
            *bad_wr = wr;
        return URMA_EINVAL;
    }
    urma_jfr_wr_t* cur = wr;
    while (cur != NULL) {
        uint64_t buf = 0;
        uint32_t len = 0;
        if (cur->src.num_sge > 0 && cur->src.sge != NULL) {
            buf = cur->src.sge[0].addr;
            len = cur->src.sge[0].len;
        }
        urma_status_t ret = urma_recv(jfr, NULL, buf, len, cur->user_ctx);
        if (ret != URMA_SUCCESS) {
            if (bad_wr)
                *bad_wr = cur;
            return ret;
        }
        cur = cur->next;
    }
    if (bad_wr)
        *bad_wr = NULL;
    return URMA_SUCCESS;
}

urma_status_t urma_post_jetty_send_wr(urma_jetty_t* jetty, urma_jfs_wr_t* wr, urma_jfs_wr_t** bad_wr)
{
    /* A jetty bundles a jfs and a shared jfr (urma_create_jetty); posting on the jetty
     * is the same operation as posting on its jfs: the SEND opcode routes the payload
     * into the shm ring addressed to the WR's target jetty (urma_send), and the
     * WRITE/READ opcodes reuse the seg-based copy paths. The rs ping data plane
     * (rs_ping_urma.c RsPingUrmaPostSend / RsPongJettyPostSend) drives the ping and
     * pong packets through this entry. */
    urma_jetty_priv_t* gp = jetty_find(jetty);
    if (gp == NULL || gp->jfs == NULL) {
        if (bad_wr)
            *bad_wr = wr;
        return URMA_EINVAL;
    }
    return urma_post_jfs_wr(gp->jfs, wr, bad_wr);
}

urma_status_t urma_post_jetty_recv_wr(urma_jetty_t* jetty, urma_jfr_wr_t* wr, urma_jfr_wr_t** bad_wr)
{
    /* Delegate to the recv queue of the jetty's shared jfr; urma_recv queues the buffer
     * and immediately drains any packets already sitting in the ring for this jetty id.
     * The rs ping path pre-posts its recv buffers through this entry
     * (rs_ping_urma.c RsPingCommonJfrPostRecv). */
    urma_jetty_priv_t* gp = jetty_find(jetty);
    if (gp == NULL || gp->jfr == NULL) {
        if (bad_wr)
            *bad_wr = wr;
        return URMA_EINVAL;
    }
    return urma_post_jfr_wr(gp->jfr, wr, bad_wr);
}

int urma_flush_jfs(urma_jfs_t* jfs, int cr_cnt, urma_cr_t* cr)
{
    (void)jfs;
    (void)cr_cnt;
    (void)cr;
    return 0;
}

int urma_flush_jetty(urma_jetty_t* jetty, int cr_cnt, urma_cr_t* cr)
{
    (void)jetty;
    (void)cr_cnt;
    (void)cr;
    return 0;
}

urma_status_t urma_alloc_jfc(urma_context_t* urma_ctx, urma_jfc_cfg_t* cfg, urma_jfc_t** jfc)
{
    urma_jfc_priv_t* p = NULL;
    if (urma_ctx == NULL || cfg == NULL || jfc == NULL)
        return URMA_EINVAL;
    *jfc = urma_create_jfc(urma_ctx, cfg);
    if (*jfc == NULL)
        return URMA_ENOMEM;
    /* the ext API serves STARS_POLL / CCU_POLL / USER_CTL_NORMAL cqs: tag the jfc so
     * completion recording mirrors hardware (region polling, no jfc CR events) */
    p = jfc_find(*jfc);
    if (p != NULL)
        p->ext_mode = 1;
    return URMA_SUCCESS;
}

urma_status_t urma_set_jfc_opt(urma_jfc_t* jfc, uint64_t opt, void* buf, uint32_t len)
{
    (void)jfc;
    (void)opt;
    (void)buf;
    (void)len;
    return URMA_SUCCESS;
}

urma_status_t urma_active_jfc(urma_jfc_t* jfc)
{
    (void)jfc;
    return URMA_SUCCESS;
}

urma_status_t urma_get_jfc_opt(urma_jfc_t* jfc, uint64_t opt, void* buf, uint32_t len)
{
    (void)jfc;
    (void)opt;
    (void)buf;
    (void)len;
    return URMA_SUCCESS;
}

urma_status_t urma_deactive_jfc(urma_jfc_t* jfc)
{
    (void)jfc;
    return URMA_SUCCESS;
}

urma_status_t urma_free_jfc(urma_jfc_t* jfc) { return urma_delete_jfc(jfc); }

urma_status_t urma_delete_jfc_batch(urma_jfc_t** jfc_arr, int jfc_num, urma_jfc_t** bad_jfc)
{
    if (jfc_arr == NULL || jfc_num <= 0)
        return URMA_EINVAL;
    for (int i = 0; i < jfc_num; i++) {
        if (jfc_arr[i] != NULL) {
            urma_status_t ret = urma_delete_jfc(jfc_arr[i]);
            if (ret != URMA_SUCCESS) {
                if (bad_jfc)
                    *bad_jfc = jfc_arr[i];
                return ret;
            }
        }
    }
    if (bad_jfc)
        *bad_jfc = NULL;
    return URMA_SUCCESS;
}

urma_status_t urma_alloc_jfs(urma_context_t* urma_ctx, urma_jfs_cfg_t* cfg, urma_jfs_t** jfs)
{
    if (urma_ctx == NULL || cfg == NULL || jfs == NULL)
        return URMA_EINVAL;
    *jfs = urma_create_jfs(urma_ctx, cfg);
    return (*jfs != NULL) ? URMA_SUCCESS : URMA_ENOMEM;
}

urma_status_t urma_set_jfs_opt(urma_jfs_t* jfs, uint64_t opt, void* buf, uint32_t len)
{
    (void)jfs;
    (void)opt;
    (void)buf;
    (void)len;
    return URMA_SUCCESS;
}

urma_status_t urma_active_jfs(urma_jfs_t* jfs)
{
    (void)jfs;
    return URMA_SUCCESS;
}

urma_status_t urma_get_jfs_opt(urma_jfs_t* jfs, uint64_t opt, void* buf, uint32_t len)
{
    (void)jfs;
    (void)opt;
    (void)buf;
    (void)len;
    return URMA_SUCCESS;
}

urma_status_t urma_deactive_jfs(urma_jfs_t* jfs)
{
    (void)jfs;
    return URMA_SUCCESS;
}

urma_status_t urma_free_jfs(urma_jfs_t* jfs) { return urma_delete_jfs(jfs); }

urma_status_t urma_delete_jfs_batch(urma_jfs_t** jfs_arr, int jfs_num, urma_jfs_t** bad_jfs)
{
    if (jfs_arr == NULL || jfs_num <= 0)
        return URMA_EINVAL;
    for (int i = 0; i < jfs_num; i++) {
        if (jfs_arr[i] != NULL) {
            urma_status_t ret = urma_delete_jfs(jfs_arr[i]);
            if (ret != URMA_SUCCESS) {
                if (bad_jfs)
                    *bad_jfs = jfs_arr[i];
                return ret;
            }
        }
    }
    if (bad_jfs)
        *bad_jfs = NULL;
    return URMA_SUCCESS;
}

urma_status_t urma_alloc_jfr(urma_context_t* urma_ctx, urma_jfr_cfg_t* cfg, urma_jfr_t** jfr)
{
    if (urma_ctx == NULL || cfg == NULL || jfr == NULL)
        return URMA_EINVAL;
    *jfr = urma_create_jfr(urma_ctx, cfg);
    return (*jfr != NULL) ? URMA_SUCCESS : URMA_ENOMEM;
}

urma_status_t urma_set_jfr_opt(urma_jfr_t* jfr, uint64_t opt, void* buf, uint32_t len)
{
    (void)jfr;
    (void)opt;
    (void)buf;
    (void)len;
    return URMA_SUCCESS;
}

urma_status_t urma_active_jfr(urma_jfr_t* jfr)
{
    if (jfr == NULL)
        return URMA_EINVAL;
    urma_jfr_priv_t* p = jfr_find(jfr);
    if (p != NULL)
        p->state = URMA_JFR_STATE_READY;
    return URMA_SUCCESS;
}

urma_status_t urma_get_jfr_opt(urma_jfr_t* jfr, uint64_t opt, void* buf, uint32_t len)
{
    (void)jfr;
    (void)opt;
    (void)buf;
    (void)len;
    return URMA_SUCCESS;
}

urma_status_t urma_deactive_jfr(urma_jfr_t* jfr)
{
    (void)jfr;
    return URMA_SUCCESS;
}

urma_status_t urma_free_jfr(urma_jfr_t* jfr) { return urma_delete_jfr(jfr); }

urma_status_t urma_delete_jfr_batch(urma_jfr_t** jfr_arr, int jfr_num, urma_jfr_t** bad_jfr)
{
    if (jfr_arr == NULL || jfr_num <= 0)
        return URMA_EINVAL;
    for (int i = 0; i < jfr_num; i++) {
        if (jfr_arr[i] != NULL) {
            urma_status_t ret = urma_delete_jfr(jfr_arr[i]);
            if (ret != URMA_SUCCESS) {
                if (bad_jfr)
                    *bad_jfr = jfr_arr[i];
                return ret;
            }
        }
    }
    if (bad_jfr)
        *bad_jfr = NULL;
    return URMA_SUCCESS;
}

static uint64_t g_jetty_dummy_dwqe[16];
static uint64_t g_jetty_dummy_db[2];
static uint64_t g_jetty_dummy_ctrl[16];

static urma_jetty_priv_t* jetty_find(urma_jetty_t* jetty)
{
    for (int i = 0; i < URMA_MAX_JETTYS; i++) {
        if (g_jetty_table[i].in_use && g_jetty_table[i].jetty == jetty) {
            return &g_jetty_table[i];
        }
    }
    return NULL;
}

static void urma_jetty_post_recv(urma_context_t* ctx, urma_jfr_priv_t* frp)
{
    if (ctx == NULL || frp == NULL)
        return;
    urma_seg_priv_t* recv_seg = NULL;
    for (int i = 0; i < URMA_MAX_SEGS; i++) {
        if (g_seg_table[i].in_use && g_seg_table[i].tseg != NULL && g_seg_table[i].tseg->urma_ctx == ctx) {
            recv_seg = &g_seg_table[i];
        }
    }
    if (recv_seg == NULL)
        return;
    /* the completion pump may already be draining this jfr (it starts on the first
     * urma_create_jfr, which precedes urma_create_jetty) */
    pthread_mutex_lock(&g_dp_mutex);
    for (int k = 0; k < URMA_JETTY_RECV_WR_NUM; k++) {
        if (frp->recv_count >= URMA_MAX_RECV_WR)
            break;
        urma_recv_wr_priv_t* r = &frp->recv_q[frp->recv_tail];
        r->in_use = 1;
        r->buf = recv_seg->addr;
        r->len = (uint32_t)recv_seg->len;
        r->user_ctx = 0;
        frp->recv_tail = (frp->recv_tail + 1) % URMA_MAX_RECV_WR;
        frp->recv_count++;
    }
    pthread_mutex_unlock(&g_dp_mutex);
}

static void urma_jetty_drain(urma_jetty_priv_t* gp)
{
    if (gp == NULL || gp->jfr == NULL)
        return;
    urma_jfr_priv_t* frp = jfr_find(gp->jfr);
    if (frp != NULL)
        jfr_drain_recv(frp);
}

urma_jetty_t* urma_create_jetty(urma_context_t* ctx, urma_jetty_cfg_t* jetty_cfg)
{
    if (ctx == NULL || jetty_cfg == NULL || jetty_cfg->shared.jfr == NULL) {
        return NULL;
    }

    urma_jetty_t* jetty = (urma_jetty_t*)calloc(1, sizeof(urma_jetty_t));
    if (jetty == NULL)
        return NULL;
    jetty->urma_ctx = ctx;
    jetty->jetty_cfg = *jetty_cfg;

    uint32_t jetty_id = (jetty_cfg->id != 0) ?
                            jetty_cfg->id :
                            ((g_urma_shm != NULL) ? hccp_alloc_unique_id(&g_urma_shm->next_jetty_id) : 0);
    if (jetty_id == 0)
        jetty_id = 1;
    jetty->jetty_id.id = jetty_id;
    jetty->jetty_id.uasid = ctx->uasid;
    jetty->jetty_id.eid = ctx->eid;
    jetty->remote_jetty = NULL;

    urma_jfs_t* jfs = urma_create_jfs(ctx, &jetty_cfg->jfs_cfg);
    if (jfs == NULL) {
        free(jetty);
        return NULL;
    }
    jfs->jfs_id.id = jetty_id;
    urma_jfs_priv_t* jp = jfs_find(jfs);
    if (jp != NULL) {
        jp->jetty_id = jetty_id;
        jp->state = URMA_JETTY_STATE_READY;
    }

    urma_jfr_priv_t* frp = jfr_find(jetty_cfg->shared.jfr);
    if (frp != NULL) {
        frp->jetty_id = jetty_id;
        frp->jfr->jfr_id.id = jetty_id;
        frp->jfc = jetty_cfg->shared.jfc;
        urma_jetty_post_recv(ctx, frp);
    }

    urma_jetty_priv_t* gp = NULL;
    for (int i = 0; i < URMA_MAX_JETTYS; i++) {
        if (!g_jetty_table[i].in_use) {
            gp = &g_jetty_table[i];
            break;
        }
    }
    if (gp == NULL) {
        free(jetty);
        return NULL;
    }
    memset(gp, 0, sizeof(*gp));
    gp->in_use = 1;
    gp->jetty = jetty;
    gp->jfs = jfs;
    gp->jfr = jetty_cfg->shared.jfr;
    gp->send_jfc = jetty_cfg->jfs_cfg.jfc;
    gp->jetty_id = jetty_id;

    urma_token_t zero_token = {0};
    urma_rdv_publish(jetty_id, jetty_cfg->jfs_cfg.trans_mode, zero_token);
    return jetty;
}

urma_status_t urma_modify_jetty(urma_jetty_t* jetty, urma_jetty_attr_t* attr)
{
    (void)attr;
    if (jetty == NULL)
        return URMA_EINVAL;
    return URMA_SUCCESS;
}

urma_status_t urma_query_jetty(urma_jetty_t* jetty, urma_jetty_cfg_t* cfg, urma_jetty_attr_t* attr)
{
    if (jetty == NULL)
        return URMA_EINVAL;
    if (cfg)
        *cfg = jetty->jetty_cfg;
    if (attr) {
        memset(attr, 0, sizeof(*attr));
        attr->state = URMA_JETTY_STATE_READY;
    }
    return URMA_SUCCESS;
}

urma_status_t urma_delete_jetty(urma_jetty_t* jetty)
{
    if (jetty == NULL)
        return URMA_EINVAL;
    pthread_mutex_lock(&g_dp_mutex);
    /* release the rendezvous entry and the jetty table slot; without this every
     * created jetty leaks one rdv entry and the shared table (URMA_MAX_RDV) fills
     * after ~32 jetty-creating cases, silently breaking the owner lookup that
     * routes cross-process writes for every later case (heap memory is reclaimed
     * at process exit, the shared table is not) */
    urma_rdv_unpublish(jetty->jetty_id.id);
    urma_jetty_priv_t* gp = jetty_find(jetty);
    if (gp != NULL)
        memset(gp, 0, sizeof(*gp));
    pthread_mutex_unlock(&g_dp_mutex);
    free(jetty);
    return URMA_SUCCESS;
}

urma_status_t urma_delete_jetty_batch(urma_jetty_t** jetty_arr, int jetty_num, urma_jetty_t** bad_jetty)
{
    if (jetty_arr == NULL || jetty_num <= 0)
        return URMA_EINVAL;
    for (int i = 0; i < jetty_num; i++) {
        if (urma_delete_jetty(jetty_arr[i]) != URMA_SUCCESS) {
            if (bad_jetty != NULL)
                *bad_jetty = jetty_arr[i];
            return URMA_EINVAL;
        }
    }
    return URMA_SUCCESS;
}

urma_target_jetty_t* urma_import_jetty(urma_context_t* ctx, urma_rjetty_t* rjetty, urma_token_t* token_value)
{
    if (ctx == NULL || rjetty == NULL)
        return NULL;
    urma_target_jetty_t* tjetty = (urma_target_jetty_t*)calloc(1, sizeof(urma_target_jetty_t));
    if (tjetty == NULL)
        return NULL;
    tjetty->urma_ctx = ctx;
    tjetty->id = rjetty->jetty_id;
    tjetty->trans_mode = rjetty->trans_mode;
    tjetty->tp_type = rjetty->tp_type;
    tjetty->flag = rjetty->flag;
    tjetty->policy = rjetty->policy;
    tjetty->type = rjetty->type;
    if (token_value != NULL) {
        tjetty->flag.user_tag.stag = token_value->token;
    }

    urma_tjfr_priv_t* p = NULL;
    for (int i = 0; i < URMA_MAX_TJFRS; i++) {
        if (!g_tjfr_table[i].in_use) {
            p = &g_tjfr_table[i];
            break;
        }
    }
    if (p == NULL) {
        free(tjetty);
        return NULL;
    }
    memset(p, 0, sizeof(*p));
    p->in_use = 1;
    p->tjfr = tjetty;
    p->remote_jetty_id = rjetty->jetty_id.id;
    return tjetty;
}

urma_target_jetty_t* urma_import_jetty_ex(
    urma_context_t* ctx, urma_rjetty_t* rjetty, urma_token_t* token_value, urma_import_jetty_ex_cfg_t* cfg)
{
    (void)cfg;
    return urma_import_jetty(ctx, rjetty, token_value);
}

urma_status_t urma_unimport_jetty(urma_target_jetty_t* tjetty)
{
    (void)tjetty;
    return URMA_SUCCESS;
}

urma_status_t urma_advise_jetty(urma_jetty_t* jetty, urma_target_jetty_t* tjetty)
{
    (void)jetty;
    (void)tjetty;
    return ENOTSUP;
}

urma_status_t urma_unadvise_jetty(urma_jetty_t* jetty, urma_target_jetty_t* tjetty)
{
    (void)jetty;
    (void)tjetty;
    return URMA_SUCCESS;
}

urma_status_t urma_bind_jetty(urma_jetty_t* jetty, urma_target_jetty_t* tjetty)
{
    if (jetty == NULL || tjetty == NULL)
        return URMA_EINVAL;
    jetty->remote_jetty = tjetty;
    return URMA_SUCCESS;
}

urma_status_t urma_bind_jetty_ex(urma_jetty_t* jetty, urma_target_jetty_t* tjetty, urma_bind_jetty_ex_cfg_t* cfg)
{
    (void)jetty;
    (void)tjetty;
    (void)cfg;
    return ENOTSUP;
}

urma_status_t urma_unbind_jetty(urma_jetty_t* jetty)
{
    (void)jetty;
    return URMA_SUCCESS;
}

urma_target_jetty_t* urma_import_jetty_async(
    urma_notifier_t* notifier, const urma_rjetty_t* rjetty, const urma_token_t* token_value, uint64_t user_ctx,
    int timeout)
{
    (void)notifier;
    (void)rjetty;
    (void)token_value;
    (void)user_ctx;
    (void)timeout;
    return NULL;
}

urma_status_t urma_unimport_jetty_async(urma_target_jetty_t* tjetty)
{
    (void)tjetty;
    return URMA_SUCCESS;
}

urma_status_t urma_bind_jetty_async(
    urma_notifier_t* notifier, urma_jetty_t* jetty, urma_target_jetty_t* tjetty, uint64_t user_ctx, int timeout)
{
    (void)notifier;
    (void)jetty;
    (void)tjetty;
    (void)user_ctx;
    (void)timeout;
    return ENOTSUP;
}

urma_status_t urma_unbind_jetty_async(urma_jetty_t* jetty)
{
    (void)jetty;
    return URMA_SUCCESS;
}

urma_notifier_t* urma_create_notifier(urma_context_t* ctx)
{
    (void)ctx;
    return NULL;
}

urma_status_t urma_delete_notifier(urma_notifier_t* notifier)
{
    (void)notifier;
    return URMA_SUCCESS;
}

urma_status_t urma_alloc_jetty(urma_context_t* urma_ctx, urma_jetty_cfg_t* cfg, urma_jetty_t** jetty)
{
    /* production allocates a jetty by cfg for the ext modes (USER_CTL_NORMAL and
     * friends go through RsUbCtxExtJettyCreate -> rs_urma_alloc_jetty); the sim
     * delegates to the same table-backed create used by the URMA_NORMAL path */
    if (urma_ctx == NULL || cfg == NULL || jetty == NULL) {
        return URMA_EINVAL;
    }
    *jetty = urma_create_jetty(urma_ctx, cfg);
    return (*jetty == NULL) ? URMA_ENOMEM : URMA_SUCCESS;
}

urma_status_t urma_set_jetty_opt(urma_jetty_t* jetty, uint64_t opt, void* buf, uint32_t len)
{
    (void)jetty;
    (void)opt;
    (void)buf;
    (void)len;
    return URMA_SUCCESS;
}

urma_status_t urma_active_jetty(urma_jetty_t* jetty)
{
    (void)jetty;
    return URMA_SUCCESS;
}

urma_status_t urma_get_jetty_opt(urma_jetty_t* jetty, uint64_t opt, void* buf, uint32_t len)
{
    if (jetty == NULL || buf == NULL)
        return URMA_EINVAL;
    /* hardware reports the jetty's sqe buffer base and doorbell address; the sim
     * serves them from the standing dummy buffers. rs maps the ranges into the
     * caller's va space via halResAddrMapV2 for the modes that need it (the map
     * ignores the input va), and the CCU jetty db registration
     * (RsCcuJettyDbReg -> RsUbCtxRegJettyDb) registers the page-aligned db
     * address, which only has to be non-NULL for the sim lmem bookkeeping. */
    if (opt == URMA_JFS_SQE_BASE_ADDR || opt == URMA_JFS_DB_ADDR) {
        if (len < sizeof(uint64_t))
            return URMA_EINVAL;
        *(uint64_t*)buf = (opt == URMA_JFS_SQE_BASE_ADDR) ? (uint64_t)(uintptr_t)g_jetty_dummy_ctrl :
                                                            (uint64_t)(uintptr_t)g_jetty_dummy_db;
    }
    /* other options (URMA_JETTY_FULL_CTX for async events) carry no sim payload */
    return URMA_SUCCESS;
}

urma_status_t urma_deactive_jetty(urma_jetty_t* jetty)
{
    (void)jetty;
    return URMA_SUCCESS;
}

urma_status_t urma_free_jetty(urma_jetty_t* jetty)
{
    (void)jetty;
    return URMA_SUCCESS;
}

int urma_wait_notify(urma_notifier_t* notifier, uint32_t cnt, urma_notify_t* notify, int timeout)
{
    (void)notifier;
    (void)cnt;
    (void)notify;
    (void)timeout;
    return 0;
}

urma_status_t urma_ack_notify(urma_context_t* ctx, uint32_t cnt, urma_notify_t* notify)
{
    (void)ctx;
    (void)cnt;
    (void)notify;
    return URMA_SUCCESS;
}

urma_jetty_grp_t* urma_create_jetty_grp(urma_context_t* ctx, urma_jetty_grp_cfg_t* cfg)
{
    (void)ctx;
    (void)cfg;
    return NULL;
}

urma_status_t urma_delete_jetty_grp(urma_jetty_grp_t* jetty_grp)
{
    (void)jetty_grp;
    return URMA_SUCCESS;
}

urma_status_t urma_get_async_event(urma_context_t* ctx, urma_async_event_t* event)
{
    (void)ctx;
    (void)event;
    return URMA_EAGAIN;
}

void urma_ack_async_event(urma_async_event_t* event) { (void)event; }

urma_token_id_t* urma_alloc_token_id(urma_context_t* ctx)
{
    if (ctx == NULL)
        return NULL;
    urma_token_id_t* tid = (urma_token_id_t*)calloc(1, sizeof(urma_token_id_t));
    if (tid == NULL)
        return NULL;
    tid->urma_ctx = ctx;
    tid->token_id = (g_urma_shm != NULL) ? hccp_alloc_unique_id(&g_urma_shm->next_jetty_id) : 0;
    if (tid->token_id == 0)
        tid->token_id = 1;
    return tid;
}

urma_token_id_t* urma_alloc_token_id_ex(urma_context_t* ctx, urma_token_id_flag_t flag)
{
    (void)flag;
    return urma_alloc_token_id(ctx);
}

urma_status_t urma_free_token_id(urma_token_id_t* token_id)
{
    if (token_id == NULL)
        return URMA_EINVAL;
    free(token_id);
    return URMA_SUCCESS;
}

urma_target_seg_t* urma_import_seg(
    urma_context_t* ctx, urma_seg_t* seg, urma_token_t* token_value, uint64_t addr, urma_import_seg_flag_t flag)
{
    (void)flag;
    if (ctx == NULL)
        return NULL;
    urma_target_seg_t* tseg = (urma_target_seg_t*)calloc(1, sizeof(urma_target_seg_t));
    if (tseg == NULL)
        return NULL;
    tseg->urma_ctx = ctx;
    if (seg != NULL) {
        tseg->seg = *seg;
    }
    tseg->mva = addr;

    urma_seg_priv_t* p = NULL;
    for (int i = 0; i < URMA_MAX_SEGS; i++) {
        if (!g_seg_table[i].in_use) {
            p = &g_seg_table[i];
            break;
        }
    }
    if (p == NULL) {
        free(tseg);
        return NULL;
    }
    memset(p, 0, sizeof(*p));
    p->in_use = 1;
    p->tseg = tseg;
    p->addr = (seg != NULL) ? seg->ubva.va : addr;
    p->len = (seg != NULL) ? seg->len : 0;
    p->seg_id = (g_urma_shm != NULL) ? hccp_alloc_unique_id(&g_urma_shm->next_jetty_id) : 0;
    if (p->seg_id == 0)
        p->seg_id = 1;
    p->lkey = p->seg_id;
    if (token_value != NULL)
        p->token = *token_value;
    tseg->handle = p->lkey;
    return tseg;
}

urma_status_t urma_unimport_seg(urma_target_seg_t* tseg)
{
    if (tseg == NULL)
        return URMA_EINVAL;
    urma_seg_priv_t* p = seg_find(tseg);
    if (p != NULL) {
        memset(p, 0, sizeof(*p));
    }
    free(tseg);
    return URMA_SUCCESS;
}

urma_status_t urma_user_ctl(urma_context_t* ctx, urma_user_ctl_in_t* in, urma_user_ctl_out_t* out)
{
    (void)ctx;
    if (in == NULL)
        return URMA_EINVAL;

    switch (in->opcode) {
        case UDMA_U_USER_CTL_POST_WR: {
            if (in->addr == 0 || in->len < sizeof(struct udma_u_wr_ex))
                return URMA_EINVAL;
            struct udma_u_wr_ex* wrEx = (struct udma_u_wr_ex*)(uintptr_t)in->addr;

            urma_jfc_t* send_jfc = NULL;
            uint32_t src_jetty_id = 0;
            urma_target_jetty_t* target = NULL;
            urma_jetty_priv_t* gp = NULL;

            if (wrEx->is_jetty && wrEx->jetty != NULL) {
                gp = jetty_find(wrEx->jetty);
                if (gp == NULL)
                    return URMA_EINVAL;
                urma_jetty_drain(gp);
                send_jfc = gp->send_jfc;
                src_jetty_id = gp->jetty_id;
                target = (wrEx->wr != NULL) ? wrEx->wr->wr.tjetty : NULL;
                if (target == NULL)
                    target = wrEx->jetty->remote_jetty;
            } else if (wrEx->jfs != NULL) {
                urma_jfs_priv_t* jp = jfs_find(wrEx->jfs);
                if (jp == NULL)
                    return URMA_EINVAL;
                send_jfc = jp->jfc;
                src_jetty_id = jp->jetty_id;
                target = (wrEx->wr != NULL) ? wrEx->wr->wr.tjetty : NULL;
            } else {
                return URMA_EINVAL;
            }
            if (target == NULL)
                return URMA_EINVAL;
            uint32_t dst_jetty_id = target->id.id;
            /* peer process owning the remote jetty (and its devmem segments) */
            pid_t peer_pid = urma_rdv_owner_pid(dst_jetty_id);

            uint64_t src = 0;
            uint32_t len = 0;
            hccp_seg_rw_kind_t rw_kind = HCCP_SEG_RW_SEND;
            urma_opcode_t opc = (wrEx->wr != NULL) ? wrEx->wr->wr.opcode : URMA_OPC_NOP;
            if (wrEx->wr != NULL) {
                urma_jfs_wr_t* w = &wrEx->wr->wr;
                if (opc == URMA_OPC_WRITE || opc == URMA_OPC_WRITE_IMM || opc == URMA_OPC_WRITE_NOTIFY) {
                    rw_kind = HCCP_SEG_RW_WRITE;
                    urma_target_seg_t* src_tseg = NULL;
                    if (w->rw.src.num_sge > 0 && w->rw.src.sge != NULL) {
                        src = w->rw.src.sge[0].addr;
                        len = w->rw.src.sge[0].len;
                        src_tseg = w->rw.src.sge[0].tseg;
                        if (src == 0 && src_tseg != NULL) {
                            urma_seg_priv_t* sp = seg_find_by_lkey((uint32_t)src_tseg->handle);
                            if (sp != NULL)
                                src = sp->addr;
                        }
                    }
                    uint64_t dst_va = 0;
                    if (w->rw.dst.num_sge > 0 && w->rw.dst.sge != NULL) {
                        dst_va = w->rw.dst.sge[0].addr;
                        if (dst_va == 0 && w->rw.dst.sge[0].tseg != NULL) {
                            urma_seg_priv_t* dp = seg_find(w->rw.dst.sge[0].tseg);
                            if (dp != NULL)
                                dst_va = dp->addr;
                        }
                    }
                    uint32_t lkey = (src_tseg != NULL) ? (uint32_t)src_tseg->handle : 0;
                    (void)urma_seg_write_routed(lkey, dst_va, (const void*)(uintptr_t)src, len, peer_pid);
                    /* WRITE_WITH_NOTIFY side effect: hardware records the notify value
                     * into the notify segment (rsge[1] + notify_data) when the data
                     * write completes; mirror it so the peer's rtNotifyWait observes
                     * the record (rs only fills rsge[1] for URMA_OPC_WRITE_NOTIFY). */
                    if (opc == URMA_OPC_WRITE_NOTIFY && w->rw.dst.num_sge > 1 && w->rw.dst.sge != NULL
                        && w->rw.notify_data != 0) {
                        uint64_t notify_va = w->rw.dst.sge[1].addr;
                        if (notify_va == 0 && w->rw.dst.sge[1].tseg != NULL) {
                            urma_seg_priv_t* np = seg_find(w->rw.dst.sge[1].tseg);
                            if (np != NULL)
                                notify_va = np->addr;
                        }
                        uint64_t notify_val = w->rw.notify_data;
                        (void)urma_seg_write_routed(0, notify_va, &notify_val, sizeof(notify_val), peer_pid);
                    }
                } else if (opc == URMA_OPC_READ) {
                    rw_kind = HCCP_SEG_RW_READ;
                    urma_target_seg_t* dst_tseg = NULL;
                    if (w->rw.dst.num_sge > 0 && w->rw.dst.sge != NULL) {
                        src = w->rw.dst.sge[0].addr;
                        len = w->rw.dst.sge[0].len;
                        dst_tseg = w->rw.dst.sge[0].tseg;
                    }
                    uint64_t remote_src_va = 0;
                    if (w->rw.src.num_sge > 0 && w->rw.src.sge != NULL) {
                        remote_src_va = w->rw.src.sge[0].addr;
                        if (remote_src_va == 0 && w->rw.src.sge[0].tseg != NULL) {
                            urma_seg_priv_t* sp = seg_find(w->rw.src.sge[0].tseg);
                            if (sp != NULL)
                                remote_src_va = sp->addr;
                        }
                    }
                    uint32_t lkey = (dst_tseg != NULL) ? (uint32_t)dst_tseg->handle : 0;
                    (void)urma_seg_read_routed(lkey, (void*)(uintptr_t)src, remote_src_va, len, peer_pid);
                } else if (opc == URMA_OPC_SEND || opc == URMA_OPC_SEND_IMM || opc == URMA_OPC_SEND_INVALIDATE) {
                    rw_kind = HCCP_SEG_RW_SEND;
                    if (w->send.src.num_sge > 0 && w->send.src.sge != NULL) {
                        src = w->send.src.sge[0].addr;
                        len = w->send.src.sge[0].len;
                    }
                }
            }

            if (rw_kind == HCCP_SEG_RW_SEND && len > 0) {
                int enq_rc = (g_urma_shm != NULL) ?
                                 hccp_pkt_enqueue(
                                     &g_urma_shm->lock, g_urma_shm->slots, URMA_RING_SLOTS, &g_urma_shm->next_seq,
                                     dst_jetty_id, src_jetty_id, (const uint8_t*)(uintptr_t)src, len) :
                                 -1;
                if (enq_rc != 0) {
                    if (wrEx->bad_wr != NULL)
                        *wrEx->bad_wr = wrEx->wr;
                    return URMA_EAGAIN;
                }
            }

            if (send_jfc != NULL) {
                urma_cr_t cr;
                memset(&cr, 0, sizeof(cr));
                cr.status = URMA_CR_SUCCESS;
                cr.opcode = hccp_seg_rw_to_cr_opcode(rw_kind);
                URMA_LOG_DEBUG(
                    "[URMA_SEG_RW] post_wr rw_kind=%s (opcode=%d)", hccp_seg_rw_name(rw_kind), (int)cr.opcode);
                cr.flag.bs.s_r = 0;
                cr.completion_len = len;
                cr.local_id = src_jetty_id;
                cr.remote_id.id = dst_jetty_id;
                jfc_push_cr(send_jfc, &cr);
            }

            if (out != NULL && out->addr != 0 && out->len >= sizeof(struct udma_u_post_info)) {
                struct udma_u_post_info* pi = (struct udma_u_post_info*)(uintptr_t)out->addr;
                pi->pi = (gp != NULL) ? (++gp->pi_counter) : 1;
                pi->ctrl = g_jetty_dummy_ctrl;
                pi->dwqe_addr = g_jetty_dummy_dwqe;
                pi->db_addr = (void volatile*)g_jetty_dummy_db;
            }
            if (wrEx->bad_wr != NULL)
                *wrEx->bad_wr = NULL;
            return URMA_SUCCESS;
        }
        case UDMA_U_USER_CTL_UPDATE_CI: {
            if (in->addr != 0 && in->len >= sizeof(struct udma_u_update_ci)) {
                struct udma_u_update_ci* ci = (struct udma_u_update_ci*)(uintptr_t)in->addr;
                if (ci->is_jetty && ci->jetty != NULL) {
                    urma_jetty_priv_t* gp = jetty_find(ci->jetty);
                    urma_jetty_drain(gp);
                }
            }
            return URMA_SUCCESS;
        }
        case UDMA_U_USER_CTL_QUERY_UE_INFO: {
            if (out != NULL && out->addr != 0 && out->len >= sizeof(struct udma_u_ue_info)) {
                struct udma_u_ue_info* ue = (struct udma_u_ue_info*)(uintptr_t)out->addr;
                ue->ue_id = 0;
                ue->chip_id = 0;
                ue->die_id = 0;
            }
            return URMA_SUCCESS;
        }
        case UDMA_U_USER_CTL_QUERY_CQE_AUX_INFO: {
            /* production decodes a cqe (status + s/r) into driver-specific aux fields;
             * the sim returns a fixed representative pair for any input, bounded by
             * the caller-provided capacity (rs presets aux_info_num to the array size) */
            if (out == NULL || out->addr == 0 || out->len < sizeof(struct udma_u_cqe_aux_info_out)) {
                return URMA_EINVAL;
            }
            struct udma_u_cqe_aux_info_out* auxOut = (struct udma_u_cqe_aux_info_out*)(uintptr_t)out->addr;
            if (auxOut->aux_info_type == NULL || auxOut->aux_info_value == NULL)
                return URMA_EINVAL;
            uint32_t cap = auxOut->aux_info_num;
            auxOut->aux_info_num = 0;
            if (cap >= 1) {
                auxOut->aux_info_type[0] = (enum udma_u_cqe_aux_info_type)0;
                auxOut->aux_info_value[0] = 0x5A5Au;
                auxOut->aux_info_num = 1;
            }
            if (cap >= 2) {
                auxOut->aux_info_type[1] = (enum udma_u_cqe_aux_info_type)1;
                auxOut->aux_info_value[1] = 0xA5A5u;
                auxOut->aux_info_num = 2;
            }
            return URMA_SUCCESS;
        }
        case UDMA_U_USER_CTL_QUERY_AE_AUX_INFO: {
            /* same shape as the cqe query, for an async event type */
            if (out == NULL || out->addr == 0 || out->len < sizeof(struct udma_u_ae_aux_info_out)) {
                return URMA_EINVAL;
            }
            struct udma_u_ae_aux_info_out* auxOut = (struct udma_u_ae_aux_info_out*)(uintptr_t)out->addr;
            if (auxOut->aux_info_type == NULL || auxOut->aux_info_value == NULL)
                return URMA_EINVAL;
            uint32_t cap = auxOut->aux_info_num;
            auxOut->aux_info_num = 0;
            if (cap >= 1) {
                auxOut->aux_info_type[0] = (enum udma_u_ae_aux_info_type)0;
                auxOut->aux_info_value[0] = 0x5A5Au;
                auxOut->aux_info_num = 1;
            }
            if (cap >= 2) {
                auxOut->aux_info_type[1] = (enum udma_u_ae_aux_info_type)1;
                auxOut->aux_info_value[1] = 0xA5A5u;
                auxOut->aux_info_num = 2;
            }
            return URMA_SUCCESS;
        }
        default:
            return ENOTSUP;
    }
}

int urma_get_tpn(urma_jetty_t* jetty)
{
    (void)jetty;
    return -1;
}

urma_net_addr_info_t* urma_get_net_addr_list(urma_context_t* ctx, uint32_t* cnt)
{
    (void)ctx;
    if (cnt)
        *cnt = 0;
    return NULL;
}

void urma_free_net_addr_list(urma_net_addr_info_t* net_addr_list) { (void)net_addr_list; }

int urma_modify_tp(
    urma_context_t* ctx, uint32_t tpn, urma_tp_cfg_t* cfg, urma_tp_attr_t* attr, urma_tp_attr_mask_t mask)
{
    (void)ctx;
    (void)tpn;
    (void)cfg;
    (void)attr;
    (void)mask;
    return 0;
}

/* ---------------------------------------------------------------- transport points
 * Production tp semantics: RaGetTpInfoList/RaCtxGetTpInfoList returns the TPs that
 * reach cfg->peer_eid from cfg->local_eid under the requested flag (ctp/rtp/utp/uboe);
 * a first query for an unseen (local, peer, flag, trans_mode) tuple creates the tp and
 * later queries return the same handle. RaSetTpAttr/RaGetTpAttr then carry a per-tp
 * attribute value table. The simulation models exactly that: a process-local tp table
 * keyed by (ctx, cfg), plus one shadow attribute value per tp handle that set stores
 * and get returns. The table lives in the device process, which is the only writer for
 * its own tps, so no cross-process sharing is needed. */
#define URMA_MAX_TPS 64

struct urma_tp_entry {
    int in_use;
    urma_context_t* ctx;
    urma_get_tp_cfg_t cfg;
    uint64_t tp_handle;
    uint8_t attr_set;
    urma_tp_attr_value_t attr;
};

static struct urma_tp_entry g_urma_tp_table[URMA_MAX_TPS];
static uint64_t g_urma_next_tp_handle = 0x1000;

static struct urma_tp_entry* urma_tp_find_or_create(const urma_context_t* ctx, const urma_get_tp_cfg_t* cfg)
{
    struct urma_tp_entry* free_slot = NULL;
    for (int i = 0; i < URMA_MAX_TPS; i++) {
        struct urma_tp_entry* e = &g_urma_tp_table[i];
        if (e->in_use) {
            if (e->ctx == ctx && e->cfg.flag.value == cfg->flag.value && e->cfg.trans_mode == cfg->trans_mode
                && memcmp(e->cfg.local_eid.raw, cfg->local_eid.raw, sizeof(cfg->local_eid.raw)) == 0
                && memcmp(e->cfg.peer_eid.raw, cfg->peer_eid.raw, sizeof(cfg->peer_eid.raw)) == 0) {
                return e;
            }
        } else if (free_slot == NULL) {
            free_slot = e;
        }
    }
    if (free_slot == NULL) {
        return NULL;
    }
    free_slot->in_use = 1;
    free_slot->ctx = (urma_context_t*)ctx;
    free_slot->cfg = *cfg;
    free_slot->tp_handle = g_urma_next_tp_handle++;
    free_slot->attr_set = 0;
    memset(&free_slot->attr, 0, sizeof(free_slot->attr));
    return free_slot;
}

static struct urma_tp_entry* urma_tp_find(const urma_context_t* ctx, uint64_t tp_handle)
{
    if (tp_handle == 0) {
        return NULL;
    }
    for (int i = 0; i < URMA_MAX_TPS; i++) {
        struct urma_tp_entry* e = &g_urma_tp_table[i];
        if (e->in_use && e->ctx == ctx && e->tp_handle == tp_handle) {
            return e;
        }
    }
    return NULL;
}

urma_status_t urma_get_tp_list(urma_context_t* ctx, urma_get_tp_cfg_t* cfg, uint32_t* tp_cnt, urma_tp_info_t* tp_list)
{
    if (ctx == NULL || cfg == NULL || tp_cnt == NULL || tp_list == NULL)
        return URMA_EINVAL;
    if (*tp_cnt == 0)
        return URMA_EINVAL;
    struct urma_tp_entry* tp = urma_tp_find_or_create(ctx, cfg);
    if (tp == NULL)
        return URMA_ENOMEM;
    tp_list[0].tp_handle = tp->tp_handle;
    *tp_cnt = 1;
    return URMA_SUCCESS;
}

urma_status_t urma_set_tp_attr(
    const urma_context_t* ctx, const uint64_t tp_handle, const uint8_t tp_attr_cnt, const uint32_t tp_attr_bitmap,
    const urma_tp_attr_value_t* tp_attr)
{
    (void)tp_attr_cnt;
    if (ctx == NULL || tp_attr == NULL)
        return URMA_EINVAL;
    struct urma_tp_entry* tp = urma_tp_find(ctx, tp_handle);
    if (tp == NULL)
        return URMA_EINVAL;
    /* store the raw value: which bitmap bit owns which field stays a caller concern */
    tp->attr = *tp_attr;
    tp->attr_set = 1;
    (void)tp_attr_bitmap;
    return URMA_SUCCESS;
}

urma_status_t urma_get_tp_attr(
    const urma_context_t* ctx, const uint64_t tp_handle, uint8_t* tp_attr_cnt, uint32_t* tp_attr_bitmap,
    urma_tp_attr_value_t* tp_attr)
{
    if (ctx == NULL || tp_attr == NULL)
        return URMA_EINVAL;
    struct urma_tp_entry* tp = urma_tp_find(ctx, tp_handle);
    if (tp == NULL)
        return URMA_EINVAL;
    *tp_attr = tp->attr;
    /* report every requested bit as carried: the shadow holds the full value */
    if (tp_attr_cnt != NULL && tp_attr_bitmap != NULL) {
        uint32_t bitmap = *tp_attr_bitmap;
        uint8_t cnt = 0;
        while (bitmap != 0) {
            bitmap &= (uint32_t)(bitmap - 1);
            cnt++;
        }
        *tp_attr_cnt = cnt;
    }
    return URMA_SUCCESS;
}

urma_status_t urma_get_eid_by_ip(const urma_context_t* ctx, const urma_net_addr_t* net_addr, urma_eid_t* eid)
{
    (void)ctx;
    if (net_addr == NULL || eid == NULL)
        return URMA_EINVAL;
    memset(eid, 0, sizeof(*eid));
    eid->in4.prefix = htonl(URMA_IPV4_MAP_IPV6_PREFIX);
    if (net_addr->sin_family == AF_INET) {
        eid->in4.addr = net_addr->in4.s_addr;
    }
    return URMA_SUCCESS;
}

urma_status_t urma_get_ip_by_eid(const urma_context_t* ctx, const urma_eid_t* eid, urma_net_addr_t* net_addr)
{
    (void)ctx;
    if (eid == NULL || net_addr == NULL)
        return URMA_EINVAL;
    memset(net_addr, 0, sizeof(*net_addr));
    net_addr->sin_family = AF_INET;
    if (eid->in4.prefix == htonl(URMA_IPV4_MAP_IPV6_PREFIX)) {
        net_addr->in4.s_addr = eid->in4.addr;
    }
    return URMA_SUCCESS;
}

urma_status_t urma_get_smac(const urma_context_t* ctx, uint8_t* mac)
{
    (void)ctx;
    if (mac == NULL)
        return URMA_EINVAL;
    memset(mac, 0, URMA_MAC_BYTES);
    mac[0] = 0x02;
    return URMA_SUCCESS;
}

urma_status_t urma_get_dmac(const urma_context_t* ctx, const urma_net_addr_t* net_addr, uint8_t* mac)
{
    (void)ctx;
    (void)net_addr;
    if (mac == NULL)
        return URMA_EINVAL;
    memset(mac, 0, URMA_MAC_BYTES);
    mac[0] = 0x02;
    mac[5] = 0x01;
    return URMA_SUCCESS;
}

void urma_u32_to_eid(uint32_t ipv4, urma_eid_t* eid)
{
    if (eid == NULL)
        return;
    memset(eid, 0, sizeof(*eid));
    eid->in4.reserved = 0;
    eid->in4.prefix = htonl(URMA_IPV4_MAP_IPV6_PREFIX);
    eid->in4.addr = htonl(ipv4);
}

int urma_str_to_eid(const char* buf, urma_eid_t* eid)
{
    if (buf == NULL || eid == NULL)
        return -EINVAL;
    memset(eid, 0, sizeof(*eid));
    struct in_addr in4;
    if (inet_pton(AF_INET, buf, &in4) == 1) {
        eid->in4.prefix = htonl(URMA_IPV4_MAP_IPV6_PREFIX);
        eid->in4.addr = in4.s_addr;
        return 0;
    }
    return -EINVAL;
}
