/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define _GNU_SOURCE

#include <ccan/list.h>
#include <infiniband/driver.h>
#include <infiniband/verbs.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <pthread.h>
#include <sys/types.h>
#include <signal.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <sys/eventfd.h>
#include <sys/uio.h>
#include <sys/prctl.h>

#include "hccp_shm_infra.h"
#include "stub_log.h"

#define IBV_DEV_NUM_MAX (1)

#define IBV_H2D_MAX_DEV 64
#define IBV_HOST_OS_MAX 4
typedef struct {
    pid_t hostPid;
    pid_t devPid;
    unsigned int hostLogicId;
    unsigned int devLogicId;
    unsigned int hostPhyId;
    unsigned int devPhyId;
    volatile unsigned int deviceReady;
} ibv_h2d_info_t;

static void ibv_stub_set_ptracer_any(void) { (void)prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0); }

#ifdef IBV_STUB_HOST
static pid_t ibv_host_get_peer_pid(void)
{
    static ibv_h2d_info_t(*h2d)[IBV_H2D_MAX_DEV] = NULL;
    static int attached = 0;
    if (!attached) {
        /* Attach to the precreated H2D registry (born-zero memfd table). */
        h2d = (ibv_h2d_info_t(*)[IBV_H2D_MAX_DEV])hccp_shm_open(
            HCCP_SHM_KEY_H2D, sizeof(ibv_h2d_info_t) * IBV_H2D_MAX_DEV * IBV_HOST_OS_MAX, 0);
        attached = 1;
    }
    if (h2d == NULL) {
        return 0;
    }
    pid_t me = getpid();
    for (int i = 0; i < IBV_H2D_MAX_DEV; i++) {
        if (h2d[0][i].hostPid == me || h2d[0][i].hostPid == 0) {
            continue;
        }
        if (kill(h2d[0][i].hostPid, 0) != 0 && errno == ESRCH) {
            continue;
        }
        return h2d[0][i].hostPid;
    }
    return 0;
}

static ssize_t ibv_pvm_xfer(int is_write, pid_t peer_pid, void* local_addr, void* remote_addr, size_t len)
{
    size_t done = 0;
    while (done < len) {
        struct iovec l = {.iov_base = (char*)local_addr + done, .iov_len = len - done};
        struct iovec r = {.iov_base = (char*)remote_addr + done, .iov_len = len - done};
        ssize_t n
            = is_write ? process_vm_writev(peer_pid, &l, 1, &r, 1, 0) : process_vm_readv(peer_pid, &l, 1, &r, 1, 0);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            ROCE_LOG_ERROR(
                "[PVM-ERR] op=%s peer=%d local=%p remote=%p off=%zu len=%zu errno=%d(%s)", is_write ? "write" : "read",
                (int)peer_pid, local_addr, remote_addr, done, len, errno, strerror(errno));
            return -1;
        }
        if (n == 0) {
            return -1;
        }
        done += (size_t)n;
    }
    return (ssize_t)done;
}

/* Notify imm encoding, mirroring WRITE_NOTIFY_OFFSET_MASK / WRITE_NOTIFY_VALUE_RECORD
 * of the rs notify path: bit 24 marks "write a record value into the peer notify
 * page", bits [23:0] select the notify unit offset inside that page. */
#define IBV_STUB_NOTIFY_OFFSET_MASK 0xffffffu
#define IBV_STUB_NOTIFY_RECORD_FLAG 0x1000000u

/* Emulate the WRITE_WITH_NOTIFY hardware behavior on the ST host data plane:
 * once the WRITE payload has landed in the peer host process, a record value is
 * written into the peer notify unit selected by the imm encoding. The peer
 * notify page base address comes from the shm registry published by the test
 * case (hccp_notify_publish). This is a simulation convention of the ST
 * environment, not a test of the hns driver. */
static void ibv_stub_notify_side_effect(const struct ibv_send_wr* wr)
{
    uint32_t imm = ntohl(wr->imm_data);
    if ((imm & IBV_STUB_NOTIFY_RECORD_FLAG) == 0) {
        return;
    }
    pid_t peer_pid = ibv_host_get_peer_pid();
    if (peer_pid <= 0) {
        return;
    }
    uint64_t peerNotifyVa = hccp_notify_query(peer_pid);
    if (peerNotifyVa == 0) {
        return;
    }
    uint64_t record = 1;
    uint64_t unit = peerNotifyVa + (uint64_t)(imm & IBV_STUB_NOTIFY_OFFSET_MASK);
    (void)ibv_pvm_xfer(1, peer_pid, &record, (void*)(uintptr_t)unit, sizeof(record));
}
#endif

#define RDMA_SHM_KEY 0x1234ABCE
#define RDMA_SHM_MAGIC 0x52444D41u
#define RDMA_RING_SLOTS 64
#define RDMA_MAX_MSG (64 * 1024)
#define RDMA_MAX_CQE 1024
#define RDMA_MAX_RECV_WR 64
#define RDMA_MAX_SGE 16
#define MAX_CQS 256

typedef struct {
    uint64_t wr_id;
    int num_sge;
    struct ibv_sge sge[RDMA_MAX_SGE];
} rdma_recv_wr_t;

struct _compat_ibv_port_attr {
    enum ibv_port_state state;
    enum ibv_mtu max_mtu;
    enum ibv_mtu active_mtu;
    int gid_tbl_len;
    uint32_t port_cap_flags;
    uint32_t max_msg_sz;
    uint32_t bad_pkey_cntr;
    uint32_t qkey_viol_cntr;
    uint16_t pkey_tbl_len;
    uint16_t lid;
    uint16_t sm_lid;
    uint8_t lmc;
    uint8_t max_vl_num;
    uint8_t sm_sl;
    uint8_t subnet_timeout;
    uint8_t init_type_reply;
    uint8_t active_width;
    uint8_t active_speed;
    uint8_t phys_state;
    uint8_t link_layer;
    uint8_t flags;
};

static struct ibv_context_ops g_ibv_context_ops;

#define MAX_QPS 128

typedef struct {
    struct ibv_qp* qp;
    int in_use;
    enum ibv_qp_state qp_state;
    enum ibv_qp_state cur_qp_state;
    uint8_t port_num;
    uint16_t pkey_index;
    uint32_t qkey;
    uint32_t rq_psn;
    uint32_t sq_psn;
    uint32_t dest_qp_num;
    enum ibv_mtu path_mtu;
    uint8_t min_rnr_timer;
    uint8_t timeout;
    uint8_t retry_cnt;
    uint8_t rnr_retry;
    uint8_t max_rd_atomic;
    uint8_t max_dest_rd_atomic;
    struct ibv_qp_cap cap;
    struct ibv_qp_init_attr init_attr;
    rdma_recv_wr_t recv_q[RDMA_MAX_RECV_WR];
    int recv_head;
    int recv_tail;
    int recv_count;
} qp_state_entry_t;

static qp_state_entry_t g_qp_table[MAX_QPS] = {0};

static qp_state_entry_t* find_qp_state(struct ibv_qp* qp)
{
    for (int i = 0; i < MAX_QPS; i++) {
        if (g_qp_table[i].in_use && g_qp_table[i].qp == qp) {
            return &g_qp_table[i];
        }
    }
    return NULL;
}

#define MAX_MRS 128

typedef struct {
    struct ibv_mr ibv_mr;
    int in_use;
    struct ibv_pd* pd;
    void* addr;
    size_t length;
    int access;
    uint32_t lkey;
    uint32_t rkey;
} mr_t;

static mr_t g_mr_table[MAX_MRS] = {0};

static mr_t* find_mock_mr(struct ibv_mr* mr)
{
    for (int i = 0; i < MAX_MRS; i++) {
        if (g_mr_table[i].in_use && &g_mr_table[i].ibv_mr == mr) {
            return &g_mr_table[i];
        }
    }
    return NULL;
}

#define MAX_MOCK_MWS 128

typedef struct {
    struct ibv_mw* mw;
    int in_use;
    int is_bound;
    struct ibv_qp* qp;
    void* addr;
    size_t length;
    int access;
    uint32_t rkey;
} mw_state_entry_t;

static mw_state_entry_t g_mw_table[MAX_MOCK_MWS] = {0};

static mw_state_entry_t* find_mw_state(struct ibv_mw* mw)
{
    for (int i = 0; i < MAX_MOCK_MWS; i++) {
        if (g_mw_table[i].in_use && g_mw_table[i].mw == mw) {
            return &g_mw_table[i];
        }
    }
    return NULL;
}

static uint32_t g_next_lkey = 1;

static mr_t* find_mr_by_lkey(uint32_t lkey)
{
    for (int i = 0; i < MAX_MRS; i++) {
        if (g_mr_table[i].in_use && g_mr_table[i].lkey == lkey) {
            return &g_mr_table[i];
        }
    }
    return NULL;
}

typedef struct {
    struct ibv_cq* cq;
    int in_use;
    struct ibv_wc wcq[RDMA_MAX_CQE];
    int head;
    int tail;
    int count;
} cq_state_t;

static cq_state_t g_cq_table[MAX_CQS] = {0};

static cq_state_t* find_cq_state(struct ibv_cq* cq)
{
    for (int i = 0; i < MAX_CQS; i++) {
        if (g_cq_table[i].in_use && g_cq_table[i].cq == cq) {
            return &g_cq_table[i];
        }
    }
    return NULL;
}

static void cq_register(struct ibv_cq* cq)
{
    if (cq == NULL)
        return;
    if (find_cq_state(cq) != NULL)
        return;
    for (int i = 0; i < MAX_CQS; i++) {
        if (!g_cq_table[i].in_use) {
            memset(&g_cq_table[i], 0, sizeof(g_cq_table[i]));
            g_cq_table[i].in_use = 1;
            g_cq_table[i].cq = cq;
            return;
        }
    }
}

static void cq_unregister(struct ibv_cq* cq)
{
    cq_state_t* cqs = find_cq_state(cq);
    if (cqs != NULL) {
        memset(cqs, 0, sizeof(*cqs));
    }
}

static void cq_push(struct ibv_cq* cq, const struct ibv_wc* wc)
{
    cq_state_t* cqs = find_cq_state(cq);
    if (cqs == NULL || wc == NULL)
        return;
    if (cqs->count >= RDMA_MAX_CQE)
        return;
    cqs->wcq[cqs->tail] = *wc;
    cqs->tail = (cqs->tail + 1) % RDMA_MAX_CQE;
    cqs->count++;
}

typedef struct {
    uint32_t magic;
    pid_t creator_pid;
    pthread_mutex_t lock;
    uint64_t next_seq;
    uint32_t next_qp_num;
    hccp_pkt_slot_t slots[RDMA_RING_SLOTS];
} rdma_shm_t;

static rdma_shm_t* g_rdma_shm = NULL;

/* Segment handle with lazy reopen: the constructor's attach is the normal
 * path (the infra precreate constructor runs first and publishes the env),
 * and a first-use retry converts a constructor-time miss -- instead of a
 * permanently degraded qp_num table fallback -- back into a working attach.
 * Benign race when several threads reopen concurrently: every racer computes
 * the same mapping. */
static rdma_shm_t* rdma_shm_get(void)
{
    if (g_rdma_shm == NULL) {
        g_rdma_shm = (rdma_shm_t*)hccp_shm_open(RDMA_SHM_KEY, sizeof(rdma_shm_t), RDMA_SHM_MAGIC);
    }
    return g_rdma_shm;
}

__attribute__((constructor)) static void rdma_shm_ctor(void)
{
    ibv_stub_set_ptracer_any();
    (void)rdma_shm_get();
}

static void rdma_drain_recv(qp_state_entry_t* entry)
{
    if (entry == NULL || entry->qp == NULL)
        return;
    uint32_t my_qpn = entry->qp->qp_num;
    struct ibv_cq* recv_cq = entry->qp->recv_cq;

    hccp_pkt_slot_t pkt;
    rdma_shm_t* shm = rdma_shm_get();
    while (entry->recv_count > 0 && shm != NULL
           && hccp_pkt_dequeue(&shm->lock, shm->slots, RDMA_RING_SLOTS, my_qpn, &pkt)) {
        rdma_recv_wr_t* r = &entry->recv_q[entry->recv_head];
        uint32_t off = 0;
        for (int i = 0; i < r->num_sge && off < pkt.byte_len; i++) {
            uint32_t copy = r->sge[i].length;
            if (copy > pkt.byte_len - off)
                copy = pkt.byte_len - off;
            if (copy > 0) {
                memcpy((void*)(uintptr_t)r->sge[i].addr, pkt.data + off, copy);
            }
            off += copy;
        }
        struct ibv_wc wc;
        memset(&wc, 0, sizeof(wc));
        wc.wr_id = r->wr_id;
        wc.status = IBV_WC_SUCCESS;
        wc.opcode = IBV_WC_RECV;
        wc.byte_len = pkt.byte_len;
        wc.qp_num = my_qpn;
        cq_push(recv_cq, &wc);

        entry->recv_head = (entry->recv_head + 1) % RDMA_MAX_RECV_WR;
        entry->recv_count--;
    }
}

static int query_device(struct ibv_context* context, struct ibv_device_attr* device_attr)
{
    if (!context || !device_attr) {
        return -1;
    }

    memset(device_attr, 0, sizeof(struct ibv_device_attr));

    device_attr->vendor_id = 0x19e5;
    device_attr->vendor_part_id = 0x0001;
    device_attr->hw_ver = 0x10;
    strncpy(device_attr->fw_ver, "24.04.0000", sizeof(device_attr->fw_ver));

    device_attr->node_guid = 0x1122334455667788;
    device_attr->sys_image_guid = 0x1122334455667788;

    device_attr->max_mr_size = 0xffffffffffffffff;
    device_attr->page_size_cap = 0xfffff000;
    device_attr->local_ca_ack_delay = 16;

    device_attr->max_qp = 16384;
    device_attr->max_qp_wr = 16384;
    device_attr->max_sge = 32;
    device_attr->max_sge_rd = 32;
    device_attr->max_cq = 8192;
    device_attr->max_cqe = 16384;
    device_attr->max_mr = 16384;
    device_attr->max_pd = 8192;
    device_attr->max_ah = 8192;

    device_attr->max_mcast_grp = 4096;
    device_attr->max_mcast_qp_attach = 16384;
    device_attr->max_total_mcast_qp_attach = 16384;

    device_attr->phys_port_cnt = 1;
    device_attr->max_pkeys = 128;
    device_attr->max_srq = 8192;
    device_attr->max_srq_wr = 16384;
    device_attr->max_srq_sge = 32;

    return 0;
}

static int query_port(struct ibv_context* context, uint8_t port_num, struct _compat_ibv_port_attr* port_attr)
{
    if (!context || !port_attr || port_num == 0) {
        errno = EINVAL;
        return -1;
    }

    memset(port_attr, 0, sizeof(*port_attr));

    port_attr->state = IBV_PORT_ACTIVE;
    port_attr->max_mtu = IBV_MTU_4096;
    port_attr->active_mtu = IBV_MTU_2048;
    port_attr->gid_tbl_len = 128;
    port_attr->port_cap_flags = IBV_PORT_IP_BASED_GIDS;
    port_attr->max_msg_sz = 1073741824U;
    port_attr->bad_pkey_cntr = 0;
    port_attr->qkey_viol_cntr = 0;
    port_attr->pkey_tbl_len = 128;
    port_attr->lid = 0x0001;
    port_attr->sm_lid = 0x0001;
    port_attr->lmc = 0;
    port_attr->max_vl_num = 1;
    port_attr->sm_sl = 0;
    port_attr->subnet_timeout = 0;
    port_attr->init_type_reply = 0;
    port_attr->active_width = 4;
    port_attr->active_speed = 1;
    port_attr->phys_state = 5;
    port_attr->link_layer = IBV_LINK_LAYER_ETHERNET;
    port_attr->flags = 0;

    return 0;
}

static void* alloc_pd(struct ibv_context* context)
{
    if (!context) {
        errno = EINVAL;
        return NULL;
    }

    struct ibv_pd* pd = (struct ibv_pd*)malloc(sizeof(struct ibv_pd));
    if (!pd) {
        errno = ENOMEM;
        return NULL;
    }
    pd->context = context;
    return pd;
}

static int dealloc_pd(struct ibv_pd* pd)
{
    free(pd);
    return 0;
}

static struct ibv_mr* reg_mr(struct ibv_pd* pd, void* addr, size_t length, int access)
{
    mr_t* mock_mr = NULL;
    for (int i = 0; i < MAX_MRS; i++) {
        if (!g_mr_table[i].in_use) {
            mock_mr = &g_mr_table[i];
            break;
        }
    }
    if (!mock_mr)
        return NULL;

    memset(mock_mr, 0, sizeof(*mock_mr));
    mock_mr->in_use = 1;
    mock_mr->pd = pd;
    mock_mr->addr = addr;
    mock_mr->length = length;
    mock_mr->access = access;
    mock_mr->lkey = g_next_lkey++;
    mock_mr->rkey = mock_mr->lkey;

    mock_mr->ibv_mr.pd = pd;
    mock_mr->ibv_mr.addr = addr;
    mock_mr->ibv_mr.length = length;
    mock_mr->ibv_mr.lkey = mock_mr->lkey;
    mock_mr->ibv_mr.rkey = mock_mr->rkey;

    return &mock_mr->ibv_mr;
}

static int rereg_mr(struct ibv_mr* mr, int flags, void* addr, size_t length, int access, struct ibv_pd* pd)
{
    if (!mr)
        return -1;
    mr_t* mock_mr = find_mock_mr(mr);
    if (!mock_mr)
        return -1;

    if (flags & IBV_REREG_MR_CHANGE_TRANSLATION) {
        mock_mr->addr = addr;
        mock_mr->length = length;
        mock_mr->ibv_mr.addr = addr;
        mock_mr->ibv_mr.length = length;
    }

    if (flags & IBV_REREG_MR_CHANGE_PD) {
        mock_mr->pd = pd;
        mock_mr->ibv_mr.pd = pd;
    }

    if (flags & IBV_REREG_MR_CHANGE_ACCESS) {
        mock_mr->access = access;
    }

    return 0;
}

static int dereg_mr(struct ibv_mr* mr)
{
    if (!mr)
        return 0;
    mr_t* mock_mr = find_mock_mr(mr);
    if (!mock_mr)
        return -1;

    memset(mock_mr, 0, sizeof(*mock_mr));
    return 0;
}

static struct ibv_mw* alloc_mw(struct ibv_pd* pd, enum ibv_mw_type type)
{
    mw_state_entry_t* entry = NULL;
    for (int i = 0; i < MAX_MOCK_MWS; i++) {
        if (!g_mw_table[i].in_use) {
            entry = &g_mw_table[i];
            break;
        }
    }
    if (!entry)
        return NULL;

    struct ibv_mw* mw = calloc(1, sizeof(struct ibv_mw));
    if (!mw)
        return NULL;

    memset(entry, 0, sizeof(*entry));
    entry->mw = mw;
    entry->in_use = 1;
    entry->is_bound = 0;
    entry->rkey = 0x12345678;

    mw->pd = pd;
    mw->type = type;
    mw->rkey = entry->rkey;

    return mw;
}

static int bind_mw(struct ibv_qp* qp, struct ibv_mw* mw, struct ibv_mw_bind* mw_bind)
{
    if (!mw || !mw_bind)
        return -1;
    mw_state_entry_t* entry = find_mw_state(mw);
    if (!entry)
        return -1;

    entry->is_bound = 1;
    entry->qp = qp;
    entry->addr = (void*)mw_bind->bind_info.addr;
    entry->length = mw_bind->bind_info.length;
    entry->access = mw_bind->bind_info.mw_access_flags;

    return 0;
}

static int dealloc_mw(struct ibv_mw* mw)
{
    if (!mw)
        return 0;
    mw_state_entry_t* entry = find_mw_state(mw);
    if (!entry)
        return -1;

    free(mw);
    memset(entry, 0, sizeof(*entry));
    return 0;
}

static struct ibv_cq*
create_cq(struct ibv_context* context, int cqe, void* cq_context, struct ibv_comp_channel* channel, int comp_vector)
{
    if (!context) {
        errno = EINVAL;
        return NULL;
    }
    struct ibv_cq* cq = calloc(1, sizeof(struct ibv_cq));
    if (!cq) {
        errno = ENOMEM;
        return NULL;
    }
    cq->context = context;
    cq->channel = channel;
    cq->cq_context = cq_context;
    cq->cqe = cqe;
    cq_register(cq);
    return cq;
}

static int poll_cq(struct ibv_cq* cq, int num_entries, struct ibv_wc* wc)
{
    if (cq == NULL || wc == NULL || num_entries <= 0)
        return 0;
    cq_state_t* cqs = find_cq_state(cq);
    if (cqs == NULL)
        return 0;

    for (int i = 0; i < MAX_QPS; i++) {
        qp_state_entry_t* e = &g_qp_table[i];
        if (e->in_use && e->qp != NULL && e->qp->recv_cq == cq) {
            rdma_drain_recv(e);
        }
    }

    int n = 0;
    while (n < num_entries && cqs->count > 0) {
        wc[n] = cqs->wcq[cqs->head];
        cqs->head = (cqs->head + 1) % RDMA_MAX_CQE;
        cqs->count--;
        n++;
    }
    return n;
}

static int req_notify_cq(struct ibv_cq* cq, int solicited_only) { return 0; }

static int cq_event(struct ibv_comp_channel* channel, struct ibv_cq** cq, void** cq_context) { return 0; }

static int resize_cq(struct ibv_cq* cq, int cqe) { return 0; }

static int destroy_cq(struct ibv_cq* cq)
{
    if (!cq)
        return 0;
    cq_unregister(cq);
    free(cq);
    return 0;
}

static struct ibv_srq* create_srq(struct ibv_pd* pd, struct ibv_srq_init_attr* attr) { return NULL; }

static int modify_srq(struct ibv_srq* srq, struct ibv_srq_attr* attr, int attr_mask) { return 0; }

static int query_srq(struct ibv_srq* srq, struct ibv_srq_attr* attr) { return 0; }

static int destroy_srq(struct ibv_srq* srq) { return 0; }

static int post_srq_recv(struct ibv_srq* srq, struct ibv_recv_wr* recv_wr, struct ibv_recv_wr** bad_recv_wr)
{
    return 0;
}

static struct ibv_qp* create_qp(struct ibv_pd* pd, struct ibv_qp_init_attr* attr)
{
    qp_state_entry_t* entry = NULL;
    for (int i = 0; i < MAX_QPS; i++) {
        if (!g_qp_table[i].in_use) {
            entry = &g_qp_table[i];
            break;
        }
    }
    if (!entry)
        return NULL;

    struct ibv_qp* qp = calloc(1, sizeof(struct ibv_qp));
    if (!qp)
        return NULL;

    memset(entry, 0, sizeof(*entry));
    entry->qp = qp;
    entry->in_use = 1;
    entry->qp_state = IBV_QPS_RESET;
    entry->cur_qp_state = IBV_QPS_RESET;

    if (attr) {
        memcpy(&entry->init_attr, attr, sizeof(*attr));
        entry->cap = attr->cap;
        qp->context = (attr->send_cq != NULL) ? attr->send_cq->context : NULL;
        qp->send_cq = attr->send_cq;
        qp->recv_cq = attr->recv_cq;
        qp->srq = attr->srq;
        qp->qp_type = attr->qp_type;
    }
    qp->pd = pd;
    if (qp->context == NULL && pd != NULL) {
        qp->context = pd->context;
    }
    qp->state = IBV_QPS_RESET;
    uint32_t uniq = 0;
    rdma_shm_t* shm = rdma_shm_get();
    if (shm != NULL) {
        uniq = hccp_alloc_unique_id(&shm->next_qp_num);
    }
    qp->qp_num = (uniq != 0) ? uniq : (uint32_t)((entry - g_qp_table) + 1);

    entry->port_num = 1;
    entry->pkey_index = 0;
    entry->qkey = 0x11111111;
    entry->rq_psn = 0;
    entry->sq_psn = 0;
    entry->dest_qp_num = 0;
    entry->path_mtu = IBV_MTU_1024;
    entry->min_rnr_timer = 12;
    entry->timeout = 14;
    entry->retry_cnt = 7;
    entry->rnr_retry = 7;
    entry->max_rd_atomic = 1;
    entry->max_dest_rd_atomic = 1;

    return qp;
}

static int query_qp(struct ibv_qp* qp, struct ibv_qp_attr* attr, int attr_mask, struct ibv_qp_init_attr* init_attr)
{
    if (!qp)
        return -1;
    qp_state_entry_t* entry = find_qp_state(qp);
    if (!entry) {
        if (attr) {
            memset(attr, 0, sizeof(*attr));
            attr->qp_state = qp->state;
            attr->cur_qp_state = qp->state;
            attr->port_num = 1;
            attr->path_mtu = IBV_MTU_1024;
        }
        if (init_attr) {
            memset(init_attr, 0, sizeof(*init_attr));
            init_attr->qp_type = qp->qp_type;
            init_attr->send_cq = qp->send_cq;
            init_attr->recv_cq = qp->recv_cq;
            init_attr->srq = qp->srq;
        }
        return 0;
    }

    if (attr) {
        memset(attr, 0, sizeof(*attr));
        if (attr_mask & IBV_QP_STATE)
            attr->qp_state = entry->qp_state;
        if (attr_mask & IBV_QP_CUR_STATE)
            attr->cur_qp_state = entry->cur_qp_state;
        if (attr_mask & IBV_QP_PORT)
            attr->port_num = entry->port_num;
        if (attr_mask & IBV_QP_PKEY_INDEX)
            attr->pkey_index = entry->pkey_index;
        if (attr_mask & IBV_QP_QKEY)
            attr->qkey = entry->qkey;
        if (attr_mask & IBV_QP_PATH_MTU)
            attr->path_mtu = entry->path_mtu;
        if (attr_mask & IBV_QP_TIMEOUT)
            attr->timeout = entry->timeout;
        if (attr_mask & IBV_QP_RETRY_CNT)
            attr->retry_cnt = entry->retry_cnt;
        if (attr_mask & IBV_QP_RNR_RETRY)
            attr->rnr_retry = entry->rnr_retry;
        if (attr_mask & IBV_QP_RQ_PSN)
            attr->rq_psn = entry->rq_psn;
        if (attr_mask & IBV_QP_MAX_QP_RD_ATOMIC)
            attr->max_rd_atomic = entry->max_rd_atomic;
        if (attr_mask & IBV_QP_MIN_RNR_TIMER)
            attr->min_rnr_timer = entry->min_rnr_timer;
        if (attr_mask & IBV_QP_SQ_PSN)
            attr->sq_psn = entry->sq_psn;
        if (attr_mask & IBV_QP_MAX_DEST_RD_ATOMIC)
            attr->max_dest_rd_atomic = entry->max_dest_rd_atomic;
        if (attr_mask & IBV_QP_CAP)
            attr->cap = entry->cap;
        if (attr_mask & IBV_QP_DEST_QPN)
            attr->dest_qp_num = entry->dest_qp_num;
    }

    if (init_attr) {
        memcpy(init_attr, &entry->init_attr, sizeof(*init_attr));
    }

    return 0;
}

static int modify_qp(struct ibv_qp* qp, struct ibv_qp_attr* attr, int attr_mask)
{
    if (!qp || !attr)
        return -1;
    qp_state_entry_t* entry = find_qp_state(qp);
    if (!entry) {
        if (attr_mask & IBV_QP_STATE) {
            qp->state = attr->qp_state;
        }
        return 0;
    }

    if (attr_mask & IBV_QP_STATE) {
        enum ibv_qp_state old_state = entry->qp_state;
        enum ibv_qp_state new_state = attr->qp_state;
        int valid = 0;

        switch (old_state) {
            case IBV_QPS_RESET:
                valid = (new_state == IBV_QPS_INIT);
                break;
            case IBV_QPS_INIT:
                valid = (new_state == IBV_QPS_RTR || new_state == IBV_QPS_RESET);
                break;
            case IBV_QPS_RTR:
                valid = (new_state == IBV_QPS_RTS || new_state == IBV_QPS_INIT || new_state == IBV_QPS_RESET);
                break;
            case IBV_QPS_RTS:
                valid
                    = (new_state == IBV_QPS_SQD || new_state == IBV_QPS_ERR || new_state == IBV_QPS_RTR
                       || new_state == IBV_QPS_RESET);
                break;
            default:
                valid = 1;
                break;
        }

        if (!valid)
            return -1;
        entry->qp_state = new_state;
        entry->cur_qp_state = new_state;
    }

    if (attr_mask & IBV_QP_CUR_STATE)
        entry->cur_qp_state = attr->cur_qp_state;
    if (attr_mask & IBV_QP_PORT)
        entry->port_num = attr->port_num;
    if (attr_mask & IBV_QP_PKEY_INDEX)
        entry->pkey_index = attr->pkey_index;
    if (attr_mask & IBV_QP_QKEY)
        entry->qkey = attr->qkey;
    if (attr_mask & IBV_QP_PATH_MTU)
        entry->path_mtu = attr->path_mtu;
    if (attr_mask & IBV_QP_TIMEOUT)
        entry->timeout = attr->timeout;
    if (attr_mask & IBV_QP_RETRY_CNT)
        entry->retry_cnt = attr->retry_cnt;
    if (attr_mask & IBV_QP_RNR_RETRY)
        entry->rnr_retry = attr->rnr_retry;
    if (attr_mask & IBV_QP_RQ_PSN)
        entry->rq_psn = attr->rq_psn;
    if (attr_mask & IBV_QP_MAX_QP_RD_ATOMIC)
        entry->max_rd_atomic = attr->max_rd_atomic;
    if (attr_mask & IBV_QP_MIN_RNR_TIMER)
        entry->min_rnr_timer = attr->min_rnr_timer;
    if (attr_mask & IBV_QP_SQ_PSN)
        entry->sq_psn = attr->sq_psn;
    if (attr_mask & IBV_QP_MAX_DEST_RD_ATOMIC)
        entry->max_dest_rd_atomic = attr->max_dest_rd_atomic;
    if (attr_mask & IBV_QP_CAP)
        entry->cap = attr->cap;
    if (attr_mask & IBV_QP_DEST_QPN)
        entry->dest_qp_num = attr->dest_qp_num;

    return 0;
}

static int destroy_qp(struct ibv_qp* qp)
{
    if (!qp)
        return 0;
    qp_state_entry_t* entry = find_qp_state(qp);
    if (!entry) {
        free(qp);
        return 0;
    }

    free(qp);
    memset(entry, 0, sizeof(*entry));
    return 0;
}

static int post_send(struct ibv_qp* qp, struct ibv_send_wr* wr, struct ibv_send_wr** bad_wr)
{
    if (qp == NULL || wr == NULL) {
        if (bad_wr)
            *bad_wr = wr;
        return -1;
    }

    qp_state_entry_t* entry = find_qp_state(qp);
    if (entry == NULL) {
        if (bad_wr)
            *bad_wr = NULL;
        return 0;
    }

    if (entry->qp_state != IBV_QPS_RTS) {
        if (bad_wr)
            *bad_wr = wr;
        return -1;
    }

    struct ibv_send_wr* cur = wr;
    while (cur != NULL) {
        enum ibv_wr_opcode opc = cur->opcode;
        uint32_t total = 0;

        if (opc == IBV_WR_RDMA_WRITE || opc == IBV_WR_RDMA_WRITE_WITH_IMM) {
            uint64_t remote_addr = cur->wr.rdma.remote_addr;
            if (cur->num_sge > 0) {
                struct ibv_sge* sge = &cur->sg_list[0];
                mr_t* mr = find_mr_by_lkey(sge->lkey);
                uint64_t src_addr = (mr != NULL) ? (uint64_t)(uintptr_t)mr->addr : sge->addr;
                uint32_t copy_len = sge->length;
                if (copy_len > RDMA_MAX_MSG)
                    copy_len = RDMA_MAX_MSG;
                if (copy_len > 0 && src_addr != 0 && remote_addr != 0) {
#ifdef IBV_STUB_HOST
                    pid_t peer_pid = ibv_host_get_peer_pid();
                    if (getenv("W2_PROBE") != NULL) {
                        static int probed = 0;
                        if (!probed) {
                            ROCE_LOG_DEBUG(
                                "[W2_PROBE] post_send WRITE pid=%d (malloc/host pid) peer_pid=%d len=%u", getpid(),
                                (int)peer_pid, copy_len);
                            probed = 1;
                        }
                    }
                    ssize_t n = (peer_pid > 0) ? ibv_pvm_xfer(
                                                     1, peer_pid, (void*)(uintptr_t)src_addr,
                                                     (void*)(uintptr_t)remote_addr, copy_len) :
                                                 -1;
                    if (n < 0) {
                        if (bad_wr)
                            *bad_wr = cur;
                        return -1;
                    }
                    total = (uint32_t)n;
#else
                    memcpy((void*)(uintptr_t)remote_addr, (void*)(uintptr_t)src_addr, copy_len);
                    total = copy_len;
#endif
                }
            }
#ifdef IBV_STUB_HOST
            if (opc == IBV_WR_RDMA_WRITE_WITH_IMM) {
                ibv_stub_notify_side_effect(cur);
            }
#endif
            struct ibv_wc wc;
            memset(&wc, 0, sizeof(wc));
            wc.wr_id = cur->wr_id;
            wc.status = IBV_WC_SUCCESS;
            wc.opcode = IBV_WC_RDMA_WRITE;
            wc.byte_len = total;
            wc.qp_num = qp->qp_num;
            cq_push(qp->send_cq, &wc);
        } else if (opc == IBV_WR_RDMA_READ) {
            uint64_t remote_addr = cur->wr.rdma.remote_addr;
            if (cur->num_sge > 0) {
                struct ibv_sge* sge = &cur->sg_list[0];
                mr_t* mr = find_mr_by_lkey(sge->lkey);
                uint64_t dst_addr = (mr != NULL) ? (uint64_t)(uintptr_t)mr->addr : sge->addr;
                uint32_t copy_len = sge->length;
                if (copy_len > RDMA_MAX_MSG)
                    copy_len = RDMA_MAX_MSG;
                if (copy_len > 0 && dst_addr != 0 && remote_addr != 0) {
#ifdef IBV_STUB_HOST
                    pid_t peer_pid = ibv_host_get_peer_pid();
                    if (getenv("W2_PROBE") != NULL) {
                        static int probed_r = 0;
                        if (!probed_r) {
                            ROCE_LOG_DEBUG(
                                "[W2_PROBE] post_send READ pid=%d (malloc/host pid) peer_pid=%d len=%u", getpid(),
                                (int)peer_pid, copy_len);
                            probed_r = 1;
                        }
                    }
                    ssize_t n = (peer_pid > 0) ? ibv_pvm_xfer(
                                                     0, peer_pid, (void*)(uintptr_t)dst_addr,
                                                     (void*)(uintptr_t)remote_addr, copy_len) :
                                                 -1;
                    if (n < 0) {
                        if (bad_wr)
                            *bad_wr = cur;
                        return -1;
                    }
                    total = (uint32_t)n;
#else
                    memcpy((void*)(uintptr_t)dst_addr, (void*)(uintptr_t)remote_addr, copy_len);
                    total = copy_len;
#endif
                }
            }
            struct ibv_wc wc;
            memset(&wc, 0, sizeof(wc));
            wc.wr_id = cur->wr_id;
            wc.status = IBV_WC_SUCCESS;
            wc.opcode = IBV_WC_RDMA_READ;
            wc.byte_len = total;
            wc.qp_num = qp->qp_num;
            cq_push(qp->send_cq, &wc);
        } else if (opc == IBV_WR_SEND || opc == IBV_WR_SEND_WITH_IMM || opc == IBV_WR_SEND_WITH_INV) {
            uint8_t buf[RDMA_MAX_MSG];
            int nsge = cur->num_sge;
            if (nsge > RDMA_MAX_SGE)
                nsge = RDMA_MAX_SGE;
            for (int i = 0; i < nsge && total < RDMA_MAX_MSG; i++) {
                struct ibv_sge* sge = &cur->sg_list[i];
                (void)find_mr_by_lkey(sge->lkey);
                uint32_t copy_len = sge->length;
                if (copy_len > RDMA_MAX_MSG - total)
                    copy_len = RDMA_MAX_MSG - total;
                if (copy_len > 0 && sge->addr != 0) {
                    memcpy(buf + total, (void*)(uintptr_t)sge->addr, copy_len);
                    total += copy_len;
                }
            }

            rdma_shm_t* shm = rdma_shm_get();
            if (entry->dest_qp_num != 0 && shm != NULL) {
                (void)hccp_pkt_enqueue(
                    &shm->lock, shm->slots, RDMA_RING_SLOTS, &shm->next_seq, entry->dest_qp_num, qp->qp_num, buf,
                    total);
            }

            struct ibv_wc wc;
            memset(&wc, 0, sizeof(wc));
            wc.wr_id = cur->wr_id;
            wc.status = IBV_WC_SUCCESS;
            wc.opcode = IBV_WC_SEND;
            wc.byte_len = total;
            wc.qp_num = qp->qp_num;
            cq_push(qp->send_cq, &wc);
        }

        cur = cur->next;
    }

    if (bad_wr)
        *bad_wr = NULL;
    return 0;
}

static int post_recv(struct ibv_qp* qp, struct ibv_recv_wr* wr, struct ibv_recv_wr** bad_wr)
{
    if (qp == NULL || wr == NULL) {
        if (bad_wr)
            *bad_wr = wr;
        return -1;
    }

    qp_state_entry_t* entry = find_qp_state(qp);
    if (entry == NULL) {
        if (bad_wr)
            *bad_wr = NULL;
        return 0;
    }

    struct ibv_recv_wr* cur = wr;
    while (cur != NULL) {
        if (entry->recv_count >= RDMA_MAX_RECV_WR) {
            if (bad_wr)
                *bad_wr = cur;
            return -1;
        }
        rdma_recv_wr_t* r = &entry->recv_q[entry->recv_tail];
        r->wr_id = cur->wr_id;
        int nsge = cur->num_sge;
        if (nsge > RDMA_MAX_SGE)
            nsge = RDMA_MAX_SGE;
        r->num_sge = nsge;
        for (int i = 0; i < nsge; i++) {
            r->sge[i] = cur->sg_list[i];
        }
        entry->recv_tail = (entry->recv_tail + 1) % RDMA_MAX_RECV_WR;
        entry->recv_count++;
        cur = cur->next;
    }

    rdma_drain_recv(entry);

    if (bad_wr)
        *bad_wr = NULL;
    return 0;
}

static struct ibv_ah* create_ah(struct ibv_pd* pd, struct ibv_ah_attr* attr)
{
    if (!pd) {
        errno = EINVAL;
        return NULL;
    }
    struct ibv_ah* ah = calloc(1, sizeof(struct ibv_ah));
    if (!ah) {
        errno = ENOMEM;
        return NULL;
    }
    ah->pd = pd;
    ah->context = (pd->context != NULL) ? pd->context : NULL;
    return ah;
}

static int destroy_ah(struct ibv_ah* ah)
{
    if (!ah)
        return 0;
    free(ah);
    return 0;
}

static int attach_mcast(struct ibv_qp* qp, const union ibv_gid* gid, uint16_t lid) { return 0; }

static int detach_mcast(struct ibv_qp* qp, const union ibv_gid* gid, uint16_t lid) { return 0; }

static int async_event(struct ibv_async_event* event) { return 0; }

static void __attribute__((constructor)) init_dev_list(void)
{
    ibv_stub_set_ptracer_any();
    g_ibv_context_ops._compat_query_device = query_device;
    g_ibv_context_ops._compat_query_port = query_port;
    g_ibv_context_ops._compat_alloc_pd = (void* (*)(void))alloc_pd;
    g_ibv_context_ops._compat_dealloc_pd = (void* (*)(void))dealloc_pd;
    g_ibv_context_ops._compat_reg_mr = (void* (*)(void))reg_mr;
    g_ibv_context_ops._compat_rereg_mr = (void* (*)(void))rereg_mr;
    g_ibv_context_ops._compat_dereg_mr = (void* (*)(void))dereg_mr;
    g_ibv_context_ops.alloc_mw = alloc_mw;
    g_ibv_context_ops.bind_mw = bind_mw;
    g_ibv_context_ops.dealloc_mw = dealloc_mw;
    g_ibv_context_ops._compat_create_cq = (void* (*)(void))create_cq;
    g_ibv_context_ops.poll_cq = poll_cq;
    g_ibv_context_ops.req_notify_cq = req_notify_cq;
    g_ibv_context_ops._compat_cq_event = (void* (*)(void))cq_event;
    g_ibv_context_ops._compat_resize_cq = (void* (*)(void))resize_cq;
    g_ibv_context_ops._compat_destroy_cq = (void* (*)(void))destroy_cq;
    g_ibv_context_ops._compat_create_srq = (void* (*)(void))create_srq;
    g_ibv_context_ops._compat_modify_srq = (void* (*)(void))modify_srq;
    g_ibv_context_ops._compat_query_srq = (void* (*)(void))query_srq;
    g_ibv_context_ops._compat_destroy_srq = (void* (*)(void))destroy_srq;
    g_ibv_context_ops.post_srq_recv = post_srq_recv;
    g_ibv_context_ops._compat_create_qp = (void* (*)(void))create_qp;
    g_ibv_context_ops._compat_query_qp = (void* (*)(void))query_qp;
    g_ibv_context_ops._compat_modify_qp = (void* (*)(void))modify_qp;
    g_ibv_context_ops._compat_destroy_qp = (void* (*)(void))destroy_qp;
    g_ibv_context_ops.post_send = post_send;
    g_ibv_context_ops.post_recv = post_recv;
    g_ibv_context_ops._compat_create_ah = (void* (*)(void))create_ah;
    g_ibv_context_ops._compat_destroy_ah = (void* (*)(void))destroy_ah;
    g_ibv_context_ops._compat_attach_mcast = (void* (*)(void))attach_mcast;
    g_ibv_context_ops._compat_detach_mcast = (void* (*)(void))detach_mcast;
    g_ibv_context_ops._compat_async_event = (void* (*)(void))async_event;
}
int ibv_query_device(struct ibv_context* context, struct ibv_device_attr* device_attr)
{
    return query_device(context, device_attr);
};

#undef ibv_query_port
int ibv_query_port(struct ibv_context* context, uint8_t port_num, struct _compat_ibv_port_attr* port_attr)
{
    return query_port(context, port_num, port_attr);
}

static unsigned int ibv_host_get_phyid(void)
{
    static ibv_h2d_info_t(*h2d)[IBV_H2D_MAX_DEV] = NULL;
    static int attached = 0;
    if (!attached) {
        /* Attach to the precreated H2D registry (born-zero memfd table). */
        h2d = (ibv_h2d_info_t(*)[IBV_H2D_MAX_DEV])hccp_shm_open(
            HCCP_SHM_KEY_H2D, sizeof(ibv_h2d_info_t) * IBV_H2D_MAX_DEV * IBV_HOST_OS_MAX, 0);
        attached = 1;
    }
    if (h2d == NULL) {
        return 0;
    }
    pid_t me = getpid();
    for (int i = 0; i < IBV_H2D_MAX_DEV; i++) {
        if (h2d[0][i].hostPid == me || h2d[0][i].devPid == me) {
            return h2d[0][i].hostPhyId;
        }
    }
    return 0;
}

int ibv_query_gid(struct ibv_context* context, uint8_t port_num, int index, union ibv_gid* gid)
{
    if (!context || !gid || port_num < 1) {
        errno = EINVAL;
        return -1;
    }

    uint32_t* p = (uint32_t*)gid;
    p[0] = 0;
    p[1] = 0;
    p[2] = htonl(0x0000FFFF);
    unsigned int phyid = ibv_host_get_phyid();
    unsigned int iface = (unsigned int)index % 3u;
    if (phyid != 0) {
        p[3] = htonl((127u << 24) | ((uint32_t)phyid << 8) | iface);
    } else {
        p[3] = htonl((127u << 24) | ((uint32_t)index << 8) | iface);
    }

    return 0;
}

struct ibv_pd* ibv_alloc_pd(struct ibv_context* context) { return alloc_pd(context); }

int ibv_dealloc_pd(struct ibv_pd* pd)
{
    free(pd);
    return 0;
};

struct ibv_cq*
ibv_create_cq(struct ibv_context* context, int cqe, void* cq_context, struct ibv_comp_channel* channel, int comp_vector)
{
    return create_cq(context, cqe, cq_context, channel, comp_vector);
};

int ibv_destroy_cq(struct ibv_cq* cq) { return destroy_cq(cq); };

struct ibv_comp_channel* ibv_create_comp_channel(struct ibv_context* context)
{
    if (!context) {
        errno = EINVAL;
        return NULL;
    }

    struct ibv_comp_channel* channel = (struct ibv_comp_channel*)calloc(1, sizeof(struct ibv_comp_channel));
    if (!channel) {
        errno = ENOMEM;
        return NULL;
    }

    channel->context = context;

    int fd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    if (fd < 0) {
        free(channel);
        errno = ENOMEM;
        return NULL;
    }
    channel->fd = fd;

    return channel;
};

int ibv_destroy_comp_channel(struct ibv_comp_channel* channel)
{
    if (!channel)
        return 0;
    if (channel->fd > 0) {
        close(channel->fd);
    }
    free(channel);
    return 0;
};

struct ibv_srq* ibv_create_srq(struct ibv_pd* pd, struct ibv_srq_init_attr* srqInitAttr) { return NULL; };

int ibv_destroy_srq(struct ibv_srq* srq) { return 0; };

int ibv_query_gid_type(
    struct ibv_context* context, uint8_t port_num, unsigned int index, enum ibv_gid_type_sysfs* gid_type)
{
    *gid_type = IBV_GID_TYPE_SYSFS_ROCE_V2;
    return 0;
};

struct ibv_ah* ibv_create_ah(struct ibv_pd* pd, struct ibv_ah_attr* attr) { return create_ah(pd, attr); };

int ibv_destroy_ah(struct ibv_ah* ah) { return destroy_ah(ah); };

int ibv_query_qp(struct ibv_qp* qp, struct ibv_qp_attr* attr, int attr_mask, struct ibv_qp_init_attr* init_attr)
{
    return query_qp(qp, attr, attr_mask, init_attr);
}

int ibv_get_cq_event(struct ibv_comp_channel* channel, struct ibv_cq** cq, void** cq_context) { return 0; };

void ibv_ack_cq_events(struct ibv_cq* cq, unsigned int num_events) { return; };

int ibv_modify_qp(struct ibv_qp* qp, struct ibv_qp_attr* attr, int attr_mask) { return modify_qp(qp, attr, attr_mask); }

#undef ibv_reg_mr
struct ibv_mr* ibv_reg_mr(struct ibv_pd* pd, void* addr, size_t length, int access)
{
    return reg_mr(pd, addr, length, access);
};

int ibv_dereg_mr(struct ibv_mr* mr) { return dereg_mr(mr); };

struct ibv_qp* ibv_create_qp(struct ibv_pd* pd, struct ibv_qp_init_attr* init_attr) { return create_qp(pd, init_attr); }

int ibv_destroy_qp(struct ibv_qp* qp) { return destroy_qp(qp); }

struct ibv_device** ibv_get_device_list(int* num)
{
    static struct ibv_device** g_dev_list = NULL;
    if (g_dev_list == NULL) {
        g_dev_list = calloc(IBV_DEV_NUM_MAX + 1, sizeof(struct ibv_device*));
    }
    *num = IBV_DEV_NUM_MAX;
    for (int i = 0; i < IBV_DEV_NUM_MAX; i++) {
        if (g_dev_list[i] != NULL)
            continue;
        g_dev_list[i] = calloc(1, sizeof(struct ibv_device));
        g_dev_list[i]->node_type = IBV_NODE_RNIC;
        g_dev_list[i]->transport_type = IBV_TRANSPORT_IB;
        sprintf(g_dev_list[i]->dev_name, "uverbs%d", i);
        sprintf(g_dev_list[i]->name, "rdma%d", i);
        sprintf(g_dev_list[i]->dev_path, "/sys/class/infiniband_verbs/mock%d", i);
        sprintf(g_dev_list[i]->ibdev_path, "/sys/class/infiniband/mock%d", i);
    }
    return g_dev_list;
};

void ibv_free_device_list(struct ibv_device** list)
{
    if (!list)
        return;

    for (int i = 0; list[i] != NULL; i++) {
        free(list[i]);
    }
    free(list);
    return;
}

const char* ibv_get_device_name(struct ibv_device* device)
{
    if (device == NULL) {
        return NULL;
    }
    return device->name;
};

int ibv_close_device(struct ibv_context* context)
{
    free(context);
    return 0;
};

struct ibv_context* ibv_open_device(struct ibv_device* device)
{
    struct ibv_context* ctx = calloc(1, sizeof(struct ibv_context));
    ctx->device = device;
    ctx->ops = g_ibv_context_ops;
    return ctx;
}

const char* ibv_wc_status_str(enum ibv_wc_status status) { return NULL; };
