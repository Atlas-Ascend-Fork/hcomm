/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include "rdma_lite.h"
#include "ascend_hal.h"
#include "hccp_shm_infra.h"

#define RLITE_POOL_MAX 8
#define RLITE_POOL_BUF_ALIGN 256

typedef struct {
    int in_use;
    u64 dva;
    void* hva;
    void* winBase;
    u64 winLen;
    u32 len;
    u32 memIdx;
    u32 offset;
} rlite_pool_t;

static rlite_pool_t g_pool_tab[RLITE_POOL_MAX];
#include <infiniband/verbs.h>

/* Dual-host QP rendezvous over the precreated LITE memfd segment (layout
 * hccp_lite_seg_t). The old inline SysV replica -- creator kill-probe without
 * errno check, a 5s magic spin, an IPC_EXCL fallback race and a non-robust
 * lock -- is gone with it: the infra constructor creates and publishes the
 * segment before any fork, so attaching here is a pure env lookup, and a
 * NULL result is a visibly broken world (never a reason to re-create).
 * The lock is robust because this is its only lock site: attribute and site
 * flipped in one batch (see hccp_shm_infra.c). */
static hccp_lite_seg_t* g_rendezvous = NULL;

static void rdma_lite_rendezvous_init(void)
{
    g_rendezvous = (hccp_lite_seg_t*)hccp_shm_open(HCCP_SHM_KEY_LITE, HCCP_SHM_SIZE_LITE, HCCP_SHM_MAGIC_LITE);
}

__attribute__((constructor)) static void rdma_lite_rendezvous_ctor(void) { rdma_lite_rendezvous_init(); }

static uint32_t rdma_lite_rendezvous_pair(uint32_t my_qp_num)
{
    if (g_rendezvous == NULL)
        return 0;
    if (hccp_shm_lock_acquire(&g_rendezvous->lock) != 0)
        return 0; /* segment not recoverable */
    uint32_t idx = g_rendezvous->count++;
    if (idx >= HCCP_LITE_MAX_HOST) {
        (void)hccp_shm_lock_release(&g_rendezvous->lock);
        return 0;
    }
    g_rendezvous->qp_num[idx] = my_qp_num;
    __sync_synchronize();
    uint32_t peer_idx = (idx == 0) ? 1 : 0;
    (void)hccp_shm_lock_release(&g_rendezvous->lock);

    for (int spin = 0; spin < 10000; spin++) {
        uint32_t peer = g_rendezvous->qp_num[peer_idx];
        if (peer != 0) {
            return peer;
        }
        usleep(1000);
    }
    return 0;
}

#define RLITE_MAX_CTX 4
#define RLITE_MAX_CQ 16
#define RLITE_MAX_QP 16

typedef struct {
    struct rdma_lite_context* lite_ctx;
    struct ibv_context* ibv_ctx;
    struct ibv_pd* ibv_pd;
    int in_use;
} rlite_ctx_entry_t;

typedef struct {
    struct rdma_lite_cq* lite_cq;
    struct ibv_cq* ibv_cq;
    int in_use;
} rlite_cq_entry_t;

typedef struct {
    struct rdma_lite_qp* lite_qp;
    struct ibv_qp* ibv_qp;
    int in_use;
} rlite_qp_entry_t;

static rlite_ctx_entry_t g_ctx_tab[RLITE_MAX_CTX];
static rlite_cq_entry_t g_cq_tab[RLITE_MAX_CQ];
static rlite_qp_entry_t g_qp_tab[RLITE_MAX_QP];

static struct ibv_pd* rlite_find_pd(struct rdma_lite_context* lite_ctx)
{
    for (int i = 0; i < RLITE_MAX_CTX; i++) {
        if (g_ctx_tab[i].in_use && g_ctx_tab[i].lite_ctx == lite_ctx) {
            return g_ctx_tab[i].ibv_pd;
        }
    }
    return NULL;
}

static struct ibv_cq* rlite_find_ibv_cq(struct rdma_lite_cq* lite_cq)
{
    for (int i = 0; i < RLITE_MAX_CQ; i++) {
        if (g_cq_tab[i].in_use && g_cq_tab[i].lite_cq == lite_cq) {
            return g_cq_tab[i].ibv_cq;
        }
    }
    return NULL;
}

static struct ibv_qp* rlite_find_ibv_qp(struct rdma_lite_qp* lite_qp)
{
    for (int i = 0; i < RLITE_MAX_QP; i++) {
        if (g_qp_tab[i].in_use && g_qp_tab[i].lite_qp == lite_qp) {
            return g_qp_tab[i].ibv_qp;
        }
    }
    return NULL;
}

struct rdma_lite_context* rdma_lite_alloc_context(u8 phyId, struct dev_cap_info* cap)
{
    (void)phyId;
    (void)cap;
    int num = 0;
    struct ibv_device** dev_list = ibv_get_device_list(&num);
    if (dev_list == NULL || num <= 0) {
        if (dev_list)
            ibv_free_device_list(dev_list);
        return NULL;
    }
    struct ibv_context* ibv_ctx = ibv_open_device(dev_list[0]);
    ibv_free_device_list(dev_list);
    if (ibv_ctx == NULL)
        return NULL;

    struct ibv_pd* pd = ibv_alloc_pd(ibv_ctx);
    if (pd == NULL) {
        ibv_close_device(ibv_ctx);
        return NULL;
    }

    struct rdma_lite_context* lite_ctx = (struct rdma_lite_context*)calloc(1, sizeof(struct rdma_lite_context));
    if (lite_ctx == NULL) {
        ibv_dealloc_pd(pd);
        ibv_close_device(ibv_ctx);
        return NULL;
    }

    for (int i = 0; i < RLITE_MAX_CTX; i++) {
        if (!g_ctx_tab[i].in_use) {
            g_ctx_tab[i].lite_ctx = lite_ctx;
            g_ctx_tab[i].ibv_ctx = ibv_ctx;
            g_ctx_tab[i].ibv_pd = pd;
            g_ctx_tab[i].in_use = 1;
            break;
        }
    }
    return lite_ctx;
}

void rdma_lite_free_context(struct rdma_lite_context* lite_ctx)
{
    if (lite_ctx == NULL)
        return;
    for (int i = 0; i < RLITE_MAX_CTX; i++) {
        if (g_ctx_tab[i].in_use && g_ctx_tab[i].lite_ctx == lite_ctx) {
            if (g_ctx_tab[i].ibv_pd)
                ibv_dealloc_pd(g_ctx_tab[i].ibv_pd);
            if (g_ctx_tab[i].ibv_ctx)
                ibv_close_device(g_ctx_tab[i].ibv_ctx);
            memset(&g_ctx_tab[i], 0, sizeof(g_ctx_tab[i]));
            break;
        }
    }
    free(lite_ctx);
}

int rdma_lite_init_mem_pool(struct rdma_lite_context* lite_ctx, struct rdma_lite_mem_attr* lite_mem_attr)
{
    (void)lite_ctx;
    if (lite_mem_attr == NULL || lite_mem_attr->mem_data.va == 0 || lite_mem_attr->mem_data.mem_size == 0) {
        return -1;
    }
    int slot = -1;
    for (int i = 0; i < RLITE_POOL_MAX; i++) {
        if (!g_pool_tab[i].in_use) {
            slot = i;
            break;
        }
    }
    if (slot < 0) {
        return -1;
    }
    struct AddrMapInPara mapIn = {0};
    struct AddrMapOutPara mapOut = {0};
    size_t outLen = 0;
    mapIn.addr_type = DEV_MEM_MAP_HOST;
    mapIn.devid = 0;
    if (halMemCtl(CTRL_TYPE_ADDR_MAP, &mapIn, sizeof(mapIn), &mapOut, &outLen) != DRV_ERROR_NONE) {
        return -1;
    }
    char* hva
        = (char*)(uintptr_t)mapOut.ptr + (uintptr_t)(lite_mem_attr->mem_data.va - (u64)(uintptr_t)HCCP_DEVMEM_DVA_ADDR);
    g_pool_tab[slot].in_use = 1;
    g_pool_tab[slot].dva = lite_mem_attr->mem_data.va;
    g_pool_tab[slot].hva = hva;
    g_pool_tab[slot].winBase = (void*)(uintptr_t)mapOut.ptr;
    g_pool_tab[slot].winLen = mapOut.len;
    g_pool_tab[slot].len = lite_mem_attr->mem_data.mem_size;
    g_pool_tab[slot].memIdx = lite_mem_attr->mem_data.mem_idx;
    g_pool_tab[slot].offset = 0;
    return 0;
}

int rdma_lite_deinit_mem_pool(struct rdma_lite_context* lite_ctx, u32 mem_idx)
{
    (void)lite_ctx;
    for (int i = 0; i < RLITE_POOL_MAX; i++) {
        if (g_pool_tab[i].in_use && g_pool_tab[i].memIdx == mem_idx) {
            struct AddrUnmapInPara unmapIn = {0};
            unmapIn.addr_type = DEV_MEM_MAP_HOST;
            unmapIn.ptr = (unsigned long long)(uintptr_t)g_pool_tab[i].winBase;
            unmapIn.len = g_pool_tab[i].winLen;
            (void)halMemCtl(CTRL_TYPE_ADDR_UNMAP, &unmapIn, sizeof(unmapIn), NULL, NULL);
            memset(&g_pool_tab[i], 0, sizeof(g_pool_tab[i]));
            break;
        }
    }
    return 0;
}

int rdma_lite_stub_alloc_buf(u32 size, struct rdma_lite_host_buf* out)
{
    if (out == NULL || size == 0) {
        return -1;
    }
    u32 aligned = (size + RLITE_POOL_BUF_ALIGN - 1) & ~((u32)RLITE_POOL_BUF_ALIGN - 1);
    for (int i = 0; i < RLITE_POOL_MAX; i++) {
        if (!g_pool_tab[i].in_use) {
            continue;
        }
        if ((u64)g_pool_tab[i].offset + aligned > (u64)g_pool_tab[i].len) {
            continue;
        }
        out->dva = g_pool_tab[i].dva + g_pool_tab[i].offset;
        out->hva = (char*)g_pool_tab[i].hva + g_pool_tab[i].offset;
        out->length = size;
        out->mem_idx = g_pool_tab[i].memIdx;
        g_pool_tab[i].offset += aligned;
        return 0;
    }
    return -1;
}

struct rdma_lite_cq* rdma_lite_create_cq(struct rdma_lite_context* lite_ctx, struct rdma_lite_cq_attr* lite_cq_attr)
{
    (void)lite_cq_attr;
    struct ibv_pd* pd = rlite_find_pd(lite_ctx);
    if (pd == NULL || pd->context == NULL)
        return NULL;
    struct ibv_cq* ibv_cq = ibv_create_cq(pd->context, 64, NULL, NULL, 0);
    if (ibv_cq == NULL)
        return NULL;

    struct rdma_lite_cq* lite_cq = (struct rdma_lite_cq*)calloc(1, sizeof(struct rdma_lite_cq));
    if (lite_cq == NULL) {
        ibv_destroy_cq(ibv_cq);
        return NULL;
    }
    lite_cq->ctx = lite_ctx;

    for (int i = 0; i < RLITE_MAX_CQ; i++) {
        if (!g_cq_tab[i].in_use) {
            g_cq_tab[i].lite_cq = lite_cq;
            g_cq_tab[i].ibv_cq = ibv_cq;
            g_cq_tab[i].in_use = 1;
            break;
        }
    }
    return lite_cq;
}

int rdma_lite_destroy_cq(struct rdma_lite_cq* lite_cq)
{
    if (lite_cq == NULL)
        return 0;
    for (int i = 0; i < RLITE_MAX_CQ; i++) {
        if (g_cq_tab[i].in_use && g_cq_tab[i].lite_cq == lite_cq) {
            if (g_cq_tab[i].ibv_cq)
                ibv_destroy_cq(g_cq_tab[i].ibv_cq);
            memset(&g_cq_tab[i], 0, sizeof(g_cq_tab[i]));
            break;
        }
    }
    free(lite_cq);
    return 0;
}

static int rlite_bringup_qp(struct ibv_qp* qp, uint32_t dest_qp_num)
{
    struct ibv_qp_attr attr;
    int ret;

    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_INIT;
    attr.port_num = 1;
    attr.pkey_index = 0;
    ret = ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_PORT | IBV_QP_PKEY_INDEX);
    if (ret)
        return ret;

    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.dest_qp_num = dest_qp_num;
    ret = ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_DEST_QPN);
    if (ret)
        return ret;

    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    ret = ibv_modify_qp(qp, &attr, IBV_QP_STATE);
    if (ret)
        return ret;

    return 0;
}

struct rdma_lite_qp* rdma_lite_create_qp(struct rdma_lite_context* lite_ctx, struct rdma_lite_qp_attr* lite_qp_attr)
{
    struct ibv_pd* pd = rlite_find_pd(lite_ctx);
    if (pd == NULL || lite_qp_attr == NULL)
        return NULL;

    struct ibv_cq* send_cq = rlite_find_ibv_cq(lite_qp_attr->send_cq);
    struct ibv_cq* recv_cq = rlite_find_ibv_cq(lite_qp_attr->recv_cq);
    if (send_cq == NULL || recv_cq == NULL)
        return NULL;

    struct ibv_qp_init_attr init_attr;
    memset(&init_attr, 0, sizeof(init_attr));
    init_attr.qp_context = NULL;
    init_attr.send_cq = send_cq;
    init_attr.recv_cq = recv_cq;
    init_attr.qp_type = IBV_QPT_RC;
    init_attr.cap.max_send_wr = 64;
    init_attr.cap.max_recv_wr = 64;
    init_attr.cap.max_send_sge = 16;
    init_attr.cap.max_recv_sge = 16;
    init_attr.cap.max_inline_data = 64;

    struct ibv_qp* ibv_qp = ibv_create_qp(pd, &init_attr);
    if (ibv_qp == NULL)
        return NULL;

    uint32_t peer_qp = rdma_lite_rendezvous_pair(ibv_qp->qp_num);
    if (peer_qp == 0 || rlite_bringup_qp(ibv_qp, peer_qp) != 0) {
        ibv_destroy_qp(ibv_qp);
        return NULL;
    }

    struct rdma_lite_qp* lite_qp = (struct rdma_lite_qp*)calloc(1, sizeof(struct rdma_lite_qp));
    if (lite_qp == NULL) {
        ibv_destroy_qp(ibv_qp);
        return NULL;
    }
    lite_qp->ctx = lite_ctx;
    lite_qp->send_cq = lite_qp_attr->send_cq;
    lite_qp->recv_cq = lite_qp_attr->recv_cq;
    lite_qp->qp_type = lite_qp_attr->qp_type;
    lite_qp->qp_state = RDMA_LITE_QPS_RTS;
    lite_qp->qp_num = ibv_qp->qp_num;

    for (int i = 0; i < RLITE_MAX_QP; i++) {
        if (!g_qp_tab[i].in_use) {
            g_qp_tab[i].lite_qp = lite_qp;
            g_qp_tab[i].ibv_qp = ibv_qp;
            g_qp_tab[i].in_use = 1;
            break;
        }
    }
    return lite_qp;
}

int rdma_lite_destroy_qp(struct rdma_lite_qp* lite_qp)
{
    if (lite_qp == NULL)
        return 0;
    for (int i = 0; i < RLITE_MAX_QP; i++) {
        if (g_qp_tab[i].in_use && g_qp_tab[i].lite_qp == lite_qp) {
            if (g_qp_tab[i].ibv_qp)
                ibv_destroy_qp(g_qp_tab[i].ibv_qp);
            memset(&g_qp_tab[i], 0, sizeof(g_qp_tab[i]));
            break;
        }
    }
    free(lite_qp);
    return 0;
}

int rdma_lite_post_send(
    struct rdma_lite_qp* lite_qp, struct rdma_lite_send_wr* wr, struct rdma_lite_send_wr** bad_wr,
    struct rdma_lite_post_send_attr* attr, struct rdma_lite_post_send_resp* resp)
{
    (void)attr;
    if (resp)
        memset(resp, 0, sizeof(*resp));
    if (bad_wr)
        *bad_wr = NULL;
    struct ibv_qp* ibv_qp = rlite_find_ibv_qp(lite_qp);
    if (ibv_qp == NULL || wr == NULL) {
        if (bad_wr)
            *bad_wr = wr;
        return -EINVAL;
    }

    struct ibv_sge sge[16];
    struct ibv_send_wr swr;
    memset(&swr, 0, sizeof(swr));
    int nsge = wr->num_sge;
    if (nsge > 16)
        nsge = 16;
    for (int i = 0; i < nsge; i++) {
        sge[i].addr = wr->sg_list[i].addr;
        sge[i].length = wr->sg_list[i].length;
        sge[i].lkey = wr->sg_list[i].lkey;
    }
    swr.wr_id = wr->wr_id;
    swr.next = NULL;
    swr.sg_list = sge;
    swr.num_sge = nsge;
    if (wr->opcode == RDMA_LITE_WR_RDMA_WRITE || wr->opcode == RDMA_LITE_WR_RDMA_READ) {
        swr.opcode = (wr->opcode == RDMA_LITE_WR_RDMA_WRITE) ? IBV_WR_RDMA_WRITE : IBV_WR_RDMA_READ;
        swr.wr.rdma.remote_addr = wr->remote_addr;
        swr.wr.rdma.rkey = wr->rkey;
    } else if (wr->opcode == RDMA_LITE_WR_WRITE_WITH_NOTIFY) {
        /* WRITE_WITH_NOTIFY carries RDMA WRITE data semantics: the payload goes
         * to remote_addr, and imm_data (big endian offset | record flag, built
         * by the RA layer) is forwarded so the host ibverbs stub can emulate
         * the hardware notify side effect. */
        swr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
        swr.imm_data = wr->imm_data;
        swr.wr.rdma.remote_addr = wr->remote_addr;
        swr.wr.rdma.rkey = wr->rkey;
    } else {
        swr.opcode = IBV_WR_SEND;
    }
    swr.send_flags = (wr->send_flags & RDMA_LITE_SEND_SIGNALED) ? IBV_SEND_SIGNALED : 0;

    struct ibv_send_wr* bad = NULL;
    int ret = ibv_post_send(ibv_qp, &swr, &bad);
    if (ret) {
        if (bad_wr)
            *bad_wr = wr;
        return ret;
    }
    return 0;
}

int rdma_lite_post_recv(struct rdma_lite_qp* lite_qp, struct rdma_lite_recv_wr* wr, struct rdma_lite_recv_wr** bad_wr)
{
    if (bad_wr)
        *bad_wr = NULL;
    struct ibv_qp* ibv_qp = rlite_find_ibv_qp(lite_qp);
    if (ibv_qp == NULL || wr == NULL) {
        if (bad_wr)
            *bad_wr = wr;
        return -EINVAL;
    }

    struct ibv_sge sge[16];
    struct ibv_recv_wr rwr;
    memset(&rwr, 0, sizeof(rwr));
    int nsge = wr->num_sge;
    if (nsge > 16)
        nsge = 16;
    for (int i = 0; i < nsge; i++) {
        sge[i].addr = wr->sg_list[i].addr;
        sge[i].length = wr->sg_list[i].length;
        sge[i].lkey = wr->sg_list[i].lkey;
    }
    rwr.wr_id = wr->wr_id;
    rwr.next = NULL;
    rwr.sg_list = sge;
    rwr.num_sge = nsge;

    struct ibv_recv_wr* bad = NULL;
    int ret = ibv_post_recv(ibv_qp, &rwr, &bad);
    if (ret) {
        if (bad_wr)
            *bad_wr = wr;
        return ret;
    }
    return 0;
}

static void rlite_translate_wc(const struct ibv_wc* ibv_wc, struct rdma_lite_wc_v2* lite_wc)
{
    memset(lite_wc, 0, sizeof(*lite_wc));
    lite_wc->wc.wr_id = ibv_wc->wr_id;
    lite_wc->wc.status = (enum rdma_lite_wc_status)ibv_wc->status;
    lite_wc->wc.opcode = (enum rdma_lite_wc_opcode)ibv_wc->opcode;
    lite_wc->wc.vendor_err = ibv_wc->vendor_err;
    lite_wc->wc.byte_len = ibv_wc->byte_len;
    lite_wc->wc.qp_num = ibv_wc->qp_num;
    lite_wc->wc.wc_flags = 0;
}

int rdma_lite_poll_cq(struct rdma_lite_cq* lite_cq, int num_entries, struct rdma_lite_wc* lite_wc)
{
    (void)lite_cq;
    (void)num_entries;
    (void)lite_wc;
    return 0;
}

int rdma_lite_poll_cq_v2(struct rdma_lite_cq* lite_cq, int num_entries, struct rdma_lite_wc_v2* lite_wc)
{
    struct ibv_cq* ibv_cq = rlite_find_ibv_cq(lite_cq);
    if (ibv_cq == NULL || lite_wc == NULL || num_entries <= 0)
        return 0;

    struct ibv_wc ibv_wc[16];
    int n = (num_entries > 16) ? 16 : num_entries;
    int got = ibv_poll_cq(ibv_cq, n, ibv_wc);
    if (got <= 0)
        return 0;
    for (int i = 0; i < got; i++) {
        rlite_translate_wc(&ibv_wc[i], &lite_wc[i]);
    }
    return got;
}

int rdma_lite_set_qp_sl(struct rdma_lite_qp* lite_qp, int sl)
{
    (void)lite_qp;
    (void)sl;
    return 0;
}

int rdma_lite_clean_qp(struct rdma_lite_qp* lite_qp)
{
    (void)lite_qp;
    return 0;
}

int rdma_lite_restore_snapshot(struct rdma_lite_context* lite_ctx)
{
    (void)lite_ctx;
    return 0;
}

unsigned int rdma_lite_get_api_version(void) { return 1; }
