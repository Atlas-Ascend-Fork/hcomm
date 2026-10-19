/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <infiniband/verbs.h>
#include "verbs_exp.h"
#include "hns_roce_u_cmd.h"
#include "ascend_hal.h"

#define ROCE_POOL_MAX 8
#define ROCE_POOL_SIZE 4096

static unsigned long long g_roce_pool_va[ROCE_POOL_MAX];
static unsigned int g_roce_pool_num = 0;

struct ibv_mr*
ibv_exp_reg_mr(struct ibv_pd* pd, void* addr, size_t length, int access, struct roce_process_sign roceSign)
{
    struct ibv_mr* mr = calloc(1, sizeof(struct ibv_mr));
    if (!mr) {
        return NULL;
    }

    mr->pd = pd;
    mr->addr = addr;
    mr->length = length;
    mr->lkey = 0x12345678;
    mr->rkey = 0x87654321;

    return mr;
}

int ibv_exp_query_notify(struct ibv_context* context, unsigned long long* notifyVa, unsigned long long* size)
{
    return 0;
}

int ibv_exp_post_send(struct ibv_qp* qp, struct ibv_send_wr* wr, struct ibv_send_wr** badWr, struct wr_exp_rsp* expRsp)
{
    return 0;
}

int ibv_exp_query_device(struct ibv_context*, struct dev_cap_info*) { return 0; }

struct ibv_ah* ibv_exp_create_ah(struct ibv_pd* pd, struct ibv_exp_ah_attr* attrx) { return NULL; }

struct ibv_qp* ibv_exp_create_qp(
    struct ibv_pd* pd, struct ibv_exp_qp_init_attr* qp_init_attr, struct rdma_lite_device_qp_attr* qp_resp)
{
    if (!pd || !qp_init_attr) {
        errno = EINVAL;
        return NULL;
    }
    struct ibv_qp* qp = calloc(1, sizeof(struct ibv_qp));
    if (!qp) {
        errno = ENOMEM;
        return NULL;
    }
    qp->pd = pd;
    qp->qp_type = qp_init_attr->attr.qp_type;
    qp->send_cq = qp_init_attr->attr.send_cq;
    qp->recv_cq = qp_init_attr->attr.recv_cq;
    qp->srq = qp_init_attr->attr.srq;
    qp->state = IBV_QPS_RESET;
    if (qp_init_attr->attr.send_cq != NULL) {
        qp->context = qp_init_attr->attr.send_cq->context;
    } else if (pd != NULL) {
        qp->context = pd->context;
    }
    static unsigned int g_mock_qpn = 1;
    qp->qp_num = g_mock_qpn++;
    if (qp_resp) {
        qp_resp->qpn = qp->qp_num;
        qp_resp->qp_info = 0;
        qp_resp->max_inline_data
            = (qp_init_attr->attr.cap.max_inline_data > 0) ? qp_init_attr->attr.cap.max_inline_data : 32;
    }
    return qp;
}

struct ibv_cq* ibv_exp_create_cq(
    struct ibv_context* context, int cqe, void* cq_context, struct ibv_comp_channel* comp_channel, int comp_vector,
    struct rdma_lite_device_cq_init_attr* init_attr, struct rdma_lite_device_cq_attr* cq_attr)
{
    if (!context || cqe <= 0) {
        errno = EINVAL;
        return NULL;
    }

    struct ibv_cq* cq = (struct ibv_cq*)malloc(sizeof(struct ibv_cq));
    if (!cq) {
        errno = ENOMEM;
        return NULL;
    }

    cq->context = context;
    cq->cqe = cqe;
    cq->cq_context = cq_context;
    cq->channel = comp_channel;

    return cq;
}

int roce_get_roce_dev_data(const char* devName, struct roce_dev_data* rdevData) { return 0; }

int roce_get_tsqp_depth(
    const char* devName, unsigned int rdevIndex, unsigned int* tempDepth, unsigned int* qpNum, unsigned int* sqDepth)
{
    return 0;
}

int roce_set_tsqp_depth(
    const char* devName, unsigned int rdevIndex, unsigned int tempDepth, unsigned int* qpNum, unsigned int* sqDepth)
{
    return 0;
}

int roce_init_mem_pool(
    const struct roce_mem_cq_qp_attr* attr, struct rdma_lite_device_mem_attr* memAttr, unsigned int num)
{
    (void)attr;
    (void)num;
    if (memAttr == NULL || g_roce_pool_num >= ROCE_POOL_MAX) {
        return -1;
    }
    void* va = NULL;
    drvError_t ret = halMemAlloc(&va, ROCE_POOL_SIZE, MEM_DEV | MEM_TYPE_HBM);
    if (ret != DRV_ERROR_NONE || va == NULL) {
        return -1;
    }
    g_roce_pool_num++;
    g_roce_pool_va[g_roce_pool_num - 1] = (unsigned long long)(uintptr_t)va;
    memAttr->va = (unsigned long long)(uintptr_t)va;
    memAttr->mem_size = ROCE_POOL_SIZE;
    memAttr->mem_idx = g_roce_pool_num;
    return 0;
}

int roce_deinit_mem_pool(unsigned int memIdx)
{
    if (memIdx == 0 || memIdx > g_roce_pool_num) {
        return -1;
    }
    if (g_roce_pool_va[memIdx - 1] != 0) {
        (void)halMemFree((void*)(uintptr_t)g_roce_pool_va[memIdx - 1]);
        g_roce_pool_va[memIdx - 1] = 0;
    }
    return 0;
}

int roce_query_qpc(struct ibv_qp* qp, struct hns_roce_qpc_attr_val* attrVal, unsigned int attrMask) { return 0; }

int roce_mmap_ai_db_reg(struct ibv_context* context, unsigned int tgid) { return 0; }

int roce_unmmap_ai_db_reg(struct ibv_context* context) { return 0; }

int roce_get_cq_data_plane_info(struct ibv_cq* cq, struct hns_roce_cq_data_plane_info* info) { return 0; }

int roce_get_qp_data_plane_info(struct ibv_qp* qp, struct hns_roce_qp_data_plane_info* info) { return 0; }

int roce_remap_mr(struct ibv_mr* mr, struct hns_roce_mr_remap_info info[], unsigned int num) { return 0; }

unsigned int roce_get_api_version(void) { return 0; }
