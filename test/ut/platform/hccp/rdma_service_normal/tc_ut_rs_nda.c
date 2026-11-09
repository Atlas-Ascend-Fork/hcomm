/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include "ut_dispatch.h"
#include "ra_rs_err.h"
#include "rs_common_inner.h"
#include "rs_inner.h"
#include "dl_hal_function.h"
#include "dl_ibverbs_function.h"
#include "rs_rdma.h"
#include "rs_drv_rdma.h"
#include "rs.h"
#include "rs_nda.h"
#include "tc_ut_rs_nda.h"

#define rs_nda_ut_msg(fmt, args...) fprintf(stderr, "\t>>>>> " fmt, ##args)

extern __thread struct rs_cb* gRsCb;
extern struct ibv_context_extend* RsIbvOpenExtend(struct ibv_context* context);
extern int RsIbvQueryDeviceExtend(struct ibv_context_extend* context, struct ibv_device_attr_extend* extDevAttr);
extern int RsQueryRdevCb(unsigned int phyId, unsigned int rdevIndex, struct RsRdevCb** rdevCb);
extern int RsNdaQpCreateEx(struct RsQpCb* qpCb, struct ibv_qp_init_attr_extend* qpInitAttrEx, struct NdaQpInfo* info);
extern int RsGetNdaPcieDbCb(struct RsNdaCb* ndaCb, uint64_t hva, struct NdaPcieDbCb** ndaDbCb);
extern int RsGetNdaUbDbCb(struct RsNdaCb* ndaCb, uint64_t guidL, uint64_t guidH, struct NdaUbDbCb** ndaDbCb);
extern int RsNdaCqCreateEx(
    struct RsRdevCb* rdevCb, struct ibv_cq_init_attr_extend* cqInitAttrEx, struct NdaCqInfo* info, void** ibvCqExt);
extern void* RsNdaUbAlloc(size_t size);
extern void* RsNdaDbMmap(struct doorbell_map_desc* desc);
extern int RsNdaDbUnmap(void* ptr, struct doorbell_map_desc* desc);

int RsQueryRdevCbNdaStub(unsigned int phyId, unsigned int rdevIndex, struct RsRdevCb** rdevCb)
{
    static struct RsRdevCb ndaRdevCb = {0};
    ndaRdevCb.rsCb = gRsCb;

    *rdevCb = &ndaRdevCb;
    return 0;
}

static struct ibv_context_extend gNdaIbCtxEx;
static uint32_t gNdaExtDevCap;
static int gNdaDirectFlag;

int RsQueryRdevCbNdaExtStub(unsigned int phyId, unsigned int rdevIndex, struct RsRdevCb** rdevCb)
{
    static struct RsRdevCb ndaRdevCb = {0};
    ndaRdevCb.rsCb = gRsCb;
    ndaRdevCb.ibCtxEx = &gNdaIbCtxEx;
    ndaRdevCb.extDevCap = gNdaExtDevCap;
    ndaRdevCb.directFlag = gNdaDirectFlag;
    *rdevCb = &ndaRdevCb;
    return 0;
}

static struct ibv_context_extend gStubIbCtxEx;
static struct ibv_device_attr_extend gStubExtDevAttr;
static int gStubQueryRet;

struct ibv_context_extend* RsIbvOpenExtendStub(struct ibv_context* context)
{
    (void)context;
    return &gStubIbCtxEx;
}

struct ibv_context_extend* RsIbvOpenExtendNullStub(struct ibv_context* context)
{
    (void)context;
    return NULL;
}

int RsIbvQueryDeviceExtendStub(struct ibv_context_extend* context, struct ibv_device_attr_extend* extDevAttr)
{
    (void)context;
    *extDevAttr = gStubExtDevAttr;
    return gStubQueryRet;
}

int DlDrvMemGetAttributeStub(DVdeviceptr vptr, struct DVattribute* attr)
{
    attr->memType = DV_MEM_LOCK_DEV;
    return 0;
}

int DlHalHostRegisterStub(void* srcPtr, UINT64 size, UINT32 flag, UINT32 devId, void** dstPtr)
{
    static void* sDstPtr = NULL;
    if (dstPtr == NULL) {
        return -1;
    }

    *dstPtr = &sDstPtr;
    return 0;
}

int DlHalHostUnRegisterExStub(void* srcPtr, UINT32 devId)
{
    uint64_t alignHva = RaRsAlignDown((uint64_t)1, RA_RS_4K_PAGE_SIZE);

    if ((uint64_t)(uintptr_t)srcPtr == alignHva) {
        return 0;
    }
    return -1;
}

static int64_t gNdaMainboardId = 0;
static int gNdaMainboardRet = 0;

int DlHalGetDeviceInfoMainboardStub(uint32_t devId, int32_t moduleType, int32_t infoType, int64_t* value)
{
    (void)devId;
    (void)moduleType;
    (void)infoType;
    if (value != NULL) {
        *value = gNdaMainboardId;
    }
    return gNdaMainboardRet;
}

void TcRsNdaGetDirectFlag()
{
    unsigned int rdevIndex = 0;
    unsigned int phyId = 0;
    int directFlag = 0;
    int ret = 0;

    ret = RsNdaGetDirectFlag(RS_MAX_DEV_NUM, rdevIndex, &directFlag);
    EXPECT_INT_EQ(ret, -EINVAL);

    mocker_clean();
    mocker_invoke(RsQueryRdevCb, RsQueryRdevCbNdaStub, 1);
    ret = RsNdaGetDirectFlag(phyId, rdevIndex, &directFlag);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRsNdaGetDirectFlagExt()
{
    unsigned int rdevIndex = 0;
    unsigned int phyId = 0;
    int directFlag = 0;
    int ret = 0;

    // extDevCap 含 NDA 位 -> 返回 directFlag
    mocker_clean();
    gNdaExtDevCap = IBV_EXTEND_DEV_NDA;
    gNdaDirectFlag = DIRECT_FLAG_UB;
    mocker_invoke(RsQueryRdevCb, RsQueryRdevCbNdaExtStub, 1);
    ret = RsNdaGetDirectFlag(phyId, rdevIndex, &directFlag);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_INT_EQ(directFlag, DIRECT_FLAG_UB);
    mocker_clean();

    // extDevCap 不含 NDA 位 -> 返回 NOTSUPP
    mocker_clean();
    gNdaExtDevCap = 0;
    gNdaDirectFlag = DIRECT_FLAG_UB;
    mocker_invoke(RsQueryRdevCb, RsQueryRdevCbNdaExtStub, 1);
    ret = RsNdaGetDirectFlag(phyId, rdevIndex, &directFlag);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_INT_EQ(directFlag, DIRECT_FLAG_NOTSUPP);
    mocker_clean();
}

void TcRsInitNdaCb()
{
    struct RsRdevCb rdevCb = {0};
    int ret = 0;

    gNdaMainboardId = 0x0300;
    gNdaMainboardRet = 0;

    // 查询成功 -> extDevCap == extDevAttr.ext_cap
    gRsCb = calloc(1, sizeof(struct rs_cb));
    rdevCb.rsCb = gRsCb;
    gStubExtDevAttr.ext_cap = IBV_EXTEND_DEV_NDA | IBV_EXTEND_DEV_NDR;
    gStubQueryRet = 0;
    mocker_clean();
    mocker_invoke((stub_fn_t)DlHalGetDeviceInfo, DlHalGetDeviceInfoMainboardStub, 1);
    mocker_invoke(RsIbvOpenExtend, RsIbvOpenExtendStub, 1);
    mocker_invoke(RsIbvQueryDeviceExtend, RsIbvQueryDeviceExtendStub, 1);
    ret = RsInitNdaCb(&rdevCb);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_INT_EQ((int)rdevCb.extDevCap, (int)gStubExtDevAttr.ext_cap);
    mocker_clean();
    free(gRsCb);

    // 查询失败 -> extDevCap == 0
    memset(&rdevCb, 0, sizeof(rdevCb));
    gRsCb = calloc(1, sizeof(struct rs_cb));
    rdevCb.rsCb = gRsCb;
    gStubQueryRet = -1;
    mocker_clean();
    mocker_invoke((stub_fn_t)DlHalGetDeviceInfo, DlHalGetDeviceInfoMainboardStub, 1);
    mocker_invoke(RsIbvOpenExtend, RsIbvOpenExtendStub, 1);
    mocker_invoke(RsIbvQueryDeviceExtend, RsIbvQueryDeviceExtendStub, 1);
    ret = RsInitNdaCb(&rdevCb);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_INT_EQ((int)rdevCb.extDevCap, 0);
    mocker_clean();
    free(gRsCb);

    // ibCtxEx == NULL -> 返回 0 且 extDevCap == 0
    memset(&rdevCb, 0, sizeof(rdevCb));
    gRsCb = calloc(1, sizeof(struct rs_cb));
    rdevCb.rsCb = gRsCb;
    mocker_clean();
    mocker_invoke((stub_fn_t)DlHalGetDeviceInfo, DlHalGetDeviceInfoMainboardStub, 1);
    mocker_invoke(RsIbvOpenExtend, RsIbvOpenExtendNullStub, 1);
    ret = RsInitNdaCb(&rdevCb);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_INT_EQ((int)rdevCb.extDevCap, 0);
    mocker_clean();
    free(gRsCb);

    gRsCb = NULL;
}

void TcRsInitNdaCbDirectFlag()
{
    /* mainboard_id 命中白名单 -> DIRECT_FLAG_UB，需与 rs_nda.c 白名单逐项保持一致 */
    static const uint64_t ubMainboardIds[] = {
        0x03, 0x05, 0x07,                                           /* 天成 POD */
        0x21, 0x23, 0x25, 0x27, 0x29, 0x2B, 0x2D, 0x2F, 0x31, 0x33, /* A+K Server */
        0x44, 0x46, 0x48, 0x4A, 0x4C,                               /* A+X Server */
    };
    /* 不在白名单 -> DIRECT_FLAG_PCIE（PCIE 标卡/预留/装备/EVB 等机型） */
    static const uint64_t pcieMainboardIds[] = {
        0x00, 0x61, 0x81, 0xC1, 0xE1, 0x04,
    };
    struct RsRdevCb rdevCb = {0};
    unsigned int i = 0;
    int ret = 0;

    gRsCb = calloc(1, sizeof(struct rs_cb));

    mocker_clean();
    gNdaMainboardRet = 0;
    mocker_invoke((stub_fn_t)DlHalGetDeviceInfo, DlHalGetDeviceInfoMainboardStub, 64);
    /* ibCtxEx 返回 NULL，跳过 extDevCap 查询，仅验证 directFlag 判定 */
    mocker_invoke(RsIbvOpenExtend, RsIbvOpenExtendNullStub, 64);

    for (i = 0; i < sizeof(ubMainboardIds) / sizeof(ubMainboardIds[0]); i++) {
        gNdaMainboardId = (int64_t)ubMainboardIds[i];
        memset(&rdevCb, 0, sizeof(rdevCb));
        rdevCb.rsCb = gRsCb;
        ret = RsInitNdaCb(&rdevCb);
        EXPECT_INT_EQ(ret, 0);
        EXPECT_INT_EQ(rdevCb.directFlag, DIRECT_FLAG_UB);
    }

    for (i = 0; i < sizeof(pcieMainboardIds) / sizeof(pcieMainboardIds[0]); i++) {
        gNdaMainboardId = (int64_t)pcieMainboardIds[i];
        memset(&rdevCb, 0, sizeof(rdevCb));
        rdevCb.rsCb = gRsCb;
        ret = RsInitNdaCb(&rdevCb);
        EXPECT_INT_EQ(ret, 0);
        EXPECT_INT_EQ(rdevCb.directFlag, DIRECT_FLAG_PCIE);
    }

    /* mainboard_id 查询失败 -> 告警降级，返回 0 且 directFlag 默认 DIRECT_FLAG_PCIE */
    gNdaMainboardRet = -1;
    memset(&rdevCb, 0, sizeof(rdevCb));
    rdevCb.rsCb = gRsCb;
    ret = RsInitNdaCb(&rdevCb);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_INT_EQ(rdevCb.directFlag, DIRECT_FLAG_PCIE);
    mocker_clean();

    free(gRsCb);
    gRsCb = NULL;
}

void TcRsNdaQpCreateFailed()
{
    struct NdaQpInitAttr attr = {0};
    struct RsQpResp qpResp = {0};
    struct NdaQpInfo info = {0};
    unsigned int rdevIndex = 0;
    unsigned int phyId = 0;
    int ret = 0;
    struct RsNdaCb ndaCb = {0};
    struct NdaOps ops = {0};

    gRsCb = malloc(sizeof(struct rs_cb));
    attr.ops = &ops;
    gRsCb->ndaCb = &ndaCb;

    ret = RsNdaQpCreate(phyId, rdevIndex, NULL, &info, &qpResp);
    EXPECT_INT_EQ(ret, -EINVAL);

    attr.dmaMode = QBUF_DMA_MODE_MAX;
    ret = RsNdaQpCreate(phyId, rdevIndex, &attr, &info, &qpResp);
    EXPECT_INT_EQ(ret, -EINVAL);

    attr.dmaMode = QBUF_DMA_MODE_INDEP_UB;
    mocker_clean();
    mocker(RsQueryRdevCb, 1, -1);
    ret = RsNdaQpCreate(phyId, rdevIndex, &attr, &info, &qpResp);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker_invoke(RsQueryRdevCb, RsQueryRdevCbNdaStub, 1);
    mocker(RsNdaQpCreateEx, 1, -1);
    ret = RsNdaQpCreate(phyId, rdevIndex, &attr, &info, &qpResp);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
    free(gRsCb);
    gRsCb = NULL;
}

void TcRsNdaQpCreate()
{
    struct NdaQpInitAttr attr = {0};
    struct RsInitConfig cfg = {0};
    struct NdaQpInfo info = {0};
    struct rdev rdevInfo = {0};
    uint32_t rdevIndex = 0;
    uint32_t phyId = 0;
    int ret = 0;

    struct RsQpResp resp = {0};
    struct NdaOps ops = {0};
    uint32_t qpn = 0;

    rdevInfo.phyId = 0;
    rdevInfo.family = AF_INET;
    rdevInfo.localIp.addr.s_addr = inet_addr("127.0.0.1");

    rs_nda_ut_msg("resource prepare begin..................\n");
    /* +++++Resource Prepare+++++ */
    cfg.chipId = 0;
    cfg.hccpMode = NETWORK_OFFLINE;
    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);
    rs_nda_ut_msg("RS INIT, ret:%d !\n", ret);

    ret = RsRdevInit(rdevInfo, NO_USE, &rdevIndex);
    EXPECT_INT_EQ(ret, 0);

    attr.ops = &ops;
    ret = RsNdaQpCreate(phyId, rdevIndex, &attr, &info, &resp);
    EXPECT_INT_EQ(ret, 0);
    qpn = resp.qpn;
    rs_nda_ut_msg("RS CREATE NDA QP: QPN:%d, ret:%d\n", qpn, ret);

    ret = RsNdaQpDestroy(phyId, rdevIndex, qpn);
    EXPECT_INT_EQ(ret, 0);
    rs_nda_ut_msg("RS destroy NDA QP: ret:%d\n", ret);

    ret = RsRdevDeinit(phyId, NO_USE, rdevIndex);
    EXPECT_INT_EQ(ret, 0);

    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);
    rs_nda_ut_msg("resource free done..................\n");
}

void TcRsNdaQpCreateEx()
{
    struct ibv_qp_init_attr_extend qpInitAttrEx = {0};
    struct ibv_context_extend ibCtxEx = {0};
    struct RsRdevCb rdevCb = {0};
    struct NdaQpInfo info = {0};
    struct RsQpCb qpCb = {0};
    struct ibv_pd pd = {0};
    int ret = 0;

    qpInitAttrEx.pd = &pd;
    rdevCb.ibCtxEx = &ibCtxEx;
    qpCb.rdevCb = &rdevCb;
    mocker_clean();
    mocker(RsIbvQueryQp, 1, -1);
    ret = RsNdaQpCreateEx(&qpCb, &qpInitAttrEx, &info);
    EXPECT_INT_EQ(ret, -EOPENSRC);
    mocker_clean();

    mocker(RsDrvQpInfoRelated, 1, -1);
    ret = RsNdaQpCreateEx(&qpCb, &qpInitAttrEx, &info);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
}

void TcRsNdaUbAllocFailed()
{
    struct RsNdaCb* ndaCbCur = NULL;
    struct RsNdaCb ndaCb = {0};
    size_t size = 32;
    void* ptr = NULL;

    gRsCb = malloc(sizeof(struct rs_cb));
    memset(gRsCb, 0, sizeof(struct rs_cb));
    gRsCb->ndaCb = &ndaCb;
    ptr = RsNdaUbAlloc(size);
    EXPECT_ADDR_EQ(ptr, NULL);

    ndaCbCur = (struct RsNdaCb*)gRsCb->ndaCb;
    ndaCbCur->ndaOps.alloc = malloc;
    ptr = RsNdaUbAlloc(size);
    EXPECT_ADDR_EQ(ptr, NULL);

    ndaCbCur->ndaOps.free = free;

    // rs_nda_ut_msg("resource free 1..................\n");
    // mocker(DlDrvMemGetAttribute, 10, 0);
    // ptr = RsNdaUbAlloc(size);
    // EXPECT_ADDR_NE(ptr, NULL);
    // rs_nda_ut_msg("ptr:0x%llx llt\n", (uint64_t)(uintptr_t)ptr);

    // rs_nda_ut_msg("resource free 2..................\n");
    // gRsCb->ndaOps.free = free;
    // RsNdaUbFree(ptr);
    // mocker_clean();
    // rs_nda_ut_msg("resource free 3..................\n");

    mocker_clean();
    mocker(DlDrvMemGetAttribute, 1, -1);
    ptr = RsNdaUbAlloc(size);
    EXPECT_ADDR_EQ(ptr, NULL);
    mocker_clean();

    mocker_invoke(DlDrvMemGetAttribute, DlDrvMemGetAttributeStub, 10);
    mocker(DlHalMemRegUbSegment, 1, -1);
    ptr = RsNdaUbAlloc(size);
    EXPECT_ADDR_EQ(ptr, NULL);
    mocker_clean();

    free(gRsCb);
    gRsCb = NULL;
}

void TcRsNdaDbMmapHostVa()
{
    struct doorbell_map_desc desc = {0};
    struct RsNdaCb* ndaCbCur = NULL;
    struct RsNdaCb ndaCb = {0};
    void* ptr = NULL;
    int ret = 0;

    desc.hva = 1;
    desc.type = DB_MAP_MODE_HOST_VA;
    gRsCb = malloc(sizeof(struct rs_cb));
    memset(gRsCb, 0, sizeof(struct rs_cb));
    gRsCb->ndaCb = (void*)&ndaCb;

    ndaCbCur = (struct RsNdaCb*)gRsCb->ndaCb;
    RS_INIT_LIST_HEAD(&ndaCbCur->ndaPcieCb.ndaDbList);
    mocker_clean();
    mocker(DlDrvDeviceGetIndexByPhyId, 10, -1);
    ptr = RsNdaDbMmap(&desc);
    EXPECT_ADDR_EQ(ptr, NULL);
    mocker_clean();
    ptr = NULL;

    // build up NdaDbCb
    mocker(DlDrvDeviceGetIndexByPhyId, 10, 0);
    mocker_invoke_p5(
        "DlHalHostRegister", "DlHalHostRegisterStub", (stub_fn_t)DlHalHostRegister, (stub_fn_t)DlHalHostRegisterStub,
        10);
    ptr = RsNdaDbMmap(&desc);
    EXPECT_ADDR_NE(ptr, NULL);
    ptr = NULL;

    ptr = RsNdaDbMmap(&desc);
    EXPECT_ADDR_NE(ptr, NULL);
    mocker_clean();

    mocker(DlDrvDeviceGetIndexByPhyId, 10, 0);
    mocker_invoke_p5(
        "DlHalHostRegister", "DlHalHostRegisterStub", (stub_fn_t)DlHalHostRegister, (stub_fn_t)DlHalHostRegisterStub,
        10);
    mocker_invoke(DlHalHostUnRegisterEx, DlHalHostUnRegisterExStub, 10);
    ret = RsNdaDbUnmap(ptr, &desc);
    EXPECT_INT_EQ(ret, 0);
    ret = RsNdaDbUnmap(ptr, &desc);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    ret = RsNdaDbUnmap(ptr, &desc);
    EXPECT_INT_EQ(ret, -ENODEV);

    mocker(RsGetNdaPcieDbCb, 1, -1);
    ptr = RsNdaDbMmap(&desc);
    EXPECT_ADDR_EQ(ptr, NULL);

    desc.type = DB_MAP_MODE_UB_RES + 1;
    ptr = RsNdaDbMmap(&desc);
    EXPECT_ADDR_EQ(ptr, NULL);

    free(gRsCb);
    gRsCb = NULL;
}

void TcRsNdaDbMmapUbRes()
{
    struct doorbell_map_desc desc = {0};
    struct RsNdaCb* ndaCbCur = NULL;
    struct RsNdaCb ndaCb = {0};
    void* ptr = NULL;
    int ret = 0;

    desc.ub_res.guid_h = 1;
    desc.ub_res.guid_l = 2;
    desc.ub_res.bits.offset = 3;
    desc.type = DB_MAP_MODE_UB_RES;
    gRsCb = malloc(sizeof(struct rs_cb));
    memset(gRsCb, 0, sizeof(struct rs_cb));
    gRsCb->ndaCb = &ndaCb;

    ndaCbCur = (struct RsNdaCb*)gRsCb->ndaCb;
    RS_INIT_LIST_HEAD(&ndaCbCur->ndaUbCb.ndaDbList);
    mocker_clean();
    mocker(DlDrvDeviceGetIndexByPhyId, 10, -1);
    ptr = RsNdaDbMmap(&desc);
    EXPECT_ADDR_EQ(ptr, NULL);
    mocker_clean();
    ptr = NULL;

    mocker(DlDrvDeviceGetIndexByPhyId, 10, 0);
    mocker(DlHalResAddrMapV2, 10, -1);
    ptr = RsNdaDbMmap(&desc);
    EXPECT_ADDR_EQ(ptr, NULL);
    mocker_clean();
    ptr = NULL;

    // build up NdaGuidCb
    mocker(DlDrvDeviceGetIndexByPhyId, 10, 0);
    mocker(DlHalResAddrMapV2, 10, 0);
    ptr = RsNdaDbMmap(&desc);
    EXPECT_ADDR_NE(ptr, NULL);
    ptr = NULL;

    ptr = RsNdaDbMmap(&desc);
    EXPECT_ADDR_NE(ptr, NULL);
    mocker_clean();

    mocker(DlDrvDeviceGetIndexByPhyId, 10, 0);
    mocker(DlHalResAddrUnmapV2, 10, 0);
    ret = RsNdaDbUnmap(ptr, &desc);
    EXPECT_INT_EQ(ret, 0);
    ret = RsNdaDbUnmap(ptr, &desc);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    ret = RsNdaDbUnmap(ptr, &desc);
    EXPECT_INT_EQ(ret, -ENODEV);

    mocker(RsGetNdaUbDbCb, 1, -1);
    ptr = RsNdaDbMmap(&desc);
    EXPECT_ADDR_EQ(ptr, NULL);

    desc.type = DB_MAP_MODE_UB_RES + 1;
    ptr = RsNdaDbMmap(&desc);
    EXPECT_ADDR_EQ(ptr, NULL);

    free(gRsCb);
    gRsCb = NULL;
}

void TcRsNdaCqCreateFailed()
{
    struct NdaCqInitAttr attr = {0};
    struct NdaCqInfo info = {0};
    unsigned int rdevIndex = 0;
    unsigned int phyId = 0;
    void* ibvCqExt = NULL;
    int ret = 0;

    struct RsNdaCb ndaCb = {0};
    struct NdaOps ops = {0};

    gRsCb = malloc(sizeof(struct rs_cb));
    attr.ops = &ops;
    gRsCb->ndaCb = &ndaCb;

    ret = RsNdaCqCreate(phyId, rdevIndex, NULL, NULL, &ibvCqExt);
    EXPECT_INT_EQ(ret, -EINVAL);

    attr.dmaMode = QBUF_DMA_MODE_MAX;
    ret = RsNdaCqCreate(phyId, rdevIndex, &attr, &info, &ibvCqExt);
    EXPECT_INT_EQ(ret, -EINVAL);

    attr.dmaMode = QBUF_DMA_MODE_INDEP_UB;
    mocker_clean();
    mocker(RsQueryRdevCb, 1, -1);
    ret = RsNdaCqCreate(phyId, rdevIndex, &attr, &info, &ibvCqExt);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker_invoke(RsQueryRdevCb, RsQueryRdevCbNdaStub, 1);
    mocker(RsNdaCqCreateEx, 1, -1);
    ret = RsNdaCqCreate(phyId, rdevIndex, &attr, &info, &ibvCqExt);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
    free(gRsCb);
    gRsCb = NULL;
}

void TcRsNdaCqCreate()
{
    struct NdaCqInitAttr attr = {0};
    struct RsInitConfig cfg = {0};
    struct NdaCqInfo info = {0};
    struct rdev rdevInfo = {0};
    unsigned int rdevIndex = 0;
    unsigned int phyId = 0;
    void* ibvCqExt = NULL;
    int ret = 0;

    struct NdaOps ops = {0};

    rdevInfo.phyId = 0;
    rdevInfo.family = AF_INET;
    rdevInfo.localIp.addr.s_addr = inet_addr("127.0.0.1");

    rs_nda_ut_msg("resource prepare begin..................\n");
    /* +++++Resource Prepare+++++ */
    cfg.chipId = 0;
    cfg.hccpMode = NETWORK_OFFLINE;
    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    ret = RsRdevInit(rdevInfo, NO_USE, &rdevIndex);
    EXPECT_INT_EQ(ret, 0);

    attr.ops = &ops;
    ret = RsNdaCqCreate(phyId, rdevIndex, &attr, &info, &ibvCqExt);
    EXPECT_INT_EQ(ret, 0);

    ret = RsNdaCqDestroy(phyId, rdevIndex, ibvCqExt);
    EXPECT_INT_EQ(ret, 0);

    ret = RsRdevDeinit(phyId, NO_USE, rdevIndex);
    EXPECT_INT_EQ(ret, 0);

    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);
}

void TcRsNdaAlignDownWithMixedTypeAlign()
{
    int alignInt = (int)RA_RS_4K_PAGE_SIZE;
    unsigned int alignUint = (unsigned int)RA_RS_4K_PAGE_SIZE;
    uint64_t addr = 0x100001234ULL;

    uint64_t r1 = RaRsAlignDown(addr, alignInt);
    uint64_t r2 = RaRsAlignDown(addr, alignUint);
    EXPECT_LONG_EQ((long)r1, (long)r2);
    EXPECT_LONG_EQ((long)r1, (long)0x100001000ULL);

    uint64_t up1 = RaRsAlignUp(addr, alignInt);
    uint64_t up2 = RaRsAlignUp(addr, alignUint);
    EXPECT_LONG_EQ((long)up1, (long)up2);
    EXPECT_LONG_EQ((long)up1, (long)0x100002000ULL);
}
