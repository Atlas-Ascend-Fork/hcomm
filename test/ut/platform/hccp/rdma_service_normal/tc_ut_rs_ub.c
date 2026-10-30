/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define _GNU_SOURCE
#define SOCK_CONN_TAG_SIZE 192
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <netdb.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <ifaddrs.h>

#include "ut_dispatch.h"
#include "stub/ibverbs.h"
#include "rs.h"
#include "rs_ub_tp.h"
#include "rs_ub_dfx.h"
#include "rs_ub.h"
#include "rs_ctx.h"
#include "rs_common_inner.h"
#include "rs_inner.h"
#include "rs_ctx_inner.h"
#include "tc_ut_rs_ub.h"
#include "rs_drv_rdma.h"
#include "stub/verbs_exp.h"
#include "encrypt.h"
#include "rs_epoll.h"
#include "rs_socket.h"
#include "ra_rs_err.h"
#include "rs_ub_jfc.h"

extern int RsNetAllocJfcId(const char* udevName, unsigned int jfcMode, unsigned int* jfcId);
extern int RsNetFreeJfcId(const char* udevName, unsigned int jfcMode, unsigned int jfcId);
extern int RsNetGetCqeBaseAddr(unsigned int dieId, unsigned long long* cqeBaseAddr);

extern uint32_t RsGenerateUeInfo(uint32_t dieId, uint32_t funcId);
extern uint32_t RsGenerateDevIndex(uint32_t devCnt, uint32_t dieId, uint32_t funcId);
extern bool RsGenerateMmapResIdCheck(uint32_t exUbResType, uint32_t dieId);
extern uint32_t RsGenerateMmapResId(uint32_t ubResId, uint32_t exUbResType, uint32_t dieId, uint32_t funcId);
extern int RsUbGetRdevCb(struct rs_cb* rsCb, unsigned int rdevIndex, struct RsUbDevCb** devCb);
typedef long unsigned int (*stub_u64_fn_t)(
    long unsigned int data0, long unsigned int data1, long unsigned int data2, long unsigned int data3,
    long unsigned int data4, long unsigned int data5);
extern void mocker_u64_invoke(stub_u64_fn_t sh, stub_u64_fn_t th, int most);
extern int RsUrmaDeviceApiInit(void);
extern int RsOpenUrmaSo(void);
extern int RsUrmaApiInit(void);
extern void RsUbCtxExtJettyCreateTaCache(struct RsCtxJettyCb* jettyCb, urma_jetty_cfg_t* jettyCfg);
extern int RsUbDevCbInit(
    struct CtxInitAttr* attr, struct RsUbDevCb* devCb, struct rs_cb* rscb, int* devIndex, struct DevBaseAttr* devAttr);
extern void RsJfcCallbackProcess(struct RsCtxJettyCb* jettyCb, urma_cr_t* cr, urma_jfc_t* jfc);
extern int RsUrmaJettyApiInit(void);
extern int RsUrmaJfcApiInit(void);
extern int RsUrmaSegmentApiInit(void);
extern int RsUrmaDataApiInit(void);
extern urma_device_t** RsUrmaGetDeviceList(int* numDevices);
extern urma_eid_info_t* RsUrmaGetEidList(urma_device_t* dev, uint32_t* cnt);
extern void RsUrmaFreeDeviceList(urma_device_t** deviceList);
extern void RsUrmaFreeEidList(urma_eid_info_t* eidList);
extern int RsUrmaQueryDevice(urma_device_t* dev, urma_device_attr_t* devAttr);
extern urma_context_t* RsUrmaCreateContext(urma_device_t* dev, uint32_t eidIndex);
extern int RsUrmaDeleteContext(urma_context_t* ctx);
extern urma_status_t urma_delete_context(urma_context_t* ctx);
extern void RsUbFreeAsyncEventCbList(struct RsUbDevCb* devCb, struct RsListHead* asyncEventList);
extern void RsUbDevShareJfrDeinit(struct RsUbDevCb* devCb);
extern void RsSensorNodeUnregister(struct rs_cb* rscb);
extern int RsUrmaUserCtl(urma_context_t* ctx, urma_user_ctl_in_t* in, urma_user_ctl_out_t* out);
extern int RsUbApiInit(void);
extern void RsUbApiDeinit(void);
extern int RsUrmaGetEidByIp(const urma_context_t* ctx, const urma_net_addr_t* netAddr, urma_eid_t* eid);
extern int RsUrmaGetIpByEid(const urma_context_t* ctx, const urma_eid_t* eid, urma_net_addr_t* netAddr);
extern int RsUbCtxExtJettyCreate(struct RsCtxJettyCb* jettyCb, urma_jetty_cfg_t* jettyCfg);
extern void RsUbCtxExtJettyDelete(struct RsCtxJettyCb* jettyCb);
extern int RsUbCtxRegJettyDb(struct RsCtxJettyCb* jettyCb, void* dwqeAddr);
extern int RsInitRscbCfg(struct rs_cb* rscb, struct RsInitConfig* cfg);
extern int RsUbCreateCtx(urma_device_t* urmaDev, unsigned int eidIndex, urma_context_t** urmaCtx);
extern int RsUbGetUeInfo(urma_context_t* urmaCtx, struct DevBaseAttr* devBaseAttr);
extern int RsUbGetDevAttr(struct RsUbDevCb* devCb, struct DevBaseAttr* devAttr, unsigned int* devIndex);
extern int RsUbGetJfcCb(struct RsUbDevCb* devCb, unsigned long long addr, struct RsCtxJfcCb** jfcCb);
extern void RsUbFreeSegCbList(struct RsUbDevCb* devCb, struct RsListHead* lsegList, struct RsListHead* rsegList);
extern void RsUbFreeJettyCbList(struct RsUbDevCb* devCb, struct RsListHead* jettyList, struct RsListHead* rjettyList);
extern void RsUbFreeJfcCbList(struct RsUbDevCb* devCb, struct RsListHead* jfcList);
extern void RsUbFreeJfceCbList(struct RsUbDevCb* devCb, struct RsListHead* jfceList);
extern void RsUbFreeTokenIdCbList(struct RsUbDevCb* devCb, struct RsListHead* tokenIdList);
extern int RsUbGetTokenIdCb(struct RsUbDevCb* devCb, unsigned long long addr, struct RsTokenIdCb** tokenIdCb);
extern int RsUbInitSegCb(struct MemRegAttrT* memAttr, struct RsUbDevCb* devCb, struct RsSegCb* segCb);
extern int RsUbCtxInitJettyCb(struct RsUbDevCb* devCb, struct CtxQpAttr* attr, struct RsCtxJettyCb** jettyCb);
extern int RsUbQueryJfcCb(
    struct RsUbDevCb* devCb, unsigned long long scqIndex, unsigned long long rcqIndex, struct RsCtxJfcCb** sendJfcCb,
    struct RsCtxJfcCb** recvJfcCb);
extern void RsUbCtxFreeJettyCb(struct RsCtxJettyCb* jettyCb);
extern int
RsUbCtxDrvJettyCreate(struct RsCtxJettyCb* jettyCb, struct RsCtxJfcCb* sendJfcCb, struct RsCtxJfcCb* recvJfcCb);
extern int RsUbFillJettyInfo(struct RsCtxJettyCb* jettyCb, struct QpCreateInfo* jettyInfo);
extern void RsUbCtxDrvJettyDelete(struct RsCtxJettyCb* jettyCb);
extern int
RsUbCtxInitRjettyCb(struct RsUbDevCb* devCb, struct RsJettyImportAttr* importAttr, struct RsCtxRemJettyCb** rjettyCb);
extern int RsUbCtxDrvJettyImport(struct RsCtxRemJettyCb* rjettyCb);
extern int RsUbDevShareJfrInit(struct RsUbDevCb* devCb);
extern int RsUbFreeRemJettyCb(struct RsUbDevCb* devCb, struct RsCtxRemJettyCb* rjettyCb);
extern int RsUbAsyncEventGetJettyContext(
    urma_async_event_t* event, struct RsUbDevCb* devCb, struct RsCtxAsyncEventCb* asyncEventCb);
extern int RsUbGetDevEidInfoTotal(struct HccpDevEidInfo** arr, unsigned int* num);
extern int RsUbGetJettyCb(struct RsUbDevCb* devCb, unsigned int jettyId, struct RsCtxJettyCb** jettyCb);
extern void RsCloseUrmaSo(void);
extern int RsUbDestroyJettyCbBatch(struct JettyDestroyBatchInfo* batchInfo, unsigned int* num);
extern int RsUbGetJettyDestroyBatchInfo(
    struct RsUbDevCb* devCb, unsigned int jettyIds[], struct JettyDestroyBatchInfo* batchInfo, unsigned int* num);
extern int RsUbCallocJettyBatchInfo(struct JettyDestroyBatchInfo* batchInfo, unsigned int num);
extern void RsUbFreeJettyCbBatch(
    struct JettyDestroyBatchInfo* batchInfo, unsigned int* num, urma_jetty_t* badJetty, urma_jfr_t* badJfr);
extern int RsUbCtxJfcCreateNormal(struct RsUbDevCb* devCb, urma_jfc_cfg_t* jfcCfg, urma_jfc_t** outJfc);
extern int RsUbGetJfceCb(struct RsUbDevCb* devCb, unsigned long long addr, struct RsCtxJfceCb** jfceCb);
extern int RsHandleEpollPollJfc(struct RsUbDevCb* devCb, urma_jfce_t* jfce);
extern int RsUbJettyCbBuffAlloc(struct RsUbDevCb* devCb, struct RsCtxJettyCb* jettyCb, enum JfcMode jfcType);
extern int RsUbFreeSegCb(struct RsUbDevCb* devCb, struct RsSegCb* segCb);
extern int
RsUbGetTpInfoList(struct RsUbDevCb* devCb, struct GetTpCfg* cfg, struct HccpTpInfo infoList[], unsigned int* num);
extern int RsUbDevShareJfrInit(struct RsUbDevCb* devCb);
extern int RsUbFreeJfceCb(struct RsUbDevCb* devCb, struct RsCtxJfceCb* jfceCb);
extern int
RsUbAsyncEventGetJfcContext(urma_async_event_t* event, struct RsUbDevCb* devCb, struct RsCtxAsyncEventCb* asyncEventCb);
extern int RsUbGetSaveAsyncEvent(struct RsUbDevCb* devCb);

struct RsConnInfo gConn = {0};
char gRevBuf[RS_BUF_SIZE] = {0};
extern struct rs_cb stubRsCb;
extern struct RsUbDevCb stubDevCb;
struct RsCtxJettyCb jettyCbStub = {0};
struct RsCtxJfceCb gJfceCb = {0};

int RsUbGetJfceCbStub(struct RsUbDevCb* devCb, unsigned long long addr, struct RsCtxJfceCb** jfceCb)
{
    *jfceCb = &gJfceCb;
    return 0;
}

int RsUbGetJfceCbDynStub(struct RsUbDevCb* devCb, unsigned long long addr, struct RsCtxJfceCb** jfceCb)
{
    struct RsCtxJfceCb* dynCb = (struct RsCtxJfceCb*)calloc(1, sizeof(struct RsCtxJfceCb));
    if (dynCb == NULL) {
        return -ENOMEM;
    }
    RS_INIT_LIST_HEAD(&dynCb->list);
    dynCb->devCb = devCb;
    dynCb->jfceAddr = gJfceCb.jfceAddr;
    dynCb->dataPlaneFlag.bs.pollCqCstm = 1;
    *jfceCb = dynCb;
    return 0;
}

int RsUbGetJettyCbStub(struct RsUbDevCb* devCb, unsigned int jettyId, struct RsCtxJettyCb** jettyCb)
{
    *jettyCb = &jettyCbStub;
    return 0;
}

void RsUbFreeJettyCbBatchStub(
    struct JettyDestroyBatchInfo* batchInfo, unsigned int* num, urma_jetty_t* badJetty, urma_jfr_t* badJfr)
{
    return;
}

void TcRsUbGetRdevCb()
{
    struct RsUbDevCb* rdevCbOut = NULL;
    struct RsUbDevCb rdevCb = {0};
    unsigned int rdevIndex = 0;
    struct rs_cb rsCb = {0};
    int ret = 0;

    RS_INIT_LIST_HEAD(&rsCb.udevList);
    RsListAddTail(&rdevCb.list, &rsCb.udevList);

    ret = RsUbGetDevCb(&rsCb, rdevIndex, &rdevCbOut);
    EXPECT_INT_EQ(ret, 0);

    rdevIndex = 1;
    ret = RsUbGetDevCb(&rsCb, rdevIndex, &rdevCbOut);
    EXPECT_INT_EQ(ret, -ENODEV);

    return;
}

void TcRsUrmaApiInitAbnormal()
{
    int ret = 0;

    mocker(RsOpenUrmaSo, 100, 0);
    mocker(RsUrmaDeviceApiInit, 100, -1);
    ret = RsUrmaApiInit();
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker(RsOpenUrmaSo, 100, 0);
    mocker(RsUrmaDeviceApiInit, 100, 0);
    mocker(RsUrmaJettyApiInit, 100, -1);
    ret = RsUrmaApiInit();
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker(RsOpenUrmaSo, 100, 0);
    mocker(RsUrmaDeviceApiInit, 100, 0);
    mocker(RsUrmaJettyApiInit, 100, 0);
    mocker(RsUrmaJfcApiInit, 100, -1);
    ret = RsUrmaApiInit();
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker(RsOpenUrmaSo, 100, 0);
    mocker(RsUrmaDeviceApiInit, 100, 0);
    mocker(RsUrmaJettyApiInit, 100, 0);
    mocker(RsUrmaJfcApiInit, 100, 0);
    mocker(RsUrmaSegmentApiInit, 100, -1);
    ret = RsUrmaApiInit();
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker(RsOpenUrmaSo, 100, 0);
    mocker(RsUrmaDeviceApiInit, 100, 0);
    mocker(RsUrmaJettyApiInit, 100, 0);
    mocker(RsUrmaJfcApiInit, 100, 0);
    mocker(RsUrmaSegmentApiInit, 100, 0);
    mocker(RsUrmaDataApiInit, 100, -1);
    ret = RsUrmaApiInit();
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    return;
}

void TcRsUbV2()
{
    struct RsUbDevCb* devCb = NULL;
    struct DevBaseAttr attr = {0};
    struct RsInitConfig cfg = {0};
    struct CtxInitAttr info = {0};
    struct rs_cb rscb = {0};
    unsigned long long tokenIdAddr = 0;
    unsigned int tokenIdNum = 0;
    unsigned int devIndex = 0;
    int ret = 0;

    struct MemRegAttrT lmemAttr = {0};
    struct MemRegInfoT lmemInfo = {0};
    struct MemImportAttrT rmemAttr = {0};
    struct MemImportInfoT rmemInfo = {0};
    void* addr = malloc(1);
    lmemAttr.mem.addr = (uintptr_t)addr;
    lmemAttr.mem.size = 1;
    lmemAttr.ub.flags.bs.tokenIdValid = 1;

    cfg.chipId = 0;
    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    RS_INIT_LIST_HEAD(&rscb.udevList);
    ret = RsUbCtxInit(&rscb, &info, &devIndex, &attr);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbGetDevCb(&rscb, devIndex, &devCb);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxTokenIdAlloc(devCb, &tokenIdAddr, &tokenIdNum);
    EXPECT_INT_EQ(0, ret);

    lmemAttr.ub.tokenIdAddr = tokenIdAddr;

    ret = RsUbCtxLmemReg(devCb, &lmemAttr, &lmemInfo);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxRmemImport(devCb, &rmemAttr, &rmemInfo);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxRmemUnimport(devCb, rmemInfo.ub.targetSegHandle);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxLmemUnreg(devCb, lmemInfo.ub.targetSegHandle);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxDeinit(devCb);
    EXPECT_INT_EQ(0, ret);

    free(addr);
    addr = NULL;

    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);
}

urma_device_t tcUrmaDev = {0};
urma_device_t* tcUrmaDeviceList[1] = {&tcUrmaDev};
urma_eid_info_t tcUrmaEidList[1] = {0};

urma_device_t** TcRsUrmaGetDeviceListStub(int* numDevices)
{
    *numDevices = 1;
    return tcUrmaDeviceList;
}

urma_eid_info_t* TcRsUrmaGetEidListStub(urma_device_t* dev, uint32_t* cnt)
{
    *cnt = 1;
    return tcUrmaEidList;
}

urma_eid_info_t* TcRsUrmaGetEidListNullStub(urma_device_t* dev, uint32_t* cnt)
{
    *cnt = 0;
    return NULL;
}

urma_context_t* TcRsUrmaCreateContextStub(urma_device_t* dev, uint32_t eidIndex) { return (urma_context_t*)0x1; }

urma_device_t tcUrmaDev2[2] = {0};
urma_device_t* tcUrmaDeviceList2[2] = {&tcUrmaDev2[0], &tcUrmaDev2[1]};
urma_eid_info_t tcUrmaEidListDev0[1] = {{.eid_index = 0}};
urma_eid_info_t tcUrmaEidListDev1[2] = {{.eid_index = 0}, {.eid_index = 1}};

urma_device_t** TcRsUrmaGetDeviceList2Stub(int* numDevices)
{
    *numDevices = 2;
    return tcUrmaDeviceList2;
}

urma_eid_info_t* TcRsUrmaGetEidList2Stub(urma_device_t* dev, uint32_t* cnt)
{
    if (dev == &tcUrmaDev2[0]) {
        *cnt = 1;
        return tcUrmaEidListDev0;
    }
    if (dev == &tcUrmaDev2[1]) {
        *cnt = 2;
        return tcUrmaEidListDev1;
    }
    *cnt = 0;
    return NULL;
}

void TcRsInitDevEidInfoCbDevListNull()
{
    struct rs_cb rscb = {0};
    int ret;

    mocker_clean();
    mocker(RsGetProductType, 100, PRODUCT_TYPE_950);
    errno = 0;
    mocker(RsUrmaGetDeviceList, 1, NULL);
    ret = RsInitDevEidInfoCb(&rscb);
    EXPECT_INT_EQ(ret, -EOPENSRC);
    EXPECT_ADDR_EQ(rscb.devEidInfoCb, NULL);
    mocker_clean();
}

void TcRsInitDevEidInfoCbAlreadyInit()
{
    struct RsDevEidInfoCb* eidInfoCb = NULL;
    struct rs_cb rscb = {0};
    int ret;

    mocker_clean();
    eidInfoCb = calloc(1, sizeof(struct RsDevEidInfoCb));
    rscb.devEidInfoCb = eidInfoCb;

    ret = RsInitDevEidInfoCb(&rscb);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_ADDR_EQ(rscb.devEidInfoCb, eidInfoCb);

    free(eidInfoCb);
    mocker_clean();
}

void TcRsInitDevEidInfoCbSuccess()
{
    struct rs_cb rscb = {0};
    int ret;

    mocker_clean();
    mocker(RsGetProductType, 100, PRODUCT_TYPE_950);
    rscb.hccpMode = NETWORK_OFFLINE;
    mocker_u64_invoke((stub_u64_fn_t)RsUrmaGetDeviceList, (stub_u64_fn_t)TcRsUrmaGetDeviceListStub, 10);
    mocker_u64_invoke((stub_u64_fn_t)RsUrmaGetEidList, (stub_u64_fn_t)TcRsUrmaGetEidListStub, 10);
    mocker(RsUrmaQueryDevice, 10, 0);
    mocker_u64_invoke((stub_u64_fn_t)RsUrmaCreateContext, (stub_u64_fn_t)TcRsUrmaCreateContextStub, 10);
    mocker(RsUrmaUserCtl, 10, 0);
    mocker(RsUrmaDeleteContext, 10, 0);
    mocker(RsUrmaFreeEidList, 10, 0);
    mocker(RsUrmaFreeDeviceList, 10, 0);

    ret = RsInitDevEidInfoCb(&rscb);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_ADDR_NE(rscb.devEidInfoCb, NULL);
    if (rscb.devEidInfoCb != NULL) {
        EXPECT_INT_EQ(((struct RsDevEidInfoCb*)rscb.devEidInfoCb)->num, 1);
    }

    RsDeinitDevEidInfoCb(&rscb);
    mocker_clean();
}

void TcRsInitDevEidInfoCbSuccessNonOffline()
{
    struct rs_cb rscb = {0};
    int ret;

    mocker_clean();
    mocker(RsGetProductType, 100, PRODUCT_TYPE_950);
    rscb.hccpMode = NETWORK_PEER_ONLINE;
    mocker(RsUbApiInit, 1, 0);
    mocker(RsUbApiDeinit, 1, 0);
    mocker_u64_invoke((stub_u64_fn_t)RsUrmaGetDeviceList, (stub_u64_fn_t)TcRsUrmaGetDeviceListStub, 10);
    mocker_u64_invoke((stub_u64_fn_t)RsUrmaGetEidList, (stub_u64_fn_t)TcRsUrmaGetEidListStub, 10);
    mocker(RsUrmaQueryDevice, 10, 0);
    mocker_u64_invoke((stub_u64_fn_t)RsUrmaCreateContext, (stub_u64_fn_t)TcRsUrmaCreateContextStub, 10);
    mocker(RsUrmaUserCtl, 10, 0);
    mocker(RsUrmaDeleteContext, 10, 0);
    mocker(RsUrmaFreeEidList, 10, 0);
    mocker(RsUrmaFreeDeviceList, 10, 0);

    ret = RsInitDevEidInfoCb(&rscb);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_ADDR_NE(rscb.devEidInfoCb, NULL);
    if (rscb.devEidInfoCb != NULL) {
        EXPECT_INT_EQ(((struct RsDevEidInfoCb*)rscb.devEidInfoCb)->num, 1);
    }

    RsDeinitDevEidInfoCb(&rscb);
    mocker_clean();
}

void TcRsDeinitDevEidInfoCb()
{
    struct RsDevEidInfoCb* eidCb = NULL;
    struct HccpDevEidInfo* eidArr = NULL;
    struct rs_cb rscb = {0};

    eidCb = calloc(1, sizeof(struct RsDevEidInfoCb));
    eidArr = calloc(1, sizeof(struct HccpDevEidInfo));
    eidArr[0].eidIndex = 0;
    eidCb->infoList = eidArr;
    eidCb->num = 1;
    rscb.devEidInfoCb = eidCb;

    RsDeinitDevEidInfoCb(&rscb);
    EXPECT_ADDR_EQ(rscb.devEidInfoCb, NULL);
}

void TcRsUbGetDevEidInfoTotalSuccess()
{
    struct HccpDevEidInfo* arr = NULL;
    unsigned int num = 0;
    int ret;

    mocker_clean();
    mocker_u64_invoke((stub_u64_fn_t)RsUrmaGetDeviceList, (stub_u64_fn_t)TcRsUrmaGetDeviceListStub, 10);
    mocker_u64_invoke((stub_u64_fn_t)RsUrmaGetEidList, (stub_u64_fn_t)TcRsUrmaGetEidListStub, 10);
    mocker(RsUrmaQueryDevice, 10, 0);
    mocker_u64_invoke((stub_u64_fn_t)RsUrmaCreateContext, (stub_u64_fn_t)TcRsUrmaCreateContextStub, 10);
    mocker(RsUrmaUserCtl, 10, 0);
    mocker(RsUrmaDeleteContext, 10, 0);
    mocker(RsUrmaFreeEidList, 10, 0);
    mocker(RsUrmaFreeDeviceList, 10, 0);

    ret = RsUbGetDevEidInfoTotal(&arr, &num);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_INT_EQ(num, 1);
    EXPECT_ADDR_NE(arr, NULL);

    free(arr);
    mocker_clean();
}

void TcRsUbGetDevEidInfoTotalMultiDev()
{
    struct HccpDevEidInfo* arr = NULL;
    unsigned int num = 0;
    int ret;

    mocker_clean();
    mocker_u64_invoke((stub_u64_fn_t)RsUrmaGetDeviceList, (stub_u64_fn_t)TcRsUrmaGetDeviceList2Stub, 10);
    mocker_u64_invoke((stub_u64_fn_t)RsUrmaGetEidList, (stub_u64_fn_t)TcRsUrmaGetEidList2Stub, 10);
    mocker(RsUrmaQueryDevice, 10, 0);
    mocker_u64_invoke((stub_u64_fn_t)RsUrmaCreateContext, (stub_u64_fn_t)TcRsUrmaCreateContextStub, 10);
    mocker(RsUrmaUserCtl, 10, 0);
    mocker(RsUrmaDeleteContext, 10, 0);
    mocker(RsUrmaFreeEidList, 10, 0);
    mocker(RsUrmaFreeDeviceList, 10, 0);

    ret = RsUbGetDevEidInfoTotal(&arr, &num);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_INT_EQ(num, 3);
    EXPECT_ADDR_NE(arr, NULL);
    EXPECT_INT_EQ(arr[0].eidIndex, 0);
    EXPECT_INT_EQ(arr[1].eidIndex, 0);
    EXPECT_INT_EQ(arr[2].eidIndex, 1);

    free(arr);
    mocker_clean();
}

void TcRsUbGetDevEidInfoTotalEidListNull()
{
    struct HccpDevEidInfo* arr = NULL;
    unsigned int num = 0;
    int ret;

    mocker_clean();
    mocker_u64_invoke((stub_u64_fn_t)RsUrmaGetDeviceList, (stub_u64_fn_t)TcRsUrmaGetDeviceListStub, 10);
    mocker_u64_invoke((stub_u64_fn_t)RsUrmaGetEidList, (stub_u64_fn_t)TcRsUrmaGetEidListNullStub, 10);
    mocker(RsUrmaFreeDeviceList, 10, 0);

    ret = RsUbGetDevEidInfoTotal(&arr, &num);
    EXPECT_INT_EQ(ret, 0);
    EXPECT_INT_EQ(num, 0);
    EXPECT_ADDR_EQ(arr, NULL);

    mocker_clean();
}

struct rs_cb* TcRsUbV2Init(int mode, unsigned int* devIndex)
{
    int ret = 0;
    struct DevBaseAttr attr = {0};
    struct RsInitConfig cfg = {0};
    struct CtxInitAttr info = {0};
    struct rs_cb* rsCb = NULL;
    cfg.hccpMode = mode;

    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);
    ret = RsGetRsCb(0, &rsCb);
    EXPECT_INT_EQ(ret, 0);
    RS_INIT_LIST_HEAD(&rsCb->udevList);

    ret = RsUbCtxInit(rsCb, &info, devIndex, &attr);
    EXPECT_INT_EQ(0, ret);

    return rsCb;
}

void TcRsUbV2Deinit(struct rs_cb* rsCb, int mode, unsigned int devIndex)
{
    int ret = 0;
    struct RsInitConfig cfg = {0};
    cfg.hccpMode = mode;
    struct RsUbDevCb* devCb = NULL;

    ret = RsUbGetDevCb(rsCb, devIndex, &devCb);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxDeinit(devCb);
    EXPECT_INT_EQ(ret, 0);
    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRsUbCtxTokenIdAlloc()
{
    TcRsUbCtxTokenIdAlloc1();
    TcRsUbCtxTokenIdAlloc2();
    TcRsUbCtxTokenIdAlloc3();
}

void TcRsUbCtxTokenIdAlloc1()
{
    unsigned long long addr = 0;
    unsigned int devIndex = 0;
    unsigned int tokenId = 0;
    struct rs_cb* tcRsCb = NULL;
    struct RsUbDevCb* devCb = NULL;
    int ret = 0;

    tcRsCb = TcRsUbV2Init(NETWORK_OFFLINE, &devIndex);

    ret = RsUbGetDevCb(tcRsCb, devIndex, &devCb);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxTokenIdAlloc(devCb, &addr, &tokenId);
    EXPECT_INT_EQ(0, ret);

    mocker(RsUrmaFreeTokenId, 1, -1);
    ret = RsUbCtxTokenIdFree(devCb, addr);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    TcRsUbV2Deinit(tcRsCb, NETWORK_OFFLINE, devIndex);
}

void TcRsUbCtxTokenIdAlloc2()
{
    unsigned long long addr = 0;
    unsigned int devIndex = 0;
    unsigned int tokenId = 0;
    struct rs_cb* tcRsCb = NULL;
    struct RsUbDevCb* devCb = NULL;
    int ret = 0;

    tcRsCb = TcRsUbV2Init(NETWORK_OFFLINE, &devIndex);

    ret = RsUbGetDevCb(tcRsCb, devIndex, &devCb);
    EXPECT_INT_EQ(0, ret);

    mocker(RsUrmaAllocTokenId, 10, NULL);
    ret = RsUbCtxTokenIdAlloc(devCb, &addr, &tokenId);
    EXPECT_INT_NE(0, ret);

    TcRsUbV2Deinit(tcRsCb, NETWORK_OFFLINE, devIndex);
}

void TcRsUbCtxTokenIdAlloc3()
{
    unsigned long long addr = 0;
    unsigned long long addr1 = 0;
    unsigned int devIndex = 0;
    unsigned int tokenId = 0;
    unsigned int tokenId1 = 0;
    struct rs_cb* tcRsCb = NULL;
    struct RsUbDevCb* devCb = NULL;
    int ret = 0;

    tcRsCb = TcRsUbV2Init(NETWORK_OFFLINE, &devIndex);

    ret = RsUbGetDevCb(tcRsCb, devIndex, &devCb);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxTokenIdAlloc(devCb, &addr, &tokenId);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxTokenIdAlloc(devCb, &addr1, &tokenId1);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxTokenIdFree(devCb, addr1);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxTokenIdFree(devCb, addr);
    EXPECT_INT_EQ(0, ret);

    TcRsUbV2Deinit(tcRsCb, NETWORK_OFFLINE, devIndex);
}

void TcRsUbCtxJfceCreate()
{
    union DataPlaneCstmFlag dataPlaneFlag = {0};
    struct RsUbDevCb* devCb = NULL;
    unsigned long long addr = 0;
    unsigned int devIndex = 0;
    struct rs_cb* tcRsCb = NULL;
    int fd = 0;
    int ret = 0;

    mocker_clean();
    tcRsCb = TcRsUbV2Init(NETWORK_OFFLINE, &devIndex);

    ret = RsUbGetDevCb(tcRsCb, devIndex, &devCb);
    EXPECT_INT_EQ(0, ret);

    dataPlaneFlag.bs.pollCqCstm = 1;
    ret = RsUbCtxChanCreate(devCb, dataPlaneFlag, &addr, &fd);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxChanDestroy(devCb, addr);
    EXPECT_INT_EQ(0, ret);

    TcRsUbV2Deinit(tcRsCb, NETWORK_OFFLINE, devIndex);
}

void TcRsUbCtxJfcCreate()
{
    int ret = 0;
    unsigned int devIndex = 0;
    struct rs_cb* tcRsCb = NULL;
    struct CtxCqAttr attr = {0};
    struct CtxCqInfo info = {0};
    struct RsUbDevCb* devCb = NULL;

    tcRsCb = TcRsUbV2Init(NETWORK_OFFLINE, &devIndex);
    attr.ub.mode = JFC_MODE_STARS_POLL;

    ret = RsUbGetDevCb(tcRsCb, devIndex, &devCb);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJfcCreate(devCb, &attr, &info);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJfcDestroy(devCb, info.addr);
    EXPECT_INT_EQ(0, ret);

    attr.ub.mode = JFC_MODE_CCU_POLL;
    attr.ub.ccuExCfg.valid = 1;

    ret = RsUbCtxJfcCreate(devCb, &attr, &info);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJfcDestroy(devCb, info.addr);
    EXPECT_INT_EQ(0, ret);

    attr.ub.mode = JFC_MODE_MAX;
    ret = RsUbCtxJfcCreate(devCb, &attr, &info);
    EXPECT_INT_EQ(-EINVAL, ret);

    mocker_clean();
    attr.ub.mode = JFC_MODE_NORMAL;
    mocker(RsUbCtxJfcCreateNormal, 1, 0);
    ret = RsUbCtxJfcCreate(devCb, &attr, &info);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();

    mocker(RsUbCtxJfcCreateNormal, 1, -1);
    ret = RsUbCtxJfcCreate(devCb, &attr, &info);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    TcRsUbV2Deinit(tcRsCb, NETWORK_OFFLINE, devIndex);
}

void TcRsUbCtxJfcCreateNormal()
{
    struct RsUbDevCb devCb = {0};
    urma_jfc_cfg_t jfcCfg = {0};
    urma_jfc_t* outJfc = NULL;
    urma_jfce_t jfce = {0};
    int ret = 0;

    gJfceCb.jfceAddr = 1;
    jfcCfg.jfce = (urma_jfce_t*)(uintptr_t)gJfceCb.jfceAddr;
    mocker_clean();
    mocker(RsUbGetJfceCb, 1, -1);
    ret = RsUbCtxJfcCreateNormal(&devCb, &jfcCfg, &outJfc);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    mocker_invoke(RsUbGetJfceCb, RsUbGetJfceCbStub, 1);
    mocker(RsUrmaCreateJfc, 1, NULL);
    ret = RsUbCtxJfcCreateNormal(&devCb, &jfcCfg, &outJfc);
    EXPECT_INT_NE(0, ret);
    mocker_clean();

    mocker_invoke(RsUbGetJfceCb, RsUbGetJfceCbStub, 1);
    mocker(RsUrmaRearmJfc, 1, -1);
    ret = RsUbCtxJfcCreateNormal(&devCb, &jfcCfg, &outJfc);
    EXPECT_INT_EQ(-1, ret);
    RsUrmaDeleteJfc(outJfc);
    mocker_clean();

    mocker_invoke(RsUbGetJfceCb, RsUbGetJfceCbStub, 1);
    ret = RsUbCtxJfcCreateNormal(&devCb, &jfcCfg, &outJfc);
    EXPECT_INT_EQ(0, ret);
    RsUrmaDeleteJfc(outJfc);
    mocker_clean();
}

void TcRsUbCtxJettyCreate()
{
    int ret = 0;
    unsigned int devIndex = 0;
    struct rs_cb* tcRsCb = NULL;
    struct CtxQpAttr qpAttr = {0};
    struct QpCreateInfo qpInfo = {0};
    struct CtxCqAttr cqAttr = {0};
    struct CtxCqInfo cqInfo = {0};
    struct RsUbDevCb* devCb = NULL;
    unsigned long long tokenIdAddr = 0;
    unsigned int tokenIdNum = 0;

    tcRsCb = TcRsUbV2Init(NETWORK_OFFLINE, &devIndex);
    cqAttr.ub.mode = JFC_MODE_STARS_POLL;

    ret = RsUbGetDevCb(tcRsCb, devIndex, &devCb);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJfcCreate(devCb, &cqAttr, &cqInfo);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJettyCreate(devCb, &qpAttr, &qpInfo);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJettyDestroy(devCb, qpInfo.ub.id);
    EXPECT_INT_EQ(0, ret);

    qpAttr.ub.mode = JETTY_MODE_CCU;
    ret = RsUbCtxTokenIdAlloc(devCb, &tokenIdAddr, &tokenIdNum);
    EXPECT_INT_EQ(0, ret);
    qpAttr.ub.tokenIdAddr = tokenIdAddr;
    ret = RsUbCtxJettyCreate(devCb, &qpAttr, &qpInfo);
    EXPECT_INT_NE(0, ret);

    ret = RsUbCtxJettyDestroy(devCb, qpInfo.ub.id);
    EXPECT_INT_NE(0, ret);

    ret = RsUbCtxJfcDestroy(devCb, cqInfo.addr);
    EXPECT_INT_EQ(0, ret);

    cqAttr.ub.mode = JFC_MODE_CCU_POLL;
    cqAttr.ub.ccuExCfg.valid = 1;
    ret = RsUbCtxJfcCreate(devCb, &cqAttr, &cqInfo);
    EXPECT_INT_EQ(0, ret);

    qpAttr.ub.mode = JETTY_MODE_CCU_TA_CACHE;
    ret = RsUbCtxTokenIdAlloc(devCb, &tokenIdAddr, &tokenIdNum);
    EXPECT_INT_EQ(0, ret);
    qpAttr.ub.tokenIdAddr = tokenIdAddr;
    ret = RsUbCtxJettyCreate(devCb, &qpAttr, &qpInfo);
    EXPECT_INT_NE(0, ret);

    ret = RsUbCtxJettyDestroy(devCb, qpInfo.ub.id);
    EXPECT_INT_NE(0, ret);

    ret = RsUbCtxJfcDestroy(devCb, cqInfo.addr);
    EXPECT_INT_EQ(0, ret);

    qpAttr.ub.mode = JETTY_MODE_MAX;
    ret = RsUbCtxJettyCreate(devCb, &qpAttr, &qpInfo);
    EXPECT_INT_EQ(-EINVAL, ret);

    qpAttr.ub.mode = JETTY_MODE_CCU_TA_CACHE;
    qpAttr.ub.taCacheMode.lockFlag = 0;
    ret = RsUbCtxJettyCreate(devCb, &qpAttr, &qpInfo);
    EXPECT_INT_EQ(-EINVAL, ret);

    TcRsUbV2Deinit(tcRsCb, NETWORK_OFFLINE, devIndex);
}

void TcRsUbCtxJettyImport()
{
    int ret = 0;
    unsigned int devIndex = 0;
    struct rs_cb* tcRsCb = NULL;
    struct RsUbDevCb* devCb = NULL;
    struct CtxQpAttr qpAttr = {0};
    struct QpCreateInfo qpInfo = {0};
    struct CtxCqAttr cqAttr = {0};
    struct CtxCqInfo cqInfo = {0};
    struct RsJettyImportAttr importAttr = {0};
    struct RsJettyImportInfo importData = {0};

    tcRsCb = TcRsUbV2Init(NETWORK_OFFLINE, &devIndex);
    cqAttr.ub.mode = JFC_MODE_STARS_POLL;

    ret = RsUbGetDevCb(tcRsCb, devIndex, &devCb);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJfcCreate(devCb, &cqAttr, &cqInfo);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJettyCreate(devCb, &qpAttr, &qpInfo);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJettyImport(devCb, &importAttr, &importData);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJettyUnimport(devCb, importAttr.key.value, REM_JETTY_ID_DEPRECATED_SIZE);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJettyDestroy(devCb, qpInfo.ub.id);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJfcDestroy(devCb, cqInfo.addr);
    EXPECT_INT_EQ(0, ret);

    TcRsUbV2Deinit(tcRsCb, NETWORK_OFFLINE, devIndex);
}

void TcRsUbCtxJettyBind()
{
    int ret = 0;
    unsigned int devIndex = 0;
    struct rs_cb* tcRsCb = NULL;
    struct RsUbDevCb* devCb = NULL;
    struct CtxQpAttr qpAttr = {0};
    struct QpCreateInfo qpInfo = {0};
    struct CtxCqAttr cqAttr = {0};
    struct CtxCqInfo cqInfo = {0};
    struct RsJettyImportAttr importAttr = {0};
    struct RsJettyImportInfo importData = {0};
    struct RsCtxQpInfo rsQpInfo = {0};

    tcRsCb = TcRsUbV2Init(NETWORK_OFFLINE, &devIndex);
    cqAttr.ub.mode = JFC_MODE_STARS_POLL;

    ret = RsUbGetDevCb(tcRsCb, devIndex, &devCb);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJfcCreate(devCb, &cqAttr, &cqInfo);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJettyCreate(devCb, &qpAttr, &qpInfo);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJettyUnbind(devCb, rsQpInfo.id);
    EXPECT_INT_NE(0, ret);

    ret = RsUbCtxJettyImport(devCb, &importAttr, &importData);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJettyBind(devCb, &rsQpInfo, &rsQpInfo);
    EXPECT_INT_EQ(0, ret);

    mocker(RsUrmaUnbindJetty, 1, -URMA_FAIL);
    errno = 0;
    ret = RsUbCtxJettyUnbind(devCb, rsQpInfo.id);
    EXPECT_INT_EQ(-EOPENSRC, ret);
    mocker_clean();

    ret = RsUbCtxJettyBind(devCb, &rsQpInfo, &rsQpInfo);
    EXPECT_INT_NE(0, ret);

    ret = RsUbCtxJettyDestroy(devCb, qpInfo.ub.id);
    EXPECT_INT_NE(0, ret);

    ret = RsUbCtxJettyUnbind(devCb, rsQpInfo.id);
    EXPECT_INT_EQ(0, ret);

    mocker(RsUrmaBindJetty, 1, -URMA_FAIL);
    errno = 0;
    ret = RsUbCtxJettyBind(devCb, &rsQpInfo, &rsQpInfo);
    EXPECT_INT_EQ(-EOPENSRC, ret);
    mocker_clean();

    errno = EINVAL;
    mocker(RsUrmaBindJetty, 1, -URMA_FAIL);
    ret = RsUbCtxJettyBind(devCb, &rsQpInfo, &rsQpInfo);
    EXPECT_INT_EQ(-EINVAL, ret);
    mocker_clean();

    ret = RsUbCtxJettyUnimport(devCb, importAttr.key.value, REM_JETTY_ID_DEPRECATED_SIZE);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJettyDestroy(devCb, qpInfo.ub.id);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJettyCreate(devCb, &qpAttr, &qpInfo);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJettyImport(devCb, &importAttr, &importData);
    EXPECT_INT_EQ(0, ret);

    TcRsUbV2Deinit(tcRsCb, NETWORK_OFFLINE, devIndex);

    tcRsCb = TcRsUbV2Init(NETWORK_OFFLINE, &devIndex);
    cqAttr.ub.mode = 10000;

    ret = RsUbGetDevCb(tcRsCb, devIndex, &devCb);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJfcCreate(devCb, &cqAttr, &cqInfo);
    EXPECT_INT_NE(0, ret);

    cqAttr.ub.mode = JFC_MODE_STARS_POLL;
    ret = RsUbCtxJfcCreate(devCb, &cqAttr, &cqInfo);
    EXPECT_INT_EQ(0, ret);

    qpAttr.ub.mode = JFC_MODE_STARS_POLL;
    ret = RsUbCtxJettyCreate(devCb, &qpAttr, &qpInfo);
    EXPECT_INT_NE(0, ret);

    ret = RsUbCtxJettyImport(devCb, &importAttr, &importData);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJettyUnimport(devCb, importAttr.key.value, REM_JETTY_ID_DEPRECATED_SIZE);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJfcDestroy(devCb, cqInfo.addr);
    EXPECT_INT_EQ(0, ret);

    struct RsCtxJettyCb jettyCb = {0};
    urma_jetty_t jetty = {0};
    jettyCb.jetty = &jetty;
    jettyCb.devCb = devCb;
    RsUbCtxExtJettyDelete(&jettyCb);

    TcRsUbV2Deinit(tcRsCb, NETWORK_OFFLINE, devIndex);
}

void TcRsUbCtxBatchSendWr()
{
    int ret = 0;
    unsigned int devIndex = 0;
    struct rs_cb* tcRsCb = NULL;
    struct CtxQpAttr qpAttr = {0};
    struct QpCreateInfo qpInfo = {0};
    struct CtxCqAttr cqAttr = {0};
    struct CtxCqInfo cqInfo = {0};
    struct RsJettyImportAttr importAttr = {0};
    struct RsJettyImportInfo importData = {0};
    struct RsCtxQpInfo rsQpInfo = {0};
    struct WrlistBaseInfo baseInfo = {0};
    struct BatchSendWrData wrDataNop[1] = {0};
    struct BatchSendWrData wrData[1] = {0};
    struct SendWrResp wrResp[1] = {0};
    struct WrlistSendCompleteNum wrlistNum = {0};
    struct MemRegAttrT memRegAttr = {0};
    struct MemRegInfoT memRegInfo = {0};
    struct MemImportAttrT memImportAttr = {0};
    struct MemImportInfoT memImportInfo = {0};
    urma_token_id_t* tokenId = NULL;
    unsigned long long tokenIdAddr = 0;
    unsigned int tokenIdNum = 0;
    unsigned int completeNum = 0;
    struct RsUbDevCb* devCb = NULL;

    tcRsCb = TcRsUbV2Init(NETWORK_OFFLINE, &devIndex);
    cqAttr.ub.mode = JFC_MODE_STARS_POLL;
    wrlistNum.sendNum = 1;
    wrlistNum.completeNum = &completeNum;
    void* addr = malloc(1);
    memRegAttr.mem.addr = (uintptr_t)addr;
    memRegAttr.mem.size = 1;
    memRegAttr.ub.flags.bs.tokenIdValid = 1;

    ret = RsUbGetDevCb(tcRsCb, devIndex, &devCb);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxTokenIdAlloc(devCb, &tokenIdAddr, &tokenIdNum);
    EXPECT_INT_EQ(0, ret);

    memRegAttr.ub.tokenIdAddr = tokenIdAddr;

    ret = RsUbCtxJfcCreate(devCb, &cqAttr, &cqInfo);
    EXPECT_INT_EQ(0, ret);

    qpAttr.ub.mode = JETTY_MODE_CCU;
    qpAttr.ub.tokenIdAddr = tokenIdAddr;
    ret = RsUbCtxJettyCreate(devCb, &qpAttr, &qpInfo);
    EXPECT_INT_NE(0, ret);

    qpAttr.ub.mode = 0;
    ret = RsUbCtxJettyCreate(devCb, &qpAttr, &qpInfo);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJettyImport(devCb, &importAttr, &importData);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJettyBind(devCb, &rsQpInfo, &rsQpInfo);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxLmemReg(devCb, &memRegAttr, &memRegInfo);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxRmemImport(devCb, &memImportAttr, &memImportInfo);
    EXPECT_INT_EQ(0, ret);

    wrData[0].devRmemHandle = (uint64_t)(uintptr_t)memImportInfo.ub.targetSegHandle;
    wrData[0].numSge = 1;
    wrData[0].sges[0].addr = (uint64_t)(uintptr_t)addr;
    wrData[0].sges[0].len = 1;
    wrData[0].sges[0].devLmemHandle = (uint64_t)(uintptr_t)memRegInfo.ub.targetSegHandle;

    baseInfo.devIndex = devIndex;
    ret = RsUbCtxBatchSendWr(tcRsCb, &baseInfo, wrData, wrResp, &wrlistNum);
    EXPECT_INT_EQ(0, ret);

    wrData[0].ub.remJetty = 0xfffff;
    ret = RsUbCtxBatchSendWr(tcRsCb, &baseInfo, wrData, wrResp, &wrlistNum);
    EXPECT_INT_NE(0, ret);

    wrDataNop[0].ub.opcode = RA_UB_OPC_NOP;
    ret = RsUbCtxBatchSendWr(tcRsCb, &baseInfo, wrDataNop, wrResp, &wrlistNum);
    EXPECT_INT_EQ(0, ret);

    mocker(RsUrmaUserCtl, 1, -1);
    ret = RsUbCtxBatchSendWr(tcRsCb, &baseInfo, wrDataNop, wrResp, &wrlistNum);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    ret = RsUbCtxRmemUnimport(devCb, memImportInfo.ub.targetSegHandle);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJettyUpdateCi(devCb, 10000, 0);
    EXPECT_INT_NE(0, ret);

    ret = RsUbCtxJettyUpdateCi(devCb, 0, 0);
    EXPECT_INT_EQ(0, ret);

    mocker(RsUrmaUserCtl, 1, -1);
    ret = RsUbCtxJettyUpdateCi(devCb, 0, 0);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    ret = RsUbCtxLmemUnreg(devCb, memRegInfo.ub.targetSegHandle);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxLmemReg(devCb, &memRegAttr, &memRegInfo);
    EXPECT_INT_EQ(0, ret);
    mocker(RsUrmaUnregisterSeg, 1, -1);
    ret = RsUbCtxLmemUnreg(devCb, memRegInfo.ub.targetSegHandle);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    ret = RsUbCtxJettyUnbind(devCb, rsQpInfo.id);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJettyUnimport(devCb, importAttr.key.value, REM_JETTY_ID_DEPRECATED_SIZE);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJettyDestroy(devCb, qpInfo.ub.id);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJfcDestroy(devCb, cqInfo.addr);
    EXPECT_INT_EQ(0, ret);

    free(addr);
    addr = NULL;
    TcRsUbV2Deinit(tcRsCb, NETWORK_OFFLINE, devIndex);
}

void TcRsUbFreeCbList()
{
    int ret = 0;
    unsigned int devIndex = 0;
    struct rs_cb* tcRsCb = NULL;
    struct CtxQpAttr qpAttr = {0};
    struct QpCreateInfo qpInfo = {0};
    struct CtxCqAttr cqAttr = {0};
    struct CtxCqInfo cqInfo = {0};
    struct RsJettyImportAttr importAttr = {0};
    struct RsJettyImportInfo importData = {0};
    struct RsCtxQpInfo rsQpInfo = {0};
    struct MemRegAttrT memRegAttr = {0};
    struct MemRegInfoT memRegInfo = {0};
    struct MemImportAttrT memImportAttr = {0};
    struct MemImportInfoT memImportInfo = {0};
    unsigned long long jfceAddr = 0;
    unsigned long long tokenIdAddr = 0;
    unsigned int tokenId = 0;
    unsigned int completeNum = 0;
    struct RsUbDevCb* devCb = NULL;
    union DataPlaneCstmFlag dataPlaneFlag = {0};
    int fd = 0;

    tcRsCb = TcRsUbV2Init(NETWORK_OFFLINE, &devIndex);
    cqAttr.ub.mode = JFC_MODE_STARS_POLL;
    void* addr = malloc(1);
    memRegAttr.mem.addr = (uintptr_t)addr;
    memRegAttr.mem.size = 1;
    memRegAttr.ub.flags.bs.tokenIdValid = 1;

    ret = RsUbGetDevCb(tcRsCb, devIndex, &devCb);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxTokenIdAlloc(devCb, &tokenIdAddr, &tokenId);
    EXPECT_INT_EQ(0, ret);

    memRegAttr.ub.tokenIdAddr = tokenIdAddr;

    ret = RsUbCtxJfcCreate(devCb, &cqAttr, &cqInfo);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJettyCreate(devCb, &qpAttr, &qpInfo);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJettyImport(devCb, &importAttr, &importData);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxJettyBind(devCb, &rsQpInfo, &rsQpInfo);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxLmemReg(devCb, &memRegAttr, &memRegInfo);
    EXPECT_INT_EQ(0, ret);

    ret = RsUbCtxRmemImport(devCb, &memImportAttr, &memImportInfo);
    EXPECT_INT_EQ(0, ret);

    dataPlaneFlag.bs.pollCqCstm = 1;
    ret = RsUbCtxChanCreate(devCb, dataPlaneFlag, &jfceAddr, &fd);
    EXPECT_INT_EQ(0, ret);

    free(addr);
    addr = NULL;
    TcRsUbV2Deinit(tcRsCb, NETWORK_OFFLINE, devIndex);
}

void TcRsUbCtxExtJettyCreate()
{
    struct RsCtxJettyCb jettyCb = {0};
    struct RsUbDevCb devCb = {0};
    urma_jetty_cfg_t jettyCfg = {0};
    struct rs_cb rscb = {0};

    devCb.rscb = &rscb;
    jettyCb.devCb = &devCb;
    jettyCb.jettyMode = JETTY_MODE_USER_CTL_NORMAL;
    RsUbCtxExtJettyCreate(&jettyCb, &jettyCfg);
    RsUbCtxExtJettyDelete(&jettyCb);
    mocker_clean();
}

void TcRsUbCtxRmemImport()
{
    struct MemImportAttrT rmemAttr = {0};
    struct MemImportInfoT rmemInfo = {0};
    struct RsUbDevCb devCb = {0};
    int ret = 0;

    mocker(RsUbGetDevCb, 2, 0);
    mocker(memcpy_s, 2, -1);
    ret = RsUbCtxRmemImport(&devCb, &rmemAttr, &rmemInfo);
    EXPECT_INT_EQ(-ESAFEFUNC, ret);
    mocker_clean();
}

void TcRsUbCtxDrvJettyImport()
{
    struct RsCtxRemJettyCb rjettyCb = {0};
    struct RsUbDevCb devCb = {0};
    urma_context_t urmaCtx = {0};
    int ret = 0;

    rjettyCb.mode = JETTY_IMPORT_MODE_EXP;
    devCb.urmaCtx = &urmaCtx;
    rjettyCb.devCb = &devCb;
    ret = RsUbCtxDrvJettyImport(&rjettyCb);
    EXPECT_INT_EQ(0, ret);

    free(rjettyCb.tjetty);
    rjettyCb.tjetty = NULL;

    mocker(RsUrmaImportJettyEx, 1, NULL);
    ret = RsUbCtxDrvJettyImport(&rjettyCb);
    EXPECT_INT_NE(0, ret);
    mocker_clean();
}

void TcRsUbDevCbInit()
{
    struct DevBaseAttr baseAttr = {0};
    struct RsUbDevCb devCb = {0};
    struct CtxInitAttr attr = {0};
    struct rs_cb rscb = {0};
    int devIndex = 0;
    int ret = 0;

    mocker(pthread_mutex_init, 1, 0);
    mocker(RsUbCreateCtx, 1, -1);
    mocker(pthread_mutex_destroy, 1, -1);
    ret = RsUbDevCbInit(&attr, &devCb, &rscb, &devIndex, &baseAttr);
    EXPECT_INT_NE(0, ret);

    mocker_clean();
    mocker(pthread_mutex_init, 1, 0);
    mocker(RsUbCreateCtx, 1, 0);
    mocker(RsUbGetDevAttr, 1, -1);
    mocker(RsUrmaDeleteContext, 1, -1);
    mocker(pthread_mutex_destroy, 1, -1);
    ret = RsUbDevCbInit(&attr, &devCb, &rscb, &devIndex, &baseAttr);
    EXPECT_INT_NE(0, ret);

    mocker_clean();
}

void TcRsUbCtxInit()
{
    struct DevBaseAttr baseAttr = {0};
    struct CtxInitAttr attr = {0};
    struct rs_cb rsCb = {0};
    int devIndex = 0;
    int ret = 0;

    mocker(RsUrmaGetDeviceByEid, 1, NULL);
    ret = RsUbCtxInit(&rsCb, &attr, &devIndex, &baseAttr);
    EXPECT_INT_NE(0, ret);
    mocker_clean();
}

void TcRsUbCtxJfcDestroy()
{
    struct RsUbDevCb devCb = {0};
    unsigned long long addr = 0;
    int ret = 0;

    mocker(RsUbGetJfcCb, 1, -1);
    ret = RsUbCtxJfcDestroy(&devCb, addr);
    EXPECT_INT_NE(0, ret);
    mocker_clean();
}

void TcRsUbCtxExtJettyDelete()
{
    struct RsCtxJettyCb jettyCb = {0};
    struct RsUbDevCb devCb = {0};
    urma_jetty_t* jetty = (urma_jetty_t*)calloc(1, sizeof(urma_jetty_t));
    jettyCb.jetty = jetty;

    jettyCb.devCb = &devCb;
    mocker(RsUrmaUserCtl, 1, -1);
    RsUbCtxExtJettyDelete(&jettyCb);
    mocker_clean();
}

void TcRsUbCtxChanCreate()
{
    union DataPlaneCstmFlag dataPlaneFlag = {0};
    struct RsUbDevCb devCb = {0};
    unsigned long long addr = 0;
    struct rs_cb rsCb = {0};
    int ret = 0;
    int fd = 0;

    devCb.rscb = &rsCb;
    dataPlaneFlag.bs.pollCqCstm = 1;
    mocker(RsUrmaCreateJfce, 1, NULL);
    ret = RsUbCtxChanCreate(&devCb, dataPlaneFlag, &addr, &fd);
    EXPECT_INT_NE(0, ret);
    mocker_clean();

    dataPlaneFlag.bs.pollCqCstm = 0;
    ret = RsUbCtxChanCreate(&devCb, dataPlaneFlag, &addr, &fd);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();
}

void TcRsUbCtxDeinit()
{
    struct RsUbDevCb* devCb = (struct RsUbDevCb*)calloc(1, sizeof(struct RsUbDevCb));
    struct rs_cb rsCb = {0};
    int ret = 0;

    devCb->rscb = &rsCb;
    mocker(RsUrmaDeleteContext, 1, -1);
    mocker(RsUbFreeSegCbList, 1, -1);
    mocker(RsUbFreeJettyCbList, 1, -1);
    mocker(RsUbFreeJfcCbList, 1, -1);
    mocker(RsUbFreeJfceCbList, 1, -1);
    mocker(RsUbFreeTokenIdCbList, 1, -1);
    RS_INIT_LIST_HEAD(&devCb->list);
    ret = RsUbCtxDeinit(devCb);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();
}

void TcRsUbInitSegCb()
{
    struct MemRegAttrT memAttr = {0};
    struct RsUbDevCb devCb = {0};
    struct RsSegCb segCb = {0};
    int ret = 0;

    mocker(RsUbGetTokenIdCb, 1, 0);
    mocker(RsUrmaRegisterSeg, 1, NULL);
    ret = RsUbInitSegCb(&memAttr, &devCb, &segCb);
    EXPECT_INT_NE(0, ret);
    mocker_clean();
}

void TcRsUbCtxLmemReg()
{
    struct MemRegAttrT memAttr = {0};
    struct MemRegInfoT memInfo = {0};
    struct RsUbDevCb devCb = {0};
    int ret = 0;

    memAttr.mem.size = 1;
    mocker(RsUbInitSegCb, 1, -1);
    ret = RsUbCtxLmemReg(&devCb, &memAttr, &memInfo);
    EXPECT_INT_NE(0, ret);
    mocker_clean();

    mocker(RsUrmaImportSeg, 1, NULL);
    ret = RsUbCtxRmemImport(&devCb, (struct MemImportAttrT*)&memAttr, (struct MemImportInfoT*)&memInfo);
    EXPECT_INT_NE(0, ret);
    mocker_clean();
}

void TcRsUbCtxJfcCreateFail()
{
    struct RsUbDevCb devCb = {0};
    struct CtxCqAttr attr = {0};
    struct CtxCqInfo info = {0};
    struct rs_cb rsCb = {0};
    int ret = 0;

    devCb.rscb = &rsCb;
    attr.ub.mode = JFC_MODE_STARS_POLL;
    mocker(RsUbCtxJfcCreateExt, 1, -1);
    ret = RsUbCtxJfcCreate(&devCb, &attr, &info);
    EXPECT_INT_NE(0, ret);
    mocker_clean();
}

void TcRsUbCtxInitJettyCb()
{
    struct RsCtxJettyCb* jettyCb = NULL;
    struct RsUbDevCb devCb = {0};
    struct CtxQpAttr attr = {0};
    int ret = 0;

    mocker(pthread_mutex_init, 1, -1);
    ret = RsUbCtxInitJettyCb(&devCb, &attr, &jettyCb);
    EXPECT_INT_NE(0, ret);
    mocker_clean();
}

static struct RsCtxJfcCb gSendJfcCb = {0};

int RsUbQueryJfcCbStub(
    struct RsUbDevCb* devCb, unsigned long long scqIndex, unsigned long long rcqIndex, struct RsCtxJfcCb** sendJfcCb,
    struct RsCtxJfcCb** recvJfcCb)
{
    gSendJfcCb.jfcType = JFC_MODE_NORMAL;
    *sendJfcCb = &gSendJfcCb;
    *recvJfcCb = &gSendJfcCb;
    return 0;
}

void TcRsUbCtxJettyCreateFail()
{
    struct RsUbDevCb devCb = {0};
    struct QpCreateInfo info = {0};
    struct CtxQpAttr attr = {0};
    int ret = 0;

    mocker(RsUbCtxInitJettyCb, 1, 0);
    mocker(RsUbQueryJfcCb, 1, -1);
    mocker(RsUbCtxFreeJettyCb, 1, -1);
    ret = RsUbCtxJettyCreate(&devCb, &attr, &info);
    EXPECT_INT_NE(0, ret);
    mocker_clean();

    mocker(RsUbCtxInitJettyCb, 1, 0);
    mocker_invoke_p5(
        "RsUbQueryJfcCb", "RsUbQueryJfcCbStub", (stub_fn_t)RsUbQueryJfcCb, (stub_fn_t)RsUbQueryJfcCbStub, 10);
    mocker(RsUbCtxDrvJettyCreate, 1, 0);
    mocker(RsUbFillJettyInfo, 1, -1);
    mocker(RsUbCtxDrvJettyDelete, 1, -1);
    mocker(RsUbCtxFreeJettyCb, 1, -1);
    mocker(RsUbJettyCbBuffAlloc, 10, 0);
    ret = RsUbCtxJettyCreate(&devCb, &attr, &info);
    EXPECT_INT_NE(0, ret);
    mocker_clean();
}

void TcRsUbCtxJettyImportFail()
{
    struct RsJettyImportAttr importAttr = {0};
    struct RsJettyImportInfo importInfo = {0};
    struct RsUbDevCb devCb = {0};
    int ret = 0;

    mocker(RsUbCtxInitRjettyCb, 1, 0);
    mocker(RsUbCtxDrvJettyImport, 1, -1);
    ret = RsUbCtxJettyImport(&devCb, &importAttr, &importInfo);
    EXPECT_INT_NE(0, ret);
    mocker_clean();
}

void TcRsUbCtxBatchSendWrFail()
{
    struct WrlistSendCompleteNum wrlistNum = {0};
    struct WrlistBaseInfo baseInfo = {0};
    struct BatchSendWrData wrData = {0};
    struct SendWrResp wrResp = {0};
    struct rs_cb rsCb = {0};
    int ret = 0;

    wrlistNum.sendNum = 1;
    mocker(RsUbGetDevCb, 1, -1);
    ret = RsUbCtxBatchSendWr(&rsCb, &baseInfo, &wrData, &wrResp, &wrlistNum);
    EXPECT_INT_NE(0, ret);
    mocker_clean();

    mocker(RsUbGetDevCb, 1, 0);
    mocker(RsUbGetJettyCb, 1, -1);
    ret = RsUbCtxBatchSendWr(&rsCb, &baseInfo, &wrData, &wrResp, &wrlistNum);
    EXPECT_INT_NE(0, ret);
    mocker_clean();
}

void TcRsUbCtxJettyDestroyBatch()
{
    struct JettyDestroyBatchInfo batchInfo = {0};
    struct RsUbDevCb devCb = {0};
    urma_device_t urmaDev = {0};
    unsigned int jettyIds[1] = {0};
    unsigned int num = 0;
    int ret = 0;
    devCb.urmaDev = &urmaDev;

    ret = RsUbCtxJettyDestroyBatch(&devCb, jettyIds, &num);
    EXPECT_INT_EQ(-EINVAL, ret);

    num = 1;
    mocker(RsUbCallocJettyBatchInfo, 1, -1);
    ret = RsUbCtxJettyDestroyBatch(&devCb, jettyIds, &num);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    num = 1;
    mocker(RsUbGetJettyDestroyBatchInfo, 1, -1);
    ret = RsUbCtxJettyDestroyBatch(&devCb, jettyIds, &num);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    num = 1;
    mocker(RsUbGetJettyDestroyBatchInfo, 1, 0);
    mocker(RsUbDestroyJettyCbBatch, 1, -1);
    ret = RsUbCtxJettyDestroyBatch(&devCb, jettyIds, &num);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    mocker(RsUbGetJettyCb, 1, -1);
    ret = RsUbGetJettyDestroyBatchInfo(&devCb, jettyIds, &batchInfo, &num);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    jettyCbStub.state = RS_JETTY_STATE_INIT;
    mocker_invoke(RsUbGetJettyCb, RsUbGetJettyCbStub, 1);
    mocker(RsUbGetJettyCb, 1, 0);
    ret = RsUbCtxJettyDestroyBatch(&devCb, jettyIds, &num);
    EXPECT_INT_EQ(-EINVAL, ret);
    mocker_clean();

    num = 1;
    mocker(RsUbGetJettyDestroyBatchInfo, 1, 0);
    mocker(RsUbDestroyJettyCbBatch, 1, 0);
    ret = RsUbCtxJettyDestroyBatch(&devCb, jettyIds, &num);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();

    num = 1;
    jettyCbStub.state = RS_JETTY_STATE_CREATED;
    pthread_mutex_init(&devCb.mutex, NULL);
    RS_INIT_LIST_HEAD(&devCb.jettyList);
    RsListAddTail(&jettyCbStub.list, &devCb.jettyList);
    devCb.jettyCnt++;
    mocker_invoke(RsUbGetJettyCb, RsUbGetJettyCbStub, 1);
    mocker(RsUbDestroyJettyCbBatch, 1, 0);
    mocker(RsUbGetJettyCb, 1, 0);
    ret = RsUbCtxJettyDestroyBatch(&devCb, jettyIds, &num);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();

    num = 1;
    jettyCbStub.state = RS_JETTY_STATE_CREATED;
    jettyCbStub.devCb = &devCb;
    RsListAddTail(&jettyCbStub.list, &devCb.jettyList);
    devCb.jettyCnt++;
    mocker_invoke(RsUbGetJettyCb, RsUbGetJettyCbStub, 1);
    mocker_invoke(RsUbFreeJettyCbBatch, RsUbFreeJettyCbBatchStub, 1);
    mocker(RsUrmaDeleteJettyBatch, 1, -1);
    mocker(RsUbGetJettyCb, 1, 0);
    ret = RsUbCtxJettyDestroyBatch(&devCb, jettyIds, &num);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    num = 1;
    jettyCbStub.state = RS_JETTY_STATE_CREATED;
    jettyCbStub.devCb = &devCb;
    RsListAddTail(&jettyCbStub.list, &devCb.jettyList);
    devCb.jettyCnt++;
    mocker_invoke(RsUbGetJettyCb, RsUbGetJettyCbStub, 1);
    mocker_invoke(RsUbFreeJettyCbBatch, RsUbFreeJettyCbBatchStub, 1);
    mocker(RsUrmaDeleteJettyBatch, 1, 0);
    mocker(RsUbGetJettyCb, 1, 0);
    ret = RsUbCtxJettyDestroyBatch(&devCb, jettyIds, &num);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();

    pthread_mutex_destroy(&devCb.mutex);
}

void TcRsUbCtxQueryJettyBatch()
{
    struct RsUbDevCb devCb = {0};
    unsigned int jettyIds[] = {1, 2, 3};
    struct JettyAttr attr[3] = {{0}};
    unsigned int num = 3;
    int ret = 0;

    mocker(RsUbGetJettyCb, 1, -1);
    ret = RsUbCtxQueryJettyBatch(&devCb, jettyIds, attr, &num);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    num = 3;
    mocker_invoke(RsUbGetJettyCb, RsUbGetJettyCbStub, 1);
    mocker(RsUrmaQueryJetty, 1, -1);
    ret = RsUbCtxQueryJettyBatch(&devCb, jettyIds, attr, &num);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    num = 3;
    mocker_invoke(RsUbGetJettyCb, RsUbGetJettyCbStub, 3);
    ret = RsUbCtxQueryJettyBatch(&devCb, jettyIds, attr, &num);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();
}

void TcRsGetEidByIp()
{
    struct RaRsDevInfo devInfo = {0};
    union HccpEid eid[32] = {0};
    struct IpInfo ip[32] = {0};
    unsigned int num = 32;
    int ret = 0;

    mocker(RsGetRsCb, 1, -1);
    ret = RsGetEidByIp(&devInfo, ip, eid, &num);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    mocker(RsGetRsCb, 1, 0);
    mocker(RsUbGetDevCb, 1, -1);
    ret = RsGetEidByIp(&devInfo, ip, eid, &num);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    mocker(RsGetRsCb, 1, 0);
    mocker(RsUbGetDevCb, 1, 0);
    mocker(RsUbGetEidByIp, 1, -1);
    ret = RsGetEidByIp(&devInfo, ip, eid, &num);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    mocker(RsGetRsCb, 1, 0);
    mocker(RsUbGetDevCb, 1, 0);
    mocker(RsUbGetEidByIp, 1, 0);
    ret = RsGetEidByIp(&devInfo, ip, eid, &num);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();
}

void TcRsUbGetEidByIp()
{
    struct RsUbDevCb devCb = {0};
    union HccpEid eid[32] = {0};
    struct IpInfo ip[32] = {0};
    unsigned int num = 32;
    int ret = 0;
    int i = 0;

    for (i = 0; i < 32; i++) {
        ip[i].family = AF_INET;
    }
    ret = RsUbGetEidByIp(&devCb, ip, eid, &num);
    EXPECT_INT_EQ(0, ret);

    errno = 0;
    mocker(RsUrmaGetEidByIp, 1, -URMA_FAIL);
    for (i = 0; i < 32; i++) {
        ip[i].family = AF_INET6;
    }
    ret = RsUbGetEidByIp(&devCb, ip, eid, &num);
    EXPECT_INT_EQ(-EOPENSRC, ret);
    mocker_clean();
}

void TcRsGetIpByEid()
{
    struct RaRsDevInfo devInfo = {0};
    union HccpEid eid[32] = {0};
    struct IpInfo ip[32] = {0};
    unsigned int num = 32;
    int ret = 0;

    mocker(RsGetRsCb, 1, -1);
    ret = RsGetIpByEid(&devInfo, eid, ip, &num);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    mocker(RsGetRsCb, 1, 0);
    mocker(RsUbGetDevCb, 1, -1);
    ret = RsGetIpByEid(&devInfo, eid, ip, &num);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    mocker(RsGetRsCb, 1, 0);
    mocker(RsUbGetDevCb, 1, 0);
    mocker(RsUbGetIpByEid, 1, -1);
    ret = RsGetIpByEid(&devInfo, eid, ip, &num);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    mocker(RsGetRsCb, 1, 0);
    mocker(RsUbGetDevCb, 1, 0);
    mocker(RsUbGetIpByEid, 1, 0);
    ret = RsGetIpByEid(&devInfo, eid, ip, &num);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();
}

void TcRsUbGetIpByEid()
{
    struct RsUbDevCb devCb = {0};
    union HccpEid eid[32] = {0};
    struct IpInfo ip[32] = {0};
    unsigned int num = 32;
    int ret = 0;

    ret = RsUbGetIpByEid(&devCb, eid, ip, &num);
    EXPECT_INT_EQ(0, ret);

    mocker(RsUrmaGetIpByEid, 1, -1);
    ret = RsUbGetIpByEid(&devCb, eid, ip, &num);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();
}

void TcRsUbCtxGetAuxInfo()
{
    struct HccpAuxInfoOut infoOut = {0};
    struct HccpAuxInfoIn infoIn = {0};
    struct RsUbDevCb devCb = {0};
    int ret = 0;

    mocker_clean();
    infoIn.type = AUX_INFO_IN_TYPE_CQE;
    mocker(RsUrmaUserCtl, 1, 0);
    ret = RsUbCtxGetAuxInfo(&devCb, &infoIn, &infoOut);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();

    int savedErrno = errno;
    errno = 0;
    mocker(RsUrmaUserCtl, 1, -URMA_FAIL);
    ret = RsUbCtxGetAuxInfo(&devCb, &infoIn, &infoOut);
    EXPECT_INT_EQ(-EOPENSRC, ret);
    mocker_clean();
    errno = savedErrno;

    infoIn.type = AUX_INFO_IN_TYPE_AE;
    mocker(RsUrmaUserCtl, 1, 0);
    ret = RsUbCtxGetAuxInfo(&devCb, &infoIn, &infoOut);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();

    mocker(RsUrmaUserCtl, 1, -1);
    ret = RsUbCtxGetAuxInfo(&devCb, &infoIn, &infoOut);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    infoIn.type = AUX_INFO_IN_TYPE_MAX;
    ret = RsUbCtxGetAuxInfo(&devCb, &infoIn, &infoOut);
    EXPECT_INT_EQ(-EINVAL, ret);
    mocker_clean();
}

void TcRsUbGetTpAttr()
{
    struct RsUbDevCb devCb = {0};
    unsigned int attrBitmap = 0b101010;
    uint64_t tpHandle = 12345;
    struct TpAttr attr = {0};
    int ret = 0;

    mocker(RsUrmaGetTpAttr, 1, -1);
    ret = RsUbGetTpAttr(&devCb, &attrBitmap, tpHandle, &attr);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    ret = RsUbGetTpAttr(&devCb, &attrBitmap, tpHandle, &attr);
    EXPECT_INT_EQ(0, ret);
}

void TcRsUbSetTpAttr()
{
    struct RsUbDevCb devCb = {0};
    unsigned int attrBitmap = 0b101010;
    uint64_t tpHandle = 12345;
    struct TpAttr attr = {0};
    int ret = 0;

    mocker(RsUrmaSetTpAttr, 1, -1);
    ret = RsUbSetTpAttr(&devCb, attrBitmap, tpHandle, &attr);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    ret = RsUbSetTpAttr(&devCb, attrBitmap, tpHandle, &attr);
    EXPECT_INT_EQ(0, ret);
}

void TcRsEpollEventJfcInHandle()
{
    struct RsCtxJfceCb jfceCb1 = {0};
    struct RsCtxJfceCb jfceCb2 = {0};
    struct RsUbDevCb devCb1 = {0};
    struct RsUbDevCb devCb2 = {0};
    struct rs_cb rsCb = {0};
    urma_jfce_t jfce1 = {0};
    urma_jfce_t jfce2 = {0};
    int ret = 0;

    RS_INIT_LIST_HEAD(&rsCb.udevList);
    ret = RsEpollEventJfcInHandle(&rsCb, -ENODEV);
    EXPECT_INT_EQ(-ENODEV, ret);

    RsListAddTail(&devCb1.list, &rsCb.udevList);
    RsListAddTail(&devCb2.list, &rsCb.udevList);
    RS_INIT_LIST_HEAD(&devCb1.jfceList);
    RS_INIT_LIST_HEAD(&devCb2.jfceList);
    RsListAddTail(&jfceCb1.list, &devCb2.jfceList);
    RsListAddTail(&jfceCb2.list, &devCb2.jfceList);

    jfce1.fd = 1;
    jfce2.fd = 2;
    jfceCb1.jfceAddr = (uint64_t)(uintptr_t)(&jfce1);
    jfceCb2.jfceAddr = (uint64_t)(uintptr_t)(&jfce2);
    mocker_clean();
    mocker(RsHandleEpollPollJfc, 1, 0);
    ret = RsEpollEventJfcInHandle(&rsCb, 2);
    EXPECT_INT_EQ(0, ret);
    mocker_clean();
}

void TcRsJfcCallbackProcess()
{
    struct RsCtxJettyCb jettyCb = {0};
    struct RsUbDevCb devCb = {0};
    struct rs_cb rsCb = {0};
    urma_jetty_t jetty = {0};
    urma_jfc_t jfc = {0};
    urma_cr_t cr = {0};

    devCb.rscb = &rsCb;
    jettyCb.devCb = &devCb;
    jettyCb.jetty = &jetty;

    cr.status = URMA_CR_RNR_RETRY_CNT_EXC_ERR;
    RsJfcCallbackProcess(&jettyCb, &cr, &jfc);
}

void TcRsUbCtxInitDeviceByEidFail()
{
    struct DevBaseAttr baseAttr = {0};
    struct CtxInitAttr attr = {0};
    struct rs_cb rsCb = {0};
    int devIndex = 0;
    int ret = 0;

    mocker_clean();
    errno = 0;
    mocker(RsUrmaGetDeviceByEid, 1, NULL);
    ret = RsUbCtxInit(&rsCb, &attr, &devIndex, &baseAttr);
    EXPECT_INT_EQ(-EOPENSRC, ret);
    mocker_clean();
}

void TcRsUbCreateCtxFail()
{
    struct DevBaseAttr baseAttr = {0};
    struct RsUbDevCb devCb = {0};
    struct CtxInitAttr attr = {0};
    struct rs_cb rscb = {0};
    int devIndex = 0;
    int ret = 0;

    mocker_clean();
    mocker(pthread_mutex_init, 1, 0);
    errno = 0;
    mocker(RsUrmaCreateContext, 1, NULL);
    mocker(pthread_mutex_destroy, 1, 0);
    ret = RsUbDevCbInit(&attr, &devCb, &rscb, &devIndex, &baseAttr);
    EXPECT_INT_EQ(-EOPENSRC, ret);
    mocker_clean();
}

void TcRsUbGetDevAttrQueryFail()
{
    struct RsUbDevCb devCb = {0};
    struct DevBaseAttr devAttr = {0};
    unsigned int devIndex = 0;
    int ret = 0;

    mocker_clean();
    errno = 0;
    mocker(RsUrmaQueryDevice, 1, -URMA_FAIL);
    ret = RsUbGetDevAttr(&devCb, &devAttr, &devIndex);
    EXPECT_INT_EQ(-EOPENSRC, ret);
    mocker_clean();
}

void TcRsUbGetIpByEidUrmaFail()
{
    struct RsUbDevCb devCb = {0};
    union HccpEid eid[32] = {0};
    struct IpInfo ip[32] = {0};
    unsigned int num = 32;
    int ret = 0;

    ret = RsUbGetIpByEid(&devCb, eid, ip, &num);
    EXPECT_INT_EQ(0, ret);

    errno = 0;
    mocker(RsUrmaGetIpByEid, 1, -URMA_FAIL);
    ret = RsUbGetIpByEid(&devCb, eid, ip, &num);
    EXPECT_INT_EQ(-EOPENSRC, ret);
    mocker_clean();
}

void TcRsUbCtxDeinitDeleteFail()
{
    struct RsUbDevCb* devCb = (struct RsUbDevCb*)calloc(1, sizeof(struct RsUbDevCb));
    struct rs_cb rsCb = {0};
    int ret = 0;

    devCb->rscb = &rsCb;
    mocker_clean();
    errno = 0;
    mocker(urma_delete_context, 1, -URMA_FAIL);
    mocker(RsUbFreeSegCbList, 1, 0);
    mocker(RsUbFreeJettyCbList, 1, 0);
    mocker(RsUbFreeJfcCbList, 1, 0);
    mocker(RsUbFreeJfceCbList, 1, 0);
    mocker(RsUbFreeTokenIdCbList, 1, 0);
    mocker(RsUbFreeAsyncEventCbList, 1, 0);
    mocker(RsUbDevShareJfrDeinit, 1, 0);
    mocker(pthread_mutex_destroy, 1, 0);
    mocker(RsSensorNodeUnregister, 1, 0);
    mocker(RsUbApiDeinit, 1, 0);
    RS_INIT_LIST_HEAD(&devCb->list);
    ret = RsUbCtxDeinit(devCb);
    EXPECT_INT_EQ(-EOPENSRC, ret);
    mocker_clean();
}

void TcRsUbSetTpAttrSmacFail()
{
    struct RsUbDevCb devCb = {0};
    unsigned int attrBitmap = TP_ATTR_SIP_MASK | TP_ATTR_SMAC_MASK;
    uint64_t tpHandle = 12345;
    struct TpAttr attr = {0};
    int ret = 0;

    mocker_clean();
    errno = 0;
    mocker(RsUrmaGetSmac, 1, -URMA_FAIL);
    ret = RsUbSetTpAttr(&devCb, attrBitmap, tpHandle, &attr);
    EXPECT_INT_EQ(-EOPENSRC, ret);
    mocker_clean();
}

void TcRsUbSetTpAttrDmacFail()
{
    struct RsUbDevCb devCb = {0};
    unsigned int attrBitmap = TP_ATTR_DIP_MASK | TP_ATTR_DMAC_MASK;
    uint64_t tpHandle = 12345;
    struct TpAttr attr = {0};
    int ret = 0;

    mocker_clean();
    errno = 0;
    mocker(RsUrmaGetDmac, 1, -URMA_FAIL);
    ret = RsUbSetTpAttr(&devCb, attrBitmap, tpHandle, &attr);
    EXPECT_INT_EQ(-EOPENSRC, ret);
}

void TcRsUbDevShareJfrInitFail()
{
    struct RsUbDevCb devCb = {0};
    urma_context_t urmaCtx = {0};
    int ret = 0;

    devCb.urmaCtx = &urmaCtx;
    mocker(RsUrmaCreateJfr, 1, NULL);
    ret = RsUbDevShareJfrInit(&devCb);
    EXPECT_INT_NE(0, ret);
    mocker_clean();
}

int TcRsNetGetCqeBaseAddrNonZero(unsigned int dieId, unsigned long long* cqeBaseAddr)
{
    *cqeBaseAddr = 0x1000;
    return 0;
}

void TcRsUbDevShareJfrInitCreateJfcFail()
{
    struct RsUbDevCb devCb = {0};
    urma_context_t urmaCtx = {0};
    int ret = 0;

    devCb.urmaCtx = &urmaCtx;
    mocker(RsUrmaCreateJfc, 1, NULL);
    ret = RsUbDevShareJfrInit(&devCb);
    EXPECT_INT_NE(0, ret);
    mocker_clean();
}

void TcRsUbCtxChanDestroyJfceFail()
{
    struct RsUbDevCb devCb = {0};
    urma_jfce_t jfce = {0};
    unsigned long long addr = 1;
    int ret = 0;

    devCb.index = 0;
    devCb.rscb = NULL;
    RS_INIT_LIST_HEAD(&devCb.jfceList);
    devCb.jfceCnt = 1;
    jfce.fd = 0;
    gJfceCb.devCb = &devCb;
    gJfceCb.jfceAddr = (uint64_t)(uintptr_t)&jfce;
    gJfceCb.dataPlaneFlag.bs.pollCqCstm = 1;
    pthread_mutex_init(&devCb.mutex, NULL);
    mocker_invoke(RsUbGetJfceCb, RsUbGetJfceCbDynStub, 1);
    mocker(RsUrmaDeleteJfce, 1, -URMA_FAIL);
    ret = RsUbCtxChanDestroy(&devCb, addr);
    EXPECT_INT_NE(0, ret);
    mocker_clean();
    pthread_mutex_destroy(&devCb.mutex);
}

void TcRsUbFreeJfceCbDeleteJfceFail()
{
    struct RsUbDevCb devCb = {0};
    struct RsCtxJfceCb* jfceCb = NULL;
    urma_jfce_t jfce = {0};
    int ret = 0;

    jfceCb = (struct RsCtxJfceCb*)calloc(1, sizeof(struct RsCtxJfceCb));
    if (jfceCb == NULL) {
        return;
    }
    RS_INIT_LIST_HEAD(&devCb.jfceList);
    RsListAddTail(&jfceCb->list, &devCb.jfceList);
    devCb.jfceCnt = 1;
    jfceCb->devCb = &devCb;
    jfceCb->jfceAddr = (uint64_t)(uintptr_t)&jfce;
    jfceCb->dataPlaneFlag.bs.pollCqCstm = 1;
    mocker(RsUrmaDeleteJfce, 1, -URMA_FAIL);
    ret = RsUbFreeJfceCb(&devCb, jfceCb);
    EXPECT_INT_NE(0, ret);
    mocker_clean();
}

void TcRsUbFreeRemJettyCbFail()
{
    struct RsUbDevCb devCb = {0};
    struct RsCtxRemJettyCb rjettyCb = {0};
    urma_target_jetty_t tjetty = {0};
    int ret = 0;

    tjetty.id.id = 1;
    rjettyCb.tjetty = &tjetty;
    rjettyCb.devCb = &devCb;
    RS_INIT_LIST_HEAD(&devCb.rjettyList);
    RsListAddTail(&rjettyCb.list, &devCb.rjettyList);
    devCb.rjettyCnt = 1;

    mocker(RsUrmaUnimportJetty, 1, -URMA_FAIL);
    errno = 0;
    ret = RsUbFreeRemJettyCb(&devCb, &rjettyCb);
    EXPECT_INT_EQ(-EOPENSRC, ret);
    mocker_clean();

    RsListAddTail(&rjettyCb.list, &devCb.rjettyList);
    devCb.rjettyCnt = 1;
    errno = EINVAL;
    mocker(RsUrmaUnimportJetty, 1, -URMA_FAIL);
    ret = RsUbFreeRemJettyCb(&devCb, &rjettyCb);
    EXPECT_INT_EQ(-EINVAL, ret);
    mocker_clean();
}

void TcRsUbFreeSegCbFail()
{
    struct RsUbDevCb devCb = {0};
    struct RsSegCb segCb = {0};
    int ret = 0;

    RS_INIT_LIST_HEAD(&devCb.lsegList);
    RsListAddTail(&segCb.list, &devCb.lsegList);
    devCb.lsegCnt = 1;

    int savedErrno = errno;
    errno = 0;
    mocker(RsUrmaUnregisterSeg, 1, -URMA_FAIL);
    ret = RsUbFreeSegCb(&devCb, &segCb);
    EXPECT_INT_EQ(-EOPENSRC, ret);
    mocker_clean();
    errno = savedErrno;
}

void TcRsUbGetTpInfoListFail()
{
    struct RsUbDevCb devCb = {0};
    struct GetTpCfg cfg = {0};
    struct HccpTpInfo infoList[1] = {0};
    unsigned int num = 1;
    int ret = 0;

    int savedErrno = errno;
    errno = 0;
    mocker(RsUrmaGetTpList, 1, -URMA_FAIL);
    ret = RsUbGetTpInfoList(&devCb, &cfg, infoList, &num);
    EXPECT_INT_EQ(-EOPENSRC, ret);
    mocker_clean();
    errno = savedErrno;
}

void TcRsHandleEpollPollJfcFail()
{
    struct RsUbDevCb devCb = {0};
    urma_jfce_t jfce = {0};
    urma_jfc_t jfc = {0};
    int ret = 0;

    jfce.fd = 1;
    devCb.urmaCtx = (urma_context_t*)0x1;

    mocker(RsUrmaWaitJfc, 1, -1);
    ret = RsHandleEpollPollJfc(&devCb, &jfce);
    EXPECT_INT_EQ(-EOPENSRC, ret);
    mocker_clean();

    mocker(RsUrmaWaitJfc, 1, 0);
    ret = RsHandleEpollPollJfc(&devCb, &jfce);
    EXPECT_INT_EQ(-EAGAIN, ret);
    mocker_clean();

    mocker(RsUrmaWaitJfc, 1, 1);
    mocker(RsUrmaPollJfc, 1, -1);
    mocker(RsUrmaRearmJfc, 1, -1);
    ret = RsHandleEpollPollJfc(&devCb, &jfce);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    mocker(RsUrmaWaitJfc, 1, 1);
    mocker(RsUrmaPollJfc, 1, -1);
    mocker(RsUrmaRearmJfc, 1, 0);
    ret = RsHandleEpollPollJfc(&devCb, &jfce);
    EXPECT_INT_EQ(-EOPENSRC, ret);
    mocker_clean();
}

void TcRsUbGetUeInfoFail()
{
    urma_context_t urmaCtx = {0};
    struct DevBaseAttr devAttr = {0};
    int ret = 0;

    mocker(RsUrmaUserCtl, 1, -1);
    ret = RsUbGetUeInfo(&urmaCtx, &devAttr);
    EXPECT_INT_EQ(-1, ret);
    mocker_clean();

    int savedErrno = errno;
    errno = EINVAL;
    mocker(RsUrmaUserCtl, 1, -URMA_FAIL);
    ret = RsUbGetUeInfo(&urmaCtx, &devAttr);
    EXPECT_INT_EQ(-EINVAL, ret);
    mocker_clean();
    errno = savedErrno;
}

void TcRsUbCtxDrvJettyImportNormalFail()
{
    struct RsCtxRemJettyCb rjettyCb = {0};
    struct RsUbDevCb devCb = {0};
    urma_context_t urmaCtx = {0};
    int ret = 0;

    rjettyCb.mode = JETTY_IMPORT_MODE_NORMAL;
    devCb.urmaCtx = &urmaCtx;
    rjettyCb.devCb = &devCb;

    mocker(RsUrmaImportJetty, 1, NULL);
    errno = 0;
    ret = RsUbCtxDrvJettyImport(&rjettyCb);
    EXPECT_INT_EQ(-EOPENSRC, ret);
    mocker_clean();
}

void TcRsUbAsyncEventGetJettyContextFail()
{
    struct RsUbDevCb devCb = {0};
    struct rs_cb rscb = {0};
    struct RsCtxAsyncEventCb asyncEventCb = {0};
    urma_async_event_t event = {0};
    urma_jetty_t jetty = {0};
    int ret = 0;

    devCb.rscb = &rscb;
    rscb.hccpMode = NETWORK_OFFLINE;
    event.element.jetty = &jetty;
    mocker(RsUrmaGetJettyOpt, 1, -URMA_FAIL);
    ret = RsUbAsyncEventGetJettyContext(&event, &devCb, &asyncEventCb);
    EXPECT_INT_NE(0, ret);
    EXPECT_INT_NE(-URMA_FAIL, ret);
    mocker_clean();
}

void TcRsUbGetJettyContextFail()
{
    struct RsUbDevCb devCb = {0};
    uint8_t context[JETTY_CONTEXT_LEN] = {0};
    unsigned int len = 0;
    int ret = 0;

    mocker_invoke(RsUbGetJettyCb, RsUbGetJettyCbStub, 1);
    mocker(RsUrmaGetJettyOpt, 1, -URMA_FAIL);
    errno = 0;
    ret = RsUbGetJettyContext(&devCb, 1, context, &len);
    EXPECT_INT_EQ(-EOPENSRC, ret);
    mocker_clean();
}

void TcRsUrmaErrnoMacro()
{
    int ret;

    errno = 0;
    ret = -URMA_FAIL;
    RS_URMA_RET_TO_ERRNO(ret);
    EXPECT_INT_EQ(-EOPENSRC, ret);

    errno = EINVAL;
    ret = -URMA_FAIL;
    RS_URMA_RET_TO_ERRNO(ret);
    EXPECT_INT_EQ(-EINVAL, ret);

    ret = 0;
    RS_URMA_RET_TO_ERRNO(ret);
    EXPECT_INT_EQ(0, ret);

    ret = -EINVAL;
    RS_URMA_RET_TO_ERRNO(ret);
    EXPECT_INT_EQ(-EINVAL, ret);

    errno = 0;
    ret = 0;
    RS_URMA_PTR_TO_ERRNO(NULL, ret);
    EXPECT_INT_EQ(-EOPENSRC, ret);

    errno = EINVAL;
    ret = 0;
    RS_URMA_PTR_TO_ERRNO(NULL, ret);
    EXPECT_INT_EQ(-EINVAL, ret);

    ret = 0;
    RS_URMA_PTR_TO_ERRNO((void*)0x1, ret);
    EXPECT_INT_EQ(0, ret);
}

void TcRsUbCtxJfcCreateExtFail()
{
    struct RsCtxJfcCb jfcCb = {0};
    struct RsUbDevCb devCb = {0};
    urma_jfc_cfg_t jfcCfg = {0};
    urma_jfc_t* jfc = NULL;
    int ret = 0;

    jfcCb.jfcType = JFC_MODE_USER_CTL_NORMAL;
    jfcCb.devCb = &devCb;

    mocker(RsUrmaAllocJfc, 1, -URMA_FAIL);
    ret = RsUbCtxJfcCreateExt(&jfcCb, &jfcCfg, &jfc);
    EXPECT_INT_NE(0, ret);
    mocker_clean();

    mocker(RsUrmaActiveJfc, 1, -URMA_FAIL);
    ret = RsUbCtxJfcCreateExt(&jfcCb, &jfcCfg, &jfc);
    EXPECT_INT_NE(0, ret);
    mocker_clean();
}

void TcRsUbDeleteJfcExtFail()
{
    struct RsUbDevCb devCb = {0};
    struct RsCtxJfcCb jfcCb = {0};
    urma_jfc_t jfc = {0};
    urma_device_t urmaDev = {0};
    int ret = 0;

    jfcCb.jfcType = JFC_MODE_NORMAL;
    jfcCb.jfcAddr = (uint64_t)(uintptr_t)&jfc;
    jfcCb.devCb = &devCb;
    devCb.urmaDev = &urmaDev;
    mocker(RsUrmaDeactiveJfc, 1, -URMA_FAIL);
    mocker(RsUrmaFreeJfc, 1, -URMA_FAIL);
    mocker(RsNetFreeJfcId, 1, -EINVAL);
    ret = RsUbDeleteJfcExt(&devCb, &jfcCb);
    EXPECT_INT_NE(0, ret);
    mocker_clean();
}

void TcRsJfcOptFail()
{
    struct RsCtxJfcCb jfcCb = {0};
    struct RsUbDevCb devCb = {0};
    urma_jfc_cfg_t jfcCfg = {0};
    urma_jfc_t* jfc = NULL;
    urma_device_t urmaDev = {0};
    int ret = 0;

    jfcCb.devCb = &devCb;
    devCb.urmaDev = &urmaDev;

    // #7: SetJfcOpt URMA_JFC_ID fail (STARS_POLL path)
    jfcCb.jfcType = JFC_MODE_STARS_POLL;
    mocker_invoke(RsNetGetCqeBaseAddr, TcRsNetGetCqeBaseAddrNonZero, 10);
    mocker(RsNetAllocJfcId, 10, 0);
    mocker(RsNetFreeJfcId, 10, 0);
    mocker(RsUrmaSetJfcOpt, 1, -URMA_FAIL);
    ret = RsUbCtxJfcCreateExt(&jfcCb, &jfcCfg, &jfc);
    EXPECT_INT_NE(0, ret);
    mocker_clean();

    // #11+#8: ActiveJfc fail after SetJfcOpt succeeds (STARS_POLL path)
    jfcCb.jfcType = JFC_MODE_STARS_POLL;
    mocker_invoke(RsNetGetCqeBaseAddr, TcRsNetGetCqeBaseAddrNonZero, 10);
    mocker(RsNetAllocJfcId, 10, 0);
    mocker(RsNetFreeJfcId, 10, 0);
    mocker(RsUrmaActiveJfc, 1, -URMA_FAIL);
    ret = RsUbCtxJfcCreateExt(&jfcCb, &jfcCfg, &jfc);
    EXPECT_INT_NE(0, ret);
    mocker_clean();

    // #9: GetJfcOpt fail (USER_CTL_NORMAL path, skips SetJfcOpt)
    jfcCb.jfcType = JFC_MODE_USER_CTL_NORMAL;
    mocker(RsUrmaActiveJfc, 1, 0);
    mocker(RsUrmaGetJfcOpt, 1, -URMA_FAIL);
    ret = RsUbCtxJfcCreateExt(&jfcCb, &jfcCfg, &jfc);
    EXPECT_INT_NE(0, ret);
    mocker_clean();
}

void TcRsUbAsyncEventGetJfcContextFail()
{
    struct RsUbDevCb devCb = {0};
    struct rs_cb rscb = {0};
    struct RsCtxAsyncEventCb asyncEventCb = {0};
    urma_async_event_t event = {0};
    urma_jfc_t jfc = {0};
    int ret = 0;

    devCb.rscb = &rscb;
    event.element.jfc = &jfc;
    mocker(RsUrmaGetJfcOpt, 1, -URMA_FAIL);
    ret = RsUbAsyncEventGetJfcContext(&event, &devCb, &asyncEventCb);
    EXPECT_INT_NE(0, ret);
    mocker_clean();
}

void TcRsUbGetSaveAsyncEventFail()
{
    struct RsUbDevCb devCb = {0};
    urma_context_t urmaCtx = {0};
    int ret = 0;

    devCb.urmaCtx = &urmaCtx;
    mocker(RsUrmaGetAsyncEvent, 1, -URMA_FAIL);
    ret = RsUbGetSaveAsyncEvent(&devCb);
    EXPECT_INT_NE(0, ret);
    mocker_clean();
}
