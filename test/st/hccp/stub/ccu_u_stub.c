/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * ST simulation of the CCU user-mode driver library (libccu-user-drv.so).
 *
 * Production: the rdma_service side (hccp_service.bin) dlsyms this library
 * (dl_ccu_function.c) and forwards the CCU channel commands to the real driver
 * and CCU hardware. The simulation has no CCU hardware, so this stub keeps a
 * per-(resource class, die) shadow copy of the last SET payload and serves every
 * GET from that shadow. That makes the set->get readback assertions of the
 * migrated HLT cases (e.g. hlt_ccu_set_msid) observe consistent state, which the
 * previous empty shell (return 0, fill nothing) could never satisfy.
 *
 * Lifetime: the device process is forked per test case (rtOpenNetService), so
 * the shadow state never leaks across test cases, and ccu_init() wipes it on
 * CCU_INIT for good measure. Multi-die is modelled by keying on udie_idx.
 */

#include <string.h>
#include <pthread.h>
#include "ccu_u_comm.h"

struct ccu_mem_rsp {
    unsigned int die_id;
    unsigned int num;
    struct ccu_mem_info list[64U];
};

/*
 * Shadow resource classes. A SET opcode and its GET counterpart share one
 * class so a set->get round trip reads back what was written; opcodes without a
 * counterpart keep their own class. Raw classes store the payload verbatim, so
 * byte-wise comparisons (test_set_and_get_cmp in the original suite) succeed
 * regardless of which union member the case used.
 */
enum ccu_stub_res {
    CCU_RES_BASEINFO = 0,  /* SET_MSID_TOKEN / GET_MSID / GET_BASIC_INFO */
    CCU_RES_CHN_JETTY_MAP, /* SET_CHN_JETTY_MAP / GET_CHN_JETTY_MAP */
    CCU_RES_MISSION_TIMEOUT,
    CCU_RES_LOOP_TIMEOUT,
    CCU_RES_TIF_SPLIT_SIZE,
    CCU_RES_HIGH_PERF_XN,
    CCU_RES_XN_TOTAL_CNT,
    CCU_RES_TRANM_NTF_HVAL,
    CCU_RES_TASKKILL,
    CCU_RES_VERSION,
    CCU_RES_DIE_WORKING,
    CCU_RES_INSTRUCTION,
    CCU_RES_GSA,
    CCU_RES_XN,
    CCU_RES_CKE,
    CCU_RES_PFE,
    CCU_RES_CHANNEL,
    CCU_RES_JETTY_CTX,
    CCU_RES_MISSION_CTX,
    CCU_RES_LOOP_CTX,
    CCU_RES_MEMORY_SLICE,
    CCU_RES_LOOP_CKE_CTX,
    CCU_RES_MISSION_SQE,
    CCU_RES_CQE_BLOCK0,
    CCU_RES_CQE_BLOCK1,
    CCU_RES_CQE_BLOCK2,
    CCU_RES_WQEBB,
    CCU_RES_MS_BLOCK0,
    CCU_RES_MS_BLOCK1,
    CCU_RES_MS_BLOCK2,
    CCU_RES_MS_BLOCK3,
    CCU_RES_MAX,
};

#define CCU_STUB_MAX_DIE 4

struct ccu_stub_shadow {
    unsigned int valid;
    struct ccu_data data;
};

static struct ccu_stub_shadow g_ccuShadow[CCU_RES_MAX][CCU_STUB_MAX_DIE];
static pthread_mutex_t g_ccuShadowLock = PTHREAD_MUTEX_INITIALIZER;

/* Plausible chip/driver-derived values for the fields the caller cannot set.
 * resourceAddr points at per-die .bss of this stub so any same-process reader
 * finds readable memory; migrated cases only print it. */
static char g_ccuStubResource[CCU_STUB_MAX_DIE][4096];
static char g_ccuStubCqeBase[CCU_STUB_MAX_DIE][4096];

static int CcuStubClassOf(unsigned int op)
{
    switch (op) {
        case CCU_U_OP_SET_MSID_TOKEN:
        case CCU_U_OP_GET_MSID:
        case CCU_U_OP_GET_BASIC_INFO:
            return CCU_RES_BASEINFO;
        case CCU_U_OP_SET_CHN_JETTY_MAP:
        case CCU_U_OP_GET_CHN_JETTY_MAP:
            return CCU_RES_CHN_JETTY_MAP;
        case CCU_U_OP_GET_MISSION_TIMEOUT_DURATION:
        case CCU_U_OP_SET_MISSION_TIMEOUT_DURATION:
            return CCU_RES_MISSION_TIMEOUT;
        case CCU_U_OP_GET_LOOP_TIMEOUT_DURATION:
        case CCU_U_OP_SET_LOOP_TIMEOUT_DURATION:
            return CCU_RES_LOOP_TIMEOUT;
        case CCU_U_OP_SET_TIF_SPLIT_SIZE:
            return CCU_RES_TIF_SPLIT_SIZE;
        case CCU_U_OP_GET_HIGH_PERF_XN:
            return CCU_RES_HIGH_PERF_XN;
        case CCU_U_OP_SET_XN_TOTAL_CNT:
            return CCU_RES_XN_TOTAL_CNT;
        case CCU_U_OP_SET_TRANM_NTF_HVAL:
            return CCU_RES_TRANM_NTF_HVAL;
        case CCU_U_OP_SET_TASKKILL:
        case CCU_U_OP_CLEAN_TASKKILL_STATE:
            return CCU_RES_TASKKILL;
        case CCU_U_OP_GET_VERSION:
            return CCU_RES_VERSION;
        case CCU_U_OP_GET_DIE_WORKING:
            return CCU_RES_DIE_WORKING;
        case CCU_U_OP_GET_INSTRUCTION:
        case CCU_U_OP_SET_INSTRUCTION:
        case CCU_U_OP_CLEAN_TIF_TABLE: /* the TIF table is the instruction table */
            return CCU_RES_INSTRUCTION;
        case CCU_U_OP_GET_GSA:
        case CCU_U_OP_SET_GSA:
            return CCU_RES_GSA;
        case CCU_U_OP_GET_XN:
        case CCU_U_OP_SET_XN:
            return CCU_RES_XN;
        case CCU_U_OP_GET_CKE:
        case CCU_U_OP_SET_CKE:
            return CCU_RES_CKE;
        case CCU_U_OP_GET_PFE:
        case CCU_U_OP_SET_PFE:
            return CCU_RES_PFE;
        case CCU_U_OP_GET_CHANNEL:
        case CCU_U_OP_SET_CHANNEL:
            return CCU_RES_CHANNEL;
        case CCU_U_OP_GET_JETTY_CTX:
        case CCU_U_OP_SET_JETTY_CTX:
            return CCU_RES_JETTY_CTX;
        case CCU_U_OP_GET_MISSION_CTX:
            return CCU_RES_MISSION_CTX;
        case CCU_U_OP_GET_LOOP_CTX:
            return CCU_RES_LOOP_CTX;
        case CCU_U_OP_GET_MEMORY_SLICE:
            return CCU_RES_MEMORY_SLICE;
        case CCU_U_OP_GET_LOOP_CKE_CTX:
        case CCU_U_OP_SET_LOOP_CKE_CTX:
            return CCU_RES_LOOP_CKE_CTX;
        case CCU_U_OP_GET_MISSION_SQE:
            return CCU_RES_MISSION_SQE;
        case CCU_U_OP_GET_CQE_BLOCK0:
            return CCU_RES_CQE_BLOCK0;
        case CCU_U_OP_GET_CQE_BLOCK1:
            return CCU_RES_CQE_BLOCK1;
        case CCU_U_OP_GET_CQE_BLOCK2:
            return CCU_RES_CQE_BLOCK2;
        case CCU_U_OP_GET_WQEBB:
            return CCU_RES_WQEBB;
        case CCU_U_OP_GET_MS_BLOCK0:
            return CCU_RES_MS_BLOCK0;
        case CCU_U_OP_GET_MS_BLOCK1:
            return CCU_RES_MS_BLOCK1;
        case CCU_U_OP_GET_MS_BLOCK2:
            return CCU_RES_MS_BLOCK2;
        case CCU_U_OP_GET_MS_BLOCK3:
            return CCU_RES_MS_BLOCK3;
        default:
            return -1;
    }
}

/* opcodes that store their payload into the shadow */
static int CcuStubIsSetOp(unsigned int op)
{
    switch (op) {
        case CCU_U_OP_SET_MSID_TOKEN:
        case CCU_U_OP_SET_CHN_JETTY_MAP:
        case CCU_U_OP_SET_MISSION_TIMEOUT_DURATION:
        case CCU_U_OP_SET_LOOP_TIMEOUT_DURATION:
        case CCU_U_OP_SET_TIF_SPLIT_SIZE:
        case CCU_U_OP_SET_XN_TOTAL_CNT:
        case CCU_U_OP_SET_TRANM_NTF_HVAL:
        case CCU_U_OP_SET_TASKKILL:
        case CCU_U_OP_SET_INSTRUCTION:
        case CCU_U_OP_SET_GSA:
        case CCU_U_OP_SET_XN:
        case CCU_U_OP_SET_CKE:
        case CCU_U_OP_SET_PFE:
        case CCU_U_OP_SET_CHANNEL:
        case CCU_U_OP_SET_JETTY_CTX:
        case CCU_U_OP_SET_LOOP_CKE_CTX:
            return 1;
        default:
            return 0;
    }
}

/* Stamp the driver/chip-derived baseinfo fields the caller cannot set, so
 * GET_BASIC_INFO output looks like real hardware. */
static void CcuStubStampBaseinfo(struct ccu_data* data)
{
    unsigned int n = data->data_array_size;
    if (n > CCU_DATA_TYPE_UNION_ARRAY_SIZE) {
        n = CCU_DATA_TYPE_UNION_ARRAY_SIZE;
    }
    for (unsigned int i = 0; i < n; i++) {
        struct ccu_baseinfo* info = &data->data_array[i].baseinfo;
        info->token_valid = 1U;
        if (info->missionKey == 0U) {
            info->missionKey = 0x5A5A0000U + data->udie_idx * 0x11U + i;
        }
        if (info->resourceAddr == NULL) {
            info->resourceAddr = (data->udie_idx < CCU_STUB_MAX_DIE) ? g_ccuStubResource[data->udie_idx] : NULL;
        }
        if (info->caps.lqc_ccu_cap0 == 0U) {
            info->caps.lqc_ccu_cap0 = 0xC0C0U + i;
            info->caps.lqc_ccu_cap1 = 0xC1C1U + i;
            info->caps.lqc_ccu_cap2 = 0xC2C2U + i;
            info->caps.lqc_ccu_cap3 = 0xC3C3U + i;
            info->caps.lqc_ccu_cap4 = 0xC4C4U + i;
        }
    }
}

/* GET with no prior SET still returns sane content for these classes */
static void CcuStubSynthesize(unsigned int cls, unsigned int die, struct ccu_data* data)
{
    unsigned int n = data->data_array_size;
    if (n > CCU_DATA_TYPE_UNION_ARRAY_SIZE) {
        n = CCU_DATA_TYPE_UNION_ARRAY_SIZE;
    }
    for (unsigned int i = 0; i < n; i++) {
        switch (cls) {
            case CCU_RES_BASEINFO: {
                struct ccu_baseinfo* info = &data->data_array[i].baseinfo;
                info->token_valid = 1U;
                info->missionKey = 0x5A5A0000U + die * 0x11U + i;
                info->resourceAddr = (die < CCU_STUB_MAX_DIE) ? g_ccuStubResource[die] : NULL;
                info->caps.lqc_ccu_cap0 = 0xC0C0U + i;
                info->caps.lqc_ccu_cap1 = 0xC1C1U + i;
                info->caps.lqc_ccu_cap2 = 0xC2C2U + i;
                info->caps.lqc_ccu_cap3 = 0xC3C3U + i;
                info->caps.lqc_ccu_cap4 = 0xC4C4U + i;
                break;
            }
            case CCU_RES_VERSION:
                data->data_array[i].version = CCU_V2;
                break;
            case CCU_RES_DIE_WORKING:
                data->data_array[i].dieinfo.enable_flag = 1U;
                break;
            default:
                break; /* raw classes stay zero until something SETs them */
        }
    }
}

int ccu_init(void)
{
    /* CCU_INIT resets the device view: wipe the shadow so every case starts clean */
    pthread_mutex_lock(&g_ccuShadowLock);
    (void)memset(g_ccuShadow, 0, sizeof(g_ccuShadow));
    pthread_mutex_unlock(&g_ccuShadowLock);
    return 0;
}

int ccu_uninit(void) { return 0; }

unsigned long long ccu_get_cqe_base_addr(unsigned int die_id)
{
    /* must be non-zero: the CCU_POLL jfc path (rs_ub_jfc.c RsCcuGetCqeBaseAddr)
     * rejects a zero base address; a readable per-die .bss page stands in for
     * the mapped CQE block */
    return (unsigned long long)(uintptr_t)g_ccuStubCqeBase[die_id % CCU_STUB_MAX_DIE];
}

int ccu_custom_channel(const struct channel_info_in* in, struct channel_info_out* out)
{
    if (in == NULL || out == NULL) {
        return -1;
    }
    (void)memset(out, 0, sizeof(*out));

    int cls = CcuStubClassOf((unsigned int)in->op);
    if (cls < 0) {
        /* unknown opcode: fail loudly so migration gaps surface immediately */
        return -1;
    }
    const struct ccu_data* din = &in->data.data_info;
    unsigned int die = din->udie_idx;
    if (die >= CCU_STUB_MAX_DIE || din->data_array_size > CCU_DATA_TYPE_UNION_ARRAY_SIZE) {
        /* the channel call itself succeeded, the operation was rejected by the
         * driver: report through op_ret, mirroring the driver contract */
        out->op_ret = -1;
        return 0;
    }

    /* clear-type operations: wipe the shadow of their class (the TIF table is
     * the instruction table, so CLEAN_TIF_TABLE clears the INSTRUCTION class) */
    if (in->op == CCU_U_OP_CLEAN_TASKKILL_STATE || in->op == CCU_U_OP_CLEAN_TIF_TABLE) {
        pthread_mutex_lock(&g_ccuShadowLock);
        g_ccuShadow[cls][die].valid = 0U;
        (void)memset(&g_ccuShadow[cls][die].data, 0, sizeof(struct ccu_data));
        pthread_mutex_unlock(&g_ccuShadowLock);
        out->data.data_info = *din;
        out->offset_next = in->offset_start + din->data_array_size;
        out->op_ret = 0;
        return 0;
    }

    if (CcuStubIsSetOp((unsigned int)in->op) != 0) {
        pthread_mutex_lock(&g_ccuShadowLock);
        g_ccuShadow[cls][die].valid = 1U;
        g_ccuShadow[cls][die].data = *din;
        if (cls == CCU_RES_BASEINFO) {
            CcuStubStampBaseinfo(&g_ccuShadow[cls][die].data);
        }
        pthread_mutex_unlock(&g_ccuShadowLock);
        /* echo the request payload in the response, as the driver does */
        out->data.data_info = *din;
        if (cls == CCU_RES_BASEINFO) {
            CcuStubStampBaseinfo(&out->data.data_info);
        }
        out->offset_next = in->offset_start + din->data_array_size;
        out->op_ret = 0;
        return 0;
    }

    /* GET: serve from the shadow (zero payload when nothing was SET yet) */
    struct ccu_data payload;
    unsigned int wasValid;
    pthread_mutex_lock(&g_ccuShadowLock);
    payload = g_ccuShadow[cls][die].data;
    wasValid = g_ccuShadow[cls][die].valid;
    pthread_mutex_unlock(&g_ccuShadowLock);
    if (wasValid == 0U) {
        (void)memset(&payload, 0, sizeof(payload));
    }
    /* keep the request header (die, sizes) so the caller sees what it asked for */
    payload.udie_idx = die;
    payload.data_array_size = din->data_array_size;
    payload.data_array_len = din->data_array_len;
    if (wasValid == 0U) {
        CcuStubSynthesize((unsigned int)cls, die, &payload);
    }
    out->data.data_info = payload;
    out->offset_next = in->offset_start + din->data_array_size;
    out->op_ret = 0;
    return 0;
}

int ccu_get_mem_info(unsigned int die_id, unsigned long long mem_type_bitmap, struct ccu_mem_rsp* rsp)
{
    if (rsp == NULL) {
        return -1;
    }
    (void)memset(rsp, 0, sizeof(*rsp));
    rsp->die_id = die_id;
    unsigned int n = 0;
    for (unsigned int bit = 0; bit < 64U && n < 64U; bit++) {
        if ((mem_type_bitmap & (1ULL << bit)) == 0ULL) {
            continue;
        }
        /* plausible per-type mapped region inside a devmem-like window */
        rsp->list[n].mem_va
            = 0x500000100000ULL + (unsigned long long)die_id * 0x100000ULL + (unsigned long long)bit * 0x1000ULL;
        rsp->list[n].mem_size = 0x1000U;
        n++;
    }
    rsp->num = n;
    return 0;
}

int get_ccu_u_info(unsigned int die_id, struct ccu_u_info* info)
{
    if (info == NULL || die_id >= CCU_STUB_MAX_DIE) {
        return -1;
    }
    (void)memset(info, 0, sizeof(*info));
    pthread_mutex_lock(&g_ccuShadowLock);
    struct ccu_data base = g_ccuShadow[CCU_RES_BASEINFO][die_id].data;
    pthread_mutex_unlock(&g_ccuShadowLock);
    info->uent_num = 16U;
    info->ccu_flag = 1U;
    info->eid = 0x100U + die_id;
    info->ms_id = base.data_array[0].baseinfo.ms_id;
    info->missionKey = base.data_array[0].baseinfo.missionKey;
    info->resourceAddr = g_ccuStubResource[die_id];
    info->caps = base.data_array[0].baseinfo.caps;
    info->version.prod_version = 1U;
    info->version.arch_version = 1U;
    return 0;
}

/* Region layout table: entry sizes follow the original HLT helper
 * test_get_opcode_entry_size so migrated cases that walk a region see the same
 * geometry the real driver reports. */
struct ccu_region_table_entry {
    ccu_u_opcode_t op;
    unsigned int entry_size;
    unsigned int entry_count;
};

static const struct ccu_region_table_entry g_ccuRegions[] = {
    {CCU_U_OP_GET_INSTRUCTION, 32U, 1024U},  {CCU_U_OP_GET_GSA, 8U, 2048U},          {CCU_U_OP_GET_XN, 8U, 2048U},
    {CCU_U_OP_GET_CKE, 8U, 2048U},           {CCU_U_OP_GET_PFE, 8U, 2048U},          {CCU_U_OP_GET_CHANNEL, 64U, 512U},
    {CCU_U_OP_GET_JETTY_CTX, 32U, 1024U},    {CCU_U_OP_GET_MISSION_CTX, 64U, 512U},  {CCU_U_OP_GET_LOOP_CTX, 64U, 512U},
    {CCU_U_OP_GET_LOOP_CKE_CTX, 32U, 1024U}, {CCU_U_OP_GET_MEMORY_SLICE, 8U, 4096U},
};
static struct ccu_region g_ccuRegionOut[sizeof(g_ccuRegions) / sizeof(g_ccuRegions[0])];

int get_region_by_op(ccu_u_opcode_t op, struct ccu_region** region)
{
    if (region == NULL) {
        return -1;
    }
    for (unsigned int i = 0; i < sizeof(g_ccuRegions) / sizeof(g_ccuRegions[0]); i++) {
        if (g_ccuRegions[i].op != op) {
            continue;
        }
        struct ccu_region* r = &g_ccuRegionOut[i];
        r->ccu_u_op = op;
        r->ccu_region_saddr = 0x1000U * (i + 1U);
        r->ccu_region_size = g_ccuRegions[i].entry_size * g_ccuRegions[i].entry_count;
        r->ccu_entry_size = g_ccuRegions[i].entry_size;
        r->ccu_va_udie0 = NULL; /* no mmap in the sim; users only read the geometry */
        r->ccu_va_udie1 = NULL;
        *region = r;
        return 0;
    }
    *region = NULL;
    return -1;
}
