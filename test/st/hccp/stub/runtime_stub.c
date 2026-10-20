/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <dlfcn.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <runtime/runtime/mem.h>
#include <acl/acl_rt.h>
#include "ascend_hal_base.h"
#include "hccp_shm_infra.h"

#ifndef RT_ERROR_INVALID_VALUE
#define RT_ERROR_INVALID_VALUE ((rtError_t)0x07110001)
#endif

/*
 * Production chain: rtMalloc -> ApiImpl::DevMalloc (32B size alignment) ->
 * NpuDriver::DevMemAlloc* -> halMemAlloc(MEM_DEV | MEM_TYPE_HBM | MEM_SET_ALIGN_SIZE(9) |
 * NODE_TO_DEVICE(devid)). This stub mirrors that layering: rtMalloc routes device memory
 * through the hal stub's halMemAlloc, so the UB (runtime) and RoCE (provider) paths both
 * end in the same devmem zone of the consolidated segment (hccp_devmem_alloc,
 * HCCP_DEVMEM_DVA_ADDR window) instead of the legacy global BUF pool. The returned DVA
 * is mapped in the owning process (the whole segment is mapped at the fixed DVA base),
 * so local dereference keeps working; remote access is routed by the urma/ibverbs
 * engines, matching production semantics.
 *
 * libruntime.so must not link a pinned host/device variant of libascend_hal.so, so the hal
 * entry points are resolved with dlsym (the ST binary always loads libascend_hal.so via
 * DT_NEEDED, so this resolves in practice). If the hal library is not loaded in the process,
 * fall back to driving the same per-phyId devmem pool directly.
 */
typedef drvError_t (*rtHalMemAllocFn)(void** pp, unsigned long long size, unsigned long long flag);
typedef drvError_t (*rtHalMemFreeFn)(void* pp);

/* Device id of this process, mirroring HalPhyId() in the hal stub. */
static int RtPhyId(void)
{
    static int phyId = -1;
    if (phyId < 0) {
        const char* s = getenv("HCCP_VM_PHYID");
        if (s != NULL) {
            phyId = atoi(s);
        }
    }
    return phyId;
}

/* Device memory allocation via the hal stub (halMemAlloc MEM_DEV path -> per-phyId pool). */
static void* RtDevMalloc(uint64_t size, uint64_t flag)
{
    rtHalMemAllocFn fn = (rtHalMemAllocFn)dlsym(RTLD_DEFAULT, "halMemAlloc");
    if (fn != NULL) {
        void* va = NULL;
        if (fn(&va, size, flag) == DRV_ERROR_NONE && va != NULL) {
            return va;
        }
        return NULL;
    }
    /* hal library not loaded: same per-phyId devmem pool, direct path */
    return hccp_devmem_alloc(RtPhyId(), (size_t)size);
}

/* Device memory free via the hal stub; must go to the same pool instance that allocated. */
static void RtDevFree(void* dva)
{
    rtHalMemFreeFn fn = (rtHalMemFreeFn)dlsym(RTLD_DEFAULT, "halMemFree");
    if (fn != NULL) {
        (void)fn(dva);
        return;
    }
    hccp_devmem_free(RtPhyId(), dva);
}

/* Shared core of the rtMalloc/aclrtMalloc entry points (see the rt comment
 * above for the production layering); kept as one static core so neither
 * public entry calls the other: rt* is deprecated-declared in the CANN
 * headers, and a call site in this file would warn. */
static rtError_t RtStubMallocImpl(void** devPtr, uint64_t size, rtMemType_t type, const uint16_t moduleId)
{
    if (devPtr == NULL || size == 0) {
        return RT_ERROR_INVALID_VALUE;
    }
    switch (type) {
        case RT_MEMORY_DEFAULT:
        case RT_MEMORY_HBM:
        case RT_MEMORY_RDMA_HBM:
        case RT_MEMORY_DDR:
            break;
        default:
            return RT_ERROR_INVALID_VALUE;
    }
    /* one devmem pool slot per allocation (HCCP_DEVMEM_SIZE bytes per slot) */
    if (size > (uint64_t)HCCP_DEVMEM_SIZE) {
        return RT_ERROR_INVALID_VALUE;
    }
    /* 32B size alignment, mirroring ApiImpl::DevMalloc in the runtime */
    uint64_t aligned = (size + 0x1F) & ~(uint64_t)0x1F;

    /* hal flag layout mirrors the production driver flags (only the virt type bits are
     * consumed by the hal stub; the rest keep the production call shape for fidelity) */
    uint64_t flag = (uint64_t)MEM_DEV;
    switch (type) {
        case RT_MEMORY_DDR:
            flag |= (uint64_t)MEM_TYPE_DDR;
            break;
        case RT_MEMORY_DEFAULT:
        case RT_MEMORY_HBM:
        case RT_MEMORY_RDMA_HBM:
        default:
            flag |= (uint64_t)MEM_TYPE_HBM;
            break;
    }
    flag |= (uint64_t)MEM_SET_ALIGN_SIZE(9ULL); /* 512B alignment, as in production */
    if (RtPhyId() >= 0) {
        flag |= (uint64_t)NODE_TO_DEVICE((uint64_t)RtPhyId()); /* device id in bit0-9 */
    }
    flag |= ((uint64_t)moduleId & 0xFFULL) << MEM_MODULE_ID_BIT;

    void* va = RtDevMalloc(aligned, flag);
    if (va == NULL) {
        return RT_ERROR_INVALID_VALUE;
    }
    *devPtr = va;
    return RT_ERROR_NONE;
}

rtError_t rtMalloc(void** devPtr, uint64_t size, rtMemType_t type, const uint16_t moduleId)
{
    return RtStubMallocImpl(devPtr, size, type, moduleId);
}

/* Shared core of rtFree/aclrtFree: route by slot-validity predicate (see the
 * rtFree comment below). */
static rtError_t RtStubFreeImpl(void* devPtr)
{
    if (devPtr == NULL) {
        return RT_ERROR_INVALID_VALUE;
    }
    /* route by slot-validity predicate: halMemFree would call free() on BUF-pool shm
     * VAs and crash. Decode accepts only true slot DVAs of the consolidated devmem
     * segment (headers are rejected by construction), which is exactly the set
     * RtDevFree may consume; everything else falls to the BUF pool. */
    if (hccp_devmem_dva_decode((uint64_t)(uintptr_t)devPtr, NULL, NULL)) {
        RtDevFree(devPtr); /* devmem zone slot (rtMalloc via halMemAlloc) */
        return RT_ERROR_NONE;
    }
    hccp_shm_free(devPtr); /* legacy global BUF pool VA (urma internal buffers) */
    return RT_ERROR_NONE;
}

rtError_t rtFree(void* devPtr) { return RtStubFreeImpl(devPtr); }

/* --------------------------------------------------------- aclrt entries */
/* ACL runtime (acl_rt.h) entries for device memory. Signatures mirror the real
 * CANN acl_rt.h; the huge-page policy is meaningless for the devmem pool, so
 * every policy maps to the HBM slot pool (aclrtMalloc in production allocates
 * HBM device memory). The rest of the aclrt family lives in runtime_rt_stub.c. */

aclError aclrtMalloc(void** devPtr, size_t size, aclrtMemMallocPolicy policy)
{
    (void)policy;
    return (aclError)RtStubMallocImpl(devPtr, (uint64_t)size, RT_MEMORY_HBM, 0);
}

aclError aclrtFree(void* devPtr) { return (aclError)RtStubFreeImpl(devPtr); }
