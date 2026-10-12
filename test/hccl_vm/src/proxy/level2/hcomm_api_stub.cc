/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 日志染色: 模块 tag (须在 include sim_log.h 之前)
#define HCCL_VM_MODULE "HCOMM_STUB"

#include <cstdint>
#include <dlfcn.h>
#include <string>
#include "hccl/hccl_types.h"
#include "hccl_proxy_common.h"
#include "sim_log.h"

#ifdef __cplusplus
extern "C" {
#endif

HcclResult HcclDevMemAcquire(HcclComm comm, const char* memTag, uint64_t* size, void** addr, bool* newCreated)
{
    // 若符号解析错误(如 RTLD_NEXT 把符号解析回本桩), 重入会在此被立即打断,
    // 避免无限递归把栈打爆(stack overflow)。
    static thread_local bool inCall = false;
    if (inCall) {
        HCCL_VM_ERROR("HcclDevMemAcquire re-entered (self-recursion detected), abort.");
        return HCCL_E_INTERNAL;
    }
    inCall = true;

    HCCL_VM_INFO(
        "[HcclDevMemAcquire] comm={:p}, memTag={}, size_ptr={:p}, addr_ptr={:p}, newCreated_ptr={:p}",
        static_cast<void*>(comm), (memTag != nullptr) ? memTag : "nullptr", static_cast<void*>(size),
        static_cast<void*>(addr), static_cast<void*>(newCreated));

    using RealFunc = HcclResult (*)(HcclComm comm, const char* memTag, uint64_t* size, void** addr, bool* newCreated);

    // RTLD_NEXT 无法穿透 torch_npu 等场景中以 RTLD_LOCAL 方式加载的 libhcomm.so,
    // 失败时按 SONAME 显式加载后重试 (libhcomm.so 导出 HcclDevMemAcquire).
    const auto resolvedFn = reinterpret_cast<RealFunc>(sim::DlsymRealWithFallback(__func__, "libhcomm.so"));
    if (resolvedFn == nullptr) {
        inCall = false;
        return HCCL_E_INTERNAL;
    }

    // 校验解析结果不是 CheckerL2 自己打桩的符号(位于 libhccl_proxy_* 内),
    // 而是来自某个真实库。若被符号插入回 proxy 自己或兄弟 .so, 说明解析错误, 直接拒绝。
    Dl_info resolvedInfo{};
    if (dladdr(reinterpret_cast<void*>(resolvedFn), &resolvedInfo) == 0 || resolvedInfo.dli_fname == nullptr
        || std::string(resolvedInfo.dli_fname).find("libhccl_proxy") != std::string::npos) {
        HCCL_VM_ERROR(
            "HcclDevMemAcquire resolvedFn comes from a proxy stub ({}) , abort to avoid self-call.",
            (resolvedInfo.dli_fname != nullptr) ? resolvedInfo.dli_fname : "unknown");
        inCall = false;
        return HCCL_E_INTERNAL;
    }

    // 转发调用真实底层函数
    HcclResult ret = resolvedFn(comm, memTag, size, addr, newCreated);
    if (ret != HCCL_SUCCESS) {
        HCCL_VM_ERROR("HcclDevMemAcquire realFn failed, ret={}", static_cast<int>(ret));
        inCall = false;
        return ret;
    }

    // 仅在真实函数成功时，将*addr(host进程可访问地址)转化为device进程可访问地址
    *addr = sim::GetDevMapperAddrByHostPtr(*addr);
    if (*addr == nullptr) {
        inCall = false;
        return HCCL_E_PTR;
    }

    inCall = false;
    return ret;
}

#ifdef __cplusplus
}
#endif // __cplusplus
