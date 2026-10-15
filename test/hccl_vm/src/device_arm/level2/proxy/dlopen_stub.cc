/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define HCCL_VM_MODULE "DLOPEN_STUB"

#include <cstring>
#include <dlfcn.h>

#include "sim_log.h"

typedef void* (*real_dlopen_t)(const char*, int);

// 通过 dlsym(RTLD_NEXT, "dlopen") 获取 glibc 原生的真实 dlopen。
// 使用 C++11 magic statics 保证线程安全的 once 初始化，避免数据竞争。
// 该实现是安全的：我们仅劫持 stub dlopen，并未拦截 dlsym 本身，
// 因此 RTLD_NEXT 可以正确链式调用到 glibc 的 dlopen，不会产生递归问题。
static real_dlopen_t get_real_dlopen()
{
    static real_dlopen_t realDlopen = []() -> real_dlopen_t {
        void* addr = ::dlsym(RTLD_NEXT, "dlopen");
        if (addr == nullptr) {
            HCCL_VM_ERROR("dlopen stub: failed to resolve real dlopen via RTLD_NEXT: {}", dlerror());
            return nullptr;
        }
        HCCL_VM_INFO("dlopen stub: real dlopen resolved successfully, addr={:p}", (void*)addr);
        return reinterpret_cast<real_dlopen_t>(addr);
    }();
    return realDlopen;
}

// 仅劫持 basename 为 "libascend_hal.so" 的加载
static bool IsHalLib(const char* filename)
{
    if (filename == nullptr) {
        return false;
    }
    const char* base = strrchr(filename, '/');
    base = (base != nullptr) ? base + 1 : filename;
    return strcmp(base, "libascend_hal.so") == 0;
}

extern "C" void* dlopen(const char* filename, int flags)
{
    real_dlopen_t realDlopen = get_real_dlopen();
    if (!realDlopen) {
        return nullptr;
    }

    if (IsHalLib(filename)) {
        // 返回主程序句柄(NULL)，使后续 dlsym 走全局作用域搜索：
        // LD_PRELOAD 的本库桩符号优先于 CANN devlib 真实库符号被命中
        void* mainHandle = realDlopen(nullptr, flags);
        HCCL_VM_INFO(
            "dlopen stub: intercepted dlopen(\"{}\"), returning main-program handle {:p}", filename, mainHandle);
        return mainHandle;
    }

    // 其他 so 转发调用系统的 dlopen
    return realDlopen(filename, flags);
}
