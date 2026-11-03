/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// libhccl 经 HcclDlopen("libhcomm.so") + HcclDlsym(handle, ...) 取 hcomm 符号。
// 本文件以强别名拦截 HcclDlopen/HcclDlsym/HcclDlclose: __HcclDlsymSub 先查
// RTLD_DEFAULT(LD_PRELOAD 全局符号, 命中 level1 桩), 未命中再回退显式句柄。
// 该拦截仅依赖 dlfcn, 不含 hcomm 头文件, 必须在所有构建模式(含 level1 裁剪)下编译,
// 否则 level1 假 HcclComm 句柄会被直通进真实 libhcomm 解引用导致段错误。
// 置于 level0: LD_PRELOAD("level0:level1:level2") 的最前层级, 保证全局符号
// 解析最优先命中本拦截。

// 日志染色: 模块 tag (须在 include sim_log.h 之前)
#define HCCL_VM_MODULE "DLSYM_STUB"

#include <dlfcn.h>

#include "sim_log.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#if !defined(strong_alias)
#define strong_alias(name, aliasname) _strong_alias(name, aliasname)
#define _strong_alias(name, aliasname) extern __typeof(name) aliasname __attribute__((alias(#name)));
#endif

void* __HcclDlopenSub(const char* libName, int mode)
{
    const char* showLib = (libName != nullptr) ? libName : "(nullptr)";

    void* handle = dlopen(libName, mode);
    if (handle == nullptr) {
        HCCL_VM_ERROR("dlopen {} failed: {}", libName, dlerror());
        return nullptr;
    }
    HCCL_VM_INFO("dlopen libName={} handle={}", showLib, handle);
    return handle;
}

void* __HcclDlsymSub(void* handle, const char* funcName)
{
    const char* showFunc = (funcName != nullptr) ? funcName : "(nullptr)";
    HCCL_VM_INFO("__HcclDlsymSub enter, handle={:p}, funcName={}", handle, showFunc);

    void* addr = dlsym(RTLD_DEFAULT, funcName);
    if (addr == nullptr) {
        HCCL_VM_WARN("rtld default not find {}: [{}]", funcName, dlerror());
        addr = dlsym(handle, funcName);
        if (addr == nullptr) {
            HCCL_VM_WARN("not find {}: [{}]", funcName, dlerror());
            return nullptr;
        }
        return addr;
    }
    return addr;
}

int __HcclDlcloseSub(void* handle)
{
    if (handle == nullptr) {
        return 0;
    }
    return dlclose(handle);
}

strong_alias(__HcclDlopenSub, HcclDlopen);
strong_alias(__HcclDlcloseSub, HcclDlclose);
strong_alias(__HcclDlsymSub, HcclDlsym);

#ifdef __cplusplus
}
#endif // __cplusplus
