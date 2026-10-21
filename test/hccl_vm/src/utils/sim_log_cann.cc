/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sim_log_backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include "spdlog/fmt/fmt.h"

namespace hcclvm {
namespace {
    struct CannLogFunctions {
        void (*record)(int32_t, int32_t, const char*, ...){nullptr};
        void (*recordVaList)(int32_t, int32_t, const char*, va_list){nullptr};
        int32_t (*checkLevel)(int32_t, int32_t){nullptr};
        void (*flush)(){nullptr};
    };

    const CannLogFunctions& LoadCannLogLibrary()
    {
        // 仅避免 dlopen 的库构造器回调日志桩时递归初始化同一个 static。
        // 普通并发首入口由 C++ static 初始化同步，不丢弃竞争线程的日志。
        static thread_local bool loadingLibrary = false;
        static const CannLogFunctions unavailable{};
        if (loadingLibrary)
            return unavailable;
        static const CannLogFunctions functions = [] {
            loadingLibrary = true;
            struct LoadingGuard {
                ~LoadingGuard() { loadingLibrary = false; }
            } guard;
            CannLogFunctions resolved;
            // 只使用 set_env.sh 的标准动态库搜索环境；句柄保留至进程结束。
            void* handle = dlopen("libunified_dlog.so", RTLD_NOW | RTLD_LOCAL);
            if (!handle) {
                fprintf(stderr, "[HCCL-VM][ERROR] CANN log library load failed: %s\n", dlerror());
                return resolved;
            }
            resolved.record = reinterpret_cast<decltype(resolved.record)>(dlsym(handle, "DlogRecord"));
            resolved.recordVaList = reinterpret_cast<decltype(resolved.recordVaList)>(dlsym(handle, "DlogVaList"));
            resolved.checkLevel = reinterpret_cast<decltype(resolved.checkLevel)>(dlsym(handle, "CheckLogLevel"));
            resolved.flush = reinterpret_cast<decltype(resolved.flush)>(dlsym(handle, "DlogFlush"));
            if (!resolved.record || !resolved.recordVaList || !resolved.checkLevel) {
                fprintf(
                    stderr,
                    "[HCCL-VM][ERROR] CANN log library load failed: missing DlogRecord/DlogVaList/CheckLogLevel.\n");
                return CannLogFunctions{};
            }
            return resolved;
        }();
        return functions;
    }

    decltype(CannLogFunctions::record) cannRecord = nullptr;
    decltype(CannLogFunctions::flush) cannFlush = nullptr;

    void WriteCannLog(
        size_t moduleIndex, spdlog::level::level_enum level, spdlog::source_loc source, const char* tag,
        fmt::string_view format, fmt::format_args args)
    {
        // 宏已按缓存阈值过滤；仅为兼容 {} 语法格式化一次，真库负责最终过滤、同步和文件管理。
        const auto body = fmt::vformat(format, args);
        const char* file = source.filename ? source.filename : "unknown";
        const char* slash = strrchr(file, '/');
        constexpr int32_t levels[] = {0, 0, 1, 2, 3, 3};
        cannRecord(
            kCannLogModules[moduleIndex].id, levels[level], "[%s:%d]%s %s", slash ? slash + 1 : file, source.line, tag,
            body.c_str());
    }
} // namespace

LogBackend GetLogBackend()
{
    static const LogBackend backend = [] {
        const char* value = getenv("HCCL_VM_LOG_MODE");
        if (!value || !*value || strcmp(value, "stub") == 0)
            return LogBackend::Stub;
        if (strcmp(value, "cann") == 0)
            return LogBackend::Cann;
        fprintf(stderr, "[HCCL-VM][WARN] invalid HCCL_VM_LOG_MODE=[%s]; using stub.\n", value);
        return LogBackend::Stub;
    }();
    return backend;
}

bool IsDeviceRole()
{
#ifdef HCCLVM_LOG_PROCESS_ROLE_DEVICE
    return true;
#else
    return false;
#endif
}

LogWriter InitializeCannLog()
{
    const auto& functions = LoadCannLogLibrary();
    cannRecord = functions.record;
    cannFlush = functions.flush;
    return cannRecord ? WriteCannLog : nullptr;
}

void FlushCannLog()
{
    if (cannFlush)
        cannFlush();
}

int32_t ForwardNativeCheckLogLevel(int32_t moduleAndType, int32_t level)
{
    if (IsDeviceRole() || !IsCannBackend())
        return -1;
    auto checkLevel = LoadCannLogLibrary().checkLevel;
    return checkLevel ? checkLevel(moduleAndType, level) : 0;
}

bool ForwardNativeDlogRecord(int32_t moduleAndType, int32_t level, const char* format, va_list args)
{
    if (IsDeviceRole() || !IsCannBackend())
        return false;
    if (auto record = LoadCannLogLibrary().recordVaList)
        record(moduleAndType, level, format, args);
    return true; // 已选择 CANN；缺库也不回退到 stub
}

} // namespace hcclvm
