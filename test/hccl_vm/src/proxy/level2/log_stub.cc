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
#undef HCCL_VM_MODULE
#define HCCL_VM_MODULE "HCCL_LOG"

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <mutex>

#include "sim_log.h"

constexpr size_t LOG_MSG_BUFFER_SIZE = 1024;

std::once_flag log_initialized_flag;

#ifdef __cplusplus
extern "C" {
#endif

int32_t CheckLogLevel(int32_t moduleId, int32_t logLevel)
{
    // cann(host)：转发 CANN 真库的判定结果，保持 debug/run 通道与等级语义；
    // stub 模式 / device 角色保持既有打桩语义；cann 真库不可用则关闭输出。
    int32_t native = hcclvm::ForwardNativeCheckLogLevel(moduleId, logLevel);
    if (native >= 0) {
        return native;
    }
    return static_cast<int32_t>(true);
}

// 业务日志级别门槛 (hccl log.cc HcclCheckLogLevel → weak
// acllogCheckDebugLevel, 未定义时回退 weak dlog_getlevel): 恒返 1 (允许),
// ProbeLogLevel 由此探测得最宽松级别, 保证 INFO 级算法选择日志必达本桩,
// 免疫环境内真实 dlog 库按级别过滤; 工具侧输出仍受 log_config.yaml
// 二级控制, 不会刷屏
int32_t acllogCheckDebugLevel(int32_t moduleId, int32_t logLevel)
{
    (void)moduleId;
    (void)logLevel;
    return 1;
}

void DlogPrintStub(int level, int moduleId, const char* msgBuffer)
{
    std::call_once(log_initialized_flag, []() {
        if (g_logger != nullptr || hcclvm::g_logWriter != nullptr) {
            return;
        }
        LogConfig config = LoadLogConfig("proxy");
        InitLogger(config);
    });

    switch (level) {
        case 0:
            HCCL_VM_DEBUG("[{}]{}", moduleId & 0xFF, msgBuffer);
            break;
        case 1:
            HCCL_VM_INFO("[{}]{}", moduleId & 0xFF, msgBuffer);
            break;
        case 2:
            HCCL_VM_WARN("[{}]{}", moduleId & 0xFF, msgBuffer);
            break;
        case 3:
            HCCL_VM_ERROR("[{}]{}", moduleId & 0xFF, msgBuffer);
            break;
        default:
            HCCL_VM_TRACE("[{}]{}", moduleId & 0xFF, msgBuffer);
    }
}

void DlogRecord(int moduleId, int level, const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    // cann(host)：原生记录原样透传(保留 moduleId 中的通道掩码与 printf/va_list 语义)；
    // stub / device 角色继续走既有桩桥接，cann 失败不回退。
    if (hcclvm::ForwardNativeDlogRecord(moduleId, level, fmt, args)) {
        va_end(args);
        return;
    }
    char buffer[LOG_MSG_BUFFER_SIZE] = {0};
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);

    DlogPrintStub(level, moduleId, buffer);
}

// C++ 符号别名(GCC 中 extern "C" 无重载，这里补一个按老名字调用的入口)。
void _Z10DlogRecordiiPKcz(int moduleId, int level, const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    if (hcclvm::ForwardNativeDlogRecord(moduleId, level, fmt, args)) {
        va_end(args);
        return;
    }
    char buffer[LOG_MSG_BUFFER_SIZE] = {0};
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);

    DlogPrintStub(level, moduleId, buffer);
}

#ifdef __cplusplus
}
#endif
