/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdarg>
#include <cstdint>
#include <cstdio>

#include "sim_log.h"
#include "log.h"

constexpr size_t LOG_MSG_BUFFER_SIZE = 1024;

#ifdef __cplusplus
extern "C" {
#endif

int32_t CheckLogLevel(int32_t moduleId, int32_t logLevel)
{
    // cann(host)：转发 CANN 真库判定；其余情况保持既有"全部放行"的打桩语义。
    int32_t native = hcclvm::ForwardNativeCheckLogLevel(moduleId, logLevel);
    if (native >= 0) {
        return native;
    }
    return static_cast<int32_t>(true);
}

void DlogPrintStub(int level, const char* msgBuffer)
{
    switch (level) {
        case 0:
            HCCL_VM_DEBUG("{}", msgBuffer);
            break;
        case 1:
            HCCL_VM_INFO("{}", msgBuffer);
            break;
        case 2:
            HCCL_VM_WARN("{}", msgBuffer);
            break;
        case 3:
            HCCL_VM_ERROR("{}", msgBuffer);
            break;
        default:
            HCCL_VM_TRACE("{}", msgBuffer);
    }
}

void DlogRecord(int moduleId, int level, const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    // cann(host)：原生记录原样透传(moduleId 已含 debug/run 通道掩码)；
    // 其余情况继续走既有桩桥接。
    if (hcclvm::ForwardNativeDlogRecord(moduleId, level, fmt, args)) {
        va_end(args);
        return;
    }
    // 定义一个缓冲区，用于存储格式化后的字符串
    char buffer[LOG_MSG_BUFFER_SIZE] = {0};
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);

    DlogPrintStub(level, buffer);
}

#ifdef __cplusplus
}
#endif

bool HcclCheckLogLevel(int logType, int moduleId)
{
    // 本接口的参数顺序与真库相反(logType 在前、moduleId 在后)：cann 时转换成
    // CheckLogLevel(moduleId, logType) 再问真库，保证等级/通道判定口径一致。
    int32_t native = hcclvm::ForwardNativeCheckLogLevel(moduleId, logType);
    if (native >= 0) {
        return native == 1;
    }
    (void)logType;
    (void)moduleId;
    return true;
}

bool IsErrorToWarn() { return false; }
