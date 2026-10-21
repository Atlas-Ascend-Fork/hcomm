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

namespace {
int count = 0;
int lastModule = 0;
int lastLevel = 0;
char lastMessage[2048]{};
} // namespace

// 只捕获转发参数；等级过滤由待测日志宏负责。
extern "C" int32_t CheckLogLevel(int32_t, int32_t) { return 1; }

extern "C" void DlogVaList(int32_t module, int32_t level, const char* format, va_list args)
{
    ++count;
    lastModule = module;
    lastLevel = level;
    vsnprintf(lastMessage, sizeof(lastMessage), format, args);
}

extern "C" void DlogRecord(int32_t module, int32_t level, const char* format, ...)
{
    va_list args;
    va_start(args, format);
    DlogVaList(module, level, format, args);
    va_end(args);
}

extern "C" void DlogFlush() {}
extern "C" void TestLogReset()
{
    count = 0;
    lastMessage[0] = '\0';
}
extern "C" int TestLogCount() { return count; }
extern "C" int TestLogModule() { return lastModule; }
extern "C" int TestLogLevel() { return lastLevel; }
extern "C" const char* TestLogMessage() { return lastMessage; }
