/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * See the License in the specific file packaged with this Software for details.
 */

// CommInit 入口检查专用的受控 kernel 动态库：符号名与 HCOMM 真实内核一致
// （RunAicpuCommInit / RunAicpuIndOpCommInit），仅记录“是否被调用/收到的
// 参数”，返回 0。调用计数与收到的首块 deviceAddr 经导出变量供测试读取——
// 被测代码在必需地址映射失败时必须不调用本内核（计数保持 0）。
// 参数布局复用模拟器侧 ABI 镜像头（aicpu_args_stub.h 的 CommAicpuParam，
// 已与冻结 HCOMM 48b54e99 真身逐字段核对一致），不做手工偏移。
#include <atomic>
#include <cstdint>
#include <cstring>

#include "aicpu_args_stub.h"

extern "C" std::atomic<uint32_t> g_test_comminit_call_count;
extern "C" uint64_t g_test_comminit_last_device_addr;

std::atomic<uint32_t> g_test_comminit_call_count{0};
uint64_t g_test_comminit_last_device_addr{0};

extern "C" uint32_t RunAicpuCommInit(void* args)
{
    g_test_comminit_call_count.fetch_add(1);
    if (args != nullptr) {
        const CommAicpuParam* param = static_cast<const CommAicpuParam*>(args);
        g_test_comminit_last_device_addr = param->kfcControlTransferH2DParams.deviceAddr;
    }
    return 0;
}

extern "C" uint32_t RunAicpuIndOpCommInit(void* args) { return RunAicpuCommInit(args); }

// 线程类内核参数透传回归：四个入口共用同一组桩——记录收到的 args 指针与
// 首 8 字节内容（验证参数原样交给 kernel）；返回值经 g_test_thread_ret_value
// 由测试侧配置（验证 kernel 非零返回沿 status 链传回设备响应）。
extern "C" std::atomic<uint32_t> g_test_thread_call_count;
extern "C" void* g_test_thread_last_args;
extern "C" uint64_t g_test_thread_last_args_head;
extern "C" std::atomic<uint32_t> g_test_thread_ret_value;

std::atomic<uint32_t> g_test_thread_call_count{0};
void* g_test_thread_last_args = nullptr;
uint64_t g_test_thread_last_args_head = 0;
std::atomic<uint32_t> g_test_thread_ret_value{0};

static uint32_t RecordThreadArgs(void* args)
{
    g_test_thread_call_count.fetch_add(1);
    g_test_thread_last_args = args;
    if (args != nullptr) {
        std::memcpy(&g_test_thread_last_args_head, args, sizeof(g_test_thread_last_args_head));
    }
    return g_test_thread_ret_value.load();
}

extern "C" uint32_t RunAicpuThreadInit(void* args) { return RecordThreadArgs(args); }

extern "C" uint32_t RunAicpuThreadDestroy(void* args) { return RecordThreadArgs(args); }

extern "C" uint32_t RunAicpuThreadSupplementNotify(void* args) { return RecordThreadArgs(args); }

extern "C" uint32_t RunAicpuIndOpThreadInit(void* args) { return RecordThreadArgs(args); }
