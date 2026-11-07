/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CPU_QUEUE_KERNEL_ENTRANCE_H
#define CPU_QUEUE_KERNEL_ENTRANCE_H

#include <cstdint>

// SQ/CQ 队列轮询 kernel 入口：只声明不导出，正式路径靠算子名字符串 + json 解析（aclrtBinaryGetFunction），
// 故本头不产生任何链接依赖，host侧不需要 include 它
extern "C" {
__attribute__((visibility("default"))) uint32_t RunDpuRpcSrvLaunchNew(const uint64_t args);
}

namespace hccl {
// 入口名：host侧按此名取函数句柄，ccl_dpu.ini 的 section 名必须与之一致（json 由 ini 生成、按名解析），
// 故两处共用本常量，避免各写一份字符串
constexpr const char* DPU_QUEUE_KERNEL_ENTRY_NAME = "RunDpuRpcSrvLaunchNew";
} // namespace hccl

#endif // CPU_QUEUE_KERNEL_ENTRANCE_H
