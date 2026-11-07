/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CPU_DEVICE_QUEUE_NAMECARD_RESTORE_KERNEL_H
#define CPU_DEVICE_QUEUE_NAMECARD_RESTORE_KERNEL_H

#include <cstdint>

extern "C" {
// 队列名片投递入口：把 host 算好的 SQ/CQ 描述、布局与握手地址登记进 device 侧，
// 供 CQ 守护轮询并回填 DPU 线程实体。与 HcclDpuTaskexpShmemRestore 同形，命名沿用其风格。
__attribute__((visibility("default"))) uint32_t RunDpuQueueNamecardRestore(void* args);
}

namespace hccl {
// 入口名：host侧按此名取函数句柄（不在host侧链接本符号），ccl_kernel.ini 的 section 名与 .map 白名单
// 必须与之一致（json 由 ini 生成、按名解析），三处缺一都只在 Release 运行期报「函数解析失败」
constexpr const char* DPU_QUEUE_NAMECARD_RESTORE_KERNEL_NAME = "RunDpuQueueNamecardRestore";
} // namespace hccl

#endif // CPU_DEVICE_QUEUE_NAMECARD_RESTORE_KERNEL_H
