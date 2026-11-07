/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CPU_DEVICE_THREAD_REGISTRY_H
#define CPU_DEVICE_THREAD_REGISTRY_H

#include "cpu_device_thread.h"

// device 侧的 DPU 线程登记处：实体建立时按句柄登记、销毁时按句柄摘除；投递 kernel 与将来的 AICPU
// 编排侧按句柄反查实体（device 侧对 Thread* 做 dynamic_cast 不可靠，故需要这张显式的反查表）。
// 队列名片（SQ/CQ 描述）留在实体自身，本表只负责找到实体。

/**
 * @brief 登记 DPU 线程实体（键为句柄，即实体在 device 侧的地址）
 */
void RegisterCpuDeviceThread(ThreadHandle handle, std::shared_ptr<hccl::CpuDeviceThread> entity);

/**
 * @brief 按句柄反查 DPU 线程实体
 * @return 命中返回实体（持有一份引用，调用期间不会被销毁侧抽走）；未登记返回 nullptr
 */
std::shared_ptr<hccl::CpuDeviceThread> FindCpuDeviceThread(ThreadHandle handle);

/**
 * @brief 按句柄摘除登记
 * @return true = 该句柄确为已登记的 DPU 线程（销毁方据此跳过 AICPU 线程容器的查找）
 */
bool EraseCpuDeviceThread(ThreadHandle handle);

/**
 * @brief 清空登记表
 */
void ClearCpuDeviceThread();

#endif // CPU_DEVICE_THREAD_REGISTRY_H
