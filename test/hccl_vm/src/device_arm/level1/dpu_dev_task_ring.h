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
 * Description: HostDPU 北向劫持（level1）设备任务转发环生产者接口。
 *              DpuRingInsertDevTask 是 InsertTaskToCollectionDev 的 DPU 感知
 *              包装：当前线程已注册转发环（经 HcommSendRequest 解析共享内存时
 *              注册，见 dpu_rpc_device_stub.cc）则推送环由 host 执行器统一落库，
 *              否则原样回退 InsertTaskToCollectionDev 直写（南向/level2/纯 AICPU
 *              路径行为不变）。环协议与 host 侧常量镜像见
 *              proxy/level1/dpu/dpu_level1_common.h 第 1b 节。
 * Create: 2026-10-29
 */

#ifndef DPU_DEV_TASK_RING_H
#define DPU_DEV_TASK_RING_H

#include <cstdint>

#include "sim_common_defs.h"

/**
 * @brief 将一条设备任务写入任务表（DPU 模式经转发环，其余模式直写）。
 * @param task 输入：任务元数据（与本进程 InsertTaskToCollectionDev 入参同约定）。
 * @return int32_t 成功返回 0；失败返回非 0（与 InsertTaskToCollectionDev 一致）。
 */
int32_t DpuRingInsertDevTask(const HcclTaskMetaData* task);

#endif // DPU_DEV_TASK_RING_H
