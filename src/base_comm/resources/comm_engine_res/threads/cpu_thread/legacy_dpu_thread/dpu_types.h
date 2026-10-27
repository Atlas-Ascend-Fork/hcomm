/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LEGACY_DPU_TYPES_H
#define LEGACY_DPU_TYPES_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include "hccl_types.h"
#include "task_param.h"

namespace Hccl {

// DPU 任务回调 ABI（跨 so 共用：宿主侧 TaskService 注册、hccl_v2 兼容入口转发）。
using CallbackTemplate = std::function<int32_t(uint64_t, int32_t)>;
using ProfCallbackTemplate = std::function<HcclResult(const TaskParam&, uint64_t)>;
using ReportCallbackTemplate = std::function<HcclResult()>;
using GetDpuTaskInfoCallbackTemplate = std::function<std::pair<u32, u32>()>;

class TaskService;

/**
 * @brief 收编后的 DPU kernel 参数块（宿主侧与 kernel 侧共享的 ABI）。
 *        宿主侧先构造 `TaskService` 并持有，kernel 侧仅取裸指针下发 `TaskRun`，
 *        不再由 kernel 入口自行实例化，故参数块退化为纯 POD（16 字节）。
 */
struct DpuKernelLaunchParam {
    TaskService* taskService{nullptr};
    uint32_t deviceId{UINT32_MAX};
};
static_assert(sizeof(DpuKernelLaunchParam) == 16, "DpuKernelLaunchParam ABI layout check");
static_assert(offsetof(DpuKernelLaunchParam, deviceId) == 8, "DpuKernelLaunchParam ABI layout check");

/**
 * @brief 宿主侧构造 `TaskService` 所需的入参（由触发点在 `HcclTaskRegister` 帧内整理传入）。
 *        收编后 DPUTAG（100MB）/DPUTASKEXCEPTION（10B）/hostShareBuf 全部由 `LegacyDpuThread` 内部申请托管，
 *        故本结构只保留身份入参（commId/devId），不再携带任何内存地址。
 */
struct TaskServiceArgs {
    std::string commId;
    uint32_t devId{UINT32_MAX};
};

/**
 * @brief Dpu Kernel Launch 申请的共享内存描述（宿主申请、注册后可跨核访问的地址口径）
 */
struct DpuShmem {
    void* originVa_{nullptr}; // ub场景时申请的原始地址
    void* va_{nullptr};       // 申请的地址,pcie场景为原始地址,ub场景为4k对齐后的地址
    void* accessVA_{nullptr}; // 注册后映射的地址
    int64_t connectType_{0};  // 连接类型
};

// 共享内存尺寸
constexpr uint64_t SHARE_HBM_MEMORY_SIZE = (100 * 1024 * 1024); // DPUTAG 100MB
constexpr uint64_t DPU_TASKEXCEPTION_MEMORY_SIZE
    = 20; // |stopflag[1]|hcclret[2]|dstret[2]|opIndex[4]|taskId[4]|streamId[4]|reserved[3]|

// DPU 共享内存标签（DpuShmemMgr 按 tag 管理「申请→注册→取地址→解注册→释放」）
constexpr char DPU_TAG[] = "DPUTAG";
constexpr char DPU_TASK_EXCEPTION_TAG[] = "DPUTASKEXCEPTION";

// DPU 共享内存首字节任务状态机
constexpr uint8_t TASK_UNSET = 0;
constexpr uint8_t TASK_OK = 1;
constexpr uint8_t TASK_TERMINATE = 2;
constexpr uint8_t TASK_TERMINATE_RESPONSE = 3;

} // namespace Hccl

#endif // LEGACY_DPU_TYPES_H
