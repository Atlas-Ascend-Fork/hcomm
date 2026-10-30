/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef THREAD_ENTITY_RESOURCES_H
#define THREAD_ENTITY_RESOURCES_H

#include "hcomm_res_entity_defs.h"
#include "mem_device_pub.h"

namespace hccl {
// Host 侧持有一个 Device 导出实例的 Entity 及关联内存，不负责实际线程和 kernel 调度。
// 更新和析构前，调用方必须确保 Device 不再使用被更新或释放的资源。
class ThreadEntityResources {
public:
    ThreadEntityResources() = default;
    ThreadEntityResources(const ThreadEntityResources&) = delete;
    ThreadEntityResources& operator=(const ThreadEntityResources&) = delete;

    // initPodFields 为 true 时分配 RTSQ/SQE/Notify 内存并初始化关联字段。
    HcclResult Init(const HcommDevInfo& devInfo, uint32_t notifyNum, bool initPodFields);
    // 仅扩容 Notify 描述数组；Device kernel 随后填写 Notify ID，不重置 RTSQ 运行状态。
    HcclResult UpdateNotifyBuffer(uint32_t notifyNum);
    // 已分配的 Entity 地址；是否完成 Device 初始化，由外层发布的 handle 表示。
    ThreadHandle GetEntityHandle() const { return reinterpret_cast<ThreadHandle>(fixedMem_.ptr()); }

private:
    DeviceMem fixedMem_;  // Entity；A5 还包含 RTSQ 上下文及 SQE 缓存
    DeviceMem notifyMem_; // 可独立扩容的 Notify 描述数组，容量由 size() 推导
};
} // namespace hccl
#endif // THREAD_ENTITY_RESOURCES_H
