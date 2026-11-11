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

#include <vector>
#include <memory>
#include <unordered_map>
#include "hcomm_res_entity_defs.h"
#include "mem_device_pub.h"

namespace hccl {
class Thread;

// Host 侧持有一个 Device 导出实例的 Entity 及关联内存，不负责实际线程和 kernel 调度。
// 更新和析构前，调用方必须确保 Device 不再使用被更新或释放的资源。
class ThreadEntityResources {
public:
    struct InitParam {
        HcommDevInfo devInfo{};
        ThreadType threadType{THREAD_TYPE_INVALID};
        CommEngine engine{COMM_ENGINE_RESERVED};
        bool initPodFields{false};
        bool hasRtsq{false};
        uint32_t streamId{0};
        uint32_t sqId{0};
        std::vector<RegedNotifyEntity> notifies;
    };

    ThreadEntityResources() = default;
    ThreadEntityResources(const ThreadEntityResources&) = delete;
    ThreadEntityResources& operator=(const ThreadEntityResources&) = delete;

    // initPodFields 为 true 时初始化 POD 字段；hasRtsq 为 true 时额外分配 RTSQ/SQE 内存，Notify 段按需分配。
    HcclResult Init(const InitParam& param);
    // 仅写入已构建好的 Notify 描述数组；RTSQ 运行状态由 Device 线程持有，不在此处重置。
    HcclResult WriteNotifyBuffer(const std::vector<RegedNotifyEntity>& notifies);
    // 按当前 Thread 的 Notify 数重建并刷新描述数组；A3 在构建前短路。
    HcclResult RefreshNotifyBuffer(const Thread& thread);
    // 已分配的 Entity 地址；是否完成 Device 初始化，由外层发布的 handle 表示。
    ThreadHandle GetEntityHandle() const { return reinterpret_cast<ThreadHandle>(fixedMem_.ptr()); }

private:
    // 分配一个 Notify 段并入表；失败返回 nullptr。
    RegedNotifyEntity* NewNotifySegment();

    DeviceMem fixedMem_;          // Entity；hasRtsq 时还包含 RTSQ 上下文及 SQE 缓存
    bool podFieldsInited_{false}; // Init 时按设备代际记录；A3 恒 false，Update 据此跳过 Notify POD
    std::vector<DeviceMem> notifySegments_; // 各 Notify 段内存；只增不删，析构时统一释放
};

HcclResult BuildThreadEntityNotifies(const Thread& thread, std::vector<RegedNotifyEntity>& notifies);
struct ThreadHandleEntry {
    // 发布状态：launch 成功后登记 device entity 句柄；0 表示已预分配但未发布。
    ThreadHandle handle{0};
    // 资源所有权：PreAlloc 成功即挂载，保证返回的 entity 句柄背后内存不被释放。
    std::unique_ptr<ThreadEntityResources> entityResources;
};

HcclResult PreAllocThreadEntityCommon(
    Thread& thread, std::unordered_map<CommEngine, ThreadHandleEntry>& entries, ThreadHandle& handle,
    CommEngine targetEngine);
ThreadEntityResources* GetThreadEntityResourcesCommon(
    const std::unordered_map<CommEngine, ThreadHandleEntry>& entries, CommEngine targetEngine);
HcclResult PreAllocThreadEntity(Thread& thread, ThreadHandle& handle, CommEngine targetEngine);
ThreadEntityResources* GetThreadEntityResources(const Thread& thread, CommEngine targetEngine);

} // namespace hccl
#endif // THREAD_ENTITY_RESOURCES_H
