/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <atomic>
#include <cstdint>
#include <iostream>
#include <map>
#include <mutex>
#include <sys/mman.h>
#include <store/store_sim_store_pub.h>

#include "sim_log.h"
#include "operation_data/operation_data_ops.h"
#include "store_sim_shm_memory_common.h"

using namespace HcclSim;

void VmPtrReleaser::operator()(void* ptr) const
{
    if (ptr != nullptr) {
        sim::ReleaseInNoHostProcess(phyMem);
    }
}

namespace {
// CCU 缓冲(MS) 地址窗口：与 level1(ccu_kernel_sim.cc 的 CCU_MS_BASE) 及
// checker(task_meta_translator_v3.cc 的 CCU_MS_WINDOW_BASE) 保持一致。
// 该段为合成地址（BASE + cursor），只由 CCU 北向(level1)产出；南向 CCU 用 msId 切片、
// AICPU/AIV 用真实 device 地址，均不会落到该段。
constexpr uint64_t kCcuMsWindowBase = 0x08000000ULL;
constexpr uint64_t kCcuMsWindowCap = 64ULL * 1024ULL * 1024ULL;
// 无 device 语义的调用（如 AIV 路径）使用该保留值，MS 兜底为单份 scratch。
constexpr uint32_t kUnknownDeviceId = 0xFFFFFFFFU;

/**
 * @brief 取指定 device 的 CCU 缓冲(MS) 后备内存（进程本地、懒分配 64MB 匿名映射）。
 *
 * 必须按 device 隔离：level1 各 rank 使用同一 MS 基址（0x8000000），而一个 runner 进程
 * 服务所有 rank；若不隔离会互相踩踏。匿名映射按访问触页，未用到的部分不占物理内存。
 * @return 后备内存基址；分配失败返回 nullptr。
 */
void* GetCcuMsScratch(uint32_t deviceId)
{
    static std::mutex scratchMutex;
    static std::map<uint32_t, void*> scratchByDevice;
    std::lock_guard<std::mutex> lock(scratchMutex);
    auto iter = scratchByDevice.find(deviceId);
    if (iter != scratchByDevice.end()) {
        return iter->second;
    }
    void* base = mmap(nullptr, kCcuMsWindowCap, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED) {
        HCCL_VM_ERROR("cannot allocate ccu MS scratch memory, deviceId={}", deviceId);
        return nullptr;
    }
    scratchByDevice[deviceId] = base;
    HCCL_VM_INFO("ccu MS scratch allocated, deviceId={}, base={:p}, cap={}", deviceId, base, kCcuMsWindowCap);
    return base;
}
} // namespace

// Virtual Runtime
HcclVmResult GetAddrByOffset(uint64_t offset, VmUniquePtr& addrPtr)
{
    return GetAddrByOffset(offset, kUnknownDeviceId, addrPtr);
}

HcclVmResult GetAddrByOffset(uint64_t offset, uint32_t deviceId, VmUniquePtr& addrPtr)
{
    // 1. 入参合法性检查（避免空指针和无效输出）
    if (addrPtr) {
        HCCL_VM_ERROR("错误：指针已持有资源，不可重复获取！");
        return HcclVmResult::HCCL_SIM_E_PARA;
    }

    // 2. CCU 缓冲(MS) 窗口：进程本地、按 device 隔离的后备内存。
    //    地址 = 该 device 的 scratch 基址 + (offset - BASE)；免注册、免边界校验，
    //    语义对齐 checker 的 MS_CCU 特判。
    if (offset >= kCcuMsWindowBase && offset < kCcuMsWindowBase + kCcuMsWindowCap) {
        void* base = GetCcuMsScratch(deviceId);
        if (base == nullptr) {
            return HcclVmResult::HCCL_SIM_E_MEMORY;
        }
        // scratch 常驻进程生命周期；releaser 用空 PhyMemBlock，ReleaseMemByName 对空名安全 no-op。
        addrPtr = VmUniquePtr(static_cast<char*>(base) + (offset - kCcuMsWindowBase), VmPtrReleaser{});
        return HcclVmResult::HCCL_SIM_SUCCESS;
    }

    void* offserPtr = reinterpret_cast<void*>(static_cast<uintptr_t>(offset));
    sim::runtime::PhyMemBlock phyMem{};
    void* addr = sim::AcquireDevPtrInNoHostProcess(offserPtr, phyMem);
    if (addr == nullptr) {
        HCCL_VM_ERROR("错误：无法获取设备地址(addr= {})！", offserPtr);
        return HcclVmResult::HCCL_SIM_E_PTR;
    }

    // 5. 日志打印（可选，便于问题排查）
    HCCL_VM_DEBUG("计算成功, 虚拟地址: {}, 物理地址: {}", offset, addr);

    addrPtr = VmUniquePtr(addr, VmPtrReleaser{phyMem});

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult InsertTaskToCollection(HcclTaskMetaData* task, uint32_t* index)
{
    // 1. 入参合法性检查（避免空指针访问）
    if (task == nullptr) {
        HCCL_VM_ERROR("错误：输入任务指针 task 不能为空！");
        return HcclVmResult::HCCL_SIM_E_PARA;
    }
    if (index == nullptr) {
        HCCL_VM_ERROR("错误：输出索引指针 index 不能为空！");
        return HcclVmResult::HCCL_SIM_E_PARA;
    }

    // 2. 检查 TaskMemoryManager 是否已初始化
    // auto &taskMemMgr = sim::TaskMemoryManager::GetInstance();
    // 3. 写入任务信息到数据库
    sim::operation::OpTaskTab opTaskInfo;
    opTaskInfo.id = 0; // 数据库自增，无需设置
    opTaskInfo.deviceId = task->deviceId;
    opTaskInfo.streamId = task->streamId;
    opTaskInfo.taskType = static_cast<uint32_t>(task->taskType);

    // 序列化 HcclTaskMetaData 到 blob
    opTaskInfo.optaskMeta.assign(
        reinterpret_cast<const uint8_t*>(task), reinterpret_cast<const uint8_t*>(task) + sizeof(HcclTaskMetaData));

    auto ret = sim::operation::InsertOpTask(opTaskInfo);
    if (ret != 0) {
        HCCL_VM_ERROR("错误：插入任务到数据库失败 - {}", ret);
        return HcclVmResult::HCCL_SIM_SHM_FAIL;
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}
