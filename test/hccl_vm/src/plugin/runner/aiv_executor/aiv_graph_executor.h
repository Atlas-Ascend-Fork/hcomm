/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_AIVGRAPHEXECUTOR_H
#define AIV_AIVGRAPHEXECUTOR_H

#include <cstdint>
#include <memory>
#include <queue>
#include <vector>

#include "aiv_task.h"
#include "sim_common_defs.h"

class AivBlock {
public:
    AivBlock(uint32_t blockIdx, size_t maxEventId);
    ~AivBlock() = default;
    AivBlock(const AivBlock&) = delete;
    AivBlock& operator=(const AivBlock&) = delete;

    uint32_t GetBlockIdx() const { return blockIdx_; }
    std::vector<bool>& GetEvents() { return events_; }

private:
    uint32_t blockIdx_{UINT32_MAX};

    std::vector<bool> events_{};
};

class AivGraphExecutor {
public:
    AivGraphExecutor(uint64_t deviceId, uint64_t launchIdx) : deviceId_(deviceId), launchIdx_{launchIdx} {}
    ~AivGraphExecutor() = default;
    AivGraphExecutor(const AivGraphExecutor&) = delete;
    AivGraphExecutor& operator=(const AivGraphExecutor&) = delete;

    bool Init();
    HcclSim::HcclVmResult Execute();

private:
    bool HasTask() const;

    HcclSim::HcclVmResult ExecuteTask(std::shared_ptr<AivSim::AivTask> task);
    HcclSim::HcclVmResult ExecuteTask(std::shared_ptr<AivSim::AivTaskMemCopy> task);
    HcclSim::HcclVmResult ExecuteTask(std::shared_ptr<AivSim::AivTaskReduce> task);
    HcclSim::HcclVmResult ExecuteTask(std::shared_ptr<AivSim::AivTaskSetFlag> task);
    HcclSim::HcclVmResult ExecuteTask(std::shared_ptr<AivSim::AivTaskWaitFlag> task);
    HcclSim::HcclVmResult ExecuteTask(std::shared_ptr<AivSim::AivTaskPipeBarrier> task);
    HcclSim::HcclVmResult ExecuteTask(std::shared_ptr<AivSim::AivTaskSendFlag> task);
    HcclSim::HcclVmResult ExecuteTask(std::shared_ptr<AivSim::AivTaskRecvFlag> task);
    HcclSim::HcclVmResult ExecuteTask(std::shared_ptr<AivSim::AivTaskSyncAll> task);

    template <typename T>
    HcclSim::HcclVmResult Reduce(void* src, void* dst, size_t len, uint32_t reduceOp);

    // For Debug
    void ShowCurrentTaskStatus();

private:
    bool isInitialized_{false};

    uint64_t deviceId_{UINT64_MAX};
    uint64_t launchIdx_{UINT64_MAX};

    std::vector<std::unique_ptr<AivBlock>> aivBlocks_{};
    std::vector<bool> pipeBarrierRegisters_{};
    std::vector<std::vector<bool>> syncAllRegisters_{};
    std::vector<std::queue<std::shared_ptr<AivSim::AivTask>>> aivTaskQueues_{};
};

#endif // AIV_AIVGRAPHEXECUTOR_H
