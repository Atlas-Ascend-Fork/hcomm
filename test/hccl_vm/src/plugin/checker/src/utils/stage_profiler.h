/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STAGE_PROFILER_H
#define STAGE_PROFILER_H

#include <chrono>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>

#include "sim_log.h"

namespace HcclSim {

constexpr uint64_t PROFILING_KB_FACTOR = 1024ULL;

inline uint64_t GetProcessRssKb()
{
    std::ifstream statmFile("/proc/self/statm");
    uint64_t totalPages = 0;
    uint64_t residentPages = 0;
    if (!(statmFile >> totalPages >> residentPages)) {
        return 0;
    }
    const long pageSize = sysconf(_SC_PAGESIZE);
    if (pageSize <= 0) {
        return 0;
    }
    return residentPages * static_cast<uint64_t>(pageSize) / PROFILING_KB_FACTOR;
}

inline uint64_t GetProcessHwmKb()
{
    std::ifstream statusFile("/proc/self/status");
    std::string line;
    while (std::getline(statusFile, line)) {
        constexpr const char* HWM_PREFIX = "VmHWM:";
        if (line.rfind(HWM_PREFIX, 0) != 0) {
            continue;
        }
        std::istringstream valueStream(line.substr(sizeof(HWM_PREFIX) - 1));
        uint64_t hwmKb = 0;
        if (valueStream >> hwmKb) {
            return hwmKb;
        }
        return 0;
    }
    return 0;
}

inline void LogProcessMemorySnapshot(const std::string& tag)
{
    HCCL_VM_INFO("[MemorySnapshot] tag={}, rssKb={}, hwmKb={}", tag, GetProcessRssKb(), GetProcessHwmKb());
}

inline void LogMatrixMemory(const std::string& tag, uint64_t rows, uint64_t wordsPerRow)
{
    const uint64_t bytes = rows * wordsPerRow * sizeof(uint64_t);
    HCCL_VM_INFO(
        "[MatrixMemory] tag={}, rows={}, wordsPerRow={}, bytes={}, kb={}, mb={}", tag, rows, wordsPerRow, bytes,
        bytes / PROFILING_KB_FACTOR, bytes / (PROFILING_KB_FACTOR * PROFILING_KB_FACTOR));
}

class StageProfiler {
public:
    explicit StageProfiler(std::string stage)
        : stage_(std::move(stage)),
          startTime_(Clock::now()),
          startRssKb_(GetProcessRssKb())
    {}

    ~StageProfiler() { End(); }

    StageProfiler(const StageProfiler&) = delete;
    StageProfiler& operator=(const StageProfiler&) = delete;

    void End()
    {
        if (finished_) {
            return;
        }
        finished_ = true;
        const uint64_t endRssKb = GetProcessRssKb();
        const double costMs = std::chrono::duration<double, std::milli>(Clock::now() - startTime_).count();
        const int64_t rssDeltaKb = static_cast<int64_t>(endRssKb) - static_cast<int64_t>(startRssKb_);
        HCCL_VM_INFO(
            "[StageProfiling] stage={}, costMs={:.2f}, rssStartKb={}, rssEndKb={}, rssDeltaKb={}, "
            "hwmKb={}",
            stage_, costMs, startRssKb_, endRssKb, rssDeltaKb, GetProcessHwmKb());
    }

private:
    using Clock = std::chrono::steady_clock;

    std::string stage_;
    Clock::time_point startTime_;
    uint64_t startRssKb_{0};
    bool finished_{false};
};

} // namespace HcclSim

#endif // STAGE_PROFILER_H
