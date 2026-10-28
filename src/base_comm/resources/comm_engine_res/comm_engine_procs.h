/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_COMM_ENGINE_PROCS_H
#define HCOMM_COMM_ENGINE_PROCS_H

#include "hccl/hccl_types.h"
#include "common/exit_profile.h"
#include "aicpu_comm_engine_bin_proc.h"
#include "ccu_instance_proc.h"
#include "hdc_proc.h"

namespace hcomm {

/**
 * @brief AIV 通信引擎 bin（占位节点，当前无实体）。
 */
class AivCommEngineBinProc {
public:
    void DeInit() {}

    ~AivCommEngineBinProc() = default;

public:
    AivCommEngineBinProc() = default;

private:
    AivCommEngineBinProc(const AivCommEngineBinProc&) = delete;
    AivCommEngineBinProc& operator=(const AivCommEngineBinProc&) = delete;
};

/**
 * @brief DPU 通信引擎 bin（占位节点，当前无实体）。
 */
class DpuCommEngineBinProc {
public:
    void DeInit() {}

    ~DpuCommEngineBinProc() = default;

public:
    DpuCommEngineBinProc() = default;

private:
    DpuCommEngineBinProc(const DpuCommEngineBinProc&) = delete;
    DpuCommEngineBinProc& operator=(const DpuCommEngineBinProc&) = delete;
};

/**
 * @brief bin 族聚合节点（进程级，树根直属）。
 */
class CommEngineBinMgrProc {
public:
    void DeInit()
    {
        dpuCommEngineBinProc_.DeInit();
        aivCommEngineBinProc_.DeInit();
        aicpuCommEngineBinProc_.DeInit();
    }

    AicpuCommEngineBinProc& GetAicpuCommEngineBinProc() { return aicpuCommEngineBinProc_; }
    AivCommEngineBinProc& GetAivCommEngineBinProc() { return aivCommEngineBinProc_; }
    DpuCommEngineBinProc& GetDpuCommEngineBinProc() { return dpuCommEngineBinProc_; }

public:
    ~CommEngineBinMgrProc() = default;

public:
    CommEngineBinMgrProc() = default;

private:
    CommEngineBinMgrProc(const CommEngineBinMgrProc&) = delete;
    CommEngineBinMgrProc& operator=(const CommEngineBinMgrProc&) = delete;

    // 声明顺序 = 析构契约反序（依据见 README R3）
    DpuCommEngineBinProc dpuCommEngineBinProc_;     // [4]
    AivCommEngineBinProc aivCommEngineBinProc_;     // [4]
    AicpuCommEngineBinProc aicpuCommEngineBinProc_; // [4]
};

/**
 * @brief AICPU_TS 引擎线程节点（占位节点，设备级，当前无实体）；析构契约 [1] 停 Thread。
 */
class TsThreadMgrProc {
public:
    void DeInit() {}

public:
    ~TsThreadMgrProc() = default;

public:
    TsThreadMgrProc() = default;

private:
    TsThreadMgrProc(const TsThreadMgrProc&) = delete;
    TsThreadMgrProc& operator=(const TsThreadMgrProc&) = delete;
};

/**
 * @brief CPU_TS 引擎线程节点（占位节点，设备级，当前无实体）；析构契约 [1] 停 Thread。
 */
class CpuThreadMgrProc {
public:
    void DeInit() {}

public:
    ~CpuThreadMgrProc() = default;

public:
    CpuThreadMgrProc() = default;

private:
    CpuThreadMgrProc(const CpuThreadMgrProc&) = delete;
    CpuThreadMgrProc& operator=(const CpuThreadMgrProc&) = delete;
};

/**
 * @brief 线程族聚合节点（设备级）。
 */
class ThreadMgrProc {
public:
    void DeInit()
    {
        cpuThreadMgrProc_.DeInit();
        tsThreadMgrProc_.DeInit();
    }

    TsThreadMgrProc& GetTsThreadMgrProc() { return tsThreadMgrProc_; }
    CpuThreadMgrProc& GetCpuThreadMgrProc() { return cpuThreadMgrProc_; }

public:
    ~ThreadMgrProc() = default;

public:
    ThreadMgrProc() = default;

private:
    ThreadMgrProc(const ThreadMgrProc&) = delete;
    ThreadMgrProc& operator=(const ThreadMgrProc&) = delete;

    // 声明顺序 = 析构契约反序（依据见 README R3）
    CpuThreadMgrProc cpuThreadMgrProc_; // [1]
    TsThreadMgrProc tsThreadMgrProc_;   // [1]
};

/**
 * @brief 内存管理节点（占位节点，设备级；当前无实体）。
 */
class MemMgrProc {
public:
    void DeInit() {}

public:
    ~MemMgrProc() = default;

public:
    MemMgrProc() = default;

private:
    MemMgrProc(const MemMgrProc&) = delete;
    MemMgrProc& operator=(const MemMgrProc&) = delete;
};

/**
 * @brief 引擎族管理器（聚合节点，设备级）：无自有资源，按契约编排子节点的声明序与收尾动作。
 */
class CommEngineMgrProc {
public:
    // 契约 [1] 停 Thread：须最先调用（见 BaseCommMgr::DeInit）；RESET 跳过（停 TS 线程触设备）
    HcclResult StopThreads(ExitProfile profile = ExitProfile::EXIT_PROFILE_GRACEFUL)
    {
        if (profile == ExitProfile::EXIT_PROFILE_RESET) {
            return HCCL_SUCCESS;
        }
        threadMgrProc_.DeInit();
        return HCCL_SUCCESS;
    }

    // 契约 [4] 引擎族收尾；停线程仅经 StopThreads（契约 [1]），HDC/CCU 收尾由节点内按 profile 分流（README R5）
    void DeInit(ExitProfile profile = ExitProfile::EXIT_PROFILE_GRACEFUL)
    {
        memMgrProc_.DeInit(); // host 侧占位，两 profile 均执行
        hdcProc_.DeInit(profile);
        ccuInstanceProc_.DeInit(profile);
    }

    CcuInstanceProc& GetCcuInstanceProc() { return ccuInstanceProc_; }
    HdcProc& GetHdcProc() { return hdcProc_; }
    ThreadMgrProc& GetThreadMgrProc() { return threadMgrProc_; }
    MemMgrProc& GetMemMgrProc() { return memMgrProc_; }

public:
    ~CommEngineMgrProc() = default;

public:
    CommEngineMgrProc() = default;

private:
    CommEngineMgrProc(const CommEngineMgrProc&) = delete;
    CommEngineMgrProc& operator=(const CommEngineMgrProc&) = delete;

    // 声明顺序 = 析构契约反序（依据见 README R3）
    CcuInstanceProc ccuInstanceProc_; // 族内最后死（析构期归还依赖）
    HdcProc hdcProc_;
    MemMgrProc memMgrProc_;       // [3] Mem（占位）
    ThreadMgrProc threadMgrProc_; // [1] 停 Thread
};

} // namespace hcomm

#endif // HCOMM_COMM_ENGINE_PROCS_H
