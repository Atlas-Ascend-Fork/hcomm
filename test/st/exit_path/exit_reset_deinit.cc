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
 * @file exit_reset_deinit.cc
 * @brief 退出期专项用例：显式 reset DeInit 后退出。
 *
 * 验证 EXIT_PROFILE_RESET profile：跳过一切触碰设备的动作，只做 host 侧清理与状态标记后进程退出
 * 不 coredump（不崩即通过，由 run_exit_path_test.sh 断言退出码 0；ASAN 构建时同时
 * 要求无 use-after-free）。
 *
 * 环境适配：reset 路径不触碰设备，无卡 ST 环境可直接运行；
 * 真实硬件复位场景的动作差异不在本用例范围。
 */

#include <cstdint>
#include <cstdio>

#include "hcomm_c_adpt.h"
#include "base_comm_mgr.h"

int main()
{
    // 无卡环境走 noDevice 兜底分支，返回值不影响本用例，仅打印记录
    HcommResult ret = HcommResMgrInit();
    std::printf("[exit_reset_deinit] HcommResMgrInit ret[%d]\n", ret);

    // 显式 reset DeInit：跳过设备动作，仅 host 侧清理
    HcclResult deInitRet = hcomm::BaseCommMgr::GetInstance().DeInit(hcomm::ExitProfile::EXIT_PROFILE_RESET);
    std::printf("[exit_reset_deinit] DeInit(EXIT_PROFILE_RESET) ret[%d]\n", deInitRet);
    if (deInitRet != HCCL_SUCCESS) {
        std::printf("[exit_reset_deinit] FAILED: reset DeInit ret[%d]\n", deInitRet);
        return 1;
    }

    // 幂等验证：先 reset 后 graceful 必须 no-op，不重复执行动作
    deInitRet = hcomm::BaseCommMgr::GetInstance().DeInit(hcomm::ExitProfile::EXIT_PROFILE_GRACEFUL);
    std::printf("[exit_reset_deinit] DeInit(EXIT_PROFILE_GRACEFUL) after reset ret[%d] (expect no-op)\n", deInitRet);
    if (deInitRet != HCCL_SUCCESS) {
        std::printf("[exit_reset_deinit] FAILED: repeated DeInit ret[%d]\n", deInitRet);
        return 1;
    }

    // 拒建断言：DeInit 后未建过的 devPhyId 必须解析到同一兜底实例
    const std::uintptr_t reserveA
        = reinterpret_cast<std::uintptr_t>(&hcomm::BaseCommMgr::GetInstance().GetBaseCommResPerDev(60));
    const std::uintptr_t reserveB
        = reinterpret_cast<std::uintptr_t>(&hcomm::BaseCommMgr::GetInstance().GetBaseCommResPerDev(61));
    if (reserveA != reserveB) {
        std::printf(
            "[%s] FAILED: post-DeInit accessors must resolve to the same reserve instance\n", "exit_reset_deinit");
        return 1;
    }
    std::printf("[%s] post-DeInit reserve-reject assertion passed\n", "exit_reset_deinit");

    // 终态断言：thread 表为纯 host 清理，两 profile 均须清空；
    // endpoint/channel 残留在 RESET 下是被允许的语义（跳过设备侧归还），不在此断言
    auto& threadReg = hcomm::BaseCommMgr::GetInstance().GetHandleRegistryMgrProc().GetThreadHandleRegistryProc();
    if (!threadReg.EngineThreadMap().empty() || !threadReg.EngineThreadD2HMap().empty()
        || !threadReg.OrderCtrlThreads().empty()) {
        std::printf("[%s] FAILED: thread tables not empty after reset DeInit\n", "exit_reset_deinit");
        return 1;
    }
    std::printf("[%s] post-DeInit terminal-state assertions passed\n", "exit_reset_deinit");

    // 退出期 ~BaseCommMgr 因 isDeInited_ 跳过动作链
    return 0;
}
