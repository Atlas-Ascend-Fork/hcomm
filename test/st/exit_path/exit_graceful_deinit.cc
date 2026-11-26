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
 * @file exit_graceful_deinit.cc
 * @brief 退出期专项用例：显式 graceful DeInit 后退出。
 *
 * 验证默认 profile（EXIT_PROFILE_GRACEFUL）的显式 DeInit 路径：完整动作链执行后进程退出不 coredump；
 * 并验证幂等——重复 DeInit 不再执行任何动作（不崩即通过，由 run_exit_path_test.sh 断言
 * 退出码 0；ASAN 构建时同时要求无 use-after-free）。
 */

#include <cstdint>
#include <cstdio>

#include "hcomm_c_adpt.h"
#include "base_comm_mgr.h"

int main()
{
    // 无卡环境走 noDevice 兜底分支，返回值不影响本用例，仅打印记录
    HcommResult ret = HcommResMgrInit();
    std::printf("[exit_graceful_deinit] HcommResMgrInit ret[%d]\n", ret);

    // 显式 graceful DeInit（默认 profile）
    HcclResult deInitRet = hcomm::BaseCommMgr::GetInstance().DeInit();
    std::printf("[exit_graceful_deinit] DeInit(EXIT_PROFILE_GRACEFUL) ret[%d]\n", deInitRet);
    if (deInitRet != HCCL_SUCCESS) {
        std::printf("[exit_graceful_deinit] FAILED: graceful DeInit ret[%d]\n", deInitRet);
        return 1;
    }

    // 幂等验证：重复调用（含换 profile）必须 no-op，不重复执行动作
    deInitRet = hcomm::BaseCommMgr::GetInstance().DeInit(hcomm::ExitProfile::EXIT_PROFILE_RESET);
    std::printf("[exit_graceful_deinit] DeInit(EXIT_PROFILE_RESET) after graceful ret[%d] (expect no-op)\n", deInitRet);
    if (deInitRet != HCCL_SUCCESS) {
        std::printf("[exit_graceful_deinit] FAILED: repeated DeInit ret[%d]\n", deInitRet);
        return 1;
    }

    // 拒建断言：DeInit 后未建过的 devPhyId 必须解析到同一兜底实例
    const std::uintptr_t reserveA
        = reinterpret_cast<std::uintptr_t>(&hcomm::BaseCommMgr::GetInstance().GetBaseCommResPerDev(60));
    const std::uintptr_t reserveB
        = reinterpret_cast<std::uintptr_t>(&hcomm::BaseCommMgr::GetInstance().GetBaseCommResPerDev(61));
    if (reserveA != reserveB) {
        std::printf(
            "[%s] FAILED: post-DeInit accessors must resolve to the same reserve instance\n", "exit_graceful_deinit");
        return 1;
    }
    std::printf("[%s] post-DeInit reserve-reject assertion passed\n", "exit_graceful_deinit");

    // 终态断言：动作链执行后各注册表必须清空（验证动作真实执行，而非仅未崩）
    auto& mgr = hcomm::BaseCommMgr::GetInstance();
    if (mgr.GetEndpointMgr().GetEndpointHandleMgr().HasLeftover()) {
        std::printf("[%s] FAILED: endpoint table not empty after graceful DeInit\n", "exit_graceful_deinit");
        return 1;
    }
    auto& registry = mgr.GetHandleRegistryMgrProc();
    if (registry.GetChannelHandleRegistryProc().HasLeftover()) {
        std::printf("[%s] FAILED: channel table not empty after graceful DeInit\n", "exit_graceful_deinit");
        return 1;
    }
    auto& threadReg = registry.GetThreadHandleRegistryProc();
    if (!threadReg.EngineThreadMap().empty() || !threadReg.EngineThreadD2HMap().empty()
        || !threadReg.OrderCtrlThreads().empty()) {
        std::printf("[%s] FAILED: thread tables not empty after graceful DeInit\n", "exit_graceful_deinit");
        return 1;
    }
    std::printf("[%s] post-DeInit terminal-state assertions passed\n", "exit_graceful_deinit");

    // 退出期 ~BaseCommMgr 因 isDeInited_ 跳过动作链
    return 0;
}
