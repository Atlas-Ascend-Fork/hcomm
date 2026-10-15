/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "subcmd_mode.h"

#include <iostream>
#include <string>

#include "cmd_base_utils.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "sim_common_defs.h"
#include "sim_log.h"
#include "runtime_state/sim_models.h"
#include "store_sim_run_mode.h"

namespace HcclSim {

void ModeCommand::ShowMode()
{
    // 读进程内的 g_hcclVmRunMode：mode 命令由宿主进程执行（子 bash 经 MQ 转发过来），
    // 该全局就是宿主当前生效的模式，由 start/run 的 --mode 标志或本命令设置。
    // 重启不会恢复旧模式 —— main.cc 的 envInit() 在每次 hccl-vm start 时 remove_all("data")，
    // SQLite(data/hccl_vm_data.db) 随之被清空重建，RunModeConfig 行不会跨重启残留。
    std::cout << "Supported modes:" << std::endl;
    std::cout << "\t" << sim::ModeName(sim::RUN_MODE_CHECK_ONLY)
              << " : 仅校验模式，大块内存复用 4GB 共享池 HcclCommPool，大幅降低 /dev/shm 占用；"
                 "只保证校验链路，缓冲区数据内容不保证正确"
              << std::endl;
    std::cout << "\t" << sim::ModeName(sim::RUN_MODE_ESTIMATE)
              << " : 估算模式，单卡(rank0)构造幻影对端跳过真实通信阻塞点，"
                 "快速产出任务清单，不真跑数据传输"
              << std::endl;
    std::cout << "\t" << sim::ModeName(sim::RUN_MODE_NORMAL)
              << " : 普通模式(默认)，真实独立分配与完整多 rank 通信，结果正确性无损" << std::endl;
    std::cout << "Current mode: " << sim::ModeName(g_hcclVmRunMode) << std::endl;
}

void ModeCommand::SetMode(sim::RunModeValue mode)
{
    uint8_t cur = g_hcclVmRunMode;
    if (cur == static_cast<uint8_t>(mode)) {
        HCCL_VM_INFO("[HCCL-VM-CMD]Already in {} mode", sim::ModeName(mode));
        return;
    }
    HCCL_VM_INFO("[HCCL-VM-CMD]Switching from {} to {} mode...", sim::ModeName(cur), sim::ModeName(mode));

    // 切换到非 normal 模式前，若 runner 已加载需先卸载（与 run 命令一致），
    // 否则已驻留的 runner 仍按旧模式行为运行。
    if (mode != sim::RUN_MODE_NORMAL && IsRunnerAlive()) {
        HCCL_VM_WARN("[HCCL-VM-CMD]Runner is loaded. Unloading before switching to {} mode...", sim::ModeName(mode));
        auto exitRet = ExitRunnerPlugin();
        if (exitRet != HcclVmResult::HCCL_SIM_SUCCESS) {
            // 卸载失败时必须中止切换：已驻留的 runner 仍按旧模式行为运行，
            // 继续切换只会让模式与实际执行体不一致（且不能谎报卸载成功）。
            HCCL_VM_ERROR(
                "[HCCL-VM-CMD]Failed to unload runner (result={}), mode switch to {} aborted.",
                static_cast<int32_t>(exitRet), sim::ModeName(mode));
            return;
        }
        HCCL_VM_INFO("[HCCL-VM-CMD]Runner unloaded successfully");
    }

    // 复用区 HcclCommPool 只服务于 check-only 模式：切进则建，切出则拆。
    // 放在模式落库之前——建/拆失败即中止切换，避免模式已改而复用区状态不符。
    if (mode == sim::RUN_MODE_CHECK_ONLY) {
        auto poolRet = CreateCommPoolShm();
        if (poolRet != HcclVmResult::HCCL_SIM_SUCCESS) {
            HCCL_VM_ERROR(
                "[HCCL-VM-CMD]Failed to create HcclCommPool, mode switch to {} aborted.", sim::ModeName(mode));
            return;
        }
    } else if (cur == static_cast<uint8_t>(sim::RUN_MODE_CHECK_ONLY)) {
        auto poolRet = DestroyCommPoolShm();
        if (poolRet != HcclVmResult::HCCL_SIM_SUCCESS) {
            HCCL_VM_ERROR(
                "[HCCL-VM-CMD]Failed to destroy HcclCommPool, mode switch to {} aborted.", sim::ModeName(mode));
            return;
        }
    }

    g_hcclVmRunMode = static_cast<uint8_t>(mode);
    // 运行模式单行表：清空后写入新模式（内部 Clear+Insert，非原子顺序与迁移前一致）
    (void)sim::runtime::SetRunMode(g_hcclVmRunMode);
    sim::ResetRunModeCache();
    HCCL_VM_INFO("[HCCL-VM-CMD]Mode switched to {}", sim::ModeName(mode));
}

void ModeCommand::Setup(CLI::App& app)
{
    // 使用「单个子命令 + 单个可选位置参数 + 单回调」。
    // 旧实现是「父命令 mode + 子命令 show/estimate/...」结构，CLI11 在一次调用中会
    // 同时触发父命令与子命令的回调（见 CLI/App.hpp：parsed_ > 0 即执行父 final_callback_），
    // 导致 "Current mode:" 被打印两次。改为单回调后每次调用只输出一行，从根上消除重复。
    // 用户侧命令保持不变：hccl-vm mode show / mode estimate / mode check-only / mode normal。
    auto sub_mode = app.add_subcommand("mode", "查看或切换运行模式 (show/check-only/estimate/normal)");
    sub_mode->add_option("target", targetMode_, "目标模式: show | check-only | estimate | normal (省略则显示当前模式)")
        ->check(CLI::IsMember({"show", "check-only", "estimate", "normal"}));
    sub_mode->callback([this]() {
        if (targetMode_.empty() || targetMode_ == "show") {
            ShowMode();
            return;
        }
        sim::RunModeValue mode = sim::RUN_MODE_NORMAL;
        if (targetMode_ == "check-only") {
            mode = sim::RUN_MODE_CHECK_ONLY;
        } else if (targetMode_ == "estimate") {
            mode = sim::RUN_MODE_ESTIMATE;
        }
        SetMode(mode);
        // 切换路径不再回显支持模式列表与当前模式：SetMode 内部已用 INFO 日志
        // 打印「Mode switched to {mode}」确认切换结果，再调 ShowMode() 会重复刷屏。
        // ShowMode() 仅在 show / 无参分支调用（见本回调开头的 if）。
    });
}

static inline CommandAutoRegister<ModeCommand> g_mode_cmd_reg{};
} // namespace HcclSim
