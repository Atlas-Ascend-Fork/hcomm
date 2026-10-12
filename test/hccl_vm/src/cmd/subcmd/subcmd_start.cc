/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdlib>
#include <string>

#include "cmd_base_utils.h"
#include "operation_data/operation_data_ops.h"
#include "storage/table_access.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "sim_common_api.h"
#include "sim_common_defs.h"
#include "sim_log.h"
#include "runtime_state/sim_models.h"
#include "subcmd_start.h"

namespace HcclSim {
void StartCommand::Setup(CLI::App& app)
{
    auto sub_start = app.add_subcommand("start", "start: 启动仿真环境,请勿在子bash中重复启用");

    sub_start->add_option("configCluster", configClusterName, "加载昇腾集群组网配置目录")
        ->required()
        ->check(CheckClusterConfigFile);
    sub_start->add_option("--level", g_hcclVmLevel, "设置模拟等级, 当前支持等级为 1 和 2, 默认模拟等级 2 ");
    sub_start->add_flag(
        "--check-only", checkOnlyMode,
        "仅校验模式:大块(200MB-4GB)内存申请复用同一块 4GB 共享区,内容不保证正确,换取内存节省");

    sub_start->callback([this]() {
        Execute();
    });
}

void StartCommand::Execute()
{
    if (g_hcclVmBashFlag) {
        HCCL_VM_WARN("hccl-vm has already started. Please do not start it again in a sub-bash.");
        return;
    }

    // 启动时序（方案 §2.1）：参数解析成功后于命令起点取得数据库会话并 fail-fast——
    // 初始化失败立即返回，不进入任何 RunnerDB 写入 / InitHvmEnv / workload；
    // 已创建对象由 main 命令结束后的显式排空关闭统一释放。
    if (sim::operation::EnsureProcessStorageSession() != 0) {
        HCCL_VM_ERROR("Failed to initialize simulation database. Please check the data directory.");
        return;
    }
    if (configClusterName.find(".yaml") != std::string::npos) {
        configClusterName.erase(configClusterName.find(".yaml"), 5);
    }

    auto retStr = GenerateClusterTopo(configClusterName);
    if (!retStr.empty()) {
        HCCL_VM_ERROR("{}", retStr);
        return;
    }

    HCCL_VM_INFO("Initializing: Model={}, Level={}", configClusterName, g_hcclVmLevel);
    g_configClusterDir = InstallPath::ResolveToInstallRoot("config/network/cluster/" + configClusterName);

    // 拷贝topo.json文件到执行目录
    auto ret = CopyFile(g_configClusterDir);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("copy topo.json file failed");
        return;
    }

    if (!sim::runtime::SetRunMode(checkOnlyMode)) {
        return;
    }
    HCCL_VM_INFO("run mode: {}", checkOnlyMode ? "check-only" : "normal");

    ret = InitHvmEnv(g_configClusterDir, g_hcclVmLevel, checkOnlyMode);
    if (ret != HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD) {
        HCCL_VM_ERROR("Failed to initialize simulation environment. Cleaning up environment.");
        auto cleanRet = HcclVmExit();
        if (cleanRet != HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD) {
            HCCL_VM_ERROR("Failed to clean up environment. Please check for residual environment artifacts.");
        }
        return;
    }

    ret = StartHvmCmd();
    if (ret != HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD) {
        HCCL_VM_ERROR("Failed to start hvm command.");
        auto cleanRet = HcclVmExit();
        if (cleanRet != HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD) {
            HCCL_VM_ERROR("Failed to clean up environment. Please check for residual environment artifacts.");
        }
        return;
    }

    // start 幂等复位（复用 mock-comm 的同一复位）：每次启动把库带回复读干净态——
    // 全部代际 Device 状态归零（含上一轮崩溃残留的 status=1，否则 aclrtGetDeviceCount
    // 计数为 0 导致 rank 启动即崩）、CcuResource/Communicator 与动态残留表清空、
    // 通信域缓存复位。mock-comm 随后会做同一复位并重建逻辑 ID，此处使其成为幂等
    // 前置——运行手册收敛为 start → mock-comm → mpirun，无需外部清库。
    // check-only 模式为只读校验，跳过复位避免误写库。
    if (!checkOnlyMode) {
        ret = HcclVmResetCommDomain();
        if (ret != HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD) {
            HCCL_VM_ERROR("Failed to reset communication domain residue after start.");
            auto cleanRet = HcclVmExit();
            if (cleanRet != HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD) {
                HCCL_VM_ERROR("Failed to clean up environment. Please check for residual environment artifacts.");
            }
            return;
        }
    }
    return;
}

static inline CommandAutoRegister<StartCommand> g_start_cmd_reg{};
} // namespace HcclSim
