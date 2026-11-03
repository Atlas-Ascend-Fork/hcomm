/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdio>
#include <cstdlib>
#include <string>

#include "topo_ascend_cluster_parser.h"
#include "cmd_base_utils.h"
#include "runtime_state/db_sim_runner_common.h"
#include "sim_common_defs.h"
#include "sim_log.h"
#include "subcmd_mock_comm.h"

namespace HcclSim {
static uint32_t ResolveTotalRankNum(const TopoMeta& topoMeta)
{
    uint32_t total = ShmGetPhyDeviceTotalCount(topoMeta);
    if (total != 0) {
        return total;
    }
    for (const auto& srv : sim::runtime::g_comm_config_data.servers) {
        total += srv.devNum;
    }
    return total;
}

void MockCommCommand::Setup(CLI::App& app)
{
    auto sub_run = app.add_subcommand("mock-comm", "mock-comm: 启动一个算子的通信域");

    sub_run->add_option("configFile", configFileName, "加载昇腾集群组网配置目录")->required()->check(FileInModelDir);
    sub_run->add_option("--level", g_hcclVmLevel, kSimLevelOptDesc);

    sub_run->callback([this]() {
        Execute();
    });
}

void MockCommCommand::Execute()
{
    // level1 裁剪构建仅支持 --level 1, 其余取值 fail-fast(先于任何环境副作用)
    if (!CheckSimLevelSupported()) {
        return;
    }
    HCCL_VM_INFO("Initializing mock communication domain: Config={}, Level={}", configFileName, g_hcclVmLevel);

    // 首先清理动态数据库表项
    auto ret = ClearDbTables();
    if (ret != HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD) {
        HCCL_VM_ERROR("Failed to clear database tables.");
        return;
    }

    // 拷贝topo.json文件到执行目录
    ret = CopyFile(g_configClusterDir);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("copy topo.json file failed");
        return;
    }

    // 重置通信域
    ret = HcclVmResetCommDomain();
    if (ret != HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD) {
        HCCL_VM_ERROR("Failed to reset mock communication domain.");
        return;
    }

    if (configFileName != "ranktable") {
        if (!ParseYamlTopo(configFileName, topoMeta)) {
            return;
        }
        // 每次mock-comm执行: 刷新TopoMetaConfig表项与进程通信域配置缓存
        // (ranktable模式在ParseRanktableAndInitCommDomain内刷新)
        if (!sim::runtime::RefreshTopoMetaConfig(configFileName, &topoMeta)) {
            HCCL_VM_ERROR("refresh topo meta config failed: {}", configFileName);
            return;
        }
    }

    ret = InitHvmCommEnv(topoMeta, configFileName, g_hcclVmLevel);
    if (ret != HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD) {
        HCCL_VM_ERROR("Failed to initialize mock communication environment. Cleaning up environment.");
        return;
    }
    // 进程监控临时方案：本轮要拉起的 rank 数 = yaml 第三层 device 数之和（即 device 数量），
    // 声明给轮询线程并清空槽位文件。算例进程在 deviceKey 确定后以它作为字节偏移登记、
    // 退出时往该字节写 1；Host 侧数够 N 才打 [HCCL_VM_ROUND_PROCESSES_EXITED]，
    // 少了不发标志让框架超时。
    RoundSlotArm(ResolveTotalRankNum(topoMeta));

    // 流程结束打屏标志(参考 EmitRoundFlag)：返回前直写 stdout，与日志后端解耦
    printf(
        "[HCCL_VM_MOCK_COMM_FINISHED] config=%s level=%u\n", configFileName.c_str(),
        static_cast<unsigned>(g_hcclVmLevel));
    fflush(stdout);
    return;
}

static inline CommandAutoRegister<MockCommCommand> g_mock_comm_cmd_reg{};
} // namespace HcclSim
