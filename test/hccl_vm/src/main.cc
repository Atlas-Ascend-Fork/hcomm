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
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#include "cmd_base.h"
#include "cmd_base_utils.h"
#include "sim_common_api.h"
#include "sim_log.h"
#include "sim_resource_limits.h"
#include "storage/storage_runtime.h"
#include "storage_composition/storage_composition.h"
#include "store/store_sim_resource_root.h"

void envInit()
{
    setenv("HCCL_VM_INSTALL_ROOT", GetBinLocation().c_str(), 1);

    // 自动把本进程 RLIMIT_NOFILE/NPROC 软限抬到硬限：大规模仿真（如 1024 rank
    // 建链）fd 需求远超默认软限 1024，触顶会让尾部 rank 建链失败（hcclRet 19）。
    // 进程级、无需特权、不落系统、退出即消失（免 limits.conf+重登录/手工 ulimit）；
    // host 在此抬升后，mpirun/orted、checker/plugin、AICPU 子进程随 fork/exec 继承，
    // rank 进程另在 aclInit 边界各自抬升（双保险）。
    HcclSim::SimRaiseResourceLimits();

    // 清理上次运行残留：data/、logs/ 及 sqlite db 均位于安装目录内，属当前用户可写，无需 sudo。
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::remove_all(InstallPath::ResolveToInstallRoot("data"), ec);
    ec.clear();
    fs::remove_all(InstallPath::ResolveToInstallRoot("logs"), ec);
    ec.clear();
    if (!fs::create_directories(InstallPath::ResolveToInstallRoot("data"), ec) || ec) {
        printf("envInit failed: recreate data dir\n");
    }
    ec.clear();
    if (!fs::create_directories(InstallPath::ResolveToInstallRoot("logs"), ec) || ec) {
        printf("envInit failed: recreate logs dir\n");
    }

    // 外部后端（MariaDB/Redis）没有本地 db 文件可删：经装配层逐逻辑库创建 transient
    // 引擎调用 Wipe()，把上一轮的持久化数据整体清回全新态——等价上面 SQLite 删
    // data/ 目录（含 db 文件）的干净起点。只有 host 的 envInit 会到这里，rank/
    // checker/消费者进程不经 envInit，绝不会误删；SQLite 部署在 Wipe 内直接跳过
    // （文件已删）。这样三后端"每次 start 全新库"语义一致，复用实例无需手工清库。
    auto wiped = sim::WipeSimulatorStorage();
    if (!wiped.ok()) {
        printf("envInit: wipe external storage failed: %s\n", wiped.diagnostic.c_str());
    }

    // 只强清并重建本进程 pid 目录，不再整机清扫 /dev/shm（其他 hccl-vm 实例隔离）。
    if (!sim::SimResourceRoot::GetInstance().Init()) {
        printf("SimResourceRoot init failed\n");
    }
    // 注册退出清理器：正常退出(atexit) + 异常崩溃/终止信号(signal handler) 都删除自身 pid 目录。
    sim::SimResourceRoot::RegisterExitCleaner();
    printf("envInit success\n");
}

int main(int argc, char* argv[])
{
    const char* envCheck = std::getenv(HVM_BASH_ENV_KEY.c_str());

    if (envCheck != nullptr) {
        StartHostClient(argc, argv);
    } else {
        envInit();
        LogConfig config = LoadLogConfig("hccl_vm");
        InitLogger(config);
        // 启动时序：数据库会话不在命令解析前取得——help/非法参数不建库；
        // start/run 参数解析成功后在其回调起点显式初始化并检查，失败即返回不启动 workload。
        std::string cmd = ArgvToString(argc, argv);
        if (argc == 1) {
            cmd += " --help";
        }
        ParseCommand(cmd);
        // 迁移前由静态单例析构关闭连接；显式排空并关闭进程存储会话保证引擎关闭顺序。
        auto closed = HcclSim::Storage::StorageRuntime::CloseProcessSession();
        if (!closed.ok()) {
            HCCL_VM_ERROR("close storage session failed: {}", closed.diagnostic);
        }
    }
    return 0;
}
