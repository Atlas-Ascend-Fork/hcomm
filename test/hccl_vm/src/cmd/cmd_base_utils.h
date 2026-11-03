/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_VM_CMD_UTILS_H
#define HCCL_VM_CMD_UTILS_H

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <libgen.h>
#include <sstream>
#include <cstdlib>
#include <stdlib.h>
#include <string>
#include <unistd.h>
#include <vector>

#include "cmd_cluster_model_utils.h"
#include "sim_common_defs.h"
#include "sim_plugin_manager.h"

extern const std::string HVM_BASH_ENV_KEY;
extern bool g_hcclVmBashFlag;
extern std::string g_configClusterDir;
extern std::uint32_t g_hcclVmLevel;
extern std::uint8_t g_hcclVmRunMode;

// level1 裁剪构建(HCCLVM_LEVEL1_ONLY)仅支持 --level 1; 其余构建支持 1/2 且行为不变。
// --level 选项帮助串(start/run/mock-comm 三处注册点共享, 随构建模式条件化)
#ifdef HCCLVM_LEVEL1_ONLY
constexpr const char* kSimLevelOptDesc = "设置模拟等级, 本构建为 level1 裁剪版, 仅支持等级 1, 默认 1";
#else
constexpr const char* kSimLevelOptDesc = "设置模拟等级, 当前支持等级为 1 和 2, 默认模拟等级 2 ";
#endif
// 校验当前构建是否支持 g_hcclVmLevel 取值; 不支持时打印错误并返回 false(非 level1 构建恒 true)
bool CheckSimLevelSupported();

class CstyleCmd {
public:
    explicit CstyleCmd(const std::vector<std::string>& args)
    {
        storage_ = args;
        rebuild();
    }
    int argc() const { return static_cast<int>(pointers_.size()); }
    char** argv() { return pointers_.data(); }
    std::string cmd() { return cmd_; }

private:
    void rebuild()
    {
        std::stringstream ss;
        pointers_.clear();
        pointers_.reserve(storage_.size() + 1);
        for (auto& s : storage_) {
            pointers_.push_back(const_cast<char*>(s.c_str()));
            ss << s << " ";
        }
        cmd_ = ss.str();
    }
    std::vector<std::string> storage_;
    std::vector<char*> pointers_;
    std::string cmd_;
};

// 通用工具方法
std::string GetBinLocation();
std::string ArgvToString(int argc, char* argv[]);
std::string FileInModelDir(const std::string& fileName);
std::string CheckClusterConfigFile(const std::string& topoFileName);
std::string GenerateClusterTopo(const std::string& topoFileName);
void RemoveFromLDPreload(const std::string& targetValue);
void ShowModel();

// init hvm env
HcclVmResult InitHvmEnv(const std::string& configClusterDir, uint32_t level, bool checkOnlyMode);
HcclVmResult InitHvmCommEnv(const TopoMeta& topoMeta, const std::string& configFileName, uint32_t level);
HcclVmResult HcclVmExit();
HcclVmResult CopyFile(const std::string& clusterDir);

// check-only 模式复用区 HcclCommPool 的建/拆，随运行模式切换调用
HcclVmResult CreateCommPoolShm(uint64_t poolSize);
HcclVmResult DestroyCommPoolShm();

// plugin manager
HcclVmResult InstallUserPlugin(std::string argStr);
HcclVmResult RunUserPlugin(std::string argStr);
HcclVmResult UninstallUserPlugin(std::string argStr);
void ShowUserPlugin();

// config
HcclVmResult SetConsoleLogLevel(int level);
HcclVmResult SetFileLogLevel(int level);
HcclVmResult ShowCurrentLogLevel();

// 进程监控：轮次槽位文件 <HCCL_VM_RESOURCE_ROOT>/round_slots，第 (deviceKey) 字节非 0
// 即该 Device 上的算例进程已退出；槽位身份 = 全局唯一的 deviceKey（Device 表主键）。
// mock-comm 调用：slots = 本轮要拉起的 rank 数（yaml 第三层 device 数之和），清空槽位文件
void RoundSlotArm(uint32_t slots);
// Host 侧轮询线程：非 0 字节数够 slots → 先清空、再打 [HCCL_VM_ROUND_PROCESSES_EXITED]；
// 数不够只打 [HCCL_VM_ROUND_PROGRESS]、不发标志，由测试框架超时
void RoundSlotsListen();
void StopRoundSlotsListen();

// hccl-vm 流程类方法
HcclVmResult StartHvmCmd();

void StartHostClient(int argc, char* argv[]);
void ServerListen();
void RunnerListen();
void ParseCommand(std::string& cmd);

HcclVmResult ClearDbTables();
HcclVmResult HcclVmResetCommDomain();

// 检测 runner 进程是否存活（通过 lock file）
bool IsRunnerAlive();

// 卸载 runner 插件
HcclVmResult ExitRunnerPlugin();

#endif
