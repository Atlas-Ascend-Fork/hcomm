/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// host/src/hccl_plugin_manager.cpp
#include "sim_plugin_manager.h"

#include <cstdint>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <libgen.h>
#include <limits>
#include <mutex>
#include <stack>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

#include "cmd_base_utils.h"
#include "sim_common_api.h"
#include "store_dump_shm_data.h"
#include "sim_log.h"
#include "runtime_state/sim_models.h"
#include "runtime_state/db_sim_runner_ops.h"

using namespace HcclSim;

// 插件管理器实现（单例，作为Host的内部组件）
HcclPluginManager& HcclPluginManager::GetInstance()
{
    static HcclPluginManager instance;
    return instance;
}

bool HcclPluginManager::IsMatchingPlugin(const std::string& manifestPath, const std::string& targetTag)
{
    // 1. 预检查：文件是否可读
    if (access(manifestPath.c_str(), R_OK) != 0) {
        return false;
    }

    try {
        std::ifstream f(manifestPath);
        if (!f.is_open()) {
            return false;
        }

        // 2. 解析 JSON
        nlohmann::json data = nlohmann::json::parse(f);

        // 3. 字段校验与匹配
        // 使用我们定义的静态常量 Manifest::pluginName
        const std::string& nameKey = HcclPlugin::Manifest::pluginName;

        if (data.contains(nameKey) && data[nameKey].is_string()) {
            std::string currentPluginName = data[nameKey].get<std::string>();

            // 进行 Tag 匹配
            if (currentPluginName == targetTag) {
                // 如果必需字段缺失，认为是一个损坏的插件，返回 false 并记录日志
                if (!data.contains(HcclPlugin::Manifest::pluginEntry)) {
                    HCCL_VM_ERROR("'{}' missing entry field.", targetTag);
                    return false;
                }
                return true;
            }
        }
    } catch (const nlohmann::json::parse_error& e) {
        // 生产环境建议记录具体的解析错误位置
        HCCL_VM_ERROR("JSON Parse Error at byte {}", e.byte);
        return false;
    } catch (const std::exception& e) {
        // 处理其他可能的异常（如文件读取中途断开等）
        HCCL_VM_ERROR("Error: {}", e.what());
        return false;
    }

    return false;
}

bool HcclPluginManager::GetPluginFolderPath(const std::string& pluginTag, std::string& pluginFolderPath)
{
    std::string rootDir = InstallPath::ResolveToInstallRoot(HcclPlugin::PLUGIN_PATH.substr(1));
    std::string foundPath = "";
    bool hasConflict = false;

    struct ScanTask {
        std::string path;
        int depth;
    };

    std::stack<ScanTask> taskStack;
    taskStack.push({rootDir, 0});

    while (!taskStack.empty()) {
        ScanTask current = taskStack.top();
        taskStack.pop();

        // 1. 检查当前目录是否包含匹配的 manifest
        std::string manifestPath = current.path + HcclPlugin::MANIFEST_FILE;
        if (IsMatchingPlugin(manifestPath, pluginTag)) {
            char absPath[PATH_MAX];
            if (realpath(current.path.c_str(), absPath)) {
                std::string currentAbsPath(absPath);

                // 冲突检查：如果之前已经找到了一个路径，且不是当前路径
                if (!foundPath.empty() && foundPath != currentAbsPath) {
                    HCCL_VM_ERROR(
                        "Conflict detected: Tag '{}' exists in both: {} and {}", pluginTag, foundPath, currentAbsPath);
                    hasConflict = true;
                    break;
                }
                foundPath = currentAbsPath;
            }
            // 按照约定，插件文件夹内部不再嵌套插件，找到后不进入子目录
            continue;
        }

        // 2. 限制深度进行 DFS 遍历
        if (current.depth < HcclPlugin::MAX_SCAN_DEPTH) {
            DIR* dir = opendir(current.path.c_str());
            if (!dir) {
                continue;
            }

            struct dirent* entry;
            while ((entry = readdir(dir)) != nullptr) {
                std::string name = entry->d_name;
                if (name == "." || name == "..") {
                    continue;
                }

                std::string nextPath = current.path + "/" + name;
                struct stat st;
                // 使用 lstat 避免符号链接环路
                if (lstat(nextPath.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
                    taskStack.push({nextPath, current.depth + 1});
                }
            }
            closedir(dir);
        }
    }

    // 3. 最终结果逻辑
    if (hasConflict) {
        pluginFolderPath = "";
        return false;
    }

    if (!foundPath.empty()) {
        pluginFolderPath = foundPath;
        return true;
    }

    return false;
}

std::vector<std::string> HcclPluginManager::GetPluginStatus() const
{
    std::vector<std::string> result;

    // 1. 定义表头 (使用左对齐和固定宽度确保整齐)
    std::stringstream header;
    header << std::left << std::setw(20) << "PLUGIN_TAG" << std::setw(10) << "PID" << std::setw(15) << "STATUS";
    result.push_back(header.str());

    // 2. 遍历插件获取数据
    std::lock_guard<std::mutex> lock(m_mutex);
    for (const auto& pair : m_plugins) {
        const std::string& tag = pair.first;
        const auto& plugin = pair.second;

        // 获取 PID 和 运行状态
        int32_t pid = plugin->GetPid(); // 确保 HcclPlugin 有 GetPid() 方法
        bool running = plugin->IsRunning();
        std::string statusStr = running ? "RUNNING" : "STOPPED/EXITED";

        std::stringstream row;
        row << std::left << std::setw(20) << tag << std::setw(10) << (pid > 0 ? std::to_string(pid) : "N/A")
            << std::setw(15) << statusStr;

        result.push_back(row.str());
    }

    return result;
}

HcclVmResult HcclPluginManager::RegisterPlugin(const std::string& pluginTag)
{
    std::lock_guard<std::mutex> lock(m_mutex);

    // 防止重复注册
    if (m_plugins.find(pluginTag) != m_plugins.end()) {
        HCCL_VM_WARN("Plugin {} already registered", pluginTag);
        return HcclSim::HcclVmResult::HCCL_SIM_SUCCESS;
    }

    std::string folderPath;
    if (GetPluginFolderPath(pluginTag, folderPath) == false) {
        return HcclSim::HcclVmResult::HCCL_SIM_E_NOT_FOUND;
    }

    try {
        // 创建即启动 (符合你 HcclPlugin 构造函数的逻辑)
        auto plugin = std::make_shared<HcclPlugin>(folderPath);
        m_plugins[pluginTag] = plugin;
        AddInstallRecordToDB(pluginTag);

    } catch (const std::exception& e) {
        HCCL_VM_ERROR("Failed to register plugin {}: {}", pluginTag, e.what());
        return HcclSim::HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    return HcclSim::HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult HcclPluginManager::SendMessageToPlugin(
    const std::string& pluginTag, const std::string& action, const nlohmann::json& payload)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_plugins.find(pluginTag);
    if (it == m_plugins.end()) {
        HCCL_VM_ERROR("Plugin {} not found", pluginTag);
        return HcclVmResult::HCCL_SIM_E_NOT_FOUND;
    }
    HcclVmResult ret = it->second->SendMessage(PLUGIN_MESSAGE_TYPE::BROADCAST, action, payload);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("Failed to send message to plugin {}: {}", pluginTag, static_cast<int>(ret));
    }
    return ret;
}

HcclVmResult HcclPluginManager::BroadcastToAllPlugin(const std::string& action, const nlohmann::json& payload)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    HcclVmResult ret = HcclVmResult::HCCL_SIM_SUCCESS;
    for (auto& pair : m_plugins) {
        if (pair.second->IsRunning()) {
            HcclVmResult singleRet = pair.second->SendMessage(PLUGIN_MESSAGE_TYPE::BROADCAST, action, payload);
            if (singleRet != HcclVmResult::HCCL_SIM_SUCCESS) {
                ret = HcclVmResult::HCCL_SIM_E_INTERNAL;
            }
        }
    }
    return ret;
}

std::vector<HcclSim::HcclVmResult> HcclPluginManager::StartPlugins(const std::vector<std::string>& tags)
{
    std::vector<HcclSim::HcclVmResult> results;
    std::lock_guard<std::mutex> lock(m_mutex);

    for (const auto& tag : tags) {
        auto it = m_plugins.find(tag);
        if (it == m_plugins.end()) {
            results.push_back(HcclSim::HcclVmResult::HCCL_SIM_E_NOT_FOUND);
            continue;
        }

        // 如果插件已经在运行，通常视为成功或检查状态
        if (it->second->IsRunning()) {
            results.push_back(HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
        } else {
            // 调用 HcclPlugin 的 Start 方法
            results.push_back(it->second->Start());
        }
    }
    return results;
}

HcclVmResult ExitRunnerPlugin()
{
    // 走与 `hccl-vm plugin uninstall @runner` 完全相同的卸载通道：
    // StopPlugins -> HcclPlugin::Stop() 通过 stdin 管道下发 {"action":"stop"}，
    // runner 主循环据此置 g_keep_running=false 退出，随后 Stop() 回收子进程、
    // StopPlugins 从 m_plugins 摘除并清理安装记录。
    // 旧实现只写 proxy_ready.flag 的 kRunnerExitSignal 哨兵，而 runner 侧从不
    // 读取该文件（checkProxyReadySignal 无任何调用方），信号发出即石沉大海。
    auto rets = HcclPluginManager::GetInstance().StopPlugins({"runner"});
    HcclVmResult ret = rets.empty() ? HcclVmResult::HCCL_SIM_SUCCESS : rets.front();
    if (ret == HcclVmResult::HCCL_SIM_E_NOT_FOUND) {
        HCCL_VM_ERROR("Runner process is alive but not managed by this host process, "
                      "cannot unload it. Please terminate the runner process manually.");
        return ret;
    }
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("Failed to unload runner plugin, result={}", static_cast<int32_t>(ret));
        return ret;
    }

    // Stop() 等待 5 秒超时后同样返回 SUCCESS（只告警不强杀），故不能以返回值判定
    // 退出成功；runner.lock 上的 flock 由内核在进程真正退出时释放，是唯一可靠判据。
    if (IsRunnerAlive()) {
        HCCL_VM_ERROR("Runner did not exit within the grace period and is still holding the lock. "
                      "[ACTION REQUIRED] Please terminate the runner process manually.");
        return HcclVmResult::HCCL_SIM_E_TIMEOUT;
    }

    HCCL_VM_INFO("Runner plugin unloaded and runner process exited.");
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

std::vector<HcclSim::HcclVmResult> HcclPluginManager::StopPlugins(const std::vector<std::string>& tags)
{
    std::vector<HcclSim::HcclVmResult> results;
    std::lock_guard<std::mutex> lock(m_mutex);

    for (const auto& tag : tags) {
        // 注意：此处不得为 "runner" 开特殊分支回调 ExitRunnerPlugin()。
        // ExitRunnerPlugin() 会反过来调用 StopPlugins()，而本函数已持有非递归的
        // m_mutex，形成自死锁。runner 与其它插件一样走下面的通用卸载路径即可
        // （Stop() 经 stdin 下发 stop 命令，runner 主循环会响应）。
        auto it = m_plugins.find(tag);
        if (it == m_plugins.end()) {
            results.push_back(HcclSim::HcclVmResult::HCCL_SIM_E_NOT_FOUND);
            continue;
        }

        // 1. 调用插件内部的优雅停止逻辑（含5秒超时）
        HcclSim::HcclVmResult res = it->second->Stop();
        results.push_back(res);

        // 2. 从管理 Map 中移除（卸载）
        // 即使 Stop 失败（如超时），我们通常也从 Manager 中移除它，将其交由系统清理或用户手动处理
        m_plugins.erase(it);
        RemoveInstallRecordFromDB(tag);
    }
    return results;
}

HcclSim::HcclVmResult HcclPluginManager::StopAllPlugins()
{
    // 1. 获取当前所有插件的 Tag
    std::vector<std::string> allTags;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        for (const auto& pair : m_plugins) {
            allTags.push_back(pair.first);
        }
    }

    // 2. 调用批量停止方法
    if (allTags.empty()) {
        return HcclSim::HcclVmResult::HCCL_SIM_SUCCESS;
    }

    StopPlugins(allTags);

    return HcclSim::HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclPluginManager::HcclPluginManager()
{
    HCCL_VM_INFO("HcclPluginManager");
    m_monitorThreadStop = false;
    m_monitorThread = std::thread([this]() {
        this->MonitorThread();
    });
}

HcclPluginManager::~HcclPluginManager()
{
    m_monitorThreadStop = true;
    StopAllPlugins();
    if (m_monitorThread.joinable()) {
        m_monitorThread.join();
    }
}

void HcclPluginManager::MonitorThread()
{
    int status;
    while (!m_monitorThreadStop) {
        // -1 表示等待任意子进程；WNOHANG 避免析构时线程卡在 waitpid 里无法退出
        pid_t terminatedPid = waitpid(-1, &status, WNOHANG);
        if (terminatedPid > 0) {
            std::lock_guard<std::mutex> lock(m_mutex);
            for (auto it : m_plugins) {
                // 用户通过命令行关闭的plugin已经不在map里 不会有输出
                if (it.second->GetPid() == terminatedPid) {
                    // 解析退出原因
                    if (WIFEXITED(status)) {
                        HCCL_VM_INFO(
                            "Plugin [{}] (PID: {}) exit normally. Code {}", it.second->GetTag(), terminatedPid,
                            WEXITSTATUS(status));
                    } else if (WIFSIGNALED(status)) {
                        HCCL_VM_ERROR(
                            "Plugin [{}] (PID: {}) exit with failure. Code {} ({})", it.second->GetTag(), terminatedPid,
                            WTERMSIG(status), strsignal(WTERMSIG(status)));
                    }
                    m_plugins.erase(it.first);
                    break;
                }
            }
        } else if (terminatedPid == 0) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        } else if (terminatedPid == -1) {
            if (errno == ECHILD) {
                // 没有子进程了 进入轮询
                std::this_thread::sleep_for(std::chrono::seconds(1));
                continue;
            }
        }
    }
}

void HcclPluginManager::AddInstallRecordToDB(const std::string& pluginTag)
{
    sim::runtime::Plugin pluginRow{};
    std::strcpy(pluginRow.tag, pluginTag.c_str());
    sim::runtime::Db::Add<sim::runtime::Plugin>(pluginRow);
}

void HcclPluginManager::RemoveInstallRecordFromDB(const std::string& pluginTag)
{
    // 正常情况一个tag只会有一行db记录，使用vector接口删除此tag相关的全部行，可以应对意外场景
    auto pluginRows = sim::runtime::Db::GetByPred<sim::runtime::Plugin>(
        HcclSim::Storage::Eq(&sim::runtime::Plugin::tag, pluginTag));
    if (!pluginRows.ok() || !pluginRows.value.has_value()) {
        return;
    }
    for (const sim::runtime::Plugin& row : *pluginRows.value) {
        sim::runtime::Db::Delete<sim::runtime::Plugin>(HcclSim::Storage::Eq(&sim::runtime::Plugin::id, row.id));
    }
}
