/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cmd_base_utils.h"
#include <algorithm>
#include <cerrno>
#include <atomic>
#include <chrono>
#include <climits>
#include <CLI11.hpp>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dirent.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mqueue.h>
#include <nlohmann_json/json.hpp>
#include <sstream>
#include <stdexcept>
#include <signal.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "topo_ascend_cluster_parser.h"
#include "cmd_base.h"
#include "store_dump_shm_data.h"
#include "store_sim_comm_pool_policy.h"
#include "store_sim_run_mode.h"
#include "sim_data_dump.h"
#include "sim_log.h"
#include "store_sim_memory_manager.h"
#include "store_sim_device_memory_manager.h"
#include "store/store_sim_resource_root.h"
#include "store_sim_comm_memory_manager.h"
#include "runtime_state/db_sim_runner_common.h"
#include "operation_data/operation_data_ops.h"
#include "sim_common_api.h"
#include "storage/table_access.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "yaml-cpp/yaml.h"

using namespace HcclSim;
namespace fs = std::filesystem;

static std::atomic<bool> g_serverListenFlag{false};
static std::atomic<bool> g_runnerListenFlag{false};
static std::thread* g_serverThread = nullptr;
static std::thread* g_runnerThread = nullptr;

static std::string MakeDataBackupTimestamp();
static void BackupAivTaskFiles(const fs::path& backupDir);

const std::string HVM_BASH_ENV_KEY = "_HVM_BASH_ENV_PATH";
const std::string g_binDir = GetBinLocation();
static void StopListenThreads();
static void ArchiveLogsAndData();

// 定义队列名称和大小配置
const char* MQ_REQ_NAME = "host_mq_request";
const char* MQ_RESP_NAME = "host_mq_response";
const int MAX_MSG_SIZE = 4096; // 单条命令最大长度
const int MAX_MSG_COUNT = 5;   // 队列深度

static const int HOST_CLIENT_RESP_TIMEOUT_MS = 5000;

bool g_hcclVmBashFlag{false};
std::uint32_t g_hcclVmLevel{2};
std::string g_configClusterDir{""};

static std::string ToPosixMqName(const char* name)
{
    if (name == nullptr || name[0] == '\0') {
        throw std::invalid_argument("message queue name is empty");
    }
    std::string base;
    if (name[0] == '/') {
        base = name;
    } else {
        base = "/" + std::string(name);
    }
    // POSIX 消息队列名不能包含中间斜杠（无目录层级）, 无法落到 /dev/shm/hvm_<pid>/ 下;
    // 故把根目录名后缀（hvm_<pid>）嵌进名字，实现按 hccl-vm 进程粒度隔离。
    return base + "_" + sim::SimResourceRoot::GetTag();
}

static std::string MakeMqErrorMessage(const std::string& operation, const std::string& name, int err)
{
    return operation + " message queue " + name + " failed: " + std::strerror(err);
}

static mq_attr MakeMessageQueueAttr()
{
    mq_attr attr{};
    attr.mq_maxmsg = MAX_MSG_COUNT;
    attr.mq_msgsize = MAX_MSG_SIZE;
    return attr;
}

static timespec MakeDeadlineAfterMs(int timeoutMs)
{
    timespec deadline{};
    if (clock_gettime(CLOCK_REALTIME, &deadline) == -1) {
        throw std::runtime_error(MakeMqErrorMessage("get deadline for", "host response", errno));
    }

    deadline.tv_sec += timeoutMs / 1000;
    deadline.tv_nsec += static_cast<long>(timeoutMs % 1000) * 1000L * 1000L;
    if (deadline.tv_nsec >= 1000L * 1000L * 1000L) {
        deadline.tv_sec += 1;
        deadline.tv_nsec -= 1000L * 1000L * 1000L;
    }
    return deadline;
}

class PosixMessageQueue {
public:
    PosixMessageQueue(const char* name, int flags, const mq_attr* attr = nullptr) : name_(ToPosixMqName(name))
    {
        if ((flags & O_CREAT) != 0) {
            mq_ = mq_open(name_.c_str(), flags, S_IRUSR | S_IWUSR, const_cast<mq_attr*>(attr));
        } else {
            mq_ = mq_open(name_.c_str(), flags);
        }
        if (mq_ == static_cast<mqd_t>(-1)) {
            throw std::runtime_error(MakeMqErrorMessage("open", name_, errno));
        }
    }

    ~PosixMessageQueue()
    {
        if (mq_ != static_cast<mqd_t>(-1)) {
            mq_close(mq_);
        }
    }

    PosixMessageQueue(const PosixMessageQueue&) = delete;
    PosixMessageQueue& operator=(const PosixMessageQueue&) = delete;

    void Send(const void* data, size_t size, unsigned int priority) const
    {
        while (mq_send(mq_, static_cast<const char*>(data), size, priority) == -1) {
            if (errno == EINTR) {
                continue;
            }
            throw std::runtime_error(MakeMqErrorMessage("send to", name_, errno));
        }
    }

    ssize_t Receive(void* buffer, size_t size, unsigned int* priority) const
    {
        while (true) {
            ssize_t recvdSize = mq_receive(mq_, static_cast<char*>(buffer), size, priority);
            if (recvdSize >= 0) {
                return recvdSize;
            }
            if (errno != EINTR) {
                throw std::runtime_error(MakeMqErrorMessage("receive from", name_, errno));
            }
        }
    }

    bool
    TimedReceive(void* buffer, size_t size, unsigned int* priority, const timespec& deadline, ssize_t& recvdSize) const
    {
        while (true) {
            recvdSize = mq_timedreceive(mq_, static_cast<char*>(buffer), size, priority, &deadline);
            if (recvdSize >= 0) {
                return true;
            }
            if (errno == EINTR) {
                continue;
            }
            if (errno == ETIMEDOUT) {
                return false;
            }
            throw std::runtime_error(MakeMqErrorMessage("receive from", name_, errno));
        }
    }

private:
    std::string name_;
    mqd_t mq_{static_cast<mqd_t>(-1)};
};

static bool StartsWith(const std::string& value, const std::string& prefix) { return value.rfind(prefix, 0) == 0; }

// 统一清理本实例的消息队列：
// 1) 强制 unlink 本进程 host_mq_request / host_mq_response 两条队列(ToPosixMqName 拼出 hvm_<pid> 后缀);
// 2) 尽力回收 /dev/mqueue 里「宿主进程已死」的历史残留队列，避免反复崩溃重启累积到系统上限.
//    第 2 步按名字后缀 hvm_<pid> 解析出宿主 pid，仅当 pid 确认死亡(ESRCH)才 unlink，绝不误删并发存活实例.
static void CleanupMessageQueues()
{
    const std::string reqName = ToPosixMqName(MQ_REQ_NAME);
    const std::string respName = ToPosixMqName(MQ_RESP_NAME);
    if (mq_unlink(reqName.c_str()) == -1 && errno != ENOENT) {
        HCCL_VM_WARN("{}", MakeMqErrorMessage("remove", reqName, errno));
    }
    if (mq_unlink(respName.c_str()) == -1 && errno != ENOENT) {
        HCCL_VM_WARN("{}", MakeMqErrorMessage("remove", respName, errno));
    }

    DIR* dir = opendir("/dev/mqueue");
    if (dir == nullptr) {
        // /dev/mqueue 未挂载时无法枚举残留；本进程两条队列已在上方清理
        return;
    }
    for (struct dirent* entry = readdir(dir); entry != nullptr; entry = readdir(dir)) {
        const std::string name = entry->d_name;
        if (!StartsWith(name, "host_mq_request_hvm_") && !StartsWith(name, "host_mq_response_hvm_")) {
            continue;
        }
        const std::string marker = "hvm_";
        size_t pos = name.rfind(marker);
        if (pos == std::string::npos) {
            continue;
        }
        const std::string pidStr = name.substr(pos + marker.size());
        if (pidStr.empty() || pidStr.find_first_not_of("0123456789") != std::string::npos) {
            continue;
        }
        char* endp = nullptr;
        long pidVal = std::strtol(pidStr.c_str(), &endp, 10);
        if (endp == pidStr.c_str() || *endp != '\0' || pidVal <= 0 || pidVal > INT_MAX) {
            continue;
        }
        if (static_cast<pid_t>(pidVal) == getpid()) {
            continue; // 本进程队列已在上方处理
        }
        if (kill(static_cast<pid_t>(pidVal), 0) == 0 || errno != ESRCH) {
            continue; // 仍存活或状态未知，跳过并发实例
        }
        HCCL_VM_INFO("remove stale message queue /{}", name);
        if (mq_unlink(("/" + name).c_str()) == -1 && errno != ENOENT) {
            HCCL_VM_WARN("remove stale message queue /{} failed: {}", name, std::strerror(errno));
        }
    }
    closedir(dir);
}

std::string GetBinLocation()
{
    std::error_code ec;
    fs::path exePath = fs::read_symlink("/proc/self/exe", ec);
    if (ec) {
        throw std::runtime_error("read_symlink failed: " + ec.message());
    }
    fs::path binDir = exePath.parent_path();
    if (binDir.filename() == "bin") {
        return binDir.parent_path().string();
    }
    return binDir.string();
}

std::string ArgvToString(int argc, char* argv[])
{
    std::string cmd;
    for (int i = 0; i < argc; ++i) {
        std::string s = argv[i];
        // 如果包含空格，两头加引号
        if (s.find(' ') != std::string::npos) {
            cmd += "\"" + s + "\"";
        } else {
            cmd += s;
        }
        if (i < argc - 1) {
            cmd += " ";
        }
    }
    return cmd;
}

void RemoveFromLDPreload(const std::string& targetValue)
{
    HCCL_VM_DEBUG("clean LD_PRELOAD env: {}", targetValue);
    const char* curVal = std::getenv("LD_PRELOAD");

    if (curVal == nullptr) {
        return;
    }
    std::string envStr(curVal);

    // 如果当前值就是目标值，直接 unset
    if (envStr == targetValue) {
        unsetenv("LD_PRELOAD");
        return;
    }

    std::stringstream ss(envStr);
    std::string item;
    std::string envStrNew;
    bool first = true;

    while (std::getline(ss, item, ':')) {
        // 过滤空项（双冒号情况）和目标项
        if (item.empty() || item == targetValue) {
            continue;
        }
        if (!first) {
            envStrNew += ":";
        }
        envStrNew += item;
        first = false;
    }
    if (envStrNew.empty()) {
        // 如果结果为空，说明只包含要删除的项，直接 unset
        unsetenv("LD_PRELOAD");
    } else {
        // 覆盖原变量 (overwrite = 1)
        setenv("LD_PRELOAD", envStrNew.c_str(), 1);
    }
}

std::string FileInModelDir(const std::string& fileName)
{
    if (fileName == "ranktable") {
        std::string clusterInfo = InstallPath::ResolveToInstallRoot(fileName + ".json");
        std::ifstream f(clusterInfo.c_str());
        if (!f.good()) {
            return "config ranktable, but rank table file not found: " + clusterInfo;
        }
        return "";
    }
    std::string filePath = InstallPath::ResolveToInstallRoot("config/topo_meta/" + fileName + ".yaml");
    auto fileExistd = [&]() -> bool {
        std::ifstream f(filePath.c_str());
        return f.good();
    };
    if (fileExistd()) {
        return "";
    } else {
        return "[HVM] model File not found: " + filePath;
    }
}

std::string CheckClusterConfigFile(const std::string& topoFileName)
{
    if (topoFileName.empty()) {
        return "[HVM] topoFileName is empty";
    }
    // 检查topoFileName是否带.yaml后缀, 有则删除
    auto topoName = topoFileName;
    if (topoName.find(".yaml") != std::string::npos) {
        topoName.erase(topoName.find(".yaml"), 5);
    }

    std::string generateShellPath = InstallPath::ResolveToInstallRoot("script/generate_cluster_topo.sh");
    if (!fs::exists(generateShellPath)) {
        HCCL_VM_ERROR("generate_cluster_topo.sh not found: {}", generateShellPath);
        return "[HVM] generate_cluster_topo.sh not found: " + generateShellPath;
    }
    std::string clusterConfigFilePath = InstallPath::ResolveToInstallRoot("config/cluster/" + topoName + ".yaml");
    if (!fs::exists(clusterConfigFilePath)) {
        HCCL_VM_ERROR("cluster config file not found: {}", clusterConfigFilePath);
        return "[HVM] cluster config file not found: " + clusterConfigFilePath;
    }
    return "";
}

std::string GenerateClusterTopo(const std::string& topoFileName)
{
    std::string generateShellPath = InstallPath::ResolveToInstallRoot("script/generate_cluster_topo.sh");
    std::string clusterConfigFilePath = InstallPath::ResolveToInstallRoot("config/cluster/" + topoFileName + ".yaml");
    // 执行generate_cluster_topo.sh脚本（默认路径），生成集群组网
    std::string cmd = "bash " + generateShellPath + " " + clusterConfigFilePath;
    if (std::system(cmd.c_str()) != 0) {
        HCCL_VM_ERROR("generate_cluster_topo.sh failed: {}", cmd);
        return "[HVM] generate_cluster_topo.sh failed: " + cmd;
    }
    return "";
}

void ShowModel()
{
    std::string modelPath = InstallPath::ResolveToInstallRoot("cluster_model");
    if (!fs::exists(modelPath)) {
        HCCL_VM_ERROR("path not exist -> {}", modelPath);
        return;
    }
    if (!fs::is_directory(modelPath)) {
        HCCL_VM_ERROR("path not a dict -> {}", modelPath);
        return;
    }
    bool hasFiles = false;
    HCCL_VM_INFO("model : ");
    for (const auto& entry : fs::directory_iterator(modelPath)) {
        // 过滤：只关心"常规文件"，忽略子文件夹
        if (entry.is_regular_file()) {
            hasFiles = true;
            fs::path filePath = entry.path();
            HCCL_VM_INFO("  {}  [description] : ", filePath.stem().string());
            YAML::Node root = YAML::LoadFile(filePath);
            uint32_t podNum = root["meta"]["podNum"].as<uint32_t>();
            uint32_t serNum = root["meta"]["serNum"].as<uint32_t>();
            uint32_t rankNum = root["meta"]["rankNum"].as<uint32_t>();
            HCCL_VM_INFO("podNum: {}, serNum: {}, rankNum: {}", podNum, serNum, rankNum);
        }
    }
    if (!hasFiles) {
        HCCL_VM_WARN("there is no model");
    }
    return;
}

HcclVmResult InitHvmEnv(const std::string& configClusterDir, uint32_t level, bool checkOnlyMode)
{
    HCCL_VM_INFO("Enter InitHvmEnv: {}", configClusterDir);
    // 启动仿真环境
    // 创建用于Host-Device通信的共享内存
    // 仅校验模式才创建大块复用区 HcclCommPool，须在其它共享内存创建前，仅一次。
    if (checkOnlyMode) {
        void* commPool = sim::MemoryManager::GetInstance().AllocMemByName(
            sim::CommPoolPolicy::kPoolName, sim::CommPoolPolicy::kPoolSize);
        if (commPool == nullptr) {
            HCCL_VM_ERROR("create HcclCommPool fail");
            return HcclVmResult::HCCL_SIM_HOST_ERROR_CMD;
        }
    }

    if (!sim::CommunicationMemoryManager::GetInstance().InitPool()) {
        HCCL_VM_ERROR("InitPool failed");
        return HcclVmResult::HCCL_SIM_HOST_ERROR_CMD;
    }

    // 解析集群拓扑，并初始化静态数据模型数据
    HCCL_VM_INFO("Initializing: Cluster Topo");
    auto topoRet = AscendClusterTopoParser::GetInstance().InitClusterTopo(configClusterDir);
    if (topoRet != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("Failed to initialize cluster topology from directory: {}", configClusterDir);
        return HcclVmResult::HCCL_SIM_HOST_ERROR_CMD;
    }

    std::string checkerTag = "@checker";
    auto chkInstallRet = InstallUserPlugin(checkerTag);
    if (chkInstallRet != HCCL_SIM_HOST_SUCCESS_CMD) {
        HCCL_VM_ERROR("default plugin install fail, please check your plugin path");
    }

    HCCL_VM_INFO("====================================");
    ShowUserPlugin();

    HCCL_VM_INFO("======================================");

    return HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD;
}

HcclVmResult InitHvmCommEnv(const TopoMeta& topoMeta, const std::string& configFileName, uint32_t level)
{
    // 通信域已初始化，则无需重复初始化
    if (AscendClusterTopoParser::GetInstance().GetClusterStatus() == HvmClusterStatus::COMM_DOMAIN_INIT_DONE) {
        HCCL_VM_ERROR("communication domain already initialized");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    HcclVmResult ret;
    if (configFileName != "ranktable") {
        HCCL_VM_INFO("mock-comm cmd config topo yaml file, config by {}.yaml", configFileName);
        // 初始化本次算子通信域相关的表项
        ret = AscendClusterTopoParser::GetInstance().InitCommunicationDomain(topoMeta, false);
    } else {
        HCCL_VM_INFO("mock-comm cmd config rank table file, config by ranktable.json");
        std::string clusterInfo = InstallPath::ResolveToInstallRoot("data/ranktable.json");
        std::ifstream f(clusterInfo.c_str());
        if (!f.good()) {
            HCCL_VM_ERROR("ranktable.json file not found: {}", clusterInfo);
            return HcclVmResult::HCCL_SIM_E_INTERNAL;
        }
        // 从ranktable文件中读取通信域相关表项
        ret = AscendClusterTopoParser::GetInstance().ParseRanktableAndInitCommDomain(clusterInfo);
    }
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("InitHvmCommEnv failed");
        return ret;
    }
    setenv("RANK_TABLE_FILE", InstallPath::ResolveToInstallRoot("data/ranktable.json").c_str(), 1);

    return HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD;
}

static void StopListenThreads()
{
    g_serverListenFlag.store(false);
    g_runnerListenFlag.store(false);

    try {
        PosixMessageQueue wakeup(MQ_REQ_NAME, O_WRONLY | O_NONBLOCK);
        wakeup.Send("", 1, 0);
    } catch (...) {
    }

    if (g_serverThread && g_serverThread->joinable()) {
        g_serverThread->join();
    }
    if (g_runnerThread && g_runnerThread->joinable()) {
        g_runnerThread->join();
    }

    delete g_serverThread;
    delete g_runnerThread;
    g_serverThread = nullptr;
    g_runnerThread = nullptr;
}

static void ArchiveLogsAndData()
{
    const std::string& installRoot = InstallPath::GetHcclVmInstallAbsPath();
    if (installRoot.empty()) {
        return;
    }
    std::string archiveScript = installRoot + "/script/archive.sh";
    if (access(archiveScript.c_str(), F_OK) != 0) {
        return;
    }
    int status = system(("bash " + archiveScript).c_str());
    if (status != 0) {
        HCCL_VM_WARN("archive.sh exited with code {:d}", status);
    }
}

HcclVmResult HcclVmExit()
{
    // hccl-vm 退出清除所有使用资源
    StopListenThreads();
    HCCL_VM_INFO("start Destroy SharedMemory.");

    CleanupMessageQueues();
    HcclPluginManager& pluginManager = HcclPluginManager::GetInstance();
    auto ret = pluginManager.StopAllPlugins();

    const fs::path backupDir = fs::path(InstallPath::ResolveToInstallRoot("data/" + MakeDataBackupTimestamp()));
    BackupAivTaskFiles(backupDir);

    HCCL_VM_INFO("start Destroy ALL Resources.");
    if (sim::IsCheckOnlyMode()) {
        sim::MemoryManager::GetInstance().FreeMemByName(sim::CommPoolPolicy::kPoolName);
    }
    // 清空本进程 pid 目录（正常退出路径），不含其他实例目录。
    sim::SimResourceRoot::GetInstance().Cleanup();
    FlushLog();
    ArchiveLogsAndData();
    return ret;
}

HcclVmResult InstallUserPlugin(std::string argStr)
{
    // 处理插件tag和路径
    HcclVmResult ret{HcclVmResult::HCCL_SIM_HOST_ERROR_CMD};
    if (argStr.empty() || argStr[0] != '@') {
        HCCL_VM_ERROR("plugin tag should start with '@', invalid tag: {}", argStr);
        return ret;
    }
    argStr.erase(argStr.begin());

    // 注册插件
    HcclPluginManager& pluginManager = HcclPluginManager::GetInstance();
    ret = pluginManager.RegisterPlugin(argStr);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("Install plugin [{}] failed", argStr);
        return ret;
    }

    // 装 Runner 时若仅校验模式开着，复用池仍在、大块仍会引流，可能覆盖 Runner 真实数据，告警提示。
    if (argStr == "runner" && sim::IsCheckOnlyMode()) {
        HCCL_VM_WARN("check-only mode on while runner installed; big-block contents are not guaranteed");
    }

    return HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD;
}

HcclVmResult RunUserPlugin(std::string argStr)
{
    nlohmann::json j;
    HcclVmResult ret = DumpData(j);

    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("DumpData failed");
        return ret;
    }
    HcclPluginManager& pluginManager = HcclPluginManager::GetInstance();
    ret = pluginManager.SendMessageToPlugin(argStr.substr(1), "status", j);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("Run Plugin failed");
        return ret;
    }
    return HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD;
}

HcclVmResult UninstallUserPlugin(std::string argStr)
{
    std::vector<std::string> pluginTags{};

    size_t start = 0;
    size_t end = argStr.find(',');
    while (end != std::string::npos) {
        std::string tag = argStr.substr(start, end - start);
        tag.erase(tag.begin());
        pluginTags.push_back(tag);
        start = end + 1;
        end = argStr.find(',', start);
    }
    std::string lastTag = argStr.substr(start);
    lastTag.erase(lastTag.begin());
    pluginTags.push_back(lastTag);

    HcclPluginManager& pluginManager = HcclPluginManager::GetInstance();
    auto rets = pluginManager.StopPlugins(pluginTags);

    for (int i = 0; i < pluginTags.size(); ++i) {
        if (rets[i] != HcclVmResult::HCCL_SIM_SUCCESS) {
            HCCL_VM_ERROR("plugin Uninstall fail : {}", pluginTags[i]);
            return rets[i];
        }
    }

    return HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD;
}

void ShowUserPlugin()
{
    std::vector<std::string> listPlugins{};
    HcclPluginManager& pluginManager = HcclPluginManager::GetInstance();
    listPlugins = pluginManager.GetPluginStatus();

    if (listPlugins.empty()) {
        HCCL_VM_INFO("no plugin installed in hccl_vm");
    } else {
        for (auto& plugin : listPlugins) {
            HCCL_VM_INFO("{}", plugin);
        }
    }
    return;
}

HcclVmResult SetConsoleLogLevel(int level)
{
    (void)level;
    HCCL_VM_WARN("set console log level is disabled because ProxyConfig shared memory is removed");
    return HcclVmResult::HCCL_SIM_HOST_ERROR_CMD;
}

HcclVmResult SetFileLogLevel(int level)
{
    (void)level;
    HCCL_VM_WARN("set file log level is disabled because ProxyConfig shared memory is removed");
    return HcclVmResult::HCCL_SIM_HOST_ERROR_CMD;
}

HcclVmResult ShowCurrentLogLevel()
{
    HCCL_VM_WARN("show log level is disabled because ProxyConfig shared memory is removed");
    return HcclVmResult::HCCL_SIM_HOST_ERROR_CMD;
}

HcclVmResult StartHvmCmd()
{
    // 初始化HOST通信队列
    // 清理本进程两条消息队列，并回收历史崩溃残留的僵尸队列（不影响并发存活实例）
    CleanupMessageQueues();
    // 启动监听线程
    g_serverThread = new std::thread(ServerListen);
    g_runnerThread = new std::thread(RunnerListen);

    // Child Process(Bash)
    // 劫持库存在性判断
    std::string hcclVmbin = InstallPath::ResolveToInstallRoot("bin/hccl-vm");
    std::string libDir = "lib/" + GetArchStr() + "/";
    std::string proxyPathL0 = InstallPath::ResolveToInstallRoot(libDir + "libhccl_proxy_level0.so");
    std::string proxyPathL1 = InstallPath::ResolveToInstallRoot(libDir + "libhccl_proxy_level1.so");
    std::string proxyPathL2 = InstallPath::ResolveToInstallRoot(libDir + "libhccl_proxy_level2.so");
    if (!fs::exists(proxyPathL0) || !fs::exists(proxyPathL2)) {
        HCCL_VM_ERROR(
            "proxy hacking .so not found: l0={}, l2={}. please check your proxy hacking .so:"
            "1. Whether the hook library has been successfully built and installed. 2. Whether the simulation level "
            "matches the proxy hook library version. Current simulation level: {}, Default simulation level: 2",
            proxyPathL0, proxyPathL2, g_hcclVmLevel);
        return HCCL_SIM_HOST_ERROR_CMD;
    }
    if (g_hcclVmLevel == 1 && !fs::exists(proxyPathL1)) {
        HCCL_VM_ERROR(
            "proxy hacking .so not found: l1={}. please check your proxy hacking .so:"
            "1. Whether the hook library has been successfully built and installed. 2. Whether the simulation level "
            "matches the proxy hook library version. Current simulation level: {}, Default simulation level: 2",
            proxyPathL1, g_hcclVmLevel);
        return HCCL_SIM_HOST_ERROR_CMD;
    }
    std::string preload = proxyPathL0 + ":" + proxyPathL2;
    if (g_hcclVmLevel == 1) {
        preload = proxyPathL0 + ":" + proxyPathL1 + ":" + proxyPathL2;
    }

    // 管道处理的用途是隔绝进程终端在std::cout中的残留
    int pipefds[2] = {-1, -1};
    if (pipe(pipefds) == -1) {
        HCCL_VM_ERROR("pipe failed");
        return HCCL_SIM_HOST_ERROR_CMD;
    }

    std::string safeBin = "'" + hcclVmbin + "'";
    std::string fdNum = std::to_string(pipefds[0]);
    std::string bashrcHack = "__HVM_SAVED_PATH=\"$PATH\";\n"
                             "__HVM_SAVED_LD_PRELOAD=\"$LD_PRELOAD\";\n"
                             "if [ -f ~/.bashrc ]; then . ~/.bashrc; fi;\n"
                             "export PATH=\"$__HVM_SAVED_PATH:$PATH\";\n"
                             "export LD_PRELOAD=\"$__HVM_SAVED_LD_PRELOAD\";\n"
                             "alias hccl-vm=\""
                             + safeBin
                             + "\";\n"
                               "PS1='(hvm)$> ';\n"
                               "unset __HVM_SAVED_PATH __HVM_SAVED_LD_PRELOAD;\n"
                               "exec "
                             + fdNum + "<&-;\n";
    ssize_t written = write(pipefds[1], bashrcHack.c_str(), bashrcHack.size());
    if (written != static_cast<ssize_t>(bashrcHack.size())) {
        HCCL_VM_ERROR("Failed to write full script to pipe");
        close(pipefds[0]);
        close(pipefds[1]);
        return HCCL_SIM_HOST_ERROR_CMD;
    }
    close(pipefds[1]);
    g_hcclVmBashFlag = true;

    std::string devFdPath = "/dev/fd/" + std::to_string(pipefds[0]);

    char* bashArgv[]
        = {const_cast<char*>("bash"), const_cast<char*>("--rcfile"), const_cast<char*>(devFdPath.c_str()),
           const_cast<char*>("-i"), nullptr};

    // Fork Bash
    pid_t pid = fork();
    if (pid == -1) {
        HCCL_VM_ERROR("fork failed: {}", std::strerror(errno));
        auto ret = HcclVmExit();
        close(pipefds[0]);
        return (ret == HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD) ? HcclVmResult::HCCL_SIM_HOST_ERROR_CMD : ret;
    } else if (pid == 0) {
        setenv(HVM_BASH_ENV_KEY.c_str(), g_binDir.c_str(), 1);
        setenv("LD_PRELOAD", preload.c_str(), 1);
        setenv("HCCL_VM_INSTALL_ROOT", g_binDir.c_str(), 1);
        setenv("HCCL_VM_LEVEL", std::to_string(g_hcclVmLevel).c_str(), 1);
        setenv("RANK_TABLE_FILE", InstallPath::ResolveToInstallRoot("data/ranktable.json").c_str(), 1);
        execv("/bin/bash", bashArgv);
        // exec 失败必须 _exit：fork child 不运行继承进程的 atexit/静态析构
        // （可能触碰父进程 Storage 会话/SQLite 连接等继承对象）。
        _exit(1);
    } else {
        // Parent Process (Host)
        // 等待 bash 结束 (阻塞等待，保持Host存活)
        int status;
        waitpid(pid, &status, 0);
        HCCL_VM_INFO("Shell exited. Host shutting down.");
        auto ret = HcclVmExit();
    }
    return HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD;
}

void StartHostClient(int argc, char* argv[])
{
    try {
        // 打开队列
        PosixMessageQueue mqReq(MQ_REQ_NAME, O_RDWR);
        PosixMessageQueue mqResp(MQ_RESP_NAME, O_RDWR);

        std::string cmd = ArgvToString(argc, argv);
        if (argc == 1) {
            cmd += " --help";
        }
        // 向主host发生命令行
        mqReq.Send(cmd.data(), cmd.size(), 0);
        // 等待回执，简单回执
        timespec deadline = MakeDeadlineAfterMs(HOST_CLIENT_RESP_TIMEOUT_MS);
        char buffer[MAX_MSG_SIZE];
        ssize_t recvdSize = 0;
        unsigned int priority = 0;
        bool hasMessage = mqResp.TimedReceive(&buffer, MAX_MSG_SIZE, &priority, deadline, recvdSize);
        if (hasMessage) {
            HCCL_VM_DEBUG("Client : Command parsing response received: {}", std::string(buffer, recvdSize));
        } else {
            HCCL_VM_WARN("Client : Command parsing out of time, Client quit");
        }
    } catch (const std::exception& e) {
        HCCL_VM_ERROR("Host client aborted: {}", e.what());
    }
}

void ServerListen()
{
    try {
        // 创建队列
        g_serverListenFlag.store(true);
        mq_attr attr = MakeMessageQueueAttr();
        PosixMessageQueue mqReq(MQ_REQ_NAME, O_CREAT | O_EXCL | O_RDWR, &attr);
        PosixMessageQueue mqResp(MQ_RESP_NAME, O_CREAT | O_EXCL | O_RDWR, &attr);
        HCCL_VM_INFO("HOST Server listening...");

        while (g_serverListenFlag.load()) {
            char buffer[MAX_MSG_SIZE];
            unsigned int priority = 0;
            ssize_t recvdSize = -1;
            timespec deadline = MakeDeadlineAfterMs(500);
            if (!mqReq.TimedReceive(buffer, MAX_MSG_SIZE, &priority, deadline, recvdSize)) {
                continue;
            }

            std::string cmd(buffer, recvdSize);
            HCCL_VM_DEBUG("HOST : Command parsing request received: {}", cmd);

            // 执行接收到的命令逻辑
            ParseCommand(cmd);
            std::string resp = "Success SubCommand Received";
            // 发送回执, 简易回执
            mqResp.Send(resp.data(), resp.size(), 0);
        }
        HCCL_VM_INFO("HOST Server listening thread exit...");
    } catch (const std::exception& e) {
        HCCL_VM_ERROR("HOST Server listening thread aborted: {}", e.what());
    }
}

// 监听proxy进程，等待所有proxy进程进入同步状态。
// 循环监听：每轮 dump 后清空 DeviceStatus 继续监听，支持 mock-comm 后多次 run。
void RunnerListen()
{
    // dump 开关 HCCLVM_ENABLE_DUMP_DATA: 未设置或为 0 时不监听/不触发 dump，直接退出.
    const char* enableDumpData = std::getenv("HCCLVM_ENABLE_DUMP_DATA");
    if (enableDumpData == nullptr || std::string(enableDumpData).empty() || std::string(enableDumpData) == "0") {
        HCCL_VM_WARN("HCCLVM_ENABLE_DUMP_DATA is not set or 0, skip runner dump listening.");
        return;
    }
    // 空闲轮询间隔：就绪状态由 proxy 写入（aclrtSynchronizeStream）且持续到监听器
    // 消费（DeleteAll）期间不会自行消失，无需高频探测；代价是就绪发现最多延迟
    // 约一个轮询间隔，不承诺实时通知。就绪计数为 0 时等待期可能很长（建链/加载），
    // 保持 500ms。
    constexpr std::chrono::milliseconds kRunnerListenIdlePoll{500};
    // 收敛轮询间隔：就绪计数 > 0 且未达在线数的收敛窗口——设备陆续就绪的延迟
    // 敏感阶段，最后一台就绪后的发现延迟最多一个间隔；逐轮交接（每轮 DeleteAll
    // 后等 proxy 重写）时轮数多则每轮平均一半间隔的发现延迟按轮累计，缩短间隔
    // 直接压缩累计值。两次 Count 非原子的短暂不一致靠下一轮收敛，更短的间隔
    // 收敛更快，语义不变。
    constexpr std::chrono::milliseconds kRunnerListenConvergePoll{50};
    // 查询失败重试间隔：异常路径尽快恢复，与空闲轮询用途不同，保持原 10ms。
    constexpr std::chrono::milliseconds kRunnerListenErrorRetry{10};
    HCCL_VM_INFO("Runner listening...");
    g_runnerListenFlag.store(true);
    while (g_runnerListenFlag.load()) {
        // 1. 先计数就绪 DeviceStatus（synchronize_strategy=1，命中单列索引，数据库端
        //    计数聚合不回表）。数量为 0 时本轮无需再查 Device：原触发条件
        //    "设备数非零且数量相等"在该取值下不可能成立。
        auto readyStatusResult = sim::runtime::Db::Count<sim::runtime::DeviceStatus>(
            HcclSim::Storage::Eq(&sim::runtime::DeviceStatus::synchronize_strategy, uint64_t{1}));
        if (!readyStatusResult.ok() || !readyStatusResult.value.has_value()) {
            // 诊断串必须可见：引擎层失败不落日志（FailStatementLocked 只回填
            // DbResult），此前的固定文案把根因吞掉，一次排障 80% 时间耗在翻日志。
            HCCL_VM_WARN(
                "runner listener DeviceStatus count failed; retrying, code={}, diagnostic={}",
                static_cast<int>(readyStatusResult.code), readyStatusResult.diagnostic);
            std::this_thread::sleep_for(kRunnerListenErrorRetry);
            continue;
        }
        const auto readyStatusCount = *readyStatusResult.value;
        if (readyStatusCount == 0) {
            std::this_thread::sleep_for(kRunnerListenIdlePoll);
            continue;
        }

        // 2. 再计数在线 Device（status=1，为 (status, server_id) 索引首列，同样走
        //    数据库端计数快路）。两次计数是先后执行的独立查询，不是跨表原子快照；
        //    就绪状态持续到本轮消费，短暂不一致由下一轮轮询收敛，与原实现
        //    两次查询的语义一致。
        auto onlineDevicesResult = sim::runtime::Db::Count<sim::runtime::Device>(
            HcclSim::Storage::Eq(&sim::runtime::Device::status, uint64_t{1}));
        if (!onlineDevicesResult.ok() || !onlineDevicesResult.value.has_value()) {
            HCCL_VM_WARN(
                "runner listener Device count failed; retrying, code={}, diagnostic={}",
                static_cast<int>(onlineDevicesResult.code), onlineDevicesResult.diagnostic);
            std::this_thread::sleep_for(kRunnerListenErrorRetry);
            continue;
        }
        const auto onlineDeviceCount = *onlineDevicesResult.value;
        if (onlineDeviceCount != readyStatusCount || onlineDeviceCount == 0) {
            std::this_thread::sleep_for(kRunnerListenConvergePoll);
            continue;
        }

        HCCL_VM_INFO("{:d} rank ready, dump runner data...", readyStatusCount);
        const auto ret = DumpDataToFile("runner");
        if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
            HCCL_VM_WARN("dump data to file failed. ret: {:d}", static_cast<int>(ret));
        }

        // 清空本轮状态，下一轮由 proxy 重新写入 synchronize_strategy=1。
        (void)sim::runtime::Db::DeleteAll<sim::runtime::DeviceStatus>(HcclSim::Storage::ClearMode::KEEP_SEQUENCE);
    }
    HCCL_VM_INFO("Runner listening thread exit.");
}

void ParseCommand(std::string& cmd)
{
    CLI::App app{"Hccl Virtual Simulator"};
    app.set_help_all_flag("--help-all", "show all the sub command help");

    auto commands = CommandRegistry::CreateAll();

    for (auto& command : commands) {
        command->Setup(app);
    }
    try {
        app.parse(cmd, true);            // true 表示命令第一个字段为程序名
    } catch (const CLI::ParseError& e) { // 非help请求，强行打印help
        if (e.get_exit_code() != 0) {
            std::cerr << "[HVM] [ERROR] para error, please check:\n\n" << app.help() << std::endl;
        }
        if (g_hcclVmBashFlag) {
            app.exit(e);
        } else {
            std::exit(app.exit(e));
        }
    }
}

static void BackupDatabase(const std::string& srcPath)
{
    if (!fs::exists(srcPath)) {
        HCCL_VM_INFO("Source database {} does not exist, skipping backup.", srcPath);
        return;
    }

    fs::path dataDir = fs::path(InstallPath::ResolveToInstallRoot("data"));
    fs::create_directories(dataDir);

    auto now = std::chrono::system_clock::now();
    auto timeT = std::chrono::system_clock::to_time_t(now);
    std::tm tmBuf{};
    localtime_r(&timeT, &tmBuf);
    std::ostringstream oss;
    oss << std::put_time(&tmBuf, "%Y%m%d_%H%M%S");

    std::string destPath = (dataDir / ("hccl_vm_data_backup_" + oss.str() + ".db")).string();

    auto ret = sim::operation::BackupOperationData(destPath);
    if (ret != HcclSim::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("Failed to backup database to {}", destPath);
    } else {
        HCCL_VM_INFO("Database backed up to: {}", destPath);
    }
}

static std::string MakeDataBackupTimestamp()
{
    auto now = std::chrono::system_clock::now();
    auto timeT = std::chrono::system_clock::to_time_t(now);
    std::tm tmBuf{};
    localtime_r(&timeT, &tmBuf);
    std::ostringstream oss;
    oss << std::put_time(&tmBuf, "%Y%m%d%H%M%S");
    return oss.str();
}

static bool EnsureBackupDir(const fs::path& backupDir)
{
    std::error_code ec;
    fs::create_directories(backupDir, ec);
    if (ec) {
        HCCL_VM_WARN("failed to create backup directory {}: {}", backupDir.string(), ec.message());
        return false;
    }
    return true;
}

static void BackupBinFiles(const fs::path& backupDir)
{
    fs::path dataDir = fs::path(InstallPath::ResolveToInstallRoot("data"));
    std::vector<fs::path> binFiles
        = {dataDir / "runner_hcclvm_instr_data.bin", dataDir / "runner_hcclvm_syn_data.bin",
           dataDir / "runner_hcclvm_task_data.bin"};

    bool hasBinFile = std::any_of(binFiles.begin(), binFiles.end(), [](const fs::path& f) {
        return fs::exists(f);
    });
    if (!hasBinFile) {
        return;
    }

    if (!EnsureBackupDir(backupDir)) {
        return;
    }

    std::error_code ec;
    for (const auto& f : binFiles) {
        if (!fs::exists(f)) {
            continue;
        }
        fs::rename(f, backupDir / f.filename(), ec);
        if (ec) {
            HCCL_VM_WARN("failed to move {} to {}: {}", f.string(), (backupDir / f.filename()).string(), ec.message());
        }
    }
}

static void BackupAivTaskFiles(const fs::path& backupDir)
{
    fs::path dataDir = fs::path(InstallPath::ResolveToInstallRoot("data"));
    std::error_code ec;
    if (!fs::exists(dataDir, ec) || ec || !fs::is_directory(dataDir, ec)) {
        if (ec) {
            HCCL_VM_WARN("failed to access data directory {}: {}", dataDir.string(), ec.message());
        }
        return;
    }

    std::vector<fs::path> aivTaskFiles;
    static const std::string aivFilePrefix = "hcclvm_aiv_";
    fs::directory_iterator iter(dataDir, ec);
    if (ec) {
        HCCL_VM_WARN("failed to iterate data directory {}: {}", dataDir.string(), ec.message());
        return;
    }
    for (const fs::directory_iterator end; iter != end; iter.increment(ec)) {
        if (ec) {
            HCCL_VM_WARN("failed to iterate data directory {}: {}", dataDir.string(), ec.message());
            return;
        }

        const fs::path filePath = iter->path();
        std::error_code statusEc;
        const fs::file_status status = fs::symlink_status(filePath, statusEc);
        if (statusEc || !fs::is_regular_file(status)) {
            continue;
        }

        const std::string fileName = filePath.filename().string();
        if (StartsWith(fileName, aivFilePrefix)) {
            aivTaskFiles.emplace_back(filePath);
        }
    }

    if (aivTaskFiles.empty()) {
        return;
    }
    if (!EnsureBackupDir(backupDir)) {
        return;
    }

    for (const auto& f : aivTaskFiles) {
        ec.clear();
        fs::rename(f, backupDir / f.filename(), ec);
        if (ec) {
            HCCL_VM_WARN("failed to move {} to {}: {}", f.string(), (backupDir / f.filename()).string(), ec.message());
        }
    }
}

HcclVmResult ClearDbTables()
{
    const fs::path backupDir = fs::path(InstallPath::ResolveToInstallRoot("data/" + MakeDataBackupTimestamp()));
    BackupBinFiles(backupDir);
    BackupAivTaskFiles(backupDir);
    BackupDatabase(InstallPath::ResolveToInstallRoot("data/hccl_vm_data.db"));

    // 1. 清空静态操作数据表（迁移前 DELETE FROM 语义：保留自增序列）
    (void)sim::operation::ClearStaticData();

    // 2. 销毁全部动态任务表 (opTask_P_*)：枚举物理表并按 PID scope 逐一销毁
    (void)sim::operation::DropDynamicData();

    // 3. 清空 RunnerDB (内存/SHM) 表。单次清空表直接走一次性原子入口
    //    （CcuResource 表已随 main 分支清理移除，无逐行重写需求）。
    (void)sim::runtime::Db::DeleteAll<sim::runtime::PhyMemBlock>(HcclSim::Storage::ClearMode::RESET_SEQUENCE);
    (void)sim::runtime::Db::DeleteAll<sim::runtime::VirtualMemBlock>(HcclSim::Storage::ClearMode::RESET_SEQUENCE);
    (void)sim::runtime::Db::DeleteAll<sim::runtime::RaDevice>(HcclSim::Storage::ClearMode::RESET_SEQUENCE);
    (void)sim::runtime::Db::DeleteAll<sim::runtime::RaContext>(HcclSim::Storage::ClearMode::RESET_SEQUENCE);
    (void)sim::runtime::Db::DeleteAll<sim::runtime::RaQP>(HcclSim::Storage::ClearMode::RESET_SEQUENCE);
    (void)sim::runtime::Db::DeleteAll<sim::runtime::RaJetty>(HcclSim::Storage::ClearMode::RESET_SEQUENCE);
    (void)sim::runtime::Db::DeleteAll<sim::runtime::EndPointPair>(HcclSim::Storage::ClearMode::RESET_SEQUENCE);
    (void)sim::runtime::Db::DeleteAll<sim::runtime::Link>(HcclSim::Storage::ClearMode::RESET_SEQUENCE);
    (void)sim::runtime::Db::DeleteAll<sim::runtime::MemoryLayout>(HcclSim::Storage::ClearMode::RESET_SEQUENCE);
    (void)sim::runtime::Db::DeleteAll<sim::runtime::SimModelData>(HcclSim::Storage::ClearMode::RESET_SEQUENCE);
    (void)sim::runtime::Db::DeleteAll<sim::runtime::CcuChannel>(HcclSim::Storage::ClearMode::RESET_SEQUENCE);
    (void)sim::runtime::Db::DeleteAll<sim::runtime::DeviceStatus>(HcclSim::Storage::ClearMode::RESET_SEQUENCE);
    (void)sim::runtime::Db::DeleteAll<sim::runtime::Task>(HcclSim::Storage::ClearMode::RESET_SEQUENCE);
    (void)sim::runtime::Db::DeleteAll<sim::runtime::Runner>(HcclSim::Storage::ClearMode::RESET_SEQUENCE);
    (void)sim::runtime::Db::DeleteAll<sim::runtime::RaSocketPair>(HcclSim::Storage::ClearMode::RESET_SEQUENCE);
    (void)sim::runtime::Db::DeleteAll<sim::runtime::TopoMetaConfig>(HcclSim::Storage::ClearMode::RESET_SEQUENCE);
    (void)sim::runtime::Db::DeleteAll<sim::runtime::HcclThread>(HcclSim::Storage::ClearMode::RESET_SEQUENCE);
    (void)sim::runtime::Db::DeleteAll<sim::runtime::HcclChannel>(HcclSim::Storage::ClearMode::RESET_SEQUENCE);
    (void)sim::runtime::Db::DeleteAll<sim::runtime::CommunicatorDestroySync>(
        HcclSim::Storage::ClearMode::RESET_SEQUENCE);
    (void)sim::runtime::Db::DeleteAll<sim::runtime::Communicator>(HcclSim::Storage::ClearMode::RESET_SEQUENCE);
    (void)sim::runtime::Db::DeleteAll<sim::runtime::Notify>(HcclSim::Storage::ClearMode::RESET_SEQUENCE);
    (void)sim::runtime::Db::DeleteAll<sim::runtime::HcclBuffer>(HcclSim::Storage::ClearMode::RESET_SEQUENCE);
    (void)sim::runtime::Db::DeleteAll<sim::runtime::HcclMem>(HcclSim::Storage::ClearMode::RESET_SEQUENCE);
    (void)sim::runtime::Db::DeleteAll<sim::runtime::HcommEndpoint>(HcclSim::Storage::ClearMode::RESET_SEQUENCE);
    (void)sim::runtime::Db::DeleteAll<sim::runtime::HcommMemReg>(HcclSim::Storage::ClearMode::RESET_SEQUENCE);
    auto engineCtxResetFn = reinterpret_cast<void (*)()>(dlsym(RTLD_DEFAULT, "HcclEngineCtxResetAll"));
    if (engineCtxResetFn != nullptr) {
        engineCtxResetFn();
    }

    // 仅清理本进程 pid 目录下的遗留 DEV/ra_sock_ 文件，不整机扫 /dev/shm（跨实例隔离）。
    sim::SimResourceRoot::GetInstance().CleanShmByPrefix({"DEV", "ra_sock_"});

    return HcclVmResult::HCCL_SIM_HOST_SUCCESS_CMD;
}

extern uint64_t g_cur_server_key;

HcclVmResult HcclVmResetCommDomain()
{
    AscendClusterTopoParser::GetInstance().SetClusterStatus(HvmClusterStatus::COMM_DOMAIN_UNINIT);
    // 重置 Host 进程中缓存的 server key与通信域配置缓存，防止跨用例残留
    g_cur_server_key = 0;
    sim::runtime::ResetCommConfigData();
    // 重置device的逻辑ID
    auto ret2 = sim::runtime::ResetAllDeviceLogicId();
    if (!ret2) {
        HCCL_VM_ERROR("reset all device logic id failed.");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    // 清除Communicator表
    (void)sim::runtime::Db::DeleteAll<sim::runtime::Communicator>(HcclSim::Storage::ClearMode::RESET_SEQUENCE);
    return ClearDbTables();
}

HcclVmResult CopyFile(const std::string& clusterDir)
{
    // 1. 拼接源文件和目标文件路径
    auto srcPath = clusterDir + "/superpod0/server0/topo.json";
    fs::create_directories(fs::path(InstallPath::ResolveToInstallRoot("data")));
    auto destPath = InstallPath::ResolveToInstallRoot("data/topo.json");

    // 显式提示：文件已存在，将覆盖
    if (fs::exists(destPath)) {
        HCCL_VM_WARN("target file already exists, will overwrite: topo.json");
    }

    // 2. 二进制模式打开文件（显式指定 trunc 覆盖，语义更清晰）
    std::ifstream srcFile(srcPath, std::ios::binary);
    std::ofstream destFile(destPath, std::ios::binary | std::ios::trunc);

    // 打开失败检查
    if (!srcFile.is_open() || !destFile.is_open()) {
        HCCL_VM_ERROR("open file failed. src: {}, dest: {}", srcPath, destPath);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    // 3. 执行文件拷贝
    destFile << srcFile.rdbuf();

    // 4. 检查拷贝是否失败
    if (srcFile.bad() || destFile.bad()) {
        HCCL_VM_ERROR("copy file failed. src: {}, dest: {}", srcPath, destPath);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    // 5. 手动关闭文件（确保数据写入磁盘）
    srcFile.close();
    destFile.close();

    // 6. 校验/etc/hccl_rootinfo.json文件中topo_file_path路径是否为destPath
    std::string topoFilePath = "/etc/hccl_rootinfo.json";
    std::ifstream rootinfoFile(topoFilePath, std::ios::binary);
    if (rootinfoFile.is_open()) {
        std::string line;
        while (std::getline(rootinfoFile, line)) {
            if (line.find("topo_file_path") != std::string::npos) {
                if (line.find(destPath) == std::string::npos) {
                    HCCL_VM_ERROR("wrong topo_file_path in rootinfo.json file. path: {}", topoFilePath);
                    return HcclVmResult::HCCL_SIM_E_INTERNAL;
                }
                break;
            }
        }
        rootinfoFile.close();
    } else {
        HCCL_VM_WARN("{} not exist", topoFilePath);
    }

    HCCL_VM_INFO("topo.json copy success!");

    return HcclVmResult::HCCL_SIM_SUCCESS;
}
