/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "store_sim_resource_root.h"

#include <algorithm>
#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <mqueue.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

#include "sim_log.h"

namespace sim {

namespace {
    constexpr const char* kEnvKey = "HCCL_VM_RESOURCE_ROOT";
    const char* kShmBase = "/dev/shm";
    // 目录名前缀，使 /dev/shm 下本工具的根目录可辨识：/dev/shm/hvm_<pid>/。
    const char* kShmPrefix = "hvm_";

    std::string ResolveRoot()
    {
        const char* env = std::getenv(kEnvKey);
        if (env != nullptr && env[0] != '\0') {
            return env;
        }
        return std::string(kShmBase) + "/" + kShmPrefix + std::to_string(::getpid());
    }

    // 递归删除文件或目录（等价 std::filesystem::remove_all，GCC 7.3 无 std::filesystem）。
    bool RemoveTree(const std::string& path)
    {
        struct stat st;
        if (::lstat(path.c_str(), &st) != 0) {
            return true; // 不存在视为成功
        }
        if (!S_ISDIR(st.st_mode)) {
            return ::unlink(path.c_str()) == 0;
        }
        DIR* dir = ::opendir(path.c_str());
        if (dir == nullptr) {
            return false;
        }
        bool ok = true;
        struct dirent* entry = nullptr;
        while ((entry = ::readdir(dir)) != nullptr) {
            if (::strcmp(entry->d_name, ".") == 0 || ::strcmp(entry->d_name, "..") == 0) {
                continue;
            }
            if (!RemoveTree(path + "/" + entry->d_name)) {
                ok = false;
            }
        }
        ::closedir(dir);
        if (::rmdir(path.c_str()) != 0) {
            ok = false;
        }
        return ok;
    }

    // 逐级创建目录（等价 std::filesystem::create_directories）。
    bool MkdirP(const std::string& path)
    {
        if (path.empty()) {
            return false;
        }
        std::string cur = path;
        size_t pos = 0;
        while ((pos = cur.find('/', pos + 1)) != std::string::npos) {
            std::string prefix = cur.substr(0, pos);
            if (prefix.empty()) {
                continue;
            }
            if (::mkdir(prefix.c_str(), 0755) != 0 && errno != EEXIST) {
                return false;
            }
        }
        if (::mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST) {
            return false;
        }
        return true;
    }
} // namespace

SimResourceRoot& SimResourceRoot::GetInstance()
{
    static SimResourceRoot instance;
    return instance;
}

SimResourceRoot::SimResourceRoot()
    : m_ownerPid(-1) // 构造时未确定 owner；由 Init() 显式设为当前进程 pid。
      ,
      m_root(ResolveRoot()),
      m_syncDir(m_root + "/sync")
{}

SimResourceRoot::~SimResourceRoot() = default;

bool SimResourceRoot::Init()
{
    // 强清自身目录（pid 复用/上次崩溃残留）。
    RemoveTree(m_root);
    if (!MkdirP(m_syncDir)) {
        HCCL_VM_ERROR("[SimResourceRoot] create {} failed: errno={}", m_syncDir, errno);
        return false;
    }
    if (::setenv(kEnvKey, m_root.c_str(), 1) != 0) {
        HCCL_VM_ERROR("[SimResourceRoot] setenv {} failed, errno={}", kEnvKey, errno);
        return false;
    }
    // 创建目录成功的进程即为资源 owner；后续 Cleanup 仅由它（及其同 pid 主进程）真正删除。
    m_ownerPid = static_cast<int>(::getpid());
    HCCL_VM_INFO("[SimResourceRoot] root={}, sync={}", m_root, m_syncDir);
    return true;
}

std::string SimResourceRoot::BuildName(const std::string& logical)
{
    std::string clean = logical;
    while (!clean.empty() && clean[0] == '/') {
        clean.erase(0, 1);
    }
    const std::string& root = GetInstance().m_root;
    return root + "/" + clean;
}

const std::string& SimResourceRoot::GetRoot() const { return m_root; }

const std::string& SimResourceRoot::GetSyncDir() const { return m_syncDir; }

std::string SimResourceRoot::GetRunnerLockPath() { return BuildName("hccl_vm_runner.lock"); }

std::string SimResourceRoot::GetTag()
{
    const std::string& root = GetInstance().m_root;
    const size_t pos = root.find_last_of('/');
    return pos == std::string::npos ? root : root.substr(pos + 1);
}

void SimResourceRoot::Cleanup()
{
    // 仅 owner 进程删除宿主目录；未初始化或继承者子进程退出时绝不误删。
    if (m_ownerPid < 0 || static_cast<int>(::getpid()) != m_ownerPid) {
        return;
    }
    if (!RemoveTree(m_root)) {
        HCCL_VM_WARN("[SimResourceRoot] cleanup {} failed", m_root);
    }
}

// unlink 本实例在 mqueue 命名空间里创建的两条消息队列（名字由 tag 推导，
// 与 cmd_base_utils.cc 的 ToPosixMqName/MQ_REQ_NAME/MQ_RESP_NAME 保持一致）。
// 不写日志，供信号 handler 调用。正常退出时 HcclVmExit 已显式 unlink，此处为无害 no-op 兜底。
static void RemoveOwnMessageQueues()
{
    const std::string tag = SimResourceRoot::GetTag();
    (void)::mq_unlink(("/host_mq_request_" + tag).c_str());
    (void)::mq_unlink(("/host_mq_response_" + tag).c_str());
}

void SimResourceRoot::CleanupQuiet()
{
    if (m_ownerPid < 0 || static_cast<int>(::getpid()) != m_ownerPid) {
        return;
    }
    RemoveTree(m_root);
    RemoveOwnMessageQueues();
}

// 信号 handler：尽力删除自身目录（不写日志、避免 spdlog 重入），随后恢复默认并重抛原信号。
// 注：RemoveTree 内含 std::string 拼接，严格说非 async-signal-safe；此处为崩溃场景的尽力清理。
static void ExitCleanupSignalHandler(int signo)
{
    sim::SimResourceRoot::GetInstance().CleanupQuiet();
    ::signal(signo, SIG_DFL);
    ::raise(signo);
}

void SimResourceRoot::RegisterExitCleaner()
{
    // 仅 owner 进程注册；子进程通过环境变量继承 root，其 getpid 与 owner 不同，CleanupQuiet 已兜底。
    if (GetInstance().m_ownerPid < 0 || static_cast<int>(::getpid()) != GetInstance().m_ownerPid) {
        return;
    }

    // 正常 return/exit 路径。
    std::atexit([]() {
        sim::SimResourceRoot::GetInstance().CleanupQuiet();
    });

    // 崩溃/终止信号路径（SIGKILL/SIGSTOP 不可捕获，属「外部主动 kill」边界，无法也不应清理）。
    const int handledSignals[] = {SIGINT, SIGTERM, SIGHUP, SIGQUIT, SIGABRT, SIGSEGV, SIGBUS, SIGFPE, SIGILL};
    for (int signo : handledSignals) {
        ::signal(signo, ExitCleanupSignalHandler);
    }
}

void SimResourceRoot::CleanShmByPrefix(std::initializer_list<std::string> prefixes)
{
    struct stat st;
    if (::lstat(m_root.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
        return;
    }
    DIR* dir = ::opendir(m_root.c_str());
    if (dir == nullptr) {
        return;
    }
    struct dirent* entry = nullptr;
    while ((entry = ::readdir(dir)) != nullptr) {
        if (::strcmp(entry->d_name, ".") == 0 || ::strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        std::string filePath = m_root + "/" + entry->d_name;
        struct stat entrySt;
        if (::lstat(filePath.c_str(), &entrySt) != 0 || !S_ISREG(entrySt.st_mode)) {
            continue;
        }
        const std::string fileName = entry->d_name;
        bool matched = std::any_of(prefixes.begin(), prefixes.end(), [&fileName](const std::string& p) {
            return fileName.rfind(p, 0) == 0;
        });
        if (matched) {
            ::unlink(filePath.c_str());
        }
    }
    ::closedir(dir);
}

} // namespace sim
