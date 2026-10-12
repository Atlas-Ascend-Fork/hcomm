/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STORE_SIM_RESOURCE_ROOT_H
#define STORE_SIM_RESOURCE_ROOT_H

#include <initializer_list>
#include <string>

namespace sim {

class SimResourceRoot {
public:
    static SimResourceRoot& GetInstance();

    SimResourceRoot(const SimResourceRoot&) = delete;
    SimResourceRoot& operator=(const SimResourceRoot&) = delete;

    // 主进程 start 阶段调用：强清自身 pid 目录 + mkdir root/sync + setenv HCCL_VM_RESOURCE_ROOT。
    bool Init();

    // 逻辑名 -> 完整共享内存文件路径, 如 "ra_sock_pool_0" -> "/dev/shm/hvm_<pid>/ra_sock_pool_0"。
    // 供 open()/mmap(MAP_SHARED) 使用（POSIX shm_open 不支持带子目录的名字）。
    static std::string BuildName(const std::string& logical);

    // 根目录绝对路径, 如 "/dev/shm/hvm_21843"。
    const std::string& GetRoot() const;

    // 同步文件目录, 如 "/dev/shm/hvm_21843/sync"。
    const std::string& GetSyncDir() const;

    // runner 存活探测锁文件的完整路径, 如 "/dev/shm/hvm_21843/hccl_vm_runner.lock"。
    // 主进程与 runner 进程通过读写该锁文件探测 runner 是否在位。
    static std::string GetRunnerLockPath();

    // 根目录名后缀（不带头尾斜杠）, 如 "/dev/shm/hvm_21843" -> "hvm_21843"。
    // 用于构造不能带目录的跨进程资源名（如 POSIX 消息队列名）, 实现按 hccl-vm 进程粒度隔离。
    static std::string GetTag();

    // 清空自身目录；仅 owner 进程（首次以自身 pid 建目录者）会真正删除，
    // 继承 HCCL_VM_RESOURCE_ROOT 的子进程调用时为 no-op。目录不存在则 no-op。
    void Cleanup();

    // 静默清理（不写日志），供信号/atexit 钩子调用，避免在信号上下文内重入 spdlog。
    void CleanupQuiet();

    // 注册退出清理器：atexit + 崩溃/终止信号 handler，使进程「正常/异常退出」都能删除自身目录。
    // 仅 owner 进程应调用（非 owner 调用后，清理动作本身也是 no-op）。
    static void RegisterExitCleaner();

    // 仅清理自身 root 目录下、文件名以给定前缀开头的普通文件（不扫 /dev/shm 根）。
    void CleanShmByPrefix(std::initializer_list<std::string> prefixes);

private:
    SimResourceRoot();
    ~SimResourceRoot();

    // owner 进程 pid（= 调用 Init() 创建/重建目录的那个进程）；-1 表示尚未 Init。
    // 仅当 getpid()==m_ownerPid 时 Cleanup 才真正删除目录，继承者子进程退出时不删。
    int m_ownerPid;
    std::string m_root;
    std::string m_syncDir;
};

} // namespace sim

#endif // STORE_SIM_RESOURCE_ROOT_H
