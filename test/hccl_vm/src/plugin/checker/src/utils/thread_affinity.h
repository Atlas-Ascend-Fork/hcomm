/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef THREAD_AFFINITY_H
#define THREAD_AFFINITY_H

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

#include "sim_log.h"

namespace HcclSim {

/**
 * 独占机器场景的线程绑核开关：环境变量取 1/true/on（忽略大小写）时开启。
 *
 * 机器被本进程独占时，把并行工作线程固定到不同物理核可减少线程迁移与缓存失效；
 * 共享机器上不建议开启（绑定的核随时可能被其他进程抢占，反而劣化）。
 */
constexpr const char* THREAD_CORE_BIND_ENV = "HCCL_VM_BIND_CORE";

/** 判断字符串是否非空且全为十进制数字。 */
inline bool IsAllDigits(const std::string& text)
{
    return !text.empty() && text.find_first_not_of("0123456789") == std::string::npos;
}

/** 线程绑核开关是否开启（独占机器场景由使用者显式声明）。 */
inline bool IsThreadCoreBindEnabled()
{
    const char* envValue = std::getenv(THREAD_CORE_BIND_ENV);
    if (envValue == nullptr) {
        return false;
    }
    std::string normalized;
    const std::string envText(envValue);
    for (const char ch : envText) {
        normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }
    return normalized == "1" || normalized == "true" || normalized == "on";
}

/**
 * @brief 解析 thread_siblings_list 内容（如 "0,1"、"0-3,8"），返回其中最小的 CPU 号。
 *
 * @return 最小 CPU 号；内容为空或不含合法数字时返回 -1。
 */
inline int ParseSiblingListMinCpu(const std::string& siblingList)
{
    int minCpuId = -1;
    size_t pos = 0;
    while (pos < siblingList.size()) {
        const size_t commaPos = siblingList.find(',', pos);
        const size_t partEnd = (commaPos == std::string::npos) ? siblingList.size() : commaPos;
        // 每段形如 "X" 或 "X-Y"，取起始编号参与比较即可覆盖两种写法。
        const std::string part = siblingList.substr(pos, partEnd - pos);
        const size_t dashPos = part.find('-');
        const std::string firstNumber = part.substr(0, dashPos);
        if (IsAllDigits(firstNumber)) {
            const int cpuId = std::atoi(firstNumber.c_str());
            if (cpuId >= 0 && (minCpuId < 0 || cpuId < minCpuId)) {
                minCpuId = cpuId;
            }
        }
        if (commaPos == std::string::npos) {
            break;
        }
        pos = commaPos + 1;
    }
    return minCpuId;
}

#ifdef __linux__
/** 获取当前进程亲和性允许的 CPU 列表（升序）；读取失败返回空列表。 */
inline std::vector<int> GetProcessAllowedCpuList()
{
    cpu_set_t allowedSet;
    CPU_ZERO(&allowedSet);
    if (sched_getaffinity(0, sizeof(allowedSet), &allowedSet) != 0) {
        return {};
    }
    std::vector<int> cpuList;
    for (int cpuId = 0; cpuId < CPU_SETSIZE; ++cpuId) {
        if (CPU_ISSET(cpuId, &allowedSet) != 0) {
            cpuList.push_back(cpuId);
        }
    }
    return cpuList;
}

/**
 * @brief 读取 CPU 的超线程兄弟列表，取其中最小 CPU 号作为物理核分组键。
 *
 * sysfs 不存在或不可读（部分容器环境）时退化为 cpuId 自身，即不感知 SMT、
 * 按逻辑 CPU 顺序直接分配。
 *
 * @return 物理核分组键（兄弟列表中最小的 CPU 号）。
 */
inline int GetSiblingGroupKey(int cpuId)
{
    const std::string siblingsPath
        = "/sys/devices/system/cpu/cpu" + std::to_string(cpuId) + "/topology/thread_siblings_list";
    std::ifstream siblingsFile(siblingsPath);
    std::string siblingList;
    if (!(siblingsFile >> siblingList)) {
        return cpuId;
    }
    const int minSiblingCpu = ParseSiblingListMinCpu(siblingList);
    return (minSiblingCpu >= 0) ? minSiblingCpu : cpuId;
}

/**
 * @brief 构建绑核用的优选 CPU 列表：进程亲和集内的 CPU 按“物理核优先”重排。
 *
 * 每个物理核（SMT 兄弟组）的首个成员排在前面、其余兄弟垫后，线程按槽位
 * 依次取用即可保证先把不同物理核分给不同线程，物理核用尽才共享同一物理核。
 *
 * @return 优选 CPU 列表；进程亲和集为空或读取失败时返回空列表（调用方退化为不绑核）。
 */
inline std::vector<int> BuildPreferredBindCpuList()
{
    const std::vector<int> allowedCpuList = GetProcessAllowedCpuList();
    if (allowedCpuList.empty()) {
        return {};
    }
    std::set<int> seenGroupKey;
    std::vector<int> physicalFirst;
    std::vector<int> siblings;
    for (const int cpuId : allowedCpuList) {
        if (seenGroupKey.insert(GetSiblingGroupKey(cpuId)).second) {
            physicalFirst.push_back(cpuId);
        } else {
            siblings.push_back(cpuId);
        }
    }
    physicalFirst.insert(physicalFirst.end(), siblings.begin(), siblings.end());
    return physicalFirst;
}

/**
 * @brief 把当前线程绑定到指定 CPU。
 *
 * @return 绑定成功返回 true；cpuId 非法或绑定失败返回 false（只告警，不影响业务流程）。
 */
inline bool BindCurrentThreadToCpu(int cpuId)
{
    if (cpuId < 0 || cpuId >= CPU_SETSIZE) {
        return false;
    }
    cpu_set_t targetSet;
    CPU_ZERO(&targetSet);
    CPU_SET(cpuId, &targetSet);
    if (pthread_setaffinity_np(pthread_self(), sizeof(targetSet), &targetSet) != 0) {
        HCCL_VM_WARN("Failed to bind the current thread to cpu {}, running without core binding", cpuId);
        return false;
    }
    return true;
}
#else
inline std::vector<int> BuildPreferredBindCpuList() { return {}; }

inline bool BindCurrentThreadToCpu(int) { return false; }
#endif

/**
 * @brief RAII 主线程绑核器：构造时把当前线程临时绑定到指定 CPU，析构时恢复原亲和性。
 *
 * 工作线程退出即销毁、无需恢复；主线程还要继续跑后续阶段，因此用本类保证
 * 绑核只在并行区间生效。cpuId 无效或绑定失败时为空操作。
 */
class ScopedThreadCpuBinder {
public:
    explicit ScopedThreadCpuBinder(int cpuId)
    {
#ifdef __linux__
        if (cpuId < 0 || cpuId >= CPU_SETSIZE) {
            return;
        }
        cpu_set_t originalSet;
        CPU_ZERO(&originalSet);
        if (pthread_getaffinity_np(pthread_self(), sizeof(originalSet), &originalSet) != 0) {
            HCCL_VM_WARN("Failed to get the current thread affinity before binding to cpu {}", cpuId);
            return;
        }
        cpu_set_t targetSet;
        CPU_ZERO(&targetSet);
        CPU_SET(cpuId, &targetSet);
        if (pthread_setaffinity_np(pthread_self(), sizeof(targetSet), &targetSet) != 0) {
            HCCL_VM_WARN("Failed to bind the current thread to cpu {}", cpuId);
            return;
        }
        originalSet_ = originalSet;
        hasOriginalSet_ = true;
        bound_ = true;
#else
        (void)cpuId;
#endif
    }

    ~ScopedThreadCpuBinder()
    {
#ifdef __linux__
        if (!hasOriginalSet_) {
            return;
        }
        if (pthread_setaffinity_np(pthread_self(), sizeof(originalSet_), &originalSet_) != 0) {
            HCCL_VM_WARN("Failed to restore the thread affinity after core binding");
        }
#endif
    }

    ScopedThreadCpuBinder(const ScopedThreadCpuBinder&) = delete;
    ScopedThreadCpuBinder& operator=(const ScopedThreadCpuBinder&) = delete;

    /** 是否成功绑定（cpuId 无效、平台不支持或绑定失败返回 false）。 */
    bool IsBound() const { return bound_; }

private:
    bool bound_{false};
#ifdef __linux__
    cpu_set_t originalSet_{};
    bool hasOriginalSet_{false};
#endif
};

} // namespace HcclSim

#endif // THREAD_AFFINITY_H
