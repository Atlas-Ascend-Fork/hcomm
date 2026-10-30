/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_HANDLE_REGISTRY_PROCS_H
#define HCOMM_HANDLE_REGISTRY_PROCS_H

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include "hccl/hccl_types.h"
#include "hcomm_res_defs.h"
#include "common/exit_profile.h"

namespace hccl {
class Thread;
} // namespace hccl

namespace hcomm {

/**
 * @brief device 句柄 → host 句柄翻译键。
 */
struct DeviceThreadKey {
    int32_t deviceId;
    ThreadHandle handle;

    bool operator==(const DeviceThreadKey& other) const { return deviceId == other.deviceId && handle == other.handle; }
};

struct DeviceThreadKeyHash {
    std::size_t operator()(const DeviceThreadKey& key) const
    {
        return std::hash<int32_t>()(key.deviceId) ^ (std::hash<ThreadHandle>()(key.handle) << 1);
    }
};

class Channel;

/**
 * @brief device channel 句柄 → host 句柄翻译键。
 */
struct DeviceChannelKey {
    int32_t deviceId;
    ChannelHandle handle;

    bool operator==(const DeviceChannelKey& other) const
    {
        return deviceId == other.deviceId && handle == other.handle;
    }
};

struct DeviceChannelKeyHash {
    std::size_t operator()(const DeviceChannelKey& key) const
    {
        return std::hash<int32_t>()(key.deviceId) ^ (std::hash<ChannelHandle>()(key.handle) << 1);
    }
};

/**
 * @brief Channel 句柄注册表（进程级）。
 *
 * Channel 句柄三表与护表单锁，单锁护三表。
 * 句柄全局唯一不按卡分，故挂树根而非设备层。
 */
class ChannelHandleRegistryProc {
public:
    // 清 channel 表残留：摘表后锁外析构（对齐 EndpointHandleMgr::DeInit 纪律）；
    // RESET 跳过：channel 析构经窄访问器归还 socket/notify 等设备资源，设备不可用时不能执行
    void DeInit(ExitProfile profile = ExitProfile::EXIT_PROFILE_GRACEFUL);

    // RESET 兜底告警用：句柄表是否有残留
    bool HasLeftover()
    {
        std::lock_guard<std::mutex> lock(channelMapMtx_);
        return !channelMap_.empty();
    }

    std::unordered_map<ChannelHandle, std::shared_ptr<Channel>>& ChannelMap() { return channelMap_; }
    std::unordered_map<DeviceChannelKey, ChannelHandle, DeviceChannelKeyHash>& ChannelD2HMap()
    {
        return channelD2HMap_;
    }
    std::unordered_map<DeviceChannelKey, ChannelHandle, DeviceChannelKeyHash>& ChannelH2DMap()
    {
        return channelH2DMap_;
    }
    std::unordered_map<DeviceChannelKey, ChannelHandle, DeviceChannelKeyHash>& ChannelDestroyHandleMap()
    {
        return channelDestroyHandleMap_;
    }
    std::mutex& ChannelMapMtx() { return channelMapMtx_; }

public:
    ChannelHandleRegistryProc() = default;
    ~ChannelHandleRegistryProc() = default;

private:
    ChannelHandleRegistryProc(const ChannelHandleRegistryProc&) = delete;
    ChannelHandleRegistryProc& operator=(const ChannelHandleRegistryProc&) = delete;

    // 锁声明最前 = 最后死（README R3）：三表析构期（channel 析构经访问器加锁）锁必须仍可用
    std::mutex channelMapMtx_;                                               // 单锁护三表
    std::unordered_map<ChannelHandle, std::shared_ptr<Channel>> channelMap_; // 来源：ChannelProcess
    std::unordered_map<DeviceChannelKey, ChannelHandle, DeviceChannelKeyHash> channelD2HMap_;
    std::unordered_map<DeviceChannelKey, ChannelHandle, DeviceChannelKeyHash> channelH2DMap_;
    std::unordered_map<DeviceChannelKey, ChannelHandle, DeviceChannelKeyHash> channelDestroyHandleMap_;
};

/**
 * @brief Thread 句柄注册表（进程级）。
 *
 * 引擎线程（thread.cc）的句柄登记 + D2H 翻译表 + order 控制流线程集合。
 * 析构：先于设备子树死（HandleRegistryMgrProc 内最后声明 = 最先死），也先于 ChannelHandleRegistryProc 死。
 */
class ThreadHandleRegistryProc {
public:
    void DeInit() {}

    std::unordered_map<ThreadHandle, std::shared_ptr<hccl::Thread>>& EngineThreadMap() { return engineThreadMap_; }
    std::unordered_map<DeviceThreadKey, ThreadHandle, DeviceThreadKeyHash>& EngineThreadD2HMap()
    {
        return engineThreadD2HMap_;
    }
    std::shared_mutex& EngineThreadMapMtx() { return engineThreadMapMtx_; }
    std::mutex& OrderCtrlMtx() { return orderCtrlMtx_; }
    std::unordered_set<ThreadHandle>& OrderCtrlThreads() { return orderCtrlThreads_; }

public:
    ThreadHandleRegistryProc() = default;
    ~ThreadHandleRegistryProc() = default;

private:
    ThreadHandleRegistryProc(const ThreadHandleRegistryProc&) = delete;
    ThreadHandleRegistryProc& operator=(const ThreadHandleRegistryProc&) = delete;

    // 锁声明最前 = 最后死（README R3）：map 析构期若触发 ~Thread 回调访问器，锁须仍可用
    std::shared_mutex engineThreadMapMtx_; // 写路径 lock_guard 独占、读路径 shared_lock 共享
    std::unordered_map<ThreadHandle, std::shared_ptr<hccl::Thread>> engineThreadMap_; // 来源：thread.cc
    std::unordered_map<DeviceThreadKey, ThreadHandle, DeviceThreadKeyHash> engineThreadD2HMap_;
    // order 控制流线程集合：key 同为 host 线程句柄；独立锁（不并入 engineThreadMapMtx_，嵌套语义不变）
    std::mutex orderCtrlMtx_;
    std::unordered_set<ThreadHandle> orderCtrlThreads_;
};

/**
 * @brief 句柄注册表聚合节点（进程级）。
 *
 * 成员声明顺序 = 析构契约反序（先声明的后死）。
 */
class HandleRegistryMgrProc {
public:
    void DeInit(ExitProfile profile = ExitProfile::EXIT_PROFILE_GRACEFUL)
    {
        threadHandleRegistryProc_.DeInit();
        channelHandleRegistryProc_.DeInit(profile);
    }

    ChannelHandleRegistryProc& GetChannelHandleRegistryProc() { return channelHandleRegistryProc_; }
    ThreadHandleRegistryProc& GetThreadHandleRegistryProc() { return threadHandleRegistryProc_; }

public:
    HandleRegistryMgrProc() = default;
    ~HandleRegistryMgrProc() = default;

private:
    HandleRegistryMgrProc(const HandleRegistryMgrProc&) = delete;
    HandleRegistryMgrProc& operator=(const HandleRegistryMgrProc&) = delete;

    // 声明顺序 = 析构契约反向：channel 先声明后死（[2]），thread 后声明先死（[1] 停 Thread 最先）
    ChannelHandleRegistryProc channelHandleRegistryProc_; // [2] Channel 句柄表，后死
    ThreadHandleRegistryProc threadHandleRegistryProc_;   // [1] Thread 句柄表，先死
};

} // namespace hcomm

#endif // HCOMM_HANDLE_REGISTRY_PROCS_H
