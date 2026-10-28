/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_BASE_COMM_RES_PER_DEV_H
#define HCOMM_BASE_COMM_RES_PER_DEV_H

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>
#include "hccl_common.h"
#include "hccl/hccl_types.h"
#include "resources/comm_engine_res/comm_engine_procs.h"
#include "resources/endpoint_pairs/sockets/socket_process.h"
#include "resources/endpoints/net_dev/net_dev_proc.h"
#include "resources/endpoints/mgr/server_socket_mgr_proc.h"
#include "resources/endpoints/mgr/endpoint_ctx_mgr.h"

namespace hcomm {

// 设备级 legacy 单例依赖清单（唯一一份）：树根构造期以备份槽预热（锁死析构序），
// 设备子树构造期按本槽再调（UT/独立构造场景生效，正常路径 no-op）——README R8
void AnchorDeviceSingletons(uint32_t devPhyId);

/**
 * @brief 设备层节点：每张物理卡（devPhyId）一棵资源子树。
 *        首访懒建、构造即完成设备级锚定；成员声明序 = 析构契约（依据见 README R3/R4/R8）。
 */
class BaseCommResPerDev {
public:
    uint32_t GetDevPhyId() const { return devPhyId_; }

    EndpointCtxMgr& GetEndpointCtxMgr() { return endpointCtxMgr_; }

    // 清 socket 表与 EndpointCtxMgr 残留（幂等）；RESET 跳过（还 socket/强释 ctx 触设备）
    void DeInit(ExitProfile profile = ExitProfile::EXIT_PROFILE_GRACEFUL)
    {
        if (profile == ExitProfile::EXIT_PROFILE_RESET) {
            return;
        }
        socketProcess_.DeInit();
        endpointCtxMgr_.DeInit();
    }

    CommEngineMgrProc& GetCommEngineMgrProc() { return commEngineMgrProc_; }
    ServerSocketMgrProc& GetServerSocketMgrProc() { return serverSocketMgrProc_; }
    SocketProcess& GetSocketProcess() { return socketProcess_; }
    NetDevProc& GetNetDevProc() { return netDevProc_; }

public:
    // skipAnchor=true 仅用于 DeInit 后拒建的兜底实例
    BaseCommResPerDev(uint32_t devPhyId, bool skipAnchor);
    ~BaseCommResPerDev() = default;

private:
    BaseCommResPerDev(const BaseCommResPerDev&) = delete;
    BaseCommResPerDev& operator=(const BaseCommResPerDev&) = delete;

    // 声明顺序 = 析构契约反序（依据见 README R3）
    NetDevProc netDevProc_;
    ServerSocketMgrProc serverSocketMgrProc_;
    SocketProcess socketProcess_; // 析构经 GetSocketMgr 还 socket，须先于 serverSocketMgrProc_ 死
    CommEngineMgrProc commEngineMgrProc_;
    EndpointCtxMgr endpointCtxMgr_;
    uint32_t devPhyId_{0};
};

/**
 * @brief 设备子树存取器：有界数组 + 原子发布（稳态读无锁）+ 退役区 + 拒建兜底实例。
 *        锁与拒建闸声明最先 = 最晚死，reserve_ 晚于 retired_ 死：退役区析构期回落路径全程可用（依据见 README R3/R4）。
 *        发布后对象地址不变、随树存活；首建/摘表持内部锁（R2 死锁红线语义不变）。
 */
class BaseCommResPerDevMgr {
private:
    // 声明在最前：成员函数返回类型/成员数组类型引用它（返回类型非类完整上下文，必须先声明可见）
    static constexpr uint32_t SLOT_NUM = MAX_MODULE_DEVICE_NUM + 1;

public:
    BaseCommResPerDevMgr() = default;

    // 快路径无锁读已发布槽位；慢路径持锁首建（构造即锚定）+ 锁内复查拒建闸，置位后回落兜底实例
    BaseCommResPerDev& FindOrCreate(uint32_t devPhyId);

    // DeInit 动作链迭代用：锁内拷贝非空槽位快照，与并发插入互斥
    std::vector<BaseCommResPerDev*> SnapshotActive();

    // 拒建闸：DeInit 起始置位，此后 FindOrCreate 慢路径拒建、回落兜底实例
    void MarkDeinited() { deinited_.store(true); }

    // 摘表：锁内清发布槽 + 移交退役区（DeInit 管动作、析构管内存）
    void RetireAll();

private:
    // 锁与拒建闸声明最前 = 最晚死：退役区析构期经窄访问器回落 reserve_ 的慢路径要持锁读闸（README R3）
    std::mutex mtx_;                    // 仅护首建与摘表；稳态读不碰
    std::atomic<bool> deinited_{false}; // 拒建闸，FindOrCreate 慢路径锁内复查
    // 拒建兜底实例：跳过锚定、从不 Init、不入表。声明序三层约束（评审曾建议后移，已驳回，勿再动）：
    // 1) 必须晚于 retired_ 死——退役区析构期经窄访问器回落要拿到存活的 reserve_；
    // 2) 自身析构链不重入 FindOrCreate——从不 Init、子节点皆默认/空表析构，无访问器回调；
    // 3) 即使重入也不读已析构元素——deinited_ 闸在位时慢路径直接返回 reserve_，不碰 active_ 内容。
    BaseCommResPerDev reserve_{UINT32_MAX, true};
    std::array<std::unique_ptr<BaseCommResPerDev>, SLOT_NUM> active_;   // 所有权，仅持锁写
    std::array<std::atomic<BaseCommResPerDev*>, SLOT_NUM> published_{}; // 无锁读视图
    std::array<std::unique_ptr<BaseCommResPerDev>, SLOT_NUM> retired_;
};

} // namespace hcomm

#endif // HCOMM_BASE_COMM_RES_PER_DEV_H
