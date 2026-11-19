/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DEV_EP_MEM_TABLE_H
#define DEV_EP_MEM_TABLE_H

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include "endpoint_pair/channel/channel_param_pub.h" // DEV_EP_MEM_TABLE_MAGIC_WORD / VERSION / LocUbBufLite

namespace Hccl {

/**
 * device 侧 endpoint 本地内存活账本表（AICPU_TS）。
 * 生命周期：endpoint 创建 kernel new（RunAicpuEpMemTableInit）→ endpoint 销毁 kernel delete
 *           （RunAicpuEpMemTableDestroy）。host 经 RunAicpuEpMemTableUpdate 运行时增删；
 *           channel 建链经 uniqueIdV2 携带表指针，收发时 BuildLocRmaBufferLite 按址查询（唯一数据来源）。
 * 作用域：per-endpoint 一张表，仅 UB 三类 device endpoint（Urma/Uboe/UbRtp）持有；
 *         与 host 侧实例归属账本（memRecordsPublished_）语义对齐，同 ctx 不同 endpoint 实例互不可见。
 * 并发：不可变快照发布——写者（Update/Remove）经 wMtx_ 串行，持锁拷贝当前代、私有副本改动后
 *       std::atomic_store 发布（构造先于发布，读者见到的代必然完整，发布/撕裂/旋转竞态整体不存在）；
 *       读者（Find/FindFirst，ExecuteBatchTransfer 逐 desc 热路径）std::atomic_load 取快照后遍历，
 *       每查询 ~4 次 RMW（libstdc++ 桶锁+控制块计数，多线程进程非 lock-free，为自动回收付的已接受代价）。
 *       旧代由在途读者的 shared_ptr 副本自动保活回收；写路径每次 O(n) 拷贝（写瞬间峰值 ~2x 表大小）。
 */
class DevEpMemTable {
public:
    // ABI 头（首成员）：kernel 下发侧与查找侧校验，防悬垂/错指针写花 AICPU 堆
    struct AbiHeader {
        uint32_t magic{};
        uint32_t version{};
        uint32_t entryNum{};
        uint32_t rsv{};
    };

    // (addr,size) 复合键：先 addr 后 size 字典序（与 host mgr 注册树同构）；
    // 本地内联定义保持本头仅依赖 channel_param_pub.h，勿引外部 buffer_key.h
    struct EpMemKey {
        uint64_t addr{};
        uint64_t size{};
        bool operator<(const EpMemKey& other) const
        {
            return addr < other.addr || (addr == other.addr && size < other.size);
        }
    };

    // 表条目：info 为 DTO 投影；ref 为本实例 Reg/Unreg 计数（同 (addr,size) 重复注册=互为别名，
    // 聚合计数）。单一写入器（Reg 路径 kernel Update/Remove）下条目创建即 ref=1，不存在无账本条目
    struct EpMemEntry {
        LocUbBufLite info{};
        uint32_t ref{1};
    };

    DevEpMemTable();

    // 析构即失效哨兵：清零 abi，delete 之后任何悬垂表指针的 CheckAbi 都必须失败
    ~DevEpMemTable();

    // Reg 路径：键存在 ref++（覆盖 info），不存在插入 {info, ref=1}
    HcclResult Update(const LocUbBufLite& entry);
    // Unreg 路径：ref-- 归零删条目；键不存在返回 NOT_FOUND（kernel 侧降级放行，幂等注销良性）
    HcclResult Remove(uint64_t addr, uint64_t size);
    // 按址区间查找：upper_bound + 回退 + 区间包含判定（与 BuildLocRmaBufferLite 原算法一致）
    bool Find(uintptr_t addr, size_t size, LocUbBufLite& out) const;
    // 未命中时的既有回退语义：借用首条（保持改造前行为兼容）
    bool FindFirst(LocUbBufLite& out) const;
    // 哨兵校验：magic/version 不符返回 false（表已销毁/错指针时快速失败）；
    // "已销毁"由析构清零保证，否则 delete 后堆块未复用时哨兵仍会误放行
    bool CheckAbi() const;

    AbiHeader abi{};

private:
    using Table = std::map<EpMemKey, EpMemEntry>;
    using Snapshot = const Table;
    // 共用静态空快照：首表构造一次性分配，失败经 kernel 入口 EXCEPTION_CATCH 转错误码；magic static 一次性安全初始化
    static std::shared_ptr<Snapshot> EmptySnapshot();

    std::mutex wMtx_;                       // 仅写-写串行（读者永不上锁）
    std::shared_ptr<Snapshot> published_{}; // 发布牌：读者唯一触碰点（std::atomic_load/store 访问）
};
} // namespace Hccl
#endif // DEV_EP_MEM_TABLE_H
