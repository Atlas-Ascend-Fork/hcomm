/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dev_ep_mem_table.h"
#include "log.h"

namespace Hccl {
// 区间包含判定：查询窗口 [addr, addr+size) 须被注册窗口 [base, base+len) 完全包含（含溢出防护）。
// 与 range_utils.h IsRangeInclude / Buffer::Contains 同语义，内联避免头依赖。
static bool IsRangeIncluded(uint64_t base, uint64_t len, uint64_t addr, uint64_t size)
{
    // 与 range_utils.h IsRangeInclude 一致：任一窗口终点（base+len / addr+size）溢出即非法，判不包含
    if (len > UINT64_MAX - base || size > UINT64_MAX - addr) {
        return false;
    }
    return (base <= addr) && (addr - base + size <= len);
}

DevEpMemTable::DevEpMemTable()
{
    abi.magic = DEV_EP_MEM_TABLE_MAGIC_WORD;
    abi.version = DEV_EP_MEM_TABLE_VERSION;
    // 进程内首表构造存在一次性静态空快照分配（此后所有表零分配、读者永见非空）；首建分配失败经
    // kernel 入口 EXCEPTION_CATCH 转 HCCL_E_MEMORY（nothrow-new 判空 + 异常转错误码双兜底）
    published_ = EmptySnapshot();
}

// 全部表共用的静态空快照：magic static 保证多线程一次性安全初始化
std::shared_ptr<DevEpMemTable::Snapshot> DevEpMemTable::EmptySnapshot()
{
    static const auto empty = std::shared_ptr<Snapshot>(std::make_shared<Table>());
    return empty;
}

DevEpMemTable::~DevEpMemTable()
{
    // kernel delete（RunAicpuEpMemTableDestroy）后堆块内容尚存（复用前 magic 仍为 'MEPT'），
    // 悬垂指针的 CheckAbi 会误放行并读写已释放的表对象（map/成员）；析构时清零哨兵封堵该窗口
    abi.magic = 0;
    abi.version = 0;
}

bool DevEpMemTable::CheckAbi() const
{
    return (abi.magic == DEV_EP_MEM_TABLE_MAGIC_WORD) && (abi.version == DEV_EP_MEM_TABLE_VERSION);
}

HcclResult DevEpMemTable::Update(const LocUbBufLite& entry)
{
    if (entry.addr == 0 || entry.size == 0) {
        HCCL_ERROR("[DevEpMemTable][%s] invalid entry addr[0x%llx] size[0x%llx]", __func__, entry.addr, entry.size);
        return HCCL_E_PARA;
    }
    // 快照语义：持锁拷贝当前代 → 私有副本改动（无读者可见的中间态）→ 构造完成后 atomic_store 发布。
    // 拷贝可抛 bad_alloc——就地转错误码（kernel 入口无捕获，逃逸即 AICPU terminate）；
    // 失败时 published_ 未动（旧代仍当前代），表状态一致
    try {
        std::lock_guard<std::mutex> lock(wMtx_);
        auto next = std::make_shared<Table>(*published_);
        EpMemKey key{entry.addr, entry.size};
        auto it = next->find(key);
        if (it != next->end()) {
            // 同 (addr,size) 重复注册（mgr 侧必为别名、token 同源）：计数聚合，info 取最新
            it->second.ref++;
            it->second.info = entry;
        } else {
            next->emplace(key, EpMemEntry{entry, 1});
        }
        abi.entryNum = static_cast<uint32_t>(next->size());
        std::atomic_store(&published_, std::shared_ptr<Snapshot>(std::move(next)));
    } catch (const std::exception& e) {
        HCCL_ERROR("[DevEpMemTable][%s] snapshot copy exception caught:%s", __func__, e.what());
        return HCCL_E_MEMORY;
    }
    return HCCL_SUCCESS;
}

HcclResult DevEpMemTable::Remove(uint64_t addr, uint64_t size)
{
    std::lock_guard<std::mutex> lock(wMtx_);
    // 幂等/乱序注销的目标态已在当前代达成：免 O(n) 拷贝直接返回（良性 miss 主路径）
    if (published_->find(EpMemKey{addr, size}) == published_->end()) {
        // 键不存在：幂等/乱序注销的目标态（表内无该条目）已达成，kernel 侧降级放行
        HCCL_WARNING("[DevEpMemTable][%s] addr[0x%llx] size[0x%llx] not in table", __func__, addr, size);
        return HCCL_E_NOT_FOUND;
    }
    // 快照语义：拷贝当前代 → 私有副本 ref-- 归零删条目 → atomic_store 发布（异常转错误码，旧代不动）
    try {
        auto next = std::make_shared<Table>(*published_);
        auto it = next->find(EpMemKey{addr, size});
        if (--it->second.ref == 0) {
            next->erase(it); // 严格同步：本实例已无此注册
        }
        abi.entryNum = static_cast<uint32_t>(next->size());
        std::atomic_store(&published_, std::shared_ptr<Snapshot>(std::move(next)));
    } catch (const std::exception& e) {
        HCCL_ERROR("[DevEpMemTable][%s] snapshot copy exception caught:%s", __func__, e.what());
        return HCCL_E_MEMORY;
    }
    return HCCL_SUCCESS;
}

bool DevEpMemTable::Find(uintptr_t addr, size_t size, LocUbBufLite& out) const
{
    std::shared_ptr<Snapshot> snap = std::atomic_load(&published_);
    if (snap->empty()) {
        return false;
    }
    // (addr, UINT64_MAX) 为上界哨兵键：upper_bound 之后所有 base<=addr 的条目均在其左侧，
    // 全量回退保证嵌套/同基址多键下包含判定完备（算法骨架与原实现一致）
    auto it = snap->upper_bound(EpMemKey{static_cast<uint64_t>(addr), UINT64_MAX});
    while (it != snap->begin()) {
        --it;
        if (IsRangeIncluded(it->first.addr, it->first.size, static_cast<uint64_t>(addr), size)) {
            out = it->second.info;
            return true;
        }
    }
    return false;
}

bool DevEpMemTable::FindFirst(LocUbBufLite& out) const
{
    std::shared_ptr<Snapshot> snap = std::atomic_load(&published_);
    if (snap->empty()) {
        return false;
    }
    out = snap->begin()->second.info;
    return true;
}
} // namespace Hccl
