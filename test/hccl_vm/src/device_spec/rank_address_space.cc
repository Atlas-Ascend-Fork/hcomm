/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "device_spec/rank_address_space.h"

#include <limits>

#include "sim_log.h"

namespace sim {
namespace devspec {
    namespace {
        /// 47-bit 规范用户 VA 上界（x86-64 四级页表 / aarch64 48-bit VA 的保守取值）。
        /// 超出后模拟设备 VA 虽仍可作为 token 使用，但不可被 host 侧解释，故仅告警（R4）。
        constexpr uint64_t VA_47BIT_LIMIT = 1ULL << 47;
    } // namespace

    RankAddressSpace::RankAddressSpace(const AddrSpace& addr) : m_addr(addr) {}

    uint64_t RankAddressSpace::RankBase(uint32_t rankIdx) const
    {
        if (m_addr.rankStride == 0) {
            return 0;
        }
        // 溢出防护：rank 规模已无上界（原 MaxRankSlots/HCCL_VM_MAX_RANKS 已删除），本检查是
        // rankIdx 的唯一硬约束，故必须自身不溢出 —— 用除法求"可容纳的最大 rankIdx"，
        // 不能写成 `rankBase > UINT64_MAX - rankIdx * rankStride`（乘法先回绕则判不出溢出）。
        const uint64_t maxRankIdx = (std::numeric_limits<uint64_t>::max() - m_addr.rankBase) / m_addr.rankStride;
        if (static_cast<uint64_t>(rankIdx) > maxRankIdx) {
            HCCL_VM_ERROR(
                "RankBase overflow: rankIdx={} exceeds max addressable rankIdx={} (rankBase={:#x} rankStride={:#x})",
                rankIdx, maxRankIdx, m_addr.rankBase, m_addr.rankStride);
            return 0;
        }
        return m_addr.rankBase + static_cast<uint64_t>(rankIdx) * m_addr.rankStride;
    }

    uint32_t RankAddressSpace::Va47BitRankLimit() const
    {
        if (m_addr.rankStride == 0 || m_addr.rankBase >= VA_47BIT_LIMIT) {
            return 0;
        }
        const uint64_t budget = (VA_47BIT_LIMIT - m_addr.rankBase) / m_addr.rankStride;
        if (budget > std::numeric_limits<uint32_t>::max()) {
            return std::numeric_limits<uint32_t>::max();
        }
        return static_cast<uint32_t>(budget);
    }

    uint64_t RankAddressSpace::CcuDieBase(uint32_t rankIdx, uint32_t dieId) const
    {
        if (dieId >= m_addr.ccuDieOffsets.size()) {
            HCCL_VM_ERROR("CcuDieBase: dieId {} out of range (die_num {})", dieId, m_addr.ccuDieOffsets.size());
            return 0;
        }
        const uint64_t rb = RankBase(rankIdx);
        if (rb == 0) {
            return 0;
        }
        const uint64_t dieOff = m_addr.ccuDieOffsets[dieId];
        if (m_addr.ccuWindowOffset > std::numeric_limits<uint64_t>::max() - rb
            || dieOff > std::numeric_limits<uint64_t>::max() - rb - m_addr.ccuWindowOffset) {
            HCCL_VM_ERROR("CcuDieBase overflow: rankIdx={} dieId={}", rankIdx, dieId);
            return 0;
        }
        return rb + m_addr.ccuWindowOffset + dieOff;
    }

    bool RankAddressSpace::IsInCcuWindow(uint64_t va, uint32_t& rankIdx, uint32_t& dieId) const
    {
        rankIdx = 0;
        dieId = 0;
        if (m_addr.rankStride == 0 || m_addr.ccuWindowSize == 0 || m_addr.ccuDieOffsets.empty()) {
            return false;
        }
        if (va < m_addr.rankBase) {
            return false;
        }
        const uint64_t rel = va - m_addr.rankBase;
        const uint64_t r = rel / m_addr.rankStride;
        if (r >= m_addr.rankStride /* 防御 */ || r > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
        const uint64_t offInRank = rel - r * m_addr.rankStride;
        if (offInRank < m_addr.ccuWindowOffset || offInRank >= m_addr.ccuWindowOffset + m_addr.ccuWindowSize) {
            return false;
        }
        const uint64_t offInCcu = offInRank - m_addr.ccuWindowOffset;
        for (size_t d = 0; d < m_addr.ccuDieOffsets.size(); ++d) {
            const uint64_t base = m_addr.ccuDieOffsets[d];
            if (offInCcu >= base && offInCcu < base + m_addr.ccuDieSize) {
                rankIdx = static_cast<uint32_t>(r);
                dieId = static_cast<uint32_t>(d);
                return true;
            }
        }
        // 落在 ccu 窗口内但不在任何 die 窗口（die 之间的间隙）：仍算 ccu 窗口，dieId 取最后一个
        rankIdx = static_cast<uint32_t>(r);
        dieId = static_cast<uint32_t>(m_addr.ccuDieOffsets.size() - 1);
        return true;
    }

    uint64_t RankAddressSpace::UserMemSegOffset(size_t segIdx) const
    {
        return (segIdx < m_addr.userMemSegs.size()) ? m_addr.userMemSegs[segIdx].offset : 0;
    }

    uint64_t RankAddressSpace::UserMemSegSize(size_t segIdx) const
    {
        return (segIdx < m_addr.userMemSegs.size()) ? m_addr.userMemSegs[segIdx].size : 0;
    }

    bool RankAddressSpace::IsUserMemVa(uint64_t va, uint32_t rankIdx) const
    {
        const uint64_t rb = RankBase(rankIdx);
        if (rb == 0 || va < rb) {
            return false;
        }
        const uint64_t off = va - rb;
        if (off >= m_addr.rankWindowSize) {
            return false;
        }
        for (const auto& seg : m_addr.userMemSegs) {
            if (off >= seg.offset && off < seg.offset + seg.size) {
                return true;
            }
        }
        return false;
    }

    bool RankAddressSpace::RankIdxOfDeviceKey(uint64_t devKey, uint32_t& rankIdx)
    {
        // ★ latent bug B1：GetCurrDeviceId() 在 context/device 查不到时返回 0
        //   （db_sim_runner_ops.cc:161,167）。若直接做 devKey-1，uint64 会下溢为 UINT64_MAX，
        //   rankBase 随之回绕成垃圾值。此处显式拦截。
        if (devKey == 0) {
            return false;
        }
        if (devKey - 1 > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
        rankIdx = static_cast<uint32_t>(devKey - 1);
        return true;
    }

} // namespace devspec
} // namespace sim
