/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIM_DEVICE_SPEC_RANK_ADDRESS_SPACE_H
#define SIM_DEVICE_SPEC_RANK_ADDRESS_SPACE_H

#include "device_spec/device_spec_types.h"

namespace sim {
namespace devspec {

    /**
     * @brief rank / CCU die / user_mem 的地址计算（纯计算，无 IO、无副作用）。
     *
     * 地址公式（设计文档 §4.4.1）：
     *   rankBase(r)      = rankBase + r × rankStride
     *   ccuDieBase(r, d) = rankBase(r) + ccuWindowOffset + ccuDieOffsets[d]
     *   userMemVa(r, seg, off) = rankBase(r) + userMemSegs[seg].offset + off
     *
     * ★ 本类【不提供】UserMemVaFromLinear(rankIdx, linear)：单一全局线性水位会让分配块
     *   跨越 ccu 空洞（反例见设计文档 §4.5.7.2）。分配必须由调用方按段遍历（first-fit），
     *   见 DeviceMemoryManager::AllocVirMem。
     *
     * 生命周期：持有的 AddrSpace 引用来自 DeviceSpecRegistry 内的 DeviceSpec，
     * 后者加载后只读且进程级常驻，故引用安全。
     */
    class RankAddressSpace {
    public:
        explicit RankAddressSpace(const AddrSpace& addr);

        /// rank 窗口起点。rankIdx 越界或溢出时返回 0。
        uint64_t RankBase(uint32_t rankIdx) const;
        uint64_t RankWindowSize() const { return m_addr.rankWindowSize; }
        uint64_t RankStride() const { return m_addr.rankStride; }

        /// ★ 本类【不提供】rank 槽位上界：集群 rank 数由拓扑决定，模拟器不设人为上限
        ///   （原 `MaxRankSlots()` 已删除）。rankIdx 的合法性由 `RankBase()` 的 uint64
        ///   溢出防护兜底（溢出返回 0，调用方按失败处理），47-bit 位宽只作诊断告警。

        /// 47-bit 规范用户 VA 下"窗口起点不越界"的最大 rank 数（诊断用，非硬约束）。
        /// 超过此值的 rank 其 VA 仍是有效的模拟 token（不被 host 解引用），只是超出
        /// x86-64 四级页表的规范用户地址范围。加载器校验 7 用它输出 WARN（R4）。
        uint32_t Va47BitRankLimit() const;

        // ---- CCU ----
        uint64_t CcuWindowOffset() const { return m_addr.ccuWindowOffset; }
        uint64_t CcuWindowSize() const { return m_addr.ccuWindowSize; }
        uint32_t DieNum() const { return static_cast<uint32_t>(m_addr.ccuDieOffsets.size()); }

        /// CCU 某 die 的资源空间基址。dieId 越界或溢出时返回 0。
        uint64_t CcuDieBase(uint32_t rankIdx, uint32_t dieId) const;

        /// 判断 va 是否落在某个 rank 的 CCU 窗口内；命中时回填 rankIdx / dieId。
        bool IsInCcuWindow(uint64_t va, uint32_t& rankIdx, uint32_t& dieId) const;

        // ---- user_mem（多段；分配用 per-segment first-fit）----
        size_t UserMemSegCount() const { return m_addr.userMemSegs.size(); }
        uint64_t UserMemSegOffset(size_t segIdx) const;
        uint64_t UserMemSegSize(size_t segIdx) const;
        /// Σ user_mem 段大小（A5/A6 均为 0xFF80000000 = 1022 GiB）。仅用于报告/日志。
        uint64_t UserMemTotal() const { return m_addr.userMemTotal; }
        /// va 是否落在指定 rank 的某个 user_mem 段内。
        bool IsUserMemVa(uint64_t va, uint32_t rankIdx) const;

        /// rankIdx := Device.physical_id - 1（设计文档 §4.4.4 / D5，与现行 AllocVirMem 同源）。
        /// devKey 为 0 时返回 false —— GetCurrDeviceId() 在 context/device 查不到时返回 0，
        /// 直接做 devKey-1 会 uint64 下溢（latent bug B1）。
        static bool RankIdxOfDeviceKey(uint64_t devKey, uint32_t& rankIdx);

    private:
        const AddrSpace& m_addr;
    };

} // namespace devspec
} // namespace sim

#endif // SIM_DEVICE_SPEC_RANK_ADDRESS_SPACE_H
