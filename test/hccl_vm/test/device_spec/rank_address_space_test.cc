/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>

#include "gtest/gtest.h"

#include "device_spec/device_spec_registry.h"
#include "device_spec/rank_address_space.h"
#include "device_spec_yaml_loader.h"
#include "sim_capacity_limits.h"

namespace {

using sim::devspec::DeviceSpec;
using sim::devspec::LoadDeviceSpecYaml;
using sim::devspec::RankAddressSpace;

const std::string kDir = HCCLVM_DEVICE_SPEC_TEST_DIR;

/// 需求 3.1/3.2 的黄金值（设计文档 §4.4.2）
constexpr uint64_t RANK0_BASE = 0x8000000000ULL;
constexpr uint64_t RANK1_BASE = 0x18000000000ULL;
constexpr uint64_t RANK0_DIE0_CCU = 0x8e00000000ULL;
constexpr uint64_t RANK0_DIE1_CCU = 0x8e40000000ULL;
constexpr uint64_t RANK1_DIE0_CCU = 0x18e00000000ULL;
constexpr uint64_t RANK1_DIE1_CCU = 0x18e40000000ULL;
constexpr uint64_t RANK_STRIDE = 0x10000000000ULL;   // 1 TiB
constexpr uint64_t CCU_WIN_OFF = 0xE00000000ULL;     // 56 GiB
constexpr uint64_t CCU_WIN_SIZE = 0x80000000ULL;     // 2 GiB
constexpr uint64_t USER_MEM_TOTAL = 0xFF80000000ULL; // 1022 GiB

class RankAddrSpaceTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        std::string err;
        ASSERT_TRUE(LoadDeviceSpecYaml(kDir + "/device_spec_yaml_demo_ascend960.yaml", m_spec, err)) << err;
        m_ras = std::make_unique<RankAddressSpace>(m_spec.addr);
    }
    DeviceSpec m_spec;
    std::unique_ptr<RankAddressSpace> m_ras;
};

TEST_F(RankAddrSpaceTest, GoldenRankBase)
{
    EXPECT_EQ(m_ras->RankBase(0), RANK0_BASE);
    EXPECT_EQ(m_ras->RankBase(1), RANK1_BASE);
    EXPECT_EQ(m_ras->RankBase(2), RANK0_BASE + 2 * RANK_STRIDE);
    EXPECT_EQ(m_ras->RankWindowSize(), RANK_STRIDE);
    EXPECT_EQ(m_ras->RankStride(), RANK_STRIDE);
}

TEST_F(RankAddrSpaceTest, GoldenCcuDieBase)
{
    EXPECT_EQ(m_ras->CcuDieBase(0, 0), RANK0_DIE0_CCU);
    EXPECT_EQ(m_ras->CcuDieBase(0, 1), RANK0_DIE1_CCU);
    EXPECT_EQ(m_ras->CcuDieBase(1, 0), RANK1_DIE0_CCU);
    EXPECT_EQ(m_ras->CcuDieBase(1, 1), RANK1_DIE1_CCU);
    // per-rank 唯一（目标 G3）
    EXPECT_NE(m_ras->CcuDieBase(0, 0), m_ras->CcuDieBase(1, 0));
    EXPECT_NE(m_ras->CcuDieBase(0, 0), m_ras->CcuDieBase(0, 1));
    // dieId 越界 → 0
    EXPECT_EQ(m_ras->CcuDieBase(0, 2), 0ULL);
    EXPECT_EQ(m_ras->CcuDieBase(0, 99), 0ULL);
}

TEST_F(RankAddrSpaceTest, CcuWindowGeometry)
{
    EXPECT_EQ(m_ras->CcuWindowOffset(), CCU_WIN_OFF);
    EXPECT_EQ(m_ras->CcuWindowSize(), CCU_WIN_SIZE);
    EXPECT_EQ(m_ras->DieNum(), 2U);
}

TEST_F(RankAddrSpaceTest, UserMemIsComplementOfCcu)
{
    ASSERT_EQ(m_ras->UserMemSegCount(), 2U);
    EXPECT_EQ(m_ras->UserMemSegOffset(0), 0x0ULL);
    EXPECT_EQ(m_ras->UserMemSegSize(0), CCU_WIN_OFF);                              // 56 GiB
    EXPECT_EQ(m_ras->UserMemSegOffset(1), CCU_WIN_OFF + CCU_WIN_SIZE);             // 58 GiB
    EXPECT_EQ(m_ras->UserMemSegSize(1), RANK_STRIDE - CCU_WIN_OFF - CCU_WIN_SIZE); // 966 GiB
    EXPECT_EQ(m_ras->UserMemTotal(), USER_MEM_TOTAL);
    // 越界段号 → 0
    EXPECT_EQ(m_ras->UserMemSegSize(99), 0ULL);
    EXPECT_EQ(m_ras->UserMemSegOffset(99), 0ULL);
}

/// 段 0 与 CCU 窗口无缝衔接且不侵入
TEST_F(RankAddrSpaceTest, Seg0EndsExactlyAtCcuStart)
{
    const uint64_t seg0LastVa = RANK0_BASE + m_ras->UserMemSegOffset(0) + m_ras->UserMemSegSize(0) - 8;
    EXPECT_EQ(seg0LastVa, 0x8DFFFFFFF8ULL);
    EXPECT_EQ(RANK0_BASE + m_ras->UserMemSegOffset(0) + m_ras->UserMemSegSize(0), RANK0_DIE0_CCU);
    EXPECT_FALSE(m_ras->IsUserMemVa(RANK0_DIE0_CCU, 0));
    uint32_t r = 0, d = 0;
    EXPECT_FALSE(m_ras->IsInCcuWindow(seg0LastVa, r, d));
}

/// 段 1 起点恰为 CCU 窗口末端
TEST_F(RankAddrSpaceTest, Seg1StartsExactlyAtCcuEnd)
{
    const uint64_t seg1FirstVa = RANK0_BASE + m_ras->UserMemSegOffset(1);
    EXPECT_EQ(seg1FirstVa, 0x8E80000000ULL);
    EXPECT_EQ(seg1FirstVa, RANK0_DIE0_CCU + CCU_WIN_SIZE);
    EXPECT_TRUE(m_ras->IsUserMemVa(seg1FirstVa, 0));
    uint32_t r = 0, d = 0;
    EXPECT_FALSE(m_ras->IsInCcuWindow(seg1FirstVa, r, d));
}

/// 段 1 末端 + 1 == rank1 起点（不侵入下一 rank）
TEST_F(RankAddrSpaceTest, Seg1EndsAtNextRankBase)
{
    const uint64_t seg1End = RANK0_BASE + m_ras->UserMemSegOffset(1) + m_ras->UserMemSegSize(1);
    EXPECT_EQ(seg1End, RANK1_BASE);
    EXPECT_TRUE(m_ras->IsUserMemVa(seg1End - 8, 0));
    EXPECT_FALSE(m_ras->IsUserMemVa(seg1End, 0)) << "rank1 的地址不属于 rank0";
    EXPECT_TRUE(m_ras->IsUserMemVa(seg1End, 1)) << "它属于 rank1 的段0 起点";
}

/// 遍历整个 rank0 窗口：CCU 窗口内的地址必须全部被 IsInCcuWindow 命中，且都不是 user_mem
TEST_F(RankAddrSpaceTest, CcuWindowFullyCoveredAndExcludedFromUserMem)
{
    uint32_t r = 0, d = 0;
    // 采样检查，避免遍历 1 TiB
    const uint64_t probes[] = {
        RANK0_DIE0_CCU,          RANK0_DIE0_CCU + 0x1000,        RANK0_DIE0_CCU + 0x4000000 - 1,    RANK0_DIE1_CCU,
        RANK0_DIE1_CCU + 0x1000, RANK0_DIE1_CCU + 0x4000000 - 1, RANK0_DIE0_CCU + CCU_WIN_SIZE - 1,
    };
    for (uint64_t va : probes) {
        r = 0;
        d = 0;
        EXPECT_TRUE(m_ras->IsInCcuWindow(va, r, d)) << "va=" << std::hex << va;
        EXPECT_EQ(r, 0U);
        EXPECT_FALSE(m_ras->IsUserMemVa(va, 0)) << "CCU 地址不得被判为 user_mem, va=" << std::hex << va;
    }
    // die 归属
    r = 0;
    d = 0;
    ASSERT_TRUE(m_ras->IsInCcuWindow(RANK0_DIE0_CCU + 0x100000, r, d));
    EXPECT_EQ(d, 0U);
    r = 0;
    d = 0;
    ASSERT_TRUE(m_ras->IsInCcuWindow(RANK0_DIE1_CCU + 0x100000, r, d));
    EXPECT_EQ(d, 1U);
    // rank 归属
    r = 0;
    d = 0;
    ASSERT_TRUE(m_ras->IsInCcuWindow(RANK1_DIE0_CCU + 0x100000, r, d));
    EXPECT_EQ(r, 1U);
    EXPECT_EQ(d, 0U);
}

TEST_F(RankAddrSpaceTest, IsInCcuWindowRejectsOutside)
{
    uint32_t r = 0, d = 0;
    EXPECT_FALSE(m_ras->IsInCcuWindow(0, r, d));
    EXPECT_FALSE(m_ras->IsInCcuWindow(RANK0_BASE - 1, r, d));
    EXPECT_FALSE(m_ras->IsInCcuWindow(RANK0_BASE, r, d)); // user_mem 段0 起点
    EXPECT_FALSE(m_ras->IsInCcuWindow(RANK0_BASE + CCU_WIN_OFF - 8, r, d));
    EXPECT_FALSE(m_ras->IsInCcuWindow(RANK0_BASE + CCU_WIN_OFF + CCU_WIN_SIZE, r, d));
    EXPECT_FALSE(m_ras->IsUserMemVa(RANK0_BASE - 1, 0));
    EXPECT_FALSE(m_ras->IsUserMemVa(0, 0));
}

/// latent bug B1：devKey == 0 必须被拦截（否则 rankIdx = 0-1 下溢为 UINT64_MAX）
TEST_F(RankAddrSpaceTest, RankIdxOfDeviceKeyRejectsZero)
{
    uint32_t rankIdx = 0xDEADBEEF;
    EXPECT_FALSE(RankAddressSpace::RankIdxOfDeviceKey(0, rankIdx));
    EXPECT_TRUE(RankAddressSpace::RankIdxOfDeviceKey(1, rankIdx));
    EXPECT_EQ(rankIdx, 0U);
    EXPECT_TRUE(RankAddressSpace::RankIdxOfDeviceKey(2, rankIdx));
    EXPECT_EQ(rankIdx, 1U);
    EXPECT_TRUE(RankAddressSpace::RankIdxOfDeviceKey(1024, rankIdx));
    EXPECT_EQ(rankIdx, 1023U);
    // 超出 uint32 的 devKey 也应拒绝
    EXPECT_FALSE(RankAddressSpace::RankIdxOfDeviceKey(static_cast<uint64_t>(UINT32_MAX) + 2, rankIdx));
}

/// 风险 R4：47-bit 位宽只作**告警**，不作硬约束；且 rank 槽位**无任何规模上界**
/// （原 MaxRankSlots()/HCCL_VM_MAX_RANKS 已删除，集群规模由拓扑决定）
TEST_F(RankAddrSpaceTest, VaWidthIsDiagnosticOnlyAndRankSlotsUncapped)
{
    // 47-bit 位宽是独立的诊断量：rank_base=2^39, rank_stride=2^40
    //   → (2^47 - 2^39)/2^40 = 127.5 → 127
    const uint32_t vaLimit = m_ras->Va47BitRankLimit();
    EXPECT_EQ(vaLimit, 127U);
    // rank127 起点在 47-bit 内，rank128 起点越界
    EXPECT_LT(m_ras->RankBase(127), 1ULL << 47);
    EXPECT_GE(m_ras->RankBase(128), 1ULL << 47);
    // 但越过位宽诊断值后仍可正常计算基址（模拟 VA 是 token，不被 host 解引用）：
    // 原 1024 上界已删除，故 1024 以上的 rankIdx（如 8192 设备集群的尾部）必须可用
    EXPECT_NE(m_ras->RankBase(1023), 0ULL);
    EXPECT_NE(m_ras->RankBase(1024), 0ULL);
    EXPECT_NE(m_ras->RankBase(8191), 0ULL);
    EXPECT_NE(m_ras->CcuDieBase(1023, 1), 0ULL);
    EXPECT_NE(m_ras->CcuDieBase(8191, 1), 0ULL);
    // 唯一的硬约束是 uint64 溢出防护：溢出时返回 0，由调用方按失败处理
    EXPECT_EQ(m_ras->RankBase(UINT32_MAX), 0ULL);
}

/// 分配器（§4.5.7.3 G3 分段 first-fit）的核心不变量：任何成功分配的块都不碰 CCU
TEST_F(RankAddrSpaceTest, FirstFitNeverLandsInCcuWindow)
{
    const uint32_t rankIdx = 0;
    std::vector<uint64_t> cursors(m_ras->UserMemSegCount(), 0);
    uint32_t r = 0, d = 0;
    // 模拟连续分配：大小在 8B ~ 4 MiB 之间变化，总量远超段0（56 GiB）以强制跳段
    const uint64_t sizes[] = {8, 64, 4096, 0x100000, 0x400000, 0xDFFFFFF00ULL /* 一次性逼近段0末端 */};
    for (uint64_t sz : sizes) {
        const uint64_t aligned = (sz + 7) & ~7ULL;
        bool placed = false;
        for (size_t i = 0; i < cursors.size(); ++i) {
            const uint64_t segSize = m_ras->UserMemSegSize(i);
            const uint64_t cur = cursors[i];
            if (cur > segSize || aligned > segSize - cur) {
                continue;
            }
            const uint64_t va = m_ras->RankBase(rankIdx) + m_ras->UserMemSegOffset(i) + cur;
            // 三重不变量
            EXPECT_FALSE(m_ras->IsInCcuWindow(va, r, d)) << "块首落入 CCU, va=" << std::hex << va;
            EXPECT_FALSE(m_ras->IsInCcuWindow(va + aligned - 1, r, d))
                << "块尾落入 CCU, va=" << std::hex << va << " size=" << aligned;
            EXPECT_GE(va, m_ras->RankBase(rankIdx));
            EXPECT_LT(va + aligned, m_ras->RankBase(rankIdx) + m_ras->RankWindowSize());
            EXPECT_TRUE(m_ras->IsUserMemVa(va, rankIdx));
            EXPECT_TRUE(m_ras->IsUserMemVa(va + aligned - 1, rankIdx));
            cursors[i] = cur + aligned;
            placed = true;
            break;
        }
        EXPECT_TRUE(placed) << "分配失败, size=" << std::hex << aligned;
    }
}

/// ★ 跨洞反例（设计文档 §4.5.7.2）：证明"单一全局线性水位"会踩 CCU，而分段 first-fit 不会
TEST_F(RankAddrSpaceTest, StraddleCounterexample)
{
    const uint32_t rankIdx = 0;
    const uint64_t seg0Size = m_ras->UserMemSegSize(0);
    const uint64_t cur = seg0Size - 0x100; // 段0 只剩 0x100
    const uint64_t req = 0x400;            // 请求 0x400

    // 线性总量检查会通过（这正是缺陷所在）
    EXPECT_LE(cur + req, m_ras->UserMemTotal());

    // 错误做法：VA = rankBase + cur → 块尾侵入 CCU 0x300 字节
    const uint64_t badVa = m_ras->RankBase(rankIdx) + cur;
    uint32_t r = 0, d = 0;
    EXPECT_FALSE(m_ras->IsInCcuWindow(badVa, r, d)) << "块首仍在段0 内";
    EXPECT_TRUE(m_ras->IsInCcuWindow(badVa + req - 1, r, d)) << "块尾必须侵入 CCU（反例成立）";
    EXPECT_EQ(d, 0U) << "侵入的是 die0";

    // 正确做法：first-fit 发现段0 放不下 → 跳段1
    EXPECT_GT(req, seg0Size - cur);
    const uint64_t goodVa = m_ras->RankBase(rankIdx) + m_ras->UserMemSegOffset(1);
    EXPECT_EQ(goodVa, 0x8E80000000ULL);
    r = 0;
    d = 0;
    EXPECT_FALSE(m_ras->IsInCcuWindow(goodVa, r, d));
    EXPECT_FALSE(m_ras->IsInCcuWindow(goodVa + req - 1, r, d));
}

/// 零回归：段0 内 VA 与旧公式 rankBase + used 逐位相同（设计文档 §4.5.7.4）
TEST_F(RankAddrSpaceTest, ZeroRegressionWithinSeg0)
{
    const uint32_t rankIdx = 0;
    const uint64_t rankBase = m_ras->RankBase(rankIdx);
    // 旧公式：deviceBase + (devPhyId-1)*deviceAddrStep + used
    const uint64_t probes[] = {
        0, 8, 0x1000, 0x40000000, 0x100000000, 0x200000000, 0x405000000 /* 实测峰值 0.405 GiB 附近 */, 0xDFFFFFFF8ULL};
    for (uint64_t used : probes) {
        ASSERT_LT(used, m_ras->UserMemSegSize(0)) << "probe 超出段0";
        const uint64_t newVa = rankBase + m_ras->UserMemSegOffset(0) + used;
        const uint64_t oldVa = 0x8000000000ULL + (rankIdx) * 0x10000000000ULL + used;
        EXPECT_EQ(newVa, oldVa) << "used=" << std::hex << used;
    }
}

} // namespace
