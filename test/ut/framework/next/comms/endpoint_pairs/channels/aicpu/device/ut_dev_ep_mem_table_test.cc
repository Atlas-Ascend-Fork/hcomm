/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <thread>

#include "gtest/gtest.h"

#include "aicpu/device/dev_ep_mem_table.h"

namespace {

LocUbBufLite MakeEntry(uint64_t addr, uint64_t size, uint32_t tokenId, uint32_t tokenValue)
{
    return LocUbBufLite{addr, size, tokenId, tokenValue};
}

} // namespace

TEST(DevEpMemTableTest, Ut_Update_When_NewKey_Expect_InsertedWithRefOne)
{
    Hccl::DevEpMemTable table;
    ASSERT_EQ(table.Update(MakeEntry(0x1000, 0x50, 1, 0x11)), HCCL_SUCCESS);
    EXPECT_EQ(table.abi.entryNum, 1U);

    LocUbBufLite hit{};
    ASSERT_TRUE(table.Find(0x1000, 0x50, hit));
    EXPECT_EQ(hit.tokenId, 1U);
    EXPECT_EQ(hit.tokenValue, 0x11U);
}

TEST(DevEpMemTableTest, Ut_Update_When_InvalidEntry_Expect_ParaError)
{
    Hccl::DevEpMemTable table;
    EXPECT_EQ(table.Update(MakeEntry(0, 0x50, 1, 0x11)), HCCL_E_PARA);
    EXPECT_EQ(table.Update(MakeEntry(0x1000, 0, 1, 0x11)), HCCL_E_PARA);
    EXPECT_EQ(table.abi.entryNum, 0U);
}

TEST(DevEpMemTableTest, Ut_Update_When_SameKeyTwice_Expect_RefAggregated)
{
    Hccl::DevEpMemTable table;
    ASSERT_EQ(table.Update(MakeEntry(0x1000, 0x50, 1, 0x11)), HCCL_SUCCESS);
    ASSERT_EQ(table.Update(MakeEntry(0x1000, 0x50, 1, 0x11)), HCCL_SUCCESS);
    // 同 (addr,size) 重复注册聚合为单条目（ref=2），非覆盖
    EXPECT_EQ(table.abi.entryNum, 1U);

    // 第一次注销：ref 2->1，条目保留，查找仍命中
    EXPECT_EQ(table.Remove(0x1000, 0x50), HCCL_SUCCESS);
    EXPECT_EQ(table.abi.entryNum, 1U);
    LocUbBufLite hit{};
    ASSERT_TRUE(table.Find(0x1000, 0x50, hit));
    EXPECT_EQ(hit.tokenId, 1U);

    // 第二次注销：ref 1->0，条目删除，查找未命中
    EXPECT_EQ(table.Remove(0x1000, 0x50), HCCL_SUCCESS);
    EXPECT_EQ(table.abi.entryNum, 0U);
    EXPECT_FALSE(table.Find(0x1000, 0x50, hit));

    // 条目已删后再注销：键不存在 NOT_FOUND（幂等注销良性）
    EXPECT_EQ(table.Remove(0x1000, 0x50), HCCL_E_NOT_FOUND);
}

TEST(DevEpMemTableTest, Ut_Find_When_NestedSameBase_Expect_CoexistNoShadow)
{
    Hccl::DevEpMemTable table;
    // 同基址嵌套注册：各占一键共存（原按裸基址单键实现后者覆盖前者）
    ASSERT_EQ(table.Update(MakeEntry(0x1000, 0x50, 1, 0x11)), HCCL_SUCCESS);
    ASSERT_EQ(table.Update(MakeEntry(0x1000, 0x100, 2, 0x22)), HCCL_SUCCESS);
    EXPECT_EQ(table.abi.entryNum, 2U);

    // 仅被大区间包含的查询命中大区间条目
    LocUbBufLite hit{};
    ASSERT_TRUE(table.Find(0x10A0, 0x10, hit));
    EXPECT_EQ(hit.tokenId, 2U);

    // 注销小区间后大区间条目不受影响（原基址覆盖实现此查询将落空回退）
    EXPECT_EQ(table.Remove(0x1000, 0x50), HCCL_SUCCESS);
    EXPECT_EQ(table.abi.entryNum, 1U);
    ASSERT_TRUE(table.Find(0x10A0, 0x10, hit));
    EXPECT_EQ(hit.tokenId, 2U);
}

TEST(DevEpMemTableTest, Ut_Find_When_NearestBaseNotContaining_Expect_FallbackToFartherEntry)
{
    Hccl::DevEpMemTable table;
    ASSERT_EQ(table.Update(MakeEntry(0x1000, 0x100, 2, 0x22)), HCCL_SUCCESS);
    ASSERT_EQ(table.Update(MakeEntry(0x1080, 0x10, 3, 0x33)), HCCL_SUCCESS);

    LocUbBufLite hit{};
    // 最近低位基址 (0x1080,0x10) 不包含该查询，全量回退命中更远的 (0x1000,0x100)
    ASSERT_TRUE(table.Find(0x1090, 0x8, hit));
    EXPECT_EQ(hit.tokenId, 2U);
}

TEST(DevEpMemTableTest, Ut_Find_When_QueryNotContained_Expect_Miss)
{
    Hccl::DevEpMemTable table;
    ASSERT_EQ(table.Update(MakeEntry(0x1000, 0x50, 1, 0x11)), HCCL_SUCCESS);
    ASSERT_EQ(table.Update(MakeEntry(0x1020, 0x10, 3, 0x33)), HCCL_SUCCESS);

    LocUbBufLite hit{};
    // 部分相交（越界尾）：任何条目均不完全包含
    EXPECT_FALSE(table.Find(0x1030, 0x40, hit));
    // 部分相交（越界头）
    EXPECT_FALSE(table.Find(0xF00, 0x200, hit));
    // 完全在表外
    EXPECT_FALSE(table.Find(0x4000, 0x10, hit));
}

TEST(DevEpMemTableTest, Ut_Remove_When_KeyAbsentOrMismatch_Expect_NotFound)
{
    Hccl::DevEpMemTable table;
    LocUbBufLite hit{};
    // 空表：Remove/Find 均 NOT_FOUND/未命中
    EXPECT_EQ(table.Remove(0x1000, 0x50), HCCL_E_NOT_FOUND);
    EXPECT_FALSE(table.Find(0x1000, 0x50, hit));

    ASSERT_EQ(table.Update(MakeEntry(0x1000, 0x50, 1, 0x11)), HCCL_SUCCESS);
    // 同基址不同 size 的键不匹配（(addr,size) 精确键）
    EXPECT_EQ(table.Remove(0x1000, 0x60), HCCL_E_NOT_FOUND);
    EXPECT_EQ(table.abi.entryNum, 1U);
}

TEST(DevEpMemTableTest, Ut_FindFirst_When_TableStates_Expect_FirstOrFalse)
{
    Hccl::DevEpMemTable table;
    LocUbBufLite first{};
    EXPECT_FALSE(table.FindFirst(first));

    ASSERT_EQ(table.Update(MakeEntry(0x2000, 0x20, 3, 0x33)), HCCL_SUCCESS);
    ASSERT_EQ(table.Update(MakeEntry(0x1000, 0x50, 1, 0x11)), HCCL_SUCCESS);
    // 首条 = 键序最小（基址最小）条目
    ASSERT_TRUE(table.FindFirst(first));
    EXPECT_EQ(first.tokenId, 1U);
    EXPECT_EQ(first.addr, 0x1000U);
}

TEST(DevEpMemTableTest, Ut_CheckAbi_When_VersionMismatch_Expect_False)
{
    Hccl::DevEpMemTable table;
    EXPECT_TRUE(table.CheckAbi());

    // 哨兵字段为 public ABI 头，可观测篡改：版本不符快速失败
    const uint32_t savedVersion = table.abi.version;
    table.abi.version = savedVersion + 1U;
    EXPECT_FALSE(table.CheckAbi());
    table.abi.version = savedVersion;
    EXPECT_TRUE(table.CheckAbi());
}

// ===================== 快照并发读写与异常回滚 UT（检视整改） =====================
// Update/Remove 已改为拷贝-发布、Find/FindFirst 已改为 atomic_load 快照，
// 以下用例兜底发布代切换并发、ref 聚合跨拷贝与拷贝异常回滚三处新机制。

// Update/Remove 的 try 块内为纯 stdlib 调用（make_shared/map 拷贝/atomic_store），无 mockcpp
// 可打桩符号；依据 C++ [replacement.functions] 以程序级 operator new 替换实现确定性 bad_alloc
// 注入：武装后的首次分配经 exchange 原子自解并抛出，其余时间等价 malloc 转发，对同二进制
// 其他用例零影响。须为全局作用域外部链接（生产 TU 的分配经符号替换路由至此），不得 static。
static std::atomic<bool> g_snapshotBadAllocArmed{false};

void* operator new(std::size_t size)
{
    if (g_snapshotBadAllocArmed.exchange(false)) {
        throw std::bad_alloc();
    }
    if (size == 0) {
        size = 1;
    }
    if (void* ptr = std::malloc(size)) {
        return ptr;
    }
    throw std::bad_alloc();
}

// RAII 守卫：异常断言中途失败也不让武装标志泄漏到后续用例
class BadAllocArmedGuard {
public:
    BadAllocArmedGuard() { g_snapshotBadAllocArmed.store(true); }
    ~BadAllocArmedGuard() { g_snapshotBadAllocArmed.store(false); }
};

// 快照隔离下并发 Find/FindFirst ↔ Update/Remove 为契约内安全场景（对齐 CpuRoce 侧
// ut_host_roce_local_mem_refresh.cc 并发用例）：读者每次只可能看到某个完整发布过的代——
// 稳定条目必命中且 info 不变，变更条目要么命中要么未命中（半截代不可见）；
// 写者 Update×2+Remove×2 驱动 ref 聚合跨拷贝与删除/重插的代切换
TEST(DevEpMemTableTest, Ut_Find_When_UpdateRemoveConcurrently_Expect_ConsistentSnapshotLookup)
{
    Hccl::DevEpMemTable table;
    const LocUbBufLite stable = MakeEntry(0x1000, 0x100, 1, 0x11);
    const LocUbBufLite changing = MakeEntry(0x3000, 0x100, 2, 0x22);
    // 变更条目由写者循环从 absent 态全周期驱动（insert ref=1 起拍），稳定条目全程在表
    ASSERT_EQ(table.Update(stable), HCCL_SUCCESS);

    std::atomic<bool> failed{false};
    std::thread reader([&] {
        for (size_t i = 0; i < 1000; ++i) {
            LocUbBufLite hit{};
            if (!table.Find(stable.addr, stable.size, hit) || hit.tokenId != stable.tokenId
                || hit.tokenValue != stable.tokenValue) {
                failed = true;
            }
            if (table.Find(changing.addr, changing.size, hit) && hit.tokenId != changing.tokenId) {
                failed = true;
            }
            // 键序最小者恒为稳定条目：FindFirst 不受变更条目增删影响
            if (!table.FindFirst(hit) || hit.tokenId != stable.tokenId) {
                failed = true;
            }
        }
    });
    for (size_t i = 0; i < 1000; ++i) {
        // 稳态节奏：插入 ref=1 -> 聚合至 2 -> 回落 1（条目保留）-> 归零删除；下轮重新插入。
        // 全删后重插条目 ref=1，故每轮恰需两次 Remove 删净（3 拍节奏会使第二次 Remove 落空）
        if (table.Update(changing) != HCCL_SUCCESS || table.Update(changing) != HCCL_SUCCESS
            || table.Remove(changing.addr, changing.size) != HCCL_SUCCESS
            || table.Remove(changing.addr, changing.size) != HCCL_SUCCESS) {
            failed = true;
        }
    }
    reader.join();
    EXPECT_FALSE(failed);

    // 终态：变更条目已删净，稳定条目不受影响
    LocUbBufLite hit{};
    EXPECT_FALSE(table.Find(changing.addr, changing.size, hit));
    ASSERT_TRUE(table.Find(stable.addr, stable.size, hit));
    EXPECT_EQ(hit.tokenId, stable.tokenId);
    EXPECT_EQ(table.abi.entryNum, 1U);
}

// Update 快照拷贝抛 bad_alloc：就地转 HCCL_E_MEMORY，published_ 保持旧代
// （entryNum 不变、旧条目仍可查、新键未插入），且异常后表仍可用
TEST(DevEpMemTableTest, Ut_Update_When_SnapshotCopyThrows_Expect_OldGenerationKept)
{
    Hccl::DevEpMemTable table;
    ASSERT_EQ(table.Update(MakeEntry(0x1000, 0x100, 1, 0x11)), HCCL_SUCCESS);

    {
        BadAllocArmedGuard guard;
        EXPECT_EQ(table.Update(MakeEntry(0x3000, 0x100, 2, 0x22)), HCCL_E_MEMORY);
    }
    EXPECT_EQ(table.abi.entryNum, 1U);

    LocUbBufLite hit{};
    ASSERT_TRUE(table.Find(0x1000, 0x100, hit));
    EXPECT_EQ(hit.tokenId, 1U);
    EXPECT_FALSE(table.Find(0x3000, 0x100, hit));

    // 异常后表仍可用：正常 Update 恢复
    EXPECT_EQ(table.Update(MakeEntry(0x3000, 0x100, 2, 0x22)), HCCL_SUCCESS);
    EXPECT_EQ(table.abi.entryNum, 2U);
}

// Remove 快照拷贝抛 bad_alloc：旧代保持且失败路径未改动旧代 ref——
// 解除武装后仍需恰两次 Remove 才删净（若失败时已扣减 ref，一次 Remove 即会删条目）
TEST(DevEpMemTableTest, Ut_Remove_When_SnapshotCopyThrows_Expect_OldGenerationAndRefKept)
{
    Hccl::DevEpMemTable table;
    ASSERT_EQ(table.Update(MakeEntry(0x1000, 0x100, 1, 0x11)), HCCL_SUCCESS);
    ASSERT_EQ(table.Update(MakeEntry(0x1000, 0x100, 1, 0x11)), HCCL_SUCCESS); // ref=2

    {
        BadAllocArmedGuard guard;
        EXPECT_EQ(table.Remove(0x1000, 0x100), HCCL_E_MEMORY);
    }
    EXPECT_EQ(table.abi.entryNum, 1U);
    LocUbBufLite hit{};
    ASSERT_TRUE(table.Find(0x1000, 0x100, hit));

    // ref 未被失败路径改动：需两次 Remove 删净
    EXPECT_EQ(table.Remove(0x1000, 0x100), HCCL_SUCCESS);
    EXPECT_EQ(table.abi.entryNum, 1U);
    EXPECT_EQ(table.Remove(0x1000, 0x100), HCCL_SUCCESS);
    EXPECT_EQ(table.abi.entryNum, 0U);
    EXPECT_FALSE(table.Find(0x1000, 0x100, hit));
}
