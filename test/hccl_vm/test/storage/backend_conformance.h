/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file backend_conformance.h
 * @brief 后端公共 conformance suite：单一 canonical 测试目录 + 后端无关 cases。
 *
 * 【复用模型】
 * 本头文件是全部已实现后端（内存语义 Fake 引擎、SQLiteEngine、MariaDbEngine）共用的
 * 行为合同单一来源：
 *   - canonical 小型测试目录由 StorageCatalog 机制登记（CONFORMANCE_DB 一个逻辑库、
 *     七张表），fixture 负责把它装配为进程会话；
 *   - cases 只通过公开 Table/Storage 原子接口表达期望，不出现任何后端专有断言
 *     （SQL 渲染、PRAGMA、文件备份、statement cache、fault injection 留在各后端私有测试）；
 *   - 公共 case 失败时不得为某个后端修改期望；新后端只提供 fixture 并运行同一套 case。
 *
 * 【Fixture 协议】
 * fixture 在进程内安装一个指向被测引擎的会话（经 StorageRuntime 工厂注册或等价装配），
 * 随后 cases 通过 Storage::/AcquireTable 公开接口执行。HCCL_VM_CONFORMANCE_CASE 宏按
 * case 名生成 gtest 用例；每个用例先由 fixture 建立全新会话，保证用例间零数据耦合。
 */
#ifndef HCCL_VM_STORAGE_BACKEND_CONFORMANCE_H
#define HCCL_VM_STORAGE_BACKEND_CONFORMANCE_H

#include <atomic>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "gtest/gtest.h"

#include "storage/query_expression.h"
#include "storage/storage_catalog.h"
#include "storage/table_access.h"
#include "storage/transaction.h"

// ---- canonical conformance 测试记录 ----
namespace storage_conformance {

struct Device {
    uint64_t id{0};
    uint32_t server_id{0};
    char name[16]{};
    uint32_t status{0};
};

struct BigValue {
    uint64_t id{0};
    uint64_t value{0};
};

struct BlobRow {
    uint64_t id{0};
    uint32_t owner{0};
    uint8_t bytes[12]{};
};

struct UniqueRow {
    uint64_t id{0};
    uint32_t code{0};
};

struct ChildRow {
    uint64_t id{0};
    uint64_t parent_id{0};
    uint32_t amount{0};
};

struct GuardRow {
    uint64_t id{0};
    uint64_t parent_id{0};
    uint32_t amount{0};
};

struct DynTask {
    uint64_t id{0};
    uint64_t owner{0};
    uint32_t seq{0};
};

/// INCREMENT 域合同 fixture：无符号/有符号 64 位计数器各一
struct CounterRow {
    uint64_t id{0};
    uint64_t ucount{0};
    int64_t scount{0};
};

/// 投影回归 fixture：变长大 BLOB 行（未投影时保持默认空载荷）
struct WideRow {
    uint64_t id{0};
    uint32_t tag{0};
    std::vector<uint8_t> payload{};
};

/// 投影回归 fixture：定长数组 + hook 有效长度（生产 CcuChannelTab jetty 形态缩影）
struct SlotRow {
    uint64_t id{0};
    uint32_t count{0};
    uint64_t slots[6]{};
};

} // namespace storage_conformance

// SlotRow 有效数组长度 hook：slots 只承载 count 个有效元素（测试目录特化，
// 与生产 CcuChannelTab 同一 hook 能力）
namespace HcclSim::Storage::Ddl {
template <>
struct RecordCodecHooks<storage_conformance::SlotRow> : DefaultRecordCodecHooks<storage_conformance::SlotRow> {
    static int64_t EffectiveBytesLength(const storage_conformance::SlotRow& record, uint16_t fieldId)
    {
        return fieldId == 2U ? static_cast<int64_t>(record.count) * static_cast<int64_t>(sizeof(record.slots[0])) : -1;
    }
};
} // namespace HcclSim::Storage::Ddl

// ---- conformance 目录（单一事实源；fixture 装配时使用） ----
using namespace HcclSim::Storage::Ddl;
#define Field HCCL_VM_DDL_FIELD
#define Version HCCL_VM_DDL_VERSION
#define PrimaryKey HCCL_VM_DDL_PRIMARY_KEY
#define Unique HCCL_VM_DDL_UNIQUE
#define Index HCCL_VM_DDL_INDEX
#define ForeignKey HCCL_VM_DDL_FOREIGN_KEY
#define Scope Ddl::ScopeKind

#define HCCL_VM_CONFORMANCE_DATABASE(Table)                                                                           \
    Table(                                                                                                            \
        storage_conformance::Device, (Version(1), Scope::Static, Field(id, AutoIncrement), Field(server_id),          \
                                      Field(name, Text), Field(status), PrimaryKey(id), Index(server_id)))            \
        Table(                                                                                                        \
            storage_conformance::BigValue,                                                                            \
            (Version(1), Scope::Static, Field(id, AutoIncrement), Field(value), PrimaryKey(id)))                      \
            Table(                                                                                                    \
                storage_conformance::BlobRow, (Version(1), Scope::Static, Field(id, AutoIncrement), Field(owner),     \
                                               Field(bytes, Blob), PrimaryKey(id)))                                   \
                Table(                                                                                                \
                    storage_conformance::UniqueRow,                                                                   \
                    (Version(1), Scope::Static, Field(id, AutoIncrement), Field(code), PrimaryKey(id), Unique(code))) \
                    Table(                                                                                            \
                        storage_conformance::ChildRow,                                                                \
                        (Version(1), Scope::Static, Field(id, AutoIncrement), Field(parent_id), Field(amount),        \
                         PrimaryKey(id), Index(parent_id),                                                            \
                         ForeignKey(parent_id, storage_conformance::UniqueRow::id, OnDelete::Cascade)))               \
                        Table(                                                                                        \
                            storage_conformance::GuardRow,                                                            \
                            (Version(1), Scope::Static, Field(id, AutoIncrement), Field(parent_id), Field(amount),    \
                             PrimaryKey(id), Index(parent_id),                                                        \
                             ForeignKey(parent_id, storage_conformance::UniqueRow::id, OnDelete::Restrict)))          \
                            Table(                                                                                    \
                                storage_conformance::DynTask,                                                         \
                                (Version(1), Scope::DynamicPid, Field(id, AutoIncrement), Field(owner), Field(seq),   \
                                 PrimaryKey(id)))                                                                     \
                                Table(                                                                                \
                                    storage_conformance::CounterRow,                                                  \
                                    (Version(1), Scope::Static, Field(id, AutoIncrement), Field(ucount),              \
                                     Field(scount), PrimaryKey(id)))                                                  \
                                    Table(                                                                            \
                                        storage_conformance::WideRow,                                                 \
                                        (Version(1), Scope::Static, Field(id, AutoIncrement), Field(tag),             \
                                         Field(payload), PrimaryKey(id)))                                             \
                                        Table(                                                                        \
                                            storage_conformance::SlotRow,                                             \
                                            (Version(1), Scope::Static, Field(id, AutoIncrement), Field(count),       \
                                             Field(slots, Blob), PrimaryKey(id)))

#define HCCL_VM_DDL_CURRENT_DATABASE "CONFORMANCE_DB"
HCCL_VM_DEFINE_DATABASE(kConformanceDatabase, CONFORMANCE_DB, HCCL_VM_CONFORMANCE_DATABASE)
#undef HCCL_VM_DDL_CURRENT_DATABASE
#undef Field
#undef Version
#undef PrimaryKey
#undef Unique
#undef Index
#undef ForeignKey
#undef Scope

namespace storage_conformance {
using namespace HcclSim;
using namespace HcclSim::Storage;

// ---- 公共 conformance cases（后端无关；期望不得按后端修改） ----
namespace cases {

    /// 自增主键生成：Insert 成功写回递增 id，失败不伪造 id
    inline void GeneratedIdCase()
    {
        Device first;
        first.server_id = 1;
        const auto inserted = Storage::Insert(first);
        ASSERT_TRUE(inserted.ok());
        EXPECT_GT(first.id, 0U);
        Device second;
        second.server_id = 1;
        ASSERT_TRUE(Storage::Insert(second).ok());
        EXPECT_EQ(second.id, first.id + 1U);
    }

    /// SelectOne 命中单行/未命中 NOT_FOUND；Select 多行零行为 OK + 空行集
    inline void SelectOneAndSelectCase()
    {
        Device device;
        device.server_id = 7;
        device.status = 2;
        ASSERT_TRUE(Storage::Insert(device).ok());
        const auto one = Storage::SelectOne<Device>(Eq(&Device::id, device.id));
        ASSERT_TRUE(one.ok());
        ASSERT_TRUE(one.value.has_value());
        EXPECT_EQ(one.value->server_id, 7U);
        EXPECT_EQ(one.value->status, 2U);

        const auto missing = Storage::SelectOne<Device>(Eq(&Device::id, uint64_t{999999}));
        EXPECT_EQ(missing.code, DbCode::NOT_FOUND);

        const auto rows = Storage::Select<Device>(Eq(&Device::id, uint64_t{999999}));
        ASSERT_TRUE(rows.ok());
        EXPECT_TRUE(rows.value->empty());
    }

    /// 条件树：等值/范围/IN/前缀/AND/OR/空 IN 恒假全部下推执行
    inline void ConditionTreeCase()
    {
        Device a;
        a.server_id = 1;
        a.status = 1;
        std::memcpy(a.name, "alpha", sizeof("alpha"));
        Device b;
        b.server_id = 1;
        b.status = 2;
        std::memcpy(b.name, "beta", sizeof("beta"));
        Device c;
        c.server_id = 2;
        c.status = 2;
        std::memcpy(c.name, "gamma", sizeof("gamma"));
        ASSERT_TRUE(Storage::Insert(a).ok());
        ASSERT_TRUE(Storage::Insert(b).ok());
        ASSERT_TRUE(Storage::Insert(c).ok());

        const auto onServer = Storage::Select<Device>(Eq(&Device::server_id, uint32_t{1}));
        ASSERT_TRUE(onServer.ok());
        EXPECT_EQ(onServer.value->size(), 2U);

        const auto ranged = Storage::Select<Device>(Ge(&Device::status, uint32_t{2}));
        ASSERT_TRUE(ranged.ok());
        EXPECT_EQ(ranged.value->size(), 2U);

        const auto collected = Storage::Select<Device>(In(&Device::server_id, std::vector<uint32_t>{1, 2}));
        ASSERT_TRUE(collected.ok());
        EXPECT_EQ(collected.value->size(), 3U);

        const auto prefixed = Storage::Select<Device>(StartsWith(&Device::name, std::string{"al"}));
        ASSERT_TRUE(prefixed.ok());
        EXPECT_EQ(prefixed.value->size(), 1U);

        const auto combined
            = Storage::Select<Device>(And(Eq(&Device::server_id, uint32_t{1}), Eq(&Device::status, uint32_t{2})));
        ASSERT_TRUE(combined.ok());
        EXPECT_EQ(combined.value->size(), 1U);

        const auto either
            = Storage::Select<Device>(Or(Eq(&Device::server_id, uint32_t{1}), Eq(&Device::server_id, uint32_t{2})));
        ASSERT_TRUE(either.ok());
        EXPECT_EQ(either.value->size(), 3U);

        // 空 IN 恒假：匹配 0 行且是成功执行。
        const auto emptyIn = Storage::Select<Device>(In(&Device::server_id, std::vector<uint32_t>{}));
        ASSERT_TRUE(emptyIn.ok());
        EXPECT_TRUE(emptyIn.value->empty());
    }

    /// 排序与数量限制：多排序键按传入顺序、方向保留、Limit 截断、Limit(0) 为 OK+空
    inline void OrderByLimitCase()
    {
        for (uint32_t server = 1; server <= 3U; ++server) {
            for (uint32_t status = 1; status <= 2U; ++status) {
                Device device;
                device.server_id = server;
                device.status = status;
                ASSERT_TRUE(Storage::Insert(device).ok());
            }
        }
        const auto rows = Storage::Select<Device>(
            All<Device>(), OrderBy(&Device::status, SortDirection::DESC),
            OrderBy(&Device::server_id, SortDirection::ASC), Limit(3U));
        ASSERT_TRUE(rows.ok());
        ASSERT_EQ(rows.value->size(), 3U);
        EXPECT_EQ(rows.value->front().status, 2U);
        EXPECT_EQ(rows.value->front().server_id, 1U);
        EXPECT_EQ(rows.value->back().status, 2U);

        const auto none = Storage::Select<Device>(All<Device>(), Limit(0U));
        ASSERT_TRUE(none.ok());
        EXPECT_TRUE(none.value->empty());

        // Limit(1) 字面量路径（SelectOne 固定上限 1 的引擎内表达；SQLite 对 LIMIT 参数
        // 每次执行前重编译，字面量 1 消除该重编译）：多行命中截断为 1 行且排序保持。
        const auto single
            = Storage::Select<Device>(All<Device>(), OrderBy(&Device::server_id, SortDirection::ASC), Limit(1U));
        ASSERT_TRUE(single.ok());
        ASSERT_EQ(single.value->size(), 1U);
        EXPECT_EQ(single.value->front().server_id, 1U);

        // 多参数 WHERE + Limit(1)：WHERE 占位符编号不受字面量上限影响，行集正确。
        const auto filtered = Storage::Select<Device>(
            And(Ge(&Device::server_id, uint32_t{1}), Le(&Device::server_id, uint32_t{3})),
            OrderBy(&Device::id, SortDirection::ASC), Limit(1U));
        ASSERT_TRUE(filtered.ok());
        ASSERT_EQ(filtered.value->size(), 1U);

        // SelectOne 非唯一条件（多行命中取一行）+ OrderBy：确定性首行，limit=1 语义不变。
        const auto oneOfMany = Storage::SelectOne<Device>(
            Eq(&Device::status, uint32_t{1}), OrderBy(&Device::server_id, SortDirection::DESC));
        ASSERT_TRUE(oneOfMany.ok());
        ASSERT_TRUE(oneOfMany.value.has_value());
        EXPECT_EQ(oneOfMany.value->status, 1U);
        EXPECT_EQ(oneOfMany.value->server_id, 3U);

        // 字面量上限路径未命中仍传播 NOT_FOUND。
        const auto stillMissing = Storage::SelectOne<Device>(Eq(&Device::server_id, uint32_t{777}));
        EXPECT_EQ(stillMissing.code, DbCode::NOT_FOUND);
    }

    /// Count 是数据库端聚合：条件计数与全表计数不解码行
    inline void CountCase()
    {
        Device first;
        first.server_id = 5;
        Device second;
        second.server_id = 5;
        Device third;
        third.server_id = 6;
        ASSERT_TRUE(Storage::Insert(first).ok());
        ASSERT_TRUE(Storage::Insert(second).ok());
        ASSERT_TRUE(Storage::Insert(third).ok());
        const auto onServer = Storage::Count<Device>(Eq(&Device::server_id, uint32_t{5}));
        ASSERT_TRUE(onServer.ok());
        EXPECT_EQ(*onServer.value, 2U);
        const auto all = Storage::Count<Device>(All<Device>());
        ASSERT_TRUE(all.ok());
        EXPECT_GE(*all.value, 3U);
        const auto missing = Storage::Count<Device>(Eq(&Device::server_id, uint32_t{77}));
        ASSERT_TRUE(missing.ok());
        EXPECT_EQ(*missing.value, 0U);
    }

    /// CountMatches 条件计数聚合（公共合同，一次读取返回 total/matched）：
    /// 部分匹配/全匹配/零匹配/空集合 {0,0}；All(match) 恒真（matched == total）；
    /// where 与 match 使用不同参数（字符串前缀 + IN + AND/OR 分组 + 空 IN），
    /// 两组参数不得错位；matched 恒不大于 total
    inline void CountMatchesCase()
    {
        Device a;
        a.server_id = 1;
        a.status = 1;
        std::memcpy(a.name, "alpha", sizeof("alpha"));
        Device b;
        b.server_id = 1;
        b.status = 2;
        std::memcpy(b.name, "beta", sizeof("beta"));
        Device c;
        c.server_id = 2;
        c.status = 2;
        std::memcpy(c.name, "gamma", sizeof("gamma"));
        ASSERT_TRUE(Storage::Insert(a).ok());
        ASSERT_TRUE(Storage::Insert(b).ok());
        ASSERT_TRUE(Storage::Insert(c).ok());

        // 部分匹配：server 1 集合 2 行，其中 status=2 的 1 行。
        const auto partial
            = Storage::CountMatches<Device>(Eq(&Device::server_id, uint32_t{1}), Eq(&Device::status, uint32_t{2}));
        ASSERT_TRUE(partial.ok());
        ASSERT_TRUE(partial.value.has_value());
        EXPECT_EQ(partial.value->total, 2U);
        EXPECT_EQ(partial.value->matched, 1U);

        // 全匹配：match 与 where 同条件。
        const auto full
            = Storage::CountMatches<Device>(Eq(&Device::server_id, uint32_t{1}), Eq(&Device::server_id, uint32_t{1}));
        ASSERT_TRUE(full.ok());
        EXPECT_EQ(full.value->total, 2U);
        EXPECT_EQ(full.value->matched, 2U);

        // All(match)：恒真表达式，matched == total。
        const auto always = Storage::CountMatches<Device>(Eq(&Device::server_id, uint32_t{1}), All<Device>());
        ASSERT_TRUE(always.ok());
        EXPECT_EQ(always.value->total, 2U);
        EXPECT_EQ(always.value->matched, 2U);

        // 零匹配：match 在集合内无命中。
        const auto none
            = Storage::CountMatches<Device>(Eq(&Device::server_id, uint32_t{1}), Eq(&Device::server_id, uint32_t{77}));
        ASSERT_TRUE(none.ok());
        EXPECT_EQ(none.value->total, 2U);
        EXPECT_EQ(none.value->matched, 0U);

        // 空集合：where 无命中时两个计数均为 0（OK + {0, 0}，不是失败）。
        const auto emptySet
            = Storage::CountMatches<Device>(Eq(&Device::server_id, uint32_t{99}), Eq(&Device::status, uint32_t{1}));
        ASSERT_TRUE(emptySet.ok());
        EXPECT_EQ(emptySet.value->total, 0U);
        EXPECT_EQ(emptySet.value->matched, 0U);

        // All(where) + 条件 match：全表集合内计数。
        const auto whole = Storage::CountMatches<Device>(All<Device>(), Eq(&Device::status, uint32_t{2}));
        ASSERT_TRUE(whole.ok());
        EXPECT_GE(whole.value->total, 3U);
        EXPECT_EQ(whole.value->matched, 2U);

        // where 用 IN、match 用字符串前缀：不同参数类型混排不错位。
        const auto mixed = Storage::CountMatches<Device>(
            In(&Device::server_id, std::vector<uint32_t>{1, 2}), StartsWith(&Device::name, std::string{"al"}));
        ASSERT_TRUE(mixed.ok());
        EXPECT_EQ(mixed.value->total, 3U);
        EXPECT_EQ(mixed.value->matched, 1U);

        // match 携带 OR 分组：(status=2 OR name 前缀 al) 在 server 1 集合内命中 a、b。
        const auto grouped = Storage::CountMatches<Device>(
            Eq(&Device::server_id, uint32_t{1}),
            Or(Eq(&Device::status, uint32_t{2}), StartsWith(&Device::name, std::string{"al"})));
        ASSERT_TRUE(grouped.ok());
        EXPECT_EQ(grouped.value->total, 2U);
        EXPECT_EQ(grouped.value->matched, 2U);

        // match 携带 AND 分组：全表内 (server 2 AND status 2) 命中 1 行。
        const auto groupedAnd = Storage::CountMatches<Device>(
            All<Device>(), And(Eq(&Device::server_id, uint32_t{2}), Eq(&Device::status, uint32_t{2})));
        ASSERT_TRUE(groupedAnd.ok());
        EXPECT_EQ(groupedAnd.value->matched, 1U);

        // 空 IN 的 match 恒假（0 匹配）；空 IN 的 where 为空集合 {0, 0}。
        const auto emptyInMatch = Storage::CountMatches<Device>(
            Eq(&Device::server_id, uint32_t{1}), In(&Device::server_id, std::vector<uint32_t>{}));
        ASSERT_TRUE(emptyInMatch.ok());
        EXPECT_EQ(emptyInMatch.value->total, 2U);
        EXPECT_EQ(emptyInMatch.value->matched, 0U);
        const auto emptyInWhere = Storage::CountMatches<Device>(
            In(&Device::server_id, std::vector<uint32_t>{}), Eq(&Device::status, uint32_t{2}));
        ASSERT_TRUE(emptyInWhere.ok());
        EXPECT_EQ(emptyInWhere.value->total, 0U);
        EXPECT_EQ(emptyInWhere.value->matched, 0U);

        // matched 不大于 total（全部已取样本统一复核）。
        EXPECT_LE(partial.value->matched, partial.value->total);
        EXPECT_LE(mixed.value->matched, mixed.value->total);
        EXPECT_LE(grouped.value->matched, grouped.value->total);
    }

    /// CountMatches 的 uint64 高半区条件：无符号语义在聚合内保持（SQLite 位型
    /// 展开与 MariaDB BIGINT UNSIGNED 同一计数），高半区参数不与 where 错位
    inline void CountMatchesUint64HalfCase()
    {
        constexpr uint64_t kMid = 1ULL << 63U;
        constexpr uint64_t kMax = ~0ULL;
        BigValue low;
        low.value = 0;
        BigValue high;
        high.value = kMid;
        BigValue top;
        top.value = kMax;
        ASSERT_TRUE(Storage::Insert(low).ok());
        ASSERT_TRUE(Storage::Insert(high).ok());
        ASSERT_TRUE(Storage::Insert(top).ok());

        const auto above = Storage::CountMatches<BigValue>(Gt(&BigValue::value, kMid - 1U), Eq(&BigValue::value, kMax));
        ASSERT_TRUE(above.ok());
        EXPECT_EQ(above.value->total, 2U);
        EXPECT_EQ(above.value->matched, 1U);

        const auto highHalf = Storage::CountMatches<BigValue>(All<BigValue>(), Ge(&BigValue::value, kMid));
        ASSERT_TRUE(highHalf.ok());
        EXPECT_GE(highHalf.value->total, 3U);
        EXPECT_EQ(highHalf.value->matched, 2U);
    }

    /// CountMatches 的动态 PID scope 隔离：统计只覆盖本作用域物理表，跨 scope
    /// 数据不混入（total 与 matched 都不泄漏）
    inline void CountMatchesDynamicScopeCase()
    {
        const uint64_t pidA = 201;
        const uint64_t pidB = 202;
        DynTask a1;
        a1.owner = pidA;
        a1.seq = 1;
        DynTask a2;
        a2.owner = pidA;
        a2.seq = 2;
        DynTask b1;
        b1.owner = pidB;
        b1.seq = 1;
        ASSERT_TRUE(Storage::Insert(TableScope::Process(pidA), a1).ok());
        ASSERT_TRUE(Storage::Insert(TableScope::Process(pidA), a2).ok());
        ASSERT_TRUE(Storage::Insert(TableScope::Process(pidB), b1).ok());

        const auto onlyA
            = Storage::CountMatches<DynTask>(TableScope::Process(pidA), All<DynTask>(), Eq(&DynTask::seq, uint32_t{1}));
        ASSERT_TRUE(onlyA.ok());
        EXPECT_EQ(onlyA.value->total, 2U);
        EXPECT_EQ(onlyA.value->matched, 1U);

        const auto onlyB = Storage::CountMatches<DynTask>(TableScope::Process(pidB), All<DynTask>(), All<DynTask>());
        ASSERT_TRUE(onlyB.ok());
        EXPECT_EQ(onlyB.value->total, 1U);
        EXPECT_EQ(onlyB.value->matched, 1U);
    }

    /// Update/Delete 命中行数返回；零行是 OK + value 0 + affectedRows 0
    inline void AffectedRowsCase()
    {
        Device device;
        device.server_id = 9;
        device.status = 1;
        ASSERT_TRUE(Storage::Insert(device).ok());
        const auto changed = Storage::Update<Device>(Eq(&Device::id, device.id), Set(&Device::status, uint32_t{3}));
        ASSERT_TRUE(changed.ok());
        EXPECT_EQ(*changed.value, 1U);
        EXPECT_EQ(changed.affectedRows, 1U);

        const auto untouched
            = Storage::Update<Device>(Eq(&Device::id, uint64_t{424242}), Set(&Device::status, uint32_t{3}));
        ASSERT_TRUE(untouched.ok());
        ASSERT_TRUE(untouched.value.has_value());
        EXPECT_EQ(*untouched.value, 0U);
        EXPECT_EQ(untouched.affectedRows, 0U);

        const auto removed = Storage::Delete<Device>(Eq(&Device::id, device.id));
        ASSERT_TRUE(removed.ok());
        EXPECT_EQ(*removed.value, 1U);
        const auto removedMissing = Storage::Delete<Device>(Eq(&Device::id, device.id));
        ASSERT_TRUE(removedMissing.ok());
        EXPECT_EQ(*removedMissing.value, 0U);
    }

    /// Clear 清空整表；RESET_SEQUENCE 重置自增序列
    inline void ClearCase()
    {
        Device device;
        device.server_id = 3;
        ASSERT_TRUE(Storage::Insert(device).ok());
        const auto before = Storage::Count<Device>(All<Device>());
        ASSERT_TRUE(before.ok());
        EXPECT_GE(*before.value, 1U);
        ASSERT_TRUE(Storage::Clear<Device>(ClearMode::RESET_SEQUENCE).ok());
        const auto after = Storage::Count<Device>(All<Device>());
        ASSERT_TRUE(after.ok());
        EXPECT_EQ(*after.value, 0U);
        Device fresh;
        fresh.server_id = 3;
        ASSERT_TRUE(Storage::Insert(fresh).ok());
        EXPECT_EQ(fresh.id, 1U);
    }

    /// 动态 scope 表：不同 PID scope 数据隔离
    inline void DynamicScopeCase()
    {
        const uint64_t pidA = 101;
        const uint64_t pidB = 102;
        DynTask taskA;
        taskA.owner = pidA;
        taskA.seq = 1;
        ASSERT_TRUE(Storage::Insert(TableScope::Process(pidA), taskA).ok());
        DynTask taskB;
        taskB.owner = pidB;
        taskB.seq = 2;
        ASSERT_TRUE(Storage::Insert(TableScope::Process(pidB), taskB).ok());

        const auto onlyA = Storage::Select<DynTask>(TableScope::Process(pidA), All<DynTask>());
        ASSERT_TRUE(onlyA.ok());
        ASSERT_EQ(onlyA.value->size(), 1U);
        EXPECT_EQ(onlyA.value->front().owner, pidA);

        const auto onlyB = Storage::SelectOne<DynTask>(
            TableScope::Process(pidB), All<DynTask>(), OrderBy(&DynTask::id, SortDirection::ASC));
        ASSERT_TRUE(onlyB.ok());
        EXPECT_EQ(onlyB.value->owner, pidB);
    }

    /// Unique 约束：重复唯一键插入被拒绝，错误原样传播
    inline void UniqueConstraintCase()
    {
        UniqueRow first;
        first.code = 42;
        ASSERT_TRUE(Storage::Insert(first).ok());
        UniqueRow duplicate;
        duplicate.code = 42;
        const auto rejected = Storage::Insert(duplicate);
        EXPECT_FALSE(rejected.ok());
        EXPECT_NE(rejected.code, DbCode::OK);
        UniqueRow second;
        second.code = 43;
        ASSERT_TRUE(Storage::Insert(second).ok());
    }

    /// 多行 Update 的内部唯一冲突：一次条件更新把两行唯一字段改成同一个新值时，
    /// 后端必须在本次语句内拒绝（CONFLICT）且零行变更——两条记录与唯一索引保持
    /// 原值（单语句原子性：不得出现"唯一索引只留最后写入者、两条业务记录都已
    /// 重复"的部分写入形态）
    inline void MultiRowUniqueConflictCase()
    {
        UniqueRow first;
        first.code = 1;
        ASSERT_TRUE(Storage::Insert(first).ok());
        UniqueRow second;
        second.code = 2;
        ASSERT_TRUE(Storage::Insert(second).ok());

        const auto rejected = Storage::Update<UniqueRow>(
            In(&UniqueRow::id, std::vector<uint64_t>{first.id, second.id}), Set(&UniqueRow::code, uint32_t{77}));
        EXPECT_EQ(rejected.code, DbCode::CONFLICT);

        // 两条记录及唯一索引保持原值（零行变更）
        const auto rows = Storage::Select<UniqueRow>(All<UniqueRow>());
        ASSERT_TRUE(rows.ok());
        ASSERT_EQ(rows.value->size(), 2U);
        for (const auto& row : *rows.value) {
            EXPECT_TRUE(row.code == 1U || row.code == 2U) << "唯一字段必须保持原值";
        }
    }

    /// 自增主键显式高水位（SQLite AUTOINCREMENT 语义）：显式插入 id=100 后，
    /// 下一次自动生成主键必须大于 100；再显式插入较小 id=50 不得把序列降到 50；
    /// RESET_SEQUENCE 后重新从 1 开始
    inline void ExplicitAutoincrementHighWaterCase()
    {
        Device explicitHigh;
        explicitHigh.id = 100U;
        explicitHigh.server_id = 1;
        ASSERT_TRUE(Storage::Insert(explicitHigh).ok());
        EXPECT_EQ(explicitHigh.id, 100U);

        Device afterHigh;
        afterHigh.server_id = 1;
        ASSERT_TRUE(Storage::Insert(afterHigh).ok());
        EXPECT_EQ(afterHigh.id, 101U);

        Device explicitLow;
        explicitLow.id = 50U;
        explicitLow.server_id = 1;
        ASSERT_TRUE(Storage::Insert(explicitLow).ok());
        EXPECT_EQ(explicitLow.id, 50U);

        Device afterLow;
        afterLow.server_id = 1;
        ASSERT_TRUE(Storage::Insert(afterLow).ok());
        EXPECT_EQ(afterLow.id, 102U);

        ASSERT_TRUE(Storage::Clear<Device>(ClearMode::RESET_SEQUENCE).ok());
        Device fresh;
        fresh.server_id = 1;
        ASSERT_TRUE(Storage::Insert(fresh).ok());
        EXPECT_EQ(fresh.id, 1U);
    }

    /// 级联外键：删除父行时子行级联删除
    inline void ForeignKeyCascadeCase()
    {
        UniqueRow parent;
        parent.code = 500;
        ASSERT_TRUE(Storage::Insert(parent).ok());
        ChildRow child;
        child.parent_id = parent.id;
        child.amount = 7;
        ASSERT_TRUE(Storage::Insert(child).ok());

        const auto before = Storage::Count<ChildRow>(Eq(&ChildRow::parent_id, parent.id));
        ASSERT_TRUE(before.ok());
        EXPECT_EQ(*before.value, 1U);

        ASSERT_TRUE(Storage::Delete<UniqueRow>(Eq(&UniqueRow::id, parent.id)).ok());
        const auto after = Storage::Count<ChildRow>(Eq(&ChildRow::parent_id, parent.id));
        ASSERT_TRUE(after.ok());
        EXPECT_EQ(*after.value, 0U);
    }

    /// 受限外键：仍有引用时父删除冲突；删除引用后父删除成功。
    inline void ForeignKeyRestrictCase()
    {
        UniqueRow parent;
        parent.code = 501;
        ASSERT_TRUE(Storage::Insert(parent).ok());
        GuardRow guard;
        guard.parent_id = parent.id;
        guard.amount = 8;
        ASSERT_TRUE(Storage::Insert(guard).ok());

        const auto blocked = Storage::Delete<UniqueRow>(Eq(&UniqueRow::id, parent.id));
        EXPECT_EQ(blocked.code, DbCode::CONFLICT);
        EXPECT_TRUE(Storage::SelectOne<UniqueRow>(Eq(&UniqueRow::id, parent.id)).ok());
        EXPECT_TRUE(Storage::SelectOne<GuardRow>(Eq(&GuardRow::id, guard.id)).ok());

        ASSERT_TRUE(Storage::Delete<GuardRow>(Eq(&GuardRow::id, guard.id)).ok());
        ASSERT_TRUE(Storage::Delete<UniqueRow>(Eq(&UniqueRow::id, parent.id)).ok());
        EXPECT_EQ(Storage::SelectOne<UniqueRow>(Eq(&UniqueRow::id, parent.id)).code, DbCode::NOT_FOUND);
    }

    /// SelectMany 批量读：多作用域结果与逐作用域 Select 完全一致（含空作用域
    /// 占位、每作用域排序与 Limit 独立应用）；空列表平凡成功。后端无关期望：
    /// 继承默认循环的后端与合批实现的后端必须给出同一结果
    inline void SelectManyCase()
    {
        const std::vector<uint64_t> pids{11U, 22U, 33U};
        // 逐作用域写入：pid11 两行、pid22 空、pid33 一行
        for (const uint64_t pid : pids) {
            const std::size_t rows = pid == 22U ? 0U : (pid == 11U ? 2U : 1U);
            for (std::size_t r = 0; r < rows; ++r) {
                DynTask task;
                task.owner = pid + r;
                ASSERT_TRUE(Storage::Insert(TableScope::Process(pid), task).ok());
            }
        }
        std::vector<TableScope> scopes;
        for (const uint64_t pid : pids) {
            scopes.emplace_back(TableScope::Process(pid));
        }
        // 批量（All + 每作用域 id 升序）与逐次 Select 逐行一致
        const auto batched
            = Storage::SelectMany<DynTask>(scopes, All<DynTask>(), OrderBy(&DynTask::id, SortDirection::ASC));
        ASSERT_TRUE(batched.ok());
        ASSERT_EQ(batched.value->size(), 3U);
        for (std::size_t i = 0; i < scopes.size(); ++i) {
            const auto single
                = Storage::Select<DynTask>(scopes[i], All<DynTask>(), OrderBy(&DynTask::id, SortDirection::ASC));
            ASSERT_TRUE(single.ok()) << single.diagnostic;
            ASSERT_EQ(batched.value->at(i).size(), single.value->size());
            for (std::size_t row = 0; row < single.value->size(); ++row) {
                EXPECT_EQ(batched.value->at(i)[row].id, single.value->at(row).id);
                EXPECT_EQ(batched.value->at(i)[row].owner, single.value->at(row).owner);
            }
        }
        EXPECT_TRUE(batched.value->at(1).empty()); // 空作用域不漏（结果仍占位）

        // Limit 每作用域独立应用
        const auto limited = Storage::SelectMany<DynTask>(
            scopes, All<DynTask>(), OrderBy(&DynTask::id, SortDirection::ASC), Limit(1U));
        ASSERT_TRUE(limited.ok());
        ASSERT_EQ(limited.value->size(), 3U);
        EXPECT_EQ(limited.value->at(0).size(), 1U); // pid11 两行截断为 1
        EXPECT_TRUE(limited.value->at(1).empty());
        EXPECT_EQ(limited.value->at(2).size(), 1U);

        // 空列表平凡成功（零命令/零查询）
        const auto none = Storage::SelectMany<DynTask>(std::vector<TableScope>{}, All<DynTask>());
        ASSERT_TRUE(none.ok());
        EXPECT_TRUE(none.value->empty());
    }

    /// 事务：提交持久、回滚丢弃、析构自动回滚、空事务可提交
    inline void TransactionCase()
    {
        auto transaction = Storage::BeginTransaction();
        ASSERT_TRUE(transaction.ok());
        UniqueRow committed;
        committed.code = 600;
        ASSERT_TRUE(Storage::Insert(committed, *transaction.value).ok());
        ASSERT_TRUE(transaction.value->Commit().ok());

        const auto visible = Storage::SelectOne<UniqueRow>(Eq(&UniqueRow::code, uint32_t{600}));
        ASSERT_TRUE(visible.ok());
        EXPECT_EQ(visible.value->code, 600U);

        auto rollback = Storage::BeginTransaction();
        ASSERT_TRUE(rollback.ok());
        UniqueRow abandoned;
        abandoned.code = 601;
        ASSERT_TRUE(Storage::Insert(abandoned, *rollback.value).ok());
        ASSERT_TRUE(rollback.value->Rollback().ok());
        const auto rolledBack = Storage::SelectOne<UniqueRow>(Eq(&UniqueRow::code, uint32_t{601}));
        EXPECT_EQ(rolledBack.code, DbCode::NOT_FOUND);

        // 未提交事务析构自动回滚。
        {
            auto destructed = Storage::BeginTransaction();
            ASSERT_TRUE(destructed.ok());
            UniqueRow pending;
            pending.code = 602;
            ASSERT_TRUE(Storage::Insert(pending, *destructed.value).ok());
        }
        const auto afterDestruct = Storage::SelectOne<UniqueRow>(Eq(&UniqueRow::code, uint32_t{602}));
        EXPECT_EQ(afterDestruct.code, DbCode::NOT_FOUND);

        // 空事务可正常提交。
        auto empty = Storage::BeginTransaction();
        ASSERT_TRUE(empty.ok());
        EXPECT_TRUE(empty.value->Commit().ok());
    }

    /// uint64 完整位型与 BLOB 字节序列无损往返
    inline void Uint64BlobCase()
    {
        BigValue row;
        row.value = 0xFFFFFFFFFFFFFFFFULL; // 最大值位型：不得截断到 INT64_MAX
        ASSERT_TRUE(Storage::Insert(row).ok());
        const auto read = Storage::SelectOne<BigValue>(Eq(&BigValue::id, row.id));
        ASSERT_TRUE(read.ok());
        EXPECT_EQ(read.value->value, 0xFFFFFFFFFFFFFFFFULL);

        BlobRow blob;
        blob.owner = 3;
        blob.bytes[0] = 0xFF;
        blob.bytes[11] = 0x80;
        ASSERT_TRUE(Storage::Insert(blob).ok());
        const auto blobBack = Storage::SelectOne<BlobRow>(Eq(&BlobRow::id, blob.id));
        ASSERT_TRUE(blobBack.ok());
        EXPECT_EQ(blobBack.value->bytes[0], 0xFF);
        EXPECT_EQ(blobBack.value->bytes[11], 0x80);
    }

    /// 统一错误结果：错误码/影响行数/诊断随 DbResult 传播，不抛异常
    inline void UnifiedErrorCase()
    {
        // 无条件全表更新/删除被合同层拒绝（安全拦截），错误码一致。
        const auto updateAll = Storage::Update<Device>(All<Device>(), Set(&Device::status, uint32_t{1}));
        EXPECT_EQ(updateAll.code, DbCode::INVALID_ARGUMENT);
        const auto deleteAll = Storage::Delete<Device>(All<Device>());
        EXPECT_EQ(deleteAll.code, DbCode::INVALID_ARGUMENT);
        EXPECT_FALSE(deleteAll.diagnostic.empty());
    }

    /// INCREMENT 增量编码合同（纯表达式层，不触引擎）：无符号字段的负增量必须
    /// 以 int64 保号到达后端（修复前经无符号类别编码折叠为 UINT64_MAX，MariaDB
    /// 以 errno 1690 拒绝合法减法）；无符号正增量保持 uint64；有符号字段沿用
    /// 既有编码规则（int64 直通、uint64 经类别转换），Set/谓词规则不变。
    inline void IncrementEncodingCase()
    {
        const auto negative = Increment(&CounterRow::ucount, int64_t{-1});
        EXPECT_TRUE(negative.increment);
        const auto* negativeDelta = std::get_if<int64_t>(&negative.value);
        ASSERT_NE(negativeDelta, nullptr);
        EXPECT_EQ(*negativeDelta, -1);

        const auto positive = Increment(&CounterRow::ucount, uint64_t{1});
        EXPECT_TRUE(positive.increment);
        const auto* positiveDelta = std::get_if<uint64_t>(&positive.value);
        ASSERT_NE(positiveDelta, nullptr);
        EXPECT_EQ(*positiveDelta, 1U);

        const auto signedPositive = Increment(&CounterRow::ucount, int64_t{5});
        const auto* signedPositiveDelta = std::get_if<int64_t>(&signedPositive.value);
        ASSERT_NE(signedPositiveDelta, nullptr);
        EXPECT_EQ(*signedPositiveDelta, 5);

        const auto signedField = Increment(&CounterRow::scount, int64_t{-1});
        const auto* signedFieldDelta = std::get_if<int64_t>(&signedField.value);
        ASSERT_NE(signedFieldDelta, nullptr);
        EXPECT_EQ(*signedFieldDelta, -1);

        const auto signedFieldUnsigned = Increment(&CounterRow::scount, uint64_t{7});
        const auto* convertedDelta = std::get_if<int64_t>(&signedFieldUnsigned.value);
        ASSERT_NE(convertedDelta, nullptr);
        EXPECT_EQ(*convertedDelta, 7);
    }

    /// INCREMENT 基本算术（三后端一致）：无符号 2-1=1、1+1=2、零增量幂等、
    /// 有符号 10-3=7、7+3=10、有符号正增量到无符号字段 2+5=7、多行同减
    inline void IncrementArithmeticCase()
    {
        CounterRow row;
        row.ucount = 2;
        row.scount = 10;
        ASSERT_TRUE(Storage::Insert(row).ok());
        const auto id = row.id;

        ASSERT_TRUE(
            Storage::Update<CounterRow>(Eq(&CounterRow::id, id), Increment(&CounterRow::ucount, int64_t{-1})).ok());
        auto back = Storage::SelectOne<CounterRow>(Eq(&CounterRow::id, id));
        ASSERT_TRUE(back.ok());
        ASSERT_TRUE(back.value.has_value());
        EXPECT_EQ(back.value->ucount, 1U);

        ASSERT_TRUE(
            Storage::Update<CounterRow>(Eq(&CounterRow::id, id), Increment(&CounterRow::ucount, uint64_t{1})).ok());
        back = Storage::SelectOne<CounterRow>(Eq(&CounterRow::id, id));
        ASSERT_TRUE(back.ok());
        EXPECT_EQ(back.value->ucount, 2U);

        ASSERT_TRUE(
            Storage::Update<CounterRow>(Eq(&CounterRow::id, id), Increment(&CounterRow::ucount, uint64_t{0})).ok());
        back = Storage::SelectOne<CounterRow>(Eq(&CounterRow::id, id));
        ASSERT_TRUE(back.ok());
        EXPECT_EQ(back.value->ucount, 2U);

        ASSERT_TRUE(
            Storage::Update<CounterRow>(Eq(&CounterRow::id, id), Increment(&CounterRow::scount, int64_t{-3})).ok());
        ASSERT_TRUE(
            Storage::Update<CounterRow>(Eq(&CounterRow::id, id), Increment(&CounterRow::scount, int64_t{3})).ok());
        back = Storage::SelectOne<CounterRow>(Eq(&CounterRow::id, id));
        ASSERT_TRUE(back.ok());
        EXPECT_EQ(back.value->scount, 10);

        ASSERT_TRUE(
            Storage::Update<CounterRow>(Eq(&CounterRow::id, id), Increment(&CounterRow::ucount, int64_t{5})).ok());
        back = Storage::SelectOne<CounterRow>(Eq(&CounterRow::id, id));
        ASSERT_TRUE(back.ok());
        EXPECT_EQ(back.value->ucount, 7U);

        CounterRow second;
        second.ucount = 9;
        second.scount = 0;
        ASSERT_TRUE(Storage::Insert(second).ok());
        ASSERT_TRUE(
            Storage::Update<CounterRow>(Gt(&CounterRow::id, uint64_t{0}), Increment(&CounterRow::ucount, int64_t{-1}))
                .ok());
        back = Storage::SelectOne<CounterRow>(Eq(&CounterRow::id, id));
        ASSERT_TRUE(back.ok());
        EXPECT_EQ(back.value->ucount, 6U);
        const auto secondBack = Storage::SelectOne<CounterRow>(Eq(&CounterRow::id, second.id));
        ASSERT_TRUE(secondBack.ok());
        EXPECT_EQ(secondBack.value->ucount, 8U);
    }

    /// INCREMENT 边界合同（三后端一致）：零减一、无符号上界加一、有符号上下界
    /// 均为明确错误且数据保持（不回绕、不截断、不浮点化、不静默成功）；
    /// 多行更新中任一行越界则整条语句原子失败，全部行值保持
    inline void IncrementBoundaryCase()
    {
        CounterRow row;
        row.ucount = 0;
        row.scount = 0;
        ASSERT_TRUE(Storage::Insert(row).ok());
        const auto id = row.id;

        const auto underflow
            = Storage::Update<CounterRow>(Eq(&CounterRow::id, id), Increment(&CounterRow::ucount, int64_t{-1}));
        EXPECT_FALSE(underflow.ok());
        auto back = Storage::SelectOne<CounterRow>(Eq(&CounterRow::id, id));
        ASSERT_TRUE(back.ok());
        EXPECT_EQ(back.value->ucount, 0U);

        ASSERT_TRUE(Storage::Update<CounterRow>(Eq(&CounterRow::id, id), Set(&CounterRow::ucount, UINT64_MAX)).ok());
        const auto overflow
            = Storage::Update<CounterRow>(Eq(&CounterRow::id, id), Increment(&CounterRow::ucount, uint64_t{1}));
        EXPECT_FALSE(overflow.ok());
        back = Storage::SelectOne<CounterRow>(Eq(&CounterRow::id, id));
        ASSERT_TRUE(back.ok());
        EXPECT_EQ(back.value->ucount, UINT64_MAX);

        ASSERT_TRUE(Storage::Update<CounterRow>(Eq(&CounterRow::id, id), Set(&CounterRow::scount, INT64_MAX)).ok());
        const auto signedOverflow
            = Storage::Update<CounterRow>(Eq(&CounterRow::id, id), Increment(&CounterRow::scount, int64_t{1}));
        EXPECT_FALSE(signedOverflow.ok());
        back = Storage::SelectOne<CounterRow>(Eq(&CounterRow::id, id));
        ASSERT_TRUE(back.ok());
        EXPECT_EQ(back.value->scount, INT64_MAX);

        ASSERT_TRUE(Storage::Update<CounterRow>(Eq(&CounterRow::id, id), Set(&CounterRow::scount, INT64_MIN)).ok());
        const auto signedUnderflow
            = Storage::Update<CounterRow>(Eq(&CounterRow::id, id), Increment(&CounterRow::scount, int64_t{-1}));
        EXPECT_FALSE(signedUnderflow.ok());
        back = Storage::SelectOne<CounterRow>(Eq(&CounterRow::id, id));
        ASSERT_TRUE(back.ok());
        EXPECT_EQ(back.value->scount, INT64_MIN);

        // 多行原子性：一行可减（MAX→MAX-1 合法）、一行下溢（0-1）→ 整条语句
        // 原子失败，两行值都保持
        CounterRow underflowRow;
        underflowRow.ucount = 0;
        underflowRow.scount = 0;
        ASSERT_TRUE(Storage::Insert(underflowRow).ok());
        const auto multi = Storage::Update<CounterRow>(
            Gt(&CounterRow::id, uint64_t{0}), Increment(&CounterRow::ucount, int64_t{-1}));
        EXPECT_FALSE(multi.ok());
        back = Storage::SelectOne<CounterRow>(Eq(&CounterRow::id, underflowRow.id));
        ASSERT_TRUE(back.ok());
        EXPECT_EQ(back.value->ucount, 0U);
        back = Storage::SelectOne<CounterRow>(Eq(&CounterRow::id, id));
        ASSERT_TRUE(back.ok());
        EXPECT_EQ(back.value->ucount, UINT64_MAX);
    }

    /// INCREMENT 并发原子减法：两个线程对同一行各做 N 次单语句 -1，最终值必须
    /// 恰为初值 - 2N（单语句 field = field - ? 原子形态不丢更新；读后写会丢）
    inline void IncrementConcurrentCase()
    {
        constexpr int kRoundsPerThread = 400;
        CounterRow row;
        row.ucount = 2U * static_cast<uint64_t>(kRoundsPerThread);
        row.scount = 0;
        ASSERT_TRUE(Storage::Insert(row).ok());
        const auto id = row.id;

        std::atomic<int> failures{0};
        const auto worker = [&failures, id]() {
            for (int round = 0; round < kRoundsPerThread; ++round) {
                const auto updated
                    = Storage::Update<CounterRow>(Eq(&CounterRow::id, id), Increment(&CounterRow::ucount, int64_t{-1}));
                if (!updated.ok()) {
                    ++failures;
                }
            }
        };
        std::thread first(worker);
        std::thread second(worker);
        first.join();
        second.join();
        EXPECT_EQ(failures.load(), 0);

        const auto back = Storage::SelectOne<CounterRow>(Eq(&CounterRow::id, id));
        ASSERT_TRUE(back.ok());
        ASSERT_TRUE(back.value.has_value());
        EXPECT_EQ(back.value->ucount, 0U);
    }
    /// 嵌套条件分组：括号语义由 openGroupsBefore/closeGroupsAfter 承载，
    /// (A AND B) OR (C AND D) 与 A AND (B OR C) 的行集各自正确（SQL 括号渲染与
    /// 内存求值同形的公共合同）
    inline void NestedConditionCase()
    {
        Device a;
        a.server_id = 1;
        a.status = 1;
        Device b;
        b.server_id = 1;
        b.status = 2;
        Device c;
        c.server_id = 2;
        c.status = 2;
        Device d;
        d.server_id = 2;
        d.status = 1;
        ASSERT_TRUE(Storage::Insert(a).ok());
        ASSERT_TRUE(Storage::Insert(b).ok());
        ASSERT_TRUE(Storage::Insert(c).ok());
        ASSERT_TRUE(Storage::Insert(d).ok());

        const auto eitherGroup = Storage::Select<Device>(
            Or(And(Eq(&Device::server_id, uint32_t{1}), Eq(&Device::status, uint32_t{1})),
               And(Eq(&Device::server_id, uint32_t{2}), Eq(&Device::status, uint32_t{2}))));
        ASSERT_TRUE(eitherGroup.ok());
        ASSERT_EQ(eitherGroup.value->size(), 2U);
        EXPECT_EQ(eitherGroup.value->front().id, a.id);
        EXPECT_EQ(eitherGroup.value->back().id, c.id);

        const auto serverOneAny = Storage::Select<Device>(
            And(Eq(&Device::server_id, uint32_t{1}),
                Or(Eq(&Device::status, uint32_t{1}), Eq(&Device::status, uint32_t{2}))));
        ASSERT_TRUE(serverOneAny.ok());
        ASSERT_EQ(serverOneAny.value->size(), 2U);
        EXPECT_EQ(serverOneAny.value->front().id, a.id);
        EXPECT_EQ(serverOneAny.value->back().id, b.id);

        // 嵌套与顶层连接混排：server=1 全部 OR (server=2 AND status=2) → a, b, c
        const auto mixed = Storage::Select<Device>(
            Or(Eq(&Device::server_id, uint32_t{1}),
               And(Eq(&Device::server_id, uint32_t{2}), Eq(&Device::status, uint32_t{2}))));
        ASSERT_TRUE(mixed.ok());
        ASSERT_EQ(mixed.value->size(), 3U);
    }

    /// PREFIX 模式字符转义：值含 %/_/\ 时前缀匹配按字面量进行（转义模式由
    /// 绑定值承载；转义产物临时字符串生命周期由各后端拥有化，不得悬空）
    inline void PrefixEscapeCase()
    {
        Device percent;
        std::memcpy(percent.name, "a%b", sizeof("a%b"));
        percent.server_id = 1;
        Device underscore;
        std::memcpy(underscore.name, "a_b", sizeof("a_b"));
        underscore.server_id = 2;
        Device backslash;
        std::memcpy(backslash.name, "a\\b", sizeof("a\\b"));
        backslash.server_id = 3;
        Device plain;
        std::memcpy(plain.name, "ab", sizeof("ab"));
        plain.server_id = 4;
        ASSERT_TRUE(Storage::Insert(percent).ok());
        ASSERT_TRUE(Storage::Insert(underscore).ok());
        ASSERT_TRUE(Storage::Insert(backslash).ok());
        ASSERT_TRUE(Storage::Insert(plain).ok());

        const auto percentPrefix = Storage::Select<Device>(StartsWith(&Device::name, std::string{"a%"}));
        ASSERT_TRUE(percentPrefix.ok());
        ASSERT_EQ(percentPrefix.value->size(), 1U);
        EXPECT_EQ(percentPrefix.value->front().server_id, 1U);

        const auto underscorePrefix = Storage::Select<Device>(StartsWith(&Device::name, std::string{"a_"}));
        ASSERT_TRUE(underscorePrefix.ok());
        ASSERT_EQ(underscorePrefix.value->size(), 1U);
        EXPECT_EQ(underscorePrefix.value->front().server_id, 2U);

        const auto backslashPrefix = Storage::Select<Device>(StartsWith(&Device::name, std::string{"a\\"}));
        ASSERT_TRUE(backslashPrefix.ok());
        ASSERT_EQ(backslashPrefix.value->size(), 1U);
        EXPECT_EQ(backslashPrefix.value->front().server_id, 3U);

        const auto plainPrefix = Storage::Select<Device>(StartsWith(&Device::name, std::string{"a"}));
        ASSERT_TRUE(plainPrefix.ok());
        ASSERT_EQ(plainPrefix.value->size(), 4U);
    }

    /// uint64 跨 2^63 边界：高半区值的等值/范围/排序按无符号语义执行
    /// （SQLite 位型展开与 MariaDB BIGINT UNSIGNED 必须给出同一行集）
    inline void Uint64HalfCase()
    {
        constexpr uint64_t kMid = 1ULL << 63U;
        constexpr uint64_t kMax = ~0ULL;
        BigValue low;
        low.value = 0;
        BigValue mid;
        mid.value = kMid - 1U;
        BigValue high;
        high.value = kMid;
        BigValue top;
        top.value = kMax;
        ASSERT_TRUE(Storage::Insert(low).ok());
        ASSERT_TRUE(Storage::Insert(mid).ok());
        ASSERT_TRUE(Storage::Insert(high).ok());
        ASSERT_TRUE(Storage::Insert(top).ok());

        const auto above = Storage::Select<BigValue>(Gt(&BigValue::value, kMid - 1U));
        ASSERT_TRUE(above.ok());
        ASSERT_EQ(above.value->size(), 2U);

        const auto below = Storage::Select<BigValue>(Le(&BigValue::value, kMid - 1U));
        ASSERT_TRUE(below.ok());
        ASSERT_EQ(below.value->size(), 2U);

        const auto exactHigh = Storage::SelectOne<BigValue>(Eq(&BigValue::value, kMid));
        ASSERT_TRUE(exactHigh.ok());
        EXPECT_EQ(exactHigh.value->value, kMid);

        const auto exactTop = Storage::SelectOne<BigValue>(Eq(&BigValue::value, kMax));
        ASSERT_TRUE(exactTop.ok());
        EXPECT_EQ(exactTop.value->value, kMax);

        const auto ordered = Storage::Select<BigValue>(All<BigValue>(), OrderBy(&BigValue::value, SortDirection::ASC));
        ASSERT_TRUE(ordered.ok());
        ASSERT_EQ(ordered.value->size(), 4U);
        EXPECT_EQ(ordered.value->at(0).value, 0U);
        EXPECT_EQ(ordered.value->at(1).value, kMid - 1U);
        EXPECT_EQ(ordered.value->at(2).value, kMid);
        EXPECT_EQ(ordered.value->at(3).value, kMax);
    }

    /// UPDATE 参数顺序：SET 值先于 WHERE 值绑定（多 SET 项 + IN 多值 WHERE +
    /// PREFIX 条件混排），值与谓词互不串位；未命中行不受影响
    inline void UpdateParameterOrderCase()
    {
        Device a;
        std::memcpy(a.name, "alpha", sizeof("alpha"));
        a.server_id = 1;
        a.status = 1;
        Device b;
        std::memcpy(b.name, "beta", sizeof("beta"));
        b.server_id = 1;
        b.status = 2;
        Device c;
        std::memcpy(c.name, "gamma", sizeof("gamma"));
        c.server_id = 2;
        c.status = 3;
        ASSERT_TRUE(Storage::Insert(a).ok());
        ASSERT_TRUE(Storage::Insert(b).ok());
        ASSERT_TRUE(Storage::Insert(c).ok());

        // 两个 SET 值在前、IN 的两个 WHERE 值在后：命中 a/b，c 不动。
        const auto updated = Storage::Update<Device>(
            In(&Device::id, std::vector<uint64_t>{a.id, b.id}), Set(&Device::status, uint32_t{7}),
            Set(&Device::name, std::string{"renamed"}));
        ASSERT_TRUE(updated.ok());
        ASSERT_TRUE(updated.value.has_value());
        EXPECT_EQ(*updated.value, 2U);
        const auto renamed = Storage::Select<Device>(Eq(&Device::status, uint32_t{7}));
        ASSERT_TRUE(renamed.ok());
        ASSERT_EQ(renamed.value->size(), 2U);
        for (const auto& row : *renamed.value) {
            EXPECT_EQ(std::string(row.name), "renamed");
        }
        const auto untouched = Storage::SelectOne<Device>(Eq(&Device::id, c.id));
        ASSERT_TRUE(untouched.ok());
        EXPECT_EQ(untouched.value->status, 3U);
        EXPECT_EQ(std::string(untouched.value->name), "gamma");

        // PREFIX 条件的转义模式绑定在 SET 值之后：仅命中 gamma 行。
        const auto prefixed
            = Storage::Update<Device>(StartsWith(&Device::name, std::string{"gam"}), Set(&Device::status, uint32_t{9}));
        ASSERT_TRUE(prefixed.ok());
        ASSERT_TRUE(prefixed.value.has_value());
        EXPECT_EQ(*prefixed.value, 1U);
        const auto targeted = Storage::SelectOne<Device>(Eq(&Device::id, c.id));
        ASSERT_TRUE(targeted.ok());
        EXPECT_EQ(targeted.value->status, 9U);
    }

    /// 同值排序与主键补充：非主键排序键全同值时，同值行按主键升序（可复现），
    /// 排序方向不影响 tie-breaker 方向
    inline void SameKeyTieBreakCase()
    {
        constexpr std::size_t kRows = 4;
        uint64_t ids[kRows] = {0, 0, 0, 0};
        for (std::size_t i = 0; i < kRows; ++i) {
            Device row;
            row.server_id = 10;
            row.status = 5;
            ASSERT_TRUE(Storage::Insert(row).ok());
            ids[i] = row.id;
        }
        const auto ascending = Storage::Select<Device>(
            Eq(&Device::server_id, uint32_t{10}), OrderBy(&Device::status, SortDirection::ASC));
        ASSERT_TRUE(ascending.ok());
        ASSERT_EQ(ascending.value->size(), kRows);
        for (std::size_t i = 0; i < kRows; ++i) {
            EXPECT_EQ(ascending.value->at(i).id, ids[i]) << "同值行必须按主键升序";
        }
        const auto descending = Storage::Select<Device>(
            Eq(&Device::server_id, uint32_t{10}), OrderBy(&Device::status, SortDirection::DESC));
        ASSERT_TRUE(descending.ok());
        ASSERT_EQ(descending.value->size(), kRows);
        for (std::size_t i = 0; i < kRows; ++i) {
            EXPECT_EQ(descending.value->at(i).id, ids[i]) << "tie-breaker 方向不随排序键翻转";
        }
    }

    /// 投影基础合同（真实后端专用清单）：过滤/排序字段不要求被投影包含，
    /// 未投影字段保持记录默认值；投影列乱序/重复时值仍落位正确成员；
    /// 全列读取与显式投影结果一致；空结果与 Limit 合同不变
    inline void ProjectionFieldsCase()
    {
        Device a;
        std::memcpy(a.name, "alpha", sizeof("alpha"));
        a.server_id = 1;
        a.status = 2;
        Device b;
        std::memcpy(b.name, "beta", sizeof("beta"));
        b.server_id = 2;
        b.status = 1;
        Device c;
        std::memcpy(c.name, "gamma", sizeof("gamma"));
        c.server_id = 3;
        c.status = 2;
        ASSERT_TRUE(Storage::Insert(a).ok());
        ASSERT_TRUE(Storage::Insert(b).ok());
        ASSERT_TRUE(Storage::Insert(c).ok());

        // 过滤字段（status）与排序字段（status）都不在投影中：过滤/排序由数据库端
        // 独立完成；投影成员（server_id）按值返回，未投影成员（id/name/status）
        // 保持默认值。
        const auto projected = Storage::Select<Device>(
            Eq(&Device::status, uint32_t{2}), Fields(&Device::server_id), OrderBy(&Device::status, SortDirection::ASC));
        ASSERT_TRUE(projected.ok());
        ASSERT_EQ(projected.value->size(), 2U);
        EXPECT_EQ(projected.value->front().server_id, 1U);
        EXPECT_EQ(projected.value->front().id, 0U);
        EXPECT_EQ(std::string(projected.value->front().name), "");
        EXPECT_EQ(projected.value->front().status, 0U);
        EXPECT_EQ(projected.value->back().server_id, 3U);

        // 投影列乱序（status 在 server_id 之前，与声明顺序相反）：值落位正确成员。
        const auto reordered = Storage::Select<Device>(
            Eq(&Device::status, uint32_t{2}), Fields(&Device::status, &Device::server_id),
            OrderBy(&Device::server_id, SortDirection::ASC));
        ASSERT_TRUE(reordered.ok());
        ASSERT_EQ(reordered.value->size(), 2U);
        EXPECT_EQ(reordered.value->front().status, 2U);
        EXPECT_EQ(reordered.value->front().server_id, 1U);
        EXPECT_EQ(std::string(reordered.value->front().name), "");

        // 重复投影列去重：与 Fields(server_id, status) 结果一致。
        const auto duplicated = Storage::Select<Device>(
            Eq(&Device::status, uint32_t{2}), Fields(&Device::server_id, &Device::server_id, &Device::status),
            OrderBy(&Device::server_id, SortDirection::ASC));
        ASSERT_TRUE(duplicated.ok());
        ASSERT_EQ(duplicated.value->size(), 2U);
        EXPECT_EQ(duplicated.value->front().server_id, 1U);
        EXPECT_EQ(duplicated.value->front().status, 2U);

        // 全列对照：同一过滤/排序不投影时行集完整等值。
        const auto full = Storage::Select<Device>(
            Eq(&Device::status, uint32_t{2}), OrderBy(&Device::server_id, SortDirection::ASC));
        ASSERT_TRUE(full.ok());
        ASSERT_EQ(full.value->size(), 2U);
        EXPECT_EQ(full.value->front().id, a.id);
        EXPECT_EQ(std::string(full.value->front().name), "alpha");
        EXPECT_EQ(full.value->front().status, 2U);

        // 空结果：投影查询未命中是 OK + 空行集。
        const auto missing = Storage::Select<Device>(Eq(&Device::status, uint32_t{77}), Fields(&Device::server_id));
        ASSERT_TRUE(missing.ok());
        EXPECT_TRUE(missing.value->empty());

        // Limit 与投影叠加：截断与排序保持。
        const auto limited = Storage::Select<Device>(
            All<Device>(), Fields(&Device::server_id), OrderBy(&Device::server_id, SortDirection::ASC), Limit(2U));
        ASSERT_TRUE(limited.ok());
        ASSERT_EQ(limited.value->size(), 2U);
        EXPECT_EQ(limited.value->front().server_id, 1U);
        EXPECT_EQ(limited.value->back().server_id, 2U);

        // 同值排序 + 投影：tie-breaker 合同不因投影变化。
        const auto tied
            = Storage::Select<Device>(All<Device>(), Fields(&Device::id), OrderBy(&Device::status, SortDirection::ASC));
        ASSERT_TRUE(tied.ok());
        ASSERT_EQ(tied.value->size(), 3U);
        EXPECT_EQ(tied.value->at(0).id, b.id);
        EXPECT_EQ(tied.value->at(1).id, a.id);
        EXPECT_EQ(tied.value->at(2).id, c.id);
    }

    /// 投影大 BLOB 合同：变长大 BLOB 不投影时保持默认空载荷，全列读取时内容
    /// 逐字节等值（避免整行大 BLOB 回传是投影能力的存在理由）
    inline void ProjectionWideRowCase()
    {
        WideRow row;
        row.tag = 7;
        constexpr std::size_t kPayloadSize = 256U * 1024U;
        row.payload.resize(kPayloadSize);
        for (std::size_t i = 0; i < kPayloadSize; ++i) {
            row.payload[i] = static_cast<uint8_t>(i % 251U);
        }
        ASSERT_TRUE(Storage::Insert(row).ok());

        const auto light
            = Storage::Select<WideRow>(Eq(&WideRow::tag, uint32_t{7}), Fields(&WideRow::id, &WideRow::tag));
        ASSERT_TRUE(light.ok());
        ASSERT_EQ(light.value->size(), 1U);
        EXPECT_EQ(light.value->front().tag, 7U);
        EXPECT_TRUE(light.value->front().payload.empty()) << "未投影的大 BLOB 必须保持默认值";

        const auto full = Storage::SelectOne<WideRow>(Eq(&WideRow::id, light.value->front().id));
        ASSERT_TRUE(full.ok());
        ASSERT_EQ(full.value->payload.size(), kPayloadSize);
        EXPECT_EQ(full.value->payload.front(), row.payload.front());
        EXPECT_EQ(full.value->payload.back(), row.payload.back());
    }

    /// 投影定长数组 + 有效长度 hook 合同：数组与其长度字段同时返回但投影列顺序
    /// 颠倒（数组在前、长度在后）时，物化仍按目录字段顺序请求（长度先于数组），
    /// hook 有效长度正确约束数组容量，容量外元素保持默认值
    inline void ProjectionSlotRowCase()
    {
        SlotRow row;
        row.count = 3;
        row.slots[0] = 11;
        row.slots[1] = 22;
        row.slots[2] = 33;
        row.slots[3] = 44; // 有效长度之外：不得回填
        ASSERT_TRUE(Storage::Insert(row).ok());

        // 投影列顺序与声明顺序（id, count, slots）相反：slots 在前、count 在后。
        const auto reversed = Storage::Select<SlotRow>(All<SlotRow>(), Fields(&SlotRow::slots, &SlotRow::count));
        ASSERT_TRUE(reversed.ok());
        ASSERT_EQ(reversed.value->size(), 1U);
        EXPECT_EQ(reversed.value->front().count, 3U);
        EXPECT_EQ(reversed.value->front().slots[0], 11U);
        EXPECT_EQ(reversed.value->front().slots[1], 22U);
        EXPECT_EQ(reversed.value->front().slots[2], 33U);
        EXPECT_EQ(reversed.value->front().slots[3], 0U) << "有效长度之外的元素保持默认值";

        // 全列对照：同一行完整读回。
        const auto full = Storage::Select<SlotRow>(All<SlotRow>());
        ASSERT_TRUE(full.ok());
        ASSERT_EQ(full.value->size(), 1U);
        EXPECT_EQ(full.value->front().count, 3U);
        EXPECT_EQ(full.value->front().slots[2], 33U);
    }

    /// SelectOne + Fields 单行投影合同：直接物化到单个结果对象（不经 vector），
    /// 未投影大 BLOB 保持默认空；未命中 NOT_FOUND；重复 Fields 明确拒绝；
    /// WHERE/ORDER 引用字段可在投影外，选择结果与全行接口一致（主键 tie-breaker
    /// 不因投影变化）
    inline void SelectOneProjectionCase()
    {
        WideRow row;
        row.tag = 7;
        constexpr std::size_t kPayloadSize = 128U * 1024U;
        row.payload.resize(kPayloadSize);
        for (std::size_t i = 0; i < kPayloadSize; ++i) {
            row.payload[i] = static_cast<uint8_t>(i % 253U);
        }
        ASSERT_TRUE(Storage::Insert(row).ok());

        // 投影单行命中：投影字段返回，未投影 BLOB 保持默认空。
        const auto light
            = Storage::SelectOne<WideRow>(Eq(&WideRow::tag, uint32_t{7}), Fields(&WideRow::id, &WideRow::tag));
        ASSERT_TRUE(light.ok());
        ASSERT_TRUE(light.value.has_value());
        EXPECT_EQ(light.value->tag, 7U);
        EXPECT_TRUE(light.value->payload.empty()) << "未投影的大 BLOB 必须保持默认值";

        // 完整单行接口对照：同键全行读回 BLOB。
        const auto full = Storage::SelectOne<WideRow>(Eq(&WideRow::id, light.value->id));
        ASSERT_TRUE(full.ok());
        EXPECT_EQ(full.value->payload.size(), kPayloadSize);

        // 未命中：NOT_FOUND（不是 OK + 默认对象）。
        const auto missing = Storage::SelectOne<WideRow>(Eq(&WideRow::tag, uint32_t{77}), Fields(&WideRow::id));
        EXPECT_EQ(missing.code, DbCode::NOT_FOUND);
        EXPECT_FALSE(missing.value.has_value());

        // 重复 Fields 明确拒绝（入口错误，不触引擎）。
        const auto duplicated
            = Storage::SelectOne<WideRow>(All<WideRow>(), Fields(&WideRow::id), Fields(&WideRow::tag));
        EXPECT_EQ(duplicated.code, DbCode::INVALID_ARGUMENT);

        // 非投影 WHERE/ORDER 字段 + 主键 tie-breaker：投影不改变行选择。
        Device a;
        a.server_id = 1;
        a.status = 2;
        Device b;
        b.server_id = 2;
        b.status = 2;
        Device c;
        c.server_id = 3;
        c.status = 1;
        ASSERT_TRUE(Storage::Insert(a).ok());
        ASSERT_TRUE(Storage::Insert(b).ok());
        ASSERT_TRUE(Storage::Insert(c).ok());
        // WHERE(status)/ORDER(status) 都不在投影（只取 server_id）：status=2 两行
        // 按 status ASC + 主键 tie-breaker 首行是 a（server_id=1）。
        const auto firstOfTwo = Storage::SelectOne<Device>(
            Eq(&Device::status, uint32_t{2}), Fields(&Device::server_id), OrderBy(&Device::status, SortDirection::ASC));
        ASSERT_TRUE(firstOfTwo.ok());
        ASSERT_TRUE(firstOfTwo.value.has_value());
        EXPECT_EQ(firstOfTwo.value->server_id, 1U);
        EXPECT_EQ(firstOfTwo.value->id, 0U) << "未投影字段保持默认值";
        // 与全行 SelectOne 同条件对照：同一行。
        const auto wholeRow = Storage::SelectOne<Device>(
            Eq(&Device::status, uint32_t{2}), OrderBy(&Device::status, SortDirection::ASC));
        ASSERT_TRUE(wholeRow.ok());
        EXPECT_EQ(wholeRow.value->id, a.id);
    }
} // namespace cases
} // namespace storage_conformance

/// 按 case 名生成后端无关 gtest 用例；fixture 需先定义 OpenConformanceSession/CloseConformanceSession。
#define HCCL_VM_CONFORMANCE_CASE(CaseName)                           \
    TEST(BackendConformance, CaseName)                               \
    {                                                                \
        ASSERT_TRUE(storage_conformance::OpenConformanceSession());  \
        storage_conformance::cases::CaseName##Case();                \
        ASSERT_TRUE(storage_conformance::CloseConformanceSession()); \
    }

/// 非事务公共 conformance case 清单（后端无关；case 内容与期望不做任何后端修改）。
#define HCCL_VM_NON_TRANSACTIONAL_CONFORMANCE_CASES()  \
    HCCL_VM_CONFORMANCE_CASE(GeneratedId)              \
    HCCL_VM_CONFORMANCE_CASE(SelectOneAndSelect)       \
    HCCL_VM_CONFORMANCE_CASE(ConditionTree)            \
    HCCL_VM_CONFORMANCE_CASE(NestedCondition)          \
    HCCL_VM_CONFORMANCE_CASE(OrderByLimit)             \
    HCCL_VM_CONFORMANCE_CASE(Count)                    \
    HCCL_VM_CONFORMANCE_CASE(CountMatches)             \
    HCCL_VM_CONFORMANCE_CASE(CountMatchesUint64Half)   \
    HCCL_VM_CONFORMANCE_CASE(CountMatchesDynamicScope) \
    HCCL_VM_CONFORMANCE_CASE(AffectedRows)             \
    HCCL_VM_CONFORMANCE_CASE(Clear)                    \
    HCCL_VM_CONFORMANCE_CASE(DynamicScope)             \
    HCCL_VM_CONFORMANCE_CASE(UniqueConstraint)         \
    HCCL_VM_CONFORMANCE_CASE(ForeignKeyCascade)        \
    HCCL_VM_CONFORMANCE_CASE(Uint64Blob)               \
    HCCL_VM_CONFORMANCE_CASE(Uint64Half)               \
    HCCL_VM_CONFORMANCE_CASE(PrefixEscape)             \
    HCCL_VM_CONFORMANCE_CASE(UpdateParameterOrder)     \
    HCCL_VM_CONFORMANCE_CASE(SameKeyTieBreak)          \
    HCCL_VM_CONFORMANCE_CASE(UnifiedError)             \
    HCCL_VM_CONFORMANCE_CASE(SelectMany)               \
    HCCL_VM_CONFORMANCE_CASE(IncrementEncoding)        \
    HCCL_VM_CONFORMANCE_CASE(IncrementArithmetic)      \
    HCCL_VM_CONFORMANCE_CASE(IncrementBoundary)

/// 投影 conformance case 清单（真实后端专用：Fake 内存引擎不实现字段投影，
/// 保持其既有能力合同，不实例化本清单）
#define HCCL_VM_PROJECTION_CONFORMANCE_CASES()  \
    HCCL_VM_CONFORMANCE_CASE(ProjectionFields)  \
    HCCL_VM_CONFORMANCE_CASE(ProjectionWideRow) \
    HCCL_VM_CONFORMANCE_CASE(ProjectionSlotRow) \
    HCCL_VM_CONFORMANCE_CASE(SelectOneProjection)

/// 并发原子性 case 清单（真实后端专用：Fake 内存引擎无跨线程锁，不实例化；
/// 证明单语句 field = field - ? 原子减法在并发下不丢更新）
#define HCCL_VM_CONCURRENT_INCREMENT_CONFORMANCE_CASES() HCCL_VM_CONFORMANCE_CASE(IncrementConcurrent)

/// B2 最终有界修正批次新增 case（后端无关单一来源；行为由各后端私有测试先行
/// RED→GREEN 驱动，本清单提供公共合同编码）。暂不并入 NON_TRANSACTIONAL 共享
/// 清单：Fake 内存引擎存在既有实现缺口（显式自增主键被无条件覆盖、Update 不
/// 检查唯一约束），并入会使 Fake fixture 携带失败用例——待 Fake 补齐语义后
/// 由 Controller 决定并入（真实证据见 B2 修正进度文档）。
#define HCCL_VM_B2_FINAL_CORRECTION_CONFORMANCE_CASES()      \
    HCCL_VM_CONFORMANCE_CASE(MultiRowUniqueConflict)         \
    HCCL_VM_CONFORMANCE_CASE(ExplicitAutoincrementHighWater) \
    HCCL_VM_CONFORMANCE_CASE(ForeignKeyRestrict)

/// 全部公共 conformance case = 非事务清单 + 事务 case + 投影清单
/// （SQLite/MariaDB 真实后端实例化全部；Fake 实例化非事务与事务清单）。
#define HCCL_VM_ALL_CONFORMANCE_CASES()           \
    HCCL_VM_NON_TRANSACTIONAL_CONFORMANCE_CASES() \
    HCCL_VM_CONFORMANCE_CASE(Transaction)         \
    HCCL_VM_PROJECTION_CONFORMANCE_CASES()

#endif // HCCL_VM_STORAGE_BACKEND_CONFORMANCE_H
