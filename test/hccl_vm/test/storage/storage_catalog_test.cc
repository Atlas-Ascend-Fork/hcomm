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
 * @file storage_catalog_test.cc
 * @brief 统一逻辑 DDL 目录（StorageCatalog）机制合同测试。
 *
 * 使用 canonical 小型测试目录（两张逻辑库、三张表）验证：
 * - 一份 master catalog 登记派生 descriptor/RecordSchema/manifest/fingerprint；
 * - 字段名严格等于 C++ 成员名、类型由成员类型与 Text/Blob 属性推导；
 * - 聚合 arity 漂移门、主键/索引/唯一/外键收集、字段通道往返
 *   （visitRecord → materializeRows）与两类记录语义 hook；
 * - 短宏区段边界（#undef 后不污染包含方）。
 */
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"

#include "storage/query_expression.h"
#include "storage/storage_catalog.h"

// ---- canonical 小型测试记录（模拟两个逻辑数据库的表） ----
namespace catalog_test {

struct Device {
    uint64_t id{0};
    uint32_t server_id{0};
    char soc_version[16]{};
    uint32_t status{0};
};

/// 派生列 + BLOB 数组 + 动态 scope 记录（OpData 形态）
struct OpDetail {
    uint64_t id{0};
    uint64_t sync_iter{0};
    uint8_t payload[8]{};
    uint64_t size{0};
    uint64_t end_ptr{0};
};

/// 有效数组长度 hook 记录（字节数组有效长度由同记录另一字段决定）
struct ChannelTab {
    uint64_t id{0};
    uint32_t jetty_num{0};
    uint64_t jetty_id[6]{};
};

/// 级联外键记录（OpTask → OpDetail 形态：动态 PID 表引用主库明细行）
struct OpTask {
    uint64_t id{0};
    uint64_t op_detail_id{0};
    uint32_t task_seq{0};
};

/// fingerprint 漂移对照记录：与 Device 同名同字段数、status 类型不同
struct DeviceLayoutV2 {
    uint64_t id{0};
    uint32_t server_id{0};
    char soc_version[16]{};
    uint64_t status{0};
};

/// 不透明 aggregate 载荷：非数组、aggregate 且 trivially copyable 的嵌套成员类型
/// （显式声明 Blob 后按 sizeof(M) 固定字节载荷编码）。
struct OpaquePayload {
    uint16_t tag{0};
    uint32_t count{0};
    uint8_t body[6]{};
};

/// 仅含 id + 不透明 payload 的表记录（嵌套 aggregate 成员的 canonical 缩影）。
struct OpaqueRow {
    uint64_t id{0};
    OpaquePayload payload{};
};

/// 满容量无 NUL 的 Text 数组边界回归记录：code[4] 之后紧跟 guard[4]，
/// 编码必须以 sizeof(code) 为界，不得读取相邻 guard 字节。
struct BoundedTextRow {
    uint64_t id{0};
    char code[4]{};
    char guard[4]{};
};

/// 非平凡复制 aggregate：不满足不透明 Blob 判据（不得作为固定字节载荷字段）。
struct NonTrivialAggregate {
    std::string text;
};

} // namespace catalog_test

namespace HcclSim::Storage::Ddl {
// 派生列 hook：end_ptr = sync_iter + size（编码时计算，解码忽略）
template <>
struct RecordCodecHooks<catalog_test::OpDetail> : DefaultRecordCodecHooks<catalog_test::OpDetail> {
    static std::optional<DbScalar> EncodeDerivedField(const catalog_test::OpDetail& record, uint16_t fieldId)
    {
        return fieldId == 4U ? std::optional<DbScalar>{DbScalar{record.sync_iter + record.size}} : std::nullopt;
    }

    static bool IgnoreDecodedField(uint16_t fieldId) { return fieldId == 4U; }
};

// 有效数组长度 hook：jetty_id 只编码 jetty_num 个有效元素
template <>
struct RecordCodecHooks<catalog_test::ChannelTab> : DefaultRecordCodecHooks<catalog_test::ChannelTab> {
    static int64_t EffectiveBytesLength(const catalog_test::ChannelTab& record, uint16_t fieldId)
    {
        return fieldId == 2U ? static_cast<int64_t>(record.jetty_num) * static_cast<int64_t>(sizeof(uint64_t)) : -1;
    }
};
} // namespace HcclSim::Storage::Ddl

// ---- 业务 DDL 区（全局作用域）：短别名 → master catalog → DEFINE_DATABASE → 立即 #undef ----
using namespace HcclSim::Storage::Ddl;
#define Field HCCL_VM_DDL_FIELD
#define Version HCCL_VM_DDL_VERSION
#define PrimaryKey HCCL_VM_DDL_PRIMARY_KEY
#define Unique HCCL_VM_DDL_UNIQUE
#define Index HCCL_VM_DDL_INDEX
#define ForeignKey HCCL_VM_DDL_FOREIGN_KEY
#define Scope Ddl::ScopeKind

#define CATALOG_TEST_RUNNER_DATABASE(Table)                                                                           \
    Table(                                                                                                            \
        catalog_test::Device,                                                                                         \
        (Version(1), Scope::Static, Field(id, AutoIncrement), Field(server_id), Field(soc_version, Text),             \
         Field(status), PrimaryKey(id), Index(server_id), Index(server_id, status)))                                  \
        Table(                                                                                                        \
            catalog_test::ChannelTab, (Version(1), Scope::Static, Field(id, AutoIncrement), Field(jetty_num),         \
                                       Field(jetty_id, Blob), PrimaryKey(id)))                                        \
            Table(                                                                                                    \
                catalog_test::DeviceLayoutV2, (Version(1), Scope::Static, Field(id, AutoIncrement), Field(server_id), \
                                               Field(soc_version, Text), Field(status), PrimaryKey(id)))              \
                Table(                                                                                                \
                    catalog_test::OpaqueRow,                                                                          \
                    (Version(1), Scope::Static, Field(id, AutoIncrement), Field(payload, Blob), PrimaryKey(id)))      \
                    Table(                                                                                            \
                        catalog_test::BoundedTextRow, (Version(1), Scope::Static, Field(id, AutoIncrement),           \
                                                       Field(code, Text), Field(guard, Text), PrimaryKey(id)))

#define CATALOG_TEST_OP_DATABASE(Table)                                                                                \
    Table(                                                                                                             \
        catalog_test::OpDetail,                                                                                        \
        (Version(2), Scope::DynamicPid, Field(id, AutoIncrement), Field(sync_iter), Field(payload, Blob), Field(size), \
         Field(end_ptr, Derived), PrimaryKey(id), Unique(sync_iter), Index(sync_iter, id)))                            \
        Table(                                                                                                         \
            catalog_test::OpTask, (Version(1), Scope::DynamicPid, Field(id, AutoIncrement), Field(op_detail_id),       \
                                   Field(task_seq), PrimaryKey(id), Index(op_detail_id, task_seq),                     \
                                   ForeignKey(op_detail_id, catalog_test::OpDetail::id, OnDelete::Cascade)))

#define HCCL_VM_DDL_CURRENT_DATABASE "CATALOG_TEST_RUNNER_DB"
HCCL_VM_DEFINE_DATABASE(kCatalogTestRunnerDatabase, CATALOG_TEST_RUNNER_DB, CATALOG_TEST_RUNNER_DATABASE)
#undef HCCL_VM_DDL_CURRENT_DATABASE
#define HCCL_VM_DDL_CURRENT_DATABASE "CATALOG_TEST_OP_DB"
HCCL_VM_DEFINE_DATABASE(kCatalogTestOpDatabase, CATALOG_TEST_OP_DB, CATALOG_TEST_OP_DATABASE)
#undef HCCL_VM_DDL_CURRENT_DATABASE
#undef Field
#undef Version
#undef PrimaryKey
#undef Unique
#undef Index
#undef ForeignKey
#undef Scope
// 短宏边界：DDL 区结束即 #undef，包含方不受污染。
#ifndef Field
#define CATALOG_TEST_SHORT_MACRO_CLEANED 1
#endif

namespace catalog_test {
using namespace HcclSim::Storage;

namespace {
    const auto kTestCatalog = Ddl::MakeCatalog(Ddl::kCatalogTestRunnerDatabase, Ddl::kCatalogTestOpDatabase);

    struct BorrowedMemberCapture {
        std::size_t calls{0};
        const Ddl::CatalogTableMeta* meta{nullptr};
        const char* textData{nullptr};
        const uint8_t* bytesData{nullptr};
        std::size_t textBytes{0};
        std::size_t blobBytes{0};
    };

    DbResult<void>
    CaptureBorrowedMember(void* opaque, uint16_t fieldId, const void* data, std::size_t bytes, bool isNull)
    {
        auto& capture = *static_cast<BorrowedMemberCapture*>(opaque);
        ++capture.calls;
        // 字段语义按同一目录 FieldDesc 判定：TEXT 数组成员与 BLOB 数组成员都以原始
        // data/bytes 直发（不构造中间标量视图）。
        if (!isNull && data != nullptr && capture.meta != nullptr && fieldId < capture.meta->fieldCount) {
            const auto& field = capture.meta->fields[fieldId];
            if (field.type == LogicalFieldType::STRING) {
                capture.textData = static_cast<const char*>(data);
                capture.textBytes = bytes;
            } else if (field.type == LogicalFieldType::BYTES) {
                capture.bytesData = static_cast<const uint8_t*>(data);
                capture.blobBytes = bytes;
            }
        }
        return {DbCode::OK, 0, {}};
    }

    struct DerivedBorrowCapture {
        std::size_t calls{0};
        std::uint64_t derivedValue{0};
        std::size_t derivedBytes{0};
    };

    DbResult<void> CaptureDerivedField(void* opaque, uint16_t fieldId, const void* data, std::size_t bytes, bool isNull)
    {
        auto& capture = *static_cast<DerivedBorrowCapture*>(opaque);
        if (fieldId != 4U) {
            return {DbCode::OK, 0, {}}; // 只观察派生列（OpDetail::end_ptr）
        }
        ++capture.calls;
        capture.derivedBytes = bytes;
        if (!isNull && data != nullptr && bytes == sizeof(std::uint64_t)) {
            std::memcpy(&capture.derivedValue, data, sizeof(capture.derivedValue));
        }
        return {DbCode::OK, 0, {}};
    }

    // ---- 字段通道往返 harness（visitRecord → 值表 → materializeRows/ReadField）----
    // 生产 CRUD 的原始引用通道 ABI 合同：INSERT 侧 visitRecord 按声明顺序发射原始
    // 值，SELECT 侧 materializeRows 按声明顺序请求成员回填；本 harness 在两个通道
    // 之间持一份 fieldId → DbScalar 值表（镜像真实后端的语句绑定/结果读取）。

    /// visitRecord 收集器：按 FieldDesc::type + 实际宽度把原始值收进值表
    struct ChannelCollector {
        const Ddl::CatalogTableMeta* meta{nullptr};
        std::vector<std::pair<uint16_t, DbScalar>> fields;

        static DbResult<void> Emit(void* opaque, uint16_t fieldId, const void* data, std::size_t bytes, bool isNull)
        {
            auto& self = *static_cast<ChannelCollector*>(opaque);
            if (fieldId >= self.meta->fieldCount) {
                return {DbCode::INVALID_ARGUMENT, 0, "record emits an undeclared field"};
            }
            const auto& field = self.meta->fields[fieldId];
            DbScalar value{uint64_t{0}};
            switch (field.type) {
                case LogicalFieldType::SIGNED: {
                    int64_t number = 0;
                    std::memcpy(&number, data, std::min(bytes, sizeof(number)));
                    value = DbScalar{number};
                    break;
                }
                case LogicalFieldType::UNSIGNED: {
                    uint64_t number = 0;
                    std::memcpy(&number, data, std::min(bytes, sizeof(number)));
                    value = DbScalar{number};
                    break;
                }
                case LogicalFieldType::BOOLEAN:
                    value = DbScalar{*static_cast<const uint8_t*>(data) != 0U};
                    break;
                case LogicalFieldType::REAL:
                    if (bytes == sizeof(float)) {
                        float number = 0.0F;
                        std::memcpy(&number, data, sizeof(number));
                        value = DbScalar{static_cast<double>(number)};
                    } else {
                        double number = 0.0;
                        std::memcpy(&number, data, sizeof(number));
                        value = DbScalar{number};
                    }
                    break;
                case LogicalFieldType::STRING:
                    value = DbScalar{std::string(static_cast<const char*>(data), bytes)};
                    break;
                case LogicalFieldType::BYTES:
                default: {
                    const auto* begin = static_cast<const uint8_t*>(data);
                    value = DbScalar{std::vector<uint8_t>(begin, begin + bytes)};
                    break;
                }
            }
            (void)isNull;
            self.fields.emplace_back(fieldId, std::move(value));
            return {DbCode::OK, 0, {}};
        }
    };

    /// materializeRows 行游标：单行值表回填（镜像引擎 ReadField 的按类型写成员）
    struct ChannelCursor {
        const std::vector<std::pair<uint16_t, DbScalar>>* fields{nullptr};
        const Ddl::CatalogTableMeta* meta{nullptr};
        bool served{false};
        /// FillCtx 实际请求的字段（按请求顺序；IgnoreDecodedField 的字段不出现）
        std::vector<uint16_t> requestedFields;
        /// 每次请求携带的 capacity（数组字段为 hook 有效长度约束后的容量）
        std::vector<std::size_t> requestedCapacities;

        static DbResult<bool> Next(void* opaque)
        {
            auto& self = *static_cast<ChannelCursor*>(opaque);
            if (self.served) {
                return {DbCode::OK, false, 0, {}};
            }
            self.served = true;
            return {DbCode::OK, true, 0, {}};
        }

        static DbResult<void> ReadField(void* opaque, uint16_t fieldId, void* destination, std::size_t capacity)
        {
            auto& self = *static_cast<ChannelCursor*>(opaque);
            if (fieldId >= self.meta->fieldCount) {
                return {DbCode::INVALID_ARGUMENT, 0, "row field is unmapped"};
            }
            self.requestedFields.push_back(fieldId);
            self.requestedCapacities.push_back(capacity);
            const DbScalar* value = nullptr;
            for (const auto& field : *self.fields) {
                if (field.first == fieldId) {
                    value = &field.second;
                    break;
                }
            }
            if (value == nullptr) {
                return {DbCode::OK, 0, {}}; // 缺省字段保持目标默认值
            }
            const auto& field = self.meta->fields[fieldId];
            switch (field.type) {
                case LogicalFieldType::SIGNED:
                case LogicalFieldType::UNSIGNED: {
                    std::uint64_t bits = 0;
                    if (const auto* number = std::get_if<int64_t>(value); number != nullptr) {
                        std::memcpy(&bits, number, sizeof(bits));
                    } else if (const auto* number = std::get_if<uint64_t>(value); number != nullptr) {
                        std::memcpy(&bits, number, sizeof(bits));
                    }
                    std::memcpy(
                        destination, &bits,
                        std::min(field.byteWidth != 0 ? field.byteWidth : sizeof(bits), sizeof(bits)));
                    return {DbCode::OK, 0, {}};
                }
                case LogicalFieldType::BOOLEAN: {
                    bool flag = false;
                    if (const auto* stored = std::get_if<bool>(value); stored != nullptr) {
                        flag = *stored;
                    }
                    std::memcpy(destination, &flag, sizeof(flag));
                    return {DbCode::OK, 0, {}};
                }
                case LogicalFieldType::REAL: {
                    double number = 0.0;
                    if (const auto* stored = std::get_if<double>(value); stored != nullptr) {
                        number = *stored;
                    }
                    if (field.byteWidth == sizeof(float)) {
                        const float narrow = static_cast<float>(number);
                        std::memcpy(destination, &narrow, sizeof(narrow));
                    } else {
                        std::memcpy(destination, &number, sizeof(number));
                    }
                    return {DbCode::OK, 0, {}};
                }
                case LogicalFieldType::STRING: {
                    const auto* text = std::get_if<std::string>(value);
                    if (text == nullptr) {
                        return {DbCode::OK, 0, {}};
                    }
                    if (field.byteWidth == 0) {
                        static_cast<std::string*>(destination)->assign(text->data(), text->size());
                        return {DbCode::OK, 0, {}};
                    }
                    const std::size_t bound = capacity != 0 ? capacity : field.byteWidth;
                    const std::size_t count = std::min(text->size(), bound > 0 ? bound - 1 : 0);
                    if (count > 0) {
                        std::memcpy(destination, text->data(), count);
                    }
                    if (bound > 0) {
                        static_cast<char*>(destination)[count] = '\0';
                    }
                    return {DbCode::OK, 0, {}};
                }
                case LogicalFieldType::BYTES:
                default: {
                    const auto* bytes = std::get_if<std::vector<uint8_t>>(value);
                    if (bytes == nullptr) {
                        return {DbCode::OK, 0, {}};
                    }
                    if (field.byteWidth == 0) {
                        static_cast<std::vector<uint8_t>*>(destination)->assign(bytes->begin(), bytes->end());
                        return {DbCode::OK, 0, {}};
                    }
                    if (capacity > 0) {
                        const std::size_t count = std::min(bytes->size(), capacity);
                        if (count > 0) {
                            std::memcpy(destination, bytes->data(), count);
                        }
                        return {DbCode::OK, 0, {}};
                    }
                    // 不透明定宽对象：仅字节数精确相等时整体回写（引擎同一规则）
                    if (bytes->size() == field.byteWidth) {
                        std::memcpy(destination, bytes->data(), field.byteWidth);
                    }
                    return {DbCode::OK, 0, {}};
                }
            }
        }
    };

    /// 单记录通道往返：返回 true 时 target 已按值表回填（调用方断言成员等值）
    template <typename T>
    DbResult<void> ChannelRoundTrip(const T& source, T& target, const Ddl::CatalogTableMeta& meta)
    {
        ChannelCollector collector;
        collector.meta = &meta;
        const auto visited = meta.visitRecord(&source, &collector, ChannelCollector::Emit);
        if (!visited.ok()) {
            return visited;
        }
        ChannelCursor cursor;
        cursor.fields = &collector.fields;
        cursor.meta = &meta;
        ResultSink sink = ResultSink::For(target);
        return meta.materializeRows(sink, &cursor, ChannelCursor::Next, ChannelCursor::ReadField);
    }

    // ---- 编译期合同：表数量、字段数、arity 门、成员指针映射 ----
    static_assert(Ddl::kCatalogTestRunnerDatabase.tableCount == 5);
    static_assert(Ddl::kCatalogTestOpDatabase.tableCount == 2);
    static_assert(Ddl::TableDdl<Device>::kFieldCount == 4);
    static_assert(Ddl::kDirectAggregateFieldCount<Device> == Ddl::TableDdl<Device>::kFieldCount);
    static_assert(Ddl::kDirectAggregateFieldCount<OpDetail> == 5);
    static_assert(Ddl::kDirectAggregateFieldCount<ChannelTab> == 3);
    static_assert(Ddl::kDirectAggregateFieldCount<OpaqueRow> == 2);
    static_assert(RecordSchema<Device>::FieldIdOf(&Device::server_id) == 1U);
    static_assert(RecordSchema<Device>::FieldIdOf(&Device::soc_version) == 2U);
    static_assert(RecordSchema<OpDetail>::FieldIdOf(&OpDetail::end_ptr) == 4U);
    static_assert(RecordSchema<ChannelTab>::FieldIdOf(&ChannelTab::jetty_id) == 2U);
    static_assert(RecordSchema<OpaqueRow>::FieldIdOf(&OpaqueRow::payload) == 1U);
    // 不透明 Blob 判据：仅非数组、aggregate 且 trivially copyable 的成员满足；
    // 标量/字符串与非平凡复制 aggregate 均为 false（误用由 FieldPropsGate 编译期拒绝）。
    static_assert(Ddl::detail::kIsOpaqueAggregateMember<OpaquePayload>);
    static_assert(!Ddl::detail::kIsOpaqueAggregateMember<NonTrivialAggregate>);
    static_assert(!Ddl::detail::kIsOpaqueAggregateMember<uint32_t>);
    static_assert(!Ddl::detail::kIsOpaqueAggregateMember<std::string>);
    // 成员固定字节宽度的唯一来源：定宽标量/C 数组/不透明 aggregate 为 sizeof(M)，变长逻辑值为 0。
    static_assert(Ddl::detail::MemberByteWidth<uint32_t>() == 4U);
    static_assert(Ddl::detail::MemberByteWidth<char[16]>() == 16U);
    static_assert(Ddl::detail::MemberByteWidth<OpaquePayload>() == sizeof(OpaquePayload));
    static_assert(Ddl::detail::MemberByteWidth<std::string>() == 0U);
    static_assert(Ddl::detail::MemberByteWidth<std::vector<uint8_t>>() == 0U);
    // 目录聚合与跨库查找在运行期断言（meta 数组含函数指针/type_index，为静态只读而非 constexpr）

    // 表名 canonical 化：全限定登记类型取最后一段作为表名（不保留命名空间前缀）。
    TEST(StorageCatalogTest, TableIdIsCanonicalTypeName)
    {
        EXPECT_EQ(RecordSchema<Device>::TableId(), std::string_view{"Device"});
        EXPECT_EQ(RecordSchema<OpDetail>::TableId(), std::string_view{"OpDetail"});
        EXPECT_EQ(RecordSchema<ChannelTab>::TableId(), std::string_view{"ChannelTab"});
    }

    // 字段名严格等于 C++ 成员名；类型由成员类型与 Text/Blob 属性推导。
    TEST(StorageCatalogTest, FieldNamesMatchMembersAndTypesAreDerived)
    {
        const auto& fields = Ddl::TableDdl<Device>::kFields;
        ASSERT_EQ(Ddl::TableDdl<Device>::kFieldCount, 4U);
        EXPECT_EQ(fields[0].name, std::string_view{"id"});
        EXPECT_EQ(fields[0].type, LogicalFieldType::UNSIGNED);
        EXPECT_TRUE(fields[0].primaryKey);
        EXPECT_EQ(fields[1].name, std::string_view{"server_id"});
        EXPECT_EQ(fields[2].name, std::string_view{"soc_version"});
        EXPECT_EQ(fields[2].type, LogicalFieldType::STRING);
        EXPECT_EQ(fields[3].name, std::string_view{"status"});
        EXPECT_EQ(Ddl::TableDdl<OpDetail>::kFields[2].type, LogicalFieldType::BYTES);
        EXPECT_EQ(Ddl::TableDdl<OpDetail>::kFields[4].type, LogicalFieldType::UNSIGNED);
    }

    // descriptor 为 constexpr 静态只读数据：manifest 指向同一份地址，不复制名称。
    TEST(StorageCatalogTest, ManifestIsStaticAndSharesDescriptorStorage)
    {
        const auto* meta = kTestCatalog.FindTable("Device");
        ASSERT_NE(meta, nullptr);
        EXPECT_EQ(meta->fields, Ddl::TableDdl<Device>::kFields.data());
        EXPECT_EQ(meta->fieldCount, 4U);
        EXPECT_EQ(meta->primaryKeyCount, 1U);
        EXPECT_STREQ(meta->primaryKeyColumns[0], "id");
        EXPECT_EQ(meta->scope, Ddl::ScopeKind::Static);
        EXPECT_EQ(kTestCatalog.FindTable("OpDetail")->scope, Ddl::ScopeKind::DynamicPid);
    }

    // 索引/唯一约束由同一 Table entry 收集；复合索引列序保持声明顺序。
    TEST(StorageCatalogTest, IndexesAndUniquesComeFromSameEntry)
    {
        const auto* device = kTestCatalog.FindTable("Device");
        ASSERT_EQ(device->indexCount, 2U);
        EXPECT_FALSE(device->indexes[0].unique);
        EXPECT_EQ(device->indexes[0].columnCount, 1U);
        EXPECT_STREQ(device->indexes[0].columns[0], "server_id");
        EXPECT_EQ(device->indexes[1].columnCount, 2U);
        EXPECT_STREQ(device->indexes[1].columns[0], "server_id");
        EXPECT_STREQ(device->indexes[1].columns[1], "status");

        const auto* opDetail = kTestCatalog.FindTable("OpDetail");
        ASSERT_GE(opDetail->indexCount, 2U);
        const auto* uniqueIndex = &opDetail->indexes[0];
        bool hasUnique = false;
        for (std::size_t i = 0; i < opDetail->indexCount; ++i) {
            if (opDetail->indexes[i].unique) {
                uniqueIndex = &opDetail->indexes[i];
                hasUnique = true;
            }
        }
        EXPECT_TRUE(hasUnique);
        EXPECT_EQ(uniqueIndex->columnCount, 1U);
        EXPECT_STREQ(uniqueIndex->columns[0], "sync_iter");
    }

    // 字段通道往返：整数/文本数组/BLOB 数组全部经 visitRecord 发射原始值、
    // materializeRows 请求成员回填，通用 descriptor 处理，不逐表手写 codec。
    TEST(StorageCatalogTest, ChannelRoundTripsPlainFields)
    {
        Device device;
        device.id = 7;
        device.server_id = 3;
        std::memcpy(device.soc_version, "Ascend910", sizeof("Ascend910"));
        device.status = 1;
        Device target;
        const auto roundTripped = ChannelRoundTrip(device, target, RecordSchema<Device>::Meta());
        ASSERT_TRUE(roundTripped.ok());
        EXPECT_EQ(target.id, 7U);
        EXPECT_EQ(target.server_id, 3U);
        EXPECT_EQ(target.status, 1U);
        // char[16] 按 Text 发射：NUL 前字节承载语义，回填后语义不变。
        EXPECT_EQ(std::string(target.soc_version), "Ascend910");
    }

    TEST(StorageCatalogTest, CatalogVisitRecordBorrowsTextAndBlobMembers)
    {
        Device device;
        std::memcpy(device.soc_version, "Ascend910", sizeof("Ascend910"));
        BorrowedMemberCapture deviceCapture;
        const auto& deviceMeta = RecordSchema<Device>::Meta();
        ASSERT_NE(deviceMeta.visitRecord, nullptr);
        deviceCapture.meta = &deviceMeta;
        ASSERT_TRUE(deviceMeta.visitRecord(&device, &deviceCapture, CaptureBorrowedMember).ok());
        EXPECT_EQ(deviceCapture.calls, 4U);
        EXPECT_EQ(deviceCapture.textData, device.soc_version);
        EXPECT_EQ(deviceCapture.textBytes, std::strlen("Ascend910"));

        OpDetail detail;
        detail.payload[0] = 7;
        BorrowedMemberCapture detailCapture;
        const auto& detailMeta = RecordSchema<OpDetail>::Meta();
        ASSERT_NE(detailMeta.visitRecord, nullptr);
        detailCapture.meta = &detailMeta;
        ASSERT_TRUE(detailMeta.visitRecord(&detail, &detailCapture, CaptureBorrowedMember).ok());
        EXPECT_EQ(detailCapture.calls, 5U);
        EXPECT_EQ(detailCapture.bytesData, detail.payload);
        EXPECT_EQ(detailCapture.blobBytes, sizeof(detail.payload));
    }

    TEST(StorageCatalogTest, DerivedFieldIsEmittedInlineDuringVisit)
    {
        // 目录登记的派生列记录（OpDetail::end_ptr = sync_iter + size）：
        // 派生值以 DbScalar 原生宽度（uint64 8 字节）临时发射，回调同步消费。
        OpDetail row;
        row.sync_iter = 100;
        row.size = 28;
        DerivedBorrowCapture capture;
        const auto& meta = RecordSchema<OpDetail>::Meta();
        ASSERT_NE(meta.visitRecord, nullptr);
        ASSERT_TRUE(meta.visitRecord(&row, &capture, CaptureDerivedField).ok());
        EXPECT_EQ(capture.calls, 1U);
        EXPECT_EQ(capture.derivedBytes, sizeof(std::uint64_t));
        EXPECT_EQ(capture.derivedValue, 128U);
    }

    // BLOB 数组逐字节通道往返：uint8_t[8] 按容量整体发射与回填。
    TEST(StorageCatalogTest, ChannelRoundTripsBlobArrays)
    {
        OpDetail detail;
        detail.sync_iter = 100;
        detail.size = 5;
        detail.payload[0] = 1;
        detail.payload[7] = 9;
        OpDetail target;
        const auto roundTripped = ChannelRoundTrip(detail, target, RecordSchema<OpDetail>::Meta());
        ASSERT_TRUE(roundTripped.ok());
        EXPECT_EQ(target.sync_iter, 100U);
        EXPECT_EQ(target.payload[0], 1U);
        EXPECT_EQ(target.payload[7], 9U);
    }

    // 派生列 hook：发射值由同记录其他字段计算，物化按 hook 忽略（基址/大小为权威）。
    TEST(StorageCatalogTest, DerivedFieldHookComputesOnEmitAndSkipsMaterialize)
    {
        OpDetail detail;
        detail.sync_iter = 100;
        detail.size = 5;
        // 发射侧：派生值以 DbScalar 原生宽度（uint64 8 字节）临时发射，回调同步消费。
        DerivedBorrowCapture capture;
        const auto& meta = RecordSchema<OpDetail>::Meta();
        ASSERT_NE(meta.visitRecord, nullptr);
        ASSERT_TRUE(meta.visitRecord(&detail, &capture, CaptureDerivedField).ok());
        EXPECT_EQ(capture.calls, 1U);
        EXPECT_EQ(capture.derivedBytes, sizeof(std::uint64_t));
        EXPECT_EQ(capture.derivedValue, 105U);

        // 物化侧：IgnoreDecodedField 忽略派生列——FillCtx 不请求 fieldId 4，
        // 目标 end_ptr 保持默认值（行值 999 不回写）。
        ChannelCollector collector;
        collector.meta = &meta;
        ASSERT_TRUE(meta.visitRecord(&detail, &collector, ChannelCollector::Emit).ok());
        for (auto& field : collector.fields) {
            if (field.first == 4U) {
                field.second = DbScalar{uint64_t{999}};
            }
        }
        ChannelCursor cursor;
        cursor.fields = &collector.fields;
        cursor.meta = &meta;
        OpDetail target;
        ResultSink sink = ResultSink::For(target);
        ASSERT_TRUE(meta.materializeRows(sink, &cursor, ChannelCursor::Next, ChannelCursor::ReadField).ok());
        EXPECT_EQ(target.sync_iter, 100U);
        EXPECT_EQ(target.end_ptr, 0U);
        for (const uint16_t requested : cursor.requestedFields) {
            EXPECT_NE(requested, 4U) << "派生列不得进入物化请求";
        }
    }

    // 有效数组长度 hook：字节数组只发射有效长度个字节，物化容量按有效长度约束。
    TEST(StorageCatalogTest, EffectiveBytesLengthHookBoundsArrayPayload)
    {
        ChannelTab channel;
        channel.jetty_num = 3;
        channel.jetty_id[0] = 11;
        channel.jetty_id[5] = 66;
        // 发射侧：只发射 3 个 uint64 元素。
        BorrowedMemberCapture capture;
        const auto& meta = RecordSchema<ChannelTab>::Meta();
        capture.meta = &meta;
        ASSERT_NE(meta.visitRecord, nullptr);
        ASSERT_TRUE(meta.visitRecord(&channel, &capture, CaptureBorrowedMember).ok());
        EXPECT_EQ(capture.bytesData, reinterpret_cast<const uint8_t*>(channel.jetty_id));
        EXPECT_EQ(capture.blobBytes, 3U * sizeof(uint64_t));

        // 物化侧：FillCtx 对 jetty_id 的请求容量被 hook 约束为有效长度，
        // 容量内元素回填、容量外元素保持默认。
        ChannelTab target;
        const auto roundTripped = ChannelRoundTrip(channel, target, meta);
        ASSERT_TRUE(roundTripped.ok());
        EXPECT_EQ(target.jetty_id[0], 11U);
        EXPECT_EQ(target.jetty_id[5], 0U);
    }

    // 主键读写由目录派生：主键成员指针来自 PrimaryKey 声明，不再手写 record.id。
    TEST(StorageCatalogTest, PrimaryKeyReadWriteUsesDeclaredColumn)
    {
        Device device;
        device.id = 42;
        EXPECT_EQ(RecordCodec<Device>::ReadPrimaryKey(device).value, 42U);
        ASSERT_TRUE(RecordCodec<Device>::WritePrimaryKey(device, PrimaryKey<Device>{7}).ok());
        EXPECT_EQ(device.id, 7U);

        ChannelTab channel;
        channel.jetty_num = 1;
        ASSERT_TRUE(RecordCodec<ChannelTab>::WritePrimaryKey(channel, PrimaryKey<ChannelTab>{9}).ok());
        EXPECT_EQ(channel.id, 9U);
    }

    // fingerprint：字段类型/字段名/版本/索引变化都会改变指纹（布局漂移可检测）。
    TEST(StorageCatalogTest, FingerprintTracksLayoutAndVersion)
    {
        constexpr auto deviceFp = Ddl::TableDdl<Device>::kFingerprint;
        constexpr auto layoutV2Fp = Ddl::TableDdl<DeviceLayoutV2>::kFingerprint;
        static_assert(deviceFp != 0);
        static_assert(layoutV2Fp != 0);
        // 同字段名不同 status 类型：指纹不同。
        EXPECT_NE(deviceFp, layoutV2Fp);
        // 表间指纹互不相同。
        EXPECT_NE(deviceFp, Ddl::TableDdl<OpDetail>::kFingerprint);
        EXPECT_NE(Ddl::TableDdl<OpDetail>::kFingerprint, Ddl::TableDdl<ChannelTab>::kFingerprint);
        // 版本进入指纹（OpDetail 为 Version 2 的表；版本数包含在指纹输入里）。
        EXPECT_EQ(Ddl::TableDdl<OpDetail>::kHead.version, 2U);
    }

    // 多逻辑库聚合：表名跨库查找、库归属枚举、总表数量由同一目录给出。
    TEST(StorageCatalogTest, CatalogAggregatesDatabasesAndTables)
    {
        EXPECT_EQ(kTestCatalog.TableCount(), 7U);
        EXPECT_NE(kTestCatalog.FindTable("Device"), nullptr);
        EXPECT_NE(kTestCatalog.FindTable("OpDetail"), nullptr);
        EXPECT_NE(kTestCatalog.FindTable("OpaqueRow"), nullptr);
        EXPECT_NE(kTestCatalog.FindTable("BoundedTextRow"), nullptr);
        EXPECT_EQ(kTestCatalog.FindTable("Missing"), nullptr);
        EXPECT_EQ(kTestCatalog.FindDatabase("CATALOG_TEST_RUNNER_DB")->tableCount, 5U);
        EXPECT_EQ(kTestCatalog.FindDatabase("CATALOG_TEST_OP_DB")->tableCount, 2U);
    }

    // 外键描述以 "目标表::成员" 携带目标；级联删除策略进入描述与 fingerprint。
    TEST(StorageCatalogTest, ForeignKeysCarryTargetNamesAndCascade)
    {
        const auto* opDetail = kTestCatalog.FindTable("OpDetail");
        EXPECT_EQ(opDetail->foreignKeyCount, 0U);

        const auto* opTask = kTestCatalog.FindTable("OpTask");
        ASSERT_NE(opTask, nullptr);
        ASSERT_EQ(opTask->foreignKeyCount, 1U);
        EXPECT_STREQ(opTask->foreignKeys[0].localColumn, "op_detail_id");
        EXPECT_STREQ(opTask->foreignKeys[0].target, "catalog_test::OpDetail::id");
        EXPECT_TRUE(opTask->foreignKeys[0].cascadeDelete);

        // 关系描述由目录 foreignKeys 单一承载（Relations() 动态 vector 已删除）：
        // 本地列解析到真实成员编号（op_detail_id 为声明顺序 1）。
        const auto* taskMeta = kTestCatalog.FindTable("OpTask");
        uint16_t localFieldId = HcclSim::Storage::kUnknownFieldId;
        for (std::size_t i = 0; i < taskMeta->fieldCount; ++i) {
            if (HcclSim::Storage::Ddl::TextEquals(taskMeta->fields[i].name, "op_detail_id")) {
                localFieldId = static_cast<uint16_t>(i);
                break;
            }
        }
        EXPECT_EQ(localFieldId, 1U);
        EXPECT_EQ(Ddl::TargetTableName(opTask->foreignKeys[0].target), std::string_view{"OpDetail"});
    }

    // 不透明 aggregate Blob 字段合同：嵌套 aggregate 成员显式声明 Blob 后是固定长度
    // 逻辑 BYTES 字段，byteWidth = sizeof(成员类型)；判据对标量/非平凡复制 aggregate 为 false
    // （普通 aggregate 不写 Blob、标量误标 Blob、Text+Blob、Blob+AutoIncrement 由
    // FieldPropsGate 编译期拒绝，本文件只以判据断言表达）。
    TEST(StorageCatalogTest, OpaqueAggregateBlobFieldRule)
    {
        static_assert(std::is_aggregate_v<OpaquePayload>);
        static_assert(std::is_trivially_copyable_v<OpaquePayload>);
        const auto& fields = Ddl::TableDdl<OpaqueRow>::kFields;
        ASSERT_EQ(Ddl::TableDdl<OpaqueRow>::kFieldCount, 2U);
        EXPECT_EQ(fields[1].name, std::string_view{"payload"});
        EXPECT_EQ(fields[1].type, LogicalFieldType::BYTES);
        EXPECT_FALSE(fields[1].primaryKey);
        EXPECT_EQ(Ddl::TableDdl<OpaqueRow>::kFields[1].byteWidth, sizeof(OpaquePayload));
        // 既有字段宽度：定宽标量与 C 数组为 sizeof(M)，变长逻辑值为 0。
        EXPECT_EQ(Ddl::TableDdl<OpaqueRow>::kFields[0].byteWidth, sizeof(uint64_t));
        EXPECT_EQ(Ddl::TableDdl<Device>::kFields[2].byteWidth, sizeof(Device::soc_version));
    }

    // 不透明 aggregate 发射：精确输出 sizeof(OpaquePayload) 个字节，内容与成员对象逐字节相同；
    // 通道往返按同一规则整体回填。
    TEST(StorageCatalogTest, OpaqueAggregateBlobEncodesExactBytes)
    {
        OpaqueRow row;
        row.id = 9;
        row.payload.tag = 0xABCD;
        row.payload.count = 7;
        row.payload.body[0] = 1;
        row.payload.body[5] = 9;
        BorrowedMemberCapture capture;
        const auto& meta = RecordSchema<OpaqueRow>::Meta();
        capture.meta = &meta;
        ASSERT_NE(meta.visitRecord, nullptr);
        ASSERT_TRUE(meta.visitRecord(&row, &capture, CaptureBorrowedMember).ok());
        EXPECT_EQ(capture.blobBytes, sizeof(OpaquePayload));
        EXPECT_EQ(capture.bytesData, reinterpret_cast<const uint8_t*>(&row.payload));

        OpaqueRow target;
        const auto roundTripped = ChannelRoundTrip(row, target, meta);
        ASSERT_TRUE(roundTripped.ok());
        EXPECT_EQ(target.id, 9U);
        EXPECT_EQ(0, std::memcmp(&target.payload, &row.payload, sizeof(OpaquePayload)));
    }

    // 满容量无 NUL 的 Text 数组：发射以 sizeof(成员) 为界——精确 N 个字符，不读取相邻
    // guard 字段；短字符串（含 NUL）语义不变；通道往返内容保持。
    TEST(StorageCatalogTest, TextArrayEmissionIsBoundedByMemberWidth)
    {
        BoundedTextRow row;
        row.id = 1;
        row.code[0] = 'A';
        row.code[1] = 'B';
        row.code[2] = 'C';
        row.code[3] = 'D';
        row.guard[0] = 'X';
        row.guard[1] = 'Y';
        row.guard[2] = 'Z';
        row.guard[3] = 'W';
        const auto& meta = RecordSchema<BoundedTextRow>::Meta();
        ChannelCollector collector;
        collector.meta = &meta;
        ASSERT_NE(meta.visitRecord, nullptr);
        ASSERT_TRUE(meta.visitRecord(&row, &collector, ChannelCollector::Emit).ok());
        ASSERT_EQ(collector.fields.size(), 3U);
        const auto* code = std::get_if<std::string>(&collector.fields[1].second);
        ASSERT_NE(code, nullptr);
        // 满容量无 NUL：精确 sizeof(code) = 4 个字符，guard 字节不得混入。
        EXPECT_EQ(*code, std::string{"ABCD"});
        const auto* guard = std::get_if<std::string>(&collector.fields[2].second);
        ASSERT_NE(guard, nullptr);
        EXPECT_EQ(*guard, std::string{"XYZW"});

        BoundedTextRow target;
        const auto roundTripped = ChannelRoundTrip(row, target, meta);
        ASSERT_TRUE(roundTripped.ok());
        // 回填按引擎同一规则：固定 Text 数组最多回写 N-1 个字符（尾字节保留 NUL），
        // 满容量 4 字符发射值回填为 3 字符。
        EXPECT_EQ(std::string{target.code}, "ABC");
        EXPECT_EQ(std::string{target.guard}, "XYZ");

        // 短字符串（NUL 前字节承载语义）往返不变。
        BoundedTextRow shortRow;
        shortRow.id = 2;
        shortRow.code[0] = 'A';
        shortRow.code[1] = 'B';
        BoundedTextRow shortTarget;
        const auto shortRound = ChannelRoundTrip(shortRow, shortTarget, meta);
        ASSERT_TRUE(shortRound.ok());
        EXPECT_EQ(std::string{shortTarget.code}, "AB");
    }

    // fingerprint 混入固定字节宽度：同 tableId、同字段名/逻辑类型/属性/主键的两个手工
    // descriptor 仅 byteWidth 不同（4 与 8），指纹必须不同（直接验证宽度进入指纹输入）。
    TEST(StorageCatalogTest, FingerprintTracksByteWidth)
    {
        constexpr Ddl::TableHead head{1U, Ddl::ScopeKind::Static, Ddl::MakePrimaryKey("value"), {}, 0U, {}, 0U};
        constexpr Ddl::FieldDesc narrow[]{Ddl::FieldDesc{"value", LogicalFieldType::UNSIGNED, 0U, true, 4U}};
        constexpr Ddl::FieldDesc wide[]{Ddl::FieldDesc{"value", LogicalFieldType::UNSIGNED, 0U, true, 8U}};
        constexpr auto narrowFp = Ddl::ComputeTableFingerprint("Row", head, narrow, 1U);
        constexpr auto wideFp = Ddl::ComputeTableFingerprint("Row", head, wide, 1U);
        static_assert(narrowFp != wideFp);
        EXPECT_NE(narrowFp, wideFp);
    }
} // namespace
} // namespace catalog_test
