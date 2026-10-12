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
 * See the License for details on how you should not use this file except in compliance with the License.
 */

// 文件职责：验证 MariaDbEngine 批量插入（InsertMany override）的全部合同：
//   - 入参校验（count=0/空指针/超 256/未 Begin）与基类默认实现逐字一致；
//   - 快路径（单条参数化多值 INSERT ... RETURNING 主键）：1 行与恰好 256 行
//     全自动主键批次的逐行真实主键（按输入顺序、非 0、唯一、可回查对账）、
//     全字段类型保真（TEXT/BLOB/高位 uint64/REAL/bool/空值语义）、Derived 派生
//     列物理值、动态 PID 表；
//   - 回退路径：显式/混合主键、Unique/FK 约束表、旧服务端（<10.5，经测试
//     seam 模拟）一律走既有逐行实现，功能与失败行号合同不裁剪；
//   - 中途失败（空指针/Unique 冲突/FK 冲突）首错 0 基行号 + 整笔回滚；
//   - 快路径执行后同连接立即 Select/Update（结果集已排空，无 out-of-sync）；
//   - 私有观察点证明 256 行合格批次只执行一次 batch statement。
// RED→GREEN：旧实现无 override（基类默认逐行 Insert）——batch-statement 计数
// 用例在旧实现上失败（计数恒 0）。endpoint 由 HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT
// 提供（真实 mysqld；推荐/验收环境 10.11）。
#include <gtest/gtest.h>

#include <mysql.h>

#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include "storage_backends/mariadb/engine/mariadb_engine.h"
#include "storage/record_contract.h"
#include "storage/storage_catalog.h"
#include "runtime_state/db_sim_runner_ops.h"

namespace mariadb_insert_many_test {
struct BatchRow {
    uint64_t id{0};
    std::string name;
    int64_t count{0};
    double score{0.0};
    std::vector<uint8_t> payload;
    uint64_t wide{0};
    bool flag{false};
};
struct UniqueNameRow {
    uint64_t id{0};
    std::string name;
};
struct FkParentRow {
    uint64_t id{0};
    uint64_t code{0};
};
struct FkChildRow {
    uint64_t id{0};
    uint64_t parentCode{0};
};
struct DerivedRow {
    uint64_t id{0};
    uint64_t base{0};
    uint64_t size{0};
    uint64_t endPtr{0};
};
struct DynRow {
    uint64_t id{0};
    std::string note;
};
} // namespace mariadb_insert_many_test

using namespace HcclSim::Storage::Ddl;
#define Field HCCL_VM_DDL_FIELD
#define Version HCCL_VM_DDL_VERSION
#define PrimaryKey HCCL_VM_DDL_PRIMARY_KEY
#define Unique HCCL_VM_DDL_UNIQUE
#define Index HCCL_VM_DDL_INDEX
#define ForeignKey HCCL_VM_DDL_FOREIGN_KEY
#define Scope Ddl::ScopeKind
#define Derived Ddl::Derived

#define MARIADB_INSERT_MANY_TEST_DATABASE(Table)                                                                       \
    Table(                                                                                                             \
        mariadb_insert_many_test::BatchRow,                                                                            \
        (Version(1), Scope::Static, Field(id, AutoIncrement), Field(name), Field(count), Field(score), Field(payload), \
         Field(wide), Field(flag), PrimaryKey(id)))                                                                    \
        Table(                                                                                                         \
            mariadb_insert_many_test::UniqueNameRow,                                                                   \
            (Version(1), Scope::Static, Field(id, AutoIncrement), Field(name), PrimaryKey(id), Unique(name)))          \
            Table(                                                                                                     \
                mariadb_insert_many_test::FkParentRow,                                                                 \
                (Version(1), Scope::Static, Field(id, AutoIncrement), Field(code), PrimaryKey(id), Unique(code)))      \
                Table(                                                                                                 \
                    mariadb_insert_many_test::FkChildRow,                                                              \
                    (Version(1), Scope::Static, Field(id, AutoIncrement), Field(parentCode), PrimaryKey(id),           \
                     ForeignKey(parentCode, mariadb_insert_many_test::FkParentRow::code, OnDelete::Cascade)))          \
                    Table(                                                                                             \
                        mariadb_insert_many_test::DerivedRow,                                                          \
                        (Version(1), Scope::Static, Field(id, AutoIncrement), Field(base), Field(size),                \
                         Field(endPtr, Derived), PrimaryKey(id)))                                                      \
                        Table(                                                                                         \
                            mariadb_insert_many_test::DynRow,                                                          \
                            (Version(1), Scope::DynamicPid, Field(id, AutoIncrement), Field(note), PrimaryKey(id)))

#define HCCL_VM_DDL_CURRENT_DATABASE "MARIADB_INSERT_MANY_TEST_DB"
HCCL_VM_DEFINE_DATABASE(kMariadbInsertManyTestDatabase, MARIADB_INSERT_MANY_TEST_DB, MARIADB_INSERT_MANY_TEST_DATABASE)
#undef HCCL_VM_DDL_CURRENT_DATABASE

#undef Field
#undef Version
#undef PrimaryKey
#undef Unique
#undef Index
#undef ForeignKey
#undef Scope
#undef Derived

namespace HcclSim::Storage::Ddl {
// DerivedRow 派生列 hook：endPtr = base + size（编码入库；解码忽略，基址/大小权威）
template <>
struct RecordCodecHooks<mariadb_insert_many_test::DerivedRow> :
    DefaultRecordCodecHooks<mariadb_insert_many_test::DerivedRow> {
    static std::optional<DbScalar>
    EncodeDerivedField(const mariadb_insert_many_test::DerivedRow& record, uint16_t fieldId)
    {
        if (fieldId == 3U) {
            return DbScalar{record.base + record.size};
        }
        return std::nullopt;
    }

    static bool IgnoreDecodedField(uint16_t fieldId) { return fieldId == 3U; }
};
} // namespace HcclSim::Storage::Ddl

namespace {
using namespace HcclSim::Storage;
using mariadb_insert_many_test::BatchRow;

/// 每用例唯一 namespace 计数：物理 database 在服务器上持久存在，同一进程
/// 内共享 namespace 会让先前用例的已提交行污染后续用例的计数断言——
/// 唯一化保证用例间零数据耦合（镜像既有 mariadb_engine_test 的
/// EngineFixture 每用例唯一 namespace 约定）
std::atomic<uint64_t> gNamespaceCounter{0};

constexpr char kMariadbEndpointEnv[] = "HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT";

bool EndpointConfigured()
{
    const char* endpoint = std::getenv(kMariadbEndpointEnv);
    return endpoint != nullptr && endpoint[0] != '\0';
}

HcclSim::Storage::MariadbConnectionProfile TestProfile()
{
    HcclSim::Storage::MariadbConnectionProfile profile;
    profile.uri = std::string{std::getenv(kMariadbEndpointEnv)};
    profile.connectionEstablishmentTimeoutS = 3;
    profile.packetReadTimeoutS = 60;
    profile.packetWriteTimeoutS = 60;
    return profile;
}

/// 简易 endpoint 解析（测试专用：user:pass@host:port，不含特殊字符凭据）
struct ParsedEndpoint {
    std::string host;
    std::string user;
    std::string password;
    unsigned port{3306};
};

std::optional<ParsedEndpoint> ParseEndpointForTest(const std::string& uri)
{
    ParsedEndpoint parsed;
    const std::string prefix = "mariadb://";
    if (uri.compare(0, prefix.size(), prefix) != 0) {
        return std::nullopt;
    }
    const std::string rest = uri.substr(prefix.size());
    const auto at = rest.find('@');
    if (at == std::string::npos) {
        return std::nullopt;
    }
    const std::string credentials = rest.substr(0, at);
    const std::string hostPort = rest.substr(at + 1);
    const auto colon = credentials.find(':');
    if (colon == std::string::npos) {
        return std::nullopt;
    }
    parsed.user = credentials.substr(0, colon);
    parsed.password = credentials.substr(colon + 1);
    const auto portColon = hostPort.rfind(':');
    if (portColon == std::string::npos) {
        parsed.host = hostPort;
    } else {
        parsed.host = hostPort.substr(0, portColon);
        parsed.port = static_cast<unsigned>(std::strtoul(hostPort.substr(portColon + 1).c_str(), nullptr, 10));
    }
    return parsed;
}

/// 独立短连接读目标表单个整数列（Derived 物理值核对；不经过被测引擎）
std::optional<uint64_t>
ReadUint64Column(const std::string& database, const std::string& table, const std::string& column, uint64_t id)
{
    const auto endpoint = ParseEndpointForTest(std::string{std::getenv(kMariadbEndpointEnv)});
    if (!endpoint.has_value()) {
        return std::nullopt;
    }
    MYSQL* mysql = mysql_init(nullptr);
    if (mysql == nullptr) {
        return std::nullopt;
    }
    if (mysql_real_connect(
            mysql, endpoint->host.c_str(), endpoint->user.c_str(), endpoint->password.c_str(), database.c_str(),
            endpoint->port, nullptr, 0)
        == nullptr) {
        mysql_close(mysql);
        return std::nullopt;
    }
    std::optional<uint64_t> value;
    const std::string sql = "SELECT `" + column + "` FROM `" + table + "` WHERE `id` = ?";
    MYSQL_STMT* stmt = mysql_stmt_init(mysql);
    if (stmt != nullptr && mysql_stmt_prepare(stmt, sql.c_str(), static_cast<unsigned long>(sql.size())) == 0) {
        MYSQL_BIND param{};
        uint64_t key = id;
        param.buffer_type = MYSQL_TYPE_LONGLONG;
        param.buffer = &key;
        param.buffer_length = sizeof(key);
        param.is_unsigned = 1;
        if (mysql_stmt_bind_param(stmt, &param) == 0 && mysql_stmt_execute(stmt) == 0) {
            uint64_t out = 0;
            my_bool isNull = 0;
            unsigned long length = 0;
            MYSQL_BIND result{};
            result.buffer_type = MYSQL_TYPE_LONGLONG;
            result.buffer = &out;
            result.buffer_length = sizeof(out);
            result.is_unsigned = 1;
            result.is_null = &isNull;
            result.length = &length;
            if (mysql_stmt_bind_result(stmt, &result) == 0) {
                const int rc = mysql_stmt_fetch(stmt);
                if ((rc == 0 || rc == MYSQL_DATA_TRUNCATED) && isNull == 0) {
                    value = out;
                }
            }
        }
    }
    if (stmt != nullptr) {
        (void)mysql_stmt_close(stmt);
    }
    mysql_close(mysql);
    return value;
}
} // namespace

namespace HcclSim::Storage {
/// 测试观察点（产品代码不得引用；本名仅落在本测试二进制内）。编译期探测
/// 定义在本 friend 结构内——对引擎私有测试入口的探测表达式在 friend 语境
/// 获得访问权；旧实现无对应入口时退化为 false_type（观察点计数恒 0 /
/// 版本 seam 不可用即 RED 证据或显式 skip）。
struct MariadbEngineTestPeer {
    template <typename Engine, typename = void>
    struct HasBatchInsertExecutions : std::false_type {};

    template <typename Engine>
    struct HasBatchInsertExecutions<
        Engine, std::void_t<decltype(std::declval<Engine&>().BatchInsertExecutionsForTest())>> : std::true_type {};

    template <typename Engine, typename = void>
    struct HasServerVersionOverride : std::false_type {};

    template <typename Engine>
    struct HasServerVersionOverride<
        Engine, std::void_t<decltype(std::declval<Engine&>().SetServerVersionForTest(std::declval<unsigned long>()))>> :
        std::true_type {};

    /// 模板函数：if constexpr 的假分支在旧引擎（无对应入口）下被丢弃，
    /// 不实例化调用
    template <typename Engine>
    static uint64_t BatchInsertExecutions(const Engine& engine)
    {
        if constexpr (HasBatchInsertExecutions<Engine>::value) {
            return engine.BatchInsertExecutionsForTest();
        }
        return 0;
    }

    template <typename Engine>
    static void SetServerVersionOverride(Engine& engine, unsigned long version)
    {
        if constexpr (HasServerVersionOverride<Engine>::value) {
            engine.SetServerVersionForTest(version);
        }
    }
};
} // namespace HcclSim::Storage

namespace {
/// 夹具：唯一 namespace 的全新 database + 真实 endpoint 引擎
class MariadbInsertManyTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        if (!EndpointConfigured()) {
            GTEST_SKIP() << "HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 未设置：真实 mysqld 批插合同需要外部 endpoint";
        }
        ResetEngine(&Ddl::kMariadbInsertManyTestDatabase);
    }

    void TearDown() override
    {
        if (engine != nullptr && ready) {
            (void)engine->Close();
        }
    }

    void ResetEngine(const Ddl::DatabaseDescriptor* catalog)
    {
        if (engine != nullptr && ready) {
            (void)engine->Close();
        }
        ready = false;
        dataNamespace = "mdb_insmany_p" + std::to_string(static_cast<long>(::getpid())) + "_"
                        + std::to_string(gNamespaceCounter.fetch_add(1));
        engine = std::make_unique<MariaDbEngine>(catalog, TestProfile());
        EngineOpenRequest request;
        request.mode = StorageOpenMode::READ_WRITE_CREATE;
        request.dataNamespace = dataNamespace;
        const auto opened = engine->Open(request);
        ASSERT_TRUE(opened.ok()) << opened.diagnostic;
        const auto prepared = engine->Prepare();
        ASSERT_TRUE(prepared.ok()) << prepared.diagnostic;
        ready = true;
    }

    const Ddl::CatalogTableMeta& MetaOf(const std::string_view tableId) const
    {
        for (std::size_t i = 0; i < Ddl::kMariadbInsertManyTestDatabase.tableCount; ++i) {
            if (Ddl::kMariadbInsertManyTestDatabase.tables[i].tableId == tableId) {
                return Ddl::kMariadbInsertManyTestDatabase.tables[i];
            }
        }
        ADD_FAILURE() << "test table missing from catalog: " << tableId;
        static const Ddl::CatalogTableMeta kEmpty{};
        return kEmpty;
    }

    /// 行数组 → 原始指针视图（地址仅在本调用栈内使用，不复制业务载荷）
    template <typename T>
    static std::vector<const void*> PointerView(const std::vector<T>& rows)
    {
        std::vector<const void*> pointers;
        pointers.reserve(rows.size());
        for (const T& row : rows) {
            pointers.push_back(&row);
        }
        return pointers;
    }

    /// 造 N 行可区分数据（name/count/score/payload/wide/flag 按 i 变化）
    static void MakeRows(std::vector<BatchRow>& rows, std::size_t count)
    {
        rows.clear();
        for (std::size_t i = 0; i < count; ++i) {
            BatchRow row;
            row.name = "row-" + std::to_string(i) + "-你好";
            row.count = static_cast<int64_t>(i) * 7 - 3;
            row.score = 0.5 * static_cast<double>(i);
            row.payload = {static_cast<uint8_t>(i), 0xFEU, 0x00U, static_cast<uint8_t>(i + 1U)};
            if (i == 0U) {
                row.payload.clear(); // 空载荷与空文本语义
            }
            row.wide = (std::numeric_limits<uint64_t>::max() - 3U) - i;
            row.flag = (i % 2U) == 0U;
            rows.push_back(row);
        }
        if (count > 1U) {
            rows[1].name = ""; // 空文本：零长度值，不是 NULL
        }
    }

    /// 按 name 回查 id（证明 generatedIds 与输入行的真实对应，而非顺序假设）
    std::optional<uint64_t> SelectIdByName(const std::string& name)
    {
        std::vector<BatchRow> rows;
        auto sink = ResultSink::For(rows);
        const auto where = ConditionData{ConditionClause{
            static_cast<uint16_t>(1), Compare::EQUAL, ConditionJoin::NONE, 0, 0, ConditionValue{DbScalar{name}}}};
        const auto selected = engine->Select(MetaOf("BatchRow"), TableScope::Static(), where, QueryOptions{}, sink);
        if (!selected.ok() || rows.size() != 1U) {
            return std::nullopt;
        }
        return rows[0].id;
    }

    std::unique_ptr<MariaDbEngine> engine;
    std::string dataNamespace;
    bool ready{false};
};
} // namespace

// 入参合同：count=0 平凡成功；空 records/generatedIds/transaction 与超 256 行
// 的拒绝消息与基类默认实现逐字一致（本 override 必须先于路径决策完成同一校验）
TEST_F(MariadbInsertManyTest, InputContractMatchesBaseDefault)
{
    ASSERT_TRUE(ready);
    const auto& meta = MetaOf("BatchRow");
    TransactionBinding transaction;
    ASSERT_TRUE(engine->Begin(transaction).ok());

    std::vector<BatchRow> rows;
    MakeRows(rows, 3);
    auto pointers = PointerView(rows);
    std::array<uint64_t, 3> ids{};

    const auto trivial = engine->InsertMany(meta, TableScope::Static(), nullptr, 0U, nullptr, &transaction);
    EXPECT_TRUE(trivial.ok());

    const auto noRecords = engine->InsertMany(meta, TableScope::Static(), nullptr, 3U, ids.data(), &transaction);
    EXPECT_EQ(noRecords.code, DbCode::INVALID_ARGUMENT);
    EXPECT_NE(
        noRecords.diagnostic.find("insert-many requires records, an id array and an active transaction"),
        std::string::npos)
        << noRecords.diagnostic;

    const auto noIds = engine->InsertMany(meta, TableScope::Static(), pointers.data(), 3U, nullptr, &transaction);
    EXPECT_EQ(noIds.code, DbCode::INVALID_ARGUMENT);

    const auto noTransaction = engine->InsertMany(meta, TableScope::Static(), pointers.data(), 3U, ids.data(), nullptr);
    EXPECT_EQ(noTransaction.code, DbCode::INVALID_ARGUMENT);

    const auto tooMany = engine->InsertMany(
        meta, TableScope::Static(), pointers.data(), kInsertBatchMaxRows + 1U, ids.data(), &transaction);
    EXPECT_EQ(tooMany.code, DbCode::INVALID_ARGUMENT);
    EXPECT_NE(tooMany.diagnostic.find("insert-many batch exceeds kInsertBatchMaxRows"), std::string::npos)
        << tooMany.diagnostic;

    ASSERT_TRUE(engine->Rollback(transaction).ok());
    const auto counted = engine->Count(MetaOf("BatchRow"), TableScope::Static(), ConditionData{});
    ASSERT_TRUE(counted.ok()) << counted.diagnostic;
    EXPECT_EQ(*counted.value, 0U) << "被拒绝的入参不得产生任何写入";
}

// 快路径：1 行与恰好 256 行全自动主键批次——真实主键（非 0、唯一、按输入
// 顺序回填且与行内容对账）、全字段类型保真、affectedRows=count。两批各自
// 落入全新 database（同表两批的行名会互相重复，回查/全表对账无法区分）。
TEST_F(MariadbInsertManyTest, FastPathAutoIdsRoundtripSingleAndMax)
{
    ASSERT_TRUE(ready);
    const auto& meta = MetaOf("BatchRow");
    if (!MariadbEngineTestPeer::HasBatchInsertExecutions<MariaDbEngine>::value) {
        // 旧实现：整批观察点不可用（计数恒 0）——本用例即为 RED 证据
        ADD_FAILURE() << "旧实现：MariaDbEngine 未提供真批插（batch statement 计数恒 0）";
        return;
    }
    uint64_t totalBatchExecutions = 0;

    for (const std::size_t count : {std::size_t{1U}, kInsertBatchMaxRows}) {
        if (count != 1U) {
            ResetEngine(&Ddl::kMariadbInsertManyTestDatabase); // 每批全新 database
        }
        const uint64_t before = MariadbEngineTestPeer::BatchInsertExecutions(*engine);
        TransactionBinding transaction;
        ASSERT_TRUE(engine->Begin(transaction).ok());
        std::vector<BatchRow> rows;
        MakeRows(rows, count);
        auto pointers = PointerView(rows);
        std::vector<uint64_t> ids(count, 0U);
        const auto inserted
            = engine->InsertMany(meta, TableScope::Static(), pointers.data(), count, ids.data(), &transaction);
        ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
        EXPECT_EQ(inserted.affectedRows, count);
        const auto committed = engine->Commit(transaction);
        ASSERT_TRUE(committed.ok()) << committed.diagnostic;
        totalBatchExecutions += MariadbEngineTestPeer::BatchInsertExecutions(*engine) - before;

        // 主键：非 0、互异、与行内容一一对应（name → id 对账，不依赖顺序假设）
        std::set<uint64_t> unique;
        for (std::size_t i = 0; i < count; ++i) {
            EXPECT_NE(ids[i], 0U) << "行 " << i << " 的自增主键不得为 0";
            unique.insert(ids[i]);
            const auto byName = SelectIdByName(rows[i].name);
            ASSERT_TRUE(byName.has_value()) << "行 " << i << "（name=" << rows[i].name << "）回查失败";
            EXPECT_EQ(*byName, ids[i]) << "行 " << i << " 的回填主键与行内容不对应";
        }
        EXPECT_EQ(unique.size(), count);

        // 字段保真抽查（全量逐字段对 256 行）
        std::vector<BatchRow> back;
        auto sink = ResultSink::For(back);
        ASSERT_TRUE(engine->Select(meta, TableScope::Static(), ConditionData{}, QueryOptions{}, sink).ok());
        ASSERT_EQ(back.size(), count);
        std::sort(back.begin(), back.end(), [](const BatchRow& left, const BatchRow& right) {
            return left.count < right.count;
        });
        for (std::size_t i = 0; i < count; ++i) {
            EXPECT_EQ(back[i].name, rows[i].name);
            EXPECT_EQ(back[i].count, rows[i].count);
            EXPECT_DOUBLE_EQ(back[i].score, rows[i].score);
            EXPECT_EQ(back[i].payload, rows[i].payload);
            EXPECT_EQ(back[i].wide, rows[i].wide);
            EXPECT_EQ(back[i].flag, rows[i].flag);
        }
    }

    // 观察点：两批合格数据各执行一次 batch statement
    EXPECT_EQ(totalBatchExecutions, 2U);
}

// 回退：显式/混合主键批次走既有逐行路径，显式值保留、自动值生成
TEST_F(MariadbInsertManyTest, ExplicitAndMixedIdsFallBackToRowByRow)
{
    ASSERT_TRUE(ready);
    const auto& meta = MetaOf("BatchRow");
    const uint64_t before = MariadbEngineTestPeer::BatchInsertExecutions(*engine);

    TransactionBinding transaction;
    ASSERT_TRUE(engine->Begin(transaction).ok());
    std::vector<BatchRow> rows;
    MakeRows(rows, 4);
    rows[0].id = 1001U; // 显式
    rows[1].id = 0U;    // 自动（混合批次）
    rows[2].id = 1003U; // 显式
    rows[3].id = 0U;    // 自动
    auto pointers = PointerView(rows);
    std::vector<uint64_t> ids(4, 0U);
    const auto inserted = engine->InsertMany(meta, TableScope::Static(), pointers.data(), 4U, ids.data(), &transaction);
    ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
    ASSERT_TRUE(engine->Commit(transaction).ok());

    EXPECT_EQ(ids[0], 1001U);
    EXPECT_EQ(ids[2], 1003U);
    EXPECT_GT(ids[1], 0U);
    EXPECT_GT(ids[3], 0U);
    EXPECT_NE(ids[1], ids[3]);

    // 混合批次不走快路径（观察点计数不变）
    EXPECT_EQ(MariadbEngineTestPeer::BatchInsertExecutions(*engine), before);
}

// 中途空指针：首错 0 基行号保留（回退逐行路径报告），整笔回滚
TEST_F(MariadbInsertManyTest, NullRecordMidBatchReportsRowIndexAndRollsBack)
{
    ASSERT_TRUE(ready);
    const auto& meta = MetaOf("BatchRow");
    TransactionBinding transaction;
    ASSERT_TRUE(engine->Begin(transaction).ok());
    std::vector<BatchRow> rows;
    MakeRows(rows, 4);
    auto pointers = PointerView(rows);
    pointers[2] = nullptr;
    std::vector<uint64_t> ids(4, 0U);
    const auto inserted = engine->InsertMany(meta, TableScope::Static(), pointers.data(), 4U, ids.data(), &transaction);
    ASSERT_FALSE(inserted.ok());
    EXPECT_NE(inserted.diagnostic.find("at row 2"), std::string::npos) << inserted.diagnostic;

    ASSERT_TRUE(engine->Rollback(transaction).ok());
    const auto counted = engine->Count(meta, TableScope::Static(), ConditionData{});
    ASSERT_TRUE(counted.ok()) << counted.diagnostic;
    EXPECT_EQ(*counted.value, 0U) << "整笔回滚：已写入的前缀行不得残留";
}

// Unique 中间冲突：首错行号（重复名所在行）保留，整笔回滚
TEST_F(MariadbInsertManyTest, UniqueConflictMidBatchReportsRowIndexAndRollsBack)
{
    ASSERT_TRUE(ready);
    const auto& meta = MetaOf("UniqueNameRow");
    TransactionBinding transaction;
    ASSERT_TRUE(engine->Begin(transaction).ok());

    std::vector<mariadb_insert_many_test::UniqueNameRow> rows;
    for (std::size_t i = 0; i < 5; ++i) {
        mariadb_insert_many_test::UniqueNameRow row;
        // 第 1、3 行同名 "dup"：第 1 行先入，第 3 行撞唯一索引——首错 0 基
        // 行号必须是 3（第二个重复行）
        row.name = (i == 1U || i == 3U) ? "dup" : "uniq-" + std::to_string(i);
        rows.push_back(row);
    }
    auto pointers = PointerView(rows);
    std::vector<uint64_t> ids(5, 0U);
    const auto inserted = engine->InsertMany(meta, TableScope::Static(), pointers.data(), 5U, ids.data(), &transaction);
    ASSERT_FALSE(inserted.ok());
    EXPECT_NE(inserted.diagnostic.find("at row 3"), std::string::npos) << inserted.diagnostic;

    ASSERT_TRUE(engine->Rollback(transaction).ok());
    const auto counted = engine->Count(meta, TableScope::Static(), ConditionData{});
    ASSERT_TRUE(counted.ok()) << counted.diagnostic;
    EXPECT_EQ(*counted.value, 0U);
}

// FK 中间冲突：首错行号保留，整笔回滚
TEST_F(MariadbInsertManyTest, FkConflictMidBatchReportsRowIndexAndRollsBack)
{
    ASSERT_TRUE(ready);
    const auto& childMeta = MetaOf("FkChildRow");
    const auto& parentMeta = MetaOf("FkParentRow");

    TransactionBinding seed;
    ASSERT_TRUE(engine->Begin(seed).ok());
    mariadb_insert_many_test::FkParentRow parent;
    parent.code = 77U;
    ASSERT_TRUE(engine->Insert(parentMeta, TableScope::Static(), &parent, &seed).ok());
    ASSERT_TRUE(engine->Commit(seed).ok());

    TransactionBinding transaction;
    ASSERT_TRUE(engine->Begin(transaction).ok());
    std::vector<mariadb_insert_many_test::FkChildRow> rows;
    for (std::size_t i = 0; i < 4; ++i) {
        mariadb_insert_many_test::FkChildRow row;
        row.parentCode = (i == 1U) ? 999U : 77U; // 第 1 行引用不存在的父行
        rows.push_back(row);
    }
    auto pointers = PointerView(rows);
    std::vector<uint64_t> ids(4, 0U);
    const auto inserted
        = engine->InsertMany(childMeta, TableScope::Static(), pointers.data(), 4U, ids.data(), &transaction);
    ASSERT_FALSE(inserted.ok());
    EXPECT_NE(inserted.diagnostic.find("at row 1"), std::string::npos) << inserted.diagnostic;

    ASSERT_TRUE(engine->Rollback(transaction).ok());
    const auto counted = engine->Count(childMeta, TableScope::Static(), ConditionData{});
    ASSERT_TRUE(counted.ok()) << counted.diagnostic;
    EXPECT_EQ(*counted.value, 0U);
}

// Derived 派生列 + 快路径：base/size 往返、endPtr 物理值 = base+size（独立短连接核对）
TEST_F(MariadbInsertManyTest, DerivedColumnPhysicalValueOnFastPath)
{
    ASSERT_TRUE(ready);
    const auto& meta = MetaOf("DerivedRow");
    const uint64_t before = MariadbEngineTestPeer::BatchInsertExecutions(*engine);

    TransactionBinding transaction;
    ASSERT_TRUE(engine->Begin(transaction).ok());
    std::vector<mariadb_insert_many_test::DerivedRow> rows;
    for (std::size_t i = 0; i < 3; ++i) {
        mariadb_insert_many_test::DerivedRow row;
        row.base = 0x1000U + i;
        row.size = 0x40U + i;
        rows.push_back(row);
    }
    auto pointers = PointerView(rows);
    std::vector<uint64_t> ids(3, 0U);
    const auto inserted = engine->InsertMany(meta, TableScope::Static(), pointers.data(), 3U, ids.data(), &transaction);
    ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
    ASSERT_TRUE(engine->Commit(transaction).ok());
    EXPECT_EQ(MariadbEngineTestPeer::BatchInsertExecutions(*engine) - before, 1U)
        << "Derived 非主键字段不破坏快路径资格";

    std::vector<mariadb_insert_many_test::DerivedRow> back;
    auto sink = ResultSink::For(back);
    ASSERT_TRUE(engine->Select(meta, TableScope::Static(), ConditionData{}, QueryOptions{}, sink).ok());
    ASSERT_EQ(back.size(), 3U);
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(back[i].base, rows[i].base);
        EXPECT_EQ(back[i].size, rows[i].size);
    }

    const std::string database = "hcclvm_" + dataNamespace + "_MARIADB_INSERT_MANY_TEST_DB";
    for (std::size_t i = 0; i < 3; ++i) {
        const auto physical = ReadUint64Column(database, "DerivedRow", "endPtr", ids[i]);
        ASSERT_TRUE(physical.has_value()) << "独立连接读取 DerivedRow.endPtr 失败";
        EXPECT_EQ(*physical, rows[i].base + rows[i].size) << "派生列物理值必须是 base+size";
    }
}

// 动态 PID 表 + 快路径：物理表 <table>_P_<pid> 落库正确
TEST_F(MariadbInsertManyTest, DynamicPidTableFastPath)
{
    ASSERT_TRUE(ready);
    const auto& meta = MetaOf("DynRow");
    const uint64_t before = MariadbEngineTestPeer::BatchInsertExecutions(*engine);

    TransactionBinding transaction;
    ASSERT_TRUE(engine->Begin(transaction).ok());
    std::vector<mariadb_insert_many_test::DynRow> rows;
    for (std::size_t i = 0; i < 4; ++i) {
        mariadb_insert_many_test::DynRow row;
        row.note = "dyn-" + std::to_string(i);
        rows.push_back(row);
    }
    auto pointers = PointerView(rows);
    std::vector<uint64_t> ids(4, 0U);
    const auto inserted
        = engine->InsertMany(meta, TableScope::Process(31), pointers.data(), 4U, ids.data(), &transaction);
    ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
    ASSERT_TRUE(engine->Commit(transaction).ok());
    EXPECT_EQ(MariadbEngineTestPeer::BatchInsertExecutions(*engine) - before, 1U);

    const auto counted = engine->Count(meta, TableScope::Process(31), ConditionData{});
    ASSERT_TRUE(counted.ok()) << counted.diagnostic;
    EXPECT_EQ(*counted.value, 4U);
    const auto other = engine->Count(meta, TableScope::Process(32), ConditionData{});
    ASSERT_TRUE(other.ok()) << other.diagnostic;
    EXPECT_EQ(*other.value, 0U) << "相邻 pid 的物理表互不可见";
}

// 快路径执行后同连接立即 Select/Update：结果集已排空、无 commands out of sync
TEST_F(MariadbInsertManyTest, ConnectionReusableImmediatelyAfterFastPath)
{
    ASSERT_TRUE(ready);
    const auto& meta = MetaOf("BatchRow");
    TransactionBinding transaction;
    ASSERT_TRUE(engine->Begin(transaction).ok());
    std::vector<BatchRow> rows;
    MakeRows(rows, 256);
    auto pointers = PointerView(rows);
    std::vector<uint64_t> ids(256, 0U);
    ASSERT_TRUE(engine->InsertMany(meta, TableScope::Static(), pointers.data(), 256U, ids.data(), &transaction).ok());
    ASSERT_TRUE(engine->Commit(transaction).ok());

    // 立即查询（同连接）
    const auto counted = engine->Count(meta, TableScope::Static(), ConditionData{});
    ASSERT_TRUE(counted.ok()) << counted.diagnostic;
    EXPECT_EQ(*counted.value, 256U);

    // 立即条件更新（同连接，参数化 DML）
    const auto updated = engine->Update(
        meta, TableScope::Static(),
        ConditionData{ConditionClause{
            static_cast<uint16_t>(1), Compare::EQUAL, ConditionJoin::NONE, 0, 0,
            ConditionValue{DbScalar{std::string{"row-7-你好"}}}}},
        UpdateData{LogicalChange{static_cast<uint16_t>(2), ChangeKind::SET, DbScalar{int64_t{700}}}}, nullptr);
    ASSERT_TRUE(updated.ok()) << updated.diagnostic;
    EXPECT_EQ(updated.value.value_or(0U), 1U);
}

// 观察点：256 行合格批次只执行一次 batch statement；同形第二批复用语句缓存
TEST_F(MariadbInsertManyTest, BatchStatementExecutedOncePerEligibleBatch)
{
    ASSERT_TRUE(ready);
    const auto& meta = MetaOf("BatchRow");
    if (!MariadbEngineTestPeer::HasBatchInsertExecutions<MariaDbEngine>::value) {
        ADD_FAILURE() << "旧实现：MariaDbEngine 未提供真批插（batch statement 计数恒 0）——RED 证据";
        return;
    }
    const uint64_t before = MariadbEngineTestPeer::BatchInsertExecutions(*engine);

    for (int round = 0; round < 2; ++round) {
        TransactionBinding transaction;
        ASSERT_TRUE(engine->Begin(transaction).ok());
        std::vector<BatchRow> rows;
        MakeRows(rows, 256);
        auto pointers = PointerView(rows);
        std::vector<uint64_t> ids(256, 0U);
        ASSERT_TRUE(
            engine->InsertMany(meta, TableScope::Static(), pointers.data(), 256U, ids.data(), &transaction).ok())
            << "round " << round;
        ASSERT_TRUE(engine->Commit(transaction).ok());
    }
    EXPECT_EQ(MariadbEngineTestPeer::BatchInsertExecutions(*engine) - before, 2U);
}

// 生产表命中：RaSocketPair 元数据（自增单列主键、无 Unique/FK）+ id=0 数据
// 走快路径；不修改 proxy 生产调用
TEST_F(MariadbInsertManyTest, RaSocketPairProductionShapeHitsFastPath)
{
    if (!EndpointConfigured()) {
        GTEST_SKIP() << "HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 未设置";
    }
    // 换装 RUNNER_DB 目录（含 RaSocketPair 生产表定义）
    if (engine != nullptr && ready) {
        (void)engine->Close();
        ready = false;
    }
    dataNamespace = "mdb_insmany_runner_p" + std::to_string(static_cast<long>(::getpid()));
    engine = std::make_unique<MariaDbEngine>(&Ddl::kRunnerDatabase, TestProfile());
    EngineOpenRequest request;
    request.mode = StorageOpenMode::READ_WRITE_CREATE;
    request.dataNamespace = dataNamespace;
    ASSERT_TRUE(engine->Open(request).ok());
    ASSERT_TRUE(engine->Prepare().ok());
    ready = true;

    const auto& meta = RecordSchema<sim::runtime::RaSocketPair>::Meta();
    const uint64_t before = MariadbEngineTestPeer::BatchInsertExecutions(*engine);

    TransactionBinding transaction;
    ASSERT_TRUE(engine->Begin(transaction).ok());
    std::vector<sim::runtime::RaSocketPair> rows;
    for (std::size_t i = 0; i < 8; ++i) {
        sim::runtime::RaSocketPair row{};
        row.id = 0; // 生产调用形态（hccp_ra_socket_stub 置 0 由后端生成）
        row.server_id = 0;
        row.client_id = 0x5f1400000000U + i;
        row.ref_cnt = 0;
        row.port = 16666;
        row.tag_hash = 0xAA0000U + i;
        row.buf_status = 0;
        row.slot_idx = static_cast<uint32_t>(i * 8U);
        rows.push_back(row);
    }
    auto pointers = PointerView(rows);
    std::vector<uint64_t> ids(8, 0U);
    const auto inserted = engine->InsertMany(meta, TableScope::Static(), pointers.data(), 8U, ids.data(), &transaction);
    ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
    ASSERT_TRUE(engine->Commit(transaction).ok());

    for (std::size_t i = 0; i < 8; ++i) {
        EXPECT_NE(ids[i], 0U);
    }
    EXPECT_EQ(MariadbEngineTestPeer::BatchInsertExecutions(*engine) - before, 1U)
        << "RaSocketPair 合格批次必须命中快路径";

    // 数据回读对账
    std::vector<sim::runtime::RaSocketPair> back;
    auto sink = ResultSink::For(back);
    ASSERT_TRUE(engine->Select(meta, TableScope::Static(), ConditionData{}, QueryOptions{}, sink).ok());
    ASSERT_EQ(back.size(), 8U);
    for (std::size_t i = 0; i < 8; ++i) {
        EXPECT_EQ(back[i].client_id, rows[i].client_id);
        EXPECT_EQ(back[i].tag_hash, rows[i].tag_hash);
        EXPECT_EQ(back[i].slot_idx, rows[i].slot_idx);
    }
}

// 旧服务端（<10.5）回退：seam 模拟版本后自动走既有逐行实现，功能保持
TEST_F(MariadbInsertManyTest, OldServerVersionFallsBackToRowByRow)
{
    ASSERT_TRUE(ready);
    if (!MariadbEngineTestPeer::HasServerVersionOverride<MariaDbEngine>::value) {
        GTEST_SKIP() << "当前实现未提供版本测试 seam（本用例仅验证回退路径）";
    }
    const auto& meta = MetaOf("BatchRow");
    const uint64_t before = MariadbEngineTestPeer::BatchInsertExecutions(*engine);
    MariadbEngineTestPeer::SetServerVersionOverride(*engine, 100499UL); // 10.4.99

    TransactionBinding transaction;
    ASSERT_TRUE(engine->Begin(transaction).ok());
    std::vector<BatchRow> rows;
    MakeRows(rows, 16);
    auto pointers = PointerView(rows);
    std::vector<uint64_t> ids(16, 0U);
    const auto inserted
        = engine->InsertMany(meta, TableScope::Static(), pointers.data(), 16U, ids.data(), &transaction);
    ASSERT_TRUE(inserted.ok()) << inserted.diagnostic;
    ASSERT_TRUE(engine->Commit(transaction).ok());

    for (std::size_t i = 0; i < 16; ++i) {
        EXPECT_NE(ids[i], 0U);
    }
    EXPECT_EQ(MariadbEngineTestPeer::BatchInsertExecutions(*engine), before) << "版本不达标的批次不得走快路径";

    const auto counted = engine->Count(meta, TableScope::Static(), ConditionData{});
    ASSERT_TRUE(counted.ok()) << counted.diagnostic;
    EXPECT_EQ(*counted.value, 16U);
}
