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
 * See the License in the specific file packaged with this Software for details.
 */

// 文件职责：实现 MariaDbEngine 的全部能力：一条长连接（mysql_init/
// mysql_real_connect，连接选项统一经 ApplyMariadbConnectionOptions：
// utf8mb4 + 三类超时（秒）+ 可选线路协议压缩）、静态 schema 幂等发布
// （information_schema 齐备性预检 + 逐条幂等 DDL——MariaDB DDL 隐式提交，不能
// 像 SQLite 那样包在 BEGIN IMMEDIATE 内）、动态 PID 表惰性建表、每类 CRUD 的
// 参数化直通链（SQL 文本 → 语句缓存取 MYSQL_STMT → MYSQL_BIND 绑定原值 →
// execute → unbuffered fetch / affected_rows / insert_id）、START TRANSACTION/
// COMMIT/ROLLBACK 事务、自产 .sql 备份与错误码转换。
// 唯一锁：一把连接 mutex 覆盖同一 MYSQL 连接上的全部 API 调用（镜像 SQLite
// 引擎的 *Locked 模式）。单连接事务重叠合同：其他线程的即时 CRUD/第二笔
// Begin 在活动事务期间经条件变量有界等待（同线程立即失败防自锁，超时
// RESOURCE_EXHAUSTED）。InsertMany 覆盖：合格批次走单条参数化多值
// INSERT ... RETURNING 主键真批插（一次 execute），不满足资格自动回退基类
// 默认逐行实现。
//
// 【SQL 构造器来源与方言审计（对照 sqlite_engine.cc，逐条验证）】
// 构造器的后端无关遍历（条件连接/括号、条件绑定顺序、排序项遍历与主键补充、
// INSERT 列清单、投影解析）经 relational_sql_helpers.h 共用；全部 Compare 语法
// （EQ/NOT_EQUAL/LESS/LESS_OR_EQUAL/GREATER/GREATER_OR_EQUAL/PREFIX/IN +
// AND/OR 连接 + 括号分组）的公共遍历顺序与 SQLite 版逐段对照。方言差异
// （已验证并落码）：
//   1) 标识符引用：SQLite "id" 双引号（L109/L98）→ MariaDB `id` 反引号；
//   2) 自增主键：INTEGER PRIMARY KEY AUTOINCREMENT（L964）→
//      BIGINT UNSIGNED AUTO_INCREMENT + 表级 PRIMARY KEY(id)；
//   3) 序列清理：sqlite_sequence DELETE（L1857）→ TRUNCATE（隐式提交，仅无
//      活动事务时执行；自动重置自增计数器）；
//   4) 元数据：sqlite_master（L909/L1877）→ information_schema.tables/
//      statistics WHERE table_schema=DATABASE()；
//   5) 连接参数：PRAGMA journal/busy_timeout（L1228）→ mysql_options 三类
//      超时 + charset（秒配置直接生效，合法范围 [1,3600]，YAML 解析期与
//      引擎建连期双层校验）；
//   6) 生成主键：sqlite3_last_insert_rowid（L1491）→ mysql_stmt_insert_id；
//   7) 影响行数：sqlite3_changes（L1738）→ mysql_stmt_affected_rows；
//   8) 备份：sqlite3_backup_*（L2048）→ 经引擎自身连接的一致性快照自产
//      .sql 文本（mariadb < dump.sql 可回放）；
//   9) LIKE 转义：LIKE ? ESCAPE '\\'（L1821/L1831）语法相同（绑定值不走
//      字面量转义；服务端须未启用 NO_BACKSLASH_ESCAPES）；
//   10) 事务开始：BEGIN IMMEDIATE（L1381/L1925）→ START TRANSACTION；
//   11) 参数缓冲：SQLITE_STATIC/TRANSIENT（L250）→ MYSQL_BIND 的缓冲生命
//       周期由本调用管理（局部数组 + 拥有型槽位，execute 返回前不析构）。
//   12) SQLite 特有的 uint64 高半区展开（IsUnsignedRangeComparison/
//       IsUnsignedOrdering，L356/L2135）在 MariaDB 不需要：BIGINT UNSIGNED
//       原生支持全范围比较与排序，直接渲染普通比较符。
//   13) SELECT 的 LIMIT 1 字面量优化（L1544）不需要：语句缓存 + 服务端预备
//       语句对不同 LIMIT 绑定值无重编译，统一 LIMIT ?。
#include "storage_backends/mariadb/engine/mariadb_engine.h"
#include "storage_backends/mariadb/engine/mariadb_statement_cache.h"
#include "storage/engine/relational/relational_sql_helpers.h"

#include <mysql.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace HcclSim::Storage {
namespace {
    // ---- libmariadb 错误码（本地具名常量：客户端码在 errmsg.h、服务端码在
    // mysqld_error.h——后者不随 libmariadb-dev 发布，全部用具名常量避免头依赖。
    // 数值已对照 vendored connector-c 3.3.5 头文件核实：include/errmsg.h 与
    // include/mysqld_error.h） ----
    constexpr unsigned int kCrConnectionError = 2002;  ///< CR_CONNECTION_ERROR（connect 失败：拒连/不可达）
    constexpr unsigned int kCrConnHostError = 2003;    ///< CR_CONN_HOST_ERROR（仅消息文本，不发给客户端）
    constexpr unsigned int kCrServerGoneError = 2006;  ///< CR_SERVER_GONE_ERROR
    constexpr unsigned int kCrServerLost = 2013;       ///< CR_SERVER_LOST
    constexpr unsigned int kErConnectionKilled = 1927; ///< ER_CONNECTION_KILLED（服务端码：连接被他方 KILL）
    constexpr unsigned int kCrServerLostExtended = 2055; ///< CR_SERVER_LOST_EXTENDED
    constexpr unsigned int kErTableExistsError = 1050;   ///< ER_TABLE_EXISTS_ERROR
    constexpr unsigned int kErDupEntry = 1062;           ///< ER_DUP_ENTRY
    constexpr unsigned int kErDupKeyName = 1061;         ///< ER_DUP_KEYNAME（索引已存在）
    constexpr unsigned int kErLockWaitTimeout = 1205;    ///< ER_LOCK_WAIT_TIMEOUT
    constexpr unsigned int kErLockDeadlock = 1213;       ///< ER_LOCK_DEADLOCK
    /// 连接断开族 errno（客户端视角连接已不可用，可复活重试）：
    /// 2006 CR_SERVER_GONE_ERROR / 2013 CR_SERVER_LOST / 1927 ER_CONNECTION_KILLED /
    /// 2055 CR_SERVER_LOST_EXTENDED。swap 冻结等资源压力下服务端网络超时掐断
    /// 客户端连接、或他方 KILL 活动连接即落此族。
    constexpr bool IsConnectionGoneErrno(unsigned int raw)
    {
        return raw == 2006U || raw == 2013U || raw == 1927U || raw == 2055U;
    }

    DbResult<void> MariadbError(DbCode code, const std::string& diagnostic) { return {code, 0, diagnostic}; }

    DbResult<void> InvalidArgument(const std::string& diagnostic) { return {DbCode::INVALID_ARGUMENT, 0, diagnostic}; }

    DbResult<void> InvalidState(const std::string& diagnostic) { return {DbCode::INVALID_STATE, 0, diagnostic}; }

    template <typename T>
    DbResult<T> FailValue(DbCode code, const std::string& diagnostic)
    {
        return {code, std::nullopt, 0, diagnostic};
    }

    /// libmariadb 错误码到后端无关错误码的映射（映射表进 M1 报告）：
    /// 连接级（2002/2003/2006/2013/1927/2055）→ BACKEND_UNAVAILABLE（2002/2003
    /// 出现在连接/重连失败出口——服务不可达即后端不可用，不是 IO 错误；1927 为
    /// 服务端 ER_CONNECTION_KILLED，与 IsConnectionGoneErrno 复活集保持一致——
    /// 同族 errno 的识别/映射/终态处理不得分叉）；唯一冲突（1062）→ CONFLICT；
    /// 锁等待超时（1205）与死锁（1213）→ RESOURCE_EXHAUSTED（对齐 SQLite BUSY
    /// 语义）；其余 → IO_ERROR（诊断携带 mysql_error 原文）。
    DbCode MapMariadbError(unsigned int error)
    {
        switch (error) {
            case kCrConnectionError:
            case kCrConnHostError:
            case kCrServerGoneError:
            case kCrServerLost:
            case kErConnectionKilled:
            case kCrServerLostExtended:
                return DbCode::BACKEND_UNAVAILABLE;
            case kErDupEntry:
                return DbCode::CONFLICT;
            case kErLockWaitTimeout:
            case kErLockDeadlock:
                return DbCode::RESOURCE_EXHAUSTED;
            default:
                return DbCode::IO_ERROR;
        }
    }

    // ---- 存储失败/恢复诊断日志（现场首错出现即可定位，不必等 Checker 收尾）----
    // 通道：stderr 单行 "[HCCL-VM-STORE] mariadb ..."（与 redis 引擎取证插桩同通道，
    // 由 proxy/device 进程日志统一收编）。只记录失败/恢复，成功路径零输出；分类
    // 计数上限 2048 行，超限打一条抑制提示后静默（断线风暴下防刷爆日志）。字段
    // 仅含：阶段、逻辑库、原始 errno、SQLSTATE、映射码、耗时、恢复结果、客户端
    // 三类超时；不输出凭据、完整连接 URI（endpoint 须先经 SanitizedEndpoint 脱敏）、
    // 行数据或 BLOB。
    namespace storage_diag {
        constexpr uint32_t kLineLimit = 2048U;

        bool AllowLine()
        {
            static std::atomic<uint32_t> reported{0U};
            static std::atomic<bool> suppressed{false};
            const uint32_t seen = reported.fetch_add(1U) + 1U;
            if (seen > kLineLimit) {
                if (!suppressed.exchange(true)) {
                    std::fprintf(stderr, "[HCCL-VM-STORE] mariadb diagnostics suppressed after %u lines\n", kLineLimit);
                }
                return false;
            }
            return true;
        }

        /// 墙钟（HH:MM:SS.mmm，与 proxy/hvm 日志时间线可直接对齐；同 redis 取证口径）
        std::string WallClockText()
        {
            const auto now = std::chrono::system_clock::now();
            const std::time_t seconds = std::chrono::system_clock::to_time_t(now);
            const std::chrono::milliseconds milliseconds
                = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
            std::tm broken{};
            ::localtime_r(&seconds, &broken);
            char buffer[32];
            std::snprintf(
                buffer, sizeof(buffer), "%02d:%02d:%02d.%03d", broken.tm_hour, broken.tm_min, broken.tm_sec,
                static_cast<int>(milliseconds.count()));
            return buffer;
        }

        /// 调用点耗时（毫秒；起点由调用方记录）
        long long ElapsedMsSince(std::chrono::steady_clock::time_point startedAt)
        {
            return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startedAt)
                .count();
        }

        /// SQLSTATE 空值占位（客户端侧错误无 SQLSTATE）
        std::string SqlstateOrDash(const std::string& sqlstate)
        {
            return sqlstate.empty() ? std::string{"-"} : sqlstate;
        }

        /// 诊断截断（超长错误文本截到 160 字符；不携行数据）
        std::string ClipDetail(const std::string& detail)
        {
            if (detail.size() > 160U) {
                return detail.substr(0, 160U) + "...";
            }
            return detail;
        }

        void Report(const std::string& line)
        {
            if (AllowLine()) {
                std::fprintf(stderr, "%s\n", line.c_str());
            }
        }
    } // namespace storage_diag

    /// 在失败点捕获语句/连接上首个非零错误。libmariadb 实测错误来源不对称：
    /// prepare 类服务端拒绝只落在 stmt 上（连接 errno 为 0）；fetch 阶段的网络类
    /// 死亡只落在连接对象上（stmt errno 为 0）——两者取首个非零才是真实原因。
    /// connection 由调用方显式传入当前引擎持有的 MYSQL*（不读 connector 结构
    /// 内部字段 stmt->mysql；复活后 Impl::mysql 即当前连接）。
    struct StmtErrorCapture {
        unsigned int errnoValue{0};
        std::string message;
        std::string sqlstate;
    };

    StmtErrorCapture CaptureStmtOrConnectionError(MYSQL_STMT* stmt, MYSQL* connection, const char* what)
    {
        StmtErrorCapture captured;
        const unsigned int stmtErrno = mysql_stmt_errno(stmt);
        if (stmtErrno != 0U) {
            captured.errnoValue = stmtErrno;
            captured.message = std::string{what} + mysql_stmt_error(stmt);
            captured.sqlstate = mysql_stmt_sqlstate(stmt);
            return captured;
        }
        if (connection != nullptr) {
            const unsigned int connectionErrno = mysql_errno(connection);
            if (connectionErrno != 0U) {
                captured.errnoValue = connectionErrno;
                captured.message = std::string{what} + mysql_error(connection);
                captured.sqlstate = mysql_sqlstate(connection);
                return captured;
            }
        }
        captured.message = std::string{what} + "no diagnostic captured";
        return captured;
    }

    /// prepare 失败错误的补全捕获载体（见 Impl::CompletePrepareErrorLocked）
    struct PrepareErrorCapture {
        unsigned int errnoValue{0};
        std::string message;
        std::string sqlstate;
    };

    /// 标识符只允许字母数字与下划线（与 SQLite 版渲染语义一致；物理表名
    /// <table>_P_<pid> 与派生索引名天然满足）
    DbResult<void> ValidateIdentifier(std::string_view name)
    {
        if (name.empty()) {
            return {DbCode::INVALID_ARGUMENT, 0, "identifier is empty"};
        }
        for (const char character : name) {
            const bool safe = (character >= 'a' && character <= 'z') || (character >= 'A' && character <= 'Z')
                              || (character >= '0' && character <= '9') || character == '_';
            if (!safe) {
                return {DbCode::INVALID_ARGUMENT, 0, "identifier is unsafe: " + std::string{name}};
            }
        }
        return {DbCode::OK, 0, {}};
    }

    /// 校验通过后把带反引号的标识符直接追加到最终 SQL 文本（方言差异 #1：
    /// SQLite 用双引号；无第二份拥有字符串）
    DbResult<void> AppendQuotedIdentifier(std::string& sql, std::string_view name)
    {
        auto validated = ValidateIdentifier(name);
        if (!validated.ok()) {
            return validated;
        }
        sql += '`';
        sql += name;
        sql += '`';
        return {DbCode::OK, 0, {}};
    }

    /// 反引号引用的拥有型标识符（DDL 渲染等冷路径）
    DbResult<std::string> QuoteIdentifier(std::string_view name)
    {
        auto validated = ValidateIdentifier(name);
        if (!validated.ok()) {
            return FailValue<std::string>(validated.code, validated.diagnostic);
        }
        return {DbCode::OK, std::optional<std::string>{"`" + std::string{name} + "`"}, 0, {}};
    }

    /// MariaDB 列类型（本文件唯一的物理类型转换点；方言差异 #2 + 类型映射表）
    const char* ColumnTypeSql(LogicalFieldType type)
    {
        switch (type) {
            case LogicalFieldType::SIGNED:
                return "BIGINT";
            case LogicalFieldType::UNSIGNED:
                return "BIGINT UNSIGNED";
            case LogicalFieldType::REAL:
                return "DOUBLE";
            case LogicalFieldType::BOOLEAN:
                return "BOOLEAN";
            case LogicalFieldType::STRING:
                return "MEDIUMTEXT";
            case LogicalFieldType::BYTES:
                return "LONGBLOB";
        }
        return "LONGBLOB";
    }

    bool IsIntegerColumn(LogicalFieldType type)
    {
        return type == LogicalFieldType::SIGNED || type == LogicalFieldType::UNSIGNED
               || type == LogicalFieldType::BOOLEAN;
    }

    /// TEXT/BLOB 列建索引必须带前缀长度（方言差异：SQLite 无此限制）。191×4
    /// 字节 = 764 字节，兼容任意 InnoDB 行格式的索引字节上限。
    constexpr const char* kTextIndexPrefix = "(191)";

    /// 目录字段名查找（入参可含 "表::成员" 前缀，按最后 "::" 之后的名字匹配本表字段；
    /// 与 SQLite 版 FindField 同构）
    const Ddl::FieldDesc* FindField(const Ddl::CatalogTableMeta& meta, std::string_view name)
    {
        const auto sep = name.rfind("::");
        const std::string_view member = sep == std::string_view::npos ? name : name.substr(sep + 2);
        for (std::size_t i = 0; i < meta.fieldCount; ++i) {
            if (std::string_view{meta.fields[i].name} == member) {
                return &meta.fields[i];
            }
        }
        return nullptr;
    }

    // ---- 按声明宽度读取原始成员值（EmitFieldRef 的 data/bytes 解释；与 SQLite 版同构）----

    int64_t ReadSignedValue(const void* data, size_t bytes)
    {
        switch (bytes) {
            case 1: {
                int8_t value = 0;
                std::memcpy(&value, data, sizeof(value));
                return value;
            }
            case 2: {
                int16_t value = 0;
                std::memcpy(&value, data, sizeof(value));
                return value;
            }
            case 4: {
                int32_t value = 0;
                std::memcpy(&value, data, sizeof(value));
                return value;
            }
            default: {
                int64_t value = 0;
                std::memcpy(&value, data, sizeof(value));
                return value;
            }
        }
    }

    uint64_t ReadUnsignedValue(const void* data, size_t bytes)
    {
        switch (bytes) {
            case 1: {
                uint8_t value = 0;
                std::memcpy(&value, data, sizeof(value));
                return value;
            }
            case 2: {
                uint16_t value = 0;
                std::memcpy(&value, data, sizeof(value));
                return value;
            }
            case 4: {
                uint32_t value = 0;
                std::memcpy(&value, data, sizeof(value));
                return value;
            }
            default: {
                uint64_t value = 0;
                std::memcpy(&value, data, sizeof(value));
                return value;
            }
        }
    }

    double ReadRealValue(const void* data, size_t bytes)
    {
        if (bytes == sizeof(float)) {
            float value = 0.0F;
            std::memcpy(&value, data, sizeof(value));
            return value;
        }
        double value = 0.0;
        std::memcpy(&value, data, sizeof(value));
        return value;
    }

    /// 自增主键"值 0 视为未指定"判定（整数/布尔按声明宽度读取；与 SQLite 版同构）
    bool IsZeroIntegerValue(const Ddl::FieldDesc& field, const void* data, size_t bytes)
    {
        if (field.type == LogicalFieldType::BOOLEAN) {
            bool value = false;
            if (bytes >= sizeof(bool)) {
                std::memcpy(&value, data, sizeof(bool));
            }
            return !value;
        }
        if (field.type == LogicalFieldType::UNSIGNED) {
            return ReadUnsignedValue(data, bytes) == 0U;
        }
        if (field.type == LogicalFieldType::SIGNED) {
            return ReadSignedValue(data, bytes) == 0;
        }
        return false;
    }

    /// 空文本/空 BLOB 的绑定占位（绑零长度值而不是 NULL；与 SQLite 版语义一致）
    constexpr char kEmptyByte = '\0';

    /// 比较符固定文本（constexpr 数组按连续枚举下标解释；IN/PREFIX 由专属形态展开）
    constexpr std::array<const char*, static_cast<std::size_t>(Compare::COUNT)> kCompareText
        = {"=", "<>", "<", "<=", ">", ">=", nullptr, nullptr};

    // ---- 参数绑定（MYSQL_BIND 缓冲生命周期由本调用管理：方言差异 #11 ----)

    /// 单个参数槽的稳定存储（MYSQL_BIND 只保留指针；缓冲必须存活到 execute
    /// 返回。标量落定宽槽；Derived 字段临时值与 PREFIX 转义模式复制进 owned；
    /// 其余 string/blob 直接借用调用方存储——源对象存活覆盖 execute）
    struct ParamSlot {
        int64_t signedValue{0};
        uint64_t unsignedValue{0};
        double realValue{0.0};
        int8_t tinyValue{0};
        unsigned long length{0};
        my_bool isNull{0};
        std::vector<uint8_t> owned;
    };

    /// STRING 列绑定值的 utf8 有效性检查（方言差异 #15）。MariaDB utf8mb4 严格
    /// 模式拒绝无效 utf8 字节（errno 1366 "Incorrect string value"）；业务层
    /// 存在未初始化的定长 char 数组尾字节（如 runtime::Server.version 从未
    /// 赋值即带栈上垃圾，SQLite 原样入库、消费者按 C 字符串首个 NUL 截断读取）。
    /// 引擎在此镜像该语义：无效字节替换为空格——首个 NUL 前的逻辑前缀不受
    /// 影响，消费者可见行为与 SQLite 一致。BYTES/BLOB 列不清洗（任意字节合法）。
    bool Utf8SequenceValid(const uint8_t* data, std::size_t size, std::size_t& next)
    {
        const uint8_t lead = data[0];
        if (lead < 0x80U) {
            next = 1;
            return true;
        }
        std::size_t length = 0;
        uint8_t lowest = 0x80U;
        uint8_t highest = 0xBFU;
        if (lead >= 0xC2U && lead <= 0xDFU) {
            length = 2;
        } else if (lead == 0xE0U) {
            length = 3;
            lowest = 0xA0U;
        } else if ((lead >= 0xE1U && lead <= 0xECU) || lead == 0xEEU || lead == 0xEFU) {
            length = 3;
        } else if (lead == 0xEDU) {
            length = 3;
            highest = 0x9FU;
        } else if (lead == 0xF0U) {
            length = 4;
            lowest = 0x90U;
        } else if (lead >= 0xF1U && lead <= 0xF3U) {
            length = 4;
        } else if (lead == 0xF4U) {
            length = 4;
            highest = 0x8FU;
        } else {
            return false;
        }
        if (size < length) {
            return false;
        }
        if (data[1] < lowest || data[1] > highest) {
            return false;
        }
        for (std::size_t k = 2; k < length; ++k) {
            if (data[k] < 0x80U || data[k] > 0xBFU) {
                return false;
            }
        }
        next = length;
        return true;
    }

    bool Utf8Valid(const uint8_t* data, std::size_t size)
    {
        std::size_t i = 0;
        while (i < size) {
            std::size_t next = 0;
            if (!Utf8SequenceValid(data + i, size - i, next)) {
                return false;
            }
            i += next;
        }
        return true;
    }

    /// 无效时产出清洗副本（无效字节→空格）；有效返回 false 不动 buffer
    bool Utf8SanitizeInto(const uint8_t* data, std::size_t size, std::vector<uint8_t>& out)
    {
        if (Utf8Valid(data, size)) {
            return false;
        }
        out.assign(data, data + size);
        std::size_t i = 0;
        while (i < out.size()) {
            std::size_t next = 0;
            if (!Utf8SequenceValid(out.data() + i, out.size() - i, next)) {
                out[i] = ' ';
                i += 1;
            } else {
                i += next;
            }
        }
        return true;
    }

    /// 一条参数化语句的全部参数绑定（数组下标 0 基，对应 SQL 的第 N 个 ?）
    class BoundParams {
    public:
        explicit BoundParams(std::size_t count) : slots_(count)
        {
            binds_.resize(count);
            for (std::size_t i = 0; i < count; ++i) {
                binds_[i].length = &slots_[i].length;
                binds_[i].is_null = &slots_[i].isNull;
            }
        }

        /// 绑定一个必须由参数槽持有副本的字符串（PREFIX 转义产物等短生命周期
        /// 值）：字节复制进槽位 owned 后绑定——BindScalar 对合法 utf8 字符串是
        /// 借用（调用方存储存活覆盖 execute 的合同），不能用于本调用内析构的
        /// 临时值
        DbResult<void> BindOwnedString(std::size_t index, std::string_view text)
        {
            if (index >= slots_.size()) {
                return InvalidArgument("statement parameter index is out of range");
            }
            MYSQL_BIND& bind = binds_[index];
            ParamSlot& slot = slots_[index];
            slot.isNull = 0;
            const uint8_t* const textBegin = reinterpret_cast<const uint8_t*>(text.data());
            slot.owned.assign(textBegin, textBegin + text.size());
            bind.buffer_type = MYSQL_TYPE_STRING;
            bind.buffer
                = slot.owned.empty() ? const_cast<char*>(&kEmptyByte) : reinterpret_cast<char*>(slot.owned.data());
            bind.buffer_length = static_cast<unsigned long>(slot.owned.size());
            slot.length = static_cast<unsigned long>(slot.owned.size());
            bind.is_unsigned = 0;
            return {DbCode::OK, 0, {}};
        }

        /// 绑定一个 DbScalar（WHERE/SET/LIMIT 值）。string/blob 直接引用唯一
        /// ConditionData/UpdateData 中的存储（借用不复制，源对象存活覆盖 execute）
        DbResult<void> BindScalar(std::size_t index, const DbScalar& value)
        {
            if (index >= slots_.size()) {
                return InvalidArgument("statement parameter index is out of range");
            }
            MYSQL_BIND& bind = binds_[index];
            ParamSlot& slot = slots_[index];
            slot.isNull = 0;
            if (const auto* number = std::get_if<int64_t>(&value); number != nullptr) {
                slot.signedValue = *number;
                bind.buffer_type = MYSQL_TYPE_LONGLONG;
                bind.buffer = &slot.signedValue;
                bind.buffer_length = sizeof(slot.signedValue);
                bind.is_unsigned = 0;
            } else if (const auto* number = std::get_if<uint64_t>(&value); number != nullptr) {
                slot.unsignedValue = *number;
                bind.buffer_type = MYSQL_TYPE_LONGLONG;
                bind.buffer = &slot.unsignedValue;
                bind.buffer_length = sizeof(slot.unsignedValue);
                bind.is_unsigned = 1;
            } else if (const auto* number = std::get_if<double>(&value); number != nullptr) {
                slot.realValue = *number;
                bind.buffer_type = MYSQL_TYPE_DOUBLE;
                bind.buffer = &slot.realValue;
                bind.buffer_length = sizeof(slot.realValue);
                bind.is_unsigned = 0;
            } else if (const auto* flag = std::get_if<bool>(&value); flag != nullptr) {
                slot.tinyValue = *flag ? 1 : 0;
                bind.buffer_type = MYSQL_TYPE_TINY;
                bind.buffer = &slot.tinyValue;
                bind.buffer_length = sizeof(slot.tinyValue);
                bind.is_unsigned = 0;
            } else if (const auto* text = std::get_if<std::string>(&value); text != nullptr) {
                // 空文本绑零长度值而不是 NULL（data() 对空 string 可能返回指针
                // 之外的特殊值，统一落到 kEmptyByte 占位）；无效 utf8 清洗进槽位
                // （方言差异 #15，见 Utf8SanitizeInto）
                const uint8_t* raw = reinterpret_cast<const uint8_t*>(text->data());
                const std::size_t size = text->size();
                if (size != 0 && Utf8SanitizeInto(raw, size, slot.owned)) {
                    bind.buffer_type = MYSQL_TYPE_STRING;
                    bind.buffer = slot.owned.data();
                    bind.buffer_length = static_cast<unsigned long>(slot.owned.size());
                    slot.length = static_cast<unsigned long>(slot.owned.size());
                    bind.is_unsigned = 0;
                } else {
                    bind.buffer_type = MYSQL_TYPE_STRING;
                    bind.buffer = text->empty() ? const_cast<char*>(&kEmptyByte) : const_cast<char*>(text->data());
                    bind.buffer_length = static_cast<unsigned long>(text->size());
                    slot.length = static_cast<unsigned long>(text->size());
                    bind.is_unsigned = 0;
                }
            } else if (const auto* bytes = std::get_if<std::vector<uint8_t>>(&value); bytes != nullptr) {
                bind.buffer_type = MYSQL_TYPE_BLOB;
                bind.buffer = bytes->empty() ? const_cast<char*>(&kEmptyByte) :
                                               const_cast<char*>(reinterpret_cast<const char*>(bytes->data()));
                bind.buffer_length = static_cast<unsigned long>(bytes->size());
                slot.length = static_cast<unsigned long>(bytes->size());
                bind.is_unsigned = 0;
            }
            return {DbCode::OK, 0, {}};
        }

        /// 绑定 EmitFieldRef 发射的原始成员值（镜像 SQLite 版 BindEmittedField）。
        /// transient 仅对 Derived 字段为 true（值来自目录 EmitDerived 的栈上临时
        /// DbScalar，emit 返回后销毁）→ 复制进槽位 owned；其余值直接借用原 T 成员
        /// （record 存活覆盖 execute）。
        DbResult<void>
        BindEmitted(std::size_t index, const Ddl::FieldDesc& field, const void* data, std::size_t bytes, bool transient)
        {
            if (index >= slots_.size()) {
                return InvalidArgument("statement parameter index is out of range");
            }
            if (data == nullptr && bytes != 0) {
                return MariadbError(DbCode::INVALID_ARGUMENT, "borrowed insert value has null data");
            }
            MYSQL_BIND& bind = binds_[index];
            ParamSlot& slot = slots_[index];
            slot.isNull = 0;
            const auto bindBytes = [&](enum_field_types type, const void* buffer, std::size_t length) {
                bind.buffer_type = type;
                bind.buffer = const_cast<void*>(buffer);
                bind.buffer_length = static_cast<unsigned long>(length);
                slot.length = static_cast<unsigned long>(length);
                bind.is_unsigned = 0;
            };
            switch (field.type) {
                case LogicalFieldType::SIGNED:
                    slot.signedValue = ReadSignedValue(data, bytes);
                    bind.buffer_type = MYSQL_TYPE_LONGLONG;
                    bind.buffer = &slot.signedValue;
                    bind.buffer_length = sizeof(slot.signedValue);
                    bind.is_unsigned = 0;
                    break;
                case LogicalFieldType::UNSIGNED:
                    // BIGINT UNSIGNED 原生承载全 uint64 范围（方言差异 #12：无需
                    // SQLite 的位型直存技巧）
                    slot.unsignedValue = ReadUnsignedValue(data, bytes);
                    bind.buffer_type = MYSQL_TYPE_LONGLONG;
                    bind.buffer = &slot.unsignedValue;
                    bind.buffer_length = sizeof(slot.unsignedValue);
                    bind.is_unsigned = 1;
                    break;
                case LogicalFieldType::BOOLEAN:
                    slot.tinyValue = IsZeroIntegerValue(field, data, bytes) ? 0 : 1;
                    bind.buffer_type = MYSQL_TYPE_TINY;
                    bind.buffer = &slot.tinyValue;
                    bind.buffer_length = sizeof(slot.tinyValue);
                    bind.is_unsigned = 0;
                    break;
                case LogicalFieldType::REAL:
                    slot.realValue = ReadRealValue(data, bytes);
                    bind.buffer_type = MYSQL_TYPE_DOUBLE;
                    bind.buffer = &slot.realValue;
                    bind.buffer_length = sizeof(slot.realValue);
                    bind.is_unsigned = 0;
                    break;
                case LogicalFieldType::STRING: {
                    const auto* rawBytes = static_cast<const uint8_t*>(data);
                    const void* buffer = bytes == 0 ? static_cast<const void*>(&kEmptyByte) : data;
                    if (bytes != 0 && Utf8SanitizeInto(rawBytes, bytes, slot.owned)) {
                        // 无效 utf8（未初始化定长数组尾字节等）→ 清洗副本绑定
                        // （方言差异 #15）；有效输入零改动直通
                        bindBytes(MYSQL_TYPE_STRING, slot.owned.data(), slot.owned.size());
                    } else if (transient) {
                        slot.owned.assign(
                            static_cast<const uint8_t*>(buffer), static_cast<const uint8_t*>(buffer) + bytes);
                        bindBytes(MYSQL_TYPE_STRING, slot.owned.data(), bytes);
                    } else {
                        bindBytes(MYSQL_TYPE_STRING, buffer, bytes);
                    }
                    break;
                }
                case LogicalFieldType::BYTES: {
                    const void* buffer = bytes == 0 ? static_cast<const void*>(&kEmptyByte) : data;
                    if (transient) {
                        slot.owned.assign(
                            static_cast<const uint8_t*>(buffer), static_cast<const uint8_t*>(buffer) + bytes);
                        bindBytes(MYSQL_TYPE_BLOB, slot.owned.data(), bytes);
                    } else {
                        bindBytes(MYSQL_TYPE_BLOB, buffer, bytes);
                    }
                    break;
                }
            }
            return {DbCode::OK, 0, {}};
        }

        /// 绑定 NULL（自增主键值 0 视为未指定时由后端生成）
        void BindNull(std::size_t index)
        {
            MYSQL_BIND& bind = binds_[index];
            slots_[index].isNull = 1;
            bind.buffer_type = MYSQL_TYPE_NULL;
            bind.buffer = nullptr;
            bind.buffer_length = 0;
            bind.is_unsigned = 0;
        }

        MYSQL_BIND* Data() const { return const_cast<MYSQL_BIND*>(binds_.data()); }
        std::size_t Size() const { return binds_.size(); }

    private:
        std::vector<MYSQL_BIND> binds_{};
        std::vector<ParamSlot> slots_{};
    };

    // ---- WHERE 生成（第一遍：扫描原条件生成参数化 SQL，同步完成校验；与 SQLite
    // 版 AppendClauseText/AppendWhereText 逐段同构，差异仅标识符引用与无符号
    // 高半区展开的删除）----

    DbResult<void> AppendClauseText(
        const ConditionClause& clause, const Ddl::CatalogTableMeta& meta, std::size_t& placeholder, std::string& output)
    {
        if (clause.fieldId >= meta.fieldCount) {
            return InvalidArgument("logical field is not mapped to a column: " + std::to_string(clause.fieldId));
        }
        const Ddl::FieldDesc& field = meta.fields[clause.fieldId];
        // 先行校验（与 SQLite 版一致的失败时序：空 IN 等短路分支之前即拒绝不安全列名）
        const auto validated = ValidateIdentifier(field.name);
        if (!validated.ok()) {
            return validated;
        }
        const auto* single = std::get_if<DbScalar>(&clause.value);
        const auto* inValues = std::get_if<std::vector<DbScalar>>(&clause.value);
        const bool emptyIn = clause.compare == Compare::IN && inValues != nullptr && inValues->empty();
        if (emptyIn) {
            // 空 IN 渲染为常量假表达式：匹配 0 行，不产生占位符与绑定值。
            output += "1 = 0";
            return {DbCode::OK, 0, {}};
        }
        if (clause.compare == Compare::IN) {
            if (inValues == nullptr) {
                return InvalidArgument("predicate expects a single comparison value");
            }
            for (const auto& value : *inValues) {
                const auto checked = Relational::CheckColumnValue(field, value);
                if (!checked.ok()) {
                    return checked;
                }
            }
            const auto appended = AppendQuotedIdentifier(output, field.name);
            if (!appended.ok()) {
                return appended;
            }
            output += " IN (";
            for (std::size_t item = 0; item < inValues->size(); ++item) {
                output += item == 0 ? "?" : ", ?";
                ++placeholder;
            }
            output += ")";
            return {DbCode::OK, 0, {}};
        }
        if (inValues != nullptr) {
            return InvalidArgument("predicate expects a single comparison value");
        }
        if (single == nullptr) {
            return InvalidArgument("predicate has no comparison values");
        }
        if (clause.compare == Compare::PREFIX) {
            if (field.type != LogicalFieldType::STRING || !std::holds_alternative<std::string>(*single)) {
                return InvalidArgument("prefix comparison requires a text column and a string value");
            }
            const auto appended = AppendQuotedIdentifier(output, field.name);
            if (!appended.ok()) {
                return appended;
            }
            output += " LIKE ? ESCAPE '\\\\'"; // 方言差异 #9：语法与 SQLite 相同
            ++placeholder;
            return {DbCode::OK, 0, {}};
        }
        const auto checked = Relational::CheckColumnValue(field, *single);
        if (!checked.ok()) {
            return checked;
        }
        const char* compareText = kCompareText[static_cast<std::size_t>(clause.compare)];
        if (compareText == nullptr) {
            return InvalidArgument("compare kind is not supported");
        }
        const auto appended = AppendQuotedIdentifier(output, field.name);
        if (!appended.ok()) {
            return appended;
        }
        output += ' ';
        output += compareText;
        output += " ?";
        ++placeholder;
        return {DbCode::OK, 0, {}};
    }

    /// 条件文本（第一遍）：公共遍历（连接词/括号，见 relational_sql_helpers.h）
    /// 调用 MariaDB 单子句方言函数；与占位符/绑定值顺序严格一致（差异仅标识符
    /// 引用与无 SQLite 的无符号高半区展开）。
    DbResult<void> AppendWhereText(
        const ConditionData& where, const Ddl::CatalogTableMeta& meta, std::size_t& placeholder, std::string& output)
    {
        return Relational::AppendWhereText(
            where, output, [&meta, &placeholder, &output](const ConditionClause& clause) {
                return AppendClauseText(clause, meta, placeholder, output);
            });
    }

    // ---- WHERE 绑定（第二遍：按相同顺序绑定原值）----

    DbResult<void> BindClauseValues(const ConditionClause& clause, BoundParams& params, std::size_t& nextIndex)
    {
        const auto* single = std::get_if<DbScalar>(&clause.value);
        const auto* inValues = std::get_if<std::vector<DbScalar>>(&clause.value);
        const bool emptyIn = clause.compare == Compare::IN && inValues != nullptr && inValues->empty();
        if (emptyIn) {
            return {DbCode::OK, 0, {}};
        }
        if (clause.compare == Compare::IN) {
            for (const auto& value : *inValues) {
                const auto bound = params.BindScalar(nextIndex, value);
                if (!bound.ok()) {
                    return bound;
                }
                ++nextIndex;
            }
            return {DbCode::OK, 0, {}};
        }
        if (clause.compare == Compare::PREFIX) {
            // 转义产生的临时字符串存活仅到本分支：绑定值复制进参数槽 owned
            // （方言差异 #11：缓冲生命周期由本调用管理），execute 返回前不析构。
            // 【既有缺陷修复】BindScalar 对合法 utf8 字符串是借用而非复制——
            // 本注释声明的"复制进 owned"此前仅无效 utf8 清洗路径成立，PREFIX
            // 模式实际借用局部 pattern，execute 读悬空指针（ASan 实证
            // heap-use-after-free；业务 IpcNotify name_or_key 前缀查询在用）。
            // 改为 BindOwnedString 显式持有副本，不改变 BindScalar 既有借用合同。
            const std::string pattern = Relational::EscapePrefixPattern(std::get<std::string>(*single));
            const auto bound = params.BindOwnedString(nextIndex, pattern);
            if (!bound.ok()) {
                return bound;
            }
            ++nextIndex;
            return {DbCode::OK, 0, {}};
        }
        const auto bound = params.BindScalar(nextIndex, *single);
        if (!bound.ok()) {
            return bound;
        }
        ++nextIndex;
        return {DbCode::OK, 0, {}};
    }

    /// 条件绑定（第二遍）：公共遍历按相同顺序调用 MariaDB 原生绑定（临时值
    /// 生命周期——PREFIX 转义模式的 BindOwnedString 拥有化——留在回调内）。
    DbResult<void> BindWhereValues(const ConditionData& where, BoundParams& params, std::size_t& nextIndex)
    {
        return Relational::BindWhereValues(where, [&params, &nextIndex](const ConditionClause& clause) {
            return BindClauseValues(clause, params, nextIndex);
        });
    }

    /// MariaDB endpoint URI 解析结果（mariadb://user:password@host:port）
    struct MariadbEndpoint {
        std::string host;
        uint32_t port{3306};
        std::string user;
        std::string password;
    };

    /// uri 中不允许出现的字符（路径/查询/锚点分隔符与空白不属于 host 形态；
    /// 与 redis_engine_config 的 host 校验同构）
    bool HostCharacterValid(char value)
    {
        if (value == '/' || value == '?' || value == '#' || value == '@') {
            return false;
        }
        if (value == ' ' || value == '\t' || value == '\r' || value == '\n' || value == '\0') {
            return false;
        }
        return true;
    }

    /// 解析 mariadb://user:password@host:port（本阶段只支持 hostname/IPv4；
    /// 密码经最后一个 '@' 切分——密码可含 '@'，用户名与密码经第一个 ':' 切分）
    DbResult<MariadbEndpoint> ParseMariadbUri(const std::string& uri)
    {
        constexpr char kScheme[] = "mariadb://";
        const auto malformed = [&uri](const char* detail) {
            return FailValue<MariadbEndpoint>(DbCode::INVALID_ARGUMENT, std::string{detail} + ": " + uri);
        };
        if (uri.find(kScheme) != 0) {
            return malformed("mariadb endpoint uri must be mariadb://user:password@host:port");
        }
        const std::string authority = uri.substr(sizeof(kScheme) - 1);
        const auto at = authority.rfind('@');
        if (at == std::string::npos || at == 0 || at + 1U == authority.size()) {
            return malformed("mariadb endpoint uri must be mariadb://user:password@host:port");
        }
        const std::string userinfo = authority.substr(0, at);
        const std::string hostPort = authority.substr(at + 1U);
        const auto colon = userinfo.find(':');
        if (colon == std::string::npos || colon == 0) {
            return malformed("mariadb endpoint uri requires user:password");
        }
        MariadbEndpoint endpoint;
        endpoint.user = userinfo.substr(0, colon);
        endpoint.password = userinfo.substr(colon + 1);
        // host 与端口恰好用一个冒号分隔（IPv6 明确拒绝，与 Redis 配置合同一致）
        const auto separator = hostPort.find(':');
        if (separator == std::string::npos || separator == 0 || separator + 1U == hostPort.size()) {
            return malformed("mariadb endpoint uri must be mariadb://user:password@host:port");
        }
        if (hostPort.find(':', separator + 1U) != std::string::npos) {
            return malformed("mariadb endpoint host must be a hostname or IPv4 address (IPv6 is not supported)");
        }
        endpoint.host = hostPort.substr(0, separator);
        const std::string port = hostPort.substr(separator + 1U);
        for (const char character : endpoint.host) {
            if (!HostCharacterValid(character)) {
                return malformed("mariadb endpoint host contains invalid character");
            }
        }
        uint64_t portValue = 0;
        for (const char digit : port) {
            if (digit < '0' || digit > '9') {
                return malformed("mariadb endpoint port must be decimal digits");
            }
            portValue = portValue * 10U + static_cast<uint64_t>(digit - '0');
            if (portValue > 65535U) {
                return malformed("mariadb endpoint port is out of range");
            }
        }
        if (portValue == 0) {
            return malformed("mariadb endpoint port must not be zero");
        }
        endpoint.port = static_cast<uint32_t>(portValue);
        return {DbCode::OK, std::optional<MariadbEndpoint>{std::move(endpoint)}, 0, {}};
    }

    /// 诊断用脱敏 endpoint（密码替换 ***，避免日志/错误信息泄漏凭据）
    std::string SanitizedEndpoint(const MariadbEndpoint& endpoint)
    {
        return "mariadb://" + endpoint.user + ":***@" + endpoint.host + ":" + std::to_string(endpoint.port);
    }

    /// profile 程序化构造的越界防护（YAML 解析期已按 [kMariadbTimeoutMinS,
    /// kMariadbTimeoutMaxS] 校验；直接构造 profile 的调用方在 Open/Wipe 建连前
    /// 拒绝，不把非法秒值传给 libmariadb——镜像 Redis 侧 read_max_attempts 的
    /// 双层校验语义）
    DbResult<void> ValidateMariadbProfileTimeouts(const MariadbConnectionProfile& profile)
    {
        const char* names[3]
            = {"connection_establishment_timeout_s", "packet_read_timeout_s", "packet_write_timeout_s"};
        const uint32_t values[3]
            = {profile.connectionEstablishmentTimeoutS, profile.packetReadTimeoutS, profile.packetWriteTimeoutS};
        for (std::size_t i = 0; i < 3; ++i) {
            if (values[i] < kMariadbTimeoutMinS || values[i] > kMariadbTimeoutMaxS) {
                return DbResult<void>{
                    DbCode::INVALID_ARGUMENT, 0,
                    std::string{"mariadb profile "} + names[i] + " is out of range ["
                        + std::to_string(kMariadbTimeoutMinS) + ", " + std::to_string(kMariadbTimeoutMaxS)
                        + "] seconds"};
            }
        }
        return {DbCode::OK, 0, {}};
    }

    /// 统一连接选项（Open / ReconnectLocked / Wipe 三条 mysql_real_connect 路径
    /// 共用的最小共享函数）：charset=utf8mb4、三类超时（秒，直接取 profile 值，
    /// 无毫秒换算与取整）、按配置决定 MYSQL_OPT_COMPRESS。mysql_options 失败不
    /// 静默忽略——返回明确诊断，由调用方带原因失败（这些选项对合法 profile 不会
    /// 失败，失败即程序性错误）。不启用 Connector/C 自动重连
    /// （MYSQL_OPT_RECONNECT）：断线复活由引擎按有界语义自行管理；本函数也不
    /// 修改事务、autocommit 与语句缓存设置（方言差异 #5）
    DbResult<void> ApplyMariadbConnectionOptions(MYSQL* connection, const MariadbConnectionProfile& profile)
    {
        if (mysql_options(connection, MYSQL_SET_CHARSET_NAME, "utf8mb4") != 0) {
            return DbResult<void>{
                DbCode::INVALID_ARGUMENT, 0, "mariadb connection option MYSQL_SET_CHARSET_NAME was rejected"};
        }
        const unsigned int connectTimeout = profile.connectionEstablishmentTimeoutS;
        const unsigned int readTimeout = profile.packetReadTimeoutS;
        const unsigned int writeTimeout = profile.packetWriteTimeoutS;
        if (mysql_options(connection, MYSQL_OPT_CONNECT_TIMEOUT, &connectTimeout) != 0) {
            return DbResult<void>{
                DbCode::INVALID_ARGUMENT, 0, "mariadb connection option MYSQL_OPT_CONNECT_TIMEOUT was rejected"};
        }
        if (mysql_options(connection, MYSQL_OPT_READ_TIMEOUT, &readTimeout) != 0) {
            return DbResult<void>{
                DbCode::INVALID_ARGUMENT, 0, "mariadb connection option MYSQL_OPT_READ_TIMEOUT was rejected"};
        }
        if (mysql_options(connection, MYSQL_OPT_WRITE_TIMEOUT, &writeTimeout) != 0) {
            return DbResult<void>{
                DbCode::INVALID_ARGUMENT, 0, "mariadb connection option MYSQL_OPT_WRITE_TIMEOUT was rejected"};
        }
        // 线路协议压缩减少传输字节，但增加客户端和服务端 CPU；是否启用由部署
        // 配置决定（profile.wireProtocolCompression，缺省 true）
        if (profile.wireProtocolCompression && mysql_options(connection, MYSQL_OPT_COMPRESS, nullptr) != 0) {
            return DbResult<void>{
                DbCode::INVALID_ARGUMENT, 0, "mariadb connection option MYSQL_OPT_COMPRESS was rejected"};
        }
        return {DbCode::OK, 0, {}};
    }

    /// 物理 database 名派生：hcclvm_<namespace>_<databaseId>（非法字符替换 '_'）。
    /// 任务合同"两逻辑库=两个 database"要求名字同时携带 Open 的 dataNamespace 与
    /// 目录 databaseId（每进程两实例共享同一 namespace，仅靠 namespace 无法区分
    /// 两个逻辑库）；派生确定性保证全部 rank/进程对同一 namespace 得到同一名字。
    std::string DeriveDatabaseName(std::string_view dataNamespace, std::string_view databaseId)
    {
        const auto sanitize = [](std::string_view source) {
            std::string out;
            out.reserve(source.size());
            for (const char character : source) {
                const bool safe = (character >= 'a' && character <= 'z') || (character >= 'A' && character <= 'Z')
                                  || (character >= '0' && character <= '9') || character == '_';
                out += safe ? character : '_';
            }
            return out;
        };
        const std::string space = dataNamespace.empty() ? std::string{"default"} : sanitize(dataNamespace);
        return "hcclvm_" + space + "_" + sanitize(databaseId);
    }

    /// 进程级 mysql_library_init 单次初始化（多线程进程在首个引擎 Open 前完成；
    /// 不调用 mysql_library_end——其它引擎实例可能仍在使用，进程退出统一回收）
    void EnsureMariadbLibraryInit()
    {
        static std::once_flag once;
        std::call_once(once, [] {
            mysql_library_init(0, nullptr, nullptr);
        });
    }

    // ---- DDL 渲染（直接遍历 CatalogTableMeta；外键目标只在启动建表时在本库目录解析）----

    DbResult<std::string> RenderCreateTable(
        const Ddl::DatabaseDescriptor& catalog, const Ddl::CatalogTableMeta& meta, const std::string& physicalTableName)
    {
        auto table = QuoteIdentifier(physicalTableName);
        if (!table.ok()) {
            return FailValue<std::string>(table.code, table.diagnostic);
        }
        std::string sql = "CREATE TABLE IF NOT EXISTS " + *table.value + " (";
        bool first = true;
        for (std::size_t i = 0; i < meta.fieldCount; ++i) {
            const Ddl::FieldDesc& field = meta.fields[i];
            auto name = QuoteIdentifier(field.name);
            if (!name.ok()) {
                return FailValue<std::string>(name.code, name.diagnostic);
            }
            sql += first ? "" : ", ";
            first = false;
            sql += *name.value + " " + ColumnTypeSql(field.type);
            if (field.primaryKey && Ddl::HasProp(field.props, Ddl::Prop::AutoIncrement)) {
                // 方言差异 #2：自增主键列 + 表级 PRIMARY KEY（SQLite 是
                // "INTEGER PRIMARY KEY AUTOINCREMENT" 列内声明）
                sql += " NOT NULL AUTO_INCREMENT";
            } else {
                sql += " NOT NULL";
            }
        }
        // 表级主键（目录合同：单列主键；全部目录表为自增 id）
        for (std::size_t i = 0; i < meta.fieldCount; ++i) {
            if (meta.fields[i].primaryKey) {
                auto name = QuoteIdentifier(meta.fields[i].name);
                if (!name.ok()) {
                    return FailValue<std::string>(name.code, name.diagnostic);
                }
                sql += ", PRIMARY KEY (" + *name.value + ")";
                break;
            }
        }
        for (std::size_t i = 0; i < meta.foreignKeyCount; ++i) {
            const Ddl::ForeignKeyDesc& foreignKey = meta.foreignKeys[i];
            const auto* localField = FindField(meta, foreignKey.localColumn);
            const Ddl::CatalogTableMeta* targetTable = nullptr;
            const Ddl::FieldDesc* targetField = nullptr;
            for (std::size_t t = 0; t < catalog.tableCount; ++t) {
                if (catalog.tables[t].tableId == Ddl::TargetTableName(foreignKey.target)) {
                    targetTable = &catalog.tables[t];
                    targetField = FindField(*targetTable, Ddl::TargetMemberName(foreignKey.target));
                    break;
                }
            }
            if (localField == nullptr || targetTable == nullptr || targetField == nullptr) {
                return FailValue<std::string>(
                    DbCode::INVALID_ARGUMENT,
                    "foreign key references an unmapped column or table: " + std::string{meta.tableId});
            }
            auto own = QuoteIdentifier(localField->name);
            auto constraintName
                = QuoteIdentifier("fk_" + std::string{physicalTableName} + "_" + std::string{localField->name});
            auto targetTableId = QuoteIdentifier(targetTable->tableId);
            auto target = QuoteIdentifier(targetField->name);
            if (!own.ok() || !constraintName.ok() || !targetTableId.ok() || !target.ok()) {
                const auto& failure = !own.ok()            ? own :
                                      !constraintName.ok() ? constraintName :
                                      !targetTableId.ok()  ? targetTableId :
                                                             target;
                return FailValue<std::string>(failure.code, failure.diagnostic);
            }
            sql += ", CONSTRAINT " + *constraintName.value + " FOREIGN KEY (" + *own.value + ") REFERENCES "
                   + *targetTableId.value + " (" + *target.value + ")";
            if (foreignKey.cascadeDelete) {
                sql += " ON DELETE CASCADE";
            }
        }
        // 表选项：InnoDB + DYNAMIC 行格式（LONGBLOB 大行）；utf8mb4_bin 二进制
        // 排序对齐 SQLite BINARY 语义（等值比较大小写敏感、按字节序）
        sql += ") ENGINE=InnoDB ROW_FORMAT=COMPRESSED KEY_BLOCK_SIZE=8"
               " DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin";
        return {DbCode::OK, std::optional<std::string>{std::move(sql)}, 0, {}};
    }

    /// 标识符超 64 字符时截断为前缀 + 全名 FNV-1a 64 低 32 位十六进制后缀：
    /// 确定性（同一输入恒得同一输出，齐备性预检与 CREATE 共用，跨进程稳定），
    /// 同表内不同列组的碰撞概率 2^-32，可忽略。
    std::string CapIdentifierLength(const std::string& name)
    {
        constexpr std::size_t kMaxIdentifierLength = 64;
        if (name.size() <= kMaxIdentifierLength) {
            return name;
        }
        uint64_t hash = 14695981039346656037ULL;
        for (const char byte : name) {
            hash ^= static_cast<uint64_t>(static_cast<unsigned char>(byte));
            hash *= 1099511628211ULL;
        }
        const uint32_t suffixBits = static_cast<uint32_t>(hash & 0xFFFFFFFFULL);
        static constexpr char kHex[] = "0123456789abcdef";
        std::string suffix = "_";
        for (int shift = 28; shift >= 0; shift -= 4) {
            suffix += kHex[(suffixBits >> shift) & 0xFU];
        }
        return name.substr(0, kMaxIdentifierLength - suffix.size()) + suffix;
    }

    /// 物理索引名由表名与列名拼接（与 SQLite 版 DerivedIndexName 同构）。
    /// 【方言差异 #14：标识符长度上限】MariaDB 标识符最长 64 字符（errno 1059，
    /// SQLite 无此限制）——真实目录的复合索引（如 VirtualMemBlock 的
    /// device_id/src_type/dev_mapped_ptr/dev_mapped_end_ptr 四列组合，71 字符）
    /// 会超限。动态 scope 表的物理索引名还须追加物理表名（EnsureDynamicTable），
    /// 组合后统一经 CapIdentifierLength 收口。
    DbResult<std::string> DerivedIndexName(const Ddl::CatalogTableMeta& meta, const Ddl::IndexDesc& index)
    {
        std::string indexName = "idx_" + std::string{meta.tableId};
        for (std::size_t c = 0; c < index.columnCount; ++c) {
            const auto* field = FindField(meta, index.columns[c]);
            if (field == nullptr) {
                return FailValue<std::string>(
                    DbCode::INVALID_ARGUMENT, "index references an unmapped field: idx_" + std::string{meta.tableId});
            }
            indexName += "_" + std::string{field->name};
        }
        return {DbCode::OK, std::optional<std::string>{CapIdentifierLength(indexName)}, 0, {}};
    }

    /// 严格匹配可按区间形状使用的普通联合索引时返回该索引的物理名，供 SELECT
    /// 生成 FORCE INDEX 提示。候选索引（全部满足才参与匹配，任一不满足即跳过
    /// 该索引——继续使用优化器默认计划，不做任何通用优化器假设）：
    /// 1. 非唯一普通复合索引，列数 >= 2 且谓词子句数恰等于索引列数；
    /// 2. 末两列是互不相同的 uint64 字段（UNSIGNED 且 byteWidth ==
    ///    sizeof(uint64_t)）——半开区间端点 [start,end) 的类型合同（原区间索引
    ///    声明的编译期校验等价检查，与 BASE 一致允许 Derived 端点：VMB 的
    ///    start_end_ptr / dev_mapped_end_ptr 由 Hook 计算入库，是合法索引列；
    ///    不满足的普通索引（如 (device_id, is_freed) purge 索引）不获得提示）；
    /// 3. 谓词形状（全部满足才命中）：
    ///    a. 条件为平铺子句组合，全部 AND 连接，无括号分组；
    ///    b. 每个子句为单值比较（IN/PREFIX 不参与）；
    ///    c. 索引前导等值列（columnCount-2 列）各恰有一个 EQUAL 子句；
    ///    d. start 列恰有一个 LESS_OR_EQUAL、end 列恰有一个 GREATER——半开区间
    ///       [start,end) 的唯一谓词形态（与 FindDevMappedVirMemByDevice 同形）；
    ///    e. 无多余谓词；每列至多命中一次。
    /// 索引名经 DerivedIndexName 生成——与 CREATE INDEX 路径同一命名函数
    /// （含 64 字符截断与哈希后缀），不硬编码表名/索引名，保证提示的索引
    /// 与实际建出的索引一致。遍历顺序保持目录声明顺序。
    std::optional<std::string> MatchedIntervalIndexName(const Ddl::CatalogTableMeta& meta, const ConditionData& where)
    {
        const auto* clauses = std::get_if<std::vector<ConditionClause>>(&where);
        if (clauses == nullptr) {
            return std::nullopt;
        }
        for (std::size_t i = 0; i < meta.indexCount; ++i) {
            const Ddl::IndexDesc& index = meta.indexes[i];
            if (index.unique || index.columnCount < 2U || clauses->size() != index.columnCount) {
                continue;
            }
            uint16_t columnIds[Ddl::kMaxIndexColumns];
            bool columnsResolved = true;
            for (std::size_t c = 0; c < index.columnCount; ++c) {
                const auto* field = FindField(meta, index.columns[c]);
                if (field == nullptr) {
                    columnsResolved = false;
                    break;
                }
                columnIds[c] = static_cast<uint16_t>(field - meta.fields);
            }
            if (!columnsResolved) {
                continue;
            }
            const uint16_t startId = columnIds[index.columnCount - 2U];
            const uint16_t endId = columnIds[index.columnCount - 1U];
            // 末两列必须是互不相同的 uint64 字段（区间端点类型合同）。Derived
            // 端点合法（BASE 的区间索引校验从未排除 Derived；VMB 的
            // start_end_ptr / dev_mapped_end_ptr 由 Hook 计算后实际入库建索引）。
            const auto isIntervalEndpoint = [&meta](uint16_t fieldId) {
                const Ddl::FieldDesc& field = meta.fields[fieldId];
                return field.type == LogicalFieldType::UNSIGNED && field.byteWidth == sizeof(uint64_t);
            };
            if (startId == endId || !isIntervalEndpoint(startId) || !isIntervalEndpoint(endId)) {
                continue;
            }
            std::array<bool, Ddl::kMaxIndexColumns> columnSeen{};
            bool shapeOk = true;
            for (const auto& clause : *clauses) {
                if (clause.joinWithPrevious != ConditionJoin::NONE && clause.joinWithPrevious != ConditionJoin::AND) {
                    shapeOk = false;
                    break;
                }
                if (clause.openGroupsBefore != 0 || clause.closeGroupsAfter != 0
                    || !std::holds_alternative<DbScalar>(clause.value)) {
                    shapeOk = false;
                    break;
                }
                int slot = -1;
                if (clause.fieldId == startId && clause.compare == Compare::LESS_OR_EQUAL) {
                    slot = static_cast<int>(index.columnCount) - 2;
                } else if (clause.fieldId == endId && clause.compare == Compare::GREATER) {
                    slot = static_cast<int>(index.columnCount) - 1;
                } else if (clause.compare == Compare::EQUAL) {
                    for (std::size_t e = 0; e + 2U < index.columnCount; ++e) {
                        if (clause.fieldId == columnIds[e]) {
                            slot = static_cast<int>(e);
                            break;
                        }
                    }
                }
                if (slot < 0 || columnSeen[static_cast<std::size_t>(slot)]) {
                    shapeOk = false;
                    break;
                }
                columnSeen[static_cast<std::size_t>(slot)] = true;
            }
            if (!shapeOk) {
                continue;
            }
            auto name = DerivedIndexName(meta, index);
            if (name.ok() && name.value.has_value()) {
                return *name.value;
            }
            return std::nullopt;
        }
        return std::nullopt;
    }

    /// MariaDB 不支持 CREATE INDEX IF NOT EXISTS：齐备性由 Prepare 的
    /// information_schema 预检承担；此处只渲染普通 CREATE（UNIQUE）INDEX。
    DbResult<std::string> RenderCreateIndex(
        const Ddl::CatalogTableMeta& meta, const Ddl::IndexDesc& index, const std::string& indexName,
        const std::string& physicalTableName)
    {
        auto table = QuoteIdentifier(physicalTableName);
        auto name = QuoteIdentifier(indexName);
        if (!table.ok()) {
            return FailValue<std::string>(table.code, table.diagnostic);
        }
        if (!name.ok()) {
            return FailValue<std::string>(name.code, name.diagnostic);
        }
        std::string sql = "CREATE ";
        if (index.unique) {
            sql += "UNIQUE ";
        }
        sql += "INDEX " + *name.value + " ON " + *table.value + " (";
        bool first = true;
        for (std::size_t c = 0; c < index.columnCount; ++c) {
            const auto* field = FindField(meta, index.columns[c]);
            if (field == nullptr) {
                return FailValue<std::string>(
                    DbCode::INVALID_ARGUMENT, "index references an unmapped field: " + indexName);
            }
            auto columnName = QuoteIdentifier(field->name);
            if (!columnName.ok()) {
                return FailValue<std::string>(columnName.code, columnName.diagnostic);
            }
            sql += first ? "" : ", ";
            first = false;
            sql += *columnName.value;
            // TEXT/BLOB 列必须带前缀长度建索引（IpcNotify.name_or_key 等 TEXT 索引）
            if (field->type == LogicalFieldType::STRING || field->type == LogicalFieldType::BYTES) {
                sql += kTextIndexPrefix;
            }
        }
        sql += ")";
        return {DbCode::OK, std::optional<std::string>{std::move(sql)}, 0, {}};
    }

    /// MariaDB 方言的单字段排序表达式（公共排序遍历的方言回调）：BIGINT
    /// UNSIGNED 原生无符号语义（方言差异 #12：无需 SQLite 的
    /// "(col < 0) DIR, col DIR" 半区补丁），列名直出。direction 为 " ASC"/" DESC"。
    DbResult<void> AppendMariadbSortColumn(std::string& sql, const Ddl::FieldDesc& field, const char* direction)
    {
        const auto appended = AppendQuotedIdentifier(sql, field.name);
        if (!appended.ok()) {
            return appended;
        }
        sql += direction;
        return {DbCode::OK, 0, {}};
    }

    // ---- SELECT 行源（NextRow/ReadFieldInto 直通 mysql_stmt_fetch / fetch_column）----

    /// 取行游标：定长列绑静态缓冲直接取值；STRING/BYTES 列 buffer=nullptr +
    /// length_ptr，ReadField 经 mysql_stmt_fetch_column 按实际长度拉取（禁止
    /// CURSOR_TYPE_READ_ONLY 服务端游标与 store_result 全量缓冲）。
    struct MariadbRowCursor {
        MYSQL_STMT* stmt{nullptr};
        const Ddl::CatalogTableMeta* meta{nullptr};
        std::vector<MYSQL_BIND> binds{};        ///< 每个输出列一个绑定
        std::vector<unsigned long> lengths{};   ///< 每列实际长度（fetch 后有效）
        std::vector<my_bool> nulls{};           ///< 每列 NULL 标志
        std::vector<int64_t> signedValues{};    ///< SIGNED 列
        std::vector<uint64_t> unsignedValues{}; ///< UNSIGNED 列
        std::vector<double> realValues{};       ///< REAL 列
        std::vector<int8_t> tinyValues{};       ///< BOOLEAN 列
        /// fieldId → 输出列下标；空 = 全列恒等映射（列序 = 声明序 = fieldId，
        /// 与空投影的 SELECT 全列清单一致理解）；-1 = 未投影
        std::vector<int32_t> columnOfField{};
        bool drained{false};
        // 当前语句所属连接（SelectLocked 构造游标时显式传入引擎持有的 MYSQL*；
        // 复活只发生在游标运行前，运行期连接不变）
        MYSQL* connection{nullptr};
        // 首次取行/取列失败的原样记录：fetch 出错后语句进入终态，后续 fetch 恒
        // 返回 NO_DATA——真实错误只能由游标记忆，排空（DrainCursorLocked）据此
        // 处置连接，不得把 NO_DATA 误读为排空成功（会把断线连接继续当作可用）
        bool hadError{false};
        unsigned int lastErrno{0};
        std::string lastError;
        std::string lastSqlstate;

        /// 记录当前取行/取列失败的真实错误（调用一次；保留首个）。错误来源按
        /// libmariadb 实测不对称（见 CaptureStmtOrConnectionError）——stmt 与连接
        /// 对象取首个非零 errno，才能拿到真实原因
        void RecordStatementError(const char* what)
        {
            if (hadError) {
                return; // 保留首错，不被后续错误覆盖
            }
            hadError = true;
            const StmtErrorCapture captured = CaptureStmtOrConnectionError(stmt, connection, what);
            lastErrno = captured.errnoValue;
            lastError = captured.message;
            lastSqlstate = captured.sqlstate.empty() ? std::string{"-"} : captured.sqlstate;
        }

        static DbResult<bool> Next(void* opaque)
        {
            auto* self = static_cast<MariadbRowCursor*>(opaque);
            const int rc = mysql_stmt_fetch(self->stmt);
            if (rc == 0 || rc == MYSQL_DATA_TRUNCATED) {
                // 变长列绑 nullptr 缓冲：非 NULL 值以 MYSQL_DATA_TRUNCATED 提示，
                // 数据由 ReadField 的 fetch_column 按需拉取
                return {DbCode::OK, true, 0, {}};
            }
            if (rc == MYSQL_NO_DATA) {
                self->drained = true;
                return {DbCode::OK, false, 0, {}};
            }
            self->RecordStatementError("mariadb fetch failed: ");
            return {MapMariadbError(self->lastErrno), std::nullopt, 0, self->lastError};
        }

        /// 把当前行列值直接写最终 T 成员（镜像 SqliteRowCursor::ReadField 的
        /// byteWidth/capacity 语义；NULL 与未投影列跳过对应成员保持默认值）。
        /// 空映射 = 全列恒等映射（结果列序 = 声明序 = fieldId）。
        static DbResult<void> ReadField(void* opaque, uint16_t fieldId, void* destination, std::size_t capacity)
        {
            auto* self = static_cast<MariadbRowCursor*>(opaque);
            if (fieldId >= self->meta->fieldCount) {
                return InvalidArgument("mariadb row field is unmapped");
            }
            int32_t column = 0;
            if (self->columnOfField.empty()) {
                // 全列直读：结果列序 = 声明顺序 = fieldId
                column = static_cast<int32_t>(fieldId);
            } else {
                column = self->columnOfField[fieldId];
            }
            if (column < 0 || self->nulls[static_cast<std::size_t>(column)] != 0) {
                return {DbCode::OK, 0, {}};
            }
            const auto index = static_cast<std::size_t>(column);
            const Ddl::FieldDesc& field = self->meta->fields[fieldId];
            switch (field.type) {
                case LogicalFieldType::SIGNED: {
                    const int64_t value = self->signedValues[index];
                    switch (field.byteWidth) {
                        case 1: {
                            const int8_t narrow = static_cast<int8_t>(value);
                            std::memcpy(destination, &narrow, sizeof(narrow));
                            break;
                        }
                        case 2: {
                            const int16_t narrow = static_cast<int16_t>(value);
                            std::memcpy(destination, &narrow, sizeof(narrow));
                            break;
                        }
                        case 4: {
                            const int32_t narrow = static_cast<int32_t>(value);
                            std::memcpy(destination, &narrow, sizeof(narrow));
                            break;
                        }
                        default: {
                            std::memcpy(destination, &value, sizeof(value));
                            break;
                        }
                    }
                    break;
                }
                case LogicalFieldType::UNSIGNED: {
                    const uint64_t value = self->unsignedValues[index];
                    switch (field.byteWidth) {
                        case 1: {
                            const uint8_t narrow = static_cast<uint8_t>(value);
                            std::memcpy(destination, &narrow, sizeof(narrow));
                            break;
                        }
                        case 2: {
                            const uint16_t narrow = static_cast<uint16_t>(value);
                            std::memcpy(destination, &narrow, sizeof(narrow));
                            break;
                        }
                        case 4: {
                            const uint32_t narrow = static_cast<uint32_t>(value);
                            std::memcpy(destination, &narrow, sizeof(narrow));
                            break;
                        }
                        default: {
                            // BIGINT UNSIGNED 原生承载全 uint64 范围
                            std::memcpy(destination, &value, sizeof(value));
                            break;
                        }
                    }
                    break;
                }
                case LogicalFieldType::BOOLEAN: {
                    const bool flag = self->tinyValues[index] != 0;
                    std::memcpy(destination, &flag, sizeof(flag));
                    break;
                }
                case LogicalFieldType::REAL: {
                    if (field.byteWidth == sizeof(float)) {
                        const float narrow = static_cast<float>(self->realValues[index]);
                        std::memcpy(destination, &narrow, sizeof(narrow));
                    } else {
                        const double value = self->realValues[index];
                        std::memcpy(destination, &value, sizeof(value));
                    }
                    break;
                }
                case LogicalFieldType::STRING: {
                    const unsigned long length = self->lengths[index];
                    if (field.byteWidth == 0) {
                        // std::string：fetch_column 直写最终业务字段（唯一一次复制）
                        auto* member = static_cast<std::string*>(destination);
                        if (length == 0) {
                            member->clear();
                            return {DbCode::OK, 0, {}};
                        }
                        member->resize(length);
                        return self->FetchColumnInto(index, &*member->begin(), length, fieldId);
                    }
                    // 固定 char 数组：NUL 前字节承载语义；回写后显式补终止符
                    const std::size_t bound = capacity > 0 ? capacity : field.byteWidth;
                    const std::size_t count = std::min<std::size_t>(length, bound > 0 ? bound - 1 : 0);
                    if (count > 0) {
                        const auto fetched = self->FetchColumnInto(index, destination, count, fieldId);
                        if (!fetched.ok()) {
                            return fetched;
                        }
                    }
                    if (bound > 0) {
                        static_cast<char*>(destination)[count] = '\0';
                    }
                    break;
                }
                case LogicalFieldType::BYTES: {
                    const unsigned long length = self->lengths[index];
                    if (field.byteWidth == 0) {
                        // std::vector<uint8_t>：唯一一次拥有化复制
                        auto* member = static_cast<std::vector<uint8_t>*>(destination);
                        if (length == 0) {
                            member->clear();
                            return {DbCode::OK, 0, {}};
                        }
                        member->resize(length);
                        return self->FetchColumnInto(index, member->data(), length, fieldId);
                    }
                    if (capacity > 0) {
                        // 固定字节数组：按有效长度约束后的容量截断
                        const std::size_t count = std::min<std::size_t>(length, capacity);
                        if (count > 0) {
                            return self->FetchColumnInto(index, destination, count, fieldId);
                        }
                        break;
                    }
                    // 不透明定宽对象：仅字节数精确相等时整体回写（长度不足或超长
                    // 都保持目标默认值——先校验长度再写目标，杜绝半写）
                    if (length == field.byteWidth) {
                        return self->FetchColumnInto(index, destination, field.byteWidth, fieldId);
                    }
                    break;
                }
            }
            return {DbCode::OK, 0, {}};
        }

        /// 把当前行第 column 列的完整值拉入 buffer（长度不超过 bufferLength）
        DbResult<void> FetchColumnInto(std::size_t column, void* buffer, unsigned long bufferLength, uint16_t fieldId)
        {
            MYSQL_BIND bind{};
            unsigned long fetched = 0;
            bind.buffer_type
                = meta->fields[fieldId].type == LogicalFieldType::STRING ? MYSQL_TYPE_STRING : MYSQL_TYPE_BLOB;
            bind.buffer = buffer;
            bind.buffer_length = bufferLength;
            bind.length = &fetched;
            if (mysql_stmt_fetch_column(stmt, &bind, static_cast<unsigned int>(column), 0) != 0) {
                RecordStatementError("mariadb fetch column failed: ");
                return {MapMariadbError(lastErrno), 0, lastError};
            }
            return {DbCode::OK, 0, {}};
        }
    };

    /// 绑定结果列（execute 之后、首次 fetch 之前调用；列清单来自投影解析）
    DbResult<void> BindResultColumns(
        MariadbRowCursor& cursor, MYSQL_STMT* stmt, const Ddl::CatalogTableMeta& meta,
        const std::vector<std::size_t>& fieldIndices)
    {
        const std::size_t count = fieldIndices.size();
        cursor.stmt = stmt;
        cursor.meta = &meta;
        cursor.binds.assign(count, MYSQL_BIND{});
        cursor.lengths.assign(count, 0);
        cursor.nulls.assign(count, 0);
        cursor.signedValues.assign(count, 0);
        cursor.unsignedValues.assign(count, 0);
        cursor.realValues.assign(count, 0.0);
        cursor.tinyValues.assign(count, 0);
        for (std::size_t i = 0; i < count; ++i) {
            const Ddl::FieldDesc& field = meta.fields[fieldIndices[i]];
            MYSQL_BIND& bind = cursor.binds[i];
            bind.length = &cursor.lengths[i];
            bind.is_null = &cursor.nulls[i];
            switch (field.type) {
                case LogicalFieldType::SIGNED:
                    bind.buffer_type = MYSQL_TYPE_LONGLONG;
                    bind.buffer = &cursor.signedValues[i];
                    bind.buffer_length = sizeof(int64_t);
                    bind.is_unsigned = 0;
                    break;
                case LogicalFieldType::UNSIGNED:
                    bind.buffer_type = MYSQL_TYPE_LONGLONG;
                    bind.buffer = &cursor.unsignedValues[i];
                    bind.buffer_length = sizeof(uint64_t);
                    bind.is_unsigned = 1;
                    break;
                case LogicalFieldType::REAL:
                    bind.buffer_type = MYSQL_TYPE_DOUBLE;
                    bind.buffer = &cursor.realValues[i];
                    bind.buffer_length = sizeof(double);
                    bind.is_unsigned = 0;
                    break;
                case LogicalFieldType::BOOLEAN:
                    bind.buffer_type = MYSQL_TYPE_TINY;
                    bind.buffer = &cursor.tinyValues[i];
                    bind.buffer_length = sizeof(int8_t);
                    bind.is_unsigned = 0;
                    break;
                case LogicalFieldType::STRING:
                case LogicalFieldType::BYTES:
                    // 变长列：不取整列缓冲，长度/空标志随 fetch 就位，数据由
                    // ReadField 的 fetch_column 按需拉取
                    bind.buffer = nullptr;
                    bind.buffer_length = 0;
                    bind.is_unsigned = 0;
                    break;
            }
        }
        if (mysql_stmt_bind_result(stmt, cursor.binds.data()) != 0) {
            return {
                MapMariadbError(mysql_stmt_errno(stmt)), 0,
                "mariadb bind result failed: " + std::string(mysql_stmt_error(stmt))};
        }
        return {DbCode::OK, 0, {}};
    }

    /// dump 值字面量：按结果列元数据分派——数值列文本原样（BIGINT/DOUBLE 的
    /// 文本表示可精确往返）、带字符集的文本列转义引号（' \ NUL \n \r \Z；
    /// utf8 字节原样保留）、二进制列（binary charset 63）hex 字面量（任意字节
    /// 安全，包括无效 utf8 与嵌入 NUL）。
    void AppendDumpLiteral(std::string& out, const MYSQL_FIELD* field, const char* data, unsigned long length)
    {
        const bool numeric = field->type == MYSQL_TYPE_TINY || field->type == MYSQL_TYPE_SHORT
                             || field->type == MYSQL_TYPE_LONG || field->type == MYSQL_TYPE_FLOAT
                             || field->type == MYSQL_TYPE_DOUBLE || field->type == MYSQL_TYPE_LONGLONG
                             || field->type == MYSQL_TYPE_INT24 || field->type == MYSQL_TYPE_DECIMAL
                             || field->type == MYSQL_TYPE_NEWDECIMAL || field->type == MYSQL_TYPE_YEAR;
        if (numeric) {
            out.append(data, length);
            return;
        }
        if ((field->flags & BINARY_FLAG) != 0 && field->charsetnr == 63) {
            // 真·二进制列（BLOB 类，binary charset 63）→ hex 字面量（任意字节安全，
            // 包括无效 utf8 与嵌入 NUL）。不能只看 BINARY_FLAG：MariaDB 对 _bin
            // 排序规则的 TEXT 列（如 utf8mb4_bin，charsetnr 45）也置 BINARY_FLAG，
            // 文本列必须走字符串转义路径（10.11.14 实测）。
            static constexpr char kHex[] = "0123456789abcdef";
            out += "0x";
            for (unsigned long i = 0; i < length; ++i) {
                const unsigned char byte = static_cast<unsigned char>(data[i]);
                out += kHex[byte >> 4U];
                out += kHex[byte & 0x0FU];
            }
            return;
        }
        out += '\'';
        for (unsigned long i = 0; i < length; ++i) {
            const char character = data[i];
            switch (character) {
                case '\'':
                    out += "\\'";
                    break;
                case '\\':
                    out += "\\\\";
                    break;
                case '\0':
                    out += "\\0";
                    break;
                case '\n':
                    out += "\\n";
                    break;
                case '\r':
                    out += "\\r";
                    break;
                case '\x1A':
                    out += "\\Z";
                    break;
                default:
                    out += character;
                    break;
            }
        }
        out += '\'';
    }

    /// 元数据查询（文本协议 unbuffered：mysql_real_query + mysql_use_result 逐行
    /// 拉取 + free；禁止 store_result 全量缓冲的合同同样覆盖辅助查询）
    DbResult<std::vector<std::vector<std::string>>> QueryTextRows(MYSQL* mysql, const std::string& sql)
    {
        if (mysql_real_query(mysql, sql.c_str(), static_cast<unsigned long>(sql.size())) != 0) {
            return FailValue<std::vector<std::vector<std::string>>>(
                MapMariadbError(mysql_errno(mysql)),
                "mariadb metadata query failed: " + std::string(mysql_error(mysql)));
        }
        MYSQL_RES* result = mysql_use_result(mysql);
        if (result == nullptr) {
            return FailValue<std::vector<std::vector<std::string>>>(
                MapMariadbError(mysql_errno(mysql)),
                "mariadb metadata query has no result set: " + std::string(mysql_error(mysql)));
        }
        std::vector<std::vector<std::string>> rows;
        while (MYSQL_ROW row = mysql_fetch_row(result)) {
            std::vector<std::string> values(mysql_num_fields(result));
            for (unsigned int i = 0; i < values.size(); ++i) {
                values[i] = row[i] == nullptr ? std::string{} : std::string{row[i]};
            }
            rows.push_back(std::move(values));
        }
        const unsigned int error = mysql_errno(mysql);
        mysql_free_result(result);
        if (error != 0) {
            return FailValue<std::vector<std::vector<std::string>>>(
                MapMariadbError(error), "mariadb metadata fetch failed: " + std::string(mysql_error(mysql)));
        }
        return {DbCode::OK, std::optional<std::vector<std::vector<std::string>>>{std::move(rows)}, 0, {}};
    }

    /// 现有 schema 对象快照（information_schema；仅供 Prepare 齐备性预检与动态
    /// scope 表枚举。表名与 "表.索引" 对分别入集合——MariaDB 索引名按表命名空间）
    struct SchemaObjectSnapshot {
        std::set<std::string> tables;
        std::set<std::string> tableDotIndex;
    };

    DbResult<SchemaObjectSnapshot> ExistingSchemaObjects(MYSQL* mysql)
    {
        auto tableRows = QueryTextRows(
            mysql, "SELECT TABLE_NAME FROM information_schema.tables "
                   "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_TYPE = 'BASE TABLE'");
        if (!tableRows.ok()) {
            return FailValue<SchemaObjectSnapshot>(tableRows.code, tableRows.diagnostic);
        }
        auto indexRows = QueryTextRows(
            mysql, "SELECT TABLE_NAME, INDEX_NAME FROM information_schema.statistics "
                   "WHERE TABLE_SCHEMA = DATABASE()");
        if (!indexRows.ok()) {
            return FailValue<SchemaObjectSnapshot>(indexRows.code, indexRows.diagnostic);
        }
        SchemaObjectSnapshot snapshot;
        for (const auto& row : *tableRows.value) {
            if (!row.empty()) {
                snapshot.tables.insert(row[0]);
            }
        }
        for (const auto& row : *indexRows.value) {
            if (row.size() >= 2U) {
                snapshot.tableDotIndex.insert(row[0] + "." + row[1]);
            }
        }
        return {DbCode::OK, std::optional<SchemaObjectSnapshot>{std::move(snapshot)}, 0, {}};
    }

    /// 事务槽等待默认上限（项目既有排空口径 30 秒；私有常量，不新增配置面——
    /// 测试经 TestPeer 缩短，产品路径固定）
    constexpr std::chrono::milliseconds kTransactionWaitLimitMs{30000};

    /// InsertMany 快路径资格：目录为单列自增主键（普通非唯一 Index 不影响）、
    /// 无 Unique 约束索引、无外键。逐行记录形态（主键 0/NULL、指针非空、visit
    /// 成功）由 InsertManyFastPathEligible 在调用方数据上补充判定。
    bool EligibleCatalogShape(const Ddl::CatalogTableMeta& meta)
    {
        const Ddl::FieldDesc* primaryKey = nullptr;
        for (std::size_t f = 0; f < meta.fieldCount; ++f) {
            if (!meta.fields[f].primaryKey) {
                continue;
            }
            if (primaryKey != nullptr) {
                return false; // 复合主键
            }
            primaryKey = &meta.fields[f];
        }
        if (primaryKey == nullptr || !Ddl::HasProp(primaryKey->props, Ddl::Prop::AutoIncrement)) {
            return false;
        }
        for (std::size_t i = 0; i < meta.indexCount; ++i) {
            if (meta.indexes[i].unique) {
                return false; // Unique 约束的逐行失败定位合同留给基类路径
            }
        }
        return meta.foreignKeyCount == 0;
    }
} // namespace

// ---- 引擎主体 ----

struct MariaDbEngine::Impl {
    Impl(const Ddl::DatabaseDescriptor* descriptor, MariadbConnectionProfile connectionProfile)
        : catalog(descriptor),
          profile(std::move(connectionProfile))
    {}

    const Ddl::DatabaseDescriptor* catalog{nullptr};
    MariadbConnectionProfile profile{};
    MYSQL* mysql{nullptr};
    std::unique_ptr<MariadbStatementCache> statements{nullptr};
    StorageOpenMode mode{StorageOpenMode::READ_WRITE_CREATE};
    std::string dataNamespace;
    std::string database; ///< Open 期由 namespace+databaseId 派生
    enum class State : uint8_t { NOT_OPEN, OPENED, PREPARED, TRANSACTION_BUILDING, CLOSED };
    State state{State::NOT_OPEN};
    // 已成功创建的动态物理表名（唯一 DDL 生命周期状态；复用连接 mutex）。
    // 注意与 SQLite 的差异：MariaDB DDL 隐式提交且不可回滚——事务回滚不会
    // 撤销动态表建表，集合无需在 RollbackPhysicalLocked 失效。当前业务流
    // 全部动态表首次 DML 均为非事务路径（OpTaskTab 经 InsertOpTask 直插）。
    std::unordered_set<std::string> ensuredDynamicTables;
    // 活动事务槽：同一连接同时只允许一个事务，以 binding 对象地址识别。
    // DML 失败不改变引擎状态（物理事务仍开启），失败回滚由 Transaction
    // 统一调用 Rollback 完成。
    const TransactionBinding* activeBinding{nullptr};
    // 死锁中止标记：ER_LOCK_DEADLOCK 后 InnoDB 已回滚受害事务，该 binding
    // 的后续 DML 与 Commit 必须失败（不吞锁冲突/死锁错误）
    bool transactionAborted{false};
    // schema DDL 观察计数（仅启动/惰性建表路径；仅供测试固化结构合同）
    uint64_t staticDdlExecutions{0};
    uint64_t dynamicTableCreates{0};
    std::mutex mutex;
    // ---- 事务槽等待（单连接事务重叠合同；与 SQLite 引擎同构）----
    // 活动事务与其他线程的即时 CRUD/第二笔 Begin 重叠时：同线程立即失败
    // （防自锁，错误文本不变），其他线程经条件变量有界等待事务结束。成功
    // 等待零日志；超时返回 RESOURCE_EXHAUSTED（仅超时路径打一条既有
    // rate-limited 诊断）。
    std::condition_variable transactionIdle;
    std::thread::id transactionOwner{}; ///< 成功 Begin 的线程；离开事务时清空
    std::chrono::milliseconds transactionWaitLimit{kTransactionWaitLimitMs};
    // InsertMany 快路径（单条参数化多值 INSERT ... RETURNING 主键）观察计数：
    // 每成功执行一条批语句 +1。仅测试 peer 读取，产品代码不得依赖。
    uint64_t batchInsertExecutions{0};
    // 服务端版本测试 seam（产品恒 0 不干预；测试置如 100499UL 模拟 <10.5
    // 服务端的回退路径）。版本判定统一经 ServerVersionLocked。
    unsigned long serverVersionForTest{0};

    /// 即时操作/第二事务的事务槽等待（持锁调用；wait_until 释放引擎 mutex）。
    /// 无活动事务、或当前线程即事务 owner（不得等待自己）时原样返回 OK，由
    /// 调用方的既有状态检查给出原错误；其他线程的活动事务期间等待其结束，
    /// 被唤醒后由调用方重新完整校验状态（可能已回到 PREPARED 或进入 CLOSED）。
    DbResult<void> WaitForTransactionSlot(std::unique_lock<std::mutex>& lock)
    {
        if (state != State::TRANSACTION_BUILDING) {
            return {DbCode::OK, 0, {}};
        }
        if (transactionOwner == std::this_thread::get_id()) {
            return {DbCode::OK, 0, {}};
        }
        const auto startedAt = std::chrono::steady_clock::now();
        const bool idle = transactionIdle.wait_until(lock, startedAt + transactionWaitLimit, [this]() {
            return state != State::TRANSACTION_BUILDING;
        });
        if (idle) {
            return {DbCode::OK, 0, {}};
        }
        const long long waitedMs = storage_diag::ElapsedMsSince(startedAt);
        ReportFailLocked(
            "transaction-wait", 0, "-", DbCode::RESOURCE_EXHAUSTED, waitedMs, "transaction wait timed out");
        return MariadbError(
            DbCode::RESOURCE_EXHAUSTED, "mariadb transaction wait timed out after " + std::to_string(waitedMs)
                                            + " ms: a transaction is building on this route");
    }

    /// 离开 TRANSACTION_BUILDING 的全部路径（Commit/Rollback/Close/连接放弃）
    /// 在状态落定之后调用：清 owner 并唤醒全部等待者。
    void NotifyTransactionSlotReleasedLocked()
    {
        transactionOwner = {};
        transactionIdle.notify_all();
    }

    /// 服务端版本（mysql_get_server_version 形如 101114 = 10.11.14；测试 seam
    /// 非零时优先）。持锁路径调用。
    unsigned long ServerVersionLocked() const
    {
        if (serverVersionForTest != 0UL) {
            return serverVersionForTest;
        }
        return mysql != nullptr ? mysql_get_server_version(mysql) : 0UL;
    }

    /// prepared INSERT ... RETURNING 需要 MariaDB 10.5+（推荐/验收环境 10.11
    /// 满足；更旧服务端由 InsertMany 自动回退基类逐行实现）
    bool SupportsInsertReturningLocked() const { return ServerVersionLocked() >= 100500UL; }

    /// 统一语句失败处置（持锁路径）：映射错误码；连接级错误放弃连接（v1 无
    /// 自动重连）；死锁标记活动事务中止；缓存语句执行失败移出缓存（不留
    /// 脏 stmt）。sql 为空表示失败发生在语句进入缓存之前（如 prepare 失败）。
    DbResult<void> FailStatementLocked(const std::string& sql, const std::string& action)
    {
        const unsigned int raw = mysql_errno(mysql);
        const std::string message = mysql_error(mysql);
        const std::string sqlstate = mysql_sqlstate(mysql);
        const DbCode code = MapMariadbError(raw);
        if (code == DbCode::BACKEND_UNAVAILABLE) {
            AbandonConnectionLocked();
        } else {
            if (raw == kErLockDeadlock && state == State::TRANSACTION_BUILDING) {
                transactionAborted = true;
            }
            if (!sql.empty() && statements != nullptr) {
                statements->Discard(sql);
            }
        }
        ReportFailLocked(action, raw, sqlstate, code, -1, message);
        return MariadbError(
            code, action + " failed: " + message + " [mariadb errno " + std::to_string(raw) + " sqlstate "
                      + storage_diag::SqlstateOrDash(sqlstate) + "]");
    }

    /// 语句级失败处置（持锁路径）：raw/message/sqlstate 由调用方在失败点从
    /// MYSQL_STMT 捕获（stmt 关闭或连接清理后不可读；服务端对预备语句的拒绝
    /// 只落在 stmt 上——连接级 mysql_errno 可为 0，256p max_prepared_stmt_count
    /// 与 KILL 后 prepare 均实测此形状）。连接断开族 → 放弃连接（坏连接不得
    /// 继续当作可用）；死锁 → 标记事务中止；其余且 sql 非空 → Discard。
    /// elapsedMs>=0 时计入诊断耗时（<0 表示起点不可得）。
    DbResult<void> FailStatementAtLocked(
        unsigned int raw, const std::string& message, const std::string& sqlstate, const std::string& sql,
        const std::string& action, long long elapsedMs = -1)
    {
        const DbCode code = MapMariadbError(raw);
        if (code == DbCode::BACKEND_UNAVAILABLE) {
            AbandonConnectionLocked();
        } else {
            if (raw == kErLockDeadlock && state == State::TRANSACTION_BUILDING) {
                transactionAborted = true;
            }
            if (!sql.empty() && statements != nullptr) {
                statements->Discard(sql);
            }
        }
        ReportFailLocked(action, raw, sqlstate, code, elapsedMs, message);
        return MariadbError(
            code, action + " failed: " + message + " [mariadb errno " + std::to_string(raw) + " sqlstate "
                      + storage_diag::SqlstateOrDash(sqlstate) + "]");
    }

    /// 存储失败诊断单行（仅失败路径；字段合同见 storage_diag 注释）
    void ReportFailLocked(
        const std::string& stage, unsigned int raw, const std::string& sqlstate, DbCode code, long long elapsedMs,
        const std::string& detail)
    {
        storage_diag::Report(
            "[HCCL-VM-STORE] mariadb fail: wall=" + storage_diag::WallClockText() + " stage=" + stage + " db="
            + database + " errno=" + std::to_string(raw) + " sqlstate=" + storage_diag::SqlstateOrDash(sqlstate)
            + " code=" + std::to_string(static_cast<uint32_t>(code))
            + " elapsedMs=" + (elapsedMs >= 0 ? std::to_string(elapsedMs) : std::string{"-"})
            + " detail=" + storage_diag::ClipDetail(detail));
    }

    /// 复活重试资格：连接断开族错误 + 无活动事务 + 连接对象尚在。
    /// 事务路径一律不重试（事务上下文不能跨连接存活；失败按合同交
    /// Transaction 统一回滚）。
    bool CanReviveLocked(unsigned int raw) const
    {
        return IsConnectionGoneErrno(raw) && activeBinding == nullptr && mysql != nullptr;
    }

    /// prepare 失败错误的补全捕获：stmt 侧出参 errno 非 0 时直接用（缓存已在
    /// mysql_stmt_close 前捕获）；为 0 时仅当当前连接携带“连接断开族”错误才
    /// 提升连接侧取值——此前任何连接级断开都会使引擎放弃连接进入终态，活
    /// 连接上的断开族 errno 只可能来自本次 prepare（新鲜）；非断开族错误
    /// （如早前被容忍的 1050/1061 DDL 错误经 real_query 后驻留在连接对象上）
    /// 不提升，避免把陈旧错误误归因为本次 prepare。仍为 0 时保持 0 +
    /// 调用方补 "no diagnostic captured"（明确失败，不伪装成功）。
    PrepareErrorCapture CompletePrepareErrorLocked(
        unsigned int prepareErrno, const std::string& prepareError, const std::string& prepareSqlstate)
    {
        PrepareErrorCapture captured{prepareErrno, prepareError, prepareSqlstate};
        if (captured.errnoValue != 0U || mysql == nullptr) {
            return captured;
        }
        const unsigned int connectionErrno = mysql_errno(mysql);
        if (IsConnectionGoneErrno(connectionErrno) || connectionErrno == kCrConnectionError
            || connectionErrno == kCrConnHostError) {
            captured.errnoValue = connectionErrno;
            captured.message = mysql_error(mysql);
            captured.sqlstate = mysql_sqlstate(mysql);
        }
        return captured;
    }

    /// 复活失败诊断：原始操作错误与重连错误的必要字段都保留（两段式），
    /// 取重连失败的错误码（连接/重连失败族 → BACKEND_UNAVAILABLE）
    std::string ReviveFailureDiagnosticLocked(
        const std::string& operationAction, const PrepareErrorCapture& original, const DbResult<void>& revived) const
    {
        const std::string originalDetail
            = original.message.empty() ? std::string{"no diagnostic captured"} : original.message;
        return operationAction + " failed: " + originalDetail + " [mariadb errno " + std::to_string(original.errnoValue)
               + " sqlstate " + storage_diag::SqlstateOrDash(original.sqlstate)
               + "]; revive failed: " + revived.diagnostic;
    }

    /// 连接复活（持锁）：按 Open 期同一 profile/database 重建长连接，并重建
    /// 语句缓存（MYSQL_STMT 属于旧连接，复活后按需重新 prepare）。仅允许无
    /// 活动事务时调用（由 CanReviveLocked 保证）；动态表缓存一并清空（服务端
    /// 表仍在，惰性重建幂等——1061 容忍）。失败时引擎进入 CLOSED 终态。
    /// 语义对齐 redis_sync_client 的 BROKEN 原地重建。stage 仅供诊断日志标注
    /// 触发阶段（prepare select / execute insert 等）。
    /// 【会话一致性核对（逐项＝Open 期）】charset=utf8mb4、connect/read/write
    /// 三类超时（同 profile 秒值）、MYSQL_OPT_COMPRESS（同 profile 开关，
    /// 重连不回退硬编码值）、autocommit=1——全部经 ApplyMariadbConnectionOptions
    /// 统一施加；目标 database 直连（库在 Open 期已创建，服务端持久）；隔离级别
    /// 引擎两处都不做会话级设置（由服务端全局配置决定，复活前后同一来源）——
    /// 不引入任何 Open 期没有的会话参数。
    DbResult<void> ReconnectLocked(const char* stage = "")
    {
        const auto startedAt = std::chrono::steady_clock::now();
        const std::string action = "mariadb reconnect to database";
        if (statements != nullptr) {
            statements->ClearAll();
        }
        if (mysql != nullptr) {
            (void)mysql_close(mysql);
            mysql = nullptr;
        }
        EnsureMariadbLibraryInit();
        auto endpoint = ParseMariadbUri(profile.uri);
        if (!endpoint.ok() || endpoint.value->host.empty()) {
            state = State::CLOSED;
            ReportReviveLocked(stage, false, 0, "endpoint lost");
            return MariadbError(DbCode::INVALID_ARGUMENT, action + " failed: endpoint lost");
        }
        MYSQL* fresh = mysql_init(nullptr);
        if (fresh == nullptr) {
            state = State::CLOSED;
            ReportReviveLocked(stage, false, storage_diag::ElapsedMsSince(startedAt), "mysql_init");
            return MariadbError(DbCode::RESOURCE_EXHAUSTED, action + " failed: mysql_init");
        }
        auto options = ApplyMariadbConnectionOptions(fresh, profile);
        if (!options.ok()) {
            (void)mysql_close(fresh);
            state = State::CLOSED;
            ReportReviveLocked(stage, false, storage_diag::ElapsedMsSince(startedAt), options.diagnostic);
            return MariadbError(options.code, action + " failed: " + options.diagnostic);
        }
        // 库已在 Open 期创建/校验，直接连目标 database
        if (mysql_real_connect(
                fresh, endpoint.value->host.c_str(), endpoint.value->user.c_str(), endpoint.value->password.c_str(),
                database.c_str(), endpoint.value->port, nullptr, 0)
            == nullptr) {
            const unsigned int raw = mysql_errno(fresh);
            const std::string message = mysql_error(fresh);
            const std::string sqlstate = mysql_sqlstate(fresh);
            (void)mysql_close(fresh);
            ensuredDynamicTables.clear();
            state = State::CLOSED;
            ReportReviveLocked(
                stage, false, storage_diag::ElapsedMsSince(startedAt), "errno " + std::to_string(raw) + " " + message);
            return MariadbError(
                MapMariadbError(raw), action + " failed: " + message + " [mariadb errno " + std::to_string(raw)
                                          + " sqlstate " + storage_diag::SqlstateOrDash(sqlstate) + "]");
        }
        mysql = fresh;
        if (mysql_autocommit(mysql, 1) != 0) {
            const unsigned int raw = mysql_errno(mysql);
            const std::string message = mysql_error(mysql);
            const std::string sqlstate = mysql_sqlstate(mysql);
            AbandonConnectionLocked();
            ReportReviveLocked(
                stage, false, storage_diag::ElapsedMsSince(startedAt),
                "autocommit errno " + std::to_string(raw) + " " + message);
            return MariadbError(
                MapMariadbError(raw), action + " failed: autocommit setup: " + message + " [mariadb errno "
                                          + std::to_string(raw) + " sqlstate " + storage_diag::SqlstateOrDash(sqlstate)
                                          + "]");
        }
        statements = std::make_unique<MariadbStatementCache>(mysql);
        ensuredDynamicTables.clear();
        ReportReviveLocked(stage, true, storage_diag::ElapsedMsSince(startedAt), std::string{});
        return {DbCode::OK, 0, {}};
    }

    /// 复活诊断单行（成功/失败各一条；携带客户端三类超时便于现场对照）
    void ReportReviveLocked(const std::string& stage, bool ok, long long elapsedMs, const std::string& failDetail)
    {
        storage_diag::Report(
            "[HCCL-VM-STORE] mariadb revive: wall=" + storage_diag::WallClockText()
            + " stage=" + (stage.empty() ? std::string{"-"} : stage) + " db=" + database
            + " result=" + (ok ? "ok" : "fail") + " elapsedMs=" + std::to_string(elapsedMs)
            + " timeoutsS=" + std::to_string(profile.connectionEstablishmentTimeoutS) + "/"
            + std::to_string(profile.packetReadTimeoutS) + "/" + std::to_string(profile.packetWriteTimeoutS)
            + (failDetail.empty() ? std::string{} : " detail=" + storage_diag::ClipDetail(failDetail)));
    }

    /// 连接死亡处置：清语句缓存（先于 mysql_close——MYSQL_STMT 属于连接）、
    /// 关闭连接、引擎进入终态 CLOSED（后续操作按未打开拒绝）
    void AbandonConnectionLocked()
    {
        if (statements != nullptr) {
            statements->ClearAll();
        }
        if (mysql != nullptr) {
            (void)mysql_close(mysql);
            mysql = nullptr;
        }
        ensuredDynamicTables.clear();
        activeBinding = nullptr;
        transactionAborted = false;
        state = State::CLOSED;
        NotifyTransactionSlotReleasedLocked(); // 等待者按 CLOSED 有限返回
    }

    /// 无参 SQL 执行（DDL/事务边界/USE；mysql_real_query 直发）。
    /// tolerateTableExists=true 时 ER_TABLE_EXISTS_ERROR 按幂等成功处理
    /// （防御：正常路径由齐备性预检跳过 CREATE，不应触发）；
    /// tolerateIndexExists=true 时 ER_DUP_KEYNAME（1061）按幂等成功处理
    /// （动态表跨进程重入：rank 进程建表建索引后，checker 等其他进程首次
    /// 访问同一物理 scope 表时索引已存在——MariaDB 无 CREATE INDEX IF NOT
    /// EXISTS 语法，镜像 SQLite 该语种的幂等语义）
    DbResult<void> ExecuteSimpleLocked(
        const std::string& sql, const char* action, bool tolerateTableExists = false, bool tolerateIndexExists = false)
    {
        if (mysql == nullptr) {
            return InvalidState("mariadb connection is not open");
        }
        if (mysql_real_query(mysql, sql.c_str(), static_cast<unsigned long>(sql.size())) != 0) {
            const unsigned int raw = mysql_errno(mysql);
            if (tolerateTableExists && raw == kErTableExistsError) {
                return {DbCode::OK, 0, {}};
            }
            if (tolerateIndexExists && raw == kErDupKeyName) {
                return {DbCode::OK, 0, {}};
            }
            // 连接断开族 + 无活动事务：复活连接后重发一次。容忍语义在重试后
            // 复检（首试可能已在服务端执行、仅回包丢失——幂等 DDL 落此形状）。
            if (CanReviveLocked(raw)) {
                const auto revived = ReconnectLocked(("simple " + std::string{action}).c_str());
                if (revived.ok()) {
                    if (mysql_real_query(mysql, sql.c_str(), static_cast<unsigned long>(sql.size())) == 0) {
                        return {DbCode::OK, 0, {}};
                    }
                    const unsigned int retryRaw = mysql_errno(mysql);
                    if (tolerateTableExists && retryRaw == kErTableExistsError) {
                        return {DbCode::OK, 0, {}};
                    }
                    if (tolerateIndexExists && retryRaw == kErDupKeyName) {
                        return {DbCode::OK, 0, {}};
                    }
                    return FailStatementLocked(std::string{}, "mariadb " + std::string{action});
                }
                return {revived.code, 0, "mariadb " + std::string{action} + " failed: " + revived.diagnostic};
            }
            return FailStatementLocked(std::string{}, "mariadb " + std::string{action});
        }
        return {DbCode::OK, 0, {}};
    }

    /// 参数化 DML 执行（语句缓存取 stmt → bind → execute → affected/insert_id；
    /// 调用前已持锁。结果集类语句不适用本入口）
    /// allowRevive=false 表示本调用已是复活重试轮（有界重试，不再递归）。
    DbResult<std::pair<uint64_t, uint64_t>> ExecutePreparedDmlLocked(
        const std::string& sql, const BoundParams& params, const char* action, bool allowRevive = true)
    {
        const auto startedAt = std::chrono::steady_clock::now();
        if (mysql == nullptr || statements == nullptr) {
            return FailValue<std::pair<uint64_t, uint64_t>>(DbCode::INVALID_STATE, "mariadb connection is not open");
        }
        unsigned int prepareErrno = 0;
        std::string prepareError;
        std::string prepareSqlstate;
        MYSQL_STMT* stmt = statements->Acquire(sql, &prepareErrno, &prepareError, &prepareSqlstate);
        if (stmt == nullptr) {
            // prepare/init 失败：errno/error/sqlstate 已在 stmt 关闭前由缓存捕获
            // （256p 现场：服务端 Prepared_stmt_count 打满 max_prepared_stmt_count
            // 时 mysql_errno(mysql) 为 0 而 stmt errno 携带真实原因——必须用
            // 出参才能报出）；stmt 侧为 0 时按 CompletePrepareErrorLocked 从当前
            // 连接补全断开族错误。
            // prepare 阶段的连接断开族：复活后整体重试一次（缓存已随复活
            // 清空，重试路径重新 prepare）。事务路径不重试。
            const PrepareErrorCapture prepare = CompletePrepareErrorLocked(prepareErrno, prepareError, prepareSqlstate);
            if (allowRevive && CanReviveLocked(prepare.errnoValue)) {
                const auto revived = ReconnectLocked(("prepare " + std::string{action}).c_str());
                if (revived.ok()) {
                    return ExecutePreparedDmlLocked(sql, params, action, false);
                }
                return FailValue<std::pair<uint64_t, uint64_t>>(
                    revived.code,
                    ReviveFailureDiagnosticLocked("mariadb prepare " + std::string{action}, prepare, revived));
            }
            const std::string detail
                = prepare.message.empty() ? std::string{"no diagnostic captured"} : prepare.message;
            const DbCode code = MapMariadbError(prepare.errnoValue);
            if (IsConnectionGoneErrno(prepare.errnoValue)) {
                // 重试额度耗尽/事务路径：坏连接不得继续当作可用
                AbandonConnectionLocked();
            }
            ReportFailLocked(
                "prepare " + std::string{action}, prepare.errnoValue, prepare.sqlstate, code,
                storage_diag::ElapsedMsSince(startedAt), detail);
            return FailValue<std::pair<uint64_t, uint64_t>>(
                code, "mariadb prepare " + std::string{action} + " failed: " + detail + " [mariadb errno "
                          + std::to_string(prepare.errnoValue) + " sqlstate "
                          + storage_diag::SqlstateOrDash(prepare.sqlstate) + "]");
        }
        if (params.Size() > 0 && mysql_stmt_bind_param(stmt, params.Data()) != 0) {
            // bind 失败：真实错误在 stmt 上（客户端侧参数缓冲错误连接级 errno 为 0）
            const unsigned int raw = mysql_stmt_errno(stmt);
            const std::string message = mysql_stmt_error(stmt);
            const std::string sqlstate = mysql_stmt_sqlstate(stmt);
            const auto failed = FailStatementAtLocked(
                raw, message, sqlstate, sql, "mariadb bind " + std::string{action},
                storage_diag::ElapsedMsSince(startedAt));
            return FailValue<std::pair<uint64_t, uint64_t>>(failed.code, failed.diagnostic);
        }
        if (mysql_stmt_execute(stmt) != 0) {
            const unsigned int raw = mysql_stmt_errno(stmt);
            // 连接断开族 + 无活动事务：复活后整体重试一次。已知边界：若首试
            // 已在服务端提交而回包丢失，重试可能产生重复行——仅连接级错误触发
            // 该窗口，远小于直接失败的场景损失（1024p swap 冻结实测主路径）。
            if (allowRevive && CanReviveLocked(raw)) {
                const auto revived = ReconnectLocked(action);
                if (revived.ok()) {
                    return ExecutePreparedDmlLocked(sql, params, action, false);
                }
                return FailValue<std::pair<uint64_t, uint64_t>>(
                    revived.code, "mariadb " + std::string{action} + " failed: " + revived.diagnostic);
            }
            const std::string message = mysql_stmt_error(stmt);
            const std::string sqlstate = mysql_stmt_sqlstate(stmt);
            const auto failed = FailStatementAtLocked(
                raw, message, sqlstate, sql, "mariadb " + std::string{action}, storage_diag::ElapsedMsSince(startedAt));
            return FailValue<std::pair<uint64_t, uint64_t>>(failed.code, failed.diagnostic);
        }
        return {
            DbCode::OK,
            std::optional<std::pair<uint64_t, uint64_t>>{std::make_pair(
                static_cast<uint64_t>(mysql_stmt_affected_rows(stmt)),
                static_cast<uint64_t>(mysql_stmt_insert_id(stmt)))},
            0,
            {}};
    }

    /// 动态 scope 表首次访问时按目录惰性建表（索引随物理表名唯一化；镜像
    /// SQLite EnsureDynamicTable）。
    /// 【已知语义边界】MariaDB DDL 隐式提交：若本函数在活动事务内执行，
    /// 建表前的事务内容已被服务端提交且不可回滚（SQLite 的"建表随事务回滚"
    /// 在 MariaDB 方言下不存在）。当前业务流全部动态表首次 DML 均为非事务
    /// 路径（OpTaskTab 经 InsertOpTask 直插），本路径不触发；作为已知限制
    /// 记录于部署 runbook。
    DbResult<void> EnsureDynamicTable(const Ddl::CatalogTableMeta& meta, const std::string& physicalTableName)
    {
        if (ensuredDynamicTables.count(physicalTableName) > 0) {
            return {DbCode::OK, 0, {}};
        }
        auto createTable = RenderCreateTable(*catalog, meta, physicalTableName);
        if (!createTable.ok()) {
            return {createTable.code, createTable.affectedRows, createTable.diagnostic};
        }
        auto created = ExecuteSimpleLocked(*createTable.value, "create dynamic table", true);
        if (!created.ok()) {
            return created;
        }
        for (std::size_t i = 0; i < meta.indexCount; ++i) {
            auto baseName = DerivedIndexName(meta, meta.indexes[i]);
            if (!baseName.ok()) {
                return {baseName.code, baseName.affectedRows, baseName.diagnostic};
            }
            // 物理索引名按物理表名派生，保证不同 pid scope 表的索引互不冲突；
            // 组合后再过 64 字符上限（基名截断+物理表名可能超限，方言差异 #14）
            const std::string indexName = CapIdentifierLength(*baseName.value + "_" + physicalTableName);
            auto createIndex = RenderCreateIndex(meta, meta.indexes[i], indexName, physicalTableName);
            if (!createIndex.ok()) {
                return {createIndex.code, createIndex.affectedRows, createIndex.diagnostic};
            }
            // 1061 幂等容忍：跨进程重入时索引已存在（见 ExecuteSimpleLocked）
            auto indexed = ExecuteSimpleLocked(*createIndex.value, "create dynamic index", false, true);
            if (!indexed.ok()) {
                return indexed;
            }
        }
        ensuredDynamicTables.insert(physicalTableName);
        ++dynamicTableCreates;
        return {DbCode::OK, 0, {}};
    }

    /// 解析物理表名并按需惰性建表（CLEAR 不建表：表不存在时按缺表报错，
    /// 与既有语义一致）
    DbResult<std::string>
    ResolvePhysicalTable(const Ddl::CatalogTableMeta& meta, const TableScope& scope, bool allowDynamicCreate)
    {
        auto physical = Relational::PhysicalTableNameOf(meta, scope);
        if (!physical.ok()) {
            return physical;
        }
        if (allowDynamicCreate && meta.scope == Ddl::ScopeKind::DynamicPid) {
            auto ensured = EnsureDynamicTable(meta, *physical.value);
            if (!ensured.ok()) {
                return FailValue<std::string>(ensured.code, ensured.diagnostic);
            }
        }
        return physical;
    }

    /// unbuffered 结果集排空（materializeRows 只取首行的单对象 sink 不会
    /// 自然取尽；同连接未取尽前不得发新语句——单 mutex 串行下由本函数收尾）
    DbResult<void> DrainCursorLocked(MariadbRowCursor& cursor)
    {
        if (cursor.drained) {
            return {DbCode::OK, 0, {}};
        }
        if (cursor.hadError) {
            // 取行/取列已失败：语句进入终态，后续 fetch 恒返回 NO_DATA——
            // 真实错误只能取游标记录（errno/文本/SQLSTATE 在失败点捕获）。
            // 连接断开族在此放弃连接（坏连接不得继续当作可用）；否则错误
            // 原样上抛、不放弃连接（连接可用性优先）
            return FailStatementAtLocked(
                cursor.lastErrno, cursor.lastError, cursor.lastSqlstate, std::string{}, "mariadb drain result");
        }
        while (true) {
            const int rc = mysql_stmt_fetch(cursor.stmt);
            if (rc == 0 || rc == MYSQL_DATA_TRUNCATED) {
                continue;
            }
            if (rc == MYSQL_NO_DATA) {
                cursor.drained = true;
                return {DbCode::OK, 0, {}};
            }
            // 排空失败：错误来源不对称（见 CaptureStmtOrConnectionError）——
            // stmt 与连接对象取首个非零 errno，在连接清理前捕获
            const StmtErrorCapture captured
                = CaptureStmtOrConnectionError(cursor.stmt, mysql, "mariadb drain result failed: ");
            return FailStatementAtLocked(
                captured.errnoValue, captured.message, captured.sqlstate, std::string{}, "mariadb drain result");
        }
    }
};

MariaDbEngine::MariaDbEngine(const Ddl::DatabaseDescriptor* catalog, MariadbConnectionProfile profile)
    : impl_(std::make_unique<Impl>(catalog, std::move(profile)))
{}

// 析构只释放连接资源（先清语句缓存再 mysql_close）；不持久化、不重试。
MariaDbEngine::~MariaDbEngine()
{
    if (impl_ == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->mysql != nullptr) {
        impl_->AbandonConnectionLocked();
    }
}

// 建立长连接：mysql_init + 连接参数（utf8mb4、三类超时；方言差异 #5）+
// mysql_real_connect（READ_WRITE_CREATE 先建库再 USE；READ_EXISTING/
// BACKUP_ONLY 直接连目标库——缺库明确失败）+ autocommit=1。
DbResult<void> MariaDbEngine::Open(const EngineOpenRequest& request)
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->state == Impl::State::CLOSED) {
        return InvalidState("engine is closed");
    }
    if (impl_->state != Impl::State::NOT_OPEN) {
        return InvalidState("engine is already open");
    }
    auto endpoint = ParseMariadbUri(impl_->profile.uri);
    if (!endpoint.ok()) {
        return {endpoint.code, 0, endpoint.diagnostic};
    }
    if (impl_->profile.uri.empty() || endpoint.value->host.empty()) {
        return MariadbError(DbCode::INVALID_ARGUMENT, "mariadb endpoint uri is empty or malformed");
    }
    // 程序化构造的 profile 越界在建连前拒绝（YAML 解析期已校验同一范围）
    auto profileTimeouts = ValidateMariadbProfileTimeouts(impl_->profile);
    if (!profileTimeouts.ok()) {
        return {profileTimeouts.code, 0, profileTimeouts.diagnostic};
    }
    EnsureMariadbLibraryInit();

    impl_->database = DeriveDatabaseName(
        request.dataNamespace, impl_->catalog == nullptr ? std::string_view{} : impl_->catalog->databaseId);
    auto databaseName = QuoteIdentifier(impl_->database);
    if (!databaseName.ok()) {
        return {databaseName.code, 0, databaseName.diagnostic};
    }

    MYSQL* mysql = mysql_init(nullptr);
    if (mysql == nullptr) {
        return MariadbError(DbCode::RESOURCE_EXHAUSTED, "mariadb mysql_init failed (out of memory)");
    }
    auto profileOptions = ApplyMariadbConnectionOptions(mysql, impl_->profile);
    if (!profileOptions.ok()) {
        (void)mysql_close(mysql);
        return MariadbError(
            profileOptions.code, "failed to configure mariadb connection options: " + profileOptions.diagnostic);
    }
    // READ_EXISTING/BACKUP_ONLY 直连目标 database（缺库由服务端拒绝）；
    // READ_WRITE_CREATE 先连服务器再 CREATE DATABASE IF NOT EXISTS + USE。
    const char* database = request.mode == StorageOpenMode::READ_WRITE_CREATE ? nullptr : databaseName.value->c_str();
    const auto connectStartedAt = std::chrono::steady_clock::now();
    if (mysql_real_connect(
            mysql, endpoint.value->host.c_str(), endpoint.value->user.c_str(), endpoint.value->password.c_str(),
            database, endpoint.value->port, nullptr, 0)
        == nullptr) {
        const unsigned int raw = mysql_errno(mysql);
        const std::string message = mysql_error(mysql);
        const std::string sqlstate = mysql_sqlstate(mysql);
        const long long elapsedMs = storage_diag::ElapsedMsSince(connectStartedAt);
        (void)mysql_close(mysql);
        storage_diag::Report(
            "[HCCL-VM-STORE] mariadb connect: wall=" + storage_diag::WallClockText() + " endpoint="
            + SanitizedEndpoint(*endpoint.value) + " db=" + impl_->database + " errno=" + std::to_string(raw)
            + " sqlstate=" + storage_diag::SqlstateOrDash(sqlstate) + " elapsedMs=" + std::to_string(elapsedMs)
            + " timeoutsS=" + std::to_string(impl_->profile.connectionEstablishmentTimeoutS) + "/"
            + std::to_string(impl_->profile.packetReadTimeoutS) + "/"
            + std::to_string(impl_->profile.packetWriteTimeoutS) + " detail=" + storage_diag::ClipDetail(message));
        return MariadbError(
            MapMariadbError(raw), "failed to connect to mariadb " + SanitizedEndpoint(*endpoint.value) + " database `"
                                      + impl_->database + "`: " + message + " [mariadb errno " + std::to_string(raw)
                                      + " sqlstate " + storage_diag::SqlstateOrDash(sqlstate) + "]");
    }
    impl_->mysql = mysql;
    impl_->mode = request.mode;
    impl_->dataNamespace = request.dataNamespace;
    if (request.mode == StorageOpenMode::READ_WRITE_CREATE) {
        auto created = impl_->ExecuteSimpleLocked(
            "CREATE DATABASE IF NOT EXISTS " + *databaseName.value
                + " DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin",
            "create database");
        if (!created.ok()) {
            impl_->AbandonConnectionLocked();
            return created;
        }
        if (mysql_select_db(impl_->mysql, impl_->database.c_str()) != 0) {
            const auto failed = impl_->FailStatementLocked(std::string{}, "mariadb select database");
            impl_->AbandonConnectionLocked();
            return failed;
        }
    }
    // 非事务 DML 逐句提交（对齐 SQLite 立即模式）；事务期由 START TRANSACTION
    // 显式挂起 autocommit
    if (mysql_autocommit(impl_->mysql, 1) != 0) {
        const auto failed = impl_->FailStatementLocked(std::string{}, "mariadb autocommit setup");
        impl_->AbandonConnectionLocked();
        return failed;
    }
    impl_->statements = std::make_unique<MariadbStatementCache>(impl_->mysql);
    impl_->state = Impl::State::OPENED;
    return {DbCode::OK, 0, {}};
}

// 静态 schema 幂等发布（owner 打开意图）：先以 information_schema 只读快照做
// 齐备性预检——已齐备直接进入 PREPARED（不重放 DDL）；冷库/缺项才逐条执行
// CREATE TABLE IF NOT EXISTS / CREATE INDEX。
// 【与 SQLite 的关键差异】MariaDB DDL 隐式提交——不能像 SQLite 那样把全部
// 建表/建索引包在 BEGIN IMMEDIATE 单事务内原子发布；改为逐条幂等执行
// （CREATE TABLE 带 IF NOT EXISTS 且容忍 ER_TABLE_EXISTS_ERROR；CREATE INDEX
// 依赖预检跳过已存在项——MariaDB 无 CREATE INDEX IF NOT EXISTS 语法）。
// consumer（READ_EXISTING）与 BACKUP_ONLY 只读打开，不执行任何 DDL。
DbResult<void> MariaDbEngine::Prepare()
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->state == Impl::State::NOT_OPEN || impl_->state == Impl::State::CLOSED) {
        return InvalidState("engine is not open");
    }
    if (impl_->state != Impl::State::OPENED) {
        return InvalidState("engine is already prepared");
    }
    if (impl_->mode == StorageOpenMode::BACKUP_ONLY || impl_->mode == StorageOpenMode::READ_EXISTING) {
        impl_->state = Impl::State::PREPARED;
        return {DbCode::OK, 0, {}};
    }
    if (impl_->mysql == nullptr) {
        return MariadbError(DbCode::INVALID_STATE, "mariadb connection is not open");
    }
    if (impl_->catalog == nullptr) {
        return MariadbError(DbCode::INVALID_STATE, "mariadb engine has no database catalog");
    }
    auto existing = ExistingSchemaObjects(impl_->mysql);
    if (!existing.ok()) {
        return {existing.code, existing.affectedRows, existing.diagnostic};
    }
    // 齐备性预检：目录声明的全部静态表与索引均已存在 → 不重放 DDL（已完成
    // 发布的库按既有 schema 直接使用；动态 PID 表不参与齐备性判定）
    bool schemaComplete = true;
    for (std::size_t t = 0; schemaComplete && t < impl_->catalog->tableCount; ++t) {
        const auto& meta = impl_->catalog->tables[t];
        if (meta.scope == Ddl::ScopeKind::DynamicPid) {
            continue;
        }
        if (existing.value->tables.count(std::string{meta.tableId}) == 0) {
            schemaComplete = false;
            break;
        }
        for (std::size_t i = 0; i < meta.indexCount; ++i) {
            auto indexName = DerivedIndexName(meta, meta.indexes[i]);
            if (!indexName.ok()) {
                return {indexName.code, indexName.affectedRows, indexName.diagnostic};
            }
            if (existing.value->tableDotIndex.count(std::string{meta.tableId} + "." + *indexName.value) == 0) {
                schemaComplete = false;
                break;
            }
        }
    }
    if (schemaComplete) {
        impl_->state = Impl::State::PREPARED;
        return {DbCode::OK, 0, {}};
    }
    for (std::size_t t = 0; t < impl_->catalog->tableCount; ++t) {
        const auto& meta = impl_->catalog->tables[t];
        if (meta.scope == Ddl::ScopeKind::DynamicPid) {
            continue; // 动态表按物理 scope 惰性创建
        }
        auto createTable = RenderCreateTable(*impl_->catalog, meta, std::string{meta.tableId});
        if (!createTable.ok()) {
            return {createTable.code, createTable.affectedRows, createTable.diagnostic};
        }
        auto created = impl_->ExecuteSimpleLocked(*createTable.value, "create table", true);
        if (!created.ok()) {
            return created;
        }
        ++impl_->staticDdlExecutions;
        for (std::size_t i = 0; i < meta.indexCount; ++i) {
            auto indexName = DerivedIndexName(meta, meta.indexes[i]);
            if (!indexName.ok()) {
                return {indexName.code, indexName.affectedRows, indexName.diagnostic};
            }
            // 已存在索引（预检发现的）跳过——MariaDB 无 CREATE INDEX IF NOT
            // EXISTS；不存在才渲染执行。预检快照偶有漏判（information_schema
            // 名称口径与派生名不一致等）时，建索引会撞 ER_DUP_KEYNAME(1061)——
            // 索引已存在即目标态，按幂等成功容忍（与 EnsureDynamicTable 的
            // "create dynamic index" 同一机制，见 ExecuteSimpleLocked 注释）。
            if (existing.value->tableDotIndex.count(std::string{meta.tableId} + "." + *indexName.value) > 0) {
                continue;
            }
            auto createIndex = RenderCreateIndex(meta, meta.indexes[i], *indexName.value, std::string{meta.tableId});
            if (!createIndex.ok()) {
                return {createIndex.code, createIndex.affectedRows, createIndex.diagnostic};
            }
            auto indexed = impl_->ExecuteSimpleLocked(*createIndex.value, "create index", false, true);
            if (!indexed.ok()) {
                return indexed;
            }
            ++impl_->staticDdlExecutions;
        }
    }
    impl_->state = Impl::State::PREPARED;
    return {DbCode::OK, 0, {}};
}

// 即时/事务 INSERT 共用的唯一直通实现（调用前已持锁并完成状态校验）：
// meta 生成固定列 SQL，语句缓存取 stmt，唯一一次 visitRecord 回调内把原 T
// 成员值绑进 MYSQL_BIND（自增主键 0 → NULL 由服务端生成）。
// 即时路径（transaction 为空）在活动事务属于其他线程时有界等待事务槽
// （13 环境跑批中 worker CRUD 与 RaSocketBatchConnect 事务的重叠不再瞬态
// 失败）；同线程保持立即失败（防自锁）。
DbResult<uint64_t> MariaDbEngine::Insert(
    const Ddl::CatalogTableMeta& meta, const TableScope& scope, const void* record, TransactionBinding* transaction)
{
    std::unique_lock<std::mutex> lock(impl_->mutex);
    if (transaction == nullptr) {
        auto waited = impl_->WaitForTransactionSlot(lock);
        if (!waited.ok()) {
            return FailValue<uint64_t>(waited.code, waited.diagnostic);
        }
    }
    auto state = CheckDmlState(transaction);
    if (!state.ok()) {
        return FailValue<uint64_t>(state.code, state.diagnostic);
    }
    return InsertLocked(meta, scope, record);
}

DbResult<uint64_t>
MariaDbEngine::InsertLocked(const Ddl::CatalogTableMeta& meta, const TableScope& scope, const void* record)
{
    if (record == nullptr) {
        return FailValue<uint64_t>(DbCode::INVALID_ARGUMENT, "insert requires a record payload");
    }
    auto physical = impl_->ResolvePhysicalTable(meta, scope, true);
    if (!physical.ok()) {
        return FailValue<uint64_t>(physical.code, physical.diagnostic);
    }
    if (meta.fieldCount == 0) {
        return FailValue<uint64_t>(DbCode::INVALID_ARGUMENT, "insert has no columns");
    }
    // 固定列清单：按声明顺序包含全部字段（自增主键值 0 以 NULL 绑定由后端生成）
    std::string sql;
    sql.reserve(48 + meta.fieldCount * 12);
    sql += "INSERT INTO ";
    const auto tableAppended = AppendQuotedIdentifier(sql, *physical.value);
    if (!tableAppended.ok()) {
        return FailValue<uint64_t>(tableAppended.code, tableAppended.diagnostic);
    }
    const auto columnsAppended = Relational::AppendInsertColumnListAndValues(sql, meta, &AppendQuotedIdentifier);
    if (!columnsAppended.ok()) {
        return FailValue<uint64_t>(columnsAppended.code, columnsAppended.diagnostic);
    }

    BoundParams params(meta.fieldCount);
    // visitRecord 回调：按声明顺序把原 T 成员值立即绑进参数槽（镜像 SQLite 版
    // InsertBindContext::Emit）
    struct InsertBindContext {
        BoundParams* params{nullptr};
        const Ddl::CatalogTableMeta* meta{nullptr};
        static DbResult<void> Emit(void* opaque, uint16_t fieldId, const void* data, std::size_t bytes, bool isNull)
        {
            auto& self = *static_cast<InsertBindContext*>(opaque);
            if (fieldId >= self.meta->fieldCount) {
                return {DbCode::OK, 0, {}}; // 目录外字段忽略（防御；目录按声明顺序全量发射）
            }
            const Ddl::FieldDesc& field = self.meta->fields[fieldId];
            if (isNull) {
                self.params->BindNull(fieldId);
                return {DbCode::OK, 0, {}};
            }
            // 自增主键值 0 视为未指定：绑定 NULL 由 MariaDB 生成（显式非零主键
            // 原样绑定）
            if (field.primaryKey && Ddl::HasProp(field.props, Ddl::Prop::AutoIncrement)
                && IsZeroIntegerValue(field, data, bytes)) {
                self.params->BindNull(fieldId);
                return {DbCode::OK, 0, {}};
            }
            // Derived 字段值来自目录 EmitDerived 的栈上临时 → 复制进槽位 owned；
            // 其余直接借用原 T 成员（record 存活覆盖 execute）
            return self.params->BindEmitted(fieldId, field, data, bytes, Ddl::HasProp(field.props, Ddl::Prop::Derived));
        }
    } context{&params, &meta};
    const auto visited = meta.visitRecord(record, &context, InsertBindContext::Emit);
    if (!visited.ok()) {
        return FailValue<uint64_t>(visited.code, visited.diagnostic);
    }
    auto executed = impl_->ExecutePreparedDmlLocked(sql, params, "insert");
    if (!executed.ok()) {
        return FailValue<uint64_t>(executed.code, executed.diagnostic);
    }
    // 方言差异 #6：生成主键 mysql_stmt_insert_id（无自增主键的表恒为 0）
    const uint64_t generatedId = (*executed.value).second;
    const uint64_t affectedRows = (*executed.value).first;
    return {DbCode::OK, generatedId, affectedRows, {}};
}

// ---- 批量插入 override（T1 真批插：单条参数化多值 INSERT ... RETURNING）----
// 【为什么不是 Connector/C 数组绑定（COM_STMT_BULK_EXECUTE）】数组绑定在
// 3.3.5 客户端只有整批 affected 与首个 insert id（源码批量请求处仍标
// "todo: request to return auto generated ids"）；逐单元 ID 结果需
// MARIADB_OPT_BULK_UNIT_RESULTS（3.3.10 客户端）与 MARIADB_CLIENT_BULK_UNIT_
// RESULTS 服务端能力（MariaDB 11.5.1+），验收环境 10.11 均不具备。用
// "首 ID + i" 或步长推算会破坏并发/间隙/显式 ID 语义。MariaDB 10.5+ 的
// 多值 INSERT ... RETURNING 主键按输入顺序返回真实值（10.11.14 实测），
// 满足按行真实主键、首错行号与回滚合同。
// 快路径资格（全部满足才走单条批命令）：单列自增主键、无 Unique 约束索引、
// 无 catalog 外键、全批记录主键 0/NULL、服务端 10.5+。任一不满足时不报
// "不支持"，直接走基类默认逐行实现——显式/混合 ID、Unique/FK、空指针行、
// visit 失败的错误文本与首错 0 基行号逐字一致，功能不裁剪。
DbResult<void> MariaDbEngine::InsertMany(
    const Ddl::CatalogTableMeta& meta, const TableScope& scope, const void* const* records, std::size_t count,
    uint64_t* generatedIds, TransactionBinding* transaction)
{
    // 入参合同与基类默认实现逐字一致：count=0 平凡成功（不触碰指针/事务）；
    // 指针空或超限拒绝且零副作用
    if (count == 0U) {
        return {DbCode::OK, 0, {}};
    }
    if (records == nullptr || generatedIds == nullptr || transaction == nullptr) {
        return InvalidArgument("insert-many requires records, an id array and an active transaction");
    }
    if (count > kInsertBatchMaxRows) {
        return InvalidArgument("insert-many batch exceeds kInsertBatchMaxRows");
    }
    // 资格预检只读目录与调用方记录（不持锁、不碰连接状态）
    if (InsertManyFastPathEligible(meta, records, count)) {
        std::unique_lock<std::mutex> lock(impl_->mutex);
        if (CheckDmlState(transaction).ok() && impl_->SupportsInsertReturningLocked()) {
            return InsertManyFastPathLocked(meta, scope, records, count, generatedIds);
        }
        // 状态校验失败或旧服务端（<10.5）：释放锁走基类默认逐行（状态错误由
        // 逐行路径在第 0 行复现 " at row 0" 后缀；旧服务端功能不裁剪）
    }
    return IStorageEngine::InsertMany(meta, scope, records, count, generatedIds, transaction);
}

// 资格预检：目录形状（单列自增主键/无 Unique/无 FK，见 EligibleCatalogShape）
// + 逐行记录形态（指针非空、visit 成功、主键 0 或 NULL）。任一不满足返回
// false，由基类逐行路径复现既有错误语义（含 " at row N" 首错行号）。
bool MariaDbEngine::InsertManyFastPathEligible(
    const Ddl::CatalogTableMeta& meta, const void* const* records, std::size_t count)
{
    if (!EligibleCatalogShape(meta)) {
        return false;
    }
    const Ddl::FieldDesc* primaryKey = nullptr;
    for (std::size_t f = 0; f < meta.fieldCount; ++f) {
        if (meta.fields[f].primaryKey) {
            primaryKey = &meta.fields[f];
            break;
        }
    }
    struct EligibilityProbe {
        const Ddl::CatalogTableMeta* meta{nullptr};
        uint16_t primaryKeyField{0};
        bool explicitId{false};
        static DbResult<void> Emit(void* opaque, uint16_t fieldId, const void* data, std::size_t bytes, bool isNull)
        {
            auto& self = *static_cast<EligibilityProbe*>(opaque);
            if (fieldId != self.primaryKeyField || fieldId >= self.meta->fieldCount) {
                return {DbCode::OK, 0, {}};
            }
            if (!isNull && !IsZeroIntegerValue(self.meta->fields[fieldId], data, bytes)) {
                self.explicitId = true; // 显式或混合主键批次 → 回退逐行
            }
            return {DbCode::OK, 0, {}};
        }
    } probe{&meta, static_cast<uint16_t>(primaryKey - meta.fields), false};
    for (std::size_t index = 0; index < count; ++index) {
        if (records[index] == nullptr) {
            return false; // 空指针行 → 基类逐行报 "insert requires a record payload at row N"
        }
        const auto visited = meta.visitRecord(records[index], &probe, EligibilityProbe::Emit);
        if (!visited.ok() || probe.explicitId) {
            return false; // visit 失败/显式 ID → 基类逐行复现
        }
    }
    return true;
}

// 快路径主体（调用前已持锁、已通过 CheckDmlState 与服务端版本检查）：
// 整批一次物理表解析、一次 SQL 构造、一次语句缓存取 stmt、一次
// mysql_stmt_bind_param、一次 mysql_stmt_execute；RETURNING 结果按输入顺序
// 逐行 fetch 真实主键（恰好 count 行），完整排空后返回，同连接立即可执行
// 新语句。execute 阶段整体失败保留真实 errno/sqlstate 并按原子批命令语义
// 标记 at row 0（尚无可消费的逐行结果）；整笔回滚由 Transaction 统一执行。
DbResult<void> MariaDbEngine::InsertManyFastPathLocked(
    const Ddl::CatalogTableMeta& meta, const TableScope& scope, const void* const* records, std::size_t count,
    uint64_t* generatedIds)
{
    const auto startedAt = std::chrono::steady_clock::now();
    // 预执行阶段失败的统一行号标注（与基类默认逐行路径第 0 行失败的
    // " at row 0" 后缀一致；execute 阶段失败见下方原子批命令语义）
    const auto failAtRow0 = [](DbCode code, const std::string& diagnostic) -> DbResult<void> {
        return {code, 0, diagnostic + " at row 0"};
    };
    auto physical = impl_->ResolvePhysicalTable(meta, scope, true);
    if (!physical.ok()) {
        return failAtRow0(physical.code, physical.diagnostic);
    }
    if (meta.fieldCount == 0) {
        return InvalidArgument("insert has no columns at row 0");
    }
    const Ddl::FieldDesc* primaryKey = nullptr;
    for (std::size_t f = 0; f < meta.fieldCount; ++f) {
        if (meta.fields[f].primaryKey) {
            primaryKey = &meta.fields[f];
            break;
        }
    }
    if (primaryKey == nullptr) {
        return InvalidArgument("insert has no primary key at row 0"); // 资格预检已保证，防御
    }
    // SQL：INSERT INTO `t` (`c1`, ...) VALUES (?, ...), (?, ...) ... RETURNING `pk`。
    // 列清单与首组 VALUES 复用公共构造（与单行 INSERT 同一列序 = 声明序），
    // 其余行组按同形追加。
    std::string sql;
    sql.reserve(64 + count * (meta.fieldCount * 4 + 4));
    sql += "INSERT INTO ";
    const auto tableAppended = AppendQuotedIdentifier(sql, *physical.value);
    if (!tableAppended.ok()) {
        return failAtRow0(tableAppended.code, tableAppended.diagnostic);
    }
    const auto columnsAppended = Relational::AppendInsertColumnListAndValues(sql, meta, &AppendQuotedIdentifier);
    if (!columnsAppended.ok()) {
        return failAtRow0(columnsAppended.code, columnsAppended.diagnostic);
    }
    for (std::size_t row = 1; row < count; ++row) {
        sql += ", (";
        for (std::size_t f = 0; f < meta.fieldCount; ++f) {
            sql += f == 0 ? "?" : ", ?";
        }
        sql += ")";
    }
    sql += " RETURNING ";
    const auto keyAppended = AppendQuotedIdentifier(sql, primaryKey->name);
    if (!keyAppended.ok()) {
        return failAtRow0(keyAppended.code, keyAppended.diagnostic);
    }
    // 语句缓存取 stmt。prepare 失败：真实 errno/error/sqlstate 由缓存在 stmt
    // 关闭前捕获；事务路径不复活（事务上下文不能跨连接存活，失败交
    // Transaction 统一回滚），断开族放弃连接（坏连接不得继续当作可用）。
    unsigned int prepareErrno = 0;
    std::string prepareError;
    std::string prepareSqlstate;
    MYSQL_STMT* stmt = impl_->statements->Acquire(sql, &prepareErrno, &prepareError, &prepareSqlstate);
    if (stmt == nullptr) {
        const PrepareErrorCapture prepare
            = impl_->CompletePrepareErrorLocked(prepareErrno, prepareError, prepareSqlstate);
        if (IsConnectionGoneErrno(prepare.errnoValue)) {
            impl_->AbandonConnectionLocked();
        }
        const std::string detail = prepare.message.empty() ? std::string{"no diagnostic captured"} : prepare.message;
        const DbCode code = MapMariadbError(prepare.errnoValue);
        impl_->ReportFailLocked(
            "prepare insert-many", prepare.errnoValue, prepare.sqlstate, code, storage_diag::ElapsedMsSince(startedAt),
            detail);
        return {
            code, 0,
            "mariadb prepare insert-many failed: " + detail + " [mariadb errno " + std::to_string(prepare.errnoValue)
                + " sqlstate " + storage_diag::SqlstateOrDash(prepare.sqlstate) + "] at row 0"};
    }
    // 参数绑定：槽位按 rowIndex * fieldCount + fieldId 排列；普通字段借用原
    // 记录生命周期，Derived 临时值由 BoundParams 槽位持有（与单行 InsertLocked
    // 一致）；资格预检已保证全批主键 0/NULL → 主键槽恒绑 NULL 由服务端生成。
    BoundParams params(count * meta.fieldCount);
    struct BatchBindContext {
        BoundParams* params{nullptr};
        const Ddl::CatalogTableMeta* meta{nullptr};
        std::size_t rowBase{0};
        static DbResult<void> Emit(void* opaque, uint16_t fieldId, const void* data, std::size_t bytes, bool isNull)
        {
            auto& self = *static_cast<BatchBindContext*>(opaque);
            if (fieldId >= self.meta->fieldCount) {
                return {DbCode::OK, 0, {}}; // 目录外字段忽略（防御；目录按声明顺序全量发射）
            }
            const Ddl::FieldDesc& field = self.meta->fields[fieldId];
            const std::size_t slot = self.rowBase + fieldId;
            if (isNull) {
                self.params->BindNull(slot);
                return {DbCode::OK, 0, {}};
            }
            if (field.primaryKey && Ddl::HasProp(field.props, Ddl::Prop::AutoIncrement)) {
                self.params->BindNull(slot); // 未指定主键：服务端生成
                return {DbCode::OK, 0, {}};
            }
            return self.params->BindEmitted(slot, field, data, bytes, Ddl::HasProp(field.props, Ddl::Prop::Derived));
        }
    } context{&params, &meta, 0};
    for (std::size_t row = 0; row < count; ++row) {
        context.rowBase = row * meta.fieldCount;
        const auto visited = meta.visitRecord(records[row], &context, BatchBindContext::Emit);
        if (!visited.ok()) {
            return {visited.code, 0, visited.diagnostic + " at row " + std::to_string(row)};
        }
    }
    if (mysql_stmt_bind_param(stmt, params.Data()) != 0) {
        // bind 失败：真实错误在 stmt 上（客户端侧参数缓冲错误连接级 errno 为 0）
        const unsigned int raw = mysql_stmt_errno(stmt);
        const auto failed = impl_->FailStatementAtLocked(
            raw, mysql_stmt_error(stmt), mysql_stmt_sqlstate(stmt), sql, "mariadb bind insert-many",
            storage_diag::ElapsedMsSince(startedAt));
        return {failed.code, 0, failed.diagnostic + " at row 0"};
    }
    if (mysql_stmt_execute(stmt) != 0) {
        const unsigned int raw = mysql_stmt_errno(stmt);
        const std::string message = mysql_stmt_error(stmt);
        const std::string sqlstate = mysql_stmt_sqlstate(stmt);
        const auto failed = impl_->FailStatementAtLocked(
            raw, message, sqlstate, sql, "mariadb insert-many", storage_diag::ElapsedMsSince(startedAt));
        return {failed.code, 0, failed.diagnostic + " at row 0"};
    }
    ++impl_->batchInsertExecutions;
    // RETURNING 结果：单列 BIGINT UNSIGNED，按输入顺序逐行 fetch 真实主键；
    // 必须恰好 count 行——少行/多行/NULL/截断/fetch 失败都返回失败，不伪造 ID。
    MYSQL_BIND result{};
    uint64_t returnedId = 0;
    my_bool resultNull = 0;
    unsigned long resultLength = 0;
    result.buffer_type = MYSQL_TYPE_LONGLONG;
    result.buffer = &returnedId;
    result.buffer_length = sizeof(returnedId);
    result.is_unsigned = 1;
    result.is_null = &resultNull;
    result.length = &resultLength;
    if (mysql_stmt_bind_result(stmt, &result) != 0) {
        const unsigned int raw = mysql_stmt_errno(stmt);
        const auto failed = impl_->FailStatementAtLocked(
            raw, mysql_stmt_error(stmt), mysql_stmt_sqlstate(stmt), sql, "mariadb bind insert-many result",
            storage_diag::ElapsedMsSince(startedAt));
        return {failed.code, 0, failed.diagnostic + " at row 0"};
    }
    for (std::size_t row = 0; row < count; ++row) {
        const int rc = mysql_stmt_fetch(stmt);
        if (rc != 0) {
            // 含 MYSQL_DATA_TRUNCATED：8 字节 LONGLONG 缓冲上的截断即异常，不得
            // 当作成功行；错误来源不对称（stmt/连接取首个非零 errno）
            const StmtErrorCapture captured
                = CaptureStmtOrConnectionError(stmt, impl_->mysql, "mariadb insert-many returning fetch failed: ");
            const auto failed = impl_->FailStatementAtLocked(
                captured.errnoValue, captured.message, captured.sqlstate, sql, "mariadb insert-many returning",
                storage_diag::ElapsedMsSince(startedAt));
            return {failed.code, 0, failed.diagnostic + " at row 0"};
        }
        if (resultNull != 0) {
            impl_->statements->Discard(sql);
            return MariadbError(
                DbCode::IO_ERROR, "mariadb insert-many returning produced a NULL id at row " + std::to_string(row));
        }
        generatedIds[row] = returnedId;
    }
    // 恰好 count 行：下一次 fetch 必须到达 NO_DATA；多行 = 结果映射异常。两类
    // 结构异常都尽力排空残余行后 Discard（连接不得带着未取尽结果集返回，
    // 否则下一条语句 commands out of sync；不留脏 stmt）
    {
        const int rc = mysql_stmt_fetch(stmt);
        if (rc == 0 || rc == MYSQL_DATA_TRUNCATED) {
            while (true) {
                const int drain = mysql_stmt_fetch(stmt);
                if (drain != 0 && drain != MYSQL_DATA_TRUNCATED) {
                    break;
                }
            }
            impl_->statements->Discard(sql);
            return MariadbError(
                DbCode::IO_ERROR, "mariadb insert-many returning produced more rows than the batch size");
        }
        if (rc != MYSQL_NO_DATA) {
            const StmtErrorCapture captured
                = CaptureStmtOrConnectionError(stmt, impl_->mysql, "mariadb insert-many returning drain failed: ");
            const auto failed = impl_->FailStatementAtLocked(
                captured.errnoValue, captured.message, captured.sqlstate, sql, "mariadb insert-many returning drain",
                storage_diag::ElapsedMsSince(startedAt));
            return {failed.code, 0, failed.diagnostic + " at row 0"};
        }
    }
    // 结果集已完整排空：同连接立即可执行新语句；affectedRows=count（批命令
    // 语义），主键不依赖 mysql_stmt_insert_id 推导
    return {DbCode::OK, count, {}};
}

// 即时 SELECT：投影解析 + 两遍同序遍历（生成 SQL / 绑定原值）→ unbuffered
// 逐行 fetch → materializeRows 直写 ResultSink 指向的最终 T/vector<T> →
// 排空残余行（单对象 sink 只取首行）。
DbResult<void> MariaDbEngine::Select(
    const Ddl::CatalogTableMeta& meta, const TableScope& scope, const ConditionData& where, const QueryOptions& options,
    ResultSink& sink)
{
    std::unique_lock<std::mutex> lock(impl_->mutex);
    // 即时读在活动事务属于其他线程时有界等待事务槽（同线程立即失败防自锁）
    auto waited = impl_->WaitForTransactionSlot(lock);
    if (!waited.ok()) {
        return waited;
    }
    return SelectLocked(meta, scope, where, options, sink, true);
}

// Select 主体（调用前已持锁）。allowRevive=false 表示本调用已是复活重试轮
// （有界重试）：连接断开族错误按原样失败。重试只覆盖 execute 前失败——此时
// 尚未向 sink 输出任何行，整体重发安全；结果集流中途断线不重试（sink 已收
// 部分行，重发会重复）。
DbResult<void> MariaDbEngine::SelectLocked(
    const Ddl::CatalogTableMeta& meta, const TableScope& scope, const ConditionData& where, const QueryOptions& options,
    ResultSink& sink, bool allowRevive)
{
    auto state = CheckOperationState();
    if (!state.ok()) {
        return state;
    }
    if (meta.materializeRows == nullptr) {
        return InvalidArgument("table has no row materializer");
    }
    auto physical = impl_->ResolvePhysicalTable(meta, scope, true);
    if (!physical.ok()) {
        return {physical.code, physical.affectedRows, physical.diagnostic};
    }
    std::vector<std::size_t> fieldIndices;
    std::vector<int32_t> columnOfField;
    // 投影解析（公共实现）：空投影 = 全列（fieldIndices 覆盖全部字段，
    // columnOfField 为空 = 恒等映射）；非空投影去重并建 fieldId → 列映射。
    // WHERE/ORDER BY 引用字段不要求被投影包含——过滤/排序独立生成，由
    // 数据库执行，字段合法性在各自渲染中校验。
    const auto projection = Relational::ResolveProjection(meta, options, fieldIndices, columnOfField);
    if (!projection.ok()) {
        return projection;
    }
    std::size_t placeholder = 1;
    std::string sql;
    sql.reserve(48 + fieldIndices.size() * 12);
    sql += "SELECT ";
    for (std::size_t i = 0; i < fieldIndices.size(); ++i) {
        if (i > 0) {
            sql += ", ";
        }
        const auto appended = AppendQuotedIdentifier(sql, meta.fields[fieldIndices[i]].name);
        if (!appended.ok()) {
            return appended;
        }
    }
    sql += " FROM ";
    const auto tableAppended = AppendQuotedIdentifier(sql, *physical.value);
    if (!tableAppended.ok()) {
        return tableAppended;
    }
    // 【查询计划缺陷修复】严格命中可按区间形状使用的普通四列联合索引的查询
    // 选择该索引：公司 1024p 实测 VirtualMemBlock 区间查找默认计划仅用三列前缀
    // 并叠加 rowid_filter（实际扫描 4118 行、lookup 1048 次、18 页、2.568ms），
    // 指定四列索引后结束地址条件成为 index_condition（9 页、0.464ms）。
    // 投影/谓词/参数绑定顺序/ORDER BY id/LIMIT/首条记录语义全部不变；
    // 未命中匹配形状（OR/缺等值前缀/多余子句/IN/PREFIX/分组/末两列非互异
    // uint64 字段）一律默认计划。
    if (const auto forcedIndex = MatchedIntervalIndexName(meta, where); forcedIndex.has_value()) {
        sql += " FORCE INDEX (";
        const auto indexAppended = AppendQuotedIdentifier(sql, *forcedIndex);
        if (!indexAppended.ok()) {
            return indexAppended;
        }
        sql += ")";
    }
    if (!std::holds_alternative<std::monostate>(where)) {
        sql += " WHERE ";
        const auto appended = AppendWhereText(where, meta, placeholder, sql);
        if (!appended.ok()) {
            return appended;
        }
    }
    const auto sortFailure
        = Relational::AppendOrderBy(meta, options, sql, &AppendMariadbSortColumn, &AppendQuotedIdentifier);
    if (!sortFailure.ok()) {
        return sortFailure;
    }
    if (options.limit.has_value()) {
        // 方言差异 #13：统一 LIMIT ?（语句缓存 + 服务端预备语句对不同绑定值
        // 无重编译，无需 SQLite 的 LIMIT 1 字面量优化）
        sql += " LIMIT ?";
        ++placeholder;
    }
    // prepare 错误处理对齐 DML 直通链（ExecutePreparedDmlLocked）：真实
    // errno/error/sqlstate 由语句缓存在 stmt 关闭前捕获——服务端对预备语句的
    // 拒绝（以及 KILL 后 prepare）只落在 stmt 上，连接级 mysql_errno 为 0，
    // 不能作为错误来源。断线族复活一次（prepare/bind/execute 阶段均未向 sink
    // 输出任何行，整体重发安全；行流开始后一律不重试——重放会重复输出）。
    // 重试额度耗尽或非断线失败：按真实错误上抛；断线族同时放弃连接（坏连接
    // 不得继续当作可用）。
    const auto startedAt = std::chrono::steady_clock::now();
    unsigned int prepareErrno = 0;
    std::string prepareError;
    std::string prepareSqlstate;
    MYSQL_STMT* stmt = impl_->statements->Acquire(sql, &prepareErrno, &prepareError, &prepareSqlstate);
    if (stmt == nullptr) {
        const PrepareErrorCapture prepare
            = impl_->CompletePrepareErrorLocked(prepareErrno, prepareError, prepareSqlstate);
        if (allowRevive && impl_->CanReviveLocked(prepare.errnoValue)) {
            const auto revived = impl_->ReconnectLocked("prepare select");
            if (revived.ok()) {
                return SelectLocked(meta, scope, where, options, sink, false);
            }
            return {revived.code, 0, impl_->ReviveFailureDiagnosticLocked("mariadb prepare select", prepare, revived)};
        }
        if (IsConnectionGoneErrno(prepare.errnoValue)) {
            impl_->AbandonConnectionLocked();
        }
        const std::string detail = prepare.message.empty() ? std::string{"no diagnostic captured"} : prepare.message;
        const DbCode code = MapMariadbError(prepare.errnoValue);
        impl_->ReportFailLocked(
            "prepare select", prepare.errnoValue, prepare.sqlstate, code, storage_diag::ElapsedMsSince(startedAt),
            detail);
        return MariadbError(
            code, "mariadb prepare select failed: " + detail + " [mariadb errno " + std::to_string(prepare.errnoValue)
                      + " sqlstate " + storage_diag::SqlstateOrDash(prepare.sqlstate) + "]");
    }
    BoundParams params(placeholder - 1);
    std::size_t nextIndex = 0;
    if (!std::holds_alternative<std::monostate>(where)) {
        const auto bound = BindWhereValues(where, params, nextIndex);
        if (!bound.ok()) {
            return bound;
        }
    }
    if (options.limit.has_value()) {
        const DbScalar limitValue{*options.limit};
        const auto bound = params.BindScalar(nextIndex, limitValue);
        if (!bound.ok()) {
            return bound;
        }
        ++nextIndex;
    }
    if (params.Size() > 0 && mysql_stmt_bind_param(stmt, params.Data()) != 0) {
        // bind 失败：真实错误在 stmt 上（客户端侧参数缓冲错误连接级 errno 为 0）
        const unsigned int raw = mysql_stmt_errno(stmt);
        const std::string message = mysql_stmt_error(stmt);
        const std::string sqlstate = mysql_stmt_sqlstate(stmt);
        return impl_->FailStatementAtLocked(
            raw, message, sqlstate, sql, "mariadb bind select", storage_diag::ElapsedMsSince(startedAt));
    }
    if (mysql_stmt_execute(stmt) != 0) {
        const unsigned int raw = mysql_stmt_errno(stmt);
        // 连接断开族：复活后整体重试一次（无行输出副作用，安全重发）
        if (allowRevive && impl_->CanReviveLocked(raw)) {
            const auto revived = impl_->ReconnectLocked("execute select");
            if (revived.ok()) {
                return SelectLocked(meta, scope, where, options, sink, false);
            }
            return {revived.code, 0, revived.diagnostic};
        }
        const std::string message = mysql_stmt_error(stmt);
        const std::string sqlstate = mysql_stmt_sqlstate(stmt);
        return impl_->FailStatementAtLocked(
            raw, message, sqlstate, sql, "mariadb select", storage_diag::ElapsedMsSince(startedAt));
    }
    MariadbRowCursor cursor;
    cursor.connection = impl_->mysql; // 取行/取列错误捕获用（显式传入，不读 stmt 内部字段）
    const auto boundColumns = BindResultColumns(cursor, stmt, meta, fieldIndices);
    if (!boundColumns.ok()) {
        // 物化中断的语句不留缓存：先尽力排空残余行，再 Discard（连接死亡时
        // FailStatementLocked 已放弃连接，缓存已清空，Discard 为无操作）
        (void)impl_->DrainCursorLocked(cursor);
        impl_->statements->Discard(sql);
        return boundColumns;
    }
    cursor.columnOfField = std::move(columnOfField);
    const auto materialized = meta.materializeRows(sink, &cursor, MariadbRowCursor::Next, MariadbRowCursor::ReadField);
    if (!materialized.ok()) {
        // 物化失败也要排空/尽力清理：连接可用性优先（错误如实返回原失败）
        (void)impl_->DrainCursorLocked(cursor);
        impl_->statements->Discard(sql);
        return materialized;
    }
    return impl_->DrainCursorLocked(cursor);
}

// 即时 COUNT：数据库端聚合，标量直接返回（单行结果取值后补一次 fetch 完成
// 结果集）。
DbResult<uint64_t>
MariaDbEngine::Count(const Ddl::CatalogTableMeta& meta, const TableScope& scope, const ConditionData& where)
{
    std::unique_lock<std::mutex> lock(impl_->mutex);
    auto waited = impl_->WaitForTransactionSlot(lock);
    if (!waited.ok()) {
        return FailValue<uint64_t>(waited.code, waited.diagnostic);
    }
    return CountLocked(meta, scope, where, true);
}

// Count 主体（调用前已持锁）。allowRevive=false 表示本调用已是复活重试轮。
DbResult<uint64_t> MariaDbEngine::CountLocked(
    const Ddl::CatalogTableMeta& meta, const TableScope& scope, const ConditionData& where, bool allowRevive)
{
    auto state = CheckOperationState();
    if (!state.ok()) {
        return FailValue<uint64_t>(state.code, state.diagnostic);
    }
    auto physical = impl_->ResolvePhysicalTable(meta, scope, true);
    if (!physical.ok()) {
        return FailValue<uint64_t>(physical.code, physical.diagnostic);
    }
    std::size_t placeholder = 1;
    std::string sql;
    sql.reserve(48);
    sql += "SELECT COUNT(*) FROM ";
    const auto tableAppended = AppendQuotedIdentifier(sql, *physical.value);
    if (!tableAppended.ok()) {
        return FailValue<uint64_t>(tableAppended.code, tableAppended.diagnostic);
    }
    if (!std::holds_alternative<std::monostate>(where)) {
        sql += " WHERE ";
        const auto appended = AppendWhereText(where, meta, placeholder, sql);
        if (!appended.ok()) {
            return FailValue<uint64_t>(appended.code, appended.diagnostic);
        }
    }
    // prepare 错误处理对齐 SelectLocked/DML 直通链：真实 errno 由出参捕获，
    // 断线族复活一次（COUNT 无 sink 副作用，prepare 阶段重发安全；首行 fetch
    // 之后不再重试）；非断线失败按真实错误上抛，断线族放弃连接。
    const auto startedAt = std::chrono::steady_clock::now();
    unsigned int prepareErrno = 0;
    std::string prepareError;
    std::string prepareSqlstate;
    MYSQL_STMT* stmt = impl_->statements->Acquire(sql, &prepareErrno, &prepareError, &prepareSqlstate);
    if (stmt == nullptr) {
        const PrepareErrorCapture prepare
            = impl_->CompletePrepareErrorLocked(prepareErrno, prepareError, prepareSqlstate);
        if (allowRevive && impl_->CanReviveLocked(prepare.errnoValue)) {
            const auto revived = impl_->ReconnectLocked("prepare count");
            if (revived.ok()) {
                return CountLocked(meta, scope, where, false);
            }
            return FailValue<uint64_t>(
                revived.code, impl_->ReviveFailureDiagnosticLocked("mariadb prepare count", prepare, revived));
        }
        if (IsConnectionGoneErrno(prepare.errnoValue)) {
            impl_->AbandonConnectionLocked();
        }
        const std::string detail = prepare.message.empty() ? std::string{"no diagnostic captured"} : prepare.message;
        const DbCode code = MapMariadbError(prepare.errnoValue);
        impl_->ReportFailLocked(
            "prepare count", prepare.errnoValue, prepare.sqlstate, code, storage_diag::ElapsedMsSince(startedAt),
            detail);
        return FailValue<uint64_t>(
            code, "mariadb prepare count failed: " + detail + " [mariadb errno " + std::to_string(prepare.errnoValue)
                      + " sqlstate " + storage_diag::SqlstateOrDash(prepare.sqlstate) + "]");
    }
    BoundParams params(placeholder - 1);
    std::size_t nextIndex = 0;
    if (!std::holds_alternative<std::monostate>(where)) {
        const auto bound = BindWhereValues(where, params, nextIndex);
        if (!bound.ok()) {
            return FailValue<uint64_t>(bound.code, bound.diagnostic);
        }
    }
    if (params.Size() > 0 && mysql_stmt_bind_param(stmt, params.Data()) != 0) {
        // bind 失败：真实错误在 stmt 上
        const unsigned int raw = mysql_stmt_errno(stmt);
        const std::string message = mysql_stmt_error(stmt);
        const std::string sqlstate = mysql_stmt_sqlstate(stmt);
        const auto failed = impl_->FailStatementAtLocked(
            raw, message, sqlstate, sql, "mariadb bind count", storage_diag::ElapsedMsSince(startedAt));
        return FailValue<uint64_t>(failed.code, failed.diagnostic);
    }
    if (mysql_stmt_execute(stmt) != 0) {
        const unsigned int raw = mysql_stmt_errno(stmt);
        // 连接断开族：复活后整体重试一次（COUNT 无副作用，安全重发）
        if (allowRevive && impl_->CanReviveLocked(raw)) {
            const auto revived = impl_->ReconnectLocked("execute count");
            if (revived.ok()) {
                return CountLocked(meta, scope, where, false);
            }
            return FailValue<uint64_t>(revived.code, revived.diagnostic);
        }
        const std::string message = mysql_stmt_error(stmt);
        const std::string sqlstate = mysql_stmt_sqlstate(stmt);
        const auto failed = impl_->FailStatementAtLocked(
            raw, message, sqlstate, sql, "mariadb count", storage_diag::ElapsedMsSince(startedAt));
        return FailValue<uint64_t>(failed.code, failed.diagnostic);
    }
    // COUNT(*) 结果列：非负 BIGINT，按无符号 LONGLONG 绑定
    MYSQL_BIND bind{};
    uint64_t counted = 0;
    my_bool isNull = 0;
    unsigned long length = 0;
    bind.buffer_type = MYSQL_TYPE_LONGLONG;
    bind.buffer = &counted;
    bind.buffer_length = sizeof(counted);
    bind.is_unsigned = 1;
    bind.is_null = &isNull;
    bind.length = &length;
    if (mysql_stmt_bind_result(stmt, &bind) != 0) {
        const unsigned int raw = mysql_stmt_errno(stmt);
        const std::string message = mysql_stmt_error(stmt);
        const std::string sqlstate = mysql_stmt_sqlstate(stmt);
        const auto failed = impl_->FailStatementAtLocked(
            raw, message, sqlstate, sql, "mariadb bind count result", storage_diag::ElapsedMsSince(startedAt));
        return FailValue<uint64_t>(failed.code, failed.diagnostic);
    }
    const int rc = mysql_stmt_fetch(stmt);
    if (rc != 0 && rc != MYSQL_DATA_TRUNCATED) {
        // fetch 失败：错误来源不对称（见 CaptureStmtOrConnectionError）；首行已
        // 开始读取，不再重试
        const StmtErrorCapture captured
            = CaptureStmtOrConnectionError(stmt, impl_->mysql, "mariadb count fetch failed: ");
        const auto failed = impl_->FailStatementAtLocked(
            captured.errnoValue, captured.message, captured.sqlstate, sql, "mariadb count fetch",
            storage_diag::ElapsedMsSince(startedAt));
        return FailValue<uint64_t>(failed.code, failed.diagnostic);
    }
    // 补一次 fetch 到 NO_DATA，完成 unbuffered 结果集
    while (true) {
        const int drain = mysql_stmt_fetch(stmt);
        if (drain == MYSQL_NO_DATA) {
            break;
        }
        if (drain != 0 && drain != MYSQL_DATA_TRUNCATED) {
            const StmtErrorCapture captured
                = CaptureStmtOrConnectionError(stmt, impl_->mysql, "mariadb count drain failed: ");
            const auto failed = impl_->FailStatementAtLocked(
                captured.errnoValue, captured.message, captured.sqlstate, sql, "mariadb count drain",
                storage_diag::ElapsedMsSince(startedAt));
            return FailValue<uint64_t>(failed.code, failed.diagnostic);
        }
    }
    return {DbCode::OK, counted, 1, {}};
}

// 即时 CountMatches：单条聚合语句同时返回 where 集合总数与其中匹配 match 的
// 数量（公共 SQL 结构见 Relational::AppendCountMatchesAggregate；占位符顺序
// match 在前、where 在后，绑定遍历同序）。无故障成功调用只执行这一条聚合
// 语句；结果集单行取值后完整排空到 NO_DATA（下一条请求不受 out-of-sync 影响）。
DbResult<CountMatchResult> MariaDbEngine::CountMatches(
    const Ddl::CatalogTableMeta& meta, const TableScope& scope, const ConditionData& where, const ConditionData& match)
{
    std::unique_lock<std::mutex> lock(impl_->mutex);
    auto waited = impl_->WaitForTransactionSlot(lock);
    if (!waited.ok()) {
        return FailValue<CountMatchResult>(waited.code, waited.diagnostic);
    }
    return CountMatchesLocked(meta, scope, where, match, true);
}

// CountMatches 主体（调用前已持锁）。allowRevive=false 表示本调用已是复活
// 重试轮。断线族复活一次（聚合无 sink 副作用，prepare/bind/execute 阶段重发
// 安全；首行 fetch 之后不再重试）。
DbResult<CountMatchResult> MariaDbEngine::CountMatchesLocked(
    const Ddl::CatalogTableMeta& meta, const TableScope& scope, const ConditionData& where, const ConditionData& match,
    bool allowRevive)
{
    auto state = CheckOperationState();
    if (!state.ok()) {
        return FailValue<CountMatchResult>(state.code, state.diagnostic);
    }
    auto physical = impl_->ResolvePhysicalTable(meta, scope, true);
    if (!physical.ok()) {
        return FailValue<CountMatchResult>(physical.code, physical.diagnostic);
    }
    std::size_t placeholder = 1;
    std::string sql;
    sql.reserve(64);
    const auto aggregated = Relational::AppendCountMatchesAggregate(
        sql, meta, *physical.value, where, match, placeholder, &AppendQuotedIdentifier, &AppendWhereText);
    if (!aggregated.ok()) {
        return FailValue<CountMatchResult>(aggregated.code, aggregated.diagnostic);
    }
    const auto startedAt = std::chrono::steady_clock::now();
    unsigned int prepareErrno = 0;
    std::string prepareError;
    std::string prepareSqlstate;
    MYSQL_STMT* stmt = impl_->statements->Acquire(sql, &prepareErrno, &prepareError, &prepareSqlstate);
    if (stmt == nullptr) {
        const PrepareErrorCapture prepare
            = impl_->CompletePrepareErrorLocked(prepareErrno, prepareError, prepareSqlstate);
        if (allowRevive && impl_->CanReviveLocked(prepare.errnoValue)) {
            const auto revived = impl_->ReconnectLocked("prepare count-matches");
            if (revived.ok()) {
                return CountMatchesLocked(meta, scope, where, match, false);
            }
            return FailValue<CountMatchResult>(
                revived.code, impl_->ReviveFailureDiagnosticLocked("mariadb prepare count-matches", prepare, revived));
        }
        if (IsConnectionGoneErrno(prepare.errnoValue)) {
            impl_->AbandonConnectionLocked();
        }
        const std::string detail = prepare.message.empty() ? std::string{"no diagnostic captured"} : prepare.message;
        const DbCode code = MapMariadbError(prepare.errnoValue);
        impl_->ReportFailLocked(
            "prepare count-matches", prepare.errnoValue, prepare.sqlstate, code,
            storage_diag::ElapsedMsSince(startedAt), detail);
        return FailValue<CountMatchResult>(
            code, "mariadb prepare count-matches failed: " + detail + " [mariadb errno "
                      + std::to_string(prepare.errnoValue) + " sqlstate "
                      + storage_diag::SqlstateOrDash(prepare.sqlstate) + "]");
    }
    BoundParams params(placeholder - 1);
    std::size_t nextIndex = 0;
    if (!std::holds_alternative<std::monostate>(match)) {
        const auto matchBound = BindWhereValues(match, params, nextIndex);
        if (!matchBound.ok()) {
            return FailValue<CountMatchResult>(matchBound.code, matchBound.diagnostic);
        }
    }
    if (!std::holds_alternative<std::monostate>(where)) {
        const auto whereBound = BindWhereValues(where, params, nextIndex);
        if (!whereBound.ok()) {
            return FailValue<CountMatchResult>(whereBound.code, whereBound.diagnostic);
        }
    }
    if (params.Size() > 0 && mysql_stmt_bind_param(stmt, params.Data()) != 0) {
        const unsigned int raw = mysql_stmt_errno(stmt);
        const std::string message = mysql_stmt_error(stmt);
        const std::string sqlstate = mysql_stmt_sqlstate(stmt);
        const auto failed = impl_->FailStatementAtLocked(
            raw, message, sqlstate, sql, "mariadb bind count-matches", storage_diag::ElapsedMsSince(startedAt));
        return FailValue<CountMatchResult>(failed.code, failed.diagnostic);
    }
    if (mysql_stmt_execute(stmt) != 0) {
        const unsigned int raw = mysql_stmt_errno(stmt);
        if (allowRevive && impl_->CanReviveLocked(raw)) {
            const auto revived = impl_->ReconnectLocked("execute count-matches");
            if (revived.ok()) {
                return CountMatchesLocked(meta, scope, where, match, false);
            }
            return FailValue<CountMatchResult>(revived.code, revived.diagnostic);
        }
        const std::string message = mysql_stmt_error(stmt);
        const std::string sqlstate = mysql_stmt_sqlstate(stmt);
        const auto failed = impl_->FailStatementAtLocked(
            raw, message, sqlstate, sql, "mariadb count-matches", storage_diag::ElapsedMsSince(startedAt));
        return FailValue<CountMatchResult>(failed.code, failed.diagnostic);
    }
    // 两列 COUNT 结果：非负 BIGINT，按无符号 LONGLONG 绑定
    MYSQL_BIND binds[2] = {};
    uint64_t counts[2] = {0, 0};
    my_bool isNull[2] = {0, 0};
    unsigned long lengths[2] = {0, 0};
    for (int column = 0; column < 2; ++column) {
        binds[column].buffer_type = MYSQL_TYPE_LONGLONG;
        binds[column].buffer = &counts[column];
        binds[column].buffer_length = sizeof(counts[column]);
        binds[column].is_unsigned = 1;
        binds[column].is_null = &isNull[column];
        binds[column].length = &lengths[column];
    }
    if (mysql_stmt_bind_result(stmt, binds) != 0) {
        const unsigned int raw = mysql_stmt_errno(stmt);
        const std::string message = mysql_stmt_error(stmt);
        const std::string sqlstate = mysql_stmt_sqlstate(stmt);
        const auto failed = impl_->FailStatementAtLocked(
            raw, message, sqlstate, sql, "mariadb bind count-matches result", storage_diag::ElapsedMsSince(startedAt));
        return FailValue<CountMatchResult>(failed.code, failed.diagnostic);
    }
    const int rc = mysql_stmt_fetch(stmt);
    if (rc != 0 && rc != MYSQL_DATA_TRUNCATED) {
        const StmtErrorCapture captured
            = CaptureStmtOrConnectionError(stmt, impl_->mysql, "mariadb count-matches fetch failed: ");
        const auto failed = impl_->FailStatementAtLocked(
            captured.errnoValue, captured.message, captured.sqlstate, sql, "mariadb count-matches fetch",
            storage_diag::ElapsedMsSince(startedAt));
        return FailValue<CountMatchResult>(failed.code, failed.diagnostic);
    }
    // 补一次 fetch 到 NO_DATA，完成 unbuffered 结果集
    while (true) {
        const int drain = mysql_stmt_fetch(stmt);
        if (drain == MYSQL_NO_DATA) {
            break;
        }
        if (drain != 0 && drain != MYSQL_DATA_TRUNCATED) {
            const StmtErrorCapture captured
                = CaptureStmtOrConnectionError(stmt, impl_->mysql, "mariadb count-matches drain failed: ");
            const auto failed = impl_->FailStatementAtLocked(
                captured.errnoValue, captured.message, captured.sqlstate, sql, "mariadb count-matches drain",
                storage_diag::ElapsedMsSince(startedAt));
            return FailValue<CountMatchResult>(failed.code, failed.diagnostic);
        }
    }
    CountMatchResult counted;
    counted.total = counts[0];
    counted.matched = counts[1];
    return {DbCode::OK, counted, 1, {}};
}

// ---- 其余方法（M1 落码）----

// 即时/事务 UPDATE 共用的唯一直通实现（调用前已持锁并完成状态校验）：
// UpdateData 生成 SET 文本并绑定，随后绑定同一 ConditionData。
// 参数顺序固定为 UpdateData 在前、ConditionData 在后（与 SQLite 版一致）。
DbResult<uint64_t> MariaDbEngine::Update(
    const Ddl::CatalogTableMeta& meta, const TableScope& scope, const ConditionData& where, const UpdateData& update,
    TransactionBinding* transaction)
{
    std::unique_lock<std::mutex> lock(impl_->mutex);
    if (transaction == nullptr) {
        auto waited = impl_->WaitForTransactionSlot(lock);
        if (!waited.ok()) {
            return FailValue<uint64_t>(waited.code, waited.diagnostic);
        }
    }
    auto state = CheckDmlState(transaction);
    if (!state.ok()) {
        return FailValue<uint64_t>(state.code, state.diagnostic);
    }
    return UpdateLocked(meta, scope, where, update);
}

DbResult<uint64_t> MariaDbEngine::UpdateLocked(
    const Ddl::CatalogTableMeta& meta, const TableScope& scope, const ConditionData& where, const UpdateData& update)
{
    if (update.empty()) {
        return FailValue<uint64_t>(DbCode::INVALID_ARGUMENT, "update requires changes and a condition");
    }
    if (std::holds_alternative<std::monostate>(where)) {
        return FailValue<uint64_t>(DbCode::INVALID_ARGUMENT, "update requires changes and a condition");
    }
    for (const auto& change : update) {
        if (change.fieldId >= meta.fieldCount) {
            return FailValue<uint64_t>(
                DbCode::INVALID_ARGUMENT, "update field is not mapped to a column: " + std::to_string(change.fieldId));
        }
        const Ddl::FieldDesc& field = meta.fields[change.fieldId];
        if (field.primaryKey) {
            return FailValue<uint64_t>(DbCode::INVALID_ARGUMENT, "primary key column cannot be updated");
        }
        if (change.kind == ChangeKind::INCREMENT && !IsIntegerColumn(field.type)
            && field.type != LogicalFieldType::REAL) {
            return FailValue<uint64_t>(DbCode::INVALID_ARGUMENT, "increment requires a numeric column");
        }
        const auto checked = Relational::CheckColumnValue(field, change.value);
        if (!checked.ok()) {
            return FailValue<uint64_t>(checked.code, checked.diagnostic);
        }
    }
    // 无符号列 INCREMENT 的负增量判定（缺陷修复：负 int64 增量此前已由
    // 表达式层折叠为 UINT64_MAX，本层不再可能见到；此处防御性处理直达
    // LogicalChange 的任何 int64 负增量）：SQL 生成减法形态 col - ?，参数在
    // 绑定处按无符号幅值绑定（BIGINT UNSIGNED 域）。INT64_MIN 幅值 = 2^63
    // 经无符号减法 0 - uint64(delta) 取得（模 2^64 回绕是良定义的），禁止对
    // INT64_MIN 取负（UB）。越界（下溢/上溢）由服务端严格模式以 errno 1690
    // 拒绝，不回绕、不截断。有符号列/REAL 保持 col + ? 原生符号语义不变。
    // 【参数生命周期】BindScalar 对字符串/BLOB 借用数据指针（数值入槽复制），
    // 因此除"无符号列 INCREMENT 的数值归一化幅值"外，其余参数必须直接绑定
    // 调用方持有的 change.value（UpdateData 在整个 execute 及断线重试期间
    // 存活）——不得经按值返回 DbScalar 的中间函数（临时副本在绑定表达式
    // 结束时销毁，字符串/BLOB 借用即成悬空指针）。
    const auto unsignedNegativeIncrement = [&meta](const LogicalChange& change) {
        if (change.kind != ChangeKind::INCREMENT || meta.fields[change.fieldId].type != LogicalFieldType::UNSIGNED) {
            return false;
        }
        const auto* delta = std::get_if<int64_t>(&change.value);
        return delta != nullptr && *delta < 0;
    };
    auto physical = impl_->ResolvePhysicalTable(meta, scope, true);
    if (!physical.ok()) {
        return FailValue<uint64_t>(physical.code, physical.diagnostic);
    }
    std::size_t placeholder = 1;
    std::string sql;
    sql.reserve(48 + update.size() * 16);
    sql += "UPDATE ";
    const auto tableAppended = AppendQuotedIdentifier(sql, *physical.value);
    if (!tableAppended.ok()) {
        return FailValue<uint64_t>(tableAppended.code, tableAppended.diagnostic);
    }
    sql += " SET ";
    for (std::size_t i = 0; i < update.size(); ++i) {
        if (i > 0) {
            sql += ", ";
        }
        const LogicalChange& change = update[i];
        const auto column = AppendQuotedIdentifier(sql, meta.fields[change.fieldId].name);
        if (!column.ok()) {
            return FailValue<uint64_t>(column.code, column.diagnostic);
        }
        if (change.kind == ChangeKind::INCREMENT) {
            sql += " = ";
            const auto columnAgain = AppendQuotedIdentifier(sql, meta.fields[change.fieldId].name);
            if (!columnAgain.ok()) {
                return FailValue<uint64_t>(columnAgain.code, columnAgain.diagnostic);
            }
            sql += unsignedNegativeIncrement(change) ? " - ?" : " + ?";
        } else {
            sql += " = ?";
        }
        ++placeholder;
    }
    sql += " WHERE ";
    const auto appended = AppendWhereText(where, meta, placeholder, sql);
    if (!appended.ok()) {
        return FailValue<uint64_t>(appended.code, appended.diagnostic);
    }
    BoundParams params(placeholder - 1);
    std::size_t nextIndex = 0;
    for (const auto& change : update) {
        DbResult<void> bound{DbCode::OK, 0, {}};
        if (change.kind == ChangeKind::INCREMENT && meta.fields[change.fieldId].type == LogicalFieldType::UNSIGNED) {
            // 仅此路径做数值归一化：构造的 DbScalar 只承载整数——BindScalar
            // 对数值复制进内部槽位（无借用），临时在表达式结束后销毁是安全的；
            // 幅值经无符号回绕取得（INT64_MIN 安全，不取负）
            if (const auto* delta = std::get_if<int64_t>(&change.value); delta != nullptr) {
                if (*delta < 0) {
                    bound = params.BindScalar(nextIndex, DbScalar{uint64_t{0} - static_cast<uint64_t>(*delta)});
                } else {
                    bound = params.BindScalar(nextIndex, DbScalar{static_cast<uint64_t>(*delta)});
                }
            } else {
                // uint64 增量原值（数值载荷），直接绑定
                bound = params.BindScalar(nextIndex, change.value);
            }
        } else {
            // 其余参数直接绑定调用方持有的原始 change.value：字符串/BLOB 走
            // 借用指针，UpdateData 在整个 execute 与断线重试期间存活，无悬空
            bound = params.BindScalar(nextIndex, change.value);
        }
        if (!bound.ok()) {
            return FailValue<uint64_t>(bound.code, bound.diagnostic);
        }
        ++nextIndex;
    }
    const auto boundWhere = BindWhereValues(where, params, nextIndex);
    if (!boundWhere.ok()) {
        return FailValue<uint64_t>(boundWhere.code, boundWhere.diagnostic);
    }
    auto executed = impl_->ExecutePreparedDmlLocked(sql, params, "update");
    if (!executed.ok()) {
        return FailValue<uint64_t>(executed.code, executed.diagnostic);
    }
    // 方言差异 #7：影响行数 mysql_stmt_affected_rows
    const uint64_t affectedRows = (*executed.value).first;
    return {DbCode::OK, affectedRows, affectedRows, {}};
}

// 即时/事务 DELETE 共用的唯一直通实现（调用前已持锁并完成状态校验）：
// 条件直推 WHERE，affected_rows 即受影响行数。
DbResult<uint64_t> MariaDbEngine::Delete(
    const Ddl::CatalogTableMeta& meta, const TableScope& scope, const ConditionData& where,
    TransactionBinding* transaction)
{
    std::unique_lock<std::mutex> lock(impl_->mutex);
    if (transaction == nullptr) {
        auto waited = impl_->WaitForTransactionSlot(lock);
        if (!waited.ok()) {
            return FailValue<uint64_t>(waited.code, waited.diagnostic);
        }
    }
    auto state = CheckDmlState(transaction);
    if (!state.ok()) {
        return FailValue<uint64_t>(state.code, state.diagnostic);
    }
    return DeleteLocked(meta, scope, where);
}

DbResult<uint64_t>
MariaDbEngine::DeleteLocked(const Ddl::CatalogTableMeta& meta, const TableScope& scope, const ConditionData& where)
{
    if (std::holds_alternative<std::monostate>(where)) {
        return FailValue<uint64_t>(DbCode::INVALID_ARGUMENT, "delete requires a primary key or conditions");
    }
    auto physical = impl_->ResolvePhysicalTable(meta, scope, true);
    if (!physical.ok()) {
        return FailValue<uint64_t>(physical.code, physical.diagnostic);
    }
    std::size_t placeholder = 1;
    std::string sql;
    sql.reserve(48);
    sql += "DELETE FROM ";
    const auto tableAppended = AppendQuotedIdentifier(sql, *physical.value);
    if (!tableAppended.ok()) {
        return FailValue<uint64_t>(tableAppended.code, tableAppended.diagnostic);
    }
    sql += " WHERE ";
    const auto appended = AppendWhereText(where, meta, placeholder, sql);
    if (!appended.ok()) {
        return FailValue<uint64_t>(appended.code, appended.diagnostic);
    }
    BoundParams params(placeholder - 1);
    std::size_t nextIndex = 0;
    const auto bound = BindWhereValues(where, params, nextIndex);
    if (!bound.ok()) {
        return FailValue<uint64_t>(bound.code, bound.diagnostic);
    }
    auto executed = impl_->ExecutePreparedDmlLocked(sql, params, "delete");
    if (!executed.ok()) {
        return FailValue<uint64_t>(executed.code, executed.diagnostic);
    }
    const uint64_t affectedRows = (*executed.value).first;
    return {DbCode::OK, affectedRows, affectedRows, {}};
}

// 清空单表：镜像 SQLite L1804 语义——活动事务期间一律拒绝（INVALID_STATE），
// 三种模式都在无事务上下文执行：
//   KEEP_SEQUENCE = DELETE FROM（保留自增计数器）；
//   RESET_SEQUENCE = DELETE FROM + ALTER TABLE ... AUTO_INCREMENT = 1
//     （不能直接 TRUNCATE：被外键引用的表 TRUNCATE 报 errno 1701；镜像
//     SQLite 引擎未开启 foreign_keys 的"父表清空恒成功"语义，清数据前临时
//     关闭外键检查，结束后恢复——连接私有，无跨连接影响）；
//   DROP_DYNAMIC_SCOPE = 枚举当前库全部物理表，客户端前缀匹配 <base>_P_
//     逐个 DROP TABLE IF EXISTS（计数值即被销毁物理表数）。
// 动态表残留属已知行为：per-run 实例数据目录用后即弃，见部署 runbook。
DbResult<void> MariaDbEngine::Clear(const Ddl::CatalogTableMeta& meta, const TableScope& scope, ClearMode mode)
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto state = CheckOperationState();
    if (!state.ok()) {
        return state;
    }
    if (mode == ClearMode::DROP_DYNAMIC_SCOPE && meta.scope != Ddl::ScopeKind::DynamicPid) {
        return InvalidArgument("drop-scope clear requires a dynamic table");
    }
    auto physical = impl_->ResolvePhysicalTable(meta, scope, false);
    if (!physical.ok()) {
        return {physical.code, physical.affectedRows, physical.diagnostic};
    }
    if (mode == ClearMode::DROP_DYNAMIC_SCOPE) {
        auto rows = QueryTextRows(
            impl_->mysql, "SELECT TABLE_NAME FROM information_schema.tables "
                          "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_TYPE = 'BASE TABLE'");
        if (!rows.ok()) {
            if (rows.code == DbCode::BACKEND_UNAVAILABLE) {
                impl_->AbandonConnectionLocked();
            }
            return {rows.code, rows.affectedRows, rows.diagnostic};
        }
        const std::string prefix = std::string{meta.tableId} + "_P_";
        uint64_t droppedScopes = 0;
        for (const auto& row : *rows.value) {
            if (row.empty() || row[0].compare(0, prefix.size(), prefix) != 0) {
                continue;
            }
            auto name = QuoteIdentifier(row[0]);
            if (!name.ok()) {
                return {name.code, name.affectedRows, name.diagnostic};
            }
            auto dropped
                = impl_->ExecuteSimpleLocked("DROP TABLE IF EXISTS " + *name.value, "drop dynamic scope table");
            if (!dropped.ok()) {
                return dropped;
            }
            impl_->ensuredDynamicTables.erase(row[0]);
            ++droppedScopes;
        }
        return {DbCode::OK, droppedScopes, {}};
    }
    auto table = QuoteIdentifier(*physical.value);
    if (!table.ok()) {
        return {table.code, table.affectedRows, table.diagnostic};
    }
    // KEEP_SEQUENCE / RESET_SEQUENCE 共用 FK 安全的清数据路径（见上）
    auto fkOff = impl_->ExecuteSimpleLocked("SET FOREIGN_KEY_CHECKS = 0", "disable foreign key checks");
    if (!fkOff.ok()) {
        return fkOff;
    }
    auto cleared = impl_->ExecuteSimpleLocked("DELETE FROM " + *table.value, "clear table");
    if (!cleared.ok()) {
        (void)impl_->ExecuteSimpleLocked("SET FOREIGN_KEY_CHECKS = 1", "enable foreign key checks");
        return cleared;
    }
    if (mode == ClearMode::KEEP_SEQUENCE) {
        (void)impl_->ExecuteSimpleLocked("SET FOREIGN_KEY_CHECKS = 1", "enable foreign key checks");
        return cleared;
    }
    auto reset
        = impl_->ExecuteSimpleLocked("ALTER TABLE " + *table.value + " AUTO_INCREMENT = 1", "reset auto increment");
    (void)impl_->ExecuteSimpleLocked("SET FOREIGN_KEY_CHECKS = 1", "enable foreign key checks");
    return reset;
}

// 列出当前库的用户表（information_schema；对齐 sqlite_master 返回合同，
// 只列 BASE TABLE，按表名排序）。
DbResult<std::vector<std::string>> MariaDbEngine::ListTables()
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto state = CheckOperationState(/*allowBackupOnly*/ false);
    if (!state.ok()) {
        return FailValue<std::vector<std::string>>(state.code, state.diagnostic);
    }
    auto rows = QueryTextRows(
        impl_->mysql, "SELECT TABLE_NAME FROM information_schema.tables "
                      "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_TYPE = 'BASE TABLE' ORDER BY TABLE_NAME");
    if (!rows.ok()) {
        if (rows.code == DbCode::BACKEND_UNAVAILABLE) {
            impl_->AbandonConnectionLocked();
        }
        return FailValue<std::vector<std::string>>(rows.code, rows.diagnostic);
    }
    std::vector<std::string> tables;
    tables.reserve(rows.value->size());
    for (const auto& row : *rows.value) {
        if (!row.empty()) {
            tables.push_back(row[0]);
        }
    }
    return {DbCode::OK, std::optional<std::vector<std::string>>{std::move(tables)}, tables.size(), {}};
}

// ---- 物理事务边界（直达模型：DML 已即时执行，本层只处理 START TRANSACTION/
// COMMIT/ROLLBACK；方言差异 #10）----

// 开启一个物理事务：START TRANSACTION 后同一 binding 的 DML 直达执行。
// 只做连接状态与唯一活动事务校验，不解析表、不创建计划、不保存业务数据。
// 单连接上的第二笔事务在其他线程的活动事务期间串行等待（而非瞬态失败）；
// 同线程保持立即失败（防自锁）。
DbResult<void> MariaDbEngine::Begin(TransactionBinding& transaction)
{
    std::unique_lock<std::mutex> lock(impl_->mutex);
    if (impl_->state == Impl::State::NOT_OPEN || impl_->state == Impl::State::CLOSED) {
        return InvalidState("engine is not open");
    }
    if (impl_->state == Impl::State::OPENED) {
        return InvalidState("engine is not prepared");
    }
    if (impl_->state == Impl::State::TRANSACTION_BUILDING) {
        auto waited = impl_->WaitForTransactionSlot(lock);
        if (!waited.ok()) {
            return waited;
        }
        // 被唤醒后重新完整校验状态（不能假定仍为 PREPARED：可能已被关闭）
        if (impl_->state == Impl::State::NOT_OPEN || impl_->state == Impl::State::CLOSED) {
            return InvalidState("engine is not open");
        }
        if (impl_->state == Impl::State::OPENED) {
            return InvalidState("engine is not prepared");
        }
    }
    if (impl_->state != Impl::State::PREPARED) {
        return InvalidState("another transaction is active on this route");
    }
    if (impl_->mode == StorageOpenMode::BACKUP_ONLY) {
        return InvalidState("backup-only engine does not accept transactions");
    }
    if (impl_->mysql == nullptr) {
        return InvalidState("mariadb connection is not open");
    }
    auto begun = impl_->ExecuteSimpleLocked("START TRANSACTION", "begin");
    if (!begun.ok()) {
        // BEGIN 失败不改变连接状态（仍为 PREPARED），活动槽保持为空
        return begun;
    }
    impl_->activeBinding = &transaction;
    impl_->transactionAborted = false;
    impl_->transactionOwner = std::this_thread::get_id();
    impl_->state = Impl::State::TRANSACTION_BUILDING;
    return {DbCode::OK, 0, {}};
}

// 提交当前物理事务：只执行 COMMIT，不解释/不重放任何 DML（结果由 Transaction
// 按执行顺序记录）。死锁中止的事务必须失败返回（InnoDB 已回滚受害事务，
// 不吞错误）；提交失败尝试 ROLLBACK；回滚也失败时放弃连接（不得复用不确定
// 状态的连接，对齐 SQLite 故障合同）。
DbResult<AtomicResult> MariaDbEngine::Commit(TransactionBinding& transaction)
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->activeBinding != &transaction || impl_->state != Impl::State::TRANSACTION_BUILDING) {
        return FailValue<AtomicResult>(DbCode::INVALID_STATE, "no matching transaction is active on this route");
    }
    if (impl_->transactionAborted) {
        // 死锁已让 InnoDB 回滚整个事务：该 binding 的 Commit 明确失败（不吞
        // 错误）。不在此清理活动槽——Rollback 是该 binding 的合法恢复入口
        // （服务端事务已回滚，再发 ROLLBACK 是无害 no-op，由它统一清连接
        // 状态回到 PREPARED）；调用方若不再 Rollback，引擎保持拒绝一切新
        // 事务/普通 DML 的防御状态。
        return FailValue<AtomicResult>(
            DbCode::INVALID_STATE,
            "transaction was aborted by deadlock: InnoDB rolled back the victim, commit is rejected");
    }
    auto committed = impl_->ExecuteSimpleLocked("COMMIT", "commit");
    if (!committed.ok()) {
        if (committed.code == DbCode::BACKEND_UNAVAILABLE) {
            // 连接级失败已放弃连接（活动槽/状态已清理）
            return FailValue<AtomicResult>(committed.code, committed.diagnostic);
        }
        auto rolled = RollbackPhysicalLocked();
        impl_->activeBinding = nullptr;
        impl_->transactionAborted = false;
        impl_->state = Impl::State::PREPARED;
        impl_->NotifyTransactionSlotReleasedLocked(); // 事务槽已释放，唤醒等待者
        if (!rolled.ok()) {
            impl_->AbandonConnectionLocked();
            return FailValue<AtomicResult>(
                committed.code, committed.diagnostic + "; rollback after failed commit also failed: "
                                    + rolled.diagnostic + " [connection abandoned]");
        }
        return FailValue<AtomicResult>(committed.code, committed.diagnostic);
    }
    impl_->activeBinding = nullptr;
    impl_->transactionAborted = false;
    impl_->state = Impl::State::PREPARED;
    impl_->NotifyTransactionSlotReleasedLocked(); // 事务槽已释放，唤醒等待者
    return {DbCode::OK, AtomicResult{}, committed.affectedRows, {}};
}

// 回滚当前物理事务：执行 ROLLBACK 并清理活动槽（显式回滚、析构兜底与
// Transaction 在 DML 失败后的统一回滚共用本入口）。回滚失败放弃连接，
// 不得复用不确定状态的连接（对齐 SQLite 故障合同）。
DbResult<void> MariaDbEngine::Rollback(TransactionBinding& transaction)
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->activeBinding != &transaction || impl_->state != Impl::State::TRANSACTION_BUILDING) {
        return InvalidState("no matching transaction is active on this route");
    }
    auto rolled = RollbackPhysicalLocked();
    if (!rolled.ok()) {
        if (rolled.code == DbCode::BACKEND_UNAVAILABLE) {
            // 连接级失败已放弃连接（活动槽/状态已清理）
            return rolled;
        }
        impl_->AbandonConnectionLocked();
        return {rolled.code, 0, rolled.diagnostic + " [connection abandoned]"};
    }
    impl_->activeBinding = nullptr;
    impl_->transactionAborted = false;
    impl_->state = Impl::State::PREPARED;
    impl_->NotifyTransactionSlotReleasedLocked(); // 事务槽已释放，唤醒等待者
    return {DbCode::OK, 0, {}};
}

DbResult<void> MariaDbEngine::Close()
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->state == Impl::State::NOT_OPEN) {
        return InvalidState("engine is not open");
    }
    if (impl_->state == Impl::State::CLOSED) {
        return InvalidState("engine is already closed");
    }
    if (impl_->state == Impl::State::TRANSACTION_BUILDING) {
        // 丢弃未提交事务（不隐式提交；失败沿用连接死亡处置）
        auto rolled = RollbackPhysicalLocked();
        if (!rolled.ok() && rolled.code != DbCode::BACKEND_UNAVAILABLE) {
            impl_->AbandonConnectionLocked();
            impl_->state = Impl::State::CLOSED;
            return {rolled.code, 0, rolled.diagnostic + " [connection abandoned]"};
        }
    }
    return CloseLocked();
}

// host 启动专用（envInit 调用）：临时连 server 级（database=nullptr，不选库）
// 对本引擎物理库（hcclvm_<namespace>_<databaseId>）执行 DROP DATABASE IF EXISTS，
// 把上一轮运行的全部持久化数据一次清空——等价 SQLite 的"删库文件"，为外部
// 后端提供同样的干净起点。仅 owner（host 的 `hccl-vm start`）在建立业务会话前
// 调用一次；rank/checker 以 READ_EXISTING 连接既有库，绝不触发本路径。
// 使用独立短连接（不复用/不修改 impl_->mysql 与引擎状态），故可在尚未 Open
// 的 transient 引擎实例上安全调用；DROP 后由后续业务会话 Open 重新建库建表。
DbResult<void> MariaDbEngine::Wipe(std::string_view dataNamespace)
{
    auto endpoint = ParseMariadbUri(impl_->profile.uri);
    if (!endpoint.ok()) {
        return {endpoint.code, 0, endpoint.diagnostic};
    }
    if (impl_->profile.uri.empty() || endpoint.value->host.empty()) {
        return MariadbError(DbCode::INVALID_ARGUMENT, "mariadb endpoint uri is empty or malformed");
    }
    // 程序化构造的 profile 越界在建连前拒绝（与 Open 同一双层校验）
    auto profileTimeouts = ValidateMariadbProfileTimeouts(impl_->profile);
    if (!profileTimeouts.ok()) {
        return {profileTimeouts.code, 0, profileTimeouts.diagnostic};
    }
    const std::string physicalName = DeriveDatabaseName(
        dataNamespace, impl_->catalog == nullptr ? std::string_view{} : impl_->catalog->databaseId);
    auto quotedName = QuoteIdentifier(physicalName);
    if (!quotedName.ok()) {
        return {quotedName.code, 0, quotedName.diagnostic};
    }
    EnsureMariadbLibraryInit();

    MYSQL* mysql = mysql_init(nullptr);
    if (mysql == nullptr) {
        return MariadbError(DbCode::RESOURCE_EXHAUSTED, "mariadb wipe: mysql_init failed (out of memory)");
    }
    // 与 Open/ReconnectLocked 同一共享选项函数：charset、三类超时与
    // MYSQL_OPT_COMPRESS 使用同一 profile 值（此前 Wipe 未施加压缩，与
    // Open 会话不一致——统一后 DROP DATABASE 的短连接同样遵守部署配置）
    auto options = ApplyMariadbConnectionOptions(mysql, impl_->profile);
    if (!options.ok()) {
        (void)mysql_close(mysql);
        return MariadbError(options.code, "mariadb wipe: " + options.diagnostic);
    }
    // server 级连接（不选库）：DROP DATABASE 必须在目标库之外发起
    if (mysql_real_connect(
            mysql, endpoint.value->host.c_str(), endpoint.value->user.c_str(), endpoint.value->password.c_str(),
            nullptr, endpoint.value->port, nullptr, 0)
        == nullptr) {
        const unsigned int raw = mysql_errno(mysql);
        const std::string message = mysql_error(mysql);
        (void)mysql_close(mysql);
        return MariadbError(
            MapMariadbError(raw), "mariadb wipe: failed to connect to server " + SanitizedEndpoint(*endpoint.value)
                                      + ": " + message + " [mariadb errno " + std::to_string(raw) + "]");
    }
    const std::string dropSql = "DROP DATABASE IF EXISTS " + *quotedName;
    if (mysql_real_query(mysql, dropSql.c_str(), static_cast<unsigned long>(dropSql.size())) != 0) {
        const unsigned int raw = mysql_errno(mysql);
        const std::string message = mysql_error(mysql);
        (void)mysql_close(mysql);
        return MariadbError(
            MapMariadbError(raw), "mariadb wipe: DROP DATABASE `" + physicalName + "` failed: " + message
                                      + " [mariadb errno " + std::to_string(raw) + "]");
    }
    (void)mysql_close(mysql);
    return {DbCode::OK, 0, {}};
}

// 关系型备份（方言差异 #8）：经引擎自身连接 START TRANSACTION WITH CONSISTENT
// SNAPSHOT（活动事务存在则拒绝）→ 逐表 SHOW CREATE TABLE + 全表行读 → 生成
// 可回放 .sql 文本（mariadb < dump.sql）→ ROLLBACK 结束快照。
// 消费者结论（M1 前置检查）：唯一调用链 BackupOperationData → 落盘
// data/hccl_vm_data_backup_*.db，消费者不解析产物内容、不自动回放；恢复是
// 运维动作（mariadb 客户端回放）。备选方案 mariadb-dump --single-transaction
// 外部进程未采用：会引入部署依赖与进程编排，且无法经引擎连接保证快照一致性。
// 原子落盘：先写 destinationPath.tmp，成功后 rename。
DbResult<DatabaseBackupArtifact> MariaDbEngine::BackupRelational(const EngineBackupRequest& request)
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->state == Impl::State::NOT_OPEN || impl_->state == Impl::State::CLOSED) {
        return FailValue<DatabaseBackupArtifact>(DbCode::INVALID_STATE, "engine is not open");
    }
    if (impl_->state == Impl::State::OPENED) {
        return FailValue<DatabaseBackupArtifact>(DbCode::INVALID_STATE, "engine is not prepared");
    }
    if (impl_->state == Impl::State::TRANSACTION_BUILDING) {
        return FailValue<DatabaseBackupArtifact>(DbCode::INVALID_STATE, "a transaction is building on this route");
    }
    if (request.destinationPath.empty()) {
        return FailValue<DatabaseBackupArtifact>(DbCode::INVALID_ARGUMENT, "backup destination path is empty");
    }
    if (impl_->mysql == nullptr) {
        return FailValue<DatabaseBackupArtifact>(DbCode::INVALID_STATE, "mariadb connection is not open");
    }
    const std::string temporaryPath = request.destinationPath + ".tmp";
    std::ofstream output(temporaryPath, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!output.good()) {
        return FailValue<DatabaseBackupArtifact>(
            DbCode::IO_ERROR, "failed to open mariadb backup destination: " + temporaryPath);
    }
    const auto finishWith = [&](DbResult<DatabaseBackupArtifact> result) -> DbResult<DatabaseBackupArtifact> {
        output.close();
        (void)RollbackPhysicalLocked(); // 结束一致性快照（幂等；失败不掩盖原结果）
        if (!result.ok()) {
            std::remove(temporaryPath.c_str()); // 失败不留半截文件
            if (result.code == DbCode::BACKEND_UNAVAILABLE) {
                impl_->AbandonConnectionLocked();
            }
        }
        return result;
    };
    // 一致性快照：START TRANSACTION WITH CONSISTENT SNAPSHOT（当前连接无活动
    // 事务——上方已拒绝 TRANSACTION_BUILDING）
    auto begun = impl_->ExecuteSimpleLocked("START TRANSACTION WITH CONSISTENT SNAPSHOT", "backup snapshot");
    if (!begun.ok()) {
        return finishWith(FailValue<DatabaseBackupArtifact>(begun.code, begun.diagnostic));
    }
    auto databaseName = QuoteIdentifier(impl_->database);
    if (!databaseName.ok()) {
        return finishWith(FailValue<DatabaseBackupArtifact>(databaseName.code, databaseName.diagnostic));
    }
    output << "-- MariaDB logical dump (hccl-vm MariaDbEngine::BackupRelational)\n"
              "-- replay with: mariadb < <this file>\n"
              "SET NAMES utf8mb4 COLLATE utf8mb4_bin;\n"
              "SET FOREIGN_KEY_CHECKS = 0;\n"
              "SET UNIQUE_CHECKS = 0;\n"
              "CREATE DATABASE IF NOT EXISTS "
           << *databaseName.value
           << " DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin;\n"
              "USE "
           << *databaseName.value << ";\n";
    auto tables = QueryTextRows(
        impl_->mysql, "SELECT TABLE_NAME FROM information_schema.tables "
                      "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_TYPE = 'BASE TABLE' ORDER BY TABLE_NAME");
    if (!tables.ok()) {
        return finishWith(FailValue<DatabaseBackupArtifact>(tables.code, tables.diagnostic));
    }
    for (const auto& tableRow : *tables.value) {
        if (tableRow.empty()) {
            continue;
        }
        auto tableName = QuoteIdentifier(tableRow[0]);
        if (!tableName.ok()) {
            return finishWith(FailValue<DatabaseBackupArtifact>(tableName.code, tableName.diagnostic));
        }
        // 建表语句：SHOW CREATE TABLE 第二列即完整 DDL
        auto createRows = QueryTextRows(impl_->mysql, "SHOW CREATE TABLE " + *tableName.value);
        if (!createRows.ok()) {
            return finishWith(FailValue<DatabaseBackupArtifact>(createRows.code, createRows.diagnostic));
        }
        if (createRows.value->empty() || (*createRows.value)[0].size() < 2U) {
            return finishWith(FailValue<DatabaseBackupArtifact>(
                DbCode::IO_ERROR, "mariadb show create table returned no ddl: " + tableRow[0]));
        }
        output << "DROP TABLE IF EXISTS " << *tableName.value << ";\n" << (*createRows.value)[0][1] << ";\n";
        const auto dumped = DumpTableRowsLocked(output, *tableName.value);
        if (!dumped.ok()) {
            return finishWith(FailValue<DatabaseBackupArtifact>(dumped.code, dumped.diagnostic));
        }
    }
    output << "SET FOREIGN_KEY_CHECKS = 1;\n"
              "SET UNIQUE_CHECKS = 1;\n";
    output.flush();
    if (!output.good()) {
        return finishWith(
            FailValue<DatabaseBackupArtifact>(DbCode::IO_ERROR, "mariadb backup write failed: " + temporaryPath));
    }
    output.close();
    if (std::rename(temporaryPath.c_str(), request.destinationPath.c_str()) != 0) {
        std::remove(temporaryPath.c_str());
        return FailValue<DatabaseBackupArtifact>(
            DbCode::IO_ERROR, "mariadb backup rename failed: " + temporaryPath + " -> " + request.destinationPath);
    }
    (void)RollbackPhysicalLocked();
    return {DbCode::OK, DatabaseBackupArtifact{request.destinationPath}, 0, {}};
}

// 单表数据行 dump：文本协议 unbuffered 逐行读（use_result；禁止 store_result
// 全量缓冲的合同同样覆盖备份路径）。数值列原样、文本列转义引号、二进制列
// hex 字面量；按 64 行/4MB 上限切分 INSERT 语句（max_allowed_packet 安全界）。
DbResult<void> MariaDbEngine::DumpTableRowsLocked(std::ofstream& output, const std::string& tableName)
{
    const std::string sql = "SELECT * FROM " + tableName;
    if (mysql_real_query(impl_->mysql, sql.c_str(), static_cast<unsigned long>(sql.size())) != 0) {
        const auto failed = impl_->FailStatementLocked(std::string{}, "mariadb backup select");
        return failed;
    }
    MYSQL_RES* result = mysql_use_result(impl_->mysql);
    if (result == nullptr) {
        const auto failed = impl_->FailStatementLocked(std::string{}, "mariadb backup select result");
        return failed;
    }
    const unsigned int fieldCount = mysql_num_fields(result);
    const MYSQL_FIELD* fields = mysql_fetch_fields(result);
    if (fields == nullptr || fieldCount == 0) {
        mysql_free_result(result);
        return MariadbError(DbCode::IO_ERROR, "mariadb backup result has no fields: " + tableName);
    }
    std::string statement;
    std::size_t pendingRows = 0;
    uint64_t totalRows = 0;
    const auto flush = [&]() {
        if (pendingRows > 0) {
            statement += ";\n";
            output << statement;
            statement.clear();
            pendingRows = 0;
        }
    };
    while (MYSQL_ROW row = mysql_fetch_row(result)) {
        const unsigned long* lengths = mysql_fetch_lengths(result);
        if (statement.empty()) {
            statement = "INSERT INTO " + tableName + " VALUES ";
        } else {
            statement += ", ";
        }
        statement += "(";
        for (unsigned int i = 0; i < fieldCount; ++i) {
            if (i > 0) {
                statement += ", ";
            }
            if (row[i] == nullptr) {
                statement += "NULL";
                continue;
            }
            AppendDumpLiteral(statement, &fields[i], row[i], lengths[i]);
        }
        statement += ")";
        ++pendingRows;
        ++totalRows;
        if (pendingRows >= 64U || statement.size() > 4U * 1024U * 1024U) {
            flush();
        }
    }
    const unsigned int error = mysql_errno(impl_->mysql);
    if (error != 0) {
        mysql_free_result(result);
        return impl_->FailStatementLocked(std::string{}, "mariadb backup fetch");
    }
    mysql_free_result(result);
    flush();
    return {DbCode::OK, 0, {}};
}

// ROLLBACK 物理边界。MariaDB 下 ROLLBACK 对无活动事务幂等成功（服务端仅记
// 警告），无需 SQLite 版的 "no transaction is active" 诊断特判；动态表 DDL
// 不可回滚（隐式提交），ensuredDynamicTables 不随回滚失效——与 SQLite 的
// 关键语义差异，见 Impl 内注释。
DbResult<void> MariaDbEngine::RollbackPhysicalLocked()
{
    if (impl_->mysql == nullptr) {
        return InvalidState("mariadb connection is not open");
    }
    return impl_->ExecuteSimpleLocked("ROLLBACK", "rollback");
}

// 显式关闭：先丢弃未提交事务（MariaDB 连接关闭时服务端自动回滚未提交事务，
// 与 SQLite 显式 ROLLBACK 等价），再释放语句缓存与连接；Close 为终态。
DbResult<void> MariaDbEngine::CloseLocked()
{
    if (impl_->mysql == nullptr) {
        impl_->state = Impl::State::CLOSED;
        impl_->NotifyTransactionSlotReleasedLocked(); // 唤醒等待者按 CLOSED 有限返回
        return {DbCode::OK, 0, {}};
    }
    // 语句先于连接释放（MYSQL_STMT 属于连接）
    if (impl_->statements != nullptr) {
        impl_->statements->ClearAll();
    }
    (void)mysql_close(impl_->mysql);
    impl_->mysql = nullptr;
    impl_->statements.reset();
    impl_->ensuredDynamicTables.clear();
    impl_->activeBinding = nullptr;
    impl_->transactionAborted = false;
    impl_->state = Impl::State::CLOSED;
    impl_->NotifyTransactionSlotReleasedLocked(); // 唤醒等待者按 CLOSED 有限返回
    return {DbCode::OK, 0, {}};
}

// 即时操作公共状态检查（持锁路径内联使用）。
DbResult<void> MariaDbEngine::CheckOperationState(bool allowBackupOnly)
{
    if (impl_->state == Impl::State::NOT_OPEN || impl_->state == Impl::State::CLOSED) {
        return InvalidState("engine is not open");
    }
    if (impl_->state == Impl::State::OPENED) {
        return InvalidState("engine is not prepared");
    }
    if (impl_->state == Impl::State::TRANSACTION_BUILDING) {
        return InvalidState("a transaction is building on this route");
    }
    if (!allowBackupOnly && impl_->mode == StorageOpenMode::BACKUP_ONLY) {
        return InvalidState("backup-only engine does not accept operations");
    }
    if (impl_->mysql == nullptr) {
        return MariadbError(DbCode::INVALID_STATE, "mariadb connection is not open");
    }
    return {DbCode::OK, 0, {}};
}

// DML 状态校验：transaction 为空要求 PREPARED（普通 DML 不得混入活动事务的
// 连接）；非空要求已 Begin 且 binding 匹配且事务未被死锁中止。DML 只执行并
// 返回错误，失败回滚由 Transaction 统一调用 Rollback 完成。
DbResult<void> MariaDbEngine::CheckDmlState(TransactionBinding* transaction)
{
    if (transaction == nullptr) {
        return CheckOperationState();
    }
    if (impl_->state == Impl::State::NOT_OPEN || impl_->state == Impl::State::CLOSED) {
        return InvalidState("engine is not open");
    }
    if (impl_->state == Impl::State::OPENED) {
        return InvalidState("engine is not prepared");
    }
    if (impl_->activeBinding != transaction || impl_->state != Impl::State::TRANSACTION_BUILDING) {
        return InvalidState("no matching transaction is active on this route");
    }
    if (impl_->transactionAborted) {
        return InvalidState("transaction was aborted by deadlock on this route");
    }
    if (impl_->mysql == nullptr) {
        return MariadbError(DbCode::INVALID_STATE, "mariadb connection is not open");
    }
    return {DbCode::OK, 0, {}};
}

// 语句缓存观察快照（仅供单元测试；镜像 SQLite 的 SchemaDdlStatistics 合同）
MariaDbEngine::StatementCacheSnapshot MariaDbEngine::StatementCacheStatistics() const
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    StatementCacheSnapshot snapshot;
    if (impl_->statements != nullptr) {
        snapshot.hits = impl_->statements->Hits();
        snapshot.misses = impl_->statements->Misses();
        snapshot.size = impl_->statements->Size();
    }
    return snapshot;
}

// ---- 测试可见性入口（仅测试 peer 经 friend 调用；产品代码不得依赖）----

// 事务等待上限缩短（持锁写入私有上限）
void MariaDbEngine::SetTransactionWaitLimitForTest(std::chrono::milliseconds limit)
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->transactionWaitLimit = limit;
}

// 批插快路径（多值 INSERT ... RETURNING）执行计数
uint64_t MariaDbEngine::BatchInsertExecutionsForTest() const
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->batchInsertExecutions;
}

// 服务端版本 seam（非 0 时优先于 mysql_get_server_version）
void MariaDbEngine::SetServerVersionForTest(unsigned long version)
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->serverVersionForTest = version;
}
} // namespace HcclSim::Storage
