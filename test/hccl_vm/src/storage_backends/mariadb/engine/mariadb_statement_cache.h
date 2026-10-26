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

/**
 * @file mariadb_statement_cache.h
 * @brief SQL 文本 → MYSQL_STMT* 的进程内 LRU 语句缓存（MariaDB 后端私有）。
 *
 * 【设计边界】
 *   - 只做"同一 SQL 文本复用已预备语句"这一件事：命中省去 prepare 的两次
 *     网络往返；不解析 SQL、不感知表结构、不做参数缓存；
 *   - 上限 128 条 LRU（超出逐出最久未用语句并 mysql_stmt_close）；
 *   - 线程安全由引擎的单一连接 mutex 提供（本类自身不加锁——全部调用都
 *     在 MariaDbEngine 持锁路径内发生）；
 *   - 执行失败的语句由调用方 Discard（不留脏 stmt）；连接关闭/放弃前
 *     ClearAll（MYSQL_STMT 属于连接，必须先于 mysql_close 释放）。
 */
#ifndef HCCL_VM_STORAGE_ENGINE_MARIADB_MARIADB_STATEMENT_CACHE_H
#define HCCL_VM_STORAGE_ENGINE_MARIADB_MARIADB_STATEMENT_CACHE_H

#include <cstddef>
#include <cstdint>
#include <list>
#include <mysql.h>
#include <string>
#include <unordered_map>
#include <utility>

namespace HcclSim::Storage {
/// MariaDB 语句缓存（SQL 文本 → 已预备 MYSQL_STMT；LRU 上限 128）
class MariadbStatementCache final {
public:
    /// 同一连接上缓存的语句数上限（超出按 LRU 逐出；语句键是完整 SQL 文本，
    /// 动态 PID 表按 <表>_P_<pid> 天然产生不同键，由 LRU 自然约束总量）
    static constexpr std::size_t kDefaultCapacity = 128;

    /// mysql 为该缓存所属的连接（本类不持有连接，只借用指针预备/关闭语句）
    explicit MariadbStatementCache(MYSQL* mysql, std::size_t capacity = kDefaultCapacity)
        : mysql_(mysql),
          capacity_(capacity == 0 ? kDefaultCapacity : capacity)
    {}

    /// 关闭全部仍持有的语句（连接关闭/引擎析构路径调用）
    ~MariadbStatementCache() { ClearAll(); }

    MariadbStatementCache(const MariadbStatementCache&) = delete;
    MariadbStatementCache& operator=(const MariadbStatementCache&) = delete;

    /// 取该 SQL 的已预备语句：命中移动到 MRU 并返回；未命中执行
    /// mysql_stmt_init + mysql_stmt_prepare 后插入并返回；prepare 失败关闭
    /// 临时语句并返回 nullptr（errno/error/sqlstate 出参在关闭前捕获——stmt
    /// 关闭后其 errno 不可读，调用方用其生成诊断；失败语句不进入缓存）。容量
    /// 超限时逐出 LRU 端语句。
    MYSQL_STMT* Acquire(
        const std::string& sql, unsigned int* prepareErrno = nullptr, std::string* prepareError = nullptr,
        std::string* prepareSqlstate = nullptr)
    {
        const auto found = entries_.find(sql);
        if (found != entries_.end()) {
            order_.splice(order_.begin(), order_, found->second.second);
            found->second.second = order_.begin();
            ++hits_;
            return found->second.first;
        }
        ++misses_;
        MYSQL_STMT* stmt = mysql_stmt_init(mysql_);
        if (stmt == nullptr) {
            // 客户端侧分配失败（无服务端错误码）：errno 保持 0，错误文本由
            // 调用方补充
            if (prepareErrno != nullptr) {
                *prepareErrno = 0;
            }
            if (prepareError != nullptr) {
                *prepareError = "mysql_stmt_init returned null (client-side allocation failure)";
            }
            if (prepareSqlstate != nullptr) {
                *prepareSqlstate = "-";
            }
            return nullptr;
        }
        if (mysql_stmt_prepare(stmt, sql.c_str(), static_cast<unsigned long>(sql.size())) != 0) {
            if (prepareErrno != nullptr) {
                *prepareErrno = mysql_stmt_errno(stmt);
            }
            if (prepareError != nullptr) {
                *prepareError = mysql_stmt_error(stmt);
            }
            if (prepareSqlstate != nullptr) {
                *prepareSqlstate = mysql_stmt_sqlstate(stmt);
            }
            (void)mysql_stmt_close(stmt); // 预备失败：不留半初始化语句
            return nullptr;
        }
        order_.push_front(sql);
        const auto position = order_.begin();
        entries_.emplace(sql, std::make_pair(stmt, position));
        while (entries_.size() > capacity_) {
            const std::string& lru = order_.back();
            const auto evict = entries_.find(lru);
            if (evict != entries_.end()) {
                CloseStatement(evict->second.first);
                entries_.erase(evict);
            }
            order_.pop_back();
        }
        return stmt;
    }

    /// 丢弃一个语句并从缓存移除（执行失败时调用，不留脏 stmt；不存在时无操作）
    void Discard(const std::string& sql)
    {
        const auto found = entries_.find(sql);
        if (found == entries_.end()) {
            return;
        }
        CloseStatement(found->second.first);
        order_.erase(found->second.second);
        entries_.erase(found);
    }

    /// 关闭全部语句并清空缓存（连接关闭/放弃连接前必须调用；先于 mysql_close）
    void ClearAll()
    {
        for (auto& entry : entries_) {
            CloseStatement(entry.second.first);
        }
        entries_.clear();
        order_.clear();
    }

    /// 观察计数（仅供单元测试固化缓存合同；产品代码不得依赖）
    uint64_t Hits() const { return hits_; }
    uint64_t Misses() const { return misses_; }
    std::size_t Size() const { return entries_.size(); }

private:
    /// 统一的语句关闭（忽略返回值：关闭失败时连接已不可用，由调用方处理）
    void CloseStatement(MYSQL_STMT* stmt)
    {
        if (stmt != nullptr) {
            (void)mysql_stmt_close(stmt);
        }
    }

    MYSQL* mysql_{nullptr};
    std::size_t capacity_{kDefaultCapacity};
    /// LRU 顺序表（front = MRU）；值是 SQL 文本本身（与 entries_ 的键同源）
    std::list<std::string> order_{};
    std::unordered_map<std::string, std::pair<MYSQL_STMT*, std::list<std::string>::iterator>> entries_{};
    uint64_t hits_{0};
    uint64_t misses_{0};
};
} // namespace HcclSim::Storage

#endif // HCCL_VM_STORAGE_ENGINE_MARIADB_MARIADB_STATEMENT_CACHE_H
