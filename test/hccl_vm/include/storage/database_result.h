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
 * @file database_result.h
 * @brief 统一存储抽象层的通用结果容器（DbResult）与错误码（DbCode）定义。
 *
 * 【错误码语义映射规范】
 *
 *   DbCode                   SQLite 语义对应                    MariaDB 语义对应
 *   ─────────────────────────────────────────────────────────────────────────────
 *   OK                       SQLITE_OK / SQLITE_DONE            查询正常返回
 *   NOT_FOUND                单条主键查询无结果                 单条主键查询无结果
 *   CONFLICT                 主键/唯一索引冲突 (SQLITE_CONSTRAINT) 唯一键冲突 (ER_DUP_ENTRY)
 *   INVALID_ARGUMENT         SQL 语法错误 / 非法列绑定          非法入参/非法配置
 *   INVALID_STATE            连接已关闭 / 事务状态不合法        连接已关闭 / 事务状态不合法
 *   UNSUPPORTED_CAPABILITY   当前后端未实现该逻辑能力           当前后端不支持请求的操作
 *   RESOURCE_EXHAUSTED       SQLITE_BUSY / 锁超时 / 内存满      锁等待超时 / 死锁回滚
 *   BACKEND_UNAVAILABLE      连接断开 / socket 失败             连接断开族 (2002/2003/2006/2013...)
 *   IO_ERROR                 磁盘读取损坏 / 写入静止校验失败    其余服务端错误
 */
#ifndef HCCL_VM_STORAGE_DATABASE_RESULT_H
#define HCCL_VM_STORAGE_DATABASE_RESULT_H

#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

namespace HcclSim::Storage {
/// 后端无关的标准存储返回码
enum class DbCode : uint8_t {
    OK,                     ///< 操作成功完成
    NOT_FOUND,              ///< 目标记录不存在（仅用于单条主键查询）
    CONFLICT,               ///< 主键冲突、唯一性约束违反或并发冲突
    INVALID_ARGUMENT,       ///< 非法入参（如字段类型不匹配、空标识符）
    INVALID_STATE,          ///< 当前对象状态不允许执行该操作（如在关闭的会话上操作）
    UNSUPPORTED_CAPABILITY, ///< 当前后端或配置不支持该能力
    RESOURCE_EXHAUSTED,     ///< 资源耗尽（如排空超时、锁等待超时、内存超限）
    BACKEND_UNAVAILABLE,    ///< 底层存储引擎未就绪或连接断开
    IO_ERROR                ///< 底层文件/物理存储 I/O 错误
};

/**
 * @brief 携带返回值的通用存储操作结果包装器。
 * @tparam T 返回值的数据类型
 *
 * 【箭头/解引用访问合同】
 * operator-> 与 operator* 仅供调用方在 ok() 检查成功之后使用，
 * 用于减少 `.value` 访问噪声；失败结果上误用属于程序错误，直接终止进程
 * （与 StorageSessionLease::Get 的空句柄处置一致），不会吞掉错误码、
 * 也不会静默返回默认值。
 */
template <typename T>
struct DbResult {
    DbCode code{DbCode::OK};  ///< 状态码
    std::optional<T> value;   ///< 成功时的可选返回值载荷
    uint64_t affectedRows{0}; ///< 本次操作影响的记录行数
    std::string diagnostic;   ///< 失败时的详细诊断信息（错误原因、调试上下文）

    /// 快捷判断操作是否成功
    bool ok() const { return code == DbCode::OK; }

    /// 成功结果的成员访问（必须先通过 ok() 检查；失败时终止进程）
    T* operator->()
    {
        EnsureValue();
        return &*value;
    }

    /// 成功结果的只读成员访问（必须先通过 ok() 检查；失败时终止进程）
    const T* operator->() const
    {
        EnsureValue();
        return &*value;
    }

    /// 成功结果的左值解引用（必须先通过 ok() 检查；失败时终止进程）
    T& operator*() &
    {
        EnsureValue();
        return *value;
    }

    /// 成功结果的只读左值解引用（必须先通过 ok() 检查；失败时终止进程）
    const T& operator*() const&
    {
        EnsureValue();
        return *value;
    }

    /// 成功结果的右值解引用（用于移出 Move-Only 载荷；失败时终止进程）
    T&& operator*() &&
    {
        EnsureValue();
        return std::move(*value);
    }

    /**
     * @brief 显式默认值访问（语义对齐 std::optional::value_or）。
     * @tparam U 可转换为 T 的默认值类型
     * @param fallback 调用方显式写出的默认值
     * @return 成功且有载荷时返回实际载荷；失败或无载荷时返回 fallback
     *
     * 【使用边界】
     * 1. 仅用于迁移前就以默认值表达失败的只读查询（如计数失败按 0 处理）；
     * 2. 不提供隐式 T 转换：只有调用方明确写出 value_or(default) 才产生默认值语义；
     * 3. 不记录日志、不在接口内部选择默认值；
     * 4. 不改变 operator-> / operator* 失败即终止的合同——需要感知错误码并向上
     *    传递的调用方必须继续检查 ok() 并映射原错误码，不得改用本方法缩短代码。
     */
    template <typename U, typename = std::enable_if_t<std::is_copy_constructible_v<T> && std::is_convertible_v<U&&, T>>>
    T value_or(U&& fallback) const&
    {
        return (ok() && value.has_value()) ? *value : static_cast<T>(std::forward<U>(fallback));
    }

    /// 右值对象的显式默认值访问：成功时移出载荷，避免多余的复制
    template <typename U, typename = std::enable_if_t<std::is_move_constructible_v<T> && std::is_convertible_v<U&&, T>>>
    T value_or(U&& fallback) &&
    {
        return (ok() && value.has_value()) ? std::move(*value) : static_cast<T>(std::forward<U>(fallback));
    }

private:
    /// 值缺失时终止：箭头访问不得吞错或返回伪造载荷
    void EnsureValue() const
    {
        if (!value.has_value()) {
            std::terminate();
        }
    }
};

/**
 * @brief 无返回值的操作结果特化版本（void）。
 */
template <>
struct DbResult<void> {
    DbCode code{DbCode::OK};  ///< 状态码
    uint64_t affectedRows{0}; ///< 本次操作影响的记录行数
    std::string diagnostic;   ///< 失败时的详细诊断信息

    /// 快捷判断操作是否成功
    bool ok() const { return code == DbCode::OK; }
};
} // namespace HcclSim::Storage

#endif // HCCL_VM_STORAGE_DATABASE_RESULT_H
