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
 * @file logical_operation.h
 * @brief 类型安全结果写回槽（ResultSink）与事务原子结果。
 *
 * 【边界】公开业务 API 完成唯一参数对象构造后，中间层只传 const 引用、
 * 原始指针、静态 meta 指针和标量；不存在通用操作视图或 payload variant。
 * ResultSink 只服务 SELECT 写回：持有调用方最终目标地址与类型，提供
 * 类型校验后的目标指针访问，不构造、不复制、不接管所有权。INSERT 由
 * 引擎直接返回生成主键，COUNT/UPDATE/DELETE 直接返回整数，CLEAR 返回
 * void，LIST_TABLES 直接返回字符串 vector。
 *
 * 【事务所有权】事务 DML 直达引擎：record/ConditionData/UpdateData 以
 * const 引用/原始指针传入引擎后立即执行，事务层不保存任何业务载荷，
 * 只按执行顺序记录 AtomicItemResult。
 */
#ifndef HCCL_VM_STORAGE_LOGICAL_OPERATION_H
#define HCCL_VM_STORAGE_LOGICAL_OPERATION_H

#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <typeindex>
#include <utility>
#include <vector>

#include "storage/database_result.h"
#include "storage/query_spec.h"

namespace HcclSim::Storage {
/**
 * @brief 类型安全的结果写回槽（ResultSink）。
 * @note 仅用于 SELECT：持有调用方最终业务对象/容器的地址与类型；
 *       materialize 直接填该目标，不构造中间结果容器。
 */
class ResultSink final {
public:
    ResultSink() = default;

    /// 构造指定目标对象的写回槽
    template <typename T>
    static ResultSink For(T& value)
    {
        static_assert(!std::is_const_v<T>, "result target must be writable");
        return ResultSink(&value, typeid(T));
    }

    std::type_index Type() const { return std::type_index(*type_); }

    /// 类型校验后的目标指针访问：目标类型完全匹配时返回原调用方对象地址，
    /// 不构造、不复制、不接管所有权；类型不匹配返回 nullptr。
    template <typename T>
    T* Target() const noexcept
    {
        return target_ != nullptr && *type_ == typeid(T) ? static_cast<T*>(target_) : nullptr;
    }

private:
    ResultSink(void* target, const std::type_info& type) : target_(target), type_(&type) {}

    void* target_{nullptr};
    const std::type_info* type_{&typeid(void)};
};

/// 单个原子操作执行结果（生成主键与影响行数）
struct AtomicItemResult {
    std::optional<uint64_t> generatedId;
    uint64_t affectedRows{0};
};

/// 事务原子提交全局结果
struct AtomicResult {
    std::vector<AtomicItemResult> items;
};
} // namespace HcclSim::Storage

#endif // HCCL_VM_STORAGE_LOGICAL_OPERATION_H
