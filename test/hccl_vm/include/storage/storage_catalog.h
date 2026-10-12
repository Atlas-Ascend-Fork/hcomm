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
 * @file storage_catalog.h
 * @brief 后端无关逻辑 DDL 目录：一份 master catalog 登记派生全部表契约产物。
 *
 * 【统一目录机制】
 * 每个逻辑数据库只有一个 master catalog X-macro；每张表写一个紧凑 Table(Type, Body) entry，
 * entry 内用 Field(member, 属性...) 声明字段，随后集中声明表级 DDL（PrimaryKey/Unique/Index/
 * ForeignKey）。公共生成器对同一 entry 多次受限展开，派生：
 *   - constexpr 表 descriptor（表名/版本/scope/字段/约束/索引，静态只读数据）；
 *   - RecordSchema<T> 特化（字段编号映射，业务不再手写）；
 *   - 类型擦除 CatalogTableMeta（manifest、表数量、fingerprint 与关系 mapping 的输入）。
 * 不存在字段 X-macro、traits、mapping、索引 if-chain、数量清单等多个手写事实源。
 *
 * 【字段完整性漂移门】
 * 字段名由 #member 产生，严格等于 C++ 成员名；成员指针使改名/删除直接编译失败；
 * kDirectAggregateFieldCount<T>（每个直接成员单独一层花括号的聚合 arity 探针，数组与多维
 * 数组按一个成员计数）必须等于目录声明字段数，新增成员漏登在编译期失败；成员类型与
 * 约束进入 schema fingerprint，布局漂移可被引擎在打开期发现。
 *
 * 【DDL 编译期校验（全部在表条目展开处触发）】
 *   - PrimaryKey/Unique/Index 的每一列与 ForeignKey 的本地/目标列都经 &RecordType::member
 *     成员指针级绑定，引用不存在成员直接编译失败（列名仍由宏参数字符串化）；
 *   - PrimaryKey 列必须是非 bool 的整数或枚举成员（当前主键合同：PrimaryKey<T>
 *     值类型为 uint64_t、主键读写遍只消费整数/枚举成员、SQLite 只支持单列 INTEGER
 *     主键；string/数组/浮点/bool/Blob/aggregate 成员在编译期拒绝；Unique/Index
 *     列不受此门约束），且每张表必须恰好声明一次 PrimaryKey——两次声明不再静默
 *     后写覆盖前写；
 *   - 同一 Table 重复 Field(member) 非法（防止重复声明掩盖另一遗漏成员）；
 *   - 同一 Field 重复属性、Text+Blob 组合非法；AutoIncrement 只允许非 bool 整数成员，
 *     且必须是该表唯一单列 PrimaryKey（与 PrimaryKey 是两个独立概念）；
 *   - Text 数组编码以 sizeof(成员) 为界：遇 NUL 截断，满容量无 NUL 时精确 N 个字符，
 *     不读取相邻成员；解码最多回写 N-1 个字符（尾字节保留 NUL）。
 *
 * 【记录语义 hook（仅两类，后端无关）】
 *   1. 派生列写回：Field(member, Derived) 的编码值由 RecordCodecHooks<T>::EncodeDerivedField
 *      计算（如 base+size 端点列），解码按 hook 忽略；
 *   2. 有效数组长度：BLOB 数组的有效字节数由同记录另一字段决定
 *      （RecordCodecHooks<T>::EffectiveBytesLength）。
 * 普通 Text/Blob 字段由通用 descriptor 编解码，不逐表手写 codec。
 *
 * 【字段通道 ABI（引擎与目录之间唯一的记录数据通道）】
 *   - visitRecord：按字段声明顺序从原 T 取成员地址，同步调用 EmitFieldRef
 *     （data/bytes/isNull 原始引用；派生列以 DbScalar 原生宽度临时发射）；
 *   - materializeRows：经 sink.Target<T>()/Target<vector<T>>() 取得调用方最终
 *     容器，单对象直接填目标，vector 逐行 emplace_back 后经 ReadFieldInto
 *     填成员（字段失败 pop_back 未完成行）。
 * 不存在借用型字段视图或字段值数组等第二套通道。
 *
 * 【短宏边界】
 * 底层宏全部使用 HCCL_VM_DDL_ 前缀；业务 DDL 区可局部 #define Field/PrimaryKey/Index 等
 * 短别名（指向本文件前缀宏），区段末尾必须立即 #undef，不得污染包含方。
 */
#ifndef HCCL_VM_STORAGE_CATALOG_H
#define HCCL_VM_STORAGE_CATALOG_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <typeindex>
#include <utility>
#include <vector>
#include <variant>

#include "storage/database_result.h"
#include "storage/logical_operation.h"
#include "storage/query_spec.h"
#include "storage/record_contract.h"

namespace HcclSim::Storage {
// ---- 字段通道 ABI：目录与后端引擎之间唯一的记录数据通道（原始引用形态）----
// INSERT 路径：目录 visitRecord 按声明顺序同步调用 EmitFieldRef，把原 T 成员
// 地址/长度/空标志直发后端（后端立即绑定数据库 API，不构造字段值数组）。
using EmitFieldRef = DbResult<void> (*)(void* context, uint16_t fieldId, const void* data, size_t bytes, bool isNull);

// SELECT 路径：后端经 NextRow 推进行，目录 materializeRows 把最终 T 成员地址
// 交给 ReadFieldInto，后端从数据库结果直接写入该成员。
using NextRow = DbResult<bool> (*)(void* context);
using ReadFieldInto = DbResult<void> (*)(void* context, uint16_t fieldId, void* destination, size_t capacity);

namespace Ddl {

    // ---- 单表元素数量上限（超过即在编译期失败，防止静默截断） ----
    inline constexpr std::size_t kMaxFieldsPerTable = 64; ///< 单表字段数上限
    inline constexpr std::size_t kMaxIndexColumns = 4;    ///< 复合索引列数上限
    // 普通索引与唯一索引共享同一张目录数组；容量必须覆盖现有声明，溢出由表条目
    // 编译期门拒绝，不能静默丢弃。14 = VirtualMemBlock 的 14 条复合索引
    // （含 2026-09-20 purge 用 (device_id, is_freed)），其余表均远低于该上限。
    inline constexpr std::size_t kMaxIndexesPerTable = 14;    ///< 单表索引条目上限
    inline constexpr std::size_t kMaxKeyColumns = 4;          ///< 主键/唯一约束列数上限
    inline constexpr std::size_t kMaxForeignKeysPerTable = 4; ///< 单表外键数上限

    // ---- 字段元属性（位掩码组合；命名空间级短名供 DDL 区直接书写） ----
    enum class Prop : unsigned {
        None = 0,
        AutoIncrement = 1u << 0, ///< 自增主键生成属性（字段属性，非表约束）
        Text = 1u << 1,          ///< char[N]/字节数组按文本编码（NUL 前字节承载语义）
        Blob = 1u << 2,          ///< 字节数组按 BLOB 逐字节保存
        Derived = 1u << 3,       ///< 派生列：编码值由 RecordCodecHooks 计算（如 base+size 端点）
    };
    inline constexpr Prop AutoIncrement = Prop::AutoIncrement;
    inline constexpr Prop Text = Prop::Text;
    inline constexpr Prop Blob = Prop::Blob;
    inline constexpr Prop Derived = Prop::Derived;

    /// 表作用域（静态公共表 / PID 独占动态表）
    enum class ScopeKind : uint8_t { Static, DynamicPid };

    /// 外键删除策略
    enum class OnDelete : uint8_t { Restrict, Cascade };

    inline constexpr bool HasProp(unsigned props, Prop wanted) { return (props & static_cast<unsigned>(wanted)) != 0U; }

    // ============================================================
    // 聚合 arity 探针：每个直接成员单独一层花括号。
    // 标量成员 ← {} 值初始化合法；数组成员 ← {} 整体零初始化合法且不会被
    // 花括号省略拆开（char[N]/多维数组各计为一个直接成员）。
    // ============================================================
    template <std::size_t N>
    struct ArityPriority : ArityPriority<N - 1> {};
    template <>
    struct ArityPriority<0> {};

#define HCCL_VM_DDL_ARITY_BRACES_0
#define HCCL_VM_DDL_ARITY_BRACES_1 \
    {}
#define HCCL_VM_DDL_ARITY_BRACES_2 \
    {}, {}
#define HCCL_VM_DDL_ARITY_BRACES_3 \
    {}, {}, {}
#define HCCL_VM_DDL_ARITY_BRACES_4 \
    {}, {}, {}, {}
#define HCCL_VM_DDL_ARITY_BRACES_5 \
    {}, {}, {}, {}, {}
#define HCCL_VM_DDL_ARITY_BRACES_6 \
    {}, {}, {}, {}, {}, {}
#define HCCL_VM_DDL_ARITY_BRACES_7 \
    {}, {}, {}, {}, {}, {}, {}
#define HCCL_VM_DDL_ARITY_BRACES_8 \
    {}, {}, {}, {}, {}, {}, {}, {}
#define HCCL_VM_DDL_ARITY_BRACES_9 \
    {}, {}, {}, {}, {}, {}, {}, {}, {}
#define HCCL_VM_DDL_ARITY_BRACES_10 \
    {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
#define HCCL_VM_DDL_ARITY_BRACES_11 \
    {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
#define HCCL_VM_DDL_ARITY_BRACES_12 \
    {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
#define HCCL_VM_DDL_ARITY_BRACES_13 \
    {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
#define HCCL_VM_DDL_ARITY_BRACES_14 \
    {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
#define HCCL_VM_DDL_ARITY_BRACES_15 \
    {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
#define HCCL_VM_DDL_ARITY_BRACES_16 \
    {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
#define HCCL_VM_DDL_ARITY_BRACES_17 \
    {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
#define HCCL_VM_DDL_ARITY_BRACES_18 \
    {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
#define HCCL_VM_DDL_ARITY_BRACES_19 \
    {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
#define HCCL_VM_DDL_ARITY_BRACES_20 \
    {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
#define HCCL_VM_DDL_ARITY_BRACES_21 \
    {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
#define HCCL_VM_DDL_ARITY_BRACES_22 \
    {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
#define HCCL_VM_DDL_ARITY_BRACES_23 \
    {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
#define HCCL_VM_DDL_ARITY_BRACES_24 \
    {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
#define HCCL_VM_DDL_ARITY_BRACES_25 \
    {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
#define HCCL_VM_DDL_ARITY_BRACES_26 \
    {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
#define HCCL_VM_DDL_ARITY_BRACES_27 \
    {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
#define HCCL_VM_DDL_ARITY_BRACES_28 \
    {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
#define HCCL_VM_DDL_ARITY_BRACES_29 \
    {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
#define HCCL_VM_DDL_ARITY_BRACES_30                                                                                 \
    {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, \
        {},                                                                                                         \
    {}
#define HCCL_VM_DDL_ARITY_BRACES_31                                                                                 \
    {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, \
        {}, {},                                                                                                     \
    {}
#define HCCL_VM_DDL_ARITY_BRACES_32                                                                                 \
    {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, \
        {}, {}, {},                                                                                                 \
    {}

#define HCCL_VM_DDL_ARITY_CASE(N)                                                                                     \
    template <class T>                                                                                                \
    constexpr auto DirectArityAt(                                                                                     \
        ArityPriority<N>, int) -> decltype(T{HCCL_VM_DDL_ARITY_BRACES_##N}, std::integral_constant<std::size_t, N>()) \
    {                                                                                                                 \
        return {};                                                                                                    \
    }
    HCCL_VM_DDL_ARITY_CASE(32)
    HCCL_VM_DDL_ARITY_CASE(31)
    HCCL_VM_DDL_ARITY_CASE(30)
    HCCL_VM_DDL_ARITY_CASE(29)
    HCCL_VM_DDL_ARITY_CASE(28)
    HCCL_VM_DDL_ARITY_CASE(27)
    HCCL_VM_DDL_ARITY_CASE(26)
    HCCL_VM_DDL_ARITY_CASE(25)
    HCCL_VM_DDL_ARITY_CASE(24)
    HCCL_VM_DDL_ARITY_CASE(23)
    HCCL_VM_DDL_ARITY_CASE(22)
    HCCL_VM_DDL_ARITY_CASE(21)
    HCCL_VM_DDL_ARITY_CASE(20)
    HCCL_VM_DDL_ARITY_CASE(19)
    HCCL_VM_DDL_ARITY_CASE(18)
    HCCL_VM_DDL_ARITY_CASE(17)
    HCCL_VM_DDL_ARITY_CASE(16)
    HCCL_VM_DDL_ARITY_CASE(15)
    HCCL_VM_DDL_ARITY_CASE(14)
    HCCL_VM_DDL_ARITY_CASE(13)
    HCCL_VM_DDL_ARITY_CASE(12)
    HCCL_VM_DDL_ARITY_CASE(11)
    HCCL_VM_DDL_ARITY_CASE(10)
    HCCL_VM_DDL_ARITY_CASE(9)
    HCCL_VM_DDL_ARITY_CASE(8)
    HCCL_VM_DDL_ARITY_CASE(7)
    HCCL_VM_DDL_ARITY_CASE(6)
    HCCL_VM_DDL_ARITY_CASE(5)
    HCCL_VM_DDL_ARITY_CASE(4)
    HCCL_VM_DDL_ARITY_CASE(3)
    HCCL_VM_DDL_ARITY_CASE(2)
    HCCL_VM_DDL_ARITY_CASE(1)
    HCCL_VM_DDL_ARITY_CASE(0)

    template <class T>
    constexpr std::size_t DirectArityAt(ArityPriority<0>, ...)
    {
        return 0;
    }

    /// 直接聚合成员数（数组/多维数组按一个成员计数）；非聚合或超上限均为 0，由表条目编译期门拒绝
    template <class T>
    inline constexpr std::size_t kDirectAggregateFieldCount = decltype(DirectArityAt<T>(ArityPriority<32>{}, 0))::value;

    // ============================================================
    // 遍元素类型（Body 中每个属性宏的展开产物；宏体引用的 RecordType 等名字
    // 在表条目展开的最终语法位置惰性解析——donor 已验证的 X-macro 技巧）
    // ============================================================
    template <typename T, typename M>
    struct NamePtr {
        M T::*ptr;
        const char* name;
    };
    template <typename T, typename M>
    constexpr NamePtr<T, M> MakeNamePtr(M T::*ptr, const char* name)
    {
        return {ptr, name};
    }

    /// 约束列成员绑定：&RecordType::member 在此完成对真实成员的指针级编译绑定
    /// （成员不存在或改名直接编译失败），列名仍取宏参数字符串化结果；
    /// 约束条目（PrimaryKey/Unique/Index）继续只持列名，不引入第二套 schema。
    template <typename T, typename M>
    constexpr const char* BindKeyColumn(M T::*member, const char* name)
    {
        static_cast<void>(member);
        return name;
    }

    /// 主键列成员绑定：在 BindKeyColumn 的成员指针级绑定之上叠加主键类型门——
    /// 当前主键合同（PrimaryKey<T> 值类型为 uint64_t；主键读写遍只消费整数/枚举
    /// 成员；SQLite 只支持单列 INTEGER 主键）要求主键列必须是非 bool 的整数或枚举
    /// 成员，string/数组/浮点/bool/Blob/aggregate 成员在编译期拒绝。Unique/Index
    /// 列继续走 BindKeyColumn，文本列仍可声明唯一/索引约束，语义不变。
    template <typename T, typename M>
    constexpr const char* BindPrimaryKeyColumn(M T::*member, const char* name)
    {
        static_assert(
            (std::is_integral_v<M> || std::is_enum_v<M>) && !std::is_same_v<M, bool>,
            "Primary key columns must be non-bool integer or enum members "
            "(string/array/floating/bool/Blob/aggregate members are rejected)");
        static_cast<void>(member);
        return name;
    }

    /// 非当前遍关心的元素统一展开产物（每遍的 ctx 用模板 operator, 吸收）
    struct IgnoreItem {};

    namespace detail {
        /// 属性包内是否声明了重复属性（同一 Field 写两次同一属性属于 DDL 错误，
        /// 位掩码按位或会静默折叠，必须逐项检测；0/1 个属性不可能重复）
        template <Prop... Props>
        constexpr bool HasDuplicateProps()
        {
            if constexpr (sizeof...(Props) > 1) {
                unsigned seen = 0U;
                for (const unsigned prop : {static_cast<unsigned>(Props)...}) {
                    if ((seen & prop) != 0U) {
                        return true;
                    }
                    seen |= prop;
                }
            }
            return false;
        }
    } // namespace detail

    /// 字段元属性标签（编译期属性集合；属性顺序不改变语义；重复属性非法）
    template <Prop... Props>
    struct PropsTag {
        static_assert(
            !detail::HasDuplicateProps<Props...>(), "field properties must not be declared twice in the same Field");
        static constexpr unsigned value = (0U | ... | static_cast<unsigned>(Props));
    };
    using NoProps = PropsTag<>;

    template <typename T, typename M, typename Tag>
    struct FieldItem {
        NamePtr<T, M> member;
        static constexpr unsigned kProps = Tag::value; ///< 字段元属性（编译期参与类型推导与校验）
    };
    template <typename T, typename M, typename Tag>
    constexpr FieldItem<T, M, Tag> MakeField(NamePtr<T, M> member, Tag)
    {
        return {member};
    }

    template <typename T>
    struct VersionItem {
        uint32_t value;
    };
    template <typename T>
    constexpr VersionItem<T> MakeVersion(uint32_t value)
    {
        return {value};
    }

    /// 主键约束条目（列名即成员名；主键列本身在字段区声明）
    struct PrimaryKeyItem {
        std::array<const char*, kMaxKeyColumns> columns{};
        std::size_t columnCount{0};
    };
    constexpr PrimaryKeyItem MakePrimaryKey(
        const char* first, const char* second = nullptr, const char* third = nullptr, const char* fourth = nullptr)
    {
        PrimaryKeyItem item;
        const char* columns[kMaxKeyColumns] = {first, second, third, fourth};
        for (std::size_t i = 0; i < kMaxKeyColumns; ++i) {
            if (columns[i] == nullptr) {
                break;
            }
            item.columns[i] = columns[i];
            ++item.columnCount;
        }
        return item;
    }

    /// 唯一约束条目（正确性约束，与普通索引分开声明）
    struct UniqueItem {
        std::array<const char*, kMaxKeyColumns> columns{};
        std::size_t columnCount{0};
    };
    constexpr UniqueItem MakeUnique(
        const char* first, const char* second = nullptr, const char* third = nullptr, const char* fourth = nullptr)
    {
        UniqueItem item;
        const char* columns[kMaxKeyColumns] = {first, second, third, fourth};
        for (std::size_t i = 0; i < kMaxKeyColumns; ++i) {
            if (columns[i] == nullptr) {
                break;
            }
            item.columns[i] = columns[i];
            ++item.columnCount;
        }
        return item;
    }

    /// 复合索引条目（列名按声明顺序为复合索引顺序；物理名称由后端自动生成）
    struct IndexItem {
        std::array<const char*, kMaxIndexColumns> columns{};
        std::size_t columnCount{0};
    };
    constexpr IndexItem MakeIndex(
        const char* first, const char* second = nullptr, const char* third = nullptr, const char* fourth = nullptr)
    {
        IndexItem item;
        const char* columns[kMaxIndexColumns] = {first, second, third, fourth};
        for (std::size_t i = 0; i < kMaxIndexColumns; ++i) {
            if (columns[i] == nullptr) {
                break;
            }
            item.columns[i] = columns[i];
            ++item.columnCount;
        }
        return item;
    }

    /// 级联外键条目（local 本表成员；target 为 "目标表::成员" 表达，携带目标成员指针）
    template <typename T, typename M, typename TargetT, typename TargetM>
    struct ForeignKeyItem {
        NamePtr<T, M> local;
        NamePtr<TargetT, TargetM> target;
        const char* targetName; ///< "目标表::成员"（由宏字符串化产生，关系层解析）
        OnDelete onDelete;
    };
    template <typename T, typename M, typename TargetT, typename TargetM>
    constexpr ForeignKeyItem<T, M, TargetT, TargetM>
    MakeForeignKey(NamePtr<T, M> local, NamePtr<TargetT, TargetM> target, const char* targetName, OnDelete onDelete)
    {
        return {local, target, targetName, onDelete};
    }

    // ============================================================
    // 只读 descriptor 形态（constexpr 生成，运行期持指针/长度，不复制名称）
    // ============================================================
    /// 字段描述（字段名严格等于 C++ 成员名；类型由成员类型与属性推导）
    struct FieldDesc {
        const char* name{""};
        LogicalFieldType type{LogicalFieldType::BYTES};
        unsigned props{0};
        bool primaryKey{false};
        /// 固定字节宽度：定宽标量/枚举/浮点/bool、C 数组与不透明 aggregate Blob 为 sizeof(M)；
        /// std::string/std::vector<uint8_t> 等变长逻辑值为 0。由 detail::MemberByteWidth 唯一派生。
        std::size_t byteWidth{0};
    };

    /// 索引描述（普通索引与唯一索引统一承载）
    struct IndexDesc {
        std::array<const char*, kMaxIndexColumns> columns{};
        std::size_t columnCount{0};
        bool unique{false};
    };

    /// 外键描述（目标以 "表名::成员名" 表达，由关系层解析到目标表列）
    struct ForeignKeyDesc {
        const char* localColumn{""};
        const char* target{""};
        bool cascadeDelete{false};
    };

    /// 表头与表级约束/索引静态描述
    struct TableHead {
        uint32_t version{1};
        ScopeKind scope{ScopeKind::Static};
        PrimaryKeyItem primaryKey{};
        std::array<IndexDesc, kMaxIndexesPerTable> indexes{};
        std::size_t indexCount{0};
        std::array<ForeignKeyDesc, kMaxForeignKeysPerTable> foreignKeys{};
        std::size_t foreignKeyCount{0};
    };

    /// 判断索引 descriptor 的结构性合同（列数在 [1, kMaxIndexColumns] 内）。
    constexpr bool IndexDescriptorsValid(const TableHead& head)
    {
        for (std::size_t i = 0; i < head.indexCount; ++i) {
            const IndexDesc& index = head.indexes[i];
            if (index.columnCount == 0U || index.columnCount > kMaxIndexColumns) {
                return false;
            }
        }
        return true;
    }

    /// 类型擦除表元信息（引擎装配与 CRUD 的输入；指向 constexpr 静态数据）。
    /// 字段通道 ABI 固定为 visitRecord（原 T 成员地址直发）与 materializeRows
    /// （最终业务目标直填）两个函数指针，不允许实现者另拟接口。
    /// 回调 context 由后端持有（含当前 stmt/reply 与同一 meta），后端用 fieldId
    /// 取同一 FieldDesc，不得另建类型表。
    struct CatalogTableMeta {
        std::string_view tableId{};
        uint32_t version{0};
        ScopeKind scope{ScopeKind::Static};
        const FieldDesc* fields{nullptr};
        std::size_t fieldCount{0};
        const IndexDesc* indexes{nullptr};
        std::size_t indexCount{0};
        const ForeignKeyDesc* foreignKeys{nullptr};
        std::size_t foreignKeyCount{0};
        const char* const* primaryKeyColumns{nullptr};
        std::size_t primaryKeyCount{0};
        uint64_t fingerprint{0};
        // 类型擦除记录通道：由同一条目派生，recordType/recordSetType 供 sink 类型校验
        std::type_index recordType{typeid(void)};
        std::type_index recordSetType{typeid(void)};
        /// SELECT 行物化：直接填 ResultSink 所指的最终 T/vector<T> 成员
        DbResult<void> (*materializeRows)(ResultSink&, void* context, NextRow next, ReadFieldInto read){nullptr};
        /// INSERT 字段访问：按声明顺序同步发射原 T 成员地址
        DbResult<void> (*visitRecord)(const void* record, void* context, EmitFieldRef emit){nullptr};
    };

    /// 逻辑数据库描述（表清单；databaseId 为 canonical 逻辑库名）
    struct DatabaseDescriptor {
        std::string_view databaseId{};
        const CatalogTableMeta* tables{nullptr};
        std::size_t tableCount{0};
    };

    // ============================================================
    // 成员类型 → 逻辑字段类型推导与属性校验（编译期）
    // ============================================================
    namespace detail {
        /// 不透明 aggregate Blob 判据（单一事实判据，不做业务类型特化）：非数组、aggregate
        /// 且 trivially copyable 的成员可显式声明为固定长度不透明 BLOB 字段——编码精确
        /// sizeof(M) 字节，解码仅字节数精确相等时整体回写。标量、字符串/变长字节与非
        /// 平凡复制 aggregate 均不满足，继续走既有类型推导或被属性门拒绝。
        template <typename M>
        inline constexpr bool kIsOpaqueAggregateMember
            = !std::is_array_v<M> && std::is_aggregate_v<M> && std::is_trivially_copyable_v<M>;

        /// 成员固定字节宽度（FieldDesc.byteWidth 的唯一来源，不另建第二份宽度清单）：
        /// 定宽标量/枚举/浮点/bool、C 数组与不透明 aggregate 为 sizeof(M)；
        /// std::string/std::vector<uint8_t> 等变长逻辑值为 0。
        template <typename M>
        constexpr std::size_t MemberByteWidth()
        {
            if constexpr (std::is_array_v<M>) {
                return sizeof(M);
            } else if constexpr (std::is_same_v<M, std::string> || std::is_same_v<M, std::vector<uint8_t>>) {
                return 0;
            } else {
                return sizeof(M);
            }
        }
    } // namespace detail

    /// 字段属性合法性编译期校验（数组成员必须恰好声明 Text 或 Blob 之一；
    /// 不透明 aggregate 成员必须显式声明 Blob 且不得 Text/AutoIncrement；
    /// AutoIncrement 只允许非 bool 整数成员）
    template <typename M, unsigned Props>
    struct FieldPropsGate {
        static constexpr unsigned value = [] {
            if constexpr (std::is_array_v<M>) {
                static_assert(
                    HasProp(Props, Prop::Text) != HasProp(Props, Prop::Blob),
                    "array members require exactly one of Text or Blob");
                static_assert(!HasProp(Props, Prop::AutoIncrement), "array members cannot be auto increment");
            } else if constexpr (detail::kIsOpaqueAggregateMember<M>) {
                // 嵌套 aggregate 成员是固定长度不透明字节载荷：必须显式写 Blob。
                static_assert(HasProp(Props, Prop::Blob), "opaque aggregate members require an explicit Blob");
                static_assert(!HasProp(Props, Prop::Text), "opaque aggregate members cannot be Text");
                static_assert(
                    !HasProp(Props, Prop::AutoIncrement), "opaque aggregate members cannot be auto increment");
            } else {
                static_assert(
                    !HasProp(Props, Prop::Text) && !HasProp(Props, Prop::Blob),
                    "Text/Blob properties apply to array or opaque aggregate members only");
                static_assert(
                    !HasProp(Props, Prop::AutoIncrement) || (std::is_integral_v<M> && !std::is_same_v<M, bool>),
                    "AutoIncrement applies to non-bool integer members only");
            }
            static_assert(!HasProp(Props, Prop::Text) || !HasProp(Props, Prop::Blob), "Text and Blob are exclusive");
            return Props;
        }();
    };

    /// 属性门辅助：M 的属性必须经过 FieldPropsGate 校验后才可消费
    template <typename M, unsigned Props>
    struct PropertiesFor {
        static constexpr unsigned value = FieldPropsGate<M, Props>::value;
    };

    template <typename M, unsigned Props>
    constexpr LogicalFieldType TypeOfMember()
    {
        if constexpr (std::is_enum_v<M>) {
            return LogicalFieldType::SIGNED;
        } else if constexpr (std::is_same_v<M, bool>) {
            return LogicalFieldType::BOOLEAN;
        } else if constexpr (std::is_integral_v<M> && std::is_unsigned_v<M>) {
            return LogicalFieldType::UNSIGNED;
        } else if constexpr (std::is_integral_v<M>) {
            return LogicalFieldType::SIGNED;
        } else if constexpr (std::is_floating_point_v<M>) {
            return LogicalFieldType::REAL;
        } else if constexpr (std::is_same_v<M, std::string>) {
            return LogicalFieldType::STRING;
        } else if constexpr (std::is_same_v<M, std::vector<uint8_t>>) {
            return LogicalFieldType::BYTES;
        } else if constexpr (std::is_array_v<M>) {
            // 数组成员的文本/BLOB 语义必须显式声明；类型推导按声明属性落位。
            return HasProp(PropertiesFor<M, Props>::value, Prop::Text) ? LogicalFieldType::STRING :
                                                                         LogicalFieldType::BYTES;
        } else if constexpr (detail::kIsOpaqueAggregateMember<M>) {
            // 嵌套 aggregate 成员：显式 Blob 声明的固定长度不透明字节载荷。
            return LogicalFieldType::BYTES;
        } else {
            static_assert(!sizeof(M), "member type is not a supported database field");
            return LogicalFieldType::BYTES;
        }
    }

    // ============================================================
    // schema fingerprint：表名/版本/字段名/类型/属性/主键/索引全量哈希
    // （布局漂移检测输入；引擎打开期比对，字段或约束变化而版本未提升即拒绝）
    // ============================================================
    constexpr uint64_t FnvMixByte(uint64_t hash, char byte)
    {
        return (hash ^ static_cast<uint64_t>(static_cast<unsigned char>(byte))) * 1099511628211ULL;
    }

    constexpr uint64_t FnvMixText(uint64_t hash, const char* text)
    {
        for (const char* p = text; p != nullptr && *p != '\0'; ++p) {
            hash = FnvMixByte(hash, *p);
        }
        return FnvMixByte(hash, '|');
    }

    constexpr uint64_t FnvMixUint(uint64_t hash, uint64_t value)
    {
        for (int shift = 0; shift < 64; shift += 8) {
            hash = FnvMixByte(hash, static_cast<char>((value >> shift) & 0xFFU));
        }
        return FnvMixByte(hash, '|');
    }

    constexpr uint64_t
    ComputeTableFingerprint(const char* tableId, const TableHead& head, const FieldDesc* fields, std::size_t fieldCount)
    {
        uint64_t hash = 1469598103934665603ULL;
        hash = FnvMixText(hash, tableId);
        hash = FnvMixUint(hash, head.version);
        hash = FnvMixUint(hash, static_cast<uint64_t>(head.scope));
        for (std::size_t i = 0; i < fieldCount; ++i) {
            hash = FnvMixText(hash, fields[i].name);
            hash = FnvMixUint(hash, static_cast<uint64_t>(fields[i].type));
            hash = FnvMixUint(hash, fields[i].props);
            hash = FnvMixUint(hash, fields[i].primaryKey ? 1U : 0U);
            hash = FnvMixUint(hash, fields[i].byteWidth);
        }
        hash = FnvMixUint(hash, head.primaryKey.columnCount);
        for (std::size_t i = 0; i < head.primaryKey.columnCount; ++i) {
            hash = FnvMixText(hash, head.primaryKey.columns[i]);
        }
        for (std::size_t i = 0; i < head.indexCount; ++i) {
            hash = FnvMixUint(hash, head.indexes[i].unique ? 1U : 0U);
            hash = FnvMixUint(hash, head.indexes[i].columnCount);
            for (std::size_t c = 0; c < head.indexes[i].columnCount; ++c) {
                hash = FnvMixText(hash, head.indexes[i].columns[c]);
            }
        }
        for (std::size_t i = 0; i < head.foreignKeyCount; ++i) {
            hash = FnvMixText(hash, head.foreignKeys[i].localColumn);
            hash = FnvMixText(hash, head.foreignKeys[i].target);
            hash = FnvMixUint(hash, head.foreignKeys[i].cascadeDelete ? 1U : 0U);
        }
        return hash;
    }

    // ============================================================
    // 编译期名字匹配（主键列名 ↔ 字段名）
    // ============================================================
    constexpr bool TextEquals(const char* left, const char* right)
    {
        if (left == nullptr || right == nullptr) {
            return false;
        }
        std::size_t i = 0;
        while (left[i] != '\0' && right[i] != '\0') {
            if (left[i] != right[i]) {
                return false;
            }
            ++i;
        }
        return left[i] == right[i];
    }

    /// 取全限定类型名的最后一段作为 canonical 表名（如 "sim::runtime::Server" -> "Server"）
    constexpr const char* LastScopeName(const char* typeName)
    {
        const char* last = typeName;
        for (const char* p = typeName; p != nullptr && *p != '\0'; ++p) {
            if (*p == ':' && p[1] == ':' && p[2] != '\0') {
                last = p + 2;
            }
        }
        return last;
    }

    constexpr bool NameInKey(const PrimaryKeyItem& key, const char* name)
    {
        for (std::size_t i = 0; i < key.columnCount; ++i) {
            if (TextEquals(key.columns[i], name)) {
                return true;
            }
        }
        return false;
    }

    // ============================================================
    // 记录语义 hook：默认实现覆盖全部普通表；确有派生列或变长字节数组的
    // 记录在业务侧特化（仅允许这两类后端无关 hook 能力）
    // ============================================================
    /// 记录语义 hook 默认实现：普通表零特化直接使用
    template <typename T>
    struct DefaultRecordCodecHooks {
        /// 派生列编码：目录声明了 Derived 的字段由本函数计算编码值；返回 nullopt 回退成员原值
        static std::optional<DbScalar> EncodeDerivedField(const T& record, uint16_t fieldId)
        {
            (void)record;
            (void)fieldId;
            return std::nullopt;
        }

        /// 派生列解码忽略：默认全部回写；声明 Derived 的列通常返回 true（基址/大小为权威）
        static bool IgnoreDecodedField(uint16_t fieldId)
        {
            (void)fieldId;
            return false;
        }

        /// 有效数组长度：BLOB 数组的有效字节数由同记录另一字段决定时返回非负值；-1 表示固定容量
        static int64_t EffectiveBytesLength(const T& record, uint16_t fieldId)
        {
            (void)record;
            (void)fieldId;
            return -1;
        }
    };

    /// 记录语义 hook：业务侧特化时继承 DefaultRecordCodecHooks 并只覆盖需要的成员
    template <typename T>
    struct RecordCodecHooks : DefaultRecordCodecHooks<T> {};

    // ============================================================
    // 逗号表达式遍上下文：对同一 Body 的每次受限展开消费一种元素
    // ============================================================
    /// 第 1 遍：表头（版本/scope/主键/索引/外键；字段元素被吸收）
    template <typename T>
    struct HeadCtx {
        TableHead head{};
        /// 索引声明超过 descriptor 容量时置位；表条目 static_assert 关门，禁止
        /// 旧的 if(<limit) 行为静默丢掉新增索引。
        bool indexOverflow{false};
        /// PrimaryKey(...) 声明次数：每张表必须恰好一次（表条目 static_assert 关门；
        /// 两次声明不再静默由后写覆盖前写，零次声明同样拒绝）。
        unsigned primaryKeyDecls{0};

        constexpr HeadCtx& operator,(VersionItem<T>&& item)
        {
            head.version = item.value;
            return *this;
        }
        constexpr HeadCtx& operator,(ScopeKind scopeKind)
        {
            head.scope = scopeKind;
            return *this;
        }
        constexpr HeadCtx& operator,(PrimaryKeyItem&& item)
        {
            ++primaryKeyDecls;
            head.primaryKey = item;
            return *this;
        }
        constexpr HeadCtx& operator,(UniqueItem&& item)
        {
            if (head.indexCount >= kMaxIndexesPerTable) {
                indexOverflow = true;
                return *this;
            }
            IndexDesc desc{};
            for (std::size_t i = 0; i < item.columnCount; ++i) {
                desc.columns[i] = item.columns[i];
            }
            desc.columnCount = item.columnCount;
            desc.unique = true;
            head.indexes[head.indexCount] = desc;
            ++head.indexCount;
            return *this;
        }
        constexpr HeadCtx& operator,(IndexItem&& item)
        {
            if (head.indexCount >= kMaxIndexesPerTable) {
                indexOverflow = true;
                return *this;
            }
            IndexDesc desc{};
            for (std::size_t i = 0; i < item.columnCount; ++i) {
                desc.columns[i] = item.columns[i];
            }
            desc.columnCount = item.columnCount;
            head.indexes[head.indexCount] = desc;
            ++head.indexCount;
            return *this;
        }
        template <typename M, typename TargetT, typename TargetM>
        constexpr HeadCtx& operator,(ForeignKeyItem<T, M, TargetT, TargetM>&& item)
        {
            if (head.foreignKeyCount < kMaxForeignKeysPerTable) {
                head.foreignKeys[head.foreignKeyCount]
                    = ForeignKeyDesc{item.local.name, item.targetName, item.onDelete == OnDelete::Cascade};
                ++head.foreignKeyCount;
            }
            return *this;
        }
        template <typename M, typename Tag>
        constexpr HeadCtx& operator,(FieldItem<T, M, Tag>&&)
        {
            return *this;
        }
    };

    /// 第 2 遍：字段数组（主键列打标；属性与类型校验在此触发）
    template <typename T>
    struct FieldCtx {
        const PrimaryKeyItem* primaryKey{nullptr};
        std::array<FieldDesc, kMaxFieldsPerTable> fields{};
        std::size_t fieldCount{0};
        bool overflow{false};
        bool duplicateField{false};            ///< 同一字段名声明两次（掩盖遗漏成员）
        bool autoIncrementKeyViolation{false}; ///< AutoIncrement 字段不是唯一单列主键

        explicit constexpr FieldCtx(const PrimaryKeyItem& key) : primaryKey(&key) {}

        template <typename M, typename Tag>
        constexpr FieldCtx& operator,(FieldItem<T, M, Tag>&& item)
        {
            if (fieldCount >= kMaxFieldsPerTable) {
                overflow = true;
                return *this;
            }
            // 重复 Field(member) 检测：字段名严格等于成员名，重名字段即重复声明
            // （仅靠聚合 arity 门会被 "Field(x), Field(x)" 掩盖另一遗漏成员）。
            for (std::size_t i = 0; i < fieldCount; ++i) {
                if (TextEquals(fields[i].name, item.member.name)) {
                    duplicateField = true;
                }
            }
            // AutoIncrement 与 PrimaryKey 是两个独立概念：自增字段必须是该表唯一
            // 单列主键；普通主键不自动获得自增。
            if constexpr (HasProp(Tag::value, Prop::AutoIncrement)) {
                if (primaryKey == nullptr || primaryKey->columnCount != 1
                    || !NameInKey(*primaryKey, item.member.name)) {
                    autoIncrementKeyViolation = true;
                }
            }
            fields[fieldCount] = FieldDesc{
                item.member.name, TypeOfMember<M, Tag::value>(), PropertiesFor<M, Tag::value>::value,
                primaryKey != nullptr && NameInKey(*primaryKey, item.member.name), detail::MemberByteWidth<M>()};
            ++fieldCount;
            return *this;
        }
        template <typename X>
        constexpr FieldCtx& operator,(X&&)
        {
            return *this;
        }
    };

    /// 字段编号解析遍（成员指针 → 声明顺序编号；未登记成员返回 kUnknownFieldId）
    template <typename T, typename M>
    struct FieldIdOfCtx {
        M T::*target{nullptr};
        uint16_t found{kUnknownFieldId};
        std::size_t index{0};

        template <typename M2, typename Tag>
        constexpr FieldIdOfCtx& operator,(FieldItem<T, M2, Tag>&& item)
        {
            if constexpr (std::is_same_v<M, M2>) {
                if (found == kUnknownFieldId && target == item.member.ptr) {
                    found = static_cast<uint16_t>(index);
                }
            }
            ++index;
            return *this;
        }
        template <typename X>
        constexpr FieldIdOfCtx& operator,(X&&)
        {
            return *this;
        }
    };

    /// 主键读值遍（单列主键：主键名匹配的字段成员读值）
    template <typename T>
    struct PrimaryKeyReadCtx {
        const T* record{nullptr};
        const PrimaryKeyItem* primaryKey{nullptr};
        uint64_t value{0};
        bool seen{false};

        template <typename M, typename Tag>
        constexpr PrimaryKeyReadCtx& operator,(FieldItem<T, M, Tag>&& item)
        {
            if constexpr (std::is_integral_v<M> || std::is_enum_v<M>) {
                if (!seen && primaryKey != nullptr && NameInKey(*primaryKey, item.member.name)) {
                    value = static_cast<uint64_t>(record->*(item.member.ptr));
                    seen = true;
                }
            }
            return *this;
        }
        template <typename X>
        constexpr PrimaryKeyReadCtx& operator,(X&&)
        {
            return *this;
        }
    };

    /// 主键写值遍（单列主键：自增 id 写回）
    template <typename T>
    struct PrimaryKeyWriteCtx {
        T* record{nullptr};
        const PrimaryKeyItem* primaryKey{nullptr};
        uint64_t value{0};
        bool seen{false};

        template <typename M, typename Tag>
        PrimaryKeyWriteCtx& operator,(FieldItem<T, M, Tag>&& item)
        {
            if constexpr (std::is_integral_v<M> || std::is_enum_v<M>) {
                if (!seen && primaryKey != nullptr && NameInKey(*primaryKey, item.member.name)) {
                    record->*(item.member.ptr) = static_cast<M>(value);
                    seen = true;
                }
            }
            return *this;
        }
        template <typename X>
        PrimaryKeyWriteCtx& operator,(X&&)
        {
            return *this;
        }
    };

    /// 字段访问遍（visitRecord 实现）：逐字段把原记录成员以原始引用同步交给
    /// 后端。TEXT/BLOB 只传 data+bytes，不在目录层构造 DbScalar/字段值数组；
    /// 回调不得保存引用（同步调用栈内消费）。
    template <typename T>
    struct VisitRefCtx {
        const T* record{nullptr};
        void* context{nullptr};
        EmitFieldRef emit{nullptr};
        std::size_t fieldIndex{0};
        DbResult<void> status{DbCode::OK, 0, {}};

        VisitRefCtx(const T& source, void* targetContext, EmitFieldRef target)
            : record(&source),
              context(targetContext),
              emit(target)
        {
            if (emit == nullptr) {
                status = {DbCode::INVALID_ARGUMENT, 0, "record field emitter is null"};
            }
        }

        /// 派生列值以 DbScalar 的原生宽度发射（标量 8 字节/bool 1 字节；
        /// string/vector 按值载荷）。回调返回前同步消费，不产生拥有型副本。
        void EmitDerived(uint16_t fieldId, const DbScalar& value)
        {
            std::visit(
                [this, fieldId](const auto& scalar) {
                    using Scalar = std::decay_t<decltype(scalar)>;
                    if constexpr (std::is_same_v<Scalar, std::string> || std::is_same_v<Scalar, std::vector<uint8_t>>) {
                        status = emit(context, fieldId, scalar.data(), scalar.size(), false);
                    } else {
                        status = emit(context, fieldId, &scalar, sizeof(Scalar), false);
                    }
                },
                value);
        }

        template <typename M, typename Tag>
        VisitRefCtx& operator,(FieldItem<T, M, Tag>&& item)
        {
            const auto fieldId = static_cast<uint16_t>(fieldIndex++);
            if (!status.ok()) {
                return *this;
            }
            const auto& value = record->*(item.member.ptr);
            if constexpr (HasProp(Tag::value, Prop::Derived)) {
                auto derived = RecordCodecHooks<T>::EncodeDerivedField(*record, fieldId);
                if (derived.has_value()) {
                    EmitDerived(fieldId, *derived);
                    return *this;
                }
            }
            if constexpr (std::is_array_v<M>) {
                if constexpr (HasProp(Tag::value, Prop::Text)) {
                    // Text 数组以 sizeof(成员) 为界：遇 NUL 截断，满容量无 NUL 时
                    // 精确 sizeof(M) 字节，不读取相邻成员。
                    const char* text = reinterpret_cast<const char*>(&value);
                    std::size_t length = 0;
                    while (length < sizeof(M) && text[length] != '\0') {
                        ++length;
                    }
                    status = emit(context, fieldId, text, length, false);
                } else {
                    std::size_t bytes = sizeof(M);
                    const int64_t effective = RecordCodecHooks<T>::EffectiveBytesLength(*record, fieldId);
                    if (effective >= 0) {
                        bytes = std::min(bytes, static_cast<std::size_t>(effective));
                    }
                    status = emit(context, fieldId, &value, bytes, false);
                }
            } else if constexpr (detail::kIsOpaqueAggregateMember<M>) {
                status = emit(context, fieldId, &value, sizeof(M), false);
            } else if constexpr (std::is_same_v<M, std::string>) {
                status = emit(context, fieldId, value.data(), value.size(), false);
            } else if constexpr (std::is_same_v<M, std::vector<uint8_t>>) {
                status = emit(context, fieldId, value.data(), value.size(), false);
            } else {
                // 标量成员（整数/枚举/浮点/bool）按成员实际宽度发射；后端按
                // FieldDesc::type + bytes 解释。
                status = emit(context, fieldId, &value, sizeof(M), false);
            }
            return *this;
        }
        template <typename X>
        VisitRefCtx& operator,(X&&)
        {
            return *this;
        }
    };

    // ReadFieldInto 的目标成员直填遍（materializeRows 实现）：把最终 T 的成员
    // 地址交给后端（固定数组携带容量：BLOB 数组为 hook 有效长度约束后的容量），
    // 后端从数据库结果直接写入；目录层不出现中间值视图。
    template <typename T>
    struct FillCtx {
        void* context{nullptr};
        ReadFieldInto read{nullptr};
        T* value{nullptr};
        std::size_t fieldIndex{0};
        DbResult<void> status{DbCode::OK, 0, {}};

        FillCtx(void* sourceContext, ReadFieldInto reader, T& target)
            : context(sourceContext),
              read(reader),
              value(&target)
        {}

        template <typename M, typename Tag>
        FillCtx& operator,(FieldItem<T, M, Tag>&& item)
        {
            const auto fieldId = static_cast<uint16_t>(fieldIndex);
            ++fieldIndex;
            if (!status.ok() || RecordCodecHooks<T>::IgnoreDecodedField(fieldId)) {
                return *this;
            }
            auto& member = value->*(item.member.ptr);
            if constexpr (std::is_array_v<M>) {
                const int64_t effective = RecordCodecHooks<T>::EffectiveBytesLength(*value, fieldId);
                std::size_t capacity = sizeof(M);
                if (effective >= 0) {
                    capacity = std::min(capacity, static_cast<std::size_t>(effective));
                }
                status = read(context, fieldId, &member, capacity);
            } else {
                status = read(context, fieldId, &member, 0);
            }
            return *this;
        }
        template <typename X>
        FillCtx& operator,(X&&)
        {
            return *this;
        }
    };

} // namespace Ddl
} // namespace HcclSim::Storage

// ============================================================
// 生成器宏：属性宏固定展开为遍元素；表条目宏对同一 Body 多次受限展开。
// 宏体引用的 RecordType / Body 等名字在最终语法位置惰性解析。
// ============================================================
namespace HcclSim::Storage {
// 表条目生成前置声明（每张登记表生成一个特化）
namespace Ddl {
    template <typename T>
    struct TableDdl;

    /// 主键类型的内部别名：表条目宏体使用它避免与业务 DDL 区的 PrimaryKey(...) 短宏同名替换
    template <typename T>
    using Key = ::HcclSim::Storage::PrimaryKey<T>;
} // namespace Ddl
} // namespace HcclSim::Storage

#define HCCL_VM_DDL_UNWRAP(...) __VA_ARGS__

// ---- 属性宏（前缀版；业务 DDL 区以短别名转发到这里） ----
#define HCCL_VM_DDL_FIELD_1(m)          \
    ::HcclSim::Storage::Ddl::MakeField( \
        ::HcclSim::Storage::Ddl::MakeNamePtr(&RecordType::m, #m), ::HcclSim::Storage::Ddl::NoProps{})
#define HCCL_VM_DDL_FIELD_2(m, p1)      \
    ::HcclSim::Storage::Ddl::MakeField( \
        ::HcclSim::Storage::Ddl::MakeNamePtr(&RecordType::m, #m), ::HcclSim::Storage::Ddl::PropsTag<p1>{})
#define HCCL_VM_DDL_FIELD_3(m, p1, p2)  \
    ::HcclSim::Storage::Ddl::MakeField( \
        ::HcclSim::Storage::Ddl::MakeNamePtr(&RecordType::m, #m), ::HcclSim::Storage::Ddl::PropsTag<p1, p2>{})
#define HCCL_VM_DDL_FIELD_SEL(_1, _2, _3, N, ...) HCCL_VM_DDL_FIELD_##N
#define HCCL_VM_DDL_FIELD(...) HCCL_VM_DDL_FIELD_SEL(__VA_ARGS__, 3, 2, 1)(__VA_ARGS__)

#define HCCL_VM_DDL_VERSION(v) ::HcclSim::Storage::Ddl::MakeVersion<RecordType>(v)

#define HCCL_VM_DDL_PRIMARY_KEY_1(a) \
    ::HcclSim::Storage::Ddl::MakePrimaryKey(::HcclSim::Storage::Ddl::BindPrimaryKeyColumn(&RecordType::a, #a))
#define HCCL_VM_DDL_PRIMARY_KEY_2(a, b)                                    \
    ::HcclSim::Storage::Ddl::MakePrimaryKey(                               \
        ::HcclSim::Storage::Ddl::BindPrimaryKeyColumn(&RecordType::a, #a), \
        ::HcclSim::Storage::Ddl::BindPrimaryKeyColumn(&RecordType::b, #b))
#define HCCL_VM_DDL_PRIMARY_KEY_3(a, b, c)                                 \
    ::HcclSim::Storage::Ddl::MakePrimaryKey(                               \
        ::HcclSim::Storage::Ddl::BindPrimaryKeyColumn(&RecordType::a, #a), \
        ::HcclSim::Storage::Ddl::BindPrimaryKeyColumn(&RecordType::b, #b), \
        ::HcclSim::Storage::Ddl::BindPrimaryKeyColumn(&RecordType::c, #c))
#define HCCL_VM_DDL_PRIMARY_KEY_4(a, b, c, d)                              \
    ::HcclSim::Storage::Ddl::MakePrimaryKey(                               \
        ::HcclSim::Storage::Ddl::BindPrimaryKeyColumn(&RecordType::a, #a), \
        ::HcclSim::Storage::Ddl::BindPrimaryKeyColumn(&RecordType::b, #b), \
        ::HcclSim::Storage::Ddl::BindPrimaryKeyColumn(&RecordType::c, #c), \
        ::HcclSim::Storage::Ddl::BindPrimaryKeyColumn(&RecordType::d, #d))
#define HCCL_VM_DDL_PRIMARY_KEY_SEL(_1, _2, _3, _4, N, ...) HCCL_VM_DDL_PRIMARY_KEY_##N
#define HCCL_VM_DDL_PRIMARY_KEY(...) HCCL_VM_DDL_PRIMARY_KEY_SEL(__VA_ARGS__, 4, 3, 2, 1)(__VA_ARGS__)

#define HCCL_VM_DDL_UNIQUE_1(a) \
    ::HcclSim::Storage::Ddl::MakeUnique(::HcclSim::Storage::Ddl::BindKeyColumn(&RecordType::a, #a))
#define HCCL_VM_DDL_UNIQUE_2(a, b)                                  \
    ::HcclSim::Storage::Ddl::MakeUnique(                            \
        ::HcclSim::Storage::Ddl::BindKeyColumn(&RecordType::a, #a), \
        ::HcclSim::Storage::Ddl::BindKeyColumn(&RecordType::b, #b))
#define HCCL_VM_DDL_UNIQUE_3(a, b, c)                               \
    ::HcclSim::Storage::Ddl::MakeUnique(                            \
        ::HcclSim::Storage::Ddl::BindKeyColumn(&RecordType::a, #a), \
        ::HcclSim::Storage::Ddl::BindKeyColumn(&RecordType::b, #b), \
        ::HcclSim::Storage::Ddl::BindKeyColumn(&RecordType::c, #c))
#define HCCL_VM_DDL_UNIQUE_4(a, b, c, d)                            \
    ::HcclSim::Storage::Ddl::MakeUnique(                            \
        ::HcclSim::Storage::Ddl::BindKeyColumn(&RecordType::a, #a), \
        ::HcclSim::Storage::Ddl::BindKeyColumn(&RecordType::b, #b), \
        ::HcclSim::Storage::Ddl::BindKeyColumn(&RecordType::c, #c), \
        ::HcclSim::Storage::Ddl::BindKeyColumn(&RecordType::d, #d))
#define HCCL_VM_DDL_UNIQUE_SEL(_1, _2, _3, _4, N, ...) HCCL_VM_DDL_UNIQUE_##N
#define HCCL_VM_DDL_UNIQUE(...) HCCL_VM_DDL_UNIQUE_SEL(__VA_ARGS__, 4, 3, 2, 1)(__VA_ARGS__)

#define HCCL_VM_DDL_INDEX_1(a) \
    ::HcclSim::Storage::Ddl::MakeIndex(::HcclSim::Storage::Ddl::BindKeyColumn(&RecordType::a, #a))
#define HCCL_VM_DDL_INDEX_2(a, b)                                   \
    ::HcclSim::Storage::Ddl::MakeIndex(                             \
        ::HcclSim::Storage::Ddl::BindKeyColumn(&RecordType::a, #a), \
        ::HcclSim::Storage::Ddl::BindKeyColumn(&RecordType::b, #b))
#define HCCL_VM_DDL_INDEX_3(a, b, c)                                \
    ::HcclSim::Storage::Ddl::MakeIndex(                             \
        ::HcclSim::Storage::Ddl::BindKeyColumn(&RecordType::a, #a), \
        ::HcclSim::Storage::Ddl::BindKeyColumn(&RecordType::b, #b), \
        ::HcclSim::Storage::Ddl::BindKeyColumn(&RecordType::c, #c))
#define HCCL_VM_DDL_INDEX_4(a, b, c, d)                             \
    ::HcclSim::Storage::Ddl::MakeIndex(                             \
        ::HcclSim::Storage::Ddl::BindKeyColumn(&RecordType::a, #a), \
        ::HcclSim::Storage::Ddl::BindKeyColumn(&RecordType::b, #b), \
        ::HcclSim::Storage::Ddl::BindKeyColumn(&RecordType::c, #c), \
        ::HcclSim::Storage::Ddl::BindKeyColumn(&RecordType::d, #d))
#define HCCL_VM_DDL_INDEX_SEL(_1, _2, _3, _4, N, ...) HCCL_VM_DDL_INDEX_##N
#define HCCL_VM_DDL_INDEX(...) HCCL_VM_DDL_INDEX_SEL(__VA_ARGS__, 4, 3, 2, 1)(__VA_ARGS__)

#define HCCL_VM_DDL_FOREIGN_KEY(local, target, policy)                    \
    ::HcclSim::Storage::Ddl::MakeForeignKey(                              \
        ::HcclSim::Storage::Ddl::MakeNamePtr(&RecordType::local, #local), \
        ::HcclSim::Storage::Ddl::MakeNamePtr(&target, #target), #target, policy)

// ---- 表条目宏：生成 TableDdl/RecordSchema/RecordCodec 三位一体特化 ----
#define HCCL_VM_DDL_TABLE_ENTRY(Record, Body)                                                                                       \
    namespace HcclSim::Storage::Ddl {                                                                                               \
        template <>                                                                                                                 \
        struct TableDdl<Record> {                                                                                                   \
            using RecordType = Record;                                                                                              \
            static constexpr std::string_view kTableId = LastScopeName(#Record);                                                    \
            static constexpr std::string_view kDatabaseId = HCCL_VM_DDL_CURRENT_DATABASE;                                           \
            static constexpr HeadCtx<Record> kHeadScan = (HeadCtx<Record>{}, HCCL_VM_DDL_UNWRAP Body);                              \
            static constexpr TableHead kHead = kHeadScan.head;                                                                      \
            static constexpr FieldCtx<Record> kFieldScan                                                                            \
                = (FieldCtx<Record>{kHead.primaryKey}, HCCL_VM_DDL_UNWRAP Body);                                                    \
            static constexpr std::size_t kFieldCount = kFieldScan.fieldCount;                                                       \
            static constexpr auto kFields = kFieldScan.fields;                                                                      \
            static_assert(!kHeadScan.indexOverflow, "catalog table exceeds the index descriptor limit");                            \
            static_assert(IndexDescriptorsValid(kHead), "catalog index descriptor is structurally invalid");                        \
            static_assert(!kFieldScan.overflow, "catalog table exceeds the field limit");                                           \
            static_assert(kFieldCount > 0, "catalog table must declare at least one field");                                        \
            static_assert(!kFieldScan.duplicateField, "catalog table declares the same Field(member) twice");                       \
            static_assert(                                                                                                          \
                !kFieldScan.autoIncrementKeyViolation,                                                                              \
                "AutoIncrement fields must be the sole single-column PrimaryKey");                                                  \
            static_assert(kHeadScan.primaryKeyDecls == 1, "catalog tables must declare exactly one PrimaryKey(...)");               \
            static_assert(                                                                                                          \
                kHead.primaryKey.columnCount == 1, "catalog tables must declare a single-column PrimaryKey(...)");                  \
            static_assert(                                                                                                          \
                kDirectAggregateFieldCount<Record> == kFieldCount,                                                                  \
                "catalog field count drifted from the record aggregate; a member is not "                                           \
                "declared via Field(...)");                                                                                         \
            static constexpr uint64_t kFingerprint                                                                                  \
                = ComputeTableFingerprint(kTableId.data(), kHead, kFields.data(), kFieldCount);                                     \
            static const CatalogTableMeta& Meta();                                                                                  \
            template <typename M>                                                                                                   \
            static constexpr uint16_t FieldIdOf(M Record::*member)                                                                  \
            {                                                                                                                       \
                const auto scan = (FieldIdOfCtx<Record, M>{member}, HCCL_VM_DDL_UNWRAP Body);                                       \
                return scan.found;                                                                                                  \
            }                                                                                                                       \
            static Ddl::Key<Record> ReadPrimaryKey(const Record& record)                                                            \
            {                                                                                                                       \
                const auto scan = (PrimaryKeyReadCtx<Record>{&record, &kHead.primaryKey}, HCCL_VM_DDL_UNWRAP Body);                 \
                return Ddl::Key<Record>{scan.value};                                                                                \
            }                                                                                                                       \
            static DbResult<void> WritePrimaryKey(Record& record, Ddl::Key<Record> key)                                             \
            {                                                                                                                       \
                PrimaryKeyWriteCtx<Record> ctx{&record, &kHead.primaryKey, key.value};                                              \
                static_cast<void>((ctx, HCCL_VM_DDL_UNWRAP Body));                                                                  \
                return ctx.seen ? DbResult<void>{DbCode::OK, 0, {}} :                                                               \
                                  DbResult<void>{DbCode::INVALID_ARGUMENT, 0, "primary key member is not declared"};                \
            }                                                                                                                       \
        };                                                                                                                          \
    }                                                                                                                               \
    namespace HcclSim::Storage {                                                                                                    \
        template <>                                                                                                                 \
        struct RecordSchema<Record> {                                                                                               \
            static std::string_view TableId() { return Ddl::TableDdl<Record>::kTableId; }                                           \
            /* 表 → 逻辑库归属与 scope 类型（编译期；会话静态路由的单一来源） */                        \
            static std::string_view DatabaseId() { return Ddl::TableDdl<Record>::kDatabaseId; }                                     \
            static Ddl::ScopeKind ScopeKind() { return Ddl::TableDdl<Record>::kHead.scope; }                                        \
            static uint32_t SchemaVersion() { return Ddl::TableDdl<Record>::kHead.version; }                                        \
            static const Ddl::CatalogTableMeta& Meta() { return Ddl::TableDdl<Record>::Meta(); }                                    \
            template <typename M>                                                                                                   \
            static constexpr uint16_t FieldIdOf(M Record::*member)                                                                  \
            {                                                                                                                       \
                return Ddl::TableDdl<Record>::FieldIdOf(member);                                                                    \
            }                                                                                                                       \
        };                                                                                                                          \
        template <>                                                                                                                 \
        struct RecordCodec<Record> {                                                                                                \
            using RecordType = Record;                                                                                              \
            static Ddl::Key<Record> ReadPrimaryKey(const Record& record)                                                            \
            {                                                                                                                       \
                return Ddl::TableDdl<Record>::ReadPrimaryKey(record);                                                               \
            }                                                                                                                       \
            static DbResult<void> WritePrimaryKey(Record& record, Ddl::Key<Record> key)                                             \
            {                                                                                                                       \
                return Ddl::TableDdl<Record>::WritePrimaryKey(record, key);                                                         \
            }                                                                                                                       \
        };                                                                                                                          \
    }                                                                                                                               \
    namespace HcclSim::Storage::Ddl {                                                                                               \
        /* 类型擦除记录通道在 RecordSchema/RecordCodec 特化之后类外定义（lambda 体引用 */                      \
        /* 已特化的表契约产物，避免主模板在显式特化前被实例化）。*/                                      \
        inline const CatalogTableMeta& TableDdl<Record>::Meta()                                                                     \
        {                                                                                                                           \
            static const CatalogTableMeta kMeta = CatalogTableMeta{                                                                 \
                kTableId,                                                                                                           \
                kHead.version,                                                                                                      \
                kHead.scope,                                                                                                        \
                kFields.data(),                                                                                                     \
                kFieldCount,                                                                                                        \
                kHead.indexes.data(),                                                                                               \
                kHead.indexCount,                                                                                                   \
                kHead.foreignKeys.data(),                                                                                           \
                kHead.foreignKeyCount,                                                                                              \
                kHead.primaryKey.columns.data(),                                                                                    \
                kHead.primaryKey.columnCount,                                                                                       \
                kFingerprint,                                                                                                       \
                typeid(Record),                                                                                                     \
                typeid(std::vector<Record>),                                                                                        \
                [](ResultSink& sink, void* context, NextRow next, ReadFieldInto read) -> DbResult<void> {                           \
                    if (next == nullptr || read == nullptr) {                                                                       \
                        return {DbCode::INVALID_ARGUMENT, 0, "row source is incomplete"};                                           \
                    }                                                                                                               \
                    if (Record* single = sink.Target<Record>()) {                                                                   \
                        const auto hasRow = next(context);                                                                          \
                        if (!hasRow.ok()) {                                                                                         \
                            return {hasRow.code, hasRow.affectedRows, hasRow.diagnostic};                                           \
                        }                                                                                                           \
                        if (!*hasRow.value) {                                                                                       \
                            /* 单行未命中按合同 NOT_FOUND（不变）；诊断携带表名，使"缺哪张表"可区分  */ \
                            /* （地址翻译链 VMB 未命中与 VMB 命中/PhyMemBlock 主键未命中此前逐字同文案）*/ \
                            /* 文案在类型化通道目录层生成，SQLite/Redis 双后端一致。                 */         \
                            static const std::string kMissDiagnostic = "no matching row in " + std::string{kTableId};               \
                            return {DbCode::NOT_FOUND, 0, kMissDiagnostic};                                                         \
                        }                                                                                                           \
                        Ddl::FillCtx<Record> fill{context, read, *single};                                                          \
                        static_cast<void>((fill, HCCL_VM_DDL_UNWRAP Body));                                                         \
                        if (!fill.status.ok())                                                                                      \
                            return fill.status;                                                                                     \
                        return {DbCode::OK, 1, {}};                                                                                 \
                    }                                                                                                               \
                    if (std::vector<Record>* rows = sink.Target<std::vector<Record>>()) {                                           \
                        uint64_t count = 0;                                                                                         \
                        rows->clear(); /* replace caller result set, not append */                                                  \
                        while (true) {                                                                                              \
                            const auto hasRow = next(context);                                                                      \
                            if (!hasRow.ok()) {                                                                                     \
                                return {hasRow.code, hasRow.affectedRows, hasRow.diagnostic};                                       \
                            }                                                                                                       \
                            if (!*hasRow.value) {                                                                                   \
                                break;                                                                                              \
                            }                                                                                                       \
                            rows->emplace_back();                                                                                   \
                            Ddl::FillCtx<Record> fill{context, read, rows->back()};                                                 \
                            static_cast<void>((fill, HCCL_VM_DDL_UNWRAP Body));                                                     \
                            if (!fill.status.ok()) {                                                                                \
                                rows->pop_back();                                                                                   \
                                return fill.status;                                                                                 \
                            }                                                                                                       \
                            ++count;                                                                                                \
                        }                                                                                                           \
                        return {DbCode::OK, count, {}};                                                                             \
                    }                                                                                                               \
                    return {DbCode::INVALID_ARGUMENT, 0, "record result type does not match the sink"};                             \
                },                                                                                                                  \
                [](const void* record, void* context, EmitFieldRef emit) -> DbResult<void> {                                        \
                    if (record == nullptr) {                                                                                        \
                        return {DbCode::INVALID_ARGUMENT, 0, "record to visit is null"};                                            \
                    }                                                                                                               \
                    Ddl::VisitRefCtx<Record> ctx{*static_cast<const Record*>(record), context, emit};                               \
                    static_cast<void>((ctx, HCCL_VM_DDL_UNWRAP Body));                                                              \
                    return ctx.status;                                                                                              \
                },                                                                                                                  \
            };                                                                                                                      \
            return kMeta;                                                                                                           \
        }                                                                                                                           \
    }

#define HCCL_VM_DDL_COUNT_ENTRY(Type, Body) +1
#define HCCL_VM_DDL_META_ENTRY(Type, Body) ::HcclSim::Storage::Ddl::TableDdl<Type>::Meta(),

namespace HcclSim::Storage {
namespace Ddl {

    /// 外键目标名拆分：宏字符串化的 "TargetTable::member" → 目标表名/目标成员名
    /// （返回静态字符串视图；目标描述由关系层解析为列级映射）
    inline constexpr std::string_view TargetTableName(const char* target)
    {
        // "命名空间::表::成员" → 表名（成员段先剥掉，再取剩余部分的最后一段）
        std::string_view view{target};
        const auto memberSep = view.rfind("::");
        if (memberSep == std::string_view::npos) {
            return std::string_view{};
        }
        view = view.substr(0, memberSep);
        const auto tableSep = view.rfind("::");
        return tableSep == std::string_view::npos ? view : view.substr(tableSep + 2);
    }

    /// 外键目标名拆分：宏字符串化的 "TargetTable::member" → 目标成员名。
    /// 目标字段编号由关系装配层按目标表自己的目录解析（记录合同层只携带成员名身份，
    /// 不跨表编号）；本表与目标表的成员指针级绑定已由 ForeignKey 宏完成。
    inline constexpr std::string_view TargetMemberName(const char* target)
    {
        std::string_view view{target};
        const auto memberSep = view.rfind("::");
        return memberSep == std::string_view::npos ? std::string_view{} : view.substr(memberSep + 2);
    }

    /// 表 → 逻辑库归属（编译期只读静态数据）：值来自 TableDdl<T>::kDatabaseId——
    /// 由所在 HCCL_VM_DEFINE_DATABASE 的绑定命名空间经词法作用域解析写入。
    /// 会话静态路由据此按 T 直接定位逻辑库，不再维护逐表运行时绑定清单；
    /// 未登记目录的表没有 TableDdl 特化，使用 DatabaseOf<T> 直接编译失败。
    template <typename TableType>
    struct DatabaseOf {
        static constexpr std::string_view kDatabaseId = TableDdl<TableType>::kDatabaseId;
    };

    /// 多数据库目录聚合（Runner/OpData 等逻辑库的表闭包；表名全局唯一由装配期校验）
    template <std::size_t N>
    struct StorageCatalog {
        std::array<DatabaseDescriptor, N> databases{};

        constexpr std::size_t TableCount() const
        {
            std::size_t total = 0;
            for (const auto& database : databases) {
                total += database.tableCount;
            }
            return total;
        }

        constexpr const CatalogTableMeta* FindTable(std::string_view tableId) const
        {
            for (const auto& database : databases) {
                for (std::size_t i = 0; i < database.tableCount; ++i) {
                    if (database.tables[i].tableId == tableId) {
                        return &database.tables[i];
                    }
                }
            }
            return nullptr;
        }

        constexpr const DatabaseDescriptor* FindDatabase(std::string_view databaseId) const
        {
            for (const auto& database : databases) {
                if (database.databaseId == databaseId) {
                    return &database;
                }
            }
            return nullptr;
        }
    };

    inline constexpr StorageCatalog<0> kEmptyCatalog{};

    template <typename... Databases>
    constexpr auto MakeCatalog(const Databases&... databases)
    {
        return StorageCatalog<sizeof...(Databases)>{{databases...}};
    }

} // namespace Ddl
} // namespace HcclSim::Storage

/// 定义一个逻辑数据库的完整目录：展开全部表条目并发布只读 manifest。
/// Name 为数据库描述符常量名；DbId 为 canonical 逻辑库名字符串。
/// 表归属绑定：业务 DDL 区在调用本宏前局部 #define HCCL_VM_DDL_CURRENT_DATABASE
/// 为本库名字符串（与 Field/PrimaryKey 等短宏同一"区段内定义、区段末 #undef"模式）；
/// 每张表的 TableDdl<Type>::kDatabaseId（表 → 逻辑库，编译期，DatabaseOf<T> 消费）
/// 由该宏绑定，本宏尾部 static_assert 防止绑定值与目录 DbId 漂移。
#define HCCL_VM_DEFINE_DATABASE(Name, DbId, TablesMacro)                                                         \
    TablesMacro(HCCL_VM_DDL_TABLE_ENTRY) namespace HcclSim::Storage::Ddl                                         \
    {                                                                                                            \
        inline constexpr std::string_view k##Name##DatabaseId{#DbId};                                            \
        inline constexpr std::size_t k##Name##TableCount = 0 TablesMacro(HCCL_VM_DDL_COUNT_ENTRY);               \
        inline const CatalogTableMeta k##Name##Tables[]{TablesMacro(HCCL_VM_DDL_META_ENTRY)};                    \
        inline constexpr DatabaseDescriptor Name{std::string_view{#DbId}, k##Name##Tables, k##Name##TableCount}; \
        static_assert(                                                                                           \
            k##Name##DatabaseId == std::string_view{HCCL_VM_DDL_CURRENT_DATABASE},                               \
            "HCCL_VM_DDL_CURRENT_DATABASE binding drifted from the database catalog id");                        \
    }

#endif // HCCL_VM_STORAGE_CATALOG_H
