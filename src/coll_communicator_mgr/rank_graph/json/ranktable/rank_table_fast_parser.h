/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_RANK_TABLE_FAST_PARSER_H
#define HCCLV2_RANK_TABLE_FAST_PARSER_H

#include <cstddef>
#include <string>
#include "rank_table_info.h"
#include "rank_table_keys.h"

namespace Hccl {

// ranktable JSON 文件大小上限：1GB。
inline constexpr u64 RANKTABLE_MAX_SIZE = 1ULL * 1024 * 1024 * 1024;

// 快速解析器的编译期参数。集中放在对应头文件便于查看和调整，不导出符号。
// rank_count 触发预分配时的每 rank 最小字节估算值，避免伪造的大 rank_count 造成巨量内存预留。
inline constexpr std::size_t kMinBytesPerRank = 64;
// JSON 嵌套深度上限，防止恶意输入递归爆栈。
inline constexpr int kMaxJsonDepth = 256;
// 并行解析时每个任务的目标字节数。
inline constexpr std::size_t kChunkTargetBytes = 1u * 1024u * 1024u;
// 并行解析的最大线程数（硬件核数不足时取硬件核数）。
inline constexpr std::size_t kMaxParseThreads = 16;
// 每 32MB 输入分配一个解析线程。
inline constexpr std::size_t kBytesPerParseThread = 32u * 1024u * 1024u;

// RankTable JSON 快速解析：单遍扫描直接填充 RankTableInfo，不构建 DOM；失败返回 HCCL_E_PARA。
// 与 DOM 路径存在少量语义差异：整数按纯十进制解析，未知字段只做词法扫描。
HcclResult ParseRankTableStringFast(const std::string& jsonString, RankTableInfo& info);

// Reader 路径使用：只解析，不执行 RankTableInfo::Check()。
HcclResult ParseRankTableStringFastNoCheck(const std::string& jsonString, RankTableInfo& info);

// 仅供 UT 观测: 返回并行解析路径累计成功次数(HCCL_FAST_PARSER_THREADS 可强制线程数),
// 用于断言并行路径确实被触发。
u64 GetFastParserParallelUsedForTest();

} // namespace Hccl

#endif // HCCLV2_RANK_TABLE_FAST_PARSER_H
