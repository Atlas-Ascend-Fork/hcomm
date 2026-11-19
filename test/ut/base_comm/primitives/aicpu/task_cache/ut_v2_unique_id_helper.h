/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_UT_V2_UNIQUE_ID_HELPER_H
#define HCOMM_UT_V2_UNIQUE_ID_HELPER_H

#include <cstdint>
#include <vector>

#include "binary_stream.h"
#include "endpoint_pair/channel/ub_unique_id_pub.h" // UB_UNIQUE_ID_EXT_*：V2 扩展区契约（零依赖薄头）
#include "mem_transport_common.h" // Hccl::TransportType：与既有 UT 传递包含（ub_transport_lite_impl.h 链）同源同拼法

// 构造最小合法 V2 uniqueId（扩展区头 + 表指针 0=无表 + 6 个空 vector），供 Init 路径反序列化。
// 原 task_cache 三个 UT 各持一份逐字重复的 static 构造器，收敛为单一定义（检视整改）：
// V2 布局契约（魔数/版本/表指针位序）变更时只改此处，避免漏改某个 binary 产生仅该处复现的解析差异
inline std::vector<char> MakeEmptyUbV2UniqueId()
{
    Hccl::BinaryStream bs;
    bs << static_cast<uint32_t>(Hccl::TransportType::UB);
    bs << static_cast<uint32_t>(0); // notifyNum
    bs << static_cast<uint32_t>(0); // bufferNum
    bs << static_cast<uint32_t>(0); // rmtbufferNum
    bs << static_cast<uint32_t>(0); // connNum
    bs << UB_UNIQUE_ID_EXT_MAGIC_WORD;
    bs << UB_UNIQUE_ID_EXT_VERSION;
    bs << static_cast<uint64_t>(0); // devEpMemTablePtr，0=无表
    std::vector<char> empty;
    bs << empty; // notifyUniqueIds
    bs << empty; // rmtNotifyUniqueIds
    bs << empty; // locBufferUniqueIds
    bs << empty; // rmtBufferUniqueIds
    bs << empty; // drainBufferUniqueIds
    bs << empty; // connUniqueIds
    std::vector<char> data;
    bs.Dump(data);
    return data;
}

#endif // HCOMM_UT_V2_UNIQUE_ID_HELPER_H
