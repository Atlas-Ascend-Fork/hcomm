/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RMA_BUF_ENTRY_LITE_H
#define RMA_BUF_ENTRY_LITE_H

#include "binary_stream.h"

namespace Hccl {

// 通道内存条目：host 侧按此字段序打包，device 侧按同一字段序解包后录进 transport 的表。
// 字段与顺序对齐 device 侧 RmtUbBufLite 及既有的 uniqueId 解析（UbTransportLiteImpl::ParseRmtBufferVec）
struct RmaBufEntryLite {
    u64 addr{0};
    u64 size{0};
    u32 tokenId{0};
    u32 tokenValue{0};
    u32 notifyId{UINT32_MAX};
};

// 打包 / 解包共用同一份字段序，避免 host 打包与 device 解析两处各写一遍而漂移
inline void PackRmaBufEntry(const RmaBufEntryLite& entry, BinaryStream& binaryStream)
{
    binaryStream << entry.addr << entry.size << entry.tokenId << entry.tokenValue << entry.notifyId;
}

inline void UnpackRmaBufEntry(BinaryStream& binaryStream, RmaBufEntryLite& entry)
{
    binaryStream >> entry.addr >> entry.size >> entry.tokenId >> entry.tokenValue >> entry.notifyId;
}

} // namespace Hccl

#endif // RMA_BUF_ENTRY_LITE_H
