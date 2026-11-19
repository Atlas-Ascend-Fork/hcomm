/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_UB_UNIQUE_ID_PUB_H
#define HCOMM_UB_UNIQUE_ID_PUB_H

#include <cstdint>

// ===== UB 建链序列化（uniqueIdV2）扩展区版本契约 =====
// 完整布局（位置式字段之外唯一的自描述段，插入在 connNum 与扩展字段之间）：
//   [type u32][notifyNum u32][bufferNum u32][rmtbufferNum u32][connNum u32]
//   [extMagic u32][extVersion u32]   ← 扩展区头，由本段常量定义
//   [devEpMemTablePtr u64]           ← extVersion=1：endpoint 活账本表 device 地址，0=无表（值语义，非格式判别）
//   [notify][rmtNotify][locBuffer][rmtBuffer][drain][conn] 6 段 vector（BinaryStream 前缀长度格式）
// 生产端：UbMemTransport::GetUniqueIdV2 / AicpuTsUboeUbRtpChannelHelper::GetUniqueIdV2；
// 消费端：UbTransportLiteImpl::Init（device）、AicpuChannelEntityResources::Build（host 预解析）。
// 魔数不可能与 V1 旧格式碰撞：V1 在 connNum 后是 vector 长度前缀（size_t，实际为几十~几 KB），
// 远小于 0x55424558（≈1.4GB）。消费端据此区分"版本化数据"与"旧/异构格式"，错配快速失败。
// 约束：本头零外部依赖（仅 <cstdint>），禁止引入 legacy 头（如 910 pub_inc 的 hccl_common/
// transport_pub 等，与 950 头在部分 UT target 二义）；channel_param_pub.h 经本头转发同一常量。
constexpr uint32_t UB_UNIQUE_ID_EXT_MAGIC_WORD = 0x55424558; // 'UBEX'
constexpr uint32_t UB_UNIQUE_ID_EXT_VERSION = 1;

#endif // HCOMM_UB_UNIQUE_ID_PUB_H
