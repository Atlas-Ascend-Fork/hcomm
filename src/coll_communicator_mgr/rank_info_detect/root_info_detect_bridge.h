/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_ROOT_INFO_DETECT_BRIDGE_H
#define HCOMM_ROOT_INFO_DETECT_BRIDGE_H

#include <memory>

#include "hccl/hccl_types.h"
#include "rank_table_info.h"
#include "root_handle_v2.h"

#include <cstdint>

namespace Hccl {

// 修改回调签名或表布局时同步升级。
constexpr uint32_t ROOT_INFO_DETECT_BRIDGE_VERSION = 3;

// 同一 HCOMM 包内的内部函数表，两端随包一起构建、更新；注册时校验版本和长度。
// 链接方向为 hcomm -> hccl_v2：Register/Get 与槽位符号归属 hccl_v2；运行时回调方向为
// hccl_v2 -> hcomm：兼容入口通过该内部函数表调用 RootInfoDetect provider。不得将槽位移入 hcomm 而引入
// hccl_v2 -> hcomm 的直接链接；待后续legacy功能全部迁移，可以考虑去掉bridge和provider
struct RootInfoDetectBridge {
    // 调用方持有该类型擦除的所有权句柄，保证探测对象在通信域初始化结束前不被析构。
    using DetectContext = std::shared_ptr<void>;
    using GetRootInfoFunc = HcclResult (*)(HcclRootInfo* rootInfo, DetectContext& detectContext);
    // 仅传入口已冻结的32字节选层字段；nullptr保持默认行为，不重新读取环境变量。
    using DetectRankTableFunc = HcclResult (*)(
        u32 nRanks, u32 rank, const HcclRootHandleV2& rootHandle, RankTableInfo& rankTable,
        DetectContext& detectContext, const char* netLayerExpression);
    // scalable 多 root 建链回调（追加槽位，向后兼容：旧 provider 仅注册前两槽时两槽为 nullptr，
    // 入口侧转换为 HCCL_E_INTERNAL 明确报错）。
    using GetRootInfoScalableFunc = HcclResult (*)(HcclRootInfo* rootInfo);
    using DetectRankTableScalableFunc = HcclResult (*)(
        u32 nRanks, u32 nRoot, const HcclRootInfo* rootInfoList, u32 rank, RankTableInfo& rankTable,
        DetectContext& detectContext);

    GetRootInfoFunc getRootInfo{nullptr};
    DetectRankTableFunc detectRankTable{nullptr};
    GetRootInfoScalableFunc getRootInfoScalable{nullptr};
    DetectRankTableScalableFunc detectRankTableScalable{nullptr};

    // root 重选时，用父通信域的 topParentIdentifier + generationIndex+1 构造 identifier
    // detectContext 输出 detect 所有权，调用方（HcclCommGetRootInfo）存入 CollComm 供 Grow 使用。
    using GetRootInfoWithDerivationFunc = HcclResult (*)(
        HcclRootInfo* rootInfo, const char* topParentId, uint32_t generationIndex, DetectContext& detectContext);
    GetRootInfoWithDerivationFunc getRootInfoWithDerivation{nullptr};
};

// hcomm 加载时注册一次完整回调表；注册完成前查询返回 nullptr。
HcclResult RegisterRootInfoDetectBridge(const RootInfoDetectBridge& bridge, uint32_t version, uint32_t size);
const RootInfoDetectBridge* GetRootInfoDetectBridge();
// scalable 槽视角的查询：基础槽已注册且 scalable 两槽齐备才返回非空（旧 provider 返回 nullptr）。
const RootInfoDetectBridge* GetRootInfoDetectBridgeScalable();

} // namespace Hccl

#endif // HCOMM_ROOT_INFO_DETECT_BRIDGE_H
