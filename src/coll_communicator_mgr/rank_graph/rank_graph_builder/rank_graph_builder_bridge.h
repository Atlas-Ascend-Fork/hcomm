/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_RANK_GRAPH_BUILDER_BRIDGE_H
#define HCOMM_RANK_GRAPH_BUILDER_BRIDGE_H

#include <memory>
#include <string>

#include "hccl/hccl_types.h"
#include "rank_gph.h"
#include "rank_table_info.h"
#include "topo_info.h"

namespace Hccl {

// 跨 SO 返回的一次完整构建结果，调用方应将三项数据作为同一份拓扑快照使用。
struct RankGraphBuildResult {
    std::shared_ptr<RankGraph> rankGraph;
    RankTableInfo rankTableInfo;
    hccl::TopoInfo topoInfo;
};

// 修改回调签名（含入参所有权语义）或表布局时同步升级。
constexpr uint32_t RANK_GRAPH_BUILDER_BRIDGE_VERSION = 3;

// 同一 HCOMM 包内 hccl_v2 与 hcomm 之间的函数指针表，两端随包一起构建、更新。
// 本结构只描述调用签名，不包含 RankGraphBuilder 实现；版本与长度在注册时校验。
// 链接方向为 hcomm -> hccl_v2：Register/Get 与槽位符号归属 hccl_v2；运行时回调方向为
// hccl_v2 -> hcomm：legacy 从槽位取得 provider 函数地址后间接调用。不得通过移动槽位引入 hccl_v2 对 hcomm
// 的直接链接；若后续需要解除现有链接，应改用 dlsym 或独立注册层替换 Register/Get 直接符号。
struct RankGraphBuilderBridge {
    // netLayerExpression仅传已冻结的32字节选层字段；空串保持默认行为，不重新读取环境。
    // 从序列化 rank table 构建拓扑。
    using BuildFromStringFunc = HcclResult (*)(
        const std::string& rankTable, const std::string& topoPath, RankId myRank, const char* netLayerExpression,
        RankGraphBuildResult& result);
    // 从 RootInfoDetect 已解析的 RankTableInfo 构建拓扑。
    using BuildFromRankTableFunc = HcclResult (*)(
        const RankTableInfo& rankTable, const std::string& topoPath, RankId myRank, const char* netLayerExpression,
        RankGraphBuildResult& result);
    // 从故障恢复快照重建拓扑，对应 provider 的 RecoverBuildImpl。
    using RecoverBuildFunc = HcclResult (*)(
        RankTableInfo&& rankTable, const hccl::TopoInfo& topoInfo, RankId myRank, bool enableVirtualGroups,
        const char* netLayerExpression, RankGraphBuildResult& result);
    // 将 unique_ptr 所有权交给 hcomm，并返回携带 hcomm 侧 deleter 的 shared_ptr。
    using AdoptRankGraphFunc
        = HcclResult (*)(std::unique_ptr<RankGraph> rankGraph, std::shared_ptr<RankGraph>& sharedRankGraph);
    using CreateSubGraphFunc = HcclResult (*)(
        const RankGraph& parent, const std::vector<u32>& rankIds, const char* netLayerExpression,
        std::unique_ptr<RankGraph>& result);

    BuildFromStringFunc buildFromString{nullptr};
    BuildFromRankTableFunc buildFromRankTable{nullptr};
    RecoverBuildFunc recoverBuild{nullptr};
    AdoptRankGraphFunc adoptRankGraph{nullptr};
    CreateSubGraphFunc createSubGraph{nullptr};
};

// provider 的静态 registrar 在 hcomm 装载阶段调用 Register；legacy 在运行阶段通过 Get 取得已注册的函数表。
HcclResult RegisterRankGraphBuilderBridge(const RankGraphBuilderBridge& bridge, uint32_t version, uint32_t size);
const RankGraphBuilderBridge* GetRankGraphBuilderBridge();

} // namespace Hccl

#endif // HCOMM_RANK_GRAPH_BUILDER_BRIDGE_H
