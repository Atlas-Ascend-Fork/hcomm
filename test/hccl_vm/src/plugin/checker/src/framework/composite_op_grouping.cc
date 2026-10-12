/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "composite_op_grouping.h"

#include <algorithm>
#include <limits>
#include <map>
#include <tuple>
#include <utility>
#include <vector>

#include "runtime_state/db_sim_communicator.h"
#include "sim_log.h"

namespace HcclSim {
namespace {

    bool GetCommunicatorIdentity(
        uint64_t commId, std::map<uint64_t, std::pair<std::string, uint64_t>>& commIdentities, std::string& commName,
        uint64_t& commHash)
    {
        const auto iter = commIdentities.find(commId);
        if (iter != commIdentities.end()) {
            commName = iter->second.first;
            commHash = iter->second.second;
            return true;
        }

        if (!sim::runtime::GetCommunicatorIdentity(commId, commName, commHash)) {
            HCCL_VM_ERROR("cannot resolve communicator identity for member commId={}", commId);
            return false;
        }
        commIdentities.emplace(commId, std::make_pair(commName, commHash));
        return true;
    }

    uint32_t GetFirstOpDetailId(const CompositeOpGroup& opGroup)
    {
        uint32_t firstId = std::numeric_limits<uint32_t>::max();
        for (const auto& entry : opGroup) {
            firstId = std::min(firstId, entry.second.detail.id);
        }
        return firstId;
    }

} // namespace

HcclResult GroupCompositeOpDetails(
    const std::map<uint32_t, std::vector<sim::operation::CompositeOpDetail>>& compositeData,
    std::vector<CompositeOpGroup>& opGroups)
{
    opGroups.clear();

    std::vector<const sim::operation::CompositeOpDetail*> operators;
    for (const auto& rankEntry : compositeData) {
        for (const sim::operation::CompositeOpDetail& op : rankEntry.second) {
            operators.push_back(&op);
        }
    }
    // 旧加载器先按通信域内 rankId 分桶，需按全局 opDetail.id 恢复原始执行顺序。
    std::sort(operators.begin(), operators.end(), [](const auto* lhs, const auto* rhs) {
        return lhs->detail.id < rhs->detail.id;
    });

    using OpIdentity = std::tuple<std::string, uint64_t, uint32_t>;
    std::map<OpIdentity, CompositeOpGroup> groupedOperators;
    std::map<uint64_t, std::pair<std::string, uint64_t>> commIdentities;
    for (const sim::operation::CompositeOpDetail* op : operators) {
        std::string commName;
        uint64_t commHash;
        if (!GetCommunicatorIdentity(op->commId, commIdentities, commName, commHash)) {
            return HCCL_E_PARA;
        }

        CompositeOpGroup& group = groupedOperators[{commName, commHash, op->detail.opIter}];
        group.emplace(op->deviceId, *op);
    }

    std::vector<std::pair<uint32_t, CompositeOpGroup>> orderedGroups;
    orderedGroups.reserve(groupedOperators.size());
    for (auto& entry : groupedOperators) {
        orderedGroups.emplace_back(GetFirstOpDetailId(entry.second), std::move(entry.second));
    }
    // 按组内最早的 opDetail.id 输出，保持跨通信域的原始执行顺序。
    std::sort(orderedGroups.begin(), orderedGroups.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.first < rhs.first;
    });
    opGroups.reserve(orderedGroups.size());
    for (auto& entry : orderedGroups) {
        opGroups.emplace_back(std::move(entry.second));
    }
    return HCCL_SUCCESS;
}

} // namespace HcclSim
