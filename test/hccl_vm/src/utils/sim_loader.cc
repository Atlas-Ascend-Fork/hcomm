/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sim_loader.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <set>
#include <string>
#include <unistd.h>
#include <vector>

#include "sim_common_defs.h"
#include "sim_log.h"
#include "sim_common_api.h"
#include "sim_yaml_config.h"
#include "operation_data/operation_data_ops.h"
#include "runtime_state/db_sim_runner_common.h"
#include "storage/table_access.h"

namespace loader {
Loader::Loader() {}

Loader::~Loader() {}

HcclResult Loader::LoadOpTaskFile(const std::string dbPath)
{
    sim::operation::DBConfig config;
    std::string targetPath;

    if (!dbPath.empty()) {
        if (!std::filesystem::exists(dbPath)) {
            HCCL_VM_ERROR("Backup database file not found: {}", dbPath);
            return HcclResult::HCCL_E_PARA;
        }
        targetPath = dbPath;
        HCCL_VM_INFO("Loading from specific backup path: {}", targetPath);
    } else {
        targetPath = InstallPath::ResolveToInstallRoot("data/hccl_vm_data.db");
        std::string absPath = std::filesystem::absolute(targetPath).string();
        HCCL_VM_INFO("Loading using default configuration path: {}", absPath);
    }

    config.dbPath = targetPath;
    sim::operation::SetDbConfig(config);
    opExecutionEntriesByKey_.clear();
    HCCL_VM_INFO("Loading using default configuration path:{}.", config.dbPath);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult Loader::LoadOpExecutionKeys(std::vector<sim::operation::OpExecutionKey>& keys)
{
    std::vector<sim::operation::OpExecutionIndexEntry> entries;
    if (sim::operation::QueryOpExecutionIndexEntries(entries) != 0) {
        HCCL_VM_ERROR("QueryOpExecutionIndexEntries failed");
        return HcclResult::HCCL_E_PARA;
    }

    keys.clear();
    opExecutionEntriesByKey_.clear();
    std::map<uint64_t, std::pair<std::string, uint64_t>> commIdentities;
    for (const sim::operation::OpExecutionIndexEntry& entry : entries) {
        auto commIdentityIt = commIdentities.find(entry.commId);
        if (commIdentityIt == commIdentities.end()) {
            std::string commName;
            uint64_t commHash = 0;
            if (!sim::runtime::GetCommunicatorIdentity(entry.commId, commName, commHash)) {
                HCCL_VM_ERROR("Failed to resolve communicator identity, commId={}", entry.commId);
                return HcclResult::HCCL_E_PARA;
            }
            commIdentityIt = commIdentities.emplace(entry.commId, std::make_pair(std::move(commName), commHash)).first;
        }

        sim::operation::OpExecutionKey key{commIdentityIt->second.first, commIdentityIt->second.second, entry.opIter};
        const OpExecutionIdentity identity{key.commName, key.commHash, key.opIter};
        auto [groupIt, inserted] = opExecutionEntriesByKey_.try_emplace(identity);
        if (inserted) {
            keys.push_back(key);
        }
        groupIt->second.push_back(entry);
    }

    HCCL_VM_INFO("Loaded operator execution keys, recordCount={}, opCount={}", entries.size(), keys.size());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
Loader::LoadOpExecutionByKey(const sim::operation::OpExecutionKey& key, sim::operation::OpExecution& opExecution)
{
    opExecution = sim::operation::OpExecution{};
    opExecution.key = key;
    const OpExecutionIdentity identity{key.commName, key.commHash, key.opIter};
    const auto indexIt = opExecutionEntriesByKey_.find(identity);
    if (indexIt == opExecutionEntriesByKey_.end()) {
        HCCL_VM_ERROR("Operator execution key was not found, commName={}, opIter={}", key.commName, key.opIter);
        return HcclResult::HCCL_E_PARA;
    }

    uint32_t rankSize = 0;
    std::set<uint32_t> devices;
    std::map<uint32_t, const sim::operation::OpExecutionIndexEntry*> entriesByRank;
    std::set<uint32_t> detailIds;
    for (const sim::operation::OpExecutionIndexEntry& entry : indexIt->second) {
        if (entry.rankSize == 0 || entry.rankId >= entry.rankSize) {
            HCCL_VM_ERROR(
                "Invalid rank in operator execution, commName={}, opIter={}, rankId={}, rankSize={}", key.commName,
                key.opIter, entry.rankId, entry.rankSize);
            return HcclResult::HCCL_E_PARA;
        }
        if (rankSize == 0) {
            rankSize = entry.rankSize;
        } else if (rankSize != entry.rankSize) {
            HCCL_VM_ERROR(
                "Inconsistent rankSize in operator execution, commName={}, opIter={}", key.commName, key.opIter);
            return HcclResult::HCCL_E_PARA;
        }
        if (!devices.insert(entry.deviceId).second) {
            HCCL_VM_ERROR(
                "Duplicate device in operator execution, commName={}, opIter={}, deviceId={}", key.commName, key.opIter,
                entry.deviceId);
            return HcclResult::HCCL_E_PARA;
        }
        if (!entriesByRank.emplace(entry.rankId, &entry).second) {
            HCCL_VM_ERROR(
                "Duplicate rank in operator execution, commName={}, opIter={}, rankId={}", key.commName, key.opIter,
                entry.rankId);
            return HcclResult::HCCL_E_PARA;
        }
        if (!detailIds.insert(entry.opDetailId).second) {
            HCCL_VM_ERROR(
                "Duplicate operator detail ID in execution, commName={}, opIter={}, opDetailId={}", key.commName,
                key.opIter, entry.opDetailId);
            return HcclResult::HCCL_E_PARA;
        }
    }

    if (rankSize == 0 || entriesByRank.size() != rankSize) {
        HCCL_VM_ERROR(
            "Incomplete operator execution, commName={}, opIter={}, expectedRankCount={}, actualRankCount={}",
            key.commName, key.opIter, rankSize, indexIt->second.size());
        return HcclResult::HCCL_E_PARA;
    }
    std::vector<sim::operation::CompositeOpDetail> details;
    if (sim::operation::QueryCompositeOpDetailByOpIter(key.commName, key.commHash, key.opIter, details) != 0) {
        HCCL_VM_ERROR(
            "QueryCompositeOpDetailByOpIter failed, commName={}, commHash={}, opIter={}", key.commName, key.commHash,
            key.opIter);
        return HcclResult::HCCL_E_PARA;
    }
    if (details.size() != entriesByRank.size()) {
        HCCL_VM_ERROR(
            "Operator execution records are incomplete, commName={}, opIter={}, expected={}, actual={}", key.commName,
            key.opIter, indexIt->second.size(), details.size());
        return HcclResult::HCCL_E_PARA;
    }

    std::set<uint32_t> ranks;
    for (const sim::operation::CompositeOpDetail& detail : details) {
        sim::operation::DeviceOpExecutionRecord record;
        record.deviceId = detail.deviceId;
        record.rankId = detail.rankId;
        record.detail = detail.detail;
        record.memInfo = detail.memInfo;
        // details 是本函数内的一次性数据，move 出 tasks 避免整份深拷贝。
        record.tasks = std::move(detail.tasks);
        const uint32_t rankId = record.rankId;
        const auto expectedIt = entriesByRank.find(rankId);
        if (expectedIt == entriesByRank.end() || record.detail.opIter != key.opIter
            || record.detail.rankSize != rankSize || !ranks.insert(rankId).second) {
            HCCL_VM_ERROR(
                "Invalid rank mapping in operator execution, commName={}, opIter={}, rankId={}", key.commName,
                key.opIter, rankId);
            return HcclResult::HCCL_E_PARA;
        }
        const sim::operation::OpExecutionIndexEntry& expected = *expectedIt->second;
        if (record.detail.id != expected.opDetailId || record.deviceId != expected.deviceId
            || record.detail.commId != expected.commId) {
            HCCL_VM_ERROR(
                "Loaded operator record does not match its index, commName={}, opIter={}, rankId={}", key.commName,
                key.opIter, rankId);
            return HcclResult::HCCL_E_PARA;
        }
        if (record.detail.id != record.memInfo.opDetailId && record.memInfo.opDetailId != 0) {
            HCCL_VM_ERROR(
                "Operator memory record does not match its detail, commName={}, opIter={}, rankId={}", key.commName,
                key.opIter, rankId);
            return HcclResult::HCCL_E_PARA;
        }
        for (const sim::operation::OpTaskTab& task : record.tasks) {
            if (task.opDetailId != record.detail.id) {
                HCCL_VM_ERROR(
                    "Operator task does not match its detail, commName={}, opIter={}, rankId={}, taskId={}",
                    key.commName, key.opIter, rankId, task.id);
                return HcclResult::HCCL_E_PARA;
            }
        }
        opExecution.deviceRecords.push_back(std::move(record));
    }
    std::sort(
        opExecution.deviceRecords.begin(), opExecution.deviceRecords.end(),
        [](const sim::operation::DeviceOpExecutionRecord& lhs, const sim::operation::DeviceOpExecutionRecord& rhs) {
            return lhs.rankId < rhs.rankId;
        });
    return HcclResult::HCCL_SUCCESS;
}

HcclResult Loader::LoadAllOpExecutions(std::vector<sim::operation::OpExecution>& opExecutions)
{
    std::vector<sim::operation::OpExecutionKey> keys;
    HcclResult ret = LoadOpExecutionKeys(keys);
    if (ret != HcclResult::HCCL_SUCCESS) {
        opExecutions.clear();
        return ret;
    }

    opExecutions.clear();
    opExecutions.reserve(keys.size());
    for (const sim::operation::OpExecutionKey& key : keys) {
        sim::operation::OpExecution opExecution;
        ret = LoadOpExecutionByKey(key, opExecution);
        if (ret != HcclResult::HCCL_SUCCESS) {
            opExecutions.clear();
            return ret;
        }
        opExecutions.push_back(std::move(opExecution));
    }

    // tasks 已由 LoadOpExecutionByKey -> QueryCompositeOpDetailByOpIter 按 opDetailId 读入
    // record.tasks；此处不再对 opTask 表做第二次整表读取与覆盖（省一次全表读 + 一份全量任务拷贝，
    // 见 docs/checker_load_translate_optimization.md §2.1「任务表被完整读取两遍」）。
    size_t taskCount = 0;
    for (const sim::operation::OpExecution& execution : opExecutions) {
        for (const sim::operation::DeviceOpExecutionRecord& record : execution.deviceRecords) {
            taskCount += record.tasks.size();
        }
    }
    HCCL_VM_INFO("Loaded all operator executions, operatorCount={}, taskCount={}", opExecutions.size(), taskCount);
    return HcclResult::HCCL_SUCCESS;
}

// 返回内部已组装好的完整 Pipeline 缓存
const OpPipeline& Loader::GetOpTasks() const { return opTaskCache_; }

// Runner: 每 sync 一次调用一次，只加载下一条 pending(status=0) 记录，根据业务传入的 syncIter，从 pending(status=0)
// 记录中查找并加载对应算子数据
HcclResult Loader::LoadRunnerSingleSync(
    const uint32_t& outSyncIter, std::map<uint32_t, std::vector<sim::operation::CompositeOpDetail>>& compositeDataMap)
{
    if (sim::operation::QueryCompositeOpDetailBySyncIter(outSyncIter, compositeDataMap) != 0) {
        HCCL_VM_ERROR("QueryCompositeOpDetailBySyncIter failed for syncIter: {}", outSyncIter);
        return HcclResult::HCCL_E_PARA;
    }
    HCCL_VM_INFO("syncIter={} loaded. Rank Count: {}", outSyncIter, compositeDataMap.size());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult Loader::GetCcuChannelInfo(std::vector<sim::operation::CcuChannelTab>& channels)
{
    if (sim::operation::QueryCcuChannelAll(channels) != 0) {
        HCCL_VM_ERROR("QueryCcuChannelAll failed");
        return HcclResult::HCCL_E_PARA;
    }
    HCCL_VM_INFO("Get {} channels", channels.size());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult Loader::GetHalfRTTInfo(std::vector<sim::operation::HalfRTTTab>& halfRTT)
{
    if (sim::operation::QueryHalfRTTAll(halfRTT) != 0) {
        HCCL_VM_ERROR("QueryHalfRTTAll failed");
        return HcclResult::HCCL_E_PARA;
    }
    HCCL_VM_INFO("Get {} half RTT records", halfRTT.size());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult Loader::GetJettyMapInfo(std::vector<sim::operation::JettyMapTab>& jettyMaps)
{
    if (sim::operation::QueryJettyMapAll(jettyMaps) != 0) {
        HCCL_VM_ERROR("QueryJettyMapAll failed");
        return HcclResult::HCCL_E_PARA;
    }
    HCCL_VM_INFO("Get {} jetty maps", jettyMaps.size());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult Loader::GetInstrResInfo(std::vector<sim::operation::CcuInstrResTab>& instrRes)
{
    if (sim::operation::QueryCcuInstrResAll(instrRes) != 0) {
        HCCL_VM_ERROR("QueryCcuInstrResAll failed");
        return HcclResult::HCCL_E_PARA;
    }
    HCCL_VM_INFO("Get {} instr res records", instrRes.size());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult Loader::GetSyncInfo(std::vector<sim::operation::SyncRecordTab>& syncRecords)
{
    if (sim::operation::QuerySyncRecordAll(syncRecords) != 0) {
        HCCL_VM_ERROR("QuerySyncRecordAll failed");
        return HcclResult::HCCL_E_PARA;
    }
    HCCL_VM_INFO("Get {} sync records", syncRecords.size());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult Loader::GetSyncRecordsByStatus(uint8_t status, std::vector<sim::operation::SyncRecordTab>& syncRecords)
{
    if (sim::operation::QuerySyncRecordByStatus(status, syncRecords) != 0) {
        HCCL_VM_ERROR("QuerySyncRecordByStatus failed for status: {}", status);
        return HcclResult::HCCL_E_PARA;
    }
    HCCL_VM_INFO("Get {} sync records for status: {}", syncRecords.size(), status);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult Loader::LoadCompositeOpDetailBySyncIter(
    uint32_t syncIter, std::map<uint32_t, std::vector<sim::operation::CompositeOpDetail>>& compositeDataMap)
{
    if (sim::operation::QueryCompositeOpDetailBySyncIter(syncIter, compositeDataMap) != 0) {
        HCCL_VM_ERROR("QueryCompositeOpDetailsBySyncIter failed for syncIter: {}", syncIter);
        return HcclResult::HCCL_E_PARA;
    }
    HCCL_VM_INFO("syncIter={}, Rank Count: {}", syncIter, compositeDataMap.size());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult Loader::LoadAllOpTasks(std::vector<sim::operation::OpTaskTab>& tasks, bool filterDone)
{
    // 上游语义：枚举全部动态 opTask 表（可选过滤未完成记录），经 opdb 域接口执行。
    if (sim::operation::QueryAllOpTasks(tasks, filterDone) != 0) {
        HCCL_VM_ERROR("QueryAllOpTasks failed");
        return HcclResult::HCCL_E_PARA;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult Loader::LoadCompositeOpDetailByOpIter(
    const std::string& commName, uint64_t commHash, uint32_t opIter,
    std::vector<sim::operation::CompositeOpDetail>& details)
{
    if (sim::operation::QueryCompositeOpDetailByOpIter(commName, commHash, opIter, details) != 0) {
        HCCL_VM_ERROR("QueryCompositeOpDetailByOpIter failed");
        return HcclResult::HCCL_E_PARA;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult Loader::FinishOpTask(const sim::operation::OpTaskTab& task)
{
    if (sim::operation::FinishOpTask(task) != 0) {
        HCCL_VM_ERROR("FinishOpTask failed");
        return HcclResult::HCCL_E_PARA;
    }
    return HcclResult::HCCL_SUCCESS;
}

} // namespace loader
