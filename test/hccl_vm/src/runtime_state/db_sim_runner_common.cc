/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <algorithm>
#include <arpa/inet.h>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include "runtime_state/db_sim_runner_common.h"

#include "sim_log.h"
#include "storage/table_access.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "runtime_state/sim_models.h"
#include "store/store_sim_run_mode.h"
#include "sim_common_api.h"
#include <json.hpp>

uint64_t g_cur_server_key = 0;

namespace sim::runtime {
thread_local uint64_t g_cur_device_key = 0;
thread_local uint64_t g_cur_comm_key = 0;

// 进程级通信域配置缓存: mock-comm执行时经RefreshTopoMetaConfig刷新,
// 其余进程首次使用时经LoadCommConfigData按TopoMetaConfig表项加载
CommConfigData g_comm_config_data;

namespace {
    constexpr uint32_t kCommunicatorWaitIntervalMs = 100;
    // 销毁同步(MarkAndSyncCommunicatorDestroy)独立预算: 固定 300s(64p+ 实测 60s 不够),
    // 不随 HCCL_VM_COMMUNICATOR_WAIT_TIMEOUT_MS 覆盖。
    constexpr uint32_t kCommDestroyWaitTimeoutMs = 300000;
    // 默认等待预算保持既有行为(60s); 超时预算可经环境变量整体覆盖, 见 GetCommunicatorWaitBudgetMs。
    constexpr uint32_t kCommunicatorWaitDefaultTimeoutMs = 60000;
    constexpr uint32_t kCommunicatorWaitMinTimeoutMs = 1000;
    constexpr uint32_t kCommunicatorWaitMaxTimeoutMs = 86400000;

    /// 通信域等待预算(毫秒): 初始化(WaitCommunicatorReady)与销毁(WaitCommunicatorDestroyReady)
    /// 共用同一解析, 各调用点不单独改常量。环境变量 HCCL_VM_COMMUNICATOR_WAIT_TIMEOUT_MS
    /// 单位毫秒, 合法范围 [1000, 86400000] (1s ~ 24h); 未设置用默认 60000,
    /// 非法/越界告警一次并回退默认。首次解析时输出实际生效值(仅此一次)。
    uint32_t GetCommunicatorWaitBudgetMs()
    {
        static const uint32_t budgetMs = []() -> uint32_t {
            const char* envValue = std::getenv("HCCL_VM_COMMUNICATOR_WAIT_TIMEOUT_MS");
            if (envValue != nullptr && envValue[0] != '\0') {
                char* endptr = nullptr;
                const long value = std::strtol(envValue, &endptr, 10);
                if (endptr != envValue && (*endptr == '\0' || *endptr == '\n')
                    && value >= static_cast<long>(kCommunicatorWaitMinTimeoutMs)
                    && value <= static_cast<long>(kCommunicatorWaitMaxTimeoutMs)) {
                    HCCL_VM_INFO(
                        "communicator wait budget: {} ms (source: HCCL_VM_COMMUNICATOR_WAIT_TIMEOUT_MS)", value);
                    return static_cast<uint32_t>(value);
                }
                HCCL_VM_WARN(
                    "invalid HCCL_VM_COMMUNICATOR_WAIT_TIMEOUT_MS value '{}' (expect integer ms in "
                    "[{}, {}]), fallback to default {} ms",
                    envValue, kCommunicatorWaitMinTimeoutMs, kCommunicatorWaitMaxTimeoutMs,
                    kCommunicatorWaitDefaultTimeoutMs);
            }
            return kCommunicatorWaitDefaultTimeoutMs;
        }();
        return budgetMs;
    }

    /// 失败摘要用: 末次可见性快照中缺失 rank 的数量与前若干个 id。
    std::string SummarizeMissingRanks(const std::vector<bool>& rankSeen)
    {
        uint32_t missing = 0;
        std::string firstIds;
        for (uint32_t rank = 0; rank < rankSeen.size(); ++rank) {
            if (!rankSeen[rank]) {
                ++missing;
                if (firstIds.size() < 48U) { // 只承载前若干个 id, 防超长日志
                    if (!firstIds.empty()) {
                        firstIds += ",";
                    }
                    firstIds += std::to_string(rank);
                }
            }
        }
        return "count=" + std::to_string(missing) + " firstIds=[" + firstIds + "]";
    }

    bool IsValidCommunicatorName(const char* commName)
    {
        // 通信域名称存储在定长数组中，必须在边界内以 '\0' 结束。
        return commName != nullptr
               && ::strnlen(commName, sizeof(sim::runtime::Communicator::comm_id))
                      < sizeof(sim::runtime::Communicator::comm_id);
    }

    const char* SafeCommunicatorName(const char* commName)
    {
        // 日志不能将未终止的定长数组直接按 C 字符串输出。
        return IsValidCommunicatorName(commName) ? commName : "<invalid>";
    }

    std::string CommunicatorNameOf(const char* commId)
    {
        // 定长数组转字符串：只承载边界内的有效字节，避免未终止数组越界读。
        return std::string(commId, ::strnlen(commId, sizeof(sim::runtime::Communicator::comm_id)));
    }

    bool IsSameCommunicator(const sim::runtime::Communicator& record, const char* commName, uint64_t commHash)
    {
        return std::strncmp(record.comm_id, commName, sizeof(record.comm_id)) == 0 && record.comm_hash == commHash;
    }

    HcclSim::Storage::Condition<Communicator> CommunicatorIdentityCondition(const char* commName, uint64_t commHash)
    {
        return HcclSim::Storage::And(
            HcclSim::Storage::Eq(&Communicator::comm_id, CommunicatorNameOf(commName)),
            HcclSim::Storage::Eq(&Communicator::comm_hash, commHash));
    }

    using json = nlohmann::json;

    // 进程级通信域配置缓存锁与加载标记(仅本翻译单元访问)
    std::mutex g_comm_config_mutex;
    bool g_comm_config_loaded = false;

    // ranktable模式下(proxy进程)解析data/ranktable.json重建topoMeta:
    // 仅支持net_instance_id中携带sp/srv信息的ranktable(与宿主侧ParseRanktable主路径一致),
    // 旧格式依赖宿主侧network_索引回退的场景不支持, 无法定位的rank直接跳过
    bool ParseTopoMetaFromRankTableFile(TopoMeta& topoMeta)
    {
        std::string ranktablePath = InstallPath::ResolveToInstallRoot("data/ranktable.json");
        std::ifstream ifs(ranktablePath);
        if (!ifs.is_open()) {
            HCCL_VM_ERROR("failed to open ranktable file: {}", ranktablePath);
            return false;
        }
        json rankTable;
        try {
            rankTable = json::parse(ifs);
        } catch (const json::exception& e) {
            HCCL_VM_ERROR("failed to parse ranktable json: {}", e.what());
            return false;
        }
        if (!rankTable.contains("rank_list") || !rankTable["rank_list"].is_array()) {
            HCCL_VM_ERROR("rank_list not found or not array in ranktable");
            return false;
        }
        for (const auto& rank : rankTable["rank_list"]) {
            uint32_t localId = rank.value("local_id", 0);
            uint32_t spIdx = 0;
            uint32_t srvIdx = 0;
            bool found = false;
            const auto& levelList = rank.value("level_list", json::array());
            for (const auto& level : levelList) {
                if (level.value("net_layer", -1) != 0) {
                    continue;
                }
                std::string netInstanceId = level.value("net_instance_id", "");
                // 新格式: sp_{sp}_srv_{srv}_idx_{k}; 旧格式: superPod{sp}_rack{srv}
                if (sscanf(netInstanceId.c_str(), "sp_%u_srv_%u", &spIdx, &srvIdx) == 2
                    || sscanf(netInstanceId.c_str(), "superPod%u_rack%u", &spIdx, &srvIdx) == 2) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                HCCL_VM_WARN("rank local_id={} cannot determine superpod/server, skip", localId);
                continue;
            }
            topoMeta[spIdx][srvIdx].push_back(localId);
        }
        return !topoMeta.empty();
    }

    // topoMeta → 通信域server有序表: map迭代序(podId、serverId升序)即rankId分配顺序;
    // 一次批量读取Server表建立(pod_id, server_id)→serverKey索引, 避免逐server查询
    bool BuildCommConfigFromTopoMeta(const TopoMeta& topoMeta, CommConfigData& data)
    {
        auto allServersResult = Db::GetByPred<Server>(HcclSim::Storage::All<Server>());
        if (!allServersResult.ok() || !allServersResult.value.has_value()) {
            HCCL_VM_ERROR("cannot read servers for comm config");
            return false;
        }
        std::map<std::pair<uint32_t, uint32_t>, uint64_t> serverKeyIndex;
        for (const auto& srv : *allServersResult.value) {
            serverKeyIndex[std::make_pair(srv.pod_id, srv.server_id)] = srv.id;
        }

        data.servers.clear();
        uint32_t rankOffset = 0;
        for (const auto& [podId, superPod] : topoMeta) {
            for (const auto& [serverId, serverMeta] : superPod) {
                auto it = serverKeyIndex.find(std::make_pair(podId, serverId));
                if (it == serverKeyIndex.end()) {
                    HCCL_VM_ERROR("server not found in static topo: podId={:d}, serverId={:d}", podId, serverId);
                    return false;
                }
                CommConfigServer entry;
                entry.serverKey = it->second;
                entry.podId = podId;
                entry.serverId = serverId;
                entry.devNum = static_cast<uint32_t>(serverMeta.size());
                entry.rankOffset = rankOffset;
                entry.phyIds = serverMeta;
                rankOffset += entry.devNum;
                data.servers.push_back(std::move(entry));
            }
        }
        if (data.servers.empty()) {
            HCCL_VM_ERROR("no valid server in comm config");
            return false;
        }
        return true;
    }

    // 从TopoMetaConfig表项加载通信域配置并缓存(每进程仅一次)
    bool LoadCommConfigData()
    {
        std::lock_guard<std::mutex> lock(g_comm_config_mutex);
        if (g_comm_config_loaded) {
            return true;
        }
        auto ret = Db::GetOneByPred<TopoMetaConfig>(HcclSim::Storage::All<TopoMetaConfig>());
        if (!ret.ok()) {
            HCCL_VM_ERROR("topo meta config not found, please run mock-comm first");
            return false;
        }
        std::string fileName(ret->file_name, ::strnlen(ret->file_name, sizeof(ret->file_name)));
        CommConfigData data;
        if (!ParseCommConfigData(fileName, nullptr, data)) {
            return false;
        }
        g_comm_config_data = std::move(data);
        g_comm_config_loaded = true;
        return true;
    }
} // namespace

bool GetOrInsertCommunicator(
    const char* commName, uint32_t rankSize, uint32_t rankId, uint64_t deviceId, uint64_t commHash, uint64_t& commId)
{
    if (!IsValidCommunicatorName(commName) || commName[0] == '\0' || rankSize == 0 || rankId >= rankSize
        || deviceId == 0) {
        HCCL_VM_ERROR(
            "invalid communicator parameters: name={}, rankSize={}, rankId={}, deviceId={}",
            SafeCommunicatorName(commName), rankSize, rankId, deviceId);
        return false;
    }

    auto membersResult = Db::GetByPred<Communicator>(CommunicatorIdentityCondition(commName, commHash));
    if (membersResult.ok() && membersResult.value.has_value()) {
        for (const auto& member : *membersResult.value) {
            if (member.rank_size != rankSize) {
                HCCL_VM_ERROR("communicator domain conflict: name={}, rankSize={}", commName, rankSize);
                return false;
            }
            if (member.rank_id == rankId) {
                if (member.device_id != deviceId) {
                    HCCL_VM_ERROR(
                        "communicator rank mapping conflict: name={}, rankId={}, "
                        "oldDeviceId={}, newDeviceId={}",
                        commName, rankId, member.device_id, deviceId);
                    return false;
                }
                commId = member.id;
                // 复用表行时清残留销毁状态, 避免旧状态误放行.
                (void)Db::Update<Communicator>(
                    HcclSim::Storage::Eq(&Communicator::id, commId),
                    HcclSim::Storage::Set(&Communicator::status, static_cast<uint8_t>(COMM_STATUS_ACTIVE)));
                return true;
            }
            if (member.device_id == deviceId) {
                HCCL_VM_ERROR(
                    "communicator device mapping conflict: name={}, deviceId={}, "
                    "oldRankId={}, newRankId={}",
                    commName, deviceId, member.rank_id, rankId);
                return false;
            }
        }
    }

    sim::runtime::Communicator record{};
    std::strncpy(record.comm_id, commName, sizeof(record.comm_id) - 1);
    record.comm_hash = commHash;
    record.rank_size = rankSize;
    record.rank_id = rankId;
    record.device_id = deviceId;
    auto inserted = Db::Add<Communicator>(record);
    if (!inserted.ok() || record.id == 0) {
        HCCL_VM_ERROR("insert communicator member failed: name={}, rankId={}, deviceId={}", commName, rankId, deviceId);
        return false;
    }
    commId = record.id;
    HCCL_VM_DEBUG(
        "Insert Communicator member, id={}, commName={}, commHash={}, rankSize={}, rankId={}, deviceId={}", commId,
        std::string(commName), record.comm_hash, record.rank_size, record.rank_id, record.device_id);
    return true;
}

bool WaitCommunicatorReady(const char* commName, uint64_t commHash, uint32_t rankSize)
{
    if (sim::IsEstimateMode()) {
        // estimate 模式为单进程评估: 无跨 rank 成员就绪可等, 直接放行
        HCCL_VM_DEBUG("estimate mode: skip communicator readiness barrier, name={}", SafeCommunicatorName(commName));
        return true;
    }
    if (!IsValidCommunicatorName(commName) || commName[0] == '\0' || rankSize == 0) {
        HCCL_VM_ERROR(
            "invalid communicator readiness parameters: name={}, rankSize={}", SafeCommunicatorName(commName),
            rankSize);
        return false;
    }

    const uint32_t budgetMs = GetCommunicatorWaitBudgetMs();
    const auto startTs = std::chrono::steady_clock::now();
    const auto deadline = startTs + std::chrono::milliseconds(budgetMs);
    // 查询失败与成员未到齐分开计数; 失败时一次性输出实际预算/总耗时/查询统计/
    // 末次数据库返回码/最大已见唯一 rank 数/末次缺失摘要, 轮询期间不打日志。
    uint64_t queryCount = 0;
    uint64_t queryErrorCount = 0;
    int lastDbCode = 0;
    std::string lastDbDiagnostic;
    uint32_t maxUniqueRanks = 0;
    std::vector<bool> lastRankSeen(rankSize, false);
    while (std::chrono::steady_clock::now() < deadline) {
        ++queryCount;
        std::vector<bool> rankSeen(rankSize, false);
        // 就绪轮询只消费 id/rank_id：按需投影收窄读取（不回传通信域名称等
        // 无用字段）；条件与 id 升序保持 GetByPred 既有语义（判据不变）。
        auto recordsResult = HcclSim::Storage::Select<Communicator>(
            CommunicatorIdentityCondition(commName, commHash),
            HcclSim::Storage::OrderBy(&Communicator::id, HcclSim::Storage::SortDirection::ASC),
            HcclSim::Storage::Fields(&Communicator::id, &Communicator::rank_id));
        if (!recordsResult.ok() || !recordsResult.value.has_value()) {
            ++queryErrorCount;
            lastDbCode = static_cast<int>(recordsResult.code);
            lastDbDiagnostic = recordsResult.diagnostic;
            std::this_thread::sleep_for(std::chrono::milliseconds(kCommunicatorWaitIntervalMs));
            continue;
        }
        uint32_t uniqueRanks = 0;
        for (const auto& record : *recordsResult.value) {
            if (record.rank_id < rankSize && !rankSeen[record.rank_id]) {
                rankSeen[record.rank_id] = true;
                ++uniqueRanks;
            }
        }
        if (uniqueRanks > maxUniqueRanks) {
            maxUniqueRanks = uniqueRanks;
        }
        lastRankSeen = std::move(rankSeen);
        if (uniqueRanks == rankSize) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(kCommunicatorWaitIntervalMs));
    }
    const auto elapsedMs
        = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTs).count();
    HCCL_VM_ERROR(
        "wait communicator ready timeout: name={}, commHash={}, rankSize={}, budgetMs={}, "
        "elapsedMs={}, queries={}, queryErrors={}, lastDbCode={}, lastDbDiagnostic={}, "
        "maxUniqueRanks={}, lastMissing={}",
        SafeCommunicatorName(commName), commHash, rankSize, budgetMs, elapsedMs, queryCount, queryErrorCount,
        lastDbCode, lastDbDiagnostic, maxUniqueRanks, SummarizeMissingRanks(lastRankSeen));
    return false;
}

bool WaitCommunicatorDestroyReady(uint64_t commId)
{
    const auto self = Db::GetById<Communicator>(commId);
    if (!self.ok() || !IsValidCommunicatorName(self->comm_id) || self->comm_id[0] == '\0' || self->rank_size == 0
        || self->rank_id >= self->rank_size) {
        HCCL_VM_ERROR("invalid communicator destroy barrier member id={}", commId);
        return false;
    }

    CommunicatorDestroySync record{};
    std::strncpy(record.comm_id, self->comm_id, sizeof(record.comm_id) - 1);
    record.comm_hash = self->comm_hash;
    record.rank_size = self->rank_size;
    record.rank_id = self->rank_id;
    auto inserted = Db::Add<CommunicatorDestroySync>(record);
    if (!inserted.ok() || record.id == 0) {
        HCCL_VM_ERROR("failed to mark communicator destroy ready: id={}", commId);
        return false;
    }

    const uint32_t budgetMs = GetCommunicatorWaitBudgetMs();
    const auto startTs = std::chrono::steady_clock::now();
    const auto deadline = startTs + std::chrono::milliseconds(budgetMs);
    // 与就绪等待同一预算与同一诊断口径: 查询失败与成员未到齐分开计数。
    uint64_t queryCount = 0;
    uint64_t queryErrorCount = 0;
    int lastDbCode = 0;
    std::string lastDbDiagnostic;
    size_t maxMembersSeen = 0;
    while (std::chrono::steady_clock::now() < deadline) {
        ++queryCount;
        // 数量判据改用数据库端 COUNT 聚合：不再生成成员行集；判据保持
        // 记录数 >= self.rank_size（不改为 COUNT DISTINCT，不添加新判据），
        // maxMembersSeen 统计含义不变（已见记录数）。
        const auto countResult = Db::Count<CommunicatorDestroySync>(HcclSim::Storage::And(
            HcclSim::Storage::Eq(&CommunicatorDestroySync::comm_id, CommunicatorNameOf(self->comm_id)),
            HcclSim::Storage::Eq(&CommunicatorDestroySync::comm_hash, self->comm_hash)));
        if (!countResult.ok() || !countResult.value.has_value()) {
            ++queryErrorCount;
            lastDbCode = static_cast<int>(countResult.code);
            lastDbDiagnostic = countResult.diagnostic;
            std::this_thread::sleep_for(std::chrono::milliseconds(kCommunicatorWaitIntervalMs));
            continue;
        }
        if (*countResult.value > maxMembersSeen) {
            maxMembersSeen = *countResult.value;
        }
        if (*countResult.value >= self->rank_size) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(kCommunicatorWaitIntervalMs));
    }
    const auto elapsedMs
        = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTs).count();
    HCCL_VM_ERROR(
        "wait communicator destroy barrier timeout: name={}, commHash={}, rankSize={}, "
        "budgetMs={}, elapsedMs={}, queries={}, queryErrors={}, lastDbCode={}, lastDbDiagnostic={}, "
        "maxMembersSeen={}",
        SafeCommunicatorName(self->comm_id), self->comm_hash, self->rank_size, budgetMs, elapsedMs, queryCount,
        queryErrorCount, lastDbCode, lastDbDiagnostic, maxMembersSeen);
    return false;
}

// level 0 专用: 避免快卡提前销毁, 标记本 rank 已调用销毁接口并等待同通信域全部成员到达后再返回
bool MarkAndSyncCommunicatorDestroy(uint64_t commId)
{
    // 首错诊断口径：self 读取失败 / 状态 Update 失败 / 轮询查询失败 / 成员未到齐
    // 四类分开（code+diagnostic）；预算（kCommDestroyWaitTimeoutMs 固定 300000ms，
    // 不受 HCCL_VM_COMMUNICATOR_WAIT_TIMEOUT_MS 覆盖）与完成判据不变。
    const auto self = Db::GetById<Communicator>(commId);
    if (!self.ok() || !IsValidCommunicatorName(self->comm_id) || self->comm_id[0] == '\0' || self->rank_size == 0
        || self->rank_id >= self->rank_size) {
        HCCL_VM_ERROR(
            "communicator destroy sync self-read FAILED: id={}, code={}, diagnostic={}", commId,
            static_cast<int>(self.code), self.diagnostic);
        return false;
    }

    // 先标记本 rank 已销毁, 供同通信域其他 rank 感知.
    auto marked = Db::Update<Communicator>(
        HcclSim::Storage::Eq(&Communicator::id, commId),
        HcclSim::Storage::Set(&Communicator::status, static_cast<uint8_t>(COMM_STATUS_DESTROYED)));
    if (!marked.ok()) {
        HCCL_VM_ERROR(
            "communicator destroy status update FAILED: id={}, code={}, diagnostic={}", commId,
            static_cast<int>(marked.code), marked.diagnostic);
        return false;
    }

    // 等待同通信域(相同 comm_id + comm_hash)表内全部成员标记销毁;
    // 轮询统计全部来自既有 GetByPred 返回集（不新增 COUNT/全表扫描）：
    // 查询/错误次数、首个后端错误、末次结果、已见记录数/唯一 rank 数/
    // DESTROYED/ACTIVE 计数与 ≤8 个未完成 rank 样本；只在超时收尾打印。
    const uint32_t budgetMs = kCommDestroyWaitTimeoutMs;
    const auto startTs = std::chrono::steady_clock::now();
    const auto deadline = startTs + std::chrono::milliseconds(budgetMs);
    uint64_t queryCount = 0;
    uint64_t queryErrorCount = 0;
    bool sawBackendError = false;
    int firstDbCode = 0;
    std::string firstDbDiagnostic;
    int lastDbCode = 0;
    size_t maxMembersSeen = 0;
    size_t lastUniqueRanks = 0;
    size_t lastDestroyed = 0;
    size_t lastActive = 0;
    std::vector<uint32_t> lastNotDestroyedSample;
    while (std::chrono::steady_clock::now() < deadline) {
        ++queryCount;
        // 轮询只消费 id/rank_id/status：按需投影收窄读取；条件与 id 升序保持
        // GetByPred 既有语义（诊断统计与样本顺序不变，判据不变）。
        const auto recordsResult = HcclSim::Storage::Select<Communicator>(
            CommunicatorIdentityCondition(self->comm_id, self->comm_hash),
            HcclSim::Storage::OrderBy(&Communicator::id, HcclSim::Storage::SortDirection::ASC),
            HcclSim::Storage::Fields(&Communicator::id, &Communicator::rank_id, &Communicator::status));
        if (!recordsResult.ok() || !recordsResult.value.has_value()) {
            ++queryErrorCount;
            lastDbCode = static_cast<int>(recordsResult.code);
            if (!sawBackendError) {
                sawBackendError = true;
                firstDbCode = lastDbCode;
                firstDbDiagnostic = recordsResult.diagnostic;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(kCommunicatorWaitIntervalMs));
            continue;
        }
        const auto& records = *recordsResult.value;
        if (records.size() > maxMembersSeen) {
            maxMembersSeen = records.size();
        }
        bool allDestroyed = !records.empty();
        size_t destroyedCount = 0;
        size_t activeCount = 0;
        std::set<uint32_t> uniqueRanks;
        lastNotDestroyedSample.clear();
        for (const auto& record : records) {
            uniqueRanks.insert(record.rank_id);
            if (record.status == COMM_STATUS_DESTROYED) {
                ++destroyedCount;
            } else {
                allDestroyed = false;
                ++activeCount;
                if (lastNotDestroyedSample.size() < 8U) {
                    lastNotDestroyedSample.push_back(record.rank_id);
                }
            }
        }
        lastUniqueRanks = uniqueRanks.size();
        lastDestroyed = destroyedCount;
        lastActive = activeCount;
        lastDbCode = 0; // 末次查询成功
        if (allDestroyed) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(kCommunicatorWaitIntervalMs));
    }
    const auto elapsedMs
        = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTs).count();
    std::string sampleText;
    for (size_t i = 0; i < lastNotDestroyedSample.size(); ++i) {
        if (i > 0) {
            sampleText += ",";
        }
        sampleText += std::to_string(lastNotDestroyedSample[i]);
    }
    HCCL_VM_ERROR(
        "wait communicator destroy sync timeout: name={}, commHash={}, rankSize={}, "
        "budgetMs={}, elapsedMs={}, queries={}, queryErrors={}, firstDbCode={}, "
        "firstDbDiagnostic={}, lastDbCode={}, maxMembersSeen={}, uniqueRanks={}, "
        "destroyed={}, active={}, notDestroyedSample=[{}]",
        SafeCommunicatorName(self->comm_id), self->comm_hash, self->rank_size, budgetMs, elapsedMs, queryCount,
        queryErrorCount, firstDbCode, firstDbDiagnostic, lastDbCode, maxMembersSeen, lastUniqueRanks, lastDestroyed,
        lastActive, sampleText);
    return false;
}

bool GetCommunicatorMembers(uint64_t commId, std::vector<CommunicatorMemberInfo>& members)
{
    members.clear();
    if (commId == 0) {
        HCCL_VM_ERROR("communicator id is not set");
        return false;
    }

    const auto selfMember = Db::GetById<Communicator>(commId);
    if (!selfMember.ok()) {
        HCCL_VM_ERROR("communicator member not found by id:{}", commId);
        return false;
    }

    const auto communicatorMembers
        = Db::GetByPred<Communicator>(CommunicatorIdentityCondition(selfMember->comm_id, selfMember->comm_hash));
    if (!communicatorMembers.ok() || !communicatorMembers.value.has_value() || communicatorMembers.value->empty()) {
        HCCL_VM_ERROR("communicator has no members: commName={}", selfMember->comm_id);
        return false;
    }
    members.reserve(communicatorMembers.value->size());
    for (const sim::runtime::Communicator& member : *communicatorMembers.value) {
        members.push_back({member.id, member.device_id, member.rank_id});
    }
    return true;
}

bool GetCommunicatorName(uint64_t commId, std::string& commName)
{
    commName.clear();
    if (commId == 0) {
        HCCL_VM_ERROR("communicator id is not set");
        return false;
    }

    const auto selfMember = Db::GetById<Communicator>(commId);
    if (!selfMember.ok()) {
        HCCL_VM_ERROR("communicator member not found by id:{}", commId);
        return false;
    }

    if (!IsValidCommunicatorName(selfMember->comm_id) || selfMember->comm_id[0] == '\0') {
        HCCL_VM_ERROR("invalid communicator name for member id:{}", commId);
        return false;
    }
    commName = selfMember->comm_id;
    return true;
}

bool GetCommunicatorIdentity(uint64_t commId, std::string& commName, uint64_t& commHash)
{
    commName.clear();
    commHash = 0;
    if (commId == 0) {
        HCCL_VM_ERROR("communicator id is not set");
        return false;
    }
    const auto selfMember = Db::GetById<Communicator>(commId);
    if (!selfMember.ok() || !IsValidCommunicatorName(selfMember->comm_id) || selfMember->comm_id[0] == '\0') {
        HCCL_VM_ERROR("invalid communicator identity for member id:{}", commId);
        return false;
    }
    commName = selfMember->comm_id;
    commHash = selfMember->comm_hash;
    return true;
}

aclError GetDeviceByLogicId(uint32_t deviceId, sim::runtime::Device& device)
{
    auto ret = Db::GetOneByPred<Device>(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&Device::server_id, g_cur_server_key), HcclSim::Storage::Eq(&Device::logic_id, deviceId)));
    if (!ret.ok()) {
        HCCL_VM_ERROR("cannot find device by logical id {:d}", deviceId);
        return ACL_ERROR_INVALID_PARAM;
    }

    device = *ret;
    return ACL_SUCCESS;
}

aclError GetDeviceByCommRank(uint64_t commId, uint32_t rankId, sim::runtime::Device& device)
{
    if (commId == 0) {
        HCCL_VM_ERROR("communicator id is not set");
        return ACL_ERROR_INVALID_PARAM;
    }
    auto selfMember = Db::GetById<Communicator>(commId);
    if (!selfMember.ok()) {
        HCCL_VM_ERROR("communicator member not found by id:{}", commId);
        return ACL_ERROR_INVALID_PARAM;
    }
    auto rankRet = Db::GetOneByPred<Communicator>(HcclSim::Storage::And(
        CommunicatorIdentityCondition(selfMember->comm_id, selfMember->comm_hash),
        HcclSim::Storage::Eq(&Communicator::rank_id, rankId)));
    if (!rankRet.ok()) {
        HCCL_VM_ERROR("cannot find communicator member: commName={}, rankId={}", selfMember->comm_id, rankId);
        return ACL_ERROR_INVALID_PARAM;
    }
    auto deviceKey = rankRet->device_id;

    auto deviceRet = Db::GetById<Device>(deviceKey);
    if (!deviceRet.ok()) {
        // not find
        HCCL_VM_ERROR(
            "cannot find device for communicator member: commName={}, rankId={}", selfMember->comm_id, rankId);
        return ACL_ERROR_INVALID_PARAM;
    }
    device = *deviceRet;

    return ACL_SUCCESS;
}

aclError GetDeviceByPhysicalId(uint32_t deviceId, sim::runtime::Device& device)
{
    auto ret = Db::GetOneByPred<Device>(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&Device::server_id, g_cur_server_key),
        HcclSim::Storage::Eq(&Device::physical_id, deviceId)));
    if (!ret.ok()) {
        HCCL_VM_ERROR(
            "[{}] cannot find device by physical id {:d}, server_key {:d}", __func__, deviceId, g_cur_server_key);
        return ACL_ERROR_INVALID_PARAM;
    }
    HCCL_VM_INFO(
        "GetDeviceByPhysicalId: device.physical_id = {:d}, serverkey = {:d}", ret->physical_id, g_cur_server_key);

    device = *ret;
    return ACL_SUCCESS;
}

aclError GetServerByKey(uint64_t serverKey, sim::runtime::Server& server)
{
    auto ret = Db::GetById<Server>(serverKey);
    if (!ret.ok()) {
        HCCL_VM_ERROR("cannot find server by key {:d}", serverKey);
        return ACL_ERROR_INVALID_PARAM;
    }

    server = *ret;
    return ACL_SUCCESS;
}

static aclError GetDeviceByServerKeyAndPhysicalId(uint64_t serverKey, uint32_t deviceId, sim::runtime::Device& device)
{
    auto ret = Db::GetOneByPred<Device>(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&Device::server_id, serverKey), HcclSim::Storage::Eq(&Device::physical_id, deviceId)));
    if (!ret.ok()) {
        HCCL_VM_ERROR("cannot find device by server key {:d}, physical id {:d}", serverKey, deviceId);
        return ACL_ERROR_INVALID_PARAM;
    }

    device = *ret;
    return ACL_SUCCESS;
}

aclError UpdateDeviceLogicId(uint64_t serverKey, uint32_t phyDevId, uint32_t logicDevId, uint32_t userId)
{
    sim::runtime::Device device{};
    auto ret = GetDeviceByServerKeyAndPhysicalId(serverKey, phyDevId, device);
    if (ret != ACL_SUCCESS) {
        return ACL_ERROR_INVALID_PARAM;
    }

    auto deviceKey = device.id;
    Db::Update<Device>(
        HcclSim::Storage::Eq(&Device::id, deviceKey), HcclSim::Storage::Set(&Device::logic_id, logicDevId),
        HcclSim::Storage::Set(&Device::user_id, userId),
        // 标记设备已编入通信域（在线）：runner listener 按 status=1 计数在线设备，
        // drvGetDevNum 按 (server_id, status=1) 计数本 server 在域设备数
        HcclSim::Storage::Set(&Device::status, (uint32_t)1));

    return ACL_SUCCESS;
}

bool ResetAllDeviceLogicId()
{
    auto allDevices = Db::GetByPred<Device>(HcclSim::Storage::All<Device>());
    if (!allDevices.ok() || !allDevices.value.has_value() || allDevices.value->empty()) {
        HCCL_VM_ERROR("cannot find any device");
        return false;
    }

    for (auto& device : *allDevices.value) {
        auto deviceKey = device.id;
        Db::Update<Device>(
            HcclSim::Storage::Eq(&Device::id, deviceKey), HcclSim::Storage::Set(&Device::logic_id, (uint32_t)0xFFFF),
            // 重置设备状态，防止残留 status=1 干扰后续 aclrtGetDeviceCount
            HcclSim::Storage::Set(&Device::status, (uint32_t)0),
            HcclSim::Storage::Set(&Device::user_id, (uint32_t)0xFFFF));
    }

    return true;
}

aclError UpdateSuperDeviceId(uint32_t logicDevId, uint32_t superDeviceId)
{
    sim::runtime::Device device{};
    auto ret = GetDeviceByLogicId(logicDevId, device);
    if (ret != ACL_SUCCESS) {
        return ACL_ERROR_INVALID_PARAM;
    }

    auto deviceKey = device.id;
    Db::Update<Device>(
        HcclSim::Storage::Eq(&Device::id, deviceKey), HcclSim::Storage::Set(&Device::super_device_id, superDeviceId));

    return ACL_SUCCESS;
}

aclError GetCcuFromDeviceByDieId(uint64_t deviceKey, uint8_t dieId, sim::runtime::Ccu& ccu)
{
    auto ret = Db::GetOneByPred<Ccu>(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&Ccu::device_id, deviceKey), HcclSim::Storage::Eq(&Ccu::die_id, dieId)));
    if (!ret.ok()) {
        HCCL_VM_ERROR("cannot find ccu from device {:d} by die {:d}", deviceKey, static_cast<uint32_t>(dieId));
        return ACL_ERROR_INVALID_PARAM;
    }

    ccu = *ret;
    return ACL_SUCCESS;
}

aclError GetContextByDevId(uint32_t deviceId, sim::runtime::Context& context)
{
    auto ret = Db::GetOneByPred<Context>(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&Context::device_id, (uint64_t)deviceId),
        HcclSim::Storage::Eq(&Context::is_default, (uint8_t)1)));
    if (!ret.ok()) {
        HCCL_VM_ERROR("cannot find context by logical device id {:d}", deviceId);
        return ACL_ERROR_INVALID_PARAM;
    }

    context = *ret;
    return ACL_SUCCESS;
}

aclError GetPortByName(uint64_t serverKey, uint32_t phyDevId, const std::string& name, sim::runtime::Port& port)
{
    sim::runtime::Device device{};
    auto ret1 = GetDeviceByServerKeyAndPhysicalId(serverKey, phyDevId, device);
    if (ret1 != ACL_SUCCESS) {
        HCCL_VM_ERROR("get device by physical device id({:d}) failed", phyDevId);
        return ret1;
    }

    auto deviceKey = device.id;
    auto ret2 = Db::GetOneByPred<Port>(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&Port::device_id, deviceKey), HcclSim::Storage::Eq(&Port::name, name)));
    if (!ret2.ok()) {
        HCCL_VM_ERROR("cannot find port by device:{:d} name:{}", deviceKey, name.c_str());
        return ACL_ERROR_INVALID_PARAM;
    }

    port = *ret2;
    return ACL_SUCCESS;
}

aclError GetEndPointByIpAddr(const std::string& ip, sim::runtime::EndPoint& endPoint)
{
    auto ret = Db::GetOneByPred<EndPoint>(HcclSim::Storage::Eq(&EndPoint::ip_addr, ip));
    if (!ret.ok()) {
        HCCL_VM_ERROR("cannot find EndPoint by ip addr:{}", ip.c_str());
        return ACL_ERROR_INVALID_PARAM;
    }
    endPoint = *ret;
    return ACL_SUCCESS;
}

aclError GetEndPointByEid(const IpAddress& addr, sim::runtime::EndPoint& endPoint)
{
    const auto addrEid = addr.GetEid();
    std::vector<uint8_t> eidBytes(addrEid.raw, addrEid.raw + URMA_EID_LEN);
    auto ret = Db::GetOneByPred<EndPoint>(HcclSim::Storage::Eq(&EndPoint::eid, eidBytes));
    if (!ret.ok()) {
        HCCL_VM_ERROR("cannot find EndPoint by eid:{}", addr.EidToHexString().c_str());
        return ACL_ERROR_INVALID_PARAM;
    }
    endPoint = *ret;
    return ACL_SUCCESS;
}

aclError GetEndPointByEidOrIpV4(const IpAddress& addr, sim::runtime::EndPoint& endPoint)
{
    // 先按EID字节精确匹配(真实EID高位含die/port等编码)
    if (GetEndPointByEid(addr, endPoint) == ACL_SUCCESS) {
        return ACL_SUCCESS;
    }
    const Eid eid = addr.GetEid();
    const bool isV4Mapped = (eid.in4.reserved == 0 && eid.in4.prefix == 0 && eid.in4.addr != 0);
    if (!isV4Mapped) {
        HCCL_VM_ERROR("eid: {} is not mapped to IPV4", addr.EidToHexString().c_str());
        return ACL_ERROR_INVALID_PARAM;
    }
    struct in_addr v4 {};
    memcpy(&v4, eid.raw + URMA_EID_LEN - sizeof(v4), sizeof(v4));
    char ipStr[INET_ADDRSTRLEN] = {0};
    if (inet_ntop(AF_INET, &v4, ipStr, sizeof(ipStr)) == nullptr) {
        HCCL_VM_ERROR("inet_ntop failed");
        return ACL_ERROR_INVALID_PARAM;
    }
    return GetEndPointByIpAddr(std::string(ipStr), endPoint);
}

bool GetRankIdByEid(const IpAddress& addr, uint32_t& rankId)
{
    // 按EID查ranktable.json记录的rankId(RankEidMap表); estimate模式Communicator表无对端行, 对端rank以此为准
    const auto addrEid = addr.GetEid();
    std::vector<uint8_t> eidBytes(addrEid.raw, addrEid.raw + URMA_EID_LEN);
    auto ret = Db::GetOneByPred<RankEidMap>(HcclSim::Storage::Eq(&RankEidMap::eid, eidBytes));
    if (!ret.ok()) {
        HCCL_VM_ERROR("failed to find rank by eid={}", addr.EidToHexString().c_str());
        return false;
    }
    rankId = ret->rank_id;
    return true;
}

aclError GetPortById(uint64_t portId, sim::runtime::Port& port)
{
    auto ret = Db::GetById<Port>(portId);
    if (!ret.ok()) {
        HCCL_VM_ERROR("cannot find port by id: {:d}", portId);
        return ACL_ERROR_INVALID_PARAM;
    }

    port = *ret;
    return ACL_SUCCESS;
}

bool GetCommRankByDeviceId(uint64_t commId, uint64_t deviceId, uint32_t& rankId)
{
    // 建域并发窗口（2026-09-17 修复）：256 宿主并发建域时，rank 启动参差
    // 导致部分宿主的成员行尚未插入，而其他 rank 的设备进程已经开始解析
    // 引用这些行的 SQE（FR1/R3K4/R5K1 取证 391,680 次 NOT_FOUND）。
    // 修复：短暂退避重试等待写入方完成——上限 60s（可通过环境变量
    // HCCL_VM_DB_WAIT_TIMEOUT_MS 调整），覆盖实测最大初始化参差 16.7s。
    // 超时后返回 false（与旧行为一致），但日志明确标识等待超时。
    if (commId == 0) {
        HCCL_VM_ERROR("communicator id is not set");
        return false;
    }

    static const auto waitTimeout = [] {
        const char* env = std::getenv("HCCL_VM_DB_WAIT_TIMEOUT_MS");
        return env != nullptr ? std::chrono::milliseconds(std::atoi(env)) : std::chrono::milliseconds(60000);
    }();
    const auto deadline = std::chrono::steady_clock::now() + waitTimeout;
    constexpr auto kRetryInterval = std::chrono::milliseconds(100);
    bool firstRetryLogged = false;

    auto selfMember = Db::GetById<Communicator>(commId);
    auto member = decltype(selfMember.ok(), Db::GetOneByPred<Communicator>(
        HcclSim::Storage::And(
            CommunicatorIdentityCondition(selfMember->comm_id, selfMember->comm_hash),
            HcclSim::Storage::Eq(&Communicator::device_id, deviceId)))){};

    while (true) {
        if (selfMember.ok()) {
            member = Db::GetOneByPred<Communicator>(HcclSim::Storage::And(
                CommunicatorIdentityCondition(selfMember->comm_id, selfMember->comm_hash),
                HcclSim::Storage::Eq(&Communicator::device_id, deviceId)));
            if (member.ok()) {
                break; // 全链命中
            }
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            HCCL_VM_ERROR(
                "communicator member not found by id:{} after {}ms wait "
                "(peer host may not have completed init)",
                commId, waitTimeout.count());
            return false;
        }
        if (!firstRetryLogged) {
            HCCL_VM_WARN(
                "communicator {} member not yet available, waiting up to {}ms...", commId, waitTimeout.count());
            firstRetryLogged = true;
        }
        std::this_thread::sleep_for(kRetryInterval);
        // 重试时重新读自身行（自身行不太可能消失，但保持一致性）
        selfMember = Db::GetById<Communicator>(commId);
    }

    rankId = member->rank_id;
    return true;
}

uint32_t GetAICpuCount(uint64_t deviceId)
{
    // ACL_DEV_ATTR_AICPU_CORE_NUM AI CPU数量。
    auto ret = Db::GetOneByPred<Device>(HcclSim::Storage::Eq(&Device::logic_id, deviceId));
    if (!ret.ok()) {
        HCCL_VM_ERROR("cannot find device by physical id {:d}", deviceId);
        return 0;
    }

    auto deviceIdx = ret->id;

    // 数据库端 COUNT 聚合：只取 AI CPU scheduler 数量，不加载行集。
    return static_cast<uint32_t>(
        Db::Count<TaskSchedulerDevice>(HcclSim::Storage::And(
                                           HcclSim::Storage::Eq(&TaskSchedulerDevice::device_id, deviceIdx),
                                           HcclSim::Storage::Eq(&TaskSchedulerDevice::type, (uint8_t)TS_DEV_TYPE_CPU)))
            .value_or(0U));
}

uint32_t GetAICoreCount(uint64_t deviceId)
{
    // ACL_DEV_ATTR_AICORE_CORE_NUM AI Core数量。
    auto ret = Db::GetOneByPred<Device>(HcclSim::Storage::Eq(&Device::logic_id, deviceId));
    if (!ret.ok()) {
        HCCL_VM_ERROR("cannot find device by physical id {:d}", deviceId);
        return 0;
    }

    auto deviceIdx = ret->id;

    // 数据库端 COUNT 聚合：只取 Scalar scheduler 数量，不加载行集。
    return static_cast<uint32_t>(Db::Count<TaskSchedulerDevice>(
                                     HcclSim::Storage::And(
                                         HcclSim::Storage::Eq(&TaskSchedulerDevice::device_id, deviceIdx),
                                         HcclSim::Storage::Eq(&TaskSchedulerDevice::type, (uint8_t)TS_DEV_TYPE_SCALAR)))
                                     .value_or(0U));
}

uint32_t GetVectorCoreCount(uint64_t deviceId)
{
    // ACL_DEV_ATTR_VECTOR_CORE_NUM Vector Core数量。
    auto ret = Db::GetOneByPred<Device>(HcclSim::Storage::Eq(&Device::logic_id, deviceId));
    if (!ret.ok()) {
        HCCL_VM_ERROR("cannot find device by physical id {:d}", deviceId);
        return 0;
    }

    auto deviceIdx = ret->id;

    auto scalarsResult = Db::GetByPred<TaskSchedulerDevice>(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&TaskSchedulerDevice::device_id, deviceIdx),
        HcclSim::Storage::Eq(&TaskSchedulerDevice::type, (uint8_t)TS_DEV_TYPE_SCALAR)));
    if (!scalarsResult.ok() || !scalarsResult.value.has_value()) {
        return 0;
    }

    uint32_t vectorCoreCount = 0;
    for (const auto& scalar : *scalarsResult.value) {
        auto tsIdx = scalar.id;
        // 数据库端 COUNT 聚合：按 scheduler 逐个计数 Vector 核，不加载 ComputeDie 行集。
        vectorCoreCount += static_cast<uint32_t>(
            Db::Count<ComputeDie>(HcclSim::Storage::And(
                                      HcclSim::Storage::Eq(&ComputeDie::ts_id, tsIdx),
                                      HcclSim::Storage::Eq(&ComputeDie::type, (uint8_t)COMPUTE_TYPE_VECTOR)))
                .value_or(0U));
    }
    return vectorCoreCount;
}

std::string GetHardwareTypeByDevice(uint64_t deviceId)
{
    auto device = Db::GetOneByPred<Device>(HcclSim::Storage::Eq(&Device::logic_id, deviceId));
    if (!device.ok()) {
        HCCL_VM_ERROR("cannot find device by physical id {:d}", deviceId);
        return "";
    }

    auto server = Db::GetById<Server>(device->server_id);
    if (!server.ok()) {
        HCCL_VM_ERROR("cannot find server by id {:d}", device->server_id);
        return "";
    }

    return std::string(server->hardware_type);
}

uint32_t GetCubeCoreCount(uint64_t deviceId)
{
    auto ret = Db::GetOneByPred<Device>(HcclSim::Storage::Eq(&Device::logic_id, deviceId));
    if (!ret.ok()) {
        HCCL_VM_ERROR("cannot find device by physical id {:d}", deviceId);
        return 0;
    }

    auto deviceIdx = ret->id;

    auto scalarsResult = Db::GetByPred<TaskSchedulerDevice>(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&TaskSchedulerDevice::device_id, deviceIdx),
        HcclSim::Storage::Eq(&TaskSchedulerDevice::type, (uint8_t)TS_DEV_TYPE_SCALAR)));
    if (!scalarsResult.ok() || !scalarsResult.value.has_value()) {
        return 0;
    }

    uint32_t cubeCoreCount = 0;
    for (const auto& scalar : *scalarsResult.value) {
        auto tsIdx = scalar.id;
        // 数据库端 COUNT 聚合：按 scheduler 逐个计数 Cube 核，不加载 ComputeDie 行集。
        cubeCoreCount += static_cast<uint32_t>(
            Db::Count<ComputeDie>(HcclSim::Storage::And(
                                      HcclSim::Storage::Eq(&ComputeDie::ts_id, tsIdx),
                                      HcclSim::Storage::Eq(&ComputeDie::type, (uint8_t)COMPUTE_TYPE_CUBE)))
                .value_or(0U));
    }
    return cubeCoreCount;
}

std::set<uint64_t> GetUsedServerNum()
{
    try {
        auto allDevicesResult = Db::GetByPred<Device>(HcclSim::Storage::Eq(&Device::status, (uint32_t)1));
        if (!allDevicesResult.ok() || !allDevicesResult.value.has_value()) {
            return {};
        }
        std::set<uint64_t> usedServerIds;
        for (const auto& device : *allDevicesResult.value) {
            usedServerIds.insert(device.server_id);
        }
        return usedServerIds;
    } catch (const std::exception& e) {
        HCCL_VM_ERROR("exception: {}", e.what());
        return {};
    }
}

bool GetRankIdByMPI(uint32_t& rankId, uint64_t& serverId)
{
    // 仅显式启用 PyTorch 时使用 server 索引定位；其余场景沿用下方 MPI/单 server 逻辑。
    const char* enablePytorch = std::getenv("HCCL_VM_ENABLE_PYTORCH");
    const char* serverIndexStr = std::getenv("HCCL_VM_SERVER_INDEX");
    if (enablePytorch != nullptr && std::strcmp(enablePytorch, "1") == 0 && serverIndexStr != nullptr) {
        char* endPtr = nullptr;
        const long serverIndex = std::strtol(serverIndexStr, &endPtr, 10);
        if (serverIndexStr[0] == '\0' || endPtr == serverIndexStr || *endPtr != '\0' || serverIndex < 0) {
            HCCL_VM_ERROR("invalid HCCL_VM_SERVER_INDEX: {}", serverIndexStr);
            return false;
        }
        if (!LoadCommConfigData()) {
            return false;
        }

        std::lock_guard<std::mutex> lock(g_comm_config_mutex);
        if (static_cast<size_t>(serverIndex) >= g_comm_config_data.servers.size()) {
            HCCL_VM_ERROR(
                "HCCL_VM_SERVER_INDEX {} exceeds server count {}", serverIndex, g_comm_config_data.servers.size());
            return false;
        }
        // serverIndex 是当前通信域配置中从 0 开始的服务器序号，不是 Server 表主键。
        const auto& server = g_comm_config_data.servers[static_cast<size_t>(serverIndex)];
        serverId = server.serverKey;
        // GetCurServerId calls this function before aclrtSetDevice and uses 0xFFFF
        // as the marker that no local device has been selected yet.
        if (rankId != 0xFFFF) {
            if (rankId >= server.devNum) {
                HCCL_VM_ERROR(
                    "local device {} exceeds device count {} for server index {}", rankId, server.devNum, serverIndex);
                return false;
            }
            // 输入为 server 内逻辑设备号，加上前序 server 的设备总数得到全局 rank。
            rankId += server.rankOffset;
        }
        return true;
    }

    const char* ompiRankStr = std::getenv("OMPI_COMM_WORLD_RANK");
    const char* mpichRankStr = std::getenv("PMI_RANK");
    if (ompiRankStr == nullptr && mpichRankStr == nullptr) {
        // 单server用例，默认serverId为1
        auto usedServerIds = GetUsedServerNum();
        if (usedServerIds.size() == 1) {
            if (*usedServerIds.begin() != 1) {
                HCCL_VM_ERROR("env OMPI_COMM_WORLD_RANK or PMI_RANK are not found, serverId is not 1.");
                return false;
            }
            serverId = 1;
            return true;
        }
        HCCL_VM_ERROR("env OMPI_COMM_WORLD_RANK or PMI_RANK are not found, usedServerIds size is not 1.");
        return false;
    } else if (ompiRankStr != nullptr) {
        rankId = static_cast<uint32_t>(atoi(ompiRankStr));
    } else {
        rankId = static_cast<uint32_t>(atoi(mpichRankStr));
    }
    return true;
}

bool ParseCommConfigData(const std::string& fileName, const TopoMeta* topoMeta, CommConfigData& data)
{
    TopoMeta parsedMeta;
    const TopoMeta* meta = topoMeta;
    if (meta == nullptr) {
        if (fileName == "ranktable") {
            if (!ParseTopoMetaFromRankTableFile(parsedMeta)) {
                HCCL_VM_ERROR("parse ranktable failed");
                return false;
            }
        } else if (!ParseTopoMetaYaml(fileName, parsedMeta)) {
            HCCL_VM_ERROR("parse topo meta yaml failed: {}", fileName);
            return false;
        }
        meta = &parsedMeta;
    }
    data.fileName = fileName;
    return BuildCommConfigFromTopoMeta(*meta, data);
}

bool RefreshTopoMetaConfig(const std::string& fileName, const TopoMeta* topoMeta)
{
    if (fileName.empty() || fileName.size() >= sizeof(TopoMetaConfig::file_name)) {
        HCCL_VM_ERROR("invalid topo meta file name: {}", fileName);
        return false;
    }
    // 每次mock-comm执行覆盖表项: TopoMetaConfig仅记录当前通信域配置
    TopoMetaConfig cfg{};
    std::strncpy(cfg.file_name, fileName.c_str(), sizeof(cfg.file_name) - 1);
    (void)Db::DeleteAll<TopoMetaConfig>(HcclSim::Storage::ClearMode::RESET_SEQUENCE);
    auto inserted = Db::Add<TopoMetaConfig>(cfg);
    if (!inserted.ok() || cfg.id == 0) {
        HCCL_VM_ERROR("insert topo meta config failed: {}", fileName);
        return false;
    }
    // 同步刷新本进程缓存(其余进程为新进程, 首次使用时按表项自行加载)
    CommConfigData data;
    if (!ParseCommConfigData(fileName, topoMeta, data)) {
        HCCL_VM_ERROR("parse topo meta yaml failed: {}", fileName);
        return false;
    }
    std::lock_guard<std::mutex> lock(g_comm_config_mutex);
    g_comm_config_data = std::move(data);
    g_comm_config_loaded = true;
    HCCL_VM_INFO("refresh comm config: file={}, server num={}", fileName, g_comm_config_data.servers.size());
    return true;
}

void ResetCommConfigData()
{
    std::lock_guard<std::mutex> lock(g_comm_config_mutex);
    g_comm_config_data = CommConfigData{};
    g_comm_config_loaded = false;
}

bool GetServerKeyByRankId(uint32_t rankId, uint64_t& serverKey)
{
    if (!LoadCommConfigData()) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_comm_config_mutex);
    for (const auto& srv : g_comm_config_data.servers) {
        if (rankId >= srv.rankOffset && rankId < srv.rankOffset + srv.devNum) {
            serverKey = srv.serverKey;
            return true;
        }
    }
    HCCL_VM_ERROR("server not found by rankId:{:d}", rankId);
    return false;
}

aclError GetDevIndexByPhyId(uint64_t serverKey, uint32_t phyId, uint32_t& devIndex)
{
    if (!LoadCommConfigData()) {
        return ACL_ERROR_INVALID_PARAM;
    }
    std::lock_guard<std::mutex> lock(g_comm_config_mutex);
    for (const auto& srv : g_comm_config_data.servers) {
        if (srv.serverKey != serverKey) {
            continue;
        }
        for (size_t idx = 0; idx < srv.phyIds.size(); ++idx) {
            if (srv.phyIds[idx] == phyId) {
                devIndex = static_cast<uint32_t>(idx);
                return ACL_SUCCESS;
            }
        }
        break; // 找到server但无匹配phyId
    }
    HCCL_VM_ERROR("dev index not found by phyId:{:d}, serverKey:{:d}", phyId, serverKey);
    return ACL_ERROR_INVALID_PARAM;
}

uint64_t GetCurServerId()
{
    if (g_cur_server_key != 0) {
        return g_cur_server_key;
    }

    // 3. 获取当前rankId
    uint32_t rankId = 0xFFFF;
    uint64_t serverKey = 0;
    if (!sim::runtime::GetRankIdByMPI(rankId, serverKey)) {
        HCCL_VM_ERROR("get rankId by MPI fail serverKey:{:d}", serverKey);
        return 0;
    }

    if (serverKey == 0 && rankId != 0xFFFF) {
        // 4. 根据rankId获取serverId: 按TopoMetaConfig指向的配置推导rankId→server映射
        if (!sim::runtime::GetServerKeyByRankId(rankId, serverKey)) {
            HCCL_VM_ERROR("get serverKey by rankId:{:d} failed", rankId);
            return 0;
        }
    }
    HCCL_VM_INFO("Get current serverKey:{:d}", serverKey);

    g_cur_server_key = serverKey;
    return serverKey;
}
} // namespace sim::runtime
