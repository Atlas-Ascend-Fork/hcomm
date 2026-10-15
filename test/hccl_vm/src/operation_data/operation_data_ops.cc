/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 实现分区与 operation_data_ops.h 保持相同顺序：已有业务接口 → Op Detail/Memory →
// Sync Record → CCU/Jetty → Dynamic Task → Lifecycle/Maintenance。DDL 与 codec 在头文件。
#include "operation_data/operation_data_ops.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <unistd.h>
#include <utility>
#include <vector>

#include "sim_common_macro.h"
#include "sim_log.h"
#include "sim_capacity_limits.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "runtime_state/db_sim_communicator.h"
#include "runtime_state/sim_models.h"
#include "storage/storage_runtime.h"
#include "storage/table_access.h"
#include "storage_composition/storage_composition.h"

namespace sim::operation {
uint32_t g_syncIterCounter = 0;
uint32_t g_currOpDetailId = 0;
uint32_t g_currOpMemId = 0;
// 当前进程正在处理的算子迭代（= 最近一次 InsertOpDetail 分配的 opIter；主线语义）。
// 用途：CCU 交换表（CcuSyncResTab）需要跨 rank 一致的 op 身份来隔离不同算子的
// 交换值——两侧进程按同一算子序列推进时天然对齐。
std::atomic<uint32_t> g_curOpIter{0};
// opIter 按 (commName, commHash) 独立自增（迁移前/主线语义）：进程参与多个通信域时
// 各序列互不干扰，两侧进程在同一通信域上的算子序列天然对齐。
std::mutex g_opIterCounterMutex{};
std::map<std::pair<std::string, uint64_t>, std::atomic<uint32_t>> g_opIterCounters{};

namespace {
    using HcclSim::Storage::Compare;
    using HcclSim::Storage::DbResult;
    using HcclSim::Storage::PrimaryKey;
    using HcclSim::Storage::StorageRuntime;
    using HcclSim::Storage::TableScope;

    /// 生命周期/维护入口专用的 ready 会话 ensure（等价迁移前 OpDbOps 单例的首次建连时机）。
    /// 普通 CRUD 不调用本函数：Storage 原子 API 已统一经 ProcessStorageContext::
    /// AcquireReady() 惰性建立会话（唯一 simulator bootstrap 已注册为进程级来源）。
    /// 装配委托唯一 simulator bootstrap（后端注册/目录登记/默认路由），本文件不拼
    /// 具体 manager/engine；普通 Insert/Select/Update 只走 Storage 原子 API。
    bool EnsureStorageSession()
    {
        auto ensured = sim::EnsureSimulatorStorageSession();
        if (!ensured.ok()) {
            HCCL_VM_ERROR(
                "[OpDbOps] open storage session failed: {} ({})", ensured.diagnostic, static_cast<int>(ensured.code));
            return false;
        }
        return true;
    }

    /// 会话失败时统一转迁移前 int 返回码并输出关键错误日志。
    int FailWith(const char* operation, const std::string& diagnostic)
    {
        HCCL_VM_ERROR("[OpDbOps] {} failed: {}", operation, diagnostic);
        return -1;
    }

    /// 各流独立的 sync 迭代计数（进程内状态，不落库；迁移前 g_syncStreamIterMap 语义）。
    std::map<uint64_t, std::atomic<uint64_t>> g_syncStreamIterMap{};

    /// 业务显式事务的 CONFLICT 重建重试上限（对齐 B2 即时路径的有界重试惯例）。
    /// Redis 后端的 Commit 契约：WATCH 冲突（EXEC nil）返回 DbCode::CONFLICT 且
    /// 引擎不内部重试（provisional ID 已对调用方可见），由调用方重建整个事务
    /// 重试；SQLite 后端无此并发冲突码，迁移前语义不感知。
    constexpr int kTxConflictRetryLimit = 8;

    /// 分配下一个 opIter（按 commName/commHash 独立序列；迁移前/主线语义）并记录
    /// "当前 op"迭代（CCU 交换表 op 隔离身份）。失败时返回 false 且不改动 opIter。
    bool AssignOpIter(uint64_t commId, uint32_t& opIter)
    {
        std::string commName;
        uint64_t commHash = 0;
        if (!sim::runtime::GetCommunicatorIdentity(commId, commName, commHash)) {
            HCCL_VM_ERROR("Failed to resolve communicator identity before assigning opIter, commId={}", commId);
            return false;
        }
        {
            std::lock_guard<std::mutex> lock(g_opIterCounterMutex);
            auto iter = g_opIterCounters.try_emplace(std::make_pair(commName, commHash), 0u).first;
            opIter = iter->second.fetch_add(1, std::memory_order_relaxed);
        }
        // 记录"当前 op"：本 op 的 CCU launch 随后会在同进程发生，交换表据此打 op 身份。
        g_curOpIter.store(opIter, std::memory_order_relaxed);
        return true;
    }
} // namespace

// ==== Stream Sync 迭代分区（纯进程内状态，不落库） ====

void AddSyncStreamIter(uint64_t streamId) { g_syncStreamIterMap.try_emplace(streamId, 0u); }

void RemoveSyncStreamIter(uint64_t streamId) { g_syncStreamIterMap.erase(streamId); }

bool NextStreamSyncIdx(uint64_t streamId, uint64_t& syncIdx)
{
    auto it = g_syncStreamIterMap.find(streamId);
    if (it == g_syncStreamIterMap.end()) {
        return false;
    }
    syncIdx = it->second.fetch_add(1);
    return true;
}

// ==== Op Detail / Memory 分区 ====

uint32_t GetCurOpIter() { return g_curOpIter.load(std::memory_order_relaxed); }

int InsertOpDetail(OpDetailTab& rec)
{
    if (!AssignOpIter(rec.commId, rec.opIter)) {
        return -1;
    }
    rec.syncIter = g_syncIterCounter;

    rec.id = 0;
    auto inserted = HcclSim::Storage::Insert<OpDetailTab>(rec);
    if (!inserted.ok()) {
        return FailWith("insert opDetails", inserted.diagnostic);
    }
    g_currOpDetailId = rec.id;
    return 0;
}

int InsertOpMem(OpMemInfoTab& rec)
{
    rec.opDetailId = g_currOpDetailId;

    rec.id = 0;
    auto inserted = HcclSim::Storage::Insert<OpMemInfoTab>(rec);
    if (!inserted.ok()) {
        return FailWith("insert opMemInfo", inserted.diagnostic);
    }
    g_currOpMemId = rec.id;
    return 0;
}

int InsertOpDetailAndMem(OpDetailTab& detail, OpMemInfoTab& mem)
{
    // 迁移前语义：OpDetail 与 OpMem 在同一事务内原子提交。
    if (!AssignOpIter(detail.commId, detail.opIter)) {
        return -1;
    }
    detail.syncIter = g_syncIterCounter;

    // CONFLICT（如 Redis 后端多 rank 并发写同表触发 WATCH 冲突）时按引擎
    // 契约重建整个事务重试：失败事务零提交，detail.id/mem.id 在循环内重新
    // 清零由引擎重新分配，opIter 为内存计数器不受失败影响。
    for (int attempt = 0; attempt < kTxConflictRetryLimit; ++attempt) {
        auto transaction = HcclSim::Storage::BeginTransaction();
        if (!transaction.ok()) {
            return FailWith("insert opDetails with opMemInfo", transaction.diagnostic);
        }
        auto detailTable = HcclSim::Storage::AcquireTable<OpDetailTab>();
        if (!detailTable.ok()) {
            (void)transaction->Rollback();
            return FailWith("insert opDetails with opMemInfo", detailTable.diagnostic);
        }
        auto memTable = HcclSim::Storage::AcquireTable<OpMemInfoTab>();
        if (!memTable.ok()) {
            (void)transaction->Rollback();
            return FailWith("insert opDetails with opMemInfo", memTable.diagnostic);
        }
        detail.id = 0;
        auto stagedDetail = detailTable->Insert(detail, *transaction);
        if (!stagedDetail.ok()) {
            (void)transaction->Rollback();
            return FailWith("insert opDetails with opMemInfo", stagedDetail.diagnostic);
        }
        mem.opDetailId = detail.id;
        mem.id = 0;
        auto stagedMem = memTable->Insert(mem, *transaction);
        if (!stagedMem.ok()) {
            (void)transaction->Rollback();
            return FailWith("insert opDetails with opMemInfo", stagedMem.diagnostic);
        }
        auto committed = transaction->Commit();
        if (committed.ok()) {
            g_currOpDetailId = detail.id;
            g_currOpMemId = mem.id;
            return 0;
        }
        if (committed.code != HcclSim::Storage::DbCode::CONFLICT) {
            return FailWith("insert opDetails with opMemInfo", committed.diagnostic);
        }
    }
    return FailWith("insert opDetails with opMemInfo", "transaction conflict retries exhausted");
}

int UpdateOpMemCclBuffer(uint64_t cclAddr, uint64_t cclSize)
{
    auto updated = HcclSim::Storage::Update<OpMemInfoTab>(
        HcclSim::Storage::Eq(&OpMemInfoTab::id, g_currOpMemId), HcclSim::Storage::Set(&OpMemInfoTab::cclAddr, cclAddr),
        HcclSim::Storage::Set(&OpMemInfoTab::cclSize, cclSize));
    if (!updated.ok()) {
        return FailWith("update opMemInfo ccl buffer", updated.diagnostic);
    }
    return 0;
}

int QueryCurrentOpMemInfoByRank(uint32_t rankId, OpMemInfoTab& out)
{
    // 数据库原子操作三步分解，等价迁移前 JOIN 语义：
    // 1. 按主键取当前 OpDetail，定位其 syncIter；
    // 2. 取同一 syncIter 下目标 rank 的最新 OpDetail（id 降序取首条）；
    // 3. 取该目标 detail 的 OpMemInfo（目标 detail id，而非当前 detail id）。
    // 同一流程两次查询同一张 detail 表：只获取一次句柄并复用。
    // detail 两查均为标量消费（首查 syncIter，次查 id），Fields 投影剥离
    // opDetail/opExtInfo 大 BLOB（1024p 单行 extInfo ≈ 8MiB，点查拖全行是
    // 纯无效传输）；投影覆盖谓词与排序字段，语义与全行读取一致。
    auto detailTable = HcclSim::Storage::AcquireTable<OpDetailTab>();
    if (!detailTable.ok()) {
        HCCL_VM_ERROR("query failed, currentOpDetailId={}, rankId={}", g_currOpDetailId, rankId);
        return -1;
    }
    const auto currentDetail = detailTable->Select(
        HcclSim::Storage::Eq(&OpDetailTab::id, g_currOpDetailId), HcclSim::Storage::Limit(1),
        HcclSim::Storage::Fields(&OpDetailTab::id, &OpDetailTab::syncIter));
    if (!currentDetail.ok() || currentDetail.value->empty()) {
        HCCL_VM_ERROR("query failed, currentOpDetailId={}, rankId={}", g_currOpDetailId, rankId);
        return -1;
    }
    const auto targetDetail = detailTable->Select(
        HcclSim::Storage::And(
            HcclSim::Storage::Eq(&OpDetailTab::syncIter, currentDetail.value->front().syncIter),
            HcclSim::Storage::Eq(&OpDetailTab::rankId, rankId)),
        HcclSim::Storage::OrderBy(&OpDetailTab::id, HcclSim::Storage::SortDirection::DESC), HcclSim::Storage::Limit(1),
        HcclSim::Storage::Fields(&OpDetailTab::id, &OpDetailTab::syncIter, &OpDetailTab::rankId));
    if (!targetDetail.ok() || targetDetail.value->empty()) {
        HCCL_VM_ERROR("query failed, currentOpDetailId={}, rankId={}", g_currOpDetailId, rankId);
        return -1;
    }
    const auto mem = HcclSim::Storage::SelectOne<OpMemInfoTab>(
        HcclSim::Storage::Eq(&OpMemInfoTab::opDetailId, targetDetail.value->front().id));
    if (!mem.ok()) {
        HCCL_VM_ERROR("query failed, currentOpDetailId={}, rankId={}", g_currOpDetailId, rankId);
        return -1;
    }
    out = *mem.value;
    return 0;
}

int QueryCurrentOpMemInfo(uint64_t commId, uint32_t deviceId, OpMemInfoTab& out)
{
    // 迁移前语义：JOIN opDetails d + opMemInfo m ON m.opDetailId = d.id，
    // WHERE d.commId=? AND d.deviceId=? ORDER BY d.id DESC LIMIT 1。
    // 数据库原子操作两步分解等价该 JOIN：
    // 1. 按 commId+deviceId 取最新 OpDetail（id 降序取首条）；
    // 2. 取该 detail 的 OpMemInfo。
    if (commId == 0 || deviceId == 0) {
        HCCL_VM_ERROR("invalid current opMemInfo query, commId={}, deviceId={}", commId, deviceId);
        return -1;
    }

    // 标量消费（id），Fields 投影剥离 opDetail/opExtInfo 大 BLOB；投影覆盖
    // 谓词（commId/deviceId）与排序字段（id），首条语义不变；SelectOne 直接
    // 物化单对象（不经 vector 中转）。整体仍是两步读取（不新增 JOIN），
    // 最新 detail 没有 mem 即失败，不回退较旧 detail。
    auto targetDetail = HcclSim::Storage::SelectOne<OpDetailTab>(
        HcclSim::Storage::And(
            HcclSim::Storage::Eq(&OpDetailTab::commId, commId), HcclSim::Storage::Eq(&OpDetailTab::deviceId, deviceId)),
        HcclSim::Storage::Fields(&OpDetailTab::id, &OpDetailTab::commId, &OpDetailTab::deviceId),
        HcclSim::Storage::OrderBy(&OpDetailTab::id, HcclSim::Storage::SortDirection::DESC));
    if (!targetDetail.ok() || !targetDetail.value.has_value()) {
        HCCL_VM_ERROR("query current opMemInfo failed, commId={}, deviceId={}", commId, deviceId);
        return -1;
    }

    auto mem = HcclSim::Storage::SelectOne<OpMemInfoTab>(
        HcclSim::Storage::Eq(&OpMemInfoTab::opDetailId, targetDetail.value->id));
    if (!mem.ok()) {
        HCCL_VM_ERROR("query current opMemInfo failed, commId={}, deviceId={}", commId, deviceId);
        return -1;
    }
    out = *mem.value;
    return 0;
}

int QueryOpMemInfoByOpDetailId(uint32_t opDetailId, OpMemInfoTab& out)
{
    // 迁移前语义：SELECT ... FROM opMemInfo WHERE opDetailId = ?（首行）。
    auto found = HcclSim::Storage::Select<OpMemInfoTab>(
        HcclSim::Storage::Eq(&OpMemInfoTab::opDetailId, opDetailId),
        HcclSim::Storage::OrderBy(&OpMemInfoTab::id, HcclSim::Storage::SortDirection::ASC), HcclSim::Storage::Limit(1));
    if (!found.ok() || found.value->empty()) {
        HCCL_VM_ERROR("query opMemInfo by opDetailId failed, opDetailId={}", opDetailId);
        return -1;
    }
    out = found.value->front();
    return 0;
}

int UpdateOpExpansionMode(uint8_t mode)
{
    auto updated = HcclSim::Storage::Update<OpDetailTab>(
        HcclSim::Storage::Eq(&OpDetailTab::id, g_currOpDetailId),
        HcclSim::Storage::Set(&OpDetailTab::opExpansionMode, mode));
    if (!updated.ok()) {
        HCCL_VM_ERROR("Failed to update mode={}, id={}", mode, g_currOpDetailId);
        return -1;
    }
    return 0;
}

int UpdateOpDetailAlgName(const std::string& algName)
{
    // 算法选择结果回填 (日志截取): level0 桩先 InsertOpDetail (置
    // g_currOpDetailId) 再调真实业务, 选择日志发生在业务调用内、与插入
    // 同线程, 据此更新"当前算子"行; 同一算子多次匹配 (选择器 ->
    // HcclGetAlgRes -> 降级重选) 时后写覆盖, 与最终生效算法一致
    if (algName.empty() || g_currOpDetailId == 0) {
        return 0;
    }
    auto updated = HcclSim::Storage::Update<OpDetailTab>(
        HcclSim::Storage::Eq(&OpDetailTab::id, g_currOpDetailId),
        HcclSim::Storage::Set(&OpDetailTab::algorithmName, algName));
    if (!updated.ok()) {
        HCCL_VM_ERROR("Failed to update opDetail algorithmName, id={}, algName={}", g_currOpDetailId, algName);
        return -1;
    }
    return 0;
}

int UpdateOpDetailMainStream(uint32_t opDetailId, uint64_t streamId)
{
    // 迁移前语义：UPDATE opDetails SET streamId = ? WHERE id = ?。
    auto updated = HcclSim::Storage::Update<OpDetailTab>(
        HcclSim::Storage::Eq(&OpDetailTab::id, opDetailId), HcclSim::Storage::Set(&OpDetailTab::streamId, streamId));
    if (!updated.ok()) {
        HCCL_VM_ERROR("Failed to update opDetail main stream, id={}, streamId={}", opDetailId, streamId);
        return -1;
    }
    return 0;
}

int QueryLatestOpExpansionMode()
{
    // 标量查询：只取 opExpansionMode，Fields 投影剥离 opDetail/opExtInfo 大 BLOB；
    // SelectOne 直接物化单对象（不经 vector 中转）。
    auto found = HcclSim::Storage::SelectOne<OpDetailTab>(
        HcclSim::Storage::Eq(&OpDetailTab::id, g_currOpDetailId),
        HcclSim::Storage::Fields(&OpDetailTab::id, &OpDetailTab::opExpansionMode));
    if (!found.ok() || !found.value.has_value()) {
        HCCL_VM_WARN("Failed to query, id={}", g_currOpDetailId);
        return static_cast<int>(sim::runtime::SimOpExpansionMode::SIM_OP_EXPANSION_MODE_RESERVED);
    }
    return static_cast<int>(found.value->opExpansionMode);
}

int QueryCompositeOpDetailBySyncIter(uint32_t syncIter, std::map<uint32_t, std::vector<CompositeOpDetail>>& detail)
{
    detail.clear();

    // 1. 取本 sync 周期的全部算子明细（迁移前无排序，按查询返回顺序消费）。
    auto details = HcclSim::Storage::Select<OpDetailTab>(HcclSim::Storage::Eq(&OpDetailTab::syncIter, syncIter));
    if (!details.ok()) {
        HCCL_VM_ERROR("Query opDetails failed for syncIter: {}", syncIter);
        return -1;
    }

    auto memTable = HcclSim::Storage::AcquireTable<OpMemInfoTab>();
    if (!memTable.ok()) {
        return FailWith("query composite opDetail", memTable.diagnostic);
    }

    for (auto& detailRow : *details.value) {
        CompositeOpDetail comp = {};
        comp.deviceId = detailRow.deviceId;
        comp.rankId = detailRow.rankId;
        comp.commId = detailRow.commId;
        comp.detail = detailRow;

        const uint32_t opDetailId = detailRow.id;
        const uint32_t pid = detailRow.pid;

        // 2. 取该算子的内存信息（迁移前取查询结果首行）。
        auto mems = memTable->Select(HcclSim::Storage::Eq(&OpMemInfoTab::opDetailId, opDetailId));
        if (!mems.ok()) {
            return FailWith("query composite opDetail", mems.diagnostic);
        }
        if (!mems.value->empty()) {
            comp.memInfo = mems.value->front();
        }

        // 3. 取该算子的任务列表（动态 opTask_P_<pid> 表；迁移前无排序，按查询返回顺序消费）。
        auto taskTable = HcclSim::Storage::AcquireTable<OpTaskTab>(TableScope::Process(pid));
        if (taskTable.ok()) {
            auto tasks = taskTable->Select(HcclSim::Storage::Eq(&OpTaskTab::opDetailId, opDetailId));
            if (tasks.ok()) {
                comp.tasks = std::move(*tasks);
            }
        }

        detail[comp.rankId].push_back(std::move(comp));
    }

    return 0;
}

int QueryNewestOpDeatailIdByPid(uint64_t pid, uint32_t& OpDetailId)
{
    // 标量查询：只取最新 id，Fields 投影剥离 opDetail/opExtInfo 大 BLOB。
    auto rows = HcclSim::Storage::Select<OpDetailTab>(
        HcclSim::Storage::Eq(&OpDetailTab::pid, pid),
        HcclSim::Storage::OrderBy(&OpDetailTab::id, HcclSim::Storage::SortDirection::DESC), HcclSim::Storage::Limit(1),
        HcclSim::Storage::Fields(&OpDetailTab::id, &OpDetailTab::pid));
    if (!rows.ok() || rows.value->empty()) {
        return -1;
    }
    OpDetailId = rows.value->front().id;
    return 0;
}

int QueryOpExecutionIndexEntries(std::vector<OpExecutionIndexEntry>& out)
{
    // 迁移前语义：SELECT id, deviceId, rankId, commId, opIter, rankSize
    // FROM opDetails ORDER BY id，逐行映射为执行索引条目。
    // 字段投影：只回传上述 6 个小字段，不拖 opDetail/opExtInfo 大 BLOB——
    // 256p 全表 extInfo（每行含 N×N 矩阵 4+8×rankSize² 字节）约 268MB，
    // 并发写冲突时原子快照会撞 128MiB 固定上限；投影后整表 ~50KB。
    // 调用方（LoadOpExecutionKeys）只消费这 6 个字段，BLOB 由逐算子
    // 查询（QueryCompositeOpDetailByOpIter，单行）自行读取。
    auto rows = HcclSim::Storage::Select<OpDetailTab>(
        HcclSim::Storage::All<OpDetailTab>(),
        HcclSim::Storage::OrderBy(&OpDetailTab::id, HcclSim::Storage::SortDirection::ASC),
        HcclSim::Storage::Fields(
            &OpDetailTab::id, &OpDetailTab::deviceId, &OpDetailTab::rankId, &OpDetailTab::commId, &OpDetailTab::opIter,
            &OpDetailTab::rankSize));
    if (!rows.ok()) {
        HCCL_VM_ERROR("Failed to query operator execution index");
        return -1;
    }

    out.clear();
    out.reserve(rows.value->size());
    for (const auto& rec : *rows.value) {
        OpExecutionIndexEntry entry;
        entry.opDetailId = rec.id;
        entry.deviceId = rec.deviceId;
        entry.rankId = rec.rankId;
        entry.commId = rec.commId;
        entry.opIter = rec.opIter;
        entry.rankSize = rec.rankSize;
        out.push_back(entry);
    }
    return 0;
}

int QueryOpDetailIdentity(uint32_t opDetailId, uint64_t& commId, uint32_t& rankId, uint32_t& deviceId)
{
    // 迁移前语义：SELECT commId, rankId, deviceId FROM opDetails WHERE id = ?。
    // Fields 投影对齐迁移前列清单，剥离 opDetail/opExtInfo 大 BLOB；SelectOne
    // 直接物化单对象（不经 vector 中转）。
    auto found = HcclSim::Storage::SelectOne<OpDetailTab>(
        HcclSim::Storage::Eq(&OpDetailTab::id, opDetailId),
        HcclSim::Storage::Fields(&OpDetailTab::id, &OpDetailTab::commId, &OpDetailTab::rankId, &OpDetailTab::deviceId));
    if (!found.ok() || !found.value.has_value()) {
        return -1;
    }
    commId = found.value->commId;
    rankId = found.value->rankId;
    deviceId = found.value->deviceId;
    return 0;
}

int QueryOpDetailById(uint32_t opDetailId, OpDetailTab& out)
{
    // 迁移前语义：SELECT 全部 16 列 FROM opDetails WHERE id = ?——调用方（图模式
    // 录制复制）需要 opDetail/opExtInfo 完整 BLOB，不做投影。
    auto found = HcclSim::Storage::Select<OpDetailTab>(
        HcclSim::Storage::Eq(&OpDetailTab::id, opDetailId), HcclSim::Storage::Limit(1));
    if (!found.ok() || found.value->empty()) {
        HCCL_VM_ERROR("Query opDetails by id failed, opDetailId={}", opDetailId);
        return -1;
    }
    out = found.value->front();
    return 0;
}

int QueryLatestOpDetailByDeviceId(uint32_t deviceId, OpDetailTab& out)
{
    // 迁移前语义：SELECT 全部 16 列 FROM opDetails WHERE deviceId = ?
    // ORDER BY id DESC LIMIT 1。空表是合法业务状态（主流发现早于首个算子落库），
    // 失败按主线 43eff8fc 的裁定降级为 WARN。
    auto found = HcclSim::Storage::Select<OpDetailTab>(
        HcclSim::Storage::Eq(&OpDetailTab::deviceId, deviceId),
        HcclSim::Storage::OrderBy(&OpDetailTab::id, HcclSim::Storage::SortDirection::DESC), HcclSim::Storage::Limit(1));
    if (!found.ok() || found.value->empty()) {
        HCCL_VM_WARN("Failed to query latest opDetail by deviceId={}, OpDetail is empty", deviceId);
        return -1;
    }
    out = found.value->front();
    return 0;
}

int QueryLatestOpDetailStreamByDeviceId(uint32_t deviceId, uint32_t& opDetailId, uint64_t& streamId)
{
    // 主流发现摘要：SelectOne + Fields 只取 id/streamId 两列（不读 opDetail/
    // opExtInfo 大 BLOB——此前完整 latestOpDetail 每次轮询拖两个 BLOB）。
    // 条件/排序与完整行接口一致（deviceId 相等、id 降序取首条）；空表是合法
    // 业务状态，保持同款 WARN + -1，不升格为业务失败。
    auto found = HcclSim::Storage::SelectOne<OpDetailTab>(
        HcclSim::Storage::Eq(&OpDetailTab::deviceId, deviceId),
        HcclSim::Storage::Fields(&OpDetailTab::id, &OpDetailTab::streamId),
        HcclSim::Storage::OrderBy(&OpDetailTab::id, HcclSim::Storage::SortDirection::DESC));
    if (!found.ok() || !found.value.has_value()) {
        HCCL_VM_WARN("Failed to query latest opDetail by deviceId={}, OpDetail is empty", deviceId);
        return -1;
    }
    opDetailId = found.value->id;
    streamId = found.value->streamId;
    return 0;
}

int QueryCompositeOpDetailByOpIter(
    const std::string& commName, uint64_t commHash, uint32_t opIter, std::vector<CompositeOpDetail>& details)
{
    // 迁移前语义：RunnerDB::GetByPred<Communicator> 按 (comm_id, comm_hash) 过滤通信域成员；
    // 再逐成员取指定 opIter 的 opDetail（首行）、其 opMemInfo（首行）与动态 opTask 表任务。
    // Communicator 表属 sim::runtime 域（RUNNER_DB），经 HcclSim::Storage 原子接口跨库查询。
    details.clear();

    auto commMembers = HcclSim::Storage::Select<sim::runtime::Communicator>(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&sim::runtime::Communicator::comm_id, commName),
        HcclSim::Storage::Eq(&sim::runtime::Communicator::comm_hash, commHash)));
    if (!commMembers.ok()) {
        HCCL_VM_ERROR("Query communicator members failed, commName={}, commHash={}", commName, commHash);
        return -1;
    }

    for (const auto& comm : *commMembers.value) {
        auto opDetail = HcclSim::Storage::SelectOne<OpDetailTab>(
            HcclSim::Storage::And(
                HcclSim::Storage::Eq(&OpDetailTab::commId, comm.id),
                HcclSim::Storage::Eq(&OpDetailTab::opIter, opIter)),
            HcclSim::Storage::OrderBy(&OpDetailTab::id, HcclSim::Storage::SortDirection::ASC));
        if (!opDetail.ok()) {
            HCCL_VM_ERROR("Query opDetails failed, commId={}, opIter={}", comm.id, opIter);
            return -1;
        }

        CompositeOpDetail comp = {};
        comp.deviceId = opDetail->deviceId;
        comp.rankId = opDetail->rankId;
        comp.commId = opDetail->commId;
        comp.detail = *opDetail;

        const uint32_t opDetailId = opDetail->id;

        // 查询 MemInfo（迁移前取查询结果首行；无匹配视为失败）。
        auto mem
            = HcclSim::Storage::SelectOne<OpMemInfoTab>(HcclSim::Storage::Eq(&OpMemInfoTab::opDetailId, opDetailId));
        if (!mem.ok()) {
            HCCL_VM_ERROR("Query opDetails mem failed, opDetailId={}", opDetailId);
            return -1;
        }
        comp.memInfo = *mem;

        // 查询 Tasks（动态 opTask_P_<pid> 表；无匹配视为失败）。
        const uint32_t pid = opDetail->pid;
        auto taskTable = HcclSim::Storage::AcquireTable<OpTaskTab>(TableScope::Process(pid));
        if (!taskTable.ok()) {
            HCCL_VM_ERROR("Query opDetails tasks failed, pid={}, opDetailId={}", pid, opDetailId);
            return -1;
        }
        auto tasks = taskTable->Select(HcclSim::Storage::Eq(&OpTaskTab::opDetailId, opDetailId));
        if (!tasks.ok() || tasks.value->empty()) {
            HCCL_VM_ERROR("Query opDetails tasks failed, pid={}, opDetailId={}", pid, opDetailId);
            return -1;
        }
        comp.tasks = std::move(*tasks);

        details.push_back(std::move(comp));
    }

    return 0;
}

// ==== Sync Record 分区 ====

int InsertSyncRecord(SyncRecordTab& rec)
{
    rec.syncIter = g_syncIterCounter++;

    rec.id = 0;
    auto inserted = HcclSim::Storage::Insert<SyncRecordTab>(rec);
    if (!inserted.ok()) {
        return FailWith("insert syncRecords", inserted.diagnostic);
    }
    return 0;
}

int QuerySyncRecordByStatus(uint8_t status, std::vector<SyncRecordTab>& out)
{
    auto rows = HcclSim::Storage::Select<SyncRecordTab>(HcclSim::Storage::Eq(&SyncRecordTab::status, status));
    if (!rows.ok()) {
        return FailWith("query syncRecords by status", rows.diagnostic);
    }
    out = std::move(*rows.value);
    return 0;
}

int UpdateSyncRecordStatus(std::vector<SyncRecordTab>& syncRecord)
{
    // 迁移前语义：全部状态更新在同一事务内提交，任一失败整体回滚。
    // CONFLICT 时按引擎契约重建整个事务重试（同 InsertOpDetailAndMem）。
    for (int attempt = 0; attempt < kTxConflictRetryLimit; ++attempt) {
        auto transaction = HcclSim::Storage::BeginTransaction();
        if (!transaction.ok()) {
            return FailWith("update syncRecords status", transaction.diagnostic);
        }
        auto table = HcclSim::Storage::AcquireTable<SyncRecordTab>();
        if (!table.ok()) {
            (void)transaction->Rollback();
            return FailWith("update syncRecords status", table.diagnostic);
        }
        for (const auto& rec : syncRecord) {
            auto staged = table->Update(
                HcclSim::Storage::Eq(&SyncRecordTab::id, rec.id),
                HcclSim::Storage::Set(&SyncRecordTab::status, rec.status), *transaction);
            if (!staged.ok()) {
                (void)transaction->Rollback();
                return FailWith("update syncRecords status", staged.diagnostic);
            }
        }
        auto committed = transaction->Commit();
        if (committed.ok()) {
            return 0;
        }
        if (committed.code != HcclSim::Storage::DbCode::CONFLICT) {
            return FailWith("update syncRecords status", committed.diagnostic);
        }
    }
    return FailWith("update syncRecords status", "transaction conflict retries exhausted");
}

int QuerySyncRecordAll(std::vector<SyncRecordTab>& out)
{
    auto rows = HcclSim::Storage::Select<SyncRecordTab>(HcclSim::Storage::All<SyncRecordTab>());
    if (!rows.ok()) {
        return FailWith("query syncRecords", rows.diagnostic);
    }
    out = std::move(*rows.value);
    return 0;
}

// ==== CCU / Jetty 分区 ====

int InsertCcuChannel(CcuChannelTab& rec)
{
    if (rec.jettyNum > HcclSim::HCCL_VM_MAX_CHANNEL_JETTIES) {
        HCCL_VM_ERROR(
            "ccu channel jettyNum {} exceeds capacity {}", rec.jettyNum, HcclSim::HCCL_VM_MAX_CHANNEL_JETTIES);
        return -1;
    }
    rec.id = 0;
    auto inserted = HcclSim::Storage::Insert<CcuChannelTab>(rec);
    if (!inserted.ok()) {
        return FailWith("insert ccuChannels", inserted.diagnostic);
    }
    return 0;
}

int InsertJettyMap(JettyMapTab& rec)
{
    rec.opDetailId = g_currOpDetailId;

    rec.id = 0;
    auto inserted = HcclSim::Storage::Insert<JettyMapTab>(rec);
    if (!inserted.ok()) {
        return FailWith("insert JettyMaps", inserted.diagnostic);
    }
    return 0;
}

int UpdateAndInsertByCcuId(
    uint64_t& ccuId, uint32_t deviceId, uint32_t dieId, uint32_t startId, uint32_t instrCount, uint32_t instrOffset,
    uint32_t instrInfoSize, const void* instrInfo, std::vector<uint8_t>* mergedInstrSpace, uint32_t* totalInstrCount)
{
    constexpr size_t kInstrSpaceSize = sizeof(CcuInstrResTab::instrSpace);

    if (instrCount > HcclSim::CCU_INSTRUCTION_NUM || instrInfo == nullptr || instrOffset > kInstrSpaceSize
        || instrInfoSize > kInstrSpaceSize - instrOffset) {
        HCCL_VM_ERROR(
            "invalid instruction range: instrCount={}, offset={}, size={}, space={}, instrInfo={}", instrCount,
            instrOffset, instrInfoSize, kInstrSpaceSize, instrInfo == nullptr ? "null" : "valid");
        return -1;
    }

    auto table = HcclSim::Storage::AcquireTable<CcuInstrResTab>();
    if (!table.ok()) {
        return FailWith("update ccuInstrRes", table.diagnostic);
    }

    // 迁移前语义：按 device+die 查现有记录，命中则读改写整块指令空间，否则插入新记录。
    auto rows = table->Select(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&CcuInstrResTab::deviceId, deviceId),
        HcclSim::Storage::Eq(&CcuInstrResTab::dieId, dieId)));
    if (!rows.ok()) {
        return FailWith("update ccuInstrRes", rows.diagnostic);
    }

    if (!rows.value->empty()) {
        const auto& existing = rows.value->front();
        const uint32_t existId = existing.id;
        const uint8_t* const spaceBase = reinterpret_cast<const uint8_t*>(existing.instrSpace);
        std::vector<uint8_t> space(spaceBase, spaceBase + kInstrSpaceSize);
        std::memcpy(space.data() + instrOffset, instrInfo, instrInfoSize);

        if (startId > HcclSim::CCU_INSTRUCTION_NUM || instrCount > HcclSim::CCU_INSTRUCTION_NUM - startId) {
            HCCL_VM_ERROR(
                "ccu instruction count overflow, startId={}, appendCount={}, capacity={}", startId, instrCount,
                HcclSim::CCU_INSTRUCTION_NUM);
            return -1;
        }
        // 支持空隙，startId 为指令起始地址，instrCount 为指令数量（bubble 指令计数修正）：
        // 多批下发时总指令数取已存指令数与本次末尾地址的较大者，避免回退。
        const uint32_t newCount = std::max(existing.instrCount, startId + instrCount);

        auto updated = table->Update(
            HcclSim::Storage::Eq(&CcuInstrResTab::id, existId),
            HcclSim::Storage::Set(&CcuInstrResTab::instrSpace, space),
            HcclSim::Storage::Set(&CcuInstrResTab::instrCount, newCount));
        if (!updated.ok()) {
            return FailWith("update ccuInstrRes", updated.diagnostic);
        }
        if (mergedInstrSpace != nullptr) {
            *mergedInstrSpace = std::move(space);
        }
        if (totalInstrCount != nullptr) {
            *totalInstrCount = newCount;
        }
        ccuId = existId;
        return 0;
    }

    CcuInstrResTab rec{};
    rec.deviceId = deviceId;
    rec.dieId = dieId;
    // 支持空隙：新记录的总指令数同样按本次末尾地址记录。
    const uint32_t newCount = startId + instrCount;
    rec.instrCount = newCount;
    // 字节偏移：等价于对一维 uint8_t 缓冲的偏移，不得对二维数组做行指针偏移。
    uint8_t* const dstBase = reinterpret_cast<uint8_t*>(rec.instrSpace);
    std::memcpy(dstBase + instrOffset, instrInfo, instrInfoSize);
    rec.id = 0;
    auto inserted = table->Insert(rec);
    if (!inserted.ok()) {
        return FailWith("insert ccuInstrRes", inserted.diagnostic);
    }
    if (mergedInstrSpace != nullptr) {
        *mergedInstrSpace = std::vector<uint8_t>(dstBase, dstBase + kInstrSpaceSize);
    }
    if (totalInstrCount != nullptr) {
        *totalInstrCount = newCount;
    }
    ccuId = rec.id;
    return 0;
}

int QueryCcuChannelAll(std::vector<CcuChannelTab>& out)
{
    auto rows = HcclSim::Storage::Select<CcuChannelTab>(HcclSim::Storage::All<CcuChannelTab>());
    if (!rows.ok()) {
        return FailWith("query ccuChannels", rows.diagnostic);
    }
    for (const auto& ch : *rows.value) {
        if (ch.jettyNum > HcclSim::HCCL_VM_MAX_CHANNEL_JETTIES) {
            HCCL_VM_ERROR(
                "persisted ccu channel jettyNum {} exceeds capacity {}", ch.jettyNum,
                HcclSim::HCCL_VM_MAX_CHANNEL_JETTIES);
            return -1;
        }
    }
    out = std::move(*rows.value);
    return 0;
}

int QueryJettyMapAll(std::vector<JettyMapTab>& out)
{
    auto rows = HcclSim::Storage::Select<JettyMapTab>(HcclSim::Storage::All<JettyMapTab>());
    if (!rows.ok()) {
        return FailWith("query JettyMaps", rows.diagnostic);
    }
    out = std::move(*rows.value);
    return 0;
}

int InsertCcuInstrRes(CcuInstrResTab& rec)
{
    if (rec.instrCount > HcclSim::CCU_INSTRUCTION_NUM) {
        HCCL_VM_ERROR("ccu instruction count {} exceeds capacity {}", rec.instrCount, HcclSim::CCU_INSTRUCTION_NUM);
        return -1;
    }
    rec.id = 0;
    auto inserted = HcclSim::Storage::Insert<CcuInstrResTab>(rec);
    if (!inserted.ok()) {
        return FailWith("insert ccuInstrRes", inserted.diagnostic);
    }
    return 0;
}

int InsertCcuInstr(CcuInstrTab& rec)
{
    rec.id = 0;
    auto inserted = HcclSim::Storage::Insert<CcuInstrTab>(rec);
    if (!inserted.ok()) {
        return FailWith("insert ccuInstr", inserted.diagnostic);
    }
    return 0;
}

int QueryCcuInstrResAll(std::vector<CcuInstrResTab>& out)
{
    auto rows = HcclSim::Storage::Select<CcuInstrResTab>(HcclSim::Storage::All<CcuInstrResTab>());
    if (!rows.ok()) {
        return FailWith("query ccuInstrRes", rows.diagnostic);
    }
    for (const auto& res : *rows.value) {
        if (res.instrCount > HcclSim::CCU_INSTRUCTION_NUM) {
            HCCL_VM_ERROR(
                "persisted ccu instruction count {} exceeds capacity {}", res.instrCount, HcclSim::CCU_INSTRUCTION_NUM);
            return -1;
        }
    }
    out = std::move(*rows.value);
    return 0;
}

int QueryCcuResourceMetaCount(uint32_t& instrLoadCnt, uint32_t& channelCnt)
{
    // 迁移前语义：ccuInstr 表记录每次微码下发，ccuInstrRes 表记录已创建的资源行；
    // 两者行数之和可同时感知微码下发和资源行补齐，避免轮询时漏掉中间状态。
    instrLoadCnt = 0;
    channelCnt = 0;

    auto instrCnt = HcclSim::Storage::Count<CcuInstrTab>(HcclSim::Storage::All<CcuInstrTab>());
    if (!instrCnt.ok()) {
        HCCL_VM_ERROR("Query ccu resource meta count failed");
        return -1;
    }
    auto instrResCnt = HcclSim::Storage::Count<CcuInstrResTab>(HcclSim::Storage::All<CcuInstrResTab>());
    if (!instrResCnt.ok()) {
        HCCL_VM_ERROR("Query ccu resource meta count failed");
        return -1;
    }
    auto chCnt = HcclSim::Storage::Count<CcuChannelTab>(HcclSim::Storage::All<CcuChannelTab>());
    if (!chCnt.ok()) {
        HCCL_VM_ERROR("Query ccu resource meta count failed");
        return -1;
    }

    instrLoadCnt = static_cast<uint32_t>(*instrCnt + *instrResCnt);
    channelCnt = static_cast<uint32_t>(*chCnt);
    return 0;
}

int InsertHalfRTT(HalfRTTTab& rec)
{
    rec.id = 0;
    auto inserted = HcclSim::Storage::Insert<HalfRTTTab>(rec);
    if (!inserted.ok()) {
        return FailWith("insert halfRTT", inserted.diagnostic);
    }
    return 0;
}

int QueryHalfRTTAll(std::vector<HalfRTTTab>& out)
{
    auto rows = HcclSim::Storage::Select<HalfRTTTab>(HcclSim::Storage::All<HalfRTTTab>());
    if (!rows.ok()) {
        return FailWith("query halfRTT", rows.diagnostic);
    }
    out = std::move(*rows.value);
    return 0;
}

// ==== Dynamic Task 分区 ====

int InsertOpTask(OpTaskTab& rec, bool isDevice)
{
    rec.opDetailId = g_currOpDetailId;
    // host进程使用pid，AICPU模式的device进程使用ppid(即host进程的pid)
    const uint64_t tabPid = isDevice ? static_cast<uint64_t>(getppid()) : static_cast<uint64_t>(getpid());
    // 迁移前语义：pid 随任务行持久化，runner 完成任务时按 task.pid 解析进程 scope 表刷新 isDone。
    rec.pid = static_cast<uint32_t>(tabPid);

    // 动态 scope 表：按进程身份解析物理表（opTask_P_<pid>），首次访问时惰性建表。
    rec.id = 0;
    auto inserted = HcclSim::Storage::Insert<OpTaskTab>(TableScope::Process(tabPid), rec);
    if (!inserted.ok()) {
        return FailWith("insert opTask", inserted.diagnostic);
    }
    return 0;
}

int QueryOpTasksByStreamId(uint64_t streamId, std::vector<OpTaskTab>& tasks)
{
    // 迁移前语义：按当前进程 pid 解析物理表 opTask_P_<pid> 并过滤 streamId。
    const uint64_t tabPid = static_cast<uint64_t>(getpid());
    auto rows = HcclSim::Storage::Select<OpTaskTab>(
        TableScope::Process(tabPid), HcclSim::Storage::Eq(&OpTaskTab::streamId, streamId));
    if (!rows.ok()) {
        HCCL_VM_ERROR("Query OpTasks by streamId failed, streamId={}", streamId);
        return -1;
    }
    tasks = std::move(*rows.value);
    return 0;
}

int QueryOpTaskCompletionByStreamId(uint64_t streamId, uint64_t& total, uint64_t& done)
{
    // 单条聚合语句同时返回 stream 任务总数与已完成数：统计在数据库端完成，
    // 不加载任务行（optaskMeta BLOB 不回传，无需构造 vector<OpTaskTab>）；
    // match 判据与原"遍历统计 task.isDone"一致。scope 沿用当前 pid 动态表。
    const uint64_t tabPid = static_cast<uint64_t>(getpid());
    auto counted = HcclSim::Storage::CountMatches<OpTaskTab>(
        TableScope::Process(tabPid), HcclSim::Storage::Eq(&OpTaskTab::streamId, streamId),
        HcclSim::Storage::Ne(&OpTaskTab::isDone, false));
    if (!counted.ok() || !counted.value.has_value()) {
        HCCL_VM_ERROR("Query OpTask completion by streamId failed, streamId={}", streamId);
        return -1;
    }
    total = counted.value->total;
    done = counted.value->matched;
    return 0;
}

int FinishOpTask(const OpTaskTab& task)
{
    // 迁移前语义：UPDATE opTask_P_<task.pid> SET isDone = 1 WHERE id = task.id。
    auto updated = HcclSim::Storage::Update<OpTaskTab>(
        TableScope::Process(task.pid), HcclSim::Storage::Eq(&OpTaskTab::id, task.id),
        HcclSim::Storage::Set(&OpTaskTab::isDone, true));
    if (!updated.ok()) {
        HCCL_VM_ERROR("Update OpTask isDone failed, taskId={}, pid={}", task.id, task.pid);
        return -1;
    }
    return 0;
}

int QueryOpTaskTabNames(std::vector<std::string>& names)
{
    // 迁移前语义：SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'OpTaskTab_P_%'。
    // 动态 PID scope 表物理名 = <canonical 逻辑表名>_P_<pid>（见 InsertOpTask 约定）。
    // 经进程存储会话对 OP_DATA_DB 执行 ListTables（以 OpDetailTab 的库归属解析到
    // OP_DATA_DB 引擎，在库连接上枚举全部用户表，不受锚点表影响），再按前缀过滤。
    auto lease = HcclSim::Storage::ProcessStorageContext::AcquireReady();
    if (!lease.ok()) {
        HCCL_VM_ERROR("Query all opTask table names failed: {}", lease.diagnostic);
        return -1;
    }
    auto tables = lease.value->Get().ListTables(HcclSim::Storage::RecordSchema<OpDetailTab>::DatabaseId());
    if (!tables.ok()) {
        HCCL_VM_ERROR("Query all opTask table names failed: {}", tables.diagnostic);
        return -1;
    }
    constexpr std::string_view kOpTaskPrefix = "OpTaskTab_P_";
    names.clear();
    for (const auto& table : *tables.value) {
        if (table.compare(0, kOpTaskPrefix.size(), kOpTaskPrefix) == 0) {
            names.push_back(table);
        }
    }
    return 0;
}

int QueryAllOpTasks(std::vector<OpTaskTab>& tasks, bool filterDone)
{
    // 迁移前语义：枚举全部动态 opTask_P_<pid> 表，逐表取行（filterDone 时仅未完成）
    // 并按 id 升序汇总。取数方式改为同表多作用域批量（每 64 个作用域一窗，
    // 消除同 endpoint 内逐表两阶段等待）；枚举/表名合法性/PID 解析/每表
    // id ASC/返回码与原逐表路径一致。
    std::vector<std::string> taskTabNames;
    auto ret = QueryOpTaskTabNames(taskTabNames);
    if (ret != 0) {
        HCCL_VM_ERROR("QueryAllOpTasks failed!");
        return ret;
    }
    tasks.clear();
    constexpr std::string_view kOpTaskPrefix = "OpTaskTab_P_";
    // 先解析全部 PID（顺序与原枚举一致；任一非法表名仍返回原错误）
    std::vector<HcclSim::Storage::TableScope> scopes;
    scopes.reserve(taskTabNames.size());
    for (const std::string& tableName : taskTabNames) {
        // 表名 "OpTaskTab_P_<pid>" → 解析进程 pid，保证记录经正确的进程 scope 读取。
        const std::string pidStr = tableName.substr(kOpTaskPrefix.size());
        char* end = nullptr;
        const unsigned long long parsedPid = std::strtoull(pidStr.c_str(), &end, 10);
        if (end == pidStr.c_str() || *end != '\0') {
            HCCL_VM_ERROR("QueryAllOpTasks failed, invalid task table name={}", tableName);
            return -1;
        }
        scopes.emplace_back(HcclSim::Storage::TableScope::Process(static_cast<uint64_t>(parsedPid)));
    }
    // 一轮只构造一次条件（各窗口共用同一 const 引用）
    const HcclSim::Storage::Condition<OpTaskTab> where
        = filterDone ? HcclSim::Storage::Eq(&OpTaskTab::isDone, false) : HcclSim::Storage::All<OpTaskTab>();
    constexpr std::size_t kScopeWindow = 64; // 批量窗口（同表多作用域）
    for (std::size_t first = 0; first < scopes.size(); first += kScopeWindow) {
        const std::size_t count = std::min(kScopeWindow, scopes.size() - first);
        const std::vector<HcclSim::Storage::TableScope> window(scopes.begin() + first, scopes.begin() + first + count);
        auto rows = HcclSim::Storage::SelectMany<OpTaskTab>(
            window, where, HcclSim::Storage::OrderBy(&OpTaskTab::id, HcclSim::Storage::SortDirection::ASC));
        if (!rows.ok()) {
            // 主线语义（逐表查询失败 continue 而非整体失败）：批量窗口失败时退回
            // 逐作用域查询，仅跳过真正失败的作用域（枚举后表被并发销毁等瞬态），
            // 其余作用域照常汇总。
            for (std::size_t i = 0; i < count; ++i) {
                auto single = HcclSim::Storage::Select<OpTaskTab>(
                    window[i], where, HcclSim::Storage::OrderBy(&OpTaskTab::id, HcclSim::Storage::SortDirection::ASC));
                if (!single.ok()) {
                    HCCL_VM_WARN("Query OpTasks failed, skip scope, firstScopeIndex={}, offset={}", first, i);
                    continue;
                }
                for (OpTaskTab& record : *single.value) {
                    tasks.push_back(std::move(record));
                }
            }
            continue;
        }
        // 按原表顺序、每表原排序把返回记录 move 进 tasks（不做跨表全局排序）
        for (std::vector<OpTaskTab>& tableRows : *rows.value) {
            for (OpTaskTab& record : tableRows) {
                tasks.push_back(std::move(record));
            }
        }
    }
    return 0;
}

// ==== Lifecycle / Maintenance 分区 ====

int EnsureProcessStorageSession()
{
    // 迁移前语义：建立数据库连接并完成全部表结构初始化（幂等）。
    return EnsureStorageSession() ? 0 : -1;
}

int SetDbConfig(DBConfig& config)
{
    // 迁移前语义：关闭当前连接并按新路径重连（加载备份文件）。
    // Runner/Checker 插件共享库持有自己的冷 storage runtime 状态：先建立默认进程会话
    // （等价迁移前 OpDbOps::Instance() 首次构造默认连接），再把 OP_DATA_DB 对齐到
    // 任务文件记录的数据位置；会话已安装时 Ensure 幂等成功，不改变其他逻辑库路由。
    // 维护入口豁免：对齐不是普通 CRUD——冷共享库状态需先经同一 bootstrap
    // 建立 ready 会话；普通 CRUD 不保留逐函数 Ensure（由 AcquireReady 统一惰性建立）。
    // 业务只表达维护意图（对齐到记录的数据位置）；部署形态差异（SQLite 路径
    // 重配 / Redis 部署保持路由）由装配层裁定，业务不感知具体后端类型。
    if (!EnsureStorageSession()) {
        return -1;
    }
    auto reconfigured = sim::ReconfigureDatabaseToRecordedLocation("OP_DATA_DB", config.dbPath);
    if (!reconfigured.ok()) {
        HCCL_VM_ERROR("[operation_data] reconnect database {} failed: {}", config.dbPath, reconfigured.diagnostic);
        return -1;
    }
    return 0;
}

int BackupOperationData(const std::string& destinationPath)
{
    // 迁移前语义：OpDbOps::Backup 在线快照整库复制到目标文件。
    // 专用备份连接经唯一 simulator bootstrap 执行（sqlite3_backup 在线快照语义）。
    EnsureStorageSession();
    auto backedUp = sim::BackupSimulatorDatabase(
        "OP_DATA_DB", HcclSim::Storage::BackupRequest{destinationPath, HcclSim::Storage::BackupPurpose::OnlineReset});
    if (!backedUp.ok()) {
        return FailWith("BackupOperationData", backedUp.diagnostic);
    }
    return 0;
}

int ClearStaticData()
{
    // 迁移前语义：reset 命令逐表 DELETE FROM（保留自增序列）清空全部静态操作数据表。
    auto cleared = StorageRuntime::ClearStaticData("OP_DATA_DB");
    if (!cleared.ok()) {
        return FailWith("ClearStaticData", cleared.diagnostic);
    }
    return 0;
}

int DropDynamicData()
{
    // 迁移前语义：reset 命令枚举物理表并逐 PID 销毁全部动态任务表。
    EnsureStorageSession();
    auto dropped = StorageRuntime::DropDynamicData("OP_DATA_DB");
    if (!dropped.ok()) {
        return FailWith("DropDynamicData", dropped.diagnostic);
    }
    return 0;
}
} // namespace sim::operation
