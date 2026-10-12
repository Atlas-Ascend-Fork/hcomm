/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <cstring>
#include <sstream>
#include <utility>
#include <nlohmann_json/json.hpp>
#include "dump/dump_manager.h"
#include "dump/dump_run_manifest.h"
#include "dump/validation_issue_recorder.h"
#include "dump_v3/dump_v3_manager.h"
#include "dfx/dag_graphviz_dump.h"
#include "setting_manager.h"
#include "storage_manager.h"
#include "checker.h"
#include "framework/composite_op_grouping.h"
#include "framework/big_graph_check/big_graph_checker.h"
#include "sim_log.h"
#include "stage_profiler.h"
#include "framework/task_graph_generator_v3/ccu_graph_generator_v3/ccu_all_rank_param_recorder_v3.h"
#include "sim_loader.h"
#include "sim_common_defs.h"
#include "utils/check_utils.h"
#include "utils/error_codes.h"

using json = nlohmann::json;
loader::Loader g_loader;

// 使用原子变量控制程序生命周期
std::atomic<bool> g_keep_running{true};
std::mutex g_run_checker_mutex;
std::mutex g_worker_mutex;
std::thread g_worker_thread;

enum class CheckerStatus : uint8_t { SUCCESS, FAILED, DISABLE, NOT_EXECUTED };
static constexpr const char* CHECKER_STATUS_TEXT[] = {"success", "failed", "disable", "not_executed"};

struct SingleOpCheckerResult {
    sim::operation::OpExecutionKey key;
    CheckerStatus status;
};

json BuildOpParamSummaryJson(const HcclSim::CheckerParam& param);

static bool IsAivOpExpansionMode(uint32_t opExpansionMode)
{
    constexpr uint32_t SIM_OP_EXPANSION_MODE_AIV = 2U;
    return opExpansionMode == SIM_OP_EXPANSION_MODE_AIV;
}

static bool HasAivGraphTask(const std::vector<std::vector<sim::operation::OpTaskTab>>& allTasks)
{
    for (const auto& rankTasks : allTasks) {
        for (const auto& task : rankTasks) {
            if (task.optaskMeta.size() < sizeof(HcclTaskMetaData)) {
                continue;
            }
            HcclTaskMetaData metaData;
            std::memcpy(&metaData, task.optaskMeta.data(), sizeof(HcclTaskMetaData));
            if (metaData.taskType == HccLTaskMetaType::AIV_GRAPH) {
                return true;
            }
        }
    }
    return false;
}

static bool IsSingleRankWithNoTask(const sim::operation::OpExecution& opExecution)
{
    if (opExecution.deviceRecords.size() != 1) {
        return false;
    }

    const sim::operation::DeviceOpExecutionRecord& op = opExecution.deviceRecords.front();
    return op.detail.rankSize == 1 && op.tasks.empty();
}

static bool IsSingleRankWithNoTask(const HcclSim::BigGraphCheckV3::BigGraphData& data)
{
    if (data.operators.empty()) {
        return false;
    }

    for (const HcclSim::BigGraphCheckV3::OpParam& opParam : data.operators) {
        if (opParam.ranks.size() != 1) {
            return false;
        }
        const HcclSim::BigGraphCheckV3::OperatorRankData& rankData = opParam.ranks.front();
        if (rankData.op.detail.rankSize != 1 || !rankData.taskMetas.empty()) {
            return false;
        }
    }
    return true;
}

static const char* HcclReduceOpToString(HcclReduceOp t)
{
    switch (t) {
        case HCCL_REDUCE_SUM:
            return "SUM";
        case HCCL_REDUCE_PROD:
            return "PROD";
        case HCCL_REDUCE_MAX:
            return "MAX";
        case HCCL_REDUCE_MIN:
            return "MIN";
        default:
            return "Unknown";
    }
}

static void AppendApplicableRoleFields(std::ostringstream& os, const HcclSim::CheckerParam& param)
{
    switch (param.cmdType) {
        case HCCL_CMD_SEND:
        case HCCL_CMD_RECEIVE:
            os << ", sourceRank=" << param.srcRank << ", targetRank=" << param.dstRank;
            break;
        case HCCL_CMD_BROADCAST:
        case HCCL_CMD_REDUCE:
        case HCCL_CMD_SCATTER:
            os << ", rootRank=" << param.root;
            break;
        default:
            break;
    }
}

static HcclResult LoadCheckerGlobalResources(
    loader::Loader& loader, HcclSim::StorageManager& storage, std::vector<sim::operation::CcuChannelTab>& channels,
    std::vector<sim::operation::HalfRTTTab>& halfRTT, std::vector<sim::operation::CcuInstrResTab>& instrRes)
{
    HcclResult ret = loader.GetCcuChannelInfo(channels);
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "{} Failed to load CCU channel information",
            HcclSim::MakeErrorCodeText(HcclSim::ErrorCode::CHECKER_RUNTIME_ERROR));
        return ret;
    }

    ret = loader.GetHalfRTTInfo(halfRTT);
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "{} Failed to load CCU half RTT information",
            HcclSim::MakeErrorCodeText(HcclSim::ErrorCode::CHECKER_RUNTIME_ERROR));
        return ret;
    }

    ret = loader.GetInstrResInfo(instrRes);
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "{} Failed to load CCU instruction resource information",
            HcclSim::MakeErrorCodeText(HcclSim::ErrorCode::CHECKER_RUNTIME_ERROR));
        return ret;
    }

    ret = storage.LoadHcclVmInstrData(instrRes);
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "{} Failed to load CCU instruction data into checker resources",
            HcclSim::MakeErrorCodeText(HcclSim::ErrorCode::CHECKER_RUNTIME_ERROR));
        return ret;
    }
    HCCL_VM_INFO(
        "Checker global resources loaded, channelCount={}, halfRTTCount={}, instrResCount={}", channels.size(),
        halfRTT.size(), instrRes.size());
    // instrRes 已解码进 storage；原始表每行固定 ~1MB（instrSpace[32K][32]），
    // 解码后不再使用，立即释放避免本轮全程驻留。
    instrRes.clear();
    instrRes.shrink_to_fit();
    return HcclResult::HCCL_SUCCESS;
}

static HcclResult PrepareOpParam(HcclSim::StorageManager& storage, const sim::operation::OpExecution& opExecution)
{
    for (const sim::operation::DeviceOpExecutionRecord& record : opExecution.deviceRecords) {
        if (record.detail.opDetail.size() < sizeof(::OpDetails)) {
            HCCL_VM_ERROR(
                "{} Op detail payload is too small to parse, rankId={}",
                HcclSim::MakeErrorCodeText(HcclSim::ErrorCode::CHECKER_RUNTIME_ERROR), record.rankId);
            return HcclResult::HCCL_E_PARA;
        }

        ::OpDetails opDetails{};
        std::memcpy(&opDetails, record.detail.opDetail.data(), sizeof(::OpDetails));
        sim::operation::OpDetailTab detail = record.detail;
        HcclResult ret = storage.Trans2CheckerParam(detail, opDetails);
        if (ret != HcclResult::HCCL_SUCCESS) {
            HCCL_VM_ERROR(
                "{} Failed to merge operator parameters for rankId={}",
                HcclSim::MakeErrorCodeText(HcclSim::ErrorCode::CHECKER_RUNTIME_ERROR), record.rankId);
            return ret;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

static HcclResult CheckOneOp(
    HcclSim::StorageManager& storage, std::vector<sim::operation::CcuChannelTab>& channels,
    std::vector<sim::operation::HalfRTTTab>& halfRTT, uint32_t opIdx, const sim::operation::OpExecution& opExecution,
    CheckerStatus& checkerStatus)
{
    HcclSim::ValidationIssueRecorder::GetInstance().Reset();
    HcclSim::TaskGraphGeneratorV3::AllRankParamRecorder::Global()->Reset();
    HcclSim::DumpManager& dumpManager = HcclSim::DumpManager::GetInstance();
    HcclSim::SettingManager& settingManager = HcclSim::SettingManager::GetInstance();
    const bool enableNewChecker = settingManager.IsNewCheckerEnabled();
    bool usesAivExpansionMode = false;
    checkerStatus = enableNewChecker ? CheckerStatus::NOT_EXECUTED : CheckerStatus::DISABLE;

    if (IsSingleRankWithNoTask(opExecution)) {
        checkerStatus = enableNewChecker ? CheckerStatus::SUCCESS : CheckerStatus::DISABLE;
        HCCL_VM_WARN(
            "Single-op check is skipped and treated as success because this is a single-rank "
            "operation with no tasks, opIndex={}",
            opIdx);
        return HcclResult::HCCL_SUCCESS;
    }

    HCCL_VM_INFO(
        "Start checking one op, commName={}, opIter={}, rankCount={}", opExecution.key.commName, opExecution.key.opIter,
        opExecution.deviceRecords.size());
    storage.BeginOpGroup(opExecution.key.commName, opExecution.key.commHash, opExecution.key.opIter);
    std::vector<std::vector<sim::operation::OpTaskTab>> allTasks;
    // 导入 task 耗时：逐 rank 加载算子数据（内存信息/指令资源/算子参数）。
    HcclSim::StageProfiler loadOpDataStage("OpGroup.loadOpData");
    for (const sim::operation::DeviceOpExecutionRecord& record : opExecution.deviceRecords) {
        HCCL_VM_INFO("Load one rank from this op, deviceId={}, rankId={}", record.deviceId, record.rankId);
        usesAivExpansionMode = usesAivExpansionMode || IsAivOpExpansionMode(record.detail.opExpansionMode);
        HcclResult ret = storage.LoadHcclVmSynthesisData(
            record.deviceId, record.detail.commId, opExecution.key.commName, opExecution.key.commHash,
            opExecution.key.opIter, record.rankId, record.memInfo, channels, halfRTT);
        if (ret != HcclResult::HCCL_SUCCESS) {
            HCCL_VM_ERROR(
                "{} Failed to load one rank from this op, opIndex={}, deviceId={}, rankId={}",
                HcclSim::MakeErrorCodeText(HcclSim::ErrorCode::CHECKER_RUNTIME_ERROR), opIdx, record.deviceId,
                record.rankId);
            return ret;
        }
    }
    loadOpDataStage.End();

    if (!enableNewChecker) {
        HCCL_VM_ERROR(
            "{} This op is skipped because the new checker is disabled, opIndex={}, "
            "newCheckerEnabled={}",
            HcclSim::MakeErrorCodeText(HcclSim::ErrorCode::SETTING_WARNING), opIdx, enableNewChecker);
        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult ret = PrepareOpParam(storage, opExecution);
    if (ret != HcclResult::HCCL_SUCCESS) {
        return ret;
    }

    ret = storage.FinalizeOpGroup();
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "{} Failed to finalize operator parameters for this op group, opIndex={}",
            HcclSim::MakeErrorCodeText(HcclSim::ErrorCode::CHECKER_RUNTIME_ERROR), opIdx);
        return ret;
    }

    // AllToAll family continues to use the existing matrix merge path.
    storage.MergeAll2AllVSendCountMatrix();

    if (dumpManager.IsEnabled()) {
        HcclSim::DumpRunManifest::GetInstance().SetOpParam(BuildOpParamSummaryJson(storage.GetCheckerParam()));
    }

    // 任务元数据加载耗时：allTasks 解码为 V3 task meta。
    HcclSim::StageProfiler loadTaskMetaStage("OpGroup.loadTaskMetaData");
    allTasks.reserve(opExecution.deviceRecords.size());
    for (const sim::operation::DeviceOpExecutionRecord& record : opExecution.deviceRecords) {
        allTasks.push_back(record.tasks);
    }
    ret = storage.LoadHcclVmTaskMetaData(allTasks);
    loadTaskMetaStage.End();
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "{} Failed to load V3 task metadata for this op group",
            HcclSim::MakeErrorCodeText(HcclSim::ErrorCode::CHECKER_RUNTIME_ERROR));
        return ret;
    }
    const bool hasAivGraphTask = HasAivGraphTask(allTasks);

    {
        auto checkerParamBrief = storage.GetCheckerParam();
        std::ostringstream summary;
        summary << "[Main] Op summary, opIndex=" << opIdx
                << ", collectiveType=" << HcclSim::HcclCmdTypeToString(checkerParamBrief.cmdType)
                << ", rankCount=" << checkerParamBrief.rankSize
                << ", dataType=" << HcclSim::HcclDataTypeToString(checkerParamBrief.dataType)
                << ", elementCount=" << checkerParamBrief.dataCount
                << ", reduceType=" << HcclReduceOpToString(checkerParamBrief.reduceType);
        AppendApplicableRoleFields(summary, checkerParamBrief);
        summary << ", rankCountInExecution=" << opExecution.deviceRecords.size()
                << ", usesAivExpansionMode=" << usesAivExpansionMode << ", hasAivGraphTask=" << hasAivGraphTask;
        HCCL_VM_INFO("{}", summary.str());
    }

    HCCL_VM_INFO("----------[Start CheckerV3]----------");
    // V3 检查总耗时：内部各阶段（翻译成图/单任务/同步/内存冲突/语义）见
    // checker.cc 的 "CheckerV3 stage finished" 日志。
    HcclSim::StageProfiler genAndCheckStage("OpGroup.genAndCheckGraphV3");
    HcclResult newCheckerRet = HcclSim::GenAndCheckGraphV3();
    genAndCheckStage.End();
    HCCL_VM_INFO("----------[CheckerV3 Finished]----------");
    HCCL_VM_INFO(
        "CheckerV3 finished for this op, commName={}, opIter={}, opIndex={}", opExecution.key.commName,
        opExecution.key.opIter, opIdx);
    checkerStatus = newCheckerRet == HcclResult::HCCL_SUCCESS ? CheckerStatus::SUCCESS : CheckerStatus::FAILED;

    return newCheckerRet;
}

// 大图单轮校验的阶段耗时打点：输出导入 task、翻译节点、成图、同步检查各阶段与整轮总耗时，
// 与内存冲突检查的 "Stage finished, stage=..., costMs=..." 日志风格保持一致。
static void LogBigGraphStageTime(const char* stage, HcclResult ret, std::chrono::steady_clock::time_point start)
{
    const auto elapsedMs
        = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
    HCCL_VM_INFO(
        "Big graph stage finished, stage={}, status={}, costMs={}", stage,
        ret == HcclResult::HCCL_SUCCESS ? "success" : "failed", static_cast<uint64_t>(elapsedMs));
}

static HcclResult ProcessBigGraph(loader::Loader& loader, HcclSim::BigGraphCheckV3::BigGraphCheckerV3& bigGraphChecker)
{
    HCCL_VM_INFO("----------[Start BigGraphCheckerV3]----------");
    const auto iterStart = std::chrono::steady_clock::now();

    // Each sync window owns an independent CCU register state. Keep this reset at
    // the window boundary; the V3 CCU expansion must remain continuous within the window.
    HcclSim::TaskGraphGeneratorV3::AllRankParamRecorder::Global()->Reset();
    const auto loadStart = std::chrono::steady_clock::now();
    HcclResult ret = bigGraphChecker.LoadOpData(loader);
    LogBigGraphStageTime("LoadOpData", ret, loadStart);
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "{} Failed to load all operator data for big graph, ret={}",
            HcclSim::MakeErrorCodeText(HcclSim::ErrorCode::CHECKER_RUNTIME_ERROR), static_cast<uint32_t>(ret));
        return ret;
    }

    if (IsSingleRankWithNoTask(bigGraphChecker.GetData())) {
        HCCL_VM_WARN("Big-graph check is skipped and treated as success because all operators "
                     "contain only single-rank operations with no tasks");
        return HcclResult::HCCL_SUCCESS;
    }

    const auto translateStart = std::chrono::steady_clock::now();
    ret = bigGraphChecker.TranslateTask();
    LogBigGraphStageTime("TranslateTask", ret, translateStart);
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "{} Failed to translate all operator tasks for big graph, ret={}",
            HcclSim::MakeErrorCodeText(HcclSim::ErrorCode::CHECKER_RUNTIME_ERROR), static_cast<uint32_t>(ret));
        return ret;
    }

    const auto generateStart = std::chrono::steady_clock::now();
    ret = bigGraphChecker.GenerateBigGraph();
    LogBigGraphStageTime("GenerateBigGraph", ret, generateStart);
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "{} Failed to generate big graph, ret={}",
            HcclSim::MakeErrorCodeText(HcclSim::ErrorCode::CHECKER_RUNTIME_ERROR), static_cast<uint32_t>(ret));
        return ret;
    }

    const auto* graph = bigGraphChecker.GetGraph();
    if (HcclSim::SettingManager::GetInstance().IsDagGraphvizDumpEnabled()) {
        std::string dumpPath;
        const HcclResult dumpRet
            = HcclSim::DumpDagGraphvizDot(graph == nullptr ? nullptr : graph->GetMainStartNode(), &dumpPath);
        if (dumpRet != HcclResult::HCCL_SUCCESS) {
            HCCL_VM_WARN(
                "[BigGraphCheckerV3][GraphvizDot] Failed to dump big-graph DAG dot file, ret={}",
                static_cast<uint32_t>(dumpRet));
        } else {
            const size_t nodeCount = graph == nullptr ? 0 : graph->GetNodes().size();
            HCCL_VM_INFO(
                "[BigGraphCheckerV3][GraphvizDot] Dumped all-operator DAG dot file, path={}, "
                "operatorCount={}, nodeCount={}",
                dumpPath, bigGraphChecker.GetOpParams().size(), nodeCount);
        }
    }

    const auto syncCheckStart = std::chrono::steady_clock::now();
    ret = bigGraphChecker.SyncCheck();
    LogBigGraphStageTime("SyncCheck", ret, syncCheckStart);
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "{} Big graph sync-conflict check failed, ret={}",
            HcclSim::MakeErrorCodeText(HcclSim::ErrorCode::CHECKER_RUNTIME_ERROR), static_cast<uint32_t>(ret));
        return ret;
    }

    const size_t nodeCount = graph == nullptr ? 0 : graph->GetNodes().size();
    const size_t rankCount = graph == nullptr ? 0 : graph->GetTaskQueues().size();
    HCCL_VM_INFO(
        "BigGraphCheckerV3 generated graph successfully, operatorCount={}, "
        "nodeCount={}, rankCount={}",
        bigGraphChecker.GetOpParams().size(), nodeCount, rankCount);
    LogBigGraphStageTime("Total", HcclResult::HCCL_SUCCESS, iterStart);
    HCCL_VM_INFO("----------[BigGraphCheckerV3 Finished]----------");
    return HcclResult::HCCL_SUCCESS;
}

json BuildOpParamSummaryJson(const HcclSim::CheckerParam& param)
{
    json opParamJson = json::object();
    opParamJson["cmd_type"] = static_cast<uint32_t>(param.cmdType);
    opParamJson["rank_size"] = param.rankSize;
    opParamJson["data_type"] = static_cast<uint32_t>(param.dataType);
    opParamJson["data_count"] = param.dataCount;

    if (param.cmdType == HcclCMDType::HCCL_CMD_ALLREDUCE || param.cmdType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER
        || param.cmdType == HcclCMDType::HCCL_CMD_REDUCE || param.cmdType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V) {
        opParamJson["reduce_type"] = static_cast<uint32_t>(param.reduceType);
    }

    if (param.cmdType == HcclCMDType::HCCL_CMD_SEND || param.cmdType == HcclCMDType::HCCL_CMD_RECEIVE) {
        opParamJson["src_rank"] = param.srcRank;
        opParamJson["dst_rank"] = param.dstRank;
    }

    if (param.cmdType == HcclCMDType::HCCL_CMD_BROADCAST || param.cmdType == HcclCMDType::HCCL_CMD_REDUCE
        || param.cmdType == HcclCMDType::HCCL_CMD_SCATTER) {
        opParamJson["root"] = param.root;
    }

    if (param.cmdType == HcclCMDType::HCCL_CMD_ALLGATHER_V || param.cmdType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V) {
        json vDataDesJson = json::object();
        vDataDesJson["data_type"] = param.vDataDes.dataType;
        vDataDesJson["rank_count"] = param.vDataDes.count;
        vDataDesJson["counts_size"] = param.vDataDes.counts.size();
        vDataDesJson["displs_size"] = param.vDataDes.displs.size();
        vDataDesJson["counts"] = param.vDataDes.counts;
        vDataDesJson["displs"] = param.vDataDes.displs;
        opParamJson["v_data_des"] = std::move(vDataDesJson);
    }

    if (param.cmdType == HcclCMDType::HCCL_CMD_ALLTOALL || param.cmdType == HcclCMDType::HCCL_CMD_ALLTOALLVC
        || param.cmdType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
        json all2AllDataDesJson = json::object();
        all2AllDataDesJson["send_type"] = param.all2AllDataDes.sendType;
        all2AllDataDesJson["recv_type"] = param.all2AllDataDes.recvType;
        all2AllDataDesJson["send_count"] = param.all2AllDataDes.sendCount;
        all2AllDataDesJson["recv_count"] = param.all2AllDataDes.recvCount;
        all2AllDataDesJson["count"] = param.all2AllDataDes.count;
        all2AllDataDesJson["send_count_matrix_size"] = param.all2AllDataDes.sendCountMatrix.size();
        opParamJson["all2all_data_des"] = std::move(all2AllDataDesJson);
    }
    return opParamJson;
}

// --- 业务函数修正 ---
void RunChecker(const std::string& data_id)
{
    std::lock_guard<std::mutex> runLock(g_run_checker_mutex);
    HcclSim::StorageManager& storage = HcclSim::StorageManager::GetInstance();
    storage.Reset();
    storage.SetDataId(data_id);
    const HcclResult settingRefreshRet = HcclSim::SettingManager::GetInstance().Refresh();
    if (settingRefreshRet != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_WARN("Failed to refresh manifest settings, the previous checker settings will be kept");
    }
    HcclSim::DumpManager& dumpManager = HcclSim::DumpManager::GetInstance();
    dumpManager.Reset();
    HcclResult dumpInitRet = dumpManager.Initialize(data_id);
    if (dumpInitRet != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "{} Failed to initialize the dump manager, checker output files "
            "cannot be written, dataId={}",
            HcclSim::MakeErrorCodeText(HcclSim::ErrorCode::DUMP_FAILED), data_id);
        return;
    }
    HcclSim::DumpV3Manager& dumpV3Manager = HcclSim::DumpV3Manager::GetInstance();
    dumpV3Manager.Reset();
    dumpInitRet = dumpV3Manager.Initialize(data_id);
    if (dumpInitRet != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "{} Failed to initialize the V3 dump manager, checker output files cannot be written, "
            "dataId={}",
            HcclSim::MakeErrorCodeText(HcclSim::ErrorCode::DUMP_FAILED), data_id);
        return;
    }
    HcclSim::DumpRunManifest::GetInstance().Reset(data_id);
    HcclSim::TaskGraphGeneratorV3::AllRankParamRecorder::Global()->Reset();
    storage.InitCcuInfo(
        HcclSim::TaskGraphGeneratorV3::AllRankParamRecorder::Global()->devType_,
        HcclSim::TaskGraphGeneratorV3::AllRankParamRecorder::Global()->ccu_resource_base_addr_);

    HcclResult loadRet = g_loader.LoadOpTaskFile();
    if (loadRet != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR("Failed to load the op task file");
        return;
    }

    std::vector<sim::operation::CcuChannelTab> channels;
    std::vector<sim::operation::HalfRTTTab> halfRTT;
    std::vector<sim::operation::CcuInstrResTab> instrRes;
    HcclResult ret = LoadCheckerGlobalResources(g_loader, storage, channels, halfRTT, instrRes);
    if (ret != HcclResult::HCCL_SUCCESS) {
        return;
    }

    const HcclSim::CheckerSettings checkerSettings = HcclSim::SettingManager::GetInstance().GetSettings();
    const bool enableBigGraphChecker = checkerSettings.enableBigGraphChecker;
    const bool enableSingleOpChecker = checkerSettings.enableNewChecker;
    std::vector<CheckerStatus> multiOpCheckerResults;
    std::vector<SingleOpCheckerResult> opCheckerResults;
    HcclSim::BigGraphCheckV3::BigGraphCheckerV3 bigGraphChecker;

    if (enableBigGraphChecker) {
        const HcclResult bigGraphRet = ProcessBigGraph(g_loader, bigGraphChecker);
        multiOpCheckerResults.push_back(
            bigGraphRet == HcclResult::HCCL_SUCCESS ? CheckerStatus::SUCCESS : CheckerStatus::FAILED);
        if (bigGraphRet != HcclResult::HCCL_SUCCESS) {
            HCCL_VM_ERROR("BigGraphCheckerV3 failed, ret={}", static_cast<uint32_t>(bigGraphRet));
        }
    }

    if (enableSingleOpChecker) {
        std::vector<sim::operation::OpExecutionKey> opExecutionKeys;
        ret = g_loader.LoadOpExecutionKeys(opExecutionKeys);
        if (ret != HcclResult::HCCL_SUCCESS) {
            HCCL_VM_ERROR(
                "{} Failed to load operator execution keys",
                HcclSim::MakeErrorCodeText(HcclSim::ErrorCode::CHECKER_RUNTIME_ERROR));
            return;
        }

        uint32_t opIdx = 0;
        for (const sim::operation::OpExecutionKey& key : opExecutionKeys) {
            sim::operation::OpExecution opExecution;
            ret = g_loader.LoadOpExecutionByKey(key, opExecution);
            if (ret != HcclResult::HCCL_SUCCESS) {
                HCCL_VM_ERROR("Failed to load one op, commName={}, opIter={}", key.commName, key.opIter);
                return;
            }
            const uint32_t currentOpIdx = opIdx++;
            CheckerStatus opCheckerStatus = CheckerStatus::DISABLE;
            ret = CheckOneOp(storage, channels, halfRTT, currentOpIdx, opExecution, opCheckerStatus);
            opCheckerResults.push_back({key, opCheckerStatus});
            if (dumpManager.IsEnabled()) {
                HcclSim::DumpRunManifest::GetInstance().SetCheckResult(ret);
                const HcclResult flushRet = HcclSim::ValidationIssueRecorder::GetInstance().Flush();
                if (flushRet != HcclResult::HCCL_SUCCESS) {
                    HCCL_VM_WARN(
                        "{} Failed to flush the validation issue dump, dataId={}, opIndex={}, "
                        "dumpType=validation_issues",
                        HcclSim::MakeErrorCodeText(HcclSim::ErrorCode::DUMP_FAILED), data_id, currentOpIdx);
                }
                const HcclResult manifestRet = HcclSim::DumpRunManifest::GetInstance().Flush();
                if (manifestRet != HcclResult::HCCL_SUCCESS) {
                    HCCL_VM_WARN(
                        "{} Failed to flush the dump manifest, dataId={}, opIndex={}",
                        HcclSim::MakeErrorCodeText(HcclSim::ErrorCode::DUMP_FAILED), data_id, currentOpIdx);
                }
            }
            if (ret != HcclResult::HCCL_SUCCESS || opCheckerStatus == CheckerStatus::FAILED) {
                HCCL_VM_ERROR("Checker result: commName={}, opIter={}, result=failed", key.commName, key.opIter);
                continue;
            }
            if (opCheckerStatus == CheckerStatus::NOT_EXECUTED) {
                HCCL_VM_INFO("Checker result: commName={}, opIter={}, result=not_executed", key.commName, key.opIter);
                continue;
            }
            HCCL_VM_INFO("Checker result: commName={}, opIter={}, result=success", key.commName, key.opIter);
        }
    } else {
        HCCL_VM_INFO("Single-op checker is disabled");
    }
    if (!opCheckerResults.empty()) {
        constexpr int COMM_NAME_COLUMN_WIDTH = 32;
        constexpr int COMM_HASH_COLUMN_WIDTH = 20;
        constexpr int OP_ITER_COLUMN_WIDTH = 8;
        constexpr int CHECKER_COLUMN_WIDTH = 13;
        HCCL_VM_INFO("Checker execution result (success/failed/disable/not_executed):");
        HCCL_VM_INFO("Single-op checker result:");
        std::ostringstream header;
        header << "| " << std::left << std::setw(COMM_NAME_COLUMN_WIDTH) << "commName"
               << " | " << std::setw(COMM_HASH_COLUMN_WIDTH) << "commHash"
               << " | " << std::setw(OP_ITER_COLUMN_WIDTH) << "opIter"
               << " | " << std::setw(CHECKER_COLUMN_WIDTH) << "checker" << " |";
        HCCL_VM_INFO("{}", header.str());
        for (const SingleOpCheckerResult& result : opCheckerResults) {
            std::ostringstream row;
            row << "| " << std::left << std::setw(COMM_NAME_COLUMN_WIDTH) << result.key.commName << " | "
                << std::setw(COMM_HASH_COLUMN_WIDTH) << result.key.commHash << " | " << std::setw(OP_ITER_COLUMN_WIDTH)
                << result.key.opIter << " | " << std::setw(CHECKER_COLUMN_WIDTH)
                << CHECKER_STATUS_TEXT[static_cast<size_t>(result.status)] << " |";
            HCCL_VM_INFO("{}", row.str());
        }
    } else {
        HCCL_VM_WARN("Checker execution result is unavailable because no single-op checker was executed");
    }
    if (!multiOpCheckerResults.empty()) {
        constexpr int MULTI_OP_COLUMN_WIDTH = 17;
        HCCL_VM_INFO("Big-graph checker result:");
        std::ostringstream header;
        header << "| " << std::left << std::setw(MULTI_OP_COLUMN_WIDTH) << "big graph checker" << " |";
        HCCL_VM_INFO("{}", header.str());
        for (const CheckerStatus status : multiOpCheckerResults) {
            std::ostringstream row;
            row << "| " << std::left << std::setw(MULTI_OP_COLUMN_WIDTH)
                << CHECKER_STATUS_TEXT[static_cast<size_t>(status)] << " |";
            HCCL_VM_INFO("{}", row.str());
        }
    }
    bool hasCheckerFailure = false;
    bool hasCheckerExecution = false;
    bool hasCheckerNotExecuted = false;
    for (const SingleOpCheckerResult& result : opCheckerResults) {
        hasCheckerFailure = hasCheckerFailure || result.status == CheckerStatus::FAILED;
        hasCheckerExecution
            = hasCheckerExecution || result.status == CheckerStatus::SUCCESS || result.status == CheckerStatus::FAILED;
        hasCheckerNotExecuted = hasCheckerNotExecuted || result.status == CheckerStatus::NOT_EXECUTED;
    }
    for (const CheckerStatus status : multiOpCheckerResults) {
        hasCheckerFailure = hasCheckerFailure || status == CheckerStatus::FAILED;
        hasCheckerExecution
            = hasCheckerExecution || status == CheckerStatus::SUCCESS || status == CheckerStatus::FAILED;
        hasCheckerNotExecuted = hasCheckerNotExecuted || status == CheckerStatus::NOT_EXECUTED;
    }
    if (hasCheckerExecution && !hasCheckerFailure && !hasCheckerNotExecuted) {
        HCCL_VM_INFO(
            "[CHECKER_RUN_SUMMARY] All Success (Total Op: {}, Big Graph: {})", opCheckerResults.size(),
            multiOpCheckerResults.size());
    } else {
        HCCL_VM_INFO(
            "[CHECKER_RUN_SUMMARY] Failed (Total Op: {}, Big Graph: {})", opCheckerResults.size(),
            multiOpCheckerResults.size());
    }
    std::cout << "(hvm)$> " << std::flush;
    FlushLog(); // 将本轮完整日志落盘
}

void StartCheckerWorker(const std::string& dataId)
{
    std::lock_guard<std::mutex> workerLock(g_worker_mutex);
    if (g_worker_thread.joinable()) {
        g_worker_thread.join();
    }
    g_worker_thread = std::thread([dataId]() {
        RunChecker(dataId);
    });
}

void JoinCheckerWorker()
{
    std::lock_guard<std::mutex> workerLock(g_worker_mutex);
    if (g_worker_thread.joinable()) {
        g_worker_thread.join();
    }
}

// --- 分发函数修正 ---
// 不再使用 exit(0)，而是通过标记位通知主线程
void ProcessCommand(const std::string& line)
{
    try {
        auto j = json::parse(line);
        std::string action = j.value("action", "");
        auto payload = j.value("payload", json::object());

        if (action == "status") {
            std::string status = payload.value("status", "");
            HCCL_VM_INFO("Received checker status signal, status={}", status);
            if (status != "finish") {
                return;
            }
            // 运行前刷新设置，确保最新的配置生效
            const HcclResult settingRefreshRet = HcclSim::SettingManager::GetInstance().Refresh();
            if (settingRefreshRet != HcclResult::HCCL_SUCCESS) {
                HCCL_VM_WARN("Failed to refresh manifest settings, use the previous settings");
            }

            std::string data_id = payload.value("data_id", "");
            StartCheckerWorker(data_id);
        } else if (action == "stop") {
            HCCL_VM_INFO("Received checker stop signal, shutdown...");
            g_keep_running.store(false); // 仅仅修改标志位
        }
        // 其他 action...
    } catch (const std::exception& e) {
        HCCL_VM_ERROR("Command processing failed: {}", e.what());
    }
}

int main()
{
    LogConfig config = LoadLogConfig("checker");
    InitLogger(config);

    HCCL_VM_INFO("Plugin process active. Listening for commands...");

    // 主循环检查标志位
    std::string line;
    // 注意：std::getline 是阻塞的。如果 stop 指令后没有后续输入，
    // 循环会卡在 getline。但在插件管理场景下，发送完 stop 后通常会关闭管道，
    // 导致 getline 返回 false。
    while (g_keep_running.load() && std::getline(std::cin, line)) {
        if (line.empty()) {
            continue;
        }
        ProcessCommand(line);
    }

    JoinCheckerWorker();
    HCCL_VM_INFO("shutdown.");
    return 0; // 整个进程唯一的正常出口
}
