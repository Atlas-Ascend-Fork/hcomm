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
#include <map>
#include <set>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <utility>
#include <unistd.h>
#include <nlohmann_json/json.hpp>
#include "dump/dump_manager.h"
#include "dump/dump_run_manifest.h"
#include "dump/validation_issue_recorder.h"
#include "dump_v3/dump_v3_manager.h"
#include "dfx/dag_graphviz_dump.h"
#include "dfx/ccu_resource_stats_dfx.h"
#include "dfx/ccu_resource_stats_reporter.h"
#include "dfx/ccu_resource_stats_topo.h"
#include "setting_manager.h"
#include "storage_manager.h"
#include "checker.h"
#include "file_utils.h"
#include "framework/composite_op_grouping.h"
#include "framework/big_graph_check/big_graph_checker.h"
#include "sim_log.h"
#include "stage_profiler.h"
#include "framework/task_graph_generator_v3/ccu_graph_generator_v3/ccu_all_rank_param_recorder_v3.h"
#include "sim_loader.h"
#include "sim_common_defs.h"
#include "store/store_sim_run_mode.h"
#include "utils/check_utils.h"
#include "utils/error_codes.h"

using json = nlohmann::json;
loader::Loader g_loader;

// 全局 channel 表（定义于 storage_manager.cc），供 CCU 资源统计的差分参考口径快照使用
extern std::map<DeviceId, std::map<uint32_t, HcclSim::ChannelsPerDie>> g_allRankChannelInfo;

// 使用原子变量控制程序生命周期
std::atomic<bool> g_keep_running{true};
std::mutex g_run_checker_mutex;
std::mutex g_worker_mutex;
std::thread g_worker_thread;

// 交互式 shell 是否已回到前台、停在提示符处等输入：前台进程组就是该 shell 自己时成立；
// 前台还在跑命令时，前台进程组是那条命令的进程组。与 cmd_base_utils.cc 的 ShellAtPrompt()
// 同一思路，区别是 checker 拿不到子 bash 的 pid（@checker 插件先于 bash 启动），
// 改为按 /proc/<pgid>/comm 识别前台进程组是不是交互式 shell。
static bool ShellAtPrompt()
{
    if (::isatty(STDOUT_FILENO) == 0) {
        return false; // 标准输出被重定向（测试框架抓取日志）时不需要补提示符
    }
    const pid_t foregroundPgrp = ::tcgetpgrp(STDOUT_FILENO);
    if (foregroundPgrp <= 0 || foregroundPgrp == ::getpgrp()) {
        return false;
    }
    std::ifstream commFile("/proc/" + std::to_string(foregroundPgrp) + "/comm");
    std::string comm;
    if (!std::getline(commFile, comm)) {
        return false;
    }
    // hccl-vm 的交互 shell 固定由 /bin/bash -i 拉起；用户再嵌套的交互 shell 同样识别
    return comm == "bash" || comm == "sh" || comm == "dash" || comm == "zsh";
}

// 异步打屏（校验结果/完成标记）会盖掉 shell 已打出的提示符：确认 shell 已在前台等输入后补打一次，
// 用户不必再敲回车才看到 "(hvm)$>"；shell 还在跑前台命令时不补，避免它随后自己再打一遍出现两个提示符。
// 100ms 让"刚交还前台但提示符还没打出来"的窗口先落定，再用 \r 覆盖，保证任何时序下只剩一个提示符。
static void RestoreShellPromptIfIdle()
{
    if (!ShellAtPrompt()) {
        return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::cout << '\r' << sim::GetRunModePrompt() << std::flush;
}

// checker 完成标记：checker 跑在 worker 线程，宿主侧 plugin run @checker 早已返回，
// 上层测试框架只靠这一行感知本轮 checker 已结束，因此任何退出路径（正常完成、提前失败返回、
// 异常）都必须打印。标记只表示本轮结束，不承载校验结果，结果以落盘日志为准。
static void EmitRunFinishedMarker()
{
    FlushLog(); // 先落盘日志，保证看到标记即可读结果
    std::cout << "[HCCL_VM_PLUGIN_RUN_FINISHED] plugin=@checker\n" << std::flush;
    // 标记之后补回提示符：所有退出路径共用同一收尾，失败提前返回也不会把用户晾在没有提示符的终端上
    RestoreShellPromptIfIdle();
}

// RAII 兜底：作用域内未显式 Emit（提前 return / 异常展开）时由析构补打，且每轮只打一次
class RunFinishedMarker {
public:
    RunFinishedMarker() = default;
    ~RunFinishedMarker() { Emit(); }

    RunFinishedMarker(const RunFinishedMarker&) = delete;
    RunFinishedMarker& operator=(const RunFinishedMarker&) = delete;

    void Emit()
    {
        if (emitted_) {
            return;
        }
        emitted_ = true;
        EmitRunFinishedMarker();
    }

private:
    bool emitted_{false};
};

enum class CheckerStatus : uint8_t { SUCCESS, FAILED, DISABLE, NOT_EXECUTED };
static constexpr const char* CHECKER_STATUS_TEXT[] = {"success", "failed", "disable", "not_executed"};

struct SingleOpCheckerResult {
    sim::operation::OpExecutionKey key;
    CheckerStatus status;
};

json BuildOpParamSummaryJson(const HcclSim::CheckerParam& param);

// CCU 资源统计（dfx）：run 末尾生成 per-op 粒度 CSV 报告（仅统计开关打开时）。
// 拓扑标签解析见 dfx/ccu_resource_stats_topo.cc（失败兜底 -1，不阻塞报告）。
static void GenerateCcuResourceStatsReportIfNeeded()
{
    if (!HcclSim::SettingManager::GetInstance().IsCcuResourceStatsDumpEnabled()) {
        return;
    }
    std::string statsRootDir = HcclSim::DumpV3Manager::GetInstance().GetRootDir();
    if (statsRootDir.empty()) {
        statsRootDir = HcclSim::JoinPath(HcclSim::GetCurrentPath(), "data");
    }
    // A6（DEV_TYPE_960）无 GSA 寄存器，报告 gsaNum 列输出 NA；A5 及其余设备输出数值
    const auto ccuDevType = HcclSim::TaskGraphGeneratorV3::AllRankParamRecorder::Global()->GetDevType();
    const bool gsaApplicable = (ccuDevType != decltype(ccuDevType)::DEV_TYPE_960);
    (void)HcclSim::CcuResourceStats::GenerateCcuResourceStatsReport(
        HcclSim::JoinPath(statsRootDir, "ccu_resource_stats.csv"), gsaApplicable,
        HcclSim::CcuResourceStats::BuildCcuStatsTopoLabels());
}

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

// CCU 资源统计（dfx）：构建当前算子的展示信息（CSV 节头附带）。
// 数据量按 rank0 视角计算（send/recv 等 rank 敏感算子取 myRank=0）；
// reduceType 仅 reduce 类算子输出，其余 NA；算法名取各 rank detail 中首个非空 algorithmName。
static HcclSim::CcuResourceStats::OpInfo
BuildCcuStatsOpInfo(const HcclSim::CheckerParam& param, const sim::operation::OpExecution& opExecution)
{
    HcclSim::CcuResourceStats::OpInfo info;
    info.collectiveType = HcclSim::HcclCmdTypeToString(param.cmdType);
    info.dataType = HcclSim::HcclDataTypeToString(param.dataType);
    info.elementCount = param.dataCount;
    info.rankSize = param.rankSize;
    if (param.cmdType == HcclCMDType::HCCL_CMD_ALLREDUCE || param.cmdType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER
        || param.cmdType == HcclCMDType::HCCL_CMD_REDUCE || param.cmdType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V) {
        info.reduceType = HcclReduceOpToString(param.reduceType);
    }
    uint64_t inputBytes = 0;
    uint64_t outputBytes = 0;
    HcclSim::CalcInputOutputSize(
        param.cmdType, param.rankSize, param.dataCount, param.dataType, inputBytes, outputBytes, 0, param.srcRank,
        param.dstRank, param.vDataDes, param.all2AllDataDes);
    info.inputBytes = inputBytes;
    info.outputBytes = outputBytes;
    for (const sim::operation::DeviceOpExecutionRecord& record : opExecution.deviceRecords) {
        const std::string algName(
            record.detail.algorithmName, ::strnlen(record.detail.algorithmName, sizeof(record.detail.algorithmName)));
        if (!algName.empty()) {
            info.algName = algName;
            break;
        }
    }
    return info;
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

enum class ResultLevel { INFO, WARN, ERROR };

struct ResultLine {
    std::string text;
    ResultLevel level;
};

// 结果块先攒后发：先统一写日志，再统一写屏（stub 模式下 logger 也上屏，逐行会交错成双份）
static std::vector<ResultLine> g_checkerResultLines;

static void PushResultLine(const std::string& text, ResultLevel level = ResultLevel::INFO)
{
    g_checkerResultLines.push_back(ResultLine{text, level});
}

static void EmitResultLines()
{
    if (g_checkerResultLines.empty()) {
        return;
    }
    for (const ResultLine& line : g_checkerResultLines) {
        switch (line.level) {
            case ResultLevel::WARN:
                HCCL_VM_WARN("{}", line.text);
                break;
            case ResultLevel::ERROR:
                HCCL_VM_ERROR("{}", line.text);
                break;
            default:
                HCCL_VM_INFO("{}", line.text);
                break;
        }
    }
    std::string block;
    for (const ResultLine& line : g_checkerResultLines) {
        block += '\r';
        block += HCCL_VM_TAG;
        block += line.text;
        block += '\n';
    }
    (void)fwrite(block.data(), 1, block.size(), stdout);
    fflush(stdout);
    g_checkerResultLines.clear();
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
    // CCU 资源统计：RAII guard 保证任意退出路径（含下方多处失败提前 return）都正确封桶，
    // 避免算子上下文悬挂导致下一个 op 的数据串桶。须先于 LoadHcclVmSynthesisData。
    auto ccuStatsGuard = HcclSim::CcuResourceStats::Collector::GetInstance().BeginOp(
        {opExecution.key.commName, opExecution.key.commHash, opExecution.key.opIter});
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
    // CCU 资源统计（dfx）：为当前算子桶附带展示信息（开关关闭/无上下文时内部短路）
    HcclSim::CcuResourceStats::Collector::GetInstance().SetCurrentOpInfo(
        BuildCcuStatsOpInfo(storage.GetCheckerParam(), opExecution));
    // V3 检查总耗时：内部各阶段（翻译成图/单任务/同步/内存冲突/语义）见
    // checker.cc 的 "CheckerV3 stage finished" 日志。
    HcclSim::StageProfiler genAndCheckStage("OpGroup.genAndCheckGraphV3");
    HcclResult newCheckerRet = HcclSim::GenAndCheckGraphV3();
    genAndCheckStage.End();
    PushResultLine("----------[CheckerV3 Finished]----------");
    PushResultLine(fmt::format(
        "CheckerV3 finished for this op, commName={}, opIter={}, opIndex={}", opExecution.key.commName,
        opExecution.key.opIter, opIdx));
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
// 单轮 checker 执行主体：正常完成时显式打完成标记，其余退出路径由 RunChecker 的 RAII 兜底
static void RunCheckerImpl(const std::string& data_id, RunFinishedMarker& runFinishedMarker)
{
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
    // CCU 资源统计：run 级复位 + 开关刷新（采集只发生在单算子路径，见 CheckOneOp 的 BeginOp）
    HcclSim::CcuResourceStats::Collector& ccuStatsCollector = HcclSim::CcuResourceStats::Collector::GetInstance();
    ccuStatsCollector.Reset();
    ccuStatsCollector.SetEnabled(HcclSim::SettingManager::GetInstance().IsCcuResourceStatsDumpEnabled());
    // channel 差分参考口径（newChannelIds）：注入全局 channel 表快照提供者，
    // BeginOp 时快照、EndOp 时差分；主口径（引用收集）在 HasChannelInfo/GetChannelInfo 封装内完成
    ccuStatsCollector.SetChannelTableProvider([]() {
        HcclSim::CcuResourceStats::ChannelTableSnapshot snapshot;
        for (const auto& devEntry : g_allRankChannelInfo) {
            for (const auto& dieEntry : devEntry.second) {
                std::set<uint32_t> channelIds;
                for (const auto& chEntry : dieEntry.second) {
                    channelIds.insert(chEntry.first);
                }
                snapshot[devEntry.first][dieEntry.first] = std::move(channelIds);
            }
        }
        return snapshot;
    });
    if (HcclSim::SettingManager::GetInstance().IsCcuResourceStatsDumpEnabled()
        && !HcclSim::SettingManager::GetInstance().IsNewCheckerEnabled()) {
        HCCL_VM_WARN("CCU resource statistics is enabled but the single-op checker is disabled; "
                     "no statistics will be collected because the big-graph path is excluded by design.");
    }
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
            EmitResultLines();
            return;
        }

        uint32_t opIdx = 0;
        for (const sim::operation::OpExecutionKey& key : opExecutionKeys) {
            sim::operation::OpExecution opExecution;
            ret = g_loader.LoadOpExecutionByKey(key, opExecution);
            if (ret != HcclResult::HCCL_SUCCESS) {
                HCCL_VM_ERROR("Failed to load one op, commName={}, opIter={}", key.commName, key.opIter);
                EmitResultLines();
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
                PushResultLine(
                    fmt::format("Checker result: commName={}, opIter={}, result=failed", key.commName, key.opIter),
                    ResultLevel::ERROR);
                continue;
            }
            if (opCheckerStatus == CheckerStatus::NOT_EXECUTED) {
                PushResultLine(fmt::format(
                    "Checker result: commName={}, opIter={}, result=not_executed", key.commName, key.opIter));
                continue;
            }
            PushResultLine(
                fmt::format("Checker result: commName={}, opIter={}, result=success", key.commName, key.opIter));
        }
    } else {
        PushResultLine("Single-op checker is disabled");
    }
    if (!opCheckerResults.empty()) {
        constexpr int COMM_NAME_COLUMN_WIDTH = 32;
        constexpr int COMM_HASH_COLUMN_WIDTH = 20;
        constexpr int OP_ITER_COLUMN_WIDTH = 8;
        constexpr int CHECKER_COLUMN_WIDTH = 13;
        PushResultLine("Checker execution result (success/failed/disable/not_executed):");
        PushResultLine("Single-op checker result:");
        std::ostringstream header;
        header << "| " << std::left << std::setw(COMM_NAME_COLUMN_WIDTH) << "commName"
               << " | " << std::setw(COMM_HASH_COLUMN_WIDTH) << "commHash"
               << " | " << std::setw(OP_ITER_COLUMN_WIDTH) << "opIter"
               << " | " << std::setw(CHECKER_COLUMN_WIDTH) << "checker" << " |";
        PushResultLine(header.str());
        for (const SingleOpCheckerResult& result : opCheckerResults) {
            std::ostringstream row;
            row << "| " << std::left << std::setw(COMM_NAME_COLUMN_WIDTH) << result.key.commName << " | "
                << std::setw(COMM_HASH_COLUMN_WIDTH) << result.key.commHash << " | " << std::setw(OP_ITER_COLUMN_WIDTH)
                << result.key.opIter << " | " << std::setw(CHECKER_COLUMN_WIDTH)
                << CHECKER_STATUS_TEXT[static_cast<size_t>(result.status)] << " |";
            PushResultLine(row.str());
        }
    } else {
        PushResultLine(
            "Checker execution result is unavailable because no single-op checker was executed", ResultLevel::WARN);
    }
    if (!multiOpCheckerResults.empty()) {
        constexpr int MULTI_OP_COLUMN_WIDTH = 17;
        PushResultLine("Big-graph checker result:");
        std::ostringstream header;
        header << "| " << std::left << std::setw(MULTI_OP_COLUMN_WIDTH) << "big graph checker" << " |";
        PushResultLine(header.str());
        for (const CheckerStatus status : multiOpCheckerResults) {
            std::ostringstream row;
            row << "| " << std::left << std::setw(MULTI_OP_COLUMN_WIDTH)
                << CHECKER_STATUS_TEXT[static_cast<size_t>(status)] << " |";
            PushResultLine(row.str());
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
        PushResultLine(fmt::format(
            "[CHECKER_RUN_SUMMARY] All Success (Total Op: {}, Big Graph: {})", opCheckerResults.size(),
            multiOpCheckerResults.size()));
    } else {
        PushResultLine(fmt::format(
            "[CHECKER_RUN_SUMMARY] Failed (Total Op: {}, Big Graph: {})", opCheckerResults.size(),
            multiOpCheckerResults.size()));
    }
    EmitResultLines();
    GenerateCcuResourceStatsReportIfNeeded();
    // 本轮正常结束：打完成标记；提示符由标记的收尾逻辑按"shell 是否已在前台等输入"补回
    runFinishedMarker.Emit();
}

// 单轮 checker 的唯一入口：标记对象先于业务逻辑构造，
// 正常完成、提前失败返回、异常展开都会在退出前打一行 [HCCL_VM_PLUGIN_RUN_FINISHED]，
// 让上层测试框架总能感知本轮结束，不会因缺少标记而超时。
void RunChecker(const std::string& data_id)
{
    std::lock_guard<std::mutex> runLock(g_run_checker_mutex);
    RunFinishedMarker runFinishedMarker;
    try {
        RunCheckerImpl(data_id, runFinishedMarker);
    } catch (const std::exception& e) {
        HCCL_VM_ERROR("Checker run aborted by exception: {}", e.what());
    } catch (...) {
        HCCL_VM_ERROR("Checker run aborted by unknown exception");
    }
}

void StartCheckerWorker(const std::string& dataId)
{
    std::lock_guard<std::mutex> workerLock(g_worker_mutex);
    if (g_worker_thread.joinable()) {
        g_worker_thread.join();
    }
    try {
        g_worker_thread = std::thread([dataId]() {
            RunChecker(dataId);
        });
    } catch (const std::exception& e) {
        // 线程未起来 = 本轮 checker 无法执行：同样打完成标记，避免上层一直等待
        HCCL_VM_ERROR("Failed to start the checker worker thread: {}", e.what());
        EmitRunFinishedMarker();
    }
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
