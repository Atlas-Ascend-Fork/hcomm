/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_DFX_PROFILING_HANDLER_PUB_H
#define HCOMM_DFX_PROFILING_HANDLER_PUB_H
#include <unordered_map>
#include <queue>
#include <mutex>
#include <atomic>
#include <memory>
#include <map>
#include <vector>
#include <string>
#include "hccl/hccl_types.h"
#include "task_info.h"
#include "rt_external.h"
#include "profiling_common.h"
#include "stream_manager.h"
#include "task_param.h"

namespace Hccl {
// ccu 上报数据结构
constexpr unsigned int MSPROF_REPORT_CCU_TASK_INFO = 14U;
constexpr unsigned int MSPROF_REPORT_CCU_WAIT_SIGNAL_INFO = 15U;
constexpr unsigned int MSPROF_REPORT_CCU_GROUP_INFO = 16U;

struct MsprofCcuTaskInfo {
    uint8_t version;
    uint8_t workFlowMode;
    uint64_t itemId;    // CCU任务名 hash id
    uint64_t groupName; // 通信域 hash id
    uint32_t rankId;
    uint32_t ranksize; // CCU任务设计的Chip数目

    uint16_t streamId;
    uint32_t taskId;
    uint8_t dieId;     // CCU任务执行的DieId
    uint8_t missionId; // CCU任务执行的MissionId
    uint16_t instrId;
};

// CCU 上报二进制结构体：与 profiling SDK 解析端约定的内存布局，公共字段须保持平铺，不可提取基类
// （平铺即接口约定），故与 MsprofCcuWaitSignalInfo 字段重复为有意为之。
struct MsprofCcuGroupInfo {
    uint8_t version;
    uint64_t itemId;    // CCU任务名 hash id
    uint64_t groupName; // 通信域 hash id
    uint32_t rankId;
    uint32_t ranksize; // CCU任务设计的Chip数目
    uint8_t workFlowMode;

    uint16_t streamId;
    uint32_t taskId;
    uint8_t dieId; // CCU任务执行的DieId
    uint16_t instrId;
    uint8_t missionId; // CCU任务执行的MissionId

    uint8_t reduceOpType;   // 与HcclReduceOp类型保持一致
    uint8_t inputDataType;  // 与HcclDataType类型保持一致
    uint8_t outputDataType; // 与HcclDataType类型保持一致
    uint64_t dataSize;      // 输入数据大小

    uint16_t channelId[CCU_MAX_CHANNEL_NUM];    // LoopGroup所包含的搬运指令使用的ChannelId
    uint32_t remoteRankId[CCU_MAX_CHANNEL_NUM]; // LoopGroup所包含的搬运指令的对端
};

// CCU 上报二进制结构体：与 profiling SDK 解析端约定的内存布局，公共字段须保持平铺，不可提取基类，
// 故与 MsprofCcuGroupInfo 字段重复为有意为之。
struct MsprofCcuWaitSignalInfo {
    uint8_t version;
    uint64_t itemId;    // CCU任务名 hash id
    uint64_t groupName; // 通信域 hash id
    uint32_t rankId;
    uint32_t ranksize; // CCU任务设计的Chip数目
    uint8_t workFlowMode;

    uint16_t streamId;
    uint32_t taskId;
    uint8_t dieId; // CCU任务执行的DieId
    uint16_t instrId;
    uint8_t missionId; // CCU任务执行的MissionId

    uint32_t ckeId;
    uint32_t mask;
    uint16_t channelId[CCU_MAX_CHANNEL_NUM];    // LoopGroup所包含的搬运指令使用的ChannelId
    uint32_t remoteRankId[CCU_MAX_CHANNEL_NUM]; // LoopGroup所包含的搬运指令的对端
};

// 即 base_comm 内部类定义，供 coll_comm 直接调用接口（同主线）；禁止在 L2 派生或取 sizeof。
class DfxProfilingHandler {
public:
    ~DfxProfilingHandler();

    DfxProfilingHandler(const DfxProfilingHandler& that) = delete;

    DfxProfilingHandler& operator=(const DfxProfilingHandler& that) = delete;

    static DfxProfilingHandler& GetInstance();

    static int32_t CommandHandleWrapper(uint32_t rtType, void* data, uint32_t len);

    void ReportKernel() const;

    HcclResult ReportHostApi(OpType opType, uint64_t beginTime, uint64_t endTime, bool cachedReq, bool isAiCpu);

    HcclResult ReportHcclOp(const DfxOpInfo& opInfo, bool cachedReq);

    HcclResult ReportHcclTaskApi(
        TaskParamType taskType, uint64_t beginTime, uint64_t endTime, bool isMasterStream, bool cachedReq,
        bool ignoreLevel = false, uint32_t threadId = 0);

    HcclResult ReportHcclTaskDetails(const TaskInfo& taskInfo, bool cachedReq);
    HcclResult ReportHcclTaskDetailsBatch(const std::vector<TaskInfo*>& taskInfos, bool cachedReq);

    bool GetHostApiState() const;
    bool GetHcclNodeState() const;
    bool GetHcclL0State() const;
    bool GetHcclL1State() const;
    HcclResult CommandHandle(uint32_t rtType, void* data, uint32_t len) const;
    HcclResult Init();
    HcclResult ReportHcclMC2CommInfo(
        const Stream& kfcStream, const Stream& stream, const std::vector<Stream*>& aicpuStreams, const std::string& id,
        RankId myRank, u32 rankSize, RankId rankInParentComm);
    HcclResult ReportHcclMC2CommInfo(
        const u32 kfcStreamId, const std::vector<u32>& aicpuStreamsId, const std::string& id, RankId myRank,
        u32 rankSize, RankId rankInParentComm);
    HcclResult
    ReportNodeApi(uint64_t beginTime, uint64_t endTime, uint64_t cmdItemId, uint32_t threadId, bool cachedReq);
    HcclResult ReportNodeBasicInfo(uint64_t timeStamp, uint64_t cmdItemId, uint32_t threadId, bool cachedReq);
    uint64_t GetProfHashId(const char* name, uint32_t len) const;
    uint64_t GetCachedAlgTypeHashId() const { return cachedAlgTypeHashId_.load(); }
    inline void SetOpModeFlags(bool isOpBase, bool isCached)
    {
        isOpbase_ = isOpBase;
        isCached_ = isCached;
    }
    inline bool GetCachedFlag() const { return isCached_; }
    inline bool GetOpBaseFlag() const { return isOpbase_; }

    static uint32_t GetCachedTid();
    static uint64_t GetProfSwitchVersion() { return g_profSwitchVersion.load(std::memory_order_relaxed); }
    static void NotifyProfSwitchChanged() { g_profSwitchVersion.fetch_add(1, std::memory_order_relaxed); }

private:
    explicit DfxProfilingHandler();

    HcclResult ReportAclApi(
        uint32_t cmdType, uint64_t beginTime, uint64_t endTime, uint64_t cmdItemId, uint32_t threadId, bool cachedReq);

    HcclResult ReportHcclOpInfo(uint64_t timeStamp, const DfxOpInfo& opInfo, uint32_t threadId, bool cachedReq);
    HcclResult ReportAdditionInfo(MsprofAdditionalInfo& reporterData) const;

    HcclResult StartSubscribe(uint64_t profconfig);
    HcclResult StartTaskApiSubscribe();
    HcclResult StartHostApiSubscribe();
    HcclResult StartAdditionInfoSubscribe();
    HcclResult StartHostHcclOpSubscribe();
    HcclResult StartCcuSubscribe();
    void StopSubscribe();

    HcclResult CallProfRegHostApi() const;
    HcclResult ReportStoragedCompactInfo();
    HcclResult ReportStoragedNodeBasicInfo();
    HcclResult ReportMc2AdditionInfo();

    HcclResult CallProfRegTaskTypeApi() const;
    HcclResult ReportStoragedTaskApi();
    HcclResult ReportStoragedAclApi();

    HcclResult CallProfRegHcclOpApi() const;

    HcclResult ReportStoragedAdditionInfo();

    void FillProfCommonInfo(const TaskInfo& taskInfo, MsprofAdditionalInfo& reporterData) const;
    HcclResult FillProfTaskSpecificInfo(const TaskInfo& taskInfo, MsprofHcclInfo* profInfo) const;
    void FillDpuProfInfo(const TaskInfo& taskInfo, MsprofAdditionalInfo& reporterData) const;
    void FillDpuTaskParaDetails(const TaskInfo& taskInfo, MsprofDpuHcclTrack* dpuProfInfo) const;
    void ConvertHcclInfoToDpuTrack(MsprofAdditionalInfo& reporterData) const;
    HcclResult FillTaskAdditionInfo(const TaskInfo& taskInfo, MsprofAdditionalInfo& reporterData) const;
    uint32_t GetTaskTypeValue(TaskParamType taskType) const;
    bool IsSkipReportTaskType(TaskParamType taskType) const;

    HcclResult ReportCcuInfo(const TaskInfo& taskInfo) const;
    HcclResult GetCcuTaskInfo(const TaskInfo& taskInfo, const CcuProfilingInfo& info) const;
    HcclResult GetCcuWaitSignalInfo(const TaskInfo& taskInfo, const CcuProfilingInfo& info) const;
    HcclResult GetCcuGroupInfo(const TaskInfo& taskInfo, const CcuProfilingInfo& info) const;

    void DumpHCCLReportData(const TaskInfo& taskInfo, const MsprofAdditionalInfo& reporterData) const;
    void DumpCcuGroupInfo(const MsprofCcuGroupInfo& ccuGroupInfo) const;
    HcclResult ReportMc2AdditionInfo(uint64_t timeStamp, const void* data, int len);
    void SetCachedCclTag();
    void InitLog() const;
    void ReportHcclMC2CommInfoLog(
        const Stream& kfcStream, const Stream& stream, const std::vector<Stream*>& aicpuStreams, const std::string& id,
        RankId myRank, u32 rankSize, RankId rankInParentComm) const;
    void ReportHcclMC2CommInfoLog(
        const u32 kfcStreamId, const std::vector<u32>& aicpuStreamsId, const std::string& id, RankId myRank,
        u32 rankSize, RankId rankInParentComm) const;
    void ReportCcuInfoLog(const TaskInfo& taskInfo) const;
    void LogCcuTaskInfo(
        const CcuProfilingInfo& info, const TaskInfo& taskInfo, uint64_t itemId, uint64_t groupName, u32 rankId,
        u32 ranksize) const;
    void LogCcuWaitSignalInfo(
        const CcuProfilingInfo& info, const TaskInfo& taskInfo, uint64_t itemId, uint64_t groupName, u32 rankId,
        u32 ranksize) const;
    void LogCcuGroupInfo(
        const CcuProfilingInfo& info, const TaskInfo& taskInfo, uint64_t itemId, uint64_t groupName, u32 rankId,
        u32 ranksize) const;
    void ReportHcclTaskDetailsBatchLog(const std::vector<TaskInfo*>& taskInfos) const;
    void ReportStoragedAdditionInfoLog() const;

private:
    static DfxProfilingHandler instance_;
    static std::atomic<uint64_t> g_profSwitchVersion;
    std::atomic<bool> initializedFlag_{false};
    std::atomic<bool> enableHostApi_{false};
    std::atomic<bool> enableHcclNode_{false};
    std::atomic<bool> enableHcclL0_{false};
    std::atomic<bool> enableHcclL1_{false};
    bool isOpbase_{false};
    bool isCached_{false};

    std::vector<TaskInfo> cacheTaskInfos_{};
    std::queue<MsprofApi> cachedTaskApiInfo_{};
    std::queue<MsprofApi> cachedAclApiInfo_{};
    std::queue<MsprofCompactInfo> cacheHcclOpInfo_{};
    std::queue<MsprofCompactInfo> cacheNodeBasicInfo_{};
    std::queue<MsprofAdditionalInfo> cacheHcclAdditionInfo_{};
    std::atomic<uint64_t> cachedAlgTypeHashId_{0};
    std::map<uint32_t, uint64_t> cachedNewCclTag_{};
    mutable std::mutex cachedCclTagMutex_;
    mutable std::mutex cacheTaskInfosMutex_;
    std::mutex cachedTaskApiInfoMutex_;
    std::mutex cachedAclApiInfoMutex_;
    std::mutex cacheHcclOpInfoMutex_;
    std::mutex cacheNodeBasicInfoMutex_;
    std::mutex cacheHcclAdditionInfoMutex_;
};
} // namespace Hccl

#endif // HCOMM_DFX_PROFILING_HANDLER_PUB_H
