/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/cast_utils_pub.h"
#include "dfx/device/dfx_res_pub.h"
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "aicpu_task_cache_entry.h"
#include "aicpu_task_cache_manager.h"
#include "aicpu_task_utils.h"

#include "common/device/aicpu_indop_env_pub.h"
#include "log.h"
#include "rma_buffer_lite.h"
#include "buffer.h"
#include "rmt_rma_buf_slice_lite.h"
#include "sqe_v82.h"

using Hccl::GetPlfDebugConfigValue;

namespace {
constexpr uint8_t SDMA_OPCODE_MEMCPY = 0;
constexpr uint32_t TASK_CACHE_LOCAL_MEM_COUNT = 2;
} // namespace
using Hccl::PLF_TASK;
#ifdef HCCL_V2 // hccl_v2
using Hccl::HCCL_LOG_DEBUG;
using Hccl::HCCL_LOG_INFO;
using Hccl::HcclCheckLogLevel;
#endif

using Hccl::AC_SQE_SIZE;
using hccl::AicpuTsThread;
using Hccl::DbSqeProfInfo;
using Hccl::Rt91095StarsMemcpySqe;
using Hccl::Rt91095StarsSqeHeader;
using Hccl::Rt91095StarsSqeType;
using Hccl::Rt91095StarsUbdmaDBmodeSqe;
using Hccl::Rt91095StarsWriteValueSqe;
using Hccl::RtsqA5;
using Hccl::StreamLite;
using Hccl::TaskParamTypeVal;
using Hccl::UbConnLite;
using Hccl::UbTransportLiteImpl;
using Hccl::UdmaSqeCommon;
using Hccl::UdmaSqOpcode;
using Hccl::WqeTask;
using std::vector;

namespace hcomm {

AddrRefreshInfo::AddrRefreshInfo() : needRefresh(false), memIdx(0), offset(0) {}

AddrRefreshInfo::AddrRefreshInfo(const uint32_t curMemIdx) : needRefresh(true), memIdx(curMemIdx), offset(0) {}

AddrRefreshInfo::AddrRefreshInfo(const AddrRefreshInfo& other)
    : needRefresh(other.needRefresh),
      memIdx(other.memIdx),
      rankId(other.rankId),
      offset(other.offset)
{}

AddrRefreshInfo::~AddrRefreshInfo() {}

const AddrRefreshInfo& AddrRefreshInfo::operator=(const AddrRefreshInfo& other)
{
    if (this != &other) {
        needRefresh = other.needRefresh;
        offset = other.offset;
        memIdx = other.memIdx;
        rankId = other.rankId;
    }
    return *this;
}

AicpuTaskCacheEntry::AicpuTaskCacheEntry()
{
    if ((UNLIKELY((GetPlfDebugConfigValue() & PLF_TASK) != 0)) || UNLIKELY(HcclCheckLogLevel(HCCL_LOG_INFO))) {
        isTaskConfigDebug_ = true;
    }
}

AicpuTaskCacheEntry::~AicpuTaskCacheEntry()
{
    size_t sqeArrayCount = sqeArrayInfos_.size();
    size_t totalSqeCount = 0;
    for (size_t arrayIdx = 0; arrayIdx < sqeArrayCount; ++arrayIdx) {
        totalSqeCount += sqeArrayInfos_[arrayIdx].srcAddrRefreshInfoArray.size();

        uint8_t* curSqeArray = sqeArrayInfos_[arrayIdx].sqeArray;
        if (UNLIKELY(curSqeArray == nullptr)) {
            HCCL_ERROR("[AicpuTaskCacheEntry][~AicpuTaskCacheEntry] curSqeArray is nullptr");
        } else {
            free(curSqeArray);
            sqeArrayInfos_[arrayIdx].sqeArray = nullptr;
        }
    }

    size_t wqeArrayCount = wqeTaskArrayInfos_.size();
    size_t totalWqeCount = 0;
    for (size_t arrayIdx = 0; arrayIdx < wqeArrayCount; ++arrayIdx) {
        totalWqeCount += wqeTaskArrayInfos_[arrayIdx].wqeTaskArray.size();
    }

    HCCL_INFO(
        "[AicpuTaskCacheEntry][~AicpuTaskCacheEntry] release %u SQE arrays "
        "(%u SQEs in total) and %u WQE arrays (%u WQEs in total) from a cache entry",
        sqeArrayCount, totalSqeCount, wqeArrayCount, totalWqeCount);
}

HcclResult AicpuTaskCacheEntry::InitCacheEntry(
    const uint64_t* baseAddrs, const uint64_t* memSizes, const uint32_t count, const VOpInfo& vOpInfo)
{
    // V 类录制与回放共用 rankId，四个非空数组必须对应同一组 rank。
    const bool hasVOpInfo = !vOpInfo.sendCounts.empty() && !vOpInfo.recvCounts.empty() && !vOpInfo.sdispls.empty()
                            && !vOpInfo.rdispls.empty();
    if (hasVOpInfo) {
        const size_t rankSize = vOpInfo.sendCounts.size();
        CHK_PRT_RET(
            vOpInfo.recvCounts.size() != rankSize || vOpInfo.sdispls.size() != rankSize
                || vOpInfo.rdispls.size() != rankSize,
            HCCL_ERROR("[AicpuTaskCacheEntry][%s] inconsistent V metadata sizes", __func__), HCCL_E_PARA);
    }

    // 前两个地址固定为当前rank的user input/output，后续地址统一作为remote memory range解析。
    CHK_PRT_RET(
        count < TASK_CACHE_LOCAL_MEM_COUNT,
        HCCL_ERROR(
            "[AicpuTaskCacheEntry][%s] count[%u] < TASK_CACHE_LOCAL_MEM_COUNT[%u]", __func__, count,
            TASK_CACHE_LOCAL_MEM_COUNT),
        HCCL_E_PARA);

    // 空指针及溢出检查
    for (uint32_t i = 0; i < count; i++) {
        CHK_PRT_RET(
            baseAddrs[i] == 0,
            HCCL_ERROR("[AicpuTaskCacheEntry][InitCacheEntry] baseAddrs[%u] is 0, memSize[%llu]", i, memSizes[i]),
            HCCL_E_PARA);
        CHK_PRT_RET(
            baseAddrs[i] + memSizes[i] < baseAddrs[i],
            HCCL_ERROR("[AicpuTaskCacheEntry][InitCacheEntry] baseAddrs[%u] + memSizes[%u] overflows", i, i),
            HCCL_E_PARA);
        HCCL_DEBUG(
            "[AicpuTaskCacheEntry][InitCacheEntry] memIdx[%u] addrType[%s] baseAddr[0x%llx] memSize[%llu]", i,
            i < TASK_CACHE_LOCAL_MEM_COUNT ? "local" : "remote", baseAddrs[i], memSizes[i]);
    }
    HCCL_INFO(
        "[AicpuTaskCacheEntry][InitCacheEntry] parsed localMemCount[%u] remoteMemCount[%u] totalMemCount[%u]",
        TASK_CACHE_LOCAL_MEM_COUNT, count - TASK_CACHE_LOCAL_MEM_COUNT, count);

    // 缓存baseAddrs
    CHK_PRT_RET(
        cachedBaseAddrs_.size() != 0,
        HCCL_ERROR("[AicpuTaskCacheEntry][%s] cachedBaseAddrs_.size[%u] != 0", __func__, cachedBaseAddrs_.size()),
        HCCL_E_INTERNAL);
    cachedBaseAddrs_.resize(count);
    entryBytes_ += count * sizeof(uint64_t);
    CHK_SAFETY_FUNC_RET(
        memcpy_s(cachedBaseAddrs_.data(), count * sizeof(uint64_t), baseAddrs, count * sizeof(uint64_t)));

    // 缓存sizes
    CHK_PRT_RET(
        cachedMemSizes_.size() != 0,
        HCCL_ERROR("[AicpuTaskCacheEntry][%s] cachedMemSizes_.size[%u] != 0", __func__, cachedMemSizes_.size()),
        HCCL_E_INTERNAL);
    cachedMemSizes_.resize(count);
    entryBytes_ += count * sizeof(uint64_t);
    CHK_SAFETY_FUNC_RET(memcpy_s(cachedMemSizes_.data(), count * sizeof(uint64_t), memSizes, count * sizeof(uint64_t)));

    if (hasVOpInfo) {
        const uint64_t oldBytes = vOpInfo_.GetSize();
        vOpInfo_ = vOpInfo;
        entryBytes_ = entryBytes_ - oldBytes + vOpInfo_.GetSize();
        isVOp_ = true;
    }

    return HCCL_SUCCESS;
}

HcclResult AicpuTaskCacheEntry::AddSqeArray(
    RtsqA5* rtsqPtr, AicpuTsThread* aicpuTsThreadPtr, const uint64_t sqeCount, const uint8_t* sqeArray,
    const uint32_t streamId, const u64* sqeTransports, const u8* sqeLinkTypes, const u8* sqeTransportTypes)
{
    CHK_PTR_NULL(rtsqPtr);
    CHK_PTR_NULL(aicpuTsThreadPtr);
    CHK_PTR_NULL(sqeArray);
    CHK_PTR_NULL(sqeTransports);
    CHK_PTR_NULL(sqeLinkTypes);
    CHK_PTR_NULL(sqeTransportTypes);
    CHK_PRT_RET(sqeCount == 0, HCCL_ERROR("[AicpuTaskCacheEntry][AddSqeArray] sqeCount is 0"), HCCL_E_INTERNAL);

    const size_t sqeBytes = sqeCount * AC_SQE_SIZE;
    uint8_t* newSqeArray = ReinterpretAs<uint8_t*>(malloc(sqeBytes));
    CHK_PTR_NULL(newSqeArray);
    HcclResult ret = AddSqeArray_(newSqeArray, sqeBytes, sqeArray, streamId);
    if (ret != HCCL_SUCCESS) {
        free(newSqeArray);
        return ret;
    }

    // 更新SQE数组
    sqeArrayInfos_.emplace_back();
    SqeArrayInfo& sqeArrayInfo = sqeArrayInfos_.back();
    sqeArrayInfo.sqeArray = newSqeArray;
    sqeArrayInfo.rtsqPtr = rtsqPtr;
    sqeArrayInfo.aicpuTsThreadPtr = aicpuTsThreadPtr;
    sqeArrayInfo.sqeCount = sqeCount;
    sqeArrayInfo.srcAddrRefreshInfoArray.resize(sqeCount);
    sqeArrayInfo.dstAddrRefreshInfoArray.resize(sqeCount);
    sqeArrayInfo.sqeTransports.resize(sqeCount);
    sqeArrayInfo.sqeLinkTypes.resize(sqeCount);
    sqeArrayInfo.sqeTransportTypes.resize(sqeCount);
    for (uint64_t i = 0; i < sqeCount; i++) {
        sqeArrayInfo.sqeTransports[i] = sqeTransports[i];
        sqeArrayInfo.sqeLinkTypes[i] = sqeLinkTypes[i];
        sqeArrayInfo.sqeTransportTypes[i] = sqeTransportTypes[i];
    }
    entryBytes_ += sqeArrayInfo.GetSize();

    // 更新下发顺序
    launchOrder_.emplace_back(TaskArrayType::kTaskArrayTypeSqe);
    entryBytes_ += sizeof(TaskArrayType);

    HCCL_INFO(
        "[AicpuTaskCacheEntry][AddSqeArray] add %uth sqe array with sqeCount[%llu] streamId[%u]",
        sqeArrayInfos_.size() - 1, sqeCount, streamId);

    return HCCL_SUCCESS;
}

HcclResult AicpuTaskCacheEntry::AddWqeArray(
    UbConnLite* ubConnLitePtr, UbTransportLiteImpl* ubTransportLiteImplPtr, const vector<WqeTask>& wqeTasks,
    const uint32_t streamId, const uint32_t dbSqeIdx, const bool isReportTask, const DbSqeProfInfo& dbSqeProfInfo)
{
    CHK_PRT_RET(
        isReportTask != dbSqeProfInfo.isValid,
        HCCL_ERROR(
            "[AicpuTaskCacheEntry][AddWqeArray] isReportTask[%d] != dbSqeProfInfo.isValid[%d]", isReportTask,
            dbSqeProfInfo.isValid),
        HCCL_E_INTERNAL);
    CHK_PTR_NULL(ubConnLitePtr);
    CHK_PTR_NULL(ubTransportLiteImplPtr);

    const uint32_t wqeCount = wqeTasks.size();
    CHK_PRT_RET(wqeCount == 0, HCCL_ERROR("[AicpuTaskCacheEntry][AddWqeArray] wqeCount is 0"), HCCL_E_INTERNAL);

    // 更新streamIdToDbSqeTmpInfoMap_ (仅cache miss时临时维护, 展开完成后一定为空)
    DbSqeTmpInfo dbSqeTmpInfo;
    dbSqeTmpInfo.wqeArrayIdx = wqeTaskArrayInfos_.size();
    dbSqeTmpInfo.dbSqeIdx = dbSqeIdx;
    dbSqeTmpInfo.isReportTask = isReportTask;
    if (isReportTask) {
        dbSqeTmpInfo.dbSqeProfInfo = dbSqeProfInfo;
    }
    std::unordered_map<uint32_t, vector<DbSqeTmpInfo>>::iterator mapIter = streamIdToDbSqeTmpInfoMap_.find(streamId);
    if (mapIter == streamIdToDbSqeTmpInfoMap_.end()) {
        mapIter = streamIdToDbSqeTmpInfoMap_.emplace(streamId, vector<DbSqeTmpInfo>()).first;
    }
    mapIter->second.push_back(dbSqeTmpInfo);

    // 更新WQE数组
    wqeTaskArrayInfos_.emplace_back();
    WqeTaskArrayInfo& wqeTaskArrayInfo = wqeTaskArrayInfos_.back();
    wqeTaskArrayInfo.wqeTaskArray = wqeTasks;
    wqeTaskArrayInfo.originalOpcodes.reserve(wqeCount);
    wqeTaskArrayInfo.originalSgeNums.reserve(wqeCount);
    for (const WqeTask& task : wqeTasks) {
        const auto* common = static_cast<const UdmaSqeCommon*>(static_cast<const void*>(&task));
        wqeTaskArrayInfo.originalOpcodes.push_back(static_cast<uint8_t>(common->opcode));
        wqeTaskArrayInfo.originalSgeNums.push_back(static_cast<uint8_t>(common->sgeNum));
    }
    wqeTaskArrayInfo.ubConnLitePtr = ubConnLitePtr;
    wqeTaskArrayInfo.ubTransportLiteImplPtr = ubTransportLiteImplPtr;
    wqeTaskArrayInfo.dbSqeLocation.dbSqeIdx = dbSqeIdx;
    wqeTaskArrayInfo.locAddrRefreshInfoArray.resize(wqeCount);
    wqeTaskArrayInfo.rmtAddrRefreshInfoArray.resize(wqeCount);
    entryBytes_ += wqeTaskArrayInfo.GetSize();

    // 更新下发顺序
    launchOrder_.emplace_back(TaskArrayType::kTaskArrayTypeWqe);
    entryBytes_ += sizeof(TaskArrayType);

    // 初始化token info
    if (tokenInfosMap_.find(ubTransportLiteImplPtr) == tokenInfosMap_.end()) {
        const uint64_t addrCnt = cachedBaseAddrs_.size();
        tokenInfosMap_.emplace(ubTransportLiteImplPtr, vector<TokenInfo>(addrCnt));
        entryBytes_ += (sizeof(UbTransportLiteImplHandle) + addrCnt * sizeof(TokenInfo));
    }

    HCCL_INFO(
        "[AicpuTaskCacheEntry][AddWqeArray] add %uth wqe array with wqeCount[%u] dbSqeTmpInfo[%u, %d, %u] streamId[%u]",
        wqeTaskArrayInfos_.size() - 1, wqeCount, dbSqeTmpInfo.wqeArrayIdx, dbSqeTmpInfo.isReportTask,
        dbSqeTmpInfo.dbSqeIdx, streamId);

    return HCCL_SUCCESS;
}

HcclResult AicpuTaskCacheEntry::SubmitCacheEntry()
{
    // 校验streamIdToDbSqeTmpInfoMap_
    // 注意: SubmitCacheEntry时, 算子展开一定完成且task全部launch, 因此所有DbSqe的临时信息一定均已消耗
    if (UNLIKELY(streamIdToDbSqeTmpInfoMap_.size() != 0)) {
        HCCL_ERROR(
            "[AicpuTaskCacheEntry][SubmitCacheEntry] streamIdToDbSqeTmpInfoMap_.size[%u] != 0",
            streamIdToDbSqeTmpInfoMap_.size());
        for (const auto& mapIter : streamIdToDbSqeTmpInfoMap_) {
            HCCL_ERROR(
                "[AicpuTaskCacheEntry][SubmitCacheEntry] streamId[%u] wqeArrayIdx[%u] dbSqeIdx[%u] isReportTask[%d]",
                mapIter.first, mapIter.second[0].wqeArrayIdx, mapIter.second[0].dbSqeIdx,
                mapIter.second[0].isReportTask);
        }
        return HCCL_E_INTERNAL;
    }

    // 地址信息已经通过InitCacheEntry保存
    const uint64_t count = cachedBaseAddrs_.size();
    CHK_PRT_RET(
        count == 0, HCCL_ERROR("[AicpuTaskCacheEntry][SubmitCacheEntry] cachedBaseAddrs_.size is 0"), HCCL_E_INTERNAL);

    // 打印dynamic memory ranges
    if (UNLIKELY(HcclCheckLogLevel(HCCL_LOG_INFO))) {
        for (size_t memIdx = 0; memIdx < count; memIdx++) {
            HCCL_INFO(
                "[AicpuTaskCacheEntry][SubmitCacheEntry] memRanges[%u]: [0x%llx, 0x%llx)", memIdx,
                cachedBaseAddrs_[memIdx], cachedBaseAddrs_[memIdx] + cachedMemSizes_[memIdx]);
        }
    }

    CHK_RET(SubmitSqeAddrRefreshInfo_());
    CHK_RET(SubmitWqeAddrRefreshInfoAndTokenInfo_());
    CHK_RET(SubmitDbSqeProfRefreshInfo_());
    CHK_RET(ValidateLaunchOrder_());
    CHK_RET(BuildConnOverflowInfos_());

    return HCCL_SUCCESS;
}

inline HcclResult AicpuTaskCacheEntry::SubmitSqeAddrRefreshInfo_()
{
    // 更新每段SQE数组的AddrRefreshInfo
    for (size_t arrayIdx = 0; arrayIdx < sqeArrayInfos_.size(); arrayIdx++) {
        SqeArrayInfo& sqeArrayInfo = sqeArrayInfos_[arrayIdx];
        const uint64_t sqeCount = sqeArrayInfo.sqeCount;
        const uint8_t* sqePtr = sqeArrayInfo.sqeArray;
        CHK_PTR_NULL(sqePtr);
        for (size_t sqeIdx = 0; sqeIdx < sqeCount; sqeIdx++) {
            CHK_RET(UpdateSqeAddrRefreshInfo_(
                sqePtr, sqeArrayInfo.srcAddrRefreshInfoArray[sqeIdx], sqeArrayInfo.dstAddrRefreshInfoArray[sqeIdx]));

            // 打印SQE AddrRefreshInfo
            HCCL_INFO(
                "[AicpuTaskCacheEntry][SubmitCacheEntry] sqeArrayInfos_[%u][%u]: "
                "srcAddrRefreshInfo[needRefresh-%d memIdx-%u] dstAddrRefreshInfo[needRefresh-%d memIdx-%u]",
                arrayIdx, sqeIdx, sqeArrayInfo.srcAddrRefreshInfoArray[sqeIdx].needRefresh,
                sqeArrayInfo.srcAddrRefreshInfoArray[sqeIdx].memIdx,
                sqeArrayInfo.dstAddrRefreshInfoArray[sqeIdx].needRefresh,
                sqeArrayInfo.dstAddrRefreshInfoArray[sqeIdx].memIdx);
            sqePtr += AC_SQE_SIZE;
        }
    }
    return HCCL_SUCCESS;
}

inline HcclResult AicpuTaskCacheEntry::SubmitWqeAddrRefreshInfoAndTokenInfo_()
{
    // 更新每段WQE数组的AddrRefreshInfo, 并更新token info
    for (size_t arrayIdx = 0; arrayIdx < wqeTaskArrayInfos_.size(); arrayIdx++) {
        // 获取token info
        WqeTaskArrayInfo& wqeTaskArrayInfo = wqeTaskArrayInfos_[arrayIdx];
        UbTransportLiteImpl* ubTransportLiteImplPtr
            = wqeTaskArrayInfo.ubTransportLiteImplPtr; // 注意: AddWqeArray时已校验非空
        std::unordered_map<UbTransportLiteImplHandle, vector<TokenInfo>>::iterator iter
            = tokenInfosMap_.find(ubTransportLiteImplPtr);

        CHK_PRT_RET(
            iter == tokenInfosMap_.end(),
            HCCL_ERROR(
                "[AicpuTaskCacheEntry][SubmitCacheEntry] ubTransportLiteImplPtr[%p] not found in tokenInfosMap_",
                ubTransportLiteImplPtr),
            HCCL_E_INTERNAL);
        vector<TokenInfo>& tokenInfos = iter->second;

        uint32_t remoteRankId = 0;
        if (isVOp_) {
            const ChannelHandle channelHandle = ReinterpretAs<ChannelHandle>(ubTransportLiteImplPtr);
            CHK_RET(AicpuTaskCacheManager::GetChannelRemoteRankId(channelHandle, remoteRankId));
            CHK_PRT_RET(
                remoteRankId >= vOpInfo_.sendCounts.size() || remoteRankId >= vOpInfo_.recvCounts.size()
                    || remoteRankId >= vOpInfo_.sdispls.size() || remoteRankId >= vOpInfo_.rdispls.size(),
                HCCL_WARNING(
                    "[AicpuTaskCacheEntry][%s] channelHandle[0x%llx] remoteRankId[%u] exceeds V arrays", __func__,
                    static_cast<unsigned long long>(channelHandle), remoteRankId),
                HCCL_E_PARA);
        }

        // 更新AddrRefreshInfo和token info
        const uint32_t wqeCount = wqeTaskArrayInfo.wqeTaskArray.size();
        vector<WqeTask>& wqeTasks = wqeTaskArrayInfo.wqeTaskArray;
        for (size_t wqeIdx = 0; wqeIdx < wqeCount; wqeIdx++) {
            CHK_RET(UpdateWqeAddrRefreshInfoAndTokenInfo_(
                wqeTasks[wqeIdx], wqeTaskArrayInfo.locAddrRefreshInfoArray[wqeIdx],
                wqeTaskArrayInfo.rmtAddrRefreshInfoArray[wqeIdx], tokenInfos));

            if (isVOp_) {
                // 记录通道对端 rank，供后续刷新索引本端的 counts/displs；不修改 WQE 内容。
                wqeTaskArrayInfo.locAddrRefreshInfoArray[wqeIdx].rankId = remoteRankId;
                wqeTaskArrayInfo.rmtAddrRefreshInfoArray[wqeIdx].rankId = remoteRankId;
            }

            // 打印WQE AddrRefreshInfo
            HCCL_INFO(
                "[AicpuTaskCacheEntry][SubmitCacheEntry] wqeTaskArrayInfos_[%u][%u]: "
                "locAddrRefreshInfo[needRefresh-%d memIdx-%u] needLocTokenIdFlag[%d] "
                "rmtAddrRefreshInfo[needRefresh-%d memIdx-%u] needRmtTokenIdAndValueFlag[%d]",
                arrayIdx, wqeIdx, wqeTaskArrayInfo.locAddrRefreshInfoArray[wqeIdx].needRefresh,
                wqeTaskArrayInfo.locAddrRefreshInfoArray[wqeIdx].memIdx,
                tokenInfos[wqeTaskArrayInfo.locAddrRefreshInfoArray[wqeIdx].memIdx].needLocTokenIdFlag,
                wqeTaskArrayInfo.rmtAddrRefreshInfoArray[wqeIdx].needRefresh,
                wqeTaskArrayInfo.rmtAddrRefreshInfoArray[wqeIdx].memIdx,
                tokenInfos[wqeTaskArrayInfo.rmtAddrRefreshInfoArray[wqeIdx].memIdx].needRmtTokenIdAndValueFlag);
        }
    }
    return HCCL_SUCCESS;
}

inline HcclResult AicpuTaskCacheEntry::SubmitDbSqeProfRefreshInfo_()
{
    // 更新每个DbSqeProfAndRefreshInfo的AddrRefreshInfo
    for (std::unordered_map<DbSqeLocation, DbSqeProfAndRefreshInfo>::iterator iter = dbSqeLocInfoMap_.begin();
         iter != dbSqeLocInfoMap_.end(); iter++) {
        DbSqeProfAndRefreshInfo& dbSqeProfAndRefreshInfo = iter->second;
        const DbSqeProfInfo& dbSqeProfInfo = dbSqeProfAndRefreshInfo.dbSqeProfInfo;
        switch (static_cast<u8>(dbSqeProfInfo.taskParamType)) {
            case TaskParamTypeVal::TASK_UB:
            case TaskParamTypeVal::TASK_UB_REDUCE_INLINE:
            case TaskParamTypeVal::TASK_WRITE_WITH_NOTIFY:
            case TaskParamTypeVal::TASK_WRITE_REDUCE_WITH_NOTIFY:
                CHK_RET(UpdateAddrRefreshInfo_(
                    dbSqeProfInfo.locAddr, dbSqeProfAndRefreshInfo.locAddrRefreshInfo, AddrType::kAddrTypeLocal));
                CHK_RET(UpdateAddrRefreshInfo_(
                    dbSqeProfInfo.rmtAddr, dbSqeProfAndRefreshInfo.rmtAddrRefreshInfo, AddrType::kAddrTypeRemote));
                break;
            case TaskParamTypeVal::TASK_UB_INLINE_WRITE:
                CHK_RET(UpdateAddrRefreshInfo_(
                    dbSqeProfInfo.rmtAddr, dbSqeProfAndRefreshInfo.rmtAddrRefreshInfo, AddrType::kAddrTypeRemote));
                break;
            default:
                HCCL_ERROR(
                    "[AicpuTaskCacheEntry][%s] invalid taskParamType[%u]", __func__, dbSqeProfInfo.taskParamType);
                return HCCL_E_INTERNAL;
        }
    }
    return HCCL_SUCCESS;
}

inline HcclResult AicpuTaskCacheEntry::ValidateLaunchOrder_()
{
    // 按照缓存时的下发顺序, 校验launchOrder_对应的sqe/wqeArrayIdx, 避免后续命中时重复校验
    size_t sqeArrayIdx = 0;
    size_t wqeArrayIdx = 0;
    for (size_t launchIdx = 0; launchIdx < launchOrder_.size(); launchIdx++) {
        const TaskArrayType taskType = launchOrder_[launchIdx];
        if (taskType == TaskArrayType::kTaskArrayTypeSqe) {
            // 校验arrayIdx
            CHK_PRT_RET(
                sqeArrayIdx >= sqeArrayInfos_.size(),
                HCCL_ERROR(
                    "[AicpuTaskCacheEntry][SubmitCacheEntry] sqeArrayIdx[%u] >= sqeArrayInfos_.size[%u]", sqeArrayIdx,
                    sqeArrayInfos_.size()),
                HCCL_E_PARA);
            ++sqeArrayIdx;
        } else if (taskType == TaskArrayType::kTaskArrayTypeWqe) {
            // 校验arrayIdx
            CHK_PRT_RET(
                wqeArrayIdx >= wqeTaskArrayInfos_.size(),
                HCCL_ERROR(
                    "[AicpuTaskCacheEntry][SubmitCacheEntry] wqeArrayIdx[%u] >= wqeTaskArrayInfos_.size[%u]",
                    wqeArrayIdx, wqeTaskArrayInfos_.size()),
                HCCL_E_PARA);
            ++wqeArrayIdx;
        } else {
            HCCL_ERROR("[AicpuTaskCacheEntry][SubmitCacheEntry] invalid task array type");
            return HCCL_E_INTERNAL;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuTaskCacheEntry::RefreshAndLaunch(
    const uint64_t* baseAddrs, const uint64_t* memSizes, const uint32_t count, const VOpInfo& vOpInfo)
{
    // 校验count
    CHK_PRT_RET(
        cachedBaseAddrs_.size() != count || cachedMemSizes_.size() != count,
        HCCL_ERROR(
            "[AicpuTaskCacheEntry][RefreshAndLaunch] cachedBaseAddrs_.size[%u] / cachedMemSizes_.size[%u] != count[%u]",
            cachedBaseAddrs_.size(), cachedMemSizes_.size(), count),
        HCCL_E_INTERNAL);

    // 校验memSizes
    for (uint32_t memIdx = 0; memIdx < count; memIdx++) {
        CHK_PRT_RET(
            !isVOp_ && cachedMemSizes_[memIdx] != memSizes[memIdx],
            HCCL_ERROR("[AicpuTaskCacheEntry][RefreshAndLaunch] cachedMemSizes_[%u] != memSizes[%u]", memIdx, memIdx),
            HCCL_E_INTERNAL);
        CHK_PRT_RET(
            ((!isVOp_ || memIdx >= TASK_CACHE_LOCAL_MEM_COUNT) && baseAddrs[memIdx] == 0)
                || memSizes[memIdx] > UINT64_MAX - baseAddrs[memIdx],
            HCCL_ERROR("[AicpuTaskCacheEntry][RefreshAndLaunch] invalid memory range[%u]", memIdx), HCCL_E_PARA);
        HCCL_DEBUG(
            "[AicpuTaskCacheEntry][RefreshAndLaunch] memIdx[%u] addrType[%s] baseAddr[0x%llx] memSize[%llu]", memIdx,
            memIdx < TASK_CACHE_LOCAL_MEM_COUNT ? "local" : "remote", baseAddrs[memIdx], memSizes[memIdx]);
    }
    HCCL_INFO(
        "[AicpuTaskCacheEntry][RefreshAndLaunch] parsed localMemCount[%u] remoteMemCount[%u] totalMemCount[%u]",
        TASK_CACHE_LOCAL_MEM_COUNT, count - TASK_CACHE_LOCAL_MEM_COUNT, count);

    if (isVOp_) {
        const size_t rankSize = vOpInfo_.sendCounts.size();
        CHK_PRT_RET(
            rankSize == 0 || vOpInfo.sendCounts.size() != rankSize || vOpInfo.recvCounts.size() != rankSize
                || vOpInfo.sdispls.size() != rankSize || vOpInfo.rdispls.size() != rankSize,
            HCCL_ERROR("[AicpuTaskCacheEntry][RefreshAndLaunch] invalid V metadata sizes"), HCCL_E_PARA);
        for (size_t rank = 0; rank < rankSize; ++rank) {
            CHK_PRT_RET(
                (vOpInfo.sendCounts[rank] != 0
                 && (baseAddrs[0] == 0 || vOpInfo.sdispls[rank] > memSizes[0]
                     || vOpInfo.sendCounts[rank] > memSizes[0] - vOpInfo.sdispls[rank]))
                    || (vOpInfo.recvCounts[rank] != 0
                        && (baseAddrs[1] == 0 || vOpInfo.rdispls[rank] > memSizes[1]
                            || vOpInfo.recvCounts[rank] > memSizes[1] - vOpInfo.rdispls[rank])),
                HCCL_ERROR("[AicpuTaskCacheEntry][RefreshAndLaunch] V range exceeds buffer, rank[%zu]", rank),
                HCCL_E_PARA);
        }

        // 调用方通过全peer非零预热和准入约束保证任务模板完整；这里只校验元数据合法性。
        CHK_RET(ValidateVOpWqeMetadata_(vOpInfo));
    }

    // 下发前检查jetty SQ深度, 避免cache hit时WQE溢出
    // 溢出时阻塞等待空间释放, 等待超时抛出InternalException
    CHK_RET(CheckWqeOverflow_());

    // V 类暂沿用录制时的 token；非 V 类保持原有的按新地址刷新 token 行为。
    if (!isVOp_) {
        CHK_RET(RefreshTokenInfos_(baseAddrs, memSizes, count));
    }

    const bool enableTaskException = hcomm::GetTaskExceptionEnable();
    const bool l1State = Hccl::DfxProfilingHandlerLite::GetInstance().GetProfL1State();
    const bool needTaskParam = (l1State || enableTaskException);
    HCCL_INFO(
        "[AicpuTaskCacheEntry][RefreshAndLaunch] l1State[%d] enableTaskException[%d] -> needTaskParam[%d]", l1State,
        enableTaskException, needTaskParam);

    CHK_RET(LaunchTasksByOrder_(baseAddrs, memSizes, count, needTaskParam, vOpInfo));

    if (UNLIKELY(HcclCheckLogLevel(HCCL_LOG_INFO))) {
        CHK_RET(PrintRefreshResult_(baseAddrs, memSizes, count));
    }

    return HCCL_SUCCESS;
}

inline HcclResult AicpuTaskCacheEntry::BuildConnOverflowInfos_()
{
    // 按UbConnLite分组, 统计每个connection的WQE实际占用的wqebb总数, 预计算后供CheckWqeOverflow_使用
    // 注意1: 同一个UbConnLite可能被多个WQE数组引用, 需要累加占用数量
    // 注意2: 每个WQE占用1个wqebb, 带notify的WQE额外多占1个(参考LaunchOneWqeWithNotify中pi += 2)
    connOverflowInfos_.clear();
    for (size_t i = 0; i < wqeTaskArrayInfos_.size(); i++) {
        UbConnLite* ubConnLitePtr = wqeTaskArrayInfos_[i].ubConnLitePtr;
        const vector<WqeTask>& wqeTasks = wqeTaskArrayInfos_[i].wqeTaskArray;
        uint32_t wqebbCount = static_cast<uint32_t>(wqeTasks.size());
        for (size_t wqeIdx = 0; wqeIdx < wqeTasks.size(); wqeIdx++) {
            const UdmaSqeCommon* wqeCommonPtr
                = static_cast<const UdmaSqeCommon*>(static_cast<const void*>(&wqeTasks[wqeIdx]));
            if (static_cast<uint8_t>(wqeCommonPtr->opcode) == WRITE_WITH_NOTIFY_OPCODE) {
                wqebbCount += 1;
            }
        }
        bool found = false;
        for (auto& info : connOverflowInfos_) {
            if (info.ubConnLitePtr == ubConnLitePtr) {
                info.wqebbCount += wqebbCount;
                found = true;
                break;
            }
        }
        if (!found) {
            connOverflowInfos_.push_back({ubConnLitePtr, wqebbCount});
        }
    }
    return HCCL_SUCCESS;
}

inline HcclResult AicpuTaskCacheEntry::CheckWqeOverflow_()
{
    for (const auto& info : connOverflowInfos_) {
        EXCEPTION_CATCH(info.ubConnLitePtr->MakeSureAvailableSpace(info.wqebbCount), return HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuTaskCacheEntry::ValidateVOpWqeMetadata_(const VOpInfo& vOpInfo) const
{
    for (const WqeTaskArrayInfo& arrayInfo : wqeTaskArrayInfos_) {
        CHK_PRT_RET(
            arrayInfo.originalOpcodes.size() != arrayInfo.wqeTaskArray.size()
                || arrayInfo.originalSgeNums.size() != arrayInfo.wqeTaskArray.size()
                || arrayInfo.locAddrRefreshInfoArray.size() != arrayInfo.wqeTaskArray.size()
                || arrayInfo.rmtAddrRefreshInfoArray.size() != arrayInfo.wqeTaskArray.size(),
            HCCL_ERROR("[AicpuTaskCacheEntry][%s] invalid WQE cache metadata", __func__), HCCL_E_INTERNAL);
        for (size_t wqeIdx = 0; wqeIdx < arrayInfo.wqeTaskArray.size(); ++wqeIdx) {
            const auto* wqeCommonPtr
                = static_cast<const UdmaSqeCommon*>(static_cast<const void*>(&arrayInfo.wqeTaskArray[wqeIdx]));
            const uint8_t wqeCode = arrayInfo.originalOpcodes[wqeIdx];
            if (wqeCode == UdmaSqOpcode::UDMA_OPC_WRITE && wqeCommonPtr->inlineEn != 0) {
                continue;
            }
            const uint32_t rankId = arrayInfo.locAddrRefreshInfoArray[wqeIdx].rankId;
            const bool isRead = wqeCode == UdmaSqOpcode::UDMA_OPC_READ;
            const auto& oldCounts = isRead ? vOpInfo_.recvCounts : vOpInfo_.sendCounts;
            const auto& newCounts = isRead ? vOpInfo.recvCounts : vOpInfo.sendCounts;
            CHK_PRT_RET(
                rankId >= oldCounts.size() || rankId >= newCounts.size() || newCounts[rankId] > UINT32_MAX,
                HCCL_ERROR("[AicpuTaskCacheEntry][%s] invalid V WQE count, rank[%u]", __func__, rankId), HCCL_E_PARA);
        }
    }
    return HCCL_SUCCESS;
}

inline HcclResult AicpuTaskCacheEntry::LaunchTasksByOrder_(
    const uint64_t* baseAddrs, const uint64_t* memSizes, const uint32_t count, bool needTaskParam,
    const VOpInfo& vOpInfo)
{
    // 按照缓存时的下发顺序, 依次刷新并下发task
    size_t sqeArrayIdx = 0;
    size_t wqeArrayIdx = 0;

    for (size_t launchIdx = 0; launchIdx < launchOrder_.size(); launchIdx++) {
        const TaskArrayType taskType = launchOrder_[launchIdx];
        if (taskType == TaskArrayType::kTaskArrayTypeSqe) {
            // 注意: sqeArrayIdx已在SubmitCacheEntry校验, 无需重复校验
            const SqeArrayInfo& sqeArrayInfo = sqeArrayInfos_[sqeArrayIdx];

            // V 类命中时保存每条 SQE 的类型，成功或失败均在返回前恢复。
            std::vector<Rt91095StarsSqeType> originalSqeType;
            const uint64_t sqeCount = sqeArrayInfo.sqeCount;
            if (isVOp_) {
                originalSqeType.reserve(sqeCount);
                for (uint64_t sqeIndex = 0; sqeIndex < sqeCount; ++sqeIndex) {
                    auto* sqeHeaderPtr = static_cast<Rt91095StarsSqeHeader*>(
                        static_cast<void*>(sqeArrayInfo.sqeArray + sqeIndex * AC_SQE_SIZE));
                    const Rt91095StarsSqeType sqeType = static_cast<Rt91095StarsSqeType>(sqeHeaderPtr->type);
                    originalSqeType.push_back(sqeType);
                }
            }
            // 刷新SQE数组
            HcclResult sqeResult = RefreshSqeTasks_(sqeArrayInfo, baseAddrs, vOpInfo);

            // 下发SQE
            if (sqeResult == HCCL_SUCCESS) {
                sqeResult = LaunchSqeTasks_(sqeArrayInfo);
            }

            // 按实际下发的 SQE 类型上报；V 类 SDMA 可能已变成 PLACE_HOLDER。
            if (sqeResult == HCCL_SUCCESS && needTaskParam) {
                sqeResult = ReportSqeArrayProfiling_(sqeArrayIdx, baseAddrs, memSizes, count);
            }

            if (isVOp_) {
                for (uint64_t sqeIndex = 0; sqeIndex < sqeCount; ++sqeIndex) {
                    auto* sqeHeaderPtr = static_cast<Rt91095StarsSqeHeader*>(
                        static_cast<void*>(sqeArrayInfo.sqeArray + sqeIndex * AC_SQE_SIZE));
                    sqeHeaderPtr->type = static_cast<uint8_t>(originalSqeType[sqeIndex]);
                }
            }
            CHK_RET(sqeResult);
            ++sqeArrayIdx;
        } else if (taskType == TaskArrayType::kTaskArrayTypeWqe) {
            // 注意: wqeArrayIdx已在SubmitCacheEntry校验, 无需重复校验
            WqeTaskArrayInfo& wqeTaskArrayInfo = wqeTaskArrayInfos_[wqeArrayIdx];

            // 刷新WQE数组
            CHK_RET(RefreshWqeTasks_(wqeTaskArrayInfo, baseAddrs, memSizes, count, vOpInfo));

            // 下发WQE数组
            CHK_RET(LaunchWqeTasks_(wqeTaskArrayInfo));

            // 刷新对应DbSqe
            CHK_RET(RefreshDbSqe_(wqeTaskArrayInfo));
            ++wqeArrayIdx;
        } else {
            HCCL_ERROR("[AicpuTaskCacheEntry][RefreshAndLaunch] invalid task array type[%u]", taskType);
            return HCCL_E_INTERNAL;
        }
    }
    return HCCL_SUCCESS;
}

inline HcclResult
AicpuTaskCacheEntry::RefreshTokenInfos_(const uint64_t* baseAddrs, const uint64_t* memSizes, const uint32_t count)
{
    // 按需统一获取新传入内存地址的token信息用于WQE刷新, 避免重复构造Buffer
    for (std::unordered_map<UbTransportLiteImplHandle, vector<TokenInfo>>::iterator iter = tokenInfosMap_.begin();
         iter != tokenInfosMap_.end(); iter++) {
        vector<TokenInfo>& tokenInfos = iter->second;
        CHK_PRT_RET(
            tokenInfos.size() != count,
            HCCL_ERROR(
                "[AicpuTaskCacheEntry][RefreshAndLaunch] tokenInfos.size[%u] != count[%u]", tokenInfos.size(), count),
            HCCL_E_INTERNAL);

        // 注意: AddWqeArray时已校验非空, 无需重复校验
        UbTransportLiteImpl* ubTransportLitePtr = ReinterpretAs<UbTransportLiteImpl*>(iter->first);
        for (uint32_t memIdx = 0; memIdx < count; memIdx++) {
            const uint64_t baseAddr = baseAddrs[memIdx];
            const uint64_t memSize = memSizes[memIdx];
            TokenInfo& tokenInfo = tokenInfos[memIdx];

            if (tokenInfo.needLocTokenIdFlag) {
                // 参考hccl_api_data_aicpu_ts.cc (例如HcommWriteOnThread), 获取新loc token id
                // 注意: 无需通过ubTransportLitePtr->GetRmaBufSlicelite(locRmaBuf)构造Hccl::RmaBufSliceLite
                // locRmaBufSlicelite,
                //     再调用locRmaBufSlicelite.GetTokenId()获取token id (一定与Hccl::RmaBufferLite locRmaBuf的token
                //     id相同)
                Hccl::RmaBufferLite locRmaBuf;
                CHK_RET(
                    ubTransportLitePtr->BuildLocRmaBufferLite(ReinterpretAs<uintptr_t>(baseAddr), memSize, locRmaBuf));
                tokenInfo.locTokenId = locRmaBuf.GetTokenId();
            }

            if (tokenInfo.needRmtTokenIdAndValueFlag) {
                // 参考hccl_api_data_aicpu_ts.cc (例如HcommRead/Write/WriteReduce/WriteWithNotifyOnThread), 获取新rmt
                // token id/value 注意: Hccl::Buffer本身不含token id/value,
                // 必须通过调用ubTransportLitePtr->GetRmtRmaBufSliceLite,
                //     构造Hccl::RmtRmaBufSliceLite, 再调用GetTokenId/Value获取token id/value
                const Hccl::Buffer rmtBuf{ReinterpretAs<uintptr_t>(baseAddr), memSize};
                Hccl::RmtRmaBufSliceLite rmtRmaBufSlicelite = ubTransportLitePtr->GetRmtRmaBufSliceLite(rmtBuf);
                tokenInfo.rmtTokenId = rmtRmaBufSlicelite.GetTokenId();
                tokenInfo.rmtTokenValue = rmtRmaBufSlicelite.GetTokenValue();
            }
        }
    }
    return HCCL_SUCCESS;
}

inline HcclResult AicpuTaskCacheEntry::PrintRefreshResult_(
    [[maybe_unused]] const uint64_t* baseAddrs, [[maybe_unused]] const uint64_t* memSizes, const uint32_t count)
{
    // 打印录制时收集的 token 需求标记；非 V 回放按新地址刷新 token。
    for (std::unordered_map<UbTransportLiteImplHandle, vector<TokenInfo>>::iterator iter = tokenInfosMap_.begin();
         iter != tokenInfosMap_.end(); iter++) {
        UbTransportLiteImpl* ubTransportLitePtr = ReinterpretAs<UbTransportLiteImpl*>(iter->first);
        vector<TokenInfo>& tokenInfos = iter->second;
        for (uint32_t memIdx = 0; memIdx < count; memIdx++) {
            TokenInfo& tokenInfo = tokenInfos[memIdx];
            HCCL_INFO(
                "[AicpuTaskCacheEntry][RefreshAndLaunch] tokenInfosMap_[0x%llx][%u]: "
                "needLocTokenIdFlag[%d]; rmtTokenIdAndValueFlag[%d]",
                ubTransportLitePtr, memIdx, tokenInfo.needLocTokenIdFlag, tokenInfo.needRmtTokenIdAndValueFlag);
        }
    }
    // 打印刷新的顺序
    size_t sqeArrayIdx = 0;
    size_t wqeArrayIdx = 0;
    for (size_t launchIdx = 0; launchIdx < launchOrder_.size(); launchIdx++) {
        const TaskArrayType taskType = launchOrder_[launchIdx];
        if (taskType == TaskArrayType::kTaskArrayTypeSqe) {
            HCCL_INFO(
                "[AicpuTaskCacheEntry][RefreshAndLaunch] launchedArray[%u] is sqeArray[%u]", launchIdx, sqeArrayIdx);
            ++sqeArrayIdx;
        } else if (taskType == TaskArrayType::kTaskArrayTypeWqe) {
            HCCL_INFO(
                "[AicpuTaskCacheEntry][RefreshAndLaunch] launchedArray[%u] is wqeArray[%u]", launchIdx, wqeArrayIdx);
            ++wqeArrayIdx;
        } else {
            HCCL_ERROR("[AicpuTaskCacheEntry][RefreshAndLaunch] invalid task array type[%u]", taskType);
            return HCCL_E_INTERNAL;
        }
    }
    return HCCL_SUCCESS;
}

inline HcclResult AicpuTaskCacheEntry::AddSqeArray_(
    uint8_t* newSqeArray, const size_t sqeBytes, const uint8_t* sqeArray, const uint32_t streamId)
{
    CHK_SAFETY_FUNC_RET(memcpy_s(newSqeArray, sqeBytes, sqeArray, sqeBytes));

    // 按需更新DbSqe相关信息
    std::unordered_map<uint32_t, vector<DbSqeTmpInfo>>::iterator mapIter = streamIdToDbSqeTmpInfoMap_.find(streamId);
    if (mapIter != streamIdToDbSqeTmpInfoMap_.end()) {
        // 当前SQE数组可能存在多个DbSqe, 需要逐一更新
        for (size_t i = 0; i < mapIter->second.size(); ++i) {
            // 更新DbSqeLocation (AddWqeArray时已分配并设置相关字段, 此处只需设置sqeArrayIdx)
            const DbSqeTmpInfo& dbSqeTmpInfo = mapIter->second[i];
            CHK_PRT_RET(
                dbSqeTmpInfo.wqeArrayIdx >= wqeTaskArrayInfos_.size(),
                HCCL_ERROR(
                    "[AicpuTaskCacheEntry][AddSqeArray_] streamIdToDbSqeTmpInfoMap_[%u][%u].wqeArrayIdx[%u] >= "
                    "wqeTaskArrayInfos_.size[%u]",
                    streamId, i, dbSqeTmpInfo.wqeArrayIdx, wqeTaskArrayInfos_.size()),
                HCCL_E_INTERNAL);
            DbSqeLocation& dbSqeLocation = wqeTaskArrayInfos_[dbSqeTmpInfo.wqeArrayIdx].dbSqeLocation;
            dbSqeLocation.sqeArrayIdx = static_cast<uint32_t>(sqeArrayInfos_.size());

            // 按需更新dbSqeLocInfoMap_
            if (dbSqeTmpInfo.isReportTask) {
                // dbSqeLocInfoMap_中一定不存在对应的DbSqeProfInfo
                CHK_PRT_RET(
                    dbSqeLocInfoMap_.find(dbSqeLocation) != dbSqeLocInfoMap_.end(),
                    HCCL_ERROR(
                        "[AicpuTaskCacheEntry][AddSqeArray_] DbSqeLocation[%u, %u] already exists in "
                        "dbSqeLocInfoMap_ for streamId[%u]",
                        dbSqeLocation.sqeArrayIdx, dbSqeLocation.dbSqeIdx, streamId),
                    HCCL_E_INTERNAL);

                // 保存到dbSqeLocInfoMap_
                // 注意: dbSqeProfAndRefreshInfo中的src/dstAddrRefreshInfo在SubmitCacheEntry时更新
                DbSqeProfAndRefreshInfo dbSqeProfAndRefreshInfo;
                dbSqeProfAndRefreshInfo.dbSqeProfInfo = dbSqeTmpInfo.dbSqeProfInfo;
                dbSqeProfAndRefreshInfo.wqeArrayIdx
                    = dbSqeTmpInfo.wqeArrayIdx; // 注意: 本函数已校验dbSqeTmpInfo.wqeArrayIdx
                dbSqeLocInfoMap_.emplace(dbSqeLocation, dbSqeProfAndRefreshInfo);
            }

            HCCL_INFO(
                "[AicpuTaskCacheEntry][AddSqeArray_] update dbSqeLocation[%u, %u] isReportTask[%d]",
                dbSqeLocation.sqeArrayIdx, dbSqeLocation.dbSqeIdx, dbSqeTmpInfo.isReportTask);
        }

        // 更新后清理当前SQE数组对应的DbSqe临时信息
        streamIdToDbSqeTmpInfoMap_.erase(mapIter);
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuTaskCacheEntry::UpdateSqeAddrRefreshInfo_(
    const uint8_t* sqePtr, AddrRefreshInfo& srcAddrRefreshInfo, AddrRefreshInfo& dstAddrRefreshInfo) const
{
    // 参考sqe_build_a5.h, 提取给定SQE的AddrRefreshInfo

    // 提取sqeType，频繁调用私有函数，上层保证指针不为空
    const Rt91095StarsSqeHeader* sqeHeaderPtr
        = static_cast<const Rt91095StarsSqeHeader*>(static_cast<const void*>(sqePtr));
    const Rt91095StarsSqeType sqeType = static_cast<Rt91095StarsSqeType>(sqeHeaderPtr->type);

    // 根据sqeType提取AddrRefreshInfo
    switch (sqeType) {
        case Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT:
        case Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_RECORD:
        case Rt91095StarsSqeType::RT_91095_SQE_TYPE_UBDMA:
            // 无地址字段, 直接跳过
            break;
        case Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA: {
            const Rt91095StarsMemcpySqe* memcpySqePtr
                = static_cast<const Rt91095StarsMemcpySqe*>(static_cast<const void*>(sqePtr));
            CHK_RET(UpdateAddrRefreshInfo_(
                memcpySqePtr->u.strideMode0.srcAddrLow, memcpySqePtr->u.strideMode0.srcAddrHigh, srcAddrRefreshInfo,
                AddrType::kAddrTypeUnknown));
            CHK_RET(UpdateAddrRefreshInfo_(
                memcpySqePtr->u.strideMode0.dstAddrLow, memcpySqePtr->u.strideMode0.dstAddrHigh, dstAddrRefreshInfo,
                AddrType::kAddrTypeUnknown));
            if (isVOp_ && memcpySqePtr->opcode == SDMA_OPCODE_MEMCPY) {
                CHK_RET(UpdateVAddrRankId_(srcAddrRefreshInfo));
                CHK_RET(UpdateVAddrRankId_(dstAddrRefreshInfo));
            }
            break;
        }
        case Rt91095StarsSqeType::RT_91095_SQE_TYPE_WRITE_VALUE: {
            const Rt91095StarsWriteValueSqe* writeValueSqePtr
                = static_cast<const Rt91095StarsWriteValueSqe*>(static_cast<const void*>(sqePtr));
            CHK_RET(UpdateAddrRefreshInfo_(
                writeValueSqePtr->writeAddrLow, writeValueSqePtr->writeAddrHigh, dstAddrRefreshInfo,
                AddrType::kAddrTypeUnknown));
            break;
        }
        default:
            // sqe_build_a5.h中未使用的SQE类型, 告警后报错
            HCCL_ERROR("[AicpuTaskCacheEntry][UpdateSqeAddrRefreshInfo_] unexpected sqeType[%u]", sqeType);
            return HCCL_E_INTERNAL;
    }

    return HCCL_SUCCESS;
}

HcclResult AicpuTaskCacheEntry::UpdateWqeAddrRefreshInfoAndTokenInfo_(
    const WqeTask& wqeTask, AddrRefreshInfo& locAddrRefreshInfo, AddrRefreshInfo& rmtAddrRefreshInfo,
    vector<TokenInfo>& tokenInfos) const
{
    // 参考ub_conn_lite.cc, 提取给定WQE的AddrRefreshInfo

    // 提取wqeCode
    // 注意: BatchOneSidedRead/Write只是对multi-slice封装的接口, 最终还是规约到normal read/write
    const UdmaSqeCommon* wqeCommonPtr = static_cast<const UdmaSqeCommon*>(static_cast<const void*>(&wqeTask));
    const uint8_t wqeCode = static_cast<uint8_t>(wqeCommonPtr->opcode); // opcode来自于UdmaSqOpcode, 一定在uint8范围内
    switch (wqeCode) {
        case UdmaSqOpcode::UDMA_OPC_READ: // UdmaSqeWrite
            // normal read
            CHK_RET(UpdateAddrRefreshInfo_(
                wqeTask.wqeWrite.u.sge.dataAddrLow, wqeTask.wqeWrite.u.sge.dataAddrHigh, locAddrRefreshInfo,
                AddrType::kAddrTypeLocal));
            CHK_RET(UpdateAddrRefreshInfo_(
                wqeTask.wqeWrite.comm.rmtAddrLow, wqeTask.wqeWrite.comm.rmtAddrHigh, rmtAddrRefreshInfo,
                AddrType::kAddrTypeRemote));
            break;
        case UdmaSqOpcode::UDMA_OPC_WRITE: { // UdmaSqeWrite
            const uint32_t inlineEn = wqeTask.wqeWrite.comm.inlineEn;
            if (inlineEn != 0) { // inline write
                CHK_RET(UpdateAddrRefreshInfo_(
                    wqeTask.wqeWrite.comm.rmtAddrLow, wqeTask.wqeWrite.comm.rmtAddrHigh, rmtAddrRefreshInfo,
                    AddrType::kAddrTypeRemote));
            } else { // normal write or write reduce
                CHK_RET(UpdateAddrRefreshInfo_(
                    wqeTask.wqeWrite.u.sge.dataAddrLow, wqeTask.wqeWrite.u.sge.dataAddrHigh, locAddrRefreshInfo,
                    AddrType::kAddrTypeLocal));
                CHK_RET(UpdateAddrRefreshInfo_(
                    wqeTask.wqeWrite.comm.rmtAddrLow, wqeTask.wqeWrite.comm.rmtAddrHigh, rmtAddrRefreshInfo,
                    AddrType::kAddrTypeRemote));
            }
            break;
        }
        case WRITE_WITH_NOTIFY_OPCODE: // UdmaSqeWriteWithNotify
            // write with notify (对于给定slice的最后一个UB chunk)
            CHK_RET(UpdateAddrRefreshInfo_(
                wqeTask.wqeWriteWithNotify.localU.sge.dataAddrLow, wqeTask.wqeWriteWithNotify.localU.sge.dataAddrHigh,
                locAddrRefreshInfo, AddrType::kAddrTypeLocal));
            CHK_RET(UpdateAddrRefreshInfo_(
                wqeTask.wqeWriteWithNotify.comm.rmtAddrLow, wqeTask.wqeWriteWithNotify.comm.rmtAddrHigh,
                rmtAddrRefreshInfo, AddrType::kAddrTypeRemote));
            break;
        default:
            // ub_conn_lite.cc中未使用的WQE类型, 告警后报错
            HCCL_ERROR("[AicpuTaskCacheEntry][UpdateWqeAddrRefreshInfo_] unexpected wqeCode[%u]", wqeCode);
            return HCCL_E_INTERNAL;
    }

    CHK_RET(UpdateTokenFlagsByAddrRefreshInfo_(locAddrRefreshInfo, tokenInfos, true));
    CHK_RET(UpdateTokenFlagsByAddrRefreshInfo_(rmtAddrRefreshInfo, tokenInfos, false));

    return HCCL_SUCCESS;
}

inline HcclResult AicpuTaskCacheEntry::UpdateTokenFlagsByAddrRefreshInfo_(
    const AddrRefreshInfo& addrRefreshInfo, vector<TokenInfo>& tokenInfos, bool isLoc) const
{
    if (!addrRefreshInfo.needRefresh) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(
        addrRefreshInfo.memIdx >= tokenInfos.size(),
        HCCL_ERROR(
            "[AicpuTaskCacheEntry][UpdateWqeAddrRefreshInfo_] %s memIdx[%u] >= tokenInfos.size[%u]",
            isLoc ? "loc" : "rmt", addrRefreshInfo.memIdx, tokenInfos.size()),
        HCCL_E_INTERNAL);
    if (isLoc) {
        tokenInfos[addrRefreshInfo.memIdx].needLocTokenIdFlag = true;
    } else {
        tokenInfos[addrRefreshInfo.memIdx].needRmtTokenIdAndValueFlag = true;
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuTaskCacheEntry::UpdateVAddrRankId_(AddrRefreshInfo& addrRefreshInfo) const
{
    if (!addrRefreshInfo.needRefresh) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(
        addrRefreshInfo.memIdx >= TASK_CACHE_LOCAL_MEM_COUNT,
        HCCL_WARNING("[AicpuTaskCacheEntry][%s] unsupported V memory index[%u]", __func__, addrRefreshInfo.memIdx),
        HCCL_E_PARA);
    const auto& counts = addrRefreshInfo.memIdx == 0 ? vOpInfo_.sendCounts : vOpInfo_.recvCounts;
    const auto& displs = addrRefreshInfo.memIdx == 0 ? vOpInfo_.sdispls : vOpInfo_.rdispls;
    CHK_PRT_RET(
        counts.empty() || counts.size() != displs.size() || counts.size() > UINT32_MAX,
        HCCL_WARNING(
            "[AicpuTaskCacheEntry][%s] invalid V array sizes: counts[%zu] displs[%zu] memIdx[%u]", __func__,
            counts.size(), displs.size(), addrRefreshInfo.memIdx),
        HCCL_E_PARA);

    bool found = false;
    uint32_t matchedRank = 0;
    for (size_t rank = 0; rank < counts.size(); ++rank) {
        if (counts[rank] == 0) {
            continue;
        }
        CHK_PRT_RET(
            counts[rank] > UINT64_MAX - displs[rank],
            HCCL_WARNING(
                "[AicpuTaskCacheEntry][%s] V range overflows: memIdx[%u] rank[%zu]", __func__, addrRefreshInfo.memIdx,
                rank),
            HCCL_E_PARA);
        if (addrRefreshInfo.offset < displs[rank] || addrRefreshInfo.offset - displs[rank] >= counts[rank]) {
            continue;
        }
        CHK_PRT_RET(
            found,
            HCCL_WARNING(
                "[AicpuTaskCacheEntry][%s] ambiguous V rank: memIdx[%u] offset[%llu] ranks[%u, %zu]", __func__,
                addrRefreshInfo.memIdx, static_cast<unsigned long long>(addrRefreshInfo.offset), matchedRank, rank),
            HCCL_E_PARA);
        found = true;
        matchedRank = static_cast<uint32_t>(rank);
    }
    CHK_PRT_RET(
        !found,
        HCCL_WARNING(
            "[AicpuTaskCacheEntry][%s] no V rank matches: memIdx[%u] offset[%llu]", __func__, addrRefreshInfo.memIdx,
            static_cast<unsigned long long>(addrRefreshInfo.offset)),
        HCCL_E_PARA);
    CHK_PRT_RET(
        addrRefreshInfo.offset != displs[matchedRank],
        HCCL_WARNING(
            "[AicpuTaskCacheEntry][%s] split V copy is unsupported: memIdx[%u] rank[%u] offset[%llu]", __func__,
            addrRefreshInfo.memIdx, matchedRank, static_cast<unsigned long long>(addrRefreshInfo.offset)),
        HCCL_E_PARA);
    addrRefreshInfo.rankId = matchedRank;
    return HCCL_SUCCESS;
}

HcclResult AicpuTaskCacheEntry::UpdateAddrRefreshInfo_(
    const uint64_t addr, AddrRefreshInfo& addrRefreshInfo, const AddrType addrType) const
{
    // 默认不是dynamic memory (e.g., user input/output), 认为无需刷新
    addrRefreshInfo.needRefresh = false;

    // 注意: 假设地址vector中前两个为local地址, 后续为remote地址
    uint32_t startMemIdx = 0;
    uint32_t endMemIdx = 0;
    if (addrType == AddrType::kAddrTypeUnknown) { // 遍历所有地址信息
        startMemIdx = 0;
        endMemIdx = cachedBaseAddrs_.size();
    } else if (addrType == AddrType::kAddrTypeLocal) { // 遍历local地址信息
        startMemIdx = 0;
        endMemIdx = TASK_CACHE_LOCAL_MEM_COUNT;
    } else { // 遍历remote地址信息
        startMemIdx = TASK_CACHE_LOCAL_MEM_COUNT;
        endMemIdx = cachedBaseAddrs_.size();
    }

    // 检查是否为任意一段dynamic memory
    for (uint32_t memIdx = startMemIdx; memIdx < endMemIdx; memIdx++) {
        // Memory range: [baseAddr, baseAddr + memSize)
        const uint64_t baseAddr = cachedBaseAddrs_[memIdx];
        const uint64_t memSize = cachedMemSizes_[memIdx];

        // 在InitCacheEntry已经做了baseAddr的空指针，及baseAddr + memsize的溢出检查。

        // 检查是否在当前dynamic memory range
        if (AicpuTaskCacheEntry::InRange(baseAddr, memSize, addr)) {
            addrRefreshInfo.needRefresh = true;
            addrRefreshInfo.memIdx = memIdx;
            addrRefreshInfo.offset = addr - baseAddr;
            break;
        }
    }

    return HCCL_SUCCESS;
}

inline bool AicpuTaskCacheEntry::InRange(const uint64_t baseAddr, const uint64_t memSize, const uint64_t addr)
{
    return (addr >= baseAddr && addr < baseAddr + memSize);
}

inline HcclResult AicpuTaskCacheEntry::RefreshSqeTasks_(
    const SqeArrayInfo& sqeArrayInfo, const uint64_t* baseAddrs, const VOpInfo& vOpInfo)
{
    // sqe数组
    uint8_t* sqeArrayPtr = sqeArrayInfo.sqeArray; // 注意: sqeArrayPtr已在AddSqeArray校验, 无需再校验
    uint64_t sqeCount = sqeArrayInfo.sqeCount;
    RtsqA5* rtsqA5Ptr = sqeArrayInfo.rtsqPtr; // 注意: rtsqPtr已在AddSqeArray校验, 无需再校验
    const vector<AddrRefreshInfo>& sqeSrcAddrRefreshInfoArray = sqeArrayInfo.srcAddrRefreshInfoArray;
    const vector<AddrRefreshInfo>& sqeDstAddrRefreshInfoArray = sqeArrayInfo.dstAddrRefreshInfoArray;
    if (UNLIKELY(isTaskConfigDebug_)) {
        PLF_CONFIG_INFO(
            PLF_TASK, "[AicpuTaskCacheEntry][RefreshSqeTasks_] dump %llu cached SQEs in stream[%u]", sqeCount,
            rtsqA5Ptr->GetStreamId());
    }
    for (size_t sqeIdx = 0; sqeIdx < sqeCount; ++sqeIdx) {
        // 获取当前SQE的信息
        Rt91095StarsSqeHeader* sqeHeaderPtr = static_cast<Rt91095StarsSqeHeader*>(static_cast<void*>(sqeArrayPtr));

        CHK_RET(RefreshOneSqe_(
            sqeArrayPtr, sqeSrcAddrRefreshInfoArray[sqeIdx], sqeDstAddrRefreshInfoArray[sqeIdx], baseAddrs, vOpInfo));

        // 刷新streamId, taskId
        rtsqA5Ptr->RefreshSqeHeaderTaskField(sqeHeaderPtr);

        // 切换至下一个SQE
        sqeArrayPtr += AC_SQE_SIZE;
    }

    // 按需打印刷新后的SQE内容
    if (UNLIKELY(isTaskConfigDebug_)) {
        // 循环打印刷新后的SQE内容
        sqeArrayPtr = sqeArrayInfo.sqeArray;
        for (size_t sqeIdx = 0; sqeIdx < sqeCount; ++sqeIdx) {
            // 打印当前SQE
            PLF_CONFIG_INFO(
                PLF_TASK, "[AicpuTaskCacheEntry][RefreshSqeTasks_] %uth cached SQE in stream[%u]", sqeIdx,
                rtsqA5Ptr->GetStreamId());
            CHK_RET(AicpuTaskUtils::DumpSqeContent(sqeArrayPtr));

            // 切换至下一个SQE
            sqeArrayPtr += AC_SQE_SIZE;
        }
    }
    return HCCL_SUCCESS;
}

inline HcclResult AicpuTaskCacheEntry::RefreshOneSqe_(
    uint8_t* sqeArrayPtr, const AddrRefreshInfo& srcAddrRefreshInfo, const AddrRefreshInfo& dstAddrRefreshInfo,
    const uint64_t* baseAddrs, const VOpInfo& vOpInfo) const
{
    Rt91095StarsSqeHeader* sqeHeaderPtr = static_cast<Rt91095StarsSqeHeader*>(static_cast<void*>(sqeArrayPtr));
    const Rt91095StarsSqeType sqeType = static_cast<Rt91095StarsSqeType>(sqeHeaderPtr->type);
    bool isPlaceHolder = false;
    // 根据SQE type进行对应刷新 (task id始终要刷新; addr相关字段有条件刷新)
    switch (sqeType) {
        case Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT:
        case Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_RECORD:
        case Rt91095StarsSqeType::RT_91095_SQE_TYPE_UBDMA:
            // 无地址字段, 直接跳过
            break;
        case Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA: {
            Rt91095StarsMemcpySqe* memcpySqePtr = static_cast<Rt91095StarsMemcpySqe*>(static_cast<void*>(sqeArrayPtr));
            // 刷新地址
            if (srcAddrRefreshInfo.needRefresh) {
                RefreshTaskAddr_(
                    memcpySqePtr->u.strideMode0.srcAddrLow, memcpySqePtr->u.strideMode0.srcAddrHigh, srcAddrRefreshInfo,
                    baseAddrs, vOpInfo, isPlaceHolder);
            }
            if (dstAddrRefreshInfo.needRefresh) {
                RefreshTaskAddr_(
                    memcpySqePtr->u.strideMode0.dstAddrLow, memcpySqePtr->u.strideMode0.dstAddrHigh, dstAddrRefreshInfo,
                    baseAddrs, vOpInfo, isPlaceHolder);
            }
            if (isVOp_ && memcpySqePtr->opcode == SDMA_OPCODE_MEMCPY && !isPlaceHolder) {
                // PreCopy 和本卡自拷贝以 input 为源，PostCopy 以 output 为目的；CCL buffer 一端不需要刷新。
                const AddrRefreshInfo* dataAddrInfo = srcAddrRefreshInfo.needRefresh ? &srcAddrRefreshInfo :
                                                      dstAddrRefreshInfo.needRefresh ? &dstAddrRefreshInfo :
                                                                                       nullptr;
                if (dataAddrInfo != nullptr) {
                    const auto& counts = dataAddrInfo->memIdx == 0 ? vOpInfo.sendCounts : vOpInfo.recvCounts;
                    const uint64_t lengthMove = counts[dataAddrInfo->rankId];
                    CHK_PRT_RET(
                        lengthMove > UINT32_MAX,
                        HCCL_ERROR(
                            "[AicpuTaskCacheEntry][%s] V SDMA length exceeds uint32, rank[%u] length[%llu]", __func__,
                            dataAddrInfo->rankId, static_cast<unsigned long long>(lengthMove)),
                        HCCL_E_PARA);
                    memcpySqePtr->u.strideMode0.lengthMove = static_cast<uint32_t>(lengthMove);
                }
            }
            break;
        }
        case Rt91095StarsSqeType::RT_91095_SQE_TYPE_WRITE_VALUE: {
            Rt91095StarsWriteValueSqe* writeValueSqePtr
                = static_cast<Rt91095StarsWriteValueSqe*>(static_cast<void*>(sqeArrayPtr));
            // 刷新地址
            if (dstAddrRefreshInfo.needRefresh) {
                // 注意: writeAddrHigh是位域, 无法直接作为u32&传入
                uint32_t tmpAddrHigh = writeValueSqePtr->writeAddrHigh;
                RefreshTaskAddr_(writeValueSqePtr->writeAddrLow, tmpAddrHigh, dstAddrRefreshInfo, baseAddrs);
                writeValueSqePtr->writeAddrHigh = tmpAddrHigh;
            }
            break;
        }
        default:
            // sqe_build_a5.h中未使用的SQE类型, 告警后报错
            HCCL_ERROR("[AicpuTaskCacheEntry][%s] unexpected sqeType[%u]", __func__, sqeType);
            return HCCL_E_INTERNAL;
    }
    if (isPlaceHolder) {
        sqeHeaderPtr->type = static_cast<uint8_t>(Hccl::Rt91095StarsSqeType::RT_91095_SQE_TYPE_PLACE_HOLDER);
    }
    return HCCL_SUCCESS;
}

inline HcclResult AicpuTaskCacheEntry::LaunchSqeTasks_(const SqeArrayInfo& sqeArrayInfo) const
{
    // 注意: rtsqPtr和sqeArray在AddSqeArray时已校验, 这里无需再校验
    RtsqA5* rtsqA5Ptr = sqeArrayInfo.rtsqPtr;
    CHK_RET(rtsqA5Ptr->LaunchNewTask(sqeArrayInfo.sqeArray, static_cast<u32>(sqeArrayInfo.sqeCount)));
    return HCCL_SUCCESS;
}
// 频繁调用函数，参数外层已经校验，不会失败，无需返回HcclResult让调用方判断。
inline void AicpuTaskCacheEntry::RefreshTaskAddr_(
    uint32_t& addrLow, uint32_t& addrHigh, const AddrRefreshInfo& addrRefreshInfo, const uint64_t* baseAddrs,
    const VOpInfo& vOpInfo, bool& isPlaceHolder) const
{
    // 注意: 调用RefreshTaskAddr_时已判断addrRefreshInfo.needRefresh, 无需再次校验

    // 注意: memIdx已经在SubmitCacheEntry的UpdateAddrRefreshInfo_中校验, 确保在cachedMemSizes_范围内

    // 计算newAddr
    uint64_t newAddr = baseAddrs[addrRefreshInfo.memIdx] + addrRefreshInfo.offset;
    if (isVOp_) {
        // count 判断是否需要占位符；地址使用本次调用的字节位移，不能把位移当作 count。
        const auto& counts = addrRefreshInfo.memIdx == 0 ? vOpInfo.sendCounts : vOpInfo.recvCounts;
        const auto& displs = addrRefreshInfo.memIdx == 0 ? vOpInfo.sdispls : vOpInfo.rdispls;
        if (counts[addrRefreshInfo.rankId] > 0) {
            newAddr = baseAddrs[addrRefreshInfo.memIdx] + displs[addrRefreshInfo.rankId];
        } else {
            isPlaceHolder = true;
        }
    }
    // 返回新地址
    AicpuTaskCacheEntry::SplitUint64ToUint32(newAddr, addrHigh, addrLow);
}
// 频繁调用函数，参数外层已经校验，不会失败，无需返回HcclResult让调用方判断。
inline void AicpuTaskCacheEntry::RefreshTaskAddr_(
    uint32_t& addrLow, uint32_t& addrHigh, const AddrRefreshInfo& addrRefreshInfo, const uint64_t* baseAddrs) const
{
    // 注意: 调用RefreshTaskAddr_时已判断addrRefreshInfo.needRefresh, 无需再次校验

    // 注意: memIdx已经在SubmitCacheEntry的UpdateAddrRefreshInfo_中校验, 确保在cachedMemSizes_范围内

    // 计算newAddr
    const uint64_t newAddr = baseAddrs[addrRefreshInfo.memIdx] + addrRefreshInfo.offset;

    // 返回新地址
    AicpuTaskCacheEntry::SplitUint64ToUint32(newAddr, addrHigh, addrLow);
}

// 频繁调用函数，参数外层已经校验，不会失败，无需返回HcclResult让上层判断。
inline void AicpuTaskCacheEntry::RefreshTaskAddr_(
    uint64_t& addr, const AddrRefreshInfo& addrRefreshInfo, const uint64_t* baseAddrs) const
{
    // 注意: 调用RefreshTaskAddr_时已判断addrRefreshInfo.needRefresh, 无需再次校验

    // 注意: memIdx已经在SubmitCacheEntry的UpdateAddrRefreshInfo_中校验, 确保在cachedMemSizes_范围内

    // 计算newAddr, 返回新地址
    addr = baseAddrs[addrRefreshInfo.memIdx] + addrRefreshInfo.offset;
}

inline void AicpuTaskCacheEntry::DumpWqeTasksHeader_(uint64_t wqeCount, const UbConnLite* ubConnLitePtr) const
{
    PLF_CONFIG_INFO(
        PLF_TASK, "[AicpuTaskCacheEntry][RefreshWqeTasks_] dump %llu cached WQEs in jetty[%u, %u, %u]", wqeCount,
        ubConnLitePtr->GetUbJettyLiteId().GetDieId(), ubConnLitePtr->GetUbJettyLiteId().GetFuncId(),
        ubConnLitePtr->GetUbJettyLiteId().GetJettyId());
}

inline HcclResult
AicpuTaskCacheEntry::DumpWqeTasksPerWqe_(size_t wqeIdx, const WqeTask& wqeTask, const UbConnLite* ubConnLitePtr) const
{
    PLF_CONFIG_INFO(
        PLF_TASK, "[AicpuTaskCacheEntry][%s] %uth cached WQE in jetty[%u, %u, %u]", __func__, wqeIdx,
        ubConnLitePtr->GetUbJettyLiteId().GetDieId(), ubConnLitePtr->GetUbJettyLiteId().GetFuncId(),
        ubConnLitePtr->GetUbJettyLiteId().GetJettyId());
    CHK_RET(AicpuTaskUtils::DumpWqeContent(ReinterpretAs<const uint8_t*>(&wqeTask)));
    return HCCL_SUCCESS;
}

inline HcclResult AicpuTaskCacheEntry::RefreshWqeTasks_(
    WqeTaskArrayInfo& wqeTaskArrayInfo, const uint64_t* baseAddrs, [[maybe_unused]] const uint64_t* memSizes,
    [[maybe_unused]] const uint32_t count, const VOpInfo& vOpInfo)
{
    // 逐个刷新地址；V 类还要根据当前 rank 的数据量刷新 SGE 与 opcode。
    vector<WqeTask>& wqeTasks = wqeTaskArrayInfo.wqeTaskArray;
    const vector<AddrRefreshInfo>& wqeLocAddrRefreshInfoArray = wqeTaskArrayInfo.locAddrRefreshInfoArray;
    const vector<AddrRefreshInfo>& wqeRmtAddrRefreshInfoArray = wqeTaskArrayInfo.rmtAddrRefreshInfoArray;
    const vector<TokenInfo>* tokenInfos = nullptr;
    if (!isVOp_) {
        auto iter = tokenInfosMap_.find(wqeTaskArrayInfo.ubTransportLiteImplPtr);
        CHK_PRT_RET(
            iter == tokenInfosMap_.end(),
            HCCL_ERROR("[AicpuTaskCacheEntry][RefreshWqeTasks_] transport token info missing"), HCCL_E_INTERNAL);
        tokenInfos = &iter->second;
    }
    const uint64_t wqeCount = wqeTasks.size();
    if (UNLIKELY(isTaskConfigDebug_)) {
        DumpWqeTasksHeader_(wqeCount, wqeTaskArrayInfo.ubConnLitePtr);
    }
    for (size_t wqeIdx = 0; wqeIdx < wqeCount; wqeIdx++) {
        WqeTask& wqeTask = wqeTasks[wqeIdx];
        const AddrRefreshInfo& locAddrRefreshInfo = wqeLocAddrRefreshInfoArray[wqeIdx];
        const AddrRefreshInfo& rmtAddrRefreshInfo = wqeRmtAddrRefreshInfoArray[wqeIdx];

        // 每次从录制值恢复，避免上一次命中改成 NOP 后无法恢复成 READ/WRITE。
        UdmaSqeCommon* wqeCommonPtr = static_cast<UdmaSqeCommon*>(static_cast<void*>(&wqeTask));
        if (isVOp_) {
            wqeCommonPtr->opcode = wqeTaskArrayInfo.originalOpcodes[wqeIdx];
            wqeCommonPtr->sgeNum = wqeTaskArrayInfo.originalSgeNums[wqeIdx];
        }
        const uint8_t wqeCode
            = static_cast<uint8_t>(wqeCommonPtr->opcode); // opcode来自于UdmaSqOpcode, 一定在uint8范围内

        if (isVOp_ && !(wqeCode == UdmaSqOpcode::UDMA_OPC_WRITE && wqeCommonPtr->inlineEn != 0)) {
            const uint32_t rankId = locAddrRefreshInfo.rankId;
            const bool isRead = wqeCode == UdmaSqOpcode::UDMA_OPC_READ;
            // 同一 entry 甚至同一 WQE 数组可混合 READ/WRITE，方向由各 WQE 的 opcode 决定。
            const auto& oldCounts = isRead ? vOpInfo_.recvCounts : vOpInfo_.sendCounts;
            const auto& newCounts = isRead ? vOpInfo.recvCounts : vOpInfo.sendCounts;
            CHK_PRT_RET(
                rankId >= oldCounts.size() || rankId >= newCounts.size() || newCounts[rankId] > UINT32_MAX,
                HCCL_ERROR("[AicpuTaskCacheEntry][%s] invalid V WQE count, rank[%u]", __func__, rankId), HCCL_E_PARA);
            const bool withNotify = wqeCode == WRITE_WITH_NOTIFY_OPCODE;

            // 带 notify 的 WRITE 必须保留两个 WQEBB 和远端通知；仅清空其数据 SGE。
            if (withNotify) {
                wqeTask.wqeWriteWithNotify.localU.sge.length = static_cast<uint32_t>(newCounts[rankId]);
            } else {
                wqeTask.wqeWrite.u.sge.length = static_cast<uint32_t>(newCounts[rankId]);
                if (newCounts[rankId] == 0) {
                    wqeCommonPtr->opcode = UdmaSqOpcode::UDMA_OPC_NOP;
                }
            }
            if (newCounts[rankId] == 0) {
                wqeCommonPtr->sgeNum = 0;
                continue;
            }
        }

        // V 类保留录制 token；非 V 类刷新地址和对应 token。
        switch (wqeCode) {
            case UdmaSqOpcode::UDMA_OPC_READ: // UdmaSqeWrite
                // normal read
                CHK_RET(RefreshWqeRead_(wqeTask, locAddrRefreshInfo, rmtAddrRefreshInfo, baseAddrs));
                break;
            case UdmaSqOpcode::UDMA_OPC_WRITE: { // UdmaSqeWrite
                CHK_RET(RefreshWqeWrite_(wqeTask, locAddrRefreshInfo, rmtAddrRefreshInfo, baseAddrs));
                break;
            }
            case WRITE_WITH_NOTIFY_OPCODE: // UdmaSqeWriteWithNotify
                // write with notify (对于给定slice的最后一个UB chunk)
                CHK_RET(RefreshWqeWriteWithNotify_(wqeTask, locAddrRefreshInfo, rmtAddrRefreshInfo, baseAddrs));
                break;
            default:
                // ub_conn_lite.cc中未使用的WQE类型, 告警后报错
                HCCL_ERROR("[AicpuTaskCacheEntry][%s] unexpected wqeCode[%u]", __func__, wqeCode);
                return HCCL_E_INTERNAL;
        }
        if (!isVOp_) {
            const bool inlineWrite = wqeCode == UdmaSqOpcode::UDMA_OPC_WRITE && wqeCommonPtr->inlineEn != 0;
            if (locAddrRefreshInfo.needRefresh && !inlineWrite) {
                uint32_t& locTokenId = wqeCode == WRITE_WITH_NOTIFY_OPCODE ?
                                           wqeTask.wqeWriteWithNotify.localU.sge.tokenId :
                                           wqeTask.wqeWrite.u.sge.tokenId;
                CHK_RET(RefreshWqeLocTokenId_(locTokenId, locAddrRefreshInfo, *tokenInfos));
            }
            if (rmtAddrRefreshInfo.needRefresh && !inlineWrite) {
                uint32_t tokenId = wqeCode == WRITE_WITH_NOTIFY_OPCODE ? wqeTask.wqeWriteWithNotify.comm.rmtObjId :
                                                                         wqeTask.wqeWrite.comm.rmtObjId;
                uint32_t& tokenValue = wqeCode == WRITE_WITH_NOTIFY_OPCODE ?
                                           wqeTask.wqeWriteWithNotify.comm.rmtTokenValue :
                                           wqeTask.wqeWrite.comm.rmtTokenValue;
                CHK_RET(RefreshWqeRmtTokenIdAndValue_(tokenId, tokenValue, rmtAddrRefreshInfo, *tokenInfos));
                if (wqeCode == WRITE_WITH_NOTIFY_OPCODE) {
                    wqeTask.wqeWriteWithNotify.comm.rmtObjId = tokenId;
                } else {
                    wqeTask.wqeWrite.comm.rmtObjId = tokenId;
                }
            }
        }
        if (isVOp_ && locAddrRefreshInfo.needRefresh && locAddrRefreshInfo.memIdx < TASK_CACHE_LOCAL_MEM_COUNT
            && wqeCommonPtr->inlineEn == 0) {
            const uint32_t rankId = locAddrRefreshInfo.rankId;
            const bool isInput = locAddrRefreshInfo.memIdx == 0;
            const auto& oldDispls = isInput ? vOpInfo_.sdispls : vOpInfo_.rdispls;
            const auto& newDispls = isInput ? vOpInfo.sdispls : vOpInfo.rdispls;
            CHK_PRT_RET(
                rankId >= oldDispls.size() || rankId >= newDispls.size()
                    || locAddrRefreshInfo.offset != oldDispls[rankId]
                    || newDispls[rankId] > UINT64_MAX - baseAddrs[locAddrRefreshInfo.memIdx],
                HCCL_ERROR("[AicpuTaskCacheEntry][%s] invalid V WQE address, rank[%u]", __func__, rankId), HCCL_E_PARA);
            const uint64_t newAddr = baseAddrs[locAddrRefreshInfo.memIdx] + newDispls[rankId];
            if (wqeCode == WRITE_WITH_NOTIFY_OPCODE) {
                SplitUint64ToUint32(
                    newAddr, wqeTask.wqeWriteWithNotify.localU.sge.dataAddrHigh,
                    wqeTask.wqeWriteWithNotify.localU.sge.dataAddrLow);
            } else {
                SplitUint64ToUint32(newAddr, wqeTask.wqeWrite.u.sge.dataAddrHigh, wqeTask.wqeWrite.u.sge.dataAddrLow);
            }
        }
        if (UNLIKELY(isTaskConfigDebug_)) {
            CHK_RET(DumpWqeTasksPerWqe_(wqeIdx, wqeTask, wqeTaskArrayInfo.ubConnLitePtr));
        }
    }
    return HCCL_SUCCESS;
}

inline HcclResult AicpuTaskCacheEntry::RefreshWqeRead_(
    WqeTask& wqeTask, const AddrRefreshInfo& locAddrRefreshInfo, const AddrRefreshInfo& rmtAddrRefreshInfo,
    const uint64_t* baseAddrs) const
{
    // 如果需要刷新loc地址信息
    if (locAddrRefreshInfo.needRefresh) {
        // 刷新loc addr
        RefreshTaskAddr_(
            wqeTask.wqeWrite.u.sge.dataAddrLow, wqeTask.wqeWrite.u.sge.dataAddrHigh, locAddrRefreshInfo, baseAddrs);
    }
    // 如果需要刷新rmt地址信息
    if (rmtAddrRefreshInfo.needRefresh) {
        // 刷新rmt addr
        RefreshTaskAddr_(
            wqeTask.wqeWrite.comm.rmtAddrLow, wqeTask.wqeWrite.comm.rmtAddrHigh, rmtAddrRefreshInfo, baseAddrs);
    }
    return HCCL_SUCCESS;
}

inline HcclResult AicpuTaskCacheEntry::RefreshWqeWrite_(
    WqeTask& wqeTask, const AddrRefreshInfo& locAddrRefreshInfo, const AddrRefreshInfo& rmtAddrRefreshInfo,
    const uint64_t* baseAddrs)
{
    const uint32_t inlineEn = wqeTask.wqeWrite.comm.inlineEn;
    if (inlineEn != 0) { // inline write
        // 注意: inline write使用WriteWqe实现notify功能 (类似A3使用WriteValue实现notify功能)
        // 因为notify token id/value以及notify addr不会改变, 因此inline write无需刷新WQE
        if (UNLIKELY(rmtAddrRefreshInfo.needRefresh)) {
            // 打印告警信息
            HCCL_ERROR(
                "[AicpuTaskCacheEntry][RefreshWqeTasks_] inline write should not refresh rmt addr: "
                "rmtAddrRefreshInfo.needRefresh[%d] rmtAddrRefreshInfo.memIdx[%u]",
                rmtAddrRefreshInfo.needRefresh, rmtAddrRefreshInfo.memIdx);

            // 打印地址信息
            // 注意: memIdx在SubmitCacheEntry中已校验, 这里直接使用
            uint64_t rmtAddr = 0;
            AicpuTaskCacheEntry::CombineUint32ToUint64(
                rmtAddr, wqeTask.wqeWrite.comm.rmtAddrHigh, wqeTask.wqeWrite.comm.rmtAddrLow);
            const uint64_t cachedBaseAddr = cachedBaseAddrs_[rmtAddrRefreshInfo.memIdx];
            const uint64_t cachedMemSize = cachedMemSizes_[rmtAddrRefreshInfo.memIdx];
            HCCL_ERROR(
                "[AicpuTaskCacheEntry][RefreshWqeTasks_] rmtAddr[0x%llx] "
                "cachedBaseAddr[0x%llx] endAddr[0x%llx] memSize[%llu]",
                rmtAddr, cachedBaseAddr, cachedBaseAddr + cachedMemSize, cachedMemSize);

            return HCCL_E_INTERNAL;
        }
        return HCCL_SUCCESS;
    }
    // normal write or write reduce
    return RefreshWqeRead_(wqeTask, locAddrRefreshInfo, rmtAddrRefreshInfo, baseAddrs);
}

inline HcclResult AicpuTaskCacheEntry::RefreshWqeWriteWithNotify_(
    WqeTask& wqeTask, const AddrRefreshInfo& locAddrRefreshInfo, const AddrRefreshInfo& rmtAddrRefreshInfo,
    const uint64_t* baseAddrs) const
{
    // 如果需要刷新loc地址信息
    if (locAddrRefreshInfo.needRefresh) {
        // 刷新loc addr
        RefreshTaskAddr_(
            wqeTask.wqeWriteWithNotify.localU.sge.dataAddrLow, wqeTask.wqeWriteWithNotify.localU.sge.dataAddrHigh,
            locAddrRefreshInfo, baseAddrs);
    }
    // 如果需要刷新rmt地址信息
    if (rmtAddrRefreshInfo.needRefresh) {
        // 刷新rmt addr
        RefreshTaskAddr_(
            wqeTask.wqeWriteWithNotify.comm.rmtAddrLow, wqeTask.wqeWriteWithNotify.comm.rmtAddrHigh, rmtAddrRefreshInfo,
            baseAddrs);
    }
    return HCCL_SUCCESS;
}

inline HcclResult AicpuTaskCacheEntry::RefreshWqeLocTokenId_(
    uint32_t& tokenId, const AddrRefreshInfo& addrRefreshInfo, const vector<TokenInfo>& tokenInfos) const
{
    // 调用方保证addrRefreshInfo.needRefresh为true

    // 当前UbTransportLiteImpl在对应内存上, 一定已经提前获取了locTokenId
    // 注意: ubTransportLitePtr已在AddWqeArray时校验非空, 这里无需再校验
    const TokenInfo& tokenInfo = tokenInfos[addrRefreshInfo.memIdx];

    CHK_PRT_RET(
        !tokenInfo.needLocTokenIdFlag,
        HCCL_ERROR("[AicpuTaskCacheEntry][RefreshWqeLocTokenId_] needLocTokenIdFlag is false"), HCCL_E_INTERNAL);

    // 刷新loc token id
    tokenId = tokenInfo.locTokenId;

    return HCCL_SUCCESS;
}

inline HcclResult AicpuTaskCacheEntry::RefreshWqeRmtTokenIdAndValue_(
    uint32_t& tokenId, uint32_t& tokenValue, const AddrRefreshInfo& addrRefreshInfo,
    const vector<TokenInfo>& tokenInfos) const
{
    // 调用方保证addrRefreshInfo.needRefresh为true

    // 当前UbTransportLiteImpl在对应内存上, 一定已经提前获取了rmtTokenId/Value
    const TokenInfo& tokenInfo = tokenInfos[addrRefreshInfo.memIdx];

    CHK_PRT_RET(
        !tokenInfo.needRmtTokenIdAndValueFlag,
        HCCL_ERROR("[AicpuTaskCacheEntry][RefreshWqeRmtTokenIdAndValue_] needRmtTokenIdAndValueFlag is false"),
        HCCL_E_INTERNAL);

    // 刷新remote token id/value
    tokenId = tokenInfo.rmtTokenId;
    tokenValue = tokenInfo.rmtTokenValue;

    return HCCL_SUCCESS;
}

inline HcclResult AicpuTaskCacheEntry::LaunchWqeTasks_(WqeTaskArrayInfo& wqeTaskArrayInfo) const
{
    // 逐个下发WQE
    vector<WqeTask>& wqeTasks = wqeTaskArrayInfo.wqeTaskArray;
    UbConnLite* ubConnLitePtr = wqeTaskArrayInfo.ubConnLitePtr; // 注意: AddWqeArray时已校验非空
    const size_t wqeCount = wqeTasks.size();
    for (size_t wqeIdx = 0; wqeIdx < wqeCount; wqeIdx++) {
        WqeTask& wqeTask = wqeTasks[wqeIdx];

        // 根据WQE类型下发 (下发过程中会更新ubConnLitePtr中的pi)
        UdmaSqeCommon* wqeCommonPtr = static_cast<UdmaSqeCommon*>(static_cast<void*>(&wqeTask));
        const uint8_t wqeCode
            = static_cast<uint8_t>(wqeCommonPtr->opcode); // opcode来自于UdmaSqOpcode, 一定在uint8范围内
        switch (wqeCode) {
            case UdmaSqOpcode::UDMA_OPC_READ: // UdmaSqeWrite
                // normal read
                ubConnLitePtr->LaunchOneWqe(&wqeTask.wqeWrite, UdmaSqOpcode::UDMA_OPC_READ);
                break;
            case UdmaSqOpcode::UDMA_OPC_WRITE: // UdmaSqeWrite
                // inline write, normal write, or write reduce
                ubConnLitePtr->LaunchOneWqe(&wqeTask.wqeWrite, UdmaSqOpcode::UDMA_OPC_WRITE);
                break;
            case WRITE_WITH_NOTIFY_OPCODE: // UdmaSqeWriteWithNotify
                // write with notify (对于给定slice的最后一个UB chunk)
                ubConnLitePtr->LaunchOneWqeWithNotify(&wqeTask.wqeWriteWithNotify, WRITE_WITH_NOTIFY_OPCODE);
                break;
            case UdmaSqOpcode::UDMA_OPC_NOP:
                ubConnLitePtr->LaunchOneWqe(&wqeTask.wqeWrite, UdmaSqOpcode::UDMA_OPC_NOP);
                break;
            default:
                // ub_conn_lite.cc中未使用的WQE类型, 告警后报错
                HCCL_ERROR("[AicpuTaskCacheEntry][LaunchWqeTasks_] unexpected wqeCode[%u]", wqeCode);
                return HCCL_E_INTERNAL;
        }
    }

    return HCCL_SUCCESS;
}

inline HcclResult AicpuTaskCacheEntry::RefreshDbSqe_(WqeTaskArrayInfo& wqeTaskArrayInfo)
{
    // 获取pi和seq
    UbConnLite* ubConnLitePtr = wqeTaskArrayInfo.ubConnLitePtr; // 注意: AddWqeArray时已校验非空
    u16 seq;
    const uint16_t pi = ubConnLitePtr->GetPiAndIncrementSeq(seq);

    // 校验dbSqeLocation, AddSqeArray_中已经校验
    const DbSqeLocation& dbSqeLocation = wqeTaskArrayInfo.dbSqeLocation;

    // 校验SQE type
    uint8_t* sqePtr = sqeArrayInfos_[dbSqeLocation.sqeArrayIdx].sqeArray + dbSqeLocation.dbSqeIdx * AC_SQE_SIZE;

    // 更新SQE pi value
    Rt91095StarsUbdmaDBmodeSqe* dbSqePtr = static_cast<Rt91095StarsUbdmaDBmodeSqe*>(static_cast<void*>(sqePtr));
    dbSqePtr->piValue1 = pi;

    // 按需记录DbSendSlotMeta, 用于ciTracker的CI同步 (参考RtsqA5::UbDbSend中的dbSendSlots_记录逻辑)
    UbTransportLiteImpl* ubTransportLiteImplPtr = wqeTaskArrayInfo.ubTransportLiteImplPtr;
    RtsqA5* rtsqA5Ptr = sqeArrayInfos_[dbSqeLocation.sqeArrayIdx].rtsqPtr;
    rtsqA5Ptr->RecordDbSendSlot(ubTransportLiteImplPtr, dbSqeLocation.dbSqeIdx, seq, pi);

    // 注意: UbTransportLiteImpl只针对WQE按需填充DfxTaskInfo上报profiling, DB SQE无需上报profiling

    HCCL_INFO(
        "[AicpuTaskCacheEntry][RefreshDbSqe_] update pi[%u] at dbSqeLocation[%u, %u]", pi, dbSqeLocation.sqeArrayIdx,
        dbSqeLocation.dbSqeIdx);

    return HCCL_SUCCESS;
}

HcclResult AicpuTaskCacheEntry::ReportDbSqeProfiling_(
    uint8_t* dbSqePtr, size_t arrayIdx, uint32_t dbSqeIdx, const uint64_t* baseAddrs, const uint64_t* memSizes,
    const uint32_t count, StreamLite* streamLite, const u32 sqId, const u32 taskId)
{
    // 注意: 参考ub_transport_lite_impl.cc填充DfxTaskInfo并经NextTaskSlot上报
    DbSqeLocation dbSqeLocation;
    dbSqeLocation.sqeArrayIdx = arrayIdx;
    dbSqeLocation.dbSqeIdx = dbSqeIdx;
    auto it = dbSqeLocInfoMap_.find(dbSqeLocation);

    // 注意: 少数场景下DbSqe不需要上报, 例如Drain, BatchOneSidedRead, BatchOneSidedWrite
    if (UNLIKELY(it == dbSqeLocInfoMap_.end())) {
        HCCL_INFO(
            "[AicpuTaskCacheEntry][ReportDbSqeProfiling_] dbSqeLocInfoMap_[%u, %u] is not found",
            dbSqeLocation.sqeArrayIdx, dbSqeLocation.dbSqeIdx);
        return HCCL_SUCCESS;
    }

    CHK_PRT_RET(
        it->second.dbSqeProfInfo.isValid == false,
        HCCL_ERROR(
            "[AicpuTaskCacheEntry][ReportDbSqeProfiling_] dbSqeLocInfoMap_[%u, %u] is invalid",
            dbSqeLocation.sqeArrayIdx, dbSqeLocation.dbSqeIdx),
        HCCL_E_INTERNAL);

    // 校验SQE type
    Rt91095StarsSqeHeader* sqeHeaderPtr
        = static_cast<Rt91095StarsSqeHeader*>(static_cast<void*>(dbSqePtr)); // 已在AddSqeArray校验, 无需再校验
    CHK_PRT_RET(
        static_cast<Rt91095StarsSqeType>(sqeHeaderPtr->type) != Rt91095StarsSqeType::RT_91095_SQE_TYPE_UBDMA,
        HCCL_ERROR(
            "[AicpuTaskCacheEntry][ReportDbSqeProfiling_] sqeHeaderPtr->type[%u] is not RT_91095_SQE_TYPE_UBDMA",
            sqeHeaderPtr->type),
        HCCL_E_INTERNAL);

    HCCL_INFO(
        "[AicpuTaskCacheEntry][ReportDbSqeProfiling_] report %lluth DbSqe profiling in %lluth SQE array: "
        "dbSqeType[%u], taskId[%u], sqId[%u]",
        dbSqeIdx, arrayIdx, sqeHeaderPtr->type, taskId, sqId);

    // 每次上报只刷新临时副本，保留录制时的地址、长度和刷新元数据。
    DbSqeProfAndRefreshInfo profAndRefreshInfo = it->second;
    if (isVOp_
        && static_cast<u8>(profAndRefreshInfo.dbSqeProfInfo.taskParamType) != TaskParamTypeVal::TASK_UB_INLINE_WRITE) {
        CHK_RET(RefreshVOpDbSqeProfInfo_(profAndRefreshInfo));
    } else {
        CHK_RET(RefreshDbSqeProfAddrs_(profAndRefreshInfo, baseAddrs, memSizes, count));
    }

    // 从DbSqeProfAndRefreshInfo中获取DbSqe对应的UbTransportLiteImpl
    // 注意: wqeArrayIdx已在AddSqeArray_校验, 无需再校验
    const uint32_t wqeArrayIdx = it->second.wqeArrayIdx;
    // 注意: ubTransportLiteImplPtr已在AddWqeArray校验, 无需再校验
    UbTransportLiteImpl* ubTransportLitePtr = wqeTaskArrayInfos_[wqeArrayIdx].ubTransportLiteImplPtr;

    Hccl::DfxTaskInfo* slot = streamLite->NextTaskSlot();
    switch (static_cast<u8>(profAndRefreshInfo.dbSqeProfInfo.taskParamType)) {
        case TaskParamTypeVal::TASK_UB_INLINE_WRITE:
        case TaskParamTypeVal::TASK_UB:
        case TaskParamTypeVal::TASK_WRITE_WITH_NOTIFY:
            CHK_RET(FillSlotUbDma_(slot, profAndRefreshInfo, ubTransportLitePtr, streamLite, taskId));
            break;
        case TaskParamTypeVal::TASK_UB_REDUCE_INLINE:
        case TaskParamTypeVal::TASK_WRITE_REDUCE_WITH_NOTIFY:
            CHK_RET(FillSlotReduce_(slot, profAndRefreshInfo, ubTransportLitePtr, streamLite, taskId));
            break;
        default:
            HCCL_ERROR(
                "[AicpuTaskCacheEntry][ReportDbSqeProfiling_] taskType[%u] is unsupported",
                it->second.dbSqeProfInfo.taskParamType);
            return HCCL_E_INTERNAL;
    }

    return HCCL_SUCCESS;
}

inline HcclResult AicpuTaskCacheEntry::RefreshVOpDbSqeProfInfo_(DbSqeProfAndRefreshInfo& profAndRefreshInfo) const
{
    // WQE 已按本次 counts/displs 刷新；这里只读取实际下发的数据任务，不重复计算地址。
    // wqeArrayIdx 和 originalOpcodes 的范围已在缓存提交及回放入口校验。
    const auto& arrayInfo = wqeTaskArrayInfos_[profAndRefreshInfo.wqeArrayIdx];
    DbSqeProfInfo& profInfo = profAndRefreshInfo.dbSqeProfInfo;
    profInfo.size = 0;
    profInfo.locAddr = 0;
    profInfo.rmtAddr = 0;
    for (size_t wqeIdx = 0; wqeIdx < arrayInfo.wqeTaskArray.size(); ++wqeIdx) {
        const WqeTask& task = arrayInfo.wqeTaskArray[wqeIdx];
        const uint8_t opcode = arrayInfo.originalOpcodes[wqeIdx];
        // inline WRITE 是通知等控制任务，不计入用户数据量。
        if (opcode == UdmaSqOpcode::UDMA_OPC_WRITE && task.wqeWrite.comm.inlineEn != 0) {
            continue;
        }
        const bool withNotify = opcode == WRITE_WITH_NOTIFY_OPCODE;
        CHK_PRT_RET(
            opcode != UdmaSqOpcode::UDMA_OPC_READ && opcode != UdmaSqOpcode::UDMA_OPC_WRITE && !withNotify,
            HCCL_ERROR("[AicpuTaskCacheEntry][%s] invalid V WQE opcode[%u]", __func__, opcode), HCCL_E_INTERNAL);
        const auto& sge = withNotify ? task.wqeWriteWithNotify.localU.sge : task.wqeWrite.u.sge;
        // 零数据 WQE 可能保留旧地址；只报告 0 字节，不读取这些地址，保留模板中的通知字段。
        if (sge.length == 0) {
            continue;
        }
        CHK_PRT_RET(
            sge.length > UINT64_MAX - profInfo.size,
            HCCL_ERROR("[AicpuTaskCacheEntry][%s] V profiling size overflow", __func__), HCCL_E_PARA);
        profInfo.size += sge.length;
        CombineUint32ToUint64(profInfo.locAddr, sge.dataAddrHigh, sge.dataAddrLow);
        if (withNotify) {
            CombineUint32ToUint64(
                profInfo.rmtAddr, task.wqeWriteWithNotify.comm.rmtAddrHigh, task.wqeWriteWithNotify.comm.rmtAddrLow);
        } else {
            CombineUint32ToUint64(profInfo.rmtAddr, task.wqeWrite.comm.rmtAddrHigh, task.wqeWrite.comm.rmtAddrLow);
        }
    }
    return HCCL_SUCCESS;
}

inline HcclResult AicpuTaskCacheEntry::RefreshDbSqeProfAddrs_(
    DbSqeProfAndRefreshInfo& profAndRefreshInfo, const uint64_t* baseAddrs, [[maybe_unused]] const uint64_t* memSizes,
    [[maybe_unused]] const uint32_t count) const
{
    // 注意: dbSqeProfInfo中的src/dstAddr, 需要根据DbSqeProfAndRefreshInfo中的src/dstAddrRefreshInfo进行刷新,
    // 才能填充DfxTaskInfo
    if (profAndRefreshInfo.locAddrRefreshInfo.needRefresh) {
        RefreshTaskAddr_(profAndRefreshInfo.dbSqeProfInfo.locAddr, profAndRefreshInfo.locAddrRefreshInfo, baseAddrs);
    }
    if (profAndRefreshInfo.rmtAddrRefreshInfo.needRefresh) {
        RefreshTaskAddr_(profAndRefreshInfo.dbSqeProfInfo.rmtAddr, profAndRefreshInfo.rmtAddrRefreshInfo, baseAddrs);
    }
    return HCCL_SUCCESS;
}

inline void AicpuTaskCacheEntry::FillSlotCommonFields_(
    Hccl::DfxTaskInfo* slot, const StreamLite* streamLite, u32 taskId, u8 linkType, u8 transportType,
    u64 channelHandle) const
{
    slot->sqId = streamLite->GetSqId();
    slot->taskId = taskId;
    const void* opInfo = streamLite->GetLatestDfxOpInfo();
    slot->dfxOpInfo = (opInfo != nullptr) ? ReinterpretAs<u64>(opInfo) : DFX_INVALID_U64;
    slot->linkType = linkType;
    slot->transportType = transportType;
    slot->channelHandle = channelHandle;
}

inline u8 AicpuTaskCacheEntry::GetUbLinkTypeVal_(const UbTransportLiteImpl* ubTransportLiteImplPtr) const
{
    return (ubTransportLiteImplPtr->linkType_ == Hccl::DfxLinkType::UB) ? Hccl::DfxLinkTypeVal::LINK_UB :
                                                                          Hccl::DfxLinkTypeVal::LINK_UBoE;
}

inline u8 AicpuTaskCacheEntry::ConvertSdmaOpCodeToReduceOp_(uint8_t opcode) const
{
    // opcode 低4位为 RtStarsMemcpyAsyncOperationKind: ADD=0x01, MAX=0x02, MIN=0x03
    // 映射到 HcclReduceOp: SUM=0, MAX=2, MIN=3
    const uint8_t opKind = opcode & 0x0F;
    switch (opKind) {
        case 0x01:
            return static_cast<u8>(HCCL_REDUCE_SUM);
        case 0x02:
            return static_cast<u8>(HCCL_REDUCE_MAX);
        case 0x03:
            return static_cast<u8>(HCCL_REDUCE_MIN);
        default:
            return static_cast<u8>(HCCL_REDUCE_RESERVED);
    }
}

inline HcclResult AicpuTaskCacheEntry::FillSlotUbDma_(
    Hccl::DfxTaskInfo* slot, const DbSqeProfAndRefreshInfo& profAndRefreshInfo,
    UbTransportLiteImpl* ubTransportLiteImplPtr, StreamLite* streamLite, u32 taskId) const
{
    const DbSqeProfInfo& profInfo = profAndRefreshInfo.dbSqeProfInfo;
    slot->taskType = static_cast<u8>(profInfo.taskParamType);
    FillSlotCommonFields_(
        slot, streamLite, taskId, GetUbLinkTypeVal_(ubTransportLiteImplPtr),
        static_cast<u8>(Hccl::DfxTransportType::DFX_TRANSPORT_TYPE_UB), ReinterpretAs<u64>(ubTransportLiteImplPtr));
    slot->taskPara.ubDma.srcAddr = profInfo.locAddr;
    slot->taskPara.ubDma.dstAddr = profInfo.rmtAddr;
    slot->taskPara.ubDma.size = profInfo.size;
    slot->taskPara.ubDma.notifyId = profInfo.notifyId;
    slot->taskPara.ubDma.jettyHandle = profInfo.jettyHandle;
    slot->taskPara.ubDma.jettyId = profInfo.jettyId;
    slot->taskPara.ubDma.tpn = profInfo.tpn;
    PLF_CONFIG_INFO(PLF_TASK, "[%s] %s", __func__, slot->Describe().c_str());
    return HCCL_SUCCESS;
}

inline HcclResult AicpuTaskCacheEntry::FillSlotReduce_(
    Hccl::DfxTaskInfo* slot, const DbSqeProfAndRefreshInfo& profAndRefreshInfo,
    UbTransportLiteImpl* ubTransportLiteImplPtr, StreamLite* streamLite, u32 taskId) const
{
    const DbSqeProfInfo& profInfo = profAndRefreshInfo.dbSqeProfInfo;
    slot->taskType = static_cast<u8>(profInfo.taskParamType);
    FillSlotCommonFields_(
        slot, streamLite, taskId, GetUbLinkTypeVal_(ubTransportLiteImplPtr),
        static_cast<u8>(Hccl::DfxTransportType::DFX_TRANSPORT_TYPE_UB), ReinterpretAs<u64>(ubTransportLiteImplPtr));
    slot->taskPara.Reduce.srcAddr = profInfo.locAddr;
    slot->taskPara.Reduce.dstAddr = profInfo.rmtAddr;
    slot->taskPara.Reduce.size = profInfo.size;
    slot->taskPara.Reduce.reduceOp = static_cast<u8>(profInfo.reduceOp);
    slot->taskPara.Reduce.notifyId = profInfo.notifyId;
    slot->taskPara.Reduce.jettyHandle = profInfo.jettyHandle;
    slot->taskPara.Reduce.jettyId = profInfo.jettyId;
    slot->taskPara.Reduce.tpn = profInfo.tpn;
    PLF_CONFIG_INFO(PLF_TASK, "[%s] %s", __func__, slot->Describe().c_str());
    return HCCL_SUCCESS;
}

HcclResult AicpuTaskCacheEntry::ReportSqeProfiling_(
    uint8_t* sqePtr, size_t arrayIdx, uint32_t sqeIdx, const uint64_t* baseAddrs, const uint64_t* memSizes,
    const uint32_t count, StreamLite* streamLite, const u32 sqId)
{
    // 注意: 参考aicpu_ts_thread.cc填充DfxTaskInfo并经NextTaskSlot上报

    // 获取SQE对应的sqeType和taskId
    Rt91095StarsSqeHeader* sqeHeaderPtr
        = static_cast<Rt91095StarsSqeHeader*>(static_cast<void*>(sqePtr)); // 已在AddSqeArray校验, 无需再校验
    const Rt91095StarsSqeType sqeType = static_cast<Rt91095StarsSqeType>(sqeHeaderPtr->type);
    const u32 taskId = (sqeHeaderPtr->taskId << 16) | (sqeHeaderPtr->rtStreamId);

    switch (sqeType) {
        case Rt91095StarsSqeType::RT_91095_SQE_TYPE_PLACE_HOLDER:
            // 950 的零长度 LocalCopy 不生成数据 task，也不创建 DFX slot。
            break;
        case Rt91095StarsSqeType::RT_91095_SQE_TYPE_UBDMA:
            // DbSqe由ReportDbSqeProfiling_内部按需获取NextTaskSlot并填充
            CHK_RET(
                ReportDbSqeProfiling_(sqePtr, arrayIdx, sqeIdx, baseAddrs, memSizes, count, streamLite, sqId, taskId));
            break;
        case Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_RECORD:
        case Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT: {
            Hccl::DfxTaskInfo* slot = streamLite->NextTaskSlot();
            CHK_RET(FillSlotNotify_(
                slot, sqePtr, streamLite, taskId, sqeArrayInfos_[arrayIdx].sqeTransports[sqeIdx],
                sqeArrayInfos_[arrayIdx].sqeLinkTypes[sqeIdx], sqeArrayInfos_[arrayIdx].sqeTransportTypes[sqeIdx]));
            break;
        }
        case Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA: {
            Hccl::DfxTaskInfo* slot = streamLite->NextTaskSlot();
            CHK_RET(FillSlotSdma_(
                slot, sqePtr, streamLite, taskId, sqeArrayInfos_[arrayIdx].sqeTransports[sqeIdx],
                sqeArrayInfos_[arrayIdx].sqeLinkTypes[sqeIdx], sqeArrayInfos_[arrayIdx].sqeTransportTypes[sqeIdx]));
            break;
        }
        case Rt91095StarsSqeType::RT_91095_SQE_TYPE_WRITE_VALUE: {
            Hccl::DfxTaskInfo* slot = streamLite->NextTaskSlot();
            FillSlotCommonFields_(
                slot, streamLite, taskId, sqeArrayInfos_[arrayIdx].sqeLinkTypes[sqeIdx],
                sqeArrayInfos_[arrayIdx].sqeTransportTypes[sqeIdx], sqeArrayInfos_[arrayIdx].sqeTransports[sqeIdx]);
            slot->taskType = static_cast<u8>(Hccl::TaskParamTypeVal::TASK_NOTIFY_RECORD);
            slot->taskPara.Notify.notifyId = 0;
            PLF_CONFIG_INFO(Hccl::PLF_TASK, "[%s] %s", __func__, slot->Describe().c_str());
            break;
        }
        default:
            HCCL_ERROR("[AicpuTaskCacheEntry][ReportSqeProfiling_] sqeType[%u] is unsupported", sqeType);
            return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

inline HcclResult AicpuTaskCacheEntry::FillSlotNotify_(
    Hccl::DfxTaskInfo* slot, const uint8_t* sqePtr, StreamLite* streamLite, u32 taskId, u64 channelHandle, u8 linkType,
    u8 transportType) const
{
    const Rt91095StarsSqeHeader* sqeHeaderPtr
        = static_cast<const Rt91095StarsSqeHeader*>(static_cast<const void*>(sqePtr));
    const Rt91095StarsSqeType sqeType = static_cast<Rt91095StarsSqeType>(sqeHeaderPtr->type);
    slot->taskType = static_cast<u8>(
        (sqeType == Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_RECORD) ? Hccl::TaskParamTypeVal::TASK_NOTIFY_RECORD :
                                                                            Hccl::TaskParamTypeVal::TASK_NOTIFY_WAIT);
    FillSlotCommonFields_(slot, streamLite, taskId, linkType, transportType, channelHandle);
    const Hccl::Rt91095StarsNotifySqe* notifySqe
        = static_cast<const Hccl::Rt91095StarsNotifySqe*>(static_cast<const void*>(sqePtr));
    slot->taskPara.Notify.notifyId = notifySqe->notifyId;
    PLF_CONFIG_INFO(PLF_TASK, "[%s] %s", __func__, slot->Describe().c_str());
    return HCCL_SUCCESS;
}

inline HcclResult AicpuTaskCacheEntry::FillSlotSdma_(
    Hccl::DfxTaskInfo* slot, const uint8_t* sqePtr, StreamLite* streamLite, u32 taskId, u64 channelHandle, u8 linkType,
    u8 transportType) const
{
    const Hccl::Rt91095StarsMemcpySqe* sdmaSqe
        = static_cast<const Hccl::Rt91095StarsMemcpySqe*>(static_cast<const void*>(sqePtr));
    FillSlotCommonFields_(slot, streamLite, taskId, linkType, transportType, channelHandle);
    constexpr uint32_t UINT32_BIT_WIDTH = 32;
    if (sdmaSqe->opcode == SDMA_OPCODE_MEMCPY) {
        slot->taskType = static_cast<u8>(Hccl::TaskParamTypeVal::TASK_SDMA);
        slot->taskPara.Dma.srcAddr = (static_cast<uint64_t>(sdmaSqe->u.strideMode0.srcAddrHigh) << UINT32_BIT_WIDTH)
                                     | sdmaSqe->u.strideMode0.srcAddrLow;
        slot->taskPara.Dma.dstAddr = (static_cast<uint64_t>(sdmaSqe->u.strideMode0.dstAddrHigh) << UINT32_BIT_WIDTH)
                                     | sdmaSqe->u.strideMode0.dstAddrLow;
        slot->taskPara.Dma.size = sdmaSqe->u.strideMode0.lengthMove;
        slot->taskPara.Dma.notifyId = INVALID_U32;
    } else {
        slot->taskType = static_cast<u8>(Hccl::TaskParamTypeVal::TASK_REDUCE_INLINE);
        slot->taskPara.Reduce.srcAddr = (static_cast<uint64_t>(sdmaSqe->u.strideMode0.srcAddrHigh) << UINT32_BIT_WIDTH)
                                        | sdmaSqe->u.strideMode0.srcAddrLow;
        slot->taskPara.Reduce.dstAddr = (static_cast<uint64_t>(sdmaSqe->u.strideMode0.dstAddrHigh) << UINT32_BIT_WIDTH)
                                        | sdmaSqe->u.strideMode0.dstAddrLow;
        slot->taskPara.Reduce.size = sdmaSqe->u.strideMode0.lengthMove;
        slot->taskPara.Reduce.notifyId = INVALID_U32;
        slot->taskPara.Reduce.reduceOp = ConvertSdmaOpCodeToReduceOp_(sdmaSqe->opcode);
    }
    PLF_CONFIG_INFO(PLF_TASK, "[%s] %s", __func__, slot->Describe().c_str());
    return HCCL_SUCCESS;
}

HcclResult AicpuTaskCacheEntry::ReportSqeArrayProfiling_(
    size_t arrayIdx, const uint64_t* baseAddrs, const uint64_t* memSizes, const uint32_t count)
{
    // 注意: arrayIdx已在SubmitCacheEntry_校验, 这里无需重复校验
    const SqeArrayInfo& sqeArrayInfo = sqeArrayInfos_[arrayIdx];

    // sqe数组
    uint8_t* sqePtr = sqeArrayInfo.sqeArray; // 注意: sqePtr已在AddSqeArray校验, 无需再校验
    uint64_t sqeCount = sqeArrayInfo.sqeCount;

    // 获取SQE对应的sqId
    // 注意: aicpuTsThreadPtr已在AddSqeArray校验, 无需再校验
    StreamLite* streamLite = ReinterpretAs<StreamLite*>(sqeArrayInfo.aicpuTsThreadPtr->GetStreamLitePtr());
    CHK_PTR_NULL(streamLite);
    const u32 sqId = streamLite->GetSqId();
    for (size_t sqeIdx = 0; sqeIdx < sqeCount; ++sqeIdx) {
        CHK_PRT(ReportSqeProfiling_(sqePtr, arrayIdx, sqeIdx, baseAddrs, memSizes, count, streamLite, sqId));

        // 切换到下一个SQE
        sqePtr += AC_SQE_SIZE;
    }

    return HCCL_SUCCESS;
}

} // namespace hcomm
