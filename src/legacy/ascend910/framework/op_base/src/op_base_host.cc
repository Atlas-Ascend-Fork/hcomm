/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <future>
#include <map>
#include <string>
#include <hccl/hccl_types.h>

#include "hccl/base.h"
#include "param_check_pub.h"
#include "externalinput_pub.h"
#include "../common/src/state_guard.h"
#include "sal_pub.h"
#include "profiling_manager_pub.h"
#include "adapter_prof.h"
#include "adapter_rts_common.h"
#include "error_codes/rt_error_codes.h"
#include "op_base.h"
#include "hccl_group.h"
#include "hcom_common.h"
#include "coll_comm_mgr.h"
#include "coll_comm.h" // HcclDevMemAcquire 的 DPUTAG 分支：CollComm/ThreadMgr 完整类型
#include "rank_consistentcy_checker.h"
#include "../common/src/topo/topoinfo_ranktable_partition.h"
#include "mmpa_api.h"
#include "../nslbdp/hccl_nslbdp.h"

using namespace std;
using namespace hccl;

HcclResult GetCaptureInfo(aclrtStream stream, aclmdlRICaptureStatus& captureStatus, uint64_t& modelId, bool& isCapture)
{
    isCapture = false;
    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        HCCL_WARNING("[%s]Stream capture only support opbase mode!", __func__);
        return HCCL_SUCCESS;
    }
    aclmdlRI rtModel = nullptr;
    aclError ret = aclmdlRICaptureGetInfo(stream, &captureStatus, &rtModel);
    if (ret == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {
        HCCL_WARNING("[%s]Stream capture does not support!", __func__);
        return HCCL_SUCCESS;
    } else {
        CHK_PRT_RET(
            ret != ACL_SUCCESS, HCCL_ERROR("[%s]rtGet stream get capture status fail. return[%d]", __func__, ret),
            HCCL_E_RUNTIME);
    }
    if (captureStatus == ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE) {
        isCapture = true;
        uint32_t mdlId;
        rtError_t rtRet = rtModelGetId(rtModel, &mdlId);
        CHK_PRT_RET(
            rtRet != RT_ERROR_NONE, HCCL_ERROR("[%s]rtGet stream get model id fail. return[%d]", __func__, rtRet),
            HCCL_E_RUNTIME);
        modelId = static_cast<uint64_t>(mdlId);
    }

    return HCCL_SUCCESS;
}

HcclResult HcclAllReduceInner(
    void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, HcclComm comm,
    aclrtStream stream)
{
    // 入参合法性校验
    CHK_PRT_RET(count == 0, HCCL_WARNING("input count is 0, return AllReduce success"), HCCL_SUCCESS);
    RPT_INPUT_ERR(
        comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclAllReduceInner", "nullptr", "comm", "non-null pointer"}));
    CHK_PTR_NULL(comm);
    RPT_INPUT_ERR(
        sendBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclAllReduceInner", "nullptr", "sendBuf", "non-null pointer"}));
    CHK_PTR_NULL(sendBuf);
    RPT_INPUT_ERR(
        recvBuf == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclAllReduceInner", "nullptr", "recvBuf", "non-null pointer"}));
    CHK_PTR_NULL(recvBuf);
    RPT_INPUT_ERR(
        stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "value"}),
        std::vector<std::string>({"HcclAllReduceInner", "nullptr", "stream", "non-null pointer"}));
    CHK_PTR_NULL(stream);

    if (hcclGroupDepth > 0) {
        struct hcclOpInfo info;
        info.coll = HcclCMDType::HCCL_CMD_ALLREDUCE;
        info.sendbuff = sendBuf;
        info.recvbuff = recvBuf;
        info.sendCount = count;
        info.sendType = dataType;
        info.recvType = dataType;
        info.op = op;
        info.comm = comm;
        info.stream = stream;
        CHK_RET(taskAppend(comm, info));
        HCCL_INFO(
            "[HcclAllReduce] Finish taskAppend, count [%d] dataType [%s]", count, GetDataTypeEnumStr(dataType).c_str());
        return HCCL_SUCCESS;
    }
    HcclUs startut = TIME_NOW();

    bool isCapture;
    aclmdlRICaptureStatus captureStatus = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    uint64_t modelId = 0xFFFFFFFF;
    CHK_PRT(GetCaptureInfo(stream, captureStatus, modelId, isCapture));
    if (!isCapture) {
        HcclSetIfProfile();
    }

    uint64_t beginTime = hrtMsprofSysCycleTime();

    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
        CHK_RET(HcclAllReduceV2(sendBuf, recvBuf, count, dataType, op, hcclComm->GetCommunicatorV2(), stream));
        return HCCL_SUCCESS;
    }());
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    const std::lock_guard<std::mutex> lock(hcclComm->operatorlock_);
    StateGuard<hccl::hcclComm, HcclCommState> guard(hcclComm, HcclCommState::INUSE);
    s32 threadID = SalGetTid();
    ProfilingManagerPub::SetThreadCaptureStatus(threadID, isCapture);
    // 同通信域同算子复用tag
    const string tag = "AllReduce_" + hcclComm->GetIdentifier();

    CHK_RET_AND_PRINT_IDE(HcomCheckOpParam(tag.c_str(), count, dataType, stream), tag.c_str());

    CHK_RET_AND_PRINT_IDE(HcomCheckReductionOp("HcclAllReduceInner", op), tag.c_str());
    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    CHK_RET_AND_PRINT_IDE(HcomCheckReduceDataType(dataType, op, devType), tag.c_str());

    /* 接口交互信息日志 */
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    if (GetExternalInputHcclEnableEntryLog()) {
        s32 deviceLogicId = 0;
        CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));

        u32 localRank = INVALID_VALUE_RANKID;
        CHK_RET_AND_PRINT_IDE(hcclComm->GetUserRank(localRank), tag.c_str());

        s32 streamId = 0;
        CHK_RET_AND_PRINT_IDE(hrtGetStreamId(stream, streamId), tag.c_str());

        s32 ret = snprintf_s(
            stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            "tag[%s], sendBuf[%p], recvBuf[%p], count[%llu], dataType[%s], op[%s], localRank[%u], streamId[%d], "
            "comm[%p], deviceLogicId[%d]",
            tag.c_str(), sendBuf, recvBuf, count, GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str(),
            localRank, streamId, comm, deviceLogicId);

        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));

        std::string logInfo = "Entry-HcclAllReduceInner: " + std::string(stackLogBuffer) + ", capture status["
                              + to_string(captureStatus) + "], model id[" + to_string(modelId) + "].";
        CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(logInfo), tag.c_str());
    }

    CHK_RET_AND_PRINT_IDE(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE), tag.c_str());

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(sendBuf), tag.c_str());

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(recvBuf), tag.c_str());

    CHK_RET_AND_PRINT_IDE(SetOverFlowAddr(hcclComm), tag.c_str());
    CHK_RET_AND_PRINT_IDE(hcclComm->AllReduceOutPlace(tag, sendBuf, recvBuf, count, dataType, op, stream), tag.c_str());
    CHK_RET(CallMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_ALLREDUCE, beginTime, count, dataType, tag));

    if (!isCapture) {
        HcclResetIfProfile();
    }
    ProfilingManagerPub::DeleteThreadCaptureStatus(threadID);

    if (GetExternalInputHcclEnableEntryLog()) {
        HcclUs endut = TIME_NOW();
        /* 关键状态记录 */
        std::string endInfo
            = "HcclAllReduceInner:success,take time: " + std::to_string(DURATION_US(endut - startut).count()) + " us,"
              + std::string(stackLogBuffer);
        CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(endInfo), tag.c_str());
    }

    return HCCL_SUCCESS;
}

HcclResult HcclBarrier(HcclComm comm, aclrtStream stream)
{
    // 入参合法性校验
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);
    HcclUs startut = TIME_NOW();
    bool isCapture;
    aclmdlRICaptureStatus captureStatus = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    uint64_t modelId = 0xFFFFFFFF;
    CHK_PRT(GetCaptureInfo(stream, captureStatus, modelId, isCapture));
    if (!isCapture) {
        HcclSetIfProfile();
    }
    s32 threadID = SalGetTid();
    ProfilingManagerPub::SetThreadCaptureStatus(threadID, isCapture);
    uint64_t beginTime = hrtMsprofSysCycleTime();
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
        CHK_RET(HcclBarrierV2(hcclComm->GetCommunicatorV2(), stream));
        return HCCL_SUCCESS;
    }());

    // Allreduce入参定义
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    StateGuard<hccl::hcclComm, HcclCommState> guard(hcclComm, HcclCommState::INUSE);
    // 同通信域同算子复用tag
    const string tag = "AllReduce_" + hcclComm->GetIdentifier();

    /* 接口交互信息日志 */
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    if (GetExternalInputHcclEnableEntryLog()) {
        s32 deviceLogicId = 0;
        CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));

        u32 localRank = INVALID_VALUE_RANKID;
        CHK_RET_AND_PRINT_IDE(hcclComm->GetUserRank(localRank), tag.c_str());

        s32 streamId = 0;
        CHK_RET_AND_PRINT_IDE(hrtGetStreamId(stream, streamId), tag.c_str());

        s32 ret = snprintf_s(
            stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
            "tag[%s], sendBuf[%p], recvBuf[%p], count[%d], dataType[%s], op[%s], localRank[%u], streamId[%d], "
            "deviceLogicId[%d]",
            tag.c_str(), hcclComm->barrierSendBuf, hcclComm->barrierRecvBuf, HCCL_BARRIER_DEFAULT_COUNT,
            GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str(), localRank, streamId, deviceLogicId);

        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
        std::string logInfo = "Entry-HcclBarrier:" + std::string(stackLogBuffer) + ", capture status["
                              + to_string(captureStatus) + "], model id[" + to_string(modelId) + "].";
        CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(logInfo), tag.c_str());
    }

    CHK_RET_AND_PRINT_IDE(hcclComm->CreateBarrierMemory(), tag.c_str());

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(hcclComm->barrierSendBuf), tag.c_str());

    CHK_RET_AND_PRINT_IDE(PrintMemoryAttr(hcclComm->barrierRecvBuf), tag.c_str());

    CHK_RET_AND_PRINT_IDE(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE), tag.c_str());

    CHK_RET_AND_PRINT_IDE(
        hcclComm->AllReduceOutPlace(
            tag, hcclComm->barrierSendBuf, hcclComm->barrierRecvBuf, HCCL_BARRIER_DEFAULT_COUNT, dataType, op, stream,
            SyncMode::UNLIMITED_TIMEWAITSYNCMODE),
        tag.c_str());

    CHK_RET(CallMsprofReportHostApi(
        hcclComm, HcclCMDType::HCCL_CMD_ALLREDUCE, beginTime, HCCL_BARRIER_DEFAULT_COUNT, dataType, tag));
    if (!isCapture) {
        HcclResetIfProfile();
    }
    ProfilingManagerPub::DeleteThreadCaptureStatus(threadID);

    if (GetExternalInputHcclEnableEntryLog()) {
        HcclUs endut = TIME_NOW();
        /* 关键状态记录 */
        std::string endInfo = "HcclBarrier:success,take time: " + std::to_string(DURATION_US(endut - startut).count())
                              + " us," + std::string(stackLogBuffer);
        CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(endInfo), tag.c_str());
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommInitClusterInfoWrapper([[maybe_unused]] struct hcclAsyncJob* job_)
{
    struct hcclCommInitRankTableAsyncJob* job = static_cast<hcclCommInitRankTableAsyncJob*>(job_);
    uint32_t rank = job->rank;
    HcclComm* comm = job->initComm;
    const char* clusterInfo = job->clusterInfo;
    s32 devId = job->devId;
    HCCL_DEBUG("[HcclCommInitClusterInfoWrapper] Set device devId: %d", devId);
    CHK_PRT_RET(
        hrtSetDevice(devId) != HCCL_SUCCESS, HCCL_ERROR("[HcclCommInitClusterInfoWrapper] set fail device[%d]", devId),
        HCCL_E_INTERNAL);
    HCCL_DEBUG("[HcclCommInitClusterInfoWrapper] Done Set device devId: %d", devId);

    HcclUs startut = TIME_NOW();
    s32 deviceLogicId = 0;
    CHK_RET(HcclDeviceRefresh(deviceLogicId));
    HCCL_RUN_INFO(
        "Entry-%s: clusterInfo[%s], rank[%u], deviceLogicId[%d].", __func__, clusterInfo, rank, deviceLogicId);
    // 入参合法性校验
    CHK_PTR_NULL(clusterInfo);
    CHK_PTR_NULL(comm);
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        void* commV2 = nullptr;
        CHK_RET(HcclCommInitClusterInfoV2(clusterInfo, rank, &commV2));
        constexpr HcclCommConfig* config = nullptr; // 未配置为默认加速模式
        HcclResult ret = HcclCommInitCollComm(rank, &commV2, config, comm);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[HcclCommInitCollComm]HcclCommInitCollComm failed.Destroy comv2");
            CHK_RET(HcclCommDestroyV2(commV2));
            commV2 = nullptr;
            *comm = nullptr;
            return ret;
        }
        return HCCL_SUCCESS;
    }());
    HcclResult ret = InitExternalInput();
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s]errNo[0x%016llx] init external input error.", __func__, HCCL_ERROR_CODE(ret)), HCCL_E_PARA);
    ret = InitEnvConfig();
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s]errNo[0x%016llx] init environment config error.", __func__, HCCL_ERROR_CODE(ret)), HCCL_E_PARA);

    std::string identifier = HCCL_WORLD_GROUP;
    CommConfig commConfig(identifier);
    std::string rankTableM;
    std::string realFilePath;
    ret = HcomLoadRanktableFile(clusterInfo, rankTableM, realFilePath);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Init][HcclCommInitClusterInfoWrapper]errNo[0x%016llx], clusterInfo[%s], rank[%u], "
            "load rankTable error.",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL), clusterInfo, rank),
        HCCL_E_INTERNAL);

    HCCL_INFO("%s success, clusterInfoRealPath[%s].", __func__, realFilePath.c_str());

    HcclOpInfoCtx& opBaseHcom = CollCommMgr::GetInstance().LegacyGetHcclOpInfoCtx();
    CHK_RET(CheckOpBasedHcom(opBaseHcom, rank, commConfig));

    CHK_RET(InitCommClusterInfo(rankTableM, rank, commConfig, opBaseHcom, comm));

    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "[HCCL_TRACE]%s success, take time [%lld]us, clusterInfo[%s], rank[%u], deviceLogicId[%d].", __func__,
        DURATION_US(TIME_NOW() - startut), clusterInfo, rank, deviceLogicId);
    return HCCL_SUCCESS;
}

HcclResult HcclCommInitClusterInfo(
    [[maybe_unused]] const char* clusterInfo, [[maybe_unused]] uint32_t rank, [[maybe_unused]] HcclComm* comm)
{
    if (hcclGroupDepth > 0) {
        HcclResult ret = HCCL_SUCCESS;
        std::shared_ptr<struct hcclCommInitRankTableAsyncJob> job;
        EXCEPTION_CATCH((job = std::make_shared<struct hcclCommInitRankTableAsyncJob>()), return HCCL_E_PARA);
        job->clusterInfo = clusterInfo;
        job->rank = rank;
        job->initComm = comm;
        s32 devId = 0;
        CHK_RET(HcclDeviceRefresh(devId));
        job->devId = devId;
        ret = commInitTaskAppend(job, HcclCommInitClusterInfoWrapper, comm);
        return ret;
    }
    HcclUs startut = TIME_NOW();
    s32 deviceLogicId = 0;
    CHK_RET(HcclDeviceRefresh(deviceLogicId));
    HCCL_RUN_INFO(
        "Entry-%s: clusterInfo[%s], rank[%u], deviceLogicId[%d].", __func__, clusterInfo, rank, deviceLogicId);
    // 入参合法性校验
    CHK_PTR_NULL(clusterInfo);
    CHK_PTR_NULL(comm);
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        void* commV2 = nullptr;
        CHK_RET(HcclCommInitClusterInfoV2(clusterInfo, rank, &commV2));
        constexpr HcclCommConfig* config = nullptr; // 未配置为默认加速模式
        HcclResult ret = HcclCommInitCollComm(rank, &commV2, config, comm);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[HcclCommInitCollComm]HcclCommInitCollComm failed.Destroy comv2");
            CHK_RET(HcclCommDestroyV2(commV2));
            commV2 = nullptr;
            *comm = nullptr;
            return ret;
        }
        return HCCL_SUCCESS;
    }());
    HcclResult ret = InitExternalInput();
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s]errNo[0x%016llx] init external input error.", __func__, HCCL_ERROR_CODE(ret)), HCCL_E_PARA);
    ret = InitEnvConfig();
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s]errNo[0x%016llx] init environment config error.", __func__, HCCL_ERROR_CODE(ret)), HCCL_E_PARA);

    std::string identifier = HCCL_WORLD_GROUP;
    CommConfig commConfig(identifier);
    std::string rankTableM;
    std::string realFilePath;
    ret = HcomLoadRanktableFile(clusterInfo, rankTableM, realFilePath);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Init][HcclCommInitClusterInfo]errNo[0x%016llx], clusterInfo[%s], rank[%u], "
            "load rankTable error.",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL), clusterInfo, rank),
        HCCL_E_INTERNAL);

    HCCL_INFO("%s success, clusterInfoRealPath[%s].", __func__, realFilePath.c_str());

    HcclOpInfoCtx& opBaseHcom = CollCommMgr::GetInstance().LegacyGetHcclOpInfoCtx();
    CHK_RET(CheckOpBasedHcom(opBaseHcom, rank, commConfig));

    CHK_RET(InitCommClusterInfo(rankTableM, rank, commConfig, opBaseHcom, comm));

    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "[HCCL_TRACE]%s success, take time [%lld]us, clusterInfo[%s], rank[%u], deviceLogicId[%d].", __func__,
        DURATION_US(TIME_NOW() - startut), clusterInfo, rank, deviceLogicId);
    return HCCL_SUCCESS;
}

HcclResult HcclCommInitClusterInfoConfigWrapper([[maybe_unused]] struct hcclAsyncJob* job_)
{
    struct hcclCommInitRankTableConfigAsyncJob* job = static_cast<hcclCommInitRankTableConfigAsyncJob*>(job_);
    uint32_t rank = job->rank;
    HcclComm* comm = job->initComm;
    const char* clusterInfo = job->clusterInfo;
    HcclCommConfig* config = job->config;
    s32 devId = job->devId;
    HCCL_DEBUG("[HcclCommInitClusterInfoConfigWrapper] Set device devId: %d", devId);
    CHK_PRT_RET(
        hrtSetDevice(devId) != HCCL_SUCCESS,
        HCCL_ERROR("[HcclCommInitClusterInfoConfigWrapper] set fail device[%d]", devId), HCCL_E_INTERNAL);
    HCCL_DEBUG("[HcclCommInitClusterInfoConfigWrapper] Done Set device devId: %d", devId);

    HcclUs startut = TIME_NOW();
    s32 deviceLogicId = 0;
    CHK_RET(HcclDeviceRefresh(deviceLogicId));
    HCCL_RUN_INFO(
        "Entry-%s: clusterInfo[%s], rank[%u], deviceLogicId[%d].", __func__, clusterInfo, rank, deviceLogicId);
    // 入参合法性校验
    CHK_PTR_NULL(clusterInfo);
    CHK_PTR_NULL(comm);

    // 检查配置参数是否为空
    RPT_INPUT_ERR(
        config == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclCommInitClusterInfoConfigWrapper", "nullptr", "config", "non-null pointer"}));
    CHK_SMART_PTR_NULL(config);
    const char* socNamePtr = aclrtGetSocName();
    CHK_PTR_NULL(socNamePtr);
    HCCLV2_FUNC_RUN(
        [&]() -> HcclResult {
            void* commV2 = nullptr;
            CHK_RET(HcclCommInitClusterInfoConfigV2(clusterInfo, rank, config, &commV2));
            HcclResult ret = HcclCommInitCollComm(rank, &commV2, config, comm);
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR("[HcclCommInitCollComm]HcclCommInitCollComm failed.Destroy comv2");
                CHK_RET(HcclCommDestroyV2(commV2));
                commV2 = nullptr;
                *comm = nullptr;
                return ret;
            }
            return HCCL_SUCCESS;
        }(),
        socNamePtr);
    HcclResult ret = InitExternalInput();
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s]errNo[0x%016llx] init external input error.", __func__, HCCL_ERROR_CODE(ret)), HCCL_E_PARA);
    ret = InitEnvConfig();
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s]errNo[0x%016llx] init environment config error.", __func__, HCCL_ERROR_CODE(ret)), HCCL_E_PARA);

    std::string identifier = HCCL_WORLD_GROUP;
    CommConfig commConfig(identifier);
    ret = commConfig.Load(config);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s]errNo[0x%016llx] load comm config failed.", __func__, HCCL_ERROR_CODE(ret)), HCCL_E_PARA);

    std::string rankTableM;
    std::string realFilePath;
    ret = HcomLoadRanktableFile(clusterInfo, rankTableM, realFilePath);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Init][HcclCommInitClusterInfoConfigWrapper]errNo[0x%016llx] clusterInfo[%s] rank[%u] "
            "load rankTable error.",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL), clusterInfo, rank),
        HCCL_E_INTERNAL);

    HCCL_INFO("%s success, clusterInfoRealPath[%s].", __func__, realFilePath.c_str());

    HcclOpInfoCtx& opBaseHcom = CollCommMgr::GetInstance().LegacyGetHcclOpInfoCtx();
    CHK_RET(CheckOpBasedHcom(opBaseHcom, rank, commConfig));

    CHK_RET(InitCommClusterInfo(rankTableM, rank, commConfig, opBaseHcom, comm));

    // 记录groupName和UDI的映射
    HCCL_PROFILER_ADD_GROUP_UDI(commConfig.GetConfigCommName(), commConfig.GetConfigUdi());

    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "[HCCL_TRACE]%s success, take time [%lld]us, clusterInfo[%s], rank[%u], deviceLogicId[%d].", __func__,
        DURATION_US(TIME_NOW() - startut), clusterInfo, rank, deviceLogicId);
    return HCCL_SUCCESS;
}

HcclResult HcclCommInitClusterInfoConfig(
    [[maybe_unused]] const char* clusterInfo, [[maybe_unused]] uint32_t rank, [[maybe_unused]] HcclCommConfig* config,
    [[maybe_unused]] HcclComm* comm)
{
    if (hcclGroupDepth > 0) {
        HcclResult ret = HCCL_SUCCESS;
        std::shared_ptr<struct hcclCommInitRankTableConfigAsyncJob> job;
        EXCEPTION_CATCH((job = std::make_shared<struct hcclCommInitRankTableConfigAsyncJob>()), return HCCL_E_PARA);
        job->clusterInfo = clusterInfo;
        job->rank = rank;
        job->config = config;
        job->initComm = comm;
        s32 devId = 0;
        CHK_RET(HcclDeviceRefresh(devId));
        job->devId = devId;
        ret = commInitTaskAppend(job, HcclCommInitClusterInfoConfigWrapper, comm);
        return ret;
    }
    HcclUs startut = TIME_NOW();
    s32 deviceLogicId = 0;
    CHK_RET(HcclDeviceRefresh(deviceLogicId));
    HCCL_RUN_INFO(
        "Entry-%s: clusterInfo[%s], rank[%u], deviceLogicId[%d].", __func__, clusterInfo, rank, deviceLogicId);
    // 入参合法性校验
    CHK_PTR_NULL(clusterInfo);
    CHK_PTR_NULL(comm);

    // 检查配置参数是否为空
    RPT_INPUT_ERR(
        config == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
        std::vector<std::string>({"HcclCommInitClusterInfoConfig", "nullptr", "config", "non-null pointer"}));
    CHK_SMART_PTR_NULL(config);
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        void* commV2 = nullptr;
        CHK_RET(HcclCommInitClusterInfoConfigV2(clusterInfo, rank, config, &commV2));
        HcclResult ret = HcclCommInitCollComm(rank, &commV2, config, comm);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[HcclCommInitCollComm]HcclCommInitCollComm failed.Destroy comv2");
            CHK_RET(HcclCommDestroyV2(commV2));
            commV2 = nullptr;
            *comm = nullptr;
            return ret;
        }
        return HCCL_SUCCESS;
    }());
    HcclResult ret = InitExternalInput();
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s]errNo[0x%016llx] init external input error.", __func__, HCCL_ERROR_CODE(ret)), HCCL_E_PARA);
    ret = InitEnvConfig();
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s]errNo[0x%016llx] init environment config error.", __func__, HCCL_ERROR_CODE(ret)), HCCL_E_PARA);

    std::string identifier = HCCL_WORLD_GROUP;
    CommConfig commConfig(identifier);
    ret = commConfig.Load(config);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s]errNo[0x%016llx] load comm config failed.", __func__, HCCL_ERROR_CODE(ret)), HCCL_E_PARA);

    std::string rankTableM;
    std::string realFilePath;
    ret = HcomLoadRanktableFile(clusterInfo, rankTableM, realFilePath);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Init][HcclCommInitClusterInfoConfig]errNo[0x%016llx] clusterInfo[%s] rank[%u] "
            "load rankTable error.",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL), clusterInfo, rank),
        HCCL_E_INTERNAL);

    HCCL_INFO("%s success, clusterInfoRealPath[%s].", __func__, realFilePath.c_str());

    HcclOpInfoCtx& opBaseHcom = CollCommMgr::GetInstance().LegacyGetHcclOpInfoCtx();
    CHK_RET(CheckOpBasedHcom(opBaseHcom, rank, commConfig));

    CHK_RET(InitCommClusterInfo(rankTableM, rank, commConfig, opBaseHcom, comm));

    // 记录groupName和UDI的映射
    HCCL_PROFILER_ADD_GROUP_UDI(commConfig.GetConfigCommName(), commConfig.GetConfigUdi());

    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "[HCCL_TRACE]%s success, take time [%lld]us, clusterInfo[%s], rank[%u], deviceLogicId[%d].", __func__,
        DURATION_US(TIME_NOW() - startut), clusterInfo, rank, deviceLogicId);
    return HCCL_SUCCESS;
}

HcclResult HcclCreateSubCommConfigInner(
    [[maybe_unused]] hccl::hcclComm* globalComm, [[maybe_unused]] uint32_t rankNum, [[maybe_unused]] uint32_t* rankIds,
    [[maybe_unused]] uint32_t subCommRankId, [[maybe_unused]] CommConfig& commConfig,
    [[maybe_unused]] HcclComm* subComm)
{
    HcclResult ret = HCCL_SUCCESS;
    HcclCommParams globalParams{};
    RankTable_t globalRankTable{};
    CHK_RET(globalComm->GetCommParams(globalParams));
    CHK_RET(globalComm->GetCommRankTable(globalRankTable));

    HcclOpInfoCtx& opBaseHcom = CollCommMgr::GetInstance().LegacyGetHcclOpInfoCtx();

    const std::string commIdentifier = commConfig.GetConfigCommName();
    auto iter = opBaseHcom.opGroup2CommMap.find(commIdentifier);
    CHK_PRT_RET(
        iter != opBaseHcom.opGroup2CommMap.end(),
        HCCL_ERROR(
            "[%s]errNo[0x%016llx]The comm name[%s] already exists in Group2Comm map.", __func__,
            HCCL_ERROR_CODE(HCCL_E_PARA), commIdentifier.c_str()),
        HCCL_E_PARA);

    std::shared_ptr<hccl::hcclComm> pComm;
    pComm.reset(new (std::nothrow) hccl::hcclComm(
        commConfig.GetConfigBufferSize(), commConfig.GetConfigBufferSize(), commIdentifier,
        commConfig.GetConfigBufferName()));
    CHK_PTR_NULL(pComm);

    bool errorFlag = false;
    hccl::HcclCommParams subParams{};
    hccl::RankTable_t subRankTable{};
    do {
        RankConsistentcyChecker::GetInstance().SetCheckCannVersionSwitch(true); // 打开CANN软件版本校验开关

        std::unique_ptr<TopoinfoRanktablePartition> pTopoPartition;
        pTopoPartition.reset(new (std::nothrow) hccl::TopoinfoRanktablePartition(globalParams, globalRankTable));
        CHK_SMART_PTR_NULL(pTopoPartition);
        CHK_RET(pTopoPartition->GenerateSubRankTable(rankNum, rankIds, subRankTable));
        CHK_RET(pTopoPartition->GenerateSubParams(subRankTable, subCommRankId, subParams));

        std::string rankTableM = "";
        CHK_RET(pTopoPartition->GetRankTableStr(subRankTable, rankTableM));

        ret = InitOtherInfo(subParams, rankTableM.c_str());
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS, HCCL_ERROR("[%s]errNo[0x%016llx] init other Info.", __func__, HCCL_ERROR_CODE(ret)),
            errorFlag = true);
        ret = pComm->init(subParams, commConfig, subRankTable);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[%s]errNo[0x%016llx] hcclComm init error.", __func__, HCCL_ERROR_CODE(ret)), errorFlag = true);
        HCCL_INFO("[HcclCreateSubCommConfigInner]comm id[%s]", subParams.id.internal);

        /* 设置确定性计算配置 */
        ret = pComm->SetDeterministicConfig(commConfig.GetConfigDeterministic());
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[%s]errNo[0x%016llx] set deterministic error.", __func__, HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        // 设置TC/SL配置
        ret = pComm->SetQpQosAttr(commConfig.GetConfigTrafficClass(), commConfig.GetConfigServiceLevel());
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS, HCCL_ERROR("[%s]errNo[0x%016llx] set TC and SL error", __func__, HCCL_ERROR_CODE(ret)),
            errorFlag = true);
        if (commConfig.GetConfigJobID() != 0) {
            HCCL_RUN_INFO(
                "[NSLBDP]GetConfigJobID = %llu,GetConfigWorldRankID = %u.", commConfig.GetConfigJobID(),
                commConfig.GetConfigWorldRankID());
            hcclNslbDp::GetInstance().SetGlobalCommTaskId(commConfig.GetConfigJobID());
            hcclNslbDp::GetInstance().SetGlobalCommNodeId(commConfig.GetConfigWorldRankID());
        }
        /* 设置AIV模式 */
        ret = pComm->SetAivModeConfig(commConfig.GetConfigAivMode());
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS, HCCL_ERROR("[%s]errNo[0x%016llx] set aivMode error.", __func__, HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        /* 设置only AIV模式 */
        ret = pComm->SetOnlyAivModeConfig(commConfig.GetConfigIsOnlyAivMode());
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[%s]errNo[0x%016llx] set only aivMode error.", __func__, HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        /* 设置AICPU */
        ret = pComm->SetAicpuUnfoldConfig(commConfig.GetConfigAicpuUnfold());
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS, HCCL_ERROR("[%s]errNo[0x%016llx] set aicpu error.", __func__, HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        /* 设置HcclExecTimeOut */
        ret = pComm->SetExecTimeOutConfig(commConfig.GetConfigExecTimeOut());
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[Init][CommClusterInfo]errNo[0x%016llx] set execTimeOut error.", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        /* 设置HcclAlgo */
        ret = pComm->SetAlgoConfig(commConfig.GetConfigHcclAlgoMap());
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[Init][CommClusterInfo]errNo[0x%016llx] set hcclAlgo error.", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        ret = InitWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[%s]errNo[0x%016llx] init workflow mode error.", __func__, HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        ret = DisplayRanktableInfo(subRankTable);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[%s]errNo[0x%016llx] print ranktable info error.", __func__, HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        /* 设置独立算子参数 */
        ret = pComm->SetIndependentOpConfig(commConfig, subRankTable);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[%s] errNo[0x%016llx] set SetIndependentOpConfig error.", __func__, HCCL_ERROR_CODE(ret)),
            errorFlag = true);
        // 初始化完成的comm指针赋给出参
        *subComm = pComm.get();
        std::unique_lock<std::mutex> lock(opBaseHcom.opGroupMapMutex);
        opBaseHcom.opGroup2CommMap[pComm->GetIdentifier()] = pComm;
        lock.unlock();

        ret = HcomSetGroupTopoInfo(pComm->GetIdentifier().c_str(), rankNum);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[%s]errNo[0x%016llx] set group topo info error.", __func__, HCCL_ERROR_CODE(ret)),
            errorFlag = true);
        ret = pComm->InitHccpChannel();
        if (ret != HCCL_SUCCESS) {
            HCCL_WARNING("InitHccp channel unsuccessful ret:[%u].", ret);
        }
    } while (0);

    if (errorFlag) {
        HCCL_ERROR(
            "[%s]Create sub communication failed, return[0x%016llx], "
            "rankNum[%u], subCommRankId[%u], sub comm identifier[%s], server[%s], logicDevId[%d]",
            __func__, HCCL_ERROR_CODE(ret), rankNum, subCommRankId, commIdentifier.c_str(),
            GetLocalServerId(subParams.serverId).c_str(), subParams.logicDevId);
        (void)HcclCommDestroy(pComm.get());
        return ret;
    }
    std::string identifier = pComm->GetIdentifier();

    /* NSLB 填充 表1 */
    CHK_RET(hcclNslbDp::GetInstance().SetCommInfo_NoRankTable(subRankTable, identifier, subCommRankId));

    HCCL_RUN_INFO(
        "%s success, sub comm identifier[%s], rankNum[%u], rank[%u], server[%s], device[%d].", __func__,
        commIdentifier.c_str(), subRankTable.rankNum, subCommRankId, subParams.serverId.c_str(), subParams.logicDevId);
    return HCCL_SUCCESS;
}

HcclResult HcclGetRootInfo([[maybe_unused]] HcclRootInfo* rootInfo)
{
    HcclUs startut = TIME_NOW();
    s32 deviceLogicId = 0;
    CHK_RET(HcclDeviceRefresh(deviceLogicId));

    // input check
    CHK_PTR_NULL(rootInfo);
    HCCL_RUN_INFO("Entry-HcclGetRootInfo:rootInfo[%p], deviceLogicId[%d] ", rootInfo, deviceLogicId);
    HCCLV2_FUNC_RUN(HcclGetRootInfoV2(rootInfo));
    // get commId from env
    CHK_RET(InitExternalInput());
    CHK_RET(InitEnvConfig());

    HcclRootHandle rootHandle;
    std::shared_ptr<TopoInfoDetect> topoDetectServer;
    EXCEPTION_CATCH((topoDetectServer = std::make_shared<TopoInfoDetect>()), return HCCL_E_MEMORY);
    HcclResult ret = topoDetectServer->SetupServer(rootHandle);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]%s failed, ret[%u]", LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_DETECT.c_str(),
            __func__, ret),
        ret);

    if (sizeof(HcclRootHandle) > HCCL_ROOT_INFO_BYTES) {
        HCCL_ERROR(
            "[Get][RootInfo]hccl root info overflow. max length: %u, actual:%zu, identifier[%s]", HCCL_ROOT_INFO_BYTES,
            sizeof(HcclRootHandle), rootHandle.identifier);
        return HCCL_E_INTERNAL;
    } else {
        s32 sRet = memcpy_s(rootInfo->internal, HCCL_ROOT_INFO_BYTES, &rootHandle, sizeof(HcclRootHandle));
        CHK_PRT_RET(
            sRet != EOK,
            HCCL_ERROR(
                "[Get][RootInfo]memcpy root info fail. errorno[%d] "
                "params:destMaxSize[%u], count[%u]",
                sRet, HCCL_ROOT_INFO_BYTES, sizeof(HcclRootHandle)),
            HCCL_E_MEMORY);
    }

    HcclOpInfoCtx& opBaseInfo = CollCommMgr::GetInstance().LegacyGetHcclOpInfoCtx();
    EXCEPTION_CATCH(
        opBaseInfo.hcclCommTopoInfoDetectServer.insert({rootHandle.identifier, topoDetectServer}),
        return HCCL_E_MEMORY);
    /* 首节点诊断信息记录 */
    HCCL_RUN_INFO(
        "[HCCL_TRACE]HcclGetRootInfo success, take time [%lld]us, identifier[%s]", DURATION_US(TIME_NOW() - startut),
        rootHandle.identifier);
    return HCCL_SUCCESS;
}

HcclResult
HcclGetCommHandle([[maybe_unused]] const char* commName, [[maybe_unused]] std::shared_ptr<hccl::hcclComm>& comm)
{
    CHK_PTR_NULL(commName);
    std::string group(commName);

    s32 deviceLogicId = 0;
    HcclResult ret = HCCL_SUCCESS;
    ret = hrtGetDevice(&deviceLogicId);
    if (ret == HCCL_SUCCESS && IsCommNameExistInOneSidedComms(deviceLogicId, commName)) {
        HcclOpInfoCtx& oneSidedHcom = GetOneSidedOpInfoCtx(deviceLogicId, commName);
        comm = oneSidedHcom.pComm;
        return HCCL_SUCCESS;
    }

    HcclOpInfoCtx& opBaseHcom = CollCommMgr::GetInstance().LegacyGetHcclOpInfoCtx();
    std::unique_lock<std::mutex> lock(opBaseHcom.opGroupMapMutex);
    auto iter = opBaseHcom.opGroup2CommMap.find(group);
    if (iter == opBaseHcom.opGroup2CommMap.end()) {
        HCCL_WARNING("please check the group name is correct, group=%s", commName);
        return HCCL_E_PARA;
    } else {
        comm = iter->second;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommGetHandleWithName([[maybe_unused]] const char* commName, [[maybe_unused]] HcclComm* comm)
{
    CHK_PTR_NULL(commName);
    CHK_PTR_NULL(comm);
    std::string group(commName);

    s32 deviceLogicId = 0;
    HcclResult ret = HCCL_SUCCESS;
    ret = hrtGetDevice(&deviceLogicId);
    if (ret == HCCL_SUCCESS && IsCommNameExistInOneSidedComms(deviceLogicId, commName)) {
        HcclOpInfoCtx& oneSidedHcom = GetOneSidedOpInfoCtx(deviceLogicId, commName);
        *comm = static_cast<HcclComm>(oneSidedHcom.pComm.get());
        return HCCL_SUCCESS;
    }

    HcclOpInfoCtx& opBaseHcom = CollCommMgr::GetInstance().LegacyGetHcclOpInfoCtx();
    std::unique_lock<std::mutex> lock(opBaseHcom.opGroupMapMutex);
    auto iter = opBaseHcom.opGroup2CommMap.find(group);
    if (iter == opBaseHcom.opGroup2CommMap.end()) {
        HCCL_ERROR("please check the group name is correct, group=%s", commName);
        return HCCL_E_PARA;
    } else {
        *comm = static_cast<HcclComm>(iter->second.get());
    }
    return HCCL_SUCCESS;
}

HcclResult HcclGetCommConnections(
    [[maybe_unused]] const HcclRootHandle& rootHandle, [[maybe_unused]] const std::string& identifier,
    [[maybe_unused]] HcclCommConnections& commConnections)
{
    HcclOpInfoCtx& opBaseInfo = CollCommMgr::GetInstance().LegacyGetHcclOpInfoCtx();
    auto iterServer = opBaseInfo.hcclCommTopoInfoDetectServer.find(rootHandle.identifier);
    if (iterServer == opBaseInfo.hcclCommTopoInfoDetectServer.end()) {
        commConnections.isRoot = false;
    } else {
        commConnections.isRoot = true;
        CHK_RET(iterServer->second->GetServerConnections(commConnections.serverConnections));
    }

    auto iterAgent = opBaseInfo.hcclCommTopoInfoDetectAgent.find(identifier);
    if (iterAgent == opBaseInfo.hcclCommTopoInfoDetectAgent.end()) {
        HCCL_ERROR("hccl get agent connections failed, identifier=%s", identifier.c_str());
        return HCCL_E_PARA;
    } else {
        CHK_RET(iterAgent->second->GetAgentConnection(commConnections.agentConnection));
    }
    return HCCL_SUCCESS;
}

void HcclCloseCommConnections([[maybe_unused]] const std::string& identifier)
{
    HcclOpInfoCtx& opBaseInfo = CollCommMgr::GetInstance().LegacyGetHcclOpInfoCtx();
    EXCEPTION_CATCH(opBaseInfo.hcclCommTopoInfoDetectServer.erase(identifier), return);
    EXCEPTION_CATCH(opBaseInfo.hcclCommTopoInfoDetectAgent.erase(identifier), return);
}

HcclResult InitCommRootInfo(
    [[maybe_unused]] const u32 nRanks, [[maybe_unused]] const u32 rank,
    [[maybe_unused]] const HcclRootHandle& rootHandle, [[maybe_unused]] const CommConfig& commConfig,
    [[maybe_unused]] HcclComm* comm, [[maybe_unused]] bool isScalable)
{
    HcclResult ret = HCCL_SUCCESS;
    bool errorFlag = false;
    std::shared_ptr<hccl::hcclComm> pComm;
    HcclOpInfoCtx& opBaseHcom = CollCommMgr::GetInstance().LegacyGetHcclOpInfoCtx();
    const std::string commIdentifier = commConfig.GetConfigCommName();
    auto iter = opBaseHcom.opGroup2CommMap.find(commIdentifier);
    CHK_PRT_RET(
        iter != opBaseHcom.opGroup2CommMap.end(),
        HCCL_ERROR(
            "[Init][InitCommRootInfo]errNo[0x%016llx] The comm name[%s] already exists in Group2Comm map.",
            HCCL_ERROR_CODE(HCCL_E_PARA), commIdentifier.c_str()),
        HCCL_E_PARA);
    hccl::HcclCommParams params;
    RankTable_t rankTable;
    HcclBasicRankInfo localRankInfo;

    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    bool retryEnable
        = devType == DevType::DEV_TYPE_910_93 && !commConfig.GetConfigAivMode()
          && (commConfig.GetConfigInterServerRetryEnable() || commConfig.GetConfigInterSuperPodRetryEnable());
    HCCL_INFO("[InitCommRootInfo] retryEnable is [%d]", retryEnable);

    do {
        RankConsistentcyChecker::GetInstance().SetCheckCannVersionSwitch(true); // 打开CANN软件版本校验开关
        pComm.reset(new hccl::hcclComm(
            commConfig.GetConfigBufferSize(), commConfig.GetConfigBufferSize(), commIdentifier,
            commConfig.GetConfigBufferName()));
        CHK_SMART_PTR_NULL(pComm);

        std::shared_ptr<TopoInfoDetect> topoDetectAgent;
        EXCEPTION_CATCH((topoDetectAgent = std::make_shared<TopoInfoDetect>()), return HCCL_E_MEMORY);
        topoDetectAgent->SetIsInterSuperPodRetryEnable(commConfig.GetConfigInterSuperPodRetryEnable());
        // 32k 作为agent开启阈值；scalable 建链按 root 分组后组内规模必然小于 nRanks，始终走 flat 路径
        if (nRanks > TOPO_HIERARCHICAL_ENABLE_THRESHOLD && !isScalable) {
            HCCL_RUN_INFO("[Init][CommRootInfo][Hierarchical]nRanks[%u] entry hierarchical topo detect.", nRanks);

            std::shared_ptr<TopoInfoDetect> topoDetectMember;
            EXCEPTION_CATCH((topoDetectMember = std::make_shared<TopoInfoDetect>()), return HCCL_E_MEMORY);
            topoDetectMember->SetIsInterSuperPodRetryEnable(commConfig.GetConfigInterSuperPodRetryEnable());

            HcclRankHandle groupLeader;
            ret = SetupHierarchical(nRanks, rank, rootHandle, topoDetectAgent, topoDetectMember, groupLeader);
            CHK_PRT_BREAK(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[Init][CommRootInfo]errNo[0x%016llx] setup "
                    "hierarchical error",
                    HCCL_ERROR_CODE(ret)),
                errorFlag = true);

            ret = GetTopoDetectInfo(params, rankTable, localRankInfo, groupLeader, topoDetectAgent, topoDetectMember);
            CHK_PRT_BREAK(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[Init][CommRootInfo][Hierarchical]errNo[0x%016llx] setup "
                    "GetTopoDetectInfo error",
                    HCCL_ERROR_CODE(ret)),
                errorFlag = true);
        } else {
            HCCL_RUN_INFO("[Init][CommRootInfo][Flat]nRanks[%u] entry flat topo detect.", nRanks);

            ret = topoDetectAgent->SetupAgent(nRanks, rank, rootHandle, rootHandle, commConfig, isScalable);
            CHK_PRT_BREAK(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[Init][CommRootInfo][Flat]errNo[0x%016llx] "
                    "setup flat topo detect error",
                    HCCL_ERROR_CODE(ret)),
                errorFlag = true);
            ret = GetTopoDetectInfo(params, rankTable, localRankInfo, rootHandle, topoDetectAgent, topoDetectAgent);
            CHK_PRT_BREAK(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[Init][CommRootInfo][Flat]errNo[0x%016llx] setup "
                    "GetTopoDetectInfo error",
                    HCCL_ERROR_CODE(ret)),
                errorFlag = true);
        }

        /* 初始化hccl comm */

        CHK_RET(DisplayRanktableInfo(rankTable));

        if (retryEnable) {
            EXCEPTION_CATCH(
                opBaseHcom.hcclCommTopoInfoDetectAgent.insert({commIdentifier, topoDetectAgent}), return HCCL_E_MEMORY);
            ret = HcclGetCommConnections(rootHandle, commIdentifier, params.commConnections);
            CHK_PRT_BREAK(
                ret != HCCL_SUCCESS, HCCL_ERROR("[Init][RootInfo]HcclGetCommConnections failed."), errorFlag = true);
        } else {
            ret = topoDetectAgent->Teardown();
            CHK_PRT_BREAK(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[Init][RootInfo]errNo[0x%016llx] Teardown topo detect error", HCCL_ERROR_CODE(ret)),
                errorFlag = true);
        }

        ret = InitWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[InitCommRootInfo]errNo[0x%016llx] init work flow mode error", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        ret = InitOtherInfo(params, nullptr);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS, HCCL_ERROR("[InitCommRootInfo]errNo[0x%016llx] init other Info", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        HCCL_INFO("rootInfo[%s], params.logiceDevice[%d]", params.id.internal, params.logicDevId);
        ret = pComm->init(params, commConfig, rankTable);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[InitCommRootInfo]errNo[0x%016llx] hcclComm init error", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        /* 设置确定性计算配置 */
        ret = pComm->SetDeterministicConfig(commConfig.GetConfigDeterministic());
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[InitCommRootInfo]errNo[0x%016llx] set deterministic error", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        // 设置TC/SL配置
        ret = pComm->SetQpQosAttr(commConfig.GetConfigTrafficClass(), commConfig.GetConfigServiceLevel());
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[InitCommRootInfo]errNo[0x%016llx] set TC and SL error or Invalid configuration parameter.",
                HCCL_ERROR_CODE(ret)),
            errorFlag = true);
        if (commConfig.GetConfigJobID() != 0) {
            HCCL_RUN_INFO(
                "[NSLBDP]GetConfigJobID = %llu,GetConfigWorldRankID = %u.", commConfig.GetConfigJobID(),
                commConfig.GetConfigWorldRankID());
            hcclNslbDp::GetInstance().SetGlobalCommTaskId(commConfig.GetConfigJobID());
            hcclNslbDp::GetInstance().SetGlobalCommNodeId(commConfig.GetConfigWorldRankID());
        }

        // 设置HCCL QOS配置
        ret = pComm->SetHcclQos(commConfig.GetConfigHcclQos());
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS, HCCL_ERROR("[%s]errNo[0x%016llx] set hccl qos error.", __func__, HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        /* 设置AIV模式 */
        ret = pComm->SetAivModeConfig(commConfig.GetConfigAivMode());
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[InitCommRootInfo]errNo[0x%016llx] set aivMode error.", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        /* 设置only AIV模式 */
        ret = pComm->SetOnlyAivModeConfig(commConfig.GetConfigIsOnlyAivMode());
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[InitCommRootInfo]errNo[0x%016llx] set only aivMode error.", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        /* 设置AICPU */
        ret = pComm->SetAicpuUnfoldConfig(commConfig.GetConfigAicpuUnfold());
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[InitCommRootInfo]errNo[0x%016llx] set aicpu error.", HCCL_ERROR_CODE(ret)), errorFlag = true);

        /* 设置独立算子参数 */
        ret = pComm->SetIndependentOpConfig(commConfig, rankTable);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[InitCommRootInfo]errNo[0x%016llx] set SetIndependentOpConfig error.", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        /* 设置HcclExecTimeOut */
        ret = pComm->SetExecTimeOutConfig(commConfig.GetConfigExecTimeOut());
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[InitCommRootInfo]errNo[0x%016llx] set execTimeOut error.", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        /* 设置HcclAlgo */
        ret = pComm->SetAlgoConfig(commConfig.GetConfigHcclAlgoMap());
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[Init][CommClusterInfo]errNo[0x%016llx] set hcclAlgo error.", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        // 初始化完成的comm指针赋给出参
        *comm = pComm.get();
        std::unique_lock<std::mutex> lock(opBaseHcom.opGroupMapMutex);
        opBaseHcom.opGroup2CommMap[pComm->GetIdentifier()] = pComm;
        lock.unlock();

        // 特殊场景，当comm name被手动配置为HCCL_WORLD_GROUP时，需要将pComm赋值到hcomInfo.pComm
        if (pComm->GetIdentifier() == HCCL_WORLD_GROUP) {
            HcomGetCtxHomInfo().pComm = pComm;
        }

        ret = HcomSetGroupTopoInfo(pComm->GetIdentifier().c_str(), nRanks);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[InitCommRootInfo]errNo[0x%016llx] setGroupTopoInfo error", HCCL_ERROR_CODE(ret)),
            errorFlag = true);

        ret = pComm->InitHccpChannel();
        if (ret != HCCL_SUCCESS) {
            HCCL_WARNING("InitHccp channel unsuccessful ret:[%u].", ret);
        }
        if (hcclNslbDp::GetInstance().GetGlobalCommTaskId() != 0) {
            DevType nslb_devType;
            CHK_RET(hrtGetDeviceType(nslb_devType));
            if (nslb_devType == DevType::DEV_TYPE_910_93) {
                hcclNslbDp::GetInstance().SetDeviceType();
            }
            if (hcclNslbDp::GetInstance().InitNetCo() == HCCL_SUCCESS) {
                std::string identifier_nslb = commIdentifier;
                hcclNslbDp::GetInstance().InitCmmDesc(identifier_nslb);
                HCCL_INFO(
                    "nslb_InitCommRootInfo rankTable.rankList.size:[%zu], identifier_nslb[%s].",
                    rankTable.rankList.size(), identifier_nslb.c_str());
                hcclNslbDp::GetInstance().SetGlobalCommRankTable_RootInfo(
                    rankTable, localRankInfo, pComm->GetRankLists(), identifier_nslb, nRanks, rank);
                hcclNslbDp::GetInstance().SetGlobalDisRankTable(localRankInfo);
            } else {
                HCCL_WARNING("nslbdp try to init hccp failed.");
            }
        }
    } while (0);

    std::string defaultIdentifier = rootHandle.identifier;
    bool serverExist = opBaseHcom.hcclCommTopoInfoDetectServer.find(defaultIdentifier)
                       != opBaseHcom.hcclCommTopoInfoDetectServer.end();
    if (defaultIdentifier.compare(commIdentifier) != 0 && retryEnable && serverExist) {
        EXCEPTION_CATCH(
            opBaseHcom.hcclCommTopoInfoDetectServer.insert(
                {commIdentifier, opBaseHcom.hcclCommTopoInfoDetectServer[defaultIdentifier]}),
            return HCCL_E_MEMORY);
        EXCEPTION_CATCH(opBaseHcom.hcclCommTopoInfoDetectServer.erase(defaultIdentifier), return HCCL_E_MEMORY);
        HCCL_INFO(
            "[InitCommRootInfo] replace key of topoDetectServer from [%s] to [%s]", defaultIdentifier.c_str(),
            commIdentifier.c_str());
    } else if (!retryEnable && serverExist) {
        EXCEPTION_CATCH(opBaseHcom.hcclCommTopoInfoDetectServer.erase(defaultIdentifier), return HCCL_E_MEMORY);
        HCCL_INFO("[InitCommRootInfo] close topoDetectServer identifier[%s]", commIdentifier.c_str());
    }

    if (errorFlag) {
        HCCL_ERROR(
            "[InitCommRootInfo]Init failed, return[0x%016llx], rankNum[%u], rank[%u], "
            "rootInfo identifier[%s], server[%s], logicDevId[%d]",
            HCCL_ERROR_CODE(ret), nRanks, rank, commIdentifier.c_str(), GetLocalServerId(params.serverId).c_str(),
            params.logicDevId);
        (void)HcclCommDestroy(pComm.get());
        return ret;
    }

    HCCL_INFO(
        "[InitCommRootInfo]Init success, rankNum[%u], rank[%u], rootInfo identifier[%s], server[%s], "
        "logicDevId[%d]",
        nRanks, rank, commIdentifier.c_str(), params.serverId.c_str(), params.logicDevId);

    return HCCL_SUCCESS;
}

HcclResult HcclSetConfig(HcclConfig config, HcclConfigValue configValue)
{
    if (config == HCCL_DETERMINISTIC) {
        bool isSupportV2 = false;
        CHK_RET(hrtGetHcclV2Support(&isSupportV2));
        char* mmSysGetEnvValue = nullptr;
        MM_SYS_GET_ENV(MM_ENV_HCCL_DETERMINISTIC, mmSysGetEnvValue);
        std::string hcclDeterministicEnv = (mmSysGetEnvValue != nullptr) ? mmSysGetEnvValue : "EmptyString";
        if (isSupportV2) {
            CHK_RET(HcclSetConfigV2(config, configValue));
            if (!hcclDeterministicEnv.empty() && hcclDeterministicEnv != "EmptyString") {
                return HCCL_SUCCESS;
            }
        } else {
            if (hcclDeterministicEnv == "EmptyString") {
                if (configValue.value != DETERMINISTIC_STRICT && configValue.value != DETERMINISTIC_ENABLE
                    && configValue.value != DETERMINISTIC_DISABLE) {
                    HCCL_ERROR("[HcclSetConfig] HCCL_DETERMINISTIC is only support 0, 1 or 2");
                    return HCCL_E_PARA;
                } else {
                    DevType devType;
                    CHK_RET(hrtGetDeviceType(devType));
                    if (configValue.value == DETERMINISTIC_STRICT && devType != DevType::DEV_TYPE_910B
                        && devType != DevType::DEV_TYPE_910_93) {
                        HCCL_ERROR(
                            "[HcclSetConfig] configValue[%d], reduce order preservation is not supported for"
                            " devType[%d]",
                            configValue.value, devType);
                        return HCCL_E_NOT_SUPPORT;
                    }
                    CHK_RET(SetDeterministic(configValue.value));
                    HCCL_INFO("[HcclSetConfig] Set HCCL_DETERMINISTIC to %u", configValue.value);
                }
            } else {
                HCCL_WARNING("[HcclSetConfig] HCCL_DETERMINISTIC has been set by Env, so will not be reset again");
                return HCCL_SUCCESS;
            }
        }
        HcclOpInfoCtx& opBaseInfo = CollCommMgr::GetInstance().LegacyGetHcclOpInfoCtx();
        // 遍历所有的通信域设置其确定性计算配置参数
        for (auto it = opBaseInfo.opGroup2CommMap.begin(); it != opBaseInfo.opGroup2CommMap.end(); it++) {
            CHK_RET(it->second->SetDeterministicConfig(static_cast<u8>(configValue.value)));
        }
    }
    HCCL_RUN_INFO("Entry-HcclSetConfig successfully, config[%d], value[%u]", config, configValue.value);
    return HCCL_SUCCESS;
}

static HcclResult ResetDevice(hccl::hcclComm* hcclComm)
{
    s32 logicDeviceId = 0;
    CHK_RET(hcclComm->GetDeviceId(logicDeviceId));
    HcclSetThreadDeviceId(logicDeviceId);
    if (hcclComm->IsNeedResetDevice()) {
        HCCL_RUN_INFO("op_base com destroy, com is not global com");
        HCCL_RUN_INFO("[HcclCommDestroy] reset logicDeviceId[%d]", logicDeviceId);
        CHK_PRT_RET(
            hrtResetDevice(logicDeviceId) != HCCL_SUCCESS,
            HCCL_ERROR("[HcclCommDestroy] reset fail logicDeviceId[%d]", logicDeviceId), HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommDestroyWrapper([[maybe_unused]] struct hcclAsyncJob* job_)
{
    struct hcclCommDestroyAsyncJob* job = static_cast<hcclCommDestroyAsyncJob*>(job_);
    HcclComm comm = job->initComm;
    s32 devId = job->devId;
    HCCL_DEBUG("[HcclCommDestroyWrapper] Set device devId: %d", devId);
    CHK_PRT_RET(hrtSetDevice(devId) != HCCL_SUCCESS, HCCL_ERROR("[HcclCommDestroyWrapper] set fail"), HCCL_E_INTERNAL);
    HCCL_DEBUG("[HcclCommDestroyWrapper] Done Set device devId: %d", devId);

    HCCL_RUN_INFO("Entry-%s: op_base comm destroy begin", __func__);

    HcclUs startut = TIME_NOW();
    s32 deviceLogicId = 0;
    HcclResult ret = HcclDeviceRefresh(deviceLogicId);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[HcclCommDestroy] Get device fail, comm=%p", comm), ret);
    CHK_PRT_RET(comm == nullptr, HCCL_WARNING("[Destroy][HcclComm]An empty comm given, skip destroy."), HCCL_SUCCESS);
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
        HcclComm commV2 = hcclComm->GetCommunicatorV2();
        CHK_PTR_NULL(commV2);
        CHK_RET(HcclCommStateNotify(comm, HcclCommStatePhase::HCCL_COMM_STATE_PHASE_DESTROY_PRE));
        CHK_RET(HcclCommDestroyV2(
            commV2)); // 临时处理，dpustream的销毁要在其他资源销毁前完成。待新方案CpuThread上库后，原dpuStream删除可以恢复顺序
        string group = hcclComm->GetIdentifier();
        HcclOpInfoCtx& opBaseHcom = CollCommMgr::GetInstance().LegacyGetHcclOpInfoCtx();
        // 与同步路径一致，释放管理表锁后再通知回调，避免重入通信域管理接口时死锁。
        {
            std::unique_lock<std::mutex> lock(opBaseHcom.opGroupMapMutex);
            auto iter = opBaseHcom.opGroup2CommMap.find(group);
            if (iter != opBaseHcom.opGroup2CommMap.end()) {
                EXCEPTION_CATCH(opBaseHcom.opGroup2CommMap.erase(group), return HCCL_E_MEMORY);
            } else {
                HCCL_ERROR(
                    "[HcclCommDestroy] comm is not exist, comm=%p, group=%s, deviceLogicId=%d", comm, group.c_str(),
                    deviceLogicId);
                return HCCL_E_PARA;
            }
        }
        CHK_RET(HcclCommStateNotify(comm, HcclCommStatePhase::HCCL_COMM_STATE_PHASE_DESTROY_POST));
        return HCCL_SUCCESS;
    }());
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    HcclCommState state = hcclComm->GetState();
    if (state == HcclCommState::INUSE) {
        HCCL_WARNING("[HcclCommDestroy] comm is in use, please try again later");
        return HCCL_E_AGAIN;
    }
    hcclComm->DeinitZeroCopyMemoryAgent();
    HCCL_RUN_INFO("[HcclCommDestroy] comm state is %s", HcclCommStateToString(state));

    CHK_RET(hcclComm->SetStopFlag(true));
    CHK_RET(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE));
    CHK_RET(ResetDevice(hcclComm));

    if (IsOneSidedComm(comm)) {
        return HcclOneSidedCommDestroy(comm, deviceLogicId, startut);
    }

    HcclOpInfoCtx& opBaseHcom = CollCommMgr::GetInstance().LegacyGetHcclOpInfoCtx();
    string group;
    if (comm == opBaseHcom.pComm.get()) {
        group = opBaseHcom.pComm->GetIdentifier();
        opBaseHcom.pComm = nullptr;
        HcclCloseCommConnections(group);
    } else {
        HCCL_RUN_INFO("com is not global com");
        group = hcclComm->GetIdentifier();
    }

    // 特殊场景，当comm name被手动配置为HCCL_WORLD_GROUP时，需要将hcomInfo.pComm设为nullptr
    if (hcclComm->GetIdentifier() == HCCL_WORLD_GROUP) {
        HcomGetCtxHomInfo().pComm = nullptr;
    }

    HcomUnSetGroupTopoInfo(group.c_str());

    std::unique_lock<std::mutex> lock(opBaseHcom.opGroupMapMutex);
    auto iter = opBaseHcom.opGroup2CommMap.find(group);
    if (iter != opBaseHcom.opGroup2CommMap.end()) {
        EXCEPTION_CATCH(opBaseHcom.opGroup2CommMap.erase(group), return HCCL_E_MEMORY);
        HcclCloseCommConnections(group);
    } else {
        HCCL_ERROR(
            "[HcclCommDestroy] comm is not exist, comm=%p, group=%s, deviceLogicId=%d", comm, group.c_str(),
            deviceLogicId);
        return HCCL_E_PARA;
    }

    if (ProfilingManagerPub::GetAllState()) {
        ProfilingManagerPub::ClearStoragedProfilingInfo();
    }

    HcclUs endut = TIME_NOW();

    // 删除groupName和UDI的映射
    HCCL_PROFILER_DEL_GROUP_UDI(group);

    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "op_base comm destroy complete, take time [%lld]us, group[%s], deviceLogicId[%d].",
        DURATION_US(endut - startut), group.c_str(), deviceLogicId);
    return HCCL_SUCCESS;
}

HcclResult HcclCommDestroy([[maybe_unused]] HcclComm comm)
{
    if (hcclGroupDepth > 0) {
        std::shared_ptr<struct hcclCommDestroyAsyncJob> job;
        EXCEPTION_CATCH((job = std::make_shared<struct hcclCommDestroyAsyncJob>()), return HCCL_E_PARA);
        job->initComm = comm;
        s32 devId = 0;
        HcclResult ret = HcclDeviceRefresh(devId);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Group][HcclCommDestroy] Get device fail, comm=%p", comm), ret);
        job->devId = devId;
        ret = commInitTaskAppend(job, HcclCommDestroyWrapper, &comm);
        return ret;
    }
    HCCL_RUN_INFO("Entry-%s: op_base comm destroy begin", __func__);

    HcclUs startut = TIME_NOW();
    s32 deviceLogicId = 0;
    HcclResult ret = HcclDeviceRefresh(deviceLogicId);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[HcclCommDestroy] Get device fail, comm=%p", comm), ret);
    CHK_PRT_RET(comm == nullptr, HCCL_WARNING("[Destroy][HcclComm]An empty comm given, skip destroy."), HCCL_SUCCESS);

    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
        CHK_RET(HcclCommStateNotify(comm, HcclCommStatePhase::HCCL_COMM_STATE_PHASE_DESTROY_PRE));
        // 先拷贝orion通信域地址，避免coll comm销毁后无法获取
        HcclComm commV2 = hcclComm->GetCommunicatorV2();
        CHK_RET(HcclCommDestroyV2(
            commV2)); // 临时处理，dpustream的销毁要在其他资源销毁前完成。待新方案CpuThread上库后，原dpuStream删除可以恢复顺序
        string group = hcclComm->GetIdentifier();
        HcclOpInfoCtx& opBaseHcom = CollCommMgr::GetInstance().LegacyGetHcclOpInfoCtx();
        // opGroupMapMutex 仅覆盖对 opGroup2CommMap 的 find/erase 避免回调里销毁子通信域时同线程递归加锁死锁
        {
            std::unique_lock<std::mutex> lock(opBaseHcom.opGroupMapMutex);
            auto iter = opBaseHcom.opGroup2CommMap.find(group);
            if (iter != opBaseHcom.opGroup2CommMap.end()) {
                EXCEPTION_CATCH(opBaseHcom.opGroup2CommMap.erase(group), return HCCL_E_MEMORY);
            } else {
                HCCL_ERROR(
                    "[HcclCommDestroy] comm is not exist, comm=%p, group=%s, deviceLogicId=%d", comm, group.c_str(),
                    deviceLogicId);
                return HCCL_E_PARA;
            }
        }
        CHK_RET(HcclCommStateNotify(comm, HcclCommStatePhase::HCCL_COMM_STATE_PHASE_DESTROY_POST));
        HCCL_RUN_INFO(
            "Entry-HcclCommDestroy V2 group[%s] destroy success, deviceLogicId[%d], comm[%p]", group.c_str(),
            deviceLogicId, comm);
        return HCCL_SUCCESS;
    }());
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    HcclCommState state = hcclComm->GetState();
    if (state == HcclCommState::INUSE) {
        HCCL_WARNING("[HcclCommDestroy] comm is in use, please try again later");
        return HCCL_E_AGAIN;
    }
    hcclComm->DeinitZeroCopyMemoryAgent();
    HCCL_RUN_INFO("[HcclCommDestroy] comm state is %s", HcclCommStateToString(state));
    CHK_RET(hcclComm->RealeaseShareCCLbuffer());
    CHK_RET(hcclComm->SetStopFlag(true));
    CHK_RET(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE));
    CHK_RET(ResetDevice(hcclComm));

    std::unique_lock<std::mutex> oneSideLock(g_opHcomOneSideMutex);
    if (IsOneSidedComm(comm)) {
        return HcclOneSidedCommDestroy(comm, deviceLogicId, startut);
    }
    oneSideLock.unlock();

    HcclOpInfoCtx& opBaseHcom = CollCommMgr::GetInstance().LegacyGetHcclOpInfoCtx();
    string group;
    if (comm == opBaseHcom.pComm.get()) {
        group = opBaseHcom.pComm->GetIdentifier();
        opBaseHcom.pComm = nullptr;
        HcclCloseCommConnections(group);
    } else {
        HCCL_RUN_INFO("com is not global com");
        group = hcclComm->GetIdentifier();
    }

    // 特殊场景，当comm name被手动配置为HCCL_WORLD_GROUP时，需要将hcomInfo.pComm设为nullptr
    if (hcclComm->GetIdentifier() == HCCL_WORLD_GROUP) {
        HcomGetCtxHomInfo().pComm = nullptr;
    }

    HcomUnSetGroupTopoInfo(group.c_str());

    std::unique_lock<std::mutex> lock(opBaseHcom.opGroupMapMutex);
    auto iter = opBaseHcom.opGroup2CommMap.find(group);
    if (iter != opBaseHcom.opGroup2CommMap.end()) {
        EXCEPTION_CATCH(opBaseHcom.opGroup2CommMap.erase(group), return HCCL_E_MEMORY);
        HcclCloseCommConnections(group);
    } else {
        HCCL_ERROR(
            "[HcclCommDestroy] comm is not exist, comm=%p, group=%s, deviceLogicId=%d", comm, group.c_str(),
            deviceLogicId);
        return HCCL_E_PARA;
    }

    if (ProfilingManagerPub::GetAllState()) {
        ProfilingManagerPub::ClearStoragedProfilingInfo();
    }

    HcclUs endut = TIME_NOW();

    // 删除groupName和UDI的映射
    HCCL_PROFILER_DEL_GROUP_UDI(group);

    /* 关键状态记录 */
    HCCL_RUN_INFO(
        "Entry-HcclCommDestroy op_base comm destroy complete, take time [%lld]us, group[%s], deviceLogicId[%d].",
        DURATION_US(endut - startut), group.c_str(), deviceLogicId);
    return HCCL_SUCCESS;
}

// HcclDevMemAcquire 的 DPUTAG 分支：取址要 CollComm/ThreadMgr 完整类型，故落在本文件（仅 host 编译），
// 与 device 共编的 op_base_mc2.cc 只留声明
HcclResult HcclAcquireDpuTagShmemHost(HcclComm comm, uint64_t* size, void** addr, bool* newCreated)
{
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    hccl::CollComm* collComm = hcclComm->GetCollComm();
    CHK_PTR_NULL(collComm);
    // g_hcclDeviceId 是 thread_local 且初值 INVALID_INT，须先经 HcclGetThreadDeviceId 惰性初始化，
    // 否则会把 0xFFFFFFFF 当 devId 下发；此处不硬失败（kernel 已拉起时 devId 不被消费），
    // 兜底判参在 EnsureDpuKernelLaunched 的首次拉起分支
    s32 devId = HcclGetThreadDeviceId();
    if (devId == INVALID_INT) {
        HCCL_WARNING("[HcclDevMemAcquire] thread device id is unavailable, DPUTAG devId unset");
    }
    hccl::ThreadMgr* threadMgr = collComm->GetThreadMgr();
    CHK_PTR_NULL(threadMgr);
    hccl::DpuThreadService* dpuService = threadMgr->GetDpuThreadService();
    CHK_PTR_NULL(dpuService);
    return dpuService->AcquireDpuTagShmem(
        collComm->GetCommunicatorV2(), static_cast<uint32_t>(devId), size, addr, newCreated);
}
