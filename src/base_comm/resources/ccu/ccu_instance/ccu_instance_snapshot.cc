/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_instance.h"
#include "ccu_device.h"
#include "ccu_instance_mgr.h"
#include "hccl_common.h"

#include <utility>

#include "comm_engine/ccu/ccu_device_res_pub.h"
#include "ccu_dev_mgr_imp.h"
#include "hcomm_res_ccu_entity.h"
#include "ccu_res_repo.h"

namespace hcomm {
namespace {

    // 内部资源仓储与跨 SO 快照的 die 容量必须一致；分属实现层与 ABI 层，不能共享定义，只锚定数值。
    static_assert(
        CCU_MAX_IODIE_NUM == HCOMM_CCU_MAX_DIE_NUM, "CCU_MAX_IODIE_NUM and HCOMM_CCU_MAX_DIE_NUM must stay consistent");

    using DieRanges = std::array<std::vector<ResInfo>, CCU_MAX_IODIE_NUM>;

    // HcclResult 与 CcuResult 为两个错误码契约，不能按数值直转；未定义的底层错误统一收敛为内部错误。
    CcuResult ConvertHcclResultToCcuResult(HcclResult ret)
    {
        switch (ret) {
            case HcclResult::HCCL_SUCCESS:
                return CcuResult::CCU_SUCCESS;
            case HcclResult::HCCL_E_PARA:
                return CcuResult::CCU_E_PARA;
            case HcclResult::HCCL_E_PTR:
                return CcuResult::CCU_E_PTR;
            case HcclResult::HCCL_E_INTERNAL:
                return CcuResult::CCU_E_INTERNAL;
            case HcclResult::HCCL_E_NOT_SUPPORT:
                return CcuResult::CCU_E_NOT_SUPPORT;
            case HcclResult::HCCL_E_NOT_FOUND:
                return CcuResult::CCU_E_NOT_FOUND;
            case HcclResult::HCCL_E_UNAVAIL:
                return CcuResult::CCU_E_UNAVAIL;
            case HcclResult::HCCL_E_RUNTIME:
                return CcuResult::CCU_E_RUNTIME;
            default:
                return CcuResult::CCU_E_INTERNAL;
        }
    }

    // 统一记录底层查询失败上下文，并将错误码收敛到 CcuResult。
    CcuResult CheckMetadataQuery(HcclResult ret, const char* action, int32_t deviceLogicId, uint32_t dieId)
    {
        if (ret == HcclResult::HCCL_SUCCESS) {
            return CcuResult::CCU_SUCCESS;
        }
        HCCL_WARNING(
            "[CcuInstance][FillMetadata] %s failed, devLogicId[%d] dieId[%u] ret[%d].", action, deviceLogicId, dieId,
            ret);
        return ConvertHcclResultToCcuResult(ret);
    }

    CcuResult
    FillResourceSpaceToken(int32_t deviceLogicId, uint32_t dieId, uint64_t resSpaceSize, HcommCcuDieMetadata& metadata);

    CcuResult FillMetadata(int32_t deviceLogicId, uint32_t dieId, HcommCcuDieMetadata& metadata)
    {
        metadata = HcommCcuDieMetadata{};
        metadata.header = {HCOMM_CCU_RES_ABI_VERSION, HCOMM_CCU_DIE_METADATA_MAGIC_WORD, sizeof(metadata), 0};
        metadata.enabled = 1;

        auto ret = CcuDevMgrImp::GetMissionKey(deviceLogicId, dieId, metadata.missionKey);
        CCU_CHK_RET(CheckMetadataQuery(ret, "get mission key", deviceLogicId, dieId));

        uint64_t xnBaseAddr = 0;
        ret = CcuDevMgrImp::GetXnBaseAddr(deviceLogicId, dieId, xnBaseAddr);
        CCU_CHK_RET(CheckMetadataQuery(ret, "get xn base address", deviceLogicId, dieId));

        ret = CcuDevMgrImp::GetLoopChannelId(deviceLogicId, dieId, dieId, metadata.loopChannelId);
        CCU_CHK_RET(CheckMetadataQuery(ret, "get loop channel id", deviceLogicId, dieId));

        uint64_t resSpaceSize = 0;
        ret = CcuDevMgrImp::GetCcuResourceSpaceBufInfo(
            deviceLogicId, dieId, metadata.ccuResBuffer.bufferInfo.rma.addr, resSpaceSize);
        CCU_CHK_RET(CheckMetadataQuery(ret, "get resource space buffer", deviceLogicId, dieId));

        if (metadata.ccuResBuffer.bufferInfo.rma.addr == 0) {
            metadata.ccuResBuffer.bufferInfo.rma.addr = xnBaseAddr;
        }
        return FillResourceSpaceToken(deviceLogicId, dieId, resSpaceSize, metadata);
    }

    CcuResult
    FillResourceSpaceToken(int32_t deviceLogicId, uint32_t dieId, uint64_t resSpaceSize, HcommCcuDieMetadata& metadata)
    {
        uint64_t tokenId = 0;
        uint64_t tokenValue = 0;
        HcclResult ret = CcuDevMgrImp::GetCcuResourceSpaceTokenInfo(deviceLogicId, dieId, tokenId, tokenValue);
        if (ret != HcclResult::HCCL_SUCCESS) {
            HCCL_WARNING(
                "[CcuInstance][FillResourceSpaceToken] get resource space token failed, devLogicId[%d] dieId[%u] "
                "ret[%d].",
                deviceLogicId, dieId, ret);
            return ConvertHcclResultToCcuResult(ret);
        }
        metadata.ccuResBuffer.type = REGED_BUFFER_RMA;
        metadata.ccuResBuffer.bufferInfo.rma.size = resSpaceSize;
        metadata.ccuResBuffer.bufferInfo.rma.protectionInfo.type = PROTECTION_TYPE_UB;
        metadata.ccuResBuffer.bufferInfo.rma.protectionInfo.memInfo.ub.tokenId = static_cast<uint32_t>(tokenId);
        metadata.ccuResBuffer.bufferInfo.rma.protectionInfo.memInfo.ub.tokenValue = static_cast<uint32_t>(tokenValue);
        return CcuResult::CCU_SUCCESS;
    }

} // namespace

CcuResult CcuInstance::GetSnapshotCcuVersion(HcommCcuVersion& ccuVersion)
{
    ccuVersion = HCOMM_CCU_VERSION_INVALID;
    CcuVersion version = CcuVersion::INVALID;
    HcclResult ret = CcuDevMgrImp::GetCcuVersion(userDevId_, version);
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_WARNING("[CcuInstance][%s] query ccu version failed, userDevId[%u] ret[%d].", __func__, userDevId_, ret);
        return ConvertHcclResultToCcuResult(ret);
    }
    if (version == CcuVersion::INVALID || version == CcuVersion::CCU_INVALID) {
        HCCL_WARNING(
            "[CcuInstance][%s] ccu version is invalid, userDevId[%u] version[%s].", __func__, userDevId_,
            version.Describe().c_str());
        return CcuResult::CCU_E_UNAVAIL;
    }

    // ABI 版本值不依赖内部枚举编号，显式映射避免后续内部枚举调整破坏跨 SO 契约。
    if (version == CcuVersion::CCU_V1) {
        ccuVersion = HCOMM_CCU_VERSION_V1;
    } else if (version == CcuVersion::CCU_V2) {
        ccuVersion = HCOMM_CCU_VERSION_V2;
    } else {
        return CcuResult::CCU_E_UNAVAIL;
    }
    return CcuResult::CCU_SUCCESS;
}

void CcuInstance::ResetSnapshotRanges()
{
    for (auto& die : snapshotRanges_) {
        for (auto& ranges : die) {
            ranges.clear();
        }
    }
}

void CcuInstance::FillSnapshotDieRanges(HcommCcuDieRes& dieRes, const CcuResRepository& repo, uint32_t dieId)
{
    const std::array<const DieRanges*, HCOMM_CCU_BATCH_RES_TYPE_COUNT> groups = {{
        &repo.loopEngine,
        &repo.blockLoopEngine,
        &repo.ms,
        &repo.blockMs,
        &repo.cke,
        &repo.blockCke,
        &repo.blockXn,
        &repo.xn,
        &repo.gsa,
        &repo.blockGsa,
        &repo.mission.mission,
    }};

    for (uint32_t type = 0; type < groups.size(); ++type) {
        auto& ranges = snapshotRanges_[dieId][type];
        for (const ResInfo& range : (*groups[type])[dieId]) {
            ranges.push_back({range.startId, range.num, 0});
        }

        HcommCcuDieResRangePack& view = dieRes.resTypes[type];
        view.resourceType = type;
        if (!ranges.empty()) {
            view.resRanges = ranges.data();
            view.resRangeNum = static_cast<uint32_t>(ranges.size());
            dieRes.resTypeNum = type + 1;
        }
    }
}

CcuResult CcuInstance::FillSnapshotDie(
    HcommCcuInstanceEntity& snapshot, const CcuResRepository& repo, uint32_t dieId, bool& enabled)
{
    enabled = false;
    HcommCcuDieRes& dieRes = snapshot.ccuDieReses[dieId];
    dieRes.dieId = dieId;

    CcuResult ret = CcuGetDieEnableInfo(userDevId_, static_cast<uint8_t>(dieId), enabled);
    if (ret != CcuResult::CCU_SUCCESS) {
        return ret;
    }
    if (!enabled) {
        return CcuResult::CCU_SUCCESS;
    }

    ret = FillMetadata(userDevId_, dieId, snapshotMetadata_[dieId]);
    if (ret != CcuResult::CCU_SUCCESS) {
        return ret;
    }
    FillSnapshotDieRanges(dieRes, repo, dieId);
    dieRes.dieMetadata = &snapshotMetadata_[dieId];
    return CcuResult::CCU_SUCCESS;
}

void CcuInstance::FillSnapshotControlSlots(HcommCcuInstanceEntity& snapshot)
{
    snapshot.ascCustom[HCOMM_CCU_CONTROL_SLOT_CHANNEL_ENTITY] = reinterpret_cast<uint64_t>(HcommCcuChannelGetEntity);
    snapshot.ascCustom[HCOMM_CCU_CONTROL_SLOT_ALLOC_INST_SPACE] = reinterpret_cast<uint64_t>(HcommCcuAllocInstSpace);
    snapshot.ascCustom[HCOMM_CCU_CONTROL_SLOT_SUBMIT_INSTS] = reinterpret_cast<uint64_t>(HcommCcuSubmitInsts);
}

CcuResult
CcuInstance::FillSnapshotDies(HcommCcuInstanceEntity& snapshot, const CcuResRepository& repo, uint32_t& enabledDieNum)
{
    enabledDieNum = 0;
    ResetSnapshotRanges();
    try {
        for (uint32_t dieId = 0; dieId < HCOMM_CCU_MAX_DIE_NUM; ++dieId) {
            bool enabled = false;
            CcuResult ret = FillSnapshotDie(snapshot, repo, dieId, enabled);
            if (ret != CcuResult::CCU_SUCCESS) {
                HCCL_ERROR(
                    "[CcuInstance][%s] failed to fill die snapshot, userDevId[%u] insHandle[%llu] dieId[%u] ret[%d].",
                    __func__, userDevId_, static_cast<unsigned long long>(insHandle_), dieId, ret);
                return ret;
            }
            enabledDieNum += enabled ? 1U : 0U;
        }
    } catch (...) {
        HCCL_ERROR(
            "[CcuInstance][%s] failed, caught exception while building snapshot, userDevId[%u] insHandle[%llu].",
            __func__, userDevId_, static_cast<unsigned long long>(insHandle_));
        return CcuResult::CCU_E_INTERNAL;
    }
    return CcuResult::CCU_SUCCESS;
}

// 构建跨 SO Instance Entity；注册/预约后的剩余资源由 asc-comm 本地仓维护。
CcuResult CcuInstance::BuildHcommCcuInstanceEntity()
{
    if (snapshotReady_) {
        return CcuResult::CCU_SUCCESS;
    }
    if (resPack_ == nullptr || insHandle_ == 0) {
        HCCL_ERROR(
            "[CcuInstance][%s] failed, resource pack not ready, userDevId[%u] insHandle[%llu] resPackNull[%d].",
            __func__, userDevId_, static_cast<unsigned long long>(insHandle_), resPack_ == nullptr ? 1 : 0);
        return CcuResult::CCU_E_UNAVAIL;
    }

    HcommCcuVersion ccuVersion = HCOMM_CCU_VERSION_INVALID;
    CcuResult ret = GetSnapshotCcuVersion(ccuVersion);
    if (ret != CcuResult::CCU_SUCCESS) {
        HCCL_ERROR(
            "[CcuInstance][%s] failed to get ccu version, userDevId[%u] insHandle[%llu] ret[%d].", __func__, userDevId_,
            static_cast<unsigned long long>(insHandle_), ret);
        return ret;
    }

    HcommCcuInstanceEntity snapshot{};
    HcommCcuInstanceEntityInit(&snapshot);
    snapshot.ccuDieNum = HCOMM_CCU_MAX_DIE_NUM;
    snapshot.ccuVersion = ccuVersion;

    uint32_t enabledDieNum = 0;
    ret = FillSnapshotDies(snapshot, resPack_->GetCcuResRepo(), enabledDieNum);
    if (ret != CcuResult::CCU_SUCCESS) {
        return ret;
    }
    if (enabledDieNum == 0) {
        HCCL_ERROR(
            "[CcuInstance][%s] failed, no enabled die, userDevId[%u] insHandle[%llu].", __func__, userDevId_,
            static_cast<unsigned long long>(insHandle_));
        return CcuResult::CCU_E_UNAVAIL;
    }

    FillSnapshotControlSlots(snapshot);
    snapshot.ccuInsKey = insHandle_;
    snapshot_ = snapshot;
    snapshotReady_ = true;
    return CcuResult::CCU_SUCCESS;
}

} // namespace hcomm
