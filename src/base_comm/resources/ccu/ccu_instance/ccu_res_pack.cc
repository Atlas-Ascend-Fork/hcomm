/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_res_pack.h"

#include "comm_engine/ccu/ccu_device_pub.h"
#include "ccu_device.h"
#include "ccu_log.h"
#include "ccu_res_repo.h"
#include "ccu_types.h"
#include "opbase_adpt.h"

namespace hcomm {

CcuResPack::~CcuResPack() { CcuReleaseRes(); }

void CcuResPack::CcuReleaseRes()
{
    if (resHandle_ != nullptr) {
        auto ret = CcuReleaseResHandle(userDevId_, resHandle_);
        if (ret != HcclResult::HCCL_SUCCESS) {
            HCCL_ERROR("[CcuResPack][%s] failed, resHandle[%p] userDevId[%d].", __func__, resHandle_, userDevId_);
        }
        resHandle_ = nullptr;
    }

    // 级联计数器块已随资源句柄在设备层统一释放，这里只清理本地快照
    std::lock_guard<std::mutex> lock(cascCntBlockMutex_);
    for (auto& blocks : cascCntBlocks_) {
        blocks.clear();
    }

    return;
}

CcuResult CcuResPack::Reset()
{
    if (!resHandle_) {
        return CcuResult::CCU_SUCCESS;
    }

    CCU_CHK_RET(CcuCheckResource(userDevId_, resHandle_, resRepo_));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuResPack::InitByInsType(const CcuInstanceType insType)
{
    userDevId_ = HcclGetThreadDeviceId();
    if (insType == CcuInstanceType::CCU_UNUSED) {
        HCCL_ERROR("[CcuResPack][%s] failed, error ccu instance type[%d].", __func__, static_cast<int32_t>(insType));
        return CcuResult::CCU_E_PARA;
    }

    // 根据通信域算子展开模式申请资源
    // 如果资源不足，返回HCCL_E_UNAVAIL，表示需要回退
    auto ret = CcuAllocResHandleByInsType(userDevId_, insType, resHandle_);
    if (ret == CcuResult::CCU_E_UNAVAIL) {
        HCCL_RUN_WARNING(
            "[%s] was unsuccessful but passed, resource is not enough, "
            "userDevId[%d], ccuInsType[%d].",
            __func__, userDevId_, insType);
        return ret;
    }
    CCU_CHK_RET(ret);
    CCU_CHK_RET(Reset());
    // 级联计数器块仅在 resDesc 路径按 die 申请, insType 路径不会返回 cascCnt 资源;
    // 此处刷新用于清理上一次初始化可能残留的快照, 保证快照与本次申请结果一致
    RefreshCascCntBlocks();
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuResPack::InitByResDescs(const CcuResDesc* descs[], uint32_t descNum)
{
    userDevId_ = HcclGetThreadDeviceId();

    // 基于 resDesc 驱动的资源数量申请资源
    // 如果资源不足，返回CCU_E_UNAVAIL，表示需要回退
    auto ret = CcuAllocResHandleByResDescs(userDevId_, descs, descNum, resHandle_);
    if (ret == CcuResult::CCU_E_UNAVAIL) {
        HCCL_RUN_WARNING(
            "[%s] was unsuccessful but passed, resource is not enough, "
            "userDevId[%d], descNum[%u].",
            __func__, userDevId_, descNum);
        return ret;
    }
    CCU_CHK_RET(ret);
    CCU_CHK_RET(Reset());
    RefreshCascCntBlocks();
    return CcuResult::CCU_SUCCESS;
}

void CcuResPack::RefreshCascCntBlocks()
{
    std::lock_guard<std::mutex> lock(cascCntBlockMutex_);
    for (uint8_t dieId = 0; dieId < CCU_MAX_IODIE_NUM; dieId++) {
        cascCntBlocks_[dieId].clear();
        for (const auto& cntXnBlock : resRepo_.cascCntBlock[dieId]) {
            cascCntBlocks_[dieId].emplace_back(cntXnBlock, false);
        }
    }
}

CcuResRepository& CcuResPack::GetCcuResRepo() { return resRepo_; }

CcuResult CcuResPack::AcquireCascCntBlock(uint8_t dieId, HcommCcuCascCntHandle handle)
{
    if (dieId >= CCU_MAX_IODIE_NUM) {
        HCCL_ERROR(
            "[CcuResPack][%s] failed, userDevId[%d], dieId[%u] is out of range[0, %u).", __func__, userDevId_, dieId,
            CCU_MAX_IODIE_NUM);
        return CcuResult::CCU_E_PARA;
    }
    // handle 由上层 CcuInstanceMgr 全局唯一分配后传入, 本类仅按其登记, 不再自增;
    // 0 恒为无效句柄(HcommCcuCascCntAlloc/GetMem 均以 0 表示无效)。
    if (handle == 0) {
        HCCL_ERROR("[CcuResPack][%s] failed, userDevId[%d], invalid cascCnt handle[0].", __func__, userDevId_);
        return CcuResult::CCU_E_PARA;
    }

    std::lock_guard<std::mutex> lock(cascCntBlockMutex_);
    auto& blocks = cascCntBlocks_[dieId];
    if (blocks.empty()) {
        HCCL_ERROR(
            "[CcuResPack][%s] failed, userDevId[%d], dieId[%u], cascCntBlocks is empty.", __func__, userDevId_, dieId);
        return CcuResult::CCU_E_UNAVAIL;
    }

    for (auto& cascCntBlock : blocks) {
        if (cascCntBlock.second) {
            continue;
        }
        cascCntBlock.second = true;
        std::unique_lock<std::shared_timed_mutex> lock(handleMapMutex_);
        EXCEPTION_CATCH(handleMap_.emplace(handle, cascCntBlock.first), return CcuResult::CCU_E_INTERNAL);
        return CcuResult::CCU_SUCCESS;
    }
    return CcuResult::CCU_E_UNAVAIL;
}

CcuResult CcuResPack::GetCascCntBlock(HcommCcuCascCntHandle handle, CntXnBlock& cascCntBlock)
{
    std::shared_lock<std::shared_timed_mutex> lock(handleMapMutex_);
    auto it = handleMap_.find(handle);
    if (it == handleMap_.end()) {
        HCCL_ERROR("[CcuResPack][%s] failed, handle[%llx] is not existed.", __func__, handle);
        return CcuResult::CCU_E_NOT_FOUND;
    }

    cascCntBlock = it->second;
    return CcuResult::CCU_SUCCESS;
}
} // namespace hcomm
