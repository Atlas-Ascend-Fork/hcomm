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
#include "hcomm_primitives.h"

#include <cstring>
#include <vector>

#include "dtype_common.h"
#include "common/dev_type_utils_pub.h"
#include "aicpu_task_cache_manager.h"
#include "log.h"

using namespace hcomm;

namespace {
constexpr uint32_t TASK_CACHE_LOCAL_MEM_COUNT = 2;
constexpr uint64_t TASK_CACHE_V_FORMAT_FLAG = uint64_t{1} << 63;

void DiscardUnsubmittedTaskCacheEntry(const char* tag)
{
    if (tag == nullptr || AicpuTaskCacheManager::cacheTag != tag || AicpuTaskCacheManager::isHit) {
        return;
    }
    if (AicpuTaskCacheManager::cacheEntryPtr != nullptr) {
        const HcclResult ret = AicpuTaskCacheManager::aicpuTaskCache.ClearEntry(tag, false);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[%s] ClearEntry failed, ret[%d]", __func__, ret);
        }
    }
    AicpuTaskCacheManager::cacheEntryPtr = nullptr;
    AicpuTaskCacheManager::cacheTag.clear();
    AicpuTaskCacheManager::isHit = false;
}

HcclResult ParseTaskCacheArgs(void** addrs, const uint64_t* sizes, uint64_t count, uint32_t& memCount, VOpInfo& vOpInfo)
{
    CHK_PTR_NULL(addrs);
    CHK_PTR_NULL(sizes);
    const bool isVFormat = (count & TASK_CACHE_V_FORMAT_FLAG) != 0;
    const uint64_t arrayCount = count & ~TASK_CACHE_V_FORMAT_FLAG;
    CHK_PRT_RET(arrayCount > UINT32_MAX, HCCL_ERROR("[%s] invalid array count", __func__), HCCL_E_PARA);
    memCount = static_cast<uint32_t>(arrayCount);
    if (!isVFormat) {
        return HCCL_SUCCESS; // 旧调用方的其余内存段仍按remote memory ranges处理。
    }
    CHK_PRT_RET(
        arrayCount <= TASK_CACHE_LOCAL_MEM_COUNT || (arrayCount - TASK_CACHE_LOCAL_MEM_COUNT) % 2 != 0,
        HCCL_ERROR("[%s] invalid V array count", __func__), HCCL_E_PARA);
    const uint32_t rankSize = (memCount - TASK_CACHE_LOCAL_MEM_COUNT) / 2;
    std::vector<uint64_t>* counts[] = {&vOpInfo.sendCounts, &vOpInfo.recvCounts};
    std::vector<uint64_t>* displs[] = {&vOpInfo.sdispls, &vOpInfo.rdispls};
    // HCOMM没有data type信息：sizes已经是字节数，地址相减得到的displs也是字节偏移。
    for (uint32_t side = 0; side < TASK_CACHE_LOCAL_MEM_COUNT; ++side) {
        const uint64_t base = reinterpret_cast<uint64_t>(addrs[side]);
        CHK_PRT_RET(sizes[side] > UINT64_MAX - base, HCCL_ERROR("[%s] invalid user buffer", __func__), HCCL_E_PARA);
        counts[side]->reserve(rankSize);
        displs[side]->reserve(rankSize);
        for (uint32_t rank = 0; rank < rankSize; ++rank) {
            const uint32_t index = TASK_CACHE_LOCAL_MEM_COUNT + side * rankSize + rank;
            const uint64_t addr = reinterpret_cast<uint64_t>(addrs[index]);
            // 零数据侧可传空buffer；size可能包含零count项保留的displ，需逐peer检查count。
            CHK_PRT_RET(
                base == 0 && sizes[index] != 0, HCCL_ERROR("[%s] nonzero V count requires a user buffer", __func__),
                HCCL_E_PARA);
            CHK_PRT_RET(
                addr < base || sizes[index] > UINT64_MAX - addr, HCCL_ERROR("[%s] invalid V address", __func__),
                HCCL_E_PARA);
            const uint64_t offset = addr - base;
            CHK_PRT_RET(
                sizes[index] != 0 && (offset > sizes[side] || sizes[index] > sizes[side] - offset),
                HCCL_ERROR("[%s] V range exceeds user buffer", __func__), HCCL_E_PARA);
            counts[side]->push_back(sizes[index]);
            displs[side]->push_back(offset);
        }
    }
    // 后续项只是V元数据，不能再次作为remote memory ranges参与地址匹配或token刷新。
    memCount = TASK_CACHE_LOCAL_MEM_COUNT;
    return HCCL_SUCCESS;
}
} // namespace

HcommResult HcommAicpuTsTaskCacheLookup(const char* tag, bool* isHit)
{
    auto isGe950 = hcomm::EqualToOrGreaterThan950();
    CHK_PRT_RET(!isGe950.has_value(), HCCL_ERROR("[%s] get device type failed.", __func__), HCCL_E_RUNTIME);
    if (!(*isGe950)) {
        HCCL_ERROR("[%s] deviceType is not supported, only DEV_TYPE_950 and later supported", __func__);
        return HCCL_E_NOT_SUPPORT;
    }

    CHK_PTR_NULL(tag);
    CHK_PTR_NULL(isHit);

    // 记录当前tag, 后续校验tag一致性
    AicpuTaskCacheManager::cacheTag = tag;

    // 注意: 相同tag的算子一定不会被多个aicpu kernel threads同时展开,
    // 否则后续threads可能命中第一个thread插入的不完整的cache entry 例如: tag中有commId时,
    // 相同tag的算子一定属于同一个通信域, 必定按序展开 因此, FindEntry和AddEntry不需要统一成单个接口, 即FindEntry后,
    // 当前tag的缓存状态不会被其他threads改变, 可以直接AddEntry

    // 查询tag对应的cache entry
    AicpuTaskCacheManager::cacheEntryPtr = nullptr;
    CHK_RET(AicpuTaskCacheManager::aicpuTaskCache.FindEntry(tag, &AicpuTaskCacheManager::cacheEntryPtr));

    // 判断并记录是否为cache hit
    AicpuTaskCacheManager::isHit = (AicpuTaskCacheManager::cacheEntryPtr != nullptr);
    *isHit = AicpuTaskCacheManager::isHit;

    // 如果是cache miss, 尝试添加cache entry
    // 注意: 如果aicpu task cache容量已满, 不会添加新的cache entry, AicpuTaskCacheManager::cacheEntryPtr被设置为nullptr
    if (!(*isHit)) {
        CHK_RET(AicpuTaskCacheManager::aicpuTaskCache.AddEntry(tag, &AicpuTaskCacheManager::cacheEntryPtr));
    }

    return HCCL_SUCCESS;
}

HcommResult HcommAicpuTsTaskCacheStart(const char* tag, void** addrs, uint64_t* sizes, uint64_t count)
{
    auto isGe950 = hcomm::EqualToOrGreaterThan950();
    CHK_PRT_RET(!isGe950.has_value(), HCCL_ERROR("[%s] get device type failed.", __func__), HCCL_E_RUNTIME);
    if (!(*isGe950)) {
        HCCL_ERROR("[HcommAicpuTsTaskCacheStart] deviceType is not supported, only DEV_TYPE_950 and later supported");
        return HCCL_E_NOT_SUPPORT;
    }

    CHK_PTR_NULL(tag);
    if (addrs == nullptr || sizes == nullptr) {
        DiscardUnsubmittedTaskCacheEntry(tag);
        return HCCL_E_PTR;
    }

    // 校验tag一致性 (即Submit对应的tag一定与Lookup对应的tag一致)
    CHK_PRT_RET(
        strcmp(AicpuTaskCacheManager::cacheTag.c_str(), tag) != 0,
        HCCL_ERROR(
            "[HcommAicpuTsTaskCacheStart] submit's tag[%s] != lookup's tag[%s]", tag,
            AicpuTaskCacheManager::cacheTag.c_str()),
        HCCL_E_PARA);

    // 一定是cache miss
    CHK_PRT_RET(
        AicpuTaskCacheManager::isHit, HCCL_ERROR("[HcommAicpuTsTaskCacheStart] cache hit, but should be miss"),
        HCCL_E_INTERNAL);

    VOpInfo vOpInfo;
    uint32_t memCount = 0;
    HcclResult ret = ParseTaskCacheArgs(addrs, sizes, count, memCount, vOpInfo);
    if (ret != HCCL_SUCCESS) {
        DiscardUnsubmittedTaskCacheEntry(tag);
        return ret;
    }

    // Aicpu task cache容量未满
    if (AicpuTaskCacheManager::cacheEntryPtr != nullptr) {
        // 保存地址信息到cache entry
        ret = AicpuTaskCacheManager::cacheEntryPtr->InitCacheEntry(
            ReinterpretAs<const uint64_t*>(addrs), sizes, memCount, vOpInfo);
        if (ret != HCCL_SUCCESS) {
            DiscardUnsubmittedTaskCacheEntry(tag);
            return ret;
        }
    }

    return HCCL_SUCCESS;
}

HcommResult HcommAicpuTsTaskCacheEnd(const char* tag)
{
    auto isGe950 = hcomm::EqualToOrGreaterThan950();
    CHK_PRT_RET(!isGe950.has_value(), HCCL_ERROR("[%s] get device type failed.", __func__), HCCL_E_RUNTIME);
    if (!(*isGe950)) {
        HCCL_ERROR("[HcommAicpuTsTaskCacheEnd] deviceType is not supported, only DEV_TYPE_950 and later supported");
        return HCCL_E_NOT_SUPPORT;
    }

    CHK_PTR_NULL(tag);

    // 校验tag一致性 (即Submit对应的tag一定与Lookup对应的tag一致)
    CHK_PRT_RET(
        strcmp(AicpuTaskCacheManager::cacheTag.c_str(), tag) != 0,
        HCCL_ERROR(
            "[HcommAicpuTsTaskCacheEnd] submit's tag[%s] != lookup's tag[%s]", tag,
            AicpuTaskCacheManager::cacheTag.c_str()),
        HCCL_E_PARA);

    // 一定是cache miss
    CHK_PRT_RET(
        AicpuTaskCacheManager::isHit, HCCL_ERROR("[HcommAicpuTsTaskCacheEnd] cache hit, but should be miss"),
        HCCL_E_INTERNAL);

    HcclResult ret = HCCL_SUCCESS;
    do {
        // Aicpu task cache容量未满
        if (AicpuTaskCacheManager::cacheEntryPtr != nullptr) {
            // 提交cache entry, 更新cache entry内部信息
            ret = AicpuTaskCacheManager::cacheEntryPtr->SubmitCacheEntry();
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR("[HcommAicpuTsTaskCacheEnd] SubmitCacheEntry error,ret[%d]", ret);
                // 残缺entry不可被后续Lookup命中，且尚未计入cacheBytes_。
                const HcclResult clearRet = AicpuTaskCacheManager::aicpuTaskCache.ClearEntry(tag, false);
                if (clearRet != HCCL_SUCCESS) {
                    HCCL_ERROR("[HcommAicpuTsTaskCacheEnd] ClearEntry error,ret[%d]", clearRet);
                }
                break;
            }

            // 更新cache空间消耗
            const uint64_t entryBytes = AicpuTaskCacheManager::cacheEntryPtr->GetEntryBytes();
            ret = AicpuTaskCacheManager::aicpuTaskCache.IncCacheBytes(
                AicpuTaskCacheManager::cacheTag.c_str(), entryBytes);
            CHK_PRT_BREAK(
                ret != HCCL_SUCCESS, HCCL_ERROR("[HcommAicpuTsTaskCacheEnd] IncCacheBytes error, ret[%d]", ret),
                (void)0);
        }
    } while (0);

    // 重置cache上下文
    AicpuTaskCacheManager::cacheTag.clear();
    AicpuTaskCacheManager::isHit = false;
    AicpuTaskCacheManager::cacheEntryPtr = nullptr;

    return ret;
}

HcommResult HcommAicpuTsTaskCacheExecute(const char* tag, void** addrs, uint64_t* sizes, uint64_t count)
{
    auto isGe950 = hcomm::EqualToOrGreaterThan950();
    CHK_PRT_RET(!isGe950.has_value(), HCCL_ERROR("[%s] get device type failed.", __func__), HCCL_E_RUNTIME);
    if (!(*isGe950)) {
        HCCL_ERROR("[HcommAicpuTsTaskCacheExecute] deviceType is not supported, only DEV_TYPE_950 and later supported");
        return HCCL_E_NOT_SUPPORT;
    }

    CHK_PTR_NULL(tag);
    CHK_PTR_NULL(addrs);
    CHK_PTR_NULL(sizes);
    VOpInfo vOpInfo;
    uint32_t memCount = 0;
    CHK_RET(ParseTaskCacheArgs(addrs, sizes, count, memCount, vOpInfo));

    // 校验tag一致性 (即Submit对应的tag一定与Lookup对应的tag一致)
    CHK_PRT_RET(
        strcmp(AicpuTaskCacheManager::cacheTag.c_str(), tag) != 0,
        HCCL_ERROR(
            "[HcommAicpuTsTaskCacheExecute] submit's tag[%s] != lookup's tag[%s]", tag,
            AicpuTaskCacheManager::cacheTag.c_str()),
        HCCL_E_PARA);

    // 一定是cache hit
    CHK_PRT_RET(
        !AicpuTaskCacheManager::isHit, HCCL_ERROR("[HcommAicpuTsTaskCacheExecute] cache miss, but should be hit"),
        HCCL_E_INTERNAL);

    // cache hit一定存在对应的cache entry
    CHK_PTR_NULL(AicpuTaskCacheManager::cacheEntryPtr);

    // 刷新并下发task
    HcclResult ret = AicpuTaskCacheManager::cacheEntryPtr->RefreshAndLaunch(
        ReinterpretAs<const uint64_t*>(addrs), sizes, memCount, vOpInfo);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HcommAicpuTsTaskCacheExecute] RefreshAndLaunch error, ret[%d]", ret);
    }

    // 重置cache上下文
    AicpuTaskCacheManager::cacheTag.clear();
    AicpuTaskCacheManager::isHit = false;
    AicpuTaskCacheManager::cacheEntryPtr = nullptr;

    return ret;
}

HcommResult HcommAicpuTsTaskCacheClear(const char* tag)
{
    auto isGe950 = hcomm::EqualToOrGreaterThan950();
    CHK_PRT_RET(!isGe950.has_value(), HCCL_ERROR("[%s] get device type failed.", __func__), HCCL_E_RUNTIME);
    if (!(*isGe950)) {
        HCCL_ERROR("[%s] deviceType is not supported, only DEV_TYPE_950 and later supported", __func__);
        return HCCL_E_NOT_SUPPORT;
    }

    CHK_PTR_NULL(tag);

    // 清除tag对应的cache entry (if any)
    CHK_RET(AicpuTaskCacheManager::aicpuTaskCache.ClearEntry(tag));
    if (strcmp(AicpuTaskCacheManager::cacheTag.c_str(), tag) == 0) {
        // 重置cache上下文
        AicpuTaskCacheManager::cacheTag.clear();
        AicpuTaskCacheManager::isHit = false;
        AicpuTaskCacheManager::cacheEntryPtr = nullptr;
    }

    return HCCL_SUCCESS;
}
