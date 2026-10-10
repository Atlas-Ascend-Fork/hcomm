/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __OP_UNFOLD_CACHE_H__
#define __OP_UNFOLD_CACHE_H__

#include <unordered_map>

#include "aicpu_hccl_sqcqv2.h"
#include "op_unfold_key.h"
#include "op_unfold_cache_entry.h"
#include "rt_external_stars_define.h"

namespace hccl {

// 算子展开的动态缓存 (每个通信域单独维护一个动态缓存; 针对单算子模式的buffer copy和zero copy两种场景)
// 注意: 目前不考虑variable类型算子，也不考虑BatchSendRecv算子
// 注意: A3下scratch memory一定在HCCL buffer中, 所以目前不考虑scratch memory的更新
class OpUnfoldCache {
public:
    explicit OpUnfoldCache();
    ~OpUnfoldCache();

    bool IsCacheFull() const;

    HcclResult FindEntry(const OpUnfoldKey& key, OpUnfoldCacheEntry** entryPtrPtr)
        const; // 查看是否存在key对应的cache entry (如果不存在, *entryPtrPtr会被置为空)
    HcclResult AddEntry(
        const OpUnfoldKey& key, const std::vector<OpUnfoldMemRange>& userInputMemRanges,
        const std::vector<OpUnfoldMemRange>& userOutputMemRanges,
        OpUnfoldCacheEntry** entryPtrPtr);         // 插入新的cache entry
    HcclResult ClearEntry(const OpUnfoldKey& key); // 如果key存在对应的cache entry, 清理entry

    HcclResult ClearEntryForAlltoallv(); // 清理与alltoallv类算子相关的cache entry

    // 设置当前算子的图归属 (cache miss产生新entry时由AddEntry记录; eager算子传0)
    // 注意: OpUnfoldCache按通信域一对一创建, 同通信域算子串行执行, 无需加锁
    void SetCurCaptureModelId(const uint64_t modelId) { curCaptureModelId_ = modelId; }

    // aclgraph图销毁时清理该图capture期产生的cache entry (按entry记录的图归属modelId精确匹配)
    // 注意: capture期的SQE内容在图构建时被打散进图中, 图销毁后SQE绑定的流/任务上下文随之失效;
    //      若不清理, 下一张图构建时会按相同key命中并复用已销毁图缓存的SQE内容, 导致不可预期的错误
    HcclResult ClearEntryForCapture(const uint64_t modelId);

    // 只会在DEBUG_LEVEL下打印SQE内容 (通过比较打印算子正常展开的SQE与缓存的SQE, 判断刷新后的SQE是否正确)
    static HcclResult DumpSqeContent(const uint8_t* sqePtr, const uint8_t sqeType);

private:
    using CacheHashMap = std::unordered_map<OpUnfoldKey, OpUnfoldCacheEntry*>;

    // 只会在DEBUG_LEVEL下打印SQE header的内容
    static HcclResult DumpSqeHeader(const rtStarsSqeHeader_t& sqeHeader);
    static HcclResult DumpSqeHeader(const rtStarsSqeHeaderV2_t& sqeHeader);

    CacheHashMap cacheHashMap_; // key-entry mapping

    // 当前算子所属aclgraph的modelId (由AicpuCacheManager在算子执行前设置, AddEntry时记录到新entry)
    uint64_t curCaptureModelId_ = 0;
};

} // namespace hccl

#endif // __OP_UNFOLD_CACHE_H__
