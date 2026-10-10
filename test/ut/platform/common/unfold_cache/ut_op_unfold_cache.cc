/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>
#include <string>
#include <vector>

#include "op_unfold_cache.h"

using namespace std;
using namespace hccl;

class OpUnfoldCacheTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "OpUnfoldCacheTest SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "OpUnfoldCacheTest TearDown" << std::endl; }
    virtual void SetUp() { std::cout << "A Test SetUP" << std::endl; }
    virtual void TearDown() { std::cout << "A Test TearDown" << std::endl; }

    // 构造测试key: inputSize区分不同key, isCapture区分capture/eager key
    // 注意: OpUnfoldKey仅提供explicit拷贝构造且无移动构造, 按值返回会编译失败, 采用输出参数风格
    static void MakeKey(const uint64_t inputSize, const bool isCapture, OpUnfoldKey& key)
    {
        key.Init(
            HcclCMDType::HCCL_CMD_ALLREDUCE, HcclDataType::HCCL_DATA_TYPE_FP32, HcclReduceOp::HCCL_REDUCE_SUM, false,
            false, inputSize, false, HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE, isCapture, /*root*/ 0);
    }

    // 构造测试mem range (AddEntry入参, 内容不影响归属/清理逻辑)
    static vector<OpUnfoldMemRange> MakeMemRanges() { return {OpUnfoldMemRange(0x1000, 1024)}; }
};

// ========== 归属记录层: capture期AddEntry时记录图归属 ==========

// capture算子产生的entry应记录当前图归属modelId
TEST_F(OpUnfoldCacheTest, ut_AddEntry_CaptureKey_Expect_RecordModelId)
{
    OpUnfoldCache cache;
    const uint64_t modelId = 100;

    cache.SetCurCaptureModelId(modelId);

    OpUnfoldCacheEntry* entryPtr = nullptr;
    OpUnfoldKey key;
    MakeKey(1024, true, key);
    HcclResult ret = cache.AddEntry(key, MakeMemRanges(), MakeMemRanges(), &entryPtr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ASSERT_NE(entryPtr, nullptr);
    EXPECT_EQ(entryPtr->GetCaptureModelId(), modelId);
}

// eager算子(SetCurCaptureModelId(0))产生的entry归属应为0
TEST_F(OpUnfoldCacheTest, ut_AddEntry_EagerKey_Expect_RecordZeroModelId)
{
    OpUnfoldCache cache;

    cache.SetCurCaptureModelId(0); // eager算子归属为0

    OpUnfoldCacheEntry* entryPtr = nullptr;
    OpUnfoldKey key;
    MakeKey(1024, false, key);
    HcclResult ret = cache.AddEntry(key, MakeMemRanges(), MakeMemRanges(), &entryPtr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ASSERT_NE(entryPtr, nullptr);
    EXPECT_EQ(entryPtr->GetCaptureModelId(), 0u);
}

// eager残留防护: capture算子(modelId=100)之后紧跟eager算子, eager entry归属必须是0而非残留的100
TEST_F(OpUnfoldCacheTest, ut_SetCurCaptureModelId_EagerAfterCapture_Expect_ResetToZero)
{
    OpUnfoldCache cache;

    // 先作为图A(100)的capture算子
    cache.SetCurCaptureModelId(100);
    OpUnfoldCacheEntry* captureEntryPtr = nullptr;
    OpUnfoldKey captureKey;
    MakeKey(1024, true, captureKey);
    ASSERT_EQ(cache.AddEntry(captureKey, MakeMemRanges(), MakeMemRanges(), &captureEntryPtr), HCCL_SUCCESS);
    EXPECT_EQ(captureEntryPtr->GetCaptureModelId(), 100u);

    // 再作为eager算子 (对应LookupOpUnfoldCache中 isCapture ? modelId : 0 的语义)
    cache.SetCurCaptureModelId(0);
    OpUnfoldCacheEntry* eagerEntryPtr = nullptr;
    OpUnfoldKey eagerKey;
    MakeKey(2048, false, eagerKey);
    ASSERT_EQ(cache.AddEntry(eagerKey, MakeMemRanges(), MakeMemRanges(), &eagerEntryPtr), HCCL_SUCCESS);
    EXPECT_EQ(eagerEntryPtr->GetCaptureModelId(), 0u);
}

// ========== 精确清理层: 图销毁时按归属清理 ==========

// 核心用例: 图A/图B/eager三种entry共存, 销毁图A只清A的entry, 图B和eager不受影响
TEST_F(OpUnfoldCacheTest, ut_ClearEntryForCapture_Expect_OnlyMatchingModelIdCleared)
{
    OpUnfoldCache cache;

    // 图A(100)的capture entry
    cache.SetCurCaptureModelId(100);
    OpUnfoldKey keyA;
    MakeKey(1024, true, keyA);
    OpUnfoldCacheEntry* entryAPtr = nullptr;
    ASSERT_EQ(cache.AddEntry(keyA, MakeMemRanges(), MakeMemRanges(), &entryAPtr), HCCL_SUCCESS);

    // 图B(200)的capture entry
    cache.SetCurCaptureModelId(200);
    OpUnfoldKey keyB;
    MakeKey(2048, true, keyB);
    OpUnfoldCacheEntry* entryBPtr = nullptr;
    ASSERT_EQ(cache.AddEntry(keyB, MakeMemRanges(), MakeMemRanges(), &entryBPtr), HCCL_SUCCESS);

    // eager entry
    cache.SetCurCaptureModelId(0);
    OpUnfoldKey keyEager;
    MakeKey(4096, false, keyEager);
    OpUnfoldCacheEntry* entryEagerPtr = nullptr;
    ASSERT_EQ(cache.AddEntry(keyEager, MakeMemRanges(), MakeMemRanges(), &entryEagerPtr), HCCL_SUCCESS);

    // 销毁图A: 只清A的entry
    EXPECT_EQ(cache.ClearEntryForCapture(100), HCCL_SUCCESS);

    // 注意: FindEntry/AddEntry入参检查要求*entryPtrPtr传入时必须为null, 每次查询前需重置
    OpUnfoldCacheEntry* entryPtr = nullptr;
    EXPECT_EQ(cache.FindEntry(keyA, &entryPtr), HCCL_SUCCESS);
    EXPECT_EQ(entryPtr, nullptr); // A的entry已清理

    entryPtr = nullptr;
    EXPECT_EQ(cache.FindEntry(keyB, &entryPtr), HCCL_SUCCESS);
    ASSERT_NE(entryPtr, nullptr); // B的entry不受影响
    EXPECT_EQ(entryPtr, entryBPtr);

    entryPtr = nullptr;
    EXPECT_EQ(cache.FindEntry(keyEager, &entryPtr), HCCL_SUCCESS);
    ASSERT_NE(entryPtr, nullptr); // eager entry不受影响
    EXPECT_EQ(entryPtr, entryEagerPtr);

    // 销毁图B: 清B的entry
    EXPECT_EQ(cache.ClearEntryForCapture(200), HCCL_SUCCESS);
    entryPtr = nullptr;
    EXPECT_EQ(cache.FindEntry(keyB, &entryPtr), HCCL_SUCCESS);
    EXPECT_EQ(entryPtr, nullptr);

    entryPtr = nullptr;
    EXPECT_EQ(cache.FindEntry(keyEager, &entryPtr), HCCL_SUCCESS);
    ASSERT_NE(entryPtr, nullptr); // eager entry仍不受影响
}

// isCapture双重保护: eager entry归属虽为0, ClearEntryForCapture(0)也不清理它
TEST_F(OpUnfoldCacheTest, ut_ClearEntryForCapture_EagerEntryWithZeroModelId_Expect_NotCleared)
{
    OpUnfoldCache cache;

    cache.SetCurCaptureModelId(0);
    OpUnfoldKey keyEager;
    MakeKey(1024, false, keyEager);
    OpUnfoldCacheEntry* entryEagerPtr = nullptr;
    ASSERT_EQ(cache.AddEntry(keyEager, MakeMemRanges(), MakeMemRanges(), &entryEagerPtr), HCCL_SUCCESS);

    // 即使modelId匹配(都为0), isCapture=false的entry不应被清理
    EXPECT_EQ(cache.ClearEntryForCapture(0), HCCL_SUCCESS);

    OpUnfoldCacheEntry* entryPtr = nullptr;
    EXPECT_EQ(cache.FindEntry(keyEager, &entryPtr), HCCL_SUCCESS);
    ASSERT_NE(entryPtr, nullptr);
    EXPECT_EQ(entryPtr, entryEagerPtr);
}

// 同一张图多个entry全部清理且迭代器安全 (连续命中erase场景)
TEST_F(OpUnfoldCacheTest, ut_ClearEntryForCapture_MultipleEntriesSameModelId_Expect_AllCleared)
{
    OpUnfoldCache cache;
    const uint64_t modelId = 100;

    cache.SetCurCaptureModelId(modelId);
    for (uint64_t inputSize = 1024; inputSize <= 5120; inputSize += 1024) {
        OpUnfoldKey key;
        MakeKey(inputSize, true, key);
        OpUnfoldCacheEntry* entryPtr = nullptr;
        ASSERT_EQ(cache.AddEntry(key, MakeMemRanges(), MakeMemRanges(), &entryPtr), HCCL_SUCCESS);
    }

    EXPECT_EQ(cache.ClearEntryForCapture(modelId), HCCL_SUCCESS);

    for (uint64_t inputSize = 1024; inputSize <= 5120; inputSize += 1024) {
        OpUnfoldKey key;
        MakeKey(inputSize, true, key);
        OpUnfoldCacheEntry* entryPtr = nullptr;
        EXPECT_EQ(cache.FindEntry(key, &entryPtr), HCCL_SUCCESS);
        EXPECT_EQ(entryPtr, nullptr); // 全部清理
    }
}

// 清理后相同key可重新AddEntry (对照: 清理前AddEntry同key返回HCCL_E_INTERNAL), 证明确实从map中erase
TEST_F(OpUnfoldCacheTest, ut_ClearEntryForCapture_ThenReAddSameKey_Expect_Success)
{
    OpUnfoldCache cache;
    const uint64_t modelId = 100;

    cache.SetCurCaptureModelId(modelId);
    OpUnfoldKey key;
    MakeKey(1024, true, key);
    OpUnfoldCacheEntry* entryPtr = nullptr;
    ASSERT_EQ(cache.AddEntry(key, MakeMemRanges(), MakeMemRanges(), &entryPtr), HCCL_SUCCESS);

    // 清理前: 同key AddEntry失败 (entry已存在)
    OpUnfoldCacheEntry* dupEntryPtr = nullptr;
    EXPECT_EQ(cache.AddEntry(key, MakeMemRanges(), MakeMemRanges(), &dupEntryPtr), HCCL_E_INTERNAL);

    // 清理后: 同key AddEntry成功 (entry已从cacheHashMap_中erase)
    EXPECT_EQ(cache.ClearEntryForCapture(modelId), HCCL_SUCCESS);
    OpUnfoldCacheEntry* newEntryPtr = nullptr;
    EXPECT_EQ(cache.AddEntry(key, MakeMemRanges(), MakeMemRanges(), &newEntryPtr), HCCL_SUCCESS);
    ASSERT_NE(newEntryPtr, nullptr);
}

// 边界: 空cache清理直接返回成功
TEST_F(OpUnfoldCacheTest, ut_ClearEntryForCapture_EmptyCache_Expect_Success)
{
    OpUnfoldCache cache;
    EXPECT_EQ(cache.ClearEntryForCapture(100), HCCL_SUCCESS);
}

// 边界: 无归属匹配的entry, 清理返回成功且entry不受影响
TEST_F(OpUnfoldCacheTest, ut_ClearEntryForCapture_NoMatchingEntry_Expect_Success)
{
    OpUnfoldCache cache;

    cache.SetCurCaptureModelId(100);
    OpUnfoldKey key;
    MakeKey(1024, true, key);
    OpUnfoldCacheEntry* entryPtr = nullptr;
    ASSERT_EQ(cache.AddEntry(key, MakeMemRanges(), MakeMemRanges(), &entryPtr), HCCL_SUCCESS);

    // 销毁图200 (无归属entry), 不影响图100的entry
    EXPECT_EQ(cache.ClearEntryForCapture(200), HCCL_SUCCESS);

    OpUnfoldCacheEntry* foundPtr = nullptr;
    EXPECT_EQ(cache.FindEntry(key, &foundPtr), HCCL_SUCCESS);
    ASSERT_NE(foundPtr, nullptr);
    EXPECT_EQ(foundPtr, entryPtr);
}

// ========== re-own层: cache hit时entry归属转移给最新使用的图 ==========

// 图A创建entry, 图B命中后re-own归属B; 销毁A不清(B还在用), 销毁B才清
TEST_F(OpUnfoldCacheTest, ut_EntryReOwn_Expect_OwnershipTransferredToLatestGraph)
{
    OpUnfoldCache cache;

    // 图A(100) capture: cache miss产生新entry, 归属A
    cache.SetCurCaptureModelId(100);
    OpUnfoldKey key;
    MakeKey(1024, true, key);
    OpUnfoldCacheEntry* entryPtr = nullptr;
    ASSERT_EQ(cache.AddEntry(key, MakeMemRanges(), MakeMemRanges(), &entryPtr), HCCL_SUCCESS);
    EXPECT_EQ(entryPtr->GetCaptureModelId(), 100u);

    // 图B(200) capture相同key: cache hit, entry re-own归属B
    // (对应AicpuCacheManager::LookupOpUnfoldCache hit分支的SetCaptureModelId逻辑)
    OpUnfoldCacheEntry* hitEntryPtr = nullptr;
    ASSERT_EQ(cache.FindEntry(key, &hitEntryPtr), HCCL_SUCCESS);
    ASSERT_NE(hitEntryPtr, nullptr);
    hitEntryPtr->SetCaptureModelId(200); // re-own为最新使用图
    EXPECT_EQ(hitEntryPtr, entryPtr);

    // 销毁图A: entry归属已是B, 不应被清理
    EXPECT_EQ(cache.ClearEntryForCapture(100), HCCL_SUCCESS);
    OpUnfoldCacheEntry* checkPtr = nullptr;
    EXPECT_EQ(cache.FindEntry(key, &checkPtr), HCCL_SUCCESS);
    ASSERT_NE(checkPtr, nullptr);

    // 销毁图B: entry归属B, 被清理
    EXPECT_EQ(cache.ClearEntryForCapture(200), HCCL_SUCCESS);
    checkPtr = nullptr; // FindEntry入参检查要求*entryPtrPtr为null, 重置后再查
    EXPECT_EQ(cache.FindEntry(key, &checkPtr), HCCL_SUCCESS);
    EXPECT_EQ(checkPtr, nullptr);
}
