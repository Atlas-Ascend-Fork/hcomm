/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <shared_mutex>
#include <vector>

#include "adapter_rts_common.h"
#include "env_func.h"

#define private public
#include "stuck_detect_counter_manager.h"
#include "stuck_detect_counter_pool.h"
#undef private

using namespace hcomm;

namespace {
constexpr uint32_t ZERO_SLOT = 0;
constexpr int32_t INITIAL_COUNTER = 10;
constexpr int32_t ADVANCED_COUNTER = 11;

uint32_t g_hrtMallocCount = 0;
uint32_t g_hrtFreeCount = 0;
uint32_t g_hrtMemSyncCopyCount = 0;
uint32_t g_hrtMemSyncCopyFailAfter = UINT32_MAX;
uint32_t g_hrtMallocFailAfter = UINT32_MAX;

HcclResult CheckCounterStuck(StuckDetectCounterManager& manager, bool& isStuck)
{
    std::unique_lock<std::shared_mutex> lock(manager.managerMutex_);
    return manager.CheckCounterStuckLocked(isStuck);
}

void ResetStubState()
{
    g_hrtMallocCount = 0;
    g_hrtFreeCount = 0;
    g_hrtMemSyncCopyCount = 0;
    g_hrtMemSyncCopyFailAfter = UINT32_MAX;
    g_hrtMallocFailAfter = UINT32_MAX;
}

HcclResult StubHrtMalloc(void** devPtr, u64 size, bool level2Address)
{
    (void)level2Address;
    ++g_hrtMallocCount;
    if (g_hrtMallocCount > g_hrtMallocFailAfter) {
        return HCCL_E_INTERNAL;
    }
    *devPtr = calloc(1, static_cast<size_t>(size));
    return (*devPtr != nullptr) ? HCCL_SUCCESS : HCCL_E_MEMORY;
}

HcclResult StubHrtFree(void* devPtr)
{
    ++g_hrtFreeCount;
    free(devPtr);
    return HCCL_SUCCESS;
}

HcclResult StubHrtMemSet(void* dst, uint64_t destMax, uint64_t count)
{
    (void)destMax;
    if (dst == nullptr) {
        return HCCL_E_PTR;
    }
    memset(dst, 0, static_cast<size_t>(count));
    return HCCL_SUCCESS;
}

HcclResult StubHrtMemSyncCopy(void* dst, uint64_t destMax, const void* src, uint64_t count, HcclRtMemcpyKind kind)
{
    (void)destMax;
    (void)kind;
    ++g_hrtMemSyncCopyCount;
    if (g_hrtMemSyncCopyCount > g_hrtMemSyncCopyFailAfter) {
        return HCCL_E_INTERNAL;
    }
    if (dst == nullptr || src == nullptr) {
        return HCCL_E_PTR;
    }
    memcpy(dst, src, static_cast<size_t>(count));
    return HCCL_SUCCESS;
}

class StuckDetectCounterTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        ResetStubState();
        MOCKER(hrtMalloc).stubs().will(invoke(StubHrtMalloc));
        MOCKER(hrtFree).stubs().will(invoke(StubHrtFree));
        MOCKER(hrtMemSet).stubs().will(invoke(StubHrtMemSet));
        MOCKER(hrtMemSyncCopy).stubs().will(invoke(StubHrtMemSyncCopy));
    }

    void TearDown() override { GlobalMockObject::verify(); }

    static void SetSlotCounter(StuckDetectCounterPool& pool, uint32_t slotIdx, int32_t head, int32_t tail)
    {
        auto* counterBase = static_cast<int32_t*>(pool.poolMem_);
        const uint32_t base = slotIdx * StuckDetectCounterSlot::SLOT_SIZE / sizeof(int32_t);
        counterBase[base + StuckDetectCounterSlot::HEAD_OFFSET / sizeof(int32_t)] = head;
        counterBase[base + StuckDetectCounterSlot::TAIL_OFFSET / sizeof(int32_t)] = tail;
    }
};
} // namespace

TEST_F(StuckDetectCounterTest, Ut_StuckDetectCounterPool_AllocateSlot_When_PoolNotInitialized_Expect_InitAndFirstSlot)
{
    StuckDetectCounterPool pool(0);
    uint32_t slotIdx = 0;
    StuckDetectCounterAddress address;

    EXPECT_EQ(pool.AllocateSlotLocked(slotIdx, address), HCCL_SUCCESS);
    EXPECT_TRUE(pool.IsReadyLocked());
    EXPECT_EQ(slotIdx, ZERO_SLOT);
    EXPECT_NE(address.headAddr, 0U);
    EXPECT_EQ(g_hrtMallocCount, 1U);
}

TEST_F(StuckDetectCounterTest, Ut_StuckDetectCounterPool_AllocateSlot_When_HrtMallocFails_Expect_Error)
{
    g_hrtMallocFailAfter = 0;
    StuckDetectCounterPool pool(0);
    uint32_t slotIdx = 0;
    StuckDetectCounterAddress address;

    EXPECT_EQ(pool.AllocateSlotLocked(slotIdx, address), HCCL_E_INTERNAL);
    EXPECT_FALSE(pool.IsReadyLocked());
    EXPECT_EQ(pool.poolMem_, nullptr);
}

TEST_F(StuckDetectCounterTest, Ut_StuckDetectCounterPool_AllocateSlot_When_SecondAllocation_Expect_NextSlot)
{
    StuckDetectCounterPool pool(0);
    uint32_t firstSlot = 0;
    uint32_t secondSlot = 0;
    StuckDetectCounterAddress firstAddress;
    StuckDetectCounterAddress secondAddress;

    ASSERT_EQ(pool.AllocateSlotLocked(firstSlot, firstAddress), HCCL_SUCCESS);
    ASSERT_EQ(pool.AllocateSlotLocked(secondSlot, secondAddress), HCCL_SUCCESS);
    EXPECT_EQ(firstSlot, ZERO_SLOT);
    EXPECT_EQ(secondSlot, 1U);
    EXPECT_NE(firstAddress.headAddr, secondAddress.headAddr);
}

TEST_F(StuckDetectCounterTest, Ut_StuckDetectCounterPool_AllocateSlot_When_FreeSlotExists_Expect_ReuseSlot)
{
    StuckDetectCounterPool pool(0);
    uint32_t slotIdx = 0;
    StuckDetectCounterAddress address;

    ASSERT_EQ(pool.AllocateSlotLocked(slotIdx, address), HCCL_SUCCESS);
    pool.FreeSlotLocked(slotIdx);
    ASSERT_EQ(pool.AllocateSlotLocked(slotIdx, address), HCCL_SUCCESS);
    EXPECT_EQ(slotIdx, ZERO_SLOT);
}

TEST_F(StuckDetectCounterTest, Ut_StuckDetectCounterPool_AllocateSlot_When_PoolExhausted_Expect_MemoryError)
{
    StuckDetectCounterPool pool(0);
    pool.poolMem_ = calloc(1, 1);
    pool.isInitialized_ = true;
    pool.nextSlotIdx_ = StuckDetectCounterPool::MAX_SLOT_NUM;
    uint32_t slotIdx = 0;
    StuckDetectCounterAddress address;
    EXPECT_EQ(pool.AllocateSlotLocked(slotIdx, address), HCCL_E_MEMORY);
}

TEST_F(StuckDetectCounterTest, Ut_StuckDetectCounterPool_ResetSlot_When_PoolNotReady_Expect_PtrError)
{
    StuckDetectCounterPool pool(0);
    EXPECT_EQ(pool.ResetSlotLocked(ZERO_SLOT), HCCL_E_PTR);
}

TEST_F(StuckDetectCounterTest, Ut_StuckDetectCounterPool_CopyLatestCounters_When_D2HFails_Expect_Error)
{
    StuckDetectCounterPool pool(0);
    uint32_t slotIdx = 0;
    StuckDetectCounterAddress address;
    ASSERT_EQ(pool.AllocateSlotLocked(slotIdx, address), HCCL_SUCCESS);

    g_hrtMemSyncCopyFailAfter = g_hrtMemSyncCopyCount;
    std::vector<int32_t> counterBuf;
    EXPECT_EQ(pool.CopyLatestCountersLocked(counterBuf), HCCL_E_INTERNAL);
}

TEST_F(StuckDetectCounterTest, Ut_StuckDetectCounterPool_DeInit_When_PoolReady_Expect_StateCleared)
{
    StuckDetectCounterPool pool(0);
    uint32_t slotIdx = 0;
    StuckDetectCounterAddress address;
    ASSERT_EQ(pool.AllocateSlotLocked(slotIdx, address), HCCL_SUCCESS);

    pool.DeInitLocked();
    EXPECT_EQ(pool.poolMem_, nullptr);
    EXPECT_FALSE(pool.IsReadyLocked());
    EXPECT_EQ(g_hrtFreeCount, 1U);
}

TEST_F(StuckDetectCounterTest, Ut_StuckDetectCounterManager_AllocCounter_When_NewComm_Expect_SlotAllocated)
{
    StuckDetectCounterManager manager(0);
    StuckDetectCounterAddress address;

    EXPECT_EQ(manager.AllocCounter("comm1", address), HCCL_SUCCESS);
    EXPECT_EQ(manager.commId2CounterInfo_.size(), 1U);
    EXPECT_NE(address.headAddr, 0U);
    EXPECT_EQ(manager.commId2CounterInfo_["comm1"].slotIdx, ZERO_SLOT);
}

TEST_F(StuckDetectCounterTest, Ut_StuckDetectCounterManager_AllocCounter_When_ExistingComm_Expect_Idempotent)
{
    StuckDetectCounterManager manager(0);
    StuckDetectCounterAddress firstAddress;
    StuckDetectCounterAddress secondAddress;

    ASSERT_EQ(manager.AllocCounter("comm1", firstAddress), HCCL_SUCCESS);
    ASSERT_EQ(manager.AllocCounter("comm1", secondAddress), HCCL_SUCCESS);
    EXPECT_EQ(firstAddress.headAddr, secondAddress.headAddr);
    EXPECT_EQ(g_hrtMallocCount, 1U);
}

TEST_F(StuckDetectCounterTest, Ut_StuckDetectCounterManager_FreeCounter_When_UnknownComm_Expect_Success)
{
    StuckDetectCounterManager manager(0);
    EXPECT_EQ(manager.FreeCounter("unknown"), HCCL_SUCCESS);
    EXPECT_TRUE(manager.commId2CounterInfo_.empty());
}

TEST_F(StuckDetectCounterTest, Ut_StuckDetectCounterManager_ResetCounter_When_CommExists_Expect_StateReset)
{
    StuckDetectCounterManager manager(0);
    StuckDetectCounterAddress address;
    ASSERT_EQ(manager.AllocCounter("comm1", address), HCCL_SUCCESS);
    auto& info = manager.commId2CounterInfo_["comm1"];
    info.head = INITIAL_COUNTER;
    info.tail = INITIAL_COUNTER;
    info.status = StuckDetectCommStatus::ACTIVE;

    EXPECT_EQ(manager.ResetCounter("comm1"), HCCL_SUCCESS);
    EXPECT_EQ(info.head, 0);
    EXPECT_EQ(info.tail, 0);
    EXPECT_EQ(info.status, StuckDetectCommStatus::NEVER_ADVANCED);
}

TEST_F(StuckDetectCounterTest, Ut_StuckDetectCounterManager_UpdateCommStatus_When_CountersFrozen_Expect_Frozen)
{
    StuckDetectCounterManager manager(0);
    StuckDetectCounterInfo info;
    info.slotIdx = ZERO_SLOT;
    info.head = INITIAL_COUNTER;
    info.tail = INITIAL_COUNTER;
    info.status = StuckDetectCommStatus::ACTIVE;
    std::vector<int32_t> counterBuf(StuckDetectCounterPool::POOL_SIZE_BYTES / sizeof(int32_t), 0);
    counterBuf[StuckDetectCounterSlot::HEAD_OFFSET / sizeof(int32_t)] = INITIAL_COUNTER;
    counterBuf[StuckDetectCounterSlot::TAIL_OFFSET / sizeof(int32_t)] = INITIAL_COUNTER;

    ASSERT_EQ(manager.UpdateCommStatusLocked(info, counterBuf), HCCL_SUCCESS);
    EXPECT_EQ(info.status, StuckDetectCommStatus::FROZEN);
}

TEST_F(StuckDetectCounterTest, Ut_StuckDetectCounterManager_UpdateCommStatus_When_InvalidSlot_Expect_Error)
{
    StuckDetectCounterManager manager(0);
    StuckDetectCounterInfo info;
    info.slotIdx = StuckDetectCounterPool::MAX_SLOT_NUM;
    std::vector<int32_t> counterBuf(StuckDetectCounterPool::POOL_SIZE_BYTES / sizeof(int32_t), 0);

    EXPECT_EQ(manager.UpdateCommStatusLocked(info, counterBuf), HCCL_E_INTERNAL);
}

TEST_F(StuckDetectCounterTest, Ut_StuckDetectCounterManager_UpdateCommStatus_When_BufferTooSmall_Expect_Error)
{
    StuckDetectCounterManager manager(0);
    StuckDetectCounterInfo info;
    info.slotIdx = ZERO_SLOT;
    std::vector<int32_t> counterBuf;

    EXPECT_EQ(manager.UpdateCommStatusLocked(info, counterBuf), HCCL_E_INTERNAL);
}

TEST_F(StuckDetectCounterTest, Ut_CastDfsConfig_When_StuckDetectionOnOffInvalid_Expect_ParseResult)
{
    auto onConfig = Hccl::CastDfsConfig("stuck_detection:on");
    EXPECT_TRUE(onConfig.stuckDetectionEnable);

    auto offConfig = Hccl::CastDfsConfig("stuck_detection:off");
    EXPECT_FALSE(offConfig.stuckDetectionEnable);

    EXPECT_THROW(Hccl::CastDfsConfig("stuck_detection:foo"), Hccl::InvalidParamsException);
}

TEST_F(StuckDetectCounterTest, Ut_StuckDetectCounterManager_CheckCounterStuck_When_AllZero_Expect_NotStuck)
{
    StuckDetectCounterManager manager(0);
    StuckDetectCounterAddress address;
    ASSERT_EQ(manager.AllocCounter("comm1", address), HCCL_SUCCESS);

    bool isStuck = true;
    EXPECT_EQ(CheckCounterStuck(manager, isStuck), HCCL_SUCCESS);
    EXPECT_FALSE(isStuck);
}

TEST_F(StuckDetectCounterTest, Ut_StuckDetectCounterManager_CheckCounterStuck_When_Frozen_Expect_Stuck)
{
    StuckDetectCounterManager manager(0);
    StuckDetectCounterAddress address;
    ASSERT_EQ(manager.AllocCounter("comm1", address), HCCL_SUCCESS);
    SetSlotCounter(manager.counterPool_, ZERO_SLOT, INITIAL_COUNTER, INITIAL_COUNTER);

    bool isStuck = false;
    ASSERT_EQ(CheckCounterStuck(manager, isStuck), HCCL_SUCCESS);
    EXPECT_FALSE(isStuck);
    ASSERT_EQ(CheckCounterStuck(manager, isStuck), HCCL_SUCCESS);
    EXPECT_TRUE(isStuck);
}

TEST_F(StuckDetectCounterTest, Ut_StuckDetectCounterManager_UpdateCommStatus_When_InFlightCounterNotAdvanced)
{
    StuckDetectCounterManager manager(0);
    StuckDetectCounterInfo info;
    info.slotIdx = ZERO_SLOT;
    info.head = ADVANCED_COUNTER;
    info.tail = INITIAL_COUNTER;
    info.status = StuckDetectCommStatus::ACTIVE;
    std::vector<int32_t> counterBuf(StuckDetectCounterPool::POOL_SIZE_BYTES / sizeof(int32_t), 0);
    counterBuf[StuckDetectCounterSlot::HEAD_OFFSET / sizeof(int32_t)] = ADVANCED_COUNTER;
    counterBuf[StuckDetectCounterSlot::TAIL_OFFSET / sizeof(int32_t)] = INITIAL_COUNTER;

    ASSERT_EQ(manager.UpdateCommStatusLocked(info, counterBuf), HCCL_SUCCESS);
    EXPECT_EQ(info.status, StuckDetectCommStatus::ACTIVE);
}

TEST_F(StuckDetectCounterTest, Ut_StuckDetectCounterManager_CheckCounterStuck_When_InFlightNotAdvanced_Expect_NotStuck)
{
    StuckDetectCounterManager manager(0);
    StuckDetectCounterAddress address;
    ASSERT_EQ(manager.AllocCounter("comm1", address), HCCL_SUCCESS);
    SetSlotCounter(manager.counterPool_, ZERO_SLOT, ADVANCED_COUNTER, INITIAL_COUNTER);

    bool isStuck = true;
    ASSERT_EQ(CheckCounterStuck(manager, isStuck), HCCL_SUCCESS);
    EXPECT_FALSE(isStuck);
    ASSERT_EQ(CheckCounterStuck(manager, isStuck), HCCL_SUCCESS);
    EXPECT_FALSE(isStuck);
}

TEST_F(StuckDetectCounterTest, Ut_StuckDetectCounterManager_CheckProcessStuck_When_ThresholdReached_Expect_Reported)
{
    StuckDetectCounterManager manager(0);
    StuckDetectCounterAddress address;
    ASSERT_EQ(manager.AllocCounter("comm1", address), HCCL_SUCCESS);
    SetSlotCounter(manager.counterPool_, ZERO_SLOT, INITIAL_COUNTER, INITIAL_COUNTER);

    bool needReportStuck = false;
    ASSERT_EQ(manager.CheckProcessStuck(2, needReportStuck), HCCL_SUCCESS);
    EXPECT_FALSE(needReportStuck);
    ASSERT_EQ(manager.CheckProcessStuck(2, needReportStuck), HCCL_SUCCESS);
    EXPECT_FALSE(needReportStuck);
    ASSERT_EQ(manager.CheckProcessStuck(2, needReportStuck), HCCL_SUCCESS);
    EXPECT_TRUE(needReportStuck);
}

TEST_F(StuckDetectCounterTest, Ut_StuckDetectCounterManager_CheckProcessStuck_When_FrozenAndActive_Expect_NotStuck)
{
    StuckDetectCounterManager manager(0);
    StuckDetectCounterAddress frozenAddress;
    StuckDetectCounterAddress activeAddress;
    ASSERT_EQ(manager.AllocCounter("frozen", frozenAddress), HCCL_SUCCESS);
    ASSERT_EQ(manager.AllocCounter("active", activeAddress), HCCL_SUCCESS);
    SetSlotCounter(manager.counterPool_, ZERO_SLOT, INITIAL_COUNTER, INITIAL_COUNTER);
    SetSlotCounter(manager.counterPool_, 1U, ADVANCED_COUNTER, INITIAL_COUNTER);

    bool needReportStuck = false;
    ASSERT_EQ(manager.CheckProcessStuck(2, needReportStuck), HCCL_SUCCESS);
    EXPECT_FALSE(needReportStuck);
    ASSERT_EQ(manager.CheckProcessStuck(2, needReportStuck), HCCL_SUCCESS);
    EXPECT_FALSE(needReportStuck);
    ASSERT_EQ(manager.CheckProcessStuck(2, needReportStuck), HCCL_SUCCESS);
    EXPECT_FALSE(needReportStuck);
}

TEST_F(StuckDetectCounterTest, Ut_StuckDetectCounterManager_CheckProcessStuck_When_FrozenAndNeverAdvanced_Expect_Stuck)
{
    StuckDetectCounterManager manager(0);
    StuckDetectCounterAddress frozenAddress;
    StuckDetectCounterAddress neverAdvancedAddress;
    ASSERT_EQ(manager.AllocCounter("frozen", frozenAddress), HCCL_SUCCESS);
    ASSERT_EQ(manager.AllocCounter("neverAdvanced", neverAdvancedAddress), HCCL_SUCCESS);
    SetSlotCounter(manager.counterPool_, ZERO_SLOT, INITIAL_COUNTER, INITIAL_COUNTER);

    bool needReportStuck = false;
    ASSERT_EQ(manager.CheckProcessStuck(2, needReportStuck), HCCL_SUCCESS);
    EXPECT_FALSE(needReportStuck);
    ASSERT_EQ(manager.CheckProcessStuck(2, needReportStuck), HCCL_SUCCESS);
    EXPECT_FALSE(needReportStuck);
    ASSERT_EQ(manager.CheckProcessStuck(2, needReportStuck), HCCL_SUCCESS);
    EXPECT_TRUE(needReportStuck);
}

TEST_F(StuckDetectCounterTest, Ut_StuckDetectCounterManager_CheckProcessStuck_When_SlotReused_Expect_NoStaleCounter)
{
    StuckDetectCounterManager manager(0);
    StuckDetectCounterAddress address;
    ASSERT_EQ(manager.AllocCounter("old", address), HCCL_SUCCESS);
    SetSlotCounter(manager.counterPool_, ZERO_SLOT, INITIAL_COUNTER, INITIAL_COUNTER);

    bool isStuck = false;
    ASSERT_EQ(CheckCounterStuck(manager, isStuck), HCCL_SUCCESS);
    EXPECT_FALSE(isStuck);
    ASSERT_EQ(CheckCounterStuck(manager, isStuck), HCCL_SUCCESS);
    EXPECT_TRUE(isStuck);

    ASSERT_EQ(manager.FreeCounter("old"), HCCL_SUCCESS);
    ASSERT_EQ(manager.AllocCounter("new", address), HCCL_SUCCESS);
    ASSERT_EQ(CheckCounterStuck(manager, isStuck), HCCL_SUCCESS);
    EXPECT_FALSE(isStuck);
}

TEST_F(StuckDetectCounterTest, Ut_StuckDetectCounterManager_DeInit_When_CounterAllocated_Expect_PoolReleased)
{
    StuckDetectCounterManager manager(0);
    StuckDetectCounterAddress address;
    ASSERT_EQ(manager.AllocCounter("comm1", address), HCCL_SUCCESS);

    manager.DeInit();
    EXPECT_TRUE(manager.commId2CounterInfo_.empty());
    EXPECT_FALSE(manager.counterPool_.IsReadyLocked());
}
