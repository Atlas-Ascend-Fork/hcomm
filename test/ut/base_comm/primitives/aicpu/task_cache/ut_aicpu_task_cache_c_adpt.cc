/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#ifndef private
#define private public
#define protected public
#endif

#include "dtype_common.h"
#include "hcomm_primitives.h"
#include "aicpu_task_cache_manager.h"
#include "aicpu_task_cache_entry.h"
#include "ub_conn_lite.h"
#include "ub_transport_lite_impl.h"
#include "rtsq_a5.h"
#include "aicpu_ts_thread.h"
#include "sqe_v82.h"
#include "udma_data_struct.h"
#include "ub_jetty_lite.h"
#include "ip_address.h"

#undef private
#undef protected

using namespace hcomm;
using Hccl::Rt91095StarsSqeHeader;
using Hccl::Rt91095StarsSqeType;

namespace {
std::vector<u64> g_defaultTransports(128, DFX_INVALID_U64);
std::vector<u8> g_defaultLinkTypes(128, static_cast<u8>(Hccl::DfxLinkTypeVal::LINK_ONCHIP));
std::vector<u8> g_defaultTransportTypes(128, static_cast<u8>(Hccl::DfxTransportType::DFX_TRANSPORT_TYPE_LOCAL));
} // namespace

namespace {
DevType g_testDeviceType = DevType::DEV_TYPE_950;
}

extern "C" HcclResult hrtGetDeviceType(DevType& devType)
{
    devType = g_testDeviceType;
    return HCCL_SUCCESS;
}

namespace {
constexpr uint32_t TEST_SQ_DEPTH = 128;
constexpr uint64_t TEST_BASE_ADDR_0 = 0x10000ULL;
constexpr uint64_t TEST_MEM_SIZE_0 = 0x1000ULL;
constexpr uint64_t TEST_BASE_ADDR_1 = 0x20000ULL;
constexpr uint64_t TEST_MEM_SIZE_1 = 0x2000ULL;

std::vector<uint8_t> MakeSqeArray(Rt91095StarsSqeType sqeType)
{
    std::vector<uint8_t> sqeArray(Hccl::AC_SQE_SIZE, 0);
    Rt91095StarsSqeHeader* header = reinterpret_cast<Rt91095StarsSqeHeader*>(sqeArray.data());
    header->type = static_cast<uint8_t>(sqeType);
    return sqeArray;
}
} // namespace

class HcommAicpuTsTaskCacheCAdptTest : public testing::Test {
protected:
    std::unique_ptr<Hccl::RtsqA5> rtsq_;
    Hccl::RtsqA5* rtsqPtr_ = nullptr;
    std::unique_ptr<hccl::AicpuTsThread> aicpuTsThread_;
    std::unique_ptr<Hccl::UbConnLite> ubConnLite_;
    std::unique_ptr<Hccl::UbTransportLiteImpl> ubTransport_;
    std::vector<char> emptyUniqueId_;
    std::string savedCacheTag_;
    bool savedIsHit_ = false;
    AicpuTaskCacheEntry* savedCacheEntryPtr_ = nullptr;
    DevType savedDeviceType_ = DevType::DEV_TYPE_950;
    uint64_t savedCacheBytes_ = 0;
    bool savedCacheHitRunInfoPrinted_ = false;
    bool savedCacheFullRunInfoPrinted_ = false;

    void SetUp() override
    {
        rtsq_ = std::make_unique<Hccl::RtsqA5>(0, 0, 0);
        rtsqPtr_ = rtsq_.get();
        aicpuTsThread_ = std::make_unique<hccl::AicpuTsThread>(std::string("ut_c_adpt_thread"));
        ubConnLite_ = std::make_unique<Hccl::UbConnLite>(
            Hccl::UbJettyLiteId(0, 0, 0), Hccl::UbJettyLiteAttr(0, 0, TEST_SQ_DEPTH, 0, true), Hccl::Eid{});
        ubTransport_ = std::make_unique<Hccl::UbTransportLiteImpl>(emptyUniqueId_);

        savedCacheTag_ = AicpuTaskCacheManager::cacheTag;
        savedIsHit_ = AicpuTaskCacheManager::isHit;
        savedCacheEntryPtr_ = AicpuTaskCacheManager::cacheEntryPtr;
        savedDeviceType_ = g_testDeviceType;
        savedCacheBytes_ = AicpuTaskCacheManager::aicpuTaskCache.cacheBytes_;
        savedCacheHitRunInfoPrinted_ = AicpuTaskCacheManager::aicpuTaskCache.cacheHitRunInfoPrinted_;
        savedCacheFullRunInfoPrinted_ = AicpuTaskCacheManager::aicpuTaskCache.cacheFullRunInfoPrinted_;
        g_testDeviceType = DevType::DEV_TYPE_950;
    }

    void TearDown() override
    {
        AicpuTaskCacheManager::aicpuTaskCache.ClearEntry("c_adpt_tag_1");
        AicpuTaskCacheManager::aicpuTaskCache.ClearEntry("c_adpt_tag_2");
        AicpuTaskCacheManager::aicpuTaskCache.ClearEntry("c_adpt_full_workflow");

        AicpuTaskCacheManager::cacheTag = savedCacheTag_;
        AicpuTaskCacheManager::isHit = savedIsHit_;
        AicpuTaskCacheManager::cacheEntryPtr = savedCacheEntryPtr_;
        AicpuTaskCacheManager::aicpuTaskCache.cacheBytes_ = savedCacheBytes_;
        AicpuTaskCacheManager::aicpuTaskCache.cacheHitRunInfoPrinted_ = savedCacheHitRunInfoPrinted_;
        AicpuTaskCacheManager::aicpuTaskCache.cacheFullRunInfoPrinted_ = savedCacheFullRunInfoPrinted_;
        g_testDeviceType = savedDeviceType_;
    }

    void SetDeviceTypeNotSupport() { g_testDeviceType = DevType::DEV_TYPE_910B; }

    void SetDeviceType960() { g_testDeviceType = DevType::DEV_TYPE_960; }

    void SetCacheMissState(AicpuTaskCacheEntry* entryPtr)
    {
        AicpuTaskCacheManager::isHit = false;
        AicpuTaskCacheManager::cacheEntryPtr = entryPtr;
    }

    void SetCacheHitState(AicpuTaskCacheEntry* entryPtr)
    {
        AicpuTaskCacheManager::isHit = true;
        AicpuTaskCacheManager::cacheEntryPtr = entryPtr;
    }

    void SetCacheFullState()
    {
        AicpuTaskCacheManager::isHit = false;
        AicpuTaskCacheManager::cacheEntryPtr = nullptr;
    }
};

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Start_VFormatDecodesByteArrays)
{
    AicpuTaskCacheEntry entry;
    AicpuTaskCacheManager::cacheTag = "v_tag";
    SetCacheMissState(&entry);
    void* addrs[] = {(void*)0x10000, (void*)0x20000, (void*)0x10020, (void*)0x10080, (void*)0x20040, (void*)0x20090};
    uint64_t sizes[] = {256, 256, 16, 0, 24, 8};
    ASSERT_EQ(HcommAicpuTsTaskCacheStart("v_tag", addrs, sizes, (uint64_t{1} << 63) | 6), HCCL_SUCCESS);
    EXPECT_EQ(entry.cachedBaseAddrs_.size(), 2U);
    EXPECT_TRUE(entry.isVOp_);
    EXPECT_EQ(entry.vOpInfo_.sendCounts, (std::vector<uint64_t>{16, 0}));
    EXPECT_EQ(entry.vOpInfo_.recvCounts, (std::vector<uint64_t>{24, 8}));
    EXPECT_EQ(entry.vOpInfo_.sdispls, (std::vector<uint64_t>{32, 128}));
    EXPECT_EQ(entry.vOpInfo_.rdispls, (std::vector<uint64_t>{64, 144}));
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Start_UnmarkedArraysKeepLegacyRemoteRanges)
{
    for (uint64_t count : {2U, 4U}) {
        AicpuTaskCacheEntry entry;
        AicpuTaskCacheManager::cacheTag = "legacy_tag";
        SetCacheMissState(&entry);
        void* addrs[] = {(void*)0x10000, (void*)0x20000, (void*)0x30000, (void*)0x40000};
        uint64_t sizes[] = {256, 256, 256, 256};
        ASSERT_EQ(HcommAicpuTsTaskCacheStart("legacy_tag", addrs, sizes, count), HCCL_SUCCESS);
        EXPECT_EQ(entry.cachedBaseAddrs_.size(), count);
        EXPECT_FALSE(entry.isVOp_);
        EXPECT_TRUE(entry.vOpInfo_.sendCounts.empty());
    }
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Start_VFormatRejectsMalformedRanges)
{
    AicpuTaskCacheEntry entry;
    AicpuTaskCacheManager::cacheTag = "v_tag";
    SetCacheMissState(&entry);
    void* addrs[] = {(void*)0x10000, (void*)0x20000, (void*)0x10020, (void*)0x20040};
    uint64_t sizes[] = {256, 256, 16, 24};
    const auto start = [&](uint64_t count) {
        AicpuTaskCacheManager::cacheTag = "v_tag";
        SetCacheMissState(&entry);
        return HcommAicpuTsTaskCacheStart("v_tag", addrs, sizes, count);
    };
    for (uint64_t count : {0U, 1U, 2U, 3U, 5U}) {
        EXPECT_EQ(start((uint64_t{1} << 63) | count), HCCL_E_PARA);
    }
    addrs[2] = (void*)0xffff;
    EXPECT_EQ(start((uint64_t{1} << 63) | 4), HCCL_E_PARA);
    addrs[2] = (void*)0x100ff;
    EXPECT_EQ(start((uint64_t{1} << 63) | 4), HCCL_E_PARA);
    addrs[2] = reinterpret_cast<void*>(UINT64_MAX - 1);
    EXPECT_EQ(start((uint64_t{1} << 63) | 4), HCCL_E_PARA);
    addrs[2] = (void*)0x10020;
    addrs[0] = reinterpret_cast<void*>(UINT64_MAX - 1);
    EXPECT_EQ(start((uint64_t{1} << 63) | 4), HCCL_E_PARA);
    EXPECT_TRUE(entry.cachedBaseAddrs_.empty());
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Execute_VFormatRejectsSdmaCountAboveUint32BeforeLaunch)
{
    AicpuTaskCacheEntry entry;
    AicpuTaskCacheManager::cacheTag = "v_tag";
    SetCacheMissState(&entry);
    void* oldAddrs[] = {(void*)0x10000, (void*)0x20000, (void*)0x10000, (void*)0x20000};
    uint64_t oldSizes[] = {256, 256, 16, 16};
    constexpr uint64_t V_COUNT = (uint64_t{1} << 63) | 4;
    ASSERT_EQ(HcommAicpuTsTaskCacheStart("v_tag", oldAddrs, oldSizes, V_COUNT), HCCL_SUCCESS);

    auto sqe = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA);
    auto* sdma = reinterpret_cast<Hccl::Rt91095StarsMemcpySqe*>(sqe.data());
    sdma->u.strideMode0.srcAddrLow = 0x10000;
    sdma->u.strideMode0.dstAddrLow = 0x20000;
    sdma->u.strideMode0.lengthMove = 16;
    ASSERT_EQ(
        entry.AddSqeArray(
            rtsqPtr_, aicpuTsThread_.get(), 1, sqe.data(), 0, g_defaultTransports.data(), g_defaultLinkTypes.data(),
            g_defaultTransportTypes.data()),
        HCCL_SUCCESS);
    ASSERT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);
    const auto originalTail = rtsqPtr_->GetTail();
    const auto originalPending = rtsqPtr_->GetPendingSqeCnt();

    constexpr uint64_t LARGE_BUFFER = uint64_t{8} << 30;
    constexpr uint64_t OVERSIZED_COUNT = uint64_t{5} << 30;
    void* newAddrs[] = {(void*)0x30000, (void*)0x40000, (void*)0x30000, (void*)0x40000};
    uint64_t newSizes[] = {LARGE_BUFFER, 256, OVERSIZED_COUNT, 16};
    SetCacheHitState(&entry);
    EXPECT_EQ(HcommAicpuTsTaskCacheExecute("v_tag", newAddrs, newSizes, V_COUNT), HCCL_E_PARA);
    EXPECT_EQ(rtsqPtr_->GetTail(), originalTail);
    EXPECT_EQ(rtsqPtr_->GetPendingSqeCnt(), originalPending);
    EXPECT_EQ(entry.vOpInfo_.sendCounts, (std::vector<uint64_t>{16}));
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Execute_VFormatAcceptsValidMetadataChangesWithoutTasks)
{
    for (uint32_t change = 0; change < 5; ++change) {
        SCOPED_TRACE(change);
        AicpuTaskCacheEntry entry;
        AicpuTaskCacheManager::cacheTag = "v_tag";
        SetCacheMissState(&entry);
        void* addrs[] = {(void*)0x10000, (void*)0x20000, (void*)0x10020, (void*)0x20040};
        uint64_t sizes[] = {256, 256, 16, 24};
        ASSERT_EQ(HcommAicpuTsTaskCacheStart("v_tag", addrs, sizes, (uint64_t{1} << 63) | 4), HCCL_SUCCESS);
        // Empty task arrays test argument validation only, not actual SQE/WQE replay.
        SetCacheHitState(&entry);
        addrs[0] = (void*)0x30000;
        addrs[1] = (void*)0x40000;
        addrs[2] = (void*)0x30020;
        addrs[3] = (void*)0x40040;
        if (change == 1) {
            sizes[2] = 8;
        }
        if (change == 2) {
            sizes[3] = 8;
        }
        if (change == 3) {
            addrs[2] = (void*)0x30030;
        }
        if (change == 4) {
            addrs[3] = (void*)0x40050;
        }
        EXPECT_EQ(HcommAicpuTsTaskCacheExecute("v_tag", addrs, sizes, (uint64_t{1} << 63) | 4), HCCL_SUCCESS);
        EXPECT_EQ(entry.vOpInfo_.sendCounts, (std::vector<uint64_t>{16}));
        EXPECT_EQ(entry.vOpInfo_.recvCounts, (std::vector<uint64_t>{24}));
        EXPECT_EQ(entry.vOpInfo_.sdispls, (std::vector<uint64_t>{32}));
        EXPECT_EQ(entry.vOpInfo_.rdispls, (std::vector<uint64_t>{64}));
        EXPECT_TRUE(AicpuTaskCacheManager::cacheTag.empty());
        EXPECT_FALSE(AicpuTaskCacheManager::isHit);
        EXPECT_EQ(AicpuTaskCacheManager::cacheEntryPtr, nullptr);
    }
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Execute_VFormatRejectsRankSizeMismatchBeforeRefreshingTasks)
{
    for (uint64_t rankSize : {1U, 3U}) {
        SCOPED_TRACE(rankSize);
        AicpuTaskCacheEntry entry;
        AicpuTaskCacheManager::cacheTag = "v_tag";
        SetCacheMissState(&entry);
        void* oldAddrs[]
            = {(void*)0x10000, (void*)0x20000, (void*)0x10000, (void*)0x10020, (void*)0x20000, (void*)0x20020};
        uint64_t oldSizes[] = {256, 256, 16, 16, 16, 16};
        ASSERT_EQ(HcommAicpuTsTaskCacheStart("v_tag", oldAddrs, oldSizes, (uint64_t{1} << 63) | 6), HCCL_SUCCESS);

        auto sqe = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA);
        auto* sdma = reinterpret_cast<Hccl::Rt91095StarsMemcpySqe*>(sqe.data());
        sdma->u.strideMode0.srcAddrLow = 0x10000;
        sdma->u.strideMode0.dstAddrLow = 0x20000;
        sdma->u.strideMode0.lengthMove = 16;
        ASSERT_EQ(
            entry.AddSqeArray(
                rtsqPtr_, aicpuTsThread_.get(), 1, sqe.data(), 0, g_defaultTransports.data(), g_defaultLinkTypes.data(),
                g_defaultTransportTypes.data()),
            HCCL_SUCCESS);
        ASSERT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);
        const std::vector<uint8_t> originalSqe(
            entry.sqeArrayInfos_[0].sqeArray, entry.sqeArrayInfos_[0].sqeArray + Hccl::AC_SQE_SIZE);
        const auto originalTail = rtsqPtr_->GetTail();
        const auto originalPending = rtsqPtr_->GetPendingSqeCnt();

        // 每个地址范围均合法，仅回放 rankSize 与录制时的 2 不一致。
        std::vector<void*> newAddrs = {(void*)0x30000, (void*)0x40000};
        std::vector<uint64_t> newSizes = {256, 256};
        for (uint64_t base : {0x30000U, 0x40000U}) {
            for (uint64_t rank = 0; rank < rankSize; ++rank) {
                newAddrs.push_back(reinterpret_cast<void*>(base + rank * 32));
                newSizes.push_back(16);
            }
        }
        SetCacheHitState(&entry);
        EXPECT_EQ(
            HcommAicpuTsTaskCacheExecute(
                "v_tag", newAddrs.data(), newSizes.data(), (uint64_t{1} << 63) | (2 + 2 * rankSize)),
            HCCL_E_PARA);
        EXPECT_EQ(std::memcmp(entry.sqeArrayInfos_[0].sqeArray, originalSqe.data(), Hccl::AC_SQE_SIZE), 0);
        EXPECT_EQ(rtsqPtr_->GetTail(), originalTail);
        EXPECT_EQ(rtsqPtr_->GetPendingSqeCnt(), originalPending);
        EXPECT_EQ(entry.vOpInfo_.sendCounts, (std::vector<uint64_t>{16, 16}));
        EXPECT_EQ(entry.vOpInfo_.recvCounts, (std::vector<uint64_t>{16, 16}));
        EXPECT_TRUE(AicpuTaskCacheManager::cacheTag.empty());
        EXPECT_FALSE(AicpuTaskCacheManager::isHit);
        EXPECT_EQ(AicpuTaskCacheManager::cacheEntryPtr, nullptr);
    }
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Execute_VFormatKeepsWarmupMetadataAfterZeroCounts)
{
    for (uint32_t side = 0; side < 2; ++side) {
        SCOPED_TRACE(side);
        AicpuTaskCacheEntry entry;
        AicpuTaskCacheManager::cacheTag = "v_tag";
        SetCacheMissState(&entry);
        void* addrs[] = {(void*)0x10000, (void*)0x20000, (void*)0x10020, (void*)0x20040};
        uint64_t sizes[] = {256, 256, 16, 24};
        ASSERT_EQ(HcommAicpuTsTaskCacheStart("v_tag", addrs, sizes, (uint64_t{1} << 63) | 4), HCCL_SUCCESS);

        // Empty task arrays verify argument handling and retention of the nonzero warmup metadata only.
        SetCacheHitState(&entry);
        sizes[2 + side] = 0;
        ASSERT_EQ(HcommAicpuTsTaskCacheExecute("v_tag", addrs, sizes, (uint64_t{1} << 63) | 4), HCCL_SUCCESS);
        EXPECT_TRUE(AicpuTaskCacheManager::cacheTag.empty());
        EXPECT_FALSE(AicpuTaskCacheManager::isHit);
        EXPECT_EQ(AicpuTaskCacheManager::cacheEntryPtr, nullptr);

        AicpuTaskCacheManager::cacheTag = "v_tag";
        SetCacheHitState(&entry);
        sizes[2 + side] = 8;
        EXPECT_EQ(HcommAicpuTsTaskCacheExecute("v_tag", addrs, sizes, (uint64_t{1} << 63) | 4), HCCL_SUCCESS);
        EXPECT_EQ(entry.vOpInfo_.sendCounts, (std::vector<uint64_t>{16}));
        EXPECT_EQ(entry.vOpInfo_.recvCounts, (std::vector<uint64_t>{24}));
        EXPECT_TRUE(AicpuTaskCacheManager::cacheTag.empty());
        EXPECT_FALSE(AicpuTaskCacheManager::isHit);
        EXPECT_EQ(AicpuTaskCacheManager::cacheEntryPtr, nullptr);
    }
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Execute_VFormatAllowsNullBufferOnlyForEmptySide)
{
    for (uint32_t side = 0; side < 2; ++side) {
        for (uint64_t emptyDispl : {uint64_t{0}, uint64_t{64}}) {
            SCOPED_TRACE(side);
            SCOPED_TRACE(emptyDispl);
            AicpuTaskCacheEntry entry;
            AicpuTaskCacheManager::cacheTag = "v_tag";
            SetCacheMissState(&entry);
            void* addrs[]
                = {(void*)0x10000, (void*)0x20000, (void*)0x10000, (void*)0x10020, (void*)0x20000, (void*)0x20040};
            uint64_t sizes[] = {256, 256, 16, 16, 24, 24};
            constexpr uint64_t V_COUNT = (uint64_t{1} << 63) | 6;
            ASSERT_EQ(HcommAicpuTsTaskCacheStart("v_tag", addrs, sizes, V_COUNT), HCCL_SUCCESS);

            // Empty task arrays verify parsing/replay validation without accessing hardware queues.
            addrs[side] = nullptr;
            sizes[side] = emptyDispl;
            for (uint32_t rank = 0; rank < 2; ++rank) {
                addrs[2 + side * 2 + rank] = reinterpret_cast<void*>(emptyDispl);
                sizes[2 + side * 2 + rank] = 0;
            }
            SetCacheHitState(&entry);
            ASSERT_EQ(HcommAicpuTsTaskCacheExecute("v_tag", addrs, sizes, V_COUNT), HCCL_SUCCESS);
            EXPECT_TRUE(AicpuTaskCacheManager::cacheTag.empty());
            EXPECT_EQ(AicpuTaskCacheManager::cacheEntryPtr, nullptr);

            // A nonzero count on either peer requires a real buffer, regardless of the full buffer size.
            sizes[side] = 256;
            for (uint32_t rank = 0; rank < 2; ++rank) {
                SCOPED_TRACE(rank);
                AicpuTaskCacheManager::cacheTag = "v_tag";
                SetCacheHitState(&entry);
                sizes[2 + side * 2 + rank] = 1;
                EXPECT_EQ(HcommAicpuTsTaskCacheExecute("v_tag", addrs, sizes, V_COUNT), HCCL_E_PARA);
                sizes[2 + side * 2 + rank] = 0;
            }

            AicpuTaskCacheManager::cacheTag = "v_tag";
            SetCacheHitState(&entry);
            addrs[side] = reinterpret_cast<void*>(side == 0 ? 0x30000 : 0x40000);
            for (uint32_t rank = 0; rank < 2; ++rank) {
                addrs[2 + side * 2 + rank] = addrs[side];
                sizes[2 + side * 2 + rank] = 8;
            }
            EXPECT_EQ(HcommAicpuTsTaskCacheExecute("v_tag", addrs, sizes, V_COUNT), HCCL_SUCCESS);
            EXPECT_EQ(entry.vOpInfo_.sendCounts, (std::vector<uint64_t>{16, 16}));
            EXPECT_EQ(entry.vOpInfo_.recvCounts, (std::vector<uint64_t>{24, 24}));
        }
    }
}

// ===================== HcommAicpuTsTaskCacheLookup Tests =====================

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Lookup_DeviceNotSupport_ReturnsNotSupport)
{
    SetDeviceTypeNotSupport();
    bool isHit = false;
    EXPECT_EQ(HcommAicpuTsTaskCacheLookup("tag1", &isHit), HCCL_E_NOT_SUPPORT);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Lookup_Device960_ReturnsSuccess)
{
    SetDeviceType960();
    AicpuTaskCacheManager::aicpuTaskCache.ClearEntry("c_adpt_tag_1");
    bool isHit = true;
    EXPECT_EQ(HcommAicpuTsTaskCacheLookup("c_adpt_tag_1", &isHit), HCCL_SUCCESS);
    EXPECT_FALSE(isHit);
    EXPECT_EQ(AicpuTaskCacheManager::cacheTag, std::string("c_adpt_tag_1"));
    EXPECT_FALSE(AicpuTaskCacheManager::isHit);
    EXPECT_NE(AicpuTaskCacheManager::cacheEntryPtr, nullptr);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Lookup_NullTag_ReturnsPtr)
{
    bool isHit = false;
    EXPECT_EQ(HcommAicpuTsTaskCacheLookup(nullptr, &isHit), HCCL_E_PTR);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Lookup_NullIsHit_ReturnsPtr)
{
    EXPECT_EQ(HcommAicpuTsTaskCacheLookup("tag1", nullptr), HCCL_E_PTR);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Lookup_CacheMiss_ReturnsSuccess)
{
    AicpuTaskCacheManager::aicpuTaskCache.cacheHitRunInfoPrinted_ = false;
    AicpuTaskCacheManager::aicpuTaskCache.ClearEntry("c_adpt_tag_1");
    bool isHit = true;
    EXPECT_EQ(HcommAicpuTsTaskCacheLookup("c_adpt_tag_1", &isHit), HCCL_SUCCESS);
    EXPECT_FALSE(isHit);
    EXPECT_EQ(AicpuTaskCacheManager::cacheTag, std::string("c_adpt_tag_1"));
    EXPECT_FALSE(AicpuTaskCacheManager::isHit);
    EXPECT_NE(AicpuTaskCacheManager::cacheEntryPtr, nullptr);
    EXPECT_FALSE(AicpuTaskCacheManager::aicpuTaskCache.cacheHitRunInfoPrinted_);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Lookup_CacheHit_ReturnsSuccess)
{
    AicpuTaskCacheManager::aicpuTaskCache.cacheHitRunInfoPrinted_ = false;
    AicpuTaskCacheManager::aicpuTaskCache.ClearEntry("c_adpt_tag_1");
    bool isHit = false;
    ASSERT_EQ(HcommAicpuTsTaskCacheLookup("c_adpt_tag_1", &isHit), HCCL_SUCCESS);
    ASSERT_FALSE(isHit);

    AicpuTaskCacheEntry* entry = AicpuTaskCacheManager::cacheEntryPtr;
    ASSERT_NE(entry, nullptr);

    uint64_t baseAddrs[] = {TEST_BASE_ADDR_0, TEST_BASE_ADDR_1};
    uint64_t memSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    ASSERT_EQ(entry->InitCacheEntry(baseAddrs, memSizes, 2), HCCL_SUCCESS);
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
    ASSERT_EQ(
        entry->AddSqeArray(
            rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0, g_defaultTransports.data(),
            g_defaultLinkTypes.data(), g_defaultTransportTypes.data()),
        HCCL_SUCCESS);
    ASSERT_EQ(entry->SubmitCacheEntry(), HCCL_SUCCESS);
    uint64_t entryBytes = entry->GetEntryBytes();
    ASSERT_EQ(AicpuTaskCacheManager::aicpuTaskCache.IncCacheBytes("c_adpt_tag_1", entryBytes), HCCL_SUCCESS);

    bool isHit2 = false;
    EXPECT_EQ(HcommAicpuTsTaskCacheLookup("c_adpt_tag_1", &isHit2), HCCL_SUCCESS);
    EXPECT_TRUE(isHit2);
    EXPECT_EQ(AicpuTaskCacheManager::cacheEntryPtr, entry);
    EXPECT_TRUE(AicpuTaskCacheManager::aicpuTaskCache.cacheHitRunInfoPrinted_);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Lookup_CacheFull_MarksProcessRunInfoPrinted)
{
    AicpuTaskCacheManager::aicpuTaskCache.cacheFullRunInfoPrinted_ = false;
    AicpuTaskCacheManager::aicpuTaskCache.cacheBytes_ = AicpuTaskCacheManager::aicpuTaskCache.maxCacheBytes_;

    bool isHit = true;
    EXPECT_EQ(HcommAicpuTsTaskCacheLookup("c_adpt_full_tag_1", &isHit), HCCL_SUCCESS);
    EXPECT_FALSE(isHit);
    EXPECT_EQ(AicpuTaskCacheManager::cacheEntryPtr, nullptr);
    EXPECT_TRUE(AicpuTaskCacheManager::aicpuTaskCache.cacheFullRunInfoPrinted_);

    EXPECT_EQ(HcommAicpuTsTaskCacheLookup("c_adpt_full_tag_2", &isHit), HCCL_SUCCESS);
    EXPECT_FALSE(isHit);
    EXPECT_EQ(AicpuTaskCacheManager::cacheEntryPtr, nullptr);
    EXPECT_TRUE(AicpuTaskCacheManager::aicpuTaskCache.cacheFullRunInfoPrinted_);
}

// ===================== HcommAicpuTsTaskCacheStart Tests =====================

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Start_DeviceNotSupport_ReturnsNotSupport)
{
    SetDeviceTypeNotSupport();
    void* addrs[] = {(void*)TEST_BASE_ADDR_0, (void*)TEST_BASE_ADDR_1};
    uint64_t sizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    EXPECT_EQ(HcommAicpuTsTaskCacheStart("tag1", addrs, sizes, 2), HCCL_E_NOT_SUPPORT);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Start_Device960_ReturnsSuccess)
{
    SetDeviceType960();
    AicpuTaskCacheManager::cacheTag = "tag1";
    AicpuTaskCacheEntry entry;
    SetCacheMissState(&entry);
    void* addrs[] = {(void*)TEST_BASE_ADDR_0, (void*)TEST_BASE_ADDR_1};
    uint64_t sizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    EXPECT_EQ(HcommAicpuTsTaskCacheStart("tag1", addrs, sizes, 2), HCCL_SUCCESS);
    EXPECT_EQ(entry.cachedBaseAddrs_.size(), 2U);
    EXPECT_EQ(entry.cachedBaseAddrs_[0], TEST_BASE_ADDR_0);
    EXPECT_EQ(entry.cachedBaseAddrs_[1], TEST_BASE_ADDR_1);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Start_NullTag_ReturnsPtr)
{
    void* addrs[] = {(void*)TEST_BASE_ADDR_0, (void*)TEST_BASE_ADDR_1};
    uint64_t sizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    EXPECT_EQ(HcommAicpuTsTaskCacheStart(nullptr, addrs, sizes, 2), HCCL_E_PTR);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Start_NullAddrs_ReturnsPtr)
{
    uint64_t sizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    EXPECT_EQ(HcommAicpuTsTaskCacheStart("tag1", nullptr, sizes, 2), HCCL_E_PTR);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Start_NullSizes_ReturnsPtr)
{
    void* addrs[] = {(void*)TEST_BASE_ADDR_0, (void*)TEST_BASE_ADDR_1};
    EXPECT_EQ(HcommAicpuTsTaskCacheStart("tag1", addrs, nullptr, 2), HCCL_E_PTR);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Start_TagMismatch_ReturnsPara)
{
    AicpuTaskCacheManager::cacheTag = "lookup_tag";
    SetCacheMissState(nullptr);
    void* addrs[] = {(void*)TEST_BASE_ADDR_0, (void*)TEST_BASE_ADDR_1};
    uint64_t sizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    EXPECT_EQ(HcommAicpuTsTaskCacheStart("wrong_tag", addrs, sizes, 2), HCCL_E_PARA);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Start_CacheHit_ReturnsInternal)
{
    AicpuTaskCacheManager::cacheTag = "tag1";
    AicpuTaskCacheEntry entry;
    SetCacheHitState(&entry);
    void* addrs[] = {(void*)TEST_BASE_ADDR_0, (void*)TEST_BASE_ADDR_1};
    uint64_t sizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    EXPECT_EQ(HcommAicpuTsTaskCacheStart("tag1", addrs, sizes, 2), HCCL_E_INTERNAL);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Start_CacheFull_ReturnsSuccess)
{
    AicpuTaskCacheManager::cacheTag = "tag1";
    SetCacheFullState();
    void* addrs[] = {(void*)TEST_BASE_ADDR_0, (void*)TEST_BASE_ADDR_1};
    uint64_t sizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    EXPECT_EQ(HcommAicpuTsTaskCacheStart("tag1", addrs, sizes, 2), HCCL_SUCCESS);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Start_CacheMissWithEntry_InitSucceeds)
{
    AicpuTaskCacheManager::cacheTag = "tag1";
    AicpuTaskCacheEntry entry;
    SetCacheMissState(&entry);
    void* addrs[] = {(void*)TEST_BASE_ADDR_0, (void*)TEST_BASE_ADDR_1};
    uint64_t sizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    EXPECT_EQ(HcommAicpuTsTaskCacheStart("tag1", addrs, sizes, 2), HCCL_SUCCESS);
    EXPECT_EQ(entry.cachedBaseAddrs_.size(), 2U);
    EXPECT_EQ(entry.cachedBaseAddrs_[0], TEST_BASE_ADDR_0);
    EXPECT_EQ(entry.cachedBaseAddrs_[1], TEST_BASE_ADDR_1);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Start_CacheMissWithEntry_InvalidCount)
{
    AicpuTaskCacheManager::cacheTag = "tag1";
    AicpuTaskCacheEntry entry;
    SetCacheMissState(&entry);
    void* addrs[] = {(void*)TEST_BASE_ADDR_0};
    uint64_t sizes[] = {TEST_MEM_SIZE_0};
    EXPECT_EQ(HcommAicpuTsTaskCacheStart("tag1", addrs, sizes, 1), HCCL_E_PARA);
}

// ===================== HcommAicpuTsTaskCacheEnd Tests =====================

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Start_FailureDiscardsPendingEntryAndAllowsRetry)
{
    auto& cache = AicpuTaskCacheManager::aicpuTaskCache;
    cache.ClearEntry("c_adpt_tag_2");
    const uint64_t cacheBytes = cache.cacheBytes_;
    for (uint32_t scenario = 0; scenario < 4; ++scenario) {
        SCOPED_TRACE(scenario);
        bool isHit = true;
        ASSERT_EQ(HcommAicpuTsTaskCacheLookup("c_adpt_tag_2", &isHit), HCCL_SUCCESS);
        ASSERT_FALSE(isHit);
        ASSERT_NE(AicpuTaskCacheManager::cacheEntryPtr, nullptr);
        void* addrs[] = {(void*)TEST_BASE_ADDR_0, (void*)TEST_BASE_ADDR_1};
        uint64_t sizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
        if (scenario == 1) {
            addrs[0] = nullptr; // InitCacheEntry rejects the recording buffer.
        }
        const uint64_t count = scenario == 0 ? (uint64_t{1} << 63) | 3 : 2;
        EXPECT_EQ(
            HcommAicpuTsTaskCacheStart(
                "c_adpt_tag_2", scenario == 2 ? nullptr : addrs, scenario == 3 ? nullptr : sizes, count),
            scenario >= 2 ? HCCL_E_PTR : HCCL_E_PARA);
        EXPECT_EQ(cache.cacheHashMap_.count("c_adpt_tag_2"), 0U);
        EXPECT_EQ(cache.cacheBytes_, cacheBytes);
        EXPECT_EQ(AicpuTaskCacheManager::cacheEntryPtr, nullptr);
        EXPECT_TRUE(AicpuTaskCacheManager::cacheTag.empty());
        EXPECT_FALSE(AicpuTaskCacheManager::isHit);
        EXPECT_EQ(HcommAicpuTsTaskCacheEnd("c_adpt_tag_2"), HCCL_E_PARA);
        ASSERT_EQ(HcommAicpuTsTaskCacheLookup("c_adpt_tag_2", &isHit), HCCL_SUCCESS);
        EXPECT_FALSE(isHit);
        cache.ClearEntry("c_adpt_tag_2", false);
    }
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Start_InvalidCallPreservesOtherTagAndHitEntry)
{
    bool isHit = true;
    ASSERT_EQ(HcommAicpuTsTaskCacheLookup("c_adpt_tag_2", &isHit), HCCL_SUCCESS);
    ASSERT_FALSE(isHit);
    auto* entry = AicpuTaskCacheManager::cacheEntryPtr;
    void* addrs[] = {(void*)TEST_BASE_ADDR_0, (void*)TEST_BASE_ADDR_1};
    uint64_t sizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    EXPECT_EQ(HcommAicpuTsTaskCacheStart("wrong_tag", addrs, sizes, 2), HCCL_E_PARA);
    EXPECT_EQ(HcommAicpuTsTaskCacheStart("wrong_tag", nullptr, sizes, 2), HCCL_E_PTR);
    EXPECT_EQ(AicpuTaskCacheManager::cacheEntryPtr, entry);
    SetCacheHitState(entry);
    EXPECT_EQ(HcommAicpuTsTaskCacheStart("c_adpt_tag_2", addrs, sizes, 2), HCCL_E_INTERNAL);
    EXPECT_EQ(HcommAicpuTsTaskCacheStart("c_adpt_tag_2", nullptr, sizes, 2), HCCL_E_PTR);
    EXPECT_EQ(AicpuTaskCacheManager::cacheEntryPtr, entry);
    EXPECT_EQ(AicpuTaskCacheManager::aicpuTaskCache.cacheHashMap_.count("c_adpt_tag_2"), 1U);
    EXPECT_TRUE(AicpuTaskCacheManager::isHit);
    AicpuTaskCacheManager::aicpuTaskCache.ClearEntry("c_adpt_tag_2", false);
    AicpuTaskCacheManager::cacheEntryPtr = nullptr;
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, End_DeviceNotSupport_ReturnsNotSupport)
{
    SetDeviceTypeNotSupport();
    EXPECT_EQ(HcommAicpuTsTaskCacheEnd("tag1"), HCCL_E_NOT_SUPPORT);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, End_Device960_ReturnsSuccess)
{
    SetDeviceType960();
    AicpuTaskCacheManager::aicpuTaskCache.ClearEntry("c_adpt_tag_2");
    AicpuTaskCacheManager::cacheTag = "c_adpt_tag_2";
    AicpuTaskCacheEntry entry;
    SetCacheMissState(&entry);

    uint64_t baseAddrs[] = {TEST_BASE_ADDR_0, TEST_BASE_ADDR_1};
    uint64_t memSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    ASSERT_EQ(entry.InitCacheEntry(baseAddrs, memSizes, 2), HCCL_SUCCESS);
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
    ASSERT_EQ(
        entry.AddSqeArray(
            rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0, g_defaultTransports.data(),
            g_defaultLinkTypes.data(), g_defaultTransportTypes.data()),
        HCCL_SUCCESS);

    EXPECT_EQ(HcommAicpuTsTaskCacheEnd("c_adpt_tag_2"), HCCL_SUCCESS);
    EXPECT_EQ(AicpuTaskCacheManager::cacheTag, std::string(""));
    EXPECT_FALSE(AicpuTaskCacheManager::isHit);
    EXPECT_EQ(AicpuTaskCacheManager::cacheEntryPtr, nullptr);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, End_NullTag_ReturnsPtr)
{
    EXPECT_EQ(HcommAicpuTsTaskCacheEnd(nullptr), HCCL_E_PTR);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, End_TagMismatch_ReturnsPara)
{
    AicpuTaskCacheManager::cacheTag = "lookup_tag";
    SetCacheMissState(nullptr);
    EXPECT_EQ(HcommAicpuTsTaskCacheEnd("wrong_tag"), HCCL_E_PARA);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, End_CacheHit_ReturnsInternal)
{
    AicpuTaskCacheManager::cacheTag = "tag1";
    AicpuTaskCacheEntry entry;
    SetCacheHitState(&entry);
    EXPECT_EQ(HcommAicpuTsTaskCacheEnd("tag1"), HCCL_E_INTERNAL);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, End_CacheFull_ReturnsSuccess)
{
    AicpuTaskCacheManager::cacheTag = "tag1";
    SetCacheFullState();
    EXPECT_EQ(HcommAicpuTsTaskCacheEnd("tag1"), HCCL_SUCCESS);
    EXPECT_EQ(AicpuTaskCacheManager::cacheTag, std::string(""));
    EXPECT_FALSE(AicpuTaskCacheManager::isHit);
    EXPECT_EQ(AicpuTaskCacheManager::cacheEntryPtr, nullptr);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, End_CacheMissWithUnsubmittedEntry_ReturnsInternal)
{
    AicpuTaskCacheManager::cacheTag = "tag1";
    AicpuTaskCacheEntry entry;
    SetCacheMissState(&entry);
    EXPECT_EQ(HcommAicpuTsTaskCacheEnd("tag1"), HCCL_E_INTERNAL);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, End_CacheMissWithSubmittedEntry_ReturnsSuccess)
{
    AicpuTaskCacheManager::aicpuTaskCache.ClearEntry("c_adpt_tag_2");
    AicpuTaskCacheManager::cacheTag = "c_adpt_tag_2";
    AicpuTaskCacheEntry entry;
    SetCacheMissState(&entry);

    uint64_t baseAddrs[] = {TEST_BASE_ADDR_0, TEST_BASE_ADDR_1};
    uint64_t memSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    ASSERT_EQ(entry.InitCacheEntry(baseAddrs, memSizes, 2), HCCL_SUCCESS);
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
    ASSERT_EQ(
        entry.AddSqeArray(
            rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0, g_defaultTransports.data(),
            g_defaultLinkTypes.data(), g_defaultTransportTypes.data()),
        HCCL_SUCCESS);

    EXPECT_EQ(HcommAicpuTsTaskCacheEnd("c_adpt_tag_2"), HCCL_SUCCESS);
    EXPECT_EQ(AicpuTaskCacheManager::cacheTag, std::string(""));
    EXPECT_FALSE(AicpuTaskCacheManager::isHit);
    EXPECT_EQ(AicpuTaskCacheManager::cacheEntryPtr, nullptr);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, End_SubmitFailureRemovesEntryWithoutChangingCommittedCacheBytes)
{
    auto& cache = AicpuTaskCacheManager::aicpuTaskCache;
    uint64_t baseAddrs[] = {TEST_BASE_ADDR_0, TEST_BASE_ADDR_1};
    uint64_t memSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    bool isHit = true;
    ASSERT_EQ(HcommAicpuTsTaskCacheLookup("c_adpt_tag_1", &isHit), HCCL_SUCCESS);
    ASSERT_FALSE(isHit);
    auto* committed = AicpuTaskCacheManager::cacheEntryPtr;
    ASSERT_NE(committed, nullptr);
    ASSERT_EQ(committed->InitCacheEntry(baseAddrs, memSizes, 2), HCCL_SUCCESS);
    ASSERT_EQ(HcommAicpuTsTaskCacheEnd("c_adpt_tag_1"), HCCL_SUCCESS);
    const uint64_t committedBytes = cache.cacheBytes_;
    ASSERT_GT(committedBytes, 0U);

    // 未初始化，以及V地址匹配的歧义、分片、无匹配三种失败。
    for (uint32_t scenario = 0; scenario < 4; ++scenario) {
        SCOPED_TRACE(scenario);
        ASSERT_EQ(HcommAicpuTsTaskCacheLookup("c_adpt_tag_2", &isHit), HCCL_SUCCESS);
        ASSERT_FALSE(isHit);
        auto* entry = AicpuTaskCacheManager::cacheEntryPtr;
        ASSERT_NE(entry, nullptr);
        if (scenario != 0) {
            VOpInfo vOpInfo{{8, 8}, {8, 8}, {0, 16}, {0, 16}};
            if (scenario == 1) {
                vOpInfo.sdispls[1] = 0;
            }
            ASSERT_EQ(entry->InitCacheEntry(baseAddrs, memSizes, 2, vOpInfo), HCCL_SUCCESS);
            auto sqe = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA);
            auto* sdma = reinterpret_cast<Hccl::Rt91095StarsMemcpySqe*>(sqe.data());
            const uint64_t offset = scenario == 1 ? 0 : (scenario == 2 ? 1 : 8);
            const uint64_t srcAddr = baseAddrs[0] + offset;
            sdma->u.strideMode0.srcAddrLow = static_cast<uint32_t>(srcAddr);
            sdma->u.strideMode0.srcAddrHigh = static_cast<uint32_t>(srcAddr >> 32);
            sdma->u.strideMode0.dstAddrLow = static_cast<uint32_t>(baseAddrs[1]);
            sdma->u.strideMode0.dstAddrHigh = static_cast<uint32_t>(baseAddrs[1] >> 32);
            sdma->u.strideMode0.lengthMove = 8;
            ASSERT_EQ(
                entry->AddSqeArray(
                    rtsqPtr_, aicpuTsThread_.get(), 1, sqe.data(), 0, g_defaultTransports.data(),
                    g_defaultLinkTypes.data(), g_defaultTransportTypes.data()),
                HCCL_SUCCESS);
        }
        EXPECT_EQ(HcommAicpuTsTaskCacheEnd("c_adpt_tag_2"), scenario == 0 ? HCCL_E_INTERNAL : HCCL_E_PARA);
        EXPECT_EQ(cache.cacheHashMap_.count("c_adpt_tag_2"), 0U);
        EXPECT_EQ(cache.cacheBytes_, committedBytes);
        EXPECT_TRUE(AicpuTaskCacheManager::cacheTag.empty());
        EXPECT_FALSE(AicpuTaskCacheManager::isHit);
        EXPECT_EQ(AicpuTaskCacheManager::cacheEntryPtr, nullptr);

        ASSERT_EQ(HcommAicpuTsTaskCacheLookup("c_adpt_tag_2", &isHit), HCCL_SUCCESS);
        ASSERT_FALSE(isHit);
        ASSERT_NE(AicpuTaskCacheManager::cacheEntryPtr, nullptr);
        // 同tag重新录制成功，之后才允许命中。
        ASSERT_EQ(AicpuTaskCacheManager::cacheEntryPtr->InitCacheEntry(baseAddrs, memSizes, 2), HCCL_SUCCESS);
        ASSERT_EQ(HcommAicpuTsTaskCacheEnd("c_adpt_tag_2"), HCCL_SUCCESS);
        ASSERT_EQ(HcommAicpuTsTaskCacheLookup("c_adpt_tag_2", &isHit), HCCL_SUCCESS);
        EXPECT_TRUE(isHit);
        ASSERT_EQ(HcommAicpuTsTaskCacheClear("c_adpt_tag_2"), HCCL_SUCCESS);
        EXPECT_EQ(cache.cacheBytes_, committedBytes);
        AicpuTaskCacheEntry* found = nullptr;
        ASSERT_EQ(cache.FindEntry("c_adpt_tag_1", &found), HCCL_SUCCESS);
        EXPECT_EQ(found, committed);
    }
}

// ===================== HcommAicpuTsTaskCacheExecute Tests =====================

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Execute_DeviceNotSupport_ReturnsNotSupport)
{
    SetDeviceTypeNotSupport();
    void* addrs[] = {(void*)TEST_BASE_ADDR_0, (void*)TEST_BASE_ADDR_1};
    uint64_t sizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    EXPECT_EQ(HcommAicpuTsTaskCacheExecute("tag1", addrs, sizes, 2), HCCL_E_NOT_SUPPORT);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Execute_Device960_CountMismatch_ReturnsInternal)
{
    SetDeviceType960();
    AicpuTaskCacheManager::cacheTag = "tag1";
    AicpuTaskCacheEntry entry;
    uint64_t baseAddrs[] = {TEST_BASE_ADDR_0, TEST_BASE_ADDR_1};
    uint64_t memSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    ASSERT_EQ(entry.InitCacheEntry(baseAddrs, memSizes, 2), HCCL_SUCCESS);
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
    ASSERT_EQ(
        entry.AddSqeArray(
            rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0, g_defaultTransports.data(),
            g_defaultLinkTypes.data(), g_defaultTransportTypes.data()),
        HCCL_SUCCESS);
    ASSERT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);
    SetCacheHitState(&entry);

    // 960 通过门控后进入 Execute 主体, count 不匹配返回 INTERNAL (UT 环境无法走完整 RefreshAndLaunch)
    void* addrs[] = {(void*)TEST_BASE_ADDR_0};
    uint64_t sizes[] = {TEST_MEM_SIZE_0};
    EXPECT_EQ(HcommAicpuTsTaskCacheExecute("tag1", addrs, sizes, 1), HCCL_E_INTERNAL);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Execute_NullTag_ReturnsPtr)
{
    void* addrs[] = {(void*)TEST_BASE_ADDR_0, (void*)TEST_BASE_ADDR_1};
    uint64_t sizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    EXPECT_EQ(HcommAicpuTsTaskCacheExecute(nullptr, addrs, sizes, 2), HCCL_E_PTR);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Execute_NullAddrs_ReturnsPtr)
{
    uint64_t sizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    EXPECT_EQ(HcommAicpuTsTaskCacheExecute("tag1", nullptr, sizes, 2), HCCL_E_PTR);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Execute_NullSizes_ReturnsPtr)
{
    void* addrs[] = {(void*)TEST_BASE_ADDR_0, (void*)TEST_BASE_ADDR_1};
    EXPECT_EQ(HcommAicpuTsTaskCacheExecute("tag1", addrs, nullptr, 2), HCCL_E_PTR);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Execute_TagMismatch_ReturnsPara)
{
    AicpuTaskCacheManager::cacheTag = "lookup_tag";
    AicpuTaskCacheEntry entry;
    SetCacheHitState(&entry);
    void* addrs[] = {(void*)TEST_BASE_ADDR_0, (void*)TEST_BASE_ADDR_1};
    uint64_t sizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    EXPECT_EQ(HcommAicpuTsTaskCacheExecute("wrong_tag", addrs, sizes, 2), HCCL_E_PARA);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Execute_CacheMiss_ReturnsInternal)
{
    AicpuTaskCacheManager::cacheTag = "tag1";
    AicpuTaskCacheEntry entry;
    SetCacheMissState(&entry);
    void* addrs[] = {(void*)TEST_BASE_ADDR_0, (void*)TEST_BASE_ADDR_1};
    uint64_t sizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    EXPECT_EQ(HcommAicpuTsTaskCacheExecute("tag1", addrs, sizes, 2), HCCL_E_INTERNAL);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Execute_CacheHitWithNullPtr_ReturnsPtr)
{
    AicpuTaskCacheManager::cacheTag = "tag1";
    AicpuTaskCacheManager::isHit = true;
    AicpuTaskCacheManager::cacheEntryPtr = nullptr;
    void* addrs[] = {(void*)TEST_BASE_ADDR_0, (void*)TEST_BASE_ADDR_1};
    uint64_t sizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    EXPECT_EQ(HcommAicpuTsTaskCacheExecute("tag1", addrs, sizes, 2), HCCL_E_PTR);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Execute_CountMismatch_ReturnsInternal)
{
    AicpuTaskCacheManager::cacheTag = "tag1";
    AicpuTaskCacheEntry entry;
    uint64_t baseAddrs[] = {TEST_BASE_ADDR_0, TEST_BASE_ADDR_1};
    uint64_t memSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    ASSERT_EQ(entry.InitCacheEntry(baseAddrs, memSizes, 2), HCCL_SUCCESS);
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
    ASSERT_EQ(
        entry.AddSqeArray(
            rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0, g_defaultTransports.data(),
            g_defaultLinkTypes.data(), g_defaultTransportTypes.data()),
        HCCL_SUCCESS);
    ASSERT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);
    SetCacheHitState(&entry);

    void* addrs[] = {(void*)TEST_BASE_ADDR_0};
    uint64_t sizes[] = {TEST_MEM_SIZE_0};
    EXPECT_EQ(HcommAicpuTsTaskCacheExecute("tag1", addrs, sizes, 1), HCCL_E_INTERNAL);
}

// ===================== HcommAicpuTsTaskCacheClear Tests =====================

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Clear_DeviceNotSupport_ReturnsNotSupport)
{
    SetDeviceTypeNotSupport();
    EXPECT_EQ(HcommAicpuTsTaskCacheClear("tag1"), HCCL_E_NOT_SUPPORT);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Clear_Device960_ReturnsSuccess)
{
    SetDeviceType960();
    AicpuTaskCacheManager::aicpuTaskCache.ClearEntry("c_adpt_tag_1");
    AicpuTaskCacheEntry* entryPtr = nullptr;
    ASSERT_EQ(AicpuTaskCacheManager::aicpuTaskCache.AddEntry("c_adpt_tag_1", &entryPtr), HCCL_SUCCESS);
    ASSERT_NE(entryPtr, nullptr);

    EXPECT_EQ(HcommAicpuTsTaskCacheClear("c_adpt_tag_1"), HCCL_SUCCESS);

    AicpuTaskCacheEntry* foundPtr = nullptr;
    AicpuTaskCacheManager::aicpuTaskCache.FindEntry("c_adpt_tag_1", &foundPtr);
    EXPECT_EQ(foundPtr, nullptr);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Clear_NullTag_ReturnsPtr)
{
    EXPECT_EQ(HcommAicpuTsTaskCacheClear(nullptr), HCCL_E_PTR);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Clear_NonExistentEntry_ReturnsSuccess)
{
    EXPECT_EQ(HcommAicpuTsTaskCacheClear("non_existent_tag"), HCCL_SUCCESS);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Clear_ExistingEntry_ReturnsSuccess)
{
    AicpuTaskCacheManager::aicpuTaskCache.ClearEntry("c_adpt_tag_1");
    AicpuTaskCacheEntry* entryPtr = nullptr;
    ASSERT_EQ(AicpuTaskCacheManager::aicpuTaskCache.AddEntry("c_adpt_tag_1", &entryPtr), HCCL_SUCCESS);
    ASSERT_NE(entryPtr, nullptr);

    EXPECT_EQ(HcommAicpuTsTaskCacheClear("c_adpt_tag_1"), HCCL_SUCCESS);

    AicpuTaskCacheEntry* foundPtr = nullptr;
    AicpuTaskCacheManager::aicpuTaskCache.FindEntry("c_adpt_tag_1", &foundPtr);
    EXPECT_EQ(foundPtr, nullptr);
}

// ===================== Full Workflow Tests =====================

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Workflow_LookupStartAddEndLookupExecute)
{
    AicpuTaskCacheManager::aicpuTaskCache.ClearEntry("c_adpt_full_workflow");

    // Step 1: Lookup (cache miss)
    bool isHit = true;
    EXPECT_EQ(HcommAicpuTsTaskCacheLookup("c_adpt_full_workflow", &isHit), HCCL_SUCCESS);
    EXPECT_FALSE(isHit);
    AicpuTaskCacheEntry* entry = AicpuTaskCacheManager::cacheEntryPtr;
    ASSERT_NE(entry, nullptr);

    // Step 2: Start (init cache entry)
    void* addrs[] = {(void*)TEST_BASE_ADDR_0, (void*)TEST_BASE_ADDR_1};
    uint64_t sizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    EXPECT_EQ(HcommAicpuTsTaskCacheStart("c_adpt_full_workflow", addrs, sizes, 2), HCCL_SUCCESS);

    // Step 3: Add SQE to entry via manager
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
    EXPECT_EQ(
        AicpuTaskCacheManager::AddSqeArray(
            rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0, g_defaultTransports.data(),
            g_defaultLinkTypes.data(), g_defaultTransportTypes.data()),
        HCCL_SUCCESS);

    // Step 4: End (submit cache entry)
    EXPECT_EQ(HcommAicpuTsTaskCacheEnd("c_adpt_full_workflow"), HCCL_SUCCESS);
    EXPECT_EQ(AicpuTaskCacheManager::cacheTag, std::string(""));
    EXPECT_FALSE(AicpuTaskCacheManager::isHit);
    EXPECT_EQ(AicpuTaskCacheManager::cacheEntryPtr, nullptr);

    // Step 5: Lookup again (cache hit)
    bool isHit2 = false;
    EXPECT_EQ(HcommAicpuTsTaskCacheLookup("c_adpt_full_workflow", &isHit2), HCCL_SUCCESS);
    EXPECT_TRUE(isHit2);
    EXPECT_EQ(AicpuTaskCacheManager::cacheEntryPtr, entry);

    // Step 6: Execute (count mismatch returns error, does not reset context)
    void* wrongAddrs[] = {(void*)TEST_BASE_ADDR_0};
    uint64_t wrongSizes[] = {TEST_MEM_SIZE_0};
    EXPECT_EQ(HcommAicpuTsTaskCacheExecute("c_adpt_full_workflow", wrongAddrs, wrongSizes, 1), HCCL_E_INTERNAL);
}
