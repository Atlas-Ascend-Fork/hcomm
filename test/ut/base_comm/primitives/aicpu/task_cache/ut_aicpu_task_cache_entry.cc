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
#include <atomic>
#include "dfx/device/dfx_res_pub.h"
#include <vector>

#include "gtest/gtest.h"

#ifndef private
#define private public
#define protected public
#endif

#include "aicpu_task_cache_entry.h"
#include "aicpu_task_cache_manager.h"
#include "ub_conn_lite.h"
#include "ub_transport_lite_impl.h"
#include "rtsq_a5.h"
#include "aicpu_ts_thread.h"
#include "sqe_v82.h"
#include "udma_data_struct.h"
#include "ub_jetty_lite.h"
#include "ip_address.h"
#include "config_plf_log_v2.h"
#include "stream_lite.h"

#undef private
#undef protected

using namespace hcomm;
using namespace Hccl;

namespace {
std::vector<u64> g_defaultTransports(128, DFX_INVALID_U64);
std::vector<u8> g_defaultLinkTypes(128, static_cast<u8>(DfxLinkTypeVal::LINK_ONCHIP));
std::vector<u8> g_defaultTransportTypes(128, static_cast<u8>(DfxTransportType::DFX_TRANSPORT_TYPE_LOCAL));
} // namespace

TEST(AicpuTaskCacheEntryDebugConfigTest, TaskDebugFlagFollowsPlfTaskConfig)
{
    Hccl::SetPlfDebugConfigValue(0ULL);
    AicpuTaskCacheEntry disabledEntry;
    EXPECT_EQ(disabledEntry.isTaskConfigDebug_, HcclCheckLogLevel(HCCL_LOG_DEBUG));

    Hccl::SetPlfDebugConfigValue(Hccl::PLF_TASK);
    AicpuTaskCacheEntry enabledEntry;
    EXPECT_TRUE(enabledEntry.isTaskConfigDebug_);
    Hccl::SetPlfDebugConfigValue(0ULL);
}

// Provide strong definition for weak symbol aicpu::GetSqeId (called by RtsqBase constructor)
namespace aicpu {
void GetSqeId(const uint32_t num, uint32_t& start, uint32_t& end)
{
    start = 1;
    end = start + num;
}
} // namespace aicpu

namespace {
constexpr uint64_t TEST_BASE_ADDR_0 = 0x10000ULL;
constexpr uint64_t TEST_MEM_SIZE_0 = 0x1000ULL;
constexpr uint64_t TEST_BASE_ADDR_1 = 0x20000ULL;
constexpr uint64_t TEST_MEM_SIZE_1 = 0x2000ULL;
constexpr uint64_t TEST_REMOTE_BASE_ADDR = 0x30000ULL;
constexpr uint64_t TEST_REMOTE_MEM_SIZE = 0x3000ULL;
constexpr uint32_t TEST_SQ_DEPTH = 128;

ChannelHandle queriedWqeChannel = 0;
uint32_t wqeRankQueryCount = 0;
HcclResult GetWqeRemoteRankForTest(ChannelHandle channel, u32& rank)
{
    queriedWqeChannel = channel;
    ++wqeRankQueryCount;
    rank = 1;
    return HCCL_SUCCESS;
}

HcclResult GetWqeRemoteRankFailureForTest(ChannelHandle, u32&) { return HCCL_E_PARA; }

std::vector<uint8_t>
MakeSqeArray(Rt91095StarsSqeType sqeType, uint64_t srcAddr = 0, uint64_t dstAddr = 0, uint32_t copyBytes = 1)
{
    std::vector<uint8_t> sqeArray(Hccl::AC_SQE_SIZE, 0);
    Rt91095StarsSqeHeader* header = reinterpret_cast<Rt91095StarsSqeHeader*>(sqeArray.data());
    header->type = static_cast<uint8_t>(sqeType);

    if (sqeType == Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA) {
        Rt91095StarsMemcpySqe* sdmaSqe = reinterpret_cast<Rt91095StarsMemcpySqe*>(sqeArray.data());
        sdmaSqe->u.strideMode0.srcAddrLow = static_cast<uint32_t>(srcAddr);
        sdmaSqe->u.strideMode0.srcAddrHigh = static_cast<uint32_t>(srcAddr >> 32);
        sdmaSqe->u.strideMode0.dstAddrLow = static_cast<uint32_t>(dstAddr);
        sdmaSqe->u.strideMode0.dstAddrHigh = static_cast<uint32_t>(dstAddr >> 32);
        sdmaSqe->u.strideMode0.lengthMove = copyBytes;
    } else if (sqeType == Rt91095StarsSqeType::RT_91095_SQE_TYPE_WRITE_VALUE) {
        Rt91095StarsWriteValueSqe* wvSqe = reinterpret_cast<Rt91095StarsWriteValueSqe*>(sqeArray.data());
        wvSqe->writeAddrLow = static_cast<uint32_t>(dstAddr);
        wvSqe->writeAddrHigh = static_cast<uint32_t>(dstAddr >> 32);
    }
    return sqeArray;
}

Hccl::WqeTask MakeWqeTaskRead(uint64_t locAddr, uint64_t rmtAddr)
{
    Hccl::WqeTask wqe;
    memset(&wqe, 0, sizeof(wqe));
    UdmaSqeCommon* common = reinterpret_cast<UdmaSqeCommon*>(&wqe);
    common->opcode = static_cast<uint8_t>(Hccl::UdmaSqOpcode::UDMA_OPC_READ);
    wqe.wqeWrite.u.sge.dataAddrLow = static_cast<uint32_t>(locAddr);
    wqe.wqeWrite.u.sge.dataAddrHigh = static_cast<uint32_t>(locAddr >> 32);
    wqe.wqeWrite.comm.rmtAddrLow = static_cast<uint32_t>(rmtAddr);
    wqe.wqeWrite.comm.rmtAddrHigh = static_cast<uint32_t>(rmtAddr >> 32);
    return wqe;
}

Hccl::WqeTask MakeWqeTaskWrite(uint64_t locAddr, uint64_t rmtAddr, bool inlineEn)
{
    Hccl::WqeTask wqe;
    memset(&wqe, 0, sizeof(wqe));
    UdmaSqeCommon* common = reinterpret_cast<UdmaSqeCommon*>(&wqe);
    common->opcode = static_cast<uint8_t>(Hccl::UdmaSqOpcode::UDMA_OPC_WRITE);
    common->inlineEn = inlineEn ? 1 : 0;
    wqe.wqeWrite.u.sge.dataAddrLow = static_cast<uint32_t>(locAddr);
    wqe.wqeWrite.u.sge.dataAddrHigh = static_cast<uint32_t>(locAddr >> 32);
    wqe.wqeWrite.comm.rmtAddrLow = static_cast<uint32_t>(rmtAddr);
    wqe.wqeWrite.comm.rmtAddrHigh = static_cast<uint32_t>(rmtAddr >> 32);
    return wqe;
}

Hccl::WqeTask MakeWqeTaskWriteWithNotify(uint64_t locAddr, uint64_t rmtAddr)
{
    Hccl::WqeTask wqe;
    memset(&wqe, 0, sizeof(wqe));
    UdmaSqeCommon* common = reinterpret_cast<UdmaSqeCommon*>(&wqe);
    common->opcode = static_cast<uint8_t>(WRITE_WITH_NOTIFY_OPCODE);
    wqe.wqeWriteWithNotify.localU.sge.dataAddrLow = static_cast<uint32_t>(locAddr);
    wqe.wqeWriteWithNotify.localU.sge.dataAddrHigh = static_cast<uint32_t>(locAddr >> 32);
    wqe.wqeWriteWithNotify.comm.rmtAddrLow = static_cast<uint32_t>(rmtAddr);
    wqe.wqeWriteWithNotify.comm.rmtAddrHigh = static_cast<uint32_t>(rmtAddr >> 32);
    return wqe;
}

Hccl::WqeTask MakeWqeTaskInvalid()
{
    Hccl::WqeTask wqe;
    memset(&wqe, 0, sizeof(wqe));
    UdmaSqeCommon* common = reinterpret_cast<UdmaSqeCommon*>(&wqe);
    common->opcode = 0xFF;
    return wqe;
}

Hccl::DbSqeProfInfo MakeDbSqeProfInfo(bool isValid)
{
    Hccl::DbSqeProfInfo info{};
    memset(&info, 0, sizeof(info));
    info.isValid = isValid;
    return info;
}
} // namespace

// ===================== AddrRefreshInfo Tests =====================

TEST(AddrRefreshInfoTest, DefaultConstructor)
{
    AddrRefreshInfo info;
    EXPECT_FALSE(info.needRefresh);
    EXPECT_EQ(info.memIdx, 0U);
}

TEST(AddrRefreshInfoTest, ParamConstructor)
{
    AddrRefreshInfo info(42);
    EXPECT_TRUE(info.needRefresh);
    EXPECT_EQ(info.memIdx, 42U);
}

TEST(AddrRefreshInfoTest, CopyConstructor)
{
    AddrRefreshInfo orig(7);
    orig.rankId = 2;
    AddrRefreshInfo copy(orig);
    EXPECT_TRUE(copy.needRefresh);
    EXPECT_EQ(copy.memIdx, 7U);
    EXPECT_EQ(copy.rankId, 2U);
}

TEST(AddrRefreshInfoTest, AssignmentOperator)
{
    AddrRefreshInfo orig(5);
    orig.rankId = 3;
    AddrRefreshInfo assigned;
    assigned = orig;
    EXPECT_TRUE(assigned.needRefresh);
    EXPECT_EQ(assigned.memIdx, 5U);
    EXPECT_EQ(assigned.rankId, 3U);
}

TEST(AddrRefreshInfoTest, SelfAssignment)
{
    AddrRefreshInfo info(3);
    info = info;
    EXPECT_TRUE(info.needRefresh);
    EXPECT_EQ(info.memIdx, 3U);
}

// ===================== AicpuTaskCacheEntry Fixture =====================

class AicpuTaskCacheEntryTest : public testing::Test {
protected:
    std::unique_ptr<Hccl::RtsqA5> rtsq_;
    Hccl::RtsqA5* rtsqPtr_ = nullptr;
    std::unique_ptr<hccl::AicpuTsThread> aicpuTsThread_;
    std::unique_ptr<Hccl::UbConnLite> ubConnLite_;
    std::unique_ptr<Hccl::UbTransportLiteImpl> ubTransport_;
    std::vector<char> emptyUniqueId_;
    AicpuTaskCacheManager::GetChannelRemoteRankIdCallback savedRankCallback_ = nullptr;

    void SetUp() override
    {
        savedRankCallback_ = AicpuTaskCacheManager::getChannelRemoteRankIdCallback_.load(std::memory_order_acquire);
        queriedWqeChannel = 0;
        wqeRankQueryCount = 0;
        rtsq_ = std::make_unique<Hccl::RtsqA5>(0, 0, 0);
        rtsqPtr_ = rtsq_.get();
        aicpuTsThread_ = std::make_unique<hccl::AicpuTsThread>(std::string("ut_thread"));
        ubConnLite_ = std::make_unique<Hccl::UbConnLite>(
            Hccl::UbJettyLiteId(0, 0, 0), Hccl::UbJettyLiteAttr(0, 0, TEST_SQ_DEPTH, 0, true), Hccl::Eid{});
        ubTransport_ = std::make_unique<Hccl::UbTransportLiteImpl>(emptyUniqueId_);
    }

    void TearDown() override { AicpuTaskCacheManager::RegisterGetChannelRemoteRankIdCallback(savedRankCallback_); }

    HcclResult InitEntryWithTwoAddrs(hcomm::AicpuTaskCacheEntry& entry)
    {
        uint64_t baseAddrs[] = {TEST_BASE_ADDR_0, TEST_BASE_ADDR_1};
        uint64_t memSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
        return entry.InitCacheEntry(baseAddrs, memSizes, 2);
    }

    // AddWqeArray always creates a DbSqeTmpInfo entry that must be cleared by AddSqeArray
    // with the same streamId before SubmitCacheEntry can succeed.
    void AddUbdmaSqeToClearTmpMap(hcomm::AicpuTaskCacheEntry& entry, uint32_t streamId = 0)
    {
        auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_UBDMA);
        ASSERT_EQ(
            entry.AddSqeArray(
                rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), streamId, g_defaultTransports.data(),
                g_defaultLinkTypes.data(), g_defaultTransportTypes.data()),
            HCCL_SUCCESS);
    }
};

// ===================== GetUbLinkTypeVal Tests =====================

TEST_F(AicpuTaskCacheEntryTest, GetUbLinkTypeVal_Returns_LINK_UB_When_linkType_Is_UB)
{
    hcomm::AicpuTaskCacheEntry entry;
    ubTransport_->linkType_ = Hccl::DfxLinkType::UB;
    EXPECT_EQ(entry.GetUbLinkTypeVal_(ubTransport_.get()), static_cast<u8>(Hccl::DfxLinkTypeVal::LINK_UB));
}

TEST_F(AicpuTaskCacheEntryTest, GetUbLinkTypeVal_Returns_LINK_UBoE_When_linkType_Is_UBoE)
{
    hcomm::AicpuTaskCacheEntry entry;
    ubTransport_->linkType_ = Hccl::DfxLinkType::UBoE;
    EXPECT_EQ(entry.GetUbLinkTypeVal_(ubTransport_.get()), static_cast<u8>(Hccl::DfxLinkTypeVal::LINK_UBoE));
}

// ===================== Constructor / Destructor Tests =====================

TEST_F(AicpuTaskCacheEntryTest, Constructor_Default)
{
    hcomm::AicpuTaskCacheEntry entry;
    EXPECT_EQ(entry.entryBytes_, 0U);
    EXPECT_EQ(entry.sqeArrayInfos_.size(), 0U);
    EXPECT_EQ(entry.wqeTaskArrayInfos_.size(), 0U);
}

TEST_F(AicpuTaskCacheEntryTest, Destructor_EmptyEntry_NoCrash)
{
    {
        hcomm::AicpuTaskCacheEntry entry;
    }
    SUCCEED();
}

TEST_F(AicpuTaskCacheEntryTest, Destructor_WithSqeArray_FreesMemory)
{
    {
        hcomm::AicpuTaskCacheEntry entry;
        ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
        auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
        ASSERT_EQ(
            entry.AddSqeArray(
                rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0, g_defaultTransports.data(),
                g_defaultLinkTypes.data(), g_defaultTransportTypes.data()),
            HCCL_SUCCESS);
    }
    SUCCEED();
}

// ===================== InitCacheEntry Tests =====================

TEST_F(AicpuTaskCacheEntryTest, InitCacheEntry_Success)
{
    hcomm::AicpuTaskCacheEntry entry;
    uint64_t baseAddrs[] = {TEST_BASE_ADDR_0, TEST_BASE_ADDR_1};
    uint64_t memSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};

    EXPECT_EQ(entry.InitCacheEntry(baseAddrs, memSizes, 2), HCCL_SUCCESS);
    EXPECT_EQ(entry.cachedBaseAddrs_.size(), 2U);
    EXPECT_EQ(entry.cachedMemSizes_.size(), 2U);
    EXPECT_EQ(entry.cachedBaseAddrs_[0], TEST_BASE_ADDR_0);
    EXPECT_EQ(entry.cachedBaseAddrs_[1], TEST_BASE_ADDR_1);
    EXPECT_EQ(entry.cachedMemSizes_[0], TEST_MEM_SIZE_0);
    EXPECT_EQ(entry.cachedMemSizes_[1], TEST_MEM_SIZE_1);
    EXPECT_GT(entry.entryBytes_, 0U);
}

TEST_F(AicpuTaskCacheEntryTest, InitCacheEntry_VOpInfoCopiesAndReplacesArrays)
{
    AicpuTaskCacheEntry entry;
    uint64_t baseAddrs[] = {TEST_BASE_ADDR_0, TEST_BASE_ADDR_1};
    uint64_t memSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    entry.vOpInfo_ = VOpInfo{{9}, {8}, {7}, {6}};
    entry.entryBytes_ = entry.vOpInfo_.GetSize();
    VOpInfo input{{0, 3}, {2, 0}, {0, 4}, {1, 5}};
    const VOpInfo expected = input;

    ASSERT_EQ(entry.InitCacheEntry(baseAddrs, memSizes, 2, input), HCCL_SUCCESS);
    EXPECT_TRUE(entry.isVOp_);
    input.sendCounts.clear();
    input.recvCounts.clear();
    input.sdispls.clear();
    input.rdispls.clear();
    EXPECT_EQ(entry.vOpInfo_.sendCounts, expected.sendCounts);
    EXPECT_EQ(entry.vOpInfo_.recvCounts, expected.recvCounts);
    EXPECT_EQ(entry.vOpInfo_.sdispls, expected.sdispls);
    EXPECT_EQ(entry.vOpInfo_.rdispls, expected.rdispls);
    EXPECT_EQ(entry.GetEntryBytes(), 4 * sizeof(uint64_t) + expected.GetSize());
}

TEST_F(AicpuTaskCacheEntryTest, InitCacheEntry_RejectsUnequalVOpInfoSizes)
{
    uint64_t baseAddrs[] = {TEST_BASE_ADDR_0, TEST_BASE_ADDR_1};
    uint64_t memSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    const VOpInfo inputs[] = {
        {{1}, {1, 1}, {0}, {0}},
        {{1}, {1}, {0, 0}, {0}},
        {{1}, {1}, {0}, {0, 0}},
        {{1, 1}, {1}, {0, 0}, {0, 0}},
    };
    for (const auto& input : inputs) {
        AicpuTaskCacheEntry entry;
        EXPECT_EQ(entry.InitCacheEntry(baseAddrs, memSizes, 2, input), HCCL_E_PARA);
        EXPECT_FALSE(entry.isVOp_);
        EXPECT_TRUE(entry.cachedBaseAddrs_.empty());
        EXPECT_TRUE(entry.cachedMemSizes_.empty());
    }
}

TEST_F(AicpuTaskCacheEntryTest, InitCacheEntry_IncompleteVOpInfoPreservesAllArrays)
{
    uint64_t baseAddrs[] = {TEST_BASE_ADDR_0, TEST_BASE_ADDR_1};
    uint64_t memSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    const VOpInfo expected{{9}, {8}, {7}, {6}};
    const VOpInfo inputs[] = {{{}, {2}, {0}, {0}}, {{1}, {}, {0}, {0}}, {{1}, {2}, {}, {0}}, {{1}, {2}, {0}, {}}, {}};
    for (const auto& input : inputs) {
        AicpuTaskCacheEntry entry;
        entry.vOpInfo_ = expected;
        entry.isVOp_ = true;
        entry.entryBytes_ = expected.GetSize();
        ASSERT_EQ(entry.InitCacheEntry(baseAddrs, memSizes, 2, input), HCCL_SUCCESS);
        EXPECT_TRUE(entry.isVOp_);
        EXPECT_EQ(entry.vOpInfo_.sendCounts, expected.sendCounts);
        EXPECT_EQ(entry.vOpInfo_.recvCounts, expected.recvCounts);
        EXPECT_EQ(entry.vOpInfo_.sdispls, expected.sdispls);
        EXPECT_EQ(entry.vOpInfo_.rdispls, expected.rdispls);
        EXPECT_EQ(entry.GetEntryBytes(), 4 * sizeof(uint64_t) + expected.GetSize());
    }
}

TEST_F(AicpuTaskCacheEntryTest, InitCacheEntry_DefaultVOpInfoRemainsEmpty)
{
    AicpuTaskCacheEntry entry;
    EXPECT_FALSE(entry.isVOp_);
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    EXPECT_FALSE(entry.isVOp_);
    EXPECT_TRUE(entry.vOpInfo_.sendCounts.empty());
    EXPECT_TRUE(entry.vOpInfo_.recvCounts.empty());
    EXPECT_TRUE(entry.vOpInfo_.sdispls.empty());
    EXPECT_TRUE(entry.vOpInfo_.rdispls.empty());
    EXPECT_EQ(entry.GetEntryBytes(), 4 * sizeof(uint64_t));
}

TEST_F(AicpuTaskCacheEntryTest, InitCacheEntry_VWarmupRequiresNonNullBuffers)
{
    for (uint32_t side = 0; side < 2; ++side) {
        SCOPED_TRACE(side);
        AicpuTaskCacheEntry entry;
        uint64_t baseAddrs[] = {TEST_BASE_ADDR_0, TEST_BASE_ADDR_1};
        uint64_t memSizes[] = {0, 0};
        baseAddrs[side] = 0;
        EXPECT_EQ(entry.InitCacheEntry(baseAddrs, memSizes, 2, VOpInfo{{0}, {0}, {0}, {0}}), HCCL_E_PARA);
    }
}

TEST_F(AicpuTaskCacheEntryTest, InitCacheEntry_InvalidCount)
{
    hcomm::AicpuTaskCacheEntry entry;
    uint64_t baseAddrs[] = {TEST_BASE_ADDR_0};
    uint64_t memSizes[] = {TEST_MEM_SIZE_0};

    EXPECT_EQ(entry.InitCacheEntry(baseAddrs, memSizes, 1), HCCL_E_PARA);
}

TEST_F(AicpuTaskCacheEntryTest, InitCacheEntry_WithRemoteMemory)
{
    hcomm::AicpuTaskCacheEntry entry;
    uint64_t baseAddrs[] = {TEST_BASE_ADDR_0, TEST_BASE_ADDR_1, TEST_REMOTE_BASE_ADDR};
    uint64_t memSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1, TEST_REMOTE_MEM_SIZE};

    EXPECT_EQ(entry.InitCacheEntry(baseAddrs, memSizes, 3), HCCL_SUCCESS);
    ASSERT_EQ(entry.cachedBaseAddrs_.size(), 3U);
    ASSERT_EQ(entry.cachedMemSizes_.size(), 3U);
    EXPECT_EQ(entry.cachedBaseAddrs_[2], TEST_REMOTE_BASE_ADDR);
    EXPECT_EQ(entry.cachedMemSizes_[2], TEST_REMOTE_MEM_SIZE);

    AddrRefreshInfo localInfo;
    EXPECT_EQ(
        entry.UpdateAddrRefreshInfo_(TEST_REMOTE_BASE_ADDR + 0x100, localInfo, AddrType::kAddrTypeLocal), HCCL_SUCCESS);
    EXPECT_FALSE(localInfo.needRefresh);

    AddrRefreshInfo remoteInfo;
    EXPECT_EQ(
        entry.UpdateAddrRefreshInfo_(TEST_REMOTE_BASE_ADDR + 0x100, remoteInfo, AddrType::kAddrTypeRemote),
        HCCL_SUCCESS);
    EXPECT_TRUE(remoteInfo.needRefresh);
    EXPECT_EQ(remoteInfo.memIdx, 2U);
    EXPECT_EQ(remoteInfo.offset, 0x100U);
}

TEST_F(AicpuTaskCacheEntryTest, InitCacheEntry_DoubleInit)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);

    uint64_t baseAddrs[] = {TEST_BASE_ADDR_0, TEST_BASE_ADDR_1};
    uint64_t memSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    EXPECT_EQ(entry.InitCacheEntry(baseAddrs, memSizes, 2), HCCL_E_INTERNAL);
}

// ===================== AddSqeArray Tests =====================

TEST_F(AicpuTaskCacheEntryTest, AddSqeArray_NullRtsqPtr)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);

    EXPECT_EQ(
        entry.AddSqeArray(
            nullptr, aicpuTsThread_.get(), 1, sqeArray.data(), 0, g_defaultTransports.data(), g_defaultLinkTypes.data(),
            g_defaultTransportTypes.data()),
        HCCL_E_PTR);
}

TEST_F(AicpuTaskCacheEntryTest, AddSqeArray_NullAicpuTsThreadPtr)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);

    EXPECT_EQ(
        entry.AddSqeArray(
            rtsqPtr_, nullptr, 1, sqeArray.data(), 0, g_defaultTransports.data(), g_defaultLinkTypes.data(),
            g_defaultTransportTypes.data()),
        HCCL_E_PTR);
}

TEST_F(AicpuTaskCacheEntryTest, AddSqeArray_NullSqeArray)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);

    EXPECT_EQ(
        entry.AddSqeArray(
            rtsqPtr_, aicpuTsThread_.get(), 1, nullptr, 0, g_defaultTransports.data(), g_defaultLinkTypes.data(),
            g_defaultTransportTypes.data()),
        HCCL_E_PTR);
}

TEST_F(AicpuTaskCacheEntryTest, AddSqeArray_ZeroSqeCount)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);

    EXPECT_EQ(
        entry.AddSqeArray(
            rtsqPtr_, aicpuTsThread_.get(), 0, sqeArray.data(), 0, g_defaultTransports.data(),
            g_defaultLinkTypes.data(), g_defaultTransportTypes.data()),
        HCCL_E_INTERNAL);
}

TEST_F(AicpuTaskCacheEntryTest, AddSqeArray_Success)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);

    EXPECT_EQ(
        entry.AddSqeArray(
            rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0, g_defaultTransports.data(),
            g_defaultLinkTypes.data(), g_defaultTransportTypes.data()),
        HCCL_SUCCESS);
    EXPECT_EQ(entry.sqeArrayInfos_.size(), 1U);
    EXPECT_EQ(entry.sqeArrayInfos_[0].sqeCount, 1U);
    EXPECT_NE(entry.sqeArrayInfos_[0].sqeArray, nullptr);
    EXPECT_EQ(entry.launchOrder_.size(), 1U);
    EXPECT_EQ(entry.launchOrder_[0], TaskArrayType::kTaskArrayTypeSqe);
    EXPECT_GT(entry.entryBytes_, 0U);
}

// ===================== AddWqeArray Tests =====================

TEST_F(AicpuTaskCacheEntryTest, AddWqeArray_ReportTaskMismatch)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    std::vector<Hccl::WqeTask> wqeTasks = {MakeWqeTaskRead(0, 0)};

    EXPECT_EQ(
        entry.AddWqeArray(ubConnLite_.get(), ubTransport_.get(), wqeTasks, 0, 0, true, MakeDbSqeProfInfo(false)),
        HCCL_E_INTERNAL);
    EXPECT_EQ(
        entry.AddWqeArray(ubConnLite_.get(), ubTransport_.get(), wqeTasks, 0, 0, false, MakeDbSqeProfInfo(true)),
        HCCL_E_INTERNAL);
}

TEST_F(AicpuTaskCacheEntryTest, AddWqeArray_EmptyWqeTasks)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    std::vector<Hccl::WqeTask> emptyTasks;

    EXPECT_EQ(
        entry.AddWqeArray(ubConnLite_.get(), ubTransport_.get(), emptyTasks, 0, 0, false, MakeDbSqeProfInfo(false)),
        HCCL_E_INTERNAL);
}

TEST_F(AicpuTaskCacheEntryTest, AddWqeArray_NullUbConnLitePtr)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    std::vector<Hccl::WqeTask> wqeTasks = {MakeWqeTaskRead(0, 0)};

    EXPECT_EQ(
        entry.AddWqeArray(nullptr, ubTransport_.get(), wqeTasks, 0, 0, false, MakeDbSqeProfInfo(false)), HCCL_E_PTR);
}

TEST_F(AicpuTaskCacheEntryTest, AddWqeArray_NullUbTransportLiteImplPtr)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    std::vector<Hccl::WqeTask> wqeTasks = {MakeWqeTaskRead(0, 0)};

    EXPECT_EQ(
        entry.AddWqeArray(ubConnLite_.get(), nullptr, wqeTasks, 0, 0, false, MakeDbSqeProfInfo(false)), HCCL_E_PTR);
}

TEST_F(AicpuTaskCacheEntryTest, AddWqeArray_Success)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    std::vector<Hccl::WqeTask> wqeTasks = {MakeWqeTaskRead(0, 0)};

    EXPECT_EQ(
        entry.AddWqeArray(ubConnLite_.get(), ubTransport_.get(), wqeTasks, 0, 0, false, MakeDbSqeProfInfo(false)),
        HCCL_SUCCESS);
    EXPECT_EQ(entry.wqeTaskArrayInfos_.size(), 1U);
    EXPECT_EQ(entry.launchOrder_.size(), 1U);
    EXPECT_EQ(entry.launchOrder_[0], TaskArrayType::kTaskArrayTypeWqe);
    EXPECT_EQ(entry.tokenInfosMap_.size(), 1U);
}

TEST_F(AicpuTaskCacheEntryTest, AddWqeArray_Success_WithReportTask)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    std::vector<Hccl::WqeTask> wqeTasks = {MakeWqeTaskRead(0, 0)};

    EXPECT_EQ(
        entry.AddWqeArray(ubConnLite_.get(), ubTransport_.get(), wqeTasks, 0, 0, true, MakeDbSqeProfInfo(true)),
        HCCL_SUCCESS);
    EXPECT_EQ(entry.wqeTaskArrayInfos_.size(), 1U);
    EXPECT_EQ(entry.streamIdToDbSqeTmpInfoMap_.size(), 1U);
}

// ===================== SubmitCacheEntry Tests =====================

TEST_F(AicpuTaskCacheEntryTest, SubmitCacheEntry_EmptyCache)
{
    hcomm::AicpuTaskCacheEntry entry;
    EXPECT_EQ(entry.SubmitCacheEntry(), HCCL_E_INTERNAL);
}

TEST_F(AicpuTaskCacheEntryTest, SubmitCacheEntry_TmpMapNotEmpty)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    std::vector<Hccl::WqeTask> wqeTasks = {MakeWqeTaskRead(0, 0)};
    ASSERT_EQ(
        entry.AddWqeArray(ubConnLite_.get(), ubTransport_.get(), wqeTasks, 0, 0, true, MakeDbSqeProfInfo(true)),
        HCCL_SUCCESS);

    EXPECT_EQ(entry.SubmitCacheEntry(), HCCL_E_INTERNAL);
}

TEST_F(AicpuTaskCacheEntryTest, SubmitCacheEntry_Success_OnlySqe_NotifyWait)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
    ASSERT_EQ(
        entry.AddSqeArray(
            rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0, g_defaultTransports.data(),
            g_defaultLinkTypes.data(), g_defaultTransportTypes.data()),
        HCCL_SUCCESS);

    EXPECT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);
    EXPECT_EQ(entry.sqeArrayInfos_[0].srcAddrRefreshInfoArray.size(), 1U);
    EXPECT_FALSE(entry.sqeArrayInfos_[0].srcAddrRefreshInfoArray[0].needRefresh);
    EXPECT_FALSE(entry.sqeArrayInfos_[0].dstAddrRefreshInfoArray[0].needRefresh);
}

TEST_F(AicpuTaskCacheEntryTest, SubmitCacheEntry_Success_OnlySqe_NotifyRecord)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_RECORD);
    ASSERT_EQ(
        entry.AddSqeArray(
            rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0, g_defaultTransports.data(),
            g_defaultLinkTypes.data(), g_defaultTransportTypes.data()),
        HCCL_SUCCESS);

    EXPECT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);
}

TEST_F(AicpuTaskCacheEntryTest, SubmitCacheEntry_Success_OnlySqe_Ubdma)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_UBDMA);
    ASSERT_EQ(
        entry.AddSqeArray(
            rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0, g_defaultTransports.data(),
            g_defaultLinkTypes.data(), g_defaultTransportTypes.data()),
        HCCL_SUCCESS);

    EXPECT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);
    EXPECT_FALSE(entry.sqeArrayInfos_[0].srcAddrRefreshInfoArray[0].needRefresh);
}

TEST_F(AicpuTaskCacheEntryTest, VAddrRankIdMatchesByteRanges)
{
    AicpuTaskCacheEntry entry;
    // rank 0 的 count 为零，不产生 SQE；这里仅匹配有数据的 rank 1/2，displs 不要求按 rank 递增。
    entry.vOpInfo_ = VOpInfo{{0, 16, 8}, {0, 8, 16}, {32, 32, 8}, {64, 64, 16}};
    AddrRefreshInfo input(0);
    input.offset = 32;
    ASSERT_EQ(entry.UpdateVAddrRankId_(input), HCCL_SUCCESS);
    EXPECT_EQ(input.rankId, 1U);
    AddrRefreshInfo output(1);
    output.offset = 16;
    ASSERT_EQ(entry.UpdateVAddrRankId_(output), HCCL_SUCCESS);
    EXPECT_EQ(output.rankId, 2U);
    AddrRefreshInfo ccl;
    EXPECT_EQ(entry.UpdateVAddrRankId_(ccl), HCCL_SUCCESS);
}

TEST_F(AicpuTaskCacheEntryTest, VAddrRankIdRejectsInvalidRanges)
{
    struct TestCase {
        std::vector<uint64_t> counts;
        std::vector<uint64_t> displs;
        size_t offset;
    };
    const TestCase cases[] = {
        {{}, {}, 0},                // 空数组
        {{8, 8}, {0}, 0},           // 长度不一致
        {{8}, {16}, 24},            // 右端点不属于半开区间
        {{8, 8}, {0, 0}, 0},        // 多个 rank 匹配
        {{16, 8}, {0, 8}, 8},       // 部分重叠
        {{8}, {16}, 17},            // 不符合无分片约定
        {{8}, {UINT64_MAX - 3}, 0}, // 区间上界溢出
    };
    for (size_t idx = 0; idx < sizeof(cases) / sizeof(cases[0]); ++idx) {
        SCOPED_TRACE(idx);
        AicpuTaskCacheEntry entry;
        entry.vOpInfo_.sendCounts = cases[idx].counts;
        entry.vOpInfo_.sdispls = cases[idx].displs;
        AddrRefreshInfo info(0);
        info.offset = cases[idx].offset;
        EXPECT_EQ(entry.UpdateVAddrRankId_(info), HCCL_E_PARA);
    }
    AicpuTaskCacheEntry entry;
    AddrRefreshInfo remote(2);
    EXPECT_EQ(entry.UpdateVAddrRankId_(remote), HCCL_E_PARA);
}

TEST_F(AicpuTaskCacheEntryTest, SubmitCacheEntry_VSelfCopyRecordsBothRankIds)
{
    AicpuTaskCacheEntry entry;
    uint64_t baseAddrs[] = {TEST_BASE_ADDR_0, TEST_BASE_ADDR_1};
    uint64_t memSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    ASSERT_EQ(entry.InitCacheEntry(baseAddrs, memSizes, 2, VOpInfo{{0, 16}, {0, 16}, {0, 32}, {0, 64}}), HCCL_SUCCESS);
    auto sqe
        = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA, TEST_BASE_ADDR_0 + 32, TEST_BASE_ADDR_1 + 64, 16);
    ASSERT_EQ(
        entry.AddSqeArray(
            rtsqPtr_, aicpuTsThread_.get(), 1, sqe.data(), 0, g_defaultTransports.data(), g_defaultLinkTypes.data(),
            g_defaultTransportTypes.data()),
        HCCL_SUCCESS);
    ASSERT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);
    EXPECT_EQ(entry.sqeArrayInfos_[0].srcAddrRefreshInfoArray[0].rankId, 1U);
    EXPECT_EQ(entry.sqeArrayInfos_[0].dstAddrRefreshInfoArray[0].rankId, 1U);
}

TEST_F(AicpuTaskCacheEntryTest, SubmitCacheEntry_VRankMismatchReturnsError)
{
    AicpuTaskCacheEntry entry;
    uint64_t baseAddrs[] = {TEST_BASE_ADDR_0, TEST_BASE_ADDR_1};
    uint64_t memSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    ASSERT_EQ(entry.InitCacheEntry(baseAddrs, memSizes, 2, VOpInfo{{8}, {8}, {0}, {0}}), HCCL_SUCCESS);
    auto sqe = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA, TEST_BASE_ADDR_0 + 8, 0x900000, 8);
    ASSERT_EQ(
        entry.AddSqeArray(
            rtsqPtr_, aicpuTsThread_.get(), 1, sqe.data(), 0, g_defaultTransports.data(), g_defaultLinkTypes.data(),
            g_defaultTransportTypes.data()),
        HCCL_SUCCESS);
    EXPECT_EQ(entry.SubmitCacheEntry(), HCCL_E_PARA);
}

TEST_F(AicpuTaskCacheEntryTest, SubmitCacheEntry_Success_OnlySqe_Sdma_AddrInRange)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    uint64_t srcAddr = TEST_BASE_ADDR_0 + 0x100;
    uint64_t dstAddr = TEST_BASE_ADDR_1 + 0x200;
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA, srcAddr, dstAddr);
    ASSERT_EQ(
        entry.AddSqeArray(
            rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0, g_defaultTransports.data(),
            g_defaultLinkTypes.data(), g_defaultTransportTypes.data()),
        HCCL_SUCCESS);

    EXPECT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);
    EXPECT_TRUE(entry.sqeArrayInfos_[0].srcAddrRefreshInfoArray[0].needRefresh);
    EXPECT_EQ(entry.sqeArrayInfos_[0].srcAddrRefreshInfoArray[0].memIdx, 0U);
    EXPECT_TRUE(entry.sqeArrayInfos_[0].dstAddrRefreshInfoArray[0].needRefresh);
    EXPECT_EQ(entry.sqeArrayInfos_[0].dstAddrRefreshInfoArray[0].memIdx, 1U);
}

TEST_F(AicpuTaskCacheEntryTest, SubmitCacheEntry_Success_OnlySqe_Sdma_AddrNotInRange)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    uint64_t outOfRangeAddr = 0x999000;
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA, outOfRangeAddr, outOfRangeAddr);
    ASSERT_EQ(
        entry.AddSqeArray(
            rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0, g_defaultTransports.data(),
            g_defaultLinkTypes.data(), g_defaultTransportTypes.data()),
        HCCL_SUCCESS);

    EXPECT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);
    EXPECT_FALSE(entry.sqeArrayInfos_[0].srcAddrRefreshInfoArray[0].needRefresh);
    EXPECT_FALSE(entry.sqeArrayInfos_[0].dstAddrRefreshInfoArray[0].needRefresh);
}

TEST_F(AicpuTaskCacheEntryTest, SubmitCacheEntry_Success_OnlySqe_WriteValue)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    uint64_t dstAddr = TEST_BASE_ADDR_0 + 0x50;
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_WRITE_VALUE, 0, dstAddr);
    ASSERT_EQ(
        entry.AddSqeArray(
            rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0, g_defaultTransports.data(),
            g_defaultLinkTypes.data(), g_defaultTransportTypes.data()),
        HCCL_SUCCESS);

    EXPECT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);
    EXPECT_FALSE(entry.sqeArrayInfos_[0].srcAddrRefreshInfoArray[0].needRefresh);
    EXPECT_TRUE(entry.sqeArrayInfos_[0].dstAddrRefreshInfoArray[0].needRefresh);
    EXPECT_EQ(entry.sqeArrayInfos_[0].dstAddrRefreshInfoArray[0].memIdx, 0U);
}

TEST_F(AicpuTaskCacheEntryTest, SubmitCacheEntry_InvalidSqeType)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_AIC);
    ASSERT_EQ(
        entry.AddSqeArray(
            rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0, g_defaultTransports.data(),
            g_defaultLinkTypes.data(), g_defaultTransportTypes.data()),
        HCCL_SUCCESS);

    EXPECT_EQ(entry.SubmitCacheEntry(), HCCL_E_INTERNAL);
}

TEST_F(AicpuTaskCacheEntryTest, SubmitCacheEntry_VWqeRankComesFromChannel)
{
    AicpuTaskCacheManager::RegisterGetChannelRemoteRankIdCallback(GetWqeRemoteRankForTest);
    AicpuTaskCacheEntry entry;
    uint64_t baseAddrs[] = {TEST_BASE_ADDR_0, TEST_BASE_ADDR_1};
    uint64_t memSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    // 本端地址可匹配 rank 0，但 WQE 必须使用通道返回的 rank 1。
    ASSERT_EQ(
        entry.InitCacheEntry(baseAddrs, memSizes, 2, VOpInfo{{16, 16}, {16, 16}, {0, 32}, {0, 32}}), HCCL_SUCCESS);
    constexpr uint64_t REMOTE_CCL_ADDR = 0x900000;
    std::vector<WqeTask> tasks = {
        MakeWqeTaskRead(TEST_BASE_ADDR_1, REMOTE_CCL_ADDR),
        MakeWqeTaskWrite(TEST_BASE_ADDR_0, REMOTE_CCL_ADDR, false),
        MakeWqeTaskWriteWithNotify(TEST_BASE_ADDR_0, REMOTE_CCL_ADDR),
        MakeWqeTaskWrite(0, REMOTE_CCL_ADDR, true),
    };
    tasks[0].wqeWrite.u.sge.length = 16;
    tasks[1].wqeWrite.u.sge.length = 16;
    tasks[2].wqeWriteWithNotify.localU.sge.length = 16;
    ASSERT_EQ(
        entry.AddWqeArray(ubConnLite_.get(), ubTransport_.get(), tasks, 0, 0, false, MakeDbSqeProfInfo(false)),
        HCCL_SUCCESS);
    AddUbdmaSqeToClearTmpMap(entry);
    ASSERT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);
    EXPECT_EQ(queriedWqeChannel, reinterpret_cast<ChannelHandle>(ubTransport_.get()));
    EXPECT_EQ(wqeRankQueryCount, 1U);
    for (size_t idx = 0; idx < tasks.size(); ++idx) {
        EXPECT_EQ(entry.wqeTaskArrayInfos_[0].locAddrRefreshInfoArray[idx].rankId, 1U);
        EXPECT_EQ(entry.wqeTaskArrayInfos_[0].rmtAddrRefreshInfoArray[idx].rankId, 1U);
        EXPECT_EQ(std::memcmp(&entry.wqeTaskArrayInfos_[0].wqeTaskArray[idx], &tasks[idx], sizeof(WqeTask)), 0);
    }
}

TEST_F(AicpuTaskCacheEntryTest, SubmitCacheEntry_VWqeRankLookupErrors)
{
    using Callback = AicpuTaskCacheManager::GetChannelRemoteRankIdCallback;
    const Callback callbacks[] = {nullptr, GetWqeRemoteRankFailureForTest, GetWqeRemoteRankForTest};
    const HcclResult expected[] = {HCCL_E_INTERNAL, HCCL_E_PARA, HCCL_E_PARA};
    for (size_t idx = 0; idx < sizeof(callbacks) / sizeof(callbacks[0]); ++idx) {
        SCOPED_TRACE(idx);
        AicpuTaskCacheManager::RegisterGetChannelRemoteRankIdCallback(callbacks[idx]);
        AicpuTaskCacheEntry entry;
        uint64_t baseAddrs[] = {TEST_BASE_ADDR_0, TEST_BASE_ADDR_1};
        uint64_t memSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
        // 只有 rank 0，回调返回 rank 1 时必须报越界。
        ASSERT_EQ(entry.InitCacheEntry(baseAddrs, memSizes, 2, VOpInfo{{16}, {16}, {0}, {0}}), HCCL_SUCCESS);
        auto task = MakeWqeTaskRead(TEST_BASE_ADDR_1, 0x900000);
        task.wqeWrite.u.sge.length = 16;
        ASSERT_EQ(
            entry.AddWqeArray(ubConnLite_.get(), ubTransport_.get(), {task}, 0, 0, false, MakeDbSqeProfInfo(false)),
            HCCL_SUCCESS);
        AddUbdmaSqeToClearTmpMap(entry);
        EXPECT_EQ(entry.SubmitCacheEntry(), expected[idx]);
    }
}

TEST_F(AicpuTaskCacheEntryTest, RefreshWqeTasks_VZeroAndRestoreLengthsAndAddresses)
{
    AicpuTaskCacheManager::RegisterGetChannelRemoteRankIdCallback(GetWqeRemoteRankForTest);
    AicpuTaskCacheEntry entry;
    uint64_t oldAddrs[] = {TEST_BASE_ADDR_0, TEST_BASE_ADDR_1};
    uint64_t oldSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    ASSERT_EQ(entry.InitCacheEntry(oldAddrs, oldSizes, 2, VOpInfo{{16, 16}, {16, 16}, {0, 32}, {0, 64}}), HCCL_SUCCESS);
    constexpr uint64_t REMOTE_CCL_ADDR = 0x900000;
    auto read = MakeWqeTaskRead(TEST_BASE_ADDR_1 + 64, REMOTE_CCL_ADDR);
    auto write = MakeWqeTaskWrite(TEST_BASE_ADDR_0 + 32, REMOTE_CCL_ADDR, false);
    auto notify = MakeWqeTaskWriteWithNotify(TEST_BASE_ADDR_0 + 32, REMOTE_CCL_ADDR);
    auto inlineNotify = MakeWqeTaskWrite(0, REMOTE_CCL_ADDR, true);
    read.wqeWrite.u.sge.length = 16;
    write.wqeWrite.u.sge.length = 16;
    notify.wqeWriteWithNotify.localU.sge.length = 16;
    read.wqeWrite.u.sge.tokenId = 0xAB;
    read.wqeWrite.comm.rmtObjId = 0xBC;
    read.wqeWrite.comm.rmtTokenValue = 0xCD;
    write.wqeWrite.u.sge.tokenId = 0xDE;
    write.wqeWrite.comm.rmtObjId = 0xEF;
    write.wqeWrite.comm.rmtTokenValue = 0x12;
    notify.wqeWriteWithNotify.localU.sge.tokenId = 0x34;
    notify.wqeWriteWithNotify.comm.rmtObjId = 0x56;
    notify.wqeWriteWithNotify.comm.rmtTokenValue = 0x78;
    read.wqeWrite.comm.sgeNum = 1;
    write.wqeWrite.comm.sgeNum = 1;
    notify.wqeWriteWithNotify.comm.sgeNum = 1;
    ASSERT_EQ(
        entry.AddWqeArray(
            ubConnLite_.get(), ubTransport_.get(), {read, write, notify, inlineNotify}, 0, 0, false,
            MakeDbSqeProfInfo(false)),
        HCCL_SUCCESS);
    AddUbdmaSqeToClearTmpMap(entry);
    ASSERT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);

    uint64_t newAddrs[] = {0x50000, 0x60000};
    uint64_t newSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    auto& arrayInfo = entry.wqeTaskArrayInfos_[0];
    const auto expectRecordedTokens = [&arrayInfo]() {
        const auto& readWqe = arrayInfo.wqeTaskArray[0].wqeWrite;
        EXPECT_EQ(readWqe.u.sge.tokenId, 0xABU);
        EXPECT_EQ(readWqe.comm.rmtObjId, 0xBCU);
        EXPECT_EQ(readWqe.comm.rmtTokenValue, 0xCDU);
        const auto& writeWqe = arrayInfo.wqeTaskArray[1].wqeWrite;
        EXPECT_EQ(writeWqe.u.sge.tokenId, 0xDEU);
        EXPECT_EQ(writeWqe.comm.rmtObjId, 0xEFU);
        EXPECT_EQ(writeWqe.comm.rmtTokenValue, 0x12U);
        const auto& notifyWqe = arrayInfo.wqeTaskArray[2].wqeWriteWithNotify;
        EXPECT_EQ(notifyWqe.localU.sge.tokenId, 0x34U);
        EXPECT_EQ(notifyWqe.comm.rmtObjId, 0x56U);
        EXPECT_EQ(notifyWqe.comm.rmtTokenValue, 0x78U);
    };
    const VOpInfo zeroCounts{{0, 0}, {0, 0}, {0, 96}, {0, 128}};
    uint64_t emptyAddrs[] = {0, 0};
    uint64_t emptySizes[] = {96, 128};
    // Validate the null sides, then refresh tasks without submitting them to hardware queues.
    entry.connOverflowInfos_.clear();
    entry.launchOrder_.clear();
    ASSERT_EQ(entry.RefreshAndLaunch(emptyAddrs, emptySizes, 2, zeroCounts), HCCL_SUCCESS);
    ASSERT_EQ(entry.RefreshWqeTasks_(arrayInfo, emptyAddrs, emptySizes, 2, zeroCounts), HCCL_SUCCESS);
    expectRecordedTokens();
    EXPECT_EQ(
        static_cast<uint32_t>(arrayInfo.wqeTaskArray[0].wqeWrite.comm.opcode),
        static_cast<uint32_t>(UdmaSqOpcode::UDMA_OPC_NOP));
    EXPECT_EQ(
        static_cast<uint32_t>(arrayInfo.wqeTaskArray[1].wqeWrite.comm.opcode),
        static_cast<uint32_t>(UdmaSqOpcode::UDMA_OPC_NOP));
    EXPECT_EQ(
        static_cast<uint32_t>(arrayInfo.wqeTaskArray[2].wqeWriteWithNotify.comm.opcode), WRITE_WITH_NOTIFY_OPCODE);
    EXPECT_EQ(arrayInfo.wqeTaskArray[2].wqeWriteWithNotify.localU.sge.length, 0U);
    EXPECT_EQ(static_cast<uint32_t>(arrayInfo.wqeTaskArray[2].wqeWriteWithNotify.comm.sgeNum), 0U);
    EXPECT_EQ(
        static_cast<uint32_t>(arrayInfo.wqeTaskArray[3].wqeWrite.comm.opcode),
        static_cast<uint32_t>(UdmaSqOpcode::UDMA_OPC_WRITE));

    const VOpInfo newCounts{{16, 8}, {16, 12}, {0, 96}, {0, 128}};
    ASSERT_EQ(entry.RefreshWqeTasks_(arrayInfo, newAddrs, newSizes, 2, newCounts), HCCL_SUCCESS);
    expectRecordedTokens();
    EXPECT_EQ(
        static_cast<uint32_t>(arrayInfo.wqeTaskArray[0].wqeWrite.comm.opcode),
        static_cast<uint32_t>(UdmaSqOpcode::UDMA_OPC_READ));
    EXPECT_EQ(arrayInfo.wqeTaskArray[0].wqeWrite.u.sge.length, 12U);
    EXPECT_EQ(
        static_cast<uint32_t>(arrayInfo.wqeTaskArray[1].wqeWrite.comm.opcode),
        static_cast<uint32_t>(UdmaSqOpcode::UDMA_OPC_WRITE));
    EXPECT_EQ(arrayInfo.wqeTaskArray[1].wqeWrite.u.sge.length, 8U);
    EXPECT_EQ(arrayInfo.wqeTaskArray[2].wqeWriteWithNotify.localU.sge.length, 8U);
    EXPECT_EQ(static_cast<uint32_t>(arrayInfo.wqeTaskArray[2].wqeWriteWithNotify.comm.sgeNum), 1U);
    uint64_t readAddr = 0;
    AicpuTaskCacheEntry::CombineUint32ToUint64(
        readAddr, arrayInfo.wqeTaskArray[0].wqeWrite.u.sge.dataAddrHigh,
        arrayInfo.wqeTaskArray[0].wqeWrite.u.sge.dataAddrLow);
    EXPECT_EQ(readAddr, newAddrs[1] + 128);
    uint64_t writeAddr = 0;
    AicpuTaskCacheEntry::CombineUint32ToUint64(
        writeAddr, arrayInfo.wqeTaskArray[1].wqeWrite.u.sge.dataAddrHigh,
        arrayInfo.wqeTaskArray[1].wqeWrite.u.sge.dataAddrLow);
    EXPECT_EQ(writeAddr, newAddrs[0] + 96);
    // 不访问硬件队列，仅验证 V 类命中入口的参数校验可通过。
    entry.connOverflowInfos_.clear();
    entry.launchOrder_.clear();
    EXPECT_EQ(entry.RefreshAndLaunch(newAddrs, newSizes, 2, newCounts), HCCL_SUCCESS);

    // A valid 8 GiB buffer can contain 5 GiB for one rank, but a single WQE cannot encode that length.
    uint64_t largeSizes[] = {uint64_t{8} << 30, uint64_t{8} << 30};
    const VOpInfo oversizedCounts{{16, uint64_t{5} << 30}, {16, 12}, {0, 96}, {0, 128}};
    EXPECT_EQ(entry.RefreshAndLaunch(newAddrs, largeSizes, 2, oversizedCounts), HCCL_E_PARA);
}

TEST_F(AicpuTaskCacheEntryTest, SubmitCacheEntry_Success_OnlyWqe_Read)
{
    AicpuTaskCacheManager::RegisterGetChannelRemoteRankIdCallback(GetWqeRemoteRankForTest);
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    uint64_t locAddr = TEST_BASE_ADDR_0 + 0x10;
    uint64_t rmtAddr = TEST_BASE_ADDR_1 + 0x20;
    std::vector<Hccl::WqeTask> wqeTasks = {MakeWqeTaskRead(locAddr, rmtAddr)};
    ASSERT_EQ(
        entry.AddWqeArray(ubConnLite_.get(), ubTransport_.get(), wqeTasks, 0, 0, false, MakeDbSqeProfInfo(false)),
        HCCL_SUCCESS);
    AddUbdmaSqeToClearTmpMap(entry);

    EXPECT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);
    EXPECT_TRUE(entry.wqeTaskArrayInfos_[0].locAddrRefreshInfoArray[0].needRefresh);
    EXPECT_EQ(entry.wqeTaskArrayInfos_[0].locAddrRefreshInfoArray[0].memIdx, 0U);

    EXPECT_EQ(wqeRankQueryCount, 0U); // 非 V 类不查询通道 rank。

    // 注意: WQE remote addr不再检查本地地址, 因此needRefresh预期为false, memIdx预期为默认值0U
    EXPECT_FALSE(entry.wqeTaskArrayInfos_[0].rmtAddrRefreshInfoArray[0].needRefresh);
    EXPECT_EQ(entry.wqeTaskArrayInfos_[0].rmtAddrRefreshInfoArray[0].memIdx, 0U);
}

TEST_F(AicpuTaskCacheEntryTest, SubmitCacheEntry_Success_OnlyWqe_WriteNonInline)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    uint64_t locAddr = TEST_BASE_ADDR_0 + 0x10;
    uint64_t rmtAddr = TEST_BASE_ADDR_1 + 0x20;
    std::vector<Hccl::WqeTask> wqeTasks = {MakeWqeTaskWrite(locAddr, rmtAddr, false)};
    ASSERT_EQ(
        entry.AddWqeArray(ubConnLite_.get(), ubTransport_.get(), wqeTasks, 0, 0, false, MakeDbSqeProfInfo(false)),
        HCCL_SUCCESS);
    AddUbdmaSqeToClearTmpMap(entry);

    EXPECT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);
    EXPECT_TRUE(entry.wqeTaskArrayInfos_[0].locAddrRefreshInfoArray[0].needRefresh);

    // 注意: WQE remote addr不再检查本地地址, 因此needRefresh预期为false
    EXPECT_FALSE(entry.wqeTaskArrayInfos_[0].rmtAddrRefreshInfoArray[0].needRefresh);
}

TEST_F(AicpuTaskCacheEntryTest, SubmitCacheEntry_Success_OnlyWqe_WriteInline)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    uint64_t rmtAddr = TEST_BASE_ADDR_1 + 0x20;
    std::vector<Hccl::WqeTask> wqeTasks = {MakeWqeTaskWrite(0, rmtAddr, true)};
    ASSERT_EQ(
        entry.AddWqeArray(ubConnLite_.get(), ubTransport_.get(), wqeTasks, 0, 0, false, MakeDbSqeProfInfo(false)),
        HCCL_SUCCESS);
    AddUbdmaSqeToClearTmpMap(entry);

    EXPECT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);
    EXPECT_FALSE(entry.wqeTaskArrayInfos_[0].locAddrRefreshInfoArray[0].needRefresh);

    // 注意: WQE remote addr不再检查本地地址, 因此needRefresh预期为false
    EXPECT_FALSE(entry.wqeTaskArrayInfos_[0].rmtAddrRefreshInfoArray[0].needRefresh);
}

TEST_F(AicpuTaskCacheEntryTest, SubmitCacheEntry_Success_OnlyWqe_WriteWithNotify)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    uint64_t locAddr = TEST_BASE_ADDR_0 + 0x10;
    uint64_t rmtAddr = TEST_BASE_ADDR_1 + 0x20;
    std::vector<Hccl::WqeTask> wqeTasks = {MakeWqeTaskWriteWithNotify(locAddr, rmtAddr)};
    ASSERT_EQ(
        entry.AddWqeArray(ubConnLite_.get(), ubTransport_.get(), wqeTasks, 0, 0, false, MakeDbSqeProfInfo(false)),
        HCCL_SUCCESS);
    AddUbdmaSqeToClearTmpMap(entry);

    EXPECT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);
    EXPECT_TRUE(entry.wqeTaskArrayInfos_[0].locAddrRefreshInfoArray[0].needRefresh);

    // 注意: WQE remote addr不再检查本地地址, 因此needRefresh预期为false
    EXPECT_FALSE(entry.wqeTaskArrayInfos_[0].rmtAddrRefreshInfoArray[0].needRefresh);
}

TEST_F(AicpuTaskCacheEntryTest, SubmitCacheEntry_InvalidWqeType)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    std::vector<Hccl::WqeTask> wqeTasks = {MakeWqeTaskInvalid()};
    ASSERT_EQ(
        entry.AddWqeArray(ubConnLite_.get(), ubTransport_.get(), wqeTasks, 0, 0, false, MakeDbSqeProfInfo(false)),
        HCCL_SUCCESS);
    AddUbdmaSqeToClearTmpMap(entry);

    EXPECT_EQ(entry.SubmitCacheEntry(), HCCL_E_INTERNAL);
}

TEST_F(AicpuTaskCacheEntryTest, SubmitCacheEntry_Success_BothSqeWqe)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);

    std::vector<Hccl::WqeTask> wqeTasks = {MakeWqeTaskRead(TEST_BASE_ADDR_0 + 0x10, TEST_BASE_ADDR_1 + 0x20)};
    ASSERT_EQ(
        entry.AddWqeArray(ubConnLite_.get(), ubTransport_.get(), wqeTasks, 0, 0, false, MakeDbSqeProfInfo(false)),
        HCCL_SUCCESS);

    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_UBDMA);
    ASSERT_EQ(
        entry.AddSqeArray(
            rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0, g_defaultTransports.data(),
            g_defaultLinkTypes.data(), g_defaultTransportTypes.data()),
        HCCL_SUCCESS);

    EXPECT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);
    EXPECT_EQ(entry.launchOrder_.size(), 2U);
    EXPECT_EQ(entry.launchOrder_[0], TaskArrayType::kTaskArrayTypeWqe);
    EXPECT_EQ(entry.launchOrder_[1], TaskArrayType::kTaskArrayTypeSqe);
}

TEST_F(AicpuTaskCacheEntryTest, SubmitCacheEntry_Success_TokenInfoFlagSet)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    uint64_t locAddr = TEST_BASE_ADDR_0 + 0x10;
    uint64_t rmtAddr = TEST_BASE_ADDR_1 + 0x20;
    std::vector<Hccl::WqeTask> wqeTasks = {MakeWqeTaskRead(locAddr, rmtAddr)};
    ASSERT_EQ(
        entry.AddWqeArray(ubConnLite_.get(), ubTransport_.get(), wqeTasks, 0, 0, false, MakeDbSqeProfInfo(false)),
        HCCL_SUCCESS);
    AddUbdmaSqeToClearTmpMap(entry);
    ASSERT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);

    auto& tokenInfos = entry.tokenInfosMap_.begin()->second;
    EXPECT_TRUE(tokenInfos[0].needLocTokenIdFlag);

    // 注意: WQE remote addr不再检查本地地址, 因此needRmtTokenIdAndValueFlag预期为false
    EXPECT_FALSE(tokenInfos[1].needRmtTokenIdAndValueFlag);
}

// ===================== RefreshAndLaunch Tests =====================

TEST_F(AicpuTaskCacheEntryTest, RefreshAndLaunch_RejectsInvalidVwqeMetadataBeforeLaunchingSqe)
{
    AicpuTaskCacheEntry entry;
    uint64_t oldAddrs[] = {TEST_BASE_ADDR_0, TEST_BASE_ADDR_1};
    uint64_t memSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    ASSERT_EQ(entry.InitCacheEntry(oldAddrs, memSizes, 2, VOpInfo{{16, 16}, {16, 16}, {0, 32}, {0, 32}}), HCCL_SUCCESS);

    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA, TEST_BASE_ADDR_0, TEST_BASE_ADDR_1);
    ASSERT_EQ(
        entry.AddSqeArray(
            rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0, g_defaultTransports.data(),
            g_defaultLinkTypes.data(), g_defaultTransportTypes.data()),
        HCCL_SUCCESS);
    const std::vector<uint8_t> originalSqe(
        entry.sqeArrayInfos_[0].sqeArray, entry.sqeArrayInfos_[0].sqeArray + Hccl::AC_SQE_SIZE);

    WqeTaskArrayInfo wqeArray;
    wqeArray.wqeTaskArray
        = {MakeWqeTaskWrite(TEST_BASE_ADDR_0 + 32, 0x900000, false),
           MakeWqeTaskWrite(TEST_BASE_ADDR_0 + 32, 0x900000, false)};
    wqeArray.wqeTaskArray[0].wqeWrite.u.sge.length = 16;
    wqeArray.wqeTaskArray[1].wqeWrite.u.sge.length = 16;
    wqeArray.originalOpcodes
        = {static_cast<uint8_t>(UdmaSqOpcode::UDMA_OPC_WRITE), static_cast<uint8_t>(UdmaSqOpcode::UDMA_OPC_WRITE)};
    wqeArray.originalSgeNums = {1, 1};
    wqeArray.locAddrRefreshInfoArray.resize(2);
    wqeArray.rmtAddrRefreshInfoArray.resize(2);
    wqeArray.locAddrRefreshInfoArray[0].rankId = 1;
    wqeArray.locAddrRefreshInfoArray[1].rankId = 1;
    entry.wqeTaskArrayInfos_.push_back(wqeArray);
    entry.launchOrder_.push_back(TaskArrayType::kTaskArrayTypeWqe);

    uint64_t newAddrs[] = {0x50000, 0x60000};
    const VOpInfo newCounts{{16, 16}, {16, 16}, {0, 32}, {0, 32}};
    for (uint32_t field = 0; field < 4; ++field) {
        SCOPED_TRACE(field);
        auto& invalidArray = entry.wqeTaskArrayInfos_[0];
        invalidArray = wqeArray;
        switch (field) {
            case 0:
                invalidArray.originalOpcodes.pop_back();
                break;
            case 1:
                invalidArray.originalSgeNums.pop_back();
                break;
            case 2:
                invalidArray.locAddrRefreshInfoArray.pop_back();
                break;
            case 3:
                invalidArray.rmtAddrRefreshInfoArray.pop_back();
                break;
        }
        EXPECT_EQ(entry.RefreshAndLaunch(newAddrs, memSizes, 2, newCounts), HCCL_E_INTERNAL);
        EXPECT_EQ(std::memcmp(entry.sqeArrayInfos_[0].sqeArray, originalSqe.data(), Hccl::AC_SQE_SIZE), 0);
    }
}

TEST_F(AicpuTaskCacheEntryTest, RefreshAndLaunch_RejectsUnequalVArraySizesBeforeRefreshingTasks)
{
    AicpuTaskCacheEntry entry;
    uint64_t oldAddrs[] = {TEST_BASE_ADDR_0, TEST_BASE_ADDR_1};
    uint64_t memSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    const VOpInfo warmup{{16, 16}, {16, 16}, {0, 32}, {0, 32}};
    ASSERT_EQ(entry.InitCacheEntry(oldAddrs, memSizes, 2, warmup), HCCL_SUCCESS);
    auto sqe = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA, oldAddrs[0], oldAddrs[1], 16);
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
    uint64_t newAddrs[] = {0x50000, 0x60000};

    // C API 解析结果始终等长，直接调用 Entry 覆盖各数组独立长度异常。
    for (uint32_t field = 0; field < 4; ++field) {
        for (size_t arraySize : {0U, 1U, 3U}) {
            SCOPED_TRACE(field);
            SCOPED_TRACE(arraySize);
            VOpInfo current = warmup;
            std::vector<uint64_t>* arrays[]
                = {&current.sendCounts, &current.recvCounts, &current.sdispls, &current.rdispls};
            arrays[field]->resize(arraySize);
            EXPECT_EQ(entry.RefreshAndLaunch(newAddrs, memSizes, 2, current), HCCL_E_PARA);
            EXPECT_EQ(std::memcmp(entry.sqeArrayInfos_[0].sqeArray, originalSqe.data(), Hccl::AC_SQE_SIZE), 0);
            EXPECT_EQ(rtsqPtr_->GetTail(), originalTail);
            EXPECT_EQ(rtsqPtr_->GetPendingSqeCnt(), originalPending);
            EXPECT_EQ(entry.vOpInfo_.sendCounts, warmup.sendCounts);
            EXPECT_EQ(entry.vOpInfo_.recvCounts, warmup.recvCounts);
            EXPECT_EQ(entry.vOpInfo_.sdispls, warmup.sdispls);
            EXPECT_EQ(entry.vOpInfo_.rdispls, warmup.rdispls);
        }
    }
}

TEST_F(AicpuTaskCacheEntryTest, RefreshAndLaunch_CountMismatch)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
    ASSERT_EQ(
        entry.AddSqeArray(
            rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0, g_defaultTransports.data(),
            g_defaultLinkTypes.data(), g_defaultTransportTypes.data()),
        HCCL_SUCCESS);
    ASSERT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);

    uint64_t baseAddrs[] = {TEST_BASE_ADDR_0};
    uint64_t memSizes[] = {TEST_MEM_SIZE_0};
    EXPECT_EQ(entry.RefreshAndLaunch(baseAddrs, memSizes, 1), HCCL_E_INTERNAL);
}

TEST_F(AicpuTaskCacheEntryTest, RefreshAndLaunch_RejectsNullBufferWithNonzeroVCount)
{
    for (uint32_t side = 0; side < 2; ++side) {
        SCOPED_TRACE(side);
        AicpuTaskCacheEntry entry;
        uint64_t baseAddrs[] = {TEST_BASE_ADDR_0, TEST_BASE_ADDR_1};
        uint64_t memSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
        ASSERT_EQ(entry.InitCacheEntry(baseAddrs, memSizes, 2, VOpInfo{{16}, {16}, {0}, {0}}), HCCL_SUCCESS);
        baseAddrs[side] = 0;
        EXPECT_EQ(entry.RefreshAndLaunch(baseAddrs, memSizes, 2, entry.vOpInfo_), HCCL_E_PARA);
    }
}

TEST_F(AicpuTaskCacheEntryTest, RefreshAndLaunch_NonVRejectsNullBufferWithZeroSize)
{
    AicpuTaskCacheEntry entry;
    uint64_t baseAddrs[] = {TEST_BASE_ADDR_0, TEST_BASE_ADDR_1};
    uint64_t memSizes[] = {0, TEST_MEM_SIZE_1};
    ASSERT_EQ(entry.InitCacheEntry(baseAddrs, memSizes, 2), HCCL_SUCCESS);
    baseAddrs[0] = 0;
    EXPECT_EQ(entry.RefreshAndLaunch(baseAddrs, memSizes, 2), HCCL_E_PARA);
}

TEST_F(AicpuTaskCacheEntryTest, RefreshAndLaunch_Success_SqeAndWqe)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);

    // 添加WQE数组 (Read类型, loc/rmt地址均在cached range内)
    uint64_t locAddr = TEST_BASE_ADDR_0 + 0x10;
    uint64_t rmtAddr = TEST_BASE_ADDR_1 + 0x20;
    std::vector<Hccl::WqeTask> wqeTasks = {MakeWqeTaskRead(locAddr, rmtAddr)};
    wqeTasks[0].wqeWrite.u.sge.tokenId = 17;
    wqeTasks[0].wqeWrite.comm.rmtObjId = 23;
    wqeTasks[0].wqeWrite.comm.rmtTokenValue = 29;
    ASSERT_EQ(
        entry.AddWqeArray(ubConnLite_.get(), ubTransport_.get(), wqeTasks, 0, 0, false, MakeDbSqeProfInfo(false)),
        HCCL_SUCCESS);

    // 添加SQE数组 (SDMA类型, src/dst地址均在cached range内)
    uint64_t srcAddr = TEST_BASE_ADDR_0 + 0x100;
    uint64_t dstAddr = TEST_BASE_ADDR_1 + 0x200;
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA, srcAddr, dstAddr);
    ASSERT_EQ(
        entry.AddSqeArray(
            rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0, g_defaultTransports.data(),
            g_defaultLinkTypes.data(), g_defaultTransportTypes.data()),
        HCCL_SUCCESS);

    ASSERT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);
    auto& tokenInfos = entry.tokenInfosMap_.begin()->second;
    tokenInfos[0].locTokenId = 99; // 模拟按新地址获取的 token。

    // 验证SubmitCacheEntry后AddrRefreshInfo已正确设置
    EXPECT_TRUE(entry.wqeTaskArrayInfos_[0].locAddrRefreshInfoArray[0].needRefresh);
    EXPECT_EQ(entry.wqeTaskArrayInfos_[0].locAddrRefreshInfoArray[0].memIdx, 0U);

    // 注意: WQE remote addr不再检查本地地址, 因此needRefresh预期为false, memIdx为默认值0U
    EXPECT_FALSE(entry.wqeTaskArrayInfos_[0].rmtAddrRefreshInfoArray[0].needRefresh);
    EXPECT_EQ(entry.wqeTaskArrayInfos_[0].rmtAddrRefreshInfoArray[0].memIdx, 0U);

    EXPECT_TRUE(entry.sqeArrayInfos_[0].srcAddrRefreshInfoArray[0].needRefresh);
    EXPECT_TRUE(entry.sqeArrayInfos_[0].dstAddrRefreshInfoArray[0].needRefresh);

    // 手动刷新SQE, 验证地址刷新结果
    uint64_t newBaseAddrs[] = {0x50000, 0x60000};
    uint64_t newMemSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    ASSERT_EQ(entry.RefreshSqeTasks_(entry.sqeArrayInfos_[0], newBaseAddrs), HCCL_SUCCESS);
    Rt91095StarsMemcpySqe* sdmaSqe = reinterpret_cast<Rt91095StarsMemcpySqe*>(entry.sqeArrayInfos_[0].sqeArray);
    uint64_t newSrcAddr = 0;
    hcomm::AicpuTaskCacheEntry::CombineUint32ToUint64(
        newSrcAddr, sdmaSqe->u.strideMode0.srcAddrHigh, sdmaSqe->u.strideMode0.srcAddrLow);
    EXPECT_EQ(newSrcAddr, 0x50000ULL + 0x100);
    uint64_t newDstAddr = 0;
    hcomm::AicpuTaskCacheEntry::CombineUint32ToUint64(
        newDstAddr, sdmaSqe->u.strideMode0.dstAddrHigh, sdmaSqe->u.strideMode0.dstAddrLow);
    EXPECT_EQ(newDstAddr, 0x60000ULL + 0x200);

    // 手动刷新WQE, 验证地址刷新结果
    ASSERT_EQ(entry.RefreshWqeTasks_(entry.wqeTaskArrayInfos_[0], newBaseAddrs, newMemSizes, 2), HCCL_SUCCESS);
    uint64_t newLocAddr = 0;
    hcomm::AicpuTaskCacheEntry::CombineUint32ToUint64(
        newLocAddr, entry.wqeTaskArrayInfos_[0].wqeTaskArray[0].wqeWrite.u.sge.dataAddrHigh,
        entry.wqeTaskArrayInfos_[0].wqeTaskArray[0].wqeWrite.u.sge.dataAddrLow);
    EXPECT_EQ(newLocAddr, 0x50000ULL + 0x10);
    const auto& refreshedWqe = entry.wqeTaskArrayInfos_[0].wqeTaskArray[0].wqeWrite;
    EXPECT_EQ(refreshedWqe.u.sge.tokenId, 99U);
    EXPECT_EQ(refreshedWqe.comm.rmtObjId, 23U);
    EXPECT_EQ(refreshedWqe.comm.rmtTokenValue, 29U);
    // 注意: WQE remote addr不再检查本地地址, 因此地址不需要刷新, 仍然为初始值
    uint64_t newRmtAddr = 0;
    hcomm::AicpuTaskCacheEntry::CombineUint32ToUint64(
        newRmtAddr, entry.wqeTaskArrayInfos_[0].wqeTaskArray[0].wqeWrite.comm.rmtAddrHigh,
        entry.wqeTaskArrayInfos_[0].wqeTaskArray[0].wqeWrite.comm.rmtAddrLow);
    EXPECT_EQ(newRmtAddr, TEST_BASE_ADDR_1 + 0x20);

    // 非 V 路径使用更新后的 token；硬件回放在 ST 中验证。
    EXPECT_TRUE(tokenInfos[0].needLocTokenIdFlag);
    // 注意: WQE remote addr不再检查本地地址, 因此needRmtTokenIdAndValueFlag预期为false
    EXPECT_FALSE(tokenInfos[1].needRmtTokenIdAndValueFlag);
}

// ===================== RefreshSqeTasks_ Tests =====================

TEST_F(AicpuTaskCacheEntryTest, RefreshSqeTasks_NotifyWait_Success)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
    ASSERT_EQ(
        entry.AddSqeArray(
            rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0, g_defaultTransports.data(),
            g_defaultLinkTypes.data(), g_defaultTransportTypes.data()),
        HCCL_SUCCESS);
    ASSERT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);

    uint64_t newBaseAddrs[] = {0x50000, 0x60000};
    uint64_t newMemSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    EXPECT_EQ(entry.RefreshSqeTasks_(entry.sqeArrayInfos_[0], newBaseAddrs), HCCL_SUCCESS);

    Rt91095StarsSqeHeader* header = reinterpret_cast<Rt91095StarsSqeHeader*>(entry.sqeArrayInfos_[0].sqeArray);
    EXPECT_EQ(static_cast<Rt91095StarsSqeType>(header->type), Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
}

TEST_F(AicpuTaskCacheEntryTest, RefreshSqeTasks_NotifyRecord_Success)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_RECORD);
    ASSERT_EQ(
        entry.AddSqeArray(
            rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0, g_defaultTransports.data(),
            g_defaultLinkTypes.data(), g_defaultTransportTypes.data()),
        HCCL_SUCCESS);
    ASSERT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);

    uint64_t newBaseAddrs[] = {0x50000, 0x60000};
    uint64_t newMemSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    EXPECT_EQ(entry.RefreshSqeTasks_(entry.sqeArrayInfos_[0], newBaseAddrs), HCCL_SUCCESS);
}

TEST_F(AicpuTaskCacheEntryTest, RefreshSqeTasks_Ubdma_Success)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_UBDMA);
    ASSERT_EQ(
        entry.AddSqeArray(
            rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0, g_defaultTransports.data(),
            g_defaultLinkTypes.data(), g_defaultTransportTypes.data()),
        HCCL_SUCCESS);
    ASSERT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);

    uint64_t newBaseAddrs[] = {0x50000, 0x60000};
    uint64_t newMemSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    EXPECT_EQ(entry.RefreshSqeTasks_(entry.sqeArrayInfos_[0], newBaseAddrs), HCCL_SUCCESS);
}

TEST_F(AicpuTaskCacheEntryTest, RefreshSqeTasks_Sdma_BothAddrsRefreshed)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    uint64_t srcAddr = TEST_BASE_ADDR_0 + 0x100;
    uint64_t dstAddr = TEST_BASE_ADDR_1 + 0x200;
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA, srcAddr, dstAddr);
    ASSERT_EQ(
        entry.AddSqeArray(
            rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0, g_defaultTransports.data(),
            g_defaultLinkTypes.data(), g_defaultTransportTypes.data()),
        HCCL_SUCCESS);
    ASSERT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);

    uint64_t newBaseAddrs[] = {0x50000, 0x60000};
    uint64_t newMemSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    EXPECT_EQ(entry.RefreshSqeTasks_(entry.sqeArrayInfos_[0], newBaseAddrs), HCCL_SUCCESS);

    Rt91095StarsMemcpySqe* sdmaSqe = reinterpret_cast<Rt91095StarsMemcpySqe*>(entry.sqeArrayInfos_[0].sqeArray);
    uint64_t newSrcAddr = 0;
    hcomm::AicpuTaskCacheEntry::CombineUint32ToUint64(
        newSrcAddr, sdmaSqe->u.strideMode0.srcAddrHigh, sdmaSqe->u.strideMode0.srcAddrLow);
    EXPECT_EQ(newSrcAddr, 0x50000ULL + 0x100);

    uint64_t newDstAddr = 0;
    hcomm::AicpuTaskCacheEntry::CombineUint32ToUint64(
        newDstAddr, sdmaSqe->u.strideMode0.dstAddrHigh, sdmaSqe->u.strideMode0.dstAddrLow);
    EXPECT_EQ(newDstAddr, 0x60000ULL + 0x200);
}

TEST_F(AicpuTaskCacheEntryTest, RefreshSqeTasks_VSdma_UsesCurrentCountsAndDisplacements)
{
    hcomm::AicpuTaskCacheEntry entry;
    uint64_t baseAddrs[] = {TEST_BASE_ADDR_0, TEST_BASE_ADDR_1};
    uint64_t memSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    ASSERT_EQ(entry.InitCacheEntry(baseAddrs, memSizes, 2, VOpInfo{{16}, {16}, {0}, {0}}), HCCL_SUCCESS);

    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA, TEST_BASE_ADDR_0, TEST_BASE_ADDR_1, 16);
    ASSERT_EQ(
        entry.AddSqeArray(
            rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0, g_defaultTransports.data(),
            g_defaultLinkTypes.data(), g_defaultTransportTypes.data()),
        HCCL_SUCCESS);
    ASSERT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);

    uint64_t newBaseAddrs[] = {0x50000, 0x60000};
    const VOpInfo current{{8}, {8}, {32}, {64}};
    ASSERT_EQ(entry.RefreshSqeTasks_(entry.sqeArrayInfos_[0], newBaseAddrs, current), HCCL_SUCCESS);

    auto* sdmaSqe = reinterpret_cast<Rt91095StarsMemcpySqe*>(entry.sqeArrayInfos_[0].sqeArray);
    EXPECT_EQ(static_cast<Rt91095StarsSqeType>(sdmaSqe->header.type), Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA);
    uint64_t newSrcAddr = 0;
    uint64_t newDstAddr = 0;
    hcomm::AicpuTaskCacheEntry::CombineUint32ToUint64(
        newSrcAddr, sdmaSqe->u.strideMode0.srcAddrHigh, sdmaSqe->u.strideMode0.srcAddrLow);
    hcomm::AicpuTaskCacheEntry::CombineUint32ToUint64(
        newDstAddr, sdmaSqe->u.strideMode0.dstAddrHigh, sdmaSqe->u.strideMode0.dstAddrLow);
    EXPECT_EQ(newSrcAddr, 0x50000ULL + 32);
    EXPECT_EQ(newDstAddr, 0x60000ULL + 64);

    const VOpInfo zeroCount{{0}, {0}, {32}, {64}};
    uint64_t emptyAddrs[] = {0, 0};
    ASSERT_EQ(entry.RefreshSqeTasks_(entry.sqeArrayInfos_[0], emptyAddrs, zeroCount), HCCL_SUCCESS);
    EXPECT_EQ(
        static_cast<Rt91095StarsSqeType>(sdmaSqe->header.type), Rt91095StarsSqeType::RT_91095_SQE_TYPE_PLACE_HOLDER);
}

TEST_F(AicpuTaskCacheEntryTest, LaunchTasksByOrder_VSdmaRestoresTypesAfterRefreshOrLaunchFailure)
{
    for (bool failDuringRefresh : {true, false}) {
        SCOPED_TRACE(failDuringRefresh);
        AicpuTaskCacheEntry entry;
        uint64_t baseAddrs[] = {TEST_BASE_ADDR_0, TEST_BASE_ADDR_1};
        uint64_t memSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
        const VOpInfo warmup{{8, 8}, {8, 8}, {0, 16}, {0, 16}};
        ASSERT_EQ(entry.InitCacheEntry(baseAddrs, memSizes, 2, warmup), HCCL_SUCCESS);
        auto sqes = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA, baseAddrs[0], baseAddrs[1], 8);
        const auto secondSqe
            = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA, baseAddrs[0] + 16, baseAddrs[1] + 16, 8);
        sqes.insert(sqes.end(), secondSqe.begin(), secondSqe.end());
        ASSERT_EQ(
            entry.AddSqeArray(
                rtsqPtr_, aicpuTsThread_.get(), 2, sqes.data(), 0, g_defaultTransports.data(),
                g_defaultLinkTypes.data(), g_defaultTransportTypes.data()),
            HCCL_SUCCESS);
        ASSERT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);

        // 第一条改成PLACE_HOLDER后，让第二条长度溢出，或让下发因pending SQE失败。
        const uint64_t secondCount = failDuringRefresh ? uint64_t{UINT32_MAX} + 1 : 0;
        const VOpInfo failing{{0, secondCount}, {0, secondCount}, {0, 16}, {0, 16}};
        rtsqPtr_->pendingSqeCnt = failDuringRefresh ? 0 : 1;
        const auto result = entry.LaunchTasksByOrder_(baseAddrs, memSizes, 2, false, failing);
        rtsqPtr_->pendingSqeCnt = 0;
        EXPECT_EQ(result, failDuringRefresh ? HCCL_E_PARA : HCCL_E_INTERNAL);

        const auto& arrayInfo = entry.sqeArrayInfos_[0];
        for (uint64_t index = 0; index < arrayInfo.sqeCount; ++index) {
            const auto* header
                = reinterpret_cast<const Rt91095StarsSqeHeader*>(arrayInfo.sqeArray + index * Hccl::AC_SQE_SIZE);
            EXPECT_EQ(static_cast<Rt91095StarsSqeType>(header->type), Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA);
        }
        // 不访问硬件队列，验证失败后的缓存可按非零count再次刷新。
        const VOpInfo retry{{12, 20}, {12, 20}, {32, 64}, {48, 96}};
        ASSERT_EQ(entry.RefreshSqeTasks_(arrayInfo, baseAddrs, retry), HCCL_SUCCESS);
        for (uint64_t index = 0; index < arrayInfo.sqeCount; ++index) {
            const auto* sdma
                = reinterpret_cast<const Rt91095StarsMemcpySqe*>(arrayInfo.sqeArray + index * Hccl::AC_SQE_SIZE);
            EXPECT_EQ(static_cast<Rt91095StarsSqeType>(sdma->header.type), Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA);
            EXPECT_EQ(sdma->u.strideMode0.lengthMove, retry.sendCounts[index]);
            EXPECT_EQ(sdma->u.strideMode0.srcAddrLow, baseAddrs[0] + retry.sdispls[index]);
            EXPECT_EQ(sdma->u.strideMode0.dstAddrLow, baseAddrs[1] + retry.rdispls[index]);
        }
    }
}

TEST_F(AicpuTaskCacheEntryTest, RefreshSqeTasks_Sdma_AddrNotInRange)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    uint64_t outOfRangeAddr = 0x999000;
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA, outOfRangeAddr, outOfRangeAddr);
    ASSERT_EQ(
        entry.AddSqeArray(
            rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0, g_defaultTransports.data(),
            g_defaultLinkTypes.data(), g_defaultTransportTypes.data()),
        HCCL_SUCCESS);
    ASSERT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);

    uint64_t newBaseAddrs[] = {0x50000, 0x60000};
    uint64_t newMemSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    EXPECT_EQ(entry.RefreshSqeTasks_(entry.sqeArrayInfos_[0], newBaseAddrs), HCCL_SUCCESS);

    Rt91095StarsMemcpySqe* sdmaSqe = reinterpret_cast<Rt91095StarsMemcpySqe*>(entry.sqeArrayInfos_[0].sqeArray);
    uint64_t newSrcAddr = 0;
    hcomm::AicpuTaskCacheEntry::CombineUint32ToUint64(
        newSrcAddr, sdmaSqe->u.strideMode0.srcAddrHigh, sdmaSqe->u.strideMode0.srcAddrLow);
    EXPECT_EQ(newSrcAddr, outOfRangeAddr);

    uint64_t newDstAddr = 0;
    hcomm::AicpuTaskCacheEntry::CombineUint32ToUint64(
        newDstAddr, sdmaSqe->u.strideMode0.dstAddrHigh, sdmaSqe->u.strideMode0.dstAddrLow);
    EXPECT_EQ(newDstAddr, outOfRangeAddr);
}

TEST_F(AicpuTaskCacheEntryTest, RefreshSqeTasks_WriteValue_DstAddrRefreshed)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    uint64_t dstAddr = TEST_BASE_ADDR_0 + 0x50;
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_WRITE_VALUE, 0, dstAddr);
    ASSERT_EQ(
        entry.AddSqeArray(
            rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0, g_defaultTransports.data(),
            g_defaultLinkTypes.data(), g_defaultTransportTypes.data()),
        HCCL_SUCCESS);
    ASSERT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);

    uint64_t newBaseAddrs[] = {0x50000, 0x60000};
    uint64_t newMemSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    EXPECT_EQ(entry.RefreshSqeTasks_(entry.sqeArrayInfos_[0], newBaseAddrs), HCCL_SUCCESS);

    Rt91095StarsWriteValueSqe* wvSqe = reinterpret_cast<Rt91095StarsWriteValueSqe*>(entry.sqeArrayInfos_[0].sqeArray);
    uint64_t newDstAddr = 0;
    hcomm::AicpuTaskCacheEntry::CombineUint32ToUint64(newDstAddr, wvSqe->writeAddrHigh, wvSqe->writeAddrLow);
    EXPECT_EQ(newDstAddr, 0x50000ULL + 0x50);
}

TEST_F(AicpuTaskCacheEntryTest, RefreshSqeTasks_MultipleSqes)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);

    std::vector<uint8_t> sqeArray(3 * Hccl::AC_SQE_SIZE, 0);
    auto sqe0 = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
    memcpy(sqeArray.data(), sqe0.data(), Hccl::AC_SQE_SIZE);
    uint64_t srcAddr = TEST_BASE_ADDR_0 + 0x100;
    uint64_t dstAddr = TEST_BASE_ADDR_1 + 0x200;
    auto sqe1 = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA, srcAddr, dstAddr);
    memcpy(sqeArray.data() + Hccl::AC_SQE_SIZE, sqe1.data(), Hccl::AC_SQE_SIZE);
    uint64_t wvDstAddr = TEST_BASE_ADDR_0 + 0x50;
    auto sqe2 = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_WRITE_VALUE, 0, wvDstAddr);
    memcpy(sqeArray.data() + 2 * Hccl::AC_SQE_SIZE, sqe2.data(), Hccl::AC_SQE_SIZE);

    ASSERT_EQ(
        entry.AddSqeArray(
            rtsqPtr_, aicpuTsThread_.get(), 3, sqeArray.data(), 0, g_defaultTransports.data(),
            g_defaultLinkTypes.data(), g_defaultTransportTypes.data()),
        HCCL_SUCCESS);
    ASSERT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);

    uint64_t newBaseAddrs[] = {0x50000, 0x60000};
    uint64_t newMemSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    EXPECT_EQ(entry.RefreshSqeTasks_(entry.sqeArrayInfos_[0], newBaseAddrs), HCCL_SUCCESS);

    uint8_t* base = entry.sqeArrayInfos_[0].sqeArray;

    Rt91095StarsSqeHeader* header0 = reinterpret_cast<Rt91095StarsSqeHeader*>(base);
    EXPECT_EQ(static_cast<Rt91095StarsSqeType>(header0->type), Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);

    Rt91095StarsMemcpySqe* sdmaSqe = reinterpret_cast<Rt91095StarsMemcpySqe*>(base + Hccl::AC_SQE_SIZE);
    uint64_t newSrcAddr = 0;
    hcomm::AicpuTaskCacheEntry::CombineUint32ToUint64(
        newSrcAddr, sdmaSqe->u.strideMode0.srcAddrHigh, sdmaSqe->u.strideMode0.srcAddrLow);
    EXPECT_EQ(newSrcAddr, 0x50000ULL + 0x100);
    uint64_t newDstAddr = 0;
    hcomm::AicpuTaskCacheEntry::CombineUint32ToUint64(
        newDstAddr, sdmaSqe->u.strideMode0.dstAddrHigh, sdmaSqe->u.strideMode0.dstAddrLow);
    EXPECT_EQ(newDstAddr, 0x60000ULL + 0x200);

    Rt91095StarsWriteValueSqe* wvSqe = reinterpret_cast<Rt91095StarsWriteValueSqe*>(base + 2 * Hccl::AC_SQE_SIZE);
    uint64_t newWvDstAddr = 0;
    hcomm::AicpuTaskCacheEntry::CombineUint32ToUint64(newWvDstAddr, wvSqe->writeAddrHigh, wvSqe->writeAddrLow);
    EXPECT_EQ(newWvDstAddr, 0x50000ULL + 0x50);
}

TEST_F(AicpuTaskCacheEntryTest, RefreshSqeTasks_InvalidSqeType)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_AIC);
    ASSERT_EQ(
        entry.AddSqeArray(
            rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0, g_defaultTransports.data(),
            g_defaultLinkTypes.data(), g_defaultTransportTypes.data()),
        HCCL_SUCCESS);

    uint64_t newBaseAddrs[] = {0x50000, 0x60000};
    uint64_t newMemSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    EXPECT_EQ(entry.RefreshSqeTasks_(entry.sqeArrayInfos_[0], newBaseAddrs), HCCL_E_INTERNAL);
}

TEST_F(AicpuTaskCacheEntryTest, RefreshSqeTasks_TaskConfigDebug)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
    ASSERT_EQ(
        entry.AddSqeArray(
            rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0, g_defaultTransports.data(),
            g_defaultLinkTypes.data(), g_defaultTransportTypes.data()),
        HCCL_SUCCESS);
    ASSERT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);

    entry.isTaskConfigDebug_ = true;
    uint64_t newBaseAddrs[] = {0x50000, 0x60000};
    uint64_t newMemSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    EXPECT_EQ(entry.RefreshSqeTasks_(entry.sqeArrayInfos_[0], newBaseAddrs), HCCL_SUCCESS);
    entry.isTaskConfigDebug_ = false;
}

// ===================== CheckWqeOverflow_ Tests =====================
// 注意: 不构造溢出用例, 溢出路径会阻塞等待sqFullTimeout后抛出InternalException, UT环境无法快速返回

TEST_F(AicpuTaskCacheEntryTest, CheckWqeOverflow_NoOverflow_ReturnsSuccess)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);

    uint64_t locAddr = TEST_BASE_ADDR_0 + 0x10;
    uint64_t rmtAddr = TEST_BASE_ADDR_1 + 0x20;
    std::vector<Hccl::WqeTask> wqeTasks = {MakeWqeTaskRead(locAddr, rmtAddr)};
    ASSERT_EQ(
        entry.AddWqeArray(ubConnLite_.get(), ubTransport_.get(), wqeTasks, 0, 0, false, MakeDbSqeProfInfo(false)),
        HCCL_SUCCESS);
    AddUbdmaSqeToClearTmpMap(entry);

    ASSERT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);

    // pi=0, ci=0, inflight=0, wqebbCount=1, 0+1 <= 128
    EXPECT_EQ(entry.CheckWqeOverflow_(), HCCL_SUCCESS);
}

TEST_F(AicpuTaskCacheEntryTest, BuildConnOverflowInfos_WriteWithNotifyOccupiesTwoWqebb)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);

    uint64_t locAddr = TEST_BASE_ADDR_0 + 0x10;
    uint64_t rmtAddr = TEST_BASE_ADDR_1 + 0x20;
    // 带notify的WQE占用2个wqebb(LaunchOneWqeWithNotify中pi += 2), 普通read占用1个wqebb
    std::vector<Hccl::WqeTask> wqeTasks
        = {MakeWqeTaskRead(locAddr, rmtAddr), MakeWqeTaskWriteWithNotify(locAddr, rmtAddr)};
    ASSERT_EQ(
        entry.AddWqeArray(ubConnLite_.get(), ubTransport_.get(), wqeTasks, 0, 0, false, MakeDbSqeProfInfo(false)),
        HCCL_SUCCESS);
    AddUbdmaSqeToClearTmpMap(entry);

    ASSERT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);

    // 同一UbConnLite只有一组: 1(read) + 2(write with notify) = 3个wqebb
    ASSERT_EQ(entry.connOverflowInfos_.size(), 1U);
    EXPECT_EQ(entry.connOverflowInfos_[0].wqebbCount, 3U);
}

TEST_F(AicpuTaskCacheEntryTest, BuildConnOverflowInfos_SameConnAccumulatesAcrossArrays)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);

    uint64_t locAddr = TEST_BASE_ADDR_0 + 0x10;
    uint64_t rmtAddr = TEST_BASE_ADDR_1 + 0x20;
    std::vector<Hccl::WqeTask> wqeTasks0 = {MakeWqeTaskWriteWithNotify(locAddr, rmtAddr)};
    ASSERT_EQ(
        entry.AddWqeArray(ubConnLite_.get(), ubTransport_.get(), wqeTasks0, 0, 0, false, MakeDbSqeProfInfo(false)),
        HCCL_SUCCESS);
    std::vector<Hccl::WqeTask> wqeTasks1 = {MakeWqeTaskRead(locAddr, rmtAddr)};
    ASSERT_EQ(
        entry.AddWqeArray(ubConnLite_.get(), ubTransport_.get(), wqeTasks1, 1, 0, false, MakeDbSqeProfInfo(false)),
        HCCL_SUCCESS);
    AddUbdmaSqeToClearTmpMap(entry, 0);
    AddUbdmaSqeToClearTmpMap(entry, 1);

    ASSERT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);

    // 同一UbConnLite被两个WQE数组引用, 占用累加: 2(write with notify) + 1(read) = 3个wqebb
    ASSERT_EQ(entry.connOverflowInfos_.size(), 1U);
    EXPECT_EQ(entry.connOverflowInfos_[0].wqebbCount, 3U);
}

// ===================== ReportSqeArrayProfiling_ (profiling disabled path) =====================

TEST_F(AicpuTaskCacheEntryTest, ReportDbSqeProfiling_VReplayUsesCurrentWqeAndPreservesTemplate)
{
    AicpuTaskCacheManager::RegisterGetChannelRemoteRankIdCallback(GetWqeRemoteRankForTest);
    constexpr uint64_t REMOTE_ADDR = 0x200900000ULL;
    for (uint8_t opcode :
         {static_cast<uint8_t>(UdmaSqOpcode::UDMA_OPC_READ), static_cast<uint8_t>(UdmaSqOpcode::UDMA_OPC_WRITE),
          static_cast<uint8_t>(WRITE_WITH_NOTIFY_OPCODE)}) {
        SCOPED_TRACE(opcode);
        const bool isRead = opcode == UdmaSqOpcode::UDMA_OPC_READ;
        const bool withNotify = opcode == WRITE_WITH_NOTIFY_OPCODE;
        const uint32_t memIdx = isRead ? 1 : 0;
        const uint64_t oldDispl = isRead ? 64 : 32;
        AicpuTaskCacheEntry entry;
        uint64_t oldAddrs[] = {TEST_BASE_ADDR_0, TEST_BASE_ADDR_1};
        uint64_t oldSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
        const VOpInfo warmup{{4, 4}, {4, 4}, {0, 32}, {0, 64}};
        ASSERT_EQ(entry.InitCacheEntry(oldAddrs, oldSizes, 2, warmup), HCCL_SUCCESS);
        const uint64_t oldLocalAddr = oldAddrs[memIdx] + oldDispl;
        auto dataTask = isRead ? MakeWqeTaskRead(oldLocalAddr, REMOTE_ADDR) :
                                 (withNotify ? MakeWqeTaskWriteWithNotify(oldLocalAddr, REMOTE_ADDR) :
                                               MakeWqeTaskWrite(oldLocalAddr, REMOTE_ADDR, false));
        if (withNotify) {
            dataTask.wqeWriteWithNotify.localU.sge.length = 4;
        } else {
            dataTask.wqeWrite.u.sge.length = 4;
        }
        dataTask.wqeWrite.comm.sgeNum = 1;
        auto inlineTask = MakeWqeTaskWrite(0, REMOTE_ADDR + 128, true);
        inlineTask.wqeWrite.u.sge.length = 8; // 控制任务不计入数据长度。
        auto profInfo = MakeDbSqeProfInfo(true);
        profInfo.taskParamType = withNotify ? TaskParamType::TASK_WRITE_WITH_NOTIFY : TaskParamType::TASK_UB;
        profInfo.locAddr = oldLocalAddr;
        profInfo.rmtAddr = REMOTE_ADDR;
        profInfo.size = 4;
        profInfo.notifyId = withNotify ? 7 : Hccl::INVALID_U32;
        ASSERT_EQ(
            entry.AddWqeArray(ubConnLite_.get(), ubTransport_.get(), {dataTask, inlineTask}, 0, 0, true, profInfo),
            HCCL_SUCCESS);
        AddUbdmaSqeToClearTmpMap(entry);
        ASSERT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);
        ASSERT_EQ(entry.dbSqeLocInfoMap_.size(), 1U);
        const auto& recorded = entry.dbSqeLocInfoMap_.begin()->second;
        const auto originalOpcodes = entry.wqeTaskArrayInfos_[0].originalOpcodes;
        const auto originalSgeNums = entry.wqeTaskArrayInfos_[0].originalSgeNums;
        const uint64_t entryBytes = entry.GetEntryBytes();
        StreamLite stream(0, 0, 0, 0);
        auto* queue = stream.GetTaskInfos();
        uint64_t iteration = 0;
        for (uint64_t bytes : {uint64_t{4096}, uint64_t{0}, uint64_t{32}, uint64_t{0}, uint64_t{128}}) {
            SCOPED_TRACE(bytes);
            ++iteration;
            uint64_t addrs[] = {0x100050000ULL + iteration * 0x10000, 0x100060000ULL + iteration * 0x10000};
            uint64_t sizes[] = {0x10000, 0x10000};
            const uint64_t sendDispl = 96 + iteration * 16;
            const uint64_t recvDispl = 128 + iteration * 16;
            VOpInfo current{{4, bytes}, {4, bytes}, {0, sendDispl}, {0, recvDispl}};
            if (bytes == 0) {
                addrs[memIdx] = 0;
                (isRead ? current.recvCounts : current.sendCounts)[0] = 0;
            }
            ASSERT_EQ(entry.RefreshWqeTasks_(entry.wqeTaskArrayInfos_[0], addrs, sizes, 2, current), HCCL_SUCCESS);
            // 仅刷新 WQE 并生成 DFX 描述，不向硬件队列提交任务。
            queue->MarkAllRead();
            ASSERT_EQ(
                entry.ReportDbSqeProfiling_(entry.sqeArrayInfos_[0].sqeArray, 0, 0, addrs, sizes, 2, &stream, 0, 42),
                HCCL_SUCCESS);
            ASSERT_EQ(queue->GetCount(), 1U);
            const auto* slot = queue->GetSlot(queue->GetBegin());
            EXPECT_EQ(slot->taskType, static_cast<uint8_t>(profInfo.taskParamType));
            EXPECT_EQ(slot->taskPara.ubDma.size, bytes);
            EXPECT_EQ(slot->taskPara.ubDma.srcAddr, bytes == 0 ? 0 : addrs[memIdx] + (isRead ? recvDispl : sendDispl));
            EXPECT_EQ(slot->taskPara.ubDma.dstAddr, bytes == 0 ? 0 : REMOTE_ADDR);
            EXPECT_EQ(slot->taskPara.ubDma.notifyId, profInfo.notifyId);
            EXPECT_EQ(slot->taskId, 42U);
            EXPECT_EQ(recorded.dbSqeProfInfo.locAddr, oldLocalAddr);
            EXPECT_EQ(recorded.dbSqeProfInfo.rmtAddr, REMOTE_ADDR);
            EXPECT_EQ(recorded.dbSqeProfInfo.size, 4U);
            EXPECT_EQ(recorded.dbSqeProfInfo.notifyId, profInfo.notifyId);
            EXPECT_EQ(recorded.locAddrRefreshInfo.offset, oldDispl);
            EXPECT_EQ(entry.wqeTaskArrayInfos_[0].originalOpcodes, originalOpcodes);
            EXPECT_EQ(entry.wqeTaskArrayInfos_[0].originalSgeNums, originalSgeNums);
            EXPECT_EQ(entry.vOpInfo_.sendCounts, warmup.sendCounts);
            EXPECT_EQ(entry.vOpInfo_.recvCounts, warmup.recvCounts);
            EXPECT_EQ(entry.vOpInfo_.sdispls, warmup.sdispls);
            EXPECT_EQ(entry.vOpInfo_.rdispls, warmup.rdispls);
            EXPECT_EQ(entry.GetEntryBytes(), entryBytes);
        }
    }
}

TEST_F(AicpuTaskCacheEntryTest, ReportDbSqeProfiling_VInlineNotifyKeepsRecordedSize)
{
    AicpuTaskCacheManager::RegisterGetChannelRemoteRankIdCallback(GetWqeRemoteRankForTest);
    AicpuTaskCacheEntry entry;
    uint64_t addrs[] = {TEST_BASE_ADDR_0, TEST_BASE_ADDR_1};
    uint64_t sizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    ASSERT_EQ(entry.InitCacheEntry(addrs, sizes, 2, VOpInfo{{4, 4}, {4, 4}, {0, 32}, {0, 64}}), HCCL_SUCCESS);
    constexpr uint64_t NOTIFY_ADDR = 0x900000;
    auto task = MakeWqeTaskWrite(0, NOTIFY_ADDR, true);
    auto profInfo = MakeDbSqeProfInfo(true);
    profInfo.taskParamType = TaskParamType::TASK_UB_INLINE_WRITE;
    profInfo.rmtAddr = NOTIFY_ADDR;
    profInfo.size = 8;
    profInfo.notifyId = 9;
    ASSERT_EQ(entry.AddWqeArray(ubConnLite_.get(), ubTransport_.get(), {task}, 0, 0, true, profInfo), HCCL_SUCCESS);
    AddUbdmaSqeToClearTmpMap(entry);
    ASSERT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);
    StreamLite stream(0, 0, 0, 0);
    const VOpInfo zeroCounts{{0, 0}, {0, 0}, {0, 96}, {0, 128}};
    addrs[0] = 0;
    addrs[1] = 0;
    ASSERT_EQ(entry.RefreshWqeTasks_(entry.wqeTaskArrayInfos_[0], addrs, sizes, 2, zeroCounts), HCCL_SUCCESS);
    ASSERT_EQ(
        entry.ReportDbSqeProfiling_(entry.sqeArrayInfos_[0].sqeArray, 0, 0, addrs, sizes, 2, &stream, 0, 1),
        HCCL_SUCCESS);
    auto* queue = stream.GetTaskInfos();
    ASSERT_EQ(queue->GetCount(), 1U);
    const auto* slot = queue->GetSlot(queue->GetBegin());
    EXPECT_EQ(slot->taskType, static_cast<uint8_t>(TaskParamType::TASK_UB_INLINE_WRITE));
    EXPECT_EQ(slot->taskPara.ubDma.dstAddr, NOTIFY_ADDR);
    EXPECT_EQ(slot->taskPara.ubDma.size, 8U);
    EXPECT_EQ(slot->taskPara.ubDma.notifyId, 9U);
    EXPECT_EQ(entry.dbSqeLocInfoMap_.begin()->second.dbSqeProfInfo.rmtAddr, NOTIFY_ADDR);
}

TEST_F(AicpuTaskCacheEntryTest, ReportDbSqeProfiling_NonVRefreshesAddressesWithoutChangingTemplate)
{
    AicpuTaskCacheEntry entry;
    uint64_t addrs[] = {TEST_BASE_ADDR_0, TEST_BASE_ADDR_1, TEST_REMOTE_BASE_ADDR};
    uint64_t sizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1, TEST_REMOTE_MEM_SIZE};
    ASSERT_EQ(entry.InitCacheEntry(addrs, sizes, 3), HCCL_SUCCESS);
    auto profInfo = MakeDbSqeProfInfo(true);
    profInfo.taskParamType = TaskParamType::TASK_UB;
    profInfo.locAddr = TEST_BASE_ADDR_0 + 16;
    profInfo.rmtAddr = TEST_REMOTE_BASE_ADDR + 32;
    profInfo.size = 64;
    ASSERT_EQ(
        entry.AddWqeArray(
            ubConnLite_.get(), ubTransport_.get(), {MakeWqeTaskWrite(profInfo.locAddr, profInfo.rmtAddr, false)}, 0, 0,
            true, profInfo),
        HCCL_SUCCESS);
    AddUbdmaSqeToClearTmpMap(entry);
    ASSERT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);
    StreamLite stream(0, 0, 0, 0);
    for (uint64_t base : {uint64_t{0x50000}, uint64_t{0x100060000}}) {
        addrs[0] = base;
        addrs[2] = base + 0x10000;
        stream.GetTaskInfos()->MarkAllRead();
        ASSERT_EQ(
            entry.ReportDbSqeProfiling_(entry.sqeArrayInfos_[0].sqeArray, 0, 0, addrs, sizes, 3, &stream, 0, 1),
            HCCL_SUCCESS);
        const auto* queue = stream.GetTaskInfos();
        ASSERT_EQ(queue->GetCount(), 1U);
        const auto* slot = queue->GetSlot(queue->GetBegin());
        EXPECT_EQ(slot->taskPara.ubDma.srcAddr, addrs[0] + 16);
        EXPECT_EQ(slot->taskPara.ubDma.dstAddr, addrs[2] + 32);
        EXPECT_EQ(slot->taskPara.ubDma.size, 64U);
        EXPECT_EQ(entry.dbSqeLocInfoMap_.begin()->second.dbSqeProfInfo.locAddr, profInfo.locAddr);
        EXPECT_EQ(entry.dbSqeLocInfoMap_.begin()->second.dbSqeProfInfo.rmtAddr, profInfo.rmtAddr);
    }
}

TEST_F(AicpuTaskCacheEntryTest, GetEntryBytes_AfterInit)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    uint64_t bytesAfterInit = entry.GetEntryBytes();
    EXPECT_GT(bytesAfterInit, 0U);

    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
    ASSERT_EQ(
        entry.AddSqeArray(
            rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0, g_defaultTransports.data(),
            g_defaultLinkTypes.data(), g_defaultTransportTypes.data()),
        HCCL_SUCCESS);
    EXPECT_GT(entry.GetEntryBytes(), bytesAfterInit);
}

TEST_F(AicpuTaskCacheEntryTest, FillSlot_UbDma_JettyPassthrough)
{
    hcomm::AicpuTaskCacheEntry entry;
    const u64 jettyHandle = 0xABCDEF1234ULL;
    const u32 jettyId = 42;

    Hccl::StreamLite streamLite(0, 0, 0, 0);
    Hccl::DfxTaskInfo slot{};
    DbSqeProfAndRefreshInfo profAndRefreshInfo;
    profAndRefreshInfo.dbSqeProfInfo.jettyHandle = jettyHandle;
    profAndRefreshInfo.dbSqeProfInfo.jettyId = jettyId;

    HcclResult ret = entry.FillSlotUbDma_(&slot, profAndRefreshInfo, ubTransport_.get(), &streamLite, 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(slot.taskPara.ubDma.jettyHandle, jettyHandle);
    EXPECT_EQ(slot.taskPara.ubDma.jettyId, jettyId);
}

TEST_F(AicpuTaskCacheEntryTest, FillSlot_Reduce_JettyPassthrough)
{
    hcomm::AicpuTaskCacheEntry entry;
    const u64 jettyHandle = 0x1122334455ULL;
    const u32 jettyId = 99;

    Hccl::StreamLite streamLite(0, 0, 0, 0);
    Hccl::DfxTaskInfo slot{};
    DbSqeProfAndRefreshInfo profAndRefreshInfo;
    profAndRefreshInfo.dbSqeProfInfo.jettyHandle = jettyHandle;
    profAndRefreshInfo.dbSqeProfInfo.jettyId = jettyId;

    HcclResult ret = entry.FillSlotReduce_(&slot, profAndRefreshInfo, ubTransport_.get(), &streamLite, 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(slot.taskPara.Reduce.jettyHandle, jettyHandle);
    EXPECT_EQ(slot.taskPara.Reduce.jettyId, jettyId);
}

// ===================== SQE Transport Context Propagation Tests =====================
// 目的: 校验 AddSqeArray 缓存的 sqeTransports/sqeLinkTypes/sqeTransportTypes 与传入值一致,
//       且经 ReportSqeProfiling_ 填充 slot 时 channelHandle/linkType/transportType 与传入值一致.
//       现有 UT 全部使用 g_defaultTransports/g_defaultLinkTypes/g_defaultTransportTypes (与改造前
//       FillSlotNotify_/FillSlotSdma_ 硬编码常量相同), 即使传播链路写错 (下标错位/实参顺序颠倒/
//       WRITE_VALUE 取错数组) 也无法拦截. 以下用例以非默认值覆盖该回归场景.

namespace {
constexpr u64 TEST_NON_DEFAULT_CHANNEL_HANDLE = 0x1234ABCDEF5678ULL;
constexpr u8 TEST_NON_DEFAULT_LINK_TYPE = static_cast<u8>(Hccl::DfxLinkTypeVal::LINK_ROCE);
constexpr u8 TEST_NON_DEFAULT_TRANSPORT_TYPE = static_cast<u8>(Hccl::DfxTransportType::DFX_TRANSPORT_TYPE_ROCE);

void FillNonDefaultTransportContext(
    std::vector<u64>& transports, std::vector<u8>& linkTypes, std::vector<u8>& transportTypes, uint64_t sqeCount)
{
    transports.assign(sqeCount, TEST_NON_DEFAULT_CHANNEL_HANDLE);
    linkTypes.assign(sqeCount, TEST_NON_DEFAULT_LINK_TYPE);
    transportTypes.assign(sqeCount, TEST_NON_DEFAULT_TRANSPORT_TYPE);
}
} // namespace

// 用例1: AddSqeArray 缓存值与传入值一致 (单 SQE, 非默认值)
TEST_F(AicpuTaskCacheEntryTest, AddSqeArray_CachesNonDefaultTransportContext_SingleSqe)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_RECORD);

    std::vector<u64> transports;
    std::vector<u8> linkTypes;
    std::vector<u8> transportTypes;
    FillNonDefaultTransportContext(transports, linkTypes, transportTypes, 1);

    ASSERT_EQ(
        entry.AddSqeArray(
            rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0, transports.data(), linkTypes.data(),
            transportTypes.data()),
        HCCL_SUCCESS);

    ASSERT_EQ(entry.sqeArrayInfos_.size(), 1U);
    ASSERT_EQ(entry.sqeArrayInfos_[0].sqeTransports.size(), 1U);
    EXPECT_EQ(entry.sqeArrayInfos_[0].sqeTransports[0], TEST_NON_DEFAULT_CHANNEL_HANDLE);
    EXPECT_EQ(entry.sqeArrayInfos_[0].sqeLinkTypes[0], TEST_NON_DEFAULT_LINK_TYPE);
    EXPECT_EQ(entry.sqeArrayInfos_[0].sqeTransportTypes[0], TEST_NON_DEFAULT_TRANSPORT_TYPE);
}

// 用例2: AddSqeArray 缓存值与传入值逐下标对应 (多 SQE 不同非默认值, 拦截下标错位)
TEST_F(AicpuTaskCacheEntryTest, AddSqeArray_CachesNonDefaultTransportContext_MultiSqePerIndex)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);

    constexpr uint64_t sqeCount = 3;
    std::vector<uint8_t> sqeArray(sqeCount * Hccl::AC_SQE_SIZE, 0);
    auto sqe0 = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_RECORD);
    auto sqe1 = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA, TEST_BASE_ADDR_0, TEST_BASE_ADDR_1);
    auto sqe2 = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_WRITE_VALUE, 0, TEST_BASE_ADDR_0 + 0x50);
    memcpy(sqeArray.data(), sqe0.data(), Hccl::AC_SQE_SIZE);
    memcpy(sqeArray.data() + Hccl::AC_SQE_SIZE, sqe1.data(), Hccl::AC_SQE_SIZE);
    memcpy(sqeArray.data() + 2 * Hccl::AC_SQE_SIZE, sqe2.data(), Hccl::AC_SQE_SIZE);

    // 每个 SQE 携带不同的非默认值, 拦截下标错位
    std::vector<u64> transports = {0x1111ULL, 0x2222ULL, 0x3333ULL};
    std::vector<u8> linkTypes = {
        static_cast<u8>(Hccl::DfxLinkTypeVal::LINK_ROCE),
        static_cast<u8>(Hccl::DfxLinkTypeVal::LINK_HCCS),
        static_cast<u8>(Hccl::DfxLinkTypeVal::LINK_UB),
    };
    std::vector<u8> transportTypes = {
        static_cast<u8>(Hccl::DfxTransportType::DFX_TRANSPORT_TYPE_ROCE),
        static_cast<u8>(Hccl::DfxTransportType::DFX_TRANSPORT_TYPE_RDMA),
        static_cast<u8>(Hccl::DfxTransportType::DFX_TRANSPORT_TYPE_UB),
    };

    ASSERT_EQ(
        entry.AddSqeArray(
            rtsqPtr_, aicpuTsThread_.get(), sqeCount, sqeArray.data(), 0, transports.data(), linkTypes.data(),
            transportTypes.data()),
        HCCL_SUCCESS);

    ASSERT_EQ(entry.sqeArrayInfos_.size(), 1U);
    ASSERT_EQ(entry.sqeArrayInfos_[0].sqeTransports.size(), sqeCount);
    for (uint64_t i = 0; i < sqeCount; ++i) {
        EXPECT_EQ(entry.sqeArrayInfos_[0].sqeTransports[i], transports[i]) << "sqeIdx=" << i;
        EXPECT_EQ(entry.sqeArrayInfos_[0].sqeLinkTypes[i], linkTypes[i]) << "sqeIdx=" << i;
        EXPECT_EQ(entry.sqeArrayInfos_[0].sqeTransportTypes[i], transportTypes[i]) << "sqeIdx=" << i;
    }
}

// 用例3: FillSlotNotify_ 实参传递正确 (非默认值, 参照 FillSlot_UbDma_JettyPassthrough 模式)
TEST_F(AicpuTaskCacheEntryTest, FillSlotNotify_PropagatesNonDefaultTransportContext)
{
    hcomm::AicpuTaskCacheEntry entry;
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_RECORD);

    Hccl::StreamLite streamLite(0, 0, 0, 0);
    Hccl::DfxTaskInfo slot{};
    const u32 taskId = 0;

    HcclResult ret = entry.FillSlotNotify_(
        &slot, sqeArray.data(), &streamLite, taskId, TEST_NON_DEFAULT_CHANNEL_HANDLE, TEST_NON_DEFAULT_LINK_TYPE,
        TEST_NON_DEFAULT_TRANSPORT_TYPE);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(slot.channelHandle, TEST_NON_DEFAULT_CHANNEL_HANDLE);
    EXPECT_EQ(slot.linkType, TEST_NON_DEFAULT_LINK_TYPE);
    EXPECT_EQ(slot.transportType, TEST_NON_DEFAULT_TRANSPORT_TYPE);
    EXPECT_EQ(slot.taskType, static_cast<u8>(Hccl::TaskParamTypeVal::TASK_NOTIFY_RECORD));
}

// 用例4: FillSlotSdma_ 实参传递正确 (非默认值)
TEST_F(AicpuTaskCacheEntryTest, FillSlotSdma_PropagatesNonDefaultTransportContext)
{
    hcomm::AicpuTaskCacheEntry entry;
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA, TEST_BASE_ADDR_0, TEST_BASE_ADDR_1);

    Hccl::StreamLite streamLite(0, 0, 0, 0);
    Hccl::DfxTaskInfo slot{};
    const u32 taskId = 0;

    HcclResult ret = entry.FillSlotSdma_(
        &slot, sqeArray.data(), &streamLite, taskId, TEST_NON_DEFAULT_CHANNEL_HANDLE, TEST_NON_DEFAULT_LINK_TYPE,
        TEST_NON_DEFAULT_TRANSPORT_TYPE);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(slot.channelHandle, TEST_NON_DEFAULT_CHANNEL_HANDLE);
    EXPECT_EQ(slot.linkType, TEST_NON_DEFAULT_LINK_TYPE);
    EXPECT_EQ(slot.transportType, TEST_NON_DEFAULT_TRANSPORT_TYPE);
    EXPECT_EQ(slot.taskType, static_cast<u8>(Hccl::TaskParamTypeVal::TASK_SDMA));
}

// 用例5: ReportSqeProfiling_ WRITE_VALUE 分支填充 slot 字段与缓存值一致 (端到端, 非默认值)
TEST_F(AicpuTaskCacheEntryTest, ReportSqeProfiling_WriteValue_PropagatesNonDefaultTransportContext)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_WRITE_VALUE, 0, TEST_BASE_ADDR_0 + 0x50);

    std::vector<u64> transports;
    std::vector<u8> linkTypes;
    std::vector<u8> transportTypes;
    FillNonDefaultTransportContext(transports, linkTypes, transportTypes, 1);

    ASSERT_EQ(
        entry.AddSqeArray(
            rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0, transports.data(), linkTypes.data(),
            transportTypes.data()),
        HCCL_SUCCESS);
    ASSERT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);

    Hccl::StreamLite streamLite(0, 0, 0, 0);
    const u16 endBefore = streamLite.GetTaskInfos()->GetEnd();

    uint64_t baseAddrs[] = {TEST_BASE_ADDR_0, TEST_BASE_ADDR_1};
    uint64_t memSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    EXPECT_EQ(
        entry.ReportSqeProfiling_(entry.sqeArrayInfos_[0].sqeArray, 0, 0, baseAddrs, memSizes, 2, &streamLite, 0),
        HCCL_SUCCESS);

    const u16 writtenIdx = (streamLite.GetTaskInfos()->GetEnd() + streamLite.GetTaskInfos()->GetCapacity() - 1U)
                           % streamLite.GetTaskInfos()->GetCapacity();
    const Hccl::DfxTaskInfo* slot = streamLite.GetTaskInfos()->GetSlot(writtenIdx);
    ASSERT_NE(slot, nullptr);
    EXPECT_EQ(slot->channelHandle, TEST_NON_DEFAULT_CHANNEL_HANDLE);
    EXPECT_EQ(slot->linkType, TEST_NON_DEFAULT_LINK_TYPE);
    EXPECT_EQ(slot->transportType, TEST_NON_DEFAULT_TRANSPORT_TYPE);
}
