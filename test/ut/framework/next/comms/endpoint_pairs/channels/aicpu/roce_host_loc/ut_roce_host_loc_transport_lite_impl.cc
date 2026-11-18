/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <chrono>
#include <iostream>
#include <thread>
#include "dfx/device/dfx_res_pub.h"
#include "gtest/gtest.h"
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#include "roce_host_loc_transport_lite_impl.h"
#include "rma_buffer_lite.h"
#include "rmt_rma_buffer_lite.h"
#include "notify_lite.h"
#include "binary_stream.h"
#include "stream.h"
#include "stream_lite.h"
#include "rtsq_base.h"
#include "sqe.h"
#include "roce_host_loc_conn_lite.h"

#define private public

using namespace Hccl;

static const u32 ROCE_TYPE = 1;
static std::vector<u64> g_writeReduceSliceSizes;
static std::vector<std::pair<u32, u32>> g_notifyWaitCalls;
static std::vector<int32_t> g_pollCqTimeouts;
static std::vector<std::pair<u64, u64>> g_rwLocSlices;
static std::vector<std::pair<u64, u64>> g_rwRmtSlices;

static HcclResult CapturePollCqTimeout(
    RoceHostLocConnLite* conn, int32_t numEntries, int32_t timeOut, std::vector<int32_t>& errList, u64& dbAddr,
    u64& dbValue)
{
    (void)conn;
    (void)numEntries;
    (void)errList;
    g_pollCqTimeouts.push_back(timeOut);
    if (g_pollCqTimeouts.size() == 1) {
        // 模拟首个连接耗光总超时预算，使后续连接拿到 0
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    dbAddr = 0;
    dbValue = 0;
    return HCCL_SUCCESS;
}

static std::vector<HcclResult> g_pollCqReturns;
static size_t g_pollCqReturnIdx = 0;
static HcclResult ReturnSequencedPollCqResult(
    RoceHostLocConnLite* conn, int32_t numEntries, int32_t timeOut, std::vector<int32_t>& errList, u64& dbAddr,
    u64& dbValue)
{
    (void)conn;
    (void)numEntries;
    (void)timeOut;
    (void)errList;
    dbAddr = 0;
    dbValue = 0;
    if (g_pollCqReturnIdx >= g_pollCqReturns.size()) {
        return HCCL_SUCCESS;
    }
    return g_pollCqReturns[g_pollCqReturnIdx++];
}

static HcclResult CaptureReadWriteSlice(
    RoceHostLocConnLite* conn, const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, const SqeConfigLite& cfg,
    u64& dbAddr, u64& dbValue)
{
    (void)conn;
    (void)cfg;
    g_rwLocSlices.emplace_back(loc.GetAddr(), loc.GetSize());
    g_rwRmtSlices.emplace_back(rmt.GetAddr(), rmt.GetSize());
    dbAddr = 0;
    dbValue = 0;
    return HCCL_SUCCESS;
}

static HcclResult CaptureWriteReduceSlice(
    RoceHostLocConnLite* conn, const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, const SqeConfigLite& cfg,
    DataType dataType, ReduceOp reduceOp, u64& dbAddr, u64& dbValue)
{
    (void)conn;
    (void)rmt;
    (void)cfg;
    (void)dataType;
    (void)reduceOp;
    g_writeReduceSliceSizes.push_back(loc.GetSize());
    dbAddr = 0;
    dbValue = 0;
    return HCCL_SUCCESS;
}

static void
CaptureNotifyWait(RoceHostLocTransportLiteImpl* transport, u32 notifyId, const StreamLite& stream, u32 timeout)
{
    (void)transport;
    (void)stream;
    g_notifyWaitCalls.emplace_back(notifyId, timeout);
}

static std::vector<char> BuildSingleRmaBufferUniqueId(u64 addr, u64 size, u32 key)
{
    BinaryStream bs;
    bs << addr;
    bs << size;
    bs << key;
    std::vector<char> result;
    bs.Dump(result);
    return result;
}

static std::vector<char> BuildLocNotifyUniqueIds(u32 notifyNum)
{
    std::vector<char> result;
    for (u32 i = 0; i < notifyNum; i++) {
        u32 notifyId = i + 1;
        u32 devPhyId = i + 10;
        result.insert(
            result.end(), reinterpret_cast<char*>(&notifyId), reinterpret_cast<char*>(&notifyId) + sizeof(notifyId));
        result.insert(
            result.end(), reinterpret_cast<char*>(&devPhyId), reinterpret_cast<char*>(&devPhyId) + sizeof(devPhyId));
    }
    return result;
}

static std::vector<char> BuildRmtNotifyUniqueIds(u32 notifyNum)
{
    std::vector<char> result;
    for (u32 i = 0; i < notifyNum; i++) {
        u64 addr = 0x2000 + i * 0x100;
        u64 size = 0x100;
        u32 rkey = 0x200 + i;
        result.insert(result.end(), reinterpret_cast<char*>(&addr), reinterpret_cast<char*>(&addr) + sizeof(addr));
        result.insert(result.end(), reinterpret_cast<char*>(&size), reinterpret_cast<char*>(&size) + sizeof(size));
        result.insert(result.end(), reinterpret_cast<char*>(&rkey), reinterpret_cast<char*>(&rkey) + sizeof(rkey));
    }
    return result;
}

static std::vector<char> BuildNotifyValueBufferUniqueId() { return BuildSingleRmaBufferUniqueId(0x3000, 0x200, 0x300); }

static std::vector<char> BuildLocBufferUniqueIds(u32 bufferNum)
{
    std::vector<char> result;
    for (u32 i = 0; i < bufferNum; i++) {
        u64 addr = 0x4000 + i * 0x200;
        u64 size = 0x200;
        u32 lkey = 0x400 + i;
        result.insert(result.end(), reinterpret_cast<char*>(&addr), reinterpret_cast<char*>(&addr) + sizeof(addr));
        result.insert(result.end(), reinterpret_cast<char*>(&size), reinterpret_cast<char*>(&size) + sizeof(size));
        result.insert(result.end(), reinterpret_cast<char*>(&lkey), reinterpret_cast<char*>(&lkey) + sizeof(lkey));
    }
    return result;
}

static std::vector<char> BuildRmtBufferUniqueIds(u32 bufferNum)
{
    std::vector<char> result;
    for (u32 i = 0; i < bufferNum; i++) {
        u64 addr = 0x5000 + i * 0x200;
        u64 size = 0x200;
        u32 rkey = 0x600 + i;
        result.insert(result.end(), reinterpret_cast<char*>(&addr), reinterpret_cast<char*>(&addr) + sizeof(addr));
        result.insert(result.end(), reinterpret_cast<char*>(&size), reinterpret_cast<char*>(&size) + sizeof(size));
        result.insert(result.end(), reinterpret_cast<char*>(&rkey), reinterpret_cast<char*>(&rkey) + sizeof(rkey));
    }
    return result;
}

static std::vector<char> BuildSqUniqueId(const RdmaSqContextLite& sqCtx)
{
    BinaryStream bs;
    bs << sqCtx.qpn;
    bs << sqCtx.sqVa;
    bs << sqCtx.wqeSize;
    bs << sqCtx.depth;
    bs << sqCtx.headAddr;
    bs << sqCtx.tailAddr;
    bs << sqCtx.dbHwVa;
    bs << sqCtx.dbSwVa;
    bs << sqCtx.sl;
    bs << sqCtx.dbVendorSpecified;
    std::vector<char> result;
    bs.Dump(result);
    return result;
}

static std::vector<char> BuildCqUniqueId(const RdmaCqContextLite& cqCtx)
{
    BinaryStream bs;
    bs << cqCtx.cqn;
    bs << cqCtx.cqVa;
    bs << cqCtx.cqeSize;
    bs << cqCtx.cqDepth;
    bs << cqCtx.headAddr;
    bs << cqCtx.tailAddr;
    bs << cqCtx.dbSwVa;
    bs << cqCtx.dbVendorSpecified;
    std::vector<char> result;
    bs.Dump(result);
    return result;
}

static std::vector<char>
BuildSingleConnUniqueId(u32 dmaMode, const RdmaSqContextLite& sqCtx, const RdmaCqContextLite& cqCtx)
{
    BinaryStream bs;
    bs << dmaMode;

    std::vector<char> sqUniqueId = BuildSqUniqueId(sqCtx);
    bs << sqUniqueId;

    std::vector<char> cqUniqueId = BuildCqUniqueId(cqCtx);
    bs << cqUniqueId;

    std::vector<char> result;
    bs.Dump(result);
    return result;
}

static std::vector<char> BuildConnUniqueIds(u32 connNum)
{
    std::vector<char> result;
    for (u32 i = 0; i < connNum; i++) {
        RdmaSqContextLite sq{};
        sq.qpn = 1 + i;
        sq.sqVa = 0x10000;
        sq.wqeSize = 64;
        sq.depth = 128;
        sq.headAddr = 0x20000;
        sq.tailAddr = 0x20008;
        sq.sl = 0;
        sq.dbHwVa = 0x30000;
        sq.dbSwVa = 0x70000;
        sq.dbVendorSpecified = 0;
        sq.dbVendorSpecified
            |= (static_cast<uint64_t>(static_cast<uint8_t>(3) & UB_DB_VENDOR_FIELD_MASK)
                << UB_DB_VENDOR_MTUSHIFT_SHIFT);
        sq.dbVendorSpecified
            |= (static_cast<uint64_t>(static_cast<uint8_t>(0x7) & UB_DB_VENDOR_FIELD_MASK) << UB_DB_VENDOR_COS_SHIFT);

        RdmaCqContextLite cq{};
        cq.cqn = 2 + i;
        cq.cqVa = 0x40000;
        cq.cqeSize = 32;
        cq.cqDepth = 256;
        cq.headAddr = 0x50000;
        cq.tailAddr = 0x50008;
        cq.dbSwVa = 0x60000;

        std::vector<char> singleConn = BuildSingleConnUniqueId(1, sq, cq);
        result.insert(result.end(), singleConn.begin(), singleConn.end());
    }
    return result;
}

static std::vector<char> BuildTransportUniqueId(u32 notifyNum, u32 bufferNum, u32 connNum, u32 qpThreshold = 0)
{
    BinaryStream bs;
    u32 type = ROCE_TYPE;
    bs << type;
    bs << notifyNum;
    bs << bufferNum;
    bs << connNum;
    bs << qpThreshold;

    std::vector<char> locNotifyIds = BuildLocNotifyUniqueIds(notifyNum);
    bs << locNotifyIds;

    std::vector<char> rmtNotifyIds = BuildRmtNotifyUniqueIds(notifyNum);
    bs << rmtNotifyIds;

    std::vector<char> notifyValueId = BuildNotifyValueBufferUniqueId();
    bs << notifyValueId;

    std::vector<char> locBufferIds = BuildLocBufferUniqueIds(bufferNum);
    bs << locBufferIds;

    std::vector<char> rmtBufferIds = BuildRmtBufferUniqueIds(bufferNum);
    bs << rmtBufferIds;

    std::vector<char> connIds = BuildConnUniqueIds(connNum);
    bs << connIds;

    std::vector<char> result;
    bs.Dump(result);
    return result;
}

class RoceHostLocTransportLiteImplTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RoceHostLocTransportLiteImplTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "RoceHostLocTransportLiteImplTest tests tear down." << std::endl; }

    virtual void SetUp()
    {
        std::cout << "A Test case in RoceHostLocTransportLiteImplTest SetUP" << std::endl;
        MOCKER_CPP(&RtsqBase::QuerySqBaseAddr)
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(reinterpret_cast<u64>(&mockSq)));
        MOCKER_CPP(&RtsqBase::QuerySqDepth)
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(static_cast<u32>(AC_SQE_MAX_CNT)));
        MOCKER_CPP(&RtsqBase::QuerySqStatusByType).stubs().with(mockcpp::any()).will(returnValue(static_cast<u32>(0)));
        MOCKER_CPP(&RtsqBase::ConfigSqStatusByType).stubs();
        uniqueId_ = BuildTransportUniqueId(NOTIFY_NUM, BUFFER_NUM, CONN_NUM);
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in RoceHostLocTransportLiteImplTest TearDown" << std::endl;
    }

    u32 NOTIFY_NUM = 3;
    u32 BUFFER_NUM = 1;
    u32 CONN_NUM = 1;
    std::vector<char> uniqueId_;
    u8 mockSq[AC_SQE_SIZE * AC_SQE_MAX_CNT]{0};
};

TEST_F(RoceHostLocTransportLiteImplTest, Ut_When_Construct_Expect_Success)
{
    std::cout << "Start Ut_When_Construct_Expect_Success" << std::endl;

    RoceHostLocTransportLiteImpl transport(uniqueId_);

    std::cout << "End Ut_When_Construct_Expect_Success" << std::endl;
}

TEST_F(RoceHostLocTransportLiteImplTest, Ut_When_Describe_Expect_NotEmpty)
{
    std::cout << "Start Ut_When_Describe_Expect_NotEmpty" << std::endl;

    RoceHostLocTransportLiteImpl transport(uniqueId_);

    std::string desc = transport.Describe();
    EXPECT_FALSE(desc.empty());

    std::cout << "End Ut_When_Describe_Expect_NotEmpty" << std::endl;
}

TEST_F(RoceHostLocTransportLiteImplTest, Ut_When_NotifyNum_Expect_Correct)
{
    std::cout << "Start Ut_When_NotifyNum_Expect_Correct" << std::endl;

    RoceHostLocTransportLiteImpl transport(uniqueId_);

    EXPECT_EQ(transport.notifyNum_, NOTIFY_NUM);

    std::cout << "End Ut_When_NotifyNum_Expect_Correct" << std::endl;
}

TEST_F(RoceHostLocTransportLiteImplTest, Ut_When_BufferNum_Expect_Correct)
{
    std::cout << "Start Ut_When_BufferNum_Expect_Correct" << std::endl;

    RoceHostLocTransportLiteImpl transport(uniqueId_);

    EXPECT_EQ(transport.bufferNum_, BUFFER_NUM);

    std::cout << "End Ut_When_BufferNum_Expect_Correct" << std::endl;
}

TEST_F(RoceHostLocTransportLiteImplTest, Ut_When_ConnNum_Expect_Correct)
{
    std::cout << "Start Ut_When_ConnNum_Expect_Correct" << std::endl;

    RoceHostLocTransportLiteImpl transport(uniqueId_);

    EXPECT_EQ(transport.connNum_, CONN_NUM);

    std::cout << "End Ut_When_ConnNum_Expect_Correct" << std::endl;
}

TEST_F(RoceHostLocTransportLiteImplTest, Ut_When_LocalNotifies_Expect_NotEmpty)
{
    std::cout << "Start Ut_When_LocalNotifies_Expect_NotEmpty" << std::endl;

    RoceHostLocTransportLiteImpl transport(uniqueId_);

    EXPECT_EQ(transport.localNotifies_.size(), static_cast<size_t>(NOTIFY_NUM));

    std::cout << "End Ut_When_LocalNotifies_Expect_NotEmpty" << std::endl;
}

TEST_F(RoceHostLocTransportLiteImplTest, Ut_When_RemoteNotifies_Expect_NotEmpty)
{
    std::cout << "Start Ut_When_RemoteNotifies_Expect_NotEmpty" << std::endl;

    RoceHostLocTransportLiteImpl transport(uniqueId_);

    EXPECT_EQ(transport.remoteNotifies_.size(), static_cast<size_t>(NOTIFY_NUM));

    std::cout << "End Ut_When_RemoteNotifies_Expect_NotEmpty" << std::endl;
}

TEST_F(RoceHostLocTransportLiteImplTest, Ut_When_LocBufferVec_Expect_NotEmpty)
{
    std::cout << "Start Ut_When_LocBufferVec_Expect_NotEmpty" << std::endl;

    RoceHostLocTransportLiteImpl transport(uniqueId_);

    EXPECT_EQ(transport.locBufferVec_.size(), static_cast<size_t>(BUFFER_NUM));

    std::cout << "End Ut_When_LocBufferVec_Expect_NotEmpty" << std::endl;
}

TEST_F(RoceHostLocTransportLiteImplTest, Ut_When_RmtBufferVec_Expect_NotEmpty)
{
    std::cout << "Start Ut_When_RmtBufferVec_Expect_NotEmpty" << std::endl;

    RoceHostLocTransportLiteImpl transport(uniqueId_);

    EXPECT_EQ(transport.rmtBufferVec_.size(), static_cast<size_t>(BUFFER_NUM));

    std::cout << "End Ut_When_RmtBufferVec_Expect_NotEmpty" << std::endl;
}

TEST_F(RoceHostLocTransportLiteImplTest, Ut_When_ConnVec_Expect_NotEmpty)
{
    std::cout << "Start Ut_When_ConnVec_Expect_NotEmpty" << std::endl;

    RoceHostLocTransportLiteImpl transport(uniqueId_);

    EXPECT_EQ(transport.connVec_.size(), static_cast<size_t>(CONN_NUM));

    std::cout << "End Ut_When_ConnVec_Expect_NotEmpty" << std::endl;
}

TEST_F(RoceHostLocTransportLiteImplTest, Ut_When_ConnUniqueIdVec_Expect_NotEmpty)
{
    std::cout << "Start Ut_When_ConnUniqueIdVec_Expect_NotEmpty" << std::endl;

    RoceHostLocTransportLiteImpl transport(uniqueId_);

    EXPECT_EQ(transport.connUniqueIdVec_.size(), static_cast<size_t>(CONN_NUM));

    std::cout << "End Ut_When_ConnUniqueIdVec_Expect_NotEmpty" << std::endl;
}

TEST_F(RoceHostLocTransportLiteImplTest, Ut_When_NotifyValueBuffer_Expect_NotNull)
{
    std::cout << "Start Ut_When_NotifyValueBuffer_Expect_NotNull" << std::endl;

    RoceHostLocTransportLiteImpl transport(uniqueId_);

    EXPECT_NE(transport.notifyValueBuffer_, nullptr);

    std::cout << "End Ut_When_NotifyValueBuffer_Expect_NotNull" << std::endl;
}

TEST_F(RoceHostLocTransportLiteImplTest, Ut_When_LocalNotifyValues_Expect_Correct)
{
    std::cout << "Start Ut_When_LocalNotifyValues_Expect_Correct" << std::endl;

    RoceHostLocTransportLiteImpl transport(uniqueId_);

    ASSERT_EQ(transport.localNotifies_.size(), static_cast<size_t>(NOTIFY_NUM));
    for (u32 i = 0; i < NOTIFY_NUM; i++) {
        EXPECT_EQ(transport.localNotifies_[i]->GetId(), i + 1);
        EXPECT_EQ(transport.localNotifies_[i]->GetDevPhyId(), i + 10);
    }

    std::cout << "End Ut_When_LocalNotifyValues_Expect_Correct" << std::endl;
}

TEST_F(RoceHostLocTransportLiteImplTest, Ut_When_RemoteNotifyValues_Expect_Correct)
{
    std::cout << "Start Ut_When_RemoteNotifyValues_Expect_Correct" << std::endl;

    RoceHostLocTransportLiteImpl transport(uniqueId_);

    ASSERT_EQ(transport.remoteNotifies_.size(), static_cast<size_t>(NOTIFY_NUM));
    for (u32 i = 0; i < NOTIFY_NUM; i++) {
        EXPECT_EQ(transport.remoteNotifies_[i].GetAddr(), 0x2000 + i * 0x100);
        EXPECT_EQ(transport.remoteNotifies_[i].GetSize(), 0x100u);
        EXPECT_EQ(transport.remoteNotifies_[i].GetRkey(), 0x200 + i);
    }

    std::cout << "End Ut_When_RemoteNotifyValues_Expect_Correct" << std::endl;
}

TEST_F(RoceHostLocTransportLiteImplTest, Ut_When_NotifyValueBufferValues_Expect_Correct)
{
    std::cout << "Start Ut_When_NotifyValueBufferValues_Expect_Correct" << std::endl;

    RoceHostLocTransportLiteImpl transport(uniqueId_);

    ASSERT_NE(transport.notifyValueBuffer_, nullptr);
    EXPECT_EQ(transport.notifyValueBuffer_->GetAddr(), 0x3000u);
    EXPECT_EQ(transport.notifyValueBuffer_->GetSize(), 0x200u);
    EXPECT_EQ(transport.notifyValueBuffer_->GetLkey(), 0x300u);

    std::cout << "End Ut_When_NotifyValueBufferValues_Expect_Correct" << std::endl;
}

TEST_F(RoceHostLocTransportLiteImplTest, Ut_When_LocBufferValues_Expect_Correct)
{
    std::cout << "Start Ut_When_LocBufferValues_Expect_Correct" << std::endl;

    RoceHostLocTransportLiteImpl transport(uniqueId_);

    ASSERT_EQ(transport.locBufferVec_.size(), static_cast<size_t>(BUFFER_NUM));
    for (u32 i = 0; i < BUFFER_NUM; i++) {
        EXPECT_EQ(transport.locBufferVec_[i].GetAddr(), 0x4000 + i * 0x200);
        EXPECT_EQ(transport.locBufferVec_[i].GetSize(), 0x200u);
        EXPECT_EQ(transport.locBufferVec_[i].GetLkey(), 0x400 + i);
    }

    std::cout << "End Ut_When_LocBufferValues_Expect_Correct" << std::endl;
}

TEST_F(RoceHostLocTransportLiteImplTest, Ut_When_RmtBufferValues_Expect_Correct)
{
    std::cout << "Start Ut_When_RmtBufferValues_Expect_Correct" << std::endl;

    RoceHostLocTransportLiteImpl transport(uniqueId_);

    ASSERT_EQ(transport.rmtBufferVec_.size(), static_cast<size_t>(BUFFER_NUM));
    for (u32 i = 0; i < BUFFER_NUM; i++) {
        EXPECT_EQ(transport.rmtBufferVec_[i].GetAddr(), 0x5000 + i * 0x200);
        EXPECT_EQ(transport.rmtBufferVec_[i].GetSize(), 0x200u);
        EXPECT_EQ(transport.rmtBufferVec_[i].GetRkey(), 0x600 + i);
    }

    std::cout << "End Ut_When_RmtBufferValues_Expect_Correct" << std::endl;
}

TEST_F(RoceHostLocTransportLiteImplTest, Ut_When_ConnVecValues_Expect_Correct)
{
    std::cout << "Start Ut_When_ConnVecValues_Expect_Correct" << std::endl;

    RoceHostLocTransportLiteImpl transport(uniqueId_);

    ASSERT_EQ(transport.connVec_.size(), static_cast<size_t>(CONN_NUM));
    for (u32 i = 0; i < CONN_NUM; i++) {
        EXPECT_NE(transport.connVec_[i], nullptr);
        EXPECT_EQ(transport.connVec_[i]->dmaMode_, 1u);
        EXPECT_EQ(transport.connVec_[i]->sqContext.qpn, 1u + i);
        EXPECT_EQ(transport.connVec_[i]->cqContext.cqn, 2u + i);
    }

    std::cout << "End Ut_When_ConnVecValues_Expect_Correct" << std::endl;
}

TEST_F(RoceHostLocTransportLiteImplTest, Ut_RoceTransportLite_Write)
{
    std::cout << "Start Ut_RoceTransportLite_Write" << std::endl;

    RoceHostLocTransportLiteImpl transport(uniqueId_);

    ASSERT_EQ(transport.connVec_.size(), static_cast<size_t>(CONN_NUM));

    RmaBufferLite locBuf(0x4000, 2048, 0x400, 0);
    Hccl::Buffer rmtBuf{0x5000, 0x200};

    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(static_cast<HcclResult (RoceHostLocConnLite::*)(
                   const RmaBufSliceLite&, const RmtRmaBufSliceLite&, const SqeConfigLite&, unsigned long long&,
                   unsigned long long&)>(&RoceHostLocConnLite::Write))
        .expects(once())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&RoceHostLocConnLite::PollCq)
        .expects(once())
        .with(eq(128), eq(0), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));

    EXPECT_EQ(transport.Write(locBuf, rmtBuf, stream), HCCL_SUCCESS);
    TaskInfoCircularQueue* queue = stream.GetTaskInfos();
    ASSERT_NE(queue, nullptr);
    ASSERT_FALSE(queue->IsEmpty());
    DfxTaskInfo* slot = queue->GetSlot(0);
    ASSERT_NE(slot, nullptr);
    EXPECT_EQ(slot->taskType, static_cast<u8>(TaskParamTypeVal::TASK_RDMA));
    EXPECT_EQ(slot->linkType, static_cast<u8>(DfxLinkTypeVal::LINK_ROCE));
    EXPECT_EQ(slot->transportType, static_cast<u8>(DfxTransportType::DFX_TRANSPORT_TYPE_ROCE));

    std::cout << "End Ut_RoceTransportLite_Write" << std::endl;
}

TEST_F(RoceHostLocTransportLiteImplTest, Ut_RoceTransportLite_WriteWithNotify)
{
    std::cout << "Start Ut_RoceTransportLite_WriteWithNotify" << std::endl;

    RoceHostLocTransportLiteImpl transport(uniqueId_);

    ASSERT_EQ(transport.connVec_.size(), static_cast<size_t>(CONN_NUM));

    RmaBufferLite locBuf(0x4000, 2048, 0x400, 0);
    Hccl::Buffer rmtBuf{0x5000, 0x200};
    uint32_t remoteNotifyIdx = 0;
    Hccl::WithNotifyIn withNotify{Hccl::TransportNotifyType::NORMAL, remoteNotifyIdx};

    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(
        static_cast<HcclResult (RoceHostLocConnLite::*)(
            const RmaBufSliceLite&, const RmtRmaBufSliceLite&, const RmaBufSliceLite&, const RmtRmaBufSliceLite&,
            const SqeConfigLite&, unsigned long long&, unsigned long long&)>(&RoceHostLocConnLite::WriteWithNotify))
        .expects(once())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&RoceHostLocConnLite::PollCq)
        .expects(once())
        .with(eq(128), eq(0), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));

    EXPECT_EQ(transport.WriteWithNotify(locBuf, rmtBuf, withNotify, stream), HCCL_SUCCESS);
    TaskInfoCircularQueue* queue = stream.GetTaskInfos();
    ASSERT_NE(queue, nullptr);
    ASSERT_FALSE(queue->IsEmpty());
    DfxTaskInfo* slot = queue->GetSlot(0);
    ASSERT_NE(slot, nullptr);
    EXPECT_EQ(slot->taskType, static_cast<u8>(TaskParamTypeVal::TASK_WRITE_WITH_NOTIFY));
    EXPECT_EQ(slot->linkType, static_cast<u8>(DfxLinkTypeVal::LINK_ROCE));
    EXPECT_EQ(slot->transportType, static_cast<u8>(DfxTransportType::DFX_TRANSPORT_TYPE_ROCE));

    std::cout << "End Ut_RoceTransportLite_WriteWithNotify" << std::endl;
}

TEST_F(RoceHostLocTransportLiteImplTest, Ut_RoceTransportLite_Post)
{
    std::cout << "Start Ut_RoceTransportLite_Post" << std::endl;

    RoceHostLocTransportLiteImpl transport(uniqueId_);

    ASSERT_EQ(transport.connVec_.size(), static_cast<size_t>(CONN_NUM));

    uint32_t remoteNotifyIdx = 0;

    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(static_cast<HcclResult (RoceHostLocConnLite::*)(
                   const RmaBufSliceLite&, const RmtRmaBufSliceLite&, const SqeConfigLite&, unsigned long long&,
                   unsigned long long&)>(&RoceHostLocConnLite::Write))
        .expects(once())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&RoceHostLocConnLite::PollCq)
        .expects(once())
        .with(eq(128), eq(0), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));

    EXPECT_EQ(transport.Post(remoteNotifyIdx, stream), HCCL_SUCCESS);
    TaskInfoCircularQueue* queue = stream.GetTaskInfos();
    ASSERT_NE(queue, nullptr);
    ASSERT_FALSE(queue->IsEmpty());
    DfxTaskInfo* slot = queue->GetSlot(0);
    ASSERT_NE(slot, nullptr);
    EXPECT_EQ(slot->taskType, static_cast<u8>(TaskParamTypeVal::TASK_RDMA));
    EXPECT_EQ(slot->linkType, static_cast<u8>(DfxLinkTypeVal::LINK_ROCE));
    EXPECT_EQ(slot->transportType, static_cast<u8>(DfxTransportType::DFX_TRANSPORT_TYPE_ROCE));

    std::cout << "End Ut_RoceTransportLite_Post" << std::endl;
}

TEST_F(RoceHostLocTransportLiteImplTest, Ut_RoceTransportLite_Fence)
{
    std::cout << "Start Ut_RoceTransportLite_Fence" << std::endl;

    RoceHostLocTransportLiteImpl transport(uniqueId_);

    EXPECT_FALSE(transport.fence_);
    EXPECT_EQ(transport.Fence(), HCCL_SUCCESS);
    EXPECT_TRUE(transport.fence_);

    std::cout << "End Ut_RoceTransportLite_Fence" << std::endl;
}

TEST_F(RoceHostLocTransportLiteImplTest, Ut_RoceTransportLite_Read)
{
    RoceHostLocTransportLiteImpl transport(uniqueId_);
    RmaBufferLite locBuf(0x4000, 0x200, 0x400, 0);
    Hccl::Buffer rmtBuf{0x5000, 0x200};
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);

    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(static_cast<HcclResult (RoceHostLocConnLite::*)(
                   const RmaBufSliceLite&, const RmtRmaBufSliceLite&, const SqeConfigLite&, unsigned long long&,
                   unsigned long long&)>(&RoceHostLocConnLite::Read))
        .expects(once())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&RoceHostLocConnLite::PollCq)
        .expects(once())
        .with(eq(128), eq(0), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));

    EXPECT_EQ(transport.Read(locBuf, rmtBuf, stream), HCCL_SUCCESS);
    TaskInfoCircularQueue* queue = stream.GetTaskInfos();
    ASSERT_NE(queue, nullptr);
    ASSERT_FALSE(queue->IsEmpty());
    DfxTaskInfo* slot = queue->GetSlot(0);
    ASSERT_NE(slot, nullptr);
    EXPECT_EQ(slot->taskType, static_cast<u8>(TaskParamTypeVal::TASK_RDMA));
    EXPECT_EQ(slot->linkType, static_cast<u8>(DfxLinkTypeVal::LINK_ROCE));
    EXPECT_EQ(slot->transportType, static_cast<u8>(DfxTransportType::DFX_TRANSPORT_TYPE_ROCE));
}

TEST_F(RoceHostLocTransportLiteImplTest, Ut_RoceTransportLite_WriteReduce)
{
    RoceHostLocTransportLiteImpl transport(uniqueId_);
    RmaBufferLite locBuf(0x4000, 0x200, 0x400, 0);
    Hccl::Buffer rmtBuf{0x5000, 0x200};
    ReduceIn reduceIn(DataType::FP32, ReduceOp::SUM);
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);

    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(static_cast<HcclResult (RoceHostLocConnLite::*)(
                   const RmaBufSliceLite&, const RmtRmaBufSliceLite&, const SqeConfigLite&, DataType, ReduceOp,
                   unsigned long long&, unsigned long long&)>(&RoceHostLocConnLite::WriteReduce))
        .expects(once())
        .with(
            mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(),
            mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&RoceHostLocConnLite::PollCq)
        .expects(once())
        .with(eq(128), eq(0), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));

    EXPECT_EQ(transport.WriteReduce(locBuf, rmtBuf, reduceIn, stream), HCCL_SUCCESS);
    TaskInfoCircularQueue* queue = stream.GetTaskInfos();
    ASSERT_NE(queue, nullptr);
    ASSERT_FALSE(queue->IsEmpty());
    DfxTaskInfo* slot = queue->GetSlot(0);
    ASSERT_NE(slot, nullptr);
    EXPECT_EQ(slot->taskType, static_cast<u8>(TaskParamTypeVal::TASK_REDUCE_INLINE));
    EXPECT_EQ(slot->linkType, static_cast<u8>(DfxLinkTypeVal::LINK_ROCE));
    EXPECT_EQ(slot->transportType, static_cast<u8>(DfxTransportType::DFX_TRANSPORT_TYPE_ROCE));
    EXPECT_EQ(slot->taskPara.Reduce.srcAddr, 0x4000u);
    EXPECT_EQ(slot->taskPara.Reduce.dstAddr, 0x5000u);
    EXPECT_EQ(slot->taskPara.Reduce.size, 0x200u);
}

TEST_F(RoceHostLocTransportLiteImplTest, Ut_RoceTransportLite_WriteReduceWithNotify)
{
    RoceHostLocTransportLiteImpl transport(uniqueId_);
    RmaBufferLite locBuf(0x4000, 0x200, 0x400, 0);
    Hccl::Buffer rmtBuf{0x5000, 0x200};
    ReduceIn reduceIn(DataType::FP32, ReduceOp::MAX);
    Hccl::WithNotifyIn withNotify{Hccl::TransportNotifyType::NORMAL, 0};
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);

    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(static_cast<HcclResult (RoceHostLocConnLite::*)(
                   const RmaBufSliceLite&, const RmtRmaBufSliceLite&, const RmaBufSliceLite&, const RmtRmaBufSliceLite&,
                   const SqeConfigLite&, DataType, ReduceOp, unsigned long long&, unsigned long long&)>(
                   &RoceHostLocConnLite::WriteReduceWithNotify))
        .expects(once())
        .with(
            mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(),
            mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&RoceHostLocConnLite::PollCq)
        .expects(once())
        .with(eq(128), eq(0), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));

    EXPECT_EQ(transport.WriteReduceWithNotify(locBuf, rmtBuf, reduceIn, withNotify, stream), HCCL_SUCCESS);
    TaskInfoCircularQueue* queue = stream.GetTaskInfos();
    ASSERT_NE(queue, nullptr);
    ASSERT_FALSE(queue->IsEmpty());
    DfxTaskInfo* slot = queue->GetSlot(0);
    ASSERT_NE(slot, nullptr);
    EXPECT_EQ(slot->taskType, static_cast<u8>(TaskParamTypeVal::TASK_WRITE_REDUCE_WITH_NOTIFY));
    EXPECT_EQ(slot->linkType, static_cast<u8>(DfxLinkTypeVal::LINK_ROCE));
    EXPECT_EQ(slot->transportType, static_cast<u8>(DfxTransportType::DFX_TRANSPORT_TYPE_ROCE));
    EXPECT_EQ(slot->taskPara.Reduce.srcAddr, 0x4000u);
    EXPECT_EQ(slot->taskPara.Reduce.dstAddr, 0x5000u);
    EXPECT_EQ(slot->taskPara.Reduce.size, 0x200u);
}

TEST_F(RoceHostLocTransportLiteImplTest, Ut_RoceTransportLite_PollCq_Expect_ResultPropagated)
{
    RoceHostLocTransportLiteImpl transport(uniqueId_);
    std::vector<int32_t> errList;

    MOCKER_CPP(&RoceHostLocConnLite::PollCq)
        .stubs()
        .with(eq(1), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_REMOTE));
    EXPECT_EQ(transport.PollCq(1, 5, errList), HCCL_E_REMOTE);
}

TEST_F(RoceHostLocTransportLiteImplTest, Ut_PollCq_MultiConn_Expect_SharedTimeoutBudget)
{
    auto multiConnId = BuildTransportUniqueId(6, BUFFER_NUM, 2, 0);
    RoceHostLocTransportLiteImpl transport(multiConnId);
    std::vector<int32_t> errList;
    g_pollCqTimeouts.clear();

    MOCKER_CPP(&RoceHostLocConnLite::PollCq)
        .expects(exactly(2))
        .with(eq(1), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(CapturePollCqTimeout));

    EXPECT_EQ(transport.PollCq(1, 1, errList), HCCL_SUCCESS);
    ASSERT_EQ(g_pollCqTimeouts.size(), 2);
    EXPECT_LE(g_pollCqTimeouts[0], 1);
    // 首个连接已耗尽总预算，后续连接必须传 0 做非阻塞轮询
    EXPECT_EQ(g_pollCqTimeouts[1], 0);
}

TEST_F(RoceHostLocTransportLiteImplTest, Ut_PollCq_MultiConn_AllTimeout_Expect_Timeout)
{
    auto multiConnId = BuildTransportUniqueId(6, BUFFER_NUM, 2, 0);
    RoceHostLocTransportLiteImpl transport(multiConnId);
    std::vector<int32_t> errList;

    MOCKER_CPP(&RoceHostLocConnLite::PollCq).expects(exactly(2)).will(returnValue(HCCL_E_TIMEOUT));

    EXPECT_EQ(transport.PollCq(1, 0, errList), HCCL_E_TIMEOUT);
}

TEST_F(RoceHostLocTransportLiteImplTest, Ut_PollCq_MultiConn_TimeoutAndError_Expect_FirstError)
{
    auto multiConnId = BuildTransportUniqueId(6, BUFFER_NUM, 2, 0);
    RoceHostLocTransportLiteImpl transport(multiConnId);
    std::vector<int32_t> errList;
    g_pollCqReturns = {HCCL_E_TIMEOUT, HCCL_E_REMOTE};
    g_pollCqReturnIdx = 0;

    MOCKER_CPP(&RoceHostLocConnLite::PollCq).expects(exactly(2)).will(invoke(ReturnSequencedPollCqResult));

    EXPECT_EQ(transport.PollCq(1, 0, errList), HCCL_E_REMOTE);
}

TEST_F(RoceHostLocTransportLiteImplTest, Ut_RoceTransportLite_WaitWithTimeout)
{
    RoceHostLocTransportLiteImpl transport(uniqueId_);
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);

    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));

    EXPECT_NO_THROW(transport.WaitWithTimeout(0, stream, 1800));
}

TEST_F(RoceHostLocTransportLiteImplTest, Ut_RoceTransportLite_Write_PollCqError_Expect_ErrorPropagated)
{
    RoceHostLocTransportLiteImpl transport(uniqueId_);
    RmaBufferLite locBuf(0x4000, 0x200, 0x400, 0);
    Hccl::Buffer rmtBuf{0x5000, 0x200};
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);

    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(static_cast<HcclResult (RoceHostLocConnLite::*)(
                   const RmaBufSliceLite&, const RmtRmaBufSliceLite&, const SqeConfigLite&, unsigned long long&,
                   unsigned long long&)>(&RoceHostLocConnLite::Write))
        .expects(once())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&RoceHostLocConnLite::PollCq)
        .expects(once())
        .with(eq(128), eq(0), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_REMOTE));

    EXPECT_EQ(transport.Write(locBuf, rmtBuf, stream), HCCL_E_REMOTE);
}

TEST_F(RoceHostLocTransportLiteImplTest, Ut_RoceTransportLite_Write_PollCqTimeout_Expect_Success)
{
    RoceHostLocTransportLiteImpl transport(uniqueId_);
    RmaBufferLite locBuf(0x4000, 0x200, 0x400, 0);
    Hccl::Buffer rmtBuf{0x5000, 0x200};
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);

    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(static_cast<HcclResult (RoceHostLocConnLite::*)(
                   const RmaBufSliceLite&, const RmtRmaBufSliceLite&, const SqeConfigLite&, unsigned long long&,
                   unsigned long long&)>(&RoceHostLocConnLite::Write))
        .expects(once())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&RoceHostLocConnLite::PollCq)
        .expects(once())
        .with(eq(128), eq(0), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_TIMEOUT));

    EXPECT_EQ(transport.Write(locBuf, rmtBuf, stream), HCCL_SUCCESS);
}

TEST_F(RoceHostLocTransportLiteImplTest, Ut_CalcUseQpNum_Expect_NdrChunkSemantics)
{
    auto smallDataId = BuildTransportUniqueId(12, BUFFER_NUM, 4, 1);
    RoceHostLocTransportLiteImpl smallDataTransport(smallDataId);
    u64 tileLen = 0;
    u64 tileLenTail = 0;
    EXPECT_EQ(smallDataTransport.CalcUseQpNum(3, tileLen, tileLenTail), 3);
    EXPECT_EQ(tileLen, 1);
    EXPECT_EQ(tileLenTail, 1);

    auto splitId = BuildTransportUniqueId(6, BUFFER_NUM, 2, 6);
    RoceHostLocTransportLiteImpl splitTransport(splitId);
    EXPECT_EQ(splitTransport.CalcUseQpNum(10, tileLen, tileLenTail), 2);
    EXPECT_EQ(tileLen, 6);
    EXPECT_EQ(tileLenTail, 4);

    auto largeThresholdId = BuildTransportUniqueId(6, BUFFER_NUM, 2, 20);
    RoceHostLocTransportLiteImpl largeThresholdTransport(largeThresholdId);
    EXPECT_EQ(largeThresholdTransport.CalcUseQpNum(10, tileLen, tileLenTail), 1);
    EXPECT_EQ(tileLen, 20);
    EXPECT_EQ(tileLenTail, 10);
}

TEST_F(RoceHostLocTransportLiteImplTest, Ut_WriteReduce_MultiQp_Expect_ElementAlignedSlices)
{
    auto multiQpId = BuildTransportUniqueId(6, BUFFER_NUM, 2, 0);
    RoceHostLocTransportLiteImpl transport(multiQpId);
    RmaBufferLite locBuf(0x4000, 12, 0x400, 0);
    Hccl::Buffer rmtBuf{0x5000, 0x200};
    ReduceIn reduceIn(DataType::FP32, ReduceOp::SUM);
    std::vector<char> streamId{};
    StreamLite stream(streamId);
    g_writeReduceSliceSizes.clear();

    MOCKER_CPP(static_cast<HcclResult (RoceHostLocConnLite::*)(
                   const RmaBufSliceLite&, const RmtRmaBufSliceLite&, const SqeConfigLite&, DataType, ReduceOp,
                   unsigned long long&, unsigned long long&)>(&RoceHostLocConnLite::WriteReduce))
        .expects(exactly(2))
        .will(invoke(CaptureWriteReduceSlice));
    MOCKER_CPP(&RoceHostLocTransportLiteImpl::BuildRdmaDbSendTask).expects(exactly(2)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&RoceHostLocConnLite::PollCq).expects(exactly(2)).will(returnValue(HCCL_SUCCESS));

    EXPECT_EQ(transport.WriteReduce(locBuf, rmtBuf, reduceIn, stream), HCCL_SUCCESS);
    ASSERT_EQ(g_writeReduceSliceSizes.size(), 2);
    EXPECT_EQ(g_writeReduceSliceSizes[0], 8);
    EXPECT_EQ(g_writeReduceSliceSizes[1], 4);
}

TEST_F(RoceHostLocTransportLiteImplTest, Ut_WriteReduce_InvalidElementLength_Expect_ParameterError)
{
    auto multiQpId = BuildTransportUniqueId(6, BUFFER_NUM, 2, 0);
    RoceHostLocTransportLiteImpl transport(multiQpId);
    RmaBufferLite locBuf(0x4000, 10, 0x400, 0);
    Hccl::Buffer rmtBuf{0x5000, 0x200};
    ReduceIn reduceIn(DataType::FP32, ReduceOp::SUM);
    Hccl::WithNotifyIn withNotify{Hccl::TransportNotifyType::NORMAL, 0};
    std::vector<char> streamId{};
    StreamLite stream(streamId);

    EXPECT_EQ(transport.WriteReduce(locBuf, rmtBuf, reduceIn, stream), HCCL_E_PARA);
    EXPECT_EQ(transport.WriteReduceWithNotify(locBuf, rmtBuf, reduceIn, withNotify, stream), HCCL_E_PARA);
}

TEST_F(RoceHostLocTransportLiteImplTest, Ut_WriteWithNotify_MultiQp_Expect_DataAndNotifyOnlyQps)
{
    auto multiQpId = BuildTransportUniqueId(6, BUFFER_NUM, 2, 20);
    RoceHostLocTransportLiteImpl transport(multiQpId);
    RmaBufferLite locBuf(0x4000, 10, 0x400, 0);
    Hccl::Buffer rmtBuf{0x5000, 0x200};
    Hccl::WithNotifyIn withNotify{Hccl::TransportNotifyType::NORMAL, 1};
    std::vector<char> streamId{};
    StreamLite stream(streamId);

    MOCKER_CPP(
        static_cast<HcclResult (RoceHostLocConnLite::*)(
            const RmaBufSliceLite&, const RmtRmaBufSliceLite&, const RmaBufSliceLite&, const RmtRmaBufSliceLite&,
            const SqeConfigLite&, unsigned long long&, unsigned long long&)>(&RoceHostLocConnLite::WriteWithNotify))
        .expects(once())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(static_cast<HcclResult (RoceHostLocConnLite::*)(
                   const RmaBufSliceLite&, const RmtRmaBufSliceLite&, const SqeConfigLite&, unsigned long long&,
                   unsigned long long&)>(&RoceHostLocConnLite::Write))
        .expects(once())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&RoceHostLocTransportLiteImpl::BuildRdmaDbSendTask).expects(exactly(2)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&RoceHostLocConnLite::PollCq).expects(exactly(2)).will(returnValue(HCCL_SUCCESS));

    EXPECT_EQ(transport.WriteWithNotify(locBuf, rmtBuf, withNotify, stream), HCCL_SUCCESS);
    EXPECT_EQ(transport.MapNotifyIdx(withNotify.index_, 0), 2);
    EXPECT_EQ(transport.MapNotifyIdx(withNotify.index_, 1), 3);
}

TEST_F(RoceHostLocTransportLiteImplTest, Ut_Write_DoorbellError_Expect_ErrorPropagated)
{
    RoceHostLocTransportLiteImpl transport(uniqueId_);
    RmaBufferLite locBuf(0x4000, 0x200, 0x400, 0);
    Hccl::Buffer rmtBuf{0x5000, 0x200};
    std::vector<char> streamId{};
    StreamLite stream(streamId);

    MOCKER_CPP(static_cast<HcclResult (RoceHostLocConnLite::*)(
                   const RmaBufSliceLite&, const RmtRmaBufSliceLite&, const SqeConfigLite&, unsigned long long&,
                   unsigned long long&)>(&RoceHostLocConnLite::Write))
        .expects(once())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&RoceHostLocTransportLiteImpl::BuildRdmaDbSendTask).expects(once()).will(returnValue(HCCL_E_REMOTE));
    MOCKER_CPP(&RoceHostLocConnLite::PollCq).expects(never());

    EXPECT_EQ(transport.Write(locBuf, rmtBuf, stream), HCCL_E_REMOTE);
}

TEST_F(RoceHostLocTransportLiteImplTest, Ut_WriteWithNotify_NotifyOnlyError_Expect_ErrorPropagated)
{
    auto multiQpId = BuildTransportUniqueId(6, BUFFER_NUM, 2, 20);
    RoceHostLocTransportLiteImpl transport(multiQpId);
    RmaBufferLite locBuf(0x4000, 10, 0x400, 0);
    Hccl::Buffer rmtBuf{0x5000, 0x200};
    Hccl::WithNotifyIn withNotify{Hccl::TransportNotifyType::NORMAL, 1};
    std::vector<char> streamId{};
    StreamLite stream(streamId);

    MOCKER_CPP(
        static_cast<HcclResult (RoceHostLocConnLite::*)(
            const RmaBufSliceLite&, const RmtRmaBufSliceLite&, const RmaBufSliceLite&, const RmtRmaBufSliceLite&,
            const SqeConfigLite&, unsigned long long&, unsigned long long&)>(&RoceHostLocConnLite::WriteWithNotify))
        .expects(once())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(static_cast<HcclResult (RoceHostLocConnLite::*)(
                   const RmaBufSliceLite&, const RmtRmaBufSliceLite&, const SqeConfigLite&, unsigned long long&,
                   unsigned long long&)>(&RoceHostLocConnLite::Write))
        .expects(once())
        .will(returnValue(HCCL_E_REMOTE));
    MOCKER_CPP(&RoceHostLocTransportLiteImpl::BuildRdmaDbSendTask).expects(once()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&RoceHostLocConnLite::PollCq).expects(never());

    EXPECT_EQ(transport.WriteWithNotify(locBuf, rmtBuf, withNotify, stream), HCCL_E_REMOTE);
}

TEST_F(RoceHostLocTransportLiteImplTest, Ut_WaitWithTimeout_MultiQp_Expect_OnlyMappedWaits)
{
    auto multiQpId = BuildTransportUniqueId(6, BUFFER_NUM, 2, 0);
    RoceHostLocTransportLiteImpl transport(multiQpId);
    std::vector<char> streamId{};
    StreamLite stream(streamId);
    g_notifyWaitCalls.clear();

    MOCKER_CPP(&RoceHostLocTransportLiteImpl::BuildNotifyWaitTask).expects(exactly(2)).will(invoke(CaptureNotifyWait));

    transport.WaitWithTimeout(1, stream, 1800);
    const std::vector<std::pair<u32, u32>> expectedCalls{{3, 900}, {4, 900}};
    EXPECT_EQ(g_notifyWaitCalls, expectedCalls);
}

TEST_F(RoceHostLocTransportLiteImplTest, Ut_Init_InvalidNotifyNumConnNum_Expect_Throw)
{
    // connNum=0 应抛出异常
    auto zeroConnId = BuildTransportUniqueId(0, BUFFER_NUM, 0, 0);
    EXPECT_THROW(RoceHostLocTransportLiteImpl transport(zeroConnId), InternalException);

    // notifyNum=4, connNum=2 不满足 notifyNum == 3 * connNum，应抛出异常
    auto mismatchId = BuildTransportUniqueId(4, BUFFER_NUM, 2, 0);
    EXPECT_THROW(RoceHostLocTransportLiteImpl transport(mismatchId), InternalException);

    // notifyNum=2, connNum=1 不满足 notifyNum == 3 * connNum，应抛出异常
    auto singleMismatchId = BuildTransportUniqueId(2, BUFFER_NUM, 1, 0);
    EXPECT_THROW(RoceHostLocTransportLiteImpl transport(singleMismatchId), InternalException);
}

TEST_F(RoceHostLocTransportLiteImplTest, Ut_Post_MultiQp_Expect_PerQpNotifyFanout)
{
    auto multiQpId = BuildTransportUniqueId(6, BUFFER_NUM, 2, 0);
    RoceHostLocTransportLiteImpl transport(multiQpId);
    std::vector<char> streamId{};
    StreamLite stream(streamId);

    MOCKER_CPP(static_cast<HcclResult (RoceHostLocConnLite::*)(
                   const RmaBufSliceLite&, const RmtRmaBufSliceLite&, const SqeConfigLite&, unsigned long long&,
                   unsigned long long&)>(&RoceHostLocConnLite::Write))
        .expects(exactly(2))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&RoceHostLocTransportLiteImpl::BuildRdmaDbSendTask).expects(exactly(2)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&RoceHostLocConnLite::PollCq).expects(exactly(2)).will(returnValue(HCCL_SUCCESS));

    EXPECT_EQ(transport.Post(1, stream), HCCL_SUCCESS);
    EXPECT_EQ(transport.MapNotifyIdx(1, 0), 2);
    EXPECT_EQ(transport.MapNotifyIdx(1, 1), 3);
}

TEST_F(RoceHostLocTransportLiteImplTest, Ut_Post_InvalidNotifyIndex_Expect_ParameterError)
{
    auto multiQpId = BuildTransportUniqueId(6, BUFFER_NUM, 2, 0);
    RoceHostLocTransportLiteImpl transport(multiQpId);
    std::vector<char> streamId{};
    StreamLite stream(streamId);

    MOCKER_CPP(static_cast<HcclResult (RoceHostLocConnLite::*)(
                   const RmaBufSliceLite&, const RmtRmaBufSliceLite&, const SqeConfigLite&, unsigned long long&,
                   unsigned long long&)>(&RoceHostLocConnLite::Write))
        .expects(never());

    // connNum=2、notifyNum=6 时，逻辑 notify 3 映射到物理下标 6，越界
    EXPECT_EQ(transport.Post(3, stream), HCCL_E_PARA);
}

TEST_F(RoceHostLocTransportLiteImplTest, Ut_Read_MultiQp_Expect_SlicedAcrossQps)
{
    auto multiQpId = BuildTransportUniqueId(6, BUFFER_NUM, 2, 0);
    RoceHostLocTransportLiteImpl transport(multiQpId);
    RmaBufferLite locBuf(0x4000, 10, 0x400, 0);
    Hccl::Buffer rmtBuf{0x5000, 0x200};
    std::vector<char> streamId{};
    StreamLite stream(streamId);
    g_rwLocSlices.clear();
    g_rwRmtSlices.clear();

    MOCKER_CPP(static_cast<HcclResult (RoceHostLocConnLite::*)(
                   const RmaBufSliceLite&, const RmtRmaBufSliceLite&, const SqeConfigLite&, unsigned long long&,
                   unsigned long long&)>(&RoceHostLocConnLite::Read))
        .expects(exactly(2))
        .will(invoke(CaptureReadWriteSlice));
    MOCKER_CPP(&RoceHostLocTransportLiteImpl::BuildRdmaDbSendTask).expects(exactly(2)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&RoceHostLocConnLite::PollCq).expects(exactly(2)).will(returnValue(HCCL_SUCCESS));

    EXPECT_EQ(transport.Read(locBuf, rmtBuf, stream), HCCL_SUCCESS);
    ASSERT_EQ(g_rwLocSlices.size(), 2);
    EXPECT_EQ(g_rwLocSlices[0].first, 0x4000u);
    EXPECT_EQ(g_rwLocSlices[0].second, 5u);
    EXPECT_EQ(g_rwLocSlices[1].first, 0x4005u);
    EXPECT_EQ(g_rwLocSlices[1].second, 5u);
    EXPECT_EQ(g_rwRmtSlices[0].first, 0x5000u);
    EXPECT_EQ(g_rwRmtSlices[1].first, 0x5005u);
}

TEST_F(RoceHostLocTransportLiteImplTest, Ut_Write_MultiQp_Expect_SlicedAcrossQps)
{
    auto multiQpId = BuildTransportUniqueId(6, BUFFER_NUM, 2, 0);
    RoceHostLocTransportLiteImpl transport(multiQpId);
    RmaBufferLite locBuf(0x4000, 10, 0x400, 0);
    Hccl::Buffer rmtBuf{0x5000, 0x200};
    std::vector<char> streamId{};
    StreamLite stream(streamId);
    g_rwLocSlices.clear();
    g_rwRmtSlices.clear();

    MOCKER_CPP(static_cast<HcclResult (RoceHostLocConnLite::*)(
                   const RmaBufSliceLite&, const RmtRmaBufSliceLite&, const SqeConfigLite&, unsigned long long&,
                   unsigned long long&)>(&RoceHostLocConnLite::Write))
        .expects(exactly(2))
        .will(invoke(CaptureReadWriteSlice));
    MOCKER_CPP(&RoceHostLocTransportLiteImpl::BuildRdmaDbSendTask).expects(exactly(2)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&RoceHostLocConnLite::PollCq).expects(exactly(2)).will(returnValue(HCCL_SUCCESS));

    EXPECT_EQ(transport.Write(locBuf, rmtBuf, stream), HCCL_SUCCESS);
    ASSERT_EQ(g_rwLocSlices.size(), 2);
    EXPECT_EQ(g_rwLocSlices[0].first, 0x4000u);
    EXPECT_EQ(g_rwLocSlices[0].second, 5u);
    EXPECT_EQ(g_rwLocSlices[1].first, 0x4005u);
    EXPECT_EQ(g_rwLocSlices[1].second, 5u);
    EXPECT_EQ(g_rwRmtSlices[0].first, 0x5000u);
    EXPECT_EQ(g_rwRmtSlices[1].first, 0x5005u);
}
