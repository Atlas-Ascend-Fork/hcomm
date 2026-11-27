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
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#define private public
#include "ub_conn_lite.h"
#include "reduce_in.h"
#include "stream_lite.h"
#include "ascend_hal.h"
#include "rtsq_base.h"
#include "sqe.h"
#include "internal_exception.h"
#undef private

using namespace Hccl;

class AicpuUbConnLiteTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "UbConnLite tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "UbConnLite tests tear down." << std::endl; }

    virtual void SetUp()
    {
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
        std::cout << "A Test case in UbConnLite SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in UbConnLite TearDown" << std::endl;
    }

    u8 mockSq[AC_SQE_SIZE * AC_SQE_MAX_CNT]{0};
    uint32_t mockPiSlot{};
    uint32_t mockCiSlot{};
};

TEST_F(AicpuUbConnLiteTest, test_RmaConnLite)
{
    RmaConnLite rdma(1);
    EXPECT_EQ(1, rdma.GetQpVa());

    UbJettyLiteId id(1, 1, 1);
    UbJettyLiteAttr attr(1, 1, 1, 1, false);
    Eid rmtEid;
    rmtEid.raw[0] = 1;

    RmaConnLite ub(id, attr, rmtEid);

    std::cout << ub.Describe() << std::endl;

    EXPECT_EQ(1, ub.GetUbJettyLiteId().GetDieId());
    EXPECT_EQ(1, ub.GetUbJettyLiteAttr().dbAddr_);
    EXPECT_EQ(1, ub.GetRmtEid().raw[0]);
}

TEST_F(AicpuUbConnLiteTest, test_UBConnLite_construct_by_unique_id)
{
    std::vector<char> data(57, 0);
    UbConnLite ubConn(data);
    ubConn.SetSqPiCi(&mockPiSlot, &mockCiSlot);
}

TEST_F(AicpuUbConnLiteTest, test_UBConnLite_Read)
{
    UbJettyLiteId id(1, 1, 1);
    u8 sqVa[1024];
    u64 sqVAaddr = reinterpret_cast<u64>(sqVa);
    UbJettyLiteAttr attr(1, sqVAaddr, 1, 1, false);
    Eid rmtEid;
    rmtEid.raw[0] = 1;
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);
    UbConnLite ubConn(id, attr, rmtEid);
    ubConn.SetSqPiCi(&mockPiSlot, &mockCiSlot);
    EXPECT_EQ(1, ubConn.GetUbJettyLiteId().GetDieId());
    EXPECT_EQ(1, ubConn.GetUbJettyLiteAttr().dbAddr_);
    EXPECT_EQ(1, ubConn.GetRmtEid().raw[0]);

    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));
    RmaBufSliceLite loc(0x1111, 64, 1, 1);
    RmtRmaBufSliceLite rmt(0x2222, 64, 1, 1, 1, UINT32_MAX);
    RmtRmaBufSliceLite notify(0x2222, 64, 1, 1, 1, UINT32_MAX);
    u64 notifyData(0x000000000000ffff);
    SqeConfigLite cfg;
    ConnLiteOperationOut out;
    u8 data[512];
    out.pi = 0;
    out.data = data;
    UdmaSqeWrite* sqe = (UdmaSqeWrite*)out.data;
    out.dataSize = 64;
    rmt.GetTokenValue();
    EXPECT_EQ(ubConn.Read(loc, rmt, cfg, stream, out), HCCL_SUCCESS);
}

TEST_F(AicpuUbConnLiteTest, test_UBConnLite_Read_Slice)
{
    UbJettyLiteId id(1, 1, 1);
    u8 sqVa[1024];
    u64 sqVAaddr = reinterpret_cast<u64>(sqVa);
    UbJettyLiteAttr attr(1, sqVAaddr, 4, 1, false);
    Eid rmtEid;
    rmtEid.raw[0] = 1;

    UbConnLite ubConn(id, attr, rmtEid);
    ubConn.SetSqPiCi(&mockPiSlot, &mockCiSlot);
    EXPECT_EQ(1, ubConn.GetUbJettyLiteId().GetDieId());
    EXPECT_EQ(1, ubConn.GetUbJettyLiteAttr().dbAddr_);
    EXPECT_EQ(1, ubConn.GetRmtEid().raw[0]);

    RmaBufSliceLite loc(0x1111, 257 * 1024 * 1024, 1, 1);
    RmtRmaBufSliceLite rmt(0x2222, 257 * 1024 * 1024, 1, 1, 1, UINT32_MAX);
    RmtRmaBufSliceLite notify(0x2222, 64, 1, 1, 1, UINT32_MAX);
    u64 notifyData(0x000000000000ffff);
    SqeConfigLite cfg;
    ConnLiteOperationOut out;
    u8 data[512];
    out.pi = 0;
    out.data = data;
    UdmaSqeWrite* sqe = (UdmaSqeWrite*)out.data;
    out.dataSize = 64;
    rmt.GetTokenValue();
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));
    EXPECT_EQ(ubConn.Read(loc, rmt, cfg, stream, out), HCCL_SUCCESS);
}

TEST_F(AicpuUbConnLiteTest, test_UBConnLite_ReadReduce_Slice)
{
    UbJettyLiteId id(1, 1, 1);
    u8 sqVa[1024];
    u64 sqVAaddr = reinterpret_cast<u64>(sqVa);
    UbJettyLiteAttr attr(1, sqVAaddr, 4, 1, false);
    Eid rmtEid;
    rmtEid.raw[0] = 1;

    UbConnLite ubConn(id, attr, rmtEid);
    ubConn.SetSqPiCi(&mockPiSlot, &mockCiSlot);
    EXPECT_EQ(1, ubConn.GetUbJettyLiteId().GetDieId());
    EXPECT_EQ(1, ubConn.GetUbJettyLiteAttr().dbAddr_);
    EXPECT_EQ(1, ubConn.GetRmtEid().raw[0]);

    RmaBufSliceLite loc(0x1111, 257 * 1024 * 1024, 1, 1);
    RmtRmaBufSliceLite rmt(0x2222, 257 * 1024 * 1024, 1, 1, 1, UINT32_MAX);
    RmtRmaBufSliceLite notify(0x2222, 64, 1, 1, 1, UINT32_MAX);
    u64 notifyData(1);
    SqeConfigLite cfg;
    ConnLiteOperationOut out;
    u8 data[512];
    out.pi = 0;
    out.data = data;
    UdmaSqeWrite* sqe = (UdmaSqeWrite*)out.data;
    out.dataSize = 64;
    rmt.GetTokenValue();
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);
    ReduceIn reduceIn(DataType::INT8, ReduceOp::SUM);
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));
    EXPECT_EQ(ubConn.ReadReduce(reduceIn, loc, rmt, stream, cfg, out), HCCL_SUCCESS);
}

TEST_F(AicpuUbConnLiteTest, test_UBConnLite_ReadReduce)
{
    UbJettyLiteId id(1, 1, 1);
    u8 sqVa[1024];
    u64 sqVAaddr = reinterpret_cast<u64>(sqVa);
    UbJettyLiteAttr attr(1, sqVAaddr, 1, 1, false);
    Eid rmtEid;
    rmtEid.raw[0] = 1;

    UbConnLite ubConn(id, attr, rmtEid);
    ubConn.SetSqPiCi(&mockPiSlot, &mockCiSlot);
    EXPECT_EQ(1, ubConn.GetUbJettyLiteId().GetDieId());
    EXPECT_EQ(1, ubConn.GetUbJettyLiteAttr().dbAddr_);
    EXPECT_EQ(1, ubConn.GetRmtEid().raw[0]);

    RmaBufSliceLite loc(0x1111, 64, 1, 1);
    RmtRmaBufSliceLite rmt(0x2222, 64, 1, 1, 1, UINT32_MAX);
    RmtRmaBufSliceLite notify(0x2222, 64, 1, 1, 1, UINT32_MAX);
    u64 notifyData(1);
    SqeConfigLite cfg;
    ConnLiteOperationOut out;
    u8 data[512];
    out.pi = 0;
    out.data = data;
    UdmaSqeWrite* sqe = (UdmaSqeWrite*)out.data;
    out.dataSize = 64;
    rmt.GetTokenValue();
    ReduceIn reduceIn(DataType::INT8, ReduceOp::SUM);
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));
    EXPECT_EQ(ubConn.ReadReduce(reduceIn, loc, rmt, stream, cfg, out), HCCL_SUCCESS);
}

TEST_F(AicpuUbConnLiteTest, test_UBConnLite_Write)
{
    UbJettyLiteId id(1, 1, 1);
    u8 sqVa[1024];
    u64 sqVAaddr = reinterpret_cast<u64>(sqVa);
    UbJettyLiteAttr attr(1, sqVAaddr, 1, 1, false);
    Eid rmtEid;
    rmtEid.raw[0] = 1;

    UbConnLite ubConn(id, attr, rmtEid);
    ubConn.SetSqPiCi(&mockPiSlot, &mockCiSlot);
    EXPECT_EQ(1, ubConn.GetUbJettyLiteId().GetDieId());
    EXPECT_EQ(1, ubConn.GetUbJettyLiteAttr().dbAddr_);
    EXPECT_EQ(1, ubConn.GetRmtEid().raw[0]);

    RmaBufSliceLite loc(0x1111, 64, 1, 1);
    RmtRmaBufSliceLite rmt(0x2222, 64, 1, 1, 1, UINT32_MAX);
    RmtRmaBufSliceLite notify(0x2222, 64, 1, 1, 1, UINT32_MAX);
    u64 notifyData(0x000000000000ffff);
    SqeConfigLite cfg;
    ConnLiteOperationOut out;
    u8 data[512];
    out.pi = 0;
    out.data = data;
    UdmaSqeWrite* sqe = (UdmaSqeWrite*)out.data;
    out.dataSize = 64;
    rmt.GetTokenValue();
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));
    EXPECT_EQ(ubConn.Write(loc, rmt, cfg, stream, out), HCCL_SUCCESS);
}

TEST_F(AicpuUbConnLiteTest, test_UBConnLite_Write_Slice)
{
    UbJettyLiteId id(1, 1, 1);
    u8 sqVa[1024];
    u64 sqVAaddr = reinterpret_cast<u64>(sqVa);
    UbJettyLiteAttr attr(1, sqVAaddr, 4, 1, false);
    Eid rmtEid;
    rmtEid.raw[0] = 1;

    UbConnLite ubConn(id, attr, rmtEid);
    ubConn.SetSqPiCi(&mockPiSlot, &mockCiSlot);
    EXPECT_EQ(1, ubConn.GetUbJettyLiteId().GetDieId());
    EXPECT_EQ(1, ubConn.GetUbJettyLiteAttr().dbAddr_);
    EXPECT_EQ(1, ubConn.GetRmtEid().raw[0]);

    RmaBufSliceLite loc(0x1111, 257 * 1024 * 1024, 1, 1);
    RmtRmaBufSliceLite rmt(0x2222, 257 * 1024 * 1024, 1, 1, 1, UINT32_MAX);
    RmtRmaBufSliceLite notify(0x2222, 64, 1, 1, 1, UINT32_MAX);
    u64 notifyData(0x000000000000ffff);
    SqeConfigLite cfg;
    ConnLiteOperationOut out;
    u8 data[512];
    out.pi = 0;
    out.data = data;
    UdmaSqeWrite* sqe = (UdmaSqeWrite*)out.data;
    out.dataSize = 64;
    rmt.GetTokenValue();
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));
    EXPECT_EQ(ubConn.Write(loc, rmt, cfg, stream, out), HCCL_SUCCESS);
}

TEST_F(AicpuUbConnLiteTest, test_UBConnLite_WriteReduceWithNotify)
{
    UbJettyLiteId id(1, 1, 1);
    u8 sqVa[1024];
    u64 sqVAaddr = reinterpret_cast<u64>(sqVa);
    UbJettyLiteAttr attr(1, sqVAaddr, 4096, 1, false);
    Eid rmtEid;
    rmtEid.raw[0] = 1;

    UbConnLite ubConn(id, attr, rmtEid);
    ubConn.SetSqPiCi(&mockPiSlot, &mockCiSlot);
    std::cout << ubConn.Describe() << std::endl;

    EXPECT_EQ(1, ubConn.GetUbJettyLiteId().GetDieId());
    EXPECT_EQ(1, ubConn.GetUbJettyLiteAttr().dbAddr_);
    EXPECT_EQ(1, ubConn.GetRmtEid().raw[0]);

    RmaBufSliceLite loc(0x1111, 64, 1, 1);
    RmtRmaBufSliceLite rmt(0x2222, 64, 1, 1, 1, UINT32_MAX);
    RmtRmaBufSliceLite notify(0x2222, 64, 1, 1, 1, UINT32_MAX);
    u64 notifyData(1);
    SqeConfigLite cfg;
    ConnLiteOperationOut out;
    u8 data[512];
    out.pi = 0;
    out.data = data;
    UdmaSqeWrite* sqe = (UdmaSqeWrite*)out.data;
    out.dataSize = 64;
    rmt.GetTokenValue();
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);

    MOCKER(memset_s).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER(memcpy_s).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));

    EXPECT_EQ(
        ubConn.WriteReduceWithNotify(DataType::INT8, ReduceOp::SUM, loc, rmt, cfg, stream, out, notify, notifyData),
        HCCL_SUCCESS);
    EXPECT_EQ(
        ubConn.WriteReduceWithNotify(DataType::INT16, ReduceOp::MAX, loc, rmt, cfg, stream, out, notify, notifyData),
        HCCL_SUCCESS);
    EXPECT_EQ(
        ubConn.WriteReduceWithNotify(DataType::INT32, ReduceOp::MIN, loc, rmt, cfg, stream, out, notify, notifyData),
        HCCL_SUCCESS);
    EXPECT_EQ(
        ubConn.WriteReduceWithNotify(DataType::FP16, ReduceOp::SUM, loc, rmt, cfg, stream, out, notify, notifyData),
        HCCL_SUCCESS);
    EXPECT_EQ(
        ubConn.WriteReduceWithNotify(DataType::FP32, ReduceOp::MAX, loc, rmt, cfg, stream, out, notify, notifyData),
        HCCL_SUCCESS);
    EXPECT_EQ(
        ubConn.WriteReduceWithNotify(DataType::UINT8, ReduceOp::MAX, loc, rmt, cfg, stream, out, notify, notifyData),
        HCCL_SUCCESS);
    EXPECT_EQ(
        ubConn.WriteReduceWithNotify(DataType::UINT32, ReduceOp::MAX, loc, rmt, cfg, stream, out, notify, notifyData),
        HCCL_SUCCESS);
}

TEST_F(AicpuUbConnLiteTest, test_UBConnLite_WriteReduceWithNotify_Slice)
{
    UbJettyLiteId id(1, 1, 1);
    u8 sqVa[1024];
    u64 sqVAaddr = reinterpret_cast<u64>(sqVa);
    UbJettyLiteAttr attr(1, sqVAaddr, 4096, 1, false);
    Eid rmtEid;
    rmtEid.raw[0] = 1;

    UbConnLite ubConn(id, attr, rmtEid);
    ubConn.SetSqPiCi(&mockPiSlot, &mockCiSlot);
    std::cout << ubConn.Describe() << std::endl;

    EXPECT_EQ(1, ubConn.GetUbJettyLiteId().GetDieId());
    EXPECT_EQ(1, ubConn.GetUbJettyLiteAttr().dbAddr_);
    EXPECT_EQ(1, ubConn.GetRmtEid().raw[0]);

    RmaBufSliceLite loc(0x1111, 257 * 1024 * 1024, 1, 1);
    RmtRmaBufSliceLite rmt(0x2222, 257 * 1024 * 1024, 1, 1, 1, UINT32_MAX);
    RmtRmaBufSliceLite notify(0x2222, 64, 1, 1, 1, UINT32_MAX);
    u64 notifyData(1);
    SqeConfigLite cfg;
    ConnLiteOperationOut out;
    u8 data[512];
    out.pi = 0;
    out.data = data;
    UdmaSqeWrite* sqe = (UdmaSqeWrite*)out.data;
    out.dataSize = 64;
    rmt.GetTokenValue();
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);

    MOCKER(memset_s).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER(memcpy_s).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));
    EXPECT_EQ(
        ubConn.WriteReduceWithNotify(DataType::INT8, ReduceOp::SUM, loc, rmt, cfg, stream, out, notify, notifyData),
        HCCL_SUCCESS);
    EXPECT_EQ(
        ubConn.WriteReduceWithNotify(DataType::INT16, ReduceOp::MAX, loc, rmt, cfg, stream, out, notify, notifyData),
        HCCL_SUCCESS);
    EXPECT_EQ(
        ubConn.WriteReduceWithNotify(DataType::INT32, ReduceOp::MIN, loc, rmt, cfg, stream, out, notify, notifyData),
        HCCL_SUCCESS);
    EXPECT_EQ(
        ubConn.WriteReduceWithNotify(DataType::FP16, ReduceOp::SUM, loc, rmt, cfg, stream, out, notify, notifyData),
        HCCL_SUCCESS);
    EXPECT_EQ(
        ubConn.WriteReduceWithNotify(DataType::FP32, ReduceOp::MAX, loc, rmt, cfg, stream, out, notify, notifyData),
        HCCL_SUCCESS);
}

TEST_F(AicpuUbConnLiteTest, test_UBConnLite_WriteWithNotify)
{
    UbJettyLiteId id(1, 1, 1);
    u8 sqVa[1024];
    u64 sqVAaddr = reinterpret_cast<u64>(sqVa);
    UbJettyLiteAttr attr(1, sqVAaddr, 4096, 1, false);
    Eid rmtEid;
    rmtEid.raw[0] = 1;

    UbConnLite ubConn(id, attr, rmtEid);
    ubConn.SetSqPiCi(&mockPiSlot, &mockCiSlot);
    EXPECT_EQ(1, ubConn.GetUbJettyLiteId().GetDieId());
    EXPECT_EQ(1, ubConn.GetUbJettyLiteAttr().dbAddr_);
    EXPECT_EQ(1, ubConn.GetRmtEid().raw[0]);

    RmaBufSliceLite loc(0x1111, 64, 1, 1);
    RmtRmaBufSliceLite rmt(0x2222, 64, 1, 1, 1, UINT32_MAX);
    RmtRmaBufSliceLite notify(0x2222, 64, 1, 1, 1, UINT32_MAX);
    u64 notifyData(1);
    SqeConfigLite cfg;
    ConnLiteOperationOut out;
    u8 data[512];
    out.pi = 0;
    out.data = data;
    UdmaSqeWrite* sqe = (UdmaSqeWrite*)out.data;
    out.dataSize = 64;
    rmt.GetTokenValue();
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);

    MOCKER(memset_s).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER(memcpy_s).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));
    EXPECT_EQ(ubConn.WriteWithNotify(loc, rmt, cfg, out, notify, stream, notifyData), HCCL_SUCCESS);
}

TEST_F(AicpuUbConnLiteTest, test_UBConnLite_WriteWithNotify_Detour)
{
    UbJettyLiteId id(1, 1, 1);
    u8 sqVa[1024];
    u64 sqVAaddr = reinterpret_cast<u64>(sqVa);
    UbJettyLiteAttr attr(1, sqVAaddr, 3, 1, false);
    Eid rmtEid;
    rmtEid.raw[0] = 1;

    UbConnLite ubConn(id, attr, rmtEid);
    ubConn.SetSqPiCi(&mockPiSlot, &mockCiSlot);
    EXPECT_EQ(1, ubConn.GetUbJettyLiteId().GetDieId());
    EXPECT_EQ(1, ubConn.GetUbJettyLiteAttr().dbAddr_);
    EXPECT_EQ(1, ubConn.GetRmtEid().raw[0]);

    RmaBufSliceLite loc(0x1111, 64, 1, 1);
    RmtRmaBufSliceLite rmt(0x2222, 64, 1, 1, 1, UINT32_MAX);
    RmtRmaBufSliceLite notify(0x2222, 64, 1, 1, 1, UINT32_MAX);
    u64 notifyData(1);
    SqeConfigLite cfg;
    ConnLiteOperationOut out;
    u8 data[512];
    out.pi = 0;
    out.data = data;
    UdmaSqeWrite* sqe = (UdmaSqeWrite*)out.data;
    out.dataSize = 64;
    rmt.GetTokenValue();
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);
    ubConn.SetCi(0);
    EXPECT_EQ(ubConn.Write(loc, rmt, cfg, stream, out), HCCL_SUCCESS);
    EXPECT_EQ(ubConn.Write(loc, rmt, cfg, stream, out), HCCL_SUCCESS);

    ubConn.SetCi(1);
    MOCKER(memset_s).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER(memcpy_s).stubs().with(mockcpp::any()).will(returnValue(0));
    EXPECT_EQ(ubConn.WriteWithNotify(loc, rmt, cfg, out, notify, stream, notifyData), HCCL_SUCCESS);
    EXPECT_EQ(1, ubConn.piDetourCount);
    EXPECT_EQ(0, ubConn.ciDetourCount);

    ubConn.SetCi(4);
    ubConn.ciDetourCount = 1;
    EXPECT_EQ(ubConn.WriteWithNotify(loc, rmt, cfg, out, notify, stream, notifyData), HCCL_SUCCESS);
    EXPECT_EQ(2, ubConn.piDetourCount);
}

TEST_F(AicpuUbConnLiteTest, test_UBConnLite_WriteWithNotify_Expect_E_Internal)
{
    UbJettyLiteId id(1, 1, 1);
    u8 sqVa[1024];
    u64 sqVAaddr = reinterpret_cast<u64>(sqVa);
    UbJettyLiteAttr attr(1, sqVAaddr, 3, 1, false);
    Eid rmtEid;
    rmtEid.raw[0] = 1;

    UbConnLite ubConn(id, attr, rmtEid);
    ubConn.SetSqPiCi(&mockPiSlot, &mockCiSlot);
    EXPECT_EQ(1, ubConn.GetUbJettyLiteId().GetDieId());
    EXPECT_EQ(1, ubConn.GetUbJettyLiteAttr().dbAddr_);
    EXPECT_EQ(1, ubConn.GetRmtEid().raw[0]);

    RmaBufSliceLite loc(0x1111, 64, 1, 1);
    RmtRmaBufSliceLite rmt(0x2222, 64, 1, 1, 1, UINT32_MAX);
    RmtRmaBufSliceLite notify(0x2222, 64, 1, 1, 1, UINT32_MAX);
    u64 notifyData(1);
    SqeConfigLite cfg;
    ConnLiteOperationOut out;
    u8 data[512];
    out.pi = 0;
    out.data = data;
    UdmaSqeWrite* sqe = (UdmaSqeWrite*)out.data;
    out.dataSize = 64;
    rmt.GetTokenValue();
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);
    ubConn.SetCi(0);
    MOCKER(memset_s).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER(memcpy_s).stubs().with(mockcpp::any()).will(returnValue(0));
    EXPECT_EQ(ubConn.WriteWithNotify(loc, rmt, cfg, out, notify, stream, notifyData), HCCL_E_INTERNAL);

    ubConn.SetCi(2);
    MOCKER(memset_s).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER(memcpy_s).stubs().with(mockcpp::any()).will(returnValue(1));
    EXPECT_EQ(ubConn.WriteWithNotify(loc, rmt, cfg, out, notify, stream, notifyData), HCCL_E_INTERNAL);
    EXPECT_EQ(1, ubConn.piDetourCount);
}

TEST_F(AicpuUbConnLiteTest, test_UBConnLite_WriteWithNotify_Slice)
{
    UbJettyLiteId id(1, 1, 1);
    u8 sqVa[1024];
    u64 sqVAaddr = reinterpret_cast<u64>(sqVa);
    UbJettyLiteAttr attr(1, sqVAaddr, 4096, 1, false);
    Eid rmtEid;
    rmtEid.raw[0] = 1;

    UbConnLite ubConn(id, attr, rmtEid);
    ubConn.SetSqPiCi(&mockPiSlot, &mockCiSlot);
    EXPECT_EQ(1, ubConn.GetUbJettyLiteId().GetDieId());
    EXPECT_EQ(1, ubConn.GetUbJettyLiteAttr().dbAddr_);
    EXPECT_EQ(1, ubConn.GetRmtEid().raw[0]);

    RmaBufSliceLite loc(0x1111, 257 * 1024 * 1024, 1, 1);
    RmtRmaBufSliceLite rmt(0x2222, 257 * 1024 * 1024, 1, 1, 1, UINT32_MAX);
    RmtRmaBufSliceLite notify(0x2222, 64, 1, 1, 1, UINT32_MAX);
    u64 notifyData(1);
    SqeConfigLite cfg;
    ConnLiteOperationOut out;
    u8 data[512];
    out.pi = 0;
    out.data = data;
    UdmaSqeWrite* sqe = (UdmaSqeWrite*)out.data;
    out.dataSize = 64;
    rmt.GetTokenValue();
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);
    MOCKER(memset_s).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER(memcpy_s).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));
    EXPECT_EQ(ubConn.WriteWithNotify(loc, rmt, cfg, out, notify, stream, notifyData), HCCL_SUCCESS);
}

TEST_F(AicpuUbConnLiteTest, test_UBConnLite_InlineWrite)
{
    UbJettyLiteId id(1, 1, 1);
    u8 sqVa[1024];
    u64 sqVAaddr = reinterpret_cast<u64>(sqVa);
    UbJettyLiteAttr attr(1, sqVAaddr, 1, 1, false);
    Eid rmtEid;
    rmtEid.raw[0] = 1;

    UbConnLite ubConn(id, attr, rmtEid);
    ubConn.SetSqPiCi(&mockPiSlot, &mockCiSlot);
    EXPECT_EQ(1, ubConn.GetUbJettyLiteId().GetDieId());
    EXPECT_EQ(1, ubConn.GetUbJettyLiteAttr().dbAddr_);
    EXPECT_EQ(1, ubConn.GetRmtEid().raw[0]);

    RmaBufSliceLite loc(0x1111, 64, 1, 1);
    RmtRmaBufSliceLite rmt(0x2222, 64, 1, 1, 1, UINT32_MAX);
    RmtRmaBufSliceLite notify(0x2222, 64, 1, 1, 1, UINT32_MAX);
    u64 notifyData(1);
    SqeConfigLite cfg;
    ConnLiteOperationOut out;
    u8 data[512];
    out.pi = 0;
    out.data = data;
    UdmaSqeWrite* sqe = (UdmaSqeWrite*)out.data;
    out.dataSize = 64;
    rmt.GetTokenValue();
    u8 write_data(1);
    u16 write_dsize(1);
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));
    EXPECT_EQ(ubConn.InlineWrite(&write_data, write_dsize, rmt, cfg, stream, out), HCCL_SUCCESS);
}

TEST_F(AicpuUbConnLiteTest, test_UBConnLite_WriteReducee)
{
    UbJettyLiteId id(1, 1, 1);
    u8 sqVa[1024];
    u64 sqVAaddr = reinterpret_cast<u64>(sqVa);
    UbJettyLiteAttr attr(1, sqVAaddr, 1, 1, false);
    Eid rmtEid;
    rmtEid.raw[0] = 1;

    UbConnLite ubConn(id, attr, rmtEid);
    ubConn.SetSqPiCi(&mockPiSlot, &mockCiSlot);
    EXPECT_EQ(1, ubConn.GetUbJettyLiteId().GetDieId());
    EXPECT_EQ(1, ubConn.GetUbJettyLiteAttr().dbAddr_);
    EXPECT_EQ(1, ubConn.GetRmtEid().raw[0]);

    RmaBufSliceLite loc(0x1111, 64, 1, 1);
    RmtRmaBufSliceLite rmt(0x2222, 64, 1, 1, 1, UINT32_MAX);
    RmtRmaBufSliceLite notify(0x2222, 64, 1, 1, 1, UINT32_MAX);
    u64 notifyData(1);
    SqeConfigLite cfg;
    ConnLiteOperationOut out;
    u8 data[512];
    out.pi = 0;
    out.data = data;
    UdmaSqeWrite* sqe = (UdmaSqeWrite*)out.data;
    out.dataSize = 64;
    rmt.GetTokenValue();
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));
    EXPECT_EQ(ubConn.WriteReduce(DataType::INT8, ReduceOp::SUM, loc, stream, rmt, cfg, out), HCCL_SUCCESS);
}

TEST_F(AicpuUbConnLiteTest, test_UBConnLite_WriteReduce_Slice)
{
    UbJettyLiteId id(1, 1, 1);
    u8 sqVa[1024];
    u64 sqVAaddr = reinterpret_cast<u64>(sqVa);
    UbJettyLiteAttr attr(1, sqVAaddr, 4, 1, false);
    Eid rmtEid;
    rmtEid.raw[0] = 1;

    UbConnLite ubConn(id, attr, rmtEid);
    ubConn.SetSqPiCi(&mockPiSlot, &mockCiSlot);
    EXPECT_EQ(1, ubConn.GetUbJettyLiteId().GetDieId());
    EXPECT_EQ(1, ubConn.GetUbJettyLiteAttr().dbAddr_);
    EXPECT_EQ(1, ubConn.GetRmtEid().raw[0]);

    RmaBufSliceLite loc(0x1111, 257 * 1024 * 1024, 1, 1);
    RmtRmaBufSliceLite rmt(0x2222, 257 * 1024 * 1024, 1, 1, 1, UINT32_MAX);
    RmtRmaBufSliceLite notify(0x2222, 64, 1, 1, 1, UINT32_MAX);
    u64 notifyData(1);
    SqeConfigLite cfg;
    ConnLiteOperationOut out;
    u8 data[512];
    out.pi = 0;
    out.data = data;
    UdmaSqeWrite* sqe = (UdmaSqeWrite*)out.data;
    out.dataSize = 64;
    rmt.GetTokenValue();
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));
    EXPECT_EQ(ubConn.WriteReduce(DataType::INT8, ReduceOp::SUM, loc, stream, rmt, cfg, out), HCCL_SUCCESS);
}

TEST_F(AicpuUbConnLiteTest, test_UBConnLite_BatchOneSidedRead)
{
    UbJettyLiteId id(1, 1, 1);
    u8 sqVa[1024];
    u64 sqVAaddr = reinterpret_cast<u64>(sqVa);
    UbJettyLiteAttr attr(1, sqVAaddr, 4, 1, false);
    Eid rmtEid;
    rmtEid.raw[0] = 1;

    UbConnLite ubConn(id, attr, rmtEid);
    ubConn.SetSqPiCi(&mockPiSlot, &mockCiSlot);
    u32 dataSize = 400 * 1024 * 1024;
    RmaBufSliceLite loc(0x1111, dataSize, 1, 1);
    RmtRmaBufSliceLite rmt(0x2222, dataSize, 1, 1, 1, UINT32_MAX);
    SqeConfigLite cfg;
    ConnLiteOperationOut out;
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));
    ubConn.BatchOneSidedRead({loc}, {rmt}, cfg, stream, out);
}

TEST_F(AicpuUbConnLiteTest, test_UBConnLite_BatchOneSidedWrite)
{
    UbJettyLiteId id(1, 1, 1);
    u8 sqVa[1024];
    u64 sqVAaddr = reinterpret_cast<u64>(sqVa);
    UbJettyLiteAttr attr(1, sqVAaddr, 4, 1, false);
    Eid rmtEid;
    rmtEid.raw[0] = 1;

    UbConnLite ubConn(id, attr, rmtEid);
    ubConn.SetSqPiCi(&mockPiSlot, &mockCiSlot);
    u32 dataSize = 400 * 1024 * 1024;
    RmaBufSliceLite loc(0x1111, dataSize, 1, 1);
    RmtRmaBufSliceLite rmt(0x2222, dataSize, 1, 1, 1, UINT32_MAX);
    SqeConfigLite cfg;
    ConnLiteOperationOut out;
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));
    ubConn.BatchOneSidedWrite({loc}, {rmt}, cfg, stream, out);
}

/* ---------- owner 位相位 ---------- */

TEST_F(AicpuUbConnLiteTest, BatchOwnerBit_PhaseAlignedWithRingTailSlot)
{
    UbJettyLiteId id(1, 1, 1);
    alignas(16) u8 sqVa[1024];
    u64 sqVAaddr = reinterpret_cast<u64>(sqVa);
    const u32 sqDepth = 4; // 环深4: 落位槽 0..3, 环尾槽 = sqDepth-1 = 3
    UbJettyLiteAttr attr(1, sqVAaddr, sqDepth, 1, false);
    Eid rmtEid;
    rmtEid.raw[0] = 1;

    UbConnLite ubConn(id, attr, rmtEid);
    ubConn.SetSqPiCi(&mockPiSlot, &mockCiSlot);
    const u32 seedPi = 1; // 取非零起点，锁定相位而非依赖 pi 从 0 开始
    ubConn.SetPi(seedPi);
    ubConn.EnableWqeTasks();

    // 单 slice = 4*256MB + 64B，循环段 4 条 + 尾段 1 条，共 5 条 SQE
    const u64 dataSize = 4ULL * 256 * 1024 * 1024 + 64;
    RmaBufSliceLite loc(0x1111, dataSize, 1, 1);
    RmtRmaBufSliceLite rmt(0x2222, dataSize, 1, 1, 1, UINT32_MAX);
    SqeConfigLite cfg;
    ConnLiteOperationOut out;
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));
    ubConn.BatchOneSidedWrite({loc}, {rmt}, cfg, stream, out);
    EXPECT_EQ(seedPi + 5, static_cast<u32>(ubConn.GetPi()));
    EXPECT_EQ(ubConn.GetPi(), out.pi);

    // 批量路径物理落位：owner==1 的 SQE 恰在环尾槽 sqDepth-1 上，其余槽位全 0
    // （旧代码用自增后的 pi 判 owner，会把 owner=1 错置在环尾前一格的槽上）
    for (u32 slot = 0; slot < sqDepth; slot++) {
        UdmaSqeWrite* landed = reinterpret_cast<UdmaSqeWrite*>(sqVa + slot * 64);
        EXPECT_EQ(slot == sqDepth - 1 ? 1u : 0u, static_cast<u32>(landed->comm.owner));
    }

    // 非 batch 基准路径续写 2 条，与批量路径共用同一条 pi
    RmaBufSliceLite locSmall(0x3333, 64, 1, 1);
    RmtRmaBufSliceLite rmtSmall(0x4444, 64, 1, 1, 1, UINT32_MAX);
    ubConn.SetCi(static_cast<u16>(ubConn.GetPi() - 2)); // 抬 ci 避开 Write 路径的 SQ 满等待
    MOCKER(memset_s).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER(memcpy_s).stubs().with(mockcpp::any()).will(returnValue(0));
    EXPECT_EQ(ubConn.Write(locSmall, rmtSmall, cfg, stream, out), HCCL_SUCCESS);
    EXPECT_EQ(ubConn.Write(locSmall, rmtSmall, cfg, stream, out), HCCL_SUCCESS);

    // 两路径统一相位：第 i 条 SQE 填充时 pi = seedPi+i，owner==1 当且仅当其落位槽为环尾槽
    const std::vector<WqeTask> tasks = ubConn.GetWqeTasks();
    ASSERT_EQ(7u, tasks.size()); // 批量 5 条 + 非批量 2 条
    for (size_t i = 0; i < tasks.size(); i++) {
        u32 slot = (seedPi + static_cast<u32>(i)) % sqDepth;
        EXPECT_EQ(slot == sqDepth - 1 ? 1u : 0u, static_cast<u32>(tasks[i].wqeWrite.comm.owner));
    }
}

/* ---------- SetPi/SetCi 绑定前默认槽 ---------- */

TEST_F(AicpuUbConnLiteTest, SetPiSetCi_BeforeBind_WriteDefaultSlot)
{
    UbJettyLiteId id(1, 1, 1);
    UbJettyLiteAttr attr(1, 1, 8, 1, false);
    Eid rmtEid;
    UbConnLite ubConn(id, attr, rmtEid);
    EXPECT_EQ(0u, static_cast<u32>(ubConn.GetPi()));
    EXPECT_EQ(0u, static_cast<u32>(ubConn.GetCi()));
    // 未绑定时写默认槽并读回，与 pi{0}/ci{0} 初值语义一致
    ubConn.SetPi(3);
    ubConn.SetCi(4);
    EXPECT_EQ(3u, static_cast<u32>(ubConn.GetPi()));
    EXPECT_EQ(4u, static_cast<u32>(ubConn.GetCi()));
}

/* ---------- GetInflight / UpdateCi / CheckOverflow (反压新增接口) ---------- */

TEST_F(AicpuUbConnLiteTest, GetInflight_ExpectCorrect)
{
    UbJettyLiteId id(1, 1, 1);
    UbJettyLiteAttr attr(1, 1, 8, 1, false);
    Eid rmtEid;
    UbConnLite ubConn(id, attr, rmtEid);
    ubConn.SetSqPiCi(&mockPiSlot, &mockCiSlot);
    // pi > ci: 正常差值
    ubConn.SetPi(20);
    ubConn.SetCi(5);
    EXPECT_EQ(15u, ubConn.GetInflight());
    // pi < ci: u16回绕
    ubConn.SetPi(5);
    ubConn.SetCi(65530);
    EXPECT_EQ(11u, ubConn.GetInflight());
    // pi == ci: 无inflight
    ubConn.SetPi(100);
    ubConn.SetCi(100);
    EXPECT_EQ(0u, ubConn.GetInflight());
}

/* ---------- UpdateCi ---------- */

TEST_F(AicpuUbConnLiteTest, UpdateCi_ExpectCorrect)
{
    UbJettyLiteId id(1, 1, 1);
    UbJettyLiteAttr attr(1, 1, 8, 1, false);
    Eid rmtEid;
    UbConnLite ubConn(id, attr, rmtEid);
    ubConn.SetSqPiCi(&mockPiSlot, &mockCiSlot);
    // 顺序消费: seq 0/1连续, ci推进
    std::vector<std::pair<u16, u16>> slots = {{0, 10}, {1, 20}};
    ubConn.UpdateCi(slots.data(), slots.size());
    EXPECT_EQ(20u, ubConn.GetCi());

    // piValue回绕 + 乱序暂存: seq 2连续消费(65530), seq 4暂存
    slots = {{2, 65530}, {4, 40}};
    ubConn.UpdateCi(slots.data(), slots.size());
    EXPECT_EQ(65530u, ubConn.GetCi());

    // 补齐后连续出队: seq 3补齐(65535), 乱序的seq 4(40)随之消费
    slots = {{3, 65535}};
    ubConn.UpdateCi(slots.data(), slots.size());
    EXPECT_EQ(40u, ubConn.GetCi());

    // 重复seqIdx: 取第一个值
    slots = {{5, 10}, {5, 99}};
    ubConn.UpdateCi(slots.data(), slots.size());
    EXPECT_EQ(10u, ubConn.GetCi());

    // 空输入: 不变化
    ubConn.UpdateCi(nullptr, 0);
    EXPECT_EQ(10u, ubConn.GetCi());
}

/* ---------- CheckOverflow ---------- */

TEST_F(AicpuUbConnLiteTest, CheckOverflow_ExpectCorrect)
{
    UbJettyLiteId id(1, 1, 1);
    UbJettyLiteAttr attr(1, 1, 8, 1, false);
    Eid rmtEid;
    UbConnLite ubConn(id, attr, rmtEid);
    ubConn.SetSqPiCi(&mockPiSlot, &mockCiSlot);
    ubConn.sqDepth_ = 8;
    ubConn.SetPi(5);
    ubConn.SetCi(2); // inflight=3

    // cache版: 3+3<=8, 3+6>8
    EXPECT_FALSE(ubConn.CheckOverflow(static_cast<u32>(3)));
    EXPECT_TRUE(ubConn.CheckOverflow(static_cast<u32>(6)));
}
