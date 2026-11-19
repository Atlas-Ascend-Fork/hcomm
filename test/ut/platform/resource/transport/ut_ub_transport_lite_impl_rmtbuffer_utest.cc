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
#include <stdio.h>

#include "hccl/base.h"
#include <hccl/hccl_types.h>

#ifndef private
#define private public
#define protected public
#endif

#include "binary_stream.h"
#include "ub_transport_lite_impl.h"

#undef private
#undef protected

#include "endpoint_pair/channel/ub_unique_id_pub.h" // UB_UNIQUE_ID_EXT_*：V2 扩展区契约（零依赖薄头）
#include "dev_ep_mem_table.h"
#include "internal_exception.h"
#include <new>

using namespace std;
using namespace Hccl;

class UbTransportLiteImplRmtbuffer_UT : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "\033[36m--UbTransportLiteImplRmtbuffer_UT SetUP--\033[0m" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "\033[36m--UbTransportLiteImplRmtbuffer_UT TearDown--\033[0m" << std::endl;
    }
    virtual void SetUp() {}
    virtual void TearDown() { GlobalMockObject::verify(); }

    // 构造 UbTransportLiteImpl 反序列化所需的完整序列化数据（V2 布局：扩展区头 + 表指针）
    static std::vector<char> BuildTransportData(u32 type, u32 notifyNum, u32 bufferNum, u32 rmtbufferNum, u32 connNum)
    {
        BinaryStream binaryStream;
        binaryStream << type;
        binaryStream << notifyNum;
        binaryStream << bufferNum;
        binaryStream << rmtbufferNum;
        binaryStream << connNum;
        binaryStream << UB_UNIQUE_ID_EXT_MAGIC_WORD;
        binaryStream << UB_UNIQUE_ID_EXT_VERSION;
        binaryStream << static_cast<u64>(0); // devEpMemTablePtr，0=无表
        std::vector<char> emptyVector;
        binaryStream << emptyVector; // notifyUniqueIds
        binaryStream << emptyVector; // rmtNotifyUniqueIds
        binaryStream << emptyVector; // locBufferUniqueIds
        binaryStream << emptyVector; // rmtBufferUniqueIds
        binaryStream << emptyVector; // drainBufferUniqueIds
        binaryStream << emptyVector; // connUniqueIds
        std::vector<char> data;
        binaryStream.Dump(data);
        return data;
    }

    // 0 conn 的最小 device transport，构造期不触碰任何连接
    static std::vector<char> BuildEmptyTransportData()
    {
        return BuildTransportData(static_cast<u32>(TransportType::UB), 0, 0, 0, 0);
    }

    // V1 布局（旧格式）：固定头 + 4 段 vector，无扩展区
    static std::vector<char> BuildV1TransportData(u32 type, u32 notifyNum, u32 bufferNum, u32 rmtbufferNum, u32 connNum)
    {
        BinaryStream binaryStream;
        binaryStream << type;
        binaryStream << notifyNum;
        binaryStream << bufferNum;
        binaryStream << rmtbufferNum;
        binaryStream << connNum;
        std::vector<char> emptyVector;
        binaryStream << emptyVector; // notifyUniqueIds
        binaryStream << emptyVector; // rmtNotifyUniqueIds
        binaryStream << emptyVector; // rmtBufferUniqueIds
        binaryStream << emptyVector; // connUniqueIds
        std::vector<char> data;
        binaryStream.Dump(data);
        return data;
    }

    // V2 布局：扩展区头 + 指定表指针 + locBuffer 段含 1 条 LocUbBufLite，其余段为空
    static std::vector<char> BuildTransportDataWithLocBuffer(u64 devEpMemTablePtr)
    {
        BinaryStream binaryStream;
        binaryStream << static_cast<u32>(TransportType::UB);
        binaryStream << static_cast<u32>(0); // notifyNum
        binaryStream << static_cast<u32>(1); // bufferNum
        binaryStream << static_cast<u32>(0); // rmtbufferNum
        binaryStream << static_cast<u32>(0); // connNum
        binaryStream << UB_UNIQUE_ID_EXT_MAGIC_WORD;
        binaryStream << UB_UNIQUE_ID_EXT_VERSION;
        binaryStream << devEpMemTablePtr;
        std::vector<char> emptyVector;
        binaryStream << emptyVector; // notifyUniqueIds
        binaryStream << emptyVector; // rmtNotifyUniqueIds
        {
            BinaryStream elemStream;
            elemStream << static_cast<u64>(0x1000); // addr
            elemStream << static_cast<u64>(256);    // size
            elemStream << static_cast<u32>(11);     // tokenId
            elemStream << static_cast<u32>(22);     // tokenValue
            std::vector<char> elemData;
            elemStream.Dump(elemData);
            binaryStream << elemData; // locBufferUniqueIds：1 条 LocUbBufLite
        }
        binaryStream << emptyVector; // rmtBufferUniqueIds
        binaryStream << emptyVector; // drainBufferUniqueIds
        binaryStream << emptyVector; // connUniqueIds
        std::vector<char> data;
        binaryStream.Dump(data);
        return data;
    }
};

TEST_F(
    UbTransportLiteImplRmtbuffer_UT,
    Ut_HeaderSerialization_When_RmtbufferDifferentFromBuffer_Expect_FiveFieldsDeserialized)
{
    u32 type = static_cast<u32>(TransportType::UB);
    u32 notifyNum = 2;
    u32 bufferNum = 3;
    u32 rmtbufferNum = 5;
    u32 connNum = 1;

    BinaryStream binaryStream;
    binaryStream << type;
    binaryStream << notifyNum;
    binaryStream << bufferNum;
    binaryStream << rmtbufferNum;
    binaryStream << connNum;

    std::vector<char> data;
    binaryStream.Dump(data);

    BinaryStream readStream(data);
    u32 readType, readNotifyNum, readBufferNum, readRmtbufferNum, readConnNum;
    readStream >> readType;
    readStream >> readNotifyNum;
    readStream >> readBufferNum;
    readStream >> readRmtbufferNum;
    readStream >> readConnNum;

    EXPECT_EQ(readType, type);
    EXPECT_EQ(readNotifyNum, notifyNum);
    EXPECT_EQ(readBufferNum, bufferNum);
    EXPECT_EQ(readRmtbufferNum, rmtbufferNum);
    EXPECT_EQ(readConnNum, connNum);
}

TEST_F(UbTransportLiteImplRmtbuffer_UT, Ut_HeaderSerialization_When_RmtbufferNumIsZero_Expect_ZeroDeserialized)
{
    u32 type = static_cast<u32>(TransportType::UB);
    u32 notifyNum = 2;
    u32 bufferNum = 3;
    u32 rmtbufferNum = 0;
    u32 connNum = 1;

    BinaryStream binaryStream;
    binaryStream << type;
    binaryStream << notifyNum;
    binaryStream << bufferNum;
    binaryStream << rmtbufferNum;
    binaryStream << connNum;

    std::vector<char> data;
    binaryStream.Dump(data);

    BinaryStream readStream(data);
    u32 readType, readNotifyNum, readBufferNum, readRmtbufferNum, readConnNum;
    readStream >> readType;
    readStream >> readNotifyNum;
    readStream >> readBufferNum;
    readStream >> readRmtbufferNum;
    readStream >> readConnNum;

    EXPECT_EQ(readRmtbufferNum, 0);
    EXPECT_EQ(readBufferNum, 3);
}

TEST_F(UbTransportLiteImplRmtbuffer_UT, Ut_HeaderSerialization_When_RmtbufferEqualsBuffer_Expect_SameValue)
{
    u32 type = static_cast<u32>(TransportType::UB);
    u32 notifyNum = 2;
    u32 bufferNum = 3;
    u32 rmtbufferNum = 3;
    u32 connNum = 1;

    BinaryStream binaryStream;
    binaryStream << type;
    binaryStream << notifyNum;
    binaryStream << bufferNum;
    binaryStream << rmtbufferNum;
    binaryStream << connNum;

    std::vector<char> data;
    binaryStream.Dump(data);

    BinaryStream readStream(data);
    u32 readType, readNotifyNum, readBufferNum, readRmtbufferNum, readConnNum;
    readStream >> readType;
    readStream >> readNotifyNum;
    readStream >> readBufferNum;
    readStream >> readRmtbufferNum;
    readStream >> readConnNum;

    EXPECT_EQ(readRmtbufferNum, readBufferNum);
}

TEST_F(UbTransportLiteImplRmtbuffer_UT, Ut_RmtbufferNumMember_When_Default_Expect_Zero)
{
    u32 type = static_cast<u32>(TransportType::UB);
    std::vector<char> data = BuildTransportData(type, 0, 0, 0, 0);
    UbTransportLiteImpl transportDev(data);
    EXPECT_EQ(transportDev.rmtbufferNum, 0);
}

// 旧格式（V1）数据进入 V2 解析路径（Init）：扩展区头校验快速失败，
// 而非位置错位把 vector 长度前缀当表指针、后续向野地址 Upsert
TEST_F(UbTransportLiteImplRmtbuffer_UT, Ut_Init_When_V1Blob_Expect_FastFailThrow)
{
    std::vector<char> data = BuildV1TransportData(static_cast<u32>(TransportType::UB), 0, 0, 0, 0);
    EXPECT_THROW(UbTransportLiteImpl transportDev(data), InternalException);
}

// 扩展区版本未知：快速失败，不按当前布局盲解
TEST_F(UbTransportLiteImplRmtbuffer_UT, Ut_Init_When_UnknownExtVersion_Expect_FastFailThrow)
{
    BinaryStream binaryStream;
    binaryStream << static_cast<u32>(TransportType::UB);
    u32 zero = 0;
    binaryStream << zero; // notifyNum
    binaryStream << zero; // bufferNum
    binaryStream << zero; // rmtbufferNum
    binaryStream << zero; // connNum
    binaryStream << UB_UNIQUE_ID_EXT_MAGIC_WORD;
    binaryStream << (UB_UNIQUE_ID_EXT_VERSION + 1U); // 未知版本
    binaryStream << static_cast<u64>(0);             // devEpMemTablePtr
    std::vector<char> emptyVector;
    binaryStream << emptyVector; // notifyUniqueIds
    binaryStream << emptyVector; // rmtNotifyUniqueIds
    binaryStream << emptyVector; // locBufferUniqueIds
    binaryStream << emptyVector; // rmtBufferUniqueIds
    binaryStream << emptyVector; // drainBufferUniqueIds
    binaryStream << emptyVector; // connUniqueIds
    std::vector<char> data;
    binaryStream.Dump(data);
    EXPECT_THROW(UbTransportLiteImpl transportDev(data), InternalException);
}

// X″ 契约：locBuffer 段保留（Urma host 实体预解析消费），device 侧仅推进流位置、不补表。
// 构造成功即证明段被正确消费（错位会把段内容当后续 vector 长度前缀），表保持空
TEST_F(UbTransportLiteImplRmtbuffer_UT, Ut_ParseLocBuffer_When_TableValid_Expect_StreamAdvancedNoTableWrite)
{
    Hccl::DevEpMemTable table;
    std::vector<char> data = BuildTransportDataWithLocBuffer(ReinterpretAs<u64>(&table));
    UbTransportLiteImpl transportDev(data);

    // 表唯一写入器 = Reg/Unreg kernel 路径（建链不写表）——经公共读路径观测空表（B1 快照化后
    // bufferMap_ 成员已移除，空快照的 FindFirst 必然 miss）
    LocUbBufLite out;
    EXPECT_FALSE(table.FindFirst(out));
    EXPECT_EQ(table.abi.entryNum, 0U);
}

// 表指针悬垂（endpoint 先于 channel 销毁）：构造路径不触表（CheckAbi 哨兵职责已移至
// 运行期 BuildLocRmaBufferLite 查表路径），不向已释放内存写 map 节点
TEST_F(UbTransportLiteImplRmtbuffer_UT, Ut_ParseLocBuffer_When_TableBadAbi_Expect_ConstructionNoTouch)
{
    Hccl::DevEpMemTable table;
    table.abi.magic = 0; // 模拟表已销毁
    std::vector<char> data = BuildTransportDataWithLocBuffer(ReinterpretAs<u64>(&table));

    EXPECT_NO_THROW(UbTransportLiteImpl transportDev(data));
    LocUbBufLite out;
    EXPECT_FALSE(table.FindFirst(out));
}

// 析构即失效：delete 之后悬垂表指针的 CheckAbi 必须失败（use-after-free 兜底）
TEST_F(UbTransportLiteImplRmtbuffer_UT, Ut_DevEpMemTable_When_Destructed_Expect_CheckAbiFail)
{
    alignas(Hccl::DevEpMemTable) unsigned char raw[sizeof(Hccl::DevEpMemTable)];
    auto* table = new (raw) Hccl::DevEpMemTable();
    EXPECT_TRUE(table->CheckAbi());

    table->~DevEpMemTable(); // 模拟 kernel delete：析构后堆块未复用，悬垂指针仍可读
    EXPECT_FALSE(table->CheckAbi());
}

TEST_F(UbTransportLiteImplRmtbuffer_UT, Ut_RmtbufferNumMember_When_RmtbufferDifferentFromBuffer_Expect_IndependentValue)
{
    u32 type = static_cast<u32>(TransportType::UB);
    std::vector<char> data = BuildTransportData(type, 2, 3, 5, 1);
    UbTransportLiteImpl transportDev(data);
    transportDev.notifyNum = 2;
    transportDev.bufferNum = 3;
    transportDev.rmtbufferNum = 5;
    transportDev.connNum = 1;

    EXPECT_EQ(transportDev.bufferNum, 3);
    EXPECT_EQ(transportDev.rmtbufferNum, 5);
    EXPECT_NE(transportDev.bufferNum, transportDev.rmtbufferNum);
}

// ==================== UpdateMemInfoEntry：按地址 upsert ====================

namespace {
RmaBufEntryLite MakeRmaBufEntry(u64 addr, u64 size, u32 tokenId, u32 tokenValue, u32 notifyId)
{
    RmaBufEntryLite entry;
    entry.addr = addr;
    entry.size = size;
    entry.tokenId = tokenId;
    entry.tokenValue = tokenValue;
    entry.notifyId = notifyId;
    return entry;
}
} // namespace

TEST_F(UbTransportLiteImplRmtbuffer_UT, Ut_UpdateMemInfoEntry_When_AddrAbsent_Expect_Insert)
{
    std::vector<char> data = BuildEmptyTransportData();
    UbTransportLiteImpl transportDev(data);
    RmaBufEntryLite entry = MakeRmaBufEntry(0x1000, 512, 11, 22, 33);

    EXPECT_EQ(transportDev.UpdateMemInfoEntry(entry), HCCL_SUCCESS);
    ASSERT_EQ(transportDev.rmtBufferMap.size(), 1U);
    auto it = transportDev.rmtBufferMap.find(0x1000);
    ASSERT_NE(it, transportDev.rmtBufferMap.end());
    EXPECT_EQ(it->second.addr, 0x1000U);
    EXPECT_EQ(it->second.size, 512U);
    EXPECT_EQ(it->second.tokenId, 11U);
    EXPECT_EQ(it->second.tokenValue, 22U);
    EXPECT_EQ(it->second.notifyId, 33U);
}

TEST_F(UbTransportLiteImplRmtbuffer_UT, Ut_UpdateMemInfoEntry_When_AddrPresent_Expect_Overwrite)
{
    std::vector<char> data = BuildEmptyTransportData();
    UbTransportLiteImpl transportDev(data);
    transportDev.UpdateMemInfoEntry(MakeRmaBufEntry(0x1000, 512, 11, 22, 33));

    // host 侧下发的是当前全量表，同一地址的新条目覆盖旧值
    EXPECT_EQ(transportDev.UpdateMemInfoEntry(MakeRmaBufEntry(0x1000, 1024, 44, 55, 66)), HCCL_SUCCESS);
    ASSERT_EQ(transportDev.rmtBufferMap.size(), 1U);
    auto it = transportDev.rmtBufferMap.find(0x1000);
    ASSERT_NE(it, transportDev.rmtBufferMap.end());
    EXPECT_EQ(it->second.size, 1024U);
    EXPECT_EQ(it->second.tokenId, 44U);
    EXPECT_EQ(it->second.tokenValue, 55U);
    EXPECT_EQ(it->second.notifyId, 66U);
}

TEST_F(UbTransportLiteImplRmtbuffer_UT, Ut_UpdateMemInfoEntry_When_SmallerEntry_Expect_AllGatherSlicesRemainAccessible)
{
    std::vector<char> data = BuildEmptyTransportData();
    UbTransportLiteImpl transportDev(data);
    const u64 addr = 0x120099000000;
    RmaBufEntryLite largeEntry = MakeRmaBufEntry(addr, 128, 11, 22, 33);
    RmaBufEntryLite smallEntry = MakeRmaBufEntry(addr, 32, 44, 55, 66);

    ASSERT_EQ(transportDev.UpdateMemInfoEntry(largeEntry), HCCL_SUCCESS);
    ASSERT_EQ(transportDev.UpdateMemInfoEntry(smallEntry), HCCL_SUCCESS);
    ASSERT_EQ(transportDev.rmtBufferMap.size(), 1U);
    auto it = transportDev.rmtBufferMap.find(addr);
    ASSERT_NE(it, transportDev.rmtBufferMap.end());
    EXPECT_EQ(it->second.size, 128U);
    EXPECT_EQ(it->second.tokenId, 11U);
    EXPECT_EQ(it->second.tokenValue, 22U);
    EXPECT_EQ(it->second.notifyId, 33U);

    for (u64 offset : {0ULL, 64ULL}) {
        RmtRmaBufSliceLite slice;
        ASSERT_EQ(transportDev.GetRmtRmaBufSliceLite(Buffer(addr + offset, 64), slice), HCCL_SUCCESS);
        EXPECT_EQ(slice.GetAddr(), addr + offset);
        EXPECT_EQ(slice.GetSize(), 64U);
        EXPECT_EQ(slice.GetTokenId(), 11U);
        EXPECT_EQ(slice.GetTokenValue(), 22U);
    }
}

TEST_F(UbTransportLiteImplRmtbuffer_UT, Ut_UpdateMemInfoEntry_When_SameEntryTwice_Expect_Idempotent)
{
    std::vector<char> data = BuildEmptyTransportData();
    UbTransportLiteImpl transportDev(data);
    RmaBufEntryLite entry = MakeRmaBufEntry(0x1000, 512, 11, 22, 33);

    transportDev.UpdateMemInfoEntry(entry);
    EXPECT_EQ(transportDev.UpdateMemInfoEntry(entry), HCCL_SUCCESS);
    ASSERT_EQ(transportDev.rmtBufferMap.size(), 1U);
    auto it = transportDev.rmtBufferMap.find(0x1000);
    ASSERT_NE(it, transportDev.rmtBufferMap.end());
    EXPECT_EQ(it->second.size, 512U);
    EXPECT_EQ(it->second.notifyId, 33U);
}

TEST_F(UbTransportLiteImplRmtbuffer_UT, Ut_UpdateMemInfoEntry_When_Called_Expect_RmtBufferVecUntouched)
{
    std::vector<char> data = BuildEmptyTransportData();
    UbTransportLiteImpl transportDev(data);

    transportDev.UpdateMemInfoEntry(MakeRmaBufEntry(0x1000, 512, 11, 22, 33));
    transportDev.UpdateMemInfoEntry(MakeRmaBufEntry(0x2000, 512, 11, 22, 33));
    EXPECT_EQ(transportDev.rmtBufferMap.size(), 2U);
    // 按下标的视图不随本流程更新，消费方按地址查 rmtBufferMap
    EXPECT_EQ(transportDev.rmtBufferVec.size(), 0U);
}

TEST_F(UbTransportLiteImplRmtbuffer_UT, Ut_UpdateMemInfoEntry_When_AddrIsZero_Expect_Insert)
{
    std::vector<char> data = BuildEmptyTransportData();
    UbTransportLiteImpl transportDev(data);

    EXPECT_EQ(transportDev.UpdateMemInfoEntry(MakeRmaBufEntry(0, 512, 11, 22, 33)), HCCL_SUCCESS);
    ASSERT_EQ(transportDev.rmtBufferMap.size(), 1U);
    EXPECT_NE(transportDev.rmtBufferMap.find(0), transportDev.rmtBufferMap.end());
}

TEST_F(UbTransportLiteImplRmtbuffer_UT, Ut_UpdateMemInfoEntry_When_TwoAddrs_Expect_BothPresent)
{
    std::vector<char> data = BuildEmptyTransportData();
    UbTransportLiteImpl transportDev(data);

    transportDev.UpdateMemInfoEntry(MakeRmaBufEntry(0x1000, 512, 11, 22, 33));
    transportDev.UpdateMemInfoEntry(MakeRmaBufEntry(0x2000, 256, 12, 23, 34));
    ASSERT_EQ(transportDev.rmtBufferMap.size(), 2U);
    EXPECT_NE(transportDev.rmtBufferMap.find(0x1000), transportDev.rmtBufferMap.end());
    auto it = transportDev.rmtBufferMap.find(0x2000);
    ASSERT_NE(it, transportDev.rmtBufferMap.end());
    EXPECT_EQ(it->second.size, 256U);
    EXPECT_EQ(it->second.notifyId, 34U);
}

TEST_F(UbTransportLiteImplRmtbuffer_UT, Ut_ParseRmtBufferVecLogic_When_BufferType_Expect_UseRmtbufferNum)
{
    u32 notifyNum = 2;
    u32 bufferNum = 3;
    u32 rmtbufferNum = 5;

    u32 numForBuffer = 0;
    UbTransportLiteImpl::RmaUbBufType rmtType = UbTransportLiteImpl::RmaUbBufType::BUFFER;
    if (rmtType == UbTransportLiteImpl::RmaUbBufType::NOTIFY) {
        numForBuffer = notifyNum;
    } else {
        numForBuffer = rmtbufferNum;
    }

    EXPECT_EQ(numForBuffer, rmtbufferNum);
    EXPECT_NE(numForBuffer, bufferNum);
}

TEST_F(UbTransportLiteImplRmtbuffer_UT, Ut_ParseRmtBufferVecLogic_When_NotifyType_Expect_UseNotifyNum)
{
    u32 notifyNum = 2;
    u32 bufferNum = 3;
    u32 rmtbufferNum = 5;

    u32 numForNotify = 0;
    UbTransportLiteImpl::RmaUbBufType rmtType = UbTransportLiteImpl::RmaUbBufType::NOTIFY;
    if (rmtType == UbTransportLiteImpl::RmaUbBufType::NOTIFY) {
        numForNotify = notifyNum;
    } else {
        numForNotify = rmtbufferNum;
    }

    EXPECT_EQ(numForNotify, notifyNum);
}

TEST_F(UbTransportLiteImplRmtbuffer_UT, Ut_ParseRmtBufferVecLogic_When_RmtbufferNumIsZero_Expect_NumIsZero)
{
    u32 notifyNum = 2;
    u32 bufferNum = 3;
    u32 rmtbufferNum = 0;

    u32 numForBuffer = 0;
    UbTransportLiteImpl::RmaUbBufType rmtType = UbTransportLiteImpl::RmaUbBufType::BUFFER;
    if (rmtType == UbTransportLiteImpl::RmaUbBufType::NOTIFY) {
        numForBuffer = notifyNum;
    } else {
        numForBuffer = rmtbufferNum;
    }

    EXPECT_EQ(numForBuffer, 0);
}

TEST_F(UbTransportLiteImplRmtbuffer_UT, Ut_FullPacketDeserialize_When_RmtbufferDifferentFromBuffer_Expect_HeaderCorrect)
{
    u32 type = static_cast<u32>(TransportType::UB);
    u32 notifyNum = 2;
    u32 bufferNum = 3;
    u32 rmtbufferNum = 5;
    u32 connNum = 1;

    BinaryStream binaryStream;
    binaryStream << type;
    binaryStream << notifyNum;
    binaryStream << bufferNum;
    binaryStream << rmtbufferNum;
    binaryStream << connNum;

    std::vector<char> notifyUniqueIds(notifyNum * 4, 0);
    binaryStream << notifyUniqueIds;

    std::vector<char> rmtNotifyUniqueIds;
    for (u32 i = 0; i < notifyNum; i++) {
        BinaryStream elemStream;
        elemStream << static_cast<u64>(0x1000 + i);
        elemStream << static_cast<u64>(256);
        elemStream << static_cast<u32>(10 + i);
        elemStream << static_cast<u32>(1);
        std::vector<char> elemData;
        elemStream.Dump(elemData);
        rmtNotifyUniqueIds.insert(rmtNotifyUniqueIds.end(), elemData.begin(), elemData.end());
    }
    binaryStream << rmtNotifyUniqueIds;

    std::vector<char> rmtBufferUniqueIds;
    for (u32 i = 0; i < rmtbufferNum; i++) {
        BinaryStream elemStream;
        elemStream << static_cast<u64>(0x2000 + i);
        elemStream << static_cast<u64>(512);
        elemStream << static_cast<u32>(20 + i);
        elemStream << static_cast<u32>(2);
        std::vector<char> elemData;
        elemStream.Dump(elemData);
        rmtBufferUniqueIds.insert(rmtBufferUniqueIds.end(), elemData.begin(), elemData.end());
    }
    binaryStream << rmtBufferUniqueIds;

    std::vector<char> connUniqueIds(connNum * 4, 0);
    binaryStream << connUniqueIds;

    std::vector<char> packet;
    binaryStream.Dump(packet);

    BinaryStream readStream(packet);
    u32 readType, readNotifyNum, readBufferNum, readRmtbufferNum, readConnNum;
    readStream >> readType;
    readStream >> readNotifyNum;
    readStream >> readBufferNum;
    readStream >> readRmtbufferNum;
    readStream >> readConnNum;

    EXPECT_EQ(readRmtbufferNum, 5);
    EXPECT_EQ(readBufferNum, 3);

    std::vector<char> readNotifyUniqueIds;
    readStream >> readNotifyUniqueIds;
    EXPECT_EQ(readNotifyUniqueIds.size(), notifyNum * 4);

    std::vector<char> readRmtNotifyUniqueIds;
    readStream >> readRmtNotifyUniqueIds;

    std::vector<char> readRmtBufferUniqueIds;
    readStream >> readRmtBufferUniqueIds;
}

TEST_F(
    UbTransportLiteImplRmtbuffer_UT, Ut_FullPacketDeserializeV2_When_RmtbufferDifferentFromBuffer_Expect_HeaderCorrect)
{
    u32 type = static_cast<u32>(TransportType::UB);
    u32 notifyNum = 2;
    u32 bufferNum = 3;
    u32 rmtbufferNum = 5;
    u32 connNum = 1;

    BinaryStream binaryStream;
    binaryStream << type;
    binaryStream << notifyNum;
    binaryStream << bufferNum;
    binaryStream << rmtbufferNum;
    binaryStream << connNum;

    std::vector<char> notifyUniqueIds(notifyNum * 4, 0);
    binaryStream << notifyUniqueIds;

    std::vector<char> rmtNotifyUniqueIds;
    for (u32 i = 0; i < notifyNum; i++) {
        BinaryStream elemStream;
        elemStream << static_cast<u64>(0x1000 + i);
        elemStream << static_cast<u64>(256);
        elemStream << static_cast<u32>(10 + i);
        elemStream << static_cast<u32>(1);
        std::vector<char> elemData;
        elemStream.Dump(elemData);
        rmtNotifyUniqueIds.insert(rmtNotifyUniqueIds.end(), elemData.begin(), elemData.end());
    }
    binaryStream << rmtNotifyUniqueIds;

    std::vector<char> locBufferUniqueIds;
    for (u32 i = 0; i < bufferNum; i++) {
        BinaryStream elemStream;
        elemStream << static_cast<u64>(0x3000 + i);
        elemStream << static_cast<u64>(1024);
        elemStream << static_cast<u32>(30 + i);
        elemStream << static_cast<u32>(3);
        std::vector<char> elemData;
        elemStream.Dump(elemData);
        locBufferUniqueIds.insert(locBufferUniqueIds.end(), elemData.begin(), elemData.end());
    }
    binaryStream << locBufferUniqueIds;

    std::vector<char> rmtBufferUniqueIds;
    for (u32 i = 0; i < rmtbufferNum; i++) {
        BinaryStream elemStream;
        elemStream << static_cast<u64>(0x2000 + i);
        elemStream << static_cast<u64>(512);
        elemStream << static_cast<u32>(20 + i);
        elemStream << static_cast<u32>(2);
        std::vector<char> elemData;
        elemStream.Dump(elemData);
        rmtBufferUniqueIds.insert(rmtBufferUniqueIds.end(), elemData.begin(), elemData.end());
    }
    binaryStream << rmtBufferUniqueIds;

    std::vector<char> connUniqueIds(connNum * 4, 0);
    binaryStream << connUniqueIds;

    std::vector<char> packet;
    binaryStream.Dump(packet);

    BinaryStream readStream(packet);
    u32 readType, readNotifyNum, readBufferNum, readRmtbufferNum, readConnNum;
    readStream >> readType;
    readStream >> readNotifyNum;
    readStream >> readBufferNum;
    readStream >> readRmtbufferNum;
    readStream >> readConnNum;

    EXPECT_EQ(readRmtbufferNum, 5);
    EXPECT_EQ(readBufferNum, 3);

    std::vector<char> readNotifyUniqueIds;
    readStream >> readNotifyUniqueIds;

    std::vector<char> readRmtNotifyUniqueIds;
    readStream >> readRmtNotifyUniqueIds;

    std::vector<char> readLocBufferUniqueIds;
    readStream >> readLocBufferUniqueIds;

    std::vector<char> readRmtBufferUniqueIds;
    readStream >> readRmtBufferUniqueIds;
}
