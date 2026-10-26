/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../ut_hcomm_base.h"
#include "../../../comms/endpoint_pairs/channels/aicpu/device/aicpu_pack_data_helper.h"
#include "coll_comm_aicpu_mgr.h"
#include "coll_comm_aicpu_kernel_adpt.h"
#include "coll_comm_aicpu_kernel.h"
#include "aicpu/aicpu_ts_channel_helper.h"

#ifndef private
#define private public
#define protected public
#endif
// aicpu_channel_process.h 需在放开区间内展开：MemUpdate 在册校验需向 transportMap_ 登记栈 transport
#include "aicpu/device/aicpu_channel_process.h"
#include "ub_transport_lite_impl.h"
#undef private
#undef protected

// aicpu_indop_process 已解散，原用例迁移到 CollCommAicpuMgr 和 kernel adapter。

class TestCollCommAicpuMgr : public TestHcommCAdptBase {
public:
    void SetUp() override { TestHcommCAdptBase::SetUp(); }
    void TearDown() override { TestHcommCAdptBase::TearDown(); }
};

// InitComm 空指针测试 (原 AicpuIndOpCommInit)
TEST_F(TestCollCommAicpuMgr, Ut_InitComm_When_ParamNullptr_Return_Error)
{
    HcclResult ret = CollCommAicpuMgr::GetInstance().InitComm(nullptr);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

// kernel adapter 空指针测试 (原 AicpuIndOpThreadInit)
TEST_F(TestCollCommAicpuMgr, Ut_InitThreads_When_ParamNullptr_Return_Error)
{
    HcclResult ret = CollCommAicpuKernelAdptInitThreads(nullptr);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

// kernel adapter 空指针测试 (原 AicpuIndOpNotifyInit)
TEST_F(TestCollCommAicpuMgr, Ut_InitNotify_When_ParamNullptr_Return_Error)
{
    HcclResult ret = CollCommAicpuKernelAdptInitNotify(nullptr);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

// kernel adapter 空指针测试 (原 AicpuIndOpChannelUpdate)
TEST_F(TestCollCommAicpuMgr, Ut_UpdateChannel_When_ParamNullptr_Return_Error)
{
    HcclResult ret = CollCommAicpuKernelAdptUpdateChannel(nullptr);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

// ==================== 后台线程纯函数 Clean/RecordChannels 测试 ====================
// NS 快恢 Clean 为后台线程纯函数调用（RecordChannels 建链存档 handle 列表 + Clean 遍历清理），
// POST 窗口内 host 无法下发 kernel，不走 kernel 下发路径

// RecordChannels — handleList 为 null：参数防御
TEST_F(TestCollCommAicpuMgr, Ut_CollCommAicpu_RecordChannels_When_NullList_Return_Error)
{
    CollCommAicpu coll;
    EXPECT_NE(coll.RecordChannels(nullptr, 1), HCCL_SUCCESS);
}

// RecordChannels — num=0 空表：无操作直接成功，已有存档不被清空
TEST_F(TestCollCommAicpuMgr, Ut_CollCommAicpu_RecordChannels_When_EmptyList_Return_Success)
{
    CollCommAicpu coll;
    ChannelHandle handleList[1] = {0};
    ASSERT_EQ(coll.RecordChannels(handleList, 1), HCCL_SUCCESS);
    ASSERT_EQ(coll.channelHandleList_.size(), 1U);
    EXPECT_EQ(coll.RecordChannels(handleList, 0), HCCL_SUCCESS);
    EXPECT_EQ(coll.channelHandleList_.size(), 1U);
}

// RecordChannels + Clean 正向链：真实 transport 存档后 Clean 纯函数清理，destroy 正常
TEST_F(TestCollCommAicpuMgr, Ut_CollCommAicpu_RecordChannelsThenClean_When_OneHandle_Succeeds)
{
    std::vector<char> data = BuildRocePackData();
    uint64_t handle = 0;
    ASSERT_EQ(AicpuChannelProcess::ParsePackData(data, handle), HCCL_SUCCESS);
    ASSERT_NE(handle, 0ULL);

    CollCommAicpu coll;
    ChannelHandle handleList[1] = {handle};
    EXPECT_EQ(coll.RecordChannels(handleList, 1), HCCL_SUCCESS);
    EXPECT_EQ(coll.Clean(), HCCL_SUCCESS);

    HcclChannelUrmaRes destroyRes{};
    destroyRes.channelList = static_cast<void*>(&handle);
    destroyRes.listNum = 1U;
    EXPECT_EQ(AicpuChannelProcess::AicpuChannelDestroy(&destroyRes), HCCL_SUCCESS);
}

// RecordChannels 多批次累积：两批建链的 handle 全部在册，Clean 覆盖全量通道
TEST_F(TestCollCommAicpuMgr, Ut_CollCommAicpu_RecordChannels_When_MultiBatch_Expect_Accumulated)
{
    std::vector<char> data1 = BuildRocePackData();
    std::vector<char> data2 = BuildRocePackData();
    uint64_t handle1 = 0;
    uint64_t handle2 = 0;
    ASSERT_EQ(AicpuChannelProcess::ParsePackData(data1, handle1), HCCL_SUCCESS);
    ASSERT_EQ(AicpuChannelProcess::ParsePackData(data2, handle2), HCCL_SUCCESS);
    ASSERT_NE(handle1, 0ULL);
    ASSERT_NE(handle2, 0ULL);
    ASSERT_NE(handle1, handle2);

    CollCommAicpu coll;
    ChannelHandle batch1[1] = {handle1};
    ChannelHandle batch2[1] = {handle2};
    EXPECT_EQ(coll.RecordChannels(batch1, 1), HCCL_SUCCESS);
    EXPECT_EQ(coll.RecordChannels(batch2, 1), HCCL_SUCCESS);
    EXPECT_EQ(coll.channelHandleList_.size(), 2U);
    EXPECT_EQ(coll.Clean(), HCCL_SUCCESS);

    HcclChannelUrmaRes destroyRes1{};
    destroyRes1.channelList = static_cast<void*>(&handle1);
    destroyRes1.listNum = 1U;
    EXPECT_EQ(AicpuChannelProcess::AicpuChannelDestroy(&destroyRes1), HCCL_SUCCESS);
    HcclChannelUrmaRes destroyRes2{};
    destroyRes2.channelList = static_cast<void*>(&handle2);
    destroyRes2.listNum = 1U;
    EXPECT_EQ(AicpuChannelProcess::AicpuChannelDestroy(&destroyRes2), HCCL_SUCCESS);
}

// RecordChannels 重复 handle 去重：同批与跨批重复均只入表一次
TEST_F(TestCollCommAicpuMgr, Ut_CollCommAicpu_RecordChannels_When_DuplicateHandle_Expect_Deduped)
{
    std::vector<char> data = BuildRocePackData();
    uint64_t handle = 0;
    ASSERT_EQ(AicpuChannelProcess::ParsePackData(data, handle), HCCL_SUCCESS);
    ASSERT_NE(handle, 0ULL);

    CollCommAicpu coll;
    ChannelHandle batch[2] = {handle, handle};
    EXPECT_EQ(coll.RecordChannels(batch, 2), HCCL_SUCCESS);
    ChannelHandle batchAgain[1] = {handle};
    EXPECT_EQ(coll.RecordChannels(batchAgain, 1), HCCL_SUCCESS);
    EXPECT_EQ(coll.channelHandleList_.size(), 1U);
    EXPECT_EQ(coll.Clean(), HCCL_SUCCESS);

    HcclChannelUrmaRes destroyRes{};
    destroyRes.channelList = static_cast<void*>(&handle);
    destroyRes.listNum = 1U;
    EXPECT_EQ(AicpuChannelProcess::AicpuChannelDestroy(&destroyRes), HCCL_SUCCESS);
}

// Clean — 未存档（空列表）直接成功
TEST_F(TestCollCommAicpuMgr, Ut_CollCommAicpu_Clean_When_EmptyList_Return_Success)
{
    CollCommAicpu coll;
    EXPECT_EQ(coll.Clean(), HCCL_SUCCESS);
}

// Clean — 非在册句柄跳过不崩溃（Unwrap 失败被忽略 + IsLiteTransportHandle 在册校验防御）
TEST_F(TestCollCommAicpuMgr, Ut_CollCommAicpu_Clean_When_InvalidHandle_Skip_Return_Success)
{
    CollCommAicpu coll;
    // 裸野值（如 0xdeadbeef）会在 Unwrap 解包时崩溃，改用可读但非 ctx 的全零内存模拟悬垂句柄
    uint64_t notCtxMem[8] = {0};
    ChannelHandle handleList[2] = {0, reinterpret_cast<ChannelHandle>(notCtxMem)};
    EXPECT_EQ(coll.RecordChannels(handleList, 2), HCCL_SUCCESS);
    EXPECT_EQ(coll.Clean(), HCCL_SUCCESS);
}

// ==================== RunAicpuChannelMemUpdate：地址条目批量录入 ====================

namespace {
// 0 conn 的最小 device transport，构造期不触碰任何连接
std::vector<char> BuildLiteTransportData()
{
    Hccl::BinaryStream binaryStream;
    u32 type = static_cast<u32>(Hccl::TransportType::UB);
    u32 zero = 0;
    binaryStream << type;
    binaryStream << zero; // notifyNum
    binaryStream << zero; // bufferNum
    binaryStream << zero; // rmtbufferNum
    binaryStream << zero; // connNum
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

Hccl::RmaBufEntryLite MakeLiteEntry(u64 addr, u64 size, u32 tokenId, u32 tokenValue, u32 notifyId)
{
    Hccl::RmaBufEntryLite entry;
    entry.addr = addr;
    entry.size = size;
    entry.tokenId = tokenId;
    entry.tokenValue = tokenValue;
    entry.notifyId = notifyId;
    return entry;
}

// 按 host 侧字段序造 blob：entryNum(u32) + N × 28B
std::vector<char> MakePack(const std::vector<Hccl::RmaBufEntryLite>& entries)
{
    Hccl::BinaryStream binaryStream;
    u32 entryNum = static_cast<u32>(entries.size());
    binaryStream << entryNum;
    for (const auto& entry : entries) {
        Hccl::PackRmaBufEntry(entry, binaryStream);
    }
    std::vector<char> data;
    binaryStream.Dump(data);
    return data;
}

HcclResult RunMemUpdate(HcclAicpuChannelMemUpdateRes& res)
{
    uint64_t devAddr = reinterpret_cast<uint64_t>(&res);
    return static_cast<HcclResult>(RunAicpuChannelMemUpdate(&devAddr));
}

// 栈 transport 无法走 ParsePackData 注册，手工登记 handle（值为空仅占 key），析构移除防泄漏
class ScopedTransportRegistration {
public:
    explicit ScopedTransportRegistration(ChannelHandle handle) : handle_(handle)
    {
        AicpuChannelProcess::transportMap_[handle_] = nullptr;
    }
    ~ScopedTransportRegistration() { AicpuChannelProcess::transportMap_.erase(handle_); }
    ScopedTransportRegistration(const ScopedTransportRegistration&) = delete;
    ScopedTransportRegistration& operator=(const ScopedTransportRegistration&) = delete;

private:
    ChannelHandle handle_;
};
} // namespace

TEST_F(TestCollCommAicpuMgr, Ut_RunAicpuChannelMemUpdate_When_ArgsNull_Return_Error)
{
    HcclResult ret = static_cast<HcclResult>(RunAicpuChannelMemUpdate(nullptr));
    EXPECT_NE(ret, HCCL_SUCCESS);
}

TEST_F(TestCollCommAicpuMgr, Ut_RunAicpuChannelMemUpdate_When_CommParamNull_Return_Error)
{
    uint64_t devAddr = 0;
    HcclResult ret = static_cast<HcclResult>(RunAicpuChannelMemUpdate(&devAddr));
    EXPECT_NE(ret, HCCL_SUCCESS);
}

TEST_F(TestCollCommAicpuMgr, Ut_RunAicpuChannelMemUpdate_When_DeviceHandleListNull_Return_Error)
{
    uint64_t packAddrList[1] = {0};
    uint32_t packSizeList[1] = {0};
    HcclAicpuChannelMemUpdateRes res{};
    res.packAddrList = packAddrList;
    res.packSizeList = packSizeList;
    res.num = 1;
    EXPECT_NE(RunMemUpdate(res), HCCL_SUCCESS);
}

TEST_F(TestCollCommAicpuMgr, Ut_RunAicpuChannelMemUpdate_When_PackAddrListNull_Return_Error)
{
    uint64_t deviceHandleList[1] = {0};
    uint32_t packSizeList[1] = {0};
    HcclAicpuChannelMemUpdateRes res{};
    res.deviceHandleList = deviceHandleList;
    res.packSizeList = packSizeList;
    res.num = 1;
    EXPECT_NE(RunMemUpdate(res), HCCL_SUCCESS);
}

TEST_F(TestCollCommAicpuMgr, Ut_RunAicpuChannelMemUpdate_When_PackSizeListNull_Return_Error)
{
    uint64_t deviceHandleList[1] = {0};
    uint64_t packAddrList[1] = {0};
    HcclAicpuChannelMemUpdateRes res{};
    res.deviceHandleList = deviceHandleList;
    res.packAddrList = packAddrList;
    res.num = 1;
    EXPECT_NE(RunMemUpdate(res), HCCL_SUCCESS);
}

TEST_F(TestCollCommAicpuMgr, Ut_RunAicpuChannelMemUpdate_When_NumZero_Return_Success)
{
    uint64_t deviceHandleList[1] = {0};
    uint64_t packAddrList[1] = {0};
    uint32_t packSizeList[1] = {0};
    HcclAicpuChannelMemUpdateRes res{};
    res.deviceHandleList = deviceHandleList;
    res.packAddrList = packAddrList;
    res.packSizeList = packSizeList;
    EXPECT_EQ(RunMemUpdate(res), HCCL_SUCCESS);
}

TEST_F(TestCollCommAicpuMgr, Ut_RunAicpuChannelMemUpdate_When_PackSizeZero_Expect_SkipAndNoUpsert)
{
    std::vector<char> transportData = BuildLiteTransportData();
    Hccl::UbTransportLiteImpl transport(transportData);
    uint64_t deviceHandleList[1] = {reinterpret_cast<uint64_t>(&transport)};
    std::vector<char> pack = MakePack({MakeLiteEntry(0x1000, 512, 11, 22, 33)});
    uint64_t packAddrList[1] = {reinterpret_cast<uint64_t>(pack.data())};
    uint32_t packSizeList[1] = {0};
    HcclAicpuChannelMemUpdateRes res{};
    res.deviceHandleList = deviceHandleList;
    res.packAddrList = packAddrList;
    res.packSizeList = packSizeList;
    res.num = 1;

    EXPECT_EQ(RunMemUpdate(res), HCCL_SUCCESS);
    // size=0 即跳过该通道，blob 一个字节都不解
    EXPECT_EQ(transport.rmtBufferMap.size(), 0U);
}

TEST_F(TestCollCommAicpuMgr, Ut_RunAicpuChannelMemUpdate_When_SingleEntry_Expect_Upserted)
{
    std::vector<char> transportData = BuildLiteTransportData();
    Hccl::UbTransportLiteImpl transport(transportData);
    ScopedTransportRegistration guard(reinterpret_cast<ChannelHandle>(&transport));
    uint64_t deviceHandleList[1] = {reinterpret_cast<uint64_t>(&transport)};
    std::vector<char> pack = MakePack({MakeLiteEntry(0x1000, 512, 11, 22, 33)});
    uint64_t packAddrList[1] = {reinterpret_cast<uint64_t>(pack.data())};
    uint32_t packSizeList[1] = {static_cast<uint32_t>(pack.size())};
    HcclAicpuChannelMemUpdateRes res{};
    res.deviceHandleList = deviceHandleList;
    res.packAddrList = packAddrList;
    res.packSizeList = packSizeList;
    res.num = 1;

    EXPECT_EQ(RunMemUpdate(res), HCCL_SUCCESS);
    ASSERT_EQ(transport.rmtBufferMap.size(), 1U);
    auto it = transport.rmtBufferMap.find(0x1000);
    ASSERT_NE(it, transport.rmtBufferMap.end());
    EXPECT_EQ(it->second.addr, 0x1000U);
    EXPECT_EQ(it->second.size, 512U);
    EXPECT_EQ(it->second.tokenId, 11U);
    EXPECT_EQ(it->second.tokenValue, 22U);
    EXPECT_EQ(it->second.notifyId, 33U);
}

TEST_F(TestCollCommAicpuMgr, Ut_RunAicpuChannelMemUpdate_When_MultiEntry_Expect_AllUpserted)
{
    std::vector<char> transportData = BuildLiteTransportData();
    Hccl::UbTransportLiteImpl transport(transportData);
    ScopedTransportRegistration guard(reinterpret_cast<ChannelHandle>(&transport));
    uint64_t deviceHandleList[1] = {reinterpret_cast<uint64_t>(&transport)};
    std::vector<char> pack = MakePack({MakeLiteEntry(0x1000, 512, 11, 22, 33), MakeLiteEntry(0x2000, 256, 44, 55, 66)});
    uint64_t packAddrList[1] = {reinterpret_cast<uint64_t>(pack.data())};
    uint32_t packSizeList[1] = {static_cast<uint32_t>(pack.size())};
    HcclAicpuChannelMemUpdateRes res{};
    res.deviceHandleList = deviceHandleList;
    res.packAddrList = packAddrList;
    res.packSizeList = packSizeList;
    res.num = 1;

    EXPECT_EQ(RunMemUpdate(res), HCCL_SUCCESS);
    ASSERT_EQ(transport.rmtBufferMap.size(), 2U);
    EXPECT_NE(transport.rmtBufferMap.find(0x1000), transport.rmtBufferMap.end());
    auto it = transport.rmtBufferMap.find(0x2000);
    ASSERT_NE(it, transport.rmtBufferMap.end());
    EXPECT_EQ(it->second.size, 256U);
    EXPECT_EQ(it->second.notifyId, 66U);
}

TEST_F(TestCollCommAicpuMgr, Ut_RunAicpuChannelMemUpdate_When_BatchOneEmpty_Expect_OnlyNonEmptyUpdated)
{
    std::vector<char> transportEmptyData = BuildLiteTransportData();
    Hccl::UbTransportLiteImpl transportEmpty(transportEmptyData);
    std::vector<char> transportFilledData = BuildLiteTransportData();
    Hccl::UbTransportLiteImpl transportFilled(transportFilledData);
    ScopedTransportRegistration guard(reinterpret_cast<ChannelHandle>(&transportFilled));
    uint64_t deviceHandleList[2]
        = {reinterpret_cast<uint64_t>(&transportEmpty), reinterpret_cast<uint64_t>(&transportFilled)};
    std::vector<char> pack = MakePack({MakeLiteEntry(0x3000, 128, 7, 8, 9)});
    uint64_t packAddrList[2] = {0, reinterpret_cast<uint64_t>(pack.data())};
    uint32_t packSizeList[2] = {0, static_cast<uint32_t>(pack.size())};
    HcclAicpuChannelMemUpdateRes res{};
    res.deviceHandleList = deviceHandleList;
    res.packAddrList = packAddrList;
    res.packSizeList = packSizeList;
    res.num = 2;

    EXPECT_EQ(RunMemUpdate(res), HCCL_SUCCESS);
    EXPECT_EQ(transportEmpty.rmtBufferMap.size(), 0U);
    ASSERT_EQ(transportFilled.rmtBufferMap.size(), 1U);
    EXPECT_NE(transportFilled.rmtBufferMap.find(0x3000), transportFilled.rmtBufferMap.end());
}

TEST_F(TestCollCommAicpuMgr, Ut_RunAicpuChannelMemUpdate_When_HandleZero_Return_Error)
{
    std::vector<char> transportData = BuildLiteTransportData();
    Hccl::UbTransportLiteImpl transport(transportData);
    uint64_t deviceHandleList[1] = {0};
    std::vector<char> pack = MakePack({MakeLiteEntry(0x1000, 512, 11, 22, 33)});
    uint64_t packAddrList[1] = {reinterpret_cast<uint64_t>(pack.data())};
    uint32_t packSizeList[1] = {static_cast<uint32_t>(pack.size())};
    HcclAicpuChannelMemUpdateRes res{};
    res.deviceHandleList = deviceHandleList;
    res.packAddrList = packAddrList;
    res.packSizeList = packSizeList;
    res.num = 1;

    EXPECT_EQ(RunMemUpdate(res), HCCL_E_PTR);
    EXPECT_EQ(transport.rmtBufferMap.size(), 0U);
}

TEST_F(TestCollCommAicpuMgr, Ut_RunAicpuChannelMemUpdate_When_LaterHandleZero_Expect_EarlyReturn)
{
    std::vector<char> transportFirstData = BuildLiteTransportData();
    Hccl::UbTransportLiteImpl transportFirst(transportFirstData);
    ScopedTransportRegistration guard(reinterpret_cast<ChannelHandle>(&transportFirst));
    std::vector<char> transportLastData = BuildLiteTransportData();
    Hccl::UbTransportLiteImpl transportLast(transportLastData);
    uint64_t deviceHandleList[3]
        = {reinterpret_cast<uint64_t>(&transportFirst), 0, reinterpret_cast<uint64_t>(&transportLast)};
    std::vector<char> pack = MakePack({MakeLiteEntry(0x1000, 512, 11, 22, 33)});
    uint64_t packAddrList[3]
        = {reinterpret_cast<uint64_t>(pack.data()), reinterpret_cast<uint64_t>(pack.data()),
           reinterpret_cast<uint64_t>(pack.data())};
    uint32_t packSizeList[3]
        = {static_cast<uint32_t>(pack.size()), static_cast<uint32_t>(pack.size()), static_cast<uint32_t>(pack.size())};
    HcclAicpuChannelMemUpdateRes res{};
    res.deviceHandleList = deviceHandleList;
    res.packAddrList = packAddrList;
    res.packSizeList = packSizeList;
    res.num = 3;

    EXPECT_EQ(RunMemUpdate(res), HCCL_E_PTR);
    EXPECT_EQ(transportFirst.rmtBufferMap.size(), 1U);
    // 中途失败即返回，后面的通道不再处理
    EXPECT_EQ(transportLast.rmtBufferMap.size(), 0U);
}

// 未在册 handle（如 host 句柄兜底流入）跳过且不中断后续通道，整批成功
TEST_F(TestCollCommAicpuMgr, Ut_RunAicpuChannelMemUpdate_When_HandleNotRegistered_Expect_SkipAndContinue)
{
    std::vector<char> unregisteredData = BuildLiteTransportData();
    Hccl::UbTransportLiteImpl unregistered(unregisteredData);
    std::vector<char> registeredData = BuildLiteTransportData();
    Hccl::UbTransportLiteImpl registered(registeredData);
    ScopedTransportRegistration guard(reinterpret_cast<ChannelHandle>(&registered));
    uint64_t deviceHandleList[2] = {reinterpret_cast<uint64_t>(&unregistered), reinterpret_cast<uint64_t>(&registered)};
    std::vector<char> pack = MakePack({MakeLiteEntry(0x5000, 256, 4, 5, 6)});
    uint64_t packAddrList[2] = {reinterpret_cast<uint64_t>(pack.data()), reinterpret_cast<uint64_t>(pack.data())};
    uint32_t packSizeList[2] = {static_cast<uint32_t>(pack.size()), static_cast<uint32_t>(pack.size())};
    HcclAicpuChannelMemUpdateRes res{};
    res.deviceHandleList = deviceHandleList;
    res.packAddrList = packAddrList;
    res.packSizeList = packSizeList;
    res.num = 2;

    EXPECT_EQ(RunMemUpdate(res), HCCL_SUCCESS);
    // 未在册通道跳过，不 upsert
    EXPECT_EQ(unregistered.rmtBufferMap.size(), 0U);
    // 后续在册通道正常处理
    ASSERT_EQ(registered.rmtBufferMap.size(), 1U);
    EXPECT_NE(registered.rmtBufferMap.find(0x5000), registered.rmtBufferMap.end());
}

TEST_F(TestCollCommAicpuMgr, Ut_RunAicpuChannelMemUpdate_When_HandleIsCtxPtr_Expect_Unwrapped)
{
    std::vector<char> transportData = BuildLiteTransportData();
    Hccl::UbTransportLiteImpl transport(transportData);
    ScopedTransportRegistration guard(reinterpret_cast<ChannelHandle>(&transport));
    HcommAicpuChannelCtx ctx{};
    ctx.abiHeader.version = HCOMM_AICPU_CHANNEL_CTX_VERSION;
    ctx.abiHeader.magicWord = HCOMM_AICPU_CHANNEL_CTX_MAGIC_WORD;
    ctx.deviceChannel = &transport;
    uint64_t deviceHandleList[1] = {reinterpret_cast<uint64_t>(&ctx)};
    std::vector<char> pack = MakePack({MakeLiteEntry(0x4000, 64, 1, 2, 3)});
    uint64_t packAddrList[1] = {reinterpret_cast<uint64_t>(pack.data())};
    uint32_t packSizeList[1] = {static_cast<uint32_t>(pack.size())};
    HcclAicpuChannelMemUpdateRes res{};
    res.deviceHandleList = deviceHandleList;
    res.packAddrList = packAddrList;
    res.packSizeList = packSizeList;
    res.num = 1;

    // ctx 模式下入参是 ctxPtr，解包到 deviceChannel 后再录表
    EXPECT_EQ(RunMemUpdate(res), HCCL_SUCCESS);
    ASSERT_EQ(transport.rmtBufferMap.size(), 1U);
    EXPECT_NE(transport.rmtBufferMap.find(0x4000), transport.rmtBufferMap.end());
}

TEST_F(TestCollCommAicpuMgr, Ut_RunAicpuChannelMemUpdate_When_CtxDeviceChannelNull_Return_Error)
{
    HcommAicpuChannelCtx ctx{};
    ctx.abiHeader.version = HCOMM_AICPU_CHANNEL_CTX_VERSION;
    ctx.abiHeader.magicWord = HCOMM_AICPU_CHANNEL_CTX_MAGIC_WORD;
    ctx.deviceChannel = nullptr;
    uint64_t deviceHandleList[1] = {reinterpret_cast<uint64_t>(&ctx)};
    std::vector<char> pack = MakePack({MakeLiteEntry(0x4000, 64, 1, 2, 3)});
    uint64_t packAddrList[1] = {reinterpret_cast<uint64_t>(pack.data())};
    uint32_t packSizeList[1] = {static_cast<uint32_t>(pack.size())};
    HcclAicpuChannelMemUpdateRes res{};
    res.deviceHandleList = deviceHandleList;
    res.packAddrList = packAddrList;
    res.packSizeList = packSizeList;
    res.num = 1;

    EXPECT_EQ(RunMemUpdate(res), HCCL_E_INTERNAL);
}

// ==================== CollCommAicpuMgr 注册表操作测试 ====================
// AcquireCommForUse + ReleaseComm — 获取使用标记后释放
TEST_F(TestCollCommAicpuMgr, Ut_AcquireCommForUse_ThenRelease)
{
    MOCKER_CPP(&CollCommAicpu::InitAicpuIndOp).stubs().will(returnValue(HCCL_SUCCESS));
    CommAicpuParam param{};
    strncpy(param.hcomId, "test_acquire_use", HCOMID_MAX_SIZE - 1);
    CollCommAicpuMgr::GetInstance().InitComm(&param);

    CollCommAicpu* comm = CollCommAicpuMgr::GetInstance().AcquireCommForUse("test_acquire_use");
    EXPECT_NE(comm, nullptr);
    CollCommAicpuMgr::GetInstance().ReleaseComm("test_acquire_use");
    CollCommAicpuMgr::GetInstance().DestroyComm("test_acquire_use");
}

// ReleaseComm — 不存在的 group 不崩溃
TEST_F(TestCollCommAicpuMgr, Ut_ReleaseComm_Nonexistent_DoesNotCrash)
{
    CollCommAicpuMgr::GetInstance().ReleaseComm("nonexistent_group");
    SUCCEED();
}

TEST_F(TestCollCommAicpuMgr, Ut_GetCurrentComm_EmptyOrMismatch_ReturnsNull)
{
    EXPECT_EQ(CollCommAicpuMgr::GetInstance().GetCurrentComm(""), nullptr);
    EXPECT_EQ(CollCommAicpuMgr::GetInstance().GetCurrentComm("nonexistent"), nullptr);
}

TEST_F(TestCollCommAicpuMgr, Ut_GetMutex_ReturnsValidReference)
{
    std::shared_mutex& mtx = CollCommAicpuMgr::GetInstance().GetMutex();
    mtx.lock();
    mtx.unlock();
    SUCCEED();
}

TEST_F(TestCollCommAicpuMgr, Ut_DestroyComm_NonExistent_ReturnsError)
{
    HcclResult ret = CollCommAicpuMgr::GetInstance().DestroyComm("nonexistent_group");
    EXPECT_NE(ret, HCCL_SUCCESS);
}

TEST_F(TestCollCommAicpuMgr, Ut_GetAllComms_ReturnsEmpty)
{
    std::vector<std::pair<std::string, CollCommAicpu*>> commInfo;
    HcclResult ret = CollCommAicpuMgr::GetInstance().GetAllComms(commInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// InitComm — mock InitAicpuIndOp 跳过硬件初始化，验证通信域创建流程
TEST_F(TestCollCommAicpuMgr, Ut_InitComm_MockInit_CreatesAndDestroys)
{
    MOCKER_CPP(&CollCommAicpu::InitAicpuIndOp).stubs().will(returnValue(HCCL_SUCCESS));
    CommAicpuParam param{};
    strncpy(param.hcomId, "initcomm_test_group", HCOMID_MAX_SIZE - 1);
    HcclResult ret = CollCommAicpuMgr::GetInstance().InitComm(&param);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 初始化 commEngineResMgr_ 供后续操作使用
    auto* comm = CollCommAicpuMgr::GetInstance().AcquireCommForUse("initcomm_test_group");
    EXPECT_NE(comm, nullptr);
    if (comm->commEngineResMgr_ == nullptr) {
        comm->commEngineResMgr_ = std::make_unique<CommEngineResAicpuMgr>(comm->dfx_, [](bool) {
            return HCCL_SUCCESS;
        });
    }

    CollCommAicpuMgr::GetInstance().DestroyComm("initcomm_test_group");
}

// AcquireCommForUse — 获取通信域并标记使用中
TEST_F(TestCollCommAicpuMgr, Ut_AcquireCommForUse_Then_Release)
{
    MOCKER_CPP(&CollCommAicpu::InitAicpuIndOp).stubs().will(returnValue(HCCL_SUCCESS));
    CommAicpuParam param{};
    strncpy(param.hcomId, "acquire_use_group", HCOMID_MAX_SIZE - 1);
    CollCommAicpuMgr::GetInstance().InitComm(&param);

    CollCommAicpu* comm = CollCommAicpuMgr::GetInstance().AcquireCommForUse("acquire_use_group");
    EXPECT_NE(comm, nullptr);
    CollCommAicpuMgr::GetInstance().ReleaseComm("acquire_use_group");

    CollCommAicpuMgr::GetInstance().DestroyComm("acquire_use_group");
}

// GetCurrentComm — 设置后能获取到当前通信域
TEST_F(TestCollCommAicpuMgr, Ut_GetCurrentComm_AfterSet_ReturnsComm)
{
    MOCKER_CPP(&CollCommAicpu::InitAicpuIndOp).stubs().will(returnValue(HCCL_SUCCESS));
    CommAicpuParam param{};
    strncpy(param.hcomId, "current_comm_group", HCOMID_MAX_SIZE - 1);
    CollCommAicpuMgr::GetInstance().InitComm(&param);

    CollCommAicpu* comm = CollCommAicpuMgr::GetInstance().AcquireCommForUse("current_comm_group");
    EXPECT_NE(comm, nullptr);
    // InitAicpuIndOp 被 mock，手动设置 identifier_ 供 GetCurrentComm 校验
    comm->identifier_ = "current_comm_group";
    // AcquireCommForUse 已设置 currentComm_，GetCurrentComm 应能找到
    CollCommAicpu* found = CollCommAicpuMgr::GetInstance().GetCurrentComm("current_comm_group");
    EXPECT_EQ(found, comm);

    CollCommAicpuMgr::GetInstance().ReleaseComm("current_comm_group");
    CollCommAicpuMgr::GetInstance().DestroyComm("current_comm_group");
}

// DestroyComm — 正常创建并销毁
TEST_F(TestCollCommAicpuMgr, Ut_DestroyComm_CreateThenDestroy_Success)
{
    MOCKER_CPP(&CollCommAicpu::InitAicpuIndOp).stubs().will(returnValue(HCCL_SUCCESS));
    CommAicpuParam param{};
    strncpy(param.hcomId, "destroy_test_group", HCOMID_MAX_SIZE - 1);
    CollCommAicpuMgr::GetInstance().InitComm(&param);
    HcclResult ret = CollCommAicpuMgr::GetInstance().DestroyComm("destroy_test_group");
    EXPECT_EQ(ret, HCCL_SUCCESS);
    // 再次销毁应返回错误
    ret = CollCommAicpuMgr::GetInstance().DestroyComm("destroy_test_group");
    EXPECT_NE(ret, HCCL_SUCCESS);
}
