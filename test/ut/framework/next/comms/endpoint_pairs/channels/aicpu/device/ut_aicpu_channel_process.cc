/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>

#include "adapter_hal.h"
// transportMap_ 为 private 静态成员，观测「初始为空」需放开访问限制
#define private public
#include "aicpu/device/aicpu_channel_process.h"
#undef private
#include "endpoint_pair/channel/device/aicpu_channel_pub.h"
#include "aicpu/device/dev_aicpu_ts_channel_mgr.h"
#include "channel_param.h"
#include "dlhns_function.h"
#include "externalinput_pub.h"
#include "hccl_dispatcher_ctx.h"
#include "workflow_pub.h"
#include "aicpu_res_package_helper.h"
#include "binary_stream.h"
#include "roce_host_loc_transport_lite_impl.h"
#include "ub_transport_lite_impl.h"
#include "aicpu_pack_data_helper.h"
#include "aicpu/aicpu_ts_channel_entity_resources.h"

using namespace hccl;

namespace {
constexpr uint32_t kKindTsRoce = 2U;

HcclResult StubHrtDrvGetLocalDevIDByHostDevID(u32 /*hostUdevid*/, u32* localDevid)
{
    if (localDevid != nullptr) {
        *localDevid = 0U;
    }
    return HCCL_SUCCESS;
}

HcclResult StubHrtDrvGetPlatformInfo(uint32_t* info)
{
    if (info != nullptr) {
        *info = 1U;
    }
    return HCCL_SUCCESS;
}
} // namespace

namespace {
// 分发 + 断言 handle 非空 + 销毁单个 channel：单 channel 成功用例的公共尾部。
static void ParsePackDataAndDestroy(std::vector<char>& data)
{
    ChannelHandle handle = 0ULL;
    EXPECT_EQ(AicpuChannelProcess::ParsePackData(data, handle), HCCL_SUCCESS);
    EXPECT_NE(handle, 0ULL);
    HcclChannelUrmaRes destroyRes{};
    destroyRes.channelList = static_cast<void*>(&handle);
    destroyRes.listNum = 1U;
    EXPECT_EQ(AicpuChannelProcess::AicpuChannelDestroy(&destroyRes), HCCL_SUCCESS);
}
} // namespace

class AicpuChannelProcessTest : public testing::Test {
protected:
    void TearDown() override { GlobalMockObject::verify(); }
};

TEST_F(AicpuChannelProcessTest, Ut_InitHcommChannelRes_WhenCommParamNull_Returns_PTR)
{
    EXPECT_EQ(AicpuChannelProcess::InitHcommChannelRes(nullptr), HCCL_E_PTR);
}

// transportMap_ 为全局静态表，本断言依赖同二进制内无前置用例泄漏条目
TEST_F(AicpuChannelProcessTest, Ut_TransportMap_InitialEmpty)
{
    EXPECT_EQ(AicpuChannelProcess::transportMap_.size(), 0u);
}

TEST_F(AicpuChannelProcessTest, Ut_InitHcommChannelRes_WhenChannelListNull_Returns_PTR)
{
    HcommChannelRes res{};
    res.listNum = 1U;
    res.channelList = nullptr;
    res.channelDataListAddr = reinterpret_cast<void*>(0x1U);
    res.channelDataSizeListAddr = reinterpret_cast<void*>(0x1U);
    res.channelTypeListAddr = reinterpret_cast<void*>(0x1U);
    EXPECT_EQ(AicpuChannelProcess::InitHcommChannelRes(&res), HCCL_E_PTR);
}

TEST_F(AicpuChannelProcessTest, Ut_InitHcommChannelRes_WhenUnsupportedKind_Returns_NOT_SUPPORT)
{
    HcommRoceChannelRes blob{};
    blob.qpsPerConnection = 1U;
    blob.QpInfo[0].qpPtr = 0x7000ULL;

    void* dataPtr = static_cast<void*>(&blob);
    u64 sizeVal = sizeof(blob);
    u32 typeVal = 999U;
    ChannelHandle outHandle = 0ULL;

    HcommChannelRes res{};
    res.listNum = 1U;
    res.channelList = static_cast<void*>(&outHandle);
    res.channelDataListAddr = static_cast<void*>(&dataPtr);
    res.channelDataSizeListAddr = static_cast<void*>(&sizeVal);
    res.channelTypeListAddr = static_cast<void*>(&typeVal);
    res.deviceInfo.deviceLogicId = 0;
    res.deviceInfo.devicePhyId = 0U;

    MOCKER(hrtSetWorkModeAicpu).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtSetlocalDevice).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtSetlocalDeviceType).stubs().will(returnValue(HCCL_SUCCESS));

    EXPECT_EQ(AicpuChannelProcess::InitHcommChannelRes(&res), HCCL_E_NOT_SUPPORT);
}

TEST_F(AicpuChannelProcessTest, Ut_InitHcommChannelRes_WhenValidTsRoceBlob_Returns_SUCCESS)
{
    MOCKER_CPP(&DlHnsFunction::DlHnsFunctionInit).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtDrvGetPlatformInfo).stubs().will(invoke(StubHrtDrvGetPlatformInfo));
    MOCKER(GetExternalInputHcclAicpuUnfold).stubs().will(returnValue(true));
    DevType devType = DevType::DEV_TYPE_910_93;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(devType)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtDrvGetLocalDevIDByHostDevID).stubs().will(invoke(StubHrtDrvGetLocalDevIDByHostDevID));
    MOCKER(hrtSetWorkModeAicpu).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtSetlocalDevice).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtSetlocalDeviceType).stubs().will(returnValue(HCCL_SUCCESS));

    HcommRoceChannelRes blob{};
    blob.qpsPerConnection = 1U;
    blob.localMemCount = 0U;
    blob.remoteMemCount = 0U;
    blob.localMem = nullptr;
    blob.remoteMem = nullptr;
    blob.QpInfo[0].qpPtr = 0x7000ULL;
    blob.QpInfo[0].sqIndex = 1U;
    blob.QpInfo[0].dbIndex = 2U;
    blob.localDataNotifyAddr = reinterpret_cast<void*>(static_cast<uintptr_t>(0x8000ULL));
    blob.remoteNotifyAddr = reinterpret_cast<void*>(static_cast<uintptr_t>(0x9000ULL));
    void* dataPtr = static_cast<void*>(&blob);
    u64 sizeVal = sizeof(blob);
    u32 typeVal = kKindTsRoce;
    ChannelHandle outHandle = 0ULL;

    HcommChannelRes res{};
    res.listNum = 1U;
    res.channelList = static_cast<void*>(&outHandle);
    res.channelDataListAddr = static_cast<void*>(&dataPtr);
    res.channelDataSizeListAddr = static_cast<void*>(&sizeVal);
    res.channelTypeListAddr = static_cast<void*>(&typeVal);
    res.deviceInfo.deviceLogicId = 0;
    res.deviceInfo.devicePhyId = 0U;
    res.deviceInfo.deviceType = static_cast<u32>(DevType::DEV_TYPE_910_93);

    ASSERT_EQ(AicpuChannelProcess::InitHcommChannelRes(&res), HCCL_SUCCESS);
    ASSERT_NE(outHandle, 0ULL);
    EXPECT_TRUE(DevAicpuTsChannelMgr::Instance().DestroyChannel(outHandle));
}

// ROCE_V2 建链收编走 V2 后，device 侧构造 RoceHostLocTransportLiteImpl 的路径由 ParsePackData ROCE 分支承担。
// 等价于被删 ut_dev_aicpu_ts_roce_channel_v2.cc 的 Ut_When_Construct_Expect_Success。
// 此处只测「分发」：最小 uniqueId（type=ROCE + 各段计数为 0 + notifyValue 段），完整字段解析由
// legacy ut_roce_host_loc_transport_lite_impl.cc 覆盖。
TEST_F(AicpuChannelProcessTest, Ut_ParsePackData_WhenRoceBlob_Returns_SUCCESS_AndHandle)
{
    std::vector<char> data = BuildRocePackData();
    ParsePackDataAndDestroy(data);
}

TEST_F(AicpuChannelProcessTest, Ut_ParsePackData_WhenInvalidTransportType_Returns_PARA)
{
    Hccl::BinaryStream inner;
    u32 invalidType = 0xFFFFFFFFU;
    inner << invalidType;
    std::vector<char> transpUniqueId;
    inner.Dump(transpUniqueId);

    std::vector<char> data = BuildOuterPackData(transpUniqueId);

    ChannelHandle handle = 0ULL;
    EXPECT_EQ(AicpuChannelProcess::ParsePackData(data, handle), HCCL_E_PARA);
}

// P2P 收编走 V2 后，device 侧构造 P2PTransportLiteImpl 的路径由 ParsePackData P2P 分支承担。
// P2P 的 Init 无 connNum/notifyValue 段，最小分发 blob 仅 type=P2P + notifyNum=0 + bufferNum=0 + 4 个空 vector。
TEST_F(AicpuChannelProcessTest, Ut_ParsePackData_WhenP2pBlob_Returns_SUCCESS_AndHandle)
{
    std::vector<char> data = BuildP2pPackData();
    ParsePackDataAndDestroy(data);
}

// device ParsePackData 的 UB 分支注册 needCacheTask / addWqeArray 两个回调
// 两个 setter 是 inline 且无 getter，利用 -fno-access-control 直接断言私有成员由空置为非空。
TEST_F(AicpuChannelProcessTest, Ut_ParsePackData_WhenUbBlob_RegistersCacheCallbacks)
{
    Hccl::BinaryStream inner;
    inner << static_cast<u32>(Hccl::TransportType::UB);
    inner << static_cast<u32>(0); // notifyNum
    inner << static_cast<u32>(0); // bufferNum
    inner << static_cast<u32>(0); // rmtbufferNum
    inner << static_cast<u32>(0); // connNum
    inner << std::vector<char>(); // locNotify
    inner << std::vector<char>(); // rmtNotify
    inner << std::vector<char>(); // locBuffer
    inner << std::vector<char>(); // rmtBuffer
    inner << std::vector<char>(); // drain
    inner << std::vector<char>(); // conn
    std::vector<char> transpUniqueId;
    inner.Dump(transpUniqueId);

    std::vector<char> data = BuildOuterPackData(transpUniqueId);

    ChannelHandle handle = 0ULL;
    EXPECT_EQ(AicpuChannelProcess::ParsePackData(data, handle), HCCL_SUCCESS);
    ASSERT_NE(handle, 0ULL);

    auto* ub = reinterpret_cast<Hccl::UbTransportLiteImpl*>(handle);
    EXPECT_NE(ub->needCacheTaskCallback_, nullptr);
    EXPECT_NE(ub->addWqeArrayCallback_, nullptr);

    HcclChannelUrmaRes destroyRes{};
    destroyRes.channelList = static_cast<void*>(&handle);
    destroyRes.listNum = 1U;
    EXPECT_EQ(AicpuChannelProcess::AicpuChannelDestroy(&destroyRes), HCCL_SUCCESS);
}

// 行 26 等价迁移：V3 单 unique_ptr transport_ 覆写 bug 已随 DevAicpuTsRoceChannelV2 删除而消除，
// V2 用 type-erased transportMap_（handle→unique_ptr）存储，多 channel 各占一条、handle 互异、独立销毁互不影响。
// 注意：ROCE 分支按 transpUniqueId 去重复用已注册 transport（保持 sqHead_/cqHead_ 影子状态连续），
// 故两条并存通道必须构造不同 uniqueId（不同 notifyAddr）。

// 单条坏句柄不得阻断同批其余通道销毁：坏句柄返回首错，好句柄仍从 transportMap_ 移除。
TEST_F(AicpuChannelProcessTest, Ut_AicpuChannelDestroy_WhenOneHandleInvalid_Expect_RestDestroyed)
{
    std::vector<char> data = BuildRocePackData();
    ChannelHandle validHandle = 0ULL;
    ASSERT_EQ(AicpuChannelProcess::ParsePackData(data, validHandle), HCCL_SUCCESS);
    ASSERT_NE(validHandle, 0ULL);
    const size_t transportMapSize = AicpuChannelProcess::transportMap_.size();

    ChannelHandle handles[2] = {0ULL, validHandle};
    HcclChannelUrmaRes destroyRes{};
    destroyRes.channelList = static_cast<void*>(handles);
    destroyRes.listNum = 2U;
    EXPECT_EQ(AicpuChannelProcess::AicpuChannelDestroy(&destroyRes), HCCL_E_PTR);
    EXPECT_EQ(AicpuChannelProcess::transportMap_.size(), transportMapSize - 1U);
    EXPECT_EQ(AicpuChannelProcess::transportMap_.find(validHandle), AicpuChannelProcess::transportMap_.end());
}

static std::vector<char> BuildRemoteNotifyData(uint32_t notifyId)
{
    constexpr size_t notifyIdOffset = sizeof(uint64_t) * 2U + sizeof(uint32_t) * 2U;
    std::vector<char> data(sizeof(uint64_t) * 2U + sizeof(uint32_t) * 3U, 0);
    (void)std::memcpy(data.data() + notifyIdOffset, &notifyId, sizeof(notifyId));
    return data;
}

static std::vector<Hccl::ModuleData> BuildEntityResourcesModules(
    const std::vector<char>& notifyData, const std::vector<char>& connectionData,
    const std::vector<char>& drainData = std::vector<char>(), const uint32_t remoteNotifyId = 0U)
{
    std::vector<char> remoteNotifyData;
    if (!notifyData.empty()) {
        uint32_t localNotifyId = 0;
        (void)std::memcpy(&localNotifyId, notifyData.data(), sizeof(localNotifyId));
        remoteNotifyData = BuildRemoteNotifyData(remoteNotifyId == 0U ? localNotifyId : remoteNotifyId);
    }

    Hccl::BinaryStream inner;
    inner << static_cast<uint32_t>(0);                            // transportType
    inner << static_cast<uint32_t>(notifyData.empty() ? 0U : 1U); // notifyNum
    inner << static_cast<uint32_t>(0);                            // localBufferNum
    inner << static_cast<uint32_t>(0);                            // remoteBufferNum
    inner << static_cast<uint32_t>(1);                            // connectionNum
    inner << notifyData;
    inner << remoteNotifyData;    // remote notify
    inner << std::vector<char>(); // local buffer
    inner << std::vector<char>(); // remote buffer
    inner << drainData;
    inner << connectionData;
    std::vector<char> transportId;
    inner.Dump(transportId);

    Hccl::ModuleData module{};
    Hccl::BinaryStream outer;
    outer << transportId;
    outer.Dump(module.data);
    std::vector<Hccl::ModuleData> modules(Hccl::AicpuResMgrType::STREAM + 1U);
    modules[Hccl::AicpuResMgrType::STREAM] = module;
    return modules;
}

static std::vector<char> BuildLocalNotifyData(uint32_t notifyId)
{
    std::vector<char> data(sizeof(uint32_t) * 2U, 0);
    (void)std::memcpy(data.data(), &notifyId, sizeof(notifyId));
    return data;
}

static std::vector<char> BuildConnectionData(uint32_t dieId)
{
    // FillJfsContext 按字段顺序读取 105 字节：5 个定标量 + 3 个 u64 +
    // 2 个 u32 + 2 个 16B EID + 3 个 u64。
    std::vector<char> data(105U, 0);
    (void)std::memcpy(data.data(), &dieId, sizeof(dieId));
    return data;
}

TEST_F(AicpuChannelProcessTest, Ut_AicpuChannelEntityResourcesBuild_WhenLocalNotifyIdOverflow_Expect_EPara)
{
    auto modules = BuildEntityResourcesModules(
        BuildLocalNotifyData(static_cast<uint32_t>(INT32_MAX) + 1U), BuildConnectionData(0U));
    MOCKER_CPP(&Hccl::AicpuResPackageHelper::ParsePackedData).stubs().with(mockcpp::any()).will(returnValue(modules));

    hcomm::AicpuChannelEntityResources resources;
    ChannelHandle entityHandle = 0;
    EXPECT_EQ(
        resources.Build(
            std::vector<char>(), COMM_ENGINE_AICPU_TS, COMM_PROTOCOL_UB_CTP, static_cast<ChannelHandle>(0x1000U),
            entityHandle),
        HCCL_E_PARA);
}

TEST_F(AicpuChannelProcessTest, Ut_AicpuChannelEntityResourcesBuild_WhenDieIdOverflow_Expect_EPara)
{
    auto modules = BuildEntityResourcesModules(std::vector<char>(), BuildConnectionData(UINT8_MAX + 1U));
    MOCKER_CPP(&Hccl::AicpuResPackageHelper::ParsePackedData).stubs().with(mockcpp::any()).will(returnValue(modules));

    hcomm::AicpuChannelEntityResources resources;
    ChannelHandle entityHandle = 0;
    EXPECT_EQ(
        resources.Build(
            std::vector<char>(), COMM_ENGINE_AICPU_TS, COMM_PROTOCOL_UB_CTP, static_cast<ChannelHandle>(0x1000U),
            entityHandle),
        HCCL_E_PARA);
}

TEST_F(AicpuChannelProcessTest, Ut_AicpuChannelEntityResourcesBuild_WhenLocalNotifyIdEmptySlot_Expect_Success)
{
    // UINT32_MAX 是空槽哨兵，允许转换为 -1，且不得被非法 ID 校验拒绝。
    using DeviceMemAllocRetByValue = hccl::DeviceMem (*)(u64, bool);
    mockcpp::mockAPI<DeviceMemAllocRetByValue>::get(
        "hccl::DeviceMem::alloc", "EntityResources_DeviceMem_alloc_by_value",
        static_cast<DeviceMemAllocRetByValue>(&hccl::DeviceMem::alloc))
        .stubs()
        .will(invoke(+[](u64 size, bool) {
            void* ptr = std::malloc(static_cast<size_t>(size));
            return hccl::DeviceMem(ptr, size, false);
        }));
    MOCKER(hrtMemSyncCopy)
        .stubs()
        .will(invoke(+[](void* dst, uint64_t, const void* src, uint64_t count, HcclRtMemcpyKind) {
            (void)std::memcpy(dst, src, count);
            return HCCL_SUCCESS;
        }));
    auto modules = BuildEntityResourcesModules(BuildLocalNotifyData(UINT32_MAX), BuildConnectionData(0U));
    MOCKER_CPP(&Hccl::AicpuResPackageHelper::ParsePackedData).stubs().with(mockcpp::any()).will(returnValue(modules));

    hcomm::AicpuChannelEntityResources resources;
    ChannelHandle entityHandle = 0;
    EXPECT_EQ(
        resources.Build(
            std::vector<char>(), COMM_ENGINE_AICPU_TS, COMM_PROTOCOL_UB_CTP, static_cast<ChannelHandle>(0x1000U),
            entityHandle),
        HCCL_SUCCESS);
    EXPECT_NE(entityHandle, 0U);
}

TEST_F(AicpuChannelProcessTest, Ut_AicpuChannelEntityResourcesBuild_WhenDrainDataTooLong_Expect_EPara)
{
    // 每条远端资源固定 28B；3 条 84B 不能被当作 2 条 42B 解析，必须整批拒绝。
    std::vector<char> drainData(84U, 0);
    auto modules = BuildEntityResourcesModules(std::vector<char>(), BuildConnectionData(0U), drainData);
    MOCKER_CPP(&Hccl::AicpuResPackageHelper::ParsePackedData).stubs().with(mockcpp::any()).will(returnValue(modules));

    hcomm::AicpuChannelEntityResources resources;
    ChannelHandle entityHandle = 0;
    EXPECT_EQ(
        resources.Build(
            std::vector<char>(), COMM_ENGINE_AICPU_TS, COMM_PROTOCOL_UB_CTP, static_cast<ChannelHandle>(0x1000U),
            entityHandle),
        HCCL_E_PARA);
}

TEST_F(AicpuChannelProcessTest, Ut_AicpuChannelEntityResourcesBuild_WhenRemoteNotifyIdOverflow_Expect_EPara)
{
    auto modules = BuildEntityResourcesModules(
        BuildLocalNotifyData(0U), BuildConnectionData(0U), std::vector<char>(), static_cast<uint32_t>(INT32_MAX) + 1U);
    MOCKER_CPP(&Hccl::AicpuResPackageHelper::ParsePackedData).stubs().with(mockcpp::any()).will(returnValue(modules));

    hcomm::AicpuChannelEntityResources resources;
    ChannelHandle entityHandle = 0;
    EXPECT_EQ(
        resources.Build(
            std::vector<char>(), COMM_ENGINE_AICPU_TS, COMM_PROTOCOL_UB_CTP, static_cast<ChannelHandle>(0x1000U),
            entityHandle),
        HCCL_E_PARA);
}

TEST_F(AicpuChannelProcessTest, Ut_AicpuChannelEntityResourcesBuild_WhenDrainNotifyIdOverflow_Expect_EPara)
{
    std::vector<char> drainData(BuildRemoteNotifyData(static_cast<uint32_t>(INT32_MAX) + 1U));
    drainData.insert(drainData.end(), 28U, 0);
    auto modules = BuildEntityResourcesModules(std::vector<char>(), BuildConnectionData(0U), drainData);
    MOCKER_CPP(&Hccl::AicpuResPackageHelper::ParsePackedData).stubs().with(mockcpp::any()).will(returnValue(modules));

    hcomm::AicpuChannelEntityResources resources;
    ChannelHandle entityHandle = 0;
    EXPECT_EQ(
        resources.Build(
            std::vector<char>(), COMM_ENGINE_AICPU_TS, COMM_PROTOCOL_UB_CTP, static_cast<ChannelHandle>(0x1000U),
            entityHandle),
        HCCL_E_PARA);
}

TEST_F(AicpuChannelProcessTest, Ut_AicpuChannelEntityResourcesBuild_WhenLocalNotifyDataTooLong_Expect_EPara)
{
    std::vector<char> notifyData = BuildLocalNotifyData(0U);
    notifyData.push_back(0);
    auto modules = BuildEntityResourcesModules(notifyData, BuildConnectionData(0U));
    MOCKER_CPP(&Hccl::AicpuResPackageHelper::ParsePackedData).stubs().with(mockcpp::any()).will(returnValue(modules));

    hcomm::AicpuChannelEntityResources resources;
    ChannelHandle entityHandle = 0;
    EXPECT_EQ(
        resources.Build(
            std::vector<char>(), COMM_ENGINE_AICPU_TS, COMM_PROTOCOL_UB_CTP, static_cast<ChannelHandle>(0x1000U),
            entityHandle),
        HCCL_E_PARA);
}

TEST_F(AicpuChannelProcessTest, Ut_ParsePackData_WhenTwoRoceBlobs_IndependentHandles_NoDangling)
{
    std::vector<char> data1 = BuildRocePackData(0x3000);
    std::vector<char> data2 = BuildRocePackData(0x4000);

    ChannelHandle handle1 = 0ULL;
    ChannelHandle handle2 = 0ULL;
    ASSERT_EQ(AicpuChannelProcess::ParsePackData(data1, handle1), HCCL_SUCCESS);
    ASSERT_EQ(AicpuChannelProcess::ParsePackData(data2, handle2), HCCL_SUCCESS);
    EXPECT_NE(handle1, 0ULL);
    EXPECT_NE(handle2, 0ULL);
    EXPECT_NE(handle1, handle2);
    EXPECT_EQ(AicpuChannelProcess::transportMap_.size(), static_cast<size_t>(2));

    // 独立销毁：删 handle1 后剩 1 条，handle2 仍存活
    HcclChannelUrmaRes destroyRes1{};
    destroyRes1.channelList = static_cast<void*>(&handle1);
    destroyRes1.listNum = 1U;
    EXPECT_EQ(AicpuChannelProcess::AicpuChannelDestroy(&destroyRes1), HCCL_SUCCESS);
    EXPECT_EQ(AicpuChannelProcess::transportMap_.size(), static_cast<size_t>(1));

    HcclChannelUrmaRes destroyRes2{};
    destroyRes2.channelList = static_cast<void*>(&handle2);
    destroyRes2.listNum = 1U;
    EXPECT_EQ(AicpuChannelProcess::AicpuChannelDestroy(&destroyRes2), HCCL_SUCCESS);
    EXPECT_EQ(AicpuChannelProcess::transportMap_.size(), static_cast<size_t>(0));
}

// ROCE 按 transpUniqueId 去重：同 uniqueId 重复注册（如算法切换后 HcclChannelAcquire 重走）复用原
// transport 对象，不新建条目，保持软件影子状态与 CQ 环/NIC 硬件状态连续。
// 复用使同一 transport 被两个通道句柄共享：引用计数保证销毁其一不影响另一句柄，
// 全部共享通道销毁后 transportMap_ 才真正清空。
TEST_F(AicpuChannelProcessTest, Ut_ParsePackData_WhenSameRoceUniqueId_ReusesRegisteredTransport)
{
    std::vector<char> data = BuildRocePackData();

    ChannelHandle handle1 = 0ULL;
    ChannelHandle handle2 = 0ULL;
    ASSERT_EQ(AicpuChannelProcess::ParsePackData(data, handle1), HCCL_SUCCESS);
    ASSERT_EQ(AicpuChannelProcess::ParsePackData(data, handle2), HCCL_SUCCESS);
    EXPECT_NE(handle1, 0ULL);
    EXPECT_EQ(handle1, handle2); // 复用原 transport
    EXPECT_EQ(AicpuChannelProcess::transportMap_.size(), static_cast<size_t>(1));

    // 销毁其中一个共享句柄: 引用计数递减, transport仍存活, 另一句柄仍可用
    HcclChannelUrmaRes destroyRes{};
    destroyRes.channelList = static_cast<void*>(&handle1);
    destroyRes.listNum = 1U;
    EXPECT_EQ(AicpuChannelProcess::AicpuChannelDestroy(&destroyRes), HCCL_SUCCESS);
    EXPECT_EQ(AicpuChannelProcess::transportMap_.size(), static_cast<size_t>(1));
    EXPECT_TRUE(hcomm::IsValidLiteTransportHandle(handle2));

    // 销毁最后一个共享句柄: 计数归零, transport真正析构, 表清空
    EXPECT_EQ(AicpuChannelProcess::AicpuChannelDestroy(&destroyRes), HCCL_SUCCESS);
    EXPECT_EQ(AicpuChannelProcess::transportMap_.size(), static_cast<size_t>(0));
    EXPECT_FALSE(hcomm::IsValidLiteTransportHandle(handle2));

    // 销毁后同uniqueId重注册: 反查表已随销毁清理, 应新建transport(旧handle已随析构失效),
    // 而非取回已析构对象的陈旧handle(捕获反查表未清理/stale handle回归)。
    // 注意不断言新handle地址与旧值不同: malloc可能复用同地址, 核心契约是"新对象存活在册"而非地址相异
    ChannelHandle handle3 = 0ULL;
    ASSERT_EQ(AicpuChannelProcess::ParsePackData(data, handle3), HCCL_SUCCESS);
    EXPECT_NE(handle3, 0ULL);
    EXPECT_EQ(AicpuChannelProcess::transportMap_.size(), static_cast<size_t>(1));
    EXPECT_TRUE(hcomm::IsValidLiteTransportHandle(handle3));

    // 清理新注册的条目, 避免污染后续用例
    HcclChannelUrmaRes destroyRes3{};
    destroyRes3.channelList = static_cast<void*>(&handle3);
    destroyRes3.listNum = 1U;
    EXPECT_EQ(AicpuChannelProcess::AicpuChannelDestroy(&destroyRes3), HCCL_SUCCESS);
    EXPECT_EQ(AicpuChannelProcess::transportMap_.size(), static_cast<size_t>(0));
}
