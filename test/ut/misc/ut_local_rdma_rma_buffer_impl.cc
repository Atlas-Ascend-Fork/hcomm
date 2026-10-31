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

#include <cstdint>
#include <memory>

#include "adapter_hccp.h"
#include "hccl_network.h"
#include "network_manager_pub.h"

#define private public
#define protected public
#include "local_rdma_rma_buffer_impl.h"
#undef protected
#undef private

using namespace hccl;

namespace {
u32 g_regMrCount = 0;
u32 g_deregMrCount = 0;
u32 g_fakeMrSeq = 0;
bool g_regFailOnce = false;
s32 g_fakeSocketHandle = 1;
constexpr u32 FAKE_LKEY_BASE = 1000;

HcclResult StubGetRaResourceInfo(NetworkManager* that, RaResourceInfo& raResourceInfo)
{
    static RaResourceInfo fakeRaResourceInfo;
    static bool initialized = false;
    if (!initialized) {
        IpSocket tmpIpSocket;
        tmpIpSocket.nicRdmaHandle = &g_fakeSocketHandle;
        fakeRaResourceInfo.nicSocketMap[HcclIpAddress(1684515008)] = tmpIpSocket;
        initialized = true;
    }
    raResourceInfo = fakeRaResourceInfo;
    return HCCL_SUCCESS;
}

HcclResult StubHrtRaRegGlobalMr(const RdmaHandle rdmaHandle, struct MrInfoT& mrInfo, MrHandle& mrHandle)
{
    if (g_regFailOnce) {
        return HCCL_E_INTERNAL;
    }
    ++g_regMrCount;
    ++g_fakeMrSeq;
    mrInfo.lkey = FAKE_LKEY_BASE + g_fakeMrSeq;
    mrHandle = reinterpret_cast<MrHandle>(static_cast<uintptr_t>(0x200000 + g_fakeMrSeq));
    return HCCL_SUCCESS;
}

HcclResult StubHrtRaDeRegGlobalMr(const RdmaHandle rdmaHandle, MrHandle mrHandle)
{
    ++g_deregMrCount;
    return HCCL_SUCCESS;
}
} // namespace

class LocalRdmaRmaBufferImplTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "LocalRdmaRmaBufferImpl tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "LocalRdmaRmaBufferImpl tests tear down." << std::endl; }

    virtual void SetUp()
    {
        g_regMrCount = 0;
        g_deregMrCount = 0;
        g_fakeMrSeq = 0;
        g_regFailOnce = false;
        MOCKER_CPP(&NetworkManager::StartHostNet).stubs().will(mockcpp::returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&NetworkManager::GetRaResourceInfo).stubs().will(mockcpp::invoke(StubGetRaResourceInfo));
        MOCKER(hrtRaRegGlobalMr).stubs().will(mockcpp::invoke(StubHrtRaRegGlobalMr));
        MOCKER(hrtRaDeRegGlobalMr).stubs().will(mockcpp::invoke(StubHrtRaDeRegGlobalMr));
    }

    virtual void TearDown() { GlobalMockObject::verify(); }

    // 每个 TEST 内的 static buffer 地址互不相同，保证 devAddrID 不串扰
    std::unique_ptr<hccl::NetDevContext> MakeNetDevCtx()
    {
        auto netDevCtx = std::make_unique<hccl::NetDevContext>();
        HcclIpAddress ipAddr(1684515008);
        EXPECT_EQ(netDevCtx->Init(NicType::HOST_NIC_TYPE, 1, 1, ipAddr), HCCL_SUCCESS);
        return netDevCtx;
    }
};

// 两次 Init 相同 (addr, size, rdmaHandle) 只注册一次 MR，第二个对象复用同一 mrHandle 与 lkey
TEST_F(LocalRdmaRmaBufferImplTest, Ut_Init_SameDevAddrID_Twice_RegistersOnceAndReuses)
{
    static u8 buffer[1024];
    auto netDevCtx = MakeNetDevCtx();
    hccl::LocalRdmaRmaBufferImpl buf1(
        reinterpret_cast<HcclNetDevCtx>(netDevCtx.get()), buffer, sizeof(buffer), RmaMemType::DEVICE);
    hccl::LocalRdmaRmaBufferImpl buf2(
        reinterpret_cast<HcclNetDevCtx>(netDevCtx.get()), buffer, sizeof(buffer), RmaMemType::DEVICE);

    EXPECT_EQ(buf1.Init(), HCCL_SUCCESS);
    EXPECT_EQ(buf2.Init(), HCCL_SUCCESS);

    EXPECT_EQ(g_regMrCount, 1U);
    EXPECT_EQ(buf2.mrHandle, buf1.mrHandle);
    EXPECT_EQ(buf2.GetKey(), buf1.GetKey());

    EXPECT_EQ(buf1.Destroy(), HCCL_SUCCESS);
    EXPECT_EQ(g_deregMrCount, 0U);
    EXPECT_EQ(buf2.Destroy(), HCCL_SUCCESS);
    EXPECT_EQ(g_deregMrCount, 1U);
}

// 两个持有者先后 Destroy：非最后持有者不解注册，仅最后一个持有者解注册一次
TEST_F(LocalRdmaRmaBufferImplTest, Ut_Destroy_TwiceByTwoOwners_DeRegOnlyOnceByLastOwner)
{
    static u8 buffer[512];
    auto netDevCtx = MakeNetDevCtx();
    hccl::LocalRdmaRmaBufferImpl buf1(
        reinterpret_cast<HcclNetDevCtx>(netDevCtx.get()), buffer, sizeof(buffer), RmaMemType::DEVICE);
    hccl::LocalRdmaRmaBufferImpl buf2(
        reinterpret_cast<HcclNetDevCtx>(netDevCtx.get()), buffer, sizeof(buffer), RmaMemType::DEVICE);

    ASSERT_EQ(buf1.Init(), HCCL_SUCCESS);
    ASSERT_EQ(buf2.Init(), HCCL_SUCCESS);
    ASSERT_EQ(g_regMrCount, 1U);

    EXPECT_EQ(buf2.Destroy(), HCCL_SUCCESS);
    EXPECT_EQ(g_deregMrCount, 0U);

    EXPECT_EQ(buf1.Destroy(), HCCL_SUCCESS);
    EXPECT_EQ(g_deregMrCount, 1U);
}

// mrHandle 非空但注册表条目缺失时，Destroy 不解注册且不崩溃
TEST_F(LocalRdmaRmaBufferImplTest, Ut_Destroy_MissingRegistryEntry_ExpectNoDeReg)
{
    static u8 buffer[256];
    auto netDevCtx = MakeNetDevCtx();
    hccl::LocalRdmaRmaBufferImpl buf(
        reinterpret_cast<HcclNetDevCtx>(netDevCtx.get()), buffer, sizeof(buffer), RmaMemType::DEVICE);

    buf.mrHandle = reinterpret_cast<MrHandle>(static_cast<uintptr_t>(0x300000));
    buf.devAddrID = "fake-dev-addr-id-not-in-registry";
    buf.initialized_ = true;
    buf.deviceLogicId = 1;

    EXPECT_EQ(buf.Destroy(), HCCL_SUCCESS);
    EXPECT_EQ(g_deregMrCount, 0U);
}

// 注册失败时 Init 返回错误；失败后重新 Init 可重新注册且不残留
TEST_F(LocalRdmaRmaBufferImplTest, Ut_Init_RegMrFail_ExpectErrorAndRetriable)
{
    static u8 buffer[128];
    auto netDevCtx = MakeNetDevCtx();
    hccl::LocalRdmaRmaBufferImpl buf(
        reinterpret_cast<HcclNetDevCtx>(netDevCtx.get()), buffer, sizeof(buffer), RmaMemType::DEVICE);

    g_regFailOnce = true;
    EXPECT_EQ(buf.Init(), HCCL_E_INTERNAL);
    EXPECT_EQ(g_regMrCount, 0U);
    g_regFailOnce = false;

    EXPECT_EQ(buf.Init(), HCCL_SUCCESS);
    EXPECT_EQ(g_regMrCount, 1U);
    EXPECT_EQ(buf.GetKey(), FAKE_LKEY_BASE + 1);

    EXPECT_EQ(buf.Destroy(), HCCL_SUCCESS);
    EXPECT_EQ(g_deregMrCount, 1U);
}
