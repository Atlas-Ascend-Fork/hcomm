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
#include "../../../hccl_api_base_test.h"
#include "legacy_dpu_thread/dpu_shmem_mgr.h"
#include "orion_adapter_rts.h"
#include "orion_adapter_hal.h"
#include "driver/ascend_hal.h"

using namespace hccl;
using Hccl::HrtFree;
using Hccl::HrtHalGetDeviceInfo;
using Hccl::HrtHalHostRegister;
using Hccl::HrtHalHostUnregister;
using Hccl::HrtMalloc;

namespace {
constexpr char kTag[] = "testTag";
constexpr uint64_t kAllocSize = 64;

// PCIE 分支 HrtMalloc 的假设备内存、HrtHalHostRegister 的假映射地址：必须用静态缓冲区承接 memset_s
// 首字节写入，桩若返回伪地址会被 memset_s 解引用崩溃
uint8_t g_pcieDevMem[256] = {0};
uint8_t g_accessMem[256] = {0};
HcclResult g_registerRet = HCCL_SUCCESS;
uint32_t g_hrtFreeCount = 0;
uint32_t g_registerCount = 0; // 注册动作计数：用于断言「被拒绝的申请没有做任何申请动作」

void StubHrtFree(void* devPtr)
{
    (void)devPtr;
    ++g_hrtFreeCount;
}

HcclResult StubHostRegister(void* srcPtr, uint64_t size, uint32_t flag, int32_t deviceLogicId, void** dstPtr)
{
    (void)srcPtr;
    (void)size;
    (void)flag;
    (void)deviceLogicId;
    ++g_registerCount;
    if (dstPtr != nullptr) {
        *dstPtr = static_cast<void*>(g_accessMem);
    }
    return g_registerRet;
}

HcclResult StubHostUnregister(void* ptr, int32_t deviceLogicId)
{
    (void)ptr;
    (void)deviceLogicId;
    return HCCL_SUCCESS;
}

// 桩 HrtHalGetDeviceInfo 走指定 connect type；UB 分支用真实 malloc + 4K 对齐，PCIE 分支用假设备内存
void MockShmemDeps(int64_t connectType)
{
    MOCKER(HrtHalGetDeviceInfo)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBound(connectType))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER(HrtHalHostRegister).stubs().will(invoke(StubHostRegister));
    MOCKER(HrtHalHostUnregister).stubs().will(invoke(StubHostUnregister));
    MOCKER(HrtMalloc).stubs().will(returnValue(static_cast<void*>(g_pcieDevMem)));
    MOCKER(HrtFree).stubs().will(invoke(StubHrtFree));
}
} // namespace

class TestDpuShmemMgr : public BaseInit {
public:
    void SetUp() override
    {
        BaseInit::SetUp();
        g_registerRet = HCCL_SUCCESS;
        g_hrtFreeCount = 0;
        g_registerCount = 0;
        memset(g_pcieDevMem, 0, sizeof(g_pcieDevMem));
        memset(g_accessMem, 0, sizeof(g_accessMem));
    }
    void TearDown() override
    {
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }
};

TEST_F(TestDpuShmemMgr, Ut_Alloc_When_Ub_Expect_HostVa_DevAccessVa)
{
    MockShmemDeps(HOST_DEVICE_CONNECT_TYPE_UB);

    DpuShmemMgr mgr;
    ASSERT_EQ(HCCL_SUCCESS, mgr.Alloc(kTag, kAllocSize, 0));
    ASSERT_TRUE(mgr.Has(kTag));
    const Hccl::DpuShmem* shmem = mgr.Get(kTag);
    ASSERT_NE(nullptr, shmem);
    EXPECT_EQ(HOST_DEVICE_CONNECT_TYPE_UB, shmem->connectType_);
    // UB 分支由 malloc + std::align 自行对齐：注册基址必须 4K 对齐（HrtHalHostRegister 的硬要求）
    EXPECT_EQ(0u, reinterpret_cast<uintptr_t>(shmem->va_) % 4096u);
    ASSERT_NE(nullptr, shmem->originVa_); // 原始块由 Free 按 originVa_ 释放
    // UB：host 侧取 va_，device 侧取 accessVA_（映射到 device 的 host 内存）
    EXPECT_EQ(shmem->va_, mgr.HostPtr(kTag));
    EXPECT_EQ(shmem->accessVA_, mgr.DevPtr(kTag));
    EXPECT_EQ(static_cast<void*>(g_accessMem), mgr.DevPtr(kTag));
    EXPECT_NE(mgr.HostPtr(kTag), mgr.DevPtr(kTag));

    mgr.Free(kTag, 0);
    EXPECT_FALSE(mgr.Has(kTag));
    EXPECT_EQ(nullptr, mgr.HostPtr(kTag));
    EXPECT_FALSE(mgr.AnyAllocated());
}

TEST_F(TestDpuShmemMgr, Ut_Alloc_When_Pcie_Expect_HostAccessVa_DevVa)
{
    MockShmemDeps(HOST_DEVICE_CONNECT_TYPE_PCIE);

    DpuShmemMgr mgr;
    ASSERT_EQ(HCCL_SUCCESS, mgr.Alloc(kTag, kAllocSize, 0));
    const Hccl::DpuShmem* shmem = mgr.Get(kTag);
    ASSERT_NE(nullptr, shmem);
    EXPECT_EQ(HOST_DEVICE_CONNECT_TYPE_PCIE, shmem->connectType_);
    // PCIE：device HBM 经 SVM 映射，host 侧取 accessVA_，device 侧取 va_（与 UB 正好互补）
    EXPECT_EQ(static_cast<void*>(g_accessMem), mgr.HostPtr(kTag));
    EXPECT_EQ(static_cast<void*>(g_pcieDevMem), mgr.DevPtr(kTag));
    EXPECT_NE(mgr.HostPtr(kTag), mgr.DevPtr(kTag));

    mgr.Free(kTag, 0);
    EXPECT_FALSE(mgr.Has(kTag));
    EXPECT_EQ(1u, g_hrtFreeCount); // PCIE 释放走 HrtFree
}

TEST_F(TestDpuShmemMgr, Ut_Alloc_When_UbRegisterFailed_Expect_Error_And_InPlaceCleanup)
{
    MockShmemDeps(HOST_DEVICE_CONNECT_TYPE_UB);
    g_registerRet = HCCL_E_INTERNAL;

    DpuShmemMgr mgr;
    EXPECT_EQ(HCCL_E_INTERNAL, mgr.Alloc(kTag, kAllocSize, 0));
    // 注册失败就地回收：未登记进 map，后续可重试
    EXPECT_FALSE(mgr.Has(kTag));
    EXPECT_FALSE(mgr.AnyAllocated());
    EXPECT_EQ(nullptr, mgr.HostPtr(kTag));
}

TEST_F(TestDpuShmemMgr, Ut_Alloc_When_PcieRegisterFailed_Expect_Error_And_HrtFree)
{
    MockShmemDeps(HOST_DEVICE_CONNECT_TYPE_PCIE);
    g_registerRet = HCCL_E_INTERNAL;

    DpuShmemMgr mgr;
    EXPECT_EQ(HCCL_E_INTERNAL, mgr.Alloc(kTag, kAllocSize, 0));
    EXPECT_FALSE(mgr.Has(kTag));
    EXPECT_FALSE(mgr.AnyAllocated());
    EXPECT_EQ(1u, g_hrtFreeCount); // PCIE 注册失败就地 HrtFree 回收
}

TEST_F(TestDpuShmemMgr, Ut_Alloc_When_MemsetFailed_Expect_E_MEMORY_And_InPlaceCleanup)
{
    MockShmemDeps(HOST_DEVICE_CONNECT_TYPE_PCIE);
    // 桩日志自己也走 memset_s（DlogRecord→sal_memset→memset_s），把它打成失败后「记日志失败→再记日志」
    // 会自激成无限递归爆栈；故窗口内一并把日志用的 sal_memset 桩成成功，日志路径不再依赖 memset_s
    MOCKER(sal_memset).stubs().will(returnValue(HCCL_SUCCESS));
    // 首字节清 0 失败：注册已成功（映射已建立），必须就地解注册再释放，否则该映射泄漏
    MOCKER(memset_s).stubs().with(mockcpp::any()).will(returnValue(1));

    DpuShmemMgr mgr;
    EXPECT_EQ(HCCL_E_MEMORY, mgr.Alloc(kTag, kAllocSize, 0));
    EXPECT_FALSE(mgr.Has(kTag));
    EXPECT_FALSE(mgr.AnyAllocated());
    EXPECT_EQ(1u, g_hrtFreeCount); // PCIE：ReleaseShmem 经 HrtFree 回收
}

TEST_F(TestDpuShmemMgr, Ut_Alloc_When_DuplicateTag_Expect_E_PARA)
{
    MockShmemDeps(HOST_DEVICE_CONNECT_TYPE_UB);

    DpuShmemMgr mgr;
    ASSERT_EQ(HCCL_SUCCESS, mgr.Alloc(kTag, kAllocSize, 0));
    // 同 tag 重复申请属调用方误用：拒绝覆盖旧映射，且拒绝必须发生在申请动作之前
    // （否则这一次注定失败的调用会白申请并注册一块新映射，旧块不动、新块无人释放）
    EXPECT_EQ(HCCL_E_PARA, mgr.Alloc(kTag, kAllocSize, 0));
    EXPECT_EQ(1u, g_registerCount); // 只有首次成功申请注册过，被拒绝的调用未触碰注册接口
    EXPECT_TRUE(mgr.Has(kTag));
    EXPECT_NE(nullptr, mgr.HostPtr(kTag));

    mgr.Free(kTag, 0);
    EXPECT_FALSE(mgr.AnyAllocated());
}

TEST_F(TestDpuShmemMgr, Ut_Alloc_When_UnsupportedConnectType_Expect_NotSupport)
{
    MockShmemDeps(HOST_DEVICE_CONNECT_TYPE_HCCS);

    DpuShmemMgr mgr;
    EXPECT_EQ(HCCL_E_NOT_SUPPORT, mgr.Alloc(kTag, kAllocSize, 0));
    EXPECT_FALSE(mgr.Has(kTag));
}

TEST_F(TestDpuShmemMgr, Ut_Free_When_MissingTag_Expect_Noop)
{
    MockShmemDeps(HOST_DEVICE_CONNECT_TYPE_UB);

    DpuShmemMgr mgr;
    mgr.Free(kTag, 0); // tag 不存在：幂等空操作，不崩溃
    EXPECT_FALSE(mgr.Has(kTag));
}

TEST_F(TestDpuShmemMgr, Ut_FreeAll_When_Multiple_Expect_AllReleased)
{
    MockShmemDeps(HOST_DEVICE_CONNECT_TYPE_UB);

    DpuShmemMgr mgr;
    ASSERT_EQ(HCCL_SUCCESS, mgr.Alloc("tagA", kAllocSize, 0));
    ASSERT_EQ(HCCL_SUCCESS, mgr.Alloc("tagB", kAllocSize, 0));
    EXPECT_TRUE(mgr.AnyAllocated());

    mgr.FreeAll(0);
    EXPECT_FALSE(mgr.AnyAllocated());
    EXPECT_FALSE(mgr.Has("tagA"));
    EXPECT_FALSE(mgr.Has("tagB"));
}
