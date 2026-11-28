/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 被测目标: HcommMemHostUnmapFromDev C API
//   根据 H2D 协议选择 rtHostUnRegisterByPciAtu / aclrtHostUnregister，
//   以映射时的 hostMem（hVA）为入参，解除 Host DDR 到 Device 侧 dVA 的映射。

#include <cstdint>

#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>

#include "acl/acl_rt.h"
#include "adapter_rts_common.h"
#include "hccl/hccl_types.h"
#include "hcomm_adapter_rts.h"
#include "hcomm_res.h"
#include "rt_external.h"

namespace {

constexpr uintptr_t HOST_ADDR = 0x1000;
constexpr uint64_t TEST_SIZE = 4096U;
constexpr rtError_t TEST_RT_ERROR = 1;

// 依赖注入状态：各用例只改状态，桩在 SetUp 中一次性注册。
// mockcpp 同一依赖「先注册先命中」，若每个用例各注册一次，后注册的桩不生效，故统一用状态桩。
aclError g_getDeviceRet = ACL_SUCCESS;
int32_t g_deviceLogicId = 3;
int64_t g_connectType = ACL_HOST_DEVICE_CONNECT_TYPE_PCIE;
DevType g_devType = DevType::DEV_TYPE_950;
HcclResult g_getDeviceTypeRet = HCCL_SUCCESS;
uint32_t g_getDeviceTypeCalls = 0;
aclError g_deviceInfoRet = ACL_SUCCESS;
aclError g_aclUnregisterRet = ACL_SUCCESS;
rtError_t g_pciAtuRet = RT_ERROR_NONE;
uint32_t g_aclUnregisterCalls = 0;
uint32_t g_pciAtuCalls = 0;
void* g_aclUnregisterPtr = nullptr;
void* g_pciAtuPtr = nullptr;

HcclResult StubGetDeviceType(DevType& devType)
{
    ++g_getDeviceTypeCalls;
    if (g_getDeviceTypeRet != HCCL_SUCCESS) {
        return g_getDeviceTypeRet;
    }
    devType = g_devType;
    return HCCL_SUCCESS;
}

aclError StubGetDevice(int32_t* deviceId)
{
    if (deviceId != nullptr) {
        *deviceId = g_deviceLogicId;
    }
    return g_getDeviceRet;
}

aclError StubGetDeviceInfo(uint32_t devId, aclrtDevAttr attr, int64_t* value)
{
    (void)devId;
    if (attr == ACL_DEV_ATTR_HD_CONNECT_TYPE && value != nullptr) {
        *value = g_connectType;
    }
    return g_deviceInfoRet;
}

aclError StubAclrtHostUnregister(void* ptr)
{
    ++g_aclUnregisterCalls;
    g_aclUnregisterPtr = ptr;
    return g_aclUnregisterRet;
}

rtError_t StubHostUnregisterByPciAtu(void* ptr)
{
    ++g_pciAtuCalls;
    g_pciAtuPtr = ptr;
    return g_pciAtuRet;
}

class TestHcommMemHostUnmapFromDev : public ::testing::Test {
protected:
    void SetUp() override
    {
        g_getDeviceRet = ACL_SUCCESS;
        g_deviceLogicId = 3;
        g_connectType = ACL_HOST_DEVICE_CONNECT_TYPE_PCIE;
        g_devType = DevType::DEV_TYPE_950;
        g_getDeviceTypeRet = HCCL_SUCCESS;
        g_getDeviceTypeCalls = 0;
        g_deviceInfoRet = ACL_SUCCESS;
        g_aclUnregisterRet = ACL_SUCCESS;
        g_pciAtuRet = RT_ERROR_NONE;
        g_aclUnregisterCalls = 0;
        g_pciAtuCalls = 0;
        g_aclUnregisterPtr = nullptr;
        g_pciAtuPtr = nullptr;

        MOCKER(aclrtGetDevice).stubs().will(invoke(StubGetDevice));
        MOCKER(aclrtGetDeviceInfo).stubs().will(invoke(StubGetDeviceInfo));
        MOCKER(hrtGetDeviceType).stubs().will(invoke(StubGetDeviceType));
        MOCKER(aclrtHostUnregister).stubs().will(invoke(StubAclrtHostUnregister));
        MOCKER(rtHostUnRegisterByPciAtu).stubs().will(invoke(StubHostUnregisterByPciAtu));
    }

    void TearDown() override { GlobalMockObject::verify(); }

    static CommMem MakeHostMem()
    {
        CommMem mem{};
        mem.type = COMM_MEM_TYPE_HOST;
        mem.addr = reinterpret_cast<void*>(HOST_ADDR);
        mem.size = TEST_SIZE;
        return mem;
    }
};

TEST_F(TestHcommMemHostUnmapFromDev, NullHostMemReturnsPtrError)
{
    EXPECT_EQ(HcommMemHostUnmapFromDev(nullptr), HCCL_E_PTR);
    EXPECT_EQ(g_aclUnregisterCalls, 0U);
    EXPECT_EQ(g_pciAtuCalls, 0U);
}

TEST_F(TestHcommMemHostUnmapFromDev, InvalidHostTypeReturnsParaError)
{
    CommMem hostMem = MakeHostMem();
    hostMem.type = COMM_MEM_TYPE_DEVICE;
    EXPECT_EQ(HcommMemHostUnmapFromDev(&hostMem), HCCL_E_PARA);
    EXPECT_EQ(g_aclUnregisterCalls, 0U);
    EXPECT_EQ(g_pciAtuCalls, 0U);
}

TEST_F(TestHcommMemHostUnmapFromDev, NullHostAddrReturnsParaError)
{
    CommMem hostMem = MakeHostMem();
    hostMem.addr = nullptr;
    EXPECT_EQ(HcommMemHostUnmapFromDev(&hostMem), HCCL_E_PARA);
    EXPECT_EQ(g_aclUnregisterCalls, 0U);
    EXPECT_EQ(g_pciAtuCalls, 0U);
}

TEST_F(TestHcommMemHostUnmapFromDev, GetDeviceFailReturnsRuntimeError)
{
    g_getDeviceRet = ACL_ERROR_RT_CONTEXT_NULL;
    CommMem hostMem = MakeHostMem();
    EXPECT_EQ(HcommMemHostUnmapFromDev(&hostMem), HCCL_E_RUNTIME);
    EXPECT_EQ(g_getDeviceTypeCalls, 0U);
    EXPECT_EQ(g_aclUnregisterCalls, 0U);
    EXPECT_EQ(g_pciAtuCalls, 0U);
}

TEST_F(TestHcommMemHostUnmapFromDev, GetDeviceInfoFailReturnsRuntimeError)
{
    g_deviceInfoRet = ACL_ERROR_RT_CONTEXT_NULL;
    CommMem hostMem = MakeHostMem();
    EXPECT_EQ(HcommMemHostUnmapFromDev(&hostMem), HCCL_E_RUNTIME);
    EXPECT_EQ(g_getDeviceTypeCalls, 0U);
    EXPECT_EQ(g_aclUnregisterCalls, 0U);
    EXPECT_EQ(g_pciAtuCalls, 0U);
}

TEST_F(TestHcommMemHostUnmapFromDev, GetDeviceInfoNotSupportFallsBackToAcl)
{
    g_connectType = ACL_HOST_DEVICE_CONNECT_TYPE_PCIE;
    g_devType = DevType::DEV_TYPE_950;
    g_deviceInfoRet = ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
    CommMem hostMem = MakeHostMem();
    EXPECT_EQ(HcommMemHostUnmapFromDev(&hostMem), HCCL_SUCCESS);
    EXPECT_EQ(g_getDeviceTypeCalls, 0U);
    EXPECT_EQ(g_pciAtuCalls, 0U);
    EXPECT_EQ(g_aclUnregisterCalls, 1U);
    EXPECT_EQ(g_aclUnregisterPtr, reinterpret_cast<void*>(HOST_ADDR));
}

TEST_F(TestHcommMemHostUnmapFromDev, PcieA5UsesPciAtuUnregister)
{
    g_connectType = ACL_HOST_DEVICE_CONNECT_TYPE_PCIE;
    g_devType = DevType::DEV_TYPE_950;
    CommMem hostMem = MakeHostMem();
    EXPECT_EQ(HcommMemHostUnmapFromDev(&hostMem), HCCL_SUCCESS);
    EXPECT_EQ(g_pciAtuCalls, 1U);
    EXPECT_EQ(g_aclUnregisterCalls, 0U);
    EXPECT_EQ(g_pciAtuPtr, reinterpret_cast<void*>(HOST_ADDR));
    EXPECT_EQ(g_getDeviceTypeCalls, 1U);
}

TEST_F(TestHcommMemHostUnmapFromDev, PcieA6UsesPciAtuUnregister)
{
    g_connectType = ACL_HOST_DEVICE_CONNECT_TYPE_PCIE;
    g_devType = DevType::DEV_TYPE_960;
    CommMem hostMem = MakeHostMem();
    EXPECT_EQ(HcommMemHostUnmapFromDev(&hostMem), HCCL_SUCCESS);
    EXPECT_EQ(g_pciAtuCalls, 1U);
    EXPECT_EQ(g_aclUnregisterCalls, 0U);
    EXPECT_EQ(g_pciAtuPtr, reinterpret_cast<void*>(HOST_ADDR));
    EXPECT_EQ(g_getDeviceTypeCalls, 1U);
}

TEST_F(TestHcommMemHostUnmapFromDev, PcieNonA5A6UsesAclrtHostUnregister)
{
    g_connectType = ACL_HOST_DEVICE_CONNECT_TYPE_PCIE;
    g_devType = DevType::DEV_TYPE_910B;
    CommMem hostMem = MakeHostMem();
    EXPECT_EQ(HcommMemHostUnmapFromDev(&hostMem), HCCL_SUCCESS);
    EXPECT_EQ(g_aclUnregisterCalls, 1U);
    EXPECT_EQ(g_pciAtuCalls, 0U);
    EXPECT_EQ(g_aclUnregisterPtr, reinterpret_cast<void*>(HOST_ADDR));
    EXPECT_EQ(g_getDeviceTypeCalls, 1U);
}

TEST_F(TestHcommMemHostUnmapFromDev, GetDeviceTypeFailReturnsRuntimeError)
{
    g_connectType = ACL_HOST_DEVICE_CONNECT_TYPE_PCIE;
    g_getDeviceTypeRet = HCCL_E_INTERNAL;
    CommMem hostMem = MakeHostMem();
    EXPECT_EQ(HcommMemHostUnmapFromDev(&hostMem), HCCL_E_RUNTIME);
    EXPECT_EQ(g_aclUnregisterCalls, 0U);
    EXPECT_EQ(g_pciAtuCalls, 0U);
}

TEST_F(TestHcommMemHostUnmapFromDev, NonPcieUsesAclrtHostUnregister)
{
    g_connectType = ACL_HOST_DEVICE_CONNECT_TYPE_UB;
    CommMem hostMem = MakeHostMem();
    EXPECT_EQ(HcommMemHostUnmapFromDev(&hostMem), HCCL_SUCCESS);
    EXPECT_EQ(g_aclUnregisterCalls, 1U);
    EXPECT_EQ(g_pciAtuCalls, 0U);
    EXPECT_EQ(g_aclUnregisterPtr, reinterpret_cast<void*>(HOST_ADDR));
    EXPECT_EQ(g_getDeviceTypeCalls, 0U);
}

TEST_F(TestHcommMemHostUnmapFromDev, PciAtuFailReturnsRuntimeError)
{
    g_connectType = ACL_HOST_DEVICE_CONNECT_TYPE_PCIE;
    g_pciAtuRet = TEST_RT_ERROR;
    CommMem hostMem = MakeHostMem();
    EXPECT_EQ(HcommMemHostUnmapFromDev(&hostMem), HCCL_E_RUNTIME);
    EXPECT_EQ(g_pciAtuCalls, 1U);
    EXPECT_EQ(g_aclUnregisterCalls, 0U);
}

TEST_F(TestHcommMemHostUnmapFromDev, AclrtHostUnregisterFailReturnsRuntimeError)
{
    g_connectType = ACL_HOST_DEVICE_CONNECT_TYPE_UB;
    g_aclUnregisterRet = ACL_ERROR_RT_PARAM_INVALID;
    CommMem hostMem = MakeHostMem();
    EXPECT_EQ(HcommMemHostUnmapFromDev(&hostMem), HCCL_E_RUNTIME);
    EXPECT_EQ(g_aclUnregisterCalls, 1U);
    EXPECT_EQ(g_pciAtuCalls, 0U);
    EXPECT_EQ(g_getDeviceTypeCalls, 0U);
}

} // namespace
