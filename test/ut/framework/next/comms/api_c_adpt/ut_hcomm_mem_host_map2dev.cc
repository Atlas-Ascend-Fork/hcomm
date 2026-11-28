/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 被测目标: HcommMemHostMap2Dev C API
//   根据 H2D 协议选择 rtHostRegisterByPciAtu / aclrtHostRegister，
//   将 Host DDR 的 hVA 映射为 Device 侧可访问的 dVA。

#include <cstdint>

#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>

#include "acl/acl_rt.h"
#include "adapter_rts_common.h"
#include "driver/ascend_hal.h"
#include "hccl/hccl_types.h"
#include "hcomm_adapter_rts.h"
#include "hcomm_res.h"
#include "rt_external.h"

namespace {

constexpr uintptr_t HOST_ADDR = 0x1000;
constexpr uintptr_t DEV_ADDR = 0x2000;
constexpr HcommMemMapType TEST_MAP_TYPE = HCOMM_MEM_MAP_DEFAULT;
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
aclError g_aclRegisterRet = ACL_SUCCESS;
aclError g_aclGetDevicePtrRet = ACL_SUCCESS;
aclError g_aclUnregisterRet = ACL_SUCCESS;
rtError_t g_pciAtuRet = RT_ERROR_NONE;
uint32_t g_aclRegisterCalls = 0;
uint32_t g_aclGetDevicePtrCalls = 0;
uint32_t g_aclUnregisterCalls = 0;
uint32_t g_pciAtuCalls = 0;
uint32_t g_pciAtuUnregisterCalls = 0;
uint32_t g_aclRegisterFlag = 0U;
uint32_t g_pciAtuFlag = 0U;
bool g_pciAtuReturnNull = false;
void* g_aclRegisterPtr = nullptr;
uint64_t g_aclRegisterSize = 0U;
void* g_aclGetDevicePtrHost = nullptr;
uint32_t g_aclGetDevicePtrFlag = 0U;
void* g_pciAtuPtr = nullptr;
uint64_t g_pciAtuSize = 0U;
void* g_pciAtuUnregisterPtr = nullptr;

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

aclError StubAclrtHostRegisterV2(void* ptr, uint64_t size, uint32_t flag)
{
    ++g_aclRegisterCalls;
    g_aclRegisterPtr = ptr;
    g_aclRegisterSize = size;
    g_aclRegisterFlag = flag;
    return g_aclRegisterRet;
}

aclError StubAclrtHostGetDevicePointer(void* pHost, void** pDevice, uint32_t flag)
{
    ++g_aclGetDevicePtrCalls;
    g_aclGetDevicePtrHost = pHost;
    g_aclGetDevicePtrFlag = flag;
    if (pDevice != nullptr) {
        *pDevice = reinterpret_cast<void*>(DEV_ADDR);
    }
    return g_aclGetDevicePtrRet;
}

aclError StubAclrtHostUnregister(void* ptr)
{
    (void)ptr;
    ++g_aclUnregisterCalls;
    return g_aclUnregisterRet;
}

rtError_t StubHostRegisterByPciAtu(void* ptr, uint64_t size, uint32_t flag, void** devPtr)
{
    ++g_pciAtuCalls;
    g_pciAtuPtr = ptr;
    g_pciAtuSize = size;
    g_pciAtuFlag = flag;
    if (devPtr != nullptr && !g_pciAtuReturnNull) {
        *devPtr = reinterpret_cast<void*>(DEV_ADDR);
    }
    return g_pciAtuRet;
}

rtError_t StubHostUnregisterByPciAtu(void* ptr)
{
    ++g_pciAtuUnregisterCalls;
    g_pciAtuUnregisterPtr = ptr;
    return RT_ERROR_NONE;
}

class TestHcommMemHostMap2Dev : public ::testing::Test {
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
        g_aclRegisterRet = ACL_SUCCESS;
        g_aclGetDevicePtrRet = ACL_SUCCESS;
        g_aclUnregisterRet = ACL_SUCCESS;
        g_pciAtuRet = RT_ERROR_NONE;
        g_aclRegisterCalls = 0;
        g_aclGetDevicePtrCalls = 0;
        g_aclUnregisterCalls = 0;
        g_pciAtuCalls = 0;
        g_pciAtuUnregisterCalls = 0;
        g_aclRegisterFlag = 0U;
        g_pciAtuFlag = 0U;
        g_pciAtuReturnNull = false;
        g_aclRegisterPtr = nullptr;
        g_aclRegisterSize = 0U;
        g_aclGetDevicePtrHost = nullptr;
        g_aclGetDevicePtrFlag = 0U;
        g_pciAtuPtr = nullptr;
        g_pciAtuSize = 0U;
        g_pciAtuUnregisterPtr = nullptr;

        MOCKER(aclrtGetDevice).stubs().will(invoke(StubGetDevice));
        MOCKER(aclrtGetDeviceInfo).stubs().will(invoke(StubGetDeviceInfo));
        MOCKER(hrtGetDeviceType).stubs().will(invoke(StubGetDeviceType));
        MOCKER(aclrtHostRegisterV2).stubs().will(invoke(StubAclrtHostRegisterV2));
        MOCKER(aclrtHostGetDevicePointer).stubs().will(invoke(StubAclrtHostGetDevicePointer));
        MOCKER(aclrtHostUnregister).stubs().will(invoke(StubAclrtHostUnregister));
        MOCKER(rtHostRegisterByPciAtu).stubs().will(invoke(StubHostRegisterByPciAtu));
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

    static CommMem MakeInvalidDevMem()
    {
        CommMem mem{};
        mem.type = COMM_MEM_TYPE_INVALID;
        mem.addr = nullptr;
        mem.size = 0U;
        return mem;
    }

    static void ExpectMapped(const CommMem& devMem)
    {
        EXPECT_EQ(devMem.type, COMM_MEM_TYPE_DEVICE);
        EXPECT_EQ(devMem.addr, reinterpret_cast<void*>(DEV_ADDR));
        EXPECT_EQ(devMem.size, TEST_SIZE);
    }

    static void ExpectUntouched(const CommMem& devMem)
    {
        EXPECT_EQ(devMem.type, COMM_MEM_TYPE_INVALID);
        EXPECT_EQ(devMem.addr, nullptr);
        EXPECT_EQ(devMem.size, 0U);
    }
};

TEST_F(TestHcommMemHostMap2Dev, NullHostMemReturnsPtrError)
{
    CommMem devMem = MakeInvalidDevMem();
    EXPECT_EQ(HcommMemHostMap2Dev(nullptr, &devMem, TEST_MAP_TYPE), HCCL_E_PTR);
    ExpectUntouched(devMem);
}

TEST_F(TestHcommMemHostMap2Dev, NullDevMemReturnsPtrError)
{
    CommMem hostMem = MakeHostMem();
    EXPECT_EQ(HcommMemHostMap2Dev(&hostMem, nullptr, TEST_MAP_TYPE), HCCL_E_PTR);
}

TEST_F(TestHcommMemHostMap2Dev, InvalidHostTypeReturnsParaError)
{
    CommMem hostMem = MakeHostMem();
    hostMem.type = COMM_MEM_TYPE_DEVICE;
    CommMem devMem = MakeInvalidDevMem();
    EXPECT_EQ(HcommMemHostMap2Dev(&hostMem, &devMem, TEST_MAP_TYPE), HCCL_E_PARA);
    ExpectUntouched(devMem);
}

TEST_F(TestHcommMemHostMap2Dev, NullHostAddrReturnsParaError)
{
    CommMem hostMem = MakeHostMem();
    hostMem.addr = nullptr;
    CommMem devMem = MakeInvalidDevMem();
    EXPECT_EQ(HcommMemHostMap2Dev(&hostMem, &devMem, TEST_MAP_TYPE), HCCL_E_PARA);
    ExpectUntouched(devMem);
}

TEST_F(TestHcommMemHostMap2Dev, ZeroHostSizeReturnsParaError)
{
    CommMem hostMem = MakeHostMem();
    hostMem.size = 0U;
    CommMem devMem = MakeInvalidDevMem();
    EXPECT_EQ(HcommMemHostMap2Dev(&hostMem, &devMem, TEST_MAP_TYPE), HCCL_E_PARA);
    ExpectUntouched(devMem);
}

TEST_F(TestHcommMemHostMap2Dev, GetDeviceFailReturnsRuntimeError)
{
    g_getDeviceRet = ACL_ERROR_RT_CONTEXT_NULL;
    CommMem hostMem = MakeHostMem();
    CommMem devMem = MakeInvalidDevMem();
    EXPECT_EQ(HcommMemHostMap2Dev(&hostMem, &devMem, TEST_MAP_TYPE), HCCL_E_RUNTIME);
    ExpectUntouched(devMem);
    EXPECT_EQ(g_getDeviceTypeCalls, 0U);
    EXPECT_EQ(g_aclRegisterCalls, 0U);
    EXPECT_EQ(g_pciAtuCalls, 0U);
}

TEST_F(TestHcommMemHostMap2Dev, GetDeviceInfoFailPropagates)
{
    g_deviceInfoRet = ACL_ERROR_RT_CONTEXT_NULL;
    CommMem hostMem = MakeHostMem();
    CommMem devMem = MakeInvalidDevMem();
    EXPECT_EQ(HcommMemHostMap2Dev(&hostMem, &devMem, TEST_MAP_TYPE), HCCL_E_RUNTIME);
    ExpectUntouched(devMem);
    EXPECT_EQ(g_getDeviceTypeCalls, 0U);
    EXPECT_EQ(g_aclRegisterCalls, 0U);
    EXPECT_EQ(g_pciAtuCalls, 0U);
}

TEST_F(TestHcommMemHostMap2Dev, GetDeviceInfoNotSupportFallsBackToAcl)
{
    g_connectType = ACL_HOST_DEVICE_CONNECT_TYPE_PCIE;
    g_devType = DevType::DEV_TYPE_950;
    g_deviceInfoRet = ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
    CommMem hostMem = MakeHostMem();
    CommMem devMem = MakeInvalidDevMem();
    EXPECT_EQ(HcommMemHostMap2Dev(&hostMem, &devMem, TEST_MAP_TYPE), HCCL_SUCCESS);
    ExpectMapped(devMem);
    EXPECT_EQ(g_getDeviceTypeCalls, 0U);
    EXPECT_EQ(g_pciAtuCalls, 0U);
    EXPECT_EQ(g_aclRegisterCalls, 1U);
    EXPECT_EQ(g_aclRegisterFlag, ACL_HOST_REG_MAPPED);
}

TEST_F(TestHcommMemHostMap2Dev, PcieA5UsesPciAtu)
{
    g_connectType = ACL_HOST_DEVICE_CONNECT_TYPE_PCIE;
    g_devType = DevType::DEV_TYPE_950;
    CommMem hostMem = MakeHostMem();
    CommMem devMem = MakeInvalidDevMem();
    EXPECT_EQ(HcommMemHostMap2Dev(&hostMem, &devMem, TEST_MAP_TYPE), HCCL_SUCCESS);
    ExpectMapped(devMem);
    EXPECT_EQ(g_pciAtuCalls, 1U);
    EXPECT_EQ(g_aclRegisterCalls, 0U);
    EXPECT_EQ(g_pciAtuFlag, ACL_HOST_REG_MAPPED);
    EXPECT_EQ(g_pciAtuPtr, reinterpret_cast<void*>(HOST_ADDR));
    EXPECT_EQ(g_pciAtuSize, TEST_SIZE);
    EXPECT_EQ(g_aclGetDevicePtrCalls, 0U);
    EXPECT_EQ(g_getDeviceTypeCalls, 1U);
}

TEST_F(TestHcommMemHostMap2Dev, PciAtuNullDevPtrRollsBack)
{
    g_connectType = ACL_HOST_DEVICE_CONNECT_TYPE_PCIE;
    g_devType = DevType::DEV_TYPE_950;
    g_pciAtuReturnNull = true;
    CommMem hostMem = MakeHostMem();
    CommMem devMem = MakeInvalidDevMem();
    EXPECT_EQ(HcommMemHostMap2Dev(&hostMem, &devMem, TEST_MAP_TYPE), HCCL_E_RUNTIME);
    ExpectUntouched(devMem);
    EXPECT_EQ(g_pciAtuCalls, 1U);
    EXPECT_EQ(g_pciAtuUnregisterCalls, 1U);
    EXPECT_EQ(g_pciAtuUnregisterPtr, reinterpret_cast<void*>(HOST_ADDR));
}

TEST_F(TestHcommMemHostMap2Dev, PcieA6UsesPciAtu)
{
    g_connectType = ACL_HOST_DEVICE_CONNECT_TYPE_PCIE;
    g_devType = DevType::DEV_TYPE_960;
    CommMem hostMem = MakeHostMem();
    CommMem devMem = MakeInvalidDevMem();
    EXPECT_EQ(HcommMemHostMap2Dev(&hostMem, &devMem, TEST_MAP_TYPE), HCCL_SUCCESS);
    ExpectMapped(devMem);
    EXPECT_EQ(g_pciAtuCalls, 1U);
    EXPECT_EQ(g_aclRegisterCalls, 0U);
    EXPECT_EQ(g_pciAtuFlag, ACL_HOST_REG_MAPPED);
    EXPECT_EQ(g_pciAtuPtr, reinterpret_cast<void*>(HOST_ADDR));
    EXPECT_EQ(g_pciAtuSize, TEST_SIZE);
    EXPECT_EQ(g_getDeviceTypeCalls, 1U);
}

TEST_F(TestHcommMemHostMap2Dev, PcieA5MappedPinnedReturnsNotSupport)
{
    g_connectType = ACL_HOST_DEVICE_CONNECT_TYPE_PCIE;
    g_devType = DevType::DEV_TYPE_950;
    CommMem hostMem = MakeHostMem();
    CommMem devMem = MakeInvalidDevMem();
    EXPECT_EQ(HcommMemHostMap2Dev(&hostMem, &devMem, HCOMM_MEM_MAP_PINNED), HCCL_E_NOT_SUPPORT);
    ExpectUntouched(devMem);
    EXPECT_EQ(g_pciAtuCalls, 0U);
    EXPECT_EQ(g_aclRegisterCalls, 0U);
}

TEST_F(TestHcommMemHostMap2Dev, PcieNonA5A6UsesAclrtHostRegisterV2)
{
    g_connectType = ACL_HOST_DEVICE_CONNECT_TYPE_PCIE;
    g_devType = DevType::DEV_TYPE_910B;
    CommMem hostMem = MakeHostMem();
    CommMem devMem = MakeInvalidDevMem();
    EXPECT_EQ(HcommMemHostMap2Dev(&hostMem, &devMem, TEST_MAP_TYPE), HCCL_SUCCESS);
    ExpectMapped(devMem);
    EXPECT_EQ(g_pciAtuCalls, 0U);
    EXPECT_EQ(g_aclRegisterCalls, 1U);
    EXPECT_EQ(g_aclRegisterFlag, ACL_HOST_REG_MAPPED);
    EXPECT_EQ(g_aclRegisterPtr, reinterpret_cast<void*>(HOST_ADDR));
    EXPECT_EQ(g_aclRegisterSize, TEST_SIZE);
    EXPECT_EQ(g_aclGetDevicePtrCalls, 1U);
    EXPECT_EQ(g_aclGetDevicePtrHost, reinterpret_cast<void*>(HOST_ADDR));
    EXPECT_EQ(g_aclGetDevicePtrFlag, 0U);
    EXPECT_EQ(g_getDeviceTypeCalls, 1U);
}

TEST_F(TestHcommMemHostMap2Dev, GetDeviceTypeFailReturnsRuntimeError)
{
    g_connectType = ACL_HOST_DEVICE_CONNECT_TYPE_PCIE;
    g_getDeviceTypeRet = HCCL_E_INTERNAL;
    CommMem hostMem = MakeHostMem();
    CommMem devMem = MakeInvalidDevMem();
    EXPECT_EQ(HcommMemHostMap2Dev(&hostMem, &devMem, TEST_MAP_TYPE), HCCL_E_RUNTIME);
    ExpectUntouched(devMem);
    EXPECT_EQ(g_aclRegisterCalls, 0U);
    EXPECT_EQ(g_pciAtuCalls, 0U);
}

TEST_F(TestHcommMemHostMap2Dev, NonPcieUsesAclrtHostRegisterV2)
{
    g_connectType = ACL_HOST_DEVICE_CONNECT_TYPE_UB;
    CommMem hostMem = MakeHostMem();
    CommMem devMem = MakeInvalidDevMem();
    EXPECT_EQ(HcommMemHostMap2Dev(&hostMem, &devMem, TEST_MAP_TYPE), HCCL_SUCCESS);
    ExpectMapped(devMem);
    EXPECT_EQ(g_aclRegisterCalls, 1U);
    EXPECT_EQ(g_pciAtuCalls, 0U);
    EXPECT_EQ(g_aclRegisterFlag, ACL_HOST_REG_MAPPED);
    EXPECT_EQ(g_aclRegisterPtr, reinterpret_cast<void*>(HOST_ADDR));
    EXPECT_EQ(g_aclRegisterSize, TEST_SIZE);
    EXPECT_EQ(g_aclGetDevicePtrCalls, 1U);
    EXPECT_EQ(g_aclGetDevicePtrHost, reinterpret_cast<void*>(HOST_ADDR));
    EXPECT_EQ(g_aclGetDevicePtrFlag, 0U);
    EXPECT_EQ(g_getDeviceTypeCalls, 0U);
}

TEST_F(TestHcommMemHostMap2Dev, NonPcieMappedPinnedUsesPinnedFlag)
{
    g_connectType = ACL_HOST_DEVICE_CONNECT_TYPE_UB;
    CommMem hostMem = MakeHostMem();
    CommMem devMem = MakeInvalidDevMem();
    EXPECT_EQ(HcommMemHostMap2Dev(&hostMem, &devMem, HCOMM_MEM_MAP_PINNED), HCCL_SUCCESS);
    ExpectMapped(devMem);
    EXPECT_EQ(g_aclRegisterCalls, 1U);
    EXPECT_EQ(g_pciAtuCalls, 0U);
    EXPECT_EQ(g_aclRegisterFlag, static_cast<uint32_t>(ACL_HOST_REG_MAPPED | ACL_HOST_REG_PINNED));
    EXPECT_EQ(g_aclRegisterPtr, reinterpret_cast<void*>(HOST_ADDR));
    EXPECT_EQ(g_aclRegisterSize, TEST_SIZE);
    EXPECT_EQ(g_aclGetDevicePtrCalls, 1U);
    EXPECT_EQ(g_getDeviceTypeCalls, 0U);
}

TEST_F(TestHcommMemHostMap2Dev, InvalidRegTypeReturnsParaError)
{
    g_connectType = ACL_HOST_DEVICE_CONNECT_TYPE_UB;
    CommMem hostMem = MakeHostMem();
    CommMem devMem = MakeInvalidDevMem();
    EXPECT_EQ(HcommMemHostMap2Dev(&hostMem, &devMem, static_cast<HcommMemMapType>(99)), HCCL_E_PARA);
    ExpectUntouched(devMem);
    EXPECT_EQ(g_aclRegisterCalls, 0U);
    EXPECT_EQ(g_pciAtuCalls, 0U);
}

TEST_F(TestHcommMemHostMap2Dev, PciAtuFailReturnsRuntimeError)
{
    g_connectType = ACL_HOST_DEVICE_CONNECT_TYPE_PCIE;
    g_pciAtuRet = TEST_RT_ERROR;
    CommMem hostMem = MakeHostMem();
    CommMem devMem = MakeInvalidDevMem();
    EXPECT_EQ(HcommMemHostMap2Dev(&hostMem, &devMem, TEST_MAP_TYPE), HCCL_E_RUNTIME);
    ExpectUntouched(devMem);
    EXPECT_EQ(g_pciAtuCalls, 1U);
    EXPECT_EQ(g_aclRegisterCalls, 0U);
}

TEST_F(TestHcommMemHostMap2Dev, AclrtHostRegisterV2FailReturnsRuntimeError)
{
    g_connectType = ACL_HOST_DEVICE_CONNECT_TYPE_UB;
    g_aclRegisterRet = ACL_ERROR_RT_PARAM_INVALID;
    CommMem hostMem = MakeHostMem();
    CommMem devMem = MakeInvalidDevMem();
    EXPECT_EQ(HcommMemHostMap2Dev(&hostMem, &devMem, TEST_MAP_TYPE), HCCL_E_RUNTIME);
    ExpectUntouched(devMem);
    EXPECT_EQ(g_aclRegisterCalls, 1U);
    EXPECT_EQ(g_aclGetDevicePtrCalls, 0U);
    EXPECT_EQ(g_aclUnregisterCalls, 0U);
    EXPECT_EQ(g_pciAtuCalls, 0U);
}

TEST_F(TestHcommMemHostMap2Dev, AclrtHostGetDevicePointerFailRollsBack)
{
    g_connectType = ACL_HOST_DEVICE_CONNECT_TYPE_UB;
    g_aclGetDevicePtrRet = ACL_ERROR_RT_PARAM_INVALID;
    CommMem hostMem = MakeHostMem();
    CommMem devMem = MakeInvalidDevMem();
    EXPECT_EQ(HcommMemHostMap2Dev(&hostMem, &devMem, TEST_MAP_TYPE), HCCL_E_RUNTIME);
    ExpectUntouched(devMem);
    EXPECT_EQ(g_aclRegisterCalls, 1U);
    EXPECT_EQ(g_aclGetDevicePtrCalls, 1U);
    EXPECT_EQ(g_aclUnregisterCalls, 1U);
    EXPECT_EQ(g_pciAtuCalls, 0U);
}

} // namespace
