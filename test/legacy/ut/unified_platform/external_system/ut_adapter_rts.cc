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
#define protected public
#include "kernel_param_lite.h"
#include "not_support_exception.h"
#include "orion_adapter_rts.h"
#include "runtime_api_exception.h"
#include "invalid_params_exception.h"
#include "null_ptr_exception.h"
#undef protected
#undef private
#include "dtype_common.h"
using namespace Hccl;

static constexpr int32_t RT_NOT_SUPPORT = 207000;

rtError_t rtDevBinaryUnRegister(void* hdl);

static aclError StubAclrtNotifyGetExportKey(aclrtNotify notify, char* key, size_t len, uint64_t flags)
{
    if (key != nullptr && len > 0) {
        strncpy(key, "fakeName", len);
    }
    return ACL_SUCCESS;
}

#include "adapter_rts_test_common.h"

TEST_F(AdapterRtsTest, HrtGetSocVer_return_nok)
{
    MOCKER(aclrtGetSocName).stubs().will(returnValue((const char*)NULL));

    std::string socName;
    EXPECT_EQ(HrtGetSocVer(socName), HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, HrtGetDeviceType_return_nok)
{
    // Given
    MOCKER(hrtGetDeviceType).stubs().will(returnValue(HCCL_E_INTERNAL));

    // when

    // then
    Hccl::DevType devType;
    EXPECT_EQ(HrtGetDeviceType(devType), HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, HrtGetDevicePhyIdByUserDevId_return_zero)
{
    // Given
    Hccl::DevType globalDevType = Hccl::DevType::DEV_TYPE_NOSOC;
    MOCKER(HrtGetDeviceType).stubs().with(outBound(globalDevType)).will(returnValue(HcclResult::HCCL_SUCCESS));

    // when
    u32 deviceLogicId = 1;
    DevId devicePhyId = 1;
    EXPECT_EQ(HrtGetDevicePhyIdByUserDevId(deviceLogicId, devicePhyId), HcclResult::HCCL_SUCCESS);

    u32 result = 0;
    // then
    EXPECT_EQ(result, devicePhyId);
}

TEST_F(AdapterRtsTest, HrtGetDevicePhyIdByUserDevId_return_ok)
{
    // Given
    Hccl::DevType globalDevType = Hccl::DevType::DEV_TYPE_950;
    MOCKER(HrtGetDeviceType).stubs().with(outBound(globalDevType)).will(returnValue(HcclResult::HCCL_SUCCESS));

    int32_t fakeDevicePhyId = 0;
    MOCKER(aclrtGetPhyDevIdByUserDevId)
        .stubs()
        .with(mockcpp::any(), outBoundP(&fakeDevicePhyId, sizeof(fakeDevicePhyId)))
        .will(returnValue(RT_ERROR_NONE));
    // when
    int32_t deviceLogicId = 1;
    DevId devicePhyId = 1;
    EXPECT_EQ(HrtGetDevicePhyIdByUserDevId(deviceLogicId, devicePhyId), HcclResult::HCCL_SUCCESS);

    // then
    EXPECT_EQ(static_cast<u32>(fakeDevicePhyId), devicePhyId);
}

TEST_F(AdapterRtsTest, HrtGetDevicePhyIdByUserDevId_return_nok)
{
    // Given
    MOCKER(HrtGetDeviceType)
        .stubs()
        .with(outBound(Hccl::DevType(Hccl::DevType::DEV_TYPE_950)))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(aclrtGetPhyDevIdByUserDevId).stubs().will(returnValue(1));

    // when

    // then
    DevId devicePhyIdOut;
    EXPECT_EQ(HrtGetDevicePhyIdByUserDevId(32, devicePhyIdOut), HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, HrtDeviceGetBareTgid_return_ok)
{
    // Given
    int32_t fakePid = 123;
    MOCKER(aclrtDeviceGetBareTgid).stubs().with(outBoundP(&fakePid, sizeof(fakePid))).will(returnValue(RT_ERROR_NONE));

    // when
    int32_t pid = 0;
    EXPECT_EQ(HrtDeviceGetBareTgid(pid), HcclResult::HCCL_SUCCESS);

    // then
    EXPECT_EQ(fakePid, pid);
}

TEST_F(AdapterRtsTest, HrtDeviceGetBareTgid_return_nok)
{
    // Given
    MOCKER(aclrtDeviceGetBareTgid).stubs().will(returnValue(1));

    // when

    // then
    s32 pid;
    EXPECT_EQ(HrtDeviceGetBareTgid(pid), HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, HrtGetDevice_return_ok)
{
    // Given
    s32 fakeDeviceLogicId = 123;
    MOCKER(aclrtGetDevice)
        .stubs()
        .with(outBoundP(&fakeDeviceLogicId, sizeof(fakeDeviceLogicId)))
        .will(returnValue(RT_ERROR_NONE));

    // when
    s32 deviceLogicId = 0;
    EXPECT_EQ(HrtGetDevice(deviceLogicId), HcclResult::HCCL_SUCCESS);

    // then
    EXPECT_EQ(fakeDeviceLogicId, deviceLogicId);
}

TEST_F(AdapterRtsTest, HrtGetDevice_return_nok)
{
    // Given
    MOCKER(aclrtGetDevice).stubs().will(returnValue(1));

    // when

    // then
    s32 userDevId;
    EXPECT_EQ(HrtGetDevice(userDevId), HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, HrtSetDevice_return_nok)
{
    // Given
    MOCKER(aclrtSetDevice).stubs().will(returnValue(1));

    // then
    EXPECT_EQ(HrtSetDevice(123), HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, HrtResetDevice_return_nok)
{
    // Given
    MOCKER(aclrtResetDevice).stubs().with(mockcpp::any()).will(returnValue(1));

    // then
    EXPECT_EQ(HrtResetDevice(1), HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, HrtGetDeviceCount_return_ok)
{
    // Given
    u32 fakeCount = 2;
    MOCKER(aclrtGetDeviceCount).stubs().with(outBoundP(&fakeCount, sizeof(fakeCount))).will(returnValue(RT_ERROR_NONE));

    // when
    u32 count = 0;
    EXPECT_EQ(HrtGetDeviceCount(count), HcclResult::HCCL_SUCCESS);

    // then
    EXPECT_EQ(fakeCount, count);
}

TEST_F(AdapterRtsTest, HrtGetDeviceCount_return_nok)
{
    u32 fakeCount = 2;
    // Given
    MOCKER(aclrtGetDeviceCount).stubs().with(mockcpp::any()).will(returnValue(1));

    // then
    u32 count;
    EXPECT_EQ(HrtGetDeviceCount(count), HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, HrtIpcSetNotifyName_return_ok)
{
    // Given
    char_t fakeName[128] = "fakeName";
    MOCKER(aclrtNotifyGetExportKey)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(StubAclrtNotifyGetExportKey));

    // when
    char_t name[128] = {0};
    EXPECT_EQ(HrtIpcSetNotifyName(nullptr, name, 128), HcclResult::HCCL_SUCCESS);

    // then
    EXPECT_EQ(0, strcmp(name, fakeName));
}

TEST_F(AdapterRtsTest, HrtIpcSetNotifyName_return_nok)
{
    // Given
    MOCKER(aclrtNotifyGetExportKey).stubs().will(returnValue(1));

    // when
    char_t name[128] = "xxx";
    EXPECT_EQ(HrtIpcSetNotifyName(nullptr, name, 128), HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, HrtGetNotifyID_return_ok)
{
    // Given
    u32 fakeNotifyID = 123;
    MOCKER(aclrtGetNotifyId)
        .stubs()
        .with(mockcpp::any(), outBoundP(&fakeNotifyID, sizeof(fakeNotifyID)))
        .will(returnValue(ACL_SUCCESS));

    // when
    u32 notifyID = 0;
    EXPECT_EQ(HrtGetNotifyID(nullptr, notifyID), HcclResult::HCCL_SUCCESS);

    // then
    EXPECT_EQ(notifyID, fakeNotifyID);
}

TEST_F(AdapterRtsTest, HrtGetNotifyID_return_nok)
{
    // Given
    MOCKER(aclrtGetNotifyId).stubs().will(returnValue(1));

    // when

    // then
    u32 notifyID;
    EXPECT_EQ(HrtGetNotifyID(nullptr, notifyID), HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, HrtGetStreamId_return_ok)
{
    // Given
    s32 fakeStreamId = 123;
    MOCKER(aclrtStreamGetId)
        .stubs()
        .with(mockcpp::any(), outBoundP(&fakeStreamId, sizeof(fakeStreamId)))
        .will(returnValue(ACL_SUCCESS));

    // when
    s32 streamId = 0;
    EXPECT_EQ(HrtGetStreamId(nullptr, streamId), HcclResult::HCCL_SUCCESS);
    // then
    EXPECT_EQ(streamId, fakeStreamId);
}

TEST_F(AdapterRtsTest, HrtGetStreamId_return_nok)
{
    // Given
    MOCKER(aclrtStreamGetId).stubs().will(returnValue(1));

    // when

    // then
    s32 streamId;
    EXPECT_EQ(HrtGetStreamId(nullptr, streamId), HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, HrtStreamGetMode_return_nok)
{
    // Given
    MOCKER(aclrtStreamGetId).stubs().will(returnValue(1));

    // then
    u64 stmMode;
    EXPECT_EQ(HrtStreamGetMode(nullptr, stmMode), HcclResult::HCCL_E_PTR);
}

TEST_F(AdapterRtsTest, HrtStreamSetMode_streamPtrIsNull_return_nok)
{
    // Given
    // then
    EXPECT_EQ(HrtStreamSetMode(nullptr, 3), HcclResult::HCCL_E_PTR);
}

TEST_F(AdapterRtsTest, HrtStreamSetMode_return_nok)
{
    // Given
    MOCKER(aclrtStreamGetId).stubs().will(returnValue(0));
    MOCKER(aclrtSetStreamAttribute).stubs().will(returnValue(1));

    // then
    EXPECT_EQ(HrtStreamSetMode((void*)1, 3), HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, HrtStreamDestroy_return_nok)
{
    // Given
    rtStream_t ptr = (rtStream_t*)0x1000000;
    MOCKER(aclrtDestroyStreamForce).stubs().will(returnValue(1));

    // then
    EXPECT_EQ(HrtStreamDestroy(ptr), HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, HcclStreamSynchronize_return_nok)
{
    // Given
    rtStream_t ptr = (rtStream_t*)0x1000000;
    MOCKER(aclrtSynchronizeStreamWithTimeout).stubs().will(returnValue(1));

    // then
    EXPECT_EQ(HcclStreamSynchronize(ptr), HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, HcclStreamSynchronize_streamPtrIsNull_return_nok)
{
    // Given
    // then
    EXPECT_EQ(HcclStreamSynchronize(nullptr), HcclResult::HCCL_E_PTR);
}

TEST_F(AdapterRtsTest, HrtMalloc_return_ok)
{
    // Given
    void* fakeDevPtr = nullptr;
    MOCKER(aclrtMallocWithCfg).stubs().with(outBoundP(&fakeDevPtr, sizeof(fakeDevPtr))).will(returnValue(ACL_SUCCESS));

    // when
    u64 size = 100;
    aclrtMemType_t memType = 2;
    void* devPtr = nullptr;
    EXPECT_EQ(HrtMalloc(size, memType, devPtr), HcclResult::HCCL_SUCCESS);
    // then
    EXPECT_EQ(fakeDevPtr, devPtr);
}

TEST_F(AdapterRtsTest, HrtMalloc_return_nok)
{
    // Given
    MOCKER(aclrtMallocWithCfg).stubs().will(returnValue(1));
    u64 size = 100;
    aclrtMemType_t memType = 2;
    // then
    void* devPtr = nullptr;
    EXPECT_EQ(HrtMalloc(size, memType, devPtr), HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, HrtFree_return_nok)
{
    // Given
    MOCKER(aclrtFree).stubs().will(returnValue(1));
    // then
    EXPECT_EQ(HrtFree(nullptr), HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, HrtMemcpy_d2d_return_nok)
{
    // Given
    MOCKER(aclmdlRICaptureThreadExchangeMode).stubs().will(returnValue(0));
    MOCKER(aclrtMemcpy).stubs().will(returnValue(1));
    // then
    EXPECT_EQ(
        HrtMemcpy(nullptr, 64, nullptr, 64, Hccl::rtMemcpyKind_t::RT_MEMCPY_DEVICE_TO_DEVICE),
        HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, HrtMemcpy_h2h_return_nok)
{
    // Given
    MOCKER(aclmdlRICaptureThreadExchangeMode).stubs().will(returnValue(0));
    MOCKER(aclrtMemcpy).stubs().will(returnValue(1));
    // then
    EXPECT_EQ(
        HrtMemcpy(nullptr, 64, nullptr, 64, Hccl::rtMemcpyKind_t::RT_MEMCPY_HOST_TO_HOST), HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, HrtMemcpy_reserved_return_nok)
{
    // Given
    MOCKER(aclmdlRICaptureThreadExchangeMode).stubs().will(returnValue(0));
    MOCKER(aclrtMemcpy).stubs().will(returnValue(1));
    // then
    EXPECT_EQ(
        HrtMemcpy(nullptr, 64, nullptr, 64, Hccl::rtMemcpyKind_t::RT_MEMCPY_RESERVED), HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, HrtIpcSetMemoryName_return_nok)
{
    // Given
    MOCKER(aclrtIpcMemGetExportKey).stubs().will(returnValue(1));

    // then
    char_t fakeName[16] = "fakeName";
    EXPECT_EQ(Hccl::HrtIpcSetMemoryName(nullptr, &fakeName[0], 32, 32), HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, HrtIpcDestroyMemoryName_return_nok)
{
    // Given
    MOCKER(aclrtIpcMemClose).stubs().will(returnValue(1));

    // then
    EXPECT_EQ(HrtIpcDestroyMemoryName(nullptr), HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, HrtIpcOpenMemory_return_ok)
{
    // Given
    void* ptr = nullptr;
    char_t fakeName[128] = "fakeName";
    MOCKER(aclrtIpcMemImportByKey).stubs().with(outBoundP(&ptr, sizeof(ptr))).will(returnValue(ACL_SUCCESS));
    // when
    void* result = nullptr;
    EXPECT_EQ(HrtIpcOpenMemory(fakeName, result), HcclResult::HCCL_SUCCESS);
    // then
    EXPECT_EQ(ptr, result);
}

TEST_F(AdapterRtsTest, HrtIpcOpenMemory_return_nok)
{
    // Given
    MOCKER(aclrtIpcMemImportByKey).stubs().will(returnValue(1));
    // then
    char_t fakeName[128] = "fakeName";
    void* ptr = nullptr;
    EXPECT_EQ(HrtIpcOpenMemory(fakeName, ptr), HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, HrtIpcCloseMemory_return_nok)
{
    // Given
    MOCKER(aclrtIpcMemClose).stubs().will(returnValue(1));

    // then
    EXPECT_EQ(HrtIpcCloseMemory(nullptr), HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, HrtIpcSetMemoryPid_return_nok)
{
    // Given
    MOCKER(aclrtIpcMemSetImportPid).stubs().will(returnValue(1));
    // then
    EXPECT_EQ(HrtIpcSetMemoryPid(nullptr, 100), HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, HrtMallocHost_return_ok)
{
    // Given
    void* fakehostPtr = nullptr;
    MOCKER(aclrtMallocHostWithCfg)
        .stubs()
        .with(outBoundP(&fakehostPtr, sizeof(fakehostPtr)))
        .will(returnValue(ACL_SUCCESS));

    // when
    void* result = nullptr;
    EXPECT_EQ(HrtMallocHost(64, result), HcclResult::HCCL_SUCCESS);
    // then
    EXPECT_EQ(fakehostPtr, result);
}

TEST_F(AdapterRtsTest, HrtMallocHost_return_nok)
{
    // Given
    MOCKER(aclrtMallocHostWithCfg).stubs().will(returnValue(1));

    // then
    void* result = nullptr;
    EXPECT_EQ(HrtMallocHost(64, result), HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, HrtFreeHost_return_nok)
{
    // Given
    MOCKER(aclrtFreeHost).stubs().will(returnValue(1));

    // then
    EXPECT_EQ(HrtFreeHost(nullptr), HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, HrtNotifyCreate_return_ok)
{
    // Given
    RtNotify_t fakeRtsNotify = (RtNotify_t*)0x1000000;
    MOCKER(aclrtCreateNotify)
        .stubs()
        .with(outBoundP(&fakeRtsNotify, sizeof(fakeRtsNotify)), mockcpp::any())
        .will(returnValue(RT_ERROR_NONE));

    // when
    aclrtNotify rtNotify = nullptr;
    EXPECT_EQ(HrtNotifyCreate(100, rtNotify), HcclResult::HCCL_SUCCESS);

    // then
    EXPECT_EQ(fakeRtsNotify, rtNotify);
}

TEST_F(AdapterRtsTest, HrtNotifyCreate_return_nok)
{
    // Given
    MOCKER(aclrtCreateNotify).stubs().will(returnValue(1));

    // then
    aclrtNotify ptr = nullptr;
    EXPECT_EQ(HrtNotifyCreate(100, ptr), HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, HrtNotifyDestroy_return_nok)
{
    // Given
    MOCKER(aclrtDestroyNotify).stubs().will(returnValue(1));

    // then
    EXPECT_EQ(HrtNotifyDestroy(nullptr), HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, HrtStreamActive_return_nok)
{
    // Given
    rtStream_t ptr = (rtStream_t*)0x1000000;
    MOCKER(aclrtActiveStream).stubs().will(returnValue(1));

    // then
    EXPECT_EQ(HrtStreamActive(nullptr, nullptr), HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, HrtPointerGetAttributes_return_ok)
{
    aclrtPtrAttributes ptrAttr
        = {.location = {.id = 0, .type = aclrtMemLocationType::ACL_MEM_LOCATION_TYPE_DEVICE}, .pageSize = 32};
    // Given
    MOCKER(aclrtPointerGetAttributes)
        .stubs()
        .with(mockcpp::any(), outBoundP(&ptrAttr, sizeof(ptrAttr)))
        .will(returnValue(ACL_SUCCESS));

    // when
    aclrtPtrAttributes result{};
    EXPECT_EQ(HrtPointerGetAttributes(nullptr, result), HcclResult::HCCL_SUCCESS);

    // then
    EXPECT_EQ(ptrAttr.location.id, result.location.id);
    EXPECT_EQ(ptrAttr.location.type, result.location.type);
    EXPECT_EQ(ptrAttr.pageSize, result.pageSize);
}

TEST_F(AdapterRtsTest, HrtPointerGetAttributes_return_nok)
{
    // Given
    MOCKER(aclrtPointerGetAttributes).stubs().will(returnValue(1));

    // then
    aclrtPtrAttributes result{};
    EXPECT_EQ(HrtPointerGetAttributes(nullptr, result), HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, PrintMemoryAttr_run)
{
    aclrtPtrAttributes ptrAttr
        = {.location = {.id = 0, .type = aclrtMemLocationType::ACL_MEM_LOCATION_TYPE_DEVICE}, .pageSize = 1};
    // Given
    MOCKER(HrtPointerGetAttributes)
        .stubs()
        .with(mockcpp::any(), outBound(aclrtPtrAttributes{}))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    PrintMemoryAttr(nullptr);
}

TEST_F(AdapterRtsTest, HrtDevMemAlignWithPage_pagesize_zero)
{
    // Given
    aclrtPtrAttributes ptrAttr
        = {.location = {.id = 0, .type = aclrtMemLocationType::ACL_MEM_LOCATION_TYPE_DEVICE}, .pageSize = 0};
    MOCKER(HrtPointerGetAttributes)
        .stubs()
        .with(mockcpp::any(), outBound(ptrAttr))
        .will(returnValue(HcclResult::HCCL_SUCCESS));

    // when
    void* ptr = nullptr;
    u64 size = 64;
    void* ipcPtr = nullptr;
    u64 ipcSize = 64;
    u64 ipcOff;
    HrtDevMemAlignWithPage(ptr, size, ipcPtr, ipcSize, ipcOff);
    // then
    EXPECT_EQ(0, ipcOff);
}

TEST_F(AdapterRtsTest, HrtDevMemAlignWithPage_pargesize_nzero)
{
    // Given
    aclrtPtrAttributes ptrAttr
        = {.location = {.id = 0, .type = aclrtMemLocationType::ACL_MEM_LOCATION_TYPE_DEVICE}, .pageSize = 32};
    MOCKER(HrtPointerGetAttributes)
        .stubs()
        .with(mockcpp::any(), outBound(ptrAttr))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    // when
    // when
    void* ptr = (void*)4;
    u64 size = 64;
    void* ipcPtr = (void*)8;
    u64 ipcSize = 64;
    u64 ipcOff;
    HrtDevMemAlignWithPage(ptr, size, ipcPtr, ipcSize, ipcOff);
    // then
    EXPECT_NE(0, ipcOff);
}

TEST_F(AdapterRtsTest, HrtDevMemAlignWithPage_return_nok)
{
    // Given
    MOCKER(aclrtPointerGetAttributes).stubs().will(returnValue(1));

    // then
    void* ptr = nullptr;
    u64 size = 64;
    void* ipcPtr = nullptr;
    u64 ipcSize = 64;
    u64 ipcOff = 4;
    EXPECT_EQ(HrtDevMemAlignWithPage(ptr, size, ipcPtr, ipcSize, ipcOff), HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, HrtNotifyGetAddr_return_ok)
{
    // Given
    uint64_t fakeAddr = 1;
    MOCKER(rtGetNotifyAddress)
        .stubs()
        .with(mockcpp::any(), outBoundP(&fakeAddr, sizeof(fakeAddr)))
        .will(returnValue(RT_ERROR_NONE));

    // when
    RtNotify_t ptr = (RtNotify_t*)0x1000000;
    u64 addr = 0;
    EXPECT_EQ(HrtNotifyGetAddr(ptr, addr), HcclResult::HCCL_SUCCESS);
    // then
    EXPECT_EQ(fakeAddr, addr);
}

TEST_F(AdapterRtsTest, HrtNotifyGetAddr_return_nok)
{
    // Given
    MOCKER(rtGetNotifyAddress).stubs().will(returnValue(1));

    // then
    u64 addr;
    EXPECT_EQ(HrtNotifyGetAddr(nullptr, addr), HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, HrtSetIpcNotifyPid_return_nok)
{
    // Given
    MOCKER(aclrtNotifySetImportPid).stubs().will(returnValue(1));

    // then
    EXPECT_EQ(HrtSetIpcNotifyPid(nullptr, 100), HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, HrtIpcOpenNotify_return_ok)
{
    // Given
    RtNotify_t fakePtr = nullptr;
    MOCKER(aclrtNotifyImportByKey)
        .stubs()
        .with(outBoundP(&fakePtr, sizeof(fakePtr)), mockcpp::any(), mockcpp::any())
        .will(returnValue(RT_ERROR_NONE));

    // when
    RtNotify_t ptr = nullptr;
    EXPECT_EQ(HrtIpcOpenNotify(nullptr, ptr), HcclResult::HCCL_SUCCESS);
    // then
    EXPECT_EQ(fakePtr, ptr);
}

TEST_F(AdapterRtsTest, HrtIpcOpenNotify_return_nok)
{
    // Given
    MOCKER(aclrtNotifyImportByKey).stubs().will(returnValue(1));

    // then
    RtNotify_t ptr;
    EXPECT_EQ(HrtIpcOpenNotify(nullptr, ptr), HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, HrtNotifyGetOffset_return_ok)
{
    // Given
    uint64_t fakeOffset = 0;
    // when
    u32 offset = 0;
    EXPECT_EQ(HrtNotifyGetOffset(nullptr, offset), HcclResult::HCCL_SUCCESS);
    // then
    EXPECT_EQ(static_cast<u32>(fakeOffset), offset);
}

TEST_F(AdapterRtsTest, HrtNotifyGetOffset_return_nok)
{
    // Given
    MOCKER(aclrtGetNotifyId).stubs().will(returnValue(1));

    // then
    u32 offset;
    EXPECT_EQ(HrtNotifyGetOffset(nullptr, offset), HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, HrtNotifyWaitWithTimeOut_return_nok)
{
    // Given
    MOCKER(aclrtWaitAndResetNotify).stubs().will(returnValue(1));

    // then
    EXPECT_EQ(HrtNotifyWaitWithTimeOut(nullptr, nullptr, 1), HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, HrtNotifyWaitWithTimeOut_return_ok)
{
    // Given
    MOCKER(aclrtWaitAndResetNotify).stubs().will(returnValue(ACL_SUCCESS));

    // then
    EXPECT_EQ(HrtNotifyWaitWithTimeOut(nullptr, nullptr, 100), HcclResult::HCCL_SUCCESS);
}

TEST_F(AdapterRtsTest, HrtNotifyRecord_return_nok)
{
    // Given
    MOCKER(aclrtRecordNotify).stubs().will(returnValue(1));

    // then
    EXPECT_EQ(HrtNotifyRecord(nullptr, nullptr), HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, HrtMemAsyncCopy_return_nok)
{
    // Given
    MOCKER(aclrtMemcpyAsync).stubs().will(returnValue(1));
    // then
    EXPECT_EQ(
        HrtMemAsyncCopy(nullptr, 64, nullptr, 64, aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_DEVICE, nullptr),
        HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, HrtReduceAsync_return_nok)
{
    // Given
    MOCKER(aclrtReduceAsync).stubs().will(returnValue(1));
    // then
    EXPECT_EQ(
        HrtReduceAsync(
            nullptr, 64, nullptr, 64, aclrtReduceKind::ACL_RT_MEMCPY_SDMA_AUTOMATIC_MIN, aclDataType::ACL_INT16,
            nullptr),
        HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, HrtRDMASend_return_nok)
{
    // Given
    MOCKER(rtRDMASend).stubs().will(returnValue(1));
    // then
    EXPECT_EQ(HrtRDMASend(32, 32, nullptr), HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, HrtRDMADBSend_return_nok)
{
    // Given
    MOCKER(rtRDMADBSend).stubs().will(returnValue(1));
    // then
    EXPECT_EQ(HrtRDMADBSend(32, 32, nullptr), HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, HrtGetTaskIdAndStreamID_return_nok)
{
    // Given
    MOCKER(rtGetTaskIdAndStreamID).stubs().will(returnValue(1));
    // when
    u32 taskId;
    u32 streamId;
    // then
    EXPECT_EQ(HrtGetTaskIdAndStreamID(taskId, streamId), HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, HrtCntNotifyCreate_return_nok)
{
    // Given
    MOCKER(aclrtCntNotifyCreate).stubs().will(returnValue(1));

    // when
    u32 deviceID = 1;

    // then
    aclrtCntNotify handle;
    EXPECT_NE(HrtCntNotifyCreate(deviceID, handle), HcclResult::HCCL_SUCCESS);
}

TEST_F(AdapterRtsTest, HrtGetCntNotifyId_return_nok)
{
    // Given
    MOCKER(aclrtCntNotifyGetId).stubs().will(returnValue(1));

    // when
    u32 deviceID = 1;

    // then
    u32 notifyId;
    EXPECT_NE(HrtGetCntNotifyId(nullptr, notifyId), HcclResult::HCCL_SUCCESS);
}

TEST_F(AdapterRtsTest, HrtUbDbSend_run_exception_NotSupportException)
{
    // Given
    rtNotify_t ptr = (rtNotify_t*)0x1000000;

    HrtUbDbInfo info;
    info.dbNum = 2;
    info.wrCqe = 2;

    HrtUbDbDetailInfo hrtUbDbDetailInfo1;
    hrtUbDbDetailInfo1.functionId = 0;
    hrtUbDbDetailInfo1.dieId = 0;
    hrtUbDbDetailInfo1.rsv = 0;
    hrtUbDbDetailInfo1.jettyId = 0;
    hrtUbDbDetailInfo1.piValue = 0;

    info.info[0] = hrtUbDbDetailInfo1;

    // then
    EXPECT_EQ(HrtUbDbSend(info, ptr), HcclResult::HCCL_E_NOT_SUPPORT);
}

TEST_F(AdapterRtsTest, HrtUbDirectSend_run_exception_NotSupportException)
{
    // Given
    rtNotify_t ptr = (rtNotify_t*)0x1000000;

    HrtUbWqeInfo info;
    info.wrCqe = 0;
    info.functionId = 0;
    info.dieId = 0;
    info.jettyId = 18;
    info.wqePtrLen = 128;
    info.wqeSize = 1;

    // then
    EXPECT_EQ(HrtUbDirectSend(info, ptr), HcclResult::HCCL_E_NOT_SUPPORT);
}

TEST_F(AdapterRtsTest, test_HrtNotifyCreateWithFlag_return_ok)
{
    // Given
    rtNotify_t fakeRtsNotify = (rtNotify_t*)0x1000000;
    MOCKER(aclrtCreateNotify)
        .stubs()
        .with(outBoundP(&fakeRtsNotify, sizeof(fakeRtsNotify)), mockcpp::any())
        .will(returnValue(RT_ERROR_NONE));

    // when
    aclrtNotify rtNotify = nullptr;
    EXPECT_EQ(HrtNotifyCreateWithFlag(100, 100, rtNotify), HcclResult::HCCL_SUCCESS);

    // then
    EXPECT_EQ(fakeRtsNotify, rtNotify);
}

TEST_F(AdapterRtsTest, HrtNotifyCreateWithFlag_return_nok)
{
    // Given
    MOCKER(aclrtCreateNotify).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(1));

    // then
    aclrtNotify ptr = nullptr;
    EXPECT_EQ(HrtNotifyCreateWithFlag(100, 100, ptr), HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, test_HrtAicpuLaunchKernelWithHostArgs_ok)
{
    // Given
    MOCKER(aclrtLaunchKernelWithHostArgs).stubs().will(returnValue(0));

    const char_t* name = "aaa";
    rtAicpuArgsEx_t argsInfo;
    argsInfo.args = (void*)name;
    argsInfo.soNameAddrOffset = 0;
    argsInfo.kernelNameAddrOffset = 0;
    aclrtFuncHandle funcHandle;
    constexpr u32 numBlocks = 1;
    aclrtStream stream;
    aclrtLaunchKernelCfg cfg;
    HcclKernelParamLite kernel;
    aclrtPlaceHolderInfo placeHolderArray;
    constexpr std::size_t placeHolderNum = 0;
    EXPECT_EQ(
        HrtAicpuLaunchKernelWithHostArgs(funcHandle, numBlocks, stream, &cfg, &kernel, sizeof(HcclKernelParamLite)),
        HcclResult::HCCL_SUCCESS);
}

TEST_F(AdapterRtsTest, test_HrtAicpuLaunchKernelWithHostArgs_nok)
{
    // Given
    MOCKER(aclrtLaunchKernelWithHostArgs).stubs().will(returnValue(1));

    aclrtFuncHandle funcHandle;
    constexpr u32 numBlocks = 1;
    aclrtStream stream;
    aclrtLaunchKernelCfg cfg;
    HcclKernelParamLite kernel;
    aclrtPlaceHolderInfo placeHolderArray;
    constexpr std::size_t placeHolderNum = 0;
    EXPECT_EQ(
        HrtAicpuLaunchKernelWithHostArgs(funcHandle, numBlocks, stream, &cfg, &kernel, sizeof(HcclKernelParamLite)),
        HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, test_HrtStreamGetSqId_ok)
{
    u32 fakeSqId = 100;
    // Given
    MOCKER(rtStreamGetSqid).stubs().with(mockcpp::any(), outBoundP(&fakeSqId, sizeof(fakeSqId))).will(returnValue(0));
    u32 sqId = 0;
    EXPECT_EQ(HrtStreamGetSqId((void*)100, sqId), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(fakeSqId, sqId);
}

TEST_F(AdapterRtsTest, test_HrtStreamGetSqId_nok)
{
    // Given
    MOCKER(rtStreamGetSqid).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(1));
    u32 sqId;
    EXPECT_EQ(HrtStreamGetSqId((void*)100, sqId), HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, test_HrtIpcOpenNotifyWithFlag_nok)
{
    MOCKER(aclrtNotifyImportByKey).stubs().will(returnValue(1));
    RtNotify_t ptr;
    EXPECT_EQ(HrtIpcOpenNotifyWithFlag("test", 100, ptr), HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, test_HrtIpcOpenNotifyWithFlag_ok)
{
    // Given
    RtNotify_t fakePtr = nullptr;
    MOCKER(aclrtNotifyImportByKey)
        .stubs()
        .with(outBoundP(&fakePtr, sizeof(fakePtr)), mockcpp::any(), mockcpp::any())
        .will(returnValue(RT_ERROR_NONE));
    RtNotify_t ptr = nullptr;
    EXPECT_EQ(HrtIpcOpenNotifyWithFlag("test", 100, ptr), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(ptr, fakePtr);
}

TEST_F(AdapterRtsTest, HrtCcuLaunch_run_fail)
{
    // Given
    rtCcuTaskInfo_t taskInfo;
    aclrtStream ptr = (aclrtStream*)0x1000000;

    MOCKER(rtCCULaunch).stubs().will(returnValue(1));

    // then
    EXPECT_EQ(HrtCcuLaunch(taskInfo, ptr), HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, HrtCcuLaunch_run_fail_rts_task_full)
{
    // Given
    rtCcuTaskInfo_t taskInfo;
    aclrtStream ptr = (aclrtStream*)0x1000000;

    // Runtime 接口报错：Task 队列满
    MOCKER(rtCCULaunch).stubs().will(returnValue(ACL_ERROR_RT_STREAM_TASK_FULL));

    // then
    EXPECT_EQ(HrtCcuLaunch(taskInfo, ptr), HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, HrtUbDevQueryInfo_run_fail)
{
    // Given
    rtUbDevQueryCmd cmd = QUERY_PROCESS_TOKEN;
    rtMemUbTokenInfo devInfo;
    devInfo.va = 0x1000;
    ;
    devInfo.size = 10;

    MOCKER(rtUbDevQueryInfo).stubs().will(returnValue(1));

    // then
    EXPECT_EQ(HrtUbDevQueryInfo(cmd, &devInfo), HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, HrtUbDevQueryInfo_run_ok)
{
    // Given
    rtUbDevQueryCmd cmd = QUERY_PROCESS_TOKEN;
    rtMemUbTokenInfo devInfo;
    devInfo.va = 0x1000;
    ;
    devInfo.size = 10;

    MOCKER(rtUbDevQueryInfo).stubs().will(returnValue(0));

    // then
    EXPECT_EQ(HrtUbDevQueryInfo(cmd, &devInfo), HcclResult::HCCL_SUCCESS);
}

TEST_F(AdapterRtsTest, HrtGetDevResAddress_run_fail)
{
    // Given
    HrtDevResInfo resInfo;

    MOCKER(rtGetDevResAddress).stubs().will(returnValue(1));

    // then
    HrtDevResAddrInfo addrInfo;
    EXPECT_EQ(HrtGetDevResAddress(resInfo, addrInfo), HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, HrtReleaseDevResAddress_run_fail)
{
    HrtDevResInfo resInfo;

    MOCKER(rtReleaseDevResAddress).stubs().will(returnValue(1));

    EXPECT_EQ(HrtReleaseDevResAddress(resInfo), HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, HrtEventCreateWithFlag_return_ok)
{
    // Given
    RtEvent_t fakePtr = nullptr;
    MOCKER(aclrtCreateEventWithFlag)
        .stubs()
        .with(outBoundP(&fakePtr, sizeof(fakePtr)), mockcpp::any())
        .will(returnValue(RT_ERROR_NONE));

    // when
    aclrtEvent ptr = nullptr;
    EXPECT_EQ(HrtEventCreateWithFlag(2, ptr), HcclResult::HCCL_SUCCESS);
    // then
    EXPECT_EQ(fakePtr, ptr);
}

TEST_F(AdapterRtsTest, HrtEventCreateWithFlag_return_nok)
{
    // Given
    MOCKER(aclrtCreateEventWithFlag).stubs().will(returnValue(1));

    // then
    aclrtEvent ptr = nullptr;
    EXPECT_EQ(HrtEventCreateWithFlag(2, ptr), HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, HrtEventDestroy_return_ok)
{
    // Given
    RtEvent_t fakePtr = nullptr;
    MOCKER(aclrtDestroyEvent).stubs().will(returnValue(RT_ERROR_NONE));

    // then
    EXPECT_EQ(HrtEventDestroy(fakePtr), HcclResult::HCCL_SUCCESS);
}

TEST_F(AdapterRtsTest, HrtEventDestroy_return_nok)
{
    // Given
    RtEvent_t fakePtr = nullptr;
    MOCKER(aclrtDestroyEvent).stubs().will(returnValue(1));

    // then
    EXPECT_EQ(HrtEventDestroy(fakePtr), HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, HrtEventRecord_return_ok)
{
    // Given
    RtEvent_t fakePtr = nullptr;
    rtStream_t stream = nullptr;
    MOCKER(aclrtRecordEvent).stubs().will(returnValue(ACL_SUCCESS));

    // then
    EXPECT_EQ(HrtEventRecord(fakePtr, stream), HcclResult::HCCL_SUCCESS);
}

TEST_F(AdapterRtsTest, HrtEventRecord_return_nok)
{
    // Given
    RtEvent_t fakePtr = nullptr;
    rtStream_t stream = nullptr;
    MOCKER(aclrtRecordEvent).stubs().will(returnValue(1));

    // then
    EXPECT_EQ(HrtEventRecord(fakePtr, stream), HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, HrtEventQueryStatus_return_init_ok)
{
    // Given
    RtEvent_t fakePtr = nullptr;
    aclrtEventWaitStatus status = ACL_EVENT_WAIT_STATUS_NOT_READY;
    MOCKER(aclrtQueryEventWaitStatus)
        .stubs()
        .with(mockcpp::any(), outBoundP(&status, sizeof(status)))
        .will(returnValue(ACL_SUCCESS));

    // then
    HrtEventStatus eventStatus;
    EXPECT_EQ(HrtEventQueryStatus(fakePtr, eventStatus), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(eventStatus, HrtEventStatus::EVENT_INIT);
}

TEST_F(AdapterRtsTest, HrtEventQueryStatus_return_recorded_ok)
{
    // Given
    RtEvent_t fakePtr = nullptr;
    aclrtEventWaitStatus status = ACL_EVENT_WAIT_STATUS_COMPLETE;
    MOCKER(aclrtQueryEventWaitStatus)
        .stubs()
        .with(mockcpp::any(), outBoundP(&status, sizeof(status)))
        .will(returnValue(ACL_SUCCESS));

    // then
    HrtEventStatus eventStatus;
    EXPECT_EQ(HrtEventQueryStatus(fakePtr, eventStatus), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(eventStatus, HrtEventStatus::EVENT_RECORDED);
}

TEST_F(AdapterRtsTest, HrtEventQueryStatus_return_nok)
{
    // Given
    RtEvent_t fakePtr = nullptr;
    MOCKER(aclrtQueryEventWaitStatus).stubs().will(returnValue(1));

    // then
    HrtEventStatus status;
    EXPECT_EQ(HrtEventQueryStatus(fakePtr, status), HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, HrtWriteValue_run_fail)
{
    EXPECT_EQ(HrtWriteValue(0, 0, nullptr), HcclResult::HCCL_E_NOT_SUPPORT);
}

TEST_F(AdapterRtsTest, HrtUbDevQueryToken_run_OK)
{
    rtMemUbTokenInfo info;
    info.tokenId = 0xaaaaa;
    info.tokenValue = 20;
    // Given
    MOCKER(rtUbDevQueryInfo)
        .stubs()
        .with(mockcpp::any(), outBoundP(static_cast<void*>(&info), sizeof(rtMemUbTokenInfo)))
        .will(returnValue(RT_ERROR_NONE));
    // then
    std::pair<u32, u32> res = HrtUbDevQueryToken(0xffff, 20);
    EXPECT_EQ(res.first, info.tokenId >> 8);
    EXPECT_EQ(res.second, info.tokenValue);
}

TEST_F(AdapterRtsTest, HrtUbDevQueryToken_run_NOK)
{
    rtMemUbTokenInfo info;
    info.tokenId = 0xaaaaa;
    info.tokenValue = 20;
    // Given
    MOCKER(rtUbDevQueryInfo)
        .stubs()
        .with(mockcpp::any(), outBoundP(static_cast<void*>(&info), sizeof(info)))
        .will(returnValue(1));
    // then
    std::pair<u32, u32> res = HrtUbDevQueryToken(0xffff, 20);
    EXPECT_EQ(res.first, 0);
    EXPECT_EQ(res.second, 0);
}

TEST_F(AdapterRtsTest, HrtEnableP2P_return_ok)
{
    // Given
    MOCKER(rtEnableP2P).stubs().will(returnValue(0));

    // then
    u32 deviceLogicId = 0;
    u32 devicePhyId = 1;

    EXPECT_EQ(HrtEnableP2P(deviceLogicId, devicePhyId), HCCL_SUCCESS);
}

TEST_F(AdapterRtsTest, HrtDisableP2P_return_ok)
{
    // Given
    MOCKER(rtDisableP2P).stubs().will(returnValue(0));

    // then
    u32 deviceLogicId = 0;
    u32 devicePhyId = 1;

    EXPECT_EQ(HrtDisableP2P(deviceLogicId, devicePhyId), HCCL_SUCCESS);
}

TEST_F(AdapterRtsTest, HrtGetP2PStatus_return_ok)
{
    // Given
    MOCKER(rtGetP2PStatus).stubs().will(returnValue(0));

    // then
    u32 deviceLogicId = 0;
    u32 devicePhyId = 1;
    uint32_t status;

    EXPECT_EQ(rtGetP2PStatus(deviceLogicId, devicePhyId, &status), HCCL_SUCCESS);
}

TEST_F(AdapterRtsTest, HrtCntNotifyWaitWithTimeOut_return_ok)
{
    MOCKER(aclrtCntNotifyWaitWithTimeout).stubs().will(returnValue(ACL_SUCCESS));
    EXPECT_EQ(
        HrtCntNotifyWaitWithTimeOut(nullptr, nullptr, HrtCntNotifyWaitMode::EQUAL, 1, 100, true),
        HcclResult::HCCL_SUCCESS);
}

TEST_F(AdapterRtsTest, HrtCntNotifyWaitWithTimeOut_return_nok)
{
    MOCKER(aclrtCntNotifyWaitWithTimeout).stubs().will(returnValue(1));
    EXPECT_EQ(
        HrtCntNotifyWaitWithTimeOut(nullptr, nullptr, HrtCntNotifyWaitMode::EQUAL, 1, 100, true),
        HcclResult::HCCL_E_RUNTIME);
}

TEST_F(AdapterRtsTest, HrtGetDeviceType_950_return_ok)
{
    // Given
    ::DevType globalDevType = ::DevType::DEV_TYPE_950;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(globalDevType)).will(returnValue(HCCL_SUCCESS));

    // when
    Hccl::DevType devType;
    EXPECT_EQ(HrtGetDeviceType(devType), HcclResult::HCCL_SUCCESS);

    // then
    EXPECT_EQ(devType, Hccl::DevType::DEV_TYPE_950);
}

TEST_F(AdapterRtsTest, HrtGetDeviceType_unsupported_return_nok)
{
    // Given
    ::DevType globalDevType = ::DevType::DEV_TYPE_MC62;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(globalDevType)).will(returnValue(HCCL_SUCCESS));

    // when

    // then
    Hccl::DevType devType;
    EXPECT_EQ(HrtGetDeviceType(devType), HcclResult::HCCL_E_RUNTIME);
}

// 全局枚举(::DevType)与域内枚举(Hccl::DevType)的命名/序号不一致，逐分支校验映射结果
class HrtGetDeviceTypeMapTest :
    public AdapterRtsTest,
    public testing::WithParamInterface<std::pair<::DevType, Hccl::DevType>> {};

TEST_P(HrtGetDeviceTypeMapTest, HrtGetDeviceType_map_global_dev_type)
{
    // Given
    ::DevType globalDevType = GetParam().first;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(globalDevType)).will(returnValue(HCCL_SUCCESS));

    // when

    // then
    Hccl::DevType devType;
    EXPECT_EQ(HrtGetDeviceType(devType), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(devType, GetParam().second);
}

INSTANTIATE_TEST_SUITE_P(
    MapToHcclDevType, HrtGetDeviceTypeMapTest,
    testing::Values(
        std::make_pair(::DevType::DEV_TYPE_910, Hccl::DevType::DEV_TYPE_910A),
        std::make_pair(::DevType::DEV_TYPE_310P3, Hccl::DevType::DEV_TYPE_V51_310_P3),
        std::make_pair(::DevType::DEV_TYPE_910B, Hccl::DevType::DEV_TYPE_910A2),
        std::make_pair(::DevType::DEV_TYPE_310P1, Hccl::DevType::DEV_TYPE_V51_310_P1),
        std::make_pair(::DevType::DEV_TYPE_910_93, Hccl::DevType::DEV_TYPE_910A3),
        std::make_pair(::DevType::DEV_TYPE_950, Hccl::DevType::DEV_TYPE_950),
        std::make_pair(::DevType::DEV_TYPE_960, Hccl::DevType::DEV_TYPE_960),
        std::make_pair(::DevType::DEV_TYPE_NOSOC, Hccl::DevType::DEV_TYPE_NOSOC)));
