/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>
#include <vector>
#include <string>
#include <memory>
#include "hccl_comm_pub.h"
#define private public
#define protected public
#include "dpuTaskException.h"
#include "hcclCommTaskException.h"
#undef private
#undef protected
#include "hccl_types.h"

using namespace hcomm;
using namespace hccl;

constexpr size_t DPU_EXP_MEM_SIZE = 20;
constexpr size_t DPU_ERRCODE_OFFSET = sizeof(uint8_t) + sizeof(uint16_t);

class DpuTaskExceptionTest : public testing::Test {
protected:
    void SetUp() override { CollCommMgr::GetInstance().allCollComms_.clear(); }
    void TearDown() override
    {
        CollCommMgr::GetInstance().allCollComms_.clear();
        GlobalMockObject::verify();
    }
};

TEST_F(DpuTaskExceptionTest, Ut_ProcessDpuException_When_DpuErrCodeZero_Expect_ReturnNotFound)
{
    const s32 testDeviceId = 0;
    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = static_cast<uint32_t>(testDeviceId);
    TaskExceptionHost* host = TaskExceptionHost::GetInstance(testDeviceId);
    ASSERT_NE(host, nullptr);

    HcclResult ret = DpuTaskException::ProcessDpuException(&exceptionInfo, *host);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(DpuTaskExceptionTest, Ut_ProcessDpuException_When_DpuErrCodeNonZero_Expect_ReturnSuccess)
{
    const s32 testDeviceId = 7;
    const std::string testCommId = "dpuExpTestComm";
    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = static_cast<uint32_t>(testDeviceId);
    TaskExceptionHost* host = TaskExceptionHost::GetInstance(testDeviceId);
    ASSERT_NE(host, nullptr);

    CollCommMgr::GetInstance().allCollComms_[testCommId] = nullptr;

    auto memBuf = std::make_shared<std::vector<uint8_t>>(DPU_EXP_MEM_SIZE, 0);
    uint16_t errCode = 1;
    memcpy(memBuf->data() + DPU_ERRCODE_OFFSET, &errCode, sizeof(uint16_t));

    MOCKER_CPP(&CollCommMgr::GetDpuDevId)
        .stubs()
        .with(mockcpp::any(), outBound(static_cast<uint32_t>(testDeviceId)))
        .will(returnValue(HCCL_SUCCESS));
    DpuTaskExpMemRef mockRef;
    mockRef.addr = memBuf->data();
    mockRef.keepAlive = memBuf;
    MOCKER_CPP(&CollCommMgr::GetDpuTaskExpMem)
        .stubs()
        .with(mockcpp::any(), outBound(mockRef))
        .will(returnValue(HCCL_SUCCESS));

    HcclResult ret = DpuTaskException::ProcessDpuException(&exceptionInfo, *host);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(DpuTaskExceptionTest, Ut_ProcessDpuException_When_MemcpyFailed_Expect_ReturnInternal)
{
    const s32 testDeviceId = 5;
    const std::string testCommId = "dpuExpTestComm";
    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = static_cast<uint32_t>(testDeviceId);
    TaskExceptionHost* host = TaskExceptionHost::GetInstance(testDeviceId);
    ASSERT_NE(host, nullptr);

    CollCommMgr::GetInstance().allCollComms_[testCommId] = nullptr;

    auto memBuf = std::make_shared<std::vector<uint8_t>>(DPU_EXP_MEM_SIZE, 0);

    MOCKER_CPP(&CollCommMgr::GetDpuDevId)
        .stubs()
        .with(mockcpp::any(), outBound(static_cast<uint32_t>(testDeviceId)))
        .will(returnValue(HCCL_SUCCESS));
    DpuTaskExpMemRef mockRef;
    mockRef.addr = memBuf->data();
    mockRef.keepAlive = memBuf;
    MOCKER_CPP(&CollCommMgr::GetDpuTaskExpMem)
        .stubs()
        .with(mockcpp::any(), outBound(mockRef))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER(memcpy_s).stubs().will(returnValue(1));

    HcclResult ret = DpuTaskException::ProcessDpuException(&exceptionInfo, *host);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(DpuTaskExceptionTest, Ut_GetDpuExceptionInfo_When_NoDpuError_Expect_ReturnSuccess)
{
    rtExceptionInfo_t exceptionInfo{};
    uint16_t dpuErrCode = 0;
    std::string commId;
    hccl::DpuTaskExpMemRef expRef;
    HcclResult ret = DpuTaskException::GetDpuExceptionInfo(&exceptionInfo, dpuErrCode, commId, expRef);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(dpuErrCode, 0);
    EXPECT_EQ(expRef.addr, nullptr);
}

TEST_F(DpuTaskExceptionTest, Ut_ProcessDpuException_When_DpuCommExistsButErrCodeZero_Expect_ReturnNotFound)
{
    const s32 testDeviceId = 3;
    const std::string testCommId = "dpuExpTestComm";
    rtExceptionInfo_t exceptionInfo{};
    exceptionInfo.deviceid = static_cast<uint32_t>(testDeviceId);
    TaskExceptionHost* host = TaskExceptionHost::GetInstance(testDeviceId);
    ASSERT_NE(host, nullptr);

    CollCommMgr::GetInstance().allCollComms_[testCommId] = nullptr;

    auto memBuf = std::make_shared<std::vector<uint8_t>>(DPU_EXP_MEM_SIZE, 0);

    MOCKER_CPP(&CollCommMgr::GetDpuDevId)
        .stubs()
        .with(mockcpp::any(), outBound(static_cast<uint32_t>(testDeviceId)))
        .will(returnValue(HCCL_SUCCESS));
    DpuTaskExpMemRef mockRef;
    mockRef.addr = memBuf->data();
    mockRef.keepAlive = memBuf;
    MOCKER_CPP(&CollCommMgr::GetDpuTaskExpMem)
        .stubs()
        .with(mockcpp::any(), outBound(mockRef))
        .will(returnValue(HCCL_SUCCESS));

    HcclResult ret = DpuTaskException::ProcessDpuException(&exceptionInfo, *host);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}
