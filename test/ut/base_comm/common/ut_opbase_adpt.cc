/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>

#include "opbase_adpt.h"
#include "adapter_rts_common.h"
#include "device_capacity.h"

namespace {
HcclResult StubHrtGetDeviceSuccess(s32* deviceLogicId)
{
    *deviceLogicId = 3;
    return HCCL_SUCCESS;
}

HcclResult StubHrtGetDeviceFail(s32* deviceLogicId)
{
    *deviceLogicId = 66;
    return HCCL_E_INTERNAL;
}

HcclResult StubHrtGetDeviceRefreshSuccess(s32* deviceLogicId)
{
    *deviceLogicId = 3;
    return HCCL_SUCCESS;
}

HcclResult StubHrtGetDeviceRefreshFail(s32* deviceLogicId)
{
    (void)deviceLogicId;
    return HCCL_E_INTERNAL;
}
} // namespace

class UtOpbaseAdptTest : public testing::Test {
protected:
    void TearDown() override
    {
        GlobalMockObject::verify();
        g_hcclDeviceId = INVALID_INT;
    }
};

TEST_F(UtOpbaseAdptTest, HcclGetThreadDeviceIdReturnCachedDeviceIdWhenValid)
{
    g_hcclDeviceId = 0;
    MOCKER(hccl::GetMaxDevNum).stubs().with(outBound(8U)).will(returnValue(HCCL_SUCCESS));
    EXPECT_EQ(HcclGetThreadDeviceId(), 0);
}

TEST_F(UtOpbaseAdptTest, HcclGetThreadDeviceIdSuccessWhenFirstCall)
{
    g_hcclDeviceId = INVALID_INT;
    MOCKER(hrtGetDevice).stubs().with(any()).will(invoke(StubHrtGetDeviceSuccess));
    MOCKER(hccl::GetMaxDevNum).stubs().with(outBound(8U)).will(returnValue(HCCL_SUCCESS));
    EXPECT_EQ(HcclGetThreadDeviceId(), 3);
    EXPECT_EQ(g_hcclDeviceId, 3);
}

TEST_F(UtOpbaseAdptTest, HcclGetThreadDeviceIdFailWhenHrtGetDeviceFail)
{
    g_hcclDeviceId = INVALID_INT;
    MOCKER(hrtGetDevice).stubs().with(any()).will(invoke(StubHrtGetDeviceFail));
    EXPECT_EQ(HcclGetThreadDeviceId(), INVALID_INT);
    EXPECT_EQ(g_hcclDeviceId, INVALID_INT);
}

TEST_F(UtOpbaseAdptTest, HcclGetThreadDeviceIdRetrySuccessAfterHrtGetDeviceFail)
{
    g_hcclDeviceId = INVALID_INT;
    MOCKER(hrtGetDevice).stubs().with(any()).will(invoke(StubHrtGetDeviceFail));
    EXPECT_EQ(HcclGetThreadDeviceId(), INVALID_INT);
    EXPECT_EQ(g_hcclDeviceId, INVALID_INT);
    GlobalMockObject::verify();

    MOCKER(hrtGetDevice).stubs().with(any()).will(invoke(StubHrtGetDeviceSuccess));
    MOCKER(hccl::GetMaxDevNum).stubs().with(outBound(8U)).will(returnValue(HCCL_SUCCESS));
    EXPECT_EQ(HcclGetThreadDeviceId(), 3);
    EXPECT_EQ(g_hcclDeviceId, 3);
}

TEST_F(UtOpbaseAdptTest, HcclGetThreadDeviceIdFailWhenGetMaxDevNumFail)
{
    g_hcclDeviceId = 0;
    MOCKER(hccl::GetMaxDevNum).stubs().with(outBound(8U)).will(returnValue(HCCL_E_INTERNAL));
    EXPECT_EQ(HcclGetThreadDeviceId(), INVALID_INT);
}

TEST_F(UtOpbaseAdptTest, HcclGetThreadDeviceIdFailWhenDeviceIdExceedMaxDevNum)
{
    g_hcclDeviceId = 9;
    MOCKER(hccl::GetMaxDevNum).stubs().with(outBound(8U)).will(returnValue(HCCL_SUCCESS));
    EXPECT_EQ(HcclGetThreadDeviceId(), INVALID_INT);
}

TEST_F(UtOpbaseAdptTest, HcclDeviceRefreshSuccess)
{
    MOCKER(hrtGetDeviceRefresh).stubs().with(any()).will(invoke(StubHrtGetDeviceRefreshSuccess));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().with(any(), any(), any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_910)).will(returnValue(HCCL_SUCCESS));
    s32 deviceLogicId = INVALID_INT;
    EXPECT_EQ(HcclDeviceRefresh(deviceLogicId), HCCL_SUCCESS);
    EXPECT_EQ(deviceLogicId, 3);
    EXPECT_EQ(g_hcclDeviceId, 3);
}

TEST_F(UtOpbaseAdptTest, HcclDeviceRefreshFail)
{
    MOCKER(hrtGetDeviceRefresh).stubs().with(any()).will(invoke(StubHrtGetDeviceRefreshFail));
    s32 deviceLogicId = INVALID_INT;
    EXPECT_EQ(HcclDeviceRefresh(deviceLogicId), HCCL_E_INTERNAL);
}

TEST_F(UtOpbaseAdptTest, HcclDeviceRefreshFailWhenGetPhyIdFail)
{
    MOCKER(hrtGetDeviceRefresh).stubs().with(any()).will(invoke(StubHrtGetDeviceRefreshSuccess));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().with(any(), any(), any()).will(returnValue(HCCL_E_INTERNAL));
    s32 deviceLogicId = INVALID_INT;
    EXPECT_EQ(HcclDeviceRefresh(deviceLogicId), HCCL_E_INTERNAL);
    EXPECT_EQ(deviceLogicId, 3);
}

TEST_F(UtOpbaseAdptTest, HcclDeviceRefreshFailWhenGetDeviceTypeFail)
{
    MOCKER(hrtGetDeviceRefresh).stubs().with(any()).will(invoke(StubHrtGetDeviceRefreshSuccess));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().with(any(), any(), any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(any()).will(returnValue(HCCL_E_INTERNAL));
    s32 deviceLogicId = INVALID_INT;
    EXPECT_EQ(HcclDeviceRefresh(deviceLogicId), HCCL_E_INTERNAL);
    EXPECT_EQ(deviceLogicId, 3);
}
