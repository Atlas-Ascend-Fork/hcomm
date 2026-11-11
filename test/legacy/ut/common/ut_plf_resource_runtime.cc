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
#include "common/device/plf_resource_runtime_pub.h"

namespace {

bool TestIsBatchLaunchMode() { return true; }

uint32_t TestGetSqFullTimeOut() { return 7; }

HcclResult TestHandleDispatchAllStreams() { return HCCL_E_PARA; }

bool TestGetProfL1State() { return true; }

HcclResult TestDumpSqe(const uint8_t*) { return HCCL_SUCCESS; }

HcclResult TestDumpWqe(const uint8_t*) { return HCCL_SUCCESS; }

u64 TestGetPlfDebugConfig() { return 42; }

void TestMarkCompactReportOp(void*, const void*) {}

uint32_t TestDeviceExecTimeout() { return 99; }

const Hccl::PlfResourceRuntimeOps TEST_OPS = {
    TestIsBatchLaunchMode, TestGetSqFullTimeOut, TestHandleDispatchAllStreams, TestGetProfL1State,
    TestDumpSqe,           TestDumpWqe,          TestGetPlfDebugConfig,
};

} // namespace

TEST(PlfResourceRuntimeTest, register_ops_and_reset_to_default)
{
    HcclRegisterPlfResourceRuntimeOps(&TEST_OPS);
    EXPECT_EQ(&TEST_OPS, &Hccl::GetPlfResourceRuntimeOps());
    EXPECT_EQ(42U, Hccl::GetPlfResourceRuntimeOps().getPlfDebugConfig());

    HcclRegisterPlfResourceRuntimeOps(nullptr);
    EXPECT_NE(&TEST_OPS, &Hccl::GetPlfResourceRuntimeOps());
}

TEST(PlfResourceRuntimeTest, mark_compact_report_op_registration)
{
    HcclSetPlfMarkCompactReportOp(TestMarkCompactReportOp);
    EXPECT_EQ(TestMarkCompactReportOp, HcclGetPlfMarkCompactReportOp());

    HcclSetPlfMarkCompactReportOp(nullptr);
    EXPECT_EQ(nullptr, HcclGetPlfMarkCompactReportOp());
}

TEST(PlfResourceRuntimeTest, device_exec_timeout_getter_registration)
{
    HcclSetPlfDeviceExecTimeoutGetter(TestDeviceExecTimeout);
    EXPECT_EQ(99U, HcclGetPlfDeviceExecTimeout());

    HcclSetPlfDeviceExecTimeoutGetter(nullptr);
    EXPECT_EQ(1836U, HcclGetPlfDeviceExecTimeout());
}
