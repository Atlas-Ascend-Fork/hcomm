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

#include "runtime_state/sim_models.h"
#include "operation_data/operation_data_types.h"

using namespace sim;
using namespace sim::runtime;

class SimModelsTest : public testing::Test {
protected:
};

TEST_F(SimModelsTest, Server_StructSize)
{
    static_assert(sizeof(Server) > 0, "Server struct should have size");
    Server s{1, 100, 0, 1, "v1"};
    EXPECT_EQ(s.id, 1);
    EXPECT_EQ(s.pod_id, 100);
}

TEST_F(SimModelsTest, Device_StructSize)
{
    static_assert(sizeof(Device) > 0, "Device struct should have size");
    Device d{1, 100, 255, 0, 0, 1, 0, 0};
    EXPECT_EQ(d.id, 1);
    EXPECT_EQ(d.server_id, 100);
}

TEST_F(SimModelsTest, Runner_StructSize)
{
    static_assert(sizeof(Runner) > 0, "Runner struct should have size");
    Runner r{1, 100, 1, 1, 1000, 1};
    EXPECT_EQ(r.id, 1);
}

TEST_F(SimModelsTest, OpExecutionKey_OrdersByCommunicatorThenIteration)
{
    sim::operation::OpExecutionKey first{"comm_a", 7, 2};
    sim::operation::OpExecutionKey second{"comm_b", 0, 0};
    sim::operation::OpExecutionKey earlierIteration{"comm_a", 7, 1};
    sim::operation::OpExecutionKey sameNameAndIterationDifferentHash{"comm_a", 6, 2};

    EXPECT_TRUE(first < second);
    EXPECT_TRUE(earlierIteration < first);
    EXPECT_FALSE(sameNameAndIterationDifferentHash < first);
    EXPECT_FALSE(first < sameNameAndIterationDifferentHash);
    EXPECT_FALSE(first < earlierIteration);
}

TEST_F(SimModelsTest, OpExecution_StoresNonContiguousDevicesWithTheirRanks)
{
    sim::operation::OpExecution execution;
    sim::operation::DeviceOpExecutionRecord first;
    first.deviceId = 17;
    first.rankId = 0;
    sim::operation::DeviceOpExecutionRecord second;
    second.deviceId = 3;
    second.rankId = 1;
    execution.deviceRecords.push_back(first);
    execution.deviceRecords.push_back(second);

    ASSERT_EQ(execution.deviceRecords.size(), 2);
    EXPECT_EQ(execution.deviceRecords[0].deviceId, 17);
    EXPECT_EQ(execution.deviceRecords[1].deviceId, 3);
    EXPECT_EQ(execution.deviceRecords[0].rankId, 0);
    EXPECT_EQ(execution.deviceRecords[1].rankId, 1);
}
