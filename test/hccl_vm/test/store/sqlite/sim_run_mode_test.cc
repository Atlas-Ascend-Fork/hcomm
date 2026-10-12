/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <unistd.h>

#include <string>

#include <gtest/gtest.h>
#include "store_sim_run_mode.h"
#include "storage/internal/process_storage_context.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "storage/storage_session.h"
#include "runtime_state/sim_models.h"
#include "sqlite/simulation_storage_test_helper.h"

// 本二进制私有的 PID 唯一临时库路径：不与正式安装目录或其他测试库共享，
// 支持单例 filter、乱序与重复运行。
namespace {
std::string RunnerDbPath() { return "/tmp/hccl_vm_sim_run_mode_" + std::to_string(::getpid()) + "_runner.db"; }

std::string OpDataDbPath() { return "/tmp/hccl_vm_sim_run_mode_" + std::to_string(::getpid()) + "_opdata.db"; }
} // namespace

// fixture 逐用例安装显式测试 Session（经 ResetTestSession 产生对 composition
// bootstrap 的强符号引用），并逐用例校验数据准备结果，不再静默忽略失败。
class RunModeTest : public testing::Test {
protected:
    void SetUp() override { ASSERT_TRUE(runnerdb_test::ResetTestSession(RunnerDbPath(), OpDataDbPath())); }

    void TearDown() override { runnerdb_test::CleanUpDatabases(RunnerDbPath(), OpDataDbPath()); }
};

// 表为空时 ProbeCheckOnlyMode 返回 false。
TEST_F(RunModeTest, ProbeCheckOnlyMode_EmptyTable_False)
{
    ASSERT_TRUE(runnerdb_test::ClearRecords<sim::runtime::RunModeConfig>());
    EXPECT_FALSE(sim::ProbeCheckOnlyMode());
}

// 写入 mode=1 后 ProbeCheckOnlyMode 返回 true。
TEST_F(RunModeTest, ProbeCheckOnlyMode_CheckOnlyRow_True)
{
    ASSERT_TRUE(runnerdb_test::ClearRecords<sim::runtime::RunModeConfig>());

    sim::runtime::RunModeConfig config{};
    config.mode = 1;

    ASSERT_NE(runnerdb_test::InsertRecord(config), 0U);
    EXPECT_TRUE(sim::ProbeCheckOnlyMode());
}

// IsCheckOnlyMode 进程内 latch：第二次调用与第一次一致。
TEST_F(RunModeTest, IsCheckOnlyMode_Latches)
{
    bool first = sim::IsCheckOnlyMode();
    bool second = sim::IsCheckOnlyMode();
    EXPECT_EQ(first, second);
}
