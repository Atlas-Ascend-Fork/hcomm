/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>

#include "store_sim_resource_root.h"

namespace fs = std::filesystem;

constexpr const char* kTestRoot = "/tmp/sim_resource_root_test_99999";

class SimResourceRootTest : public testing::Test {
protected:
    void SetUp() override
    {
        // 隔离测试环境：用独立 pid 目录，不污染真实进程目录
        setenv("HCCL_VM_RESOURCE_ROOT", kTestRoot, 1);
    }
    void TearDown() override
    {
        std::error_code ec;
        fs::remove_all(kTestRoot, ec);
        unsetenv("HCCL_VM_RESOURCE_ROOT");
    }
};

TEST_F(SimResourceRootTest, BuildName_PrependsPid)
{
    // 环境变量已设 /tmp/sim_resource_root_test_99999，BuildName 返回完整文件路径
    EXPECT_EQ(sim::SimResourceRoot::BuildName("ra_sock_pool_0"), "/tmp/sim_resource_root_test_99999/ra_sock_pool_0");
    EXPECT_EQ(sim::SimResourceRoot::BuildName("/HcclCommPool"), "/tmp/sim_resource_root_test_99999/HcclCommPool");
    EXPECT_EQ(sim::SimResourceRoot::BuildName(""), "/tmp/sim_resource_root_test_99999/");
}

TEST_F(SimResourceRootTest, Init_CreatesRootAndSyncDirs)
{
    ASSERT_TRUE(sim::SimResourceRoot::GetInstance().Init());
    EXPECT_TRUE(fs::is_directory(kTestRoot));
    EXPECT_TRUE(fs::is_directory(std::string(kTestRoot) + "/sync"));
    EXPECT_EQ(sim::SimResourceRoot::GetInstance().GetRoot(), kTestRoot);
    EXPECT_EQ(sim::SimResourceRoot::GetInstance().GetSyncDir(), std::string(kTestRoot) + "/sync");
}

TEST_F(SimResourceRootTest, Init_ForcesCleanExistingDir)
{
    fs::create_directories(std::string(kTestRoot) + "/stale");
    fs::create_directories(std::string(kTestRoot) + "/sync");
    ASSERT_TRUE(sim::SimResourceRoot::GetInstance().Init());
    EXPECT_FALSE(fs::exists(std::string(kTestRoot) + "/stale"));
    EXPECT_TRUE(fs::exists(std::string(kTestRoot) + "/sync"));
}

TEST_F(SimResourceRootTest, Cleanup_RemovesRootDir)
{
    ASSERT_TRUE(sim::SimResourceRoot::GetInstance().Init());
    sim::SimResourceRoot::GetInstance().Cleanup();
    EXPECT_FALSE(fs::exists(kTestRoot));
}

TEST_F(SimResourceRootTest, BuildName_StripsLeadingSlash)
{
    EXPECT_EQ(sim::SimResourceRoot::BuildName("//ra_sock"), "/tmp/sim_resource_root_test_99999/ra_sock");
}

TEST_F(SimResourceRootTest, CleanShmByPrefix_RemovesOnlyMatchingFiles)
{
    ASSERT_TRUE(sim::SimResourceRoot::GetInstance().Init());
    {
        std::ofstream(std::string(kTestRoot) + "/DEV_1_2_3").put('x');
        std::ofstream(std::string(kTestRoot) + "/ra_sock_pool_0").put('x');
        std::ofstream(std::string(kTestRoot) + "/keep.txt").put('x');
    }
    sim::SimResourceRoot::GetInstance().CleanShmByPrefix({"DEV", "ra_sock_"});
    EXPECT_FALSE(fs::exists(std::string(kTestRoot) + "/DEV_1_2_3"));
    EXPECT_FALSE(fs::exists(std::string(kTestRoot) + "/ra_sock_pool_0"));
    EXPECT_TRUE(fs::exists(std::string(kTestRoot) + "/keep.txt"));
}
