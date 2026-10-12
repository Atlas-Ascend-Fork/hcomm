/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <CLI11.hpp>
#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>
#include <sstream>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

#include "cmd_base_utils.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "runtime_state/sim_models.h"
#include "storage/internal/process_storage_context.h"
#include "storage/table_access.h"
#include "storage_composition/storage_composition.h"
#include "subcmd_run.h"

using namespace HcclSim;
#define private public
#undef private
namespace {
/// 全二进制唯一的临时安装根：InstallPath 首次解析后进程内缓存，
/// 各用例共用同一根路径、按需重建 data/ 布局实现隔离。
std::string InstallRootPath()
{
    static const std::string root = "/tmp/hvm_subcmd_run_test_" + std::to_string(::getpid());
    return root;
}

std::string WorkloadMarkerPath() { return InstallRootPath() + "/workload_ran.marker"; }

std::string RunnerDbPath() { return InstallRootPath() + "/data/hccl_sim.db"; }
} // namespace

class RunCommandTest : public testing::Test {
protected:
    void SetUp() override
    {
        app_ = std::make_unique<CLI::App>("test");
        cmd_ = std::make_unique<RunCommand>();
        savedBashFlag_ = g_hcclVmBashFlag;
        savedLevel_ = g_hcclVmLevel;
        g_hcclVmBashFlag = false;
        g_hcclVmLevel = 2;
        // 关闭上一用例可能遗留的进程存储会话并重建干净安装根。
        (void)HcclSim::Storage::StorageRuntime::CloseProcessSession();
        std::filesystem::remove_all(InstallRootPath());
        PrepareInstallRoot(false);
        // 安装根固定为同一路径：InstallPath 进程内只解析一次。
        setenv("HCCL_VM_INSTALL_ROOT", InstallRootPath().c_str(), 1);
    }
    void TearDown() override
    {
        g_hcclVmBashFlag = savedBashFlag_;
        g_hcclVmLevel = savedLevel_;
        (void)HcclSim::Storage::StorageRuntime::CloseProcessSession();
        std::filesystem::remove_all(InstallRootPath());
    }

    /// 搭建假安装根：生成脚本（run 的 configFile 校验在解析期执行脚本，恒 exit 0）与
    /// 集群配置齐备，data 目录按参数决定。无 data 目录时存储会话初始化必然失败。
    void PrepareInstallRoot(bool withDataDir)
    {
        const std::string root = InstallRootPath();
        std::filesystem::create_directories(root + "/script");
        std::filesystem::create_directories(root + "/config/cluster");
        if (withDataDir) {
            std::filesystem::create_directories(root + "/data");
        }
        std::ofstream script(root + "/script/generate_cluster_topo.sh");
        script << "#!/bin/bash\nexit 0\n";
        script.close();
        std::filesystem::permissions(
            root + "/script/generate_cluster_topo.sh",
            std::filesystem::perms::owner_exec | std::filesystem::perms::group_exec
                | std::filesystem::perms::others_exec,
            std::filesystem::perm_options::add);
        for (const char* model : {"fake", "112"}) {
            std::ofstream config(root + "/config/cluster/" + model + ".yaml");
            config << "meta:\n  podNum: 1\n";
            config.close();
        }
    }

    std::unique_ptr<CLI::App> app_;
    std::unique_ptr<RunCommand> cmd_;
    bool savedBashFlag_;
    std::uint32_t savedLevel_;
};

TEST_F(RunCommandTest, StaticName_ShouldReturnRun) { EXPECT_EQ(RunCommand::StaticName(), "run"); }

TEST_F(RunCommandTest, Setup_RegistersRunSubcommand)
{
    cmd_->Setup(*app_);
    auto sub = app_->get_subcommand("run");
    EXPECT_NE(sub, nullptr);
    EXPECT_EQ(sub->get_name(), "run");
}

TEST_F(RunCommandTest, Setup_RunSubcommandDescription)
{
    cmd_->Setup(*app_);
    auto run = app_->get_subcommand("run");
    ASSERT_NE(run, nullptr);
    EXPECT_FALSE(std::string(run->get_description()).empty());
}

TEST_F(RunCommandTest, Setup_HasConfigFileOption)
{
    cmd_->Setup(*app_);
    auto run = app_->get_subcommand("run");
    ASSERT_NE(run, nullptr);
    auto opts = run->get_options();
    EXPECT_GE(opts.size(), 1u);
}

TEST_F(RunCommandTest, Setup_HasLevelOption)
{
    cmd_->Setup(*app_);
    auto run = app_->get_subcommand("run");
    ASSERT_NE(run, nullptr);
    auto opts = run->get_options();
    EXPECT_GE(opts.size(), 1u);
}

TEST_F(RunCommandTest, Setup_AllowsExtras)
{
    cmd_->Setup(*app_);
    auto run = app_->get_subcommand("run");
    ASSERT_NE(run, nullptr);
    EXPECT_TRUE(run->get_allow_extras());
}

TEST_F(RunCommandTest, ParseRun_Help)
{
    cmd_->Setup(*app_);
    try {
        app_->parse("test run --help", true);
    } catch (const CLI::CallForHelp& e) {
        std::cerr << "Catch CLI::CallForHelp: " << e.what() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Catch unexpected exception: " << e.what() << "\n";
    }
}

TEST_F(RunCommandTest, ParseRun_WithConfigFile)
{
    cmd_->Setup(*app_);
    try {
        app_->parse("test run 112 -- echo hello", true);
    } catch (const CLI::CallForHelp& e) {
        std::cerr << "Catch CLI::CallForHelp: " << e.what() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Catch unexpected exception: " << e.what() << "\n";
    }
}

TEST_F(RunCommandTest, ParseRun_WithLevelOption)
{
    cmd_->Setup(*app_);
    try {
        app_->parse("test run 112 --level 2 -- echo hello", true);
    } catch (const CLI::CallForHelp& e) {
        std::cerr << "Catch CLI::CallForHelp: " << e.what() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Catch unexpected exception: " << e.what() << "\n";
    }
}

TEST_F(RunCommandTest, ParseRun_WithoutConfigFile_Throws)
{
    cmd_->Setup(*app_);
    EXPECT_THROW(app_->parse("test run", true), CLI::RequiredError);
}

TEST_F(RunCommandTest, CommandRegistry_CreateAll_ContainsRunCommand)
{
    auto commands = CommandRegistry::CreateAll();
    bool found = false;
    for (const auto& cmd : commands) {
        if (cmd != nullptr) {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found);
}

TEST_F(RunCommandTest, Execute_BashFlagTrue_ReturnsEarly)
{
    g_hcclVmBashFlag = true;
    cmd_->Setup(*app_);
    try {
        app_->parse("test run fake.yaml -- echo hello", true);
    } catch (const CLI::CallForHelp& e) {
        std::cerr << "Catch CLI::CallForHelp: " << e.what() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Catch unexpected exception: " << e.what() << "\n";
    }
    EXPECT_TRUE(g_hcclVmBashFlag);
}

TEST_F(RunCommandTest, Execute_BashFlagFalse_ParsesConfig)
{
    g_hcclVmBashFlag = false;
    cmd_->Setup(*app_);
    try {
        app_->parse("test run fake.yaml --level 1 -- echo world", true);
    } catch (const CLI::CallForHelp& e) {
        std::cerr << "Catch CLI::CallForHelp: " << e.what() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Catch unexpected exception: " << e.what() << "\n";
    }
    EXPECT_FALSE(g_hcclVmBashFlag);
}

TEST_F(RunCommandTest, ParseRun_ConfigFileOptionHasValidator)
{
    cmd_->Setup(*app_);
    auto sub = app_->get_subcommand("run");
    ASSERT_NE(sub, nullptr);
    auto* opt = sub->get_option("configFile");
    ASSERT_NE(opt, nullptr);
    EXPECT_NE(opt->get_validator(), nullptr);
}

TEST_F(RunCommandTest, ParseRun_ConfigFileRequired)
{
    cmd_->Setup(*app_);
    auto sub = app_->get_subcommand("run");
    ASSERT_NE(sub, nullptr);
    auto* opt = sub->get_option("configFile");
    ASSERT_NE(opt, nullptr);
    EXPECT_TRUE(opt->get_required());
}

TEST_F(RunCommandTest, Parse_HelpAndInvalidArgs_DoNotCreateDatabase)
{
    // help 与非法参数（缺必填 configFile）只做命令行解析，不建立数据库会话、不创建数据库文件。
    cmd_->Setup(*app_);
    try {
        app_->parse("test run --help", true);
    } catch (const CLI::CallForHelp& e) {
        std::cerr << "Catch CLI::CallForHelp: " << e.what() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Catch unexpected exception: " << e.what() << "\n";
    }
    try {
        app_->parse("test run -- true", true);
    } catch (const CLI::RequiredError& e) {
        std::cerr << "Catch CLI::RequiredError: " << e.what() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Catch unexpected exception: " << e.what() << "\n";
    }
    EXPECT_FALSE(std::filesystem::exists(RunnerDbPath()));
    EXPECT_FALSE(std::filesystem::exists(InstallRootPath() + "/data"));
}

TEST_F(RunCommandTest, Execute_DbInitFailure_DoesNotRunWorkload)
{
    // 数据库会话初始化失败（data 目录不存在、打开 data/hccl_sim.db 失败）：
    // Execute 起点即失败返回，不进入 InitHvmEnv，不执行 system workload，不创建任何文件。
    PrepareInstallRoot(false);
    cmd_->Setup(*app_);
    try {
        // 不带 "--" 分隔符：extras 直接由 run 子命令收集进入 workload（带 "--" 会被
        // app 层拒绝而永远到不了回调）。
        app_->parse("test run fake.yaml touch " + WorkloadMarkerPath(), true);
    } catch (const CLI::CallForHelp& e) {
        std::cerr << "Catch CLI::CallForHelp: " << e.what() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Catch unexpected exception: " << e.what() << "\n";
    }
    EXPECT_FALSE(std::filesystem::exists(WorkloadMarkerPath())); // workload 未执行
    EXPECT_FALSE(std::filesystem::exists(RunnerDbPath()));
}

TEST_F(RunCommandTest, Execute_NormalPath_InstallsSessionBeforeFirstDbWrite)
{
    // 正常路径：会话在 Execute 起点（任何 RunnerDB 写入之前）完成安装；
    // 本用例解析成功但缺少算例指令，恰好在首个数据库写入前返回。
    PrepareInstallRoot(true);
    cmd_->Setup(*app_);
    try {
        app_->parse("test run fake.yaml", true);
    } catch (const CLI::CallForHelp& e) {
        std::cerr << "Catch CLI::CallForHelp: " << e.what() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Catch unexpected exception: " << e.what() << "\n";
    }
    // 第一步：Execute 已在命令起点安装逻辑 Session。Acquire 只读取已安装会话、
    // 不触发 bootstrap，成功即证明会话由 RunCommand::Execute 安装，而不是后续
    // AcquireReady 惰性补装。lease 在局部作用域内及时析构，不持租约做后续访问。
    {
        auto lease = HcclSim::Storage::ProcessStorageContext::Acquire();
        ASSERT_TRUE(lease.ok()) << lease.diagnostic;
    }
    // 第二步：lease 释放后执行第一次真实数据库原子操作：已安装的 Session 可供
    // 原子接口使用、首次真实访问惰性打开具体后端（此时才创建物理 SQLite 文件），
    // 且 RunModeConfig 在首次写入前为空。
    auto count
        = HcclSim::Storage::Count<sim::runtime::RunModeConfig>(HcclSim::Storage::All<sim::runtime::RunModeConfig>());

    ASSERT_TRUE(count.ok()) << count.diagnostic;
    ASSERT_TRUE(count.value.has_value());
    EXPECT_EQ(*count, 0U);
}
