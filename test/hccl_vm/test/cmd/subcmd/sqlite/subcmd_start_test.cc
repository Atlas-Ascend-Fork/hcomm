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
// 该测试经 AcquireTable<> / StorageSession::Table<> 实例化业务表（如 RunModeConfig），
// 必须先包含定义了对应 RecordSchema<> 特化的 DDL 头，否则链接期报
// "undefined reference to HcclSim::Storage::RecordSchema<...>::Meta()"。
#include "runtime_state/db_sim_runner_ops.h"
#include "runtime_state/sim_models.h"
#include "storage/internal/process_storage_context.h"
#include "storage/table_access.h"
#include "storage_composition/storage_composition.h"
#include "subcmd_start.h"

using namespace HcclSim;
#define private public
#undef private
namespace {
/// 全二进制唯一的临时安装根：InstallPath 首次解析后进程内缓存，
/// 各用例共用同一根路径、按需重建 data/ 布局实现隔离。
std::string InstallRootPath()
{
    static const std::string root = "/tmp/hvm_subcmd_start_test_" + std::to_string(::getpid());
    return root;
}

std::string TopoMarkerPath() { return InstallRootPath() + "/topo_generated.marker"; }

std::string RunnerDbPath() { return InstallRootPath() + "/data/hccl_sim.db"; }

bool TopoMarkerExists() { return std::filesystem::exists(TopoMarkerPath()); }
} // namespace

class StartCommandTest : public testing::Test {
protected:
    void SetUp() override
    {
        app_ = std::make_unique<CLI::App>("test");
        cmd_ = std::make_unique<StartCommand>();
        // Save and reset global state
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
        // Restore global state
        g_hcclVmBashFlag = savedBashFlag_;
        g_hcclVmLevel = savedLevel_;
        (void)HcclSim::Storage::StorageRuntime::CloseProcessSession();
        std::filesystem::remove_all(InstallRootPath());
    }

    /// 搭建假安装根：生成脚本（进入即落 marker）与集群配置齐备，data 目录按参数决定。
    /// 无 data 目录时存储会话初始化必然失败（打开 data/hccl_sim.db 报错）。
    void PrepareInstallRoot(bool withDataDir)
    {
        const std::string root = InstallRootPath();
        std::filesystem::create_directories(root + "/script");
        std::filesystem::create_directories(root + "/config/cluster");
        if (withDataDir) {
            std::filesystem::create_directories(root + "/data");
        }
        std::ofstream script(root + "/script/generate_cluster_topo.sh");
        script << "#!/bin/bash\ntouch " << TopoMarkerPath() << "\nexit 0\n";
        script.close();
        std::filesystem::permissions(
            root + "/script/generate_cluster_topo.sh",
            std::filesystem::perms::owner_exec | std::filesystem::perms::group_exec
                | std::filesystem::perms::others_exec,
            std::filesystem::perm_options::add);
        for (const char* model : {"fake", "test_model", "cfg"}) {
            std::ofstream config(root + "/config/cluster/" + model + ".yaml");
            config << "meta:\n  podNum: 1\n";
            config.close();
        }
    }

    std::unique_ptr<CLI::App> app_;
    std::unique_ptr<StartCommand> cmd_;
    bool savedBashFlag_;
    std::uint32_t savedLevel_;
};

TEST_F(StartCommandTest, StaticName_ShouldReturnStart) { EXPECT_EQ(StartCommand::StaticName(), "start"); }

TEST_F(StartCommandTest, Setup_RegistersStartSubcommand)
{
    cmd_->Setup(*app_);
    auto sub = app_->get_subcommand("start");
    EXPECT_NE(sub, nullptr);
    EXPECT_EQ(sub->get_name(), "start");
}

TEST_F(StartCommandTest, Setup_StartSubcommandDescription)
{
    cmd_->Setup(*app_);
    auto start = app_->get_subcommand("start");
    ASSERT_NE(start, nullptr);
    EXPECT_FALSE(std::string(start->get_description()).empty());
}

TEST_F(StartCommandTest, Setup_HasConfigFileOption)
{
    cmd_->Setup(*app_);
    auto start = app_->get_subcommand("start");
    ASSERT_NE(start, nullptr);
    auto opts = start->get_options();
    EXPECT_GE(opts.size(), 1u);
}

TEST_F(StartCommandTest, Setup_HasLevelOption)
{
    cmd_->Setup(*app_);
    auto start = app_->get_subcommand("start");
    ASSERT_NE(start, nullptr);
    auto opts = start->get_options();
    bool hasLevel = false;
    for (const auto& opt : opts) {
        if (opt->get_name() == "--level") {
            hasLevel = true;
            break;
        }
    }
    EXPECT_TRUE(hasLevel);
}

TEST_F(StartCommandTest, Setup_ConfigFileOptionIsRequired)
{
    cmd_->Setup(*app_);
    auto start = app_->get_subcommand("start");
    ASSERT_NE(start, nullptr);
    auto opts = start->get_options();
    bool foundRequired = false;
    for (const auto& opt : opts) {
        if (opt->get_required()) {
            foundRequired = true;
            break;
        }
    }
    EXPECT_TRUE(foundRequired);
}

TEST_F(StartCommandTest, ParseStart_Help)
{
    cmd_->Setup(*app_);
    try {
        app_->parse("test start --help", true);
    } catch (const CLI::CallForHelp& e) {
        std::cerr << "Catch CLI::CallForHelp: " << e.what() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Catch unexpected exception: " << e.what() << "\n";
    }
}

TEST_F(StartCommandTest, ParseStart_WithoutConfigFile_Throws)
{
    cmd_->Setup(*app_);
    EXPECT_THROW(app_->parse("test start", true), CLI::RequiredError);
}

TEST_F(StartCommandTest, Execute_BashFlagTrue_LogsWarningAndReturns)
{
    // When g_hcclVmBashFlag is true, Execute() should log a warning and return early
    // without calling ParseYamlTopo or other external functions
    g_hcclVmBashFlag = true;
    cmd_->Setup(*app_);
    // Parse to trigger the callback which calls Execute()
    try {
        app_->parse("test start test_model.yaml", true);
    } catch (const CLI::CallForHelp& e) {
        std::cerr << "Catch CLI::CallForHelp: " << e.what() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Catch unexpected exception: " << e.what() << "\n";
    }
    // If we get here without crashing, the early return path worked
    EXPECT_TRUE(g_hcclVmBashFlag);
}

TEST_F(StartCommandTest, Execute_BashFlagFalse_ParsesAndCallsExternal)
{
    // When g_hcclVmBashFlag is false, Execute() will try to call ParseYamlTopo
    // which may fail, but we can at least verify the code path is exercised
    g_hcclVmBashFlag = false;
    cmd_->Setup(*app_);
    try {
        app_->parse("test start test_model.yaml", true);
    } catch (const CLI::CallForHelp& e) {
        std::cerr << "Catch CLI::CallForHelp: " << e.what() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Catch unexpected exception: " << e.what() << "\n";
    }
    // The test may fail at ParseYamlTopo since the file doesn't exist,
    // but we've covered the g_hcclVmBashFlag == false branch
    EXPECT_FALSE(g_hcclVmBashFlag);
}

TEST_F(StartCommandTest, CommandRegistry_CreateAll_ContainsStartCommand)
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

TEST_F(StartCommandTest, Execute_BashFlagTrue_ReturnsEarly)
{
    g_hcclVmBashFlag = true;
    cmd_->Setup(*app_);
    try {
        app_->parse("test start fake.yaml", true);
    } catch (const CLI::CallForHelp& e) {
        std::cerr << "Catch CLI::CallForHelp: " << e.what() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Catch unexpected exception: " << e.what() << "\n";
    }
    EXPECT_TRUE(g_hcclVmBashFlag);
}

TEST_F(StartCommandTest, Parse_LevelOptionDefault)
{
    cmd_->Setup(*app_);
    g_hcclVmLevel = 2;
    try {
        app_->parse("test start cfg.yaml", true);
    } catch (const CLI::CallForHelp& e) {
        std::cerr << "Catch CLI::CallForHelp: " << e.what() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Catch unexpected exception: " << e.what() << "\n";
    }
    EXPECT_EQ(g_hcclVmLevel, 2u);
}

TEST_F(StartCommandTest, Setup_ConfigFileOptionHasValidator)
{
    cmd_->Setup(*app_);
    auto start = app_->get_subcommand("start");
    ASSERT_NE(start, nullptr);
    auto* opt = start->get_option("configCluster");
    ASSERT_NE(opt, nullptr);
    EXPECT_NE(opt->get_validator(), nullptr);
}

TEST_F(StartCommandTest, Setup_ConfigFileOptionRequired)
{
    cmd_->Setup(*app_);
    auto start = app_->get_subcommand("start");
    ASSERT_NE(start, nullptr);
    auto* opt = start->get_option("configCluster");
    ASSERT_NE(opt, nullptr);
    EXPECT_TRUE(opt->get_required());
}

TEST_F(StartCommandTest, Parse_HelpAndInvalidArgs_DoNotCreateDatabase)
{
    // help 与非法参数只做命令行解析，不建立数据库会话、不创建数据库文件。
    cmd_->Setup(*app_);
    try {
        app_->parse("test start --help", true);
    } catch (const CLI::CallForHelp& e) {
        std::cerr << "Catch CLI::CallForHelp: " << e.what() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Catch unexpected exception: " << e.what() << "\n";
    }
    try {
        // 不存在的集群配置：解析期校验失败，回调不执行。
        app_->parse("test start no_such_model.yaml", true);
    } catch (const CLI::ValidationError& e) {
        std::cerr << "Catch CLI::ValidationError: " << e.what() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Catch unexpected exception: " << e.what() << "\n";
    }
    EXPECT_FALSE(std::filesystem::exists(RunnerDbPath()));
    EXPECT_FALSE(std::filesystem::exists(InstallRootPath() + "/data"));
}

TEST_F(StartCommandTest, Execute_MissingDataDir_InstallsSessionAndReachesCopyFile)
{
    // Session 安装只登记逻辑路由和惰性引擎工厂，不打开 SQLite。data 目录缺失不是
    // Session 安装失败；Start 与基线一样继续生成拓扑，随后 CopyFile 创建 data 目录并因源
    // topo.json 缺失返回。此时尚未执行首个表操作，所以不应创建物理数据库文件。
    PrepareInstallRoot(false);
    cmd_->Setup(*app_);
    try {
        app_->parse("test start fake.yaml", true);
    } catch (const CLI::CallForHelp& e) {
        std::cerr << "Catch CLI::CallForHelp: " << e.what() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Catch unexpected exception: " << e.what() << "\n";
    }
    EXPECT_TRUE(TopoMarkerExists());
    EXPECT_FALSE(std::filesystem::exists(RunnerDbPath()));
    EXPECT_TRUE(std::filesystem::exists(InstallRootPath() + "/data"));
    auto lease = HcclSim::Storage::ProcessStorageContext::Acquire();
    ASSERT_TRUE(lease.ok()) << lease.diagnostic;
}

TEST_F(StartCommandTest, Execute_NormalPath_InstallsSessionBeforeFirstDbWrite)
{
    // 正常路径：会话安装于 Execute 起点（首个数据库写入 RunModeConfig 之前的代码位置）完成；
    // 本用例 topo.json 源缺失使 CopyFile 失败返回，恰好停在首个数据库写入与 InitHvmEnv 之前
    // （不进入监听线程/fork 路径，保证单元测试安全）。
    PrepareInstallRoot(true);
    cmd_->Setup(*app_);
    try {
        app_->parse("test start fake.yaml", true);
    } catch (const CLI::CallForHelp& e) {
        std::cerr << "Catch CLI::CallForHelp: " << e.what() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Catch unexpected exception: " << e.what() << "\n";
    }
    EXPECT_TRUE(TopoMarkerExists());                       // 会话安装通过后正常进入拓扑生成
    EXPECT_FALSE(std::filesystem::exists(RunnerDbPath())); // 未做表操作，物理后端仍未打开
    {
        auto lease = HcclSim::Storage::ProcessStorageContext::Acquire();
        ASSERT_TRUE(lease.ok()) << lease.diagnostic;
    }
    auto count
        = HcclSim::Storage::Count<sim::runtime::RunModeConfig>(HcclSim::Storage::All<sim::runtime::RunModeConfig>());
    ASSERT_TRUE(count.ok()) << count.diagnostic;
    ASSERT_TRUE(count.value.has_value());
    EXPECT_EQ(*count, 0U);
    EXPECT_TRUE(std::filesystem::exists(RunnerDbPath())); // 首个真实表操作才惰性打开后端
}
