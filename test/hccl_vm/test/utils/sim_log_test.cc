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
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <gtest/gtest.h>

#include "sim_log.h"

namespace fs = std::filesystem;

#ifdef HCCLVM_TEST_CANN_HOST
extern "C" int32_t CheckLogLevel(int32_t, int32_t);
extern "C" void DlogRecord(int32_t, int32_t, const char*, ...);
#endif

class HcclVmLogTest : public testing::Test {
protected:
    void SetUp() override
    {
        std::string pattern = (fs::temp_directory_path() / "hcclvm-log-test-XXXXXX").string();
        const char* directory = mkdtemp(pattern.data());
        ASSERT_NE(directory, nullptr);
        root_ = directory;
        config_.filePath = root_.string();
        config_.consoleLevel = spdlog::level::off;
        config_.fileLevel = spdlog::level::info;
#ifdef HCCLVM_LOG_PROCESS_ROLE_DEVICE
        ASSERT_EQ(setenv("ASCEND_PROCESS_LOG_PATH", root_.c_str(), 1), 0);
#endif
#ifdef HCCLVM_TEST_CANN_HOST
        // host 在 main 前加载库；通过已加载句柄读取测试桩捕获的记录。
        library_ = dlopen("libunified_dlog.so", RTLD_NOW | RTLD_NOLOAD);
        ASSERT_NE(library_, nullptr);
        reset_ = reinterpret_cast<void (*)()>(dlsym(library_, "TestLogReset"));
        count_ = reinterpret_cast<int (*)()>(dlsym(library_, "TestLogCount"));
        module_ = reinterpret_cast<int (*)()>(dlsym(library_, "TestLogModule"));
        level_ = reinterpret_cast<int (*)()>(dlsym(library_, "TestLogLevel"));
        message_ = reinterpret_cast<const char* (*)()>(dlsym(library_, "TestLogMessage"));
        ASSERT_TRUE(reset_ && count_ && module_ && level_ && message_);
        reset_();
#endif
        InitLogger(config_, 47);
#ifdef HCCLVM_TEST_CANN_HOST
        ASSERT_EQ(g_logger, nullptr);
        ASSERT_NE(hcclvm::g_logWriter, nullptr);
#elif defined(HCCLVM_LOG_PROCESS_ROLE_DEVICE)
        ASSERT_NE(g_logger, nullptr);
        ASSERT_NE(hcclvm::g_logWriter, nullptr);
        ASSERT_EQ(g_logger->sinks().size(), 1U);
#else
        ASSERT_NE(g_logger, nullptr);
        ASSERT_EQ(hcclvm::g_logWriter, nullptr);
        ASSERT_EQ(g_logger->sinks().size(), 2U);
#endif
    }

    void TearDown() override
    {
        DeInitLogger();
#ifdef HCCLVM_TEST_CANN_HOST
        if (library_)
            dlclose(library_);
#endif
        // 仅清理由本用例 mkdtemp 创建的目录。
        if (root_.parent_path() == fs::temp_directory_path()
            && root_.filename().string().rfind("hcclvm-log-test-", 0) == 0) {
            fs::remove_all(root_);
        }
    }

    std::string Output()
    {
        FlushLog();
#ifdef HCCLVM_TEST_CANN_HOST
        return message_();
#else
        std::string text;
        for (const auto& entry : fs::recursive_directory_iterator(root_)) {
            if (!entry.is_regular_file())
                continue;
#ifdef HCCLVM_LOG_PROCESS_ROLE_DEVICE
            EXPECT_EQ(entry.path().parent_path(), root_ / "run/device-47");
#endif
            std::ifstream input(entry.path());
            text.append(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
        }
        return text;
#endif
    }

    bool Enabled(spdlog::level::level_enum level) const
    {
        const char* expected = std::getenv("TEST_LOG_LEVEL");
        // CTest 指定预期的 CANN 等级；stub 用例使用文件 sink 的 INFO 等级。
        return expected ? level >= std::atoi(expected) : level >= spdlog::level::info;
    }

    fs::path root_;
    LogConfig config_;
#ifdef HCCLVM_TEST_CANN_HOST
    void* library_ = nullptr;
    void (*reset_)() = nullptr;
    int (*count_)() = nullptr;
    int (*module_)() = nullptr;
    int (*level_)() = nullptr;
    const char* (*message_)() = nullptr;
#endif
};

TEST_F(HcclVmLogTest, FormatsAndWrites)
{
    HCCL_VM_INFO("value={:02x} percent=100%", 42);
    auto text = Output();
    EXPECT_EQ(text.find("value=2a percent=100%") != std::string::npos, Enabled(spdlog::level::info));
    HCCL_VM_ERROR("error value={}", 7);
    text = Output();
    EXPECT_EQ(text.find("error value=7") != std::string::npos, Enabled(spdlog::level::err));
#ifdef HCCLVM_TEST_CANN_HOST
    if (Enabled(spdlog::level::err)) {
        EXPECT_EQ(module_(), 3); // COMM_STUB -> HCCL，模拟器记录不附加 RUN_LOG_MASK。
        EXPECT_EQ(level_(), 3);
        EXPECT_NE(text.find("[HCCL-VM-COMM_STUB]"), std::string::npos);
    }
#endif
}

TEST_F(HcclVmLogTest, FiltersBeforeCannArgumentEvaluation)
{
    int evaluated = 0;
    HCCL_VM_DEBUG("filtered-debug {}", ++evaluated);
#ifdef HCCLVM_TEST_CANN
    EXPECT_EQ(evaluated, 0);
#else
    EXPECT_EQ(evaluated, 1); // stub 保持原 spdlog 模板路径，由 sink 过滤。
#endif
    EXPECT_EQ(Output().find("filtered-debug"), std::string::npos);
}

TEST_F(HcclVmLogTest, CloseDisablesLogging)
{
    DeInitLogger();
    EXPECT_EQ(g_logger, nullptr);
    EXPECT_EQ(hcclvm::g_logWriter, nullptr);
    int evaluated = 0;
    HCCL_VM_ERROR("after-close {}", ++evaluated);
    EXPECT_EQ(evaluated, 0);
}

#ifdef HCCLVM_TEST_CANN_HOST
TEST_F(HcclVmLogTest, NativeForwardingPreservesRunMask)
{
    constexpr int runModule = 3 | 0x01000000;
    EXPECT_EQ(CheckLogLevel(runModule, 1), 1);
    DlogRecord(runModule, 1, "native value=%d percent=100%%", 42);
    EXPECT_EQ(count_(), 1);
    EXPECT_EQ(module_(), runModule);
    EXPECT_EQ(level_(), 1);
    EXPECT_EQ(Output(), "native value=42 percent=100%");
}
#endif
