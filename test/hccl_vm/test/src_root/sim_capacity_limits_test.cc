/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 文件职责：include/sim_capacity_limits.h 中环境变量派生常量的解析合同回归。
// 当前覆盖 HCCL_UB_MULTI_CHANNEL_NUM（UB 多 channel / 每 channel 的 jetty 数）：
// proxy 侧据此落库 OpDetailTab.ubMultiChannelNum，轻量性能仿真插件据此折算带宽利用率。
#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

#include "sim_capacity_limits.h"

namespace {
constexpr const char* kUbMultiChannelNumEnv = "HCCL_UB_MULTI_CHANNEL_NUM";

/// 环境变量注入守卫：构造时写入（value 为 nullptr 表示清除），析构时恢复原值，
/// 保证用例之间不串扰（解析函数不做进程内缓存，故每个用例可独立注入）。
class ScopedEnv {
public:
    ScopedEnv(const char* name, const char* value) : name_(name)
    {
        const char* previous = std::getenv(name);
        if (previous != nullptr) {
            hadPrevious_ = true;
            previousValue_ = previous;
        }
        if (value == nullptr) {
            ::unsetenv(name);
        } else {
            ::setenv(name, value, 1);
        }
    }

    ~ScopedEnv()
    {
        if (hadPrevious_) {
            ::setenv(name_.c_str(), previousValue_.c_str(), 1);
        } else {
            ::unsetenv(name_.c_str());
        }
    }

    ScopedEnv(const ScopedEnv&) = delete;
    ScopedEnv& operator=(const ScopedEnv&) = delete;

private:
    std::string name_;
    std::string previousValue_;
    bool hadPrevious_{false};
};
} // namespace

TEST(UbMultiChannelNumTest, EnvNotSetFallsBackToDefault)
{
    const ScopedEnv env(kUbMultiChannelNumEnv, nullptr);
    uint32_t parsed = 0;
    EXPECT_FALSE(HcclSim::ParseUbMultiChannelNum(parsed));
    EXPECT_EQ(parsed, HcclSim::HCCL_VM_UB_MULTI_CHANNEL_NUM_DEFAULT);
    EXPECT_EQ(HcclSim::GetUbMultiChannelNum(), 1U);
}

TEST(UbMultiChannelNumTest, EmptyValueFallsBackToDefault)
{
    const ScopedEnv env(kUbMultiChannelNumEnv, "");
    uint32_t parsed = 0;
    EXPECT_FALSE(HcclSim::ParseUbMultiChannelNum(parsed));
    EXPECT_EQ(HcclSim::GetUbMultiChannelNum(), 1U);
}

TEST(UbMultiChannelNumTest, ValidValuesAreParsed)
{
    for (const char* value : {"1", "2", "8", "16"}) {
        const ScopedEnv env(kUbMultiChannelNumEnv, value);
        uint32_t parsed = 0;
        EXPECT_TRUE(HcclSim::ParseUbMultiChannelNum(parsed)) << "value=" << value;
        EXPECT_EQ(parsed, HcclSim::GetUbMultiChannelNum()) << "value=" << value;
        EXPECT_EQ(parsed, static_cast<uint32_t>(std::strtoul(value, nullptr, 10))) << "value=" << value;
    }
}

TEST(UbMultiChannelNumTest, SurroundingWhitespaceIsTolerated)
{
    const ScopedEnv env(kUbMultiChannelNumEnv, " 4\t");
    uint32_t parsed = 0;
    EXPECT_TRUE(HcclSim::ParseUbMultiChannelNum(parsed));
    EXPECT_EQ(parsed, 4U);
    EXPECT_EQ(HcclSim::GetUbMultiChannelNum(), 4U);
}

TEST(UbMultiChannelNumTest, InvalidValuesFallBackToDefault)
{
    // 0 表示"没有 jetty"，非法；非十进制、带多余字符、溢出 uint32 同样非法。
    for (const char* value : {"0", "abc", "4x", "4 4", "-1", "4294967296", "18446744073709551616"}) {
        const ScopedEnv env(kUbMultiChannelNumEnv, value);
        uint32_t parsed = 0;
        EXPECT_FALSE(HcclSim::ParseUbMultiChannelNum(parsed)) << "value=" << value;
        EXPECT_EQ(parsed, HcclSim::HCCL_VM_UB_MULTI_CHANNEL_NUM_DEFAULT) << "value=" << value;
        EXPECT_EQ(HcclSim::GetUbMultiChannelNum(), 1U) << "value=" << value;
    }
}

TEST(UbMultiChannelNumTest, Uint32MaxIsAccepted)
{
    const ScopedEnv env(kUbMultiChannelNumEnv, "4294967295");
    uint32_t parsed = 0;
    EXPECT_TRUE(HcclSim::ParseUbMultiChannelNum(parsed));
    EXPECT_EQ(parsed, UINT32_MAX);
}
