/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NET_LAYER_TEST_COMMON_H
#define NET_LAYER_TEST_COMMON_H

#include <cstdlib>
#include <string>

#include "gtest/gtest.h"
#include "env_net_layer_config.h"

// 选层相关 UT 的公共夹具与常量；net_layer 解析与配置读取两个用例集共用。
namespace net_layer_ut {
using Mode = hccl::NetLayerSelectionMode;

constexpr uint32_t ALL_LAYERS = 0xFFU;
constexpr uint32_t UNCHANGED_OUTPUT = 0xA5A5A5A5U;
constexpr const char* NET_LAYER_ENV_NAME = "HCCL_USE_NET_LAYER";

class NetLayerSelectionLoadTest : public testing::Test {
protected:
    void SetUp() override
    {
        hccl::EnvNetLayerConfig::GetInstance().ResetParsed();
        const char* original = std::getenv(NET_LAYER_ENV_NAME);
        if (original != nullptr) {
            originalValue_ = original;
        }
        originalWasSet_ = original != nullptr;
        environmentCaptured_ = true;
        ASSERT_EQ(unsetenv(NET_LAYER_ENV_NAME), 0);
    }

    void TearDown() override
    {
        hccl::EnvNetLayerConfig::GetInstance().ResetParsed();
        // ASSERT提前退出或测试抛异常后，仍恢复测试前的唯一环境变量。
        if (!environmentCaptured_) {
            return;
        }
        if (originalWasSet_) {
            EXPECT_EQ(setenv(NET_LAYER_ENV_NAME, originalValue_.c_str(), 1), 0);
        } else {
            EXPECT_EQ(unsetenv(NET_LAYER_ENV_NAME), 0);
        }
    }

private:
    bool environmentCaptured_{false};
    bool originalWasSet_{false};
    std::string originalValue_;
};
} // namespace net_layer_ut

#endif // NET_LAYER_TEST_COMMON_H
