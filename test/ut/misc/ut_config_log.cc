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
#include <cstdlib>

// 被测的g_debugConfig/g_debugConfigInited为文件内static且无对外复位接口,
// 直接包含源文件使static进入本编译单元以复位状态。
#include "../../../src/legacy/ascend910/common/debug/config/config_log.cc"

TEST(ConfigLogTest, ut_config_log_env_not_set_expect_inited)
{
    unsetenv("HCCL_DEBUG_CONFIG");
    hccl::g_debugConfig = 0ULL;
    hccl::g_debugConfigInited = false;
    EXPECT_FALSE(hccl::GetDebugConfigInited());
    EXPECT_EQ(hccl::InitDebugConfigByEnv(), HCCL_SUCCESS);
    EXPECT_EQ(hccl::GetDebugConfig(), 0ULL);
    EXPECT_TRUE(hccl::GetDebugConfigInited());

    EXPECT_EQ(hccl::InitDebugConfigByEnv(), HCCL_SUCCESS);
    EXPECT_EQ(hccl::GetDebugConfig(), 0ULL);
    EXPECT_TRUE(hccl::GetDebugConfigInited());

    hccl::g_debugConfig = 0ULL;
    hccl::g_debugConfigInited = false;
}

TEST(ConfigLogTest, ut_config_log_env_set_expect_parsed)
{
    setenv("HCCL_DEBUG_CONFIG", "alg,task", 1);
    hccl::g_debugConfig = 0ULL;
    hccl::g_debugConfigInited = false;
    EXPECT_FALSE(hccl::GetDebugConfigInited());
    EXPECT_EQ(hccl::InitDebugConfigByEnv(), HCCL_SUCCESS);
    EXPECT_EQ(hccl::GetDebugConfig(), hccl::HCCL_ALG | hccl::HCCL_TASK);
    EXPECT_TRUE(hccl::GetDebugConfigInited());
    unsetenv("HCCL_DEBUG_CONFIG");

    hccl::g_debugConfig = 0ULL;
    hccl::g_debugConfigInited = false;
}
