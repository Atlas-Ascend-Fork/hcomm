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

#include "store_sim_shm_ops.h"

class ShmHeadTest : public testing::Test {
protected:
};

TEST_F(ShmHeadTest, ShmHead_Size)
{
    static_assert(sizeof(ShmHead) > 0, "ShmHead should have size");
    // packed 布局：magic4+version4+name256(含 pid 前缀)+size8+lock4+refCount4=280。
    // name 自 64 扩到 256（SimResourceRoot pid 前缀命名）后该断言未跟，两侧基线同红。
    EXPECT_EQ(sizeof(ShmHead), 280u);
}

TEST_F(ShmHeadTest, ShmConstants_Match)
{
    EXPECT_EQ(SHM_MAGIC, 0x53484D50);
    EXPECT_EQ(SHM_VERSION, 1);
}
