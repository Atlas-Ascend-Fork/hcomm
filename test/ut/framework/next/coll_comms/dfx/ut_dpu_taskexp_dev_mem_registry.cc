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
#include "src/coll_communicator_mgr/dfx/taskException/aicpu/dpu_taskexp_dev_mem_registry.h"

class DpuTaskExpDevMemRegistryTest : public testing::Test {
public:
    void SetUp() override
    {
        // registry_ 含 std::mutex，不可移动；每用例由框架新构造 fixture，本处仅兜底清空残留
        registry_.Clear();
    }

protected:
    DpuTaskExpDevMemRegistry registry_;
};

TEST_F(DpuTaskExpDevMemRegistryTest, Ut_Register_When_NewKey_Expect_FindReturnsAddr)
{
    void* addr = reinterpret_cast<void*>(0x1000);
    registry_.Register("comm0", addr);
    EXPECT_EQ(addr, registry_.Find("comm0"));
}

TEST_F(DpuTaskExpDevMemRegistryTest, Ut_Find_When_MissingKey_Expect_Null)
{
    EXPECT_EQ(nullptr, registry_.Find("comm0"));
}

TEST_F(DpuTaskExpDevMemRegistryTest, Ut_Register_When_SameKey_Expect_Overwrite)
{
    void* oldAddr = reinterpret_cast<void*>(0x1000);
    void* newAddr = reinterpret_cast<void*>(0x2000);
    registry_.Register("comm0", oldAddr);
    registry_.Register("comm0", newAddr);
    // 同名通信域复用重建：新地址接管旧地址，避免 DFX 误读已释放的旧地址
    EXPECT_EQ(newAddr, registry_.Find("comm0"));
}

TEST_F(DpuTaskExpDevMemRegistryTest, Ut_Erase_When_AddrMatch_Expect_Removed)
{
    void* addr = reinterpret_cast<void*>(0x1000);
    registry_.Register("comm0", addr);
    registry_.Erase("comm0", addr);
    EXPECT_EQ(nullptr, registry_.Find("comm0"));
}

TEST_F(DpuTaskExpDevMemRegistryTest, Ut_Erase_When_AddrMismatch_Expect_Kept)
{
    void* oldAddr = reinterpret_cast<void*>(0x1000);
    void* newAddr = reinterpret_cast<void*>(0x2000);
    registry_.Register("comm0", oldAddr);
    registry_.Register("comm0", newAddr);
    // 迟到的旧销毁命令只能擦自己登记的旧地址，不得误删新登记
    registry_.Erase("comm0", oldAddr);
    EXPECT_EQ(newAddr, registry_.Find("comm0"));
}

TEST_F(DpuTaskExpDevMemRegistryTest, Ut_Clear_When_Multiple_Expect_AllRemoved)
{
    registry_.Register("comm0", reinterpret_cast<void*>(0x1000));
    registry_.Register("comm1", reinterpret_cast<void*>(0x2000));
    registry_.Clear();
    EXPECT_EQ(nullptr, registry_.Find("comm0"));
    EXPECT_EQ(nullptr, registry_.Find("comm1"));
}
