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
#include <mockcpp/mockcpp.hpp>
#include <optional>

#include "dev_type_utils.h"

using namespace hcomm;

namespace {
DevType g_stubDevType = DevType::DEV_TYPE_910;
HcclResult g_stubRet = HCCL_SUCCESS;

HcclResult StubHrtGetDeviceType(DevType& devType)
{
    devType = g_stubDevType;
    return g_stubRet;
}
} // namespace

class DevTypeUtilsTest : public testing::Test {
protected:
    void SetUp() override
    {
        g_stubDevType = DevType::DEV_TYPE_910;
        g_stubRet = HCCL_SUCCESS;
        MOCKER(hrtGetDeviceType).stubs().will(invoke(StubHrtGetDeviceType));
    }
    void TearDown() override { GlobalMockObject::verify(); }
    // 切换打桩返回的设备类型与返回码（stub 函数读取全局状态，可反复切换）
    static void MockAs(DevType devType, HcclResult ret = HCCL_SUCCESS)
    {
        g_stubDevType = devType;
        g_stubRet = ret;
    }
};

namespace {
struct TruthRow {
    DevType devType;
    bool eq950;
    bool eq960;
    bool ge950;
    bool ge960;
    bool lt950;
};
} // namespace

// 全真值表：10 种设备类型 × 5 谓词 = 50 断言
TEST_F(DevTypeUtilsTest, ut_FivePredicates_TruthTable_Expect_Match)
{
    static const TruthRow rows[] = {
        {DevType::DEV_TYPE_910, false, false, false, false, true},
        {DevType::DEV_TYPE_310P3, false, false, false, false, true},
        {DevType::DEV_TYPE_910B, false, false, false, false, true},
        {DevType::DEV_TYPE_310P1, false, false, false, false, true},
        {DevType::DEV_TYPE_910_93, false, false, false, false, true},
        {DevType::DEV_TYPE_NOSOC, false, false, false, false, false},
        {DevType::DEV_TYPE_950, true, false, true, false, false},
        {DevType::DEV_TYPE_MC62, false, false, false, false, false},
        {DevType::DEV_TYPE_960, false, true, true, true, false},
        {DevType::DEV_TYPE_COUNT, false, false, false, false, false},
    };
    for (const auto& row : rows) {
        MockAs(row.devType);
        auto eq950 = EqualTo950();
        auto eq960 = EqualTo960();
        auto ge950 = EqualToOrGreaterThan950();
        auto ge960 = EqualToOrGreaterThan960();
        auto lt950 = LessThan950();
        ASSERT_TRUE(eq950.has_value());
        ASSERT_TRUE(eq960.has_value());
        ASSERT_TRUE(ge950.has_value());
        ASSERT_TRUE(ge960.has_value());
        ASSERT_TRUE(lt950.has_value());
        EXPECT_EQ(*eq950, row.eq950);
        EXPECT_EQ(*eq960, row.eq960);
        EXPECT_EQ(*ge950, row.ge950);
        EXPECT_EQ(*ge960, row.ge960);
        EXPECT_EQ(*lt950, row.lt950);
    }
}

// 失败路径：hrtGetDeviceType 返回失败时五谓词全部返回 nullopt
TEST_F(DevTypeUtilsTest, ut_FivePredicates_GetDevTypeFailed_Expect_AllNullopt)
{
    MockAs(DevType::DEV_TYPE_950, HCCL_E_INTERNAL);
    EXPECT_FALSE(EqualTo950().has_value());
    EXPECT_FALSE(EqualTo960().has_value());
    EXPECT_FALSE(EqualToOrGreaterThan950().has_value());
    EXPECT_FALSE(EqualToOrGreaterThan960().has_value());
    EXPECT_FALSE(LessThan950().has_value());
}

// 否定形回归：NOSOC/MC62 下 !*EqualToOrGreaterThan950() 为 true 而 *LessThan950() 为 false，五谓词非穷举划分
TEST_F(DevTypeUtilsTest, ut_NegationForm_NosocMc62_Expect_NotLessThan950)
{
    for (DevType devType : {DevType::DEV_TYPE_NOSOC, DevType::DEV_TYPE_MC62}) {
        MockAs(devType);
        auto ge950 = EqualToOrGreaterThan950();
        auto lt950 = LessThan950();
        ASSERT_TRUE(ge950.has_value());
        ASSERT_TRUE(lt950.has_value());
        EXPECT_TRUE(!(*ge950)); // 否定形：非 950+
        EXPECT_FALSE(*lt950);   // 但也不是 950 之前
    }
}
