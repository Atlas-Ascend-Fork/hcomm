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
#include "common/protocol_utils_pub.h"
#include "common/comm_engine_utils_pub.h"
#include "hcomm_res_defs.h"

class ProtocolUtilsTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

TEST_F(ProtocolUtilsTest, Ut_GetCommProtocolStrMap_When_AllProtocols_Expect_CorrectString)
{
    auto& strMap = GetCommProtocolStrMap();
    EXPECT_EQ(strMap.at(COMM_PROTOCOL_RESERVED), "RESERVED");
    EXPECT_EQ(strMap.at(COMM_PROTOCOL_HCCS), "HCCS");
    EXPECT_EQ(strMap.at(COMM_PROTOCOL_ROCE), "ROCE");
    EXPECT_EQ(strMap.at(COMM_PROTOCOL_PCIE), "PCIE");
    EXPECT_EQ(strMap.at(COMM_PROTOCOL_SIO), "SIO");
    EXPECT_EQ(strMap.at(COMM_PROTOCOL_UB_CTP), "UB_CTP");
    EXPECT_EQ(strMap.at(COMM_PROTOCOL_UBC_TP), "UBC_TP");
    EXPECT_EQ(strMap.at(COMM_PROTOCOL_UB_MEM), "UB_MEM");
    EXPECT_EQ(strMap.at(COMM_PROTOCOL_UBOE), "UBOE");
    EXPECT_EQ(strMap.at(COMM_PROTOCOL_HCCS_ONLY), "HCCS_ONLY");
    EXPECT_EQ(strMap.at(COMM_PROTOCOL_UB_RTP), "UB_RTP");
}

TEST_F(ProtocolUtilsTest, Ut_GetEnumToString_When_UnknownValue_Expect_Unknown)
{
    EXPECT_EQ(GetEnumToString(GetCommProtocolStrMap(), static_cast<CommProtocol>(999)), "Unknown");
    EXPECT_EQ(GetEnumToString(GetCommEngineStatusStrMap(), static_cast<CommEngine>(999)), "Unknown");
}

TEST_F(ProtocolUtilsTest, Ut_GetThreadTypeStr_When_AllTypes_Expect_CorrectString)
{
    EXPECT_STREQ(GetThreadTypeStr(THREAD_TYPE_TS), "TS");
    EXPECT_STREQ(GetThreadTypeStr(THREAD_TYPE_CPU), "CPU");
    EXPECT_STREQ(GetThreadTypeStr(THREAD_TYPE_INVALID), "INVALID");
}
