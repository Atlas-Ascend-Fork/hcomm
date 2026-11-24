/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "ping_mesh.h"

using namespace hccl;

// RC message struct size assertions (ensure stable network serialization layout)

TEST(HccnRcStruct_UT, ut_RcMsgHeader_Size)
{
    // msgType(1) + msgSeq(1) + cmdCode[4] = 6
    EXPECT_EQ(sizeof(RcMsgHeader), 6U);
}

TEST(HccnRcStruct_UT, ut_RcUnlinkBody_Size)
{
    // localEidRaw[16] + remoteEidRaw[16] = 32
    EXPECT_EQ(sizeof(RcUnlinkBody), 32U);
}

TEST(HccnRcStruct_UT, ut_RcTpInfo_Size)
{
    // tpHandle(8) + tpPsn(4) = 12, padded to u64 boundary: sizeof = 16
    EXPECT_EQ(sizeof(RcTpInfo), 16U);
}

TEST(HccnRcStruct_UT, ut_RcJettyInfo_Size)
{
    // key[28] + tokenValue(4) + errorCode(4) + size(1) = 37, padded to 4: sizeof = 40
    EXPECT_EQ(sizeof(RcJettyInfo), 40U);
}

TEST(HccnRcStruct_UT, ut_RcExchangeInfoSendable_Size)
{
    // localEidRaw[16] + remoteEidRaw[16] + RcTpInfo(16) + RcJettyInfo(40) = 88
    EXPECT_EQ(sizeof(RcExchangeInfoSendable), 88U);
}

TEST(HccnRcStruct_UT, ut_RcMsgType_EnumValues)
{
    EXPECT_EQ(static_cast<u8>(RcMsgType::LINK_REQ), 0U);
    EXPECT_EQ(static_cast<u8>(RcMsgType::LINK_RSP), 1U);
    EXPECT_EQ(static_cast<u8>(RcMsgType::UNLINK_REQ), 2U);
    EXPECT_EQ(static_cast<u8>(RcMsgType::UNLINK_RSP), 3U);
}

TEST(HccnRcStruct_UT, ut_PingMesh_initWithCfg_SetsTaMode_TpMode)
{
    PingMesh rping;
    HccnRpingInitAttr initAttr = {};
    initAttr.mode = HCCN_RPING_MODE_UB;
    HccnRpingInitCfg initCfg = {};
    initCfg.validCfgLen = sizeof(HccnRpingInitCfg);
    initCfg.ub.taMode = HCCN_RPING_UB_RC;
    initCfg.ub.tpMode = HCCN_RPING_UB_CTP;
    rping.initWithCfg(&initAttr, &initCfg);
    EXPECT_EQ(rping.GetTaMode(), HCCN_RPING_UB_RC);
    EXPECT_EQ(rping.GetTpMode(), HCCN_RPING_UB_CTP);
}

TEST(HccnRcStruct_UT, ut_PingMesh_init_DefaultTaMode_TpMode)
{
    PingMesh rping;
    HccnRpingInitAttr initAttr = {};
    initAttr.mode = HCCN_RPING_MODE_UB;
    rping.init(&initAttr);
    EXPECT_EQ(rping.GetTaMode(), HCCN_RPING_UB_UM);
    EXPECT_EQ(rping.GetTpMode(), HCCN_RPING_UB_UTP);
}

// HccnRpingInitCfg layout assertion: validCfgLen forward-compat check depends on this sizeof, lock ABI
TEST(HccnRcStruct_UT, ut_HccnRpingInitCfg_ValidCfgLen_MatchesSize)
{
    // validCfgLen(4) + ub union(264) = 268, 4-byte aligned without tail padding
    EXPECT_EQ(sizeof(HccnRpingInitCfg), 268U);
    // ub union members: taMode(4) + tpMode(4) + reserved[64](256) = 264
    EXPECT_EQ(sizeof(((HccnRpingInitCfg*)0)->ub), 264U);
}
