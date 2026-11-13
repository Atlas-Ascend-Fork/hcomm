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
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#include "json/common/json_parse_util.h"
#include "orion_adapter_rts.h"
#include "ip_address.h"

#include "rank_table_info.h"
#include "json/ranktable/rank_table_reader.h"

using namespace Hccl;

class RankTableInfoParserTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RankTableInfoParserTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "RankTableInfoParserTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in RankTableInfoParserTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in RankTableInfoParserTest TearDown" << std::endl;
    }
};

TEST_F(RankTableInfoParserTest, St_GetBinStream_When_Normal_Expect_Success)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string rankTableString = R"(
    {
    "version": "2.0",  
    "status": "completed",         
    "rank_count": 1,
    "rank_list": [
      {
        "rank_id": 0,
        "device_id": 0,  
        "local_id": 0,   
        "replaced_loacl_id": 0,      
        "level_list": [
          {
            "net_layer": 0,
            "net_instance_id": "superPod0-rack3",
            "net_type": "TOPO_FILE_DESC",  
            "net_attr": "",
            "rank_addr_list": [
              {
               "addr_type": "IPV4",
                "addr": "192.168.100.100",
                "ports": [ "0/1", "0/2" ]
              },
              {
               "addr_type": "IPV4",
                "addr": "192.168.100.100",
                "ports": [ "1/1", "1/2" ]
              }
            ]
          }
        ]
      }
      ]
     }
    )";

    RankTableReader rankTableReader(RankTableSource::RANKTABLE);
    RankTableInfo rankTableInfo;
    EXPECT_EQ(rankTableReader.Read(rankTableString, rankTableInfo), HCCL_SUCCESS);
    BinaryStream binStream;
    rankTableInfo.GetUniqueId(true);
    EXPECT_EQ(rankTableInfo.GetBinStream(true, binStream), HCCL_SUCCESS);
    RankTableInfo expecetedRankTableInfo(binStream);

    EXPECT_EQ(expecetedRankTableInfo.version, rankTableInfo.version);
    EXPECT_EQ(expecetedRankTableInfo.rankCount, rankTableInfo.rankCount);

    ASSERT_EQ(expecetedRankTableInfo.ranks.size(), rankTableInfo.ranks.size());
    for (auto i = 0; i < rankTableInfo.ranks.size(); i++) {
        EXPECT_EQ(expecetedRankTableInfo.ranks[i].rankId, rankTableInfo.ranks[i].rankId);
        EXPECT_EQ(expecetedRankTableInfo.ranks[i].localId, rankTableInfo.ranks[i].localId);
        EXPECT_EQ(expecetedRankTableInfo.ranks[i].deviceId, rankTableInfo.ranks[i].deviceId);

        ASSERT_EQ(expecetedRankTableInfo.ranks[i].rankLevelInfos.size(), rankTableInfo.ranks[i].rankLevelInfos.size());
        for (auto j = 0; j < rankTableInfo.ranks[i].rankLevelInfos.size(); j++) {
            EXPECT_EQ(
                expecetedRankTableInfo.ranks[i].rankLevelInfos[j].netLayer,
                rankTableInfo.ranks[i].rankLevelInfos[j].netLayer);
            EXPECT_EQ(
                expecetedRankTableInfo.ranks[i].rankLevelInfos[j].netInstId,
                rankTableInfo.ranks[i].rankLevelInfos[j].netInstId);
            EXPECT_EQ(
                expecetedRankTableInfo.ranks[i].rankLevelInfos[j].netType,
                rankTableInfo.ranks[i].rankLevelInfos[j].netType);

            ASSERT_EQ(
                expecetedRankTableInfo.ranks[i].rankLevelInfos[j].rankAddrs.size(),
                rankTableInfo.ranks[i].rankLevelInfos[j].rankAddrs.size());
            for (auto k = 0; k < rankTableInfo.ranks[i].rankLevelInfos[j].rankAddrs.size(); k++) {
                EXPECT_EQ(
                    expecetedRankTableInfo.ranks[i].rankLevelInfos[j].rankAddrs[k].addrType,
                    rankTableInfo.ranks[i].rankLevelInfos[j].rankAddrs[k].addrType);
                EXPECT_EQ(
                    expecetedRankTableInfo.ranks[i].rankLevelInfos[j].rankAddrs[k].addr,
                    rankTableInfo.ranks[i].rankLevelInfos[j].rankAddrs[k].addr);
                EXPECT_EQ(
                    expecetedRankTableInfo.ranks[i].rankLevelInfos[j].rankAddrs[k].ports,
                    rankTableInfo.ranks[i].rankLevelInfos[j].rankAddrs[k].ports);
            }
        }
    }
}

TEST_F(RankTableInfoParserTest, Ut_RankTableInfo_When_RankNEQ_Expect_InvalidParamsException)
{
    RankTableInfo rankTableInfo;
    rankTableInfo.version = "2.0";
    rankTableInfo.rankCount = 1;
    vector<NewRankInfo> ranks;
    NewRankInfo newRankInfo;
    newRankInfo.rankId = 0;
    newRankInfo.replacedLocalId = 1;
    ranks.push_back(newRankInfo);
    rankTableInfo.ranks = ranks;
    EXPECT_EQ(rankTableInfo.Check(), HCCL_E_PARA);
}
