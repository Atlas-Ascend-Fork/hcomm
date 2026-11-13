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
#include <cstdlib>
#include <set>
#include <utility>
#include "json/common/json_parse_util.h"
#include "orion_adapter_rts.h"
#include "ip_address.h"

#include "rank_table_info.h"
#include "topo_common_types.h"
#include "adapter_error_manager_pub.h"
#include "invalid_params_exception.h"
#include "json/ranktable/rank_table_fast_parser.h"
#include "json/ranktable/rank_table_reader.h"
#include "json/ranktable/json_scan_simd.h"
#include "json/json_api.h"
#include "changed_rank_info.h"

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

TEST_F(RankTableInfoParserTest, Ut_GetBinStream_When_Normal_Expect_Success)
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

TEST_F(RankTableInfoParserTest, Ut_RankTableInfo_When_RankNEQ_Expect_E_PARA)
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

TEST_F(RankTableInfoParserTest, Ut_Check_When_MultipleReplacedRank_Expect_Exception)
{
    MOCKER(RptInputErr).stubs().will(returnValue(HCCL_SUCCESS));

    RankTableInfo rankTableInfo;
    rankTableInfo.version = "2.0";
    rankTableInfo.rankCount = 2;

    NewRankInfo rank0;
    rank0.rankId = 0;
    rank0.localId = BACKUP_LOCAL_ID;
    rank0.replacedLocalId = 0;

    NewRankInfo rank1;
    rank1.rankId = 1;
    rank1.localId = BACKUP_LOCAL_ID;
    rank1.replacedLocalId = 1;

    rankTableInfo.ranks = {rank0, rank1};
    EXPECT_EQ(rankTableInfo.Check(), HCCL_E_PARA);
}

TEST_F(RankTableInfoParserTest, Ut_Check_When_DuplicateRankId_Expect_Exception)
{
    MOCKER(RptInputErr).stubs().will(returnValue(HCCL_SUCCESS));

    RankTableInfo rankTableInfo;
    rankTableInfo.version = "2.0";
    rankTableInfo.rankCount = 2;

    NewRankInfo rank0;
    rank0.rankId = 0;
    rank0.localId = 0;
    rank0.replacedLocalId = 0;

    NewRankInfo rank1;
    rank1.rankId = 0;
    rank1.localId = 1;
    rank1.replacedLocalId = 1;

    rankTableInfo.ranks = {rank0, rank1};
    EXPECT_EQ(rankTableInfo.Check(), HCCL_E_PARA);
}

TEST_F(RankTableInfoParserTest, Ut_Check_When_SameLocalIdWithReplaced_Expect_Exception)
{
    MOCKER(RptInputErr).stubs().will(returnValue(HCCL_SUCCESS));

    RankTableInfo rankTableInfo;
    rankTableInfo.version = "2.0";
    rankTableInfo.rankCount = 2;

    NewRankInfo rank0;
    rank0.rankId = 0;
    rank0.localId = BACKUP_LOCAL_ID;
    rank0.replacedLocalId = 1;

    NewRankInfo rank1;
    rank1.rankId = 1;
    rank1.localId = 1;
    rank1.replacedLocalId = 1;

    rankTableInfo.ranks = {rank0, rank1};
    EXPECT_EQ(rankTableInfo.Check(), HCCL_E_PARA);
}

// 验证 rank_id 逆序时快速解析后 ranks 已按 rankId 升序排列。

namespace {
const std::string FAST_RANK_TABLE_2X2_REVERSED = R"(
{
  "version": "2.0",
  "rank_count": 2,
  "rank_list": [
    {
      "rank_id": 1,
      "local_id": 1,
      "device_id": 1,
      "device_port": 16666,
      "host_port": 16666,
      "level_list": [
        {"net_layer": 0, "net_instance_id": "sp_0_srv_0", "net_type": "TOPO_FILE_DESC", "net_attr": "",
         "rank_addr_list": [{"addr_type": "IPV4", "addr": "10.0.0.2", "ports": ["0/0"]}]},
        {"net_layer": 1, "net_instance_id": "sp_0_srv_1", "net_type": "TOPO_FILE_DESC", "net_attr": "",
         "rank_addr_list": [{"addr_type": "IPV4", "addr": "10.0.1.2", "ports": ["0/0"]}]}
      ]
    },
    {
      "rank_id": 0,
      "local_id": 0,
      "device_id": 0,
      "device_port": 16666,
      "host_port": 16666,
      "level_list": [
        {"net_layer": 0, "net_instance_id": "sp_0_srv_0", "net_type": "TOPO_FILE_DESC", "net_attr": "",
         "rank_addr_list": [{"addr_type": "IPV4", "addr": "10.0.0.1", "ports": ["0/0"]}]},
        {"net_layer": 1, "net_instance_id": "sp_0_srv_1", "net_type": "TOPO_FILE_DESC", "net_attr": "",
         "rank_addr_list": [{"addr_type": "IPV4", "addr": "10.0.1.1", "ports": ["0/0"]}]}
      ]
    }
  ]
}
)";

std::string MakeBackupAddrTable(size_t count)
{
    std::string json = R"(
{
  "version": "2.0",
  "rank_count": 1,
  "rank_list": [
    {"rank_id": 0, "local_id": 0, "device_id": 0, "device_port": 16666, "host_port": 16666,
     "level_list": [
       {"net_layer": 0, "net_instance_id": "sp_0_srv_0", "net_type": "TOPO_FILE_DESC", "net_attr": "",
        "rank_addr_list": [{"addr_type": "IPV4", "addr": "10.0.0.1", "ports": ["0/0"], "backup_addr": [)"
                       +
                       [&]() {
                           std::string addrs;
                           for (size_t i = 0; i < count; ++i) {
                               if (i != 0) {
                                   addrs += ",";
                               }
                               addrs += "\"10.0.1." + std::to_string(i + 1) + "\"";
                           }
                           return addrs;
                       }()
                       + R"(]}]}
     ]}
  ]
}
)";
    return json;
}
} // namespace

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_SortsRanksByRankId)
{
    RankTableInfo info;
    EXPECT_EQ(ParseRankTableStringFast(FAST_RANK_TABLE_2X2_REVERSED, info), HCCL_SUCCESS);
    ASSERT_EQ(info.ranks.size(), 2U);
    EXPECT_EQ(info.ranks[0].rankId, 0);
    EXPECT_EQ(info.ranks[1].rankId, 1);
}

namespace {
const std::string FAST_RANK_TABLE_FULL = R"(
{
  "version": "2.0",
  "status": "completed",
  "detour": "false",
  "rank_count": 2,
  "rank_list": [
    {
      "rank_id": 0,
      "local_id": 0,
      "device_id": 0,
      "device_port": 16666,
      "host_port": 16666,
      "level_list": [
        {
          "net_layer": 0,
          "net_instance_id": "sp_0_srv_0",
          "net_type": "TOPO_FILE_DESC",
          "net_attr": "",
          "rank_addr_list": [
            {"addr_type": "IPV4", "addr": "10.0.0.1", "ports": ["0/0"], "plane_id": "plane0",
             "backup_addr": ["10.0.9.1", "10.0.9.2"]},
            {"addr_type": "IPV4", "addr": "10.0.0.11", "ports": ["0/1"], "plane_id": "plane1"}
          ]
        },
        {
          "net_layer": 1,
          "net_instance_id": "sp_1_srv_0",
          "net_type": "CLOS",
          "net_attr": "",
          "rank_addr_list": [
            {"addr_type": "IPV4", "addr": "10.1.0.1", "ports": ["1/0"], "plane_id": "plane2"}
          ]
        }
      ],
      "control_plane": {"addr_type": "IPV4", "addr": "10.0.0.1", "listen_port": 8000}
    },
    {
      "rank_id": 1,
      "local_id": 1,
      "device_id": 1,
      "device_port": 16666,
      "host_port": 16666,
      "level_list": [
        {
          "net_layer": 0,
          "net_instance_id": "sp_0_srv_1",
          "net_type": "TOPO_FILE_DESC",
          "net_attr": "",
          "rank_addr_list": [
            {"addr_type": "IPV4", "addr": "10.0.0.2", "ports": ["0/0"], "plane_id": "plane0"}
          ]
        },
        {
          "net_layer": 1,
          "net_instance_id": "sp_1_srv_0",
          "net_type": "CLOS",
          "net_attr": "",
          "rank_addr_list": [
            {"addr_type": "IPV4", "addr": "10.1.0.2", "ports": ["1/0"], "plane_id": "plane2"}
          ]
        }
      ],
      "control_plane": {"addr_type": "IPV4", "addr": "10.0.0.2", "listen_port": 8000}
    }
  ]
}
)";

std::string ReplaceOnce(std::string text, const std::string& from, const std::string& to)
{
    const size_t pos = text.find(from);
    EXPECT_NE(pos, std::string::npos) << "pattern not found: " << from;
    if (pos != std::string::npos) {
        text.replace(pos, from.size(), to);
    }
    return text;
}

std::string WithoutLevelList()
{
    return R"({"version":"2.0","rank_count":1,"rank_list":[
        {"rank_id":0,"local_id":0,"device_id":0}]})";
}

std::string WithTrailingCommaInRankList()
{
    return R"({"version":"2.0","rank_count":1,"rank_list":[
        {"rank_id":0,"local_id":0,"device_id":0,"level_list":[
          {"net_layer":0,"net_instance_id":"sp_0_srv_0","net_type":"TOPO_FILE_DESC",
           "rank_addr_list":[{"addr_type":"IPV4","addr":"10.0.0.1","ports":["0/0"]}]}]},]})";
}

const std::string FAST_CHANGED_RANK_INFO = R"(
{
  "version": "2.0",
  "rank_count": 1,
  "rank_list": [
    {
      "rank_id": 0,
      "local_id": 0,
      "device_id": 0,
      "device_port": 16666,
      "host_port": 16666,
      "level_list": [
        {
          "net_layer": 0,
          "net_instance_id": "sp_0_srv_0",
          "net_type": "TOPO_FILE_DESC",
          "net_attr": "",
          "rank_addr_list": [
            {"addr_type": "IPV4", "addr": "10.0.0.1", "ports": ["0/0"], "plane_id": "plane0"}
          ]
        }
      ]
    }
  ]
}
)";
} // namespace

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_MissingRequiredField_Expect_Para)
{
    const std::vector<std::pair<std::string, std::string>> cases = {
        {"rank_id", R"("rank_id": 0,)"},
        {"local_id", R"("local_id": 0,)"},
        {"device_id", R"("device_id": 0,)"},
        {"version", R"("version": "2.0",)"},
    };
    for (const auto& item : cases) {
        RankTableInfo info;
        const std::string json = ReplaceOnce(FAST_RANK_TABLE_FULL, item.second, "");
        EXPECT_EQ(ParseRankTableStringFast(json, info), HCCL_E_PARA) << "missing " << item.first;
    }

    RankTableInfo info;
    EXPECT_EQ(ParseRankTableStringFast(WithoutLevelList(), info), HCCL_E_PARA);
}

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_BackupRankWithoutReplacedId_Expect_Para)
{
    const std::string json = ReplaceOnce(
        FAST_RANK_TABLE_FULL, R"("local_id": 1,)", "\"local_id\": " + std::to_string(BACKUP_LOCAL_ID) + ",");
    RankTableInfo info;
    EXPECT_EQ(ParseRankTableStringFast(json, info), HCCL_E_PARA);
}

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_BackupRankWithReplacedId_Expect_Success)
{
    const std::string json = ReplaceOnce(
        FAST_RANK_TABLE_FULL, R"("local_id": 1,)",
        "\"local_id\": " + std::to_string(BACKUP_LOCAL_ID) + ", \"replaced_local_id\": 1,");
    RankTableInfo info;
    ASSERT_EQ(ParseRankTableStringFast(json, info), HCCL_SUCCESS);
    ASSERT_EQ(info.ranks.size(), 2U);
    EXPECT_EQ(info.ranks[1].localId, BACKUP_LOCAL_ID);
    EXPECT_EQ(info.ranks[1].replacedLocalId, 1U);

    const std::string badReplaced = ReplaceOnce(
        FAST_RANK_TABLE_FULL, R"("local_id": 1,)",
        "\"local_id\": " + std::to_string(BACKUP_LOCAL_ID)
            + ", \"replaced_local_id\": " + std::to_string(BACKUP_LOCAL_ID) + ",");
    RankTableInfo badInfo;
    EXPECT_EQ(ParseRankTableStringFast(badReplaced, badInfo), HCCL_E_PARA);
}

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_InvalidVersion_Expect_Para)
{
    const std::string json = ReplaceOnce(FAST_RANK_TABLE_FULL, R"("version": "2.0")", R"("version": "8.0")");
    RankTableInfo info;
    EXPECT_EQ(ParseRankTableStringFast(json, info), HCCL_E_PARA);
}

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_PortsMissing_Expect_Default)
{
    std::string json = FAST_RANK_TABLE_FULL;
    json = ReplaceOnce(json, R"("device_port": 16666,)", "");
    json = ReplaceOnce(json, R"("host_port": 16666,)", "");
    RankTableInfo info;
    ASSERT_EQ(ParseRankTableStringFast(json, info), HCCL_SUCCESS);
    EXPECT_EQ(info.ranks[0].devicePort, DEFAULT_VALUE_TCPPORT);
    EXPECT_EQ(info.ranks[0].hostPort, DEFAULT_VALUE_TCPPORT);
}

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_RankCountMismatch_Expect_Para)
{
    const std::string json = ReplaceOnce(FAST_RANK_TABLE_FULL, R"("rank_count": 2)", R"("rank_count": 3)");
    RankTableInfo info;
    EXPECT_EQ(ParseRankTableStringFast(json, info), HCCL_E_PARA);
}

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_DuplicateRankId_Expect_Para)
{
    const std::string json = ReplaceOnce(FAST_RANK_TABLE_FULL, R"("rank_id": 1,)", R"("rank_id": 0,)");
    RankTableInfo info;
    EXPECT_EQ(ParseRankTableStringFast(json, info), HCCL_E_PARA);
}

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_RankIdGap_Expect_Para)
{
    const std::string json = ReplaceOnce(FAST_RANK_TABLE_FULL, R"("rank_id": 1,)", R"("rank_id": 2,)");
    RankTableInfo info;
    EXPECT_EQ(ParseRankTableStringFast(json, info), HCCL_E_PARA);
}

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_PortOutOfRange_Expect_Para)
{
    RankTableInfo info;
    EXPECT_EQ(
        ParseRankTableStringFast(
            ReplaceOnce(FAST_RANK_TABLE_FULL, R"("device_port": 16666,)", R"("device_port": 0,)"), info),
        HCCL_E_PARA);
    EXPECT_EQ(
        ParseRankTableStringFast(
            ReplaceOnce(FAST_RANK_TABLE_FULL, R"("host_port": 16666,)", R"("host_port": 65536,)"), info),
        HCCL_E_PARA);
}

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_DevicePortPresent_Expect_AllAddrsUseIt)
{
    RankTableInfo info;
    ASSERT_EQ(ParseRankTableStringFast(FAST_RANK_TABLE_FULL, info), HCCL_SUCCESS);
    for (const auto& rank : info.ranks) {
        for (const auto& level : rank.rankLevelInfos) {
            for (const auto& addr : level.rankAddrs) {
                EXPECT_EQ(addr.socketPort_, rank.devicePort);
            }
        }
    }
}

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_BackupAddrBoundary_Expect_16Ok17Para)
{
    RankTableInfo info;
    EXPECT_EQ(ParseRankTableStringFast(MakeBackupAddrTable(16), info), HCCL_SUCCESS);
    EXPECT_EQ(ParseRankTableStringFast(MakeBackupAddrTable(17), info), HCCL_E_PARA);
}

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_InvalidNetType_Expect_Para)
{
    const std::string json = ReplaceOnce(FAST_RANK_TABLE_FULL, R"("net_type": "CLOS",)", R"("net_type": "BAD",)");
    RankTableInfo info;
    EXPECT_EQ(ParseRankTableStringFast(json, info), HCCL_E_PARA);
}

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_InvalidLevelOrder_Expect_Para)
{
    const std::string json = ReplaceOnce(FAST_RANK_TABLE_FULL, R"("net_layer": 1,)", R"("net_layer": 0,)");
    RankTableInfo info;
    EXPECT_EQ(ParseRankTableStringFast(json, info), HCCL_E_PARA);
}

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_InvalidAddress_Expect_Para)
{
    const std::string json = ReplaceOnce(FAST_RANK_TABLE_FULL, R"("addr": "10.1.0.1")", R"("addr": "999.1.1.1")");
    RankTableInfo info;
    EXPECT_EQ(ParseRankTableStringFast(json, info), HCCL_E_PARA);
}

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_InvalidDetour_Expect_Para)
{
    const std::string json = ReplaceOnce(FAST_RANK_TABLE_FULL, R"("detour": "false")", R"("detour": "maybe")");
    RankTableInfo info;
    EXPECT_EQ(ParseRankTableStringFast(json, info), HCCL_E_PARA);
}

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_JsonSyntaxInvalid_Expect_Para)
{
    RankTableInfo info;
    EXPECT_EQ(ParseRankTableStringFast(WithTrailingCommaInRankList(), info), HCCL_E_PARA);
    EXPECT_EQ(
        ParseRankTableStringFast(FAST_RANK_TABLE_FULL.substr(0, FAST_RANK_TABLE_FULL.size() / 2), info), HCCL_E_PARA);
    EXPECT_EQ(ParseRankTableStringFast(FAST_RANK_TABLE_FULL + "{}", info), HCCL_E_PARA);
}

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_UnknownFieldAndEscape_Expect_Success)
{
    std::string json
        = ReplaceOnce(FAST_RANK_TABLE_FULL, "{\n", "{\n  \"unknown_field\": {\"nested\": [1, true, null]},\n");
    json = ReplaceOnce(json, R"("net_attr": "",)", R"("net_attr": "a\"b\\c",)");

    RankTableInfo info;
    ASSERT_EQ(ParseRankTableStringFast(json, info), HCCL_SUCCESS);
    EXPECT_EQ(info.ranks[0].rankLevelInfos[0].netAttr, "a\"b\\c");
}

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_RankCountExceedsMax_Expect_Para)
{
    const std::string json = ReplaceOnce(
        FAST_RANK_TABLE_FULL, R"("rank_count": 2)", "\"rank_count\": " + std::to_string(MAX_RANKCOUNT + 1));
    RankTableInfo info;
    EXPECT_EQ(ParseRankTableStringFast(json, info), HCCL_E_PARA);
}

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_RankCountNonDecimal_Expect_Para)
{
    // ParseU32 有意收紧为纯十进制(与 nlohmann 对浮点/指数截断取整不同), 固化拒绝行为
    const std::string forms[] = {"-0", "1e0", "1.0", "166.66"};
    for (const auto& form : forms) {
        const std::string json = ReplaceOnce(FAST_RANK_TABLE_FULL, R"("rank_count": 2)", "\"rank_count\": " + form);
        RankTableInfo info;
        EXPECT_EQ(ParseRankTableStringFast(json, info), HCCL_E_PARA) << "form=" << form;
    }
}

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_BinaryRoundTrip_Expect_Equal)
{
    RankTableInfo info;
    ASSERT_EQ(ParseRankTableStringFast(FAST_RANK_TABLE_FULL, info), HCCL_SUCCESS);

    BinaryStream binStream;
    ASSERT_EQ(info.GetBinStream(true, binStream), HCCL_SUCCESS);
    RankTableInfo restored(binStream);

    EXPECT_EQ(restored.version, info.version);
    EXPECT_EQ(restored.rankCount, info.rankCount);
    ASSERT_EQ(restored.ranks.size(), info.ranks.size());
    ASSERT_EQ(restored.ranks.size(), 2U);
    EXPECT_EQ(restored.ranks[1].rankId, info.ranks[1].rankId);
    EXPECT_EQ(restored.ranks[1].devicePort, info.ranks[1].devicePort);
    EXPECT_EQ(restored.ranks[1].rankLevelInfos.size(), info.ranks[1].rankLevelInfos.size());
}

TEST_F(RankTableInfoParserTest, Ut_RankTableInfo_When_StreamRanksSizeOverLimit_Expect_ThrowInvalidParams)
{
    // 伪造流负例: ranksSize 声明超上限, 反序列化应抛 InvalidParamsException
    BinaryStream stream;
    stream << std::string("2.0");
    u32 rankCount = 1;
    stream << rankCount;
    const size_t forgedRanksSize = static_cast<size_t>(MAX_RANKCOUNT) + 1;
    stream << forgedRanksSize;
    EXPECT_THROW(RankTableInfo info(stream), InvalidParamsException);
}

TEST_F(RankTableInfoParserTest, Ut_RankTableInfo_When_StreamTruncated_Expect_ThrowInvalidParams)
{
    // 截断流负例: 声明 1 个 rank 但流已耗尽
    BinaryStream stream;
    stream << std::string("2.0");
    u32 rankCount = 1;
    stream << rankCount;
    const size_t ranksSize = 1;
    stream << ranksSize;
    EXPECT_THROW(RankTableInfo info(stream), InvalidParamsException);
}

TEST_F(RankTableInfoParserTest, Ut_ParseChangedRankInfoJson_When_ViaPublicApi_Expect_SameAsDirect)
{
    // ParseString<ChangedRankInfo> 头文件级特化已删除(ODR 风险), 对外统一走函数式入口
    ChangedRankInfo viaApi;
    ASSERT_EQ(ParseChangedRankInfoJson(FAST_CHANGED_RANK_INFO, viaApi), HCCL_SUCCESS);

    ChangedRankInfo direct;
    ASSERT_EQ(ParseChangedRankInfoStringFast(FAST_CHANGED_RANK_INFO, direct), HCCL_SUCCESS);

    EXPECT_EQ(viaApi.version, direct.version);
    EXPECT_EQ(viaApi.rankCount, direct.rankCount);
    ASSERT_EQ(viaApi.ranks.size(), direct.ranks.size());
    EXPECT_EQ(viaApi.ranks[0].rankId, direct.ranks[0].rankId);
}

TEST_F(RankTableInfoParserTest, Ut_ParseChangedRankInfoStringFast_When_MissingVersion_Expect_Para)
{
    const std::string json = ReplaceOnce(FAST_CHANGED_RANK_INFO, R"("version": "2.0",)", "");
    ChangedRankInfo info;
    EXPECT_EQ(ParseChangedRankInfoStringFast(json, info), HCCL_E_PARA);
}

TEST_F(RankTableInfoParserTest, Ut_ParseChangedRankInfoStringFast_When_RankCountOverRankTableMax_Expect_Success)
{
    // 与 DOM 一致: MAX_RANKCOUNT 上限仅约束 RankTableInfo, 变更 ranktable 不设此上限(内存仍由 SafeReserveCount 收敛)
    const std::string json = ReplaceOnce(FAST_CHANGED_RANK_INFO, R"("rank_count": 1,)", R"("rank_count": 20000000,)");
    ChangedRankInfo info;
    EXPECT_EQ(ParseChangedRankInfoStringFast(json, info), HCCL_SUCCESS);
    EXPECT_EQ(info.rankCount, 20000000U);
    ASSERT_EQ(info.ranks.size(), 1U);
}

namespace {

std::string MakeLargeRankTable(size_t rankCount, size_t paddingBytes)
{
    const std::string padding(paddingBytes, 'x');
    std::string json;
    json.reserve(rankCount * (paddingBytes + 256));
    json += R"({"version":"2.0","detour":"false","rank_count":)";
    json += std::to_string(rankCount);
    json += R"(,"rank_list":[)";
    for (size_t i = 0; i < rankCount; ++i) {
        if (i != 0) {
            json += ',';
        }
        const std::string id = std::to_string(i);
        json += R"({"rank_id":)" + id;
        json += R"(,"local_id":)" + std::to_string(i % 512);
        json += R"(,"device_id":)" + std::to_string(i % 512);
        json += R"(,"device_port":16666,"host_port":16666,"level_list":[{"net_layer":0,)";
        json += R"("net_instance_id":"sp_0_srv_)" + id;
        json += R"(","net_type":"TOPO_FILE_DESC","rank_addr_list":[{"addr_type":"IPV4",)";
        json += R"("addr":"10.)" + std::to_string(i / 250) + "." + std::to_string(i % 250);
        json += R"(.1","ports":["0/0"]}]}],"padding":")" + padding + R"("})";
    }
    json += "]}";
    return json;
}

std::string MakeRankTableWithAddrCount(size_t addrCount)
{
    std::string addrs;
    for (size_t i = 0; i < addrCount; ++i) {
        if (i != 0) {
            addrs += ',';
        }
        addrs += R"({"addr_type":"IPV4","addr":"10.0.0.1","ports":["0/0"]})";
    }
    return R"({"version":"2.0","rank_count":1,"rank_list":[{"rank_id":0,"local_id":0,"device_id":0,)"
           R"("level_list":[{"net_layer":0,"net_instance_id":"sp_0_srv_0","net_type":"TOPO_FILE_DESC",)"
           R"("rank_addr_list":[)"
           + addrs + R"(]}]}]})";
}

} // namespace

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_OutputInfoNotEmpty_Expect_Overwritten)
{
    RankTableInfo info;
    info.version = "old";
    info.rankCount = 99;
    info.ranks.resize(3);

    ASSERT_EQ(ParseRankTableStringFast(FAST_RANK_TABLE_FULL, info), HCCL_SUCCESS);
    EXPECT_EQ(info.version, "2.0");
    EXPECT_EQ(info.rankCount, 2U);
    ASSERT_EQ(info.ranks.size(), 2U);
    EXPECT_EQ(info.ranks[0].rankId, 0U);
    EXPECT_EQ(info.ranks[1].rankId, 1U);
}

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_ForgedLargeRankCount_Expect_Para)
{
    const std::string json = ReplaceOnce(FAST_RANK_TABLE_FULL, R"("rank_count": 2)", R"("rank_count": 1000000)");
    RankTableInfo info;
    EXPECT_EQ(ParseRankTableStringFast(json, info), HCCL_E_PARA);
}

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_UnknownFieldLiterals_Expect_Success)
{
    std::string json = FAST_RANK_TABLE_FULL;
    const size_t objectPos = json.find('{');
    ASSERT_NE(objectPos, std::string::npos);
    json.insert(objectPos + 1, R"("unknown_bool":true,"unknown_false":false,"unknown_null":null,)");
    RankTableInfo info;
    EXPECT_EQ(ParseRankTableStringFast(json, info), HCCL_SUCCESS);
}

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_StringEscapes_Expect_Decoded)
{
    const std::string json = ReplaceOnce(
        FAST_RANK_TABLE_FULL, R"("net_attr": "",)", R"("net_attr": "A\u00E9\u4e2d\uD83D\uDE00\/\b\f\n\r\t",)");
    RankTableInfo info;
    ASSERT_EQ(ParseRankTableStringFast(json, info), HCCL_SUCCESS);
    const std::string expected = std::string("A") + "\xC3\xA9" + "\xE4\xB8\xAD" + "\xF0\x9F\x98\x80" + "/\b\f\n\r\t";
    EXPECT_EQ(info.ranks[0].rankLevelInfos[0].netAttr, expected);
}

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_InvalidEscape_Expect_Para)
{
    RankTableInfo info;
    EXPECT_EQ(
        ParseRankTableStringFast(
            ReplaceOnce(FAST_RANK_TABLE_FULL, R"("net_attr": "",)", R"("net_attr": "A\x",)"), info),
        HCCL_E_PARA);
    EXPECT_EQ(
        ParseRankTableStringFast(
            ReplaceOnce(FAST_RANK_TABLE_FULL, R"("net_attr": "",)", R"("net_attr": "A\u12G4",)"), info),
        HCCL_E_PARA);
    const std::string rawControl
        = ReplaceOnce(FAST_RANK_TABLE_FULL, R"("net_attr": "",)", std::string("\"net_attr\": \"a\nb\","));
    EXPECT_EQ(ParseRankTableStringFast(rawControl, info), HCCL_E_PARA);
}

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_TopLevelTrailingComma_Expect_Para)
{
    std::string json = FAST_RANK_TABLE_FULL;
    const size_t pos = json.find_last_of('}');
    ASSERT_NE(pos, std::string::npos);
    json.insert(pos, ",");
    RankTableInfo info;
    EXPECT_EQ(ParseRankTableStringFast(json, info), HCCL_E_PARA);
}

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_RankCountLeadingZero_Expect_Para)
{
    const std::string json = ReplaceOnce(FAST_RANK_TABLE_FULL, R"("rank_count": 2)", R"("rank_count": 02)");
    RankTableInfo info;
    EXPECT_EQ(ParseRankTableStringFast(json, info), HCCL_E_PARA);
}

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_ControlPlaneMissingField_Expect_Para)
{
    const std::string json = ReplaceOnce(
        FAST_RANK_TABLE_FULL, R"("control_plane": {"addr_type": "IPV4", "addr": "10.0.0.1", "listen_port": 8000})",
        R"("control_plane": {"addr_type": "IPV4", "addr": "10.0.0.1"})");
    RankTableInfo info;
    EXPECT_EQ(ParseRankTableStringFast(json, info), HCCL_E_PARA);
}

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_PortsMissing_Expect_Para)
{
    const std::string json
        = ReplaceOnce(FAST_RANK_TABLE_FULL, R"("ports": ["0/0"], "plane_id": "plane0")", R"("plane_id": "plane0")");
    RankTableInfo info;
    EXPECT_EQ(ParseRankTableStringFast(json, info), HCCL_E_PARA);
}

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_RankAddrListMissing_Expect_Para)
{
    const std::string json
        = ReplaceOnce(FAST_RANK_TABLE_FULL, R"("rank_addr_list": [)", R"("rank_addr_list_removed": [)");
    RankTableInfo info;
    EXPECT_EQ(ParseRankTableStringFast(json, info), HCCL_E_PARA);
}

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_RankAddrCountBoundary_Expect_24Ok25Para)
{
    RankTableInfo info;
    EXPECT_EQ(ParseRankTableStringFast(MakeRankTableWithAddrCount(24), info), HCCL_SUCCESS);
    EXPECT_EQ(ParseRankTableStringFast(MakeRankTableWithAddrCount(25), info), HCCL_E_PARA);
}

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_ParallelLargeTable_Expect_Success)
{
    const std::string json = MakeLargeRankTable(4096, 5120);
    ASSERT_GT(json.size(), 16U * 1024U * 1024U);

    RankTableInfo info;
    ASSERT_EQ(ParseRankTableStringFast(json, info), HCCL_SUCCESS);
    EXPECT_EQ(info.rankCount, 4096U);
    ASSERT_EQ(info.ranks.size(), 4096U);
    EXPECT_EQ(info.ranks.front().rankId, 0U);
    EXPECT_EQ(info.ranks[2048].rankId, 2048U);
    EXPECT_EQ(info.ranks.back().rankId, 4095U);
    EXPECT_EQ(info.ranks[2048].rankLevelInfos.size(), 1U);
}

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_ParallelLargeTrailingComma_Expect_Para)
{
    std::string json = MakeLargeRankTable(4096, 5120);
    const size_t pos = json.find_last_of('}');
    ASSERT_NE(pos, std::string::npos);
    json.insert(pos, ",");

    RankTableInfo info;
    EXPECT_EQ(ParseRankTableStringFast(json, info), HCCL_E_PARA);
}

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_ParallelLargeLeadingZero_Expect_Para)
{
    std::string json = MakeLargeRankTable(4096, 5120);
    json = ReplaceOnce(json, R"("rank_count":4096)", R"("rank_count":04096)");

    RankTableInfo info;
    EXPECT_EQ(ParseRankTableStringFast(json, info), HCCL_E_PARA);
}

TEST_F(RankTableInfoParserTest, Ut_FindQuoteOrBackslash_When_VariousPositions_Expect_FirstMatch)
{
    const std::vector<std::string> cases = {
        "",
        "abc",
        "\"",
        "\\",
        "a\"b",
        "a\\b",
        "0123456789abcdef\"",
        "0123456789abcdef\\",
        "0123456789abcdefg\"",
        std::string(31, 'x') + "\"",
        std::string(31, 'x') + "\\",
    };
    for (const auto& text : cases) {
        const char* begin = text.data();
        const char* end = begin + text.size();
        const char* expected = end;
        for (const char* p = begin; p < end; ++p) {
            if (*p == '"' || *p == '\\') {
                expected = p;
                break;
            }
        }
        EXPECT_EQ(FindQuoteOrBackslash(begin, end), expected);
    }
}

TEST_F(RankTableInfoParserTest, Ut_FindStructuralChar_When_VariousPositions_Expect_FirstMatch)
{
    const std::vector<std::string> cases = {
        "",
        "abc",
        "{",
        "}",
        "\"",
        "\\",
        "a{b",
        "a}b",
        "a\"b",
        "a\\b",
        "0123456789abcdef{",
        "0123456789abcdef}",
        "0123456789abcdef\"",
        "0123456789abcdef\\",
        "0123456789abcdefg}",
        std::string(31, 'x') + "{",
        std::string(31, 'x') + "}",
        std::string(31, 'x') + "\"",
        std::string(31, 'x') + "\\",
    };
    for (const auto& text : cases) {
        const char* begin = text.data();
        const char* end = begin + text.size();
        const char* expected = end;
        for (const char* p = begin; p < end; ++p) {
            if (*p == '{' || *p == '}' || *p == '"' || *p == '\\') {
                expected = p;
                break;
            }
        }
        EXPECT_EQ(FindStructuralChar(begin, end), expected);
    }
}

namespace {

// 大表生成器(触发并行布局扫描路径): padding 字符串与元素间分隔可定制
std::string MakeLargeRankTableCustom(size_t rankCount, const std::string& paddingValue, const std::string& sep)
{
    std::string json;
    json.reserve(rankCount * (paddingValue.size() + 256));
    json += R"({"version":"2.0","detour":"false","rank_count":)";
    json += std::to_string(rankCount);
    json += R"(,"rank_list":[)";
    for (size_t i = 0; i < rankCount; ++i) {
        if (i != 0) {
            json += ',';
            json += sep;
        }
        const std::string id = std::to_string(i);
        json += R"({"rank_id":)" + id;
        json += R"(,"local_id":)" + std::to_string(i % 512);
        json += R"(,"device_id":)" + std::to_string(i % 512);
        json += R"(,"device_port":16666,"host_port":16666,"level_list":[{"net_layer":0,)";
        json += R"("net_instance_id":"sp_0_srv_)" + id;
        json += R"(","net_type":"TOPO_FILE_DESC","rank_addr_list":[{"addr_type":"IPV4",)";
        json += R"("addr":"10.)" + std::to_string(i / 250) + "." + std::to_string(i % 250);
        json += R"(.1","ports":["0/0"]}]}],"padding":")" + paddingValue + R"("})";
    }
    json += "]}";
    return json;
}

} // namespace

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_ParallelLargeStringStructuralChars_Expect_Success)
{
    // padding 字符串值包含未转义的 { } 以及 \" \\ \uXXXX: 字符串内结构字符不得影响元素边界
    const std::string padding = std::string(2048, 'x') + R"(xx}yy{zz\"ww\\q\u4e2d)";
    const std::string json = MakeLargeRankTableCustom(4096, padding, "");
    ASSERT_GT(json.size(), 8U * 1024U * 1024U);

    RankTableInfo info;
    ASSERT_EQ(ParseRankTableStringFast(json, info), HCCL_SUCCESS);
    EXPECT_EQ(info.rankCount, 4096U);
    ASSERT_EQ(info.ranks.size(), 4096U);
    EXPECT_EQ(info.ranks.front().rankId, 0U);
    EXPECT_EQ(info.ranks[2048].rankId, 2048U);
    EXPECT_EQ(info.ranks.back().rankId, 4095U);
    ASSERT_EQ(info.ranks[2048].rankLevelInfos.size(), 1U);
}

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_ParallelLargeWhitespace_Expect_Success)
{
    // 元素之间插入空格/换行, 走 SkipWs 后进入 SIMD 元素扫描
    const std::string json = MakeLargeRankTableCustom(4096, std::string(2048, 'x'), "\n    ");
    ASSERT_GT(json.size(), 8U * 1024U * 1024U);

    RankTableInfo info;
    ASSERT_EQ(ParseRankTableStringFast(json, info), HCCL_SUCCESS);
    EXPECT_EQ(info.rankCount, 4096U);
    ASSERT_EQ(info.ranks.size(), 4096U);
}

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_ParallelLargeNestedUnknownObject_Expect_Success)
{
    // 元素内注入深度嵌套的未知对象, depth 计数需跨嵌套正确归零
    std::string json = MakeLargeRankTable(4096, 5120);
    json = ReplaceOnce(json, R"({"rank_id":0,)", R"({"unknown":{"a":{"b":{"c":[1,2,{"d":{}}]}}},"rank_id":0,)");
    ASSERT_GT(json.size(), 8U * 1024U * 1024U);

    RankTableInfo info;
    ASSERT_EQ(ParseRankTableStringFast(json, info), HCCL_SUCCESS);
    EXPECT_EQ(info.rankCount, 4096U);
    ASSERT_EQ(info.ranks.size(), 4096U);
    EXPECT_EQ(info.ranks.front().rankId, 0U);
}

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_ParallelLargeUnclosedElement_Expect_Para)
{
    // 删除最后一个元素的闭合 '}': 布局扫描 depth 无法归零, 拒绝该表
    std::string json = MakeLargeRankTable(4096, 5120);
    const size_t pos = json.rfind("}]}");
    ASSERT_NE(pos, std::string::npos);
    json.erase(pos, 1);

    RankTableInfo info;
    EXPECT_EQ(ParseRankTableStringFast(json, info), HCCL_E_PARA);
}

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_ParallelLargeTrailingContent_Expect_Para)
{
    std::string json = MakeLargeRankTable(4096, 5120);
    json += "garbage";

    RankTableInfo info;
    EXPECT_EQ(ParseRankTableStringFast(json, info), HCCL_E_PARA);
}

namespace {

std::string MakeRankTableWithPortCount(size_t portCount)
{
    std::string ports;
    for (size_t i = 0; i < portCount; ++i) {
        if (i != 0) {
            ports += ',';
        }
        ports += "\"p" + std::to_string(i) + "\"";
    }
    return R"({"version":"2.0","rank_count":1,"rank_list":[{"rank_id":0,"local_id":0,"device_id":0,)"
           R"("level_list":[{"net_layer":0,"net_instance_id":"sp_0_srv_0","net_type":"TOPO_FILE_DESC",)"
           R"("rank_addr_list":[{"addr_type":"IPV4","addr":"10.0.0.1","ports":[)"
           + ports + R"(]}]}]}]})";
}

std::string MakeRankTableWithLevelCount(size_t levelCount)
{
    std::string levels;
    for (size_t i = 0; i < levelCount; ++i) {
        if (i != 0) {
            levels += ',';
        }
        levels += R"({"net_layer":)" + std::to_string(i);
        levels += R"(,"net_instance_id":"sp_)" + std::to_string(i) + R"(_srv_0",)";
        levels += R"("net_type":"TOPO_FILE_DESC","rank_addr_list":[{"addr_type":"IPV4",)";
        levels += R"("addr":"10.0.0.1","ports":["0/0"]}]})";
    }
    return R"({"version":"2.0","rank_count":1,"rank_list":[{"rank_id":0,"local_id":0,"device_id":0,)"
           R"("level_list":[)"
           + levels + R"(]}]})";
}

} // namespace

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_AddressRequiredFieldMissing_Expect_Para)
{
    RankTableInfo info;
    EXPECT_EQ(
        ParseRankTableStringFast(ReplaceOnce(FAST_RANK_TABLE_FULL, R"("addr": "10.0.0.1", )", ""), info), HCCL_E_PARA);
    EXPECT_EQ(
        ParseRankTableStringFast(ReplaceOnce(FAST_RANK_TABLE_FULL, R"("addr_type": "IPV4", )", ""), info), HCCL_E_PARA);
}

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_LevelRequiredFieldMissing_Expect_Para)
{
    RankTableInfo info;
    EXPECT_EQ(ParseRankTableStringFast(ReplaceOnce(FAST_RANK_TABLE_FULL, R"("net_layer": 0,)", ""), info), HCCL_E_PARA);
    EXPECT_EQ(
        ParseRankTableStringFast(ReplaceOnce(FAST_RANK_TABLE_FULL, R"("net_instance_id": "sp_0_srv_0",)", ""), info),
        HCCL_E_PARA);
}

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_PortsBoundary_Expect_ParaAnd20Ok)
{
    RankTableInfo info;
    EXPECT_EQ(
        ParseRankTableStringFast(ReplaceOnce(FAST_RANK_TABLE_FULL, R"("ports": ["0/0"])", R"("ports": [])"), info),
        HCCL_E_PARA);
    EXPECT_EQ(
        ParseRankTableStringFast(
            ReplaceOnce(
                FAST_RANK_TABLE_FULL, R"("ports": ["0/0"])", R"("ports": ["0123456789012345678901234567890123"])"),
            info),
        HCCL_E_PARA);
    EXPECT_EQ(ParseRankTableStringFast(MakeRankTableWithPortCount(20), info), HCCL_SUCCESS);
    EXPECT_EQ(ParseRankTableStringFast(MakeRankTableWithPortCount(21), info), HCCL_E_PARA);
}

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_LevelCountBoundary_Expect_8Ok9Para)
{
    RankTableInfo info;
    EXPECT_EQ(ParseRankTableStringFast(MakeRankTableWithLevelCount(8), info), HCCL_SUCCESS);
    EXPECT_EQ(ParseRankTableStringFast(MakeRankTableWithLevelCount(9), info), HCCL_E_PARA);
}

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_IdentityOutOfRange_Expect_Para)
{
    RankTableInfo info;
    EXPECT_EQ(
        ParseRankTableStringFast(ReplaceOnce(FAST_RANK_TABLE_FULL, R"("device_id": 0,)", R"("device_id": 513,)"), info),
        HCCL_E_PARA);
    EXPECT_EQ(
        ParseRankTableStringFast(ReplaceOnce(FAST_RANK_TABLE_FULL, R"("local_id": 0,)", R"("local_id": 513,)"), info),
        HCCL_E_PARA);
}

namespace {

std::string gFastReportCode;
std::vector<std::string> gFastReportKeys;
std::vector<std::string> gFastReportValues;

void CaptureFastParserReport(std::string errorCode, std::vector<std::string> keys, std::vector<std::string> values)
{
    gFastReportCode = std::move(errorCode);
    gFastReportKeys = std::move(keys);
    gFastReportValues = std::move(values);
}

} // namespace

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_FieldErrors_Expect_ReportEI0017AndEI0014)
{
    gFastReportCode.clear();
    gFastReportKeys.clear();
    gFastReportValues.clear();
    MOCKER(RptInputErr).stubs().will(invoke(CaptureFastParserReport));

    // 必选字段缺失: EI0017, keys={config}, values 仅含字段名
    RankTableInfo info;
    EXPECT_EQ(ParseRankTableStringFast(ReplaceOnce(FAST_RANK_TABLE_FULL, R"("rank_id": 0,)", ""), info), HCCL_E_PARA);
    EXPECT_EQ(gFastReportCode, "EI0017");
    EXPECT_EQ(gFastReportKeys, std::vector<std::string>({"config"}));
    EXPECT_EQ(gFastReportValues, std::vector<std::string>({"rank_id"}));

    // level_list 缺失同样报 EI0017
    EXPECT_EQ(ParseRankTableStringFast(WithoutLevelList(), info), HCCL_E_PARA);
    EXPECT_EQ(gFastReportCode, "EI0017");
    EXPECT_EQ(gFastReportValues, std::vector<std::string>({"level_list"}));

    // control_plane 内必选字段缺失: 报 EI0017 且带出字段名
    EXPECT_EQ(
        ParseRankTableStringFast(
            ReplaceOnce(
                FAST_RANK_TABLE_FULL,
                R"("control_plane": {"addr_type": "IPV4", "addr": "10.0.0.1", "listen_port": 8000})",
                R"("control_plane": {"addr_type": "IPV4", "listen_port": 8000})"),
            info),
        HCCL_E_PARA);
    EXPECT_EQ(gFastReportCode, "EI0017");
    EXPECT_EQ(gFastReportValues, std::vector<std::string>({"addr"}));

    // 字段取值非法: EI0014, keys={value, variable, expect}
    EXPECT_EQ(
        ParseRankTableStringFast(ReplaceOnce(FAST_RANK_TABLE_FULL, R"("device_id": 0,)", R"("device_id": 513,)"), info),
        HCCL_E_PARA);
    EXPECT_EQ(gFastReportCode, "EI0014");
    EXPECT_EQ(gFastReportKeys, std::vector<std::string>({"value", "variable", "expect"}));
    ASSERT_EQ(gFastReportValues.size(), 3U);
    EXPECT_EQ(gFastReportValues[0], "513");
    EXPECT_EQ(gFastReportValues[1], "device_id");
}

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_DuplicatePorts_Expect_LastWins)
{
    // DOM 语义: 同一对象内重复键后者覆盖前者, 快速解析需保持一致
    const std::string json = R"({"version":"2.0","rank_count":1,"rank_list":[{"rank_id":0,"local_id":0,)"
                             R"("device_id":0,"device_port":16666,"host_port":16666,"level_list":[{"net_layer":0,)"
                             R"("net_instance_id":"sp_0_srv_0","net_type":"TOPO_FILE_DESC","rank_addr_list":)"
                             R"([{"addr_type":"IPV4","addr":"10.0.0.1","ports":["9/9"],"ports":["0/0"]}]}]}]})";
    RankTableInfo info;
    ASSERT_EQ(ParseRankTableStringFast(json, info), HCCL_SUCCESS);
    EXPECT_EQ(info.ranks[0].rankLevelInfos[0].rankAddrs[0].ports, std::set<std::string>({"0/0"}));
}

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_DuplicateBackupAddr_Expect_LastWins)
{
    const std::string json = R"({"version":"2.0","rank_count":1,"rank_list":[{"rank_id":0,"local_id":0,)"
                             R"("device_id":0,"device_port":16666,"host_port":16666,"level_list":[{"net_layer":0,)"
                             R"("net_instance_id":"sp_0_srv_0","net_type":"TOPO_FILE_DESC","rank_addr_list":)"
                             R"([{"addr_type":"IPV4","addr":"10.0.0.1","ports":["0/0"],)"
                             R"("backup_addr":["10.0.9.1"],"backup_addr":["10.0.9.2"]}]}]}]})";
    RankTableInfo info;
    ASSERT_EQ(ParseRankTableStringFast(json, info), HCCL_SUCCESS);
    const auto& backupAddrs = info.ranks[0].rankLevelInfos[0].rankAddrs[0].backupAddrs;
    ASSERT_EQ(backupAddrs.size(), 1U);
    EXPECT_EQ(backupAddrs[0], IpAddress("10.0.9.2", AF_INET));
}

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_DuplicateRankAddrList_Expect_LastWins)
{
    const std::string json = R"({"version":"2.0","rank_count":1,"rank_list":[{"rank_id":0,"local_id":0,)"
                             R"("device_id":0,"device_port":16666,"host_port":16666,"level_list":[{"net_layer":0,)"
                             R"("net_instance_id":"sp_0_srv_0","net_type":"TOPO_FILE_DESC",)"
                             R"("rank_addr_list":[{"addr_type":"IPV4","addr":"10.0.0.1","ports":["0/0"]}],)"
                             R"("rank_addr_list":[{"addr_type":"IPV4","addr":"10.0.0.2","ports":["0/0"]}]}]}]})";
    RankTableInfo info;
    ASSERT_EQ(ParseRankTableStringFast(json, info), HCCL_SUCCESS);
    const auto& rankAddrs = info.ranks[0].rankLevelInfos[0].rankAddrs;
    ASSERT_EQ(rankAddrs.size(), 1U);
    EXPECT_EQ(rankAddrs[0].addr, IpAddress("10.0.0.2", AF_INET));
}

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_DuplicateLevelList_Expect_LastWins)
{
    const std::string json
        = R"({"version":"2.0","rank_count":1,"rank_list":[{"rank_id":0,"local_id":0,)"
          R"("device_id":0,"device_port":16666,"host_port":16666,)"
          R"("level_list":[{"net_layer":0,"net_instance_id":"sp_0_srv_0","net_type":"TOPO_FILE_DESC",)"
          R"("rank_addr_list":[{"addr_type":"IPV4","addr":"10.0.0.1","ports":["0/0"]}]}],)"
          R"("level_list":[{"net_layer":5,"net_instance_id":"sp_9_srv_0","net_type":"TOPO_FILE_DESC",)"
          R"("rank_addr_list":[{"addr_type":"IPV4","addr":"10.0.0.2","ports":["0/0"]}]}]}]})";
    RankTableInfo info;
    ASSERT_EQ(ParseRankTableStringFast(json, info), HCCL_SUCCESS);
    ASSERT_EQ(info.ranks[0].rankLevelInfos.size(), 1U);
    EXPECT_EQ(info.ranks[0].rankLevelInfos[0].netLayer, 5U);
    EXPECT_EQ(info.ranks[0].rankLevelInfos[0].rankAddrs[0].addr, IpAddress("10.0.0.2", AF_INET));
}

namespace {

// 生成 rank_list 数组: rankIdOffset 区分两段重复 rank_list 的元素(rank_id 不相交)
std::string MakeRankListArray(size_t count, size_t paddingBytes, size_t rankIdOffset)
{
    const std::string padding(paddingBytes, 'x');
    std::string list = "[";
    for (size_t i = 0; i < count; ++i) {
        if (i != 0) {
            list += ',';
        }
        const std::string id = std::to_string(i + rankIdOffset);
        list += R"({"rank_id":)" + id;
        list += R"(,"local_id":)" + std::to_string((i + rankIdOffset) % 512);
        list += R"(,"device_id":)" + std::to_string((i + rankIdOffset) % 512);
        list += R"(,"device_port":16666,"host_port":16666,"level_list":[{"net_layer":0,)";
        list += R"("net_instance_id":"sp_0_srv_)" + id + R"(","net_type":"TOPO_FILE_DESC",)";
        list += R"("rank_addr_list":[{"addr_type":"IPV4","addr":")";
        list += "10." + std::to_string(i / 250) + "." + std::to_string(i % 250) + R"(.1","ports":["0/0"])";
        list += R"(}]}],"padding":")" + padding + R"("})";
    }
    list += ']';
    return list;
}

} // namespace

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_ParallelDuplicateRankList_Expect_LastWins)
{
    // 顶层重复 rank_list, 后段为大表(>8MB)触发并行布局扫描: 只统计最后一段, 与串行/DOM一致
    const std::string first = MakeRankListArray(1024, 0, 3000);
    const std::string second = MakeRankListArray(2048, 4096, 0);
    const std::string json = R"({"version":"2.0","detour":"false","rank_count":2048,"rank_list":)" + first
                             + R"(,"rank_list":)" + second + "}";
    ASSERT_GT(json.size(), 8U * 1024U * 1024U);

    RankTableInfo info;
    ASSERT_EQ(ParseRankTableStringFast(json, info), HCCL_SUCCESS);
    EXPECT_EQ(info.rankCount, 2048U);
    ASSERT_EQ(info.ranks.size(), 2048U);
    EXPECT_EQ(info.ranks[0].rankId, 0U);
    EXPECT_EQ(info.ranks[2047].rankId, 2047U);
}

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_Utf8BomPrefix_Expect_Skipped)
{
    const std::string bom = "\xEF\xBB\xBF";
    RankTableInfo info;
    // 单个 BOM 与 DOM(nlohmann skip_bom)一致跳过
    ASSERT_EQ(ParseRankTableStringFast(bom + FAST_RANK_TABLE_FULL, info), HCCL_SUCCESS);
    EXPECT_EQ(info.rankCount, 2U);
    // 连续两个 BOM: DOM 只跳过第一个, 第二个按非法输入拒绝
    EXPECT_EQ(ParseRankTableStringFast(bom + bom + FAST_RANK_TABLE_FULL, info), HCCL_E_PARA);
    // 仅 BOM: 去除后为空输入
    EXPECT_EQ(ParseRankTableStringFast(bom, info), HCCL_E_PARA);
    // 变更 ranktable 入口同样跳过 BOM
    ChangedRankInfo changed;
    ASSERT_EQ(ParseChangedRankInfoStringFast(bom + FAST_CHANGED_RANK_INFO, changed), HCCL_SUCCESS);
    EXPECT_EQ(changed.rankCount, 1U);
}

namespace {

// RAII: 用例结束(含断言失败提前返回)后自动清理强制线程数环境变量
class ForceParseThreadsGuard {
public:
    explicit ForceParseThreadsGuard(const char* value) { setenv("HCCL_FAST_PARSER_THREADS", value, 1); }

    ~ForceParseThreadsGuard() { unsetenv("HCCL_FAST_PARSER_THREADS"); }

    ForceParseThreadsGuard(const ForceParseThreadsGuard&) = delete;
    ForceParseThreadsGuard& operator=(const ForceParseThreadsGuard&) = delete;
};

} // namespace

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_ForcedParallel_Expect_PathTakenAndSerialEqual)
{
    const std::string json = MakeLargeRankTable(2048, 2048);
    ASSERT_GT(json.size(), 4U * 1024U * 1024U);

    // 强制 2 线程: 并行路径必须被触发(累计计数递增)
    ForceParseThreadsGuard guard("2");
    const u64 before = GetFastParserParallelUsedForTest();
    RankTableInfo parallelInfo;
    ASSERT_EQ(ParseRankTableStringFast(json, parallelInfo), HCCL_SUCCESS);
    EXPECT_GT(GetFastParserParallelUsedForTest(), before);
    EXPECT_EQ(parallelInfo.rankCount, 2048U);
    ASSERT_EQ(parallelInfo.ranks.size(), 2048U);

    // 强制 1 线程(串行): 同输入逐字段与并行结果一致
    ForceParseThreadsGuard serialGuard("1");
    RankTableInfo serialInfo;
    ASSERT_EQ(ParseRankTableStringFast(json, serialInfo), HCCL_SUCCESS);
    ASSERT_EQ(serialInfo.ranks.size(), parallelInfo.ranks.size());
    for (size_t i = 0; i < serialInfo.ranks.size(); ++i) {
        EXPECT_EQ(parallelInfo.ranks[i].rankId, serialInfo.ranks[i].rankId);
        EXPECT_EQ(parallelInfo.ranks[i].localId, serialInfo.ranks[i].localId);
        EXPECT_EQ(parallelInfo.ranks[i].deviceId, serialInfo.ranks[i].deviceId);
        ASSERT_EQ(parallelInfo.ranks[i].rankLevelInfos.size(), serialInfo.ranks[i].rankLevelInfos.size());
        for (size_t level = 0; level < serialInfo.ranks[i].rankLevelInfos.size(); ++level) {
            EXPECT_EQ(
                parallelInfo.ranks[i].rankLevelInfos[level].netLayer,
                serialInfo.ranks[i].rankLevelInfos[level].netLayer);
            EXPECT_EQ(
                parallelInfo.ranks[i].rankLevelInfos[level].netInstId,
                serialInfo.ranks[i].rankLevelInfos[level].netInstId);
            ASSERT_EQ(
                parallelInfo.ranks[i].rankLevelInfos[level].rankAddrs.size(),
                serialInfo.ranks[i].rankLevelInfos[level].rankAddrs.size());
            EXPECT_EQ(
                parallelInfo.ranks[i].rankLevelInfos[level].rankAddrs[0].addr,
                serialInfo.ranks[i].rankLevelInfos[level].rankAddrs[0].addr);
            EXPECT_EQ(
                parallelInfo.ranks[i].rankLevelInfos[level].rankAddrs[0].ports,
                serialInfo.ranks[i].rankLevelInfos[level].rankAddrs[0].ports);
        }
    }
}

TEST_F(RankTableInfoParserTest, Ut_ParseRankTableStringFast_When_ForcedParallelDuplicateRankList_Expect_LastWins)
{
    // 前段小表 + 后段大表重复 rank_list 键: 并行路径直接 last-wins 成功, 不依赖串行回退
    const std::string first = MakeRankListArray(1024, 0, 3000);
    const std::string second = MakeRankListArray(1280, 4096, 0);
    const std::string json = R"({"version":"2.0","detour":"false","rank_count":1280,"rank_list":)" + first
                             + R"(,"rank_list":)" + second + "}";
    ASSERT_GT(json.size(), 4U * 1024U * 1024U);

    ForceParseThreadsGuard guard("2");
    const u64 before = GetFastParserParallelUsedForTest();
    RankTableInfo info;
    const HcclResult ret = ParseRankTableStringFast(json, info);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    // 计数递增说明并行路径直接处理重复键成功, 未回退串行
    EXPECT_GT(GetFastParserParallelUsedForTest(), before);
    EXPECT_EQ(info.rankCount, 1280U);
    ASSERT_EQ(info.ranks.size(), 1280U);
    EXPECT_EQ(info.ranks[0].rankId, 0U);
    EXPECT_EQ(info.ranks[1279].rankId, 1279U);
}
