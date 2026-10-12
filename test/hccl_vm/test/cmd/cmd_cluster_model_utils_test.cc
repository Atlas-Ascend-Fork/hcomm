/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <vector>

#include "cmd_cluster_model_utils.h"
#include "sim_common_api.h"

using namespace HcclSim;

namespace {
class EnvGuard {
public:
    EnvGuard() = default;
    ~EnvGuard() {}
};
} // namespace

class ClearHvmModelEnvTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

class ParseYamlTopoTest : public testing::Test {
protected:
    std::string modelDir_;
    void SetUp() override
    {
        modelDir_ = InstallPath::ResolveToInstallRoot("config/topo_meta");
        std::filesystem::create_directories(modelDir_);
    }
    void TearDown() override {}
    std::string GetBinLocation()
    {
        std::error_code ec;
        auto exePath = std::filesystem::read_symlink("/proc/self/exe", ec);
        if (ec) {
            return ".";
        }
        return exePath.parent_path().string();
    }
    void WriteYaml(const std::string& name, const std::string& content)
    {
        std::string path = modelDir_ + "/" + name + ".yaml";
        std::ofstream ofs(path);
        ofs << content;
        ofs.close();
    }
    void RemoveYaml(const std::string& name)
    {
        std::string path = modelDir_ + "/" + name + ".yaml";
        std::filesystem::remove(path);
    }
};

TEST_F(ParseYamlTopoTest, Parse112WithModelMeta)
{
    TopoMeta topo;

    bool result = ParseYamlTopo("112", topo);
    EXPECT_TRUE(result);
    EXPECT_EQ(topo.size(), 1u);
}

TEST_F(ParseYamlTopoTest, Parse118WithModelMeta)
{
    TopoMeta topo;

    bool result = ParseYamlTopo("118", topo);
    EXPECT_TRUE(result);
    EXPECT_EQ(topo.size(), 1u);
}

TEST_F(ParseYamlTopoTest, ParseWithoutModelMeta)
{
    TopoMeta topo;

    bool result = ParseYamlTopo("112", topo);
    EXPECT_TRUE(result);
    EXPECT_EQ(topo.size(), 1u);
}

TEST_F(ParseYamlTopoTest, ParseWithoutSocVersion)
{
    WriteYaml("ut_wrong_soc_meta", R"(
meta:
  podNum: 1
  serNum: 1
  rankNum: 2
topology:
  - podId: 0
    servers:
      - serId: 0
        ip: 192.1.5.5
        ranks: [0, 1]
)");
    TopoMeta topo;

    bool result = ParseYamlTopo("ut_wrong_soc_meta", topo);
    EXPECT_TRUE(result);
    RemoveYaml("ut_wrong_soc_meta");
}

TEST_F(ParseYamlTopoTest, Ascend950WithModelMeta)
{
    WriteYaml("ut_950_meta", R"(
meta:
  podNum: 1
  serNum: 1
  rankNum: 4
topology:
  - podId: 0
    servers:
      - serId: 0
        ip: 192.1.5.5
        ranks: [0, 1, 2, 3]
)");
    TopoMeta topo;

    bool result = ParseYamlTopo("ut_950_meta", topo);
    EXPECT_TRUE(result);
    EXPECT_EQ(topo.size(), 1u);
    RemoveYaml("ut_950_meta");
}

TEST_F(ParseYamlTopoTest, NonUniformWithModelMeta)
{
    WriteYaml("ut_non_uniform_meta", R"(
meta:
  podNum: 1
  serNum: 2
  rankNum: 3
topology:
  - podId: 0
    servers:
      - serId: 0
        ip: 192.1.5.5
        ranks: [0, 1]
      - serId: 1
        ip: 192.2.5.5
        ranks: [2]
)");
    TopoMeta topo;

    bool result = ParseYamlTopo("ut_non_uniform_meta", topo);
    EXPECT_TRUE(result);
    EXPECT_EQ(topo.size(), 1u);
    RemoveYaml("ut_non_uniform_meta");
}

TEST_F(ParseYamlTopoTest, ParseWithServerCountMismatch)
{
    WriteYaml("ut_server_mismatch_meta", R"(
meta:
  podNum: 1
  serNum: 2
  rankNum: 2
topology:
  - podId: 0
    servers:
      - serId: 0
        ip: 192.1.5.5
        ranks: [0, 1]
)");
    TopoMeta topo;

    bool result = ParseYamlTopo("ut_server_mismatch_meta", topo);
    EXPECT_TRUE(result);
    RemoveYaml("ut_server_mismatch_meta");
}

TEST_F(ParseYamlTopoTest, ZeroPodNumYaml)
{
    WriteYaml("ut_zero_pod", R"(
meta:
  podNum: 0
  serNum: 1
  rankNum: 2
topology:
  - podId: 0
    servers:
      - serId: 0
        ip: 192.1.5.5
        ranks: [0, 1]
)");
    TopoMeta topo;
    EXPECT_FALSE(ParseYamlTopo("ut_zero_pod", topo));
    RemoveYaml("ut_zero_pod");
}

TEST_F(ParseYamlTopoTest, ZeroSerNumYaml)
{
    WriteYaml("ut_zero_ser", R"(
meta:
  podNum: 1
  serNum: 0
  rankNum: 2
topology:
  - podId: 0
    servers:
      - serId: 0
        ip: 192.1.5.5
        ranks: [0, 1]
)");
    TopoMeta topo;
    EXPECT_FALSE(ParseYamlTopo("ut_zero_ser", topo));
    RemoveYaml("ut_zero_ser");
}

TEST_F(ParseYamlTopoTest, ZeroRankNumYaml)
{
    WriteYaml("ut_zero_rank", R"(
meta:
  podNum: 1
  serNum: 1
  rankNum: 0
topology:
  - podId: 0
    servers:
      - serId: 0
        ip: 192.1.5.5
        ranks: [0, 1]
)");
    TopoMeta topo;
    EXPECT_FALSE(ParseYamlTopo("ut_zero_rank", topo));
    RemoveYaml("ut_zero_rank");
}

TEST_F(ParseYamlTopoTest, ParseWithoutTopology)
{
    WriteYaml("ut_no_topo_nometa", R"(
meta:
  podNum: 1
  serNum: 1
  rankNum: 2
)");
    TopoMeta topo;
    EXPECT_TRUE(ParseYamlTopo("ut_no_topo_nometa", topo));
    RemoveYaml("ut_no_topo_nometa");
}

TEST_F(ParseYamlTopoTest, ParseEmptyTopology)
{
    WriteYaml("ut_empty_topo", R"(
meta:
  podNum: 1
  serNum: 1
  rankNum: 2
topology: []
)");
    TopoMeta topo;

    bool result = ParseYamlTopo("ut_empty_topo", topo);
    EXPECT_TRUE(result);
    RemoveYaml("ut_empty_topo");
}

TEST_F(ParseYamlTopoTest, ParsePodWithoutServers)
{
    WriteYaml("ut_no_servers", R"(
meta:
  podNum: 1
  serNum: 1
  rankNum: 2
topology:
  - podId: 0
)");
    TopoMeta topo;

    bool result = ParseYamlTopo("ut_no_servers", topo);
    EXPECT_TRUE(result);
    RemoveYaml("ut_no_servers");
}

TEST_F(ParseYamlTopoTest, ParseServerWithoutRanks)
{
    WriteYaml("ut_no_ranks", R"(
meta:
  podNum: 1
  serNum: 1
  rankNum: 2
topology:
  - podId: 0
    servers:
      - serId: 0
        ip: 192.1.5.5
)");
    TopoMeta topo;

    bool result = ParseYamlTopo("ut_no_ranks", topo);
    EXPECT_TRUE(result);
    RemoveYaml("ut_no_ranks");
}

TEST_F(ParseYamlTopoTest, CaseInsensitiveSocVersion)
{
    WriteYaml("ut_upper_soc", R"(
meta:
  podNum: 1
  serNum: 1
  rankNum: 2
topology:
  - podId: 0
    servers:
      - serId: 0
        ip: 192.1.5.5
        ranks: [0, 1]
)");
    TopoMeta topo;

    EXPECT_TRUE(ParseYamlTopo("ut_upper_soc", topo));
    RemoveYaml("ut_upper_soc");
}

TEST_F(ParseYamlTopoTest, ParseCompactRangeFormat)
{
    WriteYaml("ut_range_compact", R"(
meta:
  podNum: 1
  serNum: 4
  rankNum: 32
topology:
  - podId: 0
    servers:
      - serId_range: [0, 3]
        ranks_range: [0, 7]
)");
    TopoMeta topo;

    EXPECT_TRUE(ParseYamlTopo("ut_range_compact", topo));
    EXPECT_EQ(topo.size(), 1u);
    EXPECT_EQ(topo.at(0).size(), 4u);
    for (uint32_t serId = 0; serId < 4; ++serId) {
        ServerMeta expected = {0, 1, 2, 3, 4, 5, 6, 7};
        EXPECT_EQ(topo.at(0).at(serId), expected);
    }
    EXPECT_EQ(ShmGetPhyDeviceTotalCount(topo), 32u);
    RemoveYaml("ut_range_compact");
}

TEST_F(ParseYamlTopoTest, ParseShippedScaleFiles)
{
    struct ScaleCase {
        const char* name;
        uint32_t podCount;
        uint32_t serversPerPod;
        uint32_t ranksPerServer;
    };
    const std::vector<ScaleCase> cases = {
        {"148", 1, 4, 8},     {"11_64", 1, 1, 64},  {"116_8", 1, 16, 8},  {"1_128_8", 1, 128, 8},
        {"2_16_8", 2, 16, 8}, {"2_32_8", 2, 32, 8}, {"2_64_8", 2, 64, 8},
    };

    for (const auto& item : cases) {
        SCOPED_TRACE(item.name);
        const YAML::Node root = YAML::LoadFile(modelDir_ + "/" + item.name + ".yaml");
        EXPECT_EQ(root["meta"]["podNum"].as<uint32_t>(), item.podCount);
        EXPECT_EQ(root["meta"]["serNum"].as<uint32_t>(), item.podCount * item.serversPerPod);
        EXPECT_EQ(root["meta"]["rankNum"].as<uint32_t>(), item.podCount * item.serversPerPod * item.ranksPerServer);

        TopoMeta topo;
        ASSERT_TRUE(ParseYamlTopo(item.name, topo));
        ASSERT_EQ(topo.size(), item.podCount);
        for (uint32_t podId = 0; podId < item.podCount; ++podId) {
            ASSERT_EQ(topo.count(podId), 1u);
            const auto& servers = topo.at(podId);
            ASSERT_EQ(servers.size(), item.serversPerPod);
            for (uint32_t serverId = 0; serverId < item.serversPerPod; ++serverId) {
                ASSERT_EQ(servers.count(serverId), 1u);
                const auto& ranks = servers.at(serverId);
                ASSERT_EQ(ranks.size(), item.ranksPerServer);
                for (uint32_t rankId = 0; rankId < item.ranksPerServer; ++rankId) {
                    EXPECT_EQ(ranks[rankId], rankId);
                }
            }
        }
        EXPECT_EQ(ShmGetPhyDeviceTotalCount(topo), item.podCount * item.serversPerPod * item.ranksPerServer);
    }
}
TEST_F(ParseYamlTopoTest, ParseRangeFormatWithOffsetAndSingleRank)
{
    WriteYaml("ut_range_offset", R"(
meta:
  podNum: 1
  serNum: 3
  rankNum: 3
topology:
  - podId: 0
    servers:
      - serId_range: [10, 12]
        ranks_range: [4, 4]
)");
    TopoMeta topo;

    EXPECT_TRUE(ParseYamlTopo("ut_range_offset", topo));
    EXPECT_EQ(topo.size(), 1u);
    EXPECT_EQ(topo.at(0).size(), 3u);
    ServerMeta expected = {4};
    EXPECT_EQ(topo.at(0).at(10), expected);
    EXPECT_EQ(topo.at(0).at(11), expected);
    EXPECT_EQ(topo.at(0).at(12), expected);
    RemoveYaml("ut_range_offset");
}

TEST_F(ParseYamlTopoTest, ParseMixedRangeAndVerboseFormat)
{
    WriteYaml("ut_range_mixed", R"(
meta:
  podNum: 1
  serNum: 3
  rankNum: 10
topology:
  - podId: 0
    servers:
      - serId: 0
        ranks: [0, 1]
      - serId_range: [1, 2]
        ranks_range: [0, 3]
)");
    TopoMeta topo;

    EXPECT_TRUE(ParseYamlTopo("ut_range_mixed", topo));
    EXPECT_EQ(topo.size(), 1u);
    EXPECT_EQ(topo.at(0).size(), 3u);
    ServerMeta verboseExpected = {0, 1};
    ServerMeta rangeExpected = {0, 1, 2, 3};
    EXPECT_EQ(topo.at(0).at(0), verboseExpected);
    EXPECT_EQ(topo.at(0).at(1), rangeExpected);
    EXPECT_EQ(topo.at(0).at(2), rangeExpected);
    RemoveYaml("ut_range_mixed");
}

TEST_F(ParseYamlTopoTest, ParseMultiPodRangeFormat)
{
    WriteYaml("ut_range_multi_pod", R"(
meta:
  podNum: 2
  serNum: 4
  rankNum: 16
topology:
  - podId: 0
    servers:
      - serId_range: [0, 1]
        ranks_range: [0, 3]
  - podId: 1
    servers:
      - serId_range: [0, 1]
        ranks_range: [0, 3]
)");
    TopoMeta topo;

    EXPECT_TRUE(ParseYamlTopo("ut_range_multi_pod", topo));
    EXPECT_EQ(topo.size(), 2u);
    ServerMeta expected = {0, 1, 2, 3};
    for (uint32_t podId = 0; podId < 2; ++podId) {
        EXPECT_EQ(topo.at(podId).size(), 2u);
        EXPECT_EQ(topo.at(podId).at(0), expected);
        EXPECT_EQ(topo.at(podId).at(1), expected);
    }
    RemoveYaml("ut_range_multi_pod");
}

TEST_F(ParseYamlTopoTest, ParseRangeAlternateListStyles)
{
    WriteYaml("ut_range_alt_styles", R"(
meta:
  podNum: 1
  serNum: 3
  rankNum: 5
topology:
  - podId: 0
    servers:
      - serId_range:
          - 0
          - 1
        ranks_range: ["2", "3"]
      - serId: 5
        ranks_range:
          - 4
          - 4
)");
    TopoMeta topo;

    EXPECT_TRUE(ParseYamlTopo("ut_range_alt_styles", topo));
    EXPECT_EQ(topo.size(), 1u);
    EXPECT_EQ(topo.at(0).size(), 3u);
    EXPECT_EQ(topo.at(0).at(0), ServerMeta({2, 3}));
    EXPECT_EQ(topo.at(0).at(1), ServerMeta({2, 3}));
    EXPECT_EQ(topo.at(0).at(5), ServerMeta({4}));
    RemoveYaml("ut_range_alt_styles");
}

TEST_F(ParseYamlTopoTest, ParseRangeBeginGreaterThanEnd)
{
    WriteYaml("ut_range_reversed", R"(
meta:
  podNum: 1
  serNum: 4
  rankNum: 32
topology:
  - podId: 0
    servers:
      - serId_range: [5, 2]
        ranks_range: [0, 7]
)");
    TopoMeta topo;

    EXPECT_FALSE(ParseYamlTopo("ut_range_reversed", topo));
    RemoveYaml("ut_range_reversed");
}

TEST_F(ParseYamlTopoTest, ParseScalarSerIdRangeRejected)
{
    WriteYaml("ut_scalar_ser_range", R"(
meta:
  podNum: 1
  serNum: 4
  rankNum: 32
topology:
  - podId: 0
    servers:
      - serId_range: 0-31
        ranks_range: [0, 7]
)");
    TopoMeta topo;

    EXPECT_FALSE(ParseYamlTopo("ut_scalar_ser_range", topo));
    RemoveYaml("ut_scalar_ser_range");
}

TEST_F(ParseYamlTopoTest, ParseScalarRanksRangeRejected)
{
    WriteYaml("ut_scalar_rank_range", R"(
meta:
  podNum: 1
  serNum: 1
  rankNum: 8
topology:
  - podId: 0
    servers:
      - serId: 0
        ranks_range: 0-7
)");
    TopoMeta topo;

    EXPECT_FALSE(ParseYamlTopo("ut_scalar_rank_range", topo));
    RemoveYaml("ut_scalar_rank_range");
}

TEST_F(ParseYamlTopoTest, ParseRangeNonNumericValue)
{
    WriteYaml("ut_range_non_numeric", R"(
meta:
  podNum: 1
  serNum: 4
  rankNum: 32
topology:
  - podId: 0
    servers:
      - serId_range: [0, 3]
        ranks_range: [abc, def]
)");
    TopoMeta topo;

    EXPECT_FALSE(ParseYamlTopo("ut_range_non_numeric", topo));
    RemoveYaml("ut_range_non_numeric");
}

TEST_F(ParseYamlTopoTest, ParseServerRangeWithoutRanks)
{
    WriteYaml("ut_range_no_ranks", R"(
meta:
  podNum: 1
  serNum: 4
  rankNum: 32
topology:
  - podId: 0
    servers:
      - serId_range: [0, 3]
)");
    TopoMeta topo;

    EXPECT_TRUE(ParseYamlTopo("ut_range_no_ranks", topo));
    EXPECT_EQ(topo.size(), 1u);
    EXPECT_EQ(topo.at(0).size(), 4u);
    for (uint32_t serId = 0; serId < 4; ++serId) {
        EXPECT_TRUE(topo.at(0).at(serId).empty());
    }
    RemoveYaml("ut_range_no_ranks");
}

TEST_F(ParseYamlTopoTest, ParseRangeExceedsWidthLimit)
{
    WriteYaml("ut_range_too_wide", R"(
meta:
  podNum: 1
  serNum: 1
  rankNum: 2
topology:
  - podId: 0
    servers:
      - serId_range: [0, 1024]
        ranks_range: [0, 1]
)");
    TopoMeta topo;

    EXPECT_FALSE(ParseYamlTopo("ut_range_too_wide", topo));
    RemoveYaml("ut_range_too_wide");
}

TEST_F(ParseYamlTopoTest, ParseSerIdWithRanksRange)
{
    WriteYaml("ut_serid_rank_range", R"(
meta:
  podNum: 1
  serNum: 1
  rankNum: 4
topology:
  - podId: 0
    servers:
      - serId: 5
        ranks_range: [0, 3]
)");
    TopoMeta topo;

    EXPECT_TRUE(ParseYamlTopo("ut_serid_rank_range", topo));
    EXPECT_EQ(topo.size(), 1u);
    EXPECT_EQ(topo.at(0).size(), 1u);
    ServerMeta expected = {0, 1, 2, 3};
    EXPECT_EQ(topo.at(0).at(5), expected);
    RemoveYaml("ut_serid_rank_range");
}

TEST_F(ParseYamlTopoTest, ParseSerIdWithListRanksRange)
{
    WriteYaml("ut_serid_rank_range_list", R"(
meta:
  podNum: 1
  serNum: 1
  rankNum: 8
topology:
  - podId: 0
    servers:
      - serId: 3
        ranks_range: [0, 7]
)");
    TopoMeta topo;

    EXPECT_TRUE(ParseYamlTopo("ut_serid_rank_range_list", topo));
    EXPECT_EQ(topo.size(), 1u);
    EXPECT_EQ(topo.at(0).size(), 1u);
    ServerMeta expected = {0, 1, 2, 3, 4, 5, 6, 7};
    EXPECT_EQ(topo.at(0).at(3), expected);
    RemoveYaml("ut_serid_rank_range_list");
}

TEST_F(ParseYamlTopoTest, ParseShipped228File)
{
    TopoMeta topo;

    EXPECT_TRUE(ParseYamlTopo("228", topo));
    EXPECT_EQ(topo.size(), 2u);
    for (uint32_t podId = 0; podId < 2; ++podId) {
        EXPECT_EQ(topo.at(podId).size(), 2u);
        ServerMeta expected = {0, 1, 2, 3, 4, 5, 6, 7};
        EXPECT_EQ(topo.at(podId).at(0), expected);
        EXPECT_EQ(topo.at(podId).at(1), expected);
    }
    EXPECT_EQ(ShmGetPhyDeviceTotalCount(topo), 32u);
}

TEST_F(ParseYamlTopoTest, ParseShipped242File)
{
    TopoMeta topo;

    EXPECT_TRUE(ParseYamlTopo("242", topo));
    EXPECT_EQ(topo.size(), 2u);
    ServerMeta expected = {0, 1};
    for (uint32_t podId = 0; podId < 2; ++podId) {
        EXPECT_EQ(topo.at(podId).size(), 4u);
        for (uint32_t serId = 0; serId < 4; ++serId) {
            EXPECT_EQ(topo.at(podId).at(serId), expected);
        }
    }
    EXPECT_EQ(ShmGetPhyDeviceTotalCount(topo), 16u);
}

TEST_F(ParseYamlTopoTest, ParseShipped288File)
{
    TopoMeta topo;

    EXPECT_TRUE(ParseYamlTopo("288", topo));
    EXPECT_EQ(topo.size(), 2u);
    ServerMeta expected = {0, 1, 2, 3, 4, 5, 6, 7};
    for (uint32_t podId = 0; podId < 2; ++podId) {
        EXPECT_EQ(topo.at(podId).size(), 8u);
        for (uint32_t serId = 0; serId < 8; ++serId) {
            EXPECT_EQ(topo.at(podId).at(serId), expected);
        }
    }
    EXPECT_EQ(ShmGetPhyDeviceTotalCount(topo), 128u);
}

TEST_F(ParseYamlTopoTest, ParseSerIdRangeWithRanksList)
{
    WriteYaml("ut_serrange_ranks_list", R"(
meta:
  podNum: 1
  serNum: 3
  rankNum: 6
topology:
  - podId: 0
    servers:
      - serId_range: [2, 4]
        ranks: [1, 3, 5]
)");
    TopoMeta topo;

    EXPECT_TRUE(ParseYamlTopo("ut_serrange_ranks_list", topo));
    EXPECT_EQ(topo.size(), 1u);
    EXPECT_EQ(topo.at(0).size(), 3u);
    ServerMeta expected = {1, 3, 5};
    EXPECT_EQ(topo.at(0).at(2), expected);
    EXPECT_EQ(topo.at(0).at(3), expected);
    EXPECT_EQ(topo.at(0).at(4), expected);
    RemoveYaml("ut_serrange_ranks_list");
}

TEST_F(ParseYamlTopoTest, ParseListFormRanges)
{
    WriteYaml("ut_list_form_ranges", R"(
meta:
  podNum: 1
  serNum: 3
  rankNum: 9
topology:
  - podId: 0
    servers:
      - serId_range: [1, 3]
        ranks_range: [0, 2]
)");
    TopoMeta topo;

    EXPECT_TRUE(ParseYamlTopo("ut_list_form_ranges", topo));
    EXPECT_EQ(topo.size(), 1u);
    EXPECT_EQ(topo.at(0).size(), 3u);
    ServerMeta expected = {0, 1, 2};
    EXPECT_EQ(topo.at(0).at(1), expected);
    EXPECT_EQ(topo.at(0).at(2), expected);
    EXPECT_EQ(topo.at(0).at(3), expected);
    RemoveYaml("ut_list_form_ranges");
}

TEST_F(ParseYamlTopoTest, ParseAllCombinationsMixedInOneFile)
{
    WriteYaml("ut_all_combinations", R"(
meta:
  podNum: 1
  serNum: 7
  rankNum: 15
topology:
  - podId: 0
    servers:
      - serId: 0
        ranks: [0, 1]
      - serId: 1
        ranks_range: [0, 3]
      - serId: [2, 4]
        ranks: [5, 6]
      - serId_range: [5, 6]
        ranks: [0, 2]
      - serId_range: [7, 7]
        ranks_range: [1, 1]
)");
    TopoMeta topo;

    EXPECT_TRUE(ParseYamlTopo("ut_all_combinations", topo));
    EXPECT_EQ(topo.size(), 1u);
    EXPECT_EQ(topo.at(0).size(), 7u);
    EXPECT_EQ(topo.at(0).at(0), ServerMeta({0, 1}));
    EXPECT_EQ(topo.at(0).at(1), ServerMeta({0, 1, 2, 3}));
    EXPECT_EQ(topo.at(0).at(2), ServerMeta({5, 6}));
    EXPECT_EQ(topo.at(0).at(4), ServerMeta({5, 6}));
    EXPECT_EQ(topo.at(0).at(5), ServerMeta({0, 2}));
    EXPECT_EQ(topo.at(0).at(6), ServerMeta({0, 2}));
    EXPECT_EQ(topo.at(0).at(7), ServerMeta({1}));
    EXPECT_EQ(ShmGetPhyDeviceTotalCount(topo), 15u);
    RemoveYaml("ut_all_combinations");
}

TEST_F(ParseYamlTopoTest, ParseSerIdListForm)
{
    WriteYaml("ut_serid_list", R"(
meta:
  podNum: 1
  serNum: 3
  rankNum: 6
topology:
  - podId: 0
    servers:
      - serId: [0, 3, 5]
        ranks: [1, 2]
)");
    TopoMeta topo;

    EXPECT_TRUE(ParseYamlTopo("ut_serid_list", topo));
    EXPECT_EQ(topo.size(), 1u);
    EXPECT_EQ(topo.at(0).size(), 3u);
    ServerMeta expected = {1, 2};
    EXPECT_EQ(topo.at(0).at(0), expected);
    EXPECT_EQ(topo.at(0).at(3), expected);
    EXPECT_EQ(topo.at(0).at(5), expected);
    EXPECT_EQ(ShmGetPhyDeviceTotalCount(topo), 6u);
    RemoveYaml("ut_serid_list");
}

TEST_F(ParseYamlTopoTest, ParseSerIdListInvalidElement)
{
    WriteYaml("ut_serid_list_bad_elem", R"(
meta:
  podNum: 1
  serNum: 2
  rankNum: 4
topology:
  - podId: 0
    servers:
      - serId: [0, abc]
        ranks: [0, 1]
)");
    TopoMeta topo;

    EXPECT_FALSE(ParseYamlTopo("ut_serid_list_bad_elem", topo));
    RemoveYaml("ut_serid_list_bad_elem");
}

TEST_F(ParseYamlTopoTest, ParseAmbiguousSerIdFields)
{
    WriteYaml("ut_ambiguous_serid", R"(
meta:
  podNum: 1
  serNum: 4
  rankNum: 8
topology:
  - podId: 0
    servers:
      - serId: 0
        serId_range: [1, 3]
        ranks: [0, 1]
)");
    TopoMeta topo;

    EXPECT_FALSE(ParseYamlTopo("ut_ambiguous_serid", topo));
    RemoveYaml("ut_ambiguous_serid");
}

TEST_F(ParseYamlTopoTest, ParseAmbiguousRanksFields)
{
    WriteYaml("ut_ambiguous_ranks", R"(
meta:
  podNum: 1
  serNum: 4
  rankNum: 8
topology:
  - podId: 0
    servers:
      - serId_range: [0, 3]
        ranks: [0, 1]
        ranks_range: [0, 1]
)");
    TopoMeta topo;

    EXPECT_FALSE(ParseYamlTopo("ut_ambiguous_ranks", topo));
    RemoveYaml("ut_ambiguous_ranks");
}

TEST_F(ParseYamlTopoTest, ParseRanksRangeListWrongSize)
{
    WriteYaml("ut_rank_range_bad_size", R"(
meta:
  podNum: 1
  serNum: 1
  rankNum: 4
topology:
  - podId: 0
    servers:
      - serId: 0
        ranks_range: [0, 1, 2]
)");
    TopoMeta topo;

    EXPECT_FALSE(ParseYamlTopo("ut_rank_range_bad_size", topo));
    RemoveYaml("ut_rank_range_bad_size");
}

TEST_F(ParseYamlTopoTest, ParseRanksRangeListReversed)
{
    WriteYaml("ut_rank_range_list_rev", R"(
meta:
  podNum: 1
  serNum: 1
  rankNum: 4
topology:
  - podId: 0
    servers:
      - serId: 0
        ranks_range: [5, 2]
)");
    TopoMeta topo;

    EXPECT_FALSE(ParseYamlTopo("ut_rank_range_list_rev", topo));
    RemoveYaml("ut_rank_range_list_rev");
}

TEST_F(ParseYamlTopoTest, ParseRanksNotSequence)
{
    WriteYaml("ut_ranks_not_seq", R"(
meta:
  podNum: 1
  serNum: 1
  rankNum: 4
topology:
  - podId: 0
    servers:
      - serId: 0
        ranks: 3
)");
    TopoMeta topo;

    EXPECT_FALSE(ParseYamlTopo("ut_ranks_not_seq", topo));
    RemoveYaml("ut_ranks_not_seq");
}

class ExportAndShellCompatIntegrationTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};
