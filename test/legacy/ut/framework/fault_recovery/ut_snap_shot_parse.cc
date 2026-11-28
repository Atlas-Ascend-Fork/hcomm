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
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>

#define private public
#define protected public
#include "snap_shot_parse.h"
#include "checkcrc.h"
#include "communicator_impl.h"
#include "internal_exception.h"
#include "null_ptr_exception.h"
#include "invalid_params_exception.h"
#include "orion_adapter_tsd.h"
#include "orion_adapter_hccp.h"
#include "hccp.h"
#include "hccp_ctx.h"
#include "hccp_common.h"
#include "coll_service_default_impl.h"
#include "rank_table.h"
#include "json/topo/topo_info_reader.h"
#include "rank_table.h"
#include "rank_table_info.h"
#include "recover_info.h"
#include "net_instance.h"
#include "rank_graph_builder.h"
#include "rank_graph_builder_bridge.h"
#include "rank_graph_test_data_builder.h"
#include "hccl/hccl_comm.h"
#include "comm_config_pub.h"
#include "phy_topo_builder.h"
#include "detour_service.h"
#include "sal.h"
#include "rank_gph.h"
#include "base_config_legacy.h"
#include "env_config_v2.h"

#undef private
#undef protected

using namespace Hccl;

class SnapShotParserTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SnapShotParserTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "SnapShotParserTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        hccl::PhyTopo::GetInstance()->Clear();
        std::cout << "A Test case in SnapShotParserTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        hccl::PhyTopo::GetInstance()->Clear();
        GlobalMockObject::verify();
        std::cout << "A Test case in SnapShotParserTest TearDown" << std::endl;
    }
};

TEST_F(SnapShotParserTest, SnapshotGenerate_test_err)
{
    char a = 'a';
    size_t usnapshotBufSize = 0;
    SnapShotBuf localBuff;
    auto res = SnapShotParser::GetInstance().ParseSnapshotToLocalBuff(&a, usnapshotBufSize, localBuff);
    EXPECT_EQ(HCCL_E_PARA, res);
}

TEST_F(SnapShotParserTest, Ut_ConfigSnapshot_When_SelectedOrLegacy_Expect_PreservePolicyAndStreamBoundary)
{
    auto& parser = SnapShotParser::GetInstance();
    for (bool subComm : {false, true}) {
        for (uint32_t version : {11U, 12U, 13U, 14U}) {
            SCOPED_TRACE(subComm);
            SCOPED_TRACE(version);
            HcclCommConfig input{};
            HcclCommConfigInit(&input);
            std::memcpy(input.reserved + offsetof(CommConfigInfo, version), &version, sizeof(version));
            ASSERT_EQ(EOK, strcpy_s(input.hcclUseNetLayer, sizeof(input.hcclUseNetLayer), "1,2"));
            BinaryStream stream;
            if (subComm) {
                parser.SerializeSubCommConfigInfo(input, stream);
            } else {
                parser.SerializeCommConfigInfo(input, stream);
            }
            const uint32_t sentinel = 0x12345678;
            stream << sentinel;
            HcclCommConfig output{};
            ASSERT_EQ(EOK, strcpy_s(output.hcclUseNetLayer, sizeof(output.hcclUseNetLayer), "stale"));
            EXPECT_EQ(
                HCCL_SUCCESS, subComm ? parser.DeserializeSubCommConfigInfo(stream, output) :
                                        parser.DeserializeCommConfigInfo(stream, output));
            EXPECT_STREQ(version >= 13U ? "1,2" : "", output.hcclUseNetLayer);
            uint32_t next = 0;
            stream >> next;
            EXPECT_EQ(sentinel, next);
        }
    }
}

TEST_F(SnapShotParserTest, Ut_ConfigSnapshot_When_TruncatedSelection_Expect_Reject)
{
    HcclCommConfig config{};
    HcclCommConfigInit(&config);
    BinaryStream stream;
    stream << config.reserved << config.hcclBufferSize << config.hcclDeterministic << config.hcclCommName
           << config.hcclUdi;
    HcclCommConfig output{};
    EXPECT_EQ(HCCL_E_PARA, SnapShotParser::GetInstance().DeserializeCommConfigInfo(stream, output));
    EXPECT_STREQ("", output.hcclUseNetLayer);
}

TEST_F(SnapShotParserTest, Ut_ConfigSnapshot_When_DirtyStringTail_Expect_ZeroPadding)
{
    for (const std::string expression : {std::string(), std::string("1,2"), std::string(31, '1')}) {
        SCOPED_TRACE(expression);
        HcclCommConfig config{};
        HcclCommConfigInit(&config);
        std::memset(config.hcclUseNetLayer, 'x', sizeof(config.hcclUseNetLayer));
        std::memcpy(config.hcclUseNetLayer, expression.c_str(), expression.size() + 1);
        BinaryStream stream;
        ASSERT_EQ(HCCL_SUCCESS, SnapShotParser::GetInstance().SerializeNetLayerConfig(config, stream));
        const std::string bytes = stream.GetString();
        ASSERT_EQ(HCCL_COMM_NET_LAYER_MAX_LENGTH + sizeof(uint32_t), bytes.size());
        EXPECT_EQ(expression, bytes.substr(0, expression.size()));
        EXPECT_TRUE(std::all_of(
            bytes.begin() + expression.size(), bytes.begin() + HCCL_COMM_NET_LAYER_MAX_LENGTH, [](char value) {
                return value == '\0';
            }));
        uint32_t marker = 0;
        std::memcpy(&marker, bytes.data() + HCCL_COMM_NET_LAYER_MAX_LENGTH, sizeof(marker));
        EXPECT_EQ(0x4E4C0001U, marker);
    }
}

TEST_F(SnapShotParserTest, Ut_ConfigSnapshot_When_InvalidSelectionField_Expect_Reject)
{
    auto& parser = SnapShotParser::GetInstance();
    HcclCommConfig config{};
    HcclCommConfigInit(&config);
    const size_t shortSize = offsetof(HcclCommConfig, hcclUseNetLayer);
    std::memcpy(config.reserved, &shortSize, sizeof(shortSize));
    BinaryStream stream;
    EXPECT_EQ(HCCL_E_PARA, parser.SerializeNetLayerConfig(config, stream));
    EXPECT_EQ(0U, stream.GetSize());
    HcclCommConfigInit(&config);
    std::memset(config.hcclUseNetLayer, '2', sizeof(config.hcclUseNetLayer));
    EXPECT_EQ(HCCL_E_PARA, parser.SerializeNetLayerConfig(config, stream));
    EXPECT_EQ(0U, stream.GetSize());
}

TEST_F(SnapShotParserTest, Ut_ConfigSnapshot_When_OldShortInput_Expect_NoTailRead)
{
    // 旧配置仅有头部时，不读取选层字段，也不改变快照边界。
    alignas(HcclCommConfig) char oldHeader[HCCL_COMM_CONFIG_INFO_BYTES]{};
    const size_t size = sizeof(oldHeader);
    const uint32_t magic = HCCL_COMM_CONFIG_MAGIC_WORD;
    std::memcpy(oldHeader, &size, sizeof(size));
    std::memcpy(oldHeader + sizeof(size), &magic, sizeof(magic));
    auto& parser = SnapShotParser::GetInstance();
    BinaryStream stream;
    EXPECT_EQ(
        HCCL_SUCCESS, parser.SerializeNetLayerConfig(*reinterpret_cast<const HcclCommConfig*>(oldHeader), stream));
    EXPECT_EQ(0U, stream.GetSize());
    HcclCommConfig output{};
    std::memcpy(output.reserved, oldHeader, sizeof(oldHeader));
    ASSERT_EQ(EOK, strcpy_s(output.hcclUseNetLayer, sizeof(output.hcclUseNetLayer), "2"));
    EXPECT_EQ(HCCL_SUCCESS, parser.DeserializeNetLayerConfig(stream, output));
    EXPECT_STREQ("", output.hcclUseNetLayer);
}

TEST_F(SnapShotParserTest, Ut_ConfigSnapshot_When_GraphInitialized_Expect_ReplayBuiltLayers)
{
    const auto* bridge = GetRankGraphBuilderBridge();
    ASSERT_NE(nullptr, bridge);
    auto& parser = SnapShotParser::GetInstance();
    // 不触发设备地址检查及拓扑缓存，保留真实 InitRankGraph 和选层结果保存路径。
    MOCKER_CPP(&CommunicatorImpl::CheckRankGraphAddrs).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::SaveTopoDesc).stubs().will(returnValue(HCCL_SUCCESS));
    const auto ranks = test::MakeRankTable4pForBuilder();
    const auto topo = test::MakeFourPeerBuilderTopo();
    const struct {
        const char* expression;
        std::set<u32> expected;
        const char* savedExpression;
    } cases[]
        = {{"", {0, 1, 2}, ""}, {"0", {0}, "0"}, {"2", {2}, "2"}, {"not(0)", {1, 2}, "1,2"}, {"0,2", {0, 2}, "0,2"}};
    for (bool subComm : {false, true}) {
        for (const auto& item : cases) {
            SCOPED_TRACE(subComm);
            SCOPED_TRACE(item.expression);
            CommunicatorImpl comm{};
            comm.myRank = 0;
            HcclCommConfigInit(&comm.config);
            ASSERT_EQ(EOK, strcpy_s(comm.config.hcclUseNetLayer, sizeof(comm.config.hcclUseNetLayer), item.expression));
            RankGraphBuildResult built{};
            HcclCommConfig parentConfig{};
            HcclCommConfigInit(&parentConfig);
            ASSERT_EQ(
                HCCL_SUCCESS, bridge->recoverBuild(
                                  RankTableInfo(ranks), topo, 0, false,
                                  subComm ? parentConfig.hcclUseNetLayer : comm.config.hcclUseNetLayer, built));
            const auto parent = built.rankGraph;
            if (subComm) {
                std::unique_ptr<RankGraph> child;
                ASSERT_EQ(HCCL_SUCCESS, bridge->createSubGraph(*parent, {0, 1}, comm.config.hcclUseNetLayer, child));
                comm.rankSize = child->GetRankSize();
                ASSERT_EQ(HCCL_SUCCESS, comm.InitRankGraph(child));
            } else {
                ASSERT_EQ(
                    HCCL_SUCCESS,
                    comm.InitRankGraph(
                        std::move(built.rankGraph), std::move(built.rankTableInfo), std::move(built.topoInfo)));
            }
            EXPECT_STREQ(item.savedExpression, comm.config.hcclUseNetLayer);
            BinaryStream stream;
            if (subComm) {
                parser.SerializeSubCommConfigInfo(comm.config, stream);
            } else {
                parser.SerializeCommConfigInfo(comm.config, stream);
            }
            HcclCommConfig restored{};
            ASSERT_EQ(
                HCCL_SUCCESS, subComm ? parser.DeserializeSubCommConfigInfo(stream, restored) :
                                        parser.DeserializeCommConfigInfo(stream, restored));
            EXPECT_STREQ(item.savedExpression, restored.hcclUseNetLayer);
            if (subComm) {
                std::unique_ptr<RankGraph> replayed;
                ASSERT_EQ(HCCL_SUCCESS, bridge->createSubGraph(*parent, {0, 1}, restored.hcclUseNetLayer, replayed));
                EXPECT_EQ(item.expected, replayed->GetLevels(0));
                EXPECT_EQ(2U, replayed->GetRankSize());
            } else {
                RankGraphBuildResult replayed{};
                ASSERT_EQ(
                    HCCL_SUCCESS,
                    bridge->recoverBuild(RankTableInfo(ranks), topo, 0, false, restored.hcclUseNetLayer, replayed));
                EXPECT_EQ(item.expected, replayed.rankGraph->GetLevels(0));
                EXPECT_EQ(4U, replayed.rankGraph->GetRankSize());
                EXPECT_EQ(ranks.rankCount, replayed.rankTableInfo.rankCount);
                EXPECT_EQ(topo.edgeCount, replayed.topoInfo.edgeCount);
            }
        }
    }
}

TEST_F(SnapShotParserTest, Ut_InitRankGraph_When_InvalidBuiltLayers_Expect_ConfigUnchanged)
{
    MOCKER_CPP(&CommunicatorImpl::CheckRankGraph).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::SaveTopoDesc).expects(never());
    for (u32 mask : {0U, 1U << MAX_NET_LAYER}) {
        SCOPED_TRACE(mask);
        auto graph = std::make_unique<RankGraph>(0);
        graph->buildLayerSelection_ = RankGraph::BuildLayerSelection{7U, mask};
        CommunicatorImpl comm{};
        HcclCommConfigInit(&comm.config);
        ASSERT_EQ(EOK, strcpy_s(comm.config.hcclUseNetLayer, sizeof(comm.config.hcclUseNetLayer), "0"));
        EXPECT_EQ(HCCL_E_PARA, comm.InitRankGraph(graph));
        EXPECT_STREQ("0", comm.config.hcclUseNetLayer);
    }
}

TEST_F(SnapShotParserTest, Ut_ParseSnapshot_When_InvalidMainOrSubSelection_Expect_Reject)
{
    auto& parser = SnapShotParser::GetInstance();
    HcclCommConfig config{};
    HcclCommConfigInit(&config);
    ASSERT_EQ(EOK, strcpy_s(config.hcclUseNetLayer, sizeof(config.hcclUseNetLayer), "1"));
    CommParams params("snapshot", 0, 1, 0, Hccl::DevType::DEV_TYPE_950, false, true);
    BinaryStream stream;
    parser.SerializeCommVersionInfo(stream);
    stream << std::string("main");
    parser.SerializeCommConfigInfo(config, stream);
    const size_t selectionSize = HCCL_COMM_NET_LAYER_MAX_LENGTH + sizeof(uint32_t);
    const size_t mainOffset = stream.GetSize() - selectionSize;
    parser.SerializeParamsInfo(params, stream);
    auto rankTable = std::make_unique<RankTableInfo>();
    rankTable->version = "1.0";
    rankTable->rankCount = 1;
    rankTable->ranks.resize(1);
    rankTable->ranks[0].rankLevelInfos.resize(1);
    rankTable->ranks[0].controlPlane.addrType = AddrType::IPV4;
    rankTable->ranks[0].controlPlane.addr = IpAddress("127.0.0.1");
    parser.SerializeRankTableInfo(std::move(rankTable), stream);
    auto topo = std::make_shared<hccl::TopoInfo>();
    parser.SerializeTopoInfo(topo, stream);
    stream << u32{0}; // 当前快照格式在 TopoInfo 后保存虚拟分组策略。
    stream << size_t{1} << std::string("sub");
    parser.SerializeSubCommParamsInfo(params, stream);
    parser.SerializeSubCommConfigInfo(config, stream);
    const size_t subOffset = stream.GetSize() - selectionSize;
    parser.SerializeRankIds({0}, stream);
    // 完整动态区：主域和一个子域均无已提交算子，CCU 通信域列表为空。
    stream << u32{0} << u32{0} << u32{0} << false << u32{0} << size_t{1};
    stream << u32{0} << u32{0} << false << u32{0} << size_t{0} << size_t{0};
    for (size_t offset : {mainOffset, subOffset}) {
        for (uint32_t fault : {0U, 1U, 2U}) {
            SCOPED_TRACE(offset);
            SCOPED_TRACE(fault);
            std::string bytes = stream.GetString();
            if (fault == 1) {
                bytes[offset + HCCL_COMM_NET_LAYER_MAX_LENGTH] ^= 1; // 损坏尾标记。
            } else if (fault == 2) {
                bytes.replace(offset, HCCL_COMM_NET_LAYER_MAX_LENGTH, HCCL_COMM_NET_LAYER_MAX_LENGTH, '1');
            }
            const uint32_t dataLen = static_cast<uint32_t>(bytes.size());
            const uint32_t totalSize = dataLen + 2 * sizeof(uint32_t);
            std::vector<uint32_t> buffer((totalSize + sizeof(uint32_t) - 1) / sizeof(uint32_t), 0);
            buffer[0] = dataLen;
            CheckCrc crc;
            ASSERT_EQ(HCCL_SUCCESS, crc.Calc32Crc(bytes.data(), dataLen, &buffer[1]));
            std::memcpy(buffer.data() + 2, bytes.data(), dataLen);
            SnapShotBuf output{};
            EXPECT_EQ(
                fault == 0 ? HCCL_SUCCESS : HCCL_E_PARA,
                parser.ParseSnapshotToLocalBuff(buffer.data(), totalSize, output));
        }
    }
}

TEST_F(SnapShotParserTest, SerializeCommonInfoTest)
{
    // 初始化快照解析器实例
    SnapShotParser& parser = SnapShotParser::GetInstance();
    Snapshot snapShot;

    // 创建示例输入参数
    CommParams commParams("test_comm_id", 0, 4, 0, Hccl::DevType::DEV_TYPE_950, false, true);
    HcclCommConfig config;
    strcpy(config.reserved, "test_reserved");
    config.hcclBufferSize = 1024;
    config.hcclDeterministic = 1;
    strcpy(config.hcclCommName, "test_comm_name");
    strcpy(config.hcclUdi, "test_udi");

    std::unique_ptr<RankTableInfo> ranktableInfo = std::make_unique<RankTableInfo>();
    ranktableInfo->version = "1.0";
    ranktableInfo->rankCount = 2;

    hccl::NewRankInfo newRankInfo_0;
    newRankInfo_0.rankId = 0;
    newRankInfo_0.localId = 0;
    newRankInfo_0.deviceId = 0;

    hccl::NewRankInfo newRankInfo_1;
    newRankInfo_1.rankId = 1;
    newRankInfo_1.localId = 1;
    newRankInfo_1.deviceId = 1;

    hccl::RankLevelInfo rankLevelInfo;

    hccl::AddressInfo test_addr;
    test_addr.ports = {"0/0", "0/1", "0/8"};
    rankLevelInfo.rankAddrs.emplace_back(test_addr);
    newRankInfo_0.rankLevelInfos.emplace_back(rankLevelInfo);
    newRankInfo_1.rankLevelInfos.emplace_back(rankLevelInfo);
    ranktableInfo->ranks.emplace_back(newRankInfo_0);
    ranktableInfo->ranks.emplace_back(newRankInfo_1);

    std::string topoString = R"({
	"version": "2.0",
	"peer_count" : 2,
	"peer_list" :[
		{ "local_id" : 0},
		{ "local_id" : 1}
	],
	"edge_count" : 3,
	"edge_list": [
		{
			"net_layer": 0,
			"link_type": "PEER2PEER",
			"protocols": ["UB_CTP"],
			"local_a": 0,
			"local_a_ports": ["0/0"],
			"local_b": 1,
			"local_b_ports": ["0/1"],
			"position": "DEVICE"
		},
        {
            "net_layer": 0,
            "link_type": "PEER2NET",
            "protocols": ["UB_CTP"],
            "local_a": 0,
            "local_a_ports": ["0/1"],
            "position": "DEVICE"
        },
        {
            "net_layer": 1,
            "link_type": "PEER2NET",
            "protocols": ["UB_TP"],
            "local_a": 0,
            "local_a_ports": ["0/8"],
            "position": "DEVICE"
        }
	]
})";
    std::shared_ptr<hccl::TopoInfo> topoInfo = std::make_shared<hccl::TopoInfo>();
    EXPECT_EQ(ParseJsonString(topoString, *topoInfo), HCCL_SUCCESS); // 创建二进制流
    BinaryStream binStream;

    // 调用序列化函数
    parser.SerializeCommonInfo(commParams, config, move(ranktableInfo), topoInfo, binStream, true);

    // 检查二进制流是否被正确写入（例如，检查大小是否增加）
    EXPECT_GT(binStream.GetSize(), 0);
    HcclResult ret = parser.DeserializeCommInfo(binStream, snapShot);
    EXPECT_EQ(HCCL_SUCCESS, ret);
    EXPECT_TRUE(snapShot.snapShotComm.enableVirtualGroups);

    binStream.Clear();
}

TEST_F(SnapShotParserTest, SerializeSubCommInfoTest)
{
    // 初始化快照解析器实例
    SnapShotParser& parser = SnapShotParser::GetInstance();
    // 创建示例输入参数
    CommParams commParams("test_comm_id", 0, 4, 0, Hccl::DevType::DEV_TYPE_950, false, true);
    HcclCommConfig config;
    strcpy(config.reserved, "test_reserved");
    config.hcclBufferSize = 1024;
    config.hcclDeterministic = 1;
    strcpy(config.hcclCommName, "test_comm_name");
    strcpy(config.hcclUdi, "test_udi");

    std::unique_ptr<RankTableInfo> ranktableInfo = std::make_unique<RankTableInfo>();
    ranktableInfo->version = "1.0";
    ranktableInfo->rankCount = 2;

    hccl::NewRankInfo newRankInfo_0;
    newRankInfo_0.rankId = 0;
    newRankInfo_0.localId = 0;
    newRankInfo_0.deviceId = 0;

    hccl::NewRankInfo newRankInfo_1;
    newRankInfo_1.rankId = 1;
    newRankInfo_1.localId = 1;
    newRankInfo_1.deviceId = 1;

    hccl::RankLevelInfo rankLevelInfo;

    hccl::AddressInfo test_addr;
    test_addr.ports = {"0/0", "0/1", "0/8"};
    rankLevelInfo.rankAddrs.emplace_back(test_addr);
    newRankInfo_0.rankLevelInfos.emplace_back(rankLevelInfo);
    newRankInfo_1.rankLevelInfos.emplace_back(rankLevelInfo);
    ranktableInfo->ranks.emplace_back(newRankInfo_0);
    ranktableInfo->ranks.emplace_back(newRankInfo_1);

    std::string topoString = R"({
	"version": "2.0",
	"peer_count" : 2,
	"peer_list" :[
		{ "local_id" : 0},
		{ "local_id" : 1}
	],
	"edge_count" : 3,
	"edge_list": [
		{
			"net_layer": 0,
			"link_type": "PEER2PEER",
			"protocols": ["UB_CTP"],
			"local_a": 0,
			"local_a_ports": ["0/0"],
			"local_b": 1,
			"local_b_ports": ["0/1"],
			"position": "DEVICE"
		},
        {
            "net_layer": 0,
            "link_type": "PEER2NET",
            "protocols": ["UB_CTP"],
            "local_a": 0,
            "local_a_ports": ["0/1"],
            "position": "DEVICE"
        },
        {
            "net_layer": 1,
            "link_type": "PEER2NET",
            "protocols": ["UB_TP"],
            "local_a": 0,
            "local_a_ports": ["0/8"],
            "position": "DEVICE"
        }
	]
})";
    std::shared_ptr<hccl::TopoInfo> topoInfo = std::make_shared<hccl::TopoInfo>();
    EXPECT_EQ(ParseJsonString(topoString, *topoInfo), HCCL_SUCCESS); // 创建二进制流
    BinaryStream binStream;

    std::vector<u32> rankId = {0, 1, 2, 3};

    // 调用序列化函数
    parser.SerializeSubCommInfo(commParams, config, rankId, binStream);

    // 检查二进制流是否被正确写入
    EXPECT_GT(binStream.GetSize(), 0);

    binStream.Clear();
}

TEST_F(SnapShotParserTest, DeAllSnapShotDynamicBufTest)
{
    SnapShotParser& parser = SnapShotParser::GetInstance();
    BinaryStream buf;
    uint32_t step = 1;
    buf << step;

    u32 opAccState{0};
    u32 commAccState{0};
    bool isLoadOp = true;
    u32 submittedOpCnt = 0; // 不下发算子
    buf << opAccState << commAccState << isLoadOp << submittedOpCnt;

    u32 groupNum = 1;
    buf << groupNum;
    SnapShotBuf localBuff;
    SubSnapshot subSnapshot;
    localBuff.subSnapshot.push_back(subSnapshot);

    HcclResult ret = parser.DeAllSnapShotDynamicBuf(buf, localBuff);
    EXPECT_EQ(HCCL_SUCCESS, ret);
    buf.Clear();
}

TEST_F(SnapShotParserTest, SerializeDynamicInfoTest)
{
    SnapShotParser& parser = SnapShotParser::GetInstance();
    std::vector<std::pair<u32, RankId>> levelRankPairs = {{0, 0}, {1, 1}, {2, 2}, {3, 3}};
    u32 submittedOpCnt = 0;
    BinaryStream binStream;
    HcclResult ret = parser.SerializeDynamicInfo(levelRankPairs, submittedOpCnt, binStream);
    EXPECT_EQ(HCCL_SUCCESS, ret);

    BinaryStream binStream1;
    parser.SerializeCommVersionInfo(binStream1);
}

TEST_F(SnapShotParserTest, SnapshotGenerate_test_3)
{
    char str[] = "hello world!";
    size_t usnapshotBufSize = sizeof(str);
    SnapShotBuf localBuff;
    MOCKER_CPP(&SnapShotParser::DeAllSnapShotStaticBuf)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&SnapShotParser::DeAllSnapShotDynamicBuf)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    auto res = SnapShotParser::GetInstance().ParseSnapshotToLocalBuff(str, usnapshotBufSize, localBuff);
}

TEST_F(SnapShotParserTest, DeSerializeSubCommInfoTest)
{
    SnapShotParser& parser = SnapShotParser::GetInstance();
    BinaryStream buf;
    SubSnapshot snapShot;
    HcclCommConfig config;
    strcpy(config.reserved, "t");
    config.hcclBufferSize = 1024;
    config.hcclDeterministic = 1;
    strcpy(config.hcclCommName, "t");
    strcpy(config.hcclUdi, "t");
    std::string commId{"1"};
    RankId myRank{0};
    u32 rankSize{0};
    RankId rankInParentComm{0};
    u32 dev = 0;
    bool devUsed{false};

    buf << commId;
    buf << myRank;
    buf << rankSize;
    buf << rankInParentComm;
    buf << dev;
    buf << devUsed;
    buf << config.reserved;
    buf << config.hcclBufferSize;
    buf << config.hcclDeterministic;
    buf << config.hcclCommName;
    buf << config.hcclUdi;
    std::string version = "1";
    u32 rankCount{1};
    u32 ranksSize = 1;
    RankId rankId = 0;
    s32 localId = 0;
    u32 rankLevelNum = 1;

    u32 level{0};
    std::string id = "";
    u32 fabricTypeInt{0};
    u32 addrTypeInt{0};
    u32 addrSize{0};

    s32 family{AF_INET};
    s32 scopeID{0};
    char dst[INET6_ADDRSTRLEN]{0};

    u32 rankTableCrcVal = 0;

    buf << version;
    buf << rankCount << ranksSize;
    buf << rankId << localId << rankLevelNum;
    buf << level << id << fabricTypeInt << addrTypeInt << addrSize;
    buf << family << scopeID;
    buf << dst;
    buf << rankTableCrcVal;
    HcclResult result = parser.DeserializeSubCommInfo(buf, snapShot);

    // 检查返回结果是否为成功
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(SnapShotParserTest, DeSerializeSubCommInfoTest2)
{
    // 初始化快照解析器实例
    SnapShotParser& parser = SnapShotParser::GetInstance();

    // 创建示例输入参数
    CommParams commParams("test_comm_id", 0, 4, 0, Hccl::DevType::DEV_TYPE_950, false, true);
    HcclCommConfig config;
    strcpy(config.reserved, "t");
    config.hcclBufferSize = 1024;
    config.hcclDeterministic = 1;
    strcpy(config.hcclCommName, "t");
    strcpy(config.hcclUdi, "t");

    // 创建二进制流
    BinaryStream binStream;
    std::vector<u32> rankId = {0, 1, 2, 3};

    // 调用序列化函数
    parser.SerializeSubCommInfo(commParams, config, rankId, binStream);
    // 写入其他字段
    SubSnapshot snapShot;
    // 调用反序列化函数
    HcclResult result = parser.DeserializeSubCommInfo(binStream, snapShot);

    // 检查返回结果是否为成功
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(SnapShotParserTest, DeAllSnapShotStaticBufTest)
{
    SnapShotParser& parser = SnapShotParser::GetInstance();
    BinaryStream buf;
    SnapShotBuf localBuff;
    parser.SerializeCommVersionInfo(buf);
    buf << std::string("test_comm");
    size_t groupNum = 0;
    buf << groupNum;
    MOCKER_CPP(&SnapShotParser::DeserializeCommInfo)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult ret = parser.DeAllSnapShotStaticBuf(buf, localBuff);
    EXPECT_EQ(HCCL_SUCCESS, ret);
    buf.Clear();
}

TEST_F(SnapShotParserTest, DeAllSnapShotStaticBufTest_2)
{
    SnapShotParser& parser = SnapShotParser::GetInstance();
    BinaryStream buf;
    SnapShotBuf localBuff;
    parser.SerializeCommVersionInfo(buf);
    string a = "a";
    buf << a;
    size_t groupNum = 1;
    buf << groupNum;
    buf << a;
    MOCKER_CPP(&SnapShotParser::DeserializeCommInfo)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&SnapShotParser::DeserializeSubCommInfo)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    HcclResult ret = parser.DeAllSnapShotStaticBuf(buf, localBuff);
    EXPECT_EQ(HCCL_SUCCESS, ret);
    buf.Clear();
}

TEST_F(SnapShotParserTest, TestSuccessfulParse_ParseSnapshotToLocalBuff)
{
    void* snapshotBuf = reinterpret_cast<void*>(new uint32_t[2]{0, 0});

    uint32_t snapshotBufSize = 8; // Matching size
    SnapShotBuf localBuff;
    SnapShotParser& parser = SnapShotParser::GetInstance();
    MOCKER_CPP(&SnapShotParser::DeAllSnapShotStaticBuf)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&SnapShotParser::DeAllSnapShotDynamicBuf)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    EXPECT_EQ(HCCL_SUCCESS, parser.ParseSnapshotToLocalBuff(snapshotBuf, snapshotBufSize, localBuff));
    delete[] static_cast<uint32_t*>(snapshotBuf);
}

TEST_F(SnapShotParserTest, test_binarystream_checkCrc)
{
    BinaryStream bs;
    unsigned char bytes[10]{0x92, 0x3e, 0x8e, 0x45, 0xa7, 0xc3, 0x4d, 0xff, 0x3e, 0xa3};

    SnapShotParser& parser = SnapShotParser::GetInstance();
    for (auto byte : bytes) {
        bs << byte;
    }
    u32 crc = 0;
    parser.CalcBufCrc32(bs, crc);
    auto ret = parser.CheckBufCrc32(bs, crc);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(SnapShotParserTest, DeserializeCcuStatusBufTest)
{
    // 创建一个测试用的BinaryStream
    BinaryStream buf;

    // 写入useMsCommIds的大小和内容
    size_t useMsCommIdsSize = 1;
    buf << useMsCommIdsSize;
    std::vector<std::string> useMsCommIds = {"comm1"};
    for (const auto& id : useMsCommIds) {
        buf << id;
    }

    // 写入useSchedCommIds的大小和内容
    size_t useSchedCommIdsSize = 3;
    buf << useSchedCommIdsSize;
    std::vector<std::string> useSchedCommIds = {"sch1", "sch2", "sch3"};
    for (const auto& id : useSchedCommIds) {
        buf << id;
    }
    SnapShotParser& parser = SnapShotParser::GetInstance();
    SnapShotBuf localBuff;
    // 调用DeserializeCcuStatusBuf函数
    EXPECT_EQ(parser.DeserializeCcuStatusBuf(buf, localBuff), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(localBuff.ccuStatusSnapshot.useMsCommIds.size(), useMsCommIdsSize);
    EXPECT_EQ(localBuff.ccuStatusSnapshot.useSchedCommIds.size(), useSchedCommIdsSize);

    std::vector<std::string> targetUseMsCommIds{};
    for (auto useMsCommId : localBuff.ccuStatusSnapshot.useMsCommIds) {
        targetUseMsCommIds.push_back(useMsCommId.data());
    }

    std::vector<std::string> targetUseSchedCommIds{};
    for (auto useSchedCommId : localBuff.ccuStatusSnapshot.useSchedCommIds) {
        targetUseSchedCommIds.push_back(useSchedCommId.data());
    }

    // 验证结果
    EXPECT_EQ(targetUseMsCommIds.size(), useMsCommIdsSize);
    for (size_t i = 0; i < useMsCommIdsSize; ++i) {
        EXPECT_EQ(targetUseMsCommIds[i], useMsCommIds[i]);
    }

    EXPECT_EQ(targetUseSchedCommIds.size(), useSchedCommIdsSize);
    for (size_t i = 0; i < useSchedCommIdsSize; ++i) {
        EXPECT_EQ(targetUseSchedCommIds[i], useSchedCommIds[i]);
    }
}

TEST_F(SnapShotParserTest, Ut_ParseSnapshotToLocalBuff_When_Error_Expect_OK_ReturnIsHCCL_E_INTERNAL)
{
    void* snapshotBuf = reinterpret_cast<void*>(new uint32_t[2]{0, 0});

    uint32_t snapshotBufSize = 8; // Matching size
    SnapShotBuf localBuff;
    SnapShotParser& parser = SnapShotParser::GetInstance();
    MOCKER_CPP(&SnapShotParser::DeAllSnapShotStaticBuf)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(throws(InternalException("")));
    MOCKER_CPP(&SnapShotParser::DeAllSnapShotDynamicBuf)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    EXPECT_EQ(HCCL_E_INTERNAL, parser.ParseSnapshotToLocalBuff(snapshotBuf, snapshotBufSize, localBuff));
    delete[] static_cast<uint32_t*>(snapshotBuf);
}

TEST_F(SnapShotParserTest, ut_DeSnapShotDynamicBuf_with_all_elements_ReturnHCCL_Success)
{
    BinaryStream buf;
    u32 opAccState{0};
    u32 commAccState{0};
    bool isLoadOp = true;
    u32 submittedOpCnt = 1;
    buf << opAccState << commAccState << isLoadOp << submittedOpCnt;
    u32 opMode{0};
    buf << opMode;

    size_t levelRankPairsCnt{1};
    u32 rankOrder{0};
    u32 rankId{0};
    buf << levelRankPairsCnt << rankOrder << rankId;

    size_t linkGroupPairCount{1};
    size_t linkSize{1};
    buf << linkGroupPairCount << linkSize;

    u32 dieId{0};
    IpAddress localAddr{"10.0.0.1"};
    IpAddress remoteAddr{"10.0.0.2"};
    buf << rankId << dieId;
    localAddr.GetBinStream(buf);
    remoteAddr.GetBinStream(buf);
    u32 cntCke{3};
    buf << cntCke;

    SnapShotDynamic info{};
    vector<LinkInfo> linkInfos(1, LinkInfo{});

    SnapShotParser& parser = SnapShotParser::GetInstance();
    EXPECT_EQ(HCCL_SUCCESS, parser.DeSnapShotDynamicBuf(buf, info));
}

TEST_F(SnapShotParserTest, CurrentAndLegacySnapshotPolicyRoundTrip)
{
    auto& parser = SnapShotParser::GetInstance();
    for (u32 versionCase : {0U, 1U, 2U, 3U, 4U}) {
        SCOPED_TRACE(versionCase);
        const bool legacy = versionCase == 0;
        const bool netLayerFormat = versionCase >= 3;
        const bool enabled = versionCase == 2 || versionCase == 4;
        CommParams params("policy", 0, 1, 0, Hccl::DevType::DEV_TYPE_950, false, true);
        HcclCommConfig config{};
        if (netLayerFormat) {
            HcclCommConfigInit(&config);
            ASSERT_EQ(EOK, strcpy_s(config.hcclUseNetLayer, sizeof(config.hcclUseNetLayer), "0"));
        }
        auto table = std::make_unique<RankTableInfo>();
        table->version = "2.0";
        table->rankCount = 1;
        hccl::NewRankInfo rank{};
        rank.rankId = 0;
        rank.localId = 0;
        rank.deviceId = 0;
        rank.replacedLocalId = 0;
        hccl::RankLevelInfo level{};
        level.netLayer = 0;
        level.netInstId = "server0";
        level.netType = NetType::TOPO_FILE_DESC;
        rank.rankLevelInfos.push_back(level);
        table->ranks.push_back(rank);
        auto topo = std::make_shared<hccl::TopoInfo>();
        topo->version = "2.0";
        topo->peerCount = 1;
        hccl::PeerInfo peer{};
        peer.localId = 0;
        topo->peers.push_back(peer);
        BinaryStream stream;
        if (!netLayerFormat) {
            char oldVersion[SNAPSHOT_VERSION_SIZE] = {};
            ASSERT_EQ(
                EOK,
                strcpy_s(oldVersion, sizeof(oldVersion), legacy ? LEGACY_SNAPSHOT_VERSION : GROUPING_SNAPSHOT_VERSION));
            char otherVersion[SNAPSHOT_VERSION_SIZE] = {};
            stream << oldVersion << otherVersion << otherVersion;
        } else {
            parser.SerializeCommVersionInfo(stream);
        }
        stream << std::string("policy");
        if (legacy) {
            // Reproduce the legacy layout without the trailing u32 policy field.
            parser.SerializeCommConfigInfo(config, stream);
            parser.SerializeParamsInfo(params, stream);
            parser.SerializeRankTableInfo(std::move(table), stream);
            parser.SerializeTopoInfo(topo, stream);
        } else {
            parser.SerializeCommonInfo(params, config, std::move(table), topo, stream, enabled);
        }
        stream << size_t{1} << std::string("sub_policy");
        parser.SerializeSubCommInfo(params, config, std::vector<u32>{0}, stream);
        stream << u32{17};                             // step
        stream << u32{0} << u32{0} << false << u32{0}; // global dynamic info
        stream << size_t{1};
        stream << u32{0} << u32{0} << false << u32{0};             // sub-communicator dynamic info
        stream << size_t{1} << std::string("policy") << size_t{0}; // CCU status
        u32 crc = 0;
        ASSERT_EQ(parser.CalcBufCrc32(stream, crc), HCCL_SUCCESS);
        BinaryStream header;
        header << static_cast<u32>(stream.GetSize()) << crc;
        std::vector<char> bytes;
        header.Dump(bytes);
        stream.Dump(bytes);
        SnapShotBuf snapshot{};
        snapshot.snapshot.snapShotComm.enableVirtualGroups = true;
        ASSERT_EQ(parser.ParseSnapshotToLocalBuff(bytes.data(), bytes.size(), snapshot), HCCL_SUCCESS);
        EXPECT_EQ(snapshot.snapshot.snapShotComm.enableVirtualGroups, enabled);
        EXPECT_STREQ(netLayerFormat ? "0" : "", snapshot.snapshot.snapShotComm.config.hcclUseNetLayer);
        EXPECT_EQ(snapshot.snapshot.snapShotComm.commParams.commId, "policy");
        ASSERT_EQ(snapshot.subSnapshot.size(), 1U);
        EXPECT_STREQ(snapshot.subSnapshot[0].groupName, "sub_policy");
        EXPECT_EQ(snapshot.subSnapshot[0].snapShotSubComm.rankIds, (std::vector<RankId>{0}));
        EXPECT_STREQ(netLayerFormat ? "0" : "", snapshot.subSnapshot[0].snapShotSubComm.config.hcclUseNetLayer);
        EXPECT_EQ(snapshot.snapShotPub.step, 17U);
        ASSERT_EQ(snapshot.ccuStatusSnapshot.useMsCommIds.size(), 1U);
        EXPECT_STREQ(snapshot.ccuStatusSnapshot.useMsCommIds[0].data(), "policy");
    }
}

TEST_F(SnapShotParserTest, NetLayerSnapshotVersionIsRejectedByPreviousReaders)
{
    auto& parser = SnapShotParser::GetInstance();
    BinaryStream stream;
    parser.SerializeCommVersionInfo(stream);
    const u32 sentinel = 0x12345678;
    stream << sentinel;
    SnapShotPub header{};
    ASSERT_EQ(HCCL_SUCCESS, parser.DeserializeCommVersionInfo(stream, header));
    EXPECT_STREQ("snapshotNetLayerSelectionV1", header.snapshotVersion);
    // 固定旧读端支持的格式标识，验证它会在读取通信域字段前拒绝新格式。
    const char legacyVersion[SNAPSHOT_VERSION_SIZE] = "snapshotVersionIsFixNow";
    const char groupingVersion[SNAPSHOT_VERSION_SIZE] = "snapshotVirtualGroupsPolicyV1";
    EXPECT_NE(0, std::memcmp(header.snapshotVersion, legacyVersion, SNAPSHOT_VERSION_SIZE));
    EXPECT_NE(0, std::memcmp(header.snapshotVersion, groupingVersion, SNAPSHOT_VERSION_SIZE));
    u32 next = 0;
    stream >> next;
    EXPECT_EQ(sentinel, next);
}

TEST_F(SnapShotParserTest, UnknownSnapshotVersionIsNotSupported)
{
    BinaryStream stream;
    char oldVersion[SNAPSHOT_VERSION_SIZE] = "unknownSnapshotVersion";
    char otherVersion[SNAPSHOT_VERSION_SIZE] = {};
    stream << oldVersion << otherVersion << otherVersion;
    SnapShotBuf snapshot{};
    EXPECT_EQ(SnapShotParser::GetInstance().DeAllSnapShotStaticBuf(stream, snapshot), HCCL_E_NOT_SUPPORT);
}
