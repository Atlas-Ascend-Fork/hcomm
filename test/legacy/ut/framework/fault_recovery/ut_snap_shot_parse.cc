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
#include <stdexcept>
#include <string>

#define private public
#define protected public
#include "snap_shot_parse.h"
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
#include "json_parser.h"
#include "rank_table.h"
#include "rank_table_info.h"
#include "recover_info.h"
#include "net_instance.h"
#include "rank_graph_builder.h"
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

    virtual void SetUp() { std::cout << "A Test case in SnapShotParserTest SetUP" << std::endl; }

    virtual void TearDown()
    {
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

TEST_F(SnapShotParserTest, SerializeCommonInfoTest)
{
    // 初始化快照解析器实例
    SnapShotParser& parser = SnapShotParser::GetInstance();
    Snapshot snapShot;

    // 创建示例输入参数
    CommParams commParams("test_comm_id", 0, 4, 0, DevType::DEV_TYPE_950, false, true);
    HcclCommConfig config;
    strcpy(config.reserved, "test_reserved");
    config.hcclBufferSize = 1024;
    config.hcclDeterministic = 1;
    strcpy(config.hcclCommName, "test_comm_name");
    strcpy(config.hcclUdi, "test_udi");

    std::unique_ptr<RankTableInfo> ranktableInfo = std::make_unique<RankTableInfo>();
    ranktableInfo->version = "1.0";
    ranktableInfo->rankCount = 2;

    NewRankInfo newRankInfo_0;
    newRankInfo_0.rankId = 0;
    newRankInfo_0.localId = 0;
    newRankInfo_0.deviceId = 0;

    NewRankInfo newRankInfo_1;
    newRankInfo_1.rankId = 1;
    newRankInfo_1.localId = 1;
    newRankInfo_1.deviceId = 1;

    RankLevelInfo rankLevelInfo;

    AddressInfo test_addr;
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

    JsonParser topoParser;
    std::shared_ptr<TopoInfo> topoInfo = std::make_shared<TopoInfo>();
    topoParser.ParseString(topoString, *topoInfo);
    // 创建二进制流
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
    CommParams commParams("test_comm_id", 0, 4, 0, DevType::DEV_TYPE_950, false, true);
    HcclCommConfig config;
    strcpy(config.reserved, "test_reserved");
    config.hcclBufferSize = 1024;
    config.hcclDeterministic = 1;
    strcpy(config.hcclCommName, "test_comm_name");
    strcpy(config.hcclUdi, "test_udi");

    std::unique_ptr<RankTableInfo> ranktableInfo = std::make_unique<RankTableInfo>();
    ranktableInfo->version = "1.0";
    ranktableInfo->rankCount = 2;

    NewRankInfo newRankInfo_0;
    newRankInfo_0.rankId = 0;
    newRankInfo_0.localId = 0;
    newRankInfo_0.deviceId = 0;

    NewRankInfo newRankInfo_1;
    newRankInfo_1.rankId = 1;
    newRankInfo_1.localId = 1;
    newRankInfo_1.deviceId = 1;

    RankLevelInfo rankLevelInfo;

    AddressInfo test_addr;
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

    JsonParser topoParser;
    std::shared_ptr<TopoInfo> topoInfo = std::make_shared<TopoInfo>();
    topoParser.ParseString(topoString, *topoInfo);
    // 创建二进制流
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
    CommParams commParams("test_comm_id", 0, 4, 0, DevType::DEV_TYPE_950, false, true);
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
    for (u32 versionCase : {0U, 1U, 2U}) {
        const bool legacy = versionCase == 0;
        const bool enabled = versionCase == 2;
        CommParams params("policy", 0, 1, 0, DevType::DEV_TYPE_950, false, true);
        HcclCommConfig config{};
        auto table = std::make_unique<RankTableInfo>();
        table->version = "2.0";
        table->rankCount = 1;
        NewRankInfo rank{};
        rank.rankId = 0;
        rank.localId = 0;
        rank.deviceId = 0;
        rank.replacedLocalId = 0;
        RankLevelInfo level{};
        level.netLayer = 0;
        level.netInstId = "server0";
        level.netType = NetType::TOPO_FILE_DESC;
        rank.rankLevelInfos.push_back(level);
        table->ranks.push_back(rank);
        auto topo = std::make_shared<TopoInfo>();
        topo->version = "2.0";
        topo->peerCount = 1;
        PeerInfo peer{};
        peer.localId = 0;
        topo->peers.push_back(peer);
        BinaryStream stream;
        if (legacy) {
            char oldVersion[SNAPSHOT_VERSION_SIZE] = "snapshotVersionIsFixNow";
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
        EXPECT_EQ(snapshot.snapshot.snapShotComm.commParams.commId, "policy");
        ASSERT_EQ(snapshot.subSnapshot.size(), 1U);
        EXPECT_STREQ(snapshot.subSnapshot[0].groupName, "sub_policy");
        EXPECT_EQ(snapshot.subSnapshot[0].snapShotSubComm.rankIds, (std::vector<RankId>{0}));
        EXPECT_EQ(snapshot.snapShotPub.step, 17U);
        ASSERT_EQ(snapshot.ccuStatusSnapshot.useMsCommIds.size(), 1U);
        EXPECT_STREQ(snapshot.ccuStatusSnapshot.useMsCommIds[0].data(), "policy");
    }
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
