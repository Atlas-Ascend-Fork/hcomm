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
#include <mockcpp/mockcpp.hpp>
#include <mockcpp/mokc.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <utility>
#include "adapter_error_manager_pub.h"
#include "invalid_params_exception.h"
#include "json_parser.h"
#include "rank_graph_test_data_builder.h"
#include "types.h"

#define private public
#include "phy_topo_builder.h"
#include "topo_info.h"

#undef private

using namespace Hccl;

namespace {
std::string g_reportedErrorCode;
std::vector<std::string> g_reportedKeys;
std::vector<std::string> g_reportedValues;

void CaptureRptInputErr(std::string errorCode, std::vector<std::string> keys, std::vector<std::string> values)
{
    g_reportedErrorCode = std::move(errorCode);
    g_reportedKeys = std::move(keys);
    g_reportedValues = std::move(values);
}
} // namespace

std::shared_ptr<TopoInfo> LoadTopoInfoStub(PhyTopoBuilder* This, const std::string& topoPath)
{
    (void)This;
    (void)topoPath;
    return std::make_shared<TopoInfo>(test::MakeTopoInfo(
        {0, 1, 2}, {
                       test::MakeEdge(
                           0, LinkType::PEER2PEER, 0, {"0/0"}, 1, {"0/1"}, LinkProtocol::UB_CTP, AddrPosition::DEVICE,
                           0, TopoType::MESH_1D),
                       test::MakeEdge(
                           0, LinkType::PEER2PEER, 1, {"0/1"}, 2, {"0/2"}, LinkProtocol::UB_CTP, AddrPosition::DEVICE,
                           0, TopoType::MESH_1D),
                       test::MakeEdge(
                           0, LinkType::PEER2NET, 1, {"0/1"}, 0, {}, LinkProtocol::UB_CTP, AddrPosition::DEVICE, 0,
                           TopoType::MESH_1D),
                       test::MakeEdge(
                           1, LinkType::PEER2NET, 0, {"1/3", "1/4", "0/3", "0/4"}, 0, {}, LinkProtocol::UB_MEM,
                           AddrPosition::DEVICE, 1, TopoType::MESH_1D),
                   }));
}

std::shared_ptr<TopoInfo> LoadTopoInfoWithDiffProtocols(PhyTopoBuilder* This, const std::string& topoPath)
{
    (void)This;
    (void)topoPath;
    return std::make_shared<TopoInfo>(test::MakeTopoInfo(
        {0, 1, 2}, {
                       test::MakeEdge(
                           0, LinkType::PEER2PEER, 1, {"0/1"}, 2, {"0/2"}, LinkProtocol::UB_CTP, AddrPosition::DEVICE,
                           0, TopoType::MESH_1D),
                       test::MakeEdge(0, LinkType::PEER2PEER, 1, {"0/1"}, 0, {"0/1"}, LinkProtocol::UB_MEM),
                       test::MakeEdge(
                           0, LinkType::PEER2NET, 0, {"0/1"}, 0, {}, LinkProtocol::UB_CTP, AddrPosition::DEVICE, 0,
                           TopoType::MESH_1D),
                       test::MakeEdge(
                           0, LinkType::PEER2NET, 1, {"0/1"}, 0, {}, LinkProtocol::TCP, AddrPosition::DEVICE, 0,
                           TopoType::MESH_1D),
                   }));
}

std::shared_ptr<TopoInfo> LoadTopoInfoWithRepeatEdge(PhyTopoBuilder* This, const std::string& topoPath)
{
    (void)This;
    (void)topoPath;
    return std::make_shared<TopoInfo>(test::MakeTopoInfo(
        {0, 1}, {
                    test::MakeEdge(
                        0, LinkType::PEER2PEER, 0, {"0/1"}, 1, {"0/2"}, LinkProtocol::UB_CTP, AddrPosition::DEVICE, 0,
                        TopoType::MESH_1D),
                    test::MakeEdge(
                        0, LinkType::PEER2PEER, 0, {"0/1"}, 1, {"0/2"}, LinkProtocol::UB_CTP, AddrPosition::DEVICE, 0,
                        TopoType::MESH_1D),
                }));
}

// 350L拆层前形态桩: layer0中mesh直连边与UB CLOS上交换边混装(topo.json未分层)
std::shared_ptr<TopoInfo> LoadTopoInfoUbxClosMixed(PhyTopoBuilder* This, const std::string& topoPath)
{
    (void)This;
    (void)topoPath;
    return std::make_shared<TopoInfo>(test::MakeTopoInfo(
        {0, 1, 2},
        {
            test::MakeEdge(
                0, LinkType::PEER2PEER, 0, {"0/0"}, 1, {"0/1"}, LinkProtocol::UB_CTP, AddrPosition::DEVICE, 0,
                TopoType::MESH_1D),
            test::MakeEdge(
                0, LinkType::PEER2PEER, 1, {"0/1"}, 2, {"0/2"}, LinkProtocol::UB_CTP, AddrPosition::DEVICE, 0,
                TopoType::MESH_1D),
            test::MakeEdge(
                0, LinkType::PEER2NET, 0, {"1/3"}, 0, {}, LinkProtocol::UB_MEM, AddrPosition::DEVICE, 0,
                TopoType::CLOS),
            test::MakeEdge(
                0, LinkType::PEER2NET, 1, {"1/4"}, 0, {}, LinkProtocol::UB_TP, AddrPosition::DEVICE, 0, TopoType::CLOS),
        }));
}

// 原生两层形态桩: mesh在layer0, CLOS已在layer1(topo.json本身已分层, 如8PMESH)
std::shared_ptr<TopoInfo> LoadTopoInfoAlreadyLayered(PhyTopoBuilder* This, const std::string& topoPath)
{
    (void)This;
    (void)topoPath;
    return std::make_shared<TopoInfo>(test::MakeTopoInfo(
        {0, 1, 2}, {
                       test::MakeEdge(
                           0, LinkType::PEER2PEER, 0, {"0/0"}, 1, {"0/1"}, LinkProtocol::UB_CTP, AddrPosition::DEVICE,
                           0, TopoType::MESH_1D),
                       test::MakeEdge(
                           0, LinkType::PEER2PEER, 1, {"0/1"}, 2, {"0/2"}, LinkProtocol::UB_CTP, AddrPosition::DEVICE,
                           0, TopoType::MESH_1D),
                       test::MakeEdge(
                           1, LinkType::PEER2NET, 0, {"1/3", "1/4", "0/3", "0/4"}, 0, {}, LinkProtocol::UB_MEM,
                           AddrPosition::DEVICE, 1, TopoType::CLOS),
                   }));
}

// 统计图中双向建链后的总边数
size_t CountGraphEdges(const std::shared_ptr<Graph<PhyTopo::Node, PhyTopo::Link>>& graph)
{
    if (graph == nullptr) {
        return 0;
    }
    size_t totalEdgeNum = 0;
    for (const auto& srcEntry : graph->edges) {
        for (const auto& dstEntry : srcEntry.second) {
            totalEdgeNum += dstEntry.second.size();
        }
    }
    return totalEdgeNum;
}

std::unique_ptr<PhyTopo> PhyTopoBuilderBuildStub(const std::string& topoPath)
{
    if (topoPath.empty()) {
        THROW<InvalidParamsException>("[PhyTopoBuilder::%s]Topo path is empty.", __func__);
    }

    HCCL_DEBUG("[PhyTopoBuilder::%s]Start to build physic topo.", __func__);
    std::unique_ptr<PhyTopo> phyTopo = std::make_unique<PhyTopo>();
    PhyTopoBuilder phyTopoBuilder;
    auto topoInfo = phyTopoBuilder.LoadTopoInfo(topoPath);
    // 根据topoInfo，按netLayer构造Graph
    for (const auto& iter : topoInfo->edges) {
        auto netLayer = iter.first;
        auto graph = phyTopoBuilder.CreateGraph(iter.second);
        if (phyTopo->AddTopoGraph(netLayer, graph) != HCCL_SUCCESS) {
            HCCL_ERROR("[PhyTopoBuilderBuildStub] AddTopoGraph failed, netLayer[%u]", netLayer);
            return nullptr;
        }
        HCCL_DEBUG("[PhyTopoBuilder::%s]Build netLayer[%u] topo graph success.", __func__, netLayer);
    }
    return phyTopo;
}

int StatLargeTopoFile(const char* path, struct stat* fileStat)
{
    (void)path;
    if (fileStat == nullptr) {
        return -1;
    }
    fileStat->st_size = SUPPORT_MAX_TOPOFILE_SIZE + 1U;
    return 0;
}

class PhyTopoBuilderTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "PhyTopoBuilderTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "PhyTopoBuilderTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in PhyTopoBuilderTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in PhyTopoBuilderTest TearDown" << std::endl;
    }
};

TEST_F(PhyTopoBuilderTest, Ut_PhyTopoBuilder_When_EmptyTopoPath_Expect_ThrowInvalidParamsException)
{
    std::string topoPath = "";
    EXPECT_THROW(PhyTopoBuilderBuildStub(topoPath), InvalidParamsException);
}

TEST_F(PhyTopoBuilderTest, Ut_PhyTopoBuilder_When_EmptyTopoPath_Expect_ReportEI0004WithCorrectArguments)
{
    PhyTopo::GetInstance()->Clear();
    g_reportedErrorCode.clear();
    g_reportedKeys.clear();
    g_reportedValues.clear();
    MOCKER(RptInputErr).stubs().will(invoke(CaptureRptInputErr));

    testing::internal::CaptureStdout();
    EXPECT_EQ(PhyTopoBuilder::GetInstance().Build(""), HCCL_E_PARA);
    const std::string log = testing::internal::GetCapturedStdout();
    EXPECT_EQ(g_reportedErrorCode, "EI0004");
    EXPECT_EQ(g_reportedKeys, std::vector<std::string>({"ranktable_path", "error_reason"}));
    EXPECT_EQ(g_reportedValues, std::vector<std::string>({"<empty>", "The topo JSON file path is empty."}));
    EXPECT_NE(log.find("[InitGroupStage][RanktableConfig] errNo["), std::string::npos);
}

TEST_F(PhyTopoBuilderTest, Ut_PhyTopoBuilder_When_InValidTopoPath_Expect_ThrowInvalidParamsException)
{
    std::string topoPath = "../topo_stub";
    EXPECT_THROW(PhyTopoBuilderBuildStub(topoPath), InvalidParamsException);
}

TEST_F(PhyTopoBuilderTest, Ut_PhyTopoBuilder_When_MaxTopoSizeFile_Expect_ThrowInvalidParamsException)
{
    std::string topoPath = "./large_topo_stub";
    MOCKER(stat).stubs().with(mockcpp::any(), mockcpp::any()).will(invoke(StatLargeTopoFile));
    EXPECT_THROW(PhyTopoBuilderBuildStub(topoPath), InvalidParamsException);
}

TEST_F(PhyTopoBuilderTest, Ut_PhyTopoBuilder_When_ErrorFilePath_Expect_ThrowInvalidParamsException)
{
    const std::string topoPath = "llt/ace/comop/hccl/none_file";
    g_reportedErrorCode.clear();
    MOCKER(RptInputErr).stubs().will(invoke(CaptureRptInputErr));

    testing::internal::CaptureStdout();
    EXPECT_THROW(PhyTopoBuilderBuildStub(topoPath), InvalidParamsException);
    const std::string log = testing::internal::GetCapturedStdout();
    EXPECT_EQ(g_reportedErrorCode, "EI0004");
    EXPECT_NE(log.find("[InitGroupStage][RanktableConfig] errNo["), std::string::npos);
}

TEST_F(PhyTopoBuilderTest, Ut_JsonParser_When_InvalidPath_Expect_ReportEI0004WithCorrectArguments)
{
    const std::string topoPath = "llt/ace/comop/hccl/none_file";
    JsonParser jsonParser;
    nlohmann::json parseInformation;
    g_reportedErrorCode.clear();
    g_reportedKeys.clear();
    g_reportedValues.clear();
    MOCKER(RptInputErr).stubs().will(invoke(CaptureRptInputErr));

    testing::internal::CaptureStdout();
    EXPECT_EQ(jsonParser.ParseFileToJson(topoPath, parseInformation), HCCL_E_PARA);
    const std::string log = testing::internal::GetCapturedStdout();
    EXPECT_EQ(g_reportedErrorCode, "EI0004");
    EXPECT_EQ(g_reportedKeys, std::vector<std::string>({"ranktable_path", "error_reason"}));
    ASSERT_EQ(g_reportedValues.size(), 2);
    EXPECT_EQ(g_reportedValues[0], topoPath);
    EXPECT_FALSE(g_reportedValues[1].empty());
    EXPECT_NE(log.find("[InitGroupStage][RanktableConfig] errNo["), std::string::npos);
    EXPECT_NE(log.find("is not a valid real path"), std::string::npos);
}

TEST_F(PhyTopoBuilderTest, Ut_JsonParserParseFile_When_InvalidPath_Expect_ReportEI0004AndKeywords)
{
    const std::string topoPath = "llt/ace/comop/hccl/none_file";
    JsonParser jsonParser;
    TopoInfo topoInfo;
    g_reportedErrorCode.clear();
    MOCKER(RptInputErr).stubs().will(invoke(CaptureRptInputErr));

    testing::internal::CaptureStdout();
    EXPECT_EQ(jsonParser.ParseFile(topoPath, topoInfo), HCCL_E_PARA);
    const std::string log = testing::internal::GetCapturedStdout();
    EXPECT_EQ(g_reportedErrorCode, "EI0004");
    EXPECT_NE(log.find("[InitGroupStage][RanktableConfig] errNo["), std::string::npos);
}

TEST_F(PhyTopoBuilderTest, Ut_JsonParser_When_MalformedFile_Expect_ReportEI0004AndReturnParseError)
{
    char filePath[] = "/tmp/hcomm_json_parser_XXXXXX";
    const int fd = mkstemp(filePath);
    ASSERT_NE(fd, -1);
    const char jsonContent[] = "{";
    const ssize_t jsonLength = static_cast<ssize_t>(sizeof(jsonContent) - 1);
    const ssize_t written = write(fd, jsonContent, static_cast<size_t>(jsonLength));
    const int closeResult = close(fd);

    JsonParser jsonParser;
    nlohmann::json parseInformation;
    g_reportedErrorCode.clear();
    g_reportedKeys.clear();
    g_reportedValues.clear();
    MOCKER(RptInputErr).stubs().will(invoke(CaptureRptInputErr));

    testing::internal::CaptureStdout();
    if (written == jsonLength) {
        EXPECT_EQ(jsonParser.ParseFileToJson(filePath, parseInformation), HCCL_E_PARA);
    }
    const std::string log = testing::internal::GetCapturedStdout();
    const int unlinkResult = unlink(filePath);

    ASSERT_EQ(written, jsonLength);
    EXPECT_EQ(closeResult, 0);
    EXPECT_EQ(unlinkResult, 0);
    EXPECT_EQ(g_reportedErrorCode, "EI0004");
    EXPECT_EQ(g_reportedKeys, std::vector<std::string>({"ranktable_path", "error_reason"}));
    ASSERT_EQ(g_reportedValues.size(), 2);
    EXPECT_EQ(g_reportedValues[0], filePath);
    EXPECT_EQ(g_reportedValues[1], "The JSON format is incorrect.");
    EXPECT_NE(log.find("JSON parse error"), std::string::npos);
    EXPECT_NE(log.find("[InitGroupStage][RanktableConfig] errNo["), std::string::npos);
}

TEST_F(PhyTopoBuilderTest, Ut_PhyTopoBuilder_When_ValidTopoPath_Expect_ReturnEdgeNum)
{
    std::string topoPath = "./topo_stub";
    MOCKER_CPP(&PhyTopoBuilder::LoadTopoInfo).stubs().will(invoke(LoadTopoInfoStub));
    std::unique_ptr<PhyTopo> phyTopo = PhyTopoBuilderBuildStub(topoPath);
    auto graph = phyTopo->GetTopoGraph(0);
    EXPECT_EQ(CountGraphEdges(graph), 6);
}

TEST_F(PhyTopoBuilderTest, Ut_PhyTopoBuilder_When_EdgeRepeat_Expect_ReturnEdgeNum)
{
    std::string topoPath = "./test_topo_stub";
    MOCKER_CPP(&PhyTopoBuilder::LoadTopoInfo).stubs().will(invoke(LoadTopoInfoWithRepeatEdge));
    std::unique_ptr<PhyTopo> phyTopo = PhyTopoBuilderBuildStub(topoPath);
    auto graph = phyTopo->GetTopoGraph(0);
    ASSERT_NE(graph, nullptr);
    EXPECT_EQ(CountGraphEdges(graph), 4);
}

TEST_F(PhyTopoBuilderTest, Ut_PhyTopoBuilder_When_DiffProtocols_Expect_ReturnEdgeNum)
{
    std::string topoPath = "./topo_stub";
    MOCKER_CPP(&PhyTopoBuilder::LoadTopoInfo).stubs().will(invoke(LoadTopoInfoWithDiffProtocols));
    std::unique_ptr<PhyTopo> phyTopo = PhyTopoBuilderBuildStub(topoPath);
    auto graph = phyTopo->GetTopoGraph(0);
    EXPECT_EQ(CountGraphEdges(graph), 8);
}

// 350L机型拆层复制: 绝对路径命中atlas_850_3.json时, UB CLOS边被复制到layer1, layer0原边保留
TEST_F(PhyTopoBuilderTest, Ut_PhyTopoBuilder_When_UbxTopoAbsolutePath_Expect_ClosEdgesCopiedToLayer1)
{
    PhyTopo::GetInstance()->Clear();
    PhyTopoBuilder::ubxClosSplit_ = false;
    MOCKER_CPP(&PhyTopoBuilder::LoadTopoInfo).stubs().will(invoke(LoadTopoInfoUbxClosMixed));

    const std::string topoPath = "/usr/local/Ascend/driver/topo/950/atlas_850_3.json";
    EXPECT_EQ(PhyTopoBuilder::GetInstance().Build(topoPath), HCCL_SUCCESS);

    // 拆层标志置位, layer1图被创建
    EXPECT_TRUE(PhyTopoBuilder::ubxClosSplit_);
    EXPECT_NE(PhyTopo::GetInstance()->GetTopoGraph(1), nullptr);
    // 复制而非搬移: layer0图仍保留全部4条输入边(2条mesh+2条CLOS, 双向建链后8条)
    EXPECT_EQ(CountGraphEdges(PhyTopo::GetInstance()->GetTopoGraph(0)), 8);
    // layer1图仅含2条CLOS边(双向建链后4条)
    EXPECT_EQ(CountGraphEdges(PhyTopo::GetInstance()->GetTopoGraph(1)), 4);
}

// 350L机型拆层复制: 无分隔符的相对路径文件名也能命中(验证find_last_of返回npos的分支)
TEST_F(PhyTopoBuilderTest, Ut_PhyTopoBuilder_When_UbxTopoRelativePath_Expect_ClosEdgesCopiedToLayer1)
{
    PhyTopo::GetInstance()->Clear();
    PhyTopoBuilder::ubxClosSplit_ = false;
    MOCKER_CPP(&PhyTopoBuilder::LoadTopoInfo).stubs().will(invoke(LoadTopoInfoUbxClosMixed));

    const std::string topoPath = "atlas_850_3.json";
    EXPECT_EQ(PhyTopoBuilder::GetInstance().Build(topoPath), HCCL_SUCCESS);

    EXPECT_TRUE(PhyTopoBuilder::ubxClosSplit_);
    EXPECT_NE(PhyTopo::GetInstance()->GetTopoGraph(1), nullptr);
    EXPECT_EQ(CountGraphEdges(PhyTopo::GetInstance()->GetTopoGraph(0)), 8);
    EXPECT_EQ(CountGraphEdges(PhyTopo::GetInstance()->GetTopoGraph(1)), 4);
}

// 350L机型拆层复制: 反斜杠分隔的路径也能命中文件名(看护find_last_of中'\\'分隔符分支)
TEST_F(PhyTopoBuilderTest, Ut_PhyTopoBuilder_When_UbxTopoBackslashPath_Expect_ClosEdgesCopiedToLayer1)
{
    PhyTopo::GetInstance()->Clear();
    PhyTopoBuilder::ubxClosSplit_ = false;
    MOCKER_CPP(&PhyTopoBuilder::LoadTopoInfo).stubs().will(invoke(LoadTopoInfoUbxClosMixed));

    const std::string topoPath = "C:\\driver\\topo\\atlas_850_3.json";
    EXPECT_EQ(PhyTopoBuilder::GetInstance().Build(topoPath), HCCL_SUCCESS);

    EXPECT_TRUE(PhyTopoBuilder::ubxClosSplit_);
    EXPECT_NE(PhyTopo::GetInstance()->GetTopoGraph(1), nullptr);
    EXPECT_EQ(CountGraphEdges(PhyTopo::GetInstance()->GetTopoGraph(0)), 8);
    EXPECT_EQ(CountGraphEdges(PhyTopo::GetInstance()->GetTopoGraph(1)), 4);
}

// 非350L机型(8PMESH的atlas_850_1.json): 不执行拆层, 无layer1图, 标志不置位
TEST_F(PhyTopoBuilderTest, Ut_PhyTopoBuilder_When_NonUbxTopoPath_Expect_NoSplitAndFlagNotSet)
{
    PhyTopo::GetInstance()->Clear();
    PhyTopoBuilder::ubxClosSplit_ = false;
    MOCKER_CPP(&PhyTopoBuilder::LoadTopoInfo).stubs().will(invoke(LoadTopoInfoUbxClosMixed));

    const std::string topoPath = "/usr/local/Ascend/driver/topo/950/atlas_850_1.json";
    EXPECT_EQ(PhyTopoBuilder::GetInstance().Build(topoPath), HCCL_SUCCESS);

    EXPECT_FALSE(PhyTopoBuilder::ubxClosSplit_);
    EXPECT_EQ(PhyTopo::GetInstance()->GetTopoGraph(1), nullptr);
    // 全部4条输入边仍在layer0(双向建链后8条)
    EXPECT_EQ(CountGraphEdges(PhyTopo::GetInstance()->GetTopoGraph(0)), 8);
}

// 原生两层形态(topo.json已含layer1): 不重复拆层, 标志不置位(防止误还原原生多级行为)
TEST_F(PhyTopoBuilderTest, Ut_PhyTopoBuilder_When_TopoAlreadyLayered_Expect_NoResplitAndFlagNotSet)
{
    PhyTopo::GetInstance()->Clear();
    PhyTopoBuilder::ubxClosSplit_ = false;
    MOCKER_CPP(&PhyTopoBuilder::LoadTopoInfo).stubs().will(invoke(LoadTopoInfoAlreadyLayered));

    const std::string topoPath = "/usr/local/Ascend/driver/topo/950/atlas_850_3.json";
    EXPECT_EQ(PhyTopoBuilder::GetInstance().Build(topoPath), HCCL_SUCCESS);

    // layer1图来自输入数据本身, 但拆层标志未置位
    EXPECT_FALSE(PhyTopoBuilder::ubxClosSplit_);
    EXPECT_NE(PhyTopo::GetInstance()->GetTopoGraph(1), nullptr);
    EXPECT_EQ(CountGraphEdges(PhyTopo::GetInstance()->GetTopoGraph(0)), 4);
}
