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
#include <memory>
#include <set>
#include <string>
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#include "virtual_topo_stub.h"
#include "virtual_topo.h"
#include "link_data_adapter.h"

using namespace Hccl;

class LinkDataTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "LinkDataTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "LinkDataTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in LinkDataTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in LinkDataTest TearDown" << std::endl;
    }

    std::shared_ptr<NetInstance::Link> InitLink(
        const std::shared_ptr<NetInstance::Node>& srcNode, const std::shared_ptr<NetInstance::Node>& dstNode,
        u32 hop = 1, const std::set<std::string>& ports = {"0/1"})
    {
        IpAddress srcAddr("0.0.0.0");
        IpAddress dstAddr("0.0.0.1");
        std::set<LinkProtocol> protocols = {LinkProtocol::UB_CTP};
        NetInstance::ConnInterface srcIf(srcAddr, ports, AddrPosition::DEVICE, LinkType::PEER2PEER, protocols);
        NetInstance::ConnInterface dstIf(dstAddr, ports, AddrPosition::DEVICE, LinkType::PEER2PEER, protocols);
        return std::make_shared<NetInstance::Link>(
            srcNode, dstNode, std::make_shared<NetInstance::ConnInterface>(srcIf),
            std::make_shared<NetInstance::ConnInterface>(dstIf), LinkType::PEER2PEER, protocols, LinkDirection::BOTH,
            hop);
    }
};

TEST_F(LinkDataTest, linkData_get_uniqueId)
{
    LinkData linkData(
        PortDeploymentType::DEV_NET, LinkProtocol::UB_TP, 0, 1, IpAddress("0.0.0.0"), IpAddress("0.0.0.0"));
    auto data = linkData.GetUniqueId();

    LinkData linkData1(data);

    EXPECT_EQ(linkData.Describe(), linkData1.Describe());
}

TEST_F(LinkDataTest, makeLinkData_invalid_path_uses_default_fields)
{
    NetInstance::Path emptyPath;
    auto emptyLinkData = MakeLinkData(emptyPath);
    EXPECT_EQ(0U, emptyLinkData.GetHop());
    EXPECT_EQ(1U, emptyLinkData.GetPortGroupSize());
    EXPECT_FALSE(emptyLinkData.GetFullmesh());

    NetInstance::Path tooManyLinksPath;
    tooManyLinksPath.links.resize(3);
    auto tooManyLinksData = MakeLinkData(tooManyLinksPath);
    EXPECT_EQ(0U, tooManyLinksData.GetHop());
    EXPECT_EQ(1U, tooManyLinksData.GetPortGroupSize());
    EXPECT_FALSE(tooManyLinksData.GetFullmesh());
}

TEST_F(LinkDataTest, makeLinkData_single_link)
{
    auto srcPeer = std::make_shared<NetInstance::Peer>(0, 0, 0, 0);
    auto dstPeer = std::make_shared<NetInstance::Peer>(1, 1, 1, 0);
    NetInstance::Path path;
    path.links.push_back(*InitLink(srcPeer, dstPeer, 3));

    auto linkData = MakeLinkData(path);
    EXPECT_EQ(0U, linkData.GetLocalRankId());
    EXPECT_EQ(1U, linkData.GetRemoteRankId());
    EXPECT_EQ(3U, linkData.GetHop());
    EXPECT_EQ(1U, linkData.GetPortGroupSize());
    EXPECT_TRUE(linkData.GetFullmesh());
    EXPECT_EQ(LinkDirection::BOTH, linkData.GetDirection());
}

TEST_F(LinkDataTest, makeLinkData_double_link)
{
    auto srcPeer = std::make_shared<NetInstance::Peer>(0, 0, 0, 0);
    auto dstPeer = std::make_shared<NetInstance::Peer>(1, 1, 1, 0);
    std::set<std::string> ports = {"0/1", "2/3"};
    NetInstance::Path path;
    path.links.push_back(*InitLink(srcPeer, dstPeer, 2, ports));
    path.links.push_back(*InitLink(srcPeer, dstPeer, 2, ports));
    path.direction = LinkDirection::SEND_ONLY;

    auto linkData = MakeLinkData(path);
    EXPECT_EQ(0U, linkData.GetLocalRankId());
    EXPECT_EQ(1U, linkData.GetRemoteRankId());
    EXPECT_EQ(2U, linkData.GetHop());
    EXPECT_EQ(2U, linkData.GetPortGroupSize());
    EXPECT_FALSE(linkData.GetFullmesh());
    EXPECT_EQ(LinkDirection::SEND_ONLY, linkData.GetDirection());
}

TEST_F(LinkDataTest, makeLinkData_double_link_port_mismatch)
{
    auto srcPeer = std::make_shared<NetInstance::Peer>(0, 0, 0, 0);
    auto dstPeer = std::make_shared<NetInstance::Peer>(1, 1, 1, 0);
    std::set<LinkProtocol> protocols = {LinkProtocol::UB_CTP};
    std::set<std::string> srcPorts = {"0/1", "2/3"};
    std::set<std::string> dstPorts = {"0/1"};
    NetInstance::ConnInterface srcIf(
        IpAddress("0.0.0.0"), srcPorts, AddrPosition::DEVICE, LinkType::PEER2PEER, protocols);
    NetInstance::ConnInterface dstIf(
        IpAddress("0.0.0.1"), dstPorts, AddrPosition::DEVICE, LinkType::PEER2PEER, protocols);
    auto link = std::make_shared<NetInstance::Link>(
        srcPeer, dstPeer, std::make_shared<NetInstance::ConnInterface>(srcIf),
        std::make_shared<NetInstance::ConnInterface>(dstIf), LinkType::PEER2PEER, protocols, LinkDirection::BOTH, 1);

    NetInstance::Path path;
    path.links.push_back(*link);
    path.links.push_back(*link);

    auto linkData = MakeLinkData(path);
    EXPECT_EQ(2U, linkData.GetPortGroupSize());
    EXPECT_FALSE(linkData.GetFullmesh());
}

TEST_F(LinkDataTest, makePortData)
{
    auto peer = std::make_shared<NetInstance::Peer>(3, 0, 0, 0);
    auto link = InitLink(peer, peer);
    auto portData = MakePortData(3, *link->GetSourceIface());
    EXPECT_EQ(3U, portData.GetRankId());
    EXPECT_EQ(PortDeploymentType::DEV_NET, portData.GetType());
    EXPECT_EQ(IpAddress("0.0.0.0"), portData.GetAddr());
}
