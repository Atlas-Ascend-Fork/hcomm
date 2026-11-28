/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PHY_TOPO_H
#define PHY_TOPO_H

#include <vector>
#include <set>
#include <unordered_map>
#include <memory>
#include "topo_common_types.h"
#include "iterator.h"
#include "graph.h"
// 两份.h 必须一起改：改本文件时同步改 legacy950 侧
// src/legacy/ascend950/framework/topo/new_topo_builder/phy_topo/phy_topo_legacy950.h。
// 两份归一化 namespace 后声明逐字一致（仅 namespace 名 Hccl/hccl + guard 名 + 同步注释 + include graph
// 名不同），legacy950 侧无 .cc，仅作布局锁定声明（除内联空构造/析构（及 MAKE_ENUM 生成的内联类）外
// 成员函数无 .cc 定义、调用将链接失败，生产实现只在
// coll_comm 侧 phy_topo.cc，namespace hccl）；两份均经 Hccl:: 限定同源类型（LinkType/
// LinkProtocol/AddrPosition/TopoType/LinkDirection/LocalId/NodeId/BaseConstIterator 均定义在
// namespace Hccl 的 topo_common_types.h 等，不随本类迁 ns），避免同 SO 跨 TU 声明分叉导致序列化错位。
namespace hccl {

class PhyTopo {
public:
    static std::unique_ptr<PhyTopo>& GetInstance();
    class ConnInterface {
    public:
        // 使用地址信息、位置信息、链路类型、链路协议构造接口
        ConnInterface(
            const std::set<std::string> inputPorts, const Hccl::AddrPosition inputPos,
            const Hccl::LinkType inputLinkType, const std::set<Hccl::LinkProtocol> inputLinkProtocols);
        std::set<std::string> GetPorts() const;
        Hccl::AddrPosition GetPos() const;
        Hccl::LinkType GetLinkType() const;
        std::set<Hccl::LinkProtocol> GetLinkProtocols() const;
        std::string Describe() const;
        bool operator==(const ConnInterface& rhs) const;
        bool operator!=(const ConnInterface& rhs) const;

    private:
        std::set<std::string> ports{};
        Hccl::AddrPosition pos{};
        Hccl::LinkType linkType{};
        std::set<Hccl::LinkProtocol> linkProtocols{};
    };

    class Node {
    public:
        using IfaceIterator = Hccl::BaseConstIterator<std::vector, std::shared_ptr<PhyTopo::ConnInterface>>;
        MAKE_ENUM(NodeType, PEER, FABRIC);
        explicit Node(const NodeType inputType);
        NodeType GetType() const;
        void AddConnInterface(const std::shared_ptr<PhyTopo::ConnInterface>& interface);
        IfaceIterator IterIfaces() const;
        virtual std::string Describe() const;

    private:
        NodeType type{};
        std::vector<std::shared_ptr<PhyTopo::ConnInterface>> interfaces{};
    };

    class Peer : public Node {
    public:
        explicit Peer(const Hccl::LocalId localId);
        Hccl::LocalId GetLocalId() const;
        static Hccl::NodeId GetId(const Hccl::LocalId localId);
        std::string Describe() const override;

    private:
        Hccl::LocalId localId{};
    };

    class Fabric : public Node {
    public:
        explicit Fabric();
        static Hccl::NodeId GetId();
        std::string Describe() const override;
    };

    struct LinkAttributes {
        Hccl::LinkType linktype;
        std::set<Hccl::LinkProtocol> protocols;
    };

    class Link {
    public:
        // 使用源节点、目的节点、链路类型、链路协议、topo类型、topoInstId 构造链路
        Link(
            std::shared_ptr<PhyTopo::Node> inputSource, std::shared_ptr<PhyTopo::Node> inputTarget,
            const LinkAttributes& properties, const Hccl::TopoType inputTopoType, const u32 inputTopoInstId);
        void SetSourceIface(std::shared_ptr<PhyTopo::ConnInterface> inputSourceIface);
        void SetTargetIface(std::shared_ptr<PhyTopo::ConnInterface> inputTargetIface);
        Hccl::LinkType GetType() const;
        std::set<Hccl::LinkProtocol> GetLinkProtocols() const;
        Hccl::LinkDirection GetLinkDirection() const;
        Hccl::TopoType GetTopoType() const;
        u32 GetTopoInstId() const;
        u32 GetHop() const;
        std::shared_ptr<PhyTopo::ConnInterface> GetSourceIFace();
        std::shared_ptr<PhyTopo::ConnInterface> GetTargetIFace();
        std::shared_ptr<PhyTopo::Node> GetSourceNode();
        std::shared_ptr<PhyTopo::Node> GetTargetNode();
        std::string Describe() const;

    private:
        std::shared_ptr<PhyTopo::ConnInterface> sourceIface{nullptr};
        std::shared_ptr<PhyTopo::ConnInterface> targetIface{nullptr}; // 如果target为Fabric节点，则为空
        std::shared_ptr<PhyTopo::Node> source{nullptr};
        std::shared_ptr<PhyTopo::Node> target{nullptr};
        std::set<Hccl::LinkProtocol> linkProtocols{};
        Hccl::LinkType linkType{};
        Hccl::LinkDirection direction{Hccl::LinkDirection::BOTH};
        Hccl::TopoType topoType{Hccl::TopoType::CLOS};
        u32 topoInstId{0};
        u32 hop{1};
    };

    HcclResult AddTopoGraph(const u32 netLayer, std::shared_ptr<Graph<PhyTopo::Node, PhyTopo::Link>> topo);
    std::shared_ptr<Graph<PhyTopo::Node, PhyTopo::Link>> GetTopoGraph(const u32 netLayer) const;
    void InitFinish();
    bool IsInitFinished() const;
    void Clear();
    void Dump() const;
    bool IsNetLayerExisted(const u32 netLayer) const;

private:
    std::unordered_map<u32, std::shared_ptr<Graph<PhyTopo::Node, PhyTopo::Link>>> topos;
    bool initFlag{false};
};
} // namespace hccl

#endif // PHY_TOPO_H
