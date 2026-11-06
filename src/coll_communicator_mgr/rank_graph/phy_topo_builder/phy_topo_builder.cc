/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "phy_topo_builder.h"
#include "log.h"
#include "json_parser.h"
#include "exception_util.h"
#include "null_ptr_exception.h"
#include "invalid_params_exception.h"
#include "adapter_error_manager_pub.h"
#include "common/hccl_log_keywords_pub.h"

namespace Hccl {

PhyTopoBuilder& PhyTopoBuilder::GetInstance()
{
    static PhyTopoBuilder phyTopoBuilder;
    return phyTopoBuilder;
}

// 根据nodeType和localId创建节点
std::shared_ptr<PhyTopo::Node> CreateNode(const PhyTopo::Node::NodeType nodeType, const LocalId localId)
{
    if (nodeType == PhyTopo::Node::NodeType::PEER) {
        return std::make_shared<PhyTopo::Peer>(localId);
    } else {
        return std::make_shared<PhyTopo::Fabric>();
    }
}

// 判断链路协议是否全为UB系协议(含PCIE/ROCE等非UB协议时返回false, 保护pciesw等混合机型)
static bool IsAllUbProtocol(const std::set<LinkProtocol>& protocols)
{
    static const std::set<LinkProtocol> ubProtocols
        = {LinkProtocol::UB_CTP, LinkProtocol::UB_TP, LinkProtocol::UB_MEM, LinkProtocol::UBOE, LinkProtocol::UB_RTP};
    if (protocols.empty()) {
        return false;
    }
    for (const auto& protocol : protocols) {
        if (ubProtocols.count(protocol) == 0) {
            return false;
        }
    }
    return true;
}

// 拓扑文件名即机型标识: 两条topoPath获取路径(rootinfo的topo_file_path/TopoAddrInfoGetTopoFilePath)
// 最终都经TopoGetFilePath(mainboard_id)生成, 仅350L(mainboard 0x44)对应atlas_850_3.json;
// 8PMESH系(0x25等)对应atlas_850_1/2.json(原生分层), 550EL对应atlas_550EL_*.json, 标卡对应atlas_350_*.json
static constexpr char UBX_350L_TOPO_FILE_NAME[] = "atlas_850_3.json";

// 判断topoPath是否为350L机型的标准拓扑文件(取最后一段文件名比对, 与driver安装路径无关)
// 兼容三种路径形态: '/'分隔的绝对路径(生产常规形态, 由TopoGetFilePath拼接)、
// 无分隔符的裸文件名相对路径、'\'分隔路径(防御性兼容, 生产Linux环境不会出现)
static bool IsUbxClosSplitMachine(const std::string& topoPath)
{
    const size_t pos = topoPath.find_last_of("/\\");
    const std::string fileName = (pos == std::string::npos) ? topoPath : topoPath.substr(pos + 1);
    return fileName == UBX_350L_TOPO_FILE_NAME;
}

// UBX 350L机型(仅此机型, 见IsUbxClosSplitMachine)的topo.json未拆层:
// DEVICE位置的UB CLOS上交换边与mesh直连边混在netLayer=0。
// rootinfo侧已拆层(level0=mesh, level1=clos)时, 在内存中将这批边重分桶到netLayer=1, 与rootinfo对齐,
// 避免CheckNetLayerFromPhyTopo因物理图缺少netLayer=1而报错。等价于手工将topo.json中对应边的
// net_layer由0改为1, 但不依赖设备侧文件变更。
// 返回值: 是否发生了350L拆层复制(供Build置位ubxClosSplit_标志, 自由函数无法访问类私有成员)
static bool SplitUbClosEdgesToLayer1(const std::string& topoPath, std::shared_ptr<TopoInfo>& topoInfo)
{
    if (!IsUbxClosSplitMachine(topoPath)) {
        return false; // 仅350L机型执行拆层复制, 其他机型拓扑保持原状
    }
    if (topoInfo->edges.count(1) > 0 || topoInfo->edges.count(0) == 0) {
        return false; // topo.json已拆层或无layer0, 无需处理
    }
    std::vector<EdgeInfo> closEdges;
    for (auto& edge : topoInfo->edges[0]) {
        if (edge.linkType == LinkType::PEER2NET && edge.position == AddrPosition::DEVICE
            && edge.topoType == TopoType::CLOS && IsAllUbProtocol(edge.protocols)) {
            edge.netLayer = 1;
            closEdges.emplace_back(edge);
        }
    }
    if (closEdges.empty()) {
        return false; // layer0中无UB CLOS边, 保持原状
    }
    // 复制而非搬移: graph(0)保留clos边副本, 供对称场景的rank graph按基线单层形态建fabric链路;
    // graph(1)的副本供非对称拆层形态使用。消费侧(rank_graph_builder)按对称性决定用哪份。
    HCCL_INFO(
        "[PhyTopoBuilder] topo.json is not layered, copy [%zu] UB CLOS edges to netLayer[1] (kept in netLayer[0]).",
        closEdges.size());
    topoInfo->edges[1] = closEdges;
    return true; // 标记350L拆层形态, 供rank graph侧RestoreSymmetricLevel0View判别机型
}

HcclResult PhyTopoBuilder::Build(const std::string& topoPath)
{
    std::lock_guard<std::mutex> lock(phyTopoMutex);

    if (PhyTopo::GetInstance()->IsInitFinished()) {
        HCCL_INFO("PhyTopo has been initialized and does not need to be rebuilt");
        return HCCL_SUCCESS;
    }

    if (topoPath.empty()) {
        RPT_INPUT_ERR(
            true, "EI0004", std::vector<std::string>({"ranktable_path", "error_reason"}),
            std::vector<std::string>({"<empty>", "The topo JSON file path is empty."}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] topo path %s is not a valid real path.", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CONFIG.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), topoPath.c_str());
        HCCL_ERROR("[%s] Topo path is empty.", __func__);
        return HCCL_E_PARA;
    }

    HCCL_DEBUG("[PhyTopoBuilder::%s]Start to build physic topo.", __func__);

    std::shared_ptr<TopoInfo> topoInfo = nullptr;
    TRY_CATCH_RETURN(topoInfo = LoadTopoInfo(topoPath););
    if (SplitUbClosEdgesToLayer1(topoPath, topoInfo)) {
        ubxClosSplit_ = true; // 标记350L拆层形态, 供rank graph侧RestoreSymmetricLevel0View判别机型
    }
    for (const auto& iter : topoInfo->edges) {
        auto netLayer = iter.first;
        std::shared_ptr<Graph<PhyTopo::Node, PhyTopo::Link>> graph = nullptr;
        TRY_CATCH_RETURN(graph = CreateGraph(iter.second););
        CHK_RET(PhyTopo::GetInstance()->AddTopoGraph(netLayer, graph));
        HCCL_DEBUG("[PhyTopoBuilder::%s]Build netLayer[%u] topo graph success.", __func__, netLayer);
    }

    auto graph = PhyTopo::GetInstance()->GetTopoGraph(0);
    if (graph == nullptr) {
        HCCL_INFO("[PhyTopoBuilder::%s] layer0 graph is nullptr, build now", __func__);
        graph = make_shared<Graph<PhyTopo::Node, PhyTopo::Link>>();
        CHK_RET(PhyTopo::GetInstance()->AddTopoGraph(0, graph));
    }
    for (const auto& iter : topoInfo->peers) {
        if (!graph->HasNode(iter.localId)) {
            auto node = CreateNode(PhyTopo::Node::NodeType::PEER, iter.localId);
            graph->AddNode(iter.localId, node);
        }
    }

    PhyTopo::GetInstance()->InitFinish();

    HCCL_DEBUG("[PhyTopoBuilder::%s]build physic topo success.", __func__);
    PhyTopo::GetInstance()->Dump();
    return HCCL_SUCCESS;
}

std::shared_ptr<TopoInfo> PhyTopoBuilder::LoadTopoInfo(const std::string& topoPath)
{
    std::shared_ptr<TopoInfo> topoInfo = std::make_shared<TopoInfo>();
    // 检查是否为非法路径以及size的大小。
    struct stat fileStat;
    if (stat(topoPath.c_str(), &fileStat) != 0) {
        RPT_INPUT_ERR(
            true, "EI0004", std::vector<std::string>({"ranktable_path", "error_reason"}),
            std::vector<std::string>({topoPath, "Failed to get the topo JSON file status."}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] Get file stat failed, file path:%s, errno:%d, error: %s",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_CONFIG.c_str(),
            HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), topoPath.c_str(), errno, strerror(errno));
        THROW<InvalidParamsException>(
            "[PhyTopoBuilder][LoadTopoInfo]Get file stat failed, file path:%s, errno:%d, error: %s", topoPath.c_str(),
            errno, strerror(errno));
    }

    u64 topoFileSize = static_cast<u64>(fileStat.st_size);
    if (topoFileSize > SUPPORT_MAX_TOPOFILE_SIZE || topoFileSize <= 0) {
        RPT_INPUT_ERR(
            true, "EI0004", std::vector<std::string>({"ranktable_path", "error_reason"}),
            std::vector<std::string>({topoPath, "The topo JSON file is empty or exceeds the maximum supported size."}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] topoFileSize size: %llu exceeds max allowed size (%u bytes)",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_CONFIG.c_str(),
            HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), topoFileSize, SUPPORT_MAX_TOPOFILE_SIZE);
        THROW<InvalidParamsException>(StringFormat(
            "[PhyTopoBuilder][LoadTopoInfo]file %s size (%llu bytes) exceeds max allowed size (%u bytes)",
            topoPath.c_str(), topoFileSize, SUPPORT_MAX_TOPOFILE_SIZE));
    }

    JsonParser topoParser;
    HcclResult parseRet = topoParser.ParseFile(topoPath, *topoInfo);
    if (parseRet != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] ParseFile failed, ret[%d]", __func__, parseRet);
        THROW<InvalidParamsException>(StringFormat("[%s] ParseFile failed", __func__));
    }
    topoInfo_ = topoInfo;
    return topoInfo;
}

shared_ptr<PhyTopo::Link> CreatePeer2PeerLink(const LinkParams& params)
{
    LinkType linkType = LinkType::PEER2PEER;
    auto srcIface
        = std::make_shared<PhyTopo::ConnInterface>(params.localAPorts, params.position, linkType, params.protocols);
    auto dstIface
        = std::make_shared<PhyTopo::ConnInterface>(params.localBPorts, params.position, linkType, params.protocols);
    PhyTopo::LinkAttributes linkAttrs;
    linkAttrs.linktype = linkType;
    linkAttrs.protocols = params.protocols;
    auto link = std::make_shared<PhyTopo::Link>(
        params.srcNode, params.dstNode, linkAttrs, params.topoType, params.topoInstanceId);
    link->SetSourceIface(srcIface);
    params.srcNode->AddConnInterface(srcIface);
    link->SetTargetIface(dstIface);
    params.dstNode->AddConnInterface(dstIface);
    return {link};
}

shared_ptr<PhyTopo::Link> CreatePeer2NetLink(const LinkParams& params)
{
    LinkType linkType = LinkType::PEER2NET;
    auto srcIface
        = std::make_shared<PhyTopo::ConnInterface>(params.localAPorts, params.position, linkType, params.protocols);
    // fabric 没有ports
    std::set<std::string> ports{};
    auto dstIface = std::make_shared<PhyTopo::ConnInterface>(ports, params.position, linkType, params.protocols);
    PhyTopo::LinkAttributes linkAttrs;
    linkAttrs.linktype = linkType;
    linkAttrs.protocols = params.protocols;
    auto link = std::make_shared<PhyTopo::Link>(
        params.srcNode, params.dstNode, linkAttrs, params.topoType, params.topoInstanceId);
    link->SetSourceIface(srcIface);
    params.srcNode->AddConnInterface(srcIface);
    link->SetTargetIface(dstIface);
    params.dstNode->AddConnInterface(dstIface);
    return {link};
}

// 根据链路类型建链
shared_ptr<PhyTopo::Link> CreateLink(const LinkType linkType, const LinkParams& params)
{
    if (linkType == LinkType::PEER2PEER) {
        return CreatePeer2PeerLink(params);
    } else {
        return CreatePeer2NetLink(params);
    }
}

NodeId GetNodeId(const PhyTopo::Node::NodeType nodeType, LocalId localId)
{
    switch (nodeType) {
        case PhyTopo::Node::NodeType::PEER:
            return PhyTopo::Peer::GetId(localId);
        case PhyTopo::Node::NodeType::FABRIC:
            return PhyTopo::Fabric::GetId();
        default:
            THROW<InvalidParamsException>(StringFormat("[PhyTopoBuilder]Invalid NodeType."));
            return 0;
    }
}

std::shared_ptr<Graph<PhyTopo::Node, PhyTopo::Link>>
PhyTopoBuilder::CreateGraph(const std::vector<EdgeInfo>& edges) const
{
    auto graph = std::make_shared<Graph<PhyTopo::Node, PhyTopo::Link>>();

    for (const auto& edgeInfo : edges) {
        std::shared_ptr<PhyTopo::Node> nodeA;
        std::shared_ptr<PhyTopo::Node> nodeB;

        // 生成 nodeAId，localA 始终为 PEER 类型
        NodeId nodeAId = GetNodeId(PhyTopo::Node::NodeType::PEER, edgeInfo.localA);
        // 获取或创建 nodeA
        if (!graph->HasNode(nodeAId)) {
            nodeA = CreateNode(PhyTopo::Node::NodeType::PEER, edgeInfo.localA);
            graph->AddNode(nodeAId, nodeA);
            HCCL_DEBUG("[PhyTopoBuilder::%s] Add node[%llu] success.", __func__, nodeAId);
        } else {
            // 使用 TraverseNode 查找 nodeA
            graph->TraverseNode([&nodeA, nodeAId](NodeId id, const std::shared_ptr<PhyTopo::Node>& n) {
                if (id == nodeAId) {
                    nodeA = n;
                }
            });
            if (nodeA == nullptr) {
                HCCL_ERROR("[PhyTopoBuilder::CreateGraph] nodeAId [%llu] not found.", nodeAId);
                THROW<NullPtrException>(
                    StringFormat("[PhyTopoBuilder::CreateGraph] Unable to find node [%llu].", nodeAId));
            }
        }

        // 根据 linkType 生成 nodeBId
        PhyTopo::Node::NodeType nodeBType;
        if (edgeInfo.linkType == LinkType::PEER2PEER) {
            nodeBType = PhyTopo::Node::NodeType::PEER;
        } else {
            nodeBType = PhyTopo::Node::NodeType::FABRIC;
        }

        NodeId nodeBId = GetNodeId(nodeBType, edgeInfo.localB);
        // 获取或创建 nodeB
        if (!graph->HasNode(nodeBId)) {
            nodeB = CreateNode(nodeBType, edgeInfo.localB);
            graph->AddNode(nodeBId, nodeB);
            HCCL_DEBUG("[PhyTopoBuilder::%s] Add node[%llu] success.", __func__, nodeBId);
        } else {
            // 使用 TraverseNode 查找 nodeB
            graph->TraverseNode([&nodeB, nodeBId](NodeId id, const std::shared_ptr<PhyTopo::Node>& n) {
                if (id == nodeBId) {
                    nodeB = n;
                }
            });
            if (nodeB == nullptr) {
                HCCL_ERROR("[PhyTopoBuilder::CreateGraph] nodeBId [%llu] not found.", nodeBId);
                THROW<NullPtrException>(
                    StringFormat("[PhyTopoBuilder::CreateGraph] Unable to find node [%llu].", nodeBId));
            }
        }

        // 双向分别创建link
        LinkParams abLinkParams{
            nodeA,
            nodeB,
            edgeInfo.localAPorts,
            edgeInfo.localBPorts,
            edgeInfo.position,
            edgeInfo.topoType,
            edgeInfo.topoInstId,
            edgeInfo.protocols};
        auto abLinks = CreateLink(edgeInfo.linkType, abLinkParams);
        graph->AddEdge(nodeAId, nodeBId, abLinks);
        LinkParams baLinkParams{
            nodeB,
            nodeA,
            edgeInfo.localBPorts,
            edgeInfo.localAPorts,
            edgeInfo.position,
            edgeInfo.topoType,
            edgeInfo.topoInstId,
            edgeInfo.protocols};
        auto baLinks = CreateLink(edgeInfo.linkType, baLinkParams);
        graph->AddEdge(nodeBId, nodeAId, baLinks);
    }

    return graph;
}

HcclResult PhyTopoBuilder::RecoverBuild(const TopoInfo& topoInfo)
{
    std::lock_guard<std::mutex> lock(phyTopoMutex);

    // PhyTopo::GetInstance()->ResetInstance();
    if (PhyTopo::GetInstance()->IsInitFinished()) {
        return HCCL_SUCCESS;
    }
    // 根据topoInfo，按netLayer构造Graph
    for (const auto& iter : topoInfo.edges) {
        auto netLayer = iter.first;
        std::shared_ptr<Graph<PhyTopo::Node, PhyTopo::Link>> graph = nullptr;
        TRY_CATCH_RETURN(graph = CreateGraph(iter.second););
        if (PhyTopo::GetInstance()->AddTopoGraph(netLayer, graph) != HCCL_SUCCESS) {
            HCCL_ERROR("[%s] AddTopoGraph failed, netLayer[%u]", __func__, netLayer);
            return HCCL_E_INTERNAL;
        }
        HCCL_DEBUG("[PhyTopoBuilder::%s]Build netLayer[%u] topo graph success.", __func__, netLayer);
    }

    PhyTopo::GetInstance()->InitFinish();

    HCCL_DEBUG("[PhyTopoBuilder::%s]build physic topo success.", __func__);
    PhyTopo::GetInstance()->Dump();
    return HCCL_SUCCESS;
}

std::shared_ptr<TopoInfo> PhyTopoBuilder::GetTopoInfo() const { return topoInfo_; }

} // namespace Hccl
