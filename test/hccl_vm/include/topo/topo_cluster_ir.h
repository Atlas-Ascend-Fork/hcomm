/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TOPO_CLUSTER_IR_H
#define TOPO_CLUSTER_IR_H

#include <map>
#include <string>
#include <tuple>
#include <vector>

enum class LinkType { UNKNOWN, PEER2PEER, PEER2NET };

enum class TopoType { UNKNOWN, MESH_1D, MESH_2D, CLOS, RING };

enum class Protocol { UNKNOWN, UB_CTP, UB_MEM, UB_TP, UBG, UBOE, ROCE };

enum class Position { UNKNOWN, DEVICE, HOST };

enum class SimDevType { UNKNOWN, DEV_TYPE_950, DEV_TYPE_910B, DEV_TYPE_910C, DEV_TYPE_910D };

enum class ParseStatus { OK, FILE_NOT_FOUND, PARSE_ERROR, INVALID_FORMAT };

LinkType ParseLinkType(const std::string& str);
std::string LinkTypeToString(LinkType type);

TopoType ParseTopoType(const std::string& str);
std::string TopoTypeToString(TopoType type);

Protocol ParseProtocol(const std::string& str);
std::string ProtocolToString(Protocol proto);

Position ParsePosition(const std::string& str);
std::string PositionToString(Position pos);

SimDevType ParseDevType(const std::string& str);
std::string DevTypeToString(SimDevType type);

struct Port {
    std::string portId;
    int dieId;
    int layer;
    std::string eid;
    LinkType linkType;
    std::string planeId;
    std::vector<Protocol> protocols;
    TopoType topoType;
    int topoInstanceId;
    Position position;
    std::string netInstanceId;

    Port()
        : portId(""),
          dieId(-1),
          layer(0),
          eid(""),
          linkType(LinkType::UNKNOWN),
          planeId(""),
          protocols(),
          topoType(TopoType::UNKNOWN),
          topoInstanceId(0),
          position(Position::UNKNOWN),
          netInstanceId("")
    {}
};

struct Device {
    int localId;
    int deviceId;
    SimDevType devType;
    std::string socVersion;
    std::vector<Port> ports;

    Device() : localId(-1), deviceId(-1), devType(SimDevType::UNKNOWN), socVersion(""), ports() {}

    std::vector<const Port*> GetPortsByLayer(int layer) const;
    std::vector<const Port*> GetPortsByLinkType(LinkType type) const;
    std::vector<const Port*> GetPortsByPlane(const std::string& pid) const;
    const Port* GetPort(const std::string& pid) const;
};

struct LinkPortRef {
    int localId;
    std::string eid;
    std::vector<std::string> portIds;

    LinkPortRef() : localId(-1), portIds() {}
};

struct Link {
    std::string linkId;
    int netLayer;
    LinkType linkType;
    TopoType topoType;
    int topoInstanceId;
    std::string topoAttr;
    LinkPortRef sideA;
    LinkPortRef sideB;
    std::vector<Protocol> protocols;
    Position position;

    Link()
        : linkId(""),
          netLayer(0),
          linkType(LinkType::UNKNOWN),
          topoType(TopoType::UNKNOWN),
          topoInstanceId(0),
          topoAttr(""),
          sideA(),
          sideB(),
          protocols(),
          position(Position::UNKNOWN)
    {}

    bool IsPeer2Peer() const { return linkType == LinkType::PEER2PEER; }
    bool IsPeer2Net() const { return linkType == LinkType::PEER2NET; }
};

// Board —— 拓扑叶子节点(承载host IP与一份hccl_rootinfo.json; IR中devices/links为board粒度)。
// server形态(3层: server → 超节点 → cluster): 一个Server只有一个Board(boardId==0), 即server自身,
//   topo.json与rootinfo.json均在server目录;
// POD形态(4层: board → pod → 超节点 → cluster): 一个Server(=POD)下有多个Board,
//   每board是一个独立host节点, board内卡的物理id(localId)从0开始编号(如0~7),
//   不同board之间localId可以重复, 全局定位需结合(superPodId, serverId, boardId)三元组。
//   topo.json为POD级单份(位于server目录), 其local_id为POD全局id
//   (= boardId * 每board卡数 + board内物理id), 解析时按board切分并重映射回board内物理id,
//   故IR中Board.devices/links的localId口径与rootinfo一致(均为board内物理id)。
struct Board {
    int boardId;
    std::string hardwareType;
    // 端口类型: "x2"/"x4" (servers_info.json port_type字段, 缺省"x4");
    // 仅影响性能仿真端口带宽计算(x2带宽为x4的一半), 不影响拓扑结构
    std::string portType;
    std::string netInstanceId;
    std::string hostEid;
    std::vector<Device> devices;
    std::vector<Link> links;

    Board() : boardId(0), hardwareType(""), portType("x4"), netInstanceId(""), hostEid(""), devices(), links() {}

    const Device* GetDevice(int localId) const;
    const Link* GetLink(const std::string& lid) const;
    std::vector<const Link*> GetLinksByLayer(int layer) const;
    std::vector<const Link*> GetLinksByType(LinkType type) const;
    std::vector<const Link*> GetDeviceLinks(int localId) const;
    int GetDeviceCount() const { return static_cast<int>(devices.size()); }
    int GetLinkCount() const { return static_cast<int>(links.size()); }
};

// Server —— server形态下为叶子节点(boards.size()==1); POD形态下为POD分组节点(仅承载boards)。
// hardwareType/portType/netInstanceId/hostEid/devices/links等"每叶子节点一份"的属性均下沉到Board。
struct Server {
    int serverId;
    std::vector<Board> boards;

    Server() : serverId(-1), boards() {}

    const Board* GetBoard(int boardId) const;
    Board* GetMutableBoard(int boardId);
    // 缺省board: server形态下即唯一的boards[0]; 便于既有调用方按"单叶子节点"语义访问
    const Board* GetDefaultBoard() const { return boards.empty() ? nullptr : &boards.front(); }
    Board* GetMutableDefaultBoard() { return boards.empty() ? nullptr : &boards.front(); }
    int GetBoardCount() const { return static_cast<int>(boards.size()); }
    int GetDeviceCount() const;
    int GetLinkCount() const;
    bool IsPodForm() const { return boards.size() > 1; }
};

struct SuperPod {
    int superPodId;
    std::vector<Server> servers;

    SuperPod() : superPodId(-1), servers() {}

    const Server* GetServer(int serverId) const;
    int GetServerCount() const { return static_cast<int>(servers.size()); }
    int GetTotalDeviceCount() const;
    int GetTotalBoardCount() const;
};

struct Network {
    std::string version;
    // 叶子节点(host节点)总数: servers_info.json的server_count字段
    // (server形态=server数; POD形态=POD数×board数)
    int serverCount;
    std::vector<SuperPod> superPods;
    // deviceId → [(superPodId, serverId, boardId)]: deviceId仅叶子节点内唯一, 故值为三元组列表
    std::map<int, std::vector<std::tuple<int, int, int>>> deviceIndex;
    // eid → [(superPodId, serverId, boardId, localId)]: EID全局唯一
    std::map<std::string, std::vector<std::tuple<int, int, int, int>>> eidIndex;

    Network() : version(""), serverCount(0), superPods(), deviceIndex(), eidIndex() {}

    const SuperPod* GetSuperPod(int superPodId) const;
    const Server* GetServer(int superPodId, int serverId) const;
    const Board* GetBoard(int superPodId, int serverId, int boardId) const;
    const Device* GetDeviceByGlobalId(int deviceId) const;
    std::vector<const Device*> GetAllDevices() const;
    std::vector<const Port*> GetAllPortsByLayer(int layer) const;
    std::vector<const Port*> GetAllPortsByLinkType(LinkType type) const;
    std::vector<const Link*> GetAllLinksByType(LinkType type) const;
    int GetSuperPodCount() const { return static_cast<int>(superPods.size()); }
    int GetTotalDeviceCount() const;
    int GetTotalLinkCount() const;
    int GetTotalBoardCount() const;

    ParseStatus BuildIndexes();
};

#endif // TOPO_CLUSTER_IR_H
