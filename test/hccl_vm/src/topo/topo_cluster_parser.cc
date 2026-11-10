/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cctype>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <dirent.h>
#include <fstream>
#include <json.hpp>
#include <map>
#include <set>
#include <sstream>
#include <sys/stat.h>
#include <tuple>
#include <vector>

#include "topo_cluster_parser.h"
#include "sim_log.h"

using json = nlohmann::json;

ClusterTopoParser::ClusterTopoParser() {}

ClusterTopoParser::~ClusterTopoParser() {}

namespace {
// 从集群目录名解析层级id: 取"前缀之后的尾部连续数字"。
// 目录命名约定为 superpod<N> / server<N> / board<N> (如 superpod0、server10、board7)。
//
// 注意: 不能用"排序下标"充当id —— 目录按字符串排序时 "server10" < "server2",
// 会让 server10 拿到 id=2, 造成"目录 ↔ Server.server_id ↔ servers_info条目(host IP) ↔
// rootinfo内net_instance_id/EID网段"四者错位(单超节点内server数>=11即触发)。
// 亦不能用 find('_') 截取(旧目录名 Ascend950_superpod0 的遗留写法), 现网目录名无下划线。
bool ParseTrailingId(const std::string& dirName, const std::string& prefix, int& id)
{
    // 前缀匹配(大小写不敏感): 兼容 superpod0 / SuperPod0 / server0 / Server0 / board0
    if (dirName.size() <= prefix.size()) {
        return false;
    }
    for (size_t i = 0; i < prefix.size(); ++i) {
        if (std::tolower(static_cast<unsigned char>(dirName[i]))
            != std::tolower(static_cast<unsigned char>(prefix[i]))) {
            return false;
        }
    }
    std::string numStr = dirName.substr(prefix.size());
    for (char c : numStr) {
        if (!std::isdigit(static_cast<unsigned char>(c))) {
            return false;
        }
    }
    try {
        id = std::stoi(numStr);
    } catch (...) {
        HCCL_VM_ERROR("Convert numStr[{}] to int error", numStr.c_str());
        return false;
    }
    return true;
}

// 按目录名解析出的id升序排序(而非按路径字符串字典序), 保证
// superPods[]/servers[]/boards[] 的向量顺序与id升序一致
void SortDirsById(const std::vector<std::string>& dirs, const std::string& prefix, std::vector<std::string>& sorted)
{
    std::vector<std::pair<int, std::string>> items;
    items.reserve(dirs.size());
    for (const auto& dir : dirs) {
        std::string dirName = dir.substr(dir.find_last_of('/') + 1);
        int id = -1;
        if (!ParseTrailingId(dirName, prefix, id)) {
            HCCL_VM_WARN(
                "cannot parse id from dir name[{}](prefix[{}]), fallback to sort by name", dirName.c_str(),
                prefix.c_str());
            id = INT32_MAX; // 无法解析id的目录排在最后, 保持相对顺序由stable_sort保证
        }
        items.emplace_back(id, dir);
    }
    std::stable_sort(items.begin(), items.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.first != rhs.first) {
            return lhs.first < rhs.first;
        }
        return lhs.second < rhs.second;
    });
    sorted.clear();
    sorted.reserve(items.size());
    for (auto& item : items) {
        sorted.push_back(std::move(item.second));
    }
}

// 收集目录下匹配指定前缀的子目录(如 superpod*/server*/board*)
void CollectSubDirs(const std::string& parentDir, const std::string& prefix, std::vector<std::string>& out)
{
    out.clear();
    DIR* dir = opendir(parentDir.c_str());
    if (dir == nullptr) {
        return;
    }
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;
        if (name == "." || name == "..") {
            continue;
        }
        std::string fullPath = parentDir + "/" + name;
        struct stat st;
        if (stat(fullPath.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
            continue;
        }
        std::string lowerName = name;
        std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        std::string lowerPrefix = prefix;
        std::transform(lowerPrefix.begin(), lowerPrefix.end(), lowerPrefix.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        if (lowerName.rfind(lowerPrefix, 0) == 0) {
            out.push_back(fullPath);
        }
    }
    closedir(dir);
}

// 按统一规则生成linkId (ParseTopoJson解析与ExtractBoardFromPodTopo重映射local_id后共用,
// 保证POD级topo切分到board后linkId口径与"每board一份topo.json"时代一致)
std::string BuildLinkId(const Link& link)
{
    std::ostringstream linkIdBuilder;
    linkIdBuilder << "L" << link.netLayer << "_" << (link.IsPeer2Peer() ? "P2P" : "P2N") << "_" << link.sideA.localId
                  << "_" << link.sideB.localId << "_I" << link.topoInstanceId;
    return linkIdBuilder.str();
}
} // namespace

ParseStatus ClusterTopoParser::ParseClusterDir(const std::string& clusterDir)
{
    network_ = Network();

    ParseStatus status = ScanSuperPodDirs(clusterDir);
    if (status != ParseStatus::OK) {
        return status;
    }

    std::string serversInfoPath = clusterDir + "/servers_info.json";
    ParseStatus siStatus = ParseServersInfoJson(serversInfoPath);
    if (siStatus != ParseStatus::OK) {
        HCCL_VM_WARN("failed to parse servers_info.json");
    }

    ParseStatus vcStatus = ValidateClusterConsistency();
    if (vcStatus != ParseStatus::OK) {
        return vcStatus;
    }

    return network_.BuildIndexes();
}

ParseStatus ClusterTopoParser::ScanSuperPodDirs(const std::string& clusterDir)
{
    DIR* dir = opendir(clusterDir.c_str());
    if (!dir) {
        return ParseStatus::FILE_NOT_FOUND;
    }
    closedir(dir);

    std::vector<std::string> superpodDirs;
    CollectSubDirs(clusterDir, "superpod", superpodDirs);
    std::vector<std::string> sortedSuperpodDirs;
    SortDirsById(superpodDirs, "superpod", sortedSuperpodDirs);

    for (size_t i = 0; i < sortedSuperpodDirs.size(); ++i) {
        SuperPod superpod;
        // id取自目录名尾部数字; 解析失败时回退遍历下标(已在SortDirsById中告警)
        superpod.superPodId = static_cast<int>(i);
        std::string dirName = sortedSuperpodDirs[i].substr(sortedSuperpodDirs[i].find_last_of('/') + 1);
        int parsedId = -1;
        if (ParseTrailingId(dirName, "superpod", parsedId)) {
            superpod.superPodId = parsedId;
        }

        ParseStatus status = ScanServerDirs(sortedSuperpodDirs[i], superpod);
        if (status != ParseStatus::OK) {
            return status;
        }

        network_.superPods.push_back(superpod);
    }

    if (!network_.superPods.empty() && network_.version.empty()) {
        network_.version = "2.0";
    }

    return ParseStatus::OK;
}

ParseStatus ClusterTopoParser::ScanServerDirs(const std::string& superpodDir, SuperPod& superpod)
{
    std::vector<std::string> serverDirs;
    CollectSubDirs(superpodDir, "server", serverDirs);
    if (serverDirs.empty()) {
        // 目录下无server子目录时, 认为superpodDir本身打不开或为空
        DIR* dir = opendir(superpodDir.c_str());
        if (!dir) {
            return ParseStatus::FILE_NOT_FOUND;
        }
        closedir(dir);
    }
    std::vector<std::string> sortedServerDirs;
    SortDirsById(serverDirs, "server", sortedServerDirs);

    for (size_t i = 0; i < sortedServerDirs.size(); ++i) {
        Server server;
        server.serverId = static_cast<int>(i);
        std::string dirName = sortedServerDirs[i].substr(sortedServerDirs[i].find_last_of('/') + 1);
        int parsedId = -1;
        if (ParseTrailingId(dirName, "server", parsedId)) {
            server.serverId = parsedId;
        }

        ParseStatus status = ScanBoardDirs(sortedServerDirs[i], server);
        if (status != ParseStatus::OK) {
            return status;
        }

        superpod.servers.push_back(server);
    }

    return ParseStatus::OK;
}

ParseStatus ClusterTopoParser::ScanBoardDirs(const std::string& serverDir, Server& server)
{
    std::vector<std::string> boardDirs;
    CollectSubDirs(serverDir, "board", boardDirs);

    if (boardDirs.empty()) {
        // server形态(3层拓扑): server目录自身即叶子节点, 视为唯一的board0
        Board board;
        board.boardId = 0;
        ParseStatus status = ParseBoard(serverDir, board);
        if (status != ParseStatus::OK) {
            return status;
        }
        server.boards.push_back(std::move(board));
        return ParseStatus::OK;
    }

    // POD形态(4层拓扑): topo.json为POD级单份, 位于server目录,
    // 其peer_list/edge_list的local_id为POD全局id(= board_id * 每board卡数 + board内物理id);
    // 每个board{N}/子目录为一个叶子节点(独立host节点), 仅含rootinfo.json(board内物理id)。
    // 存量旧布局(board目录下各自一份topo.json)不再支持, 须重新生成集群产物
    std::string podTopoPath = FindFileBySuffix(serverDir, "topo");
    if (podTopoPath.empty()) {
        HCCL_VM_ERROR(
            "POD form requires a pod-level topo.json in server dir[{}], but none found "
            "(the legacy per-board topo.json layout is no longer supported), "
            "please regenerate the cluster topology",
            serverDir.c_str());
        return ParseStatus::FILE_NOT_FOUND;
    }
    Board podTopo;
    ParseStatus topoStatus = ParseTopoJson(podTopoPath, podTopo);
    if (topoStatus != ParseStatus::OK) {
        return topoStatus;
    }

    std::vector<std::string> sortedBoardDirs;
    SortDirsById(boardDirs, "board", sortedBoardDirs);

    // 每board卡数 = POD级topo的peer总数 / board目录数(均匀切分, 与生成侧规则一致)
    int boardNum = static_cast<int>(sortedBoardDirs.size());
    int peerCount = static_cast<int>(podTopo.devices.size());
    if (boardNum <= 0 || peerCount <= 0 || peerCount % boardNum != 0) {
        HCCL_VM_ERROR(
            "pod-level topo.json[{}] peer_count[{}] cannot be evenly split into [{:d}] board dirs", podTopoPath.c_str(),
            peerCount, boardNum);
        return ParseStatus::INVALID_FORMAT;
    }
    int devicesPerBoard = peerCount / boardNum;

    for (size_t i = 0; i < sortedBoardDirs.size(); ++i) {
        Board board;
        board.boardId = static_cast<int>(i);
        std::string dirName = sortedBoardDirs[i].substr(sortedBoardDirs[i].find_last_of('/') + 1);
        int parsedId = -1;
        if (ParseTrailingId(dirName, "board", parsedId)) {
            board.boardId = parsedId;
        }

        // POD级topo按local_id值域切分出本board的devices/links(重映射为board内物理id)
        ParseStatus status = ExtractBoardFromPodTopo(podTopo, board.boardId, devicesPerBoard, board);
        if (status != ParseStatus::OK) {
            return status;
        }

        // board目录下残留的topo.json为旧布局产物: 告警并忽略(以POD级topo.json为准)
        std::string staleTopoPath = FindFileBySuffix(sortedBoardDirs[i], "topo");
        if (!staleTopoPath.empty()) {
            HCCL_VM_WARN(
                "stale per-board topo.json[{}] ignored, pod-level topo.json[{}] is used instead", staleTopoPath.c_str(),
                podTopoPath.c_str());
        }

        std::string rootinfoPath = FindFileBySuffix(sortedBoardDirs[i], "rootinfo");
        if (rootinfoPath.empty()) {
            HCCL_VM_WARN("no rootinfo.json found in {}", sortedBoardDirs[i].c_str());
        } else {
            status = ParseRootinfoJson(rootinfoPath, board);
            if (status != ParseStatus::OK) {
                return status;
            }
            MergeTopoAndRootinfo(board);
        }

        server.boards.push_back(std::move(board));
    }

    return ParseStatus::OK;
}

ParseStatus
ClusterTopoParser::ExtractBoardFromPodTopo(const Board& podTopo, int boardId, int devicesPerBoard, Board& board)
{
    const int base = boardId * devicesPerBoard;
    const int end = base + devicesPerBoard;
    if (base < 0 || end > static_cast<int>(podTopo.devices.size())) {
        HCCL_VM_ERROR(
            "board[{:d}] local_id range [{:d}, {:d}) exceeds pod-level topo peer_count[{}]", boardId, base, end,
            podTopo.devices.size());
        return ParseStatus::INVALID_FORMAT;
    }

    board.hardwareType = podTopo.hardwareType;

    // devices: 截取本board值域, POD全局local_id重映射为board内物理id
    // (IR/DB的physical_id与rootinfo/ranktable的device_id均为board内物理id口径)
    for (const auto& dev : podTopo.devices) {
        if (dev.localId >= base && dev.localId < end) {
            Device boardDev = dev;
            boardDev.localId -= base;
            board.devices.push_back(boardDev);
        }
    }
    if (static_cast<int>(board.devices.size()) != devicesPerBoard) {
        HCCL_VM_ERROR(
            "board[{:d}] holds [{:d}] devices in pod-level topo, expect [{:d}] "
            "(local_id must cover [{:d}, {:d}))",
            boardId, board.devices.size(), devicesPerBoard, base, end);
        return ParseStatus::INVALID_FORMAT;
    }

    // links: PEER2NET边按local_a归属; PEER2PEER边为板内直连(两端同board)。
    // 跨board的PEER2PEER边生成侧不会产生(POD形态直连仅限板内), 防御性告警并
    // 归属sideA所在board(sideB保留POD全局id, 对端board不重复记录该边)
    for (const auto& link : podTopo.links) {
        const bool aInBoard = (link.sideA.localId >= base && link.sideA.localId < end);
        const bool bInBoard = (link.sideB.localId >= base && link.sideB.localId < end);
        if (!aInBoard && !bInBoard) {
            continue;
        }
        if (link.IsPeer2Peer() && aInBoard != bInBoard) {
            HCCL_VM_WARN(
                "cross-board PEER2PEER edge[{}] detected in pod-level topo, attributed to board[{:d}] of sideA",
                link.linkId.c_str(), boardId);
            if (!aInBoard) {
                continue;
            }
        }
        Link boardLink = link;
        if (aInBoard) {
            boardLink.sideA.localId -= base;
        }
        if (bInBoard) {
            boardLink.sideB.localId -= base;
        }
        boardLink.linkId = BuildLinkId(boardLink);
        board.links.push_back(boardLink);
    }

    return ParseStatus::OK;
}

ParseStatus ClusterTopoParser::ParseBoard(const std::string& leafDir, Board& board)
{
    std::string topoPath = FindFileBySuffix(leafDir, "topo");
    std::string rootinfoPath = FindFileBySuffix(leafDir, "rootinfo");

    if (topoPath.empty()) {
        HCCL_VM_WARN("no topo.json found in {}", leafDir.c_str());
    }
    if (rootinfoPath.empty()) {
        HCCL_VM_WARN("no rootinfo.json found in {}", leafDir.c_str());
    }

    if (!topoPath.empty()) {
        ParseStatus status = ParseTopoJson(topoPath, board);
        if (status != ParseStatus::OK) {
            return status;
        }
    }

    if (!rootinfoPath.empty()) {
        ParseStatus status = ParseRootinfoJson(rootinfoPath, board);
        if (status != ParseStatus::OK) {
            return status;
        }
    }

    if (!topoPath.empty() && !rootinfoPath.empty()) {
        MergeTopoAndRootinfo(board);
    }

    return ParseStatus::OK;
}

std::string ClusterTopoParser::FindFileBySuffix(const std::string& dir, const std::string& suffix)
{
    DIR* dp = opendir(dir.c_str());
    if (!dp) {
        return "";
    }

    struct dirent* entry;
    std::string result;

    while ((entry = readdir(dp)) != nullptr) {
        std::string name = entry->d_name;
        if (name.find("switch_description") != std::string::npos) {
            continue;
        }
        if (name.size() > 5 && name.substr(name.size() - 5) == ".json") {
            if (name.find(suffix) != std::string::npos) {
                result = dir + "/" + name;
                break;
            }
        }
    }
    closedir(dp);
    return result;
}

ParseStatus ClusterTopoParser::ParseTopoJson(const std::string& topoPath, Board& board)
{
    std::ifstream ifs(topoPath);
    if (!ifs.is_open()) {
        return ParseStatus::FILE_NOT_FOUND;
    }

    try {
        json j;
        ifs >> j;

        if (j.contains("version") && j["version"].is_string()) {
            network_.version = j["version"].get<std::string>();
        }

        if (j.contains("hardware_type") && j["hardware_type"].is_string()) {
            board.hardwareType = j["hardware_type"].get<std::string>();
        }

        if (j.contains("peer_list") && j["peer_list"].is_array()) {
            for (const auto& peer : j["peer_list"]) {
                if (peer.contains("local_id")) {
                    Device dev;
                    dev.localId = peer["local_id"].get<int>();
                    board.devices.push_back(dev);
                }
            }
        }

        if (j.contains("edge_list") && j["edge_list"].is_array()) {
            for (const auto& edge : j["edge_list"]) {
                Link link;

                if (edge.contains("net_layer")) {
                    link.netLayer = edge["net_layer"].get<int>();
                }
                if (edge.contains("link_type")) {
                    link.linkType = ParseLinkType(edge["link_type"].get<std::string>());
                }
                if (edge.contains("topo_type")) {
                    link.topoType = ParseTopoType(edge["topo_type"].get<std::string>());
                }
                if (edge.contains("topo_instance_id")) {
                    link.topoInstanceId = edge["topo_instance_id"].get<int>();
                }
                if (edge.contains("topo_attr")) {
                    link.topoAttr = edge["topo_attr"].get<std::string>();
                }

                if (edge.contains("local_a")) {
                    link.sideA.localId = edge["local_a"].get<int>();
                }
                if (edge.contains("local_a_ports") && edge["local_a_ports"].is_array()) {
                    for (const auto& p : edge["local_a_ports"]) {
                        link.sideA.portIds.push_back(p.get<std::string>());
                    }
                }

                if (edge.contains("local_b")) {
                    link.sideB.localId = edge["local_b"].get<int>();
                }
                if (edge.contains("local_b_ports") && edge["local_b_ports"].is_array()) {
                    for (const auto& p : edge["local_b_ports"]) {
                        link.sideB.portIds.push_back(p.get<std::string>());
                    }
                }

                if (edge.contains("protocols") && edge["protocols"].is_array()) {
                    for (const auto& proto : edge["protocols"]) {
                        link.protocols.push_back(ParseProtocol(proto.get<std::string>()));
                    }
                }

                if (edge.contains("position")) {
                    link.position = ParsePosition(edge["position"].get<std::string>());
                }

                link.linkId = BuildLinkId(link);

                board.links.push_back(link);
            }
        }
    } catch (const json::exception& e) {
        HCCL_VM_ERROR("JSON parse error in {}: {}", topoPath.c_str(), e.what());
        return ParseStatus::PARSE_ERROR;
    }

    return ParseStatus::OK;
}

ParseStatus ClusterTopoParser::ParseRootinfoJson(const std::string& rootinfoPath, Board& board)
{
    std::ifstream ifs(rootinfoPath);
    if (!ifs.is_open()) {
        return ParseStatus::FILE_NOT_FOUND;
    }

    try {
        json j;
        ifs >> j;

        if (j.contains("version") && j["version"].is_string() && network_.version.empty()) {
            network_.version = j["version"].get<std::string>();
        }

        if (j.contains("rank_list") && j["rank_list"].is_array()) {
            for (const auto& rank : j["rank_list"]) {
                int localId = -1;
                int deviceId = -1;

                if (rank.contains("local_id")) {
                    localId = rank["local_id"].get<int>();
                }
                if (rank.contains("device_id")) {
                    deviceId = rank["device_id"].get<int>();
                }

                Device* dev = nullptr;
                for (auto& d : board.devices) {
                    if (d.localId == localId) {
                        dev = &d;
                        break;
                    }
                }

                if (!dev) {
                    Device newDev;
                    newDev.localId = localId;
                    newDev.deviceId = deviceId;
                    board.devices.push_back(newDev);
                    dev = &board.devices.back();
                } else {
                    dev->deviceId = deviceId;
                }

                if (rank.contains("level_list") && rank["level_list"].is_array()) {
                    for (const auto& level : rank["level_list"]) {
                        int netLayer = 0;
                        std::string netInstanceId;

                        if (level.contains("net_layer")) {
                            netLayer = level["net_layer"].get<int>();
                        }
                        if (level.contains("net_instance_id")) {
                            netInstanceId = level["net_instance_id"].get<std::string>();
                        }

                        if (board.netInstanceId.empty() && !netInstanceId.empty()) {
                            board.netInstanceId = netInstanceId;
                        }

                        if (level.contains("rank_addr_list") && level["rank_addr_list"].is_array()) {
                            for (const auto& addr : level["rank_addr_list"]) {
                                std::string eid;
                                std::vector<std::string> portIds;
                                std::string planeId;

                                if (addr.contains("addr")) {
                                    eid = addr["addr"].get<std::string>();
                                }
                                if (addr.contains("ports") && addr["ports"].is_array()) {
                                    for (const auto& p : addr["ports"]) {
                                        portIds.push_back(p.get<std::string>());
                                    }
                                }
                                if (addr.contains("plane_id")) {
                                    planeId = addr["plane_id"].get<std::string>();
                                }

                                if (portIds.empty()) {
                                    Port port;
                                    port.layer = netLayer;
                                    port.eid = eid;
                                    port.planeId = planeId;
                                    port.netInstanceId = netInstanceId;
                                    dev->ports.push_back(port);
                                } else if (portIds.size() == 1) {
                                    Port port;
                                    port.layer = netLayer;
                                    port.eid = eid;
                                    port.portId = portIds[0];
                                    port.dieId = ExtractDieIdFromPortId(portIds[0]);
                                    port.planeId = planeId;
                                    port.netInstanceId = netInstanceId;
                                    dev->ports.push_back(port);
                                } else {
                                    for (const auto& pid : portIds) {
                                        Port port;
                                        port.layer = netLayer;
                                        port.eid = eid;
                                        port.portId = pid;
                                        port.dieId = ExtractDieIdFromPortId(pid);
                                        port.planeId = planeId;
                                        port.netInstanceId = netInstanceId;
                                        dev->ports.push_back(port);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    } catch (const json::exception& e) {
        HCCL_VM_ERROR("JSON parse error in {}: {}", rootinfoPath.c_str(), e.what());
        return ParseStatus::PARSE_ERROR;
    }

    return ParseStatus::OK;
}

void ClusterTopoParser::MergeTopoAndRootinfo(Board& board)
{
    std::map<int, std::map<std::string, std::string>> devicePortGroupEid;

    for (const auto& dev : board.devices) {
        std::map<std::string, std::string> portGroupEid;
        std::map<std::string, std::vector<std::string>> eidPortGroups;

        for (const auto& port : dev.ports) {
            if (!port.eid.empty() && !port.portId.empty()) {
                eidPortGroups[port.eid].push_back(port.portId);
            }
        }

        for (const auto& [eid, pids] : eidPortGroups) {
            std::string key;
            std::vector<std::string> sortedPids = pids;
            std::sort(sortedPids.begin(), sortedPids.end());
            for (const auto& pid : sortedPids) {
                if (!key.empty()) {
                    key += ",";
                }
                key += pid;
            }
            portGroupEid[key] = eid;
        }

        devicePortGroupEid[dev.localId] = portGroupEid;
    }

    for (auto& link : board.links) {
        auto fillEid = [&](LinkPortRef& side) {
            if (side.portIds.empty()) {
                return;
            }

            auto devIt = devicePortGroupEid.find(side.localId);
            if (devIt == devicePortGroupEid.end()) {
                return;
            }

            std::string key;
            std::vector<std::string> sortedPids = side.portIds;
            std::sort(sortedPids.begin(), sortedPids.end());
            for (const auto& pid : sortedPids) {
                if (!key.empty()) {
                    key += ",";
                }
                key += pid;
            }

            auto eidIt = devIt->second.find(key);
            if (eidIt != devIt->second.end()) {
                side.eid = eidIt->second;
            } else if (sortedPids.size() > 1) {
                // Fallback: subset matching for PEER2NET links.
                // When port_list (topo) differs from port_group (rootinfo),
                // the link's portIds may be a superset of a rootinfo portGroup.
                // Assign the first matching portGroup's EID.
                std::set<std::string> linkPortSet(sortedPids.begin(), sortedPids.end());
                for (const auto& [rootinfoKey, rootinfoEid] : devIt->second) {
                    std::istringstream iss(rootinfoKey);
                    std::string token;
                    bool allFound = true;
                    while (std::getline(iss, token, ',')) {
                        if (linkPortSet.find(token) == linkPortSet.end()) {
                            allFound = false;
                            break;
                        }
                    }
                    if (allFound) {
                        side.eid = rootinfoEid;
                        break;
                    }
                }
            }
        };

        fillEid(link.sideA);
        fillEid(link.sideB);
    }

    for (auto& dev : board.devices) {
        std::map<int, LinkType> layerLinkType;
        std::map<int, TopoType> layerTopoType;
        std::map<int, int> layerTopoInstanceId;
        std::map<int, Position> layerPosition;
        std::map<int, std::vector<Protocol>> layerProtocols;

        for (const auto& link : board.links) {
            bool matchA = (link.sideA.localId == dev.localId);
            bool matchB = (link.sideB.localId == dev.localId);
            if (!matchA && !matchB) {
                continue;
            }

            int layer = link.netLayer;
            if (layerLinkType.find(layer) == layerLinkType.end()) {
                layerLinkType[layer] = link.linkType;
            }
            if (layerTopoType.find(layer) == layerTopoType.end()) {
                layerTopoType[layer] = link.topoType;
            }
            if (layerTopoInstanceId.find(layer) == layerTopoInstanceId.end()) {
                layerTopoInstanceId[layer] = link.topoInstanceId;
            }
            if (layerPosition.find(layer) == layerPosition.end()) {
                layerPosition[layer] = link.position;
            }
            if (layerProtocols.find(layer) == layerProtocols.end() && !link.protocols.empty()) {
                layerProtocols[layer] = link.protocols;
            }
        }

        for (auto& port : dev.ports) {
            int layer = port.layer;

            bool exactMatch = false;
            for (const auto& link : board.links) {
                if (link.netLayer != layer) {
                    continue;
                }

                bool matchA = (link.sideA.localId == dev.localId);
                bool matchB = (link.sideB.localId == dev.localId);
                if (!matchA && !matchB) {
                    continue;
                }

                if (matchA) {
                    for (const auto& pId : link.sideA.portIds) {
                        if (pId == port.portId) {
                            port.linkType = link.linkType;
                            port.topoType = link.topoType;
                            port.topoInstanceId = link.topoInstanceId;
                            port.position = link.position;
                            if (port.protocols.empty()) {
                                port.protocols = link.protocols;
                            }
                            exactMatch = true;
                            break;
                        }
                    }
                }
                if (exactMatch) {
                    break;
                }

                if (matchB) {
                    for (const auto& pId : link.sideB.portIds) {
                        if (pId == port.portId) {
                            port.linkType = link.linkType;
                            port.topoType = link.topoType;
                            port.topoInstanceId = link.topoInstanceId;
                            port.position = link.position;
                            if (port.protocols.empty()) {
                                port.protocols = link.protocols;
                            }
                            exactMatch = true;
                            break;
                        }
                    }
                }
                if (exactMatch) {
                    break;
                }
            }

            if (!exactMatch) {
                if (layerLinkType.count(layer)) {
                    port.linkType = layerLinkType[layer];
                }
                if (layerTopoType.count(layer)) {
                    port.topoType = layerTopoType[layer];
                }
                if (layerTopoInstanceId.count(layer)) {
                    port.topoInstanceId = layerTopoInstanceId[layer];
                }
                if (layerPosition.count(layer)) {
                    port.position = layerPosition[layer];
                }
                if (layerProtocols.count(layer) && port.protocols.empty()) {
                    port.protocols = layerProtocols[layer];
                }
            }
        }

        std::set<std::string> existingPortIds;
        for (const auto& port : dev.ports) {
            existingPortIds.insert(port.portId);
        }

        for (const auto& link : board.links) {
            bool matchA = (link.sideA.localId == dev.localId);
            bool matchB = (link.sideB.localId == dev.localId);
            if (!matchA && !matchB) {
                continue;
            }

            const auto& portIds = matchA ? link.sideA.portIds : link.sideB.portIds;
            for (const auto& pid : portIds) {
                if (existingPortIds.count(pid)) {
                    continue;
                }

                Port newPort;
                newPort.portId = pid;
                newPort.layer = link.netLayer;
                newPort.dieId = ExtractDieIdFromPortId(pid);
                newPort.eid = "";
                newPort.linkType = link.linkType;
                newPort.planeId = "plane_0";
                newPort.topoType = link.topoType;
                newPort.topoInstanceId = link.topoInstanceId;
                newPort.position = link.position;
                newPort.protocols = link.protocols;

                dev.ports.push_back(newPort);
                existingPortIds.insert(pid);
            }
        }
    }
}

int ClusterTopoParser::ExtractDieIdFromPortId(const std::string& portId)
{
    size_t slashPos = portId.find('/');
    if (slashPos != std::string::npos && slashPos > 0) {
        try {
            return std::stoi(portId.substr(0, slashPos));
        } catch (...) {
            HCCL_VM_ERROR("ExtractDieIdFromPortId error");
            return -1;
        }
    }
    return -1;
}

ParseStatus ClusterTopoParser::ParseServersInfoJson(const std::string& serversInfoPath)
{
    std::ifstream ifs(serversInfoPath);
    if (!ifs.is_open()) {
        HCCL_VM_WARN("servers_info.json not found: {}", serversInfoPath.c_str());
        return ParseStatus::FILE_NOT_FOUND;
    }

    try {
        json j;
        ifs >> j;

        if (j.contains("version") && j["version"].is_string() && network_.version.empty()) {
            network_.version = j["version"].get<std::string>();
        }

        if (j.contains("server_count") && j["server_count"].is_number()) {
            network_.serverCount = j["server_count"].get<int>();
        }

        if (!j.contains("server_ip_list") || !j["server_ip_list"].is_array()) {
            return ParseStatus::OK;
        }
        const auto& ipList = j["server_ip_list"];

        // 条目数与叶子节点数一致性: server_count为叶子节点(host节点)总数
        // (server形态=server数; POD形态=POD数×board数)
        int leafCount = network_.GetTotalBoardCount();
        if (static_cast<int>(ipList.size()) != leafCount) {
            HCCL_VM_WARN(
                "servers_info.json entry count[{}] != leaf node count[{}] (superpod/server/board dirs)", ipList.size(),
                leafCount);
        }

        // 优先按显式三元id(super_pod_id/server_id/board_id)匹配, 避免依赖条目位置序。
        // 旧版servers_info.json无这些字段时回退按遍历位置序匹配(并告警)
        std::map<std::tuple<int, int, int>, size_t> idToEntryIdx;
        bool hasExplicitIds = true;
        for (size_t i = 0; i < ipList.size(); ++i) {
            const auto& entry = ipList[i];
            if (!entry.contains("super_pod_id") || !entry.contains("server_id")) {
                hasExplicitIds = false;
                break;
            }
            int spId = entry["super_pod_id"].get<int>();
            int srvId = entry["server_id"].get<int>();
            int boardId = entry.contains("board_id") ? entry["board_id"].get<int>() : 0;
            auto key = std::make_tuple(spId, srvId, boardId);
            if (idToEntryIdx.count(key) != 0) {
                HCCL_VM_ERROR(
                    "duplicated (super_pod_id={:d}, server_id={:d}, board_id={:d}) in servers_info.json", spId, srvId,
                    boardId);
                return ParseStatus::INVALID_FORMAT;
            }
            idToEntryIdx[key] = i;
        }
        if (!hasExplicitIds) {
            HCCL_VM_WARN("servers_info.json has no super_pod_id/server_id/board_id fields, "
                         "fallback to positional matching (please regenerate cluster topo)");
        }

        auto applyEntry = [](const json& entry, Board& board) {
            if (entry.contains("addr") && entry["addr"].is_string()) {
                board.hostEid = entry["addr"].get<std::string>();
            }
            // port_type: 端口类型(x2/x4, 不区分大小写), generate_cluster_topo.sh从
            // server/pod拓扑配置yaml读取写入(hccl-vm工具专属字段, 不进topo.json等
            // hccl业务原生格式文件); 缺省/非法值按x4处理, 仅性能仿真消费(x2带宽减半)
            if (entry.contains("port_type") && entry["port_type"].is_string()) {
                std::string portType = entry["port_type"].get<std::string>();
                std::transform(portType.begin(), portType.end(), portType.begin(), [](unsigned char c) {
                    return static_cast<char>(std::tolower(c));
                });
                if (portType == "x2" || portType == "x4") {
                    board.portType = portType;
                } else {
                    HCCL_VM_WARN(
                        "invalid port_type[{}] in servers_info.json, fallback to x4",
                        entry["port_type"].get<std::string>().c_str());
                }
            }
            std::string socVersion;
            if (entry.contains("soc_version") && entry["soc_version"].is_string()) {
                socVersion = entry["soc_version"].get<std::string>();
            }
            if (!socVersion.empty()) {
                for (auto& dev : board.devices) {
                    dev.socVersion = socVersion;
                    dev.devType = ParseDevType(socVersion);
                }
            }
        };

        int positionalIdx = 0;
        for (auto& superpod : network_.superPods) {
            for (auto& server : superpod.servers) {
                for (auto& board : server.boards) {
                    const json* entry = nullptr;
                    if (hasExplicitIds) {
                        auto key = std::make_tuple(superpod.superPodId, server.serverId, board.boardId);
                        auto it = idToEntryIdx.find(key);
                        if (it == idToEntryIdx.end()) {
                            HCCL_VM_WARN(
                                "no servers_info.json entry for superpod{:d}/server{:d}/board{:d}", superpod.superPodId,
                                server.serverId, board.boardId);
                        } else {
                            entry = &ipList[it->second];
                        }
                    } else if (positionalIdx < static_cast<int>(ipList.size())) {
                        entry = &ipList[positionalIdx];
                    }
                    if (entry != nullptr) {
                        applyEntry(*entry, board);
                    }
                    positionalIdx++;
                }
            }
        }
    } catch (const json::exception& e) {
        HCCL_VM_ERROR("JSON parse error in {}: {}", serversInfoPath.c_str(), e.what());
        return ParseStatus::PARSE_ERROR;
    }

    return ParseStatus::OK;
}

namespace {
// 从EID十六进制串取出内嵌IPV4的指定八位组(octetIdx: 0=AA, 1=BB, 2=CC, 3=DD)。
// EID低32bit为IPV4地址, 即字符串末8个hex字符; 解析失败返回-1
int EidIpv4Octet(const std::string& eid, int octetIdx)
{
    if (eid.size() < 8 || octetIdx < 0 || octetIdx > 3) {
        return -1;
    }
    std::string ipHex = eid.substr(eid.size() - 8, 8);
    std::string byteHex = ipHex.substr(static_cast<size_t>(octetIdx) * 2, 2);
    try {
        return static_cast<int>(std::stoul(byteHex, nullptr, 16));
    } catch (...) {
        return -1;
    }
}
} // namespace

ParseStatus ClusterTopoParser::ValidateClusterConsistency()
{
    // 一致性校验: 目录id ↔ rootinfo内net_instance_id ↔ servers_info的host EID网段。
    // 这三者分别由"目录结构"、"generate_server_topo.sh+allocate_eid.sh重写"、
    // "generate_cluster_topo.sh Step E"产出, 任一侧口径不一致(如目录id按字典序下标解析)
    // 都会静默产出"host IP与自身device IP不同段"、"ranktable声称的srv与实际EID不符"的错误集群
    for (const auto& superpod : network_.superPods) {
        for (const auto& server : superpod.servers) {
            for (const auto& board : server.boards) {
                // (1) rootinfo的net_instance_id中携带的 sp_x / srv_y / board_z 必须与目录id一致。
                //     格式(按作用域scope): sp_{i}_srv_{j}_board_{b}_idx_{k} / sp_{i}_srv_{j}_idx_{k}
                //     / sp_{i}_idx_{k} / cluster
                for (const auto& dev : board.devices) {
                    for (const auto& port : dev.ports) {
                        if (port.netInstanceId.empty() || port.netInstanceId == "cluster") {
                            continue;
                        }
                        unsigned int spId = 0;
                        unsigned int srvId = 0;
                        unsigned int boardId = 0;
                        unsigned int idxK = 0;
                        const char* niid = port.netInstanceId.c_str();
                        if (sscanf(niid, "sp_%u_srv_%u_board_%u_idx_%u", &spId, &srvId, &boardId, &idxK) == 4) {
                            if (static_cast<int>(spId) != superpod.superPodId
                                || static_cast<int>(srvId) != server.serverId
                                || static_cast<int>(boardId) != board.boardId) {
                                HCCL_VM_ERROR(
                                    "net_instance_id[{}] mismatches dir location superpod{:d}/server{:d}/board{:d}",
                                    niid, superpod.superPodId, server.serverId, board.boardId);
                                return ParseStatus::INVALID_FORMAT;
                            }
                            continue;
                        }
                        if (sscanf(niid, "sp_%u_srv_%u_idx_%u", &spId, &srvId, &idxK) == 3) {
                            if (static_cast<int>(spId) != superpod.superPodId
                                || static_cast<int>(srvId) != server.serverId) {
                                HCCL_VM_ERROR(
                                    "net_instance_id[{}] mismatches dir location superpod{:d}/server{:d}", niid,
                                    superpod.superPodId, server.serverId);
                                return ParseStatus::INVALID_FORMAT;
                            }
                            continue;
                        }
                        if (sscanf(niid, "sp_%u_idx_%u", &spId, &idxK) == 2) {
                            if (static_cast<int>(spId) != superpod.superPodId) {
                                HCCL_VM_ERROR(
                                    "net_instance_id[{}] mismatches dir location superpod{:d}", niid,
                                    superpod.superPodId);
                                return ParseStatus::INVALID_FORMAT;
                            }
                        }
                    }
                }

                // (2) host EID的网段必须与本board的device EID网段一致:
                //     server形态 AA.BB一致(CC为endPoint计数高位, 不比较);
                //     POD形态   AA.BB.CC一致(CC=board_id)
                if (board.hostEid.empty() || board.devices.empty()) {
                    continue;
                }
                std::string devEid;
                for (const auto& dev : board.devices) {
                    for (const auto& port : dev.ports) {
                        if (!port.eid.empty()) {
                            devEid = port.eid;
                            break;
                        }
                    }
                    if (!devEid.empty()) {
                        break;
                    }
                }
                if (devEid.empty()) {
                    continue;
                }
                bool isPodForm = server.boards.size() > 1 || board.boardId != 0;
                int hostAa = EidIpv4Octet(board.hostEid, 0);
                int hostBb = EidIpv4Octet(board.hostEid, 1);
                int devAa = EidIpv4Octet(devEid, 0);
                int devBb = EidIpv4Octet(devEid, 1);
                if (hostAa != devAa || hostBb != devBb) {
                    HCCL_VM_ERROR(
                        "superpod{:d}/server{:d}/board{:d}: host EID segment {}.{:d} mismatches "
                        "device EID segment {}.{:d} (servers_info.json entry order/id wrong?)",
                        superpod.superPodId, server.serverId, board.boardId, hostAa, hostBb, devAa, devBb);
                    return ParseStatus::INVALID_FORMAT;
                }
                if (isPodForm) {
                    int hostCc = EidIpv4Octet(board.hostEid, 2);
                    int devCc = EidIpv4Octet(devEid, 2);
                    if (hostCc != devCc || hostCc != board.boardId) {
                        HCCL_VM_ERROR(
                            "superpod{:d}/server{:d}/board{:d}: board segment(CC) mismatch, "
                            "host CC={:d}, device CC={:d}, expect board_id={:d}",
                            superpod.superPodId, server.serverId, board.boardId, hostCc, devCc, board.boardId);
                        return ParseStatus::INVALID_FORMAT;
                    }
                }
            }
        }
    }
    return ParseStatus::OK;
}
