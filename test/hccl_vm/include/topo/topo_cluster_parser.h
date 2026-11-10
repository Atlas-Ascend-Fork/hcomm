/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TOPO_CLUSTER_PARSER_H
#define TOPO_CLUSTER_PARSER_H

#include <string>

#include "topo_cluster_ir.h"

class ClusterTopoParser {
public:
    ClusterTopoParser();
    ~ClusterTopoParser();

    ParseStatus ParseClusterDir(const std::string& clusterDir);

    const Network& GetNetwork() const { return network_; }
    Network& GetNetwork() { return network_; }

    static int ExtractDieIdFromPortId(const std::string& portId);

private:
    ParseStatus ScanSuperPodDirs(const std::string& clusterDir);
    ParseStatus ScanServerDirs(const std::string& superpodDir, SuperPod& superpod);
    // 扫描server目录下的board子目录(POD形态4层拓扑):
    //   存在board{N}/子目录 → POD形态: topo.json必须为server级单份(POD全局local_id =
    //     board_id * 每board卡数 + board内物理id), 按board切分并重映射回board内物理id;
    //     各board{N}/目录为叶子节点(独立host节点), 仅含rootinfo.json(board内物理id)。
    //     存量旧布局(board目录下各自一份topo.json)不再支持, 报错提示重新生成集群产物;
    //   不存在 → server形态, server目录自身视为唯一叶子节点board0
    ParseStatus ScanBoardDirs(const std::string& serverDir, Server& server);
    // 解析一个叶子节点(server形态=server目录)的topo.json与rootinfo.json
    ParseStatus ParseBoard(const std::string& leafDir, Board& board);
    // 把POD级topo.json的解析结果切分出指定board的devices/links:
    // 按local_id值域[boardId*devicesPerBoard, (boardId+1)*devicesPerBoard)截取,
    // 并把POD全局local_id重映射为board内物理id(IR/ranktable的device_id口径), 重建linkId
    ParseStatus ExtractBoardFromPodTopo(const Board& podTopo, int boardId, int devicesPerBoard, Board& board);

    ParseStatus ParseTopoJson(const std::string& topoPath, Board& board);
    ParseStatus ParseRootinfoJson(const std::string& rootinfoPath, Board& board);
    ParseStatus ParseServersInfoJson(const std::string& serversInfoPath);
    // 集群一致性校验: 目录id ↔ rootinfo的net_instance_id ↔ servers_info的host EID网段
    ParseStatus ValidateClusterConsistency();

    void MergeTopoAndRootinfo(Board& board);

    std::string FindFileBySuffix(const std::string& dir, const std::string& suffix);

    Network network_;
};

#endif // TOPO_CLUSTER_PARSER_H
