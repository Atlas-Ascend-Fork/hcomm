/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sim_common_api.h"

#include <cstdlib>
#include <fstream>
#include <string>
#include <unistd.h>
#include <climits>
#include <map>
#include "sim_log.h"

static std::string GetExePath()
{
    char buf[PATH_MAX] = {0};
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len <= 0)
        return "";
    buf[len] = '\0';
    return std::string(buf);
}

static std::string GetExeDir()
{
    std::string full = GetExePath();
    if (full.empty()) {
        return ".";
    }
    size_t pos = full.find_last_of('/');
    return (pos == std::string::npos) ? "." : full.substr(0, pos);
}

static std::string ComputeInstallRoot()
{
    const char* env = std::getenv("HCCL_VM_INSTALL_ROOT");
    if (env && *env) {
        return std::string(env);
    }
    std::string exe = GetExePath();
    if (exe.empty()) {
        return ".";
    }

    std::string cur = exe;
    for (int i = 0; i < 8; ++i) {
        size_t pos = cur.find_last_of('/');
        if (pos == std::string::npos || pos == 0) {
            break;
        }
        cur = cur.substr(0, pos);
        if (std::ifstream(cur + "/config/log_config.yaml").good()) {
            return cur;
        }
    }
    return GetExeDir();
}

const std::string& InstallPath::GetHcclVmInstallAbsPath()
{
    static const std::string root = ComputeInstallRoot();
    return root;
}

std::string InstallPath::ResolveToInstallRoot(const std::string& relPath)
{
    if (relPath.empty() || relPath[0] == '/' || (relPath.size() >= 2 && relPath[0] == '.' && relPath[1] == '/')) {
        return relPath;
    }
    return GetHcclVmInstallAbsPath() + "/" + relPath;
}

static std::map<HcclDataType, std::string> g_DataType2Str = {
    {HcclDataType::HCCL_DATA_TYPE_INT8, "INT8"},       {HcclDataType::HCCL_DATA_TYPE_INT16, "INT16"},
    {HcclDataType::HCCL_DATA_TYPE_INT32, "INT32"},     {HcclDataType::HCCL_DATA_TYPE_FP16, "FP16"},
    {HcclDataType::HCCL_DATA_TYPE_UINT64, "UINT64"},   {HcclDataType::HCCL_DATA_TYPE_UINT8, "UINT8"},
    {HcclDataType::HCCL_DATA_TYPE_UINT16, "UINT16"},   {HcclDataType::HCCL_DATA_TYPE_UINT32, "UINT32"},
    {HcclDataType::HCCL_DATA_TYPE_FP64, "FP64"},       {HcclDataType::HCCL_DATA_TYPE_BFP16, "BFP16"},
    {HcclDataType::HCCL_DATA_TYPE_INT128, "INT128"},   {HcclDataType::HCCL_DATA_TYPE_INT64, "INT64"},
    {HcclDataType::HCCL_DATA_TYPE_HIF8, "HIF8"},       {HcclDataType::HCCL_DATA_TYPE_FP8E4M3, "FP8E4M3"},
    {HcclDataType::HCCL_DATA_TYPE_FP8E5M2, "FP8E5M2"},
};

static std::map<HcclReduceOp, std::string> g_ReduceOp2Str = {
    {HcclReduceOp::HCCL_REDUCE_SUM, "SUM"},
    {HcclReduceOp::HCCL_REDUCE_MIN, "MIN"},
    {HcclReduceOp::HCCL_REDUCE_MAX, "MAX"},
    {HcclReduceOp::HCCL_REDUCE_PROD, "PROD"},
};

std::string GetDataTypeStr(HcclDataType type)
{
    if (g_DataType2Str.find(type) == g_DataType2Str.end()) {
        return "UNKNOWN";
    }
    return g_DataType2Str[type];
}

std::string GetReduceOpStr(HcclReduceOp op)
{
    if (g_ReduceOp2Str.find(op) == g_ReduceOp2Str.end()) {
        return "UNKNOWN";
    }
    return g_ReduceOp2Str[op];
}

#if !defined(NO_YAML_CONFIG) && defined(HAVE_YAML_CPP)
static std::string TrimBlank(const std::string& str)
{
    size_t first = str.find_first_not_of(" \t");
    if (first == std::string::npos) {
        return "";
    }
    size_t last = str.find_last_not_of(" \t");
    return str.substr(first, last - first + 1);
}

static bool ParseUnsignedStr(const std::string& str, uint32_t& val)
{
    if (str.empty()) {
        return false;
    }
    uint64_t value = 0;
    for (char c : str) {
        if (c < '0' || c > '9') {
            return false;
        }
        value = value * 10 + static_cast<uint64_t>(c - '0');
        if (value > 0xFFFFFFFFULL) {
            return false;
        }
    }
    val = static_cast<uint32_t>(value);
    return true;
}

static bool ParseRangeValue(const YAML::Node& node, uint32_t& begin, uint32_t& end)
{
    if (node && node.IsScalar()) {
        HCCL_VM_ERROR("YAML : range '{}' invalid, only '[<begin>, <end>]' is supported.", TrimBlank(node.Scalar()));
        return false;
    }
    if (!node || !node.IsSequence() || node.size() != 2 || !node[0].IsScalar() || !node[1].IsScalar()) {
        HCCL_VM_ERROR("YAML : range node invalid, expect '[<begin>, <end>]'.");
        return false;
    }
    uint32_t beginVal = 0;
    uint32_t endVal = 0;
    if (!ParseUnsignedStr(TrimBlank(node[0].Scalar()), beginVal)
        || !ParseUnsignedStr(TrimBlank(node[1].Scalar()), endVal) || beginVal > endVal) {
        HCCL_VM_ERROR(
            "YAML : range '[{}, {}]' invalid, expect two unsigned integers with begin <= end.",
            TrimBlank(node[0].Scalar()), TrimBlank(node[1].Scalar()));
        return false;
    }
    uint64_t width = static_cast<uint64_t>(endVal) - beginVal + 1;
    if (width > 1024) {
        HCCL_VM_ERROR("YAML : range '[{}, {}]' too large, width must be less than 1024.", beginVal, endVal);
        return false;
    }
    begin = beginVal;
    end = endVal;
    return true;
}

// 解析id选择字段(二选一, 不可同时配置):
//   {idField}       : 单个id或id列表(如 3 或 [0, 1])
//   {idField}_range : 闭区间 [<起始id>, <结束id>], 宽度不超过1024
// 用于 serId/serId_range 与 boardId/boardId_range。返回false表示配置非法;
// 两个字段都未配置时ids为空(由调用方决定缺省值, 如serId缺省0、boardId缺省0)
static bool ParseIdSelection(const YAML::Node& entry, const std::string& idField, std::vector<uint32_t>& ids)
{
    const YAML::Node idNode = entry[idField];
    const YAML::Node rangeNode = entry[idField + "_range"];
    if (idNode && rangeNode) {
        HCCL_VM_ERROR("YAML : '{}' and '{}_range' cannot be set at the same time.", idField, idField);
        return false;
    }

    ids.clear();
    if (rangeNode) {
        uint32_t begin = 0;
        uint32_t end = 0;
        if (!ParseRangeValue(rangeNode, begin, end)) {
            return false;
        }
        for (uint64_t id = begin; id <= end; ++id) {
            ids.push_back(static_cast<uint32_t>(id));
        }
        return true;
    }
    if (idNode && idNode.IsSequence()) {
        for (const auto& idNodeItem : idNode) {
            uint32_t id = 0;
            if (!idNodeItem.IsScalar() || !ParseUnsignedStr(TrimBlank(idNodeItem.Scalar()), id)) {
                HCCL_VM_ERROR("YAML : '{}' list element invalid, expect unsigned integers.", idField);
                return false;
            }
            ids.push_back(id);
        }
        return true;
    }
    if (idNode) {
        ids.push_back(idNode.as<uint32_t>());
    }
    return true;
}

// 解析rank选择字段(ranks/ranks_range, 二选一), 得到叶子节点内的设备物理id列表。
// 未配置时返回空列表(与既有行为一致: 该节点的ranks为空)
static bool ParseRankSelection(const YAML::Node& entry, NodeMeta& nodeMeta)
{
    const YAML::Node ranksNode = entry["ranks"];
    const YAML::Node ranksRangeNode = entry["ranks_range"];
    if (ranksNode && ranksRangeNode) {
        HCCL_VM_ERROR("YAML : 'ranks' and 'ranks_range' cannot be set at the same time.");
        return false;
    }

    nodeMeta.clear();
    if (ranksRangeNode) {
        uint32_t rankBegin = 0;
        uint32_t rankEnd = 0;
        if (!ParseRangeValue(ranksRangeNode, rankBegin, rankEnd)) {
            return false;
        }
        for (uint64_t rankId = rankBegin; rankId <= rankEnd; ++rankId) {
            nodeMeta.push_back(static_cast<PhyDeviceId>(rankId));
        }
        return true;
    }
    if (ranksNode) {
        if (!ranksNode.IsSequence()) {
            HCCL_VM_ERROR("YAML : 'ranks' must be a list.");
            return false;
        }
        nodeMeta = ranksNode.as<std::vector<uint32_t>>();
    }
    return true;
}

bool ParseTopoMetaYaml(const std::string& fileName, TopoMeta& topo)
{
    try {
        std::string filePath = InstallPath::ResolveToInstallRoot("config/topo_meta/" + fileName + ".yaml");
        YAML::Node root = YAML::LoadFile(filePath);

        if (!root["meta"]) {
            HCCL_VM_ERROR("YAML : 'meta' node not found.");
            return false;
        }
        uint32_t podNum = root["meta"]["podNum"].as<uint32_t>();
        uint32_t serNum = root["meta"]["serNum"].as<uint32_t>();
        uint32_t rankNum = root["meta"]["rankNum"].as<uint32_t>();
        // boardNum: 可选字段(POD形态4层拓扑)。未配置时按server形态处理(每个server只有一个隐式board0)
        uint32_t boardNum = 0;
        if (root["meta"]["boardNum"]) {
            boardNum = root["meta"]["boardNum"].as<uint32_t>();
        }
        HCCL_VM_DEBUG("PodNum: {}, SerNum: {}, BoardNum: {}, RankNum: {}", podNum, serNum, boardNum, rankNum);
        if (podNum <= 0 || podNum > 1024 || serNum <= 0 || rankNum <= 0) {
            HCCL_VM_ERROR("YAML : 'meta' number not surport, please check your config.yaml.");
            return false;
        }
        if (boardNum > 1024) {
            HCCL_VM_ERROR("YAML : 'meta.boardNum' {} out of range (expect <= 1024).", boardNum);
            return false;
        }
        if (root["topology"] && root["topology"].IsSequence()) {
            for (const auto& pod : root["topology"]) {
                SuperPodMeta superPodMeta;
                uint32_t podId = 0;
                if (pod["podId"] && pod["servers"] && pod["servers"].IsSequence()) {
                    podId = pod["podId"].as<uint32_t>();
                    for (const auto& ser : pod["servers"]) {
                        // server选择: serId / serId_range
                        std::vector<uint32_t> serverIds;
                        if (!ParseIdSelection(ser, "serId", serverIds)) {
                            return false;
                        }
                        if (serverIds.empty()) {
                            // 未配置server选择字段时默认serId为0(与既有行为一致)
                            serverIds.push_back(0);
                        }

                        // board层: 配置了boards时为POD形态(4层), 逐board声明rank;
                        // 未配置时为server形态(3层), ranks/ranks_range直接归属隐式board0
                        const YAML::Node boardsNode = ser["boards"];
                        if (boardsNode) {
                            if (!boardsNode.IsSequence()) {
                                HCCL_VM_ERROR("YAML : 'boards' must be a list.");
                                return false;
                            }
                            // ranks/ranks_range与boards互斥: 二者都声明时无法确定rank归属
                            if (ser["ranks"] || ser["ranks_range"]) {
                                HCCL_VM_ERROR(
                                    "YAML : 'ranks'/'ranks_range' and 'boards' cannot be set at the same time, "
                                    "please declare ranks inside each board entry.");
                                return false;
                            }
                            for (const auto& boardEntry : boardsNode) {
                                std::vector<uint32_t> boardIds;
                                if (!ParseIdSelection(boardEntry, "boardId", boardIds)) {
                                    return false;
                                }
                                if (boardIds.empty()) {
                                    boardIds.push_back(0);
                                }
                                NodeMeta nodeMeta;
                                if (!ParseRankSelection(boardEntry, nodeMeta)) {
                                    return false;
                                }
                                for (uint32_t serverId : serverIds) {
                                    for (uint32_t boardId : boardIds) {
                                        superPodMeta[serverId][boardId] = nodeMeta;
                                    }
                                }
                            }
                            continue;
                        }

                        NodeMeta nodeMeta;
                        if (!ParseRankSelection(ser, nodeMeta)) {
                            return false;
                        }
                        // server形态(或POD形态未显式声明boards): 归属隐式board0
                        for (uint32_t serverId : serverIds) {
                            superPodMeta[serverId][0] = nodeMeta;
                        }
                    }
                }
                topo[podId] = superPodMeta;
            }
        }

        // meta.boardNum与topology中实际出现的board数一致性校验(仅提示, 不阻断:
        // server形态下boardNum可省略, 此时每个server只有隐式board0)
        if (boardNum > 0) {
            uint32_t actualBoardNum = 0;
            for (const auto& [pid, superPod] : topo) {
                for (const auto& [srvId, server] : superPod) {
                    actualBoardNum += static_cast<uint32_t>(server.size());
                }
            }
            if (actualBoardNum != boardNum) {
                HCCL_VM_WARN(
                    "YAML : meta.boardNum({}) != board count in topology({}), please check your config.", boardNum,
                    actualBoardNum);
            }
        }
        return true;
    } catch (const YAML::Exception& e) {
        HCCL_VM_ERROR("Exception when parsing YAML: {}", e.what());
        return false;
    }
}
#endif
