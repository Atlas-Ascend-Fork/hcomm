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
        HCCL_VM_DEBUG("PodNum: {}, SerNum: {}, RankNum: {}", podNum, serNum, rankNum);
        if (podNum <= 0 || podNum > 1024 || serNum <= 0 || rankNum <= 0) {
            HCCL_VM_ERROR("YAML : 'meta' number not surport, please check your config.yaml.");
            return false;
        }
        if (root["topology"] && root["topology"].IsSequence()) {
            for (const auto& pod : root["topology"]) {
                SuperPodMeta superPodMeta;
                uint32_t podId = 0;
                if (pod["podId"] && pod["servers"] && pod["servers"].IsSequence()) {
                    podId = pod["podId"].as<uint32_t>();
                    for (const auto& ser : pod["servers"]) {
                        const YAML::Node serIdNode = ser["serId"];
                        const YAML::Node serRangeNode = ser["serId_range"];
                        const YAML::Node ranksNode = ser["ranks"];
                        const YAML::Node ranksRangeNode = ser["ranks_range"];

                        if (serIdNode && serRangeNode) {
                            HCCL_VM_ERROR("YAML : 'serId' and 'serId_range' cannot be set at the same time.");
                            return false;
                        }
                        if (ranksNode && ranksRangeNode) {
                            HCCL_VM_ERROR("YAML : 'ranks' and 'ranks_range' cannot be set at the same time.");
                            return false;
                        }

                        ServerMeta serverMeta;
                        if (ranksRangeNode) {
                            uint32_t rankBegin = 0;
                            uint32_t rankEnd = 0;
                            if (!ParseRangeValue(ranksRangeNode, rankBegin, rankEnd)) {
                                return false;
                            }
                            for (uint64_t rankId = rankBegin; rankId <= rankEnd; ++rankId) {
                                serverMeta.push_back(static_cast<PhyDeviceId>(rankId));
                            }
                        } else if (ranksNode) {
                            if (!ranksNode.IsSequence()) {
                                HCCL_VM_ERROR("YAML : 'ranks' must be a list.");
                                return false;
                            }
                            serverMeta = ranksNode.as<std::vector<uint32_t>>();
                        }

                        if (serRangeNode) {
                            uint32_t serBegin = 0;
                            uint32_t serEnd = 0;
                            if (!ParseRangeValue(serRangeNode, serBegin, serEnd)) {
                                return false;
                            }
                            for (uint64_t serverId = serBegin; serverId <= serEnd; ++serverId) {
                                superPodMeta[static_cast<uint32_t>(serverId)] = serverMeta;
                            }
                        } else if (serIdNode && serIdNode.IsSequence()) {
                            for (const auto& idNode : serIdNode) {
                                uint32_t serverId = 0;
                                if (!idNode.IsScalar() || !ParseUnsignedStr(TrimBlank(idNode.Scalar()), serverId)) {
                                    HCCL_VM_ERROR("YAML : 'serId' list element invalid, expect unsigned integers.");
                                    return false;
                                }
                                superPodMeta[serverId] = serverMeta;
                            }
                        } else {
                            uint32_t serverId = 0;
                            if (serIdNode) {
                                serverId = serIdNode.as<uint32_t>();
                            }
                            superPodMeta[serverId] = serverMeta;
                        }
                    }
                }
                topo[podId] = superPodMeta;
            }
        }
        return true;
    } catch (const YAML::Exception& e) {
        HCCL_VM_ERROR("Exception when parsing YAML: {}", e.what());
        return false;
    }
}
#endif
