/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_ROOT_HANDLE_V2_H
#define HCCL_ROOT_HANDLE_V2_H

#include "orion_adapter_hccp.h"
#include "ip_address.h"
#include "binary_stream.h"
#include "log.h"
#include "../config_mgr/coll_comm_config.h"

namespace Hccl {

// 初始化探测消息的可选尾部，不改变既有 RankTable 二进制布局。
struct NetLayerCheckInfo {
    static constexpr u32 MAGIC = 0x4e4c5331;
    static constexpr u32 VERSION = 1;
    u32 version{0}; // 0 表示旧端未上报，不能视为已通过校验
    u32 explicitSelection{0};
    u32 layers{0};
    u32 result{HCCL_SUCCESS};

    // 只比较生效集合；控制连接负责确认通信域身份并收齐成员。
    HcclResult CheckConsistency(const NetLayerCheckInfo& other) const
    {
        if (result != HCCL_SUCCESS || other.result != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[NetLayerCheckInfo::CheckConsistency] unsuccessful report, result[%u], otherResult[%u]", result,
                other.result);
            return HCCL_E_PARA;
        }
        // 全默认保持原行为，不要求异构 rank 的候选层相同。
        if (explicitSelection == 0 && other.explicitSelection == 0) {
            return HCCL_SUCCESS;
        }
        if (version != VERSION || other.version != VERSION) {
            HCCL_ERROR(
                "[NetLayerCheckInfo::CheckConsistency] unsupported version[%u], otherVersion[%u], supported[%u]",
                version, other.version, VERSION);
            return HCCL_E_NOT_SUPPORT;
        }
        if (explicitSelection > 1 || other.explicitSelection > 1 || layers == 0 || layers > 0xff
            || layers != other.layers) {
            HCCL_ERROR(
                "[NetLayerCheckInfo::CheckConsistency] invalid or inconsistent selection, explicit[%u], "
                "otherExplicit[%u], layers[%s], otherLayers[%s]",
                explicitSelection, other.explicitSelection, hccl::NetLayerSelection::FormatLayers(layers).c_str(),
                hccl::NetLayerSelection::FormatLayers(other.layers).c_str());
            return HCCL_E_PARA;
        }
        return HCCL_SUCCESS;
    }

    void Serialize(BinaryStream& stream) const
    {
        stream << static_cast<u32>(MAGIC) << version << explicitSelection << layers << result;
    }

    HcclResult Deserialize(BinaryStream& stream)
    {
        // V1 是探测消息的最后一个固定长度字段；扩展须升版，不能静默接受额外尾部。
        std::vector<char> tail;
        stream.Dump(tail);
        if (tail.empty()) {
            *this = {};
            return HCCL_SUCCESS;
        }
        if (tail.size() < 2 * sizeof(u32)) {
            HCCL_ERROR(
                "[NetLayerCheckInfo::Deserialize] truncated header, size[%zu], minimum[%zu]", tail.size(),
                2 * sizeof(u32));
            return HCCL_E_PARA;
        }
        BinaryStream input(tail);
        NetLayerCheckInfo parsed;
        u32 magic = 0;
        input >> magic >> parsed.version;
        if (magic != MAGIC) {
            HCCL_ERROR("[NetLayerCheckInfo::Deserialize] invalid magic[0x%x], expected[0x%x]", magic, MAGIC);
            return HCCL_E_PARA;
        }
        if (parsed.version != VERSION) {
            HCCL_ERROR(
                "[NetLayerCheckInfo::Deserialize] unsupported version[%u], supported[%u]", parsed.version, VERSION);
            return HCCL_E_NOT_SUPPORT;
        }
        constexpr size_t FRAME_SIZE = 5 * sizeof(u32);
        if (tail.size() != FRAME_SIZE) {
            HCCL_ERROR(
                "[NetLayerCheckInfo::Deserialize] invalid frame size[%zu], expected[%zu]", tail.size(), FRAME_SIZE);
            return HCCL_E_PARA;
        }
        input >> parsed.explicitSelection >> parsed.layers >> parsed.result;
        if (parsed.explicitSelection > 1 || parsed.layers > 0xff
            || (parsed.result != HCCL_SUCCESS && parsed.result != HCCL_E_PARA && parsed.result != HCCL_E_NOT_SUPPORT)) {
            HCCL_ERROR(
                "[NetLayerCheckInfo::Deserialize] invalid fields, explicitSelection[%u], layers[%s], result[%u]",
                parsed.explicitSelection, hccl::NetLayerSelection::FormatLayers(parsed.layers).c_str(), parsed.result);
            return HCCL_E_PARA;
        }
        // 校验成功才发布；损坏或不支持的帧不改写已有结果。
        *this = parsed;
        return HCCL_SUCCESS;
    }
};

// HcclRootHandleV2 定义
constexpr u32 IP_ADDRESS_BUFFER_LEN = 64;
constexpr u32 ROOTINFO_IDENTIFIER_MAX_LENGTH = 128;
const std::string RANK_INFO_DETECT_TAG = "rank_info_detect_default_tag";

using HcclRootHandleV2 = struct HcclRootHandleDefV2 {
    char ip[IP_ADDRESS_BUFFER_LEN];
    u32 listenPort{HCCL_INVALID_PORT};
    HrtNetworkMode netMode{HrtNetworkMode::HDC};
    char identifier[ROOTINFO_IDENTIFIER_MAX_LENGTH];
    // 派生谱系：协调者填充，新 rank 回填，使 topParentIdentifier_/generationIndex_ 与 survivor 一致
    char topParentIdentifier[ROOTINFO_IDENTIFIER_MAX_LENGTH]{};
    u32 generationIndex{0};
    u32 valid{0};               // 0=未填充（首轮/首父），新 rank 检测此标志决定是否回填谱系
    u32 rootRankId{UINT32_MAX}; // 当前通信域内 root rank；UINT32_MAX 表示尚未解析/当前域没有 root
};

// scalable 多 root 建链时单个 root 的完整信息（序列化进 HcclRootInfo.internal，
// 由 HcclGetRootInfoScalableV2 填充、HcclCommInitRootInfoScalableV2 解析）。
using HcclScalableRootHandleV2 = struct HcclScalableRootHandleDefV2 {
    HcclRootHandleV2 rootHandle;     // 本 root 的组内 agent 服务地址
    u32 meshPort{HCCL_INVALID_PORT}; // root 间 mesh 监听端口
};

// buffer大小
constexpr u32 MAX_BUFFER_LEN = 100 * 1024 * 1024; // 100M

constexpr u32 ONE_MILLISECOND_OF_USLEEP = 1000;

} // namespace Hccl

#endif // HCCL_ROOT_HANDLE_V2_H
