/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_SOCKET_CONFIG_H
#define HCCL_SOCKET_CONFIG_H

#include "types.h"
#include "virtual_topo.h"
#include "hash_utils.h"
#include "log.h"

namespace Hccl {
MAKE_ENUM(SocketRole, SERVER, CLIENT)

static constexpr size_t HCCP_TAG_MAX_LEN = 191; // SOCK_CONN_TAG_SIZE - 1 (留 1 字节给 '\0')
class SocketConfig {
public:
    RankId remoteRank;
    LinkData link;
    uint32_t listeningPort{DEFAULT_LISTENING_PORT};
    const std::string tag;
    uint32_t hostNic2DeviceNicMode_{0}; // 0 normal, 1: host(host cpu roce channel) - device(transport ibv)

    SocketConfig(RankId remoteRank, const LinkData& link, const std::string& tag)
        : remoteRank(remoteRank),
          link(link),
          tag(tag),
          role(link.GetLocalRankId() < link.GetRemoteRankId() ? SocketRole::SERVER : SocketRole::CLIENT)
    {
        hccpTag = BuildHccpTagWithRankAndIpIndex(tag, link, role);
        LogIpIndexMapping(link);
        CheckHccpTag();
    }

    SocketConfig(const LinkData& link, const std::string& tag)
        : remoteRank(link.GetRemoteRankId()),
          link(link),
          tag(tag),
          role(
              (link.GetLocalAddr() < link.GetRemoteAddr()
               || (link.GetLocalAddr() == link.GetRemoteAddr() && link.GetLocalRankId() < link.GetRemoteRankId())) ?
                  SocketRole::SERVER :
                  SocketRole::CLIENT)
    {
        hccpTag = BuildHccpTagWithRankAndIpIndex(tag, link, role);
        LogIpIndexMapping(link);
        CheckHccpTag();
    }

    SocketConfig(const LinkData& link, const std::string& tag, SocketRole role, const uint32_t listenPort)
        : remoteRank(link.GetRemoteRankId()),
          link(link),
          listeningPort(listenPort),
          tag(tag),
          role(role)
    {
        hccpTag = BuildHccpTagWithAddr(tag, link, role);
        CheckHccpTag();
    }

    SocketConfig(const LinkData& link, const std::string& tag, bool noRankId)
        : remoteRank(link.GetRemoteRankId()),
          link(link),
          tag(tag),
          role(
              (link.GetLocalAddr() < link.GetRemoteAddr()
               || (link.GetLocalAddr() == link.GetRemoteAddr() && link.GetLocalRankId() < link.GetRemoteRankId())) ?
                  SocketRole::SERVER :
                  SocketRole::CLIENT),
          noRankId(noRankId)
    {
        hccpTag = BuildHccpTagWithAddr(tag, link, role);
        CheckHccpTag();
    }

    SocketConfig(
        const LinkData& link, const uint32_t listenPort, const std::string& tag, uint32_t hostNic2DeviceNicMode,
        const uint32_t myRank, const uint32_t rmtRank)
        : SocketConfig(link, listenPort, tag)
    {
        // HOST路径下 link 的 rankId 是 devPhyId 而非真实 rank，
        // IP相等时 devPhyId 可能相同导致双端角色一致，用真实 rank 覆盖
        // hostNic2DeviceNicMode为1表示A2场景，该处新增为A5场景，故在hostNic2DeviceNicMode == 0判断之前。
        if (link.GetLocalAddr() == link.GetRemoteAddr()) {
            remoteRank = rmtRank;
            role = myRank < rmtRank ? SocketRole::SERVER : SocketRole::CLIENT;
        }
        if (hostNic2DeviceNicMode == 0) {
            return;
        }
        // Parse commTag from tag prefix: tag format is "commTag_engine_X" or "commTag_engine_X_protocol_Y"
        std::string commTag = tag;
        size_t enginePos = commTag.find("_engine_");
        if (enginePos != std::string::npos) {
            commTag = commTag.substr(0, enginePos);
        } else {
            HCCL_WARNING("[SocketConfig] socketTag[%s] format abnormal, using original tag as commTag", tag.c_str());
        }
        remoteRank = rmtRank;
        role = myRank < rmtRank ? SocketRole::SERVER : SocketRole::CLIENT;
        hccpTag = BuildHccpTagWithRank(commTag, myRank, rmtRank, link, role);
        hostNic2DeviceNicMode_ = hostNic2DeviceNicMode;
        CheckHccpTag();
    }

    SocketConfig(const LinkData& link, const uint32_t listenPort, const std::string& tag)
        : remoteRank(link.GetRemoteRankId()),
          link(link),
          listeningPort(listenPort),
          tag(tag)
    {
        role = (link.GetLocalAddr() < link.GetRemoteAddr()
                || (link.GetLocalAddr() == link.GetRemoteAddr() && link.GetLocalRankId() < link.GetRemoteRankId())) ?
                   SocketRole::SERVER :
                   SocketRole::CLIENT;
        hccpTag = BuildHccpTagWithAddrAndPort(tag, link, role, listenPort);
        CheckHccpTag();
    }

    SocketConfig(const LinkData& link, const uint32_t listenPort, const std::string& tag, const bool isServer)
        : remoteRank(link.GetRemoteRankId()),
          link(link),
          listeningPort(listenPort),
          tag(tag)
    {
        role = isServer ? SocketRole::SERVER : SocketRole::CLIENT;
        hccpTag = BuildHccpTagWithAddrAndPort(tag, link, role, listenPort);
        CheckHccpTag();
    }

    // 新增：useRawTag=true 时 hccpTag 直接使用 tag，供探测等自定义 tag 场景
    SocketConfig(
        const LinkData& link, const std::string& tag, SocketRole role, const uint32_t listenPort, bool useRawTag)
        : remoteRank(link.GetRemoteRankId()),
          link(link),
          listeningPort(listenPort),
          tag(tag),
          role(role),
          hccpTag(tag)
    {
        if (!useRawTag) {
            if (role == SocketRole::SERVER) { // server: tag_local_remote
                hccpTag = tag + "_" + link.GetLocalAddr().GetIpStr() + "_" + link.GetRemoteAddr().GetIpStr();
            } else { // client: tag_remote_local
                hccpTag = tag + "_" + link.GetRemoteAddr().GetIpStr() + "_" + link.GetLocalAddr().GetIpStr();
            }
        }
    }

    SocketRole GetRole() const { return role; }

    const string& GetHccpTag() const { return hccpTag; }

    // 直接设置 hccpTag，绕过后缀拼接
    void SetHccpTag(const string& rawTag)
    {
        hccpTag = rawTag;
        CheckHccpTag();
    }

private:
    SocketRole role{};
    string hccpTag{};

    // 带 rankId 和 ipIndex 的 hccpTag: tag_localRank_remoteRank_localIpIdx_remoteIpIdx (SERVER) /
    // tag_remoteRank_localRank_remoteIpIdx_localIpIdx (CLIENT)
    static string BuildHccpTagWithRankAndIpIndex(const string& tag, const LinkData& link, SocketRole role)
    {
        if (role == SocketRole::SERVER) {
            return tag + "_" + to_string(link.GetLocalRankId()) + "_" + to_string(link.GetRemoteRankId()) + "_"
                   + to_string(link.GetLocalIpIndex()) + "_" + to_string(link.GetRemoteIpIndex());
        }
        return tag + "_" + to_string(link.GetRemoteRankId()) + "_" + to_string(link.GetLocalRankId()) + "_"
               + to_string(link.GetRemoteIpIndex()) + "_" + to_string(link.GetLocalIpIndex());
    }

    // 带 rankId 和显式 IP 地址 的 hccpTag (构造函数5: hostNic2DeviceNicMode)
    // A2场景，不使用ipIndex
    static string
    BuildHccpTagWithRank(const string& tag, uint32_t myRank, uint32_t rmtRank, const LinkData& link, SocketRole role)
    {
        if (role == SocketRole::SERVER) {
            return tag + "_" + to_string(myRank) + "_" + to_string(rmtRank) + "_" + link.GetLocalAddr().GetIpStr() + "_"
                   + link.GetRemoteAddr().GetIpStr();
        }
        return tag + "_" + to_string(rmtRank) + "_" + to_string(myRank) + "_" + link.GetRemoteAddr().GetIpStr() + "_"
               + link.GetLocalAddr().GetIpStr();
    }

    // 带显式 IP 地址的 hccpTag: tag_localIp_remoteIp (SERVER) / tag_remoteIp_localIp (CLIENT)
    static string BuildHccpTagWithAddr(const string& tag, const LinkData& link, SocketRole role)
    {
        if (role == SocketRole::SERVER) {
            return tag + "_" + link.GetLocalAddr().GetIpStr() + "_" + link.GetRemoteAddr().GetIpStr();
        }
        return tag + "_" + link.GetRemoteAddr().GetIpStr() + "_" + link.GetLocalAddr().GetIpStr();
    }

    // 带显式 IP 地址和 port 的 hccpTag: tag_localIp_remoteIp_port (SERVER) / tag_remoteIp_localIp_port (CLIENT)
    static string
    BuildHccpTagWithAddrAndPort(const string& tag, const LinkData& link, SocketRole role, uint32_t listenPort)
    {
        if (role == SocketRole::SERVER) {
            return tag + "_" + link.GetLocalAddr().GetIpStr() + "_" + link.GetRemoteAddr().GetIpStr() + "_"
                   + to_string(listenPort);
        }
        return tag + "_" + link.GetRemoteAddr().GetIpStr() + "_" + link.GetLocalAddr().GetIpStr() + "_"
               + to_string(listenPort);
    }

    // 打印 ipIndex 与原 IP 的映射关系
    static void LogIpIndexMapping(const LinkData& link)
    {
        HCCL_INFO(
            "[SocketConfig] hccpTag uses ipIndex[%u](localIp[%s]) ipIndex[%u](remoteIp[%s])", link.GetLocalIpIndex(),
            link.GetLocalAddr().GetIpStr().c_str(), link.GetRemoteIpIndex(), link.GetRemoteAddr().GetIpStr().c_str());
    }

    // 校验 hccpTag
    void CheckHccpTag()
    {
        if (hccpTag.size() > HCCP_TAG_MAX_LEN) {
            HCCL_ERROR(
                "[SocketConfig] hccpTag length[%zu] exceeds max[%zu], tag[%s]", hccpTag.size(), HCCP_TAG_MAX_LEN,
                hccpTag.c_str());
        }
    }

public:
    bool noRankId{false};
};
} // namespace Hccl

namespace std {
// 特化SocketConfig的hash和equal模板，使其可用做map的key
template <>
class hash<Hccl::SocketConfig> {
public:
    size_t operator()(const Hccl::SocketConfig& socketConfig) const
    {
        auto remoteRankHash = hash<Hccl::RankId>{}(socketConfig.remoteRank);
        auto localPortHash = hash<Hccl::PortData>{}(socketConfig.link.GetLocalPort());
        auto remotePortHash = hash<Hccl::PortData>{}(socketConfig.link.GetRemotePort());
        auto tagHash = hash<string>{}(socketConfig.tag);
        auto portHash = hash<uint32_t>{}(socketConfig.listeningPort);

        return Hccl::HashCombine({remoteRankHash, localPortHash, remotePortHash, tagHash, portHash});
    }
};

template <>
class equal_to<Hccl::SocketConfig> {
public:
    bool operator()(const Hccl::SocketConfig& config, const Hccl::SocketConfig& otherConfig) const
    {
        bool IsOthersSame = config.link.GetLocalPort().GetAddr() == otherConfig.link.GetLocalPort().GetAddr()
                            && config.link.GetRemotePort().GetAddr() == otherConfig.link.GetRemotePort().GetAddr()
                            && config.tag == otherConfig.tag && config.GetHccpTag() == otherConfig.GetHccpTag()
                            && config.listeningPort == otherConfig.listeningPort;

        if (config.noRankId && otherConfig.noRankId) {
            return IsOthersSame;
        }

        return IsOthersSame && config.remoteRank == otherConfig.remoteRank;
    }
};
} // namespace std

#endif // HCCL_SOCKET_CONFIG_H
