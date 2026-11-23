/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rank_table_info.h"

#include <unordered_set>
#include <unordered_map>
#include <sstream>
#include <string>
#include <algorithm>
#include <new>
#include "invalid_params_exception.h"
#include "exception_util.h"
#include "adapter_error_manager_pub.h"
#include "common/hccl_log_keywords_pub.h"

namespace Hccl {

HcclResult RankTableInfo::Check()
{
    if (version != "2.0") {
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] version[%s] is not \"2.0\"", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), version.c_str());
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>({version, "version", "2.0"}));
        return HCCL_E_PARA;
    }

    if (rankCount > MAX_RANKCOUNT) {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>(
                {std::to_string(rankCount), "rankCount", "lower than " + std::to_string(MAX_RANKCOUNT)}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] rankCount[%u] exceeds maximum limit[%u]", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), rankCount, MAX_RANKCOUNT);
        return HCCL_E_PARA;
    }

    if (rankCount == 0) {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>({std::to_string(rankCount), "rankCount", "should not be 0"}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] rankCount is 0", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA));
        return HCCL_E_PARA;
    }

    if (rankCount != ranks.size()) {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>(
                {std::to_string(rankCount), "rankCount",
                 "rankCount is equal to rankSize[" + std::to_string(ranks.size()) + "]"}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] rankCount[%u] != ranks.size[%zu]", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), rankCount, ranks.size());
        return HCCL_E_PARA;
    }

    // rankId 的范围/去重/连续性用按 rankId 下标的位图一次扫描完成, 避免逐 rank 的哈希查找
    std::vector<bool> rankIdSeen(rankCount, false);
    std::unordered_set<u32> localIdSet;
    u32 recordedReplaceLocalId{UNDEFIEND_LOCAL_ID};
    for (auto& rank : ranks) {
        if (static_cast<u32>(rank.rankId) >= rankCount) {
            RPT_INPUT_ERR(
                true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
                std::vector<std::string>(
                    {std::to_string(rank.rankId), "rankId", "[0," + std::to_string(rankCount) + ")"}));
            HCCL_ERROR(
                "[%s][%s] errNo[0x%016llx] rankId[%u] out of range [0,%u)", LOG_KEYWORDS_INIT_GROUP.c_str(),
                LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), rank.rankId, rankCount);
            return HCCL_E_PARA;
        }
        if (rankIdSeen[rank.rankId]) {
            RPT_INPUT_ERR(
                true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
                std::vector<std::string>({std::to_string(rank.rankId), "rankId", "rank_id is not repeat"}));
            HCCL_ERROR(
                "[%s][%s] errNo[0x%016llx] rankId[%u] is repeat", LOG_KEYWORDS_INIT_GROUP.c_str(),
                LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), rank.rankId);
            return HCCL_E_PARA;
        }
        rankIdSeen[rank.rankId] = true;

        if (rank.localId != BACKUP_LOCAL_ID && rank.localId != rank.replacedLocalId) {
            RPT_INPUT_ERR(
                true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
                std::vector<std::string>(
                    {std::to_string(rank.replacedLocalId), "replacedLocalId",
                     "replacedLocalId equal to localId[" + std::to_string(rank.localId) + "]"}));
            HCCL_ERROR(
                "[%s][%s] errNo[0x%016llx] replacedLocalId[%u] != localId[%u]", LOG_KEYWORDS_INIT_GROUP.c_str(),
                LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), rank.replacedLocalId,
                rank.localId);
            return HCCL_E_PARA;
        } else if (rank.localId == BACKUP_LOCAL_ID) {
            if (recordedReplaceLocalId == UNDEFIEND_LOCAL_ID) {
                recordedReplaceLocalId = rank.replacedLocalId;
            } else {
                RPT_INPUT_ERR(
                    true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
                    std::vector<std::string>({"NA", "NA", "multiple replaced rank is configured"}));
                HCCL_ERROR(
                    "[%s][%s] errNo[0x%016llx] multiple replaced rank is configured", LOG_KEYWORDS_INIT_GROUP.c_str(),
                    LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA));
                return HCCL_E_PARA;
            }
        } else {
            localIdSet.emplace(rank.localId);
        }
    }

    for (u32 rankRange = 0; rankRange < rankCount; ++rankRange) {
        if (!rankIdSeen[rankRange]) {
            RPT_INPUT_ERR(
                true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
                std::vector<std::string>({std::to_string(rankRange), "rankId", "rank_id is continuous"}));
            HCCL_ERROR(
                "[%s][%s] errNo[0x%016llx] rankId[%u] is not continuous", LOG_KEYWORDS_INIT_GROUP.c_str(),
                LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), rankRange);
            return HCCL_E_PARA;
        }
    }

    std::vector<std::unordered_map<std::string, u32>> verifyRankAddr;
    for (auto& rank : ranks) {
        for (auto& levelInfo : rank.rankLevelInfos) {
            CHK_RET(InsertToRank(levelInfo.netInstId, levelInfo.rankAddrs.size(), verifyRankAddr, levelInfo.netLayer));
        }
    }

    if (localIdSet.find(recordedReplaceLocalId) != localIdSet.end()) {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>(
                {std::to_string(recordedReplaceLocalId), "recordedReplacedLocalId",
                 "failed with configuring same local_id with replaced one simultaneously"}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] configuring same local_id[%u] with replaced one simultaneously",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_CHECK.c_str(),
            HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), recordedReplaceLocalId);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult RankTableInfo::CheckAndInsert(
    const std::string& levelId, u32 rankAddrSize, std::unordered_map<std::string, u32>& idRankSizeMap) const
{
    // 单次 try_emplace 同时完成"存在性判断 + 插入", 消除 find/operator[] 的多次哈希查找
    const auto insertResult = idRankSizeMap.try_emplace(levelId, rankAddrSize);
    if (!insertResult.second && insertResult.first->second != rankAddrSize) {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>(
                {std::to_string(rankAddrSize), levelId, std::to_string(insertResult.first->second)}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] rank_addrs size[%u] for id[%s] is different from existing[%u]",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_CHECK.c_str(),
            HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), rankAddrSize, levelId.c_str(), insertResult.first->second);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult RankTableInfo::InsertToRank(
    const std::string& levelId, u32 rankAddrSize, std::vector<std::unordered_map<std::string, u32>>& rankLists,
    u32 levelNum) const
{
    if (rankLists.size() <= levelNum) {
        rankLists.resize(levelNum + 1);
    }
    CHK_RET(CheckAndInsert(levelId, rankAddrSize, rankLists[levelNum]));
    return HCCL_SUCCESS;
}

std::string RankTableInfo::Describe() const
{
    return Hccl::StringFormat(
        "RankTableInfo[version=%s, rankCount=%u, ranks size=%d]", version.c_str(), rankCount, ranks.size());
}

void RankTableInfo::Dump() const
{
    HCCL_DEBUG("RankTableInfo Dump:");
    HCCL_DEBUG("%s", Describe().c_str());
    HCCL_DEBUG("ranks:");
    for (const auto& rank : ranks) {
        HCCL_DEBUG("%s", rank.Describe().c_str());
        for (const auto& levelInfo : rank.rankLevelInfos) {
            HCCL_DEBUG("- %s", levelInfo.Describe().c_str());
        }
    }
}

RankTableInfo::RankTableInfo(Hccl::BinaryStream& binaryStream)
{
    binaryStream >> version >> rankCount;
    size_t ranksSize = 0;
    binaryStream >> ranksSize;
    CHK_PRT_THROW(
        ranksSize > MAX_RANKCOUNT,
        HCCL_ERROR("[%s] ranks size[%zu] exceeds max[%u].", __func__, ranksSize, MAX_RANKCOUNT),
        Hccl::InvalidParamsException, "ranks size exceeds limit");
    HCCL_INFO("[%s] version[%s] rankCount[%u] ranks size[%zu]", __func__, version.c_str(), rankCount, ranksSize);
    // 损坏/伪造流可在上限内声明超大 ranksSize: 预留量先按流剩余可读字节数收敛
    // (单个 rank 的二进制序列化最小体积按 16 字节保守估算, 预留不足只影响性能不影响正确性),
    // 避免一次性 GB 级 reserve; 读流循环按流剩余字节检测截断, 分配失败转 Hccl::InvalidParamsException。
    static constexpr size_t kMinBytesPerSerializedRank = 16;
    const std::int64_t streamRemaining = binaryStream.GetRemainingSize();
    const size_t reserveSize
        = streamRemaining > 0 ? std::min(ranksSize, static_cast<size_t>(streamRemaining) / kMinBytesPerSerializedRank) :
                                0;
    try {
        ranks.reserve(reserveSize);
        for (u32 i = 0; i < ranksSize; i++) {
            CHK_PRT_THROW(
                binaryStream.GetRemainingSize() <= 0,
                HCCL_ERROR("[%s] stream exhausted at rank[%u]/[%zu], corrupted stream.", __func__, i, ranksSize),
                Hccl::InvalidParamsException, "ranktable stream truncated");
            hccl::NewRankInfo rankInfo(binaryStream);
            ranks.emplace_back(std::move(rankInfo));
        }
    } catch (const std::bad_alloc&) {
        CHK_PRT_THROW(
            true, HCCL_ERROR("[%s] ranks reserve[%zu] bad_alloc.", __func__, reserveSize), Hccl::InvalidParamsException,
            "ranktable reserve bad alloc");
    }
    CHK_PRT_THROW(
        binaryStream.GetRemainingSize()
            < static_cast<std::int64_t>(sizeof(detour) + sizeof(logicSuperPodSplitLayer) + sizeof(size_t)),
        HCCL_ERROR("[%s] logic group metadata is truncated.", __func__), InvalidParamsException,
        "logic group metadata truncated");
    binaryStream >> detour;
    binaryStream >> logicSuperPodSplitLayer;
    // 反序列化逻辑分组映射（快照恢复路径）。
    size_t groupCount = 0;
    binaryStream >> groupCount;
    CHK_PRT_THROW(
        groupCount > ranks.size(),
        HCCL_ERROR("[%s] logic group count[%zu] exceeds ranks size[%zu].", __func__, groupCount, ranks.size()),
        InvalidParamsException, "logic group count exceeds ranks size");
    for (size_t i = 0; i < groupCount; ++i) {
        CHK_PRT_THROW(
            binaryStream.GetRemainingSize() < static_cast<std::int64_t>(sizeof(RankId) + sizeof(size_t)),
            HCCL_ERROR("[%s] logic group[%zu] header is truncated.", __func__, i), InvalidParamsException,
            "logic group header truncated");
        RankId rankId = 0;
        size_t groupIdSize = 0;
        binaryStream >> rankId >> groupIdSize;
        CHK_PRT_THROW(
            groupIdSize > MAX_LEN_OF_LOGIC_SUPERPOD_ID
                || groupIdSize > static_cast<size_t>(binaryStream.GetRemainingSize()),
            HCCL_ERROR("[%s] logic group[%zu] has invalid ID length[%zu].", __func__, i, groupIdSize),
            InvalidParamsException, "logic group ID length invalid");
        // 先校验再分配，避免BinaryStream的字符串读取按未校验长度分配内存。
        std::string groupId(groupIdSize, '\0');
        for (char& byte : groupId) {
            binaryStream >> byte;
        }
        logicSuperPodGroupIds.emplace(rankId, std::move(groupId));
    }
}

HcclResult RankTableInfo::GetBinStream(bool isContainLocId, Hccl::BinaryStream& binaryStream) const
{
    return GetBinStream(isContainLocId, binaryStream, false);
}

HcclResult RankTableInfo::GetBinStream(bool isContainLocId, Hccl::BinaryStream& binaryStream, bool allowEmpty) const
{
    if (ranks.size() == 0) {
        if (!allowEmpty) {
            RPT_INPUT_ERR(
                true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
                std::vector<std::string>({"0", "ranks", "non-empty"}));
            HCCL_ERROR(
                "[%s][%s] errNo[0x%016llx] ranks size is zero", LOG_KEYWORDS_INIT_GROUP.c_str(),
                LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA));
            return HCCL_E_PARA;
        }
        // 空表走同一编码路径（下方 rankCount/ranks 计数天然写 0），反序列化侧完整消费；
        // 唯一差异：version 缺省空串时对齐 "2.0"，避免空表作为合并初始表时 version 传空
        if (version.empty()) {
            binaryStream << std::string("2.0");
        } else {
            binaryStream << version;
        }
        binaryStream << rankCount;
        const size_t emptyRanksSize = 0;
        binaryStream << emptyRanksSize;
        binaryStream << detour;
        binaryStream << logicSuperPodSplitLayer;
        binaryStream << logicSuperPodGroupIds.size();
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[%s] version[%s] rankCount[%u] ranks size[%u]", __func__, version.c_str(), rankCount, ranks.size());

    binaryStream << version << rankCount;
    binaryStream << ranks.size();
    for (auto& it : ranks) {
        CHK_RET(it.GetBinStream(isContainLocId, binaryStream));
    }
    binaryStream << detour;
    binaryStream << logicSuperPodSplitLayer;
    binaryStream << logicSuperPodGroupIds.size();
    for (const auto& [rankId, groupId] : logicSuperPodGroupIds) {
        binaryStream << rankId << groupId;
    }
    return HCCL_SUCCESS;
}

HcclResult RankTableInfo::SerializeAllowEmpty(Hccl::BinaryStream& binaryStream) const
{
    return GetBinStream(true, binaryStream, true);
}

vector<char> RankTableInfo::GetUniqueId(bool isContainLocId) const
{
    CHK_PRT_THROW(
        ranks.empty(), HCCL_ERROR("[%s] ranks size is zero.", __func__), Hccl::InvalidParamsException,
        "ranks size is zero");
    std::vector<char> result(0);

    Hccl::BinaryStream binaryStream;
    binaryStream << version << rankCount;

    u32 ranksSize = ranks.size();
    binaryStream << ranksSize;
    for (auto& it : ranks) {
        CHK_RET_THROW(
            Hccl::InvalidParamsException, "[RankTableInfo::GetUniqueId] NewRankInfo GetBinStream failed",
            it.GetBinStream(isContainLocId, binaryStream));
    }

    binaryStream.Dump(result);
    return result;
}

HcclResult RankTableInfo::UpdateRankTable(const RankTableInfo& localRankInfo)
{
    // version
    if (detour) {
        RPT_INPUT_ERR(
            localRankInfo.detour != true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>({"false", "detour", "true"}));
        CHK_PRT_RET(
            localRankInfo.detour != true,
            HCCL_ERROR(
                "[%s][%s] errNo[0x%016llx] detour config is not consistent", LOG_KEYWORDS_INIT_GROUP.c_str(),
                LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA)),
            HCCL_E_PARA);
    }
    detour = localRankInfo.detour;
    if (rankCount == 0) {
        version = localRankInfo.version;
    } else {
        RPT_INPUT_ERR(
            version != localRankInfo.version, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>({version, "version", localRankInfo.version}));
        CHK_PRT_RET(
            version != localRankInfo.version,
            HCCL_ERROR(
                "[%s][%s] errNo[0x%016llx] version[%s] is not consistent with local version[%s]",
                LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_CHECK.c_str(),
                HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), version.c_str(), localRankInfo.version.c_str()),
            HCCL_E_PARA);
    }

    // ranks size
    RPT_INPUT_ERR(
        localRankInfo.ranks.size() == 0, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
        std::vector<std::string>({"0", "ranks", "non-empty"}));
    CHK_PRT_RET(
        localRankInfo.ranks.size() == 0,
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] ranks size is zero", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA)),
        HCCL_E_PARA);

    ranks.insert(ranks.end(), localRankInfo.ranks.begin(), localRankInfo.ranks.end());
    rankCount++;

    HCCL_INFO("[%s] success, current rankTableInfo[%s]", __func__, Describe().c_str());
    return HCCL_SUCCESS;
}

std::unordered_map<u32, std::unordered_map<IpAddress, u32>> RankTableInfo::GetRankDeviceListenPortMap()
{
    std::unordered_map<u32, std::unordered_map<IpAddress, u32>> ranklListenPortMap;
    ranklListenPortMap.reserve(ranks.size()); // 预分配外层，消除 rehash
    for (auto& rankinfo : ranks) {
        std::unordered_map<IpAddress, u32> listenPortMap;
        size_t addrCnt = 0;
        for (auto& rankLevelInfo : rankinfo.rankLevelInfos) {
            addrCnt += rankLevelInfo.rankAddrs.size();
        }
        listenPortMap.reserve(addrCnt + 1); // 预分配内层（含 hccl::DEVICE_PORT_KEY_IPADDRESS 占位），消除 rehash
        for (auto& rankLevelInfo : rankinfo.rankLevelInfos) {
            for (auto& rankAddr : rankLevelInfo.rankAddrs) {
                listenPortMap.insert(std::make_pair(rankAddr.addr, rankAddr.socketPort_));
            }
        }
        listenPortMap.insert(std::make_pair(hccl::DEVICE_PORT_KEY_IPADDRESS, rankinfo.devicePort));
        ranklListenPortMap.emplace(rankinfo.rankId, std::move(listenPortMap)); // 移动替代一次内层深拷贝
    }
    return ranklListenPortMap;
}

HcclResult RankTableInfo::BuildLogicSuperPodNetInstId(
    const std::string& originNetInstId, const std::string& logicId, std::string& newNetInstId)
{
    if (logicId.empty()) {
        newNetInstId = originNetInstId;
        return HCCL_SUCCESS;
    }
    if (originNetInstId.size() >= MAX_VALUE_NETID || logicId.size() > MAX_VALUE_NETID - originNetInstId.size() - 1) {
        HCCL_ERROR(
            "[%s] appended netInstId exceeds limit, originIdLen[%zu] logicIdLen[%zu] limit[%u]", __func__,
            originNetInstId.size(), logicId.size(), MAX_VALUE_NETID);
        return HCCL_E_PARA;
    }
    newNetInstId = originNetInstId + "_" + logicId;
    return HCCL_SUCCESS;
}

} // namespace Hccl
