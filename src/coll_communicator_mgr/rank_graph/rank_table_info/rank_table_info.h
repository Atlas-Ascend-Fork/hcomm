/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NEW_RANK_TABLED_H
#define NEW_RANK_TABLED_H

// 两份.h 必须一起改：改本文件时同步改 legacy950 侧
// src/legacy/ascend950/framework/topo/new_topo_builder/rank_table_info/rank_table_info.h。
// 注：两份同 guard（NEW_RANK_TABLED_H）同 ns（Hccl），靠 CMake BEFORE 路径分离，根治需迁 ns 会动 bridge，
// 方案既定不动（见 CMakeLists.txt 注释），故两份未改名/未迁 ns，同步注释仍提示一起改。

#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <climits>
#include "new_rank_info.h"
#include "rank_table_source.h"
#include "port.h"
namespace Hccl {
constexpr unsigned int MAX_RANKCOUNT = 262144; // 2^18, 约 26 万卡
constexpr u32 UNDEFIEND_LOCAL_ID = UINT_MAX;
class RankTableInfo {
public:
    RankTableInfo() {};
    RankTableInfo(BinaryStream& binaryStream);
    std::string version;
    u32 rankCount{0};
    u32 logicSuperPodSplitLayer{MAX_VALUE_NETLAYER + 1}; // 随快照保存拆分层，越过最大层号表示未配置。
    // rankId→原始逻辑ID，改写时填写全表、随快照序列化；恢复建图后用于GetPaths跨组拦截。
    std::map<RankId, std::string> logicSuperPodGroupIds;
    std::vector<hccl::NewRankInfo> ranks;
    bool detour = false;
    std::string Describe() const;
    void Dump() const;
    RankTableSource GetSource() const { return source_; }
    void SetSource(RankTableSource source) { source_ = source; }
    std::vector<char> GetUniqueId(bool isContainLocId) const; // 获取rankTable的字节流，供一致性校验crc时带localId使用
    HcclResult GetBinStream(bool isContainLocId, BinaryStream& binaryStream) const;
    // allowEmpty：空表不再报 E_PARA，按同一编码布局输出全零计数帧
    HcclResult GetBinStream(bool isContainLocId, BinaryStream& binaryStream, bool allowEmpty) const;
    HcclResult SerializeAllowEmpty(BinaryStream& binaryStream) const;
    HcclResult Check();
    // 生成追加逻辑ID后的实例ID；空逻辑ID保持原值，超长时返回错误且不修改输出。
    static HcclResult BuildLogicSuperPodNetInstId(
        const std::string& originNetInstId, const std::string& logicId, std::string& newNetInstId);
    HcclResult UpdateRankTable(const RankTableInfo& localRankInfo);
    std::unordered_map<u32, std::unordered_map<Hccl::IpAddress, u32>> GetRankDeviceListenPortMap();

private:
    RankTableSource source_{RankTableSource::RANKTABLE};
    HcclResult CheckAndInsert(
        const std::string& levelId, u32 rankAddrSize, std::unordered_map<std::string, u32>& idRankSizeMap) const;
    HcclResult InsertToRank(
        const std::string& levelId, u32 rankAddrSize, std::vector<std::unordered_map<std::string, u32>>& rankLists,
        u32 levelNum) const;
};

} // namespace Hccl

#endif // NEW_RANK_TABLED_H
