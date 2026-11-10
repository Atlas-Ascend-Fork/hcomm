/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_resource_stats_reporter.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

#include "ccu_resource_stats_dfx.h"
#include "sim_log.h"

namespace HcclSim {
namespace CcuResourceStats {
    namespace {

        // 分节数据行列头（算子标识/信息在节头注释行中，数据行只含拓扑/设备 + 资源列）
        const char* const CCU_RESOURCE_STATS_SECTION_HEADER = "superPodId,serverId,boardId,phyDevId,dieId,"
                                                              "channelNum,instructionNum,loopTemplateNum,loopNum,"
                                                              "xnNum,gsaNum,ckeNum,msNum,missionNum";

        // CSV 字段转义：含逗号/引号/换行时用双引号包裹，内部引号翻倍
        std::string EscapeCsvField(const std::string& field)
        {
            const bool needQuote = field.find_first_of(",\"\n\r") != std::string::npos;
            if (!needQuote) {
                return field;
            }
            std::string escaped;
            escaped.reserve(field.size() + 2);
            escaped.push_back('"');
            for (const char ch : field) {
                if (ch == '"') {
                    escaped.push_back('"');
                }
                escaped.push_back(ch);
            }
            escaped.push_back('"');
            return escaped;
        }

        // 节内一行（dev/die + 资源），按 (superPodId, serverId, phyDevId, dieId) 排序
        struct StatsCsvRow {
            int64_t superPodId{-1};
            int64_t serverId{-1};
            // boardId: POD形态(4层拓扑)下的board序号; server形态恒为0(-1表示拓扑标签缺失)
            int64_t boardId{-1};
            uint32_t phyDevId{0};
            uint32_t dieId{0};
            uint32_t channelNum{0};
            uint32_t instructionNum{0};
            uint32_t loopTemplateNum{0};
            uint32_t loopNum{0};
            uint32_t xnNum{0};
            uint32_t gsaNum{0};
            uint32_t ckeNum{0};
            uint32_t msNum{0};
            uint32_t missionNum{0};

            bool operator<(const StatsCsvRow& other) const
            {
                if (superPodId != other.superPodId) {
                    return superPodId < other.superPodId;
                }
                if (serverId != other.serverId) {
                    return serverId < other.serverId;
                }
                if (boardId != other.boardId) {
                    return boardId < other.boardId;
                }
                if (phyDevId != other.phyDevId) {
                    return phyDevId < other.phyDevId;
                }
                return dieId < other.dieId;
            }
        };

        std::string FormatRow(const StatsCsvRow& row, bool gsaApplicable)
        {
            std::ostringstream os;
            os << row.superPodId << ',' << row.serverId << ',' << row.boardId << ',' << row.phyDevId << ',' << row.dieId
               << ',' << row.channelNum << ',' << row.instructionNum << ',' << row.loopTemplateNum << ',' << row.loopNum
               << ',' << row.xnNum << ',';
            if (gsaApplicable) {
                os << row.gsaNum;
            } else {
                os << "NA";
            }
            os << ',' << row.ckeNum << ',' << row.msNum << ',' << row.missionNum;
            return os.str();
        }

        // op 节排序键：(opIter, commName, commHash)
        bool OpKeyDisplayOrder(const OpKey& lhs, const OpKey& rhs)
        {
            if (lhs.opIter != rhs.opIter) {
                return lhs.opIter < rhs.opIter;
            }
            if (lhs.commName != rhs.commName) {
                return lhs.commName < rhs.commName;
            }
            return lhs.commHash < rhs.commHash;
        }

    } // namespace

    std::string BuildCcuResourceStatsCsv(
        const OpStatsMap& data, const std::map<OpKey, OpInfo>& opInfos, const TopoLabelMap& topoLabels,
        bool gsaApplicable)
    {
        // op 节按展示顺序（opIter, commName, commHash）排列
        std::vector<const OpKey*> orderedOps;
        orderedOps.reserve(data.size());
        for (const auto& opEntry : data) {
            orderedOps.push_back(&opEntry.first);
        }
        std::sort(orderedOps.begin(), orderedOps.end(), [](const OpKey* lhs, const OpKey* rhs) {
            return OpKeyDisplayOrder(*lhs, *rhs);
        });

        std::ostringstream csv;
        uint32_t opIdx = 0;
        for (const OpKey* opKey : orderedOps) {
            const DevStatsMap& devMap = data.at(*opKey);
            if (opIdx > 0) {
                csv << '\n'; // 节间空行分隔
            }
            // 节头：算子标识 + 算子信息。整行以 '#' 开头的单列注释（字段间用 "; " 分隔，
            // 避免逗号被 Excel 分列导致内容截断；相邻单元格为空时 Excel 自动溢出显示全文）
            csv << "# ===== op[" << opIdx << "]: commName=" << opKey->commName << "; commHash=" << opKey->commHash
                << "; opIter=" << opKey->opIter << " =====\n";
            OpInfo info;
            const auto infoIt = opInfos.find(*opKey);
            if (infoIt != opInfos.end()) {
                info = infoIt->second;
            }
            csv << "# opInfo: type=" << info.collectiveType << "; alg=" << info.algName
                << "; dataType=" << info.dataType << "; elements=" << info.elementCount
                << "; inputBytes=" << info.inputBytes << "; outputBytes=" << info.outputBytes
                << "; reduceType=" << info.reduceType << "; rankSize=" << info.rankSize << "\n";
            csv << CCU_RESOURCE_STATS_SECTION_HEADER << '\n';

            std::vector<StatsCsvRow> rows;
            for (const auto& devEntry : devMap) {
                const uint32_t phyDevId = devEntry.first;
                TopoLabel label;
                const auto labelIt = topoLabels.find(phyDevId);
                if (labelIt != topoLabels.end()) {
                    label = labelIt->second;
                }
                for (const auto& dieEntry : devEntry.second) {
                    const CcuResStats& stats = dieEntry.second;
                    StatsCsvRow row;
                    row.superPodId = label.superPodId;
                    row.serverId = label.serverId;
                    row.boardId = label.boardId;
                    row.phyDevId = phyDevId;
                    row.dieId = dieEntry.first;
                    row.channelNum = static_cast<uint32_t>(stats.channelIds.size());
                    row.instructionNum = static_cast<uint32_t>(stats.opInstrIds.size());
                    row.loopTemplateNum = stats.LoopTemplateNum();
                    row.loopNum = stats.LoopNum();
                    row.xnNum = static_cast<uint32_t>(stats.xnIds.size());
                    row.gsaNum = static_cast<uint32_t>(stats.gsaIds.size());
                    row.ckeNum = static_cast<uint32_t>(stats.ckeIds.size());
                    row.msNum = static_cast<uint32_t>(stats.msIds.size());
                    row.missionNum = static_cast<uint32_t>(stats.missionIds.size());
                    rows.push_back(row);
                }
            }
            std::sort(rows.begin(), rows.end());
            for (const StatsCsvRow& row : rows) {
                csv << FormatRow(row, gsaApplicable) << '\n';
            }
            ++opIdx;
        }
        return csv.str();
    }

    HcclResult
    GenerateCcuResourceStatsReport(const std::string& csvPath, bool gsaApplicable, const TopoLabelMap& topoLabels)
    {
        const OpStatsMap& data = Collector::GetInstance().GetData();
        if (data.empty()) {
            HCCL_VM_WARN(
                "[CCU_RESOURCE_STATS] no statistics collected (single-op checker disabled or no op expanded); "
                "an empty report will be written, path={}",
                csvPath);
        }

        std::error_code ec;
        const std::filesystem::path parentDir = std::filesystem::path(csvPath).parent_path();
        if (!parentDir.empty()) {
            std::filesystem::create_directories(parentDir, ec);
            if (ec) {
                HCCL_VM_ERROR(
                    "[CCU_RESOURCE_STATS] failed to create report directory, dir={}, error={}", parentDir.string(),
                    ec.message());
                return HcclResult::HCCL_E_INTERNAL;
            }
        }

        const std::string csvContent
            = BuildCcuResourceStatsCsv(data, Collector::GetInstance().GetOpInfos(), topoLabels, gsaApplicable);
        std::ofstream ofs(csvPath);
        if (!ofs.is_open()) {
            HCCL_VM_ERROR("[CCU_RESOURCE_STATS] failed to open report file, path={}", csvPath);
            return HcclResult::HCCL_E_INTERNAL;
        }
        ofs << csvContent;
        ofs.close();

        uint64_t rowCount = 0;
        for (const auto& opEntry : data) {
            for (const auto& devEntry : opEntry.second) {
                rowCount += devEntry.second.size();
            }
        }
        HCCL_VM_INFO(
            "[CCU_RESOURCE_STATS] report written, path={}, ops={}, rows={}, gsaApplicable={}, topoLabels={}", csvPath,
            data.size(), rowCount, gsaApplicable, topoLabels.size());
        return HcclResult::HCCL_SUCCESS;
    }

} // namespace CcuResourceStats
} // namespace HcclSim
