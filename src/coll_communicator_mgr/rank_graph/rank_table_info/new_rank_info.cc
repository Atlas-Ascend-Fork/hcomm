/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "new_rank_info.h"

#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include "const_val.h"
#include "adapter_error_manager_pub.h"
#include "common/hccl_log_keywords_pub.h"

namespace Hccl {
std::string NewRankInfo::Describe() const
{
    return StringFormat(
        "NewRankInfo[rankId=%d, localId=%d, replacedLocalId=%d, ranklevelInfos size=%d, device_port=%d, "
        "host_port=%d, tlsStatus=%d, hostDpuTlsStatus=%d]",
        rankId, localId, replacedLocalId, rankLevelInfos.size(), devicePort, hostPort, static_cast<int>(tlsStatus),
        static_cast<int>(hostDpuTlsStatus));
}

NewRankInfo::NewRankInfo(BinaryStream& binStream)
{
    binStream >> rankId >> localId >> replacedLocalId >> deviceId >> devicePort >> hostPort;
    HCCL_DEBUG("[NewRankInfo] localId[%d]", localId);
    size_t rankLevelNum;
    binStream >> rankLevelNum;
    for (u32 i = 0; i < rankLevelNum; i++) {
        RankLevelInfo levelInfo(binStream);
        rankLevelInfos.emplace_back(std::move(levelInfo));
    }
    ControlPlane controlPlanes(binStream);
    controlPlane = controlPlanes;
    binStream >> tlsStatus;
    binStream >> hostDpuTlsStatus;
}

HcclResult NewRankInfo::GetBinStream(bool isContainLoaId, BinaryStream& binStream) const
{
    if (rankLevelInfos.size() == 0) {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>({"0", "rankLevelInfos", "non-empty"}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] rankLevelInfos size is zero", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA));
        return HCCL_E_PARA;
    }
    if (isContainLoaId) {
        binStream << rankId << localId << replacedLocalId << deviceId << devicePort << hostPort;
    } else {
        binStream << rankId << INVALID_RANKID << INVALID_RANKID << deviceId << devicePort << hostPort;
    }
    binStream << rankLevelInfos.size();
    HCCL_INFO("[NewRankInfo] rankLevelInfos size[%u], rankId[%d]", rankLevelInfos.size(), rankId);
    for (auto& it : rankLevelInfos) {
        CHK_RET(it.GetBinStream(binStream));
    }
    TRY_CATCH_RETURN(controlPlane.GetBinStream(binStream););
    binStream << tlsStatus;
    binStream << hostDpuTlsStatus;
    return HCCL_SUCCESS;
}

} // namespace Hccl
