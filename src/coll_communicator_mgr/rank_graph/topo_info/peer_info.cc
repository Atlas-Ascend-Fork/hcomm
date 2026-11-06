/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "peer_info.h"
#include "json_parser.h"
#include "rank_table_report_macro.h"

namespace Hccl {

HcclResult PeerInfo::Deserialize(const nlohmann::json& peerInfoJson)
{
    HcclResult ret = GetJsonPropertyUInt(peerInfoJson, "local_id", true, 0, localId);
    if (ret != HCCL_SUCCESS) {
        ReportJsonConfigError(peerInfoJson, "local_id", "0 ~ UINT32_MAX", RankTableSource::RANKTABLE);
        return ret;
    }

    if (localId > MAX_PEER_LOCAL_ID) {
        HCCL_ERROR("[%s] localId[%u] is out of range [0, %u].", __func__, localId, MAX_PEER_LOCAL_ID);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

std::string PeerInfo::Describe() const { return StringFormat("PeerInfo{local_id=%u}", localId); }

void PeerInfo::GetBinStream(BinaryStream& binaryStream) const { binaryStream << localId; }

PeerInfo::PeerInfo(BinaryStream& binaryStream) { binaryStream >> localId; }

} // namespace Hccl
