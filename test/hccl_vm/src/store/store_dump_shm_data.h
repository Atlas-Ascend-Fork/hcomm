/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DUMP_FLAG_DATA_H
#define DUMP_FLAG_DATA_H

#include <cstdint>
#include <map>
#include <vector>
#include "rt_external_kernel.h"
#include "sim_binary_data_type_pub.h"
#include "sim_common_defs.h"
#include "operation_data/operation_data_types.h"

namespace HcclSim {
std::string GenDataId();
HcclVmResult DumpDataToFile(const std::string& dataId);

HcclVmResult DumpHcclVmFlagData(HcclSim::HcclVmFlagData& flagData);
HcclVmResult GetHcclVmFlagData(HcclSim::HcclVmFlagData& waitFlag);
HcclVmResult DumpHcclVmSynthesisData(
    const std::string& dataId, const sim::operation::OpExecutionKey& key,
    const std::map<uint32_t, uint32_t>& deviceToRank);
HcclVmResult DumpHcclVmInstrData(const std::string& dataId, const std::map<uint32_t, uint32_t>& deviceToRank);
HcclVmResult DumpHcclVmTask(
    const std::string& dataId, const sim::operation::OpExecutionKey& key,
    const std::map<uint32_t, uint32_t>& deviceToRank);
HcclVmResult CreateChannelInfo(HcclVmSynData& hvmSynData, const std::map<uint32_t, uint32_t>& deviceToRank);
HcclVmResult CreateJettyInfo(HcclVmSynData& hvmSynData, const std::map<uint32_t, uint32_t>& deviceToRank);
HcclVmResult CreateMemoryInfo(HcclVmSynData& hvmSynData, const sim::operation::OpMemInfoTab& memInfo, uint32_t rankId);
HcclVmResult CreateSimTaskMetaData(
    HcclVmTaskMetaData& hvmTaskMetaData,
    const std::map<uint32_t, std::vector<sim::operation::CompositeOpDetail>>& compositeDataMap,
    const std::map<uint32_t, uint32_t>& deviceToRank);
HcclVmResult CreateSimSynData(
    HcclVmSynData& hvmSynData, const sim::operation::OpExecutionKey& key,
    const std::map<uint32_t, uint32_t>& deviceToRank);
HcclVmResult GenCaModelCcuInstr(hcomm::CcuRep::CcuInstr* instrData, uint32_t instrCnt, uint32_t rankId, uint32_t dieId);
HcclVmResult GenCaModelCcuToml(rtCcuTaskInfo_t* taskInfo, uint32_t rankId, uint32_t streamId);
HcclVmResult GenCcuChannelJettyConfig(const std::map<uint32_t, uint32_t>& deviceToRank);
} // namespace HcclSim
#endif
