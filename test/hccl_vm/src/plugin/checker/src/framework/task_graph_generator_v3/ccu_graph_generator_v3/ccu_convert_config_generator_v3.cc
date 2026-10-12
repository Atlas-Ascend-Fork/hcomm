/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_convert_config_generator_v3.h"

#include <fstream>
#include <nlohmann_json/json.hpp>
#include <string>

#include "ccu_all_rank_param_recorder_v3.h"
#include "file_utils.h"
#include "sim_common_api.h"
#include "sim_log.h"
#include "utils/error_codes.h"

namespace HcclSim {
namespace TaskGraphGeneratorV3 {

    namespace {
        constexpr const char* CONVERT_CONFIG_FILE_NAME = "ccu_convert_config.json";
        constexpr const char* CONVERT_CONFIG_REL_DIR = "data";

        std::string MakeEntryKey(DeviceId deviceId, uint32_t dieId)
        {
            return "device" + std::to_string(deviceId) + "_die" + std::to_string(dieId);
        }
    } // namespace

    HcclResult DumpConvertConfig()
    {
        AllRankParamRecorder* recorder = AllRankParamRecorder::Global();
        if (recorder == nullptr) {
            return HCCL_E_PTR;
        }

        nlohmann::json doc;
        doc["_comment"] = "所有 device/die 的转换配置统一保存在本文件中。"
                          "key 为 \"device<deviceId>_die<dieId>\"（dieId 为 0-based），"
                          "value 为该 device/die 对应的 rule1/rule3 子配置。"
                          "由 checker 预执行产出（curXn / loopGroupRegSnapshot）。";

        nlohmann::json entries = nlohmann::json::object();

        // rule1 数据源：遍历 curXn 的所有 (deviceId, dieId)，取最大 xnId
        for (const auto& devicePair : recorder->curXn) {
            for (const auto& diePair : devicePair.second) {
                const DeviceId deviceId = devicePair.first;
                const uint32_t dieId = diePair.first;
                const uint16_t maxUsedXnId = recorder->GetMaxUsedXnId(deviceId, dieId);

                nlohmann::json entry;
                entry["rule1"]["max_used_xn_id"] = static_cast<int64_t>(maxUsedXnId);

                // rule3 数据源：从 loopGroupRegSnapshot 中提取该 (deviceId, dieId) 下的 LoopGroup 寄存器快照
                nlohmann::json regTable = nlohmann::json::object();
                auto regDeviceIt = recorder->loopGroupRegSnapshot.find(deviceId);
                if (regDeviceIt != recorder->loopGroupRegSnapshot.end()) {
                    auto regDieIt = regDeviceIt->second.find(dieId);
                    if (regDieIt != regDeviceIt->second.end()) {
                        for (const auto& pcPair : regDieIt->second) {
                            const uint32_t pc = pcPair.first;
                            const uint64_t xpValue = pcPair.second.first;
                            const uint64_t xmValue = pcPair.second.second;
                            nlohmann::json snap;
                            snap["xpId"] = static_cast<uint64_t>(xpValue);
                            snap["xmId"] = static_cast<uint64_t>(xmValue);
                            regTable[std::to_string(pc)] = std::move(snap);
                        }
                    }
                }
                entry["rule3"]["reg_value_table"] = std::move(regTable);

                entries[MakeEntryKey(deviceId, dieId)] = std::move(entry);
            }
        }

        // 补充：loopGroupRegSnapshot 中存在但 curXn 中不存在的 (deviceId, dieId)
        for (const auto& regDevicePair : recorder->loopGroupRegSnapshot) {
            for (const auto& regDiePair : regDevicePair.second) {
                const DeviceId deviceId = regDevicePair.first;
                const uint32_t dieId = regDiePair.first;
                const std::string key = MakeEntryKey(deviceId, dieId);
                if (entries.contains(key)) {
                    continue;
                }
                nlohmann::json entry;
                entry["rule1"]["max_used_xn_id"] = static_cast<int64_t>(UINT16_MAX);
                nlohmann::json regTable = nlohmann::json::object();
                for (const auto& pcPair : regDiePair.second) {
                    nlohmann::json snap;
                    snap["xpId"] = static_cast<uint64_t>(pcPair.second.first);
                    snap["xmId"] = static_cast<uint64_t>(pcPair.second.second);
                    regTable[std::to_string(pcPair.first)] = std::move(snap);
                }
                entry["rule3"]["reg_value_table"] = std::move(regTable);
                entries[key] = std::move(entry);
            }
        }

        doc["entries"] = std::move(entries);

        // 写入 hccl_vm_install/data/ccu_convert_config.json
        const std::string dirPath = InstallPath::ResolveToInstallRoot(CONVERT_CONFIG_REL_DIR);
        HcclResult ret = EnsureDirectory(dirPath);
        if (ret != HCCL_SUCCESS) {
            HCCL_VM_ERROR(
                "{} Failed to create directory for CCU convert config, dir={}",
                MakeErrorCodeText(ErrorCode::DUMP_FAILED), dirPath);
            return ret;
        }
        const std::string fullPath = HcclSim::JoinPath(dirPath, CONVERT_CONFIG_FILE_NAME);
        std::ofstream out(fullPath, std::ios::out | std::ios::trunc);
        if (!out.is_open()) {
            HCCL_VM_ERROR(
                "{} Failed to open CCU convert config file, file={}", MakeErrorCodeText(ErrorCode::DUMP_FAILED),
                fullPath);
            return HCCL_E_INTERNAL;
        }
        try {
            out << doc.dump(2) << std::endl;
        } catch (const std::exception& ex) {
            HCCL_VM_ERROR(
                "{} Failed to serialize CCU convert config JSON, file={}, reason={}",
                MakeErrorCodeText(ErrorCode::DUMP_FAILED), fullPath, ex.what());
            return HCCL_E_INTERNAL;
        }
        if (!out.good()) {
            HCCL_VM_ERROR(
                "{} Failed to write CCU convert config file, file={}", MakeErrorCodeText(ErrorCode::DUMP_FAILED),
                fullPath);
            return HCCL_E_INTERNAL;
        }

        HCCL_VM_INFO("CCU convert config JSON dumped, file={}, entryCount={}", fullPath, doc["entries"].size());
        return HCCL_SUCCESS;
    }

} // namespace TaskGraphGeneratorV3
} // namespace HcclSim
