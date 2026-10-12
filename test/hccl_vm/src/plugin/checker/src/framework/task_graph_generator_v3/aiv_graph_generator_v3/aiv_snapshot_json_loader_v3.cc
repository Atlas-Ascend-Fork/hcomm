/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aiv_snapshot_json_loader_v3.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <nlohmann_json/json.hpp>
#include <sstream>
#include <sys/stat.h>

#include "sim_common_api.h"
#include "sim_log.h"

namespace HcclSim {
namespace TaskGraphGeneratorV3 {
    namespace {
        using Json = nlohmann::json;

        constexpr const char* AIV_TASK_FILE_PREFIX = "hcclvm_aiv_device";
        constexpr const char* AIV_TASK_FILE_LAUNCH_MARKER = "_launch";
        constexpr const char* AIV_TASK_FILE_SUFFIX = "_task.json";

        bool FileExists(const std::string& path)
        {
            struct stat st {};
            return stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
        }

        void SetError(std::string& errorMessage, const std::string& message) { errorMessage = message; }

        std::string ResolveAivTaskFilePath(DeviceId deviceId, uint64_t launchIndex)
        {
            namespace fs = std::filesystem;
            const fs::path dataDir = InstallPath::ResolveToInstallRoot("data");

            return (dataDir
                    / (std::string(AIV_TASK_FILE_PREFIX) + std::to_string(deviceId) + AIV_TASK_FILE_LAUNCH_MARKER
                       + std::to_string(launchIndex) + AIV_TASK_FILE_SUFFIX))
                .string();
        }

        bool CheckObjectField(const Json& json, const char* fieldName, std::string& errorMessage)
        {
            if (!json.is_object() || !json.contains(fieldName) || !json[fieldName].is_object()) {
                SetError(errorMessage, std::string("missing object field: ") + fieldName);
                return false;
            }
            return true;
        }

        bool CheckArrayField(const Json& json, const char* fieldName, std::string& errorMessage)
        {
            if (!json.is_object() || !json.contains(fieldName) || !json[fieldName].is_array()) {
                SetError(errorMessage, std::string("missing array field: ") + fieldName);
                return false;
            }
            return true;
        }

        bool ParseDataSlice(
            const Json& payload, const char* fieldName, AivDataSliceV3& slice, std::string& errorMessage,
            bool requireVirtualAddr)
        {
            if (!CheckObjectField(payload, fieldName, errorMessage)) {
                return false;
            }
            const Json& sliceJson = payload[fieldName];
            if (!sliceJson.contains("offset") || !sliceJson["offset"].is_number_unsigned()) {
                SetError(errorMessage, std::string("missing logical offset field: ") + fieldName);
                return false;
            }
            slice.type = static_cast<AivBufferTypeV3>(
                sliceJson.value("bufferType", static_cast<uint32_t>(AivBufferTypeV3::INVALID)));
            slice.deviceId = sliceJson.value("deviceId", INVALID_DEVICE_ID);
            slice.offset = sliceJson["offset"].get<uint64_t>();
            slice.virtualAddr = sliceJson.value("virtualAddr", 0ULL);
            if (requireVirtualAddr && slice.virtualAddr == 0ULL) {
                // 数据切片（MEM_COPY/REDUCE 的 src/dst）的 virtualAddr 由生产端保证非零（ResolveGlobalDataSlice
                // 的全部成功路径都会 SetVirtualAddr）；出现 0 说明快照异常，直接失败，不做静默回退。
                SetError(errorMessage, std::string("missing or zero virtualAddr field: ") + fieldName);
                return false;
            }
            slice.size = sliceJson.value("size", 0ULL);
            return true;
        }

        std::vector<uint32_t> ParseTaskIds(const Json& payload, const char* fieldName)
        {
            std::vector<uint32_t> taskIds;
            if (!payload.is_object() || !payload.contains(fieldName) || !payload[fieldName].is_array()) {
                return taskIds;
            }
            for (const auto& taskIdJson : payload[fieldName]) {
                if (taskIdJson.is_number_unsigned()) {
                    taskIds.push_back(taskIdJson.get<uint32_t>());
                }
            }
            return taskIds;
        }

        bool ParseRuntimeTask(const Json& taskJson, AivRuntimeTaskV3& task, std::string& errorMessage)
        {
            if (!taskJson.is_object()) {
                SetError(errorMessage, "runtime task is not object");
                return false;
            }

            task = AivRuntimeTaskV3{};
            task.taskType = static_cast<AivRuntimeTaskTypeV3>(
                taskJson.value("taskType", static_cast<uint32_t>(AivRuntimeTaskTypeV3::INVALID_TYPE)));
            task.taskId = taskJson.value("taskId", std::numeric_limits<uint32_t>::max());
            task.rankId = taskJson.value("rankId", INVALID_RANK_ID);
            task.blockId = taskJson.value("blockId", std::numeric_limits<uint32_t>::max());
            task.curPipe = taskJson.value("curPipe", std::numeric_limits<uint32_t>::max());
            task.deviceId = taskJson.value("deviceId", INVALID_DEVICE_ID);
            task.commId = taskJson.value("commId", 0ULL);

            const Json payload = taskJson.value("payload", Json::object());
            switch (task.taskType) {
                case AivRuntimeTaskTypeV3::MEM_COPY:
                    if (!ParseDataSlice(payload, "src", task.src, errorMessage, true)
                        || !ParseDataSlice(payload, "dst", task.dst, errorMessage, true)) {
                        return false;
                    }
                    return true;
                case AivRuntimeTaskTypeV3::REDUCE:
                    if (!ParseDataSlice(payload, "src", task.src, errorMessage, true)
                        || !ParseDataSlice(payload, "dst", task.dst, errorMessage, true)) {
                        return false;
                    }
                    task.dataType = payload.value("dataType", 0U);
                    task.reduceOp = payload.value("reduceOp", 0U);
                    return true;
                case AivRuntimeTaskTypeV3::SET_FLAG:
                case AivRuntimeTaskTypeV3::WAIT_FLAG:
                    task.srcPipe = payload.value("srcPipe", std::numeric_limits<uint32_t>::max());
                    task.dstPipe = payload.value("dstPipe", std::numeric_limits<uint32_t>::max());
                    task.eventId = payload.value("eventId", 0);
                    return true;
                case AivRuntimeTaskTypeV3::PIPE_BARRIER:
                    task.pipeType = payload.value("pipeType", std::numeric_limits<uint32_t>::max());
                    task.barrierGroupTaskIds = ParseTaskIds(payload, "barrierGroupTaskIds");
                    return true;
                case AivRuntimeTaskTypeV3::SYNC_ALL:
                    task.syncRound = payload.value("syncRound", std::numeric_limits<uint32_t>::max());
                    return true;
                case AivRuntimeTaskTypeV3::SEND_FLAG:
                    if (!ParseDataSlice(payload, "flagBuffer", task.flagBuffer, errorMessage, false)) {
                        return false;
                    }
                    task.targetRank = payload.value("targetRank", INVALID_RANK_ID);
                    task.flagValue = payload.value("flagValue", 0);
                    return true;
                case AivRuntimeTaskTypeV3::RECV_FLAG:
                    if (!ParseDataSlice(payload, "flagBuffer", task.flagBuffer, errorMessage, false)) {
                        return false;
                    }
                    task.targetRank = payload.value("targetRank", INVALID_RANK_ID);
                    task.flagValue = payload.value("flagValue", 0);
                    return true;
                default:
                    SetError(
                        errorMessage,
                        "unsupported AIV runtime task type: " + std::to_string(static_cast<uint32_t>(task.taskType)));
                    return false;
            }
        }

        bool ParseTaskArray(const Json& arrayJson, std::vector<AivRuntimeTaskV3>& tasks, std::string& errorMessage)
        {
            tasks.clear();
            if (!arrayJson.is_array()) {
                SetError(errorMessage, "runtime task array is not array");
                return false;
            }
            tasks.reserve(arrayJson.size());
            for (const auto& taskJson : arrayJson) {
                AivRuntimeTaskV3 task;
                if (!ParseRuntimeTask(taskJson, task, errorMessage)) {
                    return false;
                }
                tasks.push_back(std::move(task));
            }
            return true;
        }

        bool ParseSnapshot(const Json& rankJson, AivRuntimeTaskSnapshotV3& snapshot, std::string& errorMessage)
        {
            if (!rankJson.is_object()) {
                SetError(errorMessage, "AIV snapshot root is not object");
                return false;
            }
            if (!CheckArrayField(rankJson, "aivCores", errorMessage)) {
                return false;
            }

            snapshot.rankId = rankJson.value("rank", INVALID_RANK_ID);
            snapshot.deviceId = rankJson.value("deviceId", INVALID_DEVICE_ID);
            snapshot.commId = rankJson.value("commId", 0ULL);
            snapshot.launchIndex = rankJson.value("launchIndex", 0ULL);
            snapshot.rankSize = rankJson.value("rankSize", 0U);
            snapshot.inBufferSize = rankJson.value("inBufferSize", 0ULL);
            snapshot.outBufferSize = rankJson.value("outBufferSize", 0ULL);
            snapshot.cclBufferSize = rankJson.value("cclBufferSize", 0ULL);
            snapshot.aivCommInfoSize = rankJson.value("aivCommInfoSize", 0ULL);
            snapshot.ubBufferSize = rankJson.value("ubBufferSize", 0ULL);
            snapshot.blocks.clear();

            const Json& cores = rankJson["aivCores"];
            snapshot.blocks.reserve(cores.size());
            for (const auto& blockJson : cores) {
                if (!blockJson.is_object()) {
                    SetError(errorMessage, "AIV core entry is not object");
                    return false;
                }
                AivRuntimeBlockSnapshotV3 block;
                block.blockIdx = blockJson.value("blockIdx", std::numeric_limits<uint32_t>::max());
                if (!ParseTaskArray(blockJson.value("scalarTasks", Json::array()), block.scalarTasks, errorMessage)
                    || !ParseTaskArray(blockJson.value("mte2Tasks", Json::array()), block.mte2Tasks, errorMessage)
                    || !ParseTaskArray(blockJson.value("mte3Tasks", Json::array()), block.mte3Tasks, errorMessage)) {
                    return false;
                }
                snapshot.blocks.push_back(std::move(block));
            }
            return true;
        }
    } // namespace

    HcclResult AivSnapshotJsonLoaderV3::LoadByDeviceAndLaunch(
        DeviceId deviceId, uint64_t launchIndex, AivRuntimeTaskSnapshotV3& snapshot, std::string& errorMessage) const
    {
        snapshot = AivRuntimeTaskSnapshotV3{};
        const std::string filePath = ResolveAivTaskFilePath(deviceId, launchIndex);
        if (!FileExists(filePath)) {
            SetError(errorMessage, "AIV task json file does not exist: " + filePath);
            return HCCL_E_NOT_FOUND;
        }

        std::ifstream ifs(filePath.c_str());
        if (!ifs.is_open()) {
            SetError(errorMessage, "failed to open AIV task json file: " + filePath);
            return HCCL_E_INTERNAL;
        }

        const Json rankJson = Json::parse(ifs, nullptr, false);
        if (rankJson.is_discarded()) {
            SetError(errorMessage, "failed to parse AIV task json file: " + filePath);
            return HCCL_E_PARA;
        }

        if (!ParseSnapshot(rankJson, snapshot, errorMessage)) {
            errorMessage += ", file=" + filePath;
            return HCCL_E_PARA;
        }
        snapshot.filePath = filePath;

        if (snapshot.launchIndex != launchIndex) {
            SetError(
                errorMessage, "launchIndex mismatch in AIV task json, file=" + filePath + ", expected="
                                  + std::to_string(launchIndex) + ", actual=" + std::to_string(snapshot.launchIndex));
            return HCCL_E_PARA;
        }
        if (snapshot.deviceId != deviceId) {
            SetError(
                errorMessage, "device id mismatch in AIV task json, file=" + filePath + ", expected="
                                  + std::to_string(deviceId) + ", actual=" + std::to_string(snapshot.deviceId));
            return HCCL_E_PARA;
        }

        HCCL_VM_INFO(
            "Loaded AIV snapshot, deviceId={}, logicalRankId={}, launchIndex={}, blockCount={}, file={}", deviceId,
            snapshot.rankId, launchIndex, snapshot.blocks.size(), filePath);
        return HCCL_SUCCESS;
    }

} // namespace TaskGraphGeneratorV3
} // namespace HcclSim
