/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_semantics_checker.h"

#include <map>

#include "base.h"
#include "check_utils.h"
#include "sim_log.h"
#include "utils/dump/dump_json_utils.h"
#include "utils/error_codes.h"

namespace HcclSim {
HcclResult TaskCheckReduceScatterSemantics(
    const PhysicalMemorySemantics& memorySemantics, const DeviceBufferAddressLayouts& bufferLayouts, uint64_t dataSize,
    HcclReduceOp reduceType, const std::vector<DeviceId>& rankToDevice)
{
    const uint32_t rankSize = memorySemantics.size();

    for (const auto& [deviceId, mem] : memorySemantics) {
        const RankId deviceRank = FindRankIndexByDeviceId(rankToDevice, deviceId);
        if (deviceRank == TaskGraphGeneratorV3::INVALID_RANK_ID) {
            // rank 映射与参与设备集不一致属于校验环境异常，继续推转会按错误的 rank 语义校验。
            HCCL_VM_ERROR(
                "{} Device {} is not found in the rank-to-device mapping, cannot resolve "
                "its rank index for the semantic check.",
                MakeErrorCodeText(ErrorCode::CHECKER_RUNTIME_ERROR), deviceId);
            return HcclResult::HCCL_E_PARA;
        }
        uint64_t outputBase = 0;
        uint64_t outputBufferSize = 0;
        GetSemanticBufferExtent(bufferLayouts, deviceId, BufferType::OUTPUT, outputBase, outputBufferSize);
        uint64_t totalSize = 0;
        for (const auto& [eleAddr, original] : mem) {
            BufferSemantic ele = original;
            if (!ClipSemanticToBuffer(ele, outputBase, outputBufferSize)) {
                continue;
            }
            if (ele.startAddr >= outputBase + dataSize) {
                // 已裁剪到注册缓冲区；超出期望结果大小的非空语义段仍应报错。
                // 注册大小未知时保留原有兼容行为。
                if (outputBufferSize != 0 && ele.size != 0) {
                    HCCL_VM_ERROR(
                        "{} ReduceScatter output for device {} contains unexpected extra data: "
                        "range [0x{:x},0x{:x}) starts beyond the expected result size 0x{:x} but inside "
                        "the output buffer (size 0x{:x}).\nCurrent result range detail:\n{}",
                        MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SIZE_ERROR), deviceId, ele.startAddr,
                        ele.startAddr + ele.size, dataSize, outputBufferSize, ele.Describe());
                    return HcclResult::HCCL_E_PARA;
                }
                break;
            }
            const uint64_t rangeEnd = ele.startAddr + ele.size;
            if (ele.startAddr != outputBase + totalSize) {
                HCCL_VM_ERROR(
                    "{} ReduceScatter output for device {} should continue at 0x{:x}, "
                    "but the next actual range starts at 0x{:x} (actual range: [0x{:x},0x{:x}))."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), deviceId, outputBase + totalSize,
                    ele.startAddr, ele.startAddr, rangeEnd, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.size() > 1 && ele.reduceType != reduceType) {
                HCCL_VM_ERROR(
                    "{} ReduceScatter output range [0x{:x},0x{:x}) for device {} uses reduce "
                    "mode {}, but the operator is set to reduce mode {}."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_REDUCE_ERROR), ele.startAddr, rangeEnd, deviceId,
                    DumpReduceOpToString(ele.reduceType), DumpReduceOpToString(reduceType), ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.size() != rankSize) {
                HCCL_VM_ERROR(
                    "{} ReduceScatter output range [0x{:x},0x{:x}) for device {} expected "
                    "{} source ranks but got {}."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_REDUCE_ERROR), ele.startAddr, rangeEnd, deviceId,
                    rankSize, ele.srcBufs.size(), ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            for (auto& srcBuf : ele.srcBufs) {
                if (srcBuf.bufType != BufferType::INPUT) {
                    HCCL_VM_ERROR(
                        "{} ReduceScatter output range [0x{:x},0x{:x}) for device {} comes "
                        "from device {} with buffer type {}, but it should come from INPUT."
                        "\nCurrent result range detail:\n{}",
                        MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), ele.startAddr, rangeEnd, deviceId,
                        srcBuf.deviceId, BufferTypeToString(srcBuf.bufType), ele.Describe());
                    return HcclResult::HCCL_E_PARA;
                }

                const uint64_t expectedSrcAddr
                    = GetSemanticBufferBase(bufferLayouts, srcBuf.deviceId, BufferType::INPUT) + deviceRank * dataSize
                      + totalSize;
                if (srcBuf.srcAddr != expectedSrcAddr) {
                    HCCL_VM_ERROR(
                        "{} ReduceScatter output range [0x{:x},0x{:x}) for device {} should "
                        "come from device {}.INPUT at 0x{:x}, but it actually comes from device {}.INPUT "
                        "at 0x{:x}.\nCurrent result range detail:\n{}",
                        MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), ele.startAddr, rangeEnd, deviceId,
                        srcBuf.deviceId, expectedSrcAddr, srcBuf.deviceId, srcBuf.srcAddr, ele.Describe());
                    return HcclResult::HCCL_E_PARA;
                }
            }
            totalSize += ele.size;
        }
        if (totalSize != dataSize) {
            HCCL_VM_ERROR(
                "{} ReduceScatter output for device {} ends too early: the checker validated "
                "0x{:x} bytes in total, but the expected result size is 0x{:x}.",
                MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), deviceId, totalSize, dataSize);
            return HcclResult::HCCL_E_PARA;
        }
    }

    return HcclResult::HCCL_SUCCESS;
}
} // namespace HcclSim
