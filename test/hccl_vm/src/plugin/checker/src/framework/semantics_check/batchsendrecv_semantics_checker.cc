/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "batchsendrecv_semantics_checker.h"

#include <map>

#include "base.h"
#include "check_utils.h"
#include "sim_log.h"
#include "utils/dump/dump_json_utils.h"
#include "utils/error_codes.h"

namespace HcclSim {
HcclResult TaskCheckBatchSendRecvSemantics(
    const PhysicalMemorySemantics& memorySemantics, const DeviceBufferAddressLayouts& bufferLayouts,
    uint32_t expectedRankSize, uint64_t dataSize, const std::vector<DeviceId>& rankToDevice)
{
    if (expectedRankSize == 0 || memorySemantics.size() != expectedRankSize) {
        HCCL_VM_ERROR(
            "{} BatchSendRecv rank set size mismatch: expected {}, actual {}.",
            MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), expectedRankSize, memorySemantics.size());
        return HcclResult::HCCL_E_PARA;
    }

    // 每个设备的期望结果窗口大小相同：expectedRankSize 份 dataSize 顺序拼接。
    const uint64_t expectedOutputSize = dataSize * expectedRankSize;
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
        uint32_t curRankIdx = 0;
        uint64_t curDataSize = 0;
        for (const auto& [eleAddr, original] : mem) {
            BufferSemantic ele = original;
            if (!ClipSemanticToBuffer(ele, outputBase, outputBufferSize)) {
                continue;
            }
            if (ele.startAddr >= outputBase + expectedOutputSize) {
                // 已裁剪到注册缓冲区；超出期望结果大小的非空语义段仍应报错。
                // 注册大小未知时保留原有兼容行为。
                if (outputBufferSize != 0 && ele.size != 0) {
                    HCCL_VM_ERROR(
                        "{} BatchSendRecv output for device {} contains unexpected extra data: "
                        "range [0x{:x},0x{:x}) starts beyond the expected result size 0x{:x} but inside "
                        "the output buffer (size 0x{:x}).\nCurrent result range detail:\n{}",
                        MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SIZE_ERROR), deviceId, ele.startAddr,
                        ele.startAddr + ele.size, expectedOutputSize, outputBufferSize, ele.Describe());
                    return HcclResult::HCCL_E_PARA;
                }
                break;
            }
            if (curRankIdx >= expectedRankSize) {
                HCCL_VM_ERROR(
                    "{} BatchSendRecv output for device {} contains data after all source ranks "
                    "have been consumed.\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SIZE_ERROR), deviceId, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }
            const uint64_t rangeEnd = ele.startAddr + ele.size;
            if (ele.startAddr != outputBase + totalSize) {
                HCCL_VM_ERROR(
                    "{} BatchSendRecv output for device {} should continue at 0x{:x}, "
                    "but the next actual range starts at 0x{:x} (actual range: [0x{:x},0x{:x}))."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), deviceId, outputBase + totalSize,
                    ele.startAddr, ele.startAddr, rangeEnd, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.size() != 1) {
                HCCL_VM_ERROR(
                    "{} BatchSendRecv output range [0x{:x},0x{:x}) for device {} should "
                    "come from exactly one source, but it actually comes from {} sources."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_REDUCE_ERROR), ele.startAddr, rangeEnd, deviceId,
                    ele.srcBufs.size(), ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            const auto& srcBuf = *ele.srcBufs.begin();
            if (srcBuf.deviceId != rankToDevice[curRankIdx]) {
                HCCL_VM_ERROR(
                    "{} BatchSendRecv output range [0x{:x},0x{:x}) for device {} should come "
                    "from source index {}, but it actually comes from device {}."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), ele.startAddr, rangeEnd, deviceId,
                    curRankIdx, srcBuf.deviceId, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (srcBuf.bufType != BufferType::INPUT) {
                HCCL_VM_ERROR(
                    "{} BatchSendRecv output range [0x{:x},0x{:x}) for device {} should come "
                    "from source{}.INPUT, but it actually comes from device{}.{}."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), ele.startAddr, rangeEnd, deviceId,
                    curRankIdx, srcBuf.deviceId, BufferTypeToString(srcBuf.bufType), ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            const uint64_t expectedSrcAddr = GetSemanticBufferBase(bufferLayouts, srcBuf.deviceId, BufferType::INPUT)
                                             + dataSize * deviceRank + curDataSize;
            if (srcBuf.srcAddr != expectedSrcAddr) {
                HCCL_VM_ERROR(
                    "{} BatchSendRecv output range [0x{:x},0x{:x}) for device {} should take "
                    "data from source index {} at input address 0x{:x}, but it actually takes data "
                    "from device {} at input address 0x{:x}."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), ele.startAddr, rangeEnd, deviceId,
                    curRankIdx, expectedSrcAddr, srcBuf.deviceId, srcBuf.srcAddr, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }
            curDataSize += ele.size;
            if (curDataSize == dataSize) {
                curDataSize = 0;
                curRankIdx++;
            } else if (curDataSize > dataSize) {
                HCCL_VM_ERROR(
                    "{} BatchSendRecv data collected from source index {} for device {} becomes "
                    "larger than expected after outputRange [0x{:x},0x{:x}). The accumulated size is "
                    "0x{:x}, but the expected size from this source is 0x{:x}."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SIZE_ERROR), curRankIdx, deviceId, ele.startAddr,
                    rangeEnd, curDataSize, dataSize, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }
            totalSize += ele.size;
        }
        // 如果curRankIdx等于expectedRankSize，表示已经接受到其他所有rank的数据
        if (curRankIdx != expectedRankSize) {
            HCCL_VM_ERROR(
                "{} BatchSendRecv output for device {} ends too early. The checker has "
                "validated 0x{:x} bytes in total, but the expected total size is 0x{:x}.",
                MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), deviceId, totalSize, dataSize * expectedRankSize);
            return HcclResult::HCCL_E_PARA;
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TaskCheckBatchSendRecvRingSemantics(
    const PhysicalMemorySemantics& memorySemantics, const DeviceBufferAddressLayouts& bufferLayouts,
    uint32_t expectedRankSize, uint64_t dataSize, const std::vector<DeviceId>& rankToDevice)
{
    if (expectedRankSize < 2 || memorySemantics.size() != expectedRankSize) {
        HCCL_VM_ERROR(
            "{} BatchSendRecv ring rank set size mismatch: expected {}, actual {}.",
            MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), expectedRankSize, memorySemantics.size());
        return HcclResult::HCCL_E_PARA;
    }

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
        const uint32_t expectedSrcRank = (deviceRank + expectedRankSize - 1U) % expectedRankSize;
        uint64_t outputBase = 0;
        uint64_t outputBufferSize = 0;
        GetSemanticBufferExtent(bufferLayouts, deviceId, BufferType::OUTPUT, outputBase, outputBufferSize);
        uint64_t totalSize = 0;
        for (const auto& outputEntry : mem) {
            BufferSemantic output = outputEntry.second;
            if (!ClipSemanticToBuffer(output, outputBase, outputBufferSize)) {
                continue;
            }
            if (output.startAddr >= outputBase + dataSize) {
                // 已裁剪到注册缓冲区；超出期望结果大小的非空语义段仍应报错。
                // 注册大小未知时保留原有兼容行为。
                if (outputBufferSize != 0 && output.size != 0) {
                    HCCL_VM_ERROR(
                        "{} BatchSendRecv ring output for device {} contains unexpected extra data: "
                        "range [0x{:x},0x{:x}) starts beyond the expected result size 0x{:x} but inside "
                        "the output buffer (size 0x{:x}).\nCurrent result range detail:\n{}",
                        MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SIZE_ERROR), deviceId, output.startAddr,
                        output.startAddr + output.size, dataSize, outputBufferSize, output.Describe());
                    return HcclResult::HCCL_E_PARA;
                }
                break;
            }
            const uint64_t rangeEnd = output.startAddr + output.size;
            if (rangeEnd < output.startAddr || output.startAddr != outputBase + totalSize
                || output.size > dataSize - totalSize) {
                HCCL_VM_ERROR(
                    "{} BatchSendRecv ring output for device {} is not a contiguous range of size "
                    "0x{:x}; next actual range is [0x{:x},0x{:x}).\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SIZE_ERROR), deviceId, dataSize, output.startAddr,
                    rangeEnd, output.Describe());
                return HcclResult::HCCL_E_PARA;
            }
            if (output.srcBufs.size() != 1) {
                HCCL_VM_ERROR(
                    "{} BatchSendRecv ring output range [0x{:x},0x{:x}) for device {} should "
                    "come from exactly one source, but it comes from {} sources.\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_REDUCE_ERROR), output.startAddr, rangeEnd, deviceId,
                    output.srcBufs.size(), output.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            const auto& srcBuf = *output.srcBufs.begin();
            const uint64_t expectedSrcAddr = GetSemanticBufferBase(bufferLayouts, srcBuf.deviceId, BufferType::INPUT)
                                             + output.startAddr - outputBase;
            if (srcBuf.deviceId != rankToDevice[expectedSrcRank] || srcBuf.bufType != BufferType::INPUT
                || srcBuf.srcAddr != expectedSrcAddr) {
                HCCL_VM_ERROR(
                    "{} BatchSendRecv ring output range [0x{:x},0x{:x}) for device {} should "
                    "come from source index {}.INPUT at address 0x{:x}, but it comes from device{}.{} at address "
                    "0x{:x}.\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), output.startAddr, rangeEnd, deviceId,
                    expectedSrcRank, expectedSrcAddr, srcBuf.deviceId, BufferTypeToString(srcBuf.bufType),
                    srcBuf.srcAddr, output.Describe());
                return HcclResult::HCCL_E_PARA;
            }
            totalSize += output.size;
        }
        // 输出缺失（没有任何段落在结果窗口内）也会在这里以 totalSize=0 暴露。
        if (totalSize != dataSize) {
            HCCL_VM_ERROR(
                "{} BatchSendRecv ring output for device {} has size 0x{:x}, expected 0x{:x}.",
                MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), deviceId, totalSize, dataSize);
            return HcclResult::HCCL_E_PARA;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}
} // namespace HcclSim
