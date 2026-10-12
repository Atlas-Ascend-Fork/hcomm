/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "allgather_semantics_checker.h"

#include <map>

#include "check_utils.h"
#include "sim_common.h"
#include "sim_log.h"
#include "utils/dump/dump_json_utils.h"
#include "utils/error_codes.h"

namespace HcclSim {
HcclResult TaskCheckAllGatherSemantics(
    const PhysicalMemorySemantics& memorySemantics, const DeviceBufferAddressLayouts& bufferLayouts, uint64_t dataSize,
    const std::vector<DeviceId>& rankToDevice)
{
    const uint32_t rankSize = memorySemantics.size();
    // 每个设备的期望结果窗口大小相同：rankSize 份 dataSize 顺序拼接。
    const uint64_t expectedOutputSize = dataSize * rankSize;

    for (const auto& [deviceId, mem] : memorySemantics) {
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
                        "{} AllGather output for device {} contains unexpected extra data: range "
                        "[0x{:x},0x{:x}) starts beyond the expected result size 0x{:x} but inside the "
                        "output buffer (size 0x{:x}).\nCurrent result range detail:\n{}",
                        MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SIZE_ERROR), deviceId, ele.startAddr,
                        ele.startAddr + ele.size, expectedOutputSize, outputBufferSize, ele.Describe());
                    return HcclResult::HCCL_E_PARA;
                }
                break;
            }
            if (curRankIdx >= rankSize) {
                HCCL_VM_ERROR(
                    "{} AllGather output for device {} contains data after all source ranks "
                    "have been consumed.\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SIZE_ERROR), deviceId, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }
            const uint64_t rangeEnd = ele.startAddr + ele.size;
            if (ele.startAddr != outputBase + totalSize) {
                HCCL_VM_ERROR(
                    "{} AllGather output for device {} should continue at 0x{:x}, "
                    "but the next actual range starts at 0x{:x} (actual range: [0x{:x},0x{:x}))."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), deviceId, outputBase + totalSize,
                    ele.startAddr, ele.startAddr, rangeEnd, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.size() != 1) {
                HCCL_VM_ERROR(
                    "{} AllGather output range [0x{:x},0x{:x}) for device {} should come "
                    "from exactly one source, but it actually comes from {} sources."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_REDUCE_ERROR), ele.startAddr, rangeEnd, deviceId,
                    ele.srcBufs.size(), ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            const auto& srcBuf = *ele.srcBufs.begin();

            if (srcBuf.deviceId != rankToDevice[curRankIdx]) {
                HCCL_VM_ERROR(
                    "{} AllGather output range [0x{:x},0x{:x}) for device {} should come from "
                    "source index {}, but it actually comes from device {}."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), ele.startAddr, rangeEnd, deviceId,
                    curRankIdx, srcBuf.deviceId, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (srcBuf.bufType != BufferType::INPUT) {
                HCCL_VM_ERROR(
                    "{} AllGather output range [0x{:x},0x{:x}) for device {} should come from "
                    "INPUT, but it actually comes from device {} with buffer type {}."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), ele.startAddr, rangeEnd, deviceId,
                    srcBuf.deviceId, BufferTypeToString(srcBuf.bufType), ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            const uint64_t expectedSrcAddr
                = GetSemanticBufferBase(bufferLayouts, srcBuf.deviceId, BufferType::INPUT) + curDataSize;
            if (srcBuf.srcAddr != expectedSrcAddr) {
                HCCL_VM_ERROR(
                    "{} AllGather output range [0x{:x},0x{:x}) for device {} should come from "
                    "source index {} at source address 0x{:x}, but the actual source address is 0x{:x}."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), ele.startAddr, rangeEnd, deviceId,
                    srcBuf.deviceId, expectedSrcAddr, srcBuf.srcAddr, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            curDataSize += ele.size;
            if (curDataSize == dataSize) {
                curDataSize = 0;
                curRankIdx++;
            } else if (curDataSize > dataSize) {
                HCCL_VM_ERROR(
                    "{} AllGather data collected from source index {} for device {} becomes larger "
                    "than expected after outputRange [0x{:x},0x{:x}). The accumulated size is 0x{:x}, "
                    "but the expected size from this source is 0x{:x}."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SIZE_ERROR), curRankIdx, deviceId, ele.startAddr,
                    rangeEnd, curDataSize, dataSize, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }
            totalSize += ele.size;
        }
        if (totalSize != dataSize * rankSize) {
            HCCL_VM_ERROR(
                "{} AllGather output for device {} ends too early: the checker validated "
                "0x{:x} bytes in total, but the expected total result size is 0x{:x} bytes.",
                MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), deviceId, totalSize, dataSize * rankSize);
            return HcclResult::HCCL_E_PARA;
        }
    }

    return HcclResult::HCCL_SUCCESS;
}
} // namespace HcclSim
