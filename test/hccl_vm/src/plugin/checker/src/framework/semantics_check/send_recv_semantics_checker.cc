/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "send_recv_semantics_checker.h"

#include <map>

#include "base.h"
#include "check_utils.h"
#include "sim_log.h"
#include "utils/dump/dump_json_utils.h"
#include "utils/error_codes.h"

namespace HcclSim {
HcclResult TaskCheckSendRecvSemantics(
    const PhysicalMemorySemantics& memorySemantics, const DeviceBufferAddressLayouts& bufferLayouts, uint64_t dataSize,
    DeviceId srcDeviceId, DeviceId dstDeviceId, const std::vector<DeviceId>& rankToDevice)
{
    const uint32_t rankSize = memorySemantics.size();

    bool foundSrc = false;
    bool foundDst = false;
    for (const auto& [deviceId, mem] : memorySemantics) {
        if (deviceId == srcDeviceId) {
            foundSrc = true;
        }
        if (deviceId != dstDeviceId) {
            continue;
        }
        foundDst = true;

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
                        "{} Send/Recv output for rank {} contains unexpected extra data: range "
                        "[0x{:x},0x{:x}) starts beyond the expected result size 0x{:x} but inside the "
                        "output buffer (size 0x{:x}).\nCurrent result range detail:\n{}",
                        MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SIZE_ERROR), dstDeviceId, ele.startAddr,
                        ele.startAddr + ele.size, dataSize, outputBufferSize, ele.Describe());
                    return HcclResult::HCCL_E_PARA;
                }
                break;
            }
            const uint64_t rangeEnd = ele.startAddr + ele.size;
            if (ele.startAddr != outputBase + totalSize) {
                HCCL_VM_ERROR(
                    "{} Send/Recv output for rank {} should continue at 0x{:x}, "
                    "but the next actual range starts at 0x{:x} (actual range: [0x{:x},0x{:x}))."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), dstDeviceId, outputBase + totalSize,
                    ele.startAddr, ele.startAddr, rangeEnd, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.size() != 1) {
                HCCL_VM_ERROR(
                    "{} Send/Recv output range [0x{:x},0x{:x}) for rank {} should come from "
                    "exactly one source, but it actually comes from {} sources."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_REDUCE_ERROR), ele.startAddr, rangeEnd, dstDeviceId,
                    ele.srcBufs.size(), ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            const auto& srcBuf = *ele.srcBufs.begin();
            if (srcBuf.deviceId != srcDeviceId) {
                HCCL_VM_ERROR(
                    "{} Send/Recv output range [0x{:x},0x{:x}) for rank {} should come from "
                    "rank {}, but it actually comes from device {}."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), ele.startAddr, rangeEnd, dstDeviceId,
                    srcDeviceId, srcBuf.deviceId, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (srcBuf.bufType != BufferType::INPUT) {
                HCCL_VM_ERROR(
                    "{} Send/Recv output range [0x{:x},0x{:x}) for rank {} should come from "
                    "rank{}.INPUT, but it actually comes from device{}.{}."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), ele.startAddr, rangeEnd, dstDeviceId,
                    srcDeviceId, srcBuf.deviceId, BufferTypeToString(srcBuf.bufType), ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }
            const uint64_t expectedSrcAddr
                = GetSemanticBufferBase(bufferLayouts, srcBuf.deviceId, BufferType::INPUT) + totalSize;
            if (srcBuf.srcAddr != expectedSrcAddr) {
                HCCL_VM_ERROR(
                    "{} Send/Recv output range [0x{:x},0x{:x}) for rank {} should take data "
                    "from source rank {} at input address 0x{:x}, but it actually takes data from "
                    "device {} at input address 0x{:x}."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), ele.startAddr, rangeEnd, dstDeviceId,
                    srcDeviceId, expectedSrcAddr, srcBuf.deviceId, srcBuf.srcAddr, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }
            totalSize += ele.size;
        }
        if (totalSize != dataSize) {
            HCCL_VM_ERROR(
                "{} Send/Recv output for rank {} ends too early. The checker has "
                "validated 0x{:x} bytes in total, but the expected total size is 0x{:x}.",
                MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), dstDeviceId, totalSize, dataSize);
            return HcclResult::HCCL_E_PARA;
        }
    }

    if (!foundSrc || !foundDst) {
        HCCL_VM_ERROR(
            "{} Send/Recv pair references a missing rank (sourceRank={}, targetRank={}).",
            MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_CHECK_FAILED), srcDeviceId, dstDeviceId);
        return HcclResult::HCCL_E_PARA;
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult TaskCheckSendRecvGroupSemantics(
    const PhysicalMemorySemantics& memorySemantics, const DeviceBufferAddressLayouts& bufferLayouts, uint64_t dataSize,
    const std::vector<SendRecvPairParam>& pairs, const std::vector<DeviceId>& rankToDevice)
{
    for (const auto& pair : pairs) {
        const HcclResult ret = TaskCheckSendRecvSemantics(
            memorySemantics, bufferLayouts, dataSize, rankToDevice[pair.srcRank], rankToDevice[pair.dstRank],
            rankToDevice);
        if (ret != HcclResult::HCCL_SUCCESS) {
            return ret;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}
} // namespace HcclSim
