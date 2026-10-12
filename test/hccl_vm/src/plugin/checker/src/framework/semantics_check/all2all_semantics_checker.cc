/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * Description: All2All语义校验函数实现
 * Author: yinding
 * Create: 2024-09-30
 */

#include "all2all_semantics_checker.h"

#include <map>
#include <vector>

#include "checker_def.h"
#include "sim_log.h"
#include "utils/dump/dump_json_utils.h"
#include "utils/error_codes.h"

namespace HcclSim {
// 计算 targetRankIdx 接收每个 srcRank 数据时各 srcRank 在其 INPUT 中的起始偏移。
// 前提：targetRankIdx 必须是有效 rank 下标（调用方需先校验），否则矩阵索引越界。
void GenSendOffsetForOneRank(
    const All2AllDataDesTagInner& all2AllDataDes, uint32_t rankSize, uint32_t targetRankIdx,
    std::vector<uint64_t>& sendOffsets)
{
    for (uint32_t srcRankIdx = 0; srcRankIdx < rankSize; srcRankIdx++) {
        uint64_t offset = 0;
        for (uint32_t dstRankIdx = 0; dstRankIdx < targetRankIdx; dstRankIdx++) {
            uint64_t count = all2AllDataDes.sendCountMatrix[srcRankIdx * rankSize + dstRankIdx];
            uint64_t size = count * CHECK_SIZE_TABLE[all2AllDataDes.recvType];
            offset += size;
        }
        sendOffsets.push_back(offset);
    }
    return;
}

HcclResult TaskCheckAll2AllSemantics(
    const PhysicalMemorySemantics& memorySemantics, const DeviceBufferAddressLayouts& bufferLayouts,
    const All2AllDataDesTagInner& all2AllDataDes, const std::vector<DeviceId>& rankToDevice)
{
    const uint32_t rankSize = memorySemantics.size();

    for (const auto& [deviceId, mem] : memorySemantics) {
        const RankId deviceRank = FindRankIndexByDeviceId(rankToDevice, deviceId);
        if (deviceRank == TaskGraphGeneratorV3::INVALID_RANK_ID) {
            // rank 映射与参与设备集不一致属于校验环境异常，继续推转会按错误的 rank 语义校验
            // （sendCountMatrix/counts 按 rank 下标索引，非法下标会越界）。
            HCCL_VM_ERROR(
                "{} Device {} is not found in the rank-to-device mapping, cannot resolve "
                "its rank index for the semantic check.",
                MakeErrorCodeText(ErrorCode::CHECKER_RUNTIME_ERROR), deviceId);
            return HcclResult::HCCL_E_PARA;
        }
        uint64_t outputBase = 0;
        uint64_t outputBufferSize = 0;
        GetSemanticBufferExtent(bufferLayouts, deviceId, BufferType::OUTPUT, outputBase, outputBufferSize);
        std::vector<uint64_t> sendOffsets;
        GenSendOffsetForOneRank(all2AllDataDes, rankSize, deviceRank, sendOffsets);
        uint64_t outputSize = 0;
        for (uint32_t srcRank = 0; srcRank < rankSize; ++srcRank) {
            outputSize += all2AllDataDes.sendCountMatrix[srcRank * rankSize + deviceRank]
                          * CHECK_SIZE_TABLE[all2AllDataDes.recvType];
        }

        uint64_t totalSize = 0;
        uint32_t curRankIdx = 0;
        uint64_t curDataSize = 0;

        // curRankIdx向当前device发送的数据量是0 直接跳过
        while (curRankIdx < rankSize && all2AllDataDes.sendCountMatrix[curRankIdx * rankSize + deviceRank] == 0) {
            curRankIdx++;
        }

        for (const auto& [eleAddr, original] : mem) {
            BufferSemantic ele = original;
            if (!ClipSemanticToBuffer(ele, outputBase, outputBufferSize)) {
                continue;
            }
            if (ele.startAddr >= outputBase + outputSize) {
                // 已裁剪到注册缓冲区；超出期望结果大小的非空语义段仍应报错。
                // 注册大小未知时保留原有兼容行为。
                if (outputBufferSize != 0 && ele.size != 0) {
                    HCCL_VM_ERROR(
                        "{} AllToAll output for device {} contains unexpected extra data: range "
                        "[0x{:x},0x{:x}) starts beyond the expected result size 0x{:x} but inside the "
                        "output buffer (size 0x{:x}).\nCurrent result range detail:\n{}",
                        MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SIZE_ERROR), deviceId, ele.startAddr,
                        ele.startAddr + ele.size, outputSize, outputBufferSize, ele.Describe());
                    return HcclResult::HCCL_E_PARA;
                }
                break;
            }
            const uint64_t rangeEnd = ele.startAddr + ele.size;
            if (ele.startAddr != outputBase + totalSize) {
                HCCL_VM_ERROR(
                    "{} AllToAll output for device {} should continue at 0x{:x}, "
                    "but the next actual range starts at 0x{:x} (actual range: [0x{:x},0x{:x}))."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), deviceId, outputBase + totalSize,
                    ele.startAddr, ele.startAddr, rangeEnd, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (ele.srcBufs.size() != 1) {
                HCCL_VM_ERROR(
                    "{} AllToAll output range [0x{:x},0x{:x}) for device {} should "
                    "come from exactly one source, but it actually comes from {} sources."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_REDUCE_ERROR), ele.startAddr, rangeEnd, deviceId,
                    ele.srcBufs.size(), ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (curRankIdx >= rankSize) {
                HCCL_VM_ERROR(
                    "{} AllToAll expected output for device {} has already ended, "
                    "but outputRange [0x{:x},0x{:x}) is still present.\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::CHECKER_RUNTIME_ERROR), deviceId, ele.startAddr, rangeEnd,
                    ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            const auto& srcBuf = *ele.srcBufs.begin();
            const uint64_t expectedSrcAddr
                = GetSemanticBufferBase(bufferLayouts, rankToDevice[curRankIdx], BufferType::INPUT)
                  + sendOffsets[curRankIdx] + curDataSize;
            const uint64_t expectedSrcEnd = expectedSrcAddr + ele.size;
            const uint64_t actualSrcAddr = srcBuf.srcAddr;
            const uint64_t actualSrcEnd = actualSrcAddr + ele.size;

            if (srcBuf.deviceId != rankToDevice[curRankIdx]) {
                HCCL_VM_ERROR(
                    "{} AllToAll output range [0x{:x},0x{:x}) should come from "
                    "rank{}.INPUT[0x{:x},0x{:x}), but it actually comes from device{}.{}[0x{:x},0x{:x})."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), ele.startAddr, rangeEnd, curRankIdx,
                    expectedSrcAddr, expectedSrcEnd, srcBuf.deviceId, BufferTypeToString(srcBuf.bufType), actualSrcAddr,
                    actualSrcEnd, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (srcBuf.bufType != BufferType::INPUT) {
                HCCL_VM_ERROR(
                    "{} AllToAll output range [0x{:x},0x{:x}) should come from "
                    "rank{}.INPUT[0x{:x},0x{:x}), but it actually comes from device{}.{}[0x{:x},0x{:x})."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), ele.startAddr, rangeEnd, curRankIdx,
                    expectedSrcAddr, expectedSrcEnd, srcBuf.deviceId, BufferTypeToString(srcBuf.bufType), actualSrcAddr,
                    actualSrcEnd, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            if (srcBuf.srcAddr != expectedSrcAddr) {
                HCCL_VM_ERROR(
                    "{} AllToAll output range [0x{:x},0x{:x}) should come from "
                    "rank{}.INPUT[0x{:x},0x{:x}), but it actually comes from device{}.INPUT[0x{:x},0x{:x})."
                    "\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SRC_ERROR), ele.startAddr, rangeEnd, curRankIdx,
                    expectedSrcAddr, expectedSrcEnd, srcBuf.deviceId, actualSrcAddr, actualSrcEnd, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }

            curDataSize += ele.size;
            uint64_t recvCountFromCurRank = all2AllDataDes.sendCountMatrix[curRankIdx * rankSize + deviceRank];
            uint64_t recvDataSizeFromCurRank = recvCountFromCurRank * CHECK_SIZE_TABLE[all2AllDataDes.recvType];
            if (curDataSize == recvDataSizeFromCurRank) {
                curDataSize = 0;
                curRankIdx++;
                while (curRankIdx < rankSize
                       && all2AllDataDes.sendCountMatrix[curRankIdx * rankSize + deviceRank] == 0) {
                    // 跳过后续所有发送量为0的rank
                    curRankIdx++;
                }
            } else if (curDataSize > recvDataSizeFromCurRank) {
                HCCL_VM_ERROR(
                    "{} AllToAll data collected from rank{}.INPUT for device {} becomes larger "
                    "than expected after outputRange [0x{:x},0x{:x}). The accumulated size is 0x{:x}, "
                    "but the expected size from this source rank is 0x{:x}.\nCurrent result range detail:\n{}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_SIZE_ERROR), curRankIdx, deviceId, ele.startAddr,
                    rangeEnd, curDataSize, recvDataSizeFromCurRank, ele.Describe());
                return HcclResult::HCCL_E_PARA;
            }
            totalSize += ele.size;
        }
        // 如果curRankIdx等于rankSize，表示已经接受到其他所有rank的数据
        if (curRankIdx != rankSize) {
            const uint64_t expectedDataSizeFromCurRank
                = all2AllDataDes.sendCountMatrix[curRankIdx * rankSize + deviceRank]
                  * CHECK_SIZE_TABLE[all2AllDataDes.recvType];
            const uint64_t missingSizeFromCurRank = expectedDataSizeFromCurRank - curDataSize;
            HCCL_VM_ERROR(
                "{} AllToAll output for device {} ends too early. The checker has "
                "validated 0x{:x} bytes in total, but data from rank{} is still incomplete: "
                "0x{:x} bytes are missing (received 0x{:x}, expected 0x{:x}).",
                MakeErrorCodeText(ErrorCode::SEMANTIC_FINAL_MISSING), deviceId, totalSize, curRankIdx,
                missingSizeFromCurRank, curDataSize, expectedDataSizeFromCurRank);
            return HcclResult::HCCL_E_PARA;
        }
    }

    return HcclResult::HCCL_SUCCESS;
}
} // namespace HcclSim
