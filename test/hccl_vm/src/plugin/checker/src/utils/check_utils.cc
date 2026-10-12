/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "check_utils.h"

#include <algorithm>
#include <cstdint>

#include "sim_log.h"

namespace HcclSim {
std::set<SrcBufDes> ShiftSourceAddresses(const std::set<SrcBufDes>& srcBufs, uint64_t addressDelta)
{
    if (addressDelta == 0 || srcBufs.empty()) {
        return srcBufs;
    }

    std::set<SrcBufDes> shiftedSrcBufs;
    for (const auto& srcBuf : srcBufs) {
        shiftedSrcBufs.emplace(srcBuf.deviceId, srcBuf.bufType, srcBuf.srcAddr + addressDelta);
    }
    return shiftedSrcBufs;
}

// 获取原语的类型
bool IsAllToAllSeries(HcclCMDType opType)
{
    return (
        opType == HcclCMDType::HCCL_CMD_ALLTOALL || opType == HcclCMDType::HCCL_CMD_ALLTOALLV
        || opType == HcclCMDType::HCCL_CMD_ALLTOALLVC);
}

bool ClipSemanticToBuffer(BufferSemantic& semantic, uint64_t baseAddr, uint64_t bufferSize)
{
    if (semantic.startAddr < baseAddr) {
        const uint64_t delta = baseAddr - semantic.startAddr;
        if (delta >= semantic.size) {
            return false;
        }
        semantic.startAddr = baseAddr;
        semantic.size -= delta;
        semantic.srcBufs = ShiftSourceAddresses(semantic.srcBufs, delta);
    }
    if (bufferSize != 0) {
        const uint64_t offset = semantic.startAddr - baseAddr;
        if (offset >= bufferSize) {
            return false;
        }
        // 用减法计算剩余范围，避免基址加注册大小发生溢出。
        semantic.size = std::min(semantic.size, bufferSize - offset);
    }
    return true;
}

bool IsSendRecvType(HcclCMDType opType)
{
    return opType == HcclCMDType::HCCL_CMD_SEND || opType == HcclCMDType::HCCL_CMD_RECEIVE;
}

void CalcInputOutputSize(
    HcclCMDType opType, uint32_t rankSize, uint64_t count, HcclDataType dataType, uint64_t& inputSize,
    uint64_t& outputSize, RankId myRank, RankId srcRank, RankId dstRank, VDataDesTagInner vDataDes,
    All2AllDataDesTagInner all2AllDataDes)
{
    uint32_t unitSize = 0;
    if (!IsAllToAllSeries(opType) && opType != HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V
        && opType != HcclCMDType::HCCL_CMD_ALLGATHER_V) {
        unitSize = CHECK_SIZE_TABLE[dataType];
    }

    if (opType == HcclCMDType::HCCL_CMD_ALLREDUCE) {
        inputSize = count * unitSize;
        outputSize = count * unitSize;
    } else if (opType == HcclCMDType::HCCL_CMD_BROADCAST) {
        inputSize = count * unitSize;
        outputSize = count * unitSize;
    } else if (IsSendRecvType(opType) && myRank == srcRank) {
        inputSize = count * unitSize;
        outputSize = 0;
    } else if (IsSendRecvType(opType) && myRank == dstRank) {
        inputSize = 0;
        outputSize = count * unitSize;
    } else if (opType == HcclCMDType::HCCL_CMD_REDUCE) {
        outputSize = count * unitSize;
        inputSize = count * unitSize;
    } else if (opType == HcclCMDType::HCCL_CMD_ALLGATHER) {
        inputSize = count * unitSize;
        outputSize = count * unitSize * rankSize;
    } else if (opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) {
        inputSize = count * unitSize * rankSize;
        outputSize = count * unitSize;
    } else if (opType == HcclCMDType::HCCL_CMD_SCATTER) {
        inputSize = count * unitSize * rankSize;
        outputSize = count * unitSize;
    } else if (opType == HcclCMDType::HCCL_CMD_ALLTOALL || opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        uint64_t curSendOffset = 0;
        uint64_t curRecvOffset = 0;
        for (uint32_t j = 0; j < rankSize; j++) {
            uint64_t curSendCounts = all2AllDataDes.sendCountMatrix[myRank * rankSize + j];
            uint64_t curSendLength = curSendCounts * CHECK_SIZE_TABLE[all2AllDataDes.sendType];
            curSendOffset += curSendLength;
            uint64_t curRecvCounts = all2AllDataDes.sendCountMatrix[myRank + rankSize * j];
            uint64_t curRecvLength = curRecvCounts * CHECK_SIZE_TABLE[all2AllDataDes.recvType];
            curRecvOffset += curRecvLength;
        }
        inputSize = curSendOffset;
        outputSize = curRecvOffset;
    } else if (opType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
        uint64_t curSendOffset = 0;
        uint64_t curRecvOffset = 0;
        for (uint32_t j = 0; j < rankSize; j++) {
            uint64_t curSendCounts = all2AllDataDes.sendCountMatrix[myRank * rankSize + j];
            uint64_t curSendLength = curSendCounts * CHECK_SIZE_TABLE[all2AllDataDes.sendType];
            curSendOffset += curSendLength;
            uint64_t curRecvCounts = all2AllDataDes.sendCountMatrix[myRank + rankSize * j];
            uint64_t curRecvLength = curRecvCounts * CHECK_SIZE_TABLE[all2AllDataDes.recvType];
            curRecvOffset += curRecvLength;
        }
        inputSize = curSendOffset;
        outputSize = curRecvOffset;
    } else if (opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
        inputSize = count * unitSize;
        outputSize = count * unitSize;
    } else if (opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V) {
        inputSize = 0;
        for (uint32_t i = 0; i < rankSize; i++) {
            uint64_t curCounts = vDataDes.counts[i];
            uint64_t curLength = curCounts * CHECK_SIZE_TABLE[vDataDes.dataType];
            inputSize += curLength;
        }
        outputSize = vDataDes.counts[myRank] * CHECK_SIZE_TABLE[vDataDes.dataType];
    } else if (opType == HcclCMDType::HCCL_CMD_ALLGATHER_V) {
        outputSize = 0;
        for (uint32_t i = 0; i < rankSize; i++) {
            uint64_t curCounts = vDataDes.counts[i];
            uint64_t curLength = curCounts * CHECK_SIZE_TABLE[vDataDes.dataType];
            outputSize += curLength;
        }
        inputSize = vDataDes.counts[myRank] * CHECK_SIZE_TABLE[vDataDes.dataType];
    } else {
        HCCL_VM_WARN("CalcInputOutputSize not support");
    }
}

// 如果输入、输出的count的大小不一样的话，那么opParam中的count是指较小的那个值
// 比如对于AllGather算子，count指输入；对于ReduceScatter算子，count指输入
// 如果输入、输出的count大小一样的话，那么opParam中的count既可以指代输入，也可以指代输出
void CalcDataSize(HcclCMDType opType, uint64_t count, HcclDataType dataType, uint64_t& dataSize)
{
    // 当前AllToAll系列以及不等长算子不使用dataSize，如果后续使用的话，需要适配这个地方
    if (!IsAllToAllSeries(opType) && opType != HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V
        && opType != HcclCMDType::HCCL_CMD_ALLGATHER_V) {
        uint32_t unitSize = CHECK_SIZE_TABLE[dataType];
        dataSize = count * unitSize;
    }
}

} // namespace HcclSim
