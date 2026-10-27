/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_DFX_RES_PUB_H
#define HCOMM_DFX_RES_PUB_H
#include <map>
#include <string>
#include "hccl/hccl_types.h"
#include <cstdint>
#include "string_util.h"
constexpr uint32_t INVALID_U32 = UINT32_MAX;
constexpr uint64_t DFX_INVALID_U64 = UINT64_MAX;
constexpr int32_t INVALID_RANKID = INT32_MAX;

// 以下枚举来源于 legacy/ascend950，独立定义使本头脱离 legacy 依赖；取值与 legacy MAKE_ENUM 对齐，
// 其中 TASK_CCORE_NOTIFY_WAIT/RECORD(24/25) 为 base_comm 侧扩展，OP_TYPE_INVALID 为 base_comm 侧非法哨兵。
namespace Hccl {

enum OpTypeVal : uint8_t {
    OP_TYPE_ALLREDUCE = 0,
    OP_TYPE_BROADCAST = 1,
    OP_TYPE_ALLGATHER = 2,
    OP_TYPE_REDUCESCATTER = 3,
    OP_TYPE_SEND = 4,
    OP_TYPE_RECV = 5,
    OP_TYPE_BARRIER = 6,
    OP_TYPE_ALLTOALL = 7,
    OP_TYPE_REDUCE = 8,
    OP_TYPE_GATHER = 9,
    OP_TYPE_SCATTER = 10,
    OP_TYPE_ALLTOALLV = 11,
    OP_TYPE_ALLTOALLVC = 12,
    OP_TYPE_HALFALLTOALLV = 13,
    OP_TYPE_BATCHSENDRECV = 14,
    OP_TYPE_BATCHGET = 15,
    OP_TYPE_BATCHPUT = 16,
    OP_TYPE_ALLGATHERV = 17,
    OP_TYPE_REDUCESCATTERV = 18,
    OP_TYPE_DEBUGCASE = 19,
    OP_TYPE_INVALID = 20,
    OP_TYPE_COUNT
};

enum AlgTypeVal : uint8_t {
    ALG_TYPE_NOT_SPECIFIED = 0,
    ALG_TYPE_RING = 1,
    ALG_TYPE_MULTI_RING = 2,
    ALG_TYPE_MESH = 3,
    ALG_TYPE_RECURSIVE_HD = 4,
    ALG_TYPE_BINARY_HD = 5,
    ALG_TYPE_PAIR_WISE = 6,
    ALG_TYPE_INVALID_VAL = 7,
    ALG_TYPE_COUNT
};

enum TaskParamTypeVal : uint8_t {
    TASK_SDMA = 0,
    TASK_RDMA = 1,
    TASK_REDUCE_INLINE = 2,
    TASK_REDUCE_TBE = 3,
    TASK_NOTIFY_RECORD = 4,
    TASK_NOTIFY_WAIT = 5,
    TASK_SEND_NOTIFY = 6,
    TASK_SEND_PAYLOAD = 7,
    TASK_WRITE_WITH_NOTIFY = 8,
    TASK_WRITE_REDUCE_WITH_NOTIFY = 9,
    TASK_CCU = 10,
    TASK_AICPU_KERNEL = 11,
    TASK_AICPU_REDUCE = 12,
    TASK_AIV = 13,
    TASK_UB_INLINE_WRITE = 14,
    TASK_UB_REDUCE_INLINE = 15,
    TASK_UB = 16,
    TASK_DPU_KERNEL = 17,
    TASK_DPU_THREAD_FENCE = 18,
    TASK_DPU_CHANNEL_FENCE = 19,
    TASK_DPU_INLINE_WRITE = 20,
    TASK_DPU_NOTIFY_WAIT = 21,
    TASK_DPU_WRITE_WITH_NOTIFY = 22,
    TASK_DPU_CHANNEL_DRAIN = 23,
    TASK_CCORE_NOTIFY_WAIT = 24,
    TASK_CCORE_NOTIFY_RECORD = 25,
    TASK_PARAM_TYPE_COUNT
};

enum DfxLinkTypeVal : uint8_t {
    LINK_ONCHIP = 0,
    LINK_HCCS = 1,
    LINK_PCIE = 2,
    LINK_ROCE = 3,
    LINK_SIO = 4,
    LINK_HCCS_SW = 5,
    LINK_STANDARD_ROCE = 6,
    LINK_UB = 7,
    LINK_UBoE = 8,
    LINK_RESERVED = 9,
    LINK_TYPE_COUNT
};

enum DfxWorkflowMode : uint8_t {
    NEW_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB = 0,
    NEW_WORKFLOW_MODE_OP_BASE = 1,
    NEW_WORKFLOW_MODE_RESERVED = 2
};

enum DfxTransportType : uint8_t {
    DFX_TRANSPORT_TYPE_SDMA = 0,
    DFX_TRANSPORT_TYPE_RDMA = 1,
    DFX_TRANSPORT_TYPE_LOCAL = 2,
    DFX_TRANSPORT_TYPE_UB = 3,
    DFX_TRANSPORT_TYPE_ROCE = 4
};

constexpr uint32_t DFX_INVALID_RANKID = 0xFFFFFFFF;

enum DfxTaskRole : uint8_t { NEW_TASK_ROLE_DST = 0, NEW_TASK_ROLE_SRC = 1 };

inline std::string TaskParamTypeValToStr(uint8_t val)
{
    static const std::map<TaskParamTypeVal, std::string> taskParamTypeValStrMap = {
        {TaskParamTypeVal::TASK_SDMA, "TASK_SDMA"},
        {TaskParamTypeVal::TASK_RDMA, "TASK_RDMA"},
        {TaskParamTypeVal::TASK_REDUCE_INLINE, "TASK_REDUCE_INLINE"},
        {TaskParamTypeVal::TASK_REDUCE_TBE, "TASK_REDUCE_TBE"},
        {TaskParamTypeVal::TASK_NOTIFY_RECORD, "TASK_NOTIFY_RECORD"},
        {TaskParamTypeVal::TASK_NOTIFY_WAIT, "TASK_NOTIFY_WAIT"},
        {TaskParamTypeVal::TASK_SEND_NOTIFY, "TASK_SEND_NOTIFY"},
        {TaskParamTypeVal::TASK_SEND_PAYLOAD, "TASK_SEND_PAYLOAD"},
        {TaskParamTypeVal::TASK_WRITE_WITH_NOTIFY, "TASK_WRITE_WITH_NOTIFY"},
        {TaskParamTypeVal::TASK_WRITE_REDUCE_WITH_NOTIFY, "TASK_WRITE_REDUCE_WITH_NOTIFY"},
        {TaskParamTypeVal::TASK_CCU, "TASK_CCU"},
        {TaskParamTypeVal::TASK_AICPU_KERNEL, "TASK_AICPU_KERNEL"},
        {TaskParamTypeVal::TASK_AICPU_REDUCE, "TASK_AICPU_REDUCE"},
        {TaskParamTypeVal::TASK_AIV, "TASK_AIV"},
        {TaskParamTypeVal::TASK_UB_INLINE_WRITE, "TASK_UB_INLINE_WRITE"},
        {TaskParamTypeVal::TASK_UB_REDUCE_INLINE, "TASK_UB_REDUCE_INLINE"},
        {TaskParamTypeVal::TASK_UB, "TASK_UB"},
        {TaskParamTypeVal::TASK_DPU_KERNEL, "TASK_DPU_KERNEL"},
        {TaskParamTypeVal::TASK_DPU_THREAD_FENCE, "TASK_DPU_THREAD_FENCE"},
        {TaskParamTypeVal::TASK_DPU_CHANNEL_FENCE, "TASK_DPU_CHANNEL_FENCE"},
        {TaskParamTypeVal::TASK_DPU_INLINE_WRITE, "TASK_DPU_INLINE_WRITE"},
        {TaskParamTypeVal::TASK_DPU_NOTIFY_WAIT, "TASK_DPU_NOTIFY_WAIT"},
        {TaskParamTypeVal::TASK_DPU_WRITE_WITH_NOTIFY, "TASK_DPU_WRITE_WITH_NOTIFY"},
        {TaskParamTypeVal::TASK_DPU_CHANNEL_DRAIN, "TASK_DPU_CHANNEL_DRAIN"},
        {TaskParamTypeVal::TASK_CCORE_NOTIFY_WAIT, "TASK_CCORE_NOTIFY_WAIT"},
        {TaskParamTypeVal::TASK_CCORE_NOTIFY_RECORD, "TASK_CCORE_NOTIFY_RECORD"},
    };
    auto iter = taskParamTypeValStrMap.find(static_cast<TaskParamTypeVal>(val));
    if (iter == taskParamTypeValStrMap.end()) {
        return "TaskParamTypeVal(" + std::to_string(val) + ")";
    }
    return iter->second;
}

inline std::string DfxLinkTypeValToStr(uint8_t val)
{
    static const std::map<DfxLinkTypeVal, std::string> dfxLinkTypeValStrMap = {
        {DfxLinkTypeVal::LINK_ONCHIP, "LINK_ONCHIP"},
        {DfxLinkTypeVal::LINK_HCCS, "LINK_HCCS"},
        {DfxLinkTypeVal::LINK_PCIE, "LINK_PCIE"},
        {DfxLinkTypeVal::LINK_ROCE, "LINK_ROCE"},
        {DfxLinkTypeVal::LINK_SIO, "LINK_SIO"},
        {DfxLinkTypeVal::LINK_HCCS_SW, "LINK_HCCS_SW"},
        {DfxLinkTypeVal::LINK_STANDARD_ROCE, "LINK_STANDARD_ROCE"},
        {DfxLinkTypeVal::LINK_UB, "LINK_UB"},
        {DfxLinkTypeVal::LINK_UBoE, "LINK_UBoE"},
        {DfxLinkTypeVal::LINK_RESERVED, "LINK_RESERVED"},
    };
    auto iter = dfxLinkTypeValStrMap.find(static_cast<DfxLinkTypeVal>(val));
    if (iter == dfxLinkTypeValStrMap.end()) {
        return "DfxLinkTypeVal(" + std::to_string(val) + ")";
    }
    return iter->second;
}

inline std::string DfxTransportTypeToStr(uint8_t val)
{
    static const std::map<DfxTransportType, std::string> dfxTransportTypeStrMap = {
        {DfxTransportType::DFX_TRANSPORT_TYPE_SDMA, "DFX_TRANSPORT_TYPE_SDMA"},
        {DfxTransportType::DFX_TRANSPORT_TYPE_RDMA, "DFX_TRANSPORT_TYPE_RDMA"},
        {DfxTransportType::DFX_TRANSPORT_TYPE_LOCAL, "DFX_TRANSPORT_TYPE_LOCAL"},
        {DfxTransportType::DFX_TRANSPORT_TYPE_UB, "DFX_TRANSPORT_TYPE_UB"},
        {DfxTransportType::DFX_TRANSPORT_TYPE_ROCE, "DFX_TRANSPORT_TYPE_ROCE"},
    };
    auto iter = dfxTransportTypeStrMap.find(static_cast<DfxTransportType>(val));
    if (iter == dfxTransportTypeStrMap.end()) {
        return "DfxTransportType(" + std::to_string(val) + ")";
    }
    return iter->second;
}

} // namespace Hccl

#endif // HCOMM_DFX_RES_PUB_H
