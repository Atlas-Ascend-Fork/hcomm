/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_DFX_PROFILING_HANDLER_H
#define HCOMM_DFX_PROFILING_HANDLER_H

#include "dfx/dfx_profiling_handler_pub.h"

namespace Hccl {
MAKE_ENUM(kernelType, AICPU_KERNEL = 0, CCU_KERNEL);

constexpr uint8_t INVALID_TYPE_VALUE = 0xFF; // reduceOpType、inputDataType、outputDataType非法值

MAKE_ENUM(ProfTaskType, TASK_HCCL_INFO, TASK_DPU_HCCL_INFO);

const std::map<OpType, std::string> PROF_OP_NAME_V2
    = {{OpType::INVALID, "hcom_invalid_"},
       {OpType::ALLREDUCE, "hcom_allReduce_"},
       {OpType::BROADCAST, "hcom_broadcast_"},
       {OpType::REDUCE, "hcom_reduce_"},
       {OpType::SEND, "hcom_send_"},
       {OpType::RECV, "hcom_receive_"},
       {OpType::ALLGATHER, "hcom_allGather_"},
       {OpType::REDUCESCATTER, "hcom_reduceScatter_"},
       {OpType::SCATTER, "hcom_scatter_"},
       {OpType::ALLTOALL, "hcom_alltoall_"},
       {OpType::ALLTOALLV, "hcom_alltoallv_"},
       {OpType::ALLGATHERV, "hcom_allGatherv_"},
       {OpType::REDUCESCATTERV, "hcom_reduceScatterv_"},
       {OpType::ALLTOALLVC, "hcom_alltoallvc_"},
       {OpType::BATCHSENDRECV, "hcom_batchSendRecv_"},
       {OpType::BATCHPUT, "hccl_batchPut_"},
       {OpType::BATCHGET, "hccl_batchGet_"},
       {OpType::DEBUGCASE, "hccl_debugCase_"},
       {OpType::BARRIER, "hccl_barrier_"},
       {OpType::HALFALLTOALLV, "hccl_halfAlltoallv_"},
       {OpType::HCCLGROUPOP, "hccl_groupOp_"}};

inline std::string GetProfOpName(OpType opType)
{
    CHK_PRT_RET(PROF_OP_NAME_V2.empty(), HCCL_ERROR("PROF_OP_NAME_V2 has not inited."), "hcom_invalid_");
    auto it = PROF_OP_NAME_V2.find(opType);
    if (it != PROF_OP_NAME_V2.end()) {
        return it->second;
    }
    return PROF_OP_NAME_V2.begin()->second;
}
} // namespace Hccl

#endif // HCOMM_DFX_PROFILING_HANDLER_H
