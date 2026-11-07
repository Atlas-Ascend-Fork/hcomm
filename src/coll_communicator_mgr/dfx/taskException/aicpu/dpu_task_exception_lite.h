/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DPU_TASK_EXCEPTION_LITE_H
#define DPU_TASK_EXCEPTION_LITE_H

#include "coll_comm_aicpu.h"

namespace hccl {
struct QueueDescriptor;
}

namespace hcomm {

class DpuTaskExceptionLite {
public:
    DpuTaskExceptionLite() = default;
    ~DpuTaskExceptionLite() = default;
    static HcclResult HandleDpuTaskexception(CollCommAicpu* aicpuComm);
    static HcclResult
    HandleDpuCq(CollCommAicpu* aicpuComm, const hccl::QueueDescriptor& cqDesc); // SQ/CQ 队列的 CQ 回执轮询

private:
    static HcclResult IsHandleDpuStop(uint8_t* taskexceptionVa, bool& isStop);
    static HcclResult ReportDpuCqError(CollCommAicpu* aicpuComm); // 取 notifyId 投 MBox，成功后置「已上报」
};

} // namespace hcomm

#endif
