/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_GROUP_QUEUE_MGR_H
#define OP_GROUP_QUEUE_MGR_H

#include <deque>
#include <vector>
#include <hccl/hccl_types.h>
#include <hccl/hccl_launch.h>
#include "acl/acl_base_rt.h"

constexpr int32_t MAX_GROUP_OP_TASK_NUM = 2048;

namespace hccl {

// group内入队的send/recv算子任务缓存
struct GroupOpItem {
    HcclOpDesc desc{};
    aclrtStream stream{nullptr};
    void* func{nullptr};
};

class OpGroupQueueMgr {
public:
    OpGroupQueueMgr() = default;
    ~OpGroupQueueMgr();

    // group内入队任务的通信域注册表，与AICPU路径的hcclGroupCommListV2相互独立
    static std::vector<HcclComm>& GetGroupQueueCommList();
    static void ClearGroupQueueCommList();
    // group内经HcclGroupEnqueueOp入队的p2p算子任务总数（跨comm累计）
    static int32_t GetGroupQueueTaskNums();
    static void SetGroupQueueTaskNums(int32_t targetTaskNums);

    HcclResult AppendGroupOpTask(HcclComm comm, const HcclOpDesc& opDesc, aclrtStream stream, void* func);
    // 批量执行传入comm内经HcclGroupEnqueueOp入队的任务
    HcclResult LaunchGroupOpTask(HcclComm comm);
    void ClearGroupOpTask();

private:
    // group内入队任务的通信域注册表，与AICPU路径的hcclGroupCommListV2相互独立
    inline static thread_local std::vector<HcclComm> hcclGroupQueueCommListV2_;
    // group内经HcclGroupEnqueueOp入队的p2p算子任务总数（跨comm累计）
    inline static thread_local int32_t hcclP2pTaskNums_ = 0;
    std::deque<GroupOpItem> groupOpItems_;
};

} // namespace hccl

#endif // OP_GROUP_QUEUE_MGR_H
