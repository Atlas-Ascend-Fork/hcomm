/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_task_collection.h"

#include <atomic>
#include <cstdint>

#include "sim_log.h"
#include "operation_data/operation_data_ops.h"

int32_t InsertTaskToCollectionDev(const HcclTaskMetaData* task)
{
    // 1. 入参合法性检查（避免空指针访问）
    if (task == nullptr) {
        HCCL_VM_ERROR("错误：输入任务指针 task 不能为空！");
        return -1;
    }

    // 2. 填写任务表记录的关联信息和全局采集序号
    sim::operation::OpTaskTab opTaskInfo;
    opTaskInfo.id = 0; // 数据库自增，无需设置
    opTaskInfo.deviceId = task->deviceId;
    opTaskInfo.streamId = task->streamId;
    opTaskInfo.taskType = static_cast<uint32_t>(task->taskType);

    // 3. 将HcclTaskMetaData保存为任务表blob
    opTaskInfo.optaskMeta.assign(
        reinterpret_cast<const uint8_t*>(task), reinterpret_cast<const uint8_t*>(task) + sizeof(HcclTaskMetaData));

    // 4. 将设备侧任务写入当前进程对应的任务表
    auto ret = sim::operation::InsertOpTask(opTaskInfo, true);
    if (ret != 0) {
        HCCL_VM_ERROR("错误：插入任务到数据库失败 - {}", ret);
        return ret;
    }
    return 0;
}
