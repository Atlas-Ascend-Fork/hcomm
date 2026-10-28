/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_NOTIFY_MGR_PROC_H
#define HCOMM_NOTIFY_MGR_PROC_H

#include "hccl/hccl_types.h"
#include "resources/endpoint_pairs/channels/host/dpu_notify/dpu_notify_manager.h"

namespace hcomm {

/**
 * @brief Notify 管理节点（进程级），值成员持有 DpuNotifyManager（进程级语义：ID 空间全进程共享）
 */
class NotifyMgrProc {
public:
    void DeInit() {}

    static constexpr int DPU_NOTIFY_ENTRIES = 8192;

    DpuNotifyManager& GetDpuNotifyManager() { return dpuNotifyManager_; }

public:
    ~NotifyMgrProc() = default;

public:
    NotifyMgrProc() = default;

private:
    NotifyMgrProc(const NotifyMgrProc&) = delete;
    NotifyMgrProc& operator=(const NotifyMgrProc&) = delete;

    DpuNotifyManager dpuNotifyManager_{DPU_NOTIFY_ENTRIES};
};

} // namespace hcomm

#endif // HCOMM_NOTIFY_MGR_PROC_H
