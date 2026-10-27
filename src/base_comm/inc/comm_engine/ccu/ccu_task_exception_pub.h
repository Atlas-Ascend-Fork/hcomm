/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// CCU task exception public interface. The decoder lives in base_comm and must not
// depend on the L2 coll_comm types, so all communication-domain queries are inverted
// through callbacks registered by the L2 layer. `comm` is the opaque communicator
// handle carried by Hccl::TaskInfo::dfxOpInfo_->comm_.
#ifndef HCOMM_CCU_TASK_EXCEPTION_PUB_H
#define HCOMM_CCU_TASK_EXCEPTION_PUB_H

#include <string>
#include <vector>
#include "hccl_types.h"
#include "hccl/base.h"
#include "rt_external.h"
#include "task_info.h"

namespace hcomm {

// Callbacks provided by the L2 layer to feed CCU exception decoding with
// communication-domain data.
// 约定：回调失败时静默返回、不修改任何 out 参数，调用方必须先为 out 参数置初值；
// 失败与合法空值不可区分，均为调用方初值。
struct CcuTaskExceptionCallbacks {
    // Fill commId/udi/rankSize/rankId of the given communicator.
    void (*getGroupRankInfo)(void* comm, std::string& commId, std::string& udi, u32& rankSize, u32& rankId);
    // Resolve the remote local id and net instance id of the given rank.
    void (*getCcuCqeErrInfoByRankId)(void* comm, u32 rankId, u32& remoteLocalId, std::string& netInstanceId);
    // Resolve the remote rank id of the given channel handle.
    HcclResult (*getChannelRemoteRankId)(void* comm, u64 channelHandle, u32& remoteRank);
};

void RegisterCcuTaskExceptionCallbacks(const CcuTaskExceptionCallbacks& callbacks);

using GetCcuCqeErrInfoCallBackHcomm = void (*)(
    u32 RemoteLocalId, u32 locDeviceId, uint16_t status, std::string LocalEid, std::string RemoteEid,
    std::string RemoteInsId);
void RegisterGetCcuCqeErrInfoCallBackHcomm(GetCcuCqeErrInfoCallBackHcomm callback);

using CcuGetErrStatusVecCallBack = std::vector<std::string> (*)(s32 deviceLogicID);
void RegisterCcuGetErrStatusVecCallBack(CcuGetErrStatusVecCallBack callback);

// Entry called by the L2 layer when a CCU task raises an exception.
void ProcessCcuTaskException(const rtExceptionInfo_t* exceptionInfo, const Hccl::TaskInfo& taskInfo);

} // namespace hcomm

#endif // HCOMM_CCU_TASK_EXCEPTION_PUB_H
