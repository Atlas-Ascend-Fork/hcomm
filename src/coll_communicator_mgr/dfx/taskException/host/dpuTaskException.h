/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DPU_TASKEXCEPTION_H
#define DPU_TASKEXCEPTION_H

#include "global_mirror_tasks.h"
#include "orion_adapter_rts.h"
#include "coll_comm.h"

namespace hcomm {

class TaskExceptionHost;

class DpuTaskException {
public:
    DpuTaskException() = default;
    ~DpuTaskException() = default;
    static HcclResult ProcessDpuException(const rtExceptionInfo_t* exceptionInfo, const TaskExceptionHost& host);

private:
    static HcclResult GetDpuExceptionInfo(
        const rtExceptionInfo_t* exceptionInfo, uint16_t& dpuErrCode, std::string& commId, uint8_t*& taskExpPtr);
    static HcclResult ReportDpuException(
        const rtExceptionInfo_t* exceptionInfo, uint16_t dpuErrCode, const std::string& commId, uint8_t* taskExpPtr,
        const TaskExceptionHost& host);
    static HcclResult ClearDpuExceptionInfo(const rtExceptionInfo_t* exceptionInfo, const std::string& commId);
    static void HandleDpuErrorReport(
        const rtExceptionInfo_t* exceptionInfo, const Hccl::TaskInfo& taskInfo, uint16_t hcclRet,
        const std::string& commId, const TaskExceptionHost& host);
    static void
    PrintDpuTaskContextInfo(uint32_t deviceId, const Hccl::TaskInfo& taskInfo, const std::string& stageErrInfo);
    static std::string GetGroupRankInfo(const Hccl::TaskInfo& taskInfo);
    static u32 GetOpIndex(const Hccl::TaskInfo* taskInfo);
    static u32 StoredToReportStreamId(const std::string& commId, u32 stored);
    static std::string GetDpuIndopDataInfo(const Hccl::TaskInfo* taskInfo);
};
} // namespace hcomm

#endif
