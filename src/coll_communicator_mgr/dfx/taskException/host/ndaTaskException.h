/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NDA_TASKEXCEPTION_H
#define NDA_TASKEXCEPTION_H

#include <string>
#include "orion_adapter_rts.h"
#include "global_mirror_tasks.h"
#include "error_message_v2.h"

namespace hcomm {

class TaskExceptionHost;

/**
 * @note 职责：NDA（RoCE）链路 taskException 的独立处理类。
 * 仿照 NDR 的 DpuTaskException 设计：在 TaskExceptionHost::Process 中作为第三分支插入，
 * 入口处按 linkType 一次性分流（LINK_ROCE / LINK_STANDARD_ROCE），命中则走 RoCE 专用路，
 * 否则返回 HCCL_E_NOT_FOUND 交回 ProcessHcclTaskException 兜底。
 * 这样把原先散落在 7 处的 linkType 判断收敛到入口 1 处。
 */
class NdaTaskException {
public:
    NdaTaskException() = default;
    ~NdaTaskException() = default;

    /**
     * NDA 异常处理入口：遍历通信域快照 → 读 KFC D2H → linkType 分流 → 找任务 → 分派填字段 → 打印上报。
     * @param exceptionInfo RTS 回调传入的异常上下文
     * @param host TaskExceptionHost 实例，复用其 ShouldReportError / PrintTaskContextInfo 等私有能力
     * @return HCCL_SUCCESS 表示已处理；HCCL_E_NOT_FOUND 表示非 RoCE 链路，交回兜底路径
     */
    static HcclResult ProcessNdaException(const rtExceptionInfo_t* exceptionInfo, TaskExceptionHost& host);

private:
    // 遍历通信域，从 KFC D2H 读出 aicpu 写的 ErrorMessageReport；命中 RoCE 链路则返回 SUCCESS
    static HcclResult
    ReadNdaErrorMessage(const rtExceptionInfo_t* exceptionInfo, Hccl::ErrorMessageReport& errMsg, std::string& commId);

    // 复用 TaskExceptionHost::PrintTaskContextInfo 打印任务前序上下文
    static void PrintNdaTaskContextInfo(
        const rtExceptionInfo_t* exceptionInfo, const Hccl::TaskInfo& task, const std::string& stageErrInfo,
        TaskExceptionHost& host);

    // 分级打印：基础信息 / 参数信息 / groupRank 信息 + 任务上下文序列
    static void PrintNdaErrorLog(
        const rtExceptionInfo_t* exceptionInfo, const Hccl::TaskInfo& task, const std::string& stageErrInfo,
        TaskExceptionHost& host);

    // RoCE 错误码上报：超时走 EI0002，其余走 EI0018
    static void ReportNdaError(
        const Hccl::TaskInfo& task, const Hccl::ErrorMessageReport& errMsg, const rtExceptionInfo_t* exceptionInfo,
        const std::string& stageErrInfo, TaskExceptionHost& host);
};

} // namespace hcomm

#endif // NDA_TASKEXCEPTION_H
