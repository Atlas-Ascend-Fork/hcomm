/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_entrance.h"
#include "aicpu_comm_destroy_func.h"
#include "communicator_impl_lite_manager.h"
#include "ub_conn_lite_mgr.h"
#include "aicpu_daemon_service.h"
#include "task_exception_func.h"
#include "ns_recovery_handler_func.h"
#include "task_exception_handler_lite.h"
#include "log.h"
#include "dpu_taskexp_kernel_param.h" // taskexp kernel 入参块：与下发侧（coll_communicator_mgr）共享同一份定义
#include "inc/aicpu_utils.h"
#ifdef CCL_KERNEL_AICPU
#include "profiling_command_handle_lite.h"
#include "coll_comm_aicpu_mgr.h"
#endif

extern "C" {
using namespace Hccl;

uint32_t HcclKernelEntrance(void* args)
{
    if (args == nullptr) {
        HCCL_ERROR("HcclKernelEntrance Args is null.");
        return 1;
    }

#ifdef CCL_KERNEL_AICPU
    RegisterProfCallBack();
#endif

    auto* kernelParam = reinterpret_cast<HcclKernelParamLite*>(args);
    AicpuUtils::GetInstance().CreateSingleInstance(args);
    CHK_RET(AicpuUtils::GetInstance().Init());
    NsRecoveryHandlerFunc::GetInstance();
    CHK_RET(DlHalFunctionV2::GetInstance().DlHalFunctionInit());

    u32 commIdIndex = kernelParam->comm.idIndex;
    HCCL_RUN_INFO(
        "HcclKernelEntrance begin, OpType[%s] algName[%s] commIdIndex[%u] commId[%s] opTag[%s], devPhyId[%u] "
        "myRank[%u] rankSize[%u] oneSidedComm[%d] opIndex[%u]",
        kernelParam->op.algOperator.opType.Describe().c_str(), kernelParam->algName, commIdIndex,
        kernelParam->comm.commId, kernelParam->opTag, kernelParam->comm.devPhyId, kernelParam->comm.myRank,
        kernelParam->comm.rankSize, kernelParam->oneSidedComm, kernelParam->comm.opIndex_);

    Hccl::CommunicatorImplLite* communicatorImplLite = CommunicatorImplLiteMgr::GetInstance().Get(commIdIndex);
    if (communicatorImplLite == nullptr) {
        HCCL_ERROR("HcclKernelEntrance communicatorImplLite is null.");
        return 1;
    }

    uint32_t ret = 0;
    CHK_RET(AicpuUtils::GetInstance().WaitCommFree(communicatorImplLite, __func__));
    if (communicatorImplLite->LoadWithOpBasedMode(kernelParam) != 0) {
        HCCL_ERROR("HcclKernelEntrance LoadWithOpBasedMode failed.");
        ret = 1;
    } else {
        HCCL_INFO("HcclKernelEntrance success.");
        ret = 0;
    }

    unique_lock<std::mutex> aicpuLock(communicatorImplLite->GetAicpuMc2Mutex());
    communicatorImplLite->SetIsUsed(false);
    aicpuLock.unlock();
    return ret;
}

uint32_t HcclUpdateCommKernelEntrance(void* args)
{
    if (args == nullptr) {
        HCCL_ERROR("[NsRecovery] HcclUpdateCommKernelEntrance Args is null.");
        return 1;
    }

    auto* kernelParam = reinterpret_cast<HcclKernelParamLite*>(args);
    u32 commIdIndex = kernelParam->comm.idIndex;
    HCCL_INFO("[NsRecovery] HcclUpdateCommKernelEntrance begin, commIdIndex[%u]", commIdIndex);

    Hccl::CommunicatorImplLite* communicatorImplLite = CommunicatorImplLiteMgr::GetInstance().Get(commIdIndex);
    if (communicatorImplLite == nullptr) {
        HCCL_ERROR("HcclUpdateCommKernelEntrance communicatorImplLite is null.");
        return 1;
    }

    CHK_RET(AicpuUtils::GetInstance().WaitCommFree(communicatorImplLite, __func__));
    communicatorImplLite->UpdateComm(kernelParam);
    unique_lock<std::mutex> aicpuLock(communicatorImplLite->GetAicpuMc2Mutex());
    communicatorImplLite->SetIsUsed(false);
    aicpuLock.unlock();
    HCCL_INFO("[NsRecovery] HcclUpdateCommKernelEntrance success.");
    return 0;
}

uint32_t HcclDpuTaskexpShmemRestore(void* args)
{
    if (args == nullptr) {
        HCCL_ERROR("HcclDpuTaskexpShmemRestore Args is null.");
        return 1;
    }
    // 入参块与下发侧共用同一份定义（kernel 按裸地址解读 args），避免两侧各维护一份等价结构体后漂移
    auto* kernelParam = reinterpret_cast<DpuTaskexpKernelParam*>(args);
    if (kernelParam->taskexceptionVa == nullptr) {
        HCCL_ERROR("taskexceptionVa is nullptr, please check communicatorImpl init");
        return 1;
    }
    errno_t ret = memset_s(
        kernelParam->taskexceptionVa, kernelParam->memorySize, 0, kernelParam->memorySize); // 避免背景线程读到脏数据
    if (ret != EOK) {
        HCCL_ERROR("kernelParam->taskexceptionVa[%p] set 0 Fail, return[%d]", kernelParam->taskexceptionVa, ret);
        return 1;
    }
    std::string commId = kernelParam->commId;
#ifdef CCL_KERNEL_AICPU
    // 登记点在 ccl_kernel 内；本文件也编入 hccl_v2（host 侧不执行本 kernel），不保护会引用到不存在的符号
    CollCommAicpuMgr::GetInstance().GetTaskExpDevMemRegistry().Register(commId, kernelParam->taskexceptionVa);
    // 回填 comm 自身缓存：dfx 写路径与销毁路径此后都用这份「自己的」VA，不再现查可能已失效的全局登记值
    CollCommAicpu* comm = CollCommAicpuMgr::GetInstance().FindCommByGroup(commId);
    if (comm != nullptr) {
        comm->GetHcclCommDfxLite()->SetTaskExpDevMem(kernelParam->taskexceptionVa);
    }
#endif
    HCCL_INFO(
        "HcclDpuTaskexpShmemRestore success. commId[%s], deviceId[%u], taskexceptionVa[%p], memorySize[%llu]",
        commId.c_str(), kernelParam->deviceId, kernelParam->taskexceptionVa,
        static_cast<unsigned long long>(kernelParam->memorySize));
    return 0;
}
}
