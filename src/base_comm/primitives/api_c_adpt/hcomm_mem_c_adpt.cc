/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include "hcomm_c_adpt.h"
#include "hcomm_c_adpt_common.h"
#include "hcomm_res_mgr.h"
#include "log.h"
#include "endpoint.h"
#include "param_check_pub.h"
#include "exception_handler.h"
#include "hcomm_res_defs.h"
#include "hcomm_res.h"
#include "hcomm_mem_alloc.h"
#include "hccs_reged_mem_mgr.h"

using namespace hcomm;

namespace {
EndpointMgr& GetEndpointMgrWithInit()
{
    (void)HcommResMgrInit();
    return HcommResMgr::GetInstance().GetEndpointMgr();
}
} // namespace

HcommResult
HcommMemReg(EndpointHandle endpointHandle, const char* memTag, const CommMem* mem, HcommMemHandle* memHandle)
{
    if (memTag != nullptr && mem != nullptr && memHandle != nullptr) {
        size_t memTagLen = strnlen(memTag, HCOMM_RES_TAG_MAX_LEN + 1);
        CHK_PRT_RET(
            memTagLen > HCOMM_RES_TAG_MAX_LEN,
            HCCL_ERROR("[%s] memTag too long, len[%zu], max[%u].", __func__, memTagLen, HCOMM_RES_TAG_MAX_LEN),
            HCCL_E_PARA);
    }
    auto endpoint = GetEndpointMgrWithInit().Get(endpointHandle);
    CHK_PRT_RET(
        endpoint == nullptr, HCCL_ERROR("[%s] endpoint not found, endpointHandle[%p]", __func__, endpointHandle),
        HCCL_E_NOT_FOUND);
    auto mgr = endpoint->GetLocalRegMemMgr();
    CHK_PTR_NULL(mgr);
    CHK_RET(RefreshEndpointContext(endpoint->GetEndpointDesc()));
    CHK_RET(static_cast<HcclResult>(mgr->RegisterMemory(mem, memTag, static_cast<void**>(memHandle))));
    // 注册成功后记入本 endpoint 归属，供 GetAllLocalMemHandles 按实例枚举；记账失败回滚注册，避免泄漏与串数据
    HcclResult recordRet = endpoint->RecordLocalMemHandle(*memHandle);
    if (recordRet != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[%s] RecordLocalMemHandle failed, ret[%d], rollback register", __func__, static_cast<int>(recordRet));
        (void)mgr->UnregisterMemory(*memHandle);
        return static_cast<HcommResult>(recordRet);
    }
    return HCCL_SUCCESS;
}

HcommResult HcommMemUnreg(EndpointHandle endpointHandle, HcommMemHandle memHandle)
{
    auto endpoint = GetEndpointMgrWithInit().Get(endpointHandle);
    CHK_PRT_RET(
        endpoint == nullptr, HCCL_ERROR("[%s] endpoint not found, endpointHandle[%p]", __func__, endpointHandle),
        HCCL_E_NOT_FOUND);
    auto mgr = endpoint->GetLocalRegMemMgr();
    CHK_PTR_NULL(mgr);
    CHK_RET(RefreshEndpointContext(endpoint->GetEndpointDesc()));
    // 先摘本实例归属账：handle 非本实例注册时在此拒绝（硬件未动），避免跨实例注销导致两份账本静默发散；
    // 摘账命中则 mgr 必有记录（注册路径同源保证），随后注销不会因记录缺失失败
    CHK_RET(endpoint->RemoveLocalMemHandle(memHandle));
    CHK_RET(static_cast<HcclResult>(mgr->UnregisterMemory(memHandle)));
    return HCCL_SUCCESS;
}

HcommResult
HcommMemExport(EndpointHandle endpointHandle, HcommMemHandle memHandle, void** memDesc, uint32_t* memDescLen)
{
    auto endpoint = GetEndpointMgrWithInit().Get(endpointHandle);
    CHK_PRT_RET(
        endpoint == nullptr, HCCL_ERROR("[%s] endpoint not found, endpointHandle[%p]", __func__, endpointHandle),
        HCCL_E_NOT_FOUND);
    auto mgr = endpoint->GetLocalRegMemMgr();
    CHK_PTR_NULL(mgr);
    CHK_RET(RefreshEndpointContext(endpoint->GetEndpointDesc()));
    return static_cast<HcclResult>(mgr->MemoryExport(endpoint->GetEndpointDesc(), memHandle, memDesc, memDescLen));
}

HcommResult HcommMemImport(EndpointHandle endpointHandle, const void* memDesc, uint32_t descLen, CommMem* outMem)
{
    auto endpoint = GetEndpointMgrWithInit().Get(endpointHandle);
    CHK_PRT_RET(
        endpoint == nullptr, HCCL_ERROR("[%s] endpoint not found, endpointHandle[%p]", __func__, endpointHandle),
        HCCL_E_NOT_FOUND);
    auto remoteMgr = endpoint->GetRemoteRegMemMgr();
    CHK_PTR_NULL(remoteMgr);
    CHK_RET(RefreshEndpointContext(endpoint->GetEndpointDesc()));
    return static_cast<HcclResult>(remoteMgr->MemoryImport(memDesc, descLen, outMem));
}

HcommResult HcommMemUnimport(EndpointHandle endpointHandle, const void* memDesc, uint32_t descLen)
{
    auto endpoint = GetEndpointMgrWithInit().Get(endpointHandle);
    CHK_PRT_RET(
        endpoint == nullptr, HCCL_ERROR("[%s] endpoint not found, endpointHandle[%p]", __func__, endpointHandle),
        HCCL_E_NOT_FOUND);
    auto remoteMgr = endpoint->GetRemoteRegMemMgr();
    CHK_PTR_NULL(remoteMgr);
    CHK_RET(RefreshEndpointContext(endpoint->GetEndpointDesc()));
    return static_cast<HcclResult>(remoteMgr->MemoryUnimport(memDesc, descLen));
}

HcommResult HcommMemGrant(EndpointHandle endpointHandle, const HcommMemGrantInfo* remoteGrantInfo)
{
    CHK_PTR_NULL(remoteGrantInfo);
    HCCL_INFO("[%s] START. endpointHandle[0x%llx].", __func__, endpointHandle);

    auto endpoint = HcommResMgr::GetInstance().GetEndpointMgr().Get(endpointHandle);
    CHK_PRT_RET(
        endpoint == nullptr, HCCL_ERROR("[%s] endpoint not found, endpointHandle[0x%llx]", __func__, endpointHandle),
        HCCL_E_NOT_FOUND);
    auto mgr = endpoint->GetLocalRegMemMgr();
    CHK_PTR_NULL(mgr);
    // MemoryGrant 由 HccsRegedMemMgr 承载；非 HCCS 类 mgr 无此能力，跳过（与下移前基类默认 SUCCESS 行为一致）
    auto* hccsMgr = dynamic_cast<HccsRegedMemMgr*>(mgr);
    if (hccsMgr == nullptr) {
        return HCCL_SUCCESS;
    }
    return hccsMgr->MemoryGrant(remoteGrantInfo);
}

/* 暂未实现 */
HcommResult HcommMemRemap(
    [[maybe_unused]] const EndpointHandle endpointHandle, [[maybe_unused]] const CommMem* memArray,
    [[maybe_unused]] uint64_t arraySize)
{
    return HCCL_E_NOT_SUPPORT;
}

HcommResult HcommMemGetAllMemHandles(EndpointHandle endpointHandle, void** memHandles, uint32_t* memHandleNum)
{
    CHK_PTR_NULL(memHandles);
    CHK_PTR_NULL(memHandleNum);

    auto endpoint = HcommResMgr::GetInstance().GetEndpointMgr().Get(endpointHandle);
    CHK_PRT_RET(
        endpoint == nullptr, HCCL_ERROR("[%s] endpoint not found, endpointHandle[0x%llx]", __func__, endpointHandle),
        HCCL_E_NOT_FOUND);
    CHK_RET(endpoint->GetAllLocalMemHandles(memHandles, memHandleNum));
    return HCCL_SUCCESS;
}

HcommResult HcommMemAlloc(void** ptr, size_t size) { return hcomm::MemAlloc(ptr, size); }

HcommResult HcommMemFree(void* ptr) { return hcomm::MemFree(ptr); }
