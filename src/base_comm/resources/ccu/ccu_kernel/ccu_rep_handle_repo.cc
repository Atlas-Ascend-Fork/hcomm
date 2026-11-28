/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_rep_handle_repo.h"

#include "ccu_log.h"

namespace hcomm {

namespace {
    template <typename HandleType, typename ResourceType>
    CcuResult GetResourceByHandle(
        std::unordered_map<HandleType, ResourceType>& resourceMap, HandleType handle, ResourceType** resource,
        const char* resourceType)
    {
        auto iter = resourceMap.find(handle);
        if (iter == resourceMap.end()) {
            HCCL_ERROR("[%s] failed to find %s by handle: 0x%llx", __func__, resourceType, handle);
            return CcuResult::CCU_E_NOT_FOUND;
        }

        // ccu资源本身可能重载=，对象赋值会被转换成指令，导致流程失败
        *resource = &(iter->second);
        return CcuResult::CCU_SUCCESS;
    }
} // namespace

CcuResult CcuRepHandleRepo::ResolveVariable(CcuVariableHandle varHandle, CcuRep::Variable** variable)
{
    return GetResourceByHandle(ccuVarMap_, varHandle, variable, "variable");
}

CcuResult CcuRepHandleRepo::ResolveAddress(CcuAddressHandle addrHandle, CcuRep::Address** address)
{
    return GetResourceByHandle(ccuAddrMap_, addrHandle, address, "address");
}

CcuResult CcuRepHandleRepo::ResolveEvent(CcuEventHandle eventHandle, CcuRep::CompletedEvent** event)
{
    return GetResourceByHandle(ccuEventMap_, eventHandle, event, "completedEvent");
}

CcuResult CcuRepHandleRepo::ResolveBuffer(CcuBufferHandle bufferHandle, CcuRep::CcuBuf** buffer)
{
    return GetResourceByHandle(ccuBufferMap_, bufferHandle, buffer, "buffer");
}

CcuResult CcuRepHandleRepo::ResolveLocalAddr(CcuLocalAddrHandle handle, CcuRep::LocalAddr** localAddr)
{
    return GetResourceByHandle(ccuLocalAddrMap_, handle, localAddr, "localAddr");
}

CcuResult CcuRepHandleRepo::ResolveRemoteAddr(CcuRemoteAddrHandle handle, CcuRep::RemoteAddr** remoteAddr)
{
    return GetResourceByHandle(ccuRemoteAddrMap_, handle, remoteAddr, "remoteAddr");
}

CcuVariableHandle CcuRepHandleRepo::RegisterVariable(const CcuRep::Variable& variable)
{
    CcuVariableHandle handle = ccuVarMap_.size();
    ccuVarMap_.emplace(handle, variable);
    return handle;
}

CcuAddressHandle CcuRepHandleRepo::RegisterAddress(const CcuRep::Address& address)
{
    CcuAddressHandle handle = ccuAddrMap_.size();
    ccuAddrMap_.emplace(handle, address);
    return handle;
}

CcuEventHandle CcuRepHandleRepo::RegisterEvent(const CcuRep::CompletedEvent& event)
{
    CcuEventHandle handle = ccuEventMap_.size();
    ccuEventMap_.emplace(handle, event);
    return handle;
}

CcuBufferHandle CcuRepHandleRepo::RegisterBuffer(const CcuRep::CcuBuf& buffer)
{
    CcuBufferHandle handle = ccuBufferMap_.size();
    ccuBufferMap_.emplace(handle, buffer);
    return handle;
}

CcuLocalAddrHandle CcuRepHandleRepo::RegisterLocalAddr(
    const CcuRep::LocalAddr& localAddr, CcuAddressHandle& addrHandle, CcuVariableHandle& tokenHandle)
{
    addrHandle = RegisterAddress(localAddr.addr);
    tokenHandle = RegisterVariable(localAddr.token);

    CcuLocalAddrHandle laHandle = ccuLocalAddrMap_.size();
    ccuLocalAddrMap_.emplace(laHandle, localAddr);
    return laHandle;
}

CcuRemoteAddrHandle CcuRepHandleRepo::RegisterRemoteAddr(
    const CcuRep::RemoteAddr& remoteAddr, CcuAddressHandle& addrHandle, CcuVariableHandle& tokenHandle)
{
    addrHandle = RegisterAddress(remoteAddr.addr);
    tokenHandle = RegisterVariable(remoteAddr.token);

    CcuRemoteAddrHandle raHandle = ccuRemoteAddrMap_.size();
    ccuRemoteAddrMap_.emplace(raHandle, remoteAddr);
    return raHandle;
}

} // namespace hcomm
