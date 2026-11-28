/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CCU_REP_HANDLE_REPO_H
#define HCOMM_CCU_REP_HANDLE_REPO_H

#include <unordered_map>

#include "ccu_datatype_v1.h"
#include "comm_engine/ccu/ccu_common_pub.h"

namespace hcomm {

// 句柄表：持有六张“句柄 -> rep 资源对象”映射，负责登记与解析。
// 资源对象的创建仍在 CcuKernel（走 res_ 账本），本类只在创建完成后接管句柄映射。
class CcuRepHandleRepo {
public:
    // 解析：按句柄取资源对象。ccu 资源本身可能重载 =，对象赋值会被转换成指令，故返回地址而非值。
    CcuResult ResolveVariable(CcuVariableHandle varHandle, CcuRep::Variable** variable);
    CcuResult ResolveAddress(CcuAddressHandle addrHandle, CcuRep::Address** address);
    CcuResult ResolveEvent(CcuEventHandle eventHandle, CcuRep::CompletedEvent** event);
    CcuResult ResolveBuffer(CcuBufferHandle bufferHandle, CcuRep::CcuBuf** buffer);
    CcuResult ResolveLocalAddr(CcuLocalAddrHandle handle, CcuRep::LocalAddr** localAddr);
    CcuResult ResolveRemoteAddr(CcuRemoteAddrHandle handle, CcuRep::RemoteAddr** remoteAddr);

    // 登记：句柄取自对应表的当前大小，与登记顺序绑定。
    // 约定只由 CcuKernel 的 *Alloc 接口调用：资源对象须先经 res_ 账本创建，直接登记会绕过该账本。
    CcuVariableHandle RegisterVariable(const CcuRep::Variable& variable);
    CcuAddressHandle RegisterAddress(const CcuRep::Address& address);
    CcuEventHandle RegisterEvent(const CcuRep::CompletedEvent& event);
    CcuBufferHandle RegisterBuffer(const CcuRep::CcuBuf& buffer);

    // LocalAddr / RemoteAddr 携带 addr 与 token 两个子对象，三者需分别登记，
    // 顺序固定为 addr、token、自身，改变顺序会使三张表的句柄整体错位。
    CcuLocalAddrHandle
    RegisterLocalAddr(const CcuRep::LocalAddr& localAddr, CcuAddressHandle& addrHandle, CcuVariableHandle& tokenHandle);
    CcuRemoteAddrHandle RegisterRemoteAddr(
        const CcuRep::RemoteAddr& remoteAddr, CcuAddressHandle& addrHandle, CcuVariableHandle& tokenHandle);

private:
    std::unordered_map<CcuVariableHandle, CcuRep::Variable> ccuVarMap_{};
    std::unordered_map<CcuEventHandle, CcuRep::CompletedEvent> ccuEventMap_{};
    std::unordered_map<CcuBufferHandle, CcuRep::CcuBuf> ccuBufferMap_{};
    std::unordered_map<CcuAddressHandle, CcuRep::Address> ccuAddrMap_{};
    std::unordered_map<CcuLocalAddrHandle, CcuRep::LocalAddr> ccuLocalAddrMap_{};
    std::unordered_map<CcuRemoteAddrHandle, CcuRep::RemoteAddr> ccuRemoteAddrMap_{};
};

} // namespace hcomm

#endif // HCOMM_CCU_REP_HANDLE_REPO_H
