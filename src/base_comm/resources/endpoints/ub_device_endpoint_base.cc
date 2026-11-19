/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ub_device_endpoint_base.h"

#include "common/cast_utils_pub.h"
#include "common/process_exit_flag.h"
#include "ep_mem_table_launch.h"
#include "local_ub_rma_buffer.h"
#include "log.h"

namespace hcomm {

HcclResult UbDeviceEndpointBase::InitDevEpMemTable()
{
    // 退出期守卫锚点：首个 kernel 资源（device 表）创建时注册 atexit 置位回调（幂等，进程仅一次）。
    // 本入口仅由 HcommEndpointCreate 末端触达，必然晚于 BaseCommMgr/CollCommMgr 树的析构登记——
    // atexit LIFO 下 SetExiting 先于 ~CollCommMgr 执行，退出期守卫（IsExiting）必然读到 true
    ProcessExitFlag::RegisterExitHook();
    // 创建末端下发 init kernel 并读回表地址；为创建流程最后一个可失败步骤（Add/monitor 之后、
    // 返回之前），失败由编排层 epMgr.Remove 回滚发布
    return LaunchEpMemTableInit(devEpMemTablePtr_);
}

HcclResult UbDeviceEndpointBase::ReleaseDevEpMemTable()
{
    if (devEpMemTablePtr_ == 0) {
        return HCCL_SUCCESS;
    }
    if (ProcessExitFlag::IsExiting()) {
        // 退出期（不销毁通信域直接退出进程的析构链）：runtime 资源不稳定期下发 destroy kernel
        // 会 core，跳过下发；表内存由进程退出统一回收，此处仅清 host 侧表地址
        HCCL_INFO(
            "[%s] process exiting, skip destroy kernel, tablePtr[0x%llx] reclaimed at process exit", __func__,
            devEpMemTablePtr_);
        devEpMemTablePtr_ = 0;
        return HCCL_SUCCESS;
    }
    HcclResult ret = LaunchEpMemTableDestroy(devEpMemTablePtr_);
    devEpMemTablePtr_ = 0;
    return ret;
}

uint64_t UbDeviceEndpointBase::GetDevEpMemTablePtr() const { return devEpMemTablePtr_; }

HcclResult UbDeviceEndpointBase::OnLocalMemRegistered(void* memHandle)
{
    // Reg 成功后对象存活，此处读四字段安全
    auto* buffer = ReinterpretAs<Hccl::LocalUbRmaBuffer*>(memHandle);
    CHK_PTR_NULL(buffer);
    return SyncUbLocMemToEpTable(
        devEpMemTablePtr_, EpMemTableOp::UPSERT, buffer->GetAddr(), buffer->GetSize(), buffer->GetTokenId(),
        buffer->GetTokenValue());
}

HcclResult UbDeviceEndpointBase::OnLocalMemUnregistered(uint64_t addr, uint64_t size)
{
    if (ProcessExitFlag::IsExiting()) {
        // 退出期（~CollCommMgr→~EndpointMgr→~TaggedMemMap 兜底注销链）：runtime 资源不稳定期
        // 下发 REMOVE kernel 会 core，跳过下发；host 账本已由 HcommMemUnreg 主流程摘除，
        // device 表残留条目随进程退出回收
        HCCL_INFO(
            "[%s] process exiting, skip REMOVE kernel, addr[0x%llx] size[0x%llx] reclaimed at process exit", __func__,
            addr, size);
        return HCCL_SUCCESS;
    }
    // 严格同步：host 账本已摘除即无条件 REMOVE；纯值入参（对象可能已被 mgr 注销析构）。
    // 同 (addr,size) 重复注册由表内 refcount 聚合，归零才删条目；原 IsRangeRegistered 启发式已删除。
    // REMOVE 仅消费 addr/size，token 字段填 0（kernel 侧忽略）
    return SyncUbLocMemToEpTable(devEpMemTablePtr_, EpMemTableOp::REMOVE, addr, size, 0, 0);
}
} // namespace hcomm
