/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <new>

#include "endpoint_pair/channel/channel_param_pub.h"
#include "dev_ep_mem_table.h"
#include "common/cast_utils_pub.h"
#include "log.h"

// 符号导出方式同 RunAicpuChannelMemUpdate（coll_comm_aicpu_kernel.cc）：extern "C" + visibility(default)。
// 本文件仅编入 libbase_comm_aicpu.so（该库不挂 version-script，default 符号全量导出，主线形态 AICPU
// 按名解析经 ccl_kernel 的 DT_NEEDED 命中，无需 ccl_kernel.map 白名单）；aicpu_custom 形态以
// --whole-archive 静态打入 base_comm_aicpu_a，按名解析白名单见 aicpu_custom.map（RunAicpuThread* 先例）。
// args 为 device 内存中的 context 指针（*ReinterpretAs<uint64_t*>(args)），同 :146 先例。
extern "C" {
// endpoint 创建：device 侧 new 空表，回填 {tablePtr, magic, version} 到输出槽（host D2H 读回校验）
__attribute__((visibility("default"))) uint32_t RunAicpuEpMemTableInit(void* args)
{
    CHK_PTR_NULL(args);
    uint64_t devAddr = *ReinterpretAs<uint64_t*>(args);
    auto* param = ReinterpretAs<HcclAicpuEpMemTableInitRes*>(devAddr);
    CHK_PTR_NULL(param);
    CHK_PTR_NULL(ReinterpretAs<void*>(param->outAddr));

    Hccl::DevEpMemTable* table = nullptr;
    // EmptySnapshot 首次调用存在一次性静态初始化分配（可抛 bad_alloc）——nothrow 仅覆盖 operator new、
    // 不捕获构造函数异常，逃逸 kernel 入口 = AICPU terminate；就地转错误码（同 Update/Remove 的
    // EXCEPTION_CATCH 款），走 InitDevEpMemTable 失败 → epMgr.Remove 回滚 → HcommEndpointCreate 干净失败
    EXCEPTION_CATCH(table = new (std::nothrow) Hccl::DevEpMemTable(), return HCCL_E_MEMORY);
    CHK_PTR_NULL(table);

    auto* output = ReinterpretAs<EpMemTableInitOutput*>(param->outAddr);
    output->tablePtr = ReinterpretAs<uint64_t>(table);
    output->magic = table->abi.magic;
    output->version = table->abi.version;
    HCCL_INFO("[RunAicpuEpMemTableInit] tablePtr[0x%llx]", output->tablePtr);
    return HCCL_SUCCESS;
}

// HcommMemReg / HcommMemUnreg 成功后的运行时增量：单条 upsert / remove
__attribute__((visibility("default"))) uint32_t RunAicpuEpMemTableUpdate(void* args)
{
    CHK_PTR_NULL(args);
    uint64_t devAddr = *ReinterpretAs<uint64_t*>(args);
    auto* param = ReinterpretAs<HcclAicpuEpMemTableUpdateRes*>(devAddr);
    CHK_PTR_NULL(param);
    // kernel ABI 入口 fail-safe：空表指针/非法 op 直接拒绝（host 侧已有守卫，此处兜底畸形入参）
    CHK_PRT_RET(param->tablePtr == 0, HCCL_ERROR("[RunAicpuEpMemTableUpdate] tablePtr is null"), HCCL_E_PARA);
    CHK_PRT_RET(
        param->op != static_cast<uint32_t>(EpMemTableOp::UPSERT)
            && param->op != static_cast<uint32_t>(EpMemTableOp::REMOVE),
        HCCL_ERROR("[RunAicpuEpMemTableUpdate] invalid op[%u]", param->op), HCCL_E_PARA);

    // 哨兵校验：表已销毁/错指针快速失败，避免写花 AICPU 堆
    auto* table = ReinterpretAs<Hccl::DevEpMemTable*>(param->tablePtr);
    CHK_PRT_RET(
        !table->CheckAbi(), HCCL_ERROR("[RunAicpuEpMemTableUpdate] bad abi, tablePtr[0x%llx]", param->tablePtr),
        HCCL_E_INTERNAL);

    HcclResult ret = HCCL_SUCCESS;
    if (param->op == static_cast<uint32_t>(EpMemTableOp::UPSERT)) {
        const LocUbBufLite entry{param->addr, param->size, param->tokenId, param->tokenValue};
        // 快照拷贝可抛 bad_alloc——kernel 入口异常逃逸 = AICPU terminate，EXCEPTION_CATCH 就地转错误码
        EXCEPTION_CATCH(ret = table->Update(entry), return HCCL_E_MEMORY); // 同键 ref++ / 新键插入（ref=1）
    } else {
        // 同上：快照拷贝异常转错误码（NOT_FOUND 下方降级放行）
        EXCEPTION_CATCH(ret = table->Remove(param->addr, param->size), return HCCL_E_MEMORY); // ref-- 归零删
    }
    // 未命中放行：幂等/乱序注销的目标态（表内无该条目）已达成，降级 WARNING 并按成功返回，
    // 与 host 侧"仅告警不回滚"语义对齐，避免良性未命中以 ERROR 级污染日志/DFX
    if (ret == HCCL_E_NOT_FOUND) {
        HCCL_WARNING(
            "[RunAicpuEpMemTableUpdate] op[%u] addr[0x%llx] not in table, treat as idempotent remove", param->op,
            param->addr);
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[RunAicpuEpMemTableUpdate] op[%u] addr[0x%llx] failed, ret[%d]", param->op, param->addr, ret), ret);
    return HCCL_SUCCESS;
}

// endpoint 销毁：delete（正常路径 = channel 已全部销毁；异常顺序本迭代不设防，靠哨兵快速失败）
__attribute__((visibility("default"))) uint32_t RunAicpuEpMemTableDestroy(void* args)
{
    CHK_PTR_NULL(args);
    uint64_t devAddr = *ReinterpretAs<uint64_t*>(args);
    auto* param = ReinterpretAs<HcclAicpuEpMemTableDestroyRes*>(devAddr);
    CHK_PTR_NULL(param);
    CHK_PRT_RET(param->tablePtr == 0, HCCL_ERROR("[RunAicpuEpMemTableDestroy] tablePtr is null"), HCCL_E_PARA);

    auto* table = ReinterpretAs<Hccl::DevEpMemTable*>(param->tablePtr);
    CHK_PRT_RET(
        !table->CheckAbi(), HCCL_ERROR("[RunAicpuEpMemTableDestroy] bad abi, tablePtr[0x%llx]", param->tablePtr),
        HCCL_E_INTERNAL);
    delete table;
    HCCL_INFO("[RunAicpuEpMemTableDestroy] tablePtr[0x%llx] deleted", param->tablePtr);
    return HCCL_SUCCESS;
}
}
