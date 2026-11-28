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

extern "C" {
__attribute__((visibility("default"))) uint32_t RunAicpuEpMemTableInit(void* args)
{
    CHK_PTR_NULL(args);
    uint64_t devAddr = *ReinterpretAs<uint64_t*>(args);
    auto* param = ReinterpretAs<HcclAicpuEpMemTableInitRes*>(devAddr);
    CHK_PTR_NULL(param);
    CHK_PTR_NULL(ReinterpretAs<void*>(param->outAddr));

    Hccl::DevEpMemTable* table = nullptr;
    EXCEPTION_CATCH(table = new (std::nothrow) Hccl::DevEpMemTable(), return HCCL_E_MEMORY);
    CHK_PTR_NULL(table);

    auto* output = ReinterpretAs<EpMemTableInitOutput*>(param->outAddr);
    output->tablePtr = ReinterpretAs<uint64_t>(table);
    output->magic = table->abi.magic;
    output->version = table->abi.version;
    HCCL_INFO("[RunAicpuEpMemTableInit] tablePtr[0x%llx]", output->tablePtr);
    return HCCL_SUCCESS;
}

__attribute__((visibility("default"))) uint32_t RunAicpuEpMemTableUpdate(void* args)
{
    CHK_PTR_NULL(args);
    uint64_t devAddr = *ReinterpretAs<uint64_t*>(args);
    auto* param = ReinterpretAs<HcclAicpuEpMemTableUpdateRes*>(devAddr);
    CHK_PTR_NULL(param);
    CHK_PRT_RET(param->tablePtr == 0, HCCL_ERROR("[RunAicpuEpMemTableUpdate] tablePtr is null"), HCCL_E_PARA);
    CHK_PRT_RET(
        param->op != static_cast<uint32_t>(EpMemTableOp::UPSERT)
            && param->op != static_cast<uint32_t>(EpMemTableOp::REMOVE),
        HCCL_ERROR("[RunAicpuEpMemTableUpdate] invalid op[%u]", param->op), HCCL_E_PARA);

    auto* table = ReinterpretAs<Hccl::DevEpMemTable*>(param->tablePtr);
    CHK_PRT_RET(
        !table->CheckAbi(), HCCL_ERROR("[RunAicpuEpMemTableUpdate] bad abi, tablePtr[0x%llx]", param->tablePtr),
        HCCL_E_INTERNAL);

    HcclResult ret = HCCL_SUCCESS;
    if (param->op == static_cast<uint32_t>(EpMemTableOp::UPSERT)) {
        const LocUbBufLite entry{param->addr, param->size, param->tokenId, param->tokenValue};
        EXCEPTION_CATCH(ret = table->Update(entry), return HCCL_E_MEMORY);
    } else {
        EXCEPTION_CATCH(ret = table->Remove(param->addr, param->size), return HCCL_E_MEMORY);
    }
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
