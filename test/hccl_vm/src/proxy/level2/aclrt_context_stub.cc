/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 日志染色: 模块 tag (须在 include sim_log.h 之前)
#define HCCL_VM_MODULE "CTX_STUB"

#include <atomic>
#include <cstdint>
#include <iostream>
#include <unistd.h>
#include <utility>
#include <vector>

#include "acl/acl_base.h"
#include "acl/acl_rt.h"
#include "hccl/hccl_types.h"
#include "sim_common_macro.h"
#include "sim_log.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "runtime_state/db_sim_runner_common.h"
#include "storage/table_access.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
aclError aclrtCreateContext(aclrtContext* context, int32_t deviceId)
{
    auto serverId = sim::runtime::GetCurServerId();
    if (serverId == 0) {
        return ACL_ERROR_INVALID_PARAM;
    }
    sim::runtime::Runner runner;
    if (!sim::runtime::GetCurrRunnerTls(serverId, runner)) {
        return ACL_ERROR_INVALID_PARAM;
    }
    auto ret = sim::runtime::Db::GetOneByPred<sim::runtime::Device>(
        HcclSim::Storage::Eq(&sim::runtime::Device::logic_id, (uint32_t)deviceId));
    if (!ret.ok()) {
        HCCL_VM_ERROR("device not found:{:d}", deviceId);
        return HcclResult::HCCL_E_NOT_FOUND;
    }
    // 找不到context，则默认新增一个context
    sim::runtime::Context ctx{};
    ctx.device_id = ret->id;
    ctx.run_id = runner.id;
    ctx.id = 0;
    auto insertedContext = sim::runtime::Db::Add<sim::runtime::Context>(ctx);
    if (!insertedContext.ok() || !insertedContext.value.has_value()) {
        return ACL_ERROR_INVALID_PARAM;
    }
    auto currCtxId = insertedContext.value->value;

    // 新建context，默认新建一个stream
    sim::runtime::Stream stream{};
    stream.ctx_id = currCtxId;
    stream.is_primary_default = 1;
    stream.activated = 1;
    stream.id = 0;
    auto insertedStream = sim::runtime::Db::Add<sim::runtime::Stream>(stream);
    if (!insertedStream.ok() || !insertedStream.value.has_value()) {
        return ACL_ERROR_INVALID_PARAM;
    }
    *context = (aclrtContext)currCtxId;
    return ACL_SUCCESS;
}

aclError aclrtDestroyContext(aclrtContext context)
{
    uint64_t ctxId = (uint64_t)(uintptr_t)context;

    auto ret = sim::runtime::Db::GetOneByPred<sim::runtime::Stream>(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&sim::runtime::Stream::ctx_id, ctxId),
        HcclSim::Storage::Eq(&sim::runtime::Stream::is_primary_default, 1)));
    if (!ret.ok()) {
        HCCL_VM_ERROR("stream not found ctxId:{:d}", ctxId);
        return HcclResult::HCCL_E_NOT_FOUND;
    }

    (void)sim::runtime::Db::Delete<sim::runtime::Stream>(HcclSim::Storage::Eq(&sim::runtime::Stream::id, ret->id));
    (void)sim::runtime::Db::Delete<sim::runtime::Context>(HcclSim::Storage::Eq(&sim::runtime::Context::id, ctxId));
    return ACL_SUCCESS;
}

aclError aclrtSetCurrentContext(aclrtContext context)
{
    uint64_t ctxId = (uint64_t)(uintptr_t)context;
    sim::runtime::SetCurrCtxTls(ctxId);
    return ACL_SUCCESS;
}

aclError aclrtGetCurrentContext(aclrtContext* context)
{
    auto serverId = sim::runtime::GetCurServerId();
    if (serverId == 0) {
        return ACL_ERROR_INVALID_PARAM;
    }
    sim::runtime::Runner runner;
    if (!sim::runtime::GetCurrRunnerTls(serverId, runner)) {
        return ACL_ERROR_INVALID_PARAM;
    }
    *context = (aclrtContext)runner.current_ctx_id;
    return ACL_SUCCESS;
}

aclError aclrtCtxGetCurrentDefaultStream(aclrtStream* stream)
{
    auto serverId = sim::runtime::GetCurServerId();
    if (serverId == 0) {
        return ACL_ERROR_INVALID_PARAM;
    }
    sim::runtime::Runner runner;
    if (!sim::runtime::GetCurrRunnerTls(serverId, runner)) {
        return ACL_ERROR_INVALID_PARAM;
    }
    auto curCtxId = runner.current_ctx_id;
    auto stm = sim::runtime::Db::GetOneByPred<sim::runtime::Stream>(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&sim::runtime::Stream::ctx_id, curCtxId),
        HcclSim::Storage::Eq(&sim::runtime::Stream::is_primary_default, 1)));
    if (!stm.ok()) {
        HCCL_VM_ERROR("stream not found ctxId:{:d}", curCtxId);
        return HcclResult::HCCL_E_NOT_FOUND;
    }

    *stream = (aclrtStream)stm->id;
    return ACL_SUCCESS;
}

aclError aclrtGetPrimaryCtxState(int32_t deviceId, uint32_t* flags, int32_t* active)
{
    (void)deviceId;
    (void)flags;
    auto serverId = sim::runtime::GetCurServerId();
    if (serverId == 0) {
        return ACL_ERROR_INVALID_PARAM;
    }
    sim::runtime::Runner runner;
    if (!sim::runtime::GetCurrRunnerTls(serverId, runner)) {
        return ACL_ERROR_INVALID_PARAM;
    }
    auto currCtx = sim::runtime::Db::GetById<sim::runtime::Context>(runner.current_ctx_id);
    if (!currCtx.ok()) {
        // not find
        HCCL_VM_ERROR("ctx not found:{:d}", runner.current_ctx_id);
        return ACL_ERROR_INVALID_PARAM;
    }

    *active = (currCtx.value->is_default == 1) ? 1 : 0;
    return ACL_SUCCESS;
}

aclError aclrtCtxSetSysParamOpt(aclSysParamOpt opt, int64_t value)
{
    (void)opt;
    (void)value;
    return ACL_SUCCESS;
}

aclError aclrtCtxGetSysParamOpt(aclSysParamOpt opt, int64_t* value)
{
    (void)opt;
    (void)value;
    return ACL_SUCCESS;
}

aclError aclrtCtxGetFloatOverflowAddr(void** overflowAddr)
{
    (void)overflowAddr;
    return ACL_SUCCESS;
}

#ifdef __cplusplus
}
#endif // __cplusplus
