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
#define HCCL_VM_MODULE "CONFIG_STUB"

#include <atomic>
#include <cstdint>
#include <iostream>
#include <unistd.h>
#include <vector>

#include "acl/acl_base.h"
#include "acl/acl_rt.h"
#include "sim_log.h"
#include "runtime/base.h"
#include "runtime/event.h"
#include "runtime_state/db_sim_runner_common.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "ascend_hal.h"

// 将 aclrt 设备资源限制类型映射到 halGetDeviceInfo 的 moduleType, 对齐真实 runtime 数据源:
// RawDevice::InitResource 用 halGetDeviceInfo(AICORE/VECTOR_CORE, CORE_NUM) 初始化 resLimitArray_,
// 之后 aclrtGetStreamResLimit / aclrtGetResInCurrentThread 读取该缓存值(dev->GetResValue)。
static int32_t MapResLimitTypeToModule(aclrtDevResLimitType type)
{
    switch (type) {
        case ACL_RT_DEV_RES_CUBE_CORE:
            return static_cast<int32_t>(MODULE_TYPE_AICORE);
        case ACL_RT_DEV_RES_VECTOR_CORE:
            return static_cast<int32_t>(MODULE_TYPE_VECTOR_CORE);
        default:
            return -1;
    }
}

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

aclError aclrtSetSysParamOpt(aclSysParamOpt opt, int64_t value)
{
    (void)opt;
    (void)value;
    HCCL_VM_WARN("not support");
    return ACL_SUCCESS;
}

aclError aclrtGetSysParamOpt(aclSysParamOpt opt, int64_t* value)
{
    (void)opt;
    (void)value;
    HCCL_VM_WARN("not support");
    return ACL_SUCCESS;
}

aclError aclrtGetDeviceResLimit(int32_t deviceId, aclrtDevResLimitType type, uint32_t* value)
{
    (void)deviceId;
    (void)type;
    (void)value;
    HCCL_VM_WARN("not support");
    return ACL_SUCCESS;
}

aclError aclrtSetDeviceResLimit(int32_t deviceId, aclrtDevResLimitType type, uint32_t value)
{
    (void)deviceId;
    (void)type;
    (void)value;
    HCCL_VM_WARN("not support");
    return ACL_SUCCESS;
}

aclError aclrtResetDeviceResLimit(int32_t deviceId)
{
    (void)deviceId;
    HCCL_VM_WARN("not support");
    return ACL_SUCCESS;
}

aclError aclrtGetStreamResLimit(aclrtStream stream, aclrtDevResLimitType type, uint32_t* value)
{
    // 模拟环境不持久化 stream 级 res limit(aclrtSetStreamResLimit 为空实现),
    // 等价于真实 runtime 中"stream 未设置 res limit 时回退到 device 级"的 GetStreamResLimitByType 逻辑。
    (void)stream;
    if (value == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    int32_t moduleType = MapResLimitTypeToModule(type);
    if (moduleType < 0) {
        HCCL_VM_WARN("unsupported res limit type[{}]", static_cast<int32_t>(type));
        return ACL_ERROR_INVALID_PARAM;
    }
    int64_t coreNum = 0;
    drvError_t drvRet = halGetDeviceInfo(
        static_cast<uint32_t>(sim::runtime::GetCurrDeviceId()), moduleType, static_cast<int32_t>(INFO_TYPE_CORE_NUM),
        &coreNum);
    if (drvRet != DRV_ERROR_NONE) {
        HCCL_VM_ERROR("halGetDeviceInfo fail, moduleType[{}], drvRet[{}]", moduleType, static_cast<int32_t>(drvRet));
        return ACL_ERROR_RT_FAILURE;
    }
    *value = static_cast<uint32_t>(coreNum);
    return ACL_SUCCESS;
}

aclError aclrtSetStreamResLimit(aclrtStream stream, aclrtDevResLimitType type, uint32_t value)
{
    (void)stream;
    (void)type;
    (void)value;
    HCCL_VM_WARN("not support");
    return ACL_SUCCESS;
}

aclError aclrtResetStreamResLimit(aclrtStream stream)
{
    (void)stream;
    HCCL_VM_WARN("not support");
    return ACL_SUCCESS;
}

aclError aclrtUseStreamResInCurrentThread(aclrtStream stream)
{
    (void)stream;
    HCCL_VM_WARN("not support");
    return ACL_SUCCESS;
}

aclError aclrtUnuseStreamResInCurrentThread(aclrtStream stream)
{
    (void)stream;
    HCCL_VM_WARN("not support");
    return ACL_SUCCESS;
}

aclError aclrtGetResInCurrentThread(aclrtDevResLimitType type, uint32_t* value)
{
    // 对齐真实 runtime ApiImpl::GetResInCurrentThread: 未绑定 stream 级 res limit 时返回 device 级 core 数
    if (value == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    int32_t moduleType = MapResLimitTypeToModule(type);
    if (moduleType < 0) {
        HCCL_VM_WARN("unsupported res limit type[{}]", static_cast<int32_t>(type));
        return ACL_ERROR_INVALID_PARAM;
    }
    int64_t coreNum = 0;
    drvError_t drvRet = halGetDeviceInfo(
        static_cast<uint32_t>(sim::runtime::GetCurrDeviceId()), moduleType, static_cast<int32_t>(INFO_TYPE_CORE_NUM),
        &coreNum);
    if (drvRet != DRV_ERROR_NONE) {
        HCCL_VM_ERROR("halGetDeviceInfo fail, moduleType[{}], drvRet[{}]", moduleType, static_cast<int32_t>(drvRet));
        return ACL_ERROR_RT_FAILURE;
    }
    *value = static_cast<uint32_t>(coreNum);
    return ACL_SUCCESS;
}

aclError aclrtGetOpTimeOutInterval(uint64_t* interval)
{
    if (interval == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    *interval = 5ULL * 1000ULL * 1000ULL;
    return ACL_SUCCESS;
}

#ifdef __cplusplus
}
#endif // __cplusplus
