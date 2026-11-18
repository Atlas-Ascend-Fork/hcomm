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
#define HCCL_VM_MODULE "NEW_STUB"

#include <atomic>
#include <cstdint>
#include <iostream>
#include <unistd.h>
#include <vector>

#include "acl/acl_base.h"
#include "acl/acl_rt.h"
#include "ascend_hal.h"
#include "atrace_types.h"
#include "dtype_common.h"
#include <cstring>
#include "sim_log.h"
#include "rts_device.h"
#include "sim_capacity_limits.h"
#include "runtime_state/sim_models.h"
#include "runtime_state/db_sim_runner_common.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "storage/table_access.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

HcclResult hrtGetDeviceType(DevType& devType)
{
    // 设备类型按 SoC 版本判定（与真机 hrtGetSocVer -> aclrtGetSocName 对齐），不依赖 aclrtSetDevice
    // 建立的当前 context；首次查询时取数据库中第一条 Device 记录（主键 id 升序）并缓存到静态变量，
    // 后续查询直接返回缓存值（设备类型在进程生命周期内不会变化）。
    // 仅缓存查询成功的结果，Device 表尚为空时下次调用仍会重试。
    static DevType cachedDevType = DevType::DEV_TYPE_COUNT;
    if (cachedDevType != DevType::DEV_TYPE_COUNT) {
        devType = cachedDevType;
        return HCCL_SUCCESS;
    }

    auto device = sim::runtime::Db::GetOneByPred<sim::runtime::Device>(HcclSim::Storage::All<sim::runtime::Device>());
    if (!device.ok()) {
        HCCL_VM_ERROR("no Device record found in db");
        return HCCL_E_NOT_FOUND;
    }

    DevType queriedDevType = DevType::DEV_TYPE_COUNT;
    if (strncmp(device.value->soc_version, "Ascend950", strlen("Ascend950")) == 0) {
        queriedDevType = DevType::DEV_TYPE_950;
    } else if (strncmp(device.value->soc_version, "Ascend960", strlen("Ascend960")) == 0) {
        queriedDevType = DevType::DEV_TYPE_960;
    } else {
        HCCL_VM_ERROR("not support device soc version: {:s}", device.value->soc_version);
        return HCCL_E_NOT_SUPPORT;
    }

    cachedDevType = queriedDevType;
    devType = queriedDevType;
    HCCL_VM_TRACE("Get current device type: {}", static_cast<int>(devType));
    return HCCL_SUCCESS;
}

rtError_t rtOpenNetService(const rtNetServiceOpenArgs* args)
{
    (void)args;
    // hccpThreadStatus = 1;
    return ACL_SUCCESS;
}

rtError_t rtGetDeviceInfo(uint32_t deviceId, int32_t moduleType, int32_t infoType, int64_t* value)
{
    (void)deviceId;
    (void)moduleType;
    (void)infoType;
    (void)value;
    return RT_ERROR_NONE;
}

rtError_t rtUbDevQueryInfo(rtUbDevQueryCmd cmd, void* devInfo)
{
    (void)cmd;
    (void)devInfo;
    return RT_ERROR_NONE;
}

rtError_t rtCntNotifyCreateServer(rtCntNotify_t* const cntNotify, uint64_t flags)
{
    (void)cntNotify;
    (void)flags;
    return 0;
}

rtError_t rtsCntNotifyGetId(rtCntNotify_t cntNotify, uint32_t* notifyId)
{
    (void)cntNotify;
    (void)notifyId;
    return ACL_SUCCESS;
}

rtError_t rtGetDevResAddress(rtDevResInfo* const resInfo, rtDevResAddrInfo* const addrInfo)
{
    if (resInfo == nullptr
        || (resInfo->resType != rtDevResType_t::RT_RES_TYPE_STARS_NOTIFY_RECORD
            && resInfo->resType != rtDevResType_t::RT_RES_TYPE_STARS_CNT_NOTIFY_BIT_WR)) {
        // 非NotifyRecord场景暂不处理
        return RT_ERROR_NONE;
    }
    uint32_t len = 8;
    *(addrInfo->resAddress) = resInfo->resId;
    *(addrInfo->len) = len;
    return RT_ERROR_NONE;
}

HcclResult HcclSymWinGetPeerDevicePointer(HcclCommSymWindow winHandle, size_t offset, uint32_t peerMemberId, void** ptr)
{
    HCCL_VM_INFO(
        "HcclSymWinGetPeerDevicePointer: winHandle={:p}, offset={:d}, peerMemberId={:d}, ptr={:p}", winHandle, offset,
        peerMemberId, static_cast<void*>(ptr));
    (void)winHandle;
    (void)offset;
    (void)peerMemberId;
    // 仿真环境无 UB 对称窗口：按契约置空输出并返回 NOT_SUPPORT，让调用方回退 legacy 远端地址查询
    if (ptr != nullptr) {
        *ptr = nullptr;
    }
    return HCCL_E_NOT_SUPPORT;
}

rtError_t rtGetP2PStatus(uint32_t devIdDes, uint32_t phyIdSrc, uint32_t* status)
{
    *status = DRV_P2P_STATUS_ENABLE;
    return ACL_SUCCESS;
}

#ifdef __cplusplus
}
#endif // __cplusplus
