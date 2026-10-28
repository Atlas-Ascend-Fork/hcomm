/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "base_comm_node_accessors.h"

#include "hccl_common.h"
#include "adapter_rts_common.h"

namespace hcomm {

EidInfoMgr& GetEidInfoMgr(uint32_t devPhyId)
{
    if (devPhyId >= MAX_MODULE_DEVICE_NUM) {
        HCCL_WARNING(
            "[EidInfoMgr][%s] use the backup device, devPhyId[%u] should be "
            "less than %u.",
            __func__, devPhyId, MAX_MODULE_DEVICE_NUM);
        devPhyId = MAX_MODULE_DEVICE_NUM; // 使用备份设备
    }

    auto& eidInfoMgr = BaseCommMgr::GetInstance().GetEndpointMgr().GetEidInfoMgrs()[devPhyId];
    eidInfoMgr.SetDevPhyId(devPhyId);
    return eidInfoMgr;
}

EndpointCtxMgr& GetEndpointCtxMgr(uint32_t devPhyId)
{
    // 0~65 均合法（65 备用卡静默：基线 GetDeviceResMgr 对显式 65 逐次告警，本 PR 有意改静默，
    // 65 是合法备用槽、合法使用逐次告警为噪音）；>65 钳位 65 并告警
    if (devPhyId > MAX_MODULE_DEVICE_NUM) {
        HCCL_WARNING(
            "[EndpointCtxMgr][%s] devPhyId[%u] out of range, clamp to backup device %u.", __func__, devPhyId,
            MAX_MODULE_DEVICE_NUM);
        devPhyId = MAX_MODULE_DEVICE_NUM;
    }
    return BaseCommMgr::GetInstance().GetBaseCommResPerDev(devPhyId).GetEndpointCtxMgr();
}

SocketMgr& GetSocketMgr(int32_t devPhyId)
{
    if (static_cast<u32>(devPhyId) >= MAX_MODULE_DEVICE_NUM) {
        HCCL_WARNING(
            "[SocketMgr] devicePhyId >= MAX_MODULE_DEVICE_NUM, devicePhyId=%d, MAX_MODULE_DEVICE_NUM=%d", devPhyId,
            MAX_MODULE_DEVICE_NUM);
        // 越界回落 device 0 子树实例，不回写 devicePhyId_
        return BaseCommMgr::GetInstance().GetBaseCommResPerDev(0).GetServerSocketMgrProc().GetSocketMgr();
    }
    auto& inst = BaseCommMgr::GetInstance()
                     .GetBaseCommResPerDev(static_cast<u32>(devPhyId))
                     .GetServerSocketMgrProc()
                     .GetSocketMgr();
    inst.SetDevPhyId(devPhyId);
    return inst;
}

// 设备级访问器：logicId→phyId，越界/转换失败回落 0 号设备（同基线 GetInstance 槽位 0 语义）
SocketProcess& GetSocketProcess(int32_t deviceLogicId)
{
    uint32_t devPhyId = 0;
    if (deviceLogicId < 0 || static_cast<uint32_t>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM
        || hrtTryGetDevicePhyIdByIndex(static_cast<uint32_t>(deviceLogicId), devPhyId) != HCCL_SUCCESS) {
        HCCL_WARNING("[SocketProcess][%s] invalid deviceLogicId: %d, fallback to device 0", __func__, deviceLogicId);
        devPhyId = 0;
    }
    return BaseCommMgr::GetInstance().GetBaseCommResPerDev(devPhyId).GetSocketProcess();
}

hccl::GlobalNetDevMgr& GetGlobalNetDevMgr(uint32_t devPhyId)
{
    if (devPhyId >= MAX_MODULE_DEVICE_NUM + 1) {
        HCCL_RUN_WARNING("[GetGlobalNetDevMgr] devicePhyId[%u] is invalid, use the backup device", devPhyId);
        devPhyId = MAX_MODULE_DEVICE_NUM;
    }

    auto& mgr = BaseCommMgr::GetInstance().GetBaseCommResPerDev(devPhyId).GetNetDevProc().GetNetDev();
    u32 deviceLogicId = 0;
    // 探测性回落路径：用 quiet 转换，避免 legacy 内部对预期失败刷 ERROR（同 base_comm_res_per_dev.cc 探测写法）
    HcclResult hcclRet = hrtTryGetDeviceIndexByPhyId(devPhyId, deviceLogicId);
    if (hcclRet != HCCL_SUCCESS) {
        HCCL_RUN_WARNING(
            "[GetGlobalNetDevMgr] hrtTryGetDeviceIndexByPhyId failed, ret[%d], "
            "return uninitialized instance",
            hcclRet);
        return mgr;
    }

    if (deviceLogicId >= MAX_MODULE_DEVICE_NUM) {
        HCCL_RUN_WARNING(
            "[GetGlobalNetDevMgr] deviceLogicId[%u] is invalid, return uninitialized instance", deviceLogicId);
        return mgr;
    }

    hcclRet = mgr.EnsureInited(devPhyId, deviceLogicId);
    if (hcclRet != HCCL_SUCCESS) {
        HCCL_RUN_WARNING(
            "[GetGlobalNetDevMgr] Init deviceLogicId[%u] fail, return uninitialized instance", deviceLogicId);
        return mgr;
    }

    HCCL_DEBUG("[GetGlobalNetDevMgr] deviceLogicId[%u], devicePhyId[%u] done.", deviceLogicId, devPhyId);
    return mgr;
}

// CCU 族设备级访问器公共定位逻辑：越界钳位+告警、logicId→phyId 转换，
// 越界或转换失败回落共享 reserve 兜底实例（不按设备隔离，正常路径不触达）；
// id 成员由 owner 在 Init 期设定，访问器不回写（共享 reserve 并发回写 = 数据竞争）
template <typename MgrT>
static MgrT& GetCcuSlotMgr(
    int32_t deviceLogicId, const char* tag, const char* func, MgrT& (*reserveGet)(), MgrT& (CcuInstanceProc::*devGet)())
{
    int32_t userDevId = deviceLogicId;
    if (userDevId < 0 || static_cast<uint32_t>(userDevId) >= MAX_MODULE_DEVICE_NUM) {
        HCCL_WARNING(
            "[%s][%s] use the backup device, userDevId[%d] should be "
            "less than %u.",
            tag, func, userDevId, MAX_MODULE_DEVICE_NUM);
        userDevId = MAX_MODULE_DEVICE_NUM; // 使用备份设备
    }

    uint32_t devPhyId = 0;
    const bool convertFailed
        = (userDevId != MAX_MODULE_DEVICE_NUM)
          && (hrtTryGetDevicePhyIdByIndex(static_cast<uint32_t>(userDevId), devPhyId) != HCCL_SUCCESS);
    if (convertFailed) {
        HCCL_ERROR("[%s][%s] userDevId[%d] to phyId failed, fallback to shared reserve instance", tag, func, userDevId);
    }
    if (userDevId == MAX_MODULE_DEVICE_NUM || convertFailed) {
        return reserveGet(); // reserve 从不 Init、id 保持默认 -1：不再回写共享实例
    }

    auto& proc = BaseCommMgr::GetInstance().GetBaseCommResPerDev(devPhyId).GetCommEngineMgrProc().GetCcuInstanceProc();
    return (proc.*devGet)();
}

CcuKernelMgr& GetCcuKernelMgr(int32_t deviceLogicId)
{
    return GetCcuSlotMgr<CcuKernelMgr>(
        deviceLogicId, "CcuKernelMgr", __func__, &CcuInstanceProc::GetReserveCcuKernelMgr,
        &CcuInstanceProc::GetCcuKernelMgr);
}

CcuInstanceMgr& GetCcuInstanceMgr(int32_t deviceLogicId)
{
    return GetCcuSlotMgr<CcuInstanceMgr>(
        deviceLogicId, "CcuInstanceMgr", __func__, &CcuInstanceProc::GetReserveCcuInstanceMgr,
        &CcuInstanceProc::GetCcuInstanceMgr);
}

CcuVarEventResMgr& GetCcuVarEventResMgr(int32_t deviceLogicId)
{
    return GetCcuSlotMgr<CcuVarEventResMgr>(
        deviceLogicId, "CcuVarEventResMgr", __func__, &CcuInstanceProc::GetReserveCcuVarEventResMgr,
        &CcuInstanceProc::GetCcuVarEventResMgr);
}

TpMgr& GetTpMgr(uint32_t devPhyId)
{
    if (devPhyId >= MAX_MODULE_DEVICE_NUM) {
        HCCL_WARNING(
            "[TpMgr][%s] use the backup device, devPhyId[%u] should be "
            "less than %u.",
            __func__, devPhyId, MAX_MODULE_DEVICE_NUM);
        devPhyId = MAX_MODULE_DEVICE_NUM; // 使用备份设备
    }

    auto& tpMgr = BaseCommMgr::GetInstance()
                      .GetBaseCommResPerDev(devPhyId)
                      .GetCommEngineMgrProc()
                      .GetCcuInstanceProc()
                      .GetTpMgr();
    tpMgr.SetDevPhyId(devPhyId);

    return tpMgr;
}

HccpTlvHdcMgr& GetHccpTlvHdcMgr(uint32_t devPhyId)
{
    if (devPhyId >= MAX_MODULE_DEVICE_NUM) {
        HCCL_WARNING(
            "[HccpTlvHdcMgr][%s] use the backup device, devPhyId[%u] should be "
            "less than %u.",
            __func__, devPhyId, MAX_MODULE_DEVICE_NUM);
        devPhyId = MAX_MODULE_DEVICE_NUM; // 使用备份设备
    }

    auto& hccpTlvHdcMgr = BaseCommMgr::GetInstance()
                              .GetBaseCommResPerDev(devPhyId)
                              .GetCommEngineMgrProc()
                              .GetHdcProc()
                              .GetHccpTlvHdcMgr();
    hccpTlvHdcMgr.SetDevPhyId(devPhyId);
    return hccpTlvHdcMgr;
}

} // namespace hcomm
