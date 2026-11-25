/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccp_hdc_manager.h"
#include "exception_util.h"
#include "internal_exception.h"
#include "orion_adapter_tsd.h"
#include "orion_adapter_rts.h"

namespace Hccl {

HccpHdcManager& HccpHdcManager::GetInstance()
{
    // Leaky Singleton: 故意不释放，规避 Static Destruction Order Fiasco，
    // hixl全局对象析构访问 BaseCommMgr 时会调ra相关接口，而HccpHdcManager析构时会销毁Ra资源，
    // 导致ra接口不可用，因此需确保Ra资源尚未Deinit
    static HccpHdcManager* hccpHdcManager = new HccpHdcManager();
    return *hccpHdcManager;
}

void HccpHdcManager::Init(u32 deviceLogicId)
{
    std::unique_lock<std::recursive_mutex> lock(managerMutex);
    if (instances.count(deviceLogicId) != 0) {
        return;
    }

    HrtOpenTsdProcess(deviceLogicId);

    HRaInitConfig cfg;
    DevId phyIdOut = 0;
    CHK_RET_THROW(
        InternalException, "[HccpHdcManager::Init] HrtGetDevicePhyIdByUserDevId failed",
        HrtGetDevicePhyIdByUserDevId(static_cast<s32>(deviceLogicId), phyIdOut));
    cfg.phyId = static_cast<u32>(phyIdOut);
    cfg.mode = HrtNetworkMode::HDC;
    HrtRaInit(cfg);

    instances.insert(deviceLogicId);
}

void HccpHdcManager::DeInit(u32 deviceLogicId)
{
    if (destroyed.load()) {
        return;
    }
    HCCL_INFO("[HccpHdcManager::%s] deviceLogicId [%u]", __func__, deviceLogicId);

    std::lock_guard<std::recursive_mutex> lock(managerMutex);
    if (instances.count(deviceLogicId) == 0) {
        HCCL_WARNING("[HccpHdcManager::%s] deviceLogicId[%u] not ra init", __func__, deviceLogicId);
        return;
    }
    instances.erase(deviceLogicId);

    HRaInitConfig cfg;
    DevId phyIdOut = 0;
    CHK_RET_THROW(
        InternalException, "[HccpHdcManager::DeInit] HrtGetDevicePhyIdByUserDevId failed",
        HrtGetDevicePhyIdByUserDevId(static_cast<s32>(deviceLogicId), phyIdOut));
    cfg.phyId = static_cast<u32>(phyIdOut);
    cfg.mode = HrtNetworkMode::HDC;
    DECTOR_TRY_CATCH("HccpHdcManager", HrtRaDeInit(cfg));
    HCCL_INFO("[HccpHdcManager::%s] devLogicId [%u] ra deinit success.", __func__, deviceLogicId);
}

void HccpHdcManager::DestroyAll()
{
    if (destroyed.load()) {
        return;
    }
    destroyed.store(true);

    // 析构前先注销reset device的回调，防止嵌套调用
    UnregisterDeviceResetCallback();

    std::lock_guard<std::recursive_mutex> lock(managerMutex);
    for (auto deviceLogicId : instances) {
        HCCL_INFO("HccpHdcManager deinit");

        HRaInitConfig cfg;
        DevId phyIdOut = 0;
        CHK_RET_THROW(
            InternalException, "[HccpHdcManager::DestroyAll] HrtGetDevicePhyIdByUserDevId failed",
            HrtGetDevicePhyIdByUserDevId(static_cast<s32>(deviceLogicId), phyIdOut));
        cfg.phyId = static_cast<u32>(phyIdOut);
        cfg.mode = HrtNetworkMode::HDC;
        DECTOR_TRY_CATCH("HccpHdcManager", HrtRaDeInit(cfg));
        HcclResult resetRet = HrtResetDevice(deviceLogicId);
        if (resetRet != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[HccpHdcManager] HrtResetDevice failed, deviceLogicId[%u], ret[%d].", static_cast<u32>(deviceLogicId),
                resetRet);
        }
    }
    instances.clear();
}

void HccpHdcManager::UnregisterDeviceResetCallback() const
{
    aclError ret = aclrtRegDeviceStateCallback("hcomm_res_mgr", nullptr, nullptr);
    if (ret != ACL_SUCCESS) {
        HCCL_WARNING("[HccpHdcManager] aclrtRegDeviceStateCallback unregister is unsuccessful, ret[%d]", ret);
        return;
    }
    HCCL_INFO("[HccpHdcManager] aclrtRegDeviceStateCallback unregister success");
}

HccpHdcManager::~HccpHdcManager() { DECTOR_TRY_CATCH("HccpHdcManager", DestroyAll()); }

} // namespace Hccl
