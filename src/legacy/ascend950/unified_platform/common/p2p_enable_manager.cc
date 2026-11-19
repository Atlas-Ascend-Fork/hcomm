/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "p2p_enable_manager.h"
#include "hccl_exception.h"
#include "log.h"

namespace Hccl {

P2PEnableManager& P2PEnableManager::GetInstance()
{
    static P2PEnableManager p2pEnableManager;
    return p2pEnableManager;
}

HcclResult P2PEnableManager::EnableP2P(std::vector<uint32_t> remoteDevices)
{
    s32 localDeviceLogicID = 0;
    CHK_RET(HrtGetDevice(localDeviceLogicID));

    for (auto remoteDevicePhysicID : remoteDevices) {
        CHK_RET(EnableP2P(localDeviceLogicID, remoteDevicePhysicID));
    }
    return HCCL_SUCCESS;
}

HcclResult P2PEnableManager::EnableP2P(uint32_t localDeviceLogicID, uint32_t remoteDevicePhysicID)
{
    std::unique_lock<std::mutex> lock(connectionsLock_[localDeviceLogicID]);
    auto& iterLocalDevice = connectionsInfo_[localDeviceLogicID];
    auto iterRemoteDevice = iterLocalDevice.find(remoteDevicePhysicID);
    if ((iterRemoteDevice == iterLocalDevice.end()) || (iterRemoteDevice->second.reference.load() == 0)) {
        DevId localDevicePhysicIDOut = 0;
        CHK_RET(HrtGetDevicePhyIdByUserDevId(static_cast<s32>(localDeviceLogicID), localDevicePhysicIDOut));
        auto localDevicePhysicID = static_cast<uint32_t>(localDevicePhysicIDOut);
        CHK_RET(HrtEnableP2P(localDeviceLogicID, remoteDevicePhysicID));
        HCCL_INFO(
            "[EnableP2P]enable p2p: local logic id:%u, local physic id:%u, remote physic id:%u.", localDeviceLogicID,
            localDevicePhysicID, remoteDevicePhysicID);
        iterLocalDevice[remoteDevicePhysicID].status = P2PConnStatus::P2P_CONN_STATUS_ENABLING;
        iterLocalDevice[remoteDevicePhysicID].reference.fetch_add(1);
        return HCCL_SUCCESS;
    } else {
        // 使已执行过 enable，且未执行过 disable，不重复执行 enable p2p。
        iterLocalDevice[remoteDevicePhysicID].reference.fetch_add(1);
    }
    return HCCL_SUCCESS;
}

HcclResult P2PEnableManager::WaitP2PEnabled(uint32_t localDeviceLogicID, uint32_t remoteDevicePhysicID, bool& isEnabled)
{
    // 出参入口统一置 false：保证所有返回路径（未使能/错误返回）均为确定值，仅在确认使能时置 true
    isEnabled = false;

    std::unique_lock<std::mutex> lock(connectionsLock_[localDeviceLogicID]);
    auto& iterLocalDevice = connectionsInfo_[localDeviceLogicID];
    auto iterRemoteDevice = iterLocalDevice.find(remoteDevicePhysicID);
    bool bErr = (iterRemoteDevice == iterLocalDevice.end()) || (iterRemoteDevice->second.reference.load() == 0)
                || (iterRemoteDevice->second.status == P2PConnStatus::P2P_CONN_STATUS_DISABLED);
    CHK_PRT_RET(
        bErr,
        HCCL_ERROR(
            "[Wait][P2PEnabled]wait p2p enabled failed. enable operation has not been executed, "
            "ret[%u]. device info: local logic id:%u, remote physic id:%u.",
            HCCL_E_INTERNAL, localDeviceLogicID, remoteDevicePhysicID),
        HCCL_E_INTERNAL);

    // 缓存复用：已使能过的链路直接返回，无需重复查询驱动
    if (iterRemoteDevice->second.status == P2PConnStatus::P2P_CONN_STATUS_ENABLED) {
        isEnabled = true;
        return HCCL_SUCCESS;
    }

    // 非阻塞单次查询：不再通过 while 循环阻塞等待，未使能时由调用方下次轮询
    uint32_t status = DRV_P2P_STATUS_DISABLE;
    CHK_RET(HrtGetP2PStatus(localDeviceLogicID, remoteDevicePhysicID, &status));
    if (status == DRV_P2P_STATUS_ENABLE) {
        iterLocalDevice[remoteDevicePhysicID].status = P2PConnStatus::P2P_CONN_STATUS_ENABLED;
        isEnabled = true;
        HCCL_INFO(
            "connected p2p success. device info: local logic id:%u, remote physic id:%u.", localDeviceLogicID,
            remoteDevicePhysicID);
    }
    return HCCL_SUCCESS;
}

HcclResult P2PEnableManager::DisableP2P(uint32_t localDeviceLogicID, std::vector<uint32_t> remoteDevices)
{
    try {
        for (auto& remoteDevicePhysicID : remoteDevices) {
            CHK_RET(DisableP2P(localDeviceLogicID, remoteDevicePhysicID));
        }
    } catch (HcclException& e) {
        HCCL_ERROR("%s", e.what());
        return e.GetErrorCode();
    } catch (...) {
        HCCL_ERROR("Unknown error occurs!");
        return HcclResult::HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult P2PEnableManager::DisableP2P(uint32_t localDeviceLogicID, uint32_t remoteDevicePhysicID)
{
    std::unique_lock<std::mutex> lock(connectionsLock_[localDeviceLogicID]);
    auto& iterLocalDevice = connectionsInfo_[localDeviceLogicID];
    auto iterRemoteDevice = iterLocalDevice.find(remoteDevicePhysicID);
    if ((iterRemoteDevice == iterLocalDevice.end()) || (iterRemoteDevice->second.reference.load() == 0)) {
        HCCL_WARNING(
            "there is no p2p connections, no need to disable p2p. "
            "device info: local logic id:%u, remote physic id:%u.",
            localDeviceLogicID, remoteDevicePhysicID);
        return HCCL_SUCCESS;
    }

    iterRemoteDevice->second.reference.fetch_sub(1);
    if (iterRemoteDevice->second.reference.load() == 0) {
        DevId localDevicePhysicIDOut = 0;
        CHK_RET(HrtGetDevicePhyIdByUserDevId(static_cast<s32>(localDeviceLogicID), localDevicePhysicIDOut));
        auto localDevicePhysicID = static_cast<uint32_t>(localDevicePhysicIDOut);
        HCCL_INFO(
            "disable p2p: local logic id:%u, local physic id:%u, remote physic id:%u.", localDeviceLogicID,
            localDevicePhysicID, remoteDevicePhysicID);
        CHK_RET(HrtDisableP2P(localDeviceLogicID, remoteDevicePhysicID));
        iterLocalDevice[remoteDevicePhysicID].status = P2PConnStatus::P2P_CONN_STATUS_DISABLED;
    }
    return HCCL_SUCCESS;
}

P2PEnableManager::~P2PEnableManager() {}

} // namespace Hccl
