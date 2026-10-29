/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "base_comm_mgr.h"

#include <mutex>

#include "acl/acl_rt.h"
#include "hccl_common.h"
#include "adapter_rts_common.h"
#include "base_comm_res_per_dev.h"
#include "resources/comm_engine_res/comm_engine_procs.h"
#include "common/dlsym_proc.h"
#include "config_mgr/config_mgr_proc.h"
#include "dfx/dfx_procs.h"
#include "legacy950_res_mgr_proc.h"
#include "resources/endpoints/net_dev/endpoint_proc_mgr.h"
#include "handle_registry_procs.h"
#include "resources/endpoints/mgr/endpoint_mgr.h"
#include "base_comm_node_accessors.h"
#include "socket_mgr.h"
#include "server_socket_mgr.h"

#include "hccp_hdc_manager.h"
#include "rdma_handle_manager.h"
#include "socket_handle_manager.h"
#include "hcomm_adapter_hccp.h"

namespace hcomm {

BaseCommMgr& BaseCommMgr::GetInstance()
{
    static BaseCommMgr instance;
    return instance;
}

namespace {
    std::atomic<bool>& TeardownFlag()
    {
        static std::atomic<bool> flag{false}; // 平凡可析构，树死后读取安全
        return flag;
    }
} // namespace

bool IsResourceTreeTeardownFinished() { return TeardownFlag().load(std::memory_order_acquire); }

BaseCommMgr::TeardownFlagGuard::~TeardownFlagGuard() { TeardownFlag().store(true, std::memory_order_release); }

BaseCommMgr::BaseCommMgr() = default;

BaseCommMgr::~BaseCommMgr()
{
    // 兜底：未显式 DeInit 时走同一动作链（幂等）
    if (!isDeInited_) {
        (void)DeInit(ExitProfile::EXIT_PROFILE_GRACEFUL);
    }
    // 走到析构仍有残留 endpoint 不符合预期（RESET 下为已知限制：其成员析构仍会触设备），告警暴露
    if (endpointMgr_.GetEndpointHandleMgr().HasLeftover()) {
        if (lastDeinitWasReset_) {
            HCCL_RUN_WARNING(
                "[BaseCommMgr][%s] leftover endpoints destruct after EXIT_PROFILE_RESET DeInit, "
                "device-side release in endpoint destructors may touch an unavailable device.",
                __func__);
        } else {
            // graceful 下 DeInit 已清表，仍有残留属异常（如 DeInit 后新建漏网），告警暴露
            HCCL_RUN_WARNING(
                "[BaseCommMgr][%s] leftover endpoints after graceful DeInit, "
                "released at member destruction; check for post-DeInit creation.",
                __func__);
        }
    }
    if (handleRegistryMgrProc_.GetChannelHandleRegistryProc().HasLeftover()) {
        if (lastDeinitWasReset_) {
            HCCL_RUN_WARNING(
                "[BaseCommMgr][%s] leftover channels destruct after EXIT_PROFILE_RESET DeInit, "
                "device-side release in channel destructors may touch an unavailable device.",
                __func__);
        } else {
            HCCL_RUN_WARNING(
                "[BaseCommMgr][%s] leftover channels after graceful DeInit, "
                "released at member destruction; check for post-DeInit creation.",
                __func__);
        }
    }
}

// ===== 设备状态回调 =====
// regName 字符串（"hcomm_res_mgr"/"hcomm_refresh_device"）是 ACL 侧按名注销的配对键，严禁改名
static std::mutex g_deviceResetRegMutex;
static std::mutex g_deviceRefreshRegMutex;
static std::mutex g_deviceRefreshMutex;
static bool g_deviceRefreshCallbackRegistered = false;
static bool g_deviceResetCallbackRegistered = false;

static void OnDeviceResetPre(int32_t deviceId, aclrtDeviceState state, [[maybe_unused]] void* args)
{
    try {
        if (state != ACL_RT_DEVICE_STATE_RESET_PRE) {
            return;
        }
        if (IsResourceTreeTeardownFinished()) {
            return; // 树已亡：禁止再经窄访问器触树（R10）
        }
        HCCL_INFO("[%s] deviceId[%d] state[%d] ", __func__, deviceId, static_cast<int>(state));

        u32 devPhyId = 0;
        HcclResult ret = hrtGetDevicePhyIdByIndex(static_cast<u32>(deviceId), devPhyId);
        if (ret != HCCL_SUCCESS) {
            HCCL_WARNING("[%s] hrtGetDevicePhyIdByIndex failed, deviceId[%d] ret[%d]", __func__, deviceId, ret);
            return;
        }
        // 先让快照账本失效（按设备清空全部记账），再销毁句柄：
        // 消除「账本可读但句柄已销毁」窗口，窗口内到达的 end 因账本为空直接返回，不触碰已销毁句柄
        HccpResetDfxSnapshot(devPhyId);
        SocketMgr::DeInit(devPhyId);
        GetServerSocketMgr().DeInit(devPhyId);
        Hccl::RdmaHandleManager::GetInstance().DeInit(devPhyId);
        Hccl::SocketHandleManager::GetInstance().DeInit(devPhyId);
        Hccl::HccpHdcManager::GetInstance().DeInit(deviceId);
    } catch (const std::exception& e) {
        HCCL_WARNING("[%s] exception caught:%s", __func__, e.what());
    } catch (...) {
        HCCL_WARNING("[%s] unknown exception caught", __func__);
    }
}

void BaseCommMgr::RegisterDeviceResetCallback()
{
    std::lock_guard<std::mutex> lock(g_deviceResetRegMutex);
    if (g_deviceResetCallbackRegistered) {
        return;
    }
    aclError ret = aclrtRegDeviceStateCallback("hcomm_res_mgr", OnDeviceResetPre, nullptr);
    if (ret != ACL_SUCCESS) {
        HCCL_WARNING("[RegisterDeviceResetCallback] aclrtRegDeviceStateCallback failed, ret[%d]", ret);
        return;
    }
    g_deviceResetCallbackRegistered = true;
    HCCL_INFO("[%s] aclrtRegDeviceStateCallback success", __func__);
}

static void OnDeviceStateRefresh(int32_t deviceId, aclrtDeviceState state, [[maybe_unused]] void* args)
{
    std::lock_guard<std::mutex> lock(g_deviceRefreshMutex);
    try {
        if (state != ACL_RT_DEVICE_STATE_SET_POST) {
            return;
        }
        HCCL_INFO("[%s] deviceId[%d] state[%d]", __func__, deviceId, static_cast<int>(state));
        s32 deviceLogicId = 0;
        HcclResult ret = hrtGetDeviceRefresh(&deviceLogicId);
        if (ret != HCCL_SUCCESS) {
            HCCL_WARNING("[%s] hrtGetDeviceRefresh failed, deviceId[%d] ret[%d]", __func__, deviceId, ret);
            return;
        }

        u32 devicePhyId = 0;
        ret = hrtGetDevicePhyIdByIndex(static_cast<u32>(deviceLogicId), devicePhyId, true);
        if (ret != HCCL_SUCCESS) {
            HCCL_WARNING("[%s] hrtGetDevicePhyIdByIndex failed, deviceId[%d] ret[%d]", __func__, deviceId, ret);
            return;
        }

        DevType deviceType = DevType::DEV_TYPE_COUNT;
        ret = hrtGetDeviceType(deviceType);
        if (ret != HCCL_SUCCESS) {
            HCCL_WARNING("[%s] hrtGetDeviceType failed, deviceId[%d] ret[%d]", __func__, deviceId, ret);
            return;
        }
        HCCL_INFO(
            "[%s] refresh success, deviceLogicId[%d] devicePhyId[%d], deviceType[%d]", __func__, deviceLogicId,
            devicePhyId, static_cast<int>(deviceType));
    } catch (const std::exception& e) {
        HCCL_WARNING("[%s] exception caught:%s", __func__, e.what());
    } catch (...) {
        HCCL_WARNING("[%s] unknown exception caught", __func__);
    }
}

void BaseCommMgr::RegisterDeviceRefreshCallback()
{
    std::lock_guard<std::mutex> lock(g_deviceRefreshRegMutex);
    if (g_deviceRefreshCallbackRegistered) {
        return;
    }
    aclError ret = aclrtRegDeviceStateCallback("hcomm_refresh_device", OnDeviceStateRefresh, nullptr);
    if (ret != ACL_SUCCESS) {
        HCCL_WARNING("[%s] aclrtRegDeviceStateCallback failed, ret[%d]", __func__, ret);
        return;
    }
    g_deviceRefreshCallbackRegistered = true;
    HCCL_INFO("[%s] aclrtRegDeviceStateCallback success, regName[%s]", __func__, "hcomm_refresh_device");
}

void BaseCommMgr::UnregisterDeviceRefreshCallback()
{
    // 与 Register 同一把锁串行化：标志检查、ACL 注销、标志复位对并发注册原子
    std::lock_guard<std::mutex> lock(g_deviceRefreshRegMutex);
    if (!g_deviceRefreshCallbackRegistered) {
        return; // 未注册过：不调 ACL、不打误导日志
    }
    aclError ret = aclrtRegDeviceStateCallback("hcomm_refresh_device", nullptr, nullptr);
    if (ret != ACL_SUCCESS) {
        HCCL_WARNING(
            "[%s] aclrtRegDeviceStateCallback unregister failed, "
            "regName[%s] ret[%d]",
            __func__, "hcomm_refresh_device", ret);
    }
    g_deviceRefreshCallbackRegistered = false;
    HCCL_INFO("[%s] unregister success", __func__);
}

BaseCommResPerDev& BaseCommMgr::GetBaseCommResPerDev(uint32_t devPhyId) { return perDevResMgr_.FindOrCreate(devPhyId); }

// 显式动作路径：按序清理、幂等；RESET 跳过触设备动作（双 profile 见 exit_profile.h，顺序依据见 README R4）
HcclResult BaseCommMgr::DeInit(ExitProfile profile)
{
    perDevResMgr_.MarkDeinited();     // 先锁拒建闸：任何 DeInit 发起即禁止新建子树
    if (isDeInited_.exchange(true)) { // 幂等：并发下仅一方执行
        return HCCL_SUCCESS;
    }
    lastDeinitWasReset_ = (profile == ExitProfile::EXIT_PROFILE_RESET);

    // 步骤标记日志：exit_path 用例按序 grep 校验动作链顺序（R4 契约的可观测出口）
    // [1] 停 Thread
    HCCL_INFO("[BaseCommMgr][DeInit] step[1/9] stop threads "
              "(placeholder: threads die with handle registry)");
    for (auto* perDev : perDevResMgr_.SnapshotActive()) {
        (void)perDev->GetCommEngineMgrProc().StopThreads(profile);
    }
    // [2] 清 Endpoint
    HCCL_INFO("[BaseCommMgr][DeInit] step[2/9] clear endpoints");
    endpointMgr_.DeInit(profile);
    // [3] 清残留 Channel：channel 析构经窄访问器归还 socket/notify，须在引擎族收尾 [5] 前执行
    HCCL_INFO("[BaseCommMgr][DeInit] step[3/9] clear leftover channels");
    handleRegistryMgrProc_.DeInit(profile);
    // [4] 清 EndpointCtxMgr 残留
    HCCL_INFO("[BaseCommMgr][DeInit] step[4/9] clear endpoint ctx");
    for (auto* perDev : perDevResMgr_.SnapshotActive()) {
        perDev->DeInit(profile);
    }
    // [5] 引擎族收尾
    HCCL_INFO("[BaseCommMgr][DeInit] step[5/9] comm engine cleanup");
    for (auto* perDev : perDevResMgr_.SnapshotActive()) {
        perDev->GetCommEngineMgrProc().DeInit(profile);
    }
    // [6] 卸 bin（占位：无卸载逻辑，bin 句柄随进程退出由 OS 回收）
    if (profile == ExitProfile::EXIT_PROFILE_GRACEFUL) {
        HCCL_INFO("[BaseCommMgr][DeInit] step[6/9] unload bin");
        commEngineBinMgrProc_.DeInit();
    } else {
        HCCL_INFO("[BaseCommMgr][DeInit] step[6/9] unload bin skipped (reset)");
    }
    // [7] 进程级网络资源
    HCCL_INFO("[BaseCommMgr][DeInit] step[7/9] network base cleanup "
              "(no-op: device-side release deferred to destructor)");
    endpointProcMgr_.DeInit(profile);
    // [8] 基础设施 + 注销 refresh 回调并复位注册标志（reset 回调不注销）
    HCCL_INFO("[BaseCommMgr][DeInit] step[8/9] infra cleanup and unregister callbacks");
    configMgrProc_.DeInit();
    dfxProc_.DeInit();
    dlsymProc_.DeInit();
    legacy950ResMgrProc_.DeInit();
    UnregisterDeviceRefreshCallback(); // 锁内注销 + 复位标志
    // reset 回调不注销（ACL 按名配对，进程退出即失效），仅持锁复位标志
    {
        std::lock_guard<std::mutex> lock(g_deviceResetRegMutex);
        g_deviceResetCallbackRegistered = false;
    }

    // [9] 摘表移交退役区
    HCCL_INFO("[BaseCommMgr][DeInit] step[9/9] retire device table");
    perDevResMgr_.RetireAll();
    return HCCL_SUCCESS;
}

} // namespace hcomm
