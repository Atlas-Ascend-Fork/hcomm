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
#define HCCL_VM_MODULE "DEVICE_STUB"

#include <atomic>
#include <csignal>
#include <cstdint>
#include <iostream>
#include <sys/types.h>
#include <unistd.h>
#include <cstring>
#include <dlfcn.h>
#include <strings.h>
#include <utility>
#include <vector>

#include "acl/acl_base.h"
#include "acl/acl_rt.h"
#include "dtype_common.h"
#include "platform/platform_info.h"
#include "operation_data/operation_data_ops.h"
#include "sim_common_macro.h"
#include "sim_log.h"
#include "sim_capacity_limits.h"
#include "sim_resource_limits.h"
#include "runtime/base.h"
#include "runtime_state/db_sim_runner_common.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "sim_sub_process_manager.h"
#include "hccl_proxy_common.h"
#include "sim_dpu_kernel_lib_mgr.h"
#include "storage/table_access.h"

// current host id
uint64_t g_host_id;
extern uint64_t g_cur_server_key;

extern pid_t g_devicePid;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// 说明：aclmdlRICaptureGetInfo 已迁移至 aclrt_graph_capture_stub.cc（aclGraph 第一阶段），
// 会话状态由 graph_capture 的 ModelRecord 体系提供，不再使用固定 NONE 假实现。

HcclResult hrtGetDeviceIndexByPhyId(uint32_t devicePhyId, uint32_t& deviceLogicId)
{
    try {
        auto ret = sim::runtime::Db::GetOneByPred<sim::runtime::Device>(HcclSim::Storage::And(
            HcclSim::Storage::Eq(&sim::runtime::Device::server_id, g_cur_server_key),
            HcclSim::Storage::Eq(&sim::runtime::Device::physical_id, (uint32_t)devicePhyId)));
        if (!ret.ok()) {
            HCCL_VM_ERROR("device not found by phyId:{:d}", devicePhyId);
            return HcclResult::HCCL_E_NOT_FOUND;
        }
        deviceLogicId = ret->logic_id;
        return HCCL_SUCCESS;
    } catch (const std::exception& e) {
        HCCL_VM_ERROR("exception:{}", e.what());
        return HcclResult::HCCL_E_INTERNAL;
    }
}

// 真机 ACL 层在 aclrtSetDevice 内通过 UpdatePlatformInfoWithDevice（acl_rt_impl_base.cpp:42-66）
// 调 fe::PlatformInfoManager::InitRuntimePlatformInfos 加载 platform_config ini，
static void InitRuntimePlatformInfoFromDevice(const sim::runtime::Device& device)
{
    // 仅在显式启用 PyTorch 时补调平台信息初始化。
    const char* enablePytorch = std::getenv("HCCL_VM_ENABLE_PYTORCH");
    if (enablePytorch == nullptr || std::strcmp(enablePytorch, "1") != 0) {
        HCCL_VM_INFO("HCCL_VM_ENABLE_PYTORCH not set, skip platform info init");
        return;
    }
    if (device.soc_version[0] == '\0') {
        HCCL_VM_WARN("device {:d} soc_version is empty, skip platform info init", device.id);
        return;
    }
    uint32_t ret = fe::PlatformInfoManager::GeInstance().InitRuntimePlatformInfos(device.soc_version);
    if (ret != 0U) {
        HCCL_VM_WARN("InitRuntimePlatformInfos({}) failed, ret={:d}", device.soc_version, ret);
        return;
    }
    HCCL_VM_INFO("InitRuntimePlatformInfos({}) success", device.soc_version);
}

aclError aclrtSetDevice(int32_t deviceId)
{
    try {
        HCCL_VM_DEBUG("set id:{:d}", deviceId);
        uint64_t serverKey = sim::runtime::GetCurServerId();
        if (serverKey == 0) {
            HCCL_VM_ERROR("GetCurServerKey failed");
            return ACL_ERROR_INVALID_PARAM;
        }

        uint32_t rankId = deviceId;
        uint64_t serverTmp = 0;
        if (!sim::runtime::GetRankIdByMPI(rankId, serverTmp)) {
            HCCL_VM_ERROR("get rankId by MPI fail serverKey:{:d}", serverTmp);
            return ACL_ERROR_INVALID_PARAM;
        }
        // MPI env未提供rank时(单server直跑python/mp.spawn)，GetRankIdByMPI返回true但不写rankId；
        // 按"deviceId == rankId"约定回退，否则0xFFFF会写入Rank表导致notify/collective拿不到合法rank
        if (rankId == 0xFFFF) {
            rankId = static_cast<uint32_t>(deviceId);
            HCCL_VM_WARN("rank not provided by MPI env, fallback to deviceId as rankId: {}", rankId);
        }

        if (rankId >= HcclSim::GetMaxRanks()) {
            HCCL_VM_ERROR(
                "rank size {} exceeds max ranks {}, set env HCCL_VM_MAX_RANKS larger", rankId, HcclSim::GetMaxRanks());
            return ACL_ERROR_INVALID_PARAM;
        }

        sim::runtime::Device device{};
        auto ret = sim::runtime::Db::GetOneByPred<sim::runtime::Device>(HcclSim::Storage::And(
            HcclSim::Storage::Eq(&sim::runtime::Device::server_id, serverKey),
            HcclSim::Storage::Eq(&sim::runtime::Device::logic_id, (uint32_t)deviceId)));
        if (!ret.ok()) {
            HCCL_VM_ERROR("device not found logicId:{:d} serverKey:{:d}", deviceId, serverKey);
            return ACL_ERROR_INVALID_PARAM;
        }
        device = *ret;
        sim::runtime::g_cur_device_key = device.id;

        InitRuntimePlatformInfoFromDevice(device);

        auto deviceKey = device.id;

        SetDevIdPayload payload{};
        payload.rankId = rankId;
        payload.deviceKey = deviceKey;
        uint8_t rspCmd;
        uint64_t rspPayload = 0xFF;
        uint32_t rspLen = 0;
        if (sim::GetAicpuProcMgr().IsAlive()) {
            if (sim::GetAicpuProcMgr().Request(
                    PIPE_CMD_SET_DEV_ID, &payload, sizeof(payload), rspCmd, &rspPayload, sizeof(rspPayload), rspLen)
                != 0) {
                HCCL_VM_ERROR("Request PIPE_CMD_SET_DEV_ID failed.");
                return ACL_ERROR_INVALID_PARAM;
            }
            HCCL_VM_INFO("device rank id: {:d}, deviceKey: {:d}, set to sub process", rankId, deviceKey);
        }

        sim::runtime::Runner runner{};
        if (!sim::runtime::GetCurrRunnerTls(serverKey, runner)) {
            return ACL_ERROR_INVALID_PARAM;
        }
        auto curRunnerId = runner.id;

        uint64_t currCtxId = 0;
        auto ctxRet = sim::runtime::Db::GetOneByPred<sim::runtime::Context>(HcclSim::Storage::And(
            HcclSim::Storage::Eq(&sim::runtime::Context::device_id, deviceKey),
            HcclSim::Storage::Eq(&sim::runtime::Context::is_default, 1)));
        if (!ctxRet.ok()) {
            sim::runtime::Context context{};
            context.device_id = device.id;
            context.run_id = curRunnerId;
            context.is_default = 1;
            context.ref_cnt = 1;
            context.id = 0;
            auto insertedContext = sim::runtime::Db::Add<sim::runtime::Context>(context);
            if (!insertedContext.ok() || !insertedContext.value.has_value()) {
                return ACL_ERROR_INVALID_PARAM;
            }
            currCtxId = insertedContext.value->value;

            sim::runtime::Stream stream{};
            stream.ctx_id = currCtxId;
            stream.activated = 1;
            stream.is_primary_default = 1;
            stream.id = 0;
            if (!sim::runtime::Db::Add<sim::runtime::Stream>(stream).ok()) {
                return ACL_ERROR_INVALID_PARAM;
            }
        } else {
            currCtxId = ctxRet->id;
            if (!sim::runtime::Db::Update<sim::runtime::Context>(
                     HcclSim::Storage::Eq(&sim::runtime::Context::id, currCtxId),
                     HcclSim::Storage::Increment(&sim::runtime::Context::ref_cnt, uint64_t{1}))
                     .ok()) {
                return ACL_ERROR_INVALID_PARAM;
            }
        }

        sim::runtime::SetCurrCtxTls(currCtxId);
        return ACL_SUCCESS;
    } catch (const std::exception& e) {
        HCCL_VM_ERROR("exception:{}", e.what());
        return ACL_ERROR_INTERNAL_ERROR;
    }
}

aclError aclrtResetDevice(int32_t deviceId)
{
    try {
        HCCL_VM_INFO("deviceId:{:d}", deviceId);
        auto serverId = sim::runtime::GetCurServerId();
        if (serverId == 0) {
            return ACL_ERROR_INVALID_PARAM;
        }
        sim::runtime::Runner runner{};
        if (!sim::runtime::GetCurrRunnerTls(serverId, runner)) {
            return ACL_ERROR_INVALID_PARAM;
        }
        auto curCtxId = runner.current_ctx_id;
        if (curCtxId == 0) {
            return ACL_SUCCESS;
        }
        auto currCtx = sim::runtime::Db::GetById<sim::runtime::Context>(curCtxId);
        if (!currCtx.ok()) {
            HCCL_VM_WARN("ctx not found:{:d}", runner.current_ctx_id);
            return ACL_SUCCESS;
        }

        auto streamRet = sim::runtime::Db::GetOneByPred<sim::runtime::Stream>(HcclSim::Storage::And(
            HcclSim::Storage::Eq(&sim::runtime::Stream::ctx_id, curCtxId),
            HcclSim::Storage::Eq(&sim::runtime::Stream::is_primary_default, 1)));
        if (!streamRet.ok()) {
            HCCL_VM_ERROR("stream not found ctxId:{:d}", curCtxId);
            return ACL_ERROR_INVALID_PARAM;
        }

        if (currCtx.value->ref_cnt > 1) {
            (void)sim::runtime::Db::Update<sim::runtime::Context>(
                HcclSim::Storage::Eq(&sim::runtime::Context::id, curCtxId),
                HcclSim::Storage::Increment(&sim::runtime::Context::ref_cnt, int64_t{-1}));
            return ACL_SUCCESS;
        }

        (void)sim::runtime::Db::Delete<sim::runtime::Stream>(
            HcclSim::Storage::Eq(&sim::runtime::Stream::id, streamRet->id));
        (void)sim::runtime::Db::Delete<sim::runtime::Context>(
            HcclSim::Storage::Eq(&sim::runtime::Context::id, curCtxId));

        curCtxId = 0;
        sim::runtime::SetCurrCtxTls(curCtxId);
        return ACL_SUCCESS;
    } catch (const std::exception& e) {
        HCCL_VM_ERROR("exception:{}", e.what());
        return ACL_ERROR_INTERNAL_ERROR;
    }
}

aclError aclrtResetDeviceForce(int32_t deviceId)
{
    try {
        HCCL_VM_DEBUG("stream not found deviceId:{:d}", deviceId);
        sim::runtime::Runner runner{};
        auto serverId = sim::runtime::GetCurServerId();
        if (serverId == 0) {
            return ACL_ERROR_INVALID_PARAM;
        }
        if (!sim::runtime::GetCurrRunnerTls(serverId, runner)) {
            return ACL_ERROR_INVALID_PARAM;
        }
        auto curCtxId = runner.current_ctx_id;
        if (curCtxId == 0) {
            return ACL_SUCCESS;
        }
        auto currCtx = sim::runtime::Db::GetById<sim::runtime::Context>(curCtxId);
        if (!currCtx.ok()) {
            HCCL_VM_WARN("ctx not found:{:d}", runner.current_ctx_id);
            return ACL_SUCCESS;
        }

        auto streamRet = sim::runtime::Db::GetOneByPred<sim::runtime::Stream>(HcclSim::Storage::And(
            HcclSim::Storage::Eq(&sim::runtime::Stream::ctx_id, curCtxId),
            HcclSim::Storage::Eq(&sim::runtime::Stream::is_primary_default, 1)));
        if (!streamRet.ok()) {
            HCCL_VM_ERROR("stream not found ctxId:{:d}", curCtxId);
            return ACL_ERROR_INVALID_PARAM;
        }

        (void)sim::runtime::Db::Delete<sim::runtime::Stream>(
            HcclSim::Storage::Eq(&sim::runtime::Stream::id, streamRet->id));
        (void)sim::runtime::Db::Delete<sim::runtime::Context>(
            HcclSim::Storage::Eq(&sim::runtime::Context::id, curCtxId));
        curCtxId = 0;
        sim::runtime::SetCurrCtxTls(curCtxId);
        return ACL_SUCCESS;
    } catch (const std::exception& e) {
        HCCL_VM_ERROR("exception:{}", e.what());
        return ACL_ERROR_INTERNAL_ERROR;
    }
}

aclError aclrtGetDevice(int32_t* device)
{
    try {
        sim::runtime::Runner runner{};
        auto serverId = sim::runtime::GetCurServerId();
        if (serverId == 0) {
            return ACL_ERROR_INVALID_PARAM;
        }
        if (!sim::runtime::GetCurrRunnerTls(serverId, runner)) {
            return ACL_ERROR_INVALID_PARAM;
        }
        auto currCtx = sim::runtime::Db::GetById<sim::runtime::Context>(runner.current_ctx_id);
        if (!currCtx.ok()) {
            // torch_npu 场景：未显式 set_device 时查询当前设备，返回默认设备
            // 从当前 server 中查找第一个可用设备的 logic_id
            auto devs = sim::runtime::Db::GetByPred<sim::runtime::Device>(
                HcclSim::Storage::Eq(&sim::runtime::Device::server_id, serverId));
            if (devs.ok() && !devs.value->empty()) {
                *device = static_cast<int32_t>(devs.value->front().logic_id);
                HCCL_VM_DEBUG("default device:{:d}", *device);
                return ACL_SUCCESS;
            }
            HCCL_VM_WARN("no device found in server, default to 0");
            *device = 0;
            return ACL_SUCCESS;
        }

        auto devRes = sim::runtime::Db::GetById<sim::runtime::Device>(currCtx.value->device_id);
        if (!devRes.ok()) {
            HCCL_VM_ERROR("device not found:{:d}", currCtx.value->device_id);
            return ACL_ERROR_INVALID_PARAM;
        }
        *device = devRes.value->logic_id;
        HCCL_VM_DEBUG("id:{:d}", devRes.value->logic_id);
        return ACL_SUCCESS;
    } catch (const std::exception& e) {
        HCCL_VM_ERROR("exception:{}", e.what());
        return ACL_ERROR_INTERNAL_ERROR;
    }
}

aclError aclrtGetRunMode(aclrtRunMode* runMode)
{
    *runMode = ACL_DEVICE;
    return ACL_SUCCESS;
}

aclError aclrtSetTsDevice(aclrtTsId tsId)
{
    try {
        if (tsId == ACL_TS_ID_AICORE) {
            sim::runtime::SetTsDevice(tsId);
        }

        sim::runtime::Runner runner{};
        auto serverId = sim::runtime::GetCurServerId();
        if (serverId == 0) {
            return ACL_ERROR_INVALID_PARAM;
        }
        if (!sim::runtime::GetCurrRunnerTls(serverId, runner)) {
            return ACL_ERROR_INVALID_PARAM;
        }
        auto currCtx = sim::runtime::Db::GetById<sim::runtime::Context>(runner.current_ctx_id);
        if (!currCtx.ok()) {
            HCCL_VM_ERROR("ctx not found:{:d}", runner.current_ctx_id);
            return ACL_ERROR_INVALID_PARAM;
        }

        auto devRes = sim::runtime::Db::GetById<sim::runtime::Device>(currCtx.value->device_id);
        if (!devRes.ok()) {
            HCCL_VM_ERROR("device not found:{:d}", currCtx.value->device_id);
            return ACL_ERROR_INVALID_PARAM;
        }

        uint32_t vectoCount = sim::runtime::GetVectorCoreCount(devRes.value->logic_id);
        if (vectoCount != 0) {
            sim::runtime::SetTsDevice(tsId);
        }

        return ACL_SUCCESS;
    } catch (const std::exception& e) {
        HCCL_VM_ERROR("exception:{}", e.what());
        return ACL_ERROR_INTERNAL_ERROR;
    }
}

aclError aclrtGetDeviceCount(uint32_t* count)
{
    try {
        if (g_cur_server_key != 0) {
            auto currServer = sim::runtime::Db::GetById<sim::runtime::Server>(g_cur_server_key);
            if (!currServer.ok()) {
                HCCL_VM_ERROR("server not found:{:d}", g_cur_server_key);
                return ACL_ERROR_INVALID_PARAM;
            }

            *count = currServer.value->used_dev_num;
            return ACL_SUCCESS;
        }

        uint64_t serverId = sim::runtime::GetCurServerId();
        if (serverId == 0) {
            HCCL_VM_ERROR("GetCurServerId failed");
            return ACL_ERROR_INVALID_PARAM;
        }

        auto currServer = sim::runtime::Db::GetById<sim::runtime::Server>(serverId);
        if (!currServer.ok()) {
            HCCL_VM_ERROR("server not found:{:d}", serverId);
            return ACL_ERROR_INVALID_PARAM;
        }

        if (currServer.value->used_dev_num == 0) {
            HCCL_VM_ERROR("used_devs_num of server {:d} is 0", serverId);
            return ACL_ERROR_INVALID_PARAM;
        }
        *count = currServer.value->used_dev_num;

        return ACL_SUCCESS;
    } catch (const std::exception& e) {
        HCCL_VM_ERROR("exception:{}", e.what());
        return ACL_ERROR_INTERNAL_ERROR;
    }
}

aclError aclrtGetDeviceUtilizationRate(int32_t deviceId, aclrtUtilizationInfo* utilizationInfo)
{
    try {
        auto ret = sim::runtime::Db::GetOneByPred<sim::runtime::Device>(
            HcclSim::Storage::Eq(&sim::runtime::Device::logic_id, deviceId));
        if (!ret.ok()) {
            HCCL_VM_ERROR("device not found by phyId:{:d}", deviceId);
            return 0;
        }

        utilizationInfo->cubeUtilization = 20;
        utilizationInfo->vectorUtilization = 20;
        utilizationInfo->aicpuUtilization = 20;
        utilizationInfo->memoryUtilization = 20;
        return ACL_SUCCESS;
    } catch (const std::exception& e) {
        HCCL_VM_ERROR("exception:{}", e.what());
        return ACL_ERROR_INTERNAL_ERROR;
    }
}

aclError aclrtQueryDeviceStatus(int32_t deviceId, aclrtDeviceStatus* deviceStatus)
{
    try {
        auto ret = sim::runtime::Db::GetOneByPred<sim::runtime::Device>(
            HcclSim::Storage::Eq(&sim::runtime::Device::logic_id, (uint32_t)deviceId));
        if (!ret.ok()) {
            HCCL_VM_ERROR("device not found logicId:{:d}", deviceId);
            return HcclResult::HCCL_E_NOT_FOUND;
        }
        *deviceStatus = (aclrtDeviceStatus)ret->status;
        return HcclVmResult::HCCL_SIM_SUCCESS;
    } catch (const std::exception& e) {
        HCCL_VM_ERROR("exception:{}", e.what());
        return ACL_ERROR_INTERNAL_ERROR;
    }
}

const char* aclrtGetSocName()
{
    try {
        // GetSocName接口根据获取server内任意一个device的soc_version
        auto devRes = sim::runtime::Db::GetOneByPred<sim::runtime::Device>(
            HcclSim::Storage::Eq(&sim::runtime::Device::server_id, 1));
        if (!devRes.ok()) {
            HCCL_VM_ERROR("device not found serverId:1");
            return "";
        }

        thread_local static char SocName[128] = {0};
        memcpy(SocName, devRes->soc_version, strlen(devRes->soc_version));
        SocName[strlen(devRes->soc_version)] = '\0';
        HCCL_VM_DEBUG("soc:{}", devRes->soc_version);
        return SocName;
    } catch (const std::exception& e) {
        HCCL_VM_ERROR("exception:{}", e.what());
        static thread_local char s_errMsg[] = "";
        return s_errMsg;
    }
}

aclError aclrtSetDeviceSatMode(aclrtFloatOverflowMode mode)
{
    try {
        sim::runtime::Runner runner{};
        auto serverId = sim::runtime::GetCurServerId();
        if (serverId == 0) {
            return ACL_ERROR_INVALID_PARAM;
        }
        if (!sim::runtime::GetCurrRunnerTls(serverId, runner)) {
            return ACL_ERROR_INVALID_PARAM;
        }
        auto currCtx = sim::runtime::Db::GetById<sim::runtime::Context>(runner.current_ctx_id);
        if (!currCtx.ok()) {
            HCCL_VM_ERROR("ctx not found:{:d}", runner.current_ctx_id);
            return ACL_ERROR_INVALID_PARAM;
        }

        auto curDevId = currCtx.value->device_id;
        if (!sim::runtime::Db::Update<sim::runtime::Device>(
                 HcclSim::Storage::Eq(&sim::runtime::Device::id, curDevId),
                 HcclSim::Storage::Set(&sim::runtime::Device::overflow_mode, mode))
                 .ok()) {
            return ACL_ERROR_INVALID_PARAM;
        }
        return ACL_SUCCESS;
    } catch (const std::exception& e) {
        HCCL_VM_ERROR("exception:{}", e.what());
        return ACL_ERROR_INTERNAL_ERROR;
    }
}

aclError aclrtGetDeviceSatMode(aclrtFloatOverflowMode* mode)
{
    try {
        sim::runtime::Runner runner{};
        auto serverId = sim::runtime::GetCurServerId();
        if (serverId == 0) {
            return ACL_ERROR_INVALID_PARAM;
        }
        if (!sim::runtime::GetCurrRunnerTls(serverId, runner)) {
            return ACL_ERROR_INVALID_PARAM;
        }
        auto currCtx = sim::runtime::Db::GetById<sim::runtime::Context>(runner.current_ctx_id);
        if (!currCtx.ok()) {
            HCCL_VM_ERROR("ctx not found:{:d}", runner.current_ctx_id);
            return ACL_ERROR_INVALID_PARAM;
        }

        auto dev = sim::runtime::Db::GetById<sim::runtime::Device>(currCtx.value->device_id);
        if (!dev.ok()) {
            HCCL_VM_ERROR("device not found:{:d}", currCtx.value->device_id);
            return ACL_ERROR_INVALID_PARAM;
        }
        *mode = (aclrtFloatOverflowMode)dev.value->overflow_mode;
        return ACL_SUCCESS;
    } catch (const std::exception& e) {
        HCCL_VM_ERROR("exception:{}", e.what());
        return ACL_ERROR_INTERNAL_ERROR;
    }
}

aclError aclrtDeviceCanAccessPeer(int32_t* canAccessPeer, int32_t deviceId, int32_t peerDeviceId)
{
    try {
        auto dev1 = sim::runtime::Db::GetOneByPred<sim::runtime::Device>(
            HcclSim::Storage::Eq(&sim::runtime::Device::logic_id, deviceId));
        if (!dev1.ok()) {
            HCCL_VM_ERROR("device not found logicId:{:d}", deviceId);
            return ACL_ERROR_INVALID_PARAM;
        }

        auto dev2 = sim::runtime::Db::GetOneByPred<sim::runtime::Device>(
            HcclSim::Storage::Eq(&sim::runtime::Device::logic_id, peerDeviceId));
        if (!dev2.ok()) {
            HCCL_VM_ERROR("device not found logicId:{:d}", peerDeviceId);
            return ACL_ERROR_INVALID_PARAM;
        }

        auto dev1Id = dev1->id;
        auto dev2Id = dev2->id;

        auto ret = sim::runtime::Db::GetOneByPred<sim::runtime::DeviceConnection>(HcclSim::Storage::And(
            HcclSim::Storage::Eq(&sim::runtime::DeviceConnection::src_dev_id, dev1Id),
            HcclSim::Storage::Eq(&sim::runtime::DeviceConnection::dst_dev_id, dev2Id)));
        if (!ret.ok()) {
            HCCL_VM_ERROR("connection not found src:{:d} dst:{:d}", dev1Id, dev2Id);
            return HcclResult::HCCL_E_NOT_FOUND;
        }

        *canAccessPeer = (int32_t)ret->access_by_remote;
        return HcclVmResult::HCCL_SIM_SUCCESS;
    } catch (const std::exception& e) {
        HCCL_VM_ERROR("exception:{}", e.what());
        return ACL_ERROR_INTERNAL_ERROR;
    }
}

aclError aclrtDeviceEnablePeerAccess(int32_t peerDeviceId, uint32_t flags)
{
    (void)flags;
    try {
        sim::runtime::Runner runner{};
        auto serverId = sim::runtime::GetCurServerId();
        if (serverId == 0) {
            return ACL_ERROR_INVALID_PARAM;
        }
        if (!sim::runtime::GetCurrRunnerTls(serverId, runner)) {
            return ACL_ERROR_INVALID_PARAM;
        }
        auto currCtx = sim::runtime::Db::GetById<sim::runtime::Context>(runner.current_ctx_id);
        if (!currCtx.ok()) {
            HCCL_VM_ERROR("ctx not found:{:d}", runner.current_ctx_id);
            return ACL_ERROR_INVALID_PARAM;
        }

        auto dev1 = sim::runtime::Db::GetById<sim::runtime::Device>(currCtx.value->device_id);
        if (!dev1.ok()) {
            HCCL_VM_ERROR("device not found:{:d}", currCtx.value->device_id);
            return ACL_ERROR_INVALID_PARAM;
        }

        if (dev1.value->logic_id == peerDeviceId) {
            HCCL_VM_ERROR("invalid peerId:{:d}", peerDeviceId);
            return ACL_ERROR_INVALID_PARAM;
        }

        auto dev2 = sim::runtime::Db::GetOneByPred<sim::runtime::Device>(
            HcclSim::Storage::Eq(&sim::runtime::Device::logic_id, peerDeviceId));
        if (!dev2.ok()) {
            HCCL_VM_ERROR("device not found logicId:{:d}", peerDeviceId);
            return ACL_ERROR_INVALID_PARAM;
        }

        auto dev1Id = dev1.value->id;
        auto dev2Id = dev2->id;
        auto ret = sim::runtime::Db::GetOneByPred<sim::runtime::DeviceConnection>(HcclSim::Storage::And(
            HcclSim::Storage::Eq(&sim::runtime::DeviceConnection::src_dev_id, dev1Id),
            HcclSim::Storage::Eq(&sim::runtime::DeviceConnection::dst_dev_id, dev2Id)));
        if (!ret.ok()) {
            HCCL_VM_ERROR("connection not found src:{:d} dst:{:d}", dev1Id, dev2Id);
            return HcclResult::HCCL_E_NOT_FOUND;
        }

        (void)sim::runtime::Db::Update<sim::runtime::DeviceConnection>(
            HcclSim::Storage::Eq(&sim::runtime::DeviceConnection::id, ret->id),
            HcclSim::Storage::Set(&sim::runtime::DeviceConnection::access_by_remote, 1));
        return ACL_SUCCESS;
    } catch (const std::exception& e) {
        HCCL_VM_ERROR("exception:{}", e.what());
        return ACL_ERROR_INTERNAL_ERROR;
    }
}

aclError aclrtDeviceDisablePeerAccess(int32_t peerDeviceId)
{
    try {
        sim::runtime::Runner runner{};
        auto serverId = sim::runtime::GetCurServerId();
        if (serverId == 0) {
            return ACL_ERROR_INVALID_PARAM;
        }
        if (!sim::runtime::GetCurrRunnerTls(serverId, runner)) {
            return ACL_ERROR_INVALID_PARAM;
        }
        auto currCtx = sim::runtime::Db::GetById<sim::runtime::Context>(runner.current_ctx_id);
        if (!currCtx.ok()) {
            HCCL_VM_ERROR("ctx not found:{:d}", runner.current_ctx_id);
            return ACL_ERROR_INVALID_PARAM;
        }

        auto dev1 = sim::runtime::Db::GetById<sim::runtime::Device>(currCtx.value->device_id);
        if (!dev1.ok()) {
            HCCL_VM_ERROR("device not found:{:d}", currCtx.value->device_id);
            return ACL_ERROR_INVALID_PARAM;
        }

        if (dev1.value->logic_id == peerDeviceId) {
            HCCL_VM_ERROR("invalid peerId:{:d}", peerDeviceId);
            return ACL_ERROR_INVALID_PARAM;
        }

        auto dev2 = sim::runtime::Db::GetOneByPred<sim::runtime::Device>(
            HcclSim::Storage::Eq(&sim::runtime::Device::logic_id, peerDeviceId));
        if (!dev2.ok()) {
            HCCL_VM_ERROR("device not found logicId:{:d}", peerDeviceId);
            return ACL_ERROR_INVALID_PARAM;
        }

        auto dev1Id = dev1.value->id;
        auto dev2Id = dev2->id;
        auto ret = sim::runtime::Db::GetOneByPred<sim::runtime::DeviceConnection>(HcclSim::Storage::And(
            HcclSim::Storage::Eq(&sim::runtime::DeviceConnection::src_dev_id, dev1Id),
            HcclSim::Storage::Eq(&sim::runtime::DeviceConnection::dst_dev_id, dev2Id)));
        if (!ret.ok()) {
            HCCL_VM_ERROR("connection not found src:{:d} dst:{:d}", dev1Id, dev2Id);
            return ACL_ERROR_INVALID_PARAM;
        }

        (void)sim::runtime::Db::Update<sim::runtime::DeviceConnection>(
            HcclSim::Storage::Eq(&sim::runtime::DeviceConnection::id, ret->id),
            HcclSim::Storage::Set(&sim::runtime::DeviceConnection::access_by_remote, 0));
        return ACL_SUCCESS;
    } catch (const std::exception& e) {
        HCCL_VM_ERROR("exception:{}", e.what());
        return ACL_ERROR_INTERNAL_ERROR;
    }
}

aclError aclrtGetOverflowStatus(void* outputAddr, size_t outputSize, aclrtStream stream)
{
    (void)outputSize;
    try {
        uint64_t streamIdx = (uint64_t)(uintptr_t)stream;
        auto stmRes = sim::runtime::Db::GetById<sim::runtime::Stream>(streamIdx);
        if (!stmRes.ok()) {
            HCCL_VM_ERROR("stream not found:{:d}", streamIdx);
            return ACL_ERROR_INVALID_PARAM;
        }

        auto ctxRes = sim::runtime::Db::GetById<sim::runtime::Context>(stmRes.value->ctx_id);
        if (!ctxRes.ok()) {
            HCCL_VM_ERROR("ctx not found:{:d}", stmRes.value->ctx_id);
            return ACL_ERROR_INVALID_PARAM;
        }

        auto deviceIdx = ctxRes.value->device_id;
        auto devStatusRes = sim::runtime::Db::GetOneByPred<sim::runtime::DeviceStatus>(
            HcclSim::Storage::Eq(&sim::runtime::DeviceStatus::device_id, deviceIdx));
        if (!devStatusRes.ok()) {
            HCCL_VM_ERROR("device not found:{:d}", deviceIdx);
            return ACL_ERROR_INVALID_PARAM;
        }

        uint8_t* tmp = (uint8_t*)outputAddr;
        *tmp = devStatusRes->overflow_status;

        return ACL_SUCCESS;
    } catch (const std::exception& e) {
        HCCL_VM_ERROR("exception:{}", e.what());
        return ACL_ERROR_INTERNAL_ERROR;
    }
}

aclError aclrtResetOverflowStatus(aclrtStream stream)
{
    try {
        uint64_t streamIdx = (uint64_t)(uintptr_t)stream;
        auto stmRes = sim::runtime::Db::GetById<sim::runtime::Stream>(streamIdx);
        if (!stmRes.ok()) {
            HCCL_VM_ERROR("stream not found:{:d}", streamIdx);
            return ACL_ERROR_INVALID_PARAM;
        }

        auto ctxRes = sim::runtime::Db::GetById<sim::runtime::Context>(stmRes.value->ctx_id);
        if (!ctxRes.ok()) {
            HCCL_VM_ERROR("ctx not found:{:d}", stmRes.value->ctx_id);
            return ACL_ERROR_INVALID_PARAM;
        }

        auto deviceIdx = ctxRes.value->device_id;
        auto devStatusRes = sim::runtime::Db::GetOneByPred<sim::runtime::DeviceStatus>(
            HcclSim::Storage::Eq(&sim::runtime::DeviceStatus::device_id, deviceIdx));
        if (!devStatusRes.ok()) {
            HCCL_VM_ERROR("device not found:{:d}", deviceIdx);
            return ACL_ERROR_INVALID_PARAM;
        }
        (void)sim::runtime::Db::Update<sim::runtime::DeviceStatus>(
            HcclSim::Storage::Eq(&sim::runtime::DeviceStatus::id, devStatusRes->id),
            HcclSim::Storage::Set(&sim::runtime::DeviceStatus::overflow_status, 0));
        return ACL_SUCCESS;
    } catch (const std::exception& e) {
        HCCL_VM_ERROR("exception:{}", e.what());
        return ACL_ERROR_INTERNAL_ERROR;
    }
}

aclError aclrtSynchronizeDevice(void) { return ACL_SUCCESS; }

aclError aclrtSynchronizeDeviceWithTimeout(int32_t timeout)
{
    (void)timeout;
    return ACL_SUCCESS;
}

aclError aclrtGetDeviceInfo(uint32_t deviceId, aclrtDevAttr attr, int64_t* value)
{
    uint32_t count = 0;
    if (attr == ACL_DEV_ATTR_AICPU_CORE_NUM) {
        count = sim::runtime::GetAICpuCount(deviceId);
    } else if (attr == ACL_DEV_ATTR_AICORE_CORE_NUM) {
        count = sim::runtime::GetAICoreCount(deviceId);
    } else if (attr == ACL_DEV_ATTR_VECTOR_CORE_NUM) {
        count = sim::runtime::GetVectorCoreCount(deviceId);
    } else if (attr == ACL_DEV_ATTR_DEVICE_FORM_FACTOR) {
        std::string hardwareType = sim::runtime::GetHardwareTypeByDevice(deviceId);
        bool isPod = (hardwareType.find("POD") != std::string::npos) || (hardwareType.find("pod") != std::string::npos);
        count = isPod ? ACL_DEVICE_FORM_FACTOR_POD : ACL_DEVICE_FORM_FACTOR_A_K;
    }
    *value = static_cast<int64_t>(count);
    return ACL_SUCCESS;
}

aclError aclrtDeviceGetStreamPriorityRange(int32_t* leastPriority, int32_t* greatestPriority)
{
    (void)leastPriority;
    (void)greatestPriority;
    return ACL_SUCCESS;
}

aclError aclrtGetDeviceCapability(int32_t deviceId, aclrtDevFeatureType devFeatureType, int32_t* value)
{
    (void)deviceId;
    (void)devFeatureType;
    (void)value;
    return ACL_SUCCESS;
}

aclError aclrtGetDevicesTopo(uint32_t deviceId, uint32_t otherDeviceId, uint64_t* value)
{
    try {
        auto dev1 = sim::runtime::Db::GetOneByPred<sim::runtime::Device>(
            HcclSim::Storage::Eq(&sim::runtime::Device::logic_id, deviceId));
        if (!dev1.ok()) {
            HCCL_VM_ERROR("device not found logicId:{:d}", deviceId);
            return ACL_ERROR_INVALID_PARAM;
        }

        auto dev2 = sim::runtime::Db::GetOneByPred<sim::runtime::Device>(
            HcclSim::Storage::Eq(&sim::runtime::Device::logic_id, otherDeviceId));
        if (!dev2.ok()) {
            HCCL_VM_ERROR("device not found logicId:{:d}", otherDeviceId);
            return ACL_ERROR_INVALID_PARAM;
        }

        auto dev1Id = dev1->id;
        auto dev2Id = dev2->id;

        auto ret = sim::runtime::Db::GetOneByPred<sim::runtime::DeviceConnection>(HcclSim::Storage::And(
            HcclSim::Storage::Eq(&sim::runtime::DeviceConnection::src_dev_id, dev1Id),
            HcclSim::Storage::Eq(&sim::runtime::DeviceConnection::dst_dev_id, dev2Id)));
        if (!ret.ok()) {
            HCCL_VM_ERROR("connection not found src:{:d} dst:{:d}", dev1Id, dev2Id);
            return HcclResult::HCCL_E_NOT_FOUND;
        }

        *value = (uint64_t)ret->link_type;
        return HcclVmResult::HCCL_SIM_SUCCESS;
    } catch (const std::exception& e) {
        HCCL_VM_ERROR("exception:{}", e.what());
        return ACL_ERROR_INTERNAL_ERROR;
    }
}

aclError aclrtDevicePeerAccessStatus(int32_t deviceId, int32_t peerDeviceId, int32_t* status)
{
    return aclrtDeviceCanAccessPeer(status, deviceId, peerDeviceId);
}

aclError aclInit(const char* configPath)
{
    HCCL_VM_INFO("-----[acl start]----------");
    // rank 进程在初始化边界各自把 RLIMIT_NOFILE/NPROC 软限抬到硬限：大规模建链
    // fd 需求远超默认软限 1024，触顶会让尾部 rank 建链失败（hcclRet 19）。不依赖
    // host/mpirun 的继承（双保险）；进程级、无需特权、退出即消失、不落系统。
    HcclSim::SimRaiseResourceLimits();
    // 已知初始化边界提前触发同一个进程 Session bootstrap（LD_PRELOAD rank 进程在
    // 任何数据库访问与 AICPU 子进程创建之前 fail-fast；幂等——会话已安装时直接成功）。
    // 主控进程安装的 session 对 rank 进程不可见。该边界不是唯一正确性依赖：即使
    // aclInit 未执行，本进程首次 Storage 原子/业务 API 调用仍经唯一 ready-session
    // 入口（ProcessStorageContext::AcquireReady + 已注册 simulator bootstrap）惰性
    // 建立本进程会话——等价迁移前 RunnerDB 惰性 singleton 的首次建连时机。
    if (sim::operation::EnsureProcessStorageSession() != 0) {
        HCCL_VM_ERROR("aclInit failed to install the process storage session.");
        return ACL_ERROR_INTERNAL_ERROR;
    }
    if (sim::IsAICPUExpMode()) {
        HCCL_VM_INFO("aclInit on AICPU mode.");
        auto config = sim::CreateAicpuDeviceConfig(0);
        if (sim::GetAicpuProcMgr().CreateProcess(config) != 0) {
            HCCL_VM_ERROR("failed to create device process.");
            exit(EXIT_FAILURE);
        }
    }

    return ACL_SUCCESS;
}

aclError aclFinalize()
{
    // aclFinalize 不是数据库最后使用边界：与基线一致只做日志、销毁 AICPU 子进程与
    // FlushLog——finalize 之后 HCCL 动态库的 Socket/Context/Device 等退出清理仍会
    // 访问数据库（蓝区证据：RaSocketListenStop can not get Socket、aclrtResetDevice
    // ctx not found）。进程 session 为 no-destroy process-lifetime，由本进程退出时
    // OS 统一回收；显式关闭只属于主控/device-arm/测试/换库等真实 owner。
    HCCL_VM_INFO("-----[acl finalize]----------");
    sim::GetAicpuProcMgr().DestroyProcess();
    sim::DpuKernelLibManager::GetInstance().Cleanup();
    FlushLog();
    return ACL_SUCCESS;
}

aclError aclrtGetPhyDevIdByLogicDevId(int32_t logicDevId, int32_t* const phyDevId)
{
    auto device = sim::runtime::Db::GetOneByPred<sim::runtime::Device>(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&sim::runtime::Device::server_id, g_cur_server_key),
        HcclSim::Storage::Eq(&sim::runtime::Device::logic_id, static_cast<uint32_t>(logicDevId))));
    if (!device.ok()) {
        HCCL_VM_ERROR("cannot find device by logical id {:d}", logicDevId);
        return ACL_ERROR_INVALID_PARAM;
    }

    *phyDevId = (int32_t)device->physical_id;
    HCCL_VM_DEBUG("server:{:d} logicId:{:d} phyId:{:d}", g_cur_server_key, logicDevId, *phyDevId);
    return ACL_SUCCESS;
}

aclError aclrtGetLogicDevIdByPhyDevId(const int32_t phyDevId, int32_t* const logicDevId)
{
    auto device = sim::runtime::Db::GetOneByPred<sim::runtime::Device>(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&sim::runtime::Device::server_id, g_cur_server_key),
        HcclSim::Storage::Eq(&sim::runtime::Device::physical_id, static_cast<uint32_t>(phyDevId))));
    if (!device.ok()) {
        HCCL_VM_ERROR("cannot find device by physical id {:d}", phyDevId);
        return ACL_ERROR_INVALID_PARAM;
    }

    *logicDevId = (int32_t)device->logic_id;
    return ACL_SUCCESS;
}

aclError aclrtSetDeviceTaskAbortCallback(const char* regName, aclrtDeviceTaskAbortCallback callback, void* args)
{
    (void)regName;
    (void)callback;
    (void)args;
    return ACL_SUCCESS;
}

rtError_t rtGetDevicePhyIdByIndex(uint32_t devIndex, uint32_t* phyId)
{
    auto device = sim::runtime::Db::GetOneByPred<sim::runtime::Device>(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&sim::runtime::Device::server_id, g_cur_server_key),
        HcclSim::Storage::Eq(&sim::runtime::Device::logic_id, devIndex)));
    if (!device.ok()) {
        HCCL_VM_ERROR("cannot find device by logical id {:d}", devIndex);
        return (rtError_t)ACL_ERROR_INVALID_PARAM;
    }
    *phyId = device->physical_id;
    return ACL_SUCCESS;
}

rtError_t rtGetPhyDeviceInfo(uint32_t phyId, int32_t moduleType, int32_t infoType, int64_t* val)
{
    (void)phyId;
    (void)moduleType;
    (void)infoType;
    (void)val;
    return ACL_SUCCESS;
}

rtError_t rtGetDeviceIndexByPhyId(uint32_t phyId, uint32_t* devIndex)
{
    try {
        auto ret = sim::runtime::Db::GetOneByPred<sim::runtime::Device>(
            HcclSim::Storage::Eq(&sim::runtime::Device::physical_id, phyId));
        if (!ret.ok()) {
            HCCL_VM_ERROR("device not found by phyId:{:d}", phyId);
            return HcclResult::HCCL_E_NOT_FOUND;
        }
        *devIndex = ret->logic_id;
        return ACL_SUCCESS;
    } catch (const std::exception& e) {
        HCCL_VM_ERROR("exception:{}", e.what());
        return ACL_ERROR_INTERNAL_ERROR;
    }
}

rtError_t rtSetDevice(int32_t devId) { return aclrtSetDevice(devId); }

rtError_t rtGetPairPhyDevicesInfo(uint32_t devId, uint32_t otherDevId, int32_t infoType, int64_t* val)
{
    (void)devId;
    (void)otherDevId;
    (void)infoType;
    *val = 1;
    return ACL_SUCCESS;
}

rtError_t rtsGetLogicDevIdByPhyDevId(int32_t phyDevId, int32_t* const logicDevId)
{
    return aclrtGetLogicDevIdByPhyDevId(phyDevId, logicDevId);
}

struct rtDevResInfo;
rtError_t rtReleaseDevResAddress(rtDevResInfo* const resInfo)
{
    (void)resInfo;
    return ACL_SUCCESS;
}

aclError aclrtGetLogicDevIdByUserDevId(const int32_t userDevid, int32_t* const logicDevId)
{
    *logicDevId = userDevid;
    return ACL_SUCCESS;
}

aclError aclrtRegDeviceStateCallbackImpl(const char* regName, aclrtDeviceStateCallback callback, void* args)
{
    HCCL_VM_WARN("Enter aclrtRegDeviceStateCallbackImpl, regName: {:p}", (void*)regName);

    (void)regName;
    (void)callback;
    (void)args;
    return ACL_SUCCESS;
}

// --- XPU Device API stubs (for hostdpu mode) ---
// rtXpuDevType 枚举值见 rts_device.h: RT_DEV_TYPE_DPU = 0, RT_DEV_TYPE_REV = 1
rtError_t rtSetXpuDevice(uint32_t devType, const uint32_t devId)
{
    HCCL_VM_INFO("rtSetXpuDevice devType={}, devId={}", devType, devId);
    if (devType != 0) {
        HCCL_VM_WARN("rtSetXpuDevice: unsupported devType={}, currently only DPU(0) is supported", devType);
    }
    if (devId != 0) {
        HCCL_VM_WARN("rtSetXpuDevice: currently devId=0 is supported, got devId={}", devId);
    }
    return RT_ERROR_NONE;
}

rtError_t rtResetXpuDevice(uint32_t devType, const uint32_t devId)
{
    HCCL_VM_INFO("rtResetXpuDevice devType={}, devId={}", devType, devId);
    return RT_ERROR_NONE;
}

rtError_t rtGetXpuDevCount(uint32_t devType, uint32_t* devCount)
{
    HCCL_VM_INFO("rtGetXpuDevCount devType={}", devType);
    if (devCount == nullptr) {
        HCCL_VM_ERROR("rtGetXpuDevCount: devCount is null");
        return ACL_ERROR_INVALID_PARAM;
    }
    *devCount = 1;
    return RT_ERROR_NONE;
}

#ifdef __cplusplus
}
#endif // __cplusplus
