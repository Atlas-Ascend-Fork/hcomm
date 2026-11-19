/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/cast_utils_pub.h"
#include "orion_adapter_rts.h"
#include "hccl_common_v2.h"
#include "log.h"
#include "config_plf_log_v2.h"
#include "acl/acl_rt.h"
#include "driver/ascend_hal.h"
#include "adapter_error_manager_pub.h"
#include "dlrts_function_v2.h"
#include "dtype_common.h"
#include <map>

using namespace std;
namespace Hccl {

HcclResult HrtThreadExchangeCaptureMode(aclmdlRICaptureMode* mode);
constexpr u32 TOKEN_ID_RIGHT_SHIF = 8; // URMA_TOKEN_ID_RIGHT_SHIF，因URMA配置原因需要右移8位
namespace {
    constexpr char RT_SET_XPU_DEVICE[] = "rtSetXpuDevice";
    constexpr char RT_RESET_XPU_DEVICE[] = "rtResetXpuDevice";

    // 全局枚举(::DevType) → 域内枚举(Hccl::DevType) 映射：两套枚举命名与序号不完全一致
    // （如 950(6→5)、960(8→6)、NOSOC(5→7)），需逐项显式映射
    // 全局 DEV_TYPE_MC62 在域内无对应类型，返回 INVALID 由调用方按非法值处理
    DevType MapToHcclDevType(::DevType globalDevType)
    {
        switch (globalDevType) {
            case ::DevType::DEV_TYPE_910:
                return DevType::DEV_TYPE_910A;
            case ::DevType::DEV_TYPE_310P3:
                return DevType::DEV_TYPE_V51_310_P3;
            // 全局枚举不区分裸 Ascend910B 与 910B1~B4, 统一按 910A2 处理; 此前按 socName 查表时
            // 裸 "Ascend910B" 被映射为 910A, 与同族的 B1~B4 不一致, 属本次修复的语义变化
            case ::DevType::DEV_TYPE_910B:
                return DevType::DEV_TYPE_910A2;
            case ::DevType::DEV_TYPE_310P1:
                return DevType::DEV_TYPE_V51_310_P1;
            case ::DevType::DEV_TYPE_910_93:
                return DevType::DEV_TYPE_910A3;
            case ::DevType::DEV_TYPE_950:
                return DevType::DEV_TYPE_950;
            case ::DevType::DEV_TYPE_960:
                return DevType::DEV_TYPE_960;
            case ::DevType::DEV_TYPE_NOSOC:
                return DevType::DEV_TYPE_NOSOC;
            default:
                return DevType::INVALID;
        }
    }
} // namespace

// 统一走全局 hrtGetDeviceType 取数（含 HCCD 场景 workModeAicpu/localDeviceType 分支与结果缓存），
// 再映射回域内枚举；失败返回错误码以保持本接口既有语义
HcclResult HrtGetDeviceType(DevType& devType)
{
    ::DevType globalDevType = ::DevType::DEV_TYPE_COUNT;
    HcclResult ret = ::hrtGetDeviceType(globalDevType);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[Get][DeviceType]errNo[0x%016llx] hrtGetDeviceType failed, return[%d].", HCCL_ERROR_CODE(ret),
            static_cast<int>(ret));
        return HCCL_E_RUNTIME;
    }

    devType = MapToHcclDevType(globalDevType);
    if (devType == DevType::INVALID) {
        HCCL_ERROR(
            "[Get][DeviceType]errNo[0x%016llx] unsupported global devType[%d].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), static_cast<int>(globalDevType));
        return HCCL_E_RUNTIME;
    }
    HCCL_INFO("[HrtGetDeviceType]DeviceType = %s.", devType.Describe().c_str());
    return HCCL_SUCCESS;
}

HcclResult HrtGetDevicePhyIdByUserDevId(s32 userDevId, DevId& devicePhyIdOut)
{
    DevType deviceType = DevType::INVALID;
    HcclResult ret = HrtGetDeviceType(deviceType);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }
    if (deviceType == DevType::DEV_TYPE_NOSOC) {
        devicePhyIdOut = 0;
        return HCCL_SUCCESS;
    }

    s32 devicePhyId = 0;
    aclError aclRet = aclrtGetPhyDevIdByUserDevId(userDevId, &devicePhyId);
    if (aclRet != ACL_SUCCESS) {
        HCCL_ERROR(
            "[Get][DevicePhyId]errNo[0x%016llx] rtGet device PhyId by "
            "userDevId failed. return[%d], para: userDevId[%d], phyId[%d].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_DRV), aclRet, userDevId, devicePhyId);
        return HCCL_E_RUNTIME;
    }
    HCCL_INFO("[HrtGetDevicePhyIdByUserDevId]userDevId=%d, devicePhyId=%d.", userDevId, devicePhyId);
    devicePhyIdOut = static_cast<DevId>(devicePhyId);
    return HCCL_SUCCESS;
}

HcclResult HrtGetUserDevIdByPhyDevId(s32 devicePhyId, s32& userDevId)
{
    DevType deviceType = DevType::INVALID;
    HcclResult ret = HrtGetDeviceType(deviceType);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }
    if (deviceType == DevType::DEV_TYPE_NOSOC) {
        userDevId = 0;
        return HCCL_SUCCESS;
    }

    s32 tmpUserDevId = 0;
    aclError aclRet = aclrtGetUserDevIdByPhyDevId(devicePhyId, &tmpUserDevId);
    if (aclRet != ACL_SUCCESS) {
        HCCL_ERROR(
            "[Get][UserDevId]errNo[0x%016llx] rtGet userDevId by PhyDevId failed. "
            "return[%d], para: phyId[%d], userDevId[%d].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), aclRet, devicePhyId, tmpUserDevId);
        return HCCL_E_RUNTIME;
    }
    userDevId = tmpUserDevId;
    HCCL_INFO("[HrtGetUserDevIdByPhyDevId]devicePhyId=%d, userDevId=%d.", devicePhyId, userDevId);
    return HCCL_SUCCESS;
}

HcclResult HrtDeviceGetBareTgid(s32& pid)
{
    pid = 0;
    aclError ret = aclrtDeviceGetBareTgid(&pid);
    HCCL_INFO("Call rtDeviceGetBareTgid, return value[%d], rtGet pid[%d].", ret, pid);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR(
            "[Get][BareTgid]errNo[0x%016llx] rtGet pid fail. return[%d], rtGet pid[%d].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, pid);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

HcclResult HrtGetSocVer(std::string& socName)
{
    const char* socNamePtr = aclrtGetSocName();
    if (socNamePtr == nullptr) {
        HCCL_ERROR(
            "[Get][SocVer]errNo[0x%016llx] rtGet deviceVer failed.", HCCL_ERROR_CODE((HcclResult::HCCL_E_RUNTIME)));
        return HCCL_E_RUNTIME;
    }
    socName = socNamePtr;
    return HCCL_SUCCESS;
}

HcclResult HrtGetDevice(s32& userDevId)
{
    userDevId = 0;
    aclError ret = aclrtGetDevice(&userDevId);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR(
            "[Get][Device]errNo[0x%016llx] rtGet device fail, "
            "please make sure that device is set. return[%d], para:userDevId[%d]",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, userDevId);
        return HCCL_E_RUNTIME;
    }
    HCCL_INFO("[HrtGetDevice]userDevId=%d.", userDevId);
    return HCCL_SUCCESS;
}

HcclResult HrtSetDevice(s32 deviceLogicId)
{
    aclError ret = aclrtSetDevice(deviceLogicId);
    HCCL_INFO("Call rtSetDevice, return value[%d], para: device_id[%d].", ret, deviceLogicId);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR(
            "[Set][Device]errNo[0x%016llx] rtSet device fail. "
            "return[%d], para:userDevId[%d].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, deviceLogicId);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

HcclResult HrtResetDevice(s32 deviceLogicId)
{
    aclError ret = aclrtResetDevice(deviceLogicId);
    HCCL_INFO("Call aclrtResetDevice, return value[%d], para: device_id[%d].", ret, deviceLogicId);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR(
            "[Reset][Device]errNo[0x%016llx] rtReset device fail. "
            "return[%d], para: userDevId[%d].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, deviceLogicId);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

HcclResult HrtGetDeviceCount(u32& count)
{
    count = 0;
    aclError ret = aclrtGetDeviceCount(&count);
    HCCL_INFO("Call rtGetDeviceCount, return value[%d], para: count[%u].", ret, count);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR(
            "[Get][DeviceCount]errNo[0x%016llx] rtGet device count fail. "
            "return[%d], para:count[%u].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, count);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

HcclResult HrtGetLogicDevIdByUserDevId(s32 userDevId, s32& logicDevId)
{
    DevType deviceType = DevType::INVALID;
    HcclResult ret = HrtGetDeviceType(deviceType);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }
    if (deviceType == DevType::DEV_TYPE_NOSOC) {
        logicDevId = 0;
        return HCCL_SUCCESS;
    }

    aclError aclRet = aclrtGetLogicDevIdByUserDevId(userDevId, &logicDevId); // userDevId 转 logicDevId
    if (aclRet != ACL_SUCCESS) {
        HCCL_ERROR("aclrtGetLogicDevIdByUserDevId failed, userDevId: %d, ret: %d", userDevId, aclRet);
        return HCCL_E_RUNTIME;
    }

    return HCCL_SUCCESS;
}

constexpr char RTS_SO_NAME[] = "libruntime.so";
DlRtsFunctionV2<RTS_SO_NAME> g_dlRts;
HcclResult HrtResetXpuDevice(uint32_t devType, const uint32_t devId)
{
    static auto funcPtr = ReinterpretAs<rtError_t (*)(uint32_t, const uint32_t)>(g_dlRts.Handle<RT_RESET_XPU_DEVICE>());
    CHK_PTR_NULL(funcPtr);
    rtError_t ret = funcPtr(devType, devId);
    if (ret != RT_ERROR_NONE) {
        HCCL_ERROR("[%s] reset xpu device failed, devType[%u], devId[%u], return[%d].", __func__, devType, devId, ret);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

HcclResult HrtSetXpuDevice(uint32_t devType, const uint32_t devId)
{
    static auto funcPtr = ReinterpretAs<rtError_t (*)(uint32_t, const uint32_t)>(g_dlRts.Handle<RT_SET_XPU_DEVICE>());
    CHK_PTR_NULL(funcPtr);
    rtError_t ret = funcPtr(devType, devId);
    if (ret != RT_ERROR_NONE) {
        HCCL_ERROR("[%s] set xpu device failed, devType[%u], devId[%u], return[%d].", __func__, devType, devId, ret);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

HcclResult HrtGetStreamId(aclrtStream ptr, s32& streamId)
{
    streamId = 0;
    aclError ret = aclrtStreamGetId(ptr, &streamId);
    HCCL_INFO("Call aclrtStreamGetId, ptr[%p] return value[%d] streamId[%d].", ptr, ret, streamId);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR(
            "[Get][StreamId]errNo[0x%016llx]. rt get stream ID fail. ptr[%p], return[%d].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ptr, ret);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

HcclResult HrtStreamGetMode(HcclRtStream const ptr, u64& stmMode)
{
    if (ptr == nullptr) {
        HCCL_ERROR("ptr is null, call aclrtGetStreamAttribute failed, ptr=%p", ptr);
        return HCCL_E_PTR;
    }
    stmMode = 0;
    s32 streamId = -1;
    aclError ret = aclrtStreamGetId(ptr, &streamId);
    HCCL_DEBUG("[HrtStreamGetMode] ptr[%p], ret[%d].", ptr, ret);
    aclrtStreamAttrValue value;
    ret = aclrtGetStreamAttribute(ptr, ACL_STREAM_ATTR_FAILURE_MODE, &value);
    stmMode = value.failureMode;
    HCCL_INFO("Call rtStreamGetMode return value[%d]. stmMode[%llu].", ret, stmMode);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR(
            "[Stream][GetMode]errNo[0x%016llx] rtStreamGetMode error. "
            "ptr[%p], stmMode[%llu], streamId[%d], rtRet[%d].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ptr, stmMode, streamId, ret);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

HcclResult HrtStreamSetMode(HcclRtStream streamPtr, const uint64_t stmMode)
{
    if (streamPtr == nullptr) {
        HCCL_ERROR("ptr is null, call aclrtSetStreamAttribute failed, ptr=%p", streamPtr);
        return HCCL_E_PTR;
    }
    s32 streamId = -1;
    aclError ret = aclrtStreamGetId(streamPtr, &streamId);
    HCCL_DEBUG("Call aclrtStreamGetId, return value[%d].", ret);
    aclrtStreamAttrValue value;
    value.failureMode = stmMode;
    ret = aclrtSetStreamAttribute(streamPtr, ACL_STREAM_ATTR_FAILURE_MODE, &value);
    HCCL_INFO("[HrtStreamSetMode]streamPtr[%p], stmMode[%llu], ret[%d].", streamPtr, stmMode, ret);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR(
            "[Stream][SetMode]errNo[0x%016llx] rtStreamSetMode error. "
            "streamPtr[%p], rtRet[%d], stmMode[%llu].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), streamPtr, ret, stmMode);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

HcclResult HrtGetDeviceInfo(uint32_t deviceLogicId, int32_t moduleType, aclrtDevAttr infoType, int64_t& val)
{
    if (moduleType != DEV_MODULE_TYPE::MODULE_TYPE_SYSTEM) {
        HCCL_ERROR("[hrtGetDeviceInfo]Unsupported moduleType[%d].", moduleType);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    aclError ret = aclrtGetDeviceInfo(deviceLogicId, infoType, ReinterpretAs<int64_t*>(&val));
    HCCL_INFO(
        "[HrtGetDeviceInfo]deviceLogicId[%u], moduleType[%d], infoType[%d], return[%d], val[%lld].", deviceLogicId,
        moduleType, infoType, ret, val);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR(
            "[HrtGetDeviceInfo]errNo[0x%016llx] rt get device info failed, "
            "deviceLogicId=%u, moduleType=%d, infoType=%d",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), deviceLogicId, moduleType, infoType);
        return HcclResult::HCCL_E_RUNTIME;
    }
    return HcclResult::HCCL_SUCCESS;
}

constexpr uint64_t POD_MAINBOARD = 0x0;
constexpr uint64_t A_K_SERVER_MAINBOARD = 0x1;
constexpr uint64_t A_X_SERVER_MAINBOARD = 0x2;
constexpr uint64_t PCIE_STD_MAINBOARD = 0x3;
constexpr uint64_t RSV1_MAINBOARD = 0x4;
constexpr uint64_t RSV2_MAINBOARD = 0x5;
constexpr uint64_t EQUIP_MAINBOARD = 0x6;
constexpr uint64_t EVB_MAINBOARD = 0x7;

const std::unordered_map<uint64_t, HcclMainboardId> rtMainboardIdToHcclMainboardId
    = {{POD_MAINBOARD, HcclMainboardId::MAINBOARD_POD},
       {A_K_SERVER_MAINBOARD, HcclMainboardId::MAINBOARD_A_K_SERVER},
       {A_X_SERVER_MAINBOARD, HcclMainboardId::MAINBOARD_A_X_SERVER},
       {PCIE_STD_MAINBOARD, HcclMainboardId::MAINBOARD_PCIE_STD},
       {RSV1_MAINBOARD, HcclMainboardId::MAINBOARD_RSV},
       {RSV2_MAINBOARD, HcclMainboardId::MAINBOARD_RSV},
       {EQUIP_MAINBOARD, HcclMainboardId::MAINBOARD_EQUIPMENT},
       {EVB_MAINBOARD, HcclMainboardId::MAINBOARD_EVB}};

/*
 * 获取Mainboard ID 5-7位，输出整机形态枚举值
 * Mainboard ID描述说明
 * Mainboard ID采用了16bit，区分形态，主从，以及端口配置
 * bit[7:5] 区分整机形态(当前POD和EVB没有区分A+X或A+K)
 * {
 *  000: 天成 POD
 *  001: A+K Server
 *  010: A+X Server
 *  011: PCIE标卡
 *  100-101: RSV
 *  110: 装备
 *  111: EVB
 * }
 * bit[4:1] 整机形态细分
 * {
 *  0000-1111
 * }
 * bit[0] 主从或池化
 * {
 *  0: 主从（NPU作为某个Host的从设备，Host主控）
 *  1: 池化（NPU作为资源池，其它Host对等访问）
 * }
 */
HcclResult HrtGetMainboardId(uint32_t deviceLogicId, HcclMainboardId& hcclMainboardId)
{
    constexpr int32_t moduleType = DEV_MODULE_TYPE::MODULE_TYPE_SYSTEM;
    constexpr aclrtDevAttr infoType = aclrtDevAttr::ACL_DEV_ATTR_MAINBOARD_ID;
    constexpr uint64_t BITS_5 = 5;
    constexpr uint64_t MASK_7 = 0x7;
    int64_t val = 0;
    CHK_RET(HrtGetDeviceInfo(deviceLogicId, moduleType, infoType, val));
    HCCL_INFO("[HrtGetMainboardId] deviceLogicId[%u] val[%lld].", deviceLogicId, val);
    CHK_PRT_RET(val < 0, HCCL_ERROR("[HrtGetMainboardId]val[%lld] < 0", val), HCCL_E_RUNTIME);
    uint64_t mainboardId = (static_cast<uint64_t>(val) >> BITS_5) & MASK_7; // 提取val的5-7位，判断整机形态
    auto it = rtMainboardIdToHcclMainboardId.find(mainboardId);
    if (it != rtMainboardIdToHcclMainboardId.end()) {
        hcclMainboardId = it->second;
    } else {
        hcclMainboardId = HcclMainboardId::MAINBOARD_OTHERS;
    }
    HCCL_INFO(
        "[HrtGetMainboardId] deviceLogicId[%u] mainboardId[%llu] hcclMainboardId[%s].", deviceLogicId, mainboardId,
        hcclMainboardId.Describe().c_str());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult HrtStreamCreateWithFlags(uint32_t priority, uint32_t flag, aclrtStream& ptr)
{
    ptr = nullptr;
    aclError ret = aclrtCreateStreamWithConfig(&ptr, priority, flag);
    PLF_CONFIG_INFO(PLF_RES, "priority[%u], flags[%u], ptr[%p], ret[%d].", priority, flag, ptr, ret);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR(
            "[Stream][CreateWithFlags]errNo[0x%016llx] rtStreamCreate error, "
            "rtRet[%d], flags[%u].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, flag);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

HcclResult HrtStreamDestroy(aclrtStream ptr)
{
    HcclUs startut = TIME_NOW();
    if ((GetPlfDebugConfigValue() & PLF_RES) != 0) {
        s32 deviceId = 0;
        s32 streamId = 0;
        HcclResult devRet = HrtGetDevice(deviceId);
        if (devRet != HCCL_SUCCESS) {
            HCCL_WARNING("[HrtStreamDestroy] HrtGetDevice failed, ret=%d.", devRet);
        }
        HcclResult streamRet = HrtGetStreamId(ptr, streamId);
        if (streamRet != HCCL_SUCCESS) {
            HCCL_WARNING("[HrtStreamDestroy] HrtGetStreamId failed, ret=%d.", streamRet);
        }
        PLF_CONFIG_INFO(PLF_RES, "Destroy Stream para: deviceId[%d] streamId[%d]", deviceId, streamId);
    }
    aclError ret = aclrtDestroyStreamForce(ptr);
    HCCL_INFO(
        "[HrtStreamDestroy] ptr[%p], ret[%d], take time [%lld]us.", ptr, ret,
        DURATION_US(TIME_NOW() - startut).count());
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR(
            "[Stream][Destroy]errNo[0x%016llx] rt stream Destroy fail. "
            "return[%d], ptr[%p].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, ptr);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

HcclResult HrtStreamActive(aclrtStream activeStream, aclrtStream stream)
{
    aclError ret = aclrtActiveStream(activeStream, stream);
    HCCL_INFO("[HrtStreamActive] activeStream[%p], stream[%p], ret[%d].", activeStream, stream, ret);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR(
            "[Activate][Stream]errNo[0x%016llx] "
            "rt stream active fail. return[%d], active_stream=%p, stream=%p.",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, activeStream, stream);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

inline s32 GetMsTimeFromExecTimeout()
{
    constexpr s32 HCCL_EXEC_TIME_OUT_OFFSET_S = 5; // 避免与notifywait timeout时间冲突，增加5s的偏移值
    constexpr u32 TIME_S_TO_MS = 1000;
    s32 execTimeOut = 3000; // 从环境变量中获取超时时间
    s64 timeOutMs = (execTimeOut + HCCL_EXEC_TIME_OUT_OFFSET_S) * TIME_S_TO_MS;
    timeOutMs = (timeOutMs > 0x7FFFFFFF) ? 0x7FFFFFFF : timeOutMs;
    return static_cast<s32>(static_cast<u64>(timeOutMs) & (0x7FFFFFFF));
}

HcclResult HcclStreamSynchronize(HcclRtStream ptr)
{
    if (ptr == nullptr) {
        HCCL_ERROR("ptr is null, call aclrtSynchronizeStreamWithTimeout failed, ptr=%p", ptr);
        return HCCL_E_PTR;
    }
    HCCL_INFO("[HcclStreamSynchronize] ptr[%p].", ptr);
    s32 timeout = GetMsTimeFromExecTimeout();
    aclError ret = aclrtSynchronizeStreamWithTimeout(ptr, timeout);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR(
            "[Synchronize][Stream]errNo[0x%016llx] rt "
            "streamsynchronizewithtimeout fail. return[%d], stream[%p], timeout[%d ms].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, ptr, timeout);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

/*
 *RT_MEMORY_TS：aclrtMallocForTaskScheduler
 *RT_MEMORY_DDR：ACL_MEM_TYPE_LOW_BAND_WIDTH
 *RT_MEMORY_HBM：ACL_MEM_TYPE_HIGH_BAND_WIDTH
 */
HcclResult HrtMalloc(u64 size, aclrtMemType_t memType, void*& devPtr)
{
    devPtr = nullptr;
    aclError ret = ACL_SUCCESS;
    aclrtMallocAttrValue moduleIdValue;
    moduleIdValue.moduleId = HCCL;
    aclrtMallocAttribute attrs{.attr = ACL_RT_MEM_ATTR_MODULE_ID, .value = moduleIdValue};
    aclrtMallocConfig cfg{.attrs = &attrs, .numAttrs = 1};
    ret = aclrtMallocWithCfg(&devPtr, size, static_cast<aclrtMemMallocPolicy>(memType), &cfg);
    if ((GetPlfDebugConfigValue() & PLF_RES) != 0) {
        s32 deviceId = 0;
        HcclResult devRet = HrtGetDevice(deviceId);
        if (devRet != HCCL_SUCCESS) {
            HCCL_WARNING("[HrtMalloc] HrtGetDevice failed, ret=%d.", devRet);
        }
        PLF_CONFIG_INFO(
            PLF_RES, "ret[%d] size[%llu] memType[%d] devPtr[%p] deviceId[%d]", ret, size, memType, devPtr, deviceId);
    }
    if (ret == ACL_ERROR_RT_MEMORY_ALLOCATION) {
        RPT_INPUT_ERR(
            true, "EI0007", std::vector<std::string>({"resource_type", "resource_info"}),
            std::vector<std::string>(
                {"DeviceMemory", std::string("MallocWithCfg, size:") + std::to_string(size) + " bytes"}));
        HCCL_ERROR(
            "[%s][%s] [Malloc][Mem]errNo[0x%016llx] aclrtMallocWithCfg failed, "
            "Reason: out of memory, return[%d], para: devPtrAddr[%p], size[%llu]",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RESOURCE.c_str(), HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME),
            ret, devPtr, size);
        return HCCL_E_RUNTIME;
    }
    if (ret != ACL_SUCCESS) {
        RPT_INPUT_ERR(
            true, "EI0007", std::vector<std::string>({"resource_type", "resource_info"}),
            std::vector<std::string>(
                {"DeviceMemory", std::string("MallocWithCfg, size:") + std::to_string(size) + " bytes"}));
        HCCL_ERROR(
            "[%s][%s] [Malloc][Mem]errNo[0x%016llx] aclrtMallocWithCfg failed, "
            "return[%d], para: devPtrAddr[%p], size[%llu]",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RESOURCE.c_str(), HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME),
            ret, devPtr, size);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

HcclResult HrtFree(void* devPtr)
{
    if ((GetPlfDebugConfigValue() & PLF_RES) != 0) {
        s32 deviceId = 0;
        HcclResult devRet = HrtGetDevice(deviceId);
        if (devRet != HCCL_SUCCESS) {
            HCCL_WARNING("[HrtFree] HrtGetDevice failed, ret=%d.", devRet);
        }
        PLF_CONFIG_INFO(PLF_RES, "Free DevMem para: deviceId[%d] devPtr[%p]", deviceId, devPtr);
    }
    aclError ret = aclrtFree(devPtr);
    if (ret != RT_ERROR_NONE) {
        HCCL_ERROR(
            "[Free][Mem]errNo[0x%016llx] aclrtFree failed. "
            "return[%d], para: devPtrAddr[%p].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, devPtr);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

HcclResult MemcpyKindTranslate(rtMemcpyKind_t kind, aclrtMemcpyKind* rtKind)
{
    switch (kind) {
        case rtMemcpyKind_t::RT_MEMCPY_HOST_TO_HOST: {
            *rtKind = ACL_MEMCPY_HOST_TO_HOST;
            return HCCL_SUCCESS;
        }
        case rtMemcpyKind_t::RT_MEMCPY_HOST_TO_DEVICE: {
            *rtKind = ACL_MEMCPY_HOST_TO_DEVICE;
            return HCCL_SUCCESS;
        }
        case rtMemcpyKind_t::RT_MEMCPY_DEVICE_TO_HOST: {
            *rtKind = ACL_MEMCPY_DEVICE_TO_HOST;
            return HCCL_SUCCESS;
        }
        case rtMemcpyKind_t::RT_MEMCPY_DEVICE_TO_DEVICE: {
            *rtKind = ACL_MEMCPY_DEVICE_TO_DEVICE;
            return HCCL_SUCCESS;
        }
        default: {
            HCCL_ERROR("[MemcpyKindTranslate]Not support the memory copy type[%d].", kind);
            return HCCL_E_PARA;
        }
    }
}

HcclResult HrtMemcpy(void* dst, uint64_t destMax, const void* src, uint64_t count, rtMemcpyKind_t kind)
{
    aclmdlRICaptureMode mode = aclmdlRICaptureMode::ACL_MODEL_RI_CAPTURE_MODE_RELAXED;
    HcclResult hcclRet = HrtThreadExchangeCaptureMode(&mode);
    CHK_PRT_CONT(
        hcclRet != HCCL_SUCCESS && hcclRet != HCCL_E_NOT_SUPPORT,
        HCCL_WARNING("[hrtMemcpy] HrtThreadExchangeCaptureMode return [%d].", hcclRet));
    aclrtMemcpyKind rtKind = ACL_MEMCPY_DEFAULT;
    hcclRet = MemcpyKindTranslate(kind, &rtKind);
    aclError ret = aclrtMemcpy(dst, destMax, src, count, rtKind);
    HCCL_INFO("[HrtMemcpy] dst[%p], destMax[%llu], src[%p], count[%llu], ret[%d].", dst, destMax, src, count, ret);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR(
            "[SyncCopy][Mem]errNo[0x%016llx] rtMemcpy failed. "
            "return[%d], para: dstAddr[%p], destMax[%llu], srcAddr[%p], count[%llu], kind[%d].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, dst, destMax, src, count, kind);
        return HCCL_E_RUNTIME;
    }
    hcclRet = HrtThreadExchangeCaptureMode(&mode);
    CHK_PRT_CONT(
        hcclRet != HCCL_SUCCESS && hcclRet != HCCL_E_NOT_SUPPORT,
        HCCL_WARNING("[hrtMemcpy] HrtThreadExchangeCaptureMode return [%d].", hcclRet));
    return HCCL_SUCCESS;
}

HcclResult HrtMemset(void* dst, uint64_t destMax, uint64_t count)
{
    HCCL_INFO("[HrtMemset] dst[%p], destMax[%llu], count[%llu].", dst, destMax, count);
    aclmdlRICaptureMode mode = aclmdlRICaptureMode::ACL_MODEL_RI_CAPTURE_MODE_RELAXED;
    HcclResult hcclRet = HrtThreadExchangeCaptureMode(&mode);
    CHK_PRT_CONT(
        hcclRet != HCCL_SUCCESS && hcclRet != HCCL_E_NOT_SUPPORT,
        HCCL_WARNING("[hrtMemSet] HrtThreadExchangeCaptureMode return [%d].", hcclRet));
    aclError ret = aclrtMemset(dst, destMax, 0, count);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR(
            "[SyncSet][Mem]errNo[0x%016llx] aclrtMemset failed. "
            "return[%d], para: dstAddr[%p], destMax[%llu], count[%llu].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, dst, destMax, count);
        return HCCL_E_RUNTIME;
    }
    hcclRet = HrtThreadExchangeCaptureMode(&mode);
    CHK_PRT_CONT(
        hcclRet != HCCL_SUCCESS && hcclRet != HCCL_E_NOT_SUPPORT,
        HCCL_WARNING("[hrtMemSet] HrtThreadExchangeCaptureMode return [%d].", hcclRet));
    return HCCL_SUCCESS;
}

HcclResult HrtIpcSetMemoryName(void* ptr, char_t* name, u64 ptrMaxLen, u32 nameMaxLen)
{
    aclError ret = aclrtIpcMemGetExportKey(ptr, ptrMaxLen, name, nameMaxLen, 1UL);
    HCCL_INFO(
        "Call aclrtIpcMemGetExportKey, return value[%d], para: ptr[%p], name[%s], byteCount[%llu], nameLen[%u]", ret,
        ptr, name, ptrMaxLen, nameMaxLen);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR(
            "[Set][IpcMemoryName]errNo[0x%016llx] rtSet Ipc Memory Name. "
            "return[%d], para: ptr[%p], byteCount[%llu], name[%s].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, ptr, ptrMaxLen, name);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

HcclResult HrtIpcDestroyMemoryName(const char_t* name)
{
    aclError ret = aclrtIpcMemClose(ReinterpretAs<const char*>(name));
    HCCL_INFO("Call aclrtIpcMemClose, return[%d], para: name[%s]", ret, ReinterpretAs<const char*>(name));
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR(
            "[Destroy][IpcMemoryName]errNo[0x%016llx] "
            "rtDestroy Ipc memory name fail. return[%d], para: name[%s].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, name);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

HcclResult HrtIpcOpenMemory(const char_t* name, void*& ptr)
{
    HCCL_INFO("[HrtIpcOpenMemory] name[%s].", name);
    ptr = nullptr;
    aclError ret = aclrtIpcMemImportByKey(&ptr, name, 0UL);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR(
            "[Open][IpcMemory]errNo[0x%016llx] "
            "rtOpen ipc memory fail. return[%d], para: ptr[%p], name[%s].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, ptr, name);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

HcclResult HrtIpcCloseMemory(const void* ptr)
{
    aclError ret = aclrtIpcMemClose(ReinterpretAs<const char*>(ptr));
    HCCL_INFO("Call aclrtIpcMemClose, return[%d], para: name[%s].", ret, ReinterpretAs<const char*>(ptr));
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR(
            "[Close][IpcMemory]errNo[0x%016llx] "
            "rtClose ipc memory failed. return[%d], para: ptr[%p].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, ptr);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

HcclResult HrtIpcSetMemoryPid(const char_t* name, int pid)
{
    aclError ret = aclrtIpcMemSetImportPid(name, &pid, 1);
    HCCL_INFO("Call aclrtIpcMemSetImportPid, return value[%d], pid[%d], name[%s].", ret, pid, name);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR(
            "[Set][IpcMemoryPid]errNo[0x%016llx] "
            "rtSet ipc memory pid fail. return[%d], pid[%d], name[%s].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, pid, name);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

HcclResult HrtPointerGetAttributes(const void* ptr, aclrtPtrAttributes& ptrAttr)
{
    HCCL_INFO("[HrtPointerGetAttributes] ptr[%p].", ptr);
    aclError ret = aclrtPointerGetAttributes(ptr, ReinterpretAs<aclrtPtrAttributes*>(&ptrAttr));
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR(
            "[Get][PointAttr]errNo[0x%016llx] rt get point attr failed, "
            "return[%d], para: ptrAddr[%p].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, ptr);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

void PrintMemoryAttr(const void* memAddr)
{
    if (LIKELY(!CheckInfoLogLevel())) {
        return;
    }
    aclrtPtrAttributes memAttr;
    HcclResult ret = HrtPointerGetAttributes(memAddr, memAttr);
    if (ret != HCCL_SUCCESS) {
        HCCL_WARNING("[PrintMemoryAttr] HrtPointerGetAttributes failed, ret=%d.", ret);
        return;
    }
    HCCL_INFO(
        "[PrintMemoryAttr] address[%p], page size[%u], type[%d]", memAddr, memAttr.pageSize, memAttr.location.type);
}

HcclResult HrtDevMemAlignWithPage(void* ptr, u64 size, void*& ipcPtr, u64& ipcSize, u64& ipcOff)
{
    aclrtPtrAttributes memAttr;
    HcclResult ret = HrtPointerGetAttributes(ptr, memAttr);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }

    HCCL_INFO(
        "[HrtDevMemAlignWithPage] ptr[%p], size[%llu], ipcPtr[%p], ipcSize[%llu], ipcOff[%llu], pageSize[%u].", ptr,
        size, ipcPtr, ipcSize, ipcOff, memAttr.pageSize);
    if (memAttr.pageSize == 0) {
        ipcPtr = ptr;
        ipcSize = size;
        ipcOff = 0;
        return HCCL_SUCCESS;
    }

    u64 tmpPtr = ReinterpretAs<u64>(ptr);
    ipcPtr = ReinterpretAs<void*>((ReinterpretAs<u64>(ptr)) & (~(static_cast<u64>(memAttr.pageSize) - 1)));
    ipcOff = tmpPtr - ReinterpretAs<u64>(ipcPtr);
    ipcSize = size + ipcOff;
    return HCCL_SUCCESS;
}

HcclResult HrtMallocHost(u64 size, void*& hostPtr)
{
    hostPtr = nullptr;
    aclrtMallocAttrValue moduleIdValue;
    moduleIdValue.moduleId = HCCL;
    aclrtMallocAttribute attrs{.attr = ACL_RT_MEM_ATTR_MODULE_ID, .value = moduleIdValue};
    aclrtMallocConfig cfg{.attrs = &attrs, .numAttrs = 1};
    aclError ret = aclrtMallocHostWithCfg(&hostPtr, size, &cfg);
    PLF_CONFIG_INFO(PLF_RES, "Malloc HostMem para: return value[%d] hostPtr[%p] size[%llu]", ret, hostPtr, size);
    if (ret != ACL_SUCCESS) {
        RPT_INPUT_ERR(
            true, "EI0007", std::vector<std::string>({"resource_type", "resource_info"}),
            std::vector<std::string>(
                {"HostMemory", std::string("MallocHostWithCfg, size:") + std::to_string(size) + " bytes"}));
        HCCL_ERROR(
            "[%s][%s] [Malloc][Host]errNo[0x%016llx] rt malloc host fail. return[%d], para: size[%llu].",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RESOURCE.c_str(), HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME),
            ret, size);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

HcclResult HrtFreeHost(void* hostPtr)
{
    PLF_CONFIG_INFO(PLF_RES, "Free HostMem para: hostPtr[%p]", hostPtr);
    aclError ret = aclrtFreeHost(hostPtr);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR(
            "[Free][Host]errNo[0x%016llx] rt free host fail. return[%d], para: hostPtr[%p].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, hostPtr);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

HcclResult HrtNotifyCreate(s32 deviceLogicId, aclrtNotify& ptr)
{
    ptr = nullptr;
    // aclrtCreateNotify 中通过 aclrtGetDevice 获取 deviceId，所以要求当前线程设置过 setDevice
    aclError ret = aclrtCreateNotify(&ptr, ACL_NOTIFY_DEFAULT);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR(
            "[Notify][Create]errNo[0x%016llx] rtNotifyCreate error. "
            "return[%d], deviceId[%d], ptr[%p].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, deviceLogicId, ptr);
        return HCCL_E_RUNTIME;
    }
    if ((GetPlfDebugConfigValue() & PLF_RES) != 0) {
        s32 deviceId = 0;
        HcclResult devRet = HrtGetDevice(deviceId);
        if (devRet != HCCL_SUCCESS) {
            HCCL_WARNING("[HrtNotifyCreate] HrtGetDevice failed, ret=%d.", devRet);
        }
        u32 notifyId = 0;
        HcclResult nidRet = HrtGetNotifyID(ptr, notifyId);
        if (nidRet != HCCL_SUCCESS) {
            HCCL_WARNING("[HrtNotifyCreate] HrtGetNotifyID failed, ret=%d.", nidRet);
        }
        PLF_CONFIG_INFO(PLF_RES, "Create Notify para: deviceId[%d] notifyId[%u]", deviceId, notifyId);
    }
    return HCCL_SUCCESS;
}

HcclResult HrtNotifyDestroy(RtNotify_t ptr)
{
    HcclUs startut = TIME_NOW();
    if ((GetPlfDebugConfigValue() & PLF_RES) != 0) {
        s32 deviceId = 0;
        HcclResult devRet = HrtGetDevice(deviceId);
        if (devRet != HCCL_SUCCESS) {
            HCCL_WARNING("[HrtNotifyDestroy] HrtGetDevice failed, ret=%d.", devRet);
        }
        u32 notifyId = 0;
        HcclResult nidRet = HrtGetNotifyID(ptr, notifyId);
        if (nidRet != HCCL_SUCCESS) {
            HCCL_WARNING("[HrtNotifyDestroy] HrtGetNotifyID failed, ret=%d.", nidRet);
        }
        PLF_CONFIG_INFO(PLF_RES, "Destroy Notify para: deviceId[%d] notifyId[%u]", deviceId, notifyId);
    }
    aclError ret = aclrtDestroyNotify(ptr);
    HCCL_INFO(
        "[HrtNotifyDestroy] ptr[%p], ret[%d], take time [%lld]us.", ptr, ret,
        DURATION_US(TIME_NOW() - startut).count());
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR(
            "[Notify][Destroy]errNo[0x%016llx] aclrtDestroyNotify error. ptr[%p], return[%d].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ptr, ret);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

HcclResult HrtIpcSetNotifyName(RtNotify_t ptr, char_t* name, uint32_t len)
{
    aclError ret = aclrtNotifyGetExportKey(ptr, name, len, 2UL);
    HCCL_INFO("[HrtIpcSetNotifyName] ptr[%p], name[%s], len[%u], ret[%d].", ptr, name, len, ret);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR(
            "[Set][IPCNotify]errNo[0x%016llx] IPC set notify name fail. "
            "ptr[%p], name[%s], len[%u], return[%d].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ptr, name, len, ret);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

HcclResult HrtGetNotifyID(RtNotify_t notifyHandle, u32& notifyID)
{
    notifyID = 0;
    aclError ret = aclrtGetNotifyId(notifyHandle, &notifyID);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR(
            "[HrtGetNotifyID]rt get notify id failed. "
            "notifyHandle[%p], notifyID[%u], return[%d].",
            notifyHandle, notifyID, ret);
        return HCCL_E_RUNTIME;
    }
    HCCL_INFO("[HrtGetNotifyID] notifyHandle[%p], notifyID[%u].", notifyHandle, notifyID);
    return HCCL_SUCCESS;
}

HcclResult HrtNotifyGetAddr(RtNotify_t notifyHandle, u64& addr)
{
    HCCL_INFO("[HrtNotifyGetAddr] notifyHandle[%p].", notifyHandle);
    addr = 0;
    uint64_t addrVal = 0;
    rtError_t ret = rtGetNotifyAddress(notifyHandle, &addrVal);
    if (ret != RT_ERROR_NONE) {
        HCCL_ERROR(
            "[rtGetNotifyAddress]rt get notify address failed. "
            "notifyHandle[%p], addr[%llu], return[%d].",
            notifyHandle, addr, ret);
        return HCCL_E_RUNTIME;
    }
    addr = static_cast<u64>(addrVal);
    return HCCL_SUCCESS;
}

HcclResult HrtSetIpcNotifyPid(aclrtNotify notify, int32_t pid)
{
    aclError ret = aclrtNotifySetImportPid(notify, &pid, 1);
    HCCL_INFO("[HrtSetIpcNotifyPid] notify[%p], pid[%d], ret[%d].", notify, pid, ret);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR(
            "[Set][IpcNotifyPid]errNo[0x%016llx] "
            "rtSet ipc Notify pid fail. notify[%p], return[%d], pid[%d].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), notify, ret, pid);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

HcclResult HrtIpcOpenNotify(const char_t* name, RtNotify_t& notify)
{
    uint64_t flags = 0;
    notify = nullptr;
    aclError ret = aclrtNotifyImportByKey(&notify, name, static_cast<uint64_t>(flags));
    HCCL_INFO("[HrtIpcOpenNotify] ret[%d], para: notify[%p], name[%s].", ret, notify, name);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR(
            "[rt][IpcOpenNotify]errNo[0x%016llx] rt ipc notify open fail. "
            "return[%d]. para: notify[%p], name[%s], flags[%llu].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, notify, name, flags);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

HcclResult HrtNotifyGetOffset(RtNotify_t ptr, u32& offset)
{
    offset = 0;
    aclError ret = aclrtGetNotifyId(ptr, &offset);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR(
            "[rt][NotifyGetOffset]errNo[0x%016llx] rt ipc notify open fail. "
            "return[%d], ptr[%p], offset[%u].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, ptr, offset);
        return HCCL_E_RUNTIME;
    }
    HCCL_INFO("[HrtNotifyGetOffset] ptr[%p], offset[%u].", ptr, offset);
    return HCCL_SUCCESS;
}

HcclResult HrtNotifyWaitWithTimeOut(RtNotify_t notifyPtr, aclrtStream streamPtr, uint32_t timeOut)
{
    aclError ret = aclrtWaitAndResetNotify(notifyPtr, streamPtr, timeOut);
    HCCL_INFO(
        "[HrtNotifyWaitWithTimeOut] notifyPtr[%p], streamPtr[%p], timeOut[%u ms], ret[%d].", notifyPtr, streamPtr,
        timeOut, ret);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR(
            "call aclrtWaitAndResetNotify failed. notifyPtr=[%p], streamPtr=[%p], timeout=[%u ms], return=[%d].",
            notifyPtr, streamPtr, timeOut, ret);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

HcclResult HrtNotifyRecord(RtNotify_t notifyPtr, aclrtStream streamPtr)
{
    aclError ret = aclrtRecordNotify(notifyPtr, streamPtr);
    HCCL_INFO("[HrtNotifyRecord] notifyPtr[%p], streamPtr[%p], ret[%d].", notifyPtr, streamPtr, ret);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR("call HrtNotifyRecord failed. notifyPtr=%p, streamPtr=%p, return=%d.", notifyPtr, streamPtr, ret);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

HcclResult HrtMemAsyncCopy(
    void* dst, uint64_t destMax, const void* src, uint64_t count, aclrtMemcpyKind kind, aclrtStream streamPtr)
{
    aclError ret = aclrtMemcpyAsync(dst, destMax, src, count, kind, streamPtr);
    HCCL_DEBUG(
        "[HrtMemAsyncCopy] ret[%d], para: dstAddr[%p], destMax[%llu], "
        "srcAddr[%p], count[%llu], rtKind[%d], streamPtr[%p].",
        ret, dst, destMax, src, count, kind, streamPtr);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR(
            "[AsyncCopy][Mem]errNo[0x%016llx] rt memory async copy failed. "
            "return[%d], para: dstAddr[%p], destMax[%llu], srcAddr[%p], count[%llu], kind[%d], stream[%p].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, dst, destMax, src, count, kind, streamPtr);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

HcclResult HrtReduceAsync(
    void* dst, uint64_t destMax, const void* src, uint64_t count, aclrtReduceKind kind, aclDataType type,
    aclrtStream streamPtr)
{
    // reserve 预留字段填 nullptr
    aclError ret = aclrtReduceAsync(dst, src, count, kind, type, streamPtr, nullptr);
    HCCL_INFO(
        "Call rtReduceAsync, return value[%d], para: dst[%p], destMax[%llu], src[%p], count[%llu], rtReduceOp[%d], "
        "rtDataType[%d], streamPtr[%p].",
        ret, dst, destMax, src, count, kind, type, streamPtr);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR(
            "[rt][ReduceAsync]errNo[0x%016llx] rt reduce async fail. "
            "return[%d], para: dst[%p], destMax[%llu], src[%p], count[%llu], rtReduceOp[%d], rtDataType[%d], "
            "streamPtr[%p].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, dst, destMax, src, count, kind, type, streamPtr);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

HcclResult HrtRDMASend(u32 qpn, u32 wqeIndex, aclrtStream streamPtr)
{
    rtError_t ret = rtRDMASend(qpn, wqeIndex, streamPtr);
    HCCL_INFO(
        "Call rtRDMASend, return value[%d]. Params: qpn[%u], wqeIndex[%u], streamPtr[%p].", ret, qpn, wqeIndex,
        streamPtr);
    if (ret != RT_ERROR_NONE) {
        HCCL_ERROR(
            "[rt][RdmaSend]errNo[0x%016llx] rt rdma send fail. "
            "return[%d]. para: qpn[%u], wqeIndex[%u], streamPtr[%p].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, qpn, wqeIndex, streamPtr);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

HcclResult HrtRDMADBSend(uint32_t dbindex, uint64_t dbinfo, aclrtStream streamPtr)
{
    HCCL_INFO("[HrtRDMADBSend] dbindex[%u], dbinfo[%llu], streamPtr[%p].", dbindex, dbinfo, streamPtr);
    rtError_t ret = rtRDMADBSend(dbindex, dbinfo, streamPtr);
    if (ret != RT_ERROR_NONE) {
        HCCL_ERROR(
            "[rtRDMADBSend]errNo[0x%016llx] rt rdma send fail. "
            "return[%d]. para: dbindex[%u], dbinfo[%llu], streamPtr[%p].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, dbindex, dbinfo, streamPtr);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

HcclResult HrtGetTaskIdAndStreamID(u32& taskId, u32& streamId)
{
    rtError_t ret = rtGetTaskIdAndStreamID(&taskId, &streamId);
    HCCL_INFO("[HrtGetTaskIdAndStreamID] ret[%d], para: taskId[%u], streamId[%u].", ret, taskId, streamId);
    if (ret != RT_ERROR_NONE) {
        HCCL_ERROR(
            "[Get][TaskIdAndStreamID]errNo[0x%016llx] "
            "rt get task ID and stream ID fail. return[%d], taskId[%u], streamId[%u].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, taskId, streamId);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

HcclResult HrtUbDbSend([[maybe_unused]] const HrtUbDbInfo& info, [[maybe_unused]] aclrtStream streamPtr)
{
    HCCL_ERROR("[%s] Unsupported rtUbDbSend", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult HrtUbDirectSend([[maybe_unused]] const HrtUbWqeInfo& info, [[maybe_unused]] aclrtStream streamPtr)
{
    HCCL_ERROR("[%s] Unsupported rtUbDirectSend", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult HrtCntNotifyCreate(u32 deviceId, aclrtCntNotify& handle)
{
    handle = nullptr;
    aclError ret = aclrtCntNotifyCreate(&handle, RT_NOTIFY_FLAG_DEFAULT);
    HCCL_INFO("Call aclrtCntNotifyCreate, return value[%d] devId[%u].", ret, deviceId);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR("[%s] Call aclrtCntNotifyCreate failed, ret[%d], devId[%u]", __func__, ret, deviceId);
        return HCCL_E_RUNTIME;
    }
    if ((GetPlfDebugConfigValue() & PLF_RES) != 0) {
        s32 devId = 0;
        HcclResult devRet = HrtGetDevice(devId);
        if (devRet != HCCL_SUCCESS) {
            HCCL_WARNING("[HrtCntNotifyCreate] HrtGetDevice failed, ret=%d.", devRet);
        }
        u32 notifyId = 0;
        HcclResult nidRet = HrtGetCntNotifyId(handle, notifyId);
        if (nidRet != HCCL_SUCCESS) {
            HCCL_WARNING("[HrtCntNotifyCreate] HrtGetCntNotifyId failed, ret=%d.", nidRet);
        }
        PLF_CONFIG_INFO(PLF_RES, "Create Notify para: deviceId[%d] notifyId[%u]", devId, notifyId);
    }
    return HCCL_SUCCESS;
}

HcclResult HrtGetCntNotifyId(const aclrtCntNotify inCntNotify, u32& notifyId)
{
    notifyId = 0; // 待接口rtGetCntNotifyId(inCntNotify, notifyId)上库，目前打桩;
    aclError ret = aclrtCntNotifyGetId(inCntNotify, &notifyId);
    HCCL_INFO("[HrtGetCntNotifyId] ret[%d], inCntNotify[%p], notifyId[%u]", ret, inCntNotify, notifyId);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR(
            "[%s] Call rtGetCntNotifyId failed. return[%d], inCntNotify[%p], notifyId[%u].", __func__, ret, inCntNotify,
            notifyId);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

HcclResult HrtCntNotifyDestroy(const aclrtCntNotify inCntNotify)
{
    if ((GetPlfDebugConfigValue() & PLF_RES) != 0) {
        s32 deviceId = 0;
        HcclResult devRet = HrtGetDevice(deviceId);
        if (devRet != HCCL_SUCCESS) {
            HCCL_WARNING("[HrtCntNotifyDestroy] HrtGetDevice failed, ret=%d.", devRet);
        }
        u32 notifyId = 0;
        HcclResult nidRet = HrtGetCntNotifyId(inCntNotify, notifyId);
        if (nidRet != HCCL_SUCCESS) {
            HCCL_WARNING("[HrtCntNotifyDestroy] HrtGetCntNotifyId failed, ret=%d.", nidRet);
        }
        PLF_CONFIG_INFO(PLF_RES, "Destroy Notify para: deviceId[%d] notifyId[%u]", deviceId, notifyId);
    }
    aclError ret = aclrtCntNotifyDestroy(inCntNotify);
    HCCL_INFO("[HrtCntNotifyDestroy] ret[%d], inCntNotify[%p].", ret, inCntNotify);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR("[%s] Call aclrtCntNotifyDestroy failed, ret[%d], inCntNotify[%p]", __func__, ret, inCntNotify);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

const std::map<HrtCntNotifyRecordMode, aclrtCntNotifyRecordMode> HRT_CNT_NOTIFY_RECORD_MODE_MAP
    = {{HrtCntNotifyRecordMode::WRITE_BIT, aclrtCntNotifyRecordMode::ACL_RT_CNT_NOTIFY_RECORD_BIT_OR_MODE},
       {HrtCntNotifyRecordMode::STORE, aclrtCntNotifyRecordMode::ACL_RT_CNT_NOTIFY_RECORD_SET_VALUE_MODE}};
HcclResult HrtCntNotifyRecord(
    const aclrtCntNotify inCntNotify, const aclrtStream streamPtr, HrtCntNotifyRecordMode mode, u32 value)
{
    aclrtCntNotifyRecordInfo recordInfo{};
    auto recordIt = HRT_CNT_NOTIFY_RECORD_MODE_MAP.find(mode);
    CHK_PRT_RET(
        recordIt == HRT_CNT_NOTIFY_RECORD_MODE_MAP.end(),
        HCCL_ERROR("[%s] invalid CntNotifyRecord mode[%u].", __func__, static_cast<u32>(mode)), HCCL_E_PARA);
    recordInfo.mode = recordIt->second;
    recordInfo.value = value;
    aclError ret = aclrtCntNotifyRecord(inCntNotify, streamPtr, &recordInfo);
    HCCL_INFO(
        "[HrtCntNotifyRecord] inCntNotify[%p], streamPtr[%p], mode[%d], value[%u], ret[%d].", inCntNotify, streamPtr,
        mode, value, ret);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR(
            "[%s] Call aclrtCntNotifyRecord failed, ret[%d], inCntNotify[%p], streamPtr[%p], mode[%d], value[%u]",
            __func__, ret, inCntNotify, streamPtr, mode, value);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}
// david接口 包间接口
const std::map<HrtCntNotifyWaitMode, aclrtCntNotifyWaitMode> HRT_CNT_NOTIFY_WAIT_MODE_MAP
    = {{HrtCntNotifyWaitMode::EQUAL, aclrtCntNotifyWaitMode::ACL_RT_CNT_NOTIFY_WAIT_EQUAL_MODE},
       {HrtCntNotifyWaitMode::BITMAP, aclrtCntNotifyWaitMode::ACL_RT_CNT_NOTIFY_WAIT_EQUAL_WITH_BITMASK_MODE}};
HcclResult HrtCntNotifyWaitWithTimeOut(
    const aclrtCntNotify inCntNotify, const aclrtStream streamPtr, HrtCntNotifyWaitMode mode, u32 value, u32 timeout,
    bool isClear)
{
    aclrtCntNotifyWaitInfo waitInfo{};
    auto waitIt = HRT_CNT_NOTIFY_WAIT_MODE_MAP.find(mode);
    CHK_PRT_RET(
        waitIt == HRT_CNT_NOTIFY_WAIT_MODE_MAP.end(),
        HCCL_ERROR("[%s] invalid CntNotifyWait mode[%u].", __func__, static_cast<u32>(mode)), HCCL_E_PARA);
    waitInfo.mode = waitIt->second;
    waitInfo.value = value;
    waitInfo.isClear = isClear;
    waitInfo.timeout = timeout;
    aclError ret = aclrtCntNotifyWaitWithTimeout(inCntNotify, streamPtr, &waitInfo);
    HCCL_INFO(
        "[HrtCntNotifyWaitWithTimeOut] inCntNotify[%p], streamPtr[%p], mode[%d], "
        "value[%u], timeout[%u ms], isClear[%d], ret[%d].",
        inCntNotify, streamPtr, mode, value, timeout, isClear, ret);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR(
            "[%s] Call rtCntNotifyWaitWithTimeout failed. return[%d], inCntNotify[%p], "
            "streamPtr[%p], mode[%d], value[%u], timeout[%u ms], isClear[%d].",
            __func__, ret, inCntNotify, streamPtr, mode, value, timeout, isClear);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

HcclResult HrtNotifyCreateWithFlag(u32 devId, u32 flag, aclrtNotify& ptr)
{
    ptr = nullptr;
    aclError ret = aclrtCreateNotify(&ptr, flag);
    HCCL_INFO("Call HrtNotifyCreateWithFlag, return value[%d], flag[%u] devid[%u].", ret, flag, devId);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR(
            "[%s] Call rtNotifyCreateWithFlag failed. return[%d], devId[%u], flag[%u].", __func__, ret, devId, flag);
        return HCCL_E_RUNTIME;
    }
    if ((GetPlfDebugConfigValue() & PLF_RES) != 0) {
        s32 deviceId = 0;
        HcclResult devRet = HrtGetDevice(deviceId);
        if (devRet != HCCL_SUCCESS) {
            HCCL_WARNING("[HrtNotifyCreateWithFlag] HrtGetDevice failed, ret=%d.", devRet);
        }
        u32 notifyId = 0;
        HcclResult nidRet = HrtGetNotifyID(ptr, notifyId);
        if (nidRet != HCCL_SUCCESS) {
            HCCL_WARNING("[HrtNotifyCreateWithFlag] HrtGetNotifyID failed, ret=%d.", nidRet);
        }
        PLF_CONFIG_INFO(PLF_RES, "Create Notify para: deviceId[%d] notifyId[%u]", deviceId, notifyId);
    }
    return HCCL_SUCCESS;
}

HcclResult HrtIpcOpenNotifyWithFlag(const char_t* name, uint32_t flags, RtNotify_t& ptr)
{
    HCCL_INFO("[HrtIpcOpenNotifyWithFlag] name[%s], flags[%u].", name, flags);
    ptr = nullptr;
    aclError ret = aclrtNotifyImportByKey(&ptr, name, static_cast<uint64_t>(flags));
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR(
            "[%s] Call rtIpcOpenNotifyWithFlag failed. return[%d], name[%p], flags[%u], ptr[%p].", __func__, ret, name,
            flags, ptr);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

// 兜底extern形式
HcclResult HrtAicpuLaunchKernelWithHostArgs(
    aclrtFuncHandle funcHandle, uint32_t numBlocks, aclrtStream stream, aclrtLaunchKernelCfg* cfg, void* hostArgs,
    size_t argsSize, aclrtPlaceHolderInfo* placeHolderArray, size_t placeHolderNum)
{
    rtError_t ret = aclrtLaunchKernelWithHostArgs(
        funcHandle, numBlocks, stream, cfg, hostArgs, argsSize, placeHolderArray, placeHolderNum);
    if (ret != RT_ERROR_NONE) {
        HCCL_ERROR("[%s] Call aclrtLaunchKernelWithHostArgs failed, with ret[%d]", __func__, ret);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

HcclResult HrtRegTaskExceptionCallbackByModule(aclrtExceptionInfoCallback callback)
{
    HCCL_INFO("[HrtRegTaskExceptionCallbackByModule] callback[%p].", callback);
    aclError ret = aclrtExceptionInfoCallbackRegister(callback);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR(
            "[%s] Call aclrtExceptionInfoCallbackRegister failed. return[%d], callback[%p].", __func__, ret, callback);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

HcclResult HrtUnregTaskExceptionCallbackByModule(aclrtExceptionInfoCallback callback)
{
    HCCL_INFO("[HrtUnregTaskExceptionCallbackByModule] callback[%p].", callback);
    aclError ret = aclrtExceptionInfoCallbackUnregister(callback);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR("Call aclrtExceptionInfoCallbackUnregister failed. return[%d], callback[%p].", ret, callback);
        return HcclResult::HCCL_E_RUNTIME;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult HrtStreamGetSqId(const aclrtStream ptr, u32& sqId)
{
    HCCL_INFO("[HrtStreamGetSqId] ptr[%p].", ptr);
    sqId = 0;
    rtError_t ret = rtStreamGetSqid(ptr, &sqId);
    if (ret != RT_ERROR_NONE) {
        HCCL_ERROR("[%s] Call rtStreamGetSqid failed. return[%d], ptr[%p], sqId[%u].", __func__, ret, ptr, sqId);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

HcclResult HrtStreamGetCqId(const aclrtStream ptr, u32& logicCqId)
{
    HCCL_INFO("[HrtStreamGetCqId] ptr[%p].", ptr);
    u32 cqId = 0;
    logicCqId = 0;
    rtError_t ret = rtStreamGetCqid(ptr, &cqId, &logicCqId);
    if (ret != RT_ERROR_NONE) {
        HCCL_ERROR("[%s] Call rtStreamGetCqid failed, with ret[%d]", __func__, ret);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

HcclResult HrtCcuLaunch(rtCcuTaskInfo_t& taskInfo, aclrtStream const streamPtr)
{
    HCCL_INFO("[HrtCcuLaunch] taskInfo[%p], streamPtr[%p].", &taskInfo, streamPtr);
    auto ret = rtCCULaunch(&taskInfo, streamPtr);
    if (ret != RT_ERROR_NONE) {
        if (ret == ACL_ERROR_RT_STREAM_TASK_FULL) {
            // task 队列满
            HCCL_ERROR(
                "[%s] rtCCULaunch failed because the task queue on the stream is full. "
                "ret[%d], param: taskInfo[%p], streamPtr[%p]. "
                "Possible causes: 1) too many tasks have been submitted to the stream; "
                "2) the HCCL buffer is configured too small.",
                __func__, ret, &taskInfo, streamPtr);
        } else {
            HCCL_ERROR(
                "[%s] Call rtCCULaunch failed. return[%d], taskInfo[%p], streamPtr[%p].", __func__, ret, &taskInfo,
                streamPtr);
        }
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

HcclResult HrtUbDevQueryInfo(rtUbDevQueryCmd cmd, void* devInfo)
{
    HCCL_INFO("[HrtUbDevQueryInfo] cmd[%d], devInfo[%p].", cmd, devInfo);
    auto ret = rtUbDevQueryInfo(cmd, devInfo);
    if (ret != RT_ERROR_NONE) {
        HCCL_ERROR("[%s] Call rtUbDevQueryInfo failed. return[%d], cmd[%d], devInfo[%p].", __func__, ret, cmd, devInfo);
        return HCCL_E_RUNTIME;
    }
    if (cmd == QUERY_PROCESS_TOKEN) {
        rtMemUbTokenInfo* info = static_cast<rtMemUbTokenInfo*>(devInfo);
        info->tokenId = info->tokenId >> TOKEN_ID_RIGHT_SHIF;
    }
    return HCCL_SUCCESS;
}
// pair<tokendId, tokenValue>
std::pair<u32, u32> HrtUbDevQueryToken(u64 addr, u64 size)
{
    HCCL_INFO("[HrtUbDevQueryToken] addr[%llu], size[%llu].", addr, size);
    rtMemUbTokenInfo info;
    info.va = addr;
    info.size = size;
    auto ret = rtUbDevQueryInfo(QUERY_PROCESS_TOKEN, &info);
    if (ret != RT_ERROR_NONE) {
        HCCL_WARNING("query(va=0x%llx, size=0x%llx) token unsuccessfully, ret=%d", addr, size, ret);
        return std::make_pair(0, 0);
    }

    return {info.tokenId >> TOKEN_ID_RIGHT_SHIF, info.tokenValue};
}

const std::map<HrtDevResProcType, rtDevResProcType_t> HRT_DEV_RES_PROC_TYPE_MAP
    = {{HrtDevResProcType::PROCESS_CP1, RT_PROCESS_CP1}, {HrtDevResProcType::PROCESS_HCCP, RT_PROCESS_HCCP}};

const std::map<HrtDevResType, rtDevResType_t> HRT_DEV_RES_TYPE_MAP
    = {{HrtDevResType::RES_TYPE_STARS_NOTIFY_RECORD, RT_RES_TYPE_STARS_NOTIFY_RECORD},
       {HrtDevResType::RES_TYPE_CCU_CKE, RT_RES_TYPE_CCU_CKE},
       {HrtDevResType::RES_TYPE_CCU_XN, RT_RES_TYPE_CCU_XN},
       {HrtDevResType::RES_TYPE_STARS_CNT_NOTIFY_BIT_WR, RT_RES_TYPE_STARS_CNT_NOTIFY_BIT_WR}};
HcclResult HrtGetDevResAddress(const HrtDevResInfo& devResInfo, HrtDevResAddrInfo& devResAddrInfo)
{
    HCCL_INFO(
        "[HrtGetDevResAddress] devResInfo.dieId[%u], devResInfo.procType[%u], "
        "devResInfo.resType[%u], devResInfo.flag[%u], devResInfo.resId[%u].",
        devResInfo.dieId, devResInfo.procType, devResInfo.resType, devResInfo.flag, devResInfo.resId);
    rtDevResInfo resInfo;
    resInfo.dieId = devResInfo.dieId;
    auto procIt = HRT_DEV_RES_PROC_TYPE_MAP.find(devResInfo.procType);
    CHK_PRT_RET(
        procIt == HRT_DEV_RES_PROC_TYPE_MAP.end(),
        HCCL_ERROR("[%s] invalid procType[%u].", __func__, static_cast<u32>(devResInfo.procType)), HCCL_E_PARA);
    resInfo.procType = procIt->second;
    auto resIt = HRT_DEV_RES_TYPE_MAP.find(devResInfo.resType);
    CHK_PRT_RET(
        resIt == HRT_DEV_RES_TYPE_MAP.end(),
        HCCL_ERROR("[%s] invalid resType[%u].", __func__, static_cast<u32>(devResInfo.resType)), HCCL_E_PARA);
    resInfo.resType = resIt->second;
    resInfo.flag = devResInfo.flag;
    resInfo.resId = devResInfo.resId;

    uint64_t addr = 0;
    u32 len = 0;
    rtDevResAddrInfo addrInfo;
    addrInfo.resAddress = &addr;
    addrInfo.len = &len;
    auto ret = rtGetDevResAddress(&resInfo, &addrInfo);
    if (ret != RT_ERROR_NONE) {
        HCCL_ERROR(
            "[%s] Call rtGetDevResAddress failed. return[%d], devResInfo.dieId[%u], "
            "devResInfo.procType[%u], devResInfo.resType[%u], devResInfo.flag[%u], devResInfo.resId[%u].",
            __func__, ret, devResInfo.dieId, devResInfo.procType, devResInfo.resType, devResInfo.flag,
            devResInfo.resId);
        return HCCL_E_RUNTIME;
    }
    devResAddrInfo.address = addr;
    devResAddrInfo.len = len;
    HCCL_INFO("devResAddrInfo.address[%llu], devResAddrInfo.len[%u].", devResAddrInfo.address, devResAddrInfo.len);
    return HCCL_SUCCESS;
}

HcclResult HrtReleaseDevResAddress(const HrtDevResInfo& devResInfo)
{
    HCCL_INFO(
        "[HrtReleaseDevResAddress] devResInfo.dieId[%u], devResInfo.procType[%u], "
        "devResInfo.resType[%u], devResInfo.flag[%u], devResInfo.resId[%u].",
        devResInfo.dieId, devResInfo.procType, devResInfo.resType, devResInfo.flag, devResInfo.resId);
    rtDevResInfo resInfo;
    resInfo.dieId = devResInfo.dieId;
    auto procIt = HRT_DEV_RES_PROC_TYPE_MAP.find(devResInfo.procType);
    CHK_PRT_RET(
        procIt == HRT_DEV_RES_PROC_TYPE_MAP.end(),
        HCCL_ERROR("[%s] invalid procType[%u].", __func__, static_cast<u32>(devResInfo.procType)), HCCL_E_PARA);
    resInfo.procType = procIt->second;
    auto resIt = HRT_DEV_RES_TYPE_MAP.find(devResInfo.resType);
    CHK_PRT_RET(
        resIt == HRT_DEV_RES_TYPE_MAP.end(),
        HCCL_ERROR("[%s] invalid resType[%u].", __func__, static_cast<u32>(devResInfo.resType)), HCCL_E_PARA);
    resInfo.resType = resIt->second;
    resInfo.flag = devResInfo.flag;
    resInfo.resId = devResInfo.resId;

    rtError_t ret = rtReleaseDevResAddress(&resInfo);
    if (ret != RT_ERROR_NONE) {
        HCCL_ERROR(
            "[%s] Call rtReleaseDevResAddress failed. return[%d], devResInfo.dieId[%u], "
            "devResInfo.procType[%u],devResInfo.resType[%u], devResInfo.flag[%u], devResInfo.resId[%u].",
            __func__, ret, devResInfo.dieId, devResInfo.procType, devResInfo.resType, devResInfo.flag,
            devResInfo.resId);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

HcclResult HrtEventCreateWithFlag(u32 flag, aclrtEvent& ptr)
{
    HCCL_INFO("[HrtEventCreateWithFlag] flag[%u].", flag);
    ptr = nullptr;
    aclError ret = aclrtCreateEventWithFlag(&ptr, flag);
    if ((GetPlfDebugConfigValue() & PLF_RES) != 0) {
        s32 deviceId = 0;
        HcclResult devRet = HrtGetDevice(deviceId);
        if (devRet != HCCL_SUCCESS) {
            HCCL_WARNING("[HrtEventCreateWithFlag] HrtGetDevice failed, ret=%d.", devRet);
        }
        PLF_CONFIG_INFO(PLF_RES, "Create Event para: deviceId[%d] event[%p]", deviceId, ptr);
    }
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR("[%s] Call rtEventCreateWithFlag failed. return[%d], flag[%u], ptr[%p].", __func__, ret, flag, ptr);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

HcclResult HrtEventDestroy(RtEvent_t eventPtr)
{
    if ((GetPlfDebugConfigValue() & PLF_RES) != 0) {
        s32 deviceId = 0;
        HcclResult devRet = HrtGetDevice(deviceId);
        if (devRet != HCCL_SUCCESS) {
            HCCL_WARNING("[HrtEventDestroy] HrtGetDevice failed, ret=%d.", devRet);
        }
        PLF_CONFIG_INFO(PLF_RES, "Destroy Event para: deviceId[%d] event[%p]", deviceId, eventPtr);
    }
    aclError ret = aclrtDestroyEvent(eventPtr);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR("[%s] Call aclrtDestroyEvent failed. return[%d], eventPtr[%p].", __func__, ret, eventPtr);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

HcclResult HrtEventRecord(RtEvent_t eventPtr, aclrtStream streamPtr)
{
    HCCL_INFO("[HrtEventRecord] eventPtr[%p], streamPtr[%p].", eventPtr, streamPtr);
    aclError ret = aclrtRecordEvent(eventPtr, streamPtr);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR(
            "[%s] Call aclrtRecordEvent failed. return[%d], eventPtr[%p], streamPtr[%p].", __func__, ret, eventPtr,
            streamPtr);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

const std::map<aclrtEventWaitStatus, HrtEventStatus> HRT_EVENT_STATUS_MAP{
    {ACL_EVENT_WAIT_STATUS_NOT_READY, HrtEventStatus::EVENT_INIT},
    {ACL_EVENT_WAIT_STATUS_COMPLETE, HrtEventStatus::EVENT_RECORDED},
};

HcclResult HrtEventQueryStatus(RtEvent_t eventPtr, HrtEventStatus& status)
{
    HCCL_INFO("[HrtEventQueryStatus] eventPtr[%p].", eventPtr);
    aclrtEventWaitStatus rtStatus = ACL_EVENT_WAIT_STATUS_NOT_READY;
    aclError ret = aclrtQueryEventWaitStatus(eventPtr, &rtStatus);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR("[%s] Call aclrtQueryEventWaitStatus failed. return[%d], eventPtr[%p].", __func__, ret, eventPtr);
        return HCCL_E_RUNTIME;
    }
    if (HRT_EVENT_STATUS_MAP.find(rtStatus) == HRT_EVENT_STATUS_MAP.end()) {
        HCCL_ERROR("[%s] event status[%u] not in HRT_EVENT_STATUS_MAP.", __func__, static_cast<u32>(rtStatus));
        return HCCL_E_PARA;
    }
    status = HRT_EVENT_STATUS_MAP.at(rtStatus);
    return HCCL_SUCCESS;
}

HcclResult
HrtWriteValue([[maybe_unused]] u64 addr, [[maybe_unused]] u32 piVal, [[maybe_unused]] const aclrtStream streamPtr)
{
    HCCL_ERROR("[%s] Unsupported rtWriteValue", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult HrtDeviceAbortRegCallBack(aclrtDeviceTaskAbortCallback callback, void* args, const std::string& name)
{
    aclError ret = aclrtSetDeviceTaskAbortCallback(name.c_str(), callback, args);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR("[%s] call rtSetTaskAbortCallBack failed. ret=[%d].", __func__, ret);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

HcclResult HrtThreadExchangeCaptureMode(aclmdlRICaptureMode* mode)
{
    HCCL_INFO("[HrtThreadExchangeCaptureMode] mode[%p].", mode);
    aclError ret = aclmdlRICaptureThreadExchangeMode(mode);
    if (ret == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {
        HCCL_WARNING(
            "[HrtThreadExchangeCaptureMode] rtThreadExchangeCaptureMode not support!, ret=%d, mode=%p.", ret, mode);
        return HCCL_E_NOT_SUPPORT;
    } else {
        CHK_PRT_RET(
            ret != ACL_SUCCESS,
            HCCL_ERROR(
                "[HrtThreadExchangeCaptureMode]rtThreadExchangeCaptureMode "
                "failed mode:%d, return value[%d].",
                *mode, ret),
            HCCL_E_RUNTIME);
    }
    return HCCL_SUCCESS;
}

HcclResult HrtMemPrefetchToDevice(void* devPtr, uint64_t len)
{
    CHK_PRT_RET(
        aclrtMemP2PMap == nullptr,
        HCCL_ERROR("aclrtMemP2PMap is nullptr, "
                   "Does not support this interface."),
        HCCL_E_RUNTIME);
    s32 deviceId = 0;
    HcclResult devRet = HrtGetDevice(deviceId);
    if (devRet != HCCL_SUCCESS) {
        return devRet;
    }
    aclError ret = aclrtMemP2PMap(devPtr, static_cast<size_t>(len), deviceId, 0);
    HCCL_INFO("[HrtMemPrefetchToDevice] devPtr[%p], len[%llu], ret[%d].", devPtr, len, ret);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR("aclrtMemP2PMap fail ret = %d", ret);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

HcclResult HrtEnableP2P(u32 deviceLogicId, u32 devicePhyId)
{
    rtError_t ret = rtEnableP2P(deviceLogicId, devicePhyId, 0);

    HCCL_INFO("rt enableP2P deviceLogicId[%u] and devicePhyId[%u] ret[%d]", deviceLogicId, devicePhyId, ret);

    CHK_PRT_RET(
        ret != RT_ERROR_NONE,
        HCCL_ERROR(
            "[Enable][P2P]errNo[0x%016llx] rt enableP2P deviceLogicId[%u] and "
            "devicePhyId[%u] fail[%d]",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), deviceLogicId, devicePhyId, ret),
        HCCL_E_RUNTIME);

    return HCCL_SUCCESS;
}

HcclResult HrtDisableP2P(u32 deviceLogicId, u32 devicePhyId)
{
    rtError_t ret = rtDisableP2P(deviceLogicId, devicePhyId);

    HCCL_INFO("rt disableP2P deviceLogicId[%u] and devicePhyId[%u] ret[%d]", deviceLogicId, devicePhyId, ret);

    CHK_PRT_RET(
        ret != RT_ERROR_NONE,
        HCCL_ERROR(
            "[Disable][P2P]errNo[0x%016llx] rt disableP2P deviceLogicId[%u] and "
            "devicePhyId[%u] fail[%d]",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), deviceLogicId, devicePhyId, ret),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
}

HcclResult HrtGetP2PStatus(u32 deviceLogicId, u32 devicePhyId, uint32_t* status)
{
    rtError_t ret = rtGetP2PStatus(deviceLogicId, devicePhyId, status);

    HCCL_DEBUG(
        "rt getp2pstatus deviceLogicId[%u] and devicePhyId[%u] ret[%d], status[%u]", deviceLogicId, devicePhyId, ret,
        *status);
    CHK_PRT_RET(
        ret != RT_ERROR_NONE,
        HCCL_ERROR(
            "[Get][P2PStatus]errNo[0x%016llx]Call rtGetP2PStatus failed, "
            "ret[%d], deviceLogicId[%u], devicePhyId[%u]",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, deviceLogicId, devicePhyId),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
}
} // namespace Hccl
