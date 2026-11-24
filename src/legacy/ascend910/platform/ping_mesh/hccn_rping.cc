/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "network/hccp_ping.h"
#include "ping_mesh.h"
#include "hccl_ip_address.h"
#include "dlra_function.h"
#include "adapter_rts_common.h"
#include "hccn_rping.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#define HCCN_RPING_MIN_TIMEOUT 1
#define HCCN_RPING_MAX_TIMEOUT 3600000
#define HCCN_RPING_DEFAULT_TIMEOUT 120000

using namespace hccl;
constexpr u32 BUFFER_SIZE_UNIT = 4096;
constexpr u32 NPU_NUM_MAX = 32768;
constexpr u32 NPU_NUM_MIN = 128;
constexpr u32 NPU_NUM_MAX_BITWIDTH = 128;
constexpr u32 LINK_TYPE_MODE_ROCE = 3;
constexpr u32 LINK_TYPE_MODE_UB = 7;
constexpr u32 RPING_RESULT_STATE_VALID = 2;
constexpr u32 TARGET_NUM_MAX = 16;

inline HccnResult HccnRpingInitInter(
    uint32_t& devLogicIdInter, HccnRpingInitAttr* initAttrInter, PingMesh* rpingInter, u32& npuNumInter,
    std::string& ipAddrDesInter)
{
    u32 bufferSizeInter = 0;
    // npuNum必须为2的整次幂
    npuNumInter = (initAttrInter->npuNum > NPU_NUM_MIN) ? initAttrInter->npuNum : NPU_NUM_MIN;
    npuNumInter = (npuNumInter > NPU_NUM_MAX) ? (NPU_NUM_MAX - 1) : (npuNumInter - 1);
    for (u32 i = 0; i < NPU_NUM_MAX_BITWIDTH; i++) {
        npuNumInter |= (npuNumInter >> 1);
    }
    npuNumInter++;
    npuNumInter = npuNumInter < NPU_NUM_MIN ? NPU_NUM_MIN : npuNumInter;
    // bufferSize必须为4k的倍数
    if (initAttrInter->bufferSize > 0) {
        bufferSizeInter = ((initAttrInter->bufferSize - 1) / BUFFER_SIZE_UNIT + 1) * BUFFER_SIZE_UNIT;
    }
    // 初始化pingmesh实例,利用用户输入的ip地址构造HcclIpAddress类
    HcclIpAddress ipAddr;
    HcclResult ret = HCCL_SUCCESS;
    if (initAttrInter->mode == HCCN_RPING_MODE_ROCE) {
        if (!HcclIpAddress::IsIPv4(std::string(initAttrInter->ipAddr))
            && !HcclIpAddress::IsIPv6(std::string(initAttrInter->ipAddr))) {
            HCCL_ERROR("[HccnRpingInitInter] invalid ip: %s, bufferSize:%u", initAttrInter->ipAddr, bufferSizeInter);
            return HCCN_E_PARA;
        }
        ipAddr = HcclIpAddress(std::string(initAttrInter->ipAddr));
        ret = rpingInter->HccnRpingInit(
            devLogicIdInter, LINK_TYPE_MODE_ROCE, ipAddr, initAttrInter->port, npuNumInter, bufferSizeInter,
            initAttrInter->sl, initAttrInter->tc);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[HccnRpingInitInter] init fail, bufferSize:%u", bufferSizeInter);
            return HCCN_E_FAIL;
        }
        ipAddrDesInter = ipAddr.GetReadableIP();
    }
    const char* socNamePtr = aclrtGetSocName();
    CHK_PRT_RET(socNamePtr == nullptr, HCCL_ERROR("[HccnRpingInitInter]socNamePtr is null."), HCCN_E_PARA);
    ;
    if (initAttrInter->mode == HCCN_RPING_MODE_UB && IsSupportHCCLV2(socNamePtr)) {
        if (!HcclIpAddress::IsEID(std::string(initAttrInter->eid))) {
            HCCL_ERROR("[HccnRpingInitInter] invalid eid: %s, bufferSize:%u", initAttrInter->eid, bufferSizeInter);
            return HCCN_E_PARA;
        }
        Eid eid = HcclIpAddress::StrToEID(std::string(initAttrInter->eid));
        ipAddr = HcclIpAddress(eid);
        ret = rpingInter->HccnRpingInit(
            devLogicIdInter, LINK_TYPE_MODE_UB, ipAddr, initAttrInter->port, npuNumInter, bufferSizeInter,
            initAttrInter->sl, initAttrInter->tc);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[HccnRpingInitInter] init fail, bufferSize:%u, ret:%d", bufferSizeInter, ret);
            return HCCN_E_FAIL;
        }
        ipAddrDesInter = ipAddr.Describe();
    }
    HCCL_RUN_INFO("[HccnRpingInitInter]bufferSize:%u, ret:%d", bufferSizeInter, ret);
    return HCCN_SUCCESS;
}

// 新增:带 cfg 的内部初始化,透传 taMode/tpMode 给 PingMesh 9 参数重载
inline HccnResult HccnRpingInitInterWithCfg(
    uint32_t& devLogicIdInter, HccnRpingInitAttr* initAttrInter, HccnRpingInitCfg* initCfgInter, PingMesh* rpingInter,
    u32& npuNumInter, std::string& ipAddrDesInter)
{
    u32 bufferSizeInter = 0;
    // npuNum必须为2的整次幂
    npuNumInter = (initAttrInter->npuNum > NPU_NUM_MIN) ? initAttrInter->npuNum : NPU_NUM_MIN;
    npuNumInter = (npuNumInter > NPU_NUM_MAX) ? (NPU_NUM_MAX - 1) : (npuNumInter - 1);
    for (u32 i = 0; i < NPU_NUM_MAX_BITWIDTH; i++) {
        npuNumInter |= (npuNumInter >> 1);
    }
    npuNumInter++;
    npuNumInter = npuNumInter < NPU_NUM_MIN ? NPU_NUM_MIN : npuNumInter;
    // bufferSize必须为4k的倍数
    if (initAttrInter->bufferSize > 0) {
        bufferSizeInter = ((initAttrInter->bufferSize - 1) / BUFFER_SIZE_UNIT + 1) * BUFFER_SIZE_UNIT;
    }
    // 初始化pingmesh实例,利用用户输入的ip地址构造HcclIpAddress类
    HcclIpAddress ipAddr;
    HcclResult ret = HCCL_SUCCESS;
    if (initAttrInter->mode == HCCN_RPING_MODE_ROCE) {
        if (!HcclIpAddress::IsIPv4(std::string(initAttrInter->ipAddr))
            && !HcclIpAddress::IsIPv6(std::string(initAttrInter->ipAddr))) {
            HCCL_ERROR(
                "[HccnRpingInitInterWithCfg] invalid ip: %s, bufferSize:%u", initAttrInter->ipAddr, bufferSizeInter);
            return HCCN_E_PARA;
        }
        ipAddr = HcclIpAddress(std::string(initAttrInter->ipAddr));
        ret = rpingInter->HccnRpingInit(
            devLogicIdInter, LINK_TYPE_MODE_ROCE, ipAddr, initAttrInter->port, npuNumInter, bufferSizeInter,
            initAttrInter->sl, initAttrInter->tc, initCfgInter->ub.taMode, initCfgInter->ub.tpMode);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[HccnRpingInitInterWithCfg] init fail, bufferSize:%u", bufferSizeInter);
            return HCCN_E_FAIL;
        }
        ipAddrDesInter = ipAddr.GetReadableIP();
    }
    const char* socNamePtr = aclrtGetSocName();
    CHK_PRT_RET(socNamePtr == nullptr, HCCL_ERROR("[HccnRpingInitInterWithCfg]socNamePtr is null."), HCCN_E_PARA);
    // UB 模式在非 V2 形态上不支持: 显式报错, 避免静默跳过初始化导致"假成功"
    if (initAttrInter->mode == HCCN_RPING_MODE_UB && !IsSupportHCCLV2(socNamePtr)) {
        HCCL_ERROR("[HccnRpingInitInterWithCfg]UB mode not supported on soc[%s].", socNamePtr);
        return HCCN_E_FAIL;
    }

    if (initAttrInter->mode == HCCN_RPING_MODE_UB && IsSupportHCCLV2(socNamePtr)) {
        if (!HcclIpAddress::IsEID(std::string(initAttrInter->eid))) {
            HCCL_ERROR(
                "[HccnRpingInitInterWithCfg] invalid eid: %s, bufferSize:%u", initAttrInter->eid, bufferSizeInter);
            return HCCN_E_PARA;
        }
        Eid eid = HcclIpAddress::StrToEID(std::string(initAttrInter->eid));
        ipAddr = HcclIpAddress(eid);
        ret = rpingInter->HccnRpingInit(
            devLogicIdInter, LINK_TYPE_MODE_UB, ipAddr, initAttrInter->port, npuNumInter, bufferSizeInter,
            initAttrInter->sl, initAttrInter->tc, initCfgInter->ub.taMode, initCfgInter->ub.tpMode);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[HccnRpingInitInterWithCfg] init fail, bufferSize:%u, taMode:%d tpMode:%d, ret:%d", bufferSizeInter,
                initCfgInter->ub.taMode, initCfgInter->ub.tpMode, ret);
            return HCCN_E_FAIL;
        }
        ipAddrDesInter = ipAddr.Describe();
    }
    HCCL_RUN_INFO("[HccnRpingInitInterWithCfg]bufferSize:%u, ret:%d", bufferSizeInter, ret);
    return HCCN_SUCCESS;
}

HccnResult HccnRpingInit(uint32_t devLogicId, HccnRpingInitAttr* initAttr, HccnRpingCtx* rpingCtx)
{
    // 校验入参
    CHK_PRT_RET(initAttr == nullptr, HCCL_ERROR("[HccnRpingInit]initAttr is null."), HCCN_E_PARA);
    CHK_PRT_RET(
        initAttr->mode == PING_MODE_IPoUB || initAttr->mode >= HCCN_RPING_MODE_RESERVED,
        HCCL_ERROR("[HccnRpingInit]LinkMode[%d] not support.", initAttr->mode), HCCN_E_PARA);
    CHK_PRT_RET(rpingCtx == nullptr, HCCL_ERROR("[HccnRpingInit]rpingCtx is null."), HCCN_E_PARA);
    // 初始化ra接口
    CHK_PRT_RET(
        DlRaFunction::GetInstance().DlRaFunctionInit() != HCCL_SUCCESS,
        HCCL_ERROR("[HccnRpingInit]dlrafunction failed."), HCCN_E_FAIL);
    std::string initAttrDes;
    if (initAttr->mode == HCCN_RPING_MODE_ROCE) {
        initAttrDes = std::string(initAttr->ipAddr).c_str();
    }
    const char* socNamePtr = aclrtGetSocName();
    CHK_PRT_RET(socNamePtr == nullptr, HCCL_ERROR("[HccnRpingInit]socNamePtr is null."), HCCN_E_PARA);
    if (initAttr->mode == HCCN_RPING_MODE_UB && IsSupportHCCLV2(socNamePtr)) {
        initAttrDes = std::string(initAttr->eid).c_str();
    }
    HCCL_RUN_INFO(
        "[HccnRpingInit]devLogicId:%u, mode:%d port:%u npuNum:%u bufferSize:%u sl:%u tc:%u ip:%s", devLogicId,
        initAttr->mode, initAttr->port, initAttr->npuNum, initAttr->bufferSize, initAttr->sl, initAttr->tc,
        initAttrDes);
    // 获取device id
    s32 currDevLogicId = 0;
    HcclResult ret = hrtGetDeviceRefresh(&currDevLogicId);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[HccnRpingInit]cannot get device logic id, ret[%d].", ret), HCCN_E_FAIL);

    // 判断devLogicId与pingmesh所记录的是否一致
    if (devLogicId != static_cast<u32>(currDevLogicId)) {
        HCCL_ERROR(
            "[HccnRpingInit]Input device logicId[%u] don't match current logicId[%d].", devLogicId, currDevLogicId);
        return HCCN_E_FAIL;
    }
    // 构造一个pingmesh实例, 用于存放初始化的pingmesh信息
    PingMesh* rping = new (std::nothrow) PingMesh();
    CHK_PRT_RET(rping == nullptr, HCCL_ERROR("[HccnRpingInit]rping alloc failed."), HCCN_E_MEM);
    rping->init(initAttr); // 定义现在pingmesh的mode

    u32 npuNum = 0;
    std::string ipAddrDes;

    if (HccnRpingInitInter(devLogicId, initAttr, rping, npuNum, ipAddrDes) != 0) {
        HCCL_ERROR(
            "[HccnRpingInit]Pingmesh init failed, devLogicId:%u, port:%u, npuNum:%u, sl:%u"
            " tc:%u ip:%s",
            devLogicId, initAttr->port, npuNum, initAttr->sl, initAttr->tc, ipAddrDes);
        delete rping; // 初始化失败释放内存
        return HCCN_E_FAIL;
    }

    HCCL_RUN_INFO(
        "[HccnRpingInit]Pingmesh init success, devLogicId:%u, mode:%d port:%u npuNum:%u sl:%u"
        " tc:%u ip:%s",
        devLogicId, initAttr->mode, initAttr->port, npuNum, initAttr->sl, initAttr->tc, ipAddrDes);
    // 记录pingmesh指针
    *rpingCtx = rping;
    return HCCN_SUCCESS;
}

HccnResult HccnRpingInitWithCfg(
    uint32_t devLogicId, HccnRpingInitAttr* initAttr, HccnRpingInitCfg* initCfg, HccnRpingCtx* rpingCtx)
{
    CHK_PRT_RET(initAttr == nullptr, HCCL_ERROR("[HccnRpingInitWithCfg]initAttr is null."), HCCN_E_PARA);
    // 新接口 mode 应为 ROCE 或 UB;UB_RC_TP 是老接口值,新接口由 initCfg.ub.taMode 表达 RC,不应传入
    CHK_PRT_RET(
        initAttr->mode != HCCN_RPING_MODE_ROCE && initAttr->mode != HCCN_RPING_MODE_UB,
        HCCL_ERROR(
            "[HccnRpingInitWithCfg]LinkMode[%d] not support, use HCCN_RPING_MODE_UB or HCCN_RPING_MODE_ROCE.",
            initAttr->mode),
        HCCN_E_PARA);
    CHK_PRT_RET(initCfg == nullptr, HCCL_ERROR("[HccnRpingInitWithCfg]initCfg is null."), HCCN_E_PARA);
    CHK_PRT_RET(rpingCtx == nullptr, HCCL_ERROR("[HccnRpingInitWithCfg]rpingCtx is null."), HCCN_E_PARA);
    CHK_PRT_RET(
        initCfg->validCfgLen < sizeof(HccnRpingInitCfg),
        HCCL_ERROR(
            "[HccnRpingInitWithCfg]invalid validCfgLen[%u], expect no less than %zu.", initCfg->validCfgLen,
            sizeof(HccnRpingInitCfg)),
        HCCN_E_PARA);
    // ROCE 模式不使用 taMode/tpMode, 必须保持默认组合(UM/UTP), 显式拒绝歧义配置
    CHK_PRT_RET(
        initAttr->mode == HCCN_RPING_MODE_ROCE
            && (initCfg->ub.taMode != HCCN_RPING_UB_UM || initCfg->ub.tpMode != HCCN_RPING_UB_UTP),
        HCCL_ERROR(
            "[HccnRpingInitWithCfg]ROCE mode does not take taMode/tpMode, taMode[%d] tpMode[%d] should be default "
            "UM/UTP.",
            initCfg->ub.taMode, initCfg->ub.tpMode),
        HCCN_E_PARA);
    CHK_PRT_RET(
        initCfg->ub.taMode != HCCN_RPING_UB_RC && initCfg->ub.taMode != HCCN_RPING_UB_UM,
        HCCL_ERROR("[HccnRpingInitWithCfg]invalid taMode[%d].", initCfg->ub.taMode), HCCN_E_PARA);
    CHK_PRT_RET(
        initCfg->ub.tpMode != HCCN_RPING_UB_RTP && initCfg->ub.tpMode != HCCN_RPING_UB_CTP
            && initCfg->ub.tpMode != HCCN_RPING_UB_UTP,
        HCCL_ERROR("[HccnRpingInitWithCfg]invalid tpMode[%d].", initCfg->ub.tpMode), HCCN_E_PARA);
    // 组合校验(仅 UB 模式生效):UM 仅支持 UTP(hccp 层 RsPingCommonImportJetty 硬编码 UTP);RC 支持 RTP/CTP
    CHK_PRT_RET(
        initAttr->mode == HCCN_RPING_MODE_UB && initCfg->ub.taMode == HCCN_RPING_UB_UM
            && initCfg->ub.tpMode != HCCN_RPING_UB_UTP,
        HCCL_ERROR(
            "[HccnRpingInitWithCfg]UM only supports UTP, taMode[%d] tpMode[%d] not support.", initCfg->ub.taMode,
            initCfg->ub.tpMode),
        HCCN_E_PARA);
    // 组合校验(仅 UB 模式生效):RC 仅支持 RTP/CTP(下游 RsPingUrmaGetLocalTpInfo 将非 CTP 一律按 RTP 查询)
    CHK_PRT_RET(
        initAttr->mode == HCCN_RPING_MODE_UB && initCfg->ub.taMode == HCCN_RPING_UB_RC
            && initCfg->ub.tpMode == HCCN_RPING_UB_UTP,
        HCCL_ERROR(
            "[HccnRpingInitWithCfg]RC only supports RTP/CTP, taMode[%d] tpMode[%d] not support.", initCfg->ub.taMode,
            initCfg->ub.tpMode),
        HCCN_E_PARA);
    CHK_PRT_RET(
        DlRaFunction::GetInstance().DlRaFunctionInit() != HCCL_SUCCESS,
        HCCL_ERROR("[HccnRpingInitWithCfg]dlrafunction failed."), HCCN_E_FAIL);
    std::string initAttrDes;
    if (initAttr->mode == HCCN_RPING_MODE_ROCE) {
        initAttrDes = std::string(initAttr->ipAddr).c_str();
    }
    const char* socNamePtr = aclrtGetSocName();
    CHK_PRT_RET(socNamePtr == nullptr, HCCL_ERROR("[HccnRpingInitWithCfg]socNamePtr is null."), HCCN_E_PARA);
    if (initAttr->mode == HCCN_RPING_MODE_UB && IsSupportHCCLV2(socNamePtr)) {
        initAttrDes = std::string(initAttr->eid).c_str();
    }
    HCCL_RUN_INFO(
        "[HccnRpingInitWithCfg]devLogicId:%u, mode:%d taMode:%d tpMode:%d port:%u npuNum:%u "
        "bufferSize:%u sl:%u tc:%u ip:%s",
        devLogicId, initAttr->mode, initCfg->ub.taMode, initCfg->ub.tpMode, initAttr->port, initAttr->npuNum,
        initAttr->bufferSize, initAttr->sl, initAttr->tc, initAttrDes.c_str());
    s32 currDevLogicId = 0;
    HcclResult ret = hrtGetDeviceRefresh(&currDevLogicId);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[HccnRpingInitWithCfg]cannot get device logic id, ret[%d].", ret),
        HCCN_E_FAIL);

    if (devLogicId != static_cast<u32>(currDevLogicId)) {
        HCCL_ERROR(
            "[HccnRpingInitWithCfg]Input device logicId[%u] don't match current logicId[%d].", devLogicId,
            currDevLogicId);
        return HCCN_E_FAIL;
    }
    PingMesh* rping = new (std::nothrow) PingMesh();
    CHK_PRT_RET(rping == nullptr, HCCL_ERROR("[HccnRpingInitWithCfg]rping alloc failed."), HCCN_E_MEM);
    rping->initWithCfg(initAttr, initCfg); // 记录 mode + taMode + tpMode

    u32 npuNum = 0;
    std::string ipAddrDes;

    if (HccnRpingInitInterWithCfg(devLogicId, initAttr, initCfg, rping, npuNum, ipAddrDes) != 0) {
        HCCL_ERROR(
            "[HccnRpingInitWithCfg]Pingmesh init failed, devLogicId:%u, port:%u, npuNum:%u, sl:%u"
            " tc:%u ip:%s",
            devLogicId, initAttr->port, npuNum, initAttr->sl, initAttr->tc, ipAddrDes.c_str());
        delete rping;
        return HCCN_E_FAIL;
    }

    HCCL_RUN_INFO(
        "[HccnRpingInitWithCfg]Pingmesh init success, devLogicId:%u, mode:%d taMode:%d tpMode:%d "
        "port:%u npuNum:%u sl:%u tc:%u ip:%s",
        devLogicId, initAttr->mode, initCfg->ub.taMode, initCfg->ub.tpMode, initAttr->port, npuNum, initAttr->sl,
        initAttr->tc, ipAddrDes.c_str());
    *rpingCtx = rping;
    return HCCN_SUCCESS;
}

HccnResult HccnRpingDeinit(HccnRpingCtx rpingCtx)
{
    // 初始化ra接口
    CHK_PRT_RET(
        DlRaFunction::GetInstance().DlRaFunctionInit() != HCCL_SUCCESS,
        HCCL_ERROR("[HccnRpingDeinit]dlrafunction failed."), HCCN_E_FAIL);
    // 校验指针，并将其转换为pingmesh指针
    CHK_PRT_RET(rpingCtx == nullptr, HCCL_ERROR("[HccnRpingDeinit]rpingCtx is null."), HCCN_E_PARA);
    PingMesh* rping = static_cast<PingMesh*>(rpingCtx);
    // 获取device id
    s32 devLogicId = 0;
    HcclResult ret = hrtGetDeviceRefresh(&devLogicId);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[HccnRpingDeinit]cannot get device logic id, ret[%d].", ret), HCCN_E_FAIL);

    // 判断devLogicId与pingmesh所记录的是否一致
    if (devLogicId != rping->GetDeviceLogicId()) {
        HCCL_ERROR(
            "[HccnRpingDeinit]curr devId[%d] don't match record logicId[%d].", devLogicId, rping->GetDeviceLogicId());
        return HCCN_E_PARA;
    }

    ret = rping->HccnRpingDeinit(static_cast<u32>(devLogicId));
    delete rping; // 先释放资源
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HccnRpingDeinit]Device[%d] deinit fail, ret[%d]", devLogicId, ret);
        return HCCN_E_FAIL;
    }

    HCCL_RUN_INFO("[HccnRpingDeinit]Device[%d] deinit success", devLogicId);
    return HCCN_SUCCESS;
}

HccnResult HccnRpingAddTarget(HccnRpingCtx rpingCtx, uint32_t targetNum, HccnRpingTargetInfo* target)
{
    HccnRpingAddTargetConfig config;
    config.connectTimeout = HCCN_RPING_DEFAULT_TIMEOUT;
    return HccnRpingAddTargetWithCfg(rpingCtx, targetNum, target, &config);
}

inline HccnResult HccnRpingInitInputTargetAttr(HccnRpingTargetInfo* targetInter, RpingInput* inputInter, uint32_t& n)
{
    u32 addressType = 0;
    HcclResult addrTypeRet = GetAddrType(&addressType);
    if (addrTypeRet != HCCL_SUCCESS) {
        HCCL_ERROR("[HccnRpingInitInputTargetAttr]GetAddrType Fail ret %d", addrTypeRet);
        return HCCN_E_PARA;
    }
    if (addressType == HCCN_RPING_ADDR_TYPE_IP) {
        if (!HcclIpAddress::IsIPv4(std::string(targetInter[n].srcIp))
            && !HcclIpAddress::IsIPv6(std::string(targetInter[n].srcIp))) {
            HCCL_ERROR("[HccnRpingInitInputTargetAttr] invalid source ip.");
            return HCCN_E_PARA;
        }
        inputInter[n].sip = HcclIpAddress(std::string(targetInter[n].srcIp));
        if (!HcclIpAddress::IsIPv4(std::string(targetInter[n].dstIp))
            && !HcclIpAddress::IsIPv6(std::string(targetInter[n].dstIp))) {
            HCCL_ERROR("[HccnRpingInitInputTargetAttr] invalid destination ip.");
            return HCCN_E_PARA;
        }
        inputInter[n].dip = HcclIpAddress(std::string(targetInter[n].dstIp));
    }
    const char* socNamePtr = aclrtGetSocName();
    CHK_PRT_RET(socNamePtr == nullptr, HCCL_ERROR("[HccnRpingInitInputTargetAttr]socNamePtr is null."), HCCN_E_PARA);
    if (addressType == HCCN_RPING_ADDR_TYPE_EID && IsSupportHCCLV2(socNamePtr)) {
        if (!HcclIpAddress::IsEID(std::string(targetInter[n].srcEid))
            && !HcclIpAddress::IsEID(std::string(targetInter[n].dstEid))) {
            HCCL_ERROR("[HccnRpingInitInputTargetAttr] invalid eid");
            return HCCN_E_PARA;
        }
        inputInter[n].sip = HcclIpAddress(HcclIpAddress::StrToEID(std::string(targetInter[n].srcEid)));
        inputInter[n].dip = HcclIpAddress(HcclIpAddress::StrToEID(std::string(targetInter[n].dstEid)));
    }
    return HCCN_SUCCESS;
}

inline HccnResult HccnRpingInitTargetAttr(HccnRpingTargetInfo* targetInter, RpingInput* inputInter, uint32_t& n)
{
    HccnResult ret = HccnRpingInitInputTargetAttr(targetInter, inputInter, n);
    if (ret != HCCN_SUCCESS) {
        HCCL_ERROR("[HccnRpingInitTargetAttr] invalid eid");
        return HCCN_E_PARA;
    }
    inputInter[n].srcPort = targetInter[n].srcPort;
    inputInter[n].sl = targetInter[n].sl;
    inputInter[n].tc = targetInter[n].tc;
    inputInter[n].port = targetInter[n].port;
    inputInter[n].len = targetInter[n].payloadLen;
    inputInter[n].addrType = targetInter[n].addrType;

    return HCCN_SUCCESS;
}

HccnResult HccnRpingAddTargetV2(
    HccnRpingCtx rpingCtx, uint32_t targetNum, HccnRpingTargetInfo* target, HccnRpingAddTargetConfig* config)
{
    return HccnRpingAddTargetWithCfg(rpingCtx, targetNum, target, config);
}

HccnResult HccnRpingAddTargetWithCfg(
    HccnRpingCtx rpingCtx, uint32_t targetNum, HccnRpingTargetInfo* target, HccnRpingAddTargetConfig* config)
{
    // 校验入参
    CHK_PRT_RET(config == nullptr, HCCL_ERROR("[HccnRpingAddTargetWithCfg]config is null."), HCCN_E_PARA);
    // 超时设置判断 小于1ms,大于1h
    if (config->connectTimeout < HCCN_RPING_MIN_TIMEOUT || config->connectTimeout > HCCN_RPING_MAX_TIMEOUT) {
        HCCL_ERROR(
            "[HCCN][HccnRpingAddTargetWithCfg] timeout [%u ms], need to be in the range 1 to 3600000",
            config->connectTimeout);
        return HCCN_E_PARA;
    }
    // 校验入参
    CHK_PRT_RET(target == nullptr, HCCL_ERROR("[HccnRpingAddTargetWithCfg]target is null."), HCCN_E_PARA);
    CHK_PRT_RET(
        targetNum > TARGET_NUM_MAX,
        HCCL_ERROR("[HccnRpingAddTargetWithCfg]targetNum[%u] is more than %u!", targetNum, TARGET_NUM_MAX),
        HCCN_E_PARA);
    // 校验指针，并将其转换为pingmesh指针
    CHK_PRT_RET(rpingCtx == nullptr, HCCL_ERROR("[HccnRpingAddTargetWithCfg]rpingCtx is null."), HCCN_E_PARA);
    HCCL_DEBUG("[HccnRpingAddTargetWithCfg]targetNum:[%u] connectTimeout:[%u ms]", targetNum, config->connectTimeout);
    PingMesh* rping = static_cast<PingMesh*>(rpingCtx);
    // RC 模式 client jetty 仅支持绑定 1 个 target jetty(urma 单绑定槽位), 单次 add 仅允许 1 个 target
    CHK_PRT_RET(
        rping->GetTaMode() == HCCN_RPING_UB_RC && targetNum != 1,
        HCCL_ERROR("[HccnRpingAddTargetWithCfg]RC mode only supports 1 target per add, targetNum[%u].", targetNum),
        HCCN_E_PARA);
    // 获取device id
    s32 devLogicId = 0;
    HcclResult ret = hrtGetDeviceRefresh(&devLogicId);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[HccnRpingAddTargetWithCfg]cannot get device logic id, ret[%d].", ret),
        HCCN_E_FAIL);

    // 判断devLogicId与pingmesh所记录的是否一致
    if (devLogicId != rping->GetDeviceLogicId()) {
        HCCL_ERROR(
            "[HccnRpingAddTargetWithCfg]curr devId[%d] don't match record logicId[%d].", devLogicId,
            rping->GetDeviceLogicId());
        return HCCN_E_PARA;
    }
    HCCL_DEBUG("[HccnRpingAddTargetWithCfg] device id is %d.", devLogicId);

    // 将入参转换为内部接口可以使用的类型
    RpingInput* input = new (std::nothrow) RpingInput[targetNum];
    CHK_PRT_RET(input == nullptr, HCCL_ERROR("[HccnRpingAddTargetWithCfg]memory alloc failed."), HCCN_E_MEM);
    for (uint32_t m = 0; m < targetNum; m++) {
        HccnResult res = HccnRpingInitTargetAttr(target, input, m);
        if (res != HCCN_SUCCESS) {
            delete[] input;
            HCCL_ERROR("[HccnRpingAddTargetWithCfg]init target attr fail, ret[%d].", res);
            return HCCN_E_PARA;
        }
        s32 sRet = memcpy_s(input[m].payload, input[m].len, target[m].payload, target[m].payloadLen);
        if (sRet != EOK) {
            HCCL_ERROR(
                "[HccnRpingAddTargetWithCfg]memcpy payload fail. errorno[%d] params:dstMaxSize[%u] srclen[%d]", sRet,
                input[m].len, target[m].payloadLen);
            delete[] input;
            return HCCN_E_MEM;
        }
    }
    ret = rping->HccnRpingAddTarget(static_cast<u32>(devLogicId), targetNum, input, config);
    if (ret != 0) {
        HCCL_ERROR(
            "[HccnRpingAddTarget]Device[%d] add targetNum %u fail, ret[%d], mode:%d taMode:%d tpMode:%d.", devLogicId,
            targetNum, ret, rping->GetMode(), rping->GetTaMode(), rping->GetTpMode());
        delete[] input;
        return HCCN_E_FAIL;
    }

    delete[] input;
    HCCL_RUN_INFO(
        "[HccnRpingAddTargetWithCfg]Device[%d] add targetNum %u success, mode:%d taMode:%d tpMode:%d.", devLogicId,
        targetNum, rping->GetMode(), rping->GetTaMode(), rping->GetTpMode());
    return HCCN_SUCCESS;
}

HccnResult HccnRpingRemoveTarget(HccnRpingCtx rpingCtx, uint32_t targetNum, HccnRpingTargetInfo* target)
{
    // 校验入参
    CHK_PRT_RET(target == nullptr, HCCL_ERROR("[HccnRpingRemoveTarget]target is null."), HCCN_E_PARA);
    CHK_PRT_RET(
        targetNum > TARGET_NUM_MAX,
        HCCL_ERROR("[HccnRpingRemoveTarget]targetNum[%u] is more than %u!", targetNum, TARGET_NUM_MAX), HCCN_E_PARA);
    // 校验指针，并将其转换为pingmesh指针
    CHK_PRT_RET(rpingCtx == nullptr, HCCL_ERROR("[HccnRpingRemoveTarget]rpingCtx is null."), HCCN_E_PARA);
    HCCL_DEBUG("[HccnRpingRemoveTarget]targetNum:%u", targetNum);
    PingMesh* rping = static_cast<PingMesh*>(rpingCtx);
    // 获取device id
    s32 devLogicId = 0;
    HcclResult ret = hrtGetDeviceRefresh(&devLogicId);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[HccnRpingRemoveTarget]cannot get device logic id, ret[%d].", ret),
        HCCN_E_FAIL);

    // 判断devLogicId与pingmesh所记录的是否一致
    if (devLogicId != rping->GetDeviceLogicId()) {
        HCCL_ERROR(
            "[HccnRpingRemoveTarget]curr devId[%d] don't match record logicId[%d].", devLogicId,
            rping->GetDeviceLogicId());
        return HCCN_E_PARA;
    }
    HCCL_DEBUG("[HccnRpingRemoveTarget] device id is %d", devLogicId);

    // 将入参转换为内部接口可以使用的类型
    RpingInput* input = new (std::nothrow) RpingInput[targetNum];
    CHK_PRT_RET(input == nullptr, HCCL_ERROR("[HccnRpingRemoveTarget]memory alloc failed."), HCCN_E_FAIL);
    for (uint32_t in = 0; in < targetNum; in++) {
        HccnResult ret = HccnRpingInitInputTargetAttr(target, input, in);
        if (ret != HCCN_SUCCESS) {
            HCCL_ERROR("[HccnRpingRemoveTarget] invalid eid, ret[%d]", ret);
            delete[] input;
            return HCCN_E_PARA;
        }
        input[in].sl = target[in].sl;
        input[in].tc = target[in].tc;
        input[in].addrType = target[in].addrType;
    }

    ret = rping->HccnRpingRemoveTarget(static_cast<u32>(devLogicId), targetNum, input);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HccnRpingRemoveTarget]Device[%d] remove targetNum %u fail, ret[%d]", devLogicId, targetNum, ret);
        delete[] input;
        return HCCN_E_FAIL;
    }

    delete[] input;
    HCCL_RUN_INFO("[HccnRpingRemoveTarget]Device[%d] remove targetNum %u success.", devLogicId, targetNum);
    return HCCN_SUCCESS;
}

inline void ConvertTargetState(uint32_t targetNum, int* state, HccnRpingAddTargetState* targetState)
{
    for (uint32_t i = 0; i < targetNum; i++) {
        switch (static_cast<RpingLinkState>(state[i])) {
            case RpingLinkState::CONNECTED:
                targetState[i] = HCCN_RPING_ADDTARGET_STATE_DONE;
                break;
            case RpingLinkState::CONNECTING:
                targetState[i] = HCCN_RPING_ADDTARGET_STATE_DOING;
                break;
            case RpingLinkState::DISCONNECTED:
                targetState[i] = HCCN_RPING_ADDTARGET_STATE_FAIL;
                break;
            case RpingLinkState::TIMEOUT:
                targetState[i] = HCCN_RPING_ADDTARGET_STATE_TIMEOUT;
                break;
            default:
                targetState[i] = HCCN_RPING_ADDTARGET_STATE_RESERVED;
                break;
        }
    }
}

HccnResult HccnRpingGetTarget(
    HccnRpingCtx rpingCtx, uint32_t targetNum, HccnRpingTargetInfo* target, HccnRpingAddTargetState* targetState)
{
    // 校验入参
    CHK_PRT_RET(target == nullptr, HCCL_ERROR("[HccnRpingGetTarget]target is null."), HCCN_E_PARA);
    CHK_PRT_RET(targetState == nullptr, HCCL_ERROR("[HccnRpingGetTarget]targetState is null."), HCCN_E_PARA);
    CHK_PRT_RET(
        targetNum > TARGET_NUM_MAX,
        HCCL_ERROR("[HccnRpingGetTarget]targetNum[%u] is more than %u!", targetNum, TARGET_NUM_MAX), HCCN_E_PARA);
    // 校验指针，并将其转换为pingmesh指针
    CHK_PRT_RET(rpingCtx == nullptr, HCCL_ERROR("[HccnRpingGetTarget]rpingCtx is null."), HCCN_E_PARA);
    HCCL_DEBUG("[HccnRpingGetTarget]targetNum:%u", targetNum);
    PingMesh* rping = static_cast<PingMesh*>(rpingCtx);
    // 获取device id
    s32 devLogicId = 0;
    HcclResult ret = hrtGetDeviceRefresh(&devLogicId);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[HccnRpingGetTarget]cannot get device logic id, ret[%d].", ret), HCCN_E_FAIL);

    // 判断devLogicId与pingmesh所记录的是否一致
    if (devLogicId != rping->GetDeviceLogicId()) {
        HCCL_ERROR(
            "[HccnRpingGetTarget]curr devId[%d] don't match record logicId[%d].", devLogicId,
            rping->GetDeviceLogicId());
        return HCCN_E_PARA;
    }
    HCCL_DEBUG("[HccnRpingGetTarget] device id is %d", devLogicId);

    // 将入参转换为内部接口可以使用的类型
    RpingInput* input = new (std::nothrow) RpingInput[targetNum];
    CHK_PRT_RET(input == nullptr, HCCL_ERROR("[HccnRpingGetTarget]memory alloc failed."), HCCN_E_MEM);
    for (uint32_t h = 0; h < targetNum; h++) {
        HccnResult ret = HccnRpingInitInputTargetAttr(target, input, h);
        if (ret != HCCN_SUCCESS) {
            HCCL_ERROR("[HccnRpingGetTarget] invalid eid");
            delete[] input;
            return HCCN_E_FAIL;
        }
        input[h].addrType = target[h].addrType;
    }
    int* state = new (std::nothrow) int[targetNum];
    if (state == nullptr) {
        HCCL_ERROR("[HccnRpingGetTarget]memory alloc failed.");
        delete[] input;
        return HCCN_E_MEM;
    }
    ret = rping->HccnRpingGetTarget(static_cast<u32>(devLogicId), targetNum, input, state);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HccnRpingGetTarget]Device[%d] get targetNum %u fail, ret[%d]", devLogicId, targetNum, ret);
        delete[] input;
        delete[] state;
        return HCCN_E_FAIL;
    }

    ConvertTargetState(targetNum, state, targetState);
    delete[] input;
    delete[] state;
    HCCL_RUN_INFO("[HccnRpingGetTarget]Device[%d] get targetNum %u success.", devLogicId, targetNum);
    return HCCN_SUCCESS;
}

HccnResult HccnRpingBatchPingStart(HccnRpingCtx rpingCtx, uint32_t pktNum, uint32_t interval, uint32_t timeout)
{
    // 校验指针，并将其转换为pingmesh指针
    CHK_PRT_RET(rpingCtx == nullptr, HCCL_ERROR("[HccnRpingBatchPingStart]rpingCtx is null."), HCCN_E_PARA);
    HCCL_DEBUG("[HccnRpingBatchPingStart]pktNum:[%u], interval:[%u s], timeout:[%u s]", pktNum, interval, timeout);
    PingMesh* rping = static_cast<PingMesh*>(rpingCtx);
    // 获取device id
    s32 devLogicId = 0;
    HcclResult ret = hrtGetDeviceRefresh(&devLogicId);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[HccnRpingBatchPingStart]cannot get device logic id, ret[%d].", ret),
        HCCN_E_PARA);

    // 判断devLogicId与pingmesh所记录的是否一致
    if (devLogicId != rping->GetDeviceLogicId()) {
        HCCL_ERROR(
            "[HccnRpingBatchPingStart]curr devId[%d] don't match record logicId[%d].", devLogicId,
            rping->GetDeviceLogicId());
        return HCCN_E_PARA;
    }

    ret = rping->HccnRpingBatchPingStart(static_cast<u32>(devLogicId), pktNum, interval, timeout);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[HccnRpingBatchPingStart]task start failed, devLogicId[%d], ret[%d], pktNum:[%u], interval:[%u s], "
            "timeout:[%u s].",
            devLogicId, ret, pktNum, interval, timeout),
        HCCN_E_FAIL);

    HCCL_RUN_INFO(
        "[HccnRpingBatchPingStart]task start success, devLogicId:[%d], pktNum:[%u], interval:[%u s], timeout:[%u s], "
        "mode:%d taMode:%d tpMode:%d.",
        devLogicId, pktNum, interval, timeout, rping->GetMode(), rping->GetTaMode(), rping->GetTpMode());
    return HCCN_SUCCESS;
}

HccnResult HccnRpingBatchPingStop(HccnRpingCtx rpingCtx)
{
    // 校验指针，并将其转换为pingmesh指针
    CHK_PRT_RET(rpingCtx == nullptr, HCCL_ERROR("[HccnRpingBatchPingStop]rpingCtx is null."), HCCN_E_PARA);
    PingMesh* rping = static_cast<PingMesh*>(rpingCtx);
    // 获取device id
    s32 devLogicId = 0;
    HcclResult ret = hrtGetDeviceRefresh(&devLogicId);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[HccnRpingBatchPingStop]cannot get device logic id, ret[%d].", ret),
        HCCN_E_PARA);

    // 判断devLogicId与pingmesh所记录的是否一致
    if (devLogicId != rping->GetDeviceLogicId()) {
        HCCL_ERROR(
            "[HccnRpingBatchPingStop]curr devId[%d] don't match record logicId[%d].", devLogicId,
            rping->GetDeviceLogicId());
        return HCCN_E_PARA;
    }

    ret = rping->HccnRpingBatchPingStop(static_cast<u32>(devLogicId));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[HccnRpingBatchPingStop]task stop failed, devLogicId[%d] ret[%d].", devLogicId, ret), HCCN_E_FAIL);

    HCCL_RUN_INFO("[HccnRpingBatchPingStop]task stop success, devLogicId[%d].", devLogicId);
    return HCCN_SUCCESS;
}

inline void ConvertResultState(uint32_t state, HccnRpingResultState& resultState)
{
    if (state == RPING_RESULT_STATE_VALID) {
        resultState = HCCN_RPING_RESULT_STATE_VALID;
    } else {
        resultState = HCCN_RPING_RESULT_STATE_INVALID;
    }
}

inline HccnResult ResultGetTarget(uint32_t& targetNumInter, HccnRpingTargetInfo* targetInter, RpingInput* inputInter)
{
    for (uint32_t k = 0; k < targetNumInter; k++) {
        HccnResult ret = HccnRpingInitInputTargetAttr(targetInter, inputInter, k);
        if (ret != HCCN_SUCCESS) {
            HCCL_ERROR("[ResultGetTarget] invalid eid, ret[%d].", ret);
            return HCCN_E_PARA;
        }
        inputInter[k].addrType = targetInter[k].addrType;
    }
    return HCCN_SUCCESS;
}

inline void PutResult(uint32_t& targetNumInter, HccnRpingResultInfo* resultInter, RpingOutput* outputInter)
{
    for (uint32_t i = 0; i < targetNumInter; i++) {
        ConvertResultState(outputInter[i].state, resultInter[i].state);
        if (outputInter[i].state != PingResultState::PING_RESULT_STATE_VALID) {
            HCCL_INFO("[HccnRpingGetResult]Target[%u]'s state is not valid, state[%d].", i, outputInter[i].state);
            continue;
        }
        resultInter[i].txPkt = outputInter[i].txPkt;
        resultInter[i].rxPkt = outputInter[i].rxPkt;
        resultInter[i].minRTT = outputInter[i].minRTT;
        resultInter[i].maxRTT = outputInter[i].maxRTT;
        resultInter[i].avgRTT = outputInter[i].avgRTT;
    }
}

HccnResult
HccnRpingGetResult(HccnRpingCtx rpingCtx, uint32_t targetNum, HccnRpingTargetInfo* target, HccnRpingResultInfo* result)
{
    // 校验入参
    CHK_PRT_RET(target == nullptr, HCCL_ERROR("[HccnRpingGetResult]target is null."), HCCN_E_PARA);
    CHK_PRT_RET(result == nullptr, HCCL_ERROR("[HccnRpingGetResult]result is null."), HCCN_E_PARA);
    CHK_PRT_RET(
        targetNum > TARGET_NUM_MAX,
        HCCL_ERROR("[HccnRpingGetResult]targetNum[%u] is more than %u!", targetNum, TARGET_NUM_MAX), HCCN_E_PARA);
    // 校验指针，并将其转换为pingmesh指针
    CHK_PRT_RET(rpingCtx == nullptr, HCCL_ERROR("[HccnRpingGetResult]rpingCtx is null."), HCCN_E_PARA);
    HCCL_DEBUG("[HccnRpingGetResult]targetNum:%u", targetNum);
    PingMesh* rping = static_cast<PingMesh*>(rpingCtx);
    // 获取device id
    s32 devLogicId = 0;
    HcclResult ret = hrtGetDeviceRefresh(&devLogicId);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[HccnRpingGetResult]cannot get device logic id, ret[%d].", ret), HCCN_E_FAIL);

    // 判断devLogicId与pingmesh所记录的是否一致
    if (devLogicId != rping->GetDeviceLogicId()) {
        HCCL_ERROR("[HccnRpingGetResult]curr devId[%d] not match record[%d].", devLogicId, rping->GetDeviceLogicId());
        return HCCN_E_PARA;
    }

    // 将入参转换为内部接口可以使用的类型
    RpingInput* input = new (std::nothrow) RpingInput[targetNum];
    CHK_PRT_RET(input == nullptr, HCCL_ERROR("[HccnRpingGetResult]memory alloc failed."), HCCN_E_MEM);
    RpingOutput* output = new (std::nothrow) RpingOutput[targetNum];
    if (output == nullptr) {
        delete[] input;
        HCCL_ERROR("[HccnRpingGetResult]memory alloc failed.");
        return HCCN_E_MEM;
    }
    HccnResult res = ResultGetTarget(targetNum, target, input);
    if (res != HCCN_SUCCESS) {
        delete[] input;
        delete[] output;
        return HCCN_E_FAIL;
    }
    ret = rping->HccnRpingGetResult(static_cast<u32>(devLogicId), targetNum, input, output);
    if (ret == HCCL_E_AGAIN) {
        delete[] input;
        delete[] output;
        return HCCN_E_AGAIN;
    }
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HccnRpingGetResult]Device[%d] get result fail, targetNum[%u] ret[%d]", devLogicId, targetNum, ret);
        delete[] input;
        delete[] output;
        return HCCN_E_FAIL;
    }
    PutResult(targetNum, result, output);
    delete[] input;
    delete[] output;
    HCCL_RUN_INFO("[HccnRpingGetResult]Device[%d] get result success, targetNum[%u].", devLogicId, targetNum);
    return HCCN_SUCCESS;
}

HccnResult HccnRpingGetPayload(HccnRpingCtx rpingCtx, void** payload, uint32_t* payloadLen)
{
    // 校验入参
    CHK_PRT_RET(payload == nullptr, HCCL_ERROR("[HccnRpingGetPayload]payload is null."), HCCN_E_PARA);
    CHK_PRT_RET(payloadLen == nullptr, HCCL_ERROR("[HccnRpingGetPayload]payloadLen is null."), HCCN_E_PARA);
    // 校验指针，并将其转换为pingmesh指针
    CHK_PRT_RET(rpingCtx == nullptr, HCCL_ERROR("[HccnRpingGetPayload]rpingCtx is null."), HCCN_E_PARA);
    PingMesh* rping = static_cast<PingMesh*>(rpingCtx);
    // 入口拦截：RC模式(RC+RTP/RC+CTP)不支持GetPayload（内层未实现payload head重填
    CHK_PRT_RET(
        rping->GetTaMode() == HCCN_RPING_UB_RC,
        HCCL_ERROR("[HccnRpingGetPayload]RC mode is not supported, devId[%d].", rping->GetDeviceLogicId()),
        HCCN_E_FAIL);
    // 获取device id
    s32 devLogicId = 0;
    HcclResult ret = hrtGetDeviceRefresh(&devLogicId);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[HccnRpingGetPayload]cannot get device logic id, ret[%d].", ret), HCCN_E_FAIL);

    // 判断devLogicId与pingmesh所记录的是否一致
    if (devLogicId != rping->GetDeviceLogicId()) {
        HCCL_ERROR("[HccnRpingGetPayload]curr devId[%d] not match record[%d].", devLogicId, rping->GetDeviceLogicId());
        return HCCN_E_PARA;
    }

    ret = rping->HccnRpingGetPayload(static_cast<u32>(devLogicId), payload, payloadLen, rping->GetMode());
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[HccnRpingGetPayload]Device[%d] get payload fail, ret[%d]", devLogicId, ret),
        HCCN_E_FAIL);

    HCCL_RUN_INFO("[HccnRpingGetPayload]Device[%d] get payload success, payloadLen[%u].", devLogicId, *payloadLen);
    return HCCN_SUCCESS;
}

HccnResult
HccnRpingGetCtxInfo(HccnRpingCtx* rpingCtx, HccnRpingCtxInfoType infoType, HccnRpingCtxInfo* info, uint32_t infoLen)
{
    CHK_PRT_RET(rpingCtx == nullptr, HCCL_ERROR("[HccnRpingGetCtxInfo]rpingCtx is null."), HCCN_E_PARA);
    CHK_PRT_RET(*rpingCtx == nullptr, HCCL_ERROR("[HccnRpingGetCtxInfo]*rpingCtx is null."), HCCN_E_PARA);
    CHK_PRT_RET(info == nullptr, HCCL_ERROR("[HccnRpingGetCtxInfo]info is null."), HCCN_E_PARA);
    CHK_PRT_RET(*info == nullptr, HCCL_ERROR("[HccnRpingGetCtxInfo]*info is null."), HCCN_E_PARA);
    PingMesh* rping = static_cast<PingMesh*>(*rpingCtx);
    s32 devLogicId = 0;
    HcclResult ret = hrtGetDeviceRefresh(&devLogicId);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[HccnRpingGetCtxInfo]cannot get device logic id, ret[%d].", ret), HCCN_E_FAIL);

    if (devLogicId != rping->GetDeviceLogicId()) {
        HCCL_ERROR(
            "[HccnRpingGetCtxInfo]curr devId[%d] don't match record logicId[%d].", devLogicId,
            rping->GetDeviceLogicId());
        return HCCN_E_PARA;
    }

    ret = rping->HccnRpingGetCtxInfo(infoType, info, infoLen);
    // 参数错误(infoType 不支持/infoLen 不足)透传为 HCCN_E_PARA,与 RFC 返回值表一致
    if (ret == HCCL_E_PARA) {
        HCCL_ERROR(
            "[HccnRpingGetCtxInfo]Device[%d] get ctx info fail, infoType[%d] ret[%d].", devLogicId, infoType, ret);
        return HCCN_E_PARA;
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[HccnRpingGetCtxInfo]Device[%d] get ctx info fail, infoType[%d] ret[%d].", devLogicId, infoType, ret),
        HCCN_E_FAIL);

    HCCL_RUN_INFO("[HccnRpingGetCtxInfo]Device[%d] get ctx info success, infoType[%d].", devLogicId, infoType);
    return HCCN_SUCCESS;
}

#ifdef __cplusplus
}
#endif // __cplusplus
