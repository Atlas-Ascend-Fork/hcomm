/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hcomm_mem_alloc.h"

#include "acl/acl_rt.h"
#include "common/dev_type_utils_pub.h"
#include "hcomm_adapter_rts.h"
#include "log.h"
#include "rt_external.h"

namespace hcomm {
namespace {
#define ALIGN_SIZE(size, align) ({ (size) = (((size) + (align) - 1) / (align)) * (align); })

#ifndef ACL_DEV_ATTR_HD_CONNECT_TYPE
    // CANN 8.5.0 头文件未定义该属性，取值与 CANN 9.2.0 acl/acl_rt.h 保持一致
    constexpr aclrtDevAttr kHdConnectType = static_cast<aclrtDevAttr>(408U);
#else
    constexpr aclrtDevAttr kHdConnectType = ACL_DEV_ATTR_HD_CONNECT_TYPE;
    static_assert(static_cast<uint32_t>(kHdConnectType) == 408U, "ACL_DEV_ATTR_HD_CONNECT_TYPE changed");
#endif

#ifndef ACL_HOST_DEVICE_CONNECT_TYPE_PCIE
    // CANN 8.5.0 头文件未定义该枚举，取值与 CANN 9.2.0 acl/acl_rt.h 保持一致
    constexpr int64_t kConnectTypePcie = 0;
#else
    constexpr int64_t kConnectTypePcie = ACL_HOST_DEVICE_CONNECT_TYPE_PCIE;
    static_assert(kConnectTypePcie == 0, "ACL_HOST_DEVICE_CONNECT_TYPE_PCIE changed");
#endif

    HcclResult QueryH2DConnectType(int64_t& connectType)
    {
        int32_t deviceLogicId = 0;
        aclError aclRet = aclrtGetDevice(&deviceLogicId);
        CHK_PRT_RET(
            aclRet != ACL_SUCCESS, HCCL_ERROR("[%s] GetDevice failed, ret[%d]", __func__, aclRet), HCCL_E_RUNTIME);

        // H2D 协议类型；老版本Runtime不支持该属性时按非PCIe处理，回退ACL注册
        aclError connectRet = aclrtGetDeviceInfo(static_cast<uint32_t>(deviceLogicId), kHdConnectType, &connectType);
        if (connectRet == ACL_ERROR_RT_FEATURE_NOT_SUPPORT || connectRet == ACL_ERROR_RT_PARAM_INVALID) {
            HCCL_WARNING(
                "[%s] ACL_DEV_ATTR_HD_CONNECT_TYPE not supported, deviceLogicId[%d], ret[%d], fallback to non-PCIe",
                __func__, deviceLogicId, static_cast<int32_t>(connectRet));
            connectType = -1;
            return HCCL_SUCCESS;
        }
        CHK_PRT_RET(
            connectRet != ACL_SUCCESS,
            HCCL_ERROR(
                "[%s] aclrtGetDeviceInfo failed, deviceLogicId[%d], ret[%d]", __func__, deviceLogicId,
                static_cast<int32_t>(connectRet)),
            HCCL_E_RUNTIME);
        return HCCL_SUCCESS;
    }

    struct HostMapMode {
        int64_t connectType = 0;
        bool isPcie = false;
        bool usePciAtu = false;
    };

    HcclResult ValidateHostMem(const char* caller, const CommMem* hostMem, bool needSize)
    {
        CHK_PRT_RET(hostMem == nullptr, HCCL_ERROR("[%s] hostMem is nullptr", caller), HCCL_E_PTR);
        CHK_PRT_RET(
            hostMem->type != COMM_MEM_TYPE_HOST,
            HCCL_ERROR("[%s] invalid hostMem type[%d]", caller, static_cast<int32_t>(hostMem->type)), HCCL_E_PARA);
        CHK_PRT_RET(
            hostMem->addr == nullptr || (needSize && hostMem->size == 0),
            HCCL_ERROR("[%s] invalid hostMem addr[%p] size[%llu]", caller, hostMem->addr, hostMem->size), HCCL_E_PARA);
        return HCCL_SUCCESS;
    }

    HcclResult CheckPciAtuSymbols(const char* caller, bool needRegister)
    {
        bool registerSupported = rtHostRegisterByPciAtu != nullptr;
        bool unregisterSupported = rtHostUnRegisterByPciAtu != nullptr;
        CHK_PRT_RET(
            (needRegister && !registerSupported) || !unregisterSupported,
            HCCL_ERROR(
                "[%s] PCIe ATU runtime symbols missing, registerSupported[%d] unregisterSupported[%d]", caller,
                registerSupported, unregisterSupported),
            HCCL_E_NOT_SUPPORT);
        return HCCL_SUCCESS;
    }

    HcclResult ConvertMemMapType(HcommMemMapType mapType, bool usePciAtu, uint32_t& aclFlag)
    {
        switch (mapType) {
            case HCOMM_MEM_MAP_DEFAULT:
                aclFlag = ACL_HOST_REG_MAPPED;
                return HCCL_SUCCESS;
            case HCOMM_MEM_MAP_PINNED:
                // ATU接口当前仅接受ACL_HOST_REG_MAPPED，不支持锁页语义
                CHK_PRT_RET(
                    usePciAtu, HCCL_ERROR("[%s] PCIe ATU does not support pinned host memory registration", __func__),
                    HCCL_E_NOT_SUPPORT);
                aclFlag = static_cast<uint32_t>(ACL_HOST_REG_MAPPED | ACL_HOST_REG_PINNED);
                return HCCL_SUCCESS;
            default:
                HCCL_ERROR("[%s] invalid mapType[%d]", __func__, static_cast<int32_t>(mapType));
                return HCCL_E_PARA;
        }
    }

    HcclResult QueryHostMapMode(HostMapMode& mode)
    {
        CHK_RET(QueryH2DConnectType(mode.connectType));

        // 仅 PCIe 协议且芯片为 950/960 时使用 PCIe ATU 方式，其余场景使用 ACL 的 Host 内存注册接口
        mode.isPcie = (mode.connectType == kConnectTypePcie);
        mode.usePciAtu = false;
        if (!mode.isPcie) {
            return HCCL_SUCCESS;
        }

        // 芯片类型只查询一次，避免 EqualTo950/EqualTo960 分别取数
        auto devType = hcomm::devtype_detail::FetchDevType(__func__);
        CHK_PRT_RET(
            !devType.has_value(),
            HCCL_ERROR("[%s] get device type failed, connectType[%lld]", __func__, mode.connectType), HCCL_E_RUNTIME);
        bool is950 = (*devType == DevType::DEV_TYPE_950);
        bool is960 = (*devType == DevType::DEV_TYPE_960);
        mode.usePciAtu = is950 || is960;
        HCCL_INFO(
            "[%s] connectType[%lld] is950[%d] is960[%d] usePciAtu[%d]", __func__, mode.connectType, is950, is960,
            mode.usePciAtu);
        return HCCL_SUCCESS;
    }
} // namespace

HcommResult MemAlloc(void** ptr, size_t size)
{
    CHK_PTR_NULL(ptr);
    CHK_PRT_RET(size == 0, HCCL_ERROR("[%s] size is zero", __func__), HCCL_E_PARA);

    aclError ret = ACL_SUCCESS;
    int32_t deviceId = 0;
    ret = aclrtGetDevice(&deviceId);
    CHK_PRT_RET(ret != ACL_SUCCESS, HCCL_ERROR("[%s] GetDevice failed, ret[%d]", __func__, ret), HCCL_E_RUNTIME);

    aclrtPhysicalMemProp prop;
    prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
    prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
    prop.memAttr = ACL_HBM_MEM_HUGE;
    prop.location.id = deviceId;
    prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
    prop.reserve = 0;

    size_t allocSize = size;
    size_t granularity = 0;
    ret = aclrtMemGetAllocationGranularity(&prop, ACL_RT_MEM_ALLOC_GRANULARITY_RECOMMENDED, &granularity);
    CHK_PRT_RET(
        ret != ACL_SUCCESS || granularity == 0,
        HCCL_ERROR("[%s] GetAllocationGranularity failed, granularity[%llu], ret[%d]", __func__, granularity, ret),
        HCCL_E_RUNTIME);
    ALIGN_SIZE(allocSize, granularity);
    HCCL_INFO(
        "[%s] deviceId[%d], granularity[%llu], size[%llu], allocSize[%llu].", __func__, deviceId, granularity, size,
        allocSize);

    ret = aclrtReserveMemAddress(ptr, allocSize, 0, nullptr, 1);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR("[%s] ReserveMemAddress failed, virPtr[%p] size[%llu], ret[%d]", __func__, ptr, allocSize, ret),
        HCCL_E_RUNTIME);

    void* virPtr = *ptr;
    aclrtDrvMemHandle handle;
    ret = aclrtMallocPhysical(&handle, allocSize, &prop, 0);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR("[%s] MallocPhysical failed, size[%llu], ret[%d]", __func__, allocSize, ret);
        aclrtReleaseMemAddress(virPtr);
        return HCCL_E_RUNTIME;
    }
    HCCL_INFO("[%s] Start to MapMem virPtr[%p], handle[%p]", __func__, virPtr, handle);
    ret = aclrtMapMem(virPtr, allocSize, 0, handle, 0);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR(
            "[%s] MapMem virPtr[%p] size[%llu] handle[%p] failed, ret[%d]", __func__, virPtr, allocSize, handle, ret);
        aclrtFreePhysical(handle);
        aclrtReleaseMemAddress(virPtr);
        return HCCL_E_RUNTIME;
    }

    return HCCL_SUCCESS;
}

HcommResult MemFree(void* ptr)
{
    if (ptr == nullptr) {
        HCCL_DEBUG("[%s] virPtr is nullptr.", __func__);
        return HCCL_SUCCESS;
    }
    aclError ret = ACL_SUCCESS;
    aclrtDrvMemHandle handle;
    ret = aclrtMemRetainAllocationHandle(ptr, &handle);
    CHK_PRT_RET(
        ret != ACL_SUCCESS, HCCL_ERROR("[%s] RetainAllocationHandle virPtr[%p] failed, ret[%d]", __func__, ptr, ret),
        HCCL_E_RUNTIME);
    HCCL_INFO("[%s] Start to UnmapMem virPtr[%p], handle[%p]", __func__, ptr, handle);
    ret = aclrtUnmapMem(ptr);
    CHK_PRT_RET(
        ret != ACL_SUCCESS, HCCL_ERROR("[%s] UnmapMem virPtr[%p] failed, ret[%d]", __func__, ptr, ret), HCCL_E_RUNTIME);
    ret = aclrtFreePhysical(handle);
    CHK_PRT_RET(
        ret != ACL_SUCCESS, HCCL_ERROR("[%s] FreePhysical handle[%p] failed, ret[%d]", __func__, handle, ret),
        HCCL_E_RUNTIME);
    ret = aclrtReleaseMemAddress(ptr);
    CHK_PRT_RET(
        ret != ACL_SUCCESS, HCCL_ERROR("[%s] ReleaseMemAddress virPtr[%p] failed, ret[%d]", __func__, ptr, ret),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
}

HcommResult MemHostMap2Dev(const CommMem* hostMem, CommMem* devMem, HcommMemMapType mapType)
{
    CHK_PTR_NULL(devMem);
    CHK_RET(ValidateHostMem(__func__, hostMem, true));

    HostMapMode mode;
    CHK_RET(QueryHostMapMode(mode));

    uint32_t aclFlag = 0U;
    CHK_RET(ConvertMemMapType(mapType, mode.usePciAtu, aclFlag));

    // PCIe协议且芯片为950/960时使用RTS的PCIe ATU方式注册，其余场景使用ACL的Host内存注册接口
    void* devPtr = nullptr;
    if (mode.usePciAtu) {
        CHK_RET(CheckPciAtuSymbols(__func__, true));
        // ATU接口的flag当前未启用，固定传ACL_HOST_REG_MAPPED，不透传用户入参
        rtError_t rtRet = rtHostRegisterByPciAtu(hostMem->addr, hostMem->size, ACL_HOST_REG_MAPPED, &devPtr);
        CHK_PRT_RET(
            rtRet != RT_ERROR_NONE,
            HCCL_ERROR(
                "[%s] rtHostRegisterByPciAtu failed, addr[%p] size[%llu] mapType[%d] connectType[%lld] ret[%d]",
                __func__, hostMem->addr, hostMem->size, static_cast<int32_t>(mapType), mode.connectType,
                static_cast<int32_t>(rtRet)),
            HCCL_E_RUNTIME);
        if (devPtr == nullptr) {
            HCCL_ERROR("[%s] rtHostRegisterByPciAtu returned null devPtr, rollback registration", __func__);
            rtError_t unregRet = rtHostUnRegisterByPciAtu(hostMem->addr);
            if (unregRet != RT_ERROR_NONE) {
                HCCL_ERROR(
                    "[%s] rollback rtHostUnRegisterByPciAtu failed, hostAddr[%p] ret[%d]", __func__, hostMem->addr,
                    static_cast<int32_t>(unregRet));
            }
            return HCCL_E_RUNTIME;
        }
    } else {
        aclError registerRet = aclrtHostRegisterV2(hostMem->addr, hostMem->size, aclFlag);
        CHK_PRT_RET(
            registerRet != ACL_SUCCESS,
            HCCL_ERROR(
                "[%s] aclrtHostRegisterV2 failed, addr[%p] size[%llu] mapType[%d] connectType[%lld] ret[%d]", __func__,
                hostMem->addr, hostMem->size, static_cast<int32_t>(mapType), mode.connectType,
                static_cast<int32_t>(registerRet)),
            HCCL_E_RUNTIME);

        aclError getPtrRet = aclrtHostGetDevicePointer(hostMem->addr, &devPtr, 0);
        if (getPtrRet != ACL_SUCCESS || devPtr == nullptr) {
            HCCL_ERROR(
                "[%s] aclrtHostGetDevicePointer failed, addr[%p] devPtr[%p] connectType[%lld] ret[%d]", __func__,
                hostMem->addr, devPtr, mode.connectType, static_cast<int32_t>(getPtrRet));
            aclError unregisterRet = aclrtHostUnregister(hostMem->addr);
            if (unregisterRet != ACL_SUCCESS) {
                HCCL_ERROR(
                    "[%s] rollback aclrtHostUnregister failed, addr[%p] ret[%d]", __func__, hostMem->addr,
                    static_cast<int32_t>(unregisterRet));
            }
            return HCCL_E_RUNTIME;
        }
    }

    devMem->type = COMM_MEM_TYPE_DEVICE;
    devMem->addr = devPtr;
    devMem->size = hostMem->size;
    HCCL_INFO(
        "[%s] map success, hostAddr[%p] devAddr[%p] size[%llu] mapType[%d] connectType[%lld] isPcie[%d] usePciAtu[%d]",
        __func__, hostMem->addr, devMem->addr, devMem->size, static_cast<int32_t>(mapType), mode.connectType,
        mode.isPcie, mode.usePciAtu);
    return HCCL_SUCCESS;
}

HcommResult MemHostUnmapFromDev(const CommMem* hostMem)
{
    CHK_RET(ValidateHostMem(__func__, hostMem, false));

    HostMapMode mode;
    CHK_RET(QueryHostMapMode(mode));

    // PCIe协议且芯片为950/960时使用RTS的PCIe ATU方式注销，其余场景使用ACL的Host内存注销接口
    // 两种注销接口均以Host基地址（hVA）为入参
    if (mode.usePciAtu) {
        CHK_RET(CheckPciAtuSymbols(__func__, false));
        rtError_t rtRet = rtHostUnRegisterByPciAtu(hostMem->addr);
        CHK_PRT_RET(
            rtRet != RT_ERROR_NONE,
            HCCL_ERROR(
                "[%s] rtHostUnRegisterByPciAtu failed, hostAddr[%p] connectType[%lld] ret[%d]", __func__, hostMem->addr,
                mode.connectType, static_cast<int32_t>(rtRet)),
            HCCL_E_RUNTIME);
    } else {
        aclError aclRet = aclrtHostUnregister(hostMem->addr);
        CHK_PRT_RET(
            aclRet != ACL_SUCCESS,
            HCCL_ERROR(
                "[%s] aclrtHostUnregister failed, hostAddr[%p] connectType[%lld] ret[%d]", __func__, hostMem->addr,
                mode.connectType, static_cast<int32_t>(aclRet)),
            HCCL_E_RUNTIME);
    }

    HCCL_INFO(
        "[%s] unmap success, hostAddr[%p] connectType[%lld] isPcie[%d] usePciAtu[%d]", __func__, hostMem->addr,
        mode.connectType, mode.isPcie, mode.usePciAtu);
    return HCCL_SUCCESS;
}
} // namespace hcomm
