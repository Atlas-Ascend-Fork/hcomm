/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log.h"
#include "hccl_net_dev_v1.h"
#include "adapter_rts_common.h"
#include "hccl_net_dev.h"
#include "dev_type_utils.h"

#ifdef __cplusplus
extern "C" {
#endif
HcclResult __attribute__((weak)) HcclNetDevOpenV2(const HcclNetDevInfos* info, HcclNetDev* netDev);
HcclResult __attribute__((weak)) HcclNetDevCloseV2(HcclNetDev netDev);
HcclResult __attribute__((weak)) HcclNetDevGetAddrV2(HcclNetDev netDev, HcclAddress* addr);
HcclResult __attribute__((weak)) HcclNetDevGetBusAddrV2(HcclDeviceId dstDevId, HcclAddress* busAddr);
HcclResult __attribute__((weak)) HcclNetDevGetNicAddrV2(int32_t devicePhyId, HcclAddress** addr, uint32_t* addrNum);
#ifdef __cplusplus
}
#endif

using namespace std;

HcclResult HcclNetDevOpen(const HcclNetDevInfos* info, HcclNetDev* netDev)
{
    CHK_PTR_NULL(netDev);
    CHK_PTR_NULL(info);
    auto isGe950 = hcomm::EqualToOrGreaterThan950();
    CHK_PRT_RET(!isGe950.has_value(), HCCL_ERROR("[%s] get device type failed.", __func__), HCCL_E_RUNTIME);
    if (*isGe950) {
        return HcclNetDevOpenV2(info, netDev);
    }
    return HcclNetDevOpenV1(info, netDev);
}

HcclResult HcclNetDevClose(HcclNetDev netDev)
{
    // 先销毁设备
    CHK_PTR_NULL(netDev);

    auto isGe950 = hcomm::EqualToOrGreaterThan950();
    CHK_PRT_RET(!isGe950.has_value(), HCCL_ERROR("[%s] get device type failed.", __func__), HCCL_E_RUNTIME);
    if (*isGe950) {
        return HcclNetDevCloseV2(netDev);
    }

    return HcclNetDevCloseV1(netDev);
}

HcclResult HcclNetDevGetAddr(HcclNetDev netDev, HcclAddress* addr)
{
    CHK_PTR_NULL(netDev);
    CHK_PTR_NULL(addr);

    auto isGe950 = hcomm::EqualToOrGreaterThan950();
    CHK_PRT_RET(!isGe950.has_value(), HCCL_ERROR("[%s] get device type failed.", __func__), HCCL_E_RUNTIME);
    if (*isGe950) {
        return HcclNetDevGetAddrV2(netDev, addr);
    }

    return HcclNetDevGetAddrV1(netDev, addr);
}

HcclResult HcclNetDevGetBusAddr(HcclDeviceId dstDevId, HcclAddress* busAddr)
{
    CHK_PTR_NULL(busAddr);

    auto isGe950 = hcomm::EqualToOrGreaterThan950();
    CHK_PRT_RET(!isGe950.has_value(), HCCL_ERROR("[%s] get device type failed.", __func__), HCCL_E_RUNTIME);
    if (*isGe950) {
        return HcclNetDevGetBusAddrV2(dstDevId, busAddr);
    }

    return HcclNetDevGetBusAddrV1(dstDevId, busAddr);
}

HcclResult HcclNetDevGetNicAddr(int32_t devicePhyId, HcclAddress** addr, uint32_t* addrNum)
{
    CHK_PTR_NULL(addrNum);
    CHK_PTR_NULL(addr);

    auto isGe950 = hcomm::EqualToOrGreaterThan950();
    CHK_PRT_RET(!isGe950.has_value(), HCCL_ERROR("[%s] get device type failed.", __func__), HCCL_E_RUNTIME);
    if (*isGe950) {
        return HcclNetDevGetNicAddrV2(devicePhyId, addr, addrNum);
    }

    return HcclNetDevGetNicAddrV1(devicePhyId, addr, addrNum);
}
