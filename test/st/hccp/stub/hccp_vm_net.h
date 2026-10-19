/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCP_VM_NET_H
#define HCCP_VM_NET_H

#include <stdint.h>
#include <netinet/in.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HCCP_VM_VNET_BASE ((uint32_t)0xC0A80000u)
#define HCCP_VM_VNET_MASK ((uint32_t)0xFFFF0000u)

#define HCCP_VM_PHYID_MIN 0u
#define HCCP_VM_PHYID_ZERO_MAP 100u
#define HCCP_VM_PHYID_MAX 255u
#define HCCP_VM_IFACE_MAX 255u

#define HCCP_VM_IFACE_SOCKET 0u
#define HCCP_VM_IFACE_H2D 1u
#define HCCP_VM_IFACE_ROCE 2u
#define HCCP_VM_IFACE_NUM 3u

uint32_t HccpVmNicIp(uint32_t phyId, uint32_t iface);

struct in_addr HccpVmToLoopback(struct in_addr virtIp);

typedef void (*HccpVmNicVisitor)(uint32_t phyId, uint32_t iface, void* ctx);

void HccpVmForeachNic(const uint32_t* phyIds, uint32_t len, HccpVmNicVisitor visit, void* ctx);

#ifdef __cplusplus
}
#endif

#endif
