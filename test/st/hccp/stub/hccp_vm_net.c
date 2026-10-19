/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccp_vm_net.h"
#include <stddef.h>
#include <arpa/inet.h>

uint32_t HccpVmNicIp(uint32_t phyId, uint32_t iface)
{
    if (phyId > HCCP_VM_PHYID_MAX || iface > HCCP_VM_IFACE_MAX) {
        return 0;
    }
    uint32_t slot = phyId * 10u + iface;
    if (slot > 0xFFFFu) {
        return 0;
    }
    return HCCP_VM_VNET_BASE | slot;
}

struct in_addr HccpVmToLoopback(struct in_addr virtIp)
{
    uint32_t host = ntohl(virtIp.s_addr);
    if ((host & HCCP_VM_VNET_MASK) != HCCP_VM_VNET_BASE) {
        struct in_addr none;
        none.s_addr = htonl(INADDR_NONE);
        return none;
    }
    uint32_t slot = host & ~HCCP_VM_VNET_MASK;
    uint32_t phyId = slot / 10u;
    uint32_t iface = slot % 10u;
    if (phyId > HCCP_VM_PHYID_MAX || iface > HCCP_VM_IFACE_MAX) {
        struct in_addr none;
        none.s_addr = htonl(INADDR_NONE);
        return none;
    }
    uint32_t byte2 = (phyId == 0u) ? HCCP_VM_PHYID_ZERO_MAP : phyId;
    struct in_addr loopback;
    loopback.s_addr = htonl((127u << 24) | (byte2 << 8) | iface);
    return loopback;
}

void HccpVmForeachNic(const uint32_t* phyIds, uint32_t len, HccpVmNicVisitor visit, void* ctx)
{
    if (phyIds == NULL || visit == NULL) {
        return;
    }
    for (uint32_t p = 0; p < len; p++) {
        if (phyIds[p] > HCCP_VM_PHYID_MAX) {
            continue;
        }
        for (uint32_t iface = 0; iface < HCCP_VM_IFACE_NUM; iface++) {
            visit(phyIds[p], iface, ctx);
        }
    }
}
