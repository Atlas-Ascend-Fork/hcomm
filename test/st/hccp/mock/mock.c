/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccp_vm_net.h"
#include <sys/epoll.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <dlfcn.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <net/if.h>
#include <ifaddrs.h>
#include "securec.h"
#include "dl_hal_function.h"
#include "hccp.h"
#include "ra_client_host.h"
#include "ra.h"
#include "ra_init.h"
#include "ra_rdma.h"
#include "ra_hdc.h"
#include "ra_hdc_rdma_notify.h"
#include "ra_hdc_rdma.h"
#include "ra_hdc_socket.h"
#include "ra_peer.h"
#include "ra_peer_socket.h"
#include "ra_rs_comm.h"
#include "ra_rs_err.h"

__attribute__((constructor)) static void mock_init(void) {}

int RsGetIfaddrsV2(struct InterfaceInfo interfaceInfos[], unsigned int* num, unsigned int phyId, bool isAll)
{
    *num = 3;
    for (unsigned int iface = 0; iface < 3; iface++) {
        interfaceInfos[iface].family = AF_INET;
        struct in_addr vmLoopback = HccpVmToLoopback((struct in_addr){.s_addr = htonl(HccpVmNicIp(phyId, iface))});
        interfaceInfos[iface].ifaddr.ip.addr = vmLoopback;
        char ifname[MAX_INTERFACE_NAME_LEN] = {0};
        snprintf(ifname, sizeof(ifname), "eth%u", iface);
        strcpy(interfaceInfos[iface].ifname, ifname);
        interfaceInfos[iface].scopeId = 0;
        interfaceInfos[iface].ifaddr.mask.s_addr = 0x000000FF;
    }
    return 0;
}

struct IfAppendCtx {
    struct ifaddrs* tail;
};

static void AppendLoopbackNic(uint32_t phyId, uint32_t iface, void* opaque)
{
    struct IfAppendCtx* ctx = (struct IfAppendCtx*)opaque;
    struct ifaddrs* node = (struct ifaddrs*)calloc(1, sizeof(struct ifaddrs));
    struct sockaddr_in* sa = (struct sockaddr_in*)calloc(1, sizeof(struct sockaddr_in));
    struct sockaddr_in* mask = (struct sockaddr_in*)calloc(1, sizeof(struct sockaddr_in));
    char* name = (char*)calloc(1, 24);
    if (node == NULL || sa == NULL || mask == NULL || name == NULL) {
        free(node);
        free(sa);
        free(mask);
        free(name);
        return;
    }
    uint32_t virtIp = HccpVmNicIp(phyId, iface);
    struct in_addr loopback = HccpVmToLoopback((struct in_addr){.s_addr = htonl(virtIp)});
    sa->sin_family = AF_INET;
    sa->sin_addr = loopback;
    mask->sin_family = AF_INET;
    mask->sin_addr.s_addr = htonl(0xFF000000u);
    struct in_addr virt;
    virt.s_addr = htonl(virtIp);
    snprintf(name, 24, "vmnic_%s", inet_ntoa(virt));
    node->ifa_name = name;
    node->ifa_addr = (struct sockaddr*)sa;
    node->ifa_netmask = (struct sockaddr*)mask;
    node->ifa_flags = (unsigned int)(IFF_UP | IFF_RUNNING);
    node->ifa_next = NULL;
    ctx->tail->ifa_next = node;
    ctx->tail = node;
}

int getifaddrs(struct ifaddrs** ifap)
{
    static int (*real_getifaddrs)(struct ifaddrs**) = NULL;
    if (!real_getifaddrs) {
        real_getifaddrs = (int (*)(struct ifaddrs**))dlsym(RTLD_NEXT, "getifaddrs");
    }
    int ret = real_getifaddrs ? real_getifaddrs(ifap) : -1;
    if (ret != 0 || ifap == NULL || *ifap == NULL) {
        return ret;
    }
    struct ifaddrs* tail = *ifap;
    while (tail->ifa_next != NULL) {
        tail = tail->ifa_next;
    }
    static const uint32_t vmPhyIds[2] = {1, 5};
    struct IfAppendCtx ctx;
    ctx.tail = tail;
    HccpVmForeachNic(vmPhyIds, 2, AppendLoopbackNic, &ctx);
    return ret;
}
