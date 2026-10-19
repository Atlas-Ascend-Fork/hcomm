/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "hccp_system_test_base.h"
#include "hccp.h"
#include "hccp_ctx.h"
#include "hccp_ctx_data_plane.h"
#include <runtime/runtime/mem.h>
#include "rt_net_service.h"
#include "hccp_vm_net.h"
#include "hccp_test_helpers.h"

struct CaseEnv {
    bool netServiceOpened = false;
    struct RaInitConfig cfg = {};
    struct SocketInitInfoT sockInit = {};

    void* socketHandle = nullptr;
    struct SocketInfoT sock = {};
    void* rdmaHandle = nullptr;
    void* qpHandle = nullptr;

    void* ctxHandle = nullptr;
    void* sendCq = nullptr;
    void* recvCq = nullptr;
    void* sendLmem = nullptr;
    void* recvLmem = nullptr;
    void* remQp = nullptr;
    void* rmem = nullptr;

    void* rtBuf[4] = {nullptr, nullptr, nullptr, nullptr};
    char* heapBuf[2] = {nullptr, nullptr};
};

class HccpSystemTest : public HccpSystemTestBase {
public:
    void CaseCleanup(CaseEnv& env);
    int CaseOfflineInit(CaseEnv& env, int phyid, uint32_t iface, unsigned short port, const char* tag);
    int CaseOfflineInitNoRdv(CaseEnv& env, int phyid, uint32_t iface);
    int CasePeerInit(CaseEnv& env, int phyid, uint32_t iface, unsigned short port, const char* tag);
    int CaseRdevInit(CaseEnv& env, int phyid, int mode, unsigned int notifyType, uint32_t iface);
};
