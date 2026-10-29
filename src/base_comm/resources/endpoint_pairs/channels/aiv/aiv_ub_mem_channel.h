/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_UB_MEM_CHANNEL_H
#define AIV_UB_MEM_CHANNEL_H

#include <cstdint>
#include <memory>
#include "../channel.h"

// Orion
#include "../../../../../../src/legacy/ascend950/unified_platform/resource/socket/socket.h"
#include "buffer_key.h"
#include "aiv_ub_mem_transport.h"

namespace hcomm {

class AivUbMemChannel : public Channel {
public:
    AivUbMemChannel(
        EndpointHandle endpointHandle, const HcommChannelDesc& channelDesc, CommEngine engine = COMM_ENGINE_RESERVED);
    ~AivUbMemChannel() override;

    HcclResult GetNotifyNum(uint32_t* notifyNum) const override;
    HcclResult GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos) override;
    ChannelStatus GetStatus() override;
    HcclResult UpdateMemInfo(HcommMemHandle* memHandles, uint32_t memHandleNum) override;

    HcclResult Clean() override;
    HcclResult Resume() override;

private:
    HcclResult ParseInputParam() override;
    HcclResult StartListen() override;
    HcclResult BuildSocket() override;
    HcclResult BuildConnection() override;
    HcclResult BuildNotify() override;
    HcclResult BuildBuffer() override;

    void StopListen();

    // --------------------- 入参 ---------------------

    // --------------------- 具体成员 ---------------------
    std::unique_ptr<AivUbMemTransport> transport_{nullptr};
};

} // namespace hcomm

#endif // AIV_UB_MEM_CHANNEL_H
