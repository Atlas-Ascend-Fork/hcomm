/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_URMA_CHANNEL_H
#define AIV_URMA_CHANNEL_H

#include <cstdint>
#include <memory>
#include <vector>
#include "../channel.h"

// Orion
#include "aiv_urma_transport.h"
#include "aiv_channel_entity.h"
#include "buffer_key.h"
#include "resource/notify/ub_local_notify.h"
#include "resource/socket/socket.h"
#include "hcomm_adapter_hccp.h"

namespace hcomm {

struct UbConnBuildContext;

class AivUrmaChannel : public Channel {
public:
    AivUrmaChannel(
        EndpointHandle endpointHandle, const HcommChannelDesc& channelDesc, CommEngine engine = COMM_ENGINE_RESERVED);
    ~AivUrmaChannel() override;

    HcclResult GetNotifyNum(uint32_t* notifyNum) const override;
    HcclResult GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos) override;
    ChannelStatus GetStatus() override;
    HcclResult UpdateMemInfo(HcommMemHandle* memHandles, uint32_t memHandleNum) override;

    HcclResult Clean() override;
    HcclResult Resume() override;

    IChannelEntity* GetChannelEntity() override { return &aivChannelHelper_; }
    HcclResult BuildChannelEntityToDevice(void** devChannelPtr);

private:
    friend class ::AivChannelEntity;

    HcclResult ParseInputParam() override;
    HcclResult StartListen() override;
    HcclResult BuildSocket() override;
    HcclResult BuildConnection() override;
    HcclResult BuildNotify() override;
    HcclResult BuildBuffer() override;

    HcclResult BuildAttr();
    HcclResult CreateUbConnectionByProtocol(
        const UbConnBuildContext& ctx, std::unique_ptr<Hccl::DevUbConnection>& ubConn,
        Hccl::DevUbConnection::JettyMode jettyMode = Hccl::DevUbConnection::JettyMode::SELF_CREATE);
    HcclResult AcquireSharedJettyInBuildConnection(const UbConnBuildContext& ctx, Hccl::DevUbConnection* connection);
    static std::unique_ptr<Hccl::DevUbConnection> CreateSharedJettyConnection(
        Hccl::RdmaHandle rdmaHandle, const Hccl::IpAddress& locAddr, const Hccl::IpAddress& rmtAddr, u8 qosPre,
        Hccl::LinkProtocol protocol, u8 taTimeOut, u32 sqDepth);

    // --------------------- 转换参数 ---------------------
    DevBaseAttr devBaseAttr_{};

    // --------------------- 具体成员 ---------------------
    AivChannelEntity aivChannelHelper_{*this};
    std::unique_ptr<Hccl::AivUrmaTransport> transport_{nullptr};
    std::vector<std::unique_ptr<Hccl::DevUbConnection>> connections_{};

    // 共享 jetty 模式下从 JettyContext::Ctx 取得的 PI/CI device 内存指针，
    void* sharedSqPiPtr_{nullptr};
    void* sharedSqCiPtr_{nullptr};
    void* sharedCqPiPtr_{nullptr};
    void* sharedCqCiPtr_{nullptr};
};

} // namespace hcomm

#endif // AIV_URMA_CHANNEL_H
