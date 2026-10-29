/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_TS_P2P_CHANNEL_H
#define AICPU_TS_P2P_CHANNEL_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "../channel.h"
#include "aicpu_ts_channel_entity.h"

#include "../../../../../../legacy/ascend950/unified_platform/resource/socket/socket.h"
#include "p2p_transport.h"
#include "p2p_connection.h"
#include "ipc_local_notify.h"
#include "aicpu_res_package_helper.h"

namespace hcomm {

class AicpuTsP2pChannel : public Channel {
public:
    AicpuTsP2pChannel(
        EndpointHandle endpointHandle, const HcommChannelDesc& channelDesc, CommEngine engine = COMM_ENGINE_RESERVED);
    ~AicpuTsP2pChannel() override;

    HcclResult GetNotifyNum(uint32_t* notifyNum) const override;
    HcclResult GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos) override;
    ChannelStatus GetStatus() override;
    HcclResult UpdateMemInfo(HcommMemHandle* memHandles, uint32_t memHandleNum) override;

    HcclResult Clean() override;
    HcclResult Resume() override;

    HcclResult H2DResPack(std::vector<char>& buffer);

    IChannelEntity* GetChannelEntity() override { return &aicpuTsHelper_; }

private:
    HcclResult ParseInputParam() override;
    HcclResult StartListen() override;
    HcclResult BuildSocket() override;
    HcclResult BuildConnection() override;
    HcclResult BuildNotify() override;
    HcclResult BuildBuffer() override;

    HcclResult BuildAttr();
    HcclResult PackOpData(std::vector<char>& data);
    HcclResult SetModuleDataName(Hccl::ModuleData& module, const std::string& name) const;

    AicpuTsChannelEntity aicpuTsHelper_;
    std::unique_ptr<Hccl::P2PTransport> memTransport_{nullptr};
    std::vector<std::unique_ptr<Hccl::P2PConnection>> connections_{};
};

} // namespace hcomm

#endif // AICPU_TS_P2P_CHANNEL_H
