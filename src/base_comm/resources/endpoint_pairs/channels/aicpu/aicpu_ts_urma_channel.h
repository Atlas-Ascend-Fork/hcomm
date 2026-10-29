/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_TS_URMA_CHANNEL_H
#define AICPU_TS_URMA_CHANNEL_H

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>
#include "../channel.h"
#include "aicpu_ts_channel_entity.h"

// Orion
#include "../../../../../../src/legacy/ascend950/unified_platform/resource/socket/socket.h"
#include "buffer_key.h"
#include "rma_connection.h"
#include "ub_mem_transport.h"
#include "dev_ub_connection.h"
#include "ub_local_notify.h"
#include "hcomm_adapter_hccp.h"

namespace hcomm {

class AicpuTsUrmaChannel : public Channel {
public:
    AicpuTsUrmaChannel(
        EndpointHandle endpointHandle, const HcommChannelDesc& channelDesc, CommEngine engine = COMM_ENGINE_RESERVED);
    ~AicpuTsUrmaChannel() override;

    HcclResult GetNotifyNum(uint32_t* notifyNum) const override;
    HcclResult GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos) override;
    ChannelStatus GetStatus() override;
    HcclResult UpdateMemInfo(HcommMemHandle* memHandles, uint32_t memHandleNum) override;
    std::vector<char> GetMemUpdatePack() override;

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
    HcclResult ResetLocalNotifies();

    // --------------------- 转换参数 ---------------------
    DevBaseAttr devBaseAttr_{};
    bool devBaseAttrReady_{false};

    // --------------------- 具体成员 ---------------------
    AicpuTsChannelEntity aicpuTsHelper_;
    std::unique_ptr<Hccl::UbMemTransport> memTransport_{nullptr};
    std::vector<std::unique_ptr<Hccl::DevUbConnection>> connections_{};
    std::atomic<bool> isFirstPrintChannelInfo_{true}; // 是否第一次打印通道建链信息，避免重复打印日志刷屏
};

} // namespace hcomm

#endif // AICPU_TS_URMA_CHANNEL_H
