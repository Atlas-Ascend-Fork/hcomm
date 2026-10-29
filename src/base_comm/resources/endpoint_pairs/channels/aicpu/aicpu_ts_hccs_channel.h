/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_TS_HCCS_CHANNEL_H
#define AICPU_TS_HCCS_CHANNEL_H
#include <memory>
#include <vector>
#include "hccl_mem_defs.h"
#include "transport_mem.h"
#include "hccl_res.h"
#include "hccl_socket.h"
#include "channel.h"
#include "aicpu_ts_channel_entity.h"
#include "channel_param.h"
#include "buffer.h"
#include "hccl_dispatcher_ctx.h"
#include "rma_buffer.h"
#include "hcomm_c_adpt.h"
#include "../../../endpoints/aicputs_hccs_endpoint.h"

namespace hcomm {
/**
 * @note 职责：Channel的AicpuTs通信引擎、HCCS协议的类派生
 */
class AicpuTsHccsChannel : public Channel {
public:
    struct HccsExchangeInfo {
        s32 memNum = 0;
    };
    AicpuTsHccsChannel(
        EndpointHandle endpointHandle, const HcommChannelDesc& channelDesc, CommEngine engine = COMM_ENGINE_RESERVED);
    ~AicpuTsHccsChannel() override;

    HcclResult GetNotifyNum(uint32_t* notifyNum) const override;
    HcclResult GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos) override;
    ChannelStatus GetStatus() override;

    HcclResult Clean() override;
    HcclResult Resume() override;

    std::shared_ptr<hccl::Transport> GetTransport() { return transport_; }

    // for launch channel kernel data
    HcclResult Serialize(std::shared_ptr<hccl::DeviceMem>& out) override;

    IChannelEntity* GetChannelEntity() override { return &aicpuTsHelper_; }

private:
    HcclResult ParseInputParam() override;
    HcclResult DecideSocketRole() override;
    HcclResult StartListen() override;
    HcclResult BuildSocket() override;
    HcclResult BuildConnection() override;
    HcclResult BuildNotify() override;
    HcclResult BuildBuffer() override;

    AicpuTsHccsEndpoint* HccsLocalEp() const { return static_cast<AicpuTsHccsEndpoint*>(local_.epPtr); }
    HcclResult EnableP2P();
    HcclResult EnableMemAccess();
    HcclResult TransportInit();
    HcclResult SetMachinePara(hccl::MachinePara& machinePara);
    void SetTransportParam(hccl::TransportPara& para) const;
    HcclResult BuildHcclChannelHccsRes(HcclChannelHccsRes& channelHccsRes);
    void TransportDeInit();
    void DisableMemAccess() const;
    void DestroyConnection();
    void DisableP2P();

    // --------------------- 转换参数 ---------------------
    bool serverInited_{false};

    // --------------------- 具体成员 ---------------------
    AicpuTsChannelEntity aicpuTsHelper_;
    std::shared_ptr<hccl::HcclSocket> socket_{nullptr};
    std::string socketTag_{};
    // for create TransportMem
    HcclDispatcher dispatcher_{nullptr}; // dispatcher放到最后析构
    DispatcherCtxPtr dispatcherCtx_{nullptr};
    std::unique_ptr<hccl::NotifyPool> notifyPool_;
    std::shared_ptr<hccl::Transport> transport_{nullptr};

    // for get mem temp
    std::vector<CommMem> remoteIpcRmaBufferVec_;
    std::vector<HcclMemEx> localIpcRmaBufferVecEx_;
    std::vector<HcclMemEx> remoteIpcRmaBufferVecEx_;
};
} // namespace hcomm

#endif // AICPU_TS_HCCS_CHANNEL_H
