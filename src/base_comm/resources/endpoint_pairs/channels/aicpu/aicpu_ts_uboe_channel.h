/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_TS_UBOE_CHANNEL_H
#define AICPU_TS_UBOE_CHANNEL_H

#include "aicpu_ts_uboe_ub_rtp_channel_helper.h"

namespace hcomm {

class AicpuTsUboeChannel : public AicpuTsUboeUbRtpChannelHelper {
public:
    AicpuTsUboeChannel(
        EndpointHandle endpointHandle, const HcommChannelDesc& channelDesc, CommEngine engine = COMM_ENGINE_RESERVED);
    ~AicpuTsUboeChannel() override;

    ChannelStatus GetStatus() override;

    HcclResult Clean() override;
    HcclResult Resume() override;

protected:
    HcclResult BuildStateMachineConn() override;
    HcclResult SendFinish() override;
    HcclResult RecvFinish() override;

private:
    HcclResult ParseInputParam() override;
    HcclResult StartListen() override;
    HcclResult BuildSocket() override;
    HcclResult BuildConnection() override;
    HcclResult BuildNotify() override;
    HcclResult BuildBuffer() override;

    void EidPack();
    HcclResult SendEidData();
    HcclResult RecvEidData();
    void RecvEidDataProcess();
    void RmtEidUnpackProc(Hccl::IpAddress& rmtAddr);
    HcclResult HandleProcessData();
    HcclResult ProcessUboeState();

    std::vector<char> sendEidData_{};
    std::vector<char> recvEidData_{};

    MAKE_ENUM(
        UboeStatus, INIT, SEND_EID, RECV_EID, PROCESS_EID_DATA, BUILD_CONN, SEND_SIZE, RECV_SIZE, SEND_DATA, RECV_DATA,
        SEND_FIN, RECV_FIN, PROCESS_DATA, SET_READY, READY)
    UboeStatus uboeStatus{UboeStatus::INIT};
};

} // namespace hcomm

#endif // AICPU_TS_UBOE_CHANNEL_H
