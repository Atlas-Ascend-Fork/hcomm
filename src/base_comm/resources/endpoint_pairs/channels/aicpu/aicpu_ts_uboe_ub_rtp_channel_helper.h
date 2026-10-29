/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_TS_UBOE_UB_RTP_CHANNEL_HELPER_H
#define AICPU_TS_UBOE_UB_RTP_CHANNEL_HELPER_H

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "channel.h"
#include "aicpu_ts_channel_entity.h"

// Orion
#include "../../../../../../src/legacy/ascend950/unified_platform/resource/socket/socket.h"
#include "rma_connection.h"
#include "ub_mem_transport.h"
#include "dev_ub_connection.h"
#include "ub_local_notify.h"
#include "hcomm_adapter_hccp.h"

namespace hcomm {

constexpr u32 FINISH_MSG_SIZE = 128;
constexpr char_t FINISH_MSG[FINISH_MSG_SIZE] = "Uboe Comm Pipe ready!";

class AicpuTsUboeUbRtpChannelHelper : public Channel {
public:
    AicpuTsUboeUbRtpChannelHelper(
        EndpointHandle endpointHandle, const HcommChannelDesc& channelDesc, CommEngine engine,
        HcommChannelKind channelKind, const char* channelClassName);
    ~AicpuTsUboeUbRtpChannelHelper() override;

    HcclResult GetNotifyNum(uint32_t* notifyNum) const override;
    HcclResult GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos) override;
    HcclResult UpdateMemInfo(HcommMemHandle* memHandles, uint32_t memHandleNum) override;
    std::vector<char> GetMemUpdatePack() override;

    HcclResult H2DResPack(std::vector<char>& buffer);
    HcclResult Clean() override;
    HcclResult Resume() override;

    IChannelEntity* GetChannelEntity() override { return &aicpuTsHelper_; }

protected:
    static constexpr u64 NORMAL_NOTIFY_VAL = 1;

    MAKE_ENUM(UboeRmtBufType, NOTIFY, BUFFER)
    using RemoteBufferVec = std::vector<std::unique_ptr<Hccl::RemoteUbRmaBuffer>>;
    using LocalBufferVec = std::vector<Hccl::LocalUbRmaBuffer*>;

    virtual HcclResult BuildStateMachineConn() = 0;
    virtual HcclResult SendFinish() = 0;
    virtual HcclResult RecvFinish() = 0;

    HcclResult CheckSocketStatus(const std::string& socketOperator);

    HcclResult ParseInputParam() override;
    HcclResult BuildDrainResource();
    HcclResult BuildConn();

    bool IsSocketReady();
    HcclResult IsResReady(bool& isReady);
    bool IsConnsReady();
    HcclResult RecvDataProcess(bool& needFinish);
    HcclResult SendDataSize();
    HcclResult RecvDataSize();
    HcclResult SendExchangeData();
    HcclResult RecvExchangeData();

    void NotifyVecPack(Hccl::BinaryStream& binaryStream);
    void BufferVecPack(Hccl::BinaryStream& binaryStream, std::vector<Hccl::LocalRmaBuffer*>& bufferVec) const;
    void DrainBufferPack(Hccl::BinaryStream& binaryStream);
    void ConnVecPack(Hccl::BinaryStream& binaryStream);
    HcclResult RmtBufferVecUnpackProc(
        u32 locNum, Hccl::BinaryStream& binaryStream, RemoteBufferVec& bufferVec, UboeRmtBufType type);
    void RmtDrainBufferUnpackProc(Hccl::BinaryStream& binaryStream);
    HcclResult ConnVecUnpackProc(Hccl::BinaryStream& binaryStream, bool& result);

    HcclResult GetUniqueIdV2(std::vector<char>& result);
    std::vector<char> GetNotifyUniqueIds();
    std::vector<char> GetRmtBufferUniqueIds(RemoteBufferVec& bufferVec, UboeRmtBufType type) const;
    std::vector<char> GetLocBufferUniqueIds(LocalBufferVec& bufferVec, UboeRmtBufType type) const;
    std::vector<char> GetSingleRmtBufferUniqueId(u64 addr, u64 size, u32 tokenId, u32 tokenValue, u32 notifyId) const;
    std::vector<char> GetSingleLocBufferUniqueId(u64 addr, u64 size, u32 tokenId, u32 tokenValue) const;
    std::vector<char> GetDrainUniqueIds() const;
    std::vector<char> GetConnUniqueIds();

    // --------------------- 具体成员 ---------------------
    std::vector<std::unique_ptr<Hccl::DevUbConnection>> connections_{};

    std::unique_ptr<Hccl::UbLocalNotify> drainNotify_;        // 本端drain notify
    std::unique_ptr<Hccl::LocalUbRmaBuffer> drainBuffer_;     // 本端常量1 buffer
    std::unique_ptr<Hccl::RemoteUbRmaBuffer> rmtDrainBuffer_; // 对端常量1 buffer

    ChannelStatus channelStatus{ChannelStatus::INIT};

    u32 type_{1}; // 默认为UB
    u32 bufferNum_{0};
    u32 connNum_{0};
    u32 recvDataSize_{0};

    RemoteBufferVec rmtNotifyVec_;
    RemoteBufferVec rmtBufferVec_;
    LocalBufferVec locBufferVec_;
    std::vector<char> recvData_{};
    std::vector<char> recvFinishMsg_{};
    std::vector<char> sendData_{};
    std::vector<char> sendFinishMsg_{};

    DevBaseAttr devBaseAttr_{};

private:
    AicpuTsChannelEntity aicpuTsHelper_;
};

} // namespace hcomm

#endif // AICPU_TS_UBOE_UB_RTP_CHANNEL_HELPER_H
