/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HOST_CPU_ROCE_CHANNEL_H
#define HOST_CPU_ROCE_CHANNEL_H

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "../channel.h"
#include "enum_factory.h"
#include "hccl_common.h"
#include "infiniband/verbs.h"

// Orion
#include "../../../../../../legacy/ascend950/unified_platform/resource/socket/socket.h"
#include "../../../../../../legacy/ascend950/unified_platform/resource/buffer/local_rdma_rma_buffer_v2.h"
#include "remote_rma_buffer.h"
#include "host_rdma_connection.h"
#include "task_param.h"

#include "exchange_data_format.h"
#include "private_types.h"

namespace hcomm {

class HostCpuRoceChannel final : public Channel {
public:
    MAKE_ENUM(RdmaStatus, INIT, SOCKET_OK, CAP_EXCHANGED, QP_CREATED, DATA_EXCHANGE, QP_MODIFIED, CONN_OK)

    HostCpuRoceChannel(
        EndpointHandle endpointHandle, HcommChannelDesc channelDesc, CommEngine engine = COMM_ENGINE_RESERVED);
    ~HostCpuRoceChannel();

    HcclResult GetNotifyNum(uint32_t* notifyNum) const override;
    HcclResult GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos) override;
    ChannelStatus GetStatus() override;
    HcclResult GetStatus(ChannelStatus& status);

    HcclResult Clean() override;
    HcclResult Resume() override;

    HcclResult ProcessStatus();
    std::string Describe() const;
    HcclResult SetDfxCallback(std::function<HcclResult(const Hccl::TaskParam&, u64)> callback);

    // 数据面调用verbs接口
    HcclResult NotifyRecord(const uint32_t remoteNotifyIdx) override;
    HcclResult NotifyWait(const uint32_t localNotifyIdx, const uint32_t timeout) override;
    HcclResult WriteWithNotify(void* dst, const void* src, const uint64_t len, const uint32_t remoteNotifyIdx) override;
    HcclResult Write(void* dst, const void* src, const uint64_t len) override;
    HcclResult Read(void* dst, const void* src, const uint64_t len) override;
    HcclResult ChannelFence() override;
    HcclResult ChannelDrain() override;
    HcclResult GetHcclBuffer(void*& addr, uint64_t& size);

private:
    HcclResult ReportDfxTaskEnd(Hccl::TaskParam& taskParam, bool isFailed) const;
    HcclResult ParseInputParam() override;
    HcclResult DecideSocketRole() override;
    HcclResult StartListen() override;
    HcclResult BuildSocket() override;
    HcclResult BuildConnection() override;
    HcclResult BuildNotify() override;
    HcclResult BuildBuffer() override;

    HcclResult CheckSocketStatus();
    HcclResult CreateQp();
    HcclResult ExchangeData();
    HcclResult ModifyQp();
    HcclResult SyncAfterModifyQp() const;

    void NotifyVecPack(Hccl::BinaryStream& binaryStream);
    HcclResult BufferVecPack(Hccl::BinaryStream& binaryStream);
    HcclResult ConnVecPack(Hccl::BinaryStream& binaryStream);
    // void HandshakeMsgPack(Hccl::BinaryStream &binaryStream);

    // HcclResult HandshakeMsgUnpack(Hccl::BinaryStream &binaryStream);
    HcclResult NotifyVecUnpack(Hccl::BinaryStream& binaryStream);
    HcclResult RmtBufferVecUnpackProc(Hccl::BinaryStream& binaryStream);
    HcclResult ConnVecUnpackProc(Hccl::BinaryStream& binaryStream);

    std::vector<Hccl::QpInfo> GetQpInfos() const; // in Connection

    HcclResult ExchangeCapability();
    HcclResult ExchangeDataHybird();
    HcclResult GetRemoteAddrHybird(hccl::MemType memType, u8*& data, u64& size);
    HcclResult ParseRecvExchangeDataHybird();
    HcclResult BuildExchangeDataHybird();
    HcclResult BuildExchangeDataLengthHybird();

    HcclResult RegisterUserMemHybird();
    HcclResult BuildNotifyWrHybird(const uint32_t remoteNotifyIdx, struct ibv_send_wr& notifRecordWr);
    HcclResult WriteWithNotifyHybrid(void* dst, const void* src, uint64_t len, uint32_t remoteNotifyIdx);
    HcclResult NotifyWaitHybrid(uint32_t localNotifyIdx, uint32_t timeout);

    HcclResult CreateNotifyHybird(hccl::MemType notifyType, uint32_t notifyId);
    HcclResult CreateNotifyValueBufferHybird();
    HcclResult CreateNotifyBufferHybird(hccl::MemType notifyType, uint32_t notifyId, u8*& data, u64& size);
    hccl::MemType NotifyIdToMemtypeHybird(uint32_t remoteNotifyIdx) const;
    HcclResult ConnectSingleQpHybrid(std::function<bool()> needStop);

    HcclResult IbvPostRecv() const;
    HcclResult PrepareNotifyWrResource(
        uint32_t qpIdx, const uint64_t len, const uint32_t remoteNotifyIdx, struct ibv_send_wr& notifyRecordWr,
        Hccl::TaskParam& taskParam) const;
    HcclResult PrepareWriteWrResource(
        const void* dst, const void* src, const uint64_t len, const uint32_t remoteNotifyIdx,
        struct ibv_send_wr& writeWithNotifyWr, Hccl::TaskParam& taskParam) const;

    HcclResult PostRdmaOp(
        const char* caller, ibv_wr_opcode opcode, const void* localAddr, const void* remoteAddr, const uint64_t len);
    void BuildRdmaWr(
        const char* caller, ibv_wr_opcode opcode, const void* localAddr, const void* remoteAddr, uint64_t len,
        size_t localIdx, size_t rmtIdx, struct ibv_send_wr& wr, struct ibv_sge& sg) const;
    HcclResult PostAndCheckSend(struct ibv_qp* qp, const uint32_t qpIdx, const char* caller, struct ibv_send_wr& wr);
    HcclResult FindLocalBuffer(const uint64_t addr, const uint64_t len, size_t& targetIdx) const;
    HcclResult FindRemoteBuffer(const uint64_t addr, const uint64_t len, size_t& targetIdx) const;
    HcclResult ReportWcStatusError(enum ibv_wc_status status);

    HcclResult WaitForWqeCompletion();
    HcclResult WaitForSingleQpWqeCompletion(const Hccl::QpInfo& qpInfo, uint32_t qpIdx, uint64_t timeoutMs);
    HcclResult CheckQpInfoForPolling(const Hccl::QpInfo& qpInfo, uint32_t qpIdx) const;
    HcclResult
    CheckPollCqResult(const Hccl::QpInfo& qpInfo, uint32_t qpIdx, int actualNum, const struct ibv_wc* wc) const;

    // Wrapper for stub
    int IbvPollCq(ibv_cq* sendCq, uint32_t numEntries, ibv_wc* wc) const { return ibv_poll_cq(sendCq, numEntries, wc); }

    std::vector<uint16_t> srcPortBuf_{};

    uint32_t bufferNum_{0};
    uint32_t connNum_{0};

    std::vector<std::unique_ptr<HostRdmaConnection>> connections_{};
    std::vector<Hccl::LocalRdmaRmaBuffer*> localRmaBuffers_{};
    std::vector<uint32_t> localDpuNotifyIds_{};
    std::vector<uint32_t> remoteDpuNotifyIds_;
    std::vector<std::unique_ptr<Hccl::RemoteRdmaRmaBuffer>> rmtRmaBuffers_{};
    std::vector<ExchangeRdmaConnDto> rmtConnDtos_;
    std::vector<int> wqeNums_;
    bool fenceFlag_{false};
    uint64_t maxMsgSize_{0};
    uint32_t lbMax_{0}; // 多QP负载均衡
    std::function<HcclResult(const Hccl::TaskParam&, u64)> dfxCallback_;
    std::mutex cq_mutex;
    std::mutex sendCq_mutex;

    RdmaStatus rdmaStatus_{RdmaStatus::INIT};

    // 1. 能力协商结果
    RoCECapability remoteCap_;  // 对端能力
    bool isHybridMode_ = false; // 是否为混合模式

    uint32_t localNotifySize_;
    uint32_t localNotifyAccess_;

    std::array<hccl::MemMsg, static_cast<u32>(hccl::MemType::MEM_TYPE_RESERVED)> localMemMsg_;
    std::array<hccl::MemMsg, static_cast<u32>(hccl::MemType::MEM_TYPE_RESERVED)> remoteMemMsg_;
    uint64_t exchangeDataTotalSize_;
    std::vector<uint8_t> exchangeDataForSend_;
    std::vector<uint8_t> exchangeDataForRecv_;
};

} // namespace hcomm

#endif // HOST_CPU_ROCE_CHANNEL_H
