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
#include "step_transfer.h"
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
    HcclResult WriteReduceWithNotify(
        void* dst, const void* src, const uint64_t len, HcclDataType dataType, HcclReduceOp reduceOp,
        const uint32_t remoteNotifyIdx) override;
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

    HostRdmaConnection::RoceAttr BuildRoceAttrFromDesc() const;

    HcclResult CheckSocketStatus();
    HcclResult CreateQp();
    HcclResult ExchangeData();
    HcclResult ModifyQp();
    HcclResult SyncAfterModifyQp();

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
    // 构造单块 inline-reduce WR，buffer 索引由 WriteReduceWithNotify 整段预校验后传入
    HcclResult PrepareReduceWrResource(
        const void* dst, const void* src, const uint64_t len, const bool withNotify, const uint32_t dpuNotifyId,
        const size_t localIdx, const size_t rmtIdx, struct ibv_send_wr& reduceWr) const;

    // 将整段数据按 max(qpThreshold, ceil(len/qpNum)) 分配到各 QP，再由每个 QP 内部按硬件单 WR
    // 上限逐片下发。分配不到数据的 QP 仍下发空 WR，保证每个 QP 都有 CQE。
    HcclResult PostRdmaOpAllQps(
        const char* caller, ibv_wr_opcode opcode, const void* localAddr, const void* remoteAddr, const uint64_t len,
        uint32_t immData);
    // 复用 PostRdmaOpAllQps 已获取的 QP 句柄，避免重复获取 QPs
    struct QpRef {
        uint32_t idx;
        struct ibv_qp* qp;
    };
    // 将本 QP 上的一段数据按硬件单 WR 上限切片，并逐 WR 单次下发；
    // opcode 为尾片语义（WRITE/READ/WRITE_WITH_IMM），中间片统一为普通 WRITE/READ。
    // immData 仅在尾片为 WRITE_WITH_IMM 时有效。
    HcclResult PostRdmaMemOp(
        const char* caller, ibv_wr_opcode opcode, const QpRef& qpRef, const void* localAddr, const void* remoteAddr,
        const uint64_t len, uint32_t immData);
    // 单次 reduce 下发的上下文，收敛 PostReduceOp 的入参
    struct ReduceOpCtx {
        const char* caller;
        QpRef qpRef;
        const void* localAddr;
        const void* remoteAddr;
        uint64_t len;
        uint64_t elemSize;
        uint8_t mappedReduceOp;
        uint8_t mappedReduceType;
        uint32_t dpuNotifyId;
        size_t localIdx;
        size_t rmtIdx;
        // 标记本次操作是否已有 WR 成功下发（inline-reduce 不可重放）
        bool& anyPosted;
    };
    // 将本 QP 上的一段 reduce 数据按硬件单 WR 上限做元素对齐切片并逐 WR 下发；
    // 仅尾片携带 notify。
    HcclResult PostReduceOp(const ReduceOpCtx& ctx);
    // WriteReduceWithNotify 前置校验：通道就绪、QP/注册 buffer 非空、notify 索引合法，输出 DPU notify id
    HcclResult CheckWriteReducePrecondition(
        const char* caller, const std::vector<Hccl::QpInfo>& qpInfo, uint32_t remoteNotifyIdx,
        uint32_t& dpuNotifyId) const;
    // 映射并校验 reduce 数据类型/操作，长度需按元素大小对齐，输出映射结果与元素大小
    HcclResult MapAndCheckReduceParam(
        const char* caller, HcclDataType dataType, HcclReduceOp reduceOp, uint64_t len, uint8_t& mappedReduceType,
        uint8_t& mappedReduceOp, uint32_t& elemSize) const;
    Hccl::TaskParam BuildWriteReduceTaskParam(
        void* dst, const void* src, uint64_t len, uint32_t dpuNotifyId, HcclReduceOp reduceOp,
        HcclDataType dataType) const;
    // 单个 QP 的下发：无数据时下发空 WRITE_WITH_IMM 传递 notify，否则按 maxReduceChunk 切片 reduce
    HcclResult PostWriteReducePerQp(
        const char* caller, const QpRef& qpRef, const void* localAddr, void* remoteAddr, uint64_t chunkLen,
        uint32_t elemSize, uint8_t mappedReduceOp, uint8_t mappedReduceType, uint32_t dpuNotifyId, size_t localIdx,
        size_t rmtIdx, bool& anyPosted);
    // 第一级切片：按 QP 分配数据量并逐 QP 下发；任一 QP 失败即终止（inline-reduce 不可重放）
    HcclResult PostWriteReduceChunks(
        const char* caller, const std::vector<Hccl::QpInfo>& qpInfo, const void* localAddr, void* remoteAddr,
        uint64_t len, uint32_t elemSize, uint8_t mappedReduceOp, uint8_t mappedReduceType, uint32_t dpuNotifyId,
        size_t localIdx, size_t rmtIdx);
    void BuildRdmaWr(
        const char* caller, ibv_wr_opcode opcode, const void* localAddr, const void* remoteAddr, uint64_t len,
        size_t localIdx, size_t rmtIdx, int sendFlags, uint32_t immData, struct ibv_send_wr& wr,
        struct ibv_sge& sg) const;
    // signaled 为 true 时该次 post 会产生 CQE，才计入 wqeNums_（逐 WR 下发时仅尾片计数）。
    HcclResult PostAndCheckSend(
        struct ibv_qp* qp, const uint32_t qpIdx, const char* caller, struct ibv_send_wr& wr, bool signaled = true);
    // 单次扩展 WR 下发的属性，收敛 PostAndCheckSendExt 的入参
    struct PostExtAttr {
        uint8_t reduceOp;
        uint8_t reduceType;
        // true 表示该 WR 会产生 CQE，需要计入 wqeNums_
        bool signaled;
    };
    // 通过 hns 驱动扩展接口 ibv_ext_post_send 下发带 inline-reduce 属性的 WR。
    HcclResult PostAndCheckSendExt(
        struct ibv_qp* qp, const uint32_t qpIdx, const char* caller, struct ibv_send_wr& wr, const PostExtAttr& attr);
    // 驱动扩展接口的实际调用点，抽出为成员函数便于 UT 打桩；
    // 返回 HCCL_E_NOT_SUPPORT 表示扩展库或符号不可用，driverRet 透传驱动原始返回值。
    HcclResult ExtPostSend(
        struct ibv_qp* qp, struct ibv_send_wr& wr, uint8_t mappedReduceOp, uint8_t mappedReduceType,
        struct ibv_send_wr** badWr, int& driverRet);
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
    int IbvPostSend(struct ibv_qp* qp, struct ibv_send_wr* wr, struct ibv_send_wr** badWr)
    {
        return ibv_post_send(qp, wr, badWr);
    }

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
    uint64_t exchangeDataTotalSize_{0};
    std::vector<uint8_t> exchangeDataForSend_;
    std::vector<uint8_t> exchangeDataForRecv_;

    // —— GetStatus 拆步进度变量 ——
    u32 capabilitySendDone_{0};        // ExchangeCapability 步：0=待发能力 1=待收能力
    RoCECapability localCapability_{}; // 能力发送缓冲（跨步复用）
    RoCECapability recvCapability_{};  // 能力接收缓冲（跨步复用）
    StepTransfer capabilityTransfer_;  // 能力交换单步收发器
    StepTransfer exchangeTransfer_;    // ExchangeData 四阶段交换器
    bool hybridSetupDone_{false};      // hybrid 注册/打包一次性 setup 标识
    u32 hybridStep_{0};                // ExchangeDataHybird 步：0=发数据 1=收数据
    StepTransfer hybridTransfer_;      // hybrid 大数据单步收发器
    u32 syncFlagStep_{0};              // SyncAfterModifyQp 步：0=发标志 1=收标志
    uint8_t syncRecvFlag_{0};          // ModifyQp 完成标志接收缓冲
    StepTransfer syncFlagTransfer_;    // ModifyQp 同步标志单步收发器
};

} // namespace hcomm

#endif // HOST_CPU_ROCE_CHANNEL_H
