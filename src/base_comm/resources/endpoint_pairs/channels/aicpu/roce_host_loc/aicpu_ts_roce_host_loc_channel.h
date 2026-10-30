/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_TS_ROCE_HOST_LOC_CHANNEL_H
#define AICPU_TS_ROCE_HOST_LOC_CHANNEL_H

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "../../channel.h"
#include "aicpu_ts_channel_entity.h"
#include "enum_factory.h"
#include "hccl_common.h"
#include "mem_device_pub.h"
#include "hcomm_res_entity_defs.h"
#include "../host/exchange_rdma_conn_dto.h"

// Orion
#include "../../../../../../legacy/ascend950/unified_platform/resource/socket/socket.h"
#include "../../../../../../legacy/ascend950/unified_platform/resource/buffer/local_rdma_rma_buffer_v2.h"
#include "remote_rma_buffer.h"
#include "./roce_host_loc_connection.h"
#include "rdma_local_notify.h"
#include "dev_buffer.h"

namespace hcomm {
/**
 * @note 职责：AicpuTs 引擎（AICPU/AICPU_TS）的 RoCE 通道。
 *       channel 创建完成后调用 H2DResPack 将通道序列化，并在 device 侧 RoceHostLocTransportLiteImpl 反序列化。
 */
constexpr u32 RDMA_NOTIFY_NUM = 3;

using EndpointDeviceLoc = decltype(EndpointLoc::device);

class AicpuTsRoceHostLocChannel final : public Channel {
public:
    MAKE_ENUM(RdmaStatus, INIT, SOCKET_OK, QP_CREATED, DATA_EXCHANGE, QP_MODIFIED, CONN_OK)

    AicpuTsRoceHostLocChannel(EndpointHandle endpointHandle, HcommChannelDesc channelDesc, CommEngine engine);
    ~AicpuTsRoceHostLocChannel() override;

    HcclResult GetNotifyNum(uint32_t* notifyNum) const override;
    HcclResult GetBufferNum(uint32_t* bufferNum) const;
    HcclResult GetQpNum(uint32_t* qpNum) const;
    HcclResult GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos) override;
    ChannelStatus GetStatus() override;
    HcclResult GetStatus(ChannelStatus& status);

    HcclResult Clean() override;
    HcclResult Resume() override;

    CommEngine GetCommEngine() const { return engine_; }
    CommProtocol GetCommProtocol() const { return channelDesc_.remoteEndpoint.protocol; }

    HcclResult ProcessStatus();
    std::string Describe() const;
    HcclResult H2DResPack(std::vector<char>& buffer) const;

    HcclResult Serialize(std::shared_ptr<hccl::DeviceMem>& out) override;

    IChannelEntity* GetChannelEntity() override { return &aicpuTsEntity_; }

private:
    friend class AicpuTsRoceV2ChannelEntity;

    HcclResult ParseInputParam() override;
    HcclResult DecideSocketRole() override;
    HcclResult StartListen() override;
    HcclResult BuildSocket() override;
    HcclResult BuildConnection() override;
    HcclResult BuildNotify() override;
    HcclResult BuildBuffer() override;

    HcclResult BuildNotifyValueBuffer();
    HcclResult CheckSocketStatus();
    HcclResult CreateQp();
    HcclResult ExchangeData();
    HcclResult ModifyQp();
    void NotifyVecPack(Hccl::BinaryStream& binaryStream);
    HcclResult BufferVecPack(Hccl::BinaryStream& binaryStream);
    HcclResult ConnVecPack(Hccl::BinaryStream& binaryStream);
    HcclResult NotifyVecUnpack(Hccl::BinaryStream& binaryStream);
    HcclResult RmtBufferVecUnpackProc(Hccl::BinaryStream& binaryStream);
    HcclResult ConnVecUnpackProc(Hccl::BinaryStream& binaryStream);
    HcclResult BuildAndGetLocNotifyInfo(RegedNotifyEntity** notify) const;
    HcclResult BuildAndGetRmtNotifyInfo(RegedNotifyEntity** notify) const;
    HcclResult BuildAndGetRmtBufInfo(std::vector<RegedBufferEntity>& bufList, RegedBufferEntity** bufferEntityPtr);
    HcclResult BuildAndGetLocBufInfo(std::vector<RegedBufferEntity>& bufList, RegedBufferEntity** bufferEntityPtr);
    HcclResult BuildAndGetSqContext(std::vector<SqContext>& sqList, SqContext** sqContextPtr);
    HcclResult BuildAndGetCqContext(std::vector<CqContext>& cqList, CqContext** cqContextPtr);
    HcclResult BuildHostEntity(
        ChannelEntity& hostEntity, std::vector<RegedBufferEntity>& locBufList,
        std::vector<RegedBufferEntity>& rmtBufList, std::vector<SqContext>& sqList, std::vector<CqContext>& cqList);
    void FreeDeviceMemories();
    std::vector<char> GetLocalNotifyUniqueIds() const;
    std::vector<char> GetRemoteNotifyUniqueIds() const;
    std::vector<char> GetLocBufferUniqueIds() const;
    std::vector<char> GetRmtBufferUniqueIds() const;
    std::vector<char> GetNotifyValueBufferUniqueIds() const;
    std::vector<char> GetSingleRmaBufferUniqueId(u64 addr, u64 size, u32 key) const;
    std::vector<char> GetConnUniqueIds() const;
    std::vector<char> GetUniqueId() const;
    HcclResult PackOpData(std::vector<char>& data) const;

    uint32_t bufferNum_{0};
    uint32_t connNum_{0};

    AicpuTsRoceV2ChannelEntity aicpuTsEntity_{*this};
    std::vector<std::unique_ptr<RoceHostLocConnection>> connections_{};
    std::vector<Hccl::LocalRdmaRmaBuffer*> localRmaBuffers_{};
    std::shared_ptr<Hccl::DevBuffer> notifyValueMem_{nullptr};
    std::unique_ptr<Hccl::LocalRdmaRmaBuffer> notifyValueBuffer_{nullptr};
    std::vector<std::unique_ptr<Hccl::RemoteRdmaRmaBuffer>> remoteNotifies_{};
    std::vector<std::unique_ptr<Hccl::RemoteRdmaRmaBuffer>> rmtRmaBuffers_{};
    ExchangeRdmaConnDto rmtConnDto_;

    RdmaStatus rdmaStatus_{RdmaStatus::INIT};
};

} // namespace hcomm

#endif // AICPU_TS_ROCE_HOST_LOC_CHANNEL_H
