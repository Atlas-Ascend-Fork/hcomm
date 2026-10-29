/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_ROCE_HOST_LOC_CHANNEL_H
#define AIV_ROCE_HOST_LOC_CHANNEL_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "../channel.h"
#include "../aicpu/aicpu_ts_channel_entity.h"
#include "enum_factory.h"
#include "hccl_common.h"
#include "mem_device_pub.h"
#include "hcomm/hcomm_res_entity_defs.h"

// Orion
#include "../../../../../../legacy/ascend950/unified_platform/resource/socket/socket.h"
#include "../../../../../../legacy/ascend950/unified_platform/resource/buffer/local_rdma_rma_buffer_v2.h"
#include "remote_rma_buffer.h"
#include "../aicpu/roce_host_loc/roce_host_loc_connection.h"
#include "dev_buffer.h"
#include "../host/exchange_rdma_conn_dto.h"

namespace hcomm {
/**
 * @note 职责：AIV 引擎的 RoCE 通道。
 *       channel 创建完成后，将通道封装为 device 侧的 ChannelEntity 供 AIV 直接使用。
 *       与 AicpuTsRoceHostLocChannel 平级派生，均继承 Channel，共用 RoceHostLocConnection；AIV 无 notify，故不含 notify
 * 相关逻辑。
 */

class AivRoceHostLocChannel;

/**
 * @note 职责：AIV RoCE 通道实体：ctx 能力继承 AicpuTsChannelEntity 白拿，设备实体 slab 三段式
 *       （Build/PreAlloc/Fill）在本类覆写，经 owner_ 驱动宿主通道。
 */
class AivRoceHostLocChannelEntity : public AicpuTsChannelEntity {
public:
    explicit AivRoceHostLocChannelEntity(AivRoceHostLocChannel& owner);

    HcclResult BuildChannelEntityToDevice(void** devChannelPtr) override;
    HcclResult PreAllocChannelEntityToDevice(void** devChannelPtr) override;
    HcclResult FillChannelEntityToDevice() override;

private:
    AivRoceHostLocChannel& owner_;
};

class AivRoceHostLocChannel final : public Channel {
public:
    MAKE_ENUM(RdmaStatus, INIT, SOCKET_OK, QP_CREATED, DATA_EXCHANGE, QP_MODIFIED, CONN_OK)

    AivRoceHostLocChannel(EndpointHandle endpointHandle, HcommChannelDesc channelDesc, CommEngine engine);
    ~AivRoceHostLocChannel() override;

    HcclResult GetNotifyNum(uint32_t* notifyNum) const override;
    HcclResult GetBufferNum(uint32_t* bufferNum) const;
    HcclResult GetQpNum(uint32_t* qpNum) const;
    HcclResult GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos) override;
    ChannelStatus GetStatus() override;
    HcclResult GetStatus(ChannelStatus& status);
    HcclResult ProcessStatus();
    const HcommChannelDesc& GetChannelDesc() const override { return channelDesc_; }

    std::string Describe() const;

    CommProtocol GetCommProtocol() const { return channelDesc_.remoteEndpoint.protocol; }

    IChannelEntity* GetChannelEntity() override { return &aicpuTsEntity_; }
    HcclResult BuildAndGetDevChannelEntity(uint64_t* devChannelEntity);

    HcclResult Clean() override;
    HcclResult Resume() override;

private:
    friend class AivRoceHostLocChannelEntity;

    HcclResult ParseInputParam() override;
    HcclResult BuildConnection() override;
    HcclResult BuildBuffer() override;
    HcclResult BuildSocket() override;
    HcclResult StartListen() override;
    HcclResult DecideSocketRole() override;

    HcclResult CheckSocketStatus();
    HcclResult CreateQp();
    HcclResult ExchangeData();
    HcclResult ModifyQp();

    HcclResult BufferVecPack(Hccl::BinaryStream& binaryStream);
    HcclResult ConnVecPack(Hccl::BinaryStream& binaryStream);

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

    std::vector<std::unique_ptr<RoceHostLocConnection>> connections_{};
    std::vector<Hccl::LocalRdmaRmaBuffer*> localRmaBuffers_{};
    uint32_t bufferNum_{0};
    uint32_t connNum_{0};
    RdmaStatus rdmaStatus_{RdmaStatus::INIT};
    std::vector<std::unique_ptr<Hccl::RemoteRdmaRmaBuffer>> rmtRmaBuffers_{};
    ExchangeRdmaConnDto rmtConnDto_;

    AivRoceHostLocChannelEntity aicpuTsEntity_{*this};
};

} // namespace hcomm

#endif // AIV_ROCE_HOST_LOC_CHANNEL_H
