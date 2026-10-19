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
#include <mutex>
#include <string>
#include <vector>

#include "../channel.h"
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
class AivRoceHostLocChannel final : public Channel {
public:
    MAKE_ENUM(RdmaStatus, INIT, SOCKET_OK, QP_CREATED, DATA_EXCHANGE, QP_MODIFIED, CONN_OK)

    AivRoceHostLocChannel(EndpointHandle endpointHandle, HcommChannelDesc channelDesc);
    ~AivRoceHostLocChannel() override;

    HcclResult Init() override;
    HcclResult GetNotifyNum(uint32_t* notifyNum) const override;
    HcclResult GetBufferNum(uint32_t* bufferNum) const;
    HcclResult GetQpNum(uint32_t* qpNum) const;
    HcclResult GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos) override;
    ChannelStatus GetStatus() override;
    HcclResult GetStatus(ChannelStatus& status);
    HcclResult ProcessStatus();
    HcommChannelKind GetChannelKind() const override;
    const HcommChannelDesc& GetChannelDesc() const override { return channelDesc_; }

    std::string Describe() const;

    CommProtocol GetCommProtocol() const { return channelDesc_.remoteEndpoint.protocol; }

    HcclResult BuildAndGetDevChannelEntity(uint64_t* devChannelEntityPtr);
    HcclResult PreAllocDevChannelEntity(uint64_t* devChannelEntityPtr);
    HcclResult FillDevChannelEntity();

    HcclResult Clean() override;
    HcclResult Resume() override;

    HcclResult NotifyRecord([[maybe_unused]] const uint32_t remoteNotifyIdx) override { return HCCL_SUCCESS; }
    HcclResult
    NotifyWait([[maybe_unused]] const uint32_t localNotifyIdx, [[maybe_unused]] const uint32_t timeout) override
    {
        return HCCL_SUCCESS;
    }
    HcclResult WriteWithNotify(
        [[maybe_unused]] void* dst, [[maybe_unused]] const void* src, [[maybe_unused]] const uint64_t len,
        [[maybe_unused]] uint32_t remoteNotifyIdx) override
    {
        return HCCL_SUCCESS;
    }
    HcclResult
    Write([[maybe_unused]] void* dst, [[maybe_unused]] const void* src, [[maybe_unused]] uint64_t len) override
    {
        return HCCL_SUCCESS;
    }
    HcclResult
    Read([[maybe_unused]] void* dst, [[maybe_unused]] const void* src, [[maybe_unused]] uint64_t len) override
    {
        return HCCL_SUCCESS;
    }
    HcclResult ChannelFence() override { return HCCL_SUCCESS; }

private:
    HcclResult ParseInputParam();
    HcclResult BuildConnection();
    HcclResult BuildBuffer();
    HcclResult BuildSocket();
    HcclResult StartListen();

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
    void ReleaseDeviceEntitySlab();

    // 入参
    EndpointHandle endpointHandle_;
    HcommChannelDesc channelDesc_;

    // 转换参数
    EndpointDesc localEp_;
    EndpointDesc remoteEp_;
    Hccl::Socket* socket_{nullptr};
    // 保存 socketConfig 的拷贝并记录裸指针，供析构时 DestroySocket 回收引用计数。须在 GetSocket 之前调用，
    // 避免 GetSocket 成功后 holder 构造异常（bad_alloc）导致 socketConfig_ 未设置、析构无法回收引用计数。
    void SaveSocketConfig(const Hccl::SocketConfig& socketConfig)
    {
        socketConfigHolder_ = std::make_unique<Hccl::SocketConfig>(socketConfig);
        socketConfig_ = socketConfigHolder_.get();
    }

    std::unique_ptr<Hccl::SocketConfig> socketConfigHolder_{nullptr};
    const Hccl::SocketConfig* socketConfig_{nullptr};
    RdmaHandle rdmaHandle_{nullptr};
    uint32_t devicePhyId_{};

    std::vector<std::unique_ptr<RoceHostLocConnection>> connections_{};
    std::vector<Hccl::LocalRdmaRmaBuffer*> localRmaBuffers_{};
    uint32_t bufferNum_{0};
    uint32_t connNum_{0};
    ChannelStatus channelStatus_{ChannelStatus::INIT};
    RdmaStatus rdmaStatus_{RdmaStatus::INIT};
    std::vector<std::unique_ptr<Hccl::RemoteRdmaRmaBuffer>> rmtRmaBuffers_{};
    ExchangeRdmaConnDto rmtConnDto_;
    std::mutex remoteMemsMutex_{};
    std::vector<CommMem> remoteUserMems_{};
    std::vector<std::string> memInfoCopies_{};
    std::vector<char*> memInfoPointers_{};
    bool cacheValid_{false};

    void* devChannelEntitySlab_{nullptr};
    size_t devChannelEntitySlabSize_{0};
};

} // namespace hcomm

#endif // AIV_ROCE_HOST_LOC_CHANNEL_H
