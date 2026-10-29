/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_URMA_CHANNEL_H
#define CCU_URMA_CHANNEL_H

#include <memory>

#include "hcomm/ccu/hcomm_res_ccu_entity_defs.h" // CCU Channel 实体快照定义
#include <vector>
#include <atomic>
#include <mutex>

#include "../channel.h"

#include "urma_endpoint.h"
#include "ccu_transport_.h"

namespace hcomm {

class CcuUrmaChannel : public Channel {
public:
    // 当前仅支持交换hccl buffer
    CcuUrmaChannel(
        const EndpointHandle locEndpointHandle, const HcommChannelDesc& channelDesc,
        CommEngine engine = COMM_ENGINE_RESERVED);
    ~CcuUrmaChannel() override;

    HcclResult GetNotifyNum(uint32_t* notifyNum) const override;
    HcclResult GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos) override;
    ChannelStatus GetStatus() override;
    HcclResult UpdateMemInfo(HcommMemHandle* memHandles, uint32_t memHandleNum) override;

    HcclResult Clean() override;
    HcclResult Resume() override;

    EndpointHandle GetlocEndPointHandle() { return locEndpointHandle_; }

    // 获取设备逻辑 ID；impl 未建立时返回 -1。
    int32_t GetDevLogicId() const;
    // 将内部 CcuVersion 转换为 ABI 版本号。
    HcclResult GetCcuVersion(uint32_t& ccuVersion) const;
    uint32_t GetDieId() const;
    uint32_t GetChannelId() const;
    HcclResult GetRmtSignalAddrByIndex(uint32_t index, uint64_t& rmtCkeAddr) const;
    HcclResult GetRmtCcuBufferTokenInfo(uint32_t& rmtTokenId, uint32_t& rmtTokenValue) const;
    HcclResult CcuGetRmtMemToken(uint64_t srcVa, uint64_t& tokenInfo);
    HcclResult GetRmtVarAddrByIndex(uint32_t index, uint64_t& rmtXnAddr) const;
    HcclResult GetLocCkeByIndex(const uint32_t index, uint32_t& locCkeId) const;
    HcclResult GetLocXnByIndex(const uint32_t index, uint32_t& locXnId) const;

    HcclResult GetRmtCkeByIndex(const uint32_t index, uint32_t& rmtCkeId) const;
    HcclResult GetRmtXnByIndex(const uint32_t index, uint32_t& rmtXnId) const;
    HcclResult
    GetResourceNums(uint32_t& localCkeNum, uint32_t& localXnNum, uint32_t& remoteCkeNum, uint32_t& remoteXnNum) const;

    HcclResult GetRmtBuffer(uint64_t& addr, uint32_t& size, uint32_t& tokenId, uint32_t& tokenValue) const;
    // 获取对端 CCU 资源空间信息；与 GetRmtBuffer 的远端通信内存语义不同。
    HcclResult GetRmtCcuBufInfo(uint64_t& rmtCcuBufAddr, uint64_t& rmtCcuBufSize) const;
    // 在通道状态锁内构建 CCU Channel 实体；实体与远端 buffer 数组由调用方持有。
    HcclResult
    BuildHcommCcuChannelEntity(HcommCcuChannelEntity& channelEntity, std::vector<RegedBufferEntity>& remoteBuffers);

private:
    HcclResult ParseInputParam() override;
    HcclResult StartListen() override;
    HcclResult BuildSocket() override;
    HcclResult BuildConnection() override;
    HcclResult BuildNotify() override;
    HcclResult BuildBuffer() override;

    ChannelStatus TryPrepareAndConstruct();
    ChannelStatus GetStatusLocked();
    HcclResult CheckSnapshotResourceNums(
        uint32_t localCkeNum, uint32_t localXnNum, uint32_t remoteCkeNum, uint32_t remoteXnNum) const;
    void FillSnapshotBaseFields(
        HcommCcuChannelEntity& snapshot, uint32_t localCkeNum, uint32_t localXnNum, uint32_t remoteCkeNum,
        uint32_t remoteXnNum) const;
    HcclResult FillSnapshotResourceIds(
        HcommCcuChannelEntity& snapshot, uint32_t localCkeNum, uint32_t localXnNum, uint32_t remoteCkeNum,
        uint32_t remoteXnNum) const;
    HcclResult FillSnapshotRemoteCcuBuffer(HcommCcuChannelEntity& snapshot) const;
    HcclResult BuildRemoteBufferEntities(std::vector<RegedBufferEntity>& remoteBuffers);

    EndpointHandle locEndpointHandle_{nullptr};

    std::unique_ptr<CcuTransport> impl_{nullptr};
    UrmaEndpoint* ccuEndpoint_{nullptr};
    std::vector<HcommMemHandle> memHandles_{};

    std::atomic<bool> isFirstPrintChannelInfo_{true}; // 是否第一次打印通道建链信息，避免重复打印日志刷屏
    // 保护 GetStatus 懒建链、状态读写、快照构建与资源清理：避免 impl_/channelStatus_/资源状态竞态
    mutable std::mutex statusMtx_{};
};

} // namespace hcomm
#endif // CCU_URMA_CHANNEL_H
