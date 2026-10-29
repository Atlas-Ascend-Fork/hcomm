/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_UB_MEM_TRANSPORT_H
#define AIV_UB_MEM_TRANSPORT_H
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "hccl/hccl_res.h"
#include "../../../../../../legacy/ascend950/unified_platform/resource/socket/socket.h"
#include "../../../../../../legacy/ascend950/unified_platform/resource/buffer/local_ipc_rma_buffer_v2.h"
#include "../../../../../../legacy/ascend950/unified_platform/resource/buffer/remote_rma_buffer.h"
#include "binary_stream.h"
#include "buffer.h"
#include "transport_status.h"

namespace hcomm {
class AivUbMemTransport {
public:
    MAKE_ENUM(
        AivUbMemTransportStatus, INIT, P2P_ENABLING, SOCKET_OK, SEND_DATA_SIZE, RECV_DATA_SIZE, SEND_MEM_INFO,
        RECV_MEM_INFO, RECV_MEM_FIN, CONNECT_FAILED, SOCKET_TIMEOUT, READY);
    AivUbMemTransport(
        std::vector<Hccl::LocalRmaBuffer*>& localRmaBufferVec, Hccl::Socket* socket, HcommChannelDesc& channelDesc);
    ~AivUbMemTransport();
    HcclResult
    FillBufferVec(HcommMemHandle* memHandles, uint32_t bufferNum, std::vector<Hccl::LocalRmaBuffer*>& bufferVec);
    HcclResult Init();
    Hccl::TransportStatus GetStatus();
    HcclResult GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos);
    HcclResult CheckSocketStatus(std::string socketOperator);
    HcclResult UpdateMemInfo(HcommMemHandle* memHandles, uint32_t memHandleNum);

private:
    Hccl::Socket* socket_{}; // 交换所用的socket
    HcommChannelDesc channelDesc_;
    uint32_t exchangeDataSize_{0};
    std::vector<CommMem> remoteUserMems_;    // 储存内存位置、地址和大小信息
    std::vector<std::string> memInfoCopies_; // 储存memInfo字符串副本
    std::vector<char*> memInfoPointers_;     // 储存指针
    bool cacheValid_ = false;                // 当前缓存是否有效

    std::vector<Hccl::LocalRmaBuffer*> localRmaBufferVec_{};
    std::vector<Hccl::LocalRmaBuffer*> locMemTemp_{};
    std::vector<std::unique_ptr<Hccl::RemoteIpcRmaBuffer>> rmtBufferVec_{};
    std::vector<Hccl::RemoteRmaBuffer*> rmtRmaBufferVec_{};
    AivUbMemTransportStatus aivUbStatus_{AivUbMemTransportStatus::INVALID};
    Hccl::TransportStatus baseStatus_{Hccl::TransportStatus::INVALID};
    bool p2pEnableStarted_{false}; // enable p2p 是否已成功下发（与 manager 引用计数对应，失败不置位）
    uint32_t localDeviceLogicId_{0}; // 触发 enable p2p 时的本地逻辑 device id，供析构对称 disable 使用
    std::mutex remoteMemsMutex_;     // 远端内存列表互斥锁

    std::vector<char> sendData_{};
    std::vector<char> recvData_{};

    HcclResult IsSocketReady(bool& isReady);
    // 按需触发 enable p2p：HCCL_SUCCESS=已触发使能流程, HCCL_E_NOT_SUPPORT=自环链路跳过, 其他=失败
    HcclResult TriggerEnableP2P();
    // 非阻塞单次查询驱动侧 p2p 使能状态，未使能时由状态机下轮轮询推进
    HcclResult IsP2PEnabled(bool& isEnabled);
    HcclResult SendDataSize();
    HcclResult RecvDataSize();
    HcclResult SendMemInfo();
    HcclResult RecvMemInfo();
    HcclResult RecvDataProcess();
    HcclResult BufferPack(Hccl::BinaryStream& binaryStream, std::vector<Hccl::LocalRmaBuffer*>& bufferVec) const;
    void RmtBufferUnpackProc(Hccl::BinaryStream& binaryStream);
    HcclResult StateMachine();
    Hccl::TransportStatus UpdateStatus();
    void CheckStatusFuncResult(std::string funcName, HcclResult ret);
};

} // namespace hcomm

#endif // AIV_UB_MEM_TRANSPORT_H
