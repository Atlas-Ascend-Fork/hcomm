/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aiv_ub_mem_transport.h"
#include "exception_handler.h"
#include "../../../../../../legacy/ascend950/unified_platform/resource/socket/socket.h"
#include "../../../../../../legacy/ascend950/unified_platform/resource/buffer/exchange_ipc_buffer_dto.h"
#include "../../../../../../legacy/ascend950/unified_platform/resource/mem/user_remote_mem_getter.h"
#include "../../../../../../legacy/ascend950/unified_platform/common/p2p_enable_manager.h"
#include "env_config/env_config_v2.h"

namespace hcomm {

AivUbMemTransport::AivUbMemTransport(Hccl::Socket* socket, HcommChannelDesc& channelDesc)
    : socket_(socket),
      channelDesc_(channelDesc)
{}

AivUbMemTransport::~AivUbMemTransport()
{
    // 与 TriggerEnableP2P 对称：仅对已按需触发过 enable p2p 的链路，析构时对称执行 disable p2p
    if (!p2pEnableStarted_) {
        return;
    }
    std::vector<uint32_t> remoteDevices{channelDesc_.remoteEndpoint.loc.device.devPhyId};
    (void)Hccl::P2PEnableManager::GetInstance().DisableP2P(localDeviceLogicId_, remoteDevices);
}

HcclResult AivUbMemTransport::FillBufferVec(
    HcommMemHandle* memHandles, uint32_t bufferNum, std::vector<Hccl::LocalIpcRmaBuffer*>& bufferVec)
{
    uint32_t totalBufferNum = localRmaBufferVec_.size() + bufferNum;
    if (UNLIKELY(totalBufferNum > MAX_BUFFER_NUM)) {
        HCCL_ERROR(
            "[AivUbMemTransport][FillBufferVec] totalBufferNum[%u] exceeds limit[%u]", totalBufferNum, MAX_BUFFER_NUM);
        return HCCL_E_PARA;
    }
    for (uint32_t i = 0; i < bufferNum; ++i) {
        auto localIpcRmaBuffer = reinterpret_cast<Hccl::LocalIpcRmaBuffer*>(memHandles[i]);
        CHK_PTR_NULL(localIpcRmaBuffer);
        auto buf = localIpcRmaBuffer->GetBuf();
        CHK_PTR_NULL(buf);
        bufferVec.push_back(localIpcRmaBuffer);
        HCCL_INFO(
            "[AivUbMemTransport][FillBufferVec] memHandleNum[%u] buffer[%s]", i, localIpcRmaBuffer->Describe().data());
    }
    return HCCL_SUCCESS;
}

HcclResult AivUbMemTransport::Init()
{
    uint32_t bufferNum = channelDesc_.memHandleNum;
    if (bufferNum == 0) {
        HCCL_ERROR("[AivUbMemTransport][Init] bufferNum is 0.");
        return HCCL_E_PARA;
    }
    HCCL_INFO("[AivUbMemTransport][Init] channelDesc_.memHandleNum: %u", bufferNum);
    CHK_RET(FillBufferVec(channelDesc_.memHandles, bufferNum, localRmaBufferVec_));

    baseStatus_ = Hccl::TransportStatus::INIT;
    return HCCL_SUCCESS;
}

HcclResult AivUbMemTransport::IsSocketReady(bool& isReady)
{
    CHK_PTR_NULL(socket_);
    EXCEPTION_HANDLE_BEGIN
    Hccl::SocketStatus socketStatus = socket_->GetAsyncStatus();
    if (socketStatus == Hccl::SocketStatus::OK) {
        baseStatus_ = Hccl::TransportStatus::SOCKET_OK;
        isReady = true;
    } else if (socketStatus == Hccl::SocketStatus::TIMEOUT) {
        baseStatus_ = Hccl::TransportStatus::SOCKET_TIMEOUT;
        isReady = false;
    }
    EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}

void AivUbMemTransport::CheckStatusFuncResult(std::string funcName, HcclResult ret)
{
    if (UNLIKELY(ret != HCCL_SUCCESS)) {
        HCCL_ERROR(
            "[%s] fail ret[%d], aivUbStatus_[%d], baseStatus_[%d]", funcName.c_str(), ret, aivUbStatus_, baseStatus_);
        baseStatus_ = Hccl::TransportStatus::INVALID;
    }
}

Hccl::TransportStatus AivUbMemTransport::GetStatus()
{
    if (baseStatus_ == Hccl::TransportStatus::READY || baseStatus_ == Hccl::TransportStatus::INVALID) {
        return baseStatus_;
    } else if (baseStatus_ == Hccl::TransportStatus::INIT && !p2pEnableStarted_) {
        aivUbStatus_ = AivUbMemTransportStatus::INIT;
    }

    // p2p 状态机：按需触发 enable p2p 并轮询驱动侧状态。必须在 socket 检查之前：
    // PCIE 链路 socket 连接走 vinc 通路依赖 p2p 使能，先查 socket 会互等死锁
    switch (aivUbStatus_) {
        case AivUbMemTransportStatus::INIT:
            if (channelDesc_.remoteEndpoint.protocol == COMM_PROTOCOL_PCIE) {
                HcclResult ret = TriggerEnableP2P();
                if (ret == HCCL_SUCCESS) {
                    aivUbStatus_ = AivUbMemTransportStatus::P2P_ENABLING;
                    break;
                } else if (ret != HCCL_E_NOT_SUPPORT) {
                    CheckStatusFuncResult("TriggerEnableP2P", ret);
                    return baseStatus_;
                }
            }
            // 非 PCIE 协议或自环链路：无需 enable p2p，直接进入 socket 建链
            aivUbStatus_ = AivUbMemTransportStatus::SOCKET_OK;
            break;
        case AivUbMemTransportStatus::P2P_ENABLING: {
            bool isEnabled = false;
            HcclResult ret = IsP2PEnabled(isEnabled);
            if (ret != HCCL_SUCCESS) {
                CheckStatusFuncResult("IsP2PEnabled", ret);
                return baseStatus_;
            }
            if (!isEnabled) {
                return baseStatus_; // 驱动侧尚未完成使能，等待下轮轮询
            }
            HCCL_INFO(
                "[AivUbMemTransport][%s] connected p2p success, local logic id:%u, remote physic id:%u.", __func__,
                localDeviceLogicId_, channelDesc_.remoteEndpoint.loc.device.devPhyId);
            aivUbStatus_ = AivUbMemTransportStatus::SOCKET_OK;
            break;
        }
        default:
            break;
    }

    bool isReady = false;
    if (UNLIKELY(IsSocketReady(isReady) != HCCL_SUCCESS)) {
        HCCL_ERROR("[%s] IsSocketReady fail, aivUbStatus_[%d], baseStatus_[%d]", __func__, aivUbStatus_, baseStatus_);
        baseStatus_ = Hccl::TransportStatus::INVALID;
        return baseStatus_;
    }
    if (!isReady) {
        return baseStatus_;
    }
    return UpdateStatus();
}

HcclResult AivUbMemTransport::TriggerEnableP2P()
{
    localDeviceLogicId_ = Hccl::HrtGetDevice();

    // enable p2p 不支持本设备与自身执行：自环链路无需使能，跳过 enable/wait 全流程
    uint32_t localDevicePhysicId = Hccl::HrtGetDevicePhyIdByUserDevId(localDeviceLogicId_);
    uint32_t remoteDevicePhyId = channelDesc_.remoteEndpoint.loc.device.devPhyId;
    if (remoteDevicePhyId == localDevicePhysicId) {
        HCCL_INFO(
            "[AivUbMemTransport][%s] self loop link, skip enable p2p. local physic id:%u, remote physic id:%u.",
            __func__, localDevicePhysicId, remoteDevicePhyId);
        return HCCL_E_NOT_SUPPORT;
    }

    // 按需对当前链路的对端设备执行 enable p2p，不再于通信域初始化阶段全量下发
    std::vector<uint32_t> remoteDevices{remoteDevicePhyId};
    HcclResult ret = Hccl::P2PEnableManager::GetInstance().EnableP2P(remoteDevices);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[AivUbMemTransport][%s] EnableP2P failed, ret:%d, remote physic id:%u.", __func__, ret, remoteDevicePhyId);
        return ret;
    }
    // enable p2p 成功下发后才置位：失败时 manager 引用计数未增加，不置位，
    // 保证 p2pEnableStarted_ 与引用计数严格对应，析构时仅对已持有的引用执行对称 disable
    p2pEnableStarted_ = true;
    return HCCL_SUCCESS;
}

HcclResult AivUbMemTransport::IsP2PEnabled(bool& isEnabled)
{
    // 通过 P2PEnableManager 的非阻塞接口单次查询驱动侧 p2p 状态，未使能时由状态机下轮轮询推进
    return Hccl::P2PEnableManager::GetInstance().WaitP2PEnabled(
        localDeviceLogicId_, channelDesc_.remoteEndpoint.loc.device.devPhyId, isEnabled);
}

Hccl::TransportStatus AivUbMemTransport::UpdateStatus()
{
    HCCL_INFO(
        "%s aivUbStatus_[%d], baseStatus_[%d] start, aivUbStatus_::SOCKET_OK[%d]", __func__, aivUbStatus_, baseStatus_,
        AivUbMemTransportStatus::SOCKET_OK);
    HcclResult ret;
    // INIT 已在 GetStatus 的 p2p 状态机中消费（转出为 P2P_ENABLING 或 SOCKET_OK），此处不会收到
    switch (aivUbStatus_) {
        case AivUbMemTransportStatus::SOCKET_OK:
            ret = SendDataSize();
            CheckStatusFuncResult("SendDataSize", ret);
            aivUbStatus_ = AivUbMemTransportStatus::SEND_DATA_SIZE;
            break;
        case AivUbMemTransportStatus::SEND_DATA_SIZE:
            ret = RecvDataSize();
            CheckStatusFuncResult("RecvDataSize", ret);
            aivUbStatus_ = AivUbMemTransportStatus::RECV_DATA_SIZE;
            break;
        case AivUbMemTransportStatus::RECV_DATA_SIZE:
            ret = SendMemInfo();
            CheckStatusFuncResult("SendMemInfo", ret);
            aivUbStatus_ = AivUbMemTransportStatus::SEND_MEM_INFO;
            break;
        case AivUbMemTransportStatus::SEND_MEM_INFO:
            ret = RecvMemInfo();
            CheckStatusFuncResult("RecvMemInfo", ret);
            aivUbStatus_ = AivUbMemTransportStatus::RECV_MEM_INFO;
            break;
        case AivUbMemTransportStatus::RECV_MEM_INFO:
            ret = RecvDataProcess();
            CheckStatusFuncResult("RecvDataProcess", ret);
            aivUbStatus_ = AivUbMemTransportStatus::RECV_MEM_FIN;
            break;
        case AivUbMemTransportStatus::RECV_MEM_FIN:
            aivUbStatus_ = AivUbMemTransportStatus::READY;
            baseStatus_ = Hccl::TransportStatus::READY;
            break;
        default:
            break;
    }
    HCCL_INFO("%s aivUbStatus_[%d], baseStatus_[%d]", __func__, aivUbStatus_, baseStatus_);
    return baseStatus_;
}

HcclResult AivUbMemTransport::SendDataSize()
{
    HCCL_INFO("[%s] start", __func__);

    Hccl::BinaryStream binaryStream;
    CHK_RET(BufferPack(binaryStream, localRmaBufferVec_));

    binaryStream.Dump(sendData_);
    u32 sendSize = sendData_.size();
    EXCEPTION_HANDLE_BEGIN
    socket_->SendAsync(&sendSize, sizeof(sendSize));
    EXCEPTION_HANDLE_END
    HCCL_INFO("[%s] finished", __func__);
    return HCCL_SUCCESS;
}

HcclResult AivUbMemTransport::RecvDataSize()
{
    HCCL_INFO("[%s] start", __func__);

    EXCEPTION_HANDLE_BEGIN
    socket_->RecvAsync(&exchangeDataSize_, sizeof(exchangeDataSize_));
    EXCEPTION_HANDLE_END
    HCCL_INFO("[%s] finished", __func__);
    return HCCL_SUCCESS;
}

HcclResult AivUbMemTransport::SendMemInfo()
{
    HCCL_INFO("[%s] start", __func__);

    EXCEPTION_HANDLE_BEGIN
    socket_->SendAsync(&sendData_[0], sendData_.size());
    EXCEPTION_HANDLE_END
    HCCL_INFO("[%s] finished", __func__);
    return HCCL_SUCCESS;
}

HcclResult
AivUbMemTransport::BufferPack(Hccl::BinaryStream& binaryStream, std::vector<Hccl::LocalIpcRmaBuffer*>& bufferVec) const
{
    u32 vecSize = bufferVec.size();
    binaryStream << vecSize;
    HCCL_INFO("BufferPack vecSize=%u", vecSize);

    for (uint32_t i = 0; i < vecSize; ++i) {
        std::unique_ptr<Hccl::Serializable> dto = bufferVec[i]->GetExchangeDto();
        CHK_PTR_NULL(dto);
        dto->Serialize(binaryStream);
        HCCL_INFO("[%s] dto[%s]", __func__, dto->Describe().c_str());
    }
    return HCCL_SUCCESS;
}

HcclResult AivUbMemTransport::RecvMemInfo()
{
    recvData_.resize(exchangeDataSize_);
    EXCEPTION_HANDLE_BEGIN
    socket_->RecvAsync(&recvData_[0], recvData_.size());
    EXCEPTION_HANDLE_END
    // HCCL_INFO("recv data, size=%llu, data=%s", data.size(), Hccl::Bytes2hex(data.data(), data.size()).c_str());
    return HCCL_SUCCESS;
}

HcclResult AivUbMemTransport::RecvDataProcess()
{
    Hccl::BinaryStream binaryStream(recvData_);
    rmtBufferVec_.clear();
    rmtRmaBufferVec_.clear();
    EXCEPTION_HANDLE_BEGIN
    RmtBufferUnpackProc(binaryStream);
    EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}

void AivUbMemTransport::RmtBufferUnpackProc(Hccl::BinaryStream& binaryStream)
{
    u32 vecSize{0};
    binaryStream >> vecSize;
    HCCL_INFO("vecSize=%u", vecSize);
    uint32_t totalBufferNum = rmtBufferVec_.size() + vecSize;
    if (UNLIKELY(totalBufferNum > MAX_BUFFER_NUM)) {
        EXCEPTION_THROW_IF_ERR(HCCL_E_PARA, "[AivUbMemTransport][RmtBufferUnpackProc] vecSize exceeds limit.");
    }

    for (u32 pos = 0; pos < vecSize; ++pos) {
        Hccl::ExchangeIpcBufferDto dto;
        dto.Deserialize(binaryStream);
        HCCL_INFO("[%s] dto[%s]", __func__, dto.Describe().c_str());
        if (dto.size == 0) { // size为0，则为 remote 空buffer
            HCCL_INFO("unpack nullptr, pos=%u", pos);
            rmtBufferVec_.push_back(nullptr);
            rmtRmaBufferVec_.push_back(nullptr);
        } else { // size非0，则构造一个remote buffer
            HCCL_INFO("[AivUbMemTransport][RmtBufferUnpackProc] unpack buffer memInfo[%s]", dto.memInfo.c_str());
            rmtBufferVec_.push_back(std::make_unique<Hccl::RemoteIpcRmaBuffer>(dto, channelDesc_.ubMemAttr.pathMode));
            rmtRmaBufferVec_.push_back(rmtBufferVec_.back().get());
        }
    }
}

HcclResult AivUbMemTransport::GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos)
{
    std::lock_guard<std::mutex> lock(remoteMemsMutex_);
    Hccl::RemoteMemCtx<std::unique_ptr<Hccl::RemoteIpcRmaBuffer>> remoteMemCtx{
        cacheValid_, rmtBufferVec_, remoteUserMems_, memInfoCopies_, memInfoPointers_, remoteMem, memInfos, memNum};
    CHK_RET(GetRemoteUserMems(remoteMemCtx));
    return HCCL_SUCCESS;
}

HcclResult AivUbMemTransport::CheckSocketStatus(std::string socketOperator)
{
    CHK_PTR_NULL(socket_);
    auto timeout = std::chrono::seconds(Hccl::EnvConfig::GetInstance().GetSocketConfig().GetLinkTimeOut());
    auto startTime = std::chrono::steady_clock::now();
    uint32_t retryCount = 0;
    while (true) {
        EXCEPTION_HANDLE_BEGIN
        Hccl::SocketStatus socketStatus = socket_->GetAsyncStatus();
        if (socketStatus == Hccl::SocketStatus::OK) {
            auto elapsed
                = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime)
                      .count();
            HCCL_INFO(
                "[AivUbMemTransport][%s] socket transport operation[%s] success, elapsed[%lld]ms, retryCount[%u]",
                __func__, socketOperator.c_str(), elapsed, retryCount);
            break;
        }
        if ((std::chrono::steady_clock::now() - startTime) >= timeout || socketStatus == Hccl::SocketStatus::TIMEOUT) {
            auto elapsed
                = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime)
                      .count();
            HCCL_ERROR(
                "[AivUbMemTransport][%s] socket transport operation[%s] timeout after %lld sec, elapsed[%lld]ms, "
                "retryCount[%u]",
                __func__, socketOperator.c_str(), timeout, elapsed, retryCount);
            return HCCL_E_TIMEOUT;
        }
        EXCEPTION_HANDLE_END
        retryCount++;
    }
    return HCCL_SUCCESS;
}

HcclResult AivUbMemTransport::UpdateMemInfo(HcommMemHandle* memHandles, uint32_t memHandleNum)
{
    if (memHandleNum == 0) {
        HCCL_WARNING("[AivUbMemTransport][UpdateMemInfo] bufferNum is 0.");
        return HCCL_SUCCESS;
    }
    locMemTemp_.clear();
    CHK_RET(FillBufferVec(memHandles, memHandleNum, locMemTemp_));
    HCCL_INFO("[AivUbMemTransport][UpdateMemInfo] bufferNum[%zu]", locMemTemp_.size());
    sendData_.clear();
    Hccl::BinaryStream sendStream;
    CHK_RET(BufferPack(sendStream, locMemTemp_));
    sendStream.Dump(sendData_);
    u32 sendSize = sendData_.size();
    EXCEPTION_HANDLE_BEGIN
    socket_->SendAsync(&sendSize, sizeof(sendSize));
    EXCEPTION_HANDLE_END
    CHK_RET(CheckSocketStatus("SendDataSize"));
    CHK_RET(RecvDataSize());
    CHK_RET(CheckSocketStatus("RecvDataSize"));
    CHK_RET(SendMemInfo());
    CHK_RET(CheckSocketStatus("SendMemInfo"));
    CHK_RET(RecvMemInfo());
    CHK_RET(CheckSocketStatus("RecvMemInfo"));
    Hccl::BinaryStream recvStream(recvData_);
    EXCEPTION_HANDLE_BEGIN
    RmtBufferUnpackProc(recvStream);
    EXCEPTION_HANDLE_END
    localRmaBufferVec_.insert(localRmaBufferVec_.end(), locMemTemp_.begin(), locMemTemp_.end());
    // 流程中已有新增内存数量判断，故执行到此位置一定存在新增内存，需要将标识置位false，使得再次调用GetRemoteMems时重新构造缓存
    cacheValid_ = false;
    return HCCL_SUCCESS;
}
} // namespace hcomm
