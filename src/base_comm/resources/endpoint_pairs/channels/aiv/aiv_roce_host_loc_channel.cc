/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/cast_utils_pub.h"
#include "endpoint.h"
#include "hcomm_c_adpt.h"
#include "exception_handler.h"
#include "mem_transport_common.h"
#include "acl_device_slab_guard.h"

#include "acl/acl_rt.h"

// Orion
#include "exchange_rdma_buffer_dto.h"
#include "orion_adapter_rts.h"
#include "aiv_roce_host_loc_channel.h"
#include "../../../../common/orion_adpt_utils.h"
#include "../../sockets/socket_mgr.h"
#include "adapter_rts.h"

namespace hcomm {

constexpr uint32_t TC_TEMP = 132;
constexpr uint32_t SL_TEMP = 4;
constexpr uint32_t RETRY_CNT_TEMP = 7;
constexpr uint32_t RETRY_TIME_TEMP = 20;

AivRoceHostLocChannel::AivRoceHostLocChannel(
    EndpointHandle endpointHandle, HcommChannelDesc channelDesc, CommEngine engine)
    : Channel(endpointHandle, channelDesc, engine, HcommChannelKind::AIV_ROCE, "AivRoceHostLocChannel")
{}

AivRoceHostLocChannel::~AivRoceHostLocChannel() { FreeDeviceMemories(); }

HcclResult AivRoceHostLocChannel::ParseInputParam()
{
    CHK_RET(Channel::ParseInputParam());
    return HCCL_SUCCESS;
}

HcclResult AivRoceHostLocChannel::DecideSocketRole()
{
    local_.isSocketServer = channelDesc_.exchangeAllMems && (channelDesc_.role != HCOMM_SOCKET_ROLE_CLIENT);
    return HCCL_SUCCESS;
}

HcclResult AivRoceHostLocChannel::StartListen()
{
    CHK_RET(Channel::StartListen());
    return HCCL_SUCCESS;
}

HcclResult AivRoceHostLocChannel::BuildSocket()
{
    CHK_RET(Channel::BuildSocket());
    return HCCL_SUCCESS;
}

HcclResult AivRoceHostLocChannel::BuildConnection()
{
    CHK_RET(Channel::BuildConnection());
    std::unique_ptr<RoceHostLocConnection> conn;
    EXCEPTION_CATCH(
        conn
        = std::make_unique<RoceHostLocConnection>(local_.socket, local_.rdmaHandle, channelDesc_.roceAttr.cqAttrFlags),
        return HCCL_E_INTERNAL);
    CHK_PTR_NULL(conn);
    CHK_RET(conn->Init());
    Hccl::QpInfo& qpInfo = conn->GetQpInfo();
    qpInfo.serviceLevel = channelDesc_.roceAttr.sl == 0xFF ? SL_TEMP : channelDesc_.roceAttr.sl;
    qpInfo.trafficClass = channelDesc_.roceAttr.tc == 0xFF ? TC_TEMP : channelDesc_.roceAttr.tc;
    qpInfo.retryCnt = channelDesc_.roceAttr.retryCnt == 0xFFFFFFFF ? RETRY_CNT_TEMP : channelDesc_.roceAttr.retryCnt;
    qpInfo.retryInterval
        = channelDesc_.roceAttr.retryInterval == 0xFFFFFFFF ? RETRY_TIME_TEMP : channelDesc_.roceAttr.retryInterval;
    HCCL_INFO(
        "[%s::%s] QpInfo: serviceLevel[%u], trafficClass[%u], retryCnt[%u], "
        "retryInterval[%u].",
        channelClassName_, __func__, qpInfo.serviceLevel, qpInfo.trafficClass, qpInfo.retryCnt, qpInfo.retryInterval);
    connections_.emplace_back(std::move(conn));
    connNum_ = connections_.size();
    HCCL_INFO("[AivRoceHostLocChannel::%s] connection num [%u]", __func__, connNum_);
    return HCCL_SUCCESS;
}

HcclResult AivRoceHostLocChannel::BuildBuffer()
{
    CHK_RET(Channel::BuildBuffer());
    if (channelDesc_.exchangeAllMems) {
        // Get memHandles from endpoint
        HCCL_INFO("[%s::%s] exchangeAllMems == True. Get memHandles from endpoint.", channelClassName_, __func__);
        std::shared_ptr<Hccl::LocalRdmaRmaBuffer>* memHandles = nullptr;
        uint32_t memHandleNum = 0;
        CHK_RET(static_cast<HcclResult>(
            HcommMemGetAllMemHandles(endpointHandle_, ReinterpretAs<void**>(&memHandles), &memHandleNum)));
        HCCL_INFO("[%s::%s] Got memHandleNum[%u].", channelClassName_, __func__, memHandleNum);
        for (uint32_t i = 0; i < memHandleNum; ++i) {
            std::shared_ptr<Hccl::LocalRdmaRmaBuffer>& localRdmaBuffer = memHandles[i];
            CHK_SMART_PTR_NULL(localRdmaBuffer);
            Hccl::Buffer* buf = localRdmaBuffer->GetBuf();
            CHK_PTR_NULL(buf);
            HCCL_INFO(
                "[%s::%s] Got memHandle No.%u: addr[0x%llx], size[0x%llx], memType[%d], memInfo[%s].",
                channelClassName_, __func__, i, static_cast<unsigned long long>(localRdmaBuffer->GetAddr()),
                static_cast<unsigned long long>(localRdmaBuffer->GetSize()), static_cast<int>(buf->GetMemType()),
                buf->GetMemInfo().c_str());
            localRmaBuffers_.emplace_back(localRdmaBuffer.get());
        }
    } else {
        // 从 channelDesc 的 memHandle，获得 localRmaBuffers_
        HCCL_INFO("[%s::%s] exchangeAllMems == false. Get memHandles from channelDesc.", channelClassName_, __func__);
        CHK_PTR_NULL(channelDesc_.memHandles);
        for (uint32_t i = 0; i < channelDesc_.memHandleNum; ++i) {
            CHK_PTR_NULL(channelDesc_.memHandles[i]);
            auto* localRdmaBuffer = ReinterpretAs<Hccl::LocalRdmaRmaBuffer*>(channelDesc_.memHandles[i]);
            HCCL_INFO(
                "[%s::%s] Got memHandle No.%u: addr[0x%llx], size[0x%llx], memType[%d], memInfo[%s].",
                channelClassName_, __func__, i, static_cast<unsigned long long>(localRdmaBuffer->GetAddr()),
                static_cast<unsigned long long>(localRdmaBuffer->GetSize()),
                static_cast<int>(localRdmaBuffer->GetBuf()->GetMemType()),
                localRdmaBuffer->GetBuf()->GetMemInfo().c_str());
            localRmaBuffers_.emplace_back(localRdmaBuffer);
        }
    }
    bufferNum_ = localRmaBuffers_.size();
    HCCL_INFO("[%s::%s] buffer num [%u]", channelClassName_, __func__, bufferNum_);
    return HCCL_SUCCESS;
}

HcclResult AivRoceHostLocChannel::GetNotifyNum(uint32_t* notifyNum) const
{
    CHK_PTR_NULL(notifyNum);
    *notifyNum = 0; // AIV 模式无 notify
    return HCCL_SUCCESS;
}

HcclResult AivRoceHostLocChannel::GetBufferNum(uint32_t* bufferNum) const
{
    CHK_PTR_NULL(bufferNum);
    *bufferNum = bufferNum_;
    return HCCL_SUCCESS;
}

HcclResult AivRoceHostLocChannel::GetQpNum(uint32_t* qpNum) const
{
    CHK_PTR_NULL(qpNum);
    *qpNum = connNum_;
    return HCCL_SUCCESS;
}

HcclResult AivRoceHostLocChannel::GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos)
{
    return remote_.memCache.Get(rmtRmaBuffers_, memNum, remoteMem, memInfos);
}

// 当前AICPU和框架没有改为返回错误码形式，所有暂时使用该方法转换
ChannelStatus AivRoceHostLocChannel::GetStatus()
{
    ChannelStatus status;
    HcclResult ret = GetStatus(status);
    if (ret != HCCL_SUCCESS && ret != HCCL_E_AGAIN) {
        HCCL_ERROR("[%s::%s] get status exception occurred, HcclResult=[%d]", channelClassName_, __func__, ret);
        return ChannelStatus::FAILED;
    }
    return status;
}

HcclResult AivRoceHostLocChannel::ProcessStatus()
{
    switch (channelStatus_) {
        case ChannelStatus::READY:
            return HCCL_SUCCESS;
        case ChannelStatus::SOCKET_TIMEOUT:
            HCCL_ERROR("[AivRoceHostLocChannel::ProcessStatus] get socket timeout");
            return HCCL_E_ROCE_CONNECT;
        default:
            return HCCL_E_AGAIN;
    }
}

HcclResult AivRoceHostLocChannel::GetStatus(ChannelStatus& status)
{
    switch (rdmaStatus_) {
        case RdmaStatus::INIT:
            // 检查socket状态
            CHK_RET(CheckSocketStatus());
            break;
        case RdmaStatus::SOCKET_OK:
            // 准备资源
            CHK_RET(CreateQp());
            rdmaStatus_ = RdmaStatus::QP_CREATED;
            break;
        case RdmaStatus::QP_CREATED:
            // 发送交换数据
            CHK_RET(ExchangeData());
            rdmaStatus_ = RdmaStatus::DATA_EXCHANGE;
            break;
        case RdmaStatus::DATA_EXCHANGE:
            CHK_RET(ModifyQp());
            rdmaStatus_ = RdmaStatus::QP_MODIFIED;
            [[fallthrough]];
        case RdmaStatus::QP_MODIFIED:
        default:
            rdmaStatus_ = RdmaStatus::CONN_OK;
            channelStatus_ = ChannelStatus::READY;
    }

    status = channelStatus_;
    return ProcessStatus();
}

HcclResult AivRoceHostLocChannel::CheckSocketStatus()
{
    CHK_PTR_NULL(local_.socket);
    Hccl::SocketStatus socketStatus = local_.socket->GetStatus(); // socket状态机
    HCCL_DEBUG("[AivRoceHostLocChannel::CheckSocketStatus] socket status = %s", socketStatus.Describe().c_str());
    if (socketStatus == Hccl::SocketStatus::OK) {
        rdmaStatus_ = RdmaStatus::SOCKET_OK;
        channelStatus_ = ChannelStatus::SOCKET_OK;
    } else if (socketStatus == Hccl::SocketStatus::TIMEOUT) {
        channelStatus_ = ChannelStatus::SOCKET_TIMEOUT;
    }
    return HCCL_SUCCESS;
}

// 准备资源（创建QP）
HcclResult AivRoceHostLocChannel::CreateQp()
{
    for (auto& conn : connections_) {
        CHK_PTR_NULL(conn);
        HcclResult ret = conn->CreateQp();
        if (ret == HCCL_E_AGAIN) {
            return HCCL_SUCCESS;
        }
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
    }
    HCCL_INFO("[AivRoceHostLocChannel::%s] all connections resources connected.", __func__);
    return HCCL_SUCCESS;
}

// 交换数据
HcclResult AivRoceHostLocChannel::ExchangeData()
{
    HCCL_INFO(
        "[AivRoceHostLocChannel::%s] Start to SendExchangeData, bufferNum=%u, connNum=%u", __func__, bufferNum_,
        connNum_);

    // 同步数据打包
    Hccl::BinaryStream binaryStream;
    CHK_RET(BufferVecPack(binaryStream));
    CHK_RET(ConnVecPack(binaryStream));

    std::vector<char> sendData{};
    binaryStream.Dump(sendData);
    uint64_t sendSize = sendData.size();
    std::vector<char> recvData{};
    uint64_t recvSize = 0;

    EXCEPTION_HANDLE_BEGIN
    // 同步发送数据包尺寸
    CHK_PRT_RET(
        !local_.socket->Send(static_cast<void*>(&sendSize), sizeof(sendSize)),
        HCCL_ERROR("[AivRoceHostLocChannel::%s] Send sendSize failed", __func__), HCCL_E_NETWORK);
    HCCL_INFO(
        "[AivRoceHostLocChannel::%s] Send size[%llu] of data success. [%llu] bytes sent.", __func__, sendSize,
        sizeof(sendSize));

    // 同步接收数据包尺寸
    CHK_PRT_RET(
        !local_.socket->Recv(static_cast<void*>(&recvSize), sizeof(recvSize)),
        HCCL_ERROR("[AivRoceHostLocChannel::%s] Recv recvSize failed", __func__), HCCL_E_NETWORK);
    HCCL_INFO(
        "[AivRoceHostLocChannel::%s] Receive size[%llu] of data success. [%llu] bytes received.", __func__, recvSize,
        sizeof(recvSize));

    // 对端声明的数据长度不可信，为0或超过上限时拒绝分配，防止超大内存申请导致OOM
    CHK_PRT_RET(
        recvSize == 0 || recvSize > MAX_EXCHANGE_DATA_SIZE,
        HCCL_ERROR(
            "[AivRoceHostLocChannel::%s] recvSize[%llu] is invalid, limit[%llu]", __func__, recvSize,
            MAX_EXCHANGE_DATA_SIZE),
        HCCL_E_PARA);

    // 同步发送数据
    CHK_PRT_RET(
        !local_.socket->Send(static_cast<void*>(sendData.data()), sendSize),
        HCCL_ERROR("[AivRoceHostLocChannel::%s] Send exchange data failed", __func__), HCCL_E_NETWORK);
    HCCL_INFO("[AivRoceHostLocChannel::%s] Send Exchange Data success. [%llu] bytes sent.", __func__, sendSize);

    // 同步接收数据
    HCCL_INFO("[AivRoceHostLocChannel::%s] Start to Receive Exchange Data", __func__);
    recvData.resize(recvSize);
    CHK_PRT_RET(
        !local_.socket->Recv(static_cast<void*>(recvData.data()), recvSize),
        HCCL_ERROR("[AivRoceHostLocChannel::%s] Recv exchange data failed", __func__), HCCL_E_NETWORK);
    HCCL_INFO("[AivRoceHostLocChannel::%s] Receive Exchange Data success. [%llu] bytes received.", __func__, recvSize);
    EXCEPTION_HANDLE_END

    // 同步数据解包
    Hccl::BinaryStream recvBinStream(recvData);
    CHK_RET(RmtBufferVecUnpackProc(recvBinStream));
    CHK_RET(ConnVecUnpackProc(recvBinStream));

    HCCL_INFO("[AivRoceHostLocChannel::%s] Unpack exchange Data success. ", __func__);
    return HCCL_SUCCESS;
}

HcclResult AivRoceHostLocChannel::BufferVecPack(Hccl::BinaryStream& binaryStream)
{
    binaryStream << bufferNum_;
    HCCL_INFO("[AivRoceHostLocChannel::%s] start to pack RmaBuffers", __func__);
    u32 pos = 0;
    for (auto& it : localRmaBuffers_) {
        binaryStream << pos;
        if (it != nullptr) { // 非空的buffer，从buffer中获取 dto
            std::unique_ptr<Hccl::Serializable> dto = it->GetExchangeDto();
            dto->Serialize(binaryStream);
            HCCL_INFO("pack buffer pos=%u dto %s", pos, dto->Describe().c_str());
        } else { // 空的buffer，dto所有字段为0(size=0)
            Hccl::ExchangeRdmaBufferDto exchangeDto;
            exchangeDto.Serialize(binaryStream);
            HCCL_INFO("pack buffer pos=%u, dto is null %s", pos, exchangeDto.Describe().c_str());
        }
        pos++;
    }
    HCCL_INFO("[AivRoceHostLocChannel::%s] pack RmaBuffers finish", __func__);
    return HCCL_SUCCESS;
}

HcclResult AivRoceHostLocChannel::ConnVecPack(Hccl::BinaryStream& binaryStream)
{
    binaryStream << connNum_;
    HCCL_INFO("[AivRoceHostLocChannel::%s] start to pack connections", __func__);
    u32 pos = 0;
    for (auto& it : connections_) {
        binaryStream << pos;
        std::unique_ptr<Hccl::Serializable> dto = nullptr;
        CHK_RET(it->GetExchangeDto(dto));
        dto->Serialize(binaryStream);
        HCCL_INFO("pack connection pos=%u, dto %s", pos, dto->Describe().c_str());
        pos++;
    }
    HCCL_INFO("[AivRoceHostLocChannel::%s] pack connections finish", __func__);
    return HCCL_SUCCESS;
}

HcclResult AivRoceHostLocChannel::RmtBufferVecUnpackProc(Hccl::BinaryStream& binaryStream)
{
    u32 rmtNum;
    binaryStream >> rmtNum;

    HCCL_INFO("[AivRoceHostLocChannel::%s] bufferNum_=%u, rmtNum=%u", __func__, bufferNum_, rmtNum);

    // 对端声明的buffer数量不可信，超过上限时拒绝，防止直接resize导致超大内存申请
    CHK_PRT_RET(
        rmtNum > MAX_BUFFER_NUM,
        HCCL_ERROR("[AivRoceHostLocChannel::%s] rmtNum[%u] exceeds maxNum[%u]", __func__, rmtNum, MAX_BUFFER_NUM),
        HCCL_E_PARA);

    rmtRmaBuffers_.resize(rmtNum);
    for (u32 i = 0; i < rmtNum; i++) {
        u32 pos;
        binaryStream >> pos;
        if (pos >= rmtNum) {
            HCCL_ERROR("[AivRoceHostLocChannel::%s] pos=%u out of range (rmtNum=%u)", __func__, pos, rmtNum);
            return HCCL_E_INTERNAL;
        }
        Hccl::ExchangeRdmaBufferDto dto;
        dto.Deserialize(binaryStream);

        HCCL_INFO("[AivRoceHostLocChannel::%s] pos=%u, dto %s", __func__, pos, dto.Describe().c_str());
        EXCEPTION_CATCH(
            rmtRmaBuffers_[pos] = std::make_unique<Hccl::RemoteRdmaRmaBuffer>(local_.rdmaHandle, dto),
            HCCL_ERROR(
                "[AivRoceHostLocChannel::%s] make_unique<Hccl::RemoteRdmaRmaBuffer> throws an exception!", __func__);
            return HCCL_E_INTERNAL);
        HCCL_INFO(
            "[AivRoceHostLocChannel::%s] pos=%u, rmtRmaBuffer=%s", __func__, pos,
            rmtRmaBuffers_[pos]->Describe().c_str());
    }

    return HCCL_SUCCESS;
}

HcclResult AivRoceHostLocChannel::ConnVecUnpackProc(Hccl::BinaryStream& binaryStream)
{
    u32 rmtConnNum;
    binaryStream >> rmtConnNum;
    HCCL_INFO("start unpack conn, connNum=%u, rmtConnNum=%u", connNum_, rmtConnNum);
    if (connNum_ != rmtConnNum) {
        HCCL_ERROR("connNum=%u is not equal to rmtConnNum=%u", connNum_, rmtConnNum);
        return HCCL_E_ROCE_CONNECT;
    }

    for (u32 i = 0; i < rmtConnNum; i++) {
        u32 pos;
        binaryStream >> pos;
        rmtConnDto_.Deserialize(binaryStream);
    }
    return HCCL_SUCCESS;
}

HcclResult AivRoceHostLocChannel::ModifyQp()
{
    for (auto& conn : connections_) {
        CHK_PTR_NULL(conn);
        CHK_RET(conn->ParseRmtExchangeDto(rmtConnDto_));
        HcclResult ret = conn->ModifyQp();
        if (ret == HCCL_E_AGAIN) {
            return HCCL_SUCCESS;
        }
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
    }
    HCCL_INFO("[AivRoceHostLocChannel::%s] all connections resources modify success.", __func__);
    return HCCL_SUCCESS;
}

HcclResult AivRoceHostLocChannel::Clean()
{
    aicpuTsEntity_.ReleaseDeviceEntitySlab();
    return HCCL_SUCCESS;
}

HcclResult AivRoceHostLocChannel::Resume() { return HCCL_SUCCESS; }

std::string AivRoceHostLocChannel::Describe() const
{
    std::string msg = "AivRoceHostLocChannel{";
    msg += Hccl::StringFormat("bufferNum:%u, localRmaBuffers:[", bufferNum_);
    for (auto& buf : localRmaBuffers_) {
        msg += buf->Describe();
        msg += ", ";
    }
    msg += "]";
    msg += Hccl::StringFormat(", connNum:%u, connections:[", connNum_);
    for (auto& conn : connections_) {
        msg += conn->Describe();
        msg += ", ";
    }
    msg += "]";
    msg += Hccl::StringFormat(", rdmaHandle:%p, %s, ", local_.rdmaHandle, channelStatus_.Describe().c_str());
    if (local_.socket != nullptr) {
        msg += local_.socket->Describe();
    }
    msg += "}";
    return msg;
}

HcclResult AivRoceHostLocChannel::BuildAndGetLocNotifyInfo([[maybe_unused]] RegedNotifyEntity** notify) const
{
    return HCCL_SUCCESS;
}

HcclResult AivRoceHostLocChannel::BuildAndGetRmtNotifyInfo([[maybe_unused]] RegedNotifyEntity** notify) const
{
    // AIV 模式无 notify
    return HCCL_SUCCESS;
}

HcclResult AivRoceHostLocChannel::BuildAndGetRmtBufInfo(
    std::vector<RegedBufferEntity>& bufList, RegedBufferEntity** bufferEntityPtr)
{
    if (channelStatus_ != ChannelStatus::READY) {
        HCCL_ERROR(
            "[AivRoceHostLocChannel::%s] channel status[%d] is not ready[%d], please check.", __func__, channelStatus_,
            ChannelStatus::READY);
        return HCCL_E_INTERNAL;
    }

    if (bufferNum_ == 0) {
        HCCL_INFO("[AivRoceHostLocChannel::%s] No Remote memory regions available", __func__);
        return HCCL_SUCCESS;
    }

    if (bufferEntityPtr == nullptr) {
        HCCL_ERROR("[AivRoceHostLocChannel::%s] input param is null", __func__);
        return HCCL_E_PARA;
    }

    for (uint32_t i = 0; i < bufferNum_; i++) {
        auto& rmtRmaBuffer = rmtRmaBuffers_[i];
        bufList[i].type = REGED_BUFFER_RMA;
        bufList[i].bufferInfo.rma.addr = static_cast<uint64_t>(rmtRmaBuffer->GetAddr());
        bufList[i].bufferInfo.rma.size = rmtRmaBuffer->GetSize();
        bufList[i].bufferInfo.rma.protectionInfo.type = PROTECTION_TYPE_ROCE;
        bufList[i].bufferInfo.rma.protectionInfo.memInfo.roce.rkey = rmtRmaBuffer->GetRkey();
        HCCL_INFO(
            "[AivRoceHostLocChannel::%s] rmtBuf[addr[%p], size[%lu]]", __func__, bufList[i].bufferInfo.rma.addr,
            bufList[i].bufferInfo.rma.size);
    }
    *bufferEntityPtr = bufList.data();
    return HCCL_SUCCESS;
}

HcclResult AivRoceHostLocChannel::BuildAndGetLocBufInfo(
    std::vector<RegedBufferEntity>& bufList, RegedBufferEntity** bufferEntityPtr)
{
    if (channelStatus_ != ChannelStatus::READY) {
        HCCL_ERROR(
            "[AivRoceHostLocChannel::%s] channel status[%d] is not ready[%d], please check.", __func__, channelStatus_,
            ChannelStatus::READY);
        return HCCL_E_INTERNAL;
    }

    if (bufferNum_ == 0) {
        HCCL_INFO("[AivRoceHostLocChannel::%s] No local memory regions available", __func__);
        return HCCL_SUCCESS;
    }

    if (bufferEntityPtr == nullptr) {
        HCCL_ERROR("[AivRoceHostLocChannel::%s] input param is null", __func__);
        return HCCL_E_PARA;
    }

    for (uint32_t i = 0; i < bufferNum_; i++) {
        auto& locRmaBuffer = localRmaBuffers_[i];
        bufList[i].type = REGED_BUFFER_RMA;
        bufList[i].bufferInfo.rma.addr = static_cast<uint64_t>(locRmaBuffer->GetAddr());
        bufList[i].bufferInfo.rma.size = locRmaBuffer->GetSize();
        bufList[i].bufferInfo.rma.protectionInfo.type = PROTECTION_TYPE_ROCE;
        bufList[i].bufferInfo.rma.protectionInfo.memInfo.roce.lkey = locRmaBuffer->GetLkey();
        HCCL_INFO(
            "[AivRoceHostLocChannel::%s] locBuf[addr[%p], size[%lu]]", __func__, bufList[i].bufferInfo.rma.addr,
            bufList[i].bufferInfo.rma.size);
    }
    *bufferEntityPtr = bufList.data();
    return HCCL_SUCCESS;
}

HcclResult AivRoceHostLocChannel::BuildAndGetSqContext(std::vector<SqContext>& sqList, SqContext** sqContextPtr)
{
    if (channelStatus_ != ChannelStatus::READY) {
        HCCL_ERROR(
            "[AivRoceHostLocChannel::%s] channel status[%d] is not ready[%d], please check.", __func__, channelStatus_,
            ChannelStatus::READY);
        return HCCL_E_INTERNAL;
    }

    if (connNum_ == 0) {
        HCCL_INFO("[AivRoceHostLocChannel::%s] No conn available", __func__);
        return HCCL_SUCCESS;
    }

    if (sqContextPtr == nullptr) {
        HCCL_ERROR("[AivRoceHostLocChannel::%s] input param is null", __func__);
        return HCCL_E_PARA;
    }

    for (uint32_t i = 0; i < connNum_; i++) {
        auto& conn = connections_[i];
        CHK_PTR_NULL(conn);
        SqContext sqContext{};
        CHK_RET(conn->BuildSqContext(&sqContext));
        sqList[i] = sqContext;
    }
    *sqContextPtr = sqList.data();
    return HCCL_SUCCESS;
}

HcclResult AivRoceHostLocChannel::BuildAndGetCqContext(std::vector<CqContext>& cqList, CqContext** cqContextPtr)
{
    if (channelStatus_ != ChannelStatus::READY) {
        HCCL_ERROR(
            "[AivRoceHostLocChannel::%s] channel status[%d] is not ready[%d], please check.", __func__, channelStatus_,
            ChannelStatus::READY);
        return HCCL_E_INTERNAL;
    }

    if (connNum_ == 0) {
        HCCL_INFO("[AivRoceHostLocChannel::%s] No conn available", __func__);
        return HCCL_SUCCESS;
    }

    if (cqContextPtr == nullptr) {
        HCCL_ERROR("[AivRoceHostLocChannel::%s] input param is null", __func__);
        return HCCL_E_PARA;
    }

    for (uint32_t i = 0; i < connNum_; i++) {
        auto& conn = connections_[i];
        CHK_PTR_NULL(conn);
        CqContext cqContext{};
        CHK_RET(conn->BuildCqContext(&cqContext));
        cqList[i] = cqContext;
    }
    *cqContextPtr = cqList.data();
    return HCCL_SUCCESS;
}

HcclResult AivRoceHostLocChannel::BuildHostEntity(
    ChannelEntity& hostEntity, std::vector<RegedBufferEntity>& locBufList, std::vector<RegedBufferEntity>& rmtBufList,
    std::vector<SqContext>& sqList, std::vector<CqContext>& cqList)
{
    hostEntity.abiHeader.version = HCCL_CHANNEL_VERSION;
    hostEntity.abiHeader.magicWord = HCCL_CHANNEL_MAGIC_WORD;
    hostEntity.abiHeader.size = sizeof(ChannelEntity);
    hostEntity.abiHeader.reserved = 0;
    hostEntity.engine = COMM_ENGINE_AIV;
    hostEntity.protocol = GetCommProtocol();
    hostEntity.devInfo = HcommDevInfo{};
    hostEntity.devInfo.devPhyId = local_.devicePhyId;

    hostEntity.localNotifyNum = 0;
    CHK_RET(BuildAndGetLocNotifyInfo(&hostEntity.localNotifyAddr));
    hostEntity.remoteNotifyNum = 0;
    CHK_RET(BuildAndGetRmtNotifyInfo(&hostEntity.remoteNotifyAddr));

    locBufList.resize(bufferNum_);
    hostEntity.localBufferNum = bufferNum_;
    CHK_RET(BuildAndGetLocBufInfo(locBufList, &hostEntity.localBufferAddr));

    rmtBufList.resize(bufferNum_);
    hostEntity.remoteBufferNum = bufferNum_;
    CHK_RET(BuildAndGetRmtBufInfo(rmtBufList, &hostEntity.remoteBufferAddr));

    sqList.resize(connNum_);
    hostEntity.sqNum = connNum_;
    CHK_RET(BuildAndGetSqContext(sqList, &hostEntity.sqContextAddr));

    cqList.resize(connNum_);
    hostEntity.cqNum = connNum_;
    CHK_RET(BuildAndGetCqContext(cqList, &hostEntity.cqContextAddr));

    return HCCL_SUCCESS;
}

void AivRoceHostLocChannel::FreeDeviceMemories() { aicpuTsEntity_.ReleaseDeviceEntitySlab(); }

AivRoceHostLocChannelEntity::AivRoceHostLocChannelEntity(AivRoceHostLocChannel& owner) : owner_(owner) {}

HcclResult AivRoceHostLocChannelEntity::BuildChannelEntityToDevice(void** devChannelPtr)
{
    CHK_PTR_NULL(devChannelPtr);

    if (devChannelEntitySlab_ != nullptr) {
        *devChannelPtr = devChannelEntitySlab_;
        HCCL_INFO("[AivRoceHostLocChannelEntity::%s] already built, return cached devPtr=%p", __func__, *devChannelPtr);
        return HCCL_SUCCESS;
    }

    ChannelEntity hostEntity{};
    std::vector<RegedBufferEntity> locBufList;
    std::vector<RegedBufferEntity> rmtBufList;
    std::vector<SqContext> sqList;
    std::vector<CqContext> cqList;
    CHK_RET(owner_.BuildHostEntity(hostEntity, locBufList, rmtBufList, sqList, cqList));

    DeviceChannelEntityLayout layout;
    CHK_RET(BuildDeviceChannelEntityLayout(hostEntity, layout, false));
    void* slabPtr = nullptr;
    CHK_RET(AllocDeviceEntitySlab(layout.slabSize, slabPtr));
    AclDeviceSlabGuard slabGuard;
    slabGuard.Reset(slabPtr, layout.slabSize);

    ChannelEntity devEntity;
    CHK_RET(CopyChannelEntityArraysToSlab(slabPtr, hostEntity, layout, devEntity));
    void* entityDevPtr = nullptr;
    CHK_RET(CopyChannelEntityHeaderToSlab(slabPtr, layout, devEntity, entityDevPtr));

    ReleaseDeviceEntitySlab();
    devChannelEntitySlab_ = slabGuard.Release();
    devChannelEntitySlabSize_ = layout.slabSize;
    devChannelEntity_ = entityDevPtr;

    *devChannelPtr = entityDevPtr;
    HCCL_INFO(
        "[AivRoceHostLocChannelEntity::%s] Success, devPtr=%p, slabPtr=%p, slabSize=%zu", __func__, *devChannelPtr,
        devChannelEntitySlab_, devChannelEntitySlabSize_);
    return HCCL_SUCCESS;
}

HcclResult AivRoceHostLocChannelEntity::PreAllocChannelEntityToDevice(void** devChannelPtr)
{
    CHK_PTR_NULL(devChannelPtr);

    if (devChannelEntitySlab_ != nullptr) {
        *devChannelPtr = devChannelEntitySlab_;
        HCCL_INFO("[AivRoceHostLocChannelEntity::%s] already built, return cached devPtr=%p", __func__, *devChannelPtr);
        return HCCL_SUCCESS;
    }

    ChannelEntity tmp{};
    tmp.localNotifyNum = 0;
    tmp.remoteNotifyNum = 0;
    tmp.localBufferNum = owner_.bufferNum_;
    tmp.remoteBufferNum = owner_.bufferNum_;
    tmp.sqNum = owner_.connNum_;
    tmp.cqNum = owner_.connNum_;

    DeviceChannelEntityLayout layout;
    CHK_RET(BuildDeviceChannelEntityLayout(tmp, layout, false));

    void* slabPtr = nullptr;
    CHK_RET(AllocDeviceEntitySlab(layout.slabSize, slabPtr));
    AclDeviceSlabGuard slabGuard;
    slabGuard.Reset(slabPtr, layout.slabSize);

    devChannelEntitySlab_ = slabGuard.Release();
    devChannelEntitySlabSize_ = layout.slabSize;
    devChannelEntity_ = GetSlabPtr(devChannelEntitySlab_, layout.entitySection);
    *devChannelPtr = devChannelEntitySlab_;

    HCCL_INFO(
        "[AivRoceHostLocChannelEntity::%s] pre-alloc success, slabPtr=%p, slabSize=%zu", __func__,
        devChannelEntitySlab_, devChannelEntitySlabSize_);
    return HCCL_SUCCESS;
}

HcclResult AivRoceHostLocChannelEntity::FillChannelEntityToDevice()
{
    if (devChannelEntitySlab_ == nullptr) {
        HCCL_ERROR("[AivRoceHostLocChannelEntity::%s] devChannelEntitySlab_ is nullptr, not pre-allocated.", __func__);
        return HCCL_E_INTERNAL;
    }
    if (owner_.channelStatus_ != ChannelStatus::READY) {
        HCCL_ERROR(
            "[AivRoceHostLocChannelEntity::%s] channel status[%d] is not ready[%d], please check.", __func__,
            owner_.channelStatus_, ChannelStatus::READY);
        return HCCL_E_INTERNAL;
    }

    ChannelEntity hostEntity{};
    std::vector<RegedBufferEntity> locBufList;
    std::vector<RegedBufferEntity> rmtBufList;
    std::vector<SqContext> sqList;
    std::vector<CqContext> cqList;
    CHK_RET(owner_.BuildHostEntity(hostEntity, locBufList, rmtBufList, sqList, cqList));

    DeviceChannelEntityLayout layout;
    CHK_RET(BuildDeviceChannelEntityLayout(hostEntity, layout, false));
    CHK_RET(CheckSlabCapacity(layout));

    ChannelEntity devEntity;
    CHK_RET(CopyChannelEntityArraysToSlab(devChannelEntitySlab_, hostEntity, layout, devEntity));
    void* entityDevPtr = nullptr;
    CHK_RET(CopyChannelEntityHeaderToSlab(devChannelEntitySlab_, layout, devEntity, entityDevPtr));

    devChannelEntity_ = entityDevPtr;
    HCCL_INFO("[AivRoceHostLocChannelEntity::%s] fill success, devPtr=%p", __func__, entityDevPtr);
    return HCCL_SUCCESS;
}
HcclResult AivRoceHostLocChannel::BuildAndGetDevChannelEntity(uint64_t* devChannelEntity)
{
    void* devChannelPtr = nullptr;
    CHK_RET(GetChannelEntity()->BuildChannelEntityToDevice(&devChannelPtr));
    CHK_PTR_NULL(devChannelPtr);
    *devChannelEntity = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(devChannelPtr));
    return HcclResult::HCCL_SUCCESS;
}

} // namespace hcomm
