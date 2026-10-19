/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cast_utils.h"
#include "endpoint.h"
#include "aicpu_res_package_helper.h"
#include "hcomm_c_adpt.h"
#include "exception_handler.h"
#include "mem_transport_common.h"

// Orion
#include "exchange_rdma_buffer_dto.h"
#include "dev_capability.h"
#include "orion_adapter_rts.h"
#include "aicpu_ts_roce_host_loc_channel.h"
#include "../../../../../common/orion_adpt_utils.h"
#include "../../../sockets/socket_mgr.h"
#include "user_remote_mem_getter.h"

namespace hcomm {

constexpr uint16_t DEFAULT_LISTENING_PORT = 60001;
constexpr uint32_t TC_TEMP = 132;
constexpr uint32_t SL_TEMP = 4;
constexpr uint32_t RETRY_CNT_TEMP = 7;
constexpr uint32_t RETRY_TIME_TEMP = 20;

AicpuTsRoceHostLocChannel::AicpuTsRoceHostLocChannel(EndpointHandle endpointHandle, HcommChannelDesc channelDesc)
    : endpointHandle_(endpointHandle),
      channelDesc_(channelDesc),
      remoteEp_{}
{}

AicpuTsRoceHostLocChannel::~AicpuTsRoceHostLocChannel()
{
    if (channelDesc_.socket == nullptr && socket_ != nullptr && socketConfig_ != nullptr) {
        SocketMgr::GetInstance(devicePhyId_).DestroySocket(*socketConfig_);
        socket_ = nullptr;
    }
}

HcclResult AicpuTsRoceHostLocChannel::ParseInputParam()
{
    // 1. 从 endpointHandle_，获得 localEp_ 和 rdmaHandle_
    CHK_PTR_NULL(endpointHandle_);
    HCCL_INFO(
        "[AicpuTsRoceHostLocChannel][%s] Start. endpointHandle[0x%llx]", __func__,
        ReinterpretAs<uint64_t>(endpointHandle_));
    Endpoint* localEpPtr = static_cast<Endpoint*>(endpointHandle_);
    localEp_ = localEpPtr->GetEndpointDesc();
    rdmaHandle_ = localEpPtr->GetRdmaHandle();
    CHK_PTR_NULL(rdmaHandle_);

    // 2. 从 channelDesc_，获得 remoteEp_, socket_ 和 notifyNum_
    remoteEp_ = channelDesc_.remoteEndpoint;
    socket_ = static_cast<Hccl::Socket*>(channelDesc_.socket);
    notifyNum_ = channelDesc_.notifyNum;

    return HCCL_SUCCESS;
}

HcclResult AicpuTsRoceHostLocChannel::StartListen()
{
    uint16_t port = channelDesc_.port;
    HCCL_INFO(
        "[AicpuTsRoceHostLocChannel::%s] Start. EndpointHandle[0x%llx], port[%u]", __func__,
        ReinterpretAs<uint64_t>(endpointHandle_), port);
    if (port == 0) {
        port = DEFAULT_LISTENING_PORT;
        HCCL_INFO("[AicpuTsRoceHostLocChannel::%s] channelDesc port is 0, use default port [%u]", __func__, port);
    }
    CHK_RET(static_cast<HcclResult>(HcommEndpointStartListen(endpointHandle_, port, nullptr)));
    HCCL_INFO("[AicpuTsRoceHostLocChannel::%s] SUCCESS. port[%u].", __func__, port);
    return HCCL_SUCCESS;
}

HcclResult AicpuTsRoceHostLocChannel::BuildSocket()
{
    if (socket_ != nullptr) {
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[AicpuTsRoceHostLocChannel::%s] socket ptr is NULL, rebuild Socket", __func__);

    Hccl::LinkData linkData = BuildDefaultLinkData();
    CHK_RET(EndpointDescPairToLinkData(localEp_, remoteEp_, linkData));
    HCCL_INFO("[AicpuTsRoceHostLocChannel::%s] built linkData: %s", __func__, linkData.Describe().c_str());
    uint16_t port = channelDesc_.port;
    if (port == 0) {
        port = DEFAULT_LISTENING_PORT;
        HCCL_INFO("[AicpuTsRoceHostLocChannel::%s] channelDesc port is 0, use default port [%u]", __func__, port);
    }
    std::string socketTag
        = (channelDesc_.channelName != nullptr) ? std::string(channelDesc_.channelName) : "AUTOMATIC_SOCKET_TAG";
    Hccl::SocketConfig socketConfig
        = (channelDesc_.role != HCOMM_SOCKET_ROLE_RESERVED) ?
              Hccl::SocketConfig(linkData, port, socketTag, channelDesc_.role == HCOMM_SOCKET_ROLE_SERVER) :
              Hccl::SocketConfig(linkData, port, socketTag);
    if (channelDesc_.channelName != nullptr && channelDesc_.channelName[0] != '\0') {
        socketConfig.SetHccpTag(socketTag);
    }
    SaveSocketConfig(socketConfig);
    CHK_RET(SocketMgr::GetInstance(devicePhyId_).GetSocket(*socketConfig_, socket_));
    HCCL_INFO("[AicpuTsRoceHostLocChannel::%s] SUCCESS. port[%u].", __func__, port);
    return HCCL_SUCCESS;
}

HcclResult AicpuTsRoceHostLocChannel::BuildConnection()
{
    std::unique_ptr<RoceHostLocConnection> conn;
    EXCEPTION_CATCH(
        conn = std::make_unique<RoceHostLocConnection>(socket_, rdmaHandle_, channelDesc_.roceAttr.cqAttrFlags),
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
        "[AicpuTsRoceHostLocChannel::BuildConnection] QpInfo: serviceLevel[%u], trafficClass[%u], retryCnt[%u], "
        "retryInterval[%u].",
        qpInfo.serviceLevel, qpInfo.trafficClass, qpInfo.retryCnt, qpInfo.retryInterval);
    connections_.emplace_back(std::move(conn));
    connNum_ = connections_.size();
    HCCL_INFO("[AicpuTsRoceHostLocChannel::%s] connection num [%u]", __func__, connNum_);
    return HCCL_SUCCESS;
}

HcclResult AicpuTsRoceHostLocChannel::BuildNotify()
{
    CHK_PRT_RET(
        notifyNum_ != RDMA_NOTIFY_NUM,
        HCCL_ERROR(
            "[AicpuTsRoceHostLocChannel::%s] rdma notify num false, actual num [%u], expected num [%u]", __func__,
            notifyNum_, RDMA_NOTIFY_NUM),
        HCCL_E_PARA);

    localNotifies_.clear();
    bool devUsed = true;
    for (uint32_t i = 0; i < notifyNum_; ++i) {
        std::unique_ptr<Hccl::RdmaLocalNotify> notifyPtr = nullptr;
        EXCEPTION_CATCH(notifyPtr = std::make_unique<Hccl::RdmaLocalNotify>(rdmaHandle_, devUsed), return HCCL_E_PTR);
        localNotifies_.emplace_back(std::move(notifyPtr));
    }
    HCCL_INFO("[AicpuTsRoceHostLocChannel::%s] notify num [%u]", __func__, notifyNum_);
    return HCCL_SUCCESS;
}

HcclResult AicpuTsRoceHostLocChannel::BuildBuffer()
{
    if (channelDesc_.exchangeAllMems) {
        // Get memHandles from endpoint
        HCCL_INFO("[AicpuTsRoceHostLocChannel][%s] exchangeAllMems == True. Get memHandles from endpoint.", __func__);
        std::shared_ptr<Hccl::LocalRdmaRmaBuffer>* memHandles = nullptr;
        uint32_t memHandleNum = 0;
        CHK_RET(static_cast<HcclResult>(
            HcommMemGetAllMemHandles(endpointHandle_, ReinterpretAs<void**>(&memHandles), &memHandleNum)));
        HCCL_INFO("[AicpuTsRoceHostLocChannel][%s] Got memHandleNum[%u].", __func__, memHandleNum);
        for (uint32_t i = 0; i < memHandleNum; ++i) {
            std::shared_ptr<Hccl::LocalRdmaRmaBuffer>& localRdmaBuffer = memHandles[i];
            CHK_SMART_PTR_NULL(localRdmaBuffer);
            Hccl::Buffer* buf = localRdmaBuffer->GetBuf();
            CHK_PTR_NULL(buf);
            HCCL_INFO(
                "[AicpuTsRoceHostLocChannel][%s] Got memHandle No.%u: addr[0x%llx], size[0x%llx], memType[%d], "
                "memInfo[%s].",
                __func__, i, static_cast<unsigned long long>(localRdmaBuffer->GetAddr()),
                static_cast<unsigned long long>(localRdmaBuffer->GetSize()), static_cast<int>(buf->GetMemType()),
                buf->GetMemInfo().c_str());
            localRmaBuffers_.emplace_back(localRdmaBuffer.get());
        }
    } else {
        // 从 channelDesc 的 memHandle，获得 localRmaBuffers_
        HCCL_INFO(
            "[AicpuTsRoceHostLocChannel][%s] exchangeAllMems == false. Get memHandles from channelDesc.", __func__);
        CHK_PTR_NULL(channelDesc_.memHandles);
        for (uint32_t i = 0; i < channelDesc_.memHandleNum; ++i) {
            CHK_PTR_NULL(channelDesc_.memHandles[i]);
            auto* localRdmaBuffer = ReinterpretAs<Hccl::LocalRdmaRmaBuffer*>(channelDesc_.memHandles[i]);
            HCCL_INFO(
                "[AicpuTsRoceHostLocChannel][%s] Got memHandle No.%u: addr[0x%llx], size[0x%llx], memType[%d], "
                "memInfo[%s].",
                __func__, i, static_cast<unsigned long long>(localRdmaBuffer->GetAddr()),
                static_cast<unsigned long long>(localRdmaBuffer->GetSize()),
                static_cast<int>(localRdmaBuffer->GetBuf()->GetMemType()),
                localRdmaBuffer->GetBuf()->GetMemInfo().c_str());
            localRmaBuffers_.emplace_back(localRdmaBuffer);
        }
    }
    bufferNum_ = localRmaBuffers_.size();
    HCCL_INFO("[AicpuTsRoceHostLocChannel::%s] buffer num [%u]", __func__, bufferNum_);
    return HCCL_SUCCESS;
}

HcclResult AicpuTsRoceHostLocChannel::BuildNotifyValueBuffer()
{
    Hccl::DevCapability::GetInstance().Init(Hccl::HrtGetDeviceType());
    u32 notifysize = Hccl::DevCapability::GetInstance().GetNotifySize();
    EXCEPTION_CATCH((notifyValueMem_ = std::make_shared<Hccl::DevBuffer>(notifysize)), return HCCL_E_PTR);
    HCCL_DEBUG(
        "[AicpuTsRoceHostLocChannel::%s] create notify value buffer[%p], size[%u]", __func__, notifyValueMem_.get(),
        notifyValueMem_->GetSize());
    u64 notifyValue = 1; // notify值写1表示record
    Hccl::HrtMemcpy(
        ReinterpretAs<void*>(notifyValueMem_->GetAddr()), notifyValueMem_->GetSize(), &notifyValue, notifysize,
        Hccl::tagRtMemcpyKind::RT_MEMCPY_HOST_TO_DEVICE);
    EXCEPTION_CATCH(
        (notifyValueBuffer_ = std::make_unique<Hccl::LocalRdmaRmaBuffer>(notifyValueMem_, rdmaHandle_)),
        return HCCL_E_PTR);
    HCCL_INFO("[AicpuTsRoceHostLocChannel::%s] build notify value buffer success.", __func__);
    return HCCL_SUCCESS;
}

HcclResult AicpuTsRoceHostLocChannel::Init()
{
    s32 userDevId = Hccl::HrtGetDevice();
    devicePhyId_ = Hccl::HrtGetDevicePhyIdByUserDevId(static_cast<u32>(userDevId));

    CHK_RET(ParseInputParam());
    if (channelDesc_.exchangeAllMems && channelDesc_.role != HCOMM_SOCKET_ROLE_CLIENT) {
        CHK_RET(StartListen());
    }
    CHK_RET(BuildSocket());
    CHK_RET(BuildConnection());
    CHK_RET(BuildNotify());
    CHK_RET(BuildBuffer());
    CHK_RET(BuildNotifyValueBuffer());
    return HCCL_SUCCESS;
}

// 当前AICPU和框架没有改为返回错误码形式，所有暂时使用该方法转换
ChannelStatus AicpuTsRoceHostLocChannel::GetStatus()
{
    ChannelStatus status;
    HcclResult ret = GetStatus(status);
    if (ret != HCCL_SUCCESS && ret != HCCL_E_AGAIN) {
        HCCL_ERROR("[AicpuTsRoceHostLocChannel::GetStatus] get status exception occurred, HcclResult=[%d]", ret);
        return ChannelStatus::FAILED;
    }
    return status;
}

HcclResult AicpuTsRoceHostLocChannel::ProcessStatus()
{
    switch (channelStatus_) {
        case ChannelStatus::READY:
            return HCCL_SUCCESS;
        case ChannelStatus::SOCKET_TIMEOUT:
            HCCL_ERROR("[AicpuTsRoceHostLocChannel::ProcessStatus] get socket timeout");
            return HCCL_E_ROCE_CONNECT;
        default:
            return HCCL_E_AGAIN;
    }
}

HcclResult AicpuTsRoceHostLocChannel::GetStatus(ChannelStatus& status)
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

HcclResult AicpuTsRoceHostLocChannel::CheckSocketStatus()
{
    CHK_PTR_NULL(socket_);
    Hccl::SocketStatus socketStatus = socket_->GetStatus(); // socket状态机
    HCCL_DEBUG("[AicpuTsRoceHostLocChannel::CheckSocketStatus] socket status = %s", socketStatus.Describe().c_str());
    if (socketStatus == Hccl::SocketStatus::OK) {
        rdmaStatus_ = RdmaStatus::SOCKET_OK;
        channelStatus_ = ChannelStatus::SOCKET_OK;
    } else if (socketStatus == Hccl::SocketStatus::TIMEOUT) {
        channelStatus_ = ChannelStatus::SOCKET_TIMEOUT;
    }
    return HCCL_SUCCESS;
}

// 准备资源（创建QP）
HcclResult AicpuTsRoceHostLocChannel::CreateQp()
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
    HCCL_INFO("[AicpuTsRoceHostLocChannel::%s] all connections resources connected.", __func__);
    return HCCL_SUCCESS;
}

// 交换数据
HcclResult AicpuTsRoceHostLocChannel::ExchangeData()
{
    HCCL_INFO(
        "[AicpuTsRoceHostLocChannel::%s] Start to SendExchangeData, notifyNum=%u, bufferNum=%u, connNum=%u", __func__,
        notifyNum_, bufferNum_, connNum_);

    // 同步数据打包
    Hccl::BinaryStream binaryStream;
    NotifyVecPack(binaryStream);
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
        !socket_->Send(static_cast<void*>(&sendSize), sizeof(sendSize)),
        HCCL_ERROR("[AicpuTsRoceHostLocChannel::%s] Send sendSize failed", __func__), HCCL_E_NETWORK);
    HCCL_INFO(
        "[AicpuTsRoceHostLocChannel::%s] Send size[%llu] of data success. [%llu] bytes sent.", __func__, sendSize,
        sizeof(sendSize));

    // 同步接收数据包尺寸
    CHK_PRT_RET(
        !socket_->Recv(static_cast<void*>(&recvSize), sizeof(recvSize)),
        HCCL_ERROR("[AicpuTsRoceHostLocChannel::%s] Recv recvSize failed", __func__), HCCL_E_NETWORK);
    HCCL_INFO(
        "[AicpuTsRoceHostLocChannel::%s] Receive size[%llu] of data success. [%llu] bytes received.", __func__,
        recvSize, sizeof(recvSize));

    // 对端声明的数据长度不可信，为0或超过上限时拒绝分配，防止超大内存申请导致OOM
    CHK_PRT_RET(
        recvSize == 0 || recvSize > MAX_EXCHANGE_DATA_SIZE,
        HCCL_ERROR(
            "[AicpuTsRoceHostLocChannel::%s] recvSize[%llu] is invalid, limit[%llu]", __func__, recvSize,
            MAX_EXCHANGE_DATA_SIZE),
        HCCL_E_PARA);

    // 同步发送数据
    CHK_PRT_RET(
        !socket_->Send(static_cast<void*>(sendData.data()), sendSize),
        HCCL_ERROR("[AicpuTsRoceHostLocChannel::%s] Send exchange data failed", __func__), HCCL_E_NETWORK);
    HCCL_INFO("[AicpuTsRoceHostLocChannel::%s] Send Exchange Data success. [%llu] bytes sent.", __func__, sendSize);

    // 同步接收数据
    HCCL_INFO("[AicpuTsRoceHostLocChannel::%s] Start to Receive Exchange Data", __func__);
    recvData.resize(recvSize);
    CHK_PRT_RET(
        !socket_->Recv(static_cast<void*>(recvData.data()), recvSize),
        HCCL_ERROR("[AicpuTsRoceHostLocChannel::%s] Recv exchange data failed", __func__), HCCL_E_NETWORK);
    HCCL_INFO(
        "[AicpuTsRoceHostLocChannel::%s] Receive Exchange Data success. [%llu] bytes received.", __func__, recvSize);
    EXCEPTION_HANDLE_END

    // 同步数据解包
    Hccl::BinaryStream recvBinStream(recvData);
    CHK_RET(NotifyVecUnpack(recvBinStream));
    CHK_RET(RmtBufferVecUnpackProc(recvBinStream));
    CHK_RET(ConnVecUnpackProc(recvBinStream));

    HCCL_INFO("[AicpuTsRoceHostLocChannel::%s] Unpack exchange Data success. ", __func__);
    return HCCL_SUCCESS;
}

void AicpuTsRoceHostLocChannel::NotifyVecPack(Hccl::BinaryStream& binaryStream)
{
    binaryStream << notifyNum_;
    HCCL_INFO("start pack notifyVec");
    u32 pos = 0;
    for (auto& it : localNotifies_) {
        binaryStream << pos;
        std::unique_ptr<Hccl::Serializable> dto = it->GetExchangeDto();
        dto->Serialize(binaryStream);
        HCCL_INFO("pack notify pos=%u, dto %s", pos, dto->Describe().c_str());
        pos++;
    }
}

HcclResult AicpuTsRoceHostLocChannel::BufferVecPack(Hccl::BinaryStream& binaryStream)
{
    binaryStream << bufferNum_;
    HCCL_INFO("[AicpuTsRoceHostLocChannel::%s] start to pack RmaBuffers", __func__);
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
    HCCL_INFO("[AicpuTsRoceHostLocChannel::%s] pack RmaBuffers finish", __func__);
    return HCCL_SUCCESS;
}

HcclResult AicpuTsRoceHostLocChannel::ConnVecPack(Hccl::BinaryStream& binaryStream)
{
    binaryStream << connNum_;
    HCCL_INFO("[AicpuTsRoceHostLocChannel::%s] start to pack connections", __func__);
    u32 pos = 0;
    for (auto& it : connections_) {
        binaryStream << pos;
        std::unique_ptr<Hccl::Serializable> dto = nullptr;
        CHK_RET(it->GetExchangeDto(dto));
        dto->Serialize(binaryStream);
        HCCL_INFO("pack connection pos=%u, dto %s", pos, dto->Describe().c_str());
        pos++;
    }
    HCCL_INFO("[AicpuTsRoceHostLocChannel::%s] pack connections finish", __func__);
    return HCCL_SUCCESS;
}

HcclResult AicpuTsRoceHostLocChannel::RmtBufferVecUnpackProc(Hccl::BinaryStream& binaryStream)
{
    u32 rmtNum;
    binaryStream >> rmtNum;

    HCCL_INFO("[AicpuTsRoceHostLocChannel::%s] bufferNum_=%u, rmtNum=%u", __func__, bufferNum_, rmtNum);

    // 对端声明的buffer数量不可信，超过上限时拒绝，防止直接resize导致超大内存申请
    CHK_PRT_RET(
        rmtNum > MAX_BUFFER_NUM,
        HCCL_ERROR("[AicpuTsRoceHostLocChannel::%s] rmtNum[%u] exceeds maxNum[%u]", __func__, rmtNum, MAX_BUFFER_NUM),
        HCCL_E_PARA);

    rmtRmaBuffers_.resize(rmtNum);
    for (u32 i = 0; i < rmtNum; i++) {
        u32 pos;
        binaryStream >> pos;
        if (pos >= rmtNum) {
            HCCL_ERROR("[AicpuTsRoceHostLocChannel::%s] pos=%u out of range (rmtNum=%u)", __func__, pos, rmtNum);
            return HCCL_E_INTERNAL;
        }
        Hccl::ExchangeRdmaBufferDto dto;
        dto.Deserialize(binaryStream);

        HCCL_INFO("[AicpuTsRoceHostLocChannel::%s] pos=%u, dto %s", __func__, pos, dto.Describe().c_str());
        EXCEPTION_CATCH(
            rmtRmaBuffers_[pos] = std::make_unique<Hccl::RemoteRdmaRmaBuffer>(rdmaHandle_, dto),
            HCCL_ERROR(
                "[AicpuTsRoceHostLocChannel::%s] make_unique<Hccl::RemoteRdmaRmaBuffer> throws an exception!",
                __func__);
            return HCCL_E_INTERNAL);
        HCCL_INFO(
            "[AicpuTsRoceHostLocChannel::%s] pos=%u, rmtRmaBuffer=%s", __func__, pos,
            rmtRmaBuffers_[pos]->Describe().c_str());
    }

    return HCCL_SUCCESS;
}

HcclResult AicpuTsRoceHostLocChannel::NotifyVecUnpack(Hccl::BinaryStream& binaryStream)
{
    uint32_t notifySize = 0;
    binaryStream >> notifySize;
    if (notifySize != notifyNum_) {
        HCCL_ERROR(
            "[AicpuTsRoceHostLocChannel::NotifyVecUnpack] rmtNum=%u is not equal to localNum=%u", notifySize,
            notifyNum_);
        return HCCL_E_ROCE_CONNECT;
    }
    remoteNotifies_.clear();
    u32 pos = 0;
    for (pos = 0; pos < notifySize; pos++) {
        binaryStream >> pos;
        Hccl::ExchangeRdmaBufferDto dto;
        dto.Deserialize(binaryStream);
        HCCL_INFO("unpack pos=%u, dto %s", pos, dto.Describe().c_str());
        remoteNotifies_.push_back(std::make_unique<Hccl::RemoteRdmaRmaBuffer>(rdmaHandle_, dto));
        HCCL_INFO("unpack notify pos=%u, rmtRmaBuffer=%s", pos, remoteNotifies_.back()->Describe().c_str());
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuTsRoceHostLocChannel::ConnVecUnpackProc(Hccl::BinaryStream& binaryStream)
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

HcclResult AicpuTsRoceHostLocChannel::ModifyQp()
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
    HCCL_INFO("[AicpuTsRoceHostLocChannel::%s] all connections resources modify success.", __func__);
    return HCCL_SUCCESS;
}

HcclResult AicpuTsRoceHostLocChannel::GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos)
{
    std::lock_guard<std::mutex> lock(remoteMemsMutex_);
    Hccl::RemoteMemCtx<std::unique_ptr<Hccl::RemoteRdmaRmaBuffer>> remoteMemCtx{
        cacheValid_, rmtRmaBuffers_, remoteUserMems_, memInfoCopies_, memInfoPointers_, remoteMem, memInfos, memNum};
    CHK_RET(Hccl::GetRemoteUserMems(remoteMemCtx));
    return HCCL_SUCCESS;
}

HcclResult AicpuTsRoceHostLocChannel::Clean() { return HCCL_SUCCESS; }

HcclResult AicpuTsRoceHostLocChannel::Resume() { return HCCL_SUCCESS; }

std::string AicpuTsRoceHostLocChannel::Describe() const
{
    std::string msg = "AicpuTsRoceHostLocChannel{";
    msg += Hccl::StringFormat("notifyNum:%u, localNotifies:[", notifyNum_);
    for (auto& notify : localNotifies_) {
        msg += notify->Describe();
        msg += ", ";
    }
    msg += "]";
    msg += Hccl::StringFormat(", bufferNum:%u, localRmaBuffers:[", bufferNum_);
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
    msg += Hccl::StringFormat(", rdmaHandle:%p, %s, ", rdmaHandle_, channelStatus_.Describe().c_str());
    if (socket_ != nullptr) {
        msg += socket_->Describe();
    }
    msg += "}";
    return msg;
}

std::vector<char> AicpuTsRoceHostLocChannel::GetLocalNotifyUniqueIds() const
{
    HCCL_DEBUG("start packing local notify uniqueIds");
    std::vector<char> result(0);
    for (auto& it : localNotifies_) {
        HCCL_INFO("AicpuTsRoceHostLocChannel local notify %s", it->Describe().c_str());
        auto uniqueId = it->GetUniqueId();
        result.insert(result.end(), uniqueId.begin(), uniqueId.end());
    }
    return result;
}

std::vector<char> AicpuTsRoceHostLocChannel::GetRemoteNotifyUniqueIds() const
{
    HCCL_DEBUG("start packing remote notify uniqueIds");
    std::vector<char> result(0);
    Hccl::BinaryStream binaryStream;
    for (auto& it : remoteNotifies_) {
        std::vector<char> uniqueId;
        uniqueId = GetSingleRmaBufferUniqueId(static_cast<uint64_t>(it->GetAddr()), it->GetSize(), it->GetRkey());
        HCCL_INFO("AicpuTsRoceHostLocChannel remote notify %s", it->Describe().c_str());
        result.insert(result.end(), uniqueId.begin(), uniqueId.end());
    }
    binaryStream.Dump(result);
    return result;
}

std::vector<char> AicpuTsRoceHostLocChannel::GetNotifyValueBufferUniqueIds() const
{
    HCCL_DEBUG("start packing notify value buffer uniqueIds");
    std::vector<char> uniqueId;
    uniqueId = GetSingleRmaBufferUniqueId(
        static_cast<uint64_t>(notifyValueBuffer_->GetAddr()), notifyValueBuffer_->GetSize(),
        notifyValueBuffer_->GetLkey());
    HCCL_INFO("AicpuTsRoceHostLocChannel notify value buffer %s", notifyValueBuffer_->Describe().c_str());
    return uniqueId;
}

std::vector<char> AicpuTsRoceHostLocChannel::GetSingleRmaBufferUniqueId(u64 addr, u64 size, u32 key) const
{
    Hccl::BinaryStream binaryStream;
    binaryStream << addr;
    binaryStream << size;
    binaryStream << key;
    std::vector<char> result;
    binaryStream.Dump(result);
    return result;
}

std::vector<char> AicpuTsRoceHostLocChannel::GetRmtBufferUniqueIds() const
{
    HCCL_DEBUG("start packing remote buffer uniqueIds");
    std::vector<char> result(0);
    for (auto& it : rmtRmaBuffers_) {
        std::vector<char> uniqueId;
        if (it != nullptr) {
            uniqueId = GetSingleRmaBufferUniqueId(static_cast<uint64_t>(it->GetAddr()), it->GetSize(), it->GetRkey());
            HCCL_INFO("AicpuTsRoceHostLocChannel::GetRmtBufferUniqueIds, %s", it->Describe().c_str());
        } else {
            uniqueId = GetSingleRmaBufferUniqueId(0, 0, 0); // 填充一个空的buffer
            HCCL_INFO("AicpuTsRoceHostLocChannel::GetRmtBufferUniqueIds, null buffer");
        }
        result.insert(result.end(), uniqueId.begin(), uniqueId.end());
    }
    return result;
}

std::vector<char> AicpuTsRoceHostLocChannel::GetLocBufferUniqueIds() const
{
    HCCL_DEBUG("start packing local buffer uniqueIds");
    std::vector<char> result(0);
    for (auto& it : localRmaBuffers_) {
        std::vector<char> uniqueId;
        if (it != nullptr) {
            uniqueId = GetSingleRmaBufferUniqueId(static_cast<uint64_t>(it->GetAddr()), it->GetSize(), it->GetLkey());
            HCCL_INFO("AicpuTsRoceHostLocChannel::GetLocBufferUniqueIds, %s", it->Describe().c_str());
        } else {
            uniqueId = GetSingleRmaBufferUniqueId(0, 0, 0); // 填充一个空的buffer
            HCCL_INFO("AicpuTsRoceHostLocChannel::GetLocBufferUniqueIds, null buffer");
        }
        result.insert(result.end(), uniqueId.begin(), uniqueId.end());
    }
    return result;
}

std::vector<char> AicpuTsRoceHostLocChannel::GetConnUniqueIds() const
{
    HCCL_DEBUG("start packing all conn uniqueIds");
    std::vector<char> result(0);
    for (auto& it : connections_) {
        HCCL_INFO("AicpuTsRoceHostLocChannel %s", it->Describe().c_str());
        auto uniqueId = it->GetUniqueId();
        result.insert(result.end(), uniqueId.begin(), uniqueId.end());
    }
    return result;
}

std::vector<char> AicpuTsRoceHostLocChannel::GetUniqueId() const
{
    if (channelStatus_ != ChannelStatus::READY) {
        HCCL_ERROR(
            "[AicpuTsRoceHostLocChannel::%s] channel status[%d] is not ready[%d], please check.", __func__,
            channelStatus_, ChannelStatus::READY);
    }
    u32 type = static_cast<u32>(Hccl::TransportType::ROCE);
    Hccl::BinaryStream binaryStream;
    binaryStream << type;
    binaryStream << notifyNum_;
    binaryStream << bufferNum_;
    binaryStream << connNum_;

    auto locNotifyUniqueIds = GetLocalNotifyUniqueIds();
    binaryStream << locNotifyUniqueIds;

    auto rmtNotifyUniqueIds = GetRemoteNotifyUniqueIds();
    binaryStream << rmtNotifyUniqueIds;

    auto notifyValueBufferUniqueIds = GetNotifyValueBufferUniqueIds();
    binaryStream << notifyValueBufferUniqueIds;

    auto locBufferUniqueIds = GetLocBufferUniqueIds();
    binaryStream << locBufferUniqueIds;

    auto rmtBufferUniqueIds = GetRmtBufferUniqueIds();
    binaryStream << rmtBufferUniqueIds;

    auto connUniqueIds = GetConnUniqueIds();
    binaryStream << connUniqueIds;

    std::vector<char> result;
    binaryStream.Dump(result);
    return result;
}

static HcclResult SetModuleDataName(Hccl::ModuleData& module, const std::string& name)
{
    int ret = strcpy_s(module.name, sizeof(module.name), name.c_str());
    if (ret != 0) {
        HCCL_ERROR("[SetModuleDataName] strcpy_s name %s failed", name.c_str());
        return HCCL_E_INTERNAL;
    }

    return HCCL_SUCCESS;
}

HcclResult AicpuTsRoceHostLocChannel::PackOpData(std::vector<char>& data) const
{
    std::vector<Hccl::ModuleData> dataVec;
    dataVec.resize(Hccl::AicpuResMgrType::__COUNT__);

    Hccl::AicpuResMgrType resType = Hccl::AicpuResMgrType::STREAM;
    CHK_RET(SetModuleDataName(dataVec[resType], "AicpuTsRoceHostLocChannel"));

    std::vector<char> result;
    Hccl::BinaryStream binaryStream;
    binaryStream << GetUniqueId();

    binaryStream.Dump(result);

    dataVec[resType].data = result;

    Hccl::AicpuResPackageHelper helper;
    data = helper.GetPackedData(dataVec);

    return HCCL_SUCCESS;
}

HcclResult AicpuTsRoceHostLocChannel::H2DResPack(std::vector<char>& buffer) const
{
    CHK_RET(PackOpData(buffer));
    HCCL_INFO(
        "[AicpuTsRoceHostLocChannel][%s] Pack Buffer data[%p], Pack Buffer size[%zu].", __func__, buffer.data(),
        buffer.size());
    return HCCL_SUCCESS;
}

HcclResult AicpuTsRoceHostLocChannel::GetNotifyNum(uint32_t* notifyNum) const
{
    CHK_PTR_NULL(notifyNum);
    *notifyNum = notifyNum_;
    return HCCL_SUCCESS;
}

HcclResult AicpuTsRoceHostLocChannel::GetBufferNum(uint32_t* bufferNum) const
{
    CHK_PTR_NULL(bufferNum);
    *bufferNum = bufferNum_;
    return HCCL_SUCCESS;
}

HcclResult AicpuTsRoceHostLocChannel::GetQpNum(uint32_t* qpNum) const
{
    CHK_PTR_NULL(qpNum);
    *qpNum = connNum_;
    return HCCL_SUCCESS;
}

HcclResult AicpuTsRoceHostLocChannel::Serialize(std::shared_ptr<hccl::DeviceMem>& out)
{
    out.reset();
    CHK_PRT_RET(
        channelStatus_ != ChannelStatus::READY,
        HCCL_ERROR("[AicpuTsRoceHostLocChannel][%s] channel not ready, status[%d]", __func__, channelStatus_),
        HCCL_E_INTERNAL);

    std::vector<char> hostBuffer;
    CHK_RET(H2DResPack(hostBuffer));

    u64 totalBytes = static_cast<u64>(hostBuffer.size());
    CHK_PRT_RET(
        totalBytes == 0, HCCL_ERROR("[AicpuTsRoceHostLocChannel][%s] serialized buffer is empty", __func__),
        HCCL_E_INTERNAL);

    hccl::DeviceMem devMem;
    EXCEPTION_CATCH(devMem = hccl::DeviceMem::alloc(totalBytes), return HCCL_E_PTR);

    Hccl::HrtMemcpy(
        devMem.ptr(), totalBytes, hostBuffer.data(), totalBytes, Hccl::tagRtMemcpyKind::RT_MEMCPY_HOST_TO_DEVICE);

    out = std::make_shared<hccl::DeviceMem>(std::move(devMem));

    HCCL_INFO("[AicpuTsRoceHostLocChannel][%s] serialize success, size[%llu]", __func__, totalBytes);
    return HCCL_SUCCESS;
}

HcommChannelKind AicpuTsRoceHostLocChannel::GetChannelKind() const { return HcommChannelKind::AICPU_TS_ROCE_HOST_LOC; }
} // namespace hcomm
