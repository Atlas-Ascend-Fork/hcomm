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
#include "aicpu_ts_uboe_ub_rtp_channel_helper.h"
#include "endpoint.h"
#include "../../sockets/socket_mgr.h"
#include "orion_adpt_utils.h"
#include "env_config/env_config_v2.h"

// Orion
#include "virtual_topo.h"
#include "dev_capability.h"
#include "dev_buffer.h"
#include "aicpu_res_package_helper.h"
#include "exchange_ub_buffer_dto.h"
#include "exchange_ub_conn_dto.h"
#include "user_remote_mem_getter.h"
#include "makebufs_helper.h"

namespace hcomm {

constexpr u32 SERVER_LISTEN_PORT = 60001;

AicpuTsUboeUbRtpChannelHelper::AicpuTsUboeUbRtpChannelHelper(
    EndpointHandle endpointHandle, const HcommChannelDesc& channelDesc)
    : endpointHandle_(endpointHandle),
      channelDesc_(channelDesc)
{}

AicpuTsUboeUbRtpChannelHelper::~AicpuTsUboeUbRtpChannelHelper()
{
    if (channelDesc_.socket == nullptr && socket_ != nullptr && socketConfig_ != nullptr) {
        SocketMgr::GetInstance(devicePhyId_).DestroySocket(*socketConfig_);
        socket_ = nullptr;
    }
}

HcclResult AicpuTsUboeUbRtpChannelHelper::ParseInputParam()
{
    // 1. 从 endpointHandle_，获得 localEp_ 和 rdmaHandle_
    Endpoint* localEpPtr = static_cast<Endpoint*>(endpointHandle_);
    CHK_PTR_NULL(localEpPtr);
    localEp_ = localEpPtr->GetEndpointDesc();
    rdmaHandle_ = localEpPtr->GetRdmaHandle();

    HCCL_INFO("[AicpuTsUboeUbRtpChannelHelper][%s] localProtocol[%d]", __func__, localEp_.protocol);

    // 2. 从 channelDesc_，获得 remoteEp_, socket_ 和 notifyNum
    remoteEp_ = channelDesc_.remoteEndpoint;
    socket_ = static_cast<Hccl::Socket*>(channelDesc_.socket);
    notifyNum_ = channelDesc_.notifyNum;
    commonRes_.bufferVec.clear();

    if (channelDesc_.exchangeAllMems) {
        // 3. Get memHandles from endpoint
        HCCL_INFO(
            "[AicpuTsUboeUbRtpChannelHelper][%s] exchangeAllMems == True. Get memHandles from endpoint.", __func__);
        std::shared_ptr<Hccl::LocalUbRmaBuffer>* memHandles = nullptr;
        uint32_t memHandleNum = 0;
        CHK_RET(static_cast<HcclResult>(
            HcommMemGetAllMemHandles(endpointHandle_, ReinterpretAs<void**>(&memHandles), &memHandleNum)));
        HCCL_INFO("[AicpuTsUboeUbRtpChannelHelper][%s] Got memHandleNum[%u].", __func__, memHandleNum);
        for (uint32_t i = 0; i < memHandleNum; ++i) {
            std::shared_ptr<Hccl::LocalUbRmaBuffer>& localUbRmaBuffer = memHandles[i];
            CHK_SMART_PTR_NULL(localUbRmaBuffer);
            Hccl::Buffer* buf = localUbRmaBuffer->GetBuf();
            CHK_PTR_NULL(buf);
            HCCL_INFO(
                "[AicpuTsUboeUbRtpChannelHelper][%s] Got memHandle No.%u: addr[0x%llx], size[0x%llx], "
                "memType[%d], memInfo[%s].",
                __func__, i, static_cast<unsigned long long>(localUbRmaBuffer->GetAddr()),
                static_cast<unsigned long long>(localUbRmaBuffer->GetSize()), static_cast<int>(buf->GetMemType()),
                buf->GetMemInfo().c_str());
            commonRes_.bufferVec.push_back(localUbRmaBuffer.get());
        }
    } else {
        // 3. 从 channelDesc 的 memHandle 填充 commonRes_.bufferVec
        HCCL_INFO(
            "[AicpuTsUboeUbRtpChannelHelper][%s] exchangeAllMems == false. Get memHandles from channelDesc.", __func__);
        CHK_RET(MakeRmaBufferVecFromMemHandles(
            channelDesc_.memHandles, channelDesc_.memHandleNum, commonRes_.bufferVec, "AicpuTsUboeUbRtpChannelHelper"));
    }

    return HCCL_SUCCESS;
}

void AicpuTsUboeUbRtpChannelHelper::BuildConn()
{
    if (BuildConnection() != HCCL_SUCCESS) {
        HCCL_ERROR("[AicpuTsUboeUbRtpChannelHelper::%s] BuildConnection failed", __func__);
    }
}

HcclResult AicpuTsUboeUbRtpChannelHelper::BuildNotify()
{
    localNotifies_.clear();
    commonRes_.notifyVec.clear();
    bool devUsed = true;
    for (uint32_t i = 0; i < notifyNum_; ++i) {
        std::unique_ptr<Hccl::UbLocalNotify> notifyPtr = nullptr;
        EXCEPTION_CATCH(notifyPtr = std::make_unique<Hccl::UbLocalNotify>(rdmaHandle_, devUsed), return HCCL_E_PTR);
        commonRes_.notifyVec.push_back(notifyPtr.get());
        localNotifies_.push_back(std::move(notifyPtr));
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUboeUbRtpChannelHelper::BuildDrainResource()
{
    // 申请创建channel drain阻塞等待的相关资源
    // 申请notify作为read的落点
    bool devUsed = true;
    EXCEPTION_CATCH(drainNotify_ = std::make_unique<Hccl::UbLocalNotify>(rdmaHandle_, devUsed), return HCCL_E_PTR);
    HCCL_INFO(
        "[AicpuTsUboeUbRtpChannelHelper][%s] drainNotify created: %s", __func__, drainNotify_->Describe().c_str());

    // 常量1内存供远端读取
    u32 notifySize = Hccl::DevCapability::GetInstance().GetNotifySize();

    std::shared_ptr<Hccl::DevBuffer> constMem;
    EXCEPTION_CATCH(constMem = std::make_shared<Hccl::DevBuffer>(notifySize), return HCCL_E_PTR);

    Hccl::HrtMemcpy(
        ReinterpretAs<void*>(constMem->GetAddr()), constMem->GetSize(), &NORMAL_NOTIFY_VAL, sizeof(NORMAL_NOTIFY_VAL),
        Hccl::tagRtMemcpyKind::RT_MEMCPY_HOST_TO_DEVICE);

    EXCEPTION_CATCH(drainBuffer_ = std::make_unique<Hccl::LocalUbRmaBuffer>(constMem, rdmaHandle_), return HCCL_E_PTR);
    HCCL_INFO(
        "[AicpuTsUboeUbRtpChannelHelper][%s] drain buffer created: addr[0x%llx], size[%zu]", __func__,
        static_cast<unsigned long long>(drainBuffer_->GetAddr()), drainBuffer_->GetSize());

    return HCCL_SUCCESS;
}

HcclResult AicpuTsUboeUbRtpChannelHelper::BuildSocket()
{
    if (socket_ != nullptr) {
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[AicpuTsUboeUbRtpChannelHelper][%s] socket ptr is NULL, rebuildSocket", __func__);

    Hccl::IpAddress ipaddr{};
    CHK_RET(CommAddrToIpAddress(localEp_.commAddr, ipaddr));
    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::UB);
    Hccl::PortData localPort = Hccl::PortData(static_cast<Hccl::RankId>(localEp_.loc.device.devPhyId), type, 0, ipaddr);
    Hccl::SocketHandle socketHandle
        = Hccl::SocketHandleManager::GetInstance().Create(localEp_.loc.device.devPhyId, localPort);
    EXCEPTION_CATCH(
        serverSocket_ = std::make_unique<Hccl::Socket>(
            socketHandle, ipaddr, SERVER_LISTEN_PORT, ipaddr, "server", Hccl::SocketRole::SERVER,
            Hccl::NicType::DEVICE_NIC_TYPE),
        return HCCL_E_PARA);
    HCCL_INFO(
        "[AicpuTsUboeUbRtpChannelHelper][%s] listen_socket_info[%s]", __func__, serverSocket_->Describe().c_str());
    EXCEPTION_CATCH(serverSocket_->Listen(), return HCCL_E_INTERNAL);

    Hccl::LinkData linkData = BuildDefaultLinkData();
    CHK_RET(EndpointDescPairToLinkData(localEp_, remoteEp_, linkData));
    HCCL_INFO("[AicpuTsUboeUbRtpChannelHelper][%s] built linkData: %s", __func__, linkData.Describe().c_str());
    std::string socketTag
        = (channelDesc_.channelName != nullptr) ? std::string(channelDesc_.channelName) : "AUTOMATIC_SOCKET_TAG";
    bool noRankId = true;
    Hccl::SocketConfig socketConfig = Hccl::SocketConfig(linkData, socketTag, noRankId);
    if (channelDesc_.channelName != nullptr && channelDesc_.channelName[0] != '\0') {
        socketConfig.SetHccpTag(socketTag);
    }
    SaveSocketConfig(socketConfig);
    CHK_RET(SocketMgr::GetInstance(devicePhyId_).GetSocket(*socketConfig_, socket_));
    isRecvFirst_ = socket_->GetRole() == Hccl::SocketRole::CLIENT ? true : false;

    return HCCL_SUCCESS;
}

HcclResult AicpuTsUboeUbRtpChannelHelper::GetNotifyNum(uint32_t* notifyNum) const
{
    *notifyNum = this->notifyNum_;
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUboeUbRtpChannelHelper::GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos)
{
    std::lock_guard<std::mutex> lock(remoteMemsMutex_);
    Hccl::RemoteMemCtx<std::unique_ptr<Hccl::RemoteUbRmaBuffer>> remoteMemCtx{
        cacheValid_, rmtBufferVec_, remoteUserMems_, memInfoCopies_, memInfoPointers_, remoteMem, memInfos, memNum};
    CHK_RET(GetRemoteUserMems(remoteMemCtx));
    return HCCL_SUCCESS;
}

bool AicpuTsUboeUbRtpChannelHelper::IsSocketReady()
{
    if (socket_ == nullptr) {
        HCCL_ERROR("[%s] socket is nullptr, please check", __func__);
        channelStatus = ChannelStatus::INVALID;
        return false;
    }

    Hccl::SocketStatus socketStatus = socket_->GetAsyncStatus();
    if (socketStatus == Hccl::SocketStatus::OK) {
        channelStatus = ChannelStatus::SOCKET_OK;
        return true;
    } else if (socketStatus == Hccl::SocketStatus::TIMEOUT) {
        channelStatus = ChannelStatus::SOCKET_TIMEOUT;
        return false;
    }

    return false;
}

bool AicpuTsUboeUbRtpChannelHelper::IsResReady()
{
    for (auto& it : commonRes_.connVec) {
        if (it == nullptr) {
            Hccl::THROW<Hccl::InternalException>(
                "[AicpuTsUboeUbRtpChannelHelper::%s] failed, connection pointer is nullptr", __func__);
        }
        Hccl::RmaConnType connType = it->GetRmaConnType();
        if (connType != Hccl::RmaConnType::UB) {
            Hccl::THROW<Hccl::InternalException>(
                "[AicpuTsUboeUbRtpChannelHelper::%s] connection type[%s] is not ub", __func__,
                connType.Describe().c_str());
        }

        auto status = it->GetStatus();
        if (status != Hccl::RmaConnStatus::EXCHANGEABLE && status != Hccl::RmaConnStatus::READY) {
            return false;
        }
    }
    HCCL_INFO("[AicpuTsUboeUbRtpChannelHelper::%s] all resources ready.", __func__);
    return true;
}

bool AicpuTsUboeUbRtpChannelHelper::IsConnsReady()
{
    for (u32 i = 0; i < connNum_; i++) {
        if (commonRes_.connVec[i]->GetStatus() != Hccl::RmaConnStatus::READY) {
            return false;
        }
    }
    HCCL_INFO("[AicpuTsUboeUbRtpChannelHelper::%s] conns are ready.", __func__);
    return true;
}

void AicpuTsUboeUbRtpChannelHelper::NotifyVecPack(Hccl::BinaryStream& binaryStream)
{
    binaryStream << notifyNum_;
    u32 pos = 0;
    HCCL_INFO("[AicpuTsUboeUbRtpChannelHelper::%s] pack notify size[%d]", __func__, commonRes_.notifyVec.size());
    for (auto& it : commonRes_.notifyVec) {
        binaryStream << pos;
        std::unique_ptr<Hccl::Serializable> dto = it->GetExchangeDto();
        dto->Serialize(binaryStream);
        HCCL_INFO(
            "[AicpuTsUboeUbRtpChannelHelper::%s] pack notify pos=%u, dto %s", __func__, pos, dto->Describe().c_str());
        pos++;
    }
}

void AicpuTsUboeUbRtpChannelHelper::BufferVecPack(
    Hccl::BinaryStream& binaryStream, std::vector<Hccl::LocalRmaBuffer*>& bufferVec) const
{
    binaryStream << static_cast<u32>(bufferVec.size());
    u32 pos = 0;
    HCCL_INFO("[AicpuTsUboeUbRtpChannelHelper::%s] pack buffer size[%d]", __func__, bufferVec.size());
    for (auto& it : bufferVec) {
        binaryStream << pos;
        if (it != nullptr) {
            std::unique_ptr<Hccl::Serializable> dto = it->GetExchangeDto();
            dto->Serialize(binaryStream);
            HCCL_INFO(
                "[AicpuTsUboeUbRtpChannelHelper::%s] pack buffer pos=%u dto %s", __func__, pos,
                dto->Describe().c_str());
        } else {
            Hccl::ExchangeUbBufferDto exchangeDto;
            exchangeDto.Serialize(binaryStream);
            HCCL_INFO(
                "[AicpuTsUboeUbRtpChannelHelper::%s] pack buffer pos=%u, dto is null %s", __func__, pos,
                exchangeDto.Describe().c_str());
        }
        pos++;
    }
}

void AicpuTsUboeUbRtpChannelHelper::DrainBufferPack(Hccl::BinaryStream& binaryStream)
{
    // 只需交换常量buffer信息供对端读
    HCCL_INFO("start pack drain buffer");
    if (drainBuffer_ != nullptr) { // 非空的buffer，从buffer中获取 dto
        std::unique_ptr<Hccl::Serializable> dto = drainBuffer_->GetExchangeDto();
        dto->Serialize(binaryStream);
        HCCL_INFO("[AicpuTsUboeUbRtpChannelHelper::%s] pack drain buffer dto %s", __func__, dto->Describe().c_str());
    } else { // 空的buffer，dto所有字段为0(size=0)
        Hccl::ExchangeUbBufferDto exchangeDto;
        exchangeDto.Serialize(binaryStream);
        HCCL_INFO(
            "[AicpuTsUboeUbRtpChannelHelper::%s] pack drain buffer dto is null %s", __func__,
            exchangeDto.Describe().c_str());
    }
}

void AicpuTsUboeUbRtpChannelHelper::ConnVecPack(Hccl::BinaryStream& binaryStream)
{
    binaryStream << connNum_;
    u32 pos = 0;
    HCCL_INFO("[AicpuTsUboeUbRtpChannelHelper::%s] pack conn size[%d]", __func__, commonRes_.connVec.size());
    for (auto& it : commonRes_.connVec) {
        binaryStream << pos;
        std::unique_ptr<Hccl::Serializable> dto = it->GetExchangeDto();
        dto->Serialize(binaryStream);
        HCCL_INFO(
            "[AicpuTsUboeUbRtpChannelHelper::%s] pack connection pos=%u, dto %s", __func__, pos,
            dto->Describe().c_str());
        pos++;
    }
}

void AicpuTsUboeUbRtpChannelHelper::SendDataSize()
{
    sendData_.clear();
    bufferNum_ = commonRes_.bufferVec.size();
    connNum_ = commonRes_.connVec.size();

    HCCL_INFO("notifyNum=%u, bufferNum=%u, connNum=%u", notifyNum_, bufferNum_, connNum_);

    Hccl::BinaryStream binaryStream;
    NotifyVecPack(binaryStream);
    BufferVecPack(binaryStream, commonRes_.bufferVec);
    DrainBufferPack(binaryStream);
    ConnVecPack(binaryStream);

    binaryStream.Dump(sendData_);
    u32 sendSize = sendData_.size();

    // 发送数据包尺寸
    socket_->SendAsync(&sendSize, sizeof(sendSize));
    HCCL_INFO(
        "[AicpuTsUboeUbRtpChannelHelper::%s] Send size[%u] of data success. [%zu] bytes sent.", __func__, sendSize,
        sizeof(sendSize));
}

void AicpuTsUboeUbRtpChannelHelper::RecvDataSize()
{
    // 接收数据包尺寸
    socket_->RecvAsync(ReinterpretAs<u8*>(&recvDataSize_), sizeof(recvDataSize_));
    HCCL_INFO("[AicpuTsUboeUbRtpChannelHelper::%s] Receive Data Size", __func__);
}

void AicpuTsUboeUbRtpChannelHelper::SendExchangeData()
{
    socket_->SendAsync(sendData_.data(), sendData_.size());
    HCCL_INFO("[AicpuTsUboeUbRtpChannelHelper::%s] send data, size=%llu", __func__, sendData_.size());
}

void AicpuTsUboeUbRtpChannelHelper::RecvExchangeData()
{
    recvData_.resize(recvDataSize_);
    socket_->RecvAsync(ReinterpretAs<u8*>(recvData_.data()), recvData_.size());
    HCCL_INFO("[AicpuTsUboeUbRtpChannelHelper::%s] recv data", __func__);
}

bool AicpuTsUboeUbRtpChannelHelper::RecvDataProcess()
{
    HCCL_INFO("RecvDataProcess: size=%llu, recvDataSize=%u", recvData_.size(), recvDataSize_);
    Hccl::BinaryStream binaryStream(recvData_);
    RmtBufferVecUnpackProc(notifyNum_, binaryStream, rmtNotifyVec_, UboeRmtBufType::NOTIFY);
    RmtBufferVecUnpackProc(bufferNum_, binaryStream, rmtBufferVec_, UboeRmtBufType::BUFFER);
    RmtDrainBufferUnpackProc(binaryStream);
    bool result = false;
    if (ConnVecUnpackProc(binaryStream, result) != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR("[%s] ConnVecUnpackProc failed", __func__);
        return false;
    }
    return result;
}

void AicpuTsUboeUbRtpChannelHelper::RmtBufferVecUnpackProc(
    u32 locNum, Hccl::BinaryStream& binaryStream, RemoteBufferVec& bufferVec, UboeRmtBufType type)
{
    u32 rmtNum;
    binaryStream >> rmtNum;
    if (type == UboeRmtBufType::BUFFER && rmtNum > MAX_BUFFER_NUM) {
        MACRO_THROW(
            Hccl::InvalidParamsException,
            Hccl::StringFormat(
                "[AicpuTsUboeUbRtpChannelHelper][RmtBufferVecUnpackProc] rmtNum[%u] exceeds limit[%u]", rmtNum,
                MAX_BUFFER_NUM));
    }

    HCCL_INFO("unpack %s, locNum=%u, rmtNum=%u", type.Describe().c_str(), locNum, rmtNum);

    for (u32 i = 0; i < rmtNum; i++) {
        u32 pos;
        binaryStream >> pos;
        Hccl::ExchangeUbBufferDto dto;
        dto.Deserialize(binaryStream);
        if (bufferVec.size() > pos) {
            continue;
        }
        HCCL_INFO("unpack %s pos=%u, dto %s", type.Describe().c_str(), pos, dto.Describe().c_str());
        if (dto.size == 0) {
            HCCL_INFO("unpack nullptr, pos=%u", pos);
            bufferVec.push_back(nullptr);
        } else {
            bufferVec.push_back(std::make_unique<Hccl::RemoteUbRmaBuffer>(rdmaHandle_, dto));
            HCCL_INFO("unpack buffer pos=%u, rmtRmaBuffer=%s", pos, bufferVec.back()->Describe().c_str());
        }
    }
}

void AicpuTsUboeUbRtpChannelHelper::RmtDrainBufferUnpackProc(Hccl::BinaryStream& binaryStream)
{
    HCCL_INFO("[AicpuTsUboeUbRtpChannelHelper::%s] start unpack drain buffer", __func__);
    Hccl::ExchangeUbBufferDto dto;
    dto.Deserialize(binaryStream);

    if (dto.size == 0) {
        rmtDrainBuffer_ = nullptr;
        HCCL_WARNING("[AicpuTsUboeUbRtpChannelHelper::%s] unpack drain buffer dto is null", __func__);
    } else {
        rmtDrainBuffer_ = std::make_unique<Hccl::RemoteUbRmaBuffer>(rdmaHandle_, dto);
        HCCL_INFO(
            "[AicpuTsUboeUbRtpChannelHelper::%s] unpack drain buffer rmtDrainBuffer=%s", __func__,
            rmtDrainBuffer_->Describe().c_str());
    }
}

HcclResult AicpuTsUboeUbRtpChannelHelper::ConnVecUnpackProc(Hccl::BinaryStream& binaryStream, bool& result)
{
    u32 rmtConnNum;
    binaryStream >> rmtConnNum;
    HCCL_INFO("start unpack conn connNum=%u, rmtConnNum=%u", connNum_, rmtConnNum);
    if (connNum_ != rmtConnNum) {
        HCCL_ERROR("[%s] connNum=%u is not equal to rmtConnNum=%u", __func__, connNum_, rmtConnNum);
        return HcclResult::HCCL_E_PARA;
    }

    result = false;
    for (u32 i = 0; i < rmtConnNum; i++) {
        u32 pos;
        binaryStream >> pos;
        Hccl::ExchangeUbConnDto rmtDto;
        rmtDto.Deserialize(binaryStream);
        HCCL_INFO("unpack connection pos=%u dto %s", pos, rmtDto.Describe().c_str());
        if (commonRes_.connVec[i]->GetStatus() != Hccl::RmaConnStatus::READY) {
            HCCL_INFO(
                "parse and import pos=%u, rmt dto to connection[%s]", pos, commonRes_.connVec[i]->Describe().c_str());
            commonRes_.connVec[i]->ParseRmtExchangeDto(rmtDto);
            commonRes_.connVec[i]->ImportRmtDto();
            result = true;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

static HcclResult SetUboeModuleDataName(Hccl::ModuleData& module, const std::string& name)
{
    int ret = strcpy_s(module.name, sizeof(module.name), name.c_str());
    if (ret != 0) {
        HCCL_ERROR("[SetModuleDataName] strcpy_s name %s failed", name.c_str());
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

std::vector<char> AicpuTsUboeUbRtpChannelHelper::GetNotifyUniqueIds()
{
    HCCL_INFO("start packing all notify uniqueIds");
    std::vector<char> result(0);
    for (auto& it : commonRes_.notifyVec) {
        HCCL_INFO("AicpuTsUboeUbRtpChannelHelper Notify %s", it->Describe().c_str());
        auto uniqueId = it->GetUniqueId();
        result.insert(result.end(), uniqueId.begin(), uniqueId.end());
    }
    return result;
}

std::vector<char> AicpuTsUboeUbRtpChannelHelper::GetSingleRmtBufferUniqueId(
    u64 addr, u64 size, u32 tokenId, u32 tokenValue, u32 notifyId) const
{
    Hccl::BinaryStream binaryStream;
    binaryStream << addr;
    binaryStream << size;
    binaryStream << tokenId;
    binaryStream << tokenValue;
    binaryStream << notifyId;
    HCCL_INFO("AicpuTsUboeUbRtpChannelHelper RmtBuffer[addr=0x%llx, size=0x%llx, notifyId=%u]", addr, size, notifyId);
    std::vector<char> result;
    binaryStream.Dump(result);
    return result;
}

std::vector<char>
AicpuTsUboeUbRtpChannelHelper::GetSingleLocBufferUniqueId(u64 addr, u64 size, u32 tokenId, u32 tokenValue) const
{
    Hccl::BinaryStream binaryStream;
    binaryStream << addr;
    binaryStream << size;
    binaryStream << tokenId;
    binaryStream << tokenValue;
    HCCL_INFO("AicpuTsUboeUbRtpChannelHelper LocBuffer[addr=0x%llx, size=0x%llx]", addr, size);
    std::vector<char> result;
    binaryStream.Dump(result);
    return result;
}

std::vector<char>
AicpuTsUboeUbRtpChannelHelper::GetRmtBufferUniqueIds(RemoteBufferVec& bufferVec, UboeRmtBufType type) const
{
    HCCL_INFO("start packing all remote buffer %s uniqueIds", type.Describe().c_str());
    std::vector<char> result(0);
    for (auto& it : bufferVec) {
        std::vector<char> uniqueId;
        if (it != nullptr) {
            uniqueId = GetSingleRmtBufferUniqueId(
                it->GetAddr(), it->GetSize(), it->GetTokenId(), it->GetTokenValue(), it->GetNotifyId());
            HCCL_INFO("AicpuTsUboeUbRtpChannelHelper::GetRmtBufferUniqueIds, %s", it->Describe().c_str());
        } else {
            uniqueId = GetSingleRmtBufferUniqueId(0, 0, 0, 0, UINT32_MAX);
            HCCL_INFO("AicpuTsUboeUbRtpChannelHelper::GetRmtBufferUniqueIds, null buffer");
        }
        result.insert(result.end(), uniqueId.begin(), uniqueId.end());
    }
    return result;
}

std::vector<char>
AicpuTsUboeUbRtpChannelHelper::GetLocBufferUniqueIds(LocalBufferVec& bufferVec, UboeRmtBufType type) const
{
    HCCL_INFO("start packing all local buffer %s uniqueIds", type.Describe().c_str());
    std::vector<char> result(0);
    for (auto& it : bufferVec) {
        std::vector<char> uniqueId;
        if (it != nullptr) {
            uniqueId = GetSingleLocBufferUniqueId(it->GetAddr(), it->GetSize(), it->GetTokenId(), it->GetTokenValue());
            HCCL_INFO("UbMemTransport::GetLocBufferUniqueIds, %s", it->Describe().c_str());
        } else {
            uniqueId = GetSingleLocBufferUniqueId(0, 0, 0, 0);
            HCCL_INFO("UbMemTransport::GetLocBufferUniqueIds, null buffer");
        }
        result.insert(result.end(), uniqueId.begin(), uniqueId.end());
    }
    return result;
}

std::vector<char> AicpuTsUboeUbRtpChannelHelper::GetDrainUniqueIds() const
{
    HCCL_INFO("[AicpuTsUboeUbRtpChannelHelper::%s] start get drain resource uniqueIds", __func__);
    std::vector<char> result(0);
    std::vector<char> uniqueId;

    // drain notify UniqueId
    if (drainNotify_ != nullptr) {
        auto dto = drainNotify_->GetExchangeDto();
        Hccl::ExchangeUbBufferDto* rawDto = static_cast<Hccl::ExchangeUbBufferDto*>(dto.get());
        uniqueId = GetSingleRmtBufferUniqueId(
            rawDto->addr, rawDto->size, rawDto->tokenId, rawDto->tokenValue, rawDto->notifyId);
        HCCL_INFO("[AicpuTsUboeUbRtpChannelHelper::%s] %s", __func__, drainNotify_->Describe().c_str());
    } else {
        uniqueId = GetSingleRmtBufferUniqueId(0, 0, 0, 0, UINT32_MAX); // 填充一个空的buffer
        HCCL_WARNING("[AicpuTsUboeUbRtpChannelHelper::%s] drainNotify_ null buffer", __func__);
    }
    result.insert(result.end(), uniqueId.begin(), uniqueId.end());

    // drain buffer UniqueId
    if (rmtDrainBuffer_ != nullptr) {
        uniqueId = GetSingleRmtBufferUniqueId(
            rmtDrainBuffer_->GetAddr(), rmtDrainBuffer_->GetSize(), rmtDrainBuffer_->GetTokenId(),
            rmtDrainBuffer_->GetTokenValue(), rmtDrainBuffer_->GetNotifyId());
        HCCL_INFO("[AicpuTsUboeUbRtpChannelHelper::%s] %s", __func__, rmtDrainBuffer_->Describe().c_str());
    } else {
        uniqueId = GetSingleRmtBufferUniqueId(0, 0, 0, 0, UINT32_MAX); // 填充一个空的buffer
        HCCL_WARNING("[AicpuTsUboeUbRtpChannelHelper::%s] rmtDrainBuffer_ null buffer", __func__);
    }
    result.insert(result.end(), uniqueId.begin(), uniqueId.end());

    return result;
}

std::vector<char> AicpuTsUboeUbRtpChannelHelper::GetConnUniqueIds()
{
    HCCL_INFO("start packing all conn uniqueIds");
    std::vector<char> result(0);
    for (auto& it : commonRes_.connVec) {
        HCCL_INFO("[AicpuTsUboeUbRtpChannelHelper::%s] conn[%s]", __func__, it->Describe().c_str());
        auto uniqueId = it->GetUniqueId();
        result.insert(result.end(), uniqueId.begin(), uniqueId.end());
    }
    return result;
}

HcclResult AicpuTsUboeUbRtpChannelHelper::GetUniqueIdV2(std::vector<char>& result)
{
    if (channelStatus != ChannelStatus::READY) {
        HCCL_ERROR(
            "[%s] channel status[%d] is not ready[%d], please check.", __func__, channelStatus, ChannelStatus::READY);
        return HcclResult::HCCL_E_INTERNAL;
    }
    std::lock_guard<std::mutex> lock(remoteMemsMutex_);
    Hccl::BinaryStream binaryStream;
    binaryStream << type_;
    binaryStream << notifyNum_;
    binaryStream << bufferNum_;
    binaryStream << static_cast<u32>(rmtBufferVec_.size());
    binaryStream << connNum_;

    auto notifyUniqueIds = GetNotifyUniqueIds();
    binaryStream << notifyUniqueIds;

    auto rmtNotifyUniqueIds = GetRmtBufferUniqueIds(rmtNotifyVec_, UboeRmtBufType::NOTIFY);
    binaryStream << rmtNotifyUniqueIds;

    locBufferVec_.clear();
    for (auto& it : commonRes_.bufferVec) {
        locBufferVec_.emplace_back(ReinterpretAs<Hccl::LocalUbRmaBuffer*>(it));
    }

    auto locBufferUniqueIds = GetLocBufferUniqueIds(locBufferVec_, UboeRmtBufType::BUFFER);
    binaryStream << locBufferUniqueIds;

    auto rmtBufferUniqueIds = GetRmtBufferUniqueIds(rmtBufferVec_, UboeRmtBufType::BUFFER);
    binaryStream << rmtBufferUniqueIds;

    auto drainUniqueIds = GetDrainUniqueIds();
    binaryStream << drainUniqueIds;

    auto connUniqueIds = GetConnUniqueIds();
    binaryStream << connUniqueIds;

    binaryStream.Dump(result);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult AicpuTsUboeUbRtpChannelHelper::CheckSocketStatus(const std::string& socketOperator)
{
    CHK_PTR_NULL(socket_);
    auto timeout = std::chrono::seconds(Hccl::EnvConfig::GetInstance().GetSocketConfig().GetLinkTimeOut());
    auto startTime = std::chrono::steady_clock::now();
    uint32_t retryCount = 0;
    while (true) {
        Hccl::SocketStatus socketStatus = socket_->GetAsyncStatus();
        if (socketStatus == Hccl::SocketStatus::OK) {
            auto elapsed
                = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime)
                      .count();
            HCCL_INFO(
                "[AicpuTsUboeUbRtpChannelHelper][%s] socket operation[%s] success, elapsed[%lld]ms, retryCount[%u]",
                __func__, socketOperator.c_str(), elapsed, retryCount);
            break;
        }
        if ((std::chrono::steady_clock::now() - startTime) >= timeout || socketStatus == Hccl::SocketStatus::TIMEOUT) {
            auto elapsed
                = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime)
                      .count();
            HCCL_ERROR(
                "[AicpuTsUboeUbRtpChannelHelper][%s] socket operation[%s] timeout, socketStatus[%u], elapsed[%lld]ms, "
                "retryCount[%u]",
                __func__, socketOperator.c_str(), static_cast<uint32_t>(socketStatus), elapsed, retryCount);
            return HCCL_E_TIMEOUT;
        }
        retryCount++;
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUboeUbRtpChannelHelper::UpdateMemInfo(HcommMemHandle* memHandles, uint32_t memHandleNum)
{
    std::vector<Hccl::LocalRmaBuffer*> bufferVecTemp;
    CHK_RET(MakeRmaBufferVecFromMemHandles(memHandles, memHandleNum, bufferVecTemp, "AicpuTsUboeUbRtpChannelHelper"));

    if (bufferVecTemp.size() == 0) {
        HCCL_WARNING("[AicpuTsUboeUbRtpChannelHelper][%s] bufferNum is 0.", __func__);
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(socket_);
    HCCL_INFO("[AicpuTsUboeUbRtpChannelHelper][%s] bufferNum[%zu]", __func__, bufferVecTemp.size());

    uint32_t totalBufferNum = static_cast<uint32_t>(commonRes_.bufferVec.size() + bufferVecTemp.size());
    if (UNLIKELY(totalBufferNum > MAX_BUFFER_NUM)) {
        HCCL_ERROR(
            "[AicpuTsUboeUbRtpChannelHelper][%s] totalBufferNum[%u] exceeds limit[%u]", __func__, totalBufferNum,
            MAX_BUFFER_NUM);
        return HCCL_E_PARA;
    }

    std::vector<char> localSendData;
    Hccl::BinaryStream sendStream;
    BufferVecPack(sendStream, bufferVecTemp);
    sendStream.Dump(localSendData);

    u32 sendSize = localSendData.size();
    socket_->SendAsync(&sendSize, sizeof(sendSize));
    HCCL_INFO("[AicpuTsUboeUbRtpChannelHelper][%s] Send size[%u] of data.", __func__, sendSize);
    CHK_RET(CheckSocketStatus("SendDataSize"));

    u32 recvSize = 0;
    socket_->RecvAsync(ReinterpretAs<u8*>(&recvSize), sizeof(recvSize));
    CHK_RET(CheckSocketStatus("RecvDataSize"));
    HCCL_INFO("[AicpuTsUboeUbRtpChannelHelper][%s] Recv size[%u] of data.", __func__, recvSize);

    // 对端声明的数据长度不可信，为0或超过上限时拒绝分配，防止超大内存申请导致OOM
    CHK_PRT_RET(
        recvSize == 0 || recvSize > MAX_EXCHANGE_DATA_SIZE,
        HCCL_ERROR(
            "[AicpuTsUboeUbRtpChannelHelper][%s] recvSize[%u] is invalid, limit[%llu]", __func__, recvSize,
            MAX_EXCHANGE_DATA_SIZE),
        HCCL_E_PARA);

    socket_->SendAsync(localSendData.data(), localSendData.size());
    HCCL_INFO("[AicpuTsUboeUbRtpChannelHelper][%s] Send data, size[%zu].", __func__, localSendData.size());
    CHK_RET(CheckSocketStatus("SendExchangeData"));

    std::vector<char> localRecvData(recvSize);
    socket_->RecvAsync(ReinterpretAs<u8*>(localRecvData.data()), localRecvData.size());
    CHK_RET(CheckSocketStatus("RecvExchangeData"));
    HCCL_INFO("[AicpuTsUboeUbRtpChannelHelper][%s] Recv data success.", __func__);

    std::vector<std::unique_ptr<Hccl::RemoteUbRmaBuffer>> rmtBufferTemp{};
    Hccl::BinaryStream recvStream(localRecvData);
    u32 locNum = static_cast<u32>(bufferVecTemp.size());
    RmtBufferVecUnpackProc(locNum, recvStream, rmtBufferTemp, UboeRmtBufType::BUFFER);
    if (rmtBufferTemp.size() != locNum) {
        HCCL_ERROR(
            "[AicpuTsUboeUbRtpChannelHelper][%s] buffer num mismatch, loc[%u] rmt[%zu]", __func__, locNum,
            rmtBufferTemp.size());
        return HCCL_E_INTERNAL;
    }

    {
        std::lock_guard<std::mutex> lock(remoteMemsMutex_);
        rmtBufferVec_.insert(
            rmtBufferVec_.end(), std::make_move_iterator(rmtBufferTemp.begin()),
            std::make_move_iterator(rmtBufferTemp.end()));
        commonRes_.bufferVec.insert(commonRes_.bufferVec.end(), bufferVecTemp.begin(), bufferVecTemp.end());
        cacheValid_ = false;
        bufferNum_ = static_cast<u32>(commonRes_.bufferVec.size());
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUboeUbRtpChannelHelper::H2DResPack(std::vector<char>& buffer)
{
    std::vector<Hccl::ModuleData> dataVec;
    dataVec.resize(Hccl::AicpuResMgrType::__COUNT__);

    Hccl::AicpuResMgrType resType = Hccl::AicpuResMgrType::STREAM;
    CHK_RET(SetUboeModuleDataName(dataVec[resType], "AicpuTsUboeUbRtpChannelHelper"));

    std::vector<char> result;
    Hccl::BinaryStream binaryStream;
    std::vector<char> uniqueIdV2;
    CHK_RET(GetUniqueIdV2(uniqueIdV2));
    binaryStream << uniqueIdV2;

    binaryStream.Dump(result);

    dataVec[resType].data = result;

    Hccl::AicpuResPackageHelper helper;
    buffer = helper.GetPackedData(dataVec);
    HCCL_INFO(
        "[AicpuTsUboeUbRtpChannelHelper][%s] Pack Buffer data[%p], Pack Buffer size[%zu].", __func__, buffer.data(),
        buffer.size());
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUboeUbRtpChannelHelper::Clean()
{
    // 该模式当前不支持N秒快恢
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUboeUbRtpChannelHelper::Resume()
{
    // 该模式当前不支持N秒快恢
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUboeUbRtpChannelHelper::NotifyRecord([[maybe_unused]] const uint32_t remoteNotifyIdx)
{
    HCCL_INFO("[AicpuTsUboeUbRtpChannelHelper::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AicpuTsUboeUbRtpChannelHelper::NotifyWait(
    [[maybe_unused]] const uint32_t localNotifyIdx, [[maybe_unused]] const uint32_t timeout)
{
    HCCL_INFO("[AicpuTsUboeUbRtpChannelHelper::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AicpuTsUboeUbRtpChannelHelper::WriteWithNotify(
    [[maybe_unused]] void* dst, [[maybe_unused]] const void* src, [[maybe_unused]] const uint64_t len,
    [[maybe_unused]] uint32_t remoteNotifyIdx)
{
    HCCL_INFO("[AicpuTsUboeUbRtpChannelHelper::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AicpuTsUboeUbRtpChannelHelper::Write(
    [[maybe_unused]] void* dst, [[maybe_unused]] const void* src, [[maybe_unused]] uint64_t len)
{
    HCCL_INFO("[AicpuTsUboeUbRtpChannelHelper::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AicpuTsUboeUbRtpChannelHelper::Read(
    [[maybe_unused]] void* dst, [[maybe_unused]] const void* src, [[maybe_unused]] uint64_t len)
{
    HCCL_INFO("[AicpuTsUboeUbRtpChannelHelper::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult AicpuTsUboeUbRtpChannelHelper::ChannelFence()
{
    HCCL_INFO("[AicpuTsUboeUbRtpChannelHelper::%s] not supported yet.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

} // namespace hcomm
