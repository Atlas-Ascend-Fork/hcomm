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
#include "aicpu_ts_uboe_ub_rtp_channel_helper.h"
#include "endpoint.h"
#include "../../sockets/socket_mgr.h"
#include "base_comm_node_accessors.h"
#include "orion_adpt_utils.h"
#include "env_config/env_config_v2.h"
#include "rma_buf_entry_lite.h"

// Orion
#include "virtual_topo.h"
#include "dev_capability.h"
#include "dev_buffer.h"
#include "aicpu_res_package_helper.h"
#include "exchange_ub_buffer_dto.h"
#include "exchange_ub_conn_dto.h"
#include "makebufs_helper.h"

namespace hcomm {

AicpuTsUboeUbRtpChannelHelper::AicpuTsUboeUbRtpChannelHelper(
    EndpointHandle endpointHandle, const HcommChannelDesc& channelDesc, CommEngine engine, HcommChannelKind channelKind,
    const char* channelClassName)
    : Channel(endpointHandle, channelDesc, engine, channelKind, channelClassName)
{}

AicpuTsUboeUbRtpChannelHelper::~AicpuTsUboeUbRtpChannelHelper() {}

HcclResult AicpuTsUboeUbRtpChannelHelper::ParseInputParam()
{
    CHK_RET(Channel::ParseInputParam());

    CHK_RET(CollectLocalRmaBuffers(local_.commonRes.bufferVec));

    return HCCL_SUCCESS;
}

HcclResult AicpuTsUboeUbRtpChannelHelper::BuildConn()
{
    CHK_RET(BuildStateMachineConn());
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUboeUbRtpChannelHelper::BuildDrainResource()
{
    // 申请创建channel drain阻塞等待的相关资源
    // 申请notify作为read的落点
    bool devUsed = true;
    EXCEPTION_CATCH(
        drainNotify_ = std::make_unique<Hccl::UbLocalNotify>(local_.rdmaHandle, devUsed), return HCCL_E_PTR);
    HCCL_INFO("[%s::%s] drainNotify created: %s", channelClassName_, __func__, drainNotify_->Describe().c_str());

    // 常量1内存供远端读取
    u32 notifySize = Hccl::DevCapability::GetInstance().GetNotifySize();

    std::shared_ptr<Hccl::DevBuffer> constMem;
    EXCEPTION_CATCH(constMem = std::make_shared<Hccl::DevBuffer>(notifySize), return HCCL_E_PTR);

    Hccl::HrtMemcpy(
        ReinterpretAs<void*>(constMem->GetAddr()), constMem->GetSize(), &NORMAL_NOTIFY_VAL, sizeof(NORMAL_NOTIFY_VAL),
        Hccl::tagRtMemcpyKind::RT_MEMCPY_HOST_TO_DEVICE);

    EXCEPTION_CATCH(
        drainBuffer_ = std::make_unique<Hccl::LocalUbRmaBuffer>(constMem, local_.rdmaHandle), return HCCL_E_PTR);
    HCCL_INFO(
        "[%s::%s] drain buffer created: addr[0x%llx], size[%zu]", channelClassName_, __func__,
        static_cast<unsigned long long>(drainBuffer_->GetAddr()), drainBuffer_->GetSize());

    return HCCL_SUCCESS;
}

HcclResult AicpuTsUboeUbRtpChannelHelper::GetNotifyNum(uint32_t* notifyNum) const
{
    *notifyNum = this->local_.notifyNum;
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUboeUbRtpChannelHelper::GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos)
{
    return remote_.memCache.Get(rmtBufferVec_, memNum, remoteMem, memInfos);
}

bool AicpuTsUboeUbRtpChannelHelper::IsSocketReady()
{
    if (local_.socket == nullptr) {
        HCCL_ERROR("[%s::%s] socket is nullptr, please check", channelClassName_, __func__);
        channelStatus = ChannelStatus::INVALID;
        return false;
    }

    Hccl::SocketStatus socketStatus = local_.socket->GetAsyncStatus();
    if (socketStatus == Hccl::SocketStatus::OK) {
        channelStatus = ChannelStatus::SOCKET_OK;
        return true;
    } else if (socketStatus == Hccl::SocketStatus::TIMEOUT) {
        channelStatus = ChannelStatus::SOCKET_TIMEOUT;
        return false;
    }

    return false;
}

HcclResult AicpuTsUboeUbRtpChannelHelper::IsResReady(bool& isReady)
{
    for (auto& it : local_.commonRes.connVec) {
        if (it == nullptr) {
            HCCL_ERROR("[%s] connection pointer is nullptr", __func__);
            return HCCL_E_PTR;
        }
        Hccl::RmaConnType connType = it->GetRmaConnType();
        if (connType != Hccl::RmaConnType::UB) {
            HCCL_ERROR("[%s] connection type[%s] is not ub", __func__, connType.Describe().c_str());
            return HCCL_E_INTERNAL;
        }

        auto status = it->GetStatus();
        if (status != Hccl::RmaConnStatus::EXCHANGEABLE && status != Hccl::RmaConnStatus::READY) {
            isReady = false;
            return HCCL_SUCCESS;
        }
    }
    HCCL_INFO("[%s] all resources ready.", __func__);
    isReady = true;
    return HCCL_SUCCESS;
}

bool AicpuTsUboeUbRtpChannelHelper::IsConnsReady()
{
    for (u32 i = 0; i < connNum_; i++) {
        if (local_.commonRes.connVec[i]->GetStatus() != Hccl::RmaConnStatus::READY) {
            return false;
        }
    }
    HCCL_INFO("[AicpuTsUboeUbRtpChannelHelper::%s] conns are ready.", __func__);
    return true;
}

void AicpuTsUboeUbRtpChannelHelper::NotifyVecPack(Hccl::BinaryStream& binaryStream)
{
    binaryStream << local_.notifyNum;
    u32 pos = 0;
    HCCL_INFO("[AicpuTsUboeUbRtpChannelHelper::%s] pack notify size[%d]", __func__, local_.commonRes.notifyVec.size());
    for (auto& it : local_.commonRes.notifyVec) {
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
    HCCL_INFO("[%s::%s] start pack drain buffer", channelClassName_, __func__);
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
    HCCL_INFO("[AicpuTsUboeUbRtpChannelHelper::%s] pack conn size[%d]", __func__, local_.commonRes.connVec.size());
    for (auto& it : local_.commonRes.connVec) {
        binaryStream << pos;
        std::unique_ptr<Hccl::Serializable> dto = it->GetExchangeDto();
        dto->Serialize(binaryStream);
        HCCL_INFO(
            "[AicpuTsUboeUbRtpChannelHelper::%s] pack connection pos=%u, dto %s", __func__, pos,
            dto->Describe().c_str());
        pos++;
    }
}

HcclResult AicpuTsUboeUbRtpChannelHelper::SendDataSize()
{
    sendData_.clear();
    bufferNum_ = local_.commonRes.bufferVec.size();
    connNum_ = local_.commonRes.connVec.size();

    HCCL_INFO(
        "[%s::%s] notifyNum=%u, bufferNum=%u, connNum=%u", channelClassName_, __func__, local_.notifyNum, bufferNum_,
        connNum_);

    Hccl::BinaryStream binaryStream;
    NotifyVecPack(binaryStream);
    BufferVecPack(binaryStream, local_.commonRes.bufferVec);
    DrainBufferPack(binaryStream);
    ConnVecPack(binaryStream);

    binaryStream.Dump(sendData_);
    u32 sendSize = sendData_.size();

    // 发送数据包尺寸
    TRY_CATCH_RETURN(local_.socket->SendAsync(&sendSize, sizeof(sendSize)););
    HCCL_INFO(
        "[AicpuTsUboeUbRtpChannelHelper::%s] Send size[%u] of data success. [%zu] bytes sent.", __func__, sendSize,
        sizeof(sendSize));
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUboeUbRtpChannelHelper::RecvDataSize()
{
    // 接收数据包尺寸
    TRY_CATCH_RETURN(local_.socket->RecvAsync(ReinterpretAs<u8*>(&recvDataSize_), sizeof(recvDataSize_)););
    HCCL_INFO("[AicpuTsUboeUbRtpChannelHelper::%s] Receive Data Size", __func__);
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUboeUbRtpChannelHelper::SendExchangeData()
{
    TRY_CATCH_RETURN(local_.socket->SendAsync(sendData_.data(), sendData_.size()););
    HCCL_INFO("[AicpuTsUboeUbRtpChannelHelper::%s] send data, size=%llu", __func__, sendData_.size());
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUboeUbRtpChannelHelper::RecvExchangeData()
{
    recvData_.resize(recvDataSize_);
    TRY_CATCH_RETURN(local_.socket->RecvAsync(ReinterpretAs<u8*>(recvData_.data()), recvData_.size()););
    HCCL_INFO("[AicpuTsUboeUbRtpChannelHelper::%s] recv data", __func__);
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUboeUbRtpChannelHelper::RecvDataProcess(bool& needFinish)
{
    HCCL_INFO(
        "[%s::%s] RecvDataProcess: size=%llu, recvDataSize=%u", channelClassName_, __func__, recvData_.size(),
        recvDataSize_);
    Hccl::BinaryStream binaryStream(recvData_);
    CHK_RET(RmtBufferVecUnpackProc(local_.notifyNum, binaryStream, rmtNotifyVec_, UboeRmtBufType::NOTIFY));
    CHK_RET(RmtBufferVecUnpackProc(bufferNum_, binaryStream, rmtBufferVec_, UboeRmtBufType::BUFFER));
    TRY_CATCH_RETURN(RmtDrainBufferUnpackProc(binaryStream););
    bool result = false;
    CHK_RET(ConnVecUnpackProc(binaryStream, result));
    needFinish = result;
    return HCCL_SUCCESS;
}

HcclResult AicpuTsUboeUbRtpChannelHelper::RmtBufferVecUnpackProc(
    u32 locNum, Hccl::BinaryStream& binaryStream, RemoteBufferVec& bufferVec, UboeRmtBufType type)
{
    u32 rmtNum;
    binaryStream >> rmtNum;
    if (type == UboeRmtBufType::BUFFER && rmtNum > MAX_BUFFER_NUM) {
        HCCL_ERROR("[%s] rmtNum[%u] exceeds limit[%u]", __func__, rmtNum, MAX_BUFFER_NUM);
        return HcclResult::HCCL_E_PARA;
    }

    HCCL_INFO(
        "[%s::%s] unpack %s, locNum=%u, rmtNum=%u", channelClassName_, __func__, type.Describe().c_str(), locNum,
        rmtNum);

    for (u32 i = 0; i < rmtNum; i++) {
        u32 pos;
        binaryStream >> pos;
        Hccl::ExchangeUbBufferDto dto;
        dto.Deserialize(binaryStream);
        if (bufferVec.size() > pos) {
            continue;
        }
        HCCL_INFO(
            "[%s::%s] unpack %s pos=%u, dto %s", channelClassName_, __func__, type.Describe().c_str(), pos,
            dto.Describe().c_str());
        if (dto.size == 0) {
            HCCL_INFO("[%s::%s] unpack nullptr, pos=%u", channelClassName_, __func__, pos);
            bufferVec.push_back(nullptr);
        } else {
            bufferVec.push_back(std::make_unique<Hccl::RemoteUbRmaBuffer>(local_.rdmaHandle, dto));
            HCCL_INFO(
                "[%s::%s] unpack buffer pos=%u, rmtRmaBuffer=%s", channelClassName_, __func__, pos,
                bufferVec.back()->Describe().c_str());
        }
    }
    return HcclResult::HCCL_SUCCESS;
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
        rmtDrainBuffer_ = std::make_unique<Hccl::RemoteUbRmaBuffer>(local_.rdmaHandle, dto);
        HCCL_INFO(
            "[AicpuTsUboeUbRtpChannelHelper::%s] unpack drain buffer rmtDrainBuffer=%s", __func__,
            rmtDrainBuffer_->Describe().c_str());
    }
}

HcclResult AicpuTsUboeUbRtpChannelHelper::ConnVecUnpackProc(Hccl::BinaryStream& binaryStream, bool& result)
{
    u32 rmtConnNum;
    binaryStream >> rmtConnNum;
    HCCL_INFO(
        "[%s::%s] start unpack conn connNum=%u, rmtConnNum=%u", channelClassName_, __func__, connNum_, rmtConnNum);
    if (connNum_ != rmtConnNum) {
        HCCL_ERROR(
            "[%s::%s] connNum=%u is not equal to rmtConnNum=%u", channelClassName_, __func__, connNum_, rmtConnNum);
        return HcclResult::HCCL_E_PARA;
    }

    result = false;
    for (u32 i = 0; i < rmtConnNum; i++) {
        u32 pos;
        binaryStream >> pos;
        Hccl::ExchangeUbConnDto rmtDto;
        rmtDto.Deserialize(binaryStream);
        HCCL_INFO(
            "[%s::%s] unpack connection pos=%u dto %s", channelClassName_, __func__, pos, rmtDto.Describe().c_str());
        if (local_.commonRes.connVec[i]->GetStatus() != Hccl::RmaConnStatus::READY) {
            HCCL_INFO(
                "[%s::%s] parse and import pos=%u, rmt dto to connection[%s]", channelClassName_, __func__, pos,
                local_.commonRes.connVec[i]->Describe().c_str());
            CHK_RET(local_.commonRes.connVec[i]->ParseRmtExchangeDto(rmtDto));
            CHK_RET(local_.commonRes.connVec[i]->ImportRmtDto());
            result = true;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

static HcclResult SetUboeModuleDataName(Hccl::ModuleData& module, const std::string& name)
{
    int ret = strcpy_s(module.name, sizeof(module.name), name.c_str());
    if (ret != 0) {
        HCCL_ERROR("[AicpuTsUboeUbRtpChannelHelper::%s] strcpy_s name %s failed", __func__, name.c_str());
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

std::vector<char> AicpuTsUboeUbRtpChannelHelper::GetNotifyUniqueIds()
{
    HCCL_INFO("[%s::%s] start packing all notify uniqueIds", channelClassName_, __func__);
    std::vector<char> result(0);
    for (auto& it : local_.commonRes.notifyVec) {
        HCCL_INFO(
            "[%s::%s] AicpuTsUboeUbRtpChannelHelper Notify %s", channelClassName_, __func__, it->Describe().c_str());
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
    HCCL_INFO(
        "[%s::%s] AicpuTsUboeUbRtpChannelHelper RmtBuffer[addr=0x%llx, size=0x%llx, notifyId=%u]", channelClassName_,
        __func__, addr, size, notifyId);
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
    HCCL_INFO(
        "[%s::%s] AicpuTsUboeUbRtpChannelHelper LocBuffer[addr=0x%llx, size=0x%llx]", channelClassName_, __func__, addr,
        size);
    std::vector<char> result;
    binaryStream.Dump(result);
    return result;
}

std::vector<char>
AicpuTsUboeUbRtpChannelHelper::GetRmtBufferUniqueIds(RemoteBufferVec& bufferVec, UboeRmtBufType type) const
{
    HCCL_INFO(
        "[%s::%s] start packing all remote buffer %s uniqueIds", channelClassName_, __func__, type.Describe().c_str());
    std::vector<char> result(0);
    for (auto& it : bufferVec) {
        std::vector<char> uniqueId;
        if (it != nullptr) {
            uniqueId = GetSingleRmtBufferUniqueId(
                it->GetAddr(), it->GetSize(), it->GetTokenId(), it->GetTokenValue(), it->GetNotifyId());
            HCCL_INFO(
                "[%s::%s] AicpuTsUboeUbRtpChannelHelper::GetRmtBufferUniqueIds, %s", channelClassName_, __func__,
                it->Describe().c_str());
        } else {
            uniqueId = GetSingleRmtBufferUniqueId(0, 0, 0, 0, UINT32_MAX);
            HCCL_INFO(
                "[%s::%s] AicpuTsUboeUbRtpChannelHelper::GetRmtBufferUniqueIds, null buffer", channelClassName_,
                __func__);
        }
        result.insert(result.end(), uniqueId.begin(), uniqueId.end());
    }
    return result;
}

std::vector<char>
AicpuTsUboeUbRtpChannelHelper::GetLocBufferUniqueIds(LocalBufferVec& bufferVec, UboeRmtBufType type) const
{
    HCCL_INFO(
        "[%s::%s] start packing all local buffer %s uniqueIds", channelClassName_, __func__, type.Describe().c_str());
    std::vector<char> result(0);
    for (auto& it : bufferVec) {
        std::vector<char> uniqueId;
        if (it != nullptr) {
            uniqueId = GetSingleLocBufferUniqueId(it->GetAddr(), it->GetSize(), it->GetTokenId(), it->GetTokenValue());
            HCCL_INFO(
                "[%s::%s] UbMemTransport::GetLocBufferUniqueIds, %s", channelClassName_, __func__,
                it->Describe().c_str());
        } else {
            uniqueId = GetSingleLocBufferUniqueId(0, 0, 0, 0);
            HCCL_INFO("[%s::%s] UbMemTransport::GetLocBufferUniqueIds, null buffer", channelClassName_, __func__);
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
    HCCL_INFO("[%s::%s] start packing all conn uniqueIds", channelClassName_, __func__);
    std::vector<char> result(0);
    for (auto& it : local_.commonRes.connVec) {
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
            "[%s::%s] channel status[%d] is not ready[%d], please check.", channelClassName_, __func__, channelStatus,
            ChannelStatus::READY);
        return HcclResult::HCCL_E_INTERNAL;
    }
    std::lock_guard<std::mutex> lock(remote_.memCache.mutex);
    Hccl::BinaryStream binaryStream;
    binaryStream << type_;
    binaryStream << local_.notifyNum;
    binaryStream << bufferNum_;
    binaryStream << static_cast<u32>(rmtBufferVec_.size());
    binaryStream << connNum_;

    auto notifyUniqueIds = GetNotifyUniqueIds();
    binaryStream << notifyUniqueIds;

    auto rmtNotifyUniqueIds = GetRmtBufferUniqueIds(rmtNotifyVec_, UboeRmtBufType::NOTIFY);
    binaryStream << rmtNotifyUniqueIds;

    locBufferVec_.clear();
    for (auto& it : local_.commonRes.bufferVec) {
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
    CHK_PTR_NULL(local_.socket);
    auto timeout = std::chrono::seconds(Hccl::EnvConfig::GetInstance().GetSocketConfig().GetLinkTimeOut());
    auto startTime = std::chrono::steady_clock::now();
    uint32_t retryCount = 0;
    while (true) {
        Hccl::SocketStatus socketStatus = local_.socket->GetAsyncStatus();
        if (socketStatus == Hccl::SocketStatus::OK) {
            auto elapsed
                = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime)
                      .count();
            HCCL_INFO(
                "[%s::%s] socket operation[%s] success, elapsed[%lld]ms, retryCount[%u]", channelClassName_, __func__,
                socketOperator.c_str(), elapsed, retryCount);
            break;
        }
        if ((std::chrono::steady_clock::now() - startTime) >= timeout || socketStatus == Hccl::SocketStatus::TIMEOUT) {
            auto elapsed
                = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime)
                      .count();
            HCCL_ERROR(
                "[%s::%s] socket operation[%s] timeout, socketStatus[%u], elapsed[%lld]ms, "
                "retryCount[%u]",
                channelClassName_, __func__, socketOperator.c_str(), static_cast<uint32_t>(socketStatus), elapsed,
                retryCount);
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
        HCCL_WARNING("[%s::%s] bufferNum is 0.", channelClassName_, __func__);
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(local_.socket);
    HCCL_INFO("[%s::%s] bufferNum[%zu]", channelClassName_, __func__, bufferVecTemp.size());

    uint32_t totalBufferNum = static_cast<uint32_t>(local_.commonRes.bufferVec.size() + bufferVecTemp.size());
    if (UNLIKELY(totalBufferNum > MAX_BUFFER_NUM)) {
        HCCL_ERROR(
            "[%s::%s] totalBufferNum[%u] exceeds limit[%u]", channelClassName_, __func__, totalBufferNum,
            MAX_BUFFER_NUM);
        return HCCL_E_PARA;
    }

    std::vector<char> localSendData;
    Hccl::BinaryStream sendStream;
    BufferVecPack(sendStream, bufferVecTemp);
    sendStream.Dump(localSendData);

    u32 sendSize = localSendData.size();
    TRY_CATCH_RETURN(local_.socket->SendAsync(&sendSize, sizeof(sendSize)););
    HCCL_INFO("[%s::%s] Send size[%u] of data.", channelClassName_, __func__, sendSize);
    CHK_RET(CheckSocketStatus("SendDataSize"));

    u32 recvSize = 0;
    TRY_CATCH_RETURN(local_.socket->RecvAsync(ReinterpretAs<u8*>(&recvSize), sizeof(recvSize)););
    CHK_RET(CheckSocketStatus("RecvDataSize"));
    HCCL_INFO("[%s::%s] Recv size[%u] of data.", channelClassName_, __func__, recvSize);

    // 对端声明的数据长度不可信，为0或超过上限时拒绝分配，防止超大内存申请导致OOM
    CHK_PRT_RET(
        recvSize == 0 || recvSize > MAX_EXCHANGE_DATA_SIZE,
        HCCL_ERROR(
            "[%s::%s] recvSize[%u] is invalid, limit[%llu]", channelClassName_, __func__, recvSize,
            MAX_EXCHANGE_DATA_SIZE),
        HCCL_E_PARA);

    TRY_CATCH_RETURN(local_.socket->SendAsync(localSendData.data(), localSendData.size()););
    HCCL_INFO("[%s::%s] Send data, size[%zu].", channelClassName_, __func__, localSendData.size());
    CHK_RET(CheckSocketStatus("SendExchangeData"));

    std::vector<char> localRecvData(recvSize);
    TRY_CATCH_RETURN(local_.socket->RecvAsync(ReinterpretAs<u8*>(localRecvData.data()), localRecvData.size()););
    CHK_RET(CheckSocketStatus("RecvExchangeData"));
    HCCL_INFO("[%s::%s] Recv data success.", channelClassName_, __func__);

    std::vector<std::unique_ptr<Hccl::RemoteUbRmaBuffer>> rmtBufferTemp{};
    Hccl::BinaryStream recvStream(localRecvData);
    u32 locNum = static_cast<u32>(bufferVecTemp.size());
    CHK_RET(RmtBufferVecUnpackProc(locNum, recvStream, rmtBufferTemp, UboeRmtBufType::BUFFER));
    if (rmtBufferTemp.size() != locNum) {
        HCCL_ERROR(
            "[%s::%s] buffer num mismatch, loc[%u] rmt[%zu]", channelClassName_, __func__, locNum,
            rmtBufferTemp.size());
        return HCCL_E_INTERNAL;
    }

    {
        std::lock_guard<std::mutex> lock(remote_.memCache.mutex);
        rmtBufferVec_.insert(
            rmtBufferVec_.end(), std::make_move_iterator(rmtBufferTemp.begin()),
            std::make_move_iterator(rmtBufferTemp.end()));
        local_.commonRes.bufferVec.insert(local_.commonRes.bufferVec.end(), bufferVecTemp.begin(), bufferVecTemp.end());
        remote_.memCache.Invalidate();
        bufferNum_ = static_cast<u32>(local_.commonRes.bufferVec.size());
    }
    return HCCL_SUCCESS;
}

std::vector<char> AicpuTsUboeUbRtpChannelHelper::GetMemUpdatePack()
{
    std::lock_guard<std::mutex> lock(remote_.memCache.mutex);
    u32 entryNum = 0;
    for (auto& it : rmtBufferVec_) {
        if (it != nullptr) {
            entryNum++;
        }
    }
    std::vector<char> result;
    if (entryNum == 0) {
        return result;
    }

    Hccl::BinaryStream binaryStream;
    binaryStream << entryNum;
    for (auto& it : rmtBufferVec_) {
        if (it == nullptr) {
            continue;
        }
        Hccl::RmaBufEntryLite entry;
        entry.addr = it->GetAddr();
        entry.size = it->GetSize();
        entry.tokenId = it->GetTokenId();
        entry.tokenValue = it->GetTokenValue();
        entry.notifyId = it->GetNotifyId();
        Hccl::PackRmaBufEntry(entry, binaryStream);
    }
    binaryStream.Dump(result);
    HCCL_INFO("[AicpuTsUboeUbRtpChannelHelper][%s] pack entryNum[%u], size[%zu].", __func__, entryNum, result.size());
    return result;
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
        "[%s::%s] Pack Buffer data[%p], Pack Buffer size[%zu].", channelClassName_, __func__, buffer.data(),
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

} // namespace hcomm
