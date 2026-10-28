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
#include "base_comm_node_accessors.h"
#include "user_remote_mem_getter.h"
#include "adapter_rts.h"

namespace hcomm {

constexpr uint16_t DEFAULT_LISTENING_PORT = 60001;
constexpr uint32_t TC_TEMP = 132;
constexpr uint32_t SL_TEMP = 4;
constexpr uint32_t RETRY_CNT_TEMP = 7;
constexpr uint32_t RETRY_TIME_TEMP = 20;

namespace {
    constexpr size_t AIV_ROCE_ENTITY_ALIGN_SIZE = 64;

    struct DeviceEntitySection {
        size_t offset{0};
        size_t size{0};
    };

    struct DeviceChannelEntityLayout {
        DeviceEntitySection entitySection{0, sizeof(ChannelEntity)};
        DeviceEntitySection localNotifySection;
        DeviceEntitySection remoteNotifySection;
        DeviceEntitySection localBufferSection;
        DeviceEntitySection remoteBufferSection;
        DeviceEntitySection sqContextSection;
        DeviceEntitySection cqContextSection;
        size_t slabSize{0};
    };

    size_t AlignUp(size_t value, size_t alignment) { return (value + alignment - 1) / alignment * alignment; }

    HcclResult AddDeviceEntitySection(
        size_t elemSize, uint32_t elemNum, size_t& offset, DeviceEntitySection& section, const char* sectionName)
    {
        section.offset = AlignUp(offset, AIV_ROCE_ENTITY_ALIGN_SIZE);
        if (elemNum == 0) {
            section.size = 0;
            offset = section.offset;
            return HCCL_SUCCESS;
        }
        CHK_PRT_RET(
            elemSize != 0 && elemNum > (SIZE_MAX / elemSize),
            HCCL_ERROR(
                "[AivRoceHostLocChannel::AddDeviceEntitySection] %s size overflow, elemSize[%zu], elemNum[%u]",
                sectionName, elemSize, elemNum),
            HCCL_E_PARA);
        section.size = elemSize * static_cast<size_t>(elemNum);
        CHK_PRT_RET(
            section.offset > (SIZE_MAX - section.size),
            HCCL_ERROR(
                "[AivRoceHostLocChannel::AddDeviceEntitySection] %s offset overflow, offset[%zu], size[%zu]",
                sectionName, section.offset, section.size),
            HCCL_E_PARA);
        offset = section.offset + section.size;
        return HCCL_SUCCESS;
    }

    void* GetSlabPtr(void* base, const DeviceEntitySection& section)
    {
        if (section.size == 0) {
            return nullptr;
        }
        return ReinterpretAs<void*>(ReinterpretAs<uintptr_t>(base) + section.offset);
    }

    template <typename T>
    HcclResult CopyArrayToSlab(
        void* slabBase, const T* hostArray, uint32_t arrayNum, const DeviceEntitySection& section, T** deviceArrayPtr,
        const char* arrayName)
    {
        CHK_PTR_NULL(deviceArrayPtr);
        if (arrayNum == 0 || hostArray == nullptr) {
            CHK_PRT_RET(
                arrayNum != 0,
                HCCL_ERROR(
                    "[AivRoceHostLocChannel::CopyArrayToSlab] %s hostArray is nullptr, num[%u]", arrayName, arrayNum),
                HCCL_E_PTR);
            *deviceArrayPtr = nullptr;
            return HCCL_SUCCESS;
        }
        CHK_PRT_RET(
            section.size != static_cast<size_t>(arrayNum) * sizeof(T),
            HCCL_ERROR(
                "[AivRoceHostLocChannel::CopyArrayToSlab] %s size mismatch, sectionSize[%zu], expect[%zu]", arrayName,
                section.size, static_cast<size_t>(arrayNum) * sizeof(T)),
            HCCL_E_PARA);
        void* sectionPtr = GetSlabPtr(slabBase, section);
        CHK_PTR_NULL(sectionPtr);
        Hccl::HrtMemcpy(
            sectionPtr, section.size, hostArray, section.size, Hccl::tagRtMemcpyKind::RT_MEMCPY_HOST_TO_DEVICE);
        *deviceArrayPtr = ReinterpretAs<T*>(sectionPtr);
        HCCL_INFO(
            "[AivRoceHostLocChannel::CopyArrayToSlab] %s: host[%p] -> dev[%p], num[%u], size[%zu]", arrayName,
            hostArray, sectionPtr, arrayNum, section.size);
        return HCCL_SUCCESS;
    }

    HcclResult BuildDeviceChannelEntityLayout(const ChannelEntity& hostChannel, DeviceChannelEntityLayout& layout)
    {
        layout.slabSize = AlignUp(sizeof(ChannelEntity), AIV_ROCE_ENTITY_ALIGN_SIZE);
        CHK_RET(AddDeviceEntitySection(
            sizeof(RegedNotifyEntity), hostChannel.localNotifyNum, layout.slabSize, layout.localNotifySection,
            "localNotifyAddr"));
        CHK_RET(AddDeviceEntitySection(
            sizeof(RegedNotifyEntity), hostChannel.remoteNotifyNum, layout.slabSize, layout.remoteNotifySection,
            "remoteNotifyAddr"));
        CHK_RET(AddDeviceEntitySection(
            sizeof(RegedBufferEntity), hostChannel.localBufferNum, layout.slabSize, layout.localBufferSection,
            "localBufferAddr"));
        CHK_RET(AddDeviceEntitySection(
            sizeof(RegedBufferEntity), hostChannel.remoteBufferNum, layout.slabSize, layout.remoteBufferSection,
            "remoteBufferAddr"));
        CHK_RET(AddDeviceEntitySection(
            sizeof(SqContext), hostChannel.sqNum, layout.slabSize, layout.sqContextSection, "sqContextAddr"));
        CHK_RET(AddDeviceEntitySection(
            sizeof(CqContext), hostChannel.cqNum, layout.slabSize, layout.cqContextSection, "cqContextAddr"));
        layout.slabSize = AlignUp(layout.slabSize, AIV_ROCE_ENTITY_ALIGN_SIZE);
        return HCCL_SUCCESS;
    }

    HcclResult AllocDeviceEntitySlab(size_t slabSize, AclDeviceSlabGuard& slabGuard, void*& slabPtr)
    {
        HcclResult ret = hrtMalloc(&slabPtr, slabSize);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS || slabPtr == nullptr,
            HCCL_ERROR(
                "[AivRoceHostLocChannel::%s] hrtMalloc slab failed, ret[%d], size[%zu]", __func__, ret, slabSize),
            HCCL_E_MEMORY);
        slabGuard.Reset(slabPtr, slabSize);
        return HCCL_SUCCESS;
    }

    HcclResult CopyChannelEntityArrayToSlab(
        void* slabPtr, const ChannelEntity& hostChannel, const DeviceChannelEntityLayout& layout,
        ChannelEntity& devChannel)
    {
        devChannel = hostChannel;
        CHK_RET(CopyArrayToSlab(
            slabPtr, hostChannel.localNotifyAddr, hostChannel.localNotifyNum, layout.localNotifySection,
            &devChannel.localNotifyAddr, "localNotifyAddr"));
        CHK_RET(CopyArrayToSlab(
            slabPtr, hostChannel.remoteNotifyAddr, hostChannel.remoteNotifyNum, layout.remoteNotifySection,
            &devChannel.remoteNotifyAddr, "remoteNotifyAddr"));
        CHK_RET(CopyArrayToSlab(
            slabPtr, hostChannel.localBufferAddr, hostChannel.localBufferNum, layout.localBufferSection,
            &devChannel.localBufferAddr, "localBufferAddr"));
        CHK_RET(CopyArrayToSlab(
            slabPtr, hostChannel.remoteBufferAddr, hostChannel.remoteBufferNum, layout.remoteBufferSection,
            &devChannel.remoteBufferAddr, "remoteBufferAddr"));
        CHK_RET(CopyArrayToSlab(
            slabPtr, hostChannel.sqContextAddr, hostChannel.sqNum, layout.sqContextSection, &devChannel.sqContextAddr,
            "sqContextAddr"));
        CHK_RET(CopyArrayToSlab(
            slabPtr, hostChannel.cqContextAddr, hostChannel.cqNum, layout.cqContextSection, &devChannel.cqContextAddr,
            "cqContextAddr"));
        return HCCL_SUCCESS;
    }

    HcclResult CopyChannelEntityToSlab(
        void* slabPtr, const DeviceChannelEntityLayout& layout, const ChannelEntity& devChannel, void*& entityDevPtr)
    {
        entityDevPtr = GetSlabPtr(slabPtr, layout.entitySection);
        CHK_PTR_NULL(entityDevPtr);
        Hccl::HrtMemcpy(
            entityDevPtr, sizeof(ChannelEntity), &devChannel, sizeof(ChannelEntity),
            Hccl::tagRtMemcpyKind::RT_MEMCPY_HOST_TO_DEVICE);
        return HCCL_SUCCESS;
    }
} // namespace

AivRoceHostLocChannel::AivRoceHostLocChannel(EndpointHandle endpointHandle, HcommChannelDesc channelDesc)
    : endpointHandle_(endpointHandle),
      channelDesc_(channelDesc),
      remoteEp_{}
{}

AivRoceHostLocChannel::~AivRoceHostLocChannel()
{
    FreeDeviceMemories();
    if (channelDesc_.socket == nullptr && socket_ != nullptr && socketConfig_ != nullptr) {
        GetSocketMgr(devicePhyId_).DestroySocket(*socketConfig_);
        socket_ = nullptr;
    }
}

HcclResult AivRoceHostLocChannel::ParseInputParam()
{
    // 1. 从 endpointHandle_，获得 localEp_ 和 rdmaHandle_
    CHK_PTR_NULL(endpointHandle_);
    HCCL_INFO(
        "[AivRoceHostLocChannel][%s] Start. endpointHandle[0x%llx]", __func__,
        ReinterpretAs<uint64_t>(endpointHandle_));
    Endpoint* localEpPtr = static_cast<Endpoint*>(endpointHandle_);
    localEp_ = localEpPtr->GetEndpointDesc();
    rdmaHandle_ = localEpPtr->GetRdmaHandle();
    CHK_PTR_NULL(rdmaHandle_);

    // 2. 从 channelDesc_，获得 remoteEp_ 和 socket_
    remoteEp_ = channelDesc_.remoteEndpoint;
    socket_ = static_cast<Hccl::Socket*>(channelDesc_.socket);

    return HCCL_SUCCESS;
}

HcclResult AivRoceHostLocChannel::StartListen()
{
    uint16_t port = channelDesc_.port;
    HCCL_INFO(
        "[AivRoceHostLocChannel::%s] Start. EndpointHandle[0x%llx], port[%u]", __func__,
        ReinterpretAs<uint64_t>(endpointHandle_), port);
    if (port == 0) {
        port = DEFAULT_LISTENING_PORT;
        HCCL_INFO("[AivRoceHostLocChannel::%s] channelDesc port is 0, use default port [%u]", __func__, port);
    }
    CHK_RET(static_cast<HcclResult>(HcommEndpointStartListen(endpointHandle_, port, nullptr)));
    HCCL_INFO("[AivRoceHostLocChannel::%s] SUCCESS. port[%u].", __func__, port);
    return HCCL_SUCCESS;
}

HcclResult AivRoceHostLocChannel::BuildSocket()
{
    if (socket_ != nullptr) {
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[AivRoceHostLocChannel::%s] socket ptr is NULL, rebuild Socket", __func__);

    Hccl::LinkData linkData = BuildDefaultLinkData();
    CHK_RET(EndpointDescPairToLinkData(localEp_, remoteEp_, linkData));
    HCCL_INFO("[AivRoceHostLocChannel::%s] built linkData: %s", __func__, linkData.Describe().c_str());
    uint16_t port = channelDesc_.port;
    if (port == 0) {
        port = DEFAULT_LISTENING_PORT;
        HCCL_INFO("[AivRoceHostLocChannel::%s] channelDesc port is 0, use default port [%u]", __func__, port);
    }
    std::string socketTag
        = (channelDesc_.channelName != nullptr) ? std::string(channelDesc_.channelName) : "AUTOMATIC_SOCKET_TAG";
    Hccl::SocketConfig socketConfig
        = (channelDesc_.role != HCOMM_SOCKET_ROLE_RESERVED) ?
              Hccl::SocketConfig(linkData, port, socketTag, channelDesc_.role == HCOMM_SOCKET_ROLE_SERVER) :
              Hccl::SocketConfig(linkData, port, socketTag);
    SaveSocketConfig(socketConfig);
    CHK_RET(GetSocketMgr(devicePhyId_).GetSocket(*socketConfig_, socket_));
    HCCL_INFO("[AivRoceHostLocChannel::%s] SUCCESS. port[%u].", __func__, port);
    return HCCL_SUCCESS;
}

HcclResult AivRoceHostLocChannel::BuildConnection()
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
        "[AivRoceHostLocChannel::BuildConnection] QpInfo: serviceLevel[%u], trafficClass[%u], retryCnt[%u], "
        "retryInterval[%u].",
        qpInfo.serviceLevel, qpInfo.trafficClass, qpInfo.retryCnt, qpInfo.retryInterval);
    connections_.emplace_back(std::move(conn));
    connNum_ = connections_.size();
    HCCL_INFO("[AivRoceHostLocChannel::%s] connection num [%u]", __func__, connNum_);
    return HCCL_SUCCESS;
}

HcclResult AivRoceHostLocChannel::BuildBuffer()
{
    if (channelDesc_.exchangeAllMems) {
        // Get memHandles from endpoint
        HCCL_INFO("[AivRoceHostLocChannel][%s] exchangeAllMems == True. Get memHandles from endpoint.", __func__);
        std::shared_ptr<Hccl::LocalRdmaRmaBuffer>* memHandles = nullptr;
        uint32_t memHandleNum = 0;
        CHK_RET(static_cast<HcclResult>(
            HcommMemGetAllMemHandles(endpointHandle_, ReinterpretAs<void**>(&memHandles), &memHandleNum)));
        HCCL_INFO("[AivRoceHostLocChannel][%s] Got memHandleNum[%u].", __func__, memHandleNum);
        for (uint32_t i = 0; i < memHandleNum; ++i) {
            std::shared_ptr<Hccl::LocalRdmaRmaBuffer>& localRdmaBuffer = memHandles[i];
            CHK_SMART_PTR_NULL(localRdmaBuffer);
            Hccl::Buffer* buf = localRdmaBuffer->GetBuf();
            CHK_PTR_NULL(buf);
            HCCL_INFO(
                "[AivRoceHostLocChannel][%s] Got memHandle No.%u: addr[0x%llx], size[0x%llx], memType[%d], "
                "memInfo[%s].",
                __func__, i, static_cast<unsigned long long>(localRdmaBuffer->GetAddr()),
                static_cast<unsigned long long>(localRdmaBuffer->GetSize()), static_cast<int>(buf->GetMemType()),
                buf->GetMemInfo().c_str());
            localRmaBuffers_.emplace_back(localRdmaBuffer.get());
        }
    } else {
        // 从 channelDesc 的 memHandle，获得 localRmaBuffers_
        HCCL_INFO("[AivRoceHostLocChannel][%s] exchangeAllMems == false. Get memHandles from channelDesc.", __func__);
        CHK_PTR_NULL(channelDesc_.memHandles);
        for (uint32_t i = 0; i < channelDesc_.memHandleNum; ++i) {
            CHK_PTR_NULL(channelDesc_.memHandles[i]);
            auto* localRdmaBuffer = ReinterpretAs<Hccl::LocalRdmaRmaBuffer*>(channelDesc_.memHandles[i]);
            HCCL_INFO(
                "[AivRoceHostLocChannel][%s] Got memHandle No.%u: addr[0x%llx], size[0x%llx], memType[%d], "
                "memInfo[%s].",
                __func__, i, static_cast<unsigned long long>(localRdmaBuffer->GetAddr()),
                static_cast<unsigned long long>(localRdmaBuffer->GetSize()),
                static_cast<int>(localRdmaBuffer->GetBuf()->GetMemType()),
                localRdmaBuffer->GetBuf()->GetMemInfo().c_str());
            localRmaBuffers_.emplace_back(localRdmaBuffer);
        }
    }
    bufferNum_ = localRmaBuffers_.size();
    HCCL_INFO("[AivRoceHostLocChannel::%s] buffer num [%u]", __func__, bufferNum_);
    return HCCL_SUCCESS;
}

HcclResult AivRoceHostLocChannel::Init()
{
    s32 userDevId = Hccl::HrtGetDevice();
    devicePhyId_ = Hccl::HrtGetDevicePhyIdByUserDevId(static_cast<u32>(userDevId));

    CHK_RET(ParseInputParam());
    if (channelDesc_.exchangeAllMems && channelDesc_.role != HCOMM_SOCKET_ROLE_CLIENT) {
        CHK_RET(StartListen());
    }
    CHK_RET(BuildSocket());
    CHK_RET(BuildConnection());
    CHK_RET(BuildBuffer());
    return HCCL_SUCCESS;
}

// 当前AICPU和框架没有改为返回错误码形式，所有暂时使用该方法转换
ChannelStatus AivRoceHostLocChannel::GetStatus()
{
    ChannelStatus status;
    HcclResult ret = GetStatus(status);
    if (ret != HCCL_SUCCESS && ret != HCCL_E_AGAIN) {
        HCCL_ERROR("[AivRoceHostLocChannel::GetStatus] get status exception occurred, HcclResult=[%d]", ret);
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
    CHK_PTR_NULL(socket_);
    Hccl::SocketStatus socketStatus = socket_->GetStatus(); // socket状态机
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
        !socket_->Send(static_cast<void*>(&sendSize), sizeof(sendSize)),
        HCCL_ERROR("[AivRoceHostLocChannel::%s] Send sendSize failed", __func__), HCCL_E_NETWORK);
    HCCL_INFO(
        "[AivRoceHostLocChannel::%s] Send size[%llu] of data success. [%llu] bytes sent.", __func__, sendSize,
        sizeof(sendSize));

    // 同步接收数据包尺寸
    CHK_PRT_RET(
        !socket_->Recv(static_cast<void*>(&recvSize), sizeof(recvSize)),
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
        !socket_->Send(static_cast<void*>(sendData.data()), sendSize),
        HCCL_ERROR("[AivRoceHostLocChannel::%s] Send exchange data failed", __func__), HCCL_E_NETWORK);
    HCCL_INFO("[AivRoceHostLocChannel::%s] Send Exchange Data success. [%llu] bytes sent.", __func__, sendSize);

    // 同步接收数据
    HCCL_INFO("[AivRoceHostLocChannel::%s] Start to Receive Exchange Data", __func__);
    recvData.resize(recvSize);
    CHK_PRT_RET(
        !socket_->Recv(static_cast<void*>(recvData.data()), recvSize),
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
            rmtRmaBuffers_[pos] = std::make_unique<Hccl::RemoteRdmaRmaBuffer>(rdmaHandle_, dto),
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

HcclResult AivRoceHostLocChannel::GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos)
{
    std::lock_guard<std::mutex> lock(remoteMemsMutex_);
    Hccl::RemoteMemCtx<std::unique_ptr<Hccl::RemoteRdmaRmaBuffer>> remoteMemCtx{
        cacheValid_, rmtRmaBuffers_, remoteUserMems_, memInfoCopies_, memInfoPointers_, remoteMem, memInfos, memNum};
    CHK_RET(Hccl::GetRemoteUserMems(remoteMemCtx));
    return HCCL_SUCCESS;
}

HcclResult AivRoceHostLocChannel::Clean()
{
    ReleaseDeviceEntitySlab();
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
    msg += Hccl::StringFormat(", rdmaHandle:%p, %s, ", rdmaHandle_, channelStatus_.Describe().c_str());
    if (socket_ != nullptr) {
        msg += socket_->Describe();
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
        SqContext sqContext;
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
        CqContext cqContext;
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

HcclResult AivRoceHostLocChannel::BuildAndGetDevChannelEntity(uint64_t* devChannelEntityPtr)
{
    CHK_PTR_NULL(devChannelEntityPtr);

    if (devChannelEntitySlab_ != nullptr) {
        *devChannelEntityPtr = ReinterpretAs<uint64_t>(devChannelEntitySlab_);
        HCCL_INFO(
            "[AivRoceHostLocChannel::%s] already built, return cached devPtr=0x%lx", __func__, *devChannelEntityPtr);
        return HCCL_SUCCESS;
    }

    ChannelEntity hostEntity{};
    std::vector<RegedBufferEntity> locBufList;
    std::vector<RegedBufferEntity> rmtBufList;
    std::vector<SqContext> sqList;
    std::vector<CqContext> cqList;
    CHK_RET(BuildHostEntity(hostEntity, locBufList, rmtBufList, sqList, cqList));

    DeviceChannelEntityLayout layout;
    CHK_RET(BuildDeviceChannelEntityLayout(hostEntity, layout));
    void* slabPtr = nullptr;
    AclDeviceSlabGuard slabGuard;
    CHK_RET(AllocDeviceEntitySlab(layout.slabSize, slabGuard, slabPtr));

    ChannelEntity devEntity;
    CHK_RET(CopyChannelEntityArrayToSlab(slabPtr, hostEntity, layout, devEntity));
    void* entityDevPtr = nullptr;
    CHK_RET(CopyChannelEntityToSlab(slabPtr, layout, devEntity, entityDevPtr));

    ReleaseDeviceEntitySlab();
    devChannelEntitySlab_ = slabGuard.Release();
    devChannelEntitySlabSize_ = layout.slabSize;

    *devChannelEntityPtr = ReinterpretAs<uint64_t>(entityDevPtr);
    HCCL_INFO(
        "[AivRoceHostLocChannel::%s] Success, devPtr=0x%lx, slabPtr=%p, slabSize=%zu", __func__, *devChannelEntityPtr,
        devChannelEntitySlab_, devChannelEntitySlabSize_);
    return HCCL_SUCCESS;
}

HcclResult AivRoceHostLocChannel::PreAllocDevChannelEntity(uint64_t* devChannelEntityPtr)
{
    CHK_PTR_NULL(devChannelEntityPtr);

    if (devChannelEntitySlab_ != nullptr) {
        *devChannelEntityPtr = ReinterpretAs<uint64_t>(devChannelEntitySlab_);
        HCCL_INFO(
            "[AivRoceHostLocChannel::%s] already built, return cached devPtr=0x%lx", __func__, *devChannelEntityPtr);
        return HCCL_SUCCESS;
    }

    ChannelEntity tmp{};
    tmp.localNotifyNum = 0;
    tmp.remoteNotifyNum = 0;
    tmp.localBufferNum = bufferNum_;
    tmp.remoteBufferNum = bufferNum_;
    tmp.sqNum = connNum_;
    tmp.cqNum = connNum_;

    DeviceChannelEntityLayout layout;
    CHK_RET(BuildDeviceChannelEntityLayout(tmp, layout));

    void* slabPtr = nullptr;
    AclDeviceSlabGuard slabGuard;
    CHK_RET(AllocDeviceEntitySlab(layout.slabSize, slabGuard, slabPtr));

    devChannelEntitySlab_ = slabGuard.Release();
    devChannelEntitySlabSize_ = layout.slabSize;
    *devChannelEntityPtr = ReinterpretAs<uint64_t>(devChannelEntitySlab_);

    HCCL_INFO(
        "[AivRoceHostLocChannel::%s] pre-alloc success, slabPtr=%p, slabSize=%zu", __func__, devChannelEntitySlab_,
        devChannelEntitySlabSize_);
    return HCCL_SUCCESS;
}

HcclResult AivRoceHostLocChannel::FillDevChannelEntity()
{
    if (devChannelEntitySlab_ == nullptr) {
        HCCL_ERROR("[AivRoceHostLocChannel::%s] devChannelEntitySlab_ is nullptr, not pre-allocated.", __func__);
        return HCCL_E_INTERNAL;
    }
    if (channelStatus_ != ChannelStatus::READY) {
        HCCL_ERROR(
            "[AivRoceHostLocChannel::%s] channel status[%d] is not ready[%d], please check.", __func__, channelStatus_,
            ChannelStatus::READY);
        return HCCL_E_INTERNAL;
    }

    ChannelEntity hostEntity{};
    std::vector<RegedBufferEntity> locBufList;
    std::vector<RegedBufferEntity> rmtBufList;
    std::vector<SqContext> sqList;
    std::vector<CqContext> cqList;
    CHK_RET(BuildHostEntity(hostEntity, locBufList, rmtBufList, sqList, cqList));

    DeviceChannelEntityLayout layout;
    CHK_RET(BuildDeviceChannelEntityLayout(hostEntity, layout));
    if (layout.slabSize > devChannelEntitySlabSize_) {
        HCCL_ERROR(
            "[AivRoceHostLocChannel::%s] slabSize[%zu] > preAllocSize[%zu]", __func__, layout.slabSize,
            devChannelEntitySlabSize_);
        return HCCL_E_INTERNAL;
    }

    ChannelEntity devEntity;
    CHK_RET(CopyChannelEntityArrayToSlab(devChannelEntitySlab_, hostEntity, layout, devEntity));
    void* entityDevPtr = nullptr;
    CHK_RET(CopyChannelEntityToSlab(devChannelEntitySlab_, layout, devEntity, entityDevPtr));

    HCCL_INFO("[AivRoceHostLocChannel::%s] fill success, devPtr=%p", __func__, entityDevPtr);
    return HCCL_SUCCESS;
}

void AivRoceHostLocChannel::ReleaseDeviceEntitySlab()
{
    if (devChannelEntitySlab_ != nullptr) {
        HcclResult ret = hrtFree(devChannelEntitySlab_);
        if (ret != HCCL_SUCCESS) {
            HCCL_WARNING(
                "[AivRoceHostLocChannel::%s] hrtFree devChannelEntitySlab is unsuccessful, ptr[%p], size[%zu], ret[%d]",
                __func__, devChannelEntitySlab_, devChannelEntitySlabSize_, ret);
        }
        devChannelEntitySlab_ = nullptr;
        devChannelEntitySlabSize_ = 0;
    }
}

void AivRoceHostLocChannel::FreeDeviceMemories() { ReleaseDeviceEntitySlab(); }

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

HcommChannelKind AivRoceHostLocChannel::GetChannelKind() const { return HcommChannelKind::AIV_ROCE; }

} // namespace hcomm
