/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_urma_channel.h"

#include "hcomm_c_adpt.h"

#include "orion_adpt_utils.h"

#include "exception_handler.h"
#include "common/mem_type_convert_pub.h"

#include "config_log.h"
#include "config_plf_log_v2.h"

// 暂时引入orion
#include "local_ub_rma_buffer.h"

namespace hcomm {
using Hccl::PLF_CHANNEL;
constexpr uint32_t CCU_CHANNEL_RESOURCE_CAPACITY = 16U; // channel XN/CKE 资源槽数

HcclResult BuildBufferInfos(
    HcommMemHandle* memHandles, uint32_t memHandleNum, std::vector<CcuTransport::CclBufferInfo>& bufferInfos)
{
    for (uint32_t i = 0; i < memHandleNum; ++i) {
        auto localRmaBuffer = reinterpret_cast<Hccl::LocalUbRmaBuffer*>(memHandles[i]);
        CHK_PTR_NULL(localRmaBuffer);
        auto buf = localRmaBuffer->GetBuf();
        CHK_PTR_NULL(buf);

        std::array<char, HCCL_RES_TAG_MAX_LEN> memInfo{};
        std::string tag = buf->GetMemInfo();
        if (UNLIKELY(tag.size() >= HCCL_RES_TAG_MAX_LEN)) {
            HCCL_ERROR("[CcuUrmaChannel::%s] tagSize exceeds limit[%u]", __func__, HCCL_RES_TAG_MAX_LEN);
            return HCCL_E_PARA;
        }
        CHK_SAFETY_FUNC_RET(memcpy_s(memInfo.data(), memInfo.size(), tag.c_str(), tag.size()));
        bufferInfos.emplace_back(
            localRmaBuffer->GetAddr(), static_cast<uint32_t>(localRmaBuffer->GetSize()), localRmaBuffer->GetTokenId(),
            localRmaBuffer->GetTokenValue(), hccl::ConvertHcclToCommMemType(buf->GetMemType()), memInfo);
    }
    return HCCL_SUCCESS;
}

static HcclResult CreateCcuTransport(
    UrmaEndpoint* ccuEndpoint, const Hccl::LinkData& linkData, Hccl::Socket* socket, HcommMemHandle* memHandles,
    uint32_t memHandleNum, uint32_t qos, uint32_t sqSize, std::unique_ptr<CcuTransport>& impl,
    const char* channelClassName)
{
    HCCL_INFO("[%s::%s] begin, sqSize[%u]", channelClassName, __func__, sqSize);
    // 当前ccu channel不支持按需申请cke
    CHK_PTR_NULL(ccuEndpoint);
    CHK_PTR_NULL(socket);
    CHK_PTR_NULL(memHandles);

    auto ret = HcclResult::HCCL_SUCCESS;
    auto* channelCtxPool = ccuEndpoint->GetCcuChannelCtxPool();
    CHK_PTR_NULL(channelCtxPool);
    // 申请ccu channel ctx， jetty ctx，wqebb，可能资源不足，需要回退
    ret = channelCtxPool->PrepareCreate({linkData}, sqSize);
    if (ret == HCCL_E_UNAVAIL) {
        HCCL_WARNING(
            "[%s::%s] prepare ccu channel ctx unsuccessfully, "
            "ccu resources unavailable.",
            channelClassName, __func__);
        return ret;
    }
    CHK_RET(ret);

    CcuChannelCtxPool::CcuChannelCtx channelCtx{};
    CHK_RET(channelCtxPool->GetChannelCtx(linkData, channelCtx));
    const auto& channelInfo = channelCtx.first;
    const auto& ccuJettys = channelCtx.second;

    const auto& locAddr_ = linkData.GetLocalAddr();
    const auto& rmtAddr_ = linkData.GetRemoteAddr();

    CommAddr locAddr{}, rmtAddr{};
    CHK_RET(IpAddressToCommAddr(locAddr_, locAddr));
    CHK_RET(IpAddressToCommAddr(rmtAddr_, rmtAddr));

    CcuTransport::CcuConnectionType type_ = linkData.GetLinkProtocol() == Hccl::LinkProtocol::UB_CTP ?
                                                CcuTransport::CcuConnectionType::UB_CTP :
                                                CcuTransport::CcuConnectionType::UBC_TP;

    CcuTransport::CcuConnectionInfo connectionInfo{type_, locAddr, rmtAddr, channelInfo, ccuJettys, qos};

    std::vector<CcuTransport::CclBufferInfo> bufferInfos{};
    CHK_RET(BuildBufferInfos(memHandles, memHandleNum, bufferInfos));

    // 调用底层的创建函数 (CcuCreateTransport 通常是全局函数或静态函数)
    // 申请 xn cke可能失败，需要回退
    ret = CcuCreateTransport(socket, connectionInfo, bufferInfos, impl);
    if (ret == HCCL_E_UNAVAIL) {
        HCCL_WARNING("[%s::%s] is unsuccessful, ccu resources unavailable.", channelClassName, __func__);
        return ret;
    }
    CHK_RET(ret);

    PLF_CONFIG_INFO(
        PLF_CHANNEL, "[%s::%s] transport created, sqSize[%u], linkData[%s].", channelClassName, __func__, sqSize,
        linkData.Describe().c_str());
    return HCCL_SUCCESS;
}

static HcclResult
CheckEndpointDesc(const EndpointDesc& locDesc, const EndpointDesc& rmtDesc, const char* channelClassName)
{
    if (locDesc.protocol != rmtDesc.protocol) {
        HCCL_ERROR(
            "[%s::%s] failed, endpoints protocols are not same, "
            "loc[%d] rmt[%d].",
            channelClassName, __func__, locDesc.protocol, rmtDesc.protocol);
        return HcclResult::HCCL_E_PARA;
    }

    if (locDesc.protocol != COMM_PROTOCOL_UB_CTP && locDesc.protocol != COMM_PROTOCOL_UBC_TP) {
        HCCL_ERROR(
            "[%s::%s] failed, protocol[%d] are not supported in ccu.", channelClassName, __func__, locDesc.protocol);
        return HcclResult::HCCL_E_PARA;
    }

    return HcclResult::HCCL_SUCCESS;
}

// 状态机终态 -> 通道状态映射（含 LOC/RMT 资源不足优先级）
static ChannelStatus
MapTransStatusToChannelStatus(CcuTransport::TransStatus status, const CcuTransport& impl, const char* channelClassName)
{
    switch (status) {
        case CcuTransport::TransStatus::READY:
            return ChannelStatus::READY;
        case CcuTransport::TransStatus::SOCKET_TIMEOUT:
            HCCL_ERROR("[%s::%s] error status[%s].", channelClassName, __func__, status.Describe().c_str());
            return ChannelStatus::SOCKET_TIMEOUT;
        case CcuTransport::TransStatus::CONNECT_FAILED:
            // 资源不足终态映射(LOC 优先): 本端不足报 LOC, 否则若对端不足报 RMT, 否则普通 FAILED
            if (impl.IsLocResUnavailable()) {
                return ChannelStatus::RES_LOC_UNAVAIL;
            }
            if (impl.IsRmtResUnavailable()) {
                return ChannelStatus::RES_RMT_UNAVAIL;
            }
            HCCL_ERROR("[%s::%s] error status[%s].", channelClassName, __func__, status.Describe().c_str());
            return ChannelStatus::FAILED;
        default:
            return ChannelStatus::INIT;
    }
}

CcuUrmaChannel::CcuUrmaChannel(
    const EndpointHandle locEndpointHandle, const HcommChannelDesc& channelDesc, CommEngine engine)
    : Channel(locEndpointHandle, channelDesc, engine, HcommChannelKind::CCU_URMA, "CcuUrmaChannel"),
      locEndpointHandle_(locEndpointHandle)
{}

CcuUrmaChannel::~CcuUrmaChannel()
{
    // 先释放 transport/connection（CKE/XN、TP、远端 jetty QP unimport），
    // 再向 endpoint 的 CcuChannelCtxPool 归还 channel ctx / jetty ctx / wqeBB。
    // 顺序必须如此：CcuConnection 持有 batch 内 CcuJetty 裸指针。
    impl_.reset();
    if (ccuEndpoint_ != nullptr) {
        Hccl::LinkData linkData = BuildDefaultLinkData();
        HcclResult keyRet
            = EndpointDescPairToLinkData(ccuEndpoint_->GetEndpointDesc(), channelDesc_.remoteEndpoint, linkData);
        if (keyRet != HCCL_SUCCESS) {
            HCCL_ERROR("[%s::%s] failed to build link data for release, ret[%d].", channelClassName_, __func__, keyRet);
            return;
        }
        auto* pool = ccuEndpoint_->GetCcuChannelCtxPool();
        if (pool != nullptr) {
            HcclResult releaseRet = pool->ReleaseChannel(linkData);
            if (releaseRet != HCCL_SUCCESS) {
                HCCL_WARNING(
                    "[%s::%s] release channel to pool unsuccessfully, ret[%d].", channelClassName_, __func__,
                    releaseRet);
            }
        }
    }
}

HcclResult CcuUrmaChannel::ParseInputParam()
{
    CHK_RET(Channel::ParseInputParam());
    return HCCL_SUCCESS;
}

HcclResult CcuUrmaChannel::StartListen() { return HCCL_SUCCESS; }

HcclResult CcuUrmaChannel::BuildSocket()
{
    CHK_RET(Channel::BuildSocket());
    CHK_PTR_NULL(channelDesc_.socket);
    local_.socket = static_cast<Hccl::Socket*>(channelDesc_.socket);
    // 当前socket在外部统一触发connect，建议之后改为异步建链流程内触发

    CHK_PTR_NULL(locEndpointHandle_);
    void* endpoint{nullptr};
    CHK_RET(static_cast<HcclResult>(HcommEndpointGet(locEndpointHandle_, &endpoint)));
    ccuEndpoint_ = dynamic_cast<UrmaEndpoint*>(static_cast<Endpoint*>(endpoint));
    CHK_PTR_NULL(ccuEndpoint_);
    return HCCL_SUCCESS;
}

HcclResult CcuUrmaChannel::BuildConnection()
{
    CHK_RET(Channel::BuildConnection());
    const auto& locEndpointDesc = ccuEndpoint_->GetEndpointDesc();

    CHK_RET(CheckEndpointDesc(locEndpointDesc, channelDesc_.remoteEndpoint, channelClassName_));

    Hccl::LinkData linkData = BuildDefaultLinkData();
    CHK_RET(EndpointDescPairToLinkData(locEndpointDesc, channelDesc_.remoteEndpoint, linkData));
    return HCCL_SUCCESS;
}

HcclResult CcuUrmaChannel::BuildNotify()
{
    CHK_RET(Channel::BuildNotify());
    // 当前建链不支持资源扩容，CCU资源默认固定为8
    HCCL_WARNING("[%s::%s] now only support notify num is 8.", channelClassName_, __func__);
    return HCCL_SUCCESS;
}

HcclResult CcuUrmaChannel::BuildBuffer()
{
    CHK_RET(Channel::BuildBuffer());
    if (channelDesc_.memHandleNum == 0) {
        HCCL_ERROR(
            "[%s::%s] failed, unsupported memHandleNum[%u].", channelClassName_, __func__, channelDesc_.memHandleNum);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    CHK_PTR_NULL(channelDesc_.memHandles);

    memHandles_.assign(channelDesc_.memHandles, channelDesc_.memHandles + channelDesc_.memHandleNum);
    HCCL_WARNING("[%s::%s] now only support to exchange hccl buffer.", channelClassName_, __func__);
    return HCCL_SUCCESS;
}

HcclResult CcuUrmaChannel::GetNotifyNum(uint32_t* notifyNum) const
{
    CHK_PTR_NULL(impl_);
    CHK_RET(impl_->GetCkeNum(*notifyNum));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuUrmaChannel::GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos)
{
    CHK_PTR_NULL(impl_);
    return impl_->GetRemoteMems(memNum, remoteMem, memInfos);
}

ChannelStatus CcuUrmaChannel::GetStatus()
{
    std::lock_guard<std::mutex> lock(statusMtx_);
    return GetStatusLocked();
}

ChannelStatus CcuUrmaChannel::GetStatusLocked()
{
    if (ccuEndpoint_ == nullptr || local_.socket == nullptr || memHandles_.empty()) {
        HCCL_ERROR(
            "[%s::%s] endpoint[%p], socket[%p], memHandleNum[%zu], Init() may not have succeeded.", channelClassName_,
            __func__, ccuEndpoint_, local_.socket, memHandles_.size());
        channelStatus_ = ChannelStatus::FAILED;
        return channelStatus_;
    }
    // INIT 态内: locResStatus_==RES_UNKNOWN(或 impl_ 为空) 则构造 transport + 申请资源(仅一次)
    if (channelStatus_ == ChannelStatus::INIT) {
        if (!impl_ || impl_->GetLocResStatus() == CcuTransport::CcuResStatus::RES_UNKNOWN) {
            channelStatus_ = TryPrepareAndConstruct();
            return channelStatus_;
        }
    }

    if (!impl_) {
        HCCL_ERROR("[%s::%s] failed, impl is nullptr.", channelClassName_, __func__);
        return ChannelStatus::FAILED;
    }

    CcuTransport::TransStatus status = impl_->GetStatus();
    ChannelStatus out = MapTransStatusToChannelStatus(status, *impl_, channelClassName_);
    channelStatus_ = out;

    if (isFirstPrintChannelInfo_ && out == ChannelStatus::READY) {
        std::string channelInfo = "create channel info:channel handle[";
        channelInfo.append(std::to_string(reinterpret_cast<uint64_t>(this)));
        channelInfo.append("] ");
        HcclResult ret = impl_->Describe(channelInfo);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[%s::%s] Describe channel info failed, ret=%d", channelClassName_, __func__, ret);
            out = ChannelStatus::FAILED;
            channelStatus_ = out;
        } else {
            channelInfo.append(" TA[RM]"); // 目前TA只支持RM
            HCCL_CONFIG_DEBUG(hccl::HCCL_RES, "%s", channelInfo.c_str());
        }
        isFirstPrintChannelInfo_ = false;
    }
    return channelStatus_;
}

// 获取设备逻辑 ID；impl 未建立时返回 -1。
int32_t CcuUrmaChannel::GetDevLogicId() const
{
    if (!impl_) {
        return -1;
    }

    return impl_->GetDevLogicId();
}

// 获取 CCU 版本，透传 CcuTransport 查询结果。
HcclResult CcuUrmaChannel::GetCcuVersion(uint32_t& ccuVersion) const
{
    CHK_PTR_NULL(impl_);
    return impl_->GetCcuVersion(ccuVersion);
}

HcclResult CcuUrmaChannel::UpdateMemInfo(HcommMemHandle* memHandles, uint32_t memHandleNum)
{
    std::lock_guard<std::mutex> lock(statusMtx_);
    CHK_PTR_NULL(impl_);
    std::vector<CcuTransport::CclBufferInfo> bufferVecTemp{};
    CHK_RET(BuildBufferInfos(memHandles, memHandleNum, bufferVecTemp));
    return impl_->UpdateMemInfo(bufferVecTemp);
}
HcclResult CcuUrmaChannel::Clean()
{
    std::lock_guard<std::mutex> lock(statusMtx_);
    CHK_PTR_NULL(impl_);
    PLF_CONFIG_INFO(PLF_CHANNEL, "[%s::%s] clean channel resource.", channelClassName_, __func__);
    impl_->Clean();
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuUrmaChannel::Resume() { return HCCL_SUCCESS; }

uint32_t CcuUrmaChannel::GetDieId() const
{
    if (!impl_) {
        return UINT32_MAX;
    }

    return impl_->GetDieId();
}

uint32_t CcuUrmaChannel::GetChannelId() const
{
    if (!impl_) {
        return UINT32_MAX;
    }
    return impl_->GetChannelId();
}

HcclResult CcuUrmaChannel::GetRmtSignalAddrByIndex(uint32_t index, uint64_t& rmtCkeAddr) const
{
    CHK_PTR_NULL(impl_);
    CHK_RET(impl_->GetRmtSignalAddrByIndex(index, rmtCkeAddr));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuUrmaChannel::GetRmtCcuBufferTokenInfo(uint32_t& rmtTokenId, uint32_t& rmtTokenValue) const
{
    CHK_PTR_NULL(impl_);
    CHK_RET(impl_->GetRmtCcuBufferTokenInfo(rmtTokenId, rmtTokenValue));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuUrmaChannel::CcuGetRmtMemToken(uint64_t srcVa, uint64_t& tokenInfo)
{
    return impl_->CcuGetRmtMemToken(srcVa, tokenInfo);
}

HcclResult CcuUrmaChannel::GetRmtVarAddrByIndex(uint32_t index, uint64_t& rmtXnAddr) const
{
    CHK_PTR_NULL(impl_);
    CHK_RET(impl_->GetRmtVarAddrByIndex(index, rmtXnAddr));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuUrmaChannel::GetLocCkeByIndex(const uint32_t index, uint32_t& locCkeId) const
{
    CHK_PTR_NULL(impl_);
    CHK_RET(impl_->GetLocCkeByIndex(index, locCkeId));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuUrmaChannel::GetLocXnByIndex(const uint32_t index, uint32_t& locXnId) const
{
    CHK_PTR_NULL(impl_);
    CHK_RET(impl_->GetLocXnByIndex(index, locXnId));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuUrmaChannel::GetRmtCkeByIndex(const uint32_t index, uint32_t& rmtCkeId) const
{
    CHK_PTR_NULL(impl_);
    CHK_RET(impl_->GetRmtCkeByIndex(index, rmtCkeId));
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuUrmaChannel::GetRmtXnByIndex(const uint32_t index, uint32_t& rmtXnId) const
{
    CHK_PTR_NULL(impl_);
    CHK_RET(impl_->GetRmtXnByIndex(index, rmtXnId));
    return HcclResult::HCCL_SUCCESS;
}

// 获取对端 CCU 资源空间信息，透传 CcuTransport 查询结果。
HcclResult CcuUrmaChannel::GetRmtCcuBufInfo(uint64_t& rmtCcuBufAddr, uint64_t& rmtCcuBufSize) const
{
    CHK_PTR_NULL(impl_);
    CHK_RET(impl_->GetRmtCcuBufInfo(rmtCcuBufAddr, rmtCcuBufSize));
    return HcclResult::HCCL_SUCCESS;
}

// 获取本端与对端的 CKE、XN 资源数量。
HcclResult CcuUrmaChannel::GetResourceNums(
    uint32_t& localCkeNum, uint32_t& localXnNum, uint32_t& remoteCkeNum, uint32_t& remoteXnNum) const
{
    CHK_PTR_NULL(impl_);
    return impl_->GetResourceNums(localCkeNum, localXnNum, remoteCkeNum, remoteXnNum);
}

HcclResult CcuUrmaChannel::GetRmtBuffer(uint64_t& addr, uint32_t& size, uint32_t& tokenId, uint32_t& tokenValue) const
{
    CHK_PTR_NULL(impl_);
    CcuTransport::CclBufferInfo bufInfo{};
    constexpr uint32_t bufNum = 0; // 当前不支持
    CHK_RET(impl_->GetRmtBuffer(bufInfo, bufNum));

    addr = bufInfo.addr;
    size = bufInfo.size;
    tokenId = bufInfo.tokenId;
    tokenValue = bufInfo.tokenValue;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuUrmaChannel::CheckSnapshotResourceNums(
    uint32_t localCkeNum, uint32_t localXnNum, uint32_t remoteCkeNum, uint32_t remoteXnNum) const
{
    CHK_PRT_RET(
        localCkeNum > CCU_CHANNEL_RESOURCE_CAPACITY || localXnNum > CCU_CHANNEL_RESOURCE_CAPACITY
            || remoteCkeNum > CCU_CHANNEL_RESOURCE_CAPACITY || remoteXnNum > CCU_CHANNEL_RESOURCE_CAPACITY,
        HCCL_ERROR(
            "[CcuUrmaChannel][%s] invalid CCU resource counts: localCke[%u], localXn[%u], remoteCke[%u], "
            "remoteXn[%u], capacity[%u].",
            __func__, localCkeNum, localXnNum, remoteCkeNum, remoteXnNum, CCU_CHANNEL_RESOURCE_CAPACITY),
        HcclResult::HCCL_E_INTERNAL);
    return HcclResult::HCCL_SUCCESS;
}

void CcuUrmaChannel::FillSnapshotBaseFields(
    HcommCcuChannelEntity& snapshot, uint32_t localCkeNum, uint32_t localXnNum, uint32_t remoteCkeNum,
    uint32_t remoteXnNum) const
{
    snapshot.header.version = HCOMM_CCU_CHANNEL_ABI_VERSION;
    snapshot.header.magicWord = HCOMM_CCU_CHANNEL_MAGIC_WORD;
    snapshot.header.size = sizeof(HcommCcuChannelEntity);
    snapshot.engine = GetEngine();
    snapshot.protocol = channelDesc_.remoteEndpoint.protocol;
    snapshot.deviceLogicId = GetDevLogicId();
    snapshot.dieId = GetDieId();
    snapshot.channelId = GetChannelId();
    snapshot.localVarNum = localXnNum;
    snapshot.remoteVarNum = remoteXnNum;
    snapshot.localEventNum = localCkeNum;
    snapshot.remoteEventNum = remoteCkeNum;
}

HcclResult CcuUrmaChannel::FillSnapshotResourceIds(
    HcommCcuChannelEntity& snapshot, uint32_t localCkeNum, uint32_t localXnNum, uint32_t remoteCkeNum,
    uint32_t remoteXnNum) const
{
    for (uint32_t index = 0; index < localXnNum; ++index) {
        CHK_RET(GetLocXnByIndex(index, snapshot.localVarIds[index]));
    }
    for (uint32_t index = 0; index < remoteXnNum; ++index) {
        CHK_RET(GetRmtXnByIndex(index, snapshot.remoteVarIds[index]));
    }
    for (uint32_t index = 0; index < localCkeNum; ++index) {
        CHK_RET(GetLocCkeByIndex(index, snapshot.localEventIds[index]));
    }
    for (uint32_t index = 0; index < remoteCkeNum; ++index) {
        CHK_RET(GetRmtCkeByIndex(index, snapshot.remoteEventIds[index]));
    }
    // 事件掩码为预留槽位，当前保持为 0。
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuUrmaChannel::FillSnapshotRemoteCcuBuffer(HcommCcuChannelEntity& snapshot) const
{
    uint64_t rmtBufAddr = 0;
    uint64_t rmtBufSize = 0;
    CHK_RET(GetRmtCcuBufInfo(rmtBufAddr, rmtBufSize));
    CHK_PRT_RET(
        rmtBufAddr == 0,
        HCCL_WARNING(
            "[%s] remote CCU buffer address is not ready, channel[%llu] dieId[%u].", __func__,
            static_cast<unsigned long long>(GetChannelId()), GetDieId()),
        HcclResult::HCCL_E_UNAVAIL);
    CHK_PRT_RET(
        rmtBufSize == 0,
        HCCL_WARNING(
            "[%s] remote CCU buffer is not ready, channel[%llu] dieId[%u] size[%llu].", __func__,
            static_cast<unsigned long long>(GetChannelId()), GetDieId(), static_cast<unsigned long long>(rmtBufSize)),
        HcclResult::HCCL_E_UNAVAIL);

    uint32_t rmtBufTokenId = 0;
    uint32_t rmtBufTokenValue = 0;
    CHK_RET(GetRmtCcuBufferTokenInfo(rmtBufTokenId, rmtBufTokenValue));
    snapshot.rmtCcuResBuffer.type = REGED_BUFFER_RMA;
    snapshot.rmtCcuResBuffer.bufferInfo.rma.addr = rmtBufAddr;
    snapshot.rmtCcuResBuffer.bufferInfo.rma.size = rmtBufSize;
    snapshot.rmtCcuResBuffer.bufferInfo.rma.protectionInfo.type = PROTECTION_TYPE_UB;
    snapshot.rmtCcuResBuffer.bufferInfo.rma.protectionInfo.memInfo.ub.tokenId = rmtBufTokenId;
    snapshot.rmtCcuResBuffer.bufferInfo.rma.protectionInfo.memInfo.ub.tokenValue = rmtBufTokenValue;
    return HcclResult::HCCL_SUCCESS;
}

// 在状态锁内校验 READY 并构建完整 CCU Channel 实体，避免与 Clean/UpdateMemInfo 并发修改资源状态。
HcclResult CcuUrmaChannel::BuildHcommCcuChannelEntity(
    HcommCcuChannelEntity& channelEntity, std::vector<RegedBufferEntity>& remoteBuffers)
{
    std::lock_guard<std::mutex> lock(statusMtx_);
    if (GetStatusLocked() != ChannelStatus::READY) {
        if (impl_ != nullptr) {
            HCCL_WARNING(
                "[%s] ccu channel is not ready, channel[%llu] dieId[%u].", __func__,
                static_cast<unsigned long long>(GetChannelId()), GetDieId());
        } else {
            // impl_ 未建立时 GetChannelId/GetDieId 返回 UINT32_MAX 哨兵值，打印稳定说明避免误导定位
            HCCL_WARNING("[%s] ccu channel is not ready, impl not built.", __func__);
        }
        return HcclResult::HCCL_E_UNAVAIL;
    }

    uint32_t localCkeNum = 0;
    uint32_t localXnNum = 0;
    uint32_t remoteCkeNum = 0;
    uint32_t remoteXnNum = 0;
    CHK_RET(GetResourceNums(localCkeNum, localXnNum, remoteCkeNum, remoteXnNum));
    CHK_RET(CheckSnapshotResourceNums(localCkeNum, localXnNum, remoteCkeNum, remoteXnNum));

    FillSnapshotBaseFields(channelEntity, localCkeNum, localXnNum, remoteCkeNum, remoteXnNum);
    CHK_RET(GetCcuVersion(channelEntity.ccuVersion));
    CHK_RET(FillSnapshotResourceIds(channelEntity, localCkeNum, localXnNum, remoteCkeNum, remoteXnNum));
    CHK_RET(FillSnapshotRemoteCcuBuffer(channelEntity));

    remoteBuffers.clear();
    CHK_RET(BuildRemoteBufferEntities(remoteBuffers));
    channelEntity.remoteBuffers = remoteBuffers.data();
    channelEntity.remoteBufferNum = static_cast<uint32_t>(remoteBuffers.size());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuUrmaChannel::BuildRemoteBufferEntities(std::vector<RegedBufferEntity>& remoteBuffers)
{
    std::vector<CcuTransport::CclBufferInfo> bufferInfos{};
    CHK_RET(impl_->GetRemoteBufferInfos(bufferInfos));

    remoteBuffers.reserve(bufferInfos.size());
    for (const auto& bufferInfo : bufferInfos) {
        RegedBufferEntity entity{};
        entity.type = REGED_BUFFER_RMA;
        entity.bufferInfo.rma.addr = bufferInfo.addr;
        entity.bufferInfo.rma.size = bufferInfo.size;
        entity.bufferInfo.rma.protectionInfo.type = PROTECTION_TYPE_UB;
        entity.bufferInfo.rma.protectionInfo.memInfo.ub.tokenId = bufferInfo.tokenId;
        entity.bufferInfo.rma.protectionInfo.memInfo.ub.tokenValue = bufferInfo.tokenValue;
        remoteBuffers.push_back(entity);
    }
    return HcclResult::HCCL_SUCCESS;
}

ChannelStatus CcuUrmaChannel::TryPrepareAndConstruct()
{
    Hccl::LinkData linkData = BuildDefaultLinkData();
    HcclResult keyRet
        = EndpointDescPairToLinkData(ccuEndpoint_->GetEndpointDesc(), channelDesc_.remoteEndpoint, linkData);
    if (keyRet != HCCL_SUCCESS) {
        HCCL_ERROR("[%s::%s] failed to build link data, ret[%d].", channelClassName_, __func__, keyRet);
        return ChannelStatus::FAILED;
    }

    HcclResult ret = CreateCcuTransport(
        ccuEndpoint_, linkData, local_.socket, memHandles_.data(), static_cast<uint32_t>(memHandles_.size()),
        channelDesc_.qos, channelDesc_.ubAttr.sqDepth, impl_, channelClassName_);
    if (ret == HCCL_SUCCESS) {
        impl_->SetLocResStatus(CcuTransport::CcuResStatus::RES_OK);
        return ChannelStatus::INIT;
    }
    // 失败(资源不足或硬失败): 构造 msg-only transport 通知对端快速失败, 避免对端 120s 超时
    CcuTransport::CcuResStatus failStatus
        = (ret == HCCL_E_UNAVAIL) ? CcuTransport::CcuResStatus::RES_UNAVAIL : CcuTransport::CcuResStatus::RES_FAILED;
    HCCL_RUN_WARNING(
        "[%s][%s] CreateCcuTransport is unsuccessful[%d], construct msg-only transport.", channelClassName_, __func__,
        ret);
    HcclResult transRet = CcuTransport::ConstructMsgOnlyTransport(local_.socket, impl_, failStatus);
    if (transRet != HCCL_SUCCESS) {
        HCCL_ERROR("[%s::%s] ConstructMsgOnlyTransport failed[%d].", channelClassName_, __func__, transRet);
        return ChannelStatus::FAILED;
    }
    return ChannelStatus::INIT;
}

} // namespace hcomm
