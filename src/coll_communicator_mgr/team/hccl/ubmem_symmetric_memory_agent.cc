/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ubmem_symmetric_memory_agent.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <thread>

#include "common/cast_utils_pub.h"
#include "env_config/env_config_v2.h"
#include "hccl_common.h"
#include "log.h"

namespace hccl {
namespace {
    constexpr const char* UB_MEM_SYMMETRIC_SOCKET_TAG_PREFIX = "ub_mem_sym";
    // Ring通信至少需要两个LSA成员。
    constexpr uint32_t RING_RANK_SIZE_MIN = 2U;

    uint64_t HashCommId(const std::string& commId)
    {
        constexpr uint64_t fnvOffset = 1469598103934665603ULL;
        constexpr uint64_t fnvPrime = 1099511628211ULL;
        uint64_t hash = fnvOffset;
        for (unsigned char value : commId) {
            hash ^= value;
            hash *= fnvPrime;
        }
        return hash;
    }
} // namespace

static_assert(sizeof(UbmemPacket) == UBMEM_PACKET_TOTAL_LEN, "UB Memory exchange packet layout is invalid");

UbMemSymmetricMemoryAgent::UbMemSymmetricMemoryAgent(
    RankGraph* rankGraph, int32_t deviceLogicId, uint32_t selfRank, const std::vector<uint32_t>& worldRankIds,
    uint32_t netLayer, const std::string& commId)
    : rankGraph_(rankGraph),
      deviceLogicId_(deviceLogicId),
      selfRank_(selfRank),
      lsaTeamSize_(static_cast<uint32_t>(worldRankIds.size())),
      worldRankIds_(worldRankIds),
      commId_(commId),
      commHash_(HashCommId(commId)),
      netLayer_(netLayer)
{
    auto selfIter = std::find(worldRankIds_.begin(), worldRankIds_.end(), selfRank_);
    if (lsaTeamSize_ >= RING_RANK_SIZE_MIN && selfIter != worldRankIds_.end()) {
        selfMember_ = static_cast<uint32_t>(selfIter - worldRankIds_.begin());
        leftRank_ = worldRankIds_[(selfMember_ + lsaTeamSize_ - 1U) % lsaTeamSize_];
        rightRank_ = worldRankIds_[(selfMember_ + 1U) % lsaTeamSize_];
    }
}

UbMemSymmetricMemoryAgent::~UbMemSymmetricMemoryAgent() { Finalize(); }

HcclResult UbMemSymmetricMemoryAgent::GetLink(uint32_t peerRank, CommLink& link) const
{
    CommLink* links = nullptr;
    uint32_t linkNum = 0;
    CHK_RET(rankGraph_->GetLinks(netLayer_, selfRank_, peerRank, &links, &linkNum));
    CHK_PRT_RET(
        links == nullptr || linkNum == 0,
        HCCL_ERROR("[%s] no link, layer[%u], selfRank[%u], peerRank[%u]", __func__, netLayer_, selfRank_, peerRank),
        HCCL_E_NOT_FOUND);
    for (uint32_t index = 0; index < linkNum; ++index) {
        if (links[index].linkAttr.linkProtocol == COMM_PROTOCOL_UB_MEM
            && links[index].srcEndpointDesc.loc.locType != ENDPOINT_LOC_TYPE_HOST
            && links[index].dstEndpointDesc.loc.locType != ENDPOINT_LOC_TYPE_HOST) {
            link = links[index];
            return HCCL_SUCCESS;
        }
    }
    HCCL_ERROR(
        "[%s] no UB Memory device link, layer[%u], selfRank[%u], peerRank[%u]", __func__, netLayer_, selfRank_,
        peerRank);
    return HCCL_E_NOT_FOUND;
}

HcclResult UbMemSymmetricMemoryAgent::CheckNeighborLinksAvailable() const
{
    CommLink leftLink{};
    CommLink rightLink{};
    CHK_RET(GetLink(leftRank_, leftLink));
    if (rightRank_ != leftRank_) {
        CHK_RET(GetLink(rightRank_, rightLink));
    }
    return HCCL_SUCCESS;
}

HcclResult UbMemSymmetricMemoryAgent::CheckNeighborLinks()
{
    CHK_PRT_RET(
        lsaTeamSize_ < RING_RANK_SIZE_MIN,
        HCCL_ERROR(
            "[%s] UB Memory LSA team size[%u] is less than minimum[%u]", __func__, lsaTeamSize_, RING_RANK_SIZE_MIN),
        HCCL_E_PARA);
    if (neighborLinksChecked_) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(rankGraph_);
    CHK_PRT_RET(
        selfMember_ >= lsaTeamSize_ || worldRankIds_[selfMember_] != selfRank_,
        HCCL_ERROR("[%s] self rank[%u] is not in UB Memory LSA team", __func__, selfRank_), HCCL_E_PARA);
    CHK_RET(CheckNeighborLinksAvailable());
    neighborLinksChecked_ = true;
    return HCCL_SUCCESS;
}

std::string UbMemSymmetricMemoryAgent::BuildSocketTag(uint32_t peerRank) const
{
    uint32_t lowRank = std::min(selfRank_, peerRank);
    uint32_t highRank = std::max(selfRank_, peerRank);
    return std::string(UB_MEM_SYMMETRIC_SOCKET_TAG_PREFIX) + "_" + std::to_string(commHash_) + "_"
           + std::to_string(netLayer_) + "_" + std::to_string(lowRank) + "_" + std::to_string(highRank);
}

HcclResult
UbMemSymmetricMemoryAgent::CreateNeighborSocket(uint32_t peerRank, const CommLink& link, SocketHandler& socket)
{
    SocketDesc desc{};
    desc.localEndpoint = link.srcEndpointDesc;
    desc.remoteEndpoint = link.dstEndpointDesc;
    desc.role = selfRank_ < peerRank ? HCOMM_SOCKET_ROLE_SERVER : HCOMM_SOCKET_ROLE_CLIENT;

    uint32_t port = 0;
    uint32_t portRank = desc.role == HCOMM_SOCKET_ROLE_SERVER ? selfRank_ : peerRank;
    CHK_RET(rankGraph_->GetDevicePort(portRank, &port));
    CHK_PRT_RET(port > UINT16_MAX, HCCL_ERROR("[%s] invalid port[%u]", __func__, port), HCCL_E_PARA);
    desc.listenPort = static_cast<uint16_t>(port);

    std::string tag;
    EXCEPTION_CATCH(tag = BuildSocketTag(peerRank), return HCCL_E_MEMORY);
    CHK_PRT_RET(tag.size() >= sizeof(desc.tag), HCCL_ERROR("[%s] socket tag is too long", __func__), HCCL_E_PARA);
    CHK_PRT_RET(
        memcpy_s(desc.tag, sizeof(desc.tag), tag.c_str(), tag.size() + 1U) != EOK,
        HCCL_ERROR("[%s] copy socket tag failed", __func__), HCCL_E_MEMORY);
    return SocketCreate(&desc, &socket);
}

HcclResult UbMemSymmetricMemoryAgent::WaitSocketReady(SocketHandler socket) const
{
    auto connectTimeout = std::chrono::seconds(Hccl::EnvConfig::GetInstance().GetSocketConfig().GetLinkTimeOut());
    auto deadline = std::chrono::steady_clock::now() + connectTimeout;
    while (std::chrono::steady_clock::now() < deadline) {
        SocketStates status = SOCKET_CONNECTING;
        CHK_RET(SocketGetStatus(socket, &status));
        if (status == SOCKET_OK) {
            return HCCL_SUCCESS;
        }
        CHK_PRT_RET(status == SOCKET_TIMEOUT, HCCL_ERROR("[%s] socket connect timeout", __func__), HCCL_E_TIMEOUT);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    HCCL_ERROR("[%s] wait socket ready timeout", __func__);
    return HCCL_E_TIMEOUT;
}

HcclResult UbMemSymmetricMemoryAgent::InitRecvThread()
{
    threadRun_ = true;
    EXCEPTION_CATCH(
        recvThread_ = std::make_unique<std::thread>(&UbMemSymmetricMemoryAgent::DealWithRequest, std::ref(*this)),
        return HCCL_E_MEMORY);
    CHK_SMART_PTR_NULL(recvThread_);
    return HCCL_SUCCESS;
}

HcclResult UbMemSymmetricMemoryAgent::Init()
{
    if (isExchangeInfo_) {
        return HCCL_SUCCESS;
    }
    CHK_RET(CheckNeighborLinks());

    CommLink leftLink{};
    CHK_RET(GetLink(leftRank_, leftLink));
    HcclResult ret = CreateNeighborSocket(leftRank_, leftLink, leftSocket_);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[%s] create left neighbor socket failed, selfRank[%u], peerRank[%u], netLayer[%u], ret[%d]", __func__,
            selfRank_, leftRank_, netLayer_, ret);
        Finalize();
        return ret;
    }
    if (rightRank_ == leftRank_) {
        rightSocket_ = leftSocket_;
    } else {
        CommLink rightLink{};
        ret = GetLink(rightRank_, rightLink);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[%s] get right neighbor UB Memory link failed, selfRank[%u], peerRank[%u], netLayer[%u], ret[%d]",
                __func__, selfRank_, rightRank_, netLayer_, ret);
            Finalize();
            return ret;
        }
        ret = CreateNeighborSocket(rightRank_, rightLink, rightSocket_);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[%s] create right neighbor socket failed, selfRank[%u], peerRank[%u], netLayer[%u], ret[%d]", __func__,
                selfRank_, rightRank_, netLayer_, ret);
            Finalize();
            return ret;
        }
    }
    ret = WaitSocketReady(leftSocket_);
    if (ret == HCCL_SUCCESS && rightSocket_ != leftSocket_) {
        ret = WaitSocketReady(rightSocket_);
    }
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[%s] wait UB Memory LSA ring socket ready failed, selfRank[%u], leftRank[%u], rightRank[%u], "
            "netLayer[%u], ret[%d]",
            __func__, selfRank_, leftRank_, rightRank_, netLayer_, ret);
        Finalize();
        return ret;
    }
    ret = InitRecvThread();
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] initialize UB Memory receive thread failed, ret[%d]", __func__, ret);
        Finalize();
        return ret;
    }
    isExchangeInfo_ = true;
    HCCL_RUN_INFO(
        "[%s] UB Memory LSA ring ready, comm[%s], rank[%u], left[%u], right[%u], layer[%u]", __func__, commId_.c_str(),
        selfRank_, leftRank_, rightRank_, netLayer_);
    return HCCL_SUCCESS;
}

void UbMemSymmetricMemoryAgent::Finalize()
{
    isExchangeInfo_ = false;
    threadRun_ = false;
    {
        std::lock_guard<std::mutex> lock(completionMutex_);
        isProcessingTask_ = false;
        exchangeResult_ = HCCL_E_UNAVAIL;
    }
    completionCv_.notify_all();
    if (recvThread_ != nullptr && recvThread_->joinable()) {
        recvThread_->join();
    }
    recvThread_.reset();
    ClearRequestQueue();

    if (rightSocket_ != nullptr && rightSocket_ != leftSocket_) {
        CHK_PRT(SocketDestroy(rightSocket_));
    }
    if (leftSocket_ != nullptr) {
        CHK_PRT(SocketDestroy(leftSocket_));
    }
    rightSocket_ = nullptr;
    leftSocket_ = nullptr;
}

void UbMemSymmetricMemoryAgent::ClearRequestQueue()
{
    std::lock_guard<std::mutex> lock(queueMutex_);
    std::queue<UbmemPacket> emptyQueue;
    requestQueue_.swap(emptyQueue);
}

void UbMemSymmetricMemoryAgent::CompleteTask(HcclResult result)
{
    {
        std::lock_guard<std::mutex> lock(completionMutex_);
        if (!isProcessingTask_) {
            return;
        }
        exchangeResult_ = result;
        isProcessingTask_ = false;
    }
    completionCv_.notify_all();
}

HcclResult UbMemSymmetricMemoryAgent::ProcessReceivedPacket(const UbmemPacket& packet)
{
    CHK_PRT_RET(
        packet.type != UbmemPacketType::DATA || packet.memberId >= lsaTeamSize_,
        HCCL_ERROR(
            "[%s] invalid packet, type[%u], memberId[%u]", __func__, static_cast<uint32_t>(packet.type),
            packet.memberId),
        HCCL_E_PARA);
    if (packet.memberId != selfMember_) {
        CHK_PTR_NULL(outputDataPtr_);
        uint8_t* dest = outputDataPtr_ + packet.memberId * currentInputSize_;
        CHK_SAFETY_FUNC_RET(memcpy_s(dest, currentInputSize_, packet.data, currentInputSize_));
        ++collectedCount_;
    }
    HCCL_INFO(
        "[%s] received member[%u], collected[%u/%u]", __func__, packet.memberId, collectedCount_.load(), lsaTeamSize_);
    if (packet.memberId != selfMember_ && worldRankIds_[packet.memberId] != rightRank_) {
        std::lock_guard<std::mutex> lock(queueMutex_);
        EXCEPTION_CATCH(requestQueue_.push(packet), return HCCL_E_MEMORY);
    }
    return HCCL_SUCCESS;
}

void UbMemSymmetricMemoryAgent::DealWithRequest()
{
    if (hrtSetDevice(deviceLogicId_) != HCCL_SUCCESS) {
        return;
    }

    while (threadRun_) {
        HcclResult taskProgressResult = HCCL_SUCCESS;
        bool taskComplete = false;
        {
            // 保护本轮信息交换状态；涉及发送队列时始终先锁completionMutex_、再锁queueMutex_。
            std::lock_guard<std::mutex> taskStateLock(completionMutex_);
            if (isProcessingTask_) {
                if (collectedCount_ < lsaTeamSize_) {
                    uint64_t completedSize = 0;
                    HcclResult recvResult = SocketRecvNb(
                        leftSocket_, ReinterpretAs<uint8_t*>(&recvPacket_) + receivedPacketSize_,
                        sizeof(UbmemPacket) - receivedPacketSize_, &completedSize);
                    if (recvResult == HCCL_SUCCESS) {
                        if (completedSize > sizeof(UbmemPacket) - receivedPacketSize_) {
                            HCCL_ERROR(
                                "[%s] invalid socket receive completion[%llu]", __func__,
                                static_cast<unsigned long long>(completedSize));
                            taskProgressResult = HCCL_E_INTERNAL;
                        } else {
                            receivedPacketSize_ += completedSize;
                            if (receivedPacketSize_ == sizeof(UbmemPacket)) {
                                taskProgressResult = ProcessReceivedPacket(recvPacket_);
                                recvPacket_ = {};
                                receivedPacketSize_ = 0;
                            }
                        }
                    } else if (recvResult != HCCL_E_AGAIN) {
                        // 与A3一致：接收异常只记录并等待后续重试，不阻塞本轮发送。
                        HCCL_ERROR(
                            "[%s] receive UB Memory information failed, ret[%d], remoteRank[%u], receivedSize[%zu]",
                            __func__, recvResult, leftRank_, receivedPacketSize_);
                    }
                }

                std::lock_guard<std::mutex> queueLock(queueMutex_);
                if (!requestQueue_.empty()) {
                    UbmemPacket& packet = requestQueue_.front();
                    uint64_t completedSize = 0;
                    HcclResult sendResult = SocketSendNb(
                        rightSocket_, ReinterpretAs<uint8_t*>(&packet) + sentPacketSize_,
                        sizeof(UbmemPacket) - sentPacketSize_, &completedSize);
                    if (sendResult == HCCL_SUCCESS) {
                        if (completedSize > sizeof(UbmemPacket) - sentPacketSize_) {
                            HCCL_ERROR(
                                "[%s] invalid socket send completion[%llu]", __func__,
                                static_cast<unsigned long long>(completedSize));
                            taskProgressResult = HCCL_E_INTERNAL;
                        } else {
                            sentPacketSize_ += completedSize;
                            if (sentPacketSize_ == sizeof(UbmemPacket)) {
                                requestQueue_.pop();
                                sentPacketSize_ = 0;
                            }
                        }
                    } else if (sendResult != HCCL_E_AGAIN) {
                        // 发送异常时保留队首Packet，与A3一致在后续循环中继续重试。
                        HCCL_ERROR(
                            "[%s] send UB Memory information failed, ret[%d], memberId[%u], remoteRank[%u], "
                            "sentSize[%zu]",
                            __func__, sendResult, packet.memberId, rightRank_, sentPacketSize_);
                    }
                }
                taskComplete = requestQueue_.empty() && collectedCount_ == lsaTeamSize_;
            }
        }

        if (taskProgressResult != HCCL_SUCCESS) {
            HCCL_ERROR("[%s] progress UB Memory information exchange failed, ret[%d]", __func__, taskProgressResult);
            CompleteTask(taskProgressResult);
        } else if (taskComplete) {
            CompleteTask(HCCL_SUCCESS);
        }
        // 每轮固定休眠1 ms，避免后台线程持续轮询占用CPU。
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    hrtResetDevice(deviceLogicId_);
}

HcclResult UbMemSymmetricMemoryAgent::WaitForCollectionComplete()
{
    std::unique_lock<std::mutex> lock(completionMutex_);
    auto timeout = std::chrono::seconds(Hccl::EnvConfig::GetInstance().GetSocketConfig().GetLinkTimeOut());
    bool completed = completionCv_.wait_for(lock, timeout, [this]() {
        return !isProcessingTask_.load();
    });
    if (!completed) {
        HCCL_ERROR(
            "[%s] information exchange timeout, collected[%u/%u]", __func__, collectedCount_.load(), lsaTeamSize_);
        exchangeResult_ = HCCL_E_TIMEOUT;
        isProcessingTask_ = false;
        outputDataPtr_ = nullptr;
        currentInputSize_ = 0;
        recvPacket_ = {};
        receivedPacketSize_ = 0;
        sentPacketSize_ = 0;
    }
    HcclResult result = exchangeResult_;
    lock.unlock();
    if (!completed) {
        ClearRequestQueue();
    }
    return result;
}

HcclResult UbMemSymmetricMemoryAgent::ExchangeInfo(void* inputPtr, void* outputPtr, uint64_t inputSize)
{
    CHK_PTR_NULL(inputPtr);
    CHK_PTR_NULL(outputPtr);
    CHK_PRT_RET(!isExchangeInfo_, HCCL_ERROR("[%s] UB Memory LSA ring is not initialized", __func__), HCCL_E_UNAVAIL);
    CHK_PRT_RET(inputSize == 0, HCCL_ERROR("[%s] input size is zero", __func__), HCCL_E_PARA);
    CHK_PRT_RET(
        inputSize > UBMEM_PACKET_DATA_MAX_LEN,
        HCCL_ERROR(
            "[%s] input size[%llu] exceeds maximum[%u]", __func__, static_cast<unsigned long long>(inputSize),
            UBMEM_PACKET_DATA_MAX_LEN),
        HCCL_E_PARA);

    uint8_t* output = static_cast<uint8_t*>(outputPtr);
    {
        std::lock_guard<std::mutex> taskStateLock(completionMutex_);
        outputDataPtr_ = output;
        currentInputSize_ = inputSize;
        collectedCount_ = 0U;
        exchangeResult_ = HCCL_SUCCESS;
        recvPacket_ = {};
        receivedPacketSize_ = 0;
        sentPacketSize_ = 0;
    }
    uint8_t* selfOutput = output + selfMember_ * inputSize;
    CHK_SAFETY_FUNC_RET(memcpy_s(selfOutput, inputSize, inputPtr, inputSize));
    ++collectedCount_;

    UbmemPacket sendPacket{};
    sendPacket.type = UbmemPacketType::DATA;
    sendPacket.memberId = selfMember_;
    CHK_SAFETY_FUNC_RET(memcpy_s(sendPacket.data, sizeof(sendPacket.data), inputPtr, inputSize));
    ClearRequestQueue();
    {
        std::lock_guard<std::mutex> queueLock(queueMutex_);
        EXCEPTION_CATCH(requestQueue_.push(sendPacket), return HCCL_E_MEMORY);
    }
    isProcessingTask_ = true;
    CHK_RET(WaitForCollectionComplete());
    HCCL_INFO("[%s] exchanged information for[%u] LSA members", __func__, lsaTeamSize_);
    return HCCL_SUCCESS;
}

} // namespace hccl
