/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UBMEM_SYMMETRIC_MEMORY_AGENT_H
#define UBMEM_SYMMETRIC_MEMORY_AGENT_H

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "endpoint_pair/hccl_comm_socket_c_adpt_pub.h"
#include "rank_graph_base.h"

namespace hccl {

constexpr uint32_t UBMEM_PACKET_DATA_MAX_LEN = 144U;
constexpr uint32_t UBMEM_PACKET_TOTAL_LEN = 152U;

enum class UbmemPacketType : uint32_t {
    DATA = 0,
};

// 与A3 Packet语义一致，memberId表示数据在LSA Team输出数组中的位置。
struct UbmemPacket {
    UbmemPacketType type;
    uint32_t memberId;
    uint8_t data[UBMEM_PACKET_DATA_MAX_LEN];
};

class UbMemSymmetricMemoryAgent {
public:
    UbMemSymmetricMemoryAgent(
        RankGraph* rankGraph, int32_t deviceLogicId, uint32_t selfRank, const std::vector<uint32_t>& worldRankIds,
        uint32_t netLayer, const std::string& commId);
    ~UbMemSymmetricMemoryAgent();

    HcclResult CheckNeighborLinks();
    HcclResult Init();
    void Finalize();
    HcclResult ExchangeInfo(void* inputPtr, void* outputPtr, uint64_t inputSize);

    uint32_t GetNetLayer() const { return netLayer_; }

private:
    HcclResult CheckNeighborLinksAvailable() const;
    HcclResult GetLink(uint32_t peerRank, CommLink& link) const;
    HcclResult CreateNeighborSocket(uint32_t peerRank, const CommLink& link, SocketHandler& socket);
    HcclResult WaitSocketReady(SocketHandler socket) const;
    HcclResult InitRecvThread();
    HcclResult WaitForCollectionComplete();
    HcclResult ProcessReceivedPacket(const UbmemPacket& packet);
    void DealWithRequest();
    void CompleteTask(HcclResult result);
    void ClearRequestQueue();
    std::string BuildSocketTag(uint32_t peerRank) const;

    RankGraph* rankGraph_{nullptr};
    int32_t deviceLogicId_{0};
    uint32_t selfRank_{0};
    uint32_t selfMember_{0};
    uint32_t lsaTeamSize_{0};
    std::vector<uint32_t> worldRankIds_;
    std::string commId_;
    uint64_t commHash_{0};
    uint32_t netLayer_{0};
    uint32_t leftRank_{0};
    uint32_t rightRank_{0};
    SocketHandler leftSocket_{nullptr};
    SocketHandler rightSocket_{nullptr};
    bool neighborLinksChecked_{false};
    std::atomic<bool> isExchangeInfo_{false};

    // 接收线程及当前信息交换任务的运行状态。
    std::unique_ptr<std::thread> recvThread_;
    std::atomic<bool> threadRun_{false};
    std::atomic<bool> isProcessingTask_{false};

    // 保存本地待发送Packet及从左邻居收到的待转发Packet。
    std::queue<UbmemPacket> requestQueue_;
    std::mutex queueMutex_;

    // 用于ExchangeInfo等待后台线程完成本轮信息交换。
    std::mutex completionMutex_;
    std::condition_variable completionCv_;

    // 当前信息交换的输出缓冲区及单个成员数据长度。
    uint8_t* outputDataPtr_{nullptr};
    uint64_t currentInputSize_{0};
    // 本轮已收集的LSA成员数量，包含本地成员。
    std::atomic<uint32_t> collectedCount_{0};

    // A5使用非阻塞Socket，记录当前Packet以及已经完成收发的字节数。
    UbmemPacket recvPacket_{};
    uint64_t receivedPacketSize_{0};
    uint64_t sentPacketSize_{0};

    // 后台线程执行本轮信息交换的结果。
    HcclResult exchangeResult_{HCCL_SUCCESS};
};

} // namespace hccl

#endif // UBMEM_SYMMETRIC_MEMORY_AGENT_H
