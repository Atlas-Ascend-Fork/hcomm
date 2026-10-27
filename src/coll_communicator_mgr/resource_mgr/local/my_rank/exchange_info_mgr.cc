/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exchange_info_mgr.h"
#include "env_config/env_config_v2.h"
#include "exception_handler.h"
#include "rank_consistency_checker_v2.h"
#include "hcomm_res_defs.h"
#include "hcomm_channel.h"
#include "endpoint_pair/channel/channel_process_pub.h"
#include "binary_stream.h"
#include "common/comm_engine_utils_pub.h"

namespace hccl {
// 单个数据包（不含前置size头）上限：帧大小字段 + CheckFrameV2 + 长度字段 + 用户数据
constexpr uint64_t PACKED_SIZE_MAX
    = sizeof(size_t) + sizeof(CheckFrameV2) + sizeof(size_t) + static_cast<uint64_t>(HCCL_EXCHANGE_INFO_LEN);

ExchangeInfoMgr::~ExchangeInfoMgr() { HCCL_INFO("[ExchangeInfoMgr][~ExchangeInfoMgr] ExchangeInfoMgr deinit"); }

HcclResult ExchangeInfoMgr::BatchExchangeAndCheckConsistency(
    const uint32_t* remoteRanks, const ChannelHandle* channelHandles, uint32_t channelNum,
    CollCommConfigConsistency& collCommConfigConsistency)
{
    EXCEPTION_HANDLE_BEGIN
    if (channelNum == 0) {
        HCCL_INFO("[BatchExchangeAndCheckConsistency] channelNum is 0.");
        return HCCL_SUCCESS;
    }

    // 单轮非批量交换上下文：本函数内闭环，调用结束即释放
    ExchangeCtx normalCtx;
    normalCtx.rmtRankIds.reserve(channelNum);
    std::vector<Hccl::Socket*> sockets;
    sockets.reserve(channelNum);

    for (uint32_t i = 0; i < channelNum; i++) {
        uint32_t remoteRank = remoteRanks[i];
        HcommChannelDesc hcommDesc{};
        CHK_RET(hcomm::ChannelProcess::ChannelGetDesc(channelHandles[i], &hcommDesc));
        HcommSocket rawSocket = hcommDesc.socket;
        Hccl::Socket* socket = static_cast<Hccl::Socket*>(rawSocket);
        CHK_PRT_RET(
            socket == nullptr,
            HCCL_ERROR(
                "[BatchExchangeAndCheckConsistency] socket is null for channel[%u] remoteRank[%u].", i, remoteRank),
            HCCL_E_INTERNAL);
        sockets.push_back(socket);
        normalCtx.rmtRankIds.push_back(remoteRank);
    }

    // 非批量通道：发送/接收已拼好的整包，收齐后统一解包
    CHK_RET(ExchangeParamInfos(sockets, collCommConfigConsistency, normalCtx));
    CHK_RET(collCommConfigConsistency.ResetExchangeInfo());

    EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}

// 处理交换后收到的数据：按包内checkFrameSize决定是否比对CheckFrameV2，并存储对端HCCL算子交换信息
// 接收数据布局与PackExchangeInfo的发送拼包对称：
// [size_t checkFrameSize][CheckFrameV2(可选)][hcclDataLen(size_t)][hcclData]
// 收到的packed大小由调用方先经CheckRecvedSizes校验；每个rank每轮均收发完整包，recvBuff为空视为内部异常
HcclResult
ExchangeInfoMgr::ConsistencyRecvedDataProc(CollCommConfigConsistency& collCommConfigConsistency, ExchangeCtx& ctx) const
{
    if (ctx.recvBuffs.empty()) {
        HCCL_INFO("[ConsistencyRecvedDataProc] no recv buffs, skip.");
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(
        ctx.rmtRankIds.size() != ctx.recvBuffs.size(),
        HCCL_ERROR(
            "[ConsistencyRecvedDataProc] rmtRankIds size[%zu] mismatch recvBuffs size[%zu].", ctx.rmtRankIds.size(),
            ctx.recvBuffs.size()),
        HCCL_E_INTERNAL);

    // 本端校验帧按需生成：仅当对端包内含hcommData时用于比对
    CheckFrameV2 localFrame{};
    RankConsistencyCheckerV2* checker = nullptr;

    for (size_t i = 0; i < ctx.recvBuffs.size(); ++i) {
        // 每个rank每轮均收发完整包（最小 [size_t 0][size_t 0] 共16字节），CheckRecvedSizes已保证size非0，为空属内部异常
        CHK_PRT_RET(
            ctx.recvBuffs[i].empty(),
            HCCL_ERROR(
                "[ConsistencyRecvedDataProc] recvBuff is empty for remoteRank[%u], unexpected.", ctx.rmtRankIds[i]),
            HCCL_E_INTERNAL);
        // 与PackExchangeInfo的打包对称：BinaryStream反序列化checkFrameSize + CheckFrameV2(可选) + vector<u8>
        Hccl::BinaryStream binaryStream(ctx.recvBuffs[i]);
        size_t checkFrameSize = 0;
        binaryStream >> checkFrameSize;
        if (checkFrameSize > 0) {
            // 对端CheckFrameV2大小必须与本端一致，否则结构布局不兼容，直接报错避免错位解析
            CHK_PRT_RET(
                checkFrameSize != sizeof(CheckFrameV2),
                HCCL_ERROR(
                    "[ConsistencyRecvedDataProc] remote CheckFrameV2 size[%zu] mismatch local[%zu], remoteRank[%u].",
                    checkFrameSize, sizeof(CheckFrameV2), ctx.rmtRankIds[i]),
                HCCL_E_PARA);
            if (checker == nullptr) {
                s32 deviceLogicId = 0;
                CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));
                checker = &RankConsistencyCheckerV2::GetInstance(deviceLogicId);
                CHK_RET(checker->GenerateCheckFrameV2(localFrame));
            }
            CheckFrameV2 remoteFrame{};
            binaryStream >> remoteFrame;
            CHK_RET(checker->CompareCheckFrameV2(localFrame, remoteFrame, ctx.rmtRankIds[i]));
        }

        std::vector<u8> hcclData;
        binaryStream >> hcclData;
        // 对端数据长度对称性校验、两端均为空时不存储，均由 StoreRemoteExchangeInfo 内部处理
        CHK_RET(collCommConfigConsistency.StoreRemoteExchangeInfo(ctx.rmtRankIds[i], hcclData));
    }

    HCCL_INFO("[ConsistencyRecvedDataProc] processed [%zu] channels suc.", ctx.recvBuffs.size());
    return HCCL_SUCCESS;
}

// 校验各通道收到的size头与期望一致（期望值由PrepareExchangeBuffs与发送计划同步填入）
HcclResult ExchangeInfoMgr::CheckRecvedSizes(const ExchangeCtx& ctx) const
{
    CHK_PRT_RET(
        ctx.recvedSizes.size() != ctx.expectRecvSizes.size() || ctx.recvedSizes.size() != ctx.rmtRankIds.size(),
        HCCL_ERROR(
            "[CheckRecvedSizes] recvedSizes size[%zu] mismatch expectRecvSizes size[%zu] or rmtRankIds size[%zu].",
            ctx.recvedSizes.size(), ctx.expectRecvSizes.size(), ctx.rmtRankIds.size()),
        HCCL_E_PARA);
    for (size_t i = 0; i < ctx.recvedSizes.size(); i++) {
        CHK_PRT_RET(
            ctx.recvedSizes[i] != ctx.expectRecvSizes[i],
            HCCL_ERROR(
                "[CheckRecvedSizes] recvedSize[%llu] mismatch expectRecvSize[%llu], remoteRank[%u].",
                static_cast<unsigned long long>(ctx.recvedSizes[i]),
                static_cast<unsigned long long>(ctx.expectRecvSizes[i]), ctx.rmtRankIds[i]),
            HCCL_E_PARA);
    }
    HCCL_INFO("[CheckRecvedSizes] checked [%zu] channels suc.", ctx.recvedSizes.size());
    return HCCL_SUCCESS;
}

// 按rank生成本轮发送计划并同步期望接收大小：未校验rank发allData，已校验rank发onlyHcclData
// （无hcclData时为 [size_t 0][空hcclData] 包，对端解包后跳过存储）
HcclResult ExchangeInfoMgr::PrepareExchangeBuffs(const std::vector<Hccl::Socket*>& sockets, ExchangeCtx& ctx) const
{
    CHK_PRT_RET(
        ctx.rmtRankIds.size() != sockets.size(),
        HCCL_ERROR(
            "[PrepareExchangeBuffs] rmtRankIds size[%zu] mismatch sockets size[%zu].", ctx.rmtRankIds.size(),
            sockets.size()),
        HCCL_E_PARA);
    ctx.sendBuffs.assign(sockets.size(), nullptr);
    ctx.recvBuffs.assign(sockets.size(), std::vector<char>());
    ctx.recvedSizes.assign(sockets.size(), 0);
    ctx.expectRecvSizes.assign(sockets.size(), 0);
    // 一次性持锁完成本轮所有rank的已校验判断，避免在循环内反复加解锁
    std::lock_guard<std::mutex> lock(checkedRankMtx_);
    for (size_t i = 0; i < sockets.size(); i++) {
        if (checkedRankIdList_.find(ctx.rmtRankIds[i]) == checkedRankIdList_.end()) {
            ctx.sendBuffs[i] = &ctx.allData;
            ctx.expectRecvSizes[i] = ctx.sizeAll;
        } else {
            // 无hcclData时为 [size_t 0][空hcclData] 包，对端解包后跳过存储
            ctx.sendBuffs[i] = &ctx.onlyHcclData;
            ctx.expectRecvSizes[i] = ctx.sizeOnlyHccl;
        }
    }
    return HCCL_SUCCESS;
}

// 非批量通道参数交换：拼包并发送，收齐后统一解包
HcclResult ExchangeInfoMgr::ExchangeParamInfos(
    const std::vector<Hccl::Socket*>& sockets, CollCommConfigConsistency& collCommConfigConsistency, ExchangeCtx& ctx)
{
    if (sockets.empty()) {
        HCCL_INFO("[ExchangeParamInfos] no socket to exchange, skip.");
        return HCCL_SUCCESS;
    }

    // 按socket自身NicType选收发接口：HOST_NIC_TYPE同步、DEVICE_NIC_TYPE异步（NicType由本端endpoint
    // 的locType推导，与remoteEndpoint无关）；同批本端endpoint一致，取首个socket代表整批。
    bool isHostLoc = (sockets[0]->GetNicType() == Hccl::NicType::HOST_NIC_TYPE);
    HCCL_INFO("[ExchangeParamInfos] isHostLoc[%d], socketNum[%zu].", isHostLoc, sockets.size());

    // 非批量路径：拼包时在包前预留uint64_t记录本包大小
    CHK_RET(PackExchangeInfo(collCommConfigConsistency, ctx));

    // 已校验过的rank无需再交换hcommData：发onlyHcclData（无hcclData时为空载荷包）
    CHK_RET(PrepareExchangeBuffs(sockets, ctx));

    if (isHostLoc) {
        // host网卡：同步收发，每个socket先发8字节size头并收对端size头，再按对端size收载荷
        for (size_t i = 0; i < sockets.size(); i++) {
            sockets[i]->Send(ctx.sendBuffs[i]->data(), static_cast<u32>(ctx.sendBuffs[i]->size()));
            sockets[i]->Recv(&ctx.recvedSizes[i], sizeof(uint64_t));
        }
        CHK_RET(CheckRecvedSizes(ctx));
        for (size_t i = 0; i < sockets.size(); i++) {
            if (ctx.recvedSizes[i] == 0) {
                continue;
            }
            ctx.recvBuffs[i].resize(static_cast<size_t>(ctx.recvedSizes[i]), 0);
            sockets[i]->Recv(ctx.recvBuffs[i].data(), static_cast<u32>(ctx.recvedSizes[i]));
        }
    } else {
        // device网卡：异步收发，每个socket均先发size头；收齐size头统一校验后再按对端size收载荷
        for (size_t i = 0; i < sockets.size(); i++) {
            sockets[i]->SendAsync(ctx.sendBuffs[i]->data(), static_cast<u32>(ctx.sendBuffs[i]->size()));
        }
        CHK_RET(WaitAllAsyncComplete(sockets, ctx.rmtRankIds));
        for (size_t i = 0; i < sockets.size(); i++) {
            sockets[i]->RecvAsync(&ctx.recvedSizes[i], sizeof(uint64_t));
        }
        CHK_RET(WaitAllAsyncComplete(sockets, ctx.rmtRankIds));
        CHK_RET(CheckRecvedSizes(ctx));
        for (size_t i = 0; i < sockets.size(); i++) {
            if (ctx.recvedSizes[i] == 0) {
                continue;
            }
            ctx.recvBuffs[i].resize(static_cast<size_t>(ctx.recvedSizes[i]), 0);
            sockets[i]->RecvAsync(ctx.recvBuffs[i].data(), static_cast<u32>(ctx.recvedSizes[i]));
        }
        CHK_RET(WaitAllAsyncComplete(sockets, ctx.rmtRankIds));
    }

    // 全部接收完成后统一解包
    CHK_RET(ConsistencyRecvedDataProc(collCommConfigConsistency, ctx));

    // 本轮完成后记录所有对端rank已校验，后续轮次无需再交换hcommData
    {
        std::lock_guard<std::mutex> lock(checkedRankMtx_);
        checkedRankIdList_.insert(ctx.rmtRankIds.begin(), ctx.rmtRankIds.end());
    }
    return HCCL_SUCCESS;
}

// 拼单个数据包：[uint64_t size][size_t checkFrameSize][CheckFrameV2(可选)][hcclDataLen(size_t)][hcclData]
// checkFrameSize为CheckFrameV2大小（0表示不带帧）；包前预留uint64_t记录本包大小，packetSize不含前置size头
static HcclResult PackOnePacket(
    bool withHcommData, const CheckFrameV2& localFrame, const std::vector<u8>& hcclData, std::vector<char>& packet,
    uint64_t& packetSize)
{
    packet.assign(sizeof(uint64_t), 0);
    Hccl::BinaryStream stream;
    const size_t checkFrameSize = withHcommData ? sizeof(CheckFrameV2) : 0;
    stream << checkFrameSize;
    if (withHcommData) {
        stream << localFrame;
    }
    stream << hcclData;
    stream.Dump(packet);
    packetSize = static_cast<uint64_t>(packet.size() - sizeof(uint64_t));
    CHK_SAFETY_FUNC_RET(memcpy_s(packet.data(), sizeof(uint64_t), &packetSize, sizeof(uint64_t)));
    return HCCL_SUCCESS;
}

// 生成并拼包本端一致性交换信息：
// allData = [size_t checkFrameSize=sizeof(CheckFrameV2)][CheckFrameV2][hcclDataLen(size_t)][hcclData]
// onlyHcclData = [size_t checkFrameSize=0][hcclDataLen(size_t)][hcclData]
HcclResult
ExchangeInfoMgr::PackExchangeInfo(const CollCommConfigConsistency& collCommConfigConsistency, ExchangeCtx& ctx) const
{
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));
    auto& checker = RankConsistencyCheckerV2::GetInstance(deviceLogicId);
    CheckFrameV2 localFrame{};
    CHK_RET(checker.GenerateCheckFrameV2(localFrame));

    std::vector<u8> hcclData;
    CHK_RET(collCommConfigConsistency.GetExchangeInfoBuf(hcclData));

    // 含hcommData的整包
    CHK_RET(PackOnePacket(true, localFrame, hcclData, ctx.allData, ctx.sizeAll));

    // 上限检查：packed内容（不含前置size头）= bool + CheckFrameV2 + 长度字段 + 用户数据，
    // 用户数据上限由 AddExchangeInfo 保证
    CHK_PRT_RET(
        ctx.sizeAll > PACKED_SIZE_MAX,
        HCCL_ERROR(
            "[PackExchangeInfo] packed size[%llu] exceeds limit[%llu].", static_cast<unsigned long long>(ctx.sizeAll),
            static_cast<unsigned long long>(PACKED_SIZE_MAX)),
        HCCL_E_PARA);

    // 不含hcommData的整包：hcclData为空时包内为 [size_t 0][空hcclData]，对端解包后跳过存储
    CHK_RET(PackOnePacket(false, localFrame, hcclData, ctx.onlyHcclData, ctx.sizeOnlyHccl));

    HCCL_INFO(
        "[PackExchangeInfo] sizeAll[%llu], sizeOnlyHccl[%llu].", static_cast<unsigned long long>(ctx.sizeAll),
        static_cast<unsigned long long>(ctx.sizeOnlyHccl));
    return HCCL_SUCCESS;
}

HcclResult ExchangeInfoMgr::WaitAllAsyncComplete(
    const std::vector<Hccl::Socket*>& sockets, const std::vector<uint32_t>& remoteRanks) const
{
    auto timeout = std::chrono::seconds(Hccl::EnvConfig::GetInstance().GetSocketConfig().GetLinkTimeOut());
    auto startTime = std::chrono::steady_clock::now();
    std::vector<bool> done(sockets.size(), false);
    uint32_t doneCount = 0;

    while (doneCount < sockets.size()) {
        for (size_t i = 0; i < sockets.size(); i++) {
            if (done[i]) {
                continue;
            }
            Hccl::SocketStatus status = sockets[i]->GetAsyncStatus();
            if (status == Hccl::SocketStatus::OK) {
                done[i] = true;
                doneCount++;
                continue;
            }
            if (status == Hccl::SocketStatus::TIMEOUT) {
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                   std::chrono::steady_clock::now() - startTime)
                                   .count();
                HCCL_ERROR(
                    "[WaitAllAsyncComplete] socket timeout for remoteRank[%u], elapsed[%lld]ms.", remoteRanks[i],
                    elapsed);
                return HCCL_E_TIMEOUT;
            }
        }
        if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
            auto elapsed
                = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime)
                      .count();
            for (size_t i = 0; i < sockets.size(); i++) {
                if (!done[i]) {
                    HCCL_ERROR(
                        "[WaitAllAsyncComplete] wall-clock timeout for remoteRank[%u], elapsed[%lld]ms.",
                        remoteRanks[i], elapsed);
                }
            }
            return HCCL_E_TIMEOUT;
        }
    }

    auto elapsed
        = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime).count();
    HCCL_INFO("[WaitAllAsyncComplete] all[%zu] sockets completed, elapsed[%lld]ms.", sockets.size(), elapsed);
    return HCCL_SUCCESS;
}
} // namespace hccl
