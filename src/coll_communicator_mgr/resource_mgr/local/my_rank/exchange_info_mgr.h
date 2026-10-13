/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXCHANGE_INFO_MGR_H
#define EXCHANGE_INFO_MGR_H

#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>
#include "hccl/hccl_types.h"
#include "hccl/base.h"
#include "hccl/hccl_res.h"
#include "hcomm_res_defs.h"
#include "hcomm_channel.h"
#include "common.h"
#include "coll_comm_config_consistency.h"
#include "socket/socket.h"

namespace hccl {
class ExchangeInfoMgr {
public:
    // 单轮非批量交换上下文
    struct ExchangeCtx {
        std::vector<uint32_t> rmtRankIds{}; // 远端rank列表

        // 本端发送数据包
        std::vector<char> allData{};      // 含hcommData：[size_t checkFrameSize][CheckFrameV2][hcclData]
        uint64_t sizeAll{0};              // allData包大小（不含前置size头）
        std::vector<char> onlyHcclData{}; // 不含hcommData：[size_t checkFrameSize=0][hcclData]
        uint64_t sizeOnlyHccl{0};         // onlyHcclData包大小（不含前置size头）
        std::vector<const std::vector<char>*> sendBuffs{}; // 各rank对应的发送数据包

        // 接收数据
        std::vector<std::vector<char>> recvBuffs{};
        std::vector<uint64_t> recvedSizes{};     // 对端size头
        std::vector<uint64_t> expectRecvSizes{}; // 各rank期望接收的包大小（与发送计划同步确定）
    };

    ExchangeInfoMgr() = default;
    ~ExchangeInfoMgr();

    HcclResult BatchExchangeAndCheckConsistency(
        const uint32_t* remoteRanks, const ChannelHandle* channelHandles, uint32_t channelNum,
        CollCommConfigConsistency& collCommConfigConsistency);
    // 处理交换后收到的数据：按包内checkFrameSize决定是否比对CheckFrameV2，并存储对端HCCL算子交换信息
    HcclResult ConsistencyRecvedDataProc(CollCommConfigConsistency& collCommConfigConsistency, ExchangeCtx& ctx) const;
    // 校验各通道收到的size头与期望一致（期望值由PrepareExchangeBuffs与发送计划同步填入）
    HcclResult CheckRecvedSizes(const ExchangeCtx& ctx) const;
    // 非批量通道参数交换：每个socket始终先收发8字节size头，再按对端size收载荷（无hcclData的rank
    // 发送 [size_t 0][空hcclData] 包，对端解包后跳过存储）；按socket自身NicType选择同步/异步接口
    HcclResult ExchangeParamInfos(
        const std::vector<Hccl::Socket*>& sockets, CollCommConfigConsistency& collCommConfigConsistency,
        ExchangeCtx& ctx);
    HcclResult
    WaitAllAsyncComplete(const std::vector<Hccl::Socket*>& sockets, const std::vector<uint32_t>& remoteRanks) const;

private:
    // 生成并拼包本端一致性交换信息（含/不含hcommData两种包），包前预留uint64_t记录本包大小
    HcclResult PackExchangeInfo(const CollCommConfigConsistency& collCommConfigConsistency, ExchangeCtx& ctx) const;

    // 按rank生成本轮发送计划（ctx.sendBuffs）并同步期望接收大小（ctx.expectRecvSizes）：
    // 未校验rank发allData，已校验rank发onlyHcclData（无hcclData时为 [size_t 0][空hcclData] 包）
    HcclResult PrepareExchangeBuffs(const std::vector<Hccl::Socket*>& sockets, ExchangeCtx& ctx) const;

    // 已校验过的对端rankId：后续轮次无需再交换hcommData（并发调用HcclChannelAcquire时由checkedRankMtx_保护）
    mutable std::mutex checkedRankMtx_;
    std::unordered_set<uint32_t> checkedRankIdList_{};
};
} // namespace hccl

#endif // EXCHANGE_INFO_MGR_H
