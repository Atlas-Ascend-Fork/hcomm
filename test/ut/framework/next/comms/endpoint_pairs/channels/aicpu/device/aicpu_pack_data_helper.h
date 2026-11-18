/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_PACK_DATA_HELPER_H
#define AICPU_PACK_DATA_HELPER_H

#include <vector>

#include "hccl_common.h"
#include "binary_stream.h"
#include "mem_transport_common.h"
#include "roce_host_loc_proto_defs.h"

// AICPU 通道建链分发 blob 构造 helper：以 AicpuChannelProcess::ParsePackData 可解析的最小分发数据构造真实 lite
// transport（入 transportMap_）

// 单个 RmaBuffer 的 uniqueId（addr/size/lkey），ROCE 的 ParseNotifyValueBuffer 无 ==0 guard，
// 最小分发 blob 的 notifyValue 段必须塞一个合法 buffer，故保留此 helper
inline std::vector<char> BuildSingleRmaBufferUniqueId(u64 addr, u64 size, u32 key)
{
    Hccl::BinaryStream bs;
    bs << addr;
    bs << size;
    bs << key;
    std::vector<char> result;
    bs.Dump(result);
    return result;
}

// ParsePackData 入参 data 是「首元素为 vector<char> transpUniqueId」的 BinaryStream 序列化，
// 等价于 InitUrmaChannel 里 AicpuResPackageHelper::ParsePackedData(...)[STREAM].data
inline std::vector<char> BuildOuterPackData(const std::vector<char>& transpUniqueId)
{
    Hccl::BinaryStream bs;
    bs << transpUniqueId;
    std::vector<char> result;
    bs.Dump(result);
    return result;
}

// 构造完整的最小单 QP ROCE 分发 blob（外层 pack 后），bufferNum 可以为 0，notify/conn 各保留有效项。
// notifyAddr 参与 transpUniqueId 内容：ParsePackData 的 ROCE 分支按 transpUniqueId 去重复用 transport，
// 需要并存多个通道的用例必须传不同 notifyAddr，否则拿到同一 handle。
inline std::vector<char> BuildRocePackData(u64 notifyAddr = 0x3000)
{
    Hccl::BinaryStream locNotifyStream;
    for (u32 i = 0; i < hcomm::RDMA_NOTIFY_NUM; i++) {
        locNotifyStream << static_cast<u32>(i + 1) << static_cast<u32>(i);
    }
    std::vector<char> locNotify;
    locNotifyStream.Dump(locNotify);

    Hccl::BinaryStream sqStream;
    sqStream << static_cast<u32>(1) << static_cast<u64>(0x10000) << static_cast<u32>(64) << static_cast<u32>(128)
             << static_cast<u64>(0x20000) << static_cast<u64>(0x20008) << static_cast<u64>(0x30000)
             << static_cast<u64>(0x70000) << static_cast<u8>(0) << static_cast<u64>(0);
    std::vector<char> sqUniqueId;
    sqStream.Dump(sqUniqueId);

    Hccl::BinaryStream cqStream;
    cqStream << static_cast<u32>(2) << static_cast<u64>(0x40000) << static_cast<u32>(32) << static_cast<u32>(256)
             << static_cast<u64>(0x50000) << static_cast<u64>(0x50008) << static_cast<u64>(0x60000)
             << static_cast<u64>(0);
    std::vector<char> cqUniqueId;
    cqStream.Dump(cqUniqueId);

    Hccl::BinaryStream connStream;
    connStream << static_cast<u32>(1) << sqUniqueId << cqUniqueId;
    std::vector<char> connUniqueId;
    connStream.Dump(connUniqueId);

    std::vector<char> rmtNotify;
    for (u32 i = 0; i < hcomm::RDMA_NOTIFY_NUM; i++) {
        auto single = BuildSingleRmaBufferUniqueId(notifyAddr + i * 0x100, 0x200, 0x300 + i);
        rmtNotify.insert(rmtNotify.end(), single.begin(), single.end());
    }

    Hccl::BinaryStream inner;
    inner << static_cast<u32>(Hccl::TransportType::ROCE);
    inner << static_cast<u32>(hcomm::RDMA_NOTIFY_NUM); // notifyNum
    inner << static_cast<u32>(0);                      // bufferNum
    inner << static_cast<u32>(1);                      // connNum
    inner << static_cast<u32>(0);                      // qpThreshold
    inner << locNotify;
    inner << rmtNotify;
    inner << BuildSingleRmaBufferUniqueId(notifyAddr, 0x200, 0x300); // notifyValue
    inner << std::vector<char>();                                    // locBuffer
    inner << std::vector<char>();                                    // rmtBuffer
    inner << connUniqueId;
    std::vector<char> transpUniqueId;
    inner.Dump(transpUniqueId);
    return BuildOuterPackData(transpUniqueId);
}

// 构造完整的最小 P2P 分发 blob（外层 pack 后），P2P 的 Init 无 connNum/notifyValue 段，
// 仅 type=P2P + notifyNum/bufferNum=0 + 4 个空 vector
inline std::vector<char> BuildP2pPackData()
{
    Hccl::BinaryStream inner;
    inner << static_cast<u32>(Hccl::TransportType::P2P);
    inner << static_cast<u32>(0); // notifyNum
    inner << static_cast<u32>(0); // bufferNum
    inner << std::vector<char>(); // locNotify（notifyNum==0，ParseLocNotifyVec 提前 return）
    inner << std::vector<char>(); // rmtNotify
    inner << std::vector<char>(); // locBuffer（bufferNum==0，ParseRmtBufferVec 提前 return）
    inner << std::vector<char>(); // rmtBuffer
    std::vector<char> transpUniqueId;
    inner.Dump(transpUniqueId);
    return BuildOuterPackData(transpUniqueId);
}

#endif // AICPU_PACK_DATA_HELPER_H
