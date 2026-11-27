/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>

#include "aicpu/device/aicpu_channel_process.h"
#include "binary_stream.h"
#include "endpoint_pair/channel/ub_unique_id_pub.h"

#define private public
#include "ub_transport_lite_impl.h"
#include "ub_conn_lite.h"
#undef private

using namespace hccl;

namespace {

// 最小 V2 UB transport uniqueId（0 notify / 0 buffer / 0 connection），仅用于构造空 connVec 的 transport
std::vector<char> BuildEmptyUbTransportUniqueId()
{
    Hccl::BinaryStream stream;
    stream << static_cast<uint32_t>(Hccl::TransportType::UB);
    stream << static_cast<uint32_t>(0);
    stream << static_cast<uint32_t>(0);
    stream << static_cast<uint32_t>(0);
    stream << static_cast<uint32_t>(0);
    stream << UB_UNIQUE_ID_EXT_MAGIC_WORD;
    stream << UB_UNIQUE_ID_EXT_VERSION;
    stream << static_cast<uint64_t>(0);
    stream << std::vector<char>();
    stream << std::vector<char>();
    stream << std::vector<char>();
    stream << std::vector<char>();
    stream << std::vector<char>();
    stream << std::vector<char>();
    std::vector<char> uniqueId;
    stream.Dump(uniqueId);
    return uniqueId;
}

// UbConnLite 由 uniqueId 构造仅解析字段，不注册管理器，适合 UT 独立构造
std::vector<char> BuildUbConnUniqueId() { return std::vector<char>(57U, 0); }

} // namespace

class ChannelPiCiDirectSetTest : public testing::Test {
protected:
    void TearDown() override { GlobalMockObject::verify(); }
};

TEST_F(ChannelPiCiDirectSetTest, Ut_SetSqPiCiByTransport_When_UbTransportAndAddr_Expect_SlotsSet)
{
    auto transportUniqueId = BuildEmptyUbTransportUniqueId();
    auto transport = std::make_unique<Hccl::UbTransportLiteImpl>(transportUniqueId);
    auto connUniqueId = BuildUbConnUniqueId();
    auto conn = std::make_unique<Hccl::UbConnLite>(connUniqueId);
    auto* connPtr = conn.get();
    transport->connVec.push_back(connPtr);

    SqContext sqCtx{};
    sqCtx.type = SQ_CONTEXT_TYPE_UB_JFS;
    ASSERT_EQ(AicpuChannelProcess::SetSqPiCiByTransport(transport.get(), &sqCtx, 0U), HCCL_SUCCESS);
    sqCtx.contextInfo.ubJfs.sqHead = 3U;
    sqCtx.contextInfo.ubJfs.sqTail = 4U;
    EXPECT_EQ(connPtr->GetPi(), 3U);
    EXPECT_EQ(connPtr->GetCi(), 4U);
    transport->connVec.clear();
}

// 模拟快恢 Resume：connVec 重装为新 conn 后，按同一块地址重绑到新 conn
TEST_F(ChannelPiCiDirectSetTest, Ut_SetSqPiCiByTransport_WhenConnRebuilt_Expect_SlotsSetOnNewConn)
{
    auto transportUniqueId = BuildEmptyUbTransportUniqueId();
    auto transport = std::make_unique<Hccl::UbTransportLiteImpl>(transportUniqueId);
    auto firstConnUniqueId = BuildUbConnUniqueId();
    auto firstConn = std::make_unique<Hccl::UbConnLite>(firstConnUniqueId);
    transport->connVec.push_back(firstConn.get());

    SqContext sqCtx{};
    sqCtx.type = SQ_CONTEXT_TYPE_UB_JFS;
    ASSERT_EQ(AicpuChannelProcess::SetSqPiCiByTransport(transport.get(), &sqCtx, 0U), HCCL_SUCCESS);
    sqCtx.contextInfo.ubJfs.sqHead = 5U;
    sqCtx.contextInfo.ubJfs.sqTail = 6U;
    EXPECT_EQ(firstConn->GetPi(), 5U);

    transport->connVec.clear();
    auto secondConnUniqueId = BuildUbConnUniqueId();
    auto secondConn = std::make_unique<Hccl::UbConnLite>(secondConnUniqueId);
    auto* secondConnPtr = secondConn.get();
    transport->connVec.push_back(secondConnPtr);

    ASSERT_EQ(AicpuChannelProcess::SetSqPiCiByTransport(transport.get(), &sqCtx, 0U), HCCL_SUCCESS);
    // 绑定即重锚：快恢重绑时实体残留值(5/6)被清零，与新 jetty 消费起点对齐
    EXPECT_EQ(secondConnPtr->GetPi(), 0U);
    EXPECT_EQ(secondConnPtr->GetCi(), 0U);
    transport->connVec.clear();
}
