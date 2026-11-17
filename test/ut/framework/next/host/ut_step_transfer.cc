/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>
#include <algorithm>
#include <cstring>
#include <vector>

#include "ip_address.h"
#include "socket.h"
#include "binary_stream.h"
#include "channel.h"
#include "step_transfer.h"

using namespace hcomm;

namespace {
// ===== I* 序列桩：mockcpp returnValue 不写引用出参，进度须 invoke 显式回填 =====
// g_sentPlan/g_recvPlan 为每次调用的预设进度（0 表示本步零进度返回 E_AGAIN），
// 计划耗尽后默认全量完成；RECV_SIZE 阶段（size==sizeof(u64)）按 g_remoteSizePlan 回填对端声明尺寸
static std::vector<u64> g_sentPlan;
static std::vector<u64> g_recvPlan;
static size_t g_sentIdx = 0;
static size_t g_recvIdx = 0;
static u64 g_remoteSizePlan = 8;
static int g_tryCallCnt = 0;

static void ResetStubState()
{
    g_sentPlan.clear();
    g_recvPlan.clear();
    g_sentIdx = 0;
    g_recvIdx = 0;
    g_remoteSizePlan = 8;
    g_tryCallCnt = 0;
}

static HcclResult ISendSeqStub(Hccl::Socket*, const void*, u64 size, u64& sentSize)
{
    g_tryCallCnt++;
    u64 plan = (g_sentIdx < g_sentPlan.size()) ? g_sentPlan[g_sentIdx++] : size;
    plan = std::min(plan, size);
    if (plan == 0) {
        sentSize = 0;
        return HCCL_E_AGAIN;
    }
    sentSize = plan;
    return HCCL_SUCCESS;
}

static HcclResult IRecvSeqStub(Hccl::Socket*, void* buf, u64 size, u64& recvSize)
{
    g_tryCallCnt++;
    u64 plan = (g_recvIdx < g_recvPlan.size()) ? g_recvPlan[g_recvIdx++] : size;
    plan = std::min(plan, size);
    if (plan == 0) {
        recvSize = 0;
        return HCCL_E_AGAIN;
    }
    if (size == sizeof(u64)) { // RECV_SIZE 阶段：回填对端声明尺寸
        u64 remoteSize = g_remoteSizePlan;
        memcpy_s(buf, sizeof(u64), &remoteSize, sizeof(remoteSize));
    } else {
        memset(buf, 0xAB, plan); // 数据阶段内容不参与断言（解包桩不解析）
    }
    recvSize = plan;
    return HCCL_SUCCESS;
}
} // namespace

class StepTransferTest : public ::testing::Test {
protected:
    void SetUp()
    {
        ResetStubState();
        MOCKER_CPP(&Hccl::Socket::ISend).stubs().will(invoke(ISendSeqStub));
        MOCKER_CPP(&Hccl::Socket::IRecv).stubs().will(invoke(IRecvSeqStub));
        Hccl::IpAddress localIp("1.0.0.0");
        Hccl::IpAddress remoteIp("2.0.0.0");
        socket = new Hccl::Socket(
            nullptr, localIp, 60001, remoteIp, "_0_1_", Hccl::SocketRole::SERVER, Hccl::NicType::HOST_NIC_TYPE);
    }
    void TearDown()
    {
        GlobalMockObject::verify();
        delete socket;
    }

    // 打包桩：写入 4 字节数据（sendBuf_ 非空）并计数
    static void MakePackStub(int& packCnt, std::function<HcclResult(Hccl::BinaryStream&)>& packFn)
    {
        packFn = [&packCnt](Hccl::BinaryStream& bs) -> HcclResult {
            packCnt++;
            bs << (u32)1;
            return HCCL_SUCCESS;
        };
    }

    Hccl::Socket* socket;
    StepTransfer transfer;
};

// 拆步核心语义：部分写出时偏移保留、重入续传整块完成并归零（可复用于下一块）
TEST_F(StepTransferTest, Ut_StepSend_When_PartialSent_Expect_ProgressKeptAndResume)
{
    g_sentPlan = {4, 0, 4}; // 部分写出 4 字节 → 零进度 EAGAIN → 剩余 4 字节写完
    u8 buf[8] = {0};

    HcclResult ret1 = transfer.StepSend(socket, buf, sizeof(buf));
    EXPECT_EQ(ret1, HCCL_E_AGAIN);
    EXPECT_EQ(transfer.sendOff_, 4U); // 部分进度保留

    HcclResult ret2 = transfer.StepSend(socket, buf, sizeof(buf));
    EXPECT_EQ(ret2, HCCL_E_AGAIN);
    EXPECT_EQ(transfer.sendOff_, 4U); // 零进度拍偏移不丢

    HcclResult ret3 = transfer.StepSend(socket, buf, sizeof(buf));
    EXPECT_EQ(ret3, HCCL_SUCCESS);
    EXPECT_EQ(transfer.sendOff_, 0U); // 整块完成自动归零
}

// 拆步核心语义：接收侧部分读入时偏移保留、重入续传整块完成
TEST_F(StepTransferTest, Ut_StepRecv_When_PartialRecv_Expect_ProgressKeptAndResume)
{
    g_recvPlan = {4, 0, 4};
    u8 buf[8] = {0};

    HcclResult ret1 = transfer.StepRecv(socket, buf, sizeof(buf));
    EXPECT_EQ(ret1, HCCL_E_AGAIN);
    EXPECT_EQ(transfer.recvOff_, 4U);

    HcclResult ret2 = transfer.StepRecv(socket, buf, sizeof(buf));
    EXPECT_EQ(ret2, HCCL_E_AGAIN);
    EXPECT_EQ(transfer.recvOff_, 4U);

    HcclResult ret3 = transfer.StepRecv(socket, buf, sizeof(buf));
    EXPECT_EQ(ret3, HCCL_SUCCESS);
    EXPECT_EQ(transfer.recvOff_, 0U);
}

// 调用方中途改小 size 的误用：偏移越界返回 E_PARA，不静默吞进度谎报成功（检视意见 #199771816）
TEST_F(StepTransferTest, Ut_StepSend_When_OffsetExceedsSize_Expect_EPara)
{
    transfer.sendOff_ = 16; // 模拟上一拍遗留偏移大于当前块
    u8 buf[8] = {0};
    EXPECT_EQ(transfer.StepSend(socket, buf, sizeof(buf)), HCCL_E_PARA);
    EXPECT_EQ(transfer.sendOff_, 16U); // 越界报错不清偏移，错误状态保留供定位

    transfer.recvOff_ = 16;
    EXPECT_EQ(transfer.StepRecv(socket, buf, sizeof(buf)), HCCL_E_PARA);
    EXPECT_EQ(transfer.recvOff_, 16U);
}

// 四阶段直落：socket 全量进度时单次调用走完 SEND_SIZE→RECV_SIZE→SEND_DATA→RECV_DATA，
// packFn/unpackFn 各恰好执行一次
TEST_F(StepTransferTest, Ut_StepExchange_When_SocketFast_Expect_AllPhasesInOneCall)
{
    int packCnt = 0;
    int unpackCnt = 0;
    std::function<HcclResult(Hccl::BinaryStream&)> packFn;
    MakePackStub(packCnt, packFn);
    auto unpackFn = [&unpackCnt](Hccl::BinaryStream&) -> HcclResult {
        unpackCnt++;
        return HCCL_SUCCESS;
    };
    g_remoteSizePlan = 8;

    HcclResult ret = transfer.StepExchange(socket, packFn, unpackFn);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(packCnt, 1);   // 打包恰一次
    EXPECT_EQ(unpackCnt, 1); // 解包恰一次
    EXPECT_EQ(transfer.phase_, StepTransfer::ExchangePhase::DONE);
}

// 四阶段续传：部分进度返回 E_AGAIN，phase_/偏移跨调用保留不回退，直至完成且解包恰一次
TEST_F(StepTransferTest, Ut_StepExchange_When_PartialProgress_Expect_PhaseKeptResume)
{
    int packCnt = 0;
    int unpackCnt = 0;
    std::function<HcclResult(Hccl::BinaryStream&)> packFn;
    MakePackStub(packCnt, packFn);
    auto unpackFn = [&unpackCnt](Hccl::BinaryStream&) -> HcclResult {
        unpackCnt++;
        return HCCL_SUCCESS;
    };
    g_sentPlan = {4, 4, 2, 2}; // 尺寸 8 字节两步 + 数据 4 字节两步
    g_recvPlan = {8, 2, 2, 4}; // 尺寸单步收满（回填对端尺寸），数据 8 字节三步
    g_remoteSizePlan = 8;

    auto lastPhase = StepTransfer::ExchangePhase::SEND_SIZE;
    for (int call = 1; call <= 4; call++) { // 4 次 E_AGAIN 后第 5 次完成
        HcclResult ret = transfer.StepExchange(socket, packFn, unpackFn);
        EXPECT_EQ(ret, HCCL_E_AGAIN);
        // phase_ 单调推进不回退，解包未发生
        ASSERT_GE(static_cast<u32>(transfer.phase_), static_cast<u32>(lastPhase));
        lastPhase = transfer.phase_;
        EXPECT_EQ(unpackCnt, 0);
    }

    HcclResult ret = transfer.StepExchange(socket, packFn, unpackFn);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(packCnt, 1);
    EXPECT_EQ(unpackCnt, 1);
    EXPECT_EQ(transfer.phase_, StepTransfer::ExchangePhase::DONE);
}

// DONE 重入幂等（成功终态语义）：解包成功进 DONE 后再次调用返回 SUCCESS，不重复解包、
// 无 socket 收发——完成后不自动 Reset，DONE 短路即是对意外重入的正确防御
TEST_F(StepTransferTest, Ut_StepExchange_When_DoneReentry_Expect_NoReUnpack)
{
    int packCnt = 0;
    int unpackCnt = 0;
    std::function<HcclResult(Hccl::BinaryStream&)> packFn;
    MakePackStub(packCnt, packFn);
    auto unpackFn = [&unpackCnt](Hccl::BinaryStream&) -> HcclResult {
        unpackCnt++;
        return HCCL_SUCCESS;
    };

    EXPECT_EQ(transfer.StepExchange(socket, packFn, unpackFn), HCCL_SUCCESS);
    EXPECT_EQ(unpackCnt, 1);
    int tryCntBeforeReentry = g_tryCallCnt;

    EXPECT_EQ(transfer.StepExchange(socket, packFn, unpackFn), HCCL_SUCCESS);
    EXPECT_EQ(unpackCnt, 1);                      // 不重复解包
    EXPECT_EQ(g_tryCallCnt, tryCntBeforeReentry); // 重入无收发发生
}

// 解包失败进 FAILED 终态：返回原错误码，重入仍返回同一错误码（不吞错、不假成功），
// 且重入不触发 socket 收发（数据已收齐，不重收）
TEST_F(StepTransferTest, Ut_StepExchange_When_UnpackFailed_Expect_ErrorKeptOnReentry)
{
    int packCnt = 0;
    int unpackCnt = 0;
    std::function<HcclResult(Hccl::BinaryStream&)> packFn;
    MakePackStub(packCnt, packFn);
    auto unpackFn = [&unpackCnt](Hccl::BinaryStream&) -> HcclResult {
        unpackCnt++;
        return HCCL_E_PARA; // 模拟对端数据畸形（解包校验失败）
    };

    HcclResult ret1 = transfer.StepExchange(socket, packFn, unpackFn);
    EXPECT_EQ(ret1, HCCL_E_PARA);
    EXPECT_EQ(transfer.phase_, StepTransfer::ExchangePhase::FAILED);
    EXPECT_EQ(transfer.failureCode_, HCCL_E_PARA);
    int tryCntBeforeReentry = g_tryCallCnt;

    HcclResult ret2 = transfer.StepExchange(socket, packFn, unpackFn); // 重入不吞错
    EXPECT_EQ(ret2, HCCL_E_PARA);
    EXPECT_EQ(transfer.phase_, StepTransfer::ExchangePhase::FAILED);
    EXPECT_EQ(g_tryCallCnt, tryCntBeforeReentry); // 失败重入不重收数据
    EXPECT_EQ(unpackCnt, 1);                      // 不重复解包
}

// 解包失败后显式 Reset：回到 SEND_SIZE 初态，可重新完整交换并成功——Reset 的复位语义用例
TEST_F(StepTransferTest, Ut_StepExchange_When_UnpackFailedThenReset_Expect_FullReExchange)
{
    int packCnt = 0;
    int unpackCnt = 0;
    std::function<HcclResult(Hccl::BinaryStream&)> packFn;
    MakePackStub(packCnt, packFn);
    auto failUnpack = [](Hccl::BinaryStream&) -> HcclResult {
        return HCCL_E_PARA;
    };

    EXPECT_EQ(transfer.StepExchange(socket, packFn, failUnpack), HCCL_E_PARA);
    EXPECT_EQ(transfer.phase_, StepTransfer::ExchangePhase::FAILED);

    transfer.Reset(); // 显式复位（复用场景的正确入口）
    EXPECT_EQ(transfer.phase_, StepTransfer::ExchangePhase::SEND_SIZE);
    EXPECT_EQ(transfer.failureCode_, HCCL_E_INTERNAL);
    EXPECT_EQ(transfer.sendBuf_.size(), 0U); // 打包产物随复位清空，重新打包

    auto okUnpack = [&unpackCnt](Hccl::BinaryStream&) -> HcclResult {
        unpackCnt++;
        return HCCL_SUCCESS;
    };
    ResetStubState(); // 清桩计划状态，复位后重新完整交换
    HcclResult ret = transfer.StepExchange(socket, packFn, okUnpack);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(packCnt, 2); // 复位后重新打包
    EXPECT_EQ(unpackCnt, 1);
    EXPECT_EQ(transfer.phase_, StepTransfer::ExchangePhase::DONE);
}

// 防实现回退：解包执行时 phase_ 不得为 DONE（DONE 只能由解包成功置位——先解包后终态）
TEST_F(StepTransferTest, Ut_StepExchange_When_Unpacking_Expect_PhaseNotDone)
{
    int packCnt = 0;
    std::function<HcclResult(Hccl::BinaryStream&)> packFn;
    MakePackStub(packCnt, packFn);
    auto unpackFn = [this](Hccl::BinaryStream&) -> HcclResult {
        EXPECT_NE(transfer.phase_, StepTransfer::ExchangePhase::DONE); // 解包时刻未置 DONE
        return HCCL_SUCCESS;
    };

    EXPECT_EQ(transfer.StepExchange(socket, packFn, unpackFn), HCCL_SUCCESS);
    EXPECT_EQ(transfer.phase_, StepTransfer::ExchangePhase::DONE); // 成功后才置位
}

// 对端声明尺寸超上限：返回 HCCL_E_PARA 且不按超限值扩容接收缓冲（防 OOM），phase_ 停在 RECV_SIZE
TEST_F(StepTransferTest, Ut_StepExchange_When_RemoteSizeExceedLimit_Expect_EPara)
{
    int packCnt = 0;
    int unpackCnt = 0;
    std::function<HcclResult(Hccl::BinaryStream&)> packFn;
    MakePackStub(packCnt, packFn);
    auto unpackFn = [&unpackCnt](Hccl::BinaryStream&) -> HcclResult {
        unpackCnt++;
        return HCCL_SUCCESS;
    };
    g_remoteSizePlan = Channel::MAX_EXCHANGE_DATA_SIZE + 1;

    HcclResult ret = transfer.StepExchange(socket, packFn, unpackFn);
    EXPECT_EQ(ret, HCCL_E_PARA);
    EXPECT_EQ(transfer.recvBuf_.capacity(), 0U);                        // 未按超限值分配
    EXPECT_EQ(transfer.phase_, StepTransfer::ExchangePhase::RECV_SIZE); // 状态不推进
    EXPECT_EQ(unpackCnt, 0);
}
