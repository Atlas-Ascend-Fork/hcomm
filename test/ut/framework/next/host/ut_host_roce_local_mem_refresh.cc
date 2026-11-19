/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <atomic>
#include <limits>
#include <thread>
#include "gtest/gtest.h"
#include "mockcpp/mockcpp.hpp"
#include "cpu_roce_endpoint.h"
#include "host/host_cpu_roce_channel.h"
#include "exchange_rdma_buffer_dto.h"

using namespace hcomm;

// CPU-only: aliases provide deterministic keys without registering hardware memory.
// Seed the manager record, then exercise the same Endpoint bookkeeping used by HcommMemReg.
class HostRoceLocalMemRefreshTest : public testing::Test {
protected:
    CpuRoceEndpoint endpoint{EndpointDesc{}};
    std::shared_ptr<RoceRegedMemMgr> mgr = std::make_shared<RoceRegedMemMgr>(nullptr);

    void SetUp() override { endpoint.localMemMgr_ = mgr; }
    void TearDown() override { GlobalMockObject::verify(); }

    std::shared_ptr<Hccl::LocalRdmaRmaBuffer> Register(uint64_t addr, uint64_t size, uint32_t key)
    {
        auto memory = std::make_shared<Hccl::Buffer>(addr, size, HcclMemType::HCCL_MEM_TYPE_HOST, "refresh");
        auto buffer = std::make_shared<Hccl::LocalRdmaRmaBuffer>(
            memory, reinterpret_cast<RdmaHandle>(1), key, key, reinterpret_cast<MrHandle>(2));
        mgr->handlesRecords_.push_back(buffer);
        EXPECT_EQ(endpoint.RecordLocalMemHandle(buffer.get()), HCCL_SUCCESS);
        return buffer;
    }

    void Attach(HostCpuRoceChannel& channel)
    {
        channel.local_.epPtr = &endpoint;
        Hccl::ExchangeRdmaBufferDto dto(0x8000, 0x1000, 77, "remote");
        channel.rmtRmaBuffers_.push_back(
            std::make_unique<Hccl::RemoteRdmaRmaBuffer>(reinterpret_cast<RdmaHandle>(1), dto));
        channel.remoteDpuNotifyIds_.push_back(3);
    }
};

TEST_F(HostRoceLocalMemRefreshTest, Ut_FindLocalBuffer_When_DefaultEndpoint_Expect_NotSupport)
{
    std::shared_ptr<Hccl::LocalRmaBuffer> found = Register(0x1000, 0x1000, 11);
    EXPECT_EQ(endpoint.Endpoint::FindLocalBuffer(0x1000, 1, found), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(found, nullptr);
}

TEST_F(HostRoceLocalMemRefreshTest, Ut_FindLocalBuffer_When_BoundaryOrOverflow_Expect_FullCoverageOnly)
{
    auto registered = Register(0x1000, 0x1000, 11);
    std::shared_ptr<Hccl::LocalRmaBuffer> found;
    Endpoint& base = endpoint;
    EXPECT_EQ(base.FindLocalBuffer(0x1000, 0x1000, found), HCCL_SUCCESS);
    EXPECT_EQ(found.get(), registered.get());
    EXPECT_EQ(base.FindLocalBuffer(0x1fff, 1, found), HCCL_SUCCESS);
    EXPECT_EQ(base.FindLocalBuffer(0x1fff, 2, found), HCCL_E_NOT_FOUND);
    EXPECT_EQ(found, nullptr);
    EXPECT_EQ(base.FindLocalBuffer(0x0fff, 2, found), HCCL_E_NOT_FOUND);
    EXPECT_EQ(base.FindLocalBuffer(0x2000, 1, found), HCCL_E_NOT_FOUND);
    EXPECT_EQ(base.FindLocalBuffer(0x1001, std::numeric_limits<uint64_t>::max(), found), HCCL_E_NOT_FOUND);
    EXPECT_EQ(base.FindLocalBuffer(std::numeric_limits<uint64_t>::max(), 2, found), HCCL_E_NOT_FOUND);
    // Preserve the existing query convention for zero-length WRs at the end of a registered range.
    EXPECT_EQ(base.FindLocalBuffer(0x2000, 0, found), HCCL_SUCCESS);
}

TEST_F(HostRoceLocalMemRefreshTest, Ut_FindLocalBuffer_When_OtherEndpointOwnsMemory_Expect_NotFound)
{
    auto registered = Register(0x1000, 0x1000, 11);
    CpuRoceEndpoint other{EndpointDesc{}};
    other.localMemMgr_ = mgr;
    std::shared_ptr<Hccl::LocalRmaBuffer> found;
    EXPECT_EQ(other.FindLocalBuffer(0x1000, 1, found), HCCL_E_NOT_FOUND);
    ASSERT_EQ(other.RecordLocalMemHandle(registered.get()), HCCL_SUCCESS);
    ASSERT_EQ(endpoint.RemoveLocalMemHandle(registered.get()), HCCL_SUCCESS);
    EXPECT_EQ(endpoint.FindLocalBuffer(0x1000, 1, found), HCCL_E_NOT_FOUND);
    EXPECT_EQ(other.FindLocalBuffer(0x1000, 1, found), HCCL_SUCCESS);
}

TEST_F(HostRoceLocalMemRefreshTest, Ut_FindLocalBuffer_When_AliasesRemoved_Expect_RemainingCoverage)
{
    auto parent = Register(0x1000, 0x1000, 11);
    auto sameBase = Register(0x1000, 0x100, 11);
    auto child = Register(0x1200, 0x100, 11);
    std::shared_ptr<Hccl::LocalRmaBuffer> found;
    EXPECT_EQ(endpoint.FindLocalBuffer(0x1800, 0x100, found), HCCL_SUCCESS);
    ASSERT_EQ(endpoint.RemoveLocalMemHandle(parent.get()), HCCL_SUCCESS);
    EXPECT_EQ(endpoint.FindLocalBuffer(0x1800, 0x100, found), HCCL_E_NOT_FOUND);
    EXPECT_EQ(endpoint.FindLocalBuffer(0x1000, 0x100, found), HCCL_SUCCESS);
    EXPECT_EQ(endpoint.FindLocalBuffer(0x1200, 0x100, found), HCCL_SUCCESS);
    ASSERT_EQ(endpoint.RemoveLocalMemHandle(sameBase.get()), HCCL_SUCCESS);
    ASSERT_EQ(endpoint.RemoveLocalMemHandle(child.get()), HCCL_SUCCESS);
    EXPECT_EQ(endpoint.FindLocalBuffer(0x1200, 1, found), HCCL_E_NOT_FOUND);
    auto replacement = Register(0x1000, 0x1000, 99);
    ASSERT_EQ(endpoint.FindLocalBuffer(0x1200, 1, found), HCCL_SUCCESS);
    EXPECT_EQ(found.get(), replacement.get());
}

TEST_F(HostRoceLocalMemRefreshTest, Ut_FindLocalBuffer_When_IdenticalRanges_Expect_IndependentHandles)
{
    auto first = Register(0x1000, 0x1000, 11);
    auto second = Register(0x1000, 0x1000, 11);
    ASSERT_EQ(endpoint.RemoveLocalMemHandle(first.get()), HCCL_SUCCESS);
    std::shared_ptr<Hccl::LocalRmaBuffer> found;
    ASSERT_EQ(endpoint.FindLocalBuffer(0x1000, 1, found), HCCL_SUCCESS);
    EXPECT_EQ(found.get(), second.get());
    ASSERT_EQ(endpoint.RemoveLocalMemHandle(second.get()), HCCL_SUCCESS);
    EXPECT_EQ(endpoint.FindLocalBuffer(0x1000, 1, found), HCCL_E_NOT_FOUND);
    EXPECT_EQ(found, nullptr);
}

TEST_F(HostRoceLocalMemRefreshTest, Ut_FindLocalBuffer_When_OldChannels_Expect_NewRegistrationVisible)
{
    auto original = Register(0x1000, 0x1000, 11);
    HostCpuRoceChannel first(nullptr, HcommChannelDesc{}, COMM_ENGINE_RESERVED);
    HostCpuRoceChannel second(nullptr, HcommChannelDesc{}, COMM_ENGINE_RESERVED);
    Attach(first);
    Attach(second);
    first.localRmaBuffers_.push_back(original.get());
    second.localRmaBuffers_.push_back(original.get());
    auto added = Register(0x3000, 0x1000, 33);
    for (auto* channel : {&first, &second}) {
        std::shared_ptr<Hccl::LocalRdmaRmaBuffer> found;
        ASSERT_EQ(channel->FindLocalBuffer(0x3100, 0x100, found), HCCL_SUCCESS);
        EXPECT_EQ(found.get(), added.get());
        EXPECT_EQ(channel->localRmaBuffers_.size(), 1U);
        ibv_send_wr wr{};
        ibv_sge sg{};
        for (auto opcode : {IBV_WR_RDMA_WRITE, IBV_WR_RDMA_READ}) {
            // 11 参形态：sendFlags/immData 由调用方逐 WR 计算（PostRdmaMemOp 切片语义）；
            // 本用例聚焦 lkey 刷新可见性，标志与 imm 传 0
            channel->BuildRdmaWr(
                "refresh", opcode, reinterpret_cast<void*>(0x3100), reinterpret_cast<void*>(0x8100), 0x100,
                found->GetLkey(), 0, 0, 0, wr, sg);
            EXPECT_EQ(wr.sg_list->lkey, 33U);
            EXPECT_EQ(wr.wr.rdma.rkey, 77U);
            EXPECT_EQ(wr.opcode, opcode);
        }
    }
}

TEST_F(HostRoceLocalMemRefreshTest, Ut_WriteWithNotify_When_NoLocalSnapshot_Expect_CurrentKeyOnAllSlices)
{
    HostCpuRoceChannel channel(nullptr, HcommChannelDesc{}, COMM_ENGINE_RESERVED);
    Attach(channel);
    channel.maxMsgSize_ = 0x80; // 按 maxMsgSize_ 切片：0x100 → 前块 WRITE + 尾块 WRITE_WITH_IMM
    channel.wqeNums_ = {0};
    auto added = Register(0x3000, 0x1000, 33);
    std::vector<Hccl::QpInfo> qpInfos(1);
    MOCKER_CPP(&HostCpuRoceChannel::GetQpInfos).stubs().will(returnValue(qpInfos));
    // 拦截点为 PostAndCheckSend（内部 ibv_post_send + wqeNums_ 自增均被 mock 短路）
    MOCKER_CPP(&HostCpuRoceChannel::PostAndCheckSend)
        .expects(exactly(2))
        .will(invoke(+[](HostCpuRoceChannel*, ibv_qp*, uint32_t, const char*, ibv_send_wr& wr) -> HcclResult {
            EXPECT_EQ(wr.sg_list->lkey, 33U);
            EXPECT_EQ(wr.sg_list->length, 0x80U);
            EXPECT_EQ(wr.wr.rdma.rkey, 77U);
            if (wr.sg_list->addr == 0x3100U) {
                // 前块：PostRdmaOpAllQps → PostRdmaMemOp 中间片，非尾片且 fenceFlag_ 默认 false → 无标志
                EXPECT_EQ(wr.wr.rdma.remote_addr, 0x8100U);
                EXPECT_EQ(wr.opcode, IBV_WR_RDMA_WRITE);
                EXPECT_EQ(wr.send_flags, 0U);
            } else {
                // 尾块：PostRdmaMemOp 尾片 WRITE_WITH_IMM，imm=remoteDpuNotifyIds_[0]=3，SIGNALED|FENCE
                EXPECT_EQ(wr.sg_list->addr, 0x3180U);
                EXPECT_EQ(wr.wr.rdma.remote_addr, 0x8180U);
                EXPECT_EQ(wr.opcode, IBV_WR_RDMA_WRITE_WITH_IMM);
                EXPECT_EQ(wr.imm_data, 3U);
                EXPECT_EQ(
                    wr.send_flags,
                    static_cast<unsigned int>(IBV_SEND_SIGNALED) | static_cast<unsigned int>(IBV_SEND_FENCE));
            }
            return HCCL_SUCCESS;
        }));
    ASSERT_TRUE(channel.localRmaBuffers_.empty());
    ASSERT_EQ(
        channel.WriteWithNotify(reinterpret_cast<void*>(0x8100), reinterpret_cast<void*>(0x3100), 0x100, 0),
        HCCL_SUCCESS);
    // wqeNums_ 自增点在被 mock 的 PostAndCheckSend 内，不执行；“两次下发”由 exactly(2) 承载
    EXPECT_EQ(channel.wqeNums_[0], 0);
    ASSERT_EQ(endpoint.RemoveLocalMemHandle(added.get()), HCCL_SUCCESS);
    EXPECT_EQ(
        channel.WriteWithNotify(reinterpret_cast<void*>(0x8100), reinterpret_cast<void*>(0x3100), 0x100, 0),
        HCCL_E_NOT_FOUND);
}

TEST_F(HostRoceLocalMemRefreshTest, Ut_FindLocalBuffer_When_NullEndpoint_Expect_ErrorAndEmptyOutput)
{
    HostCpuRoceChannel channel(nullptr, HcommChannelDesc{}, COMM_ENGINE_RESERVED);
    auto found = Register(0x1000, 0x1000, 11);
    EXPECT_NE(channel.FindLocalBuffer(0x1000, 1, found), HCCL_SUCCESS);
    EXPECT_EQ(found, nullptr);
}

TEST_F(HostRoceLocalMemRefreshTest, Ut_FindLocalBuffer_When_RecordRemoved_Expect_ResultObjectKeptAlive)
{
    auto record = Register(0x1000, 0x1000, 11);
    std::weak_ptr<Hccl::LocalRmaBuffer> weak = record;
    std::shared_ptr<Hccl::LocalRmaBuffer> found;
    ASSERT_EQ(endpoint.FindLocalBuffer(0x1000, 1, found), HCCL_SUCCESS);
    ASSERT_EQ(endpoint.RemoveLocalMemHandle(record.get()), HCCL_SUCCESS);
    mgr->handlesRecords_.clear();
    record.reset();
    EXPECT_FALSE(weak.expired());
    found.reset();
    EXPECT_TRUE(weak.expired()); // Only object lifetime is asserted, not in-flight hardware MR lifetime.
}

TEST_F(HostRoceLocalMemRefreshTest, Ut_GetAll_When_RegAfterGetAll_Expect_HandleStaysValid)
{
    auto first = Register(0x1000, 0x1000, 11);
    void* handles = nullptr;
    uint32_t num = 0;
    ASSERT_EQ(endpoint.GetAllLocalMemHandles(&handles, &num), HCCL_SUCCESS);
    ASSERT_EQ(num, 1U);
    auto* records = static_cast<std::shared_ptr<Hccl::LocalRdmaRmaBuffer>*>(handles);
    EXPECT_EQ(records[0].get(), first.get());

    // keep-alive：Reg 发布新一代后，旧出参数组仍可读（原 shared_lock 契约下可能已因扩容搬移失效）
    Register(0x3000, 0x1000, 33);
    EXPECT_EQ(records[0].get(), first.get());

    // 契约边界：下次 GetAll 覆盖 keep-alive（旧指针从此不再触碰）
    void* handles2 = nullptr;
    uint32_t num2 = 0;
    ASSERT_EQ(endpoint.GetAllLocalMemHandles(&handles2, &num2), HCCL_SUCCESS);
    EXPECT_EQ(num2, 2U);
}

// 快照隔离下并发 Reg/Remove ↔ Find 为契约内安全场景：读者每次只可能看到某个完整发布过的代，
// 断言返回句柄必属于已知集合（垃圾指针不可能出现），顺带压测 atomic_load 桶锁与引用计数
TEST_F(HostRoceLocalMemRefreshTest, Ut_FindLocalBuffer_When_RecordsChangeConcurrently_Expect_ConsistentLookup)
{
    auto stable = Register(0x1000, 0x1000, 11);
    auto changing = Register(0x3000, 0x1000, 33);
    ASSERT_EQ(endpoint.RemoveLocalMemHandle(changing.get()), HCCL_SUCCESS);
    std::atomic<bool> failed{false};
    std::thread reader([&] {
        for (size_t i = 0; i < 1000; ++i) {
            std::shared_ptr<Hccl::LocalRmaBuffer> found;
            if (endpoint.FindLocalBuffer(0x1000, 1, found) != HCCL_SUCCESS || found.get() != stable.get()) {
                failed = true;
            }
            auto ret = endpoint.FindLocalBuffer(0x3000, 1, found);
            if ((ret == HCCL_SUCCESS && found.get() != changing.get()) || (ret == HCCL_E_NOT_FOUND && found != nullptr)
                || (ret != HCCL_SUCCESS && ret != HCCL_E_NOT_FOUND)) {
                failed = true;
            }
        }
    });
    for (size_t i = 0; i < 1000; ++i) {
        if (endpoint.RecordLocalMemHandle(changing.get()) != HCCL_SUCCESS
            || endpoint.RemoveLocalMemHandle(changing.get()) != HCCL_SUCCESS) {
            failed = true;
        }
    }
    reader.join();
    EXPECT_FALSE(failed);
}
