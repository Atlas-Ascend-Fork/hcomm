/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>
#include <stdio.h>
#include <chrono>
#include <memory>
#include <set>
#include <vector>

#define protected public
#define private public
#include "alltoallv_continuous_pipeline_pub.h"
#undef private
#undef protected

using namespace std;
using namespace hccl;

class AlltoallvContinuousPipelineTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AlltoallvContinuousPipelineTest Testcase SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "AlltoallvContinuousPipelineTest Testcase TearDown" << std::endl; }
    virtual void SetUp() { std::cout << "AlltoallvContinuousPipelineTest SetUP" << std::endl; }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "AlltoallvContinuousPipelineTest TearDown" << std::endl;
    }
};

// 等待的rank值已经就绪时，WaitValueOfRank应立即返回成功，不依赖超时时间
TEST_F(AlltoallvContinuousPipelineTest, wait_value_of_rank_success)
{
    HcclDispatcher disp;
    AlltoallvContinuousPipeline pipeline(disp);

    std::vector<u8> buffer(4096, 0);
    *reinterpret_cast<u32*>(buffer.data()) = 42;
    pipeline.inBuffer_ = DeviceMem::create(buffer.data(), buffer.size());
    pipeline.infoOffsets_.push_back(0);
    pipeline.flagAreaRefreshFlag_ = 1;

    u32 value = 0;
    HcclUs startTimeUs = std::chrono::steady_clock::now();
    HcclResult ret = pipeline.WaitValueOfRank(0, startTimeUs, value);

    EXPECT_EQ(HcclResult::HCCL_SUCCESS, ret);
    EXPECT_EQ(42u, value);
}

// 等待的rank值一直未就绪时，WaitValueOfRank应按waitFlagTimeoutSec_超时，而不是历史硬编码的1800s
TEST_F(AlltoallvContinuousPipelineTest, wait_value_of_rank_timeout)
{
    HcclDispatcher disp;
    AlltoallvContinuousPipeline pipeline(disp);

    std::vector<u8> buffer(4096, 0);
    pipeline.inBuffer_ = DeviceMem::create(buffer.data(), buffer.size());
    pipeline.infoOffsets_.push_back(0);
    pipeline.flagAreaRefreshFlag_ = 1;
    pipeline.waitFlagTimeoutSec_ = 1; // 超时时间设置为1s

    u32 value = 0;
    // 将起始时间设置为5s前，确保首轮轮询即超过1s超时
    HcclUs startTimeUs = std::chrono::steady_clock::now() - std::chrono::seconds(5);
    HcclResult ret = pipeline.WaitValueOfRank(0, startTimeUs, value);

    EXPECT_EQ(HcclResult::HCCL_E_TIMEOUT, ret);
}

// Prepare成功后，waitFlagTimeoutSec_应被设置为传入的超时时间+60s
TEST_F(AlltoallvContinuousPipelineTest, prepare_sets_wait_flag_timeout)
{
    HcclDispatcher disp;
    AlltoallvContinuousPipeline pipeline(disp);

    std::vector<u8> userIn(1024, 0);
    std::vector<u8> userOut(1024, 0);
    std::vector<u8> cclIn(65536, 0);
    std::vector<u8> cclOut(65536, 0);

    A2aPipelineMemory a2aMem;
    a2aMem.userInput = DeviceMem::create(userIn.data(), userIn.size());
    a2aMem.userOutput = DeviceMem::create(userOut.data(), userOut.size());
    a2aMem.cclInBuffer = DeviceMem::create(cclIn.data(), cclIn.size());
    a2aMem.cclOutBuffer = DeviceMem::create(cclOut.data(), cclOut.size());

    SubCommInfo level0Comm;
    level0Comm.localRank = 0;
    level0Comm.localRankSize = 2;
    SubCommInfo level1Comm;
    level1Comm.localRank = 0;
    level1Comm.localRankSize = 2;

    Stream mainStream;
    std::vector<Stream> subStream(2);
    std::vector<std::shared_ptr<LocalNotify>> notifyMain
        = {std::make_shared<LocalNotify>(), std::make_shared<LocalNotify>()};
    std::vector<std::shared_ptr<LocalNotify>> notifySub
        = {std::make_shared<LocalNotify>(), std::make_shared<LocalNotify>()};

    SendRecvInfo info;
    info.sendCounts.assign(4, 1);
    info.sendDispls.assign(4, 0);
    info.recvCounts.assign(4, 1);
    info.recvDispls.assign(4, 0);
    std::vector<SendRecvInfo> sendRecvInfoList = {info};

    const u32 timeOut = 100;
    HcclResult ret = pipeline.Prepare(
        0, a2aMem, level0Comm, level1Comm, mainStream, subStream, notifyMain, notifySub, sendRecvInfoList,
        HcclDataType::HCCL_DATA_TYPE_INT8, HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE, timeOut);

    EXPECT_EQ(HcclResult::HCCL_SUCCESS, ret);
    EXPECT_EQ(timeOut + 60u, pipeline.waitFlagTimeoutSec_);
}

// 布局模式按拓扑选择：M <= 3 走旧布局，M > 3 走槽位布局
TEST_F(AlltoallvContinuousPipelineTest, split_buffer_layout_by_topology)
{
    struct Case {
        u32 interRankSize;
        u32 intraRankSize;
        LayoutMode expectMode;
        u32 expectBlockNum;
    };
    const std::vector<Case> cases = {
        {2, 8, LayoutMode::RANK_INDEX, 32}, // M = 2 走乒乓：2N
        {3, 8, LayoutMode::RANK_INDEX, 24}, // M = 3：N
        {4, 8, LayoutMode::SLOT_INDEX, 24}, // M = 4：w×I
        {16, 8, LayoutMode::SLOT_INDEX, 24},
    };

    for (const auto& item : cases) {
        HcclDispatcher disp;
        AlltoallvContinuousPipeline pipeline(disp);
        pipeline.interRankSize_ = item.interRankSize;
        pipeline.intraRankSize_ = item.intraRankSize;
        pipeline.userRankSize_ = item.interRankSize * item.intraRankSize;
        pipeline.unitSize_ = 2;
        pipeline.rdmaConcurrentNum_ = 1;

        std::vector<u8> buffer(4 * 1024 * 1024, 0);
        pipeline.inBuffer_ = DeviceMem::create(buffer.data(), buffer.size());
        pipeline.outBuffer_ = DeviceMem::create(buffer.data(), buffer.size());

        const bool enablePingPong = pipeline.rdmaConcurrentNum_ >= item.interRankSize - 1;
        HcclResult ret = pipeline.SplitBuffer(enablePingPong);

        EXPECT_EQ(HcclResult::HCCL_SUCCESS, ret);
        EXPECT_EQ(item.expectMode, pipeline.layoutMode_);
        EXPECT_EQ(item.expectBlockNum, pipeline.blockNum_);
        EXPECT_EQ(item.expectBlockNum, pipeline.dataBlockOffsets_.size());
        EXPECT_EQ(item.expectBlockNum, pipeline.inBufferDataSize_.size());
        EXPECT_EQ(pipeline.userRankSize_, pipeline.infoOffsets_.size());
    }
}

// 容量不足时 SplitBuffer 返回 HCCL_E_MEMORY；刚好够用时应通过并满足 info 尾界
TEST_F(AlltoallvContinuousPipelineTest, split_buffer_size_bounds)
{
    HcclDispatcher disp;
    AlltoallvContinuousPipeline pipeline(disp);
    pipeline.interRankSize_ = 8;
    pipeline.intraRankSize_ = 8;
    pipeline.userRankSize_ = 64;
    pipeline.unitSize_ = 2;
    pipeline.rdmaConcurrentNum_ = 1;

    // 8N² + 128 × 24 = 32768 + 3072 = 35840
    std::vector<u8> tooSmall(35839, 0);
    pipeline.inBuffer_ = DeviceMem::create(tooSmall.data(), tooSmall.size());
    pipeline.outBuffer_ = DeviceMem::create(tooSmall.data(), tooSmall.size());
    EXPECT_EQ(HcclResult::HCCL_E_MEMORY, pipeline.SplitBuffer(false));

    std::vector<u8> enough(35840, 0);
    pipeline.inBuffer_ = DeviceMem::create(enough.data(), enough.size());
    pipeline.outBuffer_ = DeviceMem::create(enough.data(), enough.size());
    EXPECT_EQ(HcclResult::HCCL_SUCCESS, pipeline.SplitBuffer(false));
    EXPECT_EQ(0u, pipeline.infoOffsets_[0] % pipeline.sizePerBlock_);
    EXPECT_LE(pipeline.infoOffsets_[63] + 8 * 64, enough.size());
}

// 旧布局的块下标必须与改造前一致：乒乓叠加 loopIdx % 2
TEST_F(AlltoallvContinuousPipelineTest, resolve_block_index_legacy_layout)
{
    HcclDispatcher disp;
    AlltoallvContinuousPipeline pipeline(disp);
    pipeline.interRankSize_ = 2;
    pipeline.intraRankSize_ = 4;
    pipeline.userRankSize_ = 8;
    pipeline.layoutMode_ = LayoutMode::RANK_INDEX;
    pipeline.blockNum_ = 16;

    pipeline.enablePingPong_ = true;
    EXPECT_EQ(3u, pipeline.ResolveBlockIndex(3, 0, 100));
    EXPECT_EQ(11u, pipeline.ResolveBlockIndex(3, 1, 100));

    pipeline.enablePingPong_ = false;
    EXPECT_EQ(3u, pipeline.ResolveBlockIndex(3, 1, 100));
}

// 按 RunAsync 的三级 TaskState 推进规则重放，检查同刻存活的块是否踩同一槽位。
// in buffer 的访问来自 LocalCopy 写与 InterSend 读；out buffer 来自 Level0SdmaSend 写、
// Level0LocalCopy 读、InterSend 写、IntraDistribution 读。三项在 rdmaConcurrentNum_=1 时相差一步推进。
namespace {
struct StageState {
    u32 step{0};
    u32 next{0};
    u32 loop{0};
};

u32 CollectSlotAlias(const u32 interRankSize, const u32 loops, const AlltoallvContinuousPipeline& pipeline)
{
    const u32 stepsPerLoop = interRankSize - 1;
    StageState localCopy{0, std::min(1u, stepsPerLoop), 0};
    StageState inter{0, 0, 0};
    StageState intra{0, 0, 0};
    u32 repeatLoopNum = 0;
    u32 aliasCount = 0;
    u32 guard = 0;

    while ((localCopy.step < stepsPerLoop || inter.step < stepsPerLoop || intra.step < stepsPerLoop) && guard++ < 400) {
        const bool needDoLevel0SdmaWrite = (inter.next != 0 && inter.step == 0);
        const bool needDoLevel0LocalCopy = (intra.next != 0 && intra.step == 0);
        const bool needDoInterTasks = inter.step < inter.next;

        std::vector<u64> inSteps;
        std::vector<u64> outSteps;

        if (needDoLevel0SdmaWrite) {
            outSteps.push_back(pipeline.MakeGlobalStep(inter.loop, 0));
        }
        if (inter.loop == 0 && needDoInterTasks) {
            repeatLoopNum = loops > 0 ? loops - 1 : 0;
            if (localCopy.step == stepsPerLoop && localCopy.loop < repeatLoopNum) {
                ++localCopy.loop;
                localCopy.step = 0;
                localCopy.next = std::min(1u, stepsPerLoop);
            }
        }
        for (u32 step = intra.step + 1; step < intra.next + 1; ++step) {
            outSteps.push_back(pipeline.MakeGlobalStep(intra.loop, step));
        }
        for (u32 step = inter.step + 1; step < inter.next + 1; ++step) {
            const u64 globalStep = pipeline.MakeGlobalStep(inter.loop, step);
            inSteps.push_back(globalStep);
            outSteps.push_back(globalStep);
        }
        for (u32 step = localCopy.step + 1; step < localCopy.next + 1; ++step) {
            inSteps.push_back(pipeline.MakeGlobalStep(localCopy.loop, step));
        }
        if (needDoLevel0LocalCopy) {
            outSteps.push_back(pipeline.MakeGlobalStep(intra.loop, 0));
        }

        // 同一 buffer 内两个不同 globalStep 若落到同一槽位，总有一组 rank 使下标相同
        for (const auto* steps : {&inSteps, &outSteps}) {
            for (size_t a = 0; a < steps->size(); ++a) {
                for (size_t b = a + 1; b < steps->size(); ++b) {
                    if ((*steps)[a] == (*steps)[b]) {
                        continue;
                    }
                    if (pipeline.ResolveBlockIndex(0, 0, (*steps)[a])
                        == pipeline.ResolveBlockIndex(0, 0, (*steps)[b])) {
                        ++aliasCount;
                    }
                }
            }
        }

        const u32 oldIntraNext = intra.next;
        const u32 oldInterNext = inter.next;
        const u32 oldLocalNext = localCopy.next;
        intra.step = intra.next;
        inter.step = inter.next;
        localCopy.step = localCopy.next;
        intra.next = oldInterNext;
        inter.next = oldLocalNext;
        localCopy.next = std::min(oldLocalNext + 1, stepsPerLoop);

        if (intra.step == stepsPerLoop && intra.loop < inter.loop) {
            ++intra.loop;
            intra.step = 0;
            intra.next = std::min(1u, stepsPerLoop);
        }
        if (inter.step == stepsPerLoop && inter.loop < localCopy.loop) {
            ++inter.loop;
            inter.step = 0;
            inter.next = std::min(1u, stepsPerLoop);
        }
        if (localCopy.step == stepsPerLoop && localCopy.loop < repeatLoopNum) {
            ++localCopy.loop;
            localCopy.step = 0;
            localCopy.next = std::min(1u, stepsPerLoop);
        }
    }
    return aliasCount;
}
} // namespace

// 同刻存活的块在真实推进时序下不得别名：覆盖 M=4（step3 与 Level0 step0 同余）、
// M=6 与 M=9（M 能被 3 整除）、M=8（偶数且不被 3 整除）等容易出问题的拓扑
TEST_F(AlltoallvContinuousPipelineTest, slot_no_alias_under_pipeline_schedule)
{
    struct Topo {
        u32 interRankSize;
        u32 intraRankSize;
    };
    // M >= 4 才启用槽位布局，故从 4 开始
    const std::vector<Topo> topos = {{4, 4}, {5, 3}, {6, 4}, {7, 2}, {8, 2}, {9, 3}, {12, 2}, {16, 4}};

    for (const auto& topo : topos) {
        HcclDispatcher disp;
        AlltoallvContinuousPipeline pipeline(disp);
        pipeline.interRankSize_ = topo.interRankSize;
        pipeline.intraRankSize_ = topo.intraRankSize;
        pipeline.reuseGroupNum_ = SLOT_REUSE_NUM;
        pipeline.blockNum_ = pipeline.reuseGroupNum_ * pipeline.intraRankSize_;
        pipeline.layoutMode_ = LayoutMode::SLOT_INDEX;

        for (u32 loops = 1; loops <= 3; ++loops) {
            const u32 alias = CollectSlotAlias(topo.interRankSize, loops, pipeline);
            EXPECT_EQ(0u, alias) << "alias found at M=" << topo.interRankSize << " I=" << topo.intraRankSize
                                 << " loops=" << loops;
        }
    }
}

// 对照：组数不足时同一拓扑必须出现别名，证明上面的用例有判别力
TEST_F(AlltoallvContinuousPipelineTest, slot_alias_detected_with_two_groups)
{
    HcclDispatcher disp;
    AlltoallvContinuousPipeline pipeline(disp);
    pipeline.interRankSize_ = 6;
    pipeline.intraRankSize_ = 4;
    pipeline.reuseGroupNum_ = 2; // 故意取不足的组数
    pipeline.blockNum_ = pipeline.reuseGroupNum_ * pipeline.intraRankSize_;
    pipeline.layoutMode_ = LayoutMode::SLOT_INDEX;

    EXPECT_LT(0u, CollectSlotAlias(6, 2, pipeline));
}

// 任意 globalStep 与 rank 下，槽位下标都落在 [0, blockNum_)
TEST_F(AlltoallvContinuousPipelineTest, slot_index_in_range)
{
    HcclDispatcher disp;
    AlltoallvContinuousPipeline pipeline(disp);
    pipeline.interRankSize_ = 16;
    pipeline.intraRankSize_ = 8;
    pipeline.reuseGroupNum_ = SLOT_REUSE_NUM;
    pipeline.blockNum_ = pipeline.reuseGroupNum_ * pipeline.intraRankSize_;
    pipeline.layoutMode_ = LayoutMode::SLOT_INDEX;

    for (u64 globalStep = 0; globalStep < 200; ++globalStep) {
        for (u32 rank = 0; rank < 128; ++rank) {
            EXPECT_LT(pipeline.ResolveBlockIndex(rank, 0, globalStep), pipeline.blockNum_);
        }
    }
}

// out buffer 比 in buffer 小时也必须拒绝（尾界校验的右支）
TEST_F(AlltoallvContinuousPipelineTest, split_buffer_reject_smaller_out_buffer)
{
    HcclDispatcher disp;
    AlltoallvContinuousPipeline pipeline(disp);
    pipeline.interRankSize_ = 8;
    pipeline.intraRankSize_ = 8;
    pipeline.userRankSize_ = 64;
    pipeline.unitSize_ = 2;
    pipeline.rdmaConcurrentNum_ = 1;

    // in buffer 远大于 minBufferSize，确保旧的容量校验通过，只让 out 尾界校验生效。
    // N=64、K=24 时 G=32768、S=1280、infoEnd=63488
    const u64 infoEnd = 63488;
    std::vector<u8> inBuffer(64 * 1024, 0);

    // 差 1 字节：必须拒绝，且命中的是新加的 out 尾界校验
    std::vector<u8> outTooSmall(infoEnd - 1, 0);
    pipeline.inBuffer_ = DeviceMem::create(inBuffer.data(), inBuffer.size());
    pipeline.outBuffer_ = DeviceMem::create(outTooSmall.data(), outTooSmall.size());
    EXPECT_EQ(HcclResult::HCCL_E_MEMORY, pipeline.SplitBuffer(false));

    // 恰好等于 infoEnd：必须放行，锁定 '<' 而不是 '<=' 的边界语义
    std::vector<u8> outExact(infoEnd, 0);
    pipeline.outBuffer_ = DeviceMem::create(outExact.data(), outExact.size());
    EXPECT_EQ(HcclResult::HCCL_SUCCESS, pipeline.SplitBuffer(false));
    EXPECT_EQ(
        infoEnd, pipeline.sizePerBlock_ * pipeline.blockNum_ + 8 * pipeline.userRankSize_ * pipeline.userRankSize_);
}

// 在途长度元数据：旧布局按 rank 索引，新布局与数据偏移同键
TEST_F(AlltoallvContinuousPipelineTest, meta_index_matches_layout)
{
    HcclDispatcher disp;
    AlltoallvContinuousPipeline pipeline(disp);
    pipeline.interRankSize_ = 8;
    pipeline.intraRankSize_ = 4;
    pipeline.userRankSize_ = 32;

    pipeline.layoutMode_ = LayoutMode::RANK_INDEX;
    pipeline.enablePingPong_ = false;
    EXPECT_EQ(3u, pipeline.ResolveMetaIndex(3, 2, 100));

    // 乒乓时旧布局的元数据仍按 rank 索引，不能退化成 (loopIdx % 2) * N + rank
    pipeline.enablePingPong_ = true;
    EXPECT_EQ(3u, pipeline.ResolveMetaIndex(3, 1, 100));
    pipeline.enablePingPong_ = false;

    pipeline.layoutMode_ = LayoutMode::SLOT_INDEX;
    pipeline.reuseGroupNum_ = SLOT_REUSE_NUM;
    pipeline.blockNum_ = pipeline.reuseGroupNum_ * pipeline.intraRankSize_;
    const u64 globalStep = pipeline.MakeGlobalStep(2, 5); // M=8 -> globalStep=21，21 % 3 = 0 -> 组 0，组内位 3
    EXPECT_EQ(
        static_cast<u32>((globalStep % SLOT_REUSE_NUM) * pipeline.intraRankSize_ + 3),
        pipeline.ResolveMetaIndex(3, 2, globalStep));
    EXPECT_EQ(pipeline.ResolveBlockIndex(3, 2, globalStep), pipeline.ResolveMetaIndex(3, 2, globalStep));
}
