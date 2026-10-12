/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * You may not use this file except in compliance with the License.
 * A copy of the License is located at ...
 * See the License for the specific language governing permissions and limitations under the License.
 */

// GetChannelTaskContext 成功路径的 Communicator 查询次数回归（自 data_op_device_stub_test.cc
// 拆分出的 Fake 引擎用例：以 ConformanceMemoryEngine 的查询次数观察点断言，不依赖任何
// 真实后端——留在公共目录，随全部源码形态构建执行；真实 SQLite fixture 的同名测试目标
// 在 test/device_arm/sqlite/）。
//   rank_size 边界校验复用 GetThreadTaskPosition 取得的通信域行，同一 commId 只查询
//   一次（修复前为两次）。断言三张资源表各恰好 1 次 Select，且解析结果
//   （rankId / remoteRankId）不变。
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

#include "runtime_state/db_sim_runner_ops.h"
#include "storage/internal/storage_runtime_impl.h"
#include "storage/storage_runtime.h"
#include "storage/table_access.h"
#include "../storage/conformance_memory_engine.h"
#include "../storage/test_fake_backend.h"

// 将实现纳入当前翻译单元（与 data_op_device_stub_test.cc 同一方式）
#include "../../src/device_arm/level1/data_op_device_stub.cc"

namespace {
const char* kTestCommName = "test_comm";
} // namespace

TEST(DataOpChannelContextFake, ChannelTaskContextQueriesCommunicatorExactlyOnce)
{
    using storage_conformance::ConformanceMemoryEngine;

    auto& runtime = HcclSim::Storage::detail::StorageRuntimeImpl::Instance();
    runtime.ResetForTest();
    ConformanceMemoryEngine* engine = nullptr;
    runtime.RegisterEngineFactory(
        HcclSim::Storage::kTestFakeBackendName, [&engine](const HcclSim::Storage::LogicalDatabaseRuntimeConfig&) {
            auto created = ConformanceMemoryEngine::Create(&HcclSim::Storage::Ddl::kRunnerDatabase);
            if (created.ok() && created.value.has_value()) {
                engine = static_cast<ConformanceMemoryEngine*>(created.value->get());
            }
            return created;
        });
    runtime.UseDatabase(HcclSim::Storage::Ddl::kRunnerDatabase);
    const HcclSim::Storage::StorageRuntimeConfig config{
        {{std::string{"RUNNER_DB"}, HcclSim::Storage::kTestFakeBackendName, "memory://runnerdb"}}};
    ASSERT_TRUE(HcclSim::Storage::StorageRuntime::EnsureProcessSession(config).ok());

    // 种子：通信域 self(rank1)，rank_size=3（remoteRank=2 在界内）；线程与已连接通道。
    sim::runtime::Communicator self{};
    std::strncpy(self.comm_id, kTestCommName, sizeof(self.comm_id) - 1);
    self.rank_size = 3;
    self.rank_id = 1;
    ASSERT_TRUE(HcclSim::Storage::Insert<sim::runtime::Communicator>(self).ok());
    ASSERT_NE(engine, nullptr); // 引擎按配置惰性创建：首次落库后必须可观察

    sim::runtime::HcclThread thread{};
    thread.commId = self.id;
    thread.streamId = 7;
    ASSERT_TRUE(HcclSim::Storage::Insert<sim::runtime::HcclThread>(thread).ok());
    ASSERT_GT(thread.id, 0U);

    sim::runtime::HcclChannel channel{};
    channel.commId = self.id;
    channel.remoteRankId = 2;
    channel.status = true;
    ASSERT_TRUE(HcclSim::Storage::Insert<sim::runtime::HcclChannel>(channel).ok());
    ASSERT_GT(channel.id, 0U);

    uint32_t savedRankId = 0;
    ASSERT_EQ(GetCurRankId(&savedRankId), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
    ASSERT_EQ(SetCurRankId(1), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);

    sim::runtime::HcclThread threadOut{};
    sim::runtime::HcclChannel channelOut{};
    uint32_t rankId = 0;
    uint32_t remoteRankId = 0;
    EXPECT_EQ(GetChannelTaskContext(thread.id, channel.id, threadOut, channelOut, rankId, remoteRankId), HCCL_SUCCESS);
    EXPECT_EQ(rankId, 1U);
    EXPECT_EQ(remoteRankId, 2U);
    EXPECT_EQ(threadOut.commId, self.id);
    EXPECT_EQ(channelOut.remoteRankId, 2U);
    // 成功路径查询次数：Communicator 恰好 1 次（修复前 2 次）
    EXPECT_EQ(engine->SelectCallsOf("Communicator"), 1U);
    EXPECT_EQ(engine->SelectCallsOf("HcclThread"), 1U);
    EXPECT_EQ(engine->SelectCallsOf("HcclChannel"), 1U);

    ASSERT_EQ(SetCurRankId(savedRankId), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
    ASSERT_TRUE(HcclSim::Storage::StorageRuntime::CloseProcessSession().ok());
    runtime.ResetForTest();
}
