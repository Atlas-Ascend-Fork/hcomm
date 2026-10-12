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
 * See the License for details.
 * See the License in the root of the software repository for the full text of the License.
 */

// 主机侧代理 drvMemcpy（src/proxy/level2/ascend_hal_stub.cc）的地址分类与错误
// 传播回归。IsDeviceAddress 三态合同（命中 OK+true / 确定未命中 OK+false /
// 查询无法完成原样传播错误码）与 drvMemcpy 的对应行为一并锁定：
//   - 分类失败不得冒充 host 分类进入底层拷贝（128p 拆卸期 HOST_TO_HOST 裸
//     memcpy 设备地址崩溃链的边界修复）；
//   - aclrtMemcpy 失败不得返回驱动成功；
//   - 正常 host-to-host 分类与拷贝行为保持不变。
#include <cstdint>
#include <cstdio>
#include <gtest/gtest.h>

#include "ascend_hal.h"
#include "hccl_proxy_common.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "sqlite/simulation_storage_test_helper.h"
#include "storage/storage_runtime.h"

namespace {
const std::string kTestDbPath = "/tmp/test_ascend_hal_proxy_stub.db";
} // namespace

class AscendHalProxyStubTest : public testing::Test {
protected:
    void SetUp() override
    {
        // 每用例关闭会话并删除库文件后重建（等价既有 runnerdb 测试隔离语义）
        ASSERT_TRUE(runnerdb_test::ResetTestSession(kTestDbPath, kTestDbPath));
    }

    void TearDown() override { runnerdb_test::CleanUpDatabases(kTestDbPath, kTestDbPath); }

    /// 插入一行 DEV 设备映射块，返回行 id（0 = 失败）
    static uint64_t InsertDevBlock(uint64_t ptr, uint64_t size)
    {
        sim::runtime::VirtualMemBlock block{};
        block.start_ptr = ptr;
        block.dev_mapped_ptr = ptr;
        block.size = size;
        block.device_id = 1U;
        block.rank_id = 1U;
        block.src_type = static_cast<uint8_t>(sim::runtime::VIR_MEM_TYPE_DEV);
        return runnerdb_test::InsertRecord(block);
    }
};

// ---- IsDeviceAddress 三态合同 ----

// 查询完成且命中：DEV 块覆盖的地址 → OK+true
TEST_F(AscendHalProxyStubTest, IsDeviceAddress_HitReturnsOkTrue)
{
    constexpr uint64_t kBase = 0x100000000ULL;
    ASSERT_NE(InsertDevBlock(kBase, 0x1000ULL), 0U);
    const auto hit = sim::IsDeviceAddress(reinterpret_cast<void*>(kBase + 0x800ULL));
    ASSERT_TRUE(hit.ok());
    ASSERT_TRUE(hit.value.has_value());
    EXPECT_TRUE(*hit.value);
}

// 查询完成且确定未命中（NOT_FOUND 是业务结果不是错误）→ OK+false
TEST_F(AscendHalProxyStubTest, IsDeviceAddress_MissReturnsOkFalse)
{
    ASSERT_NE(InsertDevBlock(0x100000000ULL, 0x1000ULL), 0U);
    const auto miss = sim::IsDeviceAddress(reinterpret_cast<void*>(0x200000000ULL));
    ASSERT_TRUE(miss.ok());
    ASSERT_TRUE(miss.value.has_value());
    EXPECT_FALSE(*miss.value);
}

// 查询无法完成（会话已关闭）→ 原错误码与诊断传播，不得转成 false 冒充未命中
TEST_F(AscendHalProxyStubTest, IsDeviceAddress_QueryFailurePropagatesErrorCode)
{
    const auto closed = HcclSim::Storage::StorageRuntime::CloseProcessSession();
    ASSERT_TRUE(closed.ok()) << closed.diagnostic;
    const auto failed = sim::IsDeviceAddress(reinterpret_cast<void*>(0x100000000ULL));
    EXPECT_FALSE(failed.ok());
    EXPECT_FALSE(failed.value.has_value()) << "失败结果不得携带值载荷";
    EXPECT_NE(failed.diagnostic.find("session"), std::string::npos) << failed.diagnostic;
}

// ---- drvMemcpy 分类与拷贝错误传播 ----

// 分类失败（src 查询无法完成）→ 不进入底层拷贝，返回驱动内部错误而非成功
TEST_F(AscendHalProxyStubTest, DrvMemcpy_ClassificationFailureSkipsCopyAndReturnsDriverError)
{
    const auto closed = HcclSim::Storage::StorageRuntime::CloseProcessSession();
    ASSERT_TRUE(closed.ok()) << closed.diagnostic;
    uint64_t src = 0x1111U;
    uint64_t dst = 0x2222U;
    const DVresult result = drvMemcpy((DVdeviceptr)&dst, sizeof(dst), (DVdeviceptr)&src, sizeof(src));
    EXPECT_EQ(result, DRV_ERROR_INNER_ERR);
    EXPECT_EQ(dst, 0x2222U); // 未进入底层拷贝，目的缓冲未被写
}

// 分类成功（双 host → HOST_TO_HOST）但 aclrtMemcpy 上界报错 → 不得返回驱动成功
TEST_F(AscendHalProxyStubTest, DrvMemcpy_CopyFailurePropagatesDriverError)
{
    uint64_t src = 0x1111U;
    uint64_t dst = 0x2222U;
    const DVresult result = drvMemcpy((DVdeviceptr)&dst, sizeof(dst), (DVdeviceptr)&src, sizeof(src) + 1U);
    EXPECT_EQ(result, DRV_ERROR_INNER_ERR);
    EXPECT_EQ(dst, 0x2222U); // 拷贝未发生
}

// 正常路径保持：双 host 分类 → HOST_TO_HOST 实拷贝 → 驱动成功
TEST_F(AscendHalProxyStubTest, DrvMemcpy_HostToHostCopiesAndSucceeds)
{
    uint64_t src = 0x1111U;
    uint64_t dst = 0x2222U;
    const DVresult result = drvMemcpy((DVdeviceptr)&dst, sizeof(dst), (DVdeviceptr)&src, sizeof(src));
    EXPECT_EQ(result, DRV_ERROR_NONE);
    EXPECT_EQ(dst, 0x1111U);
}

// 分类成功且目的为已登记设备地址 → H2D 路径按原 kind 选择（不因错误处理改动
// 而改变分类结果）：dst 命中 DEV 块时 isDstHost=false 的分支保持可达；src 为
// host 栈地址（OK+false）。未登记 PhyMemBlock 的虚拟设备地址在 aclrtMemcpy 的
// H2D 翻译失败（ACL_ERROR_INTERNAL_ERROR）→ 驱动错误传播，不再走 H2H 裸 memcpy。
TEST_F(AscendHalProxyStubTest, DrvMemcpy_DeviceDstClassificationStillSelectsHostToDevice)
{
    constexpr uint64_t kDevBase = 0x100000000ULL;
    ASSERT_NE(InsertDevBlock(kDevBase, 0x1000ULL), 0U);
    uint64_t src = 0x1111U;
    const DVresult result = drvMemcpy((DVdeviceptr)(kDevBase + 0x800ULL), 8U, (DVdeviceptr)&src, sizeof(src));
    EXPECT_EQ(result, DRV_ERROR_INNER_ERR);
}

// ---- drvGetDevNum：Db::Count 等价替换回归（任务书 v3 §D/回归清单9）----
// 旧实现 GetByPred<Device>(Eq(status,0)) 物化全部行后只做 empty()+size()；
// 新实现数据库端 COUNT 聚合。锁定：输出与错误语义逐项一致（server/runner/ctx
// 前置检查、空集与查询失败都返回 DRV_ERROR_NO_DEVICE 且不写出参、混合 status
// 只计 status=0、成功路径原日志出参），空结果不等于后端失败。

extern uint64_t g_cur_server_key;

class DrvGetDevNumTest : public testing::Test {
protected:
    uint64_t serverId{0};
    uint64_t firstDeviceId{0};

    void SetUp() override
    {
        ASSERT_TRUE(runnerdb_test::ResetTestSession(kTestDbPath, kTestDbPath));
        sim::runtime::Server server{};
        server.pod_id = 100;
        serverId = runnerdb_test::InsertRecord<sim::runtime::Server>(server);
        ASSERT_NE(serverId, 0U);
        sim::runtime::Host host{};
        host.server_id = serverId;
        host.arch = 1;
        ASSERT_NE(runnerdb_test::InsertRecord<sim::runtime::Host>(host), 0U);
        firstDeviceId = InsertDeviceWithStatus(0, 0);
        ASSERT_NE(firstDeviceId, 0U);

        g_cur_server_key = serverId;
        sim::runtime::Runner runner{};
        ASSERT_TRUE(sim::runtime::GetCurrRunnerTls(serverId, runner));
        sim::runtime::Context context{};
        context.device_id = firstDeviceId;
        context.is_default = 1;
        context.ref_cnt = 1;
        const uint64_t ctxId = runnerdb_test::InsertRecord<sim::runtime::Context>(context);
        ASSERT_NE(ctxId, 0U);
        ASSERT_TRUE(sim::runtime::SetCurrCtxTls(ctxId));
    }

    void TearDown() override
    {
        g_cur_server_key = 0;
        runnerdb_test::CleanUpDatabases(kTestDbPath, kTestDbPath);
    }

    static uint64_t InsertDeviceWithStatus(uint32_t physicalId, uint32_t status)
    {
        sim::runtime::Device device{};
        device.server_id = 1;
        device.logic_id = physicalId;
        device.physical_id = physicalId;
        device.status = status;
        return runnerdb_test::InsertRecord<sim::runtime::Device>(device);
    }
};

TEST_F(DrvGetDevNumTest, MixedStatusCountsOnlyAvailableDevices)
{
    // SetUp 已有 1 台 status=0；再加 2 台 status=0 与 2 台 status=1
    ASSERT_NE(InsertDeviceWithStatus(1, 0), 0U);
    ASSERT_NE(InsertDeviceWithStatus(2, 0), 0U);
    ASSERT_NE(InsertDeviceWithStatus(3, 1), 0U);
    ASSERT_NE(InsertDeviceWithStatus(4, 1), 0U);

    uint32_t numDev = 0;
    EXPECT_EQ(drvGetDevNum(&numDev), DRV_ERROR_NONE);
    EXPECT_EQ(numDev, 3U);
}

TEST_F(DrvGetDevNumTest, EmptySetReturnsNoDeviceWithoutWritingOut)
{
    // 只保留 status=1 的行：查询成功但空集 → NO_DEVICE，不写出参，
    // 不当作后端失败路径（旧实现语义逐项一致）
    ASSERT_TRUE(runnerdb_test::ClearRecords<sim::runtime::Device>());
    ASSERT_NE(InsertDeviceWithStatus(1, 1), 0U);

    uint32_t numDev = 12345;
    EXPECT_EQ(drvGetDevNum(&numDev), DRV_ERROR_NO_DEVICE);
    EXPECT_EQ(numDev, 12345U);
}

TEST_F(DrvGetDevNumTest, BackendFailureReturnsNoDevice)
{
    // 后端错误注入（会话关闭）：与旧实现相同返回 NO_DEVICE，不写出参
    const auto closed = HcclSim::Storage::StorageRuntime::CloseProcessSession();
    ASSERT_TRUE(closed.ok()) << closed.diagnostic;

    uint32_t numDev = 7;
    EXPECT_EQ(drvGetDevNum(&numDev), DRV_ERROR_NO_DEVICE);
    EXPECT_EQ(numDev, 7U);

    // 恢复会话供 TearDown 清理
    ASSERT_TRUE(runnerdb_test::ResetTestSession(kTestDbPath, kTestDbPath));
}

TEST_F(DrvGetDevNumTest, PreconditionsWithoutContextReturnNoDevice)
{
    // server/runner/ctx 前置检查保持：g_cur_server_key=0 → NO_DEVICE（不触库）
    g_cur_server_key = 0;
    uint32_t numDev = 3;
    EXPECT_EQ(drvGetDevNum(&numDev), DRV_ERROR_NO_DEVICE);
    EXPECT_EQ(numDev, 3U);
}
