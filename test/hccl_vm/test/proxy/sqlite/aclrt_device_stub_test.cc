/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 文件职责：aclInit 冷启动回归——LD_PRELOAD rank 进程在真实初始化边界提前触发
// 进程 Session bootstrap。迁移前 rank 进程首次查询经 RunnerDB 惰性 singleton 自动
// 建连；存储抽象化后已知初始化边界（aclInit）复用同一 ready-session 机制提前
// fail-fast 安装进程 session，且该边界不是唯一正确性依赖——即使未执行，首次
// Storage 原子/业务调用仍能惰性建立会话（由 test_storage_cold_process 覆盖）。
// 本文件链接真实 production aclrt_device_stub.cc 验证该生命周期合同。
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>

#include "acl/acl_base.h"
#include "operation_data/operation_data_ops.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "runtime_state/sim_models.h"
#include "sim_common_api.h"
#include "storage/storage_runtime.h"
#include "storage_composition/storage_composition.h"

// 真实 production 初始化入口声明（与 aclrt_stub_test.cc 对 production 函数的
// extern "C" 声明模式一致；本测试只声明原型，实现来自链接的真实 aclrt_device_stub.cc）。
extern "C" {
aclError aclInit(const char* configPath);
aclError aclFinalize();
}

namespace {
// 测试私有安装根：aclInit 冷 Ensure 按默认路由打开的物理库与预置数据落在同一组
// 文件，且不触碰真实安装目录。HCCL_VM_INSTALL_ROOT 在首次解析安装根前设置。
const std::string kTestInstallRoot = "/tmp/test_aclrt_device_stub_install";
const std::string kTestRunnerDb = kTestInstallRoot + "/data/hccl_sim.db";
const std::string kTestOpDataDb = kTestInstallRoot + "/data/hccl_vm_data.db";
const uint64_t kServerId = 1;
const uint32_t kDeviceLogicId = 0;
const uint32_t kDevicePhysicalId = 7;

class AclrtDeviceStubColdInitTest : public testing::Test {
protected:
    void SetUp() override
    {
        (void)setenv("HCCL_VM_INSTALL_ROOT", kTestInstallRoot.c_str(), 1);
        std::filesystem::create_directories(kTestInstallRoot + "/data");

        // 测试 peer 装配：显式按与默认路由一致的测试私有端点安装会话后准备数据。
        HcclSim::Storage::StorageRuntimeConfig config;
        config.databases.push_back({std::string{"RUNNER_DB"}, "sqlite", kTestRunnerDb});
        config.databases.push_back({std::string{"OP_DATA_DB"}, "sqlite", kTestOpDataDb});
        ASSERT_TRUE(sim::EnsureSimulatorStorageSession(config).ok());

        // 最小 Runner 数据：rank 进程首次数据库访问查询的 Server/Device 记录。
        sim::runtime::Server server{};
        server.pod_id = 0;
        server.server_id = static_cast<uint32_t>(kServerId);
        (void)snprintf(server.version, sizeof(server.version), "cold-init");
        ASSERT_TRUE(sim::runtime::Db::Add<sim::runtime::Server>(server).ok());

        sim::runtime::Device device{};
        device.server_id = static_cast<uint32_t>(kServerId);
        device.logic_id = kDeviceLogicId;
        device.physical_id = kDevicePhysicalId;
        device.status = 1;
        (void)snprintf(device.soc_version, sizeof(device.soc_version), "cold-soc");
        ASSERT_TRUE(sim::runtime::Db::Add<sim::runtime::Device>(device).ok());

        // 显式关闭进程会话：进入 rank 进程冷启动状态。
        (void)HcclSim::Storage::StorageRuntime::CloseProcessSession();
    }

    void TearDown() override
    {
        (void)HcclSim::Storage::StorageRuntime::CloseProcessSession();
        std::error_code ignored;
        std::filesystem::remove_all(kTestInstallRoot, ignored);
    }
};

// 冷 rank 进程：aclInit 安装进程 session，随后正式 Db 查询已准备的 Runner 数据成功。
// 旧实现 aclInit 不安装 session：查询以“process session is not installed”失败（RED）。
TEST_F(AclrtDeviceStubColdInitTest, aclInitInstallsProcessStorageSessionForColdRankProcess)
{
    // 冷状态前置证明：公开维护入口对未安装会话的拒绝诊断即真实缺口签名
    // （也证明后续查询所用 session 只能由 aclInit 安装，而非测试预装）。
    const auto cold = HcclSim::Storage::StorageRuntime::CloseProcessSession();
    ASSERT_FALSE(cold.ok());
    ASSERT_EQ(cold.diagnostic, std::string("process session is not installed"));

    // 真实 production aclInit（rank 进程初始化边界；非复制的测试实现）。
    ASSERT_EQ(aclInit(nullptr), ACL_SUCCESS);

    // 与 aclrtSetDevice 相同条件形态的正式 Db 查询：已准备 Device 可见。
    const auto device = sim::runtime::Db::GetOneByPred<sim::runtime::Device>(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&sim::runtime::Device::server_id, static_cast<uint32_t>(kServerId)),
        HcclSim::Storage::Eq(&sim::runtime::Device::logic_id, kDeviceLogicId)));
    ASSERT_TRUE(device.ok()) << device.diagnostic;
    EXPECT_EQ(device->physical_id, kDevicePhysicalId);
    EXPECT_EQ(std::strncmp(device->soc_version, "cold-soc", sizeof(device->soc_version)), 0);

    // 已准备 Server 同样可见（与 GetServerKeyById 相同条件形态）。
    const auto server = sim::runtime::Db::GetOneByPred<sim::runtime::Server>(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&sim::runtime::Server::pod_id, uint64_t{0}),
        HcclSim::Storage::Eq(&sim::runtime::Server::server_id, static_cast<uint32_t>(kServerId))));
    ASSERT_TRUE(server.ok()) << server.diagnostic;
    EXPECT_EQ(std::strncmp(server->version, "cold-init", sizeof(server->version)), 0);
}

// 幂等：同一进程多次 aclInit（Ensure 幂等）不破坏已安装 session。
TEST_F(AclrtDeviceStubColdInitTest, aclInitIsIdempotentForProcessStorageSession)
{
    ASSERT_EQ(aclInit(nullptr), ACL_SUCCESS);
    ASSERT_EQ(aclInit(nullptr), ACL_SUCCESS);

    const auto device = sim::runtime::Db::GetOneByPred<sim::runtime::Device>(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&sim::runtime::Device::server_id, static_cast<uint32_t>(kServerId)),
        HcclSim::Storage::Eq(&sim::runtime::Device::logic_id, kDeviceLogicId)));
    ASSERT_TRUE(device.ok()) << device.diagnostic;
    EXPECT_EQ(device->physical_id, kDevicePhysicalId);
}

// aclFinalize 不是数据库最后使用边界：rank 进程在 aclFinalize 之后仍会继续执行
// Socket/Context/Device 等退出清理（HCCL 动态库静态析构与退出路径继续以业务 API/
// 数据库原子 API 访问数据库）。真实蓝区证据（上一候选 lifecycle_round1 官方
// ReduceScatter）：aclFinalize 后 RaSocketListenStop "can not get Socket:1/2"、
// aclrtResetDevice "ctx not found"、aclrtGetSocName "device not found"——会话被
// aclFinalize 关闭（REMOVED）后，退出清理查询全部被拒。本测试以与真实退出清理
// 完全等价的 Db 调用
// 验证：aclFinalize 返回 ACL_SUCCESS 后，查询 Device/Context、插入并
// GetById/Update/Delete 一条测试私有 Event 记录均成功（原 RaSocket 冒烟段随
// 死表删除改写为存活表，CRUD 覆盖不变）。旧实现 aclFinalize 调用
// CloseProcessSession 关闭会话：post-finalize 数据操作以 "process storage session
// is not installed" 失败（RED）。
TEST_F(AclrtDeviceStubColdInitTest, aclFinalizeKeepsProcessStorageAvailableForLateCleanup)
{
    // 1. 冷 rank 进程初始化边界安装进程 session。
    ASSERT_EQ(aclInit(nullptr), ACL_SUCCESS);

    // 2. 真实 Db API 确认 Device 可查询（正式业务查询通道）。
    const auto device = sim::runtime::Db::GetOneByPred<sim::runtime::Device>(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&sim::runtime::Device::server_id, static_cast<uint32_t>(kServerId)),
        HcclSim::Storage::Eq(&sim::runtime::Device::logic_id, kDeviceLogicId)));
    ASSERT_TRUE(device.ok()) << device.diagnostic;

    // 3. 真实 production aclFinalize：返回语义保持 ACL_SUCCESS（重复调用幂等）。
    ASSERT_EQ(aclFinalize(), ACL_SUCCESS);
    ASSERT_EQ(aclFinalize(), ACL_SUCCESS);

    // 4. post-finalize 退出清理等价数据操作（与真实清理路径相同的 Db 调用形态）。
    // 4.1 Device 查询（aclrtGetSocName/aclrtGetPhyDevIdByLogicDevId 同形态）。
    const auto deviceAfter = sim::runtime::Db::GetOneByPred<sim::runtime::Device>(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&sim::runtime::Device::server_id, static_cast<uint32_t>(kServerId)),
        HcclSim::Storage::Eq(&sim::runtime::Device::logic_id, kDeviceLogicId)));
    ASSERT_TRUE(deviceAfter.ok()) << deviceAfter.diagnostic;
    EXPECT_EQ(deviceAfter->physical_id, kDevicePhysicalId);

    // 4.2 测试私有 Event 记录：Add + GetById + Update status + Delete（原 RaSocket
    //     冒烟段随死表删除改写为存活表，post-finalize 全套 CRUD 覆盖不变）。
    sim::runtime::Event event{};
    event.create_ctx_id = 0;
    event.event_flag = 1;
    event.device_res_seq = 0;
    event.created_time = 0;
    event.status = 1;
    ASSERT_TRUE(sim::runtime::Db::Add<sim::runtime::Event>(event).ok());
    const uint64_t eventId = event.id;
    ASSERT_NE(eventId, 0U);

    const auto flagged = sim::runtime::Db::GetById<sim::runtime::Event>(eventId);
    ASSERT_TRUE(flagged.ok()) << flagged.diagnostic;
    ASSERT_TRUE(sim::runtime::Db::Update<sim::runtime::Event>(
                    HcclSim::Storage::Eq(&sim::runtime::Event::id, eventId),
                    HcclSim::Storage::Set(&sim::runtime::Event::status, uint64_t{0}))
                    .ok());
    ASSERT_TRUE(
        sim::runtime::Db::Delete<sim::runtime::Event>(HcclSim::Storage::Eq(&sim::runtime::Event::id, eventId)).ok());

    // 4.3 Context 查询（aclrtResetDevice 的 GetById<Context> 形态）：准备测试私有
    //     Context 后查询并删除。
    sim::runtime::Context context{};
    context.device_id = kDevicePhysicalId;
    context.is_default = 1;
    context.ref_cnt = 1;
    ASSERT_TRUE(sim::runtime::Db::Add<sim::runtime::Context>(context).ok());
    const uint64_t ctxId = context.id;
    ASSERT_NE(ctxId, 0U);
    const auto ctx = sim::runtime::Db::GetById<sim::runtime::Context>(ctxId);
    ASSERT_TRUE(ctx.ok()) << ctx.diagnostic;
    ASSERT_TRUE(
        sim::runtime::Db::Delete<sim::runtime::Context>(HcclSim::Storage::Eq(&sim::runtime::Context::id, ctxId)).ok());

    // 5. 全部操作成功；测试无网络/线程/fork；fixture TearDown 显式 CloseProcessSession。
}
} // namespace
