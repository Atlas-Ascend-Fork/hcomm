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
 * See the License in the specific file packaged with this Software for details.
 */

/**
 * @file backend_conformance_mariadb_test.cc
 * @brief 用真实 MariaDbEngine 实例化全部公共 backend conformance case（含
 * TransactionCase——START TRANSACTION/COMMIT/ROLLBACK 直达模型）。
 *
 * fixture 协议：外部真实 MariaDB Server 的 endpoint 由环境变量
 * HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 提供（mariadb://user:password@host:port）：
 *   - 未设置 → 每个用例明确 GTEST_SKIP（本地无 mysqld 时不伪造通过）；
 *   - 已设置但不可用/不可达 → 用例必须失败（不得静默跳过）。
 * 统一 StorageCatalog 的 conformance 目录经通用 MariaDB 工厂
 *（mariadb_backend_factory.h）创建引擎并经 StorageRuntime 工厂注册装配进程
 * 会话。每个用例安装会话前先清空 canonical 目录的 9 张表（静态表 +
 * DynamicScope 用例的两个固定 PID 作用域），保证用例间零数据耦合——真实
 * mysqld 是跨用例共享的外部状态。
 *
 * 公共 case 与期望全部来自 backend_conformance.h 单一来源，不为 MariaDB
 * 修改；MariaDB 专有行为（语句缓存、错误映射、dump 备份、死锁中止语义）
 * 留在 mariadb_engine_test 私有测试。
 */
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "backend_conformance.h"
#include "storage/database_result.h"
#include "storage_backends/mariadb/engine/mariadb_backend_factory.h"
#include "storage_backends/mariadb/engine/mariadb_engine.h"
#include "storage/internal/storage_runtime_impl.h"
#include "storage/storage_runtime.h"
#include "storage/table_access.h"
#include "storage_backends/mariadb/composition/mariadb_backend_module.h"

namespace storage_conformance {

namespace {
    /// 真实 MariaDB endpoint 环境变量（由测试环境提供；per-run mysqld）
    constexpr char kMariadbEndpointEnv[] = "HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT";

    /// endpoint 是否已配置（未配置 → 用例 GTEST_SKIP；配置后不可用 → 失败）
    bool MariadbConformanceEndpointConfigured()
    {
        const char* endpoint = std::getenv(kMariadbEndpointEnv);
        return endpoint != nullptr && endpoint[0] != '\0';
    }
} // namespace

/// fixture 会话安装：真实 MariaDB 通用引擎工厂 + 进程会话装配 + 用例前全表
/// 清空。返回 false 表示装配或清空失败（endpoint 已配置但不可用 → 用例必须
/// 失败）。
bool OpenConformanceSession()
{
    const char* endpointEnv = std::getenv(kMariadbEndpointEnv);
    if (endpointEnv == nullptr || endpointEnv[0] == '\0') {
        return false; // 由本地 case 宏先行 GTEST_SKIP，不应到达
    }
    const std::string endpointUri{endpointEnv};

    // 单实例部署 profile（v1 K=1；超时对齐测试环境，单位秒）
    HcclSim::Storage::MariadbConnectionProfile profile;
    profile.uri = endpointUri;
    profile.connectionEstablishmentTimeoutS = 3;
    profile.packetReadTimeoutS = 30;
    profile.packetWriteTimeoutS = 30;

    auto& runtime = HcclSim::Storage::detail::StorageRuntimeImpl::Instance();
    runtime.ResetForTest();
    runtime.RegisterEngineFactory(
        sim::backends::kMariadbBackendName,
        [profile](const HcclSim::Storage::LogicalDatabaseRuntimeConfig& config)
            -> HcclSim::Storage::DbResult<std::unique_ptr<HcclSim::Storage::IStorageEngine>> {
            // 引擎装配的唯一入口：mapping 由统一目录派生（单一事实源）
            return HcclSim::Storage::CreateMariadbEngine(kConformanceDatabase, config, profile);
        });
    runtime.UseDatabase(kConformanceDatabase);
    const HcclSim::Storage::StorageRuntimeConfig sessionConfig{
        {{std::string{"CONFORMANCE_DB"}, sim::backends::kMariadbBackendName, endpointUri}}};
    const auto ensured = HcclSim::Storage::StorageRuntime::EnsureProcessSession(sessionConfig);
    if (!ensured.ok()) {
        ADD_FAILURE() << "session install failed: " << ensured.diagnostic;
        return false;
    }

    // 真实 mysqld 是跨用例共享状态：用例前清空 canonical 目录全部表
    // （DynamicScope 用例固定使用 PID 101/102，见 backend_conformance.h）
    bool cleaned = true;
    const auto checkClear = [&cleaned](const auto& cleared, const char* table) {
        if (!cleared.ok()) {
            cleaned = false;
            ADD_FAILURE() << "clear " << table << " failed: " << cleared.diagnostic;
        }
    };
    checkClear(HcclSim::Storage::Clear<Device>(ClearMode::RESET_SEQUENCE), "Device");
    checkClear(HcclSim::Storage::Clear<BigValue>(ClearMode::RESET_SEQUENCE), "BigValue");
    checkClear(HcclSim::Storage::Clear<BlobRow>(ClearMode::RESET_SEQUENCE), "BlobRow");
    checkClear(HcclSim::Storage::Clear<UniqueRow>(ClearMode::RESET_SEQUENCE), "UniqueRow");
    checkClear(HcclSim::Storage::Clear<ChildRow>(ClearMode::RESET_SEQUENCE), "ChildRow");
    checkClear(HcclSim::Storage::Clear<GuardRow>(ClearMode::RESET_SEQUENCE), "GuardRow");
    checkClear(HcclSim::Storage::Clear<CounterRow>(ClearMode::RESET_SEQUENCE), "CounterRow");
    checkClear(HcclSim::Storage::Clear<WideRow>(ClearMode::RESET_SEQUENCE), "WideRow");
    checkClear(HcclSim::Storage::Clear<SlotRow>(ClearMode::RESET_SEQUENCE), "SlotRow");
    // 动态 PID 表用 DROP_DYNAMIC_SCOPE 幂等清理：物理 scope 表按需懒建——
    // 用例从未运行过时 DynTask_P_101/102 尚不存在，逐 pid CLEAR 会打不存在的
    // 物理表（errno 1146）；枚举式 DROP 对"尚无任何 scope 表"安全返回 0。
    checkClear(
        HcclSim::Storage::Clear<DynTask>(TableScope::Process(101), ClearMode::DROP_DYNAMIC_SCOPE),
        "DynTask(drop scopes)");
    if (!cleaned) {
        (void)HcclSim::Storage::StorageRuntime::CloseProcessSession();
        runtime.ResetForTest();
        return false;
    }
    return true;
}

/// fixture 会话关闭：排空并关闭进程会话，清理装配登记
bool CloseConformanceSession()
{
    const auto closed = HcclSim::Storage::StorageRuntime::CloseProcessSession();
    HcclSim::Storage::detail::StorageRuntimeImpl::Instance().ResetForTest();
    return closed.ok();
}
} // namespace storage_conformance

/// MariaDB fixture 的 case 展开：未配置 endpoint 先行 GTEST_SKIP；已配置则
/// 装配真实引擎会话（不可用必须失败），实例化 ALL 清单（含事务 case——
/// InnoDB 事务原生支持 START TRANSACTION/COMMIT/ROLLBACK 直达模型）。
#define HCCL_VM_MARIADB_CONFORMANCE_CASE(CaseName)                              \
    TEST(MariadbBackendConformance, CaseName)                                   \
    {                                                                           \
        if (!storage_conformance::MariadbConformanceEndpointConfigured()) {     \
            GTEST_SKIP() << "HCCL_VM_MARIADB_CONFORMANCE_ENDPOINT 未设置：" \
                            "真实 MariaDB conformance 需要外部 endpoint";       \
        }                                                                       \
        ASSERT_TRUE(storage_conformance::OpenConformanceSession());             \
        storage_conformance::cases::CaseName##Case();                           \
        ASSERT_TRUE(storage_conformance::CloseConformanceSession());            \
    }

// 本地重定义共享 case 宏：ALL 清单（含事务 case）按 MariaDB fixture 形态展开
#undef HCCL_VM_CONFORMANCE_CASE
#define HCCL_VM_CONFORMANCE_CASE(CaseName) HCCL_VM_MARIADB_CONFORMANCE_CASE(CaseName)
HCCL_VM_ALL_CONFORMANCE_CASES()
// 并发原子减法（真实后端；endpoint 未配置时同上 GTEST_SKIP）
HCCL_VM_CONCURRENT_INCREMENT_CONFORMANCE_CASES()
