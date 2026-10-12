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

// 文件职责：composition 工厂的退出生命周期回归。
// 场景（评审问题一）：先登记 atexit 消费者 → 注册真实 composition 工厂 →
// 安装显式 SQLite 会话但【不打开路由】→ 进程正常退出时消费者首次访问惰性
// 路由。工厂闭包必须自足（按值捕获模块 createEngine 回调）：部署快照是普通
// 静态存储，退出时先于遵循 no-destroy 合同的 StorageRuntime/会话析构——
// 借用部署对象裸指针的工厂在退出消费者调用时即 use-after-free。
// 断言边界：父进程显式断言【退出消费者的真实 Storage Count 返回
// DbCode=OK】（经结果文件传递），不是只看子进程退出码；子进程退出码与
// 结果文件存在性同样断言（消费者确实执行）。
// 子进程经本二进制 marker 参数重执行：独立进程保证静态析构/atexit 顺序
// 真实，不受测试进程自身状态影响。
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

#include "runtime_state/db_sim_runner_ops.h"
#include "storage_composition/storage_composition.h"
#include "storage/storage_runtime.h"
#include "storage/table_access.h"
#include "storage_backends/sqlite/composition/sqlite_backend_module.h"
using sim::backends::kSqliteBackendName;

namespace {
constexpr char kChildMarker[] = "--composition-exit-child";

/// 退出消费者经 fopen/fprintf 写结果（无捕获、无依赖静态字符串析构顺序）。
const char* g_exitResultPath = nullptr;
} // namespace

namespace {
/// 子进程：复现评审探针的最小顺序（atexit 先于工厂注册与会话安装）。
int RunLateConsumerChild(const char* resultPath, const char* dbPath)
{
    unsetenv("HCCL_VM_STORAGE_CONFIG");
    g_exitResultPath = resultPath;
    // 1) 先登记退出消费者：此后构造的静态对象（部署快照）析构都排在它之前。
    if (std::atexit([]() {
            auto count = HcclSim::Storage::Count<sim::runtime::Device>(HcclSim::Storage::All<sim::runtime::Device>());
            std::FILE* out = std::fopen(g_exitResultPath, "w");
            if (out != nullptr) {
                std::fprintf(out, "code=%d\ndiagnostic=%s\n", int(count.code), count.diagnostic.c_str());
                std::fclose(out);
            }
        })
        != 0) {
        return 4;
    }
    // 2) 注册真实 composition 工厂（经真实模块清单与部署快照）。
    if (!sim::RegisterSimulatorStorageBackends().ok()) {
        return 2;
    }
    // 3) 安装显式 SQLite 路由会话；不触表——路由保持惰性。
    HcclSim::Storage::StorageRuntimeConfig config;
    config.databases.push_back({"RUNNER_DB", std::string{kSqliteBackendName}, std::string{dbPath}});
    if (!HcclSim::Storage::StorageRuntime::EnsureProcessSession(config).ok()) {
        return 3;
    }
    std::fprintf(stderr, "session installed, route deliberately still lazy\n");
    return 0; // 4) 正常退出：退出消费者首次打开惰性路由（真实 Storage Count）。
}

void RemoveDatabaseFiles(const std::string& path)
{
    std::remove(path.c_str());
    std::remove((path + "-wal").c_str());
    std::remove((path + "-shm").c_str());
}
} // namespace

/// 本测试需要子进程 marker 模式，使用自有 main（不链接 gtest_main）。
int main(int argc, char** argv)
{
    if (argc > 3 && std::strcmp(argv[1], kChildMarker) == 0) {
        return RunLateConsumerChild(argv[2], argv[3]);
    }
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

namespace {
TEST(StorageCompositionExitTest, LateLazyRouteAccessAtExitReturnsOk)
{
    const std::string db = "/tmp/test_composition_exit_runner.db";
    const std::string result = "/tmp/test_composition_exit_result.txt";
    RemoveDatabaseFiles(db);
    std::remove(result.c_str());

    const pid_t pid = fork();
    ASSERT_GE(pid, 0) << "fork failed"; // 父（pid>0）子（pid==0）进程都通过
    if (pid == 0) {
        execl(
            "/proc/self/exe", "composition-exit-child", kChildMarker, result.c_str(), db.c_str(),
            static_cast<char*>(nullptr));
        _exit(127); // execl 失败
    }
    int status = 0;
    ASSERT_EQ(waitpid(pid, &status, 0), pid);
    ASSERT_TRUE(WIFEXITED(status));
    // 子进程正常退出（工厂注册/会话安装成功，无崩溃）
    ASSERT_EQ(WEXITSTATUS(status), 0);

    // 退出消费者确实执行并写出真实 Storage Count 结果
    std::ifstream in{result};
    ASSERT_TRUE(in.good()) << "exit consumer did not write its result";
    int code = -1;
    std::string diagnostic;
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("code=", 0) == 0) {
            code = std::atoi(line.c_str() + 5);
        }
        if (line.rfind("diagnostic=", 0) == 0) {
            diagnostic = line.substr(std::strlen("diagnostic="));
        }
    }
    // 【核心断言】退出消费者经真实 composition 工厂首次打开惰性路由：DbCode=OK
    EXPECT_EQ(code, int(HcclSim::Storage::DbCode::OK)) << "diagnostic: " << diagnostic;
    EXPECT_TRUE(diagnostic.empty());

    RemoveDatabaseFiles(db);
    std::remove(result.c_str());
}
} // namespace
