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

// CommInit 适配入口的最小失败检查（任务书§4.C 末项）回归：
//   冻结 HCOMM 48b54e99 消费合同：HDCommunicateParams 块级可选（buffLen==0
//   不消费）、块内 deviceAddr/readCacheAddr 必需（CHK_PTR_NULL）。
//   1) 必需块地址映射失败 → 非零 status 返回，kernel 不被调用；
//   2) 可选块（buffLen==0）地址映射失败 → 照常调用 kernel（可选语义保留）；
//   3) is_dev_access=1 的 VMB 命中 → 地址原样直通，kernel 被调用。
// kernel .so（test_kern_comminit.cc）记录调用计数与收到的 deviceAddr。
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <string>
#include <unistd.h>

#include <gtest/gtest.h>

#include "hccl_device_pub.h"
#include "aicpu_args_stub.h"
#include "sim_aicpu_pipe_handler.h"
#include "sim_aicpu_pipe_msg.h"
#include "sim_common_defs.h"
#include "sim_pipe_io.h"
#include "sqlite/simulation_storage_test_helper.h"
#include "operation_data/operation_data_ops.h"
#include "runtime_state/db_sim_runner_ops.h"

#ifndef HCCL_VM_TEST_KERNEL_SO
#define HCCL_VM_TEST_KERNEL_SO "libtest_kern_comminit.so"
#endif

namespace {
const std::string kRunnerDbPath = "/tmp/test_comminit_runner.db";
const std::string kOpDbPath = "/tmp/test_comminit_op.db";

// 线程类内核透传回归的共用载荷：前 8 字节为已知模式（HCOMM 以 &context
// 8 字节 device 地址发起的合同形状），其余填充，用于核对“原样透传”
constexpr uint64_t kThreadArgsPattern = 0x0123456789ABCDEFULL;
constexpr uint32_t kThreadArgsBytes = 64;

// InstallPath 安装根进程内首次使用即缓存：main 前注入 HCCL_VM_INSTALL_ROOT
// 并拷入受控 kernel .so（只拷一次——dlopen 后文件被映射不可覆写）
std::string TestInstallRoot()
{
    static const std::string root = "/tmp/hccl_vm_comminit_root_" + std::to_string(::getpid());
    return root;
}
const bool kInstallRootGuard = [] {
    std::error_code ec;
    const std::string libDir = TestInstallRoot() + "/lib/" + GetArchStr();
    std::filesystem::create_directories(libDir, ec);
    std::filesystem::copy_file(
        HCCL_VM_TEST_KERNEL_SO, libDir + "/libtest_kern_comminit.so", std::filesystem::copy_options::overwrite_existing,
        ec);
    ::setenv("HCCL_VM_INSTALL_ROOT", TestInstallRoot().c_str(), 1);
    return true;
}();

/// 受控 .so 的完整路径（与被测代码经安装根解析到的同一文件——同一路径
/// dlopen 返回同一 handle，导出变量与 kernel 所写共享）
std::string TestKernelSoPath() { return TestInstallRoot() + "/lib/" + GetArchStr() + "/libtest_kern_comminit.so"; }

/// 读取受控 .so 的调用计数（dlsym 同进程导出变量）
std::atomic<uint32_t>* KernelCallCount()
{
    static std::atomic<uint32_t>* counter = [] {
        void* handle = dlopen(TestKernelSoPath().c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (handle == nullptr) {
            return static_cast<std::atomic<uint32_t>*>(nullptr);
        }
        return reinterpret_cast<std::atomic<uint32_t>*>(dlsym(handle, "g_test_comminit_call_count"));
    }();
    return counter;
}

/// 覆盖 addr 的 VMB 行（deviceKey 绑定当前设备；is_dev_access 控制直通分支；
/// 区间端点由 size 派生——镜像 sqe 测试 InsertDevVmb 的字段集）
void InsertVmbRow(uint64_t deviceKey, uint64_t mappedBase, uint64_t size, bool devAccess)
{
    sim::runtime::PhyMemBlock phyMem{};
    phyMem.device_id = 1;
    std::snprintf(phyMem.name, sizeof(phyMem.name), "comminit_phy_%llu", static_cast<unsigned long long>(mappedBase));
    phyMem.size = size;
    phyMem.ref_count = 1;
    const uint64_t phyId = runnerdb_test::InsertRecord<sim::runtime::PhyMemBlock>(phyMem);

    sim::runtime::VirtualMemBlock vmb{};
    vmb.start_ptr = mappedBase + 0x100000000ULL;
    vmb.dev_mapped_ptr = mappedBase;
    vmb.size = size;
    vmb.ctx_id = 1;
    vmb.device_id = deviceKey;
    vmb.phy_mem_id = phyId;
    vmb.owner_pid = 0;
    vmb.src_type = (uint8_t)sim::runtime::VIR_MEM_TYPE_DEV;
    vmb.is_dev_access = devAccess ? 1 : 0;
    ASSERT_NE(runnerdb_test::InsertRecord<sim::runtime::VirtualMemBlock>(vmb), 0U);
}
} // namespace

class AicpuCommInitCheckTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        (void)kInstallRootGuard;
        ASSERT_TRUE(std::filesystem::exists(TestInstallRoot() + "/lib/" + GetArchStr() + "/libtest_kern_comminit.so"));
        ASSERT_TRUE(runnerdb_test::ResetTestSession(kRunnerDbPath, kOpDbPath));

        sim::operation::OpDetailTab detail{};
        detail.pid = static_cast<uint32_t>(::getpid());
        detail.deviceId = 1;
        detail.rankId = 3;
        detail.commId = 77;
        detailId_ = runnerdb_test::InsertRecord<sim::operation::OpDetailTab>(detail);
        ASSERT_NE(detailId_, 0U);

        ASSERT_EQ(::pipe(pipeFds_), 0);
        InitPipeFds(pipeFds_[0], pipeFds_[1]);
        SetCurRankId(3);
        SetCurDeviceKey(kDeviceKey);

        // 计数清零（跨用例共享的 .so 全局）
        std::atomic<uint32_t>* counter = KernelCallCount();
        ASSERT_NE(counter, nullptr);
        counter->store(0);
        callCount_ = counter;
    }

    void TearDown() override
    {
        sim::operation::g_currOpDetailId = 0;
        SetCurDeviceKey(0);
        ::close(pipeFds_[0]);
        ::close(pipeFds_[1]);
        runnerdb_test::CleanUpDatabases(kRunnerDbPath, kOpDbPath);
    }

    /// 下发 RunAicpuCommInit（身份合法），返回设备响应 status
    int32_t ExecCommInit(const CommAicpuParam& param)
    {
        std::vector<uint8_t> buffer(sizeof(ExecKernelPayload) + sizeof(CommAicpuParam), 0);
        auto* req = reinterpret_cast<ExecKernelPayload*>(buffer.data());
        std::strncpy(req->kernelName, "RunAicpuCommInit", sizeof(req->kernelName) - 1);
        std::strncpy(req->soName, "libtest_kern_comminit.so", sizeof(req->soName) - 1);
        req->opDetailId = static_cast<uint32_t>(detailId_);
        std::memcpy(buffer.data() + sizeof(ExecKernelPayload), &param, sizeof(CommAicpuParam));

        const int rc = sim::HandlePipeCmdExecKernel(buffer.data(), static_cast<uint32_t>(buffer.size()));
        uint8_t rspCmd = 0;
        RspExecKernelPayload rsp{};
        uint32_t rspLen = 0;
        EXPECT_EQ(sim::PipeRecvMsg(pipeFds_[0], rspCmd, &rsp, sizeof(rsp), rspLen), 0);
        EXPECT_EQ(rspCmd, PIPE_RSP_EXEC_KERNEL);
        EXPECT_EQ(rc, static_cast<int>(rsp.status)); // handler 返回值与 status 同源
        return rsp.status;
    }

    /// 下发线程类内核（四个入口共用同一验证），返回设备响应 status；
    /// sentArgs 输出本次载荷中 args 起始地址，供“指针原样透传”核对
    int32_t ExecThreadKernel(const char* kernelName, const void** sentArgs)
    {
        std::vector<uint8_t> buffer(sizeof(ExecKernelPayload) + kThreadArgsBytes, 0);
        auto* req = reinterpret_cast<ExecKernelPayload*>(buffer.data());
        std::strncpy(req->kernelName, kernelName, sizeof(req->kernelName) - 1);
        std::strncpy(req->soName, "libtest_kern_comminit.so", sizeof(req->soName) - 1);
        req->opDetailId = 0; // 普通 kernel：无通信域身份上下文
        uint8_t* args = buffer.data() + sizeof(ExecKernelPayload);
        std::memcpy(args, &kThreadArgsPattern, sizeof(kThreadArgsPattern));
        for (uint32_t i = sizeof(kThreadArgsPattern); i < kThreadArgsBytes; ++i) {
            args[i] = static_cast<uint8_t>(0x40 + (i & 0x3F));
        }
        if (sentArgs != nullptr) {
            *sentArgs = args;
        }

        const int rc = sim::HandlePipeCmdExecKernel(buffer.data(), static_cast<uint32_t>(buffer.size()));
        uint8_t rspCmd = 0;
        RspExecKernelPayload rsp{};
        uint32_t rspLen = 0;
        EXPECT_EQ(sim::PipeRecvMsg(pipeFds_[0], rspCmd, &rsp, sizeof(rsp), rspLen), 0);
        EXPECT_EQ(rspCmd, PIPE_RSP_EXEC_KERNEL);
        EXPECT_EQ(rc, static_cast<int>(rsp.status)); // handler 返回值与 status 同源
        return rsp.status;
    }

    static constexpr uint64_t kDeviceKey = 42;
    static constexpr uint64_t kMappedBase = 0x500000000ULL;
    static constexpr uint64_t kUnmappedAddr = 0xDEAD0000000ULL;
    uint64_t detailId_{0};
    int pipeFds_[2]{-1, -1};
    std::atomic<uint32_t>* callCount_{nullptr};
};

// 必需块（buffLen!=0）地址映射失败：非零 status，kernel 不被调用
TEST_F(AicpuCommInitCheckTest, RequiredAddrTranslateFailureKernelNotCalled)
{
    CommAicpuParam param{};
    std::snprintf(param.hcomId, sizeof(param.hcomId), "comminit_req_fail");
    param.kfcControlTransferH2DParams.deviceAddr = kUnmappedAddr; // 无 VMB 覆盖
    param.kfcControlTransferH2DParams.readCacheAddr = kUnmappedAddr;
    param.kfcControlTransferH2DParams.devMemSize = 4096;
    param.kfcControlTransferH2DParams.buffLen = 4096; // 必需块

    const int32_t status = ExecCommInit(param);
    EXPECT_NE(status, 0);
    EXPECT_EQ(callCount_->load(), 0U) << "必需地址映射失败时不得调用 kernel";
}

// 可选块（buffLen==0）地址映射失败：照常调用 kernel（可选语义保留，status=0）
TEST_F(AicpuCommInitCheckTest, OptionalBlockUnmappedAddrStillCallsKernel)
{
    CommAicpuParam param{};
    std::snprintf(param.hcomId, sizeof(param.hcomId), "comminit_optional");
    param.kfcControlTransferH2DParams.deviceAddr = kUnmappedAddr; // 映射失败→0
    param.kfcControlTransferH2DParams.readCacheAddr = kUnmappedAddr;
    param.kfcControlTransferH2DParams.buffLen = 0; // 可选块：HCOMM 不消费

    const int32_t status = ExecCommInit(param);
    EXPECT_EQ(status, 0);
    EXPECT_EQ(callCount_->load(), 1U);
}

// VMB 命中且 is_dev_access=1：地址原样直通（不映射），kernel 被调用且收到原值
TEST_F(AicpuCommInitCheckTest, DevAccessHitPassesAddrThroughToKernel)
{
    InsertVmbRow(kDeviceKey, kMappedBase, 8192, true);

    CommAicpuParam param{};
    std::snprintf(param.hcomId, sizeof(param.hcomId), "comminit_devaccess");
    param.kfcControlTransferH2DParams.deviceAddr = kMappedBase + 256;
    param.kfcControlTransferH2DParams.readCacheAddr = kMappedBase + 512;
    param.kfcControlTransferH2DParams.devMemSize = 4096;
    param.kfcControlTransferH2DParams.buffLen = 4096;

    const int32_t status = ExecCommInit(param);
    EXPECT_EQ(status, 0);
    EXPECT_EQ(callCount_->load(), 1U);
    // is_dev_access=1 → GetDevMapperAddr 原样返回，kernel 收到原地址
    void* handle = dlopen(TestKernelSoPath().c_str(), RTLD_NOW | RTLD_GLOBAL);
    ASSERT_NE(handle, nullptr);
    auto* lastAddr = reinterpret_cast<const uint64_t*>(dlsym(handle, "g_test_comminit_last_device_addr"));
    ASSERT_NE(lastAddr, nullptr);
    EXPECT_EQ(*lastAddr, kMappedBase + 256);
}

// ===== 线程类内核参数透传回归（PR !923 整改后合同）=====
// 四个线程入口共用同一组参数化验证：
//   1) args 指针与首 8 字节内容原样到达 kernel（不经过任何转换/改写）；
//   2) kernel 非零返回值沿 status 链传回设备响应。
// 不复制 HCOMM 私有结构，不构造冷启动地址场景。
TEST_F(AicpuCommInitCheckTest, ThreadKernelArgsPassedThroughAndRetPropagates)
{
    void* handle = dlopen(TestKernelSoPath().c_str(), RTLD_NOW | RTLD_GLOBAL);
    ASSERT_NE(handle, nullptr);
    auto* threadCount = reinterpret_cast<std::atomic<uint32_t>*>(dlsym(handle, "g_test_thread_call_count"));
    auto* lastArgs = reinterpret_cast<void**>(dlsym(handle, "g_test_thread_last_args"));
    auto* lastHead = reinterpret_cast<uint64_t*>(dlsym(handle, "g_test_thread_last_args_head"));
    auto* retVal = reinterpret_cast<std::atomic<uint32_t>*>(dlsym(handle, "g_test_thread_ret_value"));
    ASSERT_NE(threadCount, nullptr);
    ASSERT_NE(lastArgs, nullptr);
    ASSERT_NE(lastHead, nullptr);
    ASSERT_NE(retVal, nullptr);

    const char* threadKernels[] = {
        "RunAicpuThreadInit",
        "RunAicpuThreadDestroy",
        "RunAicpuThreadSupplementNotify",
        "RunAicpuIndOpThreadInit",
    };
    for (const char* kernelName : threadKernels) {
        SCOPED_TRACE(kernelName);

        // 返回 0：args 指针与内容原样交给 kernel，status=0
        retVal->store(0);
        threadCount->store(0);
        const void* sentArgs = nullptr;
        ASSERT_EQ(ExecThreadKernel(kernelName, &sentArgs), 0);
        EXPECT_EQ(threadCount->load(), 1U);
        EXPECT_EQ(*lastArgs, sentArgs) << "args 指针须原样交给 kernel";
        EXPECT_EQ(*lastHead, kThreadArgsPattern) << "args 内容不得被转换改写";

        // kernel 非零返回：沿 status 链传回设备响应
        retVal->store(42);
        threadCount->store(0);
        ASSERT_EQ(ExecThreadKernel(kernelName, nullptr), 42);
        EXPECT_EQ(threadCount->load(), 1U);
    }
}
