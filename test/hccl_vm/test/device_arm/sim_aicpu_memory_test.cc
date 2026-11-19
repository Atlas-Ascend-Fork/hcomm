/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <set>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

#include "hccl_device_pub.h"
#include "sim_aicpu_pipe_handler.h"
#include "sim_aicpu_pipe_msg.h"
#include "sim_pipe_io.h"
#include "store_sim_comm_pool_policy.h"
#include "store_sim_resource_root.h"
#include "store_sim_shm_ops.h"

namespace {

class AicpuMemoryTest : public testing::Test {
protected:
    static constexpr size_t kPoolBytes = 2 * 1024 * 1024;
    static constexpr size_t kDataBytes = 524288;

    static void SetUpTestSuite()
    {
        const char* previous = std::getenv("HCCL_VM_RESOURCE_ROOT");
        hadRoot_ = previous != nullptr;
        oldRoot_ = previous == nullptr ? "" : previous;
        char pattern[] = "/tmp/hccl_vm_aicpu_memory_XXXXXX";
        const char* root = ::mkdtemp(pattern);
        ASSERT_NE(root, nullptr);
        root_ = root;
        ASSERT_EQ(::setenv("HCCL_VM_RESOURCE_ROOT", root_.c_str(), 1), 0);
    }

    static void TearDownTestSuite()
    {
        if (!root_.empty()) {
            std::filesystem::remove_all(root_);
        }
        if (hadRoot_) {
            ::setenv("HCCL_VM_RESOURCE_ROOT", oldRoot_.c_str(), 1);
        } else {
            ::unsetenv("HCCL_VM_RESOURCE_ROOT");
        }
    }

    void SetUp() override
    {
        ASSERT_EQ(::pipe(fds_), 0);
        InitPipeFds(fds_[0], fds_[1]);
        pool_ = ShmCreate(sim::CommPoolPolicy::kPoolName, kPoolBytes);
        ASSERT_NE(pool_, nullptr);
    }

    void TearDown() override
    {
        // 即使用例提前 ASSERT 失败，也释放所有已创建的 device 映射。
        for (const auto& name : mappings_) {
            Free(name);
        }
        mappings_.clear();
        if (pool_ != nullptr) {
            ShmClose(pool_);
        }
        InitPipeFds(-1, -1);
        for (int fd : fds_) {
            if (fd >= 0) {
                ::close(fd);
            }
        }
    }

    DevMemOpPayload Request(const std::string& name, size_t size, bool pooled = true)
    {
        DevMemOpPayload request{};
        std::strncpy(request.memName, name.c_str(), sizeof(request.memName) - 1);
        request.size = size;
        request.useCommPool = pooled;
        return request;
    }

    template <typename Response>
    Response Receive(uint8_t expectedCmd)
    {
        uint8_t cmd = 0;
        uint32_t len = 0;
        Response response{};
        EXPECT_EQ(sim::PipeRecvMsg(fds_[0], cmd, &response, sizeof(response), len), 0);
        EXPECT_EQ(cmd, expectedCmd);
        EXPECT_EQ(len, sizeof(response));
        return response;
    }

    uint8_t* Get(const std::string& name, size_t size = kDataBytes, bool pooled = true)
    {
        auto request = Request(name, size, pooled);
        EXPECT_EQ(sim::HandlePipeCmdGetDevPtr(reinterpret_cast<uint8_t*>(&request), sizeof(request)), 0);
        const auto response = Receive<RspGetDevPtrPayload>(PIPE_RSP_GET_DEV_PTR);
        if (response.ptr != 0) {
            mappings_.insert(name);
        }
        return reinterpret_cast<uint8_t*>(response.ptr);
    }

    void Free(const std::string& name)
    {
        auto request = Request(name, 0);
        EXPECT_EQ(sim::HandlePipeCmdFreeDevPtr(reinterpret_cast<uint8_t*>(&request), sizeof(request)), 0);
        EXPECT_EQ(Receive<RspFreeDevPtrPayload>(PIPE_RSP_FREE_DEV_PTR).status, 0);
    }

    void Release(const std::string& name)
    {
        Free(name);
        mappings_.erase(name);
    }

    void ExpectUnmapped(void* ptr)
    {
        const auto pageSize = static_cast<uintptr_t>(::sysconf(_SC_PAGESIZE));
        void* page = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(ptr) / pageSize * pageSize);
        unsigned char residency = 0;
        errno = 0;
        EXPECT_EQ(::mincore(page, pageSize, &residency), -1);
        EXPECT_EQ(errno, ENOMEM);
    }

    void* pool_ = nullptr;
    int fds_[2] = {-1, -1};
    std::set<std::string> mappings_;
    // SimResourceRoot 在进程内缓存根目录，因此整个套件使用同一个隔离目录。
    inline static std::string root_;
    inline static std::string oldRoot_;
    inline static bool hadRoot_ = false;
};

TEST_F(AicpuMemoryTest, DistinctAddressesSharePoolData)
{
    auto* first = Get("buffer_a");
    auto* second = Get("buffer_b");
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_NE(first, second);
    const auto firstAddr = reinterpret_cast<uintptr_t>(first);
    const auto secondAddr = reinterpret_cast<uintptr_t>(second);
    EXPECT_TRUE(firstAddr + kDataBytes <= secondAddr || secondAddr + kDataBytes <= firstAddr);
    first[0] = 0x12;
    first[kDataBytes - 1] = 0x34;
    EXPECT_EQ(second[0], 0x12);
    EXPECT_EQ(second[kDataBytes - 1], 0x34);
    EXPECT_EQ(static_cast<uint8_t*>(pool_)[kDataBytes - 1], 0x34);
}

#if defined(__FLT16_MANT_DIG__)
TEST_F(AicpuMemoryTest, CpuFp16ProdCanAccessSourceAt524288ByteOffset)
{
    // 复现本次访问形状：CCL buffer + 524288 偏移，读取 262144 个 FP16 元素。
    // 只验证映射可供 CPU 直接计算，不声称等价于 HCCL kernel 的端到端数值测试。
    static_assert(sizeof(_Float16) == 2, "FP16 element must occupy two bytes");
    auto* base = Get("reduce_source", 2 * kDataBytes);
    ASSERT_NE(base, nullptr);
    auto* src = reinterpret_cast<_Float16*>(base + kDataBytes);
    const size_t count = kDataBytes / sizeof(_Float16);
    std::vector<_Float16> dst(count, static_cast<_Float16>(3.0f));
    for (size_t i = 0; i < count; ++i) {
        src[i] = static_cast<_Float16>(2.0f);
        dst[i] = static_cast<_Float16>(static_cast<float>(src[i]) * static_cast<float>(dst[i]));
    }
    for (const auto value : dst) {
        ASSERT_EQ(static_cast<float>(value), 6.0f);
    }
    EXPECT_EQ(static_cast<float>(src[count - 1]), 2.0f);
}
#else
// x86 GCC<12 等工具链无 _Float16：退化为 uint16_t 复现同一访问形状
// （CCL buffer + 524288 偏移读写 262144 个 2 字节元素），保留 check-only
// 大块映射可供 CPU 直接访问的回归覆盖；FP16 数值语义在该平台不验证。
TEST_F(AicpuMemoryTest, CpuFp16ProdCanAccessSourceAt524288ByteOffset)
{
    auto* base = Get("reduce_source", 2 * kDataBytes);
    ASSERT_NE(base, nullptr);
    auto* src = reinterpret_cast<uint16_t*>(base + kDataBytes);
    const size_t count = kDataBytes / sizeof(uint16_t);
    std::vector<uint16_t> dst(count, 0x4200);
    for (size_t i = 0; i < count; ++i) {
        src[i] = 0x3C00;
        dst[i] = static_cast<uint16_t>(src[i] + dst[i]);
    }
    for (const auto value : dst) {
        ASSERT_EQ(value, 0x7E00);
    }
    EXPECT_EQ(src[count - 1], 0x3C00);
}
#endif

TEST_F(AicpuMemoryTest, FreeUnmapsOnlyTheNamedAllocation)
{
    auto* first = Get("buffer_a");
    auto* second = Get("buffer_b");
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    Release("buffer_a");
    ExpectUnmapped(first);
    second[kDataBytes - 1] = 0x56;
    EXPECT_EQ(static_cast<uint8_t*>(pool_)[kDataBytes - 1], 0x56);
    Release("buffer_b");
    ExpectUnmapped(second);
    EXPECT_TRUE(std::filesystem::exists(sim::SimResourceRoot::BuildName(sim::CommPoolPolicy::kPoolName)));
    static_cast<uint8_t*>(pool_)[0] = 0x78;
    EXPECT_EQ(static_cast<uint8_t*>(pool_)[0], 0x78);
}

TEST_F(AicpuMemoryTest, RepeatedGetUsesExistingNamedMapping)
{
    auto* first = Get("buffer_a");
    ASSERT_NE(first, nullptr);
    EXPECT_EQ(Get("buffer_a"), first);
    Release("buffer_a");
    ExpectUnmapped(first);
}

TEST_F(AicpuMemoryTest, ExitCleanupClosesOutstandingPoolMappings)
{
    auto* first = Get("buffer_a");
    auto* second = Get("buffer_b");
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    const auto* head = reinterpret_cast<const ShmHead*>(static_cast<const char*>(pool_) - sizeof(ShmHead));
    EXPECT_EQ(static_cast<uint32_t>(head->refCount), 3U);
    sim::CleanupCommPoolMappings();
    mappings_.clear();
    ExpectUnmapped(first);
    ExpectUnmapped(second);
    EXPECT_EQ(static_cast<uint32_t>(head->refCount), 1U);
    sim::CleanupCommPoolMappings(); // 重复清理无副作用。
}

TEST_F(AicpuMemoryTest, ExactPoolSizeIsAccessible)
{
    auto* ptr = Get("whole_pool", kPoolBytes);
    ASSERT_NE(ptr, nullptr);
    ptr[kPoolBytes - 1] = 0x9A;
    EXPECT_EQ(static_cast<uint8_t*>(pool_)[kPoolBytes - 1], 0x9A);
}

TEST_F(AicpuMemoryTest, RejectsZeroSizeAndAllowsRetry)
{
    EXPECT_EQ(Get("buffer_a", 0), nullptr);
    ASSERT_NE(Get("buffer_a"), nullptr);
}

TEST_F(AicpuMemoryTest, RejectsOversizedRequestWithoutClosingPool)
{
    EXPECT_EQ(Get("buffer_a", kPoolBytes + 1), nullptr);
    auto* ptr = Get("buffer_a");
    ASSERT_NE(ptr, nullptr);
    ptr[0] = 0xBC;
    EXPECT_EQ(static_cast<uint8_t*>(pool_)[0], 0xBC);
}

TEST_F(AicpuMemoryTest, MissingPoolReturnsNullAndAllowsRetry)
{
    ShmClose(pool_);
    pool_ = nullptr;
    EXPECT_EQ(Get("buffer_a"), nullptr);
    pool_ = ShmCreate(sim::CommPoolPolicy::kPoolName, kPoolBytes);
    ASSERT_NE(pool_, nullptr);
    ASSERT_NE(Get("buffer_a"), nullptr);
}

TEST_F(AicpuMemoryTest, OrdinaryAllocationUsesItsOwnBackingMemory)
{
    std::unique_ptr<void, decltype(&ShmClose)> owner(ShmCreate("ordinary", kDataBytes), &ShmClose);
    ASSERT_NE(owner.get(), nullptr);
    auto* ptr = Get("ordinary", kDataBytes, false);
    ASSERT_NE(ptr, nullptr);
    ptr[0] = 0xDE;
    EXPECT_EQ(static_cast<uint8_t*>(owner.get())[0], 0xDE);
    EXPECT_EQ(static_cast<uint8_t*>(pool_)[0], 0);
    Release("ordinary");
    ExpectUnmapped(ptr);
    EXPECT_EQ(static_cast<uint8_t*>(owner.get())[0], 0xDE);
}

TEST_F(AicpuMemoryTest, MalformedPayloadsReturnFailureWithoutRemovingExistingMapping)
{
    auto* ptr = Get("buffer_a");
    ASSERT_NE(ptr, nullptr);
    auto request = Request("buffer_a", kDataBytes);
    for (const uint32_t len :
         {0U, static_cast<uint32_t>(sizeof(request) - 1), static_cast<uint32_t>(sizeof(request) + 1)}) {
        EXPECT_NE(sim::HandlePipeCmdGetDevPtr(reinterpret_cast<uint8_t*>(&request), len), 0);
        EXPECT_EQ(Receive<RspGetDevPtrPayload>(PIPE_RSP_GET_DEV_PTR).ptr, 0U);
        EXPECT_NE(sim::HandlePipeCmdFreeDevPtr(reinterpret_cast<uint8_t*>(&request), len), 0);
        EXPECT_NE(Receive<RspFreeDevPtrPayload>(PIPE_RSP_FREE_DEV_PTR).status, 0);
    }
    ptr[0] = 0xEF;
    EXPECT_EQ(static_cast<uint8_t*>(pool_)[0], 0xEF);
}

} // namespace
