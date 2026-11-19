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

// RecordOpDbInfo has internal linkage. Include the implementation to exercise the
// production function without adding a test-only API to the proxy library.
#include "../../src/proxy/level0/hccl_op_stub.cc"

namespace {
constexpr uint64_t kInputBase = 0x10000;
constexpr uint64_t kOutputBase = 0x20000;
constexpr uint64_t kCclBase = 0x30000;
constexpr uint64_t kAllocationSize = 4096;
constexpr uint64_t kVirtualOffset = 0x100000;
constexpr uint64_t kCommId = 7;

struct RecordState {
    std::vector<uint64_t> lookups;
    sim::operation::OpDetailTab detail{};
    sim::operation::OpMemInfoTab mem{};
    unsigned inserts = 0;
    unsigned cclQueries = 0;
    unsigned cclUpdates = 0;
    bool emptyCcl = false;
} state;

HcclResult GetTestHcclBuffer(HcclComm comm, void** buffer, uint64_t* size)
{
    EXPECT_EQ(reinterpret_cast<uint64_t>(comm), kCommId);
    ++state.cclQueries;
    *buffer = state.emptyCcl ? nullptr : reinterpret_cast<void*>(kCclBase);
    *size = state.emptyCcl ? 0 : kAllocationSize;
    return HCCL_SUCCESS;
}

struct BufferCase {
    const char* name;
    uint64_t input;
    uint64_t inputSize;
    uint64_t output;
    uint64_t outputSize;
    uint64_t expectedInput;
    uint64_t expectedOutput;
};

class RecordOpDbInfoTest : public testing::TestWithParam<BufferCase> {
protected:
    void SetUp() override
    {
        state = {};
        savedCommId_ = sim::runtime::g_cur_comm_key;
        sim::runtime::g_cur_comm_key = kCommId;
    }

    void TearDown() override { sim::runtime::g_cur_comm_key = savedCommId_; }

    int Record(const BufferCase& buffers, uint32_t rankSize = 8)
    {
        return RecordOpDbInfo(
            HCCL_CMD_ALLTOALLV, 0, 11, reinterpret_cast<void*>(buffers.input), buffers.inputSize,
            reinterpret_cast<void*>(buffers.output), buffers.outputSize, {1, 2, 3}, 0, rankSize, 0, 0, {4, 5},
            reinterpret_cast<HcclComm>(kCommId));
    }

private:
    uint64_t savedCommId_ = 0;
};
} // namespace

// Replace only the dependencies of RecordOpDbInfo; address lookup models the
// same [base, end) contract as the real virtual-memory mapping.
namespace sim {
uint64_t GetVirPtrByDevPtr(uint64_t address)
{
    state.lookups.push_back(address);
    for (uint64_t base : {kInputBase, kOutputBase, kCclBase}) {
        if (address >= base && address < base + kAllocationSize) {
            return address + kVirtualOffset;
        }
    }
    ADD_FAILURE() << "Address lookup outside a registered allocation: " << address;
    return 0;
}
} // namespace sim

namespace sim::runtime {
thread_local uint64_t g_cur_comm_key = 0;

aclError GetDeviceByCommRank(uint64_t commId, uint32_t rankId, Device& device)
{
    EXPECT_EQ(commId, kCommId);
    device.id = rankId + 1;
    return ACL_SUCCESS;
}
} // namespace sim::runtime

namespace sim::operation {
int InsertOpDetailAndMem(OpDetailTab& detail, OpMemInfoTab& mem)
{
    ++state.inserts;
    state.detail = detail;
    state.mem = mem;
    return 0;
}

int UpdateOpMemCclBuffer(uint64_t address, uint64_t size)
{
    ++state.cclUpdates;
    state.mem.cclAddr = address;
    state.mem.cclSize = size;
    return 0;
}
} // namespace sim::operation

extern "C" HcclResult hrtGetDeviceType(DevType& deviceType)
{
    deviceType = static_cast<DevType>(0);
    return HCCL_SUCCESS;
}

extern "C" void* __real_dlsym(void* handle, const char* name);
extern "C" void* __wrap_dlsym(void* handle, const char* name)
{
    if (std::strcmp(name, "HcclGetHcclBuffer") == 0) {
        return reinterpret_cast<void*>(&GetTestHcclBuffer);
    }
    return __real_dlsym(handle, name);
}

TEST_P(RecordOpDbInfoTest, RecordsBufferLayoutAndStillUpdatesCcl)
{
    const auto& buffers = GetParam();
    ASSERT_EQ(Record(buffers), 0);
    ASSERT_EQ(state.inserts, 1u);
    EXPECT_EQ(state.mem.inputAddr, buffers.expectedInput);
    EXPECT_EQ(state.mem.inputSize, buffers.inputSize);
    EXPECT_EQ(state.mem.outputAddr, buffers.expectedOutput);
    EXPECT_EQ(state.mem.outputSize, buffers.outputSize);
    EXPECT_EQ(state.detail.commId, kCommId);
    EXPECT_EQ(state.detail.streamId, 11u);
    EXPECT_EQ(state.detail.rankSize, 8u);
    EXPECT_EQ(state.detail.opDetail, (std::vector<uint8_t>{1, 2, 3}));
    EXPECT_EQ(state.detail.opExtInfo, (std::vector<uint8_t>{4, 5}));

    std::vector<uint64_t> expectedLookups;
    if (buffers.expectedInput != 0) {
        expectedLookups.push_back(buffers.input);
    }
    if (buffers.expectedOutput != 0) {
        expectedLookups.push_back(buffers.output);
    }
    expectedLookups.push_back(kCclBase);
    EXPECT_EQ(state.lookups, expectedLookups);
    EXPECT_EQ(state.cclQueries, 1u);
    EXPECT_EQ(state.cclUpdates, 1u);
    EXPECT_EQ(state.mem.cclAddr, kCclBase + kVirtualOffset);
    EXPECT_EQ(state.mem.cclSize, kAllocationSize);
}

INSTANTIATE_TEST_SUITE_P(
    BufferLayouts, RecordOpDbInfoTest,
    testing::Values(
        BufferCase{"BothZeroAtAllocationEnd", kInputBase + kAllocationSize, 0, kOutputBase + kAllocationSize, 0, 0, 0},
        BufferCase{
            "ZeroInputAtAllocationEnd", kInputBase + kAllocationSize, 0, kOutputBase + 32, 64, 0,
            kOutputBase + 32 + kVirtualOffset},
        BufferCase{
            "ZeroOutputAtAllocationEnd", kInputBase + 16, 64, kOutputBase + kAllocationSize, 0,
            kInputBase + 16 + kVirtualOffset, 0},
        BufferCase{
            "BothNonzero", kInputBase + 16, 64, kOutputBase + 32, 128, kInputBase + 16 + kVirtualOffset,
            kOutputBase + 32 + kVirtualOffset},
        BufferCase{"ZeroSizeMappedPointers", kInputBase, 0, kOutputBase, 0, 0, 0},
        BufferCase{"BothNull", 0, 0, 0, 0, 0, 0},
        BufferCase{"NullInputWithNonzeroSize", 0, 64, kOutputBase, 64, 0, kOutputBase + kVirtualOffset},
        BufferCase{"NullOutputWithNonzeroSize", kInputBase, 64, 0, 64, kInputBase + kVirtualOffset, 0},
        BufferCase{
            "InPlaceNonzero", kInputBase, 64, kInputBase, 64, kInputBase + kVirtualOffset,
            kInputBase + kVirtualOffset}),
    [](const testing::TestParamInfo<BufferCase>& info) {
        return info.param.name;
    });

TEST_F(RecordOpDbInfoTest, SingleRankStillAcceptsEmptyCclBuffer)
{
    state.emptyCcl = true;
    const BufferCase buffers{"Zero", kInputBase + kAllocationSize, 0, kOutputBase + kAllocationSize, 0, 0, 0};
    ASSERT_EQ(Record(buffers, 1), 0);
    EXPECT_EQ(state.inserts, 1u);
    EXPECT_TRUE(state.lookups.empty());
    EXPECT_EQ(state.mem.inputAddr, 0u);
    EXPECT_EQ(state.mem.outputAddr, 0u);
    EXPECT_EQ(state.cclQueries, 1u);
    EXPECT_EQ(state.cclUpdates, 0u);
}
