/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>
#include <securec.h>
#include <cstring>
#include <string>
#include <vector>

#define private public
#include "endpoint_remote_reged_mem_mgr.h"
#undef private

using namespace hcomm;

namespace {

// 构造 memDesc：DTO 序列化 + EndpointDesc + u64 pid（与 GetMemDesc 写入布局一致）
std::vector<char> BuildMemDesc(
    uintptr_t addr, uint64_t size, const EndpointDesc& ep, uint64_t pid, HcclMemType memType = HCCL_MEM_TYPE_DEVICE)
{
    Hccl::ExchangeRdmaBufferDto dto(addr, size, 42U, memType, "ut");
    Hccl::BinaryStream stream;
    dto.Serialize(stream);
    std::vector<char> desc;
    stream.Dump(desc);

    std::vector<char> tail(sizeof(EndpointDesc) + sizeof(uint64_t));
    EXPECT_EQ(memcpy_s(tail.data(), sizeof(EndpointDesc), &ep, sizeof(EndpointDesc)), EOK);
    EXPECT_EQ(memcpy_s(tail.data() + sizeof(EndpointDesc), sizeof(uint64_t), &pid, sizeof(pid)), EOK);
    desc.insert(desc.end(), tail.begin(), tail.end());
    return desc;
}

std::vector<char>
BuildUbMemDesc(uintptr_t addr, uint64_t size, HcclMemType memType, const EndpointDesc& ep, uint64_t pid)
{
    Hccl::ExchangeUbBufferDto dto(addr, size, memType, "ut", 0U, 0U, 0U);
    Hccl::BinaryStream stream;
    dto.Serialize(stream);
    std::vector<char> desc;
    stream.Dump(desc);

    std::vector<char> tail(sizeof(EndpointDesc) + sizeof(uint64_t));
    EXPECT_EQ(memcpy_s(tail.data(), sizeof(EndpointDesc), &ep, sizeof(EndpointDesc)), EOK);
    EXPECT_EQ(memcpy_s(tail.data() + sizeof(EndpointDesc), sizeof(uint64_t), &pid, sizeof(pid)), EOK);
    desc.insert(desc.end(), tail.begin(), tail.end());
    return desc;
}

} // namespace

class EndpointRemoteMemMgrTest : public testing::Test {
protected:
    void TearDown() override
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
    }
};

TEST_F(EndpointRemoteMemMgrTest, Ut_MemoryImport_When_MemDescNull_Expect_Return_Ptr)
{
    EndpointRemoteRegedMemMgr mgr(nullptr, ParseRoceMemDesc, CreateRoceRemoteBuffer);
    HcommMem out{};
    EXPECT_EQ(mgr.MemoryImport(nullptr, 100U, &out), HCCL_E_PTR);
}

TEST_F(EndpointRemoteMemMgrTest, Ut_MemoryImport_When_OutMemNull_Expect_Return_Ptr)
{
    EndpointRemoteRegedMemMgr mgr(nullptr, ParseRoceMemDesc, CreateRoceRemoteBuffer);
    EndpointDesc ep{};
    auto desc = BuildMemDesc(0x1000, 0x100, ep, 1234U);
    EXPECT_EQ(mgr.MemoryImport(desc.data(), static_cast<uint32_t>(desc.size()), nullptr), HCCL_E_PTR);
}

// 不带 pid 的旧格式 desc（长度不足）直接报错，强制两端同版本
TEST_F(EndpointRemoteMemMgrTest, Ut_MemoryImport_When_OldFormatDesc_Expect_Return_Internal)
{
    EndpointRemoteRegedMemMgr mgr(nullptr, ParseRoceMemDesc, CreateRoceRemoteBuffer);
    EndpointDesc ep{};
    char buf[sizeof(EndpointDesc)] = {};
    EXPECT_EQ(memcpy_s(buf, sizeof(EndpointDesc), &ep, sizeof(EndpointDesc)), EOK);
    HcommMem out{};
    EXPECT_EQ(mgr.MemoryImport(buf, sizeof(EndpointDesc), &out), HCCL_E_INTERNAL);
}

// 长度达标的旧格式 desc（无 pid 且 DTO 不含 memType 字段）会被 DTO 回读校验拦截，验证不会错位解析成错误归属键
TEST_F(EndpointRemoteMemMgrTest, Ut_MemoryImport_When_OldFormatDescLongEnough_Expect_Return_Internal)
{
    EndpointRemoteRegedMemMgr mgr(nullptr, ParseRoceMemDesc, CreateRoceRemoteBuffer);
    EndpointDesc ep{};
    // 手写历史版本 DTO 字节流：仅 addr/size/rkey/memInfo 四字段，不含 memType
    Hccl::BinaryStream stream;
    u64 addr = 0x1000;
    u64 size = 0x100;
    u32 rkey = 42U;
    std::string memInfo = "ut-mem-info";
    stream << addr << size << rkey << memInfo;
    std::vector<char> desc;
    stream.Dump(desc);
    std::vector<char> epTail(sizeof(EndpointDesc));
    EXPECT_EQ(memcpy_s(epTail.data(), sizeof(EndpointDesc), &ep, sizeof(EndpointDesc)), EOK);
    desc.insert(desc.end(), epTail.begin(), epTail.end());

    HcommMem out{};
    EXPECT_EQ(mgr.MemoryImport(desc.data(), static_cast<uint32_t>(desc.size()), &out), HCCL_E_INTERNAL);
    EXPECT_EQ(mgr.remoteRmaBufferMgrs_.size(), 0U);
}

// 同 {endpointDesc, pid} 重复导入：幂等成功，ref 叠加，不报 E_AGAIN
TEST_F(EndpointRemoteMemMgrTest, Ut_MemoryImport_When_SameDescTwice_Expect_IdempotentRefInc)
{
    EndpointRemoteRegedMemMgr mgr(nullptr, ParseRoceMemDesc, CreateRoceRemoteBuffer);
    EndpointDesc ep{};
    auto desc = BuildMemDesc(0x1000, 0x100, ep, 1234U);
    uint32_t descLen = static_cast<uint32_t>(desc.size());

    HcommMem out{};
    EXPECT_EQ(mgr.MemoryImport(desc.data(), descLen, &out), HCCL_SUCCESS);
    EXPECT_EQ(mgr.MemoryImport(desc.data(), descLen, &out), HCCL_SUCCESS);
    EXPECT_EQ(mgr.remoteRmaBufferMgrs_.size(), 1U);

    auto& bufferMgr = mgr.remoteRmaBufferMgrs_.begin()->second;
    ASSERT_NE(bufferMgr, nullptr);
    EXPECT_EQ(bufferMgr->size(), 1U);
    EXPECT_EQ(bufferMgr->Begin()->second.ref, 2U);
}

// RoCE desc 携带内存类型：out.type 返回 desc 中的类型，覆盖调用者传入的初值
TEST_F(EndpointRemoteMemMgrTest, Ut_MemoryImport_When_RoceDescMemTypeValid_Expect_Return_DescriptorMemType)
{
    struct TestCase {
        HcclMemType hcclType;
        CommMemType expectedType;
        uintptr_t addr;
    };
    const TestCase cases[] = {
        {HCCL_MEM_TYPE_DEVICE, COMM_MEM_TYPE_DEVICE, 0x5800},
        {HCCL_MEM_TYPE_HOST, COMM_MEM_TYPE_HOST, 0x5900},
    };

    EndpointRemoteRegedMemMgr mgr(nullptr, ParseRoceMemDesc, CreateRoceRemoteBuffer);
    EndpointDesc ep{};
    for (const auto& testCase : cases) {
        auto desc = BuildMemDesc(testCase.addr, 0x100, ep, 1234U, testCase.hcclType);

        HcommMem out{COMM_MEM_TYPE_INVALID, nullptr, 0};
        ASSERT_EQ(mgr.MemoryImport(desc.data(), static_cast<uint32_t>(desc.size()), &out), HCCL_SUCCESS);
        EXPECT_EQ(out.type, testCase.expectedType);
        EXPECT_EQ(out.addr, reinterpret_cast<void*>(testCase.addr));
        EXPECT_EQ(out.size, 0x100U);
    }
}

// RoCE desc 内存类型非法（如默认哨兵值 NUM）：拒绝导入并复位出参
TEST_F(EndpointRemoteMemMgrTest, Ut_MemoryImport_When_RoceMemTypeInvalid_Expect_Return_ParaAndResetOutput)
{
    EndpointRemoteRegedMemMgr mgr(nullptr, ParseRoceMemDesc, CreateRoceRemoteBuffer);
    EndpointDesc ep{};
    auto desc = BuildMemDesc(0x8800, 0x100, ep, 1234U, HCCL_MEM_TYPE_NUM);

    HcommMem out{COMM_MEM_TYPE_HOST, reinterpret_cast<void*>(0x1), 1U};
    EXPECT_EQ(mgr.MemoryImport(desc.data(), static_cast<uint32_t>(desc.size()), &out), HCCL_E_PARA);
    EXPECT_EQ(out.type, COMM_MEM_TYPE_INVALID);
    EXPECT_EQ(out.addr, nullptr);
    EXPECT_EQ(out.size, 0U);
    EXPECT_TRUE(mgr.remoteRmaBufferMgrs_.empty());
}

// 同一 RoCE 内存重复导入且 desc 类型冲突：拒绝并保持首次导入的引用计数
TEST_F(EndpointRemoteMemMgrTest, Ut_MemoryImport_When_ReusedRoceMemTypeMismatch_Expect_Return_ParaAndKeepRef)
{
    EndpointRemoteRegedMemMgr mgr(nullptr, ParseRoceMemDesc, CreateRoceRemoteBuffer);
    EndpointDesc ep{};
    auto deviceDesc = BuildMemDesc(0x9800, 0x100, ep, 1234U, HCCL_MEM_TYPE_DEVICE);
    auto hostDesc = BuildMemDesc(0x9800, 0x100, ep, 1234U, HCCL_MEM_TYPE_HOST);

    HcommMem out{};
    ASSERT_EQ(mgr.MemoryImport(deviceDesc.data(), static_cast<uint32_t>(deviceDesc.size()), &out), HCCL_SUCCESS);
    EXPECT_EQ(out.type, COMM_MEM_TYPE_DEVICE);

    EXPECT_EQ(mgr.MemoryImport(hostDesc.data(), static_cast<uint32_t>(hostDesc.size()), &out), HCCL_E_PARA);
    EXPECT_EQ(out.type, COMM_MEM_TYPE_INVALID);
    EXPECT_EQ(out.addr, nullptr);
    EXPECT_EQ(out.size, 0U);

    auto& bufferMgr = mgr.remoteRmaBufferMgrs_.begin()->second;
    ASSERT_NE(bufferMgr, nullptr);
    ASSERT_EQ(bufferMgr->size(), 1U);
    EXPECT_EQ(bufferMgr->Begin()->second.ref, 1U);
}

TEST_F(EndpointRemoteMemMgrTest, Ut_MemoryImport_When_UbDeviceOrHost_Expect_Return_DescriptorMemType)
{
    struct TestCase {
        HcclMemType hcclType;
        CommMemType expectedType;
        uintptr_t addr;
    };
    const TestCase cases[] = {
        {HCCL_MEM_TYPE_DEVICE, COMM_MEM_TYPE_DEVICE, 0x6000},
        {HCCL_MEM_TYPE_HOST, COMM_MEM_TYPE_HOST, 0x7000},
    };

    EndpointDesc ep{};
    for (const auto& testCase : cases) {
        auto testUbCreator
            = [](RdmaHandle, const Hccl::Serializable& serializable) -> std::shared_ptr<Hccl::RemoteRmaBuffer> {
            const auto& dto = dynamic_cast<const Hccl::ExchangeUbBufferDto&>(serializable);
            return std::make_shared<Hccl::RemoteUbRmaBuffer>(
                dto.addr, dto.size, dto.tokenId, dto.tokenValue, dto.memType, dto.memInfo);
        };
        EndpointRemoteRegedMemMgr mgr(nullptr, ParseUbMemDesc, testUbCreator);
        auto desc = BuildUbMemDesc(testCase.addr, 0x100, testCase.hcclType, ep, 1234U);
        uint32_t descLen = static_cast<uint32_t>(desc.size());

        HcommMem out{};
        out.type = COMM_MEM_TYPE_INVALID;
        ASSERT_EQ(mgr.MemoryImport(desc.data(), descLen, &out), HCCL_SUCCESS);
        EXPECT_EQ(out.type, testCase.expectedType);
        EXPECT_EQ(out.addr, reinterpret_cast<void*>(testCase.addr));
        EXPECT_EQ(out.size, 0x100U);

        out.type = COMM_MEM_TYPE_INVALID;
        ASSERT_EQ(mgr.MemoryImport(desc.data(), descLen, &out), HCCL_SUCCESS);
        EXPECT_EQ(out.type, testCase.expectedType);
    }
}

TEST_F(EndpointRemoteMemMgrTest, Ut_MemoryImport_When_UbMemTypeInvalid_Expect_Return_ParaAndResetOutput)
{
    EndpointRemoteRegedMemMgr mgr(nullptr, ParseUbMemDesc, CreateUbRemoteBuffer);
    EndpointDesc ep{};
    auto desc = BuildUbMemDesc(0x8000, 0x100, HCCL_MEM_TYPE_NUM, ep, 1234U);

    HcommMem out{COMM_MEM_TYPE_HOST, reinterpret_cast<void*>(0x1), 1U};
    EXPECT_EQ(mgr.MemoryImport(desc.data(), static_cast<uint32_t>(desc.size()), &out), HCCL_E_PARA);
    EXPECT_EQ(out.type, COMM_MEM_TYPE_INVALID);
    EXPECT_EQ(out.addr, nullptr);
    EXPECT_EQ(out.size, 0U);
    EXPECT_TRUE(mgr.remoteRmaBufferMgrs_.empty());
}

TEST_F(EndpointRemoteMemMgrTest, Ut_MemoryImport_When_ReusedUbMemTypeMismatch_Expect_Return_ParaAndKeepRef)
{
    uint32_t creatorCalls = 0;
    auto countingCreator =
        [&creatorCalls](RdmaHandle, const Hccl::Serializable& serializable) -> std::shared_ptr<Hccl::RemoteRmaBuffer> {
        creatorCalls++;
        const auto& dto = dynamic_cast<const Hccl::ExchangeUbBufferDto&>(serializable);
        return std::make_shared<Hccl::RemoteUbRmaBuffer>(
            dto.addr, dto.size, dto.tokenId, dto.tokenValue, dto.memType, dto.memInfo);
    };
    EndpointRemoteRegedMemMgr mgr(nullptr, ParseUbMemDesc, countingCreator);
    EndpointDesc ep{};
    auto deviceDesc = BuildUbMemDesc(0x9000, 0x100, HCCL_MEM_TYPE_DEVICE, ep, 1234U);
    auto hostDesc = BuildUbMemDesc(0x9000, 0x100, HCCL_MEM_TYPE_HOST, ep, 1234U);

    HcommMem out{};
    ASSERT_EQ(mgr.MemoryImport(deviceDesc.data(), static_cast<uint32_t>(deviceDesc.size()), &out), HCCL_SUCCESS);
    EXPECT_EQ(out.type, COMM_MEM_TYPE_DEVICE);

    EXPECT_EQ(mgr.MemoryImport(hostDesc.data(), static_cast<uint32_t>(hostDesc.size()), &out), HCCL_E_PARA);
    EXPECT_EQ(out.type, COMM_MEM_TYPE_INVALID);
    EXPECT_EQ(out.addr, nullptr);
    EXPECT_EQ(out.size, 0U);
    EXPECT_EQ(creatorCalls, 1U);

    auto& bufferMgr = mgr.remoteRmaBufferMgrs_.begin()->second;
    ASSERT_NE(bufferMgr, nullptr);
    ASSERT_EQ(bufferMgr->size(), 1U);
    EXPECT_EQ(bufferMgr->Begin()->second.ref, 1U);
}

// 重复导入跳过 creator_：creator 仅在首次导入调用一次（Ub 族 creator 含硬件 import，避免重复导入空转）
TEST_F(EndpointRemoteMemMgrTest, Ut_MemoryImport_When_Duplicate_Expect_CreatorCalledOnce)
{
    uint32_t creatorCalls = 0;
    auto countingCreator = [&creatorCalls](RdmaHandle handle, const Hccl::Serializable& dto) {
        creatorCalls++;
        return CreateRoceRemoteBuffer(handle, dto);
    };
    EndpointRemoteRegedMemMgr mgr(nullptr, ParseRoceMemDesc, countingCreator);
    EndpointDesc ep{};
    auto desc = BuildMemDesc(0x5000, 0x100, ep, 1234U);
    uint32_t descLen = static_cast<uint32_t>(desc.size());

    HcommMem out{};
    ASSERT_EQ(mgr.MemoryImport(desc.data(), descLen, &out), HCCL_SUCCESS);
    ASSERT_EQ(mgr.MemoryImport(desc.data(), descLen, &out), HCCL_SUCCESS);
    EXPECT_EQ(creatorCalls, 1U);
    EXPECT_EQ(mgr.remoteRmaBufferMgrs_.begin()->second->Begin()->second.ref, 2U);

    // 两次 unimport 走对称的引用递减：第一次仅减引用，第二次真正删除
    EXPECT_EQ(mgr.MemoryUnimport(desc.data(), descLen), HCCL_SUCCESS);
    EXPECT_EQ(mgr.remoteRmaBufferMgrs_.begin()->second->size(), 1U);
    EXPECT_EQ(mgr.MemoryUnimport(desc.data(), descLen), HCCL_SUCCESS);
    EXPECT_TRUE(mgr.remoteRmaBufferMgrs_.empty());
}

// 同 endpointDesc 不同 pid：各自独立管理，互不冲突
TEST_F(EndpointRemoteMemMgrTest, Ut_MemoryImport_When_DifferentPid_Expect_SeparateOwners)
{
    EndpointRemoteRegedMemMgr mgr(nullptr, ParseRoceMemDesc, CreateRoceRemoteBuffer);
    EndpointDesc ep{};
    auto desc1 = BuildMemDesc(0x2000, 0x100, ep, 1111U);
    auto desc2 = BuildMemDesc(0x2000, 0x100, ep, 2222U);

    HcommMem out{};
    EXPECT_EQ(mgr.MemoryImport(desc1.data(), static_cast<uint32_t>(desc1.size()), &out), HCCL_SUCCESS);
    EXPECT_EQ(mgr.MemoryImport(desc2.data(), static_cast<uint32_t>(desc2.size()), &out), HCCL_SUCCESS);
    EXPECT_EQ(mgr.remoteRmaBufferMgrs_.size(), 2U);
}

// 两次导入后逐次 unimport：第一次只减引用，最后一次才真正删除并清理记录
TEST_F(EndpointRemoteMemMgrTest, Ut_MemoryUnimport_When_LastRef_Expect_OwnerErased)
{
    EndpointRemoteRegedMemMgr mgr(nullptr, ParseRoceMemDesc, CreateRoceRemoteBuffer);
    EndpointDesc ep{};
    auto desc = BuildMemDesc(0x3000, 0x100, ep, 1234U);
    uint32_t descLen = static_cast<uint32_t>(desc.size());

    HcommMem out{};
    ASSERT_EQ(mgr.MemoryImport(desc.data(), descLen, &out), HCCL_SUCCESS);
    ASSERT_EQ(mgr.MemoryImport(desc.data(), descLen, &out), HCCL_SUCCESS);

    EXPECT_EQ(mgr.MemoryUnimport(desc.data(), descLen), HCCL_SUCCESS);
    EXPECT_EQ(mgr.remoteRmaBufferMgrs_.size(), 1U);
    EXPECT_EQ(mgr.remoteRmaBufferMgrs_.begin()->second->size(), 1U);

    EXPECT_EQ(mgr.MemoryUnimport(desc.data(), descLen), HCCL_SUCCESS);
    EXPECT_TRUE(mgr.remoteRmaBufferMgrs_.empty());
}

TEST_F(EndpointRemoteMemMgrTest, Ut_MemoryUnimport_When_OwnerMissing_Expect_Return_NotFound)
{
    EndpointRemoteRegedMemMgr mgr(nullptr, ParseRoceMemDesc, CreateRoceRemoteBuffer);
    EndpointDesc ep{};
    auto desc = BuildMemDesc(0x4000, 0x100, ep, 1234U);
    EXPECT_EQ(mgr.MemoryUnimport(desc.data(), static_cast<uint32_t>(desc.size())), HCCL_E_NOT_FOUND);
}

TEST_F(EndpointRemoteMemMgrTest, Ut_MemoryUnimport_When_MemDescNull_Expect_Return_Ptr)
{
    EndpointRemoteRegedMemMgr mgr(nullptr, ParseRoceMemDesc, CreateRoceRemoteBuffer);
    EXPECT_EQ(mgr.MemoryUnimport(nullptr, 100U), HCCL_E_PTR);
}
