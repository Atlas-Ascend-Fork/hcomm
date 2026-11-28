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
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>

#include "local_ub_rma_buffer.h"
#include "local_ipc_rma_buffer_v2.h"
#include "exchange_ub_buffer_dto.h"
#include "exchange_ipc_buffer_dto.h"
#include "exchange_rdma_buffer_dto.h"
#include "acl/acl.h"
#include "internal_exception.h"
#define private public
#define protected public
#include "dev_buffer.h"
#include "rma_buffer.h"
#include "remote_rma_buffer.h"
#include "buffer_type.h"
#undef protected
#undef private

using namespace Hccl;

static aclError AclsysGetVersionNumHigh(char* pkgName, int32_t* versionNum)
{
    *versionNum = 90200000;
    return ACL_SUCCESS;
}

static uint64_t g_capturedSetAttrValue = 0;
static uint32_t g_capturedSetAttrCallCount = 0;

static aclError AclrtIpcMemSetAttrSuccess(const char* key, aclrtIpcMemAttrType type, uint64_t attr)
{
    g_capturedSetAttrValue = attr;
    ++g_capturedSetAttrCallCount;
    return ACL_SUCCESS;
}

static aclError AclrtIpcMemSetAttrLinkNotSupported(const char* key, aclrtIpcMemAttrType type, uint64_t attr)
{
    g_capturedSetAttrValue = attr;
    ++g_capturedSetAttrCallCount;
    return ACL_ERROR_RT_LINK_TYPE_NOT_SUPPORTED;
}

class RemoteRmaBufferTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RemoteRmaBuffer tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "RemoteRmaBuffer tests tear down." << std::endl; }

    virtual void SetUp()
    {
        devBuf = DevBuffer::Create(0x100, 0x100);
        std::cout << "A Test case in RemoteRmaBuffer SetUP." << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in RemoteRmaBuffer TearDown." << std::endl;
    }
    std::shared_ptr<DevBuffer> devBuf;
};

TEST_F(RemoteRmaBufferTest, remoterdmarmabuffer_construct_with_dto)
{
    ExchangeRdmaBufferDto dto;
    dto.addr = 0x114;
    dto.size = 0x514;
    dto.rkey = 1919;
    dto.memType = HcclMemType::HCCL_MEM_TYPE_HOST;
    dto.memInfo = "testTag";
    RdmaHandle rdmaHandle = (RdmaHandle)0x1000000;
    RemoteRdmaRmaBuffer buffer(rdmaHandle, dto);
    EXPECT_EQ(dto.addr, buffer.GetAddr());
    EXPECT_EQ(dto.size, buffer.GetSize());
    EXPECT_EQ(dto.rkey, buffer.GetRkey());
    EXPECT_EQ(dto.memType, buffer.GetMemType());
    EXPECT_STREQ(dto.memInfo.c_str(), buffer.GetMemInfo().c_str());
};

TEST_F(RemoteRmaBufferTest, exchangerdmabufferdto_serialize_roundtrip)
{
    // 显式携带类型：序列化往返后保持
    ExchangeRdmaBufferDto srcDto(0x114, 0x514, 1919, HcclMemType::HCCL_MEM_TYPE_HOST, "testTag");
    BinaryStream stream;
    srcDto.Serialize(stream);
    ExchangeRdmaBufferDto dstDto;
    dstDto.Deserialize(stream);
    EXPECT_EQ(dstDto.addr, 0x114);
    EXPECT_EQ(dstDto.size, 0x514);
    EXPECT_EQ(dstDto.rkey, 1919U);
    EXPECT_EQ(dstDto.memType, HcclMemType::HCCL_MEM_TYPE_HOST);
    EXPECT_STREQ(dstDto.memInfo.c_str(), "testTag");

    // 4 参构造默认哨兵值：往返后保持无效，漏传处以 INVALID 暴露
    ExchangeRdmaBufferDto defaultDto(0x200, 0x100, 42, "defaultTag");
    BinaryStream defaultStream;
    defaultDto.Serialize(defaultStream);
    ExchangeRdmaBufferDto parsedDto;
    parsedDto.Deserialize(defaultStream);
    EXPECT_EQ(parsedDto.memType, HcclMemType::HCCL_MEM_TYPE_NUM);
};

TEST_F(RemoteRmaBufferTest, remoteubrmabuffer_construct_error)
{
    EXPECT_THROW(RemoteUbRmaBuffer remoteUbRmaBuffer(nullptr), NullPtrException);
};

TEST_F(RemoteRmaBufferTest, remoteubrmabuffer_deserialize_success)
{
    // construct buffer
    BufferType type = BufferType::INPUT;
    void* ptr = nullptr;
    u64 size = 0;
    bool remoteAccess = true;

    std::shared_ptr<DevBuffer> devBuf = DevBuffer::Create(0x100, 0x100);
    Buffer* buf = devBuf.get();

    HcclMemType memType = HcclMemType::HCCL_MEM_TYPE_HOST;
    u32 tokenValue = 1;
    u32 tokenId = 0;
    u64 keySize = 4;
    u8 key[HRT_UB_MEM_KEY_MAX_LEN]{};

    BinaryStream binaryStream;
    binaryStream << buf->GetAddr() << buf->GetSize() << memType << tokenValue << tokenId << keySize << key;
    ExchangeUbBufferDto dto;
    dto.Deserialize(binaryStream);

    RdmaHandle rdmaHandle = (void*)0x1000000;
    RemoteUbRmaBuffer remoteUbRmaBuffer(rdmaHandle, dto);

    UbRmaBufferExchangeData exchangeData;
    exchangeData.addr = buf->GetAddr();
    exchangeData.size = buf->GetSize();
    exchangeData.tokenValue = 1;
    exchangeData.tokenId = 0;
    memcpy_s(exchangeData.key, HRT_UB_MEM_KEY_MAX_LEN, key, HRT_UB_MEM_KEY_MAX_LEN);

    EXPECT_EQ(exchangeData.tokenId, remoteUbRmaBuffer.tokenId);
    EXPECT_EQ(memcmp(remoteUbRmaBuffer.key, exchangeData.key, HRT_UB_MEM_KEY_MAX_LEN), 0);
    EXPECT_EQ(buf->GetAddr(), remoteUbRmaBuffer.addr);
    EXPECT_EQ(buf->GetSize(), remoteUbRmaBuffer.size);
};

TEST_F(RemoteRmaBufferTest, remoterdmarmabuffer_describe_size)
{
    RdmaHandle rdmaHandle = (void*)0x1000000;
    RemoteRdmaRmaBuffer remoteRdmaRmaBuffer(rdmaHandle);

    std::string fakeKeyDesc = "fakeKeyDesc";
    MOCKER(HrtRaGetKeyDescribe).stubs().will(returnValue(fakeKeyDesc));
    remoteRdmaRmaBuffer.Describe();
}

TEST_F(RemoteRmaBufferTest, remoteubrmabuffer_construct_with_addr_device)
{
    uintptr_t addr = 0x12345678;
    u64 size = 0x1000;
    u32 tokenId = 42;
    u32 tokenValue = 7;
    HcclMemType memType = HCCL_MEM_TYPE_DEVICE;
    std::string memInfo = "test_device_mem";

    RemoteUbRmaBuffer buffer(addr, size, tokenId, tokenValue, memType, memInfo);

    EXPECT_EQ(buffer.GetAddr(), addr);
    EXPECT_EQ(buffer.GetSize(), size);
    EXPECT_EQ(buffer.tokenId, tokenId);
    EXPECT_EQ(buffer.tokenValue, tokenValue);
    EXPECT_EQ(buffer.GetMemType(), HCCL_MEM_TYPE_DEVICE);
    EXPECT_STREQ(buffer.GetMemInfo().c_str(), memInfo.c_str());
    EXPECT_EQ(buffer.GetRmaType(), RmaType::UB);
};

TEST_F(RemoteRmaBufferTest, remoteipcrmabuffer_open_new_path_success)
{
    ExchangeIpcBufferDto dto;
    dto.addr = 0x1000;
    dto.offset = 0x100;
    dto.size = 0x200;
    dto.pid = 0;
    dto.memInfo = "test";
    (void)memcpy_s(dto.name, RTS_IPC_MEM_NAME_LEN, "fakeName", sizeof("fakeName"));

    void* fakePtr = reinterpret_cast<void*>(0x2000);
    MOCKER(HrtDeviceGetBareTgid)
        .stubs()
        .with(outBound(static_cast<s32>(100)))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(aclsysGetVersionNum).stubs().will(invoke(AclsysGetVersionNumHigh));
    MOCKER(aclrtIpcMemSetAttr).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtIpcMemImportByKey).stubs().with(outBoundP(&fakePtr, sizeof(fakePtr))).will(returnValue(ACL_SUCCESS));

    RemoteIpcRmaBuffer buffer(dto, 0);

    EXPECT_TRUE(buffer.isOpened);
    EXPECT_EQ(buffer.GetAddr(), reinterpret_cast<uintptr_t>(fakePtr) + dto.offset);
};

TEST_F(RemoteRmaBufferTest, remoteipcrmabuffer_open_new_path_with_link_type_not_supported_fallback)
{
    ExchangeIpcBufferDto dto;
    dto.addr = 0x1000;
    dto.offset = 0x100;
    dto.size = 0x200;
    dto.pid = 0;
    dto.memInfo = "test";
    (void)memcpy_s(dto.name, RTS_IPC_MEM_NAME_LEN, "fakeName", sizeof("fakeName"));

    void* fakePtr = reinterpret_cast<void*>(0x2000);
    MOCKER(HrtDeviceGetBareTgid)
        .stubs()
        .with(outBound(static_cast<s32>(100)))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(aclsysGetVersionNum).stubs().will(invoke(AclsysGetVersionNumHigh));
    MOCKER(aclrtIpcMemSetAttr)
        .stubs()
        .will(returnValue(ACL_ERROR_RT_LINK_TYPE_NOT_SUPPORTED))
        .then(returnValue(ACL_SUCCESS));
    MOCKER(aclrtIpcMemImportByKey).stubs().with(outBoundP(&fakePtr, sizeof(fakePtr))).will(returnValue(ACL_SUCCESS));

    RemoteIpcRmaBuffer buffer(dto, 0);

    EXPECT_TRUE(buffer.isOpened);
    EXPECT_EQ(buffer.GetAddr(), reinterpret_cast<uintptr_t>(fakePtr) + dto.offset);
};

TEST_F(RemoteRmaBufferTest, remoteipcrmabuffer_open_new_path_ipc_open_fail)
{
    ExchangeIpcBufferDto dto;
    dto.addr = 0x1000;
    dto.offset = 0x100;
    dto.size = 0x200;
    dto.pid = 0;
    dto.memInfo = "test";
    (void)memcpy_s(dto.name, RTS_IPC_MEM_NAME_LEN, "fakeName", sizeof("fakeName"));

    MOCKER(HrtDeviceGetBareTgid)
        .stubs()
        .with(outBound(static_cast<s32>(100)))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(aclsysGetVersionNum).stubs().will(invoke(AclsysGetVersionNumHigh));
    MOCKER(aclrtIpcMemSetAttr).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtIpcMemImportByKey).stubs().will(returnValue(1));

    EXPECT_THROW(RemoteIpcRmaBuffer buffer(dto, 0), InternalException);
};

TEST_F(RemoteRmaBufferTest, remoteipcrmabuffer_open_new_path_with_pathmode_one_maps_to_onepath)
{
    ExchangeIpcBufferDto dto;
    dto.addr = 0x1000;
    dto.offset = 0x100;
    dto.size = 0x200;
    dto.pid = 0;
    dto.memInfo = "test";
    (void)memcpy_s(dto.name, RTS_IPC_MEM_NAME_LEN, "fakeName", sizeof("fakeName"));

    g_capturedSetAttrValue = 0;
    g_capturedSetAttrCallCount = 0;
    void* fakePtr = reinterpret_cast<void*>(0x2000);
    MOCKER(HrtDeviceGetBareTgid)
        .stubs()
        .with(outBound(static_cast<s32>(100)))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(aclsysGetVersionNum).stubs().will(invoke(AclsysGetVersionNumHigh));
    MOCKER(aclrtIpcMemSetAttr).stubs().will(invoke(AclrtIpcMemSetAttrSuccess));
    MOCKER(aclrtIpcMemImportByKey).stubs().with(outBoundP(&fakePtr, sizeof(fakePtr))).will(returnValue(ACL_SUCCESS));

    RemoteIpcRmaBuffer buffer(dto, 1);

    EXPECT_TRUE(buffer.isOpened);
    EXPECT_EQ(g_capturedSetAttrCallCount, 1u);
    EXPECT_EQ(g_capturedSetAttrValue, static_cast<uint64_t>(ACL_RT_IPC_MEM_ATTR_ACCESS_LINK_UB_ONE_PORT_PATH));
};

TEST_F(RemoteRmaBufferTest, remoteipcrmabuffer_open_new_path_with_pathmode_two_maps_to_multipath)
{
    ExchangeIpcBufferDto dto;
    dto.addr = 0x1000;
    dto.offset = 0x100;
    dto.size = 0x200;
    dto.pid = 0;
    dto.memInfo = "test";
    (void)memcpy_s(dto.name, RTS_IPC_MEM_NAME_LEN, "fakeName", sizeof("fakeName"));

    g_capturedSetAttrValue = 0;
    g_capturedSetAttrCallCount = 0;
    void* fakePtr = reinterpret_cast<void*>(0x2000);
    MOCKER(HrtDeviceGetBareTgid)
        .stubs()
        .with(outBound(static_cast<s32>(100)))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(aclsysGetVersionNum).stubs().will(invoke(AclsysGetVersionNumHigh));
    MOCKER(aclrtIpcMemSetAttr).stubs().will(invoke(AclrtIpcMemSetAttrSuccess));
    MOCKER(aclrtIpcMemImportByKey).stubs().with(outBoundP(&fakePtr, sizeof(fakePtr))).will(returnValue(ACL_SUCCESS));

    RemoteIpcRmaBuffer buffer(dto, 2);

    EXPECT_TRUE(buffer.isOpened);
    EXPECT_EQ(g_capturedSetAttrCallCount, 1u);
    EXPECT_EQ(g_capturedSetAttrValue, static_cast<uint64_t>(ACL_RT_IPC_MEM_ATTR_ACCESS_LINK_UB_MULTI_PORT_PATH));
};

TEST_F(RemoteRmaBufferTest, remoteipcrmabuffer_open_with_pathmode_default_goes_legacy)
{
    ExchangeIpcBufferDto dto;
    dto.addr = 0x1000;
    dto.offset = 0x100;
    dto.size = 0x200;
    dto.pid = 0;
    dto.memInfo = "test";
    (void)memcpy_s(dto.name, RTS_IPC_MEM_NAME_LEN, "fakeName", sizeof("fakeName"));

    g_capturedSetAttrValue = 0;
    g_capturedSetAttrCallCount = 0;
    void* fakePtr = reinterpret_cast<void*>(0x2000);
    MOCKER(HrtDeviceGetBareTgid)
        .stubs()
        .with(outBound(static_cast<s32>(100)))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(aclsysGetVersionNum).stubs().will(invoke(AclsysGetVersionNumHigh));
    MOCKER(aclrtIpcMemSetAttr).stubs().will(invoke(AclrtIpcMemSetAttrSuccess));
    MOCKER(aclrtIpcMemImportByKey).stubs().with(outBoundP(&fakePtr, sizeof(fakePtr))).will(returnValue(ACL_SUCCESS));

    RemoteIpcRmaBuffer buffer(dto, 0xFF);

    EXPECT_TRUE(buffer.isOpened);
    EXPECT_EQ(g_capturedSetAttrCallCount, 0u);
    EXPECT_EQ(buffer.GetAddr(), reinterpret_cast<uintptr_t>(fakePtr) + dto.offset);
};

TEST_F(RemoteRmaBufferTest, remoteipcrmabuffer_open_new_path_with_pathmode_one_link_type_not_supported_no_fallback)
{
    ExchangeIpcBufferDto dto;
    dto.addr = 0x1000;
    dto.offset = 0x100;
    dto.size = 0x200;
    dto.pid = 0;
    dto.memInfo = "test";
    (void)memcpy_s(dto.name, RTS_IPC_MEM_NAME_LEN, "fakeName", sizeof("fakeName"));

    g_capturedSetAttrValue = 0;
    g_capturedSetAttrCallCount = 0;
    void* fakePtr = reinterpret_cast<void*>(0x2000);
    MOCKER(HrtDeviceGetBareTgid)
        .stubs()
        .with(outBound(static_cast<s32>(100)))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(aclsysGetVersionNum).stubs().will(invoke(AclsysGetVersionNumHigh));
    MOCKER(aclrtIpcMemSetAttr).stubs().will(invoke(AclrtIpcMemSetAttrLinkNotSupported));
    MOCKER(aclrtIpcMemImportByKey).stubs().with(outBoundP(&fakePtr, sizeof(fakePtr))).will(returnValue(ACL_SUCCESS));

    RemoteIpcRmaBuffer buffer(dto, 1);

    EXPECT_TRUE(buffer.isOpened);
    EXPECT_EQ(g_capturedSetAttrCallCount, 1u);
    EXPECT_EQ(g_capturedSetAttrValue, static_cast<uint64_t>(ACL_RT_IPC_MEM_ATTR_ACCESS_LINK_UB_ONE_PORT_PATH));
    EXPECT_EQ(buffer.GetAddr(), reinterpret_cast<uintptr_t>(fakePtr) + dto.offset);
};

// ===================== serverId 维度（IPC 同进程判定二元组） =====================

static HcclResult StubHrtDevMemAlignWithPage(void* ptr, u64 size, void*& ipcPtr, u64& ipcSize, u64& ipcOff)
{
    ipcPtr = ptr;
    ipcSize = size;
    ipcOff = 0;
    return HCCL_SUCCESS;
}

static HcclResult StubHrtIpcSetMemoryName(void* ptr, char_t* name, u64 ptrMaxLen, u32 nameMaxLen)
{
    (void)ptr;
    (void)ptrMaxLen;
    (void)strncpy_s(name, nameMaxLen, "stub_ipc_name", RTS_IPC_MEM_NAME_LEN - 1);
    return HCCL_SUCCESS;
}

// 同进程(pid 相等且 serverId 相等)：走 prefetch 捷径，不打开 IPC 内存
TEST_F(RemoteRmaBufferTest, remoteipcrmabuffer_samepid_sameserverid_takes_prefetch_path)
{
    ExchangeIpcBufferDto dto;
    dto.addr = 0x100000;
    dto.offset = 0x10;
    dto.size = 0x200;
    dto.pid = 100;
    dto.serverId = 5;
    dto.memInfo = "test";
    (void)memcpy_s(dto.name, RTS_IPC_MEM_NAME_LEN, "fakeName", sizeof("fakeName"));

    MOCKER(HrtDeviceGetBareTgid)
        .stubs()
        .with(outBound(static_cast<s32>(100)))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(HrtMemPrefetchToDevice).expects(once()).will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(HrtIpcOpenMemory).expects(never());

    RemoteIpcRmaBuffer buffer(dto, 0, 5);

    EXPECT_FALSE(buffer.isOpened);
    EXPECT_EQ(buffer.GetAddr(), dto.addr + dto.offset);
}

// 跨 server PID 碰撞(pid 相等但 serverId 不等)：走通用 open 路径——修复目标场景
TEST_F(RemoteRmaBufferTest, remoteipcrmabuffer_samepid_diffserverid_takes_open_path)
{
    ExchangeIpcBufferDto dto;
    dto.addr = 0x1000;
    dto.offset = 0x100;
    dto.size = 0x200;
    dto.pid = 100;
    dto.serverId = 9;
    dto.memInfo = "test";
    (void)memcpy_s(dto.name, RTS_IPC_MEM_NAME_LEN, "fakeName", sizeof("fakeName"));

    g_capturedSetAttrValue = 0;
    g_capturedSetAttrCallCount = 0;
    void* fakePtr = reinterpret_cast<void*>(0x2000);
    MOCKER(HrtDeviceGetBareTgid)
        .stubs()
        .with(outBound(static_cast<s32>(100)))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(HrtMemPrefetchToDevice).expects(never());
    MOCKER(aclsysGetVersionNum).stubs().will(invoke(AclsysGetVersionNumHigh));
    MOCKER(aclrtIpcMemSetAttr).stubs().will(invoke(AclrtIpcMemSetAttrSuccess));
    MOCKER(aclrtIpcMemImportByKey).stubs().with(outBoundP(&fakePtr, sizeof(fakePtr))).will(returnValue(ACL_SUCCESS));

    RemoteIpcRmaBuffer buffer(dto, 0, 5);

    EXPECT_TRUE(buffer.isOpened);
    EXPECT_EQ(buffer.GetAddr(), reinterpret_cast<uintptr_t>(fakePtr) + dto.offset);
}

// 同 server 跨进程(pid 不等)：走通用 open 路径，与既有行为一致
TEST_F(RemoteRmaBufferTest, remoteipcrmabuffer_diffpid_sameserverid_takes_open_path)
{
    ExchangeIpcBufferDto dto;
    dto.addr = 0x1000;
    dto.offset = 0x100;
    dto.size = 0x200;
    dto.pid = 999;
    dto.serverId = 5;
    dto.memInfo = "test";
    (void)memcpy_s(dto.name, RTS_IPC_MEM_NAME_LEN, "fakeName", sizeof("fakeName"));

    void* fakePtr = reinterpret_cast<void*>(0x2000);
    MOCKER(HrtDeviceGetBareTgid)
        .stubs()
        .with(outBound(static_cast<s32>(100)))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(aclsysGetVersionNum).stubs().will(invoke(AclsysGetVersionNumHigh));
    MOCKER(aclrtIpcMemSetAttr).stubs().will(invoke(AclrtIpcMemSetAttrSuccess));
    MOCKER(aclrtIpcMemImportByKey).stubs().with(outBoundP(&fakePtr, sizeof(fakePtr))).will(returnValue(ACL_SUCCESS));

    RemoteIpcRmaBuffer buffer(dto, 0, 5);

    EXPECT_TRUE(buffer.isOpened);
}

// 未启用方(不传 localServerId，DTO 默认 0)：判定退化为 pid 单维，同进程捷径行为不变
TEST_F(RemoteRmaBufferTest, remoteipcrmabuffer_without_localserverid_keeps_pidonly_semantics)
{
    ExchangeIpcBufferDto dto;
    dto.addr = 0x100000;
    dto.offset = 0x10;
    dto.size = 0x200;
    dto.pid = 100;
    dto.memInfo = "test";
    (void)memcpy_s(dto.name, RTS_IPC_MEM_NAME_LEN, "fakeName", sizeof("fakeName"));

    MOCKER(HrtDeviceGetBareTgid)
        .stubs()
        .with(outBound(static_cast<s32>(100)))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(HrtMemPrefetchToDevice).expects(once()).will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(HrtIpcOpenMemory).expects(never());

    RemoteIpcRmaBuffer buffer(dto);

    EXPECT_FALSE(buffer.isOpened);
    EXPECT_EQ(buffer.GetAddr(), dto.addr + dto.offset);
}

// DTO serverId 序列化往返无损（字段序紧随 pid）
TEST_F(RemoteRmaBufferTest, exchangeipcbufferdto_serialize_deserialize_serverid_roundtrip)
{
    ExchangeIpcBufferDto dto;
    dto.addr = 0x100000;
    dto.size = 0x200;
    dto.offset = 0x10;
    dto.pid = 4321;
    dto.serverId = 7;
    dto.memInfo = "roundtrip";
    dto.memType = HcclMemType::HCCL_MEM_TYPE_DEVICE;
    (void)memcpy_s(dto.name, RTS_IPC_MEM_NAME_LEN, "roundName", sizeof("roundName"));

    BinaryStream stream;
    dto.Serialize(stream);
    ExchangeIpcBufferDto parsed;
    parsed.Deserialize(stream);

    EXPECT_EQ(parsed.serverId, 7U);
    EXPECT_EQ(parsed.pid, 4321U);
    EXPECT_EQ(parsed.addr, dto.addr);
    EXPECT_EQ(parsed.size, dto.size);
    EXPECT_EQ(parsed.offset, dto.offset);
    EXPECT_EQ(parsed.memType, dto.memType);
    EXPECT_EQ(parsed.memInfo, dto.memInfo);
    EXPECT_EQ(memcmp(parsed.name, dto.name, RTS_IPC_MEM_NAME_LEN), 0);
}

// LocalIpcRmaBuffer 构造注入 serverId：GetExchangeDto 携带注入值与 pid
TEST_F(RemoteRmaBufferTest, localipcrmabuffer_serverid_injected_dto_carries_serverid)
{
    std::shared_ptr<DevBuffer> devBuf = DevBuffer::Create(0x100000, 0x1000);
    ASSERT_NE(devBuf, nullptr);

    MOCKER(HrtDevMemAlignWithPage)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(StubHrtDevMemAlignWithPage));
    MOCKER(HrtIpcSetMemoryName)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(StubHrtIpcSetMemoryName));
    MOCKER(HrtIpcDestroyMemoryName).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(HrtDeviceGetBareTgid)
        .stubs()
        .with(outBound(static_cast<s32>(4321)))
        .will(returnValue(HcclResult::HCCL_SUCCESS));

    LocalIpcRmaBuffer localBuffer(devBuf, 5);
    EXPECT_EQ(localBuffer.GetServerId(), 5U);

    std::unique_ptr<Serializable> serializable = localBuffer.GetExchangeDto();
    auto* ipcDto = dynamic_cast<ExchangeIpcBufferDto*>(serializable.get());
    ASSERT_NE(ipcDto, nullptr);
    EXPECT_EQ(ipcDto->serverId, 5U);
    EXPECT_EQ(ipcDto->pid, 4321U);
}
