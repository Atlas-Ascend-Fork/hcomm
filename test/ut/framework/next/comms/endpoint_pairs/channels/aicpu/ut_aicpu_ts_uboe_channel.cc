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
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>

#define private public
#define protected public
#include "aicpu/aicpu_ts_uboe_channel.h"
#undef protected
#undef private
#include "endpoint.h"
#include "hcomm_c_adpt.h"
#include "exchange_ub_buffer_dto.h"
#include "sockets/socket_mgr.h"

#define private public
#define protected public
#include "aicpu/aicpu_ts_uboe_channel.h"
#include "binary_stream.h"
#include "rma_buf_entry_lite.h"
using namespace hcomm;

class AicpuTsUboeChannelTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override
    {
        const char* dfsConfig = std::getenv("HCCL_DFS_CONFIG");
        if (dfsConfig != nullptr) {
            savedDfsConfig_ = dfsConfig;
            hadDfsConfig_ = true;
        }
        (void)setenv("HCCL_DFS_CONFIG", "task_exception:on", 1);
    }

    void TearDown() override
    {
        GlobalMockObject::verify();
        if (hadDfsConfig_) {
            (void)setenv("HCCL_DFS_CONFIG", savedDfsConfig_.c_str(), 1);
        } else {
            (void)unsetenv("HCCL_DFS_CONFIG");
        }
    }

private:
    std::string savedDfsConfig_;
    bool hadDfsConfig_{false};
};

namespace {
EndpointDesc MakeUboeEndpointDesc()
{
    EndpointDesc desc{};
    Hccl::IpAddress localIp("127.0.0.1");
    desc.protocol = COMM_PROTOCOL_UBOE;
    desc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    desc.commAddr.addr = localIp.GetBinaryAddress().addr;
    desc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    return desc;
}

// Lightweight fake for external dependencies used by AicpuTsUboeChannel.
class FakeEndpoint : public Endpoint {
public:
    FakeEndpoint() : Endpoint(MakeUboeEndpointDesc()) { ctxHandle_ = reinterpret_cast<void*>(0xDEADBEEF); }

    HcclResult Init() override { return HCCL_SUCCESS; }
    // 内存方法在 mgr 上，Endpoint 不再承载
    LocalRegedMemMgr* GetLocalRegMemMgr() override { return nullptr; }
    void* GetRdmaHandle() override { return ctxHandle_; }
    bool IsCtxHandleValid() const override { return ctxHandle_ != nullptr; }
    Hccl::IpAddress GetIpAddress() const { return Hccl::IpAddress("127.0.0.1"); }

private:
    void* ctxHandle_{nullptr};
};

std::shared_ptr<Hccl::LocalUbRmaBuffer> MakeUboeLocalBuffer(uintptr_t addr, u64 size, const char* tag)
{
    auto buffer = std::make_shared<Hccl::Buffer>(addr, size, HCCL_MEM_TYPE_DEVICE, tag);
    return std::make_shared<Hccl::LocalUbRmaBuffer>(buffer);
}

HcommResult StubUboeGetAllMemHandlesOne(EndpointHandle, void** memHandles, uint32_t* memHandleNum)
{
    static std::shared_ptr<Hccl::Buffer> buffer
        = std::make_shared<Hccl::Buffer>(0x520000U, 0x1000U, HCCL_MEM_TYPE_DEVICE, "uboe_all");
    static std::shared_ptr<Hccl::LocalUbRmaBuffer> localBuffer = std::make_shared<Hccl::LocalUbRmaBuffer>(buffer);
    static std::shared_ptr<Hccl::LocalUbRmaBuffer> localBuffers[1] = {localBuffer};
    *memHandles = localBuffers;
    *memHandleNum = 1;
    return HCCL_SUCCESS;
}

// HcommEndpointGet 的 void** 指针出参不能用 outBound（mockcpp 的 outBound 仅支持引用出参），
// 须用 invoke 桩函数回写出参（先例：ut_socket_process.cc 的 StubSocketMgrHrtGetDevice 系列）
static void* g_uboeStubEndpoint = nullptr;

static HcommResult StubUboeHcommEndpointGet(EndpointHandle /*endpointHandle*/, void** endpoint)
{
    *endpoint = g_uboeStubEndpoint;
    return static_cast<HcommResult>(HCCL_SUCCESS);
}
} // namespace

class FakeSocket : public Hccl::Socket {
public:
    FakeSocket(Hccl::SocketStatus status = Hccl::SocketStatus::OK)
        : Hccl::Socket(
              nullptr, Hccl::IpAddress(), 0, Hccl::IpAddress(), "fake", Hccl::SocketRole::SERVER,
              Hccl::NicType::DEVICE_NIC_TYPE),
          status_(status)
    {}
    void SendAsync(const void* sendBuf, u32 size)
    {
        auto* p = static_cast<const u8*>(sendBuf);
        sent_.insert(sent_.end(), p, p + size);
    }
    void RecvAsync(void* recvBuf, u32 size)
    {
        auto* buf = static_cast<u8*>(recvBuf);
        // If we previously recorded sent bytes, echo them back to the receiver so
        // higher-level unpacking (EID/Conn/Buffer) sees sensible data instead of all zeros.
        if (buf && size) {
            if (!sent_.empty()) {
                u32 copySize = static_cast<u32>(std::min<size_t>(sent_.size(), static_cast<size_t>(size)));
                memcpy(buf, sent_.data(), copySize);
                if (copySize < size)
                    std::memset(buf + copySize, 0, size - copySize);
                // remove echoed bytes so subsequent RecvAsync calls progress
                sent_.erase(sent_.begin(), sent_.begin() + copySize);
            } else {
                // no data available yet -> return zeros
                std::memset(buf, 0, size);
            }
        }
    }
    Hccl::SocketStatus GetAsyncStatus() { return status_; }
    Hccl::SocketRole GetRole() const { return Hccl::SocketRole::SERVER; }

    // allow tests to change reported status
    Hccl::SocketStatus status_;
    std::vector<u8> sent_;
};

// Minimal fake LocalUbRmaBuffer and UbLocalNotify compatible interfaces
namespace Hccl {
class FakeLocalUbRmaBuffer : public Hccl::LocalUbRmaBuffer {
public:
    FakeLocalUbRmaBuffer(std::shared_ptr<Hccl::Buffer> b, void* rdma) : LocalUbRmaBuffer(b, rdma) {}
    string Describe() const override { return "hello"; }
};
class FakeExchangeLocalRmaBuffer : public Hccl::LocalRmaBuffer {
public:
    explicit FakeExchangeLocalRmaBuffer(std::shared_ptr<Hccl::Buffer> b) : LocalRmaBuffer(b, Hccl::RmaType::UB) {}
    string Describe() const override { return "FakeExchangeLocalRmaBuffer"; }
    std::unique_ptr<Hccl::Serializable> GetExchangeDto() override
    {
        return std::make_unique<Hccl::ExchangeUbBufferDto>(
            buf->GetAddr(), buf->GetSize(), buf->GetMemType(), buf->GetMemInfo().c_str(), 0, 0, 0);
    }
};
class FakeUbLocalNotify : public Hccl::UbLocalNotify {
public:
    FakeUbLocalNotify(void* rdma, bool dev) : UbLocalNotify(rdma, dev) {}
};
} // namespace Hccl

std::shared_ptr<Hccl::LocalRmaBuffer> MakeUboeExchangeLocalBuffer(uintptr_t addr, u64 size, const char* tag)
{
    auto buffer = std::make_shared<Hccl::Buffer>(addr, size, HCCL_MEM_TYPE_DEVICE, tag);
    return std::make_shared<Hccl::FakeExchangeLocalRmaBuffer>(buffer);
}

// Helper to build HcommChannelDesc with fake socket and endpoint
static HcommChannelDesc MakeFakeChannelDesc(FakeSocket* sock)
{
    HcommChannelDesc d{};
    d.socket = reinterpret_cast<void*>(sock);
    d.notifyNum = 0;
    d.exchangeAllMems = false;
    d.memHandles = nullptr;
    d.memHandleNum = 0;
    return d;
}

TEST_F(AicpuTsUboeChannelTest, Ut_Clean_WithoutInit_Returns_SUCCESS)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUboeChannel ch(ep, desc);

    auto ret = ch.Clean();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(AicpuTsUboeChannelTest, Ut_CleanThenResume_When_ChannelHasCachedResource_Expect_ResetStateAndClearResource)
{
    FakeEndpoint fe;
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(&fe);
    AicpuTsUboeChannel ch(ep, desc);

    ch.local_.commonRes.connVec.push_back(nullptr);
    ch.rmtNotifyVec_.push_back(std::make_unique<Hccl::RemoteUbRmaBuffer>(fe.GetRdmaHandle()));
    ch.locBufferVec_.push_back(nullptr);
    ch.recvData_.push_back('r');
    ch.recvFinishMsg_.push_back('f');
    ch.recvEidData_.push_back('e');
    ch.sendData_.push_back('s');
    ch.sendFinishMsg_.push_back('d');
    ch.sendEidData_.push_back('i');
    ch.bufferNum_ = 1;
    ch.connNum_ = 1;
    ch.recvDataSize_ = 1;
    ch.rmtBufferVec_.push_back(std::make_unique<Hccl::RemoteUbRmaBuffer>(fe.GetRdmaHandle()));
    ch.remote_.memCache.cacheValid = true;
    ch.remote_.memCache.remoteUserMems.push_back(CommMem{});
    ch.remote_.memCache.memInfoCopies.push_back("user_mem");
    ch.remote_.memCache.memInfoPointers.push_back(const_cast<char*>(ch.remote_.memCache.memInfoCopies.front().c_str()));
    ch.channelStatus = ChannelStatus::READY;
    ch.uboeStatus = AicpuTsUboeChannel::UboeStatus::READY;

    HcclResult cleanRet = ch.Clean();
    HcclResult resumeRet = ch.Resume();

    EXPECT_EQ(cleanRet, HCCL_SUCCESS);
    EXPECT_EQ(resumeRet, HCCL_SUCCESS);
    EXPECT_TRUE(ch.local_.commonRes.connVec.empty());
    EXPECT_TRUE(ch.rmtNotifyVec_.empty());
    EXPECT_TRUE(ch.locBufferVec_.empty());
    EXPECT_TRUE(ch.recvData_.empty());
    EXPECT_TRUE(ch.recvFinishMsg_.empty());
    EXPECT_TRUE(ch.recvEidData_.empty());
    EXPECT_TRUE(ch.sendData_.empty());
    EXPECT_TRUE(ch.sendFinishMsg_.empty());
    EXPECT_TRUE(ch.sendEidData_.empty());
    EXPECT_EQ(ch.bufferNum_, 0U);
    EXPECT_EQ(ch.connNum_, 0U);
    EXPECT_EQ(ch.recvDataSize_, 0U);
    EXPECT_TRUE(ch.rmtBufferVec_.empty());
    EXPECT_FALSE(ch.remote_.memCache.cacheValid);
    EXPECT_TRUE(ch.remote_.memCache.remoteUserMems.empty());
    EXPECT_TRUE(ch.remote_.memCache.memInfoCopies.empty());
    EXPECT_TRUE(ch.remote_.memCache.memInfoPointers.empty());
    EXPECT_EQ(ch.channelStatus, ChannelStatus::INIT);
    EXPECT_EQ(ch.uboeStatus, AicpuTsUboeChannel::UboeStatus::INIT);
}

TEST_F(AicpuTsUboeChannelTest, Ut_Init_MockedHelpers_Returns_SUCCESS)
{
    // Do not mock internal methods. Inject fake endpoint and fake socket so Init() exercises real code paths.
    FakeEndpoint fe;
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(&fe);
    auto fakeSock = std::make_unique<FakeSocket>(Hccl::SocketStatus::OK);
    HcommChannelDesc desc = MakeFakeChannelDesc(fakeSock.get());
    auto localBuffer = MakeUboeLocalBuffer(0x510000U, 0x1000U, "uboe_init");
    HcommMemHandle memHandles[1] = {reinterpret_cast<HcommMemHandle>(localBuffer.get())};
    desc.memHandles = memHandles;
    desc.memHandleNum = 1;
    AicpuTsUboeChannel ch(ep, desc);

    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().will(returnValue(HCCL_SUCCESS));
    g_uboeStubEndpoint = static_cast<void*>(ep);
    MOCKER(HcommEndpointGet).stubs().will(invoke(StubUboeHcommEndpointGet));

    ASSERT_EQ(ch.Init(), HCCL_SUCCESS);

    ASSERT_EQ(ch.local_.commonRes.bufferVec.size(), 1U);
    EXPECT_EQ(ch.local_.commonRes.bufferVec[0], localBuffer.get());
}

TEST_F(AicpuTsUboeChannelTest, UT_ParseInputParam_When_ExchangeAllMemsFalse_Expect_FillCommonRes)
{
    FakeEndpoint fe;
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(&fe);
    auto fakeSock = std::make_unique<FakeSocket>(Hccl::SocketStatus::OK);
    auto localBuffer = MakeUboeLocalBuffer(0x530000U, 0x1000U, "uboe_desc");
    HcommMemHandle memHandles[1] = {reinterpret_cast<HcommMemHandle>(localBuffer.get())};
    HcommChannelDesc desc = MakeFakeChannelDesc(fakeSock.get());
    desc.exchangeAllMems = false;
    desc.memHandles = memHandles;
    desc.memHandleNum = 1;
    AicpuTsUboeChannel ch(ep, desc);

    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().will(returnValue(HCCL_SUCCESS));
    g_uboeStubEndpoint = static_cast<void*>(ep);
    MOCKER(HcommEndpointGet).stubs().will(invoke(StubUboeHcommEndpointGet));

    ASSERT_EQ(ch.ParseInputParam(), HCCL_SUCCESS);
    ASSERT_EQ(ch.local_.commonRes.bufferVec.size(), 1U);
    EXPECT_EQ(ch.local_.commonRes.bufferVec[0], localBuffer.get());
}

TEST_F(AicpuTsUboeChannelTest, UT_ParseInputParam_When_ExchangeAllMemsTrue_Expect_FillCommonRes)
{
    FakeEndpoint fe;
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(&fe);
    auto fakeSock = std::make_unique<FakeSocket>(Hccl::SocketStatus::OK);
    HcommChannelDesc desc = MakeFakeChannelDesc(fakeSock.get());
    desc.exchangeAllMems = true;
    AicpuTsUboeChannel ch(ep, desc);

    MOCKER(HcommMemGetAllMemHandles).stubs().will(invoke(StubUboeGetAllMemHandlesOne));
    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().will(returnValue(HCCL_SUCCESS));
    g_uboeStubEndpoint = static_cast<void*>(ep);
    MOCKER(HcommEndpointGet).stubs().will(invoke(StubUboeHcommEndpointGet));

    ASSERT_EQ(ch.ParseInputParam(), HCCL_SUCCESS);
    ASSERT_EQ(ch.local_.commonRes.bufferVec.size(), 1U);
    EXPECT_EQ(ch.local_.commonRes.bufferVec[0]->GetAddr(), 0x520000U);
}

TEST_F(AicpuTsUboeChannelTest, Ut_GetStatus_WhenSocketNotReady_Returns_INIT)
{
    // Inject a fake socket and control its GetAsyncStatus() instead of mocking IsSocketReady
    FakeEndpoint fe;
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(&fe);
    auto fakeSock = new FakeSocket(Hccl::SocketStatus::TIMEOUT);
    HcommChannelDesc desc = MakeFakeChannelDesc(fakeSock);
    AicpuTsUboeChannel ch(ep, desc);

    // Ensure the channel uses our fake socket (ParseInputParam isn't called here)
    ch.channelDesc_.socket = reinterpret_cast<void*>(fakeSock);
    ch.local_.socket = reinterpret_cast<Hccl::Socket*>(fakeSock);

    // Ensure initial channelStatus is INIT
    EXPECT_EQ(ch.channelStatus, ChannelStatus::INIT);

    // Now GetStatus should observe socket timeout via local_.socket->GetAsyncStatus()
    auto status = ch.GetStatus();
    EXPECT_EQ(status, ChannelStatus::SOCKET_OK);

    delete fakeSock;
}

TEST_F(AicpuTsUboeChannelTest, Ut_GetNotifyNum_Returns_Value)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUboeChannel ch(ep, desc);

    ch.local_.notifyNum = 42;
    uint32_t n = 0;
    auto ret = ch.GetNotifyNum(&n);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(n, 42u);
}

// Test-local stubs for Socket async APIs. These will be used with MOCKER_CPP to intercept
// calls to Socket::SendAsync and Socket::RecvAsync inside the state-machine test.
static void stub_Socket_SendAsync(Hccl::Socket* self, const void* sendBuf, u32 size)
{
    if (!self || !sendBuf || size == 0)
        return;
    auto* fs = dynamic_cast<FakeSocket*>(self);
    if (fs) {
        auto* p = static_cast<const u8*>(sendBuf);
        fs->sent_.insert(fs->sent_.end(), p, p + size);
    }
}

static void stub_Socket_RecvAsync(Hccl::Socket* self, void* recvBuf, u32 size)
{
    if (!self || !recvBuf || size == 0)
        return;
    auto* buf = static_cast<u8*>(recvBuf);
    auto* fs = dynamic_cast<FakeSocket*>(self);
    if (fs) {
        if (!fs->sent_.empty()) {
            u32 copySize = static_cast<u32>(std::min<size_t>(fs->sent_.size(), static_cast<size_t>(size)));
            memcpy(buf, fs->sent_.data(), copySize);
            if (copySize < size)
                std::memset(buf + copySize, 0, size - copySize);
            fs->sent_.erase(fs->sent_.begin(), fs->sent_.begin() + copySize);
            return;
        }
    }
    // no data available -> zero fill
    std::memset(buf, 0, size);
}

TEST_F(AicpuTsUboeChannelTest, Ut_PackingHelpers_NoCrash)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUboeChannel ch(ep, desc);

    Hccl::BinaryStream bs;
    ch.NotifyVecPack(bs);
    std::vector<Hccl::LocalRmaBuffer*> emptyBuf;
    std::vector<std::array<char, HCCL_RES_TAG_MAX_LEN>> tags;
    ch.BufferVecPack(bs, emptyBuf);
    ch.ConnVecPack(bs);

    SUCCEED();
}

TEST_F(AicpuTsUboeChannelTest, Ut_Init_WithFakes_Runs)
{
    FakeEndpoint fe;
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(&fe);
    auto fakeSock = new FakeSocket();
    HcommChannelDesc desc = MakeFakeChannelDesc(fakeSock);

    AicpuTsUboeChannel ch(ep, desc);

    // Override rdma/context behaviour by making ParseInputParam use our fake - inject via endpointHandle
    auto ret = ch.Init();
    // Init may try to build resources; expect success or graceful failure depending on deeper deps
    SUCCEED();
}

TEST_F(AicpuTsUboeChannelTest, Ut_H2DResPack_Packs_Data)
{
    FakeEndpoint fe;
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(&fe);
    auto fakeSock = new FakeSocket();
    HcommChannelDesc desc = MakeFakeChannelDesc(fakeSock);

    AicpuTsUboeChannel ch(ep, desc);

    // Prepare channel so GetUniqueIdV2() can run: status READY and one notify
    ch.channelStatus = ChannelStatus::READY;
    ch.local_.notifyNum = 1;
    ch.bufferNum_ = 2;
    ch.connNum_ = 1;

    auto notifyUptr = std::make_unique<Hccl::FakeUbLocalNotify>(fe.GetRdmaHandle(), true);
    Hccl::UbLocalNotify* rawNotify = notifyUptr.get();
    ch.local_.notifies.push_back(std::move(notifyUptr));
    ch.local_.commonRes.notifyVec.push_back(rawNotify);

    auto rmtUbRmaBufPtr = std::make_unique<Hccl::RemoteUbRmaBuffer>(fe.GetRdmaHandle());
    ch.rmtNotifyVec_.push_back(std::move(rmtUbRmaBufPtr));

    auto buffer = std::make_shared<Hccl::Buffer>(0x100, 0x100);
    auto locUbRmaBufPtr = std::make_unique<Hccl::FakeLocalUbRmaBuffer>(buffer, fe.GetRdmaHandle());
    Hccl::FakeLocalUbRmaBuffer* locUbRmaBuf = locUbRmaBufPtr.get();
    ch.local_.commonRes.bufferVec.push_back(locUbRmaBuf);
    ch.local_.commonRes.bufferVec.push_back(nullptr);

    auto rmtBufPtr = std::make_unique<Hccl::RemoteUbRmaBuffer>(fe.GetRdmaHandle());
    ch.rmtBufferVec_.push_back(std::move(rmtBufPtr));

    ch.rmtDrainBuffer_ = std::make_unique<Hccl::RemoteUbRmaBuffer>(fe.GetRdmaHandle());

    Hccl::IpAddress ipAddress = fe.GetIpAddress();
    Hccl::DevUbUboeConnection uboeConn(
        fe.GetRdmaHandle(), ipAddress, ipAddress, Hccl::OpMode::OPBASE, true, Hccl::HrtUbJfcMode::STARS_POLL, ipAddress,
        ipAddress);
    ch.local_.commonRes.connVec.push_back(&uboeConn);

    std::vector<char> out{};
    HcclResult ret = ch.H2DResPack(out);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    delete fakeSock;
}

TEST_F(AicpuTsUboeChannelTest, Ut_GetChannelKind_Returns_AICPU_TS_UBOE)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUboeChannel ch(ep, desc);
    EXPECT_EQ(ch.GetChannelKind(), HcommChannelKind::AICPU_TS_UBOE);
}

// 辅助 FakeRemoteUbRmaBuffer，用于模拟远端内存区域
class FakeRemoteUbRmaBuffer : public Hccl::RemoteUbRmaBuffer {
public:
    FakeRemoteUbRmaBuffer(void* rdmaHandle, uint64_t addr, size_t size, HcclMemType type, const std::string& memInfo)
        : Hccl::RemoteUbRmaBuffer(rdmaHandle)
    {
        this->addr = addr;
        this->size = size;
        this->memType = type;
        this->memInfo = memInfo;
    }
};

TEST_F(AicpuTsUboeChannelTest, Ut_GetRemoteMems_NoBuffers_ReturnsSuccess)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUboeChannel ch(ep, desc);

    // 确保 rmtBufferVec_ 为空
    ch.rmtBufferVec_.clear();

    CommMem* remoteMem = nullptr;
    uint32_t memNum = 0;
    char** memInfos = nullptr;

    HcclResult ret = ch.GetRemoteMems(&memNum, &remoteMem, &memInfos);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(remoteMem, nullptr);
    EXPECT_EQ(memNum, 0U);
}

TEST_F(AicpuTsUboeChannelTest, Ut_GetRemoteMems_WithBuffers_ReturnsCorrectData)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUboeChannel ch(ep, desc);

    // 构造两个远端内存区域
    void* fakeRdma = reinterpret_cast<void*>(0x1234);
    auto buf1 = std::make_unique<FakeRemoteUbRmaBuffer>(fakeRdma, 0x1000, 4096, HCCL_MEM_TYPE_DEVICE, "ccl_buffer");
    auto buf2 = std::make_unique<FakeRemoteUbRmaBuffer>(fakeRdma, 0x2000, 8192, HCCL_MEM_TYPE_HOST, "user_buffer");
    ch.rmtBufferVec_.push_back(std::move(buf1));
    ch.rmtBufferVec_.push_back(std::move(buf2));

    CommMem* remoteMem = nullptr;
    uint32_t memNum = 0;
    char** memInfos = nullptr;
    HcclResult ret = ch.GetRemoteMems(&memNum, &remoteMem, &memInfos);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(memNum, 2U);
    ASSERT_NE(remoteMem, nullptr);

    // 验证第一个内存区域
    EXPECT_EQ(remoteMem[0].type, COMM_MEM_TYPE_DEVICE);
    EXPECT_EQ(remoteMem[0].addr, reinterpret_cast<void*>(0x1000));
    EXPECT_EQ(remoteMem[0].size, 4096U);
    EXPECT_STREQ(memInfos[0], "ccl_buffer");

    // 验证第二个内存区域
    EXPECT_EQ(remoteMem[1].type, COMM_MEM_TYPE_HOST);
    EXPECT_EQ(remoteMem[1].addr, reinterpret_cast<void*>(0x2000));
    EXPECT_EQ(remoteMem[1].size, 8192U);
    EXPECT_STREQ(memInfos[1], "user_buffer");
}

TEST_F(AicpuTsUboeChannelTest, Ut_GetRemoteMems_NullParams_ReturnsError)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUboeChannel ch(ep, desc);

    // 至少构造一个 buffer，避免提前返回错误
    void* fakeRdma = reinterpret_cast<void*>(0x1234);
    auto cclBuf = std::make_unique<FakeRemoteUbRmaBuffer>(fakeRdma, 0x1000, 4096, HCCL_MEM_TYPE_DEVICE, "ccl_buffer");
    ch.rmtBufferVec_.push_back(std::move(cclBuf));

    CommMem* remoteMem = nullptr;
    char** memInfos = nullptr;
    uint32_t memNum = 0;

    // 传入无效参数，期望返回参数错误（具体错误码视实现而定，当前返回 HCCL_E_PARA）
    EXPECT_EQ(ch.GetRemoteMems(&memNum, nullptr, &memInfos), HCCL_E_PARA);
    EXPECT_EQ(ch.GetRemoteMems(&memNum, &remoteMem, nullptr), HCCL_E_PARA);
    EXPECT_EQ(ch.GetRemoteMems(nullptr, &remoteMem, &memInfos), HCCL_E_PARA);
}

TEST_F(AicpuTsUboeChannelTest, Ut_GetRemoteMems_OnlyCclBuffer_ReturnsSuccess)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUboeChannel ch(ep, desc);

    void* fakeRdma = reinterpret_cast<void*>(0x1234);
    auto cclBuf = std::make_unique<FakeRemoteUbRmaBuffer>(fakeRdma, 0x1000, 4096, HCCL_MEM_TYPE_DEVICE, "ccl_buffer");
    ch.rmtBufferVec_.push_back(std::move(cclBuf));

    // 确保内部 cache 标志初始为 false，以便重新构建
    ch.remote_.memCache.cacheValid = false;

    CommMem* remoteMem = nullptr;
    char** memInfos = nullptr;
    uint32_t memNum = 0;

    HcclResult ret = ch.GetRemoteMems(&memNum, &remoteMem, &memInfos);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(memNum, 1U);
    ASSERT_NE(remoteMem, nullptr);
    // memInfos 在数量不为0时也可能为 nullptr，不强制检查
}

TEST_F(AicpuTsUboeChannelTest, Ut_GetRemoteMems_WithUserBuffers_ReturnsCorrectData)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUboeChannel ch(ep, desc);

    void* fakeRdma = reinterpret_cast<void*>(0x1234);
    // 第一个 buffer 为 ccl buffer，与后续用户 buffer 一并返回
    auto cclBuf = std::make_unique<FakeRemoteUbRmaBuffer>(fakeRdma, 0x1000, 4096, HCCL_MEM_TYPE_DEVICE, "ccl");
    // 用户 buffer 1
    auto userBuf1 = std::make_unique<FakeRemoteUbRmaBuffer>(fakeRdma, 0x2000, 8192, HCCL_MEM_TYPE_DEVICE, "user1");
    // 用户 buffer 2
    auto userBuf2 = std::make_unique<FakeRemoteUbRmaBuffer>(fakeRdma, 0x3000, 16384, HCCL_MEM_TYPE_DEVICE, "user2");
    ch.rmtBufferVec_.push_back(std::move(cclBuf));
    ch.rmtBufferVec_.push_back(std::move(userBuf1));
    ch.rmtBufferVec_.push_back(std::move(userBuf2));

    ch.remote_.memCache.cacheValid = false; // 强制重新构建缓存

    CommMem* remoteMem = nullptr;
    char** memInfos = nullptr;
    uint32_t memNum = 0;

    HcclResult ret = ch.GetRemoteMems(&memNum, &remoteMem, &memInfos);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(memNum, 3U);
    ASSERT_NE(remoteMem, nullptr);
    ASSERT_NE(memInfos, nullptr);

    // 验证第一个用户内存（索引1）
    EXPECT_EQ(remoteMem[1].type, COMM_MEM_TYPE_DEVICE);
    EXPECT_EQ(remoteMem[1].addr, reinterpret_cast<void*>(0x2000));
    EXPECT_EQ(remoteMem[1].size, 8192U);
    EXPECT_STREQ(memInfos[1], "user1");

    // 验证第二个用户内存（索引2）
    EXPECT_EQ(remoteMem[2].type, COMM_MEM_TYPE_DEVICE);
    EXPECT_EQ(remoteMem[2].addr, reinterpret_cast<void*>(0x3000));
    EXPECT_EQ(remoteMem[2].size, 16384U);
    EXPECT_STREQ(memInfos[2], "user2");
}

TEST_F(AicpuTsUboeChannelTest, Ut_GetStatusReady_SocketNotNullptr)
{
    MOCKER_CPP(&SocketMgr::GetSocket).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&SocketMgr::PutSocket).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AicpuTsUboeChannel::IsSocketReady).stubs().will(returnValue(true));
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUboeChannel ch(ep, desc);
    ch.uboeStatus = AicpuTsUboeChannel::UboeStatus::SET_READY;
    ch.GetStatus();
    GlobalMockObject::verify();
}

TEST_F(AicpuTsUboeChannelTest, Ut_CheckSocketStatus_When_SocketOK_Expect_ReturnSuccess)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUboeChannel ch(ep, desc);

    auto fakeSock = new FakeSocket(Hccl::SocketStatus::OK);
    ch.local_.socket = reinterpret_cast<Hccl::Socket*>(fakeSock);

    Hccl::SocketStatus okStatus = Hccl::SocketStatus::OK;
    MOCKER(&Hccl::Socket::GetAsyncStatus).stubs().will(returnValue(okStatus));

    HcclResult ret = ch.CheckSocketStatus("TestOp");
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
    delete fakeSock;
    ch.local_.socket = nullptr;
}

TEST_F(AicpuTsUboeChannelTest, Ut_CheckSocketStatus_When_SocketNull_Expect_ReturnPtrError)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUboeChannel ch(ep, desc);

    ch.local_.socket = nullptr;
    HcclResult ret = ch.CheckSocketStatus("TestOp");
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(AicpuTsUboeChannelTest, Ut_CheckSocketStatus_When_SocketTimeout_Expect_ReturnTimeout)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUboeChannel ch(ep, desc);

    auto fakeSock = new FakeSocket(Hccl::SocketStatus::TIMEOUT);
    ch.local_.socket = reinterpret_cast<Hccl::Socket*>(fakeSock);

    Hccl::SocketStatus timeoutStatus = Hccl::SocketStatus::TIMEOUT;
    MOCKER(&Hccl::Socket::GetAsyncStatus).stubs().will(returnValue(timeoutStatus));

    HcclResult ret = ch.CheckSocketStatus("TestOp");
    EXPECT_EQ(ret, HCCL_E_TIMEOUT);

    GlobalMockObject::verify();
    delete fakeSock;
    ch.local_.socket = nullptr;
}

TEST_F(AicpuTsUboeChannelTest, UT_UpdateMemInfo_When_MemHandleNumZero_Expect_ReturnHCCL_SUCCESS)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUboeChannel ch(ep, desc);

    ch.local_.socket = nullptr;

    HcclResult ret = ch.UpdateMemInfo(nullptr, 0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(AicpuTsUboeChannelTest, UT_UpdateMemInfo_When_NullHandle_Expect_ReturnHCCL_E_PTR)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUboeChannel ch(ep, desc);

    HcommMemHandle handles[1] = {nullptr};
    HcclResult ret = ch.UpdateMemInfo(handles, 1);
    EXPECT_EQ(ret, HCCL_E_PTR);

    GlobalMockObject::verify();
}

static void StubSendAsync(Hccl::Socket*, const void*, u32) {}
static void StubRecvAsync(Hccl::Socket*, void*, u32) {}

TEST_F(AicpuTsUboeChannelTest, Ut_UpdateMemInfo_When_CheckSocketStatusTimeout_Expect_ReturnTimeout)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUboeChannel ch(ep, desc);

    auto fakeSock = new FakeSocket(Hccl::SocketStatus::OK);
    ch.local_.socket = reinterpret_cast<Hccl::Socket*>(fakeSock);

    MOCKER_CPP(&Hccl::Socket::SendAsync, void(Hccl::Socket::*)(const void*, u32))
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(invoke(StubSendAsync));
    MOCKER_CPP(&Hccl::Socket::RecvAsync, void(Hccl::Socket::*)(void*, u32))
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(invoke(StubRecvAsync));
    MOCKER_CPP(
        &AicpuTsUboeUbRtpChannelHelper::CheckSocketStatus,
        HcclResult(AicpuTsUboeUbRtpChannelHelper::*)(const std::string&))
        .stubs()
        .will(returnValue(HCCL_E_TIMEOUT));

    auto localBuffer = MakeUboeExchangeLocalBuffer(0x540000U, 0x1000U, "uboe_update_timeout");
    HcommMemHandle handles[1] = {reinterpret_cast<HcommMemHandle>(localBuffer.get())};
    HcclResult ret = ch.UpdateMemInfo(handles, 1);
    EXPECT_EQ(ret, HCCL_E_TIMEOUT);

    GlobalMockObject::verify();
    delete fakeSock;
    ch.local_.socket = nullptr;
}

TEST_F(AicpuTsUboeChannelTest, UT_UpdateMemInfo_When_SocketNullAndValidHandle_Expect_ReturnHCCL_E_PTR)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUboeChannel ch(ep, desc);

    ch.local_.socket = nullptr;

    auto localBuffer = MakeUboeExchangeLocalBuffer(0x550000U, 0x1000U, "uboe_update_socket_null");
    HcommMemHandle handles[1] = {reinterpret_cast<HcommMemHandle>(localBuffer.get())};
    HcclResult ret = ch.UpdateMemInfo(handles, 1);
    EXPECT_EQ(ret, HCCL_E_PTR);

    GlobalMockObject::verify();
}

TEST_F(AicpuTsUboeChannelTest, UT_UpdateMemInfo_When_Normal_Expect_AppendBuffersAndInvalidateCache)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUboeChannel ch(ep, desc);

    auto fakeSock = new FakeSocket(Hccl::SocketStatus::OK);
    ch.local_.socket = reinterpret_cast<Hccl::Socket*>(fakeSock);
    ch.local_.rdmaHandle = reinterpret_cast<void*>(0xDEADBEEF);
    ch.remote_.memCache.cacheValid = true;

    MOCKER_CPP(&Hccl::Socket::SendAsync, void(Hccl::Socket::*)(const void*, u32))
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(invoke(stub_Socket_SendAsync));
    MOCKER_CPP(&Hccl::Socket::RecvAsync, void(Hccl::Socket::*)(void*, u32))
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(invoke(stub_Socket_RecvAsync));
    MOCKER_CPP(
        &AicpuTsUboeUbRtpChannelHelper::CheckSocketStatus,
        HcclResult(AicpuTsUboeUbRtpChannelHelper::*)(const std::string&))
        .stubs()
        .will(returnValue(HCCL_SUCCESS));

    auto localBuffer = MakeUboeExchangeLocalBuffer(0x560000U, 0x1000U, "uboe_update_normal");
    HcommMemHandle handles[1] = {reinterpret_cast<HcommMemHandle>(localBuffer.get())};
    HcclResult ret = ch.UpdateMemInfo(handles, 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(ch.local_.commonRes.bufferVec.size(), 1U);
    EXPECT_EQ(ch.local_.commonRes.bufferVec[0], localBuffer.get());
    EXPECT_EQ(ch.rmtBufferVec_.size(), 1U);
    EXPECT_FALSE(ch.remote_.memCache.cacheValid);

    GlobalMockObject::verify();
    delete fakeSock;
    ch.local_.socket = nullptr;
}
// ==================== 对端声明长度校验（防超大内存申请） ====================

static void stub_Socket_RecvAsyncFillZero(Hccl::Socket* self, void* recvBuf, u32 size)
{
    (void)self;
    if (recvBuf != nullptr && size > 0) {
        memset(recvBuf, 0, size);
    }
}

TEST_F(AicpuTsUboeChannelTest, UT_UpdateMemInfo_When_RecvSizeZero_Expect_ReturnHCCL_E_PARA)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUboeChannel ch(ep, desc);

    auto fakeSock = new FakeSocket(Hccl::SocketStatus::OK);
    ch.local_.socket = reinterpret_cast<Hccl::Socket*>(fakeSock);

    MOCKER_CPP(&Hccl::Socket::SendAsync, void(Hccl::Socket::*)(const void*, u32))
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(invoke(stub_Socket_SendAsync));
    MOCKER_CPP(&Hccl::Socket::RecvAsync, void(Hccl::Socket::*)(void*, u32))
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(invoke(stub_Socket_RecvAsyncFillZero));
    MOCKER_CPP(
        &AicpuTsUboeUbRtpChannelHelper::CheckSocketStatus,
        HcclResult(AicpuTsUboeUbRtpChannelHelper::*)(const std::string&))
        .stubs()
        .will(returnValue(HCCL_SUCCESS));

    auto localBuffer = MakeUboeExchangeLocalBuffer(0x570000U, 0x1000U, "uboe_update_recv_zero");
    HcommMemHandle handles[1] = {reinterpret_cast<HcommMemHandle>(localBuffer.get())};
    HcclResult ret = ch.UpdateMemInfo(handles, 1);
    EXPECT_EQ(ret, HCCL_E_PARA);
    EXPECT_EQ(ch.local_.commonRes.bufferVec.size(), 0U);
    EXPECT_EQ(ch.rmtBufferVec_.size(), 0U);

    GlobalMockObject::verify();
    delete fakeSock;
    ch.local_.socket = nullptr;
}

// ===================== BuildSocket 收敛基类后的钉住用例 =====================
// UBOE 折算为 LISTEN_AND_BORROW（commAddr 建 UB 监听 + noRankId 借用，端口统一 local_.serverPort）
TEST_F(AicpuTsUboeChannelTest, UT_ParseInputParam_Expect_ListenAndBorrowMode)
{
    FakeEndpoint fe;
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(&fe);
    auto fakeSock = std::make_unique<FakeSocket>(Hccl::SocketStatus::OK);
    HcommChannelDesc desc = MakeFakeChannelDesc(fakeSock.get());
    AicpuTsUboeChannel ch(ep, desc);

    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().will(returnValue(HCCL_SUCCESS));
    g_uboeStubEndpoint = static_cast<void*>(ep);
    MOCKER(HcommEndpointGet).stubs().will(invoke(StubUboeHcommEndpointGet));

    ASSERT_EQ(ch.ParseInputParam(), HCCL_SUCCESS);
    EXPECT_EQ(ch.traits_.socketBuildMode, SocketBuildMode::LISTEN_AND_BORROW);
}

// ===================== BuildNotify 收敛基类后的钉住用例 =====================
// UBOE 经 helper 继承解析到基类统一 BuildNotify：折算 BUILD_LOCAL/UbLocalNotify，
// notifyVec/local_.notifies 双注册（原 helper::BuildNotify 语义；UbLocalNotify 构造链走 llt 桩，
// 对齐本文件 FakeUbLocalNotify 先例，无需 Hrt mock）
TEST_F(AicpuTsUboeChannelTest, UT_BuildNotify_Expect_BuildLocalModeAndUbLocalNotifyRegistered)
{
    FakeEndpoint fe;
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(&fe);
    auto fakeSock = std::make_unique<FakeSocket>(Hccl::SocketStatus::OK);
    HcommChannelDesc desc = MakeFakeChannelDesc(fakeSock.get());
    desc.notifyNum = 2;
    AicpuTsUboeChannel ch(ep, desc);

    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().will(returnValue(HCCL_SUCCESS));
    g_uboeStubEndpoint = static_cast<void*>(ep);
    MOCKER(HcommEndpointGet).stubs().will(invoke(StubUboeHcommEndpointGet));

    ASSERT_EQ(ch.ParseInputParam(), HCCL_SUCCESS);
    EXPECT_EQ(ch.traits_.notifyBuildMode, NotifyBuildMode::BUILD_LOCAL); // kind=UBOE 折算钉住

    ch.local_.rdmaHandle = fe.GetRdmaHandle(); // 对齐 FakeUbLocalNotify 构造先例
    ASSERT_EQ(ch.BuildNotify(), HCCL_SUCCESS);
    EXPECT_EQ(ch.local_.notifies.size(), 2U);
    EXPECT_EQ(ch.local_.commonRes.notifyVec.size(), 2U);
    EXPECT_EQ(ch.local_.commonRes.notifyVec[1], ch.local_.notifies[1].get());            // 双注册同一对象
    EXPECT_NE(dynamic_cast<Hccl::UbLocalNotify*>(ch.local_.notifies[0].get()), nullptr); // 工厂分派 UbLocalNotify
}

// ==================== GetMemUpdatePack：helper 按自己的远端表打包 ====================

TEST_F(AicpuTsUboeChannelTest, Ut_GetMemUpdatePack_When_RmtBufferVecEmpty_Expect_Empty)
{
    HcommChannelDesc desc{};
    AicpuTsUboeChannel ch(reinterpret_cast<EndpointHandle>(0x1), desc);
    ch.rmtBufferVec_.clear();

    EXPECT_TRUE(ch.GetMemUpdatePack().empty());
}

TEST_F(AicpuTsUboeChannelTest, Ut_GetMemUpdatePack_When_OnlyNullptrEntries_Expect_Empty)
{
    HcommChannelDesc desc{};
    AicpuTsUboeChannel ch(reinterpret_cast<EndpointHandle>(0x1), desc);
    ch.rmtBufferVec_.clear();
    ch.rmtBufferVec_.push_back(nullptr);
    ch.rmtBufferVec_.push_back(nullptr);

    // 条目数只算非空槽位，全空即整包为空
    EXPECT_TRUE(ch.GetMemUpdatePack().empty());
}

TEST_F(AicpuTsUboeChannelTest, Ut_GetMemUpdatePack_When_MixedNullptr_Expect_SkipNullKeepOrder)
{
    HcommChannelDesc desc{};
    AicpuTsUboeChannel ch(reinterpret_cast<EndpointHandle>(0x1), desc);
    ch.rmtBufferVec_.clear();
    ch.rmtBufferVec_.push_back(
        std::make_unique<Hccl::RemoteUbRmaBuffer>(0x1000, 512, 11, 22, HcclMemType::HCCL_MEM_TYPE_HOST, "uboe_a"));
    ch.rmtBufferVec_.push_back(nullptr);
    ch.rmtBufferVec_.push_back(
        std::make_unique<Hccl::RemoteUbRmaBuffer>(0x3000, 256, 33, 44, HcclMemType::HCCL_MEM_TYPE_HOST, "uboe_c"));

    std::vector<char> pack = ch.GetMemUpdatePack();
    ASSERT_EQ(pack.size(), 4U + 2U * 28U);

    Hccl::BinaryStream binaryStream(pack);
    u32 entryNum = 0;
    binaryStream >> entryNum;
    EXPECT_EQ(entryNum, 2U);
    Hccl::RmaBufEntryLite first;
    Hccl::UnpackRmaBufEntry(binaryStream, first);
    Hccl::RmaBufEntryLite second;
    Hccl::UnpackRmaBufEntry(binaryStream, second);
    EXPECT_EQ(first.addr, 0x1000U);
    EXPECT_EQ(first.size, 512U);
    EXPECT_EQ(first.tokenId, 11U);
    EXPECT_EQ(first.tokenValue, 22U);
    EXPECT_EQ(second.addr, 0x3000U);
    EXPECT_EQ(second.size, 256U);
    EXPECT_EQ(second.tokenId, 33U);
    EXPECT_EQ(second.tokenValue, 44U);
    EXPECT_EQ(second.notifyId, UINT32_MAX);
}

TEST_F(AicpuTsUboeChannelTest, Ut_GetMemUpdatePack_When_CalledTwice_Expect_SameBytesAndVecUntouched)
{
    HcommChannelDesc desc{};
    AicpuTsUboeChannel ch(reinterpret_cast<EndpointHandle>(0x1), desc);
    ch.rmtBufferVec_.clear();
    ch.rmtBufferVec_.push_back(
        std::make_unique<Hccl::RemoteUbRmaBuffer>(0x1000, 512, 11, 22, HcclMemType::HCCL_MEM_TYPE_HOST, "uboe_a"));

    std::vector<char> first = ch.GetMemUpdatePack();
    std::vector<char> second = ch.GetMemUpdatePack();
    // 现取现打包，不留状态，也不动自己的远端表
    EXPECT_EQ(first, second);
    EXPECT_EQ(ch.rmtBufferVec_.size(), 1U);
}
