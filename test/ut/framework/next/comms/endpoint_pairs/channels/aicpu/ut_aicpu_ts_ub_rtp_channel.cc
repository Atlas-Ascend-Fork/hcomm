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

#define private public
#define protected public
#include "aicpu/aicpu_ts_ub_rtp_channel.h"
#undef protected
#undef private
#include "endpoint.h"
#include "hcomm_c_adpt.h"
#include "exchange_ub_buffer_dto.h"

#define private public
#define protected public
using namespace hcomm;

class AicpuTsUbRtpChannelTest : public testing::Test {
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
        GlobalMockObject::reset();
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
EndpointDesc MakeUbRtpEndpointDesc()
{
    EndpointDesc desc{};
    Hccl::IpAddress localIp("127.0.0.1");
    desc.protocol = COMM_PROTOCOL_UB_RTP;
    desc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    desc.commAddr.addr = localIp.GetBinaryAddress().addr;
    desc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    return desc;
}

class FakeParseEndpoint : public Endpoint {
public:
    FakeParseEndpoint() : Endpoint(MakeUbRtpEndpointDesc()) { ctxHandle_ = reinterpret_cast<void*>(0xDEADBEEF); }

    HcclResult Init() override { return HCCL_SUCCESS; }
    // 内存方法在 mgr 上，Endpoint 不再承载
    LocalRegedMemMgr* GetLocalRegMemMgr() override { return nullptr; }
    void* GetRdmaHandle() override { return ctxHandle_; }
    bool IsCtxHandleValid() const override { return ctxHandle_ != nullptr; }

private:
    void* ctxHandle_{nullptr};
};

std::shared_ptr<Hccl::LocalUbRmaBuffer> MakeUbRtpLocalBuffer(uintptr_t addr, u64 size, const char* tag)
{
    auto buffer = std::make_shared<Hccl::Buffer>(addr, size, HCCL_MEM_TYPE_DEVICE, tag);
    return std::make_shared<Hccl::LocalUbRmaBuffer>(buffer);
}

HcommResult StubUbRtpGetAllMemHandlesOne(EndpointHandle, void** memHandles, uint32_t* memHandleNum)
{
    static std::shared_ptr<Hccl::LocalUbRmaBuffer> localBuffer = MakeUbRtpLocalBuffer(0x620000U, 0x1000U, "ub_rtp_all");
    static std::shared_ptr<Hccl::LocalUbRmaBuffer> localBuffers[1] = {localBuffer};
    *memHandles = localBuffers;
    *memHandleNum = 1;
    return HCCL_SUCCESS;
}

// HcommEndpointGet 的 void** 指针出参不能用 outBound（mockcpp 的 outBound 仅支持引用出参），
// 须用 invoke 桩函数回写出参（先例：ut_socket_process.cc 的 StubSocketMgrHrtGetDevice 系列）
static void* g_ubRtpStubEndpoint = nullptr;

static HcommResult StubUbRtpHcommEndpointGet(EndpointHandle /*endpointHandle*/, void** endpoint)
{
    *endpoint = g_ubRtpStubEndpoint;
    return static_cast<HcommResult>(HCCL_SUCCESS);
}
} // namespace

struct FakeEndpoint {
    EndpointDesc desc;
    void* rdmaHandle{reinterpret_cast<void*>(0xDEADBEEF)};
    FakeEndpoint()
    {
        memset(&desc, 0, sizeof(desc));
        Hccl::IpAddress localIp("127.0.0.1");
        desc.protocol = COMM_PROTOCOL_UB_RTP;
        desc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
        desc.commAddr.addr = localIp.GetBinaryAddress().addr;
        desc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    }
    EndpointDesc GetEndpointDesc() { return desc; }
    void* GetRdmaHandle() { return rdmaHandle; }
    Hccl::IpAddress GetIpAddress() const { return Hccl::IpAddress("127.0.0.1"); }
};

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
        if (buf && size) {
            if (!sent_.empty()) {
                u32 copySize = static_cast<u32>(std::min<size_t>(sent_.size(), static_cast<size_t>(size)));
                memcpy(buf, sent_.data(), copySize);
                if (copySize < size)
                    std::memset(buf + copySize, 0, size - copySize);
                sent_.erase(sent_.begin(), sent_.begin() + copySize);
            } else {
                std::memset(buf, 0, size);
            }
        }
    }
    Hccl::SocketStatus GetAsyncStatus() { return status_; }
    Hccl::SocketRole GetRole() const { return Hccl::SocketRole::SERVER; }

    Hccl::SocketStatus status_;
    std::vector<u8> sent_;
};

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

TEST_F(AicpuTsUbRtpChannelTest, Ut_GetChannelKind_Returns_AICPU_TS_UB_RTP)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUbRtpChannel ch(ep, desc);
    EXPECT_EQ(ch.GetChannelKind(), HcommChannelKind::AICPU_TS_UB_RTP);
}

TEST_F(AicpuTsUbRtpChannelTest, UT_ParseInputParam_When_ExchangeAllMemsFalse_Expect_FillCommonRes)
{
    FakeParseEndpoint endpoint;
    auto localBuffer = MakeUbRtpLocalBuffer(0x610000U, 0x1000U, "ub_rtp_desc");
    HcommMemHandle memHandles[1] = {reinterpret_cast<HcommMemHandle>(localBuffer.get())};
    HcommChannelDesc desc{};
    desc.exchangeAllMems = false;
    desc.memHandles = memHandles;
    desc.memHandleNum = 1;
    AicpuTsUbRtpChannel channel(reinterpret_cast<EndpointHandle>(&endpoint), desc);

    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().will(returnValue(HCCL_SUCCESS));
    g_ubRtpStubEndpoint = static_cast<void*>(&endpoint);
    MOCKER(HcommEndpointGet).stubs().will(invoke(StubUbRtpHcommEndpointGet));

    ASSERT_EQ(channel.ParseInputParam(), HCCL_SUCCESS);
    ASSERT_EQ(channel.local_.commonRes.bufferVec.size(), 1U);
    EXPECT_EQ(channel.local_.commonRes.bufferVec[0], localBuffer.get());
}

TEST_F(AicpuTsUbRtpChannelTest, UT_ParseInputParam_When_ExchangeAllMemsTrue_Expect_FillCommonRes)
{
    FakeParseEndpoint endpoint;
    HcommChannelDesc desc{};
    desc.exchangeAllMems = true;
    AicpuTsUbRtpChannel channel(reinterpret_cast<EndpointHandle>(&endpoint), desc);
    MOCKER(HcommMemGetAllMemHandles).stubs().will(invoke(StubUbRtpGetAllMemHandlesOne));
    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().will(returnValue(HCCL_SUCCESS));
    g_ubRtpStubEndpoint = static_cast<void*>(&endpoint);
    MOCKER(HcommEndpointGet).stubs().will(invoke(StubUbRtpHcommEndpointGet));

    ASSERT_EQ(channel.ParseInputParam(), HCCL_SUCCESS);
    ASSERT_EQ(channel.local_.commonRes.bufferVec.size(), 1U);
    EXPECT_EQ(channel.local_.commonRes.bufferVec[0]->GetAddr(), 0x620000U);
}

TEST_F(AicpuTsUbRtpChannelTest, Ut_BuildConnection_Creates_UbRtpConn)
{
    FakeEndpoint fe;
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(&fe);
    auto fakeSock = std::make_unique<FakeSocket>();
    HcommChannelDesc desc = MakeFakeChannelDesc(fakeSock.get());
    desc.qos = 3U;

    AicpuTsUbRtpChannel ch(ep, desc);
    ch.local_.ep = fe.GetEndpointDesc();
    ch.remote_.ep = fe.GetEndpointDesc();
    ch.local_.rdmaHandle = fe.GetRdmaHandle();
    Hccl::IpAddress locAddr = fe.GetIpAddress();
    Hccl::IpAddress rmtAddr = fe.GetIpAddress();
    ch.local_.ip = locAddr;
    ch.remote_.ip = rmtAddr;
    ch.devBaseAttr_.maxReadSize = 256 * 1024 * 1024;
    ch.devBaseAttr_.maxWriteSize = 256 * 1024 * 1024;

    MOCKER_CPP(&Hccl::TpManager::Init).stubs();
    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));

    auto ret = ch.BuildStateMachineConn();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ASSERT_EQ(ch.local_.commonRes.connVec.size(), 1u);
    auto* conn = dynamic_cast<Hccl::DevUbRtpConnection*>(ch.local_.commonRes.connVec[0]);
    ASSERT_NE(conn, nullptr);
    EXPECT_EQ(conn->qos_, static_cast<u8>(3));
}

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
    std::memset(buf, 0, size);
}

TEST_F(AicpuTsUbRtpChannelTest, Ut_ProcessUbRtpState_AllStates_Transitions)
{
    FakeEndpoint fe;
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(&fe);
    auto fakeSock = std::make_unique<FakeSocket>();

    MOCKER_CPP(&Hccl::Socket::SendAsync, void(Hccl::Socket::*)(const void*, u32))
        .stubs()
        .with(any(), any())
        .will(invoke(stub_Socket_SendAsync));
    MOCKER_CPP(&Hccl::Socket::RecvAsync, void(Hccl::Socket::*)(void*, u32))
        .stubs()
        .with(any(), any())
        .will(invoke(stub_Socket_RecvAsync));

    HcommChannelDesc desc = MakeFakeChannelDesc(fakeSock.get());

    AicpuTsUbRtpChannel ch(ep, desc);

    FakeEndpoint localEp;
    FakeEndpoint remoteEp;
    ch.local_.ep = localEp.GetEndpointDesc();
    ch.remote_.ep = remoteEp.GetEndpointDesc();

    ch.channelDesc_.socket = reinterpret_cast<void*>(fakeSock.get());
    ch.local_.socket = reinterpret_cast<Hccl::Socket*>(fakeSock.get());

    ch.channelDesc_.exchangeAllMems = false;

    ch.local_.commonRes.connVec = {};
    ch.local_.commonRes.bufferVec = {};
    ch.connNum_ = 0;
    ch.bufferNum_ = 0;

    ch.channelStatus = ChannelStatus::SOCKET_OK;
    ch.ubRtpStatus = AicpuTsUbRtpChannel::UbRtpStatus::SEND_SIZE;

    int iter = 0;
    while (ch.channelStatus != ChannelStatus::READY && iter < 200) {
        ch.GetStatus();
        iter++;
    }

    EXPECT_EQ(ch.channelStatus, ChannelStatus::READY);
    EXPECT_EQ(ch.ubRtpStatus, AicpuTsUbRtpChannel::UbRtpStatus::READY);
}

TEST_F(AicpuTsUbRtpChannelTest, Ut_GetStatus_WhenReady_ReturnsReady)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUbRtpChannel ch(ep, desc);
    ch.channelStatus = ChannelStatus::READY;
    EXPECT_EQ(ch.GetStatus(), ChannelStatus::READY);
}

TEST_F(AicpuTsUbRtpChannelTest, Ut_SendFinish_RecvFinish_NoCrash)
{
    FakeEndpoint fe;
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(&fe);
    auto fakeSock = std::make_unique<FakeSocket>();
    HcommChannelDesc desc = MakeFakeChannelDesc(fakeSock.get());

    MOCKER_CPP(&Hccl::Socket::SendAsync, void(Hccl::Socket::*)(const void*, u32))
        .stubs()
        .with(any(), any())
        .will(invoke(stub_Socket_SendAsync));
    MOCKER_CPP(&Hccl::Socket::RecvAsync, void(Hccl::Socket::*)(void*, u32))
        .stubs()
        .with(any(), any())
        .will(invoke(stub_Socket_RecvAsync));

    AicpuTsUbRtpChannel ch(ep, desc);
    ch.local_.socket = reinterpret_cast<Hccl::Socket*>(fakeSock.get());

    ch.SendFinish();
    ch.RecvFinish();
    SUCCEED();
}

TEST_F(AicpuTsUbRtpChannelTest, Ut_H2DResPack_Packs_Data)
{
    FakeEndpoint fe;
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(&fe);
    auto fakeSock = std::make_unique<FakeSocket>();
    HcommChannelDesc desc = MakeFakeChannelDesc(fakeSock.get());

    AicpuTsUbRtpChannel ch(ep, desc);

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

    Hccl::IpAddress ipAddress("127.0.0.1");
    Hccl::DevUbRtpConnection ubRtpConn(
        fe.GetRdmaHandle(), ipAddress, ipAddress, Hccl::OpMode::OPBASE, true, Hccl::HrtUbJfcMode::STARS_POLL, ipAddress,
        ipAddress);
    ch.local_.commonRes.connVec.push_back(&ubRtpConn);

    std::vector<char> out{};
    HcclResult ret = ch.H2DResPack(out);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(AicpuTsUbRtpChannelTest, Ut_DataPlaneInterfaces_ReturnNotSupport)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUbRtpChannel ch(ep, desc);

    EXPECT_EQ(ch.NotifyRecord(0), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(ch.NotifyWait(0, 1000), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(ch.WriteWithNotify(nullptr, nullptr, 0, 0), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(ch.Write(nullptr, nullptr, 0), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(ch.Read(nullptr, nullptr, 0), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(ch.ChannelFence(), HCCL_E_NOT_SUPPORT);
}

TEST_F(AicpuTsUbRtpChannelTest, Ut_Clean_Resume_ReturnSuccess)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUbRtpChannel ch(ep, desc);

    EXPECT_EQ(ch.Clean(), HCCL_SUCCESS);
    EXPECT_EQ(ch.Resume(), HCCL_SUCCESS);
}

TEST_F(AicpuTsUbRtpChannelTest, Ut_GetNotifyNum_ReturnsValue)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUbRtpChannel ch(ep, desc);
    ch.local_.notifyNum = 5;

    uint32_t num = 0;
    EXPECT_EQ(ch.GetNotifyNum(&num), HCCL_SUCCESS);
    EXPECT_EQ(num, 5u);
}

TEST_F(AicpuTsUbRtpChannelTest, Ut_IsSocketReady_NullptrSocket_ReturnsFalse)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUbRtpChannel ch(ep, desc);
    ch.local_.socket = nullptr;

    EXPECT_FALSE(ch.IsSocketReady());
    EXPECT_EQ(ch.channelStatus, ChannelStatus::INVALID);
}

TEST_F(AicpuTsUbRtpChannelTest, Ut_IsResReady_EmptyConnVec_ReturnsTrue)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUbRtpChannel ch(ep, desc);
    ch.local_.commonRes.connVec = {};

    bool isReady = false;
    EXPECT_EQ(ch.IsResReady(isReady), HCCL_SUCCESS);
    EXPECT_TRUE(isReady);
}

TEST_F(AicpuTsUbRtpChannelTest, Ut_IsConnsReady_ZeroConnNum_ReturnsTrue)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUbRtpChannel ch(ep, desc);
    ch.connNum_ = 0;

    EXPECT_TRUE(ch.IsConnsReady());
}

TEST_F(AicpuTsUbRtpChannelTest, Ut_PackingHelpers_EmptyVecs_NoCrash)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUbRtpChannel ch(ep, desc);
    ch.local_.notifyNum = 0;
    ch.connNum_ = 0;
    ch.bufferNum_ = 0;
    ch.local_.commonRes.connVec = {};
    ch.local_.commonRes.bufferVec = {};
    ch.local_.commonRes.notifyVec = {};

    Hccl::BinaryStream bs;
    ch.NotifyVecPack(bs);
    std::vector<Hccl::LocalRmaBuffer*> emptyBuf;
    ch.BufferVecPack(bs, emptyBuf);
    ch.ConnVecPack(bs);

    SUCCEED();
}

TEST_F(AicpuTsUbRtpChannelTest, Ut_GetUniqueIdV2_NotReady_Expect_E_INTERNAL)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUbRtpChannel ch(ep, desc);
    ch.channelStatus = ChannelStatus::INIT;

    std::vector<char> uniqueIdV2;
    EXPECT_EQ(ch.GetUniqueIdV2(uniqueIdV2), HcclResult::HCCL_E_INTERNAL);
}

std::shared_ptr<Hccl::LocalRmaBuffer> MakeUbRtpExchangeLocalBuffer(uintptr_t addr, u64 size, const char* tag)
{
    auto buffer = std::make_shared<Hccl::Buffer>(addr, size, HCCL_MEM_TYPE_DEVICE, tag);
    return std::make_shared<Hccl::FakeExchangeLocalRmaBuffer>(buffer);
}

static void StubSendAsync(Hccl::Socket* self, const void* sendBuf, u32 size)
{
    if (!self || !sendBuf || size == 0) {
        return;
    }
    auto* fs = dynamic_cast<FakeSocket*>(self);
    if (fs) {
        auto* p = static_cast<const u8*>(sendBuf);
        fs->sent_.insert(fs->sent_.end(), p, p + size);
    }
}

static void StubRecvAsync(Hccl::Socket* self, void* recvBuf, u32 size)
{
    if (!self || !recvBuf || size == 0) {
        return;
    }
    auto* buf = static_cast<u8*>(recvBuf);
    auto* fs = dynamic_cast<FakeSocket*>(self);
    if (fs && !fs->sent_.empty()) {
        u32 copySize = static_cast<u32>(std::min<size_t>(fs->sent_.size(), static_cast<size_t>(size)));
        memcpy(buf, fs->sent_.data(), copySize);
        if (copySize < size) {
            std::memset(buf + copySize, 0, size - copySize);
        }
        fs->sent_.erase(fs->sent_.begin(), fs->sent_.begin() + copySize);
        return;
    }
    std::memset(buf, 0, size);
}

TEST_F(AicpuTsUbRtpChannelTest, UT_UpdateMemInfo_When_MemHandleNumZero_Expect_ReturnHCCL_SUCCESS)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUbRtpChannel ch(ep, desc);

    HcclResult ret = ch.UpdateMemInfo(nullptr, 0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(AicpuTsUbRtpChannelTest, UT_UpdateMemInfo_When_NullHandle_Expect_ReturnHCCL_E_PTR)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUbRtpChannel ch(ep, desc);

    HcommMemHandle handles[1] = {nullptr};
    HcclResult ret = ch.UpdateMemInfo(handles, 1);
    EXPECT_EQ(ret, HCCL_E_PTR);

    GlobalMockObject::verify();
}

TEST_F(AicpuTsUbRtpChannelTest, UT_UpdateMemInfo_When_SocketNull_Expect_ReturnHCCL_E_PTR)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUbRtpChannel ch(ep, desc);
    ch.local_.socket = nullptr;

    auto localBuffer = MakeUbRtpExchangeLocalBuffer(0x650000U, 0x1000U, "ub_rtp_update_socket_null");
    HcommMemHandle handles[1] = {reinterpret_cast<HcommMemHandle>(localBuffer.get())};
    HcclResult ret = ch.UpdateMemInfo(handles, 1);
    EXPECT_EQ(ret, HCCL_E_PTR);

    GlobalMockObject::verify();
}

TEST_F(AicpuTsUbRtpChannelTest, UT_UpdateMemInfo_When_Normal_Expect_AppendBuffersAndInvalidateCache)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUbRtpChannel ch(ep, desc);

    auto fakeSock = new FakeSocket(Hccl::SocketStatus::OK);
    ch.local_.socket = reinterpret_cast<Hccl::Socket*>(fakeSock);
    ch.local_.rdmaHandle = reinterpret_cast<void*>(0xDEADBEEF);
    ch.remote_.memCache.cacheValid = true;

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
        .will(returnValue(HCCL_SUCCESS));

    auto localBuffer = MakeUbRtpExchangeLocalBuffer(0x660000U, 0x1000U, "ub_rtp_update_normal");
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

// ===================== BuildNotify 收敛基类后的钉住用例 =====================
// UB_RTP 折算 BUILD_LOCAL：与 UBOE 同经 helper 继承解析（mode 折算钉住）
TEST_F(AicpuTsUbRtpChannelTest, Ut_GetNotifyBuildMode_Expect_BUILD_LOCAL)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUbRtpChannel ch(ep, desc);
    EXPECT_EQ(ch.GetNotifyBuildMode(), NotifyBuildMode::BUILD_LOCAL);
}
