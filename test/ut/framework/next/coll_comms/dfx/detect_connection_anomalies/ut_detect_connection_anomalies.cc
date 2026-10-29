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
#include <any>
#include <sstream>
#include <thread>
#include <chrono>
#include <map>
#include <vector>
#include <securec.h>

#define private public
#define protected public
#include "detect_connection_anomalies.h"
#include "socket_mgr.h"
#include "server_socket_mgr.h"
#include "env_config_v2.h"
#include "orion_adapter_rts.h"
#include "socket_handle_manager.h"
#include "orion_adpt_utils.h"
#undef private
#undef protected

using namespace std;
using namespace hcomm;
using namespace Hccl;

// 本地常量：与业务端口默认值保持一致（device=16666，host=60001）
constexpr uint32_t DCA_DEVICE_DEFAULT_PORT = 16666;
constexpr uint32_t DCA_HOST_DEFAULT_PORT = 60001;

// 借用栈上/静态 Socket 的 shared_ptr：空删除器，仅借用不接管生命周期
static std::shared_ptr<Hccl::Socket> BorrowSocket(Hccl::Socket* socket)
{
    return std::shared_ptr<Hccl::Socket>(socket, [](Hccl::Socket*) {});
}

// ServerDetectThread 编排直测桩：记录调用次序并计数白名单下发（检视 D7）
static int g_whiteListAddCount = 0;
static std::vector<int> g_serverDetectOrder;
static HcclResult g_startListenRet = HCCL_SUCCESS;
static HcclResult g_whiteListAddRet = HCCL_SUCCESS;
static HcclResult g_acceptLoopRet = HCCL_SUCCESS;
static HcclResult g_broadcastLoopRet = HCCL_SUCCESS;
static int g_deleteWhiteListCount = 0;
static int g_stopListenCount = 0;
static HcclResult StubServerSocketStartListen(
    hcomm::ServerSocketMgr* /*self*/, const Hccl::PortData& /*localPort*/, Hccl::NicType /*nicType*/,
    uint32_t /*devPhyId*/, uint32_t* /*port*/)
{
    g_serverDetectOrder.push_back(1); // StartListen
    return g_startListenRet;
}
static HcclResult StubServerSocketAddWhiteList(
    hcomm::ServerSocketMgr* /*self*/, const Hccl::PortData& /*localPort*/, Hccl::NicType /*nicType*/,
    uint32_t /*devPhyId*/, const std::string& /*tag*/, uint32_t /*connLimit*/)
{
    ++g_whiteListAddCount;
    g_serverDetectOrder.push_back(2); // AddWhiteList
    return g_whiteListAddRet;
}
static HcclResult StubServerSocketDeleteWhiteList(
    hcomm::ServerSocketMgr* /*self*/, const Hccl::PortData& /*localPort*/, Hccl::NicType /*nicType*/,
    uint32_t /*devPhyId*/, const std::string& /*tag*/, uint32_t /*connLimit*/)
{
    ++g_deleteWhiteListCount;
    return HCCL_SUCCESS;
}
static HcclResult StubServerSocketStopListen(
    hcomm::ServerSocketMgr* /*self*/, const Hccl::PortData& /*localPort*/, Hccl::NicType /*nicType*/,
    const uint32_t /*port*/)
{
    ++g_stopListenCount;
    return HCCL_SUCCESS;
}
static HcclResult StubRunServerAcceptLoopOrder(
    DetectConnectionAnomalies* /*self*/, const LocalIpInfo& /*localInfo*/, uint32_t /*listenPort*/,
    Hccl::Socket* /*listener*/, std::vector<std::shared_ptr<Hccl::Socket>>& /*acceptSuccessSocketList*/)
{
    g_serverDetectOrder.push_back(3); // RunServerAcceptLoop
    return g_acceptLoopRet;
}
static HcclResult
StubRunBroadcastLoopOrder(DetectConnectionAnomalies* /*self*/, const std::vector<std::shared_ptr<Hccl::Socket>>&)
{
    g_serverDetectOrder.push_back(4); // RunBroadcastLoop
    return g_broadcastLoopRet;
}

// 广播发送计数桩：一次发完（RunBroadcastLoop 真实调用用）
static int g_sendHeartCalls = 0;
static HcclResult NonBlockSendHeartCountDoneStub(Hccl::FdHandle, void*, u64 size, u64* sentSize)
{
    ++g_sendHeartCalls;
    *sentSize = size;
    return HCCL_SUCCESS;
}

// 广播发送桩：首次 EAGAIN、随后成功（退避重试路径用）
static HcclResult NonBlockSendHeartEagainThenDoneStub(Hccl::FdHandle, void*, u64 size, u64* sentSize)
{
    ++g_sendHeartCalls;
    if (g_sendHeartCalls == 1) {
        *sentSize = 0;
        return HCCL_E_AGAIN;
    }
    *sentSize = size;
    return HCCL_SUCCESS;
}

// 收广播桩：一次填满 DetectInfo（解析入库路径用）
static DetectInfo g_recvDetectInfo{};
static HcclResult NonBlockRecvHeartFullStub(Hccl::FdHandle, void* data, u64 size, u64* recvSize)
{
    if (size < sizeof(DetectInfo)) {
        *recvSize = 0;
        return HCCL_E_AGAIN;
    }
    if (memcpy_s(data, size, &g_recvDetectInfo, sizeof(DetectInfo)) != EOK) {
        *recvSize = 0;
        return HCCL_E_INTERNAL;
    }
    *recvSize = sizeof(DetectInfo);
    return HCCL_SUCCESS;
}

// 收广播半包桩：首次返回半包，二次补齐（验证 recvSize 累积解析）
static int g_halfRecvCalls = 0;
static HcclResult NonBlockRecvHeartHalfThenRestStub(Hccl::FdHandle, void* data, u64 size, u64* recvSize)
{
    const u64 total = sizeof(DetectInfo);
    const u8* src = reinterpret_cast<const u8*>(&g_recvDetectInfo) + (total - size);
    u64 chunk = size;
    if (g_halfRecvCalls == 0) {
        chunk = (size + 1) / 2;
    }
    ++g_halfRecvCalls;
    if (memcpy_s(data, size, src, chunk) != EOK) {
        *recvSize = 0;
        return HCCL_E_INTERNAL;
    }
    *recvSize = chunk;
    return HCCL_SUCCESS;
}

// server accept 直测桩：listener 由 GetServerSocket 返回；AcceptSocket 首次给连接、其后 TIMEOUT
static Hccl::Socket* g_stubListener = nullptr;
static Hccl::Socket* g_stubAcceptedSocket = nullptr;
static int g_acceptCallCount = 0;
static HcclResult g_getServerSocketRet = HCCL_SUCCESS;
static HcclResult StubGetServerSocket(
    hcomm::ServerSocketMgr* /*self*/, const Hccl::PortData& /*localPort*/, Hccl::NicType /*nicType*/,
    const uint32_t /*port*/, std::shared_ptr<Hccl::Socket>& serverSocket)
{
    serverSocket = nullptr;
    if (g_getServerSocketRet != HCCL_SUCCESS) {
        return g_getServerSocketRet;
    }
    // 空删除器：stub listener 为用例内栈对象，shared_ptr 仅借用不接管生命周期
    serverSocket = BorrowSocket(g_stubListener);
    return HCCL_SUCCESS;
}
static HcclResult StubAcceptOnce(
    hcomm::SocketMgr* /*self*/, Hccl::Socket* /*listener*/, const Hccl::SocketConfig& /*socketConfig*/,
    std::shared_ptr<Hccl::Socket>& outSocket, u32 /*acceptTimeOut*/)
{
    ++g_acceptCallCount;
    if (g_acceptCallCount == 1) {
        // 空删除器：stub accepted socket 为用例内栈对象，shared_ptr 仅借用不接管生命周期
        outSocket = BorrowSocket(g_stubAcceptedSocket);
        return HCCL_SUCCESS;
    }
    outSocket = nullptr;
    return HCCL_E_TIMEOUT;
}

class DetectConnectAnomalies950Test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "DetectConnectAnomalies950Test SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "DetectConnectAnomalies950Test TearDown" << std::endl; }

    virtual void SetUp()
    {
        dca_ = &dcaInstance_;
        // threadRunning_=false 让所有 while(threadRunning_) 等待循环立即退出，UT 内线程秒退、不阻塞
        dca_->threadRunning_ = false;
        dca_->isPrint_ = false;
        dca_->localDevPhyId_ = 0;
        // 模拟 AddIpQueue 已刷新探测起始时间：新语义下收广播/收 size 窗口基于全局 startTime_（由 AddIpQueue 刷新）
        dca_->startTime_ = std::chrono::steady_clock::now();
        dca_->knownPeerIps_.clear();
        while (!dca_->peerIpQueue_.empty()) {
            dca_->peerIpQueue_.pop();
        }
        dca_->recvErrorInfoMap_.clear();
        dca_->clientThreads_.clear();
        dca_->serverThreads_.clear();
        while (!dca_->localIpQueue_.empty()) {
            dca_->localIpQueue_.pop();
        }
        dca_->knownLocalIps_.clear();
        // 清零引用计数必须有界：Referenced::Count()<0 时 IsZero() 恒假，避免死循环（检视 o#6）
        while (dca_->initRef_.Count() > 0) {
            dca_->initRef_.Unref();
        }
        // 固定探测窗口为 20s，避免上一个用例把 HCCL_DFS_CONFIG 置 0 污染后续用例
        setenv("HCCL_DFS_CONFIG", "connection_fault_detection_time:20", 1);
        Hccl::EnvConfig::GetInstance().Parse();
        std::cout << "A Test SetUp" << std::endl;
    }

    virtual void TearDown()
    {
        // 必须 join 所有线程，否则 std::thread 析构未 join 会 terminate（abort）
        dca_->threadRunning_ = false;
        JoinAllThreads();

        // 清理 device handle 单例缓存，避免用例间句柄缓存耦合（host 侧无 DeInit，测试使用唯一 IP）
        Hccl::SocketHandleManager::GetInstance().DeInit(dca_->localDevPhyId_);
        // 恢复 main.cc 的进程级基线（connection_fault_detection_time:0），避免 unsetenv 后回落默认 20 反向污染
        setenv("HCCL_DFS_CONFIG", "connection_fault_detection_time:0", 1);
        Hccl::EnvConfig::GetInstance().Parse();
        GlobalMockObject::verify();
        std::cout << "A Test TearDown" << std::endl;
    }

    void JoinAllThreads()
    {
        for (auto& t : dca_->clientThreads_) {
            if (t && t->joinable()) {
                t->join();
            }
        }
        dca_->clientThreads_.clear();
        for (auto& t : dca_->serverThreads_) {
            if (t && t->joinable()) {
                t->join();
            }
        }
        dca_->serverThreads_.clear();
        dca_->knownLocalIps_.clear();
        if (dca_->monitorThread_ && dca_->monitorThread_->joinable()) {
            dca_->monitorThread_->join();
        }
        dca_->monitorThread_.reset();
    }

    // AddIpQueue 现为 void 且带 !threadRunning_ 守卫：测试内放行守卫，
    // 并屏蔽 monitor 线程与轮次等待（保证入队后的队列状态可稳定断言）
    void PrepareAddIpQueue()
    {
        dca_->threadRunning_ = true;
        MOCKER_CPP(&DetectConnectionAnomalies::Detect).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&DetectConnectionAnomalies::WaitForDetect).stubs().will(returnValue(HCCL_SUCCESS));
    }

    static void FillServerId(char* dest, size_t destLen, const char* src)
    {
        size_t len = std::min(strlen(src), destLen - 1);
        // 安全函数返回值必须校验（CodeCheck G.FUU.11）
        if (memcpy_s(dest, destLen, src, len) != EOK) {
            dest[0] = '\0';
            return;
        }
        dest[len] = '\0';
    }

    static DetectInfo MakeInfo(const char* localSrv, int32_t localDev, const char* remoteSrv, int32_t remoteDev)
    {
        DetectInfo info{};
        info.localDeviceId = localDev;
        info.remoteDeviceId = remoteDev;
        FillServerId(info.localNetInstId, sizeof(info.localNetInstId), localSrv);
        FillServerId(info.remoteNetInstId, sizeof(info.remoteNetInstId), remoteSrv);
        return info;
    }

    static EndpointDesc MakeEndpointDesc(const char* ip, EndpointLocType locType, CommProtocol protocol)
    {
        EndpointDesc endpoint{};
        endpoint.protocol = protocol;
        endpoint.loc.locType = locType;
        (void)IpAddressToCommAddr(Hccl::IpAddress(ip), endpoint.commAddr);
        return endpoint;
    }

    static LocalIpInfo MakeLocalIp(const char* ip, EndpointLocType locType, uint32_t port)
    {
        LocalIpInfo info{};
        info.ip = Hccl::IpAddress(ip);
        info.port = port;
        info.endpoint = MakeEndpointDesc(ip, locType, COMM_PROTOCOL_UB_CTP); // UB_CTP 映射为 ConnectProtoType::UB
        return info;
    }

    static DcaPeerInfo MakePeer(
        uint32_t remoteRankId, const char* remoteIp, EndpointLocType locType, uint32_t remoteDevPhyId = 0,
        const char* localIp = "1.1.1.1", const char* remoteNetInst = "remote_net_inst")
    {
        DcaPeerInfo peer{};
        peer.remoteRankId = remoteRankId;
        FillServerId(peer.remoteNetInstId, sizeof(peer.remoteNetInstId), remoteNetInst);
        peer.remoteDevPhyId = remoteDevPhyId;
        peer.remoteIp = Hccl::IpAddress(remoteIp);
        peer.remotePort = (locType == ENDPOINT_LOC_TYPE_HOST) ? DCA_HOST_DEFAULT_PORT : DCA_DEVICE_DEFAULT_PORT;
        peer.localIp = Hccl::IpAddress(localIp);
        peer.localEndpoint = MakeEndpointDesc(localIp, locType, COMM_PROTOCOL_UB_CTP);
        peer.remoteEndpoint = MakeEndpointDesc(remoteIp, locType, COMM_PROTOCOL_UB_CTP);
        return peer;
    }

    void PresetMixedHandles() {}

    hcomm::DetectConnectionAnomalies dcaInstance_;
    DetectConnectionAnomalies* dca_ = nullptr;
};

// ==================================================================
// 1. GetTag / GetDfsConnFaultTime（基础工具）
// ==================================================================
TEST_F(DetectConnectAnomalies950Test, Ut_GetTag_When_IpV4V6_Expect_SuffixAndDistinct)
{
    // IPv4：追加固定后缀 _detect_0
    EXPECT_EQ(dca_->GetTag(IpAddress("1.2.3.4")), "1.2.3.4_detect_0");
    // IPv6：同样追加后缀
    EXPECT_EQ(dca_->GetTag(IpAddress("::1")), std::string("::1") + "_detect_0");
    // 不同 IP 产生不同 tag，同一 IP 的 tag 稳定一致
    EXPECT_NE(dca_->GetTag(IpAddress("1.1.1.1")), dca_->GetTag(IpAddress("1.1.1.2")));
    EXPECT_EQ(dca_->GetTag(IpAddress("1.2.3.4")), dca_->GetTag(IpAddress("1.2.3.4")));
}

TEST_F(DetectConnectAnomalies950Test, Ut_GetDfsConnFaultTime_When_DefaultAndConfigured_Expect_Values)
{
    // 未显式配置时返回 legacy 真默认值（20s），而非仅断言 >0（检视 o#18）
    unsetenv("HCCL_DFS_CONFIG");
    Hccl::EnvConfig::GetInstance().Parse();
    EXPECT_EQ(dca_->GetDfsConnFaultTime(), 20u);
    // 显式配置 20 / 0
    setenv("HCCL_DFS_CONFIG", "connection_fault_detection_time:20", 1);
    Hccl::EnvConfig::GetInstance().Parse();
    EXPECT_EQ(dca_->GetDfsConnFaultTime(), 20u);
    setenv("HCCL_DFS_CONFIG", "connection_fault_detection_time:0", 1);
    Hccl::EnvConfig::GetInstance().Parse();
    EXPECT_EQ(dca_->GetDfsConnFaultTime(), 0u);
}

TEST_F(DetectConnectAnomalies950Test, Ut_CommProtocolToConnectProtoType_When_AllProtocols_Expect_Mapping)
{
    // UB 家族（含旧别名 UBC_CTP / UBG）
    EXPECT_EQ(hcomm::CommProtocolToConnectProtoType(COMM_PROTOCOL_UB_CTP), Hccl::ConnectProtoType::UB);
    EXPECT_EQ(hcomm::CommProtocolToConnectProtoType(COMM_PROTOCOL_UBC_CTP), Hccl::ConnectProtoType::UB);
    EXPECT_EQ(hcomm::CommProtocolToConnectProtoType(COMM_PROTOCOL_UBC_TP), Hccl::ConnectProtoType::UB);
    EXPECT_EQ(hcomm::CommProtocolToConnectProtoType(COMM_PROTOCOL_UB_MEM), Hccl::ConnectProtoType::UB);
    // UB 家族（含 UBOE/UB_RTP/UBG 及旧别名）统一归一为 UB：监听 PortType 构造仅接受 TCP/RDMA/UB，
    // 归一后 UBOE/UB_RTP 端点走与产品通道一致的 UB 监听路径（检视 HIGH 修复）
    EXPECT_EQ(hcomm::CommProtocolToConnectProtoType(COMM_PROTOCOL_UB_RTP), Hccl::ConnectProtoType::UB);
    EXPECT_EQ(hcomm::CommProtocolToConnectProtoType(COMM_PROTOCOL_UBG), Hccl::ConnectProtoType::UB);
    // 直配协议
    EXPECT_EQ(hcomm::CommProtocolToConnectProtoType(COMM_PROTOCOL_ROCE), Hccl::ConnectProtoType::RDMA);
    EXPECT_EQ(hcomm::CommProtocolToConnectProtoType(COMM_PROTOCOL_PCIE), Hccl::ConnectProtoType::PCIE);
    EXPECT_EQ(hcomm::CommProtocolToConnectProtoType(COMM_PROTOCOL_HCCS), Hccl::ConnectProtoType::HCCS);
    EXPECT_EQ(hcomm::CommProtocolToConnectProtoType(COMM_PROTOCOL_HCCS_ONLY), Hccl::ConnectProtoType::HCCS);
    EXPECT_EQ(hcomm::CommProtocolToConnectProtoType(COMM_PROTOCOL_UBOE), Hccl::ConnectProtoType::UB);
    // 未列举协议（SIO / 保留值）→ 兜底 UB
    EXPECT_EQ(hcomm::CommProtocolToConnectProtoType(COMM_PROTOCOL_SIO), Hccl::ConnectProtoType::UB);
    EXPECT_EQ(hcomm::CommProtocolToConnectProtoType(COMM_PROTOCOL_RESERVED), Hccl::ConnectProtoType::UB);
}

// ==================================================================
// 2. Init / Deinit（无参 + 引用计数 + 生命周期搭配）
// ==================================================================
TEST_F(DetectConnectAnomalies950Test, Ut_Init_When_Non950_Expect_RefInc)
{
    // Init 无设备门禁（门禁在 AddIpQueue）：非 950 也照常 Ref
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(static_cast<Hccl::DevType>(Hccl::DevType::DEV_TYPE_910A3)));
    int refBefore = dca_->initRef_.Count();
    dca_->Init();
    EXPECT_EQ(dca_->initRef_.Count(), refBefore + 1);
    GlobalMockObject::verify();
}

TEST_F(DetectConnectAnomalies950Test, Ut_InitDeinit_When_Lifecycle_Expect_RefCountAndHandleClean)
{
    // Init 无参：仅门禁 + Ref（必须 init 才能 uninit 的生命周期搭配）
    dca_->Init();
    dca_->Init();
    dca_->Deinit();
    EXPECT_FALSE(dca_->initRef_.IsZero());
    dca_->Deinit();
    EXPECT_TRUE(dca_->initRef_.IsZero());
    EXPECT_FALSE(dca_->threadRunning_.load());
    EXPECT_EQ(dca_->monitorThread_, nullptr);
    // 过减：Deinit 幂等保护（count<0 直接 return，不再 ThreadDestroy；检视 o#28）
    dca_->threadRunning_ = true; // 若错误执行 ThreadDestroy 会被置 false
    dca_->Deinit();
    EXPECT_EQ(dca_->initRef_.Count(), -1);
    EXPECT_TRUE(dca_->threadRunning_.load()); // 未触发 ThreadDestroy
}

TEST_F(DetectConnectAnomalies950Test, Ut_AddIpQueue_When_GuardInput_Expect_NoEnqueue)
{
    PrepareAddIpQueue();
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(static_cast<Hccl::DevType>(Hccl::DevType::DEV_TYPE_950)));
    MOCKER(HrtRaSocketInit).stubs().will(returnValue(reinterpret_cast<SocketHandle>(0x1)));
    // 配置 dfsTime=0：不入队
    {
        setenv("HCCL_DFS_CONFIG", "connection_fault_detection_time:0", 1);
        Hccl::EnvConfig::GetInstance().Parse();
        std::vector<hcomm::LocalIpInfo> localIps{MakeLocalIp("1.1.1.9", ENDPOINT_LOC_TYPE_DEVICE, 16666)};
        DcaPeerInfo peer = MakePeer(1, "1.1.1.1", ENDPOINT_LOC_TYPE_DEVICE);
        dca_->AddIpQueue(-1, 0, "local_net_inst", std::vector<DcaPeerInfo>{peer}, localIps);
        EXPECT_TRUE(dca_->peerIpQueue_.empty());
        setenv("HCCL_DFS_CONFIG", "connection_fault_detection_time:20", 1);
        Hccl::EnvConfig::GetInstance().Parse();
    }
    // 空 failedPeers：不入队
    {
        std::vector<hcomm::LocalIpInfo> localIps{MakeLocalIp("1.1.1.9", ENDPOINT_LOC_TYPE_DEVICE, 16666)};
        dca_->AddIpQueue(-1, 0, "local_net_inst", std::vector<DcaPeerInfo>{}, localIps);
        EXPECT_TRUE(dca_->peerIpQueue_.empty());
    }
    GlobalMockObject::verify();
}

TEST_F(DetectConnectAnomalies950Test, Ut_AddIpQueue_When_Non950_Expect_Skip)
{
    // 非 950：门禁跳过，不入队（独立用例避免 mockcpp 同对象多次 stubs 覆盖）
    PrepareAddIpQueue();
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(static_cast<Hccl::DevType>(Hccl::DevType::DEV_TYPE_910A3)));
    std::vector<hcomm::LocalIpInfo> localIps{MakeLocalIp("1.1.1.9", ENDPOINT_LOC_TYPE_DEVICE, 16666)};
    DcaPeerInfo peer = MakePeer(1, "1.1.1.1", ENDPOINT_LOC_TYPE_DEVICE);
    dca_->AddIpQueue(-1, 0, "local_net_inst", std::vector<DcaPeerInfo>{peer}, localIps);
    EXPECT_TRUE(dca_->knownPeerIps_.empty());
    EXPECT_TRUE(dca_->peerIpQueue_.empty());
    GlobalMockObject::verify();
}

TEST_F(DetectConnectAnomalies950Test, Ut_AddIpQueue_When_DeviceFlow_Expect_CacheHandleEnqueue)
{
    // device peer：缓存本端参数 + 懒建 handle + 入队
    PrepareAddIpQueue();
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(static_cast<Hccl::DevType>(Hccl::DevType::DEV_TYPE_950)));
    MOCKER(HrtRaSocketInit).stubs().will(returnValue(reinterpret_cast<SocketHandle>(0x1)));
    std::vector<hcomm::LocalIpInfo> localIps{MakeLocalIp("1.1.1.9", ENDPOINT_LOC_TYPE_DEVICE, 16888)};
    DcaPeerInfo peer = MakePeer(1, "1.1.1.1", ENDPOINT_LOC_TYPE_DEVICE, 0, "1.1.1.9");
    dca_->AddIpQueue(-1, 3, "local_net_inst", std::vector<DcaPeerInfo>{peer}, localIps);
    EXPECT_EQ(dca_->localDevPhyId_, 3u);
    EXPECT_EQ(dca_->knownPeerIps_.size(), 1u);
    EXPECT_EQ(dca_->peerIpQueue_.front().remoteEndpoint.loc.locType, ENDPOINT_LOC_TYPE_DEVICE);
    EXPECT_EQ(dca_->peerIpQueue_.front().localIp.GetIpStr(), "1.1.1.9");
    GlobalMockObject::verify();
}

TEST_F(DetectConnectAnomalies950Test, Ut_AddIpQueue_When_HostFlow_Expect_HandlePeerEnqueue)
{
    // host peer：handle locType=HOST + host 端口 + 入队
    PrepareAddIpQueue();
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(static_cast<Hccl::DevType>(Hccl::DevType::DEV_TYPE_950)));
    MOCKER(HrtRaSocketInit).stubs().will(returnValue(reinterpret_cast<SocketHandle>(0x1)));
    std::vector<hcomm::LocalIpInfo> localIps{MakeLocalIp("2.2.2.9", ENDPOINT_LOC_TYPE_HOST, 60001)};
    DcaPeerInfo peer = MakePeer(1, "2.2.2.1", ENDPOINT_LOC_TYPE_HOST, 0, "2.2.2.9");
    dca_->AddIpQueue(-1, 6, "local_net_inst", std::vector<DcaPeerInfo>{peer}, localIps);
    EXPECT_EQ(dca_->knownPeerIps_.size(), 1u);
    EXPECT_EQ(dca_->peerIpQueue_.front().remoteEndpoint.loc.locType, ENDPOINT_LOC_TYPE_HOST);
    GlobalMockObject::verify();
}

TEST_F(DetectConnectAnomalies950Test, Ut_AddIpQueue_When_MixedPeers_Expect_BothQueued)
{
    // 一批内 device+host 混合：两条都入队
    PrepareAddIpQueue();
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(static_cast<Hccl::DevType>(Hccl::DevType::DEV_TYPE_950)));
    MOCKER(HrtRaSocketInit).stubs().will(returnValue(reinterpret_cast<SocketHandle>(0x1)));
    std::vector<hcomm::LocalIpInfo> localIps{
        MakeLocalIp("1.1.1.9", ENDPOINT_LOC_TYPE_DEVICE, 16666), MakeLocalIp("2.2.2.9", ENDPOINT_LOC_TYPE_HOST, 60001)};
    DcaPeerInfo peers[2]{
        MakePeer(1, "1.1.1.1", ENDPOINT_LOC_TYPE_DEVICE, 0, "1.1.1.9"),
        MakePeer(2, "2.2.2.1", ENDPOINT_LOC_TYPE_HOST, 0, "2.2.2.9")};
    dca_->AddIpQueue(-1, 0, "local_net_inst", std::vector<DcaPeerInfo>(peers, peers + 2), localIps);
    EXPECT_EQ(dca_->peerIpQueue_.size(), 2u);
    EXPECT_EQ(dca_->knownPeerIps_.size(), 2u);
    GlobalMockObject::verify();
}

TEST_F(DetectConnectAnomalies950Test, Ut_AddIpQueue_When_RepeatedRank_Expect_Dedup)
{
    PrepareAddIpQueue();
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(static_cast<Hccl::DevType>(Hccl::DevType::DEV_TYPE_950)));
    MOCKER(HrtRaSocketInit).stubs().will(returnValue(reinterpret_cast<SocketHandle>(0x1)));
    std::vector<hcomm::LocalIpInfo> localIps{MakeLocalIp("1.1.1.9", ENDPOINT_LOC_TYPE_DEVICE, 16666)};
    DcaPeerInfo peers[2]{
        MakePeer(1, "1.1.1.1", ENDPOINT_LOC_TYPE_DEVICE), MakePeer(1, "1.1.1.1", ENDPOINT_LOC_TYPE_DEVICE)};
    dca_->AddIpQueue(-1, 0, "local_net_inst", std::vector<DcaPeerInfo>(peers, peers + 2), localIps);
    EXPECT_EQ(dca_->knownPeerIps_.size(), 1u);
    GlobalMockObject::verify();
}

TEST_F(DetectConnectAnomalies950Test, Ut_AddIpQueue_When_SameIpDifferentRank_Expect_Dedup)
{
    // 去重维度为对端 IP：同 IP 的重复对端（即便 rankId 不同）只入队一次
    PrepareAddIpQueue();
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(static_cast<Hccl::DevType>(Hccl::DevType::DEV_TYPE_950)));
    MOCKER(HrtRaSocketInit).stubs().will(returnValue(reinterpret_cast<SocketHandle>(0x1)));
    std::vector<hcomm::LocalIpInfo> localIps{MakeLocalIp("1.1.1.9", ENDPOINT_LOC_TYPE_DEVICE, 16666)};
    DcaPeerInfo peers[2]{
        MakePeer(1, "1.1.1.1", ENDPOINT_LOC_TYPE_DEVICE, 0, "1.1.1.9"),
        MakePeer(2, "1.1.1.1", ENDPOINT_LOC_TYPE_DEVICE, 0, "1.1.1.9")};
    dca_->AddIpQueue(-1, 0, "local_net_inst", std::vector<DcaPeerInfo>(peers, peers + 2), localIps);
    EXPECT_EQ(dca_->knownPeerIps_.size(), 1u);
    EXPECT_EQ(dca_->peerIpQueue_.size(), 1u);
    EXPECT_EQ(dca_->peerIpQueue_.front().remoteRankId, 1u); // 首个 peer 生效
    GlobalMockObject::verify();
}

TEST_F(DetectConnectAnomalies950Test, Ut_EnqueuePeerIpList_When_ConcurrentSamePeer_Expect_SingleEntry)
{
    // 并发入队同一对端（无 mock 参与）：knownPeerIps_ 去重 + 队列锁保证只入队一次
    DcaPeerInfo peer = MakePeer(1, "1.1.1.1", ENDPOINT_LOC_TYPE_DEVICE, 0, "1.1.1.9");
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([this, &peer]() {
            (void)dca_->EnqueuePeerIpList(std::vector<DcaPeerInfo>{peer});
        });
    }
    for (auto& t : threads) {
        t.join();
    }
    EXPECT_EQ(dca_->knownPeerIps_.size(), 1u);
    EXPECT_EQ(dca_->peerIpQueue_.size(), 1u);
}

TEST_F(DetectConnectAnomalies950Test, Ut_EnqueueLocalIpList_When_ConcurrentSameIp_Expect_SingleEntry)
{
    // 并发入队同一本端 IP（无 mock 参与）：knownLocalIps_ 去重 + 队列锁保证只入队一次
    std::vector<hcomm::LocalIpInfo> localIps{MakeLocalIp("1.1.1.9", ENDPOINT_LOC_TYPE_DEVICE, 16666)};
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([this, &localIps]() {
            (void)dca_->EnqueueLocalIpList(localIps);
        });
    }
    for (auto& t : threads) {
        t.join();
    }
    EXPECT_EQ(dca_->knownLocalIps_.size(), 1u);
    EXPECT_EQ(dca_->localIpQueue_.size(), 1u);
}

// ==================================================================
// 6. CreateServerSocket / TryAcceptOne（per-IP 监听 socket + 连接 Socket 集合）★上板测不到
// ==================================================================
static Hccl::Socket* g_stubClientSocket = nullptr;

static HcclResult
StubGetSocketAssign(hcomm::SocketMgr* /*self*/, const Hccl::SocketConfig& /*socketConfig*/, Hccl::Socket*& socket)
{
    socket = g_stubClientSocket;
    return HCCL_SUCCESS;
}

static HcclResult StubClientConnectedFail(
    DetectConnectionAnomalies* /*self*/, const DcaPeerInfo& /*peer*/, Hccl::Socket*& socket, bool& connected)
{
    socket = g_stubClientSocket;
    connected = false;
    return HCCL_SUCCESS;
}

TEST_F(DetectConnectAnomalies950Test, Ut_ClientDetectThread_When_NotConnected_Expect_FailRecorded)
{
    PresetMixedHandles();
    DcaPeerInfo peer = MakePeer(1, "1.1.1.1", ENDPOINT_LOC_TYPE_DEVICE, 5, "1.1.1.1");
    Hccl::Socket sock(
        reinterpret_cast<Hccl::SocketHandle>(0x1), peer.localIp, peer.remotePort, peer.remoteIp, "1.1.1.1_detect_0",
        Hccl::SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    g_stubClientSocket = &sock;
    MOCKER_CPP(&DetectConnectionAnomalies::CreateClientSocketAndConnected)
        .stubs()
        .will(invoke(StubClientConnectedFail));
    MOCKER_CPP(&hcomm::SocketMgr::DestroySocket, HcclResult(hcomm::SocketMgr::*)(Hccl::Socket*))
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    dca_->threadRunning_ = false;
    EXPECT_EQ(dca_->ClientDetectThread(peer), HCCL_SUCCESS);
    ASSERT_EQ(dca_->recvErrorInfoMap_.size(), 1u);
    const DetectInfo& di = dca_->recvErrorInfoMap_.begin()->second;
    EXPECT_EQ(di.remoteDeviceId, 5);
    GlobalMockObject::verify();
}

TEST_F(DetectConnectAnomalies950Test, Ut_ClientDetectThread_When_ConnectedAndBroadcast_Expect_Recorded)
{
    // DETECT OK 全链路：线程内建 socket → 连接轮询成功 → 收广播解析入库（检视 C15/d#4）
    g_recvDetectInfo = MakeInfo("S1", 0, "S2", 1);
    PresetMixedHandles();
    DcaPeerInfo peer = MakePeer(1, "1.1.1.1", ENDPOINT_LOC_TYPE_DEVICE, 5, "1.1.1.1");
    Hccl::Socket sock(
        reinterpret_cast<Hccl::SocketHandle>(0x1), peer.localIp, peer.remotePort, peer.remoteIp, "1.1.1.1_detect_0",
        Hccl::SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    // socket 由 SocketMgr 统一创建：桩化 GetSocket 返回测试 socket，DestroySocket 直接成功
    g_stubClientSocket = &sock;
    MOCKER_CPP(&hcomm::SocketMgr::GetSocket).stubs().will(invoke(StubGetSocketAssign));
    MOCKER_CPP(&hcomm::SocketMgr::DestroySocket, HcclResult(hcomm::SocketMgr::*)(Hccl::Socket*))
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    // RA 层桩化：连接/查询/关闭/收包（避免真实驱动调用）
    MOCKER(HrtRaSocketConnectOne).stubs().will(ignoreReturnValue());
    MOCKER(HrtRaBlockGetOneSocket)
        .stubs()
        .will(returnValue(Hccl::RaSocketFdHandleParam(
            reinterpret_cast<Hccl::FdHandle>(0xA1), Hccl::SOCKET_CONNECTED, Hccl::IpAddress())));
    MOCKER(HrtRaSocketCloseOne).stubs().with(mockcpp::any());
    MOCKER(HrtRaSocketNonBlockRecvHeart).stubs().will(invoke(NonBlockRecvHeartFullStub));
    // 连接/广播窗口压到 1s：CreateClientSocketAndConnected 首轮 GetStatus 即 OK，随后 1s 后进入收广播
    MOCKER_CPP(&DetectConnectionAnomalies::GetDfsConnFaultTime).stubs().will(returnValue(1u));
    dca_->threadRunning_ = true;
    std::thread client([this, peer]() {
        (void)dca_->ClientDetectThread(peer);
    });
    for (int i = 0; i < 150; ++i) {
        {
            std::lock_guard<std::mutex> lk(dca_->recvSendErrInfoMutex_);
            if (!dca_->recvErrorInfoMap_.empty()) {
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    dca_->threadRunning_ = false;
    client.join();
    ASSERT_EQ(dca_->recvErrorInfoMap_.size(), 1u);
    const DetectInfo& di = dca_->recvErrorInfoMap_.begin()->second;
    EXPECT_STREQ(di.localNetInstId, "S1");
    EXPECT_STREQ(di.remoteNetInstId, "S2");
    GlobalMockObject::verify();
}

TEST_F(DetectConnectAnomalies950Test, Ut_CreateClientSocket_When_ClientIpInvalid_Expect_NotFound)
{
    DcaPeerInfo peer = MakePeer(1, "1.1.1.1", ENDPOINT_LOC_TYPE_DEVICE, 5, "1.1.1.1");
    peer.remoteIp = Hccl::IpAddress();
    Hccl::Socket* socket = nullptr;
    EXPECT_EQ(dca_->CreateClientSocket(peer, socket), HCCL_E_NOT_FOUND);
}

TEST_F(DetectConnectAnomalies950Test, Ut_CreateClientSocket_When_GetSocketReturnsNull_Expect_PtrError)
{
    g_stubClientSocket = nullptr;
    MOCKER_CPP(&hcomm::SocketMgr::GetSocket).stubs().will(invoke(StubGetSocketAssign));
    DcaPeerInfo peer = MakePeer(1, "1.1.1.1", ENDPOINT_LOC_TYPE_DEVICE, 5, "1.1.1.1");
    Hccl::Socket* socket = nullptr;
    EXPECT_EQ(dca_->CreateClientSocket(peer, socket), HCCL_E_PTR);
    GlobalMockObject::verify();
}

TEST_F(DetectConnectAnomalies950Test, Ut_ClientDetectThread_When_ClientSocketNull_Expect_PtrError)
{
    MOCKER_CPP(&DetectConnectionAnomalies::CreateClientSocketAndConnected).stubs().will(returnValue(HCCL_SUCCESS));
    dca_->threadRunning_ = false;
    DcaPeerInfo peer = MakePeer(1, "1.1.1.1", ENDPOINT_LOC_TYPE_DEVICE, 5, "1.1.1.1");
    EXPECT_EQ(dca_->ClientDetectThread(peer), HCCL_E_PTR);
    GlobalMockObject::verify();
}

TEST_F(
    DetectConnectAnomalies950Test,
    Ut_CreateClientSocketAndConnected_When_StatusNotReadyUntilWindowEnd_Expect_ConnectedFalse)
{
    // 真实建连轮询的未就绪路径：GetStatus 持续非 OK，窗口（dfs=1s）耗尽后返回 SUCCESS 且 connected=false
    DcaPeerInfo peer = MakePeer(1, "1.1.1.1", ENDPOINT_LOC_TYPE_DEVICE, 5, "1.1.1.1");
    Hccl::Socket sock(
        reinterpret_cast<Hccl::SocketHandle>(0x1), peer.localIp, peer.remotePort, peer.remoteIp, "1.1.1.1_detect_0",
        Hccl::SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    g_stubClientSocket = &sock;
    MOCKER_CPP(&hcomm::SocketMgr::GetSocket).stubs().will(invoke(StubGetSocketAssign));
    MOCKER(HrtRaBlockGetOneSocket)
        .stubs()
        .will(returnValue(Hccl::RaSocketFdHandleParam(
            reinterpret_cast<Hccl::FdHandle>(0xA2), Hccl::SOCKET_CONNECTING, Hccl::IpAddress())));
    MOCKER_CPP(&DetectConnectionAnomalies::GetDfsConnFaultTime).stubs().will(returnValue(1u));
    dca_->threadRunning_ = true;
    Hccl::Socket* clientSocket = nullptr;
    bool connected = true;
    EXPECT_EQ(dca_->CreateClientSocketAndConnected(peer, clientSocket, connected), HCCL_SUCCESS);
    dca_->threadRunning_ = false;
    EXPECT_NE(clientSocket, nullptr);
    EXPECT_FALSE(connected); // 窗口耗尽：未连接，交由 ClientDetectThread 记录故障根因
    GlobalMockObject::verify();
}

TEST_F(DetectConnectAnomalies950Test, Ut_ClientDetectThread_When_ConstructErrorInfoFails_Expect_SocketDestroyedAndError)
{
    // 早退收尾（检视 MEDIUM）：ConstructErrorInfo 失败返回 E_INTERNAL，但已创建的 clientSocket 必须被销毁
    DcaPeerInfo peer = MakePeer(1, "1.1.1.1", ENDPOINT_LOC_TYPE_DEVICE, 5, "1.1.1.1");
    Hccl::Socket sock(
        reinterpret_cast<Hccl::SocketHandle>(0x1), peer.localIp, peer.remotePort, peer.remoteIp, "1.1.1.1_detect_0",
        Hccl::SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    g_stubClientSocket = &sock;
    MOCKER_CPP(&DetectConnectionAnomalies::CreateClientSocketAndConnected)
        .stubs()
        .will(invoke(StubClientConnectedFail));
    MOCKER_CPP(&hcomm::SocketMgr::DestroySocket, HcclResult(hcomm::SocketMgr::*)(Hccl::Socket*))
        .expects(once())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER(strcpy_s).stubs().will(returnValue(-1));
    dca_->threadRunning_ = false;
    EXPECT_EQ(dca_->ClientDetectThread(peer), HCCL_E_INTERNAL);
    EXPECT_TRUE(dca_->recvErrorInfoMap_.empty());
    GlobalMockObject::verify();
}

TEST_F(DetectConnectAnomalies950Test, Ut_ConstructErrorInfo_When_DeviceHost_Expect_FillFields)
{
    // device/host 统一 devPhyId，IP 拷贝
    dca_->localDevPhyId_ = 3;
    dca_->localNetInstId_ = "local_net_inst";
    DcaPeerInfo peer = MakePeer(1, "1.1.1.1", ENDPOINT_LOC_TYPE_DEVICE, 5, "1.1.1.9");
    DetectInfo di{};
    EXPECT_EQ(dca_->ConstructErrorInfo(peer, di), HCCL_SUCCESS);
    EXPECT_EQ(di.localDeviceId, 3);
    EXPECT_EQ(di.remoteDeviceId, 5);
    EXPECT_STREQ(di.localIp, "1.1.1.9");
    EXPECT_STREQ(di.remoteIp, "1.1.1.1");
    // strcpy_s 失败 → 返回错误
    MOCKER(strcpy_s).stubs().will(returnValue(-1));
    EXPECT_NE(dca_->ConstructErrorInfo(peer, di), HCCL_SUCCESS);
    GlobalMockObject::verify();
}

// ==================================================================
// 10. 广播 / 聚合 / 上报
// ==================================================================
// 单个已 accept 的 Socket 广播完成的 stub：底层 NonBlockSendHeart 一次发完
static HcclResult NonBlockSendHeartDoneStub(Hccl::FdHandle, void*, u64 size, u64* sentSize)
{
    *sentSize = size;
    return HCCL_SUCCESS;
}

TEST_F(DetectConnectAnomalies950Test, Ut_RunBroadcastLoop_When_InfoQueued_Expect_SendOnce)
{
    // 真实调用 RunBroadcastLoop（不再名不副实，检视 o#20）：单条结果 + 单个 socket → 窗口内只发一次（isSent 去重）
    dca_->recvErrorInfoMap_["1.1.1.1-1.1.1.1"] = MakeInfo("0", 1, "1", 2);
    LocalIpInfo devInfo = MakeLocalIp("1.1.1.1", ENDPOINT_LOC_TYPE_DEVICE, DCA_DEVICE_DEFAULT_PORT);
    Hccl::Socket sock(
        reinterpret_cast<Hccl::SocketHandle>(0x1), devInfo.ip, devInfo.port, Hccl::IpAddress{}, "1.1.1.1_detect_0",
        Hccl::SocketRole::SERVER, Hccl::NicType::DEVICE_NIC_TYPE);
    std::vector<std::shared_ptr<Hccl::Socket>> sockets{BorrowSocket(&sock)};
    g_sendHeartCalls = 0;
    MOCKER(HrtRaSocketNonBlockSendHeart).stubs().will(invoke(NonBlockSendHeartCountDoneStub));
    dca_->threadRunning_ = true;
    std::thread stopper([this]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        dca_->threadRunning_ = false;
    });
    EXPECT_EQ(dca_->RunBroadcastLoop(sockets), HCCL_SUCCESS);
    stopper.join();
    EXPECT_EQ(g_sendHeartCalls, 1); // 已发送标记生效：窗口内不重复发送
    GlobalMockObject::verify();
}

TEST_F(DetectConnectAnomalies950Test, Ut_BroadcastToSockets_When_FirstEagain_Expect_RetryUntilDone)
{
    // 首次 EAGAIN 后成功：验证退避重试路径（检视 C15）
    LocalIpInfo devInfo = MakeLocalIp("1.1.1.1", ENDPOINT_LOC_TYPE_DEVICE, DCA_DEVICE_DEFAULT_PORT);
    Hccl::Socket sock(
        reinterpret_cast<Hccl::SocketHandle>(0x1), devInfo.ip, devInfo.port, Hccl::IpAddress{}, "1.1.1.1_detect_0",
        Hccl::SocketRole::SERVER, Hccl::NicType::DEVICE_NIC_TYPE);
    std::vector<std::shared_ptr<Hccl::Socket>> sockets{BorrowSocket(&sock)};
    g_sendHeartCalls = 0;
    MOCKER(HrtRaSocketNonBlockSendHeart).stubs().will(invoke(NonBlockSendHeartEagainThenDoneStub));
    EXPECT_EQ(dca_->BroadcastToSockets(MakeInfo("0", 1, "1", 2), sockets), HCCL_SUCCESS);
    EXPECT_EQ(g_sendHeartCalls, 2); // EAGAIN 一次 + 成功一次
    GlobalMockObject::verify();
}

TEST_F(DetectConnectAnomalies950Test, Ut_BroadcastToSockets_When_Empty_Expect_NoCrash)
{
    // 广播发送对端背压（E_AGAIN）路径的鲁棒性：ST 难造驱动背压，UT 验证退避逻辑不崩溃、循环有界。
    // 空 sockets：无接收方，直接返回 SUCCESS；E_AGAIN 语义由 BroadcastToSockets 内部退避 + SEND_BUDGET_MS 时间窗保障。
    DetectInfo info = MakeInfo("0", 1, "1", 2);
    std::vector<std::shared_ptr<Hccl::Socket>> sockets;
    EXPECT_EQ(dca_->BroadcastToSockets(info, sockets), HCCL_SUCCESS);
}

TEST_F(DetectConnectAnomalies950Test, Ut_BroadcastToSockets_When_SendSuccess_Expect_Success)
{
    // 单个已 accept 的 Socket：ISendWithHeart 一次发完（底层 NonBlockSendHeart mock 直接完成）→ 返回 SUCCESS
    LocalIpInfo devInfo = MakeLocalIp("1.1.1.1", ENDPOINT_LOC_TYPE_DEVICE, DCA_DEVICE_DEFAULT_PORT);
    Hccl::Socket sock(
        reinterpret_cast<Hccl::SocketHandle>(0x1), devInfo.ip, devInfo.port, Hccl::IpAddress{}, "1.1.1.1_detect_0",
        Hccl::SocketRole::SERVER, Hccl::NicType::DEVICE_NIC_TYPE);
    std::vector<std::shared_ptr<Hccl::Socket>> sockets{BorrowSocket(&sock)};
    MOCKER(HrtRaSocketNonBlockSendHeart).stubs().will(invoke(NonBlockSendHeartDoneStub));
    EXPECT_EQ(dca_->BroadcastToSockets(MakeInfo("0", 1, "1", 2), sockets), HCCL_SUCCESS);
    GlobalMockObject::verify();
}

// 硬错误发送桩：非 EAGAIN 失败一次（触发 BroadcastToSockets 的 break 分支）
static HcclResult NonBlockSendHeartHardErrorStub(Hccl::FdHandle, void*, u64, u64* sentSize)
{
    ++g_sendHeartCalls;
    *sentSize = 0;
    return HCCL_E_INTERNAL;
}

TEST_F(DetectConnectAnomalies950Test, Ut_BroadcastToSockets_When_SendHardError_Expect_AbortWithoutRetry)
{
    // 硬错误（非 EAGAIN）：当前 socket 立即放弃、不进入 10ms 退避重试
    LocalIpInfo devInfo = MakeLocalIp("1.1.1.1", ENDPOINT_LOC_TYPE_DEVICE, DCA_DEVICE_DEFAULT_PORT);
    Hccl::Socket sock(
        reinterpret_cast<Hccl::SocketHandle>(0x1), devInfo.ip, devInfo.port, Hccl::IpAddress{}, "1.1.1.1_detect_0",
        Hccl::SocketRole::SERVER, Hccl::NicType::DEVICE_NIC_TYPE);
    std::vector<std::shared_ptr<Hccl::Socket>> sockets{BorrowSocket(&sock)};
    g_sendHeartCalls = 0;
    MOCKER(HrtRaSocketNonBlockSendHeart).stubs().will(invoke(NonBlockSendHeartHardErrorStub));
    EXPECT_EQ(dca_->BroadcastToSockets(MakeInfo("0", 1, "1", 2), sockets), HCCL_SUCCESS);
    EXPECT_EQ(g_sendHeartCalls, 1); // 硬错误立即 break，无重试
    GlobalMockObject::verify();
}

TEST_F(DetectConnectAnomalies950Test, Ut_ReceiveDetectBroadcast_When_FullPacket_Expect_Recorded)
{
    // 收广播解析路径：一次收满 DetectInfo → 解析入库（检视 C15）
    g_recvDetectInfo = MakeInfo("S1", 0, "S2", 1);
    LocalIpInfo devInfo = MakeLocalIp("1.1.1.1", ENDPOINT_LOC_TYPE_DEVICE, DCA_DEVICE_DEFAULT_PORT);
    Hccl::Socket sock(
        reinterpret_cast<Hccl::SocketHandle>(0x1), devInfo.ip, devInfo.port, Hccl::IpAddress{}, "1.1.1.1_detect_0",
        Hccl::SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    MOCKER(HrtRaSocketNonBlockRecvHeart).stubs().will(invoke(NonBlockRecvHeartFullStub));
    dca_->threadRunning_ = true;
    std::thread receiver([this, &sock]() {
        (void)dca_->ReceiveDetectBroadcast(&sock);
    });
    for (int i = 0; i < 50; ++i) {
        {
            std::lock_guard<std::mutex> lk(dca_->recvSendErrInfoMutex_);
            if (!dca_->recvErrorInfoMap_.empty()) {
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    dca_->threadRunning_ = false;
    receiver.join();
    ASSERT_EQ(dca_->recvErrorInfoMap_.size(), 1u);
    const DetectInfo& di = dca_->recvErrorInfoMap_.begin()->second;
    EXPECT_STREQ(di.localNetInstId, "S1");
    EXPECT_STREQ(di.remoteNetInstId, "S2");
    EXPECT_EQ(di.localDeviceId, 0);
    EXPECT_EQ(di.remoteDeviceId, 1);
    GlobalMockObject::verify();
}

TEST_F(DetectConnectAnomalies950Test, Ut_ReceiveDetectBroadcast_When_PartialPackets_Expect_Recorded)
{
    // 半包累积解析：首次半包、二次补齐 → 完整 DetectInfo 入库（检视：收广播分片覆盖）
    g_recvDetectInfo = MakeInfo("S1", 0, "S2", 1);
    g_halfRecvCalls = 0;
    LocalIpInfo devInfo = MakeLocalIp("1.1.1.1", ENDPOINT_LOC_TYPE_DEVICE, DCA_DEVICE_DEFAULT_PORT);
    Hccl::Socket sock(
        reinterpret_cast<Hccl::SocketHandle>(0x1), devInfo.ip, devInfo.port, Hccl::IpAddress{}, "1.1.1.1_detect_0",
        Hccl::SocketRole::CLIENT, Hccl::NicType::DEVICE_NIC_TYPE);
    MOCKER(HrtRaSocketNonBlockRecvHeart).stubs().will(invoke(NonBlockRecvHeartHalfThenRestStub));
    dca_->threadRunning_ = true;
    std::thread receiver([this, &sock]() {
        (void)dca_->ReceiveDetectBroadcast(&sock);
    });
    for (int i = 0; i < 50; ++i) {
        {
            std::lock_guard<std::mutex> lk(dca_->recvSendErrInfoMutex_);
            if (!dca_->recvErrorInfoMap_.empty()) {
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    dca_->threadRunning_ = false;
    receiver.join();
    EXPECT_GE(g_halfRecvCalls, 2);
    ASSERT_EQ(dca_->recvErrorInfoMap_.size(), 1u);
    const DetectInfo& di = dca_->recvErrorInfoMap_.begin()->second;
    EXPECT_STREQ(di.localNetInstId, "S1");
    EXPECT_STREQ(di.remoteNetInstId, "S2");
    EXPECT_EQ(di.remoteDeviceId, 1);
}

TEST_F(DetectConnectAnomalies950Test, Ut_ReceiveDetectBroadcast_When_SocketNull_Expect_PtrError)
{
    // 空指针守卫：直接返回 E_PTR，不进入收包循环
    EXPECT_EQ(dca_->ReceiveDetectBroadcast(nullptr), HCCL_E_PTR);
}

TEST_F(DetectConnectAnomalies950Test, Ut_RunServerAcceptLoop_When_AcceptSuccess_Expect_SocketCollected)
{
    // accept 窗口内收到 1 条连接：加入 accepted 列表，其后 TIMEOUT 不新增（检视：accept 循环直测）
    LocalIpInfo info = MakeLocalIp("1.1.1.9", ENDPOINT_LOC_TYPE_DEVICE, DCA_DEVICE_DEFAULT_PORT);
    Hccl::Socket accepted(
        reinterpret_cast<Hccl::SocketHandle>(0x1), info.ip, info.port, Hccl::IpAddress{}, "1.1.1.9_detect_0",
        Hccl::SocketRole::SERVER, Hccl::NicType::DEVICE_NIC_TYPE);
    Hccl::Socket listener(
        reinterpret_cast<Hccl::SocketHandle>(0x2), info.ip, info.port, Hccl::IpAddress{}, "1.1.1.9_detect_0",
        Hccl::SocketRole::SERVER, Hccl::NicType::DEVICE_NIC_TYPE);
    g_stubListener = &listener;
    g_stubAcceptedSocket = &accepted;
    g_acceptCallCount = 0;
    MOCKER_CPP(&hcomm::ServerSocketMgr::GetServerSocket).stubs().will(invoke(StubGetServerSocket));
    MOCKER_CPP(&hcomm::SocketMgr::AcceptSocket).stubs().will(invoke(StubAcceptOnce));
    MOCKER_CPP(&DetectConnectionAnomalies::GetDfsConnFaultTime).stubs().will(returnValue(1u));
    dca_->threadRunning_ = true;
    dca_->startTime_ = std::chrono::steady_clock::now() - std::chrono::milliseconds(950);
    std::vector<std::shared_ptr<Hccl::Socket>> acceptedList;
    EXPECT_EQ(dca_->RunServerAcceptLoop(info, DCA_DEVICE_DEFAULT_PORT, &listener, acceptedList), HCCL_SUCCESS);
    dca_->threadRunning_ = false;
    ASSERT_EQ(acceptedList.size(), 1u);
    EXPECT_EQ(acceptedList[0].get(), &accepted);
    GlobalMockObject::verify();
}

TEST_F(DetectConnectAnomalies950Test, Ut_RunServerAcceptLoop_When_NoConnection_Expect_EmptyList)
{
    // accept 一直 TIMEOUT：窗口结束 accepted 列表为空，不产生假连接
    LocalIpInfo info = MakeLocalIp("1.1.1.9", ENDPOINT_LOC_TYPE_DEVICE, DCA_DEVICE_DEFAULT_PORT);
    Hccl::Socket listener(
        reinterpret_cast<Hccl::SocketHandle>(0x2), info.ip, info.port, Hccl::IpAddress{}, "1.1.1.9_detect_0",
        Hccl::SocketRole::SERVER, Hccl::NicType::DEVICE_NIC_TYPE);
    g_stubListener = &listener;
    g_stubAcceptedSocket = nullptr;
    g_acceptCallCount = 1;
    MOCKER_CPP(&hcomm::ServerSocketMgr::GetServerSocket).stubs().will(invoke(StubGetServerSocket));
    MOCKER_CPP(&hcomm::SocketMgr::AcceptSocket).stubs().will(invoke(StubAcceptOnce));
    MOCKER_CPP(&DetectConnectionAnomalies::GetDfsConnFaultTime).stubs().will(returnValue(1u));
    dca_->threadRunning_ = true;
    dca_->startTime_ = std::chrono::steady_clock::now() - std::chrono::milliseconds(950);
    std::vector<std::shared_ptr<Hccl::Socket>> acceptedList;
    EXPECT_EQ(dca_->RunServerAcceptLoop(info, DCA_DEVICE_DEFAULT_PORT, &listener, acceptedList), HCCL_SUCCESS);
    dca_->threadRunning_ = false;
    EXPECT_TRUE(acceptedList.empty());
    GlobalMockObject::verify();
}

TEST_F(DetectConnectAnomalies950Test, Ut_ServerDetectThread_When_AllStepsSuccess_Expect_OrderAndWhiteListDelivered)
{
    // ServerDetectThread 编排直测：StartListen→AddWhiteList→AcceptLoop→BroadcastLoop 次序 + 白名单下发（检视 D7）
    LocalIpInfo info = MakeLocalIp("1.1.1.10", ENDPOINT_LOC_TYPE_DEVICE, DCA_DEVICE_DEFAULT_PORT);
    Hccl::Socket listener(
        reinterpret_cast<Hccl::SocketHandle>(0x3), info.ip, info.port, Hccl::IpAddress{}, "1.1.1.10_detect_0",
        Hccl::SocketRole::SERVER, Hccl::NicType::DEVICE_NIC_TYPE);
    g_stubListener = &listener;
    g_serverDetectOrder.clear();
    g_whiteListAddCount = 0;
    g_startListenRet = HCCL_SUCCESS;
    MOCKER_CPP(&hcomm::ServerSocketMgr::ServerSocketStartListen).stubs().will(invoke(StubServerSocketStartListen));
    MOCKER_CPP(&hcomm::ServerSocketMgr::AddWhiteList).stubs().will(invoke(StubServerSocketAddWhiteList));
    MOCKER_CPP(&hcomm::ServerSocketMgr::GetServerSocket).stubs().will(invoke(StubGetServerSocket));
    MOCKER_CPP(&hcomm::ServerSocketMgr::DeleteWhiteList).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hcomm::ServerSocketMgr::ServerSocketStopListen).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&DetectConnectionAnomalies::RunServerAcceptLoop).stubs().will(invoke(StubRunServerAcceptLoopOrder));
    MOCKER_CPP(&DetectConnectionAnomalies::RunBroadcastLoop).stubs().will(invoke(StubRunBroadcastLoopOrder));

    EXPECT_EQ(dca_->ServerDetectThread(info), HCCL_SUCCESS);
    EXPECT_EQ(g_serverDetectOrder, std::vector<int>({1, 2, 3, 4}));
    EXPECT_EQ(g_whiteListAddCount, 1);
    GlobalMockObject::verify();
}

TEST_F(DetectConnectAnomalies950Test, Ut_ServerDetectThread_When_StartListenFails_Expect_NoWhiteListDelivered)
{
    // StartListen 失败短路：仅执行第 1 步，白名单不下发（使用计数桩断言，检视 D7）
    LocalIpInfo info = MakeLocalIp("1.1.1.11", ENDPOINT_LOC_TYPE_DEVICE, DCA_DEVICE_DEFAULT_PORT);
    g_serverDetectOrder.clear();
    g_whiteListAddCount = 0;
    g_startListenRet = HCCL_E_INTERNAL;
    MOCKER_CPP(&hcomm::ServerSocketMgr::ServerSocketStartListen).stubs().will(invoke(StubServerSocketStartListen));
    MOCKER_CPP(&hcomm::ServerSocketMgr::AddWhiteList).stubs().will(invoke(StubServerSocketAddWhiteList));

    EXPECT_EQ(dca_->ServerDetectThread(info), HCCL_E_INTERNAL);
    EXPECT_EQ(g_serverDetectOrder, std::vector<int>({1}));
    EXPECT_EQ(g_whiteListAddCount, 0);
    GlobalMockObject::verify();
}

TEST_F(DetectConnectAnomalies950Test, Ut_BuildLocalPortInfo_When_HostAndDeviceLoc_Expect_NicTypeAndPortTypeMatched)
{
    // 端口信息构造：host/device 各自选择 nicType 与 PortType，rankId/IP 透传
    dca_->localDevPhyId_ = 3;
    LocalIpInfo hostInfo = MakeLocalIp("1.1.1.20", ENDPOINT_LOC_TYPE_HOST, DCA_HOST_DEFAULT_PORT);
    LocalIpInfo devInfo = MakeLocalIp("1.1.1.21", ENDPOINT_LOC_TYPE_DEVICE, DCA_DEVICE_DEFAULT_PORT);

    auto [hostPort, hostNicType] = dca_->BuildLocalPortInfo(hostInfo);
    EXPECT_EQ(hostNicType, Hccl::NicType::HOST_NIC_TYPE);
    EXPECT_EQ(hostPort.GetRankId(), 3u);
    EXPECT_EQ(hostPort.GetAddr().GetIpStr(), "1.1.1.20");
    EXPECT_TRUE(hostPort == Hccl::PortData(3, Hccl::HostNetPortType(Hccl::ConnectProtoType::UB), 0, hostInfo.ip));

    auto [devPort, devNicType] = dca_->BuildLocalPortInfo(devInfo);
    EXPECT_EQ(devNicType, Hccl::NicType::DEVICE_NIC_TYPE);
    EXPECT_EQ(devPort.GetRankId(), 3u);
    EXPECT_EQ(devPort.GetAddr().GetIpStr(), "1.1.1.21");
    EXPECT_TRUE(devPort == Hccl::PortData(3, Hccl::DevNetPortType(Hccl::ConnectProtoType::UB), 0, devInfo.ip));
}

TEST_F(DetectConnectAnomalies950Test, Ut_BuildLocalPortInfo_When_HccsAndPcieDevice_Expect_P2PPortType)
{
    // HCCS/PCIE 为设备间P2P直连，端口类型应为P2P，不得走DevNetPortType抛异常
    dca_->localDevPhyId_ = 3;
    LocalIpInfo pcieInfo{};
    pcieInfo.ip = Hccl::IpAddress("1.1.1.22");
    pcieInfo.port = DCA_DEVICE_DEFAULT_PORT;
    pcieInfo.endpoint = MakeEndpointDesc("1.1.1.22", ENDPOINT_LOC_TYPE_DEVICE, COMM_PROTOCOL_PCIE);
    LocalIpInfo hccsInfo{};
    hccsInfo.ip = Hccl::IpAddress("1.1.1.23");
    hccsInfo.port = DCA_DEVICE_DEFAULT_PORT;
    hccsInfo.endpoint = MakeEndpointDesc("1.1.1.23", ENDPOINT_LOC_TYPE_DEVICE, COMM_PROTOCOL_HCCS);

    auto [pciePort, pcieNicType] = dca_->BuildLocalPortInfo(pcieInfo);
    EXPECT_EQ(pcieNicType, Hccl::NicType::DEVICE_NIC_TYPE);
    EXPECT_TRUE(pciePort == Hccl::PortData(3, Hccl::P2PPortType(Hccl::ConnectProtoType::PCIE), 0, pcieInfo.ip));

    auto [hccsPort, hccsNicType] = dca_->BuildLocalPortInfo(hccsInfo);
    EXPECT_EQ(hccsNicType, Hccl::NicType::DEVICE_NIC_TYPE);
    EXPECT_TRUE(hccsPort == Hccl::PortData(3, Hccl::P2PPortType(Hccl::ConnectProtoType::HCCS), 0, hccsInfo.ip));
}

TEST_F(DetectConnectAnomalies950Test, Ut_ServerDetectThread_When_PreAcceptStepFails_Expect_TeardownPerState)
{
    // 收尾改造分档（3 轮）：AddWhiteList 失败 / GetServerSocket 失败 / listener 为空
    LocalIpInfo info = MakeLocalIp("1.1.1.12", ENDPOINT_LOC_TYPE_DEVICE, DCA_DEVICE_DEFAULT_PORT);
    Hccl::Socket listener(
        reinterpret_cast<Hccl::SocketHandle>(0x4), info.ip, info.port, Hccl::IpAddress{}, "1.1.1.12_detect_0",
        Hccl::SocketRole::SERVER, Hccl::NicType::DEVICE_NIC_TYPE);
    MOCKER_CPP(&hcomm::ServerSocketMgr::ServerSocketStartListen).stubs().will(invoke(StubServerSocketStartListen));
    MOCKER_CPP(&hcomm::ServerSocketMgr::AddWhiteList).stubs().will(invoke(StubServerSocketAddWhiteList));
    MOCKER_CPP(&hcomm::ServerSocketMgr::GetServerSocket).stubs().will(invoke(StubGetServerSocket));
    MOCKER_CPP(&hcomm::ServerSocketMgr::DeleteWhiteList).stubs().will(invoke(StubServerSocketDeleteWhiteList));
    MOCKER_CPP(&hcomm::ServerSocketMgr::ServerSocketStopListen).stubs().will(invoke(StubServerSocketStopListen));

    // 轮1：AddWhiteList 失败 → 未下发过白名单，收尾只停监听
    g_serverDetectOrder.clear();
    g_startListenRet = HCCL_SUCCESS;
    g_whiteListAddRet = HCCL_E_INTERNAL;
    g_getServerSocketRet = HCCL_SUCCESS;
    g_stubListener = &listener;
    g_deleteWhiteListCount = 0;
    g_stopListenCount = 0;
    EXPECT_EQ(dca_->ServerDetectThread(info), HCCL_E_INTERNAL);
    EXPECT_EQ(g_serverDetectOrder, std::vector<int>({1, 2}));
    EXPECT_EQ(g_deleteWhiteListCount, 0);
    EXPECT_EQ(g_stopListenCount, 1);

    // 轮2：GetServerSocket 失败 → 白名单已下发须回收，并停监听
    g_serverDetectOrder.clear();
    g_whiteListAddRet = HCCL_SUCCESS;
    g_getServerSocketRet = HCCL_E_NOT_FOUND;
    g_deleteWhiteListCount = 0;
    g_stopListenCount = 0;
    EXPECT_EQ(dca_->ServerDetectThread(info), HCCL_E_NOT_FOUND);
    EXPECT_EQ(g_serverDetectOrder, std::vector<int>({1, 2}));
    EXPECT_EQ(g_deleteWhiteListCount, 1);
    EXPECT_EQ(g_stopListenCount, 1);

    // 轮3：GetServerSocket 成功但 listener 为空 → E_PTR 短路，收尾同样完整
    g_serverDetectOrder.clear();
    g_getServerSocketRet = HCCL_SUCCESS;
    g_stubListener = nullptr;
    g_deleteWhiteListCount = 0;
    g_stopListenCount = 0;
    EXPECT_EQ(dca_->ServerDetectThread(info), HCCL_E_PTR);
    EXPECT_EQ(g_deleteWhiteListCount, 1);
    EXPECT_EQ(g_stopListenCount, 1);
    GlobalMockObject::verify();
}

TEST_F(DetectConnectAnomalies950Test, Ut_ServerDetectThread_When_LoopStepFails_Expect_TeardownAndSkipBroadcast)
{
    // 循环步骤失败（2 轮）：AcceptLoop 失败跳过 BroadcastLoop / BroadcastLoop 失败仍走收尾
    LocalIpInfo info = MakeLocalIp("1.1.1.13", ENDPOINT_LOC_TYPE_DEVICE, DCA_DEVICE_DEFAULT_PORT);
    Hccl::Socket listener(
        reinterpret_cast<Hccl::SocketHandle>(0x5), info.ip, info.port, Hccl::IpAddress{}, "1.1.1.13_detect_0",
        Hccl::SocketRole::SERVER, Hccl::NicType::DEVICE_NIC_TYPE);
    g_stubListener = &listener;
    MOCKER_CPP(&hcomm::ServerSocketMgr::ServerSocketStartListen).stubs().will(invoke(StubServerSocketStartListen));
    MOCKER_CPP(&hcomm::ServerSocketMgr::AddWhiteList).stubs().will(invoke(StubServerSocketAddWhiteList));
    MOCKER_CPP(&hcomm::ServerSocketMgr::GetServerSocket).stubs().will(invoke(StubGetServerSocket));
    MOCKER_CPP(&hcomm::ServerSocketMgr::DeleteWhiteList).stubs().will(invoke(StubServerSocketDeleteWhiteList));
    MOCKER_CPP(&hcomm::ServerSocketMgr::ServerSocketStopListen).stubs().will(invoke(StubServerSocketStopListen));
    MOCKER_CPP(&DetectConnectionAnomalies::RunServerAcceptLoop).stubs().will(invoke(StubRunServerAcceptLoopOrder));
    MOCKER_CPP(&DetectConnectionAnomalies::RunBroadcastLoop).stubs().will(invoke(StubRunBroadcastLoopOrder));

    // 轮1：AcceptLoop 失败 → BroadcastLoop 不执行，返回错误
    g_serverDetectOrder.clear();
    g_startListenRet = HCCL_SUCCESS;
    g_whiteListAddRet = HCCL_SUCCESS;
    g_getServerSocketRet = HCCL_SUCCESS;
    g_acceptLoopRet = HCCL_E_INTERNAL;
    g_broadcastLoopRet = HCCL_SUCCESS;
    g_deleteWhiteListCount = 0;
    g_stopListenCount = 0;
    EXPECT_EQ(dca_->ServerDetectThread(info), HCCL_E_INTERNAL);
    EXPECT_EQ(g_serverDetectOrder, std::vector<int>({1, 2, 3}));
    EXPECT_EQ(g_deleteWhiteListCount, 1);
    EXPECT_EQ(g_stopListenCount, 1);

    // 轮2：BroadcastLoop 失败 → 返回错误，收尾完整
    g_serverDetectOrder.clear();
    g_acceptLoopRet = HCCL_SUCCESS;
    g_broadcastLoopRet = HCCL_E_TIMEOUT;
    g_deleteWhiteListCount = 0;
    g_stopListenCount = 0;
    EXPECT_EQ(dca_->ServerDetectThread(info), HCCL_E_TIMEOUT);
    EXPECT_EQ(g_serverDetectOrder, std::vector<int>({1, 2, 3, 4}));
    EXPECT_EQ(g_deleteWhiteListCount, 1);
    EXPECT_EQ(g_stopListenCount, 1);
    GlobalMockObject::verify();
}

TEST_F(DetectConnectAnomalies950Test, Ut_BuildGroupedDetectMessage_When_MultiRoot_Expect_Grouping)
{
    // 多根按 (netInstId, deviceId) 分组，输出 device ID 列表（返回 vector<string>）
    // "This node" 用本机 localNetInstId_ + localDevPhyId_，需预置
    dca_->localDevPhyId_ = 0;
    dca_->localNetInstId_ = "S1";
    std::unordered_map<std::string, DetectInfo> snapshot;
    DetectInfo a{};
    snprintf_s(a.localIp, DEST_MAX_LEN, DEST_MAX_LEN, "10.0.0.1");
    snprintf_s(a.remoteIp, DEST_MAX_LEN, DEST_MAX_LEN, "10.0.0.2");
    a.localDeviceId = 0;
    a.remoteDeviceId = 1;
    FillServerId(a.localNetInstId, sizeof(a.localNetInstId), "S1");
    FillServerId(a.remoteNetInstId, sizeof(a.remoteNetInstId), "S2");
    snapshot["10.0.0.1-10.0.0.2"] = a;
    DetectInfo b{};
    snprintf_s(b.localIp, DEST_MAX_LEN, DEST_MAX_LEN, "10.0.0.1");
    snprintf_s(b.remoteIp, DEST_MAX_LEN, DEST_MAX_LEN, "10.0.0.3");
    b.localDeviceId = 0;
    b.remoteDeviceId = 2;
    FillServerId(b.localNetInstId, sizeof(b.localNetInstId), "S1");
    FillServerId(b.remoteNetInstId, sizeof(b.remoteNetInstId), "S2");
    snapshot["10.0.0.1-10.0.0.3"] = b;
    std::vector<std::string> msgLines = dca_->BuildGroupedDetectMessage(snapshot);
    ASSERT_FALSE(msgLines.empty());
    std::string msg = msgLines[0];
    EXPECT_NE(msg.find("[1,2]"), std::string::npos);
    EXPECT_NE(msg.find("This node (server S1"), std::string::npos);
    EXPECT_NE(msg.find("srcRank (server S1"), std::string::npos);
    EXPECT_NE(msg.find("device ID 0"), std::string::npos);
}

TEST_F(DetectConnectAnomalies950Test, Ut_ProcessDetectionResults_When_States_Expect_BranchSuccess)
{
    // 无事件：不置 isPrint_
    dca_->recvErrorInfoMap_.clear();
    EXPECT_EQ(dca_->ProcessDetectionResults(), HCCL_SUCCESS);
    EXPECT_FALSE(dca_->isPrint_.load());
    // 有事件
    dca_->recvErrorInfoMap_["k1"] = MakeInfo("0", 1, "1", 2);
    EXPECT_EQ(dca_->ProcessDetectionResults(), HCCL_SUCCESS);
}

TEST_F(DetectConnectAnomalies950Test, Ut_WaitForDetect_When_PrintOnceAndKeepStartTime_Expect_Flag)
{
    dca_->threadRunning_ = false;
    // 首次：isPrint_ 置位（上报一次）
    dca_->isPrint_ = false;
    EXPECT_EQ(dca_->WaitForDetect(), HCCL_SUCCESS);
    EXPECT_TRUE(dca_->isPrint_.load());
    // startTime_ 刷新由 AddIpQueue 负责；WaitForDetect 入口只读、不改变 startTime_
    const auto preset = std::chrono::steady_clock::now();
    dca_->startTime_ = preset;
    EXPECT_EQ(dca_->WaitForDetect(), HCCL_SUCCESS);
    EXPECT_EQ(dca_->startTime_.time_since_epoch().count(), preset.time_since_epoch().count());
    // 已上报：保持不重复上报
    dca_->isPrint_ = true;
    EXPECT_EQ(dca_->WaitForDetect(), HCCL_SUCCESS);
    EXPECT_TRUE(dca_->isPrint_.load());
}

TEST_F(DetectConnectAnomalies950Test, Ut_WaitForDetect_When_ThreadExitDuringWait_Expect_RealWaitLoop)
{
    // 不再用 threadRunning_=false 短路：真实进入等待循环，250ms 后置 false 退出（检视 o#8）
    dca_->isPrint_ = false;
    dca_->threadRunning_ = true;
    std::thread stopper([this]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        dca_->threadRunning_ = false;
    });
    auto t0 = std::chrono::steady_clock::now();
    EXPECT_EQ(dca_->WaitForDetect(), HCCL_SUCCESS);
    stopper.join();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    EXPECT_GE(ms, 200);                 // 确认真实等待（循环 100ms 粒度）
    EXPECT_LT(ms, 5000);                // 且不挂死
    EXPECT_TRUE(dca_->isPrint_.load()); // 退出时上报一次
}

TEST_F(DetectConnectAnomalies950Test, Ut_WaitForDetect_When_StartTimeRefreshed_Expect_WaitExtended)
{
    // 等待期间持续 RefreshStartTime → 窗口被延长；刷新停止并置 threadRunning_=false 后才退出（检视 o#9）
    dca_->isPrint_ = true; // 跳过上报
    dca_->threadRunning_ = true;
    std::thread refresher([this]() {
        for (int i = 0; i < 8; ++i) {
            {
                std::lock_guard<std::mutex> lk(dca_->timeMutex_);
                dca_->startTime_ = std::chrono::steady_clock::now(); // 阶段刷新（A3 语义）
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        dca_->threadRunning_ = false;
    });
    auto t0 = std::chrono::steady_clock::now();
    EXPECT_EQ(dca_->WaitForDetect(), HCCL_SUCCESS);
    refresher.join();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    EXPECT_GE(ms, 700); // 刷新持续期间未退出，验证 startTime_ 刷新生效
    EXPECT_LT(ms, 5000);
}

// ==================================================================
// 11. ServerDetectThread / DetectMonitor / GetPeerIpQueue（线程基础）
// ==================================================================
TEST_F(DetectConnectAnomalies950Test, Ut_DetectMonitor_When_ThreadExitFalse_Expect_ReturnImmediately)
{
    dca_->threadRunning_ = false;
    const auto t0 = std::chrono::steady_clock::now();
    dca_->DetectMonitor();
    const auto cost = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - t0).count();
    EXPECT_LT(cost, 2);
}

TEST_F(DetectConnectAnomalies950Test, Ut_DetectMonitor_When_QueuesNonEmpty_Expect_DispatchOnce)
{
    // DetectMonitor 单次迭代消费两个队列并分发线程（检视 o#29）：线程体桩化避免真实 socket/RA 依赖
    MOCKER_CPP(&DetectConnectionAnomalies::ServerDetectThread).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&DetectConnectionAnomalies::ClientDetectThread).stubs().will(returnValue(HCCL_SUCCESS));
    Hccl::Socket* fakeClientSocket = reinterpret_cast<Hccl::Socket*>(0x1);
    MOCKER_CPP(&hcomm::SocketMgr::GetSocket)
        .stubs()
        .with(mockcpp::any(), mockcpp::outBound(fakeClientSocket))
        .will(returnValue(HCCL_SUCCESS));
    PresetMixedHandles();
    std::vector<hcomm::LocalIpInfo> localIps{MakeLocalIp("1.1.1.9", ENDPOINT_LOC_TYPE_DEVICE, DCA_DEVICE_DEFAULT_PORT)};
    EXPECT_EQ(dca_->EnqueueLocalIpList(localIps), HCCL_SUCCESS);
    EXPECT_EQ(
        dca_->EnqueuePeerIpList(
            std::vector<DcaPeerInfo>{MakePeer(1, "1.1.1.1", ENDPOINT_LOC_TYPE_DEVICE, 5, "1.1.1.1")}),
        HCCL_SUCCESS);
    dca_->threadRunning_ = true;
    std::thread monitor([this]() {
        dca_->DetectMonitor();
    });
    for (int i = 0; i < 50 && (!dca_->localIpQueue_.empty() || !dca_->peerIpQueue_.empty()); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    dca_->threadRunning_ = false;
    monitor.join();
    EXPECT_TRUE(dca_->localIpQueue_.empty());
    EXPECT_TRUE(dca_->peerIpQueue_.empty());
    EXPECT_EQ(dca_->serverThreads_.size(), 1u);
    EXPECT_EQ(dca_->clientThreads_.size(), 1u);
    JoinAllThreads();
    GlobalMockObject::verify();
}

TEST_F(DetectConnectAnomalies950Test, Ut_GetPeerIpQueue_When_ThreadExited_Expect_NoConsume)
{
    // 队列空：返回成功
    dca_->threadRunning_ = false;
    EXPECT_EQ(dca_->GetPeerIpQueue(), HCCL_SUCCESS);
    // 线程退出态（threadRunning_=false）：不消费
    dca_->peerIpQueue_.push(MakePeer(1, "1.1.1.1", ENDPOINT_LOC_TYPE_DEVICE, 5));
    EXPECT_EQ(dca_->GetPeerIpQueue(), HCCL_SUCCESS);
    EXPECT_EQ(dca_->peerIpQueue_.size(), 1u);
}

TEST_F(DetectConnectAnomalies950Test, Ut_GetPeerIpQueue_When_QueueNonEmpty_Expect_Drain)
{
    // 非空队列：排空（含非法 IP）
    dca_->threadRunning_ = true;
    dca_->peerIpQueue_.push(MakePeer(1, "1.1.1.1", ENDPOINT_LOC_TYPE_DEVICE, 5));
    dca_->peerIpQueue_.push(MakePeer(2, "2.2.2.1", ENDPOINT_LOC_TYPE_HOST, 5));
    HcclResult ret = dca_->GetPeerIpQueue();
    dca_->threadRunning_ = false;
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(dca_->peerIpQueue_.empty());
}

// ==================================================================
// 12. Detect / GetLocalIpQueue（本端 IP 队列：每 localIp 一条 server 线程）
// ==================================================================
TEST_F(DetectConnectAnomalies950Test, Ut_GetLocalIpQueue_When_Enqueued_Expect_PerIpThread)
{
    // Detect：首次起 monitor，再调不新起
    (void)dca_->Detect();
    EXPECT_NE(dca_->monitorThread_, nullptr);
    auto* monitor = dca_->monitorThread_.get();
    (void)dca_->Detect();
    EXPECT_EQ(dca_->monitorThread_.get(), monitor);
    // EnqueueLocalIpList + GetLocalIpQueue：逐 IP 起线程，幂等不重复起
    std::vector<hcomm::LocalIpInfo> localIps{
        MakeLocalIp("1.1.1.9", ENDPOINT_LOC_TYPE_DEVICE, DCA_DEVICE_DEFAULT_PORT),
        MakeLocalIp("2.2.2.9", ENDPOINT_LOC_TYPE_HOST, DCA_HOST_DEFAULT_PORT)};
    EXPECT_EQ(dca_->EnqueueLocalIpList(localIps), HCCL_SUCCESS);
    // 消费循环需 threadRunning_ 放行
    dca_->threadRunning_ = true;
    EXPECT_EQ(dca_->GetLocalIpQueue(), HCCL_SUCCESS);
    dca_->threadRunning_ = false;
    EXPECT_EQ(dca_->serverThreads_.size(), 2u);
    EXPECT_EQ(dca_->knownLocalIps_.size(), 2u);
    // 幂等：重复入队同一批 IP（全部已 known）→ 不新增队列项、不新增线程
    EXPECT_EQ(dca_->EnqueueLocalIpList(localIps), HCCL_SUCCESS);
    EXPECT_TRUE(dca_->localIpQueue_.empty());
    dca_->threadRunning_ = true;
    EXPECT_EQ(dca_->GetLocalIpQueue(), HCCL_SUCCESS);
    dca_->threadRunning_ = false;
    EXPECT_EQ(dca_->serverThreads_.size(), 2u);
    EXPECT_EQ(dca_->knownLocalIps_.size(), 2u);
}

// ==================================================================
// 13. AddIpQueue：server/client 两侧独立入队（对端去重不影响新增本端 IP）
// ==================================================================
TEST_F(DetectConnectAnomalies950Test, Ut_AddIpQueue_When_PeerDeduped_Expect_NewLocalIpEnqueued)
{
    // 本端/对端两侧独立入队：本轮对端 rank 已被 knownPeerIps_ 去重（peerIpQueue_ 无新条目），
    // 新增本端 IP 仍要入 localIpQueue_（server 侧不被 client 侧去重影响）
    PrepareAddIpQueue();
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(static_cast<Hccl::DevType>(Hccl::DevType::DEV_TYPE_950)));
    MOCKER(HrtRaSocketInit).stubs().will(returnValue(reinterpret_cast<SocketHandle>(0x1)));
    // 预置：A 已入队并消费（server 线程无 handle 秒退）
    std::vector<hcomm::LocalIpInfo> firstIps{MakeLocalIp("1.1.1.9", ENDPOINT_LOC_TYPE_DEVICE, 16666)};
    EXPECT_EQ(dca_->EnqueueLocalIpList(firstIps), HCCL_SUCCESS);
    EXPECT_EQ(dca_->GetLocalIpQueue(), HCCL_SUCCESS);
    EXPECT_EQ(dca_->knownLocalIps_.size(), 1u);
    // 第一轮 AddIpQueue：A 已 known 不入队；peer 入队
    DcaPeerInfo peer = MakePeer(1, "1.1.1.1", ENDPOINT_LOC_TYPE_DEVICE, 0, "1.1.1.9");
    dca_->AddIpQueue(-1, 0, "local_net_inst", std::vector<DcaPeerInfo>{peer}, firstIps);
    EXPECT_EQ(dca_->peerIpQueue_.size(), 1u);
    // 第二轮 AddIpQueue：同 rank peer 被去重（对端无新条目）；新增 IP B 仍入本端队列
    std::vector<hcomm::LocalIpInfo> secondIps{
        MakeLocalIp("1.1.1.9", ENDPOINT_LOC_TYPE_DEVICE, 16666),
        MakeLocalIp("1.1.1.10", ENDPOINT_LOC_TYPE_DEVICE, 16666)};
    dca_->AddIpQueue(-1, 0, "local_net_inst", std::vector<DcaPeerInfo>{peer}, secondIps);
    EXPECT_EQ(dca_->knownPeerIps_.size(), 1u);
    EXPECT_EQ(dca_->peerIpQueue_.size(), 1u);  // 对端无新增（仍是第一轮那条）
    EXPECT_EQ(dca_->localIpQueue_.size(), 1u); // 仅新增 B
    EXPECT_EQ(dca_->localIpQueue_.front().ip.GetIpStr(), "1.1.1.10");
    EXPECT_EQ(dca_->knownLocalIps_.size(), 2u); // A + B
}
