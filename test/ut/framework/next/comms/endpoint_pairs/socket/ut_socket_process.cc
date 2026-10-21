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
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>
#include <stdexcept>

#include "../../../ut_hcomm_base.h"
#define private public
#include "socket_process.h"
#undef private
#include "log.h"
#include "hccl_comm_socket_c_adpt.h"
#include "sal_pub.h"
#include "hccl_types.h"

using namespace hcomm;

class SocketProcessTest : public TestHcommCAdptBase {
public:
    SocketProcessTest() { InitSocketDesc(); }
    ~SocketProcessTest()
    {
        if (socketHandle != nullptr) {
            hcomm::SocketProcess::GetInstance(0).DestroySocketHandle(socketHandle);
            socketHandle = nullptr;
        }
    }
    void SetUp() override { TestHcommCAdptBase::SetUp(); }
    void TearDown() override
    {
        TestHcommCAdptBase::TearDown();
        GlobalMockObject::verify();
    }

    void InitSocketDesc()
    {
        // 初始化socketDesc
        string testTag = "socket_test_tag";
        struct in_addr localAddr;
        struct in_addr remoteAddr;
        inet_pton(AF_INET, "127.0.0.1", &localAddr);
        SocketProcessTest::socketDesc.localEndpoint.commAddr.addr = localAddr;
        SocketProcessTest::socketDesc.localEndpoint.protocol = CommProtocol::COMM_PROTOCOL_ROCE;
        inet_pton(AF_INET, "192.186.0.1", &remoteAddr);
        SocketProcessTest::socketDesc.remoteEndpoint.commAddr.addr = remoteAddr;
        SocketProcessTest::socketDesc.remoteEndpoint.protocol = CommProtocol::COMM_PROTOCOL_ROCE;
        s32 ret
            = memcpy_s(SocketProcessTest::socketDesc.tag, HCCL_SOCKET_TAG_LEN, testTag.c_str(), testTag.length() + 1);
        EXPECT_EQ(ret, EOK);
        SocketProcessTest::socketDesc.role = HCOMM_SOCKET_ROLE_CLIENT;
        SocketProcessTest::socketDesc.listenPort = 8080;
    }

    static SocketHandle socketHandle;
    static SocketDesc socketDesc;
};

SocketHandle SocketProcessTest::socketHandle = nullptr;
SocketDesc SocketProcessTest::socketDesc{};

namespace {
constexpr uint32_t SOCKET_MGR_TEMP_LISTEN_PORT = 60001U;

Hccl::SocketConfig BuildSocketMgrConfig(Hccl::PortDeploymentType portType)
{
    Hccl::BasePortType basePortType(portType, Hccl::ConnectProtoType::RDMA);
    Hccl::LinkData linkData(basePortType, 0, 1, 0, 1);
    return Hccl::SocketConfig(linkData, "UT_SOCKET_MGR_INIT", true);
}

HcclResult StubSocketMgrHrtGetDevice(s32* deviceLogicId)
{
    *deviceLogicId = 3;
    return HCCL_SUCCESS;
}

HcclResult StubSocketMgrHrtGetDevicePhyIdByUserDevId(u32 deviceLogicId, u32& devicePhyId)
{
    (void)deviceLogicId;
    devicePhyId = 7;
    return HCCL_SUCCESS;
}

HcclResult StubSocketMgrHrtGetDeviceCountNoDevice(u32* deviceCount)
{
    *deviceCount = 0;
    return HCCL_SUCCESS;
}

HcclResult StubSocketMgrHrtGetDeviceCountWithDevice(u32* deviceCount)
{
    *deviceCount = 1;
    return HCCL_SUCCESS;
}

void ResetSocketMgrForTest(SocketMgr& socketMgr, uint32_t devicePhyId = 0)
{
    socketMgr.isLoaded_ = false;
    socketMgr.isHostOnlyInit_ = false;
    socketMgr.devicePhyId_ = devicePhyId;
    socketMgr.serverListenPort_ = 0;
    socketMgr.socketMap_.clear();
    socketMgr.handle2WhiteListMap_.clear();
    socketMgr.socketInUseMap_.clear();
}

// 构造一个仅用于引用计数测试的 SocketConfig（不同 tag 即不同 key，避免相互影响）
Hccl::SocketConfig BuildRefCountConfig(const std::string& tag)
{
    Hccl::LinkData linkData = BuildDefaultLinkData();
    return Hccl::SocketConfig(linkData, tag, true);
}

// 绕过真实建链（GetSocketHandle/Connect 依赖设备环境），直接向 socketMap_ 预置一个 SocketEntry，
// 聚焦验证引用计数的增减与销毁逻辑。预置的 Socket 未 Listen/Connect，Destroy 为无副作用空操作。
Hccl::Socket* InsertSocketEntryForRefCount(SocketMgr& socketMgr, const Hccl::SocketConfig& config, uint32_t refCount)
{
    auto socket = std::make_unique<Hccl::Socket>(
        reinterpret_cast<Hccl::SocketHandle>(0x1), Hccl::IpAddress(), 0, Hccl::IpAddress(), "UT_REF_COUNT",
        Hccl::SocketRole::CLIENT, Hccl::NicType::HOST_NIC_TYPE);
    Hccl::Socket* rawSocket = socket.get();
    auto& entry = socketMgr.socketMap_[config];
    entry.socket = std::move(socket);
    entry.refCount = refCount;
    socketMgr.socketInUseMap_[rawSocket] = false;
    return rawSocket;
}
} // namespace

TEST_F(SocketProcessTest, Ut_GetSocket_When_NullptrInput_Expect_ReturnError)
{
    SocketDesc* tempSocketDesc = nullptr;
    SocketHandle tempSocketHandle = nullptr;
    HcclResult ret = hcomm::SocketProcess::GetInstance(0).GetSocket(tempSocketDesc, tempSocketHandle);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(SocketProcessTest, Ut_GetSocket_When_NormalInput_Expect_GetSocketHandle)
{
    HcclResult ret = hcomm::SocketProcess::GetInstance(0).GetSocket(
        &SocketProcessTest::socketDesc, SocketProcessTest::socketHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(SocketProcessTest, Ut_GetStatus_When_InvalidInput_Expect_ReturnError)
{
    SocketHandle tempSocketHandle = nullptr;
    SocketStates socketStatus;
    HcclResult ret = hcomm::SocketProcess::GetInstance(0).GetStatus(tempSocketHandle, socketStatus);
    EXPECT_EQ(ret, HCCL_E_PARA);

    tempSocketHandle = reinterpret_cast<SocketHandle>(0x1);
    ret = hcomm::SocketProcess::GetInstance(0).GetStatus(tempSocketHandle, socketStatus);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(SocketProcessTest, Ut_GetStatus_When_NormalInput_Expect_GetSocketStatus)
{
    HcclResult ret;
    SocketStates socketStatus;
    if (SocketProcessTest::socketHandle == nullptr) {
        ret = hcomm::SocketProcess::GetInstance(0).GetSocket(
            &SocketProcessTest::socketDesc, SocketProcessTest::socketHandle);
        EXPECT_EQ(ret, HCCL_SUCCESS);
    }
    while (SocketProcessTest::socketHandle != nullptr && socketStatus != SocketStates::SOCKET_OK) {
        HcclResult ret = hcomm::SocketProcess::GetInstance(0).GetStatus(SocketProcessTest::socketHandle, socketStatus);
        EXPECT_EQ(ret, HCCL_SUCCESS);
        if (socketStatus == SocketStates::SOCKET_TIMEOUT) {
            EXPECT_EQ(socketStatus, SocketStates::SOCKET_OK);
            break;
        }
    }
}

TEST_F(SocketProcessTest, Ut_SendNoBlock_When_InvalidInput_Expect_ReturnError)
{
    SocketHandle tempSocketHandle = nullptr;
    u64 sendbuffer = 123;
    u64 sendSize = sizeof(sendbuffer);
    u64* sentSize = nullptr;
    HcclResult ret
        = hcomm::SocketProcess::GetInstance(0).SendNoBlock(tempSocketHandle, &sendbuffer, sendSize, sentSize);
    EXPECT_EQ(ret, HCCL_E_PARA);

    tempSocketHandle = reinterpret_cast<SocketHandle>(0x1);
    ret = hcomm::SocketProcess::GetInstance(0).SendNoBlock(tempSocketHandle, &sendbuffer, sendSize, sentSize);
    EXPECT_EQ(ret, HCCL_E_PARA);

    void* errorBuffer = nullptr;
    ret = hcomm::SocketProcess::GetInstance(0).SendNoBlock(
        SocketProcessTest::socketHandle, &errorBuffer, sendSize, sentSize);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(SocketProcessTest, Ut_SendNoBlock_When_NormalInput_Expect_SendData)
{
    HcclResult ret;
    if (SocketProcessTest::socketHandle == nullptr) {
        ret = hcomm::SocketProcess::GetInstance(0).GetSocket(
            &SocketProcessTest::socketDesc, SocketProcessTest::socketHandle);
        EXPECT_EQ(ret, HCCL_SUCCESS);
    }

    u64 sendbuffer = 123;
    u64 sendSize = sizeof(sendbuffer);
    u64 sentSize = 0;
    u64* sentSizePtr = &sentSize;
    ret = hcomm::SocketProcess::GetInstance(0).SendNoBlock(
        SocketProcessTest::socketHandle, &sendbuffer, sendSize, sentSizePtr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(sendSize, *sentSizePtr);
}

TEST_F(SocketProcessTest, Ut_RecvNoBlock_When_InvalidInput_Expect_ReturnError)
{
    SocketHandle tempSocketHandle = nullptr;
    u64 recvbuffer = 0;
    u64 recvSize = sizeof(recvbuffer);
    u64 recvedSize = 0;
    u64* recvedSizePtr = &recvedSize;
    HcclResult ret
        = hcomm::SocketProcess::GetInstance(0).RecvNoBlock(tempSocketHandle, &recvbuffer, recvSize, recvedSizePtr);
    EXPECT_EQ(ret, HCCL_E_PARA);

    tempSocketHandle = reinterpret_cast<SocketHandle>(0x1);
    ret = hcomm::SocketProcess::GetInstance(0).RecvNoBlock(tempSocketHandle, &recvbuffer, recvSize, recvedSizePtr);
    EXPECT_EQ(ret, HCCL_E_PARA);

    void* errorBuffer = nullptr;
    ret = hcomm::SocketProcess::GetInstance(0).RecvNoBlock(
        SocketProcessTest::socketHandle, &errorBuffer, recvSize, recvedSizePtr);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(SocketProcessTest, Ut_RecvNoBlock_When_NormalInput_Expect_RecvData)
{
    HcclResult ret;
    if (SocketProcessTest::socketHandle == nullptr) {
        ret = hcomm::SocketProcess::GetInstance(0).GetSocket(
            &SocketProcessTest::socketDesc, SocketProcessTest::socketHandle);
        EXPECT_EQ(ret, HCCL_SUCCESS);
    }

    u64 recvbuffer = 0;
    u64 recvSize = sizeof(recvbuffer);
    u64 recvedSize = 0;
    u64* recvedSizePtr = &recvedSize;
    ret = hcomm::SocketProcess::GetInstance(0).RecvNoBlock(
        SocketProcessTest::socketHandle, &recvbuffer, recvSize, recvedSizePtr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(recvSize, *recvedSizePtr);
}

TEST_F(SocketProcessTest, Ut_ConvertToHcclSocketRole_When_NormalInput_Expect_ConvertRole)
{
    Hccl::SocketRole role;
    HcommSocketRole hcommRole;
    hcommRole = HCOMM_SOCKET_ROLE_CLIENT;
    role = hcomm::SocketProcess::GetInstance(0).ConvertToHcclSocketRole(hcommRole);
    EXPECT_EQ(role, Hccl::SocketRole::CLIENT);

    hcommRole = HCOMM_SOCKET_ROLE_SERVER;
    role = hcomm::SocketProcess::GetInstance(0).ConvertToHcclSocketRole(hcommRole);
    EXPECT_EQ(role, Hccl::SocketRole::SERVER);

    hcommRole = HCOMM_SOCKET_ROLE_RESERVED;
    role = hcomm::SocketProcess::GetInstance(0).ConvertToHcclSocketRole(hcommRole);
    EXPECT_EQ(role, Hccl::SocketRole::CLIENT);
}

TEST_F(SocketProcessTest, Ut_GetInstance_SocketProcessRef_When_CalledTwice_Expect_SameInstance)
{
    s32 deviceLogicId = 0;
    hcomm::SocketProcess& process1 = hcomm::SocketProcess::GetInstance(deviceLogicId);
    hcomm::SocketProcess& process2 = hcomm::SocketProcess::GetInstance(deviceLogicId);
    EXPECT_EQ(&process1, &process2);
}

TEST_F(SocketProcessTest, Ut_GetInstance_When_InvalidDeviceLogicId_Expect_ReturnDefaultInstance)
{
    s32 invalidDeviceLogicId = 999;
    hcomm::SocketProcess& process = hcomm::SocketProcess::GetInstance(invalidDeviceLogicId);
    hcomm::SocketProcess& expectedProcess = hcomm::SocketProcess::GetInstance(0);
    EXPECT_EQ(&process, &expectedProcess);
}

TEST_F(SocketProcessTest, Ut_DestroySocketHandle_When_InvalidInput_Expect_ReturnError)
{
    SocketHandle tempSocketHandle = nullptr;
    HcclResult ret = hcomm::SocketProcess::GetInstance(0).DestroySocketHandle(tempSocketHandle);
    EXPECT_EQ(ret, HCCL_E_PARA);

    tempSocketHandle = reinterpret_cast<SocketHandle>(0x1);
    ret = hcomm::SocketProcess::GetInstance(0).DestroySocketHandle(tempSocketHandle);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(SocketProcessTest, Ut_DestroySocketHandle_When_NormalInput_Expect_Success)
{
    HcclResult ret;
    if (SocketProcessTest::socketHandle == nullptr) {
        ret = hcomm::SocketProcess::GetInstance(0).GetSocket(
            &SocketProcessTest::socketDesc, SocketProcessTest::socketHandle);
        EXPECT_EQ(ret, HCCL_SUCCESS);
    }

    ret = hcomm::SocketProcess::GetInstance(0).DestroySocketHandle(SocketProcessTest::socketHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    SocketProcessTest::socketHandle = nullptr;
}

TEST_F(SocketProcessTest, Ut_SocketMgr_GetSocket)
{
    uint32_t devicePhyId = 0;
    Hccl::LinkData linkData = BuildDefaultLinkData();
    HCCL_INFO("[AicpuTsUboeChannel::%s] built linkData: %s", __func__, linkData.Describe().c_str());
    std::string socketTag = "UT_SOCKET_TAG";
    bool noRankId = true;
    Hccl::SocketConfig socketConfig = Hccl::SocketConfig(linkData, socketTag, noRankId);
    Hccl::Socket* socketTmp = nullptr;
    const Hccl::SocketConfig* configPtr = &socketConfig;
    SocketMgr::GetInstance(devicePhyId).GetSocket(socketConfig, socketTmp);
    SocketMgr::GetInstance(devicePhyId).PutSocket(configPtr, socketTmp);
    SocketMgr::GetInstance(devicePhyId).GetSocket(socketConfig, socketTmp);
}

TEST_F(SocketProcessTest, Ut_SocketMgr_Init_When_HostNet_Expect_HostOnlyInit)
{
    SocketMgr& socketMgr = SocketMgr::GetInstance(0);
    ResetSocketMgrForTest(socketMgr);
    MOCKER(hrtGetDeviceCount).stubs().with(mockcpp::any()).will(invoke(StubSocketMgrHrtGetDeviceCountNoDevice));

    HcclResult ret = socketMgr.Init();

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(socketMgr.isLoaded_);
    EXPECT_TRUE(socketMgr.isHostOnlyInit_);
    EXPECT_EQ(socketMgr.devicePhyId_, 0U);
    EXPECT_EQ(socketMgr.serverListenPort_, SOCKET_MGR_TEMP_LISTEN_PORT);
}

TEST_F(SocketProcessTest, Ut_SocketMgr_Init_When_DevNet_Expect_DeviceInit)
{
    SocketMgr& socketMgr = SocketMgr::GetInstance(7);
    ResetSocketMgrForTest(socketMgr, 7U);
    MOCKER(hrtGetDeviceCount).stubs().with(mockcpp::any()).will(invoke(StubSocketMgrHrtGetDeviceCountWithDevice));
    MOCKER(hrtGetDevice).stubs().with(mockcpp::any()).will(invoke(StubSocketMgrHrtGetDevice));
    MOCKER(hrtGetDevicePhyIdByIndex)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(invoke(StubSocketMgrHrtGetDevicePhyIdByUserDevId));

    HcclResult ret = socketMgr.Init();

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(socketMgr.isLoaded_);
    EXPECT_FALSE(socketMgr.isHostOnlyInit_);
    EXPECT_EQ(socketMgr.devicePhyId_, 7U);
    EXPECT_EQ(socketMgr.serverListenPort_, SOCKET_MGR_TEMP_LISTEN_PORT);
}

TEST_F(SocketProcessTest, Ut_SocketMgr_GetSocket_When_UnsupportedPortType_Expect_NotSupport)
{
    SocketMgr& socketMgr = SocketMgr::GetInstance(0);
    ResetSocketMgrForTest(socketMgr);
    Hccl::SocketConfig socketConfig = BuildSocketMgrConfig(Hccl::PortDeploymentType::P2P);
    MOCKER(hrtGetDeviceCount).stubs().with(mockcpp::any()).will(invoke(StubSocketMgrHrtGetDeviceCountNoDevice));
    Hccl::Socket* socket = nullptr;

    HcclResult ret = socketMgr.GetSocket(socketConfig, socket);

    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
    EXPECT_TRUE(socketMgr.isLoaded_);
    EXPECT_TRUE(socketMgr.isHostOnlyInit_);
    EXPECT_EQ(socket, nullptr);
}

// ① 同一 SocketConfig 复用：GetSocket 复用已有 socket 递增引用计数到 2，须两次 DestroySocket 后 socket 才从 socketMap_
// 移除
TEST_F(SocketProcessTest, Ut_SocketMgr_RefCount_TwoDestroySocket_RemoveAtZero)
{
    SocketMgr& socketMgr = SocketMgr::GetInstance(0);
    ResetSocketMgrForTest(socketMgr);
    MOCKER(hrtGetDeviceCount).stubs().with(mockcpp::any()).will(invoke(StubSocketMgrHrtGetDeviceCountNoDevice));

    Hccl::SocketConfig socketConfig = BuildRefCountConfig("UT_REFCOUNT_SHARE");
    // 已有第一个 holder（refCount=1 且已释放借用），第二个 holder 复用同一 socket
    Hccl::Socket* storedSocket = InsertSocketEntryForRefCount(socketMgr, socketConfig, 1U);

    Hccl::Socket* holder2 = nullptr;
    EXPECT_EQ(socketMgr.GetSocket(socketConfig, holder2), HCCL_SUCCESS);
    EXPECT_EQ(holder2, storedSocket);
    EXPECT_EQ(socketMgr.socketMap_.begin()->second.refCount, 2U);

    // refCount 未归零时不销毁：第一次 DestroySocket 仅递减计数，entry 仍在
    EXPECT_EQ(socketMgr.DestroySocket(socketConfig), HCCL_SUCCESS);
    EXPECT_EQ(socketMgr.socketMap_.size(), 1U);
    EXPECT_EQ(socketMgr.socketMap_.begin()->second.refCount, 1U);

    // 第二次 DestroySocket 后 refCount 归零，socket 从 socketMap_ 移除
    EXPECT_EQ(socketMgr.DestroySocket(socketConfig), HCCL_SUCCESS);
    EXPECT_EQ(socketMgr.socketMap_.size(), 0U);
}

// ② DestroySocket 未命中 config 时，不影响其他 entry 的引用计数
TEST_F(SocketProcessTest, Ut_SocketMgr_DestroySocket_MissConfig_NotAffectOtherEntries)
{
    SocketMgr& socketMgr = SocketMgr::GetInstance(0);
    ResetSocketMgrForTest(socketMgr);

    Hccl::SocketConfig configA = BuildRefCountConfig("UT_REFCOUNT_A");
    Hccl::SocketConfig configB = BuildRefCountConfig("UT_REFCOUNT_B");
    Hccl::SocketConfig missConfig = BuildRefCountConfig("UT_REFCOUNT_MISS");

    InsertSocketEntryForRefCount(socketMgr, configA, 1U);
    InsertSocketEntryForRefCount(socketMgr, configB, 2U);

    EXPECT_EQ(socketMgr.DestroySocket(missConfig), HCCL_SUCCESS);
    EXPECT_EQ(socketMgr.socketMap_.size(), 2U);
    EXPECT_EQ(socketMgr.socketMap_.find(configA)->second.refCount, 1U);
    EXPECT_EQ(socketMgr.socketMap_.find(configB)->second.refCount, 2U);
}

// ③ GetSocket→PutSocket 配对：GetSocket 递增、PutSocket 对称递减并释放借用标记
TEST_F(SocketProcessTest, Ut_SocketMgr_PutSocket_RefCount_Pairing)
{
    SocketMgr& socketMgr = SocketMgr::GetInstance(0);
    ResetSocketMgrForTest(socketMgr);
    MOCKER(hrtGetDeviceCount).stubs().with(mockcpp::any()).will(invoke(StubSocketMgrHrtGetDeviceCountNoDevice));

    Hccl::SocketConfig socketConfig = BuildRefCountConfig("UT_REFCOUNT_PUT");
    Hccl::Socket* storedSocket = InsertSocketEntryForRefCount(socketMgr, socketConfig, 0U);

    Hccl::Socket* holder = nullptr;
    EXPECT_EQ(socketMgr.GetSocket(socketConfig, holder), HCCL_SUCCESS);
    EXPECT_EQ(holder, storedSocket);
    EXPECT_EQ(socketMgr.socketMap_.begin()->second.refCount, 1U);
    EXPECT_TRUE(socketMgr.socketInUseMap_[storedSocket].load());

    // PutSocket 与 GetSocket 对称：释放借用并递减引用计数
    const Hccl::SocketConfig* configPtr = &socketConfig;
    EXPECT_EQ(socketMgr.PutSocket(configPtr, holder), HCCL_SUCCESS);
    EXPECT_EQ(holder, nullptr);
    EXPECT_EQ(socketMgr.socketMap_.begin()->second.refCount, 0U);
    EXPECT_FALSE(socketMgr.socketInUseMap_[storedSocket].load());
}

// ④ HrtRaSocketWhiteListDel 抛异常（RA 白名单删除失败）时，DeleteWhiteListLocked 仍清理本地白名单记录，
// socket 也照常销毁，避免 handle2WhiteListMap_ 残留孤立条目。
TEST_F(SocketProcessTest, Ut_SocketMgr_DeleteWhiteListFail_StillEraseMap)
{
    SocketMgr& socketMgr = SocketMgr::GetInstance(0);
    ResetSocketMgrForTest(socketMgr);

    Hccl::SocketConfig socketConfig = BuildRefCountConfig("UT_REFCOUNT_WLIST");
    Hccl::Socket* storedSocket = InsertSocketEntryForRefCount(socketMgr, socketConfig, 1U);

    // 预置一条白名单记录，使 DeleteWhiteListLocked 走到 HrtRaSocketWhiteListDel 调用
    Hccl::RaSocketWhitelist wlist{};
    wlist.connLimit = 1;
    wlist.tag = "UT_WLIST";
    socketMgr.handle2WhiteListMap_[storedSocket->GetFdHandle()].push_back(wlist);

    // 让 RA 白名单删除失败（抛异常）
    MOCKER(Hccl::HrtRaSocketWhiteListDel)
        .expects(once())
        .with(mockcpp::any(), mockcpp::any())
        .will(throws(std::runtime_error("del whitelist failed")));

    // refCount 归零触发销毁：白名单删除失败不阻断，socket 仍被销毁
    EXPECT_EQ(socketMgr.DestroySocket(socketConfig), HCCL_SUCCESS);
    // 核心断言：本地白名单记录已被清理，未残留孤立条目
    EXPECT_EQ(socketMgr.handle2WhiteListMap_.size(), 0U);
    // socket 本体仍正常销毁
    EXPECT_EQ(socketMgr.socketMap_.size(), 0U);
}
