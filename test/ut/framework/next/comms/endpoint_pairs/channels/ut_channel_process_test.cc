/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../ut_hcomm_base.h"
#include "channel_process.h"
#include "aiv_urma_channel.h"
#include "aicpu_ts_roce_host_loc_channel.h"
#include "aicpu_ts_channel_helper.h"
#include "aiv_channel_helper.h"
#include "mem_device_pub.h"
#include "opbase_adpt.h"
#include "ccu/ccu_urma_channel.h"
#include "hcomm_res_ccu_entity.h"

// ConvertToLinkStatus 是 channel_process.cc 内 hcomm 命名空间下的自由函数(外部链接),
// 头文件未声明, 这里前向声明以便 UT 直接验证终态映射契约。
namespace hcomm {
void ConvertToLinkStatus(const std::vector<ChannelStatus>& internalStatus, std::vector<int32_t>& linkStatusList);
}

class TestChannelProcess : public TestHcommCAdptBase {
public:
    void SetUp() override { TestHcommCAdptBase::SetUp(); }
    void TearDown() override
    {
        {
            std::lock_guard<std::mutex> lock(hcomm::ChannelProcess::g_ChannelMapMtx);
            hcomm::ChannelProcess::g_ChannelMap.clear();
            hcomm::ChannelProcess::g_ChannelD2HMap.clear();
        }
        TestHcommCAdptBase::TearDown();
    }
};

class FakeChannelForProcess : public hcomm::Channel {
public:
    explicit FakeChannelForProcess(hcomm::HcommChannelKind kind) : kind_(kind) {}

    hcomm::HcommChannelKind GetChannelKind() const override { return kind_; }
    HcclResult Init() override { return HCCL_SUCCESS; }
    HcclResult GetNotifyNum(uint32_t*) const override { return HCCL_SUCCESS; }
    HcclResult GetRemoteMems(uint32_t*, CommMem**, char***) override { return HCCL_SUCCESS; }
    hcomm::ChannelStatus GetStatus() override { return hcomm::ChannelStatus::READY; }
    const HcommChannelDesc& GetChannelDesc() const override
    {
        static HcommChannelDesc defaultDesc{};
        return defaultDesc;
    }
    HcclResult Clean() override { return HCCL_SUCCESS; }
    HcclResult Resume() override { return HCCL_SUCCESS; }
    HcclResult NotifyRecord(const uint32_t) override { return HCCL_SUCCESS; }
    HcclResult NotifyWait(const uint32_t, const uint32_t) override { return HCCL_SUCCESS; }
    HcclResult WriteWithNotify(void*, const void*, const uint64_t, uint32_t) override { return HCCL_SUCCESS; }
    HcclResult Write(void*, const void*, uint64_t) override { return HCCL_SUCCESS; }
    HcclResult Read(void*, const void*, uint64_t) override { return HCCL_SUCCESS; }
    HcclResult ChannelFence() override { return HCCL_SUCCESS; }

private:
    hcomm::HcommChannelKind kind_;
};

// CreateChannelsLoop 空指针测试
TEST_F(TestChannelProcess, Ut_TestCreateChannelsLoop_When_EndpointHandleNullptr_Return_HCCL_E_PTR)
{
    HcommChannelDesc desc = {};
    ChannelHandle outHandles[1] = {};
    HcclResult ret = hcomm::ChannelProcess::CreateChannelsLoop(nullptr, COMM_ENGINE_AICPU, &desc, 1, outHandles);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// FillChannelD2HMap 空指针和参数校验测试
TEST_F(TestChannelProcess, Ut_TestFillChannelD2HMap_When_DeviceChannelHandlesNullptr_Return_HCCL_E_PTR)
{
    ChannelHandle hostHandles[1] = {};
    HcclResult ret = hcomm::ChannelProcess::FillChannelD2HMap(nullptr, hostHandles, 1);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestChannelProcess, Ut_TestFillChannelD2HMap_When_HostChannelHandlesNullptr_Return_HCCL_E_PTR)
{
    ChannelHandle deviceHandles[1] = {};
    HcclResult ret = hcomm::ChannelProcess::FillChannelD2HMap(deviceHandles, nullptr, 1);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestChannelProcess, Ut_TestFillChannelD2HMap_When_ListNumZero_Return_HCCL_E_PARA)
{
    ChannelHandle deviceHandles[1] = {};
    ChannelHandle hostHandles[1] = {};
    HcclResult ret = hcomm::ChannelProcess::FillChannelD2HMap(deviceHandles, hostHandles, 0);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// LaunchChannelKernelCommon 空指针和参数校验测试
TEST_F(TestChannelProcess, Ut_TestLaunchChannelKernelCommon_When_ChannelHandlesNullptr_Return_HCCL_E_PTR)
{
    HcommChannelDesc hcommDescs[1] = {};
    ChannelHandle hostHandles[1] = {};
    HcclResult ret = hcomm::ChannelProcess::LaunchChannelKernelCommon(
        nullptr, hostHandles, hcommDescs, 1, "test_tag", nullptr, "test_kernel", false);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestChannelProcess, Ut_TestLaunchChannelKernelCommon_When_HostChannelHandlesNullptr_Return_HCCL_E_PTR)
{
    HcommChannelDesc hcommDescs[1] = {};
    ChannelHandle deviceHandles[1] = {};
    HcclResult ret = hcomm::ChannelProcess::LaunchChannelKernelCommon(
        deviceHandles, nullptr, hcommDescs, 1, "test_tag", nullptr, "test_kernel", false);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestChannelProcess, Ut_TestLaunchChannelKernelCommon_When_ListNumZero_Return_HCCL_E_PARA)
{
    HcommChannelDesc hcommDescs[1] = {};
    ChannelHandle deviceHandles[1] = {};
    ChannelHandle hostHandles[1] = {};
    HcclResult ret = hcomm::ChannelProcess::LaunchChannelKernelCommon(
        deviceHandles, hostHandles, hcommDescs, 0, "test_tag", nullptr, "test_kernel", false);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// ChannelGetRemoteMems 空指针测试
TEST_F(TestChannelProcess, Ut_TestChannelGetRemoteMems_When_RemoteMemNullptr_Return_HCCL_E_PTR)
{
    char** memInfos = nullptr;
    uint32_t memNum = 0;
    HcclResult ret = hcomm::ChannelProcess::ChannelGetRemoteMems(0, &memNum, nullptr, &memInfos);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestChannelProcess, Ut_TestChannelGetRemoteMems_When_MemInfosNullptr_Return_HCCL_E_PTR)
{
    CommMem* remoteMem = nullptr;
    uint32_t memNum = 0;
    HcclResult ret = hcomm::ChannelProcess::ChannelGetRemoteMems(0, &memNum, &remoteMem, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestChannelProcess, Ut_TestChannelGetRemoteMems_When_MemNumNullptr_Return_HCCL_E_PTR)
{
    CommMem* remoteMem = nullptr;
    char** memInfos = nullptr;
    HcclResult ret = hcomm::ChannelProcess::ChannelGetRemoteMems(0, nullptr, &remoteMem, &memInfos);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// CCU Channel 实体出参为空指针
TEST_F(TestChannelProcess, Ut_CcuChannelGetEntity_When_EntityNullptr_Return_HCCL_E_PTR)
{
    HcclResult ret = hcomm::ChannelProcess::CcuChannelGetEntity(1, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// 对外入口直接透传 ChannelProcess 查询结果，空出参应保持错误码语义一致
TEST_F(TestChannelProcess, Ut_HcommCcuChannelGetEntity_When_EntityNullptr_Return_HCCL_E_PTR)
{
    HcommResult ret = HcommCcuChannelGetEntity(1, nullptr);
    EXPECT_EQ(ret, static_cast<HcommResult>(HCCL_E_PTR));
}

TEST_F(TestChannelProcess, Ut_ChannelClean_NullList_Returns_E_PARA)
{
    // Passing null pointer should return parameter error
    auto ret = hcomm::ChannelProcess::ChannelClean(nullptr, 1);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestChannelProcess, Ut_ChannelResumeConcurrency_ZeroChannels_Returns_SUCCESS)
{
    // Zero channel count should be a no-op and return success
    ChannelHandle dummyList[1] = {0};
    auto ret = hcomm::ChannelProcess::ChannelResumeConcurrency(dummyList, 0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(TestChannelProcess, Ut_ChannelResume_When_ResumeConcurrencyFails_ReturnsError)
{
    ChannelHandle list[1] = {(ChannelHandle)0x1};
    // Mock ChannelResumeConcurrency to return internal error
    MOCKER_CPP(&hcomm::ChannelProcess::ChannelResumeConcurrency, HcclResult(const ChannelHandle*, uint32_t))
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_INTERNAL));

    auto ret = hcomm::ChannelProcess::ChannelResume(list, 1);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(TestChannelProcess, Ut_ChannelResume_NullList_Returns_E_PARA)
{
    auto ret = hcomm::ChannelProcess::ChannelResume(nullptr, 1);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// 公共：mock LaunchChannelKernelCommon 短接 V2 路径，验证指定 kind 命中 V2 分流
// （ChannelKernelLaunchForBase → LaunchChannelKernelCommon）。UBOE/ROCE_HOST_LOC/PCIE 三个 kind 共用。
static void LaunchChannelKernelForKindExpectsV2(hcomm::HcommChannelKind kind)
{
    HcommChannelDesc hcommDescs[1] = {};
    ChannelHandle deviceHandles[1] = {};

    FakeChannelForProcess fakeChannel(kind);
    ChannelHandle hostHandles[1] = {reinterpret_cast<ChannelHandle>(&fakeChannel)};

    MOCKER_CPP(
        &hcomm::ChannelProcess::LaunchChannelKernelCommon,
        HcclResult(
            ChannelHandle*, ChannelHandle*, const HcommChannelDesc*, uint32_t, const std::string&, aclrtBinHandle,
            const std::string&, bool))
        .stubs()
        .with(
            mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(),
            mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    auto ret = hcomm::ChannelProcess::LaunchChannelKernel(deviceHandles, hostHandles, hcommDescs, 1, nullptr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(TestChannelProcess, Ut_LaunchChannelKernel_When_ChannelKindIsUBOE_CallsChannelKernelLaunchForBase)
{
    LaunchChannelKernelForKindExpectsV2(hcomm::HcommChannelKind::AICPU_TS_UBOE);
}

TEST_F(TestChannelProcess, Ut_LaunchChannelKernel_When_ChannelKindIsRoceHostLoc_CallsChannelKernelLaunchForBase)
{
    LaunchChannelKernelForKindExpectsV2(hcomm::HcommChannelKind::AICPU_TS_ROCE_HOST_LOC);
}

TEST_F(TestChannelProcess, Ut_LaunchChannelKernel_When_ChannelKindIsPcie_CallsChannelKernelLaunchForBase)
{
    LaunchChannelKernelForKindExpectsV2(hcomm::HcommChannelKind::AICPU_TS_PCIE);
}

TEST_F(TestChannelProcess, Ut_ChannelUpdateKernelLaunch_When_ChannelKindUnsupported_Expect_NotSupport)
{
    FakeChannelForProcess unsupportedChannel(hcomm::HcommChannelKind::INVALID);
    ChannelHandle deviceHandles[1] = {static_cast<ChannelHandle>(0x1234)};
    ChannelHandle hostHandles[1] = {reinterpret_cast<ChannelHandle>(&unsupportedChannel)};

    HcclResult ret
        = hcomm::ChannelProcess::ChannelUpdateKernelLaunch(deviceHandles, hostHandles, 1, "test_tag", nullptr);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(TestChannelProcess, Ut_CopyUpdateKernelPackResToDevice_When_TotalSizeTooSmall_Expect_Para)
{
    std::vector<std::vector<char>> hostPackBuffers{{'a', 'b'}, {'c', 'd'}};
    std::vector<u32> channelSizeVec{2U, 2U};
    hccl::DeviceMem channelSizeAddr;
    hccl::DeviceMem devicePackBuf;
    MOCKER(memcpy_s).stubs().will(returnValue(EOK));

    HcclResult ret = hcomm::ChannelProcess::CopyUpdateKernelPackResToDevice(
        hostPackBuffers, channelSizeVec, 3U, channelSizeAddr, devicePackBuf);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// ===================== 非阻塞建链 UT =====================

// 通用 FakeChannel，支持配置 engine/kind/status/deviceEntityReady/ctxMem
class FakeChannel : public hcomm::Channel {
public:
    FakeChannel(CommEngine eng = COMM_ENGINE_AICPU, hcomm::ChannelStatus sta = hcomm::ChannelStatus::READY)
    {
        engine_ = eng;
        channelKind_ = hcomm::HcommChannelKind::AICPU_TS_UBOE;
        fakeStatus_ = sta;
    }
    ~FakeChannel() = default;

    void SetStatus(hcomm::ChannelStatus s) { fakeStatus_ = s; }
    void SetDesc(const HcommChannelDesc& d) { desc_ = d; }

    hcomm::HcommChannelKind GetChannelKind() const override { return channelKind_; }
    HcclResult Init() override { return HCCL_SUCCESS; }
    HcclResult GetNotifyNum(uint32_t* notifyNum) const override { return HCCL_SUCCESS; }
    HcclResult GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos) override { return HCCL_SUCCESS; }
    hcomm::ChannelStatus GetStatus() override { return fakeStatus_; }
    const HcommChannelDesc& GetChannelDesc() const override { return desc_; }
    HcclResult Clean() override { return HCCL_SUCCESS; }
    HcclResult Resume() override { return HCCL_SUCCESS; }
    HcclResult NotifyRecord(const uint32_t remoteNotifyIdx) override { return HCCL_SUCCESS; }
    HcclResult NotifyWait(const uint32_t localNotifyIdx, const uint32_t timeout) override { return HCCL_SUCCESS; }
    HcclResult WriteWithNotify(void* dst, const void* src, const uint64_t len, uint32_t remoteNotifyIdx) override
    {
        return HCCL_SUCCESS;
    }
    HcclResult Write(void* dst, const void* src, uint64_t len) override { return HCCL_SUCCESS; }
    HcclResult Read(void* dst, const void* src, uint64_t len) override { return HCCL_SUCCESS; }
    HcclResult ChannelFence() override { return HCCL_SUCCESS; }

private:
    hcomm::ChannelStatus fakeStatus_{hcomm::ChannelStatus::INIT};
    HcommChannelDesc desc_{};
};

// 辅助：注册 FakeChannel 到全局 map
static ChannelHandle RegisterFakeChannel(std::shared_ptr<FakeChannel>& ch)
{
    ChannelHandle handle = reinterpret_cast<ChannelHandle>(ch.get());
    int32_t deviceId = 0;
    // mock hrtGetDevice 返回 0
    hcomm::DeviceChannelKey key{0, handle};
    std::lock_guard<std::mutex> lock(hcomm::ChannelProcess::g_ChannelMapMtx);
    hcomm::ChannelProcess::g_ChannelMap.emplace(handle, ch);
    hcomm::ChannelProcess::g_ChannelD2HMap.emplace(key, handle);
    return handle;
}

// --- GetChannelsInfo 参数校验 ---
TEST_F(TestChannelProcess, Ut_GetChannelsInfo_When_Nullptr_Return_E_PTR)
{
    std::vector<CommEngine> engines;
    std::vector<HcommChannelDesc> descs;
    std::vector<hcomm::ChannelStatus> status;
    EXPECT_EQ(hcomm::ChannelProcess::GetChannelsInfo(nullptr, 1, engines, descs, status), HCCL_E_PTR);
}

TEST_F(TestChannelProcess, Ut_GetChannelsInfo_When_ZeroNum_Return_E_PARA)
{
    ChannelHandle list[1] = {};
    std::vector<CommEngine> engines;
    std::vector<HcommChannelDesc> descs;
    std::vector<hcomm::ChannelStatus> status;
    EXPECT_EQ(hcomm::ChannelProcess::GetChannelsInfo(list, 0, engines, descs, status), HCCL_E_PARA);
}

// --- GetChannelsInfo 正常路径 ---
TEST_F(TestChannelProcess, Ut_GetChannelsInfo_When_Normal_Return_Success)
{
    auto ch = std::make_shared<FakeChannel>(COMM_ENGINE_AICPU, hcomm::ChannelStatus::READY);
    ChannelHandle handle = RegisterFakeChannel(ch);

    std::vector<CommEngine> engines;
    std::vector<HcommChannelDesc> descs;
    std::vector<hcomm::ChannelStatus> status;
    EXPECT_EQ(hcomm::ChannelProcess::GetChannelsInfo(&handle, 1, engines, descs, status), HCCL_SUCCESS);
    EXPECT_EQ(engines[0], COMM_ENGINE_AICPU);
    EXPECT_EQ(status[0], hcomm::ChannelStatus::READY);
}

// --- GetChannelsInfo channel not found ---
TEST_F(TestChannelProcess, Ut_GetChannelsInfo_When_NotFound_Return_Error)
{
    ChannelHandle handle = static_cast<ChannelHandle>(0x9999);
    std::vector<CommEngine> engines;
    std::vector<HcommChannelDesc> descs;
    std::vector<hcomm::ChannelStatus> status;
    EXPECT_NE(hcomm::ChannelProcess::GetChannelsInfo(&handle, 1, engines, descs, status), HCCL_SUCCESS);
}

// --- LaunchAicpuKernel: 全部已 ready 则跳过 ---
TEST_F(TestChannelProcess, Ut_LaunchAicpuKernel_When_AllReady_Skip)
{
    auto ch = std::make_shared<FakeChannel>();
    ch->SetDeviceEntityReady();
    ChannelHandle handle = RegisterFakeChannel(ch);

    HcommChannelDesc descs[1] = {};
    std::vector<int32_t> linkStatus = {hcomm::HCOMM_CHANNEL_STATUS_READY};
    int32_t statusList[1] = {0};
    EXPECT_EQ(
        AicpuTsChannelHelper::HandleStatus(&handle, 1, COMM_ENGINE_AICPU, descs, linkStatus, statusList), HCCL_SUCCESS);
}

// --- FillAivDevEntities: 全 ready 跳过 ---
TEST_F(TestChannelProcess, Ut_FillAivDevEntities_When_AllReady_Skip)
{
    auto ch = std::make_shared<FakeChannel>();
    ch->SetDeviceEntityReady();
    ChannelHandle handle = RegisterFakeChannel(ch);

    HcommChannelDesc descs[1] = {};
    std::vector<int32_t> linkStatus = {hcomm::HCOMM_CHANNEL_STATUS_READY};
    int32_t statusList[1] = {0};
    EXPECT_EQ(AivChannelHelper::HandleStatus(&handle, 1, descs, linkStatus, statusList), HCCL_SUCCESS);
}

// --- FillAivDevEntities: 非 READY 状态跳过 ---
TEST_F(TestChannelProcess, Ut_FillAivDevEntities_When_NotReadyStatus_Skip)
{
    auto ch = std::make_shared<FakeChannel>();
    ChannelHandle handle = RegisterFakeChannel(ch);

    HcommChannelDesc descs[1] = {};
    std::vector<int32_t> linkStatus = {hcomm::HCOMM_CHANNEL_STATUS_CONNECTING};
    int32_t statusList[1] = {0};
    EXPECT_EQ(AivChannelHelper::HandleStatus(&handle, 1, descs, linkStatus, statusList), HCCL_SUCCESS);
    EXPECT_FALSE(ch->IsDeviceEntityReady());
}

// --- PreAllocChannels: 非 AICPU/AIV engine 直接拷贝 ---
TEST_F(TestChannelProcess, Ut_PreAllocChannels_When_NotSupported_Return_HostHandle)
{
    auto ch = std::make_shared<FakeChannel>(COMM_ENGINE_CPU);
    ChannelHandle target = reinterpret_cast<ChannelHandle>(ch.get());

    ChannelHandle userChannels[1] = {};
    HcommChannelDesc descs[1] = {};
    EXPECT_EQ(
        hcomm::ChannelProcess::PrepareUserChannels(&target, userChannels, descs, 1, COMM_ENGINE_CPU), HCCL_SUCCESS);
    EXPECT_EQ(userChannels[0], target);
}

// --- PreAllocChannels: 参数校验 ---
TEST_F(TestChannelProcess, Ut_PreAllocChannels_When_Nullptr_Return_E_PTR)
{
    ChannelHandle userChannels[1] = {};
    HcommChannelDesc descs[1] = {};
    EXPECT_EQ(
        hcomm::ChannelProcess::PrepareUserChannels(nullptr, userChannels, descs, 1, COMM_ENGINE_AICPU), HCCL_E_PTR);
}

// --- PreAllocChannels: channelNum=0 ---
TEST_F(TestChannelProcess, Ut_PreAllocChannels_When_ZeroNum_Return_E_PARA)
{
    ChannelHandle target[1] = {};
    ChannelHandle userChannels[1] = {};
    HcommChannelDesc descs[1] = {};
    EXPECT_EQ(
        hcomm::ChannelProcess::PrepareUserChannels(target, userChannels, descs, 0, COMM_ENGINE_AICPU), HCCL_E_PARA);
}

// --- UnwrapChannelHandle: handle=0 ---
TEST_F(TestChannelProcess, Ut_UnwrapChannelHandle_When_HandleZero_Return_E_PTR)
{
    ChannelHandle handle = 0;
    EXPECT_EQ(UnwrapChannelHandle(handle), HCCL_E_PTR);
}

// --- UnwrapChannelHandle: magic word 匹配则解包 ---
TEST_F(TestChannelProcess, Ut_UnwrapChannelHandle_When_MagicMatch_Unwrap)
{
    HcommAicpuChannelCtx ctx{};
    ctx.abiHeader.version = HCOMM_AICPU_CHANNEL_CTX_VERSION;
    ctx.abiHeader.magicWord = HCOMM_AICPU_CHANNEL_CTX_MAGIC_WORD;
    ctx.deviceChannel = reinterpret_cast<void*>(static_cast<uintptr_t>(0xABCD));

    ChannelHandle handle = reinterpret_cast<ChannelHandle>(&ctx);
    EXPECT_EQ(UnwrapChannelHandle(handle), HCCL_SUCCESS);
    EXPECT_EQ(handle, static_cast<ChannelHandle>(reinterpret_cast<uintptr_t>(ctx.deviceChannel)));
}

// --- UnwrapChannelHandle: magic word 不匹配则保持不变 ---
TEST_F(TestChannelProcess, Ut_UnwrapChannelHandle_When_MagicNotMatch_KeepHandle)
{
    HcommAicpuChannelCtx ctx{};
    ctx.abiHeader.version = 0;
    ctx.abiHeader.magicWord = 0;
    ctx.deviceChannel = reinterpret_cast<void*>(static_cast<uintptr_t>(0xABCD));

    ChannelHandle handle = reinterpret_cast<ChannelHandle>(&ctx);
    ChannelHandle original = handle;
    EXPECT_EQ(UnwrapChannelHandle(handle), HCCL_SUCCESS);
    EXPECT_EQ(handle, original);
}

// 可配置 GetStatus 返回值的 Fake Channel，用于 ChannelGetStatus 测试
class FakeStatusChannel : public hcomm::Channel {
public:
    explicit FakeStatusChannel(hcomm::ChannelStatus status) : status_(status) {}
    hcomm::HcommChannelKind GetChannelKind() const override { return hcomm::HcommChannelKind::AICPU_TS_UBOE; }
    const HcommChannelDesc& GetChannelDesc() const override { return channelDesc_; }
    void SetChannelDesc(const HcommChannelDesc& channelDesc) { channelDesc_ = channelDesc; }
    HcclResult Init() override { return HCCL_SUCCESS; }
    HcclResult GetNotifyNum(uint32_t* notifyNum) const override { return HCCL_SUCCESS; }
    HcclResult GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos) override { return HCCL_SUCCESS; }
    hcomm::ChannelStatus GetStatus() override { return status_; }
    HcclResult Clean() override { return HCCL_SUCCESS; }
    HcclResult Resume() override { return HCCL_SUCCESS; }
    HcclResult NotifyRecord(const uint32_t remoteNotifyIdx) override { return HCCL_SUCCESS; }
    HcclResult NotifyWait(const uint32_t localNotifyIdx, const uint32_t timeout) override { return HCCL_SUCCESS; }
    HcclResult WriteWithNotify(void* dst, const void* src, const uint64_t len, uint32_t remoteNotifyIdx) override
    {
        return HCCL_SUCCESS;
    }
    HcclResult Write(void* dst, const void* src, uint64_t len) override { return HCCL_SUCCESS; }
    HcclResult Read(void* dst, const void* src, uint64_t len) override { return HCCL_SUCCESS; }
    HcclResult ChannelFence() override { return HCCL_SUCCESS; }

private:
    hcomm::ChannelStatus status_;
    HcommChannelDesc channelDesc_{};
};

// 将 channel 注册到 g_ChannelMap / g_ChannelD2HMap（deviceId=0，handle 即 channel 指针值）
static ChannelHandle RegisterFakeChannel(std::shared_ptr<hcomm::Channel> channel)
{
    ChannelHandle handle = reinterpret_cast<ChannelHandle>(channel.get());
    hcomm::ChannelProcess::g_ChannelMap[handle] = std::move(channel);
    hcomm::DeviceChannelKey key{0, handle};
    hcomm::ChannelProcess::g_ChannelD2HMap[key] = handle;
    return handle;
}

// 注册到通道表中的非 CCU 通道应返回不支持，避免错误解读实体地址
TEST_F(TestChannelProcess, Ut_CcuChannelGetEntity_When_ChannelIsNotCcu_Return_HCCL_E_NOT_SUPPORT)
{
    auto channel = std::make_shared<FakeChannelForProcess>(hcomm::HcommChannelKind::AICPU_TS_UBOE);
    ChannelHandle handle = RegisterFakeChannel(channel);
    HcclSetThreadDeviceId(0);

    HcommCcuChannelEntity channelEntity{};
    auto ret = hcomm::ChannelProcess::CcuChannelGetEntity(handle, &channelEntity);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(channelEntity.header.size, 0U);
}

// 未建立 transport 的 CCU 通道应返回未就绪，覆盖入口到 CcuUrmaChannel 的分发
TEST_F(TestChannelProcess, Ut_CcuChannelGetEntity_When_CcuChannelNotReady_Return_HCCL_E_UNAVAIL)
{
    HcommChannelDesc desc{};
    auto channel = std::make_shared<hcomm::CcuUrmaChannel>(nullptr, desc);
    ChannelHandle handle = RegisterFakeChannel(channel);
    HcclSetThreadDeviceId(0);

    HcommCcuChannelEntity channelEntity{};
    auto ret = hcomm::ChannelProcess::CcuChannelGetEntity(handle, &channelEntity);
    EXPECT_EQ(ret, HCCL_E_UNAVAIL);
    EXPECT_EQ(channelEntity.header.size, 0U);
}

// 正常链路应完成 entity 装配，且同线程两次查询复用同一个远端 buffer 数组地址。
TEST_F(TestChannelProcess, Ut_CcuChannelGetEntity_When_Normal_Expect_EntityAndTlsBufferReused)
{
    auto channel = std::make_shared<hcomm::CcuUrmaChannel>(nullptr, HcommChannelDesc{});
    hcomm::CcuChannelInfo channelInfo{};
    channelInfo.channelId = 101U;
    channelInfo.dieId = 1U;
    static hcomm::UrmaEndpoint endpoint{EndpointDesc{}};
    channel->ccuEndpoint_ = &endpoint;
    channel->socket_ = reinterpret_cast<Hccl::Socket*>(0x1);
    channel->memHandles_.push_back(reinterpret_cast<HcommMemHandle>(0x1));
    channel->isFirstPrintChannelInfo_ = false;
    channel->channelStatus_ = hcomm::ChannelStatus::SOCKET_OK;

    auto connection = std::make_unique<hcomm::CcuConnection>(
        CommAddr{}, CommAddr{}, channelInfo, std::vector<hcomm::CcuJetty*>{}, Hccl::UB_QOS_DEFAULT);
    connection->rmtCcuBufAddr_ = 0x10000U;
    connection->rmtCcuBufSize_ = 0x1000U;
    connection->rmtCcuBufTokenId_ = 10U;
    connection->rmtCcuBufTokenValue_ = 20U;
    auto impl = std::make_unique<hcomm::CcuTransport>(
        reinterpret_cast<Hccl::Socket*>(0x1), std::move(connection), hcomm::CcuTransport::CclBufferInfo{});
    impl->dieId_ = 1U;
    impl->userDevId_ = 0;
    impl->SetLocResStatus(hcomm::CcuTransport::CcuResStatus::RES_OK);
    impl->locRes_.ckes = {10U, 11U};
    impl->locRes_.xns = {20U, 21U};
    impl->rmtRes_.ckes = {30U, 31U};
    impl->rmtRes_.xns = {40U, 41U};
    impl->transStatus_ = hcomm::CcuTransport::TransStatus::READY;
    impl->rmtBufferVec_.push_back(
        std::make_unique<Hccl::RemoteUbRmaBuffer>(0x20000U, 0x2000U, 11U, 12U, HCCL_MEM_TYPE_DEVICE, "ccu_user"));
    channel->impl_ = std::move(impl);

    ChannelHandle handle = RegisterFakeChannel(channel);
    MOCKER(hrtGetDevice).stubs().will(invoke(+[](int32_t* deviceId) -> HcclResult {
        *deviceId = 0;
        return HCCL_SUCCESS;
    }));
    const hcomm::CcuVersion version = hcomm::CcuVersion::CCU_V2;
    MOCKER(hcomm::CcuDevMgrImp::GetCcuVersion)
        .stubs()
        .with(mockcpp::any(), outBound(version))
        .will(returnValue(HCCL_SUCCESS));

    HcommCcuChannelEntity firstEntity{};
    ASSERT_EQ(hcomm::ChannelProcess::CcuChannelGetEntity(handle, &firstEntity), HCCL_SUCCESS);
    ASSERT_EQ(firstEntity.remoteBufferNum, 1U);
    ASSERT_NE(firstEntity.remoteBuffers, nullptr);
    EXPECT_EQ(firstEntity.remoteBuffers[0].bufferInfo.rma.addr, 0x20000U);

    HcommCcuChannelEntity secondEntity{};
    ASSERT_EQ(hcomm::ChannelProcess::CcuChannelGetEntity(handle, &secondEntity), HCCL_SUCCESS);
    EXPECT_EQ(secondEntity.remoteBuffers, firstEntity.remoteBuffers);

    GlobalMockObject::verify();
    GlobalMockObject::reset();
}

// ChannelGetStatus channelList 空指针测试
TEST_F(TestChannelProcess, Ut_ChannelGetStatus_When_ChannelListNullptr_Return_HCCL_E_PTR)
{
    int32_t statusList[1] = {0};
    HcclResult ret = hcomm::ChannelProcess::ChannelGetStatus(nullptr, 1, statusList);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// ChannelGetStatus statusList 空指针测试
TEST_F(TestChannelProcess, Ut_ChannelGetStatus_When_StatusListNullptr_Return_HCCL_E_PTR)
{
    ChannelHandle channelList[1] = {0};
    HcclResult ret = hcomm::ChannelProcess::ChannelGetStatus(channelList, 1, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// 全部通道 READY 返回 SUCCESS，并回写状态
TEST_F(TestChannelProcess, Ut_ChannelGetStatus_When_AllReady_Return_SUCCESS)
{
    auto channel = std::make_shared<FakeStatusChannel>(hcomm::ChannelStatus::READY);
    ChannelHandle handle = RegisterFakeChannel(channel);
    ChannelHandle channelList[1] = {handle};
    int32_t statusList[1] = {0};
    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = hcomm::ChannelProcess::ChannelGetStatus(channelList, 1, statusList);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(statusList[0], static_cast<int32_t>(hcomm::ChannelStatus::READY));
}

// 存在 FAILED 通道（全部到终态但非全 READY）返回 HCCL_E_NETWORK
TEST_F(TestChannelProcess, Ut_ChannelGetStatus_When_HasFailed_Return_HCCL_E_NETWORK)
{
    auto channel = std::make_shared<FakeStatusChannel>(hcomm::ChannelStatus::FAILED);
    ChannelHandle handle = RegisterFakeChannel(channel);
    ChannelHandle channelList[1] = {handle};
    int32_t statusList[1] = {0};
    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = hcomm::ChannelProcess::ChannelGetStatus(channelList, 1, statusList);
    EXPECT_EQ(ret, HCCL_E_NETWORK);
    EXPECT_EQ(statusList[0], static_cast<int32_t>(hcomm::ChannelStatus::FAILED));
}

// 存在 SOCKET_TIMEOUT 通道（全部到终态但非全 READY）返回 HCCL_E_NETWORK
TEST_F(TestChannelProcess, Ut_ChannelGetStatus_When_HasSocketTimeout_Return_HCCL_E_NETWORK)
{
    auto channel = std::make_shared<FakeStatusChannel>(hcomm::ChannelStatus::SOCKET_TIMEOUT);
    ChannelHandle handle = RegisterFakeChannel(channel);
    ChannelHandle channelList[1] = {handle};
    int32_t statusList[1] = {0};
    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = hcomm::ChannelProcess::ChannelGetStatus(channelList, 1, statusList);
    EXPECT_EQ(ret, HCCL_E_NETWORK);
    EXPECT_EQ(statusList[0], static_cast<int32_t>(hcomm::ChannelStatus::SOCKET_TIMEOUT));
}

// 存在未到终态通道（INIT）返回 HCCL_E_AGAIN
TEST_F(TestChannelProcess, Ut_ChannelGetStatus_When_HasNotReady_Return_HCCL_E_AGAIN)
{
    auto channel = std::make_shared<FakeStatusChannel>(hcomm::ChannelStatus::INIT);
    ChannelHandle handle = RegisterFakeChannel(channel);
    ChannelHandle channelList[1] = {handle};
    int32_t statusList[1] = {0};
    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = hcomm::ChannelProcess::ChannelGetStatus(channelList, 1, statusList);
    EXPECT_EQ(ret, HCCL_E_AGAIN);
}

// READY + FAILED 混合（全部到终态但非全 READY）返回 HCCL_E_NETWORK
TEST_F(TestChannelProcess, Ut_ChannelGetStatus_When_MixReadyAndFailed_Return_HCCL_E_NETWORK)
{
    auto chReady = std::make_shared<FakeStatusChannel>(hcomm::ChannelStatus::READY);
    auto chFailed = std::make_shared<FakeStatusChannel>(hcomm::ChannelStatus::FAILED);
    ChannelHandle hReady = RegisterFakeChannel(chReady);
    ChannelHandle hFailed = RegisterFakeChannel(chFailed);
    ChannelHandle channelList[2] = {hReady, hFailed};
    int32_t statusList[2] = {0, 0};
    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = hcomm::ChannelProcess::ChannelGetStatus(channelList, 2, statusList);
    EXPECT_EQ(ret, HCCL_E_NETWORK);
}

// READY + INIT 混合（存在未到终态通道）返回 HCCL_E_AGAIN
TEST_F(TestChannelProcess, Ut_ChannelGetStatus_When_MixReadyAndInit_Return_HCCL_E_AGAIN)
{
    auto chReady = std::make_shared<FakeStatusChannel>(hcomm::ChannelStatus::READY);
    auto chInit = std::make_shared<FakeStatusChannel>(hcomm::ChannelStatus::INIT);
    ChannelHandle hReady = RegisterFakeChannel(chReady);
    ChannelHandle hInit = RegisterFakeChannel(chInit);
    ChannelHandle channelList[2] = {hReady, hInit};
    int32_t statusList[2] = {0, 0};
    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = hcomm::ChannelProcess::ChannelGetStatus(channelList, 2, statusList);
    EXPECT_EQ(ret, HCCL_E_AGAIN);
}

// statusList 已预标记 FAILED 的通道被 continue 跳过，不再获取状态，但计入 failCount
TEST_F(TestChannelProcess, Ut_ChannelGetStatus_When_StatusListPreMarkedFailed_SkipGetStatus)
{
    auto chReady = std::make_shared<FakeStatusChannel>(hcomm::ChannelStatus::READY);
    auto chFailed = std::make_shared<FakeStatusChannel>(hcomm::ChannelStatus::FAILED);
    ChannelHandle hReady = RegisterFakeChannel(chReady);
    ChannelHandle hFailed = RegisterFakeChannel(chFailed);
    ChannelHandle channelList[2] = {hReady, hFailed};
    // 第二个通道预标记为 FAILED，触发 continue 分支，不再调用其 GetStatus
    int32_t statusList[2] = {0, static_cast<int32_t>(hcomm::ChannelStatus::FAILED)};
    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = hcomm::ChannelProcess::ChannelGetStatus(channelList, 2, statusList);
    // 被跳过的通道计入 failCount，故 readyCount(1)+failCount(1)=2 == listNum，且非全 READY，返回 HCCL_E_NETWORK
    EXPECT_EQ(ret, HCCL_E_NETWORK);
    // 被跳过通道的状态保持预标记值不变
    EXPECT_EQ(statusList[1], static_cast<int32_t>(hcomm::ChannelStatus::FAILED));
    // 第一个通道状态被刷新为 READY
    EXPECT_EQ(statusList[0], static_cast<int32_t>(hcomm::ChannelStatus::READY));
}

// handle 未注册时 WithChannelByHandleLocked 返回 HCCL_E_NOT_FOUND
TEST_F(TestChannelProcess, Ut_ChannelGetStatus_When_HandleNotFound_Return_HCCL_E_NOT_FOUND)
{
    ChannelHandle channelList[1] = {static_cast<ChannelHandle>(0xDEADBEEFULL)};
    int32_t statusList[1] = {0};
    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = hcomm::ChannelProcess::ChannelGetStatus(channelList, 1, statusList);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(TestChannelProcess, Ut_ChannelGetStatus_When_RmtUnavail_Return_HCCL_E_UNAVAIL)
{
    auto channel = std::make_shared<FakeStatusChannel>(hcomm::ChannelStatus::RES_RMT_UNAVAIL);
    ChannelHandle handle = RegisterFakeChannel(channel);
    ChannelHandle channelList[1] = {handle};
    int32_t statusList[1] = {0};
    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = hcomm::ChannelProcess::ChannelGetStatus(channelList, 1, statusList);
    EXPECT_EQ(ret, HCCL_E_UNAVAIL);
    EXPECT_EQ(statusList[0], static_cast<int32_t>(hcomm::ChannelStatus::RES_RMT_UNAVAIL));
}

TEST_F(TestChannelProcess, Ut_ChannelGetStatus_When_LocUnavail_Return_HCCL_E_UNAVAIL)
{
    auto channel = std::make_shared<FakeStatusChannel>(hcomm::ChannelStatus::RES_LOC_UNAVAIL);
    ChannelHandle handle = RegisterFakeChannel(channel);
    ChannelHandle channelList[1] = {handle};
    int32_t statusList[1] = {0};
    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = hcomm::ChannelProcess::ChannelGetStatus(channelList, 1, statusList);
    EXPECT_EQ(ret, HCCL_E_UNAVAIL);
    EXPECT_EQ(statusList[0], static_cast<int32_t>(hcomm::ChannelStatus::RES_LOC_UNAVAIL));
}

// TC-CP-013: 混合终态(READY + FAILED + RES_LOC_UNAVAIL), 存在网络类失败时优先返回 NETWORK
TEST_F(TestChannelProcess, Ut_ChannelGetStatus_When_MixUnavailAndFailed_Expect_NETWORK)
{
    auto chReady = std::make_shared<FakeStatusChannel>(hcomm::ChannelStatus::READY);
    auto chFailed = std::make_shared<FakeStatusChannel>(hcomm::ChannelStatus::FAILED);
    auto chLocUnavail = std::make_shared<FakeStatusChannel>(hcomm::ChannelStatus::RES_LOC_UNAVAIL);
    ChannelHandle hReady = RegisterFakeChannel(chReady);
    ChannelHandle hFailed = RegisterFakeChannel(chFailed);
    ChannelHandle hLocUnavail = RegisterFakeChannel(chLocUnavail);
    ChannelHandle channelList[3] = {hReady, hFailed, hLocUnavail};
    int32_t statusList[3] = {0, 0, 0};
    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = hcomm::ChannelProcess::ChannelGetStatus(channelList, 3, statusList);
    // 设计约定: 资源不足仅在整批无其他失败时返回 UNAVAIL, 混有网络类失败时优先返回 NETWORK
    EXPECT_EQ(ret, HCCL_E_NETWORK);
    EXPECT_EQ(statusList[0], static_cast<int32_t>(hcomm::ChannelStatus::READY));
    EXPECT_EQ(statusList[1], static_cast<int32_t>(hcomm::ChannelStatus::FAILED));
    EXPECT_EQ(statusList[2], static_cast<int32_t>(hcomm::ChannelStatus::RES_LOC_UNAVAIL));
}

// TC-CP-013b: 混合终态(READY + SOCKET_TIMEOUT + RES_LOC_UNAVAIL), 存在超时类失败时优先返回 NETWORK
TEST_F(TestChannelProcess, Ut_ChannelGetStatus_When_MixTimeoutAndUnavail_Expect_NETWORK)
{
    auto chReady = std::make_shared<FakeStatusChannel>(hcomm::ChannelStatus::READY);
    auto chTimeout = std::make_shared<FakeStatusChannel>(hcomm::ChannelStatus::SOCKET_TIMEOUT);
    auto chLocUnavail = std::make_shared<FakeStatusChannel>(hcomm::ChannelStatus::RES_LOC_UNAVAIL);
    ChannelHandle hReady = RegisterFakeChannel(chReady);
    ChannelHandle hTimeout = RegisterFakeChannel(chTimeout);
    ChannelHandle hLocUnavail = RegisterFakeChannel(chLocUnavail);
    ChannelHandle channelList[3] = {hReady, hTimeout, hLocUnavail};
    int32_t statusList[3] = {0, 0, 0};
    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = hcomm::ChannelProcess::ChannelGetStatus(channelList, 3, statusList);
    EXPECT_EQ(ret, HCCL_E_NETWORK);
}

// statusList 已预标记 RES_* 的通道被 continue 跳过, 计入 failCount 且整批无网络类失败时返回 UNAVAIL
TEST_F(TestChannelProcess, Ut_ChannelGetStatus_When_StatusPreMarkedResUnavail_SkipGetStatus)
{
    auto chReady = std::make_shared<FakeStatusChannel>(hcomm::ChannelStatus::READY);
    auto chUnavail = std::make_shared<FakeStatusChannel>(hcomm::ChannelStatus::RES_RMT_UNAVAIL);
    ChannelHandle hReady = RegisterFakeChannel(chReady);
    ChannelHandle hUnavail = RegisterFakeChannel(chUnavail);
    ChannelHandle channelList[2] = {hReady, hUnavail};
    int32_t statusList[2] = {0, static_cast<int32_t>(hcomm::ChannelStatus::RES_RMT_UNAVAIL)};
    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = hcomm::ChannelProcess::ChannelGetStatus(channelList, 2, statusList);
    EXPECT_EQ(ret, HCCL_E_UNAVAIL);
    // 被跳过通道的状态保持预标记值不变
    EXPECT_EQ(statusList[1], static_cast<int32_t>(hcomm::ChannelStatus::RES_RMT_UNAVAIL));
}

// 资源不足终态与仍在连接的通道并存时, 先返回 E_AGAIN, 不提前报 UNAVAIL
TEST_F(TestChannelProcess, Ut_ChannelGetStatus_When_ResUnavailAndConnecting_Expect_AGAIN)
{
    auto chReady = std::make_shared<FakeStatusChannel>(hcomm::ChannelStatus::READY);
    auto chUnavail = std::make_shared<FakeStatusChannel>(hcomm::ChannelStatus::RES_LOC_UNAVAIL);
    auto chInit = std::make_shared<FakeStatusChannel>(hcomm::ChannelStatus::INIT);
    ChannelHandle hReady = RegisterFakeChannel(chReady);
    ChannelHandle hUnavail = RegisterFakeChannel(chUnavail);
    ChannelHandle hInit = RegisterFakeChannel(chInit);
    ChannelHandle channelList[3] = {hReady, hUnavail, hInit};
    int32_t statusList[3] = {0, 0, 0};
    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = hcomm::ChannelProcess::ChannelGetStatus(channelList, 3, statusList);
    EXPECT_EQ(ret, HCCL_E_AGAIN);
}

// TC-CP-014: ConvertToLinkStatus 终态映射契约, 资源不足状态映射到对应 link status 码
TEST_F(TestChannelProcess, Ut_ConvertToLinkStatus_When_ResUnavail_Expect_CorrectMapping)
{
    std::vector<hcomm::ChannelStatus> internalStatus = {
        hcomm::ChannelStatus::RES_LOC_UNAVAIL, hcomm::ChannelStatus::RES_RMT_UNAVAIL, hcomm::ChannelStatus::READY,
        hcomm::ChannelStatus::FAILED,          hcomm::ChannelStatus::SOCKET_TIMEOUT,
    };
    std::vector<int32_t> linkStatusList(internalStatus.size(), -1);
    hcomm::ConvertToLinkStatus(internalStatus, linkStatusList);

    EXPECT_EQ(linkStatusList[0], hcomm::HCOMM_CHANNEL_STATUS_RES_LOC_UNAVAIL);
    EXPECT_EQ(linkStatusList[1], hcomm::HCOMM_CHANNEL_STATUS_RES_RMT_UNAVAIL);
    // 回归既有映射, 确保新增资源不足分支不破坏默认行为
    EXPECT_EQ(linkStatusList[2], hcomm::HCOMM_CHANNEL_STATUS_READY);
    EXPECT_EQ(linkStatusList[3], hcomm::HCOMM_CHANNEL_STATUS_FAILED);
    EXPECT_EQ(linkStatusList[4], hcomm::HCOMM_CHANNEL_STATUS_TIMEOUT);
    // 显式校验资源不足码值, 锁定 STC V2.0 契约
    EXPECT_EQ(hcomm::HCOMM_CHANNEL_STATUS_RES_LOC_UNAVAIL, 4);
    EXPECT_EQ(hcomm::HCOMM_CHANNEL_STATUS_RES_RMT_UNAVAIL, 5);
}

// ChannelGetDesc: channelDesc 空指针返回 E_PTR
TEST_F(TestChannelProcess, Ut_ChannelGetDesc_When_DescNullptr_Return_HCCL_E_PTR)
{
    HcclResult ret = hcomm::ChannelProcess::ChannelGetDesc((ChannelHandle)0x1, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// ChannelGetDesc: handle 未注册时返回错误
TEST_F(TestChannelProcess, Ut_ChannelGetDesc_When_HandleNotFound_Return_Error)
{
    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));
    HcommChannelDesc desc{};
    HcclResult ret = hcomm::ChannelProcess::ChannelGetDesc((ChannelHandle)0x1234, &desc);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

// ChannelGetDesc: 正常返回 channel 内保存的 desc
TEST_F(TestChannelProcess, Ut_ChannelGetDesc_When_Normal_Return_ChannelDesc)
{
    auto channel = std::make_shared<FakeStatusChannel>(hcomm::ChannelStatus::READY);
    HcommChannelDesc expectDesc{};
    expectDesc.role = HCOMM_SOCKET_ROLE_SERVER;
    expectDesc.socket = reinterpret_cast<HcommSocket>(0xABCD);
    channel->SetChannelDesc(expectDesc);
    ChannelHandle handle = RegisterFakeChannel(channel);
    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));

    HcommChannelDesc desc{};
    HcclResult ret = hcomm::ChannelProcess::ChannelGetDesc(handle, &desc);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(desc.role, HCOMM_SOCKET_ROLE_SERVER);
    EXPECT_EQ(desc.socket, reinterpret_cast<HcommSocket>(0xABCD));
}
