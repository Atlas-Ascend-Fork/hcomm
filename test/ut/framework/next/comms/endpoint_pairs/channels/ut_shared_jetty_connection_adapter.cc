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
#include <stdexcept>

#define private public
#define protected public
#include "shared_jetty_connection_adapter.h"
#include "endpoint.h"
#include "urma_endpoint.h"
#include "comm_queue_context/jetty_context.h"
#include "dev_ub_connection.h"
#include "ut_shared_jetty_test_helper.h"
#undef private
#undef protected

using namespace hcomm;
using namespace Hccl;

namespace hcomm {
// 适配层 .cc 内部具名函数（无公共头声明，防内联）：UT 经此 extern 声明 mock 注入构造失败
std::function<HcclResult()> BuildTpReleaseCallback(Hccl::DevUbConnection::TpOwnership ownership);
} // namespace hcomm

namespace {
// 与 dev_ub_connection.cc 中 OPBASED_UB_SQ_DEPTH_MAX 对齐（构造函数把 UB_SQ_DEPTH_NOT_SET 转换为此值）
constexpr uint32_t DEFAULT_OPBASE_UB_SQ_DEPTH = 8192U;

class StubEndpointForAdapt : public UrmaEndpoint {
public:
    StubEndpointForAdapt() : UrmaEndpoint(MakeDesc())
    {
        // GetCommQueueContext 已上移为 Endpoint 基类虚函数（stub 经 UrmaEndpoint override 生效）；
        // 此处预创建供测试断言，与 GetCommQueueContext() 首调的延迟创建效果等价。
        jettyContext_ = std::make_unique<JettyContext>();
    }

    HcclResult Init() override { return HCCL_SUCCESS; }

private:
    static EndpointDesc MakeDesc()
    {
        EndpointDesc desc{};
        Hccl::IpAddress localIp("127.0.0.1");
        desc.protocol = COMM_PROTOCOL_UBC_CTP;
        desc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
        desc.commAddr.addr = localIp.GetBinaryAddress().addr;
        desc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
        desc.loc.device.devPhyId = 3;
        return desc;
    }
};

JettyContext::Ctx MakeTestCtx()
{
    JettyContext::Ctx ctx;
    ctx.handle = 0x1234;
    ctx.handlePtr = reinterpret_cast<void*>(0x5678);
    ctx.jettyId = 100;
    ctx.sqBuffVa = 0xABCDEF00;
    ctx.dbAddr = 0x12345678;
    ctx.keySize = Hccl::HRT_UB_QP_KEY_MAX_LEN;
    for (uint32_t i = 0; i < ctx.keySize; ++i) {
        ctx.localQpKey[i] = static_cast<uint8_t>(i);
    }
    ctx.sqDepth = 32;
    ctx.jfcHandle = 0xFCEULL;
    ctx.cqInfo.va = 0xC0FFEEULL;
    ctx.cqInfo.id = 0x1111U;
    ctx.cqInfo.cqDepth = 64U;
    ctx.localPsn = 846930886U;
    return ctx;
}

// ReleaseUbConnectionTp 实参捕获（invoke 桩需要普通函数指针，状态走文件级变量）：
// 校验移交回调以快照实参释放且恰好一次，防止 TpOwnership 字段拷贝漂移导致 useCnt 扣错
struct ReleaseTpCaptured {
    int32_t devLogicId{0};
    IpAddress locAddr{};
    IpAddress rmtAddr{};
    Hccl::TpProtocol tpProtocol{Hccl::TpProtocol::INVALID};
    uint32_t qos{0};
    uint64_t tpHandle{0};
    uint32_t calls{0};
} g_releaseTpCaptured;

HcclResult CaptureReleaseUbConnectionTp(
    int32_t devLogicId, const IpAddress& locAddr, const IpAddress& rmtAddr, Hccl::TpProtocol tpProtocol, TpInfo& tpInfo,
    uint32_t qos)
{
    g_releaseTpCaptured.devLogicId = devLogicId;
    g_releaseTpCaptured.locAddr = locAddr;
    g_releaseTpCaptured.rmtAddr = rmtAddr;
    g_releaseTpCaptured.tpProtocol = tpProtocol;
    g_releaseTpCaptured.qos = qos;
    g_releaseTpCaptured.tpHandle = tpInfo.tpHandle;
    g_releaseTpCaptured.calls++;
    return HCCL_SUCCESS;
}

// 构造失败注入桩：模拟 BuildTpReleaseCallback 捕获 IpAddress 堆分配 bad_alloc 抛异常
std::function<HcclResult()> ThrowingTpCallbackBuilder(Hccl::DevUbConnection::TpOwnership)
{
    throw std::runtime_error("mock callback build throw");
}
} // namespace

class SharedJettyConnAdaptTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
    }
    static void TearDownTestCase()
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
    }
    void SetUp() override
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
    }
    void TearDown() override
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
    }
};

TEST_F(SharedJettyConnAdaptTest, Ut_Inject_When_NullConnection_Expect_HCCL_E_PARA)
{
    JettyContext::Ctx ctx = MakeTestCtx();
    auto releaseCb = [](void*) {};
    EXPECT_EQ(SetSharedJettyFieldsToConn(nullptr, ctx, reinterpret_cast<void*>(0x1), releaseCb), HCCL_E_PARA);
}

TEST_F(SharedJettyConnAdaptTest, Ut_Inject_When_NullEndpointTag_Expect_HCCL_E_PARA)
{
    auto conn = MakeTestJettyConnection();
    JettyContext::Ctx ctx = MakeTestCtx();
    auto releaseCb = [](void*) {};
    EXPECT_EQ(SetSharedJettyFieldsToConn(conn.get(), ctx, nullptr, releaseCb), HCCL_E_PARA);
}

TEST_F(SharedJettyConnAdaptTest, Ut_Inject_When_Normal_Expect_Success)
{
    auto conn = MakeTestJettyConnection(DevUbConnection::JettyMode::EXTERNAL_INJECT);
    StubEndpointForAdapt endpoint;
    JettyContext::Ctx ctx = MakeTestCtx();
    auto releaseCb = [](void*) {};
    EXPECT_EQ(SetSharedJettyFieldsToConn(conn.get(), ctx, &endpoint, releaseCb), HCCL_SUCCESS);
    EXPECT_EQ(conn->jettyMode_, DevUbConnection::JettyMode::EXTERNAL_INJECT);
    EXPECT_EQ(conn->jettyHandle, static_cast<Hccl::JettyHandle>(0x1234));
    EXPECT_EQ(conn->jettyId, 100U);
    EXPECT_EQ(conn->sqBuffVa, 0xABCDEF00ULL);
    EXPECT_EQ(conn->sqDepth, 32U);
    EXPECT_EQ(conn->endpointTag_, static_cast<void*>(&endpoint));
    EXPECT_EQ(conn->jfcHandle, ctx.jfcHandle);
    EXPECT_EQ(conn->cqInfo_.va, ctx.cqInfo.va);
    EXPECT_EQ(conn->cqInfo_.id, ctx.cqInfo.id);
    EXPECT_EQ(conn->cqInfo_.cqDepth, ctx.cqInfo.cqDepth);
    EXPECT_EQ(conn->jettyImportCfg.localPsn, ctx.localPsn);
}

TEST_F(SharedJettyConnAdaptTest, Ut_Inject_When_ZeroKeySize_Expect_Success)
{
    auto conn = MakeTestJettyConnection(DevUbConnection::JettyMode::EXTERNAL_INJECT);
    StubEndpointForAdapt endpoint;
    JettyContext::Ctx ctx = MakeTestCtx();
    ctx.keySize = 0;
    auto releaseCb = [](void*) {};
    EXPECT_EQ(SetSharedJettyFieldsToConn(conn.get(), ctx, &endpoint, releaseCb), HCCL_SUCCESS);
}

TEST_F(SharedJettyConnAdaptTest, Ut_ExtractInfo_When_NullConnection_Expect_HCCL_E_PARA)
{
    JettyContext::Ctx ctx;
    EXPECT_EQ(ExtractJettyInfoFromConn(nullptr, ctx), HCCL_E_PARA);
}

TEST_F(SharedJettyConnAdaptTest, Ut_ExtractInfo_When_Normal_Expect_Success)
{
    auto conn = MakeTestJettyConnection(DevUbConnection::JettyMode::EXTERNAL_INJECT);
    StubEndpointForAdapt endpoint;
    JettyContext::Ctx injectCtx = MakeTestCtx();
    auto releaseCb = [](void*) {};
    ASSERT_EQ(SetSharedJettyFieldsToConn(conn.get(), injectCtx, &endpoint, releaseCb), HCCL_SUCCESS);

    // 设置 JFC 相关字段，验证 ExtractJettyInfoFromConn 能正确提取
    conn->rdmaHandle = reinterpret_cast<void*>(0xABCD);
    conn->jfcHandle = 12345;

    JettyContext::Ctx extractCtx;
    EXPECT_EQ(ExtractJettyInfoFromConn(conn.get(), extractCtx), HCCL_SUCCESS);
    EXPECT_EQ(extractCtx.handle, injectCtx.handle);
    EXPECT_EQ(extractCtx.jettyId, injectCtx.jettyId);
    EXPECT_EQ(extractCtx.sqBuffVa, injectCtx.sqBuffVa);
    EXPECT_EQ(extractCtx.dbAddr, injectCtx.dbAddr);
    EXPECT_EQ(extractCtx.keySize, injectCtx.keySize);
    EXPECT_EQ(extractCtx.sqDepth, injectCtx.sqDepth);
    EXPECT_EQ(extractCtx.rdmaHandle, reinterpret_cast<void*>(0xABCD));
    EXPECT_EQ(extractCtx.jfcHandle, 12345U);
    EXPECT_EQ(extractCtx.cqInfo.va, conn->cqInfo_.va);
    EXPECT_EQ(extractCtx.cqInfo.id, conn->cqInfo_.id);
    EXPECT_EQ(extractCtx.localPsn, injectCtx.localPsn);
    EXPECT_EQ(memcmp(extractCtx.localQpKey, injectCtx.localQpKey, Hccl::HRT_UB_QP_KEY_MAX_LEN), 0);
}

TEST_F(SharedJettyConnAdaptTest, Ut_ExtractInfo_When_EmptyConnection_Expect_DefaultSqDepth)
{
    auto conn = MakeTestJettyConnection();
    JettyContext::Ctx ctx;
    EXPECT_EQ(ExtractJettyInfoFromConn(conn.get(), ctx), HCCL_SUCCESS);
    EXPECT_EQ(ctx.handle, static_cast<Hccl::JettyHandle>(0));
    EXPECT_EQ(ctx.jettyId, 0U);
    EXPECT_EQ(ctx.sqDepth, DEFAULT_OPBASE_UB_SQ_DEPTH);
    EXPECT_EQ(ctx.rdmaHandle, nullptr);
    EXPECT_EQ(ctx.jfcHandle, 0U);
}

TEST_F(SharedJettyConnAdaptTest, Ut_TransferOwnership_When_NullConnection_Expect_HCCL_E_PARA)
{
    EXPECT_EQ(DetachConnJetty(nullptr), HCCL_E_PARA);
}

TEST_F(SharedJettyConnAdaptTest, Ut_TransferOwnership_When_Normal_Expect_Success)
{
    auto conn = MakeTestJettyConnection();
    EXPECT_FALSE(conn->jettyDetached_);
    EXPECT_EQ(DetachConnJetty(conn.get()), HCCL_SUCCESS);
    EXPECT_TRUE(conn->jettyDetached_);
}

TEST_F(SharedJettyConnAdaptTest, Ut_RoundTrip_InjectExtractTransfer_Expect_Consistent)
{
    auto conn = MakeTestJettyConnection(DevUbConnection::JettyMode::EXTERNAL_INJECT);
    StubEndpointForAdapt endpoint;
    JettyContext::Ctx injectCtx = MakeTestCtx();
    auto releaseCb = [](void*) {};
    ASSERT_EQ(SetSharedJettyFieldsToConn(conn.get(), injectCtx, &endpoint, releaseCb), HCCL_SUCCESS);

    conn->rdmaHandle = reinterpret_cast<void*>(0xBEEF);
    conn->jfcHandle = 67890;

    ASSERT_EQ(DetachConnJetty(conn.get()), HCCL_SUCCESS);

    JettyContext::Ctx extractCtx;
    ASSERT_EQ(ExtractJettyInfoFromConn(conn.get(), extractCtx), HCCL_SUCCESS);
    EXPECT_EQ(extractCtx.handle, injectCtx.handle);
    EXPECT_EQ(extractCtx.handlePtr, injectCtx.handlePtr);
    EXPECT_EQ(extractCtx.jettyId, injectCtx.jettyId);
    EXPECT_EQ(extractCtx.sqBuffVa, injectCtx.sqBuffVa);
    EXPECT_EQ(extractCtx.dbAddr, injectCtx.dbAddr);
    EXPECT_EQ(extractCtx.sqDepth, injectCtx.sqDepth);
    EXPECT_EQ(extractCtx.keySize, injectCtx.keySize);
    EXPECT_EQ(extractCtx.rdmaHandle, reinterpret_cast<void*>(0xBEEF));
    EXPECT_EQ(extractCtx.jfcHandle, 67890U);
    EXPECT_EQ(extractCtx.localPsn, injectCtx.localPsn);
}

// ============ DetachConnTpOwnership（TP 引用移交） =====
TEST_F(SharedJettyConnAdaptTest, Ut_DetachTpOwnership_When_NullConnection_Expect_HCCL_E_PARA)
{
    // 契约：所有失败路径（含空指针）outReleaseTp 一致置空，
    // 调用方复用已持有回调的 std::function 时不残留过期回调
    std::function<HcclResult()> releaseTp = []() {
        return HCCL_SUCCESS;
    };
    EXPECT_EQ(DetachConnTpOwnership(nullptr, releaseTp), HCCL_E_PARA);
    EXPECT_TRUE(releaseTp == nullptr);
}

TEST_F(SharedJettyConnAdaptTest, Ut_DetachTpOwnership_When_NoTpHandle_Expect_EmptyCallback)
{
    // tpHandle=0：无有效 TP 引用可移交，回调置空，releaseTpOnDestroy_ 保持 true（旧行为不变）
    auto conn = MakeTestJettyConnection();
    std::function<HcclResult()> releaseTp = nullptr;
    EXPECT_EQ(DetachConnTpOwnership(conn.get(), releaseTp), HCCL_SUCCESS);
    EXPECT_TRUE(releaseTp == nullptr);
    EXPECT_TRUE(conn->releaseTpOnDestroy_);
}

TEST_F(SharedJettyConnAdaptTest, Ut_DetachTpOwnership_When_ValidTpHandle_Expect_CallbackSetAndDestroySkipRelease)
{
    // tpHandle 有效：返回释放回调、析构不再 ReleaseTp（releaseTpOnDestroy_=false）；
    // ownership 字段经 TpOwnership 快照导出，构造的 locAddr/rmtAddr/qos 与 connection 一致
    auto conn = MakeTestJettyConnection();
    conn->tpProtocol = Hccl::TpProtocol::CTP;
    conn->tpInfo.tpHandle = 0x1234567890ABCDEFULL;
    std::function<HcclResult()> releaseTp = nullptr;
    EXPECT_EQ(DetachConnTpOwnership(conn.get(), releaseTp), HCCL_SUCCESS);
    EXPECT_TRUE(releaseTp != nullptr);
    EXPECT_FALSE(conn->releaseTpOnDestroy_);

    // 回调真实执行（stub 环境 ReleaseUbConnectionTp 恒成功）：结果传回供调用方审计
    EXPECT_EQ(releaseTp(), HCCL_SUCCESS);
}

TEST_F(SharedJettyConnAdaptTest, Ut_DetachTpOwnership_When_ValidTpHandle_Expect_CallbackReleasesWithSnapshotArgs)
{
    // 校验回调以移交快照实参调用 ReleaseUbConnectionTp 且恰好一次：devLogicId/地址对/协议/qos
    // 与 connection 成员一致、tpHandle 为移交值——快照拷错不会崩溃，只会表现为 useCnt 扣错或
    // 缓存条目永不回收，须以实参断言固化
    auto conn = MakeTestJettyConnection();
    conn->tpProtocol = Hccl::TpProtocol::CTP;
    conn->tpInfo.tpHandle = 0x1234567890ABCDEFULL;
    std::function<HcclResult()> releaseTp = nullptr;
    ASSERT_EQ(DetachConnTpOwnership(conn.get(), releaseTp), HCCL_SUCCESS);

    g_releaseTpCaptured = {}; // 防御性重置捕获状态（文件级变量可能被其他用例置位）
    MOCKER(Hccl::ReleaseUbConnectionTp).stubs().will(invoke(CaptureReleaseUbConnectionTp));
    EXPECT_EQ(releaseTp(), HCCL_SUCCESS);
    GlobalMockObject::verify();

    EXPECT_EQ(g_releaseTpCaptured.calls, 1U);
    EXPECT_EQ(g_releaseTpCaptured.devLogicId, conn->devLogicId);
    EXPECT_TRUE(g_releaseTpCaptured.locAddr == conn->locAddr);
    EXPECT_TRUE(g_releaseTpCaptured.rmtAddr == conn->rmtAddr);
    EXPECT_EQ(g_releaseTpCaptured.tpProtocol, conn->tpProtocol);
    EXPECT_EQ(g_releaseTpCaptured.qos, static_cast<uint32_t>(conn->qos_));
    EXPECT_EQ(g_releaseTpCaptured.tpHandle, 0x1234567890ABCDEFULL);
}

TEST_F(SharedJettyConnAdaptTest, Ut_DetachTpOwnership_When_AlreadyDetached_Expect_NoSecondCallback)
{
    // 幂等防御：重复移交时 DetachTpOwnership 返回 NOT_FOUND，适配层翻译为 SUCCESS 且回调为空，
    // 不产出第二个释放回调（避免 TpManager useCnt 重复扣减）
    auto conn = MakeTestJettyConnection();
    conn->tpInfo.tpHandle = 0x1234U;
    std::function<HcclResult()> first = nullptr;
    EXPECT_EQ(DetachConnTpOwnership(conn.get(), first), HCCL_SUCCESS);
    EXPECT_TRUE(first != nullptr);

    std::function<HcclResult()> second = nullptr;
    EXPECT_EQ(DetachConnTpOwnership(conn.get(), second), HCCL_SUCCESS);
    EXPECT_TRUE(second == nullptr);
    EXPECT_FALSE(conn->releaseTpOnDestroy_);
}

TEST_F(SharedJettyConnAdaptTest, Ut_DetachTpOwnership_When_RealError_Expect_Propagate)
{
    // 真实错误码不被适配层吞掉：上抛给 helper 告警/兜底（区别于 NOT_FOUND 的旧行为），
    // 失败时 releaseTpOnDestroy_ 未翻转，退化为临时 connection 析构释放的旧行为
    auto conn = MakeTestJettyConnection();
    MOCKER_CPP(&Hccl::DevUbConnection::DetachTpOwnership).stubs().will(returnValue(HcclResult::HCCL_E_INTERNAL));
    std::function<HcclResult()> releaseTp = nullptr;
    EXPECT_EQ(DetachConnTpOwnership(conn.get(), releaseTp), HCCL_E_INTERNAL);
    EXPECT_TRUE(releaseTp == nullptr);
    GlobalMockObject::verify();
}

TEST_F(SharedJettyConnAdaptTest, Ut_DetachTpOwnership_When_CallbackBuildThrows_Expect_RestoreAndError)
{
    // 回调构造失败（bad_alloc 窗口）注入：BuildTpReleaseCallback 抛异常时，适配层须还原
    // releaseTpOnDestroy_（否则 TP 引用既不随析构释放也无回调接管，useCnt 永久漏扣）、
    // 置空回调并以 HCCL_E_INTERNAL 返回（不抛出，避免异常穿出 Acquire 令 creating 卡死）
    auto conn = MakeTestJettyConnection();
    conn->tpProtocol = Hccl::TpProtocol::CTP;
    conn->tpInfo.tpHandle = 0x1234567890ABCDEFULL;

    MOCKER(hcomm::BuildTpReleaseCallback).stubs().will(invoke(ThrowingTpCallbackBuilder));
    std::function<HcclResult()> releaseTp = nullptr;
    EXPECT_EQ(DetachConnTpOwnership(conn.get(), releaseTp), HCCL_E_INTERNAL);
    EXPECT_TRUE(releaseTp == nullptr);
    // DetachTpOwnership 成功已翻转标志，构造失败后经 Restore 还原为析构释放（旧行为回退）
    EXPECT_TRUE(conn->releaseTpOnDestroy_);
    GlobalMockObject::verify();

    // 回滚后可重新移交：真实构造路径恢复，回调正常产出
    std::function<HcclResult()> releaseTp2 = nullptr;
    EXPECT_EQ(DetachConnTpOwnership(conn.get(), releaseTp2), HCCL_SUCCESS);
    EXPECT_TRUE(releaseTp2 != nullptr);
    EXPECT_FALSE(conn->releaseTpOnDestroy_);
}
