/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../ut_hcomm_base.h"
#include "cpu_roce_endpoint.h"
#include "cpu_urma_endpoint.h"
#include "ub_reged_mem_mgr.h"
#include "hccp.h"
#include "hcomm_c_adpt.h"
#include "hcomm_res.h"
#include "nic_plugin_manager.h"

using namespace hcomm;

namespace {
struct FakePluginEndpointState {
    bool inited = false;
};

struct FakePluginEndpointScenario {
    uint32_t createEndpointCalls = 0;
    uint32_t initEndpointCalls = 0;
    uint32_t destroyEndpointCalls = 0;
    uint32_t registerMemoryCalls = 0;
    uint32_t unregisterMemoryCalls = 0;
    uint32_t memoryExportCalls = 0;
    uint32_t memoryImportCalls = 0;
    uint32_t memoryUnimportCalls = 0;
    uint32_t getListenPortCalls = 0;
};

FakePluginEndpointScenario g_fakeEpScenario;

int32_t FakeEpInit(void* ctx)
{
    g_fakeEpScenario.initEndpointCalls++;
    static_cast<FakePluginEndpointState*>(ctx)->inited = true;
    return HCCL_SUCCESS;
}

int32_t FakeEpDestroy(void* ctx)
{
    g_fakeEpScenario.destroyEndpointCalls++;
    delete static_cast<FakePluginEndpointState*>(ctx);
    return HCCL_SUCCESS;
}

int32_t FakeEpRegisterMemory(void* ctx, const CommMem* mem, const char* tag, void** handle)
{
    g_fakeEpScenario.registerMemoryCalls++;
    *handle = reinterpret_cast<void*>(0x12345678);
    return HCCL_SUCCESS;
}

int32_t FakeEpUnregisterMemory(void* ctx, void* handle)
{
    g_fakeEpScenario.unregisterMemoryCalls++;
    return HCCL_SUCCESS;
}

int32_t FakeEpMemoryExport(void* ctx, void* handle, void** desc, uint32_t* descLen)
{
    g_fakeEpScenario.memoryExportCalls++;
    static char fakeDesc[8] = {};
    *desc = fakeDesc;
    *descLen = sizeof(fakeDesc);
    return HCCL_SUCCESS;
}

int32_t FakeEpMemoryImport(void* ctx, const void* desc, uint32_t descLen, CommMem* outMem)
{
    g_fakeEpScenario.memoryImportCalls++;
    outMem->type = COMM_MEM_TYPE_HOST;
    outMem->addr = const_cast<void*>(desc);
    outMem->size = descLen;
    return HCCL_SUCCESS;
}

int32_t FakeEpMemoryUnimport(void* ctx, const void* desc, uint32_t descLen)
{
    g_fakeEpScenario.memoryUnimportCalls++;
    return HCCL_SUCCESS;
}

int32_t FakeEpGetListenPort(void* ctx, uint32_t* port)
{
    g_fakeEpScenario.getListenPortCalls++;
    *port = 12345;
    return HCCL_SUCCESS;
}

int32_t FakeEpCreateEndpoint(const EndpointDesc* endpointDesc, void** outCtx, HcommNicEndpointOps** outOps);

HcommNicEndpointOps g_fakeEpOps = {
    {HCOMM_NIC_ENDPOINT_OPS_VERSION, HCOMM_NIC_ENDPOINT_OPS_MAGIC_WORD, sizeof(HcommNicEndpointOps), 0},
    FakeEpInit,             // init
    FakeEpDestroy,          // destroy
    FakeEpRegisterMemory,   // registerMemory
    FakeEpUnregisterMemory, // unregisterMemory
    FakeEpMemoryExport,     // memoryExport
    FakeEpMemoryImport,     // memoryImport
    FakeEpMemoryUnimport,   // memoryUnimport
    FakeEpGetListenPort,    // getListenPort
};

HcommNicEndpointOps g_unsupportedEpOps = {
    {HCOMM_NIC_ENDPOINT_OPS_VERSION, HCOMM_NIC_ENDPOINT_OPS_MAGIC_WORD, sizeof(HcommNicEndpointOps), 0},
    nullptr,       // init
    FakeEpDestroy, // destroy — FillDefaultEndpointOps 要求必须实现
    nullptr,       // registerMemory
    nullptr,       // unregisterMemory
    nullptr,       // memoryExport
    nullptr,       // memoryImport
    nullptr,       // memoryUnimport
    nullptr,       // getListenPort
};

HcommNicPluginInfo g_fakePluginInfo
    = {{HCOMM_NIC_PLUGIN_INFO_VERSION, HCOMM_NIC_PLUGIN_INFO_MAGIC_WORD, sizeof(HcommNicPluginInfo), 0},
       "fake_ep_plugin",
       1,
       {COMM_PROTOCOL_ROCE}};

int32_t FakeEpCreateEndpoint(const EndpointDesc* endpointDesc, void** outCtx, HcommNicEndpointOps** outOps)
{
    g_fakeEpScenario.createEndpointCalls++;
    *outCtx = new FakePluginEndpointState();
    *outOps = &g_fakeEpOps;
    return HCCL_SUCCESS;
}

int32_t FakeEpCreateUnsupportedEndpoint(const EndpointDesc* endpointDesc, void** outCtx, HcommNicEndpointOps** outOps)
{
    g_fakeEpScenario.createEndpointCalls++;
    *outCtx = new FakePluginEndpointState();
    *outOps = &g_unsupportedEpOps;
    return HCCL_SUCCESS;
}

NicPluginEntry g_fakeEpPluginEntry = {nullptr, &g_fakePluginInfo, FakeEpCreateEndpoint, nullptr};

void ResetFakeEpScenario() { g_fakeEpScenario = {}; }

// ---- Builtin 测试辅助变量：用于手动设置 mock 未覆盖的输出参数 ----
static int g_builtinDummyHandle = 0;
static char g_builtinDummyDesc[8] = {};

using RoceRegisterMemoryFp = HcclResult (hcomm::RoceRegedMemMgr::*)(const HcommMem*, const char*, void**);
using RoceUnregisterMemoryFp = HcclResult (hcomm::RoceRegedMemMgr::*)(void*);
using RoceMemoryExportFp = HcclResult (hcomm::RoceRegedMemMgr::*)(const EndpointDesc&, void*, void**, uint32_t*);
using RemoteMemoryImportFp = HcclResult (hcomm::RemoteRegedMemMgr::*)(const void*, uint32_t, HcommMem*);
using RemoteMemoryUnimportFp = HcclResult (hcomm::RemoteRegedMemMgr::*)(const void*, uint32_t);
using GetListenPortFp = HcclResult (hcomm::ServerSocketContext::*)(uint32_t*);
} // namespace

class TestHcommEndpoint : public TestHcommCAdptBase {
public:
    void SetUp() override
    {
        TestHcommCAdptBase::SetUp();
        ResetFakeEpScenario();
    }
    void TearDown() override { TestHcommCAdptBase::TearDown(); }
};

class StubRoceRegedMemMgr : public hcomm::RoceRegedMemMgr {
public:
    StubRoceRegedMemMgr() : hcomm::RoceRegedMemMgr(nullptr, false) {}
    ~StubRoceRegedMemMgr() override = default;
    HcclResult RegisterMemory(const HcommMem*, const char*, void**) override { return HCCL_SUCCESS; }
    HcclResult UnregisterMemory(void*) override { return HCCL_SUCCESS; }
    HcclResult MemoryExport(const EndpointDesc&, void*, void**, uint32_t*) override { return HCCL_SUCCESS; }
};

class StubEndpointRemoteRegedMemMgr : public hcomm::RemoteRegedMemMgr {
public:
    HcclResult MemoryImport(const void*, uint32_t, HcommMem*) override { return HCCL_SUCCESS; }
    HcclResult MemoryUnimport(const void*, uint32_t) override { return HCCL_SUCCESS; }
};

TEST_F(TestHcommEndpoint, Ut_TestHcommEndpointCreate_When_EndpointNullptr_Return_HCCL_E_PTR)
{
    EndpointHandle handle;
    HcommResult ret = HcommEndpointCreate(nullptr, &handle);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcommEndpoint, Ut_TestHcommEndpointCreate_When_HandleNullptr_Return_HCCL_E_PTR)
{
    EndpointDesc endpointDesc;
    endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;

    HcommResult ret = HcommEndpointCreate(&endpointDesc, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcommEndpoint, Ut_TestHcommEndpointDestroy_When_HandleNullptr_Return_HCCL_E_NOT_FOUND)
{
    HcommResult ret = HcommEndpointDestroy(nullptr);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(TestHcommEndpoint, Ut_TestHcommEndpointGet_When_EndpointNullptr_Return_HCCL_E_PTR)
{
    HcommResult ret = HcommEndpointGet(nullptr, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcommEndpoint, Ut_PluginEndpoint_Expect_DispatchToPlugin)
{
    MOCKER((hcomm::FindHostNicPlugin))
        .stubs()
        .will(returnValue(static_cast<const hcomm::NicPluginEntry*>(&g_fakeEpPluginEntry)));

    EndpointDesc endpointDesc{};
    endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    endpointDesc.protocol = COMM_PROTOCOL_ROCE;
    EndpointHandle epHandle = nullptr;
    EXPECT_EQ(HcommEndpointCreate(&endpointDesc, &epHandle), HCCL_SUCCESS);
    EXPECT_EQ(g_fakeEpScenario.createEndpointCalls, 1U);

    // mem ops
    CommMem mem{COMM_MEM_TYPE_HOST, const_cast<char*>("test"), 4};
    HcommMemHandle memHandle = nullptr;
    EXPECT_EQ(HcommMemReg(epHandle, "tag", &mem, &memHandle), HCCL_SUCCESS);
    EXPECT_EQ(g_fakeEpScenario.registerMemoryCalls, 1U);

    EXPECT_EQ(HcommMemUnreg(epHandle, memHandle), HCCL_SUCCESS);
    EXPECT_EQ(g_fakeEpScenario.unregisterMemoryCalls, 1U);

    void* memDesc = nullptr;
    uint32_t memDescLen = 0;
    EXPECT_EQ(HcommMemExport(epHandle, memHandle, &memDesc, &memDescLen), HCCL_SUCCESS);
    EXPECT_EQ(g_fakeEpScenario.memoryExportCalls, 1U);

    CommMem outMem{};
    EXPECT_EQ(HcommMemImport(epHandle, memDesc, memDescLen, &outMem), HCCL_SUCCESS);
    EXPECT_EQ(g_fakeEpScenario.memoryImportCalls, 1U);

    EXPECT_EQ(HcommMemUnimport(epHandle, memDesc, memDescLen), HCCL_SUCCESS);
    EXPECT_EQ(g_fakeEpScenario.memoryUnimportCalls, 1U);

    // listen port
    uint32_t port = 0;
    EXPECT_EQ(HcommEndpointGetListenPort(epHandle, &port), HCCL_SUCCESS);
    EXPECT_EQ(g_fakeEpScenario.getListenPortCalls, 1U);
    EXPECT_EQ(port, 12345U);

    EXPECT_EQ(HcommEndpointDestroy(epHandle), HCCL_SUCCESS);
    EXPECT_EQ(g_fakeEpScenario.destroyEndpointCalls, 1U);
}

TEST_F(TestHcommEndpoint, Ut_PluginEndpoint_Expect_DispatchToPlugin_NotSupport)
{
    static NicPluginEntry unsupportedEntry = {nullptr, &g_fakePluginInfo, FakeEpCreateUnsupportedEndpoint, nullptr};
    MOCKER((hcomm::FindHostNicPlugin))
        .stubs()
        .will(returnValue(static_cast<const hcomm::NicPluginEntry*>(&unsupportedEntry)));

    EndpointDesc endpointDesc{};
    endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    endpointDesc.protocol = COMM_PROTOCOL_ROCE;
    EndpointHandle epHandle = nullptr;
    EXPECT_EQ(HcommEndpointCreate(&endpointDesc, &epHandle), HCCL_SUCCESS);

    CommMem mem{COMM_MEM_TYPE_HOST, const_cast<char*>("test"), 4};
    HcommMemHandle memHandle = nullptr;
    EXPECT_EQ(HcommMemReg(epHandle, "tag", &mem, &memHandle), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(HcommMemUnreg(epHandle, memHandle), HCCL_E_NOT_SUPPORT);

    void* memDesc = nullptr;
    uint32_t memDescLen = 0;
    EXPECT_EQ(HcommMemExport(epHandle, memHandle, &memDesc, &memDescLen), HCCL_E_NOT_SUPPORT);

    CommMem outMem{};
    EXPECT_EQ(HcommMemImport(epHandle, memDesc, memDescLen, &outMem), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(HcommMemUnimport(epHandle, memDesc, memDescLen), HCCL_E_NOT_SUPPORT);

    uint32_t port = 0;
    EXPECT_EQ(HcommEndpointGetListenPort(epHandle, &port), HCCL_E_NOT_SUPPORT);

    EXPECT_EQ(HcommEndpointDestroy(epHandle), HCCL_SUCCESS);
}

TEST_F(TestHcommEndpoint, Ut_BuiltinEndpoint_Expect_DispatchToBuiltinOps)
{
    EndpointDesc endpointDesc{};
    endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    endpointDesc.protocol = COMM_PROTOCOL_ROCE;
    EndpointHandle epHandle = nullptr;
    EXPECT_EQ(HcommEndpointCreate(&endpointDesc, &epHandle), HCCL_SUCCESS);

    CpuRoceEndpoint* ep = nullptr;
    ASSERT_EQ(HcommEndpointGet(epHandle, reinterpret_cast<void**>(&ep)), HCCL_SUCCESS);
    ASSERT_NE(ep, nullptr);

    // 内存方法从 Endpoint 移至 mgr；本端操作经 GetLocalRegMemMgr()、远端操作经 GetRemoteRegMemMgr() 访问。
    // 此处分别 mock Local/Remote 接口的虚方法，验证 HcommMemReg 等正确转发到对应 mgr。
    ep->localMemMgr_ = std::make_shared<StubRoceRegedMemMgr>();
    ep->remoteMemMgr_ = std::make_shared<StubEndpointRemoteRegedMemMgr>();
    hcomm::RoceRegedMemMgr* roceMgr = static_cast<hcomm::RoceRegedMemMgr*>(ep->GetLocalRegMemMgr());
    hcomm::RemoteRegedMemMgr* remoteMgr = ep->GetRemoteRegMemMgr();
    MOCKER_CPP_VIRTUAL(*roceMgr, &hcomm::RoceRegedMemMgr::RegisterMemory, RoceRegisterMemoryFp)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(*roceMgr, &hcomm::RoceRegedMemMgr::UnregisterMemory, RoceUnregisterMemoryFp)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(*roceMgr, &hcomm::RoceRegedMemMgr::MemoryExport, RoceMemoryExportFp)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(*remoteMgr, &hcomm::RemoteRegedMemMgr::MemoryImport, RemoteMemoryImportFp)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(*remoteMgr, &hcomm::RemoteRegedMemMgr::MemoryUnimport, RemoteMemoryUnimportFp)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    // 归属记账为 endpoint 自身行为，且 mock 的 RegisterMemory 不写回 memHandle，非本用例验证目标；
    // mockcpp 要求对象类型与成员所属类一致，故取基类指针
    Endpoint* epBase = ep;
    MOCKER_CPP_VIRTUAL(epBase, &hcomm::Endpoint::RecordLocalMemHandle).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(epBase, &hcomm::Endpoint::RemoveLocalMemHandle).stubs().will(returnValue(HCCL_SUCCESS));
    // Socket 监听接口由 ServerSocketContext 承载，mock 上下文虚方法验证转发路径
    MOCKER_CPP_VIRTUAL(
        *ep->GetServerSocketContext(), &hcomm::ServerSocketContext::ServerSocketGetListenPort, GetListenPortFp)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));

    // ---- 业务接口全覆盖 ----

    // 1. registerMemory（mock 返回成功，手动设置 memHandle）
    CommMem mem{COMM_MEM_TYPE_HOST, const_cast<char*>("test"), 4};
    HcommMemHandle memHandle = nullptr;
    EXPECT_EQ(HcommMemReg(epHandle, "tag", &mem, &memHandle), HCCL_SUCCESS);
    memHandle = reinterpret_cast<HcommMemHandle>(&g_builtinDummyHandle);

    // 2. unregisterMemory
    EXPECT_EQ(HcommMemUnreg(epHandle, memHandle), HCCL_SUCCESS);

    // 3. memoryExport（mock 返回成功，手动设置 desc）
    void* memDesc = nullptr;
    uint32_t memDescLen = 0;
    EXPECT_EQ(HcommMemExport(epHandle, memHandle, &memDesc, &memDescLen), HCCL_SUCCESS);
    memDesc = g_builtinDummyDesc;
    memDescLen = sizeof(g_builtinDummyDesc);

    // 4. memoryImport
    CommMem outMem{};
    EXPECT_EQ(HcommMemImport(epHandle, memDesc, memDescLen, &outMem), HCCL_SUCCESS);

    // 5. memoryUnimport
    EXPECT_EQ(HcommMemUnimport(epHandle, memDesc, memDescLen), HCCL_SUCCESS);

    // 6. getListenPort
    uint32_t port = 0;
    EXPECT_EQ(HcommEndpointGetListenPort(epHandle, &port), HCCL_SUCCESS);

    EXPECT_EQ(HcommEndpointDestroy(epHandle), HCCL_SUCCESS);
}

// 核心场景：同 desc 两次创建 endpoint，经进程级 mgr 缓存共享同一 RoceRegedMemMgr；
// 各实例枚举只含本实例注册的 handle（归属隔离），Unreg 后仅从本实例摘除。
TEST_F(TestHcommEndpoint, Ut_TwoBuiltinEndpoints_When_SharedLocalMgr_Expect_HandlesIsolatedAndUnregRemoved)
{
    EndpointDesc endpointDesc{};
    endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    endpointDesc.protocol = COMM_PROTOCOL_ROCE;
    EndpointHandle epHandle1 = nullptr;
    EndpointHandle epHandle2 = nullptr;
    ASSERT_EQ(HcommEndpointCreate(&endpointDesc, &epHandle1), HCCL_SUCCESS);
    ASSERT_EQ(HcommEndpointCreate(&endpointDesc, &epHandle2), HCCL_SUCCESS);

    CpuRoceEndpoint* ep1 = nullptr;
    CpuRoceEndpoint* ep2 = nullptr;
    ASSERT_EQ(HcommEndpointGet(epHandle1, reinterpret_cast<void**>(&ep1)), HCCL_SUCCESS);
    ASSERT_EQ(HcommEndpointGet(epHandle2, reinterpret_cast<void**>(&ep2)), HCCL_SUCCESS);
    // 同缓存 key 命中同一进程级 mgr
    ASSERT_EQ(ep1->GetLocalRegMemMgr(), ep2->GetLocalRegMemMgr());

    // HOST 内存走 legacy 注册链，仅 MR 硬件边界打桩
    void* fakeMr = reinterpret_cast<void*>(0xABCDEF);
    MOCKER(RaRegisterMr)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&fakeMr, sizeof(fakeMr)))
        .will(returnValue(0));
    MOCKER(RaDeregisterMr).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(0));

    CommMem mem{COMM_MEM_TYPE_HOST, reinterpret_cast<void*>(0x5000), 2048};
    HcommMemHandle h1 = nullptr;
    HcommMemHandle h2 = nullptr;
    ASSERT_EQ(HcommMemReg(epHandle1, "ep1", &mem, &h1), HCCL_SUCCESS);
    ASSERT_EQ(HcommMemReg(epHandle2, "ep2", &mem, &h2), HCCL_SUCCESS);
    ASSERT_NE(h1, nullptr);
    ASSERT_NE(h2, nullptr);
    ASSERT_NE(h1, h2);

    // 归属记录按 shared_ptr 数组布局返回（与 channel 消费契约同构），验证只含本实例 handle
    void* handles1 = nullptr;
    uint32_t num1 = 0;
    ASSERT_EQ(HcommMemGetAllMemHandles(epHandle1, &handles1, &num1), HCCL_SUCCESS);
    ASSERT_EQ(num1, 1U);
    auto* records1 = static_cast<std::shared_ptr<Hccl::LocalRdmaRmaBuffer>*>(handles1);
    EXPECT_EQ(static_cast<void*>(records1[0].get()), h1);

    void* handles2 = nullptr;
    uint32_t num2 = 0;
    ASSERT_EQ(HcommMemGetAllMemHandles(epHandle2, &handles2, &num2), HCCL_SUCCESS);
    ASSERT_EQ(num2, 1U);
    auto* records2 = static_cast<std::shared_ptr<Hccl::LocalRdmaRmaBuffer>*>(handles2);
    EXPECT_EQ(static_cast<void*>(records2[0].get()), h2);

    // ep1 注销后仅本实例摘除，ep2 不受影响
    ASSERT_EQ(HcommMemUnreg(epHandle1, h1), HCCL_SUCCESS);
    ASSERT_EQ(HcommMemGetAllMemHandles(epHandle1, &handles1, &num1), HCCL_SUCCESS);
    EXPECT_EQ(num1, 0U);
    EXPECT_EQ(handles1, nullptr);
    ASSERT_EQ(HcommMemGetAllMemHandles(epHandle2, &handles2, &num2), HCCL_SUCCESS);
    ASSERT_EQ(num2, 1U);
    records2 = static_cast<std::shared_ptr<Hccl::LocalRdmaRmaBuffer>*>(handles2);
    EXPECT_EQ(static_cast<void*>(records2[0].get()), h2);

    ASSERT_EQ(HcommMemUnreg(epHandle2, h2), HCCL_SUCCESS);
    ASSERT_EQ(HcommMemGetAllMemHandles(epHandle2, &handles2, &num2), HCCL_SUCCESS);
    EXPECT_EQ(num2, 0U);

    EXPECT_EQ(HcommEndpointDestroy(epHandle1), HCCL_SUCCESS);
    EXPECT_EQ(HcommEndpointDestroy(epHandle2), HCCL_SUCCESS);
}

// Ub 系链路覆盖：同 desc 两次创建 CpuUrmaEndpoint，共享同一 UbRegedMemMgr；
// 各实例枚举只含本实例注册的 handle，Unreg 后仅从本实例摘除。
TEST_F(TestHcommEndpoint, Ut_TwoCpuUrmaEndpoints_When_SharedLocalMgr_Expect_HandlesIsolatedAndUnregRemoved)
{
    EndpointDesc endpointDesc{};
    endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    endpointDesc.protocol = COMM_PROTOCOL_UBC_TP;
    EndpointHandle epHandle1 = nullptr;
    EndpointHandle epHandle2 = nullptr;
    ASSERT_EQ(HcommEndpointCreate(&endpointDesc, &epHandle1), HCCL_SUCCESS);
    ASSERT_EQ(HcommEndpointCreate(&endpointDesc, &epHandle2), HCCL_SUCCESS);

    CpuUrmaEndpoint* ep1 = nullptr;
    CpuUrmaEndpoint* ep2 = nullptr;
    ASSERT_EQ(HcommEndpointGet(epHandle1, reinterpret_cast<void**>(&ep1)), HCCL_SUCCESS);
    ASSERT_EQ(HcommEndpointGet(epHandle2, reinterpret_cast<void**>(&ep2)), HCCL_SUCCESS);
    // 同缓存 key 命中同一进程级 mgr
    ASSERT_EQ(ep1->GetLocalRegMemMgr(), ep2->GetLocalRegMemMgr());

    // HOST 内存走 legacy 注册链，仅 MR 硬件边界打桩
    void* fakeMr = reinterpret_cast<void*>(0xABCDEF);
    MOCKER(RaRegisterMr)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&fakeMr, sizeof(fakeMr)))
        .will(returnValue(0));
    MOCKER(RaDeregisterMr).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(0));

    CommMem mem{COMM_MEM_TYPE_HOST, reinterpret_cast<void*>(0x5000), 2048};
    HcommMemHandle h1 = nullptr;
    HcommMemHandle h2 = nullptr;
    ASSERT_EQ(HcommMemReg(epHandle1, "ep1", &mem, &h1), HCCL_SUCCESS);
    ASSERT_EQ(HcommMemReg(epHandle2, "ep2", &mem, &h2), HCCL_SUCCESS);
    ASSERT_NE(h1, nullptr);
    ASSERT_NE(h2, nullptr);
    ASSERT_NE(h1, h2);

    // 归属记录按 shared_ptr 数组布局返回，验证只含本实例 handle
    void* handles1 = nullptr;
    uint32_t num1 = 0;
    ASSERT_EQ(HcommMemGetAllMemHandles(epHandle1, &handles1, &num1), HCCL_SUCCESS);
    ASSERT_EQ(num1, 1U);
    auto* records1 = static_cast<std::shared_ptr<Hccl::LocalUbRmaBuffer>*>(handles1);
    EXPECT_EQ(static_cast<void*>(records1[0].get()), h1);

    void* handles2 = nullptr;
    uint32_t num2 = 0;
    ASSERT_EQ(HcommMemGetAllMemHandles(epHandle2, &handles2, &num2), HCCL_SUCCESS);
    ASSERT_EQ(num2, 1U);
    auto* records2 = static_cast<std::shared_ptr<Hccl::LocalUbRmaBuffer>*>(handles2);
    EXPECT_EQ(static_cast<void*>(records2[0].get()), h2);

    // ep1 注销后仅本实例摘除，ep2 不受影响
    ASSERT_EQ(HcommMemUnreg(epHandle1, h1), HCCL_SUCCESS);
    ASSERT_EQ(HcommMemGetAllMemHandles(epHandle1, &handles1, &num1), HCCL_SUCCESS);
    EXPECT_EQ(num1, 0U);
    EXPECT_EQ(handles1, nullptr);
    ASSERT_EQ(HcommMemGetAllMemHandles(epHandle2, &handles2, &num2), HCCL_SUCCESS);
    ASSERT_EQ(num2, 1U);
    records2 = static_cast<std::shared_ptr<Hccl::LocalUbRmaBuffer>*>(handles2);
    EXPECT_EQ(static_cast<void*>(records2[0].get()), h2);

    ASSERT_EQ(HcommMemUnreg(epHandle2, h2), HCCL_SUCCESS);
    ASSERT_EQ(HcommMemGetAllMemHandles(epHandle2, &handles2, &num2), HCCL_SUCCESS);
    EXPECT_EQ(num2, 0U);

    EXPECT_EQ(HcommEndpointDestroy(epHandle1), HCCL_SUCCESS);
    EXPECT_EQ(HcommEndpointDestroy(epHandle2), HCCL_SUCCESS);
}

// 错误路径：RegisterMemory mock 成功但不产生真实记录，FindHandleRecord 反查必然失败；
// 断言 HcommMemReg 返回 HCCL_E_NOT_FOUND，且回滚恰好注销一次硬件 MR。
TEST_F(TestHcommEndpoint, Ut_HcommMemReg_When_FindHandleRecordMiss_Expect_NotFoundAndRollback)
{
    EndpointDesc endpointDesc{};
    endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    endpointDesc.protocol = COMM_PROTOCOL_ROCE;
    EndpointHandle epHandle = nullptr;
    ASSERT_EQ(HcommEndpointCreate(&endpointDesc, &epHandle), HCCL_SUCCESS);

    CpuRoceEndpoint* ep = nullptr;
    ASSERT_EQ(HcommEndpointGet(epHandle, reinterpret_cast<void**>(&ep)), HCCL_SUCCESS);
    auto* localMgr = ep->GetLocalRegMemMgr();
    ASSERT_NE(localMgr, nullptr);
    // 不 mock Endpoint::RecordLocalMemHandle，让真实记账链路走 FindHandleRecord 失败分支；
    // RegisterMemory mock 经 outBoundP 写出非空哑句柄，绕过空指针防护直达反查
    void* dummyHandle = reinterpret_cast<void*>(0xBEEF);
    MOCKER_CPP_VIRTUAL(localMgr, &hcomm::LocalRegedMemMgr::RegisterMemory)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&dummyHandle, sizeof(dummyHandle)))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(localMgr, &hcomm::LocalRegedMemMgr::UnregisterMemory)
        .expects(once())
        .will(returnValue(HCCL_SUCCESS));

    CommMem mem{COMM_MEM_TYPE_HOST, const_cast<char*>("test"), 4};
    HcommMemHandle memHandle = nullptr;
    EXPECT_EQ(HcommMemReg(epHandle, "tag", &mem, &memHandle), HCCL_E_NOT_FOUND);

    EXPECT_EQ(HcommEndpointDestroy(epHandle), HCCL_SUCCESS);
}
