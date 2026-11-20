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

#include <atomic>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <vector>

#include "adapter_rts.h"
#include "adapter_error_manager_pub.h"
#include "coll_comm.h"
#include "coll_comm_config.h"
#include "env_net_layer_config.h"
#include "hcom_common.h"
#include "hcomm_c_adpt.h"
#include "common/result_defs_pub.h"
#include "my_rank.h"
#include "op_base.h"
#include "coll_comm_mgr.h"
#include "hccl_group.h"

extern thread_local s32 g_hcclDeviceId;

// 当前测试目标不链接 V2 创建实现，为弱符号提供可打桩的函数实体。
// 未设置 mock 时显式失败，避免把未执行创建误判为成功。
HcclResult HcclCommInitRootInfoConfigV2(uint32_t, const HcclRootInfo*, uint32_t, const HcclCommConfig*, HcclComm*)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult HcclCommInitClusterInfoMemConfigV2(const char*, uint32_t, HcclCommConfig*, HcclComm*)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult HcclCommInitClusterInfoConfigV2(const char*, uint32_t, HcclCommConfig*, HcclComm*)
{
    return HCCL_E_NOT_SUPPORT;
}

namespace {
const HcclCommConfig* g_forwardedConfig = nullptr;
std::unique_ptr<hccl::hcclComm> g_forwardedComm;
std::shared_ptr<hccl::hcclAsyncJob> g_queuedConfigJob;
HcclResult (*g_queuedConfigCallback)(hccl::hcclAsyncJob*) = nullptr;
std::string g_expectedNetLayers;

HcclResult
CaptureConfigJob(std::shared_ptr<hccl::hcclAsyncJob> job, HcclResult (*callback)(hccl::hcclAsyncJob*), HcclComm*)
{
    g_queuedConfigJob = std::move(job);
    g_queuedConfigCallback = callback;
    return HCCL_SUCCESS;
}

HcclResult ObserveRootConfig(uint32_t, const HcclRootInfo*, uint32_t, const HcclCommConfig* config, HcclComm* comm)
{
    g_forwardedConfig = config;
    EXPECT_EQ(config->hcclDeterministic, HCCL_COMM_DETERMINISTIC_CONFIG_NOT_SET);
    EXPECT_EQ(config->hcclOpExpansionMode, 0U);
    EXPECT_EQ(config->hcclUdi[0], '\0');
    EXPECT_STREQ(config->hccl_use_net_layer, g_expectedNetLayers.c_str());
    *comm = reinterpret_cast<HcclComm>(0x1);
    return HCCL_SUCCESS;
}

HcclResult ObserveCollConfig(uint32_t rank, void** commV2, const HcclCommConfig* config, HcclComm* comm)
{
    EXPECT_EQ(config, g_forwardedConfig);
    // 后续入口会设置建域来源，桩也需返回持有 CollComm 的有效对象。
    g_forwardedComm = std::make_unique<hccl::hcclComm>();
    g_forwardedComm->collComm_ = std::make_unique<hccl::CollComm>(
        *commV2, rank, "config_forwarding", hccl::ManagerCallbacks{}, hccl::CollCommInitMode::simpleMode);
    *comm = g_forwardedComm.get();
    return HCCL_SUCCESS;
}

HcclResult ObserveRankTableConfig(const char*, uint32_t rank, HcclCommConfig* config, HcclComm* comm)
{
    return ObserveRootConfig(1, nullptr, rank, config, comm);
}

void ObserveNullRankTableConfigError(std::string code, std::vector<std::string> keys, std::vector<std::string> values)
{
    EXPECT_EQ(code, "EI0003");
    EXPECT_EQ(keys, std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}));
    EXPECT_EQ(
        values, std::vector<std::string>({"HcclCommInitClusterInfoConfig", "nullptr", "config", "non-null pointer"}));
}
} // namespace

class VersionedConfigForwardingTest : public testing::Test {
protected:
    enum class Entry { ROOT_INFO, RANK_TABLE, MEMORY, SCALABLE };

    HcclResult Init(Entry entry, HcclCommConfig* config)
    {
        switch (entry) {
            case Entry::ROOT_INFO:
                return HcclCommInitRootInfoConfig(1, &root_, 0, config, &comm_);
            case Entry::RANK_TABLE:
                return HcclCommInitClusterInfoConfig("unused.json", 0, config, &comm_);
            case Entry::MEMORY:
                return HcclCommInitClusterInfoMemConfig("{}", 0, config, &comm_);
            case Entry::SCALABLE:
                return HcclCommInitRootInfoScalable(1, 1, &root_, 0, 0, config, &comm_);
        }
        return HCCL_E_PARA;
    }

    const HcclCommConfig* QueuedConfig(Entry entry)
    {
        if (!g_queuedConfigJob) {
            return nullptr;
        }
        switch (entry) {
            case Entry::ROOT_INFO:
                return std::static_pointer_cast<hccl::hcclCommInitConfigAsyncJob>(g_queuedConfigJob)->config;
            case Entry::RANK_TABLE:
                return std::static_pointer_cast<hccl::hcclCommInitRankTableConfigAsyncJob>(g_queuedConfigJob)->config;
            case Entry::SCALABLE:
                return std::static_pointer_cast<hccl::hcclCommInitScalableAsyncJob>(g_queuedConfigJob)->config;
            default:
                return nullptr;
        }
    }

    void SetUp() override
    {
        hccl::EnvNetLayerConfig::GetInstance().ResetParsed();
        HcclCommConfigInit(&config_);
        savedDepth_ = hcclGroupDepth;
        hcclGroupDepth = 0;
        g_forwardedConfig = nullptr;
        g_queuedConfigJob.reset();
        g_queuedConfigCallback = nullptr;
        g_expectedNetLayers.clear();
        const char* environment = std::getenv("HCCL_USE_NET_LAYER");
        hadEnvironment_ = environment != nullptr;
        environment_ = hadEnvironment_ ? environment : "";
        ASSERT_EQ(unsetenv("HCCL_USE_NET_LAYER"), 0);
        MOCKER(&hccl::hcclComm::UnRegistTaskAbortHandler).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER(HcclDeviceRefresh).stubs().with(outBound(s32{0})).will(returnValue(HCCL_SUCCESS));
        MOCKER(hrtGetHcclV2Support)
            .stubs()
            .with(outBoundP(&supportsV2_, sizeof(supportsV2_)))
            .will(returnValue(HCCL_SUCCESS));
    }
    void TearDown() override
    {
        hccl::EnvNetLayerConfig::GetInstance().ResetParsed();
        hcclGroupDepth = savedDepth_;
        g_queuedConfigJob.reset();
        g_forwardedComm.reset();
        comm_ = nullptr;
        g_forwardedConfig = nullptr;
        g_queuedConfigCallback = nullptr;
        if (hadEnvironment_) {
            EXPECT_EQ(setenv("HCCL_USE_NET_LAYER", environment_.c_str(), 1), 0);
        } else {
            EXPECT_EQ(unsetenv("HCCL_USE_NET_LAYER"), 0);
        }
        GlobalMockObject::verify();
    }
    s32 savedDepth_ = 0;
    bool supportsV2_ = true;
    bool hadEnvironment_ = false;
    std::string environment_;
    HcclRootInfo root_{};
    HcclComm comm_ = nullptr;
    HcclCommConfig config_{};
};

TEST_F(VersionedConfigForwardingTest, RootInfoVersionZeroUsesSameNormalizedConfigForBothConsumers)
{
    CommConfigInfo input{sizeof(CommConfigInfo), HCCL_COMM_CONFIG_MAGIC_WORD, 0U, {0}};
    MOCKER(HcclCommInitRootInfoConfigV2).expects(once()).will(invoke(ObserveRootConfig));
    MOCKER(HcclCommInitCollComm).expects(once()).will(invoke(ObserveCollConfig));
    EXPECT_EQ(Init(Entry::ROOT_INFO, reinterpret_cast<HcclCommConfig*>(&input)), HCCL_SUCCESS);
    EXPECT_EQ(input.configSize, sizeof(CommConfigInfo));
    EXPECT_EQ(input.version, 0U);
}

TEST_F(VersionedConfigForwardingTest, InvalidConfigRejectedBeforeConsumersAndEnqueue)
{
    alignas(HcclCommConfig) size_t sizeOnly = sizeof(size_t);
    auto* config = reinterpret_cast<HcclCommConfig*>(&sizeOnly);
    CommConfigInfo badMagic{sizeof(CommConfigInfo), 0U, 0U, {0}};
    auto* badCfg = reinterpret_cast<HcclCommConfig*>(&badMagic);
    MOCKER(hccl::commInitTaskAppend).expects(never());
    MOCKER(HcclCommInitRootInfoConfigV2).expects(never());
    MOCKER(HcclCommInitClusterInfoMemConfigV2).expects(never());
    MOCKER(HcclCommInitClusterInfoConfigV2).expects(never());
    MOCKER(HcclCommInitCollComm).expects(never());
    // RANK_TABLE 入口的空配置需先上报 EI0003，同步与异步路径各一次。
    MOCKER(RptInputErr).expects(exactly(2)).will(invoke(ObserveNullRankTableConfigError));
    struct Case {
        s32 depth;
        Entry entry;
        HcclCommConfig* input;
        HcclResult expected;
    };
    const Case cases[] = {
        {0, Entry::ROOT_INFO, config, HCCL_E_PARA},  {0, Entry::MEMORY, config, HCCL_E_PARA},
        {0, Entry::RANK_TABLE, config, HCCL_E_PARA}, {1, Entry::ROOT_INFO, config, HCCL_E_PARA},
        {1, Entry::RANK_TABLE, config, HCCL_E_PARA}, {1, Entry::SCALABLE, config, HCCL_E_PARA},
        {0, Entry::ROOT_INFO, nullptr, HCCL_E_PTR},  {1, Entry::ROOT_INFO, nullptr, HCCL_E_PTR},
        {0, Entry::RANK_TABLE, nullptr, HCCL_E_PTR}, {1, Entry::RANK_TABLE, nullptr, HCCL_E_PTR},
        {0, Entry::MEMORY, nullptr, HCCL_E_PTR},     {0, Entry::SCALABLE, nullptr, HCCL_E_PTR},
        {0, Entry::ROOT_INFO, badCfg, HCCL_E_PARA},  {1, Entry::ROOT_INFO, badCfg, HCCL_E_PARA},
        {0, Entry::RANK_TABLE, badCfg, HCCL_E_PARA}, {1, Entry::RANK_TABLE, badCfg, HCCL_E_PARA},
        {0, Entry::MEMORY, badCfg, HCCL_E_PARA},     {1, Entry::SCALABLE, badCfg, HCCL_E_PARA},
        {1, Entry::SCALABLE, nullptr, HCCL_E_PTR},
    };
    for (const auto& item : cases) {
        SCOPED_TRACE(testing::Message() << item.depth << ":" << static_cast<int>(item.entry) << ":" << item.input);
        hcclGroupDepth = item.depth;
        EXPECT_EQ(Init(item.entry, item.input), item.expected);
    }
}

TEST_F(VersionedConfigForwardingTest, GroupedCreationOwnsVersionZeroConfigAfterInputExpires)
{
    hcclGroupDepth = 1;
    MOCKER(hccl::commInitTaskAppend).expects(exactly(3)).will(invoke(CaptureConfigJob));
    for (Entry entry : {Entry::ROOT_INFO, Entry::RANK_TABLE, Entry::SCALABLE}) {
        SCOPED_TRACE(static_cast<int>(entry));
        {
            CommConfigInfo input{sizeof(CommConfigInfo), HCCL_COMM_CONFIG_MAGIC_WORD, 0U, {0}};
            ASSERT_EQ(Init(entry, reinterpret_cast<HcclCommConfig*>(&input)), HCCL_SUCCESS);
            input.magicWord = 0;
        }
        const HcclCommConfig* snapshot = QueuedConfig(entry);
        ASSERT_NE(snapshot, nullptr);
        CommConfigInfo header{};
        ASSERT_EQ(memcpy_s(&header, sizeof(header), snapshot->reserved, sizeof(header)), EOK);
        EXPECT_EQ(header.magicWord, HCCL_COMM_CONFIG_MAGIC_WORD);
        EXPECT_EQ(header.configSize, sizeof(HcclCommConfig));
        EXPECT_EQ(header.version, 0U); // 异步副本保留输入版本，不再整体提升版本。
        EXPECT_EQ(snapshot->hcclDeterministic, HCCL_COMM_DETERMINISTIC_CONFIG_NOT_SET);
    }
}

TEST_F(VersionedConfigForwardingTest, ConfiguredEntriesForwardEnvironmentAndHonorParameterPriority)
{
    MOCKER(HcclCommInitRootInfoConfigV2).expects(exactly(2)).will(invoke(ObserveRootConfig));
    MOCKER(HcclCommInitClusterInfoConfigV2).expects(exactly(2)).will(invoke(ObserveRankTableConfig));
    MOCKER(HcclCommInitClusterInfoMemConfigV2).expects(exactly(2)).will(invoke(ObserveRankTableConfig));
    MOCKER(HcclCommInitCollComm).expects(exactly(6)).will(invoke(ObserveCollConfig));
    for (bool useParameter : {false, true}) {
        ASSERT_EQ(setenv("HCCL_USE_NET_LAYER", useParameter ? "invalid" : "not(0,0)", 1), 0);
        if (useParameter) {
            ASSERT_EQ(strcpy_s(config_.hccl_use_net_layer, sizeof(config_.hccl_use_net_layer), "2,1,2"), EOK);
        }
        g_expectedNetLayers = useParameter ? "1,2" : "not(0)";
        for (Entry entry : {Entry::ROOT_INFO, Entry::RANK_TABLE, Entry::MEMORY}) {
            SCOPED_TRACE(testing::Message() << useParameter << ":" << static_cast<int>(entry));
            EXPECT_EQ(Init(entry, &config_), HCCL_SUCCESS);
        }
        EXPECT_STREQ(config_.hccl_use_net_layer, useParameter ? "2,1,2" : "");
    }
}

TEST_F(VersionedConfigForwardingTest, QueuedConfigFreezesEnvironmentIncludingDefaultBeforeExecution)
{
    hcclGroupDepth = 1;
    MOCKER(hccl::commInitTaskAppend).expects(exactly(4)).will(invoke(CaptureConfigJob));
    MOCKER(hrtSetDevice).stubs().will(returnValue(HCCL_SUCCESS));
    const char* socName = "Ascend950PR_958b";
    MOCKER(aclrtGetSocName).stubs().will(returnValue(socName));
    MOCKER(HcclCommInitRootInfoConfigV2).expects(exactly(2)).will(invoke(ObserveRootConfig));
    MOCKER(HcclCommInitClusterInfoConfigV2).expects(exactly(2)).will(invoke(ObserveRankTableConfig));
    MOCKER(HcclCommInitCollComm).expects(exactly(4)).will(invoke(ObserveCollConfig));
    for (const char* expression : {"", "not(0)"}) {
        hccl::EnvNetLayerConfig::GetInstance().ResetParsed();
        for (Entry entry : {Entry::ROOT_INFO, Entry::RANK_TABLE}) {
            SCOPED_TRACE(testing::Message() << expression << ":" << static_cast<int>(entry));
            ASSERT_EQ(setenv("HCCL_USE_NET_LAYER", expression, 1), 0);
            {
                HcclCommConfig config;
                HcclCommConfigInit(&config);
                ASSERT_EQ(Init(entry, &config), HCCL_SUCCESS);
            }
            ASSERT_NE(g_queuedConfigJob, nullptr);
            ASSERT_NE(g_queuedConfigCallback, nullptr);
            ASSERT_EQ(setenv("HCCL_USE_NET_LAYER", "invalid", 1), 0);
            g_expectedNetLayers = expression;
            EXPECT_EQ(g_queuedConfigCallback(g_queuedConfigJob.get()), HCCL_SUCCESS);
        }
    }
}

TEST_F(VersionedConfigForwardingTest, InvalidOrUnsupportedSelectionRejectedBeforeDispatchOrEnqueue)
{
    MOCKER(hccl::commInitTaskAppend).expects(never());
    MOCKER(HcclCommInitRootInfoConfigV2).expects(never());
    MOCKER(HcclCommInitClusterInfoConfigV2).expects(never());
    MOCKER(HcclCommInitClusterInfoMemConfigV2).expects(never());
    // 配置或环境任一处表达式非法，都在下发/入队前失败。
    for (bool invalidParameter : {false, true}) {
        ASSERT_EQ(setenv("HCCL_USE_NET_LAYER", invalidParameter ? "2" : "invalid", 1), 0);
        if (invalidParameter) {
            ASSERT_EQ(strcpy_s(config_.hccl_use_net_layer, sizeof(config_.hccl_use_net_layer), "invalid"), EOK);
        }
        for (s32 depth : {0, 1}) {
            hcclGroupDepth = depth;
            for (Entry entry : {Entry::ROOT_INFO, Entry::RANK_TABLE, Entry::MEMORY, Entry::SCALABLE}) {
                SCOPED_TRACE(testing::Message() << invalidParameter << ":" << depth << ":" << static_cast<int>(entry));
                EXPECT_EQ(Init(entry, &config_), HCCL_E_PARA);
            }
        }
    }

    // 暂不支持选层的入口：Scalable 与 InitAll 明确返回不支持。
    for (bool useParameter : {false, true}) {
        HcclCommConfigInit(&config_);
        ASSERT_EQ(setenv("HCCL_USE_NET_LAYER", useParameter ? "invalid" : "2", 1), 0);
        if (useParameter) {
            ASSERT_EQ(strcpy_s(config_.hccl_use_net_layer, sizeof(config_.hccl_use_net_layer), "2"), EOK);
        }
        for (s32 depth : {0, 1}) {
            hcclGroupDepth = depth;
            SCOPED_TRACE(testing::Message() << useParameter << ":" << depth);
            EXPECT_EQ(Init(Entry::SCALABLE, &config_), HCCL_E_NOT_SUPPORT);
        }
    }
    ASSERT_EQ(setenv("HCCL_USE_NET_LAYER", "2", 1), 0);
    int32_t device = 0;
    EXPECT_EQ(HcclCommInitAll(1, &device, &comm_), HCCL_E_NOT_SUPPORT);
    hccl::EnvNetLayerConfig::GetInstance().ResetParsed();
    ASSERT_EQ(setenv("HCCL_USE_NET_LAYER", "invalid", 1), 0);
    EXPECT_EQ(HcclCommInitAll(1, &device, &comm_), HCCL_E_PARA);
}

TEST_F(VersionedConfigForwardingTest, NoConfigEntriesUseConfiguredPathForEnvironmentSelection)
{
    ASSERT_EQ(setenv("HCCL_USE_NET_LAYER", "2", 1), 0);
    g_expectedNetLayers = "2";
    MOCKER(HcclCommInitRootInfoConfigV2).expects(once()).will(invoke(ObserveRootConfig));
    MOCKER(HcclCommInitClusterInfoConfigV2).expects(once()).will(invoke(ObserveRankTableConfig));
    MOCKER(HcclCommInitCollComm).expects(exactly(2)).will(invoke(ObserveCollConfig));
    EXPECT_EQ(HcclCommInitRootInfo(1, &root_, 0, &comm_), HCCL_SUCCESS);
    EXPECT_EQ(HcclCommInitClusterInfo("unused.json", 0, &comm_), HCCL_SUCCESS);
    // 无配置入口直接拒绝非法环境，不应先读取一份人为构造的空配置。
    MOCKER(hccl::ReadHcclCommConfig).expects(never());
    hccl::EnvNetLayerConfig::GetInstance().ResetParsed();
    ASSERT_EQ(setenv("HCCL_USE_NET_LAYER", "invalid", 1), 0);
    EXPECT_EQ(HcclCommInitRootInfo(1, &root_, 0, &comm_), HCCL_E_PARA);
    EXPECT_EQ(HcclCommInitClusterInfo("unused.json", 0, &comm_), HCCL_E_PARA);
}

TEST_F(VersionedConfigForwardingTest, NonV2IgnoresSelectionAndPreservesLegacyInitialization)
{
    GlobalMockObject::verify();
    supportsV2_ = false;
    MOCKER(HcclDeviceRefresh).stubs().with(outBound(s32{0})).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetHcclV2Support)
        .stubs()
        .with(outBoundP(&supportsV2_, sizeof(supportsV2_)))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtSetDevice).stubs().will(returnValue(HCCL_SUCCESS));
    const char* socName = "Ascend910";
    MOCKER(aclrtGetSocName).stubs().will(returnValue(socName));
    MOCKER(hccl::commInitTaskAppend).expects(exactly(3)).will(invoke(CaptureConfigJob));
    MOCKER(HcclCommInitRootInfoConfigV2).expects(never());
    MOCKER(HcclCommInitClusterInfoConfigV2).expects(never());
    MOCKER(HcclCommInitClusterInfoMemConfigV2).expects(never());
    MOCKER(HcclCommInitCollComm).expects(never());
    MOCKER(hccl::PrepareIndependentCommConfig).expects(never());
    // 用原流程的环境初始化失败作为到达标记，不在 UT 中建立真实通信域。
    MOCKER(InitExternalInput).expects(exactly(9)).will(returnValue(HCCL_E_INTERNAL));
    MOCKER(HcclGetCommAll).expects(once()).will(returnValue(HCCL_SUCCESS));
    ASSERT_EQ(setenv("HCCL_USE_NET_LAYER", "invalid", 1), 0);
    ASSERT_EQ(strcpy_s(config_.hccl_use_net_layer, sizeof(config_.hccl_use_net_layer), "invalid"), EOK);
    for (Entry entry : {Entry::ROOT_INFO, Entry::RANK_TABLE, Entry::MEMORY, Entry::SCALABLE}) {
        SCOPED_TRACE(static_cast<int>(entry));
        EXPECT_EQ(Init(entry, &config_), HCCL_E_PARA);
    }
    EXPECT_EQ(HcclCommInitRootInfo(1, &root_, 0, &comm_), HCCL_E_PARA);
    EXPECT_EQ(HcclCommInitClusterInfo("unused.json", 0, &comm_), HCCL_E_PARA);
    int32_t device = 0;
    EXPECT_EQ(HcclCommInitAll(1, &device, &comm_), HCCL_SUCCESS);
    hcclGroupDepth = 1;
    for (Entry entry : {Entry::ROOT_INFO, Entry::RANK_TABLE, Entry::SCALABLE}) {
        SCOPED_TRACE(static_cast<int>(entry));
        ASSERT_EQ(Init(entry, &config_), HCCL_SUCCESS);
        EXPECT_EQ(QueuedConfig(entry), &config_);
        ASSERT_NE(g_queuedConfigCallback, nullptr);
        EXPECT_EQ(g_queuedConfigCallback(g_queuedConfigJob.get()), HCCL_E_PARA);
    }
}

namespace {
constexpr uint32_t DEFAULT_MODE = 0;
constexpr uint32_t TEST_OP_EXPANSION_MODE = 1;

std::atomic<uint32_t> g_commNameIndex(0);
std::atomic<uint32_t> g_initCallCount(0);
std::atomic<uint32_t> g_topoInfoCallCount(0);
std::atomic<bool> g_failFirstTopoInfo(false);
std::mutex g_modeMutex;
std::vector<uint32_t> g_observedConfigModes;

HcclResult StubGetCommNameV2(HcclComm comm, char* commName)
{
    (void)comm;
    int32_t ret = snprintf_s(
        commName, Hccl::ROOTINFO_IDENTIFIER_MAX_LENGTH, Hccl::ROOTINFO_IDENTIFIER_MAX_LENGTH - 1, "coll_comm_%u",
        g_commNameIndex.fetch_add(1));
    return ret < 0 ? HCCL_E_INTERNAL : HCCL_SUCCESS;
}

HcclResult StubInitCollComm(
    hccl::hcclComm* comm, void* commV2, void* rankGraph, uint32_t userRank, HcclMem cclBuffer,
    const std::string& commName, const HcclCommConfig* config, hccl::CollCommInitMode initMode)
{
    (void)rankGraph;
    (void)cclBuffer;
    uint32_t callIndex = g_initCallCount.fetch_add(1);
    uint32_t configMode = config == nullptr ? DEFAULT_MODE : config->hcclOpExpansionMode;
    {
        std::lock_guard<std::mutex> lock(g_modeMutex);
        if (g_observedConfigModes.size() <= callIndex) {
            g_observedConfigModes.resize(callIndex + 1, DEFAULT_MODE);
        }
        g_observedConfigModes[callIndex] = configMode;
    }

    comm->collComm_ = std::make_unique<hccl::CollComm>(commV2, userRank, commName, hccl::ManagerCallbacks{}, initMode);
    comm->collComm_->myRank_ = std::make_shared<hccl::MyRank>(
        nullptr, userRank, comm->collComm_->GetCommConfig(), hccl::ManagerCallbacks{}, nullptr, nullptr);
    comm->collComm_->myRank_->opExpansionMode_ = configMode;
    return HCCL_SUCCESS;
}

HcclResult StubSetGroupTopoInfo(const char* group, uint32_t rankSize)
{
    (void)group;
    (void)rankSize;
    uint32_t callIndex = g_topoInfoCallCount.fetch_add(1);
    return g_failFirstTopoInfo.load() && callIndex == 0 ? HCCL_E_INTERNAL : HCCL_SUCCESS;
}
} // namespace

class HcomInitCollCommTest : public testing::Test {
protected:
    virtual void SetUp() override
    {
        const char* fakeA5SocName = "Ascend950PR_958b";
        MOCKER(aclrtGetSocName).stubs().will(returnValue(fakeA5SocName));
    }

    virtual void TearDown() override { GlobalMockObject::verify(); }
};

TEST_F(HcomInitCollCommTest, ut_HcomInitCollComm_When_Normal_Expect_ReturnIsHCCL_SUCCESS)
{
    HcomInfo& hcomInfo = HcomGetCtxHomInfo();
    void* commV2 = nullptr;
    MOCKER(&HcclGetCommNameV2).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(&HcclGetCclBuffer).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(&HcclGetRankGraphV2).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(&hccl::hcclComm::InitCollComm).stubs().will(returnValue(HCCL_SUCCESS));
    HcclResult ret = HcomInitCollComm(0, &commV2, hcomInfo.pComm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(hcomInfo.pComm, nullptr);
}

TEST_F(HcomInitCollCommTest, ut_HcomInitCollComm_When_commV2IsNullptr_Expect_ReturnIsHCCL_E_PARA)
{
    HcomInfo& hcomInfo = HcomGetCtxHomInfo();
    HcclResult ret = HcomInitCollComm(0, nullptr, hcomInfo.pComm);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// 看护空输出指针在解引用前被拒绝。
TEST_F(HcomInitCollCommTest, Ut_HcclCommInitCollComm_When_CommIsNullptr_Expect_ReturnHcclEPtr)
{
    void* commV2 = reinterpret_cast<void*>(0x1);
    EXPECT_EQ(HcclCommInitCollComm(0, &commV2, nullptr, nullptr), HCCL_E_PTR);
}

class HcclCommInitCollCommGuardTest : public testing::Test {
protected:
    void SetUp() override
    {
        g_hcclDeviceId = 0;
        g_commNameIndex.store(0);
        g_initCallCount.store(0);
        g_topoInfoCallCount.store(0);
        g_failFirstTopoInfo.store(false);
        {
            std::lock_guard<std::mutex> lock(g_modeMutex);
            g_observedConfigModes.clear();
        }

        const char* fakeA5SocName = "Ascend950PR_958b";
        MOCKER(aclrtGetSocName).stubs().will(returnValue(fakeA5SocName));
        MOCKER(hccl::GetMaxDevNum).stubs().with(outBound(8U)).will(returnValue(HCCL_SUCCESS));
        MOCKER(HcommResMgrInit).stubs().will(returnValue(static_cast<HcommResult>(HCOMM_SUCCESS)));
        MOCKER(&HcclGetCommNameV2).stubs().will(invoke(StubGetCommNameV2));
        MOCKER(&HcclGetCclBuffer).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER(&HcclGetRankGraphV2).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER(&hccl::hcclComm::InitCollComm).stubs().will(invoke(StubInitCollComm));
        MOCKER(HcomSetGroupTopoInfo).stubs().will(invoke(StubSetGroupTopoInfo));

        ResetOpBaseState();
    }

    void TearDown() override
    {
        ResetOpBaseState();
        GlobalMockObject::verify();
    }

    static HcclResult InitComm(const HcclCommConfig& config, HcclComm& comm)
    {
        void* commV2 = reinterpret_cast<void*>(0x1);
        return HcclCommInitCollComm(0, &commV2, &config, &comm);
    }

    static std::vector<uint32_t> GetObservedConfigModes()
    {
        std::lock_guard<std::mutex> lock(g_modeMutex);
        return g_observedConfigModes;
    }

    static void ResetOpBaseState()
    {
        HcclOpInfoCtx& opBaseHcom = hccl::CollCommMgr::GetInstance().LegacyGetHcclOpInfoCtx();
        std::lock_guard<std::mutex> lock(opBaseHcom.opGroupMapMutex);
        opBaseHcom.opGroup2CommMap.clear();
    }
};

// 看护 CollComm 初始化只读传递配置，不得修改调用者输入。
TEST_F(HcclCommInitCollCommGuardTest, Ut_HcclCommInitCollComm_When_UsesConstConfig_Expect_InputConfigUnchanged)
{
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    config.hcclOpExpansionMode = TEST_OP_EXPANSION_MODE;
    const HcclCommConfig constConfig = config;
    HcclComm comm = nullptr;

    ASSERT_EQ(InitComm(constConfig, comm), HCCL_SUCCESS);

    EXPECT_EQ(constConfig.hcclOpExpansionMode, TEST_OP_EXPANSION_MODE);
    const auto observedModes = GetObservedConfigModes();
    ASSERT_EQ(observedModes.size(), 1U);
    EXPECT_EQ(observedModes[0], TEST_OP_EXPANSION_MODE);
}

// 看护拓扑信息设置失败时清理输出句柄和通信域映射。
TEST_F(HcclCommInitCollCommGuardTest, Ut_HcclCommInitCollComm_When_SetGroupTopoInfoFails_Expect_CompleteRollback)
{
    g_failFirstTopoInfo.store(true);
    HcclCommConfig failedConfig;
    HcclCommConfigInit(&failedConfig);
    failedConfig.hcclOpExpansionMode = TEST_OP_EXPANSION_MODE;
    HcclComm failedComm = nullptr;

    EXPECT_EQ(InitComm(failedConfig, failedComm), HCCL_E_INTERNAL);
    EXPECT_EQ(failedComm, nullptr);
    HcclOpInfoCtx& opBaseHcom = hccl::CollCommMgr::GetInstance().LegacyGetHcclOpInfoCtx();
    {
        std::lock_guard<std::mutex> lock(opBaseHcom.opGroupMapMutex);
        EXPECT_TRUE(opBaseHcom.opGroup2CommMap.empty());
    }

    HcclCommConfig retryConfig;
    HcclCommConfigInit(&retryConfig);
    retryConfig.hcclOpExpansionMode = TEST_OP_EXPANSION_MODE;
    HcclComm retryComm = nullptr;
    ASSERT_EQ(InitComm(retryConfig, retryComm), HCCL_SUCCESS);

    const auto observedModes = GetObservedConfigModes();
    ASSERT_EQ(observedModes.size(), 2U);
    EXPECT_EQ(observedModes[0], TEST_OP_EXPANSION_MODE);
    EXPECT_EQ(observedModes[1], TEST_OP_EXPANSION_MODE);
}
