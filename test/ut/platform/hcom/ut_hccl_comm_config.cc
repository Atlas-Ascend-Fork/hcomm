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

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <array>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

#include <hccl/hccl_comm.h>
#include "securec.h"
#include "comm_config_pub.h"
#include "coll_comm_config.h"
#include "net_layer_test_common.h"

using namespace hccl;

namespace {

using net_layer_ut::Mode;
using net_layer_ut::NET_LAYER_ENV_NAME;
using net_layer_ut::NetLayerSelectionLoadTest;
using net_layer_ut::UNCHANGED_OUTPUT;

HcclCommConfig MakeVersionedCommConfig(uint32_t version, size_t size)
{
    HcclCommConfig config{};
    HcclCommConfigInit(&config);
    CommConfigInfo info{};
    info.configSize = size;
    info.magicWord = HCCL_COMM_CONFIG_MAGIC_WORD;
    info.version = version;
    info.reserved[0] = 'R';
    EXPECT_EQ(memcpy_s(config.reserved, sizeof(config.reserved), &info, sizeof(info)), EOK);
    return config;
}

std::vector<unsigned char> CopyCommConfigPrefix(const HcclCommConfig& config, size_t size)
{
    // 精确分配旧配置长度，ASan可检测新字段越界读取。
    std::vector<unsigned char> bytes(size);
    EXPECT_EQ(memcpy_s(bytes.data(), bytes.size(), &config, size), EOK);
    return bytes;
}

bool SetConsumedCommConfigFields(HcclCommConfig& config)
{
    config.hcclBufferSize = 256U;
    config.hcclDeterministic = 1U;
    config.hcclOpExpansionMode = 3U;
    config.hcclRdmaTrafficClass = 4U;
    config.hcclRdmaServiceLevel = 7U;
    config.hcclQos = 3U;
    config.hcclChannelSqDepth = 8193U;
    config.hcclChannelScqDepth = 256U;
    return strcpy_s(config.hcclCommName, sizeof(config.hcclCommName), "input_comm") == EOK
           && strcpy_s(config.hcclUdi, sizeof(config.hcclUdi), "input_udi") == EOK
           && strcpy_s(config.hcclAlgo, sizeof(config.hcclAlgo), "input_algo") == EOK
           && strcpy_s(config.hcclUseNetLayer, sizeof(config.hcclUseNetLayer), "not(0)") == EOK;
}

void ExpectConfigReadFailurePreserves(const HcclCommConfig* input, HcclResult expected = HCCL_E_PARA)
{
    std::array<unsigned char, sizeof(HcclCommConfig)> original;
    original.fill(0x5AU);
    HcclCommConfig output{};
    ASSERT_EQ(memcpy_s(&output, sizeof(output), original.data(), original.size()), EOK);
    EXPECT_EQ(ReadHcclCommConfig(input, output), expected);
    EXPECT_EQ(std::memcmp(&output, original.data(), original.size()), 0);
}

TEST(HcclCommConfigReadTest, Ut_InitDefault_When_SelectionContainsData_Expect_Cleared)
{
    HcclCommConfig config{};
    std::fill(std::begin(config.hcclUseNetLayer), std::end(config.hcclUseNetLayer), '7');
    HcclCommConfigInit(&config);
    for (char value : config.hcclUseNetLayer) {
        EXPECT_EQ('\0', value);
    }
}

TEST(HcclCommConfigReadTest, Ut_Read_When_BoundedInput_Expect_PreserveVersionAndSuppliedFields)
{
    const struct {
        uint32_t version;
        size_t size;
    } cases[] = {
        {0U, sizeof(CommConfigInfo)},
        {1U, offsetof(HcclCommConfig, hcclDeterministic) + sizeof(uint32_t)},
        {10U, offsetof(HcclCommConfig, hcclQos) + sizeof(uint32_t)},
        {12U, offsetof(HcclCommConfig, hcclUseNetLayer)},
        {HCCL_COMM_CONFIG_VERSION, sizeof(HcclCommConfig)},
    };
    for (const auto& item : cases) {
        SCOPED_TRACE(item.version);
        auto config = MakeVersionedCommConfig(item.version, item.size);
        ASSERT_TRUE(SetConsumedCommConfigFields(config));
        const auto input = CopyCommConfigPrefix(config, item.size);
        const auto original = input;
        HcclCommConfig output{};
        ASSERT_EQ(ReadHcclCommConfig(reinterpret_cast<const HcclCommConfig*>(input.data()), output), HCCL_SUCCESS);
        CommConfigInfo info{};
        ASSERT_EQ(memcpy_s(&info, sizeof(info), output.reserved, sizeof(info)), EOK);
        EXPECT_EQ(info.configSize, sizeof(HcclCommConfig));
        EXPECT_EQ(info.version, item.version);
        EXPECT_EQ(info.reserved[0], 'R');
        const size_t copiedEnd = std::min(item.size, offsetof(HcclCommConfig, hcclUseNetLayer));
        EXPECT_EQ(
            std::memcmp(
                reinterpret_cast<const unsigned char*>(&output) + sizeof(size_t), input.data() + sizeof(size_t),
                copiedEnd - sizeof(size_t)),
            0);
        EXPECT_STREQ(output.hcclUseNetLayer, item.version >= HCCL_COMM_CONFIG_NET_LAYER_VERSION ? "not(0)" : "");
        if (item.size == sizeof(CommConfigInfo)) {
            EXPECT_EQ(output.hcclBufferSize, HCCL_COMM_BUFFSIZE_CONFIG_NOT_SET);
        }
        EXPECT_EQ(input, original);
    }
}

TEST(HcclCommConfigReadTest, Ut_Read_When_NullOrSizeOnlyBuffer_Expect_ErrorWithoutHeaderOverread)
{
    ExpectConfigReadFailurePreserves(nullptr, HCCL_E_PTR);
    for (size_t declaredSize : {size_t{0}, sizeof(size_t), sizeof(CommConfigInfo) - 1U}) {
        SCOPED_TRACE(declaredSize);
        ExpectConfigReadFailurePreserves(reinterpret_cast<const HcclCommConfig*>(&declaredSize));
    }
}

TEST(HcclCommConfigReadTest, Ut_Read_When_InvalidHeaderOrSelectionField_Expect_UnchangedOutput)
{
    auto input = MakeVersionedCommConfig(HCCL_COMM_CONFIG_VERSION, sizeof(HcclCommConfig));
    CommConfigInfo info{};
    ASSERT_EQ(memcpy_s(&info, sizeof(info), input.reserved, sizeof(info)), EOK);
    info.magicWord = 0;
    ASSERT_EQ(memcpy_s(input.reserved, sizeof(input.reserved), &info, sizeof(info)), EOK);
    ExpectConfigReadFailurePreserves(&input);
    for (uint32_t version : {HCCL_COMM_CONFIG_VERSION, HCCL_COMM_CONFIG_VERSION + 1U}) {
        SCOPED_TRACE(version);
        const size_t shortSize = offsetof(HcclCommConfig, hcclUseNetLayer) + HCCL_COMM_NET_LAYER_MAX_LENGTH - 1U;
        input = MakeVersionedCommConfig(version, shortSize);
        const auto bytes = CopyCommConfigPrefix(input, shortSize);
        ExpectConfigReadFailurePreserves(reinterpret_cast<const HcclCommConfig*>(bytes.data()));
        input = MakeVersionedCommConfig(version, sizeof(HcclCommConfig));
        std::fill(std::begin(input.hcclUseNetLayer), std::end(input.hcclUseNetLayer), '0');
        ExpectConfigReadFailurePreserves(&input);
    }
}

TEST(HcclCommConfigReadTest, Ut_Read_When_FutureVersion_Expect_BoundedReusableCopy)
{
    for (uint32_t version : {HCCL_COMM_CONFIG_VERSION + 1U, std::numeric_limits<uint32_t>::max()}) {
        SCOPED_TRACE(version);
        const size_t declaredSize = sizeof(HcclCommConfig) + 64U;
        auto config = MakeVersionedCommConfig(version, declaredSize);
        ASSERT_TRUE(SetConsumedCommConfigFields(config));
        std::vector<unsigned char> input(declaredSize, 0xEEU);
        ASSERT_EQ(memcpy_s(input.data(), input.size(), &config, sizeof(config)), EOK);
        HcclCommConfig output{};
        ASSERT_EQ(ReadHcclCommConfig(reinterpret_cast<const HcclCommConfig*>(input.data()), output), HCCL_SUCCESS);
        ASSERT_EQ(ReadHcclCommConfig(&output, output), HCCL_SUCCESS);
        CommConfigInfo info{};
        ASSERT_EQ(memcpy_s(&info, sizeof(info), output.reserved, sizeof(info)), EOK);
        EXPECT_EQ(info.version, version);
        EXPECT_EQ(info.configSize, sizeof(HcclCommConfig));
        EXPECT_EQ(output.hcclChannelScqDepth, 256U);
        EXPECT_STREQ(output.hcclUseNetLayer, "not(0)");
    }
}

TEST(HcclCommConfigReadTest, Ut_Read_When_OldVersionAliasesOutput_Expect_OnlyNewFieldCleared)
{
    auto config = MakeVersionedCommConfig(1U, sizeof(HcclCommConfig));
    ASSERT_TRUE(SetConsumedCommConfigFields(config));
    std::fill(std::begin(config.hcclUseNetLayer), std::end(config.hcclUseNetLayer), '9');
    ASSERT_EQ(ReadHcclCommConfig(&config, config), HCCL_SUCCESS);
    // 业务字段的版本语义交给原解析流程，不在安全复制时重写。
    EXPECT_EQ(config.hcclDeterministic, 1U);
    EXPECT_EQ(config.hcclOpExpansionMode, 3U);
    EXPECT_EQ(config.hcclQos, 3U);
    EXPECT_STREQ(config.hcclUseNetLayer, "");
}

class HcclCommConfigApplyTest : public testing::Test {
protected:
    HcclCommConfig config_ = MakeVersionedCommConfig(11U, sizeof(HcclCommConfig));
    CommConfig target_{"apply_config"};
    uint32_t opMode_ = UNCHANGED_OUTPUT;

    void SetUp() override
    {
        ASSERT_EQ(target_.SetConfigUdi("previous_udi"), HCCL_SUCCESS);
        ASSERT_EQ(target_.SetConfigHcclAlgoStr("previous_algo"), HCCL_SUCCESS);
    }
};

TEST_F(HcclCommConfigApplyTest, Ut_Apply_When_NullConfig_Expect_OnlyProcessDeterministicAndDefaultOpMode)
{
    ASSERT_EQ(target_.SetConfigTrafficClass(4U), HCCL_SUCCESS);
    ASSERT_EQ(target_.SetConfigServiceLevel(7U), HCCL_SUCCESS);
    ASSERT_EQ(target_.SetConfigHcclQos(3U), HCCL_SUCCESS);
    ASSERT_EQ(target_.SetConfigSqDepth(15U), HCCL_SUCCESS);
    ASSERT_EQ(ApplyHcclCommConfig(nullptr, target_, opMode_, 2), HCCL_SUCCESS);
    EXPECT_EQ(opMode_, 0U);
    EXPECT_EQ(target_.GetConfigDeterministic(), 2U);
    EXPECT_EQ(target_.GetConfigTrafficClass(), 4U);
    EXPECT_EQ(target_.GetConfigServiceLevel(), 7U);
    EXPECT_EQ(target_.GetConfigHcclQos(), 3U);
    EXPECT_EQ(target_.GetConfigSqDepth(), 15U);
    EXPECT_EQ(target_.GetConfigUdi(), "previous_udi");
    EXPECT_EQ(target_.GetConfigHcclAlgoStr(), "previous_algo");
}

TEST_F(HcclCommConfigApplyTest, Ut_Apply_When_DeterministicInvalid_Expect_ExistingParameterErrors)
{
    for (int32_t processValue : {-1, 3, std::numeric_limits<int32_t>::max()}) {
        SCOPED_TRACE(processValue);
        EXPECT_EQ(ApplyHcclCommConfig(&config_, target_, opMode_, processValue), HCCL_E_PARA);
        EXPECT_EQ(ApplyHcclCommConfig(nullptr, target_, opMode_, processValue), HCCL_E_PARA);
    }
    for (uint32_t configValue : {3U, 255U}) {
        SCOPED_TRACE(configValue);
        config_.hcclDeterministic = configValue;
        EXPECT_EQ(ApplyHcclCommConfig(&config_, target_, opMode_, 0), HCCL_E_PARA);
    }
}

TEST_F(HcclCommConfigApplyTest, Ut_Apply_When_AlgoAndUdiFillEntireArrays_Expect_BoundedStrings)
{
    std::fill(std::begin(config_.hcclUdi), std::end(config_.hcclUdi), 'U');
    std::fill(std::begin(config_.hcclAlgo), std::end(config_.hcclAlgo), 'A');
    ASSERT_EQ(ApplyHcclCommConfig(&config_, target_, opMode_, 0), HCCL_SUCCESS);
    EXPECT_EQ(target_.GetConfigUdi(), std::string(UDI_MAX_LENGTH, 'U'));
    EXPECT_EQ(target_.GetConfigHcclAlgoStr(), std::string(HCCL_COMM_ALGO_MAX_LENGTH, 'A'));
}

TEST_F(NetLayerSelectionLoadTest, Ut_Prepare_When_OldVersions_Expect_PreserveVersionAndFreezeSelection)
{
    ASSERT_EQ(setenv(NET_LAYER_ENV_NAME, "not(0,0)", 1), 0);
    for (uint32_t version : {0U, 1U, 2U, 3U, 4U, 11U, 12U}) {
        SCOPED_TRACE(version);
        auto input = MakeVersionedCommConfig(version, sizeof(HcclCommConfig));
        ASSERT_TRUE(SetConsumedCommConfigFields(input));
        HcclCommConfig snapshot{};
        ASSERT_EQ(PrepareIndependentCommConfig(&input, snapshot), HCCL_SUCCESS);
        CommConfigInfo originalInfo{};
        ASSERT_EQ(memcpy_s(&originalInfo, sizeof(originalInfo), snapshot.reserved, sizeof(originalInfo)), EOK);
        EXPECT_EQ(originalInfo.version, version);
        EXPECT_EQ(originalInfo.configSize, sizeof(HcclCommConfig));
        EXPECT_EQ(originalInfo.reserved[0], 'R');
        EXPECT_EQ(std::memcmp(&input, &snapshot, offsetof(HcclCommConfig, hcclUseNetLayer)), 0);
        EXPECT_EQ(snapshot.hcclQos, 3U);
        EXPECT_EQ(snapshot.hcclChannelScqDepth, 256U);
        EXPECT_STREQ(snapshot.hcclUseNetLayer, "not(0)");
        if (version == 11U) {
            CommConfig target("old_version_snapshot");
            uint32_t opMode = 0;
            ASSERT_EQ(ApplyHcclCommConfig(&snapshot, target, opMode, 0), HCCL_SUCCESS);
            EXPECT_EQ(target.GetConfigScqDepth(), HCCL_COMM_SCQ_DEPTH_CONFIG_NOT_SET);
        }
        NetLayerSelection selection;
        ASSERT_EQ(ReadNetLayerSelection(snapshot.hcclUseNetLayer, selection), HCCL_SUCCESS);
        EXPECT_EQ(selection.GetRequestedLayers(), 1U);
    }
}

TEST_F(NetLayerSelectionLoadTest, Ut_Prepare_When_ParameterSet_Expect_PriorityAndOwnedSnapshot)
{
    ASSERT_EQ(setenv(NET_LAYER_ENV_NAME, "invalid", 1), 0);
    HcclCommConfig input{};
    HcclCommConfigInit(&input);
    ASSERT_EQ(strcpy_s(input.hcclUseNetLayer, sizeof(input.hcclUseNetLayer), "2,0,2"), EOK);
    HcclCommConfig snapshot{};
    ASSERT_EQ(PrepareIndependentCommConfig(&input, snapshot), HCCL_SUCCESS);
    input.hcclUseNetLayer[0] = '7';
    ASSERT_EQ(setenv(NET_LAYER_ENV_NAME, "7", 1), 0);
    EXPECT_STREQ(snapshot.hcclUseNetLayer, "0,2");
    NetLayerSelection selection;
    ASSERT_EQ(ReadNetLayerSelection(snapshot.hcclUseNetLayer, selection), HCCL_SUCCESS);
    EXPECT_EQ(selection.GetNormalizedExpression(), "0,2");
    EXPECT_EQ(selection.GetRequestedLayers(), 5U);
}

TEST_F(NetLayerSelectionLoadTest, Ut_Prepare_When_DefaultOrEnvironment_Expect_NoReadAfterCapture)
{
    HcclCommConfig input{};
    HcclCommConfigInit(&input);
    for (const auto* expression : {"", "2"}) {
        SCOPED_TRACE(expression);
        hccl::EnvNetLayerConfig::GetInstance().ResetParsed();
        ASSERT_EQ(setenv(NET_LAYER_ENV_NAME, expression, 1), 0);
        HcclCommConfig snapshot{};
        ASSERT_EQ(PrepareIndependentCommConfig(&input, snapshot), HCCL_SUCCESS);
        ASSERT_EQ(setenv(NET_LAYER_ENV_NAME, "invalid", 1), 0);
        NetLayerSelection selection;
        ASSERT_EQ(ReadNetLayerSelection(snapshot.hcclUseNetLayer, selection), HCCL_SUCCESS);
        EXPECT_EQ(selection.GetMode(), expression[0] == '\0' ? Mode::DEFAULT : Mode::INCLUDE);
        EXPECT_EQ(selection.GetRequestedLayers(), expression[0] == '\0' ? 0U : 4U);
    }
}

TEST_F(NetLayerSelectionLoadTest, Ut_Prepare_When_Invalid_Expect_AtomicFailure)
{
    ASSERT_EQ(setenv(NET_LAYER_ENV_NAME, "2", 1), 0);
    HcclCommConfig input{};
    HcclCommConfigInit(&input);
    for (const auto& expression : {"8", " ", "0,", "00000000000000000000000000000000"}) {
        ASSERT_EQ(
            memset_s(input.hcclUseNetLayer, sizeof(input.hcclUseNetLayer), 0, sizeof(input.hcclUseNetLayer)), EOK);
        const size_t length = std::strlen(expression);
        ASSERT_EQ(memcpy_s(input.hcclUseNetLayer, sizeof(input.hcclUseNetLayer), expression, length), EOK);
        const auto before = CopyCommConfigPrefix(input, sizeof(input));
        EXPECT_EQ(PrepareIndependentCommConfig(&input, input), HCCL_E_PARA);
        EXPECT_EQ(CopyCommConfigPrefix(input, sizeof(input)), before);
    }
}

TEST_F(NetLayerSelectionLoadTest, Ut_Prepare_When_OldShortConfig_Expect_NoTailReadAndFrozenEnvironment)
{
    ASSERT_EQ(setenv(NET_LAYER_ENV_NAME, "2", 1), 0);
    for (size_t size : {sizeof(CommConfigInfo), offsetof(HcclCommConfig, hcclUseNetLayer), sizeof(HcclCommConfig)}) {
        SCOPED_TRACE(size);
        auto input = MakeVersionedCommConfig(12U, size);
        std::fill(std::begin(input.hcclUseNetLayer), std::end(input.hcclUseNetLayer), '9');
        const auto bytes = CopyCommConfigPrefix(input, size);
        HcclCommConfig snapshot{};
        ASSERT_EQ(
            PrepareIndependentCommConfig(reinterpret_cast<const HcclCommConfig*>(bytes.data()), snapshot),
            HCCL_SUCCESS);
        NetLayerSelection selection;
        ASSERT_EQ(ReadNetLayerSelection(snapshot.hcclUseNetLayer, selection), HCCL_SUCCESS);
        EXPECT_EQ(selection.GetRequestedLayers(), 4U);
        EXPECT_STREQ(snapshot.hcclUseNetLayer, "2");
    }
}

} // namespace
