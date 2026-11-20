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

#include <cstdlib>
#include <cstdint>
#include <cstddef>
#include <string>
#include <thread>
#include <vector>

#include "net_layer_config.h"
#include "hccl/hccl_comm.h"
#include "net_layer_test_common.h"
#include "securec.h"

namespace {

using hccl::NetLayerSelection;
using net_layer_ut::ALL_LAYERS;
using net_layer_ut::Mode;
using net_layer_ut::NET_LAYER_ENV_NAME;
using net_layer_ut::NetLayerSelectionLoadTest;
using net_layer_ut::UNCHANGED_OUTPUT;

void ExpectSelection(
    const NetLayerSelection& selection, Mode mode, uint32_t requestedLayers, const std::string& expression)
{
    SCOPED_TRACE(expression);
    EXPECT_EQ(selection.GetMode(), mode);
    EXPECT_EQ(selection.GetRequestedLayers(), requestedLayers);
    EXPECT_EQ(selection.GetNormalizedExpression(), expression);
}

void ExpectParseFailurePreserves(const std::string& expression)
{
    SCOPED_TRACE(expression);
    NetLayerSelection selection;
    ASSERT_EQ(NetLayerSelection::Parse("not(7)", selection), HCCL_SUCCESS);
    EXPECT_EQ(NetLayerSelection::Parse(expression, selection), HCCL_E_PARA);
    ExpectSelection(selection, Mode::EXCLUDE, 0x80U, "not(7)");
}

TEST(NetLayerSelectionTest, Ut_Construct_When_Default_Expect_Unconfigured)
{
    NetLayerSelection selection;
    ExpectSelection(selection, Mode::DEFAULT, 0U, "");
}

TEST(NetLayerSelectionTest, Ut_FormatLayers_When_SparseOrInvalidBits_Expect_ReadableList)
{
    EXPECT_EQ(NetLayerSelection::FormatLayers(0U), "none");
    EXPECT_EQ(NetLayerSelection::FormatLayers(1U), "0");
    EXPECT_EQ(NetLayerSelection::FormatLayers(0x09U), "0,3");
    EXPECT_EQ(NetLayerSelection::FormatLayers(ALL_LAYERS), "0,1,2,3,4,5,6,7");
    EXPECT_EQ(NetLayerSelection::FormatLayers(0x80000101U), "0,8,31");
}

TEST(NetLayerSelectionTest, Ut_Parse_When_ValidExpression_Expect_NormalizedPolicy)
{
    const struct {
        std::string expression;
        Mode mode;
        uint32_t mask;
        const char* normalized;
    } cases[]
        = {{"", Mode::DEFAULT, 0U, ""},
           {"2", Mode::INCLUDE, 0x04U, "2"},
           {"0,7", Mode::INCLUDE, 0x81U, "0,7"},
           {"7,0,2,7,0", Mode::INCLUDE, 0x85U, "0,2,7"},
           {"not(7,0,7,2)", Mode::EXCLUDE, 0x85U, "not(0,2,7)"},
           {"7,6,5,4,3,2,1,0", Mode::INCLUDE, ALL_LAYERS, "0,1,2,3,4,5,6,7"},
           // 包含正选、反选表达式恰好占满31字节的边界。
           {std::string(30U, ' ') + "7", Mode::INCLUDE, 0x80U, "7"},
           {std::string(25U, ' ') + "not(0)", Mode::EXCLUDE, 0x01U, "not(0)"}};
    for (const auto& item : cases) {
        SCOPED_TRACE(item.expression);
        NetLayerSelection selection;
        ASSERT_EQ(NetLayerSelection::Parse(item.expression, selection), HCCL_SUCCESS);
        ExpectSelection(selection, item.mode, item.mask, item.normalized);
    }
}

TEST(NetLayerSelectionTest, Ut_Parse_When_AsciiWhitespaceAroundTokens_Expect_NormalizedExpression)
{
    for (char whitespace : std::string(" \t\n\r\f\v")) {
        SCOPED_TRACE(::testing::Message() << "ASCII whitespace=" << static_cast<int>(whitespace));
        const std::string space(1U, whitespace);
        NetLayerSelection selection;
        ASSERT_EQ(NetLayerSelection::Parse(space + "7" + space + "," + space + "0" + space, selection), HCCL_SUCCESS);
        ExpectSelection(selection, Mode::INCLUDE, 0x81U, "0,7");
        const std::string expression = space + "not" + space + "(" + space + "0" + space + ")" + space;
        ASSERT_EQ(NetLayerSelection::Parse(expression, selection), HCCL_SUCCESS);
        ExpectSelection(selection, Mode::EXCLUDE, 0x01U, "not(0)");
    }
}

TEST(NetLayerSelectionTest, Ut_Parse_When_InvalidLayerId_Expect_ErrorAndUnchangedSelection)
{
    const std::vector<std::string> expressions = {"8", "-1", "+1", "01", "00", "10", "1.0", "0x1", "a", "1 2"};
    for (const auto& expression : expressions) {
        SCOPED_TRACE(expression);
        ExpectParseFailurePreserves(expression);
        ExpectParseFailurePreserves("not(" + expression + ")");
    }
}

TEST(NetLayerSelectionTest, Ut_Parse_When_SyntaxEncodingOrLengthInvalid_Expect_ErrorAndUnchangedSelection)
{
    // 来源优先级由 Load 用例覆盖；这里只验证语法、编码和长度。
    const std::vector<std::string> expressions
        = {",0",
           "0,",
           "0,,2",
           "0, ,2",
           "0;2",
           "0:2",
           "0 2",
           "(0)",
           "[0]",
           "all",
           "auto",
           "0-2",
           "0,not(2)",
           "0\t2",
           "not",
           "not()",
           "not( )",
           "not(0",
           "not0)",
           "not((0))",
           "NOT(0)",
           "Not(0)",
           "n ot(0)",
           "not(0),1",
           "not(0)tail",
           "not(0,)",
           "not(,0)",
           "not(0,,1)",
           "not(not(0))",
           "not(0))",
           "(not(0))",
           "not(0 2)",
           "no\tt(0)",
           "not(all)",
           "not(auto)",
           "not(0-2)",
           " \t\n\r\f\v",
           std::string("0\0,7", 4U),
           std::string("not(0)\0", 7U),
           std::string("\0", 1U),
           std::string("\xC2\xA0") + "0",
           std::string(31U, ' ') + "7",
           std::string(26U, ' ') + "not(0)",
           std::string(100U, '0')};
    for (const auto& expression : expressions) {
        SCOPED_TRACE(::testing::Message() << "expression=" << expression << ", length=" << expression.size());
        ExpectParseFailurePreserves(expression);
    }
}

TEST(NetLayerSelectionTest, Ut_Resolve_When_Default_Expect_UnmodifiedCandidateSetIncludingEmpty)
{
    const NetLayerSelection selection;
    for (uint32_t availableLayers : {0U, 0x01U, 0x80U, 0x85U, ALL_LAYERS}) {
        SCOPED_TRACE(::testing::Message() << "availableLayers=" << availableLayers);
        uint32_t effectiveLayers = UNCHANGED_OUTPUT;
        EXPECT_EQ(selection.Resolve(availableLayers, effectiveLayers), HCCL_SUCCESS);
        EXPECT_EQ(effectiveLayers, availableLayers);
    }
}

TEST(NetLayerSelectionTest, Ut_Resolve_When_ExplicitSelection_Expect_StrictSubsetOrUnchangedOutput)
{
    const struct {
        const char* expression;
        uint32_t available;
        HcclResult result;
        uint32_t expected;
    } cases[]
        = {{"0,7", 0x85U, HCCL_SUCCESS, 0x81U},
           {"not(0,7)", 0x85U, HCCL_SUCCESS, 0x04U},
           {"0,7", 0x01U, HCCL_E_PARA, UNCHANGED_OUTPUT},
           {"0,7", 0x80U, HCCL_E_PARA, UNCHANGED_OUTPUT},
           {"0,7", 0x04U, HCCL_E_PARA, UNCHANGED_OUTPUT},
           {"not(7)", 0x05U, HCCL_E_PARA, UNCHANGED_OUTPUT},
           {"not(0)", 0x01U, HCCL_E_PARA, UNCHANGED_OUTPUT},
           {"not(0,1,2,3,4,5,6,7)", ALL_LAYERS, HCCL_E_PARA, UNCHANGED_OUTPUT},
           {"0", 0U, HCCL_E_PARA, UNCHANGED_OUTPUT},
           {"not(0)", 0U, HCCL_E_PARA, UNCHANGED_OUTPUT}};
    for (const auto& item : cases) {
        SCOPED_TRACE(::testing::Message() << item.expression << ", available=" << item.available);
        NetLayerSelection selection;
        ASSERT_EQ(NetLayerSelection::Parse(item.expression, selection), HCCL_SUCCESS);
        uint32_t effectiveLayers = UNCHANGED_OUTPUT;
        EXPECT_EQ(selection.Resolve(item.available, effectiveLayers), item.result);
        EXPECT_EQ(effectiveLayers, item.expected);
    }
}

TEST(NetLayerSelectionTest, Ut_Resolve_When_CandidateMaskOutsideEightBits_Expect_ErrorForEveryMode)
{
    for (const auto& expression : {"", "0", "not(0)"}) {
        SCOPED_TRACE(expression);
        NetLayerSelection selection;
        ASSERT_EQ(NetLayerSelection::Parse(expression, selection), HCCL_SUCCESS);
        for (uint32_t availableLayers : {0x100U, 0x1FFU, 0x800000FFU, 0xFFFFFFFFU}) {
            SCOPED_TRACE(::testing::Message() << "availableLayers=" << availableLayers);
            uint32_t effectiveLayers = UNCHANGED_OUTPUT;
            EXPECT_EQ(selection.Resolve(availableLayers, effectiveLayers), HCCL_E_PARA);
            EXPECT_EQ(effectiveLayers, UNCHANGED_OUTPUT);
        }
    }
}

HcclCommConfig MakeSelectionConfig(const std::string& expression)
{
    HcclCommConfig config{};
    HcclCommConfigInit(&config);
    // 复制完整输入；32字节用例刻意不补终止符，不能截断成合法参数。
    EXPECT_EQ(
        EOK, memcpy_s(config.hcclUseNetLayer, sizeof(config.hcclUseNetLayer), expression.data(), expression.size()));
    return config;
}

void ExpectLoadFailurePreserves(const HcclCommConfig* commConfig)
{
    NetLayerSelection selection;
    ASSERT_EQ(NetLayerSelection::Parse("not(7)", selection), HCCL_SUCCESS);
    EXPECT_EQ(NetLayerSelection::LoadForIndependentComm(commConfig, selection), HCCL_E_PARA);
    ExpectSelection(selection, Mode::EXCLUDE, 0x80U, "not(7)");
}

void ExpectSubCommFailurePreserves(const char* expression, uint32_t parentLayers, uint32_t availableLayers)
{
    uint32_t effectiveLayers = UNCHANGED_OUTPUT;
    EXPECT_EQ(
        NetLayerSelection::ResolveForSubComm(expression, parentLayers, availableLayers, effectiveLayers), HCCL_E_PARA);
    EXPECT_EQ(effectiveLayers, UNCHANGED_OUTPUT);
}

TEST_F(NetLayerSelectionLoadTest, Ut_Load_When_ParameterAbsentOrEmpty_Expect_DefaultOrEnvironment)
{
    const auto config = MakeSelectionConfig("");
    for (const char* environment : {static_cast<const char*>(nullptr), "", "not(7,0,7)"}) {
        SCOPED_TRACE(environment ? environment : "<unset>");
        hccl::EnvNetLayerConfig::GetInstance().ResetParsed();
        ASSERT_EQ(environment ? setenv(NET_LAYER_ENV_NAME, environment, 1) : unsetenv(NET_LAYER_ENV_NAME), 0);
        for (const HcclCommConfig* parameter : {static_cast<const HcclCommConfig*>(nullptr), &config}) {
            NetLayerSelection selection;
            ASSERT_EQ(NetLayerSelection::LoadForIndependentComm(parameter, selection), HCCL_SUCCESS);
            const bool selected = environment && environment[0] != '\0';
            ExpectSelection(
                selection, selected ? Mode::EXCLUDE : Mode::DEFAULT, selected ? 0x81U : 0U, selected ? "not(0,7)" : "");
        }
    }
}

TEST_F(NetLayerSelectionLoadTest, Ut_Load_When_ConcurrentAndEnvironmentChanges_Expect_ProcessCacheAndParameterPriority)
{
    for (const auto* expression : {"", "2,0,2", "not(0)"}) {
        SCOPED_TRACE(expression);
        hccl::EnvNetLayerConfig::GetInstance().ResetParsed();
        ASSERT_EQ(setenv(NET_LAYER_ENV_NAME, expression, 1), 0);
        NetLayerSelection expected;
        ASSERT_EQ(NetLayerSelection::Parse(expression, expected), HCCL_SUCCESS);
        std::vector<NetLayerSelection> selections(4);
        std::vector<HcclResult> results(4, HCCL_E_INTERNAL);
        std::vector<std::thread> threads;
        for (size_t index = 0; index < selections.size(); ++index) {
            threads.emplace_back([&, index] {
                results[index] = NetLayerSelection::LoadForIndependentComm(nullptr, selections[index]);
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }
        for (size_t index = 0; index < selections.size(); ++index) {
            ASSERT_EQ(results[index], HCCL_SUCCESS);
            ExpectSelection(
                selections[index], expected.GetMode(), expected.GetRequestedLayers(),
                expected.GetNormalizedExpression());
        }
        ASSERT_EQ(setenv(NET_LAYER_ENV_NAME, "invalid", 1), 0);
        NetLayerSelection selection;
        ASSERT_EQ(NetLayerSelection::LoadForIndependentComm(nullptr, selection), HCCL_SUCCESS);
        ExpectSelection(
            selection, expected.GetMode(), expected.GetRequestedLayers(), expected.GetNormalizedExpression());
        const auto config = MakeSelectionConfig("7");
        ASSERT_EQ(NetLayerSelection::LoadForIndependentComm(&config, selection), HCCL_SUCCESS);
        ExpectSelection(selection, Mode::INCLUDE, 0x80U, "7");
    }
}

TEST_F(NetLayerSelectionLoadTest, Ut_Load_When_ParameterValid_Expect_IgnoreLowerPriorityEnvironment)
{
    const auto config = MakeSelectionConfig("0");
    const std::vector<std::string> environments = {"2", "invalid", std::string(4096U, '9')};
    for (const auto& environment : environments) {
        SCOPED_TRACE(::testing::Message() << "environment length=" << environment.size());
        ASSERT_EQ(setenv(NET_LAYER_ENV_NAME, environment.c_str(), 1), 0);
        NetLayerSelection selection;
        ASSERT_EQ(NetLayerSelection::LoadForIndependentComm(&config, selection), HCCL_SUCCESS);
        ExpectSelection(selection, Mode::INCLUDE, 0x01U, "0");
    }
}

TEST_F(NetLayerSelectionLoadTest, Ut_Load_When_ParameterInvalid_Expect_ErrorWithoutEnvironmentFallback)
{
    ASSERT_EQ(setenv(NET_LAYER_ENV_NAME, "2", 1), 0);
    const std::vector<std::string> expressions = {"8", " ", "0,", std::string(31U, ' ') + "0"};
    for (const auto& expression : expressions) {
        SCOPED_TRACE(expression);
        const auto config = MakeSelectionConfig(expression);
        ExpectLoadFailurePreserves(&config);
    }
}

TEST_F(NetLayerSelectionLoadTest, Ut_LoadAndSubComm_When_LengthBoundary_Expect_NoTrimmingOrTruncation)
{
    for (bool fromEnvironment : {false, true}) {
        for (size_t length : {31U, 32U, 4096U}) {
            if (!fromEnvironment && length > HCCL_COMM_NET_LAYER_MAX_LENGTH) {
                continue;
            }
            SCOPED_TRACE(::testing::Message() << "environment=" << fromEnvironment << ", length=" << length);
            hccl::EnvNetLayerConfig::GetInstance().ResetParsed();
            const std::string expression = std::string(length - 1U, ' ') + "7";
            ASSERT_EQ(setenv(NET_LAYER_ENV_NAME, fromEnvironment ? expression.c_str() : "", 1), 0);
            const auto config = MakeSelectionConfig(fromEnvironment ? "" : expression);
            const HcclCommConfig* parameter = fromEnvironment ? nullptr : &config;
            if (!fromEnvironment) {
                uint32_t effective = UNCHANGED_OUTPUT;
                ASSERT_EQ(
                    NetLayerSelection::ResolveForSubComm(parameter->hcclUseNetLayer, ALL_LAYERS, ALL_LAYERS, effective),
                    length == 31U ? HCCL_SUCCESS : HCCL_E_PARA);
                EXPECT_EQ(effective, length == 31U ? 0x80U : UNCHANGED_OUTPUT);
            }
            if (length != 31U) {
                ExpectLoadFailurePreserves(parameter);
                continue;
            }
            NetLayerSelection selection;
            ASSERT_EQ(NetLayerSelection::LoadForIndependentComm(parameter, selection), HCCL_SUCCESS);
            ExpectSelection(selection, Mode::INCLUDE, 0x80U, "7");
        }
    }
}

TEST_F(NetLayerSelectionLoadTest, Ut_LoadAndSubComm_When_GarbageAfterFirstTerminator_Expect_IgnoreTail)
{
    for (char layer : {'0', '2'}) {
        SCOPED_TRACE(layer);
        const auto config = MakeSelectionConfig(std::string{layer, '\0', '8', ',', '9'});
        const uint32_t expected = 1U << (layer - '0');
        NetLayerSelection selection;
        ASSERT_EQ(NetLayerSelection::LoadForIndependentComm(&config, selection), HCCL_SUCCESS);
        ExpectSelection(selection, Mode::INCLUDE, expected, std::string(1U, layer));
        uint32_t effective = UNCHANGED_OUTPUT;
        ASSERT_EQ(
            NetLayerSelection::ResolveForSubComm(config.hcclUseNetLayer, 0x05U, ALL_LAYERS, effective), HCCL_SUCCESS);
        EXPECT_EQ(effective, expected);
    }
}

TEST_F(NetLayerSelectionLoadTest, Ut_Load_When_EnvironmentInvalid_Expect_ErrorAndUnchangedSelection)
{
    for (const auto& expression : {"invalid", " \t", "not()", "8"}) {
        SCOPED_TRACE(expression);
        ASSERT_EQ(setenv(NET_LAYER_ENV_NAME, expression, 1), 0);
        ExpectLoadFailurePreserves(nullptr);
    }
    ASSERT_EQ(setenv(NET_LAYER_ENV_NAME, "2", 1), 0);
    NetLayerSelection selection;
    ASSERT_EQ(NetLayerSelection::LoadForIndependentComm(nullptr, selection), HCCL_SUCCESS);
    ExpectSelection(selection, Mode::INCLUDE, 0x04U, "2");
}

TEST(NetLayerSelectionSubCommTest, Ut_Resolve_When_ParentAndChildVary_Expect_StrictSubsetOrAtomicFailure)
{
    const struct {
        const char* expression;
        uint32_t parent;
        uint32_t available;
        uint32_t expected;
    } cases[]
        = {{nullptr, 0x85U, ALL_LAYERS, 0x85U},
           {"", 0x85U, ALL_LAYERS, 0x85U},
           {nullptr, 0x85U, 0x07U, 0x05U},
           {"2", 0x85U, ALL_LAYERS, 0x04U},
           {"7,0,7", 0x85U, 0x81U, 0x81U},
           {"not(0)", 0x85U, ALL_LAYERS, 0x84U},
           {"not(7,0,7)", 0x85U, ALL_LAYERS, 0x04U},
           {"2", 0x01U, 0x05U, UNCHANGED_OUTPUT},
           {"0,2", 0x01U, 0x05U, UNCHANGED_OUTPUT},
           {"7", 0x85U, 0x05U, UNCHANGED_OUTPUT},
           {"0,7", 0x85U, 0x05U, UNCHANGED_OUTPUT},
           {"not(2)", 0x01U, 0x05U, UNCHANGED_OUTPUT},
           {"not(7)", 0x85U, 0x05U, UNCHANGED_OUTPUT},
           {"not(0,7)", 0x85U, 0x05U, UNCHANGED_OUTPUT},
           {nullptr, 0U, ALL_LAYERS, 0U},
           {nullptr, ALL_LAYERS, 0U, 0U},
           {nullptr, 0x01U, 0x04U, 0U},
           {"0", 0x01U, 0x04U, UNCHANGED_OUTPUT},
           {"not(0)", 0x01U, 0x04U, UNCHANGED_OUTPUT},
           {"not(0)", 0x01U, ALL_LAYERS, UNCHANGED_OUTPUT},
           {"0", 0U, 0U, UNCHANGED_OUTPUT}};
    for (const auto& item : cases) {
        SCOPED_TRACE(
            ::testing::Message() << "expression=" << (item.expression ? item.expression : "<null>")
                                 << ", parent=" << item.parent << ", available=" << item.available);
        const auto config = MakeSelectionConfig(item.expression ? item.expression : "");
        uint32_t effectiveLayers = UNCHANGED_OUTPUT;
        ASSERT_EQ(
            NetLayerSelection::ResolveForSubComm(
                item.expression ? config.hcclUseNetLayer : nullptr, item.parent, item.available, effectiveLayers),
            item.expected == UNCHANGED_OUTPUT ? HCCL_E_PARA : HCCL_SUCCESS);
        EXPECT_EQ(effectiveLayers, item.expected);
    }
}

TEST(NetLayerSelectionSubCommTest, Ut_Resolve_When_SyntaxInvalid_Expect_ErrorAndUnchangedOutput)
{
    const std::vector<std::string> expressions = {" ", "8", "0,", "not()", "0,not(2)", "not(not(0))"};
    for (const auto& expression : expressions) {
        SCOPED_TRACE(expression);
        const auto config = MakeSelectionConfig(expression);
        ExpectSubCommFailurePreserves(config.hcclUseNetLayer, ALL_LAYERS, ALL_LAYERS);
    }
}

TEST(NetLayerSelectionSubCommTest, Ut_Resolve_When_ParentOrChildMaskHasHighBits_Expect_ErrorBeforeIntersection)
{
    const auto config = MakeSelectionConfig("0");
    for (uint32_t invalid : {0x100U, 0x1FFU, 0x80000001U, 0xFFFFFFFFU}) {
        SCOPED_TRACE(::testing::Message() << "invalid mask=" << invalid);
        ExpectSubCommFailurePreserves(nullptr, invalid, 0U);
        ExpectSubCommFailurePreserves(config.hcclUseNetLayer, invalid, ALL_LAYERS);
        ExpectSubCommFailurePreserves(nullptr, 0U, invalid);
        ExpectSubCommFailurePreserves(config.hcclUseNetLayer, ALL_LAYERS, invalid);
    }
}

TEST_F(NetLayerSelectionLoadTest, Ut_SubComm_When_EnvironmentSet_Expect_IgnoreForInheritedAndExplicitSelection)
{
    const auto emptyConfig = MakeSelectionConfig("");
    const auto selectedConfig = MakeSelectionConfig("0");
    const auto invalidConfig = MakeSelectionConfig("8");
    const std::vector<std::string> environments = {"2", "invalid", std::string(4096U, '9')};
    for (const auto& environment : environments) {
        SCOPED_TRACE(::testing::Message() << "environment length=" << environment.size());
        ASSERT_EQ(setenv(NET_LAYER_ENV_NAME, environment.c_str(), 1), 0);
        uint32_t effectiveLayers = UNCHANGED_OUTPUT;
        ASSERT_EQ(NetLayerSelection::ResolveForSubComm(nullptr, 0x81U, ALL_LAYERS, effectiveLayers), HCCL_SUCCESS);
        EXPECT_EQ(effectiveLayers, 0x81U);
        ASSERT_EQ(
            NetLayerSelection::ResolveForSubComm(emptyConfig.hcclUseNetLayer, 0x81U, ALL_LAYERS, effectiveLayers),
            HCCL_SUCCESS);
        EXPECT_EQ(effectiveLayers, 0x81U);
        ASSERT_EQ(
            NetLayerSelection::ResolveForSubComm(selectedConfig.hcclUseNetLayer, 0x81U, ALL_LAYERS, effectiveLayers),
            HCCL_SUCCESS);
        EXPECT_EQ(effectiveLayers, 0x01U);
        ExpectSubCommFailurePreserves(invalidConfig.hcclUseNetLayer, 0x81U, ALL_LAYERS);
    }
}

} // namespace
