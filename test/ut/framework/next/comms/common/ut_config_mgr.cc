/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
#include "mockcpp/mockcpp.hpp"
#include "config/base_config_pub.h"
#include "coll_comm_config_mgr.h"
#include "env_logic_superpod_config.h"
#include "base_comm_node_accessors.h"
#include "env_ub_config.h"

using namespace hcomm;

namespace {

/// 保存/恢复单个环境变量，RAII 保证用例间隔离
class EnvGuard {
public:
    explicit EnvGuard(const char* name) : name_(name)
    {
        const char* val = std::getenv(name);
        if (val != nullptr) {
            savedValue_ = val;
            hadValue_ = true;
        }
    }
    ~EnvGuard()
    {
        if (hadValue_) {
            (void)setenv(name_, savedValue_.c_str(), 1);
        } else {
            (void)unsetenv(name_);
        }
    }
    void Set(const char* value) { (void)setenv(name_, value, 1); }
    void Unset() { (void)unsetenv(name_); }

private:
    const char* name_;
    std::string savedValue_;
    bool hadValue_{false};
};

} // namespace

// ==================== HCOMM_TA_CTP_UB_TIMEOUT ====================

TEST(ConfigMgrTest, Ut_TaCtpUbTimeOut_NotSet_Expect_Default8)
{
    EnvGuard guard("HCOMM_TA_CTP_UB_TIMEOUT");
    guard.Unset();
    GetConfigMgr().GetRdmaConfig().ResetParsed();
    uint32_t value = 0;
    EXPECT_EQ(GetConfigMgr().GetRdmaConfig().GetTaCtpUbTimeOut(value), HCCL_SUCCESS);
    EXPECT_EQ(value, 8U);
}

TEST(ConfigMgrTest, Ut_TaCtpUbTimeOut_ValidValue0_Expect_0)
{
    EnvGuard guard("HCOMM_TA_CTP_UB_TIMEOUT");
    guard.Set("0");
    GetConfigMgr().GetRdmaConfig().ResetParsed();
    uint32_t value = 0;
    EXPECT_EQ(GetConfigMgr().GetRdmaConfig().GetTaCtpUbTimeOut(value), HCCL_SUCCESS);
    EXPECT_EQ(value, 0U);
}

TEST(ConfigMgrTest, Ut_TaCtpUbTimeOut_ValidValue31_Expect_31)
{
    EnvGuard guard("HCOMM_TA_CTP_UB_TIMEOUT");
    guard.Set("31");
    GetConfigMgr().GetRdmaConfig().ResetParsed();
    uint32_t value = 0;
    EXPECT_EQ(GetConfigMgr().GetRdmaConfig().GetTaCtpUbTimeOut(value), HCCL_SUCCESS);
    EXPECT_EQ(value, 31U);
}

TEST(ConfigMgrTest, Ut_TaCtpUbTimeOut_InvalidValue32_Expect_Error)
{
    EnvGuard guard("HCOMM_TA_CTP_UB_TIMEOUT");
    guard.Set("32"); // 超过 UB_TIMEOUT_MAX=31，校验失败返回错误码
    GetConfigMgr().GetRdmaConfig().ResetParsed();
    uint32_t value = 0;
    EXPECT_NE(GetConfigMgr().GetRdmaConfig().GetTaCtpUbTimeOut(value), HCCL_SUCCESS);
}

TEST(ConfigMgrTest, Ut_TaCtpUbTimeOut_InvalidString_Expect_Error)
{
    EnvGuard guard("HCOMM_TA_CTP_UB_TIMEOUT");
    guard.Set("abc"); // 非数字，解析失败返回错误码
    GetConfigMgr().GetRdmaConfig().ResetParsed();
    uint32_t value = 0;
    EXPECT_NE(GetConfigMgr().GetRdmaConfig().GetTaCtpUbTimeOut(value), HCCL_SUCCESS);
}

TEST(ConfigMgrTest, Ut_TaCtpUbTimeOut_NegativeValue_Expect_Error)
{
    EnvGuard guard("HCOMM_TA_CTP_UB_TIMEOUT");
    guard.Set("-1"); // StrToNum 检查全数字，负号被拒绝，解析失败返回错误码
    GetConfigMgr().GetRdmaConfig().ResetParsed();
    uint32_t value = 0;
    EXPECT_NE(GetConfigMgr().GetRdmaConfig().GetTaCtpUbTimeOut(value), HCCL_SUCCESS);
}

TEST(ConfigMgrTest, Ut_TaCtpUbTimeOut_EmptyString_Expect_Default8)
{
    EnvGuard guard("HCOMM_TA_CTP_UB_TIMEOUT");
    guard.Set(""); // 空字符串视为未设置
    GetConfigMgr().GetRdmaConfig().ResetParsed();
    uint32_t value = 0;
    EXPECT_EQ(GetConfigMgr().GetRdmaConfig().GetTaCtpUbTimeOut(value), HCCL_SUCCESS);
    EXPECT_EQ(value, 8U);
}

// ==================== HCOMM_TA_RTP_UB_TIMEOUT ====================

TEST(ConfigMgrTest, Ut_TaRtpUbTimeOut_NotSet_Expect_Default16)
{
    EnvGuard guard("HCOMM_TA_RTP_UB_TIMEOUT");
    guard.Unset();
    GetConfigMgr().GetRdmaConfig().ResetParsed();
    uint32_t value = 0;
    EXPECT_EQ(GetConfigMgr().GetRdmaConfig().GetTaRtpUbTimeOut(value), HCCL_SUCCESS);
    EXPECT_EQ(value, 16U);
}

TEST(ConfigMgrTest, Ut_TaRtpUbTimeOut_ValidValue0_Expect_0)
{
    EnvGuard guard("HCOMM_TA_RTP_UB_TIMEOUT");
    guard.Set("0");
    GetConfigMgr().GetRdmaConfig().ResetParsed();
    uint32_t value = 0;
    EXPECT_EQ(GetConfigMgr().GetRdmaConfig().GetTaRtpUbTimeOut(value), HCCL_SUCCESS);
    EXPECT_EQ(value, 0U);
}

TEST(ConfigMgrTest, Ut_TaRtpUbTimeOut_ValidValue31_Expect_31)
{
    EnvGuard guard("HCOMM_TA_RTP_UB_TIMEOUT");
    guard.Set("31");
    GetConfigMgr().GetRdmaConfig().ResetParsed();
    uint32_t value = 0;
    EXPECT_EQ(GetConfigMgr().GetRdmaConfig().GetTaRtpUbTimeOut(value), HCCL_SUCCESS);
    EXPECT_EQ(value, 31U);
}

TEST(ConfigMgrTest, Ut_TaRtpUbTimeOut_InvalidValue32_Expect_Error)
{
    EnvGuard guard("HCOMM_TA_RTP_UB_TIMEOUT");
    guard.Set("32"); // 超过 UB_TIMEOUT_MAX=31，校验失败返回错误码
    GetConfigMgr().GetRdmaConfig().ResetParsed();
    uint32_t value = 0;
    EXPECT_NE(GetConfigMgr().GetRdmaConfig().GetTaRtpUbTimeOut(value), HCCL_SUCCESS);
}

TEST(ConfigMgrTest, Ut_TaRtpUbTimeOut_NegativeValue_Expect_Error)
{
    EnvGuard guard("HCOMM_TA_RTP_UB_TIMEOUT");
    guard.Set("-1"); // StrToNum 检查全数字，负号被拒绝，解析失败返回错误码
    GetConfigMgr().GetRdmaConfig().ResetParsed();
    uint32_t value = 0;
    EXPECT_NE(GetConfigMgr().GetRdmaConfig().GetTaRtpUbTimeOut(value), HCCL_SUCCESS);
}

// ==================== HCOMM_TA_RTP_UBOE_TIMEOUT ====================

TEST(ConfigMgrTest, Ut_TaRtpUboeTimeOut_NotSet_Expect_Default16)
{
    EnvGuard guard("HCOMM_TA_RTP_UBOE_TIMEOUT");
    guard.Unset();
    GetConfigMgr().GetRdmaConfig().ResetParsed();
    uint32_t value = 0;
    EXPECT_EQ(GetConfigMgr().GetRdmaConfig().GetTaRtpUboeTimeOut(value), HCCL_SUCCESS);
    EXPECT_EQ(value, 16U);
}

TEST(ConfigMgrTest, Ut_TaRtpUboeTimeOut_ValidValue0_Expect_0)
{
    EnvGuard guard("HCOMM_TA_RTP_UBOE_TIMEOUT");
    guard.Set("0");
    GetConfigMgr().GetRdmaConfig().ResetParsed();
    uint32_t value = 0;
    EXPECT_EQ(GetConfigMgr().GetRdmaConfig().GetTaRtpUboeTimeOut(value), HCCL_SUCCESS);
    EXPECT_EQ(value, 0U);
}

TEST(ConfigMgrTest, Ut_TaRtpUboeTimeOut_ValidValue31_Expect_31)
{
    EnvGuard guard("HCOMM_TA_RTP_UBOE_TIMEOUT");
    guard.Set("31");
    GetConfigMgr().GetRdmaConfig().ResetParsed();
    uint32_t value = 0;
    EXPECT_EQ(GetConfigMgr().GetRdmaConfig().GetTaRtpUboeTimeOut(value), HCCL_SUCCESS);
    EXPECT_EQ(value, 31U);
}

TEST(ConfigMgrTest, Ut_TaRtpUboeTimeOut_InvalidValue100_Expect_Error)
{
    EnvGuard guard("HCOMM_TA_RTP_UBOE_TIMEOUT");
    guard.Set("100"); // 超过 UB_TIMEOUT_MAX=31，校验失败返回错误码
    GetConfigMgr().GetRdmaConfig().ResetParsed();
    uint32_t value = 0;
    EXPECT_NE(GetConfigMgr().GetRdmaConfig().GetTaRtpUboeTimeOut(value), HCCL_SUCCESS);
}

TEST(ConfigMgrTest, Ut_TaRtpUboeTimeOut_InvalidString_Expect_Error)
{
    EnvGuard guard("HCOMM_TA_RTP_UBOE_TIMEOUT");
    guard.Set("xyz"); // 非数字，解析失败返回错误码
    GetConfigMgr().GetRdmaConfig().ResetParsed();
    uint32_t value = 0;
    EXPECT_NE(GetConfigMgr().GetRdmaConfig().GetTaRtpUboeTimeOut(value), HCCL_SUCCESS);
}

TEST(ConfigMgrTest, Ut_TaRtpUboeTimeOut_NegativeValue_Expect_Error)
{
    EnvGuard guard("HCOMM_TA_RTP_UBOE_TIMEOUT");
    guard.Set("-1"); // StrToNum 检查全数字，负号被拒绝，解析失败返回错误码
    GetConfigMgr().GetRdmaConfig().ResetParsed();
    uint32_t value = 0;
    EXPECT_NE(GetConfigMgr().GetRdmaConfig().GetTaRtpUboeTimeOut(value), HCCL_SUCCESS);
}

// ==================== HCCL_UB_MULTI_CHANNEL_NUM ====================

TEST(ConfigMgrTest, Ut_UbMultiChannelNum_NotSet_Expect_Default1)
{
    EnvGuard guard("HCCL_UB_MULTI_CHANNEL_NUM");
    guard.Unset();
    hccl::EnvUbConfig ubCfg;
    ubCfg.ResetParsed();
    EXPECT_EQ(ubCfg.Parse(), HCCL_SUCCESS);
    EXPECT_EQ(ubCfg.GetUbMultiChannelNum(), 1U);
}

TEST(ConfigMgrTest, Ut_UbMultiChannelNum_ValidValue16_Expect_16)
{
    EnvGuard guard("HCCL_UB_MULTI_CHANNEL_NUM");
    guard.Set("16");
    hccl::EnvUbConfig ubCfg;
    ubCfg.ResetParsed();
    EXPECT_EQ(ubCfg.Parse(), HCCL_SUCCESS);
    EXPECT_EQ(ubCfg.GetUbMultiChannelNum(), 16U);
}

TEST(ConfigMgrTest, Ut_UbMultiChannelNum_InvalidString_Expect_Error)
{
    EnvGuard guard("HCCL_UB_MULTI_CHANNEL_NUM");
    guard.Set("abc"); // 非数字，解析失败返回错误码
    hccl::EnvUbConfig ubCfg;
    ubCfg.ResetParsed();
    EXPECT_NE(ubCfg.Parse(), HCCL_SUCCESS);
}

TEST(ConfigMgrTest, Ut_UbMultiChannelNum_NegativeValue_Expect_Error)
{
    EnvGuard guard("HCCL_UB_MULTI_CHANNEL_NUM");
    guard.Set("-1"); // StrToNum 检查全数字，负号被拒绝，解析失败返回错误码
    hccl::EnvUbConfig ubCfg;
    ubCfg.ResetParsed();
    EXPECT_NE(ubCfg.Parse(), HCCL_SUCCESS);
}

TEST(ConfigMgrTest, Ut_UbMultiChannelNum_InvalidValue0_Expect_Error)
{
    EnvGuard guard("HCCL_UB_MULTI_CHANNEL_NUM");
    guard.Set("0"); // 低于最小值1，范围校验失败返回错误码
    hccl::EnvUbConfig ubCfg;
    ubCfg.ResetParsed();
    EXPECT_NE(ubCfg.Parse(), HCCL_SUCCESS);
}

TEST(ConfigMgrTest, Ut_UbMultiChannelNum_InvalidValue17_Expect_Error)
{
    EnvGuard guard("HCCL_UB_MULTI_CHANNEL_NUM");
    guard.Set("17"); // 超过最大值16，范围校验失败返回错误码
    hccl::EnvUbConfig ubCfg;
    ubCfg.ResetParsed();
    EXPECT_NE(ubCfg.Parse(), HCCL_SUCCESS);
}

TEST(ConfigMgrTest, Ut_UbMultiChannelNum_ParseOnceIdempotent_Expect_SameResult)
{
    EnvGuard guard("HCCL_UB_MULTI_CHANNEL_NUM");
    hccl::EnvUbConfig ubCfg;

    // 合法值：首次解析成功后缓存，重复 Parse() 结果一致
    guard.Set("16");
    ubCfg.ResetParsed();
    EXPECT_EQ(ubCfg.Parse(), HCCL_SUCCESS);
    EXPECT_EQ(ubCfg.Parse(), HCCL_SUCCESS);
    EXPECT_EQ(ubCfg.GetUbMultiChannelNum(), 16U);

    // 非法值：失败不缓存，重复 Parse() 均返回错误
    guard.Set("abc");
    ubCfg.ResetParsed();
    EXPECT_NE(ubCfg.Parse(), HCCL_SUCCESS);
    EXPECT_NE(ubCfg.Parse(), HCCL_SUCCESS);
}

// ==================== CollCommConfigMgr ====================

class CollCommConfigMgrTest : public testing::Test {
protected:
    void TearDown() override { mockcpp::GlobalMockObject::verify(); }
};

TEST_F(CollCommConfigMgrTest, Ut_Init_When_AllConfigsValid_Expect_ParseOnce)
{
    MOCKER_CPP(&hccl::EnvUbConfig::Parse).expects(mockcpp::once()).will(mockcpp::returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hccl::HostMultiQpConfig::Parse).expects(mockcpp::once()).will(mockcpp::returnValue(HCCL_SUCCESS));
    hccl::CollCommConfigMgr configMgr;

    EXPECT_EQ(configMgr.Init(), HCCL_SUCCESS);
    EXPECT_EQ(configMgr.Init(), HCCL_SUCCESS);
}

TEST_F(CollCommConfigMgrTest, Ut_Init_When_UbConfigInvalid_Expect_SkipHostMultiQpConfig)
{
    MOCKER_CPP(&hccl::EnvUbConfig::Parse).expects(mockcpp::once()).will(mockcpp::returnValue(HCCL_E_PARA));
    MOCKER_CPP(&hccl::HostMultiQpConfig::Parse).expects(mockcpp::never());
    hccl::CollCommConfigMgr configMgr;

    EXPECT_EQ(configMgr.Init(), HCCL_E_PARA);
}

// ==================== HCCL_LOGIC_SUPERPOD_ID ====================

TEST(ConfigMgrTest, Ut_LogicSuperPodId_When_ParseSucceeds_Expect_LogEffectiveValueOnce)
{
    EnvGuard guard("HCCL_LOGIC_SUPERPOD_ID");
    for (const char* value : {"group_A", static_cast<const char*>(nullptr), "", "EmptyString"}) {
        SCOPED_TRACE(value == nullptr ? "<unset>" : value);
        if (value == nullptr) {
            guard.Unset();
        } else {
            guard.Set(value);
        }
        hccl::EnvLogicSuperPodConfig config;
        extern s32 log_level_get_stub();
        extern void log_level_set_stub(s32 logLevel);
        const s32 originalLogLevel = log_level_get_stub();
        log_level_set_stub(DLOG_INFO);
        testing::internal::CaptureStdout();
        const HcclResult firstRet = config.Parse();
        guard.Set("group_B");
        const HcclResult cachedRet = config.Parse();
        const std::string output = testing::internal::GetCapturedStdout();
        log_level_set_stub(originalLogLevel);

        EXPECT_EQ(firstRet, HCCL_SUCCESS);
        EXPECT_EQ(cachedRet, HCCL_SUCCESS);
        const std::string expectedId = value != nullptr && std::string(value) != "EmptyString" ? value : "";
        EXPECT_EQ(config.GetLogicSuperPodId(), expectedId);
        const std::string prefix = "[HCCL_ENV] HCCL_LOGIC_SUPERPOD_ID ";
        const std::string message = expectedId.empty() ? prefix + "is not set, default value[]" :
                                                         prefix + "set by environment to [" + expectedId + "]";
        EXPECT_NE(output.find(message), std::string::npos);
        const size_t first = output.find(prefix);
        if (first != std::string::npos) {
            EXPECT_EQ(output.find(prefix, first + prefix.size()), std::string::npos);
        }
    }
}

TEST(ConfigMgrTest, Ut_LogicSuperPodId_When_Unconfigured_Expect_SuccessAndEmptyId)
{
    EnvGuard guard("HCCL_LOGIC_SUPERPOD_ID");
    for (const char* value : {static_cast<const char*>(nullptr), "", "EmptyString"}) {
        hccl::EnvLogicSuperPodConfig config;
        if (value == nullptr) {
            guard.Unset();
        } else {
            guard.Set(value);
        }
        ASSERT_EQ(config.Parse(), HCCL_SUCCESS);
        EXPECT_TRUE(config.GetLogicSuperPodId().empty());
    }
}

// 字节边界：127、128字节有效，129字节返回HCCL_E_PARA
TEST(ConfigMgrTest, Ut_LogicSuperPodId_When_LengthBoundary_Expect_128Ok129Para)
{
    EnvGuard guard("HCCL_LOGIC_SUPERPOD_ID");
    for (size_t length : {127U, 128U, 129U}) {
        SCOPED_TRACE(length);
        hccl::EnvLogicSuperPodConfig config;
        const std::string value(length, 'a');
        guard.Set(value.c_str());
        EXPECT_EQ(config.Parse(), length <= 128 ? HCCL_SUCCESS : HCCL_E_PARA);
        EXPECT_EQ(config.GetLogicSuperPodId(), length <= 128 ? value : std::string{});
    }
}

TEST(ConfigMgrTest, Ut_LogicSuperPodId_When_EnvChanged_Expect_ParseKeepsFirstSuccessfulValue)
{
    EnvGuard guard("HCCL_LOGIC_SUPERPOD_ID");
    hccl::EnvLogicSuperPodConfig config;
    const std::string logicId = " group_A ";

    guard.Set(logicId.c_str());
    ASSERT_EQ(config.Parse(), HCCL_SUCCESS);
    EXPECT_EQ(config.GetLogicSuperPodId(), logicId);

    guard.Set("group_B");
    ASSERT_EQ(config.Parse(), HCCL_SUCCESS);
    EXPECT_EQ(config.GetLogicSuperPodId(), logicId);

    guard.Set(std::string(129, 'a').c_str());
    EXPECT_EQ(config.Parse(), HCCL_SUCCESS);
    EXPECT_EQ(config.GetLogicSuperPodId(), logicId);
}

TEST(ConfigMgrTest, Ut_LogicSuperPodId_When_InitiallyEmpty_Expect_EmptyValueCached)
{
    EnvGuard guard("HCCL_LOGIC_SUPERPOD_ID");
    for (const char* value : {static_cast<const char*>(nullptr), "", "EmptyString"}) {
        hccl::EnvLogicSuperPodConfig config;
        if (value == nullptr) {
            guard.Unset();
        } else {
            guard.Set(value);
        }
        ASSERT_EQ(config.Parse(), HCCL_SUCCESS);
        guard.Set("group_A");
        EXPECT_EQ(config.Parse(), HCCL_SUCCESS);
        EXPECT_TRUE(config.GetLogicSuperPodId().empty());
    }
}

TEST(ConfigMgrTest, Ut_LogicSuperPodId_When_FirstParseFailed_Expect_CorrectedValueCanRetry)
{
    EnvGuard guard("HCCL_LOGIC_SUPERPOD_ID");
    guard.Set(std::string(129, 'a').c_str());
    hccl::EnvLogicSuperPodConfig config;
    extern s32 log_level_get_stub();
    extern void log_level_set_stub(s32 logLevel);
    const s32 originalLogLevel = log_level_get_stub();
    log_level_set_stub(DLOG_INFO);
    testing::internal::CaptureStdout();
    const HcclResult ret = config.Parse();
    const std::string output = testing::internal::GetCapturedStdout();
    log_level_set_stub(originalLogLevel);

    ASSERT_EQ(ret, HCCL_E_PARA);
    EXPECT_NE(
        output.find("[Parse][LogicSuperPodId]Invalid HCCL_LOGIC_SUPERPOD_ID env len, len is bigger than [128]."),
        std::string::npos);
    EXPECT_EQ(output.find("[HCCL_ENV] HCCL_LOGIC_SUPERPOD_ID "), std::string::npos);
    guard.Set("group_A");
    EXPECT_EQ(config.Parse(), HCCL_SUCCESS);
    EXPECT_EQ(config.GetLogicSuperPodId(), "group_A");
}

TEST(ConfigMgrTest, Ut_LogicSuperPodId_When_ConcurrentParse_Expect_SameCachedValue)
{
    EnvGuard guard("HCCL_LOGIC_SUPERPOD_ID");
    guard.Set("group_A");
    hccl::EnvLogicSuperPodConfig config;
    std::vector<HcclResult> results(8, HCCL_E_INTERNAL);
    std::vector<std::thread> threads;
    for (size_t i = 0; i < results.size(); ++i) {
        threads.emplace_back([&config, &results, i]() {
            results[i] = config.Parse();
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    for (HcclResult result : results) {
        EXPECT_EQ(result, HCCL_SUCCESS);
    }
    EXPECT_EQ(config.GetLogicSuperPodId(), "group_A");
    guard.Set("group_B");
    EXPECT_EQ(config.Parse(), HCCL_SUCCESS);
    EXPECT_EQ(config.GetLogicSuperPodId(), "group_A");
}

// ==================== EnvField 无 parser 函数 ====================

TEST(ConfigMgrTest, Ut_EnvField_NoParser_Expect_Error)
{
    EnvGuard guard("HCOMM_TEST_NO_PARSER");
    guard.Set("123");
    EnvField<uint32_t> field("HCOMM_TEST_NO_PARSER", 0U, nullptr);
    EXPECT_NE(field.Parse(), HCCL_SUCCESS);
}
