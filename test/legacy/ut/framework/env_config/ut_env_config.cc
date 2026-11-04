/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "env_config_stub.h"

#include "env_config_v2.h"
#include "orion_adapter_rts.h"
#include "dtype_common.h" // 全局 hrtGetDeviceType 声明（代际谓词真值源）
#include "gtest/gtest.h"
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#include <stdexcept>
#include <climits>
#include <algorithm>
#include <string>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include "invalid_params_exception.h"
#include "adapter_error_manager_pub.h"
#include "env_func.h"
#include "plf_debug_config.h"
#include "config_plf_log_v2.h"

using namespace Hccl;

std::map<std::string, std::string> envCfgMap = defaultEnvCfgMap;

char* getenv_stub(const char* __name)
{
    char* ret = const_cast<char*>(envCfgMap[std::string(__name)].c_str());
    return ret;
}

class EnvConfigTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "EnvConfigTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "EnvConfigTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in EnvConfigTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in EnvConfigTest TearDown" << std::endl;
    }

    bool CmpIpAddress(const IpAddress& ip1, const IpAddress& ip2)
    {
        if (ip1.GetFamily() != ip2.GetFamily()) {
            return false;
        }
        if (ip1.GetFamily() == AF_INET) {
            return (ip1.GetBinaryAddress().addr.s_addr == ip2.GetBinaryAddress().addr.s_addr);
        } else {
            auto biAddr1 = ip1.GetBinaryAddress();
            auto biAddr2 = ip2.GetBinaryAddress();
            return (memcmp(&biAddr1.addr6, &biAddr2.addr6, sizeof(biAddr1.addr6)) == 0);
        }
    }

    bool CmpSocketIfName(const SocketIfName& fiName1, const SocketIfName& fiName2)
    {
        return (fiName1.configIfNames == fiName2.configIfNames) && (fiName1.searchNot == fiName2.searchNot)
               && (fiName1.searchExact == fiName2.searchExact);
    }

protected:
    void MockFunc()
    {
        MOCKER(getenv).stubs().with(mockcpp::any()).will(invoke(getenv_stub));

        char c = '1';
        MOCKER(realpath).stubs().with(mockcpp::any()).will(returnValue(&c));

        MOCKER(HrtGetDeviceType).stubs().will(returnValue(static_cast<Hccl::DevType>(Hccl::DevType::DEV_TYPE_910A)));

        // 代际谓词 hcomm::EqualToOrGreaterThan950 的真值源为全局 hrtGetDeviceType（UT 桩默认 950），
        // 需与 HrtGetDeviceType 桩保持一致返回 910 代，保证 910A 超时取整分支被覆盖
        MOCKER(hrtGetDeviceType).stubs().with(outBound(::DevType::DEV_TYPE_910)).will(returnValue(HCCL_SUCCESS));
    }

    void ResetEnvCfgMap()
    {
        envCfgMap.clear();
        envCfgMap = defaultEnvCfgMap;
    }

    void GenFile(const std::string& filePath, const std::string fileContent)
    {
        try {
            std::ofstream out(filePath.c_str(), std::ofstream::out);
            out << fileContent;
        } catch (...) {
            std::cout << filePath << " generate failed!" << std::endl;
            return;
        }
        std::cout << filePath << " generated." << std::endl;
    }

    void DelFile(const std::string& filePath)
    {
        int res = unlink(filePath.c_str());
        if (res == -1) {
            std::cout << filePath << " delete failed!" << std::endl;
            return;
        }
        std::cout << filePath << " deleted." << std::endl;
    }
};

TEST_F(EnvConfigTest, parse_env_config)
{
    // 使用真实的单例EnvConfig，提升覆盖率
    // 由于是单例，只能使用一个测试用例
    ResetEnvCfgMap();

    MockFunc();

    EnvConfig::GetInstance().GetHostNicConfig();
    EnvConfig::GetInstance().GetSocketConfig();
    EnvConfig::GetInstance().GetRtsConfig();
    EnvConfig::GetInstance().GetRdmaConfig();
    EnvConfig::GetInstance().GetAlgoConfig();
    EnvConfig::GetInstance().GetLogConfig();
    EnvConfig::GetInstance().GetDetourConfig();
}

TEST_F(EnvConfigTest, Ut_CastDeterministic_WhenEnvValueIsValidOrInvalid_ExpectMappedOrThrow)
{
    EXPECT_EQ(CastDeterministic("false"), 0U);
    EXPECT_EQ(CastDeterministic("true"), 1U);
    EXPECT_EQ(CastDeterministic("TRUE"), 1U);
    EXPECT_EQ(CastDeterministic("strict"), 2U);
    EXPECT_EQ(CastDeterministic("StRiCt"), 2U);
    EXPECT_THROW(CastDeterministic("invalid"), InvalidParamsException);
}

TEST_F(EnvConfigTest, Ut_EnvAlgoConfigParse_WhenDeterministicSetByEnv_ExpectEnvironmentSource)
{
    MOCKER(getenv).stubs().with(mockcpp::any()).will(invoke(getenv_stub));
    struct TestCase {
        const char* environmentValue;
        u8 expectedValue;
        bool expectedFromEnvironment;
    };
    const TestCase testCases[] = {
        {nullptr, 1U, false}, {"EmptyString", 1U, false}, {"false", 0U, true}, {"TRUE", 1U, true}, {"StRiCt", 2U, true},
    };

    for (const auto& testCase : testCases) {
        envCfgMap.clear();
        if (testCase.environmentValue != nullptr) {
            envCfgMap["HCCL_DETERMINISTIC"] = testCase.environmentValue;
        }
        EnvAlgoConfig algoConfig;
        algoConfig.Parse();

        EXPECT_EQ(algoConfig.GetDeterministic(), testCase.expectedValue);
        EXPECT_EQ(algoConfig.IsDeterministicSetByEnvironment(), testCase.expectedFromEnvironment);
    }
}

TEST_F(EnvConfigTest, Ut_EnvAlgoConfigParse_WhenDeterministicInvalid_ExpectThrow)
{
    MOCKER(getenv).stubs().with(mockcpp::any()).will(invoke(getenv_stub));
    envCfgMap = {};
    envCfgMap["HCCL_DETERMINISTIC"] = "invalid";
    EnvAlgoConfig algoConfig;

    EXPECT_THROW(algoConfig.Parse(), InvalidParamsException);
}

TEST_F(EnvConfigTest, Ut_EnvConfigParse_WhenEnvironmentInvalid_ExpectEI0001Report)
{
    static std::string reportedErrorCode;
    static std::vector<std::string> reportedKeys;
    static std::vector<std::string> reportedValues;
    MOCKER(getenv).stubs().with(mockcpp::any()).will(invoke(getenv_stub));
    MOCKER(RptInputErr)
        .stubs()
        .will(invoke(+[](std::string errorCode, std::vector<std::string> keys, std::vector<std::string> values) {
            reportedErrorCode = errorCode;
            reportedKeys = keys;
            reportedValues = values;
        }));
    struct InvalidEnvCase {
        const char* name;
        const char* value;
        const char* expectedReason;
    };
    constexpr InvalidEnvCase invalidEnvCases[]
        = {{"HCCL_RDMA_TIMEOUT", "-1", "contains non-digit char"},
           {"HCCL_BUFFSIZE", "-1", "contains non-digit char"},
           {"HCCL_DETERMINISTIC", "invalid", "Should be false, true or strict"}};

    for (const auto& envCase : invalidEnvCases) {
        ResetEnvCfgMap();
        envCfgMap[envCase.name] = envCase.value;
        reportedErrorCode.clear();
        reportedKeys.clear();
        reportedValues.clear();

        if (std::string(envCase.name) == "HCCL_RDMA_TIMEOUT") {
            EnvRdmaConfig config;
            EXPECT_THROW(config.Parse(), InvalidParamsException);
        } else {
            EnvAlgoConfig config;
            EXPECT_THROW(config.Parse(), InvalidParamsException);
        }

        EXPECT_EQ(reportedErrorCode, "EI0001");
        EXPECT_EQ(reportedKeys, std::vector<std::string>({"value", "env", "expect"}));
        ASSERT_EQ(reportedValues.size(), 3U);
        EXPECT_EQ(reportedValues[0], envCase.value);
        EXPECT_EQ(reportedValues[1], envCase.name);
        EXPECT_NE(reportedValues[2].find(envCase.expectedReason), std::string::npos);
    }
    ResetEnvCfgMap();
}

TEST_F(EnvConfigTest, Ut_EnvAlgoConfigSet_WhenDeterministicUnset_ExpectDefaultThenUpdated)
{
    MOCKER(getenv).stubs().with(mockcpp::any()).will(invoke(getenv_stub));
    envCfgMap = {};
    EnvAlgoConfig algoConfig;
    algoConfig.Parse();

    EXPECT_EQ(algoConfig.GetDeterministic(), 1U);
    EXPECT_FALSE(algoConfig.IsDeterministicSetByEnvironment());

    algoConfig.SetDeterministic(static_cast<u8>(2));
    EXPECT_EQ(algoConfig.GetDeterministic(), 2U);
}

TEST_F(EnvConfigTest, parse_env_config_should_success)
{
    ResetEnvCfgMap();

    MockFunc();

    try {
        EnvConfigStub envCfg;
        EXPECT_EQ(CmpIpAddress(envCfg.GetHostNicConfig().GetControlIfIp(), IpAddress("10.10.10.1")), true);
        EXPECT_EQ(envCfg.GetHostNicConfig().GetIfBasePort(), 50000);
        EXPECT_EQ(envCfg.GetHostNicConfig().GetWhitelistDisable(), false);
        EXPECT_EQ(envCfg.GetHostNicConfig().GetWhiteListFile(), "");
        EXPECT_EQ(envCfg.GetSocketConfig().GetSocketFamily(), AF_INET6);
        EXPECT_EQ(envCfg.GetSocketConfig().GetLinkTimeOut(), 200);
        EXPECT_EQ(envCfg.GetRtsConfig().GetExecTimeOut(), 1768);
        EXPECT_EQ(envCfg.GetRdmaConfig().GetRdmaTrafficClass(), 100);
        EXPECT_EQ(envCfg.GetRdmaConfig().GetRdmaServerLevel(), 3);
        EXPECT_EQ(envCfg.GetRdmaConfig().GetRdmaTimeOut(), 6);
        EXPECT_EQ(envCfg.GetRdmaConfig().GetRdmaRetryCnt(), 5);
        EXPECT_EQ(envCfg.GetRdmaConfig().GetRdmaQueueNum(), 1);
        EXPECT_EQ(envCfg.GetRdmaConfig().GetRdmaMultiQpThreshold(), 524288); // 512 KB 转 524288 B
        EXPECT_EQ(envCfg.GetAlgoConfig().GetPrimQueueGenName(), "AllReduceRing");
        std::map<OpType, std::vector<HcclAlgoType>> algoMap = {};
        EXPECT_EQ(envCfg.GetAlgoConfig().GetAlgoConfig(), algoMap);
        EXPECT_EQ(envCfg.GetAlgoConfig().GetBuffSize(), 200 * 1024 * 1024);
        EXPECT_EQ(envCfg.GetAlgoConfig().GetDeterministic(), 0U);
        EXPECT_EQ(envCfg.GetLogConfig().GetEntryLogEnable(), true);
        EXPECT_EQ(envCfg.GetLogConfig().GetCannVersion(), "");
        EXPECT_EQ(envCfg.GetDetourConfig().GetDetourType(), HcclDetourType::HCCL_DETOUR_ENABLE_2P);
    } catch (...) {
    }
}
/*
TEST_F(EnvConfigTest, parse_env_config_should_success2)
{
    ResetEnvCfgMap();
    envCfgMap["HCCL_NPU_NET_PROTOCOL"] = "TCP";
    envCfgMap["HCCL_SOCKET_FAMILY"] = "AF_INET";
    envCfgMap["LD_LIBRARY_PATH"] = "/temp:/latest";

    MOCKER(getenv)
    .stubs()
    .with(mockcpp::any())
    .will(invoke(getenv_stub));

    char c = '1';
    MOCKER(realpath)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(&c));

    MOCKER(HrtGetDeviceType)
    .stubs()
    .will(returnValue((DevType)DevType::DEV_TYPE_910A3));

    EnvConfigStub envCfg;

    EXPECT_EQ(CmpIpAddress(envCfg.GetHostNicConfig().GetControlIfIp(), IpAddress("10.10.10.1")), true);
    EXPECT_EQ(envCfg.GetHostNicConfig().GetIfBasePort(), 50000);
    EXPECT_EQ(CmpSocketIfName(envCfg.GetHostNicConfig().GetSocketIfName(), SocketIfName({{"eth0", "endvnic"}, true,
true})), true); EXPECT_EQ(envCfg.GetHostNicConfig().GetWhitelistDisable(), false);
    EXPECT_EQ(envCfg.GetHostNicConfig().GetWhiteListFile(), "");
    EXPECT_EQ(envCfg.GetSocketConfig().GetSocketFamily(), AF_INET);
    EXPECT_EQ(envCfg.GetSocketConfig().GetLinkTimeOut(), 200);
    EXPECT_EQ(envCfg.GetRtsConfig().GetExecTimeOut(), 1800);
    EXPECT_EQ(envCfg.GetRdmaConfig().GetRdmaTrafficClass(), 100);
    EXPECT_EQ(envCfg.GetRdmaConfig().GetRdmaServerLevel(), 3);
    EXPECT_EQ(envCfg.GetRdmaConfig().GetRdmaTimeOut(), 6);
    EXPECT_EQ(envCfg.GetRdmaConfig().GetRdmaRetryCnt(), 5);
    EXPECT_EQ(envCfg.GetAlgoConfig().GetPrimQueueGenName(), "AllReduceRing");
    EXPECT_EQ(envCfg.GetAlgoConfig().GetAlgoConfig(), vector<HcclAlgoType>({HcclAlgoType::HCCL_ALGO_TYPE_RING,
HcclAlgoType::HCCL_ALGO_TYPE_RING, HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT, HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT}));
    EXPECT_EQ(envCfg.GetAlgoConfig().GetBuffSize(), 200*1024*1024);
    EXPECT_EQ(envCfg.GetAlgoConfig().GetOpExpansionMode(), OpExpansionMode::AI_CPU);
    EXPECT_EQ(envCfg.GetLogConfig().GetEntryLogEnable(), true);
    EXPECT_EQ(envCfg.GetLogConfig().GetCannVersion(), "");
}

TEST_F(EnvConfigTest, parse_env_config_should_success3)
{
    ResetEnvCfgMap();
    envCfgMap["HCCL_NPU_NET_PROTOCOL"] = "";
    envCfgMap["HCCL_ALGO"] = "";

    MockFunc();

    EnvConfigStub envCfg;

    EXPECT_EQ(CmpIpAddress(envCfg.GetHostNicConfig().GetControlIfIp(), IpAddress("10.10.10.1")), true);
    EXPECT_EQ(envCfg.GetHostNicConfig().GetIfBasePort(), 50000);
    EXPECT_EQ(CmpSocketIfName(envCfg.GetHostNicConfig().GetSocketIfName(), SocketIfName({{"eth0", "endvnic"}, true,
true})), true); EXPECT_EQ(envCfg.GetHostNicConfig().GetWhitelistDisable(), false);
    EXPECT_EQ(envCfg.GetHostNicConfig().GetWhiteListFile(), "");
    EXPECT_EQ(envCfg.GetSocketConfig().GetSocketFamily(), AF_INET6);
    EXPECT_EQ(envCfg.GetSocketConfig().GetLinkTimeOut(), 200);
    EXPECT_EQ(envCfg.GetRtsConfig().GetExecTimeOut(), 1768);
    EXPECT_EQ(envCfg.GetRdmaConfig().GetRdmaTrafficClass(), 100);
    EXPECT_EQ(envCfg.GetRdmaConfig().GetRdmaServerLevel(), 3);
    EXPECT_EQ(envCfg.GetRdmaConfig().GetRdmaTimeOut(), 6);
    EXPECT_EQ(envCfg.GetRdmaConfig().GetRdmaRetryCnt(), 5);
    EXPECT_EQ(envCfg.GetAlgoConfig().GetPrimQueueGenName(), "AllReduceRing");
    EXPECT_EQ(envCfg.GetAlgoConfig().GetAlgoConfig(), vector<HcclAlgoType>({HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT,
HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT, HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT, HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT}));
    EXPECT_EQ(envCfg.GetAlgoConfig().GetBuffSize(), 200*1024*1024);
    EXPECT_EQ(envCfg.GetAlgoConfig().GetOpExpansionMode(), OpExpansionMode::AI_CPU);
    EXPECT_EQ(envCfg.GetLogConfig().GetEntryLogEnable(), true);
    EXPECT_EQ(envCfg.GetLogConfig().GetCannVersion(), "");
}

TEST_F(EnvConfigTest, parse_env_config_HCCL_SOCKET_IFNAME_should_fail)
{
    ResetEnvCfgMap();
    envCfgMap["HCCL_SOCKET_IFNAME"] = "^=eth0,,endvnic";

    MockFunc();

    EXPECT_THROW(EnvConfigStub envCfg, InvalidParamsException);
}

TEST_F(EnvConfigTest, parse_env_config_HCCL_INTRA_PCIE_ENABLE_should_fail)
{
    ResetEnvCfgMap();
    envCfgMap["HCCL_INTRA_PCIE_ENABLE"] = "true";
    envCfgMap["HCCL_INTRA_ROCE_ENABLE"] = "true";

    MockFunc();

    EXPECT_THROW(EnvConfigStub envCfg, InvalidParamsException);
}

TEST_F(EnvConfigTest, parse_env_config_HCCL_NPU_NET_PROTOCOL_should_fail)
{
    ResetEnvCfgMap();
    char proto[NPU_NET_PROTOCOL_MAX_LEN + 1] = {};
    std::fill_n(proto, NPU_NET_PROTOCOL_MAX_LEN, 1);
    // longer than NPU_NET_PROTOCOL_MAX_LEN
    envCfgMap["HCCL_NPU_NET_PROTOCOL"] = std::string(proto);

    MockFunc();

    EXPECT_THROW(EnvConfigStub envCfg, InvalidParamsException);
}

TEST_F(EnvConfigTest, parse_env_config_HCCL_NPU_NET_PROTOCOL_proto_should_fail)
{
    ResetEnvCfgMap();
    // not TCP or RDMA
    envCfgMap["HCCL_NPU_NET_PROTOCOL"] = "PCIE";

    MockFunc();

    EXPECT_THROW(EnvConfigStub envCfg, InvalidParamsException);
}

TEST_F(EnvConfigTest, parse_env_config_HCCL_ALGO_should_fail)
{
    ResetEnvCfgMap();
    envCfgMap["HCCL_ALGO"] = ":NA;level1:";

    MockFunc();

    EXPECT_THROW(EnvConfigStub envCfg, InvalidParamsException);
}

TEST_F(EnvConfigTest, parse_env_config_HCCL_ALGO_level_should_fail)
{
    ResetEnvCfgMap();
    // not defined level
    envCfgMap["HCCL_ALGO"] = "level0:NA;level4:ring";

    MockFunc();

    EXPECT_THROW(EnvConfigStub envCfg, InvalidParamsException);
}

TEST_F(EnvConfigTest, parse_env_config_HCCL_ALGO_algo_should_fail)
{
    ResetEnvCfgMap();
    // not defined algo
    envCfgMap["HCCL_ALGO"] = "level0:somealgo;level3:ring";

    MockFunc();

    EXPECT_THROW(EnvConfigStub envCfg, InvalidParamsException);
}

TEST_F(EnvConfigTest, parse_env_config_HCCL_ALGO_duplicate_level_should_fail)
{
    ResetEnvCfgMap();
    // duplicate level
    envCfgMap["HCCL_ALGO"] = "level0:NA;level0:ring";

    MockFunc();

    EXPECT_THROW(EnvConfigStub envCfg, InvalidParamsException);
}

TEST_F(EnvConfigTest, parse_env_config_HCCL_ALGO_level_num_should_fail)
{
    ResetEnvCfgMap();
    // too many levels
    envCfgMap["HCCL_ALGO"] = "level0:NA;level0:NA;level0:NA;level0:NA;level0:NA";

    MockFunc();

    EXPECT_THROW(EnvConfigStub envCfg, InvalidParamsException);
}

TEST_F(EnvConfigTest, parse_env_config_HCCL_SOCKET_FAMILY_should_fail)
{
    ResetEnvCfgMap();
    envCfgMap["HCCL_SOCKET_FAMILY"] = "AF_INET4";

    MockFunc();

    EXPECT_THROW(EnvConfigStub envCfg, InvalidParamsException);
}

TEST_F(EnvConfigTest, parse_env_config_HCCL_WHITELIST_FILE_should_fail)
{
    ResetEnvCfgMap();
    char path[PATH_MAX + 1] = {};
    std::fill_n(path, PATH_MAX, 1);
    envCfgMap["HCCL_WHITELIST_FILE"] = std::string(path);

    MockFunc();

    EXPECT_THROW(EnvConfigStub envCfg, InvalidParamsException);
}
*/

TEST_F(EnvConfigTest, parse_env_config_hccl_algo_invalid_test_1)
{
    std::string str1 = "level0:yyy;level1:xxxx";
    std::map<OpType, std::vector<HcclAlgoType>> algoConfig1;
    EXPECT_EQ(SetHcclAlgoConfig(str1, algoConfig1), HcclResult::HCCL_E_PARA);

    std::string str2 = "abcdefg";
    std::map<OpType, std::vector<HcclAlgoType>> algoConfig2;
    EXPECT_EQ(SetHcclAlgoConfig(str2, algoConfig2), HcclResult::HCCL_E_PARA);
}

TEST_F(EnvConfigTest, parse_env_config_HCCL_DETOUR_test)
{
    std::string input = "detour:0";
    EXPECT_EQ(CastDetourType(input), HcclDetourType::HCCL_DETOUR_DISABLE);
    input = "detour:1";
    EXPECT_EQ(CastDetourType(input), HcclDetourType::HCCL_DETOUR_ENABLE_2P);
    input = "detour:2";
    EXPECT_THROW(CastDetourType(input), NotSupportException);
    input = "detour:3";
    EXPECT_THROW(CastDetourType(input), NotSupportException);
    input = "!";
    EXPECT_THROW(CastDetourType(input), NotSupportException);
}

TEST_F(EnvConfigTest, str2T_test)
{
    std::string input = "12a";
    EXPECT_THROW(Str2T<int>(input), InvalidParamsException);
}

// 临时方案
TEST_F(EnvConfigTest, parse_env_config_socketIFName_test)
{
    std::string input = "=eth0,endvnic";
    EXPECT_NO_THROW(CastSocketIfName(input));
}

TEST_F(EnvConfigTest, CastAlgoTypeVec_test)
{
    std::string str = "level0null";
    EXPECT_THROW(CastAlgoTypeVec(str), InvalidParamsException);
}

TEST_F(EnvConfigTest, Ut_CastSocketPortRange_When_Config_Auto_Expect_Right)
{
    std::vector<SocketPortRange> rangs;
    SocketPortRange autoSocketPortRange = {HCCL_SOCKET_PORT_RANGE_AUTO, HCCL_SOCKET_PORT_RANGE_AUTO};
    rangs.push_back(autoSocketPortRange);
    EXPECT_EQ(CastSocketPortRange(HCCL_AUTO_PORT_CONFIG, "envName"), rangs);
}

TEST_F(EnvConfigTest, Ut_CastSocketPortRange_When_Config_Whitespace_Expect_Erase_Return_OK)
{
    std::vector<SocketPortRange> rangs;
    SocketPortRange autoSocketPortRange = {60000, 60050};
    rangs.push_back(autoSocketPortRange);
    EXPECT_EQ(CastSocketPortRange(" 60000-60050 ", "envName"), rangs);
}

TEST_F(EnvConfigTest, Ut_CastSocketPortRange_When_Config_More_Expect_Return_OK)
{
    std::vector<SocketPortRange> rangs;
    rangs.push_back(SocketPortRange{50000, 50000});
    rangs.push_back(SocketPortRange{60000, 60050});
    rangs.push_back(SocketPortRange{60100, 60260});
    EXPECT_EQ(CastSocketPortRange("50000,60000-60050, 60100-60260", "envName"), rangs);
}

TEST_F(EnvConfigTest, Ut_CastSocketPortRange_When_Config_Bound_Error_Expect_Throw)
{
    EXPECT_THROW(CastSocketPortRange("50000,60050-60000, 60100-60260", "envName"), InvalidParamsException);
    EXPECT_THROW(CastSocketPortRange("50000,60000-60150,60100-60260", "envName"), InvalidParamsException);
}

TEST_F(EnvConfigTest, Ut_CastSocketPortRange_When_Config_Invalid_Expect_Throw)
{
    EXPECT_THROW(CastSocketPortRange("60000-60050,0,60100-60260", "envName"), InvalidParamsException);
    EXPECT_THROW(CastSocketPortRange("50000,60000-60050,0-0", "envName"), InvalidParamsException);
    EXPECT_THROW(CastSocketPortRange("65536", "envName"), InvalidParamsException);
}

TEST_F(EnvConfigTest, Ut_CastHcclAccelerator_When_ConfigVaild_ExpectSuccess)
{
    EXPECT_EQ(CastHcclAccelerator("AI_CPU"), HcclAccelerator::AICPU_TS);
    EXPECT_EQ(CastHcclAccelerator("AICPU_TS"), HcclAccelerator::AICPU_TS);
    EXPECT_EQ(CastHcclAccelerator("AIV"), HcclAccelerator::AIV);
    EXPECT_EQ(CastHcclAccelerator("CCU_MS"), HcclAccelerator::CCU_MS);
    EXPECT_EQ(CastHcclAccelerator("CCU_SCHED"), HcclAccelerator::CCU_SCHED);
}

TEST_F(EnvConfigTest, Ut_CastHcclAccelerator_When_ConfigInvaild_ExpectThrow)
{
    EXPECT_THROW(CastHcclAccelerator("Invalid"), InvalidParamsException);
    EXPECT_THROW(CastHcclAccelerator("AIV_ONLY"), InvalidParamsException);
    EXPECT_THROW(CastHcclAccelerator("HOST"), InvalidParamsException);
    EXPECT_THROW(CastHcclAccelerator("HOST_TS"), InvalidParamsException);
}

TEST_F(EnvConfigTest, Ut_GetRdmaQueueNum_OutOfRange_ReturnsException)
{
    setenv("HCCL_RDMA_QPS_PER_CONNECTION", "33", 1);
    EnvRdmaConfig rdmaConfig;
    EXPECT_THROW(rdmaConfig.Parse(), InvalidParamsException);
    unsetenv("HCCL_RDMA_QPS_PER_CONNECTION");
}

TEST_F(EnvConfigTest, Ut_GetRdmaMultiQpThreshold_ValidValue_ReturnsCorrectValue)
{
    setenv("HCCL_MULTI_QP_THRESHOLD", "1123", 1);
    EnvRdmaConfig rdmaConfig;
    rdmaConfig.Parse();
    EXPECT_EQ(rdmaConfig.GetRdmaMultiQpThreshold(), 1149952); // 1123 KB 转 1149952 B
    unsetenv("HCCL_MULTI_QP_THRESHOLD");
}

TEST_F(EnvConfigTest, Ut_GetRdmaMultiQpThreshold_OutOfRange_ReturnsException)
{
    setenv("HCCL_MULTI_QP_THRESHOLD", "-5", 1);
    EnvRdmaConfig rdmaConfig;
    EXPECT_THROW(rdmaConfig.Parse(), InvalidParamsException);
    unsetenv("HCCL_MULTI_QP_THRESHOLD");
}

TEST_F(EnvConfigTest, Ut_EnvPlfDebugConfig_When_SetTask_Expect_PlfTaskBitSet)
{
    setenv("HCCL_DEBUG_CONFIG", "task", 1);
    EnvPlfDebugConfig plfCfg;
    plfCfg.Parse();
    EXPECT_EQ(plfCfg.GetConfigValue(), PLF_TASK);
    unsetenv("HCCL_DEBUG_CONFIG");
}

TEST_F(EnvConfigTest, Ut_EnvPlfDebugConfig_When_SetDataOp_Expect_PlfDataOpBitSet)
{
    setenv("HCOMM_DEBUG_CONFIG", "data_op", 1);
    EnvPlfDebugConfig plfCfg;
    plfCfg.Parse();
    EXPECT_EQ(plfCfg.GetConfigValue(), PLF_DATA_OP);
    unsetenv("HCOMM_DEBUG_CONFIG");
}

TEST_F(EnvConfigTest, Ut_EnvPlfDebugConfig_When_SetMultiTokens_Expect_MultiBitsSet)
{
    setenv("HCCL_DEBUG_CONFIG", "task", 1);
    setenv("HCOMM_DEBUG_CONFIG", "data_op", 1);
    EnvPlfDebugConfig plfCfg;
    plfCfg.Parse();
    EXPECT_EQ(plfCfg.GetConfigValue(), PLF_TASK | PLF_DATA_OP);
    unsetenv("HCCL_DEBUG_CONFIG");
    unsetenv("HCOMM_DEBUG_CONFIG");
}

TEST_F(EnvConfigTest, Ut_EnvPlfDebugConfig_When_InvertTask_Expect_AllBitExceptTaskSet)
{
    setenv("HCCL_DEBUG_CONFIG", "^task", 1);
    EnvPlfDebugConfig plfCfg;
    plfCfg.Parse();
    EXPECT_EQ(plfCfg.GetConfigValue(), PLF_ALG | PLF_RES);
    unsetenv("HCCL_DEBUG_CONFIG");
}

TEST_F(EnvConfigTest, Ut_EnvPlfDebugConfig_When_SetInvalidToken_Expect_ParseFailed)
{
    setenv("HCCL_DEBUG_CONFIG", "INVALID", 1);
    EnvPlfDebugConfig plfCfg;
    EXPECT_NE(plfCfg.Parse(), HCCL_SUCCESS);
    EXPECT_EQ(plfCfg.GetConfigValue(), 0ULL);
    unsetenv("HCCL_DEBUG_CONFIG");
}

TEST_F(EnvConfigTest, Ut_EnvPlfDebugConfig_When_SetAivOpsExc_Expect_ParseFailed)
{
    setenv("HCCL_DEBUG_CONFIG", "AIV_OPS_EXC", 1);
    EnvPlfDebugConfig plfCfg;
    EXPECT_NE(plfCfg.Parse(), HCCL_SUCCESS);
    unsetenv("HCCL_DEBUG_CONFIG");
}

TEST_F(EnvConfigTest, Ut_EnvPlfDebugConfig_When_HcommCrossDomainToken_Expect_ParseFailed)
{
    setenv("HCOMM_DEBUG_CONFIG", "ALG", 1);
    EnvPlfDebugConfig plfCfg;
    EXPECT_NE(plfCfg.Parse(), HCCL_SUCCESS);
    unsetenv("HCOMM_DEBUG_CONFIG");
}

TEST_F(EnvConfigTest, Ut_EnvPlfDebugConfig_When_ValueHasSpace_Expect_ParseFailed)
{
    setenv("HCCL_DEBUG_CONFIG", "alg, task ", 1);
    EnvPlfDebugConfig plfCfg;
    EXPECT_NE(plfCfg.Parse(), HCCL_SUCCESS);
    unsetenv("HCCL_DEBUG_CONFIG");
}

TEST_F(EnvConfigTest, Ut_EnvPlfDebugConfig_When_OneEnvInvalid_Expect_ParseFailed)
{
    // 非法配置整体失败（不保留另一个变量的合法项），仍由 InitFullMode 中断建链
    setenv("HCCL_DEBUG_CONFIG", "INVALID", 1);
    setenv("HCOMM_DEBUG_CONFIG", "data_op", 1);
    EnvPlfDebugConfig plfCfg;
    EXPECT_NE(plfCfg.Parse(), HCCL_SUCCESS);
    EXPECT_EQ(plfCfg.GetConfigValue(), 0ULL);
    unsetenv("HCCL_DEBUG_CONFIG");
    unsetenv("HCOMM_DEBUG_CONFIG");
}

TEST_F(EnvConfigTest, Ut_EnvPlfDebugConfig_When_CaseInsensitive_Expect_SameResult)
{
    setenv("HCCL_DEBUG_CONFIG", "TASK", 1);
    EnvPlfDebugConfig plfCfg;
    plfCfg.Parse();
    EXPECT_EQ(plfCfg.GetConfigValue(), PLF_TASK);
    unsetenv("HCCL_DEBUG_CONFIG");
}

// ==================== E1-E10: HCCL_RDMA_QP_PORT_CONFIG_PATH ====================
// These cases use real filesystem (mkdtemp, write cfg file) so realpath is NOT mocked.
// They follow the same setenv/unsetenv pattern as other EnvRdmaConfig tests above.
static std::string CreateTempDirForQpPort()
{
    char tpl[] = "/tmp/hcomm_ut_qpport_XXXXXX";
    char* dir = mkdtemp(tpl);
    return (dir != nullptr) ? std::string(dir) : "";
}

static void RemoveDirRecursive(const std::string& dir)
{
    std::string cmd = "rm -rf " + dir;
    (void)system(cmd.c_str());
}

static void WriteCfgFile(const std::string& dir, const std::string& content)
{
    std::string filePath = dir + "/MultiQpSrcPort.cfg";
    std::ofstream out(filePath.c_str(), std::ofstream::out);
    out << content;
    out.close();
}

// E1: 环境变量未设置 → GetMultiQpSrcPortConfig().IsAvailable()==false
TEST_F(EnvConfigTest, Ut_EnvRdmaConfigGetMultiQpSrcPortConfig_When_EnvNotSet_Expect_NotAvailable)
{
    unsetenv("HCCL_RDMA_QP_PORT_CONFIG_PATH");
    EnvRdmaConfig rdmaConfig;
    rdmaConfig.Parse();
    EXPECT_FALSE(rdmaConfig.GetMultiQpSrcPortConfig().IsAvailable());
    EXPECT_TRUE(rdmaConfig.GetMultiQpSrcPortConfig().configDirPath.empty());
}

// E2: 环境变量设为空字符串 → 同 E1
TEST_F(EnvConfigTest, Ut_EnvRdmaConfigGetMultiQpSrcPortConfig_When_EnvSetEmpty_Expect_NotAvailable)
{
    setenv("HCCL_RDMA_QP_PORT_CONFIG_PATH", "", 1);
    EnvRdmaConfig rdmaConfig;
    rdmaConfig.Parse();
    EXPECT_FALSE(rdmaConfig.GetMultiQpSrcPortConfig().IsAvailable());
    EXPECT_TRUE(rdmaConfig.GetMultiQpSrcPortConfig().configDirPath.empty());
    unsetenv("HCCL_RDMA_QP_PORT_CONFIG_PATH");
}

// E3: 环境变量指向有效目录+合法 cfg 文件 → IsAvailable()==true, ipPairToPorts 非空
TEST_F(EnvConfigTest, Ut_EnvRdmaConfigGetMultiQpSrcPortConfig_When_ValidDirWithCfgFile_Expect_Available)
{
    std::string tmpDir = CreateTempDirForQpPort();
    ASSERT_FALSE(tmpDir.empty());
    WriteCfgFile(tmpDir, "192.168.1.1,192.168.1.2=10001,10002\n");

    setenv("HCCL_RDMA_QP_PORT_CONFIG_PATH", tmpDir.c_str(), 1);
    EnvRdmaConfig rdmaConfig;
    rdmaConfig.Parse();
    EXPECT_TRUE(rdmaConfig.GetMultiQpSrcPortConfig().IsAvailable());
    EXPECT_EQ(rdmaConfig.GetMultiQpSrcPortConfig().ipPairToPorts.size(), 1u);
    EXPECT_NE(
        rdmaConfig.GetMultiQpSrcPortConfig().ipPairToPorts.find("192.168.1.1,192.168.1.2"),
        rdmaConfig.GetMultiQpSrcPortConfig().ipPairToPorts.end());

    unsetenv("HCCL_RDMA_QP_PORT_CONFIG_PATH");
    RemoveDirRecursive(tmpDir);
}

// E4: 环境变量指向有效目录+空 cfg 文件 → IsAvailable()==false
TEST_F(EnvConfigTest, Ut_EnvRdmaConfigGetMultiQpSrcPortConfig_When_ValidDirButEmptyCfg_Expect_NotAvailable)
{
    std::string tmpDir = CreateTempDirForQpPort();
    ASSERT_FALSE(tmpDir.empty());
    WriteCfgFile(tmpDir, "");

    setenv("HCCL_RDMA_QP_PORT_CONFIG_PATH", tmpDir.c_str(), 1);
    EnvRdmaConfig rdmaConfig;
    rdmaConfig.Parse();
    EXPECT_FALSE(rdmaConfig.GetMultiQpSrcPortConfig().IsAvailable());

    unsetenv("HCCL_RDMA_QP_PORT_CONFIG_PATH");
    RemoveDirRecursive(tmpDir);
}

// E5: 环境变量指向不存在的路径，或有效目录中缺少 cfg 文件 → 抛 InvalidParamsException
TEST_F(EnvConfigTest, Ut_EnvRdmaConfigParse_When_InvalidPath_Expect_Throw)
{
    setenv("HCCL_RDMA_QP_PORT_CONFIG_PATH", "/nonexistent/path/for/ut/test", 1);
    EnvRdmaConfig rdmaConfig;
    EXPECT_THROW(rdmaConfig.Parse(), InvalidParamsException);
    unsetenv("HCCL_RDMA_QP_PORT_CONFIG_PATH");

    std::string tmpDir = CreateTempDirForQpPort();
    ASSERT_FALSE(tmpDir.empty());
    setenv("HCCL_RDMA_QP_PORT_CONFIG_PATH", tmpDir.c_str(), 1);
    EnvRdmaConfig missingFileConfig;
    EXPECT_THROW(missingFileConfig.Parse(), InvalidParamsException);
    EXPECT_FALSE(missingFileConfig.GetMultiQpSrcPortConfig().IsAvailable());
    unsetenv("HCCL_RDMA_QP_PORT_CONFIG_PATH");
    RemoveDirRecursive(tmpDir);
}

// E6: 环境变量长度 >= PATH_MAX → CfgField CheckFilePath validate 抛 InvalidParamsException
TEST_F(EnvConfigTest, Ut_EnvRdmaConfigParse_When_PathTooLong_Expect_Throw)
{
    std::string longPath(PATH_MAX, 'a');
    setenv("HCCL_RDMA_QP_PORT_CONFIG_PATH", longPath.c_str(), 1);
    EnvRdmaConfig rdmaConfig;
    EXPECT_THROW(rdmaConfig.Parse(), InvalidParamsException);
    unsetenv("HCCL_RDMA_QP_PORT_CONFIG_PATH");
}

// E7: cfg 文件格式错误、端口越界或 IP 非法 → 抛异常，不发布部分解析结果
TEST_F(EnvConfigTest, Ut_EnvRdmaConfigParse_When_ValidDirButCfgFileMalformed_Expect_Throw)
{
    std::string tmpDir = CreateTempDirForQpPort();
    ASSERT_FALSE(tmpDir.empty());
    setenv("HCCL_RDMA_QP_PORT_CONFIG_PATH", tmpDir.c_str(), 1);
    const std::vector<std::string> invalidContents
        = {"this_is_not_a_valid_config_line\n", "192.168.1.1,192.168.1.2=65536\n",
           "10.10.23.144.200,192.168.1.2=10001\n", "192.168.1.1,192.168.1.2=10001\n192.168.1.3,192.168.1.4=65536\n"};
    for (const auto& content : invalidContents) {
        SCOPED_TRACE(content);
        WriteCfgFile(tmpDir, content);
        EnvRdmaConfig rdmaConfig;
        EXPECT_THROW(rdmaConfig.Parse(), InvalidParamsException);
        EXPECT_FALSE(rdmaConfig.GetMultiQpSrcPortConfig().IsAvailable());
    }

    unsetenv("HCCL_RDMA_QP_PORT_CONFIG_PATH");
    RemoveDirRecursive(tmpDir);
}

// E8: 单行源端口数 > 32 → 抛 InvalidParamsException
TEST_F(EnvConfigTest, Ut_EnvRdmaConfigParse_When_SrcPortCountExceedsMax_Expect_Throw)
{
    std::string tmpDir = CreateTempDirForQpPort();
    ASSERT_FALSE(tmpDir.empty());
    std::string ports = "192.168.1.1,192.168.1.2=10001";
    for (u32 i = 2; i <= 33; i++) {
        ports += "," + std::to_string(10000 + i);
    }
    WriteCfgFile(tmpDir, ports + "\n");

    setenv("HCCL_RDMA_QP_PORT_CONFIG_PATH", tmpDir.c_str(), 1);
    EnvRdmaConfig rdmaConfig;
    EXPECT_THROW(rdmaConfig.Parse(), InvalidParamsException);
    EXPECT_FALSE(rdmaConfig.GetMultiQpSrcPortConfig().IsAvailable());

    unsetenv("HCCL_RDMA_QP_PORT_CONFIG_PATH");
    RemoveDirRecursive(tmpDir);
}

// E9: 单行源端口数恰好 32（边界值）→ 解析成功
TEST_F(EnvConfigTest, Ut_EnvRdmaConfigParse_When_SrcPortCountExactlyMax_Expect_Available)
{
    std::string tmpDir = CreateTempDirForQpPort();
    ASSERT_FALSE(tmpDir.empty());
    std::string ports = "192.168.1.1,192.168.1.2=10001";
    for (u32 i = 2; i <= 32; i++) {
        ports += "," + std::to_string(10000 + i);
    }
    WriteCfgFile(tmpDir, ports + "\n");

    setenv("HCCL_RDMA_QP_PORT_CONFIG_PATH", tmpDir.c_str(), 1);
    EnvRdmaConfig rdmaConfig;
    rdmaConfig.Parse();
    EXPECT_TRUE(rdmaConfig.GetMultiQpSrcPortConfig().IsAvailable());
    EXPECT_EQ(rdmaConfig.GetMultiQpSrcPortConfig().ipPairToPorts.size(), 1u);
    EXPECT_EQ(rdmaConfig.GetMultiQpSrcPortConfig().ipPairToPorts.begin()->second.size(), 32u);

    unsetenv("HCCL_RDMA_QP_PORT_CONFIG_PATH");
    RemoveDirRecursive(tmpDir);
}

// E10: 配置文件行数达到上限 131072 → 截断，第 131073 行不被解析
TEST_F(EnvConfigTest, Ut_EnvRdmaConfigParse_When_LineCountReachesMax_Expect_Truncated)
{
    std::string tmpDir = CreateTempDirForQpPort();
    ASSERT_FALSE(tmpDir.empty());
    std::string content(MultiQpSrcPortConfig::CONFIG_FILE_LINE_MAX - 2, '\n');
    content += "192.168.1.1,192.168.1.2=10001\n";
    content += "192.168.1.3,192.168.1.4=10002\n";
    content += "192.168.1.5,192.168.1.6=10003\n";
    WriteCfgFile(tmpDir, content);

    setenv("HCCL_RDMA_QP_PORT_CONFIG_PATH", tmpDir.c_str(), 1);
    EnvRdmaConfig rdmaConfig;
    rdmaConfig.Parse();
    EXPECT_TRUE(rdmaConfig.GetMultiQpSrcPortConfig().IsAvailable());
    EXPECT_EQ(rdmaConfig.GetMultiQpSrcPortConfig().ipPairToPorts.size(), 2u);
    EXPECT_EQ(
        rdmaConfig.GetMultiQpSrcPortConfig().ipPairToPorts.find("192.168.1.5,192.168.1.6"),
        rdmaConfig.GetMultiQpSrcPortConfig().ipPairToPorts.end());

    unsetenv("HCCL_RDMA_QP_PORT_CONFIG_PATH");
    RemoveDirRecursive(tmpDir);
}
