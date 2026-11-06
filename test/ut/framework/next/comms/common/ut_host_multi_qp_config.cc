/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <streambuf>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "mockcpp/mockcpp.hpp"

#include "host_multi_qp_config.h"

namespace {

constexpr char HOST_RDMA_UDP_PORTS_LIST_ENV[] = "HCCL_HOST_RDMA_UDP_PORTS_LIST";

bool g_configFileAvailable = false;
bool g_configFileReadFailed = false;
std::string g_configFileContent;
std::istringstream g_configFileStream;

class ReadFailedStreamBuf : public std::streambuf {
protected:
    int_type underflow() override { throw std::ios_base::failure("simulated config file read failure"); }
};

ReadFailedStreamBuf g_readFailedStreamBuf;

void StubIfstreamOpen(std::ifstream* inFile, const char* filePath, std::ios_base::openmode openMode)
{
    EXPECT_STREQ(filePath, "/etc/hcomm.cfg");
    EXPECT_EQ(openMode, std::ifstream::in);
    if (!g_configFileAvailable) {
        inFile->setstate(std::ios::failbit);
        return;
    }
    g_configFileStream.clear();
    g_configFileStream.str(g_configFileContent);
    static_cast<std::istream&>(*inFile).rdbuf(g_configFileStream.rdbuf());
    if (g_configFileReadFailed) {
        // 打开时保持流正常，首次读取抛出异常，由输入流设置badbit。
        static_cast<std::istream&>(*inFile).rdbuf(&g_readFailedStreamBuf);
    }
    inFile->clear();
}

class EnvGuard {
public:
    explicit EnvGuard(const char* name) : name_(name)
    {
        const char* value = std::getenv(name);
        if (value != nullptr) {
            savedValue_ = value;
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

    void Set(const std::string& value) const { (void)setenv(name_, value.c_str(), 1); }

    void Unset() const { (void)unsetenv(name_); }

private:
    const char* name_;
    std::string savedValue_;
    bool hadValue_{false};
};

class HostMultiQpConfigTest : public testing::Test {
protected:
    void SetUp() override
    {
        g_configFileAvailable = false;
        g_configFileReadFailed = false;
        g_configFileContent.clear();
        MOCKER_CPP(static_cast<void (std::ifstream::*)(const char*, std::ios_base::openmode)>(&std::ifstream::open))
            .stubs()
            .will(invoke(StubIfstreamOpen));
    }

    void TearDown() override { GlobalMockObject::verify(); }

    void SetConfigFile(const std::string& content)
    {
        g_configFileAvailable = true;
        g_configFileContent = content;
    }
};

TEST_F(HostMultiQpConfigTest, Ut_Parse_When_EnvValidAndFileUnavailable_Expect_EnvConfigApplied)
{
    EnvGuard envGuard(HOST_RDMA_UDP_PORTS_LIST_ENV);
    envGuard.Set("0:10000,10015;1:10016,10031");
    hccl::HostMultiQpConfig config;

    EXPECT_EQ(config.Parse(), HCCL_SUCCESS);
    EXPECT_EQ(config.GetQpCount(0), 0U);
    const auto* device0Ports = config.GetUdpPorts(0);
    ASSERT_NE(device0Ports, nullptr);
    EXPECT_EQ(*device0Ports, (std::vector<uint16_t>{10000, 10015}));
    const auto* device1Ports = config.GetUdpPorts(1);
    ASSERT_NE(device1Ports, nullptr);
    EXPECT_EQ(*device1Ports, (std::vector<uint16_t>{10016, 10031}));
    EXPECT_EQ(config.GetUdpPorts(2), nullptr);
}

TEST_F(HostMultiQpConfigTest, Ut_Parse_When_EnvUnsetAndFileUnavailable_Expect_EmptyConfig)
{
    EnvGuard envGuard(HOST_RDMA_UDP_PORTS_LIST_ENV);
    envGuard.Unset();
    hccl::HostMultiQpConfig config;

    EXPECT_EQ(config.Parse(), HCCL_SUCCESS);
    EXPECT_EQ(config.GetQpCount(0), 0U);
    EXPECT_EQ(config.GetUdpPorts(0), nullptr);
}

TEST_F(HostMultiQpConfigTest, Ut_Parse_When_EnvInvalidAndFileUnavailable_Expect_ParseFailed)
{
    std::string tooManyPorts = "0:1";
    for (uint32_t index = 0; index < 32U; ++index) {
        tooManyPorts += ",1";
    }
    const std::string tooLongValue(32U * 1024U + 1U, '0');
    const std::vector<std::string> invalidValues
        = {"a:10000", "0:0",           "0:65536",         "0:10000,",   "0",         ":10000",
           "0:",      "0:10000:10001", "0:10000;0:10001", tooManyPorts, tooLongValue};
    EnvGuard envGuard(HOST_RDMA_UDP_PORTS_LIST_ENV);

    for (const auto& value : invalidValues) {
        SCOPED_TRACE(value);
        envGuard.Set(value);
        hccl::HostMultiQpConfig config;
        EXPECT_EQ(config.Parse(), HCCL_E_PARA);
        EXPECT_EQ(config.GetQpCount(0), 0U);
        EXPECT_EQ(config.GetUdpPorts(0), nullptr);
    }
}

TEST_F(HostMultiQpConfigTest, Ut_Parse_When_FileConfigValid_Expect_FileOverridesSameDeviceEnvConfig)
{
    EnvGuard envGuard(HOST_RDMA_UDP_PORTS_LIST_ENV);
    envGuard.Set("3:15001,15002;4:16001");
    SetConfigFile("udp_port_mode_3=multi_qp\n"
                  "multi_qp_count_3=4\n"
                  "multi_qp_udp_ports_3=20001,20002\n");
    hccl::HostMultiQpConfig config;

    EXPECT_EQ(config.Parse(), HCCL_SUCCESS);
    EXPECT_EQ(config.GetQpCount(3), 4U);
    const auto* device3Ports = config.GetUdpPorts(3);
    ASSERT_NE(device3Ports, nullptr);
    EXPECT_EQ(*device3Ports, (std::vector<uint16_t>{20001, 20002}));
    EXPECT_EQ(config.GetQpCount(4), 0U);
    const auto* device4Ports = config.GetUdpPorts(4);
    ASSERT_NE(device4Ports, nullptr);
    EXPECT_EQ(*device4Ports, (std::vector<uint16_t>{16001}));
}

TEST_F(HostMultiQpConfigTest, Ut_Parse_When_FilePortsEmpty_Expect_FileOverridesSameDeviceEnvConfig)
{
    EnvGuard envGuard(HOST_RDMA_UDP_PORTS_LIST_ENV);
    envGuard.Set("0:15001,15002;1:16001");
    const std::vector<std::string> emptyValues = {"", " \t\r"};

    for (const auto& value : emptyValues) {
        SCOPED_TRACE(value);
        SetConfigFile(
            "udp_port_mode_0=multi_qp\n"
            "multi_qp_count_0=4\n"
            "multi_qp_udp_ports_0="
            + value + "\n");
        hccl::HostMultiQpConfig config;

        EXPECT_EQ(config.Parse(), HCCL_SUCCESS);
        EXPECT_EQ(config.GetQpCount(0), 4U);
        const auto* device0Ports = config.GetUdpPorts(0);
        ASSERT_NE(device0Ports, nullptr);
        EXPECT_TRUE(device0Ports->empty());
        const auto* device1Ports = config.GetUdpPorts(1);
        ASSERT_NE(device1Ports, nullptr);
        EXPECT_EQ(*device1Ports, (std::vector<uint16_t>{16001}));
    }
}

TEST_F(HostMultiQpConfigTest, Ut_Parse_When_FilePortsNonEmptyAndInvalid_Expect_ParseFailed)
{
    EnvGuard envGuard(HOST_RDMA_UDP_PORTS_LIST_ENV);
    envGuard.Set("0:15001,15002");
    std::string tooManyPorts = "1";
    for (uint32_t index = 0; index < 32U; ++index) {
        tooManyPorts += ",1";
    }
    const std::vector<std::string> invalidValues
        = {"0", "65536", "abc", "-1", "20001,", ",20001", "20001,,20002", "20001, 20002", tooManyPorts};

    for (const auto& value : invalidValues) {
        SCOPED_TRACE(value);
        SetConfigFile(
            "udp_port_mode_0=multi_qp\n"
            "multi_qp_count_0=4\n"
            "multi_qp_udp_ports_0="
            + value + "\n");
        hccl::HostMultiQpConfig config;

        EXPECT_EQ(config.Parse(), HCCL_E_PARA);
        EXPECT_EQ(config.GetQpCount(0), 0U);
        EXPECT_EQ(config.GetUdpPorts(0), nullptr);
    }
}

TEST_F(HostMultiQpConfigTest, Ut_Parse_When_FileFieldsContainWhitespaceAndCrLf_Expect_ConfigApplied)
{
    EnvGuard envGuard(HOST_RDMA_UDP_PORTS_LIST_ENV);
    envGuard.Unset();
    SetConfigFile(" \tudp_port_mode_8 \t= \tmulti_qp \r\n"
                  "\tmulti_qp_count_8 = 4\t\r\n"
                  " multi_qp_udp_ports_8\t= 22001,22002 \r\n");
    hccl::HostMultiQpConfig config;

    EXPECT_EQ(config.Parse(), HCCL_SUCCESS);
    EXPECT_EQ(config.GetQpCount(8), 4U);
    const auto* device8Ports = config.GetUdpPorts(8);
    ASSERT_NE(device8Ports, nullptr);
    EXPECT_EQ(*device8Ports, (std::vector<uint16_t>{22001, 22002}));
}

TEST_F(HostMultiQpConfigTest, Ut_Parse_When_FileContainsDuplicateFields_Expect_FirstValuesKept)
{
    EnvGuard envGuard(HOST_RDMA_UDP_PORTS_LIST_ENV);
    envGuard.Unset();
    SetConfigFile("udp_port_mode_9=multi_qp\n"
                  "udp_port_mode_9=invalid\n"
                  "multi_qp_count_9=3\n"
                  "multi_qp_count_9=5\n"
                  "multi_qp_udp_ports_9=23001\n"
                  "multi_qp_udp_ports_9=24001\n");
    hccl::HostMultiQpConfig config;

    EXPECT_EQ(config.Parse(), HCCL_SUCCESS);
    EXPECT_EQ(config.GetQpCount(9), 3U);
    const auto* device9Ports = config.GetUdpPorts(9);
    ASSERT_NE(device9Ports, nullptr);
    EXPECT_EQ(*device9Ports, (std::vector<uint16_t>{23001}));
}

TEST_F(HostMultiQpConfigTest, Ut_Parse_When_FileModeInvalid_Expect_ParseFailed)
{
    EnvGuard envGuard(HOST_RDMA_UDP_PORTS_LIST_ENV);
    envGuard.Set("3:15001,15002");
    SetConfigFile("udp_port_mode_3=other\n"
                  "multi_qp_count_3=4\n"
                  "multi_qp_udp_ports_3=20001,20002\n");
    hccl::HostMultiQpConfig config;

    EXPECT_EQ(config.Parse(), HCCL_E_PARA);
    EXPECT_EQ(config.GetQpCount(3), 0U);
    EXPECT_EQ(config.GetUdpPorts(3), nullptr);
}

TEST_F(HostMultiQpConfigTest, Ut_Parse_When_FileDeviceConfigInvalid_Expect_ParseFailed)
{
    EnvGuard envGuard(HOST_RDMA_UDP_PORTS_LIST_ENV);
    envGuard.Set("3:15001,15002");
    std::vector<std::string> invalidConfigs
        = {"udp_port_mode_3=multi_qp\n"
           "multi_qp_count_3=4\n",
           "udp_port_mode_3=multi_qp\n"
           "multi_qp_count_3=4\n"
           "udp_port_mode_4=multi_qp\n"
           "multi_qp_count_4=2\n"
           "multi_qp_udp_ports_4=20001\n",
           "udp_port_mode_3=other\n"
           "multi_qp_count_3=4\n"
           "multi_qp_udp_ports_3=\n",
           "udp_port_mode_3=multi_qp\n"
           "multi_qp_count_3=0\n"
           "multi_qp_udp_ports_3=\n",
           "udp_port_mode_3=multi_qp\n"
           "multi_qp_count_3=33\n"
           "multi_qp_udp_ports_3=\n"};
    const std::vector<std::string> prefixes = {"udp_port_mode_", "multi_qp_count_", "multi_qp_udp_ports_"};
    const std::vector<std::string> invalidSuffixes = {"-1", "abc", "", "01", "4294967296"};
    for (const auto& prefix : prefixes) {
        for (const auto& suffix : invalidSuffixes) {
            invalidConfigs.emplace_back(prefix + suffix + "=1\n");
        }
    }

    for (const auto& content : invalidConfigs) {
        SCOPED_TRACE(content);
        SetConfigFile(content);
        hccl::HostMultiQpConfig config;

        EXPECT_EQ(config.Parse(), HCCL_E_PARA);
        EXPECT_EQ(config.GetQpCount(3), 0U);
        EXPECT_EQ(config.GetUdpPorts(3), nullptr);
        EXPECT_EQ(config.GetQpCount(4), 0U);
        EXPECT_EQ(config.GetUdpPorts(4), nullptr);
    }
}

TEST_F(HostMultiQpConfigTest, Ut_Parse_When_FileReadFails_Expect_ParseFailed)
{
    EnvGuard envGuard(HOST_RDMA_UDP_PORTS_LIST_ENV);
    envGuard.Set("5:17001");
    g_configFileAvailable = true;
    g_configFileReadFailed = true;
    hccl::HostMultiQpConfig config;

    EXPECT_EQ(config.Parse(), HCCL_E_PARA);
    EXPECT_EQ(config.GetQpCount(5), 0U);
    EXPECT_EQ(config.GetUdpPorts(5), nullptr);
}

TEST_F(HostMultiQpConfigTest, Ut_Parse_When_CalledRepeatedly_Expect_FirstConfigKept)
{
    EnvGuard envGuard(HOST_RDMA_UDP_PORTS_LIST_ENV);
    envGuard.Set("6:18001");
    SetConfigFile("udp_port_mode_7=multi_qp\n"
                  "multi_qp_count_7=3\n"
                  "multi_qp_udp_ports_7=21001\n");
    hccl::HostMultiQpConfig config;
    ASSERT_EQ(config.Parse(), HCCL_SUCCESS);

    envGuard.Set("6:28001");
    SetConfigFile("udp_port_mode_7=multi_qp\n"
                  "multi_qp_count_7=5\n"
                  "multi_qp_udp_ports_7=31001\n");
    EXPECT_EQ(config.Parse(), HCCL_SUCCESS);
    EXPECT_EQ(config.GetQpCount(7), 3U);
    const auto* device7Ports = config.GetUdpPorts(7);
    ASSERT_NE(device7Ports, nullptr);
    EXPECT_EQ(*device7Ports, (std::vector<uint16_t>{21001}));
    const auto* device6Ports = config.GetUdpPorts(6);
    ASSERT_NE(device6Ports, nullptr);
    EXPECT_EQ(*device6Ports, (std::vector<uint16_t>{18001}));
}

} // namespace
