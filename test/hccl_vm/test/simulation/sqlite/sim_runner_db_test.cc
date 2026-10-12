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
#include <cstring>
#include <gtest/gtest.h>

#include "storage/internal/process_storage_context.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "storage/storage_session.h"
#include "sqlite/simulation_storage_test_helper.h"

namespace {
const std::string kTestDbPath = "/tmp/test_sim_runner_db.db";
const std::string kTestOpDataDbPath = "/tmp/test_sim_runner_db.db.opdata";

void CleanUpDb() { runnerdb_test::CleanUpDatabases(kTestDbPath, kTestOpDataDbPath); }

void SetupTestData()
{
    CleanUpDb();
    (void)runnerdb_test::ResetTestSession(kTestDbPath, kTestOpDataDbPath);

    sim::runtime::Server server{};
    server.pod_id = 100;
    snprintf(server.version, sizeof(server.version), "v1.0");
    runnerdb_test::InsertRecord<sim::runtime::Server>(server);

    sim::runtime::Host host{};
    host.server_id = 1;
    snprintf(host.ip_addr, sizeof(host.ip_addr), "192.168.1.100");
    host.arch = 1;
    runnerdb_test::InsertRecord<sim::runtime::Host>(host);

    sim::runtime::Device device{};
    device.server_id = 1;
    device.logic_id = 0;
    device.physical_id = 0;
    runnerdb_test::InsertRecord<sim::runtime::Device>(device);
}
} // namespace

class SimRunnerDbTest : public testing::Test {
protected:
    void SetUp() override { SetupTestData(); }

    void TearDown() override { CleanUpDb(); }
};

TEST_F(SimRunnerDbTest, GetAllTableName_ReturnsNonEmptyList)
{
    // 表名清单由 Runner 业务目录派生（新增表无需修改本测试或任何清单）。
    std::vector<std::string> tableNames;
    const auto& runnerDatabase = HcclSim::Storage::Ddl::kRunnerDatabase;
    for (std::size_t i = 0; i < runnerDatabase.tableCount; ++i) {
        tableNames.emplace_back(runnerDatabase.tables[i].tableId);
    }

    EXPECT_GT(tableNames.size(), 0);
}

TEST_F(SimRunnerDbTest, GetAllTableName_ContainsExpectedTables)
{
    // 表名清单由 Runner 业务目录派生（新增表无需修改本测试或任何清单）。
    std::vector<std::string> tableNames;
    const auto& runnerDatabase = HcclSim::Storage::Ddl::kRunnerDatabase;
    for (std::size_t i = 0; i < runnerDatabase.tableCount; ++i) {
        tableNames.emplace_back(runnerDatabase.tables[i].tableId);
    }

    bool foundServer = false;
    bool foundHost = false;
    bool foundDevice = false;
    bool foundHcommEndpoint = false;
    bool foundHcommMemReg = false;

    for (const auto& name : tableNames) {
        if (name == "Server") {
            foundServer = true;
        }
        if (name == "Host") {
            foundHost = true;
        }
        if (name == "Device") {
            foundDevice = true;
        }
        if (name == "HcommEndpoint") {
            foundHcommEndpoint = true;
        }
        if (name == "HcommMemReg") {
            foundHcommMemReg = true;
        }
    }

    EXPECT_TRUE(foundServer);
    EXPECT_TRUE(foundHost);
    EXPECT_TRUE(foundDevice);
    EXPECT_TRUE(foundHcommEndpoint);
    EXPECT_TRUE(foundHcommMemReg);
}

TEST_F(SimRunnerDbTest, Add_And_GetById_Server)
{
    sim::runtime::Server server{};
    server.pod_id = 200;
    snprintf(server.version, sizeof(server.version), "v2.0");

    uint64_t id = runnerdb_test::InsertRecord<sim::runtime::Server>(server);

    auto result = runnerdb_test::SelectRecord<sim::runtime::Server>(id);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->pod_id, 200);
    EXPECT_STREQ(result->version, "v2.0");
}

TEST_F(SimRunnerDbTest, Add_And_GetById_Host)
{
    sim::runtime::Host host{};
    host.server_id = 2;
    snprintf(host.ip_addr, sizeof(host.ip_addr), "192.168.2.1");
    host.arch = 2;

    uint64_t id = runnerdb_test::InsertRecord<sim::runtime::Host>(host);

    auto result = runnerdb_test::SelectRecord<sim::runtime::Host>(id);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->server_id, 2);
    EXPECT_STREQ(result->ip_addr, "192.168.2.1");
    EXPECT_EQ(result->arch, 2);
}

TEST_F(SimRunnerDbTest, Find_ReturnsMatchingRecords)
{
    auto results = runnerdb_test::SelectRecords<sim::runtime::Device>(
        HcclSim::Storage::Eq(&sim::runtime::Device::server_id, uint64_t{1}));

    EXPECT_GE(results.size(), 1);
}

TEST_F(SimRunnerDbTest, Find_ReturnsEmptyForNoMatch)
{
    auto results = runnerdb_test::SelectRecords<sim::runtime::Device>(
        HcclSim::Storage::Eq(&sim::runtime::Device::server_id, uint64_t{999}));

    EXPECT_EQ(results.size(), 0);
}

TEST_F(SimRunnerDbTest, FindOne_ReturnsFirstMatch)
{
    auto result = runnerdb_test::SelectFirstRecord<sim::runtime::Device>(
        HcclSim::Storage::Eq(&sim::runtime::Device::server_id, uint64_t{1}));

    EXPECT_TRUE(result.second);
    EXPECT_EQ(result.first.server_id, 1);
}

TEST_F(SimRunnerDbTest, FindOne_ReturnsFalseForNoMatch)
{
    auto result = runnerdb_test::SelectFirstRecord<sim::runtime::Device>(
        HcclSim::Storage::Eq(&sim::runtime::Device::server_id, uint64_t{999}));

    EXPECT_FALSE(result.second);
}

TEST_F(SimRunnerDbTest, LegacyFindAndListDefaultToPrimaryKeyAscendingAndPreserveExplicitDesc)
{
    ASSERT_TRUE(runnerdb_test::ClearRecords<sim::runtime::Device>());

    sim::runtime::Device first{};
    first.server_id = 77;
    first.logic_id = 10;
    const uint64_t firstId = runnerdb_test::InsertRecord<sim::runtime::Device>(first);

    sim::runtime::Device second{};
    second.server_id = 77;
    second.logic_id = 20;
    const uint64_t secondId = runnerdb_test::InsertRecord<sim::runtime::Device>(second);

    sim::runtime::Device third{};
    third.server_id = 77;
    third.logic_id = 30;
    const uint64_t thirdId = runnerdb_test::InsertRecord<sim::runtime::Device>(third);

    // server_id is a non-primary indexed field. With no explicit order, legacy Find/ListAll use id ASC.
    const auto byServer = HcclSim::Storage::Eq(&sim::runtime::Device::server_id, uint64_t{77});
    const auto defaultMatches = runnerdb_test::SelectRecords<sim::runtime::Device>(
        byServer, HcclSim::Storage::OrderBy(&sim::runtime::Device::id, HcclSim::Storage::SortDirection::ASC));
    ASSERT_EQ(defaultMatches.size(), 3U);
    EXPECT_EQ(defaultMatches[0].id, firstId);
    EXPECT_EQ(defaultMatches[1].id, secondId);
    EXPECT_EQ(defaultMatches[2].id, thirdId);

    const auto listAll = runnerdb_test::SelectAllRecords<sim::runtime::Device>();
    ASSERT_EQ(listAll.size(), 3U);
    EXPECT_EQ(listAll[0].id, firstId);
    EXPECT_EQ(listAll[1].id, secondId);
    EXPECT_EQ(listAll[2].id, thirdId);

    const auto defaultFirst = runnerdb_test::SelectFirstRecord<sim::runtime::Device>(
        byServer, HcclSim::Storage::OrderBy(&sim::runtime::Device::id, HcclSim::Storage::SortDirection::ASC));
    ASSERT_TRUE(defaultFirst.second);
    EXPECT_EQ(defaultFirst.first.id, firstId);

    const auto descendingMatches = runnerdb_test::SelectRecords<sim::runtime::Device>(
        byServer, HcclSim::Storage::OrderBy(&sim::runtime::Device::logic_id, HcclSim::Storage::SortDirection::DESC));
    ASSERT_EQ(descendingMatches.size(), 3U);
    EXPECT_EQ(descendingMatches[0].logic_id, 30U);
    EXPECT_EQ(descendingMatches[1].logic_id, 20U);
    EXPECT_EQ(descendingMatches[2].logic_id, 10U);

    const auto descendingFirst = runnerdb_test::SelectFirstRecord<sim::runtime::Device>(
        byServer, HcclSim::Storage::OrderBy(&sim::runtime::Device::logic_id, HcclSim::Storage::SortDirection::DESC));
    ASSERT_TRUE(descendingFirst.second);
    EXPECT_EQ(descendingFirst.first.id, thirdId);
}

TEST_F(SimRunnerDbTest, Update_ModifiesRecord)
{
    auto result = runnerdb_test::SelectFirstRecord<sim::runtime::Device>(
        HcclSim::Storage::Eq(&sim::runtime::Device::physical_id, uint64_t{0}));
    ASSERT_TRUE(result.second);

    uint64_t id = result.first.id;
    runnerdb_test::UpdateRecord<sim::runtime::Device>(
        id, HcclSim::Storage::Set(&sim::runtime::Device::logic_id, uint64_t{999}));

    auto updated = runnerdb_test::SelectRecord<sim::runtime::Device>(id);
    ASSERT_TRUE(updated.has_value());
    EXPECT_EQ(updated->logic_id, 999);
}

TEST_F(SimRunnerDbTest, Update_ReturnsFalseForInvalidId)
{
    bool ret = runnerdb_test::UpdateRecord<sim::runtime::Device>(
        99999, HcclSim::Storage::Set(&sim::runtime::Device::logic_id, uint64_t{999}));

    EXPECT_FALSE(ret);
}

TEST_F(SimRunnerDbTest, Delete_RemovesRecord)
{
    sim::runtime::Device device{};
    device.server_id = 1;
    device.logic_id = 100;
    device.physical_id = 100;
    uint64_t id = runnerdb_test::InsertRecord<sim::runtime::Device>(device);

    bool ret = runnerdb_test::DeleteRecord<sim::runtime::Device>(id);
    EXPECT_TRUE(ret);

    auto result = runnerdb_test::SelectRecord<sim::runtime::Device>(id);
    EXPECT_FALSE(result.has_value());
}

TEST_F(SimRunnerDbTest, Delete_ReturnsFalseForInvalidId)
{
    bool ret = runnerdb_test::DeleteRecord<sim::runtime::Device>(99999);

    EXPECT_FALSE(ret);
}

TEST_F(SimRunnerDbTest, DeleteAll_RemovesAllRecords)
{
    sim::runtime::Server server1{};
    server1.pod_id = 1;
    snprintf(server1.version, sizeof(server1.version), "v1");
    runnerdb_test::InsertRecord<sim::runtime::Server>(server1);

    sim::runtime::Server server2{};
    server2.pod_id = 2;
    snprintf(server2.version, sizeof(server2.version), "v2");
    runnerdb_test::InsertRecord<sim::runtime::Server>(server2);

    sim::runtime::Server server3{};
    server3.pod_id = 3;
    snprintf(server3.version, sizeof(server3.version), "v3");
    runnerdb_test::InsertRecord<sim::runtime::Server>(server3);

    bool ret = runnerdb_test::ClearRecords<sim::runtime::Server>();
    EXPECT_TRUE(ret);

    auto results = runnerdb_test::SelectAllRecords<sim::runtime::Server>();
    EXPECT_EQ(results.size(), 0);
}

TEST_F(SimRunnerDbTest, MultipleAdd_MaintainsDataIntegrity)
{
    for (int i = 0; i < 10; i++) {
        sim::runtime::Device device{};
        device.server_id = 1;
        device.logic_id = static_cast<uint32_t>(i);
        device.physical_id = static_cast<uint32_t>(i + 100);
        runnerdb_test::InsertRecord<sim::runtime::Device>(device);
    }

    auto results = runnerdb_test::SelectAllRecords<sim::runtime::Device>();

    EXPECT_GE(results.size(), 11);
}

class SimRunnerDbContextTest : public testing::Test {
protected:
    void SetUp() override
    {
        SetupTestData();

        sim::runtime::Device device{};
        auto ret = runnerdb_test::SelectFirstRecord<sim::runtime::Device>(
            HcclSim::Storage::Eq(&sim::runtime::Device::physical_id, uint64_t{0}));
        ASSERT_TRUE(ret.second);

        sim::runtime::Context ctx{};
        ctx.device_id = ret.first.id;
        ctx.is_default = 1;
        ctx.run_id = 1;
        runnerdb_test::InsertRecord<sim::runtime::Context>(ctx);
    }

    void TearDown() override { CleanUpDb(); }
};

TEST_F(SimRunnerDbContextTest, GetById_Context)
{
    auto results = runnerdb_test::SelectAllRecords<sim::runtime::Context>();

    EXPECT_GE(results.size(), 1);
    EXPECT_EQ(results[0].is_default, 1);
}

TEST_F(SimRunnerDbContextTest, GetDefaultContext)
{
    auto ctx = runnerdb_test::SelectFirstRecord<sim::runtime::Context>(
        HcclSim::Storage::Eq(&sim::runtime::Context::is_default, uint64_t{1}));

    EXPECT_TRUE(ctx.second);
    EXPECT_EQ(ctx.first.is_default, 1);
}

class SimRunnerDbStreamTest : public testing::Test {
protected:
    void SetUp() override { SetupTestData(); }

    void TearDown() override { CleanUpDb(); }
};

TEST_F(SimRunnerDbStreamTest, Add_Stream)
{
    sim::runtime::Stream stream{};
    stream.ctx_id = 1;
    stream.priority = 1;
    stream.schedule_strategy = 1;

    uint64_t id = runnerdb_test::InsertRecord<sim::runtime::Stream>(stream);

    auto result = runnerdb_test::SelectRecord<sim::runtime::Stream>(id);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->ctx_id, 1);
    EXPECT_EQ(result->priority, 1);
}

TEST_F(SimRunnerDbStreamTest, Update_Stream)
{
    sim::runtime::Stream stream{};
    stream.ctx_id = 1;
    uint64_t id = runnerdb_test::InsertRecord<sim::runtime::Stream>(stream);

    runnerdb_test::UpdateRecord<sim::runtime::Stream>(
        id, HcclSim::Storage::Set(&sim::runtime::Stream::priority, uint64_t{5}));

    auto result = runnerdb_test::SelectRecord<sim::runtime::Stream>(id);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->priority, 5);
}

TEST_F(SimRunnerDbStreamTest, GetStreamsByCtxId)
{
    sim::runtime::Stream stream1{};
    stream1.ctx_id = 10;
    runnerdb_test::InsertRecord<sim::runtime::Stream>(stream1);

    sim::runtime::Stream stream2{};
    stream2.ctx_id = 10;
    runnerdb_test::InsertRecord<sim::runtime::Stream>(stream2);

    auto results = runnerdb_test::SelectRecords<sim::runtime::Stream>(
        HcclSim::Storage::Eq(&sim::runtime::Stream::ctx_id, uint64_t{10}));

    EXPECT_EQ(results.size(), 2);
}
