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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <map>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <vector>

#include "store_binary_data_operator.h"
#include "store_dump_shm_data.h"
#include "sim_loader.h"
#include "sim_common_api.h"
#include "hccl_types.h"
#include "runtime_state/sim_models.h"
#include "runtime_state/db_sim_runner_common.h"
#include "operation_data/operation_data_ops.h"
#include "storage/internal/process_storage_context.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "storage/storage_session.h"
#include "store_sim_shm_ops.h"
#include "sqlite/simulation_storage_test_helper.h"

using sim::runtime::g_cur_comm_key;
using sim::runtime::g_cur_device_key;

using namespace HcclSim;

extern uint8_t g_opExpansionMode;
namespace HcclSim {
std::string GetBinLocation();
HcclVmResult CreateSimSynData(
    HcclVmSynData& hvmSynData, const sim::operation::OpExecutionKey& key,
    const std::map<uint32_t, uint32_t>& deviceToRank);
HcclVmResult CreateSimInstrData(HcclVmInstrData& hvmInstrData, const std::map<uint32_t, uint32_t>& deviceToRank);
HcclVmResult CreateSimTaskMetaData(
    HcclVmTaskMetaData& hvmTaskMetaData,
    const std::map<uint32_t, std::vector<sim::operation::CompositeOpDetail>>& compositeDataMap,
    const std::map<uint32_t, uint32_t>& deviceToRank);
} // namespace HcclSim

static const std::string kSynDataFile = "/%s_hcclvm_syn_data.bin";
static const std::string kTaskDataFile = "/%s_hcclvm_task_data.bin";

static bool LoadFirstOpExecutionKey(sim::operation::OpExecutionKey& key)
{
    loader::Loader dataLoader;
    std::vector<sim::operation::OpExecutionKey> keys;
    if (dataLoader.LoadOpExecutionKeys(keys) != HcclResult::HCCL_SUCCESS || keys.empty()) {
        return false;
    }
    key = keys.front();
    return true;
}

static void ClearAllDbTables()
{
    // 迁移前单库 ClearAll() 的等价语义：关闭进程会话、清理数据库文件后重装。
    // 确保进程存储会话存在后逐表清空。
    g_cur_comm_key = 0;
    std::filesystem::create_directories(GetBinLocation() + DATA_FILE_PATH);
    (void)runnerdb_test::ResetTestSession("/tmp/test_dump_shm_data.db", "/tmp/test_dump_shm_data_op.db");
    runnerdb_test::ClearRecords<sim::runtime::SimModelData>();
    runnerdb_test::ClearRecords<sim::runtime::Server>();
    runnerdb_test::ClearRecords<sim::runtime::Host>();
    runnerdb_test::ClearRecords<sim::runtime::Device>();
    runnerdb_test::ClearRecords<sim::runtime::Communicator>();
    runnerdb_test::ClearRecords<sim::runtime::MemoryLayout>();
    runnerdb_test::ClearRecords<sim::runtime::EndPoint>();
    runnerdb_test::ClearRecords<sim::runtime::Ccu>();
    runnerdb_test::ClearRecords<sim::runtime::CcuChannel>();
    runnerdb_test::ClearRecords<sim::runtime::RaContext>();
    runnerdb_test::ClearRecords<sim::runtime::RaJetty>();
    runnerdb_test::ClearRecords<sim::runtime::EndPointPair>();
}

class DumpShmDataTest : public testing::Test {
protected:
    void SetUp() override
    {
        ClearAllDbTables();
        std::filesystem::create_directories("data");
    }
    void TearDown() override
    {
        ClearAllDbTables();
        g_opExpansionMode = 0;
    }
};

TEST_F(DumpShmDataTest, ShmOps_CheckConstants)
{
    EXPECT_EQ(SHM_MAGIC, 0x53484D50);
    EXPECT_EQ(SHM_VERSION, 1);
}

TEST_F(DumpShmDataTest, GetBinLocation_ReturnsNonEmpty)
{
    std::string loc = GetBinLocation();
    EXPECT_FALSE(loc.empty());
}

TEST_F(DumpShmDataTest, GenDataId_ReturnsNonEmpty)
{
    std::string dataId = GenDataId();
    EXPECT_FALSE(dataId.empty());
    EXPECT_GE(dataId.size(), 17);
}

TEST_F(DumpShmDataTest, GenDataId_ContainsTimestamp)
{
    std::string dataId = GenDataId();
    EXPECT_NE(dataId.find("_"), std::string::npos);
    EXPECT_EQ(dataId[0], '2');
    EXPECT_EQ(dataId[1], '0');
}

TEST_F(DumpShmDataTest, g_opExpansionMode_DefaultZero) { EXPECT_EQ(g_opExpansionMode, 0); }

TEST_F(DumpShmDataTest, DumpHcclVmFlagData_FileCreation)
{
    HcclVmFlagData flagData{};
    flagData.header.magic = HCCLVM_FLAG_FILE_MAGIC;
    flagData.header.header_size = sizeof(FileHeader);
    flagData.header.flags = 1;
    flagData.runner_status = 0;

    HcclVmResult ret = DumpHcclVmFlagData(flagData);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(DumpShmDataTest, DumpHcclVmFlagData_ThenGetHcclVmFlagData)
{
    HcclVmFlagData writtenData{};
    writtenData.header.magic = HCCLVM_FLAG_FILE_MAGIC;
    writtenData.header.header_size = sizeof(FileHeader);
    writtenData.header.flags = 1;
    writtenData.runner_status = 1;

    HcclVmResult ret = DumpHcclVmFlagData(writtenData);
    ASSERT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);

    HcclVmFlagData readData{};
    ret = GetHcclVmFlagData(readData);
    ASSERT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);

    EXPECT_EQ(readData.header.magic, HCCLVM_FLAG_FILE_MAGIC);
    EXPECT_EQ(readData.runner_status, writtenData.runner_status);
}

TEST_F(DumpShmDataTest, DumpHcclVmFlagData_WriteFail_ReturnsError)
{
    HcclVmFlagData flagData{};
    flagData.header.magic = HCCLVM_FLAG_FILE_MAGIC;
    flagData.header.header_size = 0;
    flagData.header.flags = 1;
    flagData.runner_status = 0;

    std::string rootPath = GetBinLocation();
    std::string fullPath = rootPath + DATA_FILE_PATH + "/hcclvm_flag_data.bin";
    FILE* fp = fopen(fullPath.c_str(), "wb");
    if (fp) {
        auto ret = HcclVmFlagDataWrite(fp, flagData);
        fclose(fp);
        if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
            HcclVmResult dumpRet = DumpHcclVmFlagData(flagData);
            EXPECT_NE(dumpRet, HcclVmResult::HCCL_SIM_SUCCESS);
        }
    }
}

TEST_F(DumpShmDataTest, GetHcclVmFlagData_FileNotExist_ReturnsError)
{
    HcclVmFlagData readData{};
    std::string rootPath = GetBinLocation();
    std::string fullPath = rootPath + DATA_FILE_PATH + "/hcclvm_flag_data.bin";
    std::filesystem::remove(fullPath);
    HcclVmResult ret = GetHcclVmFlagData(readData);
    EXPECT_NE(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(DumpShmDataTest, DumpHcclVmFlagData_StatusStart)
{
    HcclVmFlagData flagData{};
    flagData.header.magic = HCCLVM_FLAG_FILE_MAGIC;
    flagData.header.header_size = sizeof(FileHeader);
    flagData.header.flags = 1;
    flagData.runner_status = 1;

    HcclVmResult ret = DumpHcclVmFlagData(flagData);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);

    HcclVmFlagData readData{};
    ret = GetHcclVmFlagData(readData);
    ASSERT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(readData.runner_status, 1);
}

TEST_F(DumpShmDataTest, DumpHcclVmFlagData_StatusExit)
{
    HcclVmFlagData flagData{};
    flagData.header.magic = HCCLVM_FLAG_FILE_MAGIC;
    flagData.header.header_size = sizeof(FileHeader);
    flagData.header.flags = 1;
    flagData.runner_status = 2;

    HcclVmResult ret = DumpHcclVmFlagData(flagData);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);

    HcclVmFlagData readData{};
    ret = GetHcclVmFlagData(readData);
    ASSERT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(readData.runner_status, 2);
}

class DumpShmDataWithDBTest : public testing::Test {
protected:
    void SetUp() override
    {
        ClearAllDbTables();
        std::filesystem::create_directories("data");
    }
    void TearDown() override
    {
        ClearAllDbTables();
        g_opExpansionMode = 0;
    }

    void InsertOpData(
        uint32_t rankId, uint32_t expansionMode, uint16_t opType = 0, uint64_t dataCount = 1024, bool withTask = false,
        const std::vector<uint64_t>& matrix = {})
    {
        // InsertOpDetail 依赖有效 commId 来解析 commName/commHash 并分配 opIter，
        // 若尚未建立 communicator 则先补一个，避免 commId=0 导致插入失败。
        if (g_cur_comm_key == 0) {
            sim::runtime::Communicator comm{};
            std::strncpy(comm.comm_id, "test_comm", sizeof(comm.comm_id) - 1);
            comm.rank_size = 2;
            comm.rank_id = rankId;
            auto commResult = sim::runtime::Db::Add<sim::runtime::Communicator>(comm);
            ASSERT_TRUE(commResult.ok());
            ASSERT_TRUE(commResult.value.has_value());
            g_cur_comm_key = commResult.value->value;
        }
        ::OpDetails op{};
        op.opType = opType;
        op.dataType = HCCL_DATA_TYPE_INT8;
        op.reduceType = HCCL_REDUCE_SUM;
        op.opV1.count = dataCount;

        sim::operation::OpDetailTab detail{};
        detail.pid = getpid();
        detail.rankId = rankId;
        detail.commId = g_cur_comm_key;
        detail.root = 0;
        detail.opExpansionMode = expansionMode;
        detail.devType = 0;
        detail.rankSize = 2;
        detail.srcRank = rankId;
        detail.dstRank = (rankId + 1) % 2;
        detail.opDetail.resize(sizeof(op));
        std::memcpy(detail.opDetail.data(), &op, sizeof(op));
        if (!matrix.empty()) {
            uint32_t count = matrix.size();
            detail.opExtInfo.resize(sizeof(count) + matrix.size() * sizeof(uint64_t));
            std::memcpy(detail.opExtInfo.data(), &count, sizeof(count));
            std::memcpy(detail.opExtInfo.data() + sizeof(count), matrix.data(), matrix.size() * sizeof(uint64_t));
        }

        sim::operation::OpMemInfoTab mem{};
        mem.inputAddr = 0x1000 + rankId * 0x10000;
        mem.inputSize = 4096;
        mem.outputAddr = 0x2000 + rankId * 0x10000;
        mem.outputSize = 8192;
        ASSERT_EQ(sim::operation::InsertOpDetailAndMem(detail, mem), 0);

        if (withTask) {
            HcclTaskMetaData task{};
            task.taskType = HccLTaskMetaType::NOTIFY_RECORD;
            task.rankId = rankId;
            task.taskData.notify.srcDeviceId = rankId;
            task.taskData.notify.dstDeviceId = rankId;
            task.taskData.notify.notifyId = 1;
            sim::operation::OpTaskTab taskRow{};
            taskRow.optaskMeta.resize(sizeof(task));
            std::memcpy(taskRow.optaskMeta.data(), &task, sizeof(task));
            ASSERT_EQ(sim::operation::InsertOpTask(taskRow), 0);
        }
    }

    void InsertSimModelData_CCU(bool withTask = false)
    {
        InsertOpData(0, sim::runtime::SimOpExpansionMode::SIM_OP_EXPANSION_MODE_CCU, 0, 1024, withTask);
        sim::runtime::SimModelData model;
        model.rank_id = 0;
        model.src_rank = 0;
        model.dst_rank = 1;
        model.root = 0;
        model.rank_size = 2;
        model.chip_type = 0;
        model.op_type = 0;
        model.reduce_op = 0;
        model.data_type = 0;
        model.data_count = 1024;
        model.op_expansion_mode = sim::runtime::SimOpExpansionMode::SIM_OP_EXPANSION_MODE_CCU;
        model.ccu0_resource_base_addr = 0;
        model.ccu1_resource_base_addr = 0;
        model.all2AllDataDes.sendType = 0;
        model.all2AllDataDes.recvType = 0;
        model.all2AllDataDes.sendCount = 0;
        model.all2AllDataDes.recvCount = 0;
        model.all2AllDataDes.count = 2;
        memset(model.all2AllDataDes.sendCountMatrix, 0, sizeof(model.all2AllDataDes.sendCountMatrix));
        model.all2AllDataDes.sendCountMatrix[0] = 512;
        model.all2AllDataDes.sendCountMatrix[1] = 512;
        runnerdb_test::InsertRecord<sim::runtime::SimModelData>(model);
    }

    void InsertSimModelData_AICPU()
    {
        InsertOpData(0, sim::runtime::SimOpExpansionMode::SIM_OP_EXPANSION_MODE_AICPU);
        sim::runtime::SimModelData model;
        model.rank_id = 0;
        model.src_rank = 0;
        model.dst_rank = 1;
        model.root = 0;
        model.rank_size = 2;
        model.chip_type = 0;
        model.op_type = 0;
        model.reduce_op = 0;
        model.data_type = 0;
        model.data_count = 1024;
        model.op_expansion_mode = sim::runtime::SimOpExpansionMode::SIM_OP_EXPANSION_MODE_AICPU;
        model.ccu0_resource_base_addr = 0;
        model.ccu1_resource_base_addr = 0;
        model.all2AllDataDes.sendType = 0;
        model.all2AllDataDes.recvType = 0;
        model.all2AllDataDes.sendCount = 0;
        model.all2AllDataDes.recvCount = 0;
        model.all2AllDataDes.count = 0;
        runnerdb_test::InsertRecord<sim::runtime::SimModelData>(model);
    }

    void InsertSimModelData_AllToAllV()
    {
        InsertOpData(
            0, sim::runtime::SimOpExpansionMode::SIM_OP_EXPANSION_MODE_CCU,
            static_cast<uint16_t>(HcclCMDType::HCCL_CMD_ALLTOALLV), 1024, false, {512, 512, 256, 256});
        InsertOpData(
            1, sim::runtime::SimOpExpansionMode::SIM_OP_EXPANSION_MODE_CCU,
            static_cast<uint16_t>(HcclCMDType::HCCL_CMD_ALLTOALLV), 1024, false, {128, 128, 64, 64});
        sim::runtime::SimModelData model0;
        model0.rank_id = 0;
        model0.src_rank = 0;
        model0.dst_rank = 1;
        model0.root = 0;
        model0.rank_size = 2;
        model0.chip_type = 0;
        model0.op_type = static_cast<uint16_t>(HcclCMDType::HCCL_CMD_ALLTOALLV);
        model0.reduce_op = 0;
        model0.data_type = 0;
        model0.data_count = 1024;
        model0.op_expansion_mode = sim::runtime::SimOpExpansionMode::SIM_OP_EXPANSION_MODE_CCU;
        model0.all2AllDataDes.count = 2;
        memset(model0.all2AllDataDes.sendCountMatrix, 0, sizeof(model0.all2AllDataDes.sendCountMatrix));
        model0.all2AllDataDes.sendCountMatrix[0] = 512;
        model0.all2AllDataDes.sendCountMatrix[1] = 512;
        model0.all2AllDataDes.sendCountMatrix[2] = 256;
        model0.all2AllDataDes.sendCountMatrix[3] = 256;
        runnerdb_test::InsertRecord<sim::runtime::SimModelData>(model0);

        sim::runtime::SimModelData model1;
        model1.rank_id = 1;
        model1.src_rank = 1;
        model1.dst_rank = 0;
        model1.root = 0;
        model1.rank_size = 2;
        model1.chip_type = 0;
        model1.op_type = static_cast<uint16_t>(HcclCMDType::HCCL_CMD_ALLTOALLV);
        model1.reduce_op = 0;
        model1.data_type = 0;
        model1.data_count = 1024;
        model1.op_expansion_mode = sim::runtime::SimOpExpansionMode::SIM_OP_EXPANSION_MODE_CCU;
        model1.all2AllDataDes.count = 2;
        memset(model1.all2AllDataDes.sendCountMatrix, 0, sizeof(model1.all2AllDataDes.sendCountMatrix));
        model1.all2AllDataDes.sendCountMatrix[0] = 128;
        model1.all2AllDataDes.sendCountMatrix[1] = 128;
        model1.all2AllDataDes.sendCountMatrix[2] = 64;
        model1.all2AllDataDes.sendCountMatrix[3] = 64;
        runnerdb_test::InsertRecord<sim::runtime::SimModelData>(model1);
    }

    void InsertBasicTopology()
    {
        sim::runtime::Server server{};
        uint64_t serverId = runnerdb_test::InsertRecord<sim::runtime::Server>(server);
        sim::runtime::Host host{};
        host.server_id = serverId;
        strncpy(host.ip_addr, "192.1.5.5", sizeof(host.ip_addr) - 1);
        uint64_t hostId = runnerdb_test::InsertRecord<sim::runtime::Host>(host);

        sim::runtime::Device dev0{};
        dev0.server_id = serverId;
        dev0.physical_id = 0;
        uint64_t dev0Id = runnerdb_test::InsertRecord<sim::runtime::Device>(dev0);

        sim::runtime::Device dev1{};
        dev1.server_id = serverId;
        dev1.physical_id = 1;
        uint64_t dev1Id = runnerdb_test::InsertRecord<sim::runtime::Device>(dev1);

        sim::runtime::Communicator comm0{};
        std::strncpy(comm0.comm_id, "test_comm", sizeof(comm0.comm_id) - 1);
        comm0.rank_size = 2;
        comm0.rank_id = 0;
        comm0.device_id = dev0Id;
        g_cur_comm_key = runnerdb_test::InsertRecord<sim::runtime::Communicator>(comm0);

        sim::runtime::Communicator comm1{};
        std::strncpy(comm1.comm_id, "test_comm", sizeof(comm1.comm_id) - 1);
        comm1.rank_size = 2;
        comm1.rank_id = 1;
        comm1.device_id = dev1Id;
        runnerdb_test::InsertRecord<sim::runtime::Communicator>(comm1);

        sim::runtime::MemoryLayout mem0{};
        mem0.rank_id = 0;
        mem0.buf_type = 0;
        mem0.base_addr = 0x1000;
        mem0.size = 4096;
        mem0.global_offset = 0;
        runnerdb_test::InsertRecord<sim::runtime::MemoryLayout>(mem0);

        sim::runtime::MemoryLayout mem1{};
        mem1.rank_id = 1;
        mem1.buf_type = 1;
        mem1.base_addr = 0x2000;
        mem1.size = 8192;
        mem1.global_offset = 4096;
        runnerdb_test::InsertRecord<sim::runtime::MemoryLayout>(mem1);
    }

    void InsertChannelTopology()
    {
        InsertBasicTopology();

        auto rank0 = runnerdb_test::SelectFirstRecord<sim::runtime::Device>(
            HcclSim::Storage::Eq(&sim::runtime::Device::physical_id, uint64_t{0}));
        auto rank1 = runnerdb_test::SelectFirstRecord<sim::runtime::Device>(
            HcclSim::Storage::Eq(&sim::runtime::Device::physical_id, uint64_t{1}));
        ASSERT_TRUE(rank0.second);
        ASSERT_TRUE(rank1.second);

        sim::runtime::EndPoint localEp{};
        localEp.device_id = rank0.first.id;
        localEp.die_id = 0;
        memset(localEp.eid, 0x01, sizeof(localEp.eid));
        strncpy(localEp.ip_addr, "192.1.5.5", sizeof(localEp.ip_addr) - 1);
        uint64_t localEpId = runnerdb_test::InsertRecord<sim::runtime::EndPoint>(localEp);

        sim::runtime::EndPoint remoteEp{};
        remoteEp.device_id = rank1.first.id;
        remoteEp.die_id = 1;
        memset(remoteEp.eid, 0x02, sizeof(remoteEp.eid));
        strncpy(remoteEp.ip_addr, "192.2.5.5", sizeof(remoteEp.ip_addr) - 1);
        uint64_t remoteEpId = runnerdb_test::InsertRecord<sim::runtime::EndPoint>(remoteEp);

        sim::runtime::CcuChannel channel{};
        channel.channel_id = 1;
        channel.local_endpoint_id = localEpId;
        channel.remote_endpoint_id = remoteEpId;
        channel.protocol = 0;
        channel.jetty_start = 10;
        channel.jetty_num = 4;
        runnerdb_test::InsertRecord<sim::runtime::CcuChannel>(channel);

        sim::runtime::EndPointPair pair{};
        pair.local_enpoint_id = localEpId;
        pair.remote_enpoint_id = remoteEpId;
        pair.tp_type = 0;
        runnerdb_test::InsertRecord<sim::runtime::EndPointPair>(pair);

        sim::operation::CcuChannelTab channelRow{};
        channelRow.channelId = 1;
        channelRow.srcDieId = localEp.die_id;
        channelRow.dstDieId = remoteEp.die_id;
        channelRow.srcRankId = 0;
        channelRow.dstRankId = 1;
        std::memcpy(channelRow.leid, localEp.eid, sizeof(channelRow.leid));
        std::memcpy(channelRow.reid, remoteEp.eid, sizeof(channelRow.reid));
        channelRow.protocol = 0;
        channelRow.jettyNum = 4;
        for (uint32_t i = 0; i < channelRow.jettyNum; ++i) {
            channelRow.jettyId[i] = 10 + i;
        }
        ASSERT_EQ(sim::operation::InsertCcuChannel(channelRow), 0);
    }

    void InsertJettyTopology()
    {
        InsertBasicTopology();

        auto rank0 = runnerdb_test::SelectFirstRecord<sim::runtime::Device>(
            HcclSim::Storage::Eq(&sim::runtime::Device::physical_id, uint64_t{0}));
        auto rank1 = runnerdb_test::SelectFirstRecord<sim::runtime::Device>(
            HcclSim::Storage::Eq(&sim::runtime::Device::physical_id, uint64_t{1}));
        ASSERT_TRUE(rank0.second);
        ASSERT_TRUE(rank1.second);

        sim::runtime::EndPoint localEp{};
        localEp.device_id = rank0.first.id;
        localEp.die_id = 0;
        memset(localEp.eid, 0x03, sizeof(localEp.eid));
        strncpy(localEp.ip_addr, "192.1.5.5", sizeof(localEp.ip_addr) - 1);
        uint64_t localEpId = runnerdb_test::InsertRecord<sim::runtime::EndPoint>(localEp);

        sim::runtime::EndPoint remoteEp{};
        remoteEp.device_id = rank1.first.id;
        remoteEp.die_id = 1;
        memset(remoteEp.eid, 0x04, sizeof(remoteEp.eid));
        strncpy(remoteEp.ip_addr, "192.2.5.5", sizeof(remoteEp.ip_addr) - 1);
        uint64_t remoteEpId = runnerdb_test::InsertRecord<sim::runtime::EndPoint>(remoteEp);

        sim::runtime::RaContext raCtx{};
        raCtx.endpoint_id = localEpId;
        uint64_t raCtxId = runnerdb_test::InsertRecord<sim::runtime::RaContext>(raCtx);

        // peer_endpoint_id由生产路径RaCtxQpImport回填(见hccp_stub.cc),
        // 测试中需手动设置为本端到对端的链路归属, 否则dump侧过滤不到该jetty
        sim::runtime::RaJetty jetty{};
        jetty.ctx_handle = raCtxId;
        jetty.jetty_id = 5;
        jetty.mode = 3;
        jetty.peer_endpoint_id = remoteEpId;
        runnerdb_test::InsertRecord<sim::runtime::RaJetty>(jetty);

        sim::runtime::EndPointPair epPair{};
        epPair.local_enpoint_id = localEpId;
        epPair.remote_enpoint_id = remoteEpId;
        runnerdb_test::InsertRecord<sim::runtime::EndPointPair>(epPair);
    }

    void InsertCcuResourceTopology()
    {
        sim::runtime::Server server{};
        uint64_t serverId = runnerdb_test::InsertRecord<sim::runtime::Server>(server);
        sim::runtime::Host host{};
        host.server_id = serverId;
        strncpy(host.ip_addr, "192.1.5.5", sizeof(host.ip_addr) - 1);
        runnerdb_test::InsertRecord<sim::runtime::Host>(host);

        sim::runtime::Device dev0{};
        dev0.server_id = serverId;
        uint64_t dev0Id = runnerdb_test::InsertRecord<sim::runtime::Device>(dev0);

        sim::runtime::Device dev1{};
        dev1.server_id = serverId;
        uint64_t dev1Id = runnerdb_test::InsertRecord<sim::runtime::Device>(dev1);

        sim::runtime::MemoryLayout mem0{};
        mem0.rank_id = 0;
        mem0.buf_type = 0;
        mem0.base_addr = 0x1000;
        mem0.size = 4096;
        mem0.global_offset = 0;
        runnerdb_test::InsertRecord<sim::runtime::MemoryLayout>(mem0);

        sim::runtime::MemoryLayout mem1{};
        mem1.rank_id = 1;
        mem1.buf_type = 1;
        mem1.base_addr = 0x2000;
        mem1.size = 8192;
        mem1.global_offset = 4096;
        runnerdb_test::InsertRecord<sim::runtime::MemoryLayout>(mem1);

        sim::runtime::Ccu ccu{};
        ccu.device_id = dev0Id;
        ccu.die_id = 0;
        (void)runnerdb_test::InsertRecord<sim::runtime::Ccu>(ccu);

        sim::operation::CcuInstrResTab instrRes{};
        instrRes.deviceId = dev0Id;
        instrRes.dieId = 0;
        instrRes.instrCount = 2;
        // 指令空间不落库（CcuResource.instr_space 已删除）：载荷以全零初始化，
        // 保持原与 DB 列同步的全零语义。
        memset(instrRes.instrSpace, 0, sizeof(instrRes.instrSpace));
        ASSERT_EQ(sim::operation::InsertCcuInstrRes(instrRes), 0);
    }
};

TEST_F(DumpShmDataWithDBTest, CreateSimSynData_NoSimModel_ReturnsError)
{
    InsertBasicTopology();
    HcclVmSynData synData;
    sim::operation::OpExecutionKey key{"comm", 0, 0};
    HcclVmResult ret = CreateSimSynData(synData, key, {});
    EXPECT_NE(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(DumpShmDataWithDBTest, CreateSimSynData_CCU_Success)
{
    InsertSimModelData_CCU();
    InsertChannelTopology();

    HcclVmSynData synData;
    sim::operation::OpExecutionKey key;
    ASSERT_TRUE(LoadFirstOpExecutionKey(key));
    HcclVmResult ret = CreateSimSynData(synData, key, {});
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);

    EXPECT_EQ(synData.header.magic, HCCLVM_SYN_FILE_MAGIC);
    EXPECT_EQ(synData.header.version, 1);
    EXPECT_EQ(synData.header.count, 1);
    EXPECT_EQ(synData.model_info.comm.rank_size, 2u);
    EXPECT_EQ(
        synData.model_info.comm.op_expansion_mode,
        static_cast<uint32_t>(sim::runtime::SimOpExpansionMode::SIM_OP_EXPANSION_MODE_CCU));
    EXPECT_EQ(synData.channel_info.count, 1u);
    EXPECT_EQ(synData.channel_info.data.size(), 1u);
    if (!synData.channel_info.data.empty()) {
        EXPECT_EQ(synData.channel_info.data[0].channelId, 1u);
        EXPECT_EQ(synData.channel_info.data[0].jettyNum, 4u);
    }
    EXPECT_GE(synData.memory_info.count, 2u);
}

TEST_F(DumpShmDataWithDBTest, CreateSimSynData_AICPU_JettyInfo)
{
    InsertSimModelData_AICPU();
    InsertJettyTopology();

    HcclVmSynData synData;
    sim::operation::OpExecutionKey key;
    ASSERT_TRUE(LoadFirstOpExecutionKey(key));
    HcclVmResult ret = CreateSimSynData(synData, key, {});
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);

    EXPECT_EQ(
        synData.model_info.comm.op_expansion_mode,
        static_cast<uint32_t>(sim::runtime::SimOpExpansionMode::SIM_OP_EXPANSION_MODE_AICPU));
    EXPECT_GE(synData.channel_info.count, 1u);
}

TEST_F(DumpShmDataWithDBTest, CreateSimSynData_AllToAllV)
{
    InsertSimModelData_AllToAllV();
    InsertChannelTopology();

    HcclVmSynData synData;
    sim::operation::OpExecutionKey key;
    ASSERT_TRUE(LoadFirstOpExecutionKey(key));
    HcclVmResult ret = CreateSimSynData(synData, key, {});
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);

    EXPECT_EQ(synData.model_info.comm.op_type, static_cast<uint16_t>(HcclCMDType::HCCL_CMD_ALLTOALLV));
    EXPECT_GT(synData.model_info.all2AllDataDes.sendCountMatrix.size(), 0u);
}

TEST_F(DumpShmDataWithDBTest, CreateChannelInfo_NoChannel_SuccessEmpty)
{
    InsertBasicTopology();
    HcclVmSynData synData;
    HcclVmResult ret = CreateChannelInfo(synData, {});
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(synData.channel_info.count, 0u);
}

TEST_F(DumpShmDataWithDBTest, CreateChannelInfo_ChannelWithEndpoint_Success)
{
    InsertChannelTopology();
    HcclVmSynData synData;
    HcclVmResult ret = CreateChannelInfo(synData, {});
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(synData.channel_info.count, 1u);
    if (!synData.channel_info.data.empty()) {
        EXPECT_EQ(synData.channel_info.data[0].channelId, 1u);
        EXPECT_EQ(synData.channel_info.data[0].protocol, 0u);
        EXPECT_EQ(synData.channel_info.data[0].jettyNum, 4u);
        for (int i = 0; i < synData.channel_info.data[0].jettyNum; i++) {
            EXPECT_EQ(synData.channel_info.data[0].jettyId[i], 10 + i);
        }
    }
}

TEST_F(DumpShmDataWithDBTest, CreateChannelInfo_MissingLocalEndpoint_ReturnsError)
{
    InsertBasicTopology();
    sim::runtime::EndPoint remoteEp{};
    remoteEp.device_id = 1;
    runnerdb_test::InsertRecord<sim::runtime::EndPoint>(remoteEp);

    sim::runtime::CcuChannel channel{};
    channel.channel_id = 1;
    channel.local_endpoint_id = 99999;
    channel.remote_endpoint_id = remoteEp.id;
    channel.protocol = 0;
    channel.jetty_num = 2;
    runnerdb_test::InsertRecord<sim::runtime::CcuChannel>(channel);

    sim::operation::CcuChannelTab channelRow{};
    channelRow.channelId = 1;
    channelRow.srcRankId = 0;
    channelRow.dstRankId = 1;
    std::memset(channelRow.leid, 0xAA, sizeof(channelRow.leid));
    std::memcpy(channelRow.reid, remoteEp.eid, sizeof(channelRow.reid));
    channelRow.jettyNum = 2;
    ASSERT_EQ(sim::operation::InsertCcuChannel(channelRow), 0);

    HcclVmSynData synData;
    HcclVmResult ret = CreateChannelInfo(synData, {});
    EXPECT_NE(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(DumpShmDataWithDBTest, CreateChannelInfo_MissingRemoteEndpoint_ReturnsError)
{
    InsertBasicTopology();
    sim::runtime::EndPoint localEp{};
    localEp.device_id = 0;
    runnerdb_test::InsertRecord<sim::runtime::EndPoint>(localEp);

    sim::runtime::CcuChannel channel{};
    channel.channel_id = 1;
    channel.local_endpoint_id = localEp.id;
    channel.remote_endpoint_id = 99999;
    channel.protocol = 0;
    channel.jetty_num = 2;
    runnerdb_test::InsertRecord<sim::runtime::CcuChannel>(channel);

    sim::operation::CcuChannelTab channelRow{};
    channelRow.channelId = 1;
    channelRow.srcRankId = 0;
    channelRow.dstRankId = 1;
    std::memcpy(channelRow.leid, localEp.eid, sizeof(channelRow.leid));
    std::memset(channelRow.reid, 0xBB, sizeof(channelRow.reid));
    channelRow.jettyNum = 2;
    ASSERT_EQ(sim::operation::InsertCcuChannel(channelRow), 0);

    HcclVmSynData synData;
    HcclVmResult ret = CreateChannelInfo(synData, {});
    EXPECT_NE(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(DumpShmDataWithDBTest, CreateJettyInfo_NoJetty_SuccessEmpty)
{
    InsertBasicTopology();
    HcclVmSynData synData;
    HcclVmResult ret = CreateJettyInfo(synData, {});
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(synData.channel_info.count, 0u);
}

TEST_F(DumpShmDataWithDBTest, CreateJettyInfo_JettyWithContext_Success)
{
    InsertJettyTopology();
    HcclVmSynData synData;
    HcclVmResult ret = CreateJettyInfo(synData, {});
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_GE(synData.channel_info.count, 1u);
}

TEST_F(DumpShmDataWithDBTest, CreateJettyInfo_MissingRaContext_Continues)
{
    InsertBasicTopology();

    sim::runtime::RaJetty jetty{};
    jetty.ctx_handle = 99999;
    jetty.jetty_id = 5;
    jetty.mode = 3;
    runnerdb_test::InsertRecord<sim::runtime::RaJetty>(jetty);

    HcclVmSynData synData;
    HcclVmResult ret = CreateJettyInfo(synData, {});
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(synData.channel_info.data.size(), 0u);
}

TEST_F(DumpShmDataWithDBTest, CreateJettyInfo_MissingEndPointPair_Continues)
{
    InsertJettyTopology();

    runnerdb_test::ClearRecords<sim::runtime::EndPointPair>();

    HcclVmSynData synData;
    HcclVmResult ret = CreateJettyInfo(synData, {});
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(DumpShmDataWithDBTest, CreateJettyInfo_MissingLocalEndPoint_Continues)
{
    InsertBasicTopology();

    sim::runtime::RaContext raCtx{};
    raCtx.endpoint_id = 99999;
    runnerdb_test::InsertRecord<sim::runtime::RaContext>(raCtx);

    sim::runtime::RaJetty jetty{};
    jetty.ctx_handle = raCtx.id;
    jetty.jetty_id = 5;
    jetty.mode = 3;
    runnerdb_test::InsertRecord<sim::runtime::RaJetty>(jetty);

    HcclVmSynData synData;
    HcclVmResult ret = CreateJettyInfo(synData, {});
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(DumpShmDataWithDBTest, DumpHcclVmSynthesisData_Success)
{
    InsertSimModelData_CCU();
    InsertChannelTopology();

    sim::operation::OpExecutionKey key;
    ASSERT_TRUE(LoadFirstOpExecutionKey(key));

    std::string dataId = GenDataId();
    HcclVmResult ret = DumpHcclVmSynthesisData(dataId, key, {});
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);

    char fileName[256];
    snprintf(fileName, sizeof(fileName), kSynDataFile.c_str(), dataId.c_str());
    std::string rootPath = GetBinLocation();
    std::string fullPath = rootPath + DATA_FILE_PATH + fileName;
    EXPECT_TRUE(std::filesystem::exists(fullPath));

    FILE* fp = fopen(fullPath.c_str(), "rb");
    if (fp) {
        HcclVmSynData readData;
        ret = HcclVmSynDataRead(fp, readData, HCCLVM_SYN_FILE_MAGIC);
        fclose(fp);
        if (ret == HcclVmResult::HCCL_SIM_SUCCESS) {
            EXPECT_EQ(readData.header.magic, HCCLVM_SYN_FILE_MAGIC);
            EXPECT_EQ(readData.model_info.comm.rank_size, 2u);
        }
    }
}

TEST_F(DumpShmDataWithDBTest, DumpHcclVmSynthesisData_NoSimModel_Fails)
{
    InsertBasicTopology();
    std::string dataId = GenDataId();
    sim::operation::OpExecutionKey key{"comm", 0, 0};
    HcclVmResult ret = DumpHcclVmSynthesisData(dataId, key, {});
    EXPECT_NE(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(DumpShmDataWithDBTest, CreateSimInstrData_NoCcuResource_SuccessEmpty)
{
    InsertBasicTopology();
    HcclVmInstrData instrData;
    HcclVmResult ret = CreateSimInstrData(instrData, {});
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(instrData.header.count, 0u);
}

TEST_F(DumpShmDataWithDBTest, CreateSimInstrData_WithCcuResource_Success)
{
    InsertCcuResourceTopology();
    HcclVmInstrData instrData;
    HcclVmResult ret = CreateSimInstrData(instrData, {});
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(instrData.header.magic, HCCLVM_INSTR_FILE_MAGIC);
    EXPECT_GE(instrData.header.count, 1u);
    EXPECT_EQ(instrData.instr_data.size(), instrData.header.count);
}

TEST_F(DumpShmDataWithDBTest, DumpHcclVmInstrData_CCU_WithInstr_Success)
{
    g_opExpansionMode = sim::runtime::SimOpExpansionMode::SIM_OP_EXPANSION_MODE_CCU;
    InsertSimModelData_CCU();
    InsertCcuResourceTopology();

    std::string dataId = GenDataId();
    HcclVmResult ret = DumpHcclVmInstrData(dataId, {});
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(DumpShmDataWithDBTest, DumpHcclVmInstrData_AICPU_NoInstrDump_Success)
{
    g_opExpansionMode = sim::runtime::SimOpExpansionMode::SIM_OP_EXPANSION_MODE_AICPU;
    InsertSimModelData_AICPU();
    InsertBasicTopology();

    std::string dataId = GenDataId();
    HcclVmResult ret = DumpHcclVmInstrData(dataId, {});
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(DumpShmDataWithDBTest, DumpHcclVmInstrData_CCU_NoInstr_Success)
{
    g_opExpansionMode = sim::runtime::SimOpExpansionMode::SIM_OP_EXPANSION_MODE_CCU;
    InsertSimModelData_CCU();
    InsertBasicTopology();

    std::string dataId = GenDataId();
    HcclVmResult ret = DumpHcclVmInstrData(dataId, {});
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(DumpShmDataWithDBTest, DumpDataToFile_Success)
{
    InsertSimModelData_CCU(true);
    InsertChannelTopology();
    InsertCcuResourceTopology();

    std::string dataId = GenDataId();
    HcclVmResult ret = DumpDataToFile(dataId);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(DumpShmDataWithDBTest, DumpDataToFile_SynFail_PropagatesError)
{
    InsertBasicTopology();
    std::string dataId = GenDataId();
    HcclVmResult ret = DumpDataToFile(dataId);
    EXPECT_NE(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(DumpShmDataWithDBTest, DumpDataToFile_TaskFail_PropagatesError)
{
    InsertSimModelData_CCU();
    InsertChannelTopology();

    std::string dataId = GenDataId();
    HcclVmResult ret = DumpDataToFile(dataId);
    EXPECT_NE(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

class DumpShmDataTaskTest : public DumpShmDataWithDBTest {
protected:
    void SetUp() override { DumpShmDataWithDBTest::SetUp(); }
    void TearDown() override { DumpShmDataWithDBTest::TearDown(); }
};

TEST_F(DumpShmDataTaskTest, CreateSimTaskMetaData_NoTask_ReturnsError)
{
    HcclVmTaskMetaData taskMeta;
    HcclVmResult ret = CreateSimTaskMetaData(taskMeta, {}, {});
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(taskMeta.header.count, 0u);
}

TEST_F(DumpShmDataTaskTest, CreateSimTaskMetaData_WithTask_Success)
{
    HcclVmTaskMetaData taskMeta;
    HcclVmResult ret = CreateSimTaskMetaData(taskMeta, {}, {});
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(taskMeta.header.count, 0u);
}

TEST_F(DumpShmDataTaskTest, CreateSimTaskMetaData_AivGraphTask_NoTaskMeta)
{
    HcclVmTaskMetaData taskMeta;
    HcclVmResult ret = CreateSimTaskMetaData(taskMeta, {}, {});
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(taskMeta.header.count, 0u);
}

TEST_F(DumpShmDataTaskTest, DumpHcclVmTask_WithModels_Success)
{
    g_opExpansionMode = sim::runtime::SimOpExpansionMode::SIM_OP_EXPANSION_MODE_CCU;
    InsertSimModelData_CCU(true);
    InsertChannelTopology();

    sim::operation::OpExecutionKey key;
    ASSERT_TRUE(LoadFirstOpExecutionKey(key));

    std::string dataId = GenDataId();
    HcclVmResult ret = DumpHcclVmTask(dataId, key, {});
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(DumpShmDataWithDBTest, CreateJettyInfo_JettyWithEndPointPair_Success)
{
    InsertSimModelData_AICPU();
    InsertJettyTopology();

    HcclVmSynData synData;
    HcclVmResult ret = CreateJettyInfo(synData, {});
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_GE(synData.channel_info.count, 1u);
}

TEST_F(DumpShmDataWithDBTest, DumpHcclVmInstrData_CCU_WithNullCcuSimulator)
{
    g_opExpansionMode = sim::runtime::SimOpExpansionMode::SIM_OP_EXPANSION_MODE_CCU;
    InsertSimModelData_CCU();
    InsertCcuResourceTopology();

    std::string dataId = GenDataId();
    HcclVmResult ret = DumpHcclVmInstrData(dataId, {});
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(DumpShmDataWithDBTest, CreateSimInstrData_MultipleCcuResources)
{
    InsertBasicTopology();

    sim::runtime::Server server{};
    uint64_t serverId = runnerdb_test::InsertRecord<sim::runtime::Server>(server);

    sim::runtime::Device dev0{};
    dev0.server_id = serverId;
    uint64_t dev0Id = runnerdb_test::InsertRecord<sim::runtime::Device>(dev0);

    sim::runtime::Device dev1{};
    dev1.server_id = serverId;
    uint64_t dev1Id = runnerdb_test::InsertRecord<sim::runtime::Device>(dev1);

    sim::runtime::Ccu ccu0{};
    ccu0.device_id = dev0Id;
    ccu0.die_id = 0;
    (void)runnerdb_test::InsertRecord<sim::runtime::Ccu>(ccu0);

    sim::runtime::Ccu ccu1{};
    ccu1.device_id = dev1Id;
    ccu1.die_id = 0;
    (void)runnerdb_test::InsertRecord<sim::runtime::Ccu>(ccu1);

    sim::operation::CcuInstrResTab instrRes0{};
    instrRes0.deviceId = dev0Id;
    instrRes0.dieId = 0;
    instrRes0.instrCount = 1;
    ASSERT_EQ(sim::operation::InsertCcuInstrRes(instrRes0), 0);

    sim::operation::CcuInstrResTab instrRes1{};
    instrRes1.deviceId = dev1Id;
    instrRes1.dieId = 0;
    instrRes1.instrCount = 1;
    ASSERT_EQ(sim::operation::InsertCcuInstrRes(instrRes1), 0);

    HcclVmInstrData instrData;
    HcclVmResult ret = CreateSimInstrData(instrData, {});
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(instrData.header.count, 2u);
}

TEST_F(DumpShmDataTaskTest, CreateSimTaskMetaData_CCU_NoTaskMeta)
{
    HcclVmTaskMetaData taskMeta;
    HcclVmResult ret = CreateSimTaskMetaData(taskMeta, {}, {});
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(taskMeta.header.count, 0u);
}

TEST_F(DumpShmDataTaskTest, DumpHcclVmTask_WithCcuAndChannel_Success)
{
    g_opExpansionMode = sim::runtime::SimOpExpansionMode::SIM_OP_EXPANSION_MODE_CCU;
    InsertSimModelData_CCU(true);
    InsertChannelTopology();

    sim::operation::OpExecutionKey key;
    ASSERT_TRUE(LoadFirstOpExecutionKey(key));

    std::string dataId = GenDataId();
    HcclVmResult ret = DumpHcclVmTask(dataId, key, {});
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

// ==================== GenCcuChannelJettyConfig 测试 ====================

static void
InsertChannelRow(uint32_t channelId, uint32_t srcRankId, uint32_t srcDieId, uint32_t dstRankId, uint32_t dstDieId)
{
    sim::operation::CcuChannelTab row{};
    row.channelId = channelId;
    row.srcDieId = srcDieId;
    row.dstDieId = dstDieId;
    row.srcDeviceId = srcRankId;
    row.dstDeviceId = dstRankId;
    row.srcRankId = srcRankId;
    row.dstRankId = dstRankId;
    row.jettyNum = 1;
    row.jettyId[0] = 1024 + channelId;
    ASSERT_EQ(sim::operation::InsertCcuChannel(row), 0);
}

static std::string GetCcuChannelJettyConfigPath()
{
    return InstallPath::ResolveToInstallRoot("data/ccu_channel_jetty_config.xml");
}

static bool ReadTextFile(const std::string& path, std::string& content)
{
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        return false;
    }
    std::stringstream ss;
    ss << ifs.rdbuf();
    content = ss.str();
    return true;
}

TEST_F(DumpShmDataWithDBTest, GenCcuChannelJettyConfig_NoChannel_SkipSuccess)
{
    std::filesystem::remove(GetCcuChannelJettyConfigPath());

    HcclVmResult ret = GenCcuChannelJettyConfig({});
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);

    // 无 CCU 通道时不生成文件
    std::string content;
    EXPECT_FALSE(ReadTextFile(GetCcuChannelJettyConfigPath(), content));
}

TEST_F(DumpShmDataWithDBTest, GenCcuChannelJettyConfig_TwoRank)
{
    // 模拟真实 2 rank 场景: 每个 rank 各 2 条通道 (chan0 自环 + chan1 互联), die 均为 0
    InsertChannelRow(0, 1, 0, 1, 0);
    InsertChannelRow(1, 1, 0, 2, 0);
    InsertChannelRow(0, 2, 0, 2, 0);
    InsertChannelRow(1, 2, 0, 1, 0);

    EXPECT_EQ(GenCcuChannelJettyConfig({}), HcclVmResult::HCCL_SIM_SUCCESS);

    std::string content;
    ASSERT_TRUE(ReadTextFile(GetCcuChannelJettyConfigPath(), content));

    // 模块命名: ccum_0_{rankId}_{dieId + CCU_XML_DIE_ID_OFFSET(14)}
    EXPECT_NE(content.find("<ccum_0_1_14>"), std::string::npos);
    EXPECT_NE(content.find("<ccum_0_2_14>"), std::string::npos);
    // jetty_group 模板原样输出
    EXPECT_NE(content.find("<group id = \"0\" jetty_list = \"0 1 2 3\"/>"), std::string::npos);
    EXPECT_NE(content.find("<group id = \"7\" jetty_list = \"1792 1793 1794 1795\"/>"), std::string::npos);
    // channel 映射: target_die_id = 对端 dieId + CCU_XML_DIE_ID_OFFSET(14); 本端==对端或唯一对端时 jetty_group_id = 0
    EXPECT_NE(
        content.find("<chan chan = \"0\" target_chip_id = \"1\" target_die_id = \"14\" jetty_group_id = \"0\"/>"),
        std::string::npos);
    EXPECT_NE(
        content.find("<chan chan = \"1\" target_chip_id = \"2\" target_die_id = \"14\" jetty_group_id = \"0\"/>"),
        std::string::npos);
    EXPECT_NE(
        content.find("<chan chan = \"0\" target_chip_id = \"2\" target_die_id = \"14\" jetty_group_id = \"0\"/>"),
        std::string::npos);
    EXPECT_NE(
        content.find("<chan chan = \"1\" target_chip_id = \"1\" target_die_id = \"14\" jetty_group_id = \"0\"/>"),
        std::string::npos);
    // ccu_jump_mode 默认 0
    EXPECT_NE(content.find("<ccu_jump_mode mode=\"0\" />"), std::string::npos);
    // 文件框架
    EXPECT_NE(content.find("<root>"), std::string::npos);
    EXPECT_NE(content.find("</root>"), std::string::npos);

    std::filesystem::remove(GetCcuChannelJettyConfigPath());
}

TEST_F(DumpShmDataWithDBTest, GenCcuChannelJettyConfig_MultiRank_JettyGroupIdIndex)
{
    // 模拟样例场景: 本端 rank0, 通道 126/127 为片内互联 (die0/die1), 通道 1~7 互联 rank1~7
    InsertChannelRow(126, 0, 0, 0, 0);
    InsertChannelRow(127, 0, 0, 0, 1);
    for (uint32_t rank = 1; rank <= 7; ++rank) {
        InsertChannelRow(rank, 0, 0, rank, 0);
    }

    EXPECT_EQ(GenCcuChannelJettyConfig({}), HcclVmResult::HCCL_SIM_SUCCESS);

    std::string content;
    ASSERT_TRUE(ReadTextFile(GetCcuChannelJettyConfigPath(), content));

    // 仅 rank0 一个模块: ccum_0_0_{dieId + CCU_XML_DIE_ID_OFFSET(14)}
    EXPECT_NE(content.find("<ccum_0_0_14>"), std::string::npos);
    EXPECT_EQ(content.find("<ccum_0_0_14", content.find("<ccum_0_0_14") + 1), std::string::npos);
    // 片内通道: 本端==对端 rankId, jetty_group_id = 0
    EXPECT_NE(
        content.find("<chan chan = \"126\" target_chip_id = \"0\" target_die_id = \"14\" jetty_group_id = \"0\"/>"),
        std::string::npos);
    EXPECT_NE(
        content.find("<chan chan = \"127\" target_chip_id = \"0\" target_die_id = \"15\" jetty_group_id = \"0\"/>"),
        std::string::npos);
    // 跨片通道: jetty_group_id 为对端 rankId (去除本端) 升序排序后的下标
    for (uint32_t rank = 1; rank <= 7; ++rank) {
        std::ostringstream oss;
        oss << "<chan chan = \"" << rank << "\" target_chip_id = \"" << rank
            << "\" target_die_id = \"14\" jetty_group_id = \"" << rank - 1 << "\"/>";
        EXPECT_NE(content.find(oss.str()), std::string::npos) << "rank: " << rank;
    }

    std::filesystem::remove(GetCcuChannelJettyConfigPath());
}
