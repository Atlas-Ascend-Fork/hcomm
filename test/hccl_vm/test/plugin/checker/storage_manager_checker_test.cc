/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <gtest/gtest.h>
#include <map>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <vector>

#include "binary_data_operator.h"
#include "runtime_state/db_sim_communicator.h"
#include "storage_manager.h"

using namespace HcclSim;

static std::vector<sim::operation::HalfRTTTab> g_halfRTT;

extern std::map<DeviceId, std::map<u32, ChannelsPerDie>> g_allRankChannelInfo;

namespace {
constexpr CommId kTestCommId = 1;
std::vector<sim::runtime::CommunicatorMemberInfo> g_testCommunicatorMembers;
} // namespace

namespace sim::runtime {
bool GetCommunicatorMembers(uint64_t commId, std::vector<CommunicatorMemberInfo>& members)
{
    members.clear();
    const auto member = std::find_if(
        g_testCommunicatorMembers.begin(), g_testCommunicatorMembers.end(),
        [commId](const CommunicatorMemberInfo& info) {
            return info.memberId == commId;
        });
    if (member == g_testCommunicatorMembers.end()) {
        return false;
    }
    members = g_testCommunicatorMembers;
    return !members.empty();
}
} // namespace sim::runtime

class StorageManagerCheckerTest : public testing::Test {
protected:
    void SetUp() override
    {
        StorageManager::GetInstance().Reset();
        StorageManager::GetInstance().SetDataId("");
        g_testCommunicatorMembers = {{kTestCommId, 0, 0}, {kTestCommId + 1, 1, 1}};
    }

    void TearDown() override
    {
        StorageManager::GetInstance().Reset();
        StorageManager::GetInstance().SetDataId("");
        g_testCommunicatorMembers.clear();
    }

    std::string CreateTempDir()
    {
        char tmpl[] = "/tmp/sm_test_XXXXXX";
        std::string dir = mkdtemp(tmpl);
        return dir;
    }

    void RemoveDir(const std::string& dir)
    {
        std::string cmd = "rm -rf " + dir;
        system(cmd.c_str());
    }

    std::string CreateTestDataDir(const std::string& baseDir, const std::string& dataId)
    {
        std::string pluginDir = baseDir + "/plugin";
        std::string dataDir = baseDir + "/data";
        mkdir(pluginDir.c_str(), 0755);
        mkdir(dataDir.c_str(), 0755);
        return dataDir;
    }

    void WriteSynDataFile(const std::string& path, const HcclVmSynData& synData)
    {
        FILE* fp = fopen(path.c_str(), "wb");
        ASSERT_NE(fp, nullptr);
        auto ret = HcclVmSynDataWrite(fp, synData);
        fclose(fp);
        ASSERT_EQ(ret, HcclResult::HCCL_SUCCESS);
    }

    void WriteInstrDataFile(const std::string& path, const HcclVmInstrData& instrData)
    {
        FILE* fp = fopen(path.c_str(), "wb");
        ASSERT_NE(fp, nullptr);
        auto ret = HcclVmInstrDataWrite(fp, instrData);
        fclose(fp);
        ASSERT_EQ(ret, HcclResult::HCCL_SUCCESS);
    }

    void WriteTaskMetaDataFile(const std::string& path, const HcclVmTaskMetaData& taskMeta)
    {
        FILE* fp = fopen(path.c_str(), "wb");
        ASSERT_NE(fp, nullptr);
        auto ret = HcclVmTaskMetaDataWrite(fp, taskMeta);
        fclose(fp);
        ASSERT_EQ(ret, HcclResult::HCCL_SUCCESS);
    }

    static sim::operation::OpTaskTab EncodeTask(const HcclTaskMetaData& task)
    {
        sim::operation::OpTaskTab row{};
        row.optaskMeta.resize(sizeof(task));
        std::memcpy(row.optaskMeta.data(), &task, sizeof(task));
        return row;
    }

    template <typename T>
    static void AppendVParamField(std::vector<uint8_t>& payload, const T& value)
    {
        const size_t offset = payload.size();
        payload.resize(offset + sizeof(T));
        std::memcpy(payload.data() + offset, &value, sizeof(T));
    }

    static std::vector<uint8_t>
    BuildVParamPayload(uint64_t localCount, const std::vector<uint64_t>& counts, const std::vector<uint64_t>& displs)
    {
        std::vector<uint8_t> payload;
        payload.reserve(sizeof(localCount) + (counts.size() + displs.size()) * sizeof(uint64_t));
        AppendVParamField(payload, localCount);
        for (uint64_t count : counts) {
            AppendVParamField(payload, count);
        }
        for (uint64_t displ : displs) {
            AppendVParamField(payload, displ);
        }
        return payload;
    }

    static std::vector<uint8_t> BuildBatchSendRecvPayload(uint32_t itemNum)
    {
        std::vector<uint8_t> payload;
        payload.reserve(sizeof(itemNum));
        AppendVParamField(payload, itemNum);
        return payload;
    }

    HcclResult ReportBatchSendRecvRankWithParams(
        uint32_t rankId, uint32_t rankSize, uint64_t peerCount, uint32_t itemNum, uint32_t sendPeer, uint32_t recvPeer,
        HcclDataType dataType = HCCL_DATA_TYPE_INT32)
    {
        sim::operation::OpDetailTab detailTab{};
        detailTab.rankId = rankId;
        detailTab.rankSize = rankSize;
        detailTab.dstRank = sendPeer;
        detailTab.srcRank = recvPeer;
        detailTab.opExtInfo = BuildBatchSendRecvPayload(itemNum);

        ::OpDetails detail{};
        detail.opType = static_cast<uint16_t>(HcclCMDType::HCCL_CMD_BATCH_SEND_RECV);
        detail.dataType = static_cast<uint16_t>(dataType);
        detail.opV1.count = peerCount;
        detail.opV1.dataType = static_cast<uint16_t>(dataType);
        return StorageManager::GetInstance().Trans2CheckerParam(detailTab, detail);
    }

    HcclResult ReportBatchSendRecvRank(
        uint32_t rankId, uint32_t rankSize, uint64_t peerCount, uint32_t itemNum,
        HcclDataType dataType = HCCL_DATA_TYPE_INT32)
    {
        const uint32_t sendPeer = rankSize == 0 ? 0 : (rankId + 1U) % rankSize;
        const uint32_t recvPeer = rankSize == 0 ? 0 : (rankId + rankSize - 1U) % rankSize;
        return ReportBatchSendRecvRankWithParams(rankId, rankSize, peerCount, itemNum, sendPeer, recvPeer, dataType);
    }

    HcclResult ReportBatchSendRecvRank(uint32_t rankId, uint32_t rankSize, uint64_t peerCount)
    {
        return ReportBatchSendRecvRank(rankId, rankSize, peerCount, 2);
    }

    HcclResult ReportVRank(
        HcclCMDType cmdType, uint32_t rankId, uint32_t rankSize, uint64_t localCount,
        const std::vector<uint64_t>& counts, const std::vector<uint64_t>& displs,
        HcclDataType dataType = HCCL_DATA_TYPE_INT32, HcclReduceOp reduceType = HCCL_REDUCE_SUM)
    {
        sim::operation::OpDetailTab detailTab{};
        detailTab.rankId = rankId;
        detailTab.rankSize = rankSize;
        detailTab.opExtInfo = BuildVParamPayload(localCount, counts, displs);

        ::OpDetails detail{};
        detail.opType = static_cast<uint16_t>(cmdType);
        detail.dataType = static_cast<uint16_t>(dataType);
        detail.reduceType = static_cast<uint16_t>(reduceType);
        detail.opV1.count = localCount;
        return StorageManager::GetInstance().Trans2CheckerParam(detailTab, detail);
    }

    static std::vector<uint8_t> BuildMatrixPayload(const std::vector<uint64_t>& matrix)
    {
        const uint32_t count = static_cast<uint32_t>(matrix.size());
        std::vector<uint8_t> payload(sizeof(count) + matrix.size() * sizeof(uint64_t));
        std::memcpy(payload.data(), &count, sizeof(count));
        if (!matrix.empty()) {
            std::memcpy(payload.data() + sizeof(count), matrix.data(), matrix.size() * sizeof(uint64_t));
        }
        return payload;
    }

    HcclResult
    ReportAll2AllMatrix(HcclCMDType cmdType, uint32_t rankId, uint32_t rankSize, const std::vector<uint64_t>& matrix)
    {
        sim::operation::OpDetailTab detailTab{};
        detailTab.rankId = rankId;
        detailTab.rankSize = rankSize;
        detailTab.opExtInfo = BuildMatrixPayload(matrix);

        ::OpDetails detail{};
        detail.opType = static_cast<uint16_t>(cmdType);
        detail.dataType = static_cast<uint16_t>(HCCL_DATA_TYPE_INT32);
        detail.opV2.sendDataType = static_cast<uint16_t>(HCCL_DATA_TYPE_INT32);
        detail.opV2.recvDataType = static_cast<uint16_t>(HCCL_DATA_TYPE_INT32);
        return StorageManager::GetInstance().Trans2CheckerParam(detailTab, detail);
    }
};

TEST_F(StorageManagerCheckerTest, GetInstance_Singleton)
{
    StorageManager& instance1 = StorageManager::GetInstance();
    StorageManager& instance2 = StorageManager::GetInstance();
    EXPECT_EQ(&instance1, &instance2);
}

TEST_F(StorageManagerCheckerTest, Reset_NoThrow) { EXPECT_NO_THROW(StorageManager::GetInstance().Reset()); }

TEST_F(StorageManagerCheckerTest, Reset_ClearsAllState)
{
    StorageManager::GetInstance().SetDataId("test_id");
    EXPECT_EQ(StorageManager::GetInstance().GetDataId(), "test_id");
    StorageManager::GetInstance().Reset();
    EXPECT_EQ(StorageManager::GetInstance().GetRankSize(), 0);
    EXPECT_EQ(StorageManager::GetInstance().GetHvmInstrData().instr_data.size(), 0);
    EXPECT_EQ(StorageManager::GetInstance().GetHvmTaskMetaData().task_meta.size(), 0);
}

TEST_F(StorageManagerCheckerTest, Reset_ClearsGlobalChannelInfo)
{
    g_allRankChannelInfo[0][0][0] = {1, 0};
    StorageManager::GetInstance().Reset();
    EXPECT_TRUE(g_allRankChannelInfo.empty());
}

TEST_F(StorageManagerCheckerTest, SetDataId_GetDataId)
{
    StorageManager::GetInstance().SetDataId("abc123");
    EXPECT_EQ(StorageManager::GetInstance().GetDataId(), "abc123");
}

TEST_F(StorageManagerCheckerTest, SetDataId_Empty)
{
    StorageManager::GetInstance().SetDataId("");
    EXPECT_EQ(StorageManager::GetInstance().GetDataId(), "");
}

TEST_F(StorageManagerCheckerTest, SetDataId_LongString)
{
    std::string longId(256, 'x');
    StorageManager::GetInstance().SetDataId(longId);
    EXPECT_EQ(StorageManager::GetInstance().GetDataId(), longId);
}

TEST_F(StorageManagerCheckerTest, SetDataId_Overwrite)
{
    StorageManager::GetInstance().SetDataId("first");
    StorageManager::GetInstance().SetDataId("second");
    EXPECT_EQ(StorageManager::GetInstance().GetDataId(), "second");
}

TEST_F(StorageManagerCheckerTest, SetDataId_ThreadSafety)
{
    const int threadCount = 10;
    std::vector<std::thread> threads;
    for (int i = 0; i < threadCount; i++) {
        threads.emplace_back([i]() {
            StorageManager::GetInstance().SetDataId("thread_" + std::to_string(i));
        });
    }
    for (auto& t : threads) {
        t.join();
    }
    std::string result = StorageManager::GetInstance().GetDataId();
    EXPECT_TRUE(result.substr(0, 7) == "thread_");
}

TEST_F(StorageManagerCheckerTest, GetRankSize_Default) { EXPECT_EQ(StorageManager::GetInstance().GetRankSize(), 0); }

TEST_F(StorageManagerCheckerTest, GetCheckerParam_Default)
{
    CheckerParam param = StorageManager::GetInstance().GetCheckerParam();
    EXPECT_EQ(param.cmdType, static_cast<HcclCMDType>(0));
    EXPECT_EQ(param.rankSize, 0);
    EXPECT_EQ(param.dataType, static_cast<HcclDataType>(0));
    EXPECT_EQ(param.dataCount, 0);
    EXPECT_EQ(param.reduceType, static_cast<HcclReduceOp>(0));
    EXPECT_EQ(param.srcRank, 0);
    EXPECT_EQ(param.dstRank, 0);
    EXPECT_EQ(param.root, 0);
}

TEST_F(StorageManagerCheckerTest, CheckerParam_DefaultInitialization)
{
    CheckerParam param;
    EXPECT_EQ(param.cmdType, static_cast<HcclCMDType>(0));
    EXPECT_EQ(param.rankSize, 0);
    EXPECT_EQ(param.dataType, static_cast<HcclDataType>(0));
    EXPECT_EQ(param.dataCount, 0);
}

TEST_F(StorageManagerCheckerTest, CheckerParam_CustomInitialization)
{
    CheckerParam param;
    param.cmdType = HCCL_CMD_ALLREDUCE;
    param.rankSize = 4;
    param.dataType = HCCL_DATA_TYPE_INT32;
    param.dataCount = 1024;
    param.reduceType = HCCL_REDUCE_SUM;
    param.srcRank = 0;
    param.dstRank = 1;
    param.root = 2;
    EXPECT_EQ(param.cmdType, HCCL_CMD_ALLREDUCE);
    EXPECT_EQ(param.rankSize, 4);
    EXPECT_EQ(param.dataType, HCCL_DATA_TYPE_INT32);
    EXPECT_EQ(param.dataCount, 1024);
    EXPECT_EQ(param.reduceType, HCCL_REDUCE_SUM);
    EXPECT_EQ(param.srcRank, 0);
    EXPECT_EQ(param.dstRank, 1);
    EXPECT_EQ(param.root, 2);
}

TEST_F(StorageManagerCheckerTest, GetBlockSize_NonExistentRank)
{
    uint64_t size = StorageManager::GetInstance().GetBlockSize("", 0, 0, 999, BufferType::INPUT);
    EXPECT_EQ(size, 0);
}

TEST_F(StorageManagerCheckerTest, GetBlockSize_NonExistentBufferType)
{
    uint64_t size = StorageManager::GetInstance().GetBlockSize("", 0, 0, 0, BufferType::CCL);
    EXPECT_EQ(size, 0);
}

TEST_F(StorageManagerCheckerTest, GetSlice_NonExistentRank)
{
    DataSlice slice;
    uint32_t rank = 999;
    HcclResult ret = StorageManager::GetInstance().GetSlice("", 0, 0, 0x1000, 1024, slice, &rank);
    EXPECT_NE(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(StorageManagerCheckerTest, DataSlice_DefaultConstruction)
{
    DataSlice slice;
    EXPECT_EQ(slice.GetSize(), 0);
    EXPECT_EQ(slice.GetOffset(), 0);
    EXPECT_EQ(slice.GetType(), BufferType::INPUT);
}

TEST_F(StorageManagerCheckerTest, DataSlice_CustomConstruction)
{
    DataSlice slice(BufferType::OUTPUT, 0x1000, 2048);
    EXPECT_EQ(slice.GetType(), BufferType::OUTPUT);
    EXPECT_EQ(slice.GetOffset(), 0x1000);
    EXPECT_EQ(slice.GetSize(), 2048);
}

TEST_F(StorageManagerCheckerTest, GetSlice_NonExistent)
{
    DataSlice slice;
    uint32_t rank = 0;
    HcclResult ret = StorageManager::GetInstance().GetSlice("", 0, 0, 0x1000, 1024, slice, &rank);
    EXPECT_NE(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(StorageManagerCheckerTest, GetSlice_NullRankPtr)
{
    DataSlice slice;
    HcclResult ret = StorageManager::GetInstance().GetSlice("", 0, 0, 0x1000, 1024, slice, nullptr);
    EXPECT_NE(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(StorageManagerCheckerTest, GetHvmInstrData_Default)
{
    HcclVmInstrData instrData = StorageManager::GetInstance().GetHvmInstrData();
    EXPECT_EQ(instrData.instr_data.size(), 0);
}

TEST_F(StorageManagerCheckerTest, GetHvmTaskMetaData_Default)
{
    HcclVmTaskMetaData taskMeta = StorageManager::GetInstance().GetHvmTaskMetaData();
    EXPECT_EQ(taskMeta.task_meta.size(), 0);
}

TEST_F(StorageManagerCheckerTest, LoadHcclVmSynthesisData_RejectsMissingCommunicator)
{
    uint32_t rankId = 0;
    sim::operation::OpMemInfoTab memInfo{};
    std::vector<sim::operation::CcuChannelTab> channels;
    std::vector<sim::operation::HalfRTTTab> halfRTT;
    HcclResult ret = StorageManager::GetInstance().LoadHcclVmSynthesisData(
        rankId, 0, "", 0, 0, rankId, memInfo, channels, halfRTT);
    EXPECT_EQ(ret, HcclResult::HCCL_E_PARA);
}

TEST_F(StorageManagerCheckerTest, LoadHcclVmSynthesisData_ExpandsAllCommunicatorMembers)
{
    sim::operation::OpMemInfoTab memInfo{};
    std::vector<sim::operation::CcuChannelTab> channels;
    std::vector<sim::operation::HalfRTTTab> halfRTT;

    ASSERT_EQ(
        StorageManager::GetInstance().LoadHcclVmSynthesisData(
            1, kTestCommId + 1, "", 0, 0, 1, memInfo, channels, halfRTT),
        HcclResult::HCCL_SUCCESS);

    const auto firstMemberMappings = StorageManager::GetInstance().GetDeviceRankMappings(kTestCommId);
    EXPECT_EQ(firstMemberMappings, (std::map<DeviceId, RankId>{{0, 0}, {1, 1}}));
    const auto secondMemberMappings = StorageManager::GetInstance().GetDeviceRankMappings(kTestCommId + 1);
    EXPECT_EQ(secondMemberMappings, firstMemberMappings);
}

TEST_F(StorageManagerCheckerTest, LoadHcclVmInstrData_DefaultPayload)
{
    std::vector<sim::operation::CcuInstrResTab> instrRes;
    HcclResult ret = StorageManager::GetInstance().LoadHcclVmInstrData(instrRes);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(StorageManagerCheckerTest, LoadHcclVmTaskMetaData_DefaultPayload)
{
    std::vector<std::vector<sim::operation::OpTaskTab>> allTasks;
    HcclResult ret = StorageManager::GetInstance().LoadHcclVmTaskMetaData(allTasks);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(StorageManagerCheckerTest, Trans2CheckerParam_Default)
{
    sim::operation::OpDetailTab detailTab{};
    ::OpDetails detail{};
    HcclResult ret = StorageManager::GetInstance().Trans2CheckerParam(detailTab, detail);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    CheckerParam param = StorageManager::GetInstance().GetCheckerParam();
    EXPECT_EQ(param.rankSize, 0);
    EXPECT_EQ(param.dataCount, 0);
}

TEST_F(StorageManagerCheckerTest, Trans2CheckerParam_StoresOperatorMainStream)
{
    StorageManager& storage = StorageManager::GetInstance();
    sim::operation::OpDetailTab detailTab{};
    detailTab.deviceId = 7;
    detailTab.streamId = 11;
    ::OpDetails detail{};

    ASSERT_EQ(storage.Trans2CheckerParam(detailTab, detail), HcclResult::HCCL_SUCCESS);

    TaskGraphGeneratorV3::StreamId mainStreamId = TaskGraphGeneratorV3::INVALID_STREAM_ID;
    EXPECT_TRUE(storage.GetMainStreamId(detailTab.deviceId, mainStreamId));
    EXPECT_EQ(mainStreamId, 11U);

    storage.BeginOpGroup("", 0, 0);
    EXPECT_FALSE(storage.GetMainStreamId(detailTab.deviceId, mainStreamId));
}

TEST_F(StorageManagerCheckerTest, FinalizeOpGroup_BatchSendRecvRingBuildsCanonicalParameters)
{
    constexpr uint32_t kRankSize = 4;
    constexpr uint64_t kPeerCount = 128;
    StorageManager& storage = StorageManager::GetInstance();

    storage.BeginOpGroup("", 0, 0);
    for (uint32_t rankId = 0; rankId < kRankSize; ++rankId) {
        ASSERT_EQ(ReportBatchSendRecvRank(rankId, kRankSize, kPeerCount), HcclResult::HCCL_SUCCESS);
    }

    ASSERT_EQ(storage.FinalizeOpGroup(), HcclResult::HCCL_SUCCESS);
    const CheckerParam param = storage.GetCheckerParam();
    EXPECT_EQ(param.cmdType, HcclCMDType::HCCL_CMD_BATCH_SEND_RECV);
    EXPECT_EQ(param.rankSize, kRankSize);
    EXPECT_EQ(param.dataCount, kPeerCount);
    EXPECT_EQ(param.dataType, HCCL_DATA_TYPE_INT32);
    ASSERT_EQ(param.batchSendRecvRankParams.size(), kRankSize);
    for (uint32_t rankId = 0; rankId < kRankSize; ++rankId) {
        const auto& rankParam = param.batchSendRecvRankParams[rankId];
        EXPECT_EQ(rankParam.itemNum, 2U);
        EXPECT_EQ(rankParam.peerCount, kPeerCount);
        EXPECT_EQ(rankParam.dataType, HCCL_DATA_TYPE_INT32);
        EXPECT_EQ(rankParam.sendPeer, (rankId + 1U) % kRankSize);
        EXPECT_EQ(rankParam.recvPeer, (rankId + kRankSize - 1U) % kRankSize);
    }
}

TEST_F(StorageManagerCheckerTest, Trans2CheckerParam_RejectsMalformedBatchSendRecvExtInfo)
{
    StorageManager& storage = StorageManager::GetInstance();
    storage.BeginOpGroup("", 0, 0);

    sim::operation::OpDetailTab detailTab{};
    detailTab.rankId = 0;
    detailTab.rankSize = 2;
    detailTab.opExtInfo = BuildBatchSendRecvPayload(2);
    detailTab.opExtInfo.pop_back();
    ::OpDetails detail{};
    detail.opType = static_cast<uint16_t>(HcclCMDType::HCCL_CMD_BATCH_SEND_RECV);
    detail.dataType = static_cast<uint16_t>(HCCL_DATA_TYPE_INT32);
    detail.opV1.count = 128;
    detail.opV1.dataType = static_cast<uint16_t>(HCCL_DATA_TYPE_INT32);

    EXPECT_EQ(storage.Trans2CheckerParam(detailTab, detail), HcclResult::HCCL_E_PARA);
}

TEST_F(StorageManagerCheckerTest, Trans2CheckerParam_RejectsInvalidBatchSendRecvItemNum)
{
    StorageManager::GetInstance().BeginOpGroup("", 0, 0);
    EXPECT_EQ(ReportBatchSendRecvRank(0, 2, 128, 3), HcclResult::HCCL_E_PARA);
}

TEST_F(StorageManagerCheckerTest, Trans2CheckerParam_RejectsInvalidBatchSendRecvRingPeer)
{
    StorageManager::GetInstance().BeginOpGroup("", 0, 0);
    EXPECT_EQ(ReportBatchSendRecvRankWithParams(0, 4, 128, 2, 2, 3), HcclResult::HCCL_E_PARA);
}

TEST_F(StorageManagerCheckerTest, Trans2CheckerParam_RejectsSingleRankBatchSendRecvRing)
{
    StorageManager::GetInstance().BeginOpGroup("", 0, 0);
    EXPECT_EQ(ReportBatchSendRecvRank(0, 1, 128), HcclResult::HCCL_E_PARA);
}

TEST_F(StorageManagerCheckerTest, Trans2CheckerParam_RejectsDuplicateBatchSendRecvRank)
{
    StorageManager::GetInstance().BeginOpGroup("", 0, 0);
    ASSERT_EQ(ReportBatchSendRecvRank(0, 2, 128), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(ReportBatchSendRecvRank(0, 2, 128), HcclResult::HCCL_E_PARA);
}

TEST_F(StorageManagerCheckerTest, FinalizeOpGroup_RejectsMissingBatchSendRecvRank)
{
    StorageManager& storage = StorageManager::GetInstance();
    storage.BeginOpGroup("", 0, 0);
    ASSERT_EQ(ReportBatchSendRecvRank(0, 2, 128), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(storage.FinalizeOpGroup(), HcclResult::HCCL_E_PARA);
}

TEST_F(StorageManagerCheckerTest, FinalizeOpGroup_RejectsInconsistentBatchSendRecvRanks)
{
    StorageManager& storage = StorageManager::GetInstance();
    storage.BeginOpGroup("", 0, 0);
    ASSERT_EQ(ReportBatchSendRecvRank(0, 2, 128), HcclResult::HCCL_SUCCESS);
    ASSERT_EQ(ReportBatchSendRecvRank(1, 2, 64), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(storage.FinalizeOpGroup(), HcclResult::HCCL_E_PARA);
}

TEST_F(StorageManagerCheckerTest, FinalizeOpGroup_RejectsInconsistentBatchSendRecvDataTypes)
{
    StorageManager& storage = StorageManager::GetInstance();
    storage.BeginOpGroup("", 0, 0);
    ASSERT_EQ(ReportBatchSendRecvRank(0, 2, 128, 2, HCCL_DATA_TYPE_INT32), HcclResult::HCCL_SUCCESS);
    ASSERT_EQ(ReportBatchSendRecvRank(1, 2, 128, 2, HCCL_DATA_TYPE_FP32), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(storage.FinalizeOpGroup(), HcclResult::HCCL_E_PARA);
}

TEST_F(StorageManagerCheckerTest, FinalizeOpGroup_AllGatherVBuildsCanonicalDataDescriptor)
{
    constexpr uint32_t kRankSize = 4;
    const std::vector<uint64_t> counts = {2, 0, 3, 1};
    const std::vector<uint64_t> displs = {0, 2, 2, 5};
    StorageManager& storage = StorageManager::GetInstance();

    storage.BeginOpGroup("", 0, 0);
    for (uint32_t rankId = 0; rankId < kRankSize; ++rankId) {
        ASSERT_EQ(
            ReportVRank(HcclCMDType::HCCL_CMD_ALLGATHER_V, rankId, kRankSize, counts[rankId], counts, displs),
            HcclResult::HCCL_SUCCESS);
    }

    ASSERT_EQ(storage.FinalizeOpGroup(), HcclResult::HCCL_SUCCESS);
    const CheckerParam param = storage.GetCheckerParam();
    EXPECT_EQ(param.vDataDes.dataType, static_cast<uint16_t>(HCCL_DATA_TYPE_INT32));
    EXPECT_EQ(param.vDataDes.count, kRankSize);
    EXPECT_EQ(param.vDataDes.counts, counts);
    EXPECT_EQ(param.vDataDes.displs, displs);
}

TEST_F(StorageManagerCheckerTest, FinalizeOpGroup_AllGatherVRejectsCountMismatch)
{
    constexpr uint32_t kRankSize = 4;
    const std::vector<uint64_t> counts = {2, 0, 3, 1};
    const std::vector<uint64_t> displs = {0, 2, 2, 5};
    StorageManager& storage = StorageManager::GetInstance();

    storage.BeginOpGroup("", 0, 0);
    for (uint32_t rankId = 0; rankId < kRankSize; ++rankId) {
        std::vector<uint64_t> recvCounts = counts;
        if (rankId == 2) {
            recvCounts[0] = 9;
        }
        ASSERT_EQ(
            ReportVRank(HcclCMDType::HCCL_CMD_ALLGATHER_V, rankId, kRankSize, counts[rankId], recvCounts, displs),
            HcclResult::HCCL_SUCCESS);
    }

    EXPECT_EQ(storage.FinalizeOpGroup(), HcclResult::HCCL_E_PARA);
}

TEST_F(StorageManagerCheckerTest, FinalizeOpGroup_AllGatherVAcceptsAllZeroCounts)
{
    constexpr uint32_t kRankSize = 3;
    const std::vector<uint64_t> counts = {0, 0, 0};
    const std::vector<uint64_t> displs = {0, 0, 0};
    StorageManager& storage = StorageManager::GetInstance();

    storage.BeginOpGroup("", 0, 0);
    for (uint32_t rankId = 0; rankId < kRankSize; ++rankId) {
        ASSERT_EQ(
            ReportVRank(HcclCMDType::HCCL_CMD_ALLGATHER_V, rankId, kRankSize, 0, counts, displs),
            HcclResult::HCCL_SUCCESS);
    }

    ASSERT_EQ(storage.FinalizeOpGroup(), HcclResult::HCCL_SUCCESS);
    const CheckerParam param = storage.GetCheckerParam();
    EXPECT_EQ(param.vDataDes.counts, counts);
    EXPECT_EQ(param.vDataDes.displs, displs);
}

TEST_F(StorageManagerCheckerTest, FinalizeOpGroup_ReduceScatterVBuildsCanonicalDataDescriptor)
{
    constexpr uint32_t kRankSize = 4;
    const std::vector<uint64_t> counts = {2, 0, 3, 1};
    const std::vector<uint64_t> displs = {0, 2, 2, 5};
    StorageManager& storage = StorageManager::GetInstance();

    storage.BeginOpGroup("", 0, 0);
    for (uint32_t rankId = 0; rankId < kRankSize; ++rankId) {
        ASSERT_EQ(
            ReportVRank(HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V, rankId, kRankSize, counts[rankId], counts, displs),
            HcclResult::HCCL_SUCCESS);
    }

    ASSERT_EQ(storage.FinalizeOpGroup(), HcclResult::HCCL_SUCCESS);
    const CheckerParam param = storage.GetCheckerParam();
    EXPECT_EQ(param.vDataDes.dataType, static_cast<uint16_t>(HCCL_DATA_TYPE_INT32));
    EXPECT_EQ(param.vDataDes.count, kRankSize);
    EXPECT_EQ(param.vDataDes.counts, counts);
    EXPECT_EQ(param.vDataDes.displs, displs);
}

TEST_F(StorageManagerCheckerTest, FinalizeOpGroup_ReduceScatterVRejectsCountMismatch)
{
    constexpr uint32_t kRankSize = 4;
    const std::vector<uint64_t> counts = {2, 0, 3, 1};
    const std::vector<uint64_t> displs = {0, 2, 2, 5};
    StorageManager& storage = StorageManager::GetInstance();

    storage.BeginOpGroup("", 0, 0);
    for (uint32_t rankId = 0; rankId < kRankSize; ++rankId) {
        std::vector<uint64_t> sendCounts = counts;
        if (rankId == 3) {
            sendCounts[0] = 9;
        }
        ASSERT_EQ(
            ReportVRank(HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V, rankId, kRankSize, counts[rankId], sendCounts, displs),
            HcclResult::HCCL_SUCCESS);
    }

    EXPECT_EQ(storage.FinalizeOpGroup(), HcclResult::HCCL_E_PARA);
}

TEST_F(StorageManagerCheckerTest, FinalizeOpGroup_RejectsMissingVRankReport)
{
    constexpr uint32_t kRankSize = 4;
    const std::vector<uint64_t> counts = {2, 0, 3, 1};
    const std::vector<uint64_t> displs = {0, 2, 2, 5};
    StorageManager& storage = StorageManager::GetInstance();

    storage.BeginOpGroup("", 0, 0);
    for (uint32_t rankId = 0; rankId < kRankSize - 1; ++rankId) {
        ASSERT_EQ(
            ReportVRank(HcclCMDType::HCCL_CMD_ALLGATHER_V, rankId, kRankSize, counts[rankId], counts, displs),
            HcclResult::HCCL_SUCCESS);
    }

    EXPECT_EQ(storage.FinalizeOpGroup(), HcclResult::HCCL_E_PARA);
}

TEST_F(StorageManagerCheckerTest, Trans2CheckerParam_RejectsDuplicateVRankReport)
{
    constexpr uint32_t kRankSize = 2;
    const std::vector<uint64_t> counts = {1, 2};
    const std::vector<uint64_t> displs = {0, 1};
    StorageManager& storage = StorageManager::GetInstance();

    storage.BeginOpGroup("", 0, 0);
    ASSERT_EQ(
        ReportVRank(HcclCMDType::HCCL_CMD_ALLGATHER_V, 0, kRankSize, counts[0], counts, displs),
        HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(
        ReportVRank(HcclCMDType::HCCL_CMD_ALLGATHER_V, 0, kRankSize, counts[0], counts, displs),
        HcclResult::HCCL_E_PARA);
}

TEST_F(StorageManagerCheckerTest, BeginOpGroup_ClearsCompletedVRankParameters)
{
    constexpr uint32_t kFirstRankSize = 4;
    const std::vector<uint64_t> firstCounts = {2, 0, 3, 1};
    const std::vector<uint64_t> firstDispls = {0, 2, 2, 5};
    StorageManager& storage = StorageManager::GetInstance();

    storage.BeginOpGroup("", 0, 0);
    for (uint32_t rankId = 0; rankId < kFirstRankSize; ++rankId) {
        ASSERT_EQ(
            ReportVRank(
                HcclCMDType::HCCL_CMD_ALLGATHER_V, rankId, kFirstRankSize, firstCounts[rankId], firstCounts,
                firstDispls),
            HcclResult::HCCL_SUCCESS);
    }
    ASSERT_EQ(storage.FinalizeOpGroup(), HcclResult::HCCL_SUCCESS);

    constexpr uint32_t kSecondRankSize = 2;
    const std::vector<uint64_t> secondCounts = {5, 7};
    const std::vector<uint64_t> secondDispls = {0, 5};
    storage.BeginOpGroup("", 0, 0);
    for (uint32_t rankId = 0; rankId < kSecondRankSize; ++rankId) {
        ASSERT_EQ(
            ReportVRank(
                HcclCMDType::HCCL_CMD_ALLGATHER_V, rankId, kSecondRankSize, secondCounts[rankId], secondCounts,
                secondDispls),
            HcclResult::HCCL_SUCCESS);
    }
    ASSERT_EQ(storage.FinalizeOpGroup(), HcclResult::HCCL_SUCCESS);

    const CheckerParam param = storage.GetCheckerParam();
    EXPECT_EQ(param.vDataDes.count, kSecondRankSize);
    EXPECT_EQ(param.vDataDes.counts, secondCounts);
    EXPECT_EQ(param.vDataDes.displs, secondDispls);
}

TEST_F(StorageManagerCheckerTest, Trans2CheckerParam_RejectsMalformedVParamPayload)
{
    constexpr uint32_t kRankSize = 2;
    const std::vector<uint64_t> counts = {1, 2};
    const std::vector<uint64_t> displs = {0, 1};
    StorageManager& storage = StorageManager::GetInstance();

    storage.BeginOpGroup("", 0, 0);
    sim::operation::OpDetailTab detailTab{};
    detailTab.rankId = 0;
    detailTab.rankSize = kRankSize;
    detailTab.opExtInfo = BuildVParamPayload(counts[0], counts, displs);
    detailTab.opExtInfo.pop_back();

    ::OpDetails detail{};
    detail.opType = static_cast<uint16_t>(HcclCMDType::HCCL_CMD_ALLGATHER_V);
    detail.dataType = static_cast<uint16_t>(HCCL_DATA_TYPE_INT32);
    EXPECT_EQ(storage.Trans2CheckerParam(detailTab, detail), HcclResult::HCCL_E_PARA);
}

TEST_F(StorageManagerCheckerTest, Trans2CheckerParam_RejectsVLocalCountMismatch)
{
    constexpr uint32_t kRankSize = 2;
    const std::vector<uint64_t> counts = {1, 2};
    const std::vector<uint64_t> displs = {0, 1};
    StorageManager& storage = StorageManager::GetInstance();

    storage.BeginOpGroup("", 0, 0);
    sim::operation::OpDetailTab detailTab{};
    detailTab.rankId = 0;
    detailTab.rankSize = kRankSize;
    detailTab.opExtInfo = BuildVParamPayload(counts[0], counts, displs);

    ::OpDetails detail{};
    detail.opType = static_cast<uint16_t>(HcclCMDType::HCCL_CMD_ALLGATHER_V);
    detail.dataType = static_cast<uint16_t>(HCCL_DATA_TYPE_INT32);
    detail.opV1.count = counts[0] + 1;
    EXPECT_EQ(storage.Trans2CheckerParam(detailTab, detail), HcclResult::HCCL_E_PARA);
}

TEST_F(StorageManagerCheckerTest, MergeAll2AllVSendCountMatrix_PreservesExistingOrMerge)
{
    constexpr uint32_t kRankSize = 2;
    StorageManager& storage = StorageManager::GetInstance();

    storage.BeginOpGroup("", 0, 0);
    ASSERT_EQ(
        ReportAll2AllMatrix(HcclCMDType::HCCL_CMD_ALLTOALLVC, 0, kRankSize, {1, 2, 0, 4}), HcclResult::HCCL_SUCCESS);
    ASSERT_EQ(
        ReportAll2AllMatrix(HcclCMDType::HCCL_CMD_ALLTOALLVC, 1, kRankSize, {0, 8, 16, 0}), HcclResult::HCCL_SUCCESS);

    storage.MergeAll2AllVSendCountMatrix();
    const CheckerParam param = storage.GetCheckerParam();
    EXPECT_EQ(param.all2AllDataDes.count, 4u);
    EXPECT_EQ(param.all2AllDataDes.sendCountMatrix, (std::vector<uint64_t>{1, 10, 16, 4}));
}

TEST_F(StorageManagerCheckerTest, Trans2CheckerParam_WithOpDetail)
{
    sim::operation::OpDetailTab detailTab{};
    detailTab.rankSize = 8;
    detailTab.srcRank = 0;
    detailTab.dstRank = 1;
    detailTab.root = 0;
    ::OpDetails detail{};
    detail.opType = 1;
    detail.dataType = 1;
    detail.reduceType = 0;
    detail.opV1.count = 4096;
    HcclResult transRet = StorageManager::GetInstance().Trans2CheckerParam(detailTab, detail);
    EXPECT_EQ(transRet, HcclResult::HCCL_SUCCESS);

    CheckerParam param = StorageManager::GetInstance().GetCheckerParam();
    EXPECT_EQ(param.rankSize, 8);
    EXPECT_EQ(param.dataCount, 4096);
    EXPECT_EQ(param.srcRank, 0);
    EXPECT_EQ(param.dstRank, 1);
    EXPECT_EQ(param.root, 0);
}

TEST_F(StorageManagerCheckerTest, InitCcuInfo_Default)
{
    DevType devType;
    std::vector<uint64_t> resourceBaseAddr;
    StorageManager::GetInstance().InitCcuInfo(devType, resourceBaseAddr);
    EXPECT_EQ(resourceBaseAddr.size(), 2);
    EXPECT_EQ(resourceBaseAddr[0], 0x123123123);
    EXPECT_EQ(resourceBaseAddr[1], 0x456456456);
}

TEST_F(StorageManagerCheckerTest, InitCcuInfo_WithOpDetail)
{
    sim::operation::OpDetailTab detailTab{};
    detailTab.devType = 2;
    ::OpDetails detail{};
    StorageManager::GetInstance().Trans2CheckerParam(detailTab, detail);

    DevType devType;
    std::vector<uint64_t> resourceBaseAddr;
    StorageManager::GetInstance().InitCcuInfo(devType, resourceBaseAddr);
    EXPECT_EQ(static_cast<int>(devType), 2);
    EXPECT_EQ(resourceBaseAddr.size(), 2);
    EXPECT_EQ(resourceBaseAddr[0], 0x123123123);
    EXPECT_EQ(resourceBaseAddr[1], 0x456456456);
}

TEST_F(StorageManagerCheckerTest, SetDataId_DoesNotLoadCheckerParam)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_partial";
    std::string dataDir = CreateTestDataDir(tmpDir, dataId);

    std::string jsonContent = R"({"rankSize":2})";
    std::string filePath = dataDir + "/" + dataId + "_model.jsonl.gz";

    char savedCwd[4096];
    getcwd(savedCwd, sizeof(savedCwd));
    chdir(tmpDir.c_str());

    StorageManager::GetInstance().SetDataId(dataId);

    CheckerParam param = StorageManager::GetInstance().GetCheckerParam();
    EXPECT_EQ(param.rankSize, 0);
    EXPECT_EQ(param.cmdType, static_cast<HcclCMDType>(0));
    EXPECT_EQ(param.dataCount, 0);

    chdir(savedCwd);
    RemoveDir(tmpDir);
}

TEST_F(StorageManagerCheckerTest, LoadHcclVmSynthesisData_PopulatesMemLayout)
{
    std::vector<sim::operation::CcuChannelTab> channels;
    std::vector<sim::operation::HalfRTTTab> halfRTT;
    sim::operation::OpMemInfoTab rank0{};
    rank0.inputAddr = 1024;
    rank0.inputSize = 2048;
    rank0.outputAddr = 4096;
    rank0.outputSize = 1024;
    ASSERT_EQ(
        StorageManager::GetInstance().LoadHcclVmSynthesisData(0, kTestCommId, "", 0, 0, 0, rank0, channels, halfRTT),
        HcclResult::HCCL_SUCCESS);

    sim::operation::OpMemInfoTab rank1{};
    rank1.inputAddr = 8192;
    rank1.inputSize = 4096;
    ASSERT_EQ(
        StorageManager::GetInstance().LoadHcclVmSynthesisData(1, kTestCommId, "", 0, 0, 1, rank1, channels, halfRTT),
        HcclResult::HCCL_SUCCESS);

    uint64_t size0 = StorageManager::GetInstance().GetBlockSize("", 0, 0, 0, BufferType::INPUT);
    EXPECT_EQ(size0, 2048);

    uint64_t size1 = StorageManager::GetInstance().GetBlockSize("", 0, 0, 0, BufferType::OUTPUT);
    EXPECT_EQ(size1, 1024);

    uint64_t size2 = StorageManager::GetInstance().GetBlockSize("", 0, 0, 1, BufferType::INPUT);
    EXPECT_EQ(size2, 4096);
}

TEST_F(StorageManagerCheckerTest, GetSlice_WithMemLayout_ByRank)
{
    sim::operation::OpMemInfoTab memInfo{};
    memInfo.inputAddr = 1024;
    memInfo.inputSize = 2048;
    memInfo.outputAddr = 8192;
    memInfo.outputSize = 4096;
    std::vector<sim::operation::CcuChannelTab> channels;
    std::vector<sim::operation::HalfRTTTab> halfRTT;
    ASSERT_EQ(
        StorageManager::GetInstance().LoadHcclVmSynthesisData(0, kTestCommId, "", 0, 0, 0, memInfo, channels, halfRTT),
        HcclResult::HCCL_SUCCESS);

    DataSlice slice;
    uint32_t rank = 999;
    HcclResult ret = StorageManager::GetInstance().GetSlice("", 0, 0, 1024, 100, slice, &rank);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(rank, 0);
    EXPECT_EQ(slice.GetSize(), 100);
    EXPECT_EQ(slice.GetType(), BufferType::INPUT);
    EXPECT_EQ(slice.GetOffset(), 0);

    DataSlice slice2;
    uint32_t rank2 = 999;
    ret = StorageManager::GetInstance().GetSlice("", 0, 0, 2048, 100, slice2, &rank2);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(rank2, 0);
    EXPECT_EQ(slice2.GetType(), BufferType::INPUT);
    EXPECT_EQ(slice2.GetOffset(), 1024);

    DataSlice slice3;
    uint32_t rank3 = 999;
    ret = StorageManager::GetInstance().GetSlice("", 0, 0, 8192, 100, slice3, &rank3);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(rank3, 0);
    EXPECT_EQ(slice3.GetType(), BufferType::OUTPUT);
    EXPECT_EQ(slice3.GetOffset(), 0);
}

TEST_F(StorageManagerCheckerTest, GetSlice_AddrNotInAnyBlock)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_noblock";
    std::string dataDir = CreateTestDataDir(tmpDir, dataId);

    std::vector<std::string> lines;
    lines.push_back(R"({"total_count":1})");
    lines.push_back("[0, 1024, 2048, 0]");

    std::string filePath = dataDir + "/" + dataId + "_mem_layout.jsonl.gz";

    char savedCwd[4096];
    getcwd(savedCwd, sizeof(savedCwd));
    chdir(tmpDir.c_str());

    StorageManager::GetInstance().SetDataId(dataId);

    DataSlice slice;
    uint32_t rank = 999;
    HcclResult ret = StorageManager::GetInstance().GetSlice("", 0, 0, 0, 100, slice, &rank);
    EXPECT_NE(ret, HcclResult::HCCL_SUCCESS);

    DataSlice slice2;
    uint32_t rank2 = 999;
    ret = StorageManager::GetInstance().GetSlice("", 0, 0, 5000, 100, slice2, &rank2);
    EXPECT_NE(ret, HcclResult::HCCL_SUCCESS);

    chdir(savedCwd);
    RemoveDir(tmpDir);
}

TEST_F(StorageManagerCheckerTest, GetSlice_WithMemLayout)
{
    std::vector<sim::operation::CcuChannelTab> channels;
    std::vector<sim::operation::HalfRTTTab> halfRTT;
    sim::operation::OpMemInfoTab rank0{};
    rank0.inputAddr = 1024;
    rank0.inputSize = 2048;
    ASSERT_EQ(
        StorageManager::GetInstance().LoadHcclVmSynthesisData(0, kTestCommId, "", 0, 0, 0, rank0, channels, halfRTT),
        HcclResult::HCCL_SUCCESS);

    sim::operation::OpMemInfoTab rank1{};
    rank1.outputAddr = 8192;
    rank1.outputSize = 4096;
    ASSERT_EQ(
        StorageManager::GetInstance().LoadHcclVmSynthesisData(1, kTestCommId, "", 0, 0, 1, rank1, channels, halfRTT),
        HcclResult::HCCL_SUCCESS);

    DataSlice slice;
    uint32_t rank = 99;
    HcclResult sliceRet = StorageManager::GetInstance().GetSlice("", 0, 0, 1024, 100, slice, &rank);
    EXPECT_EQ(sliceRet, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(rank, 0);
    EXPECT_EQ(slice.GetType(), BufferType::INPUT);
    EXPECT_EQ(slice.GetOffset(), 0);

    DataSlice slice2;
    uint32_t rank2 = 99;
    HcclResult sliceRet2 = StorageManager::GetInstance().GetSlice("", 0, 0, 8192, 100, slice2, &rank2);
    EXPECT_EQ(sliceRet2, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(rank2, 1);
    EXPECT_EQ(slice2.GetType(), BufferType::OUTPUT);
}

TEST_F(StorageManagerCheckerTest, GetSlice_ExpectedDeviceFastPath)
{
    std::vector<sim::operation::CcuChannelTab> channels;
    std::vector<sim::operation::HalfRTTTab> halfRTT;
    // device 0: INPUT [1024, 3072)
    sim::operation::OpMemInfoTab rank0{};
    rank0.inputAddr = 1024;
    rank0.inputSize = 2048;
    ASSERT_EQ(
        StorageManager::GetInstance().LoadHcclVmSynthesisData(0, kTestCommId, "", 0, 0, 0, rank0, channels, halfRTT),
        HcclResult::HCCL_SUCCESS);
    // device 1: OUTPUT [8192, 12288)
    sim::operation::OpMemInfoTab rank1{};
    rank1.outputAddr = 8192;
    rank1.outputSize = 4096;
    ASSERT_EQ(
        StorageManager::GetInstance().LoadHcclVmSynthesisData(1, kTestCommId, "", 0, 0, 1, rank1, channels, halfRTT),
        HcclResult::HCCL_SUCCESS);

    // (a) expected 命中：device 0 上的地址
    DataSlice slice;
    uint32_t rank = 99;
    EXPECT_EQ(StorageManager::GetInstance().GetSlice("", 0, 0, 1024, 100, slice, &rank, 0), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(rank, 0);
    EXPECT_EQ(slice.GetType(), BufferType::INPUT);

    // (b) expected 未命中：device 1 上不存在 1024 → 直接报错（不再回退扫描其它设备）
    DataSlice missSlice;
    uint32_t missRank = 99;
    EXPECT_NE(
        StorageManager::GetInstance().GetSlice("", 0, 0, 1024, 100, missSlice, &missRank, 1), HcclResult::HCCL_SUCCESS);

    // (c) expected = INVALID_DEVICE_ID → 回退全设备扫描，仍能命中 device 1 的地址
    DataSlice fallbackSlice;
    uint32_t fallbackRank = 99;
    EXPECT_EQ(
        StorageManager::GetInstance().GetSlice("", 0, 0, 8192, 100, fallbackSlice, &fallbackRank, INVALID_DEVICE_ID),
        HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(fallbackRank, 1);
    EXPECT_EQ(fallbackSlice.GetType(), BufferType::OUTPUT);
}

TEST_F(StorageManagerCheckerTest, GetSlice_AddrNotFound)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_slice_miss";
    std::string dataDir = CreateTestDataDir(tmpDir, dataId);

    std::vector<std::string> lines;
    lines.push_back(R"({"total_count":1})");
    lines.push_back("[0, 1024, 2048, 0]");

    std::string filePath = dataDir + "/" + dataId + "_mem_layout.jsonl.gz";

    char savedCwd[4096];
    getcwd(savedCwd, sizeof(savedCwd));
    chdir(tmpDir.c_str());

    StorageManager::GetInstance().SetDataId(dataId);

    DataSlice slice;
    uint32_t rank = 0;
    HcclResult sliceRet = StorageManager::GetInstance().GetSlice("", 0, 0, 0, 100, slice, &rank);
    EXPECT_NE(sliceRet, HcclResult::HCCL_SUCCESS);

    chdir(savedCwd);
    RemoveDir(tmpDir);
}

TEST_F(StorageManagerCheckerTest, LoadHcclVmSynthesisData_LoadsMemoryAndChannels)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_syndata";
    std::string dataDir = CreateTestDataDir(tmpDir, dataId);

    HcclVmSynData synData{};
    synData.header.magic = HCCLVM_SYN_FILE_MAGIC;
    synData.header.version = 1;
    synData.header.header_size = sizeof(FileHeader);
    synData.header.count = 1;
    synData.model_info.comm.src_rank = 0;
    synData.model_info.comm.dst_rank = 1;
    synData.model_info.comm.root = 0;
    synData.model_info.comm.rank_size = 4;
    synData.model_info.comm.chip_type = 0;
    synData.model_info.comm.op_type = 0;
    synData.model_info.comm.reduce_op = 0;
    synData.model_info.comm.data_type = 0;
    synData.model_info.comm.data_count = 1024;
    synData.model_info.comm.op_expansion_mode = 0;

    ChannelData chData{};
    chData.channelId = 0;
    chData.srcDieId = 0;
    chData.dstDieId = 0;
    chData.srcDeviceId = 0;
    chData.dstDeviceId = 1;
    chData.protocol = 0;
    chData.jettyNum = 0;
    synData.channel_info.count = 1;
    synData.channel_info.data.push_back(chData);

    MemLayoutData memData{};
    memData.rank_id = 0;
    memData.buffer_type = 0;
    memData.start_addr = 0x1000;
    memData.size = 0x2000;
    memData.global_offset = 0;
    synData.memory_info.count = 1;
    synData.memory_info.data.push_back(memData);

    std::string synFilePath = dataDir + "/" + dataId + "_hcclvm_syn_data.bin";
    WriteSynDataFile(synFilePath, synData);

    char savedCwd[4096];
    getcwd(savedCwd, sizeof(savedCwd));
    chdir(tmpDir.c_str());

    StorageManager::GetInstance().SetDataId(dataId);
    uint32_t rankId = 0;
    sim::operation::OpMemInfoTab memInfo{};
    memInfo.inputAddr = 0x1000;
    memInfo.inputSize = 0x2000;
    sim::operation::CcuChannelTab channel{};
    channel.channelId = 0;
    channel.srcDieId = 0;
    channel.dstDieId = 0;
    channel.srcDeviceId = 0;
    channel.dstDeviceId = 1;
    channel.srcRankId = 0;
    channel.dstRankId = 1;
    std::vector<sim::operation::CcuChannelTab> channels{channel};
    std::vector<sim::operation::HalfRTTTab> halfRTT;
    HcclResult ret = StorageManager::GetInstance().LoadHcclVmSynthesisData(
        0, kTestCommId, "", 0, 0, rankId, memInfo, channels, halfRTT);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    sim::operation::OpDetailTab detailTab{};
    detailTab.rankSize = 4;
    ::OpDetails detail{};
    StorageManager::GetInstance().Trans2CheckerParam(detailTab, detail);
    EXPECT_EQ(StorageManager::GetInstance().GetRankSize(), 4);

    uint64_t blockSize = StorageManager::GetInstance().GetBlockSize("", 0, 0, 0, BufferType::INPUT);
    EXPECT_EQ(blockSize, 0x2000);

    EXPECT_FALSE(g_allRankChannelInfo.empty());
    EXPECT_EQ(g_allRankChannelInfo[0][0][0].dstDeviceId, 1);

    chdir(savedCwd);
    RemoveDir(tmpDir);
}

TEST_F(StorageManagerCheckerTest, LoadHcclVmSynthesisData_EmptyPayload)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "nonexist_syn";
    CreateTestDataDir(tmpDir, dataId);

    char savedCwd[4096];
    getcwd(savedCwd, sizeof(savedCwd));
    chdir(tmpDir.c_str());

    StorageManager::GetInstance().SetDataId(dataId);
    uint32_t rankId = 0;
    sim::operation::OpMemInfoTab memInfo{};
    std::vector<sim::operation::CcuChannelTab> channels;
    std::vector<sim::operation::HalfRTTTab> halfRTT;
    HcclResult ret = StorageManager::GetInstance().LoadHcclVmSynthesisData(
        0, kTestCommId, "", 0, 0, rankId, memInfo, channels, halfRTT);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    chdir(savedCwd);
    RemoveDir(tmpDir);
}

TEST_F(StorageManagerCheckerTest, LoadHcclVmInstrData_WithFile)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_instr";
    std::string dataDir = CreateTestDataDir(tmpDir, dataId);

    HcclVmSynData synData{};
    synData.header.magic = HCCLVM_SYN_FILE_MAGIC;
    synData.header.version = 1;
    synData.header.header_size = sizeof(FileHeader);
    synData.model_info.comm.op_expansion_mode = 0;

    std::string synFilePath = dataDir + "/" + dataId + "_hcclvm_syn_data.bin";
    WriteSynDataFile(synFilePath, synData);

    HcclVmInstrData instrData{};
    instrData.header.magic = HCCLVM_INSTR_FILE_MAGIC;
    instrData.header.version = 1;
    instrData.header.header_size = sizeof(FileHeader);
    instrData.header.count = 0;

    std::string instrFilePath = dataDir + "/" + dataId + "_hcclvm_instr_data.bin";
    WriteInstrDataFile(instrFilePath, instrData);

    char savedCwd[4096];
    getcwd(savedCwd, sizeof(savedCwd));
    chdir(tmpDir.c_str());

    StorageManager::GetInstance().SetDataId(dataId);
    uint32_t rankId = 0;
    sim::operation::OpMemInfoTab memInfo{};
    std::vector<sim::operation::CcuChannelTab> channels;
    std::vector<sim::operation::HalfRTTTab> halfRTT;
    HcclResult loadSynRet = StorageManager::GetInstance().LoadHcclVmSynthesisData(
        0, kTestCommId, "", 0, 0, rankId, memInfo, channels, halfRTT);
    EXPECT_EQ(loadSynRet, HcclResult::HCCL_SUCCESS);

    std::vector<sim::operation::CcuInstrResTab> instrRes;
    HcclResult ret = StorageManager::GetInstance().LoadHcclVmInstrData(instrRes);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    HcclVmInstrData loadedInstr = StorageManager::GetInstance().GetHvmInstrData();
    EXPECT_EQ(loadedInstr.instr_data.size(), 0);

    chdir(savedCwd);
    RemoveDir(tmpDir);
}

TEST_F(StorageManagerCheckerTest, LoadHcclVmInstrData_NonCcuMode)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_instr_nonccu";
    std::string dataDir = CreateTestDataDir(tmpDir, dataId);

    HcclVmSynData synData{};
    synData.header.magic = HCCLVM_SYN_FILE_MAGIC;
    synData.header.version = 1;
    synData.header.header_size = sizeof(FileHeader);
    synData.model_info.comm.op_expansion_mode = 1;

    std::string synFilePath = dataDir + "/" + dataId + "_hcclvm_syn_data.bin";
    WriteSynDataFile(synFilePath, synData);

    HcclVmInstrData instrData{};
    instrData.header.magic = HCCLVM_INSTR_FILE_MAGIC;
    instrData.header.version = 1;
    instrData.header.header_size = sizeof(FileHeader);
    instrData.header.count = 0;

    std::string instrFilePath = dataDir + "/" + dataId + "_hcclvm_instr_data.bin";
    WriteInstrDataFile(instrFilePath, instrData);

    char savedCwd[4096];
    getcwd(savedCwd, sizeof(savedCwd));
    chdir(tmpDir.c_str());

    StorageManager::GetInstance().SetDataId(dataId);
    uint32_t rankId = 0;
    sim::operation::OpMemInfoTab memInfo{};
    std::vector<sim::operation::CcuChannelTab> channels;
    std::vector<sim::operation::HalfRTTTab> halfRTT;
    HcclResult loadSynRet = StorageManager::GetInstance().LoadHcclVmSynthesisData(
        0, kTestCommId, "", 0, 0, rankId, memInfo, channels, halfRTT);
    EXPECT_EQ(loadSynRet, HcclResult::HCCL_SUCCESS);

    std::vector<sim::operation::CcuInstrResTab> instrRes;
    HcclResult ret = StorageManager::GetInstance().LoadHcclVmInstrData(instrRes);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    chdir(savedCwd);
    RemoveDir(tmpDir);
}

TEST_F(StorageManagerCheckerTest, LoadHcclVmInstrData_EmptyPayload)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "nonexist_instr";
    std::string dataDir = CreateTestDataDir(tmpDir, dataId);

    HcclVmSynData synData{};
    synData.header.magic = HCCLVM_SYN_FILE_MAGIC;
    synData.header.version = 1;
    synData.header.header_size = sizeof(FileHeader);
    synData.model_info.comm.op_expansion_mode = 0;

    std::string synFilePath = dataDir + "/" + dataId + "_hcclvm_syn_data.bin";
    WriteSynDataFile(synFilePath, synData);

    char savedCwd[4096];
    getcwd(savedCwd, sizeof(savedCwd));
    chdir(tmpDir.c_str());

    StorageManager::GetInstance().SetDataId(dataId);
    uint32_t rankId = 0;
    sim::operation::OpMemInfoTab memInfo{};
    std::vector<sim::operation::CcuChannelTab> channels;
    std::vector<sim::operation::HalfRTTTab> halfRTT;
    HcclResult loadSynRet = StorageManager::GetInstance().LoadHcclVmSynthesisData(
        0, kTestCommId, "", 0, 0, rankId, memInfo, channels, halfRTT);
    EXPECT_EQ(loadSynRet, HcclResult::HCCL_SUCCESS);

    std::vector<sim::operation::CcuInstrResTab> instrRes;
    HcclResult ret = StorageManager::GetInstance().LoadHcclVmInstrData(instrRes);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    chdir(savedCwd);
    RemoveDir(tmpDir);
}

TEST_F(StorageManagerCheckerTest, LoadHcclVmTaskMetaData_WithFile)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_taskmeta";
    std::string dataDir = CreateTestDataDir(tmpDir, dataId);

    HcclVmTaskMetaData taskMeta{};
    taskMeta.header.magic = HCCLVM_TASK_FILE_MAGIC;
    taskMeta.header.version = 1;
    taskMeta.header.header_size = sizeof(FileHeader);
    taskMeta.header.count = 0;

    std::string taskFilePath = dataDir + "/" + dataId + "_hcclvm_task_data.bin";
    WriteTaskMetaDataFile(taskFilePath, taskMeta);

    char savedCwd[4096];
    getcwd(savedCwd, sizeof(savedCwd));
    chdir(tmpDir.c_str());

    StorageManager::GetInstance().SetDataId(dataId);
    std::vector<std::vector<sim::operation::OpTaskTab>> allTasks;
    HcclResult ret = StorageManager::GetInstance().LoadHcclVmTaskMetaData(allTasks);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    HcclVmTaskMetaData loaded = StorageManager::GetInstance().GetHvmTaskMetaData();
    EXPECT_EQ(loaded.task_meta.size(), 0);

    chdir(savedCwd);
    RemoveDir(tmpDir);
}

TEST_F(StorageManagerCheckerTest, LoadHcclVmTaskMetaData_WithTasks)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_tasks";
    std::string dataDir = CreateTestDataDir(tmpDir, dataId);

    HcclTaskMetaData task1{};
    task1.taskType = HccLTaskMetaType::MEM_CPY;
    task1.commId = 0;
    task1.rankId = 0;
    task1.streamId = 0;
    task1.taskData.transMem.srcDeviceId = 0;
    task1.taskData.transMem.srcOffset = 0;
    task1.taskData.transMem.dstDeviceId = 1;
    task1.taskData.transMem.dstOffset = 0;
    task1.taskData.transMem.len = 1024;
    task1.taskData.transMem.protocol = 0;

    HcclTaskMetaData task2{};
    task2.taskType = HccLTaskMetaType::REDUCE;
    task2.commId = 0;
    task2.rankId = 1;
    task2.streamId = 0;
    task2.taskData.reduce.srcDeviceId = 0;
    task2.taskData.reduce.srcOffset = 0;
    task2.taskData.reduce.dstDeviceId = 1;
    task2.taskData.reduce.dstOffset = 0;
    task2.taskData.reduce.dataCount = 512;
    task2.taskData.reduce.dataType = 0;
    task2.taskData.reduce.reduceOp = 0;
    task2.taskData.reduce.protocol = 0;

    HcclVmTaskMetaData taskMeta{};
    taskMeta.header.magic = HCCLVM_TASK_FILE_MAGIC;
    taskMeta.header.version = 1;
    taskMeta.header.header_size = sizeof(FileHeader);
    taskMeta.header.count = 2;
    taskMeta.task_meta.push_back(task1);
    taskMeta.task_meta.push_back(task2);

    std::string taskFilePath = dataDir + "/" + dataId + "_hcclvm_task_data.bin";
    WriteTaskMetaDataFile(taskFilePath, taskMeta);

    char savedCwd[4096];
    getcwd(savedCwd, sizeof(savedCwd));
    chdir(tmpDir.c_str());

    StorageManager::GetInstance().SetDataId(dataId);
    std::vector<std::vector<sim::operation::OpTaskTab>> allTasks{{EncodeTask(task1), EncodeTask(task2)}};
    HcclResult ret = StorageManager::GetInstance().LoadHcclVmTaskMetaData(allTasks);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    HcclVmTaskMetaData loaded = StorageManager::GetInstance().GetHvmTaskMetaData();
    EXPECT_EQ(loaded.task_meta.size(), 2);
    EXPECT_EQ(loaded.task_meta[0].taskType, HccLTaskMetaType::MEM_CPY);
    EXPECT_EQ(loaded.task_meta[1].taskType, HccLTaskMetaType::REDUCE);

    chdir(savedCwd);
    RemoveDir(tmpDir);
}

TEST_F(StorageManagerCheckerTest, LoadHcclVmTaskMetaData_NotifyTasks)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_notify";
    std::string dataDir = CreateTestDataDir(tmpDir, dataId);

    HcclTaskMetaData notifyRecord{};
    notifyRecord.taskType = HccLTaskMetaType::NOTIFY_RECORD;
    notifyRecord.commId = 0;
    notifyRecord.rankId = 0;
    notifyRecord.deviceId = 0;
    notifyRecord.streamId = 0;
    notifyRecord.taskData.notify.srcDeviceId = 0;
    notifyRecord.taskData.notify.notifyId = 100;
    notifyRecord.taskData.notify.dstDeviceId = 0;
    notifyRecord.taskData.notify.notifyCount = 1;
    notifyRecord.taskData.notify.protocol = 0;

    HcclTaskMetaData notifyWait{};
    notifyWait.taskType = HccLTaskMetaType::NOTIFY_WAIT;
    notifyWait.commId = 0;
    notifyWait.rankId = 1;
    notifyWait.deviceId = 1;
    notifyWait.streamId = 0;
    notifyWait.taskData.notify.srcDeviceId = 0;
    notifyWait.taskData.notify.notifyId = 100;
    notifyWait.taskData.notify.dstDeviceId = 0;
    notifyWait.taskData.notify.notifyCount = 1;
    notifyWait.taskData.notify.protocol = 0;

    HcclVmTaskMetaData taskMeta{};
    taskMeta.header.magic = HCCLVM_TASK_FILE_MAGIC;
    taskMeta.header.version = 1;
    taskMeta.header.header_size = sizeof(FileHeader);
    taskMeta.header.count = 2;
    taskMeta.task_meta.push_back(notifyRecord);
    taskMeta.task_meta.push_back(notifyWait);

    std::string taskFilePath = dataDir + "/" + dataId + "_hcclvm_task_data.bin";
    WriteTaskMetaDataFile(taskFilePath, taskMeta);

    char savedCwd[4096];
    getcwd(savedCwd, sizeof(savedCwd));
    chdir(tmpDir.c_str());

    StorageManager::GetInstance().SetDataId(dataId);
    std::vector<std::vector<sim::operation::OpTaskTab>> allTasks{{EncodeTask(notifyRecord), EncodeTask(notifyWait)}};
    HcclResult ret = StorageManager::GetInstance().LoadHcclVmTaskMetaData(allTasks);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    HcclVmTaskMetaData loaded = StorageManager::GetInstance().GetHvmTaskMetaData();
    EXPECT_EQ(loaded.task_meta.size(), 2);
    EXPECT_EQ(loaded.task_meta[0].taskData.notify.dstDeviceId, 1);
    EXPECT_EQ(loaded.task_meta[1].taskData.notify.srcDeviceId, 0);

    chdir(savedCwd);
    RemoveDir(tmpDir);
}

TEST_F(StorageManagerCheckerTest, LoadHcclVmTaskMetaData_EmptyPayload)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "nonexist_task";
    CreateTestDataDir(tmpDir, dataId);

    char savedCwd[4096];
    getcwd(savedCwd, sizeof(savedCwd));
    chdir(tmpDir.c_str());

    StorageManager::GetInstance().SetDataId(dataId);
    std::vector<std::vector<sim::operation::OpTaskTab>> allTasks;
    HcclResult ret = StorageManager::GetInstance().LoadHcclVmTaskMetaData(allTasks);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    chdir(savedCwd);
    RemoveDir(tmpDir);
}

TEST_F(StorageManagerCheckerTest, FileHeader_StructLayout)
{
    FileHeader header{};
    header.magic = HCCLVM_TASK_FILE_MAGIC;
    header.version = 1;
    header.header_size = sizeof(FileHeader);
    header.flags = 0;
    header.count = 10;
    header.checksum = 0;
    EXPECT_EQ(header.magic, 0x48565444u);
    EXPECT_EQ(header.count, 10u);
}

TEST_F(StorageManagerCheckerTest, FileHeader_MagicNumbers)
{
    EXPECT_EQ(HCCLVM_TASK_FILE_MAGIC, 0x48565444u);
    EXPECT_EQ(HCCLVM_SYN_FILE_MAGIC, 0x48564D44u);
    EXPECT_EQ(HCCLVM_INSTR_FILE_MAGIC, 0x434D4349u);
    EXPECT_EQ(HCCLVM_FLAG_FILE_MAGIC, 0x464C4147u);
}

TEST_F(StorageManagerCheckerTest, MemBlock_InStorageManager)
{
    HcclSim::MemBlock sm_block{};
    EXPECT_EQ(sm_block.startAddr, 0u);
    EXPECT_EQ(sm_block.size, 0u);
    EXPECT_EQ(sm_block.globalOffset, 0u);
}

TEST_F(StorageManagerCheckerTest, RemoteDieInfo_StructDefault)
{
    RemoteDieInfo info{};
    EXPECT_EQ(info.dstDeviceId, 0u);
    EXPECT_EQ(info.remoteDieId, 0u);
}

TEST_F(StorageManagerCheckerTest, DataSlice_SetMethods)
{
    DataSlice slice;
    slice.SetBufferType(BufferType::CCL);
    slice.SetOffset(0x5000);
    slice.SetSize(4096);
    EXPECT_EQ(slice.GetType(), BufferType::CCL);
    EXPECT_EQ(slice.GetOffset(), 0x5000);
    EXPECT_EQ(slice.GetSize(), 4096);
}

TEST_F(StorageManagerCheckerTest, BufferType_EnumValues)
{
    EXPECT_EQ(static_cast<int>(BufferType::INPUT), 0);
    EXPECT_EQ(static_cast<int>(BufferType::OUTPUT), 1);
    EXPECT_EQ(static_cast<int>(BufferType::CCL), 2);
    EXPECT_EQ(static_cast<int>(BufferType::SCRATCH), 3);
    EXPECT_EQ(static_cast<int>(BufferType::INPUT_AIV), 4);
    EXPECT_EQ(static_cast<int>(BufferType::OUTPUT_AIV), 5);
    EXPECT_EQ(static_cast<int>(BufferType::AIV_COMMINFO), 6);
    EXPECT_EQ(static_cast<int>(BufferType::USERBUF_AIV), 7);
    EXPECT_EQ(static_cast<int>(BufferType::MS), 8);
    EXPECT_EQ(static_cast<int>(BufferType::RESERVED), 9);
}

TEST_F(StorageManagerCheckerTest, HccLTaskMetaType_EnumValues)
{
    EXPECT_EQ(static_cast<char>(HccLTaskMetaType::NOTIFY_WAIT), 0);
    EXPECT_EQ(static_cast<char>(HccLTaskMetaType::NOTIFY_RECORD), 1);
    EXPECT_EQ(static_cast<char>(HccLTaskMetaType::REDUCE), 2);
    EXPECT_EQ(static_cast<char>(HccLTaskMetaType::MEM_CPY), 3);
    EXPECT_EQ(static_cast<char>(HccLTaskMetaType::CCU_GRAPH), 4);
    EXPECT_EQ(static_cast<char>(HccLTaskMetaType::AIV_GRAPH), 5);
}

TEST_F(StorageManagerCheckerTest, ChannelInfoInner_StructLayout)
{
    ChannelInfoInner chInfo{};
    chInfo.count = 1;
    ChannelData chData{};
    chData.channelId = 1;
    chData.srcDieId = 0;
    chData.dstDieId = 1;
    chData.srcDeviceId = 0;
    chData.dstDeviceId = 2;
    chInfo.data.push_back(chData);
    EXPECT_EQ(chInfo.count, 1u);
    EXPECT_EQ(chInfo.data[0].channelId, 1);
    EXPECT_EQ(chInfo.data[0].srcDeviceId, 0);
    EXPECT_EQ(chInfo.data[0].dstDeviceId, 2);
}

TEST_F(StorageManagerCheckerTest, LoadHcclVmSynthesisData_MultipleChannelsAndMemLayout)
{
    sim::operation::CcuChannelTab ch1{};
    ch1.channelId = 0;
    ch1.srcDieId = 0;
    ch1.dstDieId = 0;
    ch1.srcDeviceId = 0;
    ch1.dstDeviceId = 1;
    ch1.srcRankId = 0;
    ch1.dstRankId = 1;
    ch1.protocol = 0;
    ch1.jettyNum = 0;

    sim::operation::CcuChannelTab ch2{};
    ch2.channelId = 1;
    ch2.srcDieId = 0;
    ch2.dstDieId = 0;
    ch2.srcDeviceId = 1;
    ch2.dstDeviceId = 0;
    ch2.srcRankId = 1;
    ch2.dstRankId = 0;
    ch2.protocol = 0;
    ch2.jettyNum = 0;

    std::vector<sim::operation::CcuChannelTab> channels{ch1, ch2};
    std::vector<sim::operation::HalfRTTTab> halfRTT;
    sim::operation::OpMemInfoTab mem0{};
    mem0.inputAddr = 0x1000;
    mem0.inputSize = 0x2000;
    ASSERT_EQ(
        StorageManager::GetInstance().LoadHcclVmSynthesisData(0, kTestCommId, "", 0, 0, 0, mem0, channels, halfRTT),
        HcclResult::HCCL_SUCCESS);

    sim::operation::OpMemInfoTab mem1{};
    mem1.outputAddr = 0x3000;
    mem1.outputSize = 0x4000;
    std::vector<sim::operation::CcuChannelTab> noChannels;
    ASSERT_EQ(
        StorageManager::GetInstance().LoadHcclVmSynthesisData(1, kTestCommId, "", 0, 0, 1, mem1, noChannels, halfRTT),
        HcclResult::HCCL_SUCCESS);

    EXPECT_EQ(g_allRankChannelInfo.size(), 2u);
    EXPECT_EQ(g_allRankChannelInfo[0][0].size(), 1u);
    EXPECT_EQ(g_allRankChannelInfo[1][0].size(), 1u);

    uint64_t size0 = StorageManager::GetInstance().GetBlockSize("", 0, 0, 0, BufferType::INPUT);
    EXPECT_EQ(size0, 0x2000);
    uint64_t size1 = StorageManager::GetInstance().GetBlockSize("", 0, 0, 1, BufferType::OUTPUT);
    EXPECT_EQ(size1, 0x4000);
}

TEST_F(StorageManagerCheckerTest, Trans2CheckerParam_All2AllDataDes)
{
    sim::operation::OpDetailTab detailTab{};
    detailTab.rankId = 0;
    detailTab.rankSize = 2;
    detailTab.opExtInfo = BuildMatrixPayload({10, 20, 30, 40});
    ::OpDetails detail{};
    detail.opType = static_cast<uint16_t>(HcclCMDType::HCCL_CMD_ALLTOALLV);
    detail.opV2.sendDataType = 1;
    detail.opV2.recvDataType = 2;
    detail.opV2.sendCount = 100;
    detail.opV2.recvCount = 200;
    HcclResult transRet = StorageManager::GetInstance().Trans2CheckerParam(detailTab, detail);
    EXPECT_EQ(transRet, HcclResult::HCCL_SUCCESS);

    CheckerParam param = StorageManager::GetInstance().GetCheckerParam();
    EXPECT_EQ(param.all2AllDataDes.sendType, 1);
    EXPECT_EQ(param.all2AllDataDes.recvType, 2);
    EXPECT_EQ(param.all2AllDataDes.sendCount, 100);
    EXPECT_EQ(param.all2AllDataDes.recvCount, 200);
    EXPECT_EQ(param.all2AllDataDes.count, 4u);
    EXPECT_EQ(param.all2AllDataDes.sendCountMatrix.size(), 4u);
    EXPECT_EQ(param.all2AllDataDes.sendCountMatrix[0], 10);
    EXPECT_EQ(param.all2AllDataDes.sendCountMatrix[3], 40);
}

TEST_F(StorageManagerCheckerTest, DevType_EnumValues)
{
    EXPECT_GE(static_cast<int>(DevType::DEV_TYPE_COUNT), 0);
    EXPECT_GE(static_cast<int>(DevType::DEV_TYPE_910_93), 0);
    EXPECT_GE(static_cast<int>(DevType::DEV_TYPE_910B), 0);
}

TEST_F(StorageManagerCheckerTest, LoadHcclVmInstrData_CcuModeWithData)
{
    sim::operation::CcuInstrResTab instr{};
    instr.dieId = 0;
    instr.instrCount = 2;
    std::vector<sim::operation::CcuInstrResTab> instrRes{instr};
    HcclResult ret = StorageManager::GetInstance().LoadHcclVmInstrData(instrRes);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    HcclVmInstrData loaded = StorageManager::GetInstance().GetHvmInstrData();
    EXPECT_EQ(loaded.instr_data.size(), 1);
    EXPECT_EQ(loaded.instr_data[0].desc.rank_id, 0);
    EXPECT_EQ(loaded.instr_data[0].data.size(), 2);
}

TEST_F(StorageManagerCheckerTest, LoadHcclVmInstrData_ReplacesPreviousSnapshot)
{
    sim::operation::CcuInstrResTab instr{};
    instr.deviceId = 7;
    instr.dieId = 1;
    instr.instrCount = 1;
    std::vector<sim::operation::CcuInstrResTab> instrRes{instr};

    ASSERT_EQ(StorageManager::GetInstance().LoadHcclVmInstrData(instrRes), HcclResult::HCCL_SUCCESS);
    ASSERT_EQ(StorageManager::GetInstance().GetHvmInstrData().instr_data.size(), 1);
    ASSERT_EQ(StorageManager::GetInstance().LoadHcclVmInstrData(instrRes), HcclResult::HCCL_SUCCESS);

    const HcclVmInstrData loaded = StorageManager::GetInstance().GetHvmInstrData();
    ASSERT_EQ(loaded.instr_data.size(), 1);
    EXPECT_EQ(loaded.instr_data.front().desc.rank_id, 7);
    EXPECT_EQ(loaded.instr_data.front().desc.die_id, 1);
}

TEST_F(StorageManagerCheckerTest, LoadHcclVmTaskMetaData_CcuTask)
{
    HcclTaskMetaData ccuTask{};
    ccuTask.taskType = HccLTaskMetaType::CCU_GRAPH;
    ccuTask.commId = 0;
    ccuTask.rankId = 0;
    ccuTask.streamId = 0;
    ccuTask.taskData.ccu.dieId = 1;
    ccuTask.taskData.ccu.instCnt = 10;
    ccuTask.taskData.ccu.argSize = 2;

    std::vector<std::vector<sim::operation::OpTaskTab>> allTasks{{EncodeTask(ccuTask)}};
    HcclResult ret = StorageManager::GetInstance().LoadHcclVmTaskMetaData(allTasks);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    HcclVmTaskMetaData loaded = StorageManager::GetInstance().GetHvmTaskMetaData();
    EXPECT_EQ(loaded.task_meta.size(), 1);
    EXPECT_EQ(loaded.task_meta[0].taskType, HccLTaskMetaType::CCU_GRAPH);
}
