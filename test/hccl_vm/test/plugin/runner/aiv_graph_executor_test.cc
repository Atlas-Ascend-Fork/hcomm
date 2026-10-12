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
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann_json/json.hpp>

#include "aiv_graph_executor.h"
#include "aiv_task.h"
#include "aiv_task_snapshot_loader.h"
#include "sim_common_defs.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

class AivGraphExecutorTest : public testing::Test {
protected:
    void SetUp() override
    {
        testDir_ = fs::temp_directory_path() / ("aiv_exec_test_" + std::to_string(::getpid()));
        fs::create_directories(testDir_ / "data");
        setenv("HCCL_VM_INSTALL_ROOT", testDir_.c_str(), 1);
    }

    void TearDown() override
    {
        unsetenv("HCCL_VM_INSTALL_ROOT");
        fs::remove_all(testDir_);
    }

    std::string WriteTaskFile(uint32_t deviceId, uint32_t launchIndex, const json& content)
    {
        std::string fileName
            = "hcclvm_aiv_device" + std::to_string(deviceId) + "_launch" + std::to_string(launchIndex) + "_task.json";
        fs::path filePath = testDir_ / "data" / fileName;
        std::ofstream ofs(filePath);
        ofs << content.dump();
        return filePath.string();
    }

    static json MakeDataSlice(uint32_t bufferType, uint64_t offset, uint64_t size)
    {
        const uint64_t virtualBase = bufferType == 0 ? 0x100000 :
                                     bufferType == 1 ? 0x200000 :
                                     bufferType == 2 ? 0x300000 :
                                     bufferType == 3 ? 0x110000 :
                                                       0;
        return {
            {"bufferType", bufferType},
            {"deviceId", 0},
            {"offset", offset},
            {"virtualAddr", virtualBase + offset},
            {"size", size}};
    }

    static json MakeMemCopyPayload(const json& src, const json& dst) { return {{"src", src}, {"dst", dst}}; }

    static json MakeReducePayload(const json& src, const json& dst, uint32_t dataType = 0, uint32_t reduceOp = 0)
    {
        return {{"src", src}, {"dst", dst}, {"dataType", dataType}, {"reduceOp", reduceOp}};
    }

    static json MakeSetFlagPayload(uint32_t srcPipe, uint32_t dstPipe, int32_t eventId)
    {
        return {{"srcPipe", srcPipe}, {"dstPipe", dstPipe}, {"eventId", eventId}};
    }

    static json MakeWaitFlagPayload(uint32_t srcPipe, uint32_t dstPipe, int32_t eventId)
    {
        return {{"srcPipe", srcPipe}, {"dstPipe", dstPipe}, {"eventId", eventId}};
    }

    static json MakePipeBarrierPayload(uint32_t pipeType, const std::vector<uint32_t>& barrierGroupTaskIds)
    {
        return {{"pipeType", pipeType}, {"barrierGroupTaskIds", barrierGroupTaskIds}};
    }

    static json MakeTaskJson(
        uint32_t taskType, uint32_t taskId, uint32_t rankId, uint32_t blockId, uint32_t curPipe, const json& payload)
    {
        return {{"taskType", taskType}, {"taskId", taskId},   {"rankId", rankId},
                {"blockId", blockId},   {"curPipe", curPipe}, {"payload", payload}};
    }

    static json MakeBlockJson(
        uint32_t blockIdx, const json& scalarTasks = json::array(), const json& mte2Tasks = json::array(),
        const json& mte3Tasks = json::array())
    {
        return {
            {"blockIdx", blockIdx}, {"scalarTasks", scalarTasks}, {"mte2Tasks", mte2Tasks}, {"mte3Tasks", mte3Tasks}};
    }

    fs::path testDir_;
};

// ==================== Constructor / Destructor Tests ====================

TEST_F(AivGraphExecutorTest, Destructor_BeforeInit_NoThrow)
{
    EXPECT_NO_THROW({ AivGraphExecutor executor(0, 0); });
}

// ==================== Init Tests ====================

TEST_F(AivGraphExecutorTest, Init_LoadSnapshot_Success)
{
    auto ds = MakeDataSlice(3, 0, 128);
    auto task = MakeTaskJson(0, 1, 0, 0, 0, MakeMemCopyPayload(ds, ds));
    auto block = MakeBlockJson(0, json::array({task}));
    json content = {{"rank", 0}, {"rankSize", 2}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0, 0);
    EXPECT_TRUE(executor.Init());
}

TEST_F(AivGraphExecutorTest, Init_FileNotExist_ReturnsFalse)
{
    AivGraphExecutor executor(0, 99);
    EXPECT_FALSE(executor.Init());
}

TEST_F(AivGraphExecutorTest, Init_InvalidJson_ReturnsFalse)
{
    std::ofstream ofs(testDir_ / "data" / "hcclvm_aiv_device0_launch0_task.json");
    ofs << "not valid json";
    ofs.close();

    AivGraphExecutor executor(0, 0);
    EXPECT_FALSE(executor.Init());
}

TEST_F(AivGraphExecutorTest, Init_MissingAivCores_ReturnsFalse)
{
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0, 0);
    EXPECT_FALSE(executor.Init());
}

TEST_F(AivGraphExecutorTest, Init_EmptyAivCores_Success)
{
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array()}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0, 0);
    EXPECT_TRUE(executor.Init());
}

TEST_F(AivGraphExecutorTest, Init_Idempotent_SecondCallSucceeds)
{
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array()}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0, 0);
    ASSERT_TRUE(executor.Init());
    EXPECT_TRUE(executor.Init());
}

// ==================== Execute Tests ====================

TEST_F(AivGraphExecutorTest, Execute_EmptyTasks_Success)
{
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array()}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0, 0);
    ASSERT_TRUE(executor.Init());
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(AivGraphExecutorTest, Execute_SetFlagWaitFlag_Success)
{
    auto setFlag = MakeTaskJson(2, 1, 0, 0, 0, MakeSetFlagPayload(0, 1, 5));
    auto waitFlag = MakeTaskJson(3, 2, 0, 0, 0, MakeWaitFlagPayload(1, 2, 5));
    auto block = MakeBlockJson(0, json::array({setFlag, waitFlag}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0, 0);
    ASSERT_TRUE(executor.Init());
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(AivGraphExecutorTest, Execute_WaitFlagWithoutSet_Hold)
{
    auto waitFlag = MakeTaskJson(3, 1, 0, 0, 0, MakeWaitFlagPayload(1, 2, 5));
    auto block = MakeBlockJson(0, json::array({waitFlag}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0, 0);
    ASSERT_TRUE(executor.Init());
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_VRT_HOLD_CMD);
}

TEST_F(AivGraphExecutorTest, Execute_SetWaitFlagsAcrossBlocks_Success)
{
    auto setFlag = MakeTaskJson(2, 1, 0, 0, 0, MakeSetFlagPayload(0, 1, 3));
    auto waitFlag = MakeTaskJson(3, 2, 0, 0, 0, MakeWaitFlagPayload(1, 2, 3));
    auto block0 = MakeBlockJson(0, json::array({setFlag}));
    auto block1 = MakeBlockJson(1, json::array({waitFlag}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block0, block1})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0, 0);
    ASSERT_TRUE(executor.Init());
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(AivGraphExecutorTest, Execute_WaitFlagWrongEvent_Hold)
{
    auto setFlag = MakeTaskJson(2, 1, 0, 0, 0, MakeSetFlagPayload(0, 1, 1));
    auto waitFlag = MakeTaskJson(3, 2, 0, 0, 0, MakeWaitFlagPayload(1, 2, 2));
    auto block = MakeBlockJson(0, json::array({setFlag, waitFlag}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0, 0);
    ASSERT_TRUE(executor.Init());
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_VRT_HOLD_CMD);
}

TEST_F(AivGraphExecutorTest, Execute_PipeBarrierSingleTask_Success)
{
    auto barrier = MakeTaskJson(4, 5, 0, 0, 0, MakePipeBarrierPayload(0, json::array()));
    auto block = MakeBlockJson(0, json::array({barrier}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0, 0);
    ASSERT_TRUE(executor.Init());
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(AivGraphExecutorTest, Execute_PipeBarrierGroup_Success)
{
    // 同一队列每轮只推进一个task，互相等待的barrier需分布在不同pipe队列
    auto barrier1 = MakeTaskJson(4, 10, 0, 0, 0, MakePipeBarrierPayload(0, {11}));
    auto barrier2 = MakeTaskJson(4, 11, 0, 0, 0, MakePipeBarrierPayload(0, {10}));
    auto block = MakeBlockJson(0, json::array({barrier1}), json::array({barrier2}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0, 0);
    ASSERT_TRUE(executor.Init());
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(AivGraphExecutorTest, Execute_MemCopyLenMismatch_ReturnsError)
{
    auto srcDs = MakeDataSlice(3, 0, 128);
    auto dstDs = MakeDataSlice(3, 0, 256);
    auto task = MakeTaskJson(0, 1, 0, 0, 0, MakeMemCopyPayload(srcDs, dstDs));
    auto block = MakeBlockJson(0, json::array({task}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0, 0);
    ASSERT_TRUE(executor.Init());
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_VRT_ERROR_CMD);
}

TEST_F(AivGraphExecutorTest, Execute_MemCopyUnresolvableAddr_ReturnsInternal)
{
    auto srcDs = MakeDataSlice(3, 0, 128);
    auto dstDs = MakeDataSlice(3, 0, 128);
    dstDs["virtualAddr"] = 0x990000;
    auto task = MakeTaskJson(0, 1, 0, 0, 0, MakeMemCopyPayload(srcDs, dstDs));
    auto block = MakeBlockJson(0, json::array({task}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0, 0);
    ASSERT_TRUE(executor.Init());
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_E_INTERNAL);
}

TEST_F(AivGraphExecutorTest, Execute_ReduceLenMismatch_ReturnsError)
{
    auto srcDs = MakeDataSlice(0, 0, 128);
    auto dstDs = MakeDataSlice(1, 0, 256);
    auto task = MakeTaskJson(1, 1, 0, 0, 1, MakeReducePayload(srcDs, dstDs, 4, 0));
    auto block = MakeBlockJson(0, json::array(), json::array({task}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0, 0);
    ASSERT_TRUE(executor.Init());
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_VRT_ERROR_CMD);
}

TEST_F(AivGraphExecutorTest, Execute_ReduceUnresolvableAddr_ReturnsInternal)
{
    auto srcDs = MakeDataSlice(0, 0, 16);
    auto dstDs = MakeDataSlice(1, 0, 16);
    dstDs["virtualAddr"] = 0x990000;
    auto task = MakeTaskJson(1, 1, 0, 0, 1, MakeReducePayload(srcDs, dstDs, 4, 0));
    auto block = MakeBlockJson(0, json::array(), json::array({task}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0, 0);
    ASSERT_TRUE(executor.Init());
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_E_INTERNAL);
}

TEST_F(AivGraphExecutorTest, Execute_MixedFlagAndBarrierTasks_Success)
{
    auto setFlag = MakeTaskJson(2, 1, 0, 0, 0, MakeSetFlagPayload(0, 1, 0));
    auto waitFlag = MakeTaskJson(3, 2, 0, 0, 0, MakeWaitFlagPayload(1, 2, 0));
    auto barrier = MakeTaskJson(4, 3, 0, 0, 0, MakePipeBarrierPayload(0, json::array()));
    auto block = MakeBlockJson(0, json::array({setFlag, waitFlag, barrier}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0, 0);
    ASSERT_TRUE(executor.Init());
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(AivGraphExecutorTest, Execute_MultiBlockTasks_Success)
{
    auto task0 = MakeTaskJson(2, 1, 0, 0, 0, MakeSetFlagPayload(0, 1, 0));
    auto task1 = MakeTaskJson(3, 2, 0, 0, 0, MakeWaitFlagPayload(1, 2, 0));
    auto block0 = MakeBlockJson(0, json::array({task0}));
    auto block1 = MakeBlockJson(1, json::array({task1}));
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array({block0, block1})}};
    WriteTaskFile(0, 0, content);

    AivGraphExecutor executor(0, 0);
    ASSERT_TRUE(executor.Init());
    EXPECT_EQ(executor.Execute(), HcclSim::HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(AivGraphExecutorTest, Destructor_AfterInit_NoThrow)
{
    json content = {{"rank", 0}, {"rankSize", 1}, {"launchIndex", 0}, {"aivCores", json::array()}};
    WriteTaskFile(0, 0, content);
    auto* executor = new AivGraphExecutor(0, 0);
    ASSERT_TRUE(executor->Init());
    EXPECT_NO_THROW(delete executor);
}
