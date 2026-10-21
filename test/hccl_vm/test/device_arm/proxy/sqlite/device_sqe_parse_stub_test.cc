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
#include <cstring>
#include <gtest/gtest.h>
#include <sys/mman.h>
#include <unistd.h>

#include "device_sqe_parse_stub.h"
#include "hccl_device_pub.h"
#include "sim_capacity_limits.h"
#include "sim_ip_address.h"
#include "store_sim_memory_manager.h"
#include "sqe_v82_stub.h"
#include "udma_data_struct_stub.h"
#include "operation_data/operation_data_ops.h"
#include "storage/internal/process_storage_context.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "storage/storage_session.h"
#include "runtime_state/sim_models.h"
#include "sqlite/simulation_storage_test_helper.h"

class DeviceSqeParseTest : public testing::Test {
protected:
    void SetUp() override
    {
        SetCurRankId(0);
        runnerdb_test::ClearRecords<sim::runtime::RaJetty>();
    }

    void TearDown() override
    {
        SetCurRankId(0);
        runnerdb_test::ClearRecords<sim::runtime::RaJetty>();
    }

    /// 门铃批 UDMA 解析入口：原 ParseDavidUDMASqe 公共包装删除后，UT 自建 OpDetail 身份
    /// 上下文直连主链（本夹具原本未设 g_currOpDetailId，旧用例实际未进入解析主链）。
    /// 用后还原为 0，避免影响其他依赖"身份缺失即提前返回"语义的用例。
    void ParseUdmaDoorbellSqe(uint32_t streamId, void* sqeBuf)
    {
        sim::operation::OpDetailTab detail{};
        detail.pid = static_cast<uint32_t>(getpid());
        detail.deviceId = 0;
        detail.rankId = 0;
        detail.commId = 1; // 身份解析仅要求非 0
        detail.streamId = streamId;
        const uint64_t detailId = runnerdb_test::InsertRecord<sim::operation::OpDetailTab>(detail);
        ASSERT_NE(detailId, 0U);
        sim::operation::g_currOpDetailId = static_cast<uint32_t>(detailId);

        TaskIdentity identity{};
        ASSERT_TRUE(ResolveCurrentTaskIdentity(identity));
        ParseDavidUDMASqeWithIdentity(streamId, sqeBuf, identity);

        sim::operation::g_currOpDetailId = 0;
    }
};

TEST_F(DeviceSqeParseTest, GetFull64BitAddr_LowOnly)
{
    uint32_t lowAddr = 0x12345678;
    uint32_t highAddr = 0;

    uint64_t result = GetFull64BitAddr(lowAddr, highAddr);
    EXPECT_EQ(result, 0x12345678);
}

TEST_F(DeviceSqeParseTest, GetFull64BitAddr_HighOnly)
{
    uint32_t lowAddr = 0;
    uint32_t highAddr = 0x12345678;

    uint64_t result = GetFull64BitAddr(lowAddr, highAddr);
    EXPECT_EQ(result, 0x1234567800000000ULL);
}

TEST_F(DeviceSqeParseTest, GetFull64BitAddr_Both)
{
    uint32_t lowAddr = 0x12345678;
    uint32_t highAddr = 0x9ABCDEF0;

    uint64_t result = GetFull64BitAddr(lowAddr, highAddr);
    EXPECT_EQ(result, 0x9ABCDEF012345678ULL);
}

TEST_F(DeviceSqeParseTest, GetFull64BitAddr_MaxValues)
{
    uint32_t lowAddr = 0xFFFFFFFF;
    uint32_t highAddr = 0xFFFFFFFF;

    uint64_t result = GetFull64BitAddr(lowAddr, highAddr);
    EXPECT_EQ(result, 0xFFFFFFFFFFFFFFFFULL);
}

TEST_F(DeviceSqeParseTest, GetFull64BitAddr_ZeroValues)
{
    uint32_t lowAddr = 0;
    uint32_t highAddr = 0;

    uint64_t result = GetFull64BitAddr(lowAddr, highAddr);
    EXPECT_EQ(result, 0ULL);
}

TEST_F(DeviceSqeParseTest, ParseReduceTypeDavid_Sum)
{
    uint8_t result = 0x01;
    HcclReduceOp op = ParseReduceTypeDavid(result);
    EXPECT_EQ(op, HcclReduceOp::HCCL_REDUCE_SUM);
}

TEST_F(DeviceSqeParseTest, ParseReduceTypeDavid_Max)
{
    uint8_t result = 0x02;
    HcclReduceOp op = ParseReduceTypeDavid(result);
    EXPECT_EQ(op, HcclReduceOp::HCCL_REDUCE_MAX);
}

TEST_F(DeviceSqeParseTest, ParseReduceTypeDavid_Min)
{
    uint8_t result = 0x03;
    HcclReduceOp op = ParseReduceTypeDavid(result);
    EXPECT_EQ(op, HcclReduceOp::HCCL_REDUCE_MIN);
}

TEST_F(DeviceSqeParseTest, ParseReduceTypeDavid_Invalid)
{
    uint8_t result = 0xFF;
    HcclReduceOp op = ParseReduceTypeDavid(result);
    EXPECT_EQ(op, HcclReduceOp::HCCL_REDUCE_RESERVED);
}

TEST_F(DeviceSqeParseTest, ParseDataTypeDavid_Int8)
{
    uint8_t result = 0x00;
    HcclDataType type = ParseDataTypeDavid(result);
    EXPECT_EQ(type, HcclDataType::HCCL_DATA_TYPE_INT8);
}

TEST_F(DeviceSqeParseTest, ParseDataTypeDavid_Int16)
{
    uint8_t result = 0x10;
    HcclDataType type = ParseDataTypeDavid(result);
    EXPECT_EQ(type, HcclDataType::HCCL_DATA_TYPE_INT16);
}

TEST_F(DeviceSqeParseTest, ParseDataTypeDavid_Int32)
{
    uint8_t result = 0x20;
    HcclDataType type = ParseDataTypeDavid(result);
    EXPECT_EQ(type, HcclDataType::HCCL_DATA_TYPE_INT32);
}

TEST_F(DeviceSqeParseTest, ParseDataTypeDavid_Fp32)
{
    uint8_t result = 0x70;
    HcclDataType type = ParseDataTypeDavid(result);
    EXPECT_EQ(type, HcclDataType::HCCL_DATA_TYPE_FP32);
}

TEST_F(DeviceSqeParseTest, ParseDataTypeDavid_Invalid)
{
    uint8_t result = 0xFF;
    HcclDataType type = ParseDataTypeDavid(result);
    EXPECT_EQ(type, HcclDataType::HCCL_DATA_TYPE_RESERVED);
}

TEST_F(DeviceSqeParseTest, ParseUbReduceTypeDavid_Sum)
{
    uint32_t type = 0xA;
    HcclReduceOp op = ParseUbReduceTypeDavid(type);
    EXPECT_EQ(op, HcclReduceOp::HCCL_REDUCE_SUM);
}

TEST_F(DeviceSqeParseTest, ParseUbReduceTypeDavid_Max)
{
    uint32_t type = 0x8;
    HcclReduceOp op = ParseUbReduceTypeDavid(type);
    EXPECT_EQ(op, HcclReduceOp::HCCL_REDUCE_MAX);
}

TEST_F(DeviceSqeParseTest, ParseUbReduceTypeDavid_Min)
{
    uint32_t type = 0x9;
    HcclReduceOp op = ParseUbReduceTypeDavid(type);
    EXPECT_EQ(op, HcclReduceOp::HCCL_REDUCE_MIN);
}

TEST_F(DeviceSqeParseTest, ParseUbReduceTypeDavid_Invalid)
{
    uint32_t type = 0xFF;
    HcclReduceOp op = ParseUbReduceTypeDavid(type);
    EXPECT_EQ(op, HcclReduceOp::HCCL_REDUCE_RESERVED);
}

TEST_F(DeviceSqeParseTest, ParseUbDataTypeDavid_Int8)
{
    uint32_t type = 0x0;
    HcclDataType dataType = ParseUbDataTypeDavid(type);
    EXPECT_EQ(dataType, HcclDataType::HCCL_DATA_TYPE_INT8);
}

TEST_F(DeviceSqeParseTest, ParseUbDataTypeDavid_Fp32)
{
    uint32_t type = 0x7;
    HcclDataType dataType = ParseUbDataTypeDavid(type);
    EXPECT_EQ(dataType, HcclDataType::HCCL_DATA_TYPE_FP32);
}

TEST_F(DeviceSqeParseTest, ParseUbDataTypeDavid_Invalid)
{
    uint32_t type = 0xFF;
    HcclDataType dataType = ParseUbDataTypeDavid(type);
    EXPECT_EQ(dataType, HcclDataType::HCCL_DATA_TYPE_RESERVED);
}

TEST_F(DeviceSqeParseTest, PrintTaskMetaData_MemCpy)
{
    HcclTaskMetaData taskMeta;
    memset(&taskMeta, 0, sizeof(taskMeta));
    taskMeta.taskType = HccLTaskMetaType::MEM_CPY;
    taskMeta.rankId = 0;
    taskMeta.streamId = 0;
    taskMeta.taskData.transMem.srcOffset = 0x1000;
    taskMeta.taskData.transMem.dstOffset = 0x2000;
    taskMeta.taskData.transMem.len = 1024;

    EXPECT_NO_THROW(PrintTaskMetaData(taskMeta));
}

TEST_F(DeviceSqeParseTest, PrintTaskMetaData_Reduce)
{
    HcclTaskMetaData taskMeta;
    memset(&taskMeta, 0, sizeof(taskMeta));
    taskMeta.taskType = HccLTaskMetaType::REDUCE;
    taskMeta.rankId = 0;
    taskMeta.streamId = 0;

    EXPECT_NO_THROW(PrintTaskMetaData(taskMeta));
}

TEST_F(DeviceSqeParseTest, PrintTaskMetaData_NotifyWait)
{
    HcclTaskMetaData taskMeta;
    memset(&taskMeta, 0, sizeof(taskMeta));
    taskMeta.taskType = HccLTaskMetaType::NOTIFY_WAIT;
    taskMeta.rankId = 0;
    taskMeta.streamId = 0;

    EXPECT_NO_THROW(PrintTaskMetaData(taskMeta));
}

TEST_F(DeviceSqeParseTest, PrintTaskMetaData_NotifyRecord)
{
    HcclTaskMetaData taskMeta;
    memset(&taskMeta, 0, sizeof(taskMeta));
    taskMeta.taskType = HccLTaskMetaType::NOTIFY_RECORD;
    taskMeta.rankId = 0;
    taskMeta.streamId = 0;

    EXPECT_NO_THROW(PrintTaskMetaData(taskMeta));
}

// ==================== ParseDavidSDMASqe Tests ====================

TEST_F(DeviceSqeParseTest, ParseDavidSDMASqe_Memcpy)
{
    // Create a mock SDMA SQE with opcode = 0 (memcpy)
    Rt91095StarsMemcpySqe sqe;
    memset(&sqe, 0, sizeof(sqe));
    sqe.opcode = 0; // memcpy
    sqe.u.strideMode0.lengthMove = 1024;
    sqe.u.strideMode0.srcAddrLow = 0x1000;
    sqe.u.strideMode0.srcAddrHigh = 0;
    sqe.u.strideMode0.dstAddrLow = 0x2000;
    sqe.u.strideMode0.dstAddrHigh = 0;

    // This will call GetDeviceIdByDevAddr which requires database setup
    // Just verify it doesn't crash
    EXPECT_NO_THROW(ParseDavidSDMASqe(0, &sqe));
}

TEST_F(DeviceSqeParseTest, ParseDavidSDMASqe_Reduce)
{
    // Create a mock SDMA SQE with opcode != 0 (reduce)
    Rt91095StarsMemcpySqe sqe;
    memset(&sqe, 0, sizeof(sqe));
    sqe.opcode = 0x01; // reduce (SUM)
    sqe.u.strideMode0.lengthMove = 1024;
    sqe.u.strideMode0.srcAddrLow = 0x1000;
    sqe.u.strideMode0.srcAddrHigh = 0;
    sqe.u.strideMode0.dstAddrLow = 0x2000;
    sqe.u.strideMode0.dstAddrHigh = 0;

    EXPECT_NO_THROW(ParseDavidSDMASqe(0, &sqe));
}

// ==================== ParseDavidNotifySqe Tests ====================

TEST_F(DeviceSqeParseTest, ParseDavidNotifySqe_Wait)
{
    Rt91095StarsNotifySqe sqe;
    memset(&sqe, 0, sizeof(sqe));
    sqe.notifyId = 123;

    EXPECT_NO_THROW(ParseDavidNotifySqe(0, &sqe, false));
}

TEST_F(DeviceSqeParseTest, ParseDavidNotifySqe_Record)
{
    Rt91095StarsNotifySqe sqe;
    memset(&sqe, 0, sizeof(sqe));
    sqe.notifyId = 456;

    EXPECT_NO_THROW(ParseDavidNotifySqe(0, &sqe, true));
}

// ==================== GetRmtDeviceIdByEid Tests ====================

TEST_F(DeviceSqeParseTest, GetRmtDeviceIdByEid_Zero)
{
    uint8_t eid[URMA_EID_LEN] = {};
    uint32_t deviceId = 0;
    EXPECT_NO_THROW(GetRmtDeviceIdByEid(eid, deviceId));
}

TEST_F(DeviceSqeParseTest, GetRmtDeviceIdByEid_Localhost)
{
    uint8_t eid[URMA_EID_LEN] = {};
    eid[URMA_EID_LEN - 4] = 0x7F;
    eid[URMA_EID_LEN - 1] = 0x01;
    uint32_t deviceId = 0;
    EXPECT_NO_THROW(GetRmtDeviceIdByEid(eid, deviceId));
}

TEST_F(DeviceSqeParseTest, GetRmtDeviceIdByEid_ClassA)
{
    uint8_t eid[URMA_EID_LEN] = {};
    eid[URMA_EID_LEN - 4] = 0x0A;
    eid[URMA_EID_LEN - 3] = 0x0A;
    eid[URMA_EID_LEN - 2] = 0x0A;
    eid[URMA_EID_LEN - 1] = 0x0A;
    uint32_t deviceId = 0;
    EXPECT_NO_THROW(GetRmtDeviceIdByEid(eid, deviceId));
}

// ==================== ParseA5SqeFromSqBuffer Tests ====================

TEST_F(DeviceSqeParseTest, ParseA5SqeFromSqBuffer_TailLessThanHead)
{
    // This tests the tail < head branch (data wrap-around)
    // We need to set up a scenario where tail < head
    // Since this function uses global state, we need to be careful

    // Create a mock halSqCqConfigInfo
    struct halSqCqConfigInfo info;
    memset(&info, 0, sizeof(info));
    info.sqId = 0;
    info.value[0] = 5; // tail = 5

    // Set head to a value > tail to trigger the wrap-around path
    // We need to call UpdateSqTail first to set the head
    UpdateSqTail(0, 10); // Set head = 10

    // Now tail (5) < head (10), which should trigger the wrap-around path
    EXPECT_NO_THROW(ParseA5SqeFromSqBuffer(0, &info));
}

TEST_F(DeviceSqeParseTest, ParseA5SqeFromSqBuffer_DefaultSqeType)
{
    struct halSqCqConfigInfo info;
    memset(&info, 0, sizeof(info));
    info.sqId = 0;
    info.value[0] = 1;
    UpdateSqTail(0, 0);
    EXPECT_NO_THROW(ParseA5SqeFromSqBuffer(0, &info));
}

TEST_F(DeviceSqeParseTest, ParseA5SqeFromSqBuffer_SDMAType)
{
    struct halSqCqConfigInfo info;
    memset(&info, 0, sizeof(info));
    info.sqId = 0;
    info.value[0] = 1;
    UpdateSqTail(0, 0);

    uint8_t* sqBuf = nullptr;
    GetSqBufferAddr(&sqBuf);
    if (sqBuf) {
        Rt91095StarsMemcpySqe sqe;
        memset(&sqe, 0, sizeof(sqe));
        sqe.header.type = static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA);
        sqe.opcode = 0;
        sqe.u.strideMode0.lengthMove = 1024;
        sqe.u.strideMode0.srcAddrLow = 0x1000;
        sqe.u.strideMode0.srcAddrHigh = 0;
        sqe.u.strideMode0.dstAddrLow = 0x2000;
        sqe.u.strideMode0.dstAddrHigh = 0;
        memcpy(sqBuf, &sqe, sizeof(sqe));
    }

    EXPECT_NO_THROW(ParseA5SqeFromSqBuffer(0, &info));
}
TEST_F(DeviceSqeParseTest, ParseA5SqeFromSqBuffer_NotifyWaitType)
{
    struct halSqCqConfigInfo info;
    memset(&info, 0, sizeof(info));
    info.sqId = 0;
    info.value[0] = 1;
    UpdateSqTail(0, 0);

    uint8_t* sqBuf = nullptr;
    GetSqBufferAddr(&sqBuf);
    if (sqBuf) {
        Rt91095StarsNotifySqe sqe;
        memset(&sqe, 0, sizeof(sqe));
        sqe.header.type = static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
        sqe.notifyId = 100;
        memcpy(sqBuf, &sqe, sizeof(sqe));
    }

    EXPECT_NO_THROW(ParseA5SqeFromSqBuffer(0, &info));
}
TEST_F(DeviceSqeParseTest, ParseA5SqeFromSqBuffer_NotifyRecordType)
{
    struct halSqCqConfigInfo info;
    memset(&info, 0, sizeof(info));
    info.sqId = 0;
    info.value[0] = 1;
    UpdateSqTail(0, 0);

    uint8_t* sqBuf = nullptr;
    GetSqBufferAddr(&sqBuf);
    if (sqBuf) {
        Rt91095StarsNotifySqe sqe;
        memset(&sqe, 0, sizeof(sqe));
        sqe.header.type = static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_RECORD);
        sqe.notifyId = 200;
        memcpy(sqBuf, &sqe, sizeof(sqe));
    }

    EXPECT_NO_THROW(ParseA5SqeFromSqBuffer(0, &info));
}
TEST_F(DeviceSqeParseTest, ParseA5SqeFromSqBuffer_UBDMAType)
{
    struct halSqCqConfigInfo info;
    memset(&info, 0, sizeof(info));
    info.sqId = 0;
    info.value[0] = 1;
    UpdateSqTail(0, 0);

    uint8_t* sqBuf = nullptr;
    GetSqBufferAddr(&sqBuf);
    if (sqBuf) {
        Rt91095StarsUbdmaDBmodeSqe sqe;
        memset(&sqe, 0, sizeof(sqe));
        sqe.header.type = static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_UBDMA);
        sqe.jettyId1 = HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN;
        sqe.piValue1 = 1;
        memcpy(sqBuf, &sqe, sizeof(sqe));
    }

    // With null-check added to the doorbell parse chain, this won't crash anymore
    EXPECT_NO_THROW(ParseA5SqeFromSqBuffer(0, &info));
}

// ==================== ParseDavidUDMASqe Tests ====================
// With DB-based RaJetty lookup in device_sqe_parse_stub.cc, the doorbell parse chain returns
// early when no corresponding RaJetty exists. We test both missing-jetty and valid-jetty paths.

TEST_F(DeviceSqeParseTest, ParseDavidUDMASqe_NoJettyInDb)
{
    // Without RaJetty in DB, GetWqebufferByJettyId fails
    // the parse chain should return early with error log
    Rt91095StarsUbdmaDBmodeSqe ubSqe;
    memset(&ubSqe, 0, sizeof(ubSqe));
    ubSqe.jettyId1 = HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN;
    ubSqe.piValue1 = 2;

    EXPECT_NO_THROW(ParseUdmaDoorbellSqe(0, &ubSqe));
}

TEST_F(DeviceSqeParseTest, ParseDavidUDMASqe_WriteOpcodeWithShm)
{
    alignas(64) uint8_t wqeBuffer[HCCL_WQE_SIZE * 4] = {};
    sim::runtime::RaJetty jetty{};
    jetty.id = 1;
    // 新合同：SQE 的 jettyId1 是 USER_CTL_NORMAL(AICPU) 硬件编号 [5312,9407]；设备按
    // （编号, 创建者 pid, 模式）复合查行。测试进程同时扮演创建者与解析者，创建者 pid
    // 即本进程的父进程（设备视角的 rank host）。
    jetty.jetty_id = HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN;
    jetty.mode = HcclSim::HCCL_VM_JETTY_MODE_USER_CTL_NORMAL;
    jetty.pid = static_cast<uint64_t>(getppid());
    jetty.sqBuffer = reinterpret_cast<uint64_t>(wqeBuffer);
    runnerdb_test::InsertRecord<sim::runtime::RaJetty>(jetty);

    // Setup WQE with WRITE opcode at ciVal=0
    UdmaSqeWrite* ubWqeWrite = reinterpret_cast<UdmaSqeWrite*>(wqeBuffer);
    memset(ubWqeWrite, 0, sizeof(UdmaSqeWrite));
    ubWqeWrite->comm.opcode = static_cast<int>(UdmaSqOpcode::UDMA_OPC_WRITE);
    ubWqeWrite->comm.inlineEn = 0;
    ubWqeWrite->comm.udfFlag = 0;
    ubWqeWrite->comm.rmtAddrLow = 0x2000;
    ubWqeWrite->u.sge.dataAddrLow = 0x1000;
    ubWqeWrite->u.sge.length = 128;

    Rt91095StarsUbdmaDBmodeSqe ubSqe;
    memset(&ubSqe, 0, sizeof(ubSqe));
    ubSqe.jettyId1 = HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN;
    ubSqe.piValue1 = 2;

    EXPECT_NO_THROW(ParseUdmaDoorbellSqe(0, &ubSqe));
}

TEST_F(DeviceSqeParseTest, ParseDavidUDMASqe_WriteWithNotifyOpcodeWithShm)
{
    alignas(64) uint8_t wqeBuffer[HCCL_WQE_SIZE * 4] = {};
    sim::runtime::RaJetty jetty{};
    jetty.id = 1;
    // 新合同：SQE 的 jettyId1 是 USER_CTL_NORMAL(AICPU) 硬件编号 [5312,9407]；设备按
    // （编号, 创建者 pid, 模式）复合查行。测试进程同时扮演创建者与解析者，创建者 pid
    // 即本进程的父进程（设备视角的 rank host）。
    jetty.jetty_id = HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN;
    jetty.mode = HcclSim::HCCL_VM_JETTY_MODE_USER_CTL_NORMAL;
    jetty.pid = static_cast<uint64_t>(getppid());
    jetty.sqBuffer = reinterpret_cast<uint64_t>(wqeBuffer);
    runnerdb_test::InsertRecord<sim::runtime::RaJetty>(jetty);

    // piValue1=2, ciVal = piValue1-2=0, first WQE has WRITE_WITH_NOTIFY
    UdmaSqeCommon* ubCommon = reinterpret_cast<UdmaSqeCommon*>(wqeBuffer);
    memset(ubCommon, 0, sizeof(UdmaSqeCommon));
    ubCommon->opcode = static_cast<int>(UdmaSqOpcode::UDMA_OPC_WRITE_WITH_NOTIFY);

    // Setup WriteWithNotify WQE content
    UdmaSqeWriteWithNotify* ubWqe = reinterpret_cast<UdmaSqeWriteWithNotify*>(wqeBuffer);
    ubWqe->comm.udfFlag = 0;
    ubWqe->comm.rmtAddrLow = 0x2000;
    ubWqe->comm.rmtEid[0] = 0;
    ubWqe->localU.sge.dataAddrLow = 0x1000;
    ubWqe->localU.sge.length = 256;
    ubWqe->notify.notifyAddrLow = 0x4000;

    Rt91095StarsUbdmaDBmodeSqe ubSqe;
    memset(&ubSqe, 0, sizeof(ubSqe));
    ubSqe.jettyId1 = HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN;
    ubSqe.piValue1 = 2;

    EXPECT_NO_THROW(ParseUdmaDoorbellSqe(0, &ubSqe));
}

TEST_F(DeviceSqeParseTest, ParseDavidUDMASqe_ReadOpcodeWithShm)
{
    alignas(64) uint8_t wqeBuffer[HCCL_WQE_SIZE * 4] = {};
    sim::runtime::RaJetty jetty{};
    jetty.id = 1;
    // 新合同：SQE 的 jettyId1 是 USER_CTL_NORMAL(AICPU) 硬件编号 [5312,9407]；设备按
    // （编号, 创建者 pid, 模式）复合查行。测试进程同时扮演创建者与解析者，创建者 pid
    // 即本进程的父进程（设备视角的 rank host）。
    jetty.jetty_id = HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN;
    jetty.mode = HcclSim::HCCL_VM_JETTY_MODE_USER_CTL_NORMAL;
    jetty.pid = static_cast<uint64_t>(getppid());
    jetty.sqBuffer = reinterpret_cast<uint64_t>(wqeBuffer);
    runnerdb_test::InsertRecord<sim::runtime::RaJetty>(jetty);

    UdmaSqeWrite* ubWqeRead = reinterpret_cast<UdmaSqeWrite*>(wqeBuffer);
    memset(ubWqeRead, 0, sizeof(UdmaSqeWrite));
    ubWqeRead->comm.opcode = static_cast<int>(UdmaSqOpcode::UDMA_OPC_READ);
    ubWqeRead->comm.inlineEn = 0;
    ubWqeRead->comm.udfFlag = 0;
    ubWqeRead->comm.rmtAddrLow = 0x2000;
    ubWqeRead->u.sge.dataAddrLow = 0x1000;
    ubWqeRead->u.sge.length = 64;

    Rt91095StarsUbdmaDBmodeSqe ubSqe;
    memset(&ubSqe, 0, sizeof(ubSqe));
    ubSqe.jettyId1 = HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN;
    ubSqe.piValue1 = 2;

    EXPECT_NO_THROW(ParseUdmaDoorbellSqe(0, &ubSqe));
}

TEST_F(DeviceSqeParseTest, ParseDavidUDMASqe_UnsupportedOpcodeWithShm)
{
    alignas(64) uint8_t wqeBuffer[HCCL_WQE_SIZE * 4] = {};
    sim::runtime::RaJetty jetty{};
    jetty.id = 1;
    // 新合同：SQE 的 jettyId1 是 USER_CTL_NORMAL(AICPU) 硬件编号 [5312,9407]；设备按
    // （编号, 创建者 pid, 模式）复合查行。测试进程同时扮演创建者与解析者，创建者 pid
    // 即本进程的父进程（设备视角的 rank host）。
    jetty.jetty_id = HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN;
    jetty.mode = HcclSim::HCCL_VM_JETTY_MODE_USER_CTL_NORMAL;
    jetty.pid = static_cast<uint64_t>(getppid());
    jetty.sqBuffer = reinterpret_cast<uint64_t>(wqeBuffer);
    runnerdb_test::InsertRecord<sim::runtime::RaJetty>(jetty);

    // Put unsupported opcode at ciVal position (adjusted: piValue1-1)
    UdmaSqeCommon* ubCommon = reinterpret_cast<UdmaSqeCommon*>(wqeBuffer + 1 * HCCL_WQE_SIZE);
    memset(ubCommon, 0, sizeof(UdmaSqeCommon));
    ubCommon->opcode = 63;

    Rt91095StarsUbdmaDBmodeSqe ubSqe;
    memset(&ubSqe, 0, sizeof(ubSqe));
    ubSqe.jettyId1 = HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN;
    ubSqe.piValue1 = 2;

    EXPECT_NO_THROW(ParseUdmaDoorbellSqe(0, &ubSqe));
}

TEST_F(DeviceSqeParseTest, ParseDavidUDMASqe_AdjustCiValWithShm)
{
    alignas(64) uint8_t wqeBuffer[HCCL_WQE_SIZE * 8] = {};
    sim::runtime::RaJetty jetty{};
    jetty.id = 1;
    // 新合同：SQE 的 jettyId1 是 USER_CTL_NORMAL(AICPU) 硬件编号 [5312,9407]；设备按
    // （编号, 创建者 pid, 模式）复合查行。测试进程同时扮演创建者与解析者，创建者 pid
    // 即本进程的父进程（设备视角的 rank host）。
    jetty.jetty_id = HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN;
    jetty.mode = HcclSim::HCCL_VM_JETTY_MODE_USER_CTL_NORMAL;
    jetty.pid = static_cast<uint64_t>(getppid());
    jetty.sqBuffer = reinterpret_cast<uint64_t>(wqeBuffer);
    runnerdb_test::InsertRecord<sim::runtime::RaJetty>(jetty);

    // piValue1=3, first ciVal = 3-2=1, opcode at [1] is not WRITE_WITH_NOTIFY,
    // so adjusted ciVal = 3-1=2, reads WQE at [2]
    UdmaSqeCommon* ubCommon1 = reinterpret_cast<UdmaSqeCommon*>(wqeBuffer + 1 * HCCL_WQE_SIZE);
    memset(ubCommon1, 0, sizeof(UdmaSqeCommon));
    ubCommon1->opcode = static_cast<int>(UdmaSqOpcode::UDMA_OPC_WRITE);

    UdmaSqeWrite* ubWqeWrite = reinterpret_cast<UdmaSqeWrite*>(wqeBuffer + 2 * HCCL_WQE_SIZE);
    memset(ubWqeWrite, 0, sizeof(UdmaSqeWrite));
    ubWqeWrite->comm.opcode = static_cast<int>(UdmaSqOpcode::UDMA_OPC_WRITE);
    ubWqeWrite->comm.inlineEn = 0;
    ubWqeWrite->comm.udfFlag = 0;
    ubWqeWrite->comm.rmtAddrLow = 0x2000;
    ubWqeWrite->u.sge.dataAddrLow = 0x1000;
    ubWqeWrite->u.sge.length = 128;

    Rt91095StarsUbdmaDBmodeSqe ubSqe;
    memset(&ubSqe, 0, sizeof(ubSqe));
    ubSqe.jettyId1 = HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN;
    ubSqe.piValue1 = 3;

    EXPECT_NO_THROW(ParseUdmaDoorbellSqe(0, &ubSqe));
}

TEST_F(DeviceSqeParseTest, ParseDavidUDMASqe_NullWqeBuffer)
{
    // RaJetty exists but sqBuffer=0, so wqeBuffer=0
    sim::runtime::RaJetty jetty{};
    jetty.id = 1;
    // 新合同：SQE 的 jettyId1 是 USER_CTL_NORMAL(AICPU) 硬件编号 [5312,9407]；设备按
    // （编号, 创建者 pid, 模式）复合查行。测试进程同时扮演创建者与解析者，创建者 pid
    // 即本进程的父进程（设备视角的 rank host）。
    jetty.jetty_id = HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN;
    jetty.mode = HcclSim::HCCL_VM_JETTY_MODE_USER_CTL_NORMAL;
    jetty.pid = static_cast<uint64_t>(getppid());
    jetty.sqBuffer = 0;
    runnerdb_test::InsertRecord<sim::runtime::RaJetty>(jetty);

    Rt91095StarsUbdmaDBmodeSqe ubSqe;
    memset(&ubSqe, 0, sizeof(ubSqe));
    ubSqe.jettyId1 = HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN;
    ubSqe.piValue1 = 2;

    EXPECT_NO_THROW(ParseUdmaDoorbellSqe(0, &ubSqe));
}

TEST_F(DeviceSqeParseTest, ParseDavidUDMASqe_WriteWithNotifyWithShm_Reduce)
{
    alignas(64) uint8_t wqeBuffer[HCCL_WQE_SIZE * 4] = {};
    sim::runtime::RaJetty jetty{};
    jetty.id = 1;
    // 新合同：SQE 的 jettyId1 是 USER_CTL_NORMAL(AICPU) 硬件编号 [5312,9407]；设备按
    // （编号, 创建者 pid, 模式）复合查行。测试进程同时扮演创建者与解析者，创建者 pid
    // 即本进程的父进程（设备视角的 rank host）。
    jetty.jetty_id = HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN;
    jetty.mode = HcclSim::HCCL_VM_JETTY_MODE_USER_CTL_NORMAL;
    jetty.pid = static_cast<uint64_t>(getppid());
    jetty.sqBuffer = reinterpret_cast<uint64_t>(wqeBuffer);
    runnerdb_test::InsertRecord<sim::runtime::RaJetty>(jetty);

    UdmaSqeWriteWithNotify* ubWqe = reinterpret_cast<UdmaSqeWriteWithNotify*>(wqeBuffer);
    memset(ubWqe, 0, sizeof(UdmaSqeWriteWithNotify));
    ubWqe->comm.opcode = static_cast<int>(UdmaSqOpcode::UDMA_OPC_WRITE_WITH_NOTIFY);
    ubWqe->comm.udfFlag = 1;
    ubWqe->comm.rmtAddrLow = 0x2000;
    ubWqe->comm.rmtEid[0] = 0;
    ubWqe->comm.inlinedata.udfData.reduceOp = 0xA;
    ubWqe->comm.inlinedata.udfData.reduceType = 0x7;
    ubWqe->localU.sge.dataAddrLow = 0x1000;
    ubWqe->localU.sge.length = 512;
    ubWqe->notify.notifyAddrLow = 0x4000;

    Rt91095StarsUbdmaDBmodeSqe ubSqe;
    memset(&ubSqe, 0, sizeof(ubSqe));
    ubSqe.jettyId1 = HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN;
    ubSqe.piValue1 = 2;

    EXPECT_NO_THROW(ParseUdmaDoorbellSqe(0, &ubSqe));
}

// ==================== AICPU Jetty 编号合同回归（256p 缺陷）====================
// 背景：UBDMA SQE 的 jettyId1 仅 16 位；旧实现把无界行主键当编号下发，行数超过 65535 后
// 经 16 位字段回绕（b4 实测行主键 65536~65792 回绕为 0~256，命中错误行导致数据面 SQE
// 丢弃）。新合同（sim_capacity_limits.h）：编号为 USER_CTL_NORMAL 硬件编号 [5312,9407]，
// 按 rank 进程内分配；设备按（编号, 创建者 pid, 模式）复合查行。

TEST_F(DeviceSqeParseTest, GetWqebufferByJettyId_ResolvesByHwIdAndOwnerPidNotRowId)
{
    constexpr uint32_t kHwId = HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN; // 首个合法编号

    // 另一进程（pid=1）的同编号行：设备按 owner 隔离，绝不能命中
    sim::runtime::RaJetty foreign{};
    foreign.id = kHwId;
    foreign.jetty_id = kHwId;
    foreign.mode = HcclSim::HCCL_VM_JETTY_MODE_USER_CTL_NORMAL;
    foreign.pid = 1;
    foreign.sqBuffer = 0;
    runnerdb_test::InsertRecord<sim::runtime::RaJetty>(foreign);

    // 本进程（解析者的父进程，即设备视角的 rank host）的 jetty：行主键 65629 与编号无关
    // （b4 实测形状：行主键可远超 16 位）
    alignas(64) uint8_t myBuffer[HCCL_WQE_SIZE * 4] = {};
    sim::runtime::RaJetty jetty{};
    jetty.id = 65629;
    jetty.jetty_id = kHwId;
    jetty.mode = HcclSim::HCCL_VM_JETTY_MODE_USER_CTL_NORMAL;
    jetty.pid = static_cast<uint64_t>(getppid());
    jetty.sqBuffer = reinterpret_cast<uint64_t>(myBuffer);
    runnerdb_test::InsertRecord<sim::runtime::RaJetty>(jetty);

    uint64_t wqeBuffer = 0;
    EXPECT_TRUE(GetWqebufferByJettyId(kHwId, wqeBuffer));
    EXPECT_EQ(wqeBuffer, reinterpret_cast<uint64_t>(myBuffer));

    // 同命名空间内未注册的编号不得串扰
    uint64_t unregistered = 123;
    EXPECT_FALSE(GetWqebufferByJettyId(kHwId + 1, unregistered));
}

TEST_F(DeviceSqeParseTest, GetWqebufferByJettyId_RejectsOutOfRangeBeforeNarrowing)
{
    // 入参在窄化转换前校验：越界值（含 uint64 大值）直接失败，不触库、不窄化
    uint64_t out = 123;
    EXPECT_FALSE(GetWqebufferByJettyId(HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN - 1, out));
    EXPECT_FALSE(GetWqebufferByJettyId(HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_END + 1, out));
    EXPECT_FALSE(GetWqebufferByJettyId(0, out));
    EXPECT_FALSE(GetWqebufferByJettyId(UINT32_MAX, out));
    EXPECT_FALSE(GetWqebufferByJettyId(0xFFFFFFFFFFFFFFFFULL, out));
    EXPECT_EQ(out, 123u); // 失败路径不污染出参
}

TEST_F(DeviceSqeParseTest, GetWqebufferByJettyId_ModeIsolationExcludesNonAicpuRows)
{
    // 同 owner、同编号、不同 mode 并存（假设性脏数据：CCU 行占用了 AICPU 编号）：
    // AICPU 消费者必须只命中 USER_CTL_NORMAL 行，不得命中同 owner 的 CCU 行。
    constexpr uint32_t kHwId = HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN + 7;
    alignas(64) uint8_t aicpuBuffer[HCCL_WQE_SIZE * 4] = {};
    alignas(64) uint8_t ccuBuffer[HCCL_WQE_SIZE * 4] = {};

    sim::runtime::RaJetty ccuRow{};
    ccuRow.id = 1;
    ccuRow.jetty_id = kHwId;
    ccuRow.mode = 2; // CCU
    ccuRow.pid = static_cast<uint64_t>(getppid());
    ccuRow.sqBuffer = reinterpret_cast<uint64_t>(ccuBuffer);
    runnerdb_test::InsertRecord<sim::runtime::RaJetty>(ccuRow);

    sim::runtime::RaJetty aicpuRow{};
    aicpuRow.id = 2;
    aicpuRow.jetty_id = kHwId;
    aicpuRow.mode = HcclSim::HCCL_VM_JETTY_MODE_USER_CTL_NORMAL;
    aicpuRow.pid = static_cast<uint64_t>(getppid());
    aicpuRow.sqBuffer = reinterpret_cast<uint64_t>(aicpuBuffer);
    runnerdb_test::InsertRecord<sim::runtime::RaJetty>(aicpuRow);

    uint64_t wqeBuffer = 0;
    EXPECT_TRUE(GetWqebufferByJettyId(kHwId, wqeBuffer));
    EXPECT_EQ(wqeBuffer, reinterpret_cast<uint64_t>(aicpuBuffer));
}

TEST_F(DeviceSqeParseTest, ParseDavidUDMASqe_RowIdBeyond16BitStillResolves)
{
    // b4 实测形状：本 rank 的 jetty 行主键 65629（超过 16 位），SQE 携带硬件编号
    constexpr uint32_t kHwId = HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN + 93;
    alignas(64) uint8_t wqeBuffer[HCCL_WQE_SIZE * 4] = {};
    sim::runtime::RaJetty jetty{};
    jetty.id = 65629;
    jetty.jetty_id = kHwId;
    jetty.mode = HcclSim::HCCL_VM_JETTY_MODE_USER_CTL_NORMAL;
    jetty.pid = static_cast<uint64_t>(getppid());
    jetty.sqBuffer = reinterpret_cast<uint64_t>(wqeBuffer);
    runnerdb_test::InsertRecord<sim::runtime::RaJetty>(jetty);

    UdmaSqeWrite* ubWqeWrite = reinterpret_cast<UdmaSqeWrite*>(wqeBuffer);
    memset(ubWqeWrite, 0, sizeof(UdmaSqeWrite));
    ubWqeWrite->comm.opcode = static_cast<int>(UdmaSqOpcode::UDMA_OPC_WRITE);
    ubWqeWrite->comm.inlineEn = 0;
    ubWqeWrite->comm.udfFlag = 0;
    ubWqeWrite->comm.rmtAddrLow = 0x2000;
    ubWqeWrite->u.sge.dataAddrLow = 0x1000;
    ubWqeWrite->u.sge.length = 128;

    Rt91095StarsUbdmaDBmodeSqe ubSqe;
    memset(&ubSqe, 0, sizeof(ubSqe));
    ubSqe.jettyId1 = kHwId;
    ubSqe.piValue1 = 2;

    EXPECT_NO_THROW(ParseUdmaDoorbellSqe(0, &ubSqe));
}

// ==================== Additional Coverage Tests ====================

TEST_F(DeviceSqeParseTest, ParseDataTypeDavid_Uint8)
{
    uint8_t result = 0x30;
    HcclDataType type = ParseDataTypeDavid(result);
    EXPECT_EQ(type, HcclDataType::HCCL_DATA_TYPE_UINT8);
}

TEST_F(DeviceSqeParseTest, ParseDataTypeDavid_Uint16)
{
    uint8_t result = 0x40;
    HcclDataType type = ParseDataTypeDavid(result);
    EXPECT_EQ(type, HcclDataType::HCCL_DATA_TYPE_UINT16);
}

TEST_F(DeviceSqeParseTest, ParseDataTypeDavid_Uint32)
{
    uint8_t result = 0x50;
    HcclDataType type = ParseDataTypeDavid(result);
    EXPECT_EQ(type, HcclDataType::HCCL_DATA_TYPE_UINT32);
}

TEST_F(DeviceSqeParseTest, ParseDataTypeDavid_Fp16)
{
    uint8_t result = 0x60;
    HcclDataType type = ParseDataTypeDavid(result);
    EXPECT_EQ(type, HcclDataType::HCCL_DATA_TYPE_FP16);
}

TEST_F(DeviceSqeParseTest, ParseDataTypeDavid_Bfp16Alt)
{
    uint8_t result = 0x80;
    HcclDataType type = ParseDataTypeDavid(result);
    EXPECT_EQ(type, HcclDataType::HCCL_DATA_TYPE_BFP16);
}

TEST_F(DeviceSqeParseTest, ParseUbDataTypeDavid_Int16)
{
    uint32_t type = 0x1;
    HcclDataType dataType = ParseUbDataTypeDavid(type);
    EXPECT_EQ(dataType, HcclDataType::HCCL_DATA_TYPE_INT16);
}

TEST_F(DeviceSqeParseTest, ParseUbDataTypeDavid_Int32)
{
    uint32_t type = 0x2;
    HcclDataType dataType = ParseUbDataTypeDavid(type);
    EXPECT_EQ(dataType, HcclDataType::HCCL_DATA_TYPE_INT32);
}

TEST_F(DeviceSqeParseTest, ParseUbDataTypeDavid_Uint8)
{
    uint32_t type = 0x3;
    HcclDataType dataType = ParseUbDataTypeDavid(type);
    EXPECT_EQ(dataType, HcclDataType::HCCL_DATA_TYPE_UINT8);
}

TEST_F(DeviceSqeParseTest, ParseUbDataTypeDavid_Uint16)
{
    uint32_t type = 0x4;
    HcclDataType dataType = ParseUbDataTypeDavid(type);
    EXPECT_EQ(dataType, HcclDataType::HCCL_DATA_TYPE_UINT16);
}

TEST_F(DeviceSqeParseTest, ParseUbDataTypeDavid_Uint32)
{
    uint32_t type = 0x5;
    HcclDataType dataType = ParseUbDataTypeDavid(type);
    EXPECT_EQ(dataType, HcclDataType::HCCL_DATA_TYPE_UINT32);
}

TEST_F(DeviceSqeParseTest, ParseUbDataTypeDavid_Fp16)
{
    uint32_t type = 0x6;
    HcclDataType dataType = ParseUbDataTypeDavid(type);
    EXPECT_EQ(dataType, HcclDataType::HCCL_DATA_TYPE_FP16);
}

TEST_F(DeviceSqeParseTest, PrintTaskMetaData_DefaultType)
{
    HcclTaskMetaData taskMeta;
    memset(&taskMeta, 0, sizeof(taskMeta));
    taskMeta.taskType = static_cast<HccLTaskMetaType>(99);
    EXPECT_NO_THROW(PrintTaskMetaData(taskMeta));
}

TEST_F(DeviceSqeParseTest, ParseDavidSDMASqe_MemcpyWithNonZeroRank)
{
    Rt91095StarsMemcpySqe sqe;
    memset(&sqe, 0, sizeof(sqe));
    sqe.opcode = 0;
    sqe.u.strideMode0.lengthMove = 2048;
    sqe.u.strideMode0.srcAddrLow = 0x4000;
    sqe.u.strideMode0.srcAddrHigh = 0x1;
    sqe.u.strideMode0.dstAddrLow = 0x8000;
    sqe.u.strideMode0.dstAddrHigh = 0x2;

    SetCurRankId(3);
    EXPECT_NO_THROW(ParseDavidSDMASqe(1, &sqe));
    SetCurRankId(0);
}

TEST_F(DeviceSqeParseTest, ParseDavidSDMASqe_ReduceWithCombinedOpcode)
{
    Rt91095StarsMemcpySqe sqe;
    memset(&sqe, 0, sizeof(sqe));
    sqe.opcode = 0x21;
    sqe.u.strideMode0.lengthMove = 512;
    sqe.u.strideMode0.srcAddrLow = 0x1000;
    sqe.u.strideMode0.srcAddrHigh = 0;
    sqe.u.strideMode0.dstAddrLow = 0x2000;
    sqe.u.strideMode0.dstAddrHigh = 0;

    EXPECT_NO_THROW(ParseDavidSDMASqe(0, &sqe));
}

TEST_F(DeviceSqeParseTest, ParseDavidNotifySqe_WaitWithNotifyId)
{
    Rt91095StarsNotifySqe sqe;
    memset(&sqe, 0, sizeof(sqe));
    sqe.notifyId = 999;

    SetCurRankId(2);
    EXPECT_NO_THROW(ParseDavidNotifySqe(1, &sqe, false));
    SetCurRankId(0);
}

TEST_F(DeviceSqeParseTest, ParseDavidNotifySqe_RecordWithNotifyId)
{
    Rt91095StarsNotifySqe sqe;
    memset(&sqe, 0, sizeof(sqe));
    sqe.notifyId = 777;

    SetCurRankId(1);
    EXPECT_NO_THROW(ParseDavidNotifySqe(2, &sqe, true));
    SetCurRankId(0);
}

TEST_F(DeviceSqeParseTest, ParseA5SqeFromSqBuffer_SDMATypeWithSqeCnt)
{
    struct halSqCqConfigInfo info;
    memset(&info, 0, sizeof(info));
    info.sqId = 0;
    info.value[0] = 1;
    UpdateSqTail(0, 0);

    uint8_t* sqBuf = nullptr;
    GetSqBufferAddr(&sqBuf);
    if (sqBuf) {
        Rt91095StarsMemcpySqe sqe;
        memset(&sqe, 0, sizeof(sqe));
        sqe.header.type = static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA);
        sqe.opcode = 0;
        sqe.u.strideMode0.lengthMove = 1024;
        sqe.u.strideMode0.srcAddrLow = 0x1000;
        sqe.u.strideMode0.srcAddrHigh = 0;
        sqe.u.strideMode0.dstAddrLow = 0x2000;
        sqe.u.strideMode0.dstAddrHigh = 0;
        memcpy(sqBuf, &sqe, sizeof(sqe));
    }

    EXPECT_NO_THROW(ParseA5SqeFromSqBuffer(0, &info));
}

TEST_F(DeviceSqeParseTest, ParseA5SqeFromSqBuffer_NotifyWaitTypeWithSqeCnt)
{
    struct halSqCqConfigInfo info;
    memset(&info, 0, sizeof(info));
    info.sqId = 0;
    info.value[0] = 1;
    UpdateSqTail(0, 0);

    uint8_t* sqBuf = nullptr;
    GetSqBufferAddr(&sqBuf);
    if (sqBuf) {
        Rt91095StarsNotifySqe sqe;
        memset(&sqe, 0, sizeof(sqe));
        sqe.header.type = static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
        sqe.notifyId = 100;
        memcpy(sqBuf, &sqe, sizeof(sqe));
    }

    EXPECT_NO_THROW(ParseA5SqeFromSqBuffer(0, &info));
}

TEST_F(DeviceSqeParseTest, ParseA5SqeFromSqBuffer_NotifyRecordTypeWithSqeCnt)
{
    struct halSqCqConfigInfo info;
    memset(&info, 0, sizeof(info));
    info.sqId = 0;
    info.value[0] = 1;
    UpdateSqTail(0, 0);

    uint8_t* sqBuf = nullptr;
    GetSqBufferAddr(&sqBuf);
    if (sqBuf) {
        Rt91095StarsNotifySqe sqe;
        memset(&sqe, 0, sizeof(sqe));
        sqe.header.type = static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_RECORD);
        sqe.notifyId = 200;
        memcpy(sqBuf, &sqe, sizeof(sqe));
    }

    EXPECT_NO_THROW(ParseA5SqeFromSqBuffer(0, &info));
}

TEST_F(DeviceSqeParseTest, ParseA5SqeFromSqBuffer_UBDMATypeWithSqeCnt)
{
    struct halSqCqConfigInfo info;
    memset(&info, 0, sizeof(info));
    info.sqId = 0;
    info.value[0] = 1;
    UpdateSqTail(0, 0);

    sim::runtime::RaJetty jetty{};
    jetty.id = 1;
    // 新合同：SQE 的 jettyId1 是 USER_CTL_NORMAL(AICPU) 硬件编号 [5312,9407]；设备按
    // （编号, 创建者 pid, 模式）复合查行。测试进程同时扮演创建者与解析者，创建者 pid
    // 即本进程的父进程（设备视角的 rank host）。
    jetty.jetty_id = HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN;
    jetty.mode = HcclSim::HCCL_VM_JETTY_MODE_USER_CTL_NORMAL;
    jetty.pid = static_cast<uint64_t>(getppid());
    jetty.sqBuffer = 0;
    runnerdb_test::InsertRecord<sim::runtime::RaJetty>(jetty);

    uint8_t* sqBuf = nullptr;
    GetSqBufferAddr(&sqBuf);
    if (sqBuf) {
        Rt91095StarsUbdmaDBmodeSqe sqe;
        memset(&sqe, 0, sizeof(sqe));
        sqe.header.type = static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_UBDMA);
        sqe.jettyId1 = HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN;
        sqe.piValue1 = 1;
        memcpy(sqBuf, &sqe, sizeof(sqe));
    }

    EXPECT_NO_THROW(ParseA5SqeFromSqBuffer(0, &info));
}

TEST_F(DeviceSqeParseTest, ParseA5SqeFromSqBuffer_DefaultTypeWithSqeCnt)
{
    struct halSqCqConfigInfo info;
    memset(&info, 0, sizeof(info));
    info.sqId = 0;
    info.value[0] = 1;
    UpdateSqTail(0, 0);

    uint8_t* sqBuf = nullptr;
    GetSqBufferAddr(&sqBuf);
    if (sqBuf) {
        Rt91095StarsSqeHeader header;
        memset(&header, 0, sizeof(header));
        header.type = 35;
        memcpy(sqBuf, &header, sizeof(header));
    }

    EXPECT_NO_THROW(ParseA5SqeFromSqBuffer(0, &info));
}

TEST_F(DeviceSqeParseTest, ParseA5SqeFromSqBuffer_TailLessThanHeadWithSqeCnt)
{
    struct halSqCqConfigInfo info;
    memset(&info, 0, sizeof(info));
    info.sqId = 0;
    info.value[0] = 5;

    UpdateSqTail(0, 10);

    uint8_t* sqBuf = nullptr;
    GetSqBufferAddr(&sqBuf);
    if (sqBuf) {
        Rt91095StarsSqeHeader header;
        memset(&header, 0, sizeof(header));
        header.type = static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA);
        Rt91095StarsMemcpySqe sqe;
        memset(&sqe, 0, sizeof(sqe));
        sqe.header.type = static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA);
        sqe.opcode = 0;
        sqe.u.strideMode0.lengthMove = 1024;
        sqe.u.strideMode0.srcAddrLow = 0x1000;
        sqe.u.strideMode0.srcAddrHigh = 0;
        sqe.u.strideMode0.dstAddrLow = 0x2000;
        sqe.u.strideMode0.dstAddrHigh = 0;
        memcpy(sqBuf + 10 * HCCL_SQE_SIZE, &sqe, sizeof(sqe));
    }

    EXPECT_NO_THROW(ParseA5SqeFromSqBuffer(0, &info));
}

TEST_F(DeviceSqeParseTest, ParseA5SqeFromSqBuffer_MultipleSqeTypes)
{
    struct halSqCqConfigInfo info;
    memset(&info, 0, sizeof(info));
    info.sqId = 0;
    info.value[0] = 2;
    UpdateSqTail(0, 0);

    uint8_t* sqBuf = nullptr;
    GetSqBufferAddr(&sqBuf);
    if (sqBuf) {
        Rt91095StarsMemcpySqe sqe1;
        memset(&sqe1, 0, sizeof(sqe1));
        sqe1.header.type = static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA);
        sqe1.opcode = 0;
        sqe1.u.strideMode0.lengthMove = 1024;
        sqe1.u.strideMode0.srcAddrLow = 0x1000;
        sqe1.u.strideMode0.srcAddrHigh = 0;
        sqe1.u.strideMode0.dstAddrLow = 0x2000;
        sqe1.u.strideMode0.dstAddrHigh = 0;
        memcpy(sqBuf, &sqe1, sizeof(sqe1));

        Rt91095StarsNotifySqe sqe2;
        memset(&sqe2, 0, sizeof(sqe2));
        sqe2.header.type = static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
        sqe2.notifyId = 100;
        memcpy(sqBuf + HCCL_SQE_SIZE, &sqe2, sizeof(sqe2));
    }

    EXPECT_NO_THROW(ParseA5SqeFromSqBuffer(0, &info));
}

TEST_F(DeviceSqeParseTest, ParseA5SqeFromSqBuffer_NonZeroDevId)
{
    struct halSqCqConfigInfo info;
    memset(&info, 0, sizeof(info));
    info.sqId = 1;
    info.value[0] = 1;
    UpdateSqTail(1, 0);

    uint8_t* sqBuf = nullptr;
    GetSqBufferAddr(&sqBuf);
    if (sqBuf) {
        Rt91095StarsSqeHeader header;
        memset(&header, 0, sizeof(header));
        header.type = static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA);
        memcpy(sqBuf, &header, sizeof(header));
    }

    EXPECT_NO_THROW(ParseA5SqeFromSqBuffer(3, &info));
}

TEST_F(DeviceSqeParseTest, ParseA5SqeFromSqBuffer_SDMAReduceType)
{
    struct halSqCqConfigInfo info;
    memset(&info, 0, sizeof(info));
    info.sqId = 0;
    info.value[0] = 1;
    UpdateSqTail(0, 0);

    uint8_t* sqBuf = nullptr;
    GetSqBufferAddr(&sqBuf);
    if (sqBuf) {
        Rt91095StarsMemcpySqe sqe;
        memset(&sqe, 0, sizeof(sqe));
        sqe.header.type = static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA);
        sqe.opcode = 0x01;
        sqe.u.strideMode0.lengthMove = 512;
        sqe.u.strideMode0.srcAddrLow = 0x1000;
        sqe.u.strideMode0.srcAddrHigh = 0;
        sqe.u.strideMode0.dstAddrLow = 0x2000;
        sqe.u.strideMode0.dstAddrHigh = 0;
        memcpy(sqBuf, &sqe, sizeof(sqe));
    }

    EXPECT_NO_THROW(ParseA5SqeFromSqBuffer(0, &info));
}

TEST_F(DeviceSqeParseTest, GetRmtDeviceIdByEid_MaxEid)
{
    uint8_t eid[URMA_EID_LEN];
    memset(eid, 0xFF, sizeof(eid));
    uint32_t deviceId = 0;
    EXPECT_NO_THROW(GetRmtDeviceIdByEid(eid, deviceId));
}

TEST_F(DeviceSqeParseTest, GetRmtDeviceIdByEid_SpecificIp)
{
    uint8_t eid[URMA_EID_LEN] = {};
    eid[0] = 0x09;
    eid[1] = 0x01;
    eid[2] = 0xA8;
    eid[3] = 0xC0;

    sim::runtime::EndPoint endPoint{};
    endPoint.device_id = 9;
    std::strncpy(endPoint.ip_addr, "192.168.1.9", sizeof(endPoint.ip_addr) - 1);
    // 基线红修复：产品合同已改为按 EID 字节比较（GetEndPointByEid 用
    // Eq(EndPoint::eid, ...)，见 device_sqe_parse_stub.cc 注释——新 EID 不再满足
    // IPv4-compatible 格式，不能用 ip_addr strcmp 匹配）。旧用例只写 ip_addr
    // 未写 eid 字段，在基线即失败。按 GetRmtDeviceIdByEid 相同变换
    // （字节翻转 + IpAddress 编码）写入 eid，使行与查询合同一致。
    HcclSim::Eid remoteEid{};
    for (uint32_t i = 0; i < sizeof(remoteEid.raw); ++i) {
        remoteEid.raw[i] = eid[sizeof(remoteEid.raw) - i - 1];
    }
    HcclSim::IpAddress address(remoteEid);
    const HcclSim::Eid addrEid = address.GetEid();
    std::memcpy(endPoint.eid, addrEid.raw, sizeof(endPoint.eid));
    const uint64_t endPointId = runnerdb_test::InsertRecord<sim::runtime::EndPoint>(endPoint);
    ASSERT_NE(endPointId, 0u);

    uint32_t deviceId = 0;
    EXPECT_TRUE(GetRmtDeviceIdByEid(eid, deviceId));
    EXPECT_EQ(deviceId, 9u);
    EXPECT_TRUE(runnerdb_test::DeleteRecord<sim::runtime::EndPoint>(endPointId));
}

TEST_F(DeviceSqeParseTest, ParseReduceTypeDavid_Zero)
{
    uint8_t result = 0x00;
    HcclReduceOp op = ParseReduceTypeDavid(result);
    EXPECT_EQ(op, HcclReduceOp::HCCL_REDUCE_RESERVED);
}

TEST_F(DeviceSqeParseTest, ParseDataTypeDavid_ZeroInvalid)
{
    uint8_t result = 0x90;
    HcclDataType type = ParseDataTypeDavid(result);
    EXPECT_EQ(type, HcclDataType::HCCL_DATA_TYPE_RESERVED);
}

// ==================== ParseDavidUDMASqe Tests ====================
// The doorbell parse chain uses GetWqebufferByJettyId to lookup RaJetty.sqBuffer from DB.
// These tests cover various opcode paths with valid RaJetty setup.

TEST_F(DeviceSqeParseTest, ParseDavidUDMASqe_WriteOpcodeWithValidWqe)
{
    alignas(64) uint8_t wqeBuffer[HCCL_WQE_SIZE * 4] = {};
    sim::runtime::RaJetty jetty{};
    jetty.id = 1;
    // 新合同：SQE 的 jettyId1 是 USER_CTL_NORMAL(AICPU) 硬件编号 [5312,9407]；设备按
    // （编号, 创建者 pid, 模式）复合查行。测试进程同时扮演创建者与解析者，创建者 pid
    // 即本进程的父进程（设备视角的 rank host）。
    jetty.jetty_id = HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN;
    jetty.mode = HcclSim::HCCL_VM_JETTY_MODE_USER_CTL_NORMAL;
    jetty.pid = static_cast<uint64_t>(getppid());
    jetty.sqBuffer = reinterpret_cast<uint64_t>(wqeBuffer);
    runnerdb_test::InsertRecord<sim::runtime::RaJetty>(jetty);

    UdmaSqeWrite* ubWqeWrite = reinterpret_cast<UdmaSqeWrite*>(wqeBuffer);
    memset(ubWqeWrite, 0, sizeof(UdmaSqeWrite));
    ubWqeWrite->comm.opcode = static_cast<int>(UdmaSqOpcode::UDMA_OPC_WRITE);
    ubWqeWrite->comm.inlineEn = 1;
    ubWqeWrite->comm.rmtAddrLow = 0x1000;
    ubWqeWrite->comm.rmtEid[0] = 0;

    Rt91095StarsUbdmaDBmodeSqe ubSqe;
    memset(&ubSqe, 0, sizeof(ubSqe));
    ubSqe.jettyId1 = HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN;
    ubSqe.piValue1 = 2;

    EXPECT_NO_THROW(ParseUdmaDoorbellSqe(0, &ubSqe));
}

TEST_F(DeviceSqeParseTest, ParseDavidUDMASqe_WriteWithNotifyOpcodeWithValidWqe)
{
    alignas(64) uint8_t wqeBuffer[HCCL_WQE_SIZE * 4] = {};
    sim::runtime::RaJetty jetty{};
    jetty.id = 1;
    // 新合同：SQE 的 jettyId1 是 USER_CTL_NORMAL(AICPU) 硬件编号 [5312,9407]；设备按
    // （编号, 创建者 pid, 模式）复合查行。测试进程同时扮演创建者与解析者，创建者 pid
    // 即本进程的父进程（设备视角的 rank host）。
    jetty.jetty_id = HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN;
    jetty.mode = HcclSim::HCCL_VM_JETTY_MODE_USER_CTL_NORMAL;
    jetty.pid = static_cast<uint64_t>(getppid());
    jetty.sqBuffer = reinterpret_cast<uint64_t>(wqeBuffer);
    runnerdb_test::InsertRecord<sim::runtime::RaJetty>(jetty);

    UdmaSqeWriteWithNotify* ubWqe = reinterpret_cast<UdmaSqeWriteWithNotify*>(wqeBuffer);
    memset(ubWqe, 0, sizeof(UdmaSqeWriteWithNotify));
    ubWqe->comm.opcode = static_cast<int>(UdmaSqOpcode::UDMA_OPC_WRITE_WITH_NOTIFY);
    ubWqe->comm.udfFlag = 0;
    ubWqe->comm.rmtAddrLow = 0x2000;
    ubWqe->comm.rmtEid[0] = 0;
    ubWqe->localU.sge.dataAddrLow = 0x1000;
    ubWqe->localU.sge.length = 256;
    ubWqe->notify.notifyAddrLow = 0x4000;

    Rt91095StarsUbdmaDBmodeSqe ubSqe;
    memset(&ubSqe, 0, sizeof(ubSqe));
    ubSqe.jettyId1 = HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN;
    ubSqe.piValue1 = 2;

    EXPECT_NO_THROW(ParseUdmaDoorbellSqe(0, &ubSqe));
}

TEST_F(DeviceSqeParseTest, ParseDavidUDMASqe_ReadOpcodeWithValidWqe)
{
    alignas(64) uint8_t wqeBuffer[HCCL_WQE_SIZE * 4] = {};
    sim::runtime::RaJetty jetty{};
    jetty.id = 1;
    // 新合同：SQE 的 jettyId1 是 USER_CTL_NORMAL(AICPU) 硬件编号 [5312,9407]；设备按
    // （编号, 创建者 pid, 模式）复合查行。测试进程同时扮演创建者与解析者，创建者 pid
    // 即本进程的父进程（设备视角的 rank host）。
    jetty.jetty_id = HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN;
    jetty.mode = HcclSim::HCCL_VM_JETTY_MODE_USER_CTL_NORMAL;
    jetty.pid = static_cast<uint64_t>(getppid());
    jetty.sqBuffer = reinterpret_cast<uint64_t>(wqeBuffer);
    runnerdb_test::InsertRecord<sim::runtime::RaJetty>(jetty);

    UdmaSqeWrite* ubWqeRead = reinterpret_cast<UdmaSqeWrite*>(wqeBuffer);
    memset(ubWqeRead, 0, sizeof(UdmaSqeWrite));
    ubWqeRead->comm.opcode = static_cast<int>(UdmaSqOpcode::UDMA_OPC_READ);
    ubWqeRead->comm.inlineEn = 0;
    ubWqeRead->comm.udfFlag = 1;
    ubWqeRead->comm.rmtAddrLow = 0x2000;
    ubWqeRead->comm.inlinedata.udfData.reduceOp = 0xA;
    ubWqeRead->comm.inlinedata.udfData.reduceType = 0x7;
    ubWqeRead->u.sge.dataAddrLow = 0x1000;
    ubWqeRead->u.sge.length = 64;

    Rt91095StarsUbdmaDBmodeSqe ubSqe;
    memset(&ubSqe, 0, sizeof(ubSqe));
    ubSqe.jettyId1 = HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN;
    ubSqe.piValue1 = 2;

    EXPECT_NO_THROW(ParseUdmaDoorbellSqe(0, &ubSqe));
}

TEST_F(DeviceSqeParseTest, ParseDavidUDMASqe_UnsupportedOpcodeWithValidWqe)
{
    alignas(64) uint8_t wqeBuffer[HCCL_WQE_SIZE * 4] = {};
    sim::runtime::RaJetty jetty{};
    jetty.id = 1;
    // 新合同：SQE 的 jettyId1 是 USER_CTL_NORMAL(AICPU) 硬件编号 [5312,9407]；设备按
    // （编号, 创建者 pid, 模式）复合查行。测试进程同时扮演创建者与解析者，创建者 pid
    // 即本进程的父进程（设备视角的 rank host）。
    jetty.jetty_id = HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN;
    jetty.mode = HcclSim::HCCL_VM_JETTY_MODE_USER_CTL_NORMAL;
    jetty.pid = static_cast<uint64_t>(getppid());
    jetty.sqBuffer = reinterpret_cast<uint64_t>(wqeBuffer);
    runnerdb_test::InsertRecord<sim::runtime::RaJetty>(jetty);

    UdmaSqeCommon* ubCommon = reinterpret_cast<UdmaSqeCommon*>(wqeBuffer);
    memset(ubCommon, 0, sizeof(UdmaSqeCommon));
    ubCommon->opcode = 99;

    Rt91095StarsUbdmaDBmodeSqe ubSqe;
    memset(&ubSqe, 0, sizeof(ubSqe));
    ubSqe.jettyId1 = HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN;
    ubSqe.piValue1 = 2;

    EXPECT_NO_THROW(ParseUdmaDoorbellSqe(0, &ubSqe));
}

TEST_F(DeviceSqeParseTest, ParseDavidUDMASqe_AdjustCiValWithValidWqe)
{
    alignas(64) uint8_t wqeBuffer[HCCL_WQE_SIZE * 8] = {};
    sim::runtime::RaJetty jetty{};
    jetty.id = 1;
    // 新合同：SQE 的 jettyId1 是 USER_CTL_NORMAL(AICPU) 硬件编号 [5312,9407]；设备按
    // （编号, 创建者 pid, 模式）复合查行。测试进程同时扮演创建者与解析者，创建者 pid
    // 即本进程的父进程（设备视角的 rank host）。
    jetty.jetty_id = HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN;
    jetty.mode = HcclSim::HCCL_VM_JETTY_MODE_USER_CTL_NORMAL;
    jetty.pid = static_cast<uint64_t>(getppid());
    jetty.sqBuffer = reinterpret_cast<uint64_t>(wqeBuffer);
    runnerdb_test::InsertRecord<sim::runtime::RaJetty>(jetty);

    UdmaSqeCommon* ubCommon1 = reinterpret_cast<UdmaSqeCommon*>(wqeBuffer + 1 * HCCL_WQE_SIZE);
    memset(ubCommon1, 0, sizeof(UdmaSqeCommon));
    ubCommon1->opcode = static_cast<int>(UdmaSqOpcode::UDMA_OPC_WRITE);

    UdmaSqeWrite* ubWqeWrite = reinterpret_cast<UdmaSqeWrite*>(wqeBuffer + 2 * HCCL_WQE_SIZE);
    memset(ubWqeWrite, 0, sizeof(UdmaSqeWrite));
    ubWqeWrite->comm.opcode = static_cast<int>(UdmaSqOpcode::UDMA_OPC_WRITE);
    ubWqeWrite->comm.inlineEn = 0;
    ubWqeWrite->comm.udfFlag = 0;
    ubWqeWrite->comm.rmtAddrLow = 0x2000;
    ubWqeWrite->u.sge.dataAddrLow = 0x1000;
    ubWqeWrite->u.sge.length = 128;

    Rt91095StarsUbdmaDBmodeSqe ubSqe;
    memset(&ubSqe, 0, sizeof(ubSqe));
    ubSqe.jettyId1 = HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN;
    ubSqe.piValue1 = 3;

    EXPECT_NO_THROW(ParseUdmaDoorbellSqe(0, &ubSqe));
}

// ==================== ParseA5SqeFromSqBuffer with multiple SQEs Tests ====================

TEST_F(DeviceSqeParseTest, ParseA5SqeFromSqBuffer_MultipleSqeTypesV2)
{
    struct halSqCqConfigInfo info;
    memset(&info, 0, sizeof(info));
    info.sqId = 0;
    info.value[0] = 4; // 4 SQEs
    UpdateSqTail(0, 0);

    uint8_t* sqBuf = nullptr;
    GetSqBufferAddr(&sqBuf);
    if (sqBuf) {
        // SQE 0: SDMA memcpy
        Rt91095StarsMemcpySqe sqe0;
        memset(&sqe0, 0, sizeof(sqe0));
        sqe0.header.type = static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA);
        sqe0.opcode = 0;
        sqe0.u.strideMode0.lengthMove = 1024;
        sqe0.u.strideMode0.srcAddrLow = 0x1000;
        sqe0.u.strideMode0.srcAddrHigh = 0;
        sqe0.u.strideMode0.dstAddrLow = 0x2000;
        sqe0.u.strideMode0.dstAddrHigh = 0;
        memcpy(sqBuf, &sqe0, sizeof(sqe0));

        // SQE 1: NOTIFY_WAIT
        Rt91095StarsNotifySqe sqe1;
        memset(&sqe1, 0, sizeof(sqe1));
        sqe1.header.type = static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
        sqe1.notifyId = 100;
        memcpy(sqBuf + HCCL_SQE_SIZE, &sqe1, sizeof(sqe1));

        // SQE 2: NOTIFY_RECORD
        Rt91095StarsNotifySqe sqe2;
        memset(&sqe2, 0, sizeof(sqe2));
        sqe2.header.type = static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_RECORD);
        sqe2.notifyId = 200;
        memcpy(sqBuf + 2 * HCCL_SQE_SIZE, &sqe2, sizeof(sqe2));

        // SQE 3: UBDMA
        Rt91095StarsUbdmaDBmodeSqe sqe3;
        memset(&sqe3, 0, sizeof(sqe3));
        sqe3.header.type = static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_UBDMA);
        sqe3.jettyId1 = HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN;
        sqe3.piValue1 = 1;
        memcpy(sqBuf + 3 * HCCL_SQE_SIZE, &sqe3, sizeof(sqe3));
    }

    EXPECT_NO_THROW(ParseA5SqeFromSqBuffer(0, &info));
}

TEST_F(DeviceSqeParseTest, ParseA5SqeFromSqBuffer_WrapAroundWithMultipleSqe)
{
    // Test wrap-around with head > tail
    struct halSqCqConfigInfo info;
    memset(&info, 0, sizeof(info));
    info.sqId = 0;
    info.value[0] = 3;   // tail = 3
    UpdateSqTail(0, 10); // head = 10

    uint8_t* sqBuf = nullptr;
    GetSqBufferAddr(&sqBuf);
    if (sqBuf) {
        // Place SDMA SQE at index 10
        Rt91095StarsMemcpySqe sqe0;
        memset(&sqe0, 0, sizeof(sqe0));
        sqe0.header.type = static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA);
        sqe0.opcode = 0;
        sqe0.u.strideMode0.lengthMove = 1024;
        sqe0.u.strideMode0.srcAddrLow = 0x1000;
        sqe0.u.strideMode0.srcAddrHigh = 0;
        sqe0.u.strideMode0.dstAddrLow = 0x2000;
        sqe0.u.strideMode0.dstAddrHigh = 0;
        memcpy(sqBuf + 10 * HCCL_SQE_SIZE, &sqe0, sizeof(sqe0));

        // Place NOTIFY_WAIT SQE at index 11
        Rt91095StarsNotifySqe sqe1;
        memset(&sqe1, 0, sizeof(sqe1));
        sqe1.header.type = static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
        sqe1.notifyId = 100;
        memcpy(sqBuf + 11 * HCCL_SQE_SIZE, &sqe1, sizeof(sqe1));

        // Place NOTIFY_RECORD SQE at index 0 (wrapped)
        Rt91095StarsNotifySqe sqe2;
        memset(&sqe2, 0, sizeof(sqe2));
        sqe2.header.type = static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_RECORD);
        sqe2.notifyId = 200;
        memcpy(sqBuf, &sqe2, sizeof(sqe2));
    }

    EXPECT_NO_THROW(ParseA5SqeFromSqBuffer(0, &info));
}

// ==================== ParseReduceTypeDavid and ParseDataTypeDavid edge cases ====================

TEST_F(DeviceSqeParseTest, ParseReduceTypeDavid_CombinedOpcode)
{
    // opcode with both reduce type (low 4 bits) and data type (high 4 bits)
    uint8_t result = 0x21; // reduce type = 1 (SUM), data type would be 0x20
    HcclReduceOp op = ParseReduceTypeDavid(result);
    EXPECT_EQ(op, HcclReduceOp::HCCL_REDUCE_SUM);
}

TEST_F(DeviceSqeParseTest, ParseDataTypeDavid_CombinedOpcode)
{
    uint8_t result = 0x21; // data type = 0x20 (INT32)
    HcclDataType type = ParseDataTypeDavid(result);
    EXPECT_EQ(type, HcclDataType::HCCL_DATA_TYPE_INT32);
}

// ==================== GetFull64BitAddr edge cases ====================

TEST_F(DeviceSqeParseTest, GetFull64BitAddr_AlternatingBits)
{
    uint32_t lowAddr = 0xAAAAAAAA;
    uint32_t highAddr = 0x55555555;
    uint64_t result = GetFull64BitAddr(lowAddr, highAddr);
    EXPECT_EQ(result, 0x55555555AAAAAAAAULL);
}

TEST_F(DeviceSqeParseTest, GetFull64BitAddr_SingleBit)
{
    uint32_t lowAddr = 0x1;
    uint32_t highAddr = 0x80000000;
    uint64_t result = GetFull64BitAddr(lowAddr, highAddr);
    EXPECT_EQ(result, 0x8000000000000001ULL);
}
