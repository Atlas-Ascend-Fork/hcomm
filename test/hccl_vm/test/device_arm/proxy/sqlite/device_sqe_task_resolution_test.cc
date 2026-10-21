/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// SQE 任务构造的地址解析回归（任务书 v3 §A/回归清单5）：三条数据面路径
// （SDMA memcpy/reduce、UDMA WRITE/READ/reduce、WriteWithNotify 数据段）改用
// ResolveTaskAddress 一次定位后，任务字段必须与输入语义一致；inline notify
// 分支不做地址解析；任一地址解析失败不插入伪造任务（无零偏移/零属主任务）。
// 完整身份链：OpDetailTab（g_currOpDetailId）→ TaskIdentity；远端 EID →
// EndPoint → rmtDeviceId；通信域双成员 → GetCommRankByDeviceId。
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <unistd.h>
#include <vector>

#include "device_sqe_parse_stub.h"
#include "hccl_device_pub.h"
#include "sim_capacity_limits.h"
#include "sim_ip_address.h"
#include "sqe_v82_stub.h"
#include "udma_data_struct_stub.h"
#include "operation_data/operation_data_ops.h"
#include "runtime_state/db_sim_runner_common.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "runtime_state/sim_models.h"
#include "storage/internal/process_storage_context.h"
#include "storage/storage_runtime.h"
#include "storage/storage_session.h"
#include "sqlite/simulation_storage_test_helper.h"

namespace {
const std::string kRunnerDbPath = "/tmp/test_sqe_task_resolution_runner.db";
const std::string kOpDbPath = "/tmp/test_sqe_task_resolution_op.db";

// 固定映射布局（覆盖：本地设备普通块 / 远端设备块 / 本地映射但物理属主=远端）
constexpr uint64_t kLocalMapped = 0xA00000000ULL;
constexpr uint64_t kLocalStart = 0x10000000ULL;
constexpr uint64_t kRmtMapped = 0xB00000000ULL;
constexpr uint64_t kRmtStart = 0x30000000ULL;
constexpr uint64_t kCrossMapped = 0xC00000000ULL; // 本地设备映射、物理属主=远端设备
constexpr uint64_t kCrossStart = 0x50000000ULL;
constexpr uint64_t kBlockSize = 0x10000ULL;
constexpr uint64_t kUnmappedAddr = 0xDEAD0000000ULL; // 无任何 VMB 覆盖

uint64_t InsertPhyOwnedBy(uint64_t deviceId, const char* name)
{
    sim::runtime::PhyMemBlock phyMem{};
    phyMem.device_id = deviceId;
    snprintf(phyMem.name, sizeof(phyMem.name), "%s", name);
    phyMem.size = kBlockSize;
    phyMem.ref_count = 1;
    return runnerdb_test::InsertRecord<sim::runtime::PhyMemBlock>(phyMem);
}

uint64_t InsertDevVmb(uint64_t deviceId, uint64_t startPtr, uint64_t mappedPtr, uint64_t phyMemId)
{
    sim::runtime::VirtualMemBlock virMem{};
    virMem.start_ptr = startPtr;
    virMem.dev_mapped_ptr = mappedPtr;
    virMem.size = kBlockSize;
    virMem.ctx_id = 1;
    virMem.device_id = deviceId;
    virMem.phy_mem_id = phyMemId;
    virMem.owner_pid = 0;
    virMem.src_type = (uint8_t)sim::runtime::VIR_MEM_TYPE_DEV;
    return runnerdb_test::InsertRecord<sim::runtime::VirtualMemBlock>(virMem);
}

uint64_t InsertDevice(uint64_t physicalId)
{
    sim::runtime::Device device{};
    device.server_id = 1;
    device.logic_id = physicalId;
    device.physical_id = physicalId;
    return runnerdb_test::InsertRecord<sim::runtime::Device>(device);
}

/// 把 64 位地址拆进 SQE 的低/高 32 位字段
void SplitAddr(uint64_t addr, uint32_t& low, uint32_t& high)
{
    low = static_cast<uint32_t>(addr & 0xFFFFFFFFULL);
    high = static_cast<uint32_t>(addr >> 32);
}
} // namespace

class DeviceSqeTaskResolutionTest : public testing::Test {
protected:
    uint64_t localDevId{0};
    uint64_t rmtDevId{0};
    uint64_t selfCommId{0};
    uint32_t localRank{3};
    uint32_t rmtRank{7};
    uint8_t sqeEidBytes[HcclSim::URMA_EID_LEN]{};

    void SetUp() override
    {
        ASSERT_TRUE(runnerdb_test::ResetTestSession(kRunnerDbPath, kOpDbPath));

        // 设备与物理块
        localDevId = InsertDevice(0);
        rmtDevId = InsertDevice(1);
        ASSERT_NE(localDevId, 0U);
        ASSERT_NE(rmtDevId, 0U);
        const uint64_t phyLocal = InsertPhyOwnedBy(localDevId, "sqe_phy_local");
        const uint64_t phyRmt = InsertPhyOwnedBy(rmtDevId, "sqe_phy_rmt");
        const uint64_t phyCross = InsertPhyOwnedBy(rmtDevId, "sqe_phy_cross");
        ASSERT_NE(phyLocal, 0U);
        ASSERT_NE(phyRmt, 0U);
        ASSERT_NE(phyCross, 0U);

        // VMB：本地块 / 远端块 / 本地映射但物理属主=远端（MapMem 允许的关联模型）
        ASSERT_NE(InsertDevVmb(localDevId, kLocalStart, kLocalMapped, phyLocal), 0U);
        ASSERT_NE(InsertDevVmb(rmtDevId, kRmtStart, kRmtMapped, phyRmt), 0U);
        ASSERT_NE(InsertDevVmb(localDevId, kCrossStart, kCrossMapped, phyCross), 0U);

        // 通信域：self 成员（主键=identity.commId）+ 远端成员（同名/同 hash）
        sim::runtime::Communicator selfComm{};
        std::strncpy(selfComm.comm_id, "sqe_res_comm", sizeof(selfComm.comm_id) - 1);
        selfComm.comm_hash = 0x5678;
        selfComm.rank_size = 8;
        selfComm.rank_id = localRank;
        selfComm.device_id = localDevId;
        selfCommId = runnerdb_test::InsertRecord<sim::runtime::Communicator>(selfComm);
        ASSERT_NE(selfCommId, 0U);
        sim::runtime::Communicator peerComm{};
        std::strncpy(peerComm.comm_id, "sqe_res_comm", sizeof(peerComm.comm_id) - 1);
        peerComm.comm_hash = 0x5678;
        peerComm.rank_size = 8;
        peerComm.rank_id = rmtRank;
        peerComm.device_id = rmtDevId;
        ASSERT_NE(runnerdb_test::InsertRecord<sim::runtime::Communicator>(peerComm), 0U);

        // 任务身份（OP_DATA_DB.OpDetailTab + g_currOpDetailId）
        sim::operation::OpDetailTab detail{};
        detail.pid = static_cast<uint32_t>(getpid());
        detail.deviceId = static_cast<uint32_t>(localDevId);
        detail.rankId = localRank;
        detail.commId = selfCommId;
        detail.streamId = 9;
        const uint64_t detailId = runnerdb_test::InsertRecord<sim::operation::OpDetailTab>(detail);
        ASSERT_NE(detailId, 0U);
        sim::operation::g_currOpDetailId = static_cast<uint32_t>(detailId);

        // 远端 EID：SQE 字节序与 GetRmtDeviceIdByEid 的翻转+IpAddress 编码保持一致，
        // EndPoint.eid 按产品相同变换写入（不依赖 EID 内部格式假设）
        sqeEidBytes[0] = 0x07;
        sqeEidBytes[1] = 0x01;
        sqeEidBytes[2] = 0xA8;
        sqeEidBytes[3] = 0xC0;
        HcclSim::Eid remoteEid{};
        for (uint32_t i = 0; i < sizeof(remoteEid.raw); ++i) {
            remoteEid.raw[i] = sqeEidBytes[sizeof(remoteEid.raw) - i - 1];
        }
        HcclSim::IpAddress address(remoteEid);
        const HcclSim::Eid addrEid = address.GetEid();
        sim::runtime::EndPoint endPoint{};
        endPoint.device_id = static_cast<uint32_t>(rmtDevId);
        std::memcpy(endPoint.eid, addrEid.raw, sizeof(endPoint.eid));
        ASSERT_NE(runnerdb_test::InsertRecord<sim::runtime::EndPoint>(endPoint), 0U);

        SetCurDeviceKey(static_cast<uint32_t>(localDevId));
    }

    void TearDown() override
    {
        sim::operation::g_currOpDetailId = 0;
        SetCurDeviceKey(0);
        runnerdb_test::CleanUpDatabases(kRunnerDbPath, kOpDbPath);
    }

    /// 采集本进程已插入的全部任务（解码 optaskMeta blob）
    std::vector<HcclTaskMetaData> CollectTasks()
    {
        std::vector<HcclTaskMetaData> tasks;
        std::vector<sim::operation::OpTaskTab> rows;
        if (sim::operation::QueryAllOpTasks(rows) != 0) {
            return tasks;
        }
        for (const auto& row : rows) {
            if (row.optaskMeta.size() != sizeof(HcclTaskMetaData)) {
                continue;
            }
            HcclTaskMetaData meta;
            std::memcpy(&meta, row.optaskMeta.data(), sizeof(meta));
            tasks.push_back(meta);
        }
        return tasks;
    }

    /// 身份字段断言（三条路径共用）：commId/rankId/deviceId 来自 OpDetail 身份
    void ExpectIdentity(const HcclTaskMetaData& task)
    {
        EXPECT_EQ(task.commId, selfCommId);
        EXPECT_EQ(task.rankId, localRank);
        EXPECT_EQ(task.deviceId, localDevId);
    }

    /// 插入 UDMA 路径所需的 RaJetty（AICPU 硬件编号 + 本进程父 pid 属主合同）
    void InsertAicpuJetty(uint64_t sqBuffer)
    {
        sim::runtime::RaJetty jetty{};
        jetty.jetty_id = HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN;
        jetty.mode = HcclSim::HCCL_VM_JETTY_MODE_USER_CTL_NORMAL;
        jetty.pid = static_cast<uint64_t>(getppid());
        jetty.sqBuffer = sqBuffer;
        ASSERT_NE(runnerdb_test::InsertRecord<sim::runtime::RaJetty>(jetty), 0U);
    }

    void FillRmtEid(uint32_t* rmtEidField) { std::memcpy(rmtEidField, sqeEidBytes, HcclSim::URMA_EID_LEN); }

    /// 门铃批 UDMA 解析入口：原 ParseDavidUDMASqe 公共包装删除后，UT 经身份解析直连主链
    void ParseUdmaDoorbellSqe(uint32_t streamId, void* sqeBuf)
    {
        TaskIdentity identity{};
        ASSERT_TRUE(ResolveCurrentTaskIdentity(identity));
        ParseDavidUDMASqeWithIdentity(streamId, sqeBuf, identity);
    }
};

// ==================== SDMA 路径 ====================

TEST_F(DeviceSqeTaskResolutionTest, SdmaMemcpyTaskFieldsMatchInputSemantics)
{
    // src=本地普通块、dst=本地映射但物理属主=远端（跨属主关联）
    Rt91095StarsMemcpySqe sqe;
    memset(&sqe, 0, sizeof(sqe));
    sqe.opcode = 0; // memcpy
    sqe.u.strideMode0.lengthMove = 1024;
    SplitAddr(kLocalMapped + 0x100, sqe.u.strideMode0.srcAddrLow, sqe.u.strideMode0.srcAddrHigh);
    SplitAddr(kCrossMapped + 0x200, sqe.u.strideMode0.dstAddrLow, sqe.u.strideMode0.dstAddrHigh);

    // 旧实现口径（翻译+全局反查）：新任务字段必须与其一致（回归1 的 SQE 级对照）
    SetCurDeviceKey(static_cast<uint32_t>(localDevId));
    const uint64_t legacySrc = TransLocalAddrToVirtual(kLocalMapped + 0x100);
    const uint32_t legacySrcDev = GetDeviceIdByDevAddr(legacySrc);
    const uint64_t legacyDst = TransLocalAddrToVirtual(kCrossMapped + 0x200);
    const uint32_t legacyDstDev = GetDeviceIdByDevAddr(legacyDst);

    ParseDavidSDMASqe(5, &sqe);

    const auto tasks = CollectTasks();
    ASSERT_EQ(tasks.size(), 1U);
    const auto& task = tasks[0];
    ExpectIdentity(task);
    EXPECT_EQ(task.taskType, HccLTaskMetaType::MEM_CPY);
    EXPECT_EQ(task.streamId, 5U);
    EXPECT_EQ(task.jettyId, UINT32_MAX);
    EXPECT_EQ(task.taskData.transMem.srcOffset, kLocalStart + 0x100);
    EXPECT_EQ(task.taskData.transMem.dstOffset, kCrossStart + 0x200);
    EXPECT_EQ(task.taskData.transMem.srcDeviceId, static_cast<uint32_t>(localDevId));
    EXPECT_EQ(task.taskData.transMem.dstDeviceId, static_cast<uint32_t>(rmtDevId));
    EXPECT_EQ(task.taskData.transMem.len, 1024U);
    // 与旧实现输出一致
    EXPECT_EQ(task.taskData.transMem.srcOffset, legacySrc);
    EXPECT_EQ(task.taskData.transMem.srcDeviceId, legacySrcDev);
    EXPECT_EQ(task.taskData.transMem.dstOffset, legacyDst);
    EXPECT_EQ(task.taskData.transMem.dstDeviceId, legacyDstDev);
}

TEST_F(DeviceSqeTaskResolutionTest, SdmaReduceTaskFieldsMatchInputSemantics)
{
    Rt91095StarsMemcpySqe sqe;
    memset(&sqe, 0, sizeof(sqe));
    sqe.opcode = 0x21; // SUM + INT32
    sqe.u.strideMode0.lengthMove = 512;
    SplitAddr(kLocalMapped + 0x10, sqe.u.strideMode0.srcAddrLow, sqe.u.strideMode0.srcAddrHigh);
    SplitAddr(kLocalMapped + 0x2000, sqe.u.strideMode0.dstAddrLow, sqe.u.strideMode0.dstAddrHigh);

    ParseDavidSDMASqe(6, &sqe);

    const auto tasks = CollectTasks();
    ASSERT_EQ(tasks.size(), 1U);
    const auto& task = tasks[0];
    ExpectIdentity(task);
    EXPECT_EQ(task.taskType, HccLTaskMetaType::REDUCE);
    EXPECT_EQ(task.taskData.reduce.reduceOp, HcclReduceOp::HCCL_REDUCE_SUM);
    EXPECT_EQ(task.taskData.reduce.dataType, HcclDataType::HCCL_DATA_TYPE_INT32);
    EXPECT_EQ(task.taskData.reduce.srcOffset, kLocalStart + 0x10);
    EXPECT_EQ(task.taskData.reduce.dstOffset, kLocalStart + 0x2000);
    EXPECT_EQ(task.taskData.reduce.srcDeviceId, static_cast<uint32_t>(localDevId));
    EXPECT_EQ(task.taskData.reduce.dstDeviceId, static_cast<uint32_t>(localDevId));
    EXPECT_EQ(task.taskData.reduce.dataCount, 512U);
    EXPECT_EQ(task.streamId, 6U);
}

TEST_F(DeviceSqeTaskResolutionTest, SdmaUnmappedSrcInsertsNoTask)
{
    // 解析失败：沿现有错误出口返回，不插入零偏移/零属主伪造任务
    Rt91095StarsMemcpySqe sqe;
    memset(&sqe, 0, sizeof(sqe));
    sqe.opcode = 0;
    sqe.u.strideMode0.lengthMove = 64;
    SplitAddr(kUnmappedAddr, sqe.u.strideMode0.srcAddrLow, sqe.u.strideMode0.srcAddrHigh);
    SplitAddr(kLocalMapped + 0x100, sqe.u.strideMode0.dstAddrLow, sqe.u.strideMode0.dstAddrHigh);

    ParseDavidSDMASqe(7, &sqe);

    EXPECT_TRUE(CollectTasks().empty());
}

TEST_F(DeviceSqeTaskResolutionTest, SdmaUnmappedDstInsertsNoTask)
{
    Rt91095StarsMemcpySqe sqe;
    memset(&sqe, 0, sizeof(sqe));
    sqe.opcode = 0;
    sqe.u.strideMode0.lengthMove = 64;
    SplitAddr(kLocalMapped + 0x100, sqe.u.strideMode0.srcAddrLow, sqe.u.strideMode0.srcAddrHigh);
    SplitAddr(kUnmappedAddr, sqe.u.strideMode0.dstAddrLow, sqe.u.strideMode0.dstAddrHigh);

    ParseDavidSDMASqe(7, &sqe);

    EXPECT_TRUE(CollectTasks().empty());
}

// ==================== UDMA WRITE / READ / reduce 路径 ====================

TEST_F(DeviceSqeTaskResolutionTest, UdmaWriteTaskSrcLocalDstRemote)
{
    alignas(64) uint8_t wqeBuffer[HCCL_WQE_SIZE * 4] = {};
    InsertAicpuJetty(reinterpret_cast<uint64_t>(wqeBuffer));

    auto* wqe = reinterpret_cast<UdmaSqeWrite*>(wqeBuffer);
    wqe->comm.opcode = static_cast<int>(UdmaSqOpcode::UDMA_OPC_WRITE);
    wqe->comm.inlineEn = 0;
    wqe->comm.udfFlag = 0;
    FillRmtEid(wqe->comm.rmtEid);
    SplitAddr(kRmtMapped + 0x300, wqe->comm.rmtAddrLow, wqe->comm.rmtAddrHigh);
    SplitAddr(kLocalMapped + 0x40, wqe->u.sge.dataAddrLow, wqe->u.sge.dataAddrHigh);
    wqe->u.sge.length = 128;

    Rt91095StarsUbdmaDBmodeSqe ubSqe;
    memset(&ubSqe, 0, sizeof(ubSqe));
    ubSqe.jettyId1 = HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN;
    ubSqe.piValue1 = 1;

    ParseUdmaDoorbellSqe(11, &ubSqe);

    const auto tasks = CollectTasks();
    ASSERT_EQ(tasks.size(), 1U);
    const auto& task = tasks[0];
    ExpectIdentity(task);
    EXPECT_EQ(task.taskType, HccLTaskMetaType::MEM_CPY);
    EXPECT_EQ(task.streamId, 11U);
    EXPECT_EQ(task.jettyId, HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN);
    EXPECT_EQ(0, std::memcmp(task.rmEid, sqeEidBytes, HcclSim::URMA_EID_LEN));
    // WRITE 正向：src=本地结果、dst=远端结果
    EXPECT_EQ(task.taskData.transMem.srcOffset, kLocalStart + 0x40);
    EXPECT_EQ(task.taskData.transMem.srcDeviceId, static_cast<uint32_t>(localDevId));
    EXPECT_EQ(task.taskData.transMem.dstOffset, kRmtStart + 0x300);
    EXPECT_EQ(task.taskData.transMem.dstDeviceId, static_cast<uint32_t>(rmtDevId));
    EXPECT_EQ(task.taskData.transMem.len, 128U);
}

TEST_F(DeviceSqeTaskResolutionTest, UdmaReadTaskSrcRemoteDstLocal)
{
    alignas(64) uint8_t wqeBuffer[HCCL_WQE_SIZE * 4] = {};
    InsertAicpuJetty(reinterpret_cast<uint64_t>(wqeBuffer));

    auto* wqe = reinterpret_cast<UdmaSqeWrite*>(wqeBuffer);
    wqe->comm.opcode = static_cast<int>(UdmaSqOpcode::UDMA_OPC_READ);
    wqe->comm.inlineEn = 0;
    wqe->comm.udfFlag = 0;
    FillRmtEid(wqe->comm.rmtEid);
    SplitAddr(kRmtMapped + 0x400, wqe->comm.rmtAddrLow, wqe->comm.rmtAddrHigh);
    SplitAddr(kLocalMapped + 0x50, wqe->u.sge.dataAddrLow, wqe->u.sge.dataAddrHigh);
    wqe->u.sge.length = 256;

    Rt91095StarsUbdmaDBmodeSqe ubSqe;
    memset(&ubSqe, 0, sizeof(ubSqe));
    ubSqe.jettyId1 = HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN;
    ubSqe.piValue1 = 1;

    ParseUdmaDoorbellSqe(12, &ubSqe);

    const auto tasks = CollectTasks();
    ASSERT_EQ(tasks.size(), 1U);
    const auto& task = tasks[0];
    ExpectIdentity(task);
    EXPECT_EQ(task.taskType, HccLTaskMetaType::MEM_CPY);
    // READ 反向：src=远端结果、dst=本地结果
    EXPECT_EQ(task.taskData.transMem.srcOffset, kRmtStart + 0x400);
    EXPECT_EQ(task.taskData.transMem.srcDeviceId, static_cast<uint32_t>(rmtDevId));
    EXPECT_EQ(task.taskData.transMem.dstOffset, kLocalStart + 0x50);
    EXPECT_EQ(task.taskData.transMem.dstDeviceId, static_cast<uint32_t>(localDevId));
    EXPECT_EQ(task.taskData.transMem.len, 256U);
}

TEST_F(DeviceSqeTaskResolutionTest, UdmaWriteReduceTaskFields)
{
    alignas(64) uint8_t wqeBuffer[HCCL_WQE_SIZE * 4] = {};
    InsertAicpuJetty(reinterpret_cast<uint64_t>(wqeBuffer));

    auto* wqe = reinterpret_cast<UdmaSqeWrite*>(wqeBuffer);
    wqe->comm.opcode = static_cast<int>(UdmaSqOpcode::UDMA_OPC_WRITE);
    wqe->comm.inlineEn = 0;
    wqe->comm.udfFlag = 1;
    wqe->comm.inlinedata.udfData.reduceOp = 0xA;   // SUM
    wqe->comm.inlinedata.udfData.reduceType = 0x7; // FP32
    FillRmtEid(wqe->comm.rmtEid);
    SplitAddr(kRmtMapped + 0x500, wqe->comm.rmtAddrLow, wqe->comm.rmtAddrHigh);
    SplitAddr(kLocalMapped + 0x60, wqe->u.sge.dataAddrLow, wqe->u.sge.dataAddrHigh);
    wqe->u.sge.length = 512;

    Rt91095StarsUbdmaDBmodeSqe ubSqe;
    memset(&ubSqe, 0, sizeof(ubSqe));
    ubSqe.jettyId1 = HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN;
    ubSqe.piValue1 = 1;

    ParseUdmaDoorbellSqe(13, &ubSqe);

    const auto tasks = CollectTasks();
    ASSERT_EQ(tasks.size(), 1U);
    const auto& task = tasks[0];
    ExpectIdentity(task);
    EXPECT_EQ(task.taskType, HccLTaskMetaType::REDUCE);
    EXPECT_EQ(task.taskData.reduce.reduceOp, HcclReduceOp::HCCL_REDUCE_SUM);
    EXPECT_EQ(task.taskData.reduce.dataType, HcclDataType::HCCL_DATA_TYPE_FP32);
    EXPECT_EQ(task.taskData.reduce.srcOffset, kLocalStart + 0x60);
    EXPECT_EQ(task.taskData.reduce.dstOffset, kRmtStart + 0x500);
    EXPECT_EQ(task.taskData.reduce.srcDeviceId, static_cast<uint32_t>(localDevId));
    EXPECT_EQ(task.taskData.reduce.dstDeviceId, static_cast<uint32_t>(rmtDevId));
    EXPECT_EQ(task.taskData.reduce.dataCount, 512U);
}

TEST_F(DeviceSqeTaskResolutionTest, UdmaWriteRemoteUnmappedInsertsNoTask)
{
    // 远端地址无映射（EID/rank 解析成功）：不插入数据任务
    alignas(64) uint8_t wqeBuffer[HCCL_WQE_SIZE * 4] = {};
    InsertAicpuJetty(reinterpret_cast<uint64_t>(wqeBuffer));

    auto* wqe = reinterpret_cast<UdmaSqeWrite*>(wqeBuffer);
    wqe->comm.opcode = static_cast<int>(UdmaSqOpcode::UDMA_OPC_WRITE);
    wqe->comm.inlineEn = 0;
    wqe->comm.udfFlag = 0;
    FillRmtEid(wqe->comm.rmtEid);
    SplitAddr(kUnmappedAddr, wqe->comm.rmtAddrLow, wqe->comm.rmtAddrHigh);
    SplitAddr(kLocalMapped + 0x40, wqe->u.sge.dataAddrLow, wqe->u.sge.dataAddrHigh);
    wqe->u.sge.length = 128;

    Rt91095StarsUbdmaDBmodeSqe ubSqe;
    memset(&ubSqe, 0, sizeof(ubSqe));
    ubSqe.jettyId1 = HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN;
    ubSqe.piValue1 = 1;

    ParseUdmaDoorbellSqe(14, &ubSqe);

    EXPECT_TRUE(CollectTasks().empty());
}

TEST_F(DeviceSqeTaskResolutionTest, UdmaInlineNotifySkipsAddressResolution)
{
    // inline notify 原分支不做地址解析：sge/rmt 地址均未映射，仍产出 NOTIFY_RECORD，
    // notify/rank 逻辑与身份字段不动
    alignas(64) uint8_t wqeBuffer[HCCL_WQE_SIZE * 4] = {};
    InsertAicpuJetty(reinterpret_cast<uint64_t>(wqeBuffer));

    constexpr uint64_t kNotifyAddr = 0x12345;
    auto* wqe = reinterpret_cast<UdmaSqeWrite*>(wqeBuffer);
    wqe->comm.opcode = static_cast<int>(UdmaSqOpcode::UDMA_OPC_WRITE);
    wqe->comm.inlineEn = 1;
    FillRmtEid(wqe->comm.rmtEid);
    SplitAddr(kNotifyAddr, wqe->comm.rmtAddrLow, wqe->comm.rmtAddrHigh);
    SplitAddr(kUnmappedAddr, wqe->u.sge.dataAddrLow, wqe->u.sge.dataAddrHigh);
    wqe->u.sge.length = 8;

    Rt91095StarsUbdmaDBmodeSqe ubSqe;
    memset(&ubSqe, 0, sizeof(ubSqe));
    ubSqe.jettyId1 = HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN;
    ubSqe.piValue1 = 1;

    ParseUdmaDoorbellSqe(15, &ubSqe);

    const auto tasks = CollectTasks();
    ASSERT_EQ(tasks.size(), 1U);
    const auto& task = tasks[0];
    ExpectIdentity(task);
    EXPECT_EQ(task.taskType, HccLTaskMetaType::NOTIFY_RECORD);
    EXPECT_EQ(task.taskData.notify.notifyId, HcclSim::MakeNotifyTaskId(rmtRank, static_cast<uint32_t>(kNotifyAddr)));
    EXPECT_EQ(task.taskData.notify.srcDeviceId, static_cast<uint32_t>(localDevId));
    EXPECT_EQ(task.taskData.notify.dstDeviceId, static_cast<uint32_t>(rmtDevId));
}

// ==================== WriteWithNotify 路径 ====================

TEST_F(DeviceSqeTaskResolutionTest, WriteWithNotifyEmitsDataTaskThenNotifyTask)
{
    alignas(64) uint8_t wqeBuffer[HCCL_WQE_SIZE * 4] = {};
    InsertAicpuJetty(reinterpret_cast<uint64_t>(wqeBuffer));

    constexpr uint64_t kNotifyAddr = 0x777;
    auto* wqe = reinterpret_cast<UdmaSqeWriteWithNotify*>(wqeBuffer);
    wqe->comm.opcode = static_cast<int>(UdmaSqOpcode::UDMA_OPC_WRITE_WITH_NOTIFY);
    wqe->comm.udfFlag = 0;
    FillRmtEid(wqe->comm.rmtEid);
    SplitAddr(kRmtMapped + 0x600, wqe->comm.rmtAddrLow, wqe->comm.rmtAddrHigh);
    SplitAddr(kNotifyAddr, wqe->notify.notifyAddrLow, wqe->notify.notifyAddrHigh);
    SplitAddr(kLocalMapped + 0x70, wqe->localU.sge.dataAddrLow, wqe->localU.sge.dataAddrHigh);
    wqe->localU.sge.length = 256;

    Rt91095StarsUbdmaDBmodeSqe ubSqe;
    memset(&ubSqe, 0, sizeof(ubSqe));
    ubSqe.jettyId1 = HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN;
    ubSqe.piValue1 = 1;

    ParseUdmaDoorbellSqe(16, &ubSqe);

    const auto tasks = CollectTasks();
    ASSERT_EQ(tasks.size(), 2U);
    // 任务顺序不变：先数据段后 NOTIFY_RECORD
    const auto& data = tasks[0];
    ExpectIdentity(data);
    EXPECT_EQ(data.taskType, HccLTaskMetaType::MEM_CPY);
    EXPECT_EQ(data.streamId, 16U);
    EXPECT_EQ(data.jettyId, HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN);
    EXPECT_EQ(data.taskData.transMem.srcOffset, kLocalStart + 0x70);
    EXPECT_EQ(data.taskData.transMem.srcDeviceId, static_cast<uint32_t>(localDevId));
    EXPECT_EQ(data.taskData.transMem.dstOffset, kRmtStart + 0x600);
    EXPECT_EQ(data.taskData.transMem.dstDeviceId, static_cast<uint32_t>(rmtDevId));
    EXPECT_EQ(data.taskData.transMem.len, 256U);

    const auto& notify = tasks[1];
    ExpectIdentity(notify);
    EXPECT_EQ(notify.taskType, HccLTaskMetaType::NOTIFY_RECORD);
    EXPECT_EQ(notify.taskData.notify.notifyId, HcclSim::MakeNotifyTaskId(rmtRank, static_cast<uint32_t>(kNotifyAddr)));
    EXPECT_EQ(notify.taskData.notify.srcDeviceId, static_cast<uint32_t>(localDevId));
    EXPECT_EQ(notify.taskData.notify.dstDeviceId, static_cast<uint32_t>(rmtDevId));
    EXPECT_EQ(0, std::memcmp(notify.rmEid, sqeEidBytes, HcclSim::URMA_EID_LEN));
}

TEST_F(DeviceSqeTaskResolutionTest, WriteWithNotifyReduceDataTaskFields)
{
    alignas(64) uint8_t wqeBuffer[HCCL_WQE_SIZE * 4] = {};
    InsertAicpuJetty(reinterpret_cast<uint64_t>(wqeBuffer));

    auto* wqe = reinterpret_cast<UdmaSqeWriteWithNotify*>(wqeBuffer);
    wqe->comm.opcode = static_cast<int>(UdmaSqOpcode::UDMA_OPC_WRITE_WITH_NOTIFY);
    wqe->comm.udfFlag = 1;
    wqe->comm.inlinedata.udfData.reduceOp = 0x8;   // MAX
    wqe->comm.inlinedata.udfData.reduceType = 0x0; // INT8
    FillRmtEid(wqe->comm.rmtEid);
    SplitAddr(kRmtMapped + 0x700, wqe->comm.rmtAddrLow, wqe->comm.rmtAddrHigh);
    SplitAddr(0x888, wqe->notify.notifyAddrLow, wqe->notify.notifyAddrHigh);
    SplitAddr(kLocalMapped + 0x80, wqe->localU.sge.dataAddrLow, wqe->localU.sge.dataAddrHigh);
    wqe->localU.sge.length = 64;

    Rt91095StarsUbdmaDBmodeSqe ubSqe;
    memset(&ubSqe, 0, sizeof(ubSqe));
    ubSqe.jettyId1 = HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN;
    ubSqe.piValue1 = 1;

    ParseUdmaDoorbellSqe(17, &ubSqe);

    const auto tasks = CollectTasks();
    ASSERT_EQ(tasks.size(), 2U);
    const auto& data = tasks[0];
    EXPECT_EQ(data.taskType, HccLTaskMetaType::REDUCE);
    EXPECT_EQ(data.taskData.reduce.reduceOp, HcclReduceOp::HCCL_REDUCE_MAX);
    EXPECT_EQ(data.taskData.reduce.dataType, HcclDataType::HCCL_DATA_TYPE_INT8);
    EXPECT_EQ(data.taskData.reduce.srcOffset, kLocalStart + 0x80);
    EXPECT_EQ(data.taskData.reduce.dstOffset, kRmtStart + 0x700);
    EXPECT_EQ(data.taskData.reduce.srcDeviceId, static_cast<uint32_t>(localDevId));
    EXPECT_EQ(data.taskData.reduce.dstDeviceId, static_cast<uint32_t>(rmtDevId));
    EXPECT_EQ(data.taskData.reduce.dataCount, 64U);
    EXPECT_EQ(tasks[1].taskType, HccLTaskMetaType::NOTIFY_RECORD);
}

TEST_F(DeviceSqeTaskResolutionTest, WriteWithNotifyLocalUnmappedInsertsNoTaskAtAll)
{
    // 数据段本地地址解析失败：数据任务与 notify 任务都不得插入（返回在两者之前）
    alignas(64) uint8_t wqeBuffer[HCCL_WQE_SIZE * 4] = {};
    InsertAicpuJetty(reinterpret_cast<uint64_t>(wqeBuffer));

    auto* wqe = reinterpret_cast<UdmaSqeWriteWithNotify*>(wqeBuffer);
    wqe->comm.opcode = static_cast<int>(UdmaSqOpcode::UDMA_OPC_WRITE_WITH_NOTIFY);
    wqe->comm.udfFlag = 0;
    FillRmtEid(wqe->comm.rmtEid);
    SplitAddr(kRmtMapped + 0x800, wqe->comm.rmtAddrLow, wqe->comm.rmtAddrHigh);
    SplitAddr(0x999, wqe->notify.notifyAddrLow, wqe->notify.notifyAddrHigh);
    SplitAddr(kUnmappedAddr, wqe->localU.sge.dataAddrLow, wqe->localU.sge.dataAddrHigh);
    wqe->localU.sge.length = 32;

    Rt91095StarsUbdmaDBmodeSqe ubSqe;
    memset(&ubSqe, 0, sizeof(ubSqe));
    ubSqe.jettyId1 = HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN;
    ubSqe.piValue1 = 1;

    ParseUdmaDoorbellSqe(18, &ubSqe);

    EXPECT_TRUE(CollectTasks().empty());
}

// ==================== 后端错误注入 ====================

TEST_F(DeviceSqeTaskResolutionTest, BackendErrorDuringParseInsertsNoZeroValueTask)
{
    // 注入后端错误（会话关闭）：解析失败关闭，不插入伪造任务、不崩溃
    const auto closed = HcclSim::Storage::StorageRuntime::CloseProcessSession();
    ASSERT_TRUE(closed.ok()) << closed.diagnostic;

    Rt91095StarsMemcpySqe sqe;
    memset(&sqe, 0, sizeof(sqe));
    sqe.opcode = 0;
    sqe.u.strideMode0.lengthMove = 64;
    SplitAddr(kLocalMapped + 0x100, sqe.u.strideMode0.srcAddrLow, sqe.u.strideMode0.srcAddrHigh);
    SplitAddr(kLocalMapped + 0x2000, sqe.u.strideMode0.dstAddrLow, sqe.u.strideMode0.dstAddrHigh);
    ParseDavidSDMASqe(19, &sqe);

    // 恢复会话后确认无任务写入（TearDown 也需要活会话）
    ASSERT_TRUE(runnerdb_test::ResetTestSession(kRunnerDbPath, kOpDbPath));
    EXPECT_TRUE(CollectTasks().empty());
}
