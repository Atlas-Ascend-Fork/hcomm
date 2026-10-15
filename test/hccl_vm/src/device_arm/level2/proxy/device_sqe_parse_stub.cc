/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 日志染色: 模块 tag (须在 include sim_log.h 之前)
#define HCCL_VM_MODULE "DEVICE"

#include "device_sqe_parse_stub.h"

#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "hccl_task_collection.h"
#include "operation_data/operation_data_ops.h"
#include "runtime_state/db_sim_runner_common.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "sim_capacity_limits.h"
#include "sim_log.h"
#include "sim_ip_address.h"
#include "sqe_v82_stub.h"
#include "store_sim_run_mode.h"
#include "udma_data_struct_stub.h"

constexpr int SHIFT_BIT32 = 32;
static uint32_t g_currSqeGroupId = 0;
namespace {
bool ResolveCurrentTaskIdentity(TaskIdentity& identity)
{
    if (sim::operation::g_currOpDetailId == 0) {
        HCCL_VM_ERROR("cannot resolve task identity for opDetailId={}", sim::operation::g_currOpDetailId);
        return false;
    }
    // 同一 kernel 的 SQE 解析共用同一 opDetailId, 单条目缓存避免每条 SQE 重复查库
    static thread_local uint32_t s_cachedOpDetailId{0};
    static thread_local TaskIdentity s_cachedIdentity{};
    if (sim::operation::g_currOpDetailId == s_cachedOpDetailId) {
        identity = s_cachedIdentity;
        return true;
    }
    if (sim::operation::QueryOpDetailIdentity(
            sim::operation::g_currOpDetailId, identity.commId, identity.rankId, identity.deviceId)
            != 0
        || identity.commId == 0) {
        HCCL_VM_ERROR("cannot resolve task identity for opDetailId={}", sim::operation::g_currOpDetailId);
        return false;
    }
    s_cachedOpDetailId = sim::operation::g_currOpDetailId;
    s_cachedIdentity = identity;
    return true;
}

void SetTaskIdentity(HcclTaskMetaData& task, const TaskIdentity& identity)
{
    task.commId = identity.commId;
    task.rankId = identity.rankId;
    task.deviceId = identity.deviceId;
}

// WQE 内 rmtEid 为 uint32[4] 小端字存储, 16字节整体翻转后与 ranktable.json
// 十六进制串字节序(EndPoint.eid/RankEidMap 同序)一致, rmEid 统一存该序
void CopyRmtEidFromWqe(uint8_t* dst, const uint8_t* src)
{
    for (uint32_t i = 0; i < HcclSim::URMA_EID_LEN; ++i) {
        dst[i] = src[HcclSim::URMA_EID_LEN - i - 1];
    }
}

// 评估模式: deviceId字段承载通信域rankId
void SetRemoteNotifyEndpointsEstimate(
    HcclTaskMetaData& task, const TaskIdentity& identity, uint32_t remoteDeviceId, uint32_t remoteRankId)
{
    const uint32_t localDeviceId = GetCurDeviceKey();
    task.taskData.notify.srcDeviceId = identity.rankId;
    task.taskData.notify.dstDeviceId = remoteRankId;
    HCCL_VM_INFO(
        "record Notify endpoints, localDeviceId={}, remoteDeviceId={}, srcDeviceId={}, dstDeviceId={}, notifyId={}",
        localDeviceId, remoteDeviceId, identity.rankId, remoteRankId, task.taskData.notify.notifyId);
}

// 非评估模式: deviceId字段表达针织deviceId
void SetNotifySourceDevice(HcclTaskMetaData& task)
{
    task.taskData.notify.srcDeviceId = GetCurDeviceKey();
    task.taskData.notify.dstDeviceId = 0;
}

void SetRemoteNotifyEndpoints(HcclTaskMetaData& task, uint32_t remoteDeviceId)
{
    const uint32_t localDeviceId = GetCurDeviceKey();
    task.taskData.notify.srcDeviceId = localDeviceId;
    task.taskData.notify.dstDeviceId = remoteDeviceId;
    HCCL_VM_INFO(
        "record Notify endpoints, localDeviceId={}, remoteDeviceId={}, notifyId={}", localDeviceId, remoteDeviceId,
        task.taskData.notify.notifyId);
}

} // namespace

void ParseA5SqeFromSqBuffer(uint32_t devId, struct halSqCqConfigInfo* info)
{
    TaskIdentity identity{};
    if (!ResolveCurrentTaskIdentity(identity)) {
        HCCL_VM_ERROR("cannot parse SQE without current task identity");
        return;
    }
    // 运行模式，区分评估/非评估模式路径
    const bool estimateMode = sim::IsEstimateMode();
    uint32_t streamId = info->sqId;
    int tail = info->value[0];
    int head = GetSqTail(streamId);
    UpdateSqTail(streamId, tail);

    uint8_t* sqBuffer = nullptr;
    GetSqBufferAddr(&sqBuffer);

    // 计算本轮下发的SQE的数量
    int sqeCnt = (HCCL_SQE_MAX_CNT + tail - head) % HCCL_SQE_MAX_CNT;
    //  临时缓冲区，用于存放从sqBuffer拷贝的数据
    uint8_t tempBuffer[HCCL_SQE_SIZE * HCCL_SQE_MAX_CNT];
    if (tail >= head) {
        // 数据未绕圈，直接拷贝
        memcpy(tempBuffer, sqBuffer + head * HCCL_SQE_SIZE, sqeCnt * HCCL_SQE_SIZE);
    } else {
        // 数据绕圈，分两次拷贝
        int firstPart = HCCL_SQE_MAX_CNT - head; // 从head到缓冲区  末尾的拷贝数量
        int secondPart = tail;                   // 从缓冲区开头到tail的拷贝数量
        memcpy(tempBuffer, sqBuffer + head * HCCL_SQE_SIZE, firstPart * HCCL_SQE_SIZE);
        memcpy(tempBuffer + firstPart * HCCL_SQE_SIZE, sqBuffer, secondPart * HCCL_SQE_SIZE);
    }

    for (int i = 0; i < sqeCnt; i++) {
        int sqOffIndex = i;
        void* sqeBuf = static_cast<uint8_t*>(tempBuffer) + sqOffIndex * HCCL_SQE_SIZE;
        Rt91095StarsSqeHeader* header = reinterpret_cast<Rt91095StarsSqeHeader*>(sqeBuf);
        switch (header->type) {
            case static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA): {
                estimateMode ? ParseDavidSDMASqeEstimate(streamId, sqeBuf, identity) :
                               ParseDavidSDMASqeWithIdentity(streamId, sqeBuf, identity);
                break;
            }
            case static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT): {
                estimateMode ? ParseDavidNotifySqeEstimate(streamId, sqeBuf, false, identity) :
                               ParseDavidNotifySqeWithIdentity(streamId, sqeBuf, false, identity);
                break;
            }
            case static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_RECORD): {
                estimateMode ? ParseDavidNotifySqeEstimate(streamId, sqeBuf, true, identity) :
                               ParseDavidNotifySqeWithIdentity(streamId, sqeBuf, true, identity);
                break;
            }
            case static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_UBDMA): {
                ParseDavidUDMASqeWithIdentity(streamId, sqeBuf, identity);
                break;
            }
            default: {
                HCCL_VM_ERROR("not support sqe type[{}].", static_cast<uint32_t>(header->type));
                break;
            }
        }
    }
}

// 评估模式: 保留原始设备地址, deviceId字段承载通信域rankId
void ParseDavidSDMASqeEstimate(uint32_t streamId, void* sqeBuf, const TaskIdentity& identity)
{
    Rt91095StarsMemcpySqe* sqe = reinterpret_cast<Rt91095StarsMemcpySqe*>(sqeBuf);
    HcclTaskMetaData taskMeta;
    SetTaskIdentity(taskMeta, identity);
    taskMeta.streamId = streamId;
    taskMeta.jettyId = UINT32_MAX;

    uint64_t length = sqe->u.strideMode0.lengthMove;
    uint64_t srcOffset = GetFull64BitAddr(sqe->u.strideMode0.srcAddrLow, sqe->u.strideMode0.srcAddrHigh);
    uint64_t dstOffset = GetFull64BitAddr(sqe->u.strideMode0.dstAddrLow, sqe->u.strideMode0.dstAddrHigh);

    if (sqe->opcode == 0) { // 表示是memcpy
        taskMeta.taskType = HccLTaskMetaType::MEM_CPY;
        taskMeta.profInfo = HcclTaskProfInfo(
            ProfTaskParamType::TASK_SDMA, ProfLinkType::LINK_ONCHIP,
            ProfTransportType::TRANSPORT_LOCAL); // 评估模式 src/dst 同本设备
        taskMeta.taskData.transMem.srcDeviceId = identity.rankId;
        taskMeta.taskData.transMem.srcOffset = srcOffset;
        taskMeta.taskData.transMem.dstDeviceId = identity.rankId;
        taskMeta.taskData.transMem.dstOffset = dstOffset;
        taskMeta.taskData.transMem.len = length;
        InsertTaskToCollectionDev(&taskMeta);
        PrintTaskMetaData(taskMeta);
    } else { // reduce
        taskMeta.taskType = HccLTaskMetaType::REDUCE;
        taskMeta.profInfo = HcclTaskProfInfo(
            ProfTaskParamType::TASK_REDUCE_INLINE, ProfLinkType::LINK_ONCHIP,
            ProfTransportType::TRANSPORT_LOCAL); // 评估模式 src/dst 同本设备
        taskMeta.taskData.reduce.reduceOp = ParseReduceTypeDavid(sqe->opcode);
        taskMeta.taskData.reduce.dataType = ParseDataTypeDavid(sqe->opcode);
        taskMeta.taskData.reduce.srcDeviceId = identity.rankId;
        taskMeta.taskData.reduce.srcOffset = srcOffset;
        taskMeta.taskData.reduce.dstDeviceId = identity.rankId;
        taskMeta.taskData.reduce.dstOffset = dstOffset;
        taskMeta.taskData.reduce.dataCount = length;
        InsertTaskToCollectionDev(&taskMeta);
        PrintTaskMetaData(taskMeta);
    }
}

// 非评估模式: 虚拟地址 + 地址反查 deviceId
void ParseDavidSDMASqeWithIdentity(uint32_t streamId, void* sqeBuf, const TaskIdentity& identity)
{
    Rt91095StarsMemcpySqe* sqe = reinterpret_cast<Rt91095StarsMemcpySqe*>(sqeBuf);
    HcclTaskMetaData taskMeta;
    SetTaskIdentity(taskMeta, identity);
    taskMeta.streamId = streamId;
    taskMeta.jettyId = UINT32_MAX;

    uint64_t length = sqe->u.strideMode0.lengthMove;
    // SDMA 本地搬运/归约：src、dst 都是当前设备的映射地址。一次按
    // (当前设备, 映射地址) 的区间定位同时取得虚拟地址与物理属主，
    // 不再"翻译后再按虚拟地址全局反查 VMB"。
    const uint64_t localDeviceId = GetCurDeviceKey();
    auto srcResolution = ResolveTaskAddress(
        GetFull64BitAddr(sqe->u.strideMode0.srcAddrLow, sqe->u.strideMode0.srcAddrHigh), localDeviceId);
    if (!srcResolution.ok()) {
        HCCL_VM_ERROR(
            "skip SDMA SQE because src address resolution failed, code={}, diagnostic={}",
            static_cast<uint32_t>(srcResolution.code), srcResolution.diagnostic);
        return;
    }
    auto dstResolution = ResolveTaskAddress(
        GetFull64BitAddr(sqe->u.strideMode0.dstAddrLow, sqe->u.strideMode0.dstAddrHigh), localDeviceId);
    if (!dstResolution.ok()) {
        HCCL_VM_ERROR(
            "skip SDMA SQE because dst address resolution failed, code={}, diagnostic={}",
            static_cast<uint32_t>(dstResolution.code), dstResolution.diagnostic);
        return;
    }
    const uint64_t srcOffset = srcResolution->virtualAddress;
    const uint64_t dstOffset = dstResolution->virtualAddress;
    const uint32_t srcDeviceId = srcResolution->physicalDeviceId;
    const uint32_t dstDeviceId = dstResolution->physicalDeviceId;
    // 同设备为本地搬运(ONCHIP+LOCAL), 跨设备为跨片 SDMA 通道(PCIE+SDMA), 对齐 hcomm P2P transport
    const ProfLinkType link = (srcDeviceId == dstDeviceId) ? ProfLinkType::LINK_ONCHIP : ProfLinkType::LINK_PCIE;
    const ProfTransportType transport
        = (srcDeviceId == dstDeviceId) ? ProfTransportType::TRANSPORT_LOCAL : ProfTransportType::TRANSPORT_SDMA;

    if (sqe->opcode == 0) { // 表示是memcpy
        taskMeta.taskType = HccLTaskMetaType::MEM_CPY;
        taskMeta.profInfo = HcclTaskProfInfo(ProfTaskParamType::TASK_SDMA, link, transport);
        taskMeta.taskData.transMem.srcOffset = srcOffset;
        taskMeta.taskData.transMem.dstOffset = dstOffset;
        taskMeta.taskData.transMem.dstDeviceId = dstDeviceId;
        taskMeta.taskData.transMem.srcDeviceId = srcDeviceId;
        taskMeta.taskData.transMem.len = length;
        InsertTaskToCollectionDev(&taskMeta);
        PrintTaskMetaData(taskMeta);
    } else { // reduce
        taskMeta.taskType = HccLTaskMetaType::REDUCE;
        taskMeta.profInfo = HcclTaskProfInfo(ProfTaskParamType::TASK_REDUCE_INLINE, link, transport);
        taskMeta.taskData.reduce.reduceOp = ParseReduceTypeDavid(sqe->opcode);
        taskMeta.taskData.reduce.dataType = ParseDataTypeDavid(sqe->opcode);
        taskMeta.taskData.reduce.srcOffset = srcOffset;
        taskMeta.taskData.reduce.dstOffset = dstOffset;
        taskMeta.taskData.reduce.dstDeviceId = dstDeviceId;
        taskMeta.taskData.reduce.srcDeviceId = srcDeviceId;
        taskMeta.taskData.reduce.dataCount = length;
        InsertTaskToCollectionDev(&taskMeta);
        PrintTaskMetaData(taskMeta);
    }
}

// 评估模式: deviceId字段承载通信域rankId
void ParseDavidNotifySqeEstimate(uint32_t streamId, void* sqeBuf, bool isPost, const TaskIdentity& identity)
{
    Rt91095StarsNotifySqe* sqe = reinterpret_cast<Rt91095StarsNotifySqe*>(sqeBuf);
    HcclTaskMetaData taskMeta;
    SetTaskIdentity(taskMeta, identity);
    taskMeta.streamId = streamId;
    taskMeta.jettyId = UINT32_MAX;
    taskMeta.taskType = isPost ? HccLTaskMetaType::NOTIFY_RECORD : HccLTaskMetaType::NOTIFY_WAIT;
    taskMeta.profInfo = HcclTaskProfInfo(
        isPost ? ProfTaskParamType::TASK_NOTIFY_RECORD : ProfTaskParamType::TASK_NOTIFY_WAIT, ProfLinkType::LINK_ONCHIP,
        ProfTransportType::TRANSPORT_LOCAL); // 本端 notify
    const uint64_t notifyId = sqe->notifyId;
    taskMeta.taskData.notify.notifyId = notifyId;
    taskMeta.taskData.notify.srcDeviceId = identity.rankId;
    taskMeta.taskData.notify.dstDeviceId = identity.rankId;
    PrintTaskMetaData(taskMeta);
    InsertTaskToCollectionDev(&taskMeta);
}

// 非评估模式: notify端点仅deviceId维度, 对端device置0
void ParseDavidNotifySqeWithIdentity(uint32_t streamId, void* sqeBuf, bool isPost, const TaskIdentity& identity)
{
    Rt91095StarsNotifySqe* sqe = reinterpret_cast<Rt91095StarsNotifySqe*>(sqeBuf);
    HcclTaskMetaData taskMeta;
    SetTaskIdentity(taskMeta, identity);
    taskMeta.streamId = streamId;
    taskMeta.jettyId = UINT32_MAX;
    taskMeta.taskType = isPost ? HccLTaskMetaType::NOTIFY_RECORD : HccLTaskMetaType::NOTIFY_WAIT;
    taskMeta.profInfo = HcclTaskProfInfo(
        isPost ? ProfTaskParamType::TASK_NOTIFY_RECORD : ProfTaskParamType::TASK_NOTIFY_WAIT, ProfLinkType::LINK_ONCHIP,
        ProfTransportType::TRANSPORT_LOCAL); // 本端 notify
    const uint32_t deviceNotifyId = static_cast<uint32_t>(sqe->notifyId);
    const uint64_t notifyTaskId = HcclSim::MakeNotifyTaskId(identity.rankId, deviceNotifyId);
    if (!HcclSim::IsValidNotifyTaskId(notifyTaskId)) {
        HCCL_VM_ERROR(
            "invalid rank-scoped notify SQE: rank={:d}, deviceNotifyId={:d}", identity.rankId, deviceNotifyId);
        return;
    }
    taskMeta.taskData.notify.notifyId = notifyTaskId;
    SetNotifySourceDevice(taskMeta);
    PrintTaskMetaData(taskMeta);
    InsertTaskToCollectionDev(&taskMeta);
}

uint32_t CalculateCiValue(uint64_t wqeBuffer, uint32_t piValue)
{
    if (piValue == 0) {
        return 0;
    }
    uint32_t cqeCnt = 0;
    uint32_t prePiVal = (piValue >= 2) ? piValue - 2 : 0; // 先向前查找两个WQE
    uint64_t wqeAddr = wqeBuffer + prePiVal * HCCL_WQE_SIZE;
    UdmaSqeCommon* ubCommon = reinterpret_cast<UdmaSqeCommon*>(wqeAddr);
    while (true) {
        if (ubCommon->opcode != UdmaSqOpcode::UDMA_OPC_WRITE_WITH_NOTIFY) {
            prePiVal = piValue - 1;
            wqeAddr = wqeBuffer + prePiVal * HCCL_WQE_SIZE;
            ubCommon = reinterpret_cast<UdmaSqeCommon*>(wqeAddr);
        }

        piValue = prePiVal;

        // 得到前一个真实的WQE(64字节或128字节)
        if (ubCommon->cqe == 1) {
            cqeCnt++;
        }

        // 找到了前一次dolBell的结束下标
        if (cqeCnt == 2) {
            if (ubCommon->opcode != UdmaSqOpcode::UDMA_OPC_WRITE_WITH_NOTIFY) {
                return prePiVal + 1;
            }
            return prePiVal + 2;
        }

        // 找到wqeBuffer的最头部了，直接返回
        if (prePiVal == 0) {
            return prePiVal;
        }

        // 继续向前查找两个WQE
        prePiVal = (piValue >= 2) ? piValue - 2 : 0;
        wqeAddr = wqeBuffer + prePiVal * HCCL_WQE_SIZE;
        ubCommon = reinterpret_cast<UdmaSqeCommon*>(wqeAddr);
    }

    HCCL_VM_ERROR("calculate ciValue failed.");
    return 0;
}

// UDMA 统一入口: 骨架同主线, case 内按运行模式分派;
// 评估模式额外查询本端 lcEid, 随 jetty 路径任务下发
void ParseDavidUDMASqeWithIdentity(uint32_t streamId, void* sqeBuf, const TaskIdentity& identity)
{
    Rt91095StarsUbdmaDBmodeSqe* ubSqe = reinterpret_cast<Rt91095StarsUbdmaDBmodeSqe*>(sqeBuf);
    uint32_t jettyId = ubSqe->jettyId1;
    uint64_t wqeBuffer = 0;
    if (!GetWqebufferByJettyId(jettyId, wqeBuffer) || wqeBuffer == 0) {
        HCCL_VM_ERROR("GetWqebufferByJettyId failed or returned null, jettyId[{}].", jettyId);
        return;
    }

    // SQE+多WQE场景, 组号 = 门铃批号 (环形递增 1..255, 0 保留为未打标)
    g_currSqeGroupId = g_currSqeGroupId % 255u + 1u;

    // 评估模式: 本端EID(lcEid), 仅jetty路径任务需要; 查询失败保留全0, 不丢任务
    const bool estimateMode = sim::IsEstimateMode();
    uint8_t lcEid[HcclSim::URMA_EID_LEN] = {0};
    if (estimateMode && !GetLocalEidByJettyId(jettyId, lcEid)) {
        HCCL_VM_ERROR("GetLocalEidByJettyId failed, jettyId[{}].", jettyId);
    }

    uint32_t piValue = ubSqe->piValue1;
    uint32_t ciValue = CalculateCiValue(wqeBuffer, ubSqe->piValue1);
    if (ciValue >= piValue) {
        HCCL_VM_ERROR("jettyId[{}] ciVal[{}] piValue[{}] streamId[{}].", jettyId, ciValue, piValue, streamId);
        return;
    }

    HCCL_VM_INFO("jettyId[{}] ciVal[{}] piValue[{}] streamId[{}].", jettyId, ciValue, piValue, streamId);
    for (auto index = ciValue; index < piValue; index++) {
        uint64_t wqeAddr = wqeBuffer + index * HCCL_WQE_SIZE;
        UdmaSqeCommon* ubCommon = reinterpret_cast<UdmaSqeCommon*>(wqeAddr);
        switch (ubCommon->opcode) {
            case static_cast<int>(UdmaSqOpcode::UDMA_OPC_WRITE): {
                estimateMode ? ParseDavidUBReadWriteSqeEstimate(wqeAddr, streamId, jettyId, false, identity, lcEid) :
                               ParseDavidUBReadWriteSqeWithIdentity(wqeAddr, streamId, jettyId, false, identity);
                break;
            }
            case static_cast<int>(UdmaSqOpcode::UDMA_OPC_WRITE_WITH_NOTIFY): {
                estimateMode ? ParseDavidUBWriteWithNotifySqeEstimate(wqeAddr, streamId, jettyId, identity, lcEid) :
                               ParseDavidUBWriteWithNotifySqeWithIdentity(wqeAddr, streamId, jettyId, identity);
                index++; // 占128字节两个WQE
                break;
            }
            case static_cast<int>(UdmaSqOpcode::UDMA_OPC_READ): {
                estimateMode ? ParseDavidUBReadWriteSqeEstimate(wqeAddr, streamId, jettyId, true, identity, lcEid) :
                               ParseDavidUBReadWriteSqeWithIdentity(wqeAddr, streamId, jettyId, true, identity);
                break;
            }
            default: {
                HCCL_VM_ERROR("not support opcode[{}].", static_cast<uint32_t>(ubCommon->opcode));
                return;
            }
        }
    }
}

// 评估模式: 原始地址 + EID 解析对端(deviceId字段承载rankId), 收集lcEid
void ParseDavidUBReadWriteSqeEstimate(
    uint64_t wqeAddr, uint16_t streamId, uint32_t jettyId, bool isRead, const TaskIdentity& identity,
    const uint8_t* lcEid)
{
    UdmaSqeWrite* ubWqe = reinterpret_cast<UdmaSqeWrite*>(wqeAddr);
    HcclTaskMetaData taskMeta;
    SetTaskIdentity(taskMeta, identity);
    taskMeta.streamId = streamId;
    taskMeta.jettyId = jettyId;
    taskMeta.sqeGroupId = g_currSqeGroupId;
    CopyRmtEidFromWqe(taskMeta.rmEid, reinterpret_cast<const uint8_t*>(ubWqe->comm.rmtEid));
    if (lcEid != nullptr) {
        memcpy(taskMeta.lcEid, lcEid, sizeof(taskMeta.lcEid));
    }

    uint32_t rmtDeviceId = 0;
    if (!GetRmtDeviceIdByEid(reinterpret_cast<const uint8_t*>(ubWqe->comm.rmtEid), rmtDeviceId)) {
        HCCL_VM_ERROR(
            "skip UDMA SQE because remote device resolution failed, streamId={}, jettyId={}", streamId, jettyId);
        return;
    }
    // 对端rankId: rank增强信息, 解析失败(如非EID模式)不丢任务, 以UINT32_MAX标记
    uint32_t rmtRankId = UINT32_MAX;
    if (!GetRmtRankIdByEid(reinterpret_cast<const uint8_t*>(ubWqe->comm.rmtEid), rmtRankId)) {
        rmtRankId = UINT32_MAX;
    }
    // case1:UbConnLite::InlineWrite 写Notify
    if (ubWqe->comm.inlineEn == 1) {
        uint64_t notifyAddr = GetFull64BitAddr(ubWqe->comm.rmtAddrLow, ubWqe->comm.rmtAddrHigh);
        taskMeta.taskType = HccLTaskMetaType::NOTIFY_RECORD;
        taskMeta.profInfo = HcclTaskProfInfo(
            ProfTaskParamType::TASK_UB_INLINE_WRITE, ProfLinkType::LINK_UB,
            ProfTransportType::TRANSPORT_UB); // hcomm UbTransportLite: UB+UB
        taskMeta.taskData.notify.notifyId = notifyAddr;
        SetRemoteNotifyEndpointsEstimate(taskMeta, identity, rmtDeviceId, rmtRankId);
        PrintTaskMetaData(taskMeta);
        InsertTaskToCollectionDev(&taskMeta);
        return;
    }

    uint64_t length = static_cast<uint64_t>(ubWqe->u.sge.length);
    uint64_t locAddr = GetFull64BitAddr(ubWqe->u.sge.dataAddrLow, ubWqe->u.sge.dataAddrHigh);
    uint64_t rmtAddr = GetFull64BitAddr(ubWqe->comm.rmtAddrLow, ubWqe->comm.rmtAddrHigh);
    uint64_t srcOffset = isRead ? rmtAddr : locAddr;
    uint64_t dstOffset = isRead ? locAddr : rmtAddr;
    // 评估模式: deviceId字段承载通信域rankId, write = src 本端/dst 对端, read = src 对端/dst 本端
    uint32_t srcRankId = isRead ? rmtRankId : identity.rankId;
    uint32_t dstRankId = isRead ? identity.rankId : rmtRankId;
    taskMeta.taskType = HccLTaskMetaType::MEM_CPY;
    taskMeta.profInfo = HcclTaskProfInfo(
        ProfTaskParamType::TASK_UB, ProfLinkType::LINK_UB,
        ProfTransportType::TRANSPORT_UB); // hcomm UbTransportLite: UB+UB
    taskMeta.taskData.transMem.srcDeviceId = srcRankId;
    taskMeta.taskData.transMem.srcOffset = srcOffset;
    taskMeta.taskData.transMem.dstDeviceId = dstRankId;
    taskMeta.taskData.transMem.dstOffset = dstOffset;
    taskMeta.taskData.transMem.len = length;

    // case3:UbConnLite::WriteReduce Reduce
    if (ubWqe->comm.udfFlag == 1) {
        taskMeta.taskType = HccLTaskMetaType::REDUCE;
        taskMeta.profInfo = HcclTaskProfInfo(
            ProfTaskParamType::TASK_UB_REDUCE_INLINE, ProfLinkType::LINK_UB, ProfTransportType::TRANSPORT_UB);
        taskMeta.taskData.reduce.reduceOp = ParseUbReduceTypeDavid(ubWqe->comm.inlinedata.udfData.reduceOp);
        taskMeta.taskData.reduce.dataType = ParseUbDataTypeDavid(ubWqe->comm.inlinedata.udfData.reduceType);
        taskMeta.taskData.reduce.srcDeviceId = srcRankId;
        taskMeta.taskData.reduce.srcOffset = srcOffset;
        taskMeta.taskData.reduce.dstDeviceId = dstRankId;
        taskMeta.taskData.reduce.dstOffset = dstOffset;
        taskMeta.taskData.reduce.dataCount = length;
    }

    PrintTaskMetaData(taskMeta);
    InsertTaskToCollectionDev(&taskMeta);
}

namespace {
std::mutex g_commRankCacheMtx;
std::unordered_map<uint64_t, uint32_t> g_commRankCache; // key=(commId<<32)|deviceId，只缓存正结果
} // namespace

// Communicator 成员行在建域后不再变更：按 (commId, deviceId) 缓存 rankId，命中即零 DB 往返。
// 负结果不缓存，保留 GetCommRankByDeviceId 内部对并发建域窗口的重试语义。
static bool GetCommRankCached(uint64_t commId, uint64_t deviceId, uint32_t& rankId)
{
    const uint64_t key = (commId << 32U) | (deviceId & 0xFFFFFFFFU);
    {
        std::lock_guard<std::mutex> lock(g_commRankCacheMtx);
        auto it = g_commRankCache.find(key);
        if (it != g_commRankCache.end()) {
            rankId = it->second;
            return true;
        }
    }
    if (!sim::runtime::GetCommRankByDeviceId(commId, deviceId, rankId)) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_commRankCacheMtx);
    g_commRankCache.emplace(key, rankId);
    return true;
}

// 非评估模式: 虚拟地址 + 地址反查deviceId
void ParseDavidUBReadWriteSqeWithIdentity(
    uint64_t wqeAddr, uint16_t streamId, uint32_t jettyId, bool isRead, const TaskIdentity& identity)
{
    UdmaSqeWrite* ubWqe = reinterpret_cast<UdmaSqeWrite*>(wqeAddr);
    HcclTaskMetaData taskMeta;
    SetTaskIdentity(taskMeta, identity);
    taskMeta.streamId = streamId;
    taskMeta.jettyId = jettyId;
    taskMeta.sqeGroupId = g_currSqeGroupId;
    // 主线为原序 memcpy; CI 统一翻转序, 对齐 EndPoint.eid/ranktable.json 字节序
    CopyRmtEidFromWqe(taskMeta.rmEid, reinterpret_cast<const uint8_t*>(ubWqe->comm.rmtEid));

    uint32_t rmtDeviceId = 0;
    if (!GetRmtDeviceIdByEid(reinterpret_cast<const uint8_t*>(ubWqe->comm.rmtEid), rmtDeviceId)) {
        HCCL_VM_ERROR(
            "skip UDMA SQE because remote device resolution failed, streamId={}, jettyId={}", streamId, jettyId);
        return;
    }
    uint32_t rmtRankId = 0;
    if (!GetCommRankCached(identity.commId, rmtDeviceId, rmtRankId)) {
        HCCL_VM_ERROR(
            "failed to resolve remote rank: deviceId={}, commId={}, streamId={}, jettyId={}", rmtDeviceId,
            identity.commId, streamId, jettyId);
        return;
    }
    // case1:UbConnLite::InlineWrite 写Notify
    if (ubWqe->comm.inlineEn == 1) {
        uint64_t notifyAddr = GetFull64BitAddr(ubWqe->comm.rmtAddrLow, ubWqe->comm.rmtAddrHigh);
        taskMeta.taskType = HccLTaskMetaType::NOTIFY_RECORD;
        taskMeta.profInfo = HcclTaskProfInfo(
            ProfTaskParamType::TASK_UB_INLINE_WRITE, ProfLinkType::LINK_UB,
            ProfTransportType::TRANSPORT_UB); // hcomm UbTransportLite: UB+UB
        const uint64_t notifyTaskId = HcclSim::MakeNotifyTaskId(rmtRankId, notifyAddr);
        taskMeta.taskData.notify.notifyId = notifyTaskId;
        SetRemoteNotifyEndpoints(taskMeta, rmtDeviceId);
        PrintTaskMetaData(taskMeta);
        InsertTaskToCollectionDev(&taskMeta);
        return;
    }

    uint64_t length = static_cast<uint64_t>(ubWqe->u.sge.length);
    // loc 为当前设备映射、rmt 为远端设备（EID 解析出的 rmtDeviceId）映射：
    // 两端各按 (映射设备, 映射地址) 一次定位，虚拟地址与物理属主同取；
    // READ 时 src=远端/dst=本地，WRITE 时 src=本地/dst=远端，任务字段语义不变。
    auto locResolution
        = ResolveTaskAddress(GetFull64BitAddr(ubWqe->u.sge.dataAddrLow, ubWqe->u.sge.dataAddrHigh), GetCurDeviceKey());
    if (!locResolution.ok()) {
        HCCL_VM_ERROR(
            "skip UDMA SQE because local address resolution failed, code={}, diagnostic={}, "
            "streamId={}, jettyId={}",
            static_cast<uint32_t>(locResolution.code), locResolution.diagnostic, streamId, jettyId);
        return;
    }
    auto rmtResolution
        = ResolveTaskAddress(GetFull64BitAddr(ubWqe->comm.rmtAddrLow, ubWqe->comm.rmtAddrHigh), rmtDeviceId);
    if (!rmtResolution.ok()) {
        HCCL_VM_ERROR(
            "skip UDMA SQE because remote address resolution failed, code={}, diagnostic={}, "
            "streamId={}, jettyId={}",
            static_cast<uint32_t>(rmtResolution.code), rmtResolution.diagnostic, streamId, jettyId);
        return;
    }
    const auto& srcResolution = isRead ? rmtResolution : locResolution;
    const auto& dstResolution = isRead ? locResolution : rmtResolution;
    const uint64_t srcOffset = srcResolution->virtualAddress;
    const uint64_t dstOffset = dstResolution->virtualAddress;
    const uint32_t srcDeviceId = srcResolution->physicalDeviceId;
    const uint32_t dstDeviceId = dstResolution->physicalDeviceId;
    taskMeta.taskType = HccLTaskMetaType::MEM_CPY;
    taskMeta.profInfo = HcclTaskProfInfo(
        ProfTaskParamType::TASK_UB, ProfLinkType::LINK_UB,
        ProfTransportType::TRANSPORT_UB); // hcomm UbTransportLite: UB+UB
    taskMeta.taskData.transMem.srcOffset = srcOffset;
    taskMeta.taskData.transMem.dstOffset = dstOffset;
    taskMeta.taskData.transMem.dstDeviceId = dstDeviceId;
    taskMeta.taskData.transMem.srcDeviceId = srcDeviceId;
    taskMeta.taskData.transMem.len = length;

    // case3:UbConnLite::WriteReduce Reduce
    if (ubWqe->comm.udfFlag == 1) {
        taskMeta.taskType = HccLTaskMetaType::REDUCE;
        taskMeta.profInfo = HcclTaskProfInfo(
            ProfTaskParamType::TASK_UB_REDUCE_INLINE, ProfLinkType::LINK_UB, ProfTransportType::TRANSPORT_UB);
        taskMeta.taskData.reduce.reduceOp = ParseUbReduceTypeDavid(ubWqe->comm.inlinedata.udfData.reduceOp);
        taskMeta.taskData.reduce.dataType = ParseUbDataTypeDavid(ubWqe->comm.inlinedata.udfData.reduceType);
        taskMeta.taskData.reduce.srcOffset = srcOffset;
        taskMeta.taskData.reduce.dstOffset = dstOffset;
        taskMeta.taskData.reduce.dstDeviceId = dstDeviceId;
        taskMeta.taskData.reduce.srcDeviceId = srcDeviceId;
        taskMeta.taskData.reduce.dataCount = length;
    }

    PrintTaskMetaData(taskMeta);
    InsertTaskToCollectionDev(&taskMeta);
}

// 评估模式: 原始地址 + EID 解析对端(deviceId字段承载rankId), 收集lcEid; 写(MEM_CPY/REDUCE)与notify组合为单任务
void ParseDavidUBWriteWithNotifySqeEstimate(
    uint64_t wqeAddr, uint16_t streamId, uint32_t jettyId, const TaskIdentity& identity, const uint8_t* lcEid)
{
    UdmaSqeWriteWithNotify* ubWqe = reinterpret_cast<UdmaSqeWriteWithNotify*>(wqeAddr);
    HcclTaskMetaData taskMeta;
    SetTaskIdentity(taskMeta, identity);
    taskMeta.streamId = streamId;
    taskMeta.jettyId = jettyId;
    taskMeta.sqeGroupId = g_currSqeGroupId;
    CopyRmtEidFromWqe(taskMeta.rmEid, reinterpret_cast<const uint8_t*>(ubWqe->comm.rmtEid));
    if (lcEid != nullptr) {
        memcpy(taskMeta.lcEid, lcEid, sizeof(taskMeta.lcEid));
    }
    uint32_t rmtDeviceId = 0;
    if (!GetRmtDeviceIdByEid(reinterpret_cast<const uint8_t*>(ubWqe->comm.rmtEid), rmtDeviceId)) {
        HCCL_VM_ERROR(
            "skip UDMA write-with-notify SQE because remote device resolution failed, "
            "streamId={}, jettyId={}",
            streamId, jettyId);
        return;
    }
    // 对端rankId: 解析失败(如非EID模式)不丢任务, 以UINT32_MAX标记
    uint32_t rmtRankId = UINT32_MAX;
    if (!GetRmtRankIdByEid(reinterpret_cast<const uint8_t*>(ubWqe->comm.rmtEid), rmtRankId)) {
        rmtRankId = UINT32_MAX;
    }
    uint64_t dstOffset = GetFull64BitAddr(ubWqe->comm.rmtAddrLow, ubWqe->comm.rmtAddrHigh);
    uint64_t srcOffset = GetFull64BitAddr(ubWqe->localU.sge.dataAddrLow, ubWqe->localU.sge.dataAddrHigh);
    uint64_t length = static_cast<uint64_t>(ubWqe->localU.sge.length);
    uint64_t notifyAddr = GetFull64BitAddr(ubWqe->notify.notifyAddrLow, ubWqe->notify.notifyAddrHigh);
    // 评估模式: deviceId字段承载通信域rankId, write = src 本端/dst 对端;
    // notifyId 沿用原始值, 与评估模式 NOTIFY_RECORD/WAIT 的原始 ID 空间一致(不做 rank 编码)
    taskMeta.taskType = HccLTaskMetaType::WRITE_WITH_NOTIFY;
    taskMeta.profInfo = HcclTaskProfInfo(
        ProfTaskParamType::TASK_WRITE_WITH_NOTIFY, ProfLinkType::LINK_UB,
        ProfTransportType::TRANSPORT_UB); // hcomm UbTransportLite: UB+UB
    WriteWithNotifyTask& wn = taskMeta.taskData.writeWithNotify;
    wn.srcDeviceId = identity.rankId;
    wn.srcOffset = srcOffset;
    wn.dstDeviceId = rmtRankId;
    wn.dstOffset = dstOffset;
    wn.len = length;
    wn.notifyId = notifyAddr;
    if (ubWqe->comm.udfFlag == 1) { // Reduce操作标识
        wn.isReduce = 1;
        taskMeta.profInfo = HcclTaskProfInfo(
            ProfTaskParamType::TASK_WRITE_REDUCE_WITH_NOTIFY, ProfLinkType::LINK_UB, ProfTransportType::TRANSPORT_UB);
        wn.reduceOp = ParseUbReduceTypeDavid(ubWqe->comm.inlinedata.udfData.reduceOp);
        wn.dataType = ParseUbDataTypeDavid(ubWqe->comm.inlinedata.udfData.reduceType);
    }
    PrintTaskMetaData(taskMeta);
    InsertTaskToCollectionDev(&taskMeta);
}

// 非评估模式: 虚拟地址 + 地址反查deviceId, 写与notify组合为单任务
void ParseDavidUBWriteWithNotifySqeWithIdentity(
    uint64_t wqeAddr, uint16_t streamId, uint32_t jettyId, const TaskIdentity& identity)
{
    UdmaSqeWriteWithNotify* ubWqe = reinterpret_cast<UdmaSqeWriteWithNotify*>(wqeAddr);
    HcclTaskMetaData taskMeta;
    SetTaskIdentity(taskMeta, identity);
    taskMeta.streamId = streamId;
    taskMeta.jettyId = jettyId;
    taskMeta.sqeGroupId = g_currSqeGroupId;
    // 主线为原序 memcpy; CI 统一翻转序, 对齐 EndPoint.eid/ranktable.json 字节序
    CopyRmtEidFromWqe(taskMeta.rmEid, reinterpret_cast<const uint8_t*>(ubWqe->comm.rmtEid));
    uint32_t rmtDeviceId = 0;
    if (!GetRmtDeviceIdByEid(reinterpret_cast<const uint8_t*>(ubWqe->comm.rmtEid), rmtDeviceId)) {
        HCCL_VM_ERROR(
            "skip UDMA write-with-notify SQE because remote device resolution failed, "
            "streamId={}, jettyId={}",
            streamId, jettyId);
        return;
    }
    uint32_t rmtRankId = 0;
    if (!GetCommRankCached(identity.commId, rmtDeviceId, rmtRankId)) {
        HCCL_VM_ERROR(
            "failed to resolve remote rank: deviceId={}, commId={}, streamId={}, jettyId={}", rmtDeviceId,
            identity.commId, streamId, jettyId);
        return;
    }
    // 写部分: 数据段 src=当前设备映射、dst=远端设备映射，各按 (映射设备, 映射地址) 一次定位
    auto dstResolution
        = ResolveTaskAddress(GetFull64BitAddr(ubWqe->comm.rmtAddrLow, ubWqe->comm.rmtAddrHigh), rmtDeviceId);
    if (!dstResolution.ok()) {
        HCCL_VM_ERROR(
            "skip UDMA write-with-notify SQE because remote address resolution failed, "
            "code={}, diagnostic={}, streamId={}, jettyId={}",
            static_cast<uint32_t>(dstResolution.code), dstResolution.diagnostic, streamId, jettyId);
        return;
    }
    auto srcResolution = ResolveTaskAddress(
        GetFull64BitAddr(ubWqe->localU.sge.dataAddrLow, ubWqe->localU.sge.dataAddrHigh), GetCurDeviceKey());
    if (!srcResolution.ok()) {
        HCCL_VM_ERROR(
            "skip UDMA write-with-notify SQE because local address resolution failed, "
            "code={}, diagnostic={}, streamId={}, jettyId={}",
            static_cast<uint32_t>(srcResolution.code), srcResolution.diagnostic, streamId, jettyId);
        return;
    }
    uint64_t length = static_cast<uint64_t>(ubWqe->localU.sge.length);
    const uint64_t srcOffset = srcResolution->virtualAddress;
    const uint64_t dstOffset = dstResolution->virtualAddress;
    const uint32_t srcDeviceId = srcResolution->physicalDeviceId;
    const uint32_t dstDeviceId = dstResolution->physicalDeviceId;
    const uint64_t notifyAddr = GetFull64BitAddr(ubWqe->notify.notifyAddrLow, ubWqe->notify.notifyAddrHigh);
    taskMeta.taskType = HccLTaskMetaType::WRITE_WITH_NOTIFY;
    taskMeta.profInfo = HcclTaskProfInfo(
        ProfTaskParamType::TASK_WRITE_WITH_NOTIFY, ProfLinkType::LINK_UB,
        ProfTransportType::TRANSPORT_UB); // hcomm UbTransportLite: UB+UB
    WriteWithNotifyTask& wn = taskMeta.taskData.writeWithNotify;
    wn.srcDeviceId = srcDeviceId;
    wn.srcOffset = srcOffset;
    wn.dstDeviceId = dstDeviceId;
    wn.dstOffset = dstOffset;
    wn.len = length;
    wn.notifyId = HcclSim::MakeNotifyTaskId(rmtRankId, static_cast<uint32_t>(notifyAddr));
    if (ubWqe->comm.udfFlag == 1) { // Reduce操作标识
        wn.isReduce = 1;
        taskMeta.profInfo = HcclTaskProfInfo(
            ProfTaskParamType::TASK_WRITE_REDUCE_WITH_NOTIFY, ProfLinkType::LINK_UB, ProfTransportType::TRANSPORT_UB);
        wn.reduceOp = ParseUbReduceTypeDavid(ubWqe->comm.inlinedata.udfData.reduceOp);
        wn.dataType = ParseUbDataTypeDavid(ubWqe->comm.inlinedata.udfData.reduceType);
    }
    PrintTaskMetaData(taskMeta);
    InsertTaskToCollectionDev(&taskMeta);
}

void ParseDavidSDMASqe(uint32_t streamId, void* sqeBuf)
{
    TaskIdentity identity{};
    if (!ResolveCurrentTaskIdentity(identity)) {
        return;
    }
    if (sim::IsEstimateMode()) {
        ParseDavidSDMASqeEstimate(streamId, sqeBuf, identity);
        return;
    }
    ParseDavidSDMASqeWithIdentity(streamId, sqeBuf, identity);
}

void ParseDavidNotifySqe(uint32_t streamId, void* sqeBuf, bool isPost)
{
    TaskIdentity identity{};
    if (!ResolveCurrentTaskIdentity(identity)) {
        return;
    }
    if (sim::IsEstimateMode()) {
        ParseDavidNotifySqeEstimate(streamId, sqeBuf, isPost, identity);
        return;
    }
    ParseDavidNotifySqeWithIdentity(streamId, sqeBuf, isPost, identity);
}

void ParseDavidUDMASqe(uint32_t streamId, void* sqeBuf)
{
    TaskIdentity identity{};
    if (!ResolveCurrentTaskIdentity(identity)) {
        return;
    }
    ParseDavidUDMASqeWithIdentity(streamId, sqeBuf, identity);
}

void ParseDavidUBReadWriteSqe(uint64_t wqeAddr, uint16_t streamId, uint32_t jettyId, bool isRead)
{
    TaskIdentity identity{};
    if (!ResolveCurrentTaskIdentity(identity)) {
        return;
    }
    if (sim::IsEstimateMode()) {
        uint8_t lcEid[HcclSim::URMA_EID_LEN] = {0};
        (void)GetLocalEidByJettyId(jettyId, lcEid);
        ParseDavidUBReadWriteSqeEstimate(wqeAddr, streamId, jettyId, isRead, identity, lcEid);
        return;
    }
    ParseDavidUBReadWriteSqeWithIdentity(wqeAddr, streamId, jettyId, isRead, identity);
}

void ParseDavidUBWriteWithNotifySqe(uint64_t wqeAddr, uint16_t streamId, uint32_t jettyId)
{
    TaskIdentity identity{};
    if (!ResolveCurrentTaskIdentity(identity)) {
        return;
    }
    if (sim::IsEstimateMode()) {
        uint8_t lcEid[HcclSim::URMA_EID_LEN] = {0};
        (void)GetLocalEidByJettyId(jettyId, lcEid);
        ParseDavidUBWriteWithNotifySqeEstimate(wqeAddr, streamId, jettyId, identity, lcEid);
        return;
    }
    ParseDavidUBWriteWithNotifySqeWithIdentity(wqeAddr, streamId, jettyId, identity);
}

uint64_t GetFull64BitAddr(uint32_t lowAddr, uint32_t highAddr)
{
    return (static_cast<uint64_t>(highAddr) << SHIFT_BIT32) | static_cast<uint64_t>(lowAddr);
}

std::map<uint32_t, HcclReduceOp> DavidReduceOpMap
    = {{0x01, HcclReduceOp::HCCL_REDUCE_SUM},
       {0x02, HcclReduceOp::HCCL_REDUCE_MAX},
       {0x03, HcclReduceOp::HCCL_REDUCE_MIN}};

HcclReduceOp ParseReduceTypeDavid(uint8_t result)
{
    uint8_t reduceType = result & 0x0F; // 直接与 0x0F 按位与，保留低 4 位
    if (DavidReduceOpMap.find(static_cast<uint32_t>(reduceType)) != DavidReduceOpMap.end()) {
        return DavidReduceOpMap[reduceType];
    }

    HCCL_VM_ERROR("not support reduceType[{}].", reduceType);
    return HcclReduceOp::HCCL_REDUCE_RESERVED;
}

std::map<uint32_t, HcclDataType> DavidDataTypeMap
    = {{0x00, HcclDataType::HCCL_DATA_TYPE_INT8},   {0x10, HcclDataType::HCCL_DATA_TYPE_INT16},
       {0x20, HcclDataType::HCCL_DATA_TYPE_INT32},  {0x30, HcclDataType::HCCL_DATA_TYPE_UINT8},
       {0x40, HcclDataType::HCCL_DATA_TYPE_UINT16}, {0x50, HcclDataType::HCCL_DATA_TYPE_UINT32},
       {0x60, HcclDataType::HCCL_DATA_TYPE_FP16},   {0x70, HcclDataType::HCCL_DATA_TYPE_FP32},
       {0x80, HcclDataType::HCCL_DATA_TYPE_BFP16}};

HcclDataType ParseDataTypeDavid(uint8_t result)
{
    uint8_t dataType = static_cast<uint32_t>(result) & 0xF0; // 提取高4位
    if (DavidDataTypeMap.find(static_cast<uint32_t>(dataType)) != DavidDataTypeMap.end()) {
        return DavidDataTypeMap[dataType];
    }

    HCCL_VM_ERROR("not support dataType[{}].", dataType);
    return HcclDataType::HCCL_DATA_TYPE_RESERVED;
}

std::map<uint32_t, HcclReduceOp> DavidUbReduceOpMap = {
    {0xA, HcclReduceOp::HCCL_REDUCE_SUM}, {0x8, HcclReduceOp::HCCL_REDUCE_MAX}, {0x9, HcclReduceOp::HCCL_REDUCE_MIN}};

HcclReduceOp ParseUbReduceTypeDavid(uint32_t type)
{
    if (DavidUbReduceOpMap.find(type) != DavidUbReduceOpMap.end()) {
        return DavidUbReduceOpMap[type];
    }

    HCCL_VM_ERROR("not support type[{}].", type);
    return HcclReduceOp::HCCL_REDUCE_RESERVED;
}

std::map<uint32_t, HcclDataType> DavidUbDataTypeMap
    = {{0x0, HcclDataType::HCCL_DATA_TYPE_INT8},   {0x1, HcclDataType::HCCL_DATA_TYPE_INT16},
       {0x2, HcclDataType::HCCL_DATA_TYPE_INT32},  {0x3, HcclDataType::HCCL_DATA_TYPE_UINT8},
       {0x4, HcclDataType::HCCL_DATA_TYPE_UINT16}, {0x5, HcclDataType::HCCL_DATA_TYPE_UINT32},
       {0x6, HcclDataType::HCCL_DATA_TYPE_FP16},   {0x7, HcclDataType::HCCL_DATA_TYPE_FP32},
       {0x8, HcclDataType::HCCL_DATA_TYPE_BFP16}};

HcclDataType ParseUbDataTypeDavid(uint32_t type)
{
    if (DavidUbDataTypeMap.find(type) != DavidUbDataTypeMap.end()) {
        return DavidUbDataTypeMap[type];
    }

    HCCL_VM_ERROR("not support type[{}].", type);
    return HcclDataType::HCCL_DATA_TYPE_RESERVED;
}

void PrintTaskMetaData(const HcclTaskMetaData& taskMeta)
{
    pid_t pid = getpid();
    HcclSim::Eid rmEidTmp{};
    HcclSim::Eid lcEidTmp{};
    (void)memcpy(rmEidTmp.raw, taskMeta.rmEid, HcclSim::URMA_EID_LEN);
    (void)memcpy(lcEidTmp.raw, taskMeta.lcEid, HcclSim::URMA_EID_LEN);
    const std::string eidInfo = ", rmEid[" + HcclSim::IpAddress(rmEidTmp).EidToHexString() + "], lcEid["
                                + HcclSim::IpAddress(lcEidTmp).EidToHexString() + "]";
    switch (taskMeta.taskType) {
        case HccLTaskMetaType::MEM_CPY:
            HCCL_VM_INFO(
                "rankId[{}] streamId[{}] taskType[MEM_CPY], jettyId[{}], srcOffset[{}] dstOffset[{}], len[{}], "
                "srcDeviceId[{}] dstDeviceId[{}]{}",
                taskMeta.rankId, taskMeta.streamId, taskMeta.jettyId, taskMeta.taskData.transMem.srcOffset,
                taskMeta.taskData.transMem.dstOffset, taskMeta.taskData.transMem.len,
                taskMeta.taskData.transMem.srcDeviceId, taskMeta.taskData.transMem.dstDeviceId, eidInfo);
            break;
        case HccLTaskMetaType::REDUCE:
            HCCL_VM_INFO(
                "rankId[{}] streamId[{}] taskType[REDUCE], jettyId[{}], srcOffset[{}] dstOffset[{}], len[{}], "
                "srcDeviceId[{}] dstDeviceId[{}], reduceOp[%{}], dataType[%{}]{}",
                taskMeta.rankId, taskMeta.streamId, taskMeta.jettyId, taskMeta.taskData.reduce.srcOffset,
                taskMeta.taskData.reduce.dstOffset, taskMeta.taskData.reduce.dataCount,
                taskMeta.taskData.reduce.srcDeviceId, taskMeta.taskData.reduce.dstDeviceId,
                taskMeta.taskData.reduce.reduceOp, taskMeta.taskData.reduce.dataType, eidInfo);
            break;
        case HccLTaskMetaType::NOTIFY_WAIT:
            HCCL_VM_INFO(
                "rankId[{}] streamId[{}] taskType[NOTIFY_WAIT], jettyId[{}], notifyId[{}], recordDeviceId[{}], "
                "waitDeviceId[{}]{}",
                taskMeta.rankId, taskMeta.streamId, taskMeta.jettyId, taskMeta.taskData.notify.notifyId,
                taskMeta.taskData.notify.srcDeviceId, taskMeta.taskData.notify.dstDeviceId, eidInfo);
            break;
        case HccLTaskMetaType::NOTIFY_RECORD:
            HCCL_VM_INFO(
                "rankId[{}] streamId[{}] taskType[NOTIFY_RECORD], jettyId[{}], notifyId[{}], recordDeviceId[{}], "
                "waitDeviceId[{}]{}",
                taskMeta.rankId, taskMeta.streamId, taskMeta.jettyId, taskMeta.taskData.notify.notifyId,
                taskMeta.taskData.notify.srcDeviceId, taskMeta.taskData.notify.dstDeviceId, eidInfo);
            break;
        case HccLTaskMetaType::WRITE_WITH_NOTIFY:
            HCCL_VM_INFO(
                "pid{}: rankId[{}] streamId[{}] taskType[WRITE_WITH_NOTIFY], jettyId[{}], isReduce[{}], srcOffset[{}] "
                "dstOffset[{}], len[{}], notifyId[{}], srcDeviceId[{}] dstDeviceId[{}]{}",
                pid, taskMeta.rankId, taskMeta.streamId, taskMeta.jettyId, taskMeta.taskData.writeWithNotify.isReduce,
                taskMeta.taskData.writeWithNotify.srcOffset, taskMeta.taskData.writeWithNotify.dstOffset,
                taskMeta.taskData.writeWithNotify.len, taskMeta.taskData.writeWithNotify.notifyId,
                taskMeta.taskData.writeWithNotify.srcDeviceId, taskMeta.taskData.writeWithNotify.dstDeviceId, eidInfo);
            break;
        default:
            break;
    }
}

namespace {
// EID -> {deviceId, rankId} 合并缓存: 映射由静态拓扑确定, 运行期不变, 首次访问整表灌入后纯内存命中
// (eid 藏在序列化 BLOB 内无法建索引, 逐次解析会全表扫描, 是 device 侧耗时大头)。
struct EidResolution {
    bool hasDevice{false};
    uint32_t deviceId{0};
    bool hasRank{false};
    uint32_t rankId{0};
};

std::mutex g_eidResMapMtx;
std::unordered_map<std::string, EidResolution>
    g_eidResMap; // 逐 EID 缓存(含负缓存): hasDevice/hasRank 皆 false 表示已确认不存在

// 按 EID 索引点查 EndPoint(deviceId) 与 RankEidMap(rankId)。调用方须持有 g_eidResMapMtx。
// 返回 true 表示结果可固化(含负结果); false 表示拓扑未就绪或查询出错, 不固化待重试。
bool ResolveEidByPointQueryLocked(const std::string& eidKey, EidResolution& res)
{
    std::vector<uint8_t> eidBytes(eidKey.begin(), eidKey.end());

    auto epRet = sim::runtime::Db::GetOneByPred<sim::runtime::EndPoint>(
        HcclSim::Storage::Eq(&sim::runtime::EndPoint::eid, eidBytes));
    if (epRet.ok()) {
        res.hasDevice = true;
        res.deviceId = static_cast<uint32_t>((*epRet).device_id);
    } else if (epRet.code == HcclSim::Storage::DbCode::NOT_FOUND) {
        // 表空视为拓扑未就绪(启动竞态), 不固化负结果
        if (sim::runtime::Db::Count<sim::runtime::EndPoint>(HcclSim::Storage::All<sim::runtime::EndPoint>())
                .value_or(0U)
            == 0) {
            return false;
        }
    } else {
        HCCL_VM_ERROR("query EndPoint by eid failed: {}", epRet.diagnostic);
        return false;
    }

    auto mapRet = sim::runtime::Db::GetOneByPred<sim::runtime::RankEidMap>(
        HcclSim::Storage::Eq(&sim::runtime::RankEidMap::eid, eidBytes));
    if (mapRet.ok()) {
        res.hasRank = true;
        res.rankId = static_cast<uint32_t>((*mapRet).rank_id);
    } else if (mapRet.code == HcclSim::Storage::DbCode::NOT_FOUND) {
        if (sim::runtime::Db::Count<sim::runtime::RankEidMap>(HcclSim::Storage::All<sim::runtime::RankEidMap>())
                .value_or(0U)
            == 0) {
            return false;
        }
    } else {
        HCCL_VM_ERROR("query RankEidMap by eid failed: {}", mapRet.diagnostic);
        return false;
    }
    return true;
}

// 把线上字节序的对端 EID 翻转为与 DB 存储一致的字节序。
HcclSim::Eid FlipToDbEid(const uint8_t* eid)
{
    HcclSim::Eid remoteEid{};
    for (uint32_t i = 0; i < sizeof(remoteEid.raw); ++i) {
        remoteEid.raw[i] = eid[sizeof(remoteEid.raw) - i - 1];
    }
    return remoteEid;
}

// 查 EID 解析结果, 未命中缓存时按 EID 索引点查并固化(含负结果)。
// 返回 true 表示结果就绪(命中与否由 found 给出); false 表示拓扑尚未就绪或查询出错。
bool LookupEidResolution(const HcclSim::Eid& remoteEid, EidResolution& out, bool& found)
{
    const std::string eidKey(reinterpret_cast<const char*>(remoteEid.raw), sizeof(remoteEid.raw));
    std::lock_guard<std::mutex> lock(g_eidResMapMtx);
    auto it = g_eidResMap.find(eidKey);
    if (it == g_eidResMap.end()) {
        EidResolution res;
        if (!ResolveEidByPointQueryLocked(eidKey, res)) {
            return false;
        }
        it = g_eidResMap.emplace(eidKey, res).first;
    }
    out = it->second;
    found = out.hasDevice || out.hasRank;
    return true;
}
} // namespace

bool GetRmtDeviceIdByEid(const uint8_t* eid, uint32_t& deviceId)
{
    if (eid == nullptr) {
        HCCL_VM_ERROR("remote eid is null");
        return false;
    }
    // EndPoint.eid 按字节序入库, 新 EID 高位含 die/port 编码不再满足 IPv4-compatible 格式,
    // 不能再用 inet_ntop+strcmp 匹配 ip_addr, 必须翻转后按 EID 字节比较。
    const HcclSim::Eid remoteEid = FlipToDbEid(eid);
    const HcclSim::IpAddress address(remoteEid);

    EidResolution res;
    bool found = false;
    if (!LookupEidResolution(remoteEid, res, found) || !found || !res.hasDevice) {
        HCCL_VM_ERROR("failed to resolve remote device by eid={}", address.EidToHexString());
        return false;
    }
    deviceId = res.deviceId;
    return true;
}

// 估算模式专用: 非估算(主线)路径不解析对端 rankId, rank 字段保持 0
bool GetRmtRankIdByEid(const uint8_t* eid, uint32_t& rankId)
{
    if (eid == nullptr) {
        HCCL_VM_ERROR("remote eid is null");
        return false;
    }
    const HcclSim::Eid remoteEid = FlipToDbEid(eid);
    const HcclSim::IpAddress address(remoteEid);
    // RankEidMap.eid 由 ranktable.json 按字节直译入库, 与 EndPoint.eid 同源;
    // estimate 模式 Communicator 表只有本 rank 一行, 对端 rankId 以 RankEidMap 为准。
    EidResolution res;
    bool found = false;
    if (!LookupEidResolution(remoteEid, res, found) || !found || !res.hasRank) {
        HCCL_VM_ERROR("failed to resolve rmtRankId by eid={}", address.EidToHexString());
        return false;
    }
    rankId = res.rankId;
    HCCL_VM_INFO("record rmtRankId[{}] by eid={}", rankId, address.EidToHexString());
    return true;
}
