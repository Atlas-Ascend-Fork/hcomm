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
#include <set>
#include <unordered_map>

#include "hccl_task_collection.h"
#include "operation_data/operation_data_ops.h"
#include "runtime_state/db_sim_runner_common.h"
#include "sim_capacity_limits.h"
#include "sim_log.h"
#include "sim_ip_address.h"
#include "sqe_v82_stub.h"
#include "udma_data_struct_stub.h"

constexpr int SHIFT_BIT32 = 32;

struct TaskIdentity {
    uint64_t commId;
    uint32_t rankId;
    uint32_t deviceId;
};

namespace {
bool ResolveCurrentTaskIdentity(TaskIdentity& identity)
{
    if (sim::operation::g_currOpDetailId == 0
        || sim::operation::QueryOpDetailIdentity(
               sim::operation::g_currOpDetailId, identity.commId, identity.rankId, identity.deviceId)
               != 0
        || identity.commId == 0) {
        HCCL_VM_ERROR("cannot resolve task identity for opDetailId={}", sim::operation::g_currOpDetailId);
        return false;
    }
    return true;
}

void SetTaskIdentity(HcclTaskMetaData& task, const TaskIdentity& identity)
{
    task.commId = identity.commId;
    task.rankId = identity.rankId;
    task.deviceId = identity.deviceId;
}

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

void ParseDavidSDMASqeWithIdentity(uint32_t streamId, void* sqeBuf, const TaskIdentity& identity);
void ParseDavidNotifySqeWithIdentity(uint32_t streamId, void* sqeBuf, bool isPost, const TaskIdentity& identity);
void ParseDavidUDMASqeWithIdentity(uint32_t streamId, void* sqeBuf, const TaskIdentity& identity);
void ParseDavidUBReadWriteSqeWithIdentity(
    uint64_t wqeAddr, uint16_t streamId, uint32_t jettyId, bool isRead, const TaskIdentity& identity);
void ParseDavidUBWriteWithNotifySqeWithIdentity(
    uint64_t wqeAddr, uint16_t streamId, uint32_t jettyId, const TaskIdentity& identity);

void ParseA5SqeFromSqBuffer(uint32_t devId, struct halSqCqConfigInfo* info)
{
    TaskIdentity identity{};
    if (!ResolveCurrentTaskIdentity(identity)) {
        HCCL_VM_ERROR("cannot parse SQE without current task identity");
        return;
    }
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
                ParseDavidSDMASqeWithIdentity(streamId, sqeBuf, identity);
                break;
            }
            case static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT): {
                ParseDavidNotifySqeWithIdentity(streamId, sqeBuf, false, identity);
                break;
            }
            case static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_RECORD): {
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

    if (sqe->opcode == 0) { // 表示是memcpy
        taskMeta.taskType = HccLTaskMetaType::MEM_CPY;
        taskMeta.taskData.transMem.srcOffset = srcOffset;
        taskMeta.taskData.transMem.dstOffset = dstOffset;
        taskMeta.taskData.transMem.dstDeviceId = dstDeviceId;
        taskMeta.taskData.transMem.srcDeviceId = srcDeviceId;
        taskMeta.taskData.transMem.len = length;
        InsertTaskToCollectionDev(&taskMeta);
        PrintTaskMetaData(taskMeta);
    } else { // reduce
        taskMeta.taskType = HccLTaskMetaType::REDUCE;
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

void ParseDavidNotifySqeWithIdentity(uint32_t streamId, void* sqeBuf, bool isPost, const TaskIdentity& identity)
{
    Rt91095StarsNotifySqe* sqe = reinterpret_cast<Rt91095StarsNotifySqe*>(sqeBuf);
    HcclTaskMetaData taskMeta;
    SetTaskIdentity(taskMeta, identity);
    taskMeta.streamId = streamId;
    taskMeta.jettyId = UINT32_MAX;
    taskMeta.taskType = isPost ? HccLTaskMetaType::NOTIFY_RECORD : HccLTaskMetaType::NOTIFY_WAIT;
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

void ParseDavidUDMASqeWithIdentity(uint32_t streamId, void* sqeBuf, const TaskIdentity& identity)
{
    Rt91095StarsUbdmaDBmodeSqe* ubSqe = reinterpret_cast<Rt91095StarsUbdmaDBmodeSqe*>(sqeBuf);
    uint32_t jettyId = ubSqe->jettyId1;
    uint64_t wqeBuffer = 0;
    if (!GetWqebufferByJettyId(jettyId, wqeBuffer) || wqeBuffer == 0) {
        HCCL_VM_ERROR("GetWqebufferByJettyId failed or returned null, jettyId[{}].", jettyId);
        return;
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
                ParseDavidUBReadWriteSqeWithIdentity(wqeAddr, streamId, jettyId, false, identity);
                break;
            }
            case static_cast<int>(UdmaSqOpcode::UDMA_OPC_WRITE_WITH_NOTIFY): {
                ParseDavidUBWriteWithNotifySqeWithIdentity(wqeAddr, streamId, jettyId, identity);
                index++; // 占128字节两个WQE
                break;
            }
            case static_cast<int>(UdmaSqOpcode::UDMA_OPC_READ): {
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

void ParseDavidUBReadWriteSqeWithIdentity(
    uint64_t wqeAddr, uint16_t streamId, uint32_t jettyId, bool isRead, const TaskIdentity& identity)
{
    UdmaSqeWrite* ubWqe = reinterpret_cast<UdmaSqeWrite*>(wqeAddr);
    HcclTaskMetaData taskMeta;
    SetTaskIdentity(taskMeta, identity);
    taskMeta.streamId = streamId;
    taskMeta.jettyId = jettyId;
    memcpy(taskMeta.rmEid, ubWqe->comm.rmtEid, sizeof(taskMeta.rmEid));

    uint32_t rmtDeviceId = 0;
    if (!GetRmtDeviceIdByEid(reinterpret_cast<const uint8_t*>(ubWqe->comm.rmtEid), rmtDeviceId)) {
        HCCL_VM_ERROR(
            "skip UDMA SQE because remote device resolution failed, streamId={}, jettyId={}", streamId, jettyId);
        return;
    }
    uint32_t rmtRankId = 0;
    if (!sim::runtime::GetCommRankByDeviceId(identity.commId, rmtDeviceId, rmtRankId)) {
        HCCL_VM_ERROR(
            "failed to resolve remote rank: deviceId={}, commId={}, streamId={}, jettyId={}", rmtDeviceId,
            identity.commId, streamId, jettyId);
        return;
    }
    // case1:UbConnLite::InlineWrite 写Notify
    if (ubWqe->comm.inlineEn == 1) {
        uint64_t notifyAddr = GetFull64BitAddr(ubWqe->comm.rmtAddrLow, ubWqe->comm.rmtAddrHigh);
        taskMeta.taskType = HccLTaskMetaType::NOTIFY_RECORD;
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
    taskMeta.taskData.transMem.srcOffset = srcOffset;
    taskMeta.taskData.transMem.dstOffset = dstOffset;
    taskMeta.taskData.transMem.dstDeviceId = dstDeviceId;
    taskMeta.taskData.transMem.srcDeviceId = srcDeviceId;
    taskMeta.taskData.transMem.len = length;

    // case3:UbConnLite::WriteReduce Reduce
    if (ubWqe->comm.udfFlag == 1) {
        taskMeta.taskType = HccLTaskMetaType::REDUCE;
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

void ParseDavidUBWriteWithNotifySqeWithIdentity(
    uint64_t wqeAddr, uint16_t streamId, uint32_t jettyId, const TaskIdentity& identity)
{
    UdmaSqeWriteWithNotify* ubWqe = reinterpret_cast<UdmaSqeWriteWithNotify*>(wqeAddr);
    HcclTaskMetaData taskMeta1;
    SetTaskIdentity(taskMeta1, identity);
    taskMeta1.streamId = streamId;
    taskMeta1.jettyId = jettyId;
    memcpy(taskMeta1.rmEid, ubWqe->comm.rmtEid, sizeof(taskMeta1.rmEid));
    uint32_t rmtDeviceId = 0;
    if (!GetRmtDeviceIdByEid(reinterpret_cast<const uint8_t*>(ubWqe->comm.rmtEid), rmtDeviceId)) {
        HCCL_VM_ERROR(
            "skip UDMA write-with-notify SQE because remote device resolution failed, "
            "streamId={}, jettyId={}",
            streamId, jettyId);
        return;
    }
    uint32_t rmtRankId = 0;
    if (!sim::runtime::GetCommRankByDeviceId(identity.commId, rmtDeviceId, rmtRankId)) {
        HCCL_VM_ERROR(
            "failed to resolve remote rank: deviceId={}, commId={}, streamId={}, jettyId={}", rmtDeviceId,
            identity.commId, streamId, jettyId);
        return;
    }
    // 1.先构造MEM_CPY(或SDMA_REDUCE)所需参数
    // 数据段 src=当前设备映射、dst=远端设备映射，各按 (映射设备, 映射地址)
    // 一次定位；notify 身份与任务顺序不受地址解析影响。
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
    taskMeta1.taskType = HccLTaskMetaType::MEM_CPY;
    taskMeta1.taskData.transMem.srcOffset = srcOffset;
    taskMeta1.taskData.transMem.dstOffset = dstOffset;
    taskMeta1.taskData.transMem.dstDeviceId = dstDeviceId;
    taskMeta1.taskData.transMem.srcDeviceId = srcDeviceId;
    taskMeta1.taskData.transMem.len = length;
    if (ubWqe->comm.udfFlag == 1) { // Reduce操作标识
        taskMeta1.taskType = HccLTaskMetaType::REDUCE;
        taskMeta1.taskData.reduce.reduceOp = ParseUbReduceTypeDavid(ubWqe->comm.inlinedata.udfData.reduceOp);
        taskMeta1.taskData.reduce.dataType = ParseUbDataTypeDavid(ubWqe->comm.inlinedata.udfData.reduceType);
        taskMeta1.taskData.reduce.srcOffset = srcOffset;
        taskMeta1.taskData.reduce.dstOffset = dstOffset;
        taskMeta1.taskData.reduce.dstDeviceId = dstDeviceId;
        taskMeta1.taskData.reduce.srcDeviceId = srcDeviceId;
        taskMeta1.taskData.reduce.dataCount = length;
    }
    PrintTaskMetaData(taskMeta1);
    InsertTaskToCollectionDev(&taskMeta1);

    // 2.再构造NOTIFY_RECORD所需参数
    HcclTaskMetaData taskMeta2;
    SetTaskIdentity(taskMeta2, identity);
    taskMeta2.streamId = streamId;
    taskMeta2.jettyId = jettyId;
    memcpy(taskMeta2.rmEid, ubWqe->comm.rmtEid, sizeof(taskMeta2.rmEid));

    uint64_t notifyAddr = GetFull64BitAddr(ubWqe->notify.notifyAddrLow, ubWqe->notify.notifyAddrHigh);
    taskMeta2.taskType = HccLTaskMetaType::NOTIFY_RECORD;
    const uint64_t notifyTaskId = HcclSim::MakeNotifyTaskId(rmtRankId, notifyAddr);
    taskMeta2.taskData.notify.notifyId = notifyTaskId;
    SetRemoteNotifyEndpoints(taskMeta2, rmtDeviceId);
    PrintTaskMetaData(taskMeta2);
    InsertTaskToCollectionDev(&taskMeta2);
}

void ParseDavidSDMASqe(uint32_t streamId, void* sqeBuf)
{
    TaskIdentity identity{};
    if (ResolveCurrentTaskIdentity(identity)) {
        ParseDavidSDMASqeWithIdentity(streamId, sqeBuf, identity);
    }
}

void ParseDavidNotifySqe(uint32_t streamId, void* sqeBuf, bool isPost)
{
    TaskIdentity identity{};
    if (ResolveCurrentTaskIdentity(identity)) {
        ParseDavidNotifySqeWithIdentity(streamId, sqeBuf, isPost, identity);
    }
}

void ParseDavidUDMASqe(uint32_t streamId, void* sqeBuf)
{
    TaskIdentity identity{};
    if (ResolveCurrentTaskIdentity(identity)) {
        ParseDavidUDMASqeWithIdentity(streamId, sqeBuf, identity);
    }
}

void ParseDavidUBReadWriteSqe(uint64_t wqeAddr, uint16_t streamId, uint32_t jettyId, bool isRead)
{
    TaskIdentity identity{};
    if (ResolveCurrentTaskIdentity(identity)) {
        ParseDavidUBReadWriteSqeWithIdentity(wqeAddr, streamId, jettyId, isRead, identity);
    }
}

void ParseDavidUBWriteWithNotifySqe(uint64_t wqeAddr, uint16_t streamId, uint32_t jettyId)
{
    TaskIdentity identity{};
    if (ResolveCurrentTaskIdentity(identity)) {
        ParseDavidUBWriteWithNotifySqeWithIdentity(wqeAddr, streamId, jettyId, identity);
    }
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
    switch (taskMeta.taskType) {
        case HccLTaskMetaType::MEM_CPY:
            HCCL_VM_INFO(
                "pid{}: rankId[{}] streamId[{}] taskType[MEM_CPY], srcOffset[{}] dstOffset[{}], len[{}], "
                "srcDeviceId[{}], dstDeviceId[{}]",
                pid, taskMeta.rankId, taskMeta.streamId, taskMeta.taskData.transMem.srcOffset,
                taskMeta.taskData.transMem.dstOffset, taskMeta.taskData.transMem.len,
                taskMeta.taskData.transMem.srcDeviceId, taskMeta.taskData.transMem.dstDeviceId);
            break;
        case HccLTaskMetaType::REDUCE:
            HCCL_VM_INFO(
                "pid{}: rankId[{}] streamId[{}] taskType[REDUCE], srcOffset[{}] dstOffset[{}], len[{}], "
                "srcDeviceId[{}], dstDeviceId[{}], reduceOp[%{}], dataType[%{}]",
                pid, taskMeta.rankId, taskMeta.streamId, taskMeta.taskData.reduce.srcOffset,
                taskMeta.taskData.reduce.dstOffset, taskMeta.taskData.reduce.dataCount,
                taskMeta.taskData.reduce.srcDeviceId, taskMeta.taskData.reduce.dstDeviceId,
                taskMeta.taskData.reduce.reduceOp, taskMeta.taskData.reduce.dataType);
            break;
        case HccLTaskMetaType::NOTIFY_WAIT:
            HCCL_VM_INFO(
                "pid{}: rankId[{}] streamId[{}] taskType[NOTIFY_WAIT], notifyId[{}], recordDeviceId[{}], "
                "waitDeviceId[{}]",
                pid, taskMeta.rankId, taskMeta.streamId, taskMeta.taskData.notify.notifyId,
                taskMeta.taskData.notify.srcDeviceId, taskMeta.taskData.notify.dstDeviceId);
            break;
        case HccLTaskMetaType::NOTIFY_RECORD:
            HCCL_VM_INFO(
                "pid{}: rankId[{}] streamId[{}] taskType[NOTIFY_RECORD], notifyId[{}], recordDeviceId[{}], "
                "waitDeviceId[{}]",
                pid, taskMeta.rankId, taskMeta.streamId, taskMeta.taskData.notify.notifyId,
                taskMeta.taskData.notify.srcDeviceId, taskMeta.taskData.notify.dstDeviceId);
            break;
        default:
            break;
    }
}

bool GetRmtDeviceIdByEid(const uint8_t* eid, uint32_t& deviceId)
{
    if (eid == nullptr) {
        HCCL_VM_ERROR("remote eid is null");
        return false;
    }
    HcclSim::Eid remoteEid{};
    for (uint32_t i = 0; i < sizeof(remoteEid.raw); ++i) {
        remoteEid.raw[i] = eid[sizeof(remoteEid.raw) - i - 1];
    }
    HcclSim::IpAddress address(remoteEid);
    // topo 侧 EndPoint.eid 按十六进制串字节序入库，与翻转后的 remoteEid 一致；
    // 新 EID 高位含 die/port 编码，不再满足 IPv4-compatible 格式，
    // 不能再用 inet_ntop 字符串与 ip_addr 做 strcmp 匹配，必须按 EID 字节比较。
    sim::runtime::EndPoint endPoint{};
    if (sim::runtime::GetEndPointByEidOrIpV4(address, endPoint) != 0) {
        HCCL_VM_ERROR("failed to resolve remote device by eid={}", address.EidToHexString());
        return false;
    }
    deviceId = static_cast<uint32_t>(endPoint.device_id);
    return true;
}
