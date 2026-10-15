/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_COMMON_DEFS_H
#define HCCL_COMMON_DEFS_H

#include <cstdint>
#include <cstring>
#include <vector>

#include "hccl_types.h"
#include "sim_profiling_defs.h"

#define RT_CCU_SQE_ARGS_LEN (13U)

enum class LinkProto {
    SDMA = 0,
    RDMA = 1,
    CCU = 2,
    INVALID_A = 3,
};

enum class HccLTaskMetaType : char {
    NOTIFY_WAIT,
    NOTIFY_RECORD,
    REDUCE,
    MEM_CPY,
    CCU_GRAPH,
    AIV_GRAPH,
    SYNC_STREAM,
    MODEL_EXEC,
    WRITE_WITH_NOTIFY
};

typedef enum {
    COMM_PROTOCOL_RESERVED = -1,
    COMM_PROTOCOL_HCCS = 0,
    COMM_PROTOCOL_TCP = 1,
    COMM_PROTOCOL_ROCE = 2,
    COMM_PROTOCOL_UB_CTP = 3,
    COMM_PROTOCOL_UB_TP = 4,
    COMM_PROTOCOL_PCIE = 5,
    COMM_PROTOCOL_SIO = 6,
} CommProtocol;

typedef enum {
    TASK_SRC_DEFAULT = 0,
    TASK_SRC_DPU = 1,
} HcclTaskSrc;

#pragma pack(push, 1)

// estimate模式下deviceId字段承载通信域rankId
typedef struct {
    uint32_t srcDeviceId;
    uint64_t srcOffset;
    uint32_t dstDeviceId;
    uint64_t dstOffset;
    uint64_t len;
    uint8_t protocol;
} TransMemTask;

typedef struct {
    uint64_t syncIdx;
} SyncStreamTask;

// 与 include/sim_common_defs.h 的 ModelExecTask 二进制布局保持一致（pack 1）
typedef struct {
    uint32_t execSeq; // 同一次 rtModelExecute 的 4 个任务共享
    uint8_t role;     // 0=START, 1=START_SUB, 2=END_SUB, 3=END
    uint64_t modelId;
    uint64_t peerStreamId;
} ModelExecTask;

typedef struct {
    uint32_t srcDeviceId;
    uint64_t srcOffset;
    uint32_t dstDeviceId;
    uint64_t dstOffset;
    uint64_t dataCount;
    uint8_t dataType;
    uint8_t reduceOp;
    uint8_t protocol;
    uint8_t resv;
} ReduceTask;

typedef struct {
    uint32_t srcDeviceId;
    uint64_t notifyId;
    uint32_t dstDeviceId;
    uint8_t notifyCount;
    uint8_t protocol;
} NotifyTask;

typedef struct {
    uint32_t srcDeviceId;
    uint64_t srcOffset;
    uint32_t dstDeviceId;
    uint64_t dstOffset;
    uint64_t len; // MEM_CPY: 传输长度; REDUCE: dataCount(实际为size)
    uint64_t notifyId;
    uint8_t isReduce; // 0: 写部分为MEM_CPY; 1: 写部分为REDUCE
    uint8_t dataType; // REDUCE 载荷: HcclDataType
    uint8_t reduceOp; // REDUCE 载荷: HcclReduceOp
    uint8_t protocol;
    uint8_t notifyCount;
} WriteWithNotifyTask;

typedef struct {
    uint8_t dieId;
    uint8_t missionId;
    uint16_t timeout;
    uint16_t instStartId;
    uint16_t instCnt;
    uint32_t key;
    uint32_t argSize;
    uint64_t args[RT_CCU_SQE_ARGS_LEN];
} CcuTask;

typedef struct {
    uint64_t launchIdx;
} AivTask;

typedef struct HcclTaskMetaData {
    HccLTaskMetaType taskType;
    HcclTaskProfInfo profInfo;
    uint64_t commId;
    uint32_t rankId;
    uint64_t deviceId;
    uint64_t streamId;
    uint32_t jettyId;
    uint8_t sqeGroupId; // 多wqe分组标记，环形递增1~255(0=未打标)
    uint8_t taskSrc;
    uint8_t rmEid[16];
    uint8_t lcEid[16];
    union {
        TransMemTask transMem;
        ReduceTask reduce;
        NotifyTask notify;
        CcuTask ccu;
        AivTask aiv;
        SyncStreamTask syncStreamTask;
        ModelExecTask modelExec;
        WriteWithNotifyTask writeWithNotify;
    } taskData;
    HcclTaskMetaData()
    {
        taskType = HccLTaskMetaType::SYNC_STREAM;
        profInfo = HcclTaskProfInfo();
        commId = 0;
        rankId = UINT32_MAX;
        deviceId = UINT64_MAX;
        streamId = UINT64_MAX;
        jettyId = UINT32_MAX;
        sqeGroupId = 0;
        taskSrc = TASK_SRC_DEFAULT;
        std::memset(rmEid, 0, sizeof(rmEid));
        std::memset(lcEid, 0, sizeof(lcEid));
        std::memset(&taskData, 0, sizeof(taskData));
    }
} HcclTaskMetaData;

typedef struct {
    uint64_t taskCid;
    uint16_t dispatchId;
} HcclTaskReq;

typedef struct {
    uint64_t taskCid;
    uint16_t status;
} HcclTaskRsp;

#pragma pack(pop)

// WRITE_WITH_NOTIFY 复合任务展开: 拆为 [写子任务(MEM_CPY/REDUCE), NOTIFY_RECORD] 两个子任务
inline void
SplitWriteWithNotifyTask(const HcclTaskMetaData& task, HcclTaskMetaData& outWrite, HcclTaskMetaData& outNotify)
{
    const WriteWithNotifyTask& wn = task.taskData.writeWithNotify;
    outWrite = task;
    outNotify = task;
    if (wn.isReduce == 1) {
        outWrite.taskType = HccLTaskMetaType::REDUCE;
        outWrite.taskData.reduce.srcDeviceId = wn.srcDeviceId;
        outWrite.taskData.reduce.srcOffset = wn.srcOffset;
        outWrite.taskData.reduce.dstDeviceId = wn.dstDeviceId;
        outWrite.taskData.reduce.dstOffset = wn.dstOffset;
        outWrite.taskData.reduce.dataCount = wn.len;
        outWrite.taskData.reduce.dataType = wn.dataType;
        outWrite.taskData.reduce.reduceOp = wn.reduceOp;
        outWrite.taskData.reduce.protocol = wn.protocol;
        outWrite.taskData.reduce.resv = 0;
    } else {
        outWrite.taskType = HccLTaskMetaType::MEM_CPY;
        outWrite.taskData.transMem.srcDeviceId = wn.srcDeviceId;
        outWrite.taskData.transMem.srcOffset = wn.srcOffset;
        outWrite.taskData.transMem.dstDeviceId = wn.dstDeviceId;
        outWrite.taskData.transMem.dstOffset = wn.dstOffset;
        outWrite.taskData.transMem.len = wn.len;
        outWrite.taskData.transMem.protocol = wn.protocol;
    }
    outNotify.taskType = HccLTaskMetaType::NOTIFY_RECORD;
    outNotify.taskData.notify.srcDeviceId = wn.srcDeviceId;
    outNotify.taskData.notify.notifyId = wn.notifyId;
    outNotify.taskData.notify.dstDeviceId = wn.dstDeviceId;
    outNotify.taskData.notify.notifyCount = wn.notifyCount;
    outNotify.taskData.notify.protocol = wn.protocol;
}

// 统一 Checker 解析 blob 并展开 WRITE_WITH_NOTIFY
inline void DecodeAndExpandTaskMeta(const std::vector<uint8_t>& blob, std::vector<HcclTaskMetaData>& out)
{
    HcclTaskMetaData taskMeta;
    std::memcpy(&taskMeta, blob.data(), sizeof(HcclTaskMetaData));
    if (taskMeta.taskType == HccLTaskMetaType::WRITE_WITH_NOTIFY) {
        HcclTaskMetaData writeMeta;
        HcclTaskMetaData notifyMeta;
        SplitWriteWithNotifyTask(taskMeta, writeMeta, notifyMeta);
        out.push_back(writeMeta);
        out.push_back(notifyMeta);
    } else {
        out.push_back(taskMeta);
    }
}

#pragma pack(push, 1)
typedef struct {
    uint64_t count;
    uint16_t dataType;
    uint32_t reduceOpType;
} OpInfoV1;

typedef struct {
    uint64_t sendCount;
    uint16_t sendDataType;
    uint64_t recvCount;
    uint16_t recvDataType;
    uint16_t extInfo;
} OpInfoV2;

typedef struct {
    uint16_t opType;
    uint16_t dataType;
    uint16_t reduceType;
    union {
        OpInfoV1 opV1;
        OpInfoV2 opV2;
    };
} OpDetails;
#pragma pack(pop)

#endif
