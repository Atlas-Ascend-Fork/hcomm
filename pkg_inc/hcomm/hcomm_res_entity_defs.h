/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_RES_ENTITY_DEFS_H
#define HCOMM_RES_ENTITY_DEFS_H

#include <stdint.h>
#include "hcomm_res_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef uint64_t ChannelPtr;

typedef enum {
    PROTECTION_TYPE_INVALID = -1,
    PROTECTION_TYPE_ROCE = 0,
    PROTECTION_TYPE_UB = 1,
} ProtectionType;

typedef enum {
    SQ_CONTEXT_TYPE_INVALID = -1,
    SQ_CONTEXT_TYPE_UB_JFS = 0,
    SQ_CONTEXT_TYPE_ROCE = 1,
} SqContextType;

typedef enum {
    CQ_CONTEXT_TYPE_INVALID = -1,
    CQ_CONTEXT_TYPE_UB_JFC = 0,
    CQ_CONTEXT_TYPE_ROCE = 1,
} CqContextType;

typedef enum {
    REGED_NOTIFY_INVALID = -1,
    REGED_NOTIFY_IPC_RT = 0,
    REGED_NOTIFY_IPC_MEM = 1,
    REGED_NOTIFY_RMA_RT = 2,
    REGED_NOTIFY_RMA_MEM = 3,
} RegedNotifyType;

typedef enum {
    REGED_BUFFER_INVALID = -1,
    REGED_BUFFER_IPC = 0,
    REGED_BUFFER_RMA = 1,
} RegedBufferType;

typedef struct {
    ProtectionType type;
    union {
        struct {
            uint32_t lkey;
            uint32_t rkey;
        } roce;
        struct {
            uint32_t tokenId;
            uint32_t tokenValue;
        } ub;
        uint8_t raws[24];
    } memInfo;
} ProtectionInfo; // 32B

typedef struct {
    RegedBufferType type;
    union {
        struct {
            uint64_t addr;
            uint64_t size;
        } ipc;
        struct {
            uint64_t addr;
            uint64_t size;
            ProtectionInfo protectionInfo;
        } rma;
        uint8_t raws[56];
    } bufferInfo;
} RegedBufferEntity; // 64B

typedef struct {
    RegedNotifyType type;
    union {
        struct {
            uint64_t addr;
            uint32_t size;
            int32_t notifyId;
        } ipcRt; // Thread 之间 Notify 填 ipcRt.notifyId。
        struct {
            uint64_t addr;
            uint32_t size;
        } ipcMem;
        struct {
            uint64_t addr;
            uint32_t size;
            int32_t notifyId;
            ProtectionInfo protectionInfo;
        } rmaRt;
        struct {
            uint64_t addr;
            uint32_t size;
            ProtectionInfo protectionInfo;
        } rmaMem;
        uint8_t raws[56];
    } notifyInfo;
} RegedNotifyEntity; // 64B

typedef struct {
    SqContextType type;
    union {
        struct {
            uint64_t sqVa;
            uint64_t headAddr;
            uint64_t tailAddr;
            uint64_t dbVa;
            uint32_t jfsID;
            uint32_t wqeSize;
            uint32_t sqDepth;
            uint32_t tpID;
            uint8_t remoteEID[16];
            uint32_t sqHead;
            uint32_t sqTail;
            uint32_t maxReadSize;
            uint32_t maxWriteSize;
        } ubJfs;
        struct {
            uint64_t sqVa;
            uint64_t headAddr;
            uint64_t tailAddr;
            uint64_t dbHwVa;
            uint64_t dbSwVa;
            uint32_t qpn;
            uint32_t wqeSize;
            uint32_t depth;
            uint8_t sl;
            uint64_t dbVendorSpecified;
            uint32_t sqHead;
            uint32_t sqTail;
        } roceSq;
        uint8_t raws[112];
    } contextInfo;
    uint8_t fence;
    uint8_t reserve[3];
} SqContext;

typedef struct {
    CqContextType type;
    union {
        struct {
            uint64_t scqVa;
            uint64_t headAddr;
            uint64_t tailAddr;
            uint64_t dbVa;
            uint32_t jfcID;
            uint32_t cqeSize;
            uint32_t cqDepth;
            uint32_t cqHead;
            uint32_t cqTail;
        } ubJfc;
        struct {
            uint64_t cqVa;
            uint64_t headAddr;
            uint64_t tailAddr;
            uint64_t dbHwVa;
            uint64_t dbSwVa;
            uint32_t cqn;
            uint32_t cqeSize;
            uint32_t cqDepth;
            uint64_t dbVendorSpecified;
            uint32_t cqHead;
            uint32_t cqTail;
            uint8_t cqDbFlush;
        } roceCq;
        uint8_t raws[120];
    } contextInfo;
} CqContext;

typedef struct {
    uint32_t chipVersion; // 预留芯片版本字段，当前默认0。
    uint32_t devPhyId;    // Host 侧 NPU 物理设备 ID；Device 侧可转换为本地设备 ID。
    uint8_t dieId;
    uint8_t funcId;
    uint8_t reserve[30];
} HcommDevInfo; // 40B

typedef struct {
    CommAbiHeader abiHeader;
    CommEngine engine;
    CommProtocol protocol;
    uint32_t localNotifyNum;
    uint32_t remoteNotifyNum;
    uint32_t localBufferNum;
    uint32_t remoteBufferNum;
    uint32_t sqNum;
    uint32_t cqNum;
    RegedNotifyEntity* localNotifyAddr;
    RegedNotifyEntity* remoteNotifyAddr;
    RegedBufferEntity* localBufferAddr;
    RegedBufferEntity* remoteBufferAddr;
    SqContext* sqContextAddr;
    CqContext* cqContextAddr;
    HcommDevInfo devInfo;
    ChannelHandle compatChannelHandle;
    uint8_t reserve[112];
} ChannelEntity; // 256B

typedef struct {
    uint32_t type;
    union {
        struct {
            uint32_t streamId;
            uint32_t sqId;
            uint64_t sqBaseAddr;
            uint32_t sqDepth;
            uint32_t sqHead;
            uint32_t sqTail;
            uint32_t taskId;
            uint32_t taskIdEnd;
            uint64_t localSqAddr;
            uint32_t localSqDepth;
            uint32_t pendingSqeCnt;
        } RtsqContext;
        uint8_t raws[120];
    } contextInfo;
} HcommThreadSqContext; // 128B

typedef struct {
    CommAbiHeader abiHeader;
    HcommDevInfo devInfo;
    uint32_t localNotifyNum;
    RegedNotifyEntity* localNotifyAddr; // Thread IPC_RT 仅 notifyId 有效，addr/size 为 0。
    HcommThreadSqContext* sqContextAddr;
    // Device 侧 AICPU Thread 对象地址，仅供 hcomm 内部兼容解包，跨仓消费者不得解引用。
    ThreadHandle compatThreadHandle;
    uint8_t reserve[168];
} HcommThreadEntity; // 256B

// AICPU数据面跨SO资源ABI。ChannelEntity/HcommThreadEntity同时作为hcomm内部实体和
// asc-comm按句柄直取的公开POD契约，必须保持跨仓二进制布局一致。
#define HCOMM_CHANNEL_ENTITY_VERSION 0U
#define HCOMM_CHANNEL_ENTITY_MAGIC_WORD 0x0fcf0f5fU
#define HCOMM_THREAD_ENTITY_VERSION 0U
#define HCOMM_THREAD_ENTITY_MAGIC_WORD 0x0fcf0f6fU

#ifdef __cplusplus
}

#endif

#endif // HCOMM_RES_ENTITY_DEFS_H
