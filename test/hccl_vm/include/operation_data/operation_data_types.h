/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIM_OP_DB_TYPES_H
#define SIM_OP_DB_TYPES_H

#include <string>
#include <vector>

#include "sim_capacity_limits.h"

namespace sim::operation {
enum class DbType {
    SQLITE3,
};
struct DBConfig {
    DbType type = DbType::SQLITE3;
    std::string dbPath = "data/hccl_vm_data.db";
    std::string host;
    uint16_t port = 0;
    std::string user;
    std::string password;
    std::string database;
};

struct OpDetailTab {
    uint32_t id;       // PK (自增)
    uint32_t pid;      // proxy 进程 ID
    uint32_t deviceId; // 物理设备 ID（全局唯一，内部逻辑统一使用）
    uint32_t rankId;   // 通信域内 rank ID
    uint64_t commId;   // 通信域成员 ID，FK -> Communicator.id
    uint32_t opIter;   // 算子迭代次数 (从 0 开始)
    uint32_t syncIter; // 所属 sync 周期 (从 0 开始)   // todo 可删

    uint64_t streamId;              // streamId
    uint32_t root;                  // root
    uint32_t opExpansionMode;       // 算子执行模式 0: ccu, 1: aicpu, 2: aiv
    uint32_t devType;               // 设备类型
    uint32_t rankSize;              // ransize
    uint32_t srcRank;               // srcRank
    uint32_t dstRank;               // dstRank
    std::vector<uint8_t> opDetail;  // BLOB
    std::vector<uint8_t> opExtInfo; // BLOB
};

struct OpMemInfoTab {
    uint32_t id;         // PK (自增)
    uint32_t opDetailId; // FK，关联 OpDetailTab
    uint64_t inputAddr;
    uint64_t inputSize;
    uint64_t outputAddr;
    uint64_t outputSize;
    uint64_t cclAddr;
    uint64_t cclSize;
};

struct SyncRecordTab {
    uint32_t id;  // PK (自增)
    uint32_t pid; // proxy 进程 ID
    uint32_t rankId;
    uint32_t rankSize;
    uint32_t syncIter; // sync 迭代次数 (从 0 开始)
    uint64_t streamId; // streamId
    uint8_t status;    // 0=pending, 2=completed
};

struct OpTaskTab {
    uint32_t id; // PK (自增)
    uint32_t pid{0};
    uint32_t opDetailId{0};          // FK，关联 OpDetailTables
    uint64_t deviceId{0};            // FK，关联 Device
    uint64_t streamId{0};            // FK，关联 Stream
    uint32_t taskType{UINT32_MAX};   // 任务类型
    bool isDone{false};              // runner是否执行完成
    std::vector<uint8_t> optaskMeta; // BLOB，动态大小
};

struct CcuChannelTab {
    uint32_t id;        // PK (自增)
    uint32_t channelId; // ccuChannelId
    uint32_t srcDieId;
    uint32_t dstDieId;
    uint32_t srcDeviceId; // 源端物理设备 ID
    uint32_t dstDeviceId; // 目的端物理设备 ID
    uint32_t srcRankId;   // 源端通信域 rank ID
    uint32_t dstRankId;   // 目的端通信域 rank ID
    uint8_t leid[16];
    uint8_t reid[16];
    uint16_t protocol;
    uint16_t jettyNum;
    uint32_t jettyId[HcclSim::HCCL_VM_MAX_CHANNEL_JETTIES];
};

struct JettyMapTab {
    uint32_t id;         // PK (自增)
    uint32_t opDetailId; // FK，关联 OpDetailTab
    uint32_t srcDieId;
    uint32_t dstDieId;
    uint32_t srcRankId;
    uint32_t dstRankId;
    uint8_t leid[16];
    uint8_t reid[16];
    uint16_t protocol;
};

struct CcuInstrResTab {
    uint32_t id; // PK (自增)
    uint32_t deviceId;
    uint32_t dieId;
    uint32_t instrCount;
    uint8_t instrSpace[32 * 1024][32]; // 指令空间1M: 32K个指令
};

struct CcuInstrTab {
    uint32_t id; // PK (自增)
    uint32_t ccuInstrResId;
    uint32_t startId;
    uint32_t instrInfoSize;
};

// 定义包含单卡完整信息的复合结构体
struct CompositeOpDetail {
    uint32_t deviceId; // 物理设备 ID（关键键值，用于 Map 排序）
    uint32_t rankId;   // 通信域内 rank ID
    uint64_t commId;   // 通信域成员 ID，用于解析所属通信域
    OpDetailTab detail;
    OpMemInfoTab memInfo;
    std::vector<OpTaskTab> tasks;
};

struct OpExecutionKey {
    std::string commName;
    uint64_t commHash{0};
    uint32_t opIter{0};

    bool operator<(const OpExecutionKey& other) const
    {
        if (commName != other.commName) {
            return commName < other.commName;
        }
        return opIter < other.opIter;
    }
};

struct OpExecutionIndexEntry {
    uint32_t opDetailId{0};
    uint32_t deviceId{0};
    uint32_t rankId{0};
    uint64_t commId{0};
    uint32_t opIter{0};
    uint32_t rankSize{0};
};

struct DeviceOpExecutionRecord {
    uint32_t deviceId{0};
    uint32_t rankId{0};
    OpDetailTab detail;
    OpMemInfoTab memInfo;
    std::vector<OpTaskTab> tasks;
};

struct OpExecution {
    OpExecutionKey key;
    // deviceId can be non-contiguous; use rankId for communication semantics.
    std::vector<DeviceOpExecutionRecord> deviceRecords;
};

// 0.5RTT特性数据信息
struct HalfRTTTab {
    uint32_t id; // PK (自增)
    uint32_t deviceId;
    uint16_t dieId;
    uint32_t wishCntXnIdBegin;
    uint32_t wishCntXnIdEnd;
    uint32_t totalCntId;
};
} // namespace sim::operation
#endif
