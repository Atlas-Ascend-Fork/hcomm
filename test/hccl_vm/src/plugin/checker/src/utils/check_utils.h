/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV1_CHECK_UTILS_H
#define HCCLV1_CHECK_UTILS_H

#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <vector>

#include "binary_data_type_pub.h"
#include "checker_def.h"
#include "enum_factory.h"
#include "log.h"
#include "sim_common.h"
#include "string_util.h"
#include "task_def_v3.h"

namespace HcclSim {
inline std::string BufferTypeToString(BufferType bufferType)
{
    switch (bufferType) {
        case BufferType::INPUT:
            return "INPUT";
        case BufferType::OUTPUT:
            return "OUTPUT";
        case BufferType::CCL:
            return "CCL";
        case BufferType::SCRATCH:
            return "SCRATCH";
        case BufferType::INPUT_AIV:
            return "INPUT_AIV";
        case BufferType::OUTPUT_AIV:
            return "OUTPUT_AIV";
        case BufferType::AIV_COMMINFO:
            return "AIV_COMMINFO";
        case BufferType::USERBUF_AIV:
            return "USERBUF_AIV";
        case BufferType::MS:
            return "MS";
        default:
            return "INVALID";
    }
}

inline std::string HcclCmdTypeToString(HcclCMDType cmdType)
{
    switch (cmdType) {
        case HCCL_CMD_BROADCAST:
            return "Broadcast";
        case HCCL_CMD_ALLREDUCE:
            return "AllReduce";
        case HCCL_CMD_REDUCE:
            return "Reduce";
        case HCCL_CMD_SEND:
            return "Send";
        case HCCL_CMD_RECEIVE:
            return "Recv";
        case HCCL_CMD_ALLGATHER:
            return "AllGather";
        case HCCL_CMD_REDUCE_SCATTER:
            return "ReduceScatter";
        case HCCL_CMD_ALLTOALLV:
            return "AllToAllV";
        case HCCL_CMD_ALLTOALLVC:
            return "AllToAllVC";
        case HCCL_CMD_ALLTOALL:
            return "AllToAll";
        case HCCL_CMD_SCATTER:
            return "Scatter";
        case HCCL_CMD_BATCH_SEND_RECV:
            return "BatchSendRecv";
        case HCCL_CMD_ALLGATHER_V:
            return "AllGatherV";
        case HCCL_CMD_REDUCE_SCATTER_V:
            return "ReduceScatterV";
        default:
            return "Unknown";
    }
}

inline std::string HcclDataTypeToString(HcclDataType dataType)
{
    switch (dataType) {
        case HCCL_DATA_TYPE_INT8:
            return "INT8";
        case HCCL_DATA_TYPE_INT16:
            return "INT16";
        case HCCL_DATA_TYPE_INT32:
            return "INT32";
        case HCCL_DATA_TYPE_FP16:
            return "FP16";
        case HCCL_DATA_TYPE_FP32:
            return "FP32";
        case HCCL_DATA_TYPE_INT64:
            return "INT64";
        case HCCL_DATA_TYPE_UINT64:
            return "UINT64";
        case HCCL_DATA_TYPE_UINT8:
            return "UINT8";
        case HCCL_DATA_TYPE_UINT16:
            return "UINT16";
        case HCCL_DATA_TYPE_UINT32:
            return "UINT32";
        case HCCL_DATA_TYPE_FP64:
            return "FP64";
        case HCCL_DATA_TYPE_BFP16:
            return "BFP16";
        default:
            return "Unknown";
    }
}

inline std::string ReduceOpToString(HcclReduceOp reduceOp)
{
    switch (reduceOp) {
        case HCCL_REDUCE_SUM:
            return "HCCL_REDUCE_SUM";
        case HCCL_REDUCE_PROD:
            return "HCCL_REDUCE_PROD";
        case HCCL_REDUCE_MAX:
            return "HCCL_REDUCE_MAX";
        case HCCL_REDUCE_MIN:
            return "HCCL_REDUCE_MIN";
        case HCCL_REDUCE_RESERVED:
            return "HCCL_REDUCE_RESERVED";
        default:
            return "INVALID";
    }
}

struct SrcBufDes {
    DeviceId deviceId;  // 数据源的deviceId
    BufferType bufType; // 数据源的内存类型
    uint64_t srcAddr;   // 数据源的地址
    SrcBufDes(DeviceId id, BufferType type, uint64_t addr) : deviceId(id), bufType(type), srcAddr(addr) {}
    inline bool operator<(const SrcBufDes& another) const { return deviceId < another.deviceId; }

    std::string Describe() const
    {
        std::stringstream ret;
        ret << "    - sourceDevice=" << deviceId << ", sourceBufferType=" << BufferTypeToString(bufType)
            << ", sourceAddr=0x" << std::hex << srcAddr << std::dec << '\n';
        return ret.str();
    }
};

struct BufferSemantic {
    uint64_t startAddr;                  // 起始地址
    mutable uint64_t size;               // 大小
    mutable bool isReduce;               // 是否做了reduce操作
    mutable HcclReduceOp reduceType;     // reduce操作的类型
    mutable std::set<SrcBufDes> srcBufs; // 这块数据来自哪个或哪些rank
    std::vector<uint32_t> affectedGlobalSteps; // 表示这个语义块被哪个、哪些节点影响了，用于图形化界面展示
    // V3 AR/RS 有序归约树 ID（可选）。0 表示未启用或未知；ID 仅属于某一次校验的 arena。
    // 叶子记录全局设备号；来源坐标仍在 srcBufs 中；切片继承树 ID。
    uint32_t reduceOrderExpr{0};

    BufferSemantic(
        uint64_t startAddr, uint64_t size, bool isReduce = false,
        HcclReduceOp reduceType = HcclReduceOp::HCCL_REDUCE_RESERVED)
        : startAddr(startAddr),
          size(size),
          isReduce(isReduce),
          reduceType(reduceType)
    {}

    BufferSemantic(
        uint64_t startAddr, uint64_t size, bool isReduce, HcclReduceOp reduceType, std::set<SrcBufDes> srcBufs)
        : startAddr(startAddr),
          size(size),
          isReduce(isReduce),
          reduceType(reduceType),
          srcBufs(srcBufs)
    {}

    inline bool operator<(const BufferSemantic& another) const { return startAddr < another.startAddr; }

    std::string Describe() const
    {
        std::stringstream ret;
        ret << "  range=[0x" << std::hex << startAddr << ",0x" << (startAddr + size) << ")"
            << ", size=0x" << size << std::dec;
        if (isReduce) {
            ret << ", reduce=" << ReduceOpToString(reduceType);
        }
        ret << ", sourceCount=" << srcBufs.size() << '\n';
        ret << "  sources:\n";
        for (const auto& ele : srcBufs) {
            ret << ele.Describe();
        }
        return ret.str();
    }
};

using BufferSemanticMap = std::map<uint64_t, BufferSemantic>;
// 集合通信语义按设备维护一张绝对地址表，缓冲类型仅保留在 SrcBufDes 中描述数据来源。
using PhysicalMemorySemantics = std::map<DeviceId, BufferSemanticMap>;

// 每个设备的缓冲区布局快照：各逻辑视图（INPUT/OUTPUT/CCL）在绝对地址空间中的基址与注册大小。
// 基址/大小由 InitState 从 StorageManager 的内存布局加载；CCL 缓冲不作为结果窗口校验对象，
// 因此只记录基址、不记录大小。
// 字段类型与 StorageManager 的 uint64_t& 出参保持一致。
struct BufferAddressLayout {
    bool hasInput{false};
    bool hasOutput{false};
    bool hasCcl{false};
    uint64_t inputBase{0};
    uint64_t outputBase{0};
    uint64_t cclBase{0};
    // 缓冲区的注册总大小（GetBlockSize 口径）。0 表示未知（布局未注册），
    // 未知时“多余输出数据”校验会跳过，见 GetSemanticBufferExtent。
    uint64_t inputSize{0};
    uint64_t outputSize{0};
};

using DeviceBufferAddressLayouts = std::map<DeviceId, BufferAddressLayout>;

// 读取数据源缓冲区的基址（用于校验期望源地址）。
// 布局缺失时回退为 0：回退会使期望源地址计算错误，最终以 SRC/MISSING 类错误暴露；
// 该回退不允许静默发生 —— InitState 加载布局时会对缺失的结果窗口基址输出 WARN 日志。
inline uint64_t
GetSemanticBufferBase(const DeviceBufferAddressLayouts& layouts, DeviceId deviceId, BufferType bufferType)
{
    const auto layoutIt = layouts.find(deviceId);
    if (layoutIt == layouts.end()) {
        return 0;
    }
    switch (bufferType) {
        case BufferType::INPUT:
            return layoutIt->second.inputBase;
        case BufferType::OUTPUT:
            return layoutIt->second.outputBase;
        case BufferType::CCL:
            return layoutIt->second.cclBase;
        default:
            return 0;
    }
}

// 读取结果缓冲区的布局：基址 + 注册总大小，供最终校验构建“期望结果窗口”并识别多余输出。
// 布局缺失（或对应缓冲未注册）时两者都回退为 0：
// 1) 基址回退会让校验窗口整体错位，最终以“结果缺失”类错误暴露；
// 2) 大小回退（bufferSize == 0）时无法判定哪些段越过了结果缓冲区，
//    “窗口之后的多余输出”校验对该设备自动失效，保持旧的放行行为。
// 同样地，该回退不允许静默发生，InitState 会对缺失的结果窗口基址输出 WARN 日志。
inline void GetSemanticBufferExtent(
    const DeviceBufferAddressLayouts& layouts, DeviceId deviceId, BufferType bufferType, uint64_t& baseAddr,
    uint64_t& bufferSize)
{
    baseAddr = 0;
    bufferSize = 0;
    const auto layoutIt = layouts.find(deviceId);
    if (layoutIt == layouts.end()) {
        return;
    }
    switch (bufferType) {
        case BufferType::INPUT:
            baseAddr = layoutIt->second.inputBase;
            bufferSize = layoutIt->second.inputSize;
            return;
        case BufferType::OUTPUT:
            baseAddr = layoutIt->second.outputBase;
            bufferSize = layoutIt->second.outputSize;
            return;
        default:
            return;
    }
}

// 在 rankToDevice 中查找 deviceId 对应的 rank 下标；找不到返回 INVALID_RANK_ID。
inline RankId FindRankIndexByDeviceId(const std::vector<DeviceId>& rankToDevice, DeviceId deviceId)
{
    for (size_t rank = 0; rank < rankToDevice.size(); ++rank) {
        if (rankToDevice[rank] == deviceId) {
            return static_cast<RankId>(rank);
        }
    }
    return TaskGraphGeneratorV3::INVALID_RANK_ID;
}

// SrcBufDes 会放进 std::set 里，地址平移时必须重建集合，不能原地修改 key。
std::set<SrcBufDes> ShiftSourceAddresses(const std::set<SrcBufDes>& srcBufs, uint64_t addressDelta);

// 提取语义段与注册结果缓冲区的交集，并同步平移来源地址；不相交时返回 false。
// bufferSize 为 0 表示大小未知，仅裁剪左边界；期望结果大小仍由各算子独立校验。
bool ClipSemanticToBuffer(BufferSemantic& semantic, uint64_t baseAddr, uint64_t bufferSize);

inline std::string DeviceIdListToString(const std::vector<DeviceId>& devices)
{
    std::stringstream ret;
    ret << "[";
    for (size_t i = 0; i < devices.size(); ++i) {
        if (i != 0) {
            ret << ",";
        }
        ret << devices[i];
    }
    ret << "]";
    return ret.str();
}

inline std::string SrcBufDeviceListToString(const std::set<SrcBufDes>& srcBufs)
{
    std::vector<DeviceId> devices;
    devices.reserve(srcBufs.size());
    for (const auto& srcBuf : srcBufs) {
        devices.push_back(srcBuf.deviceId);
    }
    return DeviceIdListToString(devices);
}

// 与 device/rank 顺序无关：srcBufs 的来源设备集合是否恰好等于 rankToDevice 覆盖的设备集合。
inline bool CoversAllRankDevices(const std::set<SrcBufDes>& srcBufs, const std::vector<DeviceId>& rankToDevice)
{
    const std::set<DeviceId> expectedDevices(rankToDevice.begin(), rankToDevice.end());
    if (srcBufs.size() != expectedDevices.size()) {
        return false;
    }
    for (const auto& srcBuf : srcBufs) {
        if (expectedDevices.find(srcBuf.deviceId) == expectedDevices.end()) {
            return false;
        }
    }
    return true;
}

void CalcInputOutputSize(
    HcclCMDType opType, uint32_t rankSize, uint64_t count, HcclDataType dataType, uint64_t& inputSize,
    uint64_t& outputSize, RankId myRank, RankId srcRank = 0, RankId dstRank = 0,
    VDataDesTagInner vDataDes = VDataDesTagInner{}, All2AllDataDesTagInner all2AllDataDes = All2AllDataDesTagInner{});
void CalcDataSize(HcclCMDType opType, uint64_t count, HcclDataType dataType, uint64_t& dataSize);
bool IsAllToAllSeries(HcclCMDType opType);
} // namespace HcclSim

#endif
