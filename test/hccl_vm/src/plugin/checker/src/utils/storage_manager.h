/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STORAGE_MANAGER_H
#define STORAGE_MANAGER_H

#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <vector>

#include "binary_data_type_pub.h"
#include "data_slice.h"
#include "dtype_common.h"
#include "sim_common.h"
#include "operation_data/operation_data_types.h"
#include "task_meta_defs.h"
#include "framework/task_graph_generator_v3/task_def_v3.h"

namespace HcclSim {
struct MemBlock {
    uint64_t startAddr;    // 物理起始地址
    uint64_t size;         // 块大小
    BufferType bufferType; // 类型
    uint64_t globalOffset; // 该类型总偏移
};

// Raw V-operator parameters reported by one rank in the current op group.
struct VRankParam {
    uint64_t localCount = 0;
    std::vector<uint64_t> counts;
    std::vector<uint64_t> displs;
};

struct SendRecvPairParam {
    uint32_t srcRank = 0;
    uint32_t dstRank = 0;
    bool sendSeen = false;
    bool recvSeen = false;
};

struct BatchSendRecvRankParam {
    uint32_t itemNum = 0;
    uint64_t peerCount = 0;
    HcclDataType dataType = static_cast<HcclDataType>(0);
    uint32_t sendPeer = 0;
    uint32_t recvPeer = 0;
};

struct CheckerParam {
    HcclCMDType cmdType = static_cast<HcclCMDType>(0);
    uint32_t rankSize = 0;
    HcclDataType dataType = static_cast<HcclDataType>(0);
    uint64_t dataCount = 0;

    // 以下是可能缺失的字段，初始化为安全值
    HcclReduceOp reduceType = static_cast<HcclReduceOp>(0);
    DeviceId deviceId = INVALID_DEVICE_ID;
    CommId commId = INVALID_COMM_ID;
    uint32_t srcRank = 0;
    uint32_t dstRank = 0;
    uint32_t root = 0;
    std::vector<SendRecvPairParam> sendRecvPairs;
    std::vector<BatchSendRecvRankParam> batchSendRecvRankParams;

    VDataDesTagInner vDataDes;
    std::vector<VRankParam> vRankParams;
    All2AllDataDesTagInner all2AllDataDes;
};

struct RemoteDieInfo {
    uint32_t dstDeviceId; // 远端物理设备 ID
    uint32_t remoteDieId;
};

struct HalfRTTInfo {
    uint32_t wishCntXnBegin = 0;
    uint32_t wishCntXnEnd = 0;
    uint32_t totalCntXn = 0;
};

using ChannelsPerDie = std::map<uint32_t, RemoteDieInfo>;

class StorageManager {
public:
    StorageManager() = default;
    static StorageManager& GetInstance()
    {
        static StorageManager instance;
        return instance;
    }
    // 禁用拷贝构造和赋值
    StorageManager(const StorageManager&) = delete;
    StorageManager& operator=(const StorageManager&) = delete;

    void SetDataId(const std::string& data_id)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_data_id = data_id;
    }

    std::string GetDataId() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_data_id;
    }

    /** 取当前算子的参数（按值返回，供日志/摘要等一次性使用）。 */
    CheckerParam GetCheckerParam() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_checker_param;
    }

    /**
     * 取指定算子的参数（只读引用）。
     * 翻译期间 m_checker_params / m_checker_param 不再被写入，返回引用可避免整份 CheckerParam 拷贝
     * （含多个 vector）；调用方应在持有引用期间避免触发参数重写。
     */
    const CheckerParam& GetCheckerParam(TaskGraphGeneratorV3::OperatorId operatorId) const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        const auto it = m_checker_params.find(operatorId);
        return it == m_checker_params.end() ? m_checker_param : it->second;
    }

    void SaveCheckerParam(TaskGraphGeneratorV3::OperatorId operatorId)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_checker_params[operatorId] = m_checker_param;
    }

    HcclResult LoadHcclVmSynthesisData(
        DeviceId deviceId, CommId commId, const std::string& commName, uint64_t commHash, uint32_t opIter,
        uint32_t rankId, sim::operation::OpMemInfoTab memInfo, std::vector<sim::operation::CcuChannelTab>& channels,
        std::vector<sim::operation::HalfRTTTab>& halfRTT);
    HcclResult LoadHcclVmInstrData(std::vector<sim::operation::CcuInstrResTab>& instrRes);
    HcclResult LoadHcclVmTaskMetaData(std::vector<std::vector<sim::operation::OpTaskTab>>& allTasks);
    HcclResult LoadDecodedHcclVmTaskMetaData(const std::vector<std::vector<HcclTaskMetaData>>& allTaskMetas);
    void Reset(bool clearMemLayout = true);
    uint64_t GetBlockSize(
        const std::string& commName, uint64_t commHash, uint32_t opIter, DeviceId deviceId, BufferType bufferType);
    bool GetBufferBaseAddress(
        const std::string& commName, uint64_t commHash, uint32_t opIter, DeviceId deviceId, BufferType bufferType,
        uint64_t& baseAddr) const;
    bool IsMemRangeRegistered(
        const std::string& commName, uint64_t commHash, uint32_t opIter, DeviceId deviceId, uint64_t addr,
        uint64_t len) const;
    /**
     * 解析绝对地址所属的内存块并填充 DataSlice。
     * @param commName/commHash/opIter 定位内存布局。
     * @param addr/len 待解析的地址与长度。
     * @param dataSlice 输出：命中块的 buffer 类型/原始地址/偏移。
     * @param deviceId 可选输出：命中所在设备 id。
     * @param expectedDeviceId 若有效则只在该设备上查找（O(1) 路径，用于翻译热路径）；
     *                         为 INVALID_DEVICE_ID 时退回全设备扫描（保留 CCU 展开等旧行为）。
     * @return 命中返回 HCCL_SUCCESS，未命中返回 HCCL_E_MEMORY。
     */
    HcclResult GetSlice(
        const std::string& commName, uint64_t commHash, uint32_t opIter, uint64_t addr, uint64_t len,
        DataSlice& dataSlice, DeviceId* deviceId = nullptr, DeviceId expectedDeviceId = INVALID_DEVICE_ID);
    /**
     * 取“当前算子”的内存切片（翻译热路径专用）。
     * 使用 FinalizeOpGroup 时解析好的 layout 指针，避免每次构造 pair key 的 string 拷贝；
     * 缓存未就绪时回退到通用 GetSlice。
     * @param addr/len 待解析的地址与长度。
     * @param dataSlice 输出：命中块的 buffer 类型/原始地址/偏移。
     * @param deviceId 可选输出：命中所在设备 id。
     * @param expectedDeviceId 有效时只查该设备（O(1)）。
     * @return 命中返回 HCCL_SUCCESS，未命中返回 HCCL_E_MEMORY。
     */
    HcclResult GetSliceForCurrentOp(
        uint64_t addr, uint64_t len, DataSlice& dataSlice, DeviceId* deviceId = nullptr,
        DeviceId expectedDeviceId = INVALID_DEVICE_ID);
    /** 返回当前通信域 device→rank 映射的只读引用（避免每 task 拷贝整张 map）。 */
    const std::map<DeviceId, RankId>& GetDeviceRankMappings() const;
    /** 返回指定通信域成员 device→rank 映射的只读引用（未命中返回空表引用）。 */
    const std::map<DeviceId, RankId>& GetDeviceRankMappings(CommId commId) const;
    /** 通过预构建的反向表由 commRank 反查 deviceId（O(log R)）。 */
    bool GetDeviceIdByCommRank(CommId commId, RankId rankId, DeviceId& deviceId) const;
    bool GetMainStreamId(DeviceId deviceId, TaskGraphGeneratorV3::StreamId& streamId) const;
    uint32_t GetRankSize() const;

    HcclResult ReadHeader(FILE* fp, FileHeader& header);
    HcclResult ChannelWrite(FILE* fp, const ChannelInfo& chInfo);
    HcclResult ChannelRead(FILE* fp, ChannelInfo& chInfo);

    HcclVmInstrData GetHvmInstrData() const;
    HcclResult Trans2CheckerParam(sim::operation::OpDetailTab& detailTab, ::OpDetails& detail);
    /**
     * 取当前已解码的 V3 task 元数据。
     * 返回只读引用以避免每次调用都拷贝一整份（1024 卡场景可达数百 MB）；
     * 调用方不得在持有该引用期间触发会重置 task 元数据的操作。
     */
    const HcclVmTaskMetaData& GetHvmTaskMetaData() const;
    void InitCcuInfo(DevType& devType, std::vector<uint64_t>& resourceBaseAddr);
    void BeginOpGroup(const std::string& commName, uint64_t commHash, uint32_t opIter);
    std::string GetCurrentCommName() const;
    uint64_t GetCurrentCommHash() const;
    uint32_t GetCurrentOpIter() const;
    /** 由 commId 取通信域名（加载阶段缓存、检查阶段只读；未命中返回空串引用）。 */
    const std::string& GetCommNameById(CommId commId) const;
    HcclResult FinalizeOpGroup();
    void MergeAll2AllVSendCountMatrix();

private:
    using DeviceMemLayout = std::map<DeviceId, std::map<BufferType, std::map<uint64_t, MemBlock>>>;

    std::string FindRootPath();
    bool IsDirExists(const std::string& path);

    /**
     * 在给定内存布局中解析地址所属内存块。
     * @param layout 待查找的完整布局。
     * @param addr 绝对地址。
     * @param dataSlice 命中时填充。
     * @param deviceId 可选输出：命中所在设备 id。
     * @param expectedDeviceId 有效时只查该设备（O(1)）；为 INVALID_DEVICE_ID 时全设备扫描。
     * @return 命中返回 HCCL_SUCCESS，否则 HCCL_E_MEMORY。
     */
    HcclResult ResolveSliceInLayout(
        const DeviceMemLayout& layout, uint64_t addr, DataSlice& dataSlice, DeviceId* deviceId,
        DeviceId expectedDeviceId) const;

    /**
     * 解析“当前算子”的 op/CCL layout 指针（每算子一次），供翻译热路径免去 pair key 构造与查找。
     * 需在内存布局加载完成后（FinalizeOpGroup）调用；Reset 会将其置空。
     */
    void ResolveCurrentOpLayouts();

    /**
     * 构建通信域 device→rank 映射与 commId→commName 缓存；同一通信域只构建一次，各成员 id 共享。
     * @param commId 通信域内任一成员 id。
     * @param commName 该通信域的通信域名（用于按成员 id 缓存名字）。
     * @return 成功返回 HCCL_SUCCESS；成员解析失败返回 HCCL_E_PARA。
     */
    HcclResult EnsureCommunicatorMappings(CommId commId, const std::string& commName);

    /**
     * 转换并缓存全局 channel 信息；每次 Reset 后只处理一次。
     * @param channels 本轮通信域的全部 channel 记录。
     */
    void EnsureChannelInfo(const std::vector<sim::operation::CcuChannelTab>& channels);

    /**
     * 转换并缓存全局 half RTT 信息；每次 Reset 后只处理一次。
     * @param halfRTT 本轮通信域的全部 half RTT 记录。
     */
    void EnsureHalfRttInfo(const std::vector<sim::operation::HalfRTTTab>& halfRTT);

    /**
     * 写入单个 rank 的 input/output/CCL 内存布局（单 rank 级数据，需逐 rank 调用）。
     * @param deviceId/rankId 物理设备 id 与逻辑 rank。
     * @param commName/commHash/opIter 定位算子维度的内存布局。
     * @param memInfo 该 rank 的 buffer 地址/大小。
     */
    void LoadRankMemLayout(
        DeviceId deviceId, uint32_t rankId, const std::string& commName, uint64_t commHash, uint32_t opIter,
        const sim::operation::OpMemInfoTab& memInfo);

    std::string m_data_id;
    mutable std::mutex m_mutex;

    // 算子 buffer 按通信域和 opIter 隔离保存。
    using CommIdentity = std::pair<std::string, uint64_t>;
    std::map<CommIdentity, std::map<uint32_t, DeviceMemLayout>> m_op_mem_layout;
    // CCL buffer 属于通信域，在多个 opIter 之间持续使用。
    std::map<CommIdentity, DeviceMemLayout> m_comm_ccl_layout;
    std::string m_current_comm_name;
    uint64_t m_current_comm_hash{std::numeric_limits<uint64_t>::max()};
    uint32_t m_current_op_iter{0};
    // 当前算子的 layout 指针（FinalizeOpGroup 时解析，Reset 时置空）；指向单例内 map 节点，地址稳定。
    const DeviceMemLayout* m_currentOpLayout{nullptr};
    const DeviceMemLayout* m_currentCclLayout{nullptr};
    std::map<CommId, std::map<DeviceId, RankId>> m_comm_device_rank_mappings;
    // rank→device 反向表，避免 GetDeviceIdByCommRank 每次线性扫描。
    std::map<CommId, std::map<RankId, DeviceId>> m_comm_rank_device_mappings;
    // commId→通信域名缓存（加载阶段填充，检查阶段只读）；节点不再各存 commName。
    std::map<CommId, std::string> m_comm_id_to_name;
    std::map<DeviceId, TaskGraphGeneratorV3::StreamId> m_main_stream_ids;
    std::map<DeviceId, std::map<uint32_t, ChannelsPerDie>> m_allRankChannelInfo;
    CheckerParam m_checker_param;
    std::map<TaskGraphGeneratorV3::OperatorId, CheckerParam> m_checker_params;
    // 用于收集每一轮算子中所有rank的发送矩阵数据
    std::map<uint32_t, std::vector<uint64_t>> m_all2AllvSendMatrices;

    HcclVmSynData m_synData;
    HcclVmInstrData m_instrData;
    HcclVmTaskMetaData m_taskMeataData;
    DevType devType_{DevType::DEV_TYPE_COUNT}; // 初始化无效值
    // 通信域级数据（channel / half RTT）在本轮是否已转换缓存，避免按 rank 重复处理。
    bool m_channelInfoLoaded{false};
    bool m_halfRttInfoLoaded{false};
};
} // namespace HcclSim

#endif
