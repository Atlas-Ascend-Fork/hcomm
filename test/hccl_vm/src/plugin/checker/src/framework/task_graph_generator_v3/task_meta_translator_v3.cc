/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_meta_translator_v3.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <sstream>
#include <utility>

#include "data_slice.h"
#include "sim_log.h"
#include "utils/error_codes.h"

namespace HcclSim {
namespace TaskGraphGeneratorV3 {
    namespace {
        ProtocolType ConvertProtocol(uint8_t commProtocol)
        {
            if (commProtocol == static_cast<uint8_t>(CommProtocol::COMM_PROTOCOL_ROCE)) {
                return ProtocolType::RDMA;
            }
            return ProtocolType::SDMA;
        }

        MemType ConvertMemType(BufferType inputType)
        {
            switch (inputType) {
                case ::INPUT:
                    return MemType::INPUT;
                case ::OUTPUT:
                    return MemType::OUTPUT;
                case ::CCL:
                case ::SCRATCH:
                    return MemType::CCL;
                case ::MS:
                    return MemType::MS_CCU;
                case ::INPUT_AIV:
                case ::OUTPUT_AIV:
                case ::AIV_COMMINFO:
                case ::USERBUF_AIV:
                    return MemType::AIV_UB;
                default:
                    return MemType::INVALID;
            }
        }

        MemSlice MakeMemSlice(DeviceId deviceId, RankId rankId, const DataSlice& slice)
        {
            MemSlice memSlice;
            memSlice.deviceId = deviceId;
            memSlice.rankId = rankId;
            memSlice.memType = ConvertMemType(slice.GetType());
            memSlice.offset = slice.GetOffset();
            memSlice.len = slice.GetSize();
            memSlice.rawAddr = slice.GetRawAddr();
            return memSlice;
        }

        bool ResolveTaskIdentity(
            const HcclTaskMetaData& taskMeta, StorageManager& storage, DeviceId& deviceId, RankId& rankId)
        {
            if (taskMeta.deviceId > std::numeric_limits<DeviceId>::max()) {
                return false;
            }
            deviceId = static_cast<DeviceId>(taskMeta.deviceId);

            if (taskMeta.rankId == INVALID_RANK_ID) {
                const auto& mappings = storage.GetDeviceRankMappings();
                const auto iter = mappings.find(deviceId);
                if (iter == mappings.end()) {
                    return false;
                }
                rankId = iter->second;
                return true;
            }

            DeviceId mappedDeviceId = INVALID_DEVICE_ID;
            if (!storage.GetDeviceIdByCommRank(taskMeta.commId, taskMeta.rankId, mappedDeviceId)
                || mappedDeviceId != deviceId) {
                return false;
            }
            rankId = taskMeta.rankId;
            return true;
        }

        HcclResult MakeTaskPosition(
            const HcclTaskMetaData& taskMeta, StorageManager& storage, OperatorId operatorId, TaskPosition& position)
        {
            if (taskMeta.streamId > std::numeric_limits<StreamId>::max()) {
                return HCCL_E_PARA;
            }

            position.operatorId = operatorId;
            position.commId = taskMeta.commId;
            if (!ResolveTaskIdentity(taskMeta, storage, position.deviceId, position.rankId)) {
                return HCCL_E_PARA;
            }
            position.streamId = static_cast<StreamId>(taskMeta.streamId);
            if (!storage.GetMainStreamId(position.deviceId, position.mainStreamId)) {
                HCCL_VM_ERROR(
                    "Cannot resolve operator main stream, deviceId={}, streamId={}", position.deviceId,
                    position.streamId);
                return HCCL_E_PARA;
            }
            return HCCL_SUCCESS;
        }

        HcclResult GetNotifyId(uint64_t notifyId, uint64_t& out)
        {
            // 伪 notifyId（CCU 事件 0xE000... 段 / 标签哈希）合法占用 64 位隔离空间，直接透传；
            // 原先的 uint32 上限校验会把使用 Event/LocalNotify 的 CCU 任务整个图生成打断。
            out = notifyId;
            return HCCL_SUCCESS;
        }

        HcclResult EnsureStream(AllRankNodeQueues& taskQueues, DeviceId deviceId, StreamId streamId)
        {
            if (streamId == INVALID_STREAM_ID) {
                return HCCL_E_PARA;
            }

            auto& rankStreams = taskQueues[deviceId];
            if (rankStreams.size() <= streamId) {
                rankStreams.resize(static_cast<size_t>(streamId) + 1);
            }
            return HCCL_SUCCESS;
        }

        CcuSqeParam MakeCcuSqeParam(const CcuTask& ccuTask)
        {
            CcuSqeParam param;
            param.dieId = ccuTask.dieId;
            param.missionId = ccuTask.missionId;
            param.timeout = ccuTask.timeout;
            param.instStartId = ccuTask.instStartId;
            param.instCnt = ccuTask.instCnt;
            param.key = ccuTask.key;
            param.argSize = ccuTask.argSize;

            const uint32_t argCount = std::min<uint32_t>(ccuTask.argSize, CCU_SQE_ARGS_LEN);
            for (uint32_t i = 0; i < argCount; ++i) {
                param.args[i] = ccuTask.args[i];
            }
            return param;
        }

        std::string DescribeTaskMetaForLog(const HcclTaskMetaData& taskMeta)
        {
            std::ostringstream os;
            os << "taskType=" << static_cast<int32_t>(taskMeta.taskType) << ", rankId=" << taskMeta.rankId
               << ", streamId=" << taskMeta.streamId;
            switch (taskMeta.taskType) {
                case HccLTaskMetaType::MEM_CPY:
                    os << ", srcDeviceId=" << taskMeta.taskData.transMem.srcDeviceId
                       << ", dstDeviceId=" << taskMeta.taskData.transMem.dstDeviceId << ", src=[0x" << std::hex
                       << taskMeta.taskData.transMem.srcOffset << ",0x"
                       << (taskMeta.taskData.transMem.srcOffset + taskMeta.taskData.transMem.len) << ")"
                       << ", dst=[0x" << taskMeta.taskData.transMem.dstOffset << ",0x"
                       << (taskMeta.taskData.transMem.dstOffset + taskMeta.taskData.transMem.len) << ")" << std::dec
                       << ", protocol=" << static_cast<uint32_t>(taskMeta.taskData.transMem.protocol);
                    break;
                case HccLTaskMetaType::REDUCE:
                    // 此处的 datacount 实际为 size
                    os << ", srcDeviceId=" << taskMeta.taskData.reduce.srcDeviceId
                       << ", dstDeviceId=" << taskMeta.taskData.reduce.dstDeviceId << ", src=[0x" << std::hex
                       << taskMeta.taskData.reduce.srcOffset << ",0x"
                       << (taskMeta.taskData.reduce.srcOffset + taskMeta.taskData.reduce.dataCount) << ")"
                       << ", dst=[0x" << taskMeta.taskData.reduce.dstOffset << ",0x"
                       << (taskMeta.taskData.reduce.dstOffset + taskMeta.taskData.reduce.dataCount) << ")" << std::dec
                       << ", dataType=" << static_cast<uint32_t>(taskMeta.taskData.reduce.dataType)
                       << ", reduceOp=" << static_cast<uint32_t>(taskMeta.taskData.reduce.reduceOp);
                    break;
                case HccLTaskMetaType::NOTIFY_RECORD:
                case HccLTaskMetaType::NOTIFY_WAIT:
                    os << ", recordDeviceId=" << taskMeta.taskData.notify.srcDeviceId
                       << ", waitDeviceId=" << taskMeta.taskData.notify.dstDeviceId
                       << ", notifyId=" << taskMeta.taskData.notify.notifyId
                       << ", notifyCount=" << taskMeta.taskData.notify.notifyCount
                       << ", protocol=" << static_cast<uint32_t>(taskMeta.taskData.notify.protocol);
                    break;
                case HccLTaskMetaType::CCU_GRAPH:
                    os << ", dieId=" << taskMeta.taskData.ccu.dieId << ", missionId=" << taskMeta.taskData.ccu.missionId
                       << ", instStartId=" << taskMeta.taskData.ccu.instStartId
                       << ", instCnt=" << taskMeta.taskData.ccu.instCnt << ", argSize=" << taskMeta.taskData.ccu.argSize
                       << ", key=" << taskMeta.taskData.ccu.key;
                    break;
                case HccLTaskMetaType::AIV_GRAPH:
                    os << ", launchId=" << taskMeta.taskData.aiv.launchIdx;
                    break;
                case HccLTaskMetaType::SYNC_STREAM:
                    os << ", syncIdx=" << taskMeta.taskData.syncStreamTask.syncIdx;
                    break;
                case HccLTaskMetaType::MODEL_EXEC:
                    os << ", execSeq=" << taskMeta.taskData.modelExec.execSeq
                       << ", role=" << static_cast<uint32_t>(taskMeta.taskData.modelExec.role)
                       << ", modelId=" << taskMeta.taskData.modelExec.modelId
                       << ", peerStreamId=" << taskMeta.taskData.modelExec.peerStreamId;
                    break;
                default:
                    break;
            }
            return os.str();
        }

        bool IsContiguousCcuSqe(const TaskCcuGraph& missionNode, const CcuSqeParam& sqe)
        {
            constexpr size_t DEFAULT_CCU_QUEUE_ID = 0U;
            constexpr uint32_t LEGACY_CCU_CONTINUATION_INST_CNT = 13U;
            const CcuSubGraphDesc& desc = missionNode.GetCcuDesc();
            if (desc.ccuParams.size() <= DEFAULT_CCU_QUEUE_ID || desc.ccuParams[DEFAULT_CCU_QUEUE_ID].empty()) {
                return false;
            }

            const CcuSqeParam& lastSqe = desc.ccuParams[DEFAULT_CCU_QUEUE_ID].back();
            if (lastSqe.instCnt != LEGACY_CCU_CONTINUATION_INST_CNT) {
                return false;
            }
            if (lastSqe.instCnt > std::numeric_limits<uint32_t>::max() - lastSqe.instStartId) {
                return false;
            }

            return sqe.instStartId == lastSqe.instStartId + lastSqe.instCnt;
        }

        // CCU 缓冲(MS)地址窗口（与 level1 侧的 CCU_MS_BASE 保持一致；南向不产生该段地址）。
        // 起点取南向实测的 MS 偏移起点 0x8000000（= msId 2048 × 4096），粒度 4KB。
        constexpr uint64_t CCU_MS_WINDOW_BASE = 0x08000000ULL;
        constexpr uint64_t CCU_MS_WINDOW_CAP = 64ULL * 1024ULL * 1024ULL;

        HcclResult GetTaskDataSlice(
            StorageManager& storage, const std::string& commName, uint64_t commHash, uint32_t opIter,
            DeviceId expectedDeviceId, uint64_t addr, uint64_t size, DataSlice& dataSlice)
        {
            // CCU 缓冲(MS)端点识别（仅 level1 的 taskmeta 会出现该段地址；南向 CCU 指令路径的 MS 切片
            // 由 GenSliceFromMs 直接构造、不经此处，故天然不触发）。
            // 取值与南向/MS 口径一致：offset = addr - BASE 即 msId*4096；stride/粒度 4KB。
            if (addr >= CCU_MS_WINDOW_BASE && addr < CCU_MS_WINDOW_BASE + CCU_MS_WINDOW_CAP) {
                dataSlice.SetBufferType(::MS);
                dataSlice.SetOffset(addr - CCU_MS_WINDOW_BASE);
                dataSlice.SetSize(size);
                dataSlice.SetRawAddr(addr);
                return HCCL_SUCCESS; // MS 为 CCU 内部空间，不参与 device 归属校验
            }

            DeviceId actualDeviceId = INVALID_DEVICE_ID;
            // 翻译热路径：commName/commHash/opIter 即当前算子（BeginOpGroup 设定），
            // 走缓存了 layout 指针的 O(1) 路径，避免每次构造 pair key 的 string 拷贝。
            HcclResult ret = storage.GetSliceForCurrentOp(addr, size, dataSlice, &actualDeviceId, expectedDeviceId);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }

            if (actualDeviceId != expectedDeviceId) {
                HCCL_VM_ERROR(
                    "{} Resolved data slice device mismatch, expectedDeviceId={}, actualDeviceId={}, addr={}, size={}",
                    MakeErrorCodeText(ErrorCode::GRAPH_ADDRESS_INVALID), expectedDeviceId, actualDeviceId, addr, size);
                return HCCL_E_MEMORY;
            }
            return HCCL_SUCCESS;
        }
    } // namespace

    void TaskMetaTranslatorV3::Reset()
    {
        nodes_.clear();
        taskQueues_.clear();
        nodeSourceIndices_.clear();
        ccuMissionNodes_.clear();
    }

    std::vector<std::unique_ptr<TaskNode>> TaskMetaTranslatorV3::TakeNodes()
    {
        std::vector<std::unique_ptr<TaskNode>> result = std::move(nodes_);
        nodes_.clear();
        return result;
    }

    AllRankNodeQueues TaskMetaTranslatorV3::TakeTaskQueues()
    {
        AllRankNodeQueues result = std::move(taskQueues_);
        taskQueues_.clear();
        return result;
    }

    HcclResult TaskMetaTranslatorV3::Translate(
        StorageManager& storage, OperatorId operatorId, const std::string& commName, uint64_t commHash, uint32_t opIter)
    {
        const std::string effectiveCommName = commName.empty() ? storage.GetCurrentCommName() : commName;
        const uint64_t effectiveCommHash = commName.empty() ? storage.GetCurrentCommHash() : commHash;
        const uint32_t effectiveOpIter = commName.empty() ? storage.GetCurrentOpIter() : opIter;
        if (effectiveCommName.empty() || effectiveCommHash == std::numeric_limits<uint64_t>::max()) {
            HCCL_VM_ERROR("Missing communicator identity for task metadata translation");
            return HCCL_E_PARA;
        }
        Reset();

        const HcclVmTaskMetaData& taskMetaData = storage.GetHvmTaskMetaData();
        const auto& taskMetaVec = taskMetaData.task_meta;
        HCCL_VM_INFO("Start converting task metadata into graph nodes, taskMetaCount={}", taskMetaVec.size());
        for (uint32_t i = 0; i < taskMetaVec.size(); ++i) {
            NodeId nodeId = INVALID_NODE_ID;
            const HcclResult ret = TranslateOneTaskMeta(
                taskMetaVec[i], storage, i, operatorId, effectiveCommName, effectiveCommHash, effectiveOpIter, nodeId);
            if (ret != HCCL_SUCCESS) {
                HCCL_VM_ERROR(
                    "{} Failed to convert one task into a graph node, taskIndex={}, "
                    "ret={}, taskMeta={}",
                    MakeErrorCodeText(ErrorCode::GRAPH_TRANSLATE_FAILED), i, static_cast<uint32_t>(ret),
                    DescribeTaskMetaForLog(taskMetaVec[i]));
                return ret;
            }
            nodeSourceIndices_.resize(nodes_.size(), i);
        }

        size_t streamCount = 0;
        for (const auto& rankEntry : taskQueues_) {
            for (const auto& stream : rankEntry.second) {
                if (!stream.empty()) {
                    ++streamCount;
                }
            }
        }
        // 1024 卡场景节点数可达数百万，逐 task 打印(node->Describe() + INFO)开销极大，暂时注释；需要时放开。
        // for (const auto &node : nodes_) {
        //     HCCL_VM_INFO("Translated task node, nodeId={}, task={}",
        //         node == nullptr ? INVALID_NODE_ID : node->GetNodeId(),
        //         node == nullptr ? "null" : node->Describe());
        // }
        HCCL_VM_INFO(
            "Finished converting task metadata into graph nodes, taskMetaCount={}, nodeCount={}, "
            "rankCount={}, nonEmptyStreamCount={}",
            taskMetaVec.size(), nodes_.size(), taskQueues_.size(), streamCount);
        return HCCL_SUCCESS;
    }

    HcclResult
    TaskMetaTranslatorV3::AddTaskNode(const TaskPosition& position, std::unique_ptr<TaskNode> node, NodeId& nodeId)
    {
        if (node == nullptr) {
            return HCCL_E_MEMORY;
        }
        if (nodes_.size() >= MAX_NODE_COUNT) {
            return HCCL_E_MEMORY;
        }

        HcclResult ret = EnsureStream(taskQueues_, position.deviceId, position.streamId);
        if (ret != HCCL_SUCCESS) {
            return ret;
        }

        nodeId = static_cast<NodeId>(nodes_.size());
        node->SetNodeId(nodeId);
        node->SetPosition(position);
        nodes_.emplace_back(std::move(node));
        taskQueues_[position.deviceId][position.streamId].push_back(nodeId);
        return HCCL_SUCCESS;
    }

    HcclResult TaskMetaTranslatorV3::TranslateOneTaskMeta(
        const HcclTaskMetaData& taskMeta, StorageManager& storage, uint32_t taskIndex, OperatorId operatorId,
        const std::string& commName, uint64_t commHash, uint32_t opIter, NodeId& nodeId)
    {
        // SYNC_STREAM has no communicator, but it still occupies its host stream and
        // must therefore participate in stream-order edges in the task graph.
        if (taskMeta.taskType == HccLTaskMetaType::SYNC_STREAM) {
            // The proxy records this host task without a communicator. Resolve its
            // rank using the communicator of the enclosing operator instead.
            HcclTaskMetaData positionTaskMeta = taskMeta;
            positionTaskMeta.commId = storage.GetCheckerParam(operatorId).commId;
            TaskPosition position;
            HcclResult ret = MakeTaskPosition(positionTaskMeta, storage, operatorId, position);
            if (ret != HCCL_SUCCESS) {
                HCCL_VM_ERROR(
                    "Failed to resolve SYNC_STREAM task position, taskIndex={}, deviceId={}, streamId={}", taskIndex,
                    taskMeta.deviceId, taskMeta.streamId);
                return ret;
            }
            position.opIter = opIter;
            auto node = std::make_unique<TaskSyncStream>(taskMeta.taskData.syncStreamTask.syncIdx);
            return AddTaskNode(position, std::move(node), nodeId);
        }

        // MODEL_EXEC 边界任务同样无 communicator，但须参与所在流顺序边 + 跨流 model-exec 依赖边。
        if (taskMeta.taskType == HccLTaskMetaType::MODEL_EXEC) {
            HcclTaskMetaData positionTaskMeta = taskMeta;
            positionTaskMeta.commId = storage.GetCheckerParam(operatorId).commId;
            TaskPosition position;
            HcclResult ret = MakeTaskPosition(positionTaskMeta, storage, operatorId, position);
            if (ret != HCCL_SUCCESS) {
                HCCL_VM_ERROR(
                    "Failed to resolve MODEL_EXEC task position, taskIndex={}, deviceId={}, streamId={}", taskIndex,
                    taskMeta.deviceId, taskMeta.streamId);
                return ret;
            }
            position.opIter = opIter;
            const auto& modelExec = taskMeta.taskData.modelExec;
            TaskType nodeType = TaskType::MODEL_EXEC_END;
            switch (modelExec.role) {
                case 0:
                    nodeType = TaskType::MODEL_EXEC_START;
                    break;
                case 1:
                    nodeType = TaskType::MODEL_EXEC_START_SUB;
                    break;
                case 2:
                    nodeType = TaskType::MODEL_EXEC_END_SUB;
                    break;
                case 3:
                    nodeType = TaskType::MODEL_EXEC_END;
                    break;
                default:
                    HCCL_VM_WARN("Unknown MODEL_EXEC role={}, taskIndex={}", modelExec.role, taskIndex);
                    nodeType = TaskType::MODEL_EXEC_END;
                    break;
            }
            auto node = std::make_unique<TaskModelExec>(nodeType, modelExec.execSeq, modelExec.peerStreamId);
            return AddTaskNode(position, std::move(node), nodeId);
        }

        if (taskMeta.commId == 0) {
            HCCL_VM_ERROR(
                "reject non-SYNC_STREAM task without a communicator, taskIndex={}, taskType={}, "
                "deviceId={}, streamId={}",
                taskIndex, static_cast<uint32_t>(taskMeta.taskType), taskMeta.deviceId, taskMeta.streamId);
            return HCCL_E_PARA;
        }

        TaskPosition position;
        HcclResult ret = MakeTaskPosition(taskMeta, storage, operatorId, position);
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
        position.commHash = commHash;
        position.opIter = opIter;

        switch (taskMeta.taskType) {
            case HccLTaskMetaType::MEM_CPY: {
                const auto& transMem = taskMeta.taskData.transMem;
                DataSlice srcSlice;
                DeviceId srcDeviceId = INVALID_DEVICE_ID;
                srcDeviceId = transMem.srcDeviceId;
                ret = GetTaskDataSlice(
                    storage, commName, commHash, opIter, srcDeviceId, transMem.srcOffset, transMem.len, srcSlice);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                DataSlice dstSlice;
                DeviceId dstDeviceId = INVALID_DEVICE_ID;
                dstDeviceId = transMem.dstDeviceId;
                ret = GetTaskDataSlice(
                    storage, commName, commHash, opIter, dstDeviceId, transMem.dstOffset, transMem.len, dstSlice);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }

                const ProtocolType protocol
                    = (srcDeviceId == dstDeviceId) ? ProtocolType::SDMA : ConvertProtocol(transMem.protocol);
                auto node = std::make_unique<TaskTransMem>(
                    MakeMemSlice(srcDeviceId, position.rankId, srcSlice),
                    MakeMemSlice(dstDeviceId, position.rankId, dstSlice), protocol);
                return AddTaskNode(position, std::move(node), nodeId);
            }
            case HccLTaskMetaType::REDUCE: {
                const auto& reduce = taskMeta.taskData.reduce;
                // 此处的 datacount 实际为 size
                DataSlice srcSlice;
                DeviceId srcDeviceId = INVALID_DEVICE_ID;
                srcDeviceId = reduce.srcDeviceId;
                ret = GetTaskDataSlice(
                    storage, commName, commHash, opIter, srcDeviceId, reduce.srcOffset, reduce.dataCount, srcSlice);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                DataSlice dstSlice;
                DeviceId dstDeviceId = INVALID_DEVICE_ID;
                dstDeviceId = reduce.dstDeviceId;
                ret = GetTaskDataSlice(
                    storage, commName, commHash, opIter, dstDeviceId, reduce.dstOffset, reduce.dataCount, dstSlice);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                // ReduceTask 与 TransMemTask 共用联合体，此分支必须读取 reduce.protocol。
                const ProtocolType protocol
                    = (srcDeviceId == dstDeviceId) ? ProtocolType::SDMA : ConvertProtocol(reduce.protocol);
                auto node = std::make_unique<TaskReduce>(
                    MakeMemSlice(srcDeviceId, position.rankId, srcSlice),
                    MakeMemSlice(dstDeviceId, position.rankId, dstSlice), reduce.dataType, reduce.reduceOp, protocol);
                return AddTaskNode(position, std::move(node), nodeId);
            }
            case HccLTaskMetaType::NOTIFY_RECORD: {
                AicpuNotify notify;
                notify.recordDeviceId = position.deviceId;
                notify.waitDeviceId = taskMeta.taskData.notify.dstDeviceId;
                ret = GetNotifyId(taskMeta.taskData.notify.notifyId, notify.notifyId);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }

                const ProtocolType protocol = (notify.recordDeviceId == notify.waitDeviceId) ?
                                                  ProtocolType::INVALID :
                                                  ConvertProtocol(taskMeta.taskData.notify.protocol);
                auto node = std::make_unique<TaskRecordAICPU>(notify, protocol);
                return AddTaskNode(position, std::move(node), nodeId);
            }
            case HccLTaskMetaType::NOTIFY_WAIT: {
                AicpuNotify notify;
                notify.recordDeviceId = taskMeta.taskData.notify.srcDeviceId;
                notify.waitDeviceId = position.deviceId;
                ret = GetNotifyId(taskMeta.taskData.notify.notifyId, notify.notifyId);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }

                const ProtocolType protocol = (notify.recordDeviceId == notify.waitDeviceId) ?
                                                  ProtocolType::INVALID :
                                                  ConvertProtocol(taskMeta.taskData.notify.protocol);
                auto node = std::make_unique<TaskWaitAICPU>(notify, protocol);
                return AddTaskNode(position, std::move(node), nodeId);
            }
            case HccLTaskMetaType::CCU_GRAPH: {
                if (taskMeta.taskData.ccu.argSize > CCU_SQE_ARGS_LEN) {
                    return HCCL_E_PARA;
                }

                const CcuSqeParam sqe = MakeCcuSqeParam(taskMeta.taskData.ccu);
                CcuMissionKey key;
                key.deviceId = position.deviceId;
                key.dieId = sqe.dieId;
                key.missionId = sqe.missionId;
                auto missionIter = ccuMissionNodes_.find(key);
                if (missionIter != ccuMissionNodes_.end()) {
                    const NodeId missionNodeId = missionIter->second;
                    if (missionNodeId < 0 || static_cast<size_t>(missionNodeId) >= nodes_.size()) {
                        return HCCL_E_INTERNAL;
                    }
                    auto* missionNode = dynamic_cast<TaskCcuGraph*>(nodes_[static_cast<size_t>(missionNodeId)].get());
                    if (missionNode == nullptr) {
                        return HCCL_E_INTERNAL;
                    }
                    if (IsContiguousCcuSqe(*missionNode, sqe)) {
                        missionNode->AddCcuParam(0, sqe);
                        nodeId = missionNodeId;
                        return HCCL_SUCCESS;
                    }
                }

                CcuSubGraphDesc desc;
                desc.commId = taskMeta.commId;
                desc.deviceId = position.deviceId;
                desc.ccuParams.resize(1);
                desc.ccuParams[0].push_back(sqe);

                auto node = std::make_unique<TaskCcuGraph>(std::move(desc));
                ret = AddTaskNode(position, std::move(node), nodeId);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                ccuMissionNodes_[key] = nodeId;
                return HCCL_SUCCESS;
            }
            case HccLTaskMetaType::AIV_GRAPH: {
                position.launchIdx = taskMeta.taskData.aiv.launchIdx;
                auto node = std::make_unique<TaskAivGraph>(
                    position.deviceId, taskMeta.taskData.aiv.launchIdx, taskMeta.streamId);
                return AddTaskNode(position, std::move(node), nodeId);
            }
            default:
                HCCL_VM_WARN(
                    "{} This task type is not supported for CheckerV3 graph generation, "
                    "taskIndex={}, taskMeta={}",
                    MakeErrorCodeText(ErrorCode::GRAPH_UNSUPPORTED), taskIndex, DescribeTaskMetaForLog(taskMeta));
                return HCCL_E_NOT_SUPPORT;
        }
    }
} // namespace TaskGraphGeneratorV3
} // namespace HcclSim
