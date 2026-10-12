/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * Description: ccu instruction transform to checker task
 * Author: huangweihao
 * Create: 2025-06-19
 */

#include "ccu_task_common_v3.h"

#include <deque>
#include <memory>
#include <sstream>
#include <utility>

#include "ccu_all_rank_param_recorder_v3.h"
#include "sim_log.h"
#include "storage_manager.h"
#include "type_conversion.h"
#include "utils/error_codes.h"

namespace HcclSim {
namespace TaskGraphGeneratorV3 {
    namespace {
        const char* RoleName(CcuNodeRoleV3 role)
        {
            switch (role) {
                case CcuNodeRoleV3::HEAD:
                    return "ccu_head";
                case CcuNodeRoleV3::LOCAL_COPY:
                    return "local_copy";
                case CcuNodeRoleV3::LOCAL_REDUCE:
                    return "local_reduce";
                case CcuNodeRoleV3::LOCAL_BATCH_REDUCE:
                    return "local_batch_reduce";
                case CcuNodeRoleV3::READ:
                    return "read";
                case CcuNodeRoleV3::READ_REDUCE:
                    return "read_reduce";
                case CcuNodeRoleV3::WRITE:
                    return "write";
                case CcuNodeRoleV3::WRITE_REDUCE:
                    return "write_reduce";
                case CcuNodeRoleV3::POST:
                    return "post";
                case CcuNodeRoleV3::WAIT:
                    return "wait";
                case CcuNodeRoleV3::LOCAL_POST_TO:
                    return "local_post_to";
                case CcuNodeRoleV3::LOCAL_WAIT_FROM:
                    return "local_wait_from";
                case CcuNodeRoleV3::LOOP_START:
                    return "loop_start";
                case CcuNodeRoleV3::LOOP_END:
                    return "loop_end";
                case CcuNodeRoleV3::SUB_GRAPH_END:
                    return "sub_graph_end";
                default:
                    return "unknown";
            }
        }

        bool IsRemoteAsyncRole(CcuNodeRoleV3 role)
        {
            return role == CcuNodeRoleV3::READ || role == CcuNodeRoleV3::READ_REDUCE || role == CcuNodeRoleV3::WRITE
                   || role == CcuNodeRoleV3::WRITE_REDUCE;
        }

        bool IsLocalAsyncRole(CcuNodeRoleV3 role)
        {
            return role == CcuNodeRoleV3::LOCAL_COPY || role == CcuNodeRoleV3::LOCAL_REDUCE
                   || role == CcuNodeRoleV3::LOCAL_BATCH_REDUCE;
        }

        void AddSameQueueChildrenForPrint(
            TaskNode* node, DeviceId deviceId, QueueId queueId, std::deque<TaskNode*>& candNodes,
            std::set<TaskNode*>& visitedNodes)
        {
            if (node == nullptr) {
                return;
            }

            for (TaskNode* child : node->GetChildren()) {
                if (child == nullptr || child->GetPosition().deviceId != deviceId
                    || child->GetLocation().queueId != queueId) {
                    continue;
                }
                if (visitedNodes.insert(child).second) {
                    candNodes.push_back(child);
                }
            }
        }

        void PrintCcuSingleQueueV3(TaskNode* head)
        {
            if (head == nullptr) {
                return;
            }

            const DeviceId deviceId = head->GetPosition().deviceId;
            const StreamId streamId = head->GetPosition().streamId;
            const QueueId queueId = head->GetLocation().queueId;
            std::deque<TaskNode*> candNodes;
            std::set<TaskNode*> visitedNodes;
            std::set<TaskNode*> printedNodes;

            HCCL_VM_INFO(
                "streamId={}, queueId={}, head[{:p}] {}", static_cast<uint32_t>(streamId),
                static_cast<uint32_t>(queueId), static_cast<void*>(head), head->Describe());
            printedNodes.insert(head);
            AddSameQueueChildrenForPrint(head, deviceId, queueId, candNodes, visitedNodes);

            while (!candNodes.empty()) {
                TaskNode* curNode = candNodes.front();
                candNodes.pop_front();
                AddSameQueueChildrenForPrint(curNode, deviceId, queueId, candNodes, visitedNodes);

                bool parentsAllPrinted = true;
                for (TaskNode* parent : curNode->GetParents()) {
                    if (parent == nullptr || parent->GetPosition().deviceId != deviceId
                        || parent->GetLocation().queueId != queueId) {
                        continue;
                    }
                    if (printedNodes.count(parent) == 0) {
                        parentsAllPrinted = false;
                        break;
                    }
                }

                if (parentsAllPrinted) {
                    HCCL_VM_INFO("[{:p}] {}", static_cast<void*>(curNode), curNode->Describe());
                    printedNodes.insert(curNode);
                } else {
                    candNodes.push_back(curNode);
                }
            }
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

        std::vector<MemSlice> MakeMemSlices(DeviceId deviceId, RankId rankId, const std::vector<DataSlice>& slices)
        {
            std::vector<MemSlice> memSlices;
            memSlices.reserve(slices.size());
            for (const auto& slice : slices) {
                memSlices.push_back(MakeMemSlice(deviceId, rankId, slice));
            }
            return memSlices;
        }

        const CcuSqeParam* FindTraceSqeParam(const CcuGraphStateV3& state, QueueId queueId)
        {
            if (queueId < state.ccuParams.size() && !state.ccuParams[queueId].empty()) {
                return &state.ccuParams[queueId].front();
            }
            for (const auto& queueParams : state.ccuParams) {
                if (!queueParams.empty()) {
                    return &queueParams.front();
                }
            }
            return nullptr;
        }

        void PopulateTraceContext(const CcuGraphStateV3& state, CcuTraceInfo& trace)
        {
            if (trace.position.queueId == INVALID_QUEUE_ID) {
                return;
            }
            const auto* sqe = FindTraceSqeParam(state, trace.position.queueId);
            if (sqe != nullptr) {
                trace.dieId = sqe->dieId;
                trace.missionId = sqe->missionId;
            }
            if (trace.position.deviceId == INVALID_DEVICE_ID || trace.dieId == INVALID_DIE_ID) {
                return;
            }

            auto* recorder = AllRankParamRecorder::Global();
            trace.registerState.xn = recorder->GetXnSnapshot(trace.position.deviceId, trace.dieId);
            trace.registerState.gsa = recorder->GetGSASnapshot(trace.position.deviceId, trace.dieId);
            trace.registerState.cke = recorder->GetCKESnapshot(trace.position.deviceId, trace.dieId);
        }

        TaskPosition MakeCcuPosition(DeviceId deviceId, uint32_t queId)
        {
            TaskPosition position;
            position.deviceId = deviceId;
            position.streamId = queId;
            position.queueId = queId;
            return position;
        }

        uint8_t ToTaskDataType(HcclDataType dataType) { return static_cast<uint8_t>(dataType); }

        uint8_t ToTaskReduceOp(HcclReduceOp reduceOp) { return static_cast<uint8_t>(reduceOp); }

        HcclResult FindInstrForQueue(
            StorageManager& storage, DeviceId deviceId, uint32_t dieId, hcomm::CcuRep::CcuInstrInfo& instrInfo)
        {
            HcclVmInstrData hvmInstrData = storage.GetHvmInstrData();
            for (const auto& instr : hvmInstrData.instr_data) {
                if (instr.desc.rank_id != deviceId || instr.desc.die_id != dieId) {
                    continue;
                }
                instrInfo.instrVec = instr.data;
                instrInfo.startInstrId = 0;
                instrInfo.instrCount = static_cast<uint16_t>(instr.data.size());
                return HCCL_SUCCESS;
            }
            HCCL_VM_ERROR(
                "{} Missing CCU instruction data for this device/die, deviceId={}, dieId={}",
                MakeErrorCodeText(ErrorCode::GRAPH_RESOURCE_NOT_FOUND), deviceId, dieId);
            return HCCL_E_INTERNAL;
        }

    } // namespace

    CcuGraphStateV3::CcuGraphStateV3(TaskGraphGeneratorV3& graphIn, TaskCcuGraph& ccuGraphIn)
        : graph(graphIn),
          ccuGraph(ccuGraphIn),
          commId(ccuGraphIn.GetCommId()),
          deviceId(ccuGraphIn.GetDeviceId()),
          storage_(graphIn.GetStorageManager())
    {}

    HcclResult CcuGraphStateV3::GetSlice(uint64_t addr, uint64_t len, DataSlice& dataSlice, DeviceId* deviceId) const
    {
        const TaskPosition& position = ccuGraph.GetPosition();
        // 位置不携带 commName：按 commId 解析通信域名；commId 缺失（合成位置未带 commId）时回退当前
        // 算子组，否则会以空通信域名查内存布局而必然失败。
        std::string commName = storage_.GetCommNameById(position.commId);
        if (commName.empty()) {
            commName = storage_.GetCurrentCommName();
        }
        return storage_.GetSlice(commName, position.commHash, position.opIter, addr, len, dataSlice, deviceId);
    }

    std::string CcuGraphStateV3::IdentityCtx() const
    {
        std::ostringstream os;
        os << "commId=" << commId << ", deviceId=" << deviceId;
        return os.str();
    }

    HcclResult CcuGraphStateV3::InitInstrInfo(StorageManager& storage)
    {
        const CcuSubGraphDesc& desc = ccuGraph.GetCcuDesc();
        ccuParams = desc.ccuParams;
        queueNum_ = static_cast<uint32_t>(ccuParams.size());
        instrInfo.resize(queueNum_);
        ccuParamIndexs.assign(queueNum_, 0);
        microCodePosInQue.assign(queueNum_, 0);
        startInstrIdInQue.assign(queueNum_, 0);
        instrQueStatus.assign(queueNum_, false);
        tailNodes.assign(queueNum_, nullptr);

        for (uint32_t queId = 0; queId < queueNum_; ++queId) {
            if (ccuParams[queId].empty()) {
                continue;
            }
            const auto& firstSqe = ccuParams[queId].front();
            HcclResult ret = FindInstrForQueue(storage, deviceId, firstSqe.dieId, instrInfo[queId]);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
            instrInfo[queId].missionStartInstrId = static_cast<uint16_t>(firstSqe.instStartId);
            instrInfo[queId].missionInstrCount
                = static_cast<uint16_t>(GetMissionEndInstrId(queId) - firstSqe.instStartId);
            startInstrIdInQue[queId] = instrInfo[queId].startInstrId;
            microCodePosInQue[queId] = firstSqe.instStartId - instrInfo[queId].startInstrId;
        }
        return HCCL_SUCCESS;
    }

    HcclResult CcuGraphStateV3::CreateHeadNode()
    {
        auto node = std::make_unique<TaskStart>(BoundaryType::CCU_SUB_GRAPH);
        TaskNode* rawNode = nullptr;
        HcclResult ret = AppendGeneratedNode(std::move(node), ccuGraph.GetPosition(), CcuNodeRoleV3::HEAD, rawNode);
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
        ccuHeadTaskNode = rawNode;
        return HCCL_SUCCESS;
    }

    HcclResult CcuGraphStateV3::AppendGeneratedNode(
        std::unique_ptr<TaskNode> node, DeviceId devId, uint32_t queId, CcuNodeRoleV3 role, TaskNode*& outNode,
        DeviceId peerDeviceId, uint16_t remainingCkeMask, uint32_t dieId, uint16_t ckeId, bool invalidPost)
    {
        TaskPosition position = MakeCcuPosition(devId, queId);
        position.streamId = ccuGraph.GetPosition().streamId;
        position.operatorId = ccuGraph.GetPosition().operatorId;
        // 位置不再逐节点存 commName，改用 commId 经 StorageManager::GetCommNameById 解析；
        // 因此这里必须把 commId 一并带下来（MakeCcuPosition 只设 deviceId/streamId/queueId）。
        position.commId = ccuGraph.GetPosition().commId;
        position.commHash = ccuGraph.GetPosition().commHash;
        position.opIter = ccuGraph.GetPosition().opIter;
        return AppendGeneratedNode(
            std::move(node), position, role, outNode, peerDeviceId, remainingCkeMask, dieId, ckeId, invalidPost);
    }

    HcclResult CcuGraphStateV3::AppendGeneratedNode(
        std::unique_ptr<TaskNode> node, const TaskPosition& position, CcuNodeRoleV3 role, TaskNode*& outNode,
        DeviceId peerDeviceId, uint16_t remainingCkeMask, uint32_t dieId, uint16_t ckeId, bool invalidPost)
    {
        if (node == nullptr) {
            return HCCL_E_MEMORY;
        }
        CcuTraceInfo trace;
        trace.valid = true;
        trace.opName = "ccu";
        trace.nodeRole = RoleName(role);
        trace.position = position;
        trace.taskLoc = position;
        trace.queueId = position.queueId;
        if (position.queueId < microCodePosInQue.size() && position.queueId < startInstrIdInQue.size()) {
            trace.instrId = microCodePosInQue[position.queueId] + startInstrIdInQue[position.queueId];
        }
        if (!loopNodeIdStack_.empty()) {
            trace.loopStartNodeId = loopNodeIdStack_.back();
        }
        PopulateTraceContext(*this, trace);
        node->SetCcuTrace(std::move(trace));

        NodeId nodeId = INVALID_NODE_ID;
        HcclResult ret = graph.AppendGeneratedNode(std::move(node), position, nodeId);
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
        outNode = graph.GetNode(nodeId);
        if (outNode == nullptr) {
            return HCCL_E_PTR;
        }
        CcuNodeMetaV3 meta;
        meta.role = role;
        meta.peerDeviceId = peerDeviceId;
        meta.remainingCkeMask = remainingCkeMask;
        meta.dieId = dieId;
        meta.ckeId = ckeId;
        meta.invalidPost = invalidPost;
        SetNodeMeta(outNode, meta);
        ++internalNodeCount;
        return HCCL_SUCCESS;
    }

    void CcuGraphStateV3::SetNodeMeta(TaskNode* node, const CcuNodeMetaV3& meta)
    {
        if (node == nullptr) {
            return;
        }
        nodeMetas_[node] = meta;
    }

    const CcuNodeMetaV3* CcuGraphStateV3::GetNodeMeta(const TaskNode* node) const
    {
        auto iter = nodeMetas_.find(node);
        if (iter == nodeMetas_.end()) {
            return nullptr;
        }
        return &iter->second;
    }

    CcuNodeRoleV3 CcuGraphStateV3::GetNodeRole(const TaskNode* node) const
    {
        const CcuNodeMetaV3* meta = GetNodeMeta(node);
        return (meta == nullptr) ? CcuNodeRoleV3::UNKNOWN : meta->role;
    }

    DeviceId CcuGraphStateV3::GetNodePeerDeviceId(const TaskNode* node) const
    {
        const CcuNodeMetaV3* meta = GetNodeMeta(node);
        return (meta == nullptr) ? INVALID_DEVICE_ID : meta->peerDeviceId;
    }

    uint16_t CcuGraphStateV3::GetNodeRemainingCkeMask(const TaskNode* node) const
    {
        const CcuNodeMetaV3* meta = GetNodeMeta(node);
        return (meta == nullptr) ? 0 : meta->remainingCkeMask;
    }

    void CcuGraphStateV3::SetNodeRemainingCkeMask(TaskNode* node, uint16_t remainingCkeMask)
    {
        auto iter = nodeMetas_.find(node);
        if (iter != nodeMetas_.end()) {
            iter->second.remainingCkeMask = remainingCkeMask;
        }
    }

    void CcuGraphStateV3::SetNodePeerDeviceId(TaskNode* node, DeviceId peerDeviceId)
    {
        auto iter = nodeMetas_.find(node);
        if (iter != nodeMetas_.end()) {
            iter->second.peerDeviceId = peerDeviceId;
        }
    }

    std::string CcuGraphStateV3::Describe() const
    {
        std::ostringstream os;
        os << "{commId=" << commId << ", deviceId=" << deviceId << ", queueCount=" << queueNum_
           << ", ccuGraphNodeId=" << ccuGraph.GetNodeId() << "}";
        return os.str();
    }

    std::string CcuGraphStateV3::DescribeNextInstructionByQueue() const
    {
        std::ostringstream os;
        os << "[";
        for (size_t queId = 0; queId < microCodePosInQue.size(); ++queId) {
            if (queId != 0) {
                os << ", ";
            }
            os << "{queueId=" << queId << ", nextInstructionOffset=" << microCodePosInQue[queId];
            if (queId < startInstrIdInQue.size()) {
                os << ", nextInstructionId=" << (startInstrIdInQue[queId] + microCodePosInQue[queId]);
            }
            os << "}";
        }
        os << "]";
        return os.str();
    }

    void CcuGraphStateV3::GetSqe(uint32_t queId, uint16_t sqeArgsId, uint64_t& argVal)
    {
        argVal = 0;
        if (queId >= ccuParams.size() || queId >= ccuParamIndexs.size() || sqeArgsId >= CCU_SQE_ARGS_LEN
            || ccuParamIndexs[queId] >= ccuParams[queId].size()) {
            return;
        }
        argVal = ccuParams[queId][ccuParamIndexs[queId]].args[sqeArgsId];
        if (sqeArgsId == CCU_SQE_ARGS_LEN - 1) {
            ccuParamIndexs[queId] += 1;
        }
    }

    void CcuGraphStateV3::GetDieId(uint32_t queId, uint32_t& dieId) const
    {
        dieId = INVALID_DIE_ID;
        if (queId >= ccuParams.size() || ccuParams[queId].empty()) {
            return;
        }
        dieId = ccuParams[queId][0].dieId;
    }

    uint32_t CcuGraphStateV3::GetMissionEndInstrId(uint32_t queId) const
    {
        if (queId >= ccuParams.size() || ccuParams[queId].empty()) {
            return 0;
        }
        const auto& lastSqe = ccuParams[queId].back();
        return lastSqe.instStartId + lastSqe.instCnt;
    }

    void PrintCcuGraph(TaskNode* dummyStart)
    {
        if (dummyStart == nullptr) {
            return;
        }

        HCCL_VM_INFO("=======================================================");
        std::set<QueueId> printedQueues;
        for (TaskNode* child : dummyStart->GetChildren()) {
            if (child == nullptr) {
                continue;
            }
            const QueueId queueId = child->GetLocation().queueId;
            if (queueId == INVALID_QUEUE_ID || !printedQueues.insert(queueId).second) {
                continue;
            }

            HCCL_VM_INFO("-------------------------------------------------------");
            HCCL_VM_INFO(
                "deviceId={}, rankId={}, streamId={}, queueId={}", child->GetPosition().deviceId,
                child->GetPosition().rankId, static_cast<uint32_t>(child->GetPosition().streamId),
                static_cast<uint32_t>(queueId));
            PrintCcuSingleQueueV3(child);
        }
        HCCL_VM_INFO("-------------------------------------------------------");
        HCCL_VM_INFO("=======================================================");
    }

    HcclResult CollectBilateralWaitInfo(CcuGraphStateV3* curCcuTask, uint32_t queId, TaskNode* node)
    {
        if (curCcuTask == nullptr || node == nullptr) {
            return HCCL_E_PTR;
        }
        if (curCcuTask->GetNodeRole(node) == CcuNodeRoleV3::WAIT) {
            DeviceId peerDeviceId = curCcuTask->GetNodePeerDeviceId(node);
            if (peerDeviceId == INVALID_DEVICE_ID || peerDeviceId >= curCcuTask->waitInfoTmp_.size()) {
                return HCCL_SUCCESS;
            }
            curCcuTask->waitInfoTmp_[peerDeviceId].waitNodes.push_back(node);
        } else {
            DeviceId peerDeviceId;
            CHK_RET(GetPeerDeviceIdByTaskNode(curCcuTask, node, peerDeviceId));
            if (peerDeviceId == INVALID_DEVICE_ID || peerDeviceId >= curCcuTask->waitInfoTmp_.size()) {
                return HCCL_SUCCESS;
            }
            auto waitSize = curCcuTask->waitInfoTmp_[peerDeviceId].waitNodes.size();
            TaskNode* waitNode = nullptr;
            if (waitSize > 0) {
                waitNode = curCcuTask->waitInfoTmp_[peerDeviceId].waitNodes[waitSize - 1];
            }
            curCcuTask->bilateralPart1_[queId].insert(std::make_pair(node, waitNode));
        }
        return HCCL_SUCCESS;
    }

    HcclResult CollectBilateralPostInfo(CcuGraphStateV3* curCcuTask, uint32_t queId, TaskNode* node, bool isLast)
    {
        if (curCcuTask == nullptr) {
            return HCCL_E_PTR;
        }
        if (isLast) {
            for (uint32_t devId = 0; devId < curCcuTask->postInfoTmp_.size(); devId++) {
                if (devId == curCcuTask->deviceId) {
                    continue;
                }
                for (auto* asynNode : curCcuTask->postInfoTmp_[devId].asyncNodes) {
                    curCcuTask->bilateralPart2_[queId].insert(std::make_pair(asynNode, nullptr));
                }
                curCcuTask->postInfoTmp_[devId].asyncNodes.clear();
            }
            return HCCL_SUCCESS;
        }

        if (node == nullptr) {
            return HCCL_E_PTR;
        }
        if (curCcuTask->GetNodeRole(node) == CcuNodeRoleV3::POST) {
            DeviceId peerDeviceId = curCcuTask->GetNodePeerDeviceId(node);
            if (peerDeviceId == INVALID_DEVICE_ID || peerDeviceId >= curCcuTask->postInfoTmp_.size()) {
                return HCCL_SUCCESS;
            }
            auto asynNodeSize = curCcuTask->postInfoTmp_[peerDeviceId].asyncNodes.size();
            if (asynNodeSize > 0) {
                for (auto* asynNode : curCcuTask->postInfoTmp_[peerDeviceId].asyncNodes) {
                    curCcuTask->bilateralPart2_[queId].insert(std::make_pair(asynNode, node));
                }
                curCcuTask->postInfoTmp_[peerDeviceId].asyncNodes.clear();
            }
        } else {
            DeviceId peerDeviceId;
            CHK_RET(GetPeerDeviceIdByTaskNode(curCcuTask, node, peerDeviceId));
            if (peerDeviceId == INVALID_DEVICE_ID || peerDeviceId >= curCcuTask->postInfoTmp_.size()) {
                return HCCL_SUCCESS;
            }
            curCcuTask->postInfoTmp_[peerDeviceId].asyncNodes.push_back(node);
        }
        return HCCL_SUCCESS;
    }

    void AddNodeRelation(TaskNode* parent, TaskNode* child)
    {
        if (parent == nullptr || child == nullptr) {
            return;
        }
        parent->AddChild(child);
        child->AddParent(parent);
    }

    HcclResult AppendTailNode(CcuGraphStateV3* curCcuTask, uint32_t queId, TaskNode* node)
    {
        if (curCcuTask == nullptr || node == nullptr || queId >= curCcuTask->tailNodes.size()) {
            return HCCL_E_PTR;
        }
        TaskNode* preNode
            = (curCcuTask->tailNodes[queId] == nullptr) ? curCcuTask->ccuHeadTaskNode : curCcuTask->tailNodes[queId];
        AddNodeRelation(preNode, node);
        curCcuTask->tailNodes[queId] = node;

        const CcuNodeRoleV3 role = curCcuTask->GetNodeRole(node);
        if (role == CcuNodeRoleV3::WAIT || IsRemoteAsyncRole(role)) {
            CHK_RET(CollectBilateralWaitInfo(curCcuTask, queId, node));
        }
        if (role == CcuNodeRoleV3::POST || IsRemoteAsyncRole(role)) {
            CHK_RET(CollectBilateralPostInfo(curCcuTask, queId, node));
        }
        if (IsRemoteAsyncRole(role) || IsLocalAsyncRole(role)) {
            curCcuTask->parallelNodes_.push_back(node);
        }
        return HCCL_SUCCESS;
    }

    MemSlice MakeCcuMemSlice(DeviceId deviceId, const DataSlice& slice)
    {
        return MakeMemSlice(deviceId, INVALID_RANK_ID, slice);
    }

    std::vector<MemSlice> MakeCcuMemSlices(DeviceId deviceId, const std::vector<DataSlice>& slices)
    {
        return MakeMemSlices(deviceId, INVALID_RANK_ID, slices);
    }

    HcclResult GetPeerDeviceIdByTaskNode(CcuGraphStateV3* curCcuTask, TaskNode* currNode, DeviceId& peerDeviceId)
    {
        if (curCcuTask == nullptr || currNode == nullptr) {
            return HCCL_E_PTR;
        }
        const CcuNodeRoleV3 role = curCcuTask->GetNodeRole(currNode);
        if (!IsRemoteAsyncRole(role)) {
            HCCL_VM_ERROR(
                "{} Failed to get peer device id from this task node because the node role does not "
                "represent a remote async operation, nodeRole={}",
                MakeErrorCodeText(ErrorCode::GRAPH_STRUCTURE_INVALID).c_str(), static_cast<uint32_t>(role));
            return HCCL_E_INTERNAL;
        }
        peerDeviceId = curCcuTask->GetNodePeerDeviceId(currNode);
        if (peerDeviceId == INVALID_DEVICE_ID) {
            return HCCL_E_INTERNAL;
        }
        return HCCL_SUCCESS;
    }

    HcclResult AddBatchTransMem(
        DeviceId deviceId, uint32_t queId, CcuGraphStateV3* curCcuTask, std::vector<MemSlice> srcs,
        std::vector<MemSlice> dsts, std::vector<MemSlice> mergedSrcs, std::vector<MemSlice> mergedDsts,
        CcuNodeRoleV3 role, DeviceId peerDeviceId)
    {
        auto task = std::make_unique<TaskBatchTransMem>(ProtocolType::CCU);
        task->SetSrcMemSlices(std::move(srcs));
        task->SetDstMemSlices(std::move(dsts));
        task->SetMergedSrcMemSlices(std::move(mergedSrcs));
        task->SetMergedDstMemSlices(std::move(mergedDsts));

        TaskNode* node = nullptr;
        HcclResult ret = curCcuTask->AppendGeneratedNode(std::move(task), deviceId, queId, role, node, peerDeviceId);
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
        return AppendTailNode(curCcuTask, queId, node);
    }

    HcclResult AddBatchReduce(
        DeviceId deviceId, uint32_t queId, CcuGraphStateV3* curCcuTask, std::vector<std::vector<MemSlice>> srcGroups,
        std::vector<MemSlice> dsts, std::vector<std::vector<MemSlice>> mergedSrcGroups,
        std::vector<MemSlice> mergedDsts, HcclDataType checkerDataType, HcclReduceOp checkerReduceOp,
        CcuNodeRoleV3 role, DeviceId peerDeviceId)
    {
        auto task = std::make_unique<TaskBatchReduce>(
            ToTaskDataType(checkerDataType), ToTaskReduceOp(checkerReduceOp), ProtocolType::CCU);
        task->SetSrcMemSlices(std::move(srcGroups));
        task->SetDstMemSlices(std::move(dsts));
        task->SetMergedSrcMemSlices(std::move(mergedSrcGroups));
        task->SetMergedDstMemSlices(std::move(mergedDsts));

        TaskNode* node = nullptr;
        HcclResult ret = curCcuTask->AppendGeneratedNode(std::move(task), deviceId, queId, role, node, peerDeviceId);
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
        return AppendTailNode(curCcuTask, queId, node);
    }

    void AddLocalCopy(
        DeviceId deviceId, uint32_t queId, CcuGraphStateV3* curCcuTask, const DataSlice& srcSlice,
        const DataSlice& dstSlice)
    {
        auto task = std::make_unique<TaskTransMem>(
            MakeCcuMemSlice(deviceId, srcSlice), MakeCcuMemSlice(deviceId, dstSlice), ProtocolType::CCU);
        TaskNode* node = nullptr;
        if (curCcuTask->AppendGeneratedNode(std::move(task), deviceId, queId, CcuNodeRoleV3::LOCAL_COPY, node)
            == HCCL_SUCCESS) {
            (void)AppendTailNode(curCcuTask, queId, node);
        }
    }

    void AddLocalReduce(
        DeviceId deviceId, uint32_t queId, CcuGraphStateV3* curCcuTask, const DataSlice& srcSlice,
        const DataSlice& dstSlice, HcclDataType checkerDataType, HcclReduceOp checkerReduceOp)
    {
        auto task = std::make_unique<TaskReduce>(
            MakeCcuMemSlice(deviceId, srcSlice), MakeCcuMemSlice(deviceId, dstSlice), ToTaskDataType(checkerDataType),
            ToTaskReduceOp(checkerReduceOp), ProtocolType::CCU);
        TaskNode* node = nullptr;
        if (curCcuTask->AppendGeneratedNode(std::move(task), deviceId, queId, CcuNodeRoleV3::LOCAL_REDUCE, node)
            == HCCL_SUCCESS) {
            (void)AppendTailNode(curCcuTask, queId, node);
        }
    }

    void AddLocalBatchReduce(
        DeviceId deviceId, uint32_t queId, CcuGraphStateV3* curCcuTask, const std::vector<DataSlice>& srcSlices,
        const DataSlice& dstSlice, DataType hcclDataType, HcclReduceOp checkerReduceOp)
    {
        HcclDataType checkerDataType = g_DataType2CheckerDataType_aicpu[hcclDataType];
        auto task = std::make_unique<TaskBatchReduce>(
            ToTaskDataType(checkerDataType), ToTaskReduceOp(checkerReduceOp), ProtocolType::CCU);
        const std::vector<MemSlice> srcMemSlices = MakeCcuMemSlices(deviceId, srcSlices);
        const MemSlice dstMemSlice = MakeCcuMemSlice(deviceId, dstSlice);
        task->AddSrcMemSlice(srcMemSlices);
        task->AddDstMemSlice(dstMemSlice);
        task->AddMergedSrcMemSlice(srcMemSlices);
        task->AddMergedDstMemSlice(dstMemSlice);

        TaskNode* node = nullptr;
        if (curCcuTask->AppendGeneratedNode(std::move(task), deviceId, queId, CcuNodeRoleV3::LOCAL_BATCH_REDUCE, node)
            == HCCL_SUCCESS) {
            (void)AppendTailNode(curCcuTask, queId, node);
        }
    }

    void AddWrite(
        DeviceId deviceId, DeviceId rmtDeviceId, uint32_t queId, CcuGraphStateV3* curCcuTask, const DataSlice& srcSlice,
        const DataSlice& dstSlice)
    {
        auto task = std::make_unique<TaskTransMem>(
            MakeCcuMemSlice(deviceId, srcSlice), MakeCcuMemSlice(rmtDeviceId, dstSlice), ProtocolType::CCU);
        TaskNode* node = nullptr;
        if (curCcuTask->AppendGeneratedNode(std::move(task), deviceId, queId, CcuNodeRoleV3::WRITE, node, rmtDeviceId)
            == HCCL_SUCCESS) {
            (void)AppendTailNode(curCcuTask, queId, node);
        }
    }

    void AddRead(
        DeviceId deviceId, DeviceId rmtDeviceId, uint32_t queId, CcuGraphStateV3* curCcuTask, const DataSlice& srcSlice,
        const DataSlice& dstSlice)
    {
        auto task = std::make_unique<TaskTransMem>(
            MakeCcuMemSlice(rmtDeviceId, srcSlice), MakeCcuMemSlice(deviceId, dstSlice), ProtocolType::CCU);
        TaskNode* node = nullptr;
        if (curCcuTask->AppendGeneratedNode(std::move(task), deviceId, queId, CcuNodeRoleV3::READ, node, rmtDeviceId)
            == HCCL_SUCCESS) {
            (void)AppendTailNode(curCcuTask, queId, node);
        }
    }

    void AddWriteReduce(
        DeviceId deviceId, DeviceId rmtDeviceId, uint32_t queId, CcuGraphStateV3* curCcuTask, const DataSlice& srcSlice,
        const DataSlice& dstSlice, HcclDataType checkerDataType, HcclReduceOp checkerReduceOp)
    {
        auto task = std::make_unique<TaskReduce>(
            MakeCcuMemSlice(deviceId, srcSlice), MakeCcuMemSlice(rmtDeviceId, dstSlice),
            ToTaskDataType(checkerDataType), ToTaskReduceOp(checkerReduceOp), ProtocolType::CCU);
        TaskNode* node = nullptr;
        if (curCcuTask->AppendGeneratedNode(
                std::move(task), deviceId, queId, CcuNodeRoleV3::WRITE_REDUCE, node, rmtDeviceId)
            == HCCL_SUCCESS) {
            (void)AppendTailNode(curCcuTask, queId, node);
        }
    }

    void AddReadReduce(
        DeviceId deviceId, DeviceId rmtDeviceId, uint32_t queId, CcuGraphStateV3* curCcuTask, const DataSlice& srcSlice,
        const DataSlice& dstSlice, HcclDataType checkerDataType, HcclReduceOp checkerReduceOp)
    {
        auto task = std::make_unique<TaskReduce>(
            MakeCcuMemSlice(rmtDeviceId, srcSlice), MakeCcuMemSlice(deviceId, dstSlice),
            ToTaskDataType(checkerDataType), ToTaskReduceOp(checkerReduceOp), ProtocolType::CCU);
        TaskNode* node = nullptr;
        if (curCcuTask->AppendGeneratedNode(
                std::move(task), deviceId, queId, CcuNodeRoleV3::READ_REDUCE, node, rmtDeviceId)
            == HCCL_SUCCESS) {
            (void)AppendTailNode(curCcuTask, queId, node);
        }
    }

    TaskNode* AddLoopStartTask(
        uint32_t queId, uint32_t loopIdx, uint32_t loopGroupIdx, CcuGraphStateV3* curCcuTask, uint16_t instrIdStart,
        uint16_t instrIdEnd, uint64_t loopCnt, uint64_t expandCnt)
    {
        auto task = std::make_unique<TaskStart>(BoundaryType::LOOP);
        TaskNode* node = nullptr;
        if (curCcuTask->AppendGeneratedNode(
                std::move(task), curCcuTask->GetDeviceId(), queId, CcuNodeRoleV3::LOOP_START, node)
            != HCCL_SUCCESS) {
            return nullptr;
        }
        (void)AppendTailNode(curCcuTask, queId, node);
        LoopInfoV3 loopInfo(node, nullptr);
        loopInfo.instrIdStart = instrIdStart;
        loopInfo.instrIdEnd = instrIdEnd;
        loopInfo.loopCnt = loopCnt;
        loopInfo.expandCnt = expandCnt;
        if (loopIdx == 0) {
            curCcuTask->loopGroupInfo_.push_back({loopInfo});
        } else {
            curCcuTask->loopGroupInfo_[loopGroupIdx].push_back(loopInfo);
        }
        if (node != nullptr) {
            CcuTraceInfo trace = node->GetCcuTrace();
            trace.loopInstrIdStart = instrIdStart;
            trace.loopInstrIdEnd = instrIdEnd;
            trace.loopCnt = loopCnt;
            trace.loopExpandCnt = expandCnt;
            node->SetCcuTrace(std::move(trace));
            curCcuTask->loopNodeIdStack_.push_back(node->GetNodeId());
        }
        return node;
    }

    TaskNode* AddLoopEndTask(uint32_t queId, uint32_t loopIdx, uint32_t loopGroupIdx, CcuGraphStateV3* curCcuTask)
    {
        auto task = std::make_unique<TaskEnd>(BoundaryType::LOOP);
        TaskNode* node = nullptr;
        if (curCcuTask->AppendGeneratedNode(
                std::move(task), curCcuTask->GetDeviceId(), queId, CcuNodeRoleV3::LOOP_END, node)
            != HCCL_SUCCESS) {
            return nullptr;
        }
        (void)AppendTailNode(curCcuTask, queId, node);
        curCcuTask->loopGroupInfo_[loopGroupIdx][loopIdx].loopEnd = node;
        if (node != nullptr) {
            curCcuTask->loopGroupInfo_[loopGroupIdx][loopIdx].endNodeId = node->GetNodeId();
        }
        if (!curCcuTask->loopNodeIdStack_.empty()) {
            curCcuTask->loopNodeIdStack_.pop_back();
        }
        TaskNode* startNode = curCcuTask->loopGroupInfo_[loopGroupIdx][loopIdx].loopStart;
        if (startNode != nullptr && node != nullptr) {
            CcuTraceInfo trace = startNode->GetCcuTrace();
            trace.loopEndNodeId = node->GetNodeId();
            startNode->SetCcuTrace(std::move(trace));
        }
        return node;
    }

    TaskNode* AddLoopGroupStartTask(uint32_t queId, CcuGraphStateV3* curCcuTask)
    {
        auto task = std::make_unique<TaskStart>(BoundaryType::LOOP);
        TaskNode* node = nullptr;
        if (curCcuTask->AppendGeneratedNode(
                std::move(task), curCcuTask->GetDeviceId(), queId, CcuNodeRoleV3::LOOP_START, node)
            != HCCL_SUCCESS) {
            return nullptr;
        }
        (void)AppendTailNode(curCcuTask, queId, node);
        return node;
    }

    TaskNode* CreateLoopGroupEndTask(uint32_t queId, CcuGraphStateV3* curCcuTask)
    {
        auto task = std::make_unique<TaskEnd>(BoundaryType::LOOP);
        TaskNode* node = nullptr;
        if (curCcuTask->AppendGeneratedNode(
                std::move(task), curCcuTask->GetDeviceId(), queId, CcuNodeRoleV3::LOOP_END, node)
            != HCCL_SUCCESS) {
            return nullptr;
        }
        return node;
    }

    void AddPost(
        DeviceId deviceId, DeviceId rmtDeviceId, uint32_t queId, CcuGraphStateV3* curCcuTask, uint32_t rmtDieId,
        uint16_t rmtCKEId, uint16_t setRmtCKEMask)
    {
        CcuNotify notify;
        notify.recordDeviceId = deviceId;
        notify.waitDeviceId = rmtDeviceId;
        notify.dieId = rmtDieId;
        notify.ckeId = rmtCKEId;
        notify.ckeMask = setRmtCKEMask;
        auto task = std::make_unique<TaskRecordCCU>(notify, ProtocolType::CCU);
        TaskNode* node = nullptr;
        if (curCcuTask->AppendGeneratedNode(
                std::move(task), deviceId, queId, CcuNodeRoleV3::POST, node, rmtDeviceId, setRmtCKEMask, rmtDieId,
                rmtCKEId)
            != HCCL_SUCCESS) {
            return;
        }
        (void)AppendTailNode(curCcuTask, queId, node);
        AllRankParamRecorder::Global()->seenPost[rmtDeviceId][rmtDieId][rmtCKEId].insert(node);
        CcuPostNodeMetaV3 meta;
        meta.recordDeviceId = deviceId;
        meta.waitDeviceId = rmtDeviceId;
        meta.dieId = rmtDieId;
        meta.ckeId = rmtCKEId;
        meta.remainingCkeMask = setRmtCKEMask;
        meta.isLocal = false;
        AllRankParamRecorder::Global()->RegisterPostNode(node, meta);
    }

    TaskNode* AddWait(
        DeviceId deviceId, uint32_t queId, CcuGraphStateV3* curCcuTask, uint32_t dieId, uint16_t waitCKEId,
        uint16_t remoteWaitMask)
    {
        CcuNotify notify;
        notify.waitDeviceId = deviceId;
        notify.dieId = dieId;
        notify.ckeId = waitCKEId;
        notify.ckeMask = remoteWaitMask;
        auto task = std::make_unique<TaskWaitCCU>(notify, ProtocolType::CCU);
        TaskNode* node = nullptr;
        if (curCcuTask->AppendGeneratedNode(
                std::move(task), deviceId, queId, CcuNodeRoleV3::WAIT, node, INVALID_DEVICE_ID, remoteWaitMask, dieId,
                waitCKEId)
            != HCCL_SUCCESS) {
            return nullptr;
        }
        TaskNode* preNode
            = (curCcuTask->tailNodes[queId] == nullptr) ? curCcuTask->ccuHeadTaskNode : curCcuTask->tailNodes[queId];
        AddNodeRelation(preNode, node);
        curCcuTask->tailNodes[queId] = node;
        return node;
    }

    void AddLocalPost(
        DeviceId deviceId, uint32_t queId, uint32_t dieId, CcuGraphStateV3* curCcuTask, uint16_t setCKEId,
        uint16_t setCKEMask, bool invalidPost)
    {
        CcuNotify notify;
        notify.recordDeviceId = deviceId;
        notify.waitDeviceId = deviceId;
        notify.dieId = dieId;
        notify.ckeId = setCKEId;
        notify.ckeMask = setCKEMask;
        auto task = std::make_unique<TaskRecordCCU>(notify, ProtocolType::CCU);
        TaskNode* node = nullptr;
        if (curCcuTask->AppendGeneratedNode(
                std::move(task), deviceId, queId, CcuNodeRoleV3::LOCAL_POST_TO, node, deviceId, setCKEMask, dieId,
                setCKEId, invalidPost)
            != HCCL_SUCCESS) {
            return;
        }
        (void)AppendTailNode(curCcuTask, queId, node);
        AllRankParamRecorder::Global()->seenPost[deviceId][dieId][setCKEId].insert(node);
        CcuPostNodeMetaV3 meta;
        meta.recordDeviceId = deviceId;
        meta.waitDeviceId = deviceId;
        meta.dieId = dieId;
        meta.ckeId = setCKEId;
        meta.remainingCkeMask = setCKEMask;
        meta.isLocal = true;
        AllRankParamRecorder::Global()->RegisterPostNode(node, meta);
    }

    TaskNode* AddLocalWait(
        DeviceId deviceId, uint32_t queId, CcuGraphStateV3* curCcuTask, uint32_t dieId, uint16_t waitCKEId,
        uint16_t localWaitMask)
    {
        CcuNotify notify;
        notify.recordDeviceId = deviceId;
        notify.waitDeviceId = deviceId;
        notify.dieId = dieId;
        notify.ckeId = waitCKEId;
        notify.ckeMask = localWaitMask;
        auto task = std::make_unique<TaskWaitCCU>(notify, ProtocolType::CCU);
        TaskNode* node = nullptr;
        if (curCcuTask->AppendGeneratedNode(
                std::move(task), deviceId, queId, CcuNodeRoleV3::LOCAL_WAIT_FROM, node, deviceId, localWaitMask, dieId,
                waitCKEId)
            == HCCL_SUCCESS) {
            (void)AppendTailNode(curCcuTask, queId, node);
        }
        return node;
    }

    void AddCcuSubGraphEnd(TaskNode* node, CcuGraphStateV3* curCcuTask)
    {
        (void)node;
        if (curCcuTask == nullptr || curCcuTask->ccuSubGraphEndNode != nullptr) {
            return;
        }
        TaskPosition position = curCcuTask->ccuGraph.GetPosition();
        position.queueId = INVALID_QUEUE_ID;
        auto task = std::make_unique<TaskEnd>(BoundaryType::CCU_SUB_GRAPH);
        TaskNode* endNode = nullptr;
        if (curCcuTask->AppendGeneratedNode(std::move(task), position, CcuNodeRoleV3::SUB_GRAPH_END, endNode)
            != HCCL_SUCCESS) {
            return;
        }
        curCcuTask->ccuSubGraphEndNode = endNode;
        bool hasTail = false;
        for (auto* tailNode : curCcuTask->tailNodes) {
            if (tailNode == nullptr) {
                continue;
            }
            AddNodeRelation(tailNode, endNode);
            hasTail = true;
        }
        if (!hasTail) {
            AddNodeRelation(curCcuTask->ccuHeadTaskNode, endNode);
        }
    }

    HcclResult ClearWaitMask(DeviceId deviceId, uint32_t dieId, uint16_t waitCKEId, uint16_t waitCKEMask)
    {
        uint16_t ckeValue = 0;
        CHK_RET(AllRankParamRecorder::Global()->GetCKE(deviceId, dieId, waitCKEId, ckeValue));
        CHK_RET(AllRankParamRecorder::Global()->SetCKE(deviceId, dieId, waitCKEId, ckeValue & (~waitCKEMask)));
        return HCCL_SUCCESS;
    }

    HcclResult ProcessSetMask(
        DeviceId deviceId, uint32_t dieId, CcuGraphStateV3* curCcuTask, uint32_t queId, uint16_t setCKEId,
        uint16_t setCKEMask, bool invalidPost)
    {
        if (setCKEMask != 0x0000) {
            AddLocalPost(deviceId, queId, dieId, curCcuTask, setCKEId, setCKEMask, invalidPost);
            uint16_t ckeValue = 0;
            CHK_RET(AllRankParamRecorder::Global()->GetCKE(deviceId, dieId, setCKEId, ckeValue));
            CHK_RET(AllRankParamRecorder::Global()->SetCKE(deviceId, dieId, setCKEId, ckeValue | setCKEMask));
        }

        return HCCL_SUCCESS;
    }

    uint16_t
    UpdateId(uint16_t CKEId, uint32_t curLoopIdx, uint32_t expandOffset, uint32_t ckOffset, uint32_t curExpandCnt)
    {
        if (curLoopIdx < expandOffset) {
            return CKEId;
        }
        return CKEId + curExpandCnt * ckOffset;
    }

    HcclResult GenSliceFromMs(uint16_t msId, uint64_t len, DataSlice& slice)
    {
        constexpr uint32_t MS_LEN = 4 * 1024;

        slice.SetBufferType(BufferType::MS);
        slice.SetOffset(msId * MS_LEN);
        slice.SetSize(len);

        return HCCL_SUCCESS;
    }

} // namespace TaskGraphGeneratorV3
} // namespace HcclSim
