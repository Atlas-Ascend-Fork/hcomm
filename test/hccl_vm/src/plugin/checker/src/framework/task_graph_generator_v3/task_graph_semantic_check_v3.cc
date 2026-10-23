/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_graph_semantic_check_v3.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "framework/semantics_check/all2all_semantics_checker.h"
#include "framework/semantics_check/allgather_semantics_checker.h"
#include "framework/semantics_check/allgather_v_semantics_checker.h"
#include "framework/semantics_check/allreduce_semantics_checker.h"
#include "framework/semantics_check/batchsendrecv_semantics_checker.h"
#include "framework/semantics_check/broadcast_semantics_checker.h"
#include "framework/semantics_check/reduce_scatter_semantics_checker.h"
#include "framework/semantics_check/reduce_scatter_v_semantics_checker.h"
#include "framework/semantics_check/reduce_semantics_checker.h"
#include "framework/semantics_check/scatter_semantics_checker.h"
#include "framework/semantics_check/send_recv_semantics_checker.h"
#include "sim_log.h"
#include "utils/check_utils.h"
#include "utils/dump/dump_json_utils.h"
#include "utils/error_codes.h"
#include "utils/storage_manager.h"

namespace HcclSim {
namespace TaskGraphGeneratorV3 {
    namespace {
        // CCU 的单个 MS 槽位按 4KB 对齐管理；语义校验里用它把复用的 MS 拆成独立桶。
        constexpr uint64_t kMsSliceSize = 4ULL * 1024ULL;
        enum class SliceOpV3 : uint8_t {
            OVERRIDE = 0,
            REDUCE,
        };

        std::string DescribeMemSliceForSemantic(const MemSlice& slice)
        {
            std::ostringstream os;
            os << "{deviceId=";
            if (slice.deviceId == INVALID_DEVICE_ID) {
                os << "invalid";
            } else {
                os << slice.deviceId;
            }
            os << ", memoryType=" << DescribeMemType(slice.memType) << ", rawAddr=0x" << std::hex << slice.rawAddr
               << ", offset=0x" << slice.offset << ", length=0x" << slice.len << std::dec << "}";
            return os.str();
        }

        struct SliceOpPairV3 {
            DeviceId srcDeviceId{INVALID_DEVICE_ID};
            DeviceId dstDeviceId{INVALID_DEVICE_ID};
            MemSlice src;
            MemSlice dst;
            TaskPosition position;
            SliceOpV3 op{SliceOpV3::OVERRIDE};
            HcclReduceOp reduceType{HCCL_REDUCE_RESERVED};
            uint64_t index{0};
            bool useBatchIndex{false};
            HcclDataType dataType{HCCL_DATA_TYPE_RESERVED};
            bool knownReduceOrder{true}; // 多源 reduce 的组内顺序未编码，置 false 时不臆造结合顺序

            std::string Describe() const
            {
                std::ostringstream os;
                os << "{operation=" << (op == SliceOpV3::REDUCE ? "reduce" : "overwrite")
                   << ", sourceMemorySlice=" << DescribeMemSliceForSemantic(src)
                   << ", targetMemorySlice=" << DescribeMemSliceForSemantic(dst) << ", launchIdx=" << position.launchIdx
                   << ", blockId=" << position.blockId << ", pipeId=" << position.pipe
                   << ", taskId=" << position.taskId;
                if (useBatchIndex) {
                    os << ", batchGroup=" << index;
                }
                if (op == SliceOpV3::REDUCE) {
                    os << ", reduceType=" << DumpReduceOpToString(reduceType);
                }
                os << "}";
                return os.str();
            }
        };

        // 对 MS 语义做“按 device + msId + batch 下标”分桶。
        // 普通内存仍然共用原来的 INPUT/OUTPUT/CCL 语义表，不会被 batch index 隔离。
        struct MsKey {
            DeviceId deviceId{INVALID_DEVICE_ID};
            uint64_t msId{0};
            uint64_t index{0};

            bool operator<(const MsKey& rhs) const
            {
                if (deviceId != rhs.deviceId) {
                    return deviceId < rhs.deviceId;
                }
                if (msId != rhs.msId) {
                    return msId < rhs.msId;
                }
                return index < rhs.index;
            }
        };

        struct AivMemoryKey {
            DeviceId deviceId{INVALID_DEVICE_ID};
            MemType memType{MemType::INVALID};
            uint64_t aivUBIdx{0};
            uint64_t batchIndex{0};
            bool isBatchUB{false};

            bool operator<(const AivMemoryKey& rhs) const
            {
                if (deviceId != rhs.deviceId) {
                    return deviceId < rhs.deviceId;
                }
                if (memType != rhs.memType) {
                    return static_cast<uint32_t>(memType) < static_cast<uint32_t>(rhs.memType);
                }
                if (aivUBIdx != rhs.aivUBIdx) {
                    return aivUBIdx < rhs.aivUBIdx;
                }
                if (batchIndex != rhs.batchIndex) {
                    return batchIndex < rhs.batchIndex;
                }
                return isBatchUB < rhs.isBatchUB;
            }
        };

        // 对同一个 task 先把所有源语义拷出来，再统一写回目标，
        // 这样 batch 任务内部的后一个 pair 不会读到前一个 pair 刚刚写出的结果。
        struct PreparedSliceOpPairV3 {
            SliceOpPairV3 pair;
            std::vector<BufferSemantic> srcSemantics;
        };

        using InternalBufferSemanticMap = BufferSemanticMap;

        uint64_t GetSegmentEndAddr(const BufferSemantic& semantic);

        struct SemanticState {
            CheckerParam param;
            uint64_t dataSize{0};
            uint64_t outputSize{0};
            // 绝对地址语义表：INPUT/OUTPUT/CCL 三种逻辑视图物理合一，inplace 别名天然可见，
            // 地址重叠在模拟过程中无需任何特殊处理。
            PhysicalMemorySemantics physicalSemantics;
            DeviceBufferAddressLayouts bufferLayouts;
            std::map<MsKey, InternalBufferSemanticMap> ms;
            std::map<AivMemoryKey, InternalBufferSemanticMap> AivDevMem;
            bool orderEnabled{false};
            ReduceOrderArena orderArena;
            ReduceOrderReport orderReport;
            bool orderIncomplete{false};
        };

        bool IsDataTask(const TaskNode* node)
        {
            if (node == nullptr) {
                return false;
            }
            switch (node->GetType()) {
                case TaskType::TRANS_MEM:
                case TaskType::BATCH_TRANS_MEM:
                case TaskType::REDUCE:
                case TaskType::BATCH_REDUCE:
                    return true;
                default:
                    return false;
            }
        }

        bool IsSupportedSemanticMemType(MemType memType)
        {
            switch (memType) {
                case MemType::INPUT:
                case MemType::OUTPUT:
                case MemType::CCL:
                case MemType::MS_CCU:
                case MemType::AIV_UB:
                case MemType::AIV_COMM:
                    return true;
                default:
                    return false;
            }
        }

        bool IsValidMemorySlice(const MemSlice& slice)
        {
            return slice.deviceId != INVALID_DEVICE_ID && slice.memType != MemType::INVALID;
        }

        MsKey MakeMsKey(DeviceId deviceId, uint64_t address, uint64_t index)
        {
            return MsKey{deviceId, address / kMsSliceSize, index};
        }

        uint64_t MakeAivUbIdx(const TaskPosition& position)
        {
            if (position.launchIdx == std::numeric_limits<uint64_t>::max()
                || position.blockId == std::numeric_limits<uint32_t>::max()) {
                return 0;
            }
            return (position.launchIdx << 32U) | static_cast<uint64_t>(position.blockId);
        }

        AivMemoryKey MakeAivMemoryKey(
            DeviceId deviceId, MemType memType, const TaskPosition& position, uint64_t batchIndex, bool isBatchUB)
        {
            if (memType == MemType::AIV_COMM) {
                return AivMemoryKey{deviceId, memType, 0U, 0U, false};
            }
            const bool effectiveIsBatchUB = memType == MemType::AIV_UB && isBatchUB;
            return AivMemoryKey{
                deviceId, memType, MakeAivUbIdx(position), effectiveIsBatchUB ? batchIndex : 0U, effectiveIsBatchUB};
        }

        // 全局物理内存（INPUT/OUTPUT/CCL）按 device 取同一张绝对地址语义表，inplace 别名天然可见；
        // MS 额外按 batch 下标拆桶，把“同一个 ms 在不同 batch item 复用”的语义分开记录；
        // AIV UB/COMM 维持按 launch/block 划分的独立地址空间。
        InternalBufferSemanticMap& GetBufferSemanticMap(
            SemanticState& state, DeviceId deviceId, MemType memType, uint64_t address, const TaskPosition& position,
            uint64_t index, bool useBatchIndex)
        {
            if (memType == MemType::MS_CCU) {
                return state.ms[MakeMsKey(deviceId, address, index)];
            }
            if (memType == MemType::AIV_UB || memType == MemType::AIV_COMM) {
                return state.AivDevMem[MakeAivMemoryKey(deviceId, memType, position, index, useBatchIndex)];
            }
            return state.physicalSemantics[deviceId];
        }

        const InternalBufferSemanticMap& GetBufferSemanticMap(
            const SemanticState& state, DeviceId deviceId, MemType memType, uint64_t address,
            const TaskPosition& position, uint64_t index, bool useBatchIndex)
        {
            static const InternalBufferSemanticMap empty;
            if (memType == MemType::MS_CCU) {
                const auto iter = state.ms.find(MakeMsKey(deviceId, address, index));
                return iter == state.ms.end() ? empty : iter->second;
            }
            if (memType == MemType::AIV_UB || memType == MemType::AIV_COMM) {
                const auto iter
                    = state.AivDevMem.find(MakeAivMemoryKey(deviceId, memType, position, index, useBatchIndex));
                return iter == state.AivDevMem.end() ? empty : iter->second;
            }
            const auto deviceIter = state.physicalSemantics.find(deviceId);
            return deviceIter == state.physicalSemantics.end() ? empty : deviceIter->second;
        }

        // 语义操作使用切片所属地址空间中的地址。
        uint64_t GetSliceAddress(const MemSlice& slice) { return slice.MemoryBegin(); }

        void InitDeviceBuffers(SemanticState& state, DeviceId deviceId, bool initInput, uint64_t inputSize)
        {
            state.physicalSemantics[deviceId];
            if (!initInput) {
                return;
            }
            const auto layoutIter = state.bufferLayouts.find(deviceId);
            if (layoutIter == state.bufferLayouts.end() || !layoutIter->second.hasInput) {
                // 布局缺失时跳过初始输入 seed（与输入缓冲未注册的历史行为一致），显式告警便于定位。
                HCCL_VM_WARN(
                    "Input buffer base is missing from the memory layout, skip the initial input "
                    "semantic seed, deviceId={}",
                    deviceId);
                return;
            }
            BufferSemantic input(layoutIter->second.inputBase, inputSize);
            input.srcBufs.insert(SrcBufDes(deviceId, BufferType::INPUT, layoutIter->second.inputBase));
            if (state.orderEnabled) {
                input.reduceOrderExpr = state.orderArena.Leaf(deviceId);
            }
            state.physicalSemantics[deviceId].emplace(input.startAddr, std::move(input));
        }

        HcclResult InitState(
            SemanticState& state, OperatorId operatorId, const TaskPosition& rootPosition,
            const std::map<DeviceId, RankId>& deviceToRank)
        {
            StorageManager& storage = StorageManager::GetInstance();
            // 主 START 节点可能未携带通信域身份：回退到当前算子组的通信域标识（与 GetTaskDataSlice 的回退一致）。
            // 位置不再携带 commName；按 commId 从缓存解析，未命中则回退当前算子组（与 GetTaskDataSlice 一致）。
            const std::string& rootCommName = storage.GetCommNameById(rootPosition.commId);
            const std::string effectiveCommName = rootCommName.empty() ? storage.GetCurrentCommName() : rootCommName;
            const uint64_t effectiveCommHash
                = rootCommName.empty() ? storage.GetCurrentCommHash() : rootPosition.commHash;
            const uint32_t effectiveOpIter = rootCommName.empty() ? storage.GetCurrentOpIter() : rootPosition.opIter;
            state.param = storage.GetCheckerParam(operatorId);
            if (state.param.rankSize == 0) {
                HCCL_VM_ERROR(
                    "{} Semantic check initialization failed because the rank count is 0, "
                    "collectiveType={}, dataType={}, elementCount={}, reduceType={}",
                    MakeErrorCodeText(ErrorCode::CHECKER_RUNTIME_ERROR), HcclCmdTypeToString(state.param.cmdType),
                    HcclDataTypeToString(state.param.dataType), state.param.dataCount,
                    DumpReduceOpToString(state.param.reduceType));
                return HCCL_E_PARA;
            }
            const bool orderCommand
                = state.param.cmdType == HCCL_CMD_ALLREDUCE || state.param.cmdType == HCCL_CMD_REDUCE_SCATTER;
            const bool orderType
                = state.param.dataType == HCCL_DATA_TYPE_FP16 || state.param.dataType == HCCL_DATA_TYPE_FP32
                  || state.param.dataType == HCCL_DATA_TYPE_BFP16 || state.param.dataType == HCCL_DATA_TYPE_FP64;
            const bool orderOp
                = state.param.reduceType == HCCL_REDUCE_SUM || state.param.reduceType == HCCL_REDUCE_PROD;
            // 只依据检查器进程当前的环境变量开启，不采信用例里记录下来的 opExtInfo。
            bool strictRequested = false;
            if (orderCommand) {
                const char* value = std::getenv("HCCL_DETERMINISTIC");
                std::string mode = value == nullptr ? "" : value;
                for (char& letter : mode) {
                    if (letter >= 'a' && letter <= 'z') {
                        letter = static_cast<char>(letter - 'a' + 'A');
                    }
                }
                strictRequested = mode == "STRICT";
            }
            state.orderEnabled = strictRequested && orderType && orderOp && state.param.rankSize > 2;
            if (strictRequested && !state.orderEnabled) {
                // 请求了 strict 但算子不在契约内：不算失败，仅在 detail 里说明原因。
                state.orderReport.detail
                    = "HCCL strict-order contract applies to floating-point SUM/PROD with rankSize > 2";
            }
            if (state.orderEnabled) {
                // 元素总量按字节换算不能溢出，否则环绕后的窗口会被当成合法空输出。
                const uint64_t width = CHECK_SIZE_TABLE[state.param.dataType];
                const uint64_t multiplier = state.param.cmdType == HCCL_CMD_REDUCE_SCATTER ? state.param.rankSize : 1;
                if (state.param.dataCount > std::numeric_limits<uint64_t>::max() / width
                    || state.param.dataCount * width > std::numeric_limits<uint64_t>::max() / multiplier) {
                    state.orderReport.status = ReduceOrderStatus::INCOMPLETE;
                    state.orderReport.detail = "operator input/output byte extent overflows uint64";
                    return HCCL_E_PARA;
                }
            }
            CalcDataSize(state.param.cmdType, state.param.dataCount, state.param.dataType, state.dataSize);
            for (const auto& [deviceId, rankId] : deviceToRank) {
                uint64_t inputSize = 0;
                RankId srcRank = state.param.srcRank;
                RankId dstRank = state.param.dstRank;
                for (const auto& pair : state.param.sendRecvPairs) {
                    if (pair.srcRank == rankId || pair.dstRank == rankId) {
                        srcRank = pair.srcRank;
                        dstRank = pair.dstRank;
                        break;
                    }
                }
                CalcInputOutputSize(
                    state.param.cmdType, state.param.rankSize, state.param.dataCount, state.param.dataType, inputSize,
                    state.outputSize, rankId, srcRank, dstRank, state.param.vDataDes, state.param.all2AllDataDes);
                const bool initInput
                    = !(state.param.cmdType == HCCL_CMD_BROADCAST || state.param.cmdType == HCCL_CMD_SCATTER)
                      || rankId == state.param.root;
                if (deviceId == INVALID_DEVICE_ID) {
                    HCCL_VM_ERROR(
                        "{} Semantic check initialization failed because rank has no device mapping, rankId={}",
                        MakeErrorCodeText(ErrorCode::CHECKER_RUNTIME_ERROR), rankId);
                    return HCCL_E_PARA;
                }
                BufferAddressLayout bufferLayout;
                // 出参类型与 StorageManager 接口保持一致（uint64_t&），BufferAddressLayout 字段同为 uint64_t。
                const auto loadBufferBase = [&](BufferType bufferType, bool& has, uint64_t& base) {
                    if (storage.GetBufferBaseAddress(
                            effectiveCommName, effectiveCommHash, effectiveOpIter, deviceId, bufferType, base)) {
                        has = true;
                    }
                };
                loadBufferBase(BufferType::INPUT, bufferLayout.hasInput, bufferLayout.inputBase);
                loadBufferBase(BufferType::OUTPUT, bufferLayout.hasOutput, bufferLayout.outputBase);
                loadBufferBase(BufferType::CCL, bufferLayout.hasCcl, bufferLayout.cclBase);
                // 结果窗口校验还需要缓冲区的注册大小，用于识别落在期望窗口之后、但仍位于结果缓冲区内的多余输出段。
                // GetBlockSize 对未注册的缓冲返回 0，与“大小未知则跳过多余输出校验”的约定一致。
                bufferLayout.inputSize = storage.GetBlockSize(
                    effectiveCommName, effectiveCommHash, effectiveOpIter, deviceId, BufferType::INPUT);
                bufferLayout.outputSize = storage.GetBlockSize(
                    effectiveCommName, effectiveCommHash, effectiveOpIter, deviceId, BufferType::OUTPUT);
                // 结果窗口基址缺失时，最终校验会整体回退到 0 基址，在不相干的地址上产生难以理解的报错，
                // 必须在这里显式告警暴露根因（Broadcast 以 INPUT 作为结果窗口，其余算子使用 OUTPUT）。
                const bool resultBaseMissing
                    = (state.param.cmdType == HCCL_CMD_BROADCAST) ? !bufferLayout.hasInput : !bufferLayout.hasOutput;
                if (resultBaseMissing) {
                    HCCL_VM_WARN(
                        "Result buffer base is missing from the memory layout, the final validation window "
                        "falls back to a zero base and the result may be unreliable, deviceId={}, bufferType={}, "
                        "collectiveType={}",
                        deviceId,
                        state.param.cmdType == HCCL_CMD_BROADCAST ? BufferTypeToString(BufferType::INPUT) :
                                                                    BufferTypeToString(BufferType::OUTPUT),
                        HcclCmdTypeToString(state.param.cmdType));
                }
                // 记录每个设备各自的输入/输出尺寸（send/recv 等场景各 rank 不同）。
                state.bufferLayouts[deviceId] = bufferLayout;
                InitDeviceBuffers(state, deviceId, initInput, inputSize);
            }
            return HCCL_SUCCESS;
        }

        uint64_t GetSegmentEndAddr(const BufferSemantic& semantic) { return semantic.startAddr + semantic.size; }

        BufferSemantic CloneSegmentRange(const BufferSemantic& semantic, uint64_t newStartAddr, uint64_t newEndAddr)
        {
            const uint64_t sourceAddressDelta = newStartAddr - semantic.startAddr;
            BufferSemantic cloned(
                newStartAddr, newEndAddr - newStartAddr, semantic.isReduce, semantic.reduceType,
                ShiftSourceAddresses(semantic.srcBufs, sourceAddressDelta));
            cloned.reduceOrderExpr = semantic.reduceOrderExpr;
            return cloned;
        }

        InternalBufferSemanticMap::iterator
        FindFirstOverlap(InternalBufferSemanticMap& bufferSemantics, uint64_t startAddr, uint64_t endAddr)
        {
            auto iter = bufferSemantics.lower_bound(startAddr);
            if (iter != bufferSemantics.begin()) {
                auto prevIter = std::prev(iter);
                if (GetSegmentEndAddr(prevIter->second) > startAddr) {
                    iter = prevIter;
                }
            }
            while (iter != bufferSemantics.end() && GetSegmentEndAddr(iter->second) <= startAddr) {
                ++iter;
            }
            if (iter != bufferSemantics.end() && iter->second.startAddr >= endAddr) {
                return bufferSemantics.end();
            }
            return iter;
        }

        InternalBufferSemanticMap::const_iterator
        FindFirstOverlap(const InternalBufferSemanticMap& bufferSemantics, uint64_t startAddr, uint64_t endAddr)
        {
            auto iter = bufferSemantics.lower_bound(startAddr);
            if (iter != bufferSemantics.begin()) {
                auto prevIter = std::prev(iter);
                if (GetSegmentEndAddr(prevIter->second) > startAddr) {
                    iter = prevIter;
                }
            }
            while (iter != bufferSemantics.end() && GetSegmentEndAddr(iter->second) <= startAddr) {
                ++iter;
            }
            if (iter != bufferSemantics.end() && iter->second.startAddr >= endAddr) {
                return bufferSemantics.end();
            }
            return iter;
        }

        void SplitBufferSemantic(InternalBufferSemanticMap& bufferSemantics, uint64_t splitAddr)
        {
            if (bufferSemantics.empty()) {
                return;
            }

            auto iter = bufferSemantics.upper_bound(splitAddr);
            if (iter == bufferSemantics.begin()) {
                return;
            }
            --iter;
            BufferSemantic& semantic = iter->second;
            if (splitAddr <= semantic.startAddr || splitAddr >= GetSegmentEndAddr(semantic)) {
                return;
            }

            BufferSemantic rightSemantic = CloneSegmentRange(semantic, splitAddr, GetSegmentEndAddr(semantic));
            semantic.size = splitAddr - semantic.startAddr;
            bufferSemantics.emplace(rightSemantic.startAddr, std::move(rightSemantic));
        }

        void CollectOverlappingBufferSemantics(
            InternalBufferSemanticMap& bufferSemantics, uint64_t startAddr, uint64_t endAddr,
            std::vector<BufferSemantic*>& overlaps)
        {
            if (startAddr >= endAddr) {
                return;
            }

            auto iter = FindFirstOverlap(bufferSemantics, startAddr, endAddr);
            while (iter != bufferSemantics.end() && iter->second.startAddr < endAddr) {
                overlaps.push_back(&iter->second);
                ++iter;
            }
        }

        void CollectOverlappingBufferSemantics(
            const InternalBufferSemanticMap& bufferSemantics, uint64_t startAddr, uint64_t endAddr,
            std::vector<const BufferSemantic*>& overlaps)
        {
            if (startAddr >= endAddr) {
                return;
            }

            auto iter = FindFirstOverlap(bufferSemantics, startAddr, endAddr);
            while (iter != bufferSemantics.end() && iter->second.startAddr < endAddr) {
                overlaps.push_back(&iter->second);
                ++iter;
            }
        }

        void
        EraseBufferSemanticsInRange(InternalBufferSemanticMap& bufferSemantics, uint64_t startAddr, uint64_t endAddr)
        {
            if (startAddr >= endAddr) {
                return;
            }

            auto iter = bufferSemantics.lower_bound(startAddr);
            while (iter != bufferSemantics.end() && iter->second.startAddr < endAddr) {
                iter = bufferSemantics.erase(iter);
            }
        }

        bool CanMergeSrcBufs(const BufferSemantic& lhs, const BufferSemantic& rhs)
        {
            if (lhs.srcBufs.size() != rhs.srcBufs.size()) {
                return false;
            }

            auto lhsIter = lhs.srcBufs.begin();
            auto rhsIter = rhs.srcBufs.begin();
            while (lhsIter != lhs.srcBufs.end() && rhsIter != rhs.srcBufs.end()) {
                if (lhsIter->deviceId != rhsIter->deviceId || lhsIter->bufType != rhsIter->bufType
                    || lhsIter->srcAddr + lhs.size != rhsIter->srcAddr) {
                    return false;
                }
                ++lhsIter;
                ++rhsIter;
            }
            return lhsIter == lhs.srcBufs.end() && rhsIter == rhs.srcBufs.end();
        }

        bool CanMergeSemantics(const BufferSemantic& lhs, const BufferSemantic& rhs)
        {
            return GetSegmentEndAddr(lhs) == rhs.startAddr && lhs.isReduce == rhs.isReduce
                   && lhs.reduceType == rhs.reduceType && lhs.reduceOrderExpr == rhs.reduceOrderExpr
                   && CanMergeSrcBufs(lhs, rhs);
        }

        void
        MergeAdjacentBufferSemantics(InternalBufferSemanticMap& bufferSemantics, uint64_t startAddr, uint64_t endAddr)
        {
            if (bufferSemantics.empty()) {
                return;
            }

            auto iter = bufferSemantics.lower_bound(startAddr);
            if (iter != bufferSemantics.begin()) {
                --iter;
            }
            while (iter != bufferSemantics.end()) {
                auto nextIter = std::next(iter);
                if (nextIter == bufferSemantics.end()) {
                    break;
                }
                if (iter->second.startAddr > endAddr && nextIter->second.startAddr > endAddr) {
                    break;
                }
                if (!CanMergeSemantics(iter->second, nextIter->second)) {
                    iter = nextIter;
                    continue;
                }

                iter->second.size += nextIter->second.size;
                endAddr = std::max(endAddr, GetSegmentEndAddr(nextIter->second));
                bufferSemantics.erase(nextIter);
            }
        }

        void InsertBufferSemantic(InternalBufferSemanticMap& bufferSemantics, BufferSemantic semantic)
        {
            if (semantic.size == 0) {
                return;
            }
            auto iter = bufferSemantics.lower_bound(semantic.startAddr);
            if (iter != bufferSemantics.end() && iter->first == semantic.startAddr) {
                iter->second = std::move(semantic);
                return;
            }
            bufferSemantics.emplace_hint(iter, semantic.startAddr, std::move(semantic));
        }

        bool GetSourceIntersection(
            const SliceOpPairV3& pair, const BufferSemantic& srcSemantic, uint64_t& srcStartAddr, uint64_t& srcEndAddr)
        {
            srcStartAddr = GetSliceAddress(pair.src);
            srcEndAddr = srcStartAddr + pair.src.len;
            if (srcSemantic.startAddr > srcStartAddr) {
                srcStartAddr = srcSemantic.startAddr;
            }
            if (srcSemantic.startAddr + srcSemantic.size < srcEndAddr) {
                srcEndAddr = srcSemantic.startAddr + srcSemantic.size;
            }
            return srcStartAddr < srcEndAddr;
        }

        HcclResult CheckBufSemantics(
            const std::vector<const BufferSemantic*>& bufSemantics, uint64_t startAddr, uint64_t size, bool ignoreError)
        {
            if (size == 0) {
                return HCCL_SUCCESS;
            }
            if (bufSemantics.empty()) {
                if (!ignoreError) {
                    HCCL_VM_ERROR(
                        "{} No source/output information was found for the target memory range, "
                        "startAddr=0x{:x}, size=0x{:x}",
                        MakeErrorCodeText(ErrorCode::SEMANTIC_BUFFER_EMPTY), startAddr, size);
                }
                return HCCL_E_PARA;
            }

            uint64_t totalSize = 0;
            const BufferSemantic* prev = bufSemantics.front();
            if (prev->startAddr > startAddr) {
                if (!ignoreError) {
                    HCCL_VM_ERROR(
                        "{} Output data does not start from the expected address; the beginning is "
                        "missing, expectedStart=0x{:x}, actualStart=0x{:x}",
                        MakeErrorCodeText(ErrorCode::SEMANTIC_GAP), startAddr, prev->startAddr);
                }
                return HCCL_E_PARA;
            }
            if (prev->startAddr + prev->size >= startAddr + size) {
                totalSize += size;
            } else {
                totalSize += prev->startAddr + prev->size - startAddr;
            }

            for (size_t index = 1; index < bufSemantics.size(); ++index) {
                const BufferSemantic* cur = bufSemantics[index];
                if (cur->startAddr != prev->startAddr + prev->size) {
                    if (!ignoreError) {
                        HCCL_VM_ERROR(
                            "{} Output data is broken in the middle; one piece ends at 0x{:x} "
                            "but the next starts at 0x{:x}",
                            MakeErrorCodeText(ErrorCode::SEMANTIC_GAP), prev->startAddr + prev->size, cur->startAddr);
                    }
                    return HCCL_E_PARA;
                }
                if (cur->startAddr + cur->size >= startAddr + size) {
                    totalSize += startAddr + size - cur->startAddr;
                } else {
                    totalSize += cur->size;
                }
                prev = cur;
            }

            if (totalSize != size) {
                if (!ignoreError) {
                    HCCL_VM_ERROR(
                        "{} Output data ends too early; the tail is missing, expectedEnd=0x{:x}, "
                        "actualEnd=0x{:x}",
                        MakeErrorCodeText(ErrorCode::SEMANTIC_GAP), startAddr + size, startAddr + totalSize);
                }
                return HCCL_E_PARA;
            }
            return HCCL_SUCCESS;
        }

        BufferSemantic BuildDstSemanticFromSrc(const SliceOpPairV3& pair, const BufferSemantic& srcSemantic)
        {
            uint64_t srcStartAddr = 0;
            uint64_t srcEndAddr = 0;
            if (!GetSourceIntersection(pair, srcSemantic, srcStartAddr, srcEndAddr)) {
                return BufferSemantic(0, 0);
            }

            // 使用源切片内的非负偏移映射到目标，避免绝对地址之差超出 int64_t 范围。
            const uint64_t dstStartAddr = GetSliceAddress(pair.dst) + (srcStartAddr - GetSliceAddress(pair.src));
            const uint64_t dstEndAddr = GetSliceAddress(pair.dst) + (srcEndAddr - GetSliceAddress(pair.src));

            BufferSemantic dstSemantic(
                dstStartAddr, dstEndAddr - dstStartAddr, srcSemantic.isReduce, srcSemantic.reduceType,
                ShiftSourceAddresses(srcSemantic.srcBufs, srcStartAddr - srcSemantic.startAddr));
            dstSemantic.reduceOrderExpr = srcSemantic.reduceOrderExpr;
            return dstSemantic;
        }

        void ApplyOverrideSemantics(
            const SliceOpPairV3& pair, SemanticState& state, const std::vector<const BufferSemantic*>& srcSemantics)
        {
            InternalBufferSemanticMap& dstSemantics = GetBufferSemanticMap(
                state, pair.dstDeviceId, pair.dst.memType, pair.dst.MemoryBegin(), pair.position, pair.index,
                pair.useBatchIndex);
            const uint64_t dstStartAddr = GetSliceAddress(pair.dst);
            const uint64_t dstEndAddr = dstStartAddr + pair.dst.len;
            SplitBufferSemantic(dstSemantics, dstStartAddr);
            SplitBufferSemantic(dstSemantics, dstEndAddr);
            EraseBufferSemanticsInRange(dstSemantics, dstStartAddr, dstEndAddr);

            for (const BufferSemantic* semantic : srcSemantics) {
                if (semantic == nullptr) {
                    continue;
                }
                InsertBufferSemantic(dstSemantics, BuildDstSemanticFromSrc(pair, *semantic));
            }
            MergeAdjacentBufferSemantics(dstSemantics, dstStartAddr, dstEndAddr);
        }

        HcclResult AddReduceSourcesToDstSegments(
            const SliceOpPairV3& pair, SemanticState& state, const BufferSemantic& srcSemantic,
            const std::vector<BufferSemantic*>& affectedDstSemantics, uint64_t srcStartAddr)
        {
            uint64_t sourceAddressDelta = srcStartAddr - srcSemantic.startAddr;
            for (BufferSemantic* dstSemantic : affectedDstSemantics) {
                if (dstSemantic == nullptr) {
                    return HCCL_E_PTR;
                }
                if (dstSemantic->srcBufs.size() == 1) {
                    if (dstSemantic->isReduce) {
                        HCCL_VM_ERROR(
                            "{} Target output range is already marked as a reduce result before "
                            "enough source data is merged, dataMapping={}",
                            MakeErrorCodeText(ErrorCode::SEMANTIC_REDUCE_ERROR), pair.Describe());
                        return HCCL_E_PARA;
                    }
                    dstSemantic->isReduce = true;
                    dstSemantic->reduceType = pair.reduceType;
                }
                if (srcSemantic.srcBufs.size() > 1 && dstSemantic->reduceType != srcSemantic.reduceType) {
                    HCCL_VM_ERROR(
                        "{} Reduce result type is inconsistent while merging one source data "
                        "range, dataMapping={}",
                        MakeErrorCodeText(ErrorCode::SEMANTIC_REDUCE_ERROR), pair.Describe());
                    return HCCL_E_PARA;
                }

                const std::set<SrcBufDes> shiftedSrcBufs
                    = ShiftSourceAddresses(srcSemantic.srcBufs, sourceAddressDelta);
                for (const auto& srcBuf : shiftedSrcBufs) {
                    const size_t before = dstSemantic->srcBufs.size();
                    dstSemantic->srcBufs.insert(srcBuf);
                    if (before == dstSemantic->srcBufs.size()) {
                        HCCL_VM_ERROR(
                            "{} The same source data is added twice to one output range during "
                            "reduce, dataMapping={}, sourceOffsetInThisRange=0x{:x}, outputRange=[0x{:x},0x{:x})",
                            MakeErrorCodeText(ErrorCode::SEMANTIC_REDUCE_ERROR), pair.Describe(), sourceAddressDelta,
                            dstSemantic->startAddr, GetSegmentEndAddr(*dstSemantic));
                        return HCCL_E_PARA;
                    }
                }
                if (state.orderEnabled) {
                    dstSemantic->reduceOrderExpr = pair.knownReduceOrder ?
                                                       state.orderArena.Reduce(
                                                           pair.reduceType, pair.dataType, dstSemantic->reduceOrderExpr,
                                                           srcSemantic.reduceOrderExpr) :
                                                       INVALID_REDUCE_ORDER_EXPR;
                }
                sourceAddressDelta += dstSemantic->size;
            }
            return HCCL_SUCCESS;
        }

        HcclResult
        ApplyReduceSemantic(const SliceOpPairV3& pair, SemanticState& state, const BufferSemantic& srcSemantic)
        {
            uint64_t srcStartAddr = 0;
            uint64_t srcEndAddr = 0;
            if (!GetSourceIntersection(pair, srcSemantic, srcStartAddr, srcEndAddr)) {
                return HCCL_SUCCESS;
            }
            // 使用源切片内的非负偏移映射到目标，避免绝对地址之差超出 int64_t 范围。
            const uint64_t dstStartAddr = GetSliceAddress(pair.dst) + (srcStartAddr - GetSliceAddress(pair.src));
            const uint64_t dstEndAddr = GetSliceAddress(pair.dst) + (srcEndAddr - GetSliceAddress(pair.src));

            InternalBufferSemanticMap& dstSemantics = GetBufferSemanticMap(
                state, pair.dstDeviceId, pair.dst.memType, pair.dst.MemoryBegin(), pair.position, pair.index,
                pair.useBatchIndex);
            SplitBufferSemantic(dstSemantics, dstStartAddr);
            SplitBufferSemantic(dstSemantics, dstEndAddr);

            std::vector<BufferSemantic*> affectedDstSemantics;
            CollectOverlappingBufferSemantics(dstSemantics, dstStartAddr, dstEndAddr, affectedDstSemantics);
            std::vector<const BufferSemantic*> affectedViews;
            affectedViews.reserve(affectedDstSemantics.size());
            for (const BufferSemantic* semantic : affectedDstSemantics) {
                affectedViews.push_back(semantic);
            }
            HcclResult ret = CheckBufSemantics(affectedViews, dstStartAddr, dstEndAddr - dstStartAddr, false);
            if (ret != HCCL_SUCCESS) {
                HCCL_VM_ERROR(
                    "{} Target output range is only partially filled before reduce continues, "
                    "dataMapping={}, outputRange=[0x{:x},0x{:x}), pieceCount={}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_REDUCE_ERROR), pair.Describe(), dstStartAddr, dstEndAddr,
                    affectedDstSemantics.size());
                return ret;
            }

            ret = AddReduceSourcesToDstSegments(pair, state, srcSemantic, affectedDstSemantics, srcStartAddr);
            if (ret != HCCL_SUCCESS) {
                HCCL_VM_ERROR(
                    "{} Failed to merge one source data range into the target output range during "
                    "reduce, dataMapping={}, srcRange=[0x{:x},0x{:x}), dstRange=[0x{:x},0x{:x}), ret={}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_REDUCE_ERROR), pair.Describe(), srcStartAddr, srcEndAddr,
                    dstStartAddr, dstEndAddr, static_cast<int32_t>(ret));
                return ret;
            }

            return HCCL_SUCCESS;
        }

        HcclResult ApplySrcSemanticsToDst(
            const SliceOpPairV3& pair, SemanticState& state, const std::vector<const BufferSemantic*>& srcSemantics)
        {
            if (pair.op == SliceOpV3::OVERRIDE) {
                ApplyOverrideSemantics(pair, state, srcSemantics);
                return HCCL_SUCCESS;
            }

            for (const BufferSemantic* semantic : srcSemantics) {
                if (semantic == nullptr) {
                    return HCCL_E_PTR;
                }
                HcclResult ret = ApplyReduceSemantic(pair, state, *semantic);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
            }
            return HCCL_SUCCESS;
        }

        HcclResult LoadSliceOpPairSrcSemantics(
            const SliceOpPairV3& pair, SemanticState& state, std::vector<BufferSemantic>& srcSemantics)
        {
            srcSemantics.clear();
            if (pair.src.len == 0) {
                return HCCL_SUCCESS;
            }
            if (!IsValidMemorySlice(pair.src) || !IsValidMemorySlice(pair.dst)) {
                HCCL_VM_ERROR(
                    "{} One source-to-target mapping uses an invalid memory slice, "
                    "dataMapping={}",
                    MakeErrorCodeText(ErrorCode::SINGLETASK_SLICE_INVALID), pair.Describe());
                return HCCL_E_PARA;
            }
            if (!IsSupportedSemanticMemType(pair.src.memType) || !IsSupportedSemanticMemType(pair.dst.memType)) {
                HCCL_VM_ERROR(
                    "{} This data mapping uses a memory type that semantic check does not support, "
                    "dataMapping={}",
                    MakeErrorCodeText(ErrorCode::SINGLETASK_SLICE_INVALID), pair.Describe());
                return HCCL_E_NOT_SUPPORT;
            }

            const InternalBufferSemanticMap& srcSemanticsMap = GetBufferSemanticMap(
                state, pair.srcDeviceId, pair.src.memType, pair.src.MemoryBegin(), pair.position, pair.index,
                pair.useBatchIndex);
            const uint64_t srcStartAddr = GetSliceAddress(pair.src);
            std::vector<const BufferSemantic*> srcViews;
            CollectOverlappingBufferSemantics(srcSemanticsMap, srcStartAddr, srcStartAddr + pair.src.len, srcViews);

            HcclResult ret = CheckBufSemantics(srcViews, srcStartAddr, pair.src.len, true);
            if (ret != HCCL_SUCCESS) {
                if (pair.op == SliceOpV3::REDUCE) {
                    HCCL_VM_ERROR(
                        "{} Source data needed by this reduce is missing, dataMapping={}",
                        MakeErrorCodeText(ErrorCode::SEMANTIC_REDUCE_ERROR), pair.Describe());
                    return ret;
                }
                if (state.orderEnabled) {
                    state.orderIncomplete = true;
                }
                // 老语义实现对 override 的不完整源语义是放行但告警；
                // 这里保持同样语义，至少把“并非完整 memcpy 语义”的风险显式打出来。
                HCCL_VM_WARN(
                    "{} Source data needed by this overwrite is missing, dataMapping={}",
                    MakeErrorCodeText(ErrorCode::SEMANTIC_SIMULATE_FAILED), pair.Describe());
            }

            srcSemantics.reserve(srcViews.size());
            for (const BufferSemantic* semantic : srcViews) {
                if (semantic == nullptr) {
                    return HCCL_E_PTR;
                }
                srcSemantics.push_back(*semantic);
            }
            return HCCL_SUCCESS;
        }

        HcclResult ApplyPreparedSliceOpPair(const PreparedSliceOpPairV3& prepared, SemanticState& state)
        {
            if (prepared.pair.src.len == 0) {
                return HCCL_SUCCESS;
            }
            std::vector<const BufferSemantic*> srcViews;
            srcViews.reserve(prepared.srcSemantics.size());
            for (const auto& semantic : prepared.srcSemantics) {
                srcViews.push_back(&semantic);
            }
            HcclResult ret = ApplySrcSemanticsToDst(prepared.pair, state, srcViews);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
            return HCCL_SUCCESS;
        }

        void GetSliceOpPairs(const TaskNode* node, std::vector<SliceOpPairV3>& pairs)
        {
            pairs.clear();
            if (node == nullptr || !IsDataTask(node)) {
                return;
            }

            // 这里把不同 task 类型统一翻译成“src -> dst 的语义操作对”，
            // 后续模拟阶段不再关心原始 task 类型细节。
            if (node->GetType() == TaskType::TRANS_MEM) {
                const auto* task = dynamic_cast<const TaskTransMem*>(node);
                if (task == nullptr) {
                    return;
                }
                pairs.push_back(SliceOpPairV3{
                    task->GetSrc().deviceId, task->GetDst().deviceId, task->GetSrc(), task->GetDst(),
                    node->GetPosition(), SliceOpV3::OVERRIDE, HCCL_REDUCE_RESERVED, 0});
                return;
            }

            if (node->GetType() == TaskType::REDUCE) {
                const auto* task = dynamic_cast<const TaskReduce*>(node);
                if (task == nullptr) {
                    return;
                }
                const auto& srcs = task->GetSrcs();
                for (size_t index = 0; index < srcs.size(); ++index) {
                    const auto& src = srcs[index];
                    SliceOpPairV3 pair;
                    pair.srcDeviceId = src.deviceId;
                    pair.dstDeviceId = task->GetDst().deviceId;
                    pair.src = src;
                    pair.dst = task->GetDst();
                    pair.position = node->GetPosition();
                    pair.op = SliceOpV3::REDUCE;
                    pair.reduceType = static_cast<HcclReduceOp>(task->GetReduceOp());
                    pair.dataType = static_cast<HcclDataType>(task->GetDataType());
                    pair.knownReduceOrder = srcs.size() == 1;
                    pairs.push_back(std::move(pair));
                }
                return;
            }

            if (node->GetType() == TaskType::BATCH_TRANS_MEM) {
                const auto* task = dynamic_cast<const TaskBatchTransMem*>(node);
                if (task == nullptr) {
                    return;
                }
                const auto& srcs = task->GetSrcs();
                const auto& dsts = task->GetDsts();
                const size_t count = std::min(srcs.size(), dsts.size());
                for (size_t index = 0; index < count; ++index) {
                    pairs.push_back(SliceOpPairV3{
                        srcs[index].deviceId, dsts[index].deviceId, srcs[index], dsts[index], node->GetPosition(),
                        SliceOpV3::OVERRIDE, HCCL_REDUCE_RESERVED, static_cast<uint64_t>(index), true});
                }
                return;
            }

            if (node->GetType() == TaskType::BATCH_REDUCE) {
                const auto* task = dynamic_cast<const TaskBatchReduce*>(node);
                if (task == nullptr) {
                    return;
                }
                const auto& srcGroups = task->GetSrcs();
                const auto& dsts = task->GetDsts();
                const size_t count = std::min(srcGroups.size(), dsts.size());
                for (size_t index = 0; index < count; ++index) {
                    for (const auto& src : srcGroups[index]) {
                        SliceOpPairV3 pair;
                        pair.srcDeviceId = src.deviceId;
                        pair.dstDeviceId = dsts[index].deviceId;
                        pair.src = src;
                        pair.dst = dsts[index];
                        pair.position = node->GetPosition();
                        pair.op = SliceOpV3::REDUCE;
                        pair.reduceType = static_cast<HcclReduceOp>(task->GetReduceOp());
                        pair.dataType = static_cast<HcclDataType>(task->GetDataType());
                        pair.knownReduceOrder = srcGroups[index].size() == 1;
                        pair.index = static_cast<uint64_t>(index);
                        pair.useBatchIndex = true;
                        pairs.push_back(std::move(pair));
                    }
                }
            }
        }

        HcclResult SimulateTask(const TaskNode* node, SemanticState& state)
        {
            if (!IsDataTask(node)) {
                return HCCL_SUCCESS;
            }
            std::vector<SliceOpPairV3> pairs;
            GetSliceOpPairs(node, pairs);
            if (state.orderEnabled
                && (node->GetType() == TaskType::BATCH_REDUCE || node->GetType() == TaskType::BATCH_TRANS_MEM)) {
                // 不同 batch 组之间没有编码先后关系。若多个组覆盖同一目标区间，按 vector 顺序
                // 套用就是臆造一个执行顺序，必须拒绝。按地址空间（MS/UB 各自的语义表）分桶比较，
                // 避免把两块不相干的内存误判为重叠。
                struct DestinationRange {
                    const InternalBufferSemanticMap* bucket;
                    uint64_t begin;
                    uint64_t end;
                    uint64_t index;
                };
                std::vector<DestinationRange> destinations;
                for (const auto& pair : pairs) {
                    if (pair.dst.len == 0) {
                        continue;
                    }
                    const auto& bucket = GetBufferSemanticMap(
                        state, pair.dstDeviceId, pair.dst.memType, pair.dst.MemoryBegin(), pair.position, pair.index,
                        pair.useBatchIndex);
                    destinations.push_back({&bucket, pair.dst.MemoryBegin(), pair.dst.MemoryEnd(), pair.index});
                }
                std::sort(destinations.begin(), destinations.end(), [](const auto& a, const auto& b) {
                    if (a.bucket != b.bucket) {
                        return std::less<const InternalBufferSemanticMap*>{}(a.bucket, b.bucket);
                    }
                    return a.begin < b.begin;
                });
                const InternalBufferSemanticMap* bucket = nullptr;
                uint64_t end = 0;
                uint64_t group = 0;
                for (const auto& dst : destinations) {
                    if (dst.bucket == bucket && dst.begin < end && dst.index != group) {
                        state.orderReport.status = ReduceOrderStatus::INCOMPLETE;
                        state.orderReport.detail
                            = "batch groups have overlapping destinations without an encoded order";
                        return HCCL_E_PARA;
                    }
                    if (dst.bucket != bucket || dst.end > end) {
                        end = dst.end;
                        group = dst.index;
                    }
                    bucket = dst.bucket;
                }
            }
            // 先 load 再 apply，保证一个 batch task 内所有 pair 看到的是同一时刻的源语义快照。
            std::vector<PreparedSliceOpPairV3> preparedPairs;
            preparedPairs.reserve(pairs.size());
            for (const auto& pair : pairs) {
                PreparedSliceOpPairV3 prepared;
                prepared.pair = pair;
                HcclResult ret = LoadSliceOpPairSrcSemantics(pair, state, prepared.srcSemantics);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                preparedPairs.push_back(std::move(prepared));
            }
            for (const auto& prepared : preparedPairs) {
                HcclResult ret = ApplyPreparedSliceOpPair(prepared, state);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
            }
            return HCCL_SUCCESS;
        }

        HcclResult CheckFinalOutput(const SemanticState& state, const std::vector<DeviceId>& rankToDevice)
        {
            switch (state.param.cmdType) {
                case HCCL_CMD_ALLREDUCE:
                    return TaskCheckAllReduceSemantics(
                        state.physicalSemantics, state.bufferLayouts, state.dataSize, state.param.reduceType,
                        rankToDevice);
                case HCCL_CMD_ALLGATHER:
                    return TaskCheckAllGatherSemantics(
                        state.physicalSemantics, state.bufferLayouts, state.dataSize, rankToDevice);
                case HCCL_CMD_ALLGATHER_V: {
                    return TaskCheckAllGatherVSemantics(
                        state.physicalSemantics, state.bufferLayouts, state.param.vDataDes, rankToDevice);
                }
                case HCCL_CMD_REDUCE_SCATTER:
                    return TaskCheckReduceScatterSemantics(
                        state.physicalSemantics, state.bufferLayouts, state.dataSize, state.param.reduceType,
                        rankToDevice);
                case HCCL_CMD_REDUCE_SCATTER_V: {
                    return TaskCheckReduceScatterVSemantics(
                        state.physicalSemantics, state.bufferLayouts, state.param.reduceType, state.param.vDataDes,
                        rankToDevice);
                }
                case HCCL_CMD_ALLTOALL:
                case HCCL_CMD_ALLTOALLV:
                case HCCL_CMD_ALLTOALLVC: {
                    return TaskCheckAll2AllSemantics(
                        state.physicalSemantics, state.bufferLayouts, state.param.all2AllDataDes, rankToDevice);
                }
                case HCCL_CMD_SEND:
                case HCCL_CMD_RECEIVE:
                    return TaskCheckSendRecvGroupSemantics(
                        state.physicalSemantics, state.bufferLayouts, state.dataSize, state.param.sendRecvPairs,
                        rankToDevice);
                case HCCL_CMD_BROADCAST:
                    return TaskCheckBroadcastSemantics(
                        state.physicalSemantics, state.bufferLayouts, state.dataSize, rankToDevice[state.param.root],
                        rankToDevice);
                case HCCL_CMD_REDUCE:
                    return TaskCheckReduceSemantics(
                        state.physicalSemantics, state.bufferLayouts, state.dataSize, state.param.reduceType,
                        rankToDevice[state.param.root], rankToDevice);
                case HCCL_CMD_SCATTER:
                    return TaskCheckScatterSemantics(
                        state.physicalSemantics, state.bufferLayouts, state.dataSize, rankToDevice[state.param.root],
                        rankToDevice);
                case HCCL_CMD_BATCH_SEND_RECV:
                    return TaskCheckBatchSendRecvRingSemantics(
                        state.physicalSemantics, state.bufferLayouts, rankToDevice.size(), state.dataSize,
                        rankToDevice);
                default:
                    HCCL_VM_WARN(
                        "{} Final output validation does not support this collective type yet, "
                        "collectiveType={}, rankCount={}, dataType={}, elementCount={}, reduceType={}",
                        MakeErrorCodeText(ErrorCode::CHECKER_RUNTIME_ERROR), HcclCmdTypeToString(state.param.cmdType),
                        state.param.rankSize, HcclDataTypeToString(state.param.dataType), state.param.dataCount,
                        DumpReduceOpToString(state.param.reduceType));
                    return HCCL_E_NOT_SUPPORT;
            }
        }

        HcclResult CheckFinalReduceOrder(SemanticState& state, const std::vector<DeviceId>& rankToDevice)
        {
            if (!state.orderEnabled) {
                return HCCL_SUCCESS;
            }
            auto& report = state.orderReport;
            report.expressionNodes = state.orderArena.NodeCount();
            const auto incomplete = [&](const std::string& reason) {
                report.status = ReduceOrderStatus::INCOMPLETE;
                report.detail = reason;
                return HCCL_E_PARA;
            };
            if (state.orderIncomplete) {
                return incomplete("a copy source was incomplete during semantic simulation");
            }
            // 与 TaskCheckAllReduceSemantics / TaskCheckReduceScatterSemantics 同一套取法：
            // 遍历物理语义表，从布局取结果窗口，用 totalSize 累计覆盖并要求区间连续。
            const uint64_t elementBytes = CHECK_SIZE_TABLE[state.param.dataType];
            std::vector<ReduceOrderOutputSpan> spans;
            for (const auto& [deviceId, mem] : state.physicalSemantics) {
                const RankId deviceRank = FindRankIndexByDeviceId(rankToDevice, deviceId);
                if (deviceRank == INVALID_RANK_ID) {
                    return incomplete("device is not found in the rank-to-device mapping");
                }
                uint64_t outputBase = 0;
                uint64_t outputBufferSize = 0;
                GetSemanticBufferExtent(
                    state.bufferLayouts, deviceId, BufferType::OUTPUT, outputBase, outputBufferSize);
                uint64_t totalSize = 0;
                for (const auto& [eleAddr, original] : mem) {
                    BufferSemantic ele = original;
                    if (!ClipSemanticToBuffer(ele, outputBase, outputBufferSize)) {
                        continue;
                    }
                    if (ele.startAddr >= outputBase + state.dataSize) {
                        break;
                    }
                    if (ele.startAddr != outputBase + totalSize) {
                        return incomplete("output interval coverage is invalid");
                    }
                    if (ele.srcBufs.size() != state.param.rankSize) {
                        return incomplete("output source count is invalid");
                    }
                    const uint64_t inputOffset
                        = totalSize
                          + (state.param.cmdType == HCCL_CMD_REDUCE_SCATTER ? deviceRank * state.dataSize : 0);
                    std::set<DeviceId> sourceDevices;
                    for (const auto& source : ele.srcBufs) {
                        if (source.bufType != BufferType::INPUT || !sourceDevices.insert(source.deviceId).second
                            || source.srcAddr
                                   != GetSemanticBufferBase(state.bufferLayouts, source.deviceId, BufferType::INPUT)
                                          + inputOffset) {
                            return incomplete(
                                "output provenance does not match the complete input device/coordinate set");
                        }
                    }
                    spans.push_back({deviceId, totalSize / elementBytes, ele.size / elementBytes, ele.reduceOrderExpr});
                    totalSize += ele.size;
                }
                if (totalSize != state.dataSize) {
                    return incomplete("output provenance coverage is incomplete");
                }
            }
            const HcclResult ret = CheckOutputReduceOrder(
                rankToDevice, state.param.dataType, state.param.reduceType, state.orderArena, spans, report);
            HCCL_VM_INFO(
                "ReduceOrder finished, collective={}, status={}, outputBlocks={}, exprNodes={}, "
                "comparedBlocks={}, detail={}",
                HcclCmdTypeToString(state.param.cmdType), DescribeReduceOrderStatus(report.status), report.outputSpans,
                report.expressionNodes, report.comparedSpans, report.detail);
            return ret;
        }

        void FillStats(const SemanticState& state, size_t handledNodeCount, SemanticCheckStats* stats)
        {
            if (stats == nullptr) {
                return;
            }
            stats->handledNodeCount = handledNodeCount;
            stats->rankCount = state.param.rankSize;
            stats->normalSemanticCount = 0;
            stats->normalSemanticBytes = 0;
            for (const auto& deviceEntry : state.physicalSemantics) {
                stats->normalSemanticCount += deviceEntry.second.size();
                for (const auto& semanticEntry : deviceEntry.second) {
                    stats->normalSemanticBytes += semanticEntry.second.size;
                }
            }
            stats->msBucketCount = state.ms.size();
            stats->reduceOrder = state.orderReport;
            stats->reduceOrder.expressionNodes = state.orderArena.NodeCount();
        }
    } // namespace

    HcclResult CheckSemantics(const TaskNode* start, SemanticCheckStats* stats)
    {
        if (start == nullptr) {
            HCCL_VM_ERROR(
                "{} Semantic check cannot start because the main start node is missing, "
                "mainStartNode=null",
                MakeErrorCodeText(ErrorCode::CHECKER_RUNTIME_ERROR));
            return HCCL_E_PTR;
        }

        SemanticState state;
        const OperatorId operatorId = start->GetOperatorId();
        const CheckerParam operatorParam = StorageManager::GetInstance().GetCheckerParam(operatorId);
        const std::map<DeviceId, RankId> deviceToRank
            = StorageManager::GetInstance().GetDeviceRankMappings(operatorParam.commId);
        HcclResult ret = InitState(state, operatorId, start->GetPosition(), deviceToRank);
        if (ret != HCCL_SUCCESS) {
            FillStats(state, 0, stats);
            return ret;
        }
        if (state.orderEnabled) {
            state.orderReport.status = ReduceOrderStatus::INCOMPLETE;
            state.orderReport.detail = "semantic simulation has not completed";
        }

        std::vector<DeviceId> rankToDevice(state.param.rankSize, INVALID_DEVICE_ID);
        for (const auto& [deviceId, rankId] : deviceToRank) {
            if (rankId < state.param.rankSize) {
                rankToDevice[rankId] = deviceId;
            }
        }

        // 第一步先用 BFS 把从 start 可达的节点全部收集出来，并初始化每个节点的“剩余父节点数”。
        // 这里不直接在 BFS 顺序上做语义模拟，因为 DAG 中子节点可能比某些父节点更早被扫描到。
        std::unordered_map<NodeId, const TaskNode*> nodes;
        std::unordered_map<NodeId, uint32_t> pendingParents;
        std::queue<const TaskNode*> walk;
        std::set<NodeId> visited;
        walk.push(start);
        visited.insert(start->GetNodeId());
        while (!walk.empty()) {
            const TaskNode* node = walk.front();
            walk.pop();
            nodes[node->GetNodeId()] = node;
            pendingParents[node->GetNodeId()] = static_cast<uint32_t>(node->GetParents().size());
            for (const TaskNode* child : node->GetChildren()) {
                if (child == nullptr) {
                    continue;
                }
                if (visited.insert(child->GetNodeId()).second) {
                    walk.push(child);
                }
            }
        }

        // 第二步做一次基于入度的拓扑模拟：只有 pendingParents 归零的节点才允许执行。
        // 这样可以直接复用当前 DAG 上已有的 parent/child 约束，不需要再单独识别 loop 区间。
        std::queue<const TaskNode*> ready;
        for (const auto& entry : nodes) {
            if (pendingParents[entry.first] == 0) {
                ready.push(entry.second);
            }
        }

        size_t handled = 0;
        while (!ready.empty()) {
            const TaskNode* node = ready.front();
            ready.pop();
            ++handled;
            ret = SimulateTask(node, state);
            if (ret != HCCL_SUCCESS) {
                FillStats(state, handled, stats);
                return ret;
            }
            for (const TaskNode* child : node->GetChildren()) {
                if (child == nullptr) {
                    HCCL_VM_ERROR(
                        "{} Graph structure is broken because one child node is null, parent={}",
                        MakeErrorCodeText(ErrorCode::CHECKER_RUNTIME_ERROR), node->Describe());
                    return HCCL_E_PTR;
                }
                auto iter = pendingParents.find(child->GetNodeId());
                if (iter == pendingParents.end() || iter->second == 0) {
                    continue;
                }
                --iter->second;
                if (iter->second == 0) {
                    ready.push(child);
                }
            }
        }

        if (handled != nodes.size()) {
            const TaskNode* firstRemainingNode = nullptr;
            for (const auto& entry : pendingParents) {
                if (entry.second != 0) {
                    auto nodeIter = nodes.find(entry.first);
                    if (nodeIter != nodes.end()) {
                        firstRemainingNode = nodeIter->second;
                        break;
                    }
                }
            }
            HCCL_VM_ERROR(
                "{} Output simulation stopped because some tasks still have unresolved "
                "dependencies, handledNodeCount={}, totalNodeCount={}, firstRemainingNode={}",
                MakeErrorCodeText(ErrorCode::CHECKER_RUNTIME_ERROR), handled, nodes.size(),
                firstRemainingNode == nullptr ? std::string("node=null") : firstRemainingNode->Describe());
            return HCCL_E_INTERNAL;
        }

        ret = CheckFinalOutput(state, rankToDevice);
        if (ret != HCCL_SUCCESS && ret != HCCL_E_NOT_SUPPORT) {
            HCCL_VM_ERROR("Final semantic check failed, ret={}", static_cast<uint32_t>(ret));
            if (state.orderEnabled) {
                state.orderReport.status = ReduceOrderStatus::INCOMPLETE;
                state.orderReport.detail = "final output provenance check failed";
            }
            FillStats(state, handled, stats);
            return ret;
        }

        if (ret == HCCL_E_NOT_SUPPORT && state.orderEnabled) {
            state.orderReport.status = ReduceOrderStatus::INCOMPLETE;
            state.orderReport.detail = "final provenance check is unsupported";
            FillStats(state, handled, stats);
            return ret;
        }
        ret = CheckFinalReduceOrder(state, rankToDevice);
        FillStats(state, handled, stats);
        if (ret != HCCL_SUCCESS) {
            HCCL_VM_ERROR(
                "ReduceOrder check failed, status={}, detail={}", DescribeReduceOrderStatus(state.orderReport.status),
                state.orderReport.detail);
        }
        return ret;
    }
} // namespace TaskGraphGeneratorV3
} // namespace HcclSim
