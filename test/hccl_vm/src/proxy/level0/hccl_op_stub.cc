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
#define HCCL_VM_MODULE "OP_STUB"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <execinfo.h>
#include <functional>
#include <string>
#include <unistd.h>
#include <vector>

#include "acl/acl_rt.h"
#include "dtype_common.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"
#include "sim_common_defs.h"
#include "hccl_proxy_common.h"
#include "sim_log.h"
#include "store_sim_memory_manager.h"
#include "operation_data/operation_data_ops.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "runtime_state/db_sim_runner_common.h"
#include "storage/table_access.h"
#include "sim_common_api.h"

using sim::runtime::g_cur_comm_key;
using sim::runtime::g_cur_device_key;

extern "C" uint8_t GetOpExpansionMode();

namespace {
// SetupCommunicator 在覆写 g_cur_comm_key 前暂存的“进入本算子前”值；
// 紧随其后构造的 ScopedOperatorCommunicator 取走作为弹栈恢复目标。
// （guard 构造点在 SetupCommunicator 之后，无法自行快照覆写前的值。）
thread_local uint64_t g_commKeyBeforeSetup = 0;

class ScopedOperatorCommunicator {
public:
    ScopedOperatorCommunicator() : savedCommKey_(g_commKeyBeforeSetup), savedOpContext_(sim::operation::SaveOpContext())
    {}

    ~ScopedOperatorCommunicator()
    {
        // 弹栈（算子结束）：
        // 1) 若本作用域确实落库了新算子（CurrOpDetailId 相对入栈快照发生变化），
        //    为其打上完成序号 endSeq —— checker 据此按“先出栈在前”成图；
        // 2) 恢复进入算子前的通信域键与算子上下文。嵌套调用（如算子执行期
        //    LaunchWarmupVOperator 再入 alltoallv）依赖该恢复：内层退出后外层
        //    的 g_cur_comm_key/CurrOpDetailId 回到外层值，外层后续任务/算法名/
        //    主流修正仍归属外层自己的 opDetail（此前内层析构把 g_cur_comm_key
        //    硬置 0、CurrOpDetailId 停留在内层 id，外层任务全部错挂 opDetailId=0）。
        const uint32_t ownOpDetailId = sim::operation::CurrOpDetailId();
        if (ownOpDetailId != savedOpContext_.opDetailId) {
            (void)sim::operation::AssignOpEndSeq(ownOpDetailId);
        }
        // 通信域只在当前 HCCL 算子同步执行期间有效；离开算子后不允许 TLS 状态泄漏。
        // 恢复而非清零：最外层算子恢复到进入前的 0（无泄漏），嵌套内层恢复到外层值。
        g_cur_comm_key = savedCommKey_;
        sim::operation::RestoreOpContext(savedOpContext_);
    }

    ScopedOperatorCommunicator(const ScopedOperatorCommunicator&) = delete;
    ScopedOperatorCommunicator& operator=(const ScopedOperatorCommunicator&) = delete;

private:
    uint64_t savedCommKey_;
    sim::operation::OpContextSnapshot savedOpContext_;
};

static HcclResult SetupCommunicator(HcclComm comm, uint64_t deviceKey, uint32_t& curRank, uint32_t& rankSize)
{
    if (comm == nullptr || deviceKey == 0) {
        HCCL_VM_ERROR("SetupCommunicator requires a valid comm and current device");
        return HcclResult::HCCL_E_PARA;
    }

    uint64_t communicatorId = 0;
    const char* level = std::getenv("HCCL_VM_LEVEL");
    const bool isLevel1 = level != nullptr && std::strcmp(level, "1") == 0;

    if (isLevel1) {
        // level1 将本进程 Communicator 表行 ID 直接转换成伪 HcclComm。
        // 因此这里直接还原该 ID，不能使用 level2 的真实句柄映射表。
        communicatorId = reinterpret_cast<uint64_t>(comm);
        if (communicatorId == 0) {
            HCCL_VM_ERROR("level1 HcclComm does not contain a communicator ID");
            return HcclResult::HCCL_E_INTERNAL;
        }
    } else {
        // level2 收到的是 HCCL 创建的真实 HcclComm，其地址是不可直接解释的句柄，
        // 不能当作数据库 ID，必须通过创建阶段登记的映射解析当前 rank 的表行。
        if (!SimFindHcclCommMember(comm, &communicatorId)) {
            HCCL_VM_ERROR("level2 HcclComm={:p} has no communicator binding", comm);
            return HcclResult::HCCL_E_INTERNAL;
        }
    }

    const auto member = sim::runtime::Db::GetById<sim::runtime::Communicator>(communicatorId);
    if (!member.ok()) {
        HCCL_VM_ERROR("communicator ID={} resolved from HcclComm={:p} is not present", communicatorId, comm);
        return HcclResult::HCCL_E_INTERNAL;
    }
    rankSize = member.value->rank_size;
    curRank = member.value->rank_id;
    if (rankSize == 0 || curRank >= rankSize || member.value->device_id != deviceKey) {
        HCCL_VM_ERROR("communicator ID={} does not match current deviceId={}", communicatorId, deviceKey);
        return HcclResult::HCCL_E_INTERNAL;
    }

    // 任务记录使用本进程的 Communicator 表行 ID，不能使用 HcclComm 句柄或跨 rank 的 comm_id 字符串。
    // 覆写前暂存“进入本算子前”的 comm key，供紧随构造的 ScopedOperatorCommunicator 弹栈恢复：
    // 最外层恢复到 0（无 TLS 泄漏），嵌套内层恢复到外层的 comm key（不清零、不踩坏外层）。
    g_commKeyBeforeSetup = g_cur_comm_key;
    g_cur_comm_key = member.value->id;
    return HcclResult::HCCL_SUCCESS;
}

// 图模式算子入口只拿到 group 名, 没有 HcclComm 句柄。
// 从本地已建模的 Communicator 表中按 comm_id(即通信域名称)反查, 再映射回本进程的 HcclComm 句柄。
static bool ResolveCommByGroup(const char* group, HcclComm& comm)
{
    if (group == nullptr) {
        return false;
    }
    const auto comms = sim::runtime::Db::GetByPred<sim::runtime::Communicator>(
        HcclSim::Storage::Eq(&sim::runtime::Communicator::comm_id, std::string(group)));
    if (comms.ok() && comms.value.has_value()) {
        for (const auto& record : *comms.value) {
            // GetByPred 可能命中其他 rank 的同名记录, 只有本进程已绑定的表行 id 才能反查出句柄。
            if (SimFindHcclCommHandle(record.id, &comm)) {
                return true;
            }
        }
    }
    return false;
}

static std::vector<uint8_t> BuildBatchSendRecvRingExtInfo(uint32_t itemNum)
{
    std::vector<uint8_t> extInfo(sizeof(itemNum));
    std::memcpy(extInfo.data(), &itemNum, sizeof(itemNum));
    return extInfo;
}

static std::vector<uint8_t>
BuildVOpExtInfo(uint32_t rankSize, uint64_t localCount, const uint64_t* counts, const uint64_t* displs)
{
    const size_t arrayByteSize = static_cast<size_t>(rankSize) * sizeof(uint64_t);
    std::vector<uint8_t> extInfo(sizeof(localCount) + arrayByteSize * 2);
    size_t offset = 0;
    std::memcpy(extInfo.data() + offset, &localCount, sizeof(localCount));
    offset += sizeof(localCount);
    std::memcpy(extInfo.data() + offset, counts, arrayByteSize);
    offset += arrayByteSize;
    std::memcpy(extInfo.data() + offset, displs, arrayByteSize);
    return extInfo;
}

static std::vector<uint8_t> BuildMatrixExtInfo(const uint64_t* matrix, uint32_t matrixElementCount)
{
    const size_t matrixByteSize = static_cast<size_t>(matrixElementCount) * sizeof(uint64_t);
    std::vector<uint8_t> extInfo(sizeof(matrixElementCount) + matrixByteSize);
    std::memcpy(extInfo.data(), &matrixElementCount, sizeof(matrixElementCount));
    std::memcpy(extInfo.data() + sizeof(matrixElementCount), matrix, matrixByteSize);
    return extInfo;
}

static std::vector<uint8_t>
BuildOpDetailsV1(uint64_t count, uint16_t dataType, uint32_t reduceOpType, uint32_t reduceOp, HcclCMDType opType)
{
    OpDetails details;
    std::memset(&details, 0, sizeof(details));
    details.opType = static_cast<uint16_t>(opType);
    details.dataType = dataType;
    details.reduceType = reduceOp;

    details.opV1.count = count;
    details.opV1.dataType = dataType;
    details.opV1.reduceOpType = reduceOpType;

    std::vector<uint8_t> blob(sizeof(OpDetails));
    std::memcpy(blob.data(), &details, sizeof(OpDetails));
    return blob;
}

static std::vector<uint8_t> BuildOpDetailsV2(
    uint64_t sendCount, uint16_t sendDataType, uint64_t recvCount, uint16_t recvDataType, uint16_t extInfo,
    HcclCMDType opType, uint16_t dataType, uint32_t reduceOp)
{
    OpDetails details;
    std::memset(&details, 0, sizeof(details));
    details.opType = static_cast<uint16_t>(opType);
    details.dataType = dataType;
    details.reduceType = reduceOp;

    details.opV2.sendCount = sendCount;
    details.opV2.sendDataType = sendDataType;
    details.opV2.recvCount = recvCount;
    details.opV2.recvDataType = recvDataType;
    details.opV2.extInfo = extInfo;

    std::vector<uint8_t> blob(sizeof(OpDetails));
    std::memcpy(blob.data(), &details, sizeof(OpDetails));
    return blob;
}

static uint32_t GetDeviceType()
{
    DevType devType;
    auto ret = hrtGetDeviceType(devType);
    if (ret != HcclResult::HCCL_SUCCESS) {
        return ret;
    }
    return static_cast<uint32_t>(devType);
}

// 登记卡死检测计数槽（业务侧 slot 布局：head@0 / tail@4 / addOne@8 / reserved@12，共 16B）。
// 算子会对该内存做 4 字节 int32 SUM 归约并产生 task，未登记时 checker 无法解析地址。
// 尽力而为：接口未导出、特性关闭（地址为 0）或地址转换失败均静默跳过，不算算子记录失败。
static void RecordOpCounterBuffer(HcclComm comm)
{
    using GetCounterAddrFunc = HcclResult (*)(HcclComm, uint64_t*, uint64_t*, uint64_t*);
    const auto getCounterAddr = reinterpret_cast<GetCounterAddrFunc>(
        sim::DlsymRealWithFallback("HcclStuckDetectGetCounterAddr", "libhccl.so"));
    if (getCounterAddr == nullptr) {
        HCCL_VM_DEBUG("HcclStuckDetectGetCounterAddr unavailable, skip counter buffer recording.");
        return;
    }

    uint64_t headDevAddr = 0;
    uint64_t tailDevAddr = 0;
    uint64_t addOneDevAddr = 0;
    const HcclResult ret = getCounterAddr(comm, &headDevAddr, &tailDevAddr, &addOneDevAddr);
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_DEBUG(
            "HcclStuckDetectGetCounterAddr failed, ret={}, skip counter buffer recording.", static_cast<int>(ret));
        return;
    }
    if (headDevAddr == 0 || addOneDevAddr == 0) {
        // 卡死检测未开启：业务约定未开启时全部返回 0 地址。
        HCCL_VM_DEBUG("stuck detect counter disabled, skip counter buffer recording.");
        return;
    }

    const uint64_t headAddr = sim::TryGetVirPtrByDevPtr(headDevAddr);
    const uint64_t tailAddr = sim::TryGetVirPtrByDevPtr(tailDevAddr);
    const uint64_t addOneAddr = sim::TryGetVirPtrByDevPtr(addOneDevAddr);
    if (headAddr == 0 || addOneAddr == 0 || (tailDevAddr != 0 && tailAddr == 0)) {
        HCCL_VM_WARN(
            "failed to translate counter devAddr to virAddr, head=0x{:x}, tail=0x{:x}, addOne=0x{:x}.", headDevAddr,
            tailDevAddr, addOneDevAddr);
        return;
    }

    constexpr uint64_t counterElemSize = sizeof(int32_t);
    constexpr uint64_t counterSlotSize = 16;
    const uint64_t lowAddr = std::min(headAddr, addOneAddr);
    const uint64_t highAddr = std::max(headAddr, addOneAddr);
    uint64_t baseAddr = (tailAddr != 0) ? std::min(lowAddr, tailAddr) : lowAddr;
    uint64_t endAddr = (tailAddr != 0) ? std::max(highAddr, tailAddr) : highAddr;
    endAddr += counterElemSize;
    const uint64_t size = std::max(counterSlotSize, endAddr - baseAddr);

    // 同一 slot 在业务侧（device 进程映射址）的起始地址：CCU 指令把它作为立即数烧进指令流，
    // 绕开 SQE args 的 dev→vir 翻译链，checker 需要这一份才能识别/换算计数指令地址。
    const uint64_t devLowAddr = std::min(headDevAddr, addOneDevAddr);
    const uint64_t devBaseAddr = (tailDevAddr != 0) ? std::min(devLowAddr, tailDevAddr) : devLowAddr;

    if (sim::operation::UpdateOpMemCounterBuffer(baseAddr, size, devBaseAddr) != 0) {
        HCCL_VM_WARN("failed to update current opMem counter buffer, addr=0x{:x}, size={}.", baseAddr, size);
        return;
    }
    HCCL_VM_INFO("opMem counter buffer recorded, addr=0x{:x}, devAddr=0x{:x}, size={}.", baseAddr, devBaseAddr, size);
}

int RecordOpDbInfo(
    HcclCMDType cmdType, uint32_t rankId, uint64_t streamId, const void* inputBuf, uint64_t inputSize,
    const void* outputBuf, uint64_t outputSize, const std::vector<uint8_t>& details, uint32_t root, uint32_t rankSize,
    uint32_t srcRank, uint32_t dstRank, const std::vector<uint8_t>& extInfo, HcclComm comm)
{
    (void)cmdType;
    // 本次调用内复用相同 rank 的设备身份；不同 rank 仍按原顺序查询并传播失败。
    uint32_t memoRanks[4] = {UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX};
    uint32_t memoDevices[4] = {0, 0, 0, 0};
    uint32_t memoCount = 0;
    auto convertRankToDeviceId = [&](uint32_t commRank, uint32_t& deviceId) {
        for (uint32_t m = 0; m < memoCount; ++m) {
            if (memoRanks[m] == commRank) {
                deviceId = memoDevices[m];
                return true;
            }
        }
        sim::runtime::Device device{};
        if (sim::runtime::GetDeviceByCommRank(g_cur_comm_key, commRank, device) != ACL_SUCCESS) {
            HCCL_VM_ERROR("failed to map communicator rankId={} to deviceId, commId={}", commRank, g_cur_comm_key);
            return false;
        }
        deviceId = static_cast<uint32_t>(device.id);
        memoRanks[memoCount] = commRank;
        memoDevices[memoCount] = deviceId;
        ++memoCount;
        return true;
    };

    uint32_t deviceRankId = 0;
    uint32_t deviceSrcRank = 0;
    uint32_t deviceDstRank = 0;
    uint32_t deviceRoot = 0;
    if (!convertRankToDeviceId(rankId, deviceRankId) || !convertRankToDeviceId(srcRank, deviceSrcRank)
        || !convertRankToDeviceId(dstRank, deviceDstRank) || !convertRankToDeviceId(root, deviceRoot)) {
        return -1;
    }

    sim::operation::OpDetailTab opDetailTab{};
    opDetailTab.id = 0;
    opDetailTab.pid = getpid();
    opDetailTab.deviceId = deviceRankId;
    opDetailTab.rankId = rankId;
    opDetailTab.commId = g_cur_comm_key;
    opDetailTab.opIter = 0;
    opDetailTab.syncIter = 0;
    opDetailTab.streamId = streamId;
    opDetailTab.root = root;
    opDetailTab.opExpansionMode = GetOpExpansionMode();
    // UB 多 channel(jetty)数随算子落库: 轻量性能仿真插件据此折算端口带宽利用率
    // (env HCCL_UB_MULTI_CHANNEL_NUM 未设置/非法时为默认值 1, 即不使能多 jetty)
    opDetailTab.ubMultiChannelNum = HcclSim::GetUbMultiChannelNum();
    opDetailTab.devType = GetDeviceType();
    opDetailTab.rankSize = rankSize;
    opDetailTab.srcRank = srcRank;
    opDetailTab.dstRank = dstRank;
    opDetailTab.opDetail = details;
    opDetailTab.opExtInfo = extInfo;

    sim::operation::OpMemInfoTab opMemInfoTab{};
    opMemInfoTab.id = 0;
    // 零长度缓冲区不参与内存布局，保留零初始化地址，避免转换非空的区间末尾指针。
    opMemInfoTab.inputSize = inputSize;
    if (inputBuf != nullptr && inputSize > 0) {
        opMemInfoTab.inputAddr = sim::GetVirPtrByDevPtr(reinterpret_cast<uint64_t>(inputBuf));
    }

    opMemInfoTab.outputSize = outputSize;
    if (outputBuf != nullptr && outputSize > 0) {
        opMemInfoTab.outputAddr = sim::GetVirPtrByDevPtr(reinterpret_cast<uint64_t>(outputBuf));
    }

    opMemInfoTab.cclAddr = 0;
    opMemInfoTab.cclSize = 0;

    if (sim::operation::InsertOpDetailAndMem(opDetailTab, opMemInfoTab) != 0) {
        HCCL_VM_ERROR("insert op detail+mem failed");
        return -1;
    }
    if (comm != nullptr) {
        RecordOpCounterBuffer(comm);

        using GetHcclBufferFunc = HcclResult (*)(HcclComm, void**, uint64_t*);
        const auto getHcclBuffer
            = reinterpret_cast<GetHcclBufferFunc>(sim::DlsymRealWithFallback("HcclGetHcclBuffer", "libhccl.so"));
        if (getHcclBuffer == nullptr) {
            return -1;
        }
        void* cclBuffer = nullptr;
        uint64_t cclSize = 0;
        const HcclResult cclRet = getHcclBuffer(comm, &cclBuffer, &cclSize);
        if (cclRet != HcclResult::HCCL_SUCCESS) {
            HCCL_VM_ERROR("HcclGetHcclBuffer failed while recording op, ret={}", static_cast<int>(cclRet));
            return -1;
        }
        if (cclBuffer == nullptr && cclSize == 0) {
            if (rankSize == 1) {
                return 0;
            }
            HCCL_VM_ERROR("invalid empty HCCL CCL buffer for rankSize={}", rankSize);
            return -1;
        }
        const uint64_t cclAddr = sim::GetVirPtrByDevPtr(reinterpret_cast<uint64_t>(cclBuffer));
        if (cclAddr == 0 || cclSize == 0 || sim::operation::UpdateOpMemCclBuffer(cclAddr, cclSize) != 0) {
            HCCL_VM_ERROR("failed to update current opMem CCL buffer, devAddr={:p}, size={}", cclBuffer, cclSize);
            return -1;
        }
    } else {
        HCCL_VM_ERROR("HcclGetHcclBuffer failed, comm is nullptr!");
        return -1;
    }
    return 0;
}

HcclResult RecordBatchSendRecvRing(
    HcclSendRecvItem* sendRecvInfo, uint32_t itemNum, HcclComm comm, aclrtStream stream, uint32_t curRank,
    uint32_t rankSize)
{
    if (sendRecvInfo == nullptr || comm == nullptr || stream == nullptr || rankSize < 2 || curRank >= rankSize
        || itemNum != 2) {
        HCCL_VM_ERROR(
            "invalid HcclBatchSendRecv ring parameters: rank={}, rankSize={}, itemNum={}", curRank, rankSize, itemNum);
        return HcclResult::HCCL_E_PARA;
    }

    const HcclSendRecvItem* sendItem = nullptr;
    const HcclSendRecvItem* recvItem = nullptr;
    for (uint32_t i = 0; i < itemNum; ++i) {
        const auto& item = sendRecvInfo[i];
        if (item.sendRecvType == HCCL_SEND && sendItem == nullptr) {
            sendItem = &item;
        } else if (item.sendRecvType == HCCL_RECV && recvItem == nullptr) {
            recvItem = &item;
        } else {
            HCCL_VM_ERROR("HcclBatchSendRecv ring requires exactly one SEND and one RECV item");
            return HcclResult::HCCL_E_PARA;
        }
    }

    const uint32_t sendPeer = (curRank + 1U) % rankSize;
    const uint32_t recvPeer = (curRank + rankSize - 1U) % rankSize;
    if (sendItem == nullptr || recvItem == nullptr || sendItem->remoteRank != sendPeer
        || recvItem->remoteRank != recvPeer) {
        HCCL_VM_ERROR(
            "HcclBatchSendRecv ring peer mismatch at rank {}: expected sendPeer={}, recvPeer={}", curRank, sendPeer,
            recvPeer);
        return HcclResult::HCCL_E_PARA;
    }
    if (sendItem->count != recvItem->count || sendItem->dataType != recvItem->dataType
        || (sendItem->count != 0 && (sendItem->buf == nullptr || recvItem->buf == nullptr))) {
        HCCL_VM_ERROR("HcclBatchSendRecv ring requires matching SEND/RECV count, data type and valid buffers");
        return HcclResult::HCCL_E_PARA;
    }

    uint32_t typeSize = 0;
    if (sim::GetDataTypeSize(sendItem->dataType, typeSize) != HcclResult::HCCL_SUCCESS) {
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    if (typeSize != 0 && sendItem->count > UINT64_MAX / typeSize) {
        HCCL_VM_ERROR(
            "HcclBatchSendRecv ring buffer size overflows uint64, count={}, typeSize={}", sendItem->count, typeSize);
        return HcclResult::HCCL_E_PARA;
    }
    const uint64_t dataBytes = sendItem->count * typeSize;
    const auto details
        = BuildOpDetailsV1(sendItem->count, sendItem->dataType, 0, 0, HcclCMDType::HCCL_CMD_BATCH_SEND_RECV);
    const auto extInfo = BuildBatchSendRecvRingExtInfo(itemNum);
    if (RecordOpDbInfo(
            HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, curRank, reinterpret_cast<uint64_t>(stream), sendItem->buf,
            dataBytes, recvItem->buf, dataBytes, details, 0, rankSize, recvPeer, sendPeer, extInfo, comm)
        != 0) {
        return HcclResult::HCCL_E_PARA;
    }
    return HcclResult::HCCL_SUCCESS;
}
} // namespace

using namespace HcclSim;

#ifdef __cplusplus
extern "C" {
#endif

uint8_t GetOpExpansionMode()
{
    sim::runtime::SimOpExpansionMode mode = sim::runtime::SimOpExpansionMode::SIM_OP_EXPANSION_MODE_RESERVED;
    const char* expanEnv = std::getenv("HCCL_OP_EXPANSION_MODE");
    if (expanEnv == nullptr) {
        HCCL_VM_INFO("HCCL_OP_EXPANSION_MODE env is not set, use default value: [{}]", static_cast<uint8_t>(mode));
        return static_cast<uint8_t>(mode);
    }

    std::string modeStr(expanEnv);
    if (modeStr == "AI_CPU") {
        mode = sim::runtime::SimOpExpansionMode::SIM_OP_EXPANSION_MODE_AICPU;
    } else if (modeStr == "AIV") {
        mode = sim::runtime::SimOpExpansionMode::SIM_OP_EXPANSION_MODE_AIV;
    } else if (modeStr == "CCU_SCHED" || modeStr == "CCU_MS") {
        mode = sim::runtime::SimOpExpansionMode::SIM_OP_EXPANSION_MODE_CCU;
    }

    HCCL_VM_INFO("HCCL_OP_EXPANSION_MODE = [{}]", expanEnv);
    return static_cast<uint8_t>(mode);
}

HcclResult CheckDataAndReduceOpType(HcclDataType dataType, HcclReduceOp op)
{
    if (dataType == HCCL_DATA_TYPE_INT64 || dataType == HCCL_DATA_TYPE_UINT64 || dataType == HCCL_DATA_TYPE_FP64
        || op == HcclReduceOp::HCCL_REDUCE_PROD) {
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
HcclSend(void* sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank, HcclComm comm, aclrtStream stream)
{
    uint32_t curRank = 0;
    uint32_t rankSize = 0;
    const HcclResult setupRet = SetupCommunicator(comm, g_cur_device_key, curRank, rankSize);
    if (setupRet != HcclResult::HCCL_SUCCESS) {
        g_cur_comm_key = 0;
        return setupRet;
    }
    ScopedOperatorCommunicator scopedCommunicator;
    if (stream == nullptr || rankSize == 0 || curRank >= rankSize || destRank >= rankSize || destRank == curRank
        || (count != 0 && sendBuf == nullptr)) {
        HCCL_VM_ERROR("invalid HcclSend parameters: rank={}, rankSize={}, destRank={}", curRank, rankSize, destRank);
        return HcclResult::HCCL_E_PARA;
    }
    uint32_t typeSize = 0;
    if (sim::GetDataTypeSize(dataType, typeSize) != HcclResult::HCCL_SUCCESS) {
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    const uint64_t dataSize = count * typeSize;
    const auto details = BuildOpDetailsV1(count, dataType, 0, 0, HcclCMDType::HCCL_CMD_SEND);
    if (RecordOpDbInfo(
            HcclCMDType::HCCL_CMD_SEND, curRank, reinterpret_cast<uint64_t>(stream), sendBuf, dataSize, nullptr, 0,
            details, 0, rankSize, curRank, destRank, {}, comm)
        != 0) {
        return HcclResult::HCCL_E_PARA;
    }
    using Func = HcclResult (*)(void*, uint64_t, HcclDataType, uint32_t, HcclComm, aclrtStream);
    const auto func = reinterpret_cast<Func>(sim::DlsymRealWithFallback(__func__, "libhccl.so"));

    if (func == nullptr) {
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    const HcclResult ret = func(sendBuf, count, dataType, destRank, comm, stream);
    return ret;
}

HcclResult
HcclRecv(void* recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank, HcclComm comm, aclrtStream stream)
{
    uint32_t curRank = 0;
    uint32_t rankSize = 0;
    const HcclResult setupRet = SetupCommunicator(comm, g_cur_device_key, curRank, rankSize);
    if (setupRet != HcclResult::HCCL_SUCCESS) {
        g_cur_comm_key = 0;
        return setupRet;
    }
    ScopedOperatorCommunicator scopedCommunicator;
    if (stream == nullptr || rankSize == 0 || curRank >= rankSize || srcRank >= rankSize || srcRank == curRank
        || (count != 0 && recvBuf == nullptr)) {
        HCCL_VM_ERROR("invalid HcclRecv parameters: rank={}, rankSize={}, srcRank={}", curRank, rankSize, srcRank);
        return HcclResult::HCCL_E_PARA;
    }
    uint32_t typeSize = 0;
    if (sim::GetDataTypeSize(dataType, typeSize) != HcclResult::HCCL_SUCCESS) {
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    const uint64_t dataSize = count * typeSize;
    const auto details = BuildOpDetailsV1(count, dataType, 0, 0, HcclCMDType::HCCL_CMD_RECEIVE);
    if (RecordOpDbInfo(
            HcclCMDType::HCCL_CMD_RECEIVE, curRank, reinterpret_cast<uint64_t>(stream), nullptr, 0, recvBuf, dataSize,
            details, 0, rankSize, srcRank, curRank, {}, comm)
        != 0) {
        return HcclResult::HCCL_E_PARA;
    }
    using Func = HcclResult (*)(void*, uint64_t, HcclDataType, uint32_t, HcclComm, aclrtStream);
    const auto func = reinterpret_cast<Func>(sim::DlsymRealWithFallback(__func__, "libhccl.so"));

    if (func == nullptr) {
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    const HcclResult ret = func(recvBuf, count, dataType, srcRank, comm, stream);
    return ret;
}

HcclResult HcclBatchSendRecv(HcclSendRecvItem* sendRecvInfo, uint32_t itemNum, HcclComm comm, aclrtStream stream)
{
    uint32_t curRank = 0;
    uint32_t rankSize = 0;
    const HcclResult setupRet = SetupCommunicator(comm, g_cur_device_key, curRank, rankSize);
    if (setupRet != HcclResult::HCCL_SUCCESS) {
        g_cur_comm_key = 0;
        return setupRet;
    }
    ScopedOperatorCommunicator scopedCommunicator;
    const HcclResult ret = RecordBatchSendRecvRing(sendRecvInfo, itemNum, comm, stream, curRank, rankSize);
    if (ret != HcclResult::HCCL_SUCCESS) {
        return ret;
    }
    using Func = HcclResult (*)(HcclSendRecvItem*, uint32_t, HcclComm, aclrtStream);
    const auto func = reinterpret_cast<Func>(sim::DlsymRealWithFallback(__func__, "libhccl.so"));

    if (func == nullptr) {
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    const HcclResult realRet = func(sendRecvInfo, itemNum, comm, stream);
    return realRet;
}

HcclResult HcclAlltoAll(
    const void* sendBuf, uint64_t sendCount, HcclDataType sendType, const void* recvBuf, uint64_t recvCount,
    HcclDataType recvType, HcclComm comm, aclrtStream stream)
{
    HCCL_VM_INFO("HcclAlltoAll called with parameters:");
    HCCL_VM_INFO("sendBuf = {:p}", sendBuf);
    HCCL_VM_INFO("recvBuf = {:p}", recvBuf);
    HCCL_VM_INFO("sendCount = {}", sendCount);
    HCCL_VM_INFO("recvCount = {}", recvCount);
    HCCL_VM_INFO("sendType = {}", GetDataTypeStr(sendType));
    HCCL_VM_INFO("recvType = {}", GetDataTypeStr(recvType));
    HCCL_VM_INFO("comm = {:p}", comm);
    HCCL_VM_INFO("stream = {:p}", stream);

    uint32_t curRank = 0;
    uint32_t rankSize = 0;
    const HcclResult setupRet = SetupCommunicator(comm, g_cur_device_key, curRank, rankSize);
    if (setupRet != HcclResult::HCCL_SUCCESS) {
        g_cur_comm_key = 0;
        return setupRet;
    }
    ScopedOperatorCommunicator scopedCommunicator;
    // 注册input、output buffer
    uint32_t inDataSize = 0;
    if (sim::GetDataTypeSize(sendType, inDataSize) != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "HCCL_VM not support data type {} for HcclAlltoAll send type calc size", GetDataTypeStr(sendType));
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    uint32_t outDataSize = 0;
    if (sim::GetDataTypeSize(recvType, outDataSize) != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "HCCL_VM not support data type {} for HcclAlltoAll recv type calc size", GetDataTypeStr(recvType));
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    uint64_t inputSize = static_cast<uint64_t>(inDataSize) * sendCount * rankSize;
    uint64_t outputSize = static_cast<uint64_t>(outDataSize) * recvCount * rankSize;

    // Use BuildOpDetailsV2 for AlltoAll
    auto alltoallDetails
        = BuildOpDetailsV2(sendCount, sendType, recvCount, recvType, 0, HcclCMDType::HCCL_CMD_ALLTOALL, sendType, 0);
    std::vector<uint8_t> alltoallExtInfo;
    uint32_t alltoallCount = rankSize * rankSize;
    alltoallExtInfo.resize(sizeof(uint32_t) + alltoallCount * sizeof(uint64_t));
    std::memcpy(alltoallExtInfo.data(), &alltoallCount, sizeof(uint32_t));
    for (uint32_t i = 0; i < alltoallCount; i++) {
        std::memcpy(alltoallExtInfo.data() + sizeof(uint32_t) + i * sizeof(uint64_t), &sendCount, sizeof(uint64_t));
    }
    if (RecordOpDbInfo(
            HcclCMDType::HCCL_CMD_ALLTOALL, curRank, reinterpret_cast<uint64_t>(stream), sendBuf, inputSize, recvBuf,
            outputSize, alltoallDetails, 0, rankSize, curRank, curRank, alltoallExtInfo, comm)
        != 0) {
        HCCL_VM_ERROR("record op db info failed");
        return HcclResult::HCCL_E_PARA;
    }
    HCCL_VM_INFO("get op info: allRank= {}, curRank= {}.", rankSize, curRank);

    using HcclAlltoAllFunc = HcclResult (*)(
        const void*, uint64_t, HcclDataType, const void*, uint64_t, HcclDataType, HcclComm, aclrtStream);
    const auto hcclAlltoAllFunc
        = reinterpret_cast<HcclAlltoAllFunc>(sim::DlsymRealWithFallback(__func__, "libhccl.so"));

    if (hcclAlltoAllFunc != nullptr) {
        const HcclResult ret
            = hcclAlltoAllFunc(sendBuf, sendCount, sendType, recvBuf, recvCount, recvType, comm, stream);
        return ret;
    } else {
        HCCL_VM_ERROR("dlsym failed");
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
}

HcclResult HcclAlltoAllVC(
    const void* sendBuf, const void* sendCountMatrix, HcclDataType sendType, const void* recvBuf, HcclDataType recvType,
    HcclComm comm, aclrtStream stream)
{
    HCCL_VM_INFO("HcclAlltoAllVC called with parameters:");
    HCCL_VM_INFO("sendBuf = {:p}", sendBuf);
    HCCL_VM_INFO("sendCountMatrix = {:p}", sendCountMatrix);
    HCCL_VM_INFO("recvBuf = {:p}", recvBuf);
    HCCL_VM_INFO("sendType = {}", GetDataTypeStr(sendType));
    HCCL_VM_INFO("recvType = {}", GetDataTypeStr(recvType));
    HCCL_VM_INFO("comm = {:p}", comm);
    HCCL_VM_INFO("stream = {:p}", stream);

    uint32_t curRank = 0;
    uint32_t rankSize = 0;
    const HcclResult setupRet = SetupCommunicator(comm, g_cur_device_key, curRank, rankSize);
    if (setupRet != HcclResult::HCCL_SUCCESS) {
        g_cur_comm_key = 0;
        return setupRet;
    }
    ScopedOperatorCommunicator scopedCommunicator;
    if (rankSize == 0 || curRank >= rankSize || sendCountMatrix == nullptr) {
        HCCL_VM_ERROR("invalid AlltoAllVC rank information: rankId = {}, rankSize = {}", curRank, rankSize);
        return HcclResult::HCCL_E_PARA;
    }

    const auto* matrix = static_cast<const uint64_t*>(sendCountMatrix);
    uint64_t inputCount = 0;
    uint64_t outputCount = 0;
    const size_t rowOffset = static_cast<size_t>(curRank) * static_cast<size_t>(rankSize);
    for (uint32_t peerRank = 0; peerRank < rankSize; ++peerRank) {
        inputCount += matrix[rowOffset + peerRank];
        outputCount += matrix[static_cast<size_t>(peerRank) * rankSize + curRank];
    }

    uint32_t sendDataSize = 0;
    if (sim::GetDataTypeSize(sendType, sendDataSize) != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "HCCL_VM not support data type {} for HcclAlltoAllVC send type calc size", GetDataTypeStr(sendType));
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    uint32_t recvDataSize = 0;
    if (sim::GetDataTypeSize(recvType, recvDataSize) != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "HCCL_VM not support data type {} for HcclAlltoAllVC recv type calc size", GetDataTypeStr(recvType));
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }

    const uint64_t inputSize = inputCount * sendDataSize;
    const uint64_t outputSize = outputCount * recvDataSize;

    const auto alltoallvcDetails = BuildOpDetailsV2(
        inputCount, sendType, outputCount, recvType, 0, HcclCMDType::HCCL_CMD_ALLTOALLVC, sendType, 0);
    const uint32_t matrixElementCount = rankSize * rankSize;
    const auto alltoallvcExtInfo = BuildMatrixExtInfo(matrix, matrixElementCount);
    if (RecordOpDbInfo(
            HcclCMDType::HCCL_CMD_ALLTOALLVC, curRank, reinterpret_cast<uint64_t>(stream), sendBuf, inputSize, recvBuf,
            outputSize, alltoallvcDetails, 0, rankSize, curRank, curRank, alltoallvcExtInfo, comm)
        != 0) {
        HCCL_VM_ERROR("record op db info failed");
        return HcclResult::HCCL_E_PARA;
    }

    HCCL_VM_INFO("get op info: allRank= {}, curRank= {}.", rankSize, curRank);

    using HcclAlltoAllVCFunc
        = HcclResult (*)(const void*, const void*, HcclDataType, const void*, HcclDataType, HcclComm, aclrtStream);
    const auto hcclAlltoAllVCFunc
        = reinterpret_cast<HcclAlltoAllVCFunc>(sim::DlsymRealWithFallback(__func__, "libhccl.so"));

    if (hcclAlltoAllVCFunc != nullptr) {
        const HcclResult ret = hcclAlltoAllVCFunc(sendBuf, sendCountMatrix, sendType, recvBuf, recvType, comm, stream);
        return ret;
    }

    HCCL_VM_ERROR("dlsym failed");
    return HcclResult::HCCL_E_NOT_SUPPORT;
}

HcclResult HcclAlltoAllV(
    const void* sendBuf, const void* sendCounts, const void* sdispls, HcclDataType sendType, const void* recvBuf,
    const void* recvCounts, const void* rdispls, HcclDataType recvType, HcclComm comm, aclrtStream stream)
{
    HCCL_VM_INFO("HcclAlltoAllV called with parameters:");
    HCCL_VM_INFO("sendBuf = {:p}", sendBuf);
    HCCL_VM_INFO("recvBuf = {:p}", recvBuf);
    HCCL_VM_INFO("sendType = {}", GetDataTypeStr(sendType));
    HCCL_VM_INFO("recvType = {}", GetDataTypeStr(recvType));
    HCCL_VM_INFO("comm = {:p}", comm);
    HCCL_VM_INFO("stream = {:p}", stream);

    if (sendCounts == nullptr) {
        HCCL_VM_ERROR("HcclAlltoAllV: sendCounts is nullptr");
        g_cur_comm_key = 0;
        return HcclResult::HCCL_E_PTR;
    }
    if (recvCounts == nullptr) {
        HCCL_VM_ERROR("HcclAlltoAllV: recvCounts is nullptr");
        g_cur_comm_key = 0;
        return HcclResult::HCCL_E_PTR;
    }

    uint32_t curRank = 0;
    uint32_t rankSize = 0;
    const HcclResult setupRet = SetupCommunicator(comm, g_cur_device_key, curRank, rankSize);
    if (setupRet != HcclResult::HCCL_SUCCESS) {
        g_cur_comm_key = 0;
        return setupRet;
    }
    ScopedOperatorCommunicator scopedCommunicator;
    // 注册input、output buffer
    uint32_t inDataSize = 0;
    if (sim::GetDataTypeSize(sendType, inDataSize) != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "HCCL_VM not support data type {} for HcclAlltoAllV send type calc size", GetDataTypeStr(sendType));
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    uint32_t outDataSize = 0;
    if (sim::GetDataTypeSize(recvType, outDataSize) != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "HCCL_VM not support data type {} for HcclAlltoAllV recv type calc size", GetDataTypeStr(recvType));
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    uint64_t inCountTotal = 0;
    uint64_t outCountTotal = 0;
    for (uint32_t rank = 0; rank < rankSize; rank++) {
        inCountTotal += ((uint64_t*)sendCounts)[rank];
        outCountTotal += ((uint64_t*)recvCounts)[rank];
    }
    uint64_t inputSize = static_cast<uint64_t>(inDataSize) * inCountTotal;
    uint64_t outputSize = static_cast<uint64_t>(outDataSize) * outCountTotal;

    auto alltoallvDetails = BuildOpDetailsV2(
        static_cast<uint64_t>(inCountTotal), sendType, static_cast<uint64_t>(outCountTotal), recvType, 0,
        HcclCMDType::HCCL_CMD_ALLTOALLV, sendType, 0);

    std::vector<uint64_t> sendCountMatrix(rankSize * rankSize, 0);
    for (uint32_t j = 0; j < rankSize; j++) {
        sendCountMatrix[curRank * rankSize + j] = ((uint64_t*)sendCounts)[j];
    }

    for (uint32_t j = 0; j < rankSize; j++) {
        sendCountMatrix[j * rankSize + curRank] = ((uint64_t*)recvCounts)[j];
    }

    std::vector<uint8_t> alltoallvExtInfo;
    uint32_t alltoallvCount = rankSize * rankSize;
    alltoallvExtInfo.resize(sizeof(uint32_t) + alltoallvCount * sizeof(uint64_t));
    std::memcpy(alltoallvExtInfo.data(), &alltoallvCount, sizeof(uint32_t));
    std::memcpy(alltoallvExtInfo.data() + sizeof(uint32_t), sendCountMatrix.data(), alltoallvCount * sizeof(uint64_t));
    if (RecordOpDbInfo(
            HcclCMDType::HCCL_CMD_ALLTOALLV, curRank, reinterpret_cast<uint64_t>(stream), sendBuf, inputSize, recvBuf,
            outputSize, alltoallvDetails, 0, rankSize, curRank, curRank, alltoallvExtInfo, comm)
        != 0) {
        HCCL_VM_ERROR("record op db info failed");
        return HcclResult::HCCL_E_PARA;
    }

    HCCL_VM_INFO("get op info: allRank= {}, curRank= {}.", rankSize, curRank);

    using HcclAlltoAllVFunc = HcclResult (*)(
        const void*, const void*, const void*, HcclDataType, const void*, const void*, const void*, HcclDataType,
        HcclComm, aclrtStream);
    const auto hcclAlltoAllVFunc
        = reinterpret_cast<HcclAlltoAllVFunc>(sim::DlsymRealWithFallback(__func__, "libhccl.so"));

    if (hcclAlltoAllVFunc != nullptr) {
        const HcclResult ret = hcclAlltoAllVFunc(
            sendBuf, sendCounts, sdispls, sendType, recvBuf, recvCounts, rdispls, recvType, comm, stream);
        return ret;
    } else {
        HCCL_VM_ERROR("dlsym failed");
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
}

HcclResult HcclAllGather(
    void* sendBuf, void* recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    HCCL_VM_INFO("HcclAllGather called with parameters:");
    HCCL_VM_INFO("sendBuf = {:p}", sendBuf);
    HCCL_VM_INFO("recvBuf = {:p}", recvBuf);
    HCCL_VM_INFO("sendCount = {}", sendCount);
    HCCL_VM_INFO("dataType = {}", GetDataTypeStr(dataType));
    HCCL_VM_INFO("comm = {:p}", comm);
    HCCL_VM_INFO("stream = {:p}", stream);

    uint32_t curRank = 0;
    uint32_t rankSize = 0;
    const HcclResult setupRet = SetupCommunicator(comm, g_cur_device_key, curRank, rankSize);
    if (setupRet != HcclResult::HCCL_SUCCESS) {
        g_cur_comm_key = 0;
        return setupRet;
    }
    ScopedOperatorCommunicator scopedCommunicator;
    // 注册input、output buffer
    uint32_t dataSize = 0;
    if (sim::GetDataTypeSize(dataType, dataSize) != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR("HCCL_VM not support data type {} for HcclAllGather calc size", GetDataTypeStr(dataType));
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    uint64_t inputSize = static_cast<uint64_t>(dataSize) * sendCount;
    uint64_t outputSize = static_cast<uint64_t>(dataSize) * sendCount * rankSize;

    // Use BuildOpDetailsV1 for AllGather
    auto allGatherDetails = BuildOpDetailsV1(sendCount, dataType, 0, 0, HcclCMDType::HCCL_CMD_ALLGATHER);
    if (RecordOpDbInfo(
            HcclCMDType::HCCL_CMD_ALLGATHER, curRank, reinterpret_cast<uint64_t>(stream), sendBuf, inputSize, recvBuf,
            outputSize, allGatherDetails, 0, rankSize, curRank, curRank, {}, comm)
        != 0) {
        HCCL_VM_ERROR("record op db info failed");
        return HcclResult::HCCL_E_PARA;
    }

    HCCL_VM_INFO("get op info: allRank= {}, curRank= {}.", rankSize, curRank);

    using HcclAllGatherFunc = HcclResult (*)(void*, void*, uint64_t, HcclDataType, HcclComm, aclrtStream);
    const auto hcclAllGatherFunc
        = reinterpret_cast<HcclAllGatherFunc>(sim::DlsymRealWithFallback(__func__, "libhccl.so"));

    if (hcclAllGatherFunc != nullptr) {
        const HcclResult ret = hcclAllGatherFunc(sendBuf, recvBuf, sendCount, dataType, comm, stream);
        return ret;
    } else {
        HCCL_VM_ERROR("dlsym failed");
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
}

HcclResult HcclAllGatherV(
    void* sendBuf, uint64_t sendCount, void* recvBuf, const void* recvCounts, const void* recvDispls,
    HcclDataType dataType, HcclComm comm, aclrtStream stream)
{
    HCCL_VM_INFO("HcclAllGatherV called with parameters:");
    HCCL_VM_INFO("sendBuf = {:p}", sendBuf);
    HCCL_VM_INFO("sendCount = {}", sendCount);
    HCCL_VM_INFO("recvBuf = {:p}", recvBuf);
    HCCL_VM_INFO("recvCounts = {:p}", recvCounts);
    HCCL_VM_INFO("recvDispls = {:p}", recvDispls);
    HCCL_VM_INFO("dataType = {}", GetDataTypeStr(dataType));
    HCCL_VM_INFO("comm = {:p}", comm);
    HCCL_VM_INFO("stream = {:p}", stream);

    uint32_t curRank = 0;
    uint32_t rankSize = 0;
    const HcclResult setupRet = SetupCommunicator(comm, g_cur_device_key, curRank, rankSize);
    if (setupRet != HcclResult::HCCL_SUCCESS) {
        g_cur_comm_key = 0;
        return setupRet;
    }
    ScopedOperatorCommunicator scopedCommunicator;
    if (rankSize == 0 || curRank >= rankSize) {
        HCCL_VM_ERROR("invalid AllGatherV rank information: rankId = {}, rankSize = {}", curRank, rankSize);
        return HcclResult::HCCL_E_PARA;
    }

    if (recvCounts == nullptr || recvDispls == nullptr) {
        HCCL_VM_ERROR("invalid AllGatherV count or displacement array");
        return HcclResult::HCCL_E_PARA;
    }

    const auto* recvCountValues = static_cast<const uint64_t*>(recvCounts);
    const auto* recvDisplValues = static_cast<const uint64_t*>(recvDispls);
    uint64_t totalRecvCount = 0;
    for (uint32_t rank = 0; rank < rankSize; ++rank) {
        totalRecvCount += recvCountValues[rank];
    }

    uint32_t dataSize = 0;
    if (sim::GetDataTypeSize(dataType, dataSize) != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR("HCCL_VM not support data type {} for HcclAllGatherV calc size", GetDataTypeStr(dataType));
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    const uint64_t inputSize = sendCount * dataSize;
    const uint64_t outputSize = totalRecvCount * dataSize;

    const auto allGatherVExtInfo = BuildVOpExtInfo(rankSize, sendCount, recvCountValues, recvDisplValues);
    const auto allGatherVDetails = BuildOpDetailsV1(sendCount, dataType, 0, 0, HcclCMDType::HCCL_CMD_ALLGATHER_V);
    if (RecordOpDbInfo(
            HcclCMDType::HCCL_CMD_ALLGATHER_V, curRank, reinterpret_cast<uint64_t>(stream), sendBuf, inputSize, recvBuf,
            outputSize, allGatherVDetails, 0, rankSize, curRank, curRank, allGatherVExtInfo, comm)
        != 0) {
        HCCL_VM_ERROR("record op db info failed");
        return HcclResult::HCCL_E_PARA;
    }

    HCCL_VM_INFO("get op info: allRank= {}, curRank= {}.", rankSize, curRank);

    using HcclAllGatherVFunc
        = HcclResult (*)(void*, uint64_t, void*, const void*, const void*, HcclDataType, HcclComm, aclrtStream);
    const auto hcclAllGatherVFunc
        = reinterpret_cast<HcclAllGatherVFunc>(sim::DlsymRealWithFallback(__func__, "libhccl.so"));

    if (hcclAllGatherVFunc != nullptr) {
        const HcclResult ret
            = hcclAllGatherVFunc(sendBuf, sendCount, recvBuf, recvCounts, recvDispls, dataType, comm, stream);
        return ret;
    }

    HCCL_VM_ERROR("dlsym failed");
    return HcclResult::HCCL_E_NOT_SUPPORT;
}

HcclResult
HcclBroadcast(void* buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)
{
    HCCL_VM_INFO("HcclBroadcast called with parameters:");
    HCCL_VM_INFO("buf = {:p}", buf);
    HCCL_VM_INFO("count = {}", count);
    HCCL_VM_INFO("dataType = {}", GetDataTypeStr(dataType));
    HCCL_VM_INFO("root = {}", root);
    HCCL_VM_INFO("comm = {:p}", comm);
    HCCL_VM_INFO("stream = {:p}", stream);

    uint32_t curRank = 0;
    uint32_t rankSize = 0;
    const HcclResult setupRet = SetupCommunicator(comm, g_cur_device_key, curRank, rankSize);
    if (setupRet != HcclResult::HCCL_SUCCESS) {
        g_cur_comm_key = 0;
        return setupRet;
    }
    ScopedOperatorCommunicator scopedCommunicator;
    // 注册input、output buffer
    uint32_t dataSize = 0;
    if (sim::GetDataTypeSize(dataType, dataSize) != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR("HCCL_VM not support data type {} for HcclBroadcast calc size", GetDataTypeStr(dataType));
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    uint64_t size = static_cast<uint64_t>(dataSize) * count;

    // Use BuildOpDetailsV1 for Broadcast
    auto broadcastDetails = BuildOpDetailsV1(count, dataType, 0, 0, HcclCMDType::HCCL_CMD_BROADCAST);
    if (RecordOpDbInfo(
            HcclCMDType::HCCL_CMD_BROADCAST, curRank, reinterpret_cast<uint64_t>(stream), buf, size, buf, size,
            broadcastDetails, root, rankSize, curRank, curRank, {}, comm)
        != 0) {
        HCCL_VM_ERROR("record op db info failed");
        return HcclResult::HCCL_E_PARA;
    }

    HCCL_VM_INFO("get op info: allRank= {}, curRank= {}.", rankSize, curRank);

    using HcclBroadcastFunc = HcclResult (*)(void*, uint64_t, HcclDataType, uint32_t, HcclComm, aclrtStream);
    const auto hcclBroadcastFunc
        = reinterpret_cast<HcclBroadcastFunc>(sim::DlsymRealWithFallback(__func__, "libhccl.so"));

    if (hcclBroadcastFunc != nullptr) {
        const HcclResult ret = hcclBroadcastFunc(buf, count, dataType, root, comm, stream);
        return ret;
    } else {
        HCCL_VM_ERROR("dlsym failed");
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
}

HcclResult HcclAllReduce(
    void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, HcclComm comm,
    aclrtStream stream)
{
    HCCL_VM_INFO("HcclAllReduce called with parameters:");
    HCCL_VM_INFO("sendBuf = {:p}", sendBuf);
    HCCL_VM_INFO("recvBuf = {:p}", recvBuf);
    HCCL_VM_INFO("count = {}", count);
    HCCL_VM_INFO("dataType = {}", GetDataTypeStr(dataType));
    HCCL_VM_INFO("op = {}", GetReduceOpStr(op));
    HCCL_VM_INFO("comm = {:p}", comm);
    HCCL_VM_INFO("stream = {:p}", stream);

    uint32_t curRank = 0;
    uint32_t rankSize = 0;
    const HcclResult setupRet = SetupCommunicator(comm, g_cur_device_key, curRank, rankSize);
    if (setupRet != HcclResult::HCCL_SUCCESS) {
        g_cur_comm_key = 0;
        return setupRet;
    }
    ScopedOperatorCommunicator scopedCommunicator;
    // 注册input、output buffer
    uint32_t dataSize = 0;
    if (sim::GetDataTypeSize(dataType, dataSize) != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR("HCCL_VM not support data type {} for HcclAllReduce calc size", GetDataTypeStr(dataType));
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    uint64_t size = static_cast<uint64_t>(dataSize) * count;
    // Use BuildOpDetailsV1 for AllReduce
    auto allReduceDetails = BuildOpDetailsV1(
        count, dataType, static_cast<uint32_t>(op), static_cast<uint32_t>(op), HcclCMDType::HCCL_CMD_ALLREDUCE);
    if (RecordOpDbInfo(
            HcclCMDType::HCCL_CMD_ALLREDUCE, curRank, reinterpret_cast<uint64_t>(stream), sendBuf, size, recvBuf, size,
            allReduceDetails, 0, rankSize, curRank, curRank, {}, comm)
        != 0) {
        HCCL_VM_ERROR("record op db info failed");
        return HcclResult::HCCL_E_PARA;
    }
    HCCL_VM_INFO("get op info: allRank= {}, curRank= {}.", rankSize, curRank);

    using HcclAddreduceFunc = HcclResult (*)(void*, void*, uint64_t, HcclDataType, HcclReduceOp, HcclComm, aclrtStream);
    const auto hcclAddreduceFunc
        = reinterpret_cast<HcclAddreduceFunc>(sim::DlsymRealWithFallback(__func__, "libhccl.so"));

    if (hcclAddreduceFunc != nullptr) {
        const HcclResult ret = hcclAddreduceFunc(sendBuf, recvBuf, count, dataType, op, comm, stream);
        return ret;
    } else {
        HCCL_VM_ERROR("dlsym failed");
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
}

HcclResult HcclScatter(
    void* sendBuf, void* recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root, HcclComm comm,
    aclrtStream stream)
{
    HCCL_VM_INFO("HcclScatter called with parameters:");
    HCCL_VM_INFO("sendBuf = {:p}", sendBuf);
    HCCL_VM_INFO("recvBuf = {:p}", recvBuf);
    HCCL_VM_INFO("recvCount = {}", recvCount);
    HCCL_VM_INFO("dataType = {}", GetDataTypeStr(dataType));
    HCCL_VM_INFO("root = {}", root);
    HCCL_VM_INFO("comm = {:p}", comm);
    HCCL_VM_INFO("stream = {:p}", stream);

    uint32_t curRank = 0;
    uint32_t rankSize = 0;
    const HcclResult setupRet = SetupCommunicator(comm, g_cur_device_key, curRank, rankSize);
    if (setupRet != HcclResult::HCCL_SUCCESS) {
        g_cur_comm_key = 0;
        return setupRet;
    }
    ScopedOperatorCommunicator scopedCommunicator;
    // 注册input、output buffer
    uint32_t dataSize = 0;
    if (sim::GetDataTypeSize(dataType, dataSize) != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR("HCCL_VM not support data type {} for HcclScatter calc size", GetDataTypeStr(dataType));
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    uint64_t inputValueSize = 0;
    if (curRank == root) {
        inputValueSize = static_cast<uint64_t>(dataSize) * recvCount * rankSize;
    }
    uint64_t outputValueSize = static_cast<uint64_t>(dataSize) * recvCount;

    // Use BuildOpDetailsV1 for Scatter
    auto scatterDetails = BuildOpDetailsV1(recvCount, dataType, 0, 0, HcclCMDType::HCCL_CMD_SCATTER);
    if (RecordOpDbInfo(
            HcclCMDType::HCCL_CMD_SCATTER, curRank, reinterpret_cast<uint64_t>(stream), sendBuf, inputValueSize,
            recvBuf, outputValueSize, scatterDetails, root, rankSize, curRank, curRank, {}, comm)
        != 0) {
        HCCL_VM_ERROR("record op db info failed");
        return HcclResult::HCCL_E_PARA;
    }

    HCCL_VM_INFO("get op info: allRank= {}, curRank= {}.", rankSize, curRank);

    using HcclScatterFunc = HcclResult (*)(void*, void*, uint64_t, HcclDataType, uint32_t, HcclComm, aclrtStream);
    const auto hcclScatterFunc = reinterpret_cast<HcclScatterFunc>(sim::DlsymRealWithFallback(__func__, "libhccl.so"));

    if (hcclScatterFunc != nullptr) {
        const HcclResult ret = hcclScatterFunc(sendBuf, recvBuf, recvCount, dataType, root, comm, stream);
        return ret;
    } else {
        HCCL_VM_ERROR("dlsym failed");
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
}

HcclResult HcclReduce(
    void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, uint32_t root, HcclComm comm,
    aclrtStream stream)
{
    HCCL_VM_INFO("HcclReduce called with parameters:");
    HCCL_VM_INFO("sendBuf = {:p}", sendBuf);
    HCCL_VM_INFO("recvBuf = {:p}", recvBuf);
    HCCL_VM_INFO("count = {}", count);
    HCCL_VM_INFO("dataType = {}", GetDataTypeStr(dataType));
    HCCL_VM_INFO("reduce op = {}", GetReduceOpStr(op));
    HCCL_VM_INFO("root = {}", root);
    HCCL_VM_INFO("comm = {:p}", comm);
    HCCL_VM_INFO("stream = {:p}", stream);

    uint32_t curRank = 0;
    uint32_t rankSize = 0;
    const HcclResult setupRet = SetupCommunicator(comm, g_cur_device_key, curRank, rankSize);
    if (setupRet != HcclResult::HCCL_SUCCESS) {
        g_cur_comm_key = 0;
        return setupRet;
    }
    ScopedOperatorCommunicator scopedCommunicator;
    // 注册input、output buffer
    uint32_t dataSize = 0;
    if (sim::GetDataTypeSize(dataType, dataSize) != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR("HCCL_VM not support data type {} for HcclReduce calc size", GetDataTypeStr(dataType));
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    uint64_t size = static_cast<uint64_t>(dataSize) * count;

    // Use BuildOpDetailsV1 for Reduce
    auto reduceDetails = BuildOpDetailsV1(
        count, dataType, static_cast<uint32_t>(op), static_cast<uint32_t>(op), HcclCMDType::HCCL_CMD_REDUCE);
    if (RecordOpDbInfo(
            HcclCMDType::HCCL_CMD_REDUCE, curRank, reinterpret_cast<uint64_t>(stream), sendBuf, size, recvBuf, size,
            reduceDetails, root, rankSize, curRank, curRank, {}, comm)
        != 0) {
        HCCL_VM_ERROR("record op db info failed");
        return HcclResult::HCCL_E_PARA;
    }

    HCCL_VM_INFO("get op info: allRank= {}, curRank= {}.", rankSize, curRank);

    using HcclReduceFunc
        = HcclResult (*)(void*, void*, uint64_t, HcclDataType, HcclReduceOp, uint32_t, HcclComm, aclrtStream);
    const auto hcclReduceFunc = reinterpret_cast<HcclReduceFunc>(sim::DlsymRealWithFallback(__func__, "libhccl.so"));

    if (hcclReduceFunc != nullptr) {
        const HcclResult ret = hcclReduceFunc(sendBuf, recvBuf, count, dataType, op, root, comm, stream);
        return ret;
    } else {
        HCCL_VM_ERROR("dlsym failed");
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
}

HcclResult HcclReduceScatter(
    void* sendBuf, void* recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op, HcclComm comm,
    aclrtStream stream)
{
    HCCL_VM_INFO("HcclReduceScatter called with parameters:");
    HCCL_VM_INFO("sendBuf = {:p}", sendBuf);
    HCCL_VM_INFO("recvBuf = {:p}", recvBuf);
    HCCL_VM_INFO("recvCount = {}", recvCount);
    HCCL_VM_INFO("dataType = {}", static_cast<int>(dataType));
    HCCL_VM_INFO("reduce op = {}", static_cast<int>(op));
    HCCL_VM_INFO("comm = {:p}", comm);
    HCCL_VM_INFO("stream = {:p}", stream);

    uint32_t curRank = 0;
    uint32_t rankSize = 0;
    const HcclResult setupRet = SetupCommunicator(comm, g_cur_device_key, curRank, rankSize);
    if (setupRet != HcclResult::HCCL_SUCCESS) {
        g_cur_comm_key = 0;
        return setupRet;
    }
    ScopedOperatorCommunicator scopedCommunicator;
    // 注册input、output buffer
    uint32_t dataSize = 0;
    if (sim::GetDataTypeSize(dataType, dataSize) != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR("HCCL_VM not support data type {} for HcclReduceScatter calc size", GetDataTypeStr(dataType));
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    uint64_t inputSize = static_cast<uint64_t>(dataSize) * recvCount * rankSize;
    uint64_t outputSize = static_cast<uint64_t>(dataSize) * recvCount;

    // Use BuildOpDetailsV1 for ReduceScatter
    auto reduceScatterDetails = BuildOpDetailsV1(
        recvCount, dataType, static_cast<uint32_t>(op), static_cast<uint32_t>(op),
        HcclCMDType::HCCL_CMD_REDUCE_SCATTER);
    if (RecordOpDbInfo(
            HcclCMDType::HCCL_CMD_REDUCE_SCATTER, curRank, reinterpret_cast<uint64_t>(stream), sendBuf, inputSize,
            recvBuf, outputSize, reduceScatterDetails, 0, rankSize, curRank, curRank, {}, comm)
        != 0) {
        HCCL_VM_ERROR("record op db info failed");
        return HcclResult::HCCL_E_PARA;
    }

    HCCL_VM_INFO("get op info: allRank= {}, curRank= {}.", rankSize, curRank);

    using HcclReduceScatterFunc
        = HcclResult (*)(void*, void*, uint64_t, HcclDataType, HcclReduceOp, HcclComm, aclrtStream);
    const auto hcclReduceScatterFunc
        = reinterpret_cast<HcclReduceScatterFunc>(sim::DlsymRealWithFallback(__func__, "libhccl.so"));

    if (hcclReduceScatterFunc != nullptr) {
        const HcclResult ret = hcclReduceScatterFunc(sendBuf, recvBuf, recvCount, dataType, op, comm, stream);
        return ret;
    } else {
        HCCL_VM_ERROR("dlsym failed");
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
}

HcclResult HcclReduceScatterV(
    void* sendBuf, const void* sendCounts, const void* sendDispls, void* recvBuf, uint64_t recvCount,
    HcclDataType dataType, HcclReduceOp op, HcclComm comm, aclrtStream stream)
{
    HCCL_VM_INFO("HcclReduceScatterV called with parameters:");
    HCCL_VM_INFO("sendBuf = {:p}", sendBuf);
    HCCL_VM_INFO("sendCounts = {:p}", sendCounts);
    HCCL_VM_INFO("sendDispls = {:p}", sendDispls);
    HCCL_VM_INFO("recvBuf = {:p}", recvBuf);
    HCCL_VM_INFO("recvCount = {}", recvCount);
    HCCL_VM_INFO("dataType = {}", GetDataTypeStr(dataType));
    HCCL_VM_INFO("reduce op = {}", GetReduceOpStr(op));
    HCCL_VM_INFO("comm = {:p}", comm);
    HCCL_VM_INFO("stream = {:p}", stream);

    uint32_t curRank = 0;
    uint32_t rankSize = 0;
    const HcclResult setupRet = SetupCommunicator(comm, g_cur_device_key, curRank, rankSize);
    if (setupRet != HcclResult::HCCL_SUCCESS) {
        g_cur_comm_key = 0;
        return setupRet;
    }
    ScopedOperatorCommunicator scopedCommunicator;
    if (rankSize == 0 || curRank >= rankSize) {
        HCCL_VM_ERROR("invalid ReduceScatterV rank information: rankId = {}, rankSize = {}", curRank, rankSize);
        return HcclResult::HCCL_E_PARA;
    }

    if (sendCounts == nullptr || sendDispls == nullptr) {
        HCCL_VM_ERROR("invalid ReduceScatterV count or displacement array");
        return HcclResult::HCCL_E_PARA;
    }

    const auto* sendCountValues = static_cast<const uint64_t*>(sendCounts);
    const auto* sendDisplValues = static_cast<const uint64_t*>(sendDispls);
    uint64_t totalSendCount = 0;
    for (uint32_t rank = 0; rank < rankSize; ++rank) {
        totalSendCount += sendCountValues[rank];
    }

    uint32_t dataSize = 0;
    if (sim::GetDataTypeSize(dataType, dataSize) != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR("HCCL_VM not support data type {} for HcclReduceScatterV calc size", GetDataTypeStr(dataType));
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    const uint64_t inputSize = totalSendCount * dataSize;
    const uint64_t outputSize = recvCount * dataSize;

    const auto reduceScatterVExtInfo = BuildVOpExtInfo(rankSize, recvCount, sendCountValues, sendDisplValues);
    const auto reduceScatterVDetails
        = BuildOpDetailsV1(recvCount, dataType, 0, op, HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V);
    if (RecordOpDbInfo(
            HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V, curRank, reinterpret_cast<uint64_t>(stream), sendBuf, inputSize,
            recvBuf, outputSize, reduceScatterVDetails, 0, rankSize, curRank, curRank, reduceScatterVExtInfo, comm)
        != 0) {
        HCCL_VM_ERROR("record op db info failed");
        return HcclResult::HCCL_E_PARA;
    }

    HCCL_VM_INFO("get op info: allRank= {}, curRank= {}.", rankSize, curRank);

    using HcclReduceScatterVFunc = HcclResult (*)(
        void*, const void*, const void*, void*, uint64_t, HcclDataType, HcclReduceOp, HcclComm, aclrtStream);
    const auto hcclReduceScatterVFunc
        = reinterpret_cast<HcclReduceScatterVFunc>(sim::DlsymRealWithFallback(__func__, "libhccl.so"));

    if (hcclReduceScatterVFunc != nullptr) {
        const HcclResult ret
            = hcclReduceScatterVFunc(sendBuf, sendCounts, sendDispls, recvBuf, recvCount, dataType, op, comm, stream);
        return ret;
    } else {
        HCCL_VM_ERROR("dlsym failed");
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
}

HcclResult HcclAllReduceGraphMode(
    void* sendBuf, void* recvBuf, uint64_t sendCount, HcclDataType dataType, HcclReduceOp op, const char* group,
    aclrtStream stream, const char* tag, void** streams, size_t streamCount, void* scratchMemAddr,
    uint64_t scratchMemSize)
{
    if (CheckDataAndReduceOpType(dataType, op) != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "HCCL_VM not support data type {} or reduce {} op for HcclAllReduceGraphMode", GetDataTypeStr(dataType),
            GetReduceOpStr(op));
        g_cur_comm_key = 0;
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    HCCL_VM_INFO("HcclAllReduceGraphMode called with parameters:");
    HCCL_VM_INFO("sendBuf = {:p}", sendBuf);
    HCCL_VM_INFO("recvBuf = {:p}", recvBuf);
    HCCL_VM_INFO("sendCount = {}", sendCount);
    HCCL_VM_INFO("dataType = {}", GetDataTypeStr(dataType));
    HCCL_VM_INFO("op = {}", GetReduceOpStr(op));
    HCCL_VM_INFO("group = {}", group == nullptr ? "(null)" : group);
    HCCL_VM_INFO("stream = {:p}", stream);
    HCCL_VM_INFO("tag = {}", tag == nullptr ? "(null)" : tag);
    HCCL_VM_INFO("streamCount = {}", streamCount);
    HCCL_VM_INFO("scratchMemAddr = {:p}", scratchMemAddr);
    HCCL_VM_INFO("scratchMemSize = {}", scratchMemSize);

    // 图模式入口只有 group 名, 先反查本进程已建模的 HcclComm 句柄。
    HcclComm comm = nullptr;
    if (!ResolveCommByGroup(group, comm)) {
        HCCL_VM_ERROR("resolve HcclComm by group[{}] failed", group == nullptr ? "(null)" : group);
        g_cur_comm_key = 0;
        return HcclResult::HCCL_E_PARA;
    }

    uint32_t curRank = 0;
    uint32_t rankSize = 0;
    const HcclResult setupRet = SetupCommunicator(comm, g_cur_device_key, curRank, rankSize);
    if (setupRet != HcclResult::HCCL_SUCCESS) {
        g_cur_comm_key = 0;
        return setupRet;
    }
    ScopedOperatorCommunicator scopedCommunicator;
    // 注册input、output buffer
    uint32_t dataSize = 0;
    if (sim::GetDataTypeSize(dataType, dataSize) != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "HCCL_VM not support data type {} for HcclAllReduceGraphMode calc size", GetDataTypeStr(dataType));
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    uint64_t size = static_cast<uint64_t>(dataSize) * sendCount;
    // Use BuildOpDetailsV1 for AllReduce
    auto allReduceDetails = BuildOpDetailsV1(
        sendCount, dataType, static_cast<uint32_t>(op), static_cast<uint32_t>(op), HcclCMDType::HCCL_CMD_ALLREDUCE);
    if (RecordOpDbInfo(
            HcclCMDType::HCCL_CMD_ALLREDUCE, curRank, reinterpret_cast<uint64_t>(stream), sendBuf, size, recvBuf, size,
            allReduceDetails, 0, rankSize, curRank, curRank, {}, comm)
        != 0) {
        HCCL_VM_ERROR("record op db info failed");
        return HcclResult::HCCL_E_PARA;
    }
    HCCL_VM_INFO("get op info: allRank= {}, curRank= {}.", rankSize, curRank);

    using HcclAllReduceGraphModeFunc = HcclResult (*)(
        void*, void*, uint64_t, HcclDataType, HcclReduceOp, const char*, aclrtStream, const char*, void**, size_t,
        void*, uint64_t);
    HcclAllReduceGraphModeFunc hcclAllReduceGraphModeFunc
        = reinterpret_cast<HcclAllReduceGraphModeFunc>(sim::DlsymRealWithFallback(__func__, "libhccl.so"));
    if (hcclAllReduceGraphModeFunc != nullptr) {
        const HcclResult ret = hcclAllReduceGraphModeFunc(
            sendBuf, recvBuf, sendCount, dataType, op, group, stream, tag, streams, streamCount, scratchMemAddr,
            scratchMemSize);
        return ret;
    } else {
        HCCL_VM_ERROR("dlsym failed");
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
}

HcclResult HcclBroadcastGraphMode(
    void* buf, uint64_t count, HcclDataType dataType, uint32_t root, const char* group, aclrtStream stream,
    const char* tag, void** streams, size_t streamCount, void* scratchMemAddr, uint64_t scratchMemSize)
{
    HCCL_VM_INFO("HcclBroadcastGraphMode called with parameters:");
    HCCL_VM_INFO("buf = {:p}", buf);
    HCCL_VM_INFO("count = {}", count);
    HCCL_VM_INFO("dataType = {}", GetDataTypeStr(dataType));
    HCCL_VM_INFO("root = {}", root);
    HCCL_VM_INFO("group = {}", group == nullptr ? "(null)" : group);
    HCCL_VM_INFO("stream = {:p}", stream);
    HCCL_VM_INFO("tag = {}", tag == nullptr ? "(null)" : tag);
    HCCL_VM_INFO("streamCount = {}", streamCount);
    HCCL_VM_INFO("scratchMemAddr = {:p}", scratchMemAddr);
    HCCL_VM_INFO("scratchMemSize = {}", scratchMemSize);

    HcclComm comm = nullptr;
    if (!ResolveCommByGroup(group, comm)) {
        HCCL_VM_ERROR("resolve HcclComm by group[{}] failed", group == nullptr ? "(null)" : group);
        g_cur_comm_key = 0;
        return HcclResult::HCCL_E_PARA;
    }

    uint32_t curRank = 0;
    uint32_t rankSize = 0;
    const HcclResult setupRet = SetupCommunicator(comm, g_cur_device_key, curRank, rankSize);
    if (setupRet != HcclResult::HCCL_SUCCESS) {
        g_cur_comm_key = 0;
        return setupRet;
    }
    ScopedOperatorCommunicator scopedCommunicator;
    uint32_t dataSize = 0;
    if (sim::GetDataTypeSize(dataType, dataSize) != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "HCCL_VM not support data type {} for HcclBroadcastGraphMode calc size", GetDataTypeStr(dataType));
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    uint64_t size = static_cast<uint64_t>(dataSize) * count;

    auto broadcastDetails = BuildOpDetailsV1(count, dataType, 0, 0, HcclCMDType::HCCL_CMD_BROADCAST);
    if (RecordOpDbInfo(
            HcclCMDType::HCCL_CMD_BROADCAST, curRank, reinterpret_cast<uint64_t>(stream), buf, size, buf, size,
            broadcastDetails, root, rankSize, curRank, curRank, {}, comm)
        != 0) {
        HCCL_VM_ERROR("record op db info failed");
        return HcclResult::HCCL_E_PARA;
    }
    HCCL_VM_INFO("get op info: allRank= {}, curRank= {}.", rankSize, curRank);

    using Func = HcclResult (*)(
        void*, uint64_t, HcclDataType, uint32_t, const char*, aclrtStream, const char*, void**, size_t, void*,
        uint64_t);
    const auto func = reinterpret_cast<Func>(sim::DlsymRealWithFallback(__func__, "libhccl.so"));
    if (func == nullptr) {
        HCCL_VM_ERROR("dlsym HcclBroadcastGraphMode failed");
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    return func(buf, count, dataType, root, group, stream, tag, streams, streamCount, scratchMemAddr, scratchMemSize);
}

HcclResult HcclAllGatherGraphMode(
    void* sendBuf, void* recvBuf, uint64_t sendCount, HcclDataType dataType, const char* group, aclrtStream stream,
    const char* tag, void** streams, size_t streamCount, void* scratchMemAddr, uint64_t scratchMemSize)
{
    HCCL_VM_INFO("HcclAllGatherGraphMode called with parameters:");
    HCCL_VM_INFO("sendBuf = {:p}", sendBuf);
    HCCL_VM_INFO("recvBuf = {:p}", recvBuf);
    HCCL_VM_INFO("sendCount = {}", sendCount);
    HCCL_VM_INFO("dataType = {}", GetDataTypeStr(dataType));
    HCCL_VM_INFO("group = {}", group == nullptr ? "(null)" : group);
    HCCL_VM_INFO("stream = {:p}", stream);
    HCCL_VM_INFO("tag = {}", tag == nullptr ? "(null)" : tag);
    HCCL_VM_INFO("streamCount = {}", streamCount);
    HCCL_VM_INFO("scratchMemAddr = {:p}", scratchMemAddr);
    HCCL_VM_INFO("scratchMemSize = {}", scratchMemSize);

    HcclComm comm = nullptr;
    if (!ResolveCommByGroup(group, comm)) {
        HCCL_VM_ERROR("resolve HcclComm by group[{}] failed", group == nullptr ? "(null)" : group);
        g_cur_comm_key = 0;
        return HcclResult::HCCL_E_PARA;
    }

    uint32_t curRank = 0;
    uint32_t rankSize = 0;
    const HcclResult setupRet = SetupCommunicator(comm, g_cur_device_key, curRank, rankSize);
    if (setupRet != HcclResult::HCCL_SUCCESS) {
        g_cur_comm_key = 0;
        return setupRet;
    }
    ScopedOperatorCommunicator scopedCommunicator;
    uint32_t dataSize = 0;
    if (sim::GetDataTypeSize(dataType, dataSize) != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "HCCL_VM not support data type {} for HcclAllGatherGraphMode calc size", GetDataTypeStr(dataType));
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    uint64_t inputSize = static_cast<uint64_t>(dataSize) * sendCount;
    uint64_t outputSize = static_cast<uint64_t>(dataSize) * sendCount * rankSize;

    auto allGatherDetails = BuildOpDetailsV1(sendCount, dataType, 0, 0, HcclCMDType::HCCL_CMD_ALLGATHER);
    if (RecordOpDbInfo(
            HcclCMDType::HCCL_CMD_ALLGATHER, curRank, reinterpret_cast<uint64_t>(stream), sendBuf, inputSize, recvBuf,
            outputSize, allGatherDetails, 0, rankSize, curRank, curRank, {}, comm)
        != 0) {
        HCCL_VM_ERROR("record op db info failed");
        return HcclResult::HCCL_E_PARA;
    }
    HCCL_VM_INFO("get op info: allRank= {}, curRank= {}.", rankSize, curRank);

    using Func = HcclResult (*)(
        void*, void*, uint64_t, HcclDataType, const char*, aclrtStream, const char*, void**, size_t, void*, uint64_t);
    const auto func = reinterpret_cast<Func>(sim::DlsymRealWithFallback(__func__, "libhccl.so"));
    if (func == nullptr) {
        HCCL_VM_ERROR("dlsym HcclAllGatherGraphMode failed");
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    return func(
        sendBuf, recvBuf, sendCount, dataType, group, stream, tag, streams, streamCount, scratchMemAddr,
        scratchMemSize);
}

HcclResult HcclAllGatherVGraphMode(
    void* sendBuf, void* recvBuf, uint64_t sendCount, const void* recvCounts, const void* recvDispls,
    HcclDataType dataType, const char* group, aclrtStream stream, const char* tag, void** streams, size_t streamCount,
    void* scratchMemAddr, uint64_t scratchMemSize)
{
    HCCL_VM_INFO("HcclAllGatherVGraphMode called with parameters:");
    HCCL_VM_INFO("sendBuf = {:p}", sendBuf);
    HCCL_VM_INFO("sendCount = {}", sendCount);
    HCCL_VM_INFO("recvBuf = {:p}", recvBuf);
    HCCL_VM_INFO("recvCounts = {:p}", recvCounts);
    HCCL_VM_INFO("recvDispls = {:p}", recvDispls);
    HCCL_VM_INFO("dataType = {}", GetDataTypeStr(dataType));
    HCCL_VM_INFO("group = {}", group == nullptr ? "(null)" : group);
    HCCL_VM_INFO("stream = {:p}", stream);
    HCCL_VM_INFO("tag = {}", tag == nullptr ? "(null)" : tag);
    HCCL_VM_INFO("streamCount = {}", streamCount);
    HCCL_VM_INFO("scratchMemAddr = {:p}", scratchMemAddr);
    HCCL_VM_INFO("scratchMemSize = {}", scratchMemSize);

    HcclComm comm = nullptr;
    if (!ResolveCommByGroup(group, comm)) {
        HCCL_VM_ERROR("resolve HcclComm by group[{}] failed", group == nullptr ? "(null)" : group);
        g_cur_comm_key = 0;
        return HcclResult::HCCL_E_PARA;
    }

    uint32_t curRank = 0;
    uint32_t rankSize = 0;
    const HcclResult setupRet = SetupCommunicator(comm, g_cur_device_key, curRank, rankSize);
    if (setupRet != HcclResult::HCCL_SUCCESS) {
        g_cur_comm_key = 0;
        return setupRet;
    }
    ScopedOperatorCommunicator scopedCommunicator;
    if (rankSize == 0 || curRank >= rankSize) {
        HCCL_VM_ERROR("invalid AllGatherVGraphMode rank information: rankId = {}, rankSize = {}", curRank, rankSize);
        return HcclResult::HCCL_E_PARA;
    }
    if (recvCounts == nullptr || recvDispls == nullptr) {
        HCCL_VM_ERROR("invalid AllGatherVGraphMode count or displacement array");
        return HcclResult::HCCL_E_PARA;
    }

    const auto* recvCountValues = static_cast<const uint64_t*>(recvCounts);
    const auto* recvDisplValues = static_cast<const uint64_t*>(recvDispls);
    uint64_t totalRecvCount = 0;
    for (uint32_t rank = 0; rank < rankSize; ++rank) {
        totalRecvCount += recvCountValues[rank];
    }

    uint32_t dataSize = 0;
    if (sim::GetDataTypeSize(dataType, dataSize) != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "HCCL_VM not support data type {} for HcclAllGatherVGraphMode calc size", GetDataTypeStr(dataType));
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    const uint64_t inputSize = sendCount * dataSize;
    const uint64_t outputSize = totalRecvCount * dataSize;

    const auto allGatherVExtInfo = BuildVOpExtInfo(rankSize, sendCount, recvCountValues, recvDisplValues);
    const auto allGatherVDetails = BuildOpDetailsV1(sendCount, dataType, 0, 0, HcclCMDType::HCCL_CMD_ALLGATHER_V);
    if (RecordOpDbInfo(
            HcclCMDType::HCCL_CMD_ALLGATHER_V, curRank, reinterpret_cast<uint64_t>(stream), sendBuf, inputSize, recvBuf,
            outputSize, allGatherVDetails, 0, rankSize, curRank, curRank, allGatherVExtInfo, comm)
        != 0) {
        HCCL_VM_ERROR("record op db info failed");
        return HcclResult::HCCL_E_PARA;
    }
    HCCL_VM_INFO("get op info: allRank= {}, curRank= {}.", rankSize, curRank);

    using Func = HcclResult (*)(
        void*, void*, uint64_t, const void*, const void*, HcclDataType, const char*, aclrtStream, const char*, void**,
        size_t, void*, uint64_t);
    const auto func = reinterpret_cast<Func>(sim::DlsymRealWithFallback(__func__, "libhccl.so"));
    if (func == nullptr) {
        HCCL_VM_ERROR("dlsym HcclAllGatherVGraphMode failed");
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    return func(
        sendBuf, recvBuf, sendCount, recvCounts, recvDispls, dataType, group, stream, tag, streams, streamCount,
        scratchMemAddr, scratchMemSize);
}

HcclResult HcclReduceGraphMode(
    void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, uint32_t root,
    const char* group, aclrtStream stream, const char* tag, void** streams, size_t streamCount, void* scratchMemAddr,
    uint64_t scratchMemSize)
{
    if (CheckDataAndReduceOpType(dataType, op) != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "HCCL_VM not support data type {} or reduce {} op for HcclReduceGraphMode", GetDataTypeStr(dataType),
            GetReduceOpStr(op));
        g_cur_comm_key = 0;
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    HCCL_VM_INFO("HcclReduceGraphMode called with parameters:");
    HCCL_VM_INFO("sendBuf = {:p}", sendBuf);
    HCCL_VM_INFO("recvBuf = {:p}", recvBuf);
    HCCL_VM_INFO("count = {}", count);
    HCCL_VM_INFO("dataType = {}", GetDataTypeStr(dataType));
    HCCL_VM_INFO("op = {}", GetReduceOpStr(op));
    HCCL_VM_INFO("root = {}", root);
    HCCL_VM_INFO("group = {}", group == nullptr ? "(null)" : group);
    HCCL_VM_INFO("stream = {:p}", stream);
    HCCL_VM_INFO("tag = {}", tag == nullptr ? "(null)" : tag);
    HCCL_VM_INFO("streamCount = {}", streamCount);
    HCCL_VM_INFO("scratchMemAddr = {:p}", scratchMemAddr);
    HCCL_VM_INFO("scratchMemSize = {}", scratchMemSize);

    HcclComm comm = nullptr;
    if (!ResolveCommByGroup(group, comm)) {
        HCCL_VM_ERROR("resolve HcclComm by group[{}] failed", group == nullptr ? "(null)" : group);
        g_cur_comm_key = 0;
        return HcclResult::HCCL_E_PARA;
    }

    uint32_t curRank = 0;
    uint32_t rankSize = 0;
    const HcclResult setupRet = SetupCommunicator(comm, g_cur_device_key, curRank, rankSize);
    if (setupRet != HcclResult::HCCL_SUCCESS) {
        g_cur_comm_key = 0;
        return setupRet;
    }
    ScopedOperatorCommunicator scopedCommunicator;
    uint32_t dataSize = 0;
    if (sim::GetDataTypeSize(dataType, dataSize) != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR("HCCL_VM not support data type {} for HcclReduceGraphMode calc size", GetDataTypeStr(dataType));
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    uint64_t size = static_cast<uint64_t>(dataSize) * count;

    auto reduceDetails = BuildOpDetailsV1(
        count, dataType, static_cast<uint32_t>(op), static_cast<uint32_t>(op), HcclCMDType::HCCL_CMD_REDUCE);
    if (RecordOpDbInfo(
            HcclCMDType::HCCL_CMD_REDUCE, curRank, reinterpret_cast<uint64_t>(stream), sendBuf, size, recvBuf, size,
            reduceDetails, root, rankSize, curRank, curRank, {}, comm)
        != 0) {
        HCCL_VM_ERROR("record op db info failed");
        return HcclResult::HCCL_E_PARA;
    }
    HCCL_VM_INFO("get op info: allRank= {}, curRank= {}.", rankSize, curRank);

    using Func = HcclResult (*)(
        void*, void*, uint64_t, HcclDataType, HcclReduceOp, uint32_t, const char*, aclrtStream, const char*, void**,
        size_t, void*, uint64_t);
    const auto func = reinterpret_cast<Func>(sim::DlsymRealWithFallback(__func__, "libhccl.so"));
    if (func == nullptr) {
        HCCL_VM_ERROR("dlsym HcclReduceGraphMode failed");
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    return func(
        sendBuf, recvBuf, count, dataType, op, root, group, stream, tag, streams, streamCount, scratchMemAddr,
        scratchMemSize);
}

HcclResult HcclReduceScatterGraphMode(
    void* sendBuf, void* recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op, const char* group,
    aclrtStream stream, const char* tag, void** streams, size_t streamCount, void* scratchMemAddr,
    uint64_t scratchMemSize)
{
    if (CheckDataAndReduceOpType(dataType, op) != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "HCCL_VM not support data type {} or reduce {} op for HcclReduceScatterGraphMode", GetDataTypeStr(dataType),
            GetReduceOpStr(op));
        g_cur_comm_key = 0;
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    HCCL_VM_INFO("HcclReduceScatterGraphMode called with parameters:");
    HCCL_VM_INFO("sendBuf = {:p}", sendBuf);
    HCCL_VM_INFO("recvBuf = {:p}", recvBuf);
    HCCL_VM_INFO("recvCount = {}", recvCount);
    HCCL_VM_INFO("dataType = {}", GetDataTypeStr(dataType));
    HCCL_VM_INFO("op = {}", GetReduceOpStr(op));
    HCCL_VM_INFO("group = {}", group == nullptr ? "(null)" : group);
    HCCL_VM_INFO("stream = {:p}", stream);
    HCCL_VM_INFO("tag = {}", tag == nullptr ? "(null)" : tag);
    HCCL_VM_INFO("streamCount = {}", streamCount);
    HCCL_VM_INFO("scratchMemAddr = {:p}", scratchMemAddr);
    HCCL_VM_INFO("scratchMemSize = {}", scratchMemSize);

    HcclComm comm = nullptr;
    if (!ResolveCommByGroup(group, comm)) {
        HCCL_VM_ERROR("resolve HcclComm by group[{}] failed", group == nullptr ? "(null)" : group);
        g_cur_comm_key = 0;
        return HcclResult::HCCL_E_PARA;
    }

    uint32_t curRank = 0;
    uint32_t rankSize = 0;
    const HcclResult setupRet = SetupCommunicator(comm, g_cur_device_key, curRank, rankSize);
    if (setupRet != HcclResult::HCCL_SUCCESS) {
        g_cur_comm_key = 0;
        return setupRet;
    }
    ScopedOperatorCommunicator scopedCommunicator;
    uint32_t dataSize = 0;
    if (sim::GetDataTypeSize(dataType, dataSize) != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "HCCL_VM not support data type {} for HcclReduceScatterGraphMode calc size", GetDataTypeStr(dataType));
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    uint64_t inputSize = static_cast<uint64_t>(dataSize) * recvCount * rankSize;
    uint64_t outputSize = static_cast<uint64_t>(dataSize) * recvCount;

    auto reduceScatterDetails = BuildOpDetailsV1(
        recvCount, dataType, static_cast<uint32_t>(op), static_cast<uint32_t>(op),
        HcclCMDType::HCCL_CMD_REDUCE_SCATTER);
    if (RecordOpDbInfo(
            HcclCMDType::HCCL_CMD_REDUCE_SCATTER, curRank, reinterpret_cast<uint64_t>(stream), sendBuf, inputSize,
            recvBuf, outputSize, reduceScatterDetails, 0, rankSize, curRank, curRank, {}, comm)
        != 0) {
        HCCL_VM_ERROR("record op db info failed");
        return HcclResult::HCCL_E_PARA;
    }
    HCCL_VM_INFO("get op info: allRank= {}, curRank= {}.", rankSize, curRank);

    using Func = HcclResult (*)(
        void*, void*, uint64_t, HcclDataType, HcclReduceOp, const char*, aclrtStream, const char*, void**, size_t,
        void*, uint64_t);
    const auto func = reinterpret_cast<Func>(sim::DlsymRealWithFallback(__func__, "libhccl.so"));
    if (func == nullptr) {
        HCCL_VM_ERROR("dlsym HcclReduceScatterGraphMode failed");
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    return func(
        sendBuf, recvBuf, recvCount, dataType, op, group, stream, tag, streams, streamCount, scratchMemAddr,
        scratchMemSize);
}

HcclResult HcclReduceScatterVGraphMode(
    void* sendBuf, const void* sendCounts, const void* sendDispls, void* recvBuf, uint64_t recvCount,
    HcclDataType dataType, HcclReduceOp op, const char* group, aclrtStream stream, const char* tag, void** streams,
    size_t streamCount, void* scratchMemAddr, uint64_t scratchMemSize)
{
    if (CheckDataAndReduceOpType(dataType, op) != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "HCCL_VM not support data type {} or reduce {} op for HcclReduceScatterVGraphMode",
            GetDataTypeStr(dataType), GetReduceOpStr(op));
        g_cur_comm_key = 0;
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    HCCL_VM_INFO("HcclReduceScatterVGraphMode called with parameters:");
    HCCL_VM_INFO("sendBuf = {:p}", sendBuf);
    HCCL_VM_INFO("sendCounts = {:p}", sendCounts);
    HCCL_VM_INFO("sendDispls = {:p}", sendDispls);
    HCCL_VM_INFO("recvBuf = {:p}", recvBuf);
    HCCL_VM_INFO("recvCount = {}", recvCount);
    HCCL_VM_INFO("dataType = {}", GetDataTypeStr(dataType));
    HCCL_VM_INFO("op = {}", GetReduceOpStr(op));
    HCCL_VM_INFO("group = {}", group == nullptr ? "(null)" : group);
    HCCL_VM_INFO("stream = {:p}", stream);
    HCCL_VM_INFO("tag = {}", tag == nullptr ? "(null)" : tag);
    HCCL_VM_INFO("streamCount = {}", streamCount);
    HCCL_VM_INFO("scratchMemAddr = {:p}", scratchMemAddr);
    HCCL_VM_INFO("scratchMemSize = {}", scratchMemSize);

    HcclComm comm = nullptr;
    if (!ResolveCommByGroup(group, comm)) {
        HCCL_VM_ERROR("resolve HcclComm by group[{}] failed", group == nullptr ? "(null)" : group);
        g_cur_comm_key = 0;
        return HcclResult::HCCL_E_PARA;
    }

    uint32_t curRank = 0;
    uint32_t rankSize = 0;
    const HcclResult setupRet = SetupCommunicator(comm, g_cur_device_key, curRank, rankSize);
    if (setupRet != HcclResult::HCCL_SUCCESS) {
        g_cur_comm_key = 0;
        return setupRet;
    }
    ScopedOperatorCommunicator scopedCommunicator;
    if (rankSize == 0 || curRank >= rankSize) {
        HCCL_VM_ERROR(
            "invalid ReduceScatterVGraphMode rank information: rankId = {}, rankSize = {}", curRank, rankSize);
        return HcclResult::HCCL_E_PARA;
    }
    if (sendCounts == nullptr || sendDispls == nullptr) {
        HCCL_VM_ERROR("invalid ReduceScatterVGraphMode count or displacement array");
        return HcclResult::HCCL_E_PARA;
    }

    const auto* sendCountValues = static_cast<const uint64_t*>(sendCounts);
    const auto* sendDisplValues = static_cast<const uint64_t*>(sendDispls);
    uint64_t totalSendCount = 0;
    for (uint32_t rank = 0; rank < rankSize; ++rank) {
        totalSendCount += sendCountValues[rank];
    }

    uint32_t dataSize = 0;
    if (sim::GetDataTypeSize(dataType, dataSize) != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "HCCL_VM not support data type {} for HcclReduceScatterVGraphMode calc size", GetDataTypeStr(dataType));
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    const uint64_t inputSize = totalSendCount * dataSize;
    const uint64_t outputSize = recvCount * dataSize;

    const auto reduceScatterVExtInfo = BuildVOpExtInfo(rankSize, recvCount, sendCountValues, sendDisplValues);
    const auto reduceScatterVDetails
        = BuildOpDetailsV1(recvCount, dataType, 0, static_cast<uint32_t>(op), HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V);
    if (RecordOpDbInfo(
            HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V, curRank, reinterpret_cast<uint64_t>(stream), sendBuf, inputSize,
            recvBuf, outputSize, reduceScatterVDetails, 0, rankSize, curRank, curRank, reduceScatterVExtInfo, comm)
        != 0) {
        HCCL_VM_ERROR("record op db info failed");
        return HcclResult::HCCL_E_PARA;
    }
    HCCL_VM_INFO("get op info: allRank= {}, curRank= {}.", rankSize, curRank);

    using Func = HcclResult (*)(
        void*, const void*, const void*, void*, uint64_t, HcclDataType, HcclReduceOp, const char*, aclrtStream,
        const char*, void**, size_t, void*, uint64_t);
    const auto func = reinterpret_cast<Func>(sim::DlsymRealWithFallback(__func__, "libhccl.so"));
    if (func == nullptr) {
        HCCL_VM_ERROR("dlsym HcclReduceScatterVGraphMode failed");
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    return func(
        sendBuf, sendCounts, sendDispls, recvBuf, recvCount, dataType, op, group, stream, tag, streams, streamCount,
        scratchMemAddr, scratchMemSize);
}

HcclResult HcclSendGraphMode(
    void* sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank, const char* group, aclrtStream stream,
    const char* tag, void** streams, size_t streamCount, void* scratchMemAddr, uint64_t scratchMemSize)
{
    HCCL_VM_INFO("HcclSendGraphMode called with parameters:");
    HCCL_VM_INFO("sendBuf = {:p}", sendBuf);
    HCCL_VM_INFO("count = {}", count);
    HCCL_VM_INFO("dataType = {}", GetDataTypeStr(dataType));
    HCCL_VM_INFO("destRank = {}", destRank);
    HCCL_VM_INFO("group = {}", group == nullptr ? "(null)" : group);
    HCCL_VM_INFO("stream = {:p}", stream);
    HCCL_VM_INFO("tag = {}", tag == nullptr ? "(null)" : tag);
    HCCL_VM_INFO("streamCount = {}", streamCount);
    HCCL_VM_INFO("scratchMemAddr = {:p}", scratchMemAddr);
    HCCL_VM_INFO("scratchMemSize = {}", scratchMemSize);

    HcclComm comm = nullptr;
    if (!ResolveCommByGroup(group, comm)) {
        HCCL_VM_ERROR("resolve HcclComm by group[{}] failed", group == nullptr ? "(null)" : group);
        g_cur_comm_key = 0;
        return HcclResult::HCCL_E_PARA;
    }

    uint32_t curRank = 0;
    uint32_t rankSize = 0;
    const HcclResult setupRet = SetupCommunicator(comm, g_cur_device_key, curRank, rankSize);
    if (setupRet != HcclResult::HCCL_SUCCESS) {
        g_cur_comm_key = 0;
        return setupRet;
    }
    ScopedOperatorCommunicator scopedCommunicator;
    if (stream == nullptr || rankSize == 0 || curRank >= rankSize || destRank >= rankSize || destRank == curRank
        || (count != 0 && sendBuf == nullptr)) {
        HCCL_VM_ERROR(
            "invalid HcclSendGraphMode parameters: rank={}, rankSize={}, destRank={}", curRank, rankSize, destRank);
        return HcclResult::HCCL_E_PARA;
    }
    uint32_t typeSize = 0;
    if (sim::GetDataTypeSize(dataType, typeSize) != HcclResult::HCCL_SUCCESS) {
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    const uint64_t dataSize = count * typeSize;
    const auto details = BuildOpDetailsV1(count, dataType, 0, 0, HcclCMDType::HCCL_CMD_SEND);
    if (RecordOpDbInfo(
            HcclCMDType::HCCL_CMD_SEND, curRank, reinterpret_cast<uint64_t>(stream), sendBuf, dataSize, nullptr, 0,
            details, 0, rankSize, curRank, destRank, {}, comm)
        != 0) {
        return HcclResult::HCCL_E_PARA;
    }

    using Func = HcclResult (*)(
        void*, uint64_t, HcclDataType, uint32_t, const char*, aclrtStream, const char*, void**, size_t, void*,
        uint64_t);
    const auto func = reinterpret_cast<Func>(sim::DlsymRealWithFallback(__func__, "libhccl.so"));
    if (func == nullptr) {
        HCCL_VM_ERROR("dlsym HcclSendGraphMode failed");
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    return func(
        sendBuf, count, dataType, destRank, group, stream, tag, streams, streamCount, scratchMemAddr, scratchMemSize);
}

HcclResult HcclRecvGraphMode(
    void* recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank, const char* group, aclrtStream stream,
    const char* tag, void** streams, size_t streamCount, void* scratchMemAddr, uint64_t scratchMemSize)
{
    HCCL_VM_INFO("HcclRecvGraphMode called with parameters:");
    HCCL_VM_INFO("recvBuf = {:p}", recvBuf);
    HCCL_VM_INFO("count = {}", count);
    HCCL_VM_INFO("dataType = {}", GetDataTypeStr(dataType));
    HCCL_VM_INFO("srcRank = {}", srcRank);
    HCCL_VM_INFO("group = {}", group == nullptr ? "(null)" : group);
    HCCL_VM_INFO("stream = {:p}", stream);
    HCCL_VM_INFO("tag = {}", tag == nullptr ? "(null)" : tag);
    HCCL_VM_INFO("streamCount = {}", streamCount);
    HCCL_VM_INFO("scratchMemAddr = {:p}", scratchMemAddr);
    HCCL_VM_INFO("scratchMemSize = {}", scratchMemSize);

    HcclComm comm = nullptr;
    if (!ResolveCommByGroup(group, comm)) {
        HCCL_VM_ERROR("resolve HcclComm by group[{}] failed", group == nullptr ? "(null)" : group);
        g_cur_comm_key = 0;
        return HcclResult::HCCL_E_PARA;
    }

    uint32_t curRank = 0;
    uint32_t rankSize = 0;
    const HcclResult setupRet = SetupCommunicator(comm, g_cur_device_key, curRank, rankSize);
    if (setupRet != HcclResult::HCCL_SUCCESS) {
        g_cur_comm_key = 0;
        return setupRet;
    }
    ScopedOperatorCommunicator scopedCommunicator;
    if (stream == nullptr || rankSize == 0 || curRank >= rankSize || srcRank >= rankSize || srcRank == curRank
        || (count != 0 && recvBuf == nullptr)) {
        HCCL_VM_ERROR(
            "invalid HcclRecvGraphMode parameters: rank={}, rankSize={}, srcRank={}", curRank, rankSize, srcRank);
        return HcclResult::HCCL_E_PARA;
    }
    uint32_t typeSize = 0;
    if (sim::GetDataTypeSize(dataType, typeSize) != HcclResult::HCCL_SUCCESS) {
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    const uint64_t dataSize = count * typeSize;
    const auto details = BuildOpDetailsV1(count, dataType, 0, 0, HcclCMDType::HCCL_CMD_RECEIVE);
    if (RecordOpDbInfo(
            HcclCMDType::HCCL_CMD_RECEIVE, curRank, reinterpret_cast<uint64_t>(stream), nullptr, 0, recvBuf, dataSize,
            details, 0, rankSize, srcRank, curRank, {}, comm)
        != 0) {
        return HcclResult::HCCL_E_PARA;
    }

    using Func = HcclResult (*)(
        void*, uint64_t, HcclDataType, uint32_t, const char*, aclrtStream, const char*, void**, size_t, void*,
        uint64_t);
    const auto func = reinterpret_cast<Func>(sim::DlsymRealWithFallback(__func__, "libhccl.so"));
    if (func == nullptr) {
        HCCL_VM_ERROR("dlsym HcclRecvGraphMode failed");
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    return func(
        recvBuf, count, dataType, srcRank, group, stream, tag, streams, streamCount, scratchMemAddr, scratchMemSize);
}

HcclResult HcclAlltoAllGraphMode(
    const void* sendBuf, uint64_t sendCount, HcclDataType sendType, const void* recvBuf, uint64_t recvCount,
    HcclDataType recvType, const char* group, aclrtStream stream, const char* tag, void** streams, size_t streamCount,
    void* scratchMemAddr, uint64_t scratchMemSize)
{
    HCCL_VM_INFO("HcclAlltoAllGraphMode called with parameters:");
    HCCL_VM_INFO("sendBuf = {:p}", sendBuf);
    HCCL_VM_INFO("recvBuf = {:p}", recvBuf);
    HCCL_VM_INFO("sendCount = {}", sendCount);
    HCCL_VM_INFO("recvCount = {}", recvCount);
    HCCL_VM_INFO("sendType = {}", GetDataTypeStr(sendType));
    HCCL_VM_INFO("recvType = {}", GetDataTypeStr(recvType));
    HCCL_VM_INFO("group = {}", group == nullptr ? "(null)" : group);
    HCCL_VM_INFO("stream = {:p}", stream);
    HCCL_VM_INFO("tag = {}", tag == nullptr ? "(null)" : tag);
    HCCL_VM_INFO("streamCount = {}", streamCount);
    HCCL_VM_INFO("scratchMemAddr = {:p}", scratchMemAddr);
    HCCL_VM_INFO("scratchMemSize = {}", scratchMemSize);

    HcclComm comm = nullptr;
    if (!ResolveCommByGroup(group, comm)) {
        HCCL_VM_ERROR("resolve HcclComm by group[{}] failed", group == nullptr ? "(null)" : group);
        g_cur_comm_key = 0;
        return HcclResult::HCCL_E_PARA;
    }

    uint32_t curRank = 0;
    uint32_t rankSize = 0;
    const HcclResult setupRet = SetupCommunicator(comm, g_cur_device_key, curRank, rankSize);
    if (setupRet != HcclResult::HCCL_SUCCESS) {
        g_cur_comm_key = 0;
        return setupRet;
    }
    ScopedOperatorCommunicator scopedCommunicator;
    uint32_t inDataSize = 0;
    if (sim::GetDataTypeSize(sendType, inDataSize) != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "HCCL_VM not support data type {} for HcclAlltoAllGraphMode send type calc size", GetDataTypeStr(sendType));
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    uint32_t outDataSize = 0;
    if (sim::GetDataTypeSize(recvType, outDataSize) != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "HCCL_VM not support data type {} for HcclAlltoAllGraphMode recv type calc size", GetDataTypeStr(recvType));
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    uint64_t inputSize = static_cast<uint64_t>(inDataSize) * sendCount * rankSize;
    uint64_t outputSize = static_cast<uint64_t>(outDataSize) * recvCount * rankSize;

    auto alltoallDetails
        = BuildOpDetailsV2(sendCount, sendType, recvCount, recvType, 0, HcclCMDType::HCCL_CMD_ALLTOALL, sendType, 0);
    std::vector<uint8_t> alltoallExtInfo;
    uint32_t alltoallCount = rankSize * rankSize;
    alltoallExtInfo.resize(sizeof(uint32_t) + alltoallCount * sizeof(uint64_t));
    std::memcpy(alltoallExtInfo.data(), &alltoallCount, sizeof(uint32_t));
    for (uint32_t i = 0; i < alltoallCount; i++) {
        std::memcpy(alltoallExtInfo.data() + sizeof(uint32_t) + i * sizeof(uint64_t), &sendCount, sizeof(uint64_t));
    }
    if (RecordOpDbInfo(
            HcclCMDType::HCCL_CMD_ALLTOALL, curRank, reinterpret_cast<uint64_t>(stream), sendBuf, inputSize, recvBuf,
            outputSize, alltoallDetails, 0, rankSize, curRank, curRank, alltoallExtInfo, comm)
        != 0) {
        HCCL_VM_ERROR("record op db info failed");
        return HcclResult::HCCL_E_PARA;
    }
    HCCL_VM_INFO("get op info: allRank= {}, curRank= {}.", rankSize, curRank);

    using Func = HcclResult (*)(
        const void*, uint64_t, HcclDataType, const void*, uint64_t, HcclDataType, const char*, aclrtStream, const char*,
        void**, size_t, void*, uint64_t);
    const auto func = reinterpret_cast<Func>(sim::DlsymRealWithFallback(__func__, "libhccl.so"));
    if (func == nullptr) {
        HCCL_VM_ERROR("dlsym HcclAlltoAllGraphMode failed");
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    return func(
        sendBuf, sendCount, sendType, recvBuf, recvCount, recvType, group, stream, tag, streams, streamCount,
        scratchMemAddr, scratchMemSize);
}

HcclResult HcclAlltoAllVGraphMode(
    const void* sendBuf, const void* sendCounts, const void* sdispls, HcclDataType sendType, const void* recvBuf,
    const void* recvCounts, const void* rdispls, HcclDataType recvType, const char* group, aclrtStream stream,
    const char* tag, void** streams, size_t streamCount, void* scratchMemAddr, uint64_t scratchMemSize)
{
    HCCL_VM_INFO("HcclAlltoAllVGraphMode called with parameters:");
    HCCL_VM_INFO("sendBuf = {:p}", sendBuf);
    HCCL_VM_INFO("recvBuf = {:p}", recvBuf);
    HCCL_VM_INFO("sendType = {}", GetDataTypeStr(sendType));
    HCCL_VM_INFO("recvType = {}", GetDataTypeStr(recvType));
    HCCL_VM_INFO("group = {}", group == nullptr ? "(null)" : group);
    HCCL_VM_INFO("stream = {:p}", stream);
    HCCL_VM_INFO("tag = {}", tag == nullptr ? "(null)" : tag);
    HCCL_VM_INFO("streamCount = {}", streamCount);
    HCCL_VM_INFO("scratchMemAddr = {:p}", scratchMemAddr);
    HCCL_VM_INFO("scratchMemSize = {}", scratchMemSize);

    if (sendCounts == nullptr) {
        HCCL_VM_ERROR("HcclAlltoAllVGraphMode: sendCounts is nullptr");
        g_cur_comm_key = 0;
        return HcclResult::HCCL_E_PTR;
    }
    if (recvCounts == nullptr) {
        HCCL_VM_ERROR("HcclAlltoAllVGraphMode: recvCounts is nullptr");
        g_cur_comm_key = 0;
        return HcclResult::HCCL_E_PTR;
    }

    HcclComm comm = nullptr;
    if (!ResolveCommByGroup(group, comm)) {
        HCCL_VM_ERROR("resolve HcclComm by group[{}] failed", group == nullptr ? "(null)" : group);
        g_cur_comm_key = 0;
        return HcclResult::HCCL_E_PARA;
    }

    uint32_t curRank = 0;
    uint32_t rankSize = 0;
    const HcclResult setupRet = SetupCommunicator(comm, g_cur_device_key, curRank, rankSize);
    if (setupRet != HcclResult::HCCL_SUCCESS) {
        g_cur_comm_key = 0;
        return setupRet;
    }
    ScopedOperatorCommunicator scopedCommunicator;
    uint32_t inDataSize = 0;
    if (sim::GetDataTypeSize(sendType, inDataSize) != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "HCCL_VM not support data type {} for HcclAlltoAllVGraphMode send type calc size",
            GetDataTypeStr(sendType));
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    uint32_t outDataSize = 0;
    if (sim::GetDataTypeSize(recvType, outDataSize) != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "HCCL_VM not support data type {} for HcclAlltoAllVGraphMode recv type calc size",
            GetDataTypeStr(recvType));
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    uint64_t inCountTotal = 0;
    uint64_t outCountTotal = 0;
    for (uint32_t rank = 0; rank < rankSize; rank++) {
        inCountTotal += static_cast<const uint64_t*>(sendCounts)[rank];
        outCountTotal += static_cast<const uint64_t*>(recvCounts)[rank];
    }
    uint64_t inputSize = static_cast<uint64_t>(inDataSize) * inCountTotal;
    uint64_t outputSize = static_cast<uint64_t>(outDataSize) * outCountTotal;

    auto alltoallvDetails = BuildOpDetailsV2(
        static_cast<uint64_t>(inCountTotal), sendType, static_cast<uint64_t>(outCountTotal), recvType, 0,
        HcclCMDType::HCCL_CMD_ALLTOALLV, sendType, 0);

    std::vector<uint64_t> sendCountMatrix(rankSize * rankSize, 0);
    for (uint32_t j = 0; j < rankSize; j++) {
        sendCountMatrix[curRank * rankSize + j] = static_cast<const uint64_t*>(sendCounts)[j];
    }
    for (uint32_t j = 0; j < rankSize; j++) {
        sendCountMatrix[j * rankSize + curRank] = static_cast<const uint64_t*>(recvCounts)[j];
    }

    std::vector<uint8_t> alltoallvExtInfo;
    uint32_t alltoallvCount = rankSize * rankSize;
    alltoallvExtInfo.resize(sizeof(uint32_t) + alltoallvCount * sizeof(uint64_t));
    std::memcpy(alltoallvExtInfo.data(), &alltoallvCount, sizeof(uint32_t));
    std::memcpy(alltoallvExtInfo.data() + sizeof(uint32_t), sendCountMatrix.data(), alltoallvCount * sizeof(uint64_t));
    if (RecordOpDbInfo(
            HcclCMDType::HCCL_CMD_ALLTOALLV, curRank, reinterpret_cast<uint64_t>(stream), sendBuf, inputSize, recvBuf,
            outputSize, alltoallvDetails, 0, rankSize, curRank, curRank, alltoallvExtInfo, comm)
        != 0) {
        HCCL_VM_ERROR("record op db info failed");
        return HcclResult::HCCL_E_PARA;
    }
    HCCL_VM_INFO("get op info: allRank= {}, curRank= {}.", rankSize, curRank);

    using Func = HcclResult (*)(
        const void*, const void*, const void*, HcclDataType, const void*, const void*, const void*, HcclDataType,
        const char*, aclrtStream, const char*, void**, size_t, void*, uint64_t);
    const auto func = reinterpret_cast<Func>(sim::DlsymRealWithFallback(__func__, "libhccl.so"));
    if (func == nullptr) {
        HCCL_VM_ERROR("dlsym HcclAlltoAllVGraphMode failed");
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    return func(
        sendBuf, sendCounts, sdispls, sendType, recvBuf, recvCounts, rdispls, recvType, group, stream, tag, streams,
        streamCount, scratchMemAddr, scratchMemSize);
}

HcclResult HcclAlltoAllVCGraphMode(
    const void* sendBuf, const void* sendCountMatrix, HcclDataType sendType, const void* recvBuf, HcclDataType recvType,
    const char* group, aclrtStream stream, const char* tag, void** streams, size_t streamCount, void* scratchMemAddr,
    uint64_t scratchMemSize)
{
    HCCL_VM_INFO("HcclAlltoAllVCGraphMode called with parameters:");
    HCCL_VM_INFO("sendBuf = {:p}", sendBuf);
    HCCL_VM_INFO("sendCountMatrix = {:p}", sendCountMatrix);
    HCCL_VM_INFO("recvBuf = {:p}", recvBuf);
    HCCL_VM_INFO("sendType = {}", GetDataTypeStr(sendType));
    HCCL_VM_INFO("recvType = {}", GetDataTypeStr(recvType));
    HCCL_VM_INFO("group = {}", group == nullptr ? "(null)" : group);
    HCCL_VM_INFO("stream = {:p}", stream);
    HCCL_VM_INFO("tag = {}", tag == nullptr ? "(null)" : tag);
    HCCL_VM_INFO("streamCount = {}", streamCount);
    HCCL_VM_INFO("scratchMemAddr = {:p}", scratchMemAddr);
    HCCL_VM_INFO("scratchMemSize = {}", scratchMemSize);

    HcclComm comm = nullptr;
    if (!ResolveCommByGroup(group, comm)) {
        HCCL_VM_ERROR("resolve HcclComm by group[{}] failed", group == nullptr ? "(null)" : group);
        g_cur_comm_key = 0;
        return HcclResult::HCCL_E_PARA;
    }

    uint32_t curRank = 0;
    uint32_t rankSize = 0;
    const HcclResult setupRet = SetupCommunicator(comm, g_cur_device_key, curRank, rankSize);
    if (setupRet != HcclResult::HCCL_SUCCESS) {
        g_cur_comm_key = 0;
        return setupRet;
    }
    ScopedOperatorCommunicator scopedCommunicator;
    if (rankSize == 0 || curRank >= rankSize || sendCountMatrix == nullptr) {
        HCCL_VM_ERROR("invalid AlltoAllVCGraphMode rank information: rankId = {}, rankSize = {}", curRank, rankSize);
        return HcclResult::HCCL_E_PARA;
    }

    const auto* matrix = static_cast<const uint64_t*>(sendCountMatrix);
    uint64_t inputCount = 0;
    uint64_t outputCount = 0;
    const size_t rowOffset = static_cast<size_t>(curRank) * static_cast<size_t>(rankSize);
    for (uint32_t peerRank = 0; peerRank < rankSize; ++peerRank) {
        inputCount += matrix[rowOffset + peerRank];
        outputCount += matrix[static_cast<size_t>(peerRank) * rankSize + curRank];
    }

    uint32_t sendDataSize = 0;
    if (sim::GetDataTypeSize(sendType, sendDataSize) != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "HCCL_VM not support data type {} for HcclAlltoAllVCGraphMode send type calc size",
            GetDataTypeStr(sendType));
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    uint32_t recvDataSize = 0;
    if (sim::GetDataTypeSize(recvType, recvDataSize) != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "HCCL_VM not support data type {} for HcclAlltoAllVCGraphMode recv type calc size",
            GetDataTypeStr(recvType));
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    const uint64_t inputSize = inputCount * sendDataSize;
    const uint64_t outputSize = outputCount * recvDataSize;

    const auto alltoallvcDetails = BuildOpDetailsV2(
        inputCount, sendType, outputCount, recvType, 0, HcclCMDType::HCCL_CMD_ALLTOALLVC, sendType, 0);
    const uint32_t matrixElementCount = rankSize * rankSize;
    const auto alltoallvcExtInfo = BuildMatrixExtInfo(matrix, matrixElementCount);
    if (RecordOpDbInfo(
            HcclCMDType::HCCL_CMD_ALLTOALLVC, curRank, reinterpret_cast<uint64_t>(stream), sendBuf, inputSize, recvBuf,
            outputSize, alltoallvcDetails, 0, rankSize, curRank, curRank, alltoallvcExtInfo, comm)
        != 0) {
        HCCL_VM_ERROR("record op db info failed");
        return HcclResult::HCCL_E_PARA;
    }
    HCCL_VM_INFO("get op info: allRank= {}, curRank= {}.", rankSize, curRank);

    using Func = HcclResult (*)(
        const void*, const void*, HcclDataType, const void*, HcclDataType, const char*, aclrtStream, const char*,
        void**, size_t, void*, uint64_t);
    const auto func = reinterpret_cast<Func>(sim::DlsymRealWithFallback(__func__, "libhccl.so"));
    if (func == nullptr) {
        HCCL_VM_ERROR("dlsym HcclAlltoAllVCGraphMode failed");
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    return func(
        sendBuf, sendCountMatrix, sendType, recvBuf, recvType, group, stream, tag, streams, streamCount, scratchMemAddr,
        scratchMemSize);
}

HcclResult _Z18GetRunSideIsDeviceRb(bool& isDeviceSide)
{
    isDeviceSide = false;
    return HcclResult::HCCL_SUCCESS;
}

// 自定义算子入口劫持：透明转发（不解释尾部参数）
//
// 该两个接口在固定 6 参数之后还挂过多种尾部布局（ApiTestParam 结构体、
// (uint32,uint64)、(float,uint64) 等），桩若按某一种写死，转发时就会截断/错位。
// 这里改用 GCC 的 __builtin_apply_args/__builtin_apply：把“本函数收到的全部参数”
// 原样喂给真实实现，桩完全不关心尾部参数的类型、个数和大小。
//
// 返回值：__builtin_apply 的结果是“返回值所在内存块”的指针，配合 __builtin_return
// 可把真实返回值原样带回给调用方（实测 int/enum 返回值正确，含 float 尾部参数的场景）。
// 注意 __builtin_return 是直接从本函数返回，会跳过作用域析构，故此处手动清理 g_cur_comm_key。

namespace {
    // __builtin_apply 的“栈参数区”大小，需覆盖最宽尾部布局（ApiTestParam 296B），留余量到 512B。
    constexpr size_t kHcclVmApplyStackBytes = 512;

    // BuildOpDetailsV1 返回 std::vector<uint8_t>（堆分配）。__builtin_return 是裸返回，会跳过本函数作用域
    // 内所有析构，因此把 blob 的构造/析构关在 helper 内部，保证每次调用都正常释放。
    static int RecordSendOpDetail(
        uint32_t curRank, uint32_t rankSize, uint32_t destRank, HcclComm comm, aclrtStream stream, const void* sendBuf,
        uint64_t count, HcclDataType dataType, uint64_t dataSize)
    {
        const auto details = BuildOpDetailsV1(count, dataType, 0, 0, HcclCMDType::HCCL_CMD_SEND);
        return RecordOpDbInfo(
            HcclCMDType::HCCL_CMD_SEND, curRank, reinterpret_cast<uint64_t>(stream), sendBuf, dataSize, nullptr, 0,
            details, 0, rankSize, curRank, destRank, {}, comm);
    }

    static int RecordRecvOpDetail(
        uint32_t curRank, uint32_t rankSize, uint32_t srcRank, HcclComm comm, aclrtStream stream, const void* recvBuf,
        uint64_t count, HcclDataType dataType, uint64_t dataSize)
    {
        const auto details = BuildOpDetailsV1(count, dataType, 0, 0, HcclCMDType::HCCL_CMD_RECEIVE);
        return RecordOpDbInfo(
            HcclCMDType::HCCL_CMD_RECEIVE, curRank, reinterpret_cast<uint64_t>(stream), nullptr, 0, recvBuf, dataSize,
            details, 0, rankSize, srcRank, curRank, {}, comm);
    }
} // namespace

HcclResult HcclSendCustom(
    void* sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank, HcclComm comm, aclrtStream stream)
{
    // 必须在任何调用之前捕获入参寄存器（GCC 在函数入口保存 rdi..r9/xmm0..7 与栈参数区）
    void* applyArgs = __builtin_apply_args();
    uint32_t curRank = 0;
    uint32_t rankSize = 0;
    const HcclResult setupRet = SetupCommunicator(comm, g_cur_device_key, curRank, rankSize);
    if (setupRet != HcclResult::HCCL_SUCCESS) {
        g_cur_comm_key = 0;
        return setupRet;
    }
    ScopedOperatorCommunicator scopedCommunicator;
    if (stream == nullptr || rankSize == 0 || curRank >= rankSize || destRank >= rankSize || destRank == curRank
        || (count != 0 && sendBuf == nullptr)) {
        HCCL_VM_ERROR(
            "invalid HcclSendCustom parameters: rank={}, rankSize={}, destRank={}", curRank, rankSize, destRank);
        return HcclResult::HCCL_E_PARA;
    }

    uint32_t typeSize = 0;
    if (sim::GetDataTypeSize(dataType, typeSize) != HcclResult::HCCL_SUCCESS) {
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    const uint64_t dataSize = count * typeSize;
    if (RecordSendOpDetail(curRank, rankSize, destRank, comm, stream, sendBuf, count, dataType, dataSize) != 0) {
        return HcclResult::HCCL_E_PARA;
    }

    void* realFunc = sim::DlsymRealWithFallback(__func__, "libhccl.so");
    if (realFunc == nullptr) {
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    // 真实算子在此执行（g_cur_comm_key 仍有效，与改造前一致）
    void* applyRet = __builtin_apply(reinterpret_cast<void (*)(...)>(realFunc), applyArgs, kHcclVmApplyStackBytes);
    // __builtin_return 直接返回，跳过 scopedCommunicator 析构，这里手动补齐其清理动作
    g_cur_comm_key = 0;
    __builtin_return(applyRet);
}

HcclResult HcclRecvCustom(
    void* recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank, HcclComm comm, aclrtStream stream)
{
    void* applyArgs = __builtin_apply_args();
    uint32_t curRank = 0;
    uint32_t rankSize = 0;
    const HcclResult setupRet = SetupCommunicator(comm, g_cur_device_key, curRank, rankSize);
    if (setupRet != HcclResult::HCCL_SUCCESS) {
        g_cur_comm_key = 0;
        return setupRet;
    }
    ScopedOperatorCommunicator scopedCommunicator;
    if (stream == nullptr || rankSize == 0 || curRank >= rankSize || srcRank >= rankSize || srcRank == curRank
        || (count != 0 && recvBuf == nullptr)) {
        HCCL_VM_ERROR(
            "invalid HcclRecvCustom parameters: rank={}, rankSize={}, srcRank={}", curRank, rankSize, srcRank);
        return HcclResult::HCCL_E_PARA;
    }

    uint32_t typeSize = 0;
    if (sim::GetDataTypeSize(dataType, typeSize) != HcclResult::HCCL_SUCCESS) {
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    const uint64_t dataSize = count * typeSize;
    if (RecordRecvOpDetail(curRank, rankSize, srcRank, comm, stream, recvBuf, count, dataType, dataSize) != 0) {
        return HcclResult::HCCL_E_PARA;
    }

    void* realFunc = sim::DlsymRealWithFallback(__func__, "libhccl.so");
    if (realFunc == nullptr) {
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    // 真实算子在此执行（g_cur_comm_key 仍有效，与改造前一致）
    void* applyRet = __builtin_apply(reinterpret_cast<void (*)(...)>(realFunc), applyArgs, kHcclVmApplyStackBytes);
    // __builtin_return 直接返回，跳过 scopedCommunicator 析构，这里手动补齐其清理动作
    g_cur_comm_key = 0;
    __builtin_return(applyRet);
}

#ifdef __cplusplus
}
#endif // __cplusplus
