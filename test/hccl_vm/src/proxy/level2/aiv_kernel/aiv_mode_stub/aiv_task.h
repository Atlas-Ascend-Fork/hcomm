/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_AIV_TASK_H
#define AIV_AIV_TASK_H

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "aiv_mode_stub_base.h"
#include "ascendc_base_stub.h"

namespace AivSim {
using flag_t = int32_t;

enum class AivTaskType : uint16_t {
    MEM_COPY = 0,
    REDUCE,
    SET_FLAG,
    WAIT_FLAG,
    PIPE_BARRIER,
    SYNC_ALL,
    SEND_FLAG,
    RECV_FLAG,
    INVALID_TYPE,
};

std::string GetTypeName(AivTaskType taskType);

class AivTask {
public:
    explicit AivTask(AivTaskType type) : type_(type) {}
    AivTask(
        AivTaskType type, TaskId taskId, RankId rankId, BlockId blockId, AscendC::pipe_t curPipe, uint64_t commId = 0,
        DeviceId deviceId = UINT32_MAX)
        : type_(type),
          taskId_(taskId),
          rankId_(rankId),
          blockId_(blockId),
          curPipe_(curPipe),
          commId_(commId),
          deviceId_(deviceId)
    {}
    virtual ~AivTask() {}
    AivTask(const AivTask&) = delete;
    AivTask& operator=(const AivTask&) = delete;

    virtual uint64_t GetUUID() const { return (static_cast<uint64_t>(rankId_) << 32) | taskId_; }

    virtual std::string Describe() const
    {
        std::stringstream ss;
        ss << "[" << GetTypeName(type_) << "] "
           << "TaskId=" << taskId_ << ", CommId=" << commId_ << ", RankId=" << rankId_ << ", DeviceId=" << deviceId_
           << ", BlockId=" << blockId_ << ", OnPipe=" << AscendC::GetPipeName(curPipe_);
        return ss.str();
    }

    AivTaskType GetTaskType() const { return type_; }
    void SetTaskType(AivTaskType type) { type_ = type; }
    TaskId GetTaskId() const { return taskId_; }
    void SetTaskId(TaskId taskId) { taskId_ = taskId; }
    RankId GetRankId() const { return rankId_; }
    void SetRankId(RankId rankId) { rankId_ = rankId; }
    BlockId GetBlockId() const { return blockId_; }
    void SetBlockId(BlockId blockId) { blockId_ = blockId; }
    AscendC::pipe_t GetCurPipe() const { return curPipe_; }
    void SetCurPipe(AscendC::pipe_t curPipe) { curPipe_ = curPipe; }
    uint64_t GetCommId() const { return commId_; }
    void SetCommId(uint64_t commId) { commId_ = commId; }
    DeviceId GetDeviceId() const { return deviceId_; }
    void SetDeviceId(DeviceId deviceId) { deviceId_ = deviceId; }

private:
    AivTaskType type_{AivTaskType::INVALID_TYPE};

    TaskId taskId_{UINT32_MAX};
    RankId rankId_{UINT32_MAX};
    DeviceId deviceId_{UINT32_MAX};
    uint64_t commId_{0};
    BlockId blockId_{UINT32_MAX};
    AscendC::pipe_t curPipe_{AscendC::pipe_t::PIPE_ALL};
};

class AivTaskMemCopy : public AivTask {
public:
    AivTaskMemCopy(const AivDataSlice& src, const AivDataSlice& dst)
        : AivTask(AivTaskType::MEM_COPY),
          src_(src),
          dst_(dst)
    {}
    ~AivTaskMemCopy() override {}
    AivTaskMemCopy(const AivTaskMemCopy&) = delete;
    AivTaskMemCopy& operator=(const AivTaskMemCopy&) = delete;

    std::string Describe() const override
    {
        std::stringstream ss;
        ss << AivTask::Describe() << ", Src=" << src_.Describe() << ", Dst=" << dst_.Describe();
        return ss.str();
    }

    const AivDataSlice& GetSrc() const { return src_; }
    void SetSrc(const AivDataSlice& src) { src_ = src; }
    const AivDataSlice& GetDst() const { return dst_; }
    void SetDst(const AivDataSlice& dst) { dst_ = dst; }

private:
    AivDataSlice src_{};
    AivDataSlice dst_{};
};

class AivTaskReduce : public AivTask {
public:
    AivTaskReduce(const AivDataSlice& src, const AivDataSlice& dst, uint32_t dataType, uint32_t reduceOp)
        : AivTask(AivTaskType::REDUCE),
          src_(src),
          dst_(dst),
          dataType_(dataType),
          reduceOp_(reduceOp)
    {}
    ~AivTaskReduce() override {}
    AivTaskReduce(const AivTaskReduce&) = delete;
    AivTaskReduce& operator=(const AivTaskReduce&) = delete;

    std::string Describe() const override
    {
        std::stringstream ss;
        ss << AivTask::Describe() << ", Src=" << src_.Describe() << ", Dst=" << dst_.Describe()
           << ", DataType=" << dataType_ << ", ReduceOp=" << GetReduceOpName(static_cast<ReduceOp>(reduceOp_));
        return ss.str();
    }

    const AivDataSlice& GetSrc() const { return src_; }
    void SetSrc(const AivDataSlice& src) { src_ = src; }
    const AivDataSlice& GetDst() const { return dst_; }
    void SetDst(const AivDataSlice& dst) { dst_ = dst; }
    uint32_t GetDataType() const { return dataType_; }
    void SetDataType(uint32_t dataType) { dataType_ = dataType; }
    uint32_t GetReduceOp() const { return reduceOp_; }
    void SetReduceOp(uint32_t reduceOp) { reduceOp_ = reduceOp; }

private:
    AivDataSlice src_{};
    AivDataSlice dst_{};

    uint32_t dataType_{0};
    uint32_t reduceOp_{0};
};

class AivTaskSetFlag : public AivTask {
public:
    AivTaskSetFlag(AscendC::pipe_t srcPipe, AscendC::pipe_t dstPipe, int32_t eventId)
        : AivTask(AivTaskType::SET_FLAG),
          srcPipe_(srcPipe),
          dstPipe_(dstPipe),
          eventId_(eventId)
    {}
    ~AivTaskSetFlag() override {}
    AivTaskSetFlag(const AivTaskSetFlag&) = delete;
    AivTaskSetFlag& operator=(const AivTaskSetFlag&) = delete;

    std::string Describe() const override
    {
        std::stringstream ss;
        ss << AivTask::Describe() << ", SrcPipe=" << AscendC::GetPipeName(srcPipe_)
           << ", DstPipe=" << AscendC::GetPipeName(dstPipe_) << ", EventId=" << eventId_;
        return ss.str();
    }

    AscendC::pipe_t GetSrcPipe() const { return srcPipe_; }
    void SetSrcPipe(AscendC::pipe_t srcPipe) { srcPipe_ = srcPipe; }
    AscendC::pipe_t GetDstPipe() const { return dstPipe_; }
    void SetDstPipe(AscendC::pipe_t dstPipe) { dstPipe_ = dstPipe; }
    int32_t GetEventId() const { return eventId_; }
    void SetEventId(int32_t eventId) { eventId_ = eventId; }

private:
    AscendC::pipe_t srcPipe_{AscendC::pipe_t::PIPE_ALL};
    AscendC::pipe_t dstPipe_{AscendC::pipe_t::PIPE_ALL};
    int32_t eventId_{0};
};

class AivTaskWaitFlag : public AivTask {
public:
    AivTaskWaitFlag(AscendC::pipe_t srcPipe, AscendC::pipe_t dstPipe, int32_t eventId)
        : AivTask(AivTaskType::WAIT_FLAG),
          srcPipe_(srcPipe),
          dstPipe_(dstPipe),
          eventId_(eventId)
    {}
    ~AivTaskWaitFlag() override {}
    AivTaskWaitFlag(const AivTaskWaitFlag&) = delete;
    AivTaskWaitFlag& operator=(const AivTaskWaitFlag&) = delete;

    std::string Describe() const override
    {
        std::stringstream ss;
        ss << AivTask::Describe() << ", SrcPipe=" << AscendC::GetPipeName(srcPipe_)
           << ", DstPipe=" << AscendC::GetPipeName(dstPipe_) << ", EventId=" << eventId_;
        return ss.str();
    }

    AscendC::pipe_t GetSrcPipe() const { return srcPipe_; }
    void SetSrcPipe(AscendC::pipe_t srcPipe) { srcPipe_ = srcPipe; }
    AscendC::pipe_t GetDstPipe() const { return dstPipe_; }
    void SetDstPipe(AscendC::pipe_t dstPipe) { dstPipe_ = dstPipe; }
    int32_t GetEventId() const { return eventId_; }
    void SetEventId(int32_t eventId) { eventId_ = eventId; }

private:
    AscendC::pipe_t srcPipe_{AscendC::pipe_t::PIPE_ALL};
    AscendC::pipe_t dstPipe_{AscendC::pipe_t::PIPE_ALL};
    int32_t eventId_{0};
};

class AivTaskPipeBarrier : public AivTask {
public:
    explicit AivTaskPipeBarrier(AscendC::pipe_t pipeType) : AivTask(AivTaskType::PIPE_BARRIER), pipeType_(pipeType) {}
    ~AivTaskPipeBarrier() override {}
    AivTaskPipeBarrier(const AivTaskPipeBarrier&) = delete;
    AivTaskPipeBarrier& operator=(const AivTaskPipeBarrier&) = delete;

    std::string Describe() const override
    {
        std::stringstream ss;
        ss << AivTask::Describe() << ", PipeType=" << AscendC::GetPipeName(pipeType_) << ", BarrierGroup=";
        ss << "[";
        for (size_t i = 0; i < barrierGroup_.size(); ++i) {
            if (i != 0) {
                ss << ",";
            }
            ss << barrierGroup_[i]->GetTaskId();
        }
        ss << "]";
        return ss.str();
    }

    AscendC::pipe_t GetPipeType() const { return pipeType_; }
    void SetPipeType(AscendC::pipe_t pipeType) { pipeType_ = pipeType; }
    const std::vector<std::shared_ptr<AivTaskPipeBarrier>>& GetBarrierGroup() const { return barrierGroup_; }
    void AddBarrierGroup(const std::shared_ptr<AivTaskPipeBarrier>& task) { barrierGroup_.push_back(task); }

private:
    AscendC::pipe_t pipeType_{AscendC::pipe_t::PIPE_ALL};
    std::vector<std::shared_ptr<AivTaskPipeBarrier>> barrierGroup_{};
};

class AivTaskSyncAll : public AivTask {
public:
    explicit AivTaskSyncAll(uint32_t round) : AivTask(AivTaskType::SYNC_ALL), syncRound_(round) {}
    ~AivTaskSyncAll() override {}
    AivTaskSyncAll(const AivTaskSyncAll&) = delete;
    AivTaskSyncAll& operator=(const AivTaskSyncAll&) = delete;

    uint32_t GetSyncRound() const { return syncRound_; }

    std::string Describe() const override
    {
        std::stringstream ss;
        ss << AivTask::Describe() << ", SyncRound=" << syncRound_;
        return ss.str();
    }

private:
    uint32_t syncRound_{UINT32_MAX};
};

class AivTaskSendFlag : public AivTask {
public:
    AivTaskSendFlag(RankId targetRank, const AivDataSlice& flagBuffer, flag_t flagValue)
        : AivTask(AivTaskType::SEND_FLAG),
          targetRank_(targetRank),
          flagBuffer_(flagBuffer),
          flagValue_(flagValue)
    {}
    ~AivTaskSendFlag() override {}
    AivTaskSendFlag(const AivTaskSendFlag&) = delete;
    AivTaskSendFlag& operator=(const AivTaskSendFlag&) = delete;

    std::string Describe() const override
    {
        std::stringstream ss;
        ss << AivTask::Describe() << ", TargetRank=" << targetRank_ << ", FlagBuffer=" << flagBuffer_.Describe()
           << ", Value=" << flagValue_;
        return ss.str();
    }

    RankId GetTargetRank() const { return targetRank_; }
    void SetTargetRank(RankId targetRank) { targetRank_ = targetRank; }
    const AivDataSlice& GetFlagBuffer() const { return flagBuffer_; }
    void SetFlagBuffer(const AivDataSlice& flagBuffer) { flagBuffer_ = flagBuffer; }
    flag_t GetFlagValue() const { return flagValue_; }
    void SetFlagValue(flag_t flagValue) { flagValue_ = flagValue; }

private:
    RankId targetRank_{UINT32_MAX};
    AivDataSlice flagBuffer_{};
    flag_t flagValue_{0};
};

class AivTaskRecvFlag : public AivTask {
public:
    AivTaskRecvFlag(RankId targetRank, const AivDataSlice& flagBuffer, flag_t flagValue)
        : AivTask(AivTaskType::RECV_FLAG),
          targetRank_(targetRank),
          flagBuffer_(flagBuffer),
          flagValue_(flagValue)
    {}
    ~AivTaskRecvFlag() override {}
    AivTaskRecvFlag(const AivTaskRecvFlag&) = delete;
    AivTaskRecvFlag& operator=(const AivTaskRecvFlag&) = delete;

    std::string Describe() const override
    {
        std::stringstream ss;
        ss << AivTask::Describe() << ", TargetRank=" << targetRank_ << ", FlagBuffer=" << flagBuffer_.Describe()
           << ", Value=" << flagValue_;
        return ss.str();
    }

    RankId GetTargetRank() const { return targetRank_; }
    void SetTargetRank(RankId targetRank) { targetRank_ = targetRank; }
    const AivDataSlice& GetFlagBuffer() const { return flagBuffer_; }
    void SetFlagBuffer(const AivDataSlice& flagBuffer) { flagBuffer_ = flagBuffer; }
    flag_t GetFlagValue() const { return flagValue_; }
    void SetFlagValue(flag_t flagValue) { flagValue_ = flagValue; }

private:
    RankId targetRank_{UINT32_MAX};
    AivDataSlice flagBuffer_{};
    flag_t flagValue_{0};
};
} // namespace AivSim

#endif // AIV_AIV_TASK_H
