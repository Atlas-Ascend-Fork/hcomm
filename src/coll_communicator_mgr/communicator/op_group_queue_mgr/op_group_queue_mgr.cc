/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_group_queue_mgr.h"

#include <algorithm>

#include "log.h"
#include "hccl_comm_pub.h"
#include "coll_comm.h"
#include "common/cast_utils_pub.h"

namespace hccl {

std::vector<HcclComm>& OpGroupQueueMgr::GetGroupQueueCommList() { return hcclGroupQueueCommListV2_; }

void OpGroupQueueMgr::ClearGroupQueueCommList() { hcclGroupQueueCommListV2_.clear(); }

int32_t OpGroupQueueMgr::GetGroupQueueTaskNums() { return hcclP2pTaskNums_; }

void OpGroupQueueMgr::SetGroupQueueTaskNums(int32_t targetTaskNums) { hcclP2pTaskNums_ = targetTaskNums; }

OpGroupQueueMgr::~OpGroupQueueMgr() {}

HcclResult OpGroupQueueMgr::AppendGroupOpTask(HcclComm comm, const HcclOpDesc& opDesc, aclrtStream stream, void* func)
{
    if (hcclP2pTaskNums_ >= MAX_GROUP_OP_TASK_NUM) {
        HCCL_ERROR(
            "[%s] comm[%p] hcclP2pTaskNums[%d] reaches the limit[%d].", __func__, comm, hcclP2pTaskNums_,
            MAX_GROUP_OP_TASK_NUM);
        return HCCL_E_INTERNAL;
    }

    GroupOpItem item;
    item.desc = opDesc;
    item.stream = stream;
    item.func = func;
    this->groupOpItems_.push_back(item);
    hcclP2pTaskNums_++;

    auto itComm = std::find(hcclGroupQueueCommListV2_.begin(), hcclGroupQueueCommListV2_.end(), comm);
    if (itComm == hcclGroupQueueCommListV2_.end()) {
        hcclGroupQueueCommListV2_.emplace_back(comm);
    }

    HCCL_INFO(
        "[%s] comm[%p], cmdType[%d], remoteRank[%u], stream[%p], comm queue size[%zu].", __func__, comm,
        static_cast<int32_t>(opDesc.p2p.cmdType), opDesc.p2p.remoteRank, stream, this->groupOpItems_.size());
    return HCCL_SUCCESS;
}

HcclResult OpGroupQueueMgr::LaunchGroupOpTask(HcclComm comm)
{
    if (this->groupOpItems_.empty()) {
        return HCCL_SUCCESS;
    }

    // 同一comm内只允许一种批量执行函数，存在不同func则不支持
    void* func = this->groupOpItems_.front().func;
    for (const auto& item : this->groupOpItems_) {
        if (item.func != func) {
            HCCL_ERROR("[%s] comm[%p] exists different func[%p] vs[%p], not support.", __func__, comm, func, item.func);
            this->ClearGroupOpTask(); // 清空残留任务，避免影响后续group
            return HCCL_E_PARA;
        }
    }

    // 将HcclOpDesc中缓存的send/recv算子信息转换为批量执行函数的入参
    std::vector<HcclSendRecvItem> sendRecvItems;
    sendRecvItems.reserve(this->groupOpItems_.size());
    for (const auto& item : this->groupOpItems_) {
        HcclSendRecvItem sendRecvItem{};
        sendRecvItem.sendRecvType = (item.desc.p2p.cmdType == HcclCMDType::HCCL_CMD_SEND) ? HCCL_SEND : HCCL_RECV;
        sendRecvItem.buf = item.desc.p2p.buffer;
        sendRecvItem.count = item.desc.p2p.count;
        sendRecvItem.dataType = item.desc.p2p.dataType;
        sendRecvItem.remoteRank = item.desc.p2p.remoteRank;
        sendRecvItems.push_back(sendRecvItem);
    }

    // 以组内首条任务入队时指定的stream作为批量执行的流
    aclrtStream stream = this->groupOpItems_.front().stream;
    HCCL_INFO("[%s] comm[%p], func[%p], itemNum[%zu].", __func__, comm, func, sendRecvItems.size());

    // HcclGroupEnd时批量执行函数的签名约定
    using HcclGroupOpFunc
        = HcclResult (*)(HcclSendRecvItem* sendRecvInfo, uint32_t itemNum, HcclComm comm, aclrtStream stream);
    HcclGroupOpFunc execFunc = ReinterpretAs<HcclGroupOpFunc>(func);
    HcclResult ret = execFunc(sendRecvItems.data(), static_cast<uint32_t>(sendRecvItems.size()), comm, stream);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] exec func failed, itemNum[%zu], ret[%d].", __func__, sendRecvItems.size(), ret);
    }
    this->groupOpItems_.clear();
    return ret;
}

void OpGroupQueueMgr::ClearGroupOpTask() { this->groupOpItems_.clear(); }

} // namespace hccl
