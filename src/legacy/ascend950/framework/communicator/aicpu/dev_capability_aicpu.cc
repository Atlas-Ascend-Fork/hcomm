/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 本文件仅供 aicpu（ccl_kernel）构建域使用：与 host 侧 unified_platform/common/dev_capability.cc 的唯一差异是
// Init 在 isInit 早退判断前喂入 workModeAicpu 与全局枚举 devType（参考 coll_comm_aicpu_kernel.cc），
// 使挂起恢复（UpdateComm）路径在 isInit 为 true 时仍能重新喂入。host 侧请勿将本文件编入 hccl_v2。
#include "adapter_rts_common.h"
#include "dev_capability.h"
#include "dev_type_utils.h"
#include "not_support_exception.h"
#include "string_util.h"
namespace Hccl {
using namespace std;
// 域内枚举(Hccl::DevType) → 全局枚举(::DevType) 映射：两套枚举序号与命名不完全一致（如 950(5→6)/960(6→8)/NOSOC(7→5)），
// 全部成员逐一显式映射，未映射枚举打警告并兜底 DEV_TYPE_COUNT；
// 供 hrtSetlocalDeviceType 使用，使 aicpu 侧 hcomm 谓词可经 g_localDeviceType 取数
inline ::DevType MapToGlobalDevType(const DevType& devType)
{
    if (devType == DevType::DEV_TYPE_950) {
        return ::DevType::DEV_TYPE_950;
    }
    if (devType == DevType::DEV_TYPE_960) {
        return ::DevType::DEV_TYPE_960;
    }
    if (devType == DevType::DEV_TYPE_910A) {
        return ::DevType::DEV_TYPE_910;
    }
    if (devType == DevType::DEV_TYPE_910A2) {
        return ::DevType::DEV_TYPE_910B;
    }
    if (devType == DevType::DEV_TYPE_910A3) {
        return ::DevType::DEV_TYPE_910_93;
    }
    if (devType == DevType::DEV_TYPE_V51_310_P3) {
        return ::DevType::DEV_TYPE_310P3;
    }
    if (devType == DevType::DEV_TYPE_V51_310_P1) {
        return ::DevType::DEV_TYPE_310P1;
    }
    if (devType == DevType::DEV_TYPE_NOSOC) {
        return ::DevType::DEV_TYPE_NOSOC;
    }
    HCCL_WARNING(
        "[MapToGlobalDevType] unmapped devType[%s], value[%d].", devType.Describe().c_str(), static_cast<int>(devType));
    return ::DevType::DEV_TYPE_COUNT;
}

const u64 RDMA_SEND_MAX_SIZE = 0x80000000;  // 节点间RDMA发送数据单个WQE支持的最大数据量
const u64 SDMA_SEND_MAX_SIZE = 0x100000000; // 节点内单个SDMA任务发送数据支持的最大数据量
const map<DataType, bool> CAP_INLINE_REDUCE_DATATYPE_910A = {
    {DataType::INT8, false},   {DataType::INT16, false},  {DataType::INT32, false},  {DataType::FP16, false},
    {DataType::FP32, true},    {DataType::INT64, false},  {DataType::UINT64, false}, {DataType::UINT8, false},
    {DataType::UINT16, false}, {DataType::UINT32, false}, {DataType::FP64, false},   {DataType::BFP16, false},
    {DataType::INT128, false},
};
const map<ReduceOp, bool> CAP_INLINE_REDUCE_OP_910A
    = {{ReduceOp::SUM, true}, {ReduceOp::PROD, false}, {ReduceOp::MAX, false}, {ReduceOp::MIN, false}};
const u32 CAP_NOTIFY_SIZE_910A = 8;
const u32 CAP_SDMA_INLINE_REDUCE_ALIGN_BYTES_910A = 128;
const map<DataType, bool> CAP_INLINE_REDUCE_DATATYPE_910A3 = {
    {DataType::INT8, true},    {DataType::INT16, true},   {DataType::INT32, true},   {DataType::FP16, true},
    {DataType::FP32, true},    {DataType::INT64, false},  {DataType::UINT64, false}, {DataType::UINT8, false},
    {DataType::UINT16, false}, {DataType::UINT32, false}, {DataType::FP64, false},   {DataType::BFP16, true},
    {DataType::INT128, false},
};
const map<ReduceOp, bool> CAP_INLINE_REDUCE_OP_910A3
    = {{ReduceOp::SUM, true}, {ReduceOp::PROD, false}, {ReduceOp::MAX, true}, {ReduceOp::MIN, true}};
const u32 CAP_NOTIFY_SIZE_910A3 = 4;
const u32 CAP_SDMA_INLINE_REDUCE_ALIGN_BYTES_910A3 = 32;

const map<DataType, bool> CAP_INLINE_REDUCE_DATATYPE_V82 = {
    {DataType::INT8, true},    {DataType::INT16, true},    {DataType::INT32, true},   {DataType::FP16, true},
    {DataType::FP32, true},    {DataType::INT64, false},   {DataType::UINT64, false}, {DataType::UINT8, true},
    {DataType::UINT16, true},  {DataType::UINT32, true},   {DataType::FP64, false},   {DataType::BFP16, true},
    {DataType::INT128, false}, {DataType::BF16_SAT, true},
};
const map<ReduceOp, bool> CAP_INLINE_REDUCE_OP_V82
    = {{ReduceOp::SUM, true},
       {ReduceOp::PROD, false},
       {ReduceOp::MAX, true},
       {ReduceOp::MIN, true},
       {ReduceOp::EQUAL, true}};
const u32 CAP_NOTIFY_SIZE_V82 = 8;
const u32 CAP_SDMA_INLINE_REDUCE_ALIGN_BYTES_V82 = 32;

DevCapability& DevCapability::GetInstance()
{
    static DevCapability devCapability;
    return devCapability;
}

DevCapability::DevCapability() {}

void DevCapability::Init(DevType givenDevType)
{
    // aicpu（HCCD）场景 Init 前喂入 workModeAicpu 与全局枚举 devType，使后续 hcomm 谓词经 g_localDeviceType 取数
    (void)hrtSetWorkModeAicpu(true);
    (void)hrtSetlocalDeviceType(MapToGlobalDevType(givenDevType));
    if (isInit) {
        return;
    }
    isInit = true;
    devType = givenDevType;
    if (devType == DevType::DEV_TYPE_910A) {
        Load910ACap();
    } else if (devType == DevType::DEV_TYPE_910A3 || devType == DevType::DEV_TYPE_910A2) {
        Load910A3Cap();
    } else if (hcomm::EqualToOrGreaterThan950().value_or(false)) {
        LoadV82Cap();
    } else {
        throw NotSupportException(StringFormat("we don't support %s now.", devType.Describe().c_str()));
    }
}

void DevCapability::Reset() { isInit = false; }

void DevCapability::Load910A910A3CommonCap()
{
    isSupportWriteWithNotify = false;
    isSupportStarsPollNetCq = false;
    sdmaSendMaxSize = SDMA_SEND_MAX_SIZE;
    rdmaSendMaxSize = RDMA_SEND_MAX_SIZE;
}

void DevCapability::Load910A3Cap()
{
    Load910A910A3CommonCap();
    inlineReduceDataTypeMap = CAP_INLINE_REDUCE_DATATYPE_910A3;
    inlineReduceOpMap = CAP_INLINE_REDUCE_OP_910A3;
    sdmaInlineReduceAlignBytes = CAP_SDMA_INLINE_REDUCE_ALIGN_BYTES_910A3;
    notifySize = CAP_NOTIFY_SIZE_910A3;
    isSupportDevNetInlineReduce = true;
}

void DevCapability::Load910ACap()
{
    Load910A910A3CommonCap();
    inlineReduceDataTypeMap = CAP_INLINE_REDUCE_DATATYPE_910A;
    inlineReduceOpMap = CAP_INLINE_REDUCE_OP_910A;
    sdmaInlineReduceAlignBytes = CAP_SDMA_INLINE_REDUCE_ALIGN_BYTES_910A;
    notifySize = CAP_NOTIFY_SIZE_910A;
    isSupportDevNetInlineReduce = false;
}

void DevCapability::LoadV82Cap()
{
    isSupportWriteWithNotify = true;
    isSupportStarsPollNetCq = true;
    sdmaSendMaxSize = SDMA_SEND_MAX_SIZE;
    rdmaSendMaxSize = RDMA_SEND_MAX_SIZE;

    inlineReduceDataTypeMap = CAP_INLINE_REDUCE_DATATYPE_V82;
    inlineReduceOpMap = CAP_INLINE_REDUCE_OP_V82;
    sdmaInlineReduceAlignBytes = CAP_SDMA_INLINE_REDUCE_ALIGN_BYTES_V82;
    notifySize = CAP_NOTIFY_SIZE_V82;
    isSupportDevNetInlineReduce = true;
}

} // namespace Hccl
