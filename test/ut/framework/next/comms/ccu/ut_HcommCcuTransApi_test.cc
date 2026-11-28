/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 搬运类 C 接口(本地拷贝 / 本地 reduce / 远端读写)的端到端用例:
// 在注册流程中逐个调用 C 接口, 校验返回值及每次调用新追加的 rep 类型。
// 依赖 ChannelHandle 的远端搬运接口由 Ut_CcuTransApi_When_RemoteOps_* 用例建链后覆盖正常路径。
#include "ut_HcommCcuControlApi_common.h"

#include <limits>

#include "ccu_log.h"
#include "ccu_primitives_impl.h"
#include "ccu_rep_buflocread_v1.h"
#include "ccu_rep_buflocwrite_v1.h"
#include "ccu_rep_bufreduce_v1.h"
#include "ccu_rep_loccpy_v1.h"

namespace {
namespace CcuRep = hcomm::CcuRep;
using hcomm::CcuRep::CcuRepType;

constexpr uint16_t MOVE_MASK = 1;
constexpr uint32_t BUFFER_NUM = 2;

// 期望编码直接写值, 取自 ccu_assist.cc 的 ccuSumDataTypeMap、ccuReduceTypeMap、ubDataTypeMap、ubReduceTypeMap。
constexpr uint16_t CCU_FP32_CODE = 0;
constexpr uint16_t CCU_FP16_CODE = 1;
constexpr uint16_t CCU_INT8_CODE = 6;
constexpr uint16_t CCU_SUM_CODE = 0;
constexpr uint16_t UB_FP32_CODE = 7;
constexpr uint16_t UB_SUM_CODE = 10;

struct TransOperands {
    CcuVariableHandle len{};
    CcuEventHandle event{};
    CcuBufferHandle buf[BUFFER_NUM]{};
    CcuLocalAddrHandle localSrc{};
    CcuLocalAddrHandle localDst{};
};

void AllocTransOperands(TransOperands& ops)
{
    EXPECT_EQ(CcuVariableAlloc(&ops.len), OK);
    EXPECT_EQ(CcuEventAlloc(&ops.event), OK);
    for (auto& buf : ops.buf) {
        EXPECT_EQ(CcuBufferAlloc(&buf), OK);
    }
    CcuAddressHandle addr{};
    CcuVariableHandle token{};
    EXPECT_EQ(CcuLocalAddrAlloc(&ops.localSrc, &addr, &token), OK);
    EXPECT_EQ(CcuLocalAddrAlloc(&ops.localDst, &addr, &token), OK);
}

CcuResult TransInvalidHandleKernel(CcuKernelArg arg)
{
    (void)arg;
    TransOperands o{};
    AllocTransOperands(o);
    CcuRepCursor reps;

    EXPECT_EQ(CcuLocalCopyMemToMem(INVALID_HANDLE, o.localSrc, o.len, o.event, MOVE_MASK), NOT_FOUND);
    reps.ExpectNone();
    EXPECT_EQ(CcuLocalCopyMemToBuffer(INVALID_HANDLE, o.localSrc, o.len, o.event, MOVE_MASK), NOT_FOUND);
    reps.ExpectNone();
    EXPECT_EQ(CcuLocalCopyBufferToMem(o.localDst, INVALID_HANDLE, o.len, o.event, MOVE_MASK), NOT_FOUND);
    reps.ExpectNone();
    EXPECT_EQ(
        CcuLocalMemReduce(INVALID_HANDLE, o.localSrc, o.len, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, o.event, MOVE_MASK),
        NOT_FOUND);
    reps.ExpectNone();
    // 本 kernel 无建链能力, 以下 ChannelHandle 类接口的 channel 只能传 INVALID_HANDLE,
    // 其 kernel 路径不解析 channel, 故每条用例隔离的是 local/buf 或 remote。
    CcuRemoteAddrHandle remote{};
    CcuAddressHandle remoteAddr{};
    CcuVariableHandle remoteToken{};
    EXPECT_EQ(CcuRemoteAddrAlloc(&remote, &remoteAddr, &remoteToken), OK);

    EXPECT_EQ(CcuReadMemToMem(INVALID_HANDLE, INVALID_HANDLE, remote, o.len, o.event, MOVE_MASK), NOT_FOUND);
    reps.ExpectNone();
    EXPECT_EQ(CcuReadMemToMem(INVALID_HANDLE, o.localDst, INVALID_HANDLE, o.len, o.event, MOVE_MASK), NOT_FOUND);
    reps.ExpectNone();
    EXPECT_EQ(CcuReadMemToBuffer(INVALID_HANDLE, INVALID_HANDLE, remote, o.len, o.event, MOVE_MASK), NOT_FOUND);
    reps.ExpectNone();
    EXPECT_EQ(CcuReadMemToBuffer(INVALID_HANDLE, o.buf[0], INVALID_HANDLE, o.len, o.event, MOVE_MASK), NOT_FOUND);
    reps.ExpectNone();
    EXPECT_EQ(CcuWriteMemToMem(INVALID_HANDLE, INVALID_HANDLE, o.localSrc, o.len, o.event, MOVE_MASK), NOT_FOUND);
    reps.ExpectNone();
    EXPECT_EQ(CcuWriteMemToMem(INVALID_HANDLE, remote, INVALID_HANDLE, o.len, o.event, MOVE_MASK), NOT_FOUND);
    reps.ExpectNone();
    EXPECT_EQ(CcuWriteBufferToMem(INVALID_HANDLE, INVALID_HANDLE, o.buf[0], o.len, o.event, MOVE_MASK), NOT_FOUND);
    reps.ExpectNone();
    EXPECT_EQ(CcuWriteBufferToMem(INVALID_HANDLE, remote, INVALID_HANDLE, o.len, o.event, MOVE_MASK), NOT_FOUND);
    reps.ExpectNone();
    // WriteVarAtomicAdd 的 channelId 是 CcuVariableHandle 并参与解析, 与 addValue 取不同变量,
    // 以下四条依次隔离 channelId、varAddr、addValue、event。
    CcuVariableHandle addValue{};
    EXPECT_EQ(CcuVariableAlloc(&addValue), OK);
    EXPECT_EQ(CcuWriteVarAtomicAdd(INVALID_HANDLE, remote, addValue, o.event, MOVE_MASK), NOT_FOUND);
    reps.ExpectNone();
    EXPECT_EQ(CcuWriteVarAtomicAdd(o.len, INVALID_HANDLE, addValue, o.event, MOVE_MASK), NOT_FOUND);
    reps.ExpectNone();
    EXPECT_EQ(CcuWriteVarAtomicAdd(o.len, remote, INVALID_HANDLE, o.event, MOVE_MASK), NOT_FOUND);
    reps.ExpectNone();
    EXPECT_EQ(CcuWriteVarAtomicAdd(o.len, remote, addValue, INVALID_HANDLE, MOVE_MASK), NOT_FOUND);
    reps.ExpectNone();
    return OK;
}

// 转换表之外的数据类型或 reduce 操作变成 INVALID，应在进入 GetCcuDataType / GetCcuReduceType 之前返回，且不能留下 rep。
CcuResult TransLocalBufferReduceInvalidOpKernel(CcuKernelArg arg)
{
    (void)arg;
    TransOperands o{};
    AllocTransOperands(o);
    CcuRepCursor reps;

    EXPECT_EQ(
        CcuLocalBufferReduce(
            o.buf, BUFFER_NUM, HCCL_DATA_TYPE_FP32, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_RESERVED, o.len, o.event,
            MOVE_MASK),
        CcuResult::CCU_E_PARA);
    reps.ExpectNone();
    EXPECT_EQ(
        CcuLocalBufferReduce(
            o.buf, BUFFER_NUM, HCCL_DATA_TYPE_RESERVED, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, o.len, o.event,
            MOVE_MASK),
        CcuResult::CCU_E_PARA);
    reps.ExpectNone();
    EXPECT_EQ(
        CcuLocalBufferReduce(
            o.buf, BUFFER_NUM, HCCL_DATA_TYPE_FP32, HCCL_DATA_TYPE_RESERVED, HCCL_REDUCE_SUM, o.len, o.event,
            MOVE_MASK),
        CcuResult::CCU_E_PARA);
    reps.ExpectNone();
    // 低精度输入配非低精度输出。
    EXPECT_EQ(
        CcuLocalBufferReduce(
            o.buf, BUFFER_NUM, HCCL_DATA_TYPE_INT8, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, o.len, o.event, MOVE_MASK),
        CcuResult::CCU_E_NOT_SUPPORT);
    reps.ExpectNone();
    // SUM，输入输出类型不同。
    EXPECT_EQ(
        CcuLocalBufferReduce(
            o.buf, BUFFER_NUM, HCCL_DATA_TYPE_FP32, HCCL_DATA_TYPE_FP16, HCCL_REDUCE_SUM, o.len, o.event, MOVE_MASK),
        CcuResult::CCU_E_NOT_SUPPORT);
    reps.ExpectNone();
    // 非 SUM，输入输出类型不同。
    EXPECT_EQ(
        CcuLocalBufferReduce(
            o.buf, BUFFER_NUM, HCCL_DATA_TYPE_FP32, HCCL_DATA_TYPE_FP16, HCCL_REDUCE_MAX, o.len, o.event, MOVE_MASK),
        CcuResult::CCU_E_NOT_SUPPORT);
    reps.ExpectNone();
    // 合法枚举、同类型，但 reduce 操作在微码支持表之外。
    EXPECT_EQ(
        CcuLocalBufferReduce(
            o.buf, BUFFER_NUM, HCCL_DATA_TYPE_FP32, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_PROD, o.len, o.event, MOVE_MASK),
        CcuResult::CCU_E_NOT_SUPPORT);
    reps.ExpectNone();
    // 合法枚举、同类型，但数据类型在 SUM 支持表之外。
    EXPECT_EQ(
        CcuLocalBufferReduce(
            o.buf, BUFFER_NUM, HCCL_DATA_TYPE_UINT16, HCCL_DATA_TYPE_UINT16, HCCL_REDUCE_SUM, o.len, o.event,
            MOVE_MASK),
        CcuResult::CCU_E_NOT_SUPPORT);
    reps.ExpectNone();
    // 合法枚举、同类型，但数据类型在 MAX/MIN 支持表之外。
    EXPECT_EQ(
        CcuLocalBufferReduce(
            o.buf, BUFFER_NUM, HCCL_DATA_TYPE_INT64, HCCL_DATA_TYPE_INT64, HCCL_REDUCE_MAX, o.len, o.event, MOVE_MASK),
        CcuResult::CCU_E_NOT_SUPPORT);
    reps.ExpectNone();
    // count 超过 rep 侧 uint16_t 上限。
    EXPECT_EQ(
        CcuLocalBufferReduce(
            o.buf, static_cast<uint32_t>(std::numeric_limits<uint16_t>::max()) + 1, HCCL_DATA_TYPE_FP32,
            HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, o.len, o.event, MOVE_MASK),
        CcuResult::CCU_E_PARA);
    reps.ExpectNone();
    return OK;
}

// 取句柄对应资源的 rep 侧 Id, 用于核对接口是否把实参放进了 rep 的正确位置。
uint16_t LocalAddrIdOf(hcomm::CcuKernel* kernel, CcuLocalAddrHandle handle)
{
    hcomm::CcuRep::LocalAddr* local{nullptr};
    EXPECT_EQ(kernel->Repo().ResolveLocalAddr(handle, &local), OK);
    return local == nullptr ? 0 : local->addr.Id();
}

uint16_t RemoteAddrIdOf(hcomm::CcuKernel* kernel, CcuRemoteAddrHandle handle)
{
    hcomm::CcuRep::RemoteAddr* remote{nullptr};
    EXPECT_EQ(kernel->Repo().ResolveRemoteAddr(handle, &remote), OK);
    return remote == nullptr ? 0 : remote->addr.Id();
}

uint16_t BufferIdOf(hcomm::CcuKernel* kernel, CcuBufferHandle handle)
{
    hcomm::CcuRep::CcuBuf* buf{nullptr};
    EXPECT_EQ(kernel->Repo().ResolveBuffer(handle, &buf), OK);
    return buf == nullptr ? 0 : buf->Id();
}

// 本地搬运的地址句柄都是 uint64_t，实参写串编译期发现不了，逐个核对 rep 上的操作数 Id。
CcuResult TransLocalOpsKernel(CcuKernelArg arg)
{
    (void)arg;
    TransOperands o{};
    AllocTransOperands(o);
    CcuRepCursor reps;
    auto* kernel = reps.Kernel();
    const uint16_t localDstId = LocalAddrIdOf(kernel, o.localDst);
    const uint16_t localSrcId = LocalAddrIdOf(kernel, o.localSrc);
    const uint16_t bufId = BufferIdOf(kernel, o.buf[0]);
    const uint16_t lenId = VariableIdOf(kernel, o.len);
    const uint16_t eventId = EventIdOf(kernel, o.event);

    EXPECT_EQ(CcuLocalCopyMemToMem(o.localDst, o.localSrc, o.len, o.event, MOVE_MASK), OK);
    if (auto copy = TakeRep<CcuRep::CcuRepLocCpy>(reps, CcuRepType::LOCAL_CPY)) {
        EXPECT_EQ(copy->GetDstAddrId(), localDstId);
        EXPECT_EQ(copy->GetSrcAddrId(), localSrcId);
        EXPECT_EQ(copy->GetLenId(), lenId);
        EXPECT_EQ(copy->GetSemId(), eventId);
    }

    EXPECT_EQ(CcuLocalCopyMemToBuffer(o.buf[0], o.localSrc, o.len, o.event, MOVE_MASK), OK);
    if (auto read = TakeRep<CcuRep::CcuRepBufLocRead>(reps, CcuRepType::BUF_LOC_READ)) {
        EXPECT_EQ(read->GetSrcAddrId(), localSrcId);
        EXPECT_EQ(read->GetDstId(), bufId);
        EXPECT_EQ(read->GetLenId(), lenId);
        EXPECT_EQ(read->GetSemId(), eventId);
    }

    EXPECT_EQ(CcuLocalCopyBufferToMem(o.localDst, o.buf[0], o.len, o.event, MOVE_MASK), OK);
    if (auto write = TakeRep<CcuRep::CcuRepBufLocWrite>(reps, CcuRepType::BUF_LOC_WRITE)) {
        EXPECT_EQ(write->GetSrcAddrId(), bufId);
        EXPECT_EQ(write->GetDstAddrId(), localDstId);
        EXPECT_EQ(write->GetLenId(), lenId);
        EXPECT_EQ(write->GetSemId(), eventId);
    }

    EXPECT_EQ(
        CcuLocalMemReduce(o.localDst, o.localSrc, o.len, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, o.event, MOVE_MASK), OK);
    if (auto reduce = TakeRep<CcuRep::CcuRepLocCpy>(reps, CcuRepType::LOCAL_REDUCE)) {
        EXPECT_EQ(reduce->GetDstAddrId(), localDstId);
        EXPECT_EQ(reduce->GetSrcAddrId(), localSrcId);
        EXPECT_EQ(reduce->GetLenId(), lenId);
        EXPECT_EQ(reduce->GetSemId(), eventId);
        EXPECT_EQ(reduce->GetDataType(), UB_FP32_CODE);
        EXPECT_EQ(reduce->GetOpType(), UB_SUM_CODE);
    }
    return OK;
}

// 合法组合生成 BUF_REDUCE。同类型核对操作数；低精度输入配低精度输出核对两种类型编码。
CcuResult TransLocalBufferReduceKernel(CcuKernelArg arg)
{
    (void)arg;
    TransOperands o{};
    AllocTransOperands(o);
    CcuRepCursor reps;
    auto* kernel = reps.Kernel();
    const uint16_t lenId = VariableIdOf(kernel, o.len);
    const uint16_t eventId = EventIdOf(kernel, o.event);
    const uint16_t buf0 = BufferIdOf(kernel, o.buf[0]);
    const uint16_t buf1 = BufferIdOf(kernel, o.buf[1]);

    EXPECT_EQ(
        CcuLocalBufferReduce(
            o.buf, BUFFER_NUM, HCCL_DATA_TYPE_FP32, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, o.len, o.event, MOVE_MASK),
        OK);
    if (auto reduce = TakeRep<CcuRep::CcuRepBufReduce>(reps, CcuRepType::BUF_REDUCE)) {
        EXPECT_EQ(reduce->GetCount(), static_cast<uint16_t>(BUFFER_NUM));
        EXPECT_EQ(reduce->GetMem().size(), static_cast<size_t>(BUFFER_NUM));
        if (reduce->GetMem().size() == BUFFER_NUM) {
            EXPECT_EQ(reduce->GetMem()[0].Id(), buf0);
            EXPECT_EQ(reduce->GetMem()[1].Id(), buf1);
        }
        EXPECT_EQ(reduce->GetDataType(), CCU_FP32_CODE);
        EXPECT_EQ(reduce->GetOutputDataType(), CCU_FP32_CODE);
        EXPECT_EQ(reduce->GetOpType(), CCU_SUM_CODE);
        EXPECT_EQ(reduce->GetXnLengthId(), lenId);
        EXPECT_EQ(reduce->GetSemId(), eventId);
        EXPECT_EQ(reduce->GetMask(), MOVE_MASK);
    }

    EXPECT_EQ(
        CcuLocalBufferReduce(
            o.buf, BUFFER_NUM, HCCL_DATA_TYPE_INT8, HCCL_DATA_TYPE_FP16, HCCL_REDUCE_SUM, o.len, o.event, MOVE_MASK),
        OK);
    if (auto reduce = TakeRep<CcuRep::CcuRepBufReduce>(reps, CcuRepType::BUF_REDUCE)) {
        EXPECT_EQ(reduce->GetDataType(), CCU_INT8_CODE);
        EXPECT_EQ(reduce->GetOutputDataType(), CCU_FP16_CODE);
    }
    return OK;
}

// 远端搬运接口的正常路径: 通道由用例建链后经 kernelArg 传入。
// 本地侧与远端侧句柄都是 uint64_t, 实参写串编译期发现不了, 故逐个核对 rep 上的操作数 Id。
CcuResult TransRemoteOpsKernel(CcuKernelArg arg)
{
    const auto channel = static_cast<CcuChannelKernelArg*>(arg)->channel;
    TransOperands o{};
    AllocTransOperands(o);
    CcuRemoteAddrHandle remote{};
    CcuAddressHandle remoteAddr{};
    CcuVariableHandle remoteToken{};
    EXPECT_EQ(CcuRemoteAddrAlloc(&remote, &remoteAddr, &remoteToken), OK);

    CcuRepCursor reps;
    auto* kernel = reps.Kernel();
    const uint16_t remoteId = RemoteAddrIdOf(kernel, remote);
    const uint16_t localDstId = LocalAddrIdOf(kernel, o.localDst);
    const uint16_t localSrcId = LocalAddrIdOf(kernel, o.localSrc);
    const uint16_t bufId = BufferIdOf(kernel, o.buf[0]);
    const uint16_t lenId = VariableIdOf(kernel, o.len);
    const uint16_t eventId = EventIdOf(kernel, o.event);

    EXPECT_EQ(CcuReadMemToMem(channel, o.localDst, remote, o.len, o.event, MOVE_MASK), OK);
    if (auto read = TakeRep<CcuRep::CcuRepRead>(reps, CcuRepType::READ)) {
        EXPECT_EQ(read->GetChannel(), channel);
        EXPECT_EQ(read->GetLocAddrId(), localDstId);
        EXPECT_EQ(read->GetRemAddrId(), remoteId);
        EXPECT_EQ(read->GetLenId(), lenId);
        EXPECT_EQ(read->GetSemId(), eventId);
        EXPECT_EQ(read->GetMask(), MOVE_MASK);
    }

    EXPECT_EQ(CcuReadMemToBuffer(channel, o.buf[0], remote, o.len, o.event, MOVE_MASK), OK);
    if (auto bufRead = TakeRep<CcuRep::CcuRepBufRead>(reps, CcuRepType::BUF_READ)) {
        EXPECT_EQ(bufRead->GetChannel(), channel);
        EXPECT_EQ(bufRead->GetSrcAddrId(), remoteId);
        EXPECT_EQ(bufRead->GetDstAddrId(), bufId);
        EXPECT_EQ(bufRead->GetLenId(), lenId);
        EXPECT_EQ(bufRead->GetSemId(), eventId);
    }

    EXPECT_EQ(
        CcuReadMemToMemReduce(
            channel, o.localDst, remote, o.len, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, o.event, MOVE_MASK),
        OK);
    if (auto readReduce = TakeRep<CcuRep::CcuRepRead>(reps, CcuRepType::READ)) {
        EXPECT_EQ(readReduce->GetLocAddrId(), localDstId);
        EXPECT_EQ(readReduce->GetRemAddrId(), remoteId);
        EXPECT_EQ(readReduce->GetLenId(), lenId);
        EXPECT_EQ(readReduce->GetSemId(), eventId);
        EXPECT_EQ(readReduce->GetDataType(), UB_FP32_CODE);
        EXPECT_EQ(readReduce->GetOpType(), UB_SUM_CODE);
    }

    EXPECT_EQ(CcuWriteMemToMem(channel, remote, o.localSrc, o.len, o.event, MOVE_MASK), OK);
    if (auto write = TakeRep<CcuRep::CcuRepWrite>(reps, CcuRepType::WRITE)) {
        EXPECT_EQ(write->GetChannel(), channel);
        EXPECT_EQ(write->GetLocAddrId(), localSrcId);
        EXPECT_EQ(write->GetRemAddrId(), remoteId);
        EXPECT_EQ(write->GetLenId(), lenId);
        EXPECT_EQ(write->GetSemId(), eventId);
    }

    EXPECT_EQ(CcuWriteBufferToMem(channel, remote, o.buf[0], o.len, o.event, MOVE_MASK), OK);
    if (auto bufWrite = TakeRep<CcuRep::CcuRepBufWrite>(reps, CcuRepType::BUF_WRITE)) {
        EXPECT_EQ(bufWrite->GetSrcId(), bufId);
        EXPECT_EQ(bufWrite->GetDstAddrId(), remoteId);
        EXPECT_EQ(bufWrite->GetLenId(), lenId);
        EXPECT_EQ(bufWrite->GetSemId(), eventId);
    }

    EXPECT_EQ(
        CcuWriteMemToMemReduce(
            channel, remote, o.localSrc, o.len, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, o.event, MOVE_MASK),
        OK);
    if (auto writeReduce = TakeRep<CcuRep::CcuRepWrite>(reps, CcuRepType::WRITE)) {
        EXPECT_EQ(writeReduce->GetLocAddrId(), localSrcId);
        EXPECT_EQ(writeReduce->GetRemAddrId(), remoteId);
        EXPECT_EQ(writeReduce->GetLenId(), lenId);
        EXPECT_EQ(writeReduce->GetSemId(), eventId);
        EXPECT_EQ(writeReduce->GetDataType(), UB_FP32_CODE);
        EXPECT_EQ(writeReduce->GetOpType(), UB_SUM_CODE);
    }
    return OK;
}

// WriteVarAtomicAdd 的 channelId 与 addValue 同为 CcuVariableHandle, 逐个核对 rep 上的操作数 Id。
CcuResult TransWriteVarAtomicKernel(CcuKernelArg arg)
{
    (void)arg;
    TransOperands o{};
    AllocTransOperands(o);
    CcuRemoteAddrHandle remote{};
    CcuAddressHandle remoteAddr{};
    CcuVariableHandle remoteToken{};
    EXPECT_EQ(CcuRemoteAddrAlloc(&remote, &remoteAddr, &remoteToken), OK);
    CcuVariableHandle addValue{};
    EXPECT_EQ(CcuVariableAlloc(&addValue), OK);

    CcuRepCursor reps;
    auto* kernel = reps.Kernel();
    const uint16_t channelVarId = VariableIdOf(kernel, o.len);
    const uint16_t remoteId = RemoteAddrIdOf(kernel, remote);
    const uint16_t addValueId = VariableIdOf(kernel, addValue);
    const uint16_t eventId = EventIdOf(kernel, o.event);

    EXPECT_EQ(CcuWriteVarAtomicAdd(o.len, remote, addValue, o.event, MOVE_MASK), OK);
    if (auto atomic = TakeRep<CcuRep::CcuRepWriteVarAtomic>(reps, CcuRepType::WRITE_VAR_ATOMIC)) {
        EXPECT_EQ(atomic->GetChannelVarId(), channelVarId);
        EXPECT_EQ(atomic->GetVarAddrId(), remoteId);
        EXPECT_EQ(atomic->GetTargetId(), addValueId);
        EXPECT_EQ(atomic->GetSemId(), eventId);
        EXPECT_EQ(atomic->GetMask(), MOVE_MASK);
    }
    return OK;
}

// 远端 reduce 的非法 dataType/opType 路径: 通道合法, UB 类型查表失败时不留 rep, 也不把通道登记进 kernel。
CcuResult TransRemoteReduceInvalidTypeKernel(CcuKernelArg arg)
{
    const auto channel = static_cast<CcuChannelKernelArg*>(arg)->channel;
    TransOperands o{};
    AllocTransOperands(o);
    CcuRemoteAddrHandle remote{};
    CcuAddressHandle remoteAddr{};
    CcuVariableHandle remoteToken{};
    EXPECT_EQ(CcuRemoteAddrAlloc(&remote, &remoteAddr, &remoteToken), OK);

    CcuRepCursor reps;
    auto* kernel = reps.Kernel();

    EXPECT_EQ(
        CcuReadMemToMemReduce(
            channel, o.localDst, remote, o.len, HCCL_DATA_TYPE_RESERVED, HCCL_REDUCE_SUM, o.event, MOVE_MASK),
        CcuResult::CCU_E_PARA);
    reps.ExpectNone();
    EXPECT_EQ(
        CcuReadMemToMemReduce(
            channel, o.localDst, remote, o.len, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_RESERVED, o.event, MOVE_MASK),
        CcuResult::CCU_E_PARA);
    reps.ExpectNone();
    EXPECT_EQ(
        CcuWriteMemToMemReduce(
            channel, remote, o.localSrc, o.len, HCCL_DATA_TYPE_RESERVED, HCCL_REDUCE_SUM, o.event, MOVE_MASK),
        CcuResult::CCU_E_PARA);
    reps.ExpectNone();
    EXPECT_EQ(
        CcuWriteMemToMemReduce(
            channel, remote, o.localSrc, o.len, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_RESERVED, o.event, MOVE_MASK),
        CcuResult::CCU_E_PARA);
    reps.ExpectNone();
    EXPECT_TRUE(kernel->GetChannels().empty());
    return OK;
}
} // namespace

CCU_FUNC_KERNEL_V2_TEST(Ut_CcuTransApi_When_LocalOps_Expect_MatchingRep, TransLocalOpsKernel, true)
CCU_FUNC_KERNEL_V2_TEST(Ut_CcuTransApi_When_InvalidHandle_Expect_NotFoundAndNoRep, TransInvalidHandleKernel, true)
CCU_FUNC_KERNEL_V2_TEST(
    Ut_CcuTransApi_When_LocalBufferReduceInvalidOp_Expect_ParaAndNoRep, TransLocalBufferReduceInvalidOpKernel, true)
CCU_FUNC_KERNEL_V2_TEST(
    Ut_CcuTransApi_When_LocalBufferReduce_Expect_BufReduceOperands, TransLocalBufferReduceKernel, true)
CCU_FUNC_KERNEL_V2_TEST(
    Ut_CcuTransApi_When_WriteVarAtomicAdd_Expect_MatchingRepOperands, TransWriteVarAtomicKernel, true)

TEST_F(HcommCcuControlApiTest, Ut_CcuTransApi_When_RemoteOps_Expect_MatchingRepOperands)
{
    RunKernelWithChannel("TransRemoteOpsKernel", TransRemoteOpsKernel);
}

TEST_F(HcommCcuControlApiTest, Ut_CcuTransApi_When_RemoteReduceInvalidType_Expect_ParaAndNoChannel)
{
    RunKernelWithChannel("TransRemoteReduceInvalidTypeKernel", TransRemoteReduceInvalidTypeKernel);
}
