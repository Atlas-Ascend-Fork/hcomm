/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_KERNEL_SIM_H
#define CCU_KERNEL_SIM_H

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "ccu_types.h"
#include "sim_common_defs.h"
#include "ccu_level1_common.h" // 公共类型：SimCcuHandleTag / SimCcuPrim / SimCcuKernelEntry / ToCcuResult 等

// 与 hcomm 仓 pub_inc/ccu_kernel_func.h 中的官方函数指针类型保持一致
// （CcuResult (*)() 与 CcuResult (*)(CcuKernelArg)），仅作类型引用，不改动签名。
typedef CcuResult (*SimCcuKernelFuncNoArg)();
typedef CcuResult (*SimCcuKernelFuncOneArg)(CcuKernelArg arg);

namespace HcclSim {
namespace CcuSim {

    /*==================== 录制期查询 / 句柄分配 ====================*/

    /** @brief 当前线程是否正处于注册期录制（kernelFunc 执行期）。 */
    bool IsRecording();

    /** @brief 当前线程是否正处于 Launch 回放期。 */
    bool IsExecuting();

    /** @brief 当前执行上下文的 rank（未执行时返回 UINT32_MAX）。 */
    uint32_t CurRankId();

    /**
     * @brief 分配原语句柄（录制上下文 TLS 内签发，kernel 命名空间编码）。
     * @param tag 句柄类型。
     * @return 64bit 编码句柄。
     */
    uint64_t AllocHandle(SimCcuHandleTag tag);

    /**
     * @brief 分配 Event(CKE) 句柄（全局原子序列 + EVENT tag，兼作 NOTIFY 任务 notifyId 的编码原料）。
     */
    uint64_t AllocEventHandle();

    /**
     * @brief 登记通道派生变量来源（录制期 CcuVariableCreateByChannel 调用）。
     *
     * 记录 varHandle -> (channelHandle, varIndex)，供回放期读方识别交换变量并懒解析
     * （§3.4 跨 rank 值交换；只有被声明的槽参与交换）。
     */
    void NoteChannelVarSource(uint64_t varHandle, uint64_t channelHandle, uint32_t varIndex);

    /*==================== 注册事务 ====================*/

    /** @brief 开启 kernel 注册事务：登记 insHandle 为注册中，拒绝同一 insHandle 重入。 */
    HcclVmResult BeginRegister(uint64_t insHandle);
    /**
     * @brief 注册并就地录制 CCU kernel：注册期执行 kernelFunc 采集轨迹（此刻 kernelArg 存活），
     *        Launch 期仅回放。成功返回进程内唯一的内核句柄。
     */
    HcclVmResult RegisterKernel(
        uint64_t insHandle, uint32_t dieId, const char* funcName, const void* func, const void* kernelArg,
        uint32_t argNum, uint64_t& kernelHandle);
    /** @brief 结束 kernel 注册事务（关闭注册窗口）。 */
    HcclVmResult EndRegister(uint64_t insHandle);
    /** @brief 按内核句柄查注册表，命中则拷贝出条目。 */
    bool FindKernel(uint64_t kernelHandle, SimCcuKernelEntry& entry);

    /*==================== Launch 回放 ====================*/

    HcclVmResult ExecuteKernelOnLaunch(
        uint64_t threadHandle, uint64_t kernelHandle, const SimCcuKernelEntry& entry, const uint64_t* taskArgs,
        uint32_t argNum);

    /*==================== 录制：控制流宏标签栈 ====================*/

    /** @brief CCU_IF 进入时压入 if 标签栈（供 CCU_ELSE / 自动闭合配对）。 */
    void IfStackPush(const char* label);
    /** @brief 标记栈顶 CCU_IF 的 if 体已执行完（允许后续 CCU_ELSE 配对 / flush 闭合）。 */
    void IfStackMarkBodyDone();
    /** @brief CCU_ELSE 弹出栈顶 if 标签，返回池内保活指针（无匹配返回 nullptr）。 */
    const char* IfStackPopForElse();
    /** @brief 闭合栈顶所有已 bodyDone 的挂起 if（补录 IF_END）。 */
    void FlushPendingIfs();
    /** @brief CCU_DO 完成时压入 do-while 标签（供后续 CCU_WHILE 作为其结束条件）。 */
    void DoWhileStackPush(const char* label);
    /** @brief CCU_WHILE 弹出 do-while 标签（无挂起返回 nullptr）；返回池内保活指针。 */
    const char* DoWhileStackPopForWhile();

    /*==================== 录制：while/do-while 事务校验 ====================*/

    CcuResult WhileBeginCheck(const char* label);
    CcuResult WhileEndCheck(const char* label);
    CcuResult DoWhileBeginCheck(const char* label);
    CcuResult DoWhileEndCheck(const char* label);

    /*==================== 录制：Prim* 原语录入 ====================*/

    /* 变量/地址 */
    void PrimVarAssignImm(uint64_t varHandle, uint64_t immediate);
    void PrimVarAssignVar(uint64_t dstVar, uint64_t srcVar);
    void PrimVarOpVar(uint64_t resVar, uint64_t varA, uint64_t varB, SimCcuVarOp op);
    void PrimVarOpImm(uint64_t resVar, uint64_t varA, uint64_t immediate, SimCcuVarOp op);
    void PrimVarNot(uint64_t resVar, uint64_t varA);
    void PrimAddrAssignImm(uint64_t addrHandle, uint64_t immediate);
    void PrimAddrAssignAddr(uint64_t dstAddr, uint64_t srcAddr);
    void PrimAddrAssignVar(uint64_t addrHandle, uint64_t varHandle);
    void PrimAddrOpVar(uint64_t resAddr, uint64_t lhsAddr, uint64_t rhsVar, SimCcuVarOp op);
    void PrimAddrOpAddr(uint64_t resAddr, uint64_t addrA, uint64_t addrB, SimCcuVarOp op);
    void PrimAddrOpImm(uint64_t resAddr, uint64_t addrA, uint64_t immediate, SimCcuVarOp op);
    void PrimAddrAddAssignVar(uint64_t addrHandle, uint64_t varHandle);

    /* 加载/存储/组合 */
    void PrimLoadArg(uint64_t varHandle, uint32_t argId);
    void PrimLoadMem(uint64_t varHandle, uint64_t addr, uint32_t num, bool addrIsVar);
    void PrimStoreMem(uint64_t varOrAddrHandle, uint64_t varHandle, uint32_t num, bool addrIsVar);
    void PrimLocalAddrAlloc(uint64_t localAddrHandle, uint64_t addrHandle, uint64_t tokenHandle);
    void PrimRemoteAddrAlloc(uint64_t remoteAddrHandle, uint64_t addrHandle, uint64_t tokenHandle);

    /* 同步 */
    void PrimEventRecord(uint64_t eventHandle, uint16_t mask);
    void PrimEventWait(uint64_t eventHandle, uint16_t mask);
    void PrimNotifyRecord(uint64_t channel, uint32_t notifyIdx, uint16_t mask);
    void PrimNotifyWait(uint64_t channel, uint32_t notifyIdx, uint16_t mask);
    /** @brief 录制“写远端变量 + 通知”（WriteVariableWithNotify，PreSync 交换机制）。 */
    void PrimWriteVarNotify(
        uint64_t channel, uint64_t varHandle, uint32_t remoteVarIdx, uint32_t remoteNotifyIdx, uint16_t mask);
    void PrimLocalNotifyRecord(const char* notifyTag, uint16_t mask);
    void PrimLocalNotifyWait(const char* notifyTag, uint16_t mask);

    /* 本地数据 */
    void PrimLocalCopyMm(uint64_t dstLocal, uint64_t srcLocal, uint64_t lenVar, uint64_t event, uint16_t mask);
    void PrimLocalCopyMb(uint64_t dstBuf, uint64_t srcLocal, uint64_t lenVar, uint64_t event, uint16_t mask);
    void PrimLocalCopyBm(uint64_t dstLocal, uint64_t srcBuf, uint64_t lenVar, uint64_t event, uint16_t mask);
    /** @brief 录制本地内存归约（LocalReduceMemToMem）。 */
    void PrimLocalReduceMm(
        uint64_t dstLocal, uint64_t srcLocal, uint64_t lenVar, uint64_t event, uint16_t mask, uint8_t dataType,
        uint8_t reduceOp);
    /** @brief 录制多缓冲归约（buffers[1..] 归约到 buffers[0]，count >= 2）。 */
    void PrimLocalReduceBufs(
        const uint64_t* buffers, uint32_t count, uint64_t lenVar, uint64_t event, uint16_t mask, uint8_t dataType,
        uint8_t reduceOp);

    /* 远端数据 */
    void PrimRemoteReadMm(
        uint64_t channel, uint64_t localH, uint64_t remoteH, uint64_t lenVar, uint64_t event, uint16_t mask);
    /** @brief 录制远端读入本地缓冲（ReadMemoryToBuffer，远端 mem → 本地 MS）。 */
    void
    PrimRemoteReadMb(uint64_t channel, uint64_t bufH, uint64_t remoteH, uint64_t lenVar, uint64_t event, uint16_t mask);
    /** @brief 录制远端读并归约（ReadMemoryReduce，远端 mem → 本地 mem 归约）。 */
    void PrimRemoteReadMr(
        uint64_t channel, uint64_t localH, uint64_t remoteH, uint64_t lenVar, uint64_t event, uint16_t mask,
        uint8_t dataType, uint8_t reduceOp);
    /** @brief 录制本地写远端内存（WriteMemoryToMemory）。 */
    void PrimRemoteWriteMm(
        uint64_t channel, uint64_t remoteH, uint64_t localH, uint64_t lenVar, uint64_t event, uint16_t mask);
    /** @brief 录制本地缓冲写远端内存（WriteBufferToMemory，本地 MS → 远端 mem）。 */
    void PrimRemoteWriteBm(
        uint64_t channel, uint64_t remoteH, uint64_t bufH, uint64_t lenVar, uint64_t event, uint16_t mask);
    /** @brief 录制本地写远端并归约（WriteMemoryReduce）。 */
    void PrimRemoteWriteMr(
        uint64_t channel, uint64_t remoteH, uint64_t localH, uint64_t lenVar, uint64_t event, uint16_t mask,
        uint8_t dataType, uint8_t reduceOp);

    /* 控制流 */
    void PrimIfBegin(uint64_t lhsVar, uint64_t rhs, bool isVarCompare, CcuConditionType condType, const char* label);
    void PrimIfElse(const char* label);
    void PrimIfEnd(const char* label);
    void PrimWhileBegin(uint64_t lhsVar, uint64_t rhs, bool isVarCompare, CcuConditionType condType, const char* label);
    void PrimWhileEnd(const char* label);
    void PrimDoBegin(const char* label);
    void PrimDoEnd(uint64_t lhsVar, uint64_t rhs, bool isVarCompare, CcuConditionType condType, const char* label);

    /* Loop/LoopGroup */
    void PrimLoopBodyEnter(uint32_t loopId);
    void PrimLoopBodyExit(uint32_t loopId);
    /**
     * @brief 开启 LoopGroup 录制：登记组级并行参数（clone 数/步进/来源），并回填 groupIdx。
     */
    void PrimLoopGroupBegin(
        SimCcuLoopParamSrc src, uint64_t cfgCloneNum, uint64_t cfgCloneLoopOffset, uint64_t cfgAddrOffset,
        uint64_t cfgCcuBufferOffset, uint64_t cfgEventOffset, uint64_t cfgVarOffset, uint64_t parallelVar,
        uint64_t offsetVar, uint64_t varOffsetVar, uint32_t& groupIdx);
    /** @brief 向 LoopGroup 追加一个 loop 成员（迭代数/步进及参数来源）。 */
    void PrimLoopGroupAddLoop(
        uint32_t groupIdx, uint32_t loopIdx, SimCcuLoopParamSrc src, uint64_t cfgIterNum, uint64_t cfgAddrOffset,
        uint64_t paramVar, uint64_t iterNumVar, uint64_t addrOffsetVar, uint64_t ctxIdVar);
    void PrimLoopGroupEnd(uint32_t groupIdx);

} // namespace CcuSim
} // namespace HcclSim

#endif // CCU_KERNEL_SIM_H
