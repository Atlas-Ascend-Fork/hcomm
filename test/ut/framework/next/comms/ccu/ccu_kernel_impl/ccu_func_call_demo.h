/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_FUNC_CALL_DEMO_H
#define CCU_FUNC_CALL_DEMO_H

#include "ccu_primitives.hpp"
#include "ccu_types.h"

namespace ccu = ::AscendC::ccu;

ccu::Func CcuFuncCallBasicFunc([](ccu::Variable x) {
    ccu::Variable tmp{};
    tmp = x + x;
});

ccu::Func CcuFuncCallNestedInnerFunc([](ccu::Variable x) {
    ccu::Variable tmp{};
    tmp = x + x;
});

ccu::Func CcuFuncCallNestedOuterFunc([](ccu::Variable x) {
    (void)ccu::CallFunc<CcuFuncCallNestedInnerFunc>(x);
});

inline CcuResult CcuFuncCallBasicDemoKernel(CcuKernelArg arg)
{
    (void)arg;
    ccu::Variable x{};
    x = 1;
    return ccu::CallFunc<CcuFuncCallBasicFunc>(x);
}

inline CcuResult CcuFuncCallReuseDemoKernel(CcuKernelArg arg)
{
    (void)arg;
    ccu::Variable x{};
    x = 2;
    CCU_CHK_RET(ccu::CallFunc<CcuFuncCallBasicFunc>(x));
    CCU_CHK_RET(ccu::CallFunc<CcuFuncCallNestedInnerFunc>(x));
    return ccu::CallFunc<CcuFuncCallBasicFunc>(x);
}

inline CcuResult CcuFuncCallInLoopInvalidDemoKernel(CcuKernelArg arg)
{
    (void)arg;
    ccu::Variable x{};
    x = 3;

    CcuResult bodyRet = CcuResult::CCU_SUCCESS;
    ccu::Func body([&]() {
        bodyRet = ccu::CallFunc<CcuFuncCallBasicFunc>(x);
    });
    ccu::LoopConfig dummyCfg{};
    ccu::Loop loop(dummyCfg, body);
    return bodyRet;
}

inline CcuResult CcuFuncCallNestedInvalidDemoKernel(CcuKernelArg arg)
{
    (void)arg;
    ccu::Variable x{};
    x = 4;
    return ccu::CallFunc<CcuFuncCallNestedOuterFunc>(x);
}

ccu::Func CcuFuncCallMultiArgFunc([](ccu::Variable a, ccu::Variable b, ccu::Variable c) {
    ccu::Variable tmp{};
    tmp = a + b;
    tmp = tmp + c;
});

inline CcuResult CcuFuncCallMultiArgDemoKernel(CcuKernelArg arg)
{
    (void)arg;
    ccu::Variable x{};
    ccu::Variable y{};
    ccu::Variable z{};
    x = 1;
    y = 2;
    z = 3;
    return ccu::CallFunc<CcuFuncCallMultiArgFunc>(x, y, z);
}

constexpr const char* CCU_FUNC_BY_ADDR_NAME = "CcuFuncCallByAddrFunc";
constexpr const char* CCU_FUNC_NAMED_3ARG_NAME = "CcuFuncCallNamed3ArgFunc";

// 命名 func 的形参统一以 Array 传入，按下标顺序对应调用方 Array 的实参。
// 命名 func 声明即绑定当前 kernel，所以只能写在 kernel 函数体里；body 抽成自由函数供各 kernel 复用。
inline void CcuFuncByAddr2ArgBody(ccu::Array<ccu::Variable>& args)
{
    ccu::Variable tmp{};
    tmp = args[0] + args[1];
}

inline void CcuFuncByAddr3ArgBody(ccu::Array<ccu::Variable>& args)
{
    ccu::Variable tmp{};
    tmp = args[0] + args[1];
    tmp = tmp + args[2];
}

// 注册在构造时就完成，ccu::Func 对象随后析构不影响已录入 kernel 的 microcode，
// 所以可以把声明收进辅助函数里复用。失败时构造函数抛异常，由 kernel 注册路径转成注册失败。
inline void CcuDeclareByAddr2ArgFunc()
{
    ccu::Func byAddr(
        [](ccu::Array<ccu::Variable>& args) {
            CcuFuncByAddr2ArgBody(args);
        },
        2, CCU_FUNC_BY_ADDR_NAME);
}

inline void CcuDeclareByAddr3ArgFunc()
{
    ccu::Func byAddr(
        [](ccu::Array<ccu::Variable>& args) {
            CcuFuncByAddr3ArgBody(args);
        },
        3, CCU_FUNC_NAMED_3ARG_NAME);
}

inline CcuResult CcuFuncCallByAddrDemoKernel(CcuKernelArg arg)
{
    (void)arg;
    CcuDeclareByAddr2ArgFunc();

    ccu::Variable pc{};
    CCU_CHK_RET(ccu::LoadArg(pc, 0));

    ccu::Array<ccu::Variable> args(2);
    args[0] = 1;
    args[1] = 2;
    return ccu::CallFunc(pc, args);
}

constexpr const char* CCU_FUNC_NAMED_NTTP_NAME = "CcuFuncNamedNttpFunc";

// 命名 func 的形参绑定在跨 kernel 共用的传参寄存器上，只能按 pc 调用；误传给按符号调用的
// CallFunc<Obj> 应中断注册，并在日志里提示改用 ccu::CallFunc(pc, args)。
// 命名 func 构造需要活跃的 kernel 注册，不能放命名空间作用域，故用函数内 static 兼顾静态存储期
// 与 NTTP 取址（C++17 起允许无链接实体作引用 NTTP）。
inline CcuResult CcuFuncCallNamedByNttpInvalidDemoKernel(CcuKernelArg arg)
{
    (void)arg;
    static ccu::Func named(
        [](ccu::Array<ccu::Variable>& args) {
            CcuFuncByAddr2ArgBody(args);
        },
        2, CCU_FUNC_NAMED_NTTP_NAME);

    ccu::Variable x{};
    ccu::Variable y{};
    x = 1;
    y = 2;
    return ccu::CallFunc<named>(x, y);
}

constexpr const char* CCU_FUNC_OVER_LIMIT_NAME = "CcuFuncOverLimitFunc";
constexpr uint32_t CCU_FUNC_OVER_LIMIT_ARG_NUM = CCU_FUNC_XKERNEL_ARG_MAX + 1;
constexpr uint32_t CCU_FUNC_RETRY_ARG_NUM = 2;

// numIn 超上限的命名 func 应在构造处就判错（不应先分配形参、跑完 body 才失败）
inline CcuResult CcuFuncNamedOverLimitInvalidDemoKernel(CcuKernelArg arg)
{
    (void)arg;
    ccu::Func overLimit(
        [](ccu::Array<ccu::Variable>& args) {
            CcuFuncByAddr2ArgBody(args);
        },
        CCU_FUNC_OVER_LIMIT_ARG_NUM, CCU_FUNC_OVER_LIMIT_NAME);
    return CcuResult::CCU_SUCCESS;
}

// 裸 C 路径拿到的是返回码而非异常，可在同一 kernel 内改正参数个数后同名重试：
// 上一次超限失败必须把名字登记清掉，否则重试会被误判重复
inline CcuResult CcuFuncNamedRetryAfterOverLimitDemoKernel(CcuKernelArg arg)
{
    (void)arg;
    uint64_t overLimitHandle = 0;
    CCU_CHK_RET(::CcuFuncNamedBlockBegin(CCU_FUNC_OVER_LIMIT_NAME, &overLimitHandle));
    ccu::Array<ccu::Variable> tooMany(CCU_FUNC_OVER_LIMIT_ARG_NUM);
    for (uint32_t i = 0; i < CCU_FUNC_OVER_LIMIT_ARG_NUM; i++) {
        CCU_CHK_RET(::CcuFuncDefineInArg(overLimitHandle, tooMany[i].handle));
    }
    if (::CcuFuncBlockEnd(overLimitHandle) == CcuResult::CCU_SUCCESS) {
        return CcuResult::CCU_E_INTERNAL; // 超限本应失败
    }

    uint64_t retryHandle = 0;
    CCU_CHK_RET(::CcuFuncNamedBlockBegin(CCU_FUNC_OVER_LIMIT_NAME, &retryHandle));
    ccu::Array<ccu::Variable> formals(CCU_FUNC_RETRY_ARG_NUM);
    for (uint32_t i = 0; i < CCU_FUNC_RETRY_ARG_NUM; i++) {
        CCU_CHK_RET(::CcuFuncDefineInArg(retryHandle, formals[i].handle));
    }
    CcuFuncByAddr2ArgBody(formals);
    return ::CcuFuncBlockEnd(retryHandle);
}

// 函数体录制中途抛异常时 func 未收尾，异常须传出 kernel 函数，由注册入口判本次注册失败
inline CcuResult CcuFuncNamedBodyThrowInvalidDemoKernel(CcuKernelArg arg)
{
    (void)arg;
    ccu::Func throwing(
        [](ccu::Array<ccu::Variable>& args) {
            CcuFuncByAddr2ArgBody(args);
            CCU_THROW_IF_FAILED(CcuResult::CCU_E_PARA, "body failed on purpose");
        },
        2, CCU_FUNC_BY_ADDR_NAME);
    return CcuResult::CCU_SUCCESS;
}

// 同一 kernel 内重名的命名 func 应被拒绝（name 是 kernel 内的唯一身份）
inline CcuResult CcuFuncNamedDuplicateInvalidDemoKernel(CcuKernelArg arg)
{
    (void)arg;
    CcuDeclareByAddr2ArgFunc();
    CcuDeclareByAddr2ArgFunc(); // 第二次重名，应判错
    return CcuResult::CCU_SUCCESS;
}

// 同一个命名 func 被多次间接调用，且与未命名 func 的直接调用混用
inline CcuResult CcuFuncCallMixedDemoKernel(CcuKernelArg arg)
{
    (void)arg;
    CcuDeclareByAddr2ArgFunc();

    ccu::Variable pc{};
    CCU_CHK_RET(ccu::LoadArg(pc, 0));

    ccu::Variable x{};
    x = 1;
    ccu::Array<ccu::Variable> args(2);
    args[0] = 1;
    args[1] = 2;
    CCU_CHK_RET(ccu::CallFunc(pc, args));
    CCU_CHK_RET(ccu::CallFunc<CcuFuncCallBasicFunc>(x));
    return ccu::CallFunc(pc, args);
}

// 命名 func 的声明不影响之后匿名 func 的惰性合成与重复直调
inline CcuResult CcuFuncNamedThenAnonCallDemoKernel(CcuKernelArg arg)
{
    (void)arg;
    CcuDeclareByAddr2ArgFunc();

    ccu::Variable x{};
    x = 1;
    CCU_CHK_RET(ccu::CallFunc<CcuFuncCallBasicFunc>(x));
    return ccu::CallFunc<CcuFuncCallBasicFunc>(x);
}

// 同一 kernel 内不同入参个数的命名 func 共用同一批共享传参寄存器
inline CcuResult CcuFuncCallNamedMultiArityDemoKernel(CcuKernelArg arg)
{
    (void)arg;
    CcuDeclareByAddr2ArgFunc();
    CcuDeclareByAddr3ArgFunc();

    ccu::Variable pc0{};
    ccu::Variable pc1{};
    CCU_CHK_RET(ccu::LoadArg(pc0, 0));
    CCU_CHK_RET(ccu::LoadArg(pc1, 1));

    ccu::Array<ccu::Variable> args2(2);
    ccu::Array<ccu::Variable> args3(3);
    args2[0] = 1;
    args2[1] = 2;
    args3[0] = 1;
    args3[1] = 2;
    args3[2] = 3;
    CCU_CHK_RET(ccu::CallFunc(pc0, args2));
    CCU_CHK_RET(ccu::CallFunc(pc1, args3));
    return ccu::CallFunc(pc0, args2);
}

// 两个不同入参个数的命名 func、匿名 func 直调与按 pc 调用交错混用
inline CcuResult CcuFuncCallByAddrMixedArityDemoKernel(CcuKernelArg arg)
{
    (void)arg;
    CcuDeclareByAddr2ArgFunc();
    CcuDeclareByAddr3ArgFunc();

    ccu::Variable pc0{};
    ccu::Variable pc1{};
    CCU_CHK_RET(ccu::LoadArg(pc0, 0));
    CCU_CHK_RET(ccu::LoadArg(pc1, 1));

    ccu::Variable x{};
    x = 1;
    ccu::Array<ccu::Variable> args2(2);
    ccu::Array<ccu::Variable> args3(3);
    args2[0] = 1;
    args2[1] = 2;
    args3[0] = 1;
    args3[1] = 2;
    args3[2] = 3;
    CCU_CHK_RET(ccu::CallFunc(pc0, args2));
    CCU_CHK_RET(ccu::CallFunc<CcuFuncCallBasicFunc>(x));
    CCU_CHK_RET(ccu::CallFunc(pc1, args3));
    return ccu::CallFunc(pc0, args2);
}

ccu::Func CcuFuncCallByAddrNestedFunc([](ccu::Variable x) {
    ccu::Array<ccu::Variable> args(1);
    args[0] = x;
    (void)ccu::CallFunc(x, args);
});

inline CcuResult CcuFuncCallByAddrNestedInvalidDemoKernel(CcuKernelArg arg)
{
    (void)arg;
    ccu::Variable x{};
    x = 1;
    return ccu::CallFunc<CcuFuncCallByAddrNestedFunc>(x);
}

inline CcuResult CcuFuncCallByAddrInLoopInvalidDemoKernel(CcuKernelArg arg)
{
    (void)arg;
    ccu::Variable pc{};
    CCU_CHK_RET(ccu::LoadArg(pc, 0));
    ccu::Array<ccu::Variable> args(1);
    args[0] = 1;

    CcuResult bodyRet = CcuResult::CCU_SUCCESS;
    ccu::Func body([&]() {
        bodyRet = ccu::CallFunc(pc, args);
    });
    ccu::LoopConfig dummyCfg{};
    ccu::Loop loop(dummyCfg, body);
    return bodyRet;
}

ccu::Func CcuFuncCallWithLoopFunc([](ccu::Variable x) {
    ccu::Func body([&]() {
        ccu::Variable tmp{};
        tmp = x + x;
    });
    ccu::LoopConfig cfg = {.addrOffset = 0, .iterNum = 2};
    ccu::Loop loop(cfg, body);
    ccu::LoopGroupConfig grpCfg
        = {.cloneNum = 0, .cloneLoopOffset = 0, .addrOffset = 0, .ccuBufferOffset = 0, .eventOffset = 0};
    ccu::LoopGroup group(grpCfg, 1, {loop});
});

inline CcuResult CcuFuncCallWithLoopDemoKernel(CcuKernelArg arg)
{
    (void)arg;
    ccu::Variable x{};
    x = 1;
    return ccu::CallFunc<CcuFuncCallWithLoopFunc>(x);
}

#endif // CCU_FUNC_CALL_DEMO_H
