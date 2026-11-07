/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CPU_THREAD_TEST_MOCK_DEPS_H
#define CPU_THREAD_TEST_MOCK_DEPS_H

#include <cstdint>
#include <stdexcept>
#include "../../../hccl_api_base_test.h"
#include "cpu_thread.h"
#include "orion_adapter_rts.h"
#include "orion_adapter_hal.h"
#include "driver/ascend_hal.h"

namespace {
using namespace hccl;
// Hrt* 在 namespace Hccl 且带 extern "C"，逐名引入避免与 hccl 命名空间二义
using Hccl::HrtGetDevicePhyIdByUserDevId;
using Hccl::HrtHalGetDeviceInfo;
using Hccl::HrtHalHostRegister;
using Hccl::HrtHalHostUnregister;

// Init 失败矩阵的注入点。mockcpp「先注册先命中」，同一依赖一旦注册成功桩，
// 后面再注册失败桩不会生效，故失败须在注册时选定
enum class InitFailPoint {
    kNone = 0,
    kGetDeviceType,
    kLogicDevId,
    kGetCurrentContextNpu,
    kGetCurrentContextDpu,
    kBinaryLoad,
    kCreateStream,
    kGetFunction,
    kSetContextNpuAfterLaunch, // kernel 已下发成功后，切回 npu ctx 失败
    kDepThrows,                // 链上依赖抛异常（异常边界用例）
};

// aclrtGetCurrentContext 在 Init 内被调用两次（先 npu ctx、后 dpu ctx），签名相同，
// 失败注入须按调用次序区分，故用计数桩
inline uint32_t g_getCtxCall = 0;
inline uint32_t g_getCtxFailAt = 0;
inline aclError StubGetCurrentContext(aclrtContext* ctx)
{
    ++g_getCtxCall;
    if (ctx != nullptr) {
        *ctx = reinterpret_cast<aclrtContext>(0x1000);
    }
    return (g_getCtxFailAt == g_getCtxCall) ? ACL_ERROR_INVALID_PARAM : ACL_SUCCESS;
}

// DeInit 内 aclrtSetCurrentContext 同样是两次（切 dpu ctx 销毁流、切回 npu ctx），同款计数桩
inline uint32_t g_setCtxCall = 0;
inline uint32_t g_setCtxFailAt = 0;
inline aclError StubSetCurrentContext(aclrtContext ctx)
{
    (void)ctx;
    ++g_setCtxCall;
    return (g_setCtxFailAt == g_setCtxCall) ? ACL_ERROR_INVALID_PARAM : ACL_SUCCESS;
}

// UB 分支的 device 侧地址取自注册回填的 accessVA_，桩不回填会让 AllocDputagShmem 判空失败
inline HcclResult StubHostRegister(void* srcPtr, uint64_t size, uint32_t flag, int32_t deviceLogicId, void** dstPtr)
{
    (void)size;
    (void)flag;
    (void)deviceLogicId;
    if (dstPtr != nullptr) {
        *dstPtr = srcPtr;
    }
    return HCCL_SUCCESS;
}

// 失败出口回收已建流的观测点：用例按这个假句柄断言被回收的正是这条流
constexpr uint64_t FAKE_STREAM = 0x2000;
inline aclError StubCreateStreamWithFakeHandle(aclrtStream* stream, uint32_t priority, uint32_t flag)
{
    (void)priority;
    (void)flag;
    if (stream != nullptr) {
        *stream = reinterpret_cast<aclrtStream>(FAKE_STREAM);
    }
    return ACL_SUCCESS;
}

// Init / InitQueueMode 成功路径上除 hrtGetDeviceType / HrtSetXpuDevice / HrtResetXpuDevice 以外的全部依赖。
//  1. aclrtCreateStreamWithConfig 默认不注入 fake stream，让 rtStream_ 保持 nullptr；需要观测
//     「失败出口回收已建流」的用例传 fakeStream = true；
//  2. HrtHalGetDeviceInfo 返回 UB 连接类型，让 DpuShmemMgr::Alloc 走「真实 malloc + 4K 对齐」分支，
//     避免 PCIE 分支 HrtMalloc 桩返回假地址被 memset_s 写入导致段错误；
//  3. fail 指定让哪一个依赖失败，其余保持成功，用于逐步覆盖 Init 的失败矩阵。
inline void MockInitDeps(DevType devType, InitFailPoint fail = InitFailPoint::kNone, bool fakeStream = false)
{
    static char realPathBuf[1] = {0};
    MOCKER(realpath).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(static_cast<char*>(realPathBuf)));
    g_getCtxCall = 0;
    g_getCtxFailAt = (fail == InitFailPoint::kGetCurrentContextNpu) ?
                         1u :
                         ((fail == InitFailPoint::kGetCurrentContextDpu) ? 2u : 0u);
    MOCKER(hrtGetDeviceType)
        .stubs()
        .with(outBound(devType))
        .will(returnValue(fail == InitFailPoint::kGetDeviceType ? HCCL_E_RUNTIME : HCCL_SUCCESS));
    if (fail == InitFailPoint::kDepThrows) {
        // 异常边界用例：共享内存与 TaskService 都已就绪后抛异常，验证 Init 把异常转成错误码交给调用方回滚，
        // 而不是穿出 HcclTaskRegister 的 extern "C" 边界（穿出即 terminate）
        MOCKER(aclrtGetCurrentContext).stubs().will(throws(std::runtime_error("stub throws")));
    } else {
        MOCKER(aclrtGetCurrentContext).stubs().with(mockcpp::any()).will(invoke(StubGetCurrentContext));
    }
    // aclrtSetCurrentContext 在 Init 路径上首次被调用即 LaunchKernel 里「切回 npu ctx」，故用计数桩按次注入；
    // failAt == 0 时恒成功，与固定成功桩等价
    g_setCtxCall = 0;
    g_setCtxFailAt = (fail == InitFailPoint::kSetContextNpuAfterLaunch) ? 1u : 0u;
    MOCKER(aclrtSetCurrentContext).stubs().with(mockcpp::any()).will(invoke(StubSetCurrentContext));
    MOCKER(aclrtBinaryLoadFromFile)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(fail == InitFailPoint::kBinaryLoad ? ACL_ERROR_INVALID_PARAM : ACL_SUCCESS));
    if (fakeStream) {
        MOCKER(aclrtCreateStreamWithConfig)
            .stubs()
            .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
            .will(invoke(StubCreateStreamWithFakeHandle));
    } else {
        MOCKER(aclrtCreateStreamWithConfig)
            .stubs()
            .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
            .will(returnValue(fail == InitFailPoint::kCreateStream ? ACL_ERROR_INVALID_PARAM : ACL_SUCCESS));
    }
    MOCKER(aclrtBinaryGetFunction)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(fail == InitFailPoint::kGetFunction ? ACL_ERROR_INVALID_PARAM : ACL_SUCCESS));

    // InitShmem 的 DPU 共享内存申请链（DPUTAG + DPUTASKEXCEPTION）
    int32_t logicDevId = 0;
    MOCKER(aclrtGetLogicDevIdByUserDevId)
        .stubs()
        .with(mockcpp::any(), outBoundP(&logicDevId))
        .will(returnValue(fail == InitFailPoint::kLogicDevId ? ACL_ERROR_INVALID_PARAM : ACL_SUCCESS));
    int64_t connectType = HOST_DEVICE_CONNECT_TYPE_UB;
    MOCKER(HrtHalGetDeviceInfo)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBound(connectType))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER(HrtHalHostRegister).stubs().will(invoke(StubHostRegister));
    MOCKER(HrtHalHostUnregister).stubs().will(returnValue(HCCL_SUCCESS));

    // Stream(nullptr) 构造里 InitDevPhyId 的依赖
    MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().will(returnValue(static_cast<Hccl::DevId>(0)));
}
} // namespace

#endif // CPU_THREAD_TEST_MOCK_DEPS_H
