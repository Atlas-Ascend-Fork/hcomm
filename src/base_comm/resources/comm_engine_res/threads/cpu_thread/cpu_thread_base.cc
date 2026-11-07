/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cpu_thread_base.h"
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <limits>
#include <thread>
#include "legacy_dpu_thread/dpu_kernel_entrance.h" // SetDpuExecTimeout
#include "hccl_common.h"
#include "securec.h"
#include "adapter_rts.h"
#include "orion_adapter_rts.h"
#include "orion_adapter_hal.h"
#include "driver/ascend_hal.h"
#include "log.h"
#include "env_config/env_config_v2.h"

namespace hccl {

namespace {
    // 由 legacy `communicator_impl.cc` 迁入的常量，两条通道共用
    constexpr uint8_t DEVICE_SIGNAL_SECOND = 2;
    constexpr uint8_t DEVICE_SIGNAL_THIRD = 3;
    constexpr uint32_t TEMP_DEV_TYPE_DPU = 0; // 临时适配，后续 rts 接口上库之后使用 rts 的定义
    constexpr u32 DPU_KERNEL_NUM_BLOCKS = 1;
} // namespace

std::atomic<uint32_t> CpuThreadBase::commNum_{0};

CpuThreadBase::CpuCtxRollback::~CpuCtxRollback() { (void)aclrtSetCurrentContext(npuCtx_); }

CpuThreadBase::CpuStreamRollback::CpuStreamRollback(
    rtStream_t& rtStream, bool& isKernelLaunched, aclrtContext dpuCtx, aclrtContext npuCtx,
    std::unique_ptr<Stream>& stream)
    : rtStream_(rtStream),
      isKernelLaunched_(isKernelLaunched),
      dpuContext_(dpuCtx),
      npuContext_(npuCtx),
      stream_(stream)
{}

CpuThreadBase::CpuStreamRollback::~CpuStreamRollback()
{
    if (isKernelLaunched_ || rtStream_ == nullptr) {
        return;
    }
    // 流在 dpu 域下创建，销毁须在其 ctx 内（与 DeInit 同口径）：launch 的失败分支已切回
    // npu ctx，故先切 dpu ctx，回收后再切回 npu ctx，避免调用线程停在 dpu 域
    if (dpuContext_ != nullptr) {
        (void)aclrtSetCurrentContext(dpuContext_);
    }
    if (aclrtDestroyStreamForce(rtStream_) != ACL_SUCCESS) {
        HCCL_ERROR(
            "[CpuThreadBase][%s] destroy unlaunched dpu stream failed, rtStream[%p]", __func__,
            static_cast<void*>(rtStream_));
    }
    stream_.reset();
    rtStream_ = nullptr;
    (void)aclrtSetCurrentContext(npuContext_);
}

CpuThreadBase::~CpuThreadBase()
{
    // 兜底：正常路径上派生类析构已调过 DeInit（isKernelLaunched_ 已清 ⇒ 此处空转）。
    // 之后才释放共享内存 —— 顺序不可颠倒，kernel 退出前仍在读这块内存
    (void)DeInit();

    if (binHandle_ != nullptr) {
        if (dpuContext_ != nullptr) {
            (void)aclrtSetCurrentContext(dpuContext_);
        }
        (void)aclrtBinaryUnLoad(binHandle_);
        binHandle_ = nullptr;
        (void)aclrtSetCurrentContext(npuContext_);
    }

    if (shmemMgr_.AnyAllocated()) {
        if (isDeviceLogicIdResolved_) {
            shmemMgr_.FreeAll(deviceLogicId_);
        } else {
            HCCL_ERROR(
                "[CpuThreadBase][%s] deviceLogicId never resolved, skip FreeAll, devId: %u", __func__, args_.devId);
        }
    }
    dputagAddr_ = nullptr;
    dputagDevAddr_ = nullptr;
    dputagHostPtr_ = nullptr;
}

HcclResult CpuThreadBase::ResolveDeviceLogicId(int32_t& deviceLogicId)
{
    aclError aclRet = aclrtGetLogicDevIdByUserDevId(args_.devId, &deviceLogicId);
    if (aclRet != ACL_SUCCESS) {
        HCCL_ERROR(
            "[CpuThreadBase][%s] aclrtGetLogicDevIdByUserDevId failed, devId: %u, ret: %d", __func__, args_.devId,
            aclRet);
        return HCCL_E_RUNTIME;
    }
    deviceLogicId_ = deviceLogicId;
    isDeviceLogicIdResolved_ = true;
    return HCCL_SUCCESS;
}

HcclResult CpuThreadBase::AllocDputagShmem(int32_t deviceLogicId, uint64_t size)
{
    // DPUTAG 的申请与注册全部由独立管理器 DpuShmemMgr 托管：
    // 910/950 都不再依赖 legacy 侧单独的全局表，两个实体各持一个管理器实例、各自申请各自所需（100MB / 256KB）
    CHK_RET(shmemMgr_.Alloc(Hccl::DPU_TAG, size, deviceLogicId));
    // host 侧地址供 TaskService 与终止握手写 flag；device 侧地址供 HcclDevMemAcquire（device scatter
    // kernel 解引用），UB 下两者不同、PCIE/SVM 下同址
    dputagAddr_ = shmemMgr_.HostPtr(Hccl::DPU_TAG);
    dputagDevAddr_ = shmemMgr_.DevPtr(Hccl::DPU_TAG);
    dputagHostPtr_ = dputagAddr_; // 握手字节在 DPUTAG 偏移 0（原 SetDputagHostPtr 注入口径收编为自查）
    CHK_PTR_NULL(dputagAddr_);
    CHK_PTR_NULL(dputagDevAddr_);

    const Hccl::DpuShmem* dputag = shmemMgr_.Get(Hccl::DPU_TAG);
    HCCL_INFO(
        "[CpuThreadBase][%s] dputag va_[%p] accessVA_[%p]", __func__, dputag != nullptr ? dputag->va_ : nullptr,
        dputag != nullptr ? dputag->accessVA_ : nullptr);
    return HCCL_SUCCESS;
}

HcclResult CpuThreadBase::EnterDpuContext()
{
    HCCL_INFO("[CpuThreadBase][%s] Switch to Dpu Ctx", __func__);
    // 整段 XPU 切换序列不可拆：先存 npu ctx，再切 dpu 域，最后取 dpu ctx
    if (aclrtGetCurrentContext(&npuContext_) != ACL_SUCCESS) {
        HCCL_ERROR("[CpuThreadBase][%s] Get Npu Ctx Failed", __func__);
        return HCCL_E_INTERNAL;
    }
    if (Hccl::HrtSetXpuDevice(TEMP_DEV_TYPE_DPU, 0) != HCCL_SUCCESS) {
        HCCL_ERROR("[CpuThreadBase][%s] Switch to Dpu Ctx Failed", __func__);
        return HCCL_E_INTERNAL;
    }
    if (aclrtGetCurrentContext(&dpuContext_) != ACL_SUCCESS) {
        // 已切入 dpu 域：本函数自行还原，调用方此后才用 CpuCtxRollback 接管
        HCCL_ERROR("[CpuThreadBase][%s] Get Dpu Ctx Failed", __func__);
        (void)aclrtSetCurrentContext(npuContext_);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

namespace {
    struct LoadCleanupGuard {
        explicit LoadCleanupGuard(aclrtBinHandle& handle) : handle_(handle), active_(true) {}

        ~LoadCleanupGuard()
        {
            if (active_ && handle_ != nullptr) {
                (void)aclrtBinaryUnLoad(handle_);
                handle_ = nullptr;
            }
        }

        void Dismiss() { active_ = false; }

        aclrtBinHandle& handle_;
        bool active_;
    };
} // namespace

HcclResult CpuThreadBase::PrepareDpuKernelResource(const char* entryName)
{
    // 获取二进制文件路径
    std::string jsonPath;
    const char* ascendHomePath = getenv("ASCEND_HOME_PATH"); // getenv 未命中返回 nullptr，不可直接构造 std::string
    std::string getPath = ascendHomePath != nullptr ? ascendHomePath : "";
    if (!getPath.empty()) {
        jsonPath = getPath;
    } else {
        jsonPath = "/usr/local/Ascend/cann/";
        HCCL_WARNING("[CpuThreadBase][%s] ENV:ASCEND_HOME_PATH is not set", __func__);
    }

    jsonPath += "/opp/built-in/op_impl/dpu/";
    HCCL_DEBUG("[CpuThreadBase][%s] kernel folder path[%s]", __func__, jsonPath.c_str());

    // cpuKernelMode为1时，json命名需与so命名保持一致， 即libccl_dpu.json与libccl_dpu.so
    jsonPath += "libccl_dpu.json";
    char realPath[PATH_MAX] = {0};
    CHK_PRT_RET(
        realpath(jsonPath.c_str(), realPath) == nullptr,
        HCCL_ERROR("[CpuThreadBase][%s]: %s is not a valid real path, err[%d]", __func__, jsonPath.c_str(), errno),
        HCCL_E_INTERNAL);
    HCCL_INFO("[CpuThreadBase][%s] realPath: %s", __func__, realPath);

    aclrtBinHandle binHandle;
    aclrtBinaryLoadOptions options;
    aclrtBinaryLoadOption option;
    option.type = ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE; // AI CPU算子注册模式
    option.value.cpuKernelMode = 1; // 0 ：仅需要加载json，1 ：加载cpu so & json，2: LoadFromData
    options.numOpt = 1;
    options.options = &option;
    if (aclrtBinaryLoadFromFile(realPath, &options, &binHandle) != ACL_SUCCESS) {
        HCCL_ERROR("[CpuThreadBase][%s] load binary from file error.", __func__);
        return HCCL_E_OPEN_FILE_FAILURE;
    }
    binHandle_ = binHandle;
    LoadCleanupGuard binGuard(binHandle_);

    // 创建 dpustream
    aclError ret = aclrtCreateStreamWithConfig(&rtStream_, 0, ACL_STREAM_FAST_LAUNCH);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR("[CpuThreadBase][%s] Create Local Stream Failed", __func__);
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO(
        "[CpuThreadBase][%s] Create Stream para: deviceId[%d] streamId[%d]", __func__, Hccl::HrtGetDevice(),
        Hccl::HrtGetStreamId(rtStream_));

    // 查找核函数：入口名由调用方给出（RunDpuRpcSrvLaunch / RunDpuRpcSrvLaunchNew），
    // 必须按传入名取 —— 无条件取 RunDpuRpcSrvLaunchNew 会在 json 尚未生效的环境里把 TaskService 通道一起弄挂
    if (aclrtBinaryGetFunction(binHandle, entryName, &funcHandle_) != ACL_SUCCESS) {
        HCCL_ERROR("[CpuThreadBase][%s] Get Function Failed, entryName[%s]", __func__, entryName);
        return HCCL_E_INTERNAL;
    }

    binGuard.Dismiss();
    return HCCL_SUCCESS;
}

HcclResult CpuThreadBase::LaunchDpuKernelWithArgs(const void* args, size_t argsSize)
{
    // 下发前同步执行超时配置（HCCL_EXEC_TIMEOUT，秒）：内核侧 GetDpuExecTimeout 未下发时取默认值，host侧 launch 前覆盖
    SetDpuExecTimeout(Hccl::EnvConfig::GetInstance().GetRtsConfig().GetExecTimeOut());

    HCCL_INFO("[CpuThreadBase][%s] Launch Dpu Kernel", __func__);
    aclrtLaunchKernelCfg cfg;
    aclrtLaunchKernelAttr kernelAttr;
    kernelAttr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
    kernelAttr.value.timeout = NOTIFY_DEFAULT_WAIT_TIME > std::numeric_limits<uint16_t>::max() ?
                                   std::numeric_limits<uint16_t>::max() :
                                   NOTIFY_DEFAULT_WAIT_TIME;
    cfg.numAttrs = 1;
    cfg.attrs = &kernelAttr;
    CHK_RET(RefreshStreamId());
    aclrtPlaceHolderInfo placeHolderArrays;
    size_t placeHolderNum = 0;
    if (aclrtLaunchKernelWithHostArgs(
            funcHandle_, DPU_KERNEL_NUM_BLOCKS, rtStream_, &cfg, const_cast<void*>(args), argsSize, &placeHolderArrays,
            placeHolderNum)
        != ACL_SUCCESS) {
        HCCL_ERROR("[CpuThreadBase][%s] Launch Dpu Kernel Failed", __func__);
        // launch 失败仍处于 dpu 域，切回 npu ctx 再返回，避免调用线程滞留 dpu 域
        (void)aclrtSetCurrentContext(npuContext_);
        return HCCL_E_INTERNAL;
    }

    // 下发成功即代表 kernel 已在跑：先置位再做后续动作。否则切回 npu ctx 失败时析构链会因
    // !isKernelLaunched_ 短路，终止握手/流销毁/XPU reset 全被跳过，kernel 线程永不退出
    isKernelLaunched_ = true;
    commNum_++;

    // 切换回当前Ctx
    HCCL_INFO("[CpuThreadBase][%s] Switch to Npu Ctx", __func__);
    if (ACL_SUCCESS != aclrtSetCurrentContext(npuContext_)) {
        HCCL_ERROR("[CpuThreadBase][%s] Reset Current Ctx Failed", __func__);
        return HCCL_E_INTERNAL;
    }

    HCCL_INFO("[CpuThreadBase][%s] Launch Dpu Kernel End", __func__);
    return HCCL_SUCCESS;
}

HcclResult CpuThreadBase::WaitKernelTerminate()
{
    if (dputagHostPtr_ == nullptr) {
        HCCL_ERROR("[CpuThreadBase][%s] dputag host ptr is nullptr, shmem not allocated", __func__);
        return HCCL_E_PTR;
    }

    // 先读取是否有dpu线程回写的退出信号
    uint8_t flag = 0;
    errno_t ret = memcpy_s(&flag, sizeof(flag), dputagHostPtr_, sizeof(flag));
    if (ret != EOK) {
        HCCL_ERROR("[CpuThreadBase][%s] Read Terminate TaskRun Signal Fail, return[%d]", __func__, ret);
        return HCCL_E_INTERNAL;
    }
    if (flag == DEVICE_SIGNAL_THIRD) {
        return HCCL_SUCCESS; // dpu线程已退出
    }
    flag = DEVICE_SIGNAL_SECOND;
    ret = memcpy_s(dputagHostPtr_, sizeof(flag), &flag, sizeof(flag));
    if (ret != EOK) {
        HCCL_ERROR("[CpuThreadBase][%s] Terminate TaskRun Fail, return[%d]", __func__, ret);
        return HCCL_E_INTERNAL;
    }
    // 无超时上限：kernel 未退出就释放共享内存会与其读写冲突，必须等到退出信号
    do {
        ret = memcpy_s(&flag, sizeof(flag), dputagHostPtr_, sizeof(flag));
        if (ret != EOK) {
            HCCL_ERROR("[CpuThreadBase][%s] Read Terminate TaskRun Signal Fail, return[%d]", __func__, ret);
            return HCCL_E_INTERNAL;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while (flag != DEVICE_SIGNAL_THIRD);
    return HCCL_SUCCESS;
}

HcclResult CpuThreadBase::RefreshStreamId()
{
    streamId_ = Hccl::HrtGetStreamId(rtStream_);
    HCCL_INFO("[CpuThreadBase][%s] dpuStreamId[%u]", __func__, streamId_);
    return HCCL_SUCCESS;
}

HcclResult CpuThreadBase::DeInit()
{
    // 终止Dpu Kernel的TaskRun
    if (!isKernelLaunched_) {
        return HCCL_SUCCESS;
    }
    // 先清标志再动作：本函数要求幂等，且 DeInit 后 rtStream_ 即失效；标志已清也意味着不会再有第二次进入，
    // 故下面每步都必须走到：失败只记首个错误码、不提前返回，否则流销毁/XPU reset/ctx 还原会被永久跳过
    isKernelLaunched_ = false;

    HcclResult firstErr = HCCL_SUCCESS;
    auto recordErr = [&firstErr](HcclResult err) {
        if (firstErr == HCCL_SUCCESS) {
            firstErr = err;
        }
    };

    HcclResult waitRet = WaitKernelTerminate();
    if (waitRet != HCCL_SUCCESS) {
        HCCL_ERROR("[CpuThreadBase][%s] WaitKernelTerminate failed, return[%d]", __func__, waitRet);
        recordErr(waitRet);
    }

    // 切换回 dpu ctx：流在 dpu 域下创建，销毁须在其 ctx 内进行
    aclError aclRet = aclrtSetCurrentContext(dpuContext_);
    if (ACL_SUCCESS != aclRet) {
        HCCL_ERROR("[CpuThreadBase][%s] set dpu Ctx Failed, aclReturn[%d]", __func__, aclRet);
        recordErr(HCCL_E_RUNTIME);
    } else {
        // 销毁局部流
        aclRet = aclrtDestroyStreamForce(rtStream_);
        if (ACL_SUCCESS != aclRet) {
            HCCL_ERROR("[CpuThreadBase][%s] Destroy Stream Failed, aclReturn[%d]", __func__, aclRet);
            recordErr(HCCL_E_RUNTIME);
        }
    }
    // 句柄已不可再用：同时失效包装它的 Stream，避免后续拿到悬垂指针
    stream_.reset();
    rtStream_ = nullptr;

    // 二进制在 dpu 域下加载，卸载须在 ResetXpuDevice 之前完成（之后 dpu ctx 已失效）
    if (binHandle_ != nullptr) {
        (void)aclrtBinaryUnLoad(binHandle_);
        binHandle_ = nullptr;
    }

    if (--commNum_ == 0) {
        HcclResult ret = Hccl::HrtResetXpuDevice(TEMP_DEV_TYPE_DPU, 0);
        if (HCCL_SUCCESS != ret) {
            HCCL_ERROR("[CpuThreadBase][%s] ResetXpuDevice Failed, return[%d]", __func__, ret);
            recordErr(ret);
        }
    }
    // 切回 npu ctx：即使前面有失败也必须走完，否则本线程当前 context 会停在 dpu 域
    aclRet = aclrtSetCurrentContext(npuContext_);
    if (ACL_SUCCESS != aclRet) {
        HCCL_ERROR("[CpuThreadBase][%s] set npu Ctx failed, aclReturn[%d]", __func__, aclRet);
        recordErr(HCCL_E_RUNTIME);
    }

    return firstErr;
}

} // namespace hccl
