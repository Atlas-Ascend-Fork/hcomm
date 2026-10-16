/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "legacy_dpu_thread.h"
#include "dpu_kernel_entrance.h"
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <exception>
#include <limits>
#include <thread>
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
    // 由 legacy `communicator_impl.cc` 迁入的常量。暂留本文件，待 D2′ 结构体合并（S3a）时统一收敛到 dpu_types.h
    constexpr uint8_t DEVICE_SIGNAL_SECOND = 2;
    constexpr uint8_t DEVICE_SIGNAL_THIRD = 3;
    constexpr uint32_t TEMP_DEV_TYPE_DPU = 0; // 临时适配，后续 rts 接口上库之后使用 rts 的定义
    constexpr u32 DPU_KERNEL_NUM_BLOCKS = 1;
    constexpr u32 WAIT_CMD_TIMEOUT = 10 * 1000; // 最大等待10秒（收编自 communicator_impl.cc）

    // Init 切入 dpu 域后，离开作用域前必须切回 npu ctx：后续任一步失败都靠本守卫在返回前兜底还原，
    // 避免调用线程 current context 滞留 dpu 域（成功路径 LaunchKernel 已切回，重复切回无害）
    struct DpuCtxRollback {
        explicit DpuCtxRollback(aclrtContext npuCtx) : npuCtx_(npuCtx) {}
        ~DpuCtxRollback() { (void)aclrtSetCurrentContext(npuCtx_); }
        aclrtContext npuCtx_;
    };

    // 「流已建、kernel 未下发」时 Init 失败的回收守卫。必须显式回收：DeInit 以 isKernelLaunched_ 为
    // 短路判据，失败出口上该标志恒为 false，aclrtDestroyStreamForce 永不执行；而 Stream 以 rtStream
    // 构造、不是 stream owner，流不会随包装析构释放。半拉起线程由调用方回滚销毁，故每次失败重试都漏一条
    // dpu 流。成功下发（isKernelLaunched_ 为 true）时本守卫不介入，交由 DeInit 走完整终止流程
    struct DpuStreamRollback {
        DpuStreamRollback(
            rtStream_t& rtStream, bool& isKernelLaunched, aclrtContext dpuCtx, aclrtContext npuCtx,
            std::unique_ptr<Stream>& stream)
            : rtStream_(rtStream),
              isKernelLaunched_(isKernelLaunched),
              dpuContext_(dpuCtx),
              npuContext_(npuCtx),
              stream_(stream)
        {}
        ~DpuStreamRollback()
        {
            if (isKernelLaunched_ || rtStream_ == nullptr) {
                return;
            }
            // 流在 dpu 域下创建，销毁须在其 ctx 内（与 DeInit 同口径）：LaunchKernel 的失败分支已切回
            // npu ctx，故先切 dpu ctx，回收后再切回 npu ctx，避免调用线程停在 dpu 域
            if (dpuContext_ != nullptr) {
                (void)aclrtSetCurrentContext(dpuContext_);
            }
            if (aclrtDestroyStreamForce(rtStream_) != ACL_SUCCESS) {
                HCCL_ERROR(
                    "[LegacyDpuThread][%s] destroy unlaunched dpu stream failed, rtStream[%p]", __func__,
                    static_cast<void*>(rtStream_));
            }
            stream_.reset();
            rtStream_ = nullptr;
            (void)aclrtSetCurrentContext(npuContext_);
        }
        rtStream_t& rtStream_;
        bool& isKernelLaunched_;
        aclrtContext dpuContext_;
        aclrtContext npuContext_;
        std::unique_ptr<Stream>& stream_;
    };
} // namespace

std::atomic<uint32_t> LegacyDpuThread::commNum_{0};

LegacyDpuThread::~LegacyDpuThread()
{
    // 收编后 teardown 全部收敛进析构：由 ThreadMgr 析构 → HcommThreadFree → 本析构触发，
    // 每一步均幂等（isKernelLaunched_/taskExpMem_/hostShareBuf_/shmemMgr_ 判空/判非空），重复调用安全。
    // 顺序约束：先停 AICPU taskexception 监控（taskExpMem_ 仍有效），再终止 kernel，再析构实例并释放内存。
    (void)StopTaskexpService();
    (void)DeInit();

    // TaskService 实例析构须先于 hostShareBuf_ 释放（hostMem_ 非 owning 引用，否则 UAF）
    taskService_.reset();
    if (hostShareBuf_ != nullptr) {
        free(hostShareBuf_);
        hostShareBuf_ = nullptr;
    }

    if (shmemMgr_.AnyAllocated()) {
        int32_t deviceLogicId = 0;
        if (aclrtGetLogicDevIdByUserDevId(args_.devId, &deviceLogicId) == ACL_SUCCESS) {
            shmemMgr_.FreeAll(deviceLogicId);
        } else {
            HCCL_ERROR(
                "[LegacyDpuThread][%s] aclrtGetLogicDevIdByUserDevId failed in destructor, devId: %u", __func__,
                args_.devId);
        }
    }
    dputagAddr_ = nullptr;
    dputagHostPtr_ = nullptr;
    taskExpMem_ = nullptr;
}

HcclResult LegacyDpuThread::Init()
{
    // 本函数是异常边界：调用者 HcclTaskRegister 是 extern "C" 边界，异常穿出会终止进程。
    // 链路上多处可抛（TaskService 构造、jsonPath 等 STL 分配、EnvConfig 解析、Stream 构造），
    // 统一在此兜底；转成错误码后由调用方走既有回滚（RollbackAcquiredThread 释放线程与内存）
    try {
        return InitImpl();
    } catch (const std::exception& e) {
        HCCL_ERROR("[LegacyDpuThread][%s] exception caught: %s", __func__, e.what());
        return HCCL_E_INTERNAL;
    } catch (...) {
        HCCL_ERROR("[LegacyDpuThread][%s] unknown exception caught", __func__);
        return HCCL_E_INTERNAL;
    }
}

HcclResult LegacyDpuThread::InitImpl()
{
    CHK_RET(hrtGetDeviceType(devType_));

    // 三步收编 ①：初始化共享内存（DPUTAG + DPUTASKEXCEPTION，需在 npu ctx 下、切 dpu 域之前完成）
    CHK_RET(InitShmem());

    // 三步收编 ②a：宿主侧同步构造 TaskService（纯 host 侧构造，无 device 调用，先于 kernel 就绪）
    CHK_RET(CreateTaskService());

    HCCL_INFO("[LegacyDpuThread][%s] Switch to Dpu Ctx", __func__);
    // 整段 XPU 切换序列不可拆：先存 npu ctx，再切 dpu 域，最后取 dpu ctx
    if (aclrtGetCurrentContext(&npuContext_) != ACL_SUCCESS) {
        HCCL_ERROR("[LegacyDpuThread][%s] Get Npu Ctx Failed", __func__);
        return HCCL_E_INTERNAL;
    }
    if (Hccl::HrtSetXpuDevice(TEMP_DEV_TYPE_DPU, 0) != HCCL_SUCCESS) {
        HCCL_ERROR("[LegacyDpuThread][%s] Switch to Dpu Ctx Failed", __func__);
        return HCCL_E_INTERNAL;
    }
    // 从切入 dpu 域起，后续任一步失败都由守卫在返回前切回 npu ctx，避免调用线程滞留 dpu 域
    DpuCtxRollback rollback(npuContext_);

    if (aclrtGetCurrentContext(&dpuContext_) != ACL_SUCCESS) {
        HCCL_ERROR("[LegacyDpuThread][%s] Get Dpu Ctx Failed", __func__);
        return HCCL_E_INTERNAL;
    }

    // 从此处起 rtStream_ 有可能被建出，失败出口须回收（见 DpuStreamRollback）
    DpuStreamRollback streamRollback(rtStream_, isKernelLaunched_, dpuContext_, npuContext_, stream_);

    // 准备资源：加载 libccl_dpu 并建 dpu 流
    CHK_RET(PrepareDpuKernelResource());

    // Stream 构造经 InitStream 间接调 hrt 系列可能抛异常：catch 拦截避免异常穿透 extern C 边界
    // （HcclTaskRegister），切回 npu ctx 由上方 DpuCtxRollback 守卫兜底
    try {
        stream_.reset(new (std::nothrow) Stream(rtStream_));
    } catch (...) {
        HCCL_ERROR("[LegacyDpuThread][%s] create Stream failed, exception caught", __func__);
        return HCCL_E_PTR;
    }
    CHK_SMART_PTR_NULL(stream_);

    // 三步收编 ②b：下发 kernel
    CHK_RET(LaunchKernel());

    HCCL_INFO("[LegacyDpuThread][%s] end, rtStream[%p]", __func__, rtStream_);
    return HCCL_SUCCESS;
}

HcclResult LegacyDpuThread::InitShmem()
{
    // DPUTAG（100MB）+ DPUTASKEXCEPTION（10B）两块共享内存的申请与注册，
    // 全部收编到 base_comm 由独立管理器 DpuShmemMgr 托管：910/950 都不再依赖 legacy 侧单独的全局表，
    // 其他使用方可按 tag 各自持有一份管理器复用。
    int32_t deviceLogicId = 0;
    aclError aclRet = aclrtGetLogicDevIdByUserDevId(args_.devId, &deviceLogicId);
    if (aclRet != ACL_SUCCESS) {
        HCCL_ERROR(
            "[LegacyDpuThread][%s] aclrtGetLogicDevIdByUserDevId failed, devId: %u, ret: %d", __func__, args_.devId,
            aclRet);
        return HCCL_E_RUNTIME;
    }

    // DPUTAG：TaskService::deviceMem 与终止握手信号区共用同一 host 侧地址（原 LaunchDpuKernel 口径）
    CHK_RET(shmemMgr_.Alloc(Hccl::DPU_TAG, Hccl::SHARE_HBM_MEMORY_SIZE, deviceLogicId));
    dputagAddr_ = shmemMgr_.HostPtr(Hccl::DPU_TAG);
    dputagHostPtr_ = dputagAddr_; // 由本类内部置位（原 SetDputagHostPtr 注入口径收编为自查）

    // DPUTASKEXCEPTION：异常区 host 侧地址
    CHK_RET(shmemMgr_.Alloc(Hccl::DPU_TASK_EXCEPTION_TAG, Hccl::DPU_TASKEXCEPTION_MEMORY_SIZE, deviceLogicId));
    taskExpMem_ = shmemMgr_.HostPtr(Hccl::DPU_TASK_EXCEPTION_TAG);

    const Hccl::DpuShmem* dputag = shmemMgr_.Get(Hccl::DPU_TAG);
    const Hccl::DpuShmem* taskExp = shmemMgr_.Get(Hccl::DPU_TASK_EXCEPTION_TAG);
    HCCL_INFO(
        "[LegacyDpuThread][%s] dputag va_[%p] accessVA_[%p], taskExp va_[%p] accessVA_[%p]", __func__,
        dputag != nullptr ? dputag->va_ : nullptr, dputag != nullptr ? dputag->accessVA_ : nullptr,
        taskExp != nullptr ? taskExp->va_ : nullptr, taskExp != nullptr ? taskExp->accessVA_ : nullptr);
    return HCCL_SUCCESS;
}

HcclResult LegacyDpuThread::CreateTaskService()
{
    // host 侧数据缓冲（100MB）：TaskService::hostMem_ 是非 owning 引用，析构须先析构实例再 free（UAF 放大器）
    hostShareBuf_ = malloc(Hccl::SHARE_HBM_MEMORY_SIZE);
    CHK_PTR_NULL(hostShareBuf_);

    // 收编后实例归属宿主侧：hcomm.so 构造 TaskService，kernel 侧 RunDpuRpcSrvLaunch 只取裸指针下发 TaskRun
    HCCL_INFO(
        "[LegacyDpuThread][%s] TaskServiceArgs{deviceMem:%p; deviceMemSize:%lu; hostMem:%p; hostMemSize:%lu; "
        "taskExpMem:%p; commId:%s; devId:%u}",
        __func__, dputagAddr_, Hccl::SHARE_HBM_MEMORY_SIZE, hostShareBuf_, Hccl::SHARE_HBM_MEMORY_SIZE, taskExpMem_,
        args_.commId.c_str(), args_.devId);
    taskService_ = std::make_unique<Hccl::TaskService>(
        dputagAddr_, static_cast<int32_t>(Hccl::SHARE_HBM_MEMORY_SIZE), hostShareBuf_,
        static_cast<int32_t>(Hccl::SHARE_HBM_MEMORY_SIZE), taskExpMem_, args_.commId, args_.devId);
    CHK_SMART_PTR_NULL(taskService_);
    return HCCL_SUCCESS;
}

HcclResult LegacyDpuThread::PrepareDpuKernelResource()
{
    // 获取二进制文件路径
    std::string jsonPath;
    const char* ascendHomePath = getenv("ASCEND_HOME_PATH"); // getenv 未命中返回 nullptr，不可直接构造 std::string
    std::string getPath = ascendHomePath != nullptr ? ascendHomePath : "";
    if (!getPath.empty()) {
        jsonPath = getPath;
    } else {
        jsonPath = "/usr/local/Ascend/cann/";
        HCCL_WARNING("[LegacyDpuThread][%s] ENV:ASCEND_HOME_PATH is not set", __func__);
    }

    jsonPath += "/opp/built-in/op_impl/dpu/";
    HCCL_DEBUG("[LegacyDpuThread][%s] kernel folder path[%s]", __func__, jsonPath.c_str());

    // cpuKernelMode为1时，json命名需与so命名保持一致， 即libccl_dpu.json与libccl_dpu.so
    jsonPath += "libccl_dpu.json";
    char realPath[PATH_MAX] = {0};
    CHK_PRT_RET(
        realpath(jsonPath.c_str(), realPath) == nullptr,
        HCCL_ERROR("[LegacyDpuThread][%s]: %s is not a valid real path, err[%d]", __func__, jsonPath.c_str(), errno),
        HCCL_E_INTERNAL);
    HCCL_INFO("[LegacyDpuThread][%s] realPath: %s", __func__, realPath);

    aclrtBinHandle binHandle;
    aclrtBinaryLoadOptions options;
    aclrtBinaryLoadOption option;
    option.type = ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE; // AI CPU算子注册模式
    option.value.cpuKernelMode = 1; // 0 ：仅需要加载json，1 ：加载cpu so & json，2: LoadFromData
    options.numOpt = 1;
    options.options = &option;
    if (aclrtBinaryLoadFromFile(realPath, &options, &binHandle) != ACL_SUCCESS) {
        HCCL_ERROR("[LegacyDpuThread][%s] load binary from file error.", __func__);
        return HCCL_E_OPEN_FILE_FAILURE;
    }

    // 创建 dpustream
    aclError ret = aclrtCreateStreamWithConfig(&rtStream_, 0, ACL_STREAM_FAST_LAUNCH);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR("[LegacyDpuThread][%s] Create Local Stream Failed", __func__);
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO(
        "[LegacyDpuThread][%s] Create Stream para: deviceId[%d] streamId[%d]", __func__, Hccl::HrtGetDevice(),
        Hccl::HrtGetStreamId(rtStream_));

    // 查找核函数
    if (aclrtBinaryGetFunction(binHandle, "RunDpuRpcSrvLaunch", &funcHandle_) != ACL_SUCCESS) {
        HCCL_ERROR("[LegacyDpuThread][%s] Get Function Failed", __func__);
        return HCCL_E_INTERNAL;
    }

    return HCCL_SUCCESS;
}

HcclResult LegacyDpuThread::LaunchKernel()
{
    if (isKernelLaunched_) {
        HCCL_INFO("[LegacyDpuThread][%s] kernel already launched, skip", __func__);
        return HCCL_SUCCESS;
    }
    if (taskService_ == nullptr) {
        HCCL_ERROR("[LegacyDpuThread][%s] taskService is nullptr, call CreateTaskService first", __func__);
        return HCCL_E_INTERNAL;
    }

    // 下发前同步执行超时配置（HCCL_EXEC_TIMEOUT，秒）：内核侧 GetDpuExecTimeout 未下发时取默认值，宿主侧 launch 前覆盖
    SetDpuExecTimeout(Hccl::EnvConfig::GetInstance().GetRtsConfig().GetExecTimeOut());

    // 参数块由本类内部组装：收编后只带宿主侧已构造的 TaskService 裸指针 + deviceId
    Hccl::DpuKernelLaunchParam hostArgs{taskService_.get(), args_.devId};
    size_t argsSize = sizeof(hostArgs);

    // 下发
    HCCL_INFO("[LegacyDpuThread][%s] Launch Dpu Kernel", __func__);
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
            funcHandle_, DPU_KERNEL_NUM_BLOCKS, rtStream_, &cfg, &hostArgs, argsSize, &placeHolderArrays,
            placeHolderNum)
        != ACL_SUCCESS) {
        HCCL_ERROR("[LegacyDpuThread][%s] Launch Dpu Kernel Failed", __func__);
        // launch 失败仍处于 dpu 域，切回 npu ctx 再返回，避免调用线程滞留 dpu 域
        (void)aclrtSetCurrentContext(npuContext_);
        return HCCL_E_INTERNAL;
    }

    // 下发成功即代表 kernel 已在跑：先置位再做后续动作。否则切回 npu ctx 失败时析构链会因
    // !isKernelLaunched_ 短路，终止握手/流销毁/XPU reset 全被跳过，kernel 线程永不退出
    isKernelLaunched_ = true;
    commNum_++;

    // 切换回当前Ctx
    HCCL_INFO("[LegacyDpuThread][%s] Switch to Npu Ctx", __func__);
    if (ACL_SUCCESS != aclrtSetCurrentContext(npuContext_)) {
        HCCL_ERROR("[LegacyDpuThread][%s] Reset Current Ctx Failed", __func__);
        return HCCL_E_INTERNAL;
    }

    HCCL_INFO("[LegacyDpuThread][%s] Launch Dpu Kernel End", __func__);
    return HCCL_SUCCESS;
}

HcclResult LegacyDpuThread::WaitKernelTerminate()
{
    if (dputagHostPtr_ == nullptr) {
        HCCL_ERROR("[LegacyDpuThread][%s] dputag host ptr is nullptr, InitShmem not completed", __func__);
        return HCCL_E_PTR;
    }

    // 先读取是否有dpu线程回写的退出信号
    uint8_t flag = 0;
    errno_t ret = memcpy_s(&flag, sizeof(flag), dputagHostPtr_, sizeof(flag));
    if (ret != EOK) {
        HCCL_ERROR("[LegacyDpuThread][%s] Read Terminate TaskRun Signal Fail, return[%d]", __func__, ret);
        return HCCL_E_INTERNAL;
    }
    if (flag == DEVICE_SIGNAL_THIRD) {
        return HCCL_SUCCESS; // dpu线程已退出
    }
    flag = DEVICE_SIGNAL_SECOND;
    ret = memcpy_s(dputagHostPtr_, sizeof(flag), &flag, sizeof(flag));
    if (ret != EOK) {
        HCCL_ERROR("[LegacyDpuThread][%s] Terminate TaskRun Fail, return[%d]", __func__, ret);
        return HCCL_E_INTERNAL;
    }
    // 等待 dpu 线程回写退出信号，须有上限：dpu 线程可能在未回写的情况下已退出（TaskService 预检失败等），
    // 无界自旋会让析构路径挂死、进程退不出
    auto timeout = std::chrono::milliseconds(WAIT_CMD_TIMEOUT);
    auto startTime = std::chrono::steady_clock::now();
    do {
        ret = memcpy_s(&flag, sizeof(flag), dputagHostPtr_, sizeof(flag));
        if (ret != EOK) {
            HCCL_ERROR("[LegacyDpuThread][%s] Read Terminate TaskRun Signal Fail, return[%d]", __func__, ret);
            return HCCL_E_INTERNAL;
        }
        if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
            HCCL_WARNING("[LegacyDpuThread][%s] wait dpu kernel terminate timeout[%u ms].", __func__, WAIT_CMD_TIMEOUT);
            return HCCL_E_TIMEOUT;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while (flag != DEVICE_SIGNAL_THIRD);
    return HCCL_SUCCESS;
}

HcclResult LegacyDpuThread::RefreshStreamId()
{
    streamId_ = Hccl::HrtGetStreamId(rtStream_);
    HCCL_INFO("[LegacyDpuThread][%s] dpuStreamId[%u]", __func__, streamId_);
    return HCCL_SUCCESS;
}

HcclResult LegacyDpuThread::DeInit()
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
        HCCL_ERROR("[LegacyDpuThread][%s] WaitKernelTerminate failed, return[%d]", __func__, waitRet);
        recordErr(waitRet);
    }

    // 切换回 dpu ctx：流在 dpu 域下创建，销毁须在其 ctx 内进行
    aclError aclRet = aclrtSetCurrentContext(dpuContext_);
    if (ACL_SUCCESS != aclRet) {
        HCCL_ERROR("[LegacyDpuThread][%s] set dpu Ctx Failed, aclReturn[%d]", __func__, aclRet);
        recordErr(HCCL_E_RUNTIME);
    } else {
        // 销毁局部流
        aclRet = aclrtDestroyStreamForce(rtStream_);
        if (ACL_SUCCESS != aclRet) {
            HCCL_ERROR("[LegacyDpuThread][%s] Destroy Stream Failed, aclReturn[%d]", __func__, aclRet);
            recordErr(HCCL_E_RUNTIME);
        }
    }
    // 句柄已不可再用：同时失效包装它的 Stream，避免后续拿到悬垂指针
    stream_.reset();
    rtStream_ = nullptr;

    if (commNum_ > 1) {
        commNum_--;
    } else {
        // reset DPU kernel 线程
        HcclResult ret = Hccl::HrtResetXpuDevice(TEMP_DEV_TYPE_DPU, 0);
        if (HCCL_SUCCESS != ret) {
            HCCL_ERROR("[LegacyDpuThread][%s] ResetXpuDevice Failed, return[%d]", __func__, ret);
            recordErr(ret);
        }
    }
    // 切回 npu ctx：即使前面有失败也必须走完，否则本线程当前 context 会停在 dpu 域
    aclRet = aclrtSetCurrentContext(npuContext_);
    if (ACL_SUCCESS != aclRet) {
        HCCL_ERROR("[LegacyDpuThread][%s] set npu Ctx Failed, aclReturn[%d]", __func__, aclRet);
        recordErr(HCCL_E_RUNTIME);
    }

    return firstErr;
}

HcclResult LegacyDpuThread::TaskRegister(const std::string& taskType, Hccl::CallbackTemplate callback)
{
    if (taskService_ == nullptr) {
        HCCL_ERROR("[LegacyDpuThread][%s] taskService is nullptr, register failed", __func__);
        return HCCL_E_INTERNAL;
    }
    return taskService_->TaskRegister(taskType, callback);
}

HcclResult LegacyDpuThread::TaskUnRegister(const std::string& taskType)
{
    if (taskService_ == nullptr) {
        HCCL_ERROR("[LegacyDpuThread][%s] taskService is nullptr, unregister failed", __func__);
        return HCCL_E_INTERNAL;
    }
    return taskService_->TaskUnRegister(taskType);
}

HcclResult LegacyDpuThread::StopTaskexpService()
{
    // 幂等：~CollComm 已在 DestroyAicpuComm 前置停过（taskExpStopped_ 置位），析构二次进入直接返回，
    // 避免 device 侧 IsTaskExpStopped 早退不回写 flag 导致的二次握手超时
    if (taskExpStopped_) {
        return HCCL_SUCCESS;
    }
    // 非 dpu 场景（kernel 未拉起）或异常区未申请，无共享内存可停止
    if (!isKernelLaunched_ || taskExpMem_ == nullptr) {
        return HCCL_SUCCESS;
    }
    uint8_t flag = 1;
    errno_t ret = memcpy_s(taskExpMem_, sizeof(flag), &flag, sizeof(flag)); // taskexception 首字节作为停止标志位
    if (ret != EOK) {
        HCCL_ERROR("[LegacyDpuThread][%s] set stop flag failed: %d", __func__, ret);
        return HCCL_E_INTERNAL;
    }
    auto timeout = std::chrono::milliseconds(WAIT_CMD_TIMEOUT);
    auto startTime = std::chrono::steady_clock::now();
    do { // 等待 aicpu 背景线程 response
        ret = memcpy_s(&flag, sizeof(flag), taskExpMem_, sizeof(flag));
        if (ret != EOK) {
            HCCL_ERROR("[LegacyDpuThread][%s] read device response failed: %d", __func__, ret);
            return HCCL_E_INTERNAL;
        }
        if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
            HCCL_WARNING(
                "[LegacyDpuThread][%s] wait device destroy response timeout[%u ms].", __func__, WAIT_CMD_TIMEOUT);
            return HCCL_E_TIMEOUT;
        }
    } while (flag != 0);
    taskExpStopped_ = true;
    return HCCL_SUCCESS;
}

} // namespace hccl
