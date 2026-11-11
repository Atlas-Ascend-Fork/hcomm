/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "comm_engine/aicpu_launch_manager_pub.h"
#include "thread.h"
#include "aicpu_thread_launch_mgr.h"
#include "common/dev_type_utils_pub.h"
#include "common/cast_utils_pub.h"
#include "adapter_rts_common.h"
#include "mem_device_pub.h"
#include "launch_aicpu.h"
#include "cpu_thread.h"
#include "cpu_device_queue_namecard_restore_kernel.h"
#include "device/cpu_device_notify_addr_update_kernel.h"
#include "stream_pub.h"
#include "hccl_common.h"
#include "comm_configer.h"
#include "base_comm_mgr.h"
#include "base_comm_node_accessors.h"
#include "hcom_host_profiling.h"
#include "adapter_prof.h"
#include "env_config/env_config_v2.h"
#include "common/cast_utils_pub.h"

namespace hccl {
template <typename OpParam, typename ApiParam>
HcclResult AicpuLaunchMgr::KernelLaunch(OpParam& opParam, ApiParam& apiParam, rtStream_t aicpuInitStream)
{
    return HCCL_SUCCESS;
}

// 以下为仅本文件使用的实现细节（原 AicpuLaunchMgr 内部接口），不再出现在任何头文件
static HcclResult KernelLaunchAicpuCustom(
    uint64_t context, const std::string& kernelName, rtStream_t aicpuInitStream, aclrtBinHandle binCustomHandle)
{
    uint32_t envTimeout = 0;
    auto isGe950 = hcomm::EqualToOrGreaterThan950();
    CHK_PRT_RET(!isGe950.has_value(), HCCL_ERROR("[%s] get device type failed.", __func__), HCCL_E_RUNTIME);
    if (*isGe950) {
        envTimeout = Hccl::EnvConfig::GetInstance().GetRtsConfig().GetExecTimeOut();
    } else {
        envTimeout = hcomm::GetCommConfiger().GetCommConfigExecTimeOut("");
    }
    u16 timeOut = envTimeout + 25 > std::numeric_limits<u16>::max() ? std::numeric_limits<u16>::max() :
                                                                      envTimeout + 25; // 多25s，避免超时
    CHK_RET(
        AicpuAclKernelLaunch(aicpuInitStream, &context, sizeof(context), binCustomHandle, kernelName, true, timeOut));
    return HCCL_SUCCESS;
}

static HcclResult CreateLocalStream(Stream& localStream)
{
    HCCL_INFO("[%s] create local stream", __func__);
    localStream = Stream(StreamType::STREAM_TYPE_ONLINE);
    constexpr u32 aicpuStreamMode = 1;
    CHK_RET(hrtStreamSetMode(localStream.ptr(), aicpuStreamMode));
    return HCCL_SUCCESS;
}

namespace {
    // host 侧线程参数打包：定长头 + entry 表 + 变长 uid 缓冲（下发前按实际大小拼入一块连续 device 内存）
    struct ThreadParamPack {
        ThreadMgrAicpuParam header{};
        DeviceMem deviceHandle;
        std::vector<ThreadParamEntry> entries;
        std::vector<char> uidBuf;
    };
} // namespace

static HcclResult PrepareThreadMgrParam(
    const std::vector<std::shared_ptr<Thread>>& newThreads, const ThreadKernelLaunchConfig& config,
    ThreadParamPack& pack)
{
    HCCL_INFO("[%s] fill opParam", __func__);
    CHK_PRT_RET(newThreads.empty(), HCCL_ERROR("[%s] newThreads is empty", __func__), HCCL_E_PARA);
    // pack 中仅 header 为 POD（deviceHandle/entries/uidBuf 为智能指针与容器，不可整体 memset），
    // 清零 header 含 padding，保证 H2D 下发内容确定
    errno_t mRet = memset_s(&pack.header, sizeof(pack.header), 0, sizeof(pack.header));
    CHK_PRT_RET(mRet != EOK, HCCL_ERROR("[%s] memset_s failed, return [%d].", __func__, mRet), HCCL_E_MEMORY);
    pack.header.threadNum = newThreads.size();

    // 拷贝 commId
    errno_t sRet = strncpy_s(pack.header.hcomId, HCOMID_MAX_SIZE, config.commId.c_str(), config.commId.length());
    CHK_PRT_RET(sRet != EOK, HCCL_ERROR("[%s] strncpy_s failed, return [%d].", __func__, sRet), HCCL_E_MEMORY);
    pack.header.hcomId[HCOMID_MAX_SIZE - 1] = '\0';

    // 从 Thread 对象获取 engine（同一批 threads 属于同一 engine）
    pack.header.engine = newThreads[0]->GetCommEngine();
    pack.header.threadType = newThreads[0]->GetThreadType();
    // 参数块只带一个 threadType，device 侧按它建整批实体，混批会建错实体类型
    for (u32 i = 1; i < pack.header.threadNum; ++i) {
        CHK_PRT_RET(
            newThreads[i]->GetThreadType() != pack.header.threadType,
            HCCL_ERROR(
                "[%s] thread[%u] threadType[%d] differs from batch[%d], return [%d].", __func__, i,
                static_cast<int32_t>(newThreads[i]->GetThreadType()), static_cast<int32_t>(pack.header.threadType),
                HCCL_E_PARA),
            HCCL_E_PARA);
    }

    // 收集线程 unique id 填 entry 表：uidOffset 为 u32 相对偏移，累计偏移与 uid 单条长度均须
    // 可被 u32 表达（偏移赋值前显式检查，不静默截断）。orderCtrlFlag 查进程级集合（key=host对象句柄）
    // 填充，时序天然保证：创建路径kernel先于acquire出口登记→flag=0；导出miss路径kernel晚于登记→flag=1
    // （其它路径恒不在集合）
    pack.entries.assign(newThreads.size(), ThreadParamEntry{});
    pack.uidBuf.clear();
    for (u32 i = 0; i < pack.header.threadNum; ++i) {
        const std::string& uid = newThreads[i]->GetUniqueId();
        // CPU 线程的设备实体按 threadType 分派建立，不读 uid blob，允许为空
        CHK_PRT_RET(
            uid.empty() && newThreads[i]->GetThreadType() != THREAD_TYPE_CPU,
            HCCL_ERROR("[%s] thread[%u] uid is empty, return [%d].", __func__, i, HCCL_E_INTERNAL), HCCL_E_INTERNAL);
        CHK_PRT_RET(
            uid.size() > UINT32_MAX,
            HCCL_ERROR("[%s] uid.size()[%zu] exceeds u32 limit, return [%d].", __func__, uid.size(), HCCL_E_MEMORY),
            HCCL_E_MEMORY);
        // 累计偏移赋值前校验，防止截断后静默下发错误 uidOffset
        CHK_PRT_RET(
            pack.uidBuf.size() > UINT32_MAX,
            HCCL_ERROR(
                "[%s] uid accumulated size[%zu] exceeds u32 offset limit, return [%d].", __func__, pack.uidBuf.size(),
                HCCL_E_MEMORY),
            HCCL_E_MEMORY);
        pack.entries[i].uidOffset = static_cast<uint32_t>(pack.uidBuf.size());
        pack.entries[i].uidLen = static_cast<uint32_t>(uid.size());
        pack.entries[i].orderCtrlFlag
            = hccl::IsOrderCtrlThread(ReinterpretAs<ThreadHandle>(newThreads[i].get())) ? 1U : 0U;
        pack.uidBuf.insert(pack.uidBuf.end(), uid.begin(), uid.end());
    }
    HCCL_INFO(
        "[%s] threadNum[%u], entrySize[%zu], paramBufSize[%zu]", __func__, pack.header.threadNum, pack.entries.size(),
        pack.uidBuf.size());

    // 分配 Entity 地址输入缓冲区：Host 只 H2D 写入，Device 侧回填 compatThreadHandle。
    size_t handleLen = sizeof(ThreadHandle) * newThreads.size();
    pack.deviceHandle = DeviceMem::alloc(handleLen);
    CHK_SMART_PTR_NULL(pack.deviceHandle);
    pack.header.deviceHandle = pack.deviceHandle.ptr();

    // 基础通信需要设备信息
    if (config.needDeviceInfo) {
        CHK_RET(hrtGetDevice(&pack.header.deviceLogicId));
        DevType devType;
        CHK_RET(hrtGetDeviceType(devType));
        pack.header.deviceType = static_cast<u32>(devType);
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuThreadLaunchMgr::ThreadKernelLaunchImpl(
    std::vector<std::shared_ptr<Thread>>& newThreads, std::unique_ptr<ThreadHandle[]>& aicpuHandle,
    const ThreadKernelLaunchConfig& config)
{
    uint64_t beginTime = (config.needProfiling ? hrtMsprofSysCycleTime() : 0);

    // Step 1. 创建局部 stream
    Stream localStream;
    CHK_RET(CreateLocalStream(localStream));

    // Step 2. 填写参数并分配设备内存（threadNum/uid 总长按实际大小，无定长上限，底层失败按错误码返回）
    ThreadParamPack pack;
    CHK_RET(PrepareThreadMgrParam(newThreads, config, pack));

    // Step 3. 初始化和补充 Notify 均传入调用方持有的 ThreadEntity 地址。
    size_t handleLen = sizeof(ThreadHandle) * newThreads.size();
    for (size_t i = 0; i < newThreads.size(); ++i) {
        CHK_PRT_RET(aicpuHandle[i] == 0, HCCL_ERROR("[%s] thread handle[%zu] is null", __func__, i), HCCL_E_PTR);
    }
    CHK_RET(hrtMemSyncCopy(
        pack.header.deviceHandle, handleLen, aicpuHandle.get(), handleLen,
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

    // Step 4. 分配连续参数块（header + entry 表 + 变长 uid），按实际大小分三次 H2D
    size_t entryLen = pack.entries.size() * sizeof(ThreadParamEntry);
    size_t blobSize = sizeof(pack.header) + entryLen + pack.uidBuf.size();
    DeviceMem paramBlob = DeviceMem::alloc(blobSize);
    CHK_SMART_PTR_NULL(paramBlob);
    pack.header.entryTableAddr = reinterpret_cast<uint64_t>(paramBlob.ptr()) + sizeof(pack.header);
    pack.header.paramBufAddr = pack.header.entryTableAddr + entryLen;
    pack.header.paramBufSize = pack.uidBuf.size();
    CHK_RET(hrtMemSyncCopy(
        paramBlob.ptr(), sizeof(pack.header), &pack.header, sizeof(pack.header),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    if (entryLen > 0) {
        void* entryDst = reinterpret_cast<void*>(static_cast<uintptr_t>(pack.header.entryTableAddr));
        CHK_RET(hrtMemSyncCopy(
            entryDst, entryLen, pack.entries.data(), entryLen, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    }
    if (pack.uidBuf.size() > 0) {
        void* bufDst = reinterpret_cast<void*>(static_cast<uintptr_t>(pack.header.paramBufAddr));
        CHK_RET(hrtMemSyncCopy(
            bufDst, pack.uidBuf.size(), pack.uidBuf.data(), pack.uidBuf.size(),
            HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    }

    // Step 5. 调用 KernelLaunch
    uint64_t context = reinterpret_cast<uint64_t>(paramBlob.ptr());
    HCCL_INFO("AicpuLaunchMgr::%s, call KernelLaunch", __func__);
    HcclResult ret = KernelLaunchAicpuCustom(context, config.kernelName.c_str(), localStream.ptr(), config.binHandle);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[AicpuLaunchMgr][%s] KernelLaunch failed, return [%d].", __func__, ret), ret);

    // Step 6. 等待流完成
    CHK_RET(hcclStreamSynchronize(localStream.ptr(), config.timeoutSec));

    // 性能分析上报
    if (config.needProfiling) {
        HcommProfilingReportKernel(beginTime, config.kernelName.c_str());
    }
    return HCCL_SUCCESS;
}

// 集合通信使用，待归一到基础通信接口
HcclResult AicpuThreadLaunchMgr::ThreadKernelLaunchForComm(
    std::vector<std::shared_ptr<Thread>>& newThreads, const std::string& commId,
    std::unique_ptr<ThreadHandle[]>& aicpuHandle, aclrtBinHandle binHandle)
{
    uint32_t envTimeout = 0;
    auto isGe950 = hcomm::EqualToOrGreaterThan950();
    CHK_PRT_RET(!isGe950.has_value(), HCCL_ERROR("[%s] get device type failed.", __func__), HCCL_E_RUNTIME);
    if (*isGe950) {
        envTimeout = Hccl::EnvConfig::GetInstance().GetRtsConfig().GetExecTimeOut();
    } else {
        envTimeout = hcomm::GetCommConfiger().GetCommConfigExecTimeOut(commId);
    }
    uint32_t timeOut = envTimeout + 25 > std::numeric_limits<uint32_t>::max() ? std::numeric_limits<uint32_t>::max() :
                                                                                envTimeout + 25; // 多25s，避免超时
    ThreadKernelLaunchConfig config(commId, binHandle, "RunAicpuIndOpThreadInit", false, timeOut, true, false);
    return ThreadKernelLaunchImpl(newThreads, aicpuHandle, config);
}

// 基础通信使用
HcclResult AicpuThreadLaunchMgr::ThreadKernelLaunchForBase(
    std::vector<std::shared_ptr<Thread>>& newThreads, std::unique_ptr<ThreadHandle[]>& aicpuHandle,
    aclrtBinHandle binHandle)
{
    constexpr uint32_t defaultTimeOutSec = 120;
    ThreadKernelLaunchConfig config("", binHandle, "RunAicpuThreadInit", true, defaultTimeOutSec, false, false);
    return ThreadKernelLaunchImpl(newThreads, aicpuHandle, config);
}

// 补充notify使用
HcclResult AicpuThreadLaunchMgr::SupplementNotifyKernelLaunch(
    std::vector<std::shared_ptr<Thread>>& newThreads, const std::string& commId,
    std::unique_ptr<ThreadHandle[]>& aicpuHandle, aclrtBinHandle binHandle)
{
    constexpr uint32_t defaultTimeOutSec = 120;
    ThreadKernelLaunchConfig config(
        commId, binHandle, "RunAicpuThreadSupplementNotify", true, defaultTimeOutSec, false, true);
    return ThreadKernelLaunchImpl(newThreads, aicpuHandle, config);
}

// 名片只在 host 侧算好后按句柄投给 device 侧实体；投递失败则后续任务无处可查，按失败返回
HcclResult AicpuThreadLaunchMgr::CpuQueueNamecardLaunch(
    std::vector<std::shared_ptr<Thread>>& newThreads, const ThreadHandle* threadHandles, const std::string& commId,
    aclrtBinHandle binHandle)
{
    uint32_t envTimeout = 0;
    auto isGe950 = hcomm::EqualToOrGreaterThan950();
    CHK_PRT_RET(!isGe950.has_value(), HCCL_ERROR("[%s] get device type failed.", __func__), HCCL_E_RUNTIME);
    if (*isGe950) {
        envTimeout = Hccl::EnvConfig::GetInstance().GetRtsConfig().GetExecTimeOut();
    } else {
        envTimeout = hcomm::GetCommConfiger().GetCommConfigExecTimeOut(commId);
    }
    u16 timeOut = envTimeout + 25 > std::numeric_limits<u16>::max() ? std::numeric_limits<u16>::max() :
                                                                      envTimeout + 25; // 多25s，避免超时

    for (size_t i = 0; i < newThreads.size(); ++i) {
        auto cpuThread = std::dynamic_pointer_cast<CpuThread>(newThreads[i]);
        if (cpuThread == nullptr || cpuThread->GetTaskPipe() == nullptr) {
            continue; // 未走队列通道的线程没有名片可投
        }
        DpuQueueNamecardParam param{};
        CHK_RET(cpuThread->GetDevSyncLayout(
            param.layout, param.sqDesc, param.cqDesc, param.notifyHostAddr, param.notifyDevAddr));
        param.threadHandle = threadHandles[i];
        param.notifyNum = cpuThread->GetNotifyNum();

        Stream namecardStream;
        CHK_RET(CreateLocalStream(namecardStream));
        CHK_RET(AicpuAclKernelLaunch(
            namecardStream.ptr(), &param, sizeof(param), binHandle, DPU_QUEUE_NAMECARD_RESTORE_KERNEL_NAME, true,
            timeOut));
        // 等名片落地再返回：申请入口返回即代表 device 侧已能按句柄找到这对 SQ/CQ
        CHK_RET(hcclStreamSynchronize(namecardStream.ptr(), timeOut));
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuThreadLaunchMgr::CpuNotifyAddrUpdateLaunch(
    CpuThread& cpuThread, ThreadHandle threadHandle, const std::string& commId)
{
    CHK_RET(hcomm::GetAicpuCommEngineBinProc().EnsureKernelBinLoaded(COMM_ENGINE_AICPU));
    aclrtBinHandle binHandle = hcomm::GetAicpuCommEngineBinProc().GetBinHandle();
    CHK_PRT_RET(
        binHandle == nullptr, HCCL_ERROR("[%s] aicpu kernel bin is not loaded, comm[%s]", __func__, commId.c_str()),
        HCCL_E_PTR);

    CpuThreadNotifyAddrUpdateParam param{};
    CHK_RET(cpuThread.GetNotifyShmemAddrs(param.notifyHostAddr, param.notifyDevAddr));
    param.threadHandle = threadHandle;
    param.notifyNum = cpuThread.GetNotifyNum();

    Stream localStream;
    CHK_RET(CreateLocalStream(localStream));
    CHK_RET(AicpuAclKernelLaunch(
        localStream.ptr(), &param, sizeof(param), binHandle, CPU_THREAD_NOTIFY_ADDR_UPDATE_KERNEL_NAME, true,
        NOTIFY_DEFAULT_WAIT_TIME));
    // 同步返回后 device 侧才按新地址收发，调用方补配完即可继续用这对线程
    CHK_RET(hcclStreamSynchronize(localStream.ptr(), CommConfiger::GetInstance().GetCommConfigExecTimeOut(commId)));
    HCCL_INFO(
        "[%s] deliver notify addr update done, comm[%s], kernel[%s], notifyNum[%u]", __func__, commId.c_str(),
        CPU_THREAD_NOTIFY_ADDR_UPDATE_KERNEL_NAME, param.notifyNum);
    return HCCL_SUCCESS;
}

HcclResult
AicpuThreadLaunchMgr::ThreadKernelLaunchDestroy(ThreadHandle* threadHandles, uint32_t listNum, aclrtBinHandle binHandle)
{
    HCCL_INFO(
        "[AicpuLaunchMgr][%s] Start. listNum=%u, threadHandles=%p, binHandle=%p", __func__, listNum, threadHandles,
        binHandle);
    // Step 1. 创建局部 stream
    Stream localStream;
    CHK_RET(CreateLocalStream(localStream));

    // Step 2. 填写 opParam
    ThreadMgrAicpuParam opParam{};
    (void)memset_s(&opParam, sizeof(opParam), 0, sizeof(opParam));
    opParam.threadNum = listNum;
    size_t handleLen = sizeof(ThreadHandle) * listNum;
    DeviceMem deviceHandle = DeviceMem::alloc(handleLen);
    CHK_SMART_PTR_NULL(deviceHandle);
    opParam.deviceHandle = deviceHandle.ptr();
    CHK_RET(hrtMemSyncCopy(
        deviceHandle.ptr(), handleLen, threadHandles, handleLen, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

    // Step 3. 调用 KernelLaunch
    std::string kernelName = "RunAicpuThreadDestroy";
    HCCL_INFO(
        "[AicpuLaunchMgr][%s] call KernelLaunch, kernelName=%s, stream=%p, binHandle=%p", __func__, kernelName.c_str(),
        localStream.ptr(), binHandle);
    DeviceMem addr = DeviceMem::alloc(sizeof(opParam));
    CHK_RET(hrtMemSyncCopy(
        addr.ptr(), sizeof(opParam), &opParam, sizeof(opParam), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    uint64_t context = reinterpret_cast<uint64_t>(addr.ptr());
    HcclResult ret = KernelLaunchAicpuCustom(context, kernelName.c_str(), localStream.ptr(), binHandle);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[AicpuLaunchMgr][%s] KernelLaunch failed, return [%d].", __func__, ret), ret);

    // Step 4. 等待流完成
    constexpr uint32_t defaultTimeOutSec = 120;
    CHK_RET(hcclStreamSynchronize(localStream.ptr(), defaultTimeOutSec));
    return HCCL_SUCCESS;
}
} // namespace hccl
