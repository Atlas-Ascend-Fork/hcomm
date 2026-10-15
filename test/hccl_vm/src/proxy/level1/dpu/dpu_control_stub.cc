/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * Description: HostDPU 北向劫持（level1）控制面打桩函数。
 *              替代真实 libhcomm.so 的 DPU 生命周期接口：
 *              - HcclDevMemAcquire：分配 DPUTAG 共享内存（真实 aclrtMalloc 经
 *                level2 落跨进程共享内存），完成 HcclMem/Communicator/
 *                DpuMemRegistry 三处登记，并惰性拉起 DPU 执行器；
 *              - HcclTaskRegister / HcclTaskUnRegister：DPU 回调注册/注销；
 *              - HcclGetDpuSteamIdV2：查询 DPU 任务归属流；
 *              - HcommThreadSynchronize / HcommThreadJoin：线程同步（no-op）；
 *              - HcommSetNotifyWaitTimeOut / HcommThreadResAcquireTimeOut：超时配置；
 *              - HcclChannelDestroy：通道销毁（级联释放 Notify）。
 *              铁律：本文件接口禁止 dlsym(RTLD_NEXT) 透传——真实实现会把
 *              level0 返回的 DB 主键 comm 句柄 cast 为 HcclCommunicator*，透传必崩。
 * Create: 2026-10-09
 */

#define HCCL_VM_MODULE "DPU_CTRL_STUB"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include "acl/acl_rt.h"
#include "ascend_hal.h"
#include "hccl/hccl_res.h"
#include "level1_proxy_common.h"
#include "runtime_state/db_sim_runner_common.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "runtime_state/sim_models.h"
#include "sim_log.h"

#include "dpu_level1_common.h"
#include "dpu_level1_res.h"

namespace {

/* DPUTAG 共享内存标签（与 hcomm communicator_impl.cc 的 DPUTAG 常量一致）。 */
constexpr const char* DPU_SHMEM_TAG = "DPUTAG";

/* 与 sim::runtime::HcclChannel.notifyId[64] 数组容量对齐。 */
constexpr uint32_t MAX_NOTIFY_PER_CHANNEL = 64;

/**
 * @brief 按 (commId, memTagHash) 查询已登记的 HcclMem 内存记录。
 * @param commId   输入：通信域 id（Communicator 表主键）。
 * @param tagHash  输入：memTag 的哈希值（HashCtxTag）。
 * @param out      输出：命中时回填记录。
 * @return 命中返回 true，否则 false。
 */
bool FindDpuMemRecord(uint64_t commId, uint64_t tagHash, sim::runtime::HcclMem& out)
{
    const auto records = sim::runtime::Db::GetByPred<sim::runtime::HcclMem>(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&sim::runtime::HcclMem::commId, commId),
        HcclSim::Storage::Eq(&sim::runtime::HcclMem::memTag, tagHash)));
    if (!records.ok() || records->empty()) {
        return false;
    }
    out = records->front();
    return true;
}

/**
 * @brief DPU 共享内存失败路径回滚（解除 host 注册 + 释放设备内存）。
 * @param dev      输入：aclrtMalloc 返回的设备映射地址。
 * @param deviceId 输入：设备 id。
 */
void RollbackDpuShmem(void* dev, uint32_t deviceId)
{
    (void)halHostUnregister(dev, deviceId);
    (void)aclrtFree(dev);
}

/**
 * @brief 分配 DPU 共享内存并完成三处登记（HcclMem / Communicator / DpuMemRegistry）。
 * @param commId    输入：通信域 id。
 * @param deviceId  输入：通信域所属设备 id（Device 表主键）。
 * @param memTag    输入：内存标签字符串（"DPUTAG"）。
 * @param tagHash   输入：标签哈希。
 * @param devAddr   输出：设备映射地址（aclrtMalloc 返回值，返回给 hccl 下发设备侧）。
 * @param hostPtr   输出：host 可解引用地址（halHostRegister 结果，执行器轮询用）。
 * @return HcclResult 成功返回 HCCL_SUCCESS；分配或登记失败返回对应错误码。
 * @note 复刻真实 GetKFCWorkSpaceVA 的两步分配：aclrtMalloc（level2 跨进程共享
 *       内存）+ halHostRegister（同物理内存的 host 侧注册，同时使
 *       GetDevMapperAddrByHostPtr 反向可查）；失败路径回滚，不留半初始化状态。
 */
HcclResult AllocAndRegisterDpuShmem(
    uint64_t commId, uint32_t deviceId, const char* memTag, uint64_t tagHash, void** devAddr, void** hostPtr)
{
    // 【核心步骤 1】经 level2 aclrtMalloc 分配跨进程共享内存，得到设备映射地址
    void* dev = nullptr;
    if (aclrtMalloc(&dev, HcclSim::DpuSim::DPU_SHMEM_ALLOC_SIZE, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS
        || dev == nullptr) {
        HCCL_VM_ERROR("{}: aclrtMalloc failed, commId={:d}", __func__, commId);
        return HCCL_E_MEMORY;
    }

    // 【核心步骤 2】halHostRegister 取得同物理内存的 host 可解引用地址
    // （执行器在 host 进程直接读写；语义对齐真实 HrtHalHostRegister(va_) -> accessVA_）
    void* host = nullptr;
    if (halHostRegister(dev, HcclSim::DpuSim::DPU_SHMEM_ALLOC_SIZE, DEV_SVM_MAP_HOST, deviceId, &host) != DRV_ERROR_NONE
        || host == nullptr) {
        HCCL_VM_ERROR("{}: halHostRegister failed, commId={:d}, devAddr={:p}", __func__, commId, dev);
        (void)aclrtFree(dev);
        return HCCL_E_INTERNAL;
    }

    // 【核心步骤 3】登记 HcclMem 记录（addr 记设备映射地址：重复获取时直接回传，
    // 与 hccl 侧使用面一致；host 侧地址经 DpuMemRegistry/Communicator 另行登记）
    sim::runtime::HcclMem mem{};
    mem.commId = commId;
    mem.memTag = tagHash;
    mem.memType = static_cast<uint64_t>(COMM_MEM_TYPE_DEVICE);
    mem.addr = reinterpret_cast<uint64_t>(dev);
    mem.size = HcclSim::DpuSim::DPU_SHMEM_ALLOC_SIZE;
    std::strncpy(mem.mem_tag_str, memTag, sizeof(mem.mem_tag_str) - 1);
    mem.mem_tag_str[sizeof(mem.mem_tag_str) - 1] = '\0';
    if (!sim::runtime::Db::Add<sim::runtime::HcclMem>(mem).ok()) {
        HCCL_VM_ERROR("{}: add HcclMem failed, commId={:d}", __func__, commId);
        RollbackDpuShmem(dev, deviceId);
        return HCCL_E_INTERNAL;
    }

    // 【核心步骤 4】Communicator 回填 dpuMemAddr/dpuMemSize（DB 侧权威观测记录）
    (void)sim::runtime::Db::Update<sim::runtime::Communicator>(
        HcclSim::Storage::Eq(&sim::runtime::Communicator::id, commId),
        HcclSim::Storage::Set(&sim::runtime::Communicator::dpu_mem_addr, reinterpret_cast<uint64_t>(host)),
        HcclSim::Storage::Set(&sim::runtime::Communicator::dpu_mem_size, HcclSim::DpuSim::DPU_SHMEM_ALLOC_SIZE));

    // 【核心步骤 5】登记 DpuMemRegistry（host 视角折半：前 1MB 请求区、后 1MB 响应区）
    HcclSim::DpuSim::DpuMemRegion region{};
    region.commId = commId;
    region.npu2dpu = static_cast<uint8_t*>(host);
    region.dpu2npu = static_cast<uint8_t*>(host) + HcclSim::DpuSim::DPU_SHMEM_HALF_SIZE;
    region.halfSize = HcclSim::DpuSim::DPU_SHMEM_HALF_SIZE;
    region.devAddr = dev;
    if (!HcclSim::DpuSim::DpuMemRegistry::Instance().Add(region)) {
        HCCL_VM_ERROR("{}: add DpuMemRegistry failed, commId={:d}", __func__, commId);
        RollbackDpuShmem(dev, deviceId);
        return HCCL_E_INTERNAL;
    }

    *devAddr = dev;
    *hostPtr = host;
    return HCCL_SUCCESS;
}

/**
 * @brief 归一化超时参数（对齐真实实现：float 向零截断）。
 * @param timeOut    输入：超时时间（秒）。
 * @param normalized 输出：截断后的整数值。
 * @return int32_t 成功返回 0；非法（负数/NaN/超上限）返回 HCCL_E_PARA。
 */
int32_t NormalizeDpuTimeout(float timeOut, uint32_t& normalized)
{
    if (std::isnan(timeOut) || timeOut < 0.0F || timeOut > static_cast<float>(std::numeric_limits<uint32_t>::max())) {
        return static_cast<int32_t>(HCCL_E_PARA);
    }
    normalized = static_cast<uint32_t>(timeOut);
    return 0;
}

} // namespace

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 按标签获取设备工作区内存（HostDPU 场景为 DPUTAG 共享内存）。
 *
 * 替代真实 libhcomm 的 GetKFCWorkSpaceVA：首次调用经 aclrtMalloc 真实分配
 * 2MB 跨进程共享内存（得设备映射地址）+ halHostRegister（得 host 可解引用
 * 地址）并完成三处登记，DPUTAG 首分配时惰性拉起 DPU 执行器；后续调用返回
 * 登记缓存。返回给调用方的地址为**设备映射地址**（level2 aclrtMalloc 在
 * AICPU 模式下返回即设备地址，设备侧直接使用，无需 host→device 转换），
 * 返回的尺寸为**实际分配尺寸**（保证 hccl 折半 dpu2Npu=base+size/2 落在
 * 映射范围内）。
 *
 * @param comm       输入：通信域句柄（level0 下发的 DB 主键）。
 * @param memTag     输入：内存标签（HostDPU 固定为 "DPUTAG"）。
 * @param size       输入输出：入参为请求尺寸（hccl 硬编码 100MB，仅作校验），
 *                   出参为实际分配尺寸（2MB）。
 * @param addr       输出：内存地址（设备映射地址）。
 * @param newCreated 输出：本次是否新分配（可为 nullptr）。
 * @return HcclResult 成功返回 HCCL_SUCCESS，否则返回对应错误码。
 */
HcclResult HcclDevMemAcquire(HcclComm comm, const char* memTag, uint64_t* size, void** addr, bool* newCreated)
{
    if (comm == nullptr || memTag == nullptr || size == nullptr || addr == nullptr) {
        HCCL_VM_ERROR("{}: comm/memTag/size/addr is nullptr", __func__);
        return HCCL_E_PTR;
    }
    if (*size == 0 || std::strlen(memTag) > HCCL_RES_TAG_MAX_LEN) {
        HCCL_VM_ERROR("{}: invalid args, size={:d}, memTagLen={:d}", __func__, *size, std::strlen(memTag));
        return HCCL_E_PARA;
    }

    const uint64_t commId = reinterpret_cast<uint64_t>(comm);
    const auto optComm = sim::runtime::Db::GetById<sim::runtime::Communicator>(commId);
    if (!optComm.ok()) {
        HCCL_VM_ERROR("{}: communicator {:d} not found", __func__, commId);
        return HCCL_E_INTERNAL;
    }

    // 【核心步骤 1】命中登记直接回传（幂等获取；HcclMem.addr 即设备映射地址，无需转换）
    const uint64_t tagHash = HashCtxTag(memTag);
    sim::runtime::HcclMem existing{};
    if (FindDpuMemRecord(commId, tagHash, existing)) {
        *addr = reinterpret_cast<void*>(static_cast<uintptr_t>(existing.addr));
        *size = existing.size;
        if (newCreated != nullptr) {
            *newCreated = false;
        }
        HCCL_VM_INFO("{}: reuse registered mem, commId={:d}, memTag={}", __func__, commId, memTag);
        return HCCL_SUCCESS;
    }

    // 【核心步骤 2】新分配 + 三处登记（aclrtMalloc 得设备地址，halHostRegister 得 host 地址）
    void* devAddr = nullptr;
    void* hostPtr = nullptr;
    const uint32_t deviceId = static_cast<uint32_t>(optComm->device_id);
    const HcclResult ret = AllocAndRegisterDpuShmem(commId, deviceId, memTag, tagHash, &devAddr, &hostPtr);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }

    // 【核心步骤 3】DPUTAG 首分配：拉起 DPU 执行器（每进程一次，幂等）
    if (std::strcmp(memTag, DPU_SHMEM_TAG) == 0 && !HcclSim::DpuSim::EnsureDpuExecutorStarted(deviceId)) {
        HCCL_VM_ERROR("{}: start dpu executor failed, commId={:d}", __func__, commId);
        return HCCL_E_INTERNAL;
    }

    // 直接返回设备映射地址：level2 aclrtMalloc 在 AICPU 模式下返回的即是设备映射地址
    // （VirMemBlock.is_dev_access=1），设备侧经 GetDevMapperAddrByDevAddr 原样使用；
    // 与南向透传桩的 GetDevMapperAddrByHostPtr 转换不同——那只是真实 libhcomm 返回
    // host 地址后的回换算术，北向自分配场景无此必要。
    *addr = devAddr;
    *size = HcclSim::DpuSim::DPU_SHMEM_ALLOC_SIZE;
    if (newCreated != nullptr) {
        *newCreated = true;
    }
    HCCL_VM_INFO("{} success, commId={:d}, memTag={}, addr={:p}, size={:d}", __func__, commId, memTag, *addr, *size);
    return HCCL_SUCCESS;
}

/**
 * @brief 注册 DPU 算法回调（对应真实 TaskService::TaskRegister）。
 *
 * 将 (commId, msgTag) -> 回调函数指针登记到进程内 DpuCallbackRegistry，
 * 并兜底幂等拉起 DPU 执行器（正常时序下 HcclDevMemAcquire 已拉起）。
 *
 * @param comm   输入：通信域句柄。
 * @param msgTag 输入：任务标识（hccl 算法实例 algTag）。
 * @param cb     输入：回调函数（HcclLaunchDPUKernel）。
 * @return HcclResult 成功返回 HCCL_SUCCESS，否则返回对应错误码。
 */
HcclResult HcclTaskRegister(HcclComm comm, const char* msgTag, HcclSim::DpuSim::DpuCallback cb)
{
    if (comm == nullptr || msgTag == nullptr || cb == nullptr) {
        HCCL_VM_ERROR("{}: comm/msgTag/cb is nullptr", __func__);
        return HCCL_E_PTR;
    }
    const uint64_t commId = reinterpret_cast<uint64_t>(comm);
    const auto optComm = sim::runtime::Db::GetById<sim::runtime::Communicator>(commId);
    if (!optComm.ok()) {
        HCCL_VM_ERROR("{}: communicator {:d} not found", __func__, commId);
        return HCCL_E_INTERNAL;
    }
    if (!HcclSim::DpuSim::DpuCallbackRegistry::Instance().Register(commId, msgTag, cb)) {
        return HCCL_E_INTERNAL;
    }
    // 兜底拉起执行器（首次分配已拉起的场景为幂等 no-op）
    (void)HcclSim::DpuSim::EnsureDpuExecutorStarted(static_cast<uint32_t>(optComm->device_id));
    HCCL_VM_INFO("{} success, commId={:d}, msgTag={}", __func__, commId, msgTag);
    return HCCL_SUCCESS;
}

/**
 * @brief 注销 DPU 算法回调。
 * @param comm   输入：通信域句柄。
 * @param msgTag 输入：任务标识。
 * @return HcclResult 成功返回 HCCL_SUCCESS。
 */
HcclResult HcclTaskUnRegister(HcclComm comm, const char* msgTag)
{
    if (comm == nullptr || msgTag == nullptr) {
        HCCL_VM_ERROR("{}: comm or msgTag is nullptr", __func__);
        return HCCL_E_PTR;
    }
    const uint64_t commId = reinterpret_cast<uint64_t>(comm);
    (void)HcclSim::DpuSim::DpuCallbackRegistry::Instance().UnRegister(commId, msgTag);
    return HCCL_SUCCESS;
}

/**
 * @brief 查询本设备 DPU 数据面任务归属流 id。
 *
 * 读取设备侧 HcommSendRequest 桩写入的 DpuDeviceInfo 记录（与南向同源）。
 *
 * @param comm        输入：通信域句柄。
 * @param dpuStreamId 输出：DPU 流 id（引用参数，ABI 同真实实现 u32&）。
 * @return HcclResult 成功返回 HCCL_SUCCESS；未记录返回 HCCL_E_NOT_FOUND。
 */
HcclResult HcclGetDpuSteamIdV2(HcclComm comm, uint32_t& dpuStreamId)
{
    if (comm == nullptr) {
        HCCL_VM_ERROR("{}: comm is nullptr", __func__);
        return HCCL_E_PTR;
    }
    const uint64_t commId = reinterpret_cast<uint64_t>(comm);
    const auto optComm = sim::runtime::Db::GetById<sim::runtime::Communicator>(commId);
    if (!optComm.ok()) {
        HCCL_VM_ERROR("{}: communicator {:d} not found", __func__, commId);
        return HCCL_E_INTERNAL;
    }
    const auto dpuInfo = sim::runtime::Db::GetOneByPred<sim::runtime::DpuDeviceInfo>(
        HcclSim::Storage::Eq(&sim::runtime::DpuDeviceInfo::device_id, static_cast<uint32_t>(optComm->device_id)));
    if (!dpuInfo.ok()) {
        HCCL_VM_WARN("{}: DpuDeviceInfo not found, deviceId={:d}", __func__, optComm->device_id);
        return HCCL_E_NOT_FOUND;
    }
    dpuStreamId = static_cast<uint32_t>(dpuInfo->stream_id);
    return HCCL_SUCCESS;
}

/**
 * @brief 等待线程任务队列排空（HostDPU 场景，no-op）。
 *
 * 真实实现轮询 RTSQ head==tail；VM 无真实队列，直接成功。
 *
 * @param thread 输入：线程句柄（未使用）。
 * @return int32_t 成功返回 0。
 */
int32_t HcommThreadSynchronize(ThreadHandle thread)
{
    HCCL_VM_INFO("{}: thread={:d}, no-op", __func__, thread);
    return 0;
}

/**
 * @brief 等待线程退出（HostDPU 场景，no-op）。
 * @param thread  输入：线程句柄（未使用）。
 * @param timeout 输入：超时时间（未使用）。
 * @return int32_t 成功返回 0。
 */
int32_t HcommThreadJoin(ThreadHandle thread, uint32_t timeout)
{
    (void)timeout;
    HCCL_VM_INFO("{}: thread={:d}, no-op", __func__, thread);
    return 0;
}

/**
 * @brief 设置 notify 等待默认超时时间（仅保存语义，返回成功）。
 *
 * 对应 hcomm_primitives.h 同名接口；模拟器不感知超时，
 * 按真实实现校验并截断 float，不保存状态。
 *
 * @param timeOut 输入：超时时间（秒）。
 * @return int32_t 成功返回 0，非法超时返回 HCCL_E_PARA。
 */
int32_t HcommSetNotifyWaitTimeOut(float timeOut)
{
    uint32_t normalized = 0;
    const int32_t ret = NormalizeDpuTimeout(timeOut, normalized);
    if (ret != 0) {
        HCCL_VM_ERROR("{} failed: timeOut={}", __func__, timeOut);
        return ret;
    }
    HCCL_VM_INFO("{} success, timeOutSeconds={:d}", __func__, normalized);
    return 0;
}

/**
 * @brief 设置线程资源申请默认超时时间（仅保存语义，返回成功）。
 * @param timeOut 输入：超时时间（秒）。
 * @return int32_t 成功返回 0，非法超时返回 HCCL_E_PARA。
 */
int32_t HcommThreadResAcquireTimeOut(float timeOut)
{
    uint32_t normalized = 0;
    const int32_t ret = NormalizeDpuTimeout(timeOut, normalized);
    if (ret != 0) {
        HCCL_VM_ERROR("{} failed: timeOut={}", __func__, timeOut);
        return ret;
    }
    HCCL_VM_INFO("{} success, timeOutSeconds={:d}", __func__, normalized);
    return 0;
}

/**
 * @brief 销毁通信域通道（级联释放通道挂载的 Notify 记录）。
 * @param comm       输入：通信域句柄。
 * @param channels   输入：待销毁的通道句柄数组（HcclChannel 表主键）。
 * @param channelNum 输入：通道数量。
 * @return HcclResult 成功返回 HCCL_SUCCESS，参数错误返回对应错误码。
 */
HcclResult HcclChannelDestroy(HcclComm comm, const ChannelHandle* channels, uint32_t channelNum)
{
    if (comm == nullptr || channels == nullptr) {
        HCCL_VM_ERROR("{}: comm or channels is nullptr", __func__);
        return HCCL_E_PTR;
    }
    const uint64_t commId = reinterpret_cast<uint64_t>(comm);
    if (!sim::runtime::Db::GetById<sim::runtime::Communicator>(commId).ok()) {
        HCCL_VM_ERROR("{}: communicator {:d} not found", __func__, commId);
        return HCCL_E_INTERNAL;
    }
    for (uint32_t i = 0; i < channelNum; ++i) {
        const auto channel = sim::runtime::Db::GetById<sim::runtime::HcclChannel>(channels[i]);
        if (!channel.ok()) {
            HCCL_VM_WARN("{}: channel {:d} not found, skip", __func__, channels[i]);
            continue;
        }
        // 级联释放通道挂载的 Notify 记录（与 ReleaseChannelNotifies 同逻辑）
        for (uint32_t n = 0; n < channel->notifyNum && n < MAX_NOTIFY_PER_CHANNEL; ++n) {
            if (channel->notifyId[n] != 0u) {
                (void)sim::runtime::Db::Delete<sim::runtime::Notify>(
                    HcclSim::Storage::Eq(&sim::runtime::Notify::id, channel->notifyId[n]));
            }
        }
        (void)sim::runtime::Db::Delete<sim::runtime::HcclChannel>(
            HcclSim::Storage::Eq(&sim::runtime::HcclChannel::id, channels[i]));
    }
    HCCL_VM_INFO("{} success, commId={:d}, channelNum={:d}", __func__, commId, channelNum);
    return HCCL_SUCCESS;
}

#ifdef __cplusplus
}
#endif
