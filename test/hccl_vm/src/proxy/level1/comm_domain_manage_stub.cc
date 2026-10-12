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
 * Description: 通信域管理打桩函数（北向劫持）
 */

#define HCCL_VM_MODULE "CDM_STUB"

#include <cstdint>
#include <cstring>
#include <cstdlib>

#include "acl/acl_rt.h"
#include "hccl_proxy_common.h"
#include "level1_proxy_common.h"
#include "sim_log.h"
#include "runtime_state/db_sim_runner_common.h"
#include "runtime_state/db_sim_runner_ops.h"

/* hcclAlgo 配置串的最大长度，对应 CANN 头文件 hccl_types.h 中的 HCCL_COMM_ALGO_MAX_LENGTH。 */
static constexpr uint32_t kHcclAlgoMaxLength = 1600;
/* UB 多 channel 数量默认值（1 表示不使能），对应 hccl::GetEnvUbConfig().GetUbMultiChannelNum()。 */
static constexpr uint32_t kUbMultiChannelNumDefault = 1;

#ifdef __cplusplus
extern "C" {
#endif

typedef HcclResult (*HcclCommStateCallback)(HcclComm comm, HcclCommStatePhase state, void* args);

/**
 * @brief 初始化HcclCommConfig结构体。
 *
 * @note HcclCommConfigInit为static inline函数，北向劫持无法拦截其调用，
 *       因此此桩函数仅作为占位存在，调用时应返回HCCL_E_NOT_SUPPORT。
 *       用户应直接使用HcclCommInitClusterInfo或HcclCommInitRootInfo等接口。
 *
 * @param config 输入/输出：待初始化的HcclCommConfig结构体指针。
 * @return HcclResult 始终返回HCCL_E_NOT_SUPPORT。
 */
HcclResult HcclCommConfigInit(HcclCommConfig* config)
{
    (void)config;
    HCCL_VM_ERROR(
        "{} is a static inline function, cannot be hijacked by proxy. "
        "Please use HcclCommInitClusterInfo/HcclCommInitRootInfo instead.",
        __func__);
    return HCCL_E_NOT_SUPPORT;
}

/**
 * @brief 获取指定通信域的rank总数。
 *
 * 根据通信域句柄查询通信域表，返回该通信域中rank的总数量。
 *
 * @param comm 输入：通信域句柄，由HcclCommInitXxx系列接口返回。
 * @param rankSize 输出：rank的总数。
 *
 * @return HcclResult 接口成功返回HCCL_SUCCESS，其他失败。
 * @retval HCCL_SUCCESS 查询成功
 * @retval HCCL_E_PTR rankSize指针为空
 * @retval HCCL_E_INTERNAL 内部错误（通信域不存在）
 */
HcclResult HcclGetRankSize(HcclComm comm, uint32_t* rankSize)
{
    if (comm == nullptr) {
        HCCL_VM_ERROR("{}: comm is nullptr", __func__);
        return HCCL_E_PTR;
    }

    uint64_t commId = reinterpret_cast<uint64_t>(comm);
    auto optComm = sim::runtime::Db::GetById<sim::runtime::Communicator>(commId);
    if (!optComm.ok()) {
        HCCL_VM_ERROR("{}: communicator {:d} not found", __func__, commId);
        return HCCL_E_INTERNAL;
    }

    *rankSize = optComm->rank_size;
    HCCL_VM_INFO("{} success, commId={:d}, rankSize={:d}", __func__, commId, *rankSize);
    return HCCL_SUCCESS;
}

/**
 * @brief 获取指定通信域中当前rank的ID。
 *
 * 根据通信域句柄查询通信域表，返回当前rank在该通信域中的编号。
 * 该编号与rank table中对应的"rank_id"字段取值一致。
 *
 * @param comm 输入：通信域句柄，由HcclCommInitXxx系列接口返回。
 * @param rank 输出：当前rank的ID。
 *
 * @return HcclResult 接口成功返回HCCL_SUCCESS，其他失败。
 * @retval HCCL_SUCCESS 查询成功
 * @retval HCCL_E_PTR rank指针为空
 * @retval HCCL_E_INTERNAL 内部错误（通信域不存在）
 */
HcclResult HcclGetRankId(HcclComm comm, uint32_t* rank)
{
    if (comm == nullptr) {
        HCCL_VM_ERROR("{}: comm is nullptr", __func__);
        return HCCL_E_PTR;
    }

    uint64_t commId = reinterpret_cast<uint64_t>(comm);
    auto optComm = sim::runtime::Db::GetById<sim::runtime::Communicator>(commId);
    if (!optComm.ok()) {
        HCCL_VM_ERROR("{}: communicator {:d} not found", __func__, commId);
        return HCCL_E_INTERNAL;
    }

    *rank = static_cast<uint32_t>(optComm->rank_id);
    HCCL_VM_INFO("{} success, commId={:d}, rank={:d}", __func__, commId, *rank);
    return HCCL_SUCCESS;
}

/**
 * @brief 通信域同步屏障，所有rank需调用此接口以达到同步点。
 *
 * @note 北向劫持场景下无需实际同步，直接返回成功。
 *
 * @param comm 输入：通信域句柄。
 * @param stream 输入：指定在此流上插入Barrier。
 *
 * @return HcclResult 始终返回HCCL_SUCCESS。
 */
HcclResult HcclBarrier(HcclComm comm, aclrtStream stream)
{
    if (comm == nullptr) {
        HCCL_VM_ERROR("{}: comm is nullptr", __func__);
        return HCCL_E_PTR;
    }
    (void)stream;
    return HCCL_SUCCESS;
}

/**
 * @brief 全局配置HCCL通信域属性。
 *
 * @note 当前平台（A5）不支持此接口，始终返回HCCL_E_NOT_SUPPORT。
 *
 * @param config 输入：配置项，详见HcclConfig。
 * @param configValue 输入：配置值，详见HcclConfigValue。
 *
 * @return HcclResult 始终返回HCCL_E_NOT_SUPPORT。
 */
HcclResult HcclSetConfig(HcclConfig config, HcclConfigValue configValue)
{
    (void)config;
    (void)configValue;
    HCCL_VM_ERROR("{} is not supported on this platform (A5)", __func__);
    return HCCL_E_NOT_SUPPORT;
}

/**
 * @brief 获取HCCL全局通信域属性配置。
 *
 * @note 当前平台（A5）不支持此接口，始终返回HCCL_E_NOT_SUPPORT。
 *
 * @param config 输入：配置项，详见HcclConfig。
 * @param configValue 输出：配置值，详见HcclConfigValue。
 *
 * @return HcclResult 始终返回HCCL_E_NOT_SUPPORT。
 */
HcclResult HcclGetConfig(HcclConfig config, HcclConfigValue* configValue)
{
    (void)config;
    (void)configValue;
    HCCL_VM_ERROR("{} is not supported on this platform (A5)", __func__);
    return HCCL_E_NOT_SUPPORT;
}

/**
 * @brief 获取通信域名称。
 *
 * 根据通信域句柄查询通信域表，返回该通信域的名称。若创建时未指定名称，
 * 则返回空字符串。
 *
 * @param comm 输入：通信域句柄，由HcclCommInitXxx系列接口返回。
 * @param commName 输出：通信域名称，缓冲区大小需不小于COMM_NAME_MAX_LENGTH（128字节）。
 *
 * @return HcclResult 接口成功返回HCCL_SUCCESS，其他失败。
 * @retval HCCL_SUCCESS 查询成功
 * @retval HCCL_E_PTR commName或comm为空
 * @retval HCCL_E_INTERNAL 内部错误（通信域不存在）
 */
HcclResult HcclGetCommName(HcclComm comm, char* commName)
{
    if (comm == nullptr) {
        HCCL_VM_ERROR("{}: comm is nullptr", __func__);
        return HCCL_E_PTR;
    }

    uint64_t commId = reinterpret_cast<uint64_t>(comm);
    auto optComm = sim::runtime::Db::GetById<sim::runtime::Communicator>(commId);
    if (!optComm.ok()) {
        HCCL_VM_ERROR("{}: communicator {:d} not found", __func__, commId);
        return HCCL_E_INTERNAL;
    }

    std::strncpy(commName, optComm->comm_id, COMM_NAME_MAX_LENGTH - 1);
    commName[COMM_NAME_MAX_LENGTH - 1] = '\0';

    HCCL_VM_INFO("{} success, commId={:d}, commName={}", __func__, commId, commName);
    return HCCL_SUCCESS;
}

/**
 * @brief 根据通信域名称获取通信域句柄。
 *
 * @param commName 输入：通信域名称，长度不超过COMM_NAME_MAX_LENGTH（128字节）。
 * @param comm 输出：返回的通信域句柄。
 * @return HcclResult 接口成功返回HCCL_SUCCESS，其他失败。
 * @retval HCCL_SUCCESS 成功获取句柄
 * @retval HCCL_E_PTR commName或comm为空
 * @retval HCCL_E_NOT_FOUND 未找到指定名称的通信域
 */
HcclResult HcclCommGetHandleWithName(const char* commName, HcclComm* comm)
{
    if (commName == nullptr) {
        HCCL_VM_ERROR("{}: commName is nullptr", __func__);
        return HCCL_E_PTR;
    }

    auto comms = sim::runtime::Db::GetByPred<sim::runtime::Communicator>(
        HcclSim::Storage::Eq(&sim::runtime::Communicator::comm_id, std::string(commName)));
    if (!comms.ok() || comms->empty()) {
        HCCL_VM_ERROR("{}: communicator with name '{}' not found", __func__, commName);
        return HCCL_E_NOT_FOUND;
    }

    *comm = reinterpret_cast<HcclComm>((*comms)[0].id);
    HCCL_VM_INFO("{} success, commId={:d}, name={}", __func__, (*comms)[0].id, commName);
    return HCCL_SUCCESS;
}

/**
 * @brief 获取HcclCommConfig支持的Capability位掩码。
 * @return 返回12，表示支持所有配置能力。
 */
uint32_t HcclGetCommConfigCapability()
{
    HCCL_VM_INFO("{} returns 12", __func__);
    return 12;
}

/**
 * @brief 暂停指定通信域的执行。
 * @note 当前不支持此功能。
 */
HcclResult HcclCommSuspend(HcclComm comm)
{
    if (comm == nullptr) {
        HCCL_VM_ERROR("{}: comm is nullptr", __func__);
        return HCCL_E_PTR;
    }
    (void)comm;
    HCCL_VM_ERROR("{} is not supported", __func__);
    return HCCL_E_NOT_SUPPORT;
}

/**
 * @brief 恢复指定通信域的执行。
 */
HcclResult HcclCommResume(HcclComm comm)
{
    (void)comm;
    HCCL_VM_INFO("{} stub, commId={:d}", __func__, comm ? reinterpret_cast<uint64_t>(comm) : 0);
    return HCCL_SUCCESS;
}

/**
 * @brief 获取通信域状态。
 * @param commId 输入：通信域标识符（当前未使用）。
 * @param status 输出：通信域状态。
 * @return HcclResult 接口成功返回HCCL_SUCCESS。
 */
HcclResult HcclCommGetStatus(const char* commId, HcclCommStatus* status)
{
    (void)commId;
    *status = HCCL_COMM_STATUS_READY;
    HCCL_VM_INFO("{} returns HCCL_COMM_STATUS_READY", __func__);
    return HCCL_SUCCESS;
}

/**
 * @brief 设置通信域在设备网卡故障时使用备用网卡。
 * @note 当前平台（A5）不支持此接口，始终返回HCCL_E_NOT_SUPPORT。
 */
HcclResult HcclCommWorkingDevNicSet(HcclComm comm, uint32_t* ranks, bool* useBackup, uint32_t nRanks)
{
    (void)comm;
    (void)ranks;
    (void)useBackup;
    (void)nRanks;
    HCCL_VM_ERROR("{} is not supported on this platform (A5)", __func__);
    return HCCL_E_NOT_SUPPORT;
}

/**
 * @brief 获取通信域的异步错误信息。
 * @note 当前平台（A5）不支持此接口，始终返回HCCL_E_NOT_SUPPORT。
 */
HcclResult HcclGetCommAsyncError(HcclComm comm, HcclResult* asyncError)
{
    (void)comm;
    (void)asyncError;
    HCCL_VM_ERROR("{} is not supported on this platform (A5)", __func__);
    return HCCL_E_NOT_SUPPORT;
}

/**
 * @brief 将HcclResult错误码转换为可读字符串。
 * @note 当前平台（A5）不支持此接口，始终返回nullptr。
 */
const char* HcclGetErrorString(HcclResult code)
{
    (void)code;
    HCCL_VM_ERROR("{} is not supported on this platform (A5)", __func__);
    return nullptr;
}

/**
 * @brief 设置通信域的内存范围。
 * @note 当前平台（A5）不支持此接口，始终返回HCCL_E_NOT_SUPPORT。
 */
HcclResult HcclCommSetMemoryRange(HcclComm comm, void* baseVirPtr, size_t size, size_t alignment, uint64_t flags)
{
    (void)comm;
    (void)baseVirPtr;
    (void)size;
    (void)alignment;
    (void)flags;
    HCCL_VM_ERROR("{} is not supported on this platform (A5)", __func__);
    return HCCL_E_NOT_SUPPORT;
}

/**
 * @brief 取消设置通信域的内存范围。
 * @note A5平台不支持。
 */
HcclResult HcclCommUnsetMemoryRange(HcclComm comm, void* baseVirPtr)
{
    (void)comm;
    (void)baseVirPtr;
    HCCL_VM_ERROR("{} is not supported on A5", __func__);
    return HCCL_E_NOT_SUPPORT;
}

/**
 * @brief 激活通信域内存。
 * @note A5平台不支持。
 */
HcclResult HcclCommActivateCommMemory(
    HcclComm comm, void* virPtr, size_t size, size_t offset, aclrtDrvMemHandle handle, uint64_t flags)
{
    (void)comm;
    (void)virPtr;
    (void)size;
    (void)offset;
    (void)handle;
    (void)flags;
    HCCL_VM_ERROR("{} is not supported on A5", __func__);
    return HCCL_E_NOT_SUPPORT;
}

/**
 * @brief 反激活通信域内存。
 * @note A5平台不支持。
 */
HcclResult HcclCommDeactivateCommMemory(HcclComm comm, void* virPtr)
{
    (void)comm;
    (void)virPtr;
    HCCL_VM_ERROR("{} is not supported on A5", __func__);
    return HCCL_E_NOT_SUPPORT;
}

/**
 * @brief 注册对称内存窗口。
 *
 * 将 addr 指向的 size 字节注册为当前通信域的对称内存窗口。一个通信域只允许注册
 * 一个对称内存窗口（一一对应），重复注册将返回错误。
 * winHandle 返回通信域句柄，用于后续的 SymWinGet / SymWinGetPeerPointer 接口。
 *
 * @param comm      输入：通信域句柄。
 * @param addr      输入：待注册的对称内存起始地址。
 * @param size      输入：对称内存窗口大小（字节）。
 * @param winHandle 输出：对称内存窗口资源句柄，注册成功时返回通信域句柄。
 * @param flag      输入：预留标志位，当前未使用。
 * @return HcclResult 成功返回 HCCL_SUCCESS，参数错误返回 HCCL_E_PTR。
 */
HcclResult HcclCommSymWinRegister(HcclComm comm, void* addr, uint64_t size, HcclCommSymWindow* winHandle, uint32_t flag)
{
    if (comm == nullptr || addr == nullptr || winHandle == nullptr) {
        HCCL_VM_ERROR("{}: comm, addr or winHandle is nullptr", __func__);
        return HCCL_E_PTR;
    }
    if (size == 0u) {
        HCCL_VM_ERROR("{}: size is 0", __func__);
        return HCCL_E_PARA;
    }
    (void)flag;

    uint64_t commId = reinterpret_cast<uint64_t>(comm);
    auto optComm = sim::runtime::Db::GetById<sim::runtime::Communicator>(commId);
    if (!optComm.ok()) {
        HCCL_VM_ERROR("{}: communicator {:d} not found", __func__, commId);
        return HCCL_E_INTERNAL;
    }

    if (optComm->sym_win_registered) {
        HCCL_VM_ERROR("{}: symmetric memory window already registered for comm {:d}", __func__, commId);
        return HCCL_E_INTERNAL;
    }

    sim::runtime::Db::Update<sim::runtime::Communicator>(
        HcclSim::Storage::Eq(&sim::runtime::Communicator::id, commId),
        HcclSim::Storage::Set(&sim::runtime::Communicator::sym_win_addr, reinterpret_cast<uint64_t>(addr)),
        HcclSim::Storage::Set(&sim::runtime::Communicator::sym_win_size, size),
        HcclSim::Storage::Set(&sim::runtime::Communicator::sym_win_registered, static_cast<uint8_t>(1)));

    *winHandle = reinterpret_cast<HcclCommSymWindow>(commId);
    HCCL_VM_INFO("{} success, commId={:d}, symWinAddr={:p}, symWinSize={:d}", __func__, commId, addr, size);
    return HCCL_SUCCESS;
}

/**
 * @brief 注销对称内存窗口。
 *
 * 清除指定对称内存窗口的注册信息。winHandle 由 HcclCommSymWinRegister 返回。
 *
 * @param winHandle 输入：待注销的对称内存窗口资源句柄。
 * @return HcclResult 成功返回 HCCL_SUCCESS，参数错误返回 HCCL_E_PTR。
 */
HcclResult HcclCommSymWinDeregister(HcclCommSymWindow winHandle)
{
    if (winHandle == nullptr) {
        HCCL_VM_ERROR("{}: winHandle is nullptr", __func__);
        return HCCL_E_PTR;
    }

    uint64_t commId = reinterpret_cast<uint64_t>(winHandle);
    auto optComm = sim::runtime::Db::GetById<sim::runtime::Communicator>(commId);
    if (!optComm.ok()) {
        HCCL_VM_ERROR("{}: communicator {:d} not found", __func__, commId);
        return HCCL_E_INTERNAL;
    }

    sim::runtime::Db::Update<sim::runtime::Communicator>(
        HcclSim::Storage::Eq(&sim::runtime::Communicator::id, commId),
        HcclSim::Storage::Set(&sim::runtime::Communicator::sym_win_addr, 0u),
        HcclSim::Storage::Set(&sim::runtime::Communicator::sym_win_size, 0u),
        HcclSim::Storage::Set(&sim::runtime::Communicator::sym_win_registered, static_cast<uint8_t>(0)));

    HCCL_VM_INFO("{} success, commId={:d}", __func__, commId);
    return HCCL_SUCCESS;
}

/**
 * @brief 获取通信域的对称内存窗口。
 *
 * 根据已注册对称内存的地址指针 ptr，返回对应的窗口资源句柄及其在窗口内的偏移量。
 * ptr 必须落在 HcclCommSymWinRegister 注册的 [addr, addr + size) 范围内。
 *
 * @param comm      输入：通信域句柄。
 * @param ptr       输入：对称内存窗口内的地址指针。
 * @param size      输入：预留，当前未使用。
 * @param winHandle 输出：对称内存窗口资源句柄。
 * @param offset    输出：ptr 在对称内存窗口中的偏移量（字节）。
 * @return HcclResult 成功返回 HCCL_SUCCESS，参数错误返回 HCCL_E_PTR。
 */
HcclResult HcclCommSymWinGet(HcclComm comm, void* ptr, size_t size, HcclCommSymWindow* winHandle, size_t* offset)
{
    if (comm == nullptr || ptr == nullptr || winHandle == nullptr || offset == nullptr) {
        HCCL_VM_ERROR("{}: comm, ptr, winHandle or offset is nullptr", __func__);
        return HCCL_E_PTR;
    }
    (void)size;

    uint64_t commId = reinterpret_cast<uint64_t>(comm);
    auto optComm = sim::runtime::Db::GetById<sim::runtime::Communicator>(commId);
    if (!optComm.ok()) {
        HCCL_VM_ERROR("{}: communicator {:d} not found", __func__, commId);
        return HCCL_E_INTERNAL;
    }

    if (!optComm->sym_win_registered) {
        HCCL_VM_INFO("{}: symmetric memory window not registered for comm {:d}", __func__, commId);
        return HCCL_E_INTERNAL;
    }

    if (optComm->sym_win_addr == 0u || optComm->sym_win_size == 0u) {
        HCCL_VM_ERROR("{}: symmetric memory window registered but addr/size invalid for comm {:d}", __func__, commId);
        return HCCL_E_INTERNAL;
    }

    uint64_t ptrAddr = reinterpret_cast<uint64_t>(ptr);
    uint64_t baseAddr = optComm->sym_win_addr;
    uint64_t winSize = optComm->sym_win_size;
    if (ptrAddr < baseAddr || ptrAddr >= baseAddr + winSize) {
        HCCL_VM_ERROR(
            "{}: ptr {:p} is out of symmetric memory window [addr={:p}, size={:d}) for comm {:d}", __func__, ptr,
            reinterpret_cast<void*>(baseAddr), winSize, commId);
        return HCCL_E_PARA;
    }

    *winHandle = reinterpret_cast<HcclCommSymWindow>(commId);
    *offset = static_cast<size_t>(ptrAddr - baseAddr);
    HCCL_VM_INFO("{} success, commId={:d}, ptr={:p}, offset={:d}", __func__, commId, ptr, *offset);
    return HCCL_SUCCESS;
}

/**
 * @brief 获取对端rank在对称内存窗口中的指针。
 *
 * 对称内存模型中，所有 rank 的虚拟地址布局相同，对端 rank 在相同偏移处
 * 的地址与本端一致。直接返回 SymWinAddr + offset 作为对端指针。
 *
 * @param winHandle 输入：对称内存窗口资源句柄。
 * @param offset    输入：对称内存窗口内的偏移量（字节）。
 * @param peerRank  输入：对端 rank 编号。
 * @param ptr       输出：对端 rank 在对称内存窗口中该偏移对应的地址指针。
 * @return HcclResult 成功返回 HCCL_SUCCESS，参数错误返回 HCCL_E_PTR。
 */
HcclResult HcclSymWinGetPeerPointer(HcclCommSymWindow winHandle, size_t offset, uint32_t peerRank, void** ptr)
{
    if (winHandle == nullptr || ptr == nullptr) {
        HCCL_VM_ERROR("{}: winHandle or ptr is nullptr", __func__);
        return HCCL_E_PTR;
    }
    (void)peerRank;

    uint64_t commId = reinterpret_cast<uint64_t>(winHandle);
    auto optComm = sim::runtime::Db::GetById<sim::runtime::Communicator>(commId);
    if (!optComm.ok()) {
        HCCL_VM_ERROR("{}: communicator {:d} not found", __func__, commId);
        return HCCL_E_INTERNAL;
    }

    if (!optComm->sym_win_registered) {
        HCCL_VM_INFO("{}: symmetric memory window not registered for comm {:d}", __func__, commId);
        return HCCL_E_INTERNAL;
    }

    if (optComm->sym_win_addr == 0u || optComm->sym_win_size == 0u) {
        HCCL_VM_ERROR("{}: symmetric memory window registered but addr/size invalid for comm {:d}", __func__, commId);
        return HCCL_E_INTERNAL;
    }

    uint64_t baseAddr = optComm->sym_win_addr;
    uint64_t winSize = optComm->sym_win_size;
    if (offset >= winSize) {
        HCCL_VM_ERROR(
            "{}: offset {:d} out of range (symWinSize={:d}) for comm {:d}", __func__, offset, winSize, commId);
        return HCCL_E_PARA;
    }

    // 对称内存模型中，所有 rank 的虚拟地址布局相同
    *ptr = reinterpret_cast<void*>(baseAddr + offset);
    HCCL_VM_INFO(
        "{} success, commId={:d}, peerRank={:d}, offset={:d}, ptr={:p}", __func__, commId, peerRank, offset, *ptr);
    return HCCL_SUCCESS;
}

/**
 * @brief 开始通信域分组操作。
 * @note A5平台不支持分组通信。
 */
HcclResult HcclGroupStart()
{
    HCCL_VM_ERROR("{} is not supported on A5", __func__);
    return HCCL_E_NOT_SUPPORT;
}

/**
 * @brief 结束通信域分组操作。
 * @note A5平台不支持分组通信。
 */
HcclResult HcclGroupEnd()
{
    HCCL_VM_ERROR("{} is not supported on A5", __func__);
    return HCCL_E_NOT_SUPPORT;
}

/**
 * @brief 把 HCCL_OP_EXPANSION_MODE 环境变量字符串解析为算子展开模式枚举。
 *
 * @param envValue 输入：环境变量取值（可为 nullptr 或空串）。
 * @return 对应的展开模式枚举；nullptr / 空串 / 无法识别时返回 HCCL_OP_EXPANSION_MODE_INVALID。
 */
static HcclOpExpansionMode OpExpansionModeFromString(const char* envValue)
{
    if (envValue == nullptr || envValue[0] == '\0') {
        return HCCL_OP_EXPANSION_MODE_INVALID;
    }
    if (std::strcmp(envValue, "AI_CPU") == 0 || std::strcmp(envValue, "AICPU_TS") == 0
        || std::strcmp(envValue, "AICPU_CacheDisable") == 0) {
        return HCCL_OP_EXPANSION_MODE_AI_CPU;
    }
    if (std::strcmp(envValue, "AIV") == 0) {
        return HCCL_OP_EXPANSION_MODE_AIV;
    }
    if (std::strcmp(envValue, "HOST") == 0) {
        return HCCL_OP_EXPANSION_MODE_HOST;
    }
    if (std::strcmp(envValue, "HOST_TS") == 0) {
        return HCCL_OP_EXPANSION_MODE_HOST_TS;
    }
    if (std::strcmp(envValue, "CCU_MS") == 0) {
        return HCCL_OP_EXPANSION_MODE_CCU_MS;
    }
    if (std::strcmp(envValue, "CCU_SCHED") == 0) {
        return HCCL_OP_EXPANSION_MODE_CCU_SCHED;
    }
    return HCCL_OP_EXPANSION_MODE_INVALID;
}

/**
 * @brief 解析确定性等级（与 hccl 侧 GetExternalInputHcclDeterministic 的 env 口径一致）。
 *
 * 取值与 CANN 的 DeterministicEnableLevel 一致：0=非确定、1=确定、2=严格保序。
 * 环境变量 HCCL_DETERMINISTIC 支持 true / false / strict（另兼容 1 / 0）；
 * 未设置或取值非法时返回 0（非确定，即 hccl 的默认值）。
 *
 * @return 确定性等级（0 / 1 / 2）。
 */
static uint32_t ResolveDeterministicLevelFromEnv()
{
    const char* envValue = std::getenv("HCCL_DETERMINISTIC");
    if (envValue == nullptr) {
        return 0;
    }
    if (std::strcmp(envValue, "strict") == 0) {
        return 2;
    }
    if (std::strcmp(envValue, "true") == 0 || std::strcmp(envValue, "1") == 0) {
        return 1;
    }
    return 0;
}

/**
 * @brief 读取 HCCL_CONFIG_TYPE_DETERMINISTIC：输出确定性等级（uint32_t）。
 *
 * 该查询项为本轮新 CANN/hccl 新增（hccl 侧 order_preserved_common.h 的 GetDeterministicLevel
 * 在 hcomm 版本高于 9.2.0.1 时会查）。桩若不支持会走到 default 分支打 ERROR 并返回 HCCL_E_PARA，
 * 上层只能回退 env 值并打 WARNING；此处按 env 口径直接给出等级，行为与回退值一致且不再刷错误日志。
 *
 * @param infoLen 输入：info 缓冲区长度（字节），需不小于 sizeof(uint32_t)。
 * @param info 输出：存放 uint32_t 确定性等级的缓冲区。
 * @return HcclResult 成功返回 HCCL_SUCCESS；infoLen 不足返回 HCCL_E_PARA。
 */
static HcclResult GetDeterministicConfig(uint32_t infoLen, void* info)
{
    if (infoLen < sizeof(uint32_t)) {
        HCCL_VM_ERROR(
            "{}: infoLen too small: {:d}, need at least {:d}", __func__, infoLen,
            static_cast<uint32_t>(sizeof(uint32_t)));
        return HCCL_E_PARA;
    }

    const uint32_t deterministicLevel = ResolveDeterministicLevelFromEnv();
    *(static_cast<uint32_t*>(info)) = deterministicLevel;
    HCCL_VM_INFO("{} success, deterministicLevel={:d}", __func__, deterministicLevel);
    return HCCL_SUCCESS;
}

/**
 * @brief 读取 HCCL_CONFIG_TYPE_UDI：输出通信域自定义标识（C 字符串，'\0' 结尾）。
 *
 * 本模拟器无 UDI 配置，按"空串"输出（至少写入 1 个 '\0'）；与真实实现
 * "infoLen 必须容纳 udi.size()+1" 的约束同构——空串时要求 infoLen >= 2。
 *
 * @param infoLen 输入：info 缓冲区长度（字节），需不小于 2。
 * @param info 输出：存放以 '\0' 结尾的字符串缓冲区。
 * @return HcclResult 成功返回 HCCL_SUCCESS；infoLen 不足返回 HCCL_E_PARA。
 */
static HcclResult GetUdiConfig(uint32_t infoLen, void* info)
{
    if (infoLen <= 1U) {
        HCCL_VM_ERROR("{}: infoLen too small: {:d}, need at least 2", __func__, infoLen);
        return HCCL_E_PARA;
    }

    static_cast<char*>(info)[0] = '\0';
    HCCL_VM_INFO("{} success, udi is empty", __func__);
    return HCCL_SUCCESS;
}

/**
 * @brief 读取 HCCL_CONFIG_TYPE_OP_EXPANSION_MODE：输出算子展开模式（int32_t）。
 *
 * @param infoLen 输入：info 缓冲区长度（字节），需容纳 HcclConfigTypeOpExpansionMode。
 * @param info 输出：存放算子展开模式值的缓冲区。
 * @return HcclResult 成功返回 HCCL_SUCCESS；长度不足或环境变量未设置/非法返回 HCCL_E_PARA。
 */
static HcclResult GetOpExpansionModeConfig(uint32_t infoLen, void* info)
{
    if (infoLen < sizeof(HcclConfigTypeOpExpansionMode)) {
        HCCL_VM_ERROR(
            "{}: infoLen too small: {:d}, need at least {:d}", __func__, infoLen,
            static_cast<uint32_t>(sizeof(HcclConfigTypeOpExpansionMode)));
        return HCCL_E_PARA;
    }

    const char* envValue = std::getenv("HCCL_OP_EXPANSION_MODE");
    const HcclOpExpansionMode mode = OpExpansionModeFromString(envValue);
    if (mode == HCCL_OP_EXPANSION_MODE_INVALID) {
        HCCL_VM_ERROR(
            "{}: HCCL_OP_EXPANSION_MODE not set or invalid value: {}", __func__, envValue ? envValue : "(null)");
        return HCCL_E_PARA;
    }

    *(static_cast<int32_t*>(info)) = static_cast<int32_t>(mode);
    HCCL_VM_INFO("{} success, opExpansionMode={:d}", __func__, static_cast<int>(mode));
    return HCCL_SUCCESS;
}

/**
 * @brief 读取 HCCL_CONFIG_TYPE_HCCL_ALGO：输出算法配置串（A5 走 costmodel，本域未配置，输出空串）。
 *
 * @param infoLen 输入：info 缓冲区长度（字节），需不小于 kHcclAlgoMaxLength。
 * @param info 输出：存放以 '\0' 起始填充为零的算法串缓冲区。
 * @return HcclResult 成功返回 HCCL_SUCCESS；长度不足返回 HCCL_E_PARA。
 */
static HcclResult GetHcclAlgoConfig(uint32_t infoLen, void* info)
{
    if (infoLen < kHcclAlgoMaxLength) {
        HCCL_VM_ERROR("{}: infoLen too small: {:d}, need at least {:d}", __func__, infoLen, kHcclAlgoMaxLength);
        return HCCL_E_PARA;
    }

    std::memset(info, 0, infoLen);
    HCCL_VM_INFO("{} success, hcclAlgo is empty", __func__);
    return HCCL_SUCCESS;
}

/**
 * @brief 读取 HCCL_CONFIG_TYPE_UB_MULTI_CHANNEL_NUM：输出 UB 多 channel 数量（uint32_t）。
 *
 * @param infoLen 输入：info 缓冲区长度（字节），必须等于 sizeof(uint32_t)。
 * @param info 输出：存放 UB 多 channel 数量的缓冲区。
 * @return HcclResult 成功返回 HCCL_SUCCESS；长度不符返回 HCCL_E_PARA。
 */
static HcclResult GetUbMultiChannelNumConfig(uint32_t infoLen, void* info)
{
    if (infoLen != sizeof(uint32_t)) {
        HCCL_VM_ERROR(
            "{}: infoLen not expected: {:d}, need {:d}", __func__, infoLen, static_cast<uint32_t>(sizeof(uint32_t)));
        return HCCL_E_PARA;
    }

    *(static_cast<uint32_t*>(info)) = kUbMultiChannelNumDefault;
    HCCL_VM_INFO("{} success, ubMultiChannelNum={:d}", __func__, kUbMultiChannelNumDefault);
    return HCCL_SUCCESS;
}

/**
 * @brief 获取通信域配置信息。
 *
 * 支持的 cfgType：
 *  - HCCL_CONFIG_TYPE_OP_EXPANSION_MODE：读取环境变量HCCL_OP_EXPANSION_MODE，
 *    返回当前通信域使用的算子展开模式（CCU / AICPU / AIV 等），
 *    环境变量未设置或值不合法时返回 HCCL_E_PARA；
 *  - HCCL_CONFIG_TYPE_HCCL_ALGO：A5 设备走 costmodel 流程、不解析 HCCL_ALGO 环境变量，
 *    本域未配置算法串，故在 infoLen 不小于 HCCL_COMM_ALGO_MAX_LENGTH 时输出空串并返回成功；
 *  - HCCL_CONFIG_TYPE_UB_MULTI_CHANNEL_NUM：输出 UB 多 channel 数量，取默认值 1（不使能）；
 *  - HCCL_CONFIG_TYPE_DETERMINISTIC：输出确定性等级（0=非确定 / 1=确定 / 2=严格保序），
 *    取自环境变量 HCCL_DETERMINISTIC（未设置或非法则为 0）；
 *  - HCCL_CONFIG_TYPE_UDI：输出通信域自定义标识，本模拟器无该配置，返回空串（'\0'）。
 *
 * @param comm 输入：通信域句柄（当前实现未使用）。
 * @param cfgType 输入：配置类型，详见HcclConfigType。
 * @param infoLen 输入：info缓冲区的长度（字节）。
 * @param info 输出：指向存放返回值的缓冲区，具体类型由cfgType决定。
 *
 * @return HcclResult 接口成功返回HCCL_SUCCESS，其他失败。
 * @retval HCCL_SUCCESS 查询成功
 * @retval HCCL_E_PTR info指针为空
 * @retval HCCL_E_PARA cfgType不支持或infoLen不足
 */
HcclResult HcclConfigGetInfo(HcclComm comm, HcclConfigType cfgType, uint32_t infoLen, void* info)
{
    (void)comm;

    if (info == nullptr) {
        HCCL_VM_ERROR("{}: info is nullptr", __func__);
        return HCCL_E_PTR;
    }

    switch (cfgType) {
        case HCCL_CONFIG_TYPE_OP_EXPANSION_MODE:
            return GetOpExpansionModeConfig(infoLen, info);
        case HCCL_CONFIG_TYPE_HCCL_ALGO:
            return GetHcclAlgoConfig(infoLen, info);
        case HCCL_CONFIG_TYPE_UB_MULTI_CHANNEL_NUM:
            return GetUbMultiChannelNumConfig(infoLen, info);
        case HCCL_CONFIG_TYPE_DETERMINISTIC:
            return GetDeterministicConfig(infoLen, info);
        case HCCL_CONFIG_TYPE_UDI:
            return GetUdiConfig(infoLen, info);
        default:
            HCCL_VM_ERROR("{}: unsupported cfgType {:d}", __func__, static_cast<int>(cfgType));
            return HCCL_E_PARA;
    }
}

/**
 * @brief 向通信域登记交换信息（打桩：不落盘，仅校验入参）。
 *
 * @param comm 输入：通信域句柄。
 * @param data 输入：待登记的交换信息缓冲区。
 * @param length 输入：数据长度（字节）；当前实现未使用。
 * @return HcclResult 成功返回 HCCL_SUCCESS；comm 或 data 为空返回 HCCL_E_PTR。
 */
HcclResult HcclCommAddExchangeInfo(HcclComm comm, const void* data, uint32_t length)
{
    if (comm == nullptr || data == nullptr) {
        HCCL_VM_ERROR("{}: comm or data is nullptr", __func__);
        return HCCL_E_PTR;
    }
    return HCCL_SUCCESS;
}

/**
 * @brief 查询对端 rank 的交换信息（打桩：不产生数据，仅校验入参）。
 *
 * @param comm 输入：通信域句柄。
 * @param remoteRank 输入：对端 rank；当前实现未使用。
 * @param length 输入：调用方缓冲区长度（字节）；当前实现未使用。
 * @param data 输出：接收交换信息的缓冲区。
 * @param actualLength 输出：实际写入长度；当前实现未使用。
 * @return HcclResult 成功返回 HCCL_SUCCESS；comm 或 data 为空返回 HCCL_E_PTR。
 */
HcclResult
HcclCommGetExchangeInfo(HcclComm comm, uint32_t remoteRank, uint32_t length, void* data, uint32_t* actualLength)
{
    if (comm == nullptr || data == nullptr) {
        HCCL_VM_ERROR("{}: comm or data is nullptr", __func__);
        return HCCL_E_PTR;
    }
    return HCCL_SUCCESS;
}

/**
 * @brief 注册通信域状态回调（打桩：不注册，直接返回成功）。
 *
 * @param regName 输入：回调名（仅用于日志）。
 * @param cb 输入：回调函数指针；当前实现未使用。
 * @param args 输入：回调参数；当前实现未使用。
 * @return HcclResult 恒返回 HCCL_SUCCESS。
 */
HcclResult HcclCommRegCommStateCallback(const char* regName, HcclCommStateCallback cb, void* args)
{
    HCCL_VM_INFO("regName='{}', skipped.", regName);
    (void)cb;
    (void)args;
    return HCCL_SUCCESS;
}

#ifdef __cplusplus
}
#endif
