/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 日志染色: 模块 tag (须在 include sim_log.h 之前)
#define HCCL_VM_MODULE "MEM_STUB"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <mutex>
#include <unordered_map>
#include <unistd.h>
#include <vector>

#include "acl/acl_base.h"
#include "acl/acl_rt.h"
#include "hccl_proxy_common.h"
#include "store_sim_store_pub.h"
#include "sim_log.h"
#include "stat/res_stat.h"
#include "rt_external_mem.h"
#include "store_sim_comm_pool_policy.h"
#include "store_sim_device_memory_manager.h"
#include "store_sim_memory_manager.h"
#include "store_sim_run_mode.h"
#include "runtime_state/sim_models.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "runtime_state/db_sim_runner_common.h"
#include "sim_sub_process_manager.h"
#include "storage/table_access.h"

std::mutex g_hostAllocMutx;
std::map<uint64_t, uint32_t> g_hostAllocRecord;

/// 插入前物理清除本设备已标记释放的行（is_freed=1，2026-09-17 拆卸竞态修复）。
/// 不变量（成败点）：释放路径只置 is_freed=1 不删行——没有本 purge，标记行会
/// 与同设备同址新行并存，GetOneByPred（Free/Release/MapMem 的等值查找）多命中
/// 破坏管理路径；每次插入前先清同设备全部标记行 ⇒ 标记行与活跃行永不同址
/// 共存 ⇒ 任何等值/包含查询至多命中一行，无歧义。拆卸期无插入 ⇒ 标记行存续
/// 过迟到读者窗口（正是修复目标：FR1/R3K4 取证 43/48 次远端解析未命中全部
/// 晚于目标行成功删除 16.7s+，行保留即消除该窗口）；purge 窗口
/// （free→下次插入）严格优于现状的立即删除（窗口从 0 变宽，无回归）。
/// 附带收益：运行期稳态表大小 = 活跃行数（写时压实）；拆卸尾巴由下一轮
/// FLUSHALL 兜底。尽力而为：purge 失败不阻断插入（新行已保证不同址——
/// 同址标记行先于新行存在时 purge 才可能遗漏，而同址复用恰由本 purge 清理）。
template <typename T>
static void PurgeFreedRowsOfDevice(uint64_t deviceId)
{
    (void)sim::runtime::Db::Delete<T>(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&T::device_id, deviceId), HcclSim::Storage::Eq(&T::is_freed, uint64_t{1})));
}

// rank 进程加载本库时先于 main 预热仅校验模式缓存，使后续引流判定拿到确定值。
// 只读 DB、不打日志（此刻日志组件可能尚未初始化）。
#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// 主机侧大块复用区句柄（懒加载缓存，与设备侧共用同一块 HcclCommPool）
static std::atomic<void*> g_commPoolBase{nullptr};

// 记下复用区在本进程的地址，首个大块申请时由 CAS 保证只有一个线程写入。
// 首次记录时多 acquire 一次复用区且进程内不释放，使复用区在进程退出前保持映射、地址不变，
// 后续用地址判断一块内存是否在复用区时地址不会失效。
static void CacheCommPoolBase(void* base)
{
    if (base == nullptr) {
        return;
    }
    void* expected = nullptr;
    // 写入用 release、IsInCommPool 读取用 acquire 配对，保证其它线程读到的地址是有效的。
    if (g_commPoolBase.compare_exchange_strong(expected, base, std::memory_order_release, std::memory_order_relaxed)) {
        sim::MemoryManager::GetInstance().AcquireMemByName(sim::CommPoolPolicy::kPoolName);
    }
}

std::string GenDevMemName(int deviceId)
{
    std::ostringstream oss;
    oss << "DEV" << deviceId << "_" << getpid() << "_" << std::this_thread::get_id() << "_"
        << std::chrono::steady_clock::now().time_since_epoch().count();
    return oss.str();
}

bool GetPhyMemBlockByVirPtr(const void* virPtr, uint64_t& offset, sim::runtime::PhyMemBlock& phyMem)
{
    // host侧根据虚拟地址查询时需要根据device_id匹配
    uint64_t devPtr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(virPtr));
    uint64_t deviceKey = sim::runtime::GetCurrDeviceKey();
    // host侧根据虚拟地址查询时需要根据device_id匹配；地址包含查询经派生端点列下推。
    auto virResult = sim::runtime::Db::GetOneByPred<sim::runtime::VirtualMemBlock>(HcclSim::Storage::And(
        HcclSim::Storage::Le(&sim::runtime::VirtualMemBlock::dev_mapped_ptr, devPtr),
        HcclSim::Storage::Gt(&sim::runtime::VirtualMemBlock::dev_mapped_end_ptr, devPtr),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::src_type, (uint8_t)sim::runtime::VIR_MEM_TYPE_DEV),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::device_id, deviceKey)));
    if (!virResult.ok()) {
        // 未命中（NOT_FOUND）是常规语义：传入地址不属于任何 VM 设备虚拟区间，典型为主机指针。
        // 是否算错误由调用方按自身语义决定：aclrtMemset 退化为主机写，aclrtMemcpy 在调用点报错。
        // 存储层真正的执行错误（如稳定读重试耗尽 CONFLICT）仍按 ERROR 暴露，不得伪装成未命中。
        if (virResult.code == HcclSim::Storage::DbCode::NOT_FOUND) {
            HCCL_VM_WARN("no VirtualMemBlock matched virPtr:{:p}", virPtr);
        } else {
            HCCL_VM_ERROR(
                "query VirtualMemBlock failed, virPtr:{:p}, code:{:d}, diagnostic:{}", virPtr,
                static_cast<uint32_t>(virResult.code), virResult.diagnostic);
        }
        return false;
    }

    const auto& virMem = *virResult;
    auto phyMemId = virMem.phy_mem_id;
    // 未映射的 reserve 占位行 phy_mem_id=0：这是合法"尚无物理后端"状态，按未命中返回，
    // 不再 GetById(0) 刷 "can not find PhyMemBlock by phyMemId:0"。
    if (phyMemId == 0) {
        HCCL_VM_INFO("vir mem has no phy mapping yet, virPtr:{:p}, rowId:{:d}", virPtr, virMem.id);
        return false;
    }
    auto phyMemRes = sim::runtime::Db::GetById<sim::runtime::PhyMemBlock>(phyMemId);
    if (!phyMemRes.ok()) {
        HCCL_VM_ERROR("can not find PhyMemBlock by phyMemId:{:d}", phyMemId);
        return false;
    };
    offset = devPtr - virMem.dev_mapped_ptr;
    phyMem = *phyMemRes.value;
    return true;
}

uint32_t GetDeviceIdByVirAddr(const void* virAddr)
{
    sim::runtime::PhyMemBlock phyMem{};
    uint64_t offset = 0;
    if (!GetPhyMemBlockByVirPtr(virAddr, offset, phyMem)) {
        return 0;
    }

    return phyMem.device_id;
}

void* GetRealPtrByAddr(const void* virPtr)
{
    uint64_t devPtr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(virPtr));
    uint64_t deviceKey = sim::runtime::GetCurrDeviceKey();
    // 设备地址包含查询:范围条件经派生端点列完整下推。
    auto virMemRes = sim::runtime::Db::GetOneByPred<sim::runtime::VirtualMemBlock>(HcclSim::Storage::And(
        HcclSim::Storage::Le(&sim::runtime::VirtualMemBlock::dev_mapped_ptr, devPtr),
        HcclSim::Storage::Gt(&sim::runtime::VirtualMemBlock::dev_mapped_end_ptr, devPtr),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::src_type, (uint8_t)sim::runtime::VIR_MEM_TYPE_DEV),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::device_id, deviceKey)));
    if (!virMemRes.ok()) {
        HCCL_VM_ERROR("can not find this buff offset ptr:{:p}", virPtr);
        return nullptr;
    }

    const auto& virMem = *virMemRes;
    void* virStartPtr = (void*)(uintptr_t)virMem.start_ptr;
    void* devStartPtr = (void*)(uintptr_t)virMem.dev_mapped_ptr;
    auto hostPtr = sim::DeviceMemoryManager::GetInstance().GetHostPtrByDevPtr(virStartPtr);
    if (hostPtr == nullptr) {
        HCCL_VM_ERROR("can not find host ptr by dev ptr:{:p}", virStartPtr);
        return nullptr;
    }

    hostPtr = ((char*)hostPtr + (devPtr - (uint64_t)(uintptr_t)devStartPtr));
    return hostPtr;
}

// 按调用方持有的"用户可见VA"解析活跃映射块。统一模型下 reserve 只发跨 rank 同址对称 VA，
// map 后落 dev_mapped_ptr；start_ptr 仅用于 runner/checker 在无 device 上下文时的唯一解析。
// 故用户 VA 一律按 (device_id, dev_mapped 区间) 定位。
struct UserVaRange {
    uint64_t baseVa = 0;
    uint64_t size = 0;
    uint64_t phyMemId = 0;
};

static bool FindActiveVirMemByUserVa(uint64_t userVa, UserVaRange& out)
{
    uint64_t deviceKey = sim::runtime::GetCurrDeviceKey();
    HCCL_VM_INFO("find vir mem by user va, userVa:0x{:x}, deviceKey:{:d}", userVa, deviceKey);
    // 仅服务对称内存注册路径（aclrtMemGetAddressRange / aclrtMemRetainAllocationHandle）：入参恒为
    // 用户持有的映射址（= dev_mapped_ptr，bit63 GLOBAL / bit62 UB 堆窗口，已先 MapMem），故直接按
    // (device_id, dev_mapped 区间) 定位，无需区分 start_ptr，也无需按地址段选列。形参用非 const
    // uint64_t 成员指针：FieldIdOf 按 is_same_v 精确匹配 schema 字段类型。
    auto virMemRes = sim::runtime::Db::GetOneByPred<sim::runtime::VirtualMemBlock>(HcclSim::Storage::And(
        HcclSim::Storage::Le(&sim::runtime::VirtualMemBlock::dev_mapped_ptr, userVa),
        HcclSim::Storage::Gt(&sim::runtime::VirtualMemBlock::dev_mapped_end_ptr, userVa),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::src_type, (uint8_t)sim::runtime::VIR_MEM_TYPE_DEV),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::device_id, deviceKey),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::is_freed, uint64_t{0})));
    if (!virMemRes.ok()) {
        HCCL_VM_WARN(
            "vir mem range miss for userVa 0x{:x}, code:{:d}, diagnostic:{}", userVa,
            static_cast<uint32_t>(virMemRes.code), virMemRes.diagnostic);
        return false;
    }
    // 返回对用户可见的区间基址（对称窗口即 dev_mapped_ptr）。
    out.baseVa = virMemRes.value->dev_mapped_ptr;
    out.size = virMemRes.value->size;
    out.phyMemId = virMemRes.value->phy_mem_id;
    return true;
}

// ============================================================================
// 对称内存（bit63 GLOBAL / bit62 UB Memory NoUC reserve）统一仿真模型
//   1) reserve（aclrtReserveMemAddress / NoUCMemory）只发跨 rank 同址对称 VA，
//      仅整数编址、不占物理内存、不写 DB；同时把发放区间登记进进程内表（g_reservedSymVaRanges）；
//   2) aclrtMapMem 是唯一写表点：先校验入参 VA 完整落在某个已 reserve 区间内，再经物理长度/
//      区间重复占用校验后，用 AllocVirMem 另发一个按设备全局唯一、单调不复用的 start_ptr，
//      供 Runner/Checker 无 device 唯一定位本 rank 物理 shm；dev_mapped_ptr 保留入参对称 VA
//      （承载 hcomm 窗口配对 tag 与设备侧任务地址），两者写进同一条 DEV VMB；同区间重复 map 直接报错；
//   3) aclrtUnmapMem 仅物理删除该行（AllocVirMem 游标单调递增、FreeVirMem 为空桩，start_ptr 不回收）；
//      aclrtReleaseMemAddress 为 no-op（reserve 区间只增不删，VA 永不复用）。
//   物理块走 MallocPhysical(含 shm)，跨 rank 仅借 fabric handle 透传同一主键。
// ============================================================================

// 对称 VA 分配粒度（真实 RECOMMENDED=2MiB）。
static constexpr uint64_t kSymAllocGranularity = 2ULL * 1024ULL * 1024ULL;
// 触发对称堆路径的最小预留尺寸（仅 UB 堆 16GB 起步命中，避免误判）。
static constexpr uint64_t kSymHeapReserveThreshold = 8ULL * 1024ULL * 1024ULL * 1024ULL;
// UB 对称堆单成员步距（hcomm 默认 16GB）。单堆预留跨度由 reserve 请求的实际尺寸（stride × lsaTeamSize）
// 对齐到该步距取整决定，与 DeviceMemoryManager::AllocSymHeapVirMem 的跨度算法保持一致。
static constexpr uint64_t kSymHeapStride = 16ULL * 1024ULL * 1024ULL * 1024ULL;

// reserve 发放区间登记表（仅本进程内存，仅用于 aclrtMapMem 的入参归属校验）。
// reserve 只发号、不写 DB，故这里单独记录每次 reserve 实际占有的 VA 跨度；VA 单调发放、release 为
// no-op 且地址永不复用，因此登记项只增不删。begin 互不重叠（bit63 按 2MiB 槽、bit62 按整堆跨度）。
struct ReservedSymVaRange {
    uint64_t begin = 0;
    uint64_t end = 0; // 不含
};
static std::mutex g_reservedSymVaLock;
static std::map<uint64_t, ReservedSymVaRange> g_reservedSymVaRanges; // begin -> 区间

static void RegisterReservedSymVa(uint64_t begin, uint64_t span)
{
    if (begin == 0 || span == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_reservedSymVaLock);
    g_reservedSymVaRanges.emplace(begin, ReservedSymVaRange{begin, begin + span});
}

// [va, va+windowSize) 是否完整落在某个已 reserve 区间内（调用前已保证 windowSize!=0 且区间不上溢）。
static bool IsWithinReservedSymVa(uint64_t va, uint64_t windowSize)
{
    std::lock_guard<std::mutex> lock(g_reservedSymVaLock);
    // 找 begin <= va 的最后一个区间（begin 不重叠，至多一个候选）。
    auto it = g_reservedSymVaRanges.upper_bound(va);
    if (it == g_reservedSymVaRanges.begin()) {
        return false;
    }
    --it;
    return va >= it->second.begin && va + windowSize <= it->second.end;
}

// 判断一个真实内存地址是否落在复用区内，主机侧和设备侧共用。
// 设备侧传进来的是设备指针对应的真实地址（用 GetHostPtrByDevPtr 取得）。
// 地址在 [基址, 基址+kPoolSize) 范围内，说明这块内存是引流到复用区的大块。
static bool IsInCommPool(const void* ptr)
{
    void* base = g_commPoolBase.load(std::memory_order_acquire);
    return base != nullptr && ptr >= base && ptr < static_cast<char*>(base) + sim::CommPoolPolicy::kPoolSize;
}

aclError aclrtMallocHost(void** hostPtr, size_t size)
{
    bool checkOnlyMode = sim::IsCheckOnlyMode();
    // 超过复用区上界直接报错，不回退真实分配。
    if (sim::CommPoolPolicy::ExceedsCeiling(size, checkOnlyMode)) {
        HCCL_VM_ERROR("malloc host size:{:d} exceeds pool ceiling, reject", size);
        return ACL_ERROR_INTERNAL_ERROR;
    }
    // 仅校验模式下大块引流到共享复用区 HcclCommPool
    if (sim::CommPoolPolicy::ShouldRedirect(size, checkOnlyMode)) {
        void* base = g_commPoolBase.load(std::memory_order_acquire);
        if (base == nullptr) {
            base = sim::MemoryManager::GetInstance().AcquireMemByName(sim::CommPoolPolicy::kPoolName);
            if (base == nullptr) {
                HCCL_VM_ERROR("malloc host acquire HcclCommPool failed, size:{:d}", size);
                return ACL_ERROR_INTERNAL_ERROR;
            }
            CacheCommPoolBase(base);
        }
        *hostPtr = base;
        HCCL_VM_INFO("malloc host (SHM) addr:{:p}, size:{:d}", *hostPtr, size);
    } else {
        *hostPtr = malloc(size);
        HCCL_VM_INFO("malloc host addr:{:p}, size:{:d}", *hostPtr, size);
        std::lock_guard lock(g_hostAllocMutx);
        g_hostAllocRecord[(uintptr_t)(*hostPtr)] = size;
    }
    HCCL_VM_STAT_MEM_ALLOC_HOST(0, size);
    return ACL_SUCCESS;
}

aclError aclrtMallocHostWithCfg(void** ptr, uint64_t size, aclrtMallocConfig* cfg)
{
    (void)cfg;
    return aclrtMallocHost(ptr, size);
}

aclError aclrtFreeHost(void* hostPtr)
{
    HCCL_VM_INFO("free host addr:{:p}", hostPtr);
    // 池内地址：空操作（复用区生命周期由主进程管理，不在此 free）
    if (IsInCommPool(hostPtr)) {
        HCCL_VM_INFO("free host (SHM noop) addr:{:p}", hostPtr);
        return ACL_SUCCESS;
    }

    std::lock_guard lock(g_hostAllocMutx);
    auto iter = g_hostAllocRecord.find((uintptr_t)(hostPtr));
    if (iter == g_hostAllocRecord.end()) {
        HCCL_VM_INFO("free host addr:{:p} error this ptr is not aclrtMallocHost alloc", hostPtr);
        return ACL_ERROR_BAD_ALLOC;
    }

    HCCL_VM_STAT_MEM_FREE_HOST(0, iter->second);

    g_hostAllocRecord.erase((uintptr_t)(hostPtr));
    free(hostPtr);
    return ACL_SUCCESS;
}

bool GetDevMappedMemPtr(
    uint64_t devPhyId, size_t size, std::string& memName, bool& isDevMapped, void** virPtr, void** devPtr)
{
    isDevMapped = false;
    *virPtr = sim::DeviceMemoryManager::GetInstance().AllocVirMem(devPhyId, size);
    if (*virPtr == nullptr) {
        HCCL_VM_ERROR("alloc vir mem devPhyId:{:d}, size:{:d} fail", devPhyId, size);
        return false;
    }

    // --check-only 模式下大块引流到复用区，设备侧无需真实地址映射
    if (sim::CommPoolPolicy::ShouldRedirect(size, sim::IsCheckOnlyMode())) {
        *devPtr = *virPtr;
        HCCL_VM_INFO("check-only pool-redirect: use virAddr {:p} as devPtr", *virPtr);
        return true;
    }

    if (sim::GetAicpuProcMgr().IsAlive()) {
        DevMemOpPayload payload{};
        strncpy(payload.memName, memName.c_str(), sizeof(payload.memName) - 1);

        uint8_t rspCmd;
        uint32_t rspLen = 0;
        RspGetDevPtrPayload rspPayload{};
        if (sim::GetAicpuProcMgr().Request(
                PIPE_CMD_GET_DEV_PTR, &payload, sizeof(payload), rspCmd, &rspPayload, sizeof(rspPayload), rspLen)
            != 0) {
            HCCL_VM_ERROR("Request PIPE_CMD_GET_DEV_PTR failed.");
            return false;
        }

        isDevMapped = true;
        *devPtr = (void*)rspPayload.ptr;
        HCCL_VM_INFO("PIPE_CMD_GET_DEV_PTR vir mem by device {:p}", (void*)rspPayload.ptr);
    } else {
        *devPtr = *virPtr;
    }

    return true;
}

aclError aclrtMalloc(void** devPtr, size_t size, aclrtMemMallocPolicy policy)
{
    auto serverId = sim::runtime::GetCurServerId();
    if (serverId == 0) {
        return ACL_ERROR_INVALID_PARAM;
    }
    sim::runtime::Runner runner;
    if (!sim::runtime::GetCurrRunnerTls(serverId, runner)) {
        return ACL_ERROR_INVALID_PARAM;
    }
    auto currCtx = sim::runtime::Db::GetById<sim::runtime::Context>(runner.current_ctx_id);
    if (!currCtx.ok()) {
        HCCL_VM_ERROR("can not find current context:{:d}", runner.current_ctx_id);
        return ACL_ERROR_INTERNAL_ERROR;
    }

    auto deviceId = currCtx.value->device_id;

    auto dev = sim::runtime::Db::GetById<sim::runtime::Device>(deviceId);
    if (!dev.ok()) {
        HCCL_VM_ERROR("can not find device id:{:d}", deviceId);
        return ACL_ERROR_INTERNAL_ERROR;
    }

    auto curDeviceId = (uint32_t)sim::runtime::GetCurrDeviceId();

    std::string memName = GenDevMemName(deviceId);
    auto hostPtr = sim::DeviceMemoryManager::GetInstance().AllocPhyMem(memName.c_str(), deviceId, size);
    if (hostPtr == nullptr) {
        HCCL_VM_ERROR("can not alloc phy mem deviceId:{:d}, size:{:d}", deviceId, size);
        return ACL_ERROR_INTERNAL_ERROR;
    }
    // 大块引流到复用区时缓存池基址，供 IsInCommPool 判定（与主机侧共用同一基址）。
    if (sim::CommPoolPolicy::ShouldRedirect(size, sim::IsCheckOnlyMode())) {
        CacheCommPoolBase(hostPtr);
    }

    bool isDevMapped = false;
    void* virPtr = nullptr;
    void* devMappedPtr = nullptr;
    if (!GetDevMappedMemPtr(curDeviceId, size, memName, isDevMapped, &virPtr, &devMappedPtr)) {
        HCCL_VM_ERROR("can not alloc vir mem deviceId:{:d}, curDeviceId:{:d}, size:{:d}", deviceId, curDeviceId, size);
        return ACL_ERROR_INTERNAL_ERROR;
    }

    // 记录物理内存信息到数据库（插入前先清本设备标记行：见 PurgeFreedRowsOfDevice）
    PurgeFreedRowsOfDevice<sim::runtime::PhyMemBlock>(deviceId);
    sim::runtime::PhyMemBlock phyMem{};
    phyMem.device_id = deviceId;
    memcpy(phyMem.name, memName.c_str(), memName.size());
    phyMem.size = size;
    phyMem.ref_count = 1;
    phyMem.id = 0;
    auto phyInserted = sim::runtime::Db::Add<sim::runtime::PhyMemBlock>(phyMem);
    if (!phyInserted.ok() || !phyInserted.value.has_value()) {
        // 失败路径不泄漏已分配的物理内存（迁移前此处忽略 DB 失败继续执行）。
        sim::DeviceMemoryManager::GetInstance().FreePhyMem(memName.c_str(), deviceId);
        return ACL_ERROR_INTERNAL_ERROR;
    }
    auto phyMemId = phyInserted.value->value;

    // 记录虚拟内存信息到数据库（插入前先清本设备标记行：见 PurgeFreedRowsOfDevice）
    PurgeFreedRowsOfDevice<sim::runtime::VirtualMemBlock>(deviceId);
    sim::runtime::VirtualMemBlock virMem{};
    virMem.start_ptr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(virPtr));
    virMem.dev_mapped_ptr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(devMappedPtr));
    virMem.is_dev_access = static_cast<uint8_t>(isDevMapped);
    virMem.size = size;
    virMem.ctx_id = runner.current_ctx_id;
    virMem.rank_id = curDeviceId;
    virMem.device_id = deviceId;
    virMem.phy_mem_id = phyMemId;
    virMem.owner_pid = runner.pid;
    virMem.src_type = (uint8_t)sim::runtime::VIR_MEM_TYPE_DEV;
    virMem.policy = (uint8_t)policy;
    virMem.id = 0;
    auto virInserted = sim::runtime::Db::Add<sim::runtime::VirtualMemBlock>(virMem);
    if (!virInserted.ok()) {
        // 失败路径回滚已写入的物理内存记录并释放物理内存，避免残留（回滚同走
        // is_freed 标记：行不硬删，由本设备下次插入前 purge 物理清除）。
        (void)sim::runtime::Db::Update<sim::runtime::PhyMemBlock>(
            HcclSim::Storage::Eq(&sim::runtime::PhyMemBlock::id, phyMemId),
            HcclSim::Storage::Set(&sim::runtime::PhyMemBlock::is_freed, uint64_t{1}));
        sim::DeviceMemoryManager::GetInstance().FreePhyMem(memName.c_str(), deviceId);
        return ACL_ERROR_INTERNAL_ERROR;
    }

    *devPtr = isDevMapped ? devMappedPtr : virPtr;
    // 缓存设备地址到host地址
    sim::DeviceMemoryManager::GetInstance().MapDevPtrHostPtr(virPtr, hostPtr);
    HCCL_VM_STAT_MEM_ALLOC_DEV(deviceId, size);
    HCCL_VM_INFO("malloc dev addr:{:p}, devMapAddr:{:p} memName:{}, size:{:d}", virPtr, devMappedPtr, memName, size);
    return ACL_SUCCESS;
}

aclError aclrtMallocAlign32(void** devPtr, size_t size, aclrtMemMallocPolicy policy)
{
    HCCL_VM_INFO("..");
    size_t realSize = ((size + 31) / 32) * 32;
    return aclrtMalloc(devPtr, realSize, policy);
}

aclError aclrtMallocCached(void** devPtr, size_t size, aclrtMemMallocPolicy policy)
{
    HCCL_VM_INFO("..");
    return aclrtMalloc(devPtr, size, policy);
}

aclError aclrtMemFlush(void* devPtr, size_t size)
{
    (void)devPtr;
    (void)size;
    HCCL_VM_INFO("..");
    return ACL_SUCCESS;
}

aclError aclrtMemInvalidate(void* devPtr, size_t size)
{
    (void)devPtr;
    (void)size;
    HCCL_VM_INFO("..");
    return ACL_SUCCESS;
}

aclError aclrtMallocWithCfg(void** devPtr, size_t size, aclrtMemMallocPolicy policy, aclrtMallocConfig* cfg)
{
    (void)cfg;
    HCCL_VM_INFO("..");
    return aclrtMalloc(devPtr, size, policy);
}

aclError aclrtMallocForTaskScheduler(void** devPtr, size_t size, aclrtMemMallocPolicy policy, aclrtMallocConfig* cfg)
{
    (void)cfg;
    HCCL_VM_INFO("..");
    return aclrtMalloc(devPtr, size, policy);
}

aclError aclrtFree(void* devPtr)
{
    uint64_t startPtr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(devPtr));
    uint64_t deviceKey = sim::runtime::GetCurrDeviceKey();
    auto virMemRes = sim::runtime::Db::GetOneByPred<sim::runtime::VirtualMemBlock>(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::dev_mapped_ptr, startPtr),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::src_type, (uint8_t)sim::runtime::VIR_MEM_TYPE_DEV),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::device_id, deviceKey)));
    if (!virMemRes.ok()) {
        HCCL_VM_ERROR("can not find this buff offset ptr:0x{:x}", startPtr);
        return ACL_ERROR_INTERNAL_ERROR;
    }

    const auto& virMem = *virMemRes;
    // 双释放幂等守卫：旧合同"二次释放提前失败"靠行已删 NOT_FOUND 实现；释放
    // 改为标记后行对等值读可见，由本守卫等价恢复（第一次释放已置 is_freed=1）。
    if (virMem.is_freed != 0) {
        HCCL_VM_ERROR("already freed dev addr:{:p}", devPtr);
        return ACL_ERROR_INTERNAL_ERROR;
    }
    auto phyMemId = virMem.phy_mem_id;
    auto phyMemRes = sim::runtime::Db::GetById<sim::runtime::PhyMemBlock>(phyMemId);
    if (!phyMemRes.ok()) {
        HCCL_VM_ERROR("can not find phy Mem id:{:d}", phyMemId);
        return ACL_ERROR_INTERNAL_ERROR;
    };

    const auto& phyMem = *phyMemRes.value;
    auto deviceId = phyMem.device_id;
    auto dev = sim::runtime::Db::GetById<sim::runtime::Device>(deviceId);
    if (!dev.ok()) {
        HCCL_VM_ERROR("can not find device id:{:d}", deviceId);
        return ACL_ERROR_INTERNAL_ERROR;
    }

    auto devPhyId = dev.value->physical_id;

    if (sim::GetAicpuProcMgr().IsAlive()) {
        DevMemOpPayload freePayload{};
        strncpy(freePayload.memName, phyMem.name, sizeof(freePayload.memName) - 1);

        uint8_t rspCmd;
        uint32_t rspLen = 0;
        RspFreeDevPtrPayload rspPayload{};
        if (sim::GetAicpuProcMgr().Request(
                PIPE_CMD_FREE_DEV_PTR, &freePayload, sizeof(freePayload), rspCmd, &rspPayload, sizeof(rspPayload),
                rspLen)
            != 0) {
            HCCL_VM_ERROR("Request PIPE_CMD_FREE_DEV_PTR failed for {}", phyMem.name);
        }
    }

    // 先取这块内存的真实地址，判断它是否在复用区（必须在 Unmap 之前取）。
    // 第二次释放时映射已删、取不到地址，就按非复用区处理，天然幂等。
    void* backing = sim::DeviceMemoryManager::GetInstance().GetHostPtrByDevPtr(devPtr);

    // 去缓存设备地址到host地址
    sim::DeviceMemoryManager::GetInstance().UnmapDevPtrHostPtr(devPtr);

    // 删除内存
    sim::DeviceMemoryManager::GetInstance().FreeVirMem(devPhyId, devPtr);
    if (IsInCommPool(backing)) {
        sim::MemoryManager::GetInstance().ReleaseMemByName(sim::CommPoolPolicy::kPoolName);
    } else {
        sim::DeviceMemoryManager::GetInstance().FreePhyMem(phyMem.name, deviceId);
    }

    // 删除数据库记录 → 标记释放（行保留，不硬删）：FR1 128p / R3K4 256p 拆卸期
    // 取证显示 43/48 次远端地址解析未命中全部晚于目标行成功删除 16.7s 以上
    // （删除回执全部 code=0/affectedRows=1）——行保留使慢 rank 设备进程的迟到
    // 解析仍命中，任务行不再丢失；标记行由本设备下次插入前
    // PurgeFreedRowsOfDevice 物理清除（保证标记行与活跃行永不同址共存）；
    // 二次释放由入口 is_freed 守卫挡住。已披露语义边界：对已标记缓冲的
    // MapMem/UnmapMem/IPC 导入等管理操作会命中标记行（use-after-free 场景）。
    (void)sim::runtime::Db::Update<sim::runtime::PhyMemBlock>(
        HcclSim::Storage::Eq(&sim::runtime::PhyMemBlock::id, phyMemId),
        HcclSim::Storage::Set(&sim::runtime::PhyMemBlock::is_freed, uint64_t{1}));
    (void)sim::runtime::Db::Update<sim::runtime::VirtualMemBlock>(
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::id, virMem.id),
        HcclSim::Storage::Set(&sim::runtime::VirtualMemBlock::is_freed, uint64_t{1}));
    HCCL_VM_STAT_MEM_FREE_DEV(virMem.device_id, virMem.size);
    HCCL_VM_INFO("free dev addr:{:p}, memName:{}, size:{:d}", devPtr, phyMem.name, virMem.size);
    return ACL_SUCCESS;
}

aclError aclrtMemset(void* devPtr, size_t maxCount, int32_t value, size_t count)
{
    if (count > maxCount) {
        HCCL_VM_ERROR("memset count({}) exceeds maxCount({}), ptr:{:p}", count, maxCount, devPtr);
        return ACL_ERROR_INVALID_PARAM;
    }

    sim::runtime::PhyMemBlock phyMem{};
    uint64_t offset = 0;
    if (!GetPhyMemBlockByVirPtr(devPtr, offset, phyMem)) {
        memset(devPtr, value, count);
        HCCL_VM_INFO(
            "sys mem memset ptr:{:p}, maxCount: {:d}, value: {:d}, count: {:d}", devPtr, maxCount, value, count);
        return ACL_SUCCESS;
    }

    if (offset > phyMem.size || count > phyMem.size - offset) {
        HCCL_VM_ERROR("memset range overflow, offset:{}, count:{}, phyMemSize:{}", offset, count, phyMem.size);
        return ACL_ERROR_INVALID_PARAM;
    }

    char* hostPtr
        = (char*)sim::DeviceMemoryManager::GetInstance().AcquirePhyMem(phyMem.name, phyMem.device_id, phyMem.size);
    if (hostPtr == nullptr) {
        HCCL_VM_INFO("dev mem memset ptr:{:p} name:{} can not get host ptr", devPtr, phyMem.name);
        return ACL_ERROR_INTERNAL_ERROR;
    }

    // 目的设备块按 size 判是否在复用区，与申请一致，命中则跳过实际写入并按池释放。
    bool pooled = sim::CommPoolPolicy::ShouldRedirect(phyMem.size, sim::IsCheckOnlyMode());
    if (pooled) {
        HCCL_VM_INFO("memset skip (pool) ptr:{:p}, size:{:d}, count:{:d}", devPtr, phyMem.size, count);
    } else {
        memset(hostPtr + offset, value, count);
    }
    if (pooled) {
        sim::MemoryManager::GetInstance().ReleaseMemByName(sim::CommPoolPolicy::kPoolName);
    } else {
        sim::DeviceMemoryManager::GetInstance().ReleasePhyMem(phyMem.name, phyMem.device_id);
    }
    HCCL_VM_INFO("dev mem memset ptr:{:p}, memName:{}, count: {:d}", devPtr, phyMem.name, count);
    return ACL_SUCCESS;
}

aclError aclrtMemsetAsync(void* devPtr, size_t maxCount, int32_t value, size_t count, aclrtStream stream)
{
    (void)stream;
    HCCL_VM_INFO("..");
    // 当前先不生成任务，后续有需要再根据实际情况生成任务
    return aclrtMemset(devPtr, maxCount, value, count);
}

aclError aclrtMemcpy(void* dst, size_t destMax, const void* src, size_t count, aclrtMemcpyKind kind)
{
    if (count > destMax) {
        HCCL_VM_ERROR("memcpy count({}) exceeds destMax({}), dst:{:p}", count, destMax, dst);
        return ACL_ERROR_INVALID_PARAM;
    }
    void* hostPtr = nullptr;
    void* devPtr = nullptr;
    if (kind == ACL_MEMCPY_HOST_TO_HOST) {
        memcpy(dst, src, count);
        return ACL_SUCCESS;
    } else if (kind == ACL_MEMCPY_HOST_TO_DEVICE) {
        hostPtr = const_cast<void*>(src);
        devPtr = dst;
    } else if (kind == ACL_MEMCPY_DEVICE_TO_HOST) {
        hostPtr = dst;
        devPtr = const_cast<void*>(src);
    } else if (kind == ACL_MEMCPY_DEVICE_TO_DEVICE) {
        auto dstAddr = GetRealPtrByAddr(dst);
        if (dstAddr == nullptr) {
            HCCL_VM_WARN(" only support self D2D memcpy");
            return ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
        }
        auto srcAddr = GetRealPtrByAddr(src);
        if (srcAddr == nullptr) {
            HCCL_VM_WARN("only support self D2D memcpy");
            return ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
        }
        // 目的设备块按 size 判是否在复用区，与申请一致，命中则跳过实际拷贝。
        if (IsInCommPool(dstAddr)) {
            HCCL_VM_INFO("memcpy D2D skip (dst big->pool) count:{:d}", count);
        } else {
            memcpy(dstAddr, srcAddr, count);
        }
        return ACL_SUCCESS;
    } else {
        HCCL_VM_INFO("src:{:p} to dst:{:p}, size:{:d} type: {:d} not support", src, dst, count, (int)kind);
        return ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
    }

    sim::runtime::PhyMemBlock phyMem{};
    uint64_t offset = 0;
    if (!GetPhyMemBlockByVirPtr(devPtr, offset, phyMem)) {
        HCCL_VM_ERROR(
            "aclrtMemcpy: devPtr:{:p} is not a VM device address, count:{:d}, kind:{:d}", devPtr, count, (int)kind);
        return ACL_ERROR_INTERNAL_ERROR;
    }

    if (offset > phyMem.size || count > phyMem.size - offset) {
        HCCL_VM_ERROR("memcpy range overflow, offset:{}, count:{}, phyMemSize:{}", offset, count, phyMem.size);
        return ACL_ERROR_INVALID_PARAM;
    }

    HCCL_VM_INFO("host:{:p} dev:{:p} memName:{} size:{:d} type:{:d}", hostPtr, devPtr, phyMem.name, count, (int)kind);
    char* hostDevPtr
        = (char*)sim::DeviceMemoryManager::GetInstance().AcquirePhyMem(phyMem.name, phyMem.device_id, phyMem.size);
    if (hostDevPtr == nullptr) {
        HCCL_VM_INFO(" dev mem memcpy ptr:{:p} name:{} can not get host ptr", devPtr, phyMem.name);
        return ACL_ERROR_INTERNAL_ERROR;
    }

    // 目的设备块按 size 判是否在复用区，与申请一致，命中则跳过实际拷贝并按池释放。
    // D2H 的目的是主机缓冲，仍按地址判断主机缓冲是否在复用区。
    bool devPooled = sim::CommPoolPolicy::ShouldRedirect(phyMem.size, sim::IsCheckOnlyMode());
    if (kind == ACL_MEMCPY_HOST_TO_DEVICE) {
        if (devPooled) {
            HCCL_VM_INFO("memcpy H2D skip (dst big->pool) count:{:d}", count);
        } else {
            memcpy(hostDevPtr + offset, hostPtr, count);
        }
    } else if (kind == ACL_MEMCPY_DEVICE_TO_HOST) {
        if (IsInCommPool(hostPtr)) {
            HCCL_VM_INFO("memcpy D2H skip (dst host pool) count:{:d}", count);
        } else {
            memcpy(hostPtr, hostDevPtr + offset, count);
        }
    }

    if (devPooled) {
        sim::MemoryManager::GetInstance().ReleaseMemByName(sim::CommPoolPolicy::kPoolName);
    } else {
        sim::DeviceMemoryManager::GetInstance().ReleasePhyMem(phyMem.name, phyMem.device_id);
    }

    return ACL_SUCCESS;
}

// deviceId → 通信域内rankId; 查询失败(如该device非本通信域成员)以UINT32_MAX标记
static uint32_t ResolveRankIdByDevice(uint64_t commId, uint32_t deviceId)
{
    uint32_t rankId = UINT32_MAX;
    if (!sim::runtime::GetCommRankByDeviceId(commId, deviceId, rankId)) {
        return UINT32_MAX;
    }
    return rankId;
}

aclError
aclrtMemcpyAsync(void* dst, size_t destMax, const void* src, size_t count, aclrtMemcpyKind kind, aclrtStream stream)
{
    if (count > destMax) {
        HCCL_VM_ERROR("memcpyAsync count({}) exceeds destMax({}), dst:{:p}", count, destMax, dst);
        return ACL_ERROR_INVALID_PARAM;
    }
    void* hostPtr = nullptr;
    void* devPtr = nullptr;
    if (kind == ACL_MEMCPY_HOST_TO_HOST) {
        memcpy(dst, src, count);
        return ACL_SUCCESS;
    } else if (kind == ACL_MEMCPY_HOST_TO_DEVICE) {
        hostPtr = const_cast<void*>(src);
        devPtr = dst;
    } else if (kind == ACL_MEMCPY_DEVICE_TO_HOST) {
        hostPtr = dst;
        devPtr = const_cast<void*>(src);
    } else if (kind == ACL_MEMCPY_DEVICE_TO_DEVICE) {
        if (sim::runtime::g_cur_comm_key == 0) { // 临时规避集合通信算子外的D2D内存搬运task生成
            HCCL_VM_WARN("commId invalid, D2D task do not insert, commId={}", sim::runtime::g_cur_comm_key);
            return ACL_SUCCESS;
        }
        auto deviceId = (uint32_t)sim::runtime::GetCurrDeviceId();
        HcclTaskMetaData taskMetaData{};
        taskMetaData.taskType = HccLTaskMetaType::MEM_CPY;
        taskMetaData.profInfo = HcclTaskProfInfo(
            ProfTaskParamType::TASK_SDMA, ProfLinkType::LINK_ONCHIP,
            ProfTransportType::TRANSPORT_LOCAL); // hcomm LocalCopy
        taskMetaData.deviceId = deviceId;
        taskMetaData.commId = sim::runtime::g_cur_comm_key;
        if (!sim::runtime::GetCommRankByDeviceId(taskMetaData.commId, deviceId, taskMetaData.rankId)) {
            HCCL_VM_ERROR("device memcpy task has no valid communicator, deviceId={}", deviceId);
            return ACL_ERROR_INVALID_PARAM;
        }
        taskMetaData.streamId = reinterpret_cast<uint64_t>(stream);
        taskMetaData.taskData.transMem.srcOffset = sim::GetVirPtrByDevPtr(reinterpret_cast<uint64_t>(src));
        taskMetaData.taskData.transMem.dstOffset = sim::GetVirPtrByDevPtr(reinterpret_cast<uint64_t>(dst));
        // 评估模式: deviceId字段承载通信域rankId; 主线保持真实deviceId供checker/runner消费
        if (sim::IsEstimateMode()) {
            taskMetaData.taskData.transMem.srcDeviceId
                = ResolveRankIdByDevice(taskMetaData.commId, GetDeviceIdByVirAddr(src));
            taskMetaData.taskData.transMem.dstDeviceId
                = ResolveRankIdByDevice(taskMetaData.commId, GetDeviceIdByVirAddr(dst));
        } else {
            taskMetaData.taskData.transMem.srcDeviceId = GetDeviceIdByVirAddr(src);
            taskMetaData.taskData.transMem.dstDeviceId = GetDeviceIdByVirAddr(dst);
        }
        taskMetaData.taskData.transMem.len = count;

        uint32_t index{0};
        auto ret = InsertTaskToCollection(&taskMetaData, &index);
        if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
            HCCL_VM_ERROR("InsertTaskToCollection fail");
            return ACL_ERROR_INTERNAL_ERROR;
        }

        // 下发cid
        if (taskMetaData.commId > UINT16_MAX || taskMetaData.rankId > UINT16_MAX) {
            HCCL_VM_ERROR(
                "device memcpy task identity exceeds task cid range, commId={}, rankId={}", taskMetaData.commId,
                taskMetaData.rankId);
            return ACL_ERROR_INVALID_PARAM;
        }
        HcclTaskCid taskCid{};
        taskCid.field.commId = static_cast<uint16_t>(taskMetaData.commId);
        taskCid.field.rankId = static_cast<uint16_t>(taskMetaData.rankId);
        taskCid.field.index = index;
        sim::runtime::Task task{};
        task.stream_id = taskMetaData.streamId;
        task.cid = taskCid.value;
        task.type = (uint8_t)HccLTaskMetaType::MEM_CPY;

        task.id = 0;
        auto inserted = sim::runtime::Db::Add<sim::runtime::Task>(task);
        if (!inserted.ok()) {
            return ACL_ERROR_INTERNAL_ERROR;
        }
        return ACL_SUCCESS;
    } else {
        HCCL_VM_INFO("src:{:p} to dst:{:p}, size:{:d} type: {:d} not support", src, dst, count, (int)kind);
        return ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
    }

    sim::runtime::PhyMemBlock phyMem{};
    uint64_t offset = 0;
    if (!GetPhyMemBlockByVirPtr(devPtr, offset, phyMem)) {
        HCCL_VM_ERROR(
            "aclrtMemcpyAsync: devPtr:{:p} is not a VM device address (host memory?), count:{:d}, kind:{:d}", devPtr,
            count, (int)kind);
        return ACL_ERROR_INTERNAL_ERROR;
    }

    if (offset > phyMem.size || count > phyMem.size - offset) {
        HCCL_VM_ERROR("memcpyAsync range overflow, offset:{}, count:{}, phyMemSize:{}", offset, count, phyMem.size);
        return ACL_ERROR_INVALID_PARAM;
    }

    HCCL_VM_INFO("host:{:p} dev:{:p} memName:{} size:{:d} type:{:d}", hostPtr, devPtr, phyMem.name, count, (int)kind);

    char* hostDevPtr
        = (char*)sim::DeviceMemoryManager::GetInstance().AcquirePhyMem(phyMem.name, phyMem.device_id, phyMem.size);
    if (hostDevPtr == nullptr) {
        HCCL_VM_INFO("dev mem memcpy ptr:{:p} name:{} can not get host ptr", devPtr, phyMem.name);
        return ACL_ERROR_INTERNAL_ERROR;
    }

    // 目的设备块按 size 判是否在复用区，与申请一致，命中则跳过实际拷贝并按池释放。
    // D2H 的目的是主机缓冲，仍按地址判断主机缓冲是否在复用区。
    bool devPooled = sim::CommPoolPolicy::ShouldRedirect(phyMem.size, sim::IsCheckOnlyMode());
    if (kind == ACL_MEMCPY_HOST_TO_DEVICE) {
        if (devPooled) {
            HCCL_VM_INFO("memcpy H2D skip (dst big->pool) count:{:d}", count);
        } else {
            memcpy(hostDevPtr + offset, hostPtr, count);
        }
    } else if (kind == ACL_MEMCPY_DEVICE_TO_HOST) {
        if (IsInCommPool(hostPtr)) {
            HCCL_VM_INFO("memcpy D2H skip (dst host pool) count:{:d}", count);
        } else {
            memcpy(hostPtr, hostDevPtr + offset, count);
        }
    }

    if (devPooled) {
        sim::MemoryManager::GetInstance().ReleaseMemByName(sim::CommPoolPolicy::kPoolName);
    } else {
        sim::DeviceMemoryManager::GetInstance().ReleasePhyMem(phyMem.name, phyMem.device_id);
    }

    return ACL_SUCCESS;
}

aclError aclrtMemcpyAsyncWithCondition(
    void* dst, size_t destMax, const void* src, size_t count, aclrtMemcpyKind kind, aclrtStream stream)
{
    return aclrtMemcpyAsync(dst, destMax, src, count, kind, stream);
}

aclError aclrtMemcpyBatch(
    void** dsts, size_t* destMaxs, void** srcs, size_t* sizes, size_t numBatches, aclrtMemcpyBatchAttr* attrs,
    size_t* attrsIndexes, size_t numAttrs, size_t* failIndex)
{
    (void)attrs;
    (void)attrsIndexes;
    (void)numAttrs;
    (void)failIndex;
    for (size_t i = 0; i < numBatches; i++) {
        aclrtMemcpy(dsts[i], destMaxs[i], srcs[i], sizes[i], ACL_MEMCPY_DEFAULT);
    }

    return ACL_SUCCESS;
}

aclError aclrtMemcpyBatchAsync(
    void** dsts, size_t* destMaxs, void** srcs, size_t* sizes, size_t numBatches, aclrtMemcpyBatchAttr* attrs,
    size_t* attrsIndexes, size_t numAttrs, size_t* failIndex, aclrtStream stream)
{
    (void)attrs;
    (void)attrsIndexes;
    (void)numAttrs;
    (void)failIndex;
    for (size_t i = 0; i < numBatches; i++) {
        aclrtMemcpyAsync(dsts[i], destMaxs[i], srcs[i], sizes[i], ACL_MEMCPY_DEFAULT, stream);
    }

    return ACL_SUCCESS;
}

aclError aclrtMemcpy2d(
    void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, aclrtMemcpyKind kind)
{
    (void)dst;
    (void)dpitch;
    (void)src;
    (void)spitch;
    (void)width;
    (void)height;
    if (kind != ACL_MEMCPY_HOST_TO_DEVICE && kind != ACL_MEMCPY_DEVICE_TO_HOST) {
        return ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
    }

    return ACL_SUCCESS;
}

aclError aclrtMemcpy2dAsync(
    void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, aclrtMemcpyKind kind,
    aclrtStream stream)
{
    (void)dst;
    (void)dpitch;
    (void)src;
    (void)spitch;
    (void)width;
    (void)height;
    (void)kind;
    (void)stream;
    return ACL_SUCCESS;
}

aclError aclrtGetMemcpyDescSize(aclrtMemcpyKind kind, size_t* descSize)
{
    (void)kind;
    (void)descSize;
    return ACL_SUCCESS;
}

aclError aclrtSetMemcpyDesc(void* desc, aclrtMemcpyKind kind, void* srcAddr, void* dstAddr, size_t count, void* config)
{
    (void)desc;
    (void)kind;
    (void)srcAddr;
    (void)dstAddr;
    (void)count;
    (void)config;
    return ACL_SUCCESS;
}

aclError aclrtMemcpyAsyncWithDesc(void* desc, aclrtMemcpyKind kind, aclrtStream stream)
{
    (void)desc;
    (void)kind;
    (void)stream;
    return ACL_SUCCESS;
}

aclError aclrtMallocPhysical(aclrtDrvMemHandle* handle, size_t size, const aclrtPhysicalMemProp* prop, uint64_t flags)
{
    (void)prop;
    (void)flags;
    auto serverId = sim::runtime::GetCurServerId();
    if (serverId == 0) {
        return ACL_ERROR_INVALID_PARAM;
    }
    sim::runtime::Runner runner;
    if (!sim::runtime::GetCurrRunnerTls(serverId, runner)) {
        return ACL_ERROR_INVALID_PARAM;
    }
    auto currCtx = sim::runtime::Db::GetById<sim::runtime::Context>(runner.current_ctx_id);
    if (!currCtx.ok()) {
        HCCL_VM_ERROR("can not find current context:{:d}", runner.current_ctx_id);
        return ACL_ERROR_INTERNAL_ERROR;
    }

    auto deviceId = currCtx.value->device_id;

    std::string memName = GenDevMemName(deviceId);
    auto hostPtr = sim::DeviceMemoryManager::GetInstance().AllocPhyMem(memName.c_str(), deviceId, size);
    if (hostPtr == nullptr) {
        HCCL_VM_ERROR("can not alloc phy mem deviceId:{:d}, size:{:d}", deviceId, size);
        return ACL_ERROR_INTERNAL_ERROR;
    }
    // 大块物理内存同样引流到复用区，缓存池基址，使其被 map 后的 memcpy/memset 能按地址判定。
    if (sim::CommPoolPolicy::ShouldRedirect(size, sim::IsCheckOnlyMode())) {
        CacheCommPoolBase(hostPtr);
    }

    // 记录物理内存信息到数据库（插入前先清本设备标记行：见 PurgeFreedRowsOfDevice）
    PurgeFreedRowsOfDevice<sim::runtime::PhyMemBlock>(deviceId);
    sim::runtime::PhyMemBlock phyMem{};
    phyMem.device_id = deviceId;
    memcpy(phyMem.name, memName.c_str(), memName.size());
    phyMem.size = size;
    phyMem.ref_count = 1;
    phyMem.id = 0;
    auto inserted = sim::runtime::Db::Add<sim::runtime::PhyMemBlock>(phyMem);
    if (!inserted.ok() || !inserted.value.has_value()) {
        return ACL_ERROR_INTERNAL_ERROR;
    }
    auto phyMemId = inserted.value->value;
    HCCL_VM_STAT_MEM_ALLOC_VIR(deviceId, size);

    HCCL_VM_INFO("malloc phy mem name:{}, size:{:d}, id:{:d}", memName, size, phyMemId);
    *handle = (aclrtDrvMemHandle)phyMemId;
    return ACL_SUCCESS;
}

aclError aclrtFreePhysical(aclrtDrvMemHandle handle)
{
    auto phyMemId = (uint64_t)(uintptr_t)handle;
    auto phyMemRes = sim::runtime::Db::GetById<sim::runtime::PhyMemBlock>(phyMemId);
    if (!phyMemRes.ok()) {
        HCCL_VM_ERROR("can not find phy Mem id:{:d}", phyMemId);
        return ACL_ERROR_INTERNAL_ERROR;
    };

    // 严格"哪里申请哪里释放"：仅当当前设备即属主设备时才真正释放。两端 id 同口径
    // （aclrtMallocPhysical 写入的 PhyMemBlock.device_id 与 GetCurrDeviceId 均为 Device
    // 表主键）。仿真 fabric/shareable handle 透传属主全局 phyMemId，导入端(peer)句柄数值
    // 与属主相同；真机导入句柄每端独立、free 只关闭本端导入映射而不释放属主分配，故非属主
    // 设备直接成功返回，不触碰属主 shm/PhyMemBlock。
    const uint64_t localDeviceId = sim::runtime::GetCurrDeviceId();
    if (localDeviceId != phyMemRes.value->device_id) {
        HCCL_VM_INFO(
            "skip free peer-owned phy mem, id:{:d}, ownerDeviceId:{:d}, localDeviceId:{:d}", phyMemId,
            phyMemRes.value->device_id, localDeviceId);
        return ACL_SUCCESS;
    }

    // 物理内存无 devPtr→host 映射，按 size 判定是否走了复用区（与申请同一判据）
    if (sim::CommPoolPolicy::ShouldRedirect(phyMemRes.value->size, sim::IsCheckOnlyMode())) {
        sim::MemoryManager::GetInstance().ReleaseMemByName(sim::CommPoolPolicy::kPoolName);
    } else {
        sim::DeviceMemoryManager::GetInstance().FreePhyMem(phyMemRes.value->name, phyMemRes.value->device_id);
    }
    // 删除数据库记录 → 标记释放（行保留；合同与清理时机见 aclrtFree 处注释）
    auto freed = sim::runtime::Db::Update<sim::runtime::PhyMemBlock>(
        HcclSim::Storage::Eq(&sim::runtime::PhyMemBlock::id, phyMemId),
        HcclSim::Storage::Set(&sim::runtime::PhyMemBlock::is_freed, uint64_t{1}));
    if (!freed.ok()) {
        return ACL_ERROR_INTERNAL_ERROR;
    }
    HCCL_VM_STAT_MEM_FREE_VIR(phyMemRes.value->device_id, phyMemRes.value->size);
    HCCL_VM_INFO("free phy mem, id:{:d}", phyMemId);

    return ACL_SUCCESS;
}

// 统一模型：reserve 只发一个跨 rank 同址的对称 VA（bit63 GLOBAL 段），仅整数编址，
// 不占物理内存、不写 DB。物理绑定与 VirtualMemBlock 写表全部延迟到 aclrtMapMem 统一处理。
aclError aclrtReserveMemAddress(void** virPtr, size_t size, size_t alignment, void* expectPtr, uint64_t flags)
{
    (void)alignment;
    (void)expectPtr;
    (void)flags;
    if (virPtr == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    void* symPtr = sim::DeviceMemoryManager::GetInstance().AllocGlobalVirMem(size);
    if (symPtr == nullptr) {
        HCCL_VM_ERROR("can not alloc global sym vir mem, size:{:d}", size);
        return ACL_ERROR_INTERNAL_ERROR;
    }
    // 登记本次 reserve 实际占有的 2MiB 槽跨度（与 AllocGlobalVirMem 的游标步进一致），供 map 归属校验。
    uint64_t slots = (static_cast<uint64_t>(size) + kSymAllocGranularity - 1) / kSymAllocGranularity;
    RegisterReservedSymVa(reinterpret_cast<uintptr_t>(symPtr), (slots == 0 ? 1 : slots) * kSymAllocGranularity);
    *virPtr = symPtr;
    HCCL_VM_INFO("reserve symmetric vir mem, symVa:{:p}, size:{:d}", symPtr, size);
    return ACL_SUCCESS;
}

// NoUC 版：UB 对称内存。大块（>=阈值）发跨 rank 同址的 bit62 堆基址；小块退化为普通 bit63 reserve。
// 同样只发号、不写表；堆内每个窗口的映射与写表在 aclrtMapMem 统一完成。
aclError aclrtReserveMemAddressNoUCMemory(void** virPtr, size_t size, size_t alignment, void* expectPtr, uint64_t flags)
{
    (void)alignment;
    (void)expectPtr;
    (void)flags;
    if (virPtr == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    if (size < kSymHeapReserveThreshold) {
        return aclrtReserveMemAddress(virPtr, size, alignment, expectPtr, flags);
    }
    void* heapPtr = sim::DeviceMemoryManager::GetInstance().AllocSymHeapVirMem(size);
    if (heapPtr == nullptr) {
        HCCL_VM_ERROR("can not alloc symmetric heap base, size:{:d}", size);
        return ACL_ERROR_INTERNAL_ERROR;
    }
    // 登记整个堆的 VA 跨度：跨度 = round_up(size, 16GB 步距)，覆盖 hcomm 实际 map 的任意成员窗口
    // （virPtr = 堆基址 + member*stride + windowOffset，member 上界 = lsaTeamSize - 1）。
    const uint64_t heapSpan = (static_cast<uint64_t>(size) + kSymHeapStride - 1) / kSymHeapStride * kSymHeapStride;
    RegisterReservedSymVa(reinterpret_cast<uintptr_t>(heapPtr), heapSpan);
    *virPtr = heapPtr;
    HCCL_VM_INFO("reserve symmetric heap base:{:p}, size:{:d}", heapPtr, size);
    return ACL_SUCCESS;
}

aclError aclrtReleaseMemAddress(void* virPtr)
{
    // 统一模型下 reserve 只发对称 VA、不落表；映射行已在 aclrtUnmapMem 物理删除。
    // 故 release 无需操作 DB（VA 游标单调递增、不复用，FreeVirMem 为空桩）。
    // 同时覆盖 hcomm 失败清理路径：reserve 成功但 malloc/map 失败时直接 release（尚无映射行）。
    HCCL_VM_INFO("release symmetric vir mem(no-op), va:{:p}", virPtr);
    return ACL_SUCCESS;
}

// 判断入参 VA 是否为 reserve 发放的对称 VA 段（bit63 GLOBAL 或 bit62 UB 堆窗口），
// 并校验 [va, va+size) 不越界环绕。普通 malloc 的按卡编址/主机指针一律不得走对称 map。
static bool IsValidSymmetricWindowVa(uint64_t va, uint64_t size)
{
    constexpr uint64_t kBit62Base = 1ULL << 62;
    if (va < kBit62Base || size == 0) {
        return false;
    }
    return va <= UINT64_MAX - (size - 1); // [va, va+size) 不上溢
}

aclError aclrtMapMem(void* virPtr, size_t size, size_t offset, aclrtDrvMemHandle handle, uint64_t flags)
{
    (void)offset;
    (void)flags;
    uint64_t phyMemId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(handle));
    uint64_t symVa = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(virPtr));

    // 1) 入参对称 VA 有效性 + 长度范围校验。
    if (!IsValidSymmetricWindowVa(symVa, size)) {
        HCCL_VM_ERROR("invalid symmetric window va:0x{:x} or size:{:d}", symVa, size);
        return ACL_ERROR_INVALID_PARAM;
    }
    // virPtr 必须落在本进程此前 reserve 发放的某个对称 VA 区间内，且窗口不得跨出该区间
    // （reserve 只发号不写表，故用进程内登记表做归属校验；防止传入未 reserve/越界的对称地址）。
    if (!IsWithinReservedSymVa(symVa, size)) {
        HCCL_VM_ERROR("sym window va:0x{:x} size:{:d} not within any reserved symmetric va range", symVa, size);
        return ACL_ERROR_INVALID_PARAM;
    }

    // 2) 物理块必须存在，且窗口长度不得超过物理内存长度。
    auto phyMemRes = sim::runtime::Db::GetById<sim::runtime::PhyMemBlock>(phyMemId);
    if (!phyMemRes.ok()) {
        HCCL_VM_ERROR("can not find phy Mem id:{:d}", phyMemId);
        return ACL_ERROR_INVALID_PARAM;
    }
    if (size > phyMemRes.value->size) {
        HCCL_VM_ERROR(
            "sym window size:{:d} exceeds phy mem size:{:d}, symVa:0x{:x}, phyMemId:{:d}", size, phyMemRes.value->size,
            symVa, phyMemId);
        return ACL_ERROR_INVALID_PARAM;
    }

    // bit63 GLOBAL 段与 bit62 UB 堆窗口在映射阶段统一处理：virPtr 恒为跨 rank 同址的对称 VA。
    auto serverId = sim::runtime::GetCurServerId();
    if (serverId == 0) {
        return ACL_ERROR_INVALID_PARAM;
    }
    sim::runtime::Runner runner;
    if (!sim::runtime::GetCurrRunnerTls(serverId, runner)) {
        HCCL_VM_ERROR("map symmetric window without active runner, va:0x{:x}", symVa);
        return ACL_ERROR_INVALID_PARAM;
    }
    uint64_t deviceId = sim::runtime::GetCurrDeviceKey();

    // 3) 重复占用校验：本设备上 [symVa, symVa+size) 不得与任何活跃窗口的 dev_mapped 区间重叠
    //    （含同址重复 map）。hcomm 重映射必先 UnmapMem 硬删行，故命中重叠即非法，直接报错，不做 update。
    auto overlap = sim::runtime::Db::GetOneByPred<sim::runtime::VirtualMemBlock>(HcclSim::Storage::And(
        HcclSim::Storage::Le(&sim::runtime::VirtualMemBlock::dev_mapped_ptr, symVa + size - 1),
        HcclSim::Storage::Gt(&sim::runtime::VirtualMemBlock::dev_mapped_end_ptr, symVa),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::src_type, (uint8_t)sim::runtime::VIR_MEM_TYPE_DEV),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::device_id, deviceId)));
    if (overlap.code == HcclSim::Storage::DbCode::OK) {
        HCCL_VM_ERROR(
            "symmetric va range already mapped, symVa:0x{:x}, size:{:d}, occupied:[0x{:x},0x{:x}), deviceId:{:d}",
            symVa, size, overlap.value->dev_mapped_ptr, overlap.value->dev_mapped_end_ptr, deviceId);
        return ACL_ERROR_INVALID_PARAM;
    }
    if (overlap.code != HcclSim::Storage::DbCode::NOT_FOUND) {
        HCCL_VM_ERROR(
            "query symmetric window occupancy failed, symVa:0x{:x}, code:{:d}", symVa,
            static_cast<uint32_t>(overlap.code));
        return ACL_ERROR_INTERNAL_ERROR;
    }

    // 新窗口：start_ptr 另发一个按设备全局唯一的本地址（供 Runner/Checker 无 device 唯一定位本 rank
    // 物理 shm，offset 落在本窗口 shm 范围内）；dev_mapped_ptr 保留跨 rank 同址的入参对称 VA，承载
    // hcomm 窗口配对 tag 与设备侧任务地址。import（V1/V2）透传属主同一 phyMemId，同槽多行物理等价。
    void* localPtr = sim::DeviceMemoryManager::GetInstance().AllocVirMem(deviceId, size);
    if (localPtr == nullptr) {
        HCCL_VM_ERROR("can not alloc local vir mem for symmetric window, deviceId:{:d}, size:{:d}", deviceId, size);
        return ACL_ERROR_INTERNAL_ERROR;
    }
    sim::runtime::VirtualMemBlock win{};
    win.start_ptr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(localPtr));
    win.dev_mapped_ptr = symVa;
    win.dev_mapped_end_ptr = symVa + size;
    win.is_dev_access = 1;
    win.size = size;
    win.ctx_id = runner.current_ctx_id;
    win.rank_id = deviceId;
    win.device_id = deviceId;
    win.phy_mem_id = phyMemId;
    win.owner_pid = runner.pid;
    win.src_type = (uint8_t)sim::runtime::VIR_MEM_TYPE_DEV;
    win.policy = 0;
    win.id = 0;
    if (!sim::runtime::Db::Add<sim::runtime::VirtualMemBlock>(win).ok()) {
        HCCL_VM_ERROR("add symmetric window VMB failed, symVa:0x{:x}", symVa);
        return ACL_ERROR_INTERNAL_ERROR;
    }
    HCCL_VM_INFO(
        "map symmetric window, symVa:0x{:x}, localStart:0x{:x}, size:{:d}, phyMemId:{:d}, deviceId:{:d}, "
        "ownerPid:{:d}",
        symVa, win.start_ptr, size, phyMemId, deviceId, runner.pid);
    return ACL_SUCCESS;
}

aclError aclrtUnmapMem(void* virPtr)
{
    uint64_t symVa = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(virPtr));
    uint64_t deviceId = sim::runtime::GetCurrDeviceKey();

    // 统一模型：映射行按 (本 device, dev_mapped==入参对称 VA) 唯一定位，物理删除（不使用 is_freed）。
    // start_ptr 仅为随 map 单调取号、不回收（与原 AllocVirMem 一致），故 unmap 无需处理它；
    // 窗口是否存活完全以本行是否存在为准。重复 unmap 幂等成功。
    auto winRes = sim::runtime::Db::GetOneByPred<sim::runtime::VirtualMemBlock>(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::dev_mapped_ptr, symVa),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::src_type, (uint8_t)sim::runtime::VIR_MEM_TYPE_DEV),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::device_id, deviceId)));
    if (!winRes.ok()) {
        HCCL_VM_INFO("symmetric window already unmapped or missing, symVa:0x{:x}, deviceId:{:d}", symVa, deviceId);
        return ACL_SUCCESS;
    }
    uint64_t removedPhyId = winRes.value->phy_mem_id;
    uint64_t localStart = winRes.value->start_ptr;
    if (!sim::runtime::Db::Delete<sim::runtime::VirtualMemBlock>(
             HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::id, winRes.value->id))
             .ok()) {
        return ACL_ERROR_INTERNAL_ERROR;
    }
    HCCL_VM_INFO(
        "unmap symmetric window, symVa:0x{:x}, localStart:0x{:x}, phyMemId:{:d}", symVa, localStart, removedPhyId);
    return ACL_SUCCESS;
}

// 返回 ptr 所属分配（VA 预留/映射块）的用户侧基址与整段大小。
// 对称内存注册路径（GetMemoryInfo）据此做范围与粒度校验；未命中返回 INVALID_PARAM，
// 与真实 runtime 对未注册地址的行为一致。
aclError aclrtMemGetAddressRange(void* ptr, void** pbase, size_t* psize)
{
    if (pbase == nullptr || psize == nullptr) {
        HCCL_VM_ERROR("null output param, ptr:{:p}", ptr);
        return ACL_ERROR_INVALID_PARAM;
    }
    *pbase = nullptr;
    *psize = 0;

    UserVaRange range{};
    uint64_t userVa = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr));
    if (!FindActiveVirMemByUserVa(userVa, range)) {
        HCCL_VM_ERROR("get allocation range failed, ptr:0x{:x}", userVa);
        return ACL_ERROR_INVALID_PARAM;
    }

    *pbase = reinterpret_cast<void*>(static_cast<uintptr_t>(range.baseVa));
    *psize = range.size;
    HCCL_VM_INFO("get address range, ptr:{:p}, base:{:p}, size:{:d}", ptr, *pbase, *psize);
    return ACL_SUCCESS;
}

// 从用户侧 VA 反查其当前映射的物理内存 handle（仿真中即 PhyMemBlock id）。
// phyMemId 为 0 表示尚未映射或已 Unmap，按真实语义返回失败。
aclError aclrtMemRetainAllocationHandle(void* virPtr, aclrtDrvMemHandle* handle)
{
    if (handle == nullptr) {
        HCCL_VM_ERROR("null handle output, ptr:{:p}", virPtr);
        return ACL_ERROR_INVALID_PARAM;
    }
    *handle = nullptr;

    UserVaRange range{};
    uint64_t userVa = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(virPtr));
    if (!FindActiveVirMemByUserVa(userVa, range) || range.phyMemId == 0) {
        HCCL_VM_ERROR("retain allocation handle failed, ptr:{:p}", virPtr);
        return ACL_ERROR_INVALID_PARAM;
    }

    *handle = reinterpret_cast<aclrtDrvMemHandle>(static_cast<uintptr_t>(range.phyMemId));
    HCCL_VM_INFO("retain allocation handle, ptr:{:p}, phyMemId:{:d}", virPtr, range.phyMemId);
    return ACL_SUCCESS;
}

aclError aclrtMemExportToShareableHandle(
    aclrtDrvMemHandle handle, aclrtMemHandleType handleType, uint64_t flags, uint64_t* shareableHandle)
{
    (void)handleType;
    (void)flags;
    uint64_t phyMemId = (uint64_t)(uintptr_t)handle;

    auto phyMemRes = sim::runtime::Db::GetById<sim::runtime::PhyMemBlock>(phyMemId);
    if (!phyMemRes.ok()) {
        HCCL_VM_ERROR("cannot find phy Mem offset: {:d}", phyMemId);
        return ACL_ERROR_INVALID_PARAM;
    }

    *shareableHandle = phyMemRes.value->id;
    return ACL_SUCCESS;
}

aclError aclrtDeviceGetBareTgid(int32_t* pid)
{
    auto serverId = sim::runtime::GetCurServerId();
    if (serverId == 0) {
        return ACL_ERROR_INVALID_PARAM;
    }
    sim::runtime::Runner runner;
    if (!sim::runtime::GetCurrRunnerTls(serverId, runner)) {
        return ACL_ERROR_INVALID_PARAM;
    }
    *pid = runner.pid;
    return ACL_SUCCESS;
}

aclError aclrtMemSetPidToShareableHandle(uint64_t shareableHandle, int32_t* pid, size_t pidNum)
{
    auto serverId = sim::runtime::GetCurServerId();
    if (serverId == 0) {
        return ACL_ERROR_INVALID_PARAM;
    }
    sim::runtime::Runner runner;
    if (!sim::runtime::GetCurrRunnerTls(serverId, runner)) {
        return ACL_ERROR_INVALID_PARAM;
    }
    for (size_t i = 0; i < pidNum; i++) {
        sim::runtime::FdMemWhiteList tmp{};
        tmp.name_or_key = shareableHandle;
        tmp.pid = pid[i];
        tmp.create_pid = runner.pid;
        tmp.id = 0;
        if (!sim::runtime::Db::Add<sim::runtime::FdMemWhiteList>(tmp).ok()) {
            return ACL_ERROR_INTERNAL_ERROR;
        }
    }

    return ACL_SUCCESS;
}

aclError aclrtMemImportFromShareableHandle(uint64_t shareableHandle, int32_t deviceId, aclrtDrvMemHandle* handle)
{
    (void)deviceId;
    uint64_t phyMemId = shareableHandle;

    auto phyMemRes = sim::runtime::Db::GetById<sim::runtime::PhyMemBlock>(phyMemId);
    if (!phyMemRes.ok()) {
        HCCL_VM_ERROR("cannot find phy Mem offset: {:d}", phyMemId);
        return ACL_ERROR_INVALID_PARAM;
    }

    *handle = (aclrtDrvMemHandle)phyMemId;
    return ACL_SUCCESS;
}

// ==== V2 Fabric Handle 接口族（对称内存注册路径使用） ====
// 仿真约定: aclrtMemFabricHandle 为 128 字节不透明 blob, 前 8 字节承载 PhyMemBlock
// 主键（与 V1 接口 shareableHandle=phyMemId 的编码一致），跨 rank 经 DB 共享行还原。
// 从物理内存句柄导出 fabric shareable handle（对端 Import 后可跨进程映射）。
aclError aclrtMemExportToShareableHandleV2(
    aclrtDrvMemHandle handle, uint64_t flags, aclrtMemSharedHandleType shareType, void* shareableHandle)
{
    (void)flags;
    (void)shareType;
    if (shareableHandle == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    uint64_t phyMemId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(handle));
    auto phyMemRes = sim::runtime::Db::GetById<sim::runtime::PhyMemBlock>(phyMemId);
    if (!phyMemRes.ok()) {
        HCCL_VM_ERROR("cannot find phy Mem offset: {:d}", phyMemId);
        return ACL_ERROR_INVALID_PARAM;
    }

    aclrtMemFabricHandle fabric{};
    std::memcpy(fabric.data, &phyMemId, sizeof(phyMemId));
    *static_cast<aclrtMemFabricHandle*>(shareableHandle) = fabric;
    HCCL_VM_INFO("export fabric handle, phyMemId:{:d}", phyMemId);
    return ACL_SUCCESS;
}

// 将 fabric shareable handle 授权给指定 pid 白名单（GrantLocalMemory 语义）。
aclError aclrtMemSetPidToShareableHandleV2(
    void* shareableHandle, aclrtMemSharedHandleType shareType, int32_t* pid, size_t pidNum)
{
    (void)shareType;
    if (shareableHandle == nullptr || pid == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    auto serverId = sim::runtime::GetCurServerId();
    if (serverId == 0) {
        return ACL_ERROR_INVALID_PARAM;
    }
    sim::runtime::Runner runner;
    if (!sim::runtime::GetCurrRunnerTls(serverId, runner)) {
        return ACL_ERROR_INVALID_PARAM;
    }
    uint64_t phyMemId = 0;
    std::memcpy(&phyMemId, static_cast<const aclrtMemFabricHandle*>(shareableHandle)->data, sizeof(phyMemId));
    auto phyMemRes = sim::runtime::Db::GetById<sim::runtime::PhyMemBlock>(phyMemId);
    if (!phyMemRes.ok()) {
        HCCL_VM_ERROR("cannot find phy Mem offset: {:d}", phyMemId);
        return ACL_ERROR_INVALID_PARAM;
    }

    for (size_t i = 0; i < pidNum; i++) {
        sim::runtime::FdMemWhiteList tmp{};
        tmp.name_or_key = phyMemId;
        tmp.pid = pid[i];
        tmp.create_pid = runner.pid;
        tmp.id = 0;
        if (!sim::runtime::Db::Add<sim::runtime::FdMemWhiteList>(tmp).ok()) {
            return ACL_ERROR_INTERNAL_ERROR;
        }
    }
    return ACL_SUCCESS;
}

// 从对端导出的 fabric shareable handle 导入为本进程物理内存句柄。
aclError aclrtMemImportFromShareableHandleV2(
    void* shareableHandle, aclrtMemSharedHandleType shareType, uint64_t flags, aclrtDrvMemHandle* handle)
{
    (void)shareType;
    (void)flags;
    if (shareableHandle == nullptr || handle == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    uint64_t phyMemId = 0;
    std::memcpy(&phyMemId, static_cast<const aclrtMemFabricHandle*>(shareableHandle)->data, sizeof(phyMemId));
    auto phyMemRes = sim::runtime::Db::GetById<sim::runtime::PhyMemBlock>(phyMemId);
    if (!phyMemRes.ok()) {
        HCCL_VM_ERROR("cannot find phy Mem offset: {:d}", phyMemId);
        return ACL_ERROR_INVALID_PARAM;
    }

    // 透传物理块 id；属主/导入归属在 FreePhysical、对称窗口 MapMem 处按共享 DB 的
    // PhyMemBlock.device_id 实时判定，自导入天然判为属主，无需进程内注册。
    *handle = (aclrtDrvMemHandle)phyMemId;
    HCCL_VM_INFO("import fabric handle, phyMemId:{:d}", phyMemId);
    return ACL_SUCCESS;
}

aclError
aclrtMemGetAllocationGranularity(aclrtPhysicalMemProp* prop, aclrtMemGranularityOptions option, size_t* granularity)
{
    (void)prop;
    (void)option;
    // 真实 runtime RECOMMENDED 粒度为 2MiB。hcomm UB RegisterWindow 的 GetMemoryInfo
    // 据此校验 GetAddressRange 返回 size % granularity == 0；返回 1 会让对称内存
    // 注册路径的物理块映射/交换语义与真实设备不一致，这里返回真实 2MiB。
    *granularity = static_cast<size_t>(kSymAllocGranularity);
    return ACL_SUCCESS;
}

aclError aclrtCmoAsync(void* src, size_t size, aclrtCmoType cmoType, aclrtStream stream)
{
    (void)src;
    (void)size;
    (void)cmoType;
    (void)stream;
    return ACL_SUCCESS;
}

aclError aclrtCmoAsyncWithBarrier(void* src, size_t size, aclrtCmoType cmoType, uint32_t barrierId, aclrtStream stream)
{
    (void)src;
    (void)size;
    (void)cmoType;
    (void)barrierId;
    (void)stream;
    return ACL_SUCCESS;
}

aclError aclrtCmoWaitBarrier(aclrtBarrierTaskInfo* taskInfo, aclrtStream stream, uint32_t flag)
{
    (void)taskInfo;
    (void)stream;
    (void)flag;
    return ACL_SUCCESS;
}

aclError aclrtPointerGetAttributes(const void* ptr, aclrtPtrAttributes* attributes)
{
    uint64_t devPtr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr));
    auto virMemRes = sim::runtime::Db::GetOneByPred<sim::runtime::VirtualMemBlock>(HcclSim::Storage::And(
        HcclSim::Storage::Le(&sim::runtime::VirtualMemBlock::dev_mapped_ptr, devPtr),
        HcclSim::Storage::Gt(&sim::runtime::VirtualMemBlock::dev_mapped_end_ptr, devPtr),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::src_type, (uint8_t)sim::runtime::VIR_MEM_TYPE_DEV)));
    if (!virMemRes.ok()) {
        HCCL_VM_ERROR("can not find this buff offset ptr: 0x{:x}", devPtr);
        return ACL_ERROR_INVALID_PARAM;
    }

    const auto& virMem = *virMemRes;
    auto phyMemId = virMem.phy_mem_id;
    auto phyMemRes = sim::runtime::Db::GetById<sim::runtime::PhyMemBlock>(phyMemId);
    if (!phyMemRes.ok()) {
        HCCL_VM_ERROR("cannot find phy Mem offset: {:d}", phyMemId);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto devRes = sim::runtime::Db::GetById<sim::runtime::Device>(phyMemRes.value->device_id);
    if (!devRes.ok()) {
        HCCL_VM_ERROR("cannot find phy Mem device: {:d}", phyMemRes.value->device_id);
        return ACL_ERROR_INVALID_PARAM;
    }

    attributes->location.id = devRes.value->logic_id;
    attributes->location.type = (aclrtMemLocationType)(virMem.src_type);
    attributes->pageSize = 0;
    return ACL_SUCCESS;
}

aclError aclrtHostRegister(void* ptr, uint64_t size, aclrtHostRegisterType type, void** devPtr)
{
    (void)size;
    (void)type;
    uint64_t startPtr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr));
    auto virMemRes = sim::runtime::Db::GetOneByPred<sim::runtime::VirtualMemBlock>(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::start_ptr, startPtr),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::src_type, (uint8_t)sim::runtime::VIR_MEM_TYPE_HOST)));
    if (!virMemRes.ok()) {
        HCCL_VM_ERROR("can not find this buff offset ptr: 0x{:x}", startPtr);
        return ACL_ERROR_INVALID_PARAM;
    }
    *devPtr = ptr;
    return ACL_SUCCESS;
}

aclError aclrtHostUnregister(void* ptr)
{
    uint64_t startPtr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr));
    auto virMemRes = sim::runtime::Db::GetOneByPred<sim::runtime::VirtualMemBlock>(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::start_ptr, startPtr),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::src_type, (uint8_t)sim::runtime::VIR_MEM_TYPE_HOST)));
    if (!virMemRes.ok()) {
        HCCL_VM_ERROR("can not find this buff offset ptr: 0x{:x}", startPtr);
        return ACL_ERROR_INVALID_PARAM;
    }
    return ACL_SUCCESS;
}

aclError aclrtValueWrite(void* devAddr, uint64_t value, uint32_t flag, aclrtStream stream)
{
    (void)devAddr;
    (void)value;
    (void)flag;
    (void)stream;
    return ACL_SUCCESS;
}

aclError aclrtValueWait(void* devAddr, uint64_t value, uint32_t flag, aclrtStream stream)
{
    (void)devAddr;
    (void)value;
    (void)flag;
    (void)stream;
    return ACL_SUCCESS;
}

aclError aclrtIpcMemGetExportKey(void* devPtr, size_t size, char* key, size_t len, uint64_t flags)
{
    (void)flags;
    if (devPtr == nullptr || key == nullptr || len < sizeof(uint64_t)) {
        return ACL_ERROR_INVALID_PARAM;
    }
    uint64_t deviceKey = sim::runtime::GetCurrDeviceKey();

    // aclrtMalloc exposes dev_mapped_ptr to callers when a device process is
    // present.  IPC must use that same address domain; start_ptr is only the
    // simulator's per-rank virtual address and is not accepted by memcpy,
    // memset, or the other public memory entry points.
    uint64_t devMappedPtr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(devPtr));
    auto virMemRes = sim::runtime::Db::GetOneByPred<sim::runtime::VirtualMemBlock>(HcclSim::Storage::And(
        HcclSim::Storage::Le(&sim::runtime::VirtualMemBlock::dev_mapped_ptr, devMappedPtr),
        HcclSim::Storage::Gt(&sim::runtime::VirtualMemBlock::dev_mapped_end_ptr, devMappedPtr),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::src_type, (uint8_t)sim::runtime::VIR_MEM_TYPE_DEV),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::device_id, deviceKey)));
    if (!virMemRes.ok()) {
        HCCL_VM_ERROR("can not find dev mapped buffer ptr: 0x{:x}", devMappedPtr);
        return ACL_ERROR_INVALID_PARAM;
    }
    uint64_t offset = devMappedPtr - virMemRes->dev_mapped_ptr;
    if (size > virMemRes->size - offset) {
        HCCL_VM_ERROR(
            "ipc export range overflow, ptr:0x{:x}, offset:{}, size:{}, memSize:{}", devMappedPtr, offset, size,
            virMemRes->size);
        return ACL_ERROR_INVALID_PARAM;
    }
    const auto& virMem = *virMemRes;
    auto phyMemId = virMem.phy_mem_id;
    auto phyMemRes = sim::runtime::Db::GetById<sim::runtime::PhyMemBlock>(phyMemId);
    if (!phyMemRes.ok()) {
        HCCL_VM_ERROR("cannot find phy Mem offset: {:d}", phyMemId);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto serverId = sim::runtime::GetCurServerId();
    if (serverId == 0) {
        return ACL_ERROR_INVALID_PARAM;
    }
    sim::runtime::Runner runner;
    if (!sim::runtime::GetCurrRunnerTls(serverId, runner)) {
        return ACL_ERROR_INVALID_PARAM;
    }
    sim::runtime::IpcMemRecord memRecord{};
    memRecord.vir_mem_id = virMem.id;
    memRecord.offset = offset;
    memRecord.create_pid = runner.pid;
    memRecord.id = 0;
    auto inserted = sim::runtime::Db::Add<sim::runtime::IpcMemRecord>(memRecord);
    if (!inserted.ok() || !inserted.value.has_value()) {
        return ACL_ERROR_INVALID_PARAM;
    }
    auto recordIdx = inserted.value->value;

    memcpy(key, &recordIdx, sizeof(recordIdx));

    return ACL_SUCCESS;
}

aclError aclrtIpcMemSetImportPid(const char* key, int32_t* pid, size_t num)
{
    if (key == nullptr || (pid == nullptr && num != 0)) {
        return ACL_ERROR_INVALID_PARAM;
    }
    auto serverId = sim::runtime::GetCurServerId();
    if (serverId == 0) {
        return ACL_ERROR_INVALID_PARAM;
    }
    sim::runtime::Runner runner;
    if (!sim::runtime::GetCurrRunnerTls(serverId, runner)) {
        return ACL_ERROR_INVALID_PARAM;
    }
    uint64_t ipcRecordIdx = 0;
    memcpy(&ipcRecordIdx, key, sizeof(ipcRecordIdx));
    for (int32_t i = 0; i < num; i++) {
        sim::runtime::IpcMemWhiteList tmp{};
        tmp.name_or_key = ipcRecordIdx;
        tmp.pid = pid[i];
        tmp.create_pid = runner.pid;
        tmp.id = 0;
        if (!sim::runtime::Db::Add<sim::runtime::IpcMemWhiteList>(tmp).ok()) {
            return ACL_ERROR_INTERNAL_ERROR;
        }
    }
    return ACL_SUCCESS;
}

// IPC 导入映射落库（方案A 的事实采集点）。
// 为什么必须在这里记：A5 的 P2P(PCIe) 传输把对端 buffer 地址直接写进 SDMA SQE
// （BuildA5SqeSdmaCopy 只填 srcAddr/dstAddr/lengthMove/opcode/partId），SQE 内既无 rmtEid
// 也无 deviceId；而该地址正是本接口返回给 hcomm 的值（RemoteIpcRmaBuffer::OpenIpcMemory →
// HrtIpcOpenMemory → aclrtIpcMemImportByKey，addr = ipcPtr + ipcOffset）。只有导入这一刻
// 同时握有"导入方设备 + 返回地址区间 + 属主设备"，过后无从还原，故在此落库供 device 侧反查。
// A3(910) 的 hrtIpcOpenMemory 走同一接口，映射同样成立（A3 任务口径走 level1 北向，不消费本表）。
static void RecordIpcMemImport(
    uint64_t ipcRecordIdx, const sim::runtime::VirtualMemBlock& virMem, uint64_t offset, uint64_t addrBase)
{
    // 导入方设备与 device 进程 GetCurDeviceKey() 同口径（均为 Device 表主键），device 侧据此定位本方
    const uint64_t importerDeviceId = sim::runtime::GetCurrDeviceId();
    if (importerDeviceId == 0) {
        HCCL_VM_WARN("skip ipc import record, current device unresolved, ipc record: {:d}", ipcRecordIdx);
        return;
    }

    // 物理属主：任务字段 srcDeviceId/dstDeviceId 用物理属主口径（与 ResolveTaskAddress 步骤3 一致）；
    // aclrtMapMem 允许把其他属主的物理块映射到本设备虚拟块，故不能用 VMB.device_id 代替
    uint64_t ownerPhyDeviceId = virMem.device_id;
    auto phyMemRes = sim::runtime::Db::GetById<sim::runtime::PhyMemBlock>(virMem.phy_mem_id);
    if (phyMemRes.ok()) {
        ownerPhyDeviceId = phyMemRes->device_id;
    } else {
        HCCL_VM_WARN(
            "phy mem block not found, fall back to vir mem owner, phy_mem_id: {:d}, vir_mem_id: {:d}",
            virMem.phy_mem_id, virMem.id);
    }

    // 同一 (导入方设备, 地址基址) 只保留首行：hcomm 可能对同一导出块反复 open，避免表膨胀
    auto exists = sim::runtime::Db::GetOneByPred<sim::runtime::IpcMemImport>(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&sim::runtime::IpcMemImport::importer_device_id, importerDeviceId),
        HcclSim::Storage::Eq(&sim::runtime::IpcMemImport::addr_base, addrBase)));
    if (exists.ok() && exists.value.has_value()) {
        return;
    }

    sim::runtime::IpcMemImport importRecord{};
    importRecord.ipc_record_id = ipcRecordIdx;
    importRecord.importer_device_id = importerDeviceId;
    importRecord.importer_pid = static_cast<uint32_t>(getpid());
    importRecord.addr_base = addrBase;
    importRecord.addr_end = addrBase + (virMem.size - offset); // 可导入长度即区间长度，上界不含
    importRecord.start_base = virMem.start_ptr + offset;       // 属主按卡编址基址，用于换算 virtualAddress
    importRecord.owner_device_id = virMem.device_id;
    importRecord.owner_phy_device_id = ownerPhyDeviceId;
    importRecord.id = 0;
    auto inserted = sim::runtime::Db::Add<sim::runtime::IpcMemImport>(importRecord);
    if (!inserted.ok() || !inserted.value.has_value()) {
        HCCL_VM_ERROR(
            "failed to record ipc mem import, ipc record: {:d}, importer device: {:d}, addr: 0x{:x}", ipcRecordIdx,
            importerDeviceId, addrBase);
        return;
    }
    HCCL_VM_INFO(
        "record ipc mem import: id={:d}, ipcRecord={:d}, importerDevice={:d}, addr=[0x{:x},0x{:x}), "
        "startBase=0x{:x}, ownerDevice={:d}, ownerPhyDevice={:d}",
        inserted.value->value, ipcRecordIdx, importerDeviceId, importRecord.addr_base, importRecord.addr_end,
        importRecord.start_base, importRecord.owner_device_id, importRecord.owner_phy_device_id);
}

aclError aclrtIpcMemImportByKey(void** devPtr, const char* key, uint64_t flags)
{
    (void)flags;
    if (devPtr == nullptr || key == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    uint64_t ipcRecordIdx = 0;
    memcpy(&ipcRecordIdx, key, sizeof(ipcRecordIdx));
    auto recordRes = sim::runtime::Db::GetById<sim::runtime::IpcMemRecord>(ipcRecordIdx);
    if (!recordRes.ok()) {
        HCCL_VM_ERROR("cannot find ipc record: {:d}", ipcRecordIdx);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto virMemRes = sim::runtime::Db::GetById<sim::runtime::VirtualMemBlock>(recordRes.value->vir_mem_id);
    if (!virMemRes.ok()) {
        HCCL_VM_ERROR("cannot find vir mem: {:d}", recordRes.value->vir_mem_id);
        return ACL_ERROR_INVALID_PARAM;
    }
    if (recordRes->offset >= virMemRes->size) {
        HCCL_VM_ERROR(
            "ipc mem offset out of range, ipc record: {:d}, vir mem: {:d}, offset: {:d}, size: {:d}", ipcRecordIdx,
            recordRes->vir_mem_id, recordRes->offset, virMemRes->size);
        return ACL_ERROR_INVALID_PARAM;
    }
    const uint64_t addrBase = virMemRes->dev_mapped_ptr + recordRes->offset;
    *devPtr = reinterpret_cast<void*>(addrBase);
    // 返回地址即将被 hcomm 当作对端地址写进 SDMA SQE，此处登记"地址区间 → 属主"供 device 侧反查
    RecordIpcMemImport(ipcRecordIdx, *virMemRes, recordRes->offset, addrBase);

    return ACL_SUCCESS;
}

aclError aclrtIpcMemClose(const char* key)
{
    if (key == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    uint64_t ipcRecordIdx = 0;
    memcpy(&ipcRecordIdx, key, sizeof(ipcRecordIdx));
    auto recordRes = sim::runtime::Db::GetById<sim::runtime::IpcMemRecord>(ipcRecordIdx);
    if (!recordRes.ok()) {
        HCCL_VM_ERROR("cannot find ipc record: {:d}", ipcRecordIdx);
        return ACL_ERROR_INVALID_PARAM;
    }

    auto serverId = sim::runtime::GetCurServerId();
    if (serverId == 0) {
        return ACL_ERROR_INVALID_PARAM;
    }
    sim::runtime::Runner runner;
    if (!sim::runtime::GetCurrRunnerTls(serverId, runner)) {
        return ACL_ERROR_INVALID_PARAM;
    }
    if (runner.pid == recordRes.value->create_pid) {
        if (!sim::runtime::Db::Delete<sim::runtime::IpcMemRecord>(
                 HcclSim::Storage::Eq(&sim::runtime::IpcMemRecord::id, recordRes.value->id))
                 .ok()) {
            return ACL_ERROR_INTERNAL_ERROR;
        }
        // 级联清理导入映射：导出记录已失效，残留行会让后续复用的同值地址命中过期属主
        (void)sim::runtime::Db::Delete<sim::runtime::IpcMemImport>(
            HcclSim::Storage::Eq(&sim::runtime::IpcMemImport::ipc_record_id, recordRes.value->id));
    }

    return ACL_SUCCESS;
}

aclError aclrtGetMemInfo(aclrtMemAttr attr, size_t* free, size_t* total)
{
    if (free == nullptr || total == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    // 仿真环境无真实显存，桩按HBM大页设备返回模拟容量（total=64GB、free=50GB），
    // 使对称内存预留前的容量预检（默认stride=16GB <= total）可以通过。
    constexpr size_t SIM_BYTES_PER_GB = 1024ULL * 1024ULL * 1024ULL;
    constexpr size_t SIM_TOTAL_HBM_SIZE = 64ULL * SIM_BYTES_PER_GB;
    constexpr size_t SIM_FREE_HBM_SIZE = 50ULL * SIM_BYTES_PER_GB;
    *total = SIM_TOTAL_HBM_SIZE;
    *free = SIM_FREE_HBM_SIZE;
    HCCL_VM_INFO("get mem info attr:{:d}, free:{:d}, total:{:d}", static_cast<int>(attr), *free, *total);
    return ACL_SUCCESS;
}

aclError aclrtAllocatorRegister(aclrtStream stream, aclrtAllocatorDesc allocatorDesc)
{
    (void)stream;
    (void)allocatorDesc;
    return ACL_SUCCESS;
}

aclError aclrtAllocatorGetByStream(
    aclrtStream stream, aclrtAllocatorDesc* allocatorDesc, aclrtAllocator* allocator,
    aclrtAllocatorAllocFunc* allocFunc, aclrtAllocatorFreeFunc* freeFunc,
    aclrtAllocatorAllocAdviseFunc* allocAdviseFunc, aclrtAllocatorGetAddrFromBlockFunc* getAddrFromBlockFunc)
{
    (void)stream;
    (void)allocatorDesc;
    (void)allocator;
    (void)allocFunc;
    (void)freeFunc;
    (void)allocAdviseFunc;
    (void)getAddrFromBlockFunc;
    return ACL_SUCCESS;
}

aclError aclrtAllocatorUnregister(aclrtStream stream)
{
    (void)stream;
    return ACL_SUCCESS;
}

aclError aclrtMemcpyAsyncWithOffsetImpl(
    void** dst, size_t destMax, uint64_t dstDataOffset, const void** src, size_t count, size_t srcDataOffset,
    aclrtMemcpyKind kind, aclrtStream stream)
{
    (void)dst;
    (void)destMax;
    (void)dstDataOffset;
    (void)src;
    (void)count;
    (void)srcDataOffset;
    (void)kind;
    (void)stream;
    return ACL_SUCCESS;
}

aclError aclrtIpcMemSetAttr(const char* key, aclrtIpcMemAttrType type, uint64_t attr)
{
    (void)key;
    (void)type;
    (void)attr;
    return ACL_SUCCESS;
}

aclError aclrtIpcMemImportPidInterServer(const char* name, aclrtServerPid* serverPids, size_t num)
{
    const aclrtServerPid& rtServerPid = *serverPids;
    return aclrtIpcMemSetImportPid(name, rtServerPid.pid, rtServerPid.num);
}

rtError_t rtMalloc(void** devPtr, uint64_t size, rtMemType_t type, const uint16_t moduleId)
{
    (void)type;
    (void)moduleId;
    int ret = aclrtMalloc(devPtr, size, aclrtMemMallocPolicy::ACL_MEM_MALLOC_HUGE_FIRST);
    HCCL_VM_INFO("enter into stub rtMalloc.... addr:{:p}", *devPtr);
    return ret;
}

aclError rtFree(void* devPtr)
{
    (void)devPtr;
    HCCL_VM_ERROR("enter into rtFree not support");
    return ACL_ERROR_INTERNAL_ERROR;
}

std::map<uint8_t, HcclDataType> rtDataType2CheckerDataType
    = {{aclDataType::ACL_FLOAT, HcclDataType::HCCL_DATA_TYPE_FP32},
       {aclDataType::ACL_FLOAT16, HcclDataType::HCCL_DATA_TYPE_FP16},
       {aclDataType::ACL_INT8, HcclDataType::HCCL_DATA_TYPE_INT8},
       {aclDataType::ACL_INT32, HcclDataType::HCCL_DATA_TYPE_INT32},
       {aclDataType::ACL_UINT8, HcclDataType::HCCL_DATA_TYPE_UINT8},
       {aclDataType::ACL_INT16, HcclDataType::HCCL_DATA_TYPE_INT16},
       {aclDataType::ACL_UINT16, HcclDataType::HCCL_DATA_TYPE_UINT16},
       {aclDataType::ACL_UINT32, HcclDataType::HCCL_DATA_TYPE_UINT32},
       {aclDataType::ACL_INT64, HcclDataType::HCCL_DATA_TYPE_INT64},
       {aclDataType::ACL_BF16, HcclDataType::HCCL_DATA_TYPE_BFP16}};

aclError aclrtReduceAsync(
    void* dst, const void* src, uint64_t count, aclrtReduceKind kind, aclDataType type, aclrtStream stream,
    void* reserve)
{
    (void)reserve;
    uint64_t srcOffset = 0;
    uint64_t dstOffset = 0;
    uint64_t srcCtxId = 0;
    uint64_t dstCtxId = 0;

    auto iter = rtDataType2CheckerDataType.find(type);
    if (iter == rtDataType2CheckerDataType.end()) {
        HCCL_VM_ERROR("not support data type: {}", static_cast<uint32_t>(type));
        return ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
    }

    auto dataType = iter->second;
    uint32_t dataSize = 0;
    if (sim::GetDataTypeSize(dataType, dataSize) != HcclResult::HCCL_SUCCESS) {
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }

    uint64_t streamId = reinterpret_cast<uint64_t>(stream);

    uint32_t deviceId = (uint32_t)sim::runtime::GetCurrDeviceId();

    HcclTaskMetaData taskMetaData;
    taskMetaData.taskType = HccLTaskMetaType::REDUCE;
    taskMetaData.profInfo = HcclTaskProfInfo(
        ProfTaskParamType::TASK_REDUCE_INLINE, ProfLinkType::LINK_ONCHIP,
        ProfTransportType::TRANSPORT_LOCAL); // hcomm LocalReduce
    taskMetaData.deviceId = deviceId;
    taskMetaData.commId = sim::runtime::g_cur_comm_key;
    if (taskMetaData.commId == 0
        || !sim::runtime::GetCommRankByDeviceId(taskMetaData.commId, deviceId, taskMetaData.rankId)) {
        HCCL_VM_ERROR("reduce task has no valid communicator, deviceId={}", deviceId);
        return ACL_ERROR_INVALID_PARAM;
    }
    taskMetaData.streamId = streamId;
    // 评估模式: deviceId字段承载通信域rankId; 主线保持真实deviceId供checker/runner消费
    if (sim::IsEstimateMode()) {
        taskMetaData.taskData.reduce.srcDeviceId
            = ResolveRankIdByDevice(taskMetaData.commId, GetDeviceIdByVirAddr(src));
        taskMetaData.taskData.reduce.dstDeviceId
            = ResolveRankIdByDevice(taskMetaData.commId, GetDeviceIdByVirAddr(dst));
    } else {
        taskMetaData.taskData.reduce.srcDeviceId = GetDeviceIdByVirAddr(src);
        taskMetaData.taskData.reduce.dstDeviceId = GetDeviceIdByVirAddr(dst);
    }
    taskMetaData.taskData.reduce.srcOffset = sim::GetVirPtrByDevPtr(reinterpret_cast<uint64_t>(src));
    taskMetaData.taskData.reduce.dstOffset = sim::GetVirPtrByDevPtr(reinterpret_cast<uint64_t>(dst));
    taskMetaData.taskData.reduce.dataType = static_cast<uint8_t>(dataType);
    taskMetaData.taskData.reduce.dataCount = count;
    taskMetaData.taskData.reduce.reduceOp = static_cast<uint8_t>(kind);

    uint32_t index{0};
    HCCL_VM_DEBUG("Get reduce task, streamId={:d}", streamId);
    auto ret = InsertTaskToCollection(&taskMetaData, &index);
    if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("InsertTaskToCollection fail");
        return ACL_ERROR_INTERNAL_ERROR;
    }

    return ACL_SUCCESS;
}

#ifdef __cplusplus
}
#endif // __cplusplus
