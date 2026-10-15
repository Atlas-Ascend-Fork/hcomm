/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_device_pub.h"

#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <execinfo.h>
#include <fcntl.h>
#include <link.h>
#include <algorithm>
#include <map>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "sim_log.h"
#include "sim_ip_address.h"
#include "sim_capacity_limits.h"
#include "store_sim_memory_manager.h"
#include "storage/table_access.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "sim_pipe_io.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

uint32_t g_rankId = 0;
uint32_t g_deviceKey = 0;

char g_crash_file_name[64] = "crash_default.log";

HcclVmResult SetCurRankId(uint32_t rankId)
{
    HCCL_VM_INFO("SetCurRankId, old rankId: {}, new rankId: {}", g_rankId, rankId);
    g_rankId = rankId;
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult GetCurRankId(uint32_t* rankId)
{
    *rankId = g_rankId;
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

void SetCurDeviceKey(uint32_t deviceKey)
{
    HCCL_VM_INFO("SetCurDeviceKey, old deviceKey: {}, new deviceKey: {}", g_deviceKey, deviceKey);
    g_deviceKey = deviceKey;
}

uint32_t GetCurDeviceKey() { return g_deviceKey; }

uint8_t sqBuffer[HCCL_SQE_SIZE * HCCL_SQE_MAX_CNT];
HcclVmResult GetSqBufferAddr(uint8_t** sqBuff)
{
    *sqBuff = sqBuffer;
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

std::map<uint32_t, uint32_t> jettyId2PiValMap;
HcclVmResult GetPiValByJettyId(uint32_t jettyId, uint32_t* piValue)
{
    *piValue = jettyId2PiValMap[jettyId];
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult UpdatePiValByJettyId(uint32_t jettyId, uint32_t piValue)
{
    jettyId2PiValMap[jettyId] = piValue;
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

std::map<uint32_t, uint32_t> sqTailMap;
uint32_t GetSqTail(uint32_t sqId) { return sqTailMap[sqId]; }

void UpdateSqTail(uint32_t sqId, uint32_t newTail) { sqTailMap[sqId] = newTail; }

// 设备作用域 dev_mapped_ptr 区间包含查找的唯一谓词：device_id 等值 + src_type=DEV +
// dev_mapped_ptr<=addr<dev_mapped_end_ptr（全部可下推，命中 IntervalIndex(device_id,
// src_type, dev_mapped_ptr, dev_mapped_end_ptr)）。本地/远端地址翻译、宿主映射地址
// 解析与 ResolveTaskAddress 共用；文件内不出现第二份相同谓词。
namespace {
    struct CachedVirMemBlock {
        uint64_t devMappedPtr;
        uint64_t devMappedEndPtr;
        uint64_t startPtr;
        uint64_t phyMemId;
    };
    std::mutex g_vmbCacheMtx;
    std::unordered_map<uint64_t, std::vector<CachedVirMemBlock>> g_vmbCache; // deviceId → 按 devMappedPtr 升序区间
    std::unordered_map<uint64_t, uint64_t> g_phyMemOwnerCache;               // phyMemId → 物理属主 device_id

    // 调用方须持有 g_vmbCacheMtx。整设备重载 VMB 区间；查询失败时保留旧缓存。
    void RefreshVmbCacheLocked(uint64_t deviceId)
    {
        auto rows = sim::runtime::Db::GetByPred<sim::runtime::VirtualMemBlock>(HcclSim::Storage::And(
            HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::device_id, deviceId),
            HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::src_type, (uint8_t)sim::runtime::VIR_MEM_TYPE_DEV)));
        if (!rows.ok()) {
            return;
        }
        std::vector<CachedVirMemBlock> blocks;
        blocks.reserve(rows->size());
        for (const auto& r : *rows) {
            blocks.push_back({r.dev_mapped_ptr, r.dev_mapped_end_ptr, r.start_ptr, r.phy_mem_id});
        }
        std::sort(blocks.begin(), blocks.end(), [](const CachedVirMemBlock& a, const CachedVirMemBlock& b) {
            return a.devMappedPtr < b.devMappedPtr;
        });
        g_vmbCache[deviceId] = std::move(blocks);
    }
} // namespace

// 只返回原始查询结果：NOT_FOUND/后端错误由调用方按各自业务语义处置
// （翻译函数记日志返回 0；ResolveTaskAddress 原样传播错误码），helper 不吞错、不打日志。
// 性能：VMB 区间按 deviceId 进程内缓存（分配记录静态，SQE 解析期高频点查）；
// 缓存内未命中地址时回源点查并刷新该设备缓存，覆盖运行期新增映射。
static HcclSim::Storage::DbResult<sim::runtime::VirtualMemBlock>
FindDevMappedVirMemByDevice(uint64_t mappedAddress, uint64_t deviceId)
{
    {
        std::lock_guard<std::mutex> lock(g_vmbCacheMtx);
        auto it = g_vmbCache.find(deviceId);
        if (it == g_vmbCache.end()) {
            RefreshVmbCacheLocked(deviceId);
            it = g_vmbCache.find(deviceId);
        }
        if (it != g_vmbCache.end()) {
            const auto& blocks = it->second;
            auto ub = std::upper_bound(
                blocks.begin(), blocks.end(), mappedAddress, [](uint64_t addr, const CachedVirMemBlock& b) {
                    return addr < b.devMappedPtr;
                });
            if (ub != blocks.begin()) {
                const auto& blk = *std::prev(ub);
                if (mappedAddress < blk.devMappedEndPtr) {
                    sim::runtime::VirtualMemBlock rec{};
                    rec.dev_mapped_ptr = blk.devMappedPtr;
                    rec.dev_mapped_end_ptr = blk.devMappedEndPtr;
                    rec.start_ptr = blk.startPtr;
                    rec.phy_mem_id = blk.phyMemId;
                    rec.device_id = deviceId;
                    rec.src_type = (uint8_t)sim::runtime::VIR_MEM_TYPE_DEV;
                    return {HcclSim::Storage::DbCode::OK, rec, 1, ""};
                }
            }
        }
    }
    // 缓存未命中：回源点查（运行期新增映射场景）；命中则刷新该设备缓存
    auto res = sim::runtime::Db::GetOneByPred<sim::runtime::VirtualMemBlock>(HcclSim::Storage::And(
        HcclSim::Storage::Le(&sim::runtime::VirtualMemBlock::dev_mapped_ptr, mappedAddress),
        HcclSim::Storage::Gt(&sim::runtime::VirtualMemBlock::dev_mapped_end_ptr, mappedAddress),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::src_type, (uint8_t)sim::runtime::VIR_MEM_TYPE_DEV),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::device_id, deviceId)));
    if (res.ok()) {
        std::lock_guard<std::mutex> lock(g_vmbCacheMtx);
        RefreshVmbCacheLocked(deviceId);
    }
    return res;
}

// AICPU构造Task时任务转换，将本端设备地址(本rank对应host进程申请的)转换为虚拟地址
uint64_t TransLocalAddrToVirtual(uint64_t devAddr)
{
    uint64_t deviceKey = GetCurDeviceKey();
    auto virMemRes = FindDevMappedVirMemByDevice(devAddr, deviceKey);
    if (!virMemRes.ok()) {
        // 地址/实际 deviceKey/commRank/错误码/诊断齐全：现场首错（NOT_FOUND 与
        // 后端错误可区分）；stage=VMB 标明查的是 VirtualMemBlock 阶段
        uint32_t commRank = 0;
        (void)GetCurRankId(&commRank);
        HCCL_VM_ERROR(
            "cannot find virMemRes by devAddr[0x{:x}], stage=VMB, deviceKey[{}], commRank[{}], "
            "code:{:d}, diagnostic:{}",
            devAddr, deviceKey, commRank, static_cast<uint32_t>(virMemRes.code), virMemRes.diagnostic);
        return 0;
    }

    const auto& record = *virMemRes;
    uint64_t diff = devAddr - record.dev_mapped_ptr;
    return record.start_ptr + diff;
}

// AICPU构造Task时任务转换，将对端设备地址转换为虚拟地址。
uint64_t TransRemoteAddrToVirtualByDeviceId(uint64_t devAddr, uint32_t deviceId)
{
    auto virMemRes = FindDevMappedVirMemByDevice(devAddr, deviceId);
    if (!virMemRes.ok()) {
        // deviceId 形参即谓词里的 VirtualMemBlock.device_id（deviceKey 口径），如实记录
        uint32_t commRank = 0;
        (void)GetCurRankId(&commRank);
        HCCL_VM_ERROR(
            "cannot find virMemRes by devAddr[0x{:x}], stage=VMB(remote), vmbDeviceId[{}], "
            "commRank[{}], code:{:d}, diagnostic:{}",
            devAddr, deviceId, commRank, static_cast<uint32_t>(virMemRes.code), virMemRes.diagnostic);
        return 0;
    }

    const auto& record = *virMemRes;
    uint64_t diff = devAddr - record.dev_mapped_ptr;
    return record.start_ptr + diff;
}

uint64_t GetDevMapperAddrByDevAddrImpl(uint64_t devAddr, const char* file, int line)
{
    uint64_t deviceKey = GetCurDeviceKey();
    auto virMemRes = FindDevMappedVirMemByDevice(devAddr, deviceKey);
    if (!virMemRes.ok()) {
        uint32_t commRank = 0;
        (void)GetCurRankId(&commRank);
        HCCL_VM_ERROR(
            "cannot find virMemRes by devAddr[0x{:x}], stage=VMB, deviceKey[{}], commRank[{}], "
            "code:{:d}, diagnostic:{}, called from {}:{}",
            devAddr, deviceKey, commRank, static_cast<uint32_t>(virMemRes.code), virMemRes.diagnostic, file, line);
        return 0;
    }

    // 已经映射过的这里直接取值
    const auto& virtualRecord = *virMemRes;
    if (virtualRecord.is_dev_access == 1) {
        HCCL_VM_INFO("device cann access this devAddr[0x{:x}], called from {}:{}", devAddr, file, line);
        return devAddr;
    }

    auto phyMemId = virtualRecord.phy_mem_id;
    auto phyMemRes = sim::runtime::Db::GetById<sim::runtime::PhyMemBlock>(phyMemId);
    if (!phyMemRes.ok()) {
        // stage=PhyMem：VMB 已命中、第二段物理块主键读取失败（与 VMB 阶段区分）
        uint32_t commRank = 0;
        (void)GetCurRankId(&commRank);
        HCCL_VM_ERROR(
            "cannot find phyMemRes by phyMemId[{}], stage=PhyMem, deviceKey[{}], commRank[{}], "
            "code:{:d}, diagnostic:{}, called from {}:{}",
            phyMemId, deviceKey, commRank, static_cast<uint32_t>(phyMemRes.code), phyMemRes.diagnostic, file, line);
        return 0;
    }

    std::string memName(phyMemRes.value->name);
    void* devMappedPtr = sim::MemoryManager::GetInstance().AcquireMemByName(memName.c_str());
    if (devMappedPtr == nullptr) {
        HCCL_VM_ERROR("acquire device shm for memName[{}] failed, called from {}:{}", memName, file, line);
        return 0;
    }

    uint64_t offset = devAddr - virtualRecord.dev_mapped_ptr;
    uint64_t result = reinterpret_cast<uint64_t>(devMappedPtr) + offset;
    HCCL_VM_INFO(
        "devAddr[0x{:x}] offset[{}] devMappedPtr[0x{:x}] memName[{}]", devAddr, offset,
        reinterpret_cast<uint64_t>(devMappedPtr), memName);
    return result;
}

uint32_t GetDeviceIdByDevAddr(uint64_t devAddr)
{
    auto virMemRes = sim::runtime::Db::GetOneByPred<sim::runtime::VirtualMemBlock>(HcclSim::Storage::And(
        HcclSim::Storage::Le(&sim::runtime::VirtualMemBlock::start_ptr, devAddr),
        HcclSim::Storage::Gt(&sim::runtime::VirtualMemBlock::start_end_ptr, devAddr),
        HcclSim::Storage::Eq(&sim::runtime::VirtualMemBlock::src_type, (uint8_t)sim::runtime::VIR_MEM_TYPE_DEV)));
    if (!virMemRes.ok()) {
        // start_ptr 区间谓词（无 device_id 过滤）：地址/错误码/诊断齐全
        HCCL_VM_ERROR(
            "cannot find virMemRes by devAddr[0x{:x}], stage=VMB(startRange), code:{:d}, diagnostic:{}", devAddr,
            static_cast<uint32_t>(virMemRes.code), virMemRes.diagnostic);
        return 0;
    }

    auto phyMemId = virMemRes->phy_mem_id;
    auto phyMemRes = sim::runtime::Db::GetById<sim::runtime::PhyMemBlock>(phyMemId);
    if (!phyMemRes.ok()) {
        HCCL_VM_ERROR(
            "cannot find phyMemRes by phyMemId[{}], stage=PhyMem, code:{:d}, diagnostic:{}", phyMemId,
            static_cast<uint32_t>(phyMemRes.code), phyMemRes.diagnostic);
        return 0;
    };

    return phyMemRes->device_id;
}

HcclSim::Storage::DbResult<TaskAddressResolution> ResolveTaskAddress(uint64_t mappedAddress, uint64_t mappingDeviceId)
{
    using HcclSim::Storage::DbCode;
    // 步骤0：入参防御。Device 主键自增从 1 起，0 是"无当前设备"哨兵
    // （GetCurDeviceKey 失败/未设置时的返回值），不得当作合法映射设备。
    if (mappingDeviceId == 0) {
        return {DbCode::INVALID_ARGUMENT, std::nullopt, 0, "ResolveTaskAddress: mapping device id is not set"};
    }

    // 步骤1：按 (device_id, 区间) 定位 VMB——与本地/远端翻译共用同一谓词，
    // 天然具备单分片条件（启用 PartitionBy(device_id) 后由 S2 定向）。
    auto virMemRes = FindDevMappedVirMemByDevice(mappedAddress, mappingDeviceId);
    if (!virMemRes.ok()) {
        return {
            virMemRes.code, std::nullopt, virMemRes.affectedRows,
            virMemRes.diagnostic.empty() ? "ResolveTaskAddress: no device-mapped VirtualMemBlock contains the address" :
                                           virMemRes.diagnostic};
    }

    // 步骤2：由本次命中行计算虚拟地址；不再按虚拟地址二次查询 VMB。
    // 区间语义已保证 dev_mapped_ptr <= mappedAddress < dev_mapped_end_ptr，
    // diff 不会下溢；start_ptr + diff 显式防上溢。
    const auto& record = *virMemRes;
    const uint64_t diff = mappedAddress - record.dev_mapped_ptr;
    if (record.start_ptr > UINT64_MAX - diff) {
        return {DbCode::INVALID_ARGUMENT, std::nullopt, 0, "ResolveTaskAddress: virtual address computation overflows"};
    }

    // 步骤3：用同一行的 phy_mem_id 做 PhyMemBlock 主键读取（S2 主键路由单 shard），
    // 取物理属主设备。映射设备与物理属主可以不同（aclrtMapMem 允许把其他属主的
    // 物理块映射到当前设备的虚拟块），此读取固定保留、不可用 mappingDeviceId
    // 或 VMB.device_id 替换。
    if (record.phy_mem_id == 0) {
        return {
            DbCode::NOT_FOUND, std::nullopt, 0, "ResolveTaskAddress: virtual mem block has no physical association"};
    }
    // PhyMemBlock 为静态分配记录：phy_mem_id → 物理属主缓存，命中即免一次主键查询
    uint64_t physicalDeviceId = 0;
    {
        std::lock_guard<std::mutex> lock(g_vmbCacheMtx);
        auto it = g_phyMemOwnerCache.find(record.phy_mem_id);
        if (it != g_phyMemOwnerCache.end()) {
            physicalDeviceId = it->second;
        }
    }
    if (physicalDeviceId == 0) {
        auto phyMemRes = sim::runtime::Db::GetById<sim::runtime::PhyMemBlock>(record.phy_mem_id);
        if (!phyMemRes.ok()) {
            return {phyMemRes.code, std::nullopt, phyMemRes.affectedRows, phyMemRes.diagnostic};
        }
        physicalDeviceId = phyMemRes->device_id;
        std::lock_guard<std::mutex> lock(g_vmbCacheMtx);
        g_phyMemOwnerCache.emplace(record.phy_mem_id, physicalDeviceId);
    }
    // 目标字段为 uint32：窄化前检查范围；0 属主是未初始化行，不作为有效任务身份。
    if (physicalDeviceId == 0 || physicalDeviceId > UINT32_MAX) {
        return {
            DbCode::INVALID_ARGUMENT, std::nullopt, 0, "ResolveTaskAddress: physical device id is out of uint32 range"};
    }

    // 步骤4：全部成功才返回结果。
    TaskAddressResolution resolution{};
    resolution.virtualAddress = record.start_ptr + diff;
    resolution.physicalDeviceId = static_cast<uint32_t>(physicalDeviceId);
    return {DbCode::OK, resolution, 0, {}};
}

bool GetDeviceIdByIpAddr(const std::string& ipAddr, uint32_t& deviceId)
{
    auto ret = sim::runtime::Db::GetOneByPred<sim::runtime::EndPoint>(
        HcclSim::Storage::Eq(&sim::runtime::EndPoint::ip_addr, std::string(ipAddr.c_str())));
    if (!ret.ok()) {
        HCCL_VM_ERROR("cannot find endpoint by ipAddr[{}]", ipAddr);
        return false;
    }

    deviceId = ret->device_id;
    return true;
}

void UpdateKfcStatus(uint64_t d2hAddr)
{
    if (d2hAddr == 0) {
        HCCL_VM_ERROR("d2hAddr is 0, cannot update kfc status.");
        return;
    }

    uint8_t devDoneStatus = 3;
    constexpr uint32_t headTailCnt = 66;
    constexpr uint32_t headCntOffset = 4088;
    constexpr uint32_t tailCntOffset = 4092;
    *reinterpret_cast<uint8_t*>(d2hAddr) = devDoneStatus;                // 更新设备状态
    *reinterpret_cast<uint32_t*>(d2hAddr + headCntOffset) = headTailCnt; // 更新headCnt
    *reinterpret_cast<uint32_t*>(d2hAddr + tailCntOffset) = headTailCnt; // 更新tailCnt
    HCCL_VM_INFO("update kfc status success.");
}

static int DumpModuleMap(struct dl_phdr_info* info, size_t size, void* data)
{
    int fd = *(int*)data;
    char buf[512];
    const char* name = info->dlpi_name[0] ? info->dlpi_name : "[main]";

    uint64_t vaddr_min = (uint64_t)-1;
    uint64_t vaddr_max = 0;
    for (int i = 0; i < info->dlpi_phnum; i++) {
        if (info->dlpi_phdr[i].p_type != PT_LOAD) {
            continue;
        }
        uint64_t seg_start = info->dlpi_addr + info->dlpi_phdr[i].p_vaddr;
        uint64_t seg_end = seg_start + info->dlpi_phdr[i].p_memsz;
        if (seg_start < vaddr_min) {
            vaddr_min = seg_start;
        }
        if (seg_end > vaddr_max) {
            vaddr_max = seg_end;
        }
    }

    int len = snprintf(
        buf, sizeof(buf), "  0x%016lx-0x%016lx  base=0x%016lx  %s\n", vaddr_min, vaddr_max, (uint64_t)info->dlpi_addr,
        name);
    if (len > 0) {
        write(fd, buf, len);
    }
    return 0;
}

static void DumpCallSites(int fd, void* const* stack_pointers, int nptrs)
{
    char buf[768];
    int len;

    len = snprintf(
        buf, sizeof(buf),
        "\n---- [Call Sites] (call_site = ret_addr - 4, aarch64 instruction size) ----\n"
        "     (addr2line -e <binary> -f -C -p <file_offset>)\n");
    if (len > 0) {
        write(fd, buf, len);
    }

    for (int i = 2; i < nptrs; i++) {
        uint64_t ret_addr = (uint64_t)stack_pointers[i];
        uint64_t call_addr = ret_addr - 4;

        Dl_info info;
        uint64_t base = 0;
        const char* module = "??";
        if (dladdr((void*)call_addr, &info) && info.dli_fname) {
            base = (uint64_t)info.dli_fbase;
            module = info.dli_fname;
        }

        uint64_t call_offset = call_addr - base;

        len = snprintf(
            buf, sizeof(buf), "  #%02d  call_site=0x%016lx  file_offset=0x%lx  base=0x%016lx  %s\n", i, call_addr,
            call_offset, base, module);
        if (len > 0) {
            write(fd, buf, len);
        }
    }
}

// 信号处理函数
void SignalHandler(int signum)
{
    const int MAX_STACK_FRAMES = 64;
    void* stack_pointers[MAX_STACK_FRAMES];

    int fd = open(g_crash_file_name, O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd < 0) {
        _exit(1);
    }
    char msg[512] = {0};
    int len = snprintf(msg, sizeof(msg), "\n---- [Crash] Received signal:%d ----\n", signum);
    if (len > 0) {
        write(fd, msg, len);
    }

    int nptrs = backtrace(stack_pointers, MAX_STACK_FRAMES);
    backtrace_symbols_fd(stack_pointers, nptrs, fd);

    DumpCallSites(fd, stack_pointers, nptrs);

    len = snprintf(msg, sizeof(msg), "\n---- [Module Map] (file_offset = runtime_addr - base) ----\n");
    if (len > 0) {
        write(fd, msg, len);
    }
    dl_iterate_phdr(DumpModuleMap, &fd);

    close(fd);
    signal(signum, SIG_DFL);
    _exit(signum);
}

// 注册所有崩溃相关的信号
void RegisterSignalHandler()
{
    pid_t pid = getpid();
    sprintf(g_crash_file_name, "crash_%d.log", pid);
    signal(SIGSEGV, SignalHandler); // 段错误
    signal(SIGABRT, SignalHandler); // Abort (如 assert 失败)
    signal(SIGFPE, SignalHandler);  // 算术溢出/除零
    signal(SIGBUS, SignalHandler);  // 总线错误
    signal(SIGILL, SignalHandler);  // 非法指令
}

bool GetWqebufferByJettyId(uint64_t jettyId, uint64_t& wqeBuffer)
{
    // SQE 携带的 jettyId 是 16 位硬件编号，仅 USER_CTL_NORMAL(AICPU) 模式合法（[5312,9407]，
    // 各模式编号空间按硬件合同不相交，见 sim_capacity_limits.h）。入参在窄化转换前校验，
    // 超出命名空间直接失败。行按（编号, 创建者 pid, 模式）复合定位：设备进程的父进程即
    // 创建该 jetty 的 rank 进程（生命周期合同），owner 失效时查库自然失败、不自动改查其他
    // owner；mode 限定 AICPU，不允许命中同 owner 的 CCU/URMA 行。查询复用 RaJetty 既有
    // Index(jetty_id, pid, dieId) 的 (jetty_id, pid) 前缀，mode 为残余过滤条件。
    if (!HcclSim::IsAicpuJettyHwIdInRange(jettyId)) {
        HCCL_VM_ERROR(
            "jettyId[{}] outside USER_CTL_NORMAL range [{},{}]", jettyId,
            HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_BEGIN, HcclSim::HCCL_VM_AICPU_USER_CTL_JETTY_ID_END);
        return false;
    }
    auto raJetty = sim::runtime::Db::GetOneByPred<sim::runtime::RaJetty>(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&sim::runtime::RaJetty::jetty_id, static_cast<uint32_t>(jettyId)),
        HcclSim::Storage::Eq(&sim::runtime::RaJetty::pid, static_cast<uint64_t>(getppid())),
        HcclSim::Storage::Eq(&sim::runtime::RaJetty::mode, HcclSim::HCCL_VM_JETTY_MODE_USER_CTL_NORMAL)));
    if (!raJetty.ok()) {
        HCCL_VM_ERROR(
            "RaJetty jettyId:{:d} owner pid:{:d} mode:{:d} not found", jettyId, getppid(),
            HcclSim::HCCL_VM_JETTY_MODE_USER_CTL_NORMAL);
        return false;
    }

    wqeBuffer = raJetty.value->sqBuffer;
    return true;
}

bool GetLocalEidByJettyId(uint64_t jettyId, uint8_t* lcEid)
{
    // RaJetty 主键是自增行 id, jetty_id 是独立字段; 按 (jetty_id, 属主 pid, mode) 定位,
    // 与同进程 GetWqebufferByJettyId 的判定一致
    auto raJetty = sim::runtime::Db::GetOneByPred<sim::runtime::RaJetty>(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&sim::runtime::RaJetty::jetty_id, static_cast<uint32_t>(jettyId)),
        HcclSim::Storage::Eq(&sim::runtime::RaJetty::pid, static_cast<uint64_t>(getppid())),
        HcclSim::Storage::Eq(&sim::runtime::RaJetty::mode, HcclSim::HCCL_VM_JETTY_MODE_USER_CTL_NORMAL)));
    if (!raJetty.ok()) {
        HCCL_VM_ERROR("RaJetty jettyId:{:d} owner pid:{:d} not found", jettyId, getppid());
        return false;
    }

    auto raCtx = sim::runtime::Db::GetById<sim::runtime::RaContext>(raJetty->ctx_handle);
    if (!raCtx.ok()) {
        HCCL_VM_ERROR("RaContext id:{:d} not found", raJetty->ctx_handle);
        return false;
    }

    auto endPoint = sim::runtime::Db::GetById<sim::runtime::EndPoint>(raCtx->endpoint_id);
    if (!endPoint.ok()) {
        HCCL_VM_ERROR("EndPoint id:{:d} not found", raCtx->endpoint_id);
        return false;
    }

    // EndPoint.eid 按十六进制串字节序入库, 与 ranktable.json 显示序一致;
    // lcEid 保持该序, 与 rmEid(已翻转为同序)直接可比
    (void)memcpy(lcEid, endPoint->eid, HcclSim::URMA_EID_LEN);

    // 打印lcEid字节序, 便于与 EndPoint.eid/rmEid 核对
    HcclSim::Eid localEid{};
    (void)memcpy(localEid.raw, lcEid, HcclSim::URMA_EID_LEN);
    HCCL_VM_INFO("lcEid(ranktable order):{}, jettyId:{:d}", HcclSim::IpAddress(localEid).EidToHexString(), jettyId);
    return true;
}

static int g_h2dReadFd = -1;
static int g_d2hWriteFd = -1;

void InitPipeFds(int h2dReadFd, int d2hWriteFd)
{
    g_h2dReadFd = h2dReadFd;
    g_d2hWriteFd = d2hWriteFd;
}

int DeviceSendMsg(uint8_t cmd, const void* data, uint32_t dataLen)
{
    if (g_d2hWriteFd < 0) {
        HCCL_VM_ERROR("pipe d2h write fd not initialized.");
        return -1;
    }

    return sim::PipeSendMsg(g_d2hWriteFd, cmd, data, dataLen);
}

int DeviceRecvMsg(uint8_t& outCmd, void* outData, uint32_t maxLen, uint32_t& outLen)
{
    if (g_h2dReadFd < 0) {
        HCCL_VM_ERROR("pipe h2d read fd not initialized.");
        return -1;
    }

    return sim::PipeRecvMsg(g_h2dReadFd, outCmd, outData, maxLen, outLen);
}

static thread_local uint32_t g_lastQuerySqId = 0;

void SetLastQuerySqId(uint32_t sqId) { g_lastQuerySqId = sqId; }

uint32_t GetLastQuerySqId() { return g_lastQuerySqId; }

void SetDpuStreamId(uint32_t deviceId, uint64_t streamId)
{
    auto dpuRet = sim::runtime::Db::GetOneByPred<sim::runtime::DpuDeviceInfo>(
        HcclSim::Storage::Eq(&sim::runtime::DpuDeviceInfo::device_id, deviceId));

    if (dpuRet.ok()) {
        sim::runtime::Db::Update<sim::runtime::DpuDeviceInfo>(
            HcclSim::Storage::Eq(&sim::runtime::DpuDeviceInfo::id, dpuRet->id),
            HcclSim::Storage::Set(&sim::runtime::DpuDeviceInfo::stream_id, streamId));
    } else {
        sim::runtime::DpuDeviceInfo dpuInfo{};
        dpuInfo.device_id = deviceId;
        dpuInfo.stream_id = streamId;
        sim::runtime::Db::Add<sim::runtime::DpuDeviceInfo>(dpuInfo);
    }
}

#ifdef __cplusplus
}
#endif
