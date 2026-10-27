/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UBMEM_SYMMETRIC_MEMORY_H
#define UBMEM_SYMMETRIC_MEMORY_H

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "acl/acl_rt.h"
#include "hcomm_res_defs.h"
#include "hcomm_team_defs.h"
#include "hccl/hccl_types.h"
#include "ubmem_symmetric_memory_agent.h"

namespace hccl {

class CollComm;

struct UbmemShareableInfo {
    uint64_t offset;
    uint64_t size;
    aclrtMemFabricHandle handle;
};

class UbMemSymmetricMemory {
public:
    UbMemSymmetricMemory(
        CollComm* collComm, HcommTeamHandle lsaTeam, uint32_t netLayer, const std::vector<uint32_t>& worldRankIds);
    ~UbMemSymmetricMemory();

    HcclResult Init();
    void Finalize();
    HcclResult RegisterWindow(void* ptr, size_t size, HcclCommSymWindow winHandle, HcclComm comm);
    HcclResult DeregisterWindow(HcclCommSymWindow winHandle);
    HcommTeamHandle GetLsaTeam() const { return lsaTeam_; }

private:
    class SimpleVaAllocator;

    static constexpr uint64_t BYTES_PER_GB = 1024ULL * 1024ULL * 1024ULL;

    struct PeerMappingInfo {
        void* address{nullptr};
        aclrtDrvMemHandle handle{nullptr};
        bool mapped{false};
        bool ownsHandle{false};
    };

    // 一个物理内存块复用同一份Fabric共享信息，活跃Window通过refCount共享成员映射。
    struct PaMappingInfo {
        void* baseUserVa{nullptr};
        size_t baseVaSize{0};
        aclrtDrvMemHandle paHandle{nullptr};
        aclrtMemFabricHandle shareableHandle{};
        bool shareableHandleReady{false};          // 本地PA Handle已完成Fabric Export
        bool memberAccessGranted{false};           // Shareable Handle已授权给全部LSA成员进程
        std::vector<PeerMappingInfo> peerMappings; // 各LSA成员的VA映射及导入Handle释放信息
        std::vector<CommMem> memberMems;           // 发布到HcommWindow的完整内存块信息
        size_t heapBaseOffset{0};                  // 当前内存块在单个成员对称VA空间内的偏移
        bool vaOffsetReserved{false};              // heapBaseOffset是否已从对称VA空间中分配
        uint32_t refCount{0};                      // 复用当前物理内存映射的Window数量
    };

    struct WindowRecord {
        void* userVa{nullptr}; // 本次Window注册的用户内存范围起始地址
        size_t userSize{0};    // 本次Window注册的用户内存范围大小
        // 用户注册地址相对完整内存块基址baseUserVa的偏移，
        // 用于将底层完整映射转换为当前用户子区间Window。
        size_t offsetInBaseVa{0};
        HcclCommSymWindow devWin{nullptr};
        std::shared_ptr<PaMappingInfo> paMapInfo;
    };

    HcclResult EnsureInit();
    HcclResult GetAllMemberPids();
    HcclResult ValidateRegisterRange(void* ptr, size_t size) const;
    HcclResult InitSymmetricVa();
    HcclResult InitGranularity();
    HcclResult ReserveSymmetricVa();
    void FinalizeSymmetricVa();
    HcclResult GetMemoryInfo(
        void* ptr, size_t size, void*& baseUserVa, size_t& baseVaSize, aclrtDrvMemHandle& paHandle,
        size_t& offsetInBaseVa) const;
    HcclResult ExportLocalMapping(PaMappingInfo& mapping) const;
    HcclResult GrantLocalMemory(const PaMappingInfo& mapping);
    HcclResult ValidateMappingRange(uint64_t windowOffset, size_t mapSize) const;
    HcclResult ImportMemberHandle(
        uint32_t member, const std::vector<uint8_t>& shareableDesc, const PaMappingInfo& mapping,
        aclrtDrvMemHandle& handle, bool& ownsHandle) const;
    HcclResult MapAllMembers(
        uint64_t windowOffset, const std::vector<std::vector<uint8_t>>& shareableDescs, PaMappingInfo& mapping,
        std::vector<CommMem>& memberMems);
    HcclResult CleanupPartialMapping(PaMappingInfo& mapping, HcclResult originalResult);
    HcclResult ReleasePeerMapping(PeerMappingInfo& peer);
    HcclResult ReleaseMemberMappings(PaMappingInfo& mapping);
    HcclResult RegisterInternal(PaMappingInfo& paMapping);
    HcclResult PublishWindow(WindowRecord& record);
    HcclResult AddWindowRecord(std::unique_ptr<WindowRecord>& record, HcclComm comm);
    HcclResult ReleasePaMapping(WindowRecord& record);
    HcclResult CleanupWindow(WindowRecord& record);
    void EraseWindowAddressRecord(uintptr_t userAddress, const WindowRecord* record);

    CollComm* collComm_{nullptr};
    RankGraph* rankGraph_{nullptr};
    uint32_t selfRank_{0};
    uint32_t selfMember_{0}; // 当前Rank在LSA成员列表中的下标
    uint32_t lsaTeamSize_{0};
    uint32_t netLayer_{0};
    std::string commId_;
    std::vector<uint32_t> worldRankIds_;
    HcommTeamHandle lsaTeam_{nullptr};
    std::unique_ptr<UbMemSymmetricMemoryAgent> agent_;
    void* heapBase_{nullptr};
    size_t stride_{0};
    size_t granularity_{0};
    size_t activeMappingCount_{0}; // 当前已建立的成员VA映射数量
    std::once_flag runtimeInitFlag_;
    HcclResult runtimeInitResult_{HCCL_E_INTERNAL};
    std::vector<int32_t> memberPids_; // 用于授权Fabric Shareable Handle的LSA成员bare TGID
    std::unique_ptr<SimpleVaAllocator> vaAllocator_;
    std::multimap<uintptr_t, std::unique_ptr<WindowRecord>> windowsByAddress_;
    std::unordered_map<HcclCommSymWindow, WindowRecord*> windowsByHandle_; // Window由windowsByAddress_持有
    std::unordered_map<aclrtDrvMemHandle, std::shared_ptr<PaMappingInfo>> paMappingMap_;
    mutable std::mutex mutex_;
    bool finalized_{false};
};

} // namespace hccl

#endif // UBMEM_SYMMETRIC_MEMORY_H
