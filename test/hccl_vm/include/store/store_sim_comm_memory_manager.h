/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STORE_SIM_COMM_MEMORY_MANAGER_H
#define STORE_SIM_COMM_MEMORY_MANAGER_H

#include <cstdint>
#include <mutex>
#include <vector>

namespace sim {

constexpr uint32_t kSlotSize = 8448;
constexpr uint32_t kSlotC2sOff = 256;
constexpr uint32_t kSlotS2cOff = 4352;
constexpr uint32_t kSlotBufSize = 4096;
constexpr uint32_t kSlotsPerPage = 127099;
constexpr uint64_t kCommPageSize = 1ULL << 30;
constexpr uint32_t kPageMagic = 0x5241504F;
// 槽位池页数上限：每页 1GiB shm 懒分配（仅触达才创建），容量 = 127099 × kMaxPages。
// 4 页（50.8 万槽）只够 512p 全互联（512×511≈26.2 万对）；1024p 全互联需要
// 512×1023≈104.8 万对槽位 → 至少 9 页。16 页（203 万槽/峰值 16GiB）覆盖 1024p
// 并留一倍余量。公司 1024p redis 实测：池子在半程耗尽 → 后半配对永不落库 →
// 服务端 180×500ms 空等 → 通道建立 7200s 超时 → hcclRet 9。
constexpr uint32_t kMaxPages = 16;
constexpr uint32_t kAllocBatchSize = 4;
constexpr uint32_t kSlotRefCountOff = 136;
constexpr uint32_t kSlotHeaderSize = 256;
constexpr uint32_t kC2sLockOff = 0;
constexpr uint32_t kC2sSizeOff = 4;
constexpr uint32_t kS2cLockOff = 128;
constexpr uint32_t kS2cSizeOff = 132;
constexpr int32_t kSpinCountLimit = 20;

constexpr uint32_t kRsFdClient = 0x01U;
constexpr uint32_t kRsFdServer = 0x00U;
#define MAKE_FD(slot_idx, role) (((uint64_t)(role) << 63) | (uint64_t)(slot_idx))
#define FD_SLOT_IDX(fd) ((fd) & 0x7FFFFFFFFFFFFFFFULL)
#define FD_ROLE(fd) ((fd) >> 63)

#if defined(__aarch64__) || defined(__arm__)
#define CPU_PAUSE() __asm__ volatile("yield" ::: "memory")
#elif defined(__x86_64__) || defined(__i386__)
#define CPU_PAUSE() __builtin_ia32_pause()
#else
#define CPU_PAUSE() ((void)0)
#endif

struct PageHeader {
    uint32_t magic;
    uint32_t page_id;
    uint32_t slot_count;
    uint32_t next_free;
    uint32_t free_head;
    char padding[4076];
};

struct SlotHandle {
    char* base;
    char* sendBuf;
    volatile uint32_t* sendLock;
    volatile uint32_t* sendSize;
    char* recvBuf;
    volatile uint32_t* recvLock;
    volatile uint32_t* recvSize;
};

class CommunicationMemoryManager {
public:
    static CommunicationMemoryManager& GetInstance();

    CommunicationMemoryManager(const CommunicationMemoryManager&) = delete;
    CommunicationMemoryManager& operator=(const CommunicationMemoryManager&) = delete;

    bool InitPool();

    uint32_t AllocSlot();

    SlotHandle GetSlotHandle(uint32_t slotIdx, uint32_t role);

    int64_t Send(const SlotHandle& handle, const void* data, size_t size);

    int64_t Recv(const SlotHandle& handle, void* data, size_t size);

    void AddRef(uint32_t slotIdx);

    bool CloseSlot(uint32_t slotIdx);

    void Shutdown();

private:
    struct BatchCache {
        uint32_t start = 0;
        uint32_t current = 0;
        uint32_t batch_size = 0;
    };

    CommunicationMemoryManager();
    ~CommunicationMemoryManager();

    void* GetOrCreatePage(uint32_t pageId);

    std::mutex m_mutex;
    std::vector<void*> m_pages;
};

} // namespace sim

#endif // STORE_SIM_COMM_MEMORY_MANAGER_H
