/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include "store_sim_comm_memory_manager.h"
#include "store_sim_resource_root.h"
#include "sim_log.h"

namespace sim {

static CommunicationMemoryManager* s_instance = nullptr;
static std::mutex s_instanceLock;

CommunicationMemoryManager& CommunicationMemoryManager::GetInstance()
{
    if (s_instance == nullptr) {
        std::lock_guard<std::mutex> lock(s_instanceLock);
        if (s_instance == nullptr) {
            s_instance = new CommunicationMemoryManager();
        }
    }
    return *s_instance;
}

CommunicationMemoryManager::CommunicationMemoryManager() {}

CommunicationMemoryManager::~CommunicationMemoryManager() {}

static void SlotLock(volatile uint32_t* lock)
{
    int spinCount = 0;
    while (__atomic_test_and_set(lock, __ATOMIC_ACQUIRE)) {
        if (spinCount < kSpinCountLimit) {
            CPU_PAUSE();
            spinCount++;
        } else {
            sched_yield();
        }
    }
}

static void SlotUnlock(volatile uint32_t* lock) { __atomic_clear(lock, __ATOMIC_RELEASE); }

bool CommunicationMemoryManager::InitPool()
{
    void* addr = GetOrCreatePage(0);
    if (!addr) {
        HCCL_VM_ERROR("Failed to create Page 0");
        return false;
    }
    HCCL_VM_INFO("Page 0 is ready, slots_per_page={}", kSlotsPerPage);
    return true;
}

uint32_t CommunicationMemoryManager::AllocSlot()
{
    thread_local BatchCache batchCache;
    uint32_t slotIdx;
    PageHeader* h0 = (PageHeader*)m_pages[0];

    // 本地批次缓存还有余量 → 直接分发
    if (batchCache.current < batchCache.batch_size) {
        slotIdx = batchCache.start + batchCache.current;
        batchCache.current++;
    } else {
        // 缓存耗尽 → CAS 抢新批次：先检查越界，失败时不推进 next_free
        uint32_t totalSlots = kSlotsPerPage * kMaxPages;
        uint32_t batchStart;
        uint32_t nextFree;
        do {
            nextFree = __atomic_load_n(&h0->next_free, __ATOMIC_RELAXED);
            if (nextFree > UINT32_MAX - kAllocBatchSize) {
                HCCL_VM_ERROR("AllocSlot: next_free overflow, next_free={}", nextFree);
                return UINT32_MAX;
            }
            if (nextFree >= totalSlots) {
                HCCL_VM_ERROR("AllocSlot: slot pool exhausted, next_free={}", nextFree);
                return UINT32_MAX;
            }
            batchStart = nextFree;
        } while (!__atomic_compare_exchange_n(
            &h0->next_free, &nextFree, nextFree + kAllocBatchSize, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED));
        batchCache.start = batchStart;
        batchCache.current = 1;
        batchCache.batch_size = std::min((uint32_t)kAllocBatchSize, totalSlots - batchStart);
        slotIdx = batchStart;
    }

    uint32_t pageId = slotIdx / kSlotsPerPage; // 算出 slot 在第几页
    GetOrCreatePage(pageId);                   // 确保目标页已打开
    if (!m_pages[pageId]) {
        HCCL_VM_ERROR("AllocSlot: GetOrCreatePage failed for pageId={}", pageId);
        return UINT32_MAX;
    }

    uint32_t offset = slotIdx % kSlotsPerPage; // 算出 slot 在页内第几个
    // 计算 slot 在共享内存中的绝对地址
    char* base = (char*)m_pages[pageId] + sizeof(PageHeader) + offset * kSlotSize;
    memset(base, 0, kSlotHeaderSize);

    __atomic_thread_fence(__ATOMIC_RELEASE);

    return slotIdx;
}

SlotHandle CommunicationMemoryManager::GetSlotHandle(uint32_t slotIdx, uint32_t role)
{
    SlotHandle handle{};
    // 计算 slot 所在页和页内偏移量
    uint32_t pageId = slotIdx / kSlotsPerPage;
    uint32_t offset = slotIdx % kSlotsPerPage;
    // 懒加载打开SHM文件, 确保页已打开
    GetOrCreatePage(pageId);
    if (pageId >= m_pages.size() || !m_pages[pageId]) {
        HCCL_VM_ERROR("GetSlotHandle: page {} unavailable", pageId);
        return handle;
    }

    char* base = (char*)m_pages[pageId] + sizeof(PageHeader) + offset * kSlotSize;
    handle.base = base;

    /* Slot 布局:  [c2s_lock][c2s_size]...[s2c_lock][s2c_size]...[c2s_buf 4KB]...[s2c_buf 4KB]
     *  offset:     0         4            128        132          256              4352
     */
    if (role == kRsFdClient) {
        handle.sendBuf = base + kSlotC2sOff;
        handle.sendLock = (volatile uint32_t*)(base + kC2sLockOff);
        handle.sendSize = (volatile uint32_t*)(base + kC2sSizeOff);
        handle.recvBuf = base + kSlotS2cOff;
        handle.recvLock = (volatile uint32_t*)(base + kS2cLockOff);
        handle.recvSize = (volatile uint32_t*)(base + kS2cSizeOff);
    } else {
        handle.sendBuf = base + kSlotS2cOff;
        handle.sendLock = (volatile uint32_t*)(base + kS2cLockOff);
        handle.sendSize = (volatile uint32_t*)(base + kS2cSizeOff);
        handle.recvBuf = base + kSlotC2sOff;
        handle.recvLock = (volatile uint32_t*)(base + kC2sLockOff);
        handle.recvSize = (volatile uint32_t*)(base + kC2sSizeOff);
    }
    return handle;
}

int64_t CommunicationMemoryManager::Send(const SlotHandle& handle, const void* data, size_t size)
{
    SlotLock(handle.sendLock);
    if (*handle.sendSize > kSlotBufSize) {
        SlotUnlock(handle.sendLock);
        return -1;
    }
    uint32_t free = kSlotBufSize - *handle.sendSize;
    uint32_t real = (size < free) ? size : free;
    memcpy(handle.sendBuf + *handle.sendSize, data, real);
    *handle.sendSize += real;
    SlotUnlock(handle.sendLock);
    return real;
}

int64_t CommunicationMemoryManager::Recv(const SlotHandle& handle, void* data, size_t size)
{
    SlotLock(handle.recvLock);
    if (*handle.recvSize == 0) {
        SlotUnlock(handle.recvLock);
        return 0;
    }
    uint32_t real = (size < *handle.recvSize) ? size : *handle.recvSize;
    memcpy(data, handle.recvBuf, real);
    *handle.recvSize -= real;
    if (*handle.recvSize > 0) {
        memmove(handle.recvBuf, handle.recvBuf + real, *handle.recvSize);
    }
    SlotUnlock(handle.recvLock);
    return real;
}

void CommunicationMemoryManager::AddRef(uint32_t slotIdx)
{
    SlotHandle h = GetSlotHandle(slotIdx, kRsFdClient);
    if (!h.base) {
        HCCL_VM_ERROR("AddRef: slot {} not found", slotIdx);
        return;
    }
    uint32_t* ref = (uint32_t*)(h.base + kSlotRefCountOff);
    __atomic_fetch_add(ref, 1, __ATOMIC_RELAXED);
}

bool CommunicationMemoryManager::CloseSlot(uint32_t slotIdx)
{
    SlotHandle h = GetSlotHandle(slotIdx, kRsFdClient);
    if (!h.base) {
        HCCL_VM_ERROR("CloseSlot: slot {} not found", slotIdx);
        return false;
    }
    uint32_t* ref = (uint32_t*)(h.base + kSlotRefCountOff);
    uint32_t prev = __atomic_fetch_sub(ref, 1, __ATOMIC_ACQ_REL);
    if (prev != 1) {
        return false;
    }
    memset(h.base, 0, kSlotHeaderSize);
    return true;
}

void CommunicationMemoryManager::Shutdown()
{
    for (void* addr : m_pages) {
        if (addr) {
            munmap(addr, kCommPageSize);
        }
    }
    m_pages.clear();
}

void* CommunicationMemoryManager::GetOrCreatePage(uint32_t pageId)
{
    if (pageId >= kMaxPages) {
        HCCL_VM_ERROR("GetOrCreatePage: pageId {} out of range (max {})", pageId, kMaxPages);
        return nullptr;
    }
    // 先检查是否已存在该页
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (pageId < m_pages.size() && m_pages[pageId] != nullptr) {
            return m_pages[pageId];
        }
    }
    // 该页不存在 → 创建
    char logicalName[64];
    snprintf(logicalName, sizeof(logicalName), "ra_sock_pool_%u", pageId);
    std::string name = sim::SimResourceRoot::BuildName(logicalName);
    HCCL_VM_INFO("open shm name: {}", name);

    // 创建共享内存文件
    int fd = open(name.c_str(), O_CREAT | O_RDWR | O_EXCL, 0666);
    void* addr = nullptr;
    if (fd != -1) {
        // 通过ftruncate 将文件扩展到1GB
        if (ftruncate(fd, kCommPageSize) != 0) {
            HCCL_VM_ERROR(
                "GetOrCreatePage: ftruncate failed for page {}, errno={} ({})", pageId, errno, strerror(errno));
            close(fd);
            unlink(name.c_str());
            return nullptr;
        }
        // 映射共享内存到进程地址空间
        addr = mmap(nullptr, kCommPageSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        if (addr == MAP_FAILED) {
            HCCL_VM_ERROR("GetOrCreatePage: mmap failed for page {}, errno={} ({})", pageId, errno, strerror(errno));
            unlink(name.c_str());
            return nullptr;
        }
        PageHeader* h = (PageHeader*)addr;
        h->page_id = pageId;
        h->slot_count = kSlotsPerPage;
        h->next_free = 0;
        h->free_head = 0;
        __atomic_store_n(&h->magic, kPageMagic, __ATOMIC_RELEASE);
    } else if (errno == EEXIST) { // 该页已存在 → 打开
        fd = open(name.c_str(), O_RDWR, 0666);
        if (fd == -1) {
            HCCL_VM_ERROR(
                "GetOrCreatePage: open (consumer) failed for page {}, errno={} ({})", pageId, errno, strerror(errno));
            return nullptr;
        }
        addr = mmap(nullptr, kCommPageSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        if (addr == MAP_FAILED) {
            HCCL_VM_ERROR(
                "GetOrCreatePage: mmap (consumer) failed for page {}, errno={} ({})", pageId, errno, strerror(errno));
            return nullptr;
        }
        PageHeader* h = (PageHeader*)addr;
        int spinCount = 0;
        // 等待页头初始化完成
        while (*(volatile uint32_t*)&h->magic != kPageMagic) {
            if (spinCount < kSpinCountLimit) {
                CPU_PAUSE();
                spinCount++;
            } else {
                sched_yield();
            }
        }
        __atomic_thread_fence(__ATOMIC_ACQUIRE);
    } else {
        HCCL_VM_ERROR(
            "GetOrCreatePage: open (creator) failed for page {}, errno={} ({})", pageId, errno, strerror(errno));
        return nullptr;
    }

    // 加入页表
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (pageId >= m_pages.size()) {
            m_pages.resize(pageId + 1, nullptr);
        }
        m_pages[pageId] = addr;
    }
    return addr;
}

} // namespace sim
