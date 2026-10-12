/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include <gtest/gtest.h>
#include <sys/mman.h>

#include "store_sim_comm_memory_manager.h"
#include "store_sim_resource_root.h"

class CommMemoryManagerTest : public testing::Test {
protected:
    void SetUp() override
    {
        // 先初始化 pid 目录（强清重建），保证 GetOrCreatePage 用 open() 的中间目录存在
        ASSERT_TRUE(sim::SimResourceRoot::GetInstance().Init());
        ASSERT_TRUE(sim::CommunicationMemoryManager::GetInstance().InitPool());
    }

    void TearDown() override
    {
        sim::CommunicationMemoryManager::GetInstance().Shutdown();
        // 清理本进程 pid 目录，隔离残留资源
        sim::SimResourceRoot::GetInstance().Cleanup();
    }
};

TEST_F(CommMemoryManagerTest, GetInstance_Singleton)
{
    sim::CommunicationMemoryManager& inst1 = sim::CommunicationMemoryManager::GetInstance();
    sim::CommunicationMemoryManager& inst2 = sim::CommunicationMemoryManager::GetInstance();
    EXPECT_EQ(&inst1, &inst2);
}

TEST_F(CommMemoryManagerTest, InitPool_Success)
{
    sim::CommunicationMemoryManager::GetInstance().Shutdown();
    unlink(sim::SimResourceRoot::BuildName("ra_sock_pool_0").c_str());
    EXPECT_TRUE(sim::CommunicationMemoryManager::GetInstance().InitPool());
}

TEST_F(CommMemoryManagerTest, AllocSlot_Basic)
{
    auto& mgr = sim::CommunicationMemoryManager::GetInstance();
    uint32_t first = mgr.AllocSlot();
    ASSERT_NE(first, UINT32_MAX);
    uint32_t second = mgr.AllocSlot();
    ASSERT_NE(second, UINT32_MAX);
    EXPECT_EQ(first + 1, second);
}

TEST_F(CommMemoryManagerTest, GetSlotHandle_Client)
{
    auto& mgr = sim::CommunicationMemoryManager::GetInstance();
    uint32_t slotIdx = mgr.AllocSlot();
    ASSERT_NE(slotIdx, UINT32_MAX);

    sim::SlotHandle h = mgr.GetSlotHandle(slotIdx, sim::kRsFdClient);
    ASSERT_NE(h.base, nullptr);
    EXPECT_EQ(h.sendBuf, h.base + sim::kSlotC2sOff);
    EXPECT_EQ(h.recvBuf, h.base + sim::kSlotS2cOff);
    EXPECT_EQ(h.sendLock, reinterpret_cast<volatile uint32_t*>(h.base + sim::kC2sLockOff));
    EXPECT_EQ(h.recvLock, reinterpret_cast<volatile uint32_t*>(h.base + sim::kS2cLockOff));
}

TEST_F(CommMemoryManagerTest, GetSlotHandle_Server)
{
    auto& mgr = sim::CommunicationMemoryManager::GetInstance();
    uint32_t slotIdx = mgr.AllocSlot();
    ASSERT_NE(slotIdx, UINT32_MAX);

    sim::SlotHandle h = mgr.GetSlotHandle(slotIdx, sim::kRsFdServer);
    ASSERT_NE(h.base, nullptr);
    EXPECT_EQ(h.sendBuf, h.base + sim::kSlotS2cOff);
    EXPECT_EQ(h.recvBuf, h.base + sim::kSlotC2sOff);
    EXPECT_EQ(h.sendLock, reinterpret_cast<volatile uint32_t*>(h.base + sim::kS2cLockOff));
    EXPECT_EQ(h.recvLock, reinterpret_cast<volatile uint32_t*>(h.base + sim::kC2sLockOff));
}

TEST_F(CommMemoryManagerTest, GetSlotHandle_InvalidSlot)
{
    auto& mgr = sim::CommunicationMemoryManager::GetInstance();
    sim::SlotHandle h = mgr.GetSlotHandle(UINT32_MAX, sim::kRsFdClient);
    EXPECT_EQ(h.base, nullptr);
}

TEST_F(CommMemoryManagerTest, Send_Basic)
{
    auto& mgr = sim::CommunicationMemoryManager::GetInstance();
    uint32_t slotIdx = mgr.AllocSlot();
    ASSERT_NE(slotIdx, UINT32_MAX);

    sim::SlotHandle client = mgr.GetSlotHandle(slotIdx, sim::kRsFdClient);
    const char* data = "hello";
    int64_t ret = mgr.Send(client, data, strlen(data));
    EXPECT_EQ(ret, static_cast<int64_t>(strlen(data)));
    EXPECT_EQ(*client.sendSize, static_cast<uint32_t>(strlen(data)));
}

TEST_F(CommMemoryManagerTest, Send_BufferFull)
{
    auto& mgr = sim::CommunicationMemoryManager::GetInstance();
    uint32_t slotIdx = mgr.AllocSlot();
    ASSERT_NE(slotIdx, UINT32_MAX);

    sim::SlotHandle client = mgr.GetSlotHandle(slotIdx, sim::kRsFdClient);
    char bigData[sim::kSlotBufSize + 1];
    memset(bigData, 'A', sizeof(bigData));
    int64_t ret = mgr.Send(client, bigData, sizeof(bigData));
    EXPECT_GT(ret, 0);
    EXPECT_LE(ret, static_cast<int64_t>(sim::kSlotBufSize));
}

TEST_F(CommMemoryManagerTest, Recv_Basic)
{
    auto& mgr = sim::CommunicationMemoryManager::GetInstance();
    uint32_t slotIdx = mgr.AllocSlot();
    ASSERT_NE(slotIdx, UINT32_MAX);

    sim::SlotHandle client = mgr.GetSlotHandle(slotIdx, sim::kRsFdClient);
    sim::SlotHandle server = mgr.GetSlotHandle(slotIdx, sim::kRsFdServer);

    const char* writeData = "test data";
    mgr.Send(client, writeData, strlen(writeData));

    char readBuf[64] = {0};
    int64_t ret = mgr.Recv(server, readBuf, sizeof(readBuf));
    EXPECT_EQ(ret, static_cast<int64_t>(strlen(writeData)));
    EXPECT_STREQ(readBuf, writeData);
}

TEST_F(CommMemoryManagerTest, Recv_NoData)
{
    auto& mgr = sim::CommunicationMemoryManager::GetInstance();
    uint32_t slotIdx = mgr.AllocSlot();
    ASSERT_NE(slotIdx, UINT32_MAX);

    sim::SlotHandle server = mgr.GetSlotHandle(slotIdx, sim::kRsFdServer);
    char buf[64];
    int64_t ret = mgr.Recv(server, buf, sizeof(buf));
    EXPECT_EQ(ret, 0);
}

TEST_F(CommMemoryManagerTest, SendRecv_Multiple)
{
    auto& mgr = sim::CommunicationMemoryManager::GetInstance();
    uint32_t slotIdx = mgr.AllocSlot();
    ASSERT_NE(slotIdx, UINT32_MAX);

    sim::SlotHandle client = mgr.GetSlotHandle(slotIdx, sim::kRsFdClient);
    sim::SlotHandle server = mgr.GetSlotHandle(slotIdx, sim::kRsFdServer);

    const char* data1 = "first";
    mgr.Send(client, data1, strlen(data1));
    const char* data2 = "second";
    mgr.Send(client, data2, strlen(data2));

    char readBuf[64] = {0};
    int64_t ret = mgr.Recv(server, readBuf, sizeof(readBuf));
    EXPECT_EQ(ret, static_cast<int64_t>(strlen(data1) + strlen(data2)));
    EXPECT_STREQ(readBuf, "firstsecond");
}

TEST_F(CommMemoryManagerTest, AddRef_CloseSlot_Lifecycle)
{
    auto& mgr = sim::CommunicationMemoryManager::GetInstance();
    uint32_t slotIdx = mgr.AllocSlot();
    ASSERT_NE(slotIdx, UINT32_MAX);

    mgr.AddRef(slotIdx);
    bool isLast = mgr.CloseSlot(slotIdx);
    EXPECT_TRUE(isLast);
}

TEST_F(CommMemoryManagerTest, CloseSlot_NotLast)
{
    auto& mgr = sim::CommunicationMemoryManager::GetInstance();
    uint32_t slotIdx = mgr.AllocSlot();
    ASSERT_NE(slotIdx, UINT32_MAX);

    mgr.AddRef(slotIdx);
    mgr.AddRef(slotIdx);
    bool isLast = mgr.CloseSlot(slotIdx);
    EXPECT_FALSE(isLast);
}

TEST_F(CommMemoryManagerTest, fd_EncodeDecode)
{
    uint32_t slotIdx = 12345;
    uint64_t fd = MAKE_FD(slotIdx, sim::kRsFdClient);
    EXPECT_EQ(FD_SLOT_IDX(fd), slotIdx);
    EXPECT_EQ(FD_ROLE(fd), sim::kRsFdClient);

    fd = MAKE_FD(slotIdx, sim::kRsFdServer);
    EXPECT_EQ(FD_SLOT_IDX(fd), slotIdx);
    EXPECT_EQ(FD_ROLE(fd), sim::kRsFdServer);
}

TEST_F(CommMemoryManagerTest, Shutdown_Cleanup)
{
    auto& mgr = sim::CommunicationMemoryManager::GetInstance();
    mgr.Shutdown();
    SUCCEED();
}
