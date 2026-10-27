/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include <memory>
#include <string>

#define private public
#define protected public
#include "hccl_dpu_manager.h"
#undef private

#include "llt_hccl_stub_pub.h"
#include "acl/acl_rt.h"

using namespace hccl;

// DpuManager 仅保留「非 DPUTAG」的通用 workspace 内存管理，故 UT 只覆盖存留接口：
// CreateWorkspaceBuf / GetKFCWorkSpace / GetDevMemWorkSpace。

class DpuManagerTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "DpuManagerTest SetUpTestCase" << std::endl; }

    static void TearDownTestCase() { std::cout << "DpuManagerTest TearDownTestCase" << std::endl; }

    virtual void SetUp() { std::cout << "DpuManagerTest SetUp" << std::endl; }

    virtual void TearDown()
    {
        std::cout << "DpuManagerTest TearDown" << std::endl;
        GlobalMockObject::verify();
    }
};

// ========== CreateWorkspaceBuf 测试用例 ==========

// CreateWorkspaceBuf: 正常创建工作空间
TEST_F(DpuManagerTest, Ut_CreateWorkspaceBuf_When_NewTag_Expect_Success)
{
    DpuManager dpuManager;
    const char* memTag = "TEST_TAG";
    uint64_t size = 1024;
    bool newCreated = false;

    HcclResult ret = dpuManager.CreateWorkspaceBuf(memTag, &size, &newCreated);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(newCreated, true);
}

// CreateWorkspaceBuf: 相同tag第二次调用，newCreated应为false
TEST_F(DpuManagerTest, Ut_CreateWorkspaceBuf_When_SameTagAgain_Expect_NotNewCreated)
{
    DpuManager dpuManager;
    const char* memTag = "TEST_TAG";
    uint64_t size1 = 1024;
    uint64_t size2 = 2048;
    bool newCreated1 = false;
    bool newCreated2 = false;

    HcclResult ret1 = dpuManager.CreateWorkspaceBuf(memTag, &size1, &newCreated1);
    EXPECT_EQ(ret1, HCCL_SUCCESS);
    EXPECT_EQ(newCreated1, true);

    // 第二次调用相同tag，newCreated应为false
    HcclResult ret2 = dpuManager.CreateWorkspaceBuf(memTag, &size2, &newCreated2);
    EXPECT_EQ(ret2, HCCL_SUCCESS);
    EXPECT_EQ(newCreated2, false);
}

// CreateWorkspaceBuf: nullptr tag
TEST_F(DpuManagerTest, Ut_CreateWorkspaceBuf_When_NullTag_Expect_Success)
{
    DpuManager dpuManager;
    const char* memTag = nullptr;
    uint64_t size = 1024;
    bool newCreated = false;

    HcclResult ret = dpuManager.CreateWorkspaceBuf(memTag, &size, &newCreated);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(newCreated, true);
}

// CreateWorkspaceBuf: newCreated为nullptr
TEST_F(DpuManagerTest, Ut_CreateWorkspaceBuf_When_NewCreatedIsNull_Expect_Success)
{
    DpuManager dpuManager;
    const char* memTag = "TEST_TAG";
    uint64_t size = 1024;

    HcclResult ret = dpuManager.CreateWorkspaceBuf(memTag, &size, nullptr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// ========== GetKFCWorkSpace 测试用例 ==========

// GetKFCWorkSpace: 获取已创建的工作空间
TEST_F(DpuManagerTest, Ut_GetKFCWorkSpace_When_WorkspaceExists_Expect_NotNull)
{
    DpuManager dpuManager;
    const char* memTag = "TEST_TAG";
    uint64_t size = 1024;

    // 先创建工作空间
    dpuManager.CreateWorkspaceBuf(memTag, &size, nullptr);

    // 获取工作空间
    auto workspace = dpuManager.GetKFCWorkSpace(memTag);
    EXPECT_NE(workspace, nullptr);
}

// GetKFCWorkSpace: 获取不存在的tag
TEST_F(DpuManagerTest, Ut_GetKFCWorkSpace_When_TagNotExists_Expect_Null)
{
    DpuManager dpuManager;
    const char* memTag = "NOT_EXIST_TAG";

    auto workspace = dpuManager.GetKFCWorkSpace(memTag);
    EXPECT_EQ(workspace, nullptr);
}

// GetKFCWorkSpace: nullptr tag
TEST_F(DpuManagerTest, Ut_GetKFCWorkSpace_When_NullTag_Expect_Null)
{
    DpuManager dpuManager;
    const char* memTag = nullptr;

    auto workspace = dpuManager.GetKFCWorkSpace(memTag);
    EXPECT_EQ(workspace, nullptr);
}

// ========== GetDevMemWorkSpace 测试用例 ==========

// GetDevMemWorkSpace: 首次创建新工作空间
TEST_F(DpuManagerTest, Ut_GetDevMemWorkSpace_When_NewTag_Expect_Success)
{
    DpuManager dpuManager;
    std::string memTag = "DEV_MEM_TAG";
    uint64_t size = 2048;
    void* addr = nullptr;
    bool newCreated = false;

    HcclResult ret = dpuManager.GetDevMemWorkSpace(memTag, &size, &addr, &newCreated);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(addr, nullptr);
    EXPECT_EQ(newCreated, true);
}

// GetDevMemWorkSpace: 获取已存在的工作空间，大小一致
TEST_F(DpuManagerTest, Ut_GetDevMemWorkSpace_When_WorkspaceExists_SameSize_Expect_Success)
{
    DpuManager dpuManager;
    std::string memTag = "DEV_MEM_TAG";
    uint64_t size = 2048;
    void* addr1 = nullptr;
    bool newCreated1 = false;

    // 首次创建
    HcclResult ret1 = dpuManager.GetDevMemWorkSpace(memTag, &size, &addr1, &newCreated1);
    EXPECT_EQ(ret1, HCCL_SUCCESS);
    EXPECT_EQ(newCreated1, true);

    // 再次获取，大小相同
    void* addr2 = nullptr;
    bool newCreated2 = false;
    HcclResult ret2 = dpuManager.GetDevMemWorkSpace(memTag, &size, &addr2, &newCreated2);
    EXPECT_EQ(ret2, HCCL_SUCCESS);
    EXPECT_EQ(newCreated2, false);
}

// GetDevMemWorkSpace: 获取已存在的工作空间，大小不一致
TEST_F(DpuManagerTest, Ut_GetDevMemWorkSpace_When_WorkspaceExists_DiffSize_Expect_ParaError)
{
    DpuManager dpuManager;
    std::string memTag = "DEV_MEM_TAG";
    uint64_t size1 = 2048;
    void* addr1 = nullptr;
    bool newCreated1 = false;

    // 首次创建
    HcclResult ret1 = dpuManager.GetDevMemWorkSpace(memTag, &size1, &addr1, &newCreated1);
    EXPECT_EQ(ret1, HCCL_SUCCESS);

    // 再次获取，大小不同
    uint64_t size2 = 4096;
    void* addr2 = nullptr;
    bool newCreated2 = false;
    HcclResult ret2 = dpuManager.GetDevMemWorkSpace(memTag, &size2, &addr2, &newCreated2);
    EXPECT_EQ(ret2, HCCL_E_PARA);
}

// GetDevMemWorkSpace: addr为nullptr
TEST_F(DpuManagerTest, Ut_GetDevMemWorkSpace_When_AddrIsNull_Expect_Success)
{
    DpuManager dpuManager;
    std::string memTag = "DEV_MEM_TAG";
    uint64_t size = 2048;
    void* addr = nullptr;

    HcclResult ret = dpuManager.GetDevMemWorkSpace(memTag, &size, &addr, nullptr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(addr, nullptr);
}
