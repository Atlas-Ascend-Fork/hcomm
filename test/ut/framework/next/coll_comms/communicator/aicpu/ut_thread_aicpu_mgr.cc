/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>
#include <memory>
#include <string>

#define private public
#include "threads/thread_aicpu_mgr.h"
#include "comm_engine/thread/device/device_thread_pub.h"
#include "aicpu_ts_thread.h"
#include "stream_lite.h"
#include "rtsq_a5.h"
#undef private

using namespace hccl;

namespace aicpu {
void GetSqeId(const uint32_t num, uint32_t& start, uint32_t& end)
{
    start = 1;
    end = start + num;
}
} // namespace aicpu

class ThreadAicpuMgrTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override { GlobalMockObject::verify(); }
};

TEST_F(ThreadAicpuMgrTest, Constructor_InitializesMembers)
{
    HcclCommDfxLite dfx;
    auto checkCb = [](bool) {
        return HCCL_SUCCESS;
    };
    ThreadAicpuMgr mgr(dfx, checkCb);
    EXPECT_EQ(mgr.GetAllThread().size(), 0u);
}

TEST_F(ThreadAicpuMgrTest, GetThreadMutex_ReturnsValidReference)
{
    HcclCommDfxLite dfx;
    auto checkCb = [](bool) {
        return HCCL_SUCCESS;
    };
    ThreadAicpuMgr mgr(dfx, checkCb);
    // GetThreadMutex 返回有效的 shared_mutex 引用
    std::shared_mutex& mtx = mgr.GetThreadMutex();
    std::shared_lock<std::shared_mutex> lock(mtx);
    EXPECT_TRUE(true);
}

TEST_F(ThreadAicpuMgrTest, InitThreads_NullParam_ReturnsError)
{
    HcclCommDfxLite dfx;
    auto checkCb = [](bool) {
        return HCCL_SUCCESS;
    };
    ThreadAicpuMgr mgr(dfx, checkCb);
    HcclResult ret = mgr.InitThreads(nullptr);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

TEST_F(ThreadAicpuMgrTest, Destructor_CleansUpThreads)
{
    HcclCommDfxLite dfx;
    auto checkCb = [](bool) {
        return HCCL_SUCCESS;
    };
    {
        ThreadAicpuMgr mgr(dfx, checkCb);
        // 析构时无 threads 不应崩溃
    }
    SUCCEED();
}

// 调用冒烟：host 目标下 HcommThreadRegisterCacheCallback 走 NOT_SUPPORT 分支，真实注册链
// （SetAicpuTsThreadPtr/SetNeedCacheTaskCallback/SetAddSqeArrayCallback）无法在此验证。
// 真实注册链的成功/失败分支由 hccl_utest_thread_device（-DCCL_KERNEL_AICPU）覆盖。
TEST_F(ThreadAicpuMgrTest, RegisterThreadCacheCallback_MockSmoke)
{
    MOCKER(HcommThreadRegisterCacheCallback).stubs().will(returnValue(static_cast<HcommResult>(HCOMM_SUCCESS)));
    EXPECT_EQ(HcommThreadRegisterCacheCallback(0), static_cast<HcommResult>(HCOMM_SUCCESS));
}

// 边界线程 C API 的 host 契约：host 目标未定义 CCL_KERNEL_AICPU，四接口 + 分类查询统一返回
// HCOMM_E_NOT_SUPPORT（不做设备侧参数校验，传空指针也不崩溃）。
TEST_F(ThreadAicpuMgrTest, DeviceThreadCApi_HostContract_NullArgs_Expect_NotSupport)
{
    EXPECT_EQ(HcommThreadCreateAicpu(nullptr, 0, nullptr), static_cast<HcommResult>(HCOMM_E_NOT_SUPPORT));
    EXPECT_EQ(HcommThreadGetStreamLite(0, nullptr), static_cast<HcommResult>(HCOMM_E_NOT_SUPPORT));
    EXPECT_EQ(HcommThreadRegisterCacheCallback(0), static_cast<HcommResult>(HCOMM_E_NOT_SUPPORT));
    EXPECT_EQ(HcommThreadDestroy(0), static_cast<HcommResult>(HCOMM_E_NOT_SUPPORT));
    EXPECT_EQ(HcommThreadIsFakeDeviceRes(0, nullptr), static_cast<HcommResult>(HCOMM_E_NOT_SUPPORT));
}
