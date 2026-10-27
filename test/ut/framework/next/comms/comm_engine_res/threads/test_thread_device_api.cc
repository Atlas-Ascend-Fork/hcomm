/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 设备侧线程边界 C API（thread.cc 的 CCL_KERNEL_AICPU 分支）直接用例。
// 与 hccl_utest_ccore_profiling 同范式：本目标定义 CCL_KERNEL_AICPU 并编入 thread.cc，
// AICPU 框架符号在此本地 stub，避免加载真实运行框架。

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include <memory>
#include <string>
#define private public
#define protected public
#include "comm_engine/thread/device/device_thread_pub.h"
#include "aicpu_ts_thread.h"
#include "stream_lite.h"
#include "rtsq_a5.h"
#undef protected
#undef private

namespace aicpu {
void GetSqeId(const uint32_t num, uint32_t& start, uint32_t& end)
{
    start = 0x80000400U;
    end = start + num;
}
} // namespace aicpu

bool IsBatchLaunchMode() { return true; }
uint32_t GetSqFullTimeOut() { return 30U; }
HcclResult HandleDispatchAllStreams() { return HCCL_SUCCESS; }

class ThreadDeviceApiTest : public testing::Test {
protected:
    void TearDown() override { GlobalMockObject::verify(); }
};

// 仅用于验证 GetStreamLite/IsFakeDeviceRes/Destroy 的真实 device 分支
class FakeDeviceThread : public hccl::AicpuTsThread {
public:
    explicit FakeDeviceThread(void* streamLite) : hccl::AicpuTsThread(std::string()), streamLitePtr_(streamLite) {}
    void* GetStreamLitePtr() const override { return streamLitePtr_; }

private:
    void* streamLitePtr_;
};

// 参数校验（device 分支）：空指针/空句柄 → HCOMM_E_PTR
TEST_F(ThreadDeviceApiTest, ParamChecks_NullArgs_Expect_EPtr)
{
    EXPECT_EQ(HcommThreadCreateAicpu(nullptr, 0, nullptr), static_cast<HcommResult>(HCOMM_E_PTR));
    EXPECT_EQ(HcommThreadCreateAicpu("t", 1, nullptr), static_cast<HcommResult>(HCOMM_E_PTR));
    void* streamLite = nullptr;
    EXPECT_EQ(HcommThreadGetStreamLite(0, &streamLite), static_cast<HcommResult>(HCOMM_E_PTR));
    EXPECT_EQ(HcommThreadGetStreamLite(0, nullptr), static_cast<HcommResult>(HCOMM_E_PTR));
    EXPECT_EQ(HcommThreadRegisterCacheCallback(0), static_cast<HcommResult>(HCOMM_E_PTR));
    EXPECT_EQ(HcommThreadDestroy(0), static_cast<HcommResult>(HCOMM_E_PTR));
    bool isFake = false;
    EXPECT_EQ(HcommThreadIsFakeDeviceRes(0, &isFake), static_cast<HcommResult>(HCOMM_E_PTR));
}

// 未注册（伪造）句柄：查询/分类/destroy 统一返回 HCOMM_E_PTR（拒绝裸解引用未注册句柄，防 UAF/ABA）
TEST_F(ThreadDeviceApiTest, UnregisteredHandle_QueryRejected_DestroyRejected)
{
    FakeDeviceThread thread(reinterpret_cast<void*>(0x1234));
    ThreadHandle handle = reinterpret_cast<ThreadHandle>(&thread);

    void* streamLite = nullptr;
    EXPECT_EQ(HcommThreadGetStreamLite(handle, &streamLite), static_cast<HcommResult>(HCOMM_E_PTR));

    bool isFake = true;
    EXPECT_EQ(HcommThreadIsFakeDeviceRes(handle, &isFake), static_cast<HcommResult>(HCOMM_E_PTR));

    EXPECT_EQ(HcommThreadDestroy(handle), static_cast<HcommResult>(HCOMM_E_PTR));
    EXPECT_EQ(HcommThreadDestroy(handle), static_cast<HcommResult>(HCOMM_E_PTR));
}

// StreamLite 为空（未注册句柄）→ E_PTR
TEST_F(ThreadDeviceApiTest, GetStreamLite_NullStream_Expect_EPtr)
{
    FakeDeviceThread thread(nullptr);
    ThreadHandle handle = reinterpret_cast<ThreadHandle>(&thread);
    void* streamLite = nullptr;
    EXPECT_EQ(HcommThreadGetStreamLite(handle, &streamLite), static_cast<HcommResult>(HCOMM_E_PTR));
}

// 真实创建/销毁（依赖 AicpuTsThread 桩的 Init）：成功则句柄非空可销毁，失败不崩溃
TEST_F(ThreadDeviceApiTest, CreateAndDestroy)
{
    ThreadHandle handle = 0;
    HcommResult ret = HcommThreadCreateAicpu("ut_thread", 9, &handle);
    if (ret == HCOMM_SUCCESS) {
        EXPECT_NE(handle, 0u);
        EXPECT_EQ(HcommThreadDestroy(handle), static_cast<HcommResult>(HCOMM_SUCCESS));
    } else {
        EXPECT_EQ(handle, 0u);
    }
}

// 未注册（伪造）句柄：注册回调被拒（HCOMM_E_PTR）
TEST_F(ThreadDeviceApiTest, RegisterCacheCallback_UnregisteredHandle_Expect_EPtr)
{
    auto stream = std::make_unique<Hccl::StreamLite>(126, 7, 0, 0);
    FakeDeviceThread thread(reinterpret_cast<void*>(stream.get()));
    ThreadHandle handle = reinterpret_cast<ThreadHandle>(&thread);
    EXPECT_EQ(HcommThreadRegisterCacheCallback(handle), static_cast<HcommResult>(HCOMM_E_PTR));
}

// 真实注册链（失败分支）：线程无 StreamLite → E_PTR
TEST_F(ThreadDeviceApiTest, RegisterCacheCallback_NullStream_Expect_EPtr)
{
    FakeDeviceThread thread(nullptr);
    ThreadHandle handle = reinterpret_cast<ThreadHandle>(&thread);
    EXPECT_EQ(HcommThreadRegisterCacheCallback(handle), static_cast<HcommResult>(HCOMM_E_PTR));
}
