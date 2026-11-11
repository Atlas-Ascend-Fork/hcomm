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
#include "hcomm_res_entity_defs.h"
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
    void TearDown() override
    {
        HcommThreadRegisterDestroyCallback(nullptr);
        GlobalMockObject::verify();
    }
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
    ThreadHandle nullHandle = 0;
    EXPECT_EQ(HcommThreadUnwrapHandle(nullptr), HCOMM_E_PTR);
    EXPECT_EQ(HcommThreadUnwrapHandle(&nullHandle), HCOMM_E_PTR);
    void* streamLite = nullptr;
    EXPECT_EQ(HcommThreadGetStreamLite(0, &streamLite), static_cast<HcommResult>(HCOMM_E_PTR));
    EXPECT_EQ(HcommThreadGetStreamLite(0, nullptr), static_cast<HcommResult>(HCOMM_E_PTR));
    EXPECT_EQ(HcommThreadRegisterCacheCallback(0), static_cast<HcommResult>(HCOMM_E_PTR));
    EXPECT_EQ(HcommThreadDestroy(0), static_cast<HcommResult>(HCOMM_E_PTR));
    bool isFake = false;
    EXPECT_EQ(HcommThreadIsFakeDeviceRes(0, &isFake), static_cast<HcommResult>(HCOMM_E_PTR));
}

TEST_F(ThreadDeviceApiTest, UnwrappedEntityCanQueryRegisteredThreadStream)
{
    int streamToken = 0;
    void* expectedStream = &streamToken;
    hccl::AicpuTsThread mockTarget{std::string{}};
    MOCKER_CPP_VIRTUAL(mockTarget, &hccl::AicpuTsThread::Init).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(mockTarget, &hccl::AicpuTsThread::GetStreamLitePtr).stubs().will(returnValue(expectedStream));
    ThreadHandle rawHandle = 0;
    ASSERT_EQ(HcommThreadCreateAicpu("ut", 2, &rawHandle), static_cast<HcommResult>(HCOMM_SUCCESS));
    HcommThreadEntity entity{};
    entity.abiHeader = {HCOMM_THREAD_ENTITY_VERSION, HCOMM_THREAD_ENTITY_MAGIC_WORD, sizeof(entity), 0};
    entity.compatThreadHandle = rawHandle;
    ThreadHandle handle = reinterpret_cast<ThreadHandle>(&entity);
    void* streamLite = nullptr;
    EXPECT_EQ(HcommThreadGetStreamLite(handle, &streamLite), static_cast<HcommResult>(HCOMM_E_PTR));
    EXPECT_EQ(HcommThreadUnwrapHandle(&handle), static_cast<HcommResult>(HCOMM_SUCCESS));
    EXPECT_EQ(handle, rawHandle);
    EXPECT_EQ(HcommThreadGetStreamLite(handle, &streamLite), static_cast<HcommResult>(HCOMM_SUCCESS));
    EXPECT_EQ(streamLite, expectedStream);
    EXPECT_EQ(HcommThreadDestroy(rawHandle), static_cast<HcommResult>(HCOMM_SUCCESS));
}

// 未注册（伪造）句柄：查询/分类/destroy 统一返回 HCOMM_E_PTR（拒绝裸解引用未注册句柄，防 UAF/ABA）
TEST_F(ThreadDeviceApiTest, UnregisteredHandle_QueryRejected_DestroyRejected)
{
    FakeDeviceThread thread(reinterpret_cast<void*>(0x1234));
    ThreadHandle handle = reinterpret_cast<ThreadHandle>(&thread);

    HcommThreadEntity entity{};
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

TEST_F(ThreadDeviceApiTest, UnwrappedEntityQueriesRegisteredThread)
{
    Hccl::StreamLite stream(126, 7, 0, 0);
    hccl::AicpuTsThread mockThread{std::string()};
    MOCKER_CPP_VIRTUAL(mockThread, &hccl::AicpuTsThread::Init).stubs().will(returnValue(HCCL_SUCCESS));
    void* expectedStream = &stream;
    MOCKER_CPP_VIRTUAL(mockThread, &hccl::AicpuTsThread::GetStreamLitePtr).stubs().will(returnValue(expectedStream));
    HcommThreadEntity entity{};
    entity.abiHeader = {HCOMM_THREAD_ENTITY_VERSION, HCOMM_THREAD_ENTITY_MAGIC_WORD, sizeof(HcommThreadEntity), 0};
    ThreadHandle entityHandle = reinterpret_cast<ThreadHandle>(&entity);
    ThreadHandle rawHandle = 0;
    ASSERT_EQ(HcommThreadCreateAicpu("t", 1, &rawHandle), HCOMM_SUCCESS);
    entity.compatThreadHandle = rawHandle;

    void* streamLite = nullptr;
    EXPECT_EQ(HcommThreadGetStreamLite(entityHandle, &streamLite), HCOMM_E_PTR);
    ThreadHandle resolved = entityHandle;
    ASSERT_EQ(HcommThreadUnwrapHandle(&resolved), HCOMM_SUCCESS);
    EXPECT_EQ(resolved, rawHandle);
    EXPECT_EQ(HcommThreadGetStreamLite(resolved, &streamLite), HCOMM_SUCCESS);
    EXPECT_EQ(streamLite, expectedStream);
    EXPECT_EQ(HcommThreadGetStreamLite(rawHandle, &streamLite), HCOMM_SUCCESS);
    EXPECT_EQ(streamLite, expectedStream);
    EXPECT_EQ(HcommThreadGetStreamLite(1, &streamLite), HCOMM_E_PTR);

    static ThreadHandle destroyedHandle = 0;
    destroyedHandle = 0;
    ASSERT_EQ(
        HcommThreadRegisterDestroyCallback(+[](ThreadHandle handle) {
            destroyedHandle = handle;
        }),
        HCOMM_SUCCESS);
    EXPECT_EQ(HcommThreadDestroy(entityHandle), HCOMM_E_PTR);
    EXPECT_EQ(HcommThreadDestroy(rawHandle), HCOMM_SUCCESS);
    EXPECT_EQ(destroyedHandle, rawHandle);
    EXPECT_EQ(HcommThreadGetStreamLite(entityHandle, &streamLite), HCOMM_E_PTR);
    EXPECT_EQ(HcommThreadGetStreamLite(rawHandle, &streamLite), HCOMM_E_PTR);
    // Entity 仍存活但对应线程已销毁，解包后仍须由 raw 注册表拒绝查询。
    resolved = entityHandle;
    ASSERT_EQ(HcommThreadUnwrapHandle(&resolved), HCOMM_SUCCESS);
    EXPECT_EQ(HcommThreadGetStreamLite(resolved, &streamLite), HCOMM_E_PTR);
    EXPECT_EQ(HcommThreadDestroy(rawHandle), HCOMM_E_PTR);
}

TEST_F(ThreadDeviceApiTest, UnwrapRejectsInvalidAndUninitializedEntity)
{
    HcommThreadEntity entity{};
    entity.abiHeader = {HCOMM_THREAD_ENTITY_VERSION, HCOMM_THREAD_ENTITY_MAGIC_WORD, sizeof(HcommThreadEntity), 0};
    ThreadHandle handle = reinterpret_cast<ThreadHandle>(&entity);
    EXPECT_EQ(HcommThreadUnwrapHandle(&handle), HCOMM_E_INTERNAL);
    entity.abiHeader.version = HCOMM_THREAD_ENTITY_VERSION + 1;
    EXPECT_EQ(HcommThreadUnwrapHandle(&handle), HCOMM_E_PARA);
    entity.abiHeader.version = HCOMM_THREAD_ENTITY_VERSION;
    entity.abiHeader.size = sizeof(HcommThreadEntity) - 1;
    EXPECT_EQ(HcommThreadUnwrapHandle(&handle), HCOMM_E_PARA);
}

TEST_F(ThreadDeviceApiTest, GeFakeRawHandleSurvivesUnwrap)
{
    hccl::AicpuTsThread mockThread{std::string()};
    MOCKER_CPP_VIRTUAL(mockThread, &hccl::AicpuTsThread::Init)
        .stubs()
        .will(invoke(+[](hccl::AicpuTsThread* thread) -> HcclResult {
            thread->SetFakeDeviceRes();
            return HCCL_SUCCESS;
        }));
    ThreadHandle rawHandle = 0;
    ASSERT_EQ(HcommThreadCreateAicpu("t", 1, &rawHandle), HCOMM_SUCCESS);
    ThreadHandle resolved = rawHandle;
    ASSERT_EQ(HcommThreadUnwrapHandle(&resolved), HCOMM_SUCCESS);
    EXPECT_EQ(resolved, rawHandle);
    bool isFake = false;
    EXPECT_EQ(HcommThreadIsFakeDeviceRes(resolved, &isFake), HCOMM_SUCCESS);
    EXPECT_TRUE(isFake);
    EXPECT_EQ(HcommThreadDestroy(resolved), HCOMM_SUCCESS);
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
