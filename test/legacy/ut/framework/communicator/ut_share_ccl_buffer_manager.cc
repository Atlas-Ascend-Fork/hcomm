/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>
#include <string>
#include "gtest/gtest.h"
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#include "share_ccl_buffer_manager_v2.h"
#include "orion_adapter_rts.h"
#define private public
#define protected public
#include "communicator_impl.h"
#undef private
#undef protected
#include "invalid_params_exception.h"

using namespace Hccl;

class ShareCclBufferManagerTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ShareCclBufferManager tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "ShareCclBufferManager tests tear down." << std::endl; }

    void SetUp() override { MOCKER(HrtFree).stubs().with(mockcpp::any()); }

    void TearDown() override { GlobalMockObject::verify(); }

    void MockDevice(s32 deviceId) { MOCKER(HrtGetDevice).stubs().will(returnValue(deviceId)); }

    void MockMalloc(uintptr_t addr1, uintptr_t addr2)
    {
        MOCKER(HrtMalloc)
            .stubs()
            .with(mockcpp::any(), mockcpp::any())
            .will(returnValue(reinterpret_cast<void*>(addr1)))
            .then(returnValue(reinterpret_cast<void*>(addr2)));
    }

    ShareCclBufferManager& Mgr() { return ShareCclBufferManager::GetInstance(); }
};

TEST_F(ShareCclBufferManagerTest, GetShareCclBufferReuseSameName)
{
    MockDevice(0);
    MockMalloc(0x1000, 0x2000);

    std::shared_ptr<DevBuffer> buffer1 = nullptr;
    std::shared_ptr<DevBuffer> buffer2 = nullptr;
    EXPECT_EQ(Mgr().GetShareCclBuffer("ut_share_reuse", 1024, buffer1), HCCL_SUCCESS);
    EXPECT_EQ(Mgr().GetShareCclBuffer("ut_share_reuse", 1024, buffer2), HCCL_SUCCESS);
    ASSERT_NE(buffer1, nullptr);
    ASSERT_NE(buffer2, nullptr);
    EXPECT_EQ(buffer1.get(), buffer2.get());
    EXPECT_EQ(buffer1->GetAddr(), static_cast<uintptr_t>(0x1000));
    EXPECT_EQ(buffer1.use_count(), 2);
}

TEST_F(ShareCclBufferManagerTest, GetShareCclBufferDifferentName)
{
    MockDevice(0);
    MockMalloc(0x1000, 0x2000);

    std::shared_ptr<DevBuffer> buffer1 = nullptr;
    std::shared_ptr<DevBuffer> buffer2 = nullptr;
    EXPECT_EQ(Mgr().GetShareCclBuffer("ut_share_name_a", 1024, buffer1), HCCL_SUCCESS);
    EXPECT_EQ(Mgr().GetShareCclBuffer("ut_share_name_b", 1024, buffer2), HCCL_SUCCESS);
    ASSERT_NE(buffer1, nullptr);
    ASSERT_NE(buffer2, nullptr);
    EXPECT_NE(buffer1.get(), buffer2.get());
    EXPECT_EQ(buffer1->GetAddr(), static_cast<uintptr_t>(0x1000));
    EXPECT_EQ(buffer2->GetAddr(), static_cast<uintptr_t>(0x2000));
}

TEST_F(ShareCclBufferManagerTest, GetShareCclBufferSizeMismatch)
{
    MockDevice(0);
    MockMalloc(0x1000, 0x2000);

    std::shared_ptr<DevBuffer> buffer1 = nullptr;
    std::shared_ptr<DevBuffer> buffer2 = nullptr;
    EXPECT_EQ(Mgr().GetShareCclBuffer("ut_share_size_mismatch", 1024, buffer1), HCCL_SUCCESS);
    EXPECT_EQ(Mgr().GetShareCclBuffer("ut_share_size_mismatch", 2048, buffer2), HCCL_E_PARA);
    ASSERT_NE(buffer1, nullptr);
    EXPECT_EQ(buffer2, nullptr);
}

TEST_F(ShareCclBufferManagerTest, GetShareCclBufferRecreateAfterRelease)
{
    MockDevice(0);
    MockMalloc(0x1000, 0x2000);

    std::shared_ptr<DevBuffer> buffer1 = nullptr;
    EXPECT_EQ(Mgr().GetShareCclBuffer("ut_share_release", 1024, buffer1), HCCL_SUCCESS);
    ASSERT_NE(buffer1, nullptr);
    EXPECT_EQ(buffer1->GetAddr(), static_cast<uintptr_t>(0x1000));

    const std::string commId = "ut_comm_release";
    constexpr s32 streamA = 1;
    EXPECT_EQ(Mgr().CheckCclBufferConflict(commId, "ut_share_release", streamA), HCCL_SUCCESS);

    buffer1.reset();
    std::shared_ptr<DevBuffer> buffer2 = nullptr;
    EXPECT_EQ(Mgr().GetShareCclBuffer("ut_share_release", 1024, buffer2), HCCL_SUCCESS);
    ASSERT_NE(buffer2, nullptr);
    EXPECT_EQ(buffer2->GetAddr(), static_cast<uintptr_t>(0x2000));

    // buffer释放后stream绑定同步失效，允许绑定新的stream
    constexpr s32 streamB = 2;
    EXPECT_EQ(Mgr().CheckCclBufferConflict(commId, "ut_share_release", streamB), HCCL_SUCCESS);
}

TEST_F(ShareCclBufferManagerTest, CheckCclBufferConflict)
{
    constexpr s32 streamA = 1;
    constexpr s32 streamB = 2;
    const std::string commA = "ut_comm_conflict_a";
    const std::string commB = "ut_comm_conflict_b";
    const std::string bufferName = "ut_share_conflict";

    // 空bufferName不做校验
    EXPECT_EQ(Mgr().CheckCclBufferConflict(commA, "", streamA), HCCL_SUCCESS);
    EXPECT_EQ(Mgr().CheckCclBufferConflict(commB, "", streamB), HCCL_SUCCESS);

    // 不同通信域使用同一条Stream时共享不冲突
    EXPECT_EQ(Mgr().CheckCclBufferConflict(commA, bufferName, streamA), HCCL_SUCCESS);
    EXPECT_EQ(Mgr().CheckCclBufferConflict(commB, bufferName, streamA), HCCL_SUCCESS);
    // 不同通信域使用不同Stream时判定冲突
    EXPECT_EQ(Mgr().CheckCclBufferConflict(commB, bufferName, streamB), HCCL_E_PARA);
}

TEST_F(ShareCclBufferManagerTest, CheckCclBufferConflictSameCommSwitchStream)
{
    constexpr s32 streamA = 1;
    constexpr s32 streamB = 2;
    const std::string commId = "ut_comm_single";
    const std::string bufferName = "ut_share_single_comm";

    EXPECT_EQ(Mgr().CheckCclBufferConflict(commId, bufferName, streamA), HCCL_SUCCESS);
    // 单个通信域切换Stream不视为冲突，由用户保证算子串行
    EXPECT_EQ(Mgr().CheckCclBufferConflict(commId, bufferName, streamB), HCCL_SUCCESS);
}

TEST_F(ShareCclBufferManagerTest, GetInstanceByDeviceIsolation)
{
    MockMalloc(0x1000, 0x2000);

    std::shared_ptr<DevBuffer> buffer0 = nullptr;
    std::shared_ptr<DevBuffer> buffer1 = nullptr;
    EXPECT_EQ(
        ShareCclBufferManager::GetInstance(0).GetShareCclBuffer("ut_share_device_iso", 1024, buffer0), HCCL_SUCCESS);
    EXPECT_EQ(
        ShareCclBufferManager::GetInstance(1).GetShareCclBuffer("ut_share_device_iso", 1024, buffer1), HCCL_SUCCESS);
    ASSERT_NE(buffer0, nullptr);
    ASSERT_NE(buffer1, nullptr);
    // 不同device各自维护一份共享buffer，互不影响
    EXPECT_NE(buffer0.get(), buffer1.get());
    EXPECT_EQ(buffer0->GetAddr(), static_cast<uintptr_t>(0x1000));
    EXPECT_EQ(buffer1->GetAddr(), static_cast<uintptr_t>(0x2000));
}

TEST_F(ShareCclBufferManagerTest, ReleaseCclBufferStreamBinding)
{
    constexpr s32 streamA = 1;
    constexpr s32 streamB = 2;
    const std::string commA = "ut_comm_release_a";
    const std::string commB = "ut_comm_release_b";
    const std::string bufferName = "ut_share_release_binding";

    EXPECT_EQ(Mgr().CheckCclBufferConflict(commA, bufferName, streamA), HCCL_SUCCESS);
    EXPECT_EQ(Mgr().CheckCclBufferConflict(commB, bufferName, streamA), HCCL_SUCCESS);

    // commA销毁后解除其Stream绑定，commB可切换到新的Stream
    Mgr().ReleaseCclBufferStreamBinding(commA, bufferName);
    EXPECT_EQ(Mgr().CheckCclBufferConflict(commB, bufferName, streamB), HCCL_SUCCESS);

    // bufferName为空时不做任何处理
    Mgr().ReleaseCclBufferStreamBinding(commB, "");
}

TEST_F(ShareCclBufferManagerTest, InitDataBufferManagerSharesSameNameBuffer)
{
    MockDevice(0);
    MOCKER(HrtMalloc).stubs().will(returnValue(reinterpret_cast<void*>(0x7000)));
    MOCKER(HrtMemset).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any());

    CommunicatorImpl commA;
    CommunicatorImpl commB;
    commA.rankSize = 2;
    commB.rankSize = 2;
    commA.devLogicId = 0;
    commB.devLogicId = 0;
    commA.cclBufferName_ = "ut_init_share_same_name";
    commB.cclBufferName_ = "ut_init_share_same_name";
    commA.cclBufferSize = 1; // 1MB
    commB.cclBufferSize = 1; // 1MB

    commA.InitDataBufferManager();
    commB.InitDataBufferManager();

    ASSERT_NE(commA.GetCclBuffer(), nullptr);
    ASSERT_NE(commB.GetCclBuffer(), nullptr);
    EXPECT_EQ(commA.GetCclBuffer()->GetAddr(), commB.GetCclBuffer()->GetAddr());
    EXPECT_EQ(commA.GetCclBuffer().use_count(), 2);
}

TEST_F(ShareCclBufferManagerTest, InitDataBufferManagerRejectsSizeMismatch)
{
    MockDevice(0);
    MOCKER(HrtMalloc).stubs().will(returnValue(reinterpret_cast<void*>(0x8000)));
    MOCKER(HrtMemset).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any());

    CommunicatorImpl commA;
    commA.rankSize = 2;
    commA.devLogicId = 0;
    commA.cclBufferName_ = "ut_init_share_size_mismatch";
    commA.cclBufferSize = 1; // 1MB
    commA.InitDataBufferManager();

    CommunicatorImpl commB;
    commB.rankSize = 2;
    commB.devLogicId = 0;
    commB.cclBufferName_ = "ut_init_share_size_mismatch";
    commB.cclBufferSize = 2; // 2MB
    EXPECT_THROW(commB.InitDataBufferManager(), InvalidParamsException);
}
