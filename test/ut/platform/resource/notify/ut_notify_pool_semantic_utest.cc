/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>

#include <cstdlib>
#include <memory>
#include <new>
#include <string>
#include <vector>

#include "hccl/base.h"
#include "adapter_rts_common.h"
#include "device_capacity.h"
#include "notify_pool.h"
#include "notify_pool_impl.h"
#include "sal.h"

using namespace hccl;

namespace {
u32 g_createNotifyInitCount = 0;
u32 g_failAllocCount = 0;

// 第一次申请返回不满足 4 字节对齐的 offset，第二次返回满足对齐的 offset，
// 不依赖 mockcpp 的 will/then 生效顺序
HcclResult InitNotifyWithAlternateOffset(LocalIpcNotify* notify, const s32, const s32, const NotifyLoadType)
{
    notify->offset = (g_createNotifyInitCount++ == 0) ? 6 : 8;
    notify->notifyPtr = reinterpret_cast<HcclRtNotify>(0x1);
    return HCCL_SUCCESS;
}

HcclResult InitNotifyWithOffset8(LocalIpcNotify* notify, const s32, const s32, const NotifyLoadType)
{
    notify->offset = 8;
    notify->notifyPtr = reinterpret_cast<HcclRtNotify>(0x1);
    return HCCL_SUCCESS;
}

void PrepareResetNotifies(NotifyPoolImpl& impl)
{
    for (u32 index = 0; index < NOTIFY_RES_MGR_NUM; ++index) {
        auto notify = std::make_shared<LocalIpcNotify>();
        notify->notifyPtr = reinterpret_cast<HcclRtNotify>(0x1);
        impl.notifyResMgr_[index].notifyPoolDevIPCAsignedMap[7].push_back(notify);
        impl.notifyResMgr_[index].notifyPoolDevNoIPCAsignedMap[7].push_back(notify);
    }
}
} // namespace

// 一次性分配失败注入：g_failAllocCount 大于 0 时让接下来的分配抛 bad_alloc，
// 用于覆盖 CreateNotify 中 make_shared 抛异常的清理分支
void* operator new(std::size_t size)
{
    if (size == 0) {
        size = 1;
    }
    if (g_failAllocCount > 0) {
        --g_failAllocCount;
        throw std::bad_alloc();
    }
    void* ptr = std::malloc(size);
    if (ptr == nullptr) {
        throw std::bad_alloc();
    }
    return ptr;
}

void operator delete(void* ptr) noexcept { std::free(ptr); }

void operator delete(void* ptr, std::size_t) noexcept { std::free(ptr); }

class NotifyPoolSemanticTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {}
    void TearDown() override
    {
        g_failAllocCount = 0;
        GlobalMockObject::verify();
    }
};

TEST_F(NotifyPoolSemanticTest, ut_notify_pool_merge_and_slot_membership)
{
    NotifyPool pool;
    ASSERT_EQ(pool.Init(0), HCCL_SUCCESS);

    const std::string tagAlltoAll = "ALLTOALLV_0";
    const std::string tagNormal = "ALLREDUCE_0";
    ASSERT_EQ(pool.RegisterOp(tagAlltoAll), HCCL_SUCCESS);
    ASSERT_EQ(pool.RegisterOp(tagNormal), HCCL_SUCCESS);

    RemoteRankInfo info(0, 0, 0);
    ASSERT_EQ(SalGetBareTgid(&info.remotePid), HCCL_SUCCESS); // 使 pid 与本端一致，走 no-ipc 分配路径

    // 两条 tag 的请求数相同（均为 1），命中同一槽位的同一对象：并池语义成立
    std::shared_ptr<LocalIpcNotify> notifyAlltoAll = nullptr;
    std::shared_ptr<LocalIpcNotify> notifyNormal = nullptr;
    ASSERT_EQ(pool.Alloc(tagAlltoAll, info, notifyAlltoAll, NotifyLoadType::DEVICE_NOTIFY), HCCL_SUCCESS);
    ASSERT_EQ(pool.Alloc(tagNormal, info, notifyNormal, NotifyLoadType::DEVICE_NOTIFY), HCCL_SUCCESS);
    ASSERT_NE(notifyAlltoAll, nullptr);
    EXPECT_EQ(notifyAlltoAll, notifyNormal);

    // 槽归属判据（不需要访问私有成员）：8 字节对齐走另一个池槽，对象必须不同
    std::shared_ptr<LocalIpcNotify> notifyAligned = nullptr;
    ASSERT_EQ(pool.Alloc(tagNormal, info, notifyAligned, NotifyLoadType::DEVICE_NOTIFY, 8), HCCL_SUCCESS);
    ASSERT_NE(notifyAligned, nullptr);
    EXPECT_NE(notifyAlltoAll, notifyAligned);

    ASSERT_EQ(pool.UnregisterOp(tagAlltoAll), HCCL_SUCCESS);
    ASSERT_EQ(pool.UnregisterOp(tagNormal), HCCL_SUCCESS);
    ASSERT_EQ(pool.Destroy(), HCCL_SUCCESS);
}

TEST_F(NotifyPoolSemanticTest, ut_notify_pool_alltoall_align_pool_open)
{
    NotifyPool pool;
    ASSERT_EQ(pool.Init(0), HCCL_SUCCESS);

    // 改造点：对齐池不再排除 alltoall 类 tag，注册时无条件登记
    const std::string tagAlltoAll = "ALLTOALLV_ALIGN";
    ASSERT_EQ(pool.RegisterOp(tagAlltoAll), HCCL_SUCCESS);

    RemoteRankInfo info(0, 0, 0);
    ASSERT_EQ(SalGetBareTgid(&info.remotePid), HCCL_SUCCESS);

    // 先按默认对齐分配（普通池），再按 8 字节对齐分配（对齐池）
    std::shared_ptr<LocalIpcNotify> notifyNormal = nullptr;
    ASSERT_EQ(pool.Alloc(tagAlltoAll, info, notifyNormal, NotifyLoadType::DEVICE_NOTIFY), HCCL_SUCCESS);
    std::shared_ptr<LocalIpcNotify> notifyAligned = nullptr;
    ASSERT_EQ(pool.Alloc(tagAlltoAll, info, notifyAligned, NotifyLoadType::DEVICE_NOTIFY, 8), HCCL_SUCCESS);

    // 判别判据：两次申请落在不同池槽，因此必须是不同对象。
    ASSERT_NE(notifyNormal, nullptr);
    ASSERT_NE(notifyAligned, nullptr);
    EXPECT_NE(notifyNormal, notifyAligned);
    EXPECT_EQ(notifyAligned->offset % 8, 0); // 附带校验：对齐性质仍成立

    ASSERT_EQ(pool.UnregisterOp(tagAlltoAll), HCCL_SUCCESS);
    ASSERT_EQ(pool.Destroy(), HCCL_SUCCESS);
}

TEST_F(NotifyPoolSemanticTest, ut_notify_pool_index_self_consistent)
{
    EXPECT_EQ(NOTIFY_RES_MGR_NUM, 2u);
    EXPECT_EQ(NOTIFY_NORMAL, 0u);
    EXPECT_EQ(NOTIFY_ALIGN, 1u);

    static_assert(NOTIFY_NORMAL < NOTIFY_RES_MGR_NUM, "NOTIFY_NORMAL out of range");
    static_assert(NOTIFY_ALIGN < NOTIFY_RES_MGR_NUM, "NOTIFY_ALIGN out of range");
}

TEST_F(NotifyPoolSemanticTest, ut_alloc_notify_ipc_ok)
{
    NotifyPool pool;
    ASSERT_EQ(pool.Init(0), HCCL_SUCCESS);
    const std::string tag = "test_signal_create";
    ASSERT_EQ(pool.RegisterOp(tag), HCCL_SUCCESS);

    std::shared_ptr<LocalIpcNotify> localNotify = nullptr;
    RemoteRankInfo info(0, 0, 0);
    EXPECT_EQ(pool.Alloc(tag, info, localNotify, NotifyLoadType::DEVICE_NOTIFY), HCCL_SUCCESS);

    ASSERT_EQ(pool.UnregisterOp(tag), HCCL_SUCCESS);
    ASSERT_EQ(pool.Destroy(), HCCL_SUCCESS);
}

TEST_F(NotifyPoolSemanticTest, ut_alloc_notify_ipc_fail_tag_invalid)
{
    NotifyPool pool;
    ASSERT_EQ(pool.Init(0), HCCL_SUCCESS);
    const std::string tag = "test_signal_create";
    const std::string tagInvalid = "test_signal_create_1";
    ASSERT_EQ(pool.RegisterOp(tag), HCCL_SUCCESS);
    EXPECT_EQ(pool.RegisterOp(tag), HCCL_E_PARA);

    std::shared_ptr<LocalIpcNotify> localNotify = nullptr;
    RemoteRankInfo info(0, 0, 0);
    EXPECT_EQ(pool.Alloc(tagInvalid, info, localNotify, NotifyLoadType::DEVICE_NOTIFY), HCCL_E_PARA);
    EXPECT_EQ(pool.UnregisterOp(tagInvalid), HCCL_E_PARA);

    ASSERT_EQ(pool.UnregisterOp(tag), HCCL_SUCCESS);
    ASSERT_EQ(pool.Destroy(), HCCL_SUCCESS);
}

TEST_F(NotifyPoolSemanticTest, ut_alloc_notify_no_ipc_ok)
{
    NotifyPool pool;
    ASSERT_EQ(pool.Init(0), HCCL_SUCCESS);
    const std::string tag = "test_signal_create";
    ASSERT_EQ(pool.RegisterOp(tag), HCCL_SUCCESS);

    std::shared_ptr<LocalIpcNotify> localNotify = nullptr;
    RemoteRankInfo info(0, 0, 0);
    ASSERT_EQ(SalGetBareTgid(&info.remotePid), HCCL_SUCCESS);
    EXPECT_EQ(pool.Alloc(tag, info, localNotify, NotifyLoadType::DEVICE_NOTIFY), HCCL_SUCCESS);

    ASSERT_EQ(pool.UnregisterOp(tag), HCCL_SUCCESS);
    ASSERT_EQ(pool.Destroy(), HCCL_SUCCESS);
}

TEST_F(NotifyPoolSemanticTest, ut_alloc_notify_no_ipc_fail_tag_invalid)
{
    NotifyPool pool;
    ASSERT_EQ(pool.Init(0), HCCL_SUCCESS);
    const std::string tag = "test_signal_create";
    const std::string tagInvalid = "test_signal_create_1";
    ASSERT_EQ(pool.RegisterOp(tag), HCCL_SUCCESS);
    EXPECT_EQ(pool.RegisterOp(tag), HCCL_E_PARA);

    std::shared_ptr<LocalIpcNotify> localNotify = nullptr;
    RemoteRankInfo info(0, 0, 0);
    ASSERT_EQ(SalGetBareTgid(&info.remotePid), HCCL_SUCCESS);
    EXPECT_EQ(pool.Alloc(tagInvalid, info, localNotify, NotifyLoadType::DEVICE_NOTIFY), HCCL_E_PARA);
    EXPECT_EQ(pool.UnregisterOp(tagInvalid), HCCL_E_PARA);

    ASSERT_EQ(pool.UnregisterOp(tag), HCCL_SUCCESS);
    ASSERT_EQ(pool.Destroy(), HCCL_SUCCESS);
}

TEST_F(NotifyPoolSemanticTest, ut_alloc_notify_aligned)
{
    NotifyPool pool;
    ASSERT_EQ(pool.Init(0), HCCL_SUCCESS);
    const std::string tag = "test_signal_create";
    ASSERT_EQ(pool.RegisterOp(tag), HCCL_SUCCESS);

    std::shared_ptr<LocalIpcNotify> localNotifyDefault = nullptr;
    RemoteRankInfo infoDefault(0, 0, 0);
    EXPECT_EQ(pool.Alloc(tag, infoDefault, localNotifyDefault, NotifyLoadType::DEVICE_NOTIFY), HCCL_SUCCESS);

    std::shared_ptr<LocalIpcNotify> localNotifyAligned = nullptr;
    RemoteRankInfo infoAligned(0, 0, 0);
    EXPECT_EQ(pool.Alloc(tag, infoAligned, localNotifyAligned, NotifyLoadType::DEVICE_NOTIFY, 8), HCCL_SUCCESS);

    std::shared_ptr<LocalIpcNotify> localNotifyInvalid = nullptr;
    RemoteRankInfo infoInvalid(0, 0, 0);
    EXPECT_NE(pool.Alloc(tag, infoInvalid, localNotifyInvalid, NotifyLoadType::DEVICE_NOTIFY, 3), HCCL_SUCCESS);

    ASSERT_EQ(pool.UnregisterOp(tag), HCCL_SUCCESS);
    ASSERT_EQ(pool.Destroy(), HCCL_SUCCESS);
}

TEST_F(NotifyPoolSemanticTest, ut_create_notify_retry_for_four_byte_alignment)
{
    g_createNotifyInitCount = 0;
    MOCKER_CPP(&LocalIpcNotify::Init, HcclResult(LocalIpcNotify::*)(const s32, const s32, const NotifyLoadType))
        .stubs()
        .will(invoke(InitNotifyWithAlternateOffset));
    MOCKER(Is310PDevice).stubs().will(returnValue(false));

    NotifyPoolImpl impl(0);
    std::shared_ptr<LocalIpcNotify> notify = nullptr;
    EXPECT_EQ(impl.CreateNotify(notify, 0, 0, NotifyLoadType::HOST_NOTIFY, false, -1, 4), HCCL_SUCCESS);
    ASSERT_NE(notify, nullptr);
    EXPECT_EQ(g_createNotifyInitCount, 2u); // 第一次 offset=6 不满足对齐，重新申请一次
    EXPECT_EQ(notify->offset, 8u);
}

TEST_F(NotifyPoolSemanticTest, ut_create_notify_init_failed)
{
    MOCKER_CPP(&LocalIpcNotify::Init, HcclResult(LocalIpcNotify::*)(const s32, const s32, const NotifyLoadType))
        .stubs()
        .will(returnValue(HCCL_E_INTERNAL));

    NotifyPoolImpl impl(0);
    std::shared_ptr<LocalIpcNotify> notify = nullptr;
    EXPECT_EQ(impl.CreateNotify(notify, 0, 0, NotifyLoadType::HOST_NOTIFY, false, -1, INVALID_UINT), HCCL_E_INTERNAL);
    EXPECT_EQ(notify, nullptr);
}

TEST_F(NotifyPoolSemanticTest, ut_create_notify_set_ipc_failed)
{
    LocalNotify virtualMockTarget;
    MOCKER_CPP(&LocalIpcNotify::Init, HcclResult(LocalIpcNotify::*)(const s32, const s32, const NotifyLoadType))
        .stubs()
        .will(invoke(InitNotifyWithOffset8));
    MOCKER(Is310PDevice).stubs().will(returnValue(false));
    MOCKER_CPP_VIRTUAL(virtualMockTarget, &LocalNotify::SetIpc).stubs().will(returnValue(HCCL_E_INTERNAL));

    NotifyPoolImpl impl(0);
    std::shared_ptr<LocalIpcNotify> notify = nullptr;
    EXPECT_EQ(impl.CreateNotify(notify, 0, 0, NotifyLoadType::HOST_NOTIFY, true, 1, INVALID_UINT), HCCL_E_INTERNAL);
    EXPECT_EQ(notify, nullptr);
}

TEST_F(NotifyPoolSemanticTest, ut_create_notify_grant_failed)
{
    LocalNotify virtualMockTarget;
    MOCKER_CPP(&LocalIpcNotify::Init, HcclResult(LocalIpcNotify::*)(const s32, const s32, const NotifyLoadType))
        .stubs()
        .will(invoke(InitNotifyWithOffset8));
    MOCKER(Is310PDevice).stubs().will(returnValue(false));
    MOCKER_CPP_VIRTUAL(virtualMockTarget, &LocalNotify::SetIpc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&LocalIpcNotify::Grant).stubs().will(returnValue(HCCL_E_INTERNAL));

    NotifyPoolImpl impl(0);
    std::shared_ptr<LocalIpcNotify> notify = nullptr;
    EXPECT_EQ(impl.CreateNotify(notify, 0, 0, NotifyLoadType::HOST_NOTIFY, true, 1, INVALID_UINT), HCCL_E_INTERNAL);
    EXPECT_EQ(notify, nullptr);
}

TEST_F(NotifyPoolSemanticTest, ut_create_notify_alloc_failed)
{
    MOCKER_CPP(&LocalIpcNotify::Init, HcclResult(LocalIpcNotify::*)(const s32, const s32, const NotifyLoadType))
        .stubs()
        .will(invoke(InitNotifyWithOffset8));

    NotifyPoolImpl impl(0);
    std::shared_ptr<LocalIpcNotify> notify = nullptr;
    g_failAllocCount = 1; // 让 CreateNotify 中第一次 make_shared 分配失败
    EXPECT_EQ(impl.CreateNotify(notify, 0, 0, NotifyLoadType::HOST_NOTIFY, false, -1, INVALID_UINT), HCCL_E_MEMORY);
    EXPECT_EQ(g_failAllocCount, 0u); // 注入的分配失败已被消耗
    EXPECT_EQ(notify, nullptr);      // 分配失败后不返回对象
}

TEST_F(NotifyPoolSemanticTest, ut_notify_pool_destroy_failed_warning)
{
    MOCKER_CPP(&NotifyPoolImpl::Destroy).stubs().will(returnValue(HCCL_E_INTERNAL));
    {
        NotifyPoolImpl impl(0);
    }
}

TEST_F(NotifyPoolSemanticTest, ut_destroy_all_pool_maps_with_null_notify)
{
    NotifyPoolImpl impl(0);
    auto& resMgr = impl.notifyResMgr_[NOTIFY_NORMAL];
    resMgr.notifyPoolIPCAsignedMap[0].push_back(nullptr);
    resMgr.notifyPoolDevIPCAsignedMap[0].push_back(nullptr);
    resMgr.notifyPoolNoIPCAsignedMap[0].push_back(nullptr);
    resMgr.notifyPoolDevNoIPCAsignedMap[0].push_back(nullptr);

    EXPECT_EQ(impl.Destroy(), HCCL_SUCCESS);
    EXPECT_TRUE(resMgr.notifyPoolIPCAsignedMap.empty());
    EXPECT_TRUE(resMgr.notifyPoolDevIPCAsignedMap.empty());
    EXPECT_TRUE(resMgr.notifyPoolNoIPCAsignedMap.empty());
    EXPECT_TRUE(resMgr.notifyPoolDevNoIPCAsignedMap.empty());
}

TEST_F(NotifyPoolSemanticTest, ut_alloc_ipc_host_reuse_and_alignment_retry)
{
    LocalNotify virtualMockTarget;
    MOCKER_CPP(&LocalIpcNotify::Init, HcclResult(LocalIpcNotify::*)(const s32, const s32, const NotifyLoadType))
        .stubs()
        .will(invoke(InitNotifyWithOffset8));
    MOCKER(Is310PDevice).stubs().will(returnValue(false));
    MOCKER_CPP_VIRTUAL(virtualMockTarget, &LocalNotify::SetIpc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&LocalIpcNotify::Grant).stubs().will(returnValue(HCCL_SUCCESS));

    NotifyPoolImpl impl(0);
    ASSERT_EQ(impl.Init(), HCCL_SUCCESS);
    ASSERT_EQ(impl.RegisterOp("host_ipc_create"), HCCL_SUCCESS);
    ASSERT_EQ(impl.RegisterOp("host_ipc_reuse"), HCCL_SUCCESS);
    ASSERT_EQ(impl.RegisterOp("host_ipc_align"), HCCL_SUCCESS);

    // 池中预置一个 offset=6 的对象：前两次申请复用它，第三次要求 4 字节对齐时必须跳过它并新建
    auto seeded = std::make_shared<LocalIpcNotify>();
    seeded->offset = 6;
    seeded->notifyPtr = reinterpret_cast<HcclRtNotify>(0x1);
    auto& ipcMap = impl.notifyResMgr_[NOTIFY_NORMAL].notifyPoolIPCAsignedMap;
    ipcMap[0].push_back(seeded);

    RemoteRankInfo info(0, 0, 0);
    info.remotePid = impl.pid_ + 1;
    std::shared_ptr<LocalIpcNotify> created = nullptr;
    std::shared_ptr<LocalIpcNotify> reused = nullptr;
    std::shared_ptr<LocalIpcNotify> aligned = nullptr;
    ASSERT_EQ(impl.Alloc("host_ipc_create", info, NotifyLoadType::HOST_NOTIFY, created, INVALID_UINT), HCCL_SUCCESS);
    ASSERT_EQ(impl.Alloc("host_ipc_reuse", info, NotifyLoadType::HOST_NOTIFY, reused, INVALID_UINT), HCCL_SUCCESS);
    ASSERT_EQ(impl.Alloc("host_ipc_align", info, NotifyLoadType::HOST_NOTIFY, aligned, 4), HCCL_SUCCESS);

    EXPECT_EQ(created, seeded);
    EXPECT_EQ(reused, created);
    ASSERT_NE(aligned, nullptr);
    EXPECT_NE(aligned, created);
    EXPECT_EQ(aligned->offset, 8u);
    EXPECT_EQ(ipcMap[0].size(), 2u);
    EXPECT_EQ(impl.Destroy(), HCCL_SUCCESS);
}

TEST_F(NotifyPoolSemanticTest, ut_alloc_no_ipc_host_reuse)
{
    MOCKER_CPP(&LocalIpcNotify::Init, HcclResult(LocalIpcNotify::*)(const s32, const s32, const NotifyLoadType))
        .stubs()
        .will(invoke(InitNotifyWithOffset8));
    MOCKER(Is310PDevice).stubs().will(returnValue(false));

    NotifyPoolImpl impl(0);
    ASSERT_EQ(impl.Init(), HCCL_SUCCESS);
    ASSERT_EQ(impl.RegisterOp("host_no_ipc_create"), HCCL_SUCCESS);
    ASSERT_EQ(impl.RegisterOp("host_no_ipc_reuse"), HCCL_SUCCESS);

    RemoteRankInfo info(0, 0, 0);
    info.remotePid = impl.pid_;
    std::shared_ptr<LocalIpcNotify> created = nullptr;
    std::shared_ptr<LocalIpcNotify> reused = nullptr;
    ASSERT_EQ(impl.Alloc("host_no_ipc_create", info, NotifyLoadType::HOST_NOTIFY, created, INVALID_UINT), HCCL_SUCCESS);
    ASSERT_EQ(impl.Alloc("host_no_ipc_reuse", info, NotifyLoadType::HOST_NOTIFY, reused, INVALID_UINT), HCCL_SUCCESS);

    ASSERT_NE(created, nullptr);
    EXPECT_EQ(reused, created);
    EXPECT_EQ(impl.Destroy(), HCCL_SUCCESS);
}

TEST_F(NotifyPoolSemanticTest, ut_alloc_no_ipc_alignment_skip_and_create)
{
    MOCKER_CPP(&LocalIpcNotify::Init, HcclResult(LocalIpcNotify::*)(const s32, const s32, const NotifyLoadType))
        .stubs()
        .will(invoke(InitNotifyWithOffset8));
    MOCKER(Is310PDevice).stubs().will(returnValue(false));

    NotifyPoolImpl impl(0);
    ASSERT_EQ(impl.Init(), HCCL_SUCCESS);
    ASSERT_EQ(impl.RegisterOp("host_no_ipc_align"), HCCL_SUCCESS);

    // 池中预置一个 offset=6 的对象：4 字节对齐申请必须跳过它并新建
    auto seeded = std::make_shared<LocalIpcNotify>();
    seeded->offset = 6;
    seeded->notifyPtr = reinterpret_cast<HcclRtNotify>(0x1);
    auto& noIpcMap = impl.notifyResMgr_[NOTIFY_NORMAL].notifyPoolNoIPCAsignedMap;
    noIpcMap[0].push_back(seeded);

    RemoteRankInfo info(0, 0, 0);
    info.remotePid = impl.pid_;
    std::shared_ptr<LocalIpcNotify> aligned = nullptr;
    ASSERT_EQ(impl.Alloc("host_no_ipc_align", info, NotifyLoadType::HOST_NOTIFY, aligned, 4), HCCL_SUCCESS);

    ASSERT_NE(aligned, nullptr);
    EXPECT_NE(aligned, seeded);
    EXPECT_EQ(aligned->offset, 8u);
    EXPECT_EQ(noIpcMap[0].size(), 2u);
    EXPECT_EQ(impl.Destroy(), HCCL_SUCCESS);
}

TEST_F(NotifyPoolSemanticTest, ut_reset_notify_success_and_empty_rank)
{
    MOCKER(hrtNotifyReset).stubs().will(returnValue(HCCL_SUCCESS));

    NotifyPoolImpl impl(0);
    PrepareResetNotifies(impl);
    EXPECT_EQ(impl.ResetNotifyForDestRank(7), HCCL_SUCCESS);
    EXPECT_EQ(impl.ResetNotifyForDestRank(8), HCCL_SUCCESS);
    EXPECT_EQ(impl.ResetNotify(), HCCL_SUCCESS);
    EXPECT_EQ(impl.Destroy(), HCCL_SUCCESS);
}

TEST_F(NotifyPoolSemanticTest, ut_reset_notify_failures)
{
    MOCKER(hrtNotifyReset).stubs().will(returnValue(HCCL_E_INTERNAL));

    NotifyPoolImpl impl(0);
    PrepareResetNotifies(impl);
    EXPECT_EQ(impl.ResetNotifyForDestRank(7), HCCL_E_INTERNAL);

    impl.notifyResMgr_[NOTIFY_NORMAL].notifyPoolDevIPCAsignedMap.clear();
    EXPECT_EQ(impl.ResetNotifyForDestRank(7), HCCL_E_INTERNAL);

    PrepareResetNotifies(impl);
    EXPECT_EQ(impl.ResetNotify(), HCCL_E_INTERNAL);

    impl.notifyResMgr_[NOTIFY_NORMAL].notifyPoolDevIPCAsignedMap.clear();
    EXPECT_EQ(impl.ResetNotify(), HCCL_E_INTERNAL);
    EXPECT_EQ(impl.Destroy(), HCCL_SUCCESS);
}
