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
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>

#include <atomic>
#include <chrono>
#include <future>
#include <stdexcept>
#include <thread>
#include <vector>
#include "adapter_rts_common.h"

#define private public
#define protected public
#include "comm_queue_context/jetty_context.h"
#undef private
#undef protected

using namespace hcomm;
using namespace Hccl;

namespace {
// 构造一个有效的 Ctx。PI/CI 必须是真实 hrtMalloc 分配的内存：
// DestroyJettyResources 会经真实 hrtFree→aclrtFree 释放，而 aclrtFree 桩按"共享内存管理信息"
// 解引用指针（share_mem_t*），若传占位地址（如 0x4000）会直接段错误。
// 未走到销毁路径的用例其 PI/CI 在 ~JettyContext 跳过销毁时泄漏，UT 场景可接受。
JettyContext::Ctx MakeValidCtx()
{
    JettyContext::Ctx ctx{};
    ctx.handle = 0xAAABULL;
    ctx.handlePtr = reinterpret_cast<void*>(0x1000);
    ctx.jettyId = 7;
    ctx.sqBuffVa = 0x2000;
    ctx.dbAddr = 0x3000;
    ctx.keySize = 4;
    ctx.localQpKey[0] = 0x11;
    ctx.localQpKey[1] = 0x22;
    ctx.localQpKey[2] = 0x33;
    ctx.localQpKey[3] = 0x44;
    ctx.sqDepth = 8192;
    ctx.queueIndexMemSize = 0x100;
    (void)hrtMalloc(&ctx.sqPiPtr, ctx.queueIndexMemSize);
    (void)hrtMalloc(&ctx.sqCiPtr, ctx.queueIndexMemSize);
    (void)hrtMalloc(&ctx.cqPiPtr, ctx.queueIndexMemSize);
    (void)hrtMalloc(&ctx.cqCiPtr, ctx.queueIndexMemSize);
    ctx.rdmaHandle = reinterpret_cast<void*>(0x8000);
    ctx.jfcHandle = 0xBBBBULL;
    ctx.localPsn = 100;
    return ctx;
}
} // namespace

class JettyContextTest : public testing::Test {
protected:
    JettyContext ctx;

    void SetUp() override
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
    }
    void TearDown() override
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
    }
};

// ============ Acquire ============

TEST_F(JettyContextTest, Acquire_FirstCreate_Success)
{
    JettyContext::Ctx out;
    auto provideCtx = [](JettyContext::Ctx& c) -> HcclResult {
        c = MakeValidCtx();
        return HCCL_SUCCESS;
    };
    EXPECT_EQ(ctx.Acquire(provideCtx, out), HCCL_SUCCESS);
    EXPECT_EQ(out.handle, 0xAAABULL);
    EXPECT_EQ(out.jettyId, 7U);
    EXPECT_EQ(out.keySize, 4U);
    EXPECT_EQ(out.localQpKey[0], 0x11);
    EXPECT_TRUE(ctx.inner_.valid);
    EXPECT_FALSE(ctx.inner_.creating);
    EXPECT_EQ(ctx.inner_.refCount, 1U);
}

TEST_F(JettyContextTest, Acquire_Reuse_When_AlreadyValid)
{
    JettyContext::Ctx out;
    auto provideCtx = [](JettyContext::Ctx& c) -> HcclResult {
        c = MakeValidCtx();
        return HCCL_SUCCESS;
    };
    ASSERT_EQ(ctx.Acquire(provideCtx, out), HCCL_SUCCESS);

    JettyContext::Ctx out2;
    auto provideCtx2 = [](JettyContext::Ctx&) -> HcclResult {
        ADD_FAILURE() << "provideCtx should not be called on reuse";
        return HCCL_SUCCESS;
    };
    EXPECT_EQ(ctx.Acquire(provideCtx2, out2), HCCL_SUCCESS);
    EXPECT_EQ(out2.handle, 0xAAABULL);
    EXPECT_EQ(ctx.inner_.refCount, 2U);
}

TEST_F(JettyContextTest, Acquire_When_ProvideCtxFails_ReturnsError)
{
    JettyContext::Ctx out;
    auto provideCtx = [](JettyContext::Ctx&) -> HcclResult {
        return HCCL_E_INTERNAL;
    };
    EXPECT_EQ(ctx.Acquire(provideCtx, out), HCCL_E_INTERNAL);
    EXPECT_FALSE(ctx.inner_.valid);
    EXPECT_FALSE(ctx.inner_.creating);
    EXPECT_EQ(ctx.inner_.refCount, 0U);
}

TEST_F(JettyContextTest, Acquire_When_ProvideCtxFailsAfterTpCbFilled_ExecutesTpReleaseCbFallback)
{
    // 契约防御路径：provider 在填充 tpReleaseCb 后返回错误——回调是已移交 TP 引用的唯一
    // 释放路径（临时 connection 已置 releaseTpOnDestroy_=false），Acquire 第二段失败须兜底
    // 执行恰一次，防止 TpManager useCnt 永久漏扣；错误码照常透传，Inner 不保管回调。
    // 当前 helper 保证"移交是 provideCtx 最后一步"（失败路径回调为空，兜底恒 no-op），
    // 本用例固化兜底契约，防止未来 provider 违反顺序不变量时 useCnt 漏扣回归无法被 CI 捕获。
    JettyContext::Ctx out;
    int invokeCount = 0;
    auto provideCtx = [&invokeCount](JettyContext::Ctx& c) -> HcclResult {
        c.tpReleaseCb = [&invokeCount]() -> HcclResult {
            invokeCount++;
            return HCCL_SUCCESS;
        };
        return HCCL_E_INTERNAL;
    };
    EXPECT_EQ(ctx.Acquire(provideCtx, out), HCCL_E_INTERNAL);
    EXPECT_EQ(invokeCount, 1);
    EXPECT_TRUE(ctx.inner_.tpReleaseCb == nullptr);
    EXPECT_FALSE(ctx.inner_.valid);
    EXPECT_FALSE(ctx.inner_.creating);

    // 失败后状态干净，同 endpoint 可重新创建（creating 已清、Inner 无残留回调）
    JettyContext::Ctx out2;
    auto provideCtx2 = [](JettyContext::Ctx& c) -> HcclResult {
        c = MakeValidCtx();
        return HCCL_SUCCESS;
    };
    EXPECT_EQ(ctx.Acquire(provideCtx2, out2), HCCL_SUCCESS);
}

TEST_F(JettyContextTest, Acquire_When_ProvideCtxThrows_PropagatesException)
{
    // provideCtx 异常原样上抛（不吞异常）是有意设计；creating 的清理由 Acquire 第二段的
    // try/catch 兜底保证（生产路径 provideCtx 可抛：如 WaitForJettyCreated 轮询 GetStatus 触发的
    // ThrowAbnormalStatus/CheckRequestResult THROW，生产侧 EXCEPTION_CATCH 仅包连接构造、
    // 不覆盖该轮询），异常穿出时 creating 已清理，见 ClearsCreatingAndRetrySucceeds 用例
    JettyContext::Ctx out;
    auto provideCtx = [](JettyContext::Ctx&) -> HcclResult {
        throw std::runtime_error("mock throw");
    };
    EXPECT_THROW(ctx.Acquire(provideCtx, out), std::runtime_error);
    EXPECT_FALSE(ctx.inner_.creating);
    EXPECT_FALSE(ctx.inner_.valid);
}

TEST_F(JettyContextTest, Acquire_When_ProvideCtxThrows_ClearsCreatingAndRetrySucceeds)
{
    // 复现 WaitForJettyCreated 轮询 GetStatus 抛异常穿出 provideCtx 的场景：
    // Acquire 须在异常上抛前清理 creating 并唤醒等待方——否则同 endpoint 后续 Acquire 每次
    // 等满 32s 超时（共享 jetty 槽位在 Endpoint 生命周期内永久失效）；异常后可重试创建成功
    JettyContext::Ctx out;
    int callCount = 0;
    auto provideCtx = [&callCount](JettyContext::Ctx& c) -> HcclResult {
        callCount++;
        if (callCount == 1) {
            throw std::runtime_error("mock provider throw");
        }
        c = MakeValidCtx();
        return HCCL_SUCCESS;
    };
    EXPECT_THROW(ctx.Acquire(provideCtx, out), std::runtime_error);
    EXPECT_FALSE(ctx.inner_.creating);
    EXPECT_FALSE(ctx.inner_.valid);

    EXPECT_EQ(ctx.Acquire(provideCtx, out), HCCL_SUCCESS);
    EXPECT_EQ(callCount, 2);
    EXPECT_TRUE(ctx.inner_.valid);
}

TEST_F(JettyContextTest, Acquire_When_Creating_BlocksUntilCreatingCleared)
{
    // 模拟另一个线程正在创建：手动设置 creating=true，valid=false。
    // 主线程 Acquire 应阻塞在 cv_.wait_for 上。为避免 UT 等 32s，用 async + 短超时检测
    // 验证 Acquire 确实阻塞，然后手动清除 creating 并 notify 让其继续。
    // 这覆盖了 wait_for 等待路径与被唤醒后抢占创建权的分支。
    ctx.inner_.creating = true;
    ctx.inner_.valid = false;

    JettyContext::Ctx out;
    std::atomic<bool> acquireDone(false);
    std::atomic<HcclResult> acquireRet(HCCL_SUCCESS);

    auto fut = std::async(std::launch::async, [&]() {
        // 手动清除 creating 并 notify 后，等待线程被唤醒并抢占创建权，
        // 此时 provideCtx 会被正常调用完成首次创建，因此这里不应当断言失败。
        auto provideCtx = [](JettyContext::Ctx& c) -> HcclResult {
            c = MakeValidCtx();
            return HCCL_SUCCESS;
        };
        acquireRet = ctx.Acquire(provideCtx, out);
        acquireDone = true;
    });

    // 等待 200ms，确认 Acquire 仍在阻塞（未完成）
    auto status = fut.wait_for(std::chrono::milliseconds(200));
    EXPECT_EQ(status, std::future_status::timeout);
    EXPECT_FALSE(acquireDone.load());

    // 手动清除 creating 并 notify，让 Acquire 的 wait_for 谓词成真（creating=false），
    // Acquire 将进入"抢占创建权"分支，调 provideCtx 成功后完成。
    {
        std::lock_guard<std::mutex> lk(ctx.mtx_);
        ctx.inner_.creating = false;
    }
    ctx.cv_.notify_all();

    // Acquire 现在应快速完成
    ASSERT_EQ(fut.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    EXPECT_TRUE(acquireDone.load());
    EXPECT_EQ(acquireRet.load(), HCCL_SUCCESS);
}

TEST_F(JettyContextTest, Acquire_Concurrent_Waiters_ReuseAfterCreate)
{
    // 多线程并发 Acquire：首个创建成功后，其余线程应复用（不重复调 provideCtx）
    std::atomic<int> provideCallCount(0);
    auto provideCtx = [&provideCallCount](JettyContext::Ctx& c) -> HcclResult {
        provideCallCount++;
        c = MakeValidCtx();
        return HCCL_SUCCESS;
    };

    constexpr int N = 8;
    std::vector<std::future<HcclResult>> futs;
    std::vector<JettyContext::Ctx> outs(N);
    for (int i = 0; i < N; ++i) {
        futs.push_back(std::async(std::launch::async, [&, i]() {
            return ctx.Acquire(provideCtx, outs[i]);
        }));
    }
    for (auto& f : futs) {
        EXPECT_EQ(f.get(), HCCL_SUCCESS);
    }
    EXPECT_EQ(provideCallCount.load(), 1); // 仅首次创建调一次
    EXPECT_EQ(ctx.inner_.refCount, static_cast<uint32_t>(N));
}

// ============ Release ============

TEST_F(JettyContextTest, Release_When_NotValid_Skips)
{
    EXPECT_EQ(ctx.Release(), HCCL_SUCCESS);
    EXPECT_FALSE(ctx.inner_.valid);
}

TEST_F(JettyContextTest, Release_Decrement_RefCount)
{
    JettyContext::Ctx out;
    auto provideCtx = [](JettyContext::Ctx& c) -> HcclResult {
        c = MakeValidCtx();
        return HCCL_SUCCESS;
    };
    ASSERT_EQ(ctx.Acquire(provideCtx, out), HCCL_SUCCESS);
    ASSERT_EQ(ctx.Acquire(provideCtx, out), HCCL_SUCCESS); // refCount=2

    EXPECT_EQ(ctx.Release(), HCCL_SUCCESS);
    EXPECT_TRUE(ctx.inner_.valid);
    EXPECT_EQ(ctx.inner_.refCount, 1U);
}

TEST_F(JettyContextTest, Release_ToZero_DestroysResources)
{
    // MakeValidCtx 的 PI/CI 为真实 hrtMalloc 内存，归 0 销毁时 hrtFree 安全释放，不会段错误
    JettyContext::Ctx out;
    auto provideCtx = [](JettyContext::Ctx& c) -> HcclResult {
        c = MakeValidCtx();
        return HCCL_SUCCESS;
    };
    ASSERT_EQ(ctx.Acquire(provideCtx, out), HCCL_SUCCESS);

    EXPECT_EQ(ctx.Release(), HCCL_SUCCESS);
    EXPECT_FALSE(ctx.inner_.valid);
    EXPECT_EQ(ctx.inner_.refCount, 0U);
    EXPECT_EQ(ctx.inner_.handle, 0ULL);
}

TEST_F(JettyContextTest, Release_When_RefCountAlreadyZero_Skips)
{
    // 正常路径下 Release 归 0 会直接清空 inner_（valid=false），无法再观测 refCount==0 分支；
    // 手工构造"valid 但 refCount 已为 0"的防御性状态，验证跳过且不触发销毁，
    // 从而覆盖 Release 的 refCount==0 提前返回分支且避免误销毁。
    ctx.inner_.valid = true;
    ctx.inner_.handle = 0;
    ctx.inner_.refCount = 0;

    EXPECT_EQ(ctx.Release(), HCCL_SUCCESS);
    EXPECT_TRUE(ctx.inner_.valid);
    EXPECT_EQ(ctx.inner_.refCount, 0U);
}

// ============ tpReleaseCb（TP 引用移交） ============

TEST_F(JettyContextTest, Acquire_FirstCreate_StoresTpReleaseCb)
{
    // provider 填充的 tpReleaseCb 应存入 Inner 保管；channel 视图（InnerToCtx 产物）不携带该字段
    JettyContext::Ctx out;
    bool invoked = false;
    auto provideCtx = [&invoked](JettyContext::Ctx& c) -> HcclResult {
        c = MakeValidCtx();
        c.tpReleaseCb = [&invoked]() -> HcclResult {
            invoked = true;
            return HCCL_SUCCESS;
        };
        return HCCL_SUCCESS;
    };
    EXPECT_EQ(ctx.Acquire(provideCtx, out), HCCL_SUCCESS);
    EXPECT_TRUE(ctx.inner_.tpReleaseCb != nullptr);
    EXPECT_TRUE(out.tpReleaseCb == nullptr); // InnerToCtx 不回填
    EXPECT_FALSE(invoked);                   // Acquire 阶段不触发执行
}

TEST_F(JettyContextTest, Release_LastRef_InvokesTpReleaseCbOnce)
{
    // 引用归零销毁共享资源时执行 TP 释放回调，且仅执行一次（执行后清空）
    JettyContext::Ctx out;
    int invokeCount = 0;
    auto provideCtx = [&invokeCount](JettyContext::Ctx& c) -> HcclResult {
        c = MakeValidCtx();
        c.tpReleaseCb = [&invokeCount]() -> HcclResult {
            invokeCount++;
            return HCCL_SUCCESS;
        };
        return HCCL_SUCCESS;
    };
    ASSERT_EQ(ctx.Acquire(provideCtx, out), HCCL_SUCCESS);

    EXPECT_EQ(ctx.Release(), HCCL_SUCCESS);
    EXPECT_EQ(invokeCount, 1);
    EXPECT_TRUE(ctx.inner_.tpReleaseCb == nullptr);

    // 已销毁后再 Release 走 notValid 跳过，不重复执行
    EXPECT_EQ(ctx.Release(), HCCL_SUCCESS);
    EXPECT_EQ(invokeCount, 1);
}

TEST_F(JettyContextTest, Release_When_TpReleaseCbFails_ReturnsError)
{
    // 契约用例：TP 释放回调执行失败（如缓存键漂移导致 NOT_FOUND）时，共享 jetty 资源已销毁不可回滚，
    // 但 Release 不得无条件返回 SUCCESS——以回调失败码返回供调用方审计（useCnt 可能漏扣、条目常驻），
    // 避免键漂移类缺陷静默泄漏；回调仍恰好执行一次且 Inner 不残留
    JettyContext::Ctx out;
    int invokeCount = 0;
    auto provideCtx = [&invokeCount](JettyContext::Ctx& c) -> HcclResult {
        c = MakeValidCtx();
        c.tpReleaseCb = [&invokeCount]() -> HcclResult {
            invokeCount++;
            return HCCL_E_NOT_FOUND;
        };
        return HCCL_SUCCESS;
    };
    ASSERT_EQ(ctx.Acquire(provideCtx, out), HCCL_SUCCESS);

    EXPECT_EQ(ctx.Release(), HCCL_E_NOT_FOUND);
    EXPECT_EQ(invokeCount, 1);
    // 资源已销毁（不可回滚），Inner 已清空，后续 Release 走 notValid 跳过
    EXPECT_FALSE(ctx.inner_.valid);
    EXPECT_TRUE(ctx.inner_.tpReleaseCb == nullptr);
    EXPECT_EQ(ctx.Release(), HCCL_SUCCESS);
    EXPECT_EQ(invokeCount, 1);
}

TEST_F(JettyContextTest, Release_LastRef_ExecutesTpReleaseCbOutsideMtx)
{
    // 可证伪断言：TP 释放回调执行期间 mtx_ 必须未被持有（锁外执行是本次重构的核心语义，
    // 回调链进入 TpManager 信息锁域，持锁执行会跨模块锁嵌套）。回调内起独立线程对 mtx_
    // try_lock（跨线程语义良定义：锁被占用返回 false；勿用同线程 try_lock，非递归锁同线程
    // 重入是 UB），join 后断言成功——实现若回退为持锁执行，本用例即失败，为该语义提供 CI 防线
    JettyContext::Ctx out;
    bool lockAcquired = false;
    auto provideCtx = [this, &lockAcquired](JettyContext::Ctx& c) -> HcclResult {
        c = MakeValidCtx();
        c.tpReleaseCb = [this, &lockAcquired]() -> HcclResult {
            std::thread probe([this, &lockAcquired]() {
                lockAcquired = ctx.mtx_.try_lock();
                if (lockAcquired) {
                    ctx.mtx_.unlock();
                }
            });
            probe.join();
            return HCCL_SUCCESS;
        };
        return HCCL_SUCCESS;
    };
    ASSERT_EQ(ctx.Acquire(provideCtx, out), HCCL_SUCCESS);

    EXPECT_EQ(ctx.Release(), HCCL_SUCCESS);
    EXPECT_TRUE(lockAcquired);
}

TEST_F(JettyContextTest, Release_RefRemain_KeepsTpReleaseCb)
{
    // 引用未归零不销毁资源，TP 释放回调保持保管状态
    JettyContext::Ctx out;
    int invokeCount = 0;
    auto provideCtx = [&invokeCount](JettyContext::Ctx& c) -> HcclResult {
        c = MakeValidCtx();
        c.tpReleaseCb = [&invokeCount]() -> HcclResult {
            invokeCount++;
            return HCCL_SUCCESS;
        };
        return HCCL_SUCCESS;
    };
    ASSERT_EQ(ctx.Acquire(provideCtx, out), HCCL_SUCCESS);
    ASSERT_EQ(ctx.Acquire(provideCtx, out), HCCL_SUCCESS); // refCount=2

    EXPECT_EQ(ctx.Release(), HCCL_SUCCESS);
    EXPECT_EQ(invokeCount, 0);
    EXPECT_TRUE(ctx.inner_.tpReleaseCb != nullptr);
    EXPECT_TRUE(ctx.inner_.valid);
}

TEST_F(JettyContextTest, Acquire_When_MemcpyFail_RollbackInvokesTpReleaseCbOnce)
{
    // 第三段 memcpy_s 失败回滚：tpReleaseCb 已先于其他字段写入 Inner，
    // 回滚走 DestroyJettyResources 须执行回调恰好一次并清空，不泄漏已移交的 TP 引用
    JettyContext::Ctx out;
    int invokeCount = 0;
    auto provideCtx = [&invokeCount, this](JettyContext::Ctx& c) -> HcclResult {
        c = MakeValidCtx();
        c.tpReleaseCb = [&invokeCount, this]() -> HcclResult {
            invokeCount++;
            // 回调执行时 jetty/JFC 已销毁清零（DestroyJettyResources 先销毁资源后释放 TP）
            EXPECT_EQ(ctx.inner_.handle, 0ULL);
            EXPECT_EQ(ctx.inner_.jfcHandle, 0ULL);
            return HCCL_SUCCESS;
        };
        return HCCL_SUCCESS;
    };
    MOCKER(memcpy_s).stubs().will(returnValue(1)); // 1 != EOK
    EXPECT_EQ(ctx.Acquire(provideCtx, out), HCCL_E_INTERNAL);
    EXPECT_EQ(invokeCount, 1);
    EXPECT_TRUE(ctx.inner_.tpReleaseCb == nullptr);
    EXPECT_FALSE(ctx.inner_.valid);
    EXPECT_FALSE(ctx.inner_.creating);
}

TEST_F(JettyContextTest, Release_LastRef_TpReleaseCbThrows_DoesNotTerminate)
{
    // 回调抛异常须被捕获（不能穿出导致 Release 路径崩溃/terminate），并翻译为 HCCL_E_INTERNAL
    // 告警码返回（异常与失败码同口径可审计）；~JettyContext 链内捕获语义由析构用例覆盖
    JettyContext::Ctx out;
    auto provideCtx = [](JettyContext::Ctx& c) -> HcclResult {
        c = MakeValidCtx();
        c.tpReleaseCb = []() -> HcclResult {
            throw std::runtime_error("mock tp release throw");
        };
        return HCCL_SUCCESS;
    };
    ASSERT_EQ(ctx.Acquire(provideCtx, out), HCCL_SUCCESS);

    EXPECT_EQ(ctx.Release(), HCCL_E_INTERNAL);
    EXPECT_TRUE(ctx.inner_.tpReleaseCb == nullptr);
}

TEST_F(JettyContextTest, Destructor_ForceDestroy_InvokesTpReleaseCbAfterJettyDestroyed)
{
    // refCount=0 且 rdmaHandle 有效时 force destroy 分支：jetty 销毁后执行 TP 释放回调恰好一次，
    // 且回调执行期间 mtx_ 未被持有（锁外执行语义，验证手段同 Release_LastRef_ExecutesTpReleaseCbOutsideMtx）
    MOCKER_CPP(&Hccl::RdmaHandleManager::IsHandleValid).stubs().will(returnValue(true));
    int invokeCount = 0;
    bool lockAcquired = false;
    {
        JettyContext localCtx;
        {
            std::lock_guard<std::mutex> lk(localCtx.mtx_);
            // 手工构造 valid + refCount=0（正常 Release 归零会清空 inner_，无法保留到析构）
            localCtx.inner_.valid = true;
            localCtx.inner_.handle = 0xAAABULL;
            localCtx.inner_.refCount = 0;
            localCtx.inner_.rdmaHandle = reinterpret_cast<void*>(0x8000);
            localCtx.inner_.tpReleaseCb = [&invokeCount, &localCtx, &lockAcquired]() -> HcclResult {
                invokeCount++;
                // 回调执行时 jetty/JFC 已销毁清零：验证"先销毁后释放 TP"顺序
                EXPECT_EQ(localCtx.inner_.handle, 0ULL);
                EXPECT_EQ(localCtx.inner_.jfcHandle, 0ULL);
                // 回调执行期间 mtx_ 未被持有：独立线程 try_lock（跨线程语义良定义）
                std::thread probe([&localCtx, &lockAcquired]() {
                    lockAcquired = localCtx.mtx_.try_lock();
                    if (lockAcquired) {
                        localCtx.mtx_.unlock();
                    }
                });
                probe.join();
                return HCCL_SUCCESS;
            };
        }
    } // ~JettyContext: refCount=0 + rdmaValid → force destroy → 执行回调
    EXPECT_EQ(invokeCount, 1);
    EXPECT_TRUE(lockAcquired);
}

TEST_F(JettyContextTest, Destructor_InvalidRdmaHandle_SkipsDestroyButReleasesTp)
{
    // rdmaHandle 失效（进程退出/DeInit 防御分支）：device 资源跳过 unimport/destroy（句柄失效防崩溃），
    // 但 TpManager 为 intentional leak（进程级存活，见 tp_manager.cc GetInstance），TP 引用仍须释放一次，
    // 避免进程内 DeInit 后重 Init 命中陈旧 tpHandle 缓存
    int invokeCount = 0;
    {
        JettyContext localCtx;
        std::lock_guard<std::mutex> lk(localCtx.mtx_);
        localCtx.inner_.valid = true;
        localCtx.inner_.handle = 0xAAABULL;
        localCtx.inner_.refCount = 0;
        localCtx.inner_.rdmaHandle = reinterpret_cast<void*>(0x8000); // 不在 activeHandles → IsHandleValid=false
        localCtx.inner_.tpReleaseCb = [&invokeCount]() -> HcclResult {
            invokeCount++;
            return HCCL_SUCCESS;
        };
    }
    EXPECT_EQ(invokeCount, 1);
}

// ============ AcquireSharedRemoteJetty ============

TEST_F(JettyContextTest, AcquireSharedRemoteJetty_When_NullKey_ReturnsPtrError)
{
    bool needImport = false;
    uint64_t handle = 0;
    void* handlePtr = nullptr;
    uint32_t tpn = 0;
    // CHK_PTR_NULL 对空指针返回 HCCL_E_PTR
    EXPECT_EQ(ctx.AcquireSharedRemoteJetty(nullptr, 4, needImport, handle, handlePtr, tpn), HCCL_E_PTR);
}

TEST_F(JettyContextTest, AcquireSharedRemoteJetty_When_ZeroKeySize_ReturnsPara)
{
    uint8_t key[] = {0x01};
    bool needImport = false;
    uint64_t handle = 0;
    void* handlePtr = nullptr;
    uint32_t tpn = 0;
    EXPECT_EQ(ctx.AcquireSharedRemoteJetty(key, 0, needImport, handle, handlePtr, tpn), HCCL_E_PARA);
}

TEST_F(JettyContextTest, AcquireSharedRemoteJetty_When_InvalidLocalJetty_ReturnsInternal)
{
    uint8_t key[] = {0x01, 0x02};
    bool needImport = false;
    uint64_t handle = 0;
    void* handlePtr = nullptr;
    uint32_t tpn = 0;
    // inner_.valid 默认 false
    EXPECT_EQ(ctx.AcquireSharedRemoteJetty(key, 2, needImport, handle, handlePtr, tpn), HCCL_E_INTERNAL);
}

TEST_F(JettyContextTest, AcquireSharedRemoteJetty_FirstReserve_NeedImport)
{
    // 先 Acquire 让 inner_.valid = true
    JettyContext::Ctx out;
    auto provideCtx = [](JettyContext::Ctx& c) -> HcclResult {
        c = MakeValidCtx();
        return HCCL_SUCCESS;
    };
    ASSERT_EQ(ctx.Acquire(provideCtx, out), HCCL_SUCCESS);

    uint8_t key[] = {0xAA, 0xBB, 0xCC};
    bool needImport = false;
    uint64_t handle = 999;
    void* handlePtr = reinterpret_cast<void*>(0x999);
    uint32_t tpn = 999;
    EXPECT_EQ(ctx.AcquireSharedRemoteJetty(key, 3, needImport, handle, handlePtr, tpn), HCCL_SUCCESS);
    EXPECT_TRUE(needImport);
    EXPECT_EQ(handle, 0ULL);
    EXPECT_EQ(handlePtr, nullptr);
    EXPECT_EQ(tpn, 0U);
    EXPECT_EQ(ctx.inner_.remoteJettys.size(), 1U);
}

TEST_F(JettyContextTest, AcquireSharedRemoteJetty_When_Ready_ReuseHandle)
{
    JettyContext::Ctx out;
    auto provideCtx = [](JettyContext::Ctx& c) -> HcclResult {
        c = MakeValidCtx();
        return HCCL_SUCCESS;
    };
    ASSERT_EQ(ctx.Acquire(provideCtx, out), HCCL_SUCCESS);

    uint8_t key[] = {0xAA, 0xBB};
    bool needImport = false;
    uint64_t handle = 0;
    void* handlePtr = nullptr;
    uint32_t tpn = 0;
    ASSERT_EQ(ctx.AcquireSharedRemoteJetty(key, 2, needImport, handle, handlePtr, tpn), HCCL_SUCCESS);
    ASSERT_TRUE(needImport);

    // 发布
    void* publishedPtr = reinterpret_cast<void*>(0x1234);
    ASSERT_EQ(ctx.PublishSharedRemoteJetty(key, 2, 0x5678, publishedPtr, 42), HCCL_SUCCESS);

    // 再次 Acquire 应复用已发布的 handle
    bool needImport2 = true;
    uint64_t handle2 = 0;
    void* handlePtr2 = nullptr;
    uint32_t tpn2 = 0;
    EXPECT_EQ(ctx.AcquireSharedRemoteJetty(key, 2, needImport2, handle2, handlePtr2, tpn2), HCCL_SUCCESS);
    EXPECT_FALSE(needImport2);
    EXPECT_EQ(handle2, 0x5678ULL);
    EXPECT_EQ(handlePtr2, publishedPtr);
    EXPECT_EQ(tpn2, 42U);
}

TEST_F(JettyContextTest, AcquireSharedRemoteJetty_When_NotReady_Wait)
{
    JettyContext::Ctx out;
    auto provideCtx = [](JettyContext::Ctx& c) -> HcclResult {
        c = MakeValidCtx();
        return HCCL_SUCCESS;
    };
    ASSERT_EQ(ctx.Acquire(provideCtx, out), HCCL_SUCCESS);

    uint8_t key[] = {0x01};
    bool needImport = false;
    uint64_t handle = 0;
    void* handlePtr = nullptr;
    uint32_t tpn = 0;
    ASSERT_EQ(ctx.AcquireSharedRemoteJetty(key, 1, needImport, handle, handlePtr, tpn), HCCL_SUCCESS);
    ASSERT_TRUE(needImport); // 首次预约

    // 再次 Acquire 同 key，未发布：needImport=false, handle=0（调用方应等待）
    bool needImport2 = true;
    uint64_t handle2 = 123;
    void* handlePtr2 = reinterpret_cast<void*>(0x1);
    uint32_t tpn2 = 99;
    EXPECT_EQ(ctx.AcquireSharedRemoteJetty(key, 1, needImport2, handle2, handlePtr2, tpn2), HCCL_SUCCESS);
    EXPECT_FALSE(needImport2);
    EXPECT_EQ(handle2, 0ULL);
    EXPECT_EQ(handlePtr2, nullptr);
    EXPECT_EQ(tpn2, 0U);
}

// ============ PublishSharedRemoteJetty ============

TEST_F(JettyContextTest, PublishSharedRemoteJetty_When_NullKey_ReturnsPtrError)
{
    // CHK_PTR_NULL 对空指针返回 HCCL_E_PTR
    EXPECT_EQ(ctx.PublishSharedRemoteJetty(nullptr, 2, 0x1, reinterpret_cast<void*>(0x1), 1), HCCL_E_PTR);
}

TEST_F(JettyContextTest, PublishSharedRemoteJetty_When_NullHandlePtr_ReturnsPtrError)
{
    uint8_t key[] = {0x01};
    // CHK_PTR_NULL 对空指针返回 HCCL_E_PTR
    EXPECT_EQ(ctx.PublishSharedRemoteJetty(key, 1, 0x1, nullptr, 1), HCCL_E_PTR);
}

TEST_F(JettyContextTest, PublishSharedRemoteJetty_When_ZeroHandle_ReturnsPara)
{
    uint8_t key[] = {0x01};
    EXPECT_EQ(ctx.PublishSharedRemoteJetty(key, 1, 0, reinterpret_cast<void*>(0x1), 1), HCCL_E_PARA);
}

TEST_F(JettyContextTest, PublishSharedRemoteJetty_When_ZeroKeySize_ReturnsPara)
{
    uint8_t key[] = {0x01};
    EXPECT_EQ(ctx.PublishSharedRemoteJetty(key, 0, 0x1, reinterpret_cast<void*>(0x1), 1), HCCL_E_PARA);
}

TEST_F(JettyContextTest, PublishSharedRemoteJetty_When_NotFound_ReturnsNotFound)
{
    uint8_t key[] = {0x99};
    EXPECT_EQ(ctx.PublishSharedRemoteJetty(key, 1, 0x1, reinterpret_cast<void*>(0x1), 1), HCCL_E_NOT_FOUND);
}

TEST_F(JettyContextTest, PublishSharedRemoteJetty_Success)
{
    JettyContext::Ctx out;
    auto provideCtx = [](JettyContext::Ctx& c) -> HcclResult {
        c = MakeValidCtx();
        return HCCL_SUCCESS;
    };
    ASSERT_EQ(ctx.Acquire(provideCtx, out), HCCL_SUCCESS);

    uint8_t key[] = {0x11, 0x22};
    bool needImport = false;
    uint64_t handle = 0;
    void* handlePtr = nullptr;
    uint32_t tpn = 0;
    ASSERT_EQ(ctx.AcquireSharedRemoteJetty(key, 2, needImport, handle, handlePtr, tpn), HCCL_SUCCESS);
    ASSERT_TRUE(needImport);

    void* publishedPtr = reinterpret_cast<void*>(0xABCD);
    EXPECT_EQ(ctx.PublishSharedRemoteJetty(key, 2, 0x9999, publishedPtr, 5), HCCL_SUCCESS);
    ASSERT_EQ(ctx.inner_.remoteJettys.size(), 1U);
    EXPECT_TRUE(ctx.inner_.remoteJettys[0].ready);
    EXPECT_EQ(ctx.inner_.remoteJettys[0].handle, 0x9999ULL);
    EXPECT_EQ(ctx.inner_.remoteJettys[0].handlePtr, publishedPtr);
    EXPECT_EQ(ctx.inner_.remoteJettys[0].tpn, 5U);
}
