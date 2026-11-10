/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "jetty_context.h"
#include "log.h"
#include "adapter_rts_common.h"
#include <exception>

namespace hcomm {
namespace {

    constexpr uint32_t JETTY_CREATE_TIMEOUT_SECONDS = 32; // 覆盖 provideCtx 内部 16s jetty 创建 + 余量
    bool IsSameRemoteJetty(const JettyContext::SharedRemoteJettyCtx& ctx, const uint8_t* remoteQpKey, uint32_t keySize)
    {
        return ctx.remoteQpKey.size() == keySize && std::memcmp(ctx.remoteQpKey.data(), remoteQpKey, keySize) == 0;
    }

    // 在未持有 JettyContext::mtx_ 的作用域执行 TP 释放回调（执行时机约束见 DestroyJettyResources 注释）。
    // 返回执行结果供调用方审计：NOT_FOUND 等失败多为缓存键漂移类缺陷的信号，useCnt 不回减、
    // TP 条目可能常驻，须上抛记录（析构路径 best-effort 忽略返回值）。
    // 析构路径位于 ~JettyContext（隐式 noexcept）链内：回调内部经 TpManager::GetInstance 越界 THROW、
    // 日志实参字符串分配等均可能抛异常，穿出将 std::terminate，故与 DECTOR_TRY_CATCH 同口径全量捕获。
    HcclResult ExecuteTpReleaseCb(std::function<HcclResult()> tpReleaseCb)
    {
        if (tpReleaseCb == nullptr) {
            return HCCL_SUCCESS;
        }
        try {
            HcclResult ret = tpReleaseCb();
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR("[JettyContext][ExecuteTpReleaseCb] tp release callback failed, ret[%d].", ret);
            } else {
                HCCL_INFO("[JettyContext][ExecuteTpReleaseCb] released detached tp ownership.");
            }
            return ret;
        } catch (const std::exception& e) {
            HCCL_ERROR("[JettyContext][ExecuteTpReleaseCb] tp release callback exception: %s", e.what());
            return HCCL_E_INTERNAL;
        } catch (...) {
            HCCL_ERROR("[JettyContext][ExecuteTpReleaseCb] tp release callback unknown exception.");
            return HCCL_E_INTERNAL;
        }
    }
} // namespace

JettyContext::~JettyContext()
{
    // TP 释放回调统一移出后在锁外执行：回调链经适配层进入 TpManager 信息锁域
    // （ReleaseUbConnectionTp → ReleaseTpInfo → GetInfoCtxMutex），持 mtx_ 执行会跨模块锁嵌套
    std::function<HcclResult()> tpReleaseCb;
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (inner_.valid && inner_.handle != 0) {
            if (inner_.refCount == 0) {
                // 与 DevUbConnection::ReleaseResource 的 ctxValid 防御一致：DeInit 阶段本设备句柄可能
                // 已被回收（RdmaHandleManager 为 intentional leak 进程级存活，探测自身无 UB 风险，
                // 见 rdma_handle_manager.cc），用失效句柄 unimport/destroy 会崩溃。
                const bool rdmaValid = (inner_.rdmaHandle != nullptr)
                                       && Hccl::RdmaHandleManager::GetInstance().IsHandleValid(
                                           static_cast<Hccl::RdmaHandle>(inner_.rdmaHandle));
                if (!rdmaValid) {
                    HCCL_WARNING(
                        "[JettyContext][~JettyContext] shared jetty still valid on destroy but rdmaHandle "
                        "invalid, skip unimport/destroy to avoid using invalid handle, handle[%llu].",
                        static_cast<unsigned long long>(inner_.handle));
                    // rdmaHandle 失效仅代表 device 句柄管理器不可用，不代表 TP 引用可不归还：
                    // TpManager 为 intentional leak（进程级存活，见 tp_manager.cc），此处仍移出 TP 释放回调
                    // 在锁外释放，避免 DeInit 后进程内重 Init 命中陈旧 tpHandle 缓存
                    tpReleaseCb = std::move(inner_.tpReleaseCb);
                    inner_.tpReleaseCb = nullptr;
                } else {
                    HCCL_WARNING(
                        "[JettyContext][~JettyContext] shared jetty still valid on destroy, handle[%llu], "
                        "force destroy.",
                        static_cast<unsigned long long>(inner_.handle));
                    UnimportSharedRemoteJettys(inner_);
                    DestroyJettyResources(inner_, tpReleaseCb);
                }
            } else {
                HCCL_WARNING(
                    "[JettyContext][~JettyContext] shared jetty still in use, refCount[%u], handle[%llu], "
                    "skip destroy to avoid use-after-free, shared jetty and detached tp ownership are both kept "
                    "(check refcount leak if this warning appears).",
                    inner_.refCount, static_cast<unsigned long long>(inner_.handle));
            }
        }
    }
    // 析构路径 best-effort：仅记录（ExecuteTpReleaseCb 内部已含 ERROR 日志），不上抛不影响退出
    (void)ExecuteTpReleaseCb(std::move(tpReleaseCb));
}

HcclResult JettyContext::Acquire(const std::function<HcclResult(Ctx&)>& provideCtx, Ctx& outCtx)
{
    // 第一段（持锁）：检查是否已创建或正在创建。已创建则 refCount++ 返回；未创建则标记 creating。
    // 超时上限 32s（覆盖 provideCtx 内部 16s jetty 创建 + 余量），避免创建线程异常崩溃后其他线程永久阻塞。
    {
        std::unique_lock<std::mutex> lk(mtx_);
        if (!cv_.wait_for(lk, std::chrono::seconds(JETTY_CREATE_TIMEOUT_SECONDS), [this] {
                return inner_.valid || !inner_.creating;
            })) {
            HCCL_ERROR("[JettyContext][Acquire] wait for shared jetty creation timeout[32s].");
            return HCCL_E_TIMEOUT;
        }
        if (inner_.valid) {
            CHK_RET(InnerToCtx(inner_, outCtx));
            inner_.refCount++;
            HCCL_INFO(
                "[JettyContext][Acquire] reuse shared jetty, handle[%llu], refCount[%u]",
                static_cast<unsigned long long>(outCtx.handle), inner_.refCount);
            return HCCL_SUCCESS;
        }
        // 抢占创建权
        inner_.creating = true;
    }

    // 第二段（无锁）：执行首次创建回调（含网络建链 I/O，可能耗时数秒）。
    // 创建期间不持锁，其他线程的 Acquire 在 cv_ 上等待，Release 不被阻塞。
    // provideCtx 抛异常（如 WaitForJettyCreated 轮询 GetStatus 触发的 ThrowAbnormalStatus/
    // CheckRequestResult THROW 经连接状态机上抛）时：先持锁清理 creating 并唤醒等待方再原样上抛，
    // 否则 creating 永久滞留，同 endpoint 后续 Acquire 每次等满 32s 超时（共享 jetty 槽位失效）
    Ctx createdCtx;
    HcclResult createRet;
    try {
        createRet = provideCtx(createdCtx);
    } catch (...) {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            inner_.creating = false;
            cv_.notify_all();
        }
        // 与错误码路径同口径的契约防御：provider 若在填充 tpReleaseCb 后抛异常，回调是已移交
        // TP 引用的唯一释放路径（临时 connection 已置 releaseTpOnDestroy_=false），锁外兜底执行
        // 恰一次后上抛（ExecuteTpReleaseCb 内部全量捕获，不影响异常传播）
        ExecuteTpReleaseCb(std::move(createdCtx.tpReleaseCb));
        throw;
    }
    if (createRet != HCCL_SUCCESS) {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            inner_.creating = false;
            cv_.notify_all();
            HCCL_ERROR("[JettyContext][Acquire] provideCtx failed, ret[%d].", createRet);
        }
        // 契约防御：provider 在填充 tpReleaseCb 后失败的场景（当前 helper 保证移交是 provideCtx 的
        // 最后一步、失败路径回调为空，本兜底恒为 no-op）——回调是已移交 TP 引用的唯一释放路径
        // （临时 connection 已置 releaseTpOnDestroy_=false），随 createdCtx 析构直接丢弃将令
        // TpManager useCnt 永久漏扣，故锁外兜底执行恰一次（与 Release/第三段回滚口径一致）。
        // 回调自身失败叠加审计日志（createRet 已为错误码，保持原错误码返回）
        (void)ExecuteTpReleaseCb(std::move(createdCtx.tpReleaseCb));
        return createRet;
    }

    // 第三段（持锁，委托 PublishCreatedCtx）：写入缓存，清除 creating 标记，设置 refCount=1。
    // 失败回滚的 TP 释放回调先移出到 rollbackTpReleaseCb，待锁作用域结束再执行：回调链经适配层进入
    // TpManager 信息锁域，持 mtx_ 执行会把单 endpoint 临界区与全设备同协议 TP 信息操作串行化，
    // 放大其他 channel 的 Acquire 等待方至 32s 超时风险（本特性目标恰是高频建链/销毁场景）。
    std::function<HcclResult()> rollbackTpReleaseCb;
    HcclResult publishRet = PublishCreatedCtx(createdCtx, outCtx, rollbackTpReleaseCb);
    HcclResult tpRet = ExecuteTpReleaseCb(std::move(rollbackTpReleaseCb));
    if (publishRet == HCCL_SUCCESS && tpRet != HCCL_SUCCESS) {
        // 成功路径回调恒为空（no-op），本分支防御性保留：发布成功但回调执行失败时以告警码返回
        return tpRet;
    }
    if (publishRet != HCCL_SUCCESS) {
        return publishRet;
    }
    HCCL_INFO(
        "[JettyContext][Acquire] created shared jetty, handle[%llu]", static_cast<unsigned long long>(outCtx.handle));
    return HCCL_SUCCESS;
}

HcclResult JettyContext::PublishCreatedCtx(Ctx& createdCtx, Ctx& outCtx, std::function<HcclResult()>& outTpReleaseCb)
{
    std::lock_guard<std::mutex> lk(mtx_);
    // tpReleaseCb 先于其他字段写入：后续 memcpy_s/InnerToCtx 失败回滚走 DestroyJettyResources，
    // 届时已移交的 TP 引用随之释放，不泄漏。
    // 用 std::move 赋值（noexcept）：拷贝赋值 lambda 捕获 TpOwnership（含 IpAddress）必然堆分配、
    // 可能 bad_alloc，异常穿出时 creating 尚未清除（同 endpoint 后续 Acquire 等满 32s 超时），
    // 且回调随 createdCtx 析构丢弃、已移交的 TP 引用泄漏
    inner_.tpReleaseCb = std::move(createdCtx.tpReleaseCb);
    inner_.handle = createdCtx.handle;
    inner_.handlePtr = createdCtx.handlePtr;
    inner_.jettyId = createdCtx.jettyId;
    inner_.sqBuffVa = createdCtx.sqBuffVa;
    inner_.dbAddr = createdCtx.dbAddr;
    inner_.keySize = createdCtx.keySize;
    inner_.sqDepth = createdCtx.sqDepth;
    inner_.sqPiPtr = createdCtx.sqPiPtr;
    inner_.sqCiPtr = createdCtx.sqCiPtr;
    inner_.cqPiPtr = createdCtx.cqPiPtr;
    inner_.cqCiPtr = createdCtx.cqCiPtr;
    inner_.queueIndexMemSize = createdCtx.queueIndexMemSize;
    inner_.rdmaHandle = createdCtx.rdmaHandle;
    inner_.jfcHandle = createdCtx.jfcHandle;
    inner_.cqInfo = createdCtx.cqInfo;
    inner_.localPsn = createdCtx.localPsn;
    if (createdCtx.keySize > 0 && createdCtx.keySize <= Hccl::HRT_UB_QP_KEY_MAX_LEN) {
        errno_t cpyRet
            = memcpy_s(inner_.localQpKey, Hccl::HRT_UB_QP_KEY_MAX_LEN, createdCtx.localQpKey, createdCtx.keySize);
        if (cpyRet != EOK) {
            // memcpy_s 失败：inner_ 已写入部分字段（handle/PI·CI 指针等）但 valid 仍为 false，
            // 需销毁已写入的 device 资源避免泄漏，再清空 inner_ 让其他线程可重新创建。
            RollbackCreatedInnerLocked(outTpReleaseCb);
            HCCL_ERROR("[JettyContext][Acquire] memcpy_s localQpKey failed, ret[%d].", cpyRet);
            return HCCL_E_INTERNAL;
        }
    }
    inner_.valid = true;
    inner_.creating = false;
    inner_.refCount = 1;
    HcclResult innerToCtxRet = InnerToCtx(inner_, outCtx);
    if (innerToCtxRet != HCCL_SUCCESS) {
        RollbackCreatedInnerLocked(outTpReleaseCb);
        HCCL_ERROR("[JettyContext][Acquire] InnerToCtx failed, ret[%d].", innerToCtxRet);
        return innerToCtxRet;
    }
    cv_.notify_all();
    return HCCL_SUCCESS;
}

void JettyContext::RollbackCreatedInnerLocked(std::function<HcclResult()>& outTpReleaseCb)
{
    DestroyJettyResources(inner_, outTpReleaseCb);
    inner_ = Inner{};
    inner_.creating = false;
    cv_.notify_all();
}

HcclResult JettyContext::Release()
{
    // TP 释放回调移出后在锁外执行（避免与 TpManager 信息锁跨模块嵌套，见 DestroyJettyResources 注释）
    std::function<HcclResult()> tpReleaseCb;
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (!inner_.valid) {
            HCCL_WARNING("[JettyContext][Release] shared jetty already invalid, skip release.");
            return HCCL_SUCCESS;
        }
        if (inner_.refCount == 0) {
            HCCL_WARNING("[JettyContext][Release] refCount already 0, skip release.");
            return HCCL_SUCCESS;
        }
        inner_.refCount--;
        HCCL_INFO(
            "[JettyContext][Release] release shared jetty, handle[%llu], refCount[%u]",
            static_cast<unsigned long long>(inner_.handle), inner_.refCount);
        if (inner_.refCount == 0) {
            UnimportSharedRemoteJettys(inner_);
            DestroyJettyResources(inner_, tpReleaseCb);
            inner_ = Inner{};
        }
    }
    HcclResult tpRet = ExecuteTpReleaseCb(std::move(tpReleaseCb));
    if (tpRet != HCCL_SUCCESS) {
        // 共享 jetty 资源已销毁（不可回滚），但 TP 引用释放失败（useCnt 可能漏扣、条目常驻）：
        // 以告警码返回供调用方审计，不静默吞掉
        HCCL_ERROR(
            "[JettyContext][Release] shared jetty destroyed but tp release failed, ret[%d], tpHandle may leak in "
            "TpManager cache.",
            tpRet);
        return tpRet;
    }
    return HCCL_SUCCESS;
}

HcclResult JettyContext::AcquireSharedRemoteJetty(
    const uint8_t* remoteQpKey, uint32_t keySize, bool& needImport, uint64_t& handle, void*& handlePtr, uint32_t& tpn)
{
    CHK_PTR_NULL(remoteQpKey);
    CHK_PRT_RET(
        keySize == 0 || keySize > Hccl::HRT_UB_QP_KEY_MAX_LEN,
        HCCL_ERROR("[%s] invalid keySize[%u].", __func__, keySize), HCCL_E_PARA);

    std::lock_guard<std::mutex> lk(mtx_);
    CHK_PRT_RET(!inner_.valid, HCCL_ERROR("[%s] shared local jetty is invalid.", __func__), HCCL_E_INTERNAL);

    needImport = false;
    handle = 0;
    handlePtr = nullptr;
    tpn = 0;
    for (const auto& remoteCtx : inner_.remoteJettys) {
        if (!IsSameRemoteJetty(remoteCtx, remoteQpKey, keySize)) {
            continue;
        }
        if (remoteCtx.ready) {
            handle = remoteCtx.handle;
            handlePtr = remoteCtx.handlePtr;
            tpn = remoteCtx.tpn;
            HCCL_INFO(
                "[%s] reuse shared remote jetty, handle[%llu], tpn[%u].", __func__,
                static_cast<unsigned long long>(handle), tpn);
        }
        return HCCL_SUCCESS;
    }

    SharedRemoteJettyCtx remoteCtx;
    remoteCtx.remoteQpKey.assign(remoteQpKey, remoteQpKey + keySize);
    inner_.remoteJettys.push_back(std::move(remoteCtx));
    needImport = true;
    HCCL_INFO("[%s] reserve shared remote jetty import.", __func__);
    return HCCL_SUCCESS;
}

HcclResult JettyContext::PublishSharedRemoteJetty(
    const uint8_t* remoteQpKey, uint32_t keySize, uint64_t handle, void* handlePtr, uint32_t tpn)
{
    CHK_PTR_NULL(remoteQpKey);
    CHK_PTR_NULL(handlePtr);
    CHK_PRT_RET(
        keySize == 0 || keySize > Hccl::HRT_UB_QP_KEY_MAX_LEN || handle == 0,
        HCCL_ERROR(
            "[%s] invalid params, keySize[%u], handle[%llu].", __func__, keySize,
            static_cast<unsigned long long>(handle)),
        HCCL_E_PARA);

    std::lock_guard<std::mutex> lk(mtx_);
    for (auto& remoteCtx : inner_.remoteJettys) {
        if (!IsSameRemoteJetty(remoteCtx, remoteQpKey, keySize)) {
            continue;
        }
        remoteCtx.handle = handle;
        remoteCtx.handlePtr = handlePtr;
        remoteCtx.tpn = tpn;
        remoteCtx.ready = true;
        HCCL_INFO(
            "[%s] publish shared remote jetty, handle[%llu], tpn[%u].", __func__,
            static_cast<unsigned long long>(handle), tpn);
        return HCCL_SUCCESS;
    }
    HCCL_ERROR("[%s] shared remote jetty reservation not found.", __func__);
    return HCCL_E_NOT_FOUND;
}

void JettyContext::UnimportSharedRemoteJettys(Inner& inner) const
{
    if (inner.rdmaHandle == nullptr) {
        return;
    }
    for (auto& remoteCtx : inner.remoteJettys) {
        if (!remoteCtx.ready || remoteCtx.handle == 0) {
            continue;
        }
        Hccl::HrtRaUbUnimportJetty(
            static_cast<Hccl::RdmaHandle>(inner.rdmaHandle), static_cast<Hccl::TargetJettyHandle>(remoteCtx.handle));
        HCCL_INFO(
            "[JettyContext][%s] unimport shared remote jetty, handle[%llu].", __func__,
            static_cast<unsigned long long>(remoteCtx.handle));
        remoteCtx.handle = 0;
        remoteCtx.handlePtr = nullptr;
        remoteCtx.ready = false;
    }
}

void JettyContext::DestroyJettyResources(Inner& inner, std::function<HcclResult()>& outTpReleaseCb) const
{
    if (inner.handle != 0) {
        Hccl::HrtRaUbDestroyJetty(inner.handle);
        HCCL_INFO(
            "[JettyContext][%s] destroyed shared jetty, handle[%llu].", __func__,
            static_cast<unsigned long long>(inner.handle));
    }
    if (inner.jfcHandle != 0 && inner.rdmaHandle != nullptr) {
        Hccl::HrtRaUbDestroyJfc(static_cast<Hccl::RdmaHandle>(inner.rdmaHandle), inner.jfcHandle);
        HCCL_INFO(
            "[JettyContext][%s] destroyed shared jfc, jfcHandle[%llu].", __func__,
            static_cast<unsigned long long>(inner.jfcHandle));
    }
    if (inner.sqPiPtr != nullptr) {
        (void)hrtFree(inner.sqPiPtr);
        inner.sqPiPtr = nullptr;
    }
    if (inner.sqCiPtr != nullptr) {
        (void)hrtFree(inner.sqCiPtr);
        inner.sqCiPtr = nullptr;
    }
    if (inner.cqPiPtr != nullptr) {
        (void)hrtFree(inner.cqPiPtr);
        inner.cqPiPtr = nullptr;
    }
    if (inner.cqCiPtr != nullptr) {
        (void)hrtFree(inner.cqCiPtr);
        inner.cqCiPtr = nullptr;
    }
    // 防御性清空：销毁完成后句柄类字段不再有效，避免后续误用悬挂句柄
    // （Release 路径随后会 inner_ = Inner{}，此处清空主要保护 memcpy_s 失败分支等提前销毁场景）
    inner.handle = 0;
    inner.jfcHandle = 0;
    inner.rdmaHandle = nullptr;
    // TP 引用生命周期覆盖共享 jetty 生命周期：jetty/JFC/PI/CI 销毁后最后释放。
    // 仅移出不在本函数执行：本函数假定调用方持 mtx_，而回调链经适配层进入 TpManager 信息锁域
    // （ReleaseTpInfo → GetInfoCtxMutex），持锁执行会跨模块锁嵌套，须由调用方在锁作用域外经
    // ExecuteTpReleaseCb 统一执行（仍 try/catch）。先移出清空保证失败回滚 + 后续重创建
    // （Release 归零 → 再 Acquire）路径不重复释放。
    outTpReleaseCb = std::move(inner.tpReleaseCb);
    inner.tpReleaseCb = nullptr;
}

HcclResult JettyContext::InnerToCtx(const Inner& inner, Ctx& outCtx) const
{
    Ctx ctx{};
    ctx.handle = inner.handle;
    ctx.handlePtr = inner.handlePtr;
    ctx.jettyId = inner.jettyId;
    ctx.sqBuffVa = inner.sqBuffVa;
    ctx.dbAddr = inner.dbAddr;
    ctx.keySize = inner.keySize;
    ctx.sqDepth = inner.sqDepth;
    ctx.sqPiPtr = inner.sqPiPtr;
    ctx.sqCiPtr = inner.sqCiPtr;
    ctx.cqPiPtr = inner.cqPiPtr;
    ctx.cqCiPtr = inner.cqCiPtr;
    ctx.queueIndexMemSize = inner.queueIndexMemSize;
    ctx.rdmaHandle = inner.rdmaHandle;
    ctx.jfcHandle = inner.jfcHandle;
    ctx.cqInfo = inner.cqInfo;
    ctx.localPsn = inner.localPsn;
    if (inner.keySize > 0 && inner.keySize <= Hccl::HRT_UB_QP_KEY_MAX_LEN) {
        CHK_SAFETY_FUNC_RET(memcpy_s(ctx.localQpKey, Hccl::HRT_UB_QP_KEY_MAX_LEN, inner.localQpKey, inner.keySize));
    }
    outCtx = ctx;
    return HCCL_SUCCESS;
}

} // namespace hcomm
