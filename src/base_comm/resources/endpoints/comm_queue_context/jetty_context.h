/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef JETTY_CONTEXT_H
#define JETTY_CONTEXT_H

#include <cstdint>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <vector>
#include "hccl/hccl_types.h"
#include "hcomm_res_defs.h"
#include "rdma_handle_manager.h"
#include "comm_queue_context.h"

namespace hcomm {

/**
 * @note 职责：共享 Jetty 数据面资源的内聚集合（jetty 句柄 / QP key / PI·CI device 内存 / JFC / 远端 jetty 缓存）。
 *       Endpoint（控制面）持 unique_ptr<JettyContext> 延迟创建，控制面职责不因共享 jetty 特性膨胀。
 *       引用计数 + condition_variable 管理同 endpoint 下多 channel 复用，替代 sleep 轮询。
 */
class JettyContext : public CommQueueContext {
public:
    struct SharedRemoteJettyCtx {
        std::vector<uint8_t> remoteQpKey{};
        uint64_t handle{0};
        void* handlePtr{nullptr};
        uint32_t tpn{0};
        bool ready{false};
    };

    /**
     * @brief jetty 资源字段集合。channel 通过 Acquire 取得 Ctx 视图，
     *        读取 PI/CI 指针绑给 transport，不持有资源 ownership。
     * @note tpHandle 不在此共享：一对多场景下各主 connection 到不同对端需各自申请 tpHandle，
     *       否则对端 import 时 peerTpHandle 路由不匹配。仅 jetty/SQ/JFC/CQ/psn 共享。
     * @note 同步约束：本结构与 Inner 的共享字段一一对应，新增/修改字段须同步四处：
     *       Ctx、Inner、InnerToCtx（jetty_context.cc）、Acquire 第三段（PublishCreatedCtx）逐字段拷贝，
     *       漏改会静默丢字段。例外：tpReleaseCb 仅 provider→Inner 单向传递（Acquire 第三段），
     *       InnerToCtx 不回填——channel 视图不消费该字段。
     */
    struct Ctx {
        Hccl::JettyHandle handle{0};
        void* handlePtr{nullptr};
        uint32_t jettyId{0};
        uint64_t sqBuffVa{0};
        uint64_t dbAddr{0};
        uint8_t localQpKey[Hccl::HRT_UB_QP_KEY_MAX_LEN]{0};
        uint32_t keySize{0};
        uint32_t sqDepth{0};
        void* sqPiPtr{nullptr};
        void* sqCiPtr{nullptr};
        void* cqPiPtr{nullptr};
        void* cqCiPtr{nullptr};
        uint64_t queueIndexMemSize{0};
        void* rdmaHandle{nullptr};
        uint64_t jfcHandle{0};
        Hccl::CqCreateInfo cqInfo{};
        uint32_t localPsn{0}; // 共享 jetty 统一 psn：临时 connection 生成后存入，主 connection 复用，避免多 connection
                              // 各自 GenerateLocalPsn 导致 import 同一 TP 对时 psn 互相覆盖
        std::function<HcclResult()> tpReleaseCb{nullptr}; // 首建临时 connection 移交的 TP 引用释放回调，
                                                          // 由 provider 填充、Acquire 存入 Inner 保管；
                                                          // 返回释放结果（NOT_FOUND 等失败为键漂移审计信号）
    };

    JettyContext() = default;
    ~JettyContext() override;

    JettyContext(const JettyContext&) = delete;
    JettyContext& operator=(const JettyContext&) = delete;

    /**
     * @brief 获取或创建共享 jetty。命中复用 refCount++；未命中调 provideCtx 创建并缓存。
     *        用 condition_variable 等待并发创建者，替代 sleep 轮询。
     * @param[in] provideCtx 创建回调（首次时调用，回调内创建 jetty 并填入 ctx）
     * @param[out] outCtx 输出的 jetty 上下文视图
     */
    HcclResult Acquire(const std::function<HcclResult(Ctx&)>& provideCtx, Ctx& outCtx);

    /** 释放共享 jetty 引用，refCount-- 归 0 时销毁 jetty 并清空 ctx */
    HcclResult Release();

    /**
     * 查询或预约远端共享 jetty。三种返回场景：
     *   - 已发布(ready=true)：needImport=false, handle!=0，调用方直接复用 handle
     *   - 已预约未发布(ready=false)：needImport=false, handle=0，调用方应进入等待，不使用 handle
     *   - 首次预约：needImport=true, handle=0，调用方应执行 import 并随后调 PublishSharedRemoteJetty
     */
    HcclResult AcquireSharedRemoteJetty(
        const uint8_t* remoteQpKey, uint32_t keySize, bool& needImport, uint64_t& handle, void*& handlePtr,
        uint32_t& tpn);

    HcclResult PublishSharedRemoteJetty(
        const uint8_t* remoteQpKey, uint32_t keySize, uint64_t handle, void* handlePtr, uint32_t tpn);

private:
    // @note 同步约束：共享字段与 Ctx 一一对应（见 Ctx 的同步注释），多出的 refCount/valid/creating/remoteJettys
    // 为内部状态
    struct Inner {
        Hccl::JettyHandle handle{0};
        void* handlePtr{nullptr};
        uint32_t jettyId{0};
        uint64_t sqBuffVa{0};
        uint64_t dbAddr{0};
        uint8_t localQpKey[Hccl::HRT_UB_QP_KEY_MAX_LEN]{0};
        uint32_t keySize{0};
        uint32_t sqDepth{0};
        uint32_t refCount{0};
        bool valid{false};
        bool creating{false};
        void* sqPiPtr{nullptr};
        void* sqCiPtr{nullptr};
        void* cqPiPtr{nullptr};
        void* cqCiPtr{nullptr};
        uint64_t queueIndexMemSize{0};
        void* rdmaHandle{nullptr};
        uint64_t jfcHandle{0};
        Hccl::CqCreateInfo cqInfo{};
        uint32_t localPsn{0}; // 共享 jetty 统一 psn（与 Ctx::localPsn 对应）
        std::vector<SharedRemoteJettyCtx> remoteJettys{};
        std::function<HcclResult()> tpReleaseCb{nullptr}; // 首建临时 connection 移交的 TP 引用释放回调（内部状态，
                                                          // 生命周期与共享 jetty 一致：随 DestroyJettyResources 移出，
                                                          // 由调用方在锁作用域外执行，返回释放结果供审计）
    };

    void UnimportSharedRemoteJettys(Inner& inner) const;
    /**
     * @brief 销毁 Inner 中的共享 jetty device 资源（jetty/JFC/PI·CI 内存），并把 TP 引用释放回调
     *        移出到 outTpReleaseCb。假定调用方已持有 mtx_；回调不在本函数内执行——
     *        回调链经适配层进入 TpManager 信息锁域（ReleaseTpInfo → GetInfoCtxMutex），持 mtx_ 执行
     *        会将单 endpoint 临界区与全设备同协议 TP 信息操作串行化，须由调用方在锁作用域外统一执行。
     * @param[out] outTpReleaseCb 移出的 TP 引用释放回调（inner 中同步清空，可为空）
     */
    void DestroyJettyResources(Inner& inner, std::function<HcclResult()>& outTpReleaseCb) const;
    HcclResult InnerToCtx(const Inner& inner, Ctx& outCtx) const;

    /**
     * @brief Acquire 第三段实现（内部持 mtx_）：把 createdCtx 字段搬运入 Inner 并发布
     *        （valid/creating/refCount/cv_.notify_all）。
     * @param[out] outTpReleaseCb 失败回滚时经 DestroyJettyResources 移出的 TP 释放回调，
     *             由调用方在锁作用域外执行——回调链经适配层进入 TpManager 信息锁域，禁止持 mtx_ 执行。
     * @note 失败路径统一走 RollbackCreatedInnerLocked 后返回错误码，本方法不执行回调。
     */
    HcclResult PublishCreatedCtx(Ctx& createdCtx, Ctx& outCtx, std::function<HcclResult()>& outTpReleaseCb);

    /**
     * @brief Acquire 第三段失败回滚（假定调用方持 mtx_）：销毁已写入的 device 资源
     *        （TP 释放回调移出至 outTpReleaseCb）、清空 Inner、清 creating 标记并唤醒等待方。
     */
    void RollbackCreatedInnerLocked(std::function<HcclResult()>& outTpReleaseCb);

    mutable std::mutex mtx_;
    std::condition_variable cv_;
    Inner inner_{};
};

} // namespace hcomm

#endif // JETTY_CONTEXT_H
