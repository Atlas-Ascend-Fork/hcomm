/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "shared_jetty_connection_adapter.h"
#include "dev_ub_connection.h"
#include "log.h"
#include "adapter_rts_common.h"
#include <exception>

namespace hcomm {

// 适配层当前实现：转发到 Hccl::DevUbConnection 的新增共享 jetty 方法。
// 后续演进：DevUbConnection 自 legacy/ 迁入 base_comm 后，本文件可直接操作 connection，
//                     届时移除对 dev_ub_connection.h 的 #include，base_comm 完全不依赖 legacy 实现细节。
// 注意：endpointTag 当前仅 builtin 路径传入，实际类型为 Endpoint*。
//       若将来 plugin 路径复用此函数，需将 acquireRemoteCb/publishRemoteCb 的构建提升到调用方，
//       避免适配层直接强转 void* 为 Endpoint* 造成类型不匹配。
HcclResult SetSharedJettyFieldsToConn(
    void* rawConnection, const JettyContext::Ctx& ctx, void* endpointTag, std::function<void(void*)> releaseCb)
{
    if (rawConnection == nullptr || endpointTag == nullptr) {
        return HCCL_E_PARA;
    }
    auto* endpoint = static_cast<Endpoint*>(endpointTag);
    auto acquireRemoteCb = [endpoint](
                               const uint8_t* remoteQpKey, uint32_t keySize, bool& needImport,
                               Hccl::TargetJettyHandle& handle, void*& handlePtr, uint32_t& tpn) -> HcclResult {
        JettyContext* jc = dynamic_cast<JettyContext*>(endpoint->GetCommQueueContext());
        CHK_PTR_NULL(jc);
        uint64_t cachedHandle = 0;
        HcclResult ret = jc->AcquireSharedRemoteJetty(remoteQpKey, keySize, needImport, cachedHandle, handlePtr, tpn);
        handle = static_cast<Hccl::TargetJettyHandle>(cachedHandle);
        return ret;
    };
    auto publishRemoteCb = [endpoint](
                               const uint8_t* remoteQpKey, uint32_t keySize, Hccl::TargetJettyHandle handle,
                               void* handlePtr, uint32_t tpn) -> HcclResult {
        JettyContext* jc = dynamic_cast<JettyContext*>(endpoint->GetCommQueueContext());
        CHK_PTR_NULL(jc);
        return jc->PublishSharedRemoteJetty(remoteQpKey, keySize, static_cast<uint64_t>(handle), handlePtr, tpn);
    };
    auto* conn = static_cast<Hccl::DevUbConnection*>(rawConnection);
    return conn->SetSharedJettyFields(
        ctx.handle, ctx.handlePtr, ctx.jettyId, ctx.sqBuffVa, ctx.dbAddr, ctx.localQpKey, ctx.keySize, ctx.sqDepth,
        static_cast<Hccl::JfcHandle>(ctx.jfcHandle), ctx.cqInfo, ctx.localPsn, endpointTag, std::move(releaseCb),
        std::move(acquireRemoteCb), std::move(publishRemoteCb));
}

HcclResult ExtractJettyInfoFromConn(void* rawConnection, JettyContext::Ctx& ctx)
{
    if (rawConnection == nullptr) {
        return HCCL_E_PARA;
    }
    auto* conn = static_cast<Hccl::DevUbConnection*>(rawConnection);
    Hccl::DevUbConnection::JettyInfo info;
    CHK_RET(conn->GetJettyInfo(info));
    ctx.handle = info.handle;
    ctx.handlePtr = info.handlePtr;
    ctx.jettyId = info.jettyId;
    ctx.sqBuffVa = info.sqBuffVa;
    ctx.dbAddr = info.dbAddr;
    ctx.keySize = info.keySize;
    ctx.sqDepth = info.sqDepth;
    ctx.rdmaHandle = info.rdmaHandle;
    ctx.jfcHandle = info.jfcHandle;
    ctx.cqInfo = info.cqInfo;
    ctx.localPsn = info.localPsn;
    // 严格按实际 keySize 拷贝，避免拷贝超出实际 keySize 的无效尾部
    if (info.keySize > Hccl::HRT_UB_QP_KEY_MAX_LEN) {
        HCCL_ERROR(
            "[SharedJettyConnectionAdapter::%s] invalid keySize[%u], max[%u].", __func__, info.keySize,
            Hccl::HRT_UB_QP_KEY_MAX_LEN);
        return HCCL_E_PARA;
    }
    CHK_SAFETY_FUNC_RET(memcpy_s(ctx.localQpKey, Hccl::HRT_UB_QP_KEY_MAX_LEN, info.localQpKey, info.keySize));
    return HCCL_SUCCESS;
}

HcclResult DetachConnJetty(void* rawConnection)
{
    if (rawConnection == nullptr) {
        return HCCL_E_PARA;
    }
    auto* conn = static_cast<Hccl::DevUbConnection*>(rawConnection);
    conn->DetachJetty();
    return HCCL_SUCCESS;
}

// 由 TpOwnership 快照构造 TP 引用释放回调：捕获 ownership 副本（ReleaseUbConnectionTp 参数为
// 非 const 引用、内部置 tpHandle=0，故 lambda 需 mutable 并经局部副本调用），传回释放结果供审计。
// 独立具名函数（外部链接、禁止内联）便于 UT 经 mock 注入构造失败，验证 DetachConnTpOwnership
// 的回滚路径；仅 .cc 内部使用，不进公共头文件（base_comm 调用方不感知 TpOwnership）
__attribute__((noinline)) std::function<HcclResult()>
BuildTpReleaseCallback(Hccl::DevUbConnection::TpOwnership ownership)
{
    return [ownership]() mutable -> HcclResult {
        Hccl::TpInfo tpInfo = ownership.tpInfo;
        return Hccl::ReleaseUbConnectionTp(
            ownership.devLogicId, ownership.locAddr, ownership.rmtAddr, ownership.tpProtocol, tpInfo, ownership.qos);
    };
}

HcclResult DetachConnTpOwnership(void* rawConnection, std::function<HcclResult()>& outReleaseTp)
{
    // 无条件先置空：所有失败路径（含空指针入参）输出参数一致为空，符合头文件
    // "无有效 TP 引用时置空"契约，避免调用方复用已持有回调的 std::function 时残留过期回调
    outReleaseTp = nullptr;
    if (rawConnection == nullptr) {
        return HCCL_E_PARA;
    }
    auto* conn = static_cast<Hccl::DevUbConnection*>(rawConnection);
    Hccl::DevUbConnection::TpOwnership ownership{};
    HcclResult ret = conn->DetachTpOwnership(ownership);
    if (ret == HCCL_E_NOT_FOUND) {
        // 无有效 TP 引用（tpHandle=0）或已移交过：析构路径本就跳过/不再释放，无需回调，保持旧行为
        return HCCL_SUCCESS;
    }
    if (ret != HCCL_SUCCESS) {
        // 真实错误上抛：releaseTpOnDestroy_ 保持 true，临时 connection 析构仍 ReleaseTp（退化为旧行为），
        // 由调用方（helper）告警并兜底，不静默吞掉
        HCCL_ERROR("[%s] DetachTpOwnership failed, ret[%d].", __func__, ret);
        return ret;
    }
    // 回调构造在 releaseTpOnDestroy_ 已翻转后执行（捕获 IpAddress 副本必然堆分配，可能 bad_alloc）：
    // 异常穿出将令 TP 引用既不随析构释放也无回调接管（useCnt 永久漏扣），且异常穿出 Acquire 会令
    // creating 卡死（同 endpoint 后续 Acquire 等满 32s 超时）。捕获后还原标志退化为旧行为，
    // 以错误码返回（不抛出），由调用方告警兜底
    try {
        outReleaseTp = BuildTpReleaseCallback(ownership);
    } catch (const std::exception& e) {
        HCCL_ERROR("[%s] construct tp release callback exception: %s, restore release-on-destroy.", __func__, e.what());
        outReleaseTp = nullptr;
        conn->RestoreTpOwnershipOnDestroy();
        return HCCL_E_INTERNAL;
    } catch (...) {
        HCCL_ERROR("[%s] construct tp release callback unknown exception, restore release-on-destroy.", __func__);
        outReleaseTp = nullptr;
        conn->RestoreTpOwnershipOnDestroy();
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

} // namespace hcomm
