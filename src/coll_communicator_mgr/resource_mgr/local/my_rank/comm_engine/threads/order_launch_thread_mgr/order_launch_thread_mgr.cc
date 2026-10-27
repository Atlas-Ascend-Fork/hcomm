/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "order_launch_thread_mgr.h"
#include "hcomm_res.h"
#include "acl/acl_rt.h"
#include "orion_adapter_rts.h"
#include "adapter_rts_common.h"
#include "local_notify.h" // GE保序notify以LocalNotify对象管理（所有权转移模型，正常路径创建/销毁）
#include "coll_comm.h"
#include "sal_pub.h"
#include "comm_engine/thread/dfx/thread_dfx_pub.h"
#include "dfx/prof_utils_pub.h"
#include "comm_engine/thread/thread_c_adpt_pub.h"
#include "comm_engine/thread/thread_pub.h"

namespace hccl {

constexpr u32 ORDER_THREAD_NOTIFY_NUM = 1;

static u32 GetAicpuBlockNum(s32 deviceLogicId)
{
    int64_t coreNum = 0;
    aclError ret = aclrtGetDeviceInfo(static_cast<uint32_t>(deviceLogicId), ACL_DEV_ATTR_AICPU_CORE_NUM, &coreNum);
    HCCL_INFO(
        "[GetAicpuBlockNum] deviceLogicId[%d], aclrtGetDeviceInfo ret[%d], coreNum[%lld]", deviceLogicId, ret, coreNum);
    if (ret != ACL_SUCCESS || coreNum <= 0) {
        HCCL_WARNING(
            "[GetAicpuBlockNum] aclrtGetDeviceInfo is unsuccessful, ret[%d], coreNum[%lld], use default 1", ret,
            coreNum);
        return 1U;
    }
    return static_cast<u32>(coreNum);
}

// GE保序线程统一经标准接口释放（回收流等资源，确保销毁发生在context存活期内，避免推迟到
// 进程退出时context已失效才回收）；失败仅告警，不影响主流程
static void FreeGeThread(ThreadHandle handle)
{
    if (handle == 0) {
        return;
    }
    HcommResult ret = HcommThreadFree(&handle, 1);
    if (ret != HCCL_SUCCESS) {
        HCCL_WARNING("[OrderLaunchThreadMgr] HcommThreadFree GE thread[0x%llx] is unsuccessful, ret[%d]", handle, ret);
    }
}

/* ============================ OrderLaunchContextRes ============================ */

void OrderLaunchContextRes::DestroyResources()
{
    if (opbaseThread != 0) {
        HcommResult ret = HcommThreadFree(&opbaseThread, 1);
        if (ret != HCCL_SUCCESS) {
            HCCL_WARNING(
                "[OrderLaunchContextRes] HcommThreadFree opbaseThread[0x%llx] is unsuccessful, ret[%d]", opbaseThread,
                ret);
        }
        opbaseThread = 0;
    }
    if (aclgraphThread != 0) {
        HcommResult ret = HcommThreadFree(&aclgraphThread, 1);
        if (ret != HCCL_SUCCESS) {
            HCCL_WARNING(
                "[OrderLaunchContextRes] HcommThreadFree aclgraphThread[0x%llx] is unsuccessful, ret[%d]",
                aclgraphThread, ret);
        }
        aclgraphThread = 0;
    }
    resValid = false;
    HCCL_INFO("[OrderLaunchContextRes] resources destroyed, context[0x%llx]", context);
}

/* ============================ OrderLaunchThreadMgr ============================ */

OrderLaunchThreadMgr::OrderLaunchThreadMgr() {}

OrderLaunchThreadMgr::~OrderLaunchThreadMgr()
{
    std::unique_lock<std::mutex> lock(mutex_);
    Destroy();
}

void OrderLaunchThreadMgr::Destroy()
{
    for (auto& entry : contextResMap_) {
        entry.second.DestroyResources();
    }
    contextResMap_.clear();
    contextGroupsMap_.clear();
    groupCtxMap_.clear();

    for (auto& entry : groupAttachedThreadMap_) {
        if (entry.second != 0) {
            FreeGeThread(entry.second);
            entry.second = 0;
        }
    }
    groupAttachedThreadMap_.clear();
    hcomAttachedStreamMap_.clear();
    groupGraphMap_.clear();

    for (auto& entry : groupGeNotifys_) {
        // 唯一所有者：erase/clear即unique_ptr析构自动销毁对象与notify资源
        // （~LocalNotify→Destroy，底层失败由hrt层日志覆盖）
        entry.second.clear();
    }
    groupGeNotifys_.clear();
}

HcclResult OrderLaunchThreadMgr::RegisterOrderLaunch(const std::string& group)
{
    std::unique_lock<std::mutex> lock(mutex_);
    if (groupCtxMap_.find(group) != groupCtxMap_.end()) {
        HCCL_WARNING("%s skip, group[%s] has already been registered", __func__, group.c_str());
        return HCCL_SUCCESS;
    }
    groupCtxMap_.insert({group, UINT64_MAX});
    HCCL_INFO("%s success, group[%s]", __func__, group.c_str());
    return HCCL_SUCCESS;
}

HcclResult OrderLaunchThreadMgr::UnRegisterOrderLaunch(const std::string& group)
{
    std::unique_lock<std::mutex> lock(mutex_);
    auto it = groupCtxMap_.find(group);
    if (it == groupCtxMap_.end()) {
        HCCL_WARNING("%s skip, group[%s] has not been registered", __func__, group.c_str());
        return HCCL_SUCCESS;
    }

    u64 context = it->second;

    auto ctxIt = contextGroupsMap_.find(context);
    if (ctxIt != contextGroupsMap_.end()) {
        ctxIt->second.erase(group);
        if (ctxIt->second.empty()) {
            contextGroupsMap_.erase(ctxIt);
            auto resIt = contextResMap_.find(context);
            if (resIt != contextResMap_.end()) {
                resIt->second.DestroyResources();
                contextResMap_.erase(resIt);
            }
        }
    }

    groupCtxMap_.erase(it);

    // 不释放：线程由 g_ThreadMap 持有，L1 仅摘缓存引用（运行中释放有设备侧UAF风险）；
    // 最终随进程退出由 g_ThreadMap 静态析构回收（context探测保证退出期安静）
    groupAttachedThreadMap_.erase(group);
    // 同步清理 group->graphId 映射：避免已注销 group 凭残留映射经 GetHcomAttachedThreadByGroup
    // 复活（为其重建 notify/thread）；换图回来时 SetAttachedStream 会重建映射
    groupGraphMap_.erase(group);

    auto geNotifyIt = groupGeNotifys_.find(group);
    if (geNotifyIt != groupGeNotifys_.end()) {
        // 唯一所有者：erase即析构销毁对象与notify资源；借用的线程此后不得再使用（契约）
        groupGeNotifys_.erase(geNotifyIt);
    }

    HCCL_INFO("%s success, group[%s], context[0x%llx]", __func__, group.c_str(), context);
    return HCCL_SUCCESS;
}

HcclResult OrderLaunchThreadMgr::GetCurrentContext(u64& currentContext) const
{
    aclrtContext rtCtx = nullptr;
    aclError ret = aclrtGetCurrentContext(&rtCtx);
    CHK_PRT_RET(
        ret != ACL_SUCCESS, HCCL_ERROR("[%s]aclrtGetCurrentContext failed, ret[%d]", __func__, ret), HCCL_E_RUNTIME);
    currentContext = ReinterpretAs<u64>(rtCtx);
    CHK_PRT_RET(
        currentContext == UINT64_MAX, HCCL_ERROR("[%s]GetCurrentContext failed, context is INVALID_U64", __func__),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
}

HcclResult OrderLaunchThreadMgr::EnsureContextRes(u64 context)
{
    if (contextResMap_.find(context) == contextResMap_.end()) {
        contextResMap_.emplace(context, OrderLaunchContextRes());
        HCCL_INFO(
            "[OrderLaunchThreadMgr][%s] created new OrderLaunchContextRes for context[0x%llx]", __func__, context);
    }
    return HCCL_SUCCESS;
}

HcclResult OrderLaunchThreadMgr::SetAttachedStream(const std::string& group, u32 graphId, void* stream)
{
    CHK_PRT_RET(stream == nullptr, HCCL_ERROR("[%s] stream is nullptr, graphId[%u]", __func__, graphId), HCCL_E_PTR);

    std::unique_lock<std::mutex> lock(mutex_);

    // group 换图：旧图流构建的 thread 缓存失效，待下次获取时基于新图流重建。
    // 不做释放：线程由 g_ThreadMap 持有（HcommThreadAcquireByNotify 经 SaveThreads 登记），
    // L1 仅摘缓存引用；运行中释放会导致设备侧UAF（device core），线程最终随进程退出
    // 由 g_ThreadMap 静态析构回收（Stream/DeviceMem 析构链含context探测，退出期安静）。
    auto groupGraphIt = groupGraphMap_.find(group);
    if (groupGraphIt != groupGraphMap_.end() && groupGraphIt->second != graphId) {
        HCCL_INFO(
            "[%s] group[%s] graphId changed [%u] -> [%u], thread cache invalidated", __func__, group.c_str(),
            groupGraphIt->second, graphId);
        groupAttachedThreadMap_.erase(group);
    }

    auto streamIt = hcomAttachedStreamMap_.find(graphId);
    if (streamIt != hcomAttachedStreamMap_.end() && streamIt->second == stream) {
        groupGraphMap_[group] = graphId;
        HCCL_DEBUG(
            "%s reuse existing stream, group[%s], graphId[%u], stream[%p]", __func__, group.c_str(), graphId, stream);
        return HCCL_SUCCESS;
    }

    void* oldStream = nullptr;
    if (streamIt != hcomAttachedStreamMap_.end()) {
        oldStream = streamIt->second;
        // 图流变更：同 graphId 下各 group 基于旧流构建的 thread 全部失效，待下次获取时基于新流重建。
        // 同换图分支：不释放，线程由 g_ThreadMap 持有
        for (auto& entry : groupGraphMap_) {
            if (entry.second == graphId) {
                auto threadIt = groupAttachedThreadMap_.find(entry.first);
                if (threadIt != groupAttachedThreadMap_.end()) {
                    HCCL_INFO(
                        "[%s] graph stream changed, group[%s] thread[0x%llx] invalidated, graphId[%u]", __func__,
                        entry.first.c_str(), threadIt->second, graphId);
                    groupAttachedThreadMap_.erase(threadIt);
                }
                // 同换图分支：notify与group绑定不随流失效，下次建链复用（notify身份保持）
            }
        }
    }

    hcomAttachedStreamMap_[graphId] = stream;
    groupGraphMap_[group] = graphId;
    // 流ID仅用于日志定位（换图/换流排查）：查询失败不影响主流程，保持INVALID便于识别
    s32 newStreamId = INVALID_INT;
    (void)hrtGetStreamId(stream, newStreamId);
    HCCL_INFO(
        "%s success, group[%s], graphId[%u], oldStream[%p], stream[%p](id[%d]), streamMapSize[%zu]", __func__,
        group.c_str(), graphId, oldStream, stream, newStreamId, hcomAttachedStreamMap_.size());
    return HCCL_SUCCESS;
}

void OrderLaunchThreadMgr::UpdateGroupContextMapping(const std::string& group, u64 currentContext)
{
    auto groupIt = groupCtxMap_.find(group);
    if (groupIt != groupCtxMap_.end() && groupIt->second != currentContext) {
        HCCL_INFO(
            "[OrderLaunchThreadMgr][%s] group[%s] context updated: [0x%llx] -> [0x%llx]", __func__, group.c_str(),
            groupIt->second, currentContext);
        if (groupIt->second != UINT64_MAX) {
            auto oldCtxIt = contextGroupsMap_.find(groupIt->second);
            if (oldCtxIt != contextGroupsMap_.end()) {
                oldCtxIt->second.erase(group);
                if (oldCtxIt->second.empty()) {
                    contextGroupsMap_.erase(oldCtxIt);
                }
            }
        }
        groupIt->second = currentContext;
        contextGroupsMap_[currentContext].insert(group);
    } else if (groupIt == groupCtxMap_.end()) {
        groupCtxMap_[group] = currentContext;
        contextGroupsMap_[currentContext].insert(group);
    }
}

bool OrderLaunchThreadMgr::IsOrderLaunchDisabled(u64 currentContext)
{
    if (blockNum_ == 0U) {
        blockNum_ = GetAicpuBlockNum(static_cast<s32>(Hccl::HrtGetDevice()));
    }
    if (blockNum_ == 0U) {
        return false;
    }
    auto ctxGroupsIt = contextGroupsMap_.find(currentContext);
    u32 groupCount = (ctxGroupsIt != contextGroupsMap_.end()) ? static_cast<u32>(ctxGroupsIt->second.size()) : 0U;
    HCCL_DEBUG(
        "[OrderLaunchThreadMgr][%s] blockNum[%u] groupCount[%u] context[0x%llx]", __func__, blockNum_, groupCount,
        currentContext);
    return groupCount <= blockNum_;
}

HcclResult OrderLaunchThreadMgr::EnsureOrderThread(
    OrderThreadMode mode, const std::string& group, uint32_t notifyNumPerThread, ThreadHandle& thread)
{
    std::unique_lock<std::mutex> lock(mutex_);
    thread = 0;

    u64 currentContext = UINT64_MAX;
    CHK_RET(GetCurrentContext(currentContext));
    CHK_RET(EnsureContextRes(currentContext));

    auto& ctxRes = contextResMap_[currentContext];

    UpdateGroupContextMapping(group, currentContext);

    if (IsOrderLaunchDisabled(currentContext)) {
        HCCL_DEBUG(
            "[OrderLaunchThreadMgr][%s] order launch disabled, group[%s], context[0x%llx]", __func__, group.c_str(),
            currentContext);
        thread = 0;
        return HCCL_SUCCESS;
    }

    ThreadHandle& targetThread = (mode == OrderThreadMode::ACLGRAPH) ? ctxRes.aclgraphThread : ctxRes.opbaseThread;

    if (targetThread == 0) {
        HcommResult ret = HcommThreadAlloc(COMM_ENGINE_CPU_TS, 1, &notifyNumPerThread, &targetThread);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[%s] HcommThreadAlloc failed, ret[%d], mode[%u]", __func__, ret, static_cast<u8>(mode)),
            static_cast<HcclResult>(ret));

        ctxRes.resValid = true;
        // 线程首次创建审计（仅创建时打，避免每次acquire刷日志）：流ID经L0接口查询，
        // 失败保持INVALID便于识别，不影响主流程
        s32 streamId = INVALID_INT;
        if (targetThread != 0) {
            ThreadResTypeStream stream = nullptr;
            HcommResult resRet
                = HcommThreadResGetInfo(targetThread, THREAD_RES_TYPE_STREAM, sizeof(ThreadResTypeStream), &stream);
            if (resRet == HCCL_SUCCESS && stream != nullptr) {
                (void)hrtGetStreamId(stream, streamId);
            } else {
                HCCL_WARNING("[%s] get stream info failed, ret[%d], thread[0x%llx]", __func__, resRet, targetThread);
            }
        }
        HCCL_RUN_INFO(
            "[%s] success, comm[%s], mode[%u], thread[0x%llx], streamId[%d], "
            "notifyNumPerThread[%u]",
            __func__, group.c_str(), static_cast<u8>(mode), targetThread, streamId, notifyNumPerThread);
    } else {
        HCCL_DEBUG(
            "[OrderLaunchThreadMgr][%s] order thread already exists, context[0x%llx], thread[0x%llx]", __func__,
            currentContext, targetThread);
    }

    thread = targetThread;
    return HCCL_SUCCESS;
}

HcclResult OrderLaunchThreadMgr::EnsureDeviceOrderThread(
    CollComm* collComm, const std::string& group, uint32_t notifyNumPerThread, ThreadHandle& thread)
{
    std::unique_lock<std::mutex> lock(mutex_);
    thread = 0;

    CHK_PRT_RET(
        collComm == nullptr, HCCL_ERROR("[%s] collComm is null, group[%s]", __func__, group.c_str()), HCCL_E_PTR);
    CommEngineResMgr* engineResMgr = collComm->GetCommEngineResMgr();
    CHK_PRT_RET(
        engineResMgr == nullptr, HCCL_ERROR("[%s] engineResMgr is null, group[%s]", __func__, group.c_str()),
        HCCL_E_PTR);
    CHK_RET(engineResMgr->HcclDedicatedThreadAcquire(
        HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE, notifyNumPerThread, &thread));

    HCCL_DEBUG("[OrderLaunchThreadMgr][%s] device order thread[0x%llx], group[%s]", __func__, thread, group.c_str());
    return HCCL_SUCCESS;
}

ThreadHandle OrderLaunchThreadMgr::GetHcomAttachedThreadByGroup(const std::string& group)
{
    std::unique_lock<std::mutex> lock(mutex_);

    auto graphIt = groupGraphMap_.find(group);
    if (graphIt == groupGraphMap_.end()) {
        HCCL_WARNING(
            "[%s] graphId not found for group[%s], please call HcomSetAttachedStream first", __func__, group.c_str());
        return 0;
    }
    u32 graphId = graphIt->second;

    auto streamIt = hcomAttachedStreamMap_.find(graphId);
    if (streamIt == hcomAttachedStreamMap_.end()) {
        HCCL_ERROR(
            "[%s] attached stream not found for group[%s], graphId[%u], please call HcomSetAttachedStream first",
            __func__, group.c_str(), graphId);
        return 0;
    }

    // 通信域粒度缓存：本 group 基于图流 + 自身 notify 构建的 thread，同 graphId 下各 group 各自建链互不共用
    auto threadIt = groupAttachedThreadMap_.find(group);
    if (threadIt != groupAttachedThreadMap_.end() && threadIt->second != 0) {
        HCCL_DEBUG(
            "[%s] reuse existing thread[0x%llx], group[%s], graphId[%u]", __func__, threadIt->second, group.c_str(),
            graphId);
        return threadIt->second;
    }

    // notify与group绑定仅首次创建，换图重建线程时复用（借用不排他）。
    // 本函数返回ThreadHandle（0=失败），不能用CHK_*宏（会return HcclResult被隐式转成非零
    // 句柄），失败分支须显式return 0
    auto notifyIt = groupGeNotifys_.find(group);
    if (notifyIt == groupGeNotifys_.end()) {
        std::vector<std::unique_ptr<LocalNotify>> notifys;
        notifys.reserve(ORDER_THREAD_NOTIFY_NUM);
        for (u32 i = 0; i < ORDER_THREAD_NOTIFY_NUM; i++) {
            std::unique_ptr<LocalNotify> notify;
            EXCEPTION_CATCH(notify = std::make_unique<LocalNotify>(),
                            HCCL_ERROR("[%s] new LocalNotify failed, group[%s], index[%u]", __func__, group.c_str(), i);
                            return 0);
            HcclResult createRet = notify->Init(NotifyLoadType::HOST_NOTIFY);
            if (createRet != HCCL_SUCCESS) {
                HCCL_ERROR(
                    "[%s] LocalNotify init failed, ret[%d], group[%s], index[%u]", __func__, createRet, group.c_str(),
                    i);
                return 0; // Init失败自回滚，已建的随局部unique_ptr析构自动回收
            }
            notifys.push_back(std::move(notify));
        }
        // 创建即入map（管理器唯一所有者）：建链失败对象保留，下次建链直接复用不浪费
        notifyIt = groupGeNotifys_.emplace(group, std::move(notifys)).first;
        HCCL_INFO(
            "[%s] created ge notifys, group[%s], notifyNum[%u]", __func__, group.c_str(), ORDER_THREAD_NOTIFY_NUM);
    }
    if (notifyIt->second.empty()) {
        HCCL_ERROR("[%s] ge notify list is empty for group[%s]", __func__, group.c_str());
        return 0;
    }

    // 借用注入：临时裸指针数组传给线程（C ABI面为void**；纯借用契约——对象所有权始终归
    // 本管理器，线程仅拷贝引用，成败均不触碰对象生命周期）
    std::vector<LocalNotify*> rawNotifys;
    rawNotifys.reserve(notifyIt->second.size());
    for (const auto& notify : notifyIt->second) {
        rawNotifys.push_back(notify.get());
    }

    ThreadHandle thread = 0;
    HcommResult ret = HcommThreadAcquireByNotify(
        static_cast<rtStream_t>(streamIt->second), ReinterpretAs<void**>(rawNotifys.data()),
        static_cast<uint32_t>(rawNotifys.size()), &thread);
    if (ret != HCCL_SUCCESS) {
        // 线程仅借用（其内部失败路径析构只清引用不碰对象）；对象保留在map中供下次建链复用
        HCCL_ERROR("[%s] HcommThreadAcquireByNotify failed, ret[%d], group[%s]", __func__, ret, group.c_str());
        return 0;
    }

    groupAttachedThreadMap_[group] = thread;
    HCCL_INFO("[%s] success, group[%s], graphId[%u], thread[0x%llx]", __func__, group.c_str(), graphId, thread);
    return thread;
}

HcclResult OrderLaunchThreadMgr::RegisterThreadToComm(CollComm* collComm, ThreadHandle thread) const
{
    CommEngineResMgr* engineResMgr = collComm->GetCommEngineResMgr();
    if (engineResMgr != nullptr) {
        CHK_RET(engineResMgr->RegisterOrderLaunchThread(thread));
    }
    return HCCL_SUCCESS;
}

HcclResult OrderLaunchThreadMgr::RegisterDfx(
    CollComm* collComm, HcclDedicatedThreadType useType, ThreadHandle thread, u64 beginTime,
    const std::string& commId) const
{
    if (useType == HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE) {
        HcclCommDfx* hcclCommDfx = collComm->GetHcclCommDfx();
        if (hcclCommDfx != nullptr) {
            const std::string kernelName = "RunAicpuIndOpThreadInit";
            CHK_RET(hcclCommDfx->ReportKernel(beginTime, commId, kernelName, SalGetTid(), false));
            HCCL_DEBUG("[%s] DEVICE order launch thread ReportKernel done, comm[%s]", __func__, commId.c_str());
        }
    } else {
        std::function<HcclResult(u32, u32, const Hccl::TaskParam&, u64)> dfxCallback
            = [](u32 streamId, u32 taskId, const Hccl::TaskParam& taskParam, u64 handle) {
                  (void)streamId;
                  (void)taskId;
                  (void)taskParam;
                  (void)handle;
                  return HCCL_SUCCESS;
              };
        int ret = HcommThreadRegisterDfx(thread, dfxCallback);
        if (ret != 0) {
            HCCL_WARNING("[%s] HcommThreadRegisterDfx is unsuccessful, ret[%d], thread[0x%llx]", __func__, ret, thread);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult OrderLaunchThreadMgr::OrderLaunchThreadAcquire(
    HcclDedicatedThreadType useType, CollComm* collComm, const std::string& group, uint32_t notifyNumPerThread,
    ThreadHandle& thread)
{
    thread = 0;

    HCCL_DEBUG(
        "[%s] begin, HcclDedicatedThreadType[%d], group[%s], notifyNumPerThread[%u]", __func__,
        static_cast<s32>(useType), group.c_str(), notifyNumPerThread);

    u64 beginTime = hcomm::GetProfCycleTime();

    switch (useType) {
        case HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_OPBASE: {
            HcclResult ret = EnsureOrderThread(OrderThreadMode::OPBASE, group, notifyNumPerThread, thread);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS, HCCL_ERROR("[%s] EnsureOrderThread OPBASE failed, ret[%d]", __func__, ret), ret);
            break;
        }
        case HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_ACLGRAPH: {
            HcclResult ret = EnsureOrderThread(OrderThreadMode::ACLGRAPH, group, notifyNumPerThread, thread);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS, HCCL_ERROR("[%s] EnsureOrderThread ACLGRAPH failed, ret[%d]", __func__, ret), ret);
            break;
        }
        case HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_GE: {
            ThreadHandle th = GetHcomAttachedThreadByGroup(group);
            thread = th;
            break;
        }
        case HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE: {
            HcclResult ret = EnsureDeviceOrderThread(collComm, group, notifyNumPerThread, thread);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS, HCCL_ERROR("[%s] EnsureDeviceOrderThread failed, ret[%d]", __func__, ret), ret);
            break;
        }
        default:
            HCCL_ERROR(
                "[%s] invalid HcclDedicatedThreadType[%d] for order launch", __func__, static_cast<s32>(useType));
            return HCCL_E_PARA;
    }

    if (thread != 0 && collComm != nullptr) {
        if (useType != HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE) {
            CHK_RET(RegisterThreadToComm(collComm, thread));
        }
        CHK_RET(RegisterDfx(collComm, useType, thread, beginTime, group));
    }

    HCCL_INFO("[%s] success, HcclDedicatedThreadType[%d], thread[0x%llx]", __func__, static_cast<s32>(useType), thread);
    return HCCL_SUCCESS;
}

} // namespace hccl
