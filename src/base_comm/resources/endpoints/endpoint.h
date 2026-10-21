/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ENDPOINT_H
#define ENDPOINT_H

#include <algorithm>
#include <memory>
#include <shared_mutex>
#include <vector>
#include <string>
#include "reged_mem_mgr.h"
#include "socket/socket.h"
#include "socket_handle_manager.h"
#include "rdma_handle_manager.h"
#include "hccp_hdc_manager.h"
#include "proc_reged_mem_mgr_cache.h"
#include "comm_queue_context/comm_queue_context.h"
#include "dfx/endpoint_monitor.h"
#include "server_socket_context/server_socket_context.h"
#include "mgr/endpoint_ctx_mgr.h"
#include "../../common/orion_adpt_utils.h"

namespace hcomm {

class EndpointMonitor;
/**
 * @note 职责：通信设备Endpoint的C++纯虚接口类，只承担 endpoint 生命周期与工厂职责。
 *       数据成员仅保留 endpointDesc_/monitorKeepAlive_（各子类无差别），其余全部移至各具体派生类。
 *       本端内存注册接口由 LocalRegedMemMgr 承载（经 GetLocalRegMemMgr() 访问，直达进程级
 *       mgr 或组合 mgr）；远端内存导入接口由 RemoteRegedMemMgr 承载（经 GetRemoteRegMemMgr()
 *       访问，仅 RoCE/Ub 族返回实例级 wrapper，hccs/aicpu_ts 返回组合 mgr，其余返回 nullptr）；
 *       Socket 监听接口由 ServerSocketContext 承载（经 GetServerSocketContext() 访问）；
 *       NIC 插件方法由 PluginEndpointHolder 承载；
 *       RegedMemMgr 缓存管理（AttachCache/ReleaseCache）移至各派生类 private。
 */
class Endpoint {
public:
    explicit Endpoint(const EndpointDesc& endpointDesc);

    virtual ~Endpoint();

    static HcclResult CreateEndpoint(const EndpointDesc& endpointDesc, std::unique_ptr<Endpoint>& endpointPtr);

    virtual HcclResult Init() = 0;

    // 返回本端内存 mgr（各派生类 override 返回自身成员）
    // 返回裸指针视图：生命周期由宿主 endpoint 保证，调用方不得在 endpoint 存活期外持有
    virtual LocalRegedMemMgr* GetLocalRegMemMgr() = 0;

    // 返回远端内存 mgr；不支持远端导入的 endpoint 不 override（返回 nullptr）
    virtual RemoteRegedMemMgr* GetRemoteRegMemMgr() { return nullptr; }

    // 本端 memHandle 归属记账：默认不记录，GetAllLocalMemHandles 维持 mgr 全量返回（行为不变）；
    // 共享进程级 mgr 的 endpoint（CpuUrma/Uboe/Urma/CpuRoce）override 三者，枚举只返回本实例注册的 handle；
    // UbMemEndpoint 同享进程级 mgr 但有意不接入：其 mgr 未实现枚举，UB_MEM 通道按 tag 取句柄
    // 枚举契约：返回实例/mgr 内部存储的裸指针，仅可同步立即消费，不得保存或跨越后续注册/注销使用
    virtual HcclResult RecordLocalMemHandle(void*) { return HCCL_SUCCESS; }
    virtual HcclResult RemoveLocalMemHandle(void*) { return HCCL_SUCCESS; }
    virtual HcclResult GetAllLocalMemHandles(void** memHandles, uint32_t* memHandleNum)
    {
        auto* mgr = GetLocalRegMemMgr();
        CHK_PTR_NULL(mgr);
        return mgr->GetAllMemHandles(memHandles, memHandleNum);
    }

    // 返回 RDMA 句柄（各派生类 override 返回自身 ctxHandle_）
    virtual void* GetRdmaHandle() = 0;
    virtual bool IsCtxHandleValid() const = 0;

    // 返回 ServerSocketContext（仅 Socket 相关类 override 返回非 nullptr）
    virtual ServerSocketContext* GetServerSocketContext() { return nullptr; }

    // 返回 CommQueueContext（仅 UB 类 endpoint override 返回非 nullptr，当前唯一产出为 JettyContext）
    virtual CommQueueContext* GetCommQueueContext() { return nullptr; }

    EndpointDesc GetEndpointDesc() { return endpointDesc_; }

    static HcclResult CheckFeature(const EndpointDesc& endpointDesc, HcommEndpointFeatureType featureType, bool& value);

    // Register：AttachMonitor 拷贝 GetHolder，再用持有的指针 RegisterToEndpointMonitor。
    // Destroy / 析构走 ReleaseEndpointMonitor。未 Register 的空指针直接返回。不要再调 GetHolder()。
    void AttachMonitor(s32 logicId);
    HcclResult RegisterToEndpointMonitor(s32 logicId, EndpointHandle handle);
    void ReleaseEndpointMonitor(EndpointHandle handle);

protected:
    static HcclResult CreateEndpointBase(const EndpointDesc& endpointDesc, std::unique_ptr<Endpoint>& endpointPtr);
    EndpointDesc endpointDesc_;
    std::shared_ptr<EndpointMonitor> monitorKeepAlive_;

    // 实例级归属记账共用实现：MgrT 提供跨域 FindHandleRecord 反查，RecordT 区分 Ub/Roce 记录元素类型；
    // 共享进程级 mgr 的 endpoint（Urma/CpuUrma/Uboe/CpuRoce）据此一行接入，逻辑保持各域一致；
    // UbMemEndpoint 同享 mgr 但不接入（原因见 RecordLocalMemHandle 注释）
    template <typename MgrT, typename RecordT>
    HcclResult RecordLocalMemHandleImpl(
        void* memHandle, const std::shared_ptr<LocalRegedMemMgr>& mgr, std::shared_mutex& mtx,
        std::vector<std::shared_ptr<RecordT>>& records)
    {
        CHK_PTR_NULL(memHandle);
        // 注册在进程级 mgr 完成后反查共享指针，记入本实例归属；转型失败返回明确错误码，防止 mgr 实际类型与 MgrT 不符时
        // UB
        std::shared_ptr<RecordT> record{};
        auto typedMgr = std::dynamic_pointer_cast<MgrT>(mgr);
        CHK_PTR_NULL(typedMgr);
        CHK_RET(typedMgr->FindHandleRecord(memHandle, &record));
        std::unique_lock<std::shared_mutex> lock(mtx);
        records.push_back(std::move(record));
        return HCCL_SUCCESS;
    }

    template <typename RecordT>
    HcclResult
    RemoveLocalMemHandleImpl(void* memHandle, std::shared_mutex& mtx, std::vector<std::shared_ptr<RecordT>>& records)
    {
        CHK_PTR_NULL(memHandle);
        std::unique_lock<std::shared_mutex> lock(mtx);
        // 归属校验：handle 非本实例注册（如误传其他 endpoint 的 handle）时拒绝，避免账本静默发散
        size_t erased = 0;
        records.erase(
            std::remove_if(
                records.begin(), records.end(),
                [memHandle, &erased](const auto& record) {
                    if (record.get() == memHandle) {
                        ++erased;
                        return true;
                    }
                    return false;
                }),
            records.end());
        CHK_PRT_RET(
            erased == 0,
            HCCL_ERROR("[Endpoint][%s] memHandle[%p] not registered by this endpoint", __func__, memHandle),
            HCCL_E_NOT_FOUND);
        return HCCL_SUCCESS;
    }

    template <typename RecordT>
    HcclResult GetAllLocalMemHandlesImpl(
        std::shared_mutex& mtx, std::vector<std::shared_ptr<RecordT>>& records, void** memHandles,
        uint32_t* memHandleNum)
    {
        CHK_PTR_NULL(memHandles);
        CHK_PTR_NULL(memHandleNum);
        std::shared_lock<std::shared_mutex> lock(mtx);
        *memHandleNum = static_cast<uint32_t>(records.size());
        // 返回实例内部存储裸指针：锁返回即失效，并发注册/注销可致扩容搬移，调用方须同步立即消费
        *memHandles = records.empty() ? nullptr : static_cast<void*>(records.data());
        return HCCL_SUCCESS;
    }
};

} // namespace hcomm
#endif // ENDPOINT_H
