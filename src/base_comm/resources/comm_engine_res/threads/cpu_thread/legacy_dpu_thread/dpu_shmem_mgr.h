/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DPU_SHMEM_MGR_H
#define DPU_SHMEM_MGR_H

#include <string>
#include <unordered_map>
#include "dpu_types.h"

namespace hccl {

/**
 * @brief DPU 共享内存管理器（独立可复用类，可与使用方解耦）。
 *        按 tag 管理「申请→注册→取 host 地址→解注册→释放」整段生命周期。
 *        从 LegacyDpuThread 抽出为独立实体，
 *        LegacyDpuThread 与后续扩展方各自持有一个实例、按 tag 申请各自所需的内存。
 * @note 申请/释放依赖 deviceLogicId（HrtHalHostRegister/HrtHalHostUnregister），
 *       由调用方在 npu ctx 下解析后传入；本类不感知 commId/devId。
 */
class DpuShmemMgr {
public:
    DpuShmemMgr() = default;
    ~DpuShmemMgr() = default;
    // 持有原生地址/注册映射，浅拷贝会导致双重释放，禁止拷贝与移动
    DpuShmemMgr(const DpuShmemMgr&) = delete;
    DpuShmemMgr& operator=(const DpuShmemMgr&) = delete;
    DpuShmemMgr(DpuShmemMgr&&) = delete;
    DpuShmemMgr& operator=(DpuShmemMgr&&) = delete;

    /**
     * @brief 按 tag 申请并注册一块共享内存（首字节清 0 作为信号标志位）
     * @param[in] tag 内存标签（如 Hccl::DPU_TAG / Hccl::DPU_TASK_EXCEPTION_TAG）
     * @param[in] size 申请尺寸
     * @param[in] deviceLogicId 设备逻辑 id（PCIE/UB 分支 HrtHalHostRegister 使用）
     * @return HcclResult 成功返回 HCCL_SUCCESS；tag 已存在返回 HCCL_E_PARA（调用方须先 Free）
     * @note PCIE 分支经 HrtMalloc 申请，其失败以异常上抛而非返回错误码，故本接口可能抛出异常：
     *       调用方须保证自身处于异常边界内。UB 分支用 malloc，失败仍走错误码路径返回 HCCL_E_PTR。
     */
    HcclResult Alloc(const std::string& tag, uint64_t size, int32_t deviceLogicId);

    /**
     * @brief 按 tag 解注册并释放共享内存（幂等：tag 不存在或未申请则直接返回）
     */
    void Free(const std::string& tag, int32_t deviceLogicId);

    /**
     * @brief 释放全部已申请共享内存
     */
    void FreeAll(int32_t deviceLogicId);

    /**
     * @brief 取 tag 对应的 host 侧地址（UB 取 va_，否则取 accessVA_）；tag 不存在返回 nullptr
     */
    void* HostPtr(const std::string& tag) const;

    /**
     * @brief 取 tag 对应的 device 侧可见地址（UB 取 accessVA_，PCIE 取 va_）；tag 不存在返回 nullptr
     */
    void* DevPtr(const std::string& tag) const;

    /**
     * @brief 取 tag 对应的共享内存描述（含 connectType_/va_/accessVA_）；tag 不存在返回 nullptr
     */
    const Hccl::DpuShmem* Get(const std::string& tag) const;

    /**
     * @brief 是否已登记该 tag
     */
    bool Has(const std::string& tag) const { return shmems_.find(tag) != shmems_.end(); }

    /**
     * @brief 是否存在任一块已申请（析构兜底判据）
     */
    bool AnyAllocated() const { return !shmems_.empty(); }

private:
    // 解注册并释放（不碰 map）：Free 与 Alloc 的失败回收路径共用
    void ReleaseShmem(Hccl::DpuShmem& shmem, int32_t deviceLogicId);

    std::unordered_map<std::string, Hccl::DpuShmem> shmems_;
};

} // namespace hccl

#endif // DPU_SHMEM_MGR_H
