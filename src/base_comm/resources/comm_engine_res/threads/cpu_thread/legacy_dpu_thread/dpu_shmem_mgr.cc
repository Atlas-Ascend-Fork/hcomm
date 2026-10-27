/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dpu_shmem_mgr.h"
#include <cstdlib>
#include <memory>
#include "hccl_common.h"
#include "securec.h"
#include "orion_adapter_rts.h"
#include "orion_adapter_hal.h"
#include "driver/ascend_hal.h"
#include "log.h"
#include "exception_util.h"

using Hccl::HcclException;

namespace hccl {

namespace {
    constexpr u64 ALIGN_4K = 4096U; // UB 场景向上对齐步长（保证注册基址 4K 对齐）
} // namespace

HcclResult DpuShmemMgr::Alloc(const std::string& tag, uint64_t size, int32_t deviceLogicId)
{
    // 同 tag 重复申请属调用方误用：须在任何申请动作之前拒绝，否则这一次注定失败的调用会白申请并注册
    // 一块新映射，旧块不动而新块无人持有，既泄漏内存又留下 device 侧注册残留
    if (Has(tag)) {
        HCCL_ERROR("[DpuShmemMgr][%s] tag[%s] already allocated", __func__, tag.c_str());
        return HCCL_E_PARA;
    }

    // 申请并注册一块 DPU 共享内存：
    // PCIE 用 HrtMalloc，UB 用 malloc + 4K 对齐；首字节作为信号标志位清 0。
    Hccl::DpuShmem shmem;
    CHK_RET(
        Hccl::HrtHalGetDeviceInfo(deviceLogicId, MODULE_TYPE_SYSTEM, INFO_TYPE_HD_CONNECT_TYPE, shmem.connectType_));
    HCCL_INFO("[DpuShmemMgr][%s] tag[%s] connectType_: %ld", __func__, tag.c_str(), shmem.connectType_);

    HcclResult ret = HCCL_SUCCESS;
    if (shmem.connectType_ == HOST_DEVICE_CONNECT_TYPE_PCIE) {
        // HrtMalloc 失败以异常上抛、不返回 nullptr，故本分支无判空（异常语义见头文件 Alloc 的 @note）
        shmem.va_ = Hccl::HrtMalloc(size, ACL_MEM_TYPE_HIGH_BAND_WIDTH);
        ret = Hccl::HrtHalHostRegister(shmem.va_, size, DEV_SVM_MAP_HOST, deviceLogicId, &shmem.accessVA_);
    } else if (shmem.connectType_ == HOST_DEVICE_CONNECT_TYPE_UB) {
        shmem.originVa_ = malloc(size + ALIGN_4K); // 保证足够的对齐余量（最多偏移 ALIGN_4K-1）
        CHK_PTR_NULL(shmem.originVa_);
        void* ptr = shmem.originVa_;
        std::size_t storeSize = 0; // 不预留对象空间，仅做指针向上对齐
        std::size_t space = static_cast<std::size_t>(size + ALIGN_4K);
        shmem.va_ = std::align(static_cast<std::size_t>(ALIGN_4K), storeSize, ptr, space);
        ret = Hccl::HrtHalHostRegister(shmem.va_, size, HOST_MEM_MAP_DEV_PCIE_TH, deviceLogicId, &shmem.accessVA_);
    } else {
        return HCCL_E_NOT_SUPPORT;
    }
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[DpuShmemMgr][%s] HrtHalHostRegister failed, ret: %d, connect type: %ld", __func__, ret,
            shmem.connectType_);
        if (shmem.va_ != nullptr) {
            if (shmem.connectType_ == HOST_DEVICE_CONNECT_TYPE_PCIE) {
                Hccl::HrtFree(shmem.va_);
            } else if (shmem.connectType_ == HOST_DEVICE_CONNECT_TYPE_UB) {
                free(shmem.originVa_);
                shmem.originVa_ = nullptr;
            }
            shmem.va_ = nullptr;
        }
        shmem.accessVA_ = nullptr;
        return ret;
    }

    // host 侧地址：UB 取 va_，否则取 accessVA_（与原 AllocAndRegKFCWorkSpace 口径一致）
    void* hostPtr = shmem.connectType_ == HOST_DEVICE_CONNECT_TYPE_UB ? shmem.va_ : shmem.accessVA_;
    errno_t cpyRet = memset_s(hostPtr, sizeof(uint8_t), 0, sizeof(uint8_t)); // 首字节作为信号标志位
    if (cpyRet != EOK) {
        HCCL_ERROR("[DpuShmemMgr][%s] set 0 failed: %d", __func__, cpyRet);
        ReleaseShmem(shmem, deviceLogicId); // 尚未登记进 map，调用方无从释放，就地回收
        return HCCL_E_MEMORY;
    }

    shmems_[tag] = shmem;
    return HCCL_SUCCESS;
}

void DpuShmemMgr::Free(const std::string& tag, int32_t deviceLogicId)
{
    auto it = shmems_.find(tag);
    if (it == shmems_.end()) {
        return;
    }
    ReleaseShmem(it->second, deviceLogicId);
    shmems_.erase(it);
}

void DpuShmemMgr::ReleaseShmem(Hccl::DpuShmem& shmem, int32_t deviceLogicId)
{
    if (shmem.va_ == nullptr) {
        return;
    }
    // 必须先 halHostUnregister 解除映射，再释放内存，否则 HrtFree 会因内存被 pin 住而异常
    HcclResult ret = Hccl::HrtHalHostUnregister(shmem.va_, deviceLogicId);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[DpuShmemMgr][%s] HrtHalHostUnregister failed, ret[%d]", __func__, ret);
    }
    if (shmem.connectType_ == HOST_DEVICE_CONNECT_TYPE_PCIE) {
        // HrtFree 失败经 MACRO_THROW 上抛，而本函数在析构路径上，异常穿出会直接 terminate，故就地捕获
        DECTOR_TRY_CATCH("DpuShmemMgr", Hccl::HrtFree(shmem.va_));
    } else if (shmem.connectType_ == HOST_DEVICE_CONNECT_TYPE_UB) {
        free(shmem.originVa_);
    }
}

void DpuShmemMgr::FreeAll(int32_t deviceLogicId)
{
    // 逐个释放：Free 内部会 erase，取 begin 的 tag 依次释放直至清空
    while (!shmems_.empty()) {
        Free(shmems_.begin()->first, deviceLogicId);
    }
}

void* DpuShmemMgr::HostPtr(const std::string& tag) const
{
    auto it = shmems_.find(tag);
    if (it == shmems_.end()) {
        return nullptr;
    }
    const Hccl::DpuShmem& shmem = it->second;
    return shmem.connectType_ == HOST_DEVICE_CONNECT_TYPE_UB ? shmem.va_ : shmem.accessVA_;
}

void* DpuShmemMgr::DevPtr(const std::string& tag) const
{
    // device 侧可见地址：UB 场景 host 内存映射到 device（HOST_MEM_MAP_DEV_PCIE_TH）取 accessVA_，
    // PCIE 场景 device HBM（DEV_SVM_MAP_HOST，SVM 下与 host 同址）取 va_；与 HostPtr 正好互补。
    auto it = shmems_.find(tag);
    if (it == shmems_.end()) {
        return nullptr;
    }
    const Hccl::DpuShmem& shmem = it->second;
    return shmem.connectType_ == HOST_DEVICE_CONNECT_TYPE_UB ? shmem.accessVA_ : shmem.va_;
}

const Hccl::DpuShmem* DpuShmemMgr::Get(const std::string& tag) const
{
    auto it = shmems_.find(tag);
    return it == shmems_.end() ? nullptr : &it->second;
}

} // namespace hccl
