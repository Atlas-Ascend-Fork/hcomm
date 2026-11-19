/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _SIM_HCCL_PROXY_COMMON_H_
#define _SIM_HCCL_PROXY_COMMON_H_

#include "hccl/hccl_types.h"
#include "sim_log.h"
#include <string>
#include <map>

#include <cstdint>
#include <dlfcn.h>
#include <map>
#include <mutex>

#include "storage/database_result.h"

namespace sim {
inline const std::map<HcclDataType, uint32_t> DATA_TYPE_SIZE_MAP
    = {{HcclDataType::HCCL_DATA_TYPE_INT8, 1},    {HcclDataType::HCCL_DATA_TYPE_INT16, 2},
       {HcclDataType::HCCL_DATA_TYPE_INT32, 4},   {HcclDataType::HCCL_DATA_TYPE_FP16, 2},
       {HcclDataType::HCCL_DATA_TYPE_FP32, 4},    {HcclDataType::HCCL_DATA_TYPE_INT64, 8},
       {HcclDataType::HCCL_DATA_TYPE_UINT64, 8},  {HcclDataType::HCCL_DATA_TYPE_UINT8, 1},
       {HcclDataType::HCCL_DATA_TYPE_UINT16, 2},  {HcclDataType::HCCL_DATA_TYPE_UINT32, 4},
       {HcclDataType::HCCL_DATA_TYPE_FP64, 8},    {HcclDataType::HCCL_DATA_TYPE_BFP16, 2},
       {HcclDataType::HCCL_DATA_TYPE_INT128, 16}, {HcclDataType::HCCL_DATA_TYPE_HIF8, 1},
       {HcclDataType::HCCL_DATA_TYPE_FP8E4M3, 1}, {HcclDataType::HCCL_DATA_TYPE_FP8E5M2, 1},
       {HcclDataType::HCCL_DATA_TYPE_FP8E8M0, 1}};

inline int GetDataTypeSize(HcclDataType dataType, uint32_t& size)
{
    auto iter = DATA_TYPE_SIZE_MAP.find(dataType);
    if (iter == DATA_TYPE_SIZE_MAP.end()) {
        return 1;
    }
    size = iter->second;
    return 0;
}

// RTLD_NEXT 无法穿透 torch_npu 等场景中以 RTLD_LOCAL 方式加载的真实库链,
// dlsym 失败时按 SONAME 显式 dlopen (如 libhccl.so / libhcomm.so) 后重试.
// 成功的 dlopen 句柄按 SONAME 缓存, 避免重复 dlopen 叠加引用计数; 失败不缓存, 下次调用重试.
//
// 注意: 必须使用 static(内部链接), 不能只用 inline。本头文件被多个 proxy .so
// (level0/level2 等) 共同包含, 若仅用 inline, 本函数会被编译成 weak 符号重复导出;
// 经 LD_PRELOAD("level0:level2") 的全局符号插入后, level2 里的调用会绑定到 level0
// 的那份拷贝, 使 dlsym(RTLD_NEXT) 的"当前对象"变成 level0, 从而把符号解析回 level2
// 自己的桩函数, 形成无限递归(stack overflow)。static 保证每个 .so 使用自身拷贝,
// RTLD_NEXT 的返回地址始终位于调用方 .so 内。
static inline void* DlsymRealWithFallback(const char* funcName, const char* fallbackSoName)
{
    dlerror();
    void* func = dlsym(RTLD_NEXT, funcName);
    if (func != nullptr) {
        return func;
    }
    // dlerror 返回内部缓冲区；后续动态库操作前复制，避免错误日志引用失效内存。
    const char* nextErrorMessage = dlerror();
    const std::string nextError = nextErrorMessage != nullptr ? nextErrorMessage : "unknown";

    static std::mutex handleMutex;
    static std::map<std::string, void*> handleCache;
    void* handle = nullptr;
    {
        std::lock_guard<std::mutex> lock(handleMutex);
        auto iter = handleCache.find(fallbackSoName);
        if (iter != handleCache.end()) {
            handle = iter->second;
        } else {
            handle = dlopen(fallbackSoName, RTLD_NOW | RTLD_LOCAL);
            if (handle != nullptr) {
                handleCache.emplace(fallbackSoName, handle);
            }
        }
    }

    if (handle != nullptr) {
        dlerror();
        func = dlsym(handle, funcName);
        if (func != nullptr) {
            HCCL_VM_INFO("dlsym {} resolved from {} fallback", funcName, fallbackSoName);
            return func;
        }
    }
    const char* fallbackErrorMessage = dlerror();
    const std::string fallbackError = fallbackErrorMessage != nullptr ? fallbackErrorMessage : "unknown";
    HCCL_VM_ERROR(
        "dlsym {} failed via RTLD_NEXT ({}), fallback {} ({})", funcName, nextError, fallbackSoName, fallbackError);
    return nullptr;
}

/// 地址分类（设备地址包含查询 src_type=DEV 且 dev_mapped_ptr<=addr<dev_mapped_end_ptr）。
/// 三态合同——"确定未命中"与"查询无法完成"是不同状态，不得互相冒充：
///   OK+true    查询完成且命中设备映射区间；
///   OK+false   查询完成且数据库明确 NOT_FOUND（host 地址，不是错误）；
///   其它 code  查询无法完成（CONFLICT、RESOURCE_EXHAUSTED、BACKEND_UNAVAILABLE、
///              协议/解码错误等），保留原错误码与诊断向上传播，调用方不得据此
///              判定地址类别，也不得视为未命中。
HcclSim::Storage::DbResult<bool> IsDeviceAddress(void* addr);

bool ParseKernelJson(const std::string& jsonPath, std::map<std::string, std::string>& out);

uint64_t GetVirPtrByDevPtr(uint64_t devAddr);

// CCU SQE arguments contain both device addresses and scalar values. Return
// zero for non-address arguments so callers can translate only mapped values.
uint64_t TryGetVirPtrByDevPtr(uint64_t devAddr);

uint64_t TransRemoteAddrToVirtualByRank(uint64_t devAddr, uint32_t rankId);

uint64_t TransRemoteAddrToVirtualByDeviceId(uint64_t devAddr, uint32_t deviceId);

void* GetDevMapperAddrByHostPtr(void* hostPtr);

bool IsAICPUExpMode();

} // namespace sim

// 查询真实 HcclComm 在当前进程中绑定的本地 Communicator 表行 id.
extern "C" bool SimFindHcclCommMember(HcclComm comm, uint64_t* communicatorId);
extern "C" bool SimFindHcclCommHandle(uint64_t communicatorId, HcclComm* comm);
#endif
