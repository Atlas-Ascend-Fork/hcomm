/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIM_LOG_BACKEND_H
#define SIM_LOG_BACKEND_H

#include <cstdarg>
#include <cstdint>
#include <string_view>

#include "spdlog/common.h"

namespace hcclvm {

enum class LogBackend { Stub, Cann };
LogBackend GetLogBackend();
inline bool IsCannBackend() { return GetLogBackend() == LogBackend::Cann; }
// 角色由日志实现所属目标决定，避免不同 DSO 的 inline 定义互相覆盖。
bool IsDeviceRole();

// 编译期将模拟器 tag 归入已有 CANN 模块；打印热路径不查表。
struct CannLogModule {
    int32_t id;
    const char* name;
    std::string_view tags;
};
inline constexpr CannLogModule kCannLogModules[] = {
    {0, "SLOG", ""},
    {3, "HCCL",
     "|CDM_STUB|CDR_STUB|COMM_OP|COMM_STUB|HCOMM_DEV_STUB|HCOMM_LOCAL|HCOMM_STUB|OP_STUB|OTHER_OP|SCATTER_STUB|TOPO|"
     "TOPO_ADDR_STUB|TOPO_QUERY_STUB|"},
    {28, "HCCP", "|BR_STUB|HCCP_STUB|IBV_DPU_STUB|RASOCKET_STUB|"},
    {5, "CCU",
     "|CCU_CTRL_STUB|CCU_DATA_STUB|CCU_FALLBACK_STUB|CCU_KERNEL_SIM|CCU_L1_COMMON|CCU_L1_RES|CCU_RES_STUB|CCU_STUB|DPU_"
     "CTRL_STUB|DPU_DATA_STUB|DPU_EXEC|DPU_KERNEL_MGR_STUB|DPU_RES|DPU_RPC_DEV|"},
    {7, "RUNTIME",
     "|ACLGRAPH_STUB|CTX_STUB|EVENT_STUB|EXEC_STUB|GRAPH_CAPTURE|MEM_STUB|MODEL_STUB|NEW_STUB|NOTIFY_STUB|RTS_STUB|"
     "STREAM_STUB|TSD_STUB|"},
    {76, "ACLRTC", "|ACLRT_STUB|"},
    {10, "DRV", "|HAL_STUB|"},
    {23, "KERNEL", "|KERNEL_STUB|"},
    {31, "PROFILING", "|DFX_STUB|"},
    {56, "AIVECTOR", "|AIV|"},
    {63, "OP", "|OPERATION_DATA|"},
};
inline constexpr size_t kLogModuleCount = sizeof(kCannLogModules) / sizeof(kCannLogModules[0]);
constexpr size_t LogModuleIndex(std::string_view tag)
{
    if (tag.empty())
        return 0;
    for (size_t i = 1; i < kLogModuleCount; ++i) {
        const auto tags = kCannLogModules[i].tags;
        size_t begin = 1;
        while (begin < tags.size()) {
            const auto end = tags.find('|', begin);
            if (tags.substr(begin, end - begin) == tag)
                return i;
            begin = end + 1;
        }
    }
    return 0;
}

// 初始化只为 cann 绑定写入入口；stub 保持空指针并直接走 spdlog 模板接口。
using LogWriter = void (*)(
    size_t moduleIndex, spdlog::level::level_enum level, spdlog::source_loc source, const char* tag,
    fmt::string_view format, fmt::format_args args);
extern LogWriter g_logWriter;
// 在发布 cann writer 前从环境变量初始化；热路径仅按常量下标读取并比较。
extern spdlog::level::level_enum g_cannModuleLevels[kLogModuleCount];
LogWriter InitializeCannLog();
void FlushCannLog();
int32_t ForwardNativeCheckLogLevel(int32_t moduleAndType, int32_t level);
bool ForwardNativeDlogRecord(int32_t moduleAndType, int32_t level, const char* format, va_list args);
} // namespace hcclvm
#endif // SIM_LOG_BACKEND_H
