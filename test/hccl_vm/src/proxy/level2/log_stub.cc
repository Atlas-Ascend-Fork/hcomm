/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 日志染色: 模块 tag (须在 include sim_log.h 之前)
#undef HCCL_VM_MODULE
#define HCCL_VM_MODULE "HCCL_LOG"

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>

#include "sim_log.h"
#include "operation_data/operation_data_ops.h"

constexpr size_t LOG_MSG_BUFFER_SIZE = 1024;

std::once_flag log_initialized_flag;

namespace {

// 算法选择结果日志特征 (业务侧 INFO 级, 匹配取自实际日志样本):
//   新/旧选择器公共最终路径 (op_common.cc HcclGetAlgRes):
//     "[HcclGetAlgRes] algName=%s, engine=%d, opType=%d, algoTypes.size=%zu."
//   新选择器 (selector_engine.cc):
//     "SelectMinCost: selected algName=%s, engine=%s, cost=%f."
//   旧选择器风格 (selector_engine.cc/ExecuteSelector, 行尾无分隔符):
//     "... the selected algo type is %s"
constexpr char ALG_RES_MARKER[] = "[HcclGetAlgRes] algName=";
constexpr char SELECT_MIN_COST_MARKER[] = "SelectMinCost: selected algName=";
constexpr char OLD_SELECTOR_MARKER[] = "the selected algo type is ";

// 从日志消息提取算法选择结果, 未命中返回空串。命中 marker 后取算法名字段:
// 截到首个 ','/'.' (前两类) 或行尾 (旧选择器), 去首尾空白与残留标点
std::string ExtractAlgName(const char* msg)
{
    struct MarkerSpec {
        const char* marker;
        bool toEol;
    };
    const MarkerSpec specs[] = {
        {ALG_RES_MARKER, false},
        {SELECT_MIN_COST_MARKER, false},
        {OLD_SELECTOR_MARKER, true},
    };
    for (const auto& spec : specs) {
        const char* start = std::strstr(msg, spec.marker);
        if (start == nullptr) {
            continue;
        }
        const char* field = start + std::strlen(spec.marker);
        const char* end = field;
        if (spec.toEol) {
            while (*end != '\0') {
                ++end;
            }
        } else {
            while (*end != '\0' && *end != ',' && *end != '.') {
                ++end;
            }
        }
        std::string algName(field, end);
        size_t begin = algName.find_first_not_of(" \t\r\n");
        size_t last = algName.find_last_not_of(" \t\r\n.,;");
        if (begin == std::string::npos || last == std::string::npos || last < begin) {
            return {};
        }
        return algName.substr(begin, last - begin + 1);
    }
    return {};
}

// 算法选择结果截取 (方案二): 工具即日志底层接口实现者 (业务 DlogRecord
// weak 符号解析到本桩), 在消息进入工具日志输出 (受 log_config.yaml 级别
// 二级控制) 之前匹配提取, 回填当前算子的 opDetails.algorithmName ——
// 截取免疫工具日志级别与业务日志级别两级过滤
void TryCaptureAlgName(const char* msg)
{
    std::string algName = ExtractAlgName(msg);
    if (algName.empty()) {
        return;
    }
    (void)sim::operation::UpdateOpDetailAlgName(algName);
}

} // namespace

#ifdef __cplusplus
extern "C" {
#endif

int32_t CheckLogLevel(int32_t moduleId, int32_t logLevel)
{
    (void)moduleId;
    (void)logLevel;
    return static_cast<int32_t>(true);
}

// 业务日志级别门槛 (hccl log.cc HcclCheckLogLevel → weak
// acllogCheckDebugLevel, 未定义时回退 weak dlog_getlevel): 恒返 1 (允许),
// ProbeLogLevel 由此探测得最宽松级别, 保证 INFO 级算法选择日志必达本桩,
// 免疫环境内真实 dlog 库按级别过滤; 工具侧输出仍受 log_config.yaml
// 二级控制, 不会刷屏
int32_t acllogCheckDebugLevel(int32_t moduleId, int32_t logLevel)
{
    (void)moduleId;
    (void)logLevel;
    return 1;
}

void DlogPrintStub(int level, int moduleId, const char* msgBuffer)
{
    // 算法选择结果截取: 须在任何输出路径之前 (免疫工具日志级别)
    TryCaptureAlgName(msgBuffer);

    std::call_once(log_initialized_flag, []() {
        if (g_logger != nullptr) {
            return;
        }
        LogConfig config = LoadLogConfig("proxy");
        InitLogger(config);
    });

    switch (level) {
        case 0:
            HCCL_VM_DEBUG("[{}]{}", moduleId & 0xFF, msgBuffer);
            break;
        case 1:
            HCCL_VM_INFO("[{}]{}", moduleId & 0xFF, msgBuffer);
            break;
        case 2:
            HCCL_VM_WARN("[{}]{}", moduleId & 0xFF, msgBuffer);
            break;
        case 3:
            HCCL_VM_ERROR("[{}]{}", moduleId & 0xFF, msgBuffer);
            break;
        default:
            HCCL_VM_TRACE("[{}]{}", moduleId & 0xFF, msgBuffer);
    }
}

void DlogRecord(int moduleId, int level, const char* fmt, ...)
{
    char buffer[LOG_MSG_BUFFER_SIZE] = {0};
    va_list args;
    va_start(args, fmt);
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);

    DlogPrintStub(level, moduleId, buffer);
}

void _Z10DlogRecordiiPKcz(int moduleId, int level, const char* fmt, ...)
{
    char buffer[LOG_MSG_BUFFER_SIZE] = {0};
    va_list args;
    va_start(args, fmt);
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);

    DlogPrintStub(level, moduleId, buffer);
}

#ifdef __cplusplus
}
#endif
