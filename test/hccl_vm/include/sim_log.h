/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIM_LOG_H
#define SIM_LOG_H

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE // 保留全部调用点，由已绑定的后端决定输出。

#include <iostream>
#include <memory>
#include <sstream>
#include <sys/types.h>

#include "spdlog/spdlog.h"

#include "sim_log_backend.h"

extern spdlog::logger* g_logger;

namespace hcclvm {
template <typename... Args>
inline void cannLog(
    LogWriter writer, size_t moduleIndex, spdlog::level::level_enum level, spdlog::source_loc source, const char* tag,
    fmt::format_string<Args...> format, Args&&... args)
{
    // 参数在调用期间存活；只构造类型擦除视图，是否格式化由后端决定。
    writer(moduleIndex, level, source, tag, format, fmt::make_format_args(args...));
}
} // namespace hcclvm

struct LogConfig {
    int consoleLevel{2};                  // default info
    int fileLevel{1};                     // default debug
    size_t maxFileSize{50 * 1024 * 1024}; // 50MB
    size_t maxFiles{UINT16_MAX};
    std::string filePath{"logs"};
    std::string fileBaseName{"app_log"};
    std::string fileSuffix{".log"};
    bool enableCompress{false};
};

#ifdef HCCL_VM_MODULE
#define HCCL_VM_TAG "[HCCL-VM-" HCCL_VM_MODULE "]"
#else
#define HCCL_VM_TAG "[HCCL-VM]"
#endif

#ifdef HCCL_VM_MODULE
#define HCCLVM_MODULE_NAME HCCL_VM_MODULE
#else
#define HCCLVM_MODULE_NAME ""
#endif
// 初始化只为 cann 绑定 writer；stub 直接使用原 spdlog 模板接口，不做类型擦除或手工格式化。
// cann 在参数求值/格式化前比较启动时缓存阈值，不查询后端、环境变量或真库等级。
#define HCCLVM_LOG(LEVEL, SPD, ...)                                                                                  \
    do {                                                                                                             \
        const auto hcclvmWriter = ::hcclvm::g_logWriter;                                                             \
        if (hcclvmWriter != nullptr) {                                                                               \
            constexpr auto hcclvmModuleIndex = ::hcclvm::LogModuleIndex(HCCLVM_MODULE_NAME);                         \
            if (LEVEL >= ::hcclvm::g_cannModuleLevels[hcclvmModuleIndex]) {                                          \
                ::hcclvm::cannLog(                                                                                   \
                    hcclvmWriter, hcclvmModuleIndex, LEVEL, spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, \
                    HCCL_VM_TAG, __VA_ARGS__);                                                                       \
            }                                                                                                        \
        } else if (g_logger != nullptr) {                                                                            \
            SPD(g_logger, HCCL_VM_TAG __VA_ARGS__);                                                                  \
        }                                                                                                            \
    } while (0)

#define HCCL_VM_TRACE(...) HCCLVM_LOG(spdlog::level::trace, SPDLOG_LOGGER_TRACE, __VA_ARGS__)
#define HCCL_VM_DEBUG(...) HCCLVM_LOG(spdlog::level::debug, SPDLOG_LOGGER_DEBUG, __VA_ARGS__)
#define HCCL_VM_INFO(...) HCCLVM_LOG(spdlog::level::info, SPDLOG_LOGGER_INFO, __VA_ARGS__)
#define HCCL_VM_WARN(...) HCCLVM_LOG(spdlog::level::warn, SPDLOG_LOGGER_WARN, __VA_ARGS__)
#define HCCL_VM_ERROR(...) HCCLVM_LOG(spdlog::level::err, SPDLOG_LOGGER_ERROR, __VA_ARGS__)
#define HCCL_VM_CRITICAL(...) HCCLVM_LOG(spdlog::level::critical, SPDLOG_LOGGER_CRITICAL, __VA_ARGS__)

LogConfig LoadLogConfig(const std::string& process_name);
// device cann 模式在启动时直接用 Device 表主键确定目录；host/stub 忽略 deviceKey。
void InitLogger(const LogConfig& config, uint32_t deviceKey = 0);
void FlushLog();
void DeInitLogger();

#endif // SIM_LOG_H
