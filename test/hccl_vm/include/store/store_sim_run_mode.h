/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STORE_SIM_RUN_MODE_H
#define STORE_SIM_RUN_MODE_H

#include <cstdint>
#include <string>

namespace sim {
// RunModeConfig.mode 的互斥三态取值
enum RunModeValue : uint8_t {
    RUN_MODE_NORMAL = 0,
    RUN_MODE_CHECK_ONLY = 1,
    RUN_MODE_ESTIMATE = 2,
};

// 读 DB 的 RunModeConfig，返回是否仅校验模式，不缓存。表为空返回 false。
bool ProbeCheckOnlyMode();

// 进程内只读一次并缓存结果。
bool IsCheckOnlyMode();

// 读 DB 的 RunModeConfig，返回是否评估模式，不缓存。表为空返回 false。
bool ProbeEstimateMode();

// 进程内只读一次并缓存结果。惰性求值：首次调用发生在算子执行期，
// 此时 cmd 层已写入 RunModeConfig，避免了 .so 加载期读空表。
bool IsEstimateMode();

// 重置模式缓存，供 mode 命令切换后刷新。
void ResetRunModeCache();

// mode 值到日志可读名的映射：normal / check-only / estimate。
const char* ModeName(uint8_t mode);

// mode 值到子 bash 提示符的单一映射，保证各进程显示一致。
// 0 -> "(hvm)$> "，1 -> "(chk-only)$> "，2 -> "(estimate)$> "，未知值回退 "(hvm)$> "。
std::string GetRunModePrompt(uint8_t mode);

// 读 DB 的 RunModeConfig 后映射提示符。供 checker 等无法访问 cmd 层全局的进程使用。
std::string GetRunModePrompt();
} // namespace sim

#endif // STORE_SIM_RUN_MODE_H
