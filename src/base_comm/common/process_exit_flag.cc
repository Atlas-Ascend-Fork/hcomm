/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "process_exit_flag.h"

#include <atomic>
#include <cstdlib>

#include "log.h"

namespace hcomm {

std::atomic<bool>* InitExitFlag()
{
    auto* f = new std::atomic<bool>(false);
    // 给atexit注册钩子，在进程退出时把flag改为true
    if (std::atexit(&ProcessExitFlag::SetExiting) != 0) {
        HCCL_ERROR("[ProcessExitFlag] register atexit callback failed, exit-flag fallback disabled");
    }
    return f;
}

std::atomic<bool>& ProcessExitFlag::Ref()
{
    // 首次调用时初始化标志并注册退出回调，此后直接返回已有标志
    static std::atomic<bool>* flag = InitExitFlag();
    return *flag;
}

// 注册晚于CollCommMgr单例静态对象构造，退出期逆序执行。
// 这样可保证flag赋值为true先于 ~CollCommMgr 读取，释放thread资源时可跳过kernel下发
void ProcessExitFlag::RegisterExitHook() { (void)Ref(); }

void ProcessExitFlag::SetExiting() { Ref().store(true); }

bool ProcessExitFlag::IsExiting() { return Ref().load(); }

void ProcessExitFlag::ResetForTest() { Ref().store(false); }
} // namespace hcomm
