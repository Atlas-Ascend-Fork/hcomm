/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_PROCESS_EXIT_FLAG_H
#define HCOMM_PROCESS_EXIT_FLAG_H

#include <atomic>

namespace hcomm {

class ProcessExitFlag {
public:
    static void RegisterExitHook(); // 仅注册回调
    static void SetExiting();
    static bool IsExiting();
    // 仅 UT 使用：同进程多用例间复位退出标志；业务代码禁止调用
    static void ResetForTest();

private:
    static std::atomic<bool>& Ref();
};

} // namespace hcomm
#endif // HCOMM_PROCESS_EXIT_FLAG_H
