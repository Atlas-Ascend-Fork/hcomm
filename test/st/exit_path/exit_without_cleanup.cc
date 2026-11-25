/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file exit_without_cleanup.cc
 * @brief 退出期专项用例：触发锚定层后不做任何显式释放，直接退出。
 *
 * 验证进程退出路径不发生 coredump（不崩即通过，由 run_exit_path_test.sh 断言退出码 0；
 * ASAN 构建时同时要求无 use-after-free）。BaseCommMgr 改为 Leaky Singleton，不会进行析构，由 OS 回收。
 *
 */

#include <cstdint>
#include <cstdio>

#include "hcomm_c_adpt.h"

// 构造早于 BaseCommMgr 首次 GetInstance（在 main 中触发），故析构晚于 BaseCommMgr——
// 若非 Leaky Singleton 则 BaseCommMgr 先析构（触发 DeInit 打 step 标记），本对象析构时访问已析构单例
class LateDestructCaller {
public:
    ~LateDestructCaller()
    {
        // 析构期调用 HcommResMgrInit，经 GetSharedJettyMgr -> BaseCommMgr::GetInstance() 访问单例
        // Leaky Singleton：单例存活，正常返回；非泄漏式（回归）：单例已析构，FindOrCreate 打 WARNING "already deinited"
        HcommResult ret = HcommResMgrInit();
        std::printf(
            "[exit_without_cleanup] LateDestructCaller dtor: HcommResMgrInit ret[%d] (leaky singleton alive)\n", ret);
        std::fflush(stdout);
    }
};

// 全局对象：main 返回后在静态析构期析构
LateDestructCaller g_lateDestructCaller;

int main()
{
    // 无卡环境走 noDevice 兜底分支，返回值不影响本用例，仅打印记录
    HcommResult ret = HcommResMgrInit();
    std::printf("[exit_without_cleanup] HcommResMgrInit ret[%d], exit without any explicit cleanup\n", ret);

    // 刻意不做任何显式释放直接退出；释放时序错误在此表现为非 0 退出码或 ASAN 报错
    // g_lateDestructCaller 的析构在 main 返回后由静态析构触发，验证 Leaky Singleton 在退出期仍可服务
    return 0;
}
