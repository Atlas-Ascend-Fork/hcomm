/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 进程侧轮次槽位标记（临时方案）：<HCCL_VM_RESOURCE_ROOT>/round_slots 的第 (deviceKey) 个字节。
// 判据：本进程退出时往该字节写 1；Host 数非 0 字节，够 mock-comm 声明的 N 才发完成标志。
//   - 槽位身份 = 全局唯一的 deviceKey（Device 表主键），首次拿到即锁定，后续 setdevice 不再改变；
//   - 单字节 pwrite + 各自独立 offset，跨进程无冲突，不需要锁/mmap/位运算；
//   - 每轮 mock-comm 会清空文件，故不需要代数，也不考虑"上一轮残留进程"；
//   - 失败一律静默（漏报方向安全）：不占槽 → Host 数不够 → 框架超时。
#ifndef _SIM_HCCL_PROXY_ROUND_SLOT_H_
#define _SIM_HCCL_PROXY_ROUND_SLOT_H_

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>

namespace sim {
namespace round_slot {

    static int g_slotFd = -1;      // 保持打开到进程退出（写入依赖它，不显式 close）
    static uint64_t g_slotKey = 0; // 本进程占用的槽位 = deviceKey（= Device 表主键）
    static pid_t g_slotPid = 0;    // fork 防护：只有登记进程本人写自己的槽

    // on_exit 回调：退出链上执行，只用裸 syscall（选 on_exit 而非 atexit/析构：不受 dlclose 影响）
    static void WriteSlotMark(int /*status*/, void* /*arg*/)
    {
        if (g_slotPid == 0 || g_slotPid != getpid() || g_slotFd < 0 || g_slotKey == 0) {
            return; // fork 出来的子进程（exec 前）继承的回调不写父进程的槽
        }
        const char one = 1;
        ssize_t written = pwrite(g_slotFd, &one, 1, static_cast<off_t>(g_slotKey));
        (void)written; // 失败不重试：Host 数不够就不发标志
    }

    // 门控调用（aclrtSetDevice 里 deviceKey 确定之后）：只登记第一次；deviceKey 为 0 不登记
    static inline void HvmRoundSlotAttach(uint64_t deviceKey) noexcept
    {
        static bool attached = false;
        if (attached) {
            return; // 一个进程只认第一次拿到的 deviceKey
        }
        if (deviceKey == 0) {
            return;
        }
        attached = true;

        const char* root = getenv("HCCL_VM_RESOURCE_ROOT"); // Host 已 setenv，全链继承
        if (root == nullptr || root[0] == '\0') {
            return; // 非模拟器环境
        }
        char path[512];
        int len = snprintf(path, sizeof(path), "%s/round_slots", root);
        if (len <= 0 || len >= static_cast<int>(sizeof(path))) {
            return;
        }
        // 只写不读：文件由 Host 在 mock-comm 时清空；不存在 = 本轮未装载
        int fd = open(path, O_WRONLY | O_CLOEXEC);
        if (fd < 0) {
            return;
        }
        g_slotKey = deviceKey;
        g_slotFd = fd;
        g_slotPid = getpid();
        on_exit(WriteSlotMark, nullptr);
    }

} // namespace round_slot
} // namespace sim

#endif // _SIM_HCCL_PROXY_ROUND_SLOT_H_
