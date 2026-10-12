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
 * See the License for the specific file packaged with this Software for details.
 */

// 文件职责：进程级资源软限自动抬升（RLIMIT_NOFILE / RLIMIT_NPROC）。
//
// 背景：大规模仿真（如 1024 rank 全互联建链）单进程需要的文件描述符远超系统默认
// soft limit（常见 1024）；触顶会让尾部 rank 建链失败（HCCL 报 hcclRet 19），
// 历史做法是让客户改 /etc/security/limits.conf（需 root 且重新登录）或在会话内
// 手工 `ulimit -n`。本头把这一步收进模拟器自身，客户零手工。
//
// 机制与边界：
//   - 进程无需任何特权即可用 setrlimit 把自己的 soft limit 抬到 hard limit；
//     抬 high hard limit 本身才需要 CAP_SYS_RESOURCE，这里不强求——系统默认
//     hard 通常已足够高（systemd 默认 RLIMIT_NOFILE hard=524288）。
//   - setrlimit 只作用于调用进程本身，并被其后 fork/exec 的子进程继承：host
//     在 envInit 抬升后，mpirun/orted、checker/plugin、AICPU device 子进程随
//     继承；rank 进程另在 aclInit 边界各自抬升（不依赖继承，双保险）。
//   - 进程级、不落系统、不影响其它进程；进程退出即自然消失，无需也不应在退出
//     时“恢复”（子进程已在 fork 时继承，退出再降无意义且可能误伤晚 spawn 者）。
//   - best-effort：getrlimit/setrlimit 失败（如 EPERM）静默跳过，绝不阻断启动；
//     幂等，可重复调用。
#ifndef SIM_RESOURCE_LIMITS_H
#define SIM_RESOURCE_LIMITS_H

#include <sys/resource.h>

namespace HcclSim {

/// 把本进程 RLIMIT_NOFILE 与 RLIMIT_NPROC 的 soft limit 抬到各自 hard limit。
/// best-effort、幂等、不阻断启动；详见文件头注释。
inline void SimRaiseResourceLimits()
{
    const int resources[] = {RLIMIT_NOFILE, RLIMIT_NPROC};
    for (int resource : resources) {
        struct rlimit lim;
        if (::getrlimit(resource, &lim) != 0) {
            continue; // 读取失败：跳过该资源，不影响另一个，也不阻断启动
        }
        // soft 已达 hard（或 hard 为 INFINITY 且 soft 亦然）则无需抬升
        if (lim.rlim_cur < lim.rlim_max) {
            lim.rlim_cur = lim.rlim_max;
            (void)::setrlimit(resource, &lim); // 失败（EPERM 等）静默：保持原 soft
        }
    }
}

} // namespace HcclSim

#endif // SIM_RESOURCE_LIMITS_H
