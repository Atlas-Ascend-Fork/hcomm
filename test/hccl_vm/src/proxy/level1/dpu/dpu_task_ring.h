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
 * Description: HostDPU 北向劫持（level1）设备任务转发环消费者接口。
 *              环布局与协议见 dpu_level1_common.h 第 1b 节；本文件只暴露
 *              host 执行器侧的两个钩子：按环序排空落库（DrainDevTaskRing）
 *              与通信域销毁时的消费位点清理（DiscardDevTaskRingState）。
 * Create: 2026-10-29
 */

#ifndef DPU_TASK_RING_H
#define DPU_TASK_RING_H

#include <cstdint>

#include "dpu_level1_res.h"

namespace HcclSim {
namespace DpuSim {

    /**
     * @brief 排空一个通信域转发环中的记录并按环序落库。
     * @param region 输入：目标共享内存区域（host 视角；环锚在请求半区尾部）。
     * @param markExclusive 输入：环截止水位（ticket 上界，排 [head, mark)）；
     *        缺省 UINT64_MAX = 排空全部已发布记录（空闲轮/回调后/清理路径）。
     *        执行请求回调前必须传请求头 DPU_REQ_RING_MARK_OFF 水位：只有
     *        SendRequest 之前推送的记录（< 水位）才允许落在回调直写任务之前，
     *        之后的推送由回调后的补充排空消费，落库序与设备程序序严格对齐。
     *        水位 <= 已消费位点时为空排水（兼容陈旧/初始值 0）。
     * @note 幂等可重入：消费位点按 commId 维护于进程内 map，重复调用只消费新增记录。
     *       落库写方为本进程（getpid scope，与 device 直写时 ppid scope 同一张物理表），
     *       opDetailId 原样携带 device 推送时刻的值（与 device 直写语义逐字节一致）。
     *       线程安全：执行器线程与清理线程并发排空由互斥锁串行化。
     */
    void DrainDevTaskRing(const DpuMemRegion& region, uint64_t markExclusive = UINT64_MAX);

    /**
     * @brief 丢弃指定通信域的消费位点（通信域销毁收尾）。
     * @param commId 输入：通信域 id（Communicator 表主键）。
     */
    void DiscardDevTaskRingState(uint64_t commId);

} // namespace DpuSim
} // namespace HcclSim

#endif // DPU_TASK_RING_H
