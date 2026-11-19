/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BARRIER_REAL_WAIT_H
#define BARRIER_REAL_WAIT_H

#include <cstdint>

namespace sim {

/**
 * @brief 方案二"屏障真等"共享逻辑（level1 join 桩 / level2 hal SQ_HEAD 桩共用）。
 *
 * 编入 device 可执行文件（--export-dynamic），两个代理库以未定义符号引用、加载期解析
 * （与 InsertTaskToCollectionDev 同模式）——保证同进程内门控缓存/节流状态只有一份。
 * 详见 docs/int64_prod_reduce_aicpu_hijack_design.md §9.2（2a′ 与 level1 形态）。
 */

/**
 * @brief 设置/清除屏障门控的 kernel 上下文（HandlePipeCmdExecKernel 专用）。
 *
 * kernel 执行窗口入口写入 host 显式传入、经 QueryOpDetailIdentity 校验过的
 * opDetailId；kernel 返回/普通 kernel（id=0）时清零。SQ_HEAD/join 的门控判定按
 * 当前上下文每窗口做一次（按确切 id 读属性），窗口内重复调用复用判定、零 SQL；
 * 线程安全（原子 + 释放/获取序）：写=pipe handler 主线程，读=kernel 线程与
 * aicpu_daemon_service 后台线程。编入 device 可执行（同进程唯一实例，见上）。
 */
void SetBarrierGateContext(uint32_t opDetailId);

/**
 * @brief 阻塞等待本 rank（ppid 作用域表）未完成任务清零（runner 回放完成、数据落地）。
 *
 * 自动门控（无环境变量/开关文件，按算子属性判定）：按当前 kernel 上下文的
 * opDetail（opType/dataType/reduceType），命中 reduce 类算子 + 64bit 数据
 * 类型（INT64/UINT64/FP64）或 PROD 归约——即 hccl LocalReduce→AicpuReduce 内联 CPU
 * 归约路径（无 REDUCE 任务制品、屏障假完成 ⇒ 时序倒置，方案二适用域）——才执行真等；
 * 其余算子/普通 kernel（无算子上下文）/判定失败一律回退基线时序（fail-open）。
 * 判定按 kernel 上下文 id 缓存，上下文切换时重判并打 [BarrierRealWait] gate
 * decision 日志。
 *
 * @param streamId   wholeTable=false 时按该流过滤（level2 per-SQ 语义）；wholeTable=true 时忽略。
 * @param wholeTable true=全表等待（level1 join 形态，与已验证初版语义一致）；false=按流等待（level2 hal 形态）。
 * @param callerTag  日志标注。
 * @return true=自动门控命中且执行了真等路径（含各类 fail-open 放行）；false=门控未命中（基线，未等待）。
 *
 * 内建防挂死闸门（全部 fail-open：直接返回、调用方按基线语义放行，绝不无限挂起）：
 *   ① runner 未安装（Plugin 表无 tag='runner' 行，判据同 host aclrt_stream_stub）→ 不等待；
 *   ② 等待上限 300s；③ 零进度停滞 60s（runner 已装但未消费，如崩溃/残存旧 runner）。
 */
bool BarrierRealWait(uint64_t streamId, bool wholeTable, const char* callerTag);

} // namespace sim

#endif // BARRIER_REAL_WAIT_H
