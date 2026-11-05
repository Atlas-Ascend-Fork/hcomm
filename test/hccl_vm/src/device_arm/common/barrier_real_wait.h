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
 * @brief 阻塞等待本 rank（ppid 作用域表）未完成任务清零（runner 回放完成、数据落地）。
 *
 * 自动门控（无环境变量/开关文件，按算子属性判定）：取"本设备（GetCurDeviceKey）最新
 * 落库算子"的 opDetail（opType/dataType/reduceType），命中 reduce 类算子 + 64bit 数据
 * 类型（INT64/UINT64/FP64）或 PROD 归约——即 hccl LocalReduce→AicpuReduce 内联 CPU
 * 归约路径（无 REDUCE 任务制品、屏障假完成 ⇒ 时序倒置，方案二适用域）——才执行真等；
 * 其余算子/判定失败/尚无算子落库一律回退基线时序（fail-open）。判定按算子行 id 缓存，
 * 算子切换时重判并打 [BarrierRealWait] gate decision 日志。
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
