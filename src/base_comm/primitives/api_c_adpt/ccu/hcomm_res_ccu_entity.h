/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_RES_CCU_ENTITY_H
#define HCOMM_RES_CCU_ENTITY_H

#include <stdint.h>

#include "hcomm_res_defs.h"
#include "hcomm/ccu/hcomm_res_ccu_entity_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 查询 CCU Channel 实体快照。
 *
 * 查询指定 CCU Channel 的资源视图，包括设备、协议、CCU 版本、本端/远端
 * Variable 与 Event 资源 ID、对端 CCU 资源空间及远端内存 buffer 信息。
 *
 * @param[in] channel CCU Channel 句柄，须来自 HcclChannelAcquire（engine 须为 COMM_ENGINE_CCU）。
 * @param[out] channelEntity 调用方申请并传入的实体快照；接口成功时填充。
 *
 * @return HcommResult 成功返回 0，其他失败。
 * @retval HCCL_E_PTR(2) channelEntity 为空。
 * @retval HCCL_E_NOT_SUPPORT(5) channel 不是 CCU Channel。
 * @retval HCCL_E_NOT_FOUND(6) channel 不存在。
 * @retval 其他 Channel 尚未 READY、底层资源信息不满足契约或内部异常。
 *
 * @note 接口在 Channel 状态锁内读取快照，可与 Channel Clean/UpdateMemInfo 并发调用；
 *       返回成功只代表调用瞬间快照构建成功，Channel 后续状态仍可能变化。
 * @note channelEntity.remoteBuffers 指向 hcomm 线程本地数组，其借用期到当前线程下一次调用
 *       本接口为止；调用方如需保留数据，须在借用期内完成深拷贝。
 * @note 失败时 channelEntity 会被清零，调用方不得消费其中内容。
 *
 * @code {.c}
 * HcommCcuChannelEntity entity{};
 * HcommResult ret = HcommCcuChannelGetEntity(channel, &entity);
 * if (ret == 0) {
 *     // 消费 entity；如需保留 remoteBuffers，须先完成深拷贝。
 * }
 * @endcode
 */
HcommResult HcommCcuChannelGetEntity(ChannelHandle channel, HcommCcuChannelEntity* channelEntity);

/**
 * @brief 下发 CCU 微码指令。
 *
 * 将 host 侧连续微码指令数组拷贝到设备暂存内存，并经 CCU 驱动下发到指定
 * 设备与 IO Die 的 CCU。指令宽度为 32 字节，接口按 instNum * 32 字节拷贝数据。
 *
 * @param[in] deviceLogicId 设备逻辑 ID。
 * @param[in] dieId CCU IO Die ID，当前取值范围 [0, 2)。
 * @param[in] startInstructionId 起始硬件指令槽位，须来自成功申请的连续指令空间。
 * @param[in] data host 侧指令数组首地址，至少包含 instNum 条 32 字节指令。
 * @param[in] instNum 下发指令条数，须大于 0。
 *
 * @return HcommResult 成功返回 0，其他失败。
 * @retval HCCL_E_PARA(1) data 为空、instNum 为 0、dieId 越界或总字节数超过 32 位上限。
 * @retval 其他 暂存内存申请、H2D 拷贝或 CCU 驱动调用失败、内部异常或底层驱动错误。
 *
 * @note data 仅在本接口执行期间借用，接口返回后可立即复用或释放；接口内部不保留该地址。
 * @note 本接口会临时申请设备暂存内存，H2D 拷贝和驱动下发完成后立即释放；调用方无需管理该内存。
 * @note startInstructionId 与下发的指令槽位须与 HcommCcuAllocInstSpace 申请的空间匹配，
 *       空间不足或重复下发时行为由底层驱动和硬件定义。
 * @note 本接口不保证异步执行的 CCU 指令已完成；如需确认执行结果，应通过任务同步或事件机制等待。
 *
 * @code {.c}
 * uint32_t startInstId = 0;
 * if (HcommCcuAllocInstSpace(deviceLogicId, dieId, instNum, &startInstId) == 0) {
 *     (void)HcommCcuSubmitInsts(deviceLogicId, dieId, startInstId, instructions, instNum);
 * }
 * @endcode
 */
HcommResult HcommCcuSubmitInsts(
    int32_t deviceLogicId, uint32_t dieId, uint32_t startInstructionId, const void* data, uint32_t instNum);

/**
 * @brief 按设备和 die 申请连续 CCU 指令空间。
 *
 * @param[in] deviceLogicId 设备逻辑 ID。
 * @param[in] dieId CCU IO Die ID，当前取值范围 [0, 2)，上限即设备 CCU IO Die 数。
 * @param[in] instNum 申请的连续指令条数，须大于 0。
 * @param[out] startInstId 输出申请区间的起始指令槽位。
 *
 * @return HcommResult 成功返回 0，其他失败。
 * @retval HCCL_E_PTR(2) startInstId 为空。
 * @retval HCCL_E_PARA(1) instNum 为 0 或 dieId 越界。
 * @retval 其他 指令资源不足、设备资源不可用、设备未初始化、底层分配失败或内部异常。
 *
 * @note 申请到的空间为 [startInstId, startInstId + instNum)，调用方须保存
 *       startInstId 与 instNum。当前不提供按区间单独归还的入口；指令槽位由 hcomm 的
 *       per-device CCU 分配器管理，最后一个 CCU feature 引用释放并触发 CcuDeinitFeature 后，
 *       随 CCU 组件与驱动整体反初始化回收；单个 CCU Instance 销毁不保证立即释放该设备级指令空间。
 * @note 同一设备同一 Die 的指令资源分配使用底层分配器互斥锁；并发调用不会破坏分配器内部状态。
 *       同一指令空间的使用顺序由调用方保证。
 * @note 失败时 startInstId 会被置 0；只有返回成功时才可消费该值。
 *
 * @code {.c}
 * uint32_t startInstId = 0;
 * HcommResult ret = HcommCcuAllocInstSpace(deviceLogicId, dieId, instNum, &startInstId);
 * if (ret != 0) {
 *     return ret;
 * }
 * // 消费 [startInstId, startInstId + instNum) 指令空间。
 * @endcode
 */
HcommResult HcommCcuAllocInstSpace(int32_t deviceLogicId, uint32_t dieId, uint32_t instNum, uint32_t* startInstId);

#ifdef __cplusplus
}
#endif

#endif // HCOMM_RES_CCU_ENTITY_H
