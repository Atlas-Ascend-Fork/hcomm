/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_RES_CCU_H
#define HCOMM_RES_CCU_H

#include <stdint.h>

#include "hcomm_res_defs.h"
#include "hcomm/ccu/hcomm_res_ccu_entity_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 查询 CCU Channel 实体快照。
 *
 * 持锁校验通道状态后，按当前资源信息填充调用方传入的
 * HcommCcuChannelEntity。remoteBuffers 借用 hcomm 线程本地数组，
 * 借用期到本线程下一次调用本接口为止，调用方须在借用期内完成深拷贝。
 *
 * @param channel        输入：通道句柄。
 * @param channelEntity 输出：CCU Channel 实体快照。
 *
 * @return 成功返回 HCCL_SUCCESS，其他失败。
 * @retval HCCL_SUCCESS       查询成功
 * @retval HCCL_E_PTR         参数为空
 * @retval HCCL_E_NOT_SUPPORT 通道不是 CCU 通道
 * @retval HCCL_E_UNAVAIL     通道未就绪
 */
HcommResult HcommCcuGetChannelEntity(ChannelHandle channel, HcommCcuChannelEntity* channelEntity);

/**
 * @brief 下发 CCU 微码指令。
 *
 * 将 host 侧指令数组暂存到设备内存，经驱动下发到目标 CCU。
 *
 * @param deviceLogicId      输入：设备逻辑 ID。
 * @param dieId              输入：die ID。
 * @param startInstructionId 输入：起始硬件指令槽位。
 * @param data               输入：host 侧指令数组，仅本次调用内借用。
 * @param instNum            输入：指令条数。
 *
 * @return 成功返回 HCCL_SUCCESS，其他失败。
 * @retval HCCL_SUCCESS 下发成功
 * @retval HCCL_E_PTR  data 为空
 * @retval HCCL_E_PARA dieId 或指令数量非法
 */
HcommResult HcommCcuSubmitInsts(
    int32_t deviceLogicId, uint32_t dieId, uint32_t startInstructionId, const void* data, uint32_t instNum);

/**
 * @brief 按设备和 die 申请连续 CCU 指令空间。
 *
 * @param deviceLogicId 输入：设备逻辑 ID。
 * @param dieId         输入：die ID。
 * @param instNum       输入：申请的指令条数。
 * @param startInstId  输出：申请区间的起始指令槽位。
 *
 * @return 成功返回 HCCL_SUCCESS，其他失败。
 * @retval HCCL_SUCCESS   申请成功
 * @retval HCCL_E_PTR     startInstId 为空
 * @retval HCCL_E_PARA    dieId 或指令数量非法
 * @retval HCCL_E_UNAVAIL 指令资源不足
 */
HcommResult HcommCcuAllocInstSpace(int32_t deviceLogicId, uint32_t dieId, uint32_t instNum, uint32_t* startInstId);

/**
 * @brief 归还 CCU 微码指令空间。
 *
 * @param deviceLogicId 输入：设备逻辑 ID。
 * @param dieId         输入：die ID。
 * @param instNum       输入：归还的指令条数。
 * @param startInstId   输入：归还区间的起始指令槽位。
 *
 * @return 成功返回 HCCL_SUCCESS，其他失败。
 * @retval HCCL_SUCCESS 归还成功
 * @retval HCCL_E_PARA  dieId 或指令数量非法
 */
HcommResult HcommCcuReleaseInstSpace(int32_t deviceLogicId, uint32_t dieId, uint32_t instNum, uint32_t startInstId);

#ifdef __cplusplus
}
#endif

#endif // HCOMM_RES_CCU_H
