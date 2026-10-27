/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// base_comm 内部使用的 CCU 设备接口（coll_comm 不消费）；公共类型（CcuChannelPara/CcuChannelInfo/CcuResHandle 等）
// 仍从边界头获取。
#ifndef HCOMM_CCU_DEVICE_H
#define HCOMM_CCU_DEVICE_H

#include <cstdint>
#include <vector>

#include "comm_engine/ccu/ccu_device_pub.h"

namespace hcomm {

// CcuAllocResHandleByResDescs 仅以 const CcuResDesc* 形式使用，前向声明即可，
// 避免此处包含 ccu_res_desc.h（其会引入 ccu_dev_mgr_imp.h，造成循环包含）。
class CcuResDesc;

struct CcuResRepository;
struct tagCntXnBlock;
using CntXnBlock = tagCntXnBlock;

/**
 * @brief 获取指定die是否启用
 *
 * @param userDevId 用户设备ID
 * @param dieId ccu channel 所属的 IO Die 编号
 * @param enableFlag 出参，表示该die是否启用
 * @return HcclResult 返回HcclResult类型的结果
 * @note dieId越界时返回HCCL_E_PARA
 */
CcuResult CcuGetDieEnableInfo(int32_t userDevId, uint8_t dieId, bool& enableFlag);

/**
 * @brief 查询指定 ioDie
 * 上各类资源总量，按block分的资源类型（LoopEngine、Ms、Cke）查询的是可以分配的总量（块大小*块总数）
 *
 * @param userDevId 用户设备ID
 * @param dieId ioDie ID
 * @param num 出参，返回该 die 上对应资源类型的总量
 * @return CcuResult
 * @note dieId 越界或 die 未启用时返回错误
 */
CcuResult CcuGetLoopEngineNum(int32_t userDevId, uint8_t dieId, uint32_t& num);
CcuResult CcuGetMsNum(int32_t userDevId, uint8_t dieId, uint32_t& num);
CcuResult CcuGetCkeNum(int32_t userDevId, uint8_t dieId, uint32_t& num);
CcuResult CcuGetXnNum(int32_t userDevId, uint8_t dieId, uint32_t& num);
CcuResult CcuGetGsaNum(int32_t userDevId, uint8_t dieId, uint32_t& num);
CcuResult CcuGetInstructionNum(int32_t userDevId, uint8_t dieId, uint32_t& num);
CcuResult CcuGetMissionNum(int32_t userDevId, uint8_t dieId, uint32_t& num);
CcuResult CcuGetCascCntNum(int32_t userDevId, uint8_t dieId, uint32_t& num);

/**
 * @brief 基于资源描述符数组申请批量资源
 *
 * @param userDevId 用户设备ID
 * @param descs 资源描述符指针数组（每个描述符携带 dieId 及各资源数量）
 * @param descNum 资源描述符数量
 * @param resHandle 返回的CCU批量资源句柄
 * @return HcclResult 返回HcclResult类型的结果
 * @note 资源不足时返回HCCL_E_UNAVIL，其余非HCCL_SUCCESS结果属于错误
 */
CcuResult
CcuAllocResHandleByResDescs(int32_t userDevId, const CcuResDesc* descs[], uint32_t descNum, CcuResHandle& resHandle);

/**
 * @brief 按加速引擎模式申请批量资源
 *
 * @param userDevId 用户设备ID
 * @param ccuEngine CCU通信引擎类型
 * @param resHandle 返回的CCU批量资源句柄
 * @return HcclResult 返回HcclResult类型的结果
 * @note 资源不足时返回HCCL_E_UNAVIL，其余非HCCL_SUCCESS结果属于错误
 */
CcuResult CcuAllocResHandleByInsType(int32_t userDevId, CcuInstanceType ccuInsType, CcuResHandle& resHandle);

/**
 * @brief 根据资源句柄查看对应资源信息
 *
 * @param userDevId 用户设备ID
 * @param resHandle 查询的CCU批量资源句柄
 * @param resRepo 返回的CCU批量资源信息
 * @return HcclResult 返回HcclResult类型的结果
 * @note 资源句柄无法查找到时返回HCCL_E_NOT_FOUND，其余非HCCL_SUCCESS结果属于错误
 */
CcuResult CcuCheckResource(const int32_t userDevId, const CcuResHandle resHandle, CcuResRepository& resRepo);

/**
 * @brief 根据资源句柄释放对应资源信息
 *
 * @param userDevId 用户设备ID
 * @param resHandle 查询的CCU批量资源句柄
 * @note 资源句柄无法查找到时返回HCCL_E_NOT_FOUND，其余非HCCL_SUCCESS结果属于错误
 * @note 返回批量的channel资源总数可能超过申请数量，jettyNum为0时由平台层决定分配数量
 */
HcclResult CcuReleaseResHandle(const int32_t userDevId, const CcuResHandle resHandle);

/**
 * @brief 申请批量ccu channel资源
 *
 * @param userDevId 用户设备ID
 * @param ccuChannelPara ccu channel 申请参数
 * @param ccuChannelInfos 返回的channel资源信息
 * @return HcclResult 返回HcclResult类型的结果
 * @note 返回批量的channel资源总数可能超过申请数量，jettyNum为0时由平台层决定分配数量
 */
HcclResult CcuAllocChannels(
    const int32_t userDevId, const CcuChannelPara& ccuChannelPara, std::vector<CcuChannelInfo>& ccuChannelInfos);

/**
 * @brief 释放ccu channel资源
 *
 * @param userDevId 用户设备ID
 * @param dieId ccu channel 所属的 IO Die 编号
 * @param ccuChannelId ccu channel 编号
 * @return HcclResult 返回HcclResult类型的结果
 * @note 无
 */
HcclResult CcuReleaseChannel(const int32_t userDevId, const uint8_t dieId, const uint32_t ccuChannelId);

/**
 * @brief 清空CCU Task Kill状态
 *
 * @param userDevId 用户设备ID
 * @return HcclResult 返回HcclResult类型的结果
 * @note 该接口会处理全部die，未启用die将跳过
 */
HcclResult CcuCleanTaskKillState(const int32_t userDevId);

/**
 * @brief 申请批量cnt xn资源
 *
 * @param userDevId 用户设备ID
 * @param dieId ioDie ID
 * @param cntXnBlock count xn block结构体
 * @return CcuResult
 * @note 返回申请的count xn block
 */
CcuResult CcuAllocCntXnBlock(const int32_t userDevId, const uint8_t dieId, CntXnBlock& cntXnBlock);

/**
 * @brief 释放指定count xn资源
 *
 * @param userDevId 用户设备ID
 * @param dieId ioDie ID
 * @param cntXnBlock count xn block结构体
 * @return CcuResult
 * @note 根据cntXnBlock释放对应的count xn资源
 */
CcuResult CcuReleaseCntXnBlock(const int32_t userDevId, const uint8_t dieId, const CntXnBlock& cntXnBlock);

/**
 * @brief 查询指定device指定地址的资源token信息
 *
 * @param userDevId 用户设备ID
 * @param srcVa 资源地址
 * @param size 资源size
 * @param tokenId token ID
 * @param tokenValue token value
 * @return CcuResult
 * @note 查询资源token信息，返回token信息
 */
CcuResult
CcuQueryTokenInfo(const int32_t userDevId, uint64_t srcVa, uint64_t size, uint64_t& tokenId, uint64_t& tokenValue);

}; // namespace hcomm
#endif // HCOMM_CCU_DEVICE_H
