/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CCU_DEVICE_PUB_H
#define HCOMM_CCU_DEVICE_PUB_H

#include <memory>
#include <vector>
#include <cstdint>

#include "comm_engine/ccu/ccu_common_pub.h"
#include "comm_engine/ccu/ccu_device_res_pub.h"
#include "comm_engine/ccu/ccu_drv_handle_pub.h"

#include "hccl_types.h"
#include "hccl_rank_graph.h"

// 支持新老通信域混跑，引入legacy数据结构
#include "unified_platform/pub_inc/ccu/ccu_dev_mgr.h"
#include "dev_type.h"

namespace hcomm {

using CcuResHandle = void*;

// 不复用legacy数据结构，对上层支持CommAddr，不使用Hccl::IpAddress
struct CcuChannelPara {
    CommAddr commAddr{};
    uint32_t channelNum{0};
    uint32_t jettyNum{0};
    uint32_t sqSize{0};

    CcuChannelPara() = default;
    CcuChannelPara(const CommAddr& address, const uint32_t channelNum, const uint32_t jettyNum, const uint32_t sqSize)
        : commAddr(address),
          channelNum(channelNum),
          jettyNum(jettyNum),
          sqSize(sqSize)
    {}
};

using CcuJettyType = Hccl::CcuJettyType;
/* 开源自定义算子CCU设备管理实现，当前支持新老通信域混跑，
 * 暂时改用legacy数据结构，避免反向依赖
 * MAKE_ENUM(CcuJettyType, CCUM_CACHED_JETTY, TA_CACHED_JETTY, INVALID_JETTY);
 */

using CcuJettyInfo = Hccl::CcuJettyInfo;
/* 开源自定义算子CCU设备管理实现，当前支持新老通信域混跑，
 * 暂时改用legacy数据结构，避免反向依赖
 * struct CcuJettyInfo {
 *     CcuJettyType jettyType{CcuJettyType::INVALID_JETTY};
 *     uint16_t jettyCtxId{0};
 *     uint16_t taJettyId{0};

 *     uint32_t sqDepth{0};
 *     uint32_t wqeBBStartId{0};

 *     uint64_t sqBufVa{0};
 *     uint32_t sqBufSize{0};
 * };
 */

using CcuChannelInfo = Hccl::CcuChannelInfo;
/* 开源自定义算子CCU设备管理实现，当前支持新老通信域混跑，
 * 暂时改用legacy数据结构，避免反向依赖
 * struct CcuChannelInfo {
 *     uint32_t channelId{0};
 *     uint8_t dieId{0};
 *     std::vector<CcuJettyInfo> jettyInfos;
 * };
 */

/**
 * @brief 启用CCU特性，初始化CCU平台层
 *
 * @param userDevId 用户设备ID
 * @param ccuDrvHandle CCU驱动句柄
 * @return CcuResult 返回CcuResult类型的结果
 * @note 失败时返回 CCU_E_* 错误码，非 CCU_SUCCESS 即失败
 */
CcuResult CcuInitFeature(const int32_t userDevId, std::shared_ptr<CcuDrvHandle>& ccuDrvHandle);

/**
 * @brief 关闭CCU特性，解初始化CCU平台层
 *
 * @param userDevId 用户设备ID
 * @return CcuResult 返回CcuResult类型的结果
 * @note 失败时返回 CCU_E_* 错误码，非 CCU_SUCCESS 即失败
 */
CcuResult CcuDeinitFeature(const int32_t userDevId);

/**
 * @brief 获取指定device的主板类型
 *
 * @param userDevId 用户设备ID
 * @param hcclMainboardId 出参，返回该device的主板类型
 * @return HcclResult 返回HcclResult类型的结果
 */
HcclResult CcuGetMainboardType(uint32_t userDevId, Hccl::HcclMainboardId& hcclMainboardId);

/**
 * @brief 查询CCU设备是否已完成初始化
 *
 * @param userDevId 用户设备ID
 * @return bool true表示已初始化，false表示未初始化或入参非法
 * @note 无
 */
bool CcuIsInited(const int32_t userDevId);

/**
 * @brief 触发CCU Task Kill
 *
 * @param userDevId 用户设备ID
 * @return HcclResult 返回HcclResult类型的结果
 * @note 该接口会处理全部die，未启用die将跳过
 */
HcclResult CcuSetTaskKill(const int32_t userDevId);

/**
 * @brief 配置CCU Task Kill完成状态
 *
 * @param userDevId 用户设备ID
 * @return HcclResult 返回HcclResult类型的结果
 * @note 该接口会处理全部die，未启用die将跳过
 */
HcclResult CcuSetTaskKillDone(const int32_t userDevId);

/**
 * @brief 清理指定ioDie CCU的全部CKE资源，重置为0
 *
 * @param userDevId 用户设备ID
 * @return HcclResult 返回HcclResult类型的结果
 * @note 未启用die无需清理将视为成功
 */
HcclResult CcuCleanDieCkes(const int32_t userDevId, const uint8_t dieId);
} // namespace hcomm
#endif // HCOMM_CCU_DEVICE_PUB_H
