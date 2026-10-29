/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_comm_config.h"
#include <cstring> // 包含strnlen函数

namespace hccl {
constexpr uint32_t MULTIPLE = 4;             // 用于A5判断TC是否为4的倍数
constexpr uint32_t TC_MAX = 255;             // TC的最大值（不区分芯片类型）
constexpr uint32_t SL_MAX = 7u;              // sl范围的最大值，sl即serviceLevel（不区分芯片类型）
constexpr uint32_t TC_DEFAULT = 0xFFFFFFFFu; // TC的默认值（不区分芯片类型）
constexpr uint32_t SL_DEFAULT = 0xFFFFFFFFu; // SL的默认值（不区分芯片类型）
constexpr uint32_t HCCL_COMM_CONFIG_QOS_VERSION = 10U;
constexpr uint32_t HCCL_DETERMINISTIC_MAX = 2U;

static HcclResult GetHcclCommConfigVersion(const HcclCommConfig* config, uint32_t& version)
{
    CHK_PTR_NULL(config);

    CommConfigInfo info{};
    s32 sRet = memcpy_s(&info, sizeof(info), config->reserved, sizeof(info));
    CHK_PRT_RET(sRet != EOK, HCCL_ERROR("[GetHcclCommConfigVersion] memcpy_s failed, errNo[%d]", sRet), HCCL_E_MEMORY);
    version = info.version;
    return HCCL_SUCCESS;
}

static HcclResult ApplyHcclQos(const HcclCommConfig* hcclCommConfig, CommConfig& commConfig)
{
    if (hcclCommConfig == nullptr) {
        return HCCL_SUCCESS;
    }

    // hcclQos 自 CommConfig version 10 起引入；低版本按未配置处理
    uint32_t configVersion = 0U;
    CHK_RET(GetHcclCommConfigVersion(hcclCommConfig, configVersion));
    if (configVersion < HCCL_COMM_CONFIG_QOS_VERSION) {
        HCCL_INFO(
            "[ApplyHcclQos] skip hcclQos by version, configVersion[%u] < HCCL_COMM_CONFIG_QOS_VERSION[%u]",
            configVersion, HCCL_COMM_CONFIG_QOS_VERSION);
        CHK_RET(commConfig.SetConfigHcclQos(HCCL_COMM_QOS_CONFIG_NOT_SET));
        return HCCL_SUCCESS;
    }

    u32 qos = hcclCommConfig->hcclQos;
    CHK_PRT_RET(
        (qos != HCCL_COMM_QOS_CONFIG_NOT_SET) && (qos > 7u),
        HCCL_ERROR(
            "[ApplyHcclQos]errNo[0x%016llx] invalid hcclQos[%u], must be 0xFFFFFFFF or in [0,7]",
            HCCL_ERROR_CODE(HCCL_E_PARA), qos),
        HCCL_E_PARA);
    CHK_RET(commConfig.SetConfigHcclQos(qos));
    HCCL_INFO("[ApplyHcclQos] hcclQos[%u]", qos);
    return HCCL_SUCCESS;
}

static HcclResult ApplyHcclSqDepth(const HcclCommConfig* hcclCommConfig, CommConfig& commConfig)
{
    if (hcclCommConfig == nullptr) {
        return HCCL_SUCCESS;
    }

    uint32_t configVersion = 0U;
    CHK_RET(GetHcclCommConfigVersion(hcclCommConfig, configVersion));
    if (configVersion < HCCL_COMM_CONFIG_SQ_DEPTH_VERSION) {
        HCCL_INFO(
            "[ApplyHcclSqDepth] skip hcclSqDepth by version, configVersion[%u] < "
            "HCCL_COMM_CONFIG_SQ_DEPTH_VERSION[%u]",
            configVersion, HCCL_COMM_CONFIG_SQ_DEPTH_VERSION);
        return commConfig.SetConfigSqDepth(HCCL_COMM_SQ_DEPTH_CONFIG_NOT_SET);
    }

    const uint32_t sqDepth = hcclCommConfig->hcclChannelSqDepth;
    CHK_RET(commConfig.SetConfigSqDepth(sqDepth));
    HCCL_INFO("[ApplyHcclSqDepth] hcclSqDepth[%u]", sqDepth);
    return HCCL_SUCCESS;
}

static HcclResult ApplyHcclScqDepth(const HcclCommConfig* hcclCommConfig, CommConfig& commConfig)
{
    if (hcclCommConfig == nullptr) {
        return HCCL_SUCCESS;
    }

    uint32_t configVersion = 0U;
    CHK_RET(GetHcclCommConfigVersion(hcclCommConfig, configVersion));
    if (configVersion < HCCL_COMM_CONFIG_SCQ_DEPTH_VERSION) {
        HCCL_RUN_INFO(
            "[%s] skip hcclScqDepth by version, configVersion[%u] < "
            "HCCL_COMM_CONFIG_SCQ_DEPTH_VERSION[%u]",
            __func__, configVersion, HCCL_COMM_CONFIG_SCQ_DEPTH_VERSION);
        return commConfig.SetConfigScqDepth(HCCL_COMM_SCQ_DEPTH_CONFIG_NOT_SET);
    }

    const uint32_t scqDepth = hcclCommConfig->hcclChannelScqDepth;
    CHK_RET(commConfig.SetConfigScqDepth(scqDepth));
    HCCL_RUN_INFO("[%s] hcclScqDepth[%u]", __func__, scqDepth);
    return HCCL_SUCCESS;
}

static HcclResult ApplyTrafficClassAndServiceLevel(const HcclCommConfig* hcclCommConfig, CommConfig& commConfig)
{
    if (hcclCommConfig == nullptr) {
        return HCCL_SUCCESS;
    }

    u32 tc = hcclCommConfig->hcclRdmaTrafficClass;
    CHK_PRT_RET(
        (tc != TC_DEFAULT) && (tc > TC_MAX || (tc % MULTIPLE != 0)),
        HCCL_ERROR(
            "[ApplyTrafficClassAndServiceLevel]errNo[0x%016llx] invalid hcclRdmaTrafficClass[%u], "
            "must be 0xFFFFFFFF or in [0,255] and a multiple of 4",
            HCCL_ERROR_CODE(HCCL_E_PARA), tc),
        HCCL_E_PARA);
    CHK_RET(commConfig.SetConfigTrafficClass(tc));

    u32 sl = hcclCommConfig->hcclRdmaServiceLevel;
    CHK_PRT_RET(
        (sl != SL_DEFAULT) && (sl > SL_MAX),
        HCCL_ERROR(
            "[ApplyTrafficClassAndServiceLevel]errNo[0x%016llx] invalid hcclRdmaServiceLevel[%u], "
            "must be 0xFFFFFFFF or in [0,7]",
            HCCL_ERROR_CODE(HCCL_E_PARA), sl),
        HCCL_E_PARA);
    CHK_RET(commConfig.SetConfigServiceLevel(sl));
    return HCCL_SUCCESS;
}

static HcclResult
ApplyHcclDeterministic(const HcclCommConfig* hcclCommConfig, CommConfig& commConfig, int32_t processDeterministic)
{
    uint32_t deterministic = HCCL_COMM_DETERMINISTIC_CONFIG_NOT_SET;
    if (hcclCommConfig != nullptr) {
        deterministic = hcclCommConfig->hcclDeterministic;
    }

    if (deterministic == HCCL_COMM_DETERMINISTIC_CONFIG_NOT_SET) {
        CHK_PRT_RET(
            processDeterministic < 0,
            HCCL_ERROR("[ApplyHcclDeterministic] invalid process deterministic[%d].", processDeterministic),
            HCCL_E_PARA);
        deterministic = static_cast<uint32_t>(processDeterministic);
    }

    CHK_PRT_RET(
        deterministic > HCCL_DETERMINISTIC_MAX,
        HCCL_ERROR("[ApplyHcclDeterministic] invalid hcclDeterministic[%u], which should be 0, 1 or 2.", deterministic),
        HCCL_E_PARA);
    CHK_RET(commConfig.SetConfigDeterministic(static_cast<u8>(deterministic)));
    HCCL_INFO("[ApplyHcclDeterministic] hcclDeterministic[%u].", deterministic);
    return HCCL_SUCCESS;
}

HcclResult ApplyHcclCommConfig(
    const HcclCommConfig* hcclCommConfig, CommConfig& commConfig, uint32_t& opExpansionMode,
    int32_t processDeterministic)
{
    opExpansionMode = 0;
    CHK_RET(ApplyHcclDeterministic(hcclCommConfig, commConfig, processDeterministic));
    if (hcclCommConfig == nullptr) {
        return HCCL_SUCCESS;
    }

    opExpansionMode = hcclCommConfig->hcclOpExpansionMode;
    CHK_RET(ApplyTrafficClassAndServiceLevel(hcclCommConfig, commConfig));
    CHK_RET(ApplyHcclQos(hcclCommConfig, commConfig));
    CHK_RET(commConfig.SetConfigSymmetricMemoryStride(hcclCommConfig->hcclSymWinMaxMemSizePerRank));

    if (hcclCommConfig->hcclAlgo[0] != '\0') {
        size_t algoLen = strnlen(hcclCommConfig->hcclAlgo, static_cast<size_t>(HCCL_COMM_ALGO_MAX_LENGTH));
        CHK_RET(commConfig.SetConfigHcclAlgoStr(std::string(hcclCommConfig->hcclAlgo, algoLen)));
    }
    if (hcclCommConfig->hcclUdi[0] != '\0') {
        size_t udiLen = strnlen(hcclCommConfig->hcclUdi, UDI_MAX_LENGTH);
        const std::string hcclUdi(hcclCommConfig->hcclUdi, udiLen);
        CHK_RET(commConfig.SetConfigUdi(hcclUdi));
        HCCL_INFO(
            "[ApplyHcclCommConfig] set hcclUdi, commName[%s], hcclUdi[%s].", commConfig.GetConfigCommName().c_str(),
            hcclUdi.c_str());
    }
    CHK_RET(ApplyHcclSqDepth(hcclCommConfig, commConfig));
    CHK_RET(ApplyHcclScqDepth(hcclCommConfig, commConfig));
    return HCCL_SUCCESS;
}
} // namespace hccl
