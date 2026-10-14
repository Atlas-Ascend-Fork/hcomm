/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DEV_TYPE_UTILS_H
#define DEV_TYPE_UTILS_H

#include <initializer_list>
#include <optional>

#include "dtype_common.h"
#include "hcomm_res_defs.h"
#include "log.h"

#pragma GCC visibility push(hidden)
namespace hcomm {

/* 公共取数：唯一真值源，获取失败返回 std::nullopt；aicpu（HCCD）场景须在 hrtSetlocalDeviceType 之后调用 */
namespace devtype_detail {
    inline std::optional<DevType> FetchDevType(const char* caller)
    {
        DevType devType = DevType::DEV_TYPE_COUNT;
        HcclResult ret = hrtGetDeviceType(devType);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[%s] hrtGetDeviceType failed, ret[%d].", caller, static_cast<int>(ret));
            return std::nullopt;
        }
        return devType;
    }
} // namespace devtype_detail

/* 五谓词族：[[nodiscard]] 强制处理失败态（nullopt）；NOSOC/MC62 等合法不归类返回有值 false；新增代际在各谓词 switch
 * 追加 case */

/* 仅 950 生效（如 CCU V1） */
[[nodiscard]] inline std::optional<bool> EqualTo950()
{
    auto devType = devtype_detail::FetchDevType(__func__);
    if (!devType.has_value()) {
        return std::nullopt;
    }
    return *devType == DevType::DEV_TYPE_950;
}

/* 仅 960 生效（如 CCU V2） */
[[nodiscard]] inline std::optional<bool> EqualTo960()
{
    auto devType = devtype_detail::FetchDevType(__func__);
    if (!devType.has_value()) {
        return std::nullopt;
    }
    return *devType == DevType::DEV_TYPE_960;
}

/* 950 及后续代际均生效 */
[[nodiscard]] inline std::optional<bool> EqualToOrGreaterThan950()
{
    auto devType = devtype_detail::FetchDevType(__func__);
    if (!devType.has_value()) {
        return std::nullopt;
    }
    switch (*devType) {
        case DevType::DEV_TYPE_950:
        case DevType::DEV_TYPE_960:
            return true;
        /* 后续 950 系芯片在此追加 case */
        default: /* NOSOC / MC62 / COUNT / 910 系：合法但不归类 */
            return false;
    }
}

/* 960 及后续代际均生效（当前与 EqualTo960 等价） */
[[nodiscard]] inline std::optional<bool> EqualToOrGreaterThan960()
{
    auto devType = devtype_detail::FetchDevType(__func__);
    if (!devType.has_value()) {
        return std::nullopt;
    }
    switch (*devType) {
        case DevType::DEV_TYPE_960:
            return true;
        /* 后续 960 系芯片在此追加 case */
        default: /* NOSOC / MC62 / COUNT / 910 系 / 950：合法但不归类 */
            return false;
    }
}

/* 950 之前生效（隔离 A2/A3 流程）；NOSOC / MC62 为有值的 false */
[[nodiscard]] inline std::optional<bool> LessThan950()
{
    auto devType = devtype_detail::FetchDevType(__func__);
    if (!devType.has_value()) {
        return std::nullopt;
    }
    switch (*devType) {
        case DevType::DEV_TYPE_910:
        case DevType::DEV_TYPE_310P3:
        case DevType::DEV_TYPE_910B:
        case DevType::DEV_TYPE_310P1:
        case DevType::DEV_TYPE_910_93:
            return true;
        default: /* NOSOC / MC62 / COUNT / 950 / 960：合法但不归类 */
            return false;
    }
}

} // namespace hcomm
#pragma GCC visibility pop

inline const char* GetDevTypeStr(DevType devType)
{
    switch (devType) {
        case DevType::DEV_TYPE_910:
            return "DEV_TYPE_910";
        case DevType::DEV_TYPE_310P3:
            return "DEV_TYPE_310P3";
        case DevType::DEV_TYPE_910B:
            return "DEV_TYPE_910B";
        case DevType::DEV_TYPE_310P1:
            return "DEV_TYPE_310P1";
        case DevType::DEV_TYPE_910_93:
            return "DEV_TYPE_910_93";
        case DevType::DEV_TYPE_NOSOC:
            return "DEV_TYPE_NOSOC";
        case DevType::DEV_TYPE_950:
            return "DEV_TYPE_950";
        case DevType::DEV_TYPE_MC62:
            return "DEV_TYPE_MC62";
        case DevType::DEV_TYPE_960:
            return "DEV_TYPE_960";
        case DevType::DEV_TYPE_COUNT:
            return "DEV_TYPE_COUNT";
        default:
            return "UNKNOWN";
    }
}

inline const char* GetEndpointLocTypeStr(EndpointLocType locType)
{
    switch (locType) {
        case ENDPOINT_LOC_TYPE_RESERVED:
            return "RESERVED";
        case ENDPOINT_LOC_TYPE_DEVICE:
            return "DEVICE";
        case ENDPOINT_LOC_TYPE_HOST:
            return "HOST";
        default:
            return "UNKNOWN";
    }
}

// 仅校验当前芯片是否在 supportedTypes 中；protocol / locType 只用于失败日志，不参与判断。
HcclResult CheckDevTypeSupport(
    const std::initializer_list<DevType>& supportedTypes, CommProtocol protocol, EndpointLocType locType);

#endif /* DEV_TYPE_UTILS_H */
