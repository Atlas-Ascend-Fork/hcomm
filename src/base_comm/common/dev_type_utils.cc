/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/dev_type_utils_pub.h"

#include <string>

#include "log.h"
#include "protocol_utils.h"

HcclResult CheckDevTypeSupport(
    const std::initializer_list<DevType>& supportedTypes, CommProtocol protocol, EndpointLocType locType)
{
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    std::string supportedTypesStr;
    for (const DevType& supportedType : supportedTypes) {
        if (!supportedTypesStr.empty()) {
            supportedTypesStr += ",";
        }
        supportedTypesStr += GetDevTypeStr(supportedType);
        if (deviceType == supportedType) {
            return HCCL_SUCCESS;
        }
    }
    HCCL_ERROR(
        "[CheckDevTypeSupport] protocol[%s] locType[%s] deviceType[%s] not in supportedTypes[%s]",
        GetEnumToString(GetCommProtocolStrMap(), protocol).c_str(), GetEndpointLocTypeStr(locType),
        GetDevTypeStr(deviceType), supportedTypesStr.c_str());
    return HCCL_E_NOT_SUPPORT;
}
