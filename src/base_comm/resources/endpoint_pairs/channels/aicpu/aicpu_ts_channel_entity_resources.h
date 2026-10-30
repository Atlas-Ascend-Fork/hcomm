/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_TS_CHANNEL_ENTITY_RESOURCES_H
#define AICPU_TS_CHANNEL_ENTITY_RESOURCES_H

#include <memory>
#include <string>
#include <vector>

#include "common/cast_utils_pub.h"
#include "mem_device_pub.h"
#include "hccl/hccl_types.h"
#include "hcomm_res_entity_defs.h"

namespace hcomm {

using DeviceMem = hccl::DeviceMem;

/**
 * @brief AICPU URMA Channel 的设备侧实体资源。
 */
class AicpuChannelEntityResources {
public:
    HcclResult Build(
        const std::vector<char>& packedChannelData, CommEngine engine, CommProtocol protocol,
        ChannelHandle originHandle, ChannelHandle& entityHandle);

private:
    ChannelEntity entity_{};
    DeviceMem localNotifies_;
    DeviceMem remoteNotifies_;
    DeviceMem localBuffers_;
    DeviceMem remoteBuffers_;
    DeviceMem sqContexts_;
    DeviceMem entityMem_;
};

} // namespace hcomm

#endif // AICPU_TS_CHANNEL_ENTITY_RESOURCES_H
