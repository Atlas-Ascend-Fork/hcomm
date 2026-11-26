/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UB_DEVICE_ENDPOINT_BASE_H
#define UB_DEVICE_ENDPOINT_BASE_H

#include <cstdint>
#include "endpoint.h"

namespace hcomm {
/**
 * @note 职责：UB 三类 device endpoint（Urma/Uboe/UbRtp）的公共中间基类。
 */
class UbDeviceEndpointBase : public Endpoint {
public:
    explicit UbDeviceEndpointBase(const EndpointDesc& endpointDesc) : Endpoint(endpointDesc) {}
    ~UbDeviceEndpointBase() override = default;

    HcclResult InitDevEpMemTable() override;
    HcclResult ReleaseDevEpMemTable() override;
    uint64_t GetDevEpMemTablePtr() const override;
    HcclResult OnLocalMemRegistered(void* memHandle) override;
    HcclResult OnLocalMemUnregistered(uint64_t addr, uint64_t size) override;

private:
    uint64_t devEpMemTablePtr_{0};
};
} // namespace hcomm

#endif // UB_DEVICE_ENDPOINT_BASE_H
