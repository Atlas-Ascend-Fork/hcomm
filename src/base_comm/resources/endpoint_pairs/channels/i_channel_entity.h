/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef I_CHANNEL_ENTITY_H
#define I_CHANNEL_ENTITY_H

#include "hccl/hccl_types.h"
#include "hcomm_res_defs.h"
#include "hcomm/hcomm_res_entity_defs.h"

namespace hcomm {

class IChannelEntity {
public:
    virtual ~IChannelEntity() = default;

    virtual HcclResult PreAllocCtx(ChannelHandle& outHandle)
    {
        (void)outHandle;
        return HCCL_E_NOT_SUPPORT;
    }
    virtual void* GetCtxPtr() const { return nullptr; }
    virtual bool IsDeviceEntityReady() const { return deviceEntityReady_; }
    virtual HcclResult BuildChannelEntityToDevice(void** devChannelPtr)
    {
        (void)devChannelPtr;
        return HCCL_E_NOT_SUPPORT;
    }
    virtual HcclResult PreAllocChannelEntityToDevice(void** devChannelPtr)
    {
        (void)devChannelPtr;
        return HCCL_E_NOT_SUPPORT;
    }
    virtual HcclResult FillChannelEntityToDevice() { return HCCL_E_NOT_SUPPORT; }

    virtual void SetDeviceEntityReady() { deviceEntityReady_ = true; }

protected:
    bool deviceEntityReady_{false}; // 设备实体就绪状态
};

} // namespace hcomm

#endif // I_CHANNEL_ENTITY_H
