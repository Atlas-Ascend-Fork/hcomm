/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCP_TLV_HDC_MGR_H
#define HCCP_TLV_HDC_MGR_H

#include <mutex>

#include "hccl_types.h"

namespace hcomm {

using TlvHandle = void*;

class HccpTlvHdcMgr {
public:
    HccpTlvHdcMgr() = default;
    ~HccpTlvHdcMgr();
    // 访问器回写钩子：本类唯一外部写入口
    void SetDevPhyId(uint32_t devPhyId) { devPhyId_ = devPhyId; }
    // 资源树 HdcProc 值成员（每卡一份）；调用点走窄访问器 hcomm::GetHccpTlvHdcMgr（base_comm_node_accessors.h）
    TlvHandle GetHandle();
    HcclResult Init();
    // 幂等（initFlag_ 守护）；树节点 DeInit 链与析构共用
    HcclResult Deinit();
    bool IsInited() const { return initFlag_; }

private:
    HccpTlvHdcMgr(const HccpTlvHdcMgr& hccpTlvHdcMgr) = delete;
    HccpTlvHdcMgr& operator=(const HccpTlvHdcMgr& hccpTlvHdcMgr) = delete;

private:
    uint32_t devPhyId_{0};
    std::mutex innerMutex_{};
    bool initFlag_{false};
    TlvHandle tlvHandle_{nullptr};
};

} // namespace hcomm

#endif // HCCP_TLV_HDC_MGR_H
