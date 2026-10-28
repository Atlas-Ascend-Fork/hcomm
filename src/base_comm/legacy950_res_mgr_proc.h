/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_LEGACY950_RES_MGR_PROC_H
#define HCOMM_LEGACY950_RES_MGR_PROC_H

#include "hccl/hccl_types.h"
#include "hccl_common.h"

namespace hcomm {

/**
 * @brief legacy950（Hccl::，libhccl_v2.so）单例锚定节点。
 * 单例跨二进制搬不动，只锚定不入树（README R8）；本节点声明序首位 = 全树最后死。
 * 并以备份槽调用 AnchorDeviceSingletons 预热设备级 legacy 数组（锁死晚于全树死亡）。
 */
class Legacy950ResMgrProc {
public:
    void DeInit() {}

public:
    // 构造即完成单例锚定
    Legacy950ResMgrProc();
    ~Legacy950ResMgrProc() = default;

private:
    Legacy950ResMgrProc(const Legacy950ResMgrProc&) = delete;
    Legacy950ResMgrProc& operator=(const Legacy950ResMgrProc&) = delete;
};

} // namespace hcomm

#endif // HCOMM_LEGACY950_RES_MGR_PROC_H
