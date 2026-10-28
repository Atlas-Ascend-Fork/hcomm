/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CONFIG_MGR_CONFIG_MGR_PROC_H
#define HCOMM_CONFIG_MGR_CONFIG_MGR_PROC_H

#include "hccl/hccl_types.h"
#include "hccl_common.h"
#include "config_mgr/config_mgr.h"

namespace hccl {
class CommConfiger; // legacy910（libhccl_v2.so）
}

namespace hcomm {

/**
 * @brief 通信配置节点（进程级）：ConfigMgr 值成员，另提供 legacy910 CommConfiger 转发访问器。
 * CommConfiger 单例在 libhccl_v2.so、不进树，由 Legacy950ResMgrProc 锚定（README R8）。
 */
class ConfigMgrProc {
public:
    void DeInit() {}

    ConfigMgr& GetConfigMgr() { return configMgr_; }
    // 定义在 .cc，避免本头引入 legacy 头完整定义
    hccl::CommConfiger& GetCommConfiger();

public:
    ~ConfigMgrProc() = default;

public:
    ConfigMgrProc() = default;

private:
    ConfigMgrProc(const ConfigMgrProc&) = delete;
    ConfigMgrProc& operator=(const ConfigMgrProc&) = delete;

    ConfigMgr configMgr_;
};

} // namespace hcomm

#endif // HCOMM_CONFIG_MGR_CONFIG_MGR_PROC_H
