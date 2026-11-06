/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HOST_MULTI_QP_CONFIG_H
#define HOST_MULTI_QP_CONFIG_H

#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "hccl/hccl_types.h"

namespace hccl {

struct HostMultiQpDeviceConfig {
    uint32_t qpCount{0};
    std::vector<uint16_t> udpPorts;
};

/**
 * @brief Host DPU多QP配置。
 *
 * CollCommConfigMgr初始化时先解析HCCL_HOST_RDMA_UDP_PORTS_LIST，再使用/etc/hcomm.cfg中完整合法的单卡配置覆盖，
 * 最终按物理设备缓存生效配置。配置文件不可用时保留环境变量配置；解析或校验失败时返回错误，不缓存配置。
 * 环境变量仅配置端口列表，各卡端口段须非空；文件每卡mode、count、ports三项键必填，缺键报错。
 * 文件ports值可空，以空列表缓存并覆盖同卡环境变量端口，表示文件不指定源端口。
 * 空列表在应用时回退MultiQpSrcPort.cfg；无匹配项时使用默认端口。
 */
class HostMultiQpConfig {
public:
    HcclResult Parse();
    uint32_t GetQpCount(uint32_t devicePhyId) const;
    const std::vector<uint16_t>* GetUdpPorts(uint32_t devicePhyId) const;

private:
    std::unordered_map<uint32_t, HostMultiQpDeviceConfig> configs_;
    std::mutex parseMutex_;
    bool parsed_{false};
};

} // namespace hccl

#endif // HOST_MULTI_QP_CONFIG_H
