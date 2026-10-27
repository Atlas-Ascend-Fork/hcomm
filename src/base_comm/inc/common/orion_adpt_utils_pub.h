/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_ORION_ADPT_UTILS_PUB_H
#define HCOMM_ORION_ADPT_UTILS_PUB_H

#include "hccl/hccl_types.h"
#include "hcomm_res_defs.h"
#include "hccl_net_dev_defs.h"

// Orion
#include "ip_address.h"

namespace hcomm {

// 仅暴露 L2（coll_communicator_mgr）使用的地址转换接口；
// UB 深度校验/LinkData 组装等其余接口仅 base_comm 内部使用。
HcclResult CommAddrToIpAddress(const CommAddr& commAddr, Hccl::IpAddress& ipAddr);

} // namespace hcomm

#endif // HCOMM_ORION_ADPT_UTILS_PUB_H
