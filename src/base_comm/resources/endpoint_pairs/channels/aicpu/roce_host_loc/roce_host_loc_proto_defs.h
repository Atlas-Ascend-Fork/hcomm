/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ROCE_HOST_LOC_PROTO_DEFS_H
#define ROCE_HOST_LOC_PROTO_DEFS_H

namespace hcomm {

// host/device 间 blob 协议常量：notifyNum = RDMA_NOTIFY_NUM * connNum。
// host 侧组包与 device 侧 Init 校验必须使用同一份定义，禁止各自复制。
constexpr unsigned int RDMA_NOTIFY_NUM = 3;

} // namespace hcomm

#endif // ROCE_HOST_LOC_PROTO_DEFS_H
