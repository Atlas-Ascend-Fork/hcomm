/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef HCOMM_DFX_COMPACT_REPORT_PUB_H
#define HCOMM_DFX_COMPACT_REPORT_PUB_H

#include "common/device/plf_resource_runtime_pub.h"

namespace Hccl {
// 资源层不直接依赖 coll 的 HcclCommDfxLite：
// coll 在加载时注册实现，资源层只通过函数指针通知。
using MarkCompactReportOpFn = PlfMarkCompactReportOp;
inline void SetMarkCompactReportOp(MarkCompactReportOpFn fn) { HcclSetPlfMarkCompactReportOp(fn); }
inline MarkCompactReportOpFn GetMarkCompactReportOp() { return HcclGetPlfMarkCompactReportOp(); }
} // namespace Hccl
#endif
