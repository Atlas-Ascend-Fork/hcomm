/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_C_ADPT_COMMON_H
#define HCOMM_C_ADPT_COMMON_H

#include "hccl/hccl_res.h"
#include "hcomm_res_defs.h"
#include "config_plf_log_v2.h"
#include "plf_debug_config.h"
#include "hcomm_primitives.h"

/* 跨域共享的基础设施函数 */
HcclResult RefreshEndpointContext(const EndpointDesc& endpointDesc);
HcclResult RefreshCommEngineContext(CommEngine engine);

namespace Hccl {

/* 数据面接口 512B 地址对齐检查日志：受 PLF_DATA_OP 门控，仅记录不拦截；cpu/aicpu_ts 适配文件共用 */
inline void LogAddrAlignCheck512B(const char* funcName, const void* dst, const void* src)
{
    if ((GetPlfDebugConfigValue() & PLF_DATA_OP) == 0) {
        return;
    }
    if (!IsAddrAligned512B(dst) || !IsAddrAligned512B(src)) {
        PLF_CONFIG_INFO(
            PLF_DATA_OP, "[%s] dst[%p] or src[%p] is not aligned to 512B.", funcName, const_cast<void*>(dst),
            const_cast<void*>(src));
    }
}

/* 批量版：跳过无搬运地址的 NOTIFY_RECORD/INVALID；write 与各数据类子结构 dst/src 偏移一致，借道 transferInfo.write
 * 取值覆盖所有数据类 desc */
inline void
LogDescsAddrAlignCheck512B(const char* funcName, const HcommBatchTransferDesc* transferDescs, uint32_t transferDescNum)
{
    if (((GetPlfDebugConfigValue() & PLF_DATA_OP) == 0) || (transferDescs == nullptr)) {
        return;
    }
    for (uint32_t i = 0; i < transferDescNum; i++) {
        const HcommBatchTransferDesc& desc = transferDescs[i];
        if ((desc.transType == HCOMM_TRANSFER_TYPE_NOTIFY_RECORD) || (desc.transType == HCOMM_TRANSFER_TYPE_INVALID)) {
            continue;
        }
        if (!IsAddrAligned512B(desc.transferInfo.write.dst) || !IsAddrAligned512B(desc.transferInfo.write.src)) {
            PLF_CONFIG_INFO(
                PLF_DATA_OP, "[%s] desc[%u] dst[%p] or src[%p] is not aligned to 512B.", funcName, i,
                desc.transferInfo.write.dst, desc.transferInfo.write.src);
        }
    }
}

} // namespace Hccl

#endif // HCOMM_C_ADPT_COMMON_H
