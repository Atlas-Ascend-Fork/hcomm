/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_ROCE_HOST_LOC_CONN_LITE_H_
#define HCCLV2_ROCE_HOST_LOC_CONN_LITE_H_

#include "rma_conn_lite.h"
#include "binary_stream.h"
#include "not_support_exception.h"
#include "log.h"

#include "roce_vendor_base_ops.h"

namespace Hccl {
class RoceHostLocConnLite {
public:
    explicit RoceHostLocConnLite(std::vector<char>& uniqueId);
    ~RoceHostLocConnLite();

    std::string Describe();

    // ========== 厂商初始化接口 ==========
    void GetVendorOps();
    HcclResult CheckVendorOp();

    // ========== 数据面：RMA 数据传输 ==========
    void Read(
        const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, const SqeConfigLite& cfg, u64& dbAddr, u64& dbValue);

    void Write(
        const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, const SqeConfigLite& cfg, u64& dbAddr, u64& dbValue);

    void WriteReduce(
        const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, const SqeConfigLite& cfg, DataType dataType,
        ReduceOp reduceOp, u64& dbAddr, u64& dbValue);

    void WriteWithNotify(
        const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, const RmaBufSliceLite& locNotify,
        const RmtRmaBufSliceLite& notify, const SqeConfigLite& cfg, u64& dbAddr, u64& dbValue);

    void WriteReduceWithNotify(
        const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt, const RmaBufSliceLite& locNotify,
        const RmtRmaBufSliceLite& notify, const SqeConfigLite& cfg, DataType dataType, ReduceOp reduceOp, u64& dbAddr,
        u64& dbValue);

    HcclResult PollCq(int32_t numEntries, int32_t timeOut, std::vector<int32_t>& errList, u64& dbAddr, u64& dbValue);

private:
    void ParseSqContext(std::vector<char>& data);
    void ParseCqContext(std::vector<char>& data);

    uint32_t dmaMode_{0};
    RdmaSqContextLite sqContext{};
    RdmaCqContextLite cqContext{};

    // ========== 厂商 Ops（工厂模式，负责具体厂商 ops 创建）==========
    std::unique_ptr<RoceBaseOps> rdmaOps_ = nullptr;

    // ========== 辅助分片写入函数 ==========
    void DoSlice(
        const RmaBufSliceLite& loc, const RmtRmaBufSliceLite& rmt,
        const std::function<void(const RmaBufSliceLite&, const RmtRmaBufSliceLite&)>& op) const;
};

} // namespace Hccl

#endif
