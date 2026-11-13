/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CHANGED_RANK_TABLE_READER_H
#define HCCLV2_CHANGED_RANK_TABLE_READER_H

#include <string>
#include "changed_rank_info.h"

namespace Hccl {

// changed ranktable 域的快速 Reader。
class ChangedRankTableReader {
public:
    HcclResult ReadString(const std::string& json, ChangedRankInfo& out) const;
};

} // namespace Hccl

#endif // HCCLV2_CHANGED_RANK_TABLE_READER_H
