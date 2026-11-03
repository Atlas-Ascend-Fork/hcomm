/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_UB_CONN_LITE_MGR_H_
#define HCCLV2_UB_CONN_LITE_MGR_H_

#include "ub_conn_lite.h"
#include <memory>
#include <unordered_map>
#include <mutex>
#include <shared_mutex>
#include <vector>

namespace Hccl {

class UbTransportLiteImpl;

class UbConnLiteMgr {
public:
    static UbConnLiteMgr& GetInstance();

    ~UbConnLiteMgr();

    RmaConnLite* Get(std::vector<char>& uniqueId);

    void Clear(std::vector<char>& uniqueId);

    void AppendCompletedCis(const UbTransportLiteImpl* transport, const std::pair<u16, u16>* slots, size_t count);

    void RegisterCiTracker(const UbTransportLiteImpl* transport, RmaConnLite* conn);
    void UnRegisterCiTracker(const UbTransportLiteImpl* transport);

private:
    UbConnLiteMgr();

    std::string GetKey(const UbConnLiteParam& liteParam) const;

    std::unordered_map<std::string, std::unique_ptr<UbConnLite>> ubConnLiteMap;

    mutable std::shared_mutex mtx_;
    std::unordered_map<const UbTransportLiteImpl*, RmaConnLite*> ciTrackerMap_;

    bool IsExist(const std::string& key);
};
} // namespace Hccl

#endif // HCCL_AICPU_RESOURCE_AI_CPU_RESOUCES_H_
