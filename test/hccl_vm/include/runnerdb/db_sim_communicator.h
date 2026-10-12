/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DB_SIM_COMMUNICATOR_H
#define DB_SIM_COMMUNICATOR_H

#include <cstdint>
#include <string>
#include <vector>

namespace sim {
struct CommunicatorMemberInfo {
    uint64_t memberId;
    uint64_t deviceId;
    uint32_t rankId;
};

bool GetCommunicatorMembers(uint64_t commId, std::vector<CommunicatorMemberInfo>& members);
// 根据通信域成员 ID 查询其所属通信域名称。
bool GetCommunicatorName(uint64_t commId, std::string& commName);
bool GetCommunicatorIdentity(uint64_t commId, std::string& commName, uint64_t& commHash);
} // namespace sim

#endif // DB_SIM_COMMUNICATOR_H
