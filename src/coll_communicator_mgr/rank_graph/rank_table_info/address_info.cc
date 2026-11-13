/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "address_info.h"

#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include "invalid_params_exception.h"
#include "exception_util.h"
#include "adapter_error_manager_pub.h"
#include "common/hccl_log_keywords_pub.h"

namespace Hccl {
using namespace std;

std::string AddressInfo::Describe() const
{
    return StringFormat(
        "AddressInfo[addrType=%s, addr=%s, backupAddrSize=%u, planeId=%s, portsize=%u socketPort_=%u]",
        addrType.Describe().c_str(), addr.Describe().c_str(), static_cast<u32>(backupAddrs.size()), planeId.c_str(),
        static_cast<u32>(ports.size()), socketPort_);
}

AddressInfo::AddressInfo(BinaryStream& binStream)
{
    IpAddress address(binStream);
    addr = address;
    u32 addrTypeInt{0};
    binStream >> addrTypeInt;
    addrType = static_cast<AddrType::Value>(addrTypeInt);
    size_t portsSize{0};
    binStream >> portsSize;
    for (u32 i = 0; i < portsSize; i++) {
        string port;
        binStream >> port;
        ports.emplace(port);
    }
    binStream >> planeId;
    binStream >> socketPort_;
    size_t backupAddrSize{0};
    binStream >> backupAddrSize;
    CHK_PRT_THROW(
        backupAddrSize > MAX_VALUE_BACKUP_ADDR_SIZE,
        HCCL_ERROR("[%s] backup_addr size[%zu] exceeds max[%u].", __func__, backupAddrSize, MAX_VALUE_BACKUP_ADDR_SIZE),
        InvalidParamsException, "backup_addr size exceeds limit");
    for (size_t i = 0; i < backupAddrSize; i++) {
        IpAddress backupAddr(binStream);
        backupAddrs.emplace_back(std::move(backupAddr));
    }
}

HcclResult AddressInfo::GetBinStream(BinaryStream& binStream) const
{
    if (ports.size() == 0) {
        RPT_INPUT_ERR(
            true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>({"0", "ports", "non-empty"}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] ports size is zero", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RANKTABLE_CHECK.c_str(), HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA));
        return HCCL_E_PARA;
    }
    addr.GetBinStream(binStream);
    binStream << static_cast<u32>(addrType);
    binStream << ports.size();
    for (auto& it : ports) {
        binStream << it;
    }
    binStream << planeId;
    binStream << socketPort_;
    binStream << backupAddrs.size();
    for (const auto& backupAddr : backupAddrs) {
        backupAddr.GetBinStream(binStream);
    }
    return HCCL_SUCCESS;
}

} // namespace Hccl
