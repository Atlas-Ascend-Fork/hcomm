/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_EXCHANGE_RDMA_CONN_DTO_LEGACY950_H
#define HCCLV2_EXCHANGE_RDMA_CONN_DTO_LEGACY950_H

// 两份.h 必须一起改：改本文件时同步改 base_comm 侧
// src/base_comm/resources/endpoint_pairs/channels/host/exchange_rdma_conn_dto.h。

#include <string>
#include "binary_stream.h"
#include "serializable.h"
#include "orion_adapter_hccp.h"

namespace Hccl {

constexpr uint32_t HYPER_FEATURE_LEN = 64U;

class ExchangeRdmaConnDto : public Hccl::Serializable {
public:
    ExchangeRdmaConnDto() {};

    ExchangeRdmaConnDto(uint32_t qpn, uint32_t psn, uint32_t gid_idx) : qpn_(qpn), psn_(psn), gid_idx_(gid_idx) {};

    void Serialize(Hccl::BinaryStream& stream) override { stream << qpn_ << psn_ << gid_idx_ << gid_ << feature_; }

    void Deserialize(Hccl::BinaryStream& stream) override { stream >> qpn_ >> psn_ >> gid_idx_ >> gid_ >> feature_; }

    std::string Describe() const override
    {
        return Hccl::StringFormat("ExchangeRdmaConnDto=[qpn=%u, psn=%u, gid_idx=%u]", qpn_, psn_, gid_idx_);
    }

    // RaTypicalQpModify
    uint32_t qpn_;
    uint32_t psn_;
    uint32_t gid_idx_;
    uint8_t gid_[HCCP_GID_RAW_LEN]{};
    char feature_[HYPER_FEATURE_LEN]{};
};

} // namespace Hccl

#endif // HCCLV2_EXCHANGE_RDMA_CONN_DTO_LEGACY950_H
