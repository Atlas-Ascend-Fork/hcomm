/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_RMA_HOST_NET_CONNECTION_H
#define HCCLV2_RMA_HOST_NET_CONNECTION_H

#include <cstdint>
#include <memory>

#include "enum_factory.h"
#include "hccl_common.h"
#include "exchange_rdma_conn_dto.h"

// Orion
#include "../../../../../../legacy/ascend950/unified_platform/resource/socket/socket.h"
#include "orion_adapter_hccp.h"

namespace hcomm {

class HostRdmaConnection {
public:
    struct QpAttrDto {
        uint32_t qpn{UINT32_MAX};
        uint32_t psn{UINT32_MAX};
        uint32_t gid_idx{0};
        unsigned char gid[HCCP_GID_RAW_LEN];
        char feature[HYPER_FEATURE_LEN];

        bool IsValid() const
        {
            if (qpn == UINT32_MAX || psn == UINT32_MAX) {
                return false;
            }
            return true;
        }
    };
    struct RoceAttr {
        uint32_t queueNum{0};
        uint32_t retryCnt{0};
        uint32_t retryInterval{0};
        uint32_t tc{0};
        uint32_t sl{0};
        uint32_t sqDepth{0};
        uint32_t scqDepth{0};
    };
    MAKE_ENUM(RdmaConnStatus, CLOSED, INIT, QP_CREATED, QP_MODIFIED, SOCKET_TIMEOUT)
    HostRdmaConnection(Hccl::Socket* socket, RdmaHandle rdmaHandle);

    HcclResult Init();
    HcclResult CreateQp();
    // HcclResult GetLocQpAttr(std::unique_ptr<Hccl::Serializable> &locQpAttrserial);
    // HcclResult ParseRmtQpAttr(const Hccl::Serializable &rmtQpAttrSerial);
    HcclResult GetExchangeDto(std::unique_ptr<Hccl::Serializable>& serial);
    HcclResult ParseRmtExchangeDto(const Hccl::Serializable& rmtDto); // 解析收到的远端序列化数据
    HcclResult ModifyQp();

    ~HostRdmaConnection();

    std::string Describe() const;
    Hccl::QpInfo& GetQpInfo() { return qpInfo_; }
    void SetLbValue(s32 lbValue) { qpInfo_.lbValue = lbValue; }
    void SetUdpSport(uint32_t udpSport) { qpInfo_.udpSport = udpSport; }
    void SetRoceAttr(const RoceAttr& roceAttr)
    {
        roceAttr_ = roceAttr;
        qpInfo_.serviceLevel = roceAttr.sl;
        qpInfo_.trafficClass = roceAttr.tc;
        qpInfo_.retryCnt = roceAttr.retryCnt;
        qpInfo_.retryInterval = roceAttr.retryInterval;
        qpInfo_.attr.maxWr = roceAttr.sqDepth;
    }
    const RoceAttr& GetRoceAttr() const { return roceAttr_; }

private:
    HcclResult DestroyQp();
    bool isValidQpAttr();

    Hccl::Socket* socket_{nullptr};
    Hccl::RdmaHandle rdmaHandle_{nullptr};
    // OpMode              opMode_{OpMode::OPBASE};

    Hccl::QpInfo qpInfo_;
    RoceAttr roceAttr_{};
    void* sendCompChannel_{nullptr};
    void* recvCompChannel_{nullptr};
    bool isHdcMode_{false};
    RdmaConnStatus rdmaConnStatus_{RdmaConnStatus::CLOSED};
    QpAttrDto rmtQpAttr_{};
    QpAttrDto locQpAttr_{};
};

} // namespace hcomm

#endif // HCCLV2_RMA_HOST_NET_CONNECTION_H
