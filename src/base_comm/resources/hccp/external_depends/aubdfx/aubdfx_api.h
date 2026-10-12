/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * Note: This file is copied from aubdfx. Do not modify manually.
 *       To update, sync from the original aubdfx source.
 */

#ifndef AUBDFX_API_H
#define AUBDFX_API_H

#include <stddef.h>
#include <stdint.h>

#define AUBDFX_ATTRI_VISI_DEF __attribute__((visibility("default")))

enum ServiceType { URMA_TYPE = 0, UBMEM_TYPE, UNIC_TYPE };

enum UbNotifyCmd { SERVICE_ERRINFO_CMD = 0, NET_DFX_SNAPSHOT_CMD, CMD_MAX };

struct ub_service_errinfo {
    uint16_t sub_cmd;
    uint16_t rsv2;
    unsigned char dieid;
    unsigned char ueid;
    unsigned char servicetype;
    unsigned char errortype;
    uint32_t srceid[4];
    uint32_t dsteid[4];
    uint32_t rsv[2];
    uint64_t value;
};

struct ub_net_dfx_info {
    pid_t hccl_pid;
    pid_t hccp_pid;
    uint8_t status; // status = 1 : start, status = 0 : end
    uint8_t rsv[3];
};

AUBDFX_ATTRI_VISI_DEF int aubdfx_notify_event(unsigned int die_id, unsigned int notify_cmd, void *data,
    unsigned int data_len);

#endif // AUBDFX_API_H
