/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NPU_NIC_AFFINITY_H
#define NPU_NIC_AFFINITY_H

#include <stddef.h>
#include "topo_addr_info_err.h"
#include "rank_info_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/** IP 地址及其类型，作为 GetRoceIpFromXml 等解析函数的出参 */
typedef struct {
    char addr[MAX_NET_ADDR_LEN];
    NpuNicAddrType type;
} NpuNicIp;

/**
 * 从 virtualTopology.xml 获取指定 NPU 的 RoCE IP 及其类型。
 *
 * 解析 XML → 构建 NPU-NIC 亲和矩阵 → 轮询匹配 → HCA名 → eth → getifaddrs → IP
 * 无 XML 时，通过 hal_get_topo_info_by_device_id_and_nic_name 驱动接口构建亲和矩阵。
 *
 * @param npuId NPU 物理 ID
 * @param out   输出 IP 地址（含字符串和类型）
 * @return HCCL_SUCCESS=成功, 其他=错误码
 */
TopoAddrResult GetRoceIpFromXml(int npuId, NpuNicIp* out);

#ifdef __cplusplus
}
#endif

#endif /* NPU_NIC_AFFINITY_H */
