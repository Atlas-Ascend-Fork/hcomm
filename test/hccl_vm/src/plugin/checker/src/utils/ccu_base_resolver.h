/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CHECKER_CCU_BASE_RESOLVER_H
#define HCCLV2_CHECKER_CCU_BASE_RESOLVER_H

#include <cstdint>
#include <map>
#include <vector>

namespace sim {
namespace devspec {
    class CcuAddressSpace;
}
} // namespace sim

namespace HcclSim {

/**
 * @brief 解析全部 device 的 CCU 资源基址表：deviceId → (dieId → 基址)。
 *
 * ★ 取代原 StorageManager::InitCcuInfo 中的 0x123123123/0x456456456 假基址
 *   （设计文档 §2.2 #8 / §4.5.5）：解析口径与全项目唯一入口
 *   sim::runtime::GetCcuResourceBaseAddr 完全一致（DB Ccu.resource_addr 优先、yaml 规格回退），
 *   die 数取自 device 规格 die_num；解析失败写 0（显式"未知"），不回退假地址。
 *   ★ 但**不逐 (device, die) 调用**该入口：本函数在 checker 启动时无条件执行，
 *   逐卡查询会让读库次数随卡数线性放大（双 die 1024 卡 = 3073 次），故改为
 *   Device / Ccu 各一次全表扫 + 规格走进程级常驻的 DeviceSpecRegistry，共 2 次读库。
 *
 * ★ 独立编译单元的原因：checker 内部头 task_meta_defs.h 与全局 sim_common_defs.h
 *   共用 include guard（HCCL_COMMON_DEFS_H，二者互斥），storage_manager.cc 这类
 *   先包含 checker 内部头的 TU 再包含 runtime_state DB 头会因 guard 占位丢失符号
 *   （如 URMA_EID_LEN）。本文件只包含 runtime_state/device_spec 头，
 *   与 dfx/ccu_resource_stats_topo.cc 的隔离方式一致。
 */
std::map<uint32_t, std::vector<uint64_t>> ResolveAllCcuBaseAddrs();

/**
 * @brief deviceId 对应 device 规格的 CCU 地址空间计算器（进程级缓存，只读安全）。
 *
 * 区域偏移/步长/容量/MS 几何全部来自 config/device/*.yaml（经 DeviceSpecRegistry），
 * 取代 runner/checker 各自的 XN_*_ADDR_A6 / CCU_MS_START_ADDR 等字面量常量表。
 * @return device 规格解析失败返回 nullptr（打 ERROR），调用方须显式失败。
 */
const sim::devspec::CcuAddressSpace* GetCcuAddrSpaceByDeviceId(uint32_t deviceId);

} // namespace HcclSim

#endif // HCCLV2_CHECKER_CCU_BASE_RESOLVER_H
