/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UB_DEVICE_ENDPOINT_BASE_H
#define UB_DEVICE_ENDPOINT_BASE_H

#include <cstdint>
#include "endpoint.h"

namespace hcomm {
/**
 * @note 职责：UB 三类 device endpoint（Urma/Uboe/UbRtp）的公共中间基类。
 *       承载 DevEpMemTable（device 侧 endpoint 本地内存活账本表）5 个钩子的唯一实现与表地址成员，
 *       消除三处逐字重复（检视意见）；以 override 基类默认 no-op 虚函数的形态暴露（与
 *       RecordLocalMemHandle 家族同款惯例），调用方经基类指针虚分发，非 UB 走默认 no-op。
 *       其余 endpoint 类型（Plugin/CpuUrma/CpuRoce/Hccs 等）仍直接继承 Endpoint，行为不变。
 */
class UbDeviceEndpointBase : public Endpoint {
public:
    explicit UbDeviceEndpointBase(const EndpointDesc& endpointDesc) : Endpoint(endpointDesc) {}

    // 表的生命周期仅由 HcommEndpointDestroy 负责，析构兜底下发 kernel 不安全，残留表由进程退出回收。
    // 退出期（ProcessExitFlag 置位后）InitDevEpMemTable 注册 atexit 锚点，
    // ReleaseDevEpMemTable/OnLocalMemUnregistered 跳过 kernel 下发、仅清 host 状态，残留由进程退出回收
    ~UbDeviceEndpointBase() override = default;

    // 表生命周期：endpoint 创建时建表（init kernel + D2H 读回）/ 销毁时删表（destroy kernel）
    HcclResult InitDevEpMemTable() override;
    HcclResult ReleaseDevEpMemTable() override; // 退出期跳过 destroy kernel（runtime 不稳定期下发会 core）
    // 表地址（0=无表），建链序列化（uniqueIdV2）下发用
    uint64_t GetDevEpMemTablePtr() const override;
    // Reg 成功后 UPSERT 表条目（对象存活，读四字段安全）/
    // Unreg 成功后无条件 REMOVE（纯值入参：对象可能已被 mgr 注销析构，禁传 handle）；
    // 退出期跳过 REMOVE kernel，device 表残留条目随进程退出回收
    HcclResult OnLocalMemRegistered(void* memHandle) override;
    HcclResult OnLocalMemUnregistered(uint64_t addr, uint64_t size) override;

private:
    uint64_t devEpMemTablePtr_{0}; // device 侧活账本表地址（RunAicpuEpMemTableInit 读回）
};
} // namespace hcomm

#endif // UB_DEVICE_ENDPOINT_BASE_H
