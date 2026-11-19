/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// ccu_base_resolver 的 UT 桩：test_storage_manager_checker 保持轻依赖（只链 log），
// 且测试文件自带 sim::runtime::GetCommunicatorMembers 等 mock，链入真 runtime_state
// 会产生重复符号。真实实现（DB/device 规格解析）在主 checker 二进制中编译与运行；
// UT 环境本无 DB 会话，真实实现也只会返回空表——桩的行为与之一致。

#include "ccu_base_resolver.h"

namespace HcclSim {

std::map<uint32_t, std::vector<uint64_t>> ResolveAllCcuBaseAddrs() { return {}; }

// UT 环境无 DB 会话，真实实现读 VirtualMemBlock 也只会返回空表（调用方退化为不换算）。
std::vector<DevMappedVirRange> ResolveAllDevMappedRanges() { return {}; }

const sim::devspec::CcuAddressSpace* GetCcuAddrSpaceByDeviceId(uint32_t deviceId)
{
    (void)deviceId;
    return nullptr;
}

} // namespace HcclSim
