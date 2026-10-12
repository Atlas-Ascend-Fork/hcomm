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
 * You should not use this file except in compliance with the License.
 */

// 测试本地 fe::PlatformInfoManager 桩。
// 生产形态：aclrt_device_stub.cc 的平台信息初始化仅在 HCCL_VM_ENABLE_PYTORCH=1 时调用，
// libhccl_proxy_level2.so 对 fe 符号保持未定义、由 torch_npu 宿主进程在加载期解析
// （CANN 9.2.0 运行时库不导出该符号，属新版 CANN/编译器侧实现）。
// test_aclrt_device_stub 独立链入 aclrt_device_stub.cc，可执行文件要求链接期符号闭合，
// 且测试不启用 PyTorch 路径，故此处提供 no-op 链接桩。
#include "platform/platform_info.h"

#include <cstdint>
#include <new>
#include <string>

namespace fe {
namespace {
    // 未构造的原生存储：桩实现不访问任何成员，仅满足链接闭合。
    alignas(PlatformInfoManager) unsigned char kStubPlatformInfoManagerStorage[sizeof(PlatformInfoManager)];
} // namespace

PlatformInfoManager& PlatformInfoManager::GeInstance()
{
    return *reinterpret_cast<PlatformInfoManager*>(kStubPlatformInfoManagerStorage);
}

PlatformInfoManager& PlatformInfoManager::Instance() { return GeInstance(); }

uint32_t PlatformInfoManager::InitRuntimePlatformInfos(const std::string&)
{
    return 0U; // no-op：测试不经过真实平台信息加载路径
}
} // namespace fe
