/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/device/plf_resource_runtime_pub.h"

// 临时例外：本文件仅用于 legacy 日落前的跨 DSO 回调注册，不承载业务功能。
// 清理条件：legacy 资源层迁移完成、资源层不再位于 ccl_kernel_plf 时，本文件与
// plf_resource_runtime.cc 一并删除。
namespace {

__attribute__((constructor)) void RegisterPlfResourceRuntimeOps()
{
    HcclRegisterPlfResourceRuntimeOps(HcclGetPlfResourceRuntimeOps());
}

} // namespace
