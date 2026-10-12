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
 * See the License in the specific file packaged with this Software for details.
 */

// 文件职责：公共测试注入 Fake/内存引擎时使用的后端注册名。
// 公共合同测试不引用任何真实后端模块名——Fake fixture 统一使用本测试自有
// 名称，真实后端（sqlite/mariadb）的名称常量只存在于各自模块目录。
#ifndef HCCL_VM_TEST_STORAGE_FAKE_BACKEND_NAME_H
#define HCCL_VM_TEST_STORAGE_FAKE_BACKEND_NAME_H

namespace HcclSim::Storage {
/// 公共测试 Fake 引擎的后端注册名（测试自有名称，不是真实后端模块名）
inline constexpr char kTestFakeBackendName[] = "test_fake";
} // namespace HcclSim::Storage

#endif // HCCL_VM_TEST_STORAGE_FAKE_BACKEND_NAME_H
