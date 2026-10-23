/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LLT_DCA_STUB_H
#define LLT_DCA_STUB_H

#include <cstdint>

namespace hcomm {

// DetectConnectionAnomalies 打桩计数接口：供 UT 断言 DCA 真实调用路径（Init/Deinit 配对、AddIpQueue 触发）
uint32_t DcaStubInitCallCount();
uint32_t DcaStubDeinitCallCount();
uint32_t DcaStubAddIpQueueCallCount();
void DcaStubResetCounts();

} // namespace hcomm

#endif // LLT_DCA_STUB_H
