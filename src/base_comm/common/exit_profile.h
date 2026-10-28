/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_EXIT_PROFILE_H
#define HCOMM_EXIT_PROFILE_H

namespace hcomm {

/**
 * @brief 析构链退出 profile：GRACEFUL 完整动作链（默认，进程退出兜底走它）；
 *        RESET 跳过触设备动作（设备可能已不可用），仅 host 侧清理、仅显式调用触发。
 *        语义与契约见 README R5。
 */
enum class ExitProfile { EXIT_PROFILE_GRACEFUL = 0, EXIT_PROFILE_RESET };

} // namespace hcomm

#endif // HCOMM_EXIT_PROFILE_H
