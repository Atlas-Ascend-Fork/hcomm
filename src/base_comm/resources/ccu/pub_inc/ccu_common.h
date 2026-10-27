/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 兼容转发头：保留旧文件名 ccu_common.h，转发到边界头 ccu_common_pub.h。
// 涉及checker的修改，PR5577合入后，这个转发头可以删除，但是需要checker那边同步修改头文件
#ifndef CCU_COMMON_H
#define CCU_COMMON_H

#include "comm_engine/ccu/ccu_common_pub.h"

#endif // CCU_COMMON_H
