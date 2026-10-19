/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCP_SYSTEM_TEST_COMMON_H
#define HCCP_SYSTEM_TEST_COMMON_H

#define COLOR_BLUE "\033[94m"
#define COLOR_RED "\033[31m"
#define COLOR_RESET "\033[0m"
#define COLOR_MAIN "\033[32m"
#define COLOR_CLEANUP "\033[33m"
#define COLOR_DEVICE "\033[36m"
#define COLOR_INFO "\033[94m"
#define COLOR_ERROR "\033[31m"
#define COLOR_SUCCESS "\033[1;32m"

#define TEST_LOG_INFO(phy_id, fmt, ...) \
    printf(COLOR_BLUE "[INFO]" COLOR_RESET "[HOST-%d] " fmt "\n", phy_id, ##__VA_ARGS__)

#define TEST_LOG_ERROR(phy_id, fmt, ...) \
    printf(COLOR_RED "[ERROR]" COLOR_RESET "[HOST-%d] " fmt "\n", phy_id, ##__VA_ARGS__)

#endif
