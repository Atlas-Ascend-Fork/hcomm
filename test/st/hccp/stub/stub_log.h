/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * Shared log macros for the hccp test driver stubs (urma / ibverbs / rdma_lite / hns / hrn3).
 *
 * Usage contract (t2 infrastructure): a stub library only has to link unified_dlog_stub and call
 * these macros; dlog_stub.c routes every line to the per-case log tree automatically
 * (host process -> <debug|run>/<Case>/plog/host-phy<x>-pid<p>.log, device process
 * -> <debug|run>/<Case>/device-<x>/hccp_service-<pid>.log). Stubs never build log paths
 * themselves. Module ids: DRV/NET/HCCP/ROCE come from the product log_types.h enum; URMA has no
 * product id (the enum is full), so it uses the test-private STUB_URMA_MODULE_ID below.
 */

#ifndef STUB_LOG_H
#define STUB_LOG_H

#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#include <dlog_pub.h>

/* Test-private module id for the URMA (ub engine) stub. The product module enum in log_types.h is
 * full (INVLID_MOUDLE_ID = 76), so all test stubs share this out-of-range id; dlog_stub.c renders
 * it as "URMA". Keep the value in sync with STUB_URMA_MODULE_ID in dlog_stub.c. */
#define STUB_URMA_MODULE_ID 100

/* URMA (ub engine) stub log macros, mirroring DRV_LOG_* of ascend_hal_stub.h: non-error levels
 * are gated by CheckLogLevel (ASCEND_GLOBAL_LOG_LEVEL controls them), ERROR is always recorded
 * with file/line/pid/tid/func context. */
#define URMA_LOG_DEBUG(fmt, ...)                                                       \
    do {                                                                               \
        if (CheckLogLevel(STUB_URMA_MODULE_ID, DLOG_DEBUG) == 1) {                     \
            DlogRecord(STUB_URMA_MODULE_ID, DLOG_DEBUG, "[URMA] " fmt, ##__VA_ARGS__); \
        }                                                                              \
    } while (0)
#define URMA_LOG_INFO(fmt, ...)                                                       \
    do {                                                                              \
        if (CheckLogLevel(STUB_URMA_MODULE_ID, DLOG_INFO) == 1) {                     \
            DlogRecord(STUB_URMA_MODULE_ID, DLOG_INFO, "[URMA] " fmt, ##__VA_ARGS__); \
        }                                                                             \
    } while (0)
#define URMA_LOG_WARN(fmt, ...)                                                       \
    do {                                                                              \
        if (CheckLogLevel(STUB_URMA_MODULE_ID, DLOG_WARN) == 1) {                     \
            DlogRecord(STUB_URMA_MODULE_ID, DLOG_WARN, "[URMA] " fmt, ##__VA_ARGS__); \
        }                                                                             \
    } while (0)
#define URMA_LOG_ERROR(fmt, ...)                                                                              \
    do {                                                                                                      \
        DlogRecord(                                                                                           \
            STUB_URMA_MODULE_ID, DLOG_ERROR, "[URMA][%s:%d]pid:%d,tid:%ld,%s : " fmt, __FILENAME__, __LINE__, \
            (int)getpid(), (long)syscall(SYS_gettid), __func__, ##__VA_ARGS__);                               \
    } while (0)

/* ROCE (rdma family: ibverbs / ascend_rdma_lite / hns / hrn3 provider stubs) log macros, same
 * shape as the URMA ones but on the product module id ROCE (log_types.h). */
#define ROCE_LOG_DEBUG(fmt, ...)                                        \
    do {                                                                \
        if (CheckLogLevel(ROCE, DLOG_DEBUG) == 1) {                     \
            DlogRecord(ROCE, DLOG_DEBUG, "[ROCE] " fmt, ##__VA_ARGS__); \
        }                                                               \
    } while (0)
#define ROCE_LOG_INFO(fmt, ...)                                        \
    do {                                                               \
        if (CheckLogLevel(ROCE, DLOG_INFO) == 1) {                     \
            DlogRecord(ROCE, DLOG_INFO, "[ROCE] " fmt, ##__VA_ARGS__); \
        }                                                              \
    } while (0)
#define ROCE_LOG_WARN(fmt, ...)                                        \
    do {                                                               \
        if (CheckLogLevel(ROCE, DLOG_WARN) == 1) {                     \
            DlogRecord(ROCE, DLOG_WARN, "[ROCE] " fmt, ##__VA_ARGS__); \
        }                                                              \
    } while (0)
#define ROCE_LOG_ERROR(fmt, ...)                                                                              \
    do {                                                                                                      \
        DlogRecord(                                                                                           \
            ROCE, DLOG_ERROR, "[ROCE][%s:%d]pid:%d,tid:%ld,%s : " fmt, __FILENAME__, __LINE__, (int)getpid(), \
            (long)syscall(SYS_gettid), __func__, ##__VA_ARGS__);                                              \
    } while (0)

#endif /* STUB_LOG_H */
