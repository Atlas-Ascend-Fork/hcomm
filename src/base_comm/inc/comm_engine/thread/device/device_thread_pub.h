/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Device-side (AICPU) thread interfaces for coll_comm internal use only (C++ linkage, NOT exported).
// They are implemented on the AICPU (device) side; on the host they return HCOMM_E_NOT_SUPPORT.
// Host-side thread interfaces are declared in comm_engine/thread/thread_pub.h.
#ifndef HCOMM_DEVICE_THREAD_PUB_H
#define HCOMM_DEVICE_THREAD_PUB_H

#include <stddef.h>
#include "hcomm_res_defs.h"
#include "hcomm_res_entity_defs.h"
#include "common/result_defs_pub.h"

/**
 * @brief Create and init an AICPU thread described by uniqueId; register its ownership and return the handle.
 * @param[in] uniqueId thread unique id string (not required to be null-terminated)
 * @param[in] uniqueIdLen length of uniqueId in bytes
 * @param[out] handle output thread handle
 * @return HcommResult HCOMM_SUCCESS on success, error code otherwise
 */
HcommResult HcommThreadCreateAicpu(const char* uniqueId, size_t uniqueIdLen, ThreadHandle* handle);

/**
 * @brief Fill the Host-preallocated entity from a registered AICPU thread.
 * @param[in] handle thread handle returned by HcommThreadCreateAicpu
 * @param[in,out] entity entity with preallocated SQ context and notify buffer
 * @return HcommResult HCOMM_SUCCESS on success, error code otherwise
 */
HcommResult HcommThreadFillEntity(ThreadHandle handle, HcommThreadEntity* entity);

/**
 * @brief Resolve a device entity handle to its compatibility thread handle; raw thread handles are preserved.
 * @param[in,out] handle readable device entity or raw thread handle, valid for the duration of the call
 * @return HcommResult HCOMM_SUCCESS on success, error code otherwise
 */
HcommResult HcommThreadUnwrapHandle(ThreadHandle* handle);

/**
 * @brief Query the StreamLite pointer associated with a thread (opaque, for stream/task-queue access).
 * @param[in] handle thread handle
 * @param[out] streamLite output StreamLite pointer (void*)
 * @return HcommResult HCOMM_SUCCESS on success, error code otherwise
 * @note 返回的是线程对象内部的裸指针：随线程销毁（HcommThreadDestroy）而失效。使用期间调用方须保证
 *       该句柄仍注册于 base_comm 注册表（如持有上层 ThreadMgr 读锁遍历、析构走写锁），否则可能悬垂。
 */
HcommResult HcommThreadGetStreamLite(ThreadHandle handle, void** streamLite);

/**
 * @brief Query whether a thread is a GE-exported CPU_TS stub thread (Thread::IsFakeDeviceRes), i.e. it holds no
 *        device stream resources (no StreamLite/pImpl_/Rtsq). Used to split real/stub threads by an explicit flag
 *        instead of probing StreamLite (which is null for non-950 real threads too).
 * @param[in] handle thread handle
 * @param[out] isFake true if it is a stub thread
 * @return HcommResult HCOMM_SUCCESS on success, error code otherwise
 */
HcommResult HcommThreadIsFakeDeviceRes(ThreadHandle handle, bool* isFake);

/**
 * @brief Register the rtsq task cache callbacks for a thread (used by the AICPU task cache).
 * @param[in] handle thread handle
 * @return HcommResult HCOMM_SUCCESS on success, error code otherwise
 */
HcommResult HcommThreadRegisterCacheCallback(ThreadHandle handle);

/**
 * @brief Destroy a thread created by HcommThreadCreateAicpu and release its ownership.
 * @param[in] handle thread handle
 * @return HcommResult HCOMM_SUCCESS on success, error code otherwise
 */
HcommResult HcommThreadDestroy(ThreadHandle handle);

/**
 * @brief Callback invoked after a thread created by HcommThreadCreateAicpu is destroyed.
 * @param[in] handle the destroyed thread handle
 */
typedef void (*HcommThreadDestroyCallback)(ThreadHandle handle);

/**
 * @brief Register a process-level callback fired after every HcommThreadDestroy, so upper layers holding a cached
 *        raw handle (e.g. the order-control thread Registry) can drop stale entries and avoid address-reuse
 *        misjudgment. Single slot per process; a later registration overwrites the previous one.
 * @param[in] callback callback to register (nullptr clears it)
 * @return HcommResult HCOMM_SUCCESS on success, error code otherwise
 */
HcommResult HcommThreadRegisterDestroyCallback(HcommThreadDestroyCallback callback);

#endif // HCOMM_DEVICE_THREAD_PUB_H
