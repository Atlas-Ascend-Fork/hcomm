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
 * NON-INFRINGEMENT. See the License in the specific file packaged with this Software for details.
 */

/**
 * @file session_test_peer.h
 * @brief 测试专用同伴（friend）：读取 StorageSession 内部控制块的真实同步状态。
 *
 * 仅测试目标包含本文件：StorageSession 声明 friend struct StorageSessionTestPeer，
 * 同伴经内部控制块只读观察点（惰性打开等待者数 / 静止闸门标志）供并发测试精确
 * 编排时序断言。不构成公开业务 API，生产代码不得包含。
 */
#ifndef HCCL_VM_TEST_STORAGE_SESSION_TEST_PEER_H
#define HCCL_VM_TEST_STORAGE_SESSION_TEST_PEER_H

#include "storage/internal/session_control.h"
#include "storage/storage_session.h"

namespace HcclSim::Storage {

/// 测试同伴：转发内部控制块只读同步状态（SessionControlBlock::SyncState）。
struct StorageSessionTestPeer {
    static SessionControlBlock::SyncStateForTest SyncState(const StorageSession& session) noexcept
    {
        return session.control_ == nullptr ? SessionControlBlock::SyncStateForTest{} : session.control_->SyncState();
    }
};

} // namespace HcclSim::Storage

#endif // HCCL_VM_TEST_STORAGE_SESSION_TEST_PEER_H
