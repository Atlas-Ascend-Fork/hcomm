/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_SHARE_CCL_BUFFER_MANAGER_H
#define HCCLV2_SHARE_CCL_BUFFER_MANAGER_H

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include "types.h"
#include "dev_buffer.h"

namespace Hccl {

// 按device粒度管理共享CCLBuffer：bufferName相同的通信域复用同一片DevBuffer
class ShareCclBufferManager {
public:
    static ShareCclBufferManager& GetInstance();
    static ShareCclBufferManager& GetInstance(DevId deviceLogicId);

    ShareCclBufferManager(const ShareCclBufferManager&) = delete;
    ShareCclBufferManager& operator=(const ShareCclBufferManager&) = delete;

    // 获取指定名称的共享CCLBuffer，不存在时创建；同名buffer申请大小不一致时返回错误，避免越界访问
    HcclResult GetShareCclBuffer(const std::string& bufferName, u64 bufferSize, std::shared_ptr<DevBuffer>& cclBuffer);

    // 使用同一共享CCLBuffer的不同通信域，算子必须下发到同一条Stream；同一通信域切换Stream不视为冲突
    HcclResult CheckCclBufferConflict(const std::string& commId, const std::string& bufferName, s32 streamId);

    // 通信域销毁时解除其Stream绑定，避免残留绑定影响后续使用同名buffer的通信域
    void ReleaseCclBufferStreamBinding(const std::string& commId, const std::string& bufferName);

private:
    ShareCclBufferManager() = default;
    ~ShareCclBufferManager() = default;

    std::mutex lock_;
    // 只持有弱引用，不做显式释放：最后一个持有shared_ptr的通信域销毁时buffer自动释放，
    std::unordered_map<std::string, std::weak_ptr<DevBuffer>> shareBufferMap_;
    std::unordered_map<std::string, u64> bufferSizeMap_;
    // bufferName -> (commId -> streamId)，按通信域记录Stream绑定，通信域销毁后可单独解绑
    std::unordered_map<std::string, std::unordered_map<std::string, s32>> streamMap_;
};

} // namespace Hccl
#endif // HCCLV2_SHARE_CCL_BUFFER_MANAGER_H
