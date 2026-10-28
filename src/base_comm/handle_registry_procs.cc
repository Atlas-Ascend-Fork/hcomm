/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "handle_registry_procs.h"

#include "channel.h"
#include "log.h"

namespace hcomm {

void ChannelHandleRegistryProc::DeInit(ExitProfile profile)
{
    if (profile == ExitProfile::EXIT_PROFILE_RESET) {
        return;
    }
    std::unordered_map<ChannelHandle, std::shared_ptr<Channel>> channels;
    {
        std::lock_guard<std::mutex> lock(channelMapMtx_);
        if (!channelMap_.empty()) {
            HCCL_WARNING(
                "[%s] %zu channel(s) still in map during DeInit, force releasing", __func__, channelMap_.size());
            channels = std::move(channelMap_);
        }
        channelD2HMap_.clear();
        channelH2DMap_.clear();
    }
    // 锁外触发 ~Channel：析构经窄访问器归还 socket/notify 会触设备子树
    channels.clear();
}

} // namespace hcomm
