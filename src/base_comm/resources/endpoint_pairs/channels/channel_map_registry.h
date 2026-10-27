/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CHANNEL_MAP_REGISTRY_H
#define HCOMM_CHANNEL_MAP_REGISTRY_H

#include <memory>
#include <mutex>
#include <unordered_map>
#include "endpoint_pair/channel/channel_pub.h"

namespace hcomm {

class Channel;

// Plugin-channel 注册表（handle -> Channel）访问函数。注册表变量（g_ChannelMap/g_ChannelMapMtx）为
// Plugin-channel 注册表（handle -> Channel）访问接口。注册表变量（g_ChannelMap/g_ChannelMapMtx）为
// channel_process.cc 内的文件级静态变量，不对外暴露；此处仅提供增删清接口，使注册表读写集中在
// channel_process.cc，避免其它模块直接操作静态变量。
HcclResult InsertPluginChannelToMap(ChannelHandle handle, std::shared_ptr<Channel> channelPtr);
HcclResult RemovePluginChannelFromMap(ChannelHandle handle);
void ClearPluginChannelMap();

} // namespace hcomm
#endif // HCOMM_CHANNEL_MAP_REGISTRY_H
