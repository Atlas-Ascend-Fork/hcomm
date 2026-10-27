/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CHANNEL_C_ADPT_PUB_H
#define HCOMM_CHANNEL_C_ADPT_PUB_H

#include <stdint.h>
#include "hcomm_res_defs.h"
#include "hcomm_channel.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * @brief Get the notify number of a channel.
 * @param[in] channelHandle channel handle
 * @param[out] notifyNum output notify number
 * @return HcommResult HCOMM_SUCCESS on success, error code otherwise
 */
extern HcommResult HcommChannelGetNotifyNum(ChannelHandle channelHandle, uint32_t* notifyNum);
/**
 * @brief Build the device-side channel entity for AIV direct channels.
 * @param[in] channelHandle channel handle
 * @param[out] devChannelEntity output device channel entity address
 * @return HcommResult HCOMM_SUCCESS on success, error code otherwise
 */
extern HcommResult HcommChannelBuildDeviceEntity(ChannelHandle channelHandle, uint64_t* devChannelEntity);
/**
 * @brief Ensure the AICPU channel device entity is initialized (re-issue init kernel if needed).
 * @param[in] channelList host-side channel handle list
 * @param[in] listNum handle count
 * @return HcommResult HCOMM_SUCCESS on success, error code otherwise
 */
extern HcommResult HcommChannelEnsureDeviceReady(const ChannelHandle* channelList, uint32_t listNum);
/**
 * @brief Get the serialized memory-update pack of a channel.
 * @param[in] channelHandle channel handle
 * @param[out] hostPack output pack buffer (nullptr to query size only)
 * @param[out] packSize output pack size in bytes
 * @return HcommResult HCOMM_SUCCESS on success, error code otherwise
 */
extern HcommResult HcommChannelGetMemUpdatePack(ChannelHandle channelHandle, void* hostPack, uint32_t* packSize);

/**
 * @brief Get the remote memory handles bound to a channel.
 * @param[in] channelHandle channel handle
 * @param[out] memNum output remote memory count
 * @param[out] remoteMem output remote memory array
 * @param[out] memInfos output per-memory tag string array
 * @return HcommResult HCOMM_SUCCESS on success, error code otherwise
 * @note 出参指向 Channel 内部缓存，随通道生命周期有效，调用方不得释放。
 */
HcommResult
HcommChannelGetRemoteMems(ChannelHandle channelHandle, uint32_t* memNum, CommMem** remoteMem, char*** memInfos);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // HCOMM_CHANNEL_C_ADPT_PUB_H
