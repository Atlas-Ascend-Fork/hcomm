/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_C_ADPT_H
#define HCOMM_C_ADPT_H

#include "hcomm_res.h"
#include "hcomm/hcomm_res_expt.h"
#include "mem_host_pub.h"
#include "hccl_diag.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

const uint32_t HCOMM_RES_TAG_MAX_LEN = 255;
constexpr uint32_t UB_SQ_DEPTH_NOT_SET = 0xFFFFFFFFU;
constexpr uint32_t UB_SCQ_DEPTH_NOT_SET = 0xFFFFFFFFU;

typedef struct {
    int32_t devPhyId;
    uint32_t superPodId;
} HcommDevId;

/**
 * @struct HcommBuf
 * @brief 内存缓冲区描述结构体
 * @var addr   - 虚拟地址指针
 * @var len    - 内存长度（单位字节）
 */
typedef struct {
    void* addr;
    uint64_t len;
} HcommBuf;

typedef CommMem HcommMem;

typedef HcommMemHandle MemHandle;

#ifndef HCOMM_MEM_GRANT_INFO_DEFINED
#define HCOMM_MEM_GRANT_INFO_DEFINED
typedef struct {
    uint32_t sdid;
    int32_t pid;
} HcommMemGrantInfo;
#endif

struct CommMemInfo {
    CommMem mem{};
    void* bufferHandle{nullptr};
    char memTag[HCOMM_RES_TAG_MAX_LEN] = {0};
};

/**
 * @brief 通信设备Endpoint监听配置结构体
 */
#ifndef HCOMM_ENDPOINT_LISTEN_CONFIG_DEFINED
#define HCOMM_ENDPOINT_LISTEN_CONFIG_DEFINED
typedef struct {
    union {
        uint8_t raws[24]; ///< 通用数据区，用于未来扩展，如backlog, timeout等
        struct {};
    };
} HcommEndpointListenConfig;
#endif

HcommResult HcommResMgrInit(uint32_t devPhyId = UINT32_MAX);

HcommResult HcommEndpointGet(EndpointHandle endpointHandle, void** endpoint);

/**
 * @brief 启动通信设备Endpoint监听
 * @param[in] endpointHandle Endpoint句柄
 * @param[in] port 监听端口号
 * @param[in] config 监听配置参数（可为NULL，使用默认配置）
 * @return HcommResult 执行结果状态码
 * @note 启动指定Endpoint在指定端口上的监听服务
 */
extern HcommResult
HcommEndpointStartListen(EndpointHandle endpointHandle, uint32_t port, const HcommEndpointListenConfig* config);

/**
 * @brief 停止通信设备Endpoint监听
 * @param[in] endpointHandle Endpoint句柄
 * @param[in] port 监听端口号
 * @return HcommResult 执行结果状态码
 * @note 停止指定Endpoint在指定端口上的监听服务
 */
extern HcommResult HcommEndpointStopListen(EndpointHandle endpointHandle, uint32_t port);

extern HcommResult HcommChannelGetNotifyNum(ChannelHandle channelHandle, uint32_t* notifyNum);

/**
 * @brief 确保 AICPU 通道 device 实体完成初始化（兜底补发 init kernel）
 * @param[in] channelList host 侧通道句柄列表
 * @param[in] listNum 句柄数量
 * @return HcommResult 执行结果状态码
 * @note 不查链路状态：未就绪通道补发 init kernel，已就绪跳过；仅支持 AICPU/AICPU_TS 引擎
 */
HcommResult HcommChannelEnsureDeviceReady(const ChannelHandle* channelList, uint32_t listNum);

extern HcommResult HcommMemGrant(EndpointHandle endpointHandle, const HcommMemGrantInfo* remoteGrantInfo);

HcommResult HcommChannelGet(ChannelHandle channelHandle, void** channel);

HcommResult HcommThreadAcquireByNotify(rtStream_t stream, void** notifys, uint32_t notifyNum, ThreadHandle* thread);
HcommResult HcommThreadAllocWithConfig(
    CommEngine engine, uint32_t threadNum, ThreadType type, const ThreadConfig* config, ThreadHandle* threads);

HcommResult HcommThreadAllocWithCommConfig(
    CommEngine engine, const char* commId, uint32_t threadNum, ThreadType type, const ThreadConfig* config,
    ThreadHandle* threads);

HcommResult HcommEngineCtxCreate(CommEngine engine, uint64_t size, void** ctx);

HcommResult HcommEngineCtxDestroy(CommEngine engine, void* ctx);

HcommResult HcommEngineCtxCopy(CommEngine engine, void* dstCtx, const void* srcCtx, uint64_t size);

HcommResult HcommDfxKernelLaunch(const std::string& commTag, aclrtBinHandle binHandle, HcclDfxOpInfo dfxOpInfo);

HcommResult HcommMemGetAllMemHandles(EndpointHandle endpointHandle, void** memHandles, uint32_t* memHandleNum);

HcommResult HcommChannelUpdateMemInfo(HcommMemHandle* memHandles, uint32_t memHandleNum, ChannelHandle channelHandle);

// 取通道待下发的 device 侧地址条目：hostPack 为空只回所需字节数，
// 否则按 *packSize 传入的容量拷贝并回实际字节数；未就绪 / 无条目回 0 且成功
HcommResult HcommChannelGetMemUpdatePack(ChannelHandle channelHandle, void* hostPack, uint32_t* packSize);

HcommResult CheckUbAttr(HcommChannelDesc& channelDesc, CommEngine engine);

HcommResult CheckUbMemAttr(HcommChannelDesc& channelDesc);

/// 校验 HcommChannelDesc.qos：未配置(0xFFFFFFFF) 或 [HCCL_QOS_MIN, HCCL_QOS_MAX]；与 ApplyHcclQos 对齐
HcommResult CheckChannelDescQos(const HcommChannelDesc& channelDesc);

HcommResult CheckRoceAttr(HcommChannelDesc& channelDesc, EndpointLocType localLocType);
#ifdef __cplusplus
}
#endif // __cplusplus

#endif // HCOMM_C_ADPT_H
