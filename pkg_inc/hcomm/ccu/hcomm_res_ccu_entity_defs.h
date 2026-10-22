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
 * @file hcomm_res_ccu_entity_defs.h
 * @brief CCU instance、resource、channel 跨 SO POD 定义。
 */

#ifndef HCOMM_RES_CCU_ENTITY_DEFS_H
#define HCOMM_RES_CCU_ENTITY_DEFS_H

#include <stdint.h>
#include "hcomm_res_defs.h"
#include "hcomm/hcomm_res_entity_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 跨 SO POD 的 ABI 头部常量。
 * @note 布局变化时必须同步 bump 对应 POD 的 ABI 版本。
 */
static const uint32_t HCOMM_CCU_INSTANCE_ENTITY_ABI_VERSION = 1U;
static const uint32_t HCOMM_CCU_INSTANCE_ENTITY_MAGIC_WORD = 0x43435548U; ///< "CCUH"
static const uint32_t HCOMM_CCU_CHANNEL_ABI_VERSION = 1U;
static const uint32_t HCOMM_CCU_CHANNEL_MAGIC_WORD = 0x43435543U; ///< "CCUC"
static const uint32_t HCOMM_CCU_RES_ABI_VERSION = 1U;
static const uint32_t HCOMM_CCU_DIE_METADATA_MAGIC_WORD = 0x4343444DU; ///< "CCDM"

/**
 * @brief 一段连续 CCU 资源。
 * @note 起始 id + 数量；所属资源类型与 die 由外层 HcommCcuDieResRangePack / HcommCcuDieRes 表达。
 */
typedef struct {
    uint32_t startId;  // 连续资源的起始 ID
    uint32_t count;    // 从 startId 开始的资源数量，必须大于 0
    uint64_t reserved; // 预留字段，当前必须为 0
} HcommCcuDieResRange;

/**
 * @brief 一类资源的全部 range 段。
 * @note 按资源类型枚举值挂在 HcommCcuDieRes.resTypes[type] 上；resRanges 指向 hcomm CcuInstance
 *       持有的数组，asc-comm 在 register_start 期间深拷贝。
 */
typedef struct {
    HcommCcuDieResRange* resRanges; // 该类型的 range 数组（借用）
    uint32_t resourceType;          // 资源池槽位编号（内部约定）
    uint32_t resRangeNum;           // range 数组元素数量
    uint64_t reserved;              // 预留字段，当前必须为 0
} HcommCcuDieResRangePack;

/**
 * @brief 单个 die 的资源元数据。
 * @note 承载回路 channel 与 CCU 资源空间 buffer；ccuResBuffer 按
 *       {REGED_BUFFER_RMA, addr=XN 基址, ProtectionInfo.ub{tokenId, tokenValue}} 装配。
 * @note loopChannelId 为该 die 唯一的环回 channel，硬件上 die 内与 die 间访问共用同一条。
 */
typedef struct {
    CommAbiHeader header;           // ABI 头：version/magic/size 校验
    uint32_t enabled;               // 该 die 是否启用（0/1）
    uint32_t missionKey;            // mission 密钥
    uint32_t loopChannelId;         // 环回 channel id（每 die 一条，die 内/间访问共用）
    RegedBufferEntity ccuResBuffer; // CCU 资源空间（XN 基址 + token）映射后的 buffer 信息
    uint32_t reserved;              // 预留字段，当前必须为 0
} HcommCcuDieMetadata;

/**
 * @brief 单个 die 的全部资源视图。
 * @note 按资源类型索引 range 段数组，并携带 die 元数据指针。
 */
typedef struct {
    uint32_t dieId;                       // 该资源视图所属 die
    uint32_t resTypeNum;                  // 已使用的资源类型数（<= 32）
    HcommCcuDieMetadata* dieMetadata;     // hcomm 持有的 die 元数据（借用）
    HcommCcuDieResRangePack resTypes[32]; // 按资源类型枚举值索引
    uint64_t reserved[16];                // 预留字段，当前必须清零
} HcommCcuDieRes;

/**
 * @brief hcomm 与 asc-comm 间传递的 CCU Instance 完整实体。
 * @note header 用于校验版本、魔数和结构大小；ccuInsKey 由 CcuInstanceMgr 分配，用于跨 SO 查找。
 * @note 本类型仅在 hcomm 与 asc-comm 契约面及 hcomm 内部使用；对外 API 的不透明实例句柄仍为
 *       CcuInsHandle(uint64_t token)。
 */
typedef struct {
    CommAbiHeader header;          // ABI 头：version/magic/size 校验.
    uint32_t ccuDieNum;            // POD 中 die 槽位数（<= 8）
    uint32_t ccuVersion;           // 当前设备的 CCU 版本（V1/V2）
    HcommCcuDieRes ccuDieReses[8]; // 按 die 组织的资源视图（预留 8 槽）
    uint64_t ccuInsKey;            // 不透明实例键
    uint64_t ascCustom[8];         // hcomm 提供的控制操作槽位数组
    uint64_t reserved;             // 预留字段，当前必须清零
} HcommCcuInstanceEntity;

/**
 * @brief CCU Channel 单次查询快照实体（16 槽 + 计数 + 事件掩码 + 远端 buffer 数组）。
 * @note 布局规约：remoteBuffers 紧跟 rmtCcuResBuffer（同为 8 字节对齐类型），避免计数段与指针间
 *       产生中间空洞；不足 8 字节对齐的填充只允许出现在结构体尾部。
 */
typedef struct {
    CommAbiHeader header; // ABI 头：version/magic/size 校验
    CommEngine engine;
    CommProtocol protocol;

    int32_t deviceLogicId; // 所在设备逻辑 id
    uint32_t ccuVersion;   // CCU 版本（V1/V2）
    uint32_t dieId;        // die id
    uint32_t channelId;    // channel id（DFX 记录用）

    RegedBufferEntity rmtCcuResBuffer; // 对端 CCU 资源空间（基址/大小/token 折叠为 buffer 实体）
    RegedBufferEntity* remoteBuffers;  // 远端 buffer 数组借用指针（hcomm 持有，当前数据面不消费）

    uint32_t localVarNum;     // 本端 XN 有效数量（<= 16）
    uint32_t remoteVarNum;    // 远端 XN 有效数量
    uint32_t localEventNum;   // 本端 CKE 有效数量
    uint32_t remoteEventNum;  // 远端 CKE 有效数量
    uint32_t remoteBufferNum; // 远端 buffer 数量（remoteBuffers 数组长度）

    uint32_t localVarIds[16];      // 本端 XN id 槽
    uint32_t remoteVarIds[16];     // 远端 XN id 槽
    uint32_t localEventIds[16];    // 本端 CKE id 槽
    uint32_t remoteEventIds[16];   // 远端 CKE id 槽
    uint32_t localEventMasks[16];  // 本端 CKE 掩码槽（预留，填 0）
    uint32_t remoteEventMasks[16]; // 远端 CKE 掩码槽（预留，填 0）

} HcommCcuChannelEntity;

/**
 * @brief 查询 CCU Channel 实体的回调函数原型。
 *
 * @param channel 输入：通道句柄。
 * @param channelEntity 输出：实体快照。
 *
 * @return 成功返回 0，其他失败。
 */
typedef int32_t (*HcommCcuChannelGetEntityCallBack)(ChannelHandle channel, HcommCcuChannelEntity* channelEntity);

/**
 * @brief 按 die 申请 CCU 指令空间的回调函数原型。
 *
 * @param deviceLogicId 输入：设备逻辑 id。
 * @param dieId 输入：die id。
 * @param instNum 输入：申请的指令条数。
 * @param startInstId 输出：申请到的起始指令 id。
 *
 * @return 成功返回 0，其他失败。
 */
typedef int32_t (*HcommCcuInstSpaceAllocCallBack)(
    int32_t deviceLogicId, uint32_t dieId, uint32_t instNum, uint32_t* startInstId);

/**
 * @brief 下发 CCU 指令的回调函数原型。
 *
 * @param deviceLogicId 输入：设备逻辑 id。
 * @param dieId 输入：die id。
 * @param startInstructionId 输入：起始指令 id。
 * @param data 输入：主机侧指令数据地址。
 * @param instNum 输入：下发的指令条数。
 *
 * @return 成功返回 0，其他失败。
 */
typedef int32_t (*HcommCcuInstSpaceSubmitCallBack)(
    int32_t deviceLogicId, uint32_t dieId, uint32_t startInstructionId, const void* data, uint32_t instNum);

#ifdef __cplusplus
} // extern "C"

#endif // __cplusplus

#endif // HCOMM_RES_CCU_ENTITY_DEFS_H
