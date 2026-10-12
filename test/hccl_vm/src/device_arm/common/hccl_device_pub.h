/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_DEVICE_PUB_H
#define HCCL_DEVICE_PUB_H

#include <cstdint>
#include <string>

#include "sim_common_defs.h"
#include "storage/database_result.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

constexpr uint32_t HCCL_SQE_SIZE = 64;
constexpr uint32_t HCCL_WQE_SIZE = 64;
constexpr uint32_t HCCL_SQE_MAX_CNT = 2048;

HcclSim::HcclVmResult SetCurRankId(uint32_t rankId);

HcclSim::HcclVmResult GetCurRankId(uint32_t* rankId);

void SetCurDeviceKey(uint32_t deviceKey);

uint32_t GetCurDeviceKey();

HcclSim::HcclVmResult GetSqBufferAddr(uint8_t** sqBuff);

HcclSim::HcclVmResult GetPiValByJettyId(uint32_t jettyId, uint32_t* piValue);

HcclSim::HcclVmResult UpdatePiValByJettyId(uint32_t jettyId, uint32_t piValue);

uint64_t TransLocalAddrToVirtual(uint64_t devAddr);

uint64_t TransRemoteAddrToVirtualByDeviceId(uint64_t devAddr, uint32_t deviceId);

uint32_t GetDeviceIdByDevAddr(uint64_t devAddr);

/// SQE 任务构造的地址一次解析结果：同一次按 (映射设备, 映射地址) 定位的
/// VirtualMemBlock 区间查找同时给出虚拟地址与该映射关联的物理属主设备，
/// 替代"先按设备翻译虚拟地址、再按虚拟地址全局反查 VMB→PhyMemBlock"的
/// 两步查询（旧路径每对地址 4 次 VMB 读 + 2 次 PhyMemBlock 读，
/// 新路径 2 次带设备条件 VMB 读 + 2 次 PhyMemBlock 主键读）。
struct TaskAddressResolution {
    uint64_t virtualAddress;   ///< start_ptr + (mappedAddress - dev_mapped_ptr)
    uint32_t physicalDeviceId; ///< 物理属主设备（定位到的 VMB 行 phy_mem_id 对应 PhyMemBlock.device_id）
};

/// 按 (映射设备, 映射地址) 解析任务地址：
/// - mappingDeviceId 为 Device 表主键（本地 GetCurDeviceKey() 或远端 EID 经
///   GetRmtDeviceIdByEid 的解析结果）；不同进程相同数值的映射地址不代表同一
///   映射，匹配身份是 (device_id, mappedAddress)；
/// - 命中 VMB（device_id 等值 + src_type=DEV + dev_mapped_ptr<=addr<dev_mapped_end_ptr）
///   后由同一行计算虚拟地址（校验加法溢出），再按该行 phy_mem_id 主键读取
///   PhyMemBlock 取物理属主——映射设备与物理属主可以不同（aclrtMapMem 合同），
///   不能互相替换，物理属主读取固定保留；
/// - 全部成功才返回 OK；mappingDeviceId 无效(0)、区间未命中、phy_mem_id 为空、
///   物理行不存在、属主超出 uint32 范围或后端错误均返回明确错误，
///   不生成零地址/零属主的"有效"结果；
/// - 结果复用本次读取，不持久缓存、不新增跨调用生命周期。
HcclSim::Storage::DbResult<TaskAddressResolution> ResolveTaskAddress(uint64_t mappedAddress, uint64_t mappingDeviceId);

bool GetDeviceIdByIpAddr(const std::string& ipAddr, uint32_t& deviceId);

uint64_t GetDevMapperAddrByDevAddrImpl(uint64_t devAddr, const char* file, int line);

#define GetDevMapperAddrByDevAddr(devAddr) GetDevMapperAddrByDevAddrImpl(devAddr, __FILE__, __LINE__)

void UpdateKfcStatus(uint64_t d2hAddr);

void RegisterSignalHandler();

uint32_t GetSqTail(uint32_t sqId);

void UpdateSqTail(uint32_t sqId, uint32_t newTail);

bool GetWqebufferByJettyId(uint64_t jettyId, uint64_t& wqeBuffer);

void InitPipeFds(int h2dReadFd, int d2hWriteFd);

int DeviceSendMsg(uint8_t cmd, const void* data, uint32_t dataLen);

int DeviceRecvMsg(uint8_t& outCmd, void* outData, uint32_t maxLen, uint32_t& outDataLen);

void SetLastQuerySqId(uint32_t sqId);

uint32_t GetLastQuerySqId();

void SetDpuStreamId(uint32_t deviceId, uint64_t streamId);

#ifdef __cplusplus
}
#endif

#endif
