/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_RES_H
#define HCOMM_RES_H

#include "hcomm_res_defs.h"
#include "hcomm_channel.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

extern HcommResult HcommEndpointCreate(const EndpointDesc* endpoint, EndpointHandle* endpointHandle);

extern HcommResult HcommEndpointDestroy(EndpointHandle endpointHandle);

/**
 * @brief 查询指定设备上的Endpoint描述符数量
 * @param[in] userDevId NPU用户设备ID
 * @param[out] descNum 返回的Endpoint描述符数量
 * @return HcommResult 执行结果状态码
 * @note 典型用法是先调用本接口获取数量，再按该数量申请数组并调用HcommEndpointGetDescs。
 * @note 仅 950 及后续代际设备支持，其余代际返回 HCCL_E_NOT_SUPPORT。
 */
extern HcommResult HcommEndpointGetDescNum(int32_t userDevId, uint32_t* descNum);

/**
 * @brief 获取指定设备上的Endpoint描述符
 * @param[in] userDevId NPU用户设备ID
 * @param[in,out] descNum 入参为endpointDescs数组容量，出参为实际填充的描述符数量
 * @param[out] endpointDescs 调用方申请的Endpoint描述符数组
 * @return HcommResult 执行结果状态码；数组容量小于实际Endpoint数量时返回参数错误
 * @note 推荐按HcommEndpointGetDescNum返回的数量申请数组，并将数组实际容量通过descNum传入。
 * @note 仅 950 及后续代际设备支持，其余代际返回 HCCL_E_NOT_SUPPORT。
 */
extern HcommResult HcommEndpointGetDescs(int32_t userDevId, uint32_t* descNum, EndpointDesc* endpointDescs);

extern HcommResult
HcommMemReg(EndpointHandle endpointHandle, const char* memTag, const CommMem* mem, HcommMemHandle* memHandle);

extern HcommResult HcommMemUnreg(EndpointHandle endpointHandle, HcommMemHandle memHandle);

extern HcommResult
HcommMemExport(EndpointHandle endpointHandle, HcommMemHandle memHandle, void** memDesc, uint32_t* memDescLen);

extern HcommResult
HcommMemImport(EndpointHandle endpointHandle, const void* memDesc, uint32_t descLen, CommMem* outMem);

extern HcommResult HcommMemUnimport(EndpointHandle endpointHandle, const void* memDesc, uint32_t descLen);

extern HcommResult
HcommThreadAlloc(CommEngine engine, uint32_t threadNum, const uint32_t* notifyNumPerThread, ThreadHandle* threads);

extern HcommResult HcommThreadFree(const ThreadHandle* threads, uint32_t threadNum);

extern HcommResult HcommEndpointGetListenPort(EndpointHandle endpointHandle, uint32_t* port);

extern HcommResult HcommThreadResGetInfo(ThreadHandle thread, ThreadResType resType, uint32_t infoLen, void** info);

extern HcommResult
HcommEndpointCheckFeature(HcommEndpointFeatureType featureType, const EndpointDesc* endpointDesc, bool* value);

extern HcommResult HcommMemAlloc(void** ptr, size_t size);

extern HcommResult HcommMemFree(void* ptr);

/**
 * @brief 将Host DDR内存映射为Device侧可访问的dVA
 * @param[in] hostMem Host侧内存描述，type须为COMM_MEM_TYPE_HOST，addr为Host hVA
 * @param[out] devMem 返回映射后的Device侧内存描述，addr为dVA，type为COMM_MEM_TYPE_DEVICE
 * @param[in] mapType Host内存映射方式，参见HcommMemMapType；
 * HCOMM_MEM_MAP_PINNED仅非ATU（ACL）路径支持，ATU场景返回HCCL_E_NOT_SUPPORT
 * @return HcommResult 执行结果状态码
 * @note PCIe H2D协议且芯片为950/960时使用PCIe ATU方式注册；
 * 其余场景使用aclrtHostRegisterV2注册，并通过aclrtHostGetDevicePointer获取dVA
 * @note 调用前需通过aclrtSetDevice设置当前Device；hostMem->addr需4K页对齐
 * @note 映射得到的dVA供Device侧访问Host内存使用，不能用于内存拷贝操作
 * @note 映射得到的dVA仅在完成映射的进程/Device上下文内有效，不能跨进程或跨Device传递
 * @note Host内存的生命周期由调用方保证，需与HcommMemHostUnmapFromDev在同一Device上下文中成对调用
 */
extern HcommResult HcommMemHostMap2Dev(const CommMem* hostMem, CommMem* devMem, HcommMemMapType mapType);

/**
 * @brief 解除Host DDR内存到Device侧dVA的映射
 * @param[in] hostMem Host侧内存描述，须与HcommMemHostMap2Dev的入参一致，type为COMM_MEM_TYPE_HOST，addr为Host hVA
 * @return HcommResult 执行结果状态码
 * @note 需与HcommMemHostMap2Dev成对调用，传入映射时的hostMem
 * @note 需与HcommMemHostMap2Dev在同一Device上下文中调用
 * @note 注销成功后，映射得到的dVA失效，不能再被Device侧访问
 */
extern HcommResult HcommMemHostUnmapFromDev(const CommMem* hostMem);

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // HCOMM_RES_H
