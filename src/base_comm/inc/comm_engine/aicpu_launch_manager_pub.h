/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_AICPU_LAUNCH_MANAGER_PUB_H
#define HCOMM_AICPU_LAUNCH_MANAGER_PUB_H

#include "hccl_common.h"
#include "stream_pub.h"
#include "aicpu_operator_pub.h"
#include "hccl/hccl_res.h"
#include "local_notify.h"
#include "comm_engine/aicpu_comm_def_pub.h"

constexpr uint32_t THREAD_UNIQUE_ID_MAX_SIZE = 10000;
constexpr uint32_t NOTIFY_UNIQUE_ID_MAX_SIZE = THREAD_UNIQUE_ID_MAX_SIZE * HCOMM_THREAD_NOTIFY_MAX_NUM;
constexpr uint32_t NOTIFY_DEVICE_ID_MAX_SIZE = 21 * HCOMM_THREAD_NOTIFY_MAX_NUM;
constexpr uint32_t NAME_SIZE = 64;
struct DevAicpuThreadConfig {
    // 如要新增配置类字段，在此处添加
};

struct ThreadMgrAicpuParam {
    u32 threadNum;
    char hcomId[HCOMID_MAX_SIZE];
    char threadParam[HCOMM_THREADNUM_MAX_NUM][THREAD_UNIQUE_ID_MAX_SIZE]; // 含序列化后thread信息，约40KB
    void* deviceHandle;
    s32 deviceLogicId{-1};                   // 基础通信使用
    u32 deviceType{0};                       // 基础通信使用
    DevAicpuThreadConfig threadConfig;       // 收编thread配置类变量
    CommEngine engine{COMM_ENGINE_RESERVED}; // 通信引擎类型，由Host侧Thread对象携带
    // order控制流导出标记（按线程下标，与threadParam对齐）：device侧InitThreads据此把重建对象
    // 分流进进程级Registry（生命周期与通信域解耦）。
    uint8_t orderCtrlFlag[HCOMM_THREADNUM_MAX_NUM] = {};
};

struct DevAicpuNotifyConfig {
    // 如要新增配置类字段，在此处添加
};

struct NotifyMgrAicpuParam {
    u32 notifyNum;
    char hcomId[HCOMID_MAX_SIZE];
    char notifyParam[NOTIFY_UNIQUE_ID_MAX_SIZE]; // 含序列化后notify信息
    void* deviceHandle;
    bool freeFlag;
    DevAicpuNotifyConfig notifyConfig; // 收编notify配置类变量
};

namespace hccl {

// notify handle 语义类型：与 coll_comm 侧同名同义，避免跨边界退化为裸 uint64_t
typedef uint64_t NotifyHandle;

struct ApiParamDef {
    uint64_t commContext{};
    char kernelName[NAME_SIZE] = {};
    char soName[NAME_SIZE] = {};
    char opName[NAME_SIZE] = {};

    ApiParamDef(const char* kName, const char* sName, const char* oName)
    {
        auto ret = strncpy_s(kernelName, NAME_SIZE, kName, NAME_SIZE - 1);
        if (ret != EOK) {
            HCCL_ERROR("[ApiParamDef][Check]strncpy_s kernelName failed, ret[%d]", ret);
            kernelName[0] = '\0';
        }
        ret = strncpy_s(soName, NAME_SIZE, sName, NAME_SIZE - 1);
        if (ret != EOK) {
            HCCL_ERROR("[ApiParamDef][Check]strncpy_s soName failed, ret[%d]", ret);
            soName[0] = '\0';
        }
        ret = strncpy_s(opName, NAME_SIZE, oName, NAME_SIZE - 1);
        if (ret != EOK) {
            HCCL_ERROR("[ApiParamDef][Check]strncpy_s opName failed, ret[%d]", ret);
            opName[0] = '\0';
        }
    }
};

class AicpuLaunchMgr {
public:
    AicpuLaunchMgr() = default;
    ~AicpuLaunchMgr() = default;
    template <typename OpParam, typename ApiParam>
    static HcclResult KernelLaunch(OpParam& opParam, ApiParam& apiParam, rtStream_t aicpuInitStream);
    static HcclResult NotifyKernelLaunchAlloc(
        std::vector<std::unique_ptr<LocalNotify>>& newNotifys, const std::string& commId,
        std::unique_ptr<NotifyHandle[]>& hostHandle, aclrtBinHandle binCustomHandle);
    static HcclResult NotifyKernelLaunchFree(
        std::vector<NotifyHandle>& aicpuNotifys, uint32_t notifyNum, const std::string& commId,
        aclrtBinHandle binCustomHandle);
};
} // namespace hccl
#endif
