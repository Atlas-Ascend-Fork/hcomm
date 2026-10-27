/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 自定义算子 AICPU 通信域公共初始化参数（coll_communicator_mgr 侧 ABI）。
#ifndef COLL_COMM_AICPU_INIT_PARAM_H
#define COLL_COMM_AICPU_INIT_PARAM_H

#include "hccl_common.h"
#include "hdc_pub.h"
#include "comm_engine/aicpu_comm_def_pub.h"

struct DevAicpuCommConfig {
    bool taskExceptionEnable{true};
    u32 notifyWaitTimeout{1836};
    u64 plfDebugConfig{0};
    u32 taskMonitorInterval{0}; // task执行耗时监控阈值(ms)，0表示关闭
    // 如要新增配置类字段，在此处添加
};

// 自定义算子aicpu通信域公共初始化参数
struct CommAicpuParam {
    char hcomId[HCOMID_MAX_SIZE]{};
    s32 deviceLogicId{0};
    u32 devicePhyId{0};
    u32 deviceType{0};
    u32 userRankSize{0};
    u32 userRank{0};
    hccl::HDCommunicateParams kfcControlTransferH2DParams;
    hccl::HDCommunicateParams kfcStatusTransferD2HParams;
    DevAicpuCommConfig commConfig; // 收编通信域配置类变量
    char udi[HCOMID_MAX_SIZE]{};
};

#endif // COLL_COMM_AICPU_INIT_PARAM_H
