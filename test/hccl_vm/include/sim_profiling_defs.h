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
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * for the full text of the License.
 */

#ifndef SIM_PROFILING_DEFS_H
#define SIM_PROFILING_DEFS_H

#include <cstdint>
#include <map>
#include <sstream>
#include <string>

// profiling 任务类型: 与 hcomm TaskParamType(值域 0~22)
enum class ProfTaskParamType : uint8_t {
    TASK_SDMA = 0,                 // profiling op 名: Memcpy
    TASK_RDMA,                     // RDMASend
    TASK_REDUCE_INLINE,            // Reduce_Inline
    TASK_REDUCE_TBE,               // Reduce_TBE
    TASK_NOTIFY_RECORD,            // Notify_Record
    TASK_NOTIFY_WAIT,              // Notify_Wait
    TASK_SEND_NOTIFY,              // Send_Notify
    TASK_SEND_PAYLOAD,             // Send_Payload
    TASK_WRITE_WITH_NOTIFY,        // Write_With_Notify
    TASK_WRITE_REDUCE_WITH_NOTIFY, // Reduce_With_Notify
    TASK_CCU,                      // Ccu
    TASK_AICPU_KERNEL,             // AicpuKernel
    TASK_AICPU_REDUCE,             // Aicpu_Reduce
    TASK_AIV,                      // AivKernel
    TASK_UB_INLINE_WRITE,          // Ub_Inline_Write
    TASK_UB_REDUCE_INLINE,         // Ub_Reduce_Inline
    TASK_UB,                       // Ub_Write_Or_Read
    TASK_DPU_KERNEL,               // DpuKernel
    TASK_DPU_THREAD_FENCE,         // Dpu_ThreadFence
    TASK_DPU_CHANNEL_FENCE,        // Dpu_ChannelFence
    TASK_DPU_INLINE_WRITE,         // Dpu_Notify_Record
    TASK_DPU_NOTIFY_WAIT,          // Dpu_Notify_Wait
    TASK_DPU_WRITE_WITH_NOTIFY,    // Dpu_Write_With_Notify
    UNKNOWN = 0xFF
};

// profiling 链路类型: 与 hcomm DfxLinkType(值域 0~9)
enum class ProfLinkType : uint8_t {
    LINK_ONCHIP = 0,        // 片上/本地
    LINK_HCCS = 1,          // HCCS
    LINK_PCIE = 2,          // PCIe(跨片 SDMA 通道)
    LINK_ROCE = 3,          // RoCE
    LINK_SIO = 4,           // SIO
    LINK_HCCS_SW = 5,       // HCCS_SW
    LINK_STANDARD_ROCE = 6, // 标准RoCE
    LINK_UB = 7,            // UB
    LINK_UBoE = 8,          // UB over Ethernet
    LINK_RESERVED = 9,      // 预留
    UNKNOWN = 0xFF
};

// profiling 传输类型: 与 hcomm DfxTransportType/SimpleTaskType(值域 0~4)
enum class ProfTransportType : uint8_t {
    TRANSPORT_SDMA = 0,  // 跨片 SDMA 通道搬运
    TRANSPORT_RDMA = 1,  // RDMA
    TRANSPORT_LOCAL = 2, // 本地搬运/本地算子
    TRANSPORT_UB = 3,    // UB 传输
    TRANSPORT_ROCE = 4,  // RoCE 传输
    UNKNOWN = 0xFF
};

// 名称与 hcomm PROF_TASK_OP_NAME_V2(profiling_common.h)一致
inline std::string ProfTaskParamTypeName(ProfTaskParamType type)
{
    static const std::map<ProfTaskParamType, std::string> PROF_TASK_NAMES{
        {ProfTaskParamType::TASK_SDMA, "Memcpy"},
        {ProfTaskParamType::TASK_RDMA, "RDMASend"},
        {ProfTaskParamType::TASK_REDUCE_INLINE, "Reduce_Inline"},
        {ProfTaskParamType::TASK_REDUCE_TBE, "Reduce_TBE"},
        {ProfTaskParamType::TASK_NOTIFY_RECORD, "Notify_Record"},
        {ProfTaskParamType::TASK_NOTIFY_WAIT, "Notify_Wait"},
        {ProfTaskParamType::TASK_SEND_NOTIFY, "Send_Notify"},
        {ProfTaskParamType::TASK_SEND_PAYLOAD, "Send_Payload"},
        {ProfTaskParamType::TASK_WRITE_WITH_NOTIFY, "Write_With_Notify"},
        {ProfTaskParamType::TASK_WRITE_REDUCE_WITH_NOTIFY, "Reduce_With_Notify"},
        {ProfTaskParamType::TASK_CCU, "Ccu"},
        {ProfTaskParamType::TASK_AICPU_KERNEL, "AicpuKernel"},
        {ProfTaskParamType::TASK_AICPU_REDUCE, "Aicpu_Reduce"},
        {ProfTaskParamType::TASK_AIV, "AivKernel"},
        {ProfTaskParamType::TASK_UB_INLINE_WRITE, "Ub_Inline_Write"},
        {ProfTaskParamType::TASK_UB_REDUCE_INLINE, "Ub_Reduce_Inline"},
        {ProfTaskParamType::TASK_UB, "Ub_Write_Or_Read"},
        {ProfTaskParamType::TASK_DPU_KERNEL, "DpuKernel"},
        {ProfTaskParamType::TASK_DPU_THREAD_FENCE, "Dpu_ThreadFence"},
        {ProfTaskParamType::TASK_DPU_CHANNEL_FENCE, "Dpu_ChannelFence"},
        {ProfTaskParamType::TASK_DPU_INLINE_WRITE, "Dpu_Notify_Record"},
        {ProfTaskParamType::TASK_DPU_NOTIFY_WAIT, "Dpu_Notify_Wait"},
        {ProfTaskParamType::TASK_DPU_WRITE_WITH_NOTIFY, "Dpu_Write_With_Notify"},
    };
    auto it = PROF_TASK_NAMES.find(type);
    return it != PROF_TASK_NAMES.end() ? it->second : "Unknown";
}

// 名称对应 hcomm DfxLinkType 枚举名
inline std::string ProfLinkTypeName(ProfLinkType type)
{
    static const std::map<ProfLinkType, std::string> PROF_LINK_NAMES{
        {ProfLinkType::LINK_ONCHIP, "ONCHIP"},
        {ProfLinkType::LINK_HCCS, "HCCS"},
        {ProfLinkType::LINK_PCIE, "PCIE"},
        {ProfLinkType::LINK_ROCE, "ROCE"},
        {ProfLinkType::LINK_SIO, "SIO"},
        {ProfLinkType::LINK_HCCS_SW, "HCCS_SW"},
        {ProfLinkType::LINK_STANDARD_ROCE, "STANDARD_ROCE"},
        {ProfLinkType::LINK_UB, "UB"},
        {ProfLinkType::LINK_UBoE, "UBoE"},
        {ProfLinkType::LINK_RESERVED, "RESERVED"},
    };
    auto it = PROF_LINK_NAMES.find(type);
    return it != PROF_LINK_NAMES.end() ? it->second : "Unknown";
}

// 名称对应 hcomm DfxTransportType/SimpleTaskType 枚举名
inline std::string ProfTransportTypeName(ProfTransportType type)
{
    static const std::map<ProfTransportType, std::string> PROF_TRANSPORT_NAMES{
        {ProfTransportType::TRANSPORT_SDMA, "SDMA"},   {ProfTransportType::TRANSPORT_RDMA, "RDMA"},
        {ProfTransportType::TRANSPORT_LOCAL, "LOCAL"}, {ProfTransportType::TRANSPORT_UB, "UB"},
        {ProfTransportType::TRANSPORT_ROCE, "ROCE"},
    };
    auto it = PROF_TRANSPORT_NAMES.find(type);
    return it != PROF_TRANSPORT_NAMES.end() ? it->second : "Unknown";
}

// 任务 profiling 信息: 值域对齐 hcomm TaskParamType/DfxLinkType/DfxTransportType
struct HcclTaskProfInfo {
    ProfTaskParamType profTaskType;  // profiling 类型
    ProfLinkType linkType;           // 物理链路类型
    ProfTransportType transportType; // 数据面传输方式

    HcclTaskProfInfo()
        : profTaskType(ProfTaskParamType::UNKNOWN),
          linkType(ProfLinkType::UNKNOWN),
          transportType(ProfTransportType::UNKNOWN)
    {}

    HcclTaskProfInfo(ProfTaskParamType taskType, ProfLinkType link, ProfTransportType transport)
        : profTaskType(taskType),
          linkType(link),
          transportType(transport)
    {}

    std::string Describe() const
    {
        std::ostringstream oss;
        oss << "profTaskType=" << ProfTaskParamTypeName(profTaskType) << " linkType=" << ProfLinkTypeName(linkType)
            << " transportType=" << ProfTransportTypeName(transportType);
        return oss.str();
    }
};

#endif // SIM_PROFILING_DEFS_H
