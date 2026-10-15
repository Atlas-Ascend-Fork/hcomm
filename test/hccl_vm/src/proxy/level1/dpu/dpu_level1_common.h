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
 * Description: HostDPU 北向劫持（level1）公共设施头文件。
 *              提供 DPU RPC 共享内存协议常量与编解码、执行器回调期 TLS 上下文、
 *              任务 protocol 标注以及回调函数指针类型，供 dpu 子模块各文件共用。
 *              协议布局与真实实现 aicpu_ts_sync_data_c_adpt.cc / task_service.cc 同构。
 * Create: 2026-10-09
 */

#ifndef DPU_LEVEL1_COMMON_H
#define DPU_LEVEL1_COMMON_H

#include <cstdint>
#include <cstring>

#include "sim_common_defs.h"

namespace HcclSim {
namespace DpuSim {

    /*==================== 1. DPU RPC 共享内存协议常量 ====================*/

    /*
     * NPU -> DPU 请求半区布局（与 hcomm aicpu_ts_sync_data_c_adpt.cc:71-121 一致）：
     *   +--------+---------------+---------+--------------+----------+------+
     *   | flag   | msgTag[256]   | msgId   | timeout      | dataSize | data |
     *   | (1B)   | (256B)        | (4B)    | (4B)         | (8B)     | ...  |
     *   +--------+---------------+---------+--------------+----------+------+
     *   偏移:  0      1              257       261            265        273
     *
     * DPU -> NPU 响应半区布局（本桩扩展，与真实 SynchronizeControlInfo 兼容）：
     *   +--------+---------------+---------+----------+
     *   | flag   | msgId         | retCode |          |
     *   | (1B)   | (4B)          | (4B)    |          |
     *   +--------+---------------+---------+----------+
     *   偏移:  0      257          261
     *   真实实现将请求头 flag|msgTag|msgId 共 261B 原样拷贝到响应半区再置 flag，
     *   本桩在响应半区 offset=261（原 timeout 槽位）追加 retCode 用于快速失败。
     */
    constexpr uint64_t DPU_REQ_FLAG_OFF = 0;       /**< 请求/响应半区 flag 偏移（1 字节） */
    constexpr uint64_t DPU_REQ_MSGTAG_OFF = 1;     /**< 请求半区 msgTag 偏移（256 字节定长） */
    constexpr uint64_t DPU_REQ_MSGID_OFF = 257;    /**< 请求/响应半区 msgId 偏移（4 字节） */
    constexpr uint64_t DPU_REQ_TIMEOUT_OFF = 261;  /**< 请求半区 timeout 偏移（4 字节） */
    constexpr uint64_t DPU_REQ_DATASIZE_OFF = 265; /**< 请求半区 dataSize 偏移（8 字节） */
    constexpr uint64_t DPU_REQ_DATA_OFF = 273;     /**< 请求半区 data payload 偏移 */
    constexpr uint64_t DPU_RSP_RETCODE_OFF = 261;  /**< 响应半区 retCode 偏移（复用 timeout 槽位） */

    constexpr uint32_t DPU_MSG_TAG_MAX_LEN = 256;                /**< msgTag 定长空间（同 MSG_TAG_SIZE_BYTE） */
    constexpr uint64_t DPU_SHMEM_ALLOC_SIZE = 2UL * 1024 * 1024; /**< DPUTAG 实际分配尺寸（2MB） */
    constexpr uint64_t DPU_SHMEM_HALF_SIZE = DPU_SHMEM_ALLOC_SIZE / 2; /**< 请求/响应半区各 1MB */
    constexpr uint64_t DPU_TASK_PAYLOAD_MAX = 64UL * 1024;             /**< data payload 上限（64KB） */
    constexpr uint32_t DPU_EXEC_IDLE_SLEEP_US = 100;                   /**< 执行器空转轮询间隔（微秒） */

    /* 协议布局自检：半区容量必须容纳最大 payload。 */
    static_assert(
        DPU_REQ_DATA_OFF + DPU_TASK_PAYLOAD_MAX <= DPU_SHMEM_HALF_SIZE,
        "dpu shmem half region too small for max payload");

    /** flag 取值（与真实 TaskService 同语义）。 */
    constexpr uint8_t DPU_FLAG_UNSET = 0; /**< 半区空闲（写方已消费/可写入） */
    constexpr uint8_t DPU_FLAG_SET = 1;   /**< 半区有完整消息待消费 */

    /*==================== 2. DPU 任务 protocol 标注 ====================*/

    /** DPU 数据面任务 protocol 值（与南向 ibv_dpu_stub.cc 的 DPU_PROTOCOL_ROCE 一致）。 */
    constexpr uint8_t DPU_TASK_PROTOCOL_ROCE = 2;

    /**
     * @brief 判断当前线程是否处于 DPU 执行器回调上下文。
     * @return true 表示当前任务由 DPU 数据面产生（thread=0 的调用路径）。
     */
    bool IsDpuContext();

    /**
     * @brief DPU 任务上下文（执行器回调期间由执行器置位，回调结束复位）。
     *
     * DPU 数据面调用均为 thread=0，无法经 HcclThread 解析任务归属，
     * 执行器在分发回调前把该请求的通信域/流信息置入本 TLS，
     * 数据面桩与任务插入路径据此回退填充 commId/streamId/rankId。
     */
    struct DpuExecContext {
        bool active{false};      /**< 是否处于 DPU 回调执行期 */
        uint64_t commId{0};      /**< 当前请求的通信域 id（Communicator 表主键） */
        uint32_t rankId{0};      /**< 本 rank id（取自 Communicator.rank_id） */
        uint32_t deviceId{0};    /**< 本设备 id（Device 表主键） */
        uint64_t dpuStreamId{0}; /**< DPU 任务归属流（DpuDeviceInfo.stream_id） */
    };

    /**
     * @brief 获取本线程的 DPU 执行上下文。
     * @return 线程局部 DpuExecContext 引用；非 DPU 线程时 active=false。
     */
    DpuExecContext& GetDpuExecTls();

    /**
     * @brief 给 DPU 上下文产生的任务打 protocol 标注。
     * @param task 输入输出：待标注的任务元数据；非 DPU 上下文或不支持 protocol 的类型不修改。
     */
    void ApplyDpuTaskProtocol(HcclTaskMetaData& task);

    /**
     * @brief 翻译 DPU 任务的端点地址（设备映射空间 → VM 虚拟地址空间）。
     * @param task 输入输出：任务元数据；仅处理 MEM_CPY / REDUCE，
     *             src/dst 按其（Normalize 后的）属主 deviceId 分别翻译。
     * @note DPU 数据面地址来自设备侧序列化的 DPURunInfo（dev_mapped 空间），
     *       与南向 ibv_dpu_stub 的翻译路径同源；仅 DPU 上下文调用。
     */
    void TranslateDpuTaskAddress(HcclTaskMetaData& task);

    /*==================== 3. 回调类型 ====================*/

    /**
     * @brief DPU 任务回调函数指针类型（与 hcomm Callback 同型）。
     *
     * hccl 经 HcclTaskRegister(comm, msgTag, HcclLaunchDPUKernel) 注册，
     * 签名为 int32_t(uint64_t argsAddr, int32_t argsSize) 的函数指针，
     * 纯 C ABI，无 std::function/std::string 跨边界问题。
     */
    using DpuCallback = int32_t (*)(uint64_t, int32_t);

    /*==================== 5. DPU notify 唯一标识（事件级配对） ====================*/

    /**
     * @brief DPU notify 唯一 id 的编码标记（bit63，与其他 id 空间隔离）。
     *
     * 现有 id 空间：MakeNotifyTaskId（rank<<17|行id，≤2^27）、南向唯一 id（bit31）。
     * DPU 采用 bit63 标记 + (槽位行id << 32) | 事件序号 的布局，保证全空间唯一。
     */
    constexpr uint64_t DPU_NOTIFY_ID_TAG = 0x8000'0000'0000'0000ULL;
    constexpr uint32_t DPU_NOTIFY_SEQ_MAX = 0xFFFFFFFFU;

    /**
     * @brief 为 DPU 场景编码一次 RECORD/WAIT 事件的唯一 notify id。
     * @param slotNotifyId 输入：通道槽位对应的 Notify 表行 id（跨 rank 一致的资源标识）。
     * @param seq          输入：该槽位上的事件序号（由 NextDpuNotifySeq 分配）。
     * @return 事件级唯一 notify id。
     * @note 真实 RoCE 的多次连续 record（如 NHR 多 slice）是 FIFO 队列语义，
     *       用事件级唯一 id 还原 1:1 配对，避免槽位复用被误判为多打一冲突。
     */
    uint64_t EncodeDpuNotifyTaskId(uint64_t slotNotifyId, uint32_t seq);

    /**
     * @brief 分配指定槽位的下一个事件序号（进程内单调递增，线程安全）。
     * @param slotNotifyId 输入：通道槽位对应的 Notify 表行 id。
     * @return 从 0 开始递增的事件序号。
     * @note RECORD 侧与 WAIT 侧各自进程独立计数；模板中同槽位的 record/wait
     *       顺序与数量一一对应，序号天然对齐，无需跨进程协调。
     */
    uint32_t NextDpuNotifySeq(uint64_t slotNotifyId);

    /*==================== 6. 执行器 API（实现于 dpu_executor.cc） ====================*/

    /**
     * @brief 幂等拉起 DPU 执行器线程（进程级单例）。
     * @param deviceId 输入：本进程设备 id（Device 表主键，用于绑定设备上下文）。
     * @return 已运行或拉起成功返回 true；失败返回 false。
     */
    bool EnsureDpuExecutorStarted(uint32_t deviceId);

    /**
     * @brief 按通信域清理 DPU 资源（共享内存 + 回调注册）。
     * @param commId 输入：通信域 id（Communicator 表主键）。
     * @note 由 level0 DestroyLevel1Communicator 在通信域销毁时调用。
     */
    void DpuCleanupByCommId(uint64_t commId);

} // namespace DpuSim
} // namespace HcclSim

#endif // DPU_LEVEL1_COMMON_H
