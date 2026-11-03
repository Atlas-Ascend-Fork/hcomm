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

    /*
     * 设备任务转发环（锚在请求半区尾部保留区；与 data payload 区不相交）。
     *
     * 背景：北向 HostDPU 下同一设备流的任务由两方并发落库——host 执行器线程
     * （DPU 回调数据面，proto=ROCE）与 device 进程（AICPU 展开，proto=LOCAL/MESH）。
     * 行 id 按物理到达分配，交织序可能 ≠ 设备程序序，checker 按 id 升序当流序
     * 推断先后 → 假环（ErrorCode 102）。真实 hcomm 无此问题：真机任务表仅设备侧
     * 单写者，落库序天然 = 设备程序序。本环在北向重建该单写者不变量：
     * device 的落库改经环转发，host 执行器统一排空落库（每流单写者），
     * 落库序 ≡ 设备程序序。
     *
     * 环布局（请求半区尾部 DPU_DEV_TASK_RING_OFF 起）：
     *   [0, 64)                  环头：tail（u64，ticket 计数器，fetch_add 分配槽位）
     *   [64, 64 + N*200)         槽位数组，槽位 i 布局 = DpuDevTaskRingEntry
     * 生产者为 device 进程当前 RPC 线程（thread_local 注册，一线程一环，同 AICPU
     * 一线程一流不变量）；消费者为 host 执行器线程（每轮先查请求 flag：无请求
     * 排空全部就绪记录；有请求按请求头 DPU_REQ_RING_MARK_OFF 水位精确排空后
     * 再执行回调，回调返回后再补一次排空）。协议：
     *   生产者：fetch_add(tail) 得 ticket → 自旋等槽位 FREE（有界，超时回退直写）
     *           → 填 opDetailId + meta → 屏障 → state=READY(release)
     *   消费者：acquire 读 tail → 按 ticket 序自旋等槽位 READY（有界，下轮重试）
     *           → 拷出 → state=FREE(release) → 按环序落库（id 递增 = 程序序）
     */
    constexpr uint32_t DPU_DEV_TASK_RING_CAP = 4096; /**< 环槽位数（约 800KB，远超单 op 任务峰值 ~750） */
    constexpr uint64_t DPU_DEV_TASK_RING_HDR = 64;   /**< 环头尺寸（tail u64 @0） */
    constexpr uint32_t DPU_DEV_TASK_ENTRY_FREE = 0;  /**< 槽位空闲（消费者已取走） */
    constexpr uint32_t DPU_DEV_TASK_ENTRY_READY = 1; /**< 槽位就绪（生产者已写完） */

    /** 环记录：device 推送时刻的 opDetailId + 任务元数据原样镜像。 */
    struct DpuDevTaskRingEntry {
        uint32_t state;        /**< FREE/READY（生产者写、消费者清，acquire/release 语义） */
        uint32_t opDetailId;   /**< device 进程推送时刻的 sim::operation::g_currOpDetailId */
        HcclTaskMetaData meta; /**< 任务元数据镜像（落库 BLOB 与 device 直写逐字节一致） */
        uint8_t reserved[2];   /**< 填充至 200B（4 字节对齐步长，防非对齐原子访问） */
    };

    constexpr uint64_t DPU_DEV_TASK_RING_ENTRY_BYTES = sizeof(DpuDevTaskRingEntry);
    constexpr uint64_t DPU_DEV_TASK_RING_BYTES
        = DPU_DEV_TASK_RING_HDR + DPU_DEV_TASK_RING_CAP * DPU_DEV_TASK_RING_ENTRY_BYTES;
    constexpr uint64_t DPU_DEV_TASK_RING_OFF = DPU_SHMEM_HALF_SIZE - DPU_DEV_TASK_RING_BYTES;

    /*
     * 环截止水位（u64，位于 payload 区与转发环之间的保留带）：
     * device 在 HcommSendRequest 置 flag 前写入"此刻本通信域环 tail 快照"，
     * host 执行器见到 flag 后只精确排空 ticket < 水位的记录再执行回调——
     * 水位内是本请求之前的设备程序序任务（必须先于回调直写任务落库），
     * 水位外是 SendRequest 返回之后的推送（必须后于回调直写任务落库，
     * 由回调后/响应前的补充排空消费）。初始/陈旧值恒 <= 已消费位点，
     * 按空排水语义兜底，天然兼容不写本字段的其他发送方
     * （0=从未推送；上次请求的水位 <= 已消费位点）。
     */
    constexpr uint64_t DPU_REQ_RING_MARK_OFF = DPU_DEV_TASK_RING_OFF - 16; /**< 环截止水位偏移（8B 对齐） */

    /* 与 device 侧 dpu_rpc_device_stub.cc 的镜像常量由两侧 static_assert 共同守护。 */
    static_assert(sizeof(HcclTaskMetaData) == 190, "HcclTaskMetaData layout drift");
    static_assert(DPU_DEV_TASK_RING_ENTRY_BYTES == 200, "ring entry layout drift");
    static_assert(DPU_DEV_TASK_RING_OFF % 8 == 0, "ring base must be 8B aligned");
    static_assert(
        DPU_REQ_DATA_OFF + DPU_TASK_PAYLOAD_MAX <= DPU_DEV_TASK_RING_OFF,
        "dev task ring overlaps request payload region");
    static_assert(DPU_REQ_RING_MARK_OFF % 8 == 0, "ring mark must be 8B aligned");
    static_assert(
        DPU_REQ_DATA_OFF + DPU_TASK_PAYLOAD_MAX <= DPU_REQ_RING_MARK_OFF, "ring mark overlaps request payload region");

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

    /*==================== 5. 执行器 API（实现于 dpu_executor.cc） ====================*/

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
