/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DPU_SYNC_LAYOUT_H
#define DPU_SYNC_LAYOUT_H

#include <cstddef>
#include <cstdint>

/**
 * @note SQ/CQ 共享内存任务队列的三方共用布局契约：host 侧、DPU kernel 侧、device 侧 CQ 守护器
 *       都 include 本头，故全部为定长 POD + static_assert，不引用任何 aclrt 或 Hrt 接口，
 *       也不 include 其它模块的头（半区尺寸自带一支，交叉断言在 host 侧做）。
 *       本头描述 DPUTAG 内部怎么切分，只装 SQ/CQ 队列；DpuSyncLayout 是内存中的名片、不在共享内存上。
 */
namespace hccl {

// ─────────────────────────────────────────────────────────────
// 1. 尺寸与容量
// ─────────────────────────────────────────────────────────────
constexpr uint32_t DPU_SQE_SIZE = 64U;                      // == HCCL_SQE_SIZE（stream_pub.h），一格一宽
constexpr uint32_t DPU_CQE_SIZE = 64U;                      // 与 SQE 同宽 → SQ/CQ 可复用同一个环形实现
constexpr uint32_t DPU_SQ_DEPTH = 1024U;                    // SQ 环深：一个通信域最多 32 个 DPU 线程
constexpr uint32_t DPU_CQ_DEPTH = 1U;                       // CQ 单槽：只在 SQE 执行异常时写
constexpr uint32_t DPU_SQ_USABLE_DEPTH = DPU_SQ_DEPTH - 1U; // 满/空判据同形，牺牲一格
constexpr uint32_t DPU_SQ_ELEM_BYTES = DPU_SQ_DEPTH * DPU_SQE_SIZE;
constexpr uint32_t DPU_CQ_ELEM_BYTES = DPU_CQ_DEPTH * DPU_CQE_SIZE;

// ─────────────────────────────────────────────────────────────
// 2. SQ 元素：64 字节自描述
// ─────────────────────────────────────────────────────────────
typedef enum : uint16_t {
    DPU_OP_INVALID,
    /* 控制 */
    DPU_OP_CONTROL_TERMINATE,
    /* channel 级读写 */
    DPU_OP_CHANNEL_READ,
    DPU_OP_CHANNEL_READ_NBI,
    DPU_OP_CHANNEL_READ_REDUCE,
    DPU_OP_CHANNEL_WRITE,
    DPU_OP_CHANNEL_WRITE_NBI,
    DPU_OP_CHANNEL_WRITE_WITH_NOTIFY,
    DPU_OP_CHANNEL_WRITE_WITH_NOTIFY_NBI,
    DPU_OP_CHANNEL_WRITE_REDUCE,
    DPU_OP_CHANNEL_WRITE_REDUCE_WITH_NOTIFY,
    /* channel 级同步 */
    DPU_OP_CHANNEL_NOTIFY_RECORD,
    DPU_OP_CHANNEL_NOTIFY_WAIT,
    DPU_OP_CHANNEL_NOTIFY_WAIT_DEFAULT,
    DPU_OP_CHANNEL_FENCE,
    DPU_OP_CHANNEL_DRAIN,
    /* thread 级同步 */
    DPU_OP_THREAD_NOTIFY_RECORD_TO_DPU,
    DPU_OP_THREAD_NOTIFY_RECORD_TO_NPU,
    DPU_OP_THREAD_NOTIFY_WAIT,
    /* 通信域级同步 */
    DPU_OP_COMM_FENCE,
    /* 本端内存操作 */
    DPU_OP_LOCAL_COPY,
    DPU_OP_LOCAL_REDUCE,
    /* rt 级 notify */
    DPU_OP_ACLRT_NOTIFY_RECORD,
    DPU_OP_ACLRT_NOTIFY_WAIT,
} DpuOpKind;

constexpr uint64_t DPU_CHANNEL_NONE = 0U;       // 句柄为 0 = 本操作不用 channel
constexpr uint32_t DPU_TASK_NONE = 0xFFFFFFFFU; // 只出现在 CQE 侧（此条不属于任何任务）

// SQE payload 各变体（40B union 内互斥解释）；提为命名空间级类型，供 CpuTaskPipe 等按类型名引用
struct DpuChannelReadInfo {
    uint64_t channelHandle;
    uint64_t dstAddr, srcAddr, len;
};
struct DpuChannelWriteInfo {
    uint64_t channelHandle;
    uint64_t dstAddr, srcAddr, len;
};
struct DpuChannelWriteWithNotifyInfo {
    uint64_t channelHandle;
    uint64_t dstAddr, srcAddr, len;
    uint32_t notifyIdx;
};
struct DpuChannelWriteReduceInfo {
    uint64_t channelHandle;
    uint64_t dstAddr, srcAddr, count;
    uint8_t dataType, reduceOp;
};
struct DpuChannelWriteReduceWithNotifyInfo {
    uint64_t channelHandle;
    uint64_t dstAddr, srcAddr, count;
    uint8_t dataType, reduceOp;
    uint32_t notifyIdx;
};
struct DpuChannelNotifyRecordInfo {
    uint64_t channelHandle;
    uint32_t notifyIdx;
};
struct DpuChannelNotifyWaitInfo {
    uint64_t channelHandle;
    uint32_t notifyIdx, timeOut;
};
struct DpuChannelDrainOrFenceInfo {
    uint64_t channelHandle;
};
struct DpuThreadNotifyRecordToDpuInfo {
    // host 视角绝对地址（执行段在 host 侧直写）：烧地址免执行侧查表，勿改成句柄+idx 解析
    uint64_t addr;
};
struct DpuThreadNotifyRecordToNpuInfo {
    uint32_t notifyId;
};
struct DpuThreadNotifyWaitInfo {
    uint32_t notifyIdx, timeOut;
};
struct DpuLocalCopyInfo {
    uint64_t dst, src, len;
};
struct DpuAclrtInfo {
    uint64_t notifyId;
    uint32_t timeout;
};
struct DpuOpDescInfo {
    uint64_t descAddr;
    uint32_t descNum, descKind;
};

struct DpuSqe {      // 64 字节定长
    uint16_t opKind; // +0  判别式，见 DpuOpKind；0 = 空元素
    uint32_t taskId; // +4  与 CQE 配对认领
    // 8B 识别前缀 + 本字段 16B = 24B 头部，恰好接上 +24 的 40B union 凑足 64B；
    uint64_t reserved[2]; // +8..+23
    union { // +24, 40 字节；raws 定宽，各变体只声明自己用到的字段，余量由 raws 覆盖
        uint8_t raws[40];
        DpuChannelReadInfo channelRead;
        DpuChannelWriteInfo channelWrite;
        DpuChannelWriteWithNotifyInfo channelWriteWithNotify;
        DpuChannelWriteReduceInfo channelWriteReduce;
        DpuChannelWriteReduceWithNotifyInfo channelWriteReduceWithNotify;
        DpuChannelNotifyRecordInfo channelNotifyRecord;
        DpuChannelNotifyWaitInfo channelNotifyWait;
        DpuChannelDrainOrFenceInfo channelDrainOrFence;
        DpuThreadNotifyRecordToDpuInfo threadNotifyRecordToDpu;
        DpuThreadNotifyRecordToNpuInfo threadNotifyRecordToNpu;
        DpuThreadNotifyWaitInfo threadNotifyWait;
        DpuLocalCopyInfo localCopy;
        DpuAclrtInfo aclrt;
        DpuOpDescInfo opDesc;
    } u;
};
static_assert(sizeof(DpuSqe) == DPU_SQE_SIZE, "DpuSqe must be 64B");
static_assert(offsetof(DpuSqe, u) == 24, "DpuSqe union must be 8B-aligned");

// ─────────────────────────────────────────────────────────────
// 3. CQ 元素：64 字节，与 SQE 同宽
// ─────────────────────────────────────────────────────────────
// CQE 只装「这次失败」的定位信息：错误码 + taskId/streamId；当前算子序号是持久标记，放 DpuCqMeta 不进回执。
struct DpuCqe {          // 64 字节定长
    int32_t hcclErrCode; // +0  任务执行错误码
    uint32_t taskId;     // +4  报错定位键：镜像队列里按 taskId 找失败任务
    uint32_t streamId;   // +8  报错定位键：按逻辑流分片镜像队列
    uint32_t reserved0;  // +12 对齐填充：补齐 8B 边界让 extra 落 +16
    uint64_t extra[6];   // +16 预留
};
static_assert(sizeof(DpuCqe) == DPU_CQE_SIZE, "DpuCqe must be 64B");
static_assert(offsetof(DpuCqe, taskId) == 4, "CQE taskId offset check");
static_assert(offsetof(DpuCqe, streamId) == 8, "CQE streamId offset check");
static_assert(offsetof(DpuCqe, extra) == 16, "CQE extra must be 8B-aligned");

// ─────────────────────────────────────────────────────────────
// 4. 队列元数据：一个半区一页 4KB（游标与算子序号都在这里）
// 游标 pi/ci 一律 acquire/release 访问（__atomic_* 作用在普通 uint32_t 上，布局不变）：
// 生产者先写元素再 release 发布 pi，消费者先 acquire 读 pi 再读元素；ci 的推进同样用 release。
// ─────────────────────────────────────────────────────────────
struct DpuSqMeta { // 4KB
    uint32_t depth;
    uint32_t elemSize;
    uint32_t pi; // 写位置（数据面生产侧推进）
    uint32_t ci; // 读位置（kernel 取格后推进）
    uint8_t rsv[8];
    uint8_t reserved[4096 - 24];
};
static_assert(sizeof(DpuSqMeta) == 4096, "DpuSqMeta must be 4KB");

struct DpuCqMeta { // 4KB
    uint32_t depth;
    uint32_t elemSize;
    uint32_t pi;      // kernel 写回执后推进
    uint32_t ci;      // device 侧守护消费后推进
    uint32_t opIndex; // 当前算子序号：入队侧 DFX 关联用
    uint8_t reserved[4096 - 20];
};
static_assert(sizeof(DpuCqMeta) == 4096, "DpuCqMeta must be 4KB");
static_assert(offsetof(DpuCqMeta, opIndex) == 16, "CqMeta.opIndex offset check");

// ─────────────────────────────────────────────────────────────
// 5. 队列名片：只说地址和尺寸，不说指针语义（跨核只传地址 + 尺寸）
// ─────────────────────────────────────────────────────────────
struct QueueDescriptor {
    uint64_t base;  // 元素区起始地址
    uint32_t depth; // 权威副本在同半区的 4KB 元数据里
    uint32_t elemSize;
    uint64_t piAddr; // 指向本半区 4KB 元数据里的 pi
    uint64_t ciAddr; // 同上
};
static_assert(sizeof(QueueDescriptor) == 32, "QueueDescriptor must be 32B");

// ─────────────────────────────────────────────────────────────
// 6. DPUTAG 内部切分：两个半区各自独立，偏移 0 直接落队列元数据
// ─────────────────────────────────────────────────────────────
// 终止/复位已不落共享内存（改 host 侧 destroyFlag_/resetFlag_），SQ 半区无需再为握手字节留位

// DPUTAG（256KB = 2 × 128KB）里落的数据结构与占用：
//   SQ 半区 [0, 128KB)：
//     [0, 4096)       DpuSqMeta（4KB）：depth/elemSize/pi/ci
//     [4096, 69632)   DpuSqe[1024]（64KB）：SQ 环形任务队列，每格 64B
//     [69632, 128KB)  保留（60KB）
//   CQ 半区 [128KB, 256KB)：
//     [0, 4096)       DpuCqMeta（4KB）：depth/elemSize/pi/ci/opIndex
//     [4096, 4160)    DpuCqe[1]（64B）：CQ 单槽回执队列
//     [4160, 128KB)   保留
//   32 个 DPU 线程合计 8MB
constexpr uint64_t DPU_TAG_HALF_BYTES = 128ULL * 1024;
constexpr uint64_t DPU_TAG_QUEUE_BYTES = DPU_TAG_HALF_BYTES * 2; // 单线程 DPUTAG 申请尺寸
// npu2dpu 半区 —— SQ 方向
constexpr uint64_t DPU_SQ_META_OFF = 0;
constexpr uint64_t DPU_SQ_ELEM_OFF = DPU_SQ_META_OFF + sizeof(DpuSqMeta);
// dpu2npu 半区 —— CQ 方向
constexpr uint64_t DPU_CQ_META_OFF = 0;
constexpr uint64_t DPU_CQ_ELEM_OFF = DPU_CQ_META_OFF + sizeof(DpuCqMeta);
static_assert(DPU_SQ_ELEM_OFF + DPU_SQ_ELEM_BYTES <= DPU_TAG_HALF_BYTES, "SQ 越出 npu2dpu 半区");
static_assert(DPU_CQ_ELEM_OFF + DPU_CQ_ELEM_BYTES <= DPU_TAG_HALF_BYTES, "CQ 越出 dpu2npu 半区");
// 紧凑布局下元数据/元素区不再整页对齐，靠对齐断言钉住各自类型的要求
static_assert(DPU_SQ_META_OFF % alignof(DpuSqMeta) == 0, "SQ 元数据基址未按其对齐要求对齐");
static_assert(DPU_SQ_ELEM_OFF % alignof(DpuSqe) == 0, "SQ 元素区基址未按 DpuSqe 对齐要求对齐");
static_assert(DPU_CQ_META_OFF % alignof(DpuCqMeta) == 0, "CQ 元数据基址未按其对齐要求对齐");
static_assert(DPU_CQ_ELEM_OFF % alignof(DpuCqe) == 0, "CQ 元素区基址未按 DpuCqe 对齐要求对齐");
static_assert((DPU_SQ_DEPTH & (DPU_SQ_DEPTH - 1U)) == 0, "SQ 环形取模要求深度为 2 的幂");
static_assert(DPU_CQ_DEPTH == 1U, "CQ 单槽，深度恒为 1");

// ─────────────────────────────────────────────────────────────
// 7. 启动期入参
// ─────────────────────────────────────────────────────────────
struct DpuSyncLayout { // 名片：host与 kernel 各持一份内存中的同一对象，不落共享内存
    uint64_t sqMeta;   // DpuSqMeta 地址（游标）
    uint64_t sqElem;   // SQ 元素区起始
    uint64_t cqMeta;   // DpuCqMeta 地址（游标 + 算子序号）
    uint64_t cqElem;   // CQ 元素区起始
    uint32_t sqDepth;  // SQ 环深（2 的幂）
    uint32_t cqDepth;  // CQ 槽数（恒 1：单槽）
    uint32_t elemSize;
    // 本轮host与 kernel 同址，故不区分两侧视角；若上板证伪（两地址空间）再补 host/kernel 双份字段
    // 地址一律显式给出、消费侧不做偏移算术：名片存在的意义就是让消费侧不必知道半区布局约定
};
static_assert(sizeof(DpuSyncLayout) == 48, "DpuSyncLayout must be 48B");

class CpuTaskPipe;

struct DpuQueueLaunchParam { // SQ/CQ 队列轮询 kernel 的参数块
    CpuTaskPipe* taskPipe{nullptr};
    uint32_t deviceId{UINT32_MAX};
    uint32_t reserved{0};
};
static_assert(sizeof(DpuQueueLaunchParam) == 16, "DpuQueueLaunchParam ABI layout check");
static_assert(offsetof(DpuQueueLaunchParam, deviceId) == 8, "DpuQueueLaunchParam ABI layout check");

// ─────────────────────────────────────────────────────────────
// 8. 队列名片投递：host → device 侧（DPU 线程实体回填）的 kernel 参数块
// ─────────────────────────────────────────────────────────────
struct DpuQueueNamecardParam { // 纯 POD：整张名片按句柄投给 device 侧实体，跨核不传 std::string
    QueueDescriptor cqDesc;    // 地址为 device 视角（host侧用 DpuShmemMgr::DevPtr 换算）
    // ── 以下为 DPU 线程改挂 AICPU 引擎后追加：按句柄把整张队列名片回填 device 侧实体 ──
    uint64_t threadHandle;  // device 侧线程句柄，名片按它落到实体上
    QueueDescriptor sqDesc; // SQ 半区描述（CQ 守护只读 CQ；SQ 侧留给 AICPU 编排侧提交任务）
    DpuSyncLayout layout;   // 整张名片（device 视角），让消费侧不必知道半区布局约定
    uint32_t notifyNum{0};
    uint64_t notifyHostAddr{0};
    uint64_t notifyDevAddr{0};
};
static_assert(
    offsetof(DpuQueueNamecardParam, threadHandle) == sizeof(QueueDescriptor),
    "追加字段须紧随 cqDesc：param{} 的整体零填假设依赖于此");

struct CpuThreadNotifyAddrUpdateParam {
    uint64_t threadHandle;
    uint32_t notifyNum{0};
    uint64_t notifyHostAddr{0};
    uint64_t notifyDevAddr{0};
};
static_assert(sizeof(CpuThreadNotifyAddrUpdateParam) == 32, "CpuThreadNotifyAddrUpdateParam ABI layout check");

} // namespace hccl

#endif // DPU_SYNC_LAYOUT_H
