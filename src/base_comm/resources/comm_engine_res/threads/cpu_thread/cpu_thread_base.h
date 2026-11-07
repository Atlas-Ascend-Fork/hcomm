/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CPU_THREAD_BASE_H
#define CPU_THREAD_BASE_H

#include <atomic>
#include <cstddef>
#include <memory>
#include "thread.h"
// 共享类型与共享内存管理器仍在 legacy_dpu_thread/ 目录（两条通道共用，位置不动），故按相对路径引入
#include "legacy_dpu_thread/dpu_types.h"
#include "legacy_dpu_thread/dpu_shmem_mgr.h"

namespace hccl {

/**
 * @note 职责：DPU 线程的公共机制层。host 侧两个实体都派生自本类 ——
 *       `LegacyDpuThread`（TaskService + taskexception）与 `CpuThread`（SQ/CQ 队列）——
 *       本类只承载两者逐字相同的部分：XPU 域切换与进程级记账、dpu 流、
 *       kernel 二进制与入口解析、下发、终止握手、DPUTAG 共享内存。
 *       两者各自的策略与生命周期动词留在派生类，故机制单份、无复制。
 */
class CpuThreadBase {
public:
    CpuThreadBase() = default;
    // 派生类析构体先于基类析构体执行，故各派生必须在自己析构体首条显式调一次 DeInit()：
    // 本类兜底只保护基类成员，保护不了已析构的派生内存
    virtual ~CpuThreadBase();
    // 持有裸流/上下文句柄，复制会导致重复销毁，禁止拷贝与移动
    CpuThreadBase(const CpuThreadBase&) = delete;
    CpuThreadBase& operator=(const CpuThreadBase&) = delete;
    CpuThreadBase(CpuThreadBase&&) = delete;
    CpuThreadBase& operator=(CpuThreadBase&&) = delete;

    /**
     * @brief 终止握手 + 销毁 dpu 流 + XPU 记账，幂等
     * @return HcclResult 成功返回 HCCL_SUCCESS
     */
    HcclResult DeInit();

    /**
     * @brief 注入身份入参（commId/devId），两条通道的拉起入口共用
     */
    void SetTaskServiceArgs(const Hccl::TaskServiceArgs& args) { args_ = args; }

    Stream* GetStream() const { return stream_.get(); }
    DevType GetDevType() const { return devType_; }
    bool IsKernelLaunched() const { return isKernelLaunched_; }
    uint32_t GetStreamId() const { return streamId_; }
    uint32_t GetDevId() const { return args_.devId; }

    /**
     * @brief 取 DPUTAG 共享内存 host 侧地址（TaskService 与终止握手写 flag 用）
     */
    void* GetDputagAddr() const { return dputagAddr_; }

    /**
     * @brief 取 DPUTAG 共享内存 device 侧地址（仅供 HcclDevMemAcquire 取用，device scatter kernel 解引用）
     */
    void* GetDputagDevAddr() const { return dputagDevAddr_; }

protected:
    HcclResult ResolveDeviceLogicId(int32_t& deviceLogicId);

    /**
     * @brief 申请并注册 DPUTAG 共享内存，并置位 dputagAddr_/dputagHostPtr_
     * @param[in] deviceLogicId 由 ResolveDeviceLogicId 解析后传入（派生类还要用它申请自己的内存）
     * @param[in] size 申请尺寸：默认 100MB（TaskService 通道），SQ/CQ 队列侧传 256KB（两者同 tag
     * 名、各持各的管理器实例）
     */
    HcclResult AllocDputagShmem(int32_t deviceLogicId, uint64_t size = Hccl::SHARE_HBM_MEMORY_SIZE);

    /**
     * @brief 整段 XPU 切换序列：存 npu ctx → 切 dpu 域 → 取 dpu ctx
     * @return HcclResult 成功返回 HCCL_SUCCESS
     * @note 序列不可拆；失败时本函数自行把调用线程 current context 还原，不留残余域
     */
    HcclResult EnterDpuContext();

    /**
     * @brief 加载 libccl_dpu 二进制、建 dpu 流、按入口名解析 kernel 句柄
     * @param[in] entryName json 中的算子名（TaskService 通道 RunDpuRpcSrvLaunch / 队列侧 RunDpuRpcSrvLaunchNew）
     * @return HcclResult 成功返回 HCCL_SUCCESS
     */
    HcclResult PrepareDpuKernelResource(const char* entryName = "RunDpuRpcSrvLaunch");

    /**
     * @brief 在 dpu 流上下发 kernel；参数块由派生类组装
     * @note 前置条件：EnterDpuContext 已切换域、PrepareDpuKernelResource 已建流
     */
    HcclResult LaunchDpuKernelWithArgs(const void* args, size_t argsSize);

    /**
     * @brief 等 dpu 线程回写退出信号（不设上限：提前释放共享内存会与 kernel 的读写冲突）。
     *        TaskService 通道走 DPUTAG 首字节 2→3 握手，SQ/CQ 队列通道由 CpuThread 覆盖为置 destroyFlag_ 等回执
     */
    virtual HcclResult WaitKernelTerminate();

    HcclResult RefreshStreamId();

    // Init 切入 dpu 域后，离开作用域前必须切回 npu ctx：后续任一步失败都靠本守卫在返回前兜底还原，
    // 避免调用线程 current context 滞留 dpu 域（成功路径 launch 已切回，重复切回无害）
    struct CpuCtxRollback {
        explicit CpuCtxRollback(aclrtContext npuCtx) : npuCtx_(npuCtx) {}
        ~CpuCtxRollback();

        aclrtContext npuCtx_;
    };

    // 「流已建、kernel 未下发」时拉起失败的回收守卫。必须显式回收：DeInit 以 isKernelLaunched_ 为
    // 短路判据，失败出口上该标志恒为 false，aclrtDestroyStreamForce 永不执行；而 Stream 以 rtStream
    // 构造、不是 stream owner，流不会随包装析构释放。半拉起线程由调用方回滚销毁，故每次失败重试都漏一条
    // dpu 流。成功下发（isKernelLaunched_ 为 true）时本守卫不介入，交由 DeInit 走完整终止流程
    struct CpuStreamRollback {
        CpuStreamRollback(
            rtStream_t& rtStream, bool& isKernelLaunched, aclrtContext dpuCtx, aclrtContext npuCtx,
            std::unique_ptr<Stream>& stream);
        ~CpuStreamRollback();

        rtStream_t& rtStream_;
        bool& isKernelLaunched_;
        aclrtContext dpuContext_;
        aclrtContext npuContext_;
        std::unique_ptr<Stream>& stream_;
    };

    rtStream_t rtStream_{nullptr};
    aclrtContext npuContext_{nullptr}; // 进入 dpu 域前保存的当前上下文
    aclrtContext dpuContext_{nullptr}; // dpu 域上下文
    aclrtFuncHandle funcHandle_{nullptr};
    aclrtBinHandle binHandle_{nullptr};
    void* dputagAddr_{nullptr}; // DPUTAG host 侧地址（终止握手信号区）
    void* dputagDevAddr_{nullptr}; // DPUTAG device 侧地址（device kernel解引用；宿主侧用作 aclrtMemcpy 的 device 源）
    void* dputagHostPtr_{nullptr}; // 同上，握手字节在偏移 0（两条通道同一口径）
    uint32_t streamId_{0};
    int32_t deviceLogicId_{-1};
    bool isDeviceLogicIdResolved_{false};
    DevType devType_{DevType::DEV_TYPE_COUNT};
    bool isKernelLaunched_{false};
    std::unique_ptr<Stream> stream_;
    DpuShmemMgr shmemMgr_;       // 共享内存管理器：每个实体一份，各管自己申请的内存
    Hccl::TaskServiceArgs args_; // 拉起入口传入的身份入参（commId/devId）

    /**
     * @brief 进程级通信域计数，决定 XPU 设备何时 reset
     * @note 必须是静态量：HrtSetXpuDevice 是进程级设备切换，退化成实例成员会让
     *       reset 时机变成「本通信域计数归零」，与真实语义不符；两条通道共用同一份计数
     */
    static std::atomic<uint32_t> commNum_;
};

} // namespace hccl
#endif // CPU_THREAD_BASE_H
