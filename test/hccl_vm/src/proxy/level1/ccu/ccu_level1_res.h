/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_LEVEL1_RES_H
#define CCU_LEVEL1_RES_H

#include <cstdint>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "ccu_types.h"

namespace HcclSim {
namespace CcuSim {

    /*==================== 1. CCU 实例描述 ====================*/

    /**
     * @brief CCU 实例（进程级单例池管理）。
     *
     * 对应真身 CcuInstanceMgr 中的一个实例条目。每个实例代表一次
     * HcommCcuInsCreate 调用的产物，持有 die 位掩码（标识该实例占用了
     * 哪些 die 的资源），后续的 kernel 注册、资源描述符都挂在实例上。
     *
     * 生命周期：CreateIns 创建 → BindInsToComm 绑定通信域 → 多次
     * Register/Launch → DestroyIns 或 UnbindInsFromComm 销毁。
     */
    struct SimCcuInsDesc {
        uint64_t insHandle{0}; /**< 实例句柄（CcuResMgr 自签发，进程内唯一非零）。 */
        uint32_t dieMask{0};   /**< 占用 die 位掩码（bit0=die0, bit1=die1）。 */
        bool destroyed{false}; /**< 是否已销毁（销毁后句柄不可再使用）。 */
    };

    /*==================== 2. CCU 实例资源描述符 ====================*/

    /**
     * @brief CCU 资源描述符（对应真身 CcuResDescMgr）。
     *
     * 记录 per-die 的资源请求/实批/剩余，支持多资源类型（每个 desc 可按
     * HcommCcuResType 分别 SetNum/QueryNum）。
     *
     * 生命周期：CreateResDesc 创建（全零）→ SetNum 逐项设置请求量 →
     * InsQueryResDesc 回填实批量 → DestroyResDesc 销毁。
     */
    constexpr uint32_t CCU_RES_TYPE_MAX = 8; /**< HcommCcuResType 枚举上限（0-7）。 */
    struct SimCcuInsResDesc {
        uint64_t descHandle{0};                  /**< 描述符句柄。 */
        uint64_t insHandle{0};                   /**< 所属实例句柄。 */
        uint32_t dieId{0};                       /**< Die ID。 */
        uint32_t reqNum[CCU_RES_TYPE_MAX]{0};    /**< 请求量（按 resType 索引）。 */
        uint32_t allocNum[CCU_RES_TYPE_MAX]{0};  /**< 实批量。 */
        uint32_t remainNum[CCU_RES_TYPE_MAX]{0}; /**< 剩余量。 */
        bool destroyed{false};
    };

    /*==================== 3. 实例级变量/事件预约 ====================*/

    /**
     * @brief 实例级变量/事件预约记录（对应真身 CcuVarEventResMgr）。
     *
     * 预约一段连续 XN（Variable）或 CKE（Event）资源并完成 VA 映射。
     * 与 kernel 级 CcuVariableAlloc 的区别：
     * - 实例级：HcommCcuVariableAlloc 在 Register 之前由算法模板基座调用，
     *   跨 kernel 持续存在、Host 可直接读写（VA 可用）、经 WriteVariableWithNotify
     *   跨 rank 交换（PreSync 机制）。
     * - kernel 级：CcuVariableAlloc 在 kernelFunc 执行期调用，生命周期 =
     *   注册事务（回放期每次 Launch 重建 varValues）。
     */
    struct SimCcuAcquiredVar {
        uint64_t acqHandle{0}; /**< 预约句柄（CcuResMgr 自签发）。 */
        uint64_t insHandle{0}; /**< 所属 CCU 实例。 */
        uint32_t dieId{0};     /**< 资源所在 die。 */
        uint32_t num{0};       /**< 预约个数（销毁后置 0 标记无效）。 */
        uint64_t baseVa{0};    /**< VA 基址（首元素虚拟地址，Alloc 时映射）。 */
        bool isEvent{false};   /**< true = Event(CKE)，false = Variable(XN)。 */
    };

    /*==================== 4. 资源管理器（进程级单例） ====================*/

    /**
     * @brief CCU 每 die 的资源容量（V1/V2 分档，值与 level2 SetCcuV1/V2ResourceBasicInfo 一致）。
     *
     * 数据来源：sim::runtime::Device.soc_version 决定版本；容量值为固定常量
     * （真实硬件每 die 相同，仿真中不区分 die 的容量差异）。
     */
    struct CcuCapacity {
        uint32_t xnNum{0};          /**< XN 寄存器数（V1=3072, V2=16384）。 */
        uint32_t gsaNum{0};         /**< GSA 寄存器数（V1=3072, V2=0）。 */
        uint32_t ckeNum{0};         /**< CKE 事件数（V1/V2=1024）。 */
        uint32_t msNum{0};          /**< MS 缓冲数（V1/V2=1536）。 */
        uint32_t loopEngineNum{0};  /**< Loop 引擎数（V1=200, V2=512）。 */
        uint32_t missionNum{0};     /**< Mission 数（V1/V2=16）。 */
        uint32_t instructionNum{0}; /**< 指令空间条数（V1/V2=32K）。 */
    };

    /*==================== 5. 版本化资源地址布局 ====================*/

    /**
     * @brief CCU 资源版本（决定地址布局）。
     * - V1(A5/Ascend950)：有独立 GSA（Address 经 GSA 中转）；
     * - V2(A6/Ascend960)：无 GSA（Address 本质 = Variable(XN)）。
     */
    enum class SimCcuVersion : uint8_t {
        V1 = 0,
        V2 = 1,
    };

    /**
     * @brief 版本化资源地址布局（地址 = dieBase + offset + id*stride）。
     *
     * 权威来源：hcomm src/base_comm/resources/ccu/ccu_device/ccu_res_specs.{h,cc}
     *   V1  = CCUM(0x800000) + INS(0x100000) → GSA(0x8000) → XN(0x8000) → CKE
     *   V2  = CCUM(0x0)      + INS(0x100000) → XN(0x40000) → CKE
     * XN/CKE/GSA 单元步长均 8B。V2 偏移与 checker `findTypeByAddr`、runner
     * `ccu_resource_manager.cc` 完全一致。
     */
    struct CcuResLayout {
        SimCcuVersion version{SimCcuVersion::V1};
        uint64_t xnOffset{0};  /**< XN 区相对 dieBase 偏移。 */
        uint64_t gsaOffset{0}; /**< GSA 区偏移（V2 无 GSA，返回 0）。 */
        uint64_t ckeOffset{0}; /**< CKE 区偏移。 */
        uint32_t xnStride{8};  /**< XN 单元步长（B）。 */
        uint32_t gsaStride{8}; /**< GSA 单元步长（B）。 */
        uint32_t ckeStride{8}; /**< CKE 单元步长（B）。 */
    };

    /**
     * @brief CCU 资源单例管理器（进程级，mutex 保护）。
     *
     * 管理四类资源池（全部进程内，不入库——R4 原则）：
     * 1. 实例池       insPool_      （insHandle → SimCcuInsDesc）
     * 2. 绑定表       commBinding_  （commId → insHandle）
     * 3. 描述符池     resDescPool_  （descHandle → SimCcuInsResDesc）
     * 4. 预约池       acqVarPool_   （acqHandle → SimCcuAcquiredVar）
     *
     * 资源容量与地址模型：
     * - 容量来源：sim::runtime::Device.soc_version → V1/V2 容量表（与 level2 硬编码一致）；
     * - 基址来源：per-die 固定基址（die0=0x123123123, die1=0x456456456，与 level2 一致）；
     * - 分配跟踪：per-die 分配水位（xn/gsa/ckeAllocated），起始 = SYNC_SLOT_NUM（同步区预留）；
     * - VA 编码：按版本分档（CcuResLayout）——`baseVa = dieBase(dieId) + 区偏移 + id*8`，
     *   V1 含 GSA 区（0x900000）、XN(0x908000)、CKE(0x910000)；
     *   V2 XN(0x100000)、CKE(0x140000)（与 checker/runner 反查布局一致）。
     */
    class CcuResMgr {
    public:
        static CcuResMgr& Instance();

        /*---------- CCU 实例生命周期 ----------*/

        /**
         * @brief 创建 CCU 实例（对应 HcommCcuInsCreate / InsCreateDefault）。
         * @param dieMask  占用 die 位掩码（bit0=die0, bit1=die1）。
         * @return 实例句柄（非零，进程内唯一）；失败返回 0。
         */
        uint64_t CreateIns(uint32_t dieMask);

        bool DestroyIns(uint64_t insHandle);
        bool IsValidIns(uint64_t insHandle) const;

        /*---------- 通信域绑定 ----------*/

        bool BindInsToComm(uint64_t commId, uint64_t insHandle);
        bool UnbindInsFromComm(uint64_t commId);
        bool QueryBoundIns(uint64_t commId, uint64_t& insHandle) const;

        /*---------- 资源描述符 ----------*/

        /** @brief 创建描述符（per-die，全零初始）。 */
        uint64_t CreateResDesc(uint32_t dieId);
        bool DestroyResDesc(uint64_t descHandle);
        bool SetResDescNum(uint64_t descHandle, uint32_t resType, uint32_t reqNum);
        bool QueryResDescNum(uint64_t descHandle, uint32_t resType, uint32_t& reqNum) const;
        bool QueryResDescDieId(uint64_t descHandle, uint32_t& dieId) const;
        bool QueryResDescIns(uint64_t insHandle, uint64_t descHandle);
        bool QueryRemainResDesc(uint64_t descHandle);

        /**
         * @brief 以查询模式执行 kernelFunc 统计资源需求（对应真身 GetKernelResourceRequest）。
         *
         * 在临时录制事务中执行 kernelFunc（不注册、不产轨迹），从录制上下文
         * 提取 Variable/Event/Buffer/Loop 的实际分配计数，写入 desc 的 reqNum[]。
         *
         * @param descHandle 输出目标描述符。
         * @param kernelFunc kernel 函数指针。
         * @param kernelArgs kernel 参数（0/1 个）。
         * @param argNum     参数个数。
         * @param dieId      查询目标 die。
         * @return 成功返回 true。
         */
        bool QueryResourceReq(
            uint64_t descHandle, const void* kernelFunc, const void** kernelArgs, uint32_t argNum, uint32_t dieId);

        /*---------- 实例级变量/事件预约 ----------*/

        /**
         * @brief 预约一段连续 Variable(XN) 资源。
         *
         * VA 编码：baseVa = dieBase(dieId) + xnAllocated[dieId] * 8
         * （从该 die 的 XN 分配水位开始，按步长递增；水位随分配推进）。
         */
        uint64_t VariableAlloc(uint64_t insHandle, uint32_t dieId, uint32_t num);

        /**
         * @brief 预约一段连续 Event(CKE) 资源。
         *
         * VA 编码：baseVa = dieBase(dieId) + xnCapacity * 8 + ckeAllocated[dieId] * 2
         * （CKE 空间排在 XN 空间之后，按 CKE 步长递增）。
         */
        uint64_t EventAlloc(uint64_t insHandle, uint32_t dieId, uint32_t num);

        bool VariableGetAddr(uint64_t acqHandle, uint32_t index, uint64_t& va) const;
        bool EventGetAddr(uint64_t acqHandle, uint32_t index, uint64_t& va) const;

        /*---------- 容量查询 ----------*/

        /**
         * @brief 获取当前设备的 CCU 容量（V1/V2 分档）。
         *
         * 读取 sim::runtime::Device.soc_version：Ascend950 → V1 容量，Ascend960 → V2 容量。
         * 查询失败时返回 V1 容量并打 WARN（降级，保证流程可继续）。
         */
        CcuCapacity GetCapacity() const;

        /**
         * @brief 获取指定 die 的 CCU 资源基址。
         *
         * 与 level2 SetCcuV1/V2ResourceBasicInfo 中的 resourceAddr 一致：
         * die0 → 0x123123123, die1 → 0x456456456。
         */
        uint64_t GetDieBaseAddr(uint32_t dieId) const;

        /*---------- 版本与版本化地址布局 ----------*/

        /**
         * @brief 同步区槽位数（hcomm `CcuTransport::INIT_XN_NUM` / `INIT_CKE_NUM`）。
         *
         * XN/CKE id ∈ [0, SYNC_SLOT_NUM) 为跨 rank 交换区（per-channel 预留，值入库共享）；
         * 普通（本进程）资源从 SYNC_SLOT_NUM 起分配。
         */
        static constexpr uint32_t SYNC_SLOT_NUM = 4;

        /**
         * @brief 当前 CCU 资源版本。
         *
         * 读 sim::runtime::Device.soc_version：Ascend950 → V1(A5)，Ascend960 → V2(A6)；
         * 未知/查询失败降级 V1 并打 WARN（与 level2 hccp_ccu_stub 同口径）。
         */
        SimCcuVersion GetCcuVersion() const;

        /** @brief 当前版本的资源地址布局（区偏移 + 步长）。 */
        CcuResLayout GetLayout() const;

        /** @brief XN 区基址（dieBase + xnOffset）。 */
        uint64_t GetXnBaseAddr(uint32_t dieId) const;

        /** @brief GSA 区基址（dieBase + gsaOffset；V2 无 GSA 返回 0）。 */
        uint64_t GetGsaBaseAddr(uint32_t dieId) const;

        /** @brief CKE 区基址（dieBase + ckeOffset）。 */
        uint64_t GetCkeBaseAddr(uint32_t dieId) const;

    private:
        CcuResMgr() = default;
        ~CcuResMgr() = default;
        CcuResMgr(const CcuResMgr&) = delete;
        CcuResMgr& operator=(const CcuResMgr&) = delete;

        uint64_t NextHandle();
        /** @brief 查询指定 die 的剩余 XN / GSA / CKE 数（capacity - allocated）。 */
        uint32_t RemainingXn(uint32_t dieId) const;
        uint32_t RemainingGsa(uint32_t dieId) const;
        uint32_t RemainingCke(uint32_t dieId) const;

        mutable std::mutex mutex_;
        uint64_t nextHandle_{1};

        std::map<uint64_t, SimCcuInsDesc> insPool_;
        std::map<uint64_t, uint64_t> commBinding_;
        std::map<uint64_t, SimCcuInsResDesc> resDescPool_;
        std::map<uint64_t, SimCcuAcquiredVar> acqVarPool_;

        /* per-die 分配水位（容量范围内的连续分配指针；同步区 [0,SYNC_SLOT_NUM) 预留） */
        uint32_t xnAllocated_[2]{SYNC_SLOT_NUM, SYNC_SLOT_NUM};  /**< per die 已分配 XN 数（水位推进）。 */
        uint32_t gsaAllocated_[2]{SYNC_SLOT_NUM, SYNC_SLOT_NUM}; /**< per die 已分配 GSA 数（V1 用）。 */
        uint32_t ckeAllocated_[2]{SYNC_SLOT_NUM, SYNC_SLOT_NUM}; /**< per die 已分配 CKE 数。 */

        /** @brief 本进程是否已清空交换表（`CcuSyncResTab`）——首次 CreateIns 时清一次，防跨 run 旧行。 */
        bool exchangeTableCleared_{false};
    };

} // namespace CcuSim
} // namespace HcclSim

#endif // CCU_LEVEL1_RES_H
