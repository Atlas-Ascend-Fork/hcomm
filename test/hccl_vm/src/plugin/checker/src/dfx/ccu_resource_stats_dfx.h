/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * Description: CCU 资源统计采集器（dfx）。
 *
 * 采集范围决策（详见 docs/ccu_resource_statistics_design.md）：
 *   - 只在单算子成图阶段采集（CheckOneOp 内 BeginOp/EndOp 门控）；
 *     大图成图路径从不调用 BeginOp，钩子全部 no-op，天然零采集。
 *   - 所有 Record* 接口首行短路：开关关闭或无当前算子上下文时直接返回。
 *
 * 线程模型：采集发生在单算子翻译线程内且翻译期串行（ccu_graph_generator_v3 无并发），
 * collector 状态为普通成员、不加锁；Reset、BeginOp、EndOp、Record 系列与报告读取
 * 均在同一线程执行。
 */

#ifndef CHECKER_DFX_CCU_RESOURCE_STATS_DFX_H
#define CHECKER_DFX_CCU_RESOURCE_STATS_DFX_H

#include <functional>

#include "ccu_resource_stats_types.h"

namespace HcclSim {
namespace CcuResourceStats {

    class Collector;

    // RAII guard：析构即 EndOp 封桶。
    // CheckOneOp 在 BeginOp 之后存在多处失败提前 return，guard 保证任意退出路径都正确封桶，
    // 避免算子上下文悬挂导致下一个 op 的数据串桶。
    class OpGuard {
    public:
        OpGuard() = default;
        explicit OpGuard(Collector* collector);
        ~OpGuard();

        OpGuard(const OpGuard&) = delete;
        OpGuard& operator=(const OpGuard&) = delete;
        OpGuard(OpGuard&& other) noexcept;
        OpGuard& operator=(OpGuard&& other) noexcept;

        bool IsValid() const { return m_collector != nullptr; }

    private:
        Collector* m_collector{nullptr};
    };

    class Collector {
    public:
        static Collector& GetInstance();

        Collector(const Collector&) = delete;
        Collector& operator=(const Collector&) = delete;

        void SetEnabled(bool enabled);
        bool IsEnabled() const;

        // run 级复位（挂 RunChecker 开头；不随 per-op 的 AllRankParamRecorder::Reset 复位）
        void Reset();

        // channel 表提供者（由 main.cc 注入；用于差分参考口径 newChannelIds）。
        // BeginOp 时快照、EndOp 时差分。未注入时差分口径为空，不影响主口径。
        using ChannelTableProvider = std::function<ChannelTableSnapshot()>;
        void SetChannelTableProvider(ChannelTableProvider provider);

        // 返回 RAII guard；开关关闭时返回无效 guard（析构 no-op）
        OpGuard BeginOp(const OpKey& opKey);
        void EndOp();
        bool IsInOp() const;

        // 当前算子展示信息（CheckOneOp 在 CheckerParam 就绪后调用；短路条件同 Record 系列）
        void SetCurrentOpInfo(const OpInfo& opInfo);
        const std::map<OpKey, OpInfo>& GetOpInfos() const;

        // 翻译期上下文（MS 采集方案 A：指令翻译入口按 queue 设置 dev/die）
        void SetTranslateCtx(uint32_t deviceId, uint32_t dieId);
        void ClearTranslateCtx();

        // ---- 记录接口（均短路：开关关闭或无当前算子上下文）----
        void RecordXn(uint32_t deviceId, uint32_t dieId, uint16_t xnId);
        void RecordGsa(uint32_t deviceId, uint32_t dieId, uint16_t gsaId);
        void RecordCke(uint32_t deviceId, uint32_t dieId, uint16_t ckeId);
        void RecordMs(uint32_t deviceId, uint32_t dieId, uint16_t msId);
        // 使用翻译期上下文的 MS 记录（GenSliceFromMs 用；无上下文时 no-op）
        void RecordMsByCtx(uint16_t msId);
        void RecordChannel(uint32_t deviceId, uint32_t dieId, uint32_t channelId);
        void RecordMission(uint32_t deviceId, uint32_t dieId, uint16_t missionId);
        // [beginInstrId, endInstrId) 半开区间入 opInstrIds
        void RecordInstrRange(uint32_t deviceId, uint32_t dieId, uint32_t beginInstrId, uint32_t endInstrId);
        // 幂等记账：同 key 已存在且 value 一致则忽略；不一致打 WARN 并保留首次值
        void RecordLoopGroup(uint32_t deviceId, uint32_t dieId, const LoopGroupKey& key, const LoopGroupRecord& record);
        void SetInstrResidentNum(uint32_t deviceId, uint32_t dieId, uint32_t instrCount);

        // 报告器只读访问
        const OpStatsMap& GetData() const;

    private:
        Collector() = default;

        CcuResStats* CurrentStats(uint32_t deviceId, uint32_t dieId);
        void DiffChannelsAfterOp();

        bool m_enabled{false};
        bool m_inOp{false};
        OpKey m_curOpKey;
        bool m_hasTranslateCtx{false};
        uint32_t m_ctxDeviceId{0};
        uint32_t m_ctxDieId{0};
        ChannelTableProvider m_channelTableProvider;
        ChannelTableSnapshot m_channelSnapshotBeforeOp;
        OpStatsMap m_data;
        std::map<OpKey, OpInfo> m_opInfos;
    };

} // namespace CcuResourceStats
} // namespace HcclSim

#endif // CHECKER_DFX_CCU_RESOURCE_STATS_DFX_H
