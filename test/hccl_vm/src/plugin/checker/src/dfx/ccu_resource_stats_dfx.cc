/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_resource_stats_dfx.h"

#include "sim_log.h"

namespace HcclSim {
namespace CcuResourceStats {

    OpGuard::OpGuard(Collector* collector) : m_collector(collector) {}

    OpGuard::~OpGuard()
    {
        if (m_collector != nullptr) {
            m_collector->EndOp();
        }
    }

    OpGuard::OpGuard(OpGuard&& other) noexcept : m_collector(other.m_collector) { other.m_collector = nullptr; }

    OpGuard& OpGuard::operator=(OpGuard&& other) noexcept
    {
        if (this != &other) {
            if (m_collector != nullptr) {
                m_collector->EndOp();
            }
            m_collector = other.m_collector;
            other.m_collector = nullptr;
        }
        return *this;
    }

    Collector& Collector::GetInstance()
    {
        static Collector instance;
        return instance;
    }

    void Collector::SetEnabled(bool enabled) { m_enabled = enabled; }

    bool Collector::IsEnabled() const { return m_enabled; }

    void Collector::Reset()
    {
        m_data.clear();
        m_opInfos.clear();
        m_inOp = false;
        m_curOpKey = OpKey{};
        m_hasTranslateCtx = false;
        m_ctxDeviceId = 0;
        m_ctxDieId = 0;
        m_channelSnapshotBeforeOp.clear();
    }

    void Collector::SetChannelTableProvider(ChannelTableProvider provider)
    {
        m_channelTableProvider = std::move(provider);
    }

    OpGuard Collector::BeginOp(const OpKey& opKey)
    {
        if (!m_enabled) {
            return OpGuard();
        }
        if (m_inOp) {
            HCCL_VM_WARN(
                "[CcuResourceStats] BeginOp while previous op bucket is not closed, "
                "prevCommName={}, prevOpIter={}; force closing previous bucket",
                m_curOpKey.commName, m_curOpKey.opIter);
            EndOp();
        }
        const auto existIt = m_data.find(opKey);
        if (existIt != m_data.end()) {
            HCCL_VM_WARN(
                "[CcuResourceStats] op bucket already exists (op retry or duplicate key), overwrite. "
                "commName={}, commHash={}, opIter={}",
                opKey.commName, opKey.commHash, opKey.opIter);
            m_data.erase(existIt);
        }
        m_curOpKey = opKey;
        m_inOp = true;
        m_data[m_curOpKey]; // 建空桶
        if (m_channelTableProvider) {
            m_channelSnapshotBeforeOp = m_channelTableProvider();
        }
        HCCL_VM_INFO(
            "[CcuResourceStats] BeginOp, commName={}, commHash={}, opIter={}", m_curOpKey.commName, m_curOpKey.commHash,
            m_curOpKey.opIter);
        return OpGuard(this);
    }

    void Collector::EndOp()
    {
        if (!m_inOp) {
            return;
        }
        DiffChannelsAfterOp();
        HCCL_VM_INFO(
            "[CcuResourceStats] EndOp, commName={}, commHash={}, opIter={}", m_curOpKey.commName, m_curOpKey.commHash,
            m_curOpKey.opIter);
        m_inOp = false;
        m_hasTranslateCtx = false;
        m_curOpKey = OpKey{};
        m_channelSnapshotBeforeOp.clear();
    }

    bool Collector::IsInOp() const { return m_inOp; }

    void Collector::SetCurrentOpInfo(const OpInfo& opInfo)
    {
        if (!m_enabled || !m_inOp) {
            return;
        }
        m_opInfos[m_curOpKey] = opInfo;
    }

    const std::map<OpKey, OpInfo>& Collector::GetOpInfos() const { return m_opInfos; }

    void Collector::SetTranslateCtx(uint32_t deviceId, uint32_t dieId)
    {
        m_hasTranslateCtx = true;
        m_ctxDeviceId = deviceId;
        m_ctxDieId = dieId;
    }

    void Collector::ClearTranslateCtx() { m_hasTranslateCtx = false; }

    CcuResStats* Collector::CurrentStats(uint32_t deviceId, uint32_t dieId)
    {
        if (!m_enabled || !m_inOp) {
            return nullptr;
        }
        return &m_data[m_curOpKey][deviceId][dieId];
    }

    void Collector::RecordXn(uint32_t deviceId, uint32_t dieId, uint16_t xnId)
    {
        CcuResStats* stats = CurrentStats(deviceId, dieId);
        if (stats == nullptr) {
            return;
        }
        stats->xnIds.insert(xnId);
        if (!stats->xnUsed || xnId > stats->xnMaxId) {
            stats->xnMaxId = xnId;
            stats->xnUsed = true;
        }
    }

    void Collector::RecordGsa(uint32_t deviceId, uint32_t dieId, uint16_t gsaId)
    {
        CcuResStats* stats = CurrentStats(deviceId, dieId);
        if (stats == nullptr) {
            return;
        }
        stats->gsaIds.insert(gsaId);
    }

    void Collector::RecordCke(uint32_t deviceId, uint32_t dieId, uint16_t ckeId)
    {
        CcuResStats* stats = CurrentStats(deviceId, dieId);
        if (stats == nullptr) {
            return;
        }
        stats->ckeIds.insert(ckeId);
    }

    void Collector::RecordMs(uint32_t deviceId, uint32_t dieId, uint16_t msId)
    {
        CcuResStats* stats = CurrentStats(deviceId, dieId);
        if (stats == nullptr) {
            return;
        }
        stats->msIds.insert(msId);
    }

    void Collector::RecordMsByCtx(uint16_t msId)
    {
        if (!m_enabled || !m_inOp || !m_hasTranslateCtx) {
            return;
        }
        RecordMs(m_ctxDeviceId, m_ctxDieId, msId);
    }

    void Collector::RecordChannel(uint32_t deviceId, uint32_t dieId, uint32_t channelId)
    {
        CcuResStats* stats = CurrentStats(deviceId, dieId);
        if (stats == nullptr) {
            return;
        }
        stats->channelIds.insert(channelId);
    }

    void Collector::RecordMission(uint32_t deviceId, uint32_t dieId, uint16_t missionId)
    {
        CcuResStats* stats = CurrentStats(deviceId, dieId);
        if (stats == nullptr) {
            return;
        }
        stats->missionIds.insert(missionId);
    }

    void Collector::RecordInstrRange(uint32_t deviceId, uint32_t dieId, uint32_t beginInstrId, uint32_t endInstrId)
    {
        CcuResStats* stats = CurrentStats(deviceId, dieId);
        if (stats == nullptr || beginInstrId >= endInstrId) {
            return;
        }
        for (uint32_t instrId = beginInstrId; instrId < endInstrId; ++instrId) {
            stats->opInstrIds.insert(instrId);
        }
    }

    void Collector::RecordLoopGroup(
        uint32_t deviceId, uint32_t dieId, const LoopGroupKey& key, const LoopGroupRecord& record)
    {
        CcuResStats* stats = CurrentStats(deviceId, dieId);
        if (stats == nullptr) {
            return;
        }
        const auto it = stats->loopGroups.find(key);
        if (it == stats->loopGroups.end()) {
            stats->loopGroups.emplace(key, record);
            return;
        }
        if (it->second.templateNum != record.templateNum || it->second.expandedNum != record.expandedNum) {
            HCCL_VM_WARN(
                "[CcuResourceStats] LoopGroup record conflict (keep first), deviceId={}, dieId={}, "
                "missionId={}, instStartId={}, loopGroupInstrId={}, first=(n={}, expanded={}), new=(n={}, expanded={})",
                deviceId, dieId, key.missionId, key.instStartId, key.loopGroupInstrId, it->second.templateNum,
                it->second.expandedNum, record.templateNum, record.expandedNum);
        }
    }

    void Collector::SetInstrResidentNum(uint32_t deviceId, uint32_t dieId, uint32_t instrCount)
    {
        CcuResStats* stats = CurrentStats(deviceId, dieId);
        if (stats == nullptr) {
            return;
        }
        stats->instrResidentNum = instrCount;
    }

    const OpStatsMap& Collector::GetData() const { return m_data; }

    void Collector::DiffChannelsAfterOp()
    {
        if (!m_channelTableProvider) {
            return;
        }
        const ChannelTableSnapshot after = m_channelTableProvider();
        DevStatsMap& devMap = m_data[m_curOpKey];
        for (const auto& devEntry : after) {
            const uint32_t deviceId = devEntry.first;
            const auto snapshotDevIt = m_channelSnapshotBeforeOp.find(deviceId);
            for (const auto& dieEntry : devEntry.second) {
                const uint32_t dieId = dieEntry.first;
                const std::set<uint32_t>* beforeIds = nullptr;
                if (snapshotDevIt != m_channelSnapshotBeforeOp.end()) {
                    const auto snapshotDieIt = snapshotDevIt->second.find(dieId);
                    if (snapshotDieIt != snapshotDevIt->second.end()) {
                        beforeIds = &snapshotDieIt->second;
                    }
                }
                for (const uint32_t channelId : dieEntry.second) {
                    if (beforeIds == nullptr || beforeIds->find(channelId) == beforeIds->end()) {
                        devMap[deviceId][dieId].newChannelIds.insert(channelId);
                    }
                }
            }
        }
    }

} // namespace CcuResourceStats
} // namespace HcclSim
