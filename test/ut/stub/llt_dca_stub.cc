/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "llt_dca_stub.h"
#include "../../../src/coll_communicator_mgr/dfx/detect_connection_anomalies/detect_connection_anomalies.h"
#include <atomic>

namespace hcomm {

namespace {
    std::atomic<uint32_t> g_dcaStubInitCalls{0};
    std::atomic<uint32_t> g_dcaStubDeinitCalls{0};
    std::atomic<uint32_t> g_dcaStubAddIpQueueCalls{0};
} // namespace

uint32_t DcaStubInitCallCount() { return g_dcaStubInitCalls.load(); }

uint32_t DcaStubDeinitCallCount() { return g_dcaStubDeinitCalls.load(); }

uint32_t DcaStubAddIpQueueCallCount() { return g_dcaStubAddIpQueueCalls.load(); }

void DcaStubResetCounts()
{
    g_dcaStubInitCalls.store(0);
    g_dcaStubDeinitCalls.store(0);
    g_dcaStubAddIpQueueCalls.store(0);
}

void DetectConnectionAnomalies::Init() { ++g_dcaStubInitCalls; }

void DetectConnectionAnomalies::Deinit() { ++g_dcaStubDeinitCalls; }

void DetectConnectionAnomalies::AddIpQueue(
    s32, uint32_t, const std::string&, const std::vector<DcaPeerInfo>&, const std::vector<LocalIpInfo>&)
{
    ++g_dcaStubAddIpQueueCalls;
}

HcclResult DetectConnectionAnomalies::EnqueuePeerIpList(const std::vector<DcaPeerInfo>&) { return HCCL_SUCCESS; }

HcclResult DetectConnectionAnomalies::EnqueueLocalIpList(const std::vector<LocalIpInfo>&) { return HCCL_SUCCESS; }

HcclResult DetectConnectionAnomalies::Detect() { return HCCL_SUCCESS; }

void DetectConnectionAnomalies::DetectMonitor() {}

HcclResult DetectConnectionAnomalies::GetPeerIpQueue() { return HCCL_SUCCESS; }

HcclResult DetectConnectionAnomalies::GetLocalIpQueue() { return HCCL_SUCCESS; }

HcclResult DetectConnectionAnomalies::ServerDetectThread(const LocalIpInfo&) { return HCCL_SUCCESS; }

HcclResult DetectConnectionAnomalies::ClientDetectThread(const DcaPeerInfo&) { return HCCL_SUCCESS; }

HcclResult DetectConnectionAnomalies::CreateClientSocketAndConnected(const DcaPeerInfo&, Hccl::Socket*&, bool&)
{
    return HCCL_SUCCESS;
}

HcclResult DetectConnectionAnomalies::ReceiveDetectBroadcast(Hccl::Socket*) { return HCCL_SUCCESS; }

HcclResult DetectConnectionAnomalies::RunServerAcceptLoop(
    const LocalIpInfo&, uint32_t, Hccl::Socket*, std::vector<std::shared_ptr<Hccl::Socket>>&)
{
    return HCCL_SUCCESS;
}

HcclResult DetectConnectionAnomalies::RunBroadcastLoop(const std::vector<std::shared_ptr<Hccl::Socket>>&)
{
    return HCCL_SUCCESS;
}

HcclResult
DetectConnectionAnomalies::BroadcastToSockets(const DetectInfo&, const std::vector<std::shared_ptr<Hccl::Socket>>&)
{
    return HCCL_SUCCESS;
}

HcclResult DetectConnectionAnomalies::ConstructErrorInfo(const DcaPeerInfo&, DetectInfo&) { return HCCL_SUCCESS; }

HcclResult DetectConnectionAnomalies::ProcessDetectionResults() { return HCCL_SUCCESS; }

std::vector<std::string>
DetectConnectionAnomalies::BuildGroupedDetectMessage(const std::unordered_map<std::string, DetectInfo>&) const
{
    return {};
}

void DetectConnectionAnomalies::ThreadDestroy() {}

HcclResult DetectConnectionAnomalies::WaitForDetect() { return HCCL_SUCCESS; }

uint32_t DetectConnectionAnomalies::GetDfsConnFaultTime() { return 0; }

std::string DetectConnectionAnomalies::GetTag(const Hccl::IpAddress& ip) { return ip.GetIpStr() + "_detect_0"; }

} // namespace hcomm
