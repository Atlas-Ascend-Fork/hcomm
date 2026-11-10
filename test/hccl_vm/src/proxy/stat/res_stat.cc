/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "stat/res_stat.h"

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sim_log.h"
#include "sim_common_api.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "runtime_state/sim_models.h"

namespace fs = std::filesystem;

namespace HcclSim {

namespace {
    std::mutex g_resStatMutex;
    // value 用 unique_ptr：GetResStat 返回的引用在 map rehash 后依然有效
    std::unordered_map<uint64_t, std::unique_ptr<SimResStat>> g_resStatMap;

    void WriteResStatCsv(uint64_t deviceId, const SimResStat& stat)
    {
        const auto deviceRet = sim::runtime::Db::GetById<sim::runtime::Device>(deviceId);
        if (!deviceRet.ok()) {
            HCCL_VM_ERROR("query Device failed, deviceId {}: {}", deviceId, deviceRet.diagnostic);
            return;
        }
        const uint64_t serverKey = static_cast<uint64_t>(deviceRet.value->server_id);
        const auto serverRet = sim::runtime::Db::GetById<sim::runtime::Server>(serverKey);
        if (!serverRet.ok()) {
            HCCL_VM_ERROR("query Server failed, serverKey {}: {}", serverKey, serverRet.diagnostic);
            return;
        }
        const uint64_t deviceLogicId = deviceRet.value->logic_id;
        const auto now = std::chrono::system_clock::now();
        const std::time_t nowTime = std::chrono::system_clock::to_time_t(now);
        std::tm tmBuf{};
        localtime_r(&nowTime, &tmBuf);
        std::ostringstream timeOss;
        timeOss << std::put_time(&tmBuf, "%Y%m%d_%H%M%S");
        // 文件名带 board_id: POD形态(4层拓扑)下同一POD内各board的卡物理/逻辑id值域相同(如均为0~7),
        // 不含board_id会导致同POD内不同board的同号卡文件名冲突而相互覆盖。
        // server形态(3层拓扑)下board_id恒为0
        const std::string fileName = std::to_string(serverRet.value->pod_id) + "_"
                                     + std::to_string(serverRet.value->server_id) + "_"
                                     + std::to_string(serverRet.value->board_id) + "_" + std::to_string(deviceLogicId)
                                     + "_res_" + timeOss.str() + ".csv";
        const std::string filePath = InstallPath::ResolveToInstallRoot("data/stat/" + fileName);

        std::ofstream out(filePath);
        if (!out.is_open()) {
            HCCL_VM_ERROR("open res stat csv failed: {}", filePath);
            return;
        }

        // host 内存不按 deviceId 统计，调用侧恒记到 deviceId=0，这里统一取全局计数
        uint64_t allocHostBytes = 0;
        uint64_t freeHostBytes = 0;
        {
            std::lock_guard<std::mutex> lock(g_resStatMutex);
            const auto hostStatIt = g_resStatMap.find(0);
            if (hostStatIt != g_resStatMap.end()) {
                allocHostBytes = hostStatIt->second->mem.allocHostBytes.load(std::memory_order_relaxed);
                freeHostBytes = hostStatIt->second->mem.freeHostBytes.load(std::memory_order_relaxed);
            }
        }

        out << "metric,value\n"
            << "memAllocDevBytes," << stat.mem.allocDevBytes.load(std::memory_order_relaxed) << "\n"
            << "memFreeDevBytes," << stat.mem.freeDevBytes.load(std::memory_order_relaxed) << "\n"
            << "memAllocHostBytes," << allocHostBytes << "\n"
            << "memFreeHostBytes," << freeHostBytes << "\n"
            << "memAllocVirBytes," << stat.mem.allocVirBytes.load(std::memory_order_relaxed) << "\n"
            << "memFreeVirBytes," << stat.mem.freeVirBytes.load(std::memory_order_relaxed) << "\n"
            << "memAllocPhyBytes," << stat.mem.allocPhyBytes.load(std::memory_order_relaxed) << "\n"
            << "memFreePhyBytes," << stat.mem.freePhyBytes.load(std::memory_order_relaxed) << "\n"
            << "jettyAllocCount," << stat.jetty.allocCount.load(std::memory_order_relaxed) << "\n"
            << "jettyFreeCount," << stat.jetty.freeCount.load(std::memory_order_relaxed) << "\n"
            << "qpAllocCount," << stat.qp.allocCount.load(std::memory_order_relaxed) << "\n"
            << "qpFreeCount," << stat.qp.freeCount.load(std::memory_order_relaxed) << "\n"
            << "notifyAllocCount," << stat.notify.allocCount.load(std::memory_order_relaxed) << "\n"
            << "notifyFreeCount," << stat.notify.freeCount.load(std::memory_order_relaxed) << "\n"
            << "aivUsedCoreMax," << stat.aivCore.usedCoreMax.load(std::memory_order_relaxed) << "\n";
        out.close();

        HCCL_VM_INFO("res stat dumped to {}", filePath);
    }
} // namespace

SimResStat& GetResStat(uint64_t deviceId)
{
    std::lock_guard<std::mutex> lock(g_resStatMutex);
    auto& slot = g_resStatMap[deviceId];
    if (!slot) {
        slot = std::make_unique<SimResStat>();
    }
    return *slot;
}

HcclVmResult DumpResStatToCsv()
{
    if (std::getenv("HCCL_VM_RES_STAT_DUMP") == nullptr) {
        return HCCL_SIM_SUCCESS;
    }
    std::vector<std::pair<uint64_t, const SimResStat*>> snapshot;
    {
        std::lock_guard<std::mutex> lock(g_resStatMutex);
        snapshot.reserve(g_resStatMap.size());
        for (const auto& entry : g_resStatMap) {
            snapshot.emplace_back(entry.first, entry.second.get());
        }
    }
    if (snapshot.empty()) {
        HCCL_VM_WARN("no res stat recorded, nothing to dump");
        return HCCL_SIM_SUCCESS;
    }
    // create_directories 对已存在目录不报错，多进程并发创建安全
    std::error_code createEc;
    fs::create_directories(InstallPath::ResolveToInstallRoot("data/stat"), createEc);
    if (createEc) {
        HCCL_VM_ERROR("create data/stat dir failed: {}", createEc.message());
        return HCCL_SIM_PROXY_ERROR_STAT;
    }
    for (const auto& [deviceId, stat] : snapshot) {
        WriteResStatCsv(deviceId, *stat); // 单个device失败只记日志，继续其余device
    }
    return HCCL_SIM_SUCCESS;
}

} // namespace HcclSim
