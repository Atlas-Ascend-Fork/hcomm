/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "store_dump_shm_data.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib> // strtoull
#include <cstring>
#include <filesystem>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <utility>

#include "store_binary_data_operator.h"
#include "sim_common_defs.h"
#include "sim_log.h"
#include "sim_common_api.h"
#include "sim_capacity_limits.h"
#include "sim_yaml_config.h"
#include "sim_ip_address.h"
#include "sim_loader.h"
#include "runtime_state/sim_models.h"
#include "operation_data/operation_data_ops.h"
#include "runtime_state/db_sim_runner_common.h"
#include "storage/table_access.h"
#include "runtime_state/db_sim_runner_ops.h"

uint8_t g_opExpansionMode = 0;
static const std::string HCCLVM_FLAG_DATA_FILE = "/hcclvm_flag_data.bin";
static const std::string HCCLVM_TASK_DATA_FILE = "/%s_hcclvm_task_data.bin";
static const std::string HCCLVM_SYN_DATA_FILE = "/%s_hcclvm_syn_data.bin";
static const std::string HCCLVM_INSTR_DATA_FILE = "/%s_hcclvm_instr_data.bin";

namespace HcclSim {
namespace fs = std::filesystem;

namespace {
    HcclVmResult LoadCompositeDataByOpIter(
        const sim::operation::OpExecutionKey& key,
        std::map<uint32_t, std::vector<sim::operation::CompositeOpDetail>>& compositeDataMap)
    {
        std::vector<sim::operation::CompositeOpDetail> details;
        loader::Loader dataLoader;
        if (dataLoader.LoadCompositeOpDetailByOpIter(key.commName, key.commHash, key.opIter, details)
            != HcclResult::HCCL_SUCCESS) {
            HCCL_VM_ERROR(
                "LoadCompositeOpDetailByOpIter failed, commName={}, commHash={}, opIter={}", key.commName, key.commHash,
                key.opIter);
            return HcclVmResult::HCCL_SIM_E_INTERNAL;
        }
        for (sim::operation::CompositeOpDetail& detail : details) {
            compositeDataMap[detail.rankId].push_back(std::move(detail));
        }
        return HcclVmResult::HCCL_SIM_SUCCESS;
    }

    // CCU channel -> jetty 映射表 (ccu_channel_jetty_config.xml) 生成规则常量
    // dieId 在 XML 中的偏移: 本项目 dieId + 14
    constexpr uint32_t CCU_XML_DIE_ID_OFFSET = 14;
    // ccu_jump_mode 默认值: 0 commonMode, 后续根据实际情况调整
    constexpr uint32_t CCU_JUMP_MODE_DEFAULT = 0;
    // jetty_group 模板: 完全照抄样例文件, 详细生成规则待补充
    const std::vector<std::string> CCU_JETTY_GROUP_TEMPLATE = {
        "<jetty_group>",
        "    <group id = \"0\" jetty_list = \"0 1 2 3\"/>",
        "    <group id = \"1\" jetty_list = \"256 257 258 259\"/>",
        "    <group id = \"2\" jetty_list = \"512 513 514 515\"/>",
        "    <group id = \"3\" jetty_list = \"768 769 770 771\"/>",
        "    <group id = \"4\" jetty_list = \"1024 1025 1026 1027\"/>",
        "    <group id = \"5\" jetty_list = \"1280 1281 1282 1283\"/>",
        "    <group id = \"6\" jetty_list = \"1536 1537 1538 1539\"/>",
        "    <group id = \"7\" jetty_list = \"1792 1793 1794 1795\"/>",
        "</jetty_group>",
    };

    // deviceId -> rankId 换算; 映射中查不到时报错并回退为原 deviceId.
    uint32_t ResolveRankId(uint32_t deviceId, const std::map<uint32_t, uint32_t>& deviceToRank)
    {
        auto it = deviceToRank.find(deviceId);
        if (it == deviceToRank.end()) {
            HCCL_VM_ERROR(
                "cannot resolve rank by deviceId={}, deviceToRank size={}, fallback to deviceId", deviceId,
                deviceToRank.size());
            return deviceId;
        }
        return it->second;
    }

} // namespace

std::string GetBinLocation()
{
    std::filesystem::path curPath = std::filesystem::current_path();
    return curPath.string();
}

std::string GenDataId()
{
    std::string dataId;

    // 1. 初始化随机数生成器 (static 保证只初始化一次，提高性能和随机性)
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1000, 9999); // 4位随机数
    int32_t retryCount = 10;
    while (true) {
        if (retryCount == 0) {
            HCCL_VM_ERROR("failed to gen dataId.");
            break;
        }
        retryCount--;
        // 2. 获取当前时间戳
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        struct tm buf;
        localtime_r(&in_time_t, &buf);

        // 3. 拼接时间 + 随机数
        std::stringstream ss;
        ss << std::put_time(&buf, "%Y%m%d_%H%M%S") << "_" << dis(gen);
        dataId = ss.str();

        // 4. 冲突检查：检查核心文件是否已存在
        char fileName[256];
        snprintf(fileName, sizeof(fileName), "/%s_model.jsonl.gz", dataId.c_str());
        std::string rootPath = GetBinLocation();
        std::string fullPath = rootPath + DATA_FILE_PATH + fileName;

        if (access(fullPath.c_str(), F_OK) == -1) {
            break; // 文件不存在，DataId 可用，跳出循环
        }
        // 如果冲突，循环会立即重新生成（时间或随机数会变）
    }

    return dataId;
}

HcclVmResult DumpHcclVmFlagData(HcclSim::HcclVmFlagData& flagData)
{
    HCCL_VM_INFO("Start dumping hccl vm flag data...");
    // 1. 构造完整路径
    // 假设 FindRootPath() 已经实现并返回插件根目录
    std::string rootPath = GetBinLocation();
    std::string fullPath = rootPath + DATA_FILE_PATH + HCCLVM_FLAG_DATA_FILE;

    // 2. 写文件，通知runner启动
    FILE* fp = fopen(fullPath.c_str(), "wb");
    if (!fp) {
        HCCL_VM_ERROR("Open file failed: {}, error={}", fullPath, strerror(errno));
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    auto ret = HcclVmFlagDataWrite(fp, flagData);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("HcclVmFlagDataWrite failed. ");
        fclose(fp);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    fclose(fp);
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult GetHcclVmFlagData(HcclSim::HcclVmFlagData& waitFlag)
{
    // 1. 构造完整路径
    // 假设 FindRootPath() 已经实现并返回插件根目录
    std::string rootPath = GetBinLocation();
    std::string fullPath = rootPath + DATA_FILE_PATH + HCCLVM_FLAG_DATA_FILE;
    // 2. 读取文件，等待runner状态
    FILE* fp = fopen(fullPath.c_str(), "rb");
    if (!fp) {
        HCCL_VM_ERROR("Open file failed: {}, error={}", fullPath, strerror(errno));
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    auto ret = HcclVmFlagDataRead(fp, waitFlag, HCCLVM_FLAG_FILE_MAGIC);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("HcclVmFlagDataRead failed. ");
        fclose(fp);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    fclose(fp);
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult DumpDataToFile(const std::string& dataId)
{
    loader::Loader dataLoader;
    std::vector<sim::operation::OpExecutionKey> keys;
    if (dataLoader.LoadOpExecutionKeys(keys) != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR("LoadOpExecutionKeys failed, dataId={}", dataId);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    // 仅当恰好存在一个算子执行记录时才 dump; 0 个或大于 1 个都视为异常，不 dump 并返回错误.
    if (keys.size() != 1) {
        HCCL_VM_WARN(
            "Expected exactly one operator execution to dump, but got {:d}, dataId={}, skip dump", keys.size(), dataId);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    const sim::operation::OpExecutionKey& key = keys.front();
    HCCL_VM_INFO(
        "Dumping op dataId={}, commName={}, commHash={}, opIter={}", dataId, key.commName, key.commHash, key.opIter);

    // 按当前通信域建立 deviceId -> rankId 映射, 供后续 dump 统一换算.
    std::map<uint32_t, uint32_t> deviceToRank;
    auto membersResult = sim::runtime::Db::GetByPred<sim::runtime::Communicator>(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&sim::runtime::Communicator::comm_id, key.commName),
        HcclSim::Storage::Eq(&sim::runtime::Communicator::comm_hash, key.commHash)));
    if (membersResult.ok() && membersResult.value.has_value()) {
        for (const auto& member : *membersResult.value) {
            deviceToRank[static_cast<uint32_t>(member.device_id)] = member.rank_id;
        }
    }
    HCCL_VM_INFO("deviceToRank map size={}, commName={}", deviceToRank.size(), key.commName);

    HcclVmResult ret = DumpHcclVmSynthesisData(dataId, key, deviceToRank);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("failed to dump hccl vm synthesis data, opIter={}", key.opIter);
        return ret;
    }
    ret = DumpHcclVmInstrData(dataId, deviceToRank);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("failed to dump hccl vm instruction data, opIter={}", key.opIter);
        return ret;
    }
    ret = DumpHcclVmTask(dataId, key, deviceToRank);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("failed to dump hccl vm task data, opIter={}", key.opIter);
        return ret;
    }
    ret = GenCcuChannelJettyConfig(deviceToRank);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("failed to gen ccu channel jetty config, opIter={}", key.opIter);
        return ret;
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult CreateMemoryInfo(HcclVmSynData& hvmSynData, const sim::operation::OpMemInfoTab& memInfo, uint32_t rankId)
{
    HCCL_VM_INFO("Enter into create memory info...");
    auto addBuf = [&](uint8_t bufType, uint64_t addr, uint64_t size) {
        if (addr != 0 && size > 0) {
            MemLayoutData memLayoutData{};
            memLayoutData.rank_id = rankId;
            memLayoutData.buffer_type = bufType;
            memLayoutData.start_addr = addr;
            memLayoutData.size = size;
            memLayoutData.global_offset = 0; // 简化约定，与 checker 侧一致
            hvmSynData.memory_info.data.push_back(memLayoutData);
        }
    };
    addBuf(static_cast<uint8_t>(BufferType::INPUT), memInfo.inputAddr, memInfo.inputSize);
    addBuf(static_cast<uint8_t>(BufferType::OUTPUT), memInfo.outputAddr, memInfo.outputSize);
    addBuf(static_cast<uint8_t>(BufferType::CCL), memInfo.cclAddr, memInfo.cclSize);

    hvmSynData.memory_info.count = hvmSynData.memory_info.data.size();
    HCCL_VM_INFO("memory_info count={}, dataSize={}", hvmSynData.memory_info.count, hvmSynData.memory_info.data.size());

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult CreateChannelInfo(HcclVmSynData& hvmSynData, const std::map<uint32_t, uint32_t>& deviceToRank)
{
    HCCL_VM_INFO("Enter into create channel info...");

    std::vector<sim::operation::CcuChannelTab> channels;
    loader::Loader dataLoader;
    if (dataLoader.GetCcuChannelInfo(channels) != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR("GetCcuChannelInfo() failed.");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    for (const auto& ch : channels) {
        ChannelData chData{};
        chData.channelId = static_cast<uint16_t>(ch.channelId);
        chData.srcDieId = static_cast<uint8_t>(ch.srcDieId);
        chData.dstDieId = static_cast<uint8_t>(ch.dstDieId);
        chData.srcDeviceId = ResolveRankId(ch.srcDeviceId, deviceToRank);
        chData.dstDeviceId = ResolveRankId(ch.dstDeviceId, deviceToRank);

        std::memcpy(chData.leid, ch.leid, sizeof(chData.leid));
        std::memcpy(chData.reid, ch.reid, sizeof(chData.reid));

        uint8_t* leidPtr = chData.leid;
        auto lEpRet = sim::runtime::Db::GetOneByPred<sim::runtime::EndPoint>(HcclSim::Storage::Eq(
            &sim::runtime::EndPoint::eid, std::vector<uint8_t>(
                                              reinterpret_cast<const uint8_t*>(leidPtr),
                                              reinterpret_cast<const uint8_t*>(leidPtr) + sizeof(chData.leid))));
        if (!lEpRet.ok()) {
            HCCL_VM_ERROR("cannot find EndPoint by ip addr: {}", static_cast<const void*>(leidPtr));
            return HcclVmResult::HCCL_SIM_E_NOT_FOUND;
        }

        uint8_t* reidPtr = chData.reid;
        auto rEpRet = sim::runtime::Db::GetOneByPred<sim::runtime::EndPoint>(HcclSim::Storage::Eq(
            &sim::runtime::EndPoint::eid, std::vector<uint8_t>(
                                              reinterpret_cast<const uint8_t*>(reidPtr),
                                              reinterpret_cast<const uint8_t*>(reidPtr) + sizeof(chData.reid))));
        if (!rEpRet.ok()) {
            HCCL_VM_ERROR("cannot find EndPoint by ip addr:{}", static_cast<const void*>(reidPtr));
            return HcclVmResult::HCCL_SIM_E_NOT_FOUND;
        }

        auto localEpId = lEpRet->id;
        auto remoteEpId = rEpRet->id;

        auto pairOpt = sim::runtime::Db::GetOneByPred<sim::runtime::EndPointPair>(HcclSim::Storage::And(
            HcclSim::Storage::Eq(&sim::runtime::EndPointPair::local_enpoint_id, localEpId),
            HcclSim::Storage::Eq(&sim::runtime::EndPointPair::remote_enpoint_id, remoteEpId)));

        if (!pairOpt.ok()) {
            HCCL_VM_ERROR("cannot find EndPointPair by local: {} remote: {}", localEpId, remoteEpId);
            return HcclVmResult::HCCL_SIM_E_NOT_FOUND;
        }

        chData.protocol = pairOpt->tp_type;
        if (ch.jettyNum > HcclSim::HCCL_VM_MAX_CHANNEL_JETTIES) {
            HCCL_VM_ERROR(
                "channelId={} jettyNum={} exceeds capacity={}", ch.channelId, ch.jettyNum,
                HcclSim::HCCL_VM_MAX_CHANNEL_JETTIES);
            return HcclVmResult::HCCL_SIM_E_PARA;
        }
        chData.jettyNum = ch.jettyNum;
        std::memcpy(chData.jettyId, ch.jettyId, static_cast<size_t>(chData.jettyNum) * sizeof(uint32_t));

        HCCL_VM_INFO(
            "channelId={}, srcDeviceId={}, dstDeviceId={}, jettyNum={}, protocol={}", chData.channelId,
            chData.srcDeviceId, chData.dstDeviceId, chData.jettyNum, chData.protocol);

        hvmSynData.channel_info.data.push_back(chData);
    }
    hvmSynData.channel_info.count = hvmSynData.channel_info.data.size();
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult CreateJettyInfo(HcclVmSynData& hvmSynData, const std::map<uint32_t, uint32_t>& deviceToRank)
{
    HCCL_VM_INFO("Enter into create jetty info...");
    auto endpointPairs
        = sim::runtime::Db::GetByPred<sim::runtime::EndPointPair>(HcclSim::Storage::All<sim::runtime::EndPointPair>());
    if (!endpointPairs.ok() || !endpointPairs.value.has_value()) {
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    for (const auto& endpoint : *endpointPairs.value) {
        uint64_t localEndPointId = endpoint.local_enpoint_id;
        uint64_t rmtEndPointId = endpoint.remote_enpoint_id;
        uint8_t protocol = endpoint.tp_type;

        // 根据enpoint_id查找本端和对端的EndPoint信息
        auto localEndPointOpt = sim::runtime::Db::GetById<sim::runtime::EndPoint>(localEndPointId);
        if (!localEndPointOpt.ok()) {
            HCCL_VM_ERROR("Get EndPoint failed handle = {}", localEndPointId);
            continue;
        }

        auto rmtEndPointOpt = sim::runtime::Db::GetById<sim::runtime::EndPoint>(rmtEndPointId);
        if (!rmtEndPointOpt.ok()) {
            HCCL_VM_ERROR("Get EndPoint failed handle = {}", rmtEndPointId);
            continue;
        }

        // ctx被同本端eid通往不同对端的channel共用, 按peer_endpoint_id过滤即得该eid对的jetty(由对端RaCtxQpImport回填)
        auto localRaCtxs = sim::runtime::Db::GetByPred<sim::runtime::RaContext>(
            HcclSim::Storage::Eq(&sim::runtime::RaContext::endpoint_id, localEndPointId));
        if (!localRaCtxs.ok() || !localRaCtxs.value.has_value() || localRaCtxs.value->empty()) {
            HCCL_VM_WARN("no RaContext for endpoint = {}", localEndPointId);
            continue;
        }
        std::vector<sim::runtime::RaJetty> localJettys;
        for (const auto& ctx : *localRaCtxs.value) {
            auto jettys = sim::runtime::Db::GetByPred<sim::runtime::RaJetty>(HcclSim::Storage::And(
                HcclSim::Storage::Eq(&sim::runtime::RaJetty::ctx_handle, ctx.id),
                HcclSim::Storage::Eq(&sim::runtime::RaJetty::mode, 3),
                HcclSim::Storage::Eq(&sim::runtime::RaJetty::peer_endpoint_id, rmtEndPointId)));
            if (jettys.ok() && jettys.value.has_value()) {
                localJettys.insert(localJettys.end(), jettys.value->begin(), jettys.value->end());
            }
        }
        if (localJettys.empty()) {
            HCCL_VM_WARN("no attributed jetty for pair local:{} remote:{}, skip", localEndPointId, rmtEndPointId);
            continue;
        }
        std::sort(
            localJettys.begin(), localJettys.end(), [](const sim::runtime::RaJetty& a, const sim::runtime::RaJetty& b) {
                return a.id < b.id;
            });

        // 每个eid对输出一条聚合记录: jettyId[0..jettyNum)为该eid对(本端eid, 对端eid)上全部本端(发送)jetty
        ChannelData channelData{};
        channelData.channelId = endpoint.id;
        channelData.protocol = protocol;

        memcpy(channelData.leid, localEndPointOpt->eid, sizeof(channelData.leid));
        channelData.srcDieId = localEndPointOpt->die_id;
        channelData.srcDeviceId = ResolveRankId(static_cast<uint32_t>(localEndPointOpt->device_id), deviceToRank);

        memcpy(channelData.reid, rmtEndPointOpt->eid, sizeof(channelData.reid));
        channelData.dstDieId = rmtEndPointOpt->die_id;
        channelData.dstDeviceId = ResolveRankId(static_cast<uint32_t>(rmtEndPointOpt->device_id), deviceToRank);

        // ChannelData.jettyId[32] 容量保护, 与 CreateChannelInfo 的截断写法一致
        channelData.jettyNum = static_cast<uint16_t>(
            std::min(localJettys.size(), sizeof(channelData.jettyId) / sizeof(channelData.jettyId[0])));
        for (size_t k = 0; k < channelData.jettyNum; k++) {
            channelData.jettyId[k] = static_cast<uint32_t>(localJettys[k].jetty_id);
        }

        hvmSynData.channel_info.data.push_back(channelData);
    }
    hvmSynData.channel_info.count = hvmSynData.channel_info.data.size();

    // 打印测试-JettyInfo
    std::stringstream chDataStr;
    for (auto chData : hvmSynData.channel_info.data) {
        chDataStr << "channelId=" << chData.channelId << ", srcDeviceId=" << chData.srcDeviceId
                  << ", dstDeviceId=" << chData.dstDeviceId;
        chDataStr << ", srcDieId=" << (int)chData.srcDieId << ", dstDieId=" << (int)chData.dstDieId;
        chDataStr << ", protocol=" << chData.protocol;
        chDataStr << ", leid=";
        for (size_t i = 0; i < sizeof(chData.leid); i++) {
            chDataStr << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned int>(chData.leid[i]);
        }

        chDataStr << ", reid=";
        for (size_t i = 0; i < sizeof(chData.reid); i++) {
            chDataStr << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned int>(chData.reid[i]);
        }
        chDataStr << std::dec << std::setfill(' ');
        chDataStr << std::endl;

        chDataStr << "jettyNum=" << chData.jettyNum << " ";
        for (int i = 0; i < chData.jettyNum; i++) {
            chDataStr << chData.jettyId[i] << " ";
        }
        chDataStr << std::endl;
    }
    HCCL_VM_INFO("{}", chDataStr.str());
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult CreateSimSynData(
    HcclVmSynData& hvmSynData, const sim::operation::OpExecutionKey& key,
    const std::map<uint32_t, uint32_t>& deviceToRank)
{
    HCCL_VM_INFO("Start get simulator synthesis data...");
    // header
    hvmSynData.header.magic = HCCLVM_SYN_FILE_MAGIC;
    hvmSynData.header.version = 1;
    hvmSynData.header.header_size = 20;
    hvmSynData.header.count = 1;

    std::map<uint32_t, std::vector<sim::operation::CompositeOpDetail>> compositeDataMap;
    const auto loadRet = LoadCompositeDataByOpIter(key, compositeDataMap);
    if (loadRet != HcclVmResult::HCCL_SIM_SUCCESS) {
        return loadRet;
    }
    const auto rank0It = compositeDataMap.find(0);
    if (rank0It == compositeDataMap.end() || rank0It->second.empty()) {
        HCCL_VM_ERROR(
            "No op detail found for rankId=0, commName={}, commHash={}, opIter={}", key.commName, key.commHash,
            key.opIter);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    const auto& compRank0 = rank0It->second.front();
    const auto& opTab = compRank0.detail;
    const auto& memInfo = compRank0.memInfo;
    uint32_t rankId = compRank0.rankId;
    if (opTab.opDetail.size() < sizeof(OpDetails)) {
        HCCL_VM_ERROR("opDetail BLOB too small: {} < {}", opTab.opDetail.size(), sizeof(OpDetails));
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    ::OpDetails opDetails{};
    std::memcpy(&opDetails, opTab.opDetail.data(), sizeof(OpDetails));
    g_opExpansionMode = opTab.opExpansionMode;
    hvmSynData.model_info.comm.root = opTab.root;
    hvmSynData.model_info.comm.rank_size = opTab.rankSize;
    hvmSynData.model_info.comm.chip_type = static_cast<uint16_t>(opTab.devType);
    hvmSynData.model_info.comm.op_type = opDetails.opType;
    hvmSynData.model_info.comm.reduce_op = opDetails.reduceType;
    hvmSynData.model_info.comm.data_type = opDetails.dataType;
    hvmSynData.model_info.comm.data_count = opDetails.opV1.count;
    hvmSynData.model_info.comm.op_expansion_mode = opTab.opExpansionMode;
    // 保留占位（与 checker 侧一致）
    hvmSynData.model_info.comm.ccu0_resource_base_addr = 0x123123123;
    hvmSynData.model_info.comm.ccu1_resource_base_addr = 0x456456456;
    hvmSynData.model_info.all2AllDataDes.sendType = opDetails.opV2.sendDataType;
    hvmSynData.model_info.all2AllDataDes.recvType = opDetails.opV2.recvDataType;
    hvmSynData.model_info.all2AllDataDes.sendCount = opDetails.opV2.sendCount;
    hvmSynData.model_info.all2AllDataDes.recvCount = opDetails.opV2.recvCount;
    // opExtInfo 解析 count + sendCountMatrix
    hvmSynData.model_info.all2AllDataDes.count = 0;
    const HcclCMDType cmdType = static_cast<HcclCMDType>(opDetails.opType);
    const bool isAll2AllOp = cmdType == HcclCMDType::HCCL_CMD_ALLTOALL || cmdType == HcclCMDType::HCCL_CMD_ALLTOALLV
                             || cmdType == HcclCMDType::HCCL_CMD_ALLTOALLVC;
    if (isAll2AllOp && opTab.opExtInfo.size() >= sizeof(uint32_t)) {
        uint32_t cnt = 0;
        std::memcpy(&cnt, opTab.opExtInfo.data(), sizeof(uint32_t));
        hvmSynData.model_info.all2AllDataDes.count = cnt;
        for (uint32_t i = 0; i < cnt; i++) {
            uint64_t val = 0;
            size_t offset = sizeof(uint32_t) + i * sizeof(uint64_t);
            if (offset + sizeof(uint64_t) <= opTab.opExtInfo.size()) {
                std::memcpy(&val, opTab.opExtInfo.data() + offset, sizeof(uint64_t));
            }
            hvmSynData.model_info.all2AllDataDes.sendCountMatrix.push_back(val);
        }
    }
    HCCL_VM_INFO("all2AllDataDes send count: {}", hvmSynData.model_info.all2AllDataDes.count);

    // ChannelInfo(CCU) or JettyInfo(AICPU)
    HcclVmResult channelResult;
    if (hvmSynData.model_info.comm.op_expansion_mode == sim::runtime::SimOpExpansionMode::SIM_OP_EXPANSION_MODE_AICPU) {
        channelResult = CreateJettyInfo(hvmSynData, deviceToRank);
    } else {
        channelResult = CreateChannelInfo(hvmSynData, deviceToRank);
    }
    if (channelResult != HcclVmResult::HCCL_SIM_SUCCESS) {
        return channelResult;
    }

    // MemoryLayout
    CreateMemoryInfo(hvmSynData, memInfo, rankId);

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult DumpHcclVmSynthesisData(
    const std::string& dataId, const sim::operation::OpExecutionKey& key,
    const std::map<uint32_t, uint32_t>& deviceToRank)
{
    HCCL_VM_INFO("Start dumping hccl vm synthesis data...");
    // 1. 构造完整路径
    char fileName[256];
    snprintf(fileName, sizeof(fileName), HCCLVM_SYN_DATA_FILE.c_str(), dataId.c_str());

    fs::create_directories(fs::path(InstallPath::ResolveToInstallRoot("data")));
    std::string fullPath = InstallPath::ResolveToInstallRoot("data" + std::string(fileName));

    //  构造hccl vm synthesis数据
    HcclVmSynData hvmSynData;
    auto ret = CreateSimSynData(hvmSynData, key, deviceToRank);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("Get hccl vm synthesis data failed.");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    FILE* fp = fopen(fullPath.c_str(), "wb");
    if (!fp) {
        HCCL_VM_ERROR("Open file failed: {}, err: {}", fullPath, strerror(errno));
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    // 3. dump为二进制文件
    ret = HcclVmSynDataWrite(fp, hvmSynData);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        fclose(fp);
        HCCL_VM_ERROR("Write file failed.");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    fclose(fp);
    HCCL_VM_INFO("Write file success.");
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult CreateSimInstrData(HcclVmInstrData& hvmInstrData, const std::map<uint32_t, uint32_t>& deviceToRank)
{
    HCCL_VM_INFO("Start create simulator instruction data...");
    // 原来实现中 sizeof(hcomm::CcuRep::CcuInstr) == 32判断（与 instrSpace[i][32] 对齐），加 static_assert
    static_assert(sizeof(hcomm::CcuRep::CcuInstr) == 32, "CcuInstr must be 32 bytes");

    std::vector<sim::operation::CcuInstrResTab> allInstrRes;
    loader::Loader dataLoader;
    auto ret = dataLoader.GetInstrResInfo(allInstrRes);
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR("Get instr res info failed.");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    // 过滤掉 instrCount == 0 的记录 等价原来的 Ccustatus.state == 1 的判断
    std::vector<sim::operation::CcuInstrResTab> validInstrRes;
    for (const auto& instrRes : allInstrRes) {
        if (instrRes.instrCount > 0) {
            validInstrRes.push_back(instrRes);
        }
    }

    // header
    hvmInstrData.header.magic = HCCLVM_INSTR_FILE_MAGIC;
    hvmInstrData.header.version = 1;
    hvmInstrData.header.header_size = 20;
    hvmInstrData.header.count = validInstrRes.size(); // CCU有微码指令的个数

    for (const auto& instrRes : validInstrRes) {
        MicrocodeInstrInner mcInstr;
        mcInstr.desc.rank_id = ResolveRankId(instrRes.deviceId, deviceToRank);
        mcInstr.desc.die_id = instrRes.dieId;
        mcInstr.desc.count = instrRes.instrCount;

        if (mcInstr.desc.count > HcclSim::CCU_INSTRUCTION_NUM) {
            HCCL_VM_ERROR(
                "instrCount {} exceeds instruction capacity {}", mcInstr.desc.count, HcclSim::CCU_INSTRUCTION_NUM);
            return HcclVmResult::HCCL_SIM_E_INTERNAL;
        }

        mcInstr.data.resize(mcInstr.desc.count);
        auto size = sizeof(hcomm::CcuRep::CcuInstr) * mcInstr.desc.count;
        memcpy(mcInstr.data.data(), instrRes.instrSpace, size);

        hvmInstrData.instr_data.push_back(mcInstr);
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult DumpHcclVmInstrData(const std::string& dataId, const std::map<uint32_t, uint32_t>& deviceToRank)
{
    HCCL_VM_INFO("Start dumping hccl vm instruction data...");
    // 1. 构造完整路径
    char fileName[256];
    snprintf(fileName, sizeof(fileName), HCCLVM_INSTR_DATA_FILE.c_str(), dataId.c_str());

    fs::create_directories(fs::path(InstallPath::ResolveToInstallRoot("data")));
    std::string fullPath = InstallPath::ResolveToInstallRoot("data" + std::string(fileName));
    FILE* fp = fopen(fullPath.c_str(), "wb");
    if (!fp) {
        HCCL_VM_ERROR("Open file failed: {}, err: {}", fullPath, strerror(errno));
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    // 2. 构造hccl vm instruction数据
    HcclVmInstrData hvmInstrData;
    auto ret = CreateSimInstrData(hvmInstrData, deviceToRank);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        fclose(fp);
        HCCL_VM_ERROR("Get hccl vm instruction data failed.");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    // 非CCU模式无微码指令，无需执行步骤3、步骤4
    if (g_opExpansionMode != sim::runtime::SimOpExpansionMode::SIM_OP_EXPANSION_MODE_CCU) {
        fclose(fp);
        return HcclVmResult::HCCL_SIM_SUCCESS;
    }

    // 3. 检查指令是否为空
    if (hvmInstrData.instr_data.empty()) {
        fclose(fp);
        HCCL_VM_WARN("There is no microcode instruction.");
        return HcclVmResult::HCCL_SIM_SUCCESS;
    }

    // 4. dump为二进制文件
    ret = HcclVmInstrDataWrite(fp, hvmInstrData);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        fclose(fp);
        HCCL_VM_ERROR("Write file failed.");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    fclose(fp);
    HCCL_VM_INFO("Write file success.");
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult CreateSimTaskMetaData(
    HcclVmTaskMetaData& hvmTaskMetaData,
    const std::map<uint32_t, std::vector<sim::operation::CompositeOpDetail>>& compositeDataMap,
    const std::map<uint32_t, uint32_t>& deviceToRank)
{
    HCCL_VM_INFO("Start create simulator task metadata...");

    // 遍历 compositeDataMap，收集所有 task
    for (const auto& rankEntry : compositeDataMap) {
        for (const auto& comp : rankEntry.second) {
            for (const auto& task : comp.tasks) {
                // 反序列化 optaskMeta BLOB
                if (task.optaskMeta.size() < sizeof(HcclTaskMetaData)) {
                    HCCL_VM_WARN("optaskMeta too small: {} < {}", task.optaskMeta.size(), sizeof(HcclTaskMetaData));
                    continue;
                }

                HcclTaskMetaData taskMeta;
                std::memcpy(&taskMeta, task.optaskMeta.data(), sizeof(HcclTaskMetaData));

                // AIV_GRAPH 特殊处理
                if (taskMeta.taskType == HccLTaskMetaType::AIV_GRAPH) {
                    taskMeta.taskData.aiv.launchIdx = hvmTaskMetaData.task_meta.size();
                }

                // deviceId -> rankId 换算
                if (taskMeta.deviceId != UINT64_MAX) {
                    taskMeta.deviceId = ResolveRankId(static_cast<uint32_t>(taskMeta.deviceId), deviceToRank);
                }
                switch (taskMeta.taskType) {
                    case HccLTaskMetaType::MEM_CPY:
                        taskMeta.taskData.transMem.srcDeviceId
                            = ResolveRankId(taskMeta.taskData.transMem.srcDeviceId, deviceToRank);
                        taskMeta.taskData.transMem.dstDeviceId
                            = ResolveRankId(taskMeta.taskData.transMem.dstDeviceId, deviceToRank);
                        break;
                    case HccLTaskMetaType::REDUCE:
                        taskMeta.taskData.reduce.srcDeviceId
                            = ResolveRankId(taskMeta.taskData.reduce.srcDeviceId, deviceToRank);
                        taskMeta.taskData.reduce.dstDeviceId
                            = ResolveRankId(taskMeta.taskData.reduce.dstDeviceId, deviceToRank);
                        break;
                    case HccLTaskMetaType::NOTIFY_RECORD:
                    case HccLTaskMetaType::NOTIFY_WAIT:
                        taskMeta.taskData.notify.srcDeviceId
                            = ResolveRankId(taskMeta.taskData.notify.srcDeviceId, deviceToRank);
                        // dstDeviceId=0 表示本地通知(无远端设备)，是哨兵值而非有效 device_id，跳过换算.
                        if (taskMeta.taskData.notify.dstDeviceId != 0) {
                            taskMeta.taskData.notify.dstDeviceId
                                = ResolveRankId(taskMeta.taskData.notify.dstDeviceId, deviceToRank);
                        }
                        break;
                    case HccLTaskMetaType::WRITE_WITH_NOTIFY:
                        taskMeta.taskData.writeWithNotify.srcDeviceId
                            = ResolveRankId(taskMeta.taskData.writeWithNotify.srcDeviceId, deviceToRank);
                        taskMeta.taskData.writeWithNotify.dstDeviceId
                            = ResolveRankId(taskMeta.taskData.writeWithNotify.dstDeviceId, deviceToRank);
                        break;
                    default:
                        break;
                }

                // 拆分 WRITE_WITH_NOTIFY 任务再 dump，性能仿真先不感知新任务类型
                if (taskMeta.taskType == HccLTaskMetaType::WRITE_WITH_NOTIFY) {
                    HcclTaskMetaData writeMeta;
                    HcclTaskMetaData notifyMeta;
                    SplitWriteWithNotify(taskMeta, writeMeta, notifyMeta);
                    hvmTaskMetaData.task_meta.push_back(writeMeta);
                    hvmTaskMetaData.task_meta.push_back(notifyMeta);
                    continue;
                }

                hvmTaskMetaData.task_meta.push_back(taskMeta);
            }
        }
    }

    // 构造 header
    hvmTaskMetaData.header.magic = HCCLVM_TASK_FILE_MAGIC;
    hvmTaskMetaData.header.version = 1;
    hvmTaskMetaData.header.header_size = 20;
    hvmTaskMetaData.header.count = hvmTaskMetaData.task_meta.size(); // task个数

    HCCL_VM_INFO("taskCount = {}", hvmTaskMetaData.header.count);

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult DumpHcclVmTask(
    const std::string& dataId, const sim::operation::OpExecutionKey& key,
    const std::map<uint32_t, uint32_t>& deviceToRank)
{
    HCCL_VM_INFO("Start dumping hccl vm task data...");
    // 1. 构造完整路径
    char fileName[256];
    snprintf(fileName, sizeof(fileName), HCCLVM_TASK_DATA_FILE.c_str(), dataId.c_str());

    fs::create_directories(fs::path(InstallPath::ResolveToInstallRoot("data")));
    std::string fullPath = InstallPath::ResolveToInstallRoot("data" + std::string(fileName));

    FILE* fp = fopen(fullPath.c_str(), "wb");
    if (!fp) {
        HCCL_VM_ERROR("Open file failed: {}, err: {}", fullPath, strerror(errno));
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    // 2. 获取 compositeDataMap
    std::map<uint32_t, std::vector<sim::operation::CompositeOpDetail>> compositeDataMap;
    if (LoadCompositeDataByOpIter(key, compositeDataMap) != HcclVmResult::HCCL_SIM_SUCCESS) {
        fclose(fp);
        HCCL_VM_ERROR("LoadCompositeDataByOpIter failed.");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    // 3. 构造 hccl vm task meta 数据
    HcclVmTaskMetaData hvmTaskMeta;
    auto ret = CreateSimTaskMetaData(hvmTaskMeta, compositeDataMap, deviceToRank);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        fclose(fp);
        HCCL_VM_ERROR("Get hccl vm task data failed.");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    // 4. 检查任务是否为空
    auto allDevicesResult = sim::runtime::Db::GetByPred<sim::runtime::Device>(
        HcclSim::Storage::Eq(&sim::runtime::Device::status, uint64_t{1}));
    if (!allDevicesResult.ok() || !allDevicesResult.value.has_value()) {
        fclose(fp);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    const auto& allDevices = *allDevicesResult.value;
    // 单卡流程，hccl业务不会生成任务，不需要报错
    if (hvmTaskMeta.task_meta.empty() && allDevices.size() > 1) {
        fclose(fp);
        HCCL_VM_ERROR("There is no task.");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    // 5. dump 为二进制文件
    ret = HcclVmTaskMetaDataWrite(fp, hvmTaskMeta);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        fclose(fp);
        HCCL_VM_ERROR("Write hccl vm task file failed.");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    fclose(fp);
    HCCL_VM_INFO("Write hccl vm task file success.");
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult GenCaModelCcuInstr(hcomm::CcuRep::CcuInstr* instrData, uint32_t instrCnt, uint32_t rankId, uint32_t dieId)
{
    char instrFileName[256] = {0};
    sprintf(instrFileName, "data/CaModel_instruction_0_%u_%u_sq.toml", rankId, dieId + 2);
    HCCL_VM_INFO("instrFileName = {}", instrFileName);
    auto fullPath = InstallPath::ResolveToInstallRoot(instrFileName);

    FILE* fp = fopen(fullPath.c_str(), "w");
    if (fp == nullptr) {
        HCCL_VM_ERROR("open file fail: {}", fullPath.c_str());
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    uint64_t* ptr = (uint64_t*)instrData;
    for (uint32_t idx = 0; idx < instrCnt; idx++) {
        for (uint32_t j = 0; j < 4; j++) {
            if (j == 3) {
                fprintf(fp, "%lu\r\n", ptr[idx * 4 + j]);
            } else {
                fprintf(fp, "%lu\t", ptr[idx * 4 + j]);
            }
        }
    }
    fclose(fp);
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult GenCaModelCcuToml(rtCcuTaskInfo_t* taskInfo, uint32_t rankId, uint32_t streamId)
{
    char tomlFileName[256] = {0};
    sprintf(tomlFileName, "data/CaModel_ccu_%u_%u_%u_sq.toml", rankId, taskInfo->dieId + 2, streamId);
    HCCL_VM_INFO("tomlFileName = {}", tomlFileName);

    auto fullPath = InstallPath::ResolveToInstallRoot(tomlFileName);

    FILE* fp = fopen(fullPath.c_str(), "w");
    if (fp == nullptr) {
        HCCL_VM_ERROR("open file fail: {}", fullPath.c_str());
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    fprintf(fp, "name = \" ccu \"\r\n");
    fprintf(fp, "[[tasks]]\r\n");
    fprintf(fp, "sqe_type = \"CCU\"\r\n");
    fprintf(fp, "mission_id = %u\r\n", taskInfo->missionId);
    fprintf(fp, "iodie_id = %u\r\n", taskInfo->dieId);
    fprintf(fp, "inst_start_id = %u\r\n", taskInfo->instStartId);
    fprintf(fp, "inst_cnt = %u\r\n", taskInfo->instCnt);
    // fprintf(fp, "inst_addr_key_value = %u\r\n", taskInfo->instAddrKeyValue); // key是什么？
    fprintf(fp, "args = [");
    for (uint32_t idx = 0; idx < taskInfo->argSize; idx++) {
        fprintf(fp, "%lu", taskInfo->args[idx]);
        if (idx < taskInfo->argSize - 1) {
            fprintf(fp, ",");
        }
    }
    fprintf(fp, "]");

    fclose(fp);

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

// 生成 CCU channel -> jetty 映射表: data/ccu_channel_jetty_config.xml
// 数据来源: DB ccuChannels 表 (由各 proxy 的 ConfigChannelInfo 写入, 汇聚全部 rank 通道)。
// 生成规则:
//   1. jetty_group: 按 CCU_JETTY_GROUP_TEMPLATE 原样输出;
//   2. 模块命名: ccum_0_{本端rankId}_{本端dieId + 2};
//   3. channel_map: chan=channelId, target_chip_id=对端rankId, target_die_id=对端dieId+2;
//      jetty_group_id: 本端rankId==对端rankId 时为 0, 否则为对端rankId(去除本端)
//      从小到大排序后的下标(从0开始);
//   4. ccu_jump_mode: 默认 CCU_JUMP_MODE_DEFAULT。
HcclVmResult GenCcuChannelJettyConfig(const std::map<uint32_t, uint32_t>& deviceToRank)
{
    HCCL_VM_INFO("Start gen ccu channel jetty config...");

    // 1. 读取全部 CCU 通道
    std::vector<sim::operation::CcuChannelTab> channels;
    loader::Loader dataLoader;
    if (dataLoader.GetCcuChannelInfo(channels) != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR("GetCcuChannelInfo() failed.");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    if (channels.empty()) {
        HCCL_VM_WARN("There is no ccu channel, skip gen ccu channel jetty config.");
        return HcclVmResult::HCCL_SIM_SUCCESS;
    }

    // ccuChannels 落库时 srcRankId/dstRankId 恒为 0(建链时无通信域上下文),
    // 此处依据 srcDeviceId/dstDeviceId 反查当前通信域的真正 rank;只换算 rank 字段,deviceId 保持不变.
    for (auto& ch : channels) {
        ch.srcRankId = ResolveRankId(ch.srcDeviceId, deviceToRank);
        ch.dstRankId = ResolveRankId(ch.dstDeviceId, deviceToRank);
    }

    // 2. 按落库顺序排序, 同一 (模块, channel) 重复配置时以后写的记录为准
    std::sort(
        channels.begin(), channels.end(),
        [](const sim::operation::CcuChannelTab& a, const sim::operation::CcuChannelTab& b) {
            return a.id < b.id;
        });

    // 3. 按 (本端rankId, 本端dieId) 聚合各 CCU 模块, 模块内按 channelId 排序
    std::map<std::pair<uint32_t, uint32_t>, std::map<uint32_t, const sim::operation::CcuChannelTab*>> modules;
    for (const auto& ch : channels) {
        modules[std::make_pair(ch.srcRankId, ch.srcDieId)][ch.channelId] = &ch;
    }

    // 4. 生成 XML 内容
    std::ostringstream oss;
    oss << "<root>\n";
    oss << "    <!-- node name means ccu module name-->\n";
    for (const auto& module : modules) {
        const uint32_t rankId = module.first.first;
        const uint32_t dieId = module.first.second + CCU_XML_DIE_ID_OFFSET;
        const auto& chans = module.second;

        // 对端 rankId (去除本端) 升序集合, 下标即 jetty_group_id
        std::set<uint32_t> remoteRanks;
        for (const auto& item : chans) {
            if (item.second->dstRankId != rankId) {
                remoteRanks.insert(item.second->dstRankId);
            }
        }

        oss << "    <ccum_0_" << rankId << "_" << dieId << ">\n";
        oss << "        <!-- id: jettyGroup index, jetty_list: each list has one more jetty-id -->\n";
        for (const auto& line : CCU_JETTY_GROUP_TEMPLATE) {
            oss << "        " << line << "\n";
        }
        oss << "\n";
        oss << "        <channel_map>\n";
        for (const auto& item : chans) {
            const sim::operation::CcuChannelTab& ch = *item.second;
            uint32_t jettyGroupId = 0;
            if (ch.dstRankId != rankId) {
                jettyGroupId
                    = static_cast<uint32_t>(std::distance(remoteRanks.begin(), remoteRanks.find(ch.dstRankId)));
            }
            oss << "            <chan chan = \"" << ch.channelId << "\" target_chip_id = \"" << ch.dstRankId
                << "\" target_die_id = \"" << ch.dstDieId + CCU_XML_DIE_ID_OFFSET << "\" jetty_group_id = \""
                << jettyGroupId << "\"/>\n";
        }
        oss << "        </channel_map>\n";
        oss << "        <ccu_jump_mode mode=\"" << CCU_JUMP_MODE_DEFAULT
            << "\" /> <!-- 0: commonMode 1: diretlyJump 2: jump-lock -->\n";
        oss << "    </ccum_0_" << rankId << "_" << dieId << ">\n";
        oss << "\n";
    }
    oss << "\n";
    oss << "</root>\n";

    // 5. 落盘到安装目录 data 下
    fs::create_directories(fs::path(InstallPath::ResolveToInstallRoot("data")));
    std::string fullPath = InstallPath::ResolveToInstallRoot("data/ccu_channel_jetty_config.xml");
    FILE* fp = fopen(fullPath.c_str(), "w");
    if (fp == nullptr) {
        HCCL_VM_ERROR("open file fail: {}, err: {}", fullPath, strerror(errno));
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    const std::string content = oss.str();
    if (fwrite(content.data(), 1, content.size(), fp) != content.size() || fclose(fp) != 0) {
        HCCL_VM_ERROR("write file failed: {}", fullPath);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    HCCL_VM_INFO("Gen ccu channel jetty config success: {}, module count: {}", fullPath, modules.size());
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

} // namespace HcclSim
