/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 日志染色: 模块 tag (须在 include sim_log.h 之前)
#define HCCL_VM_MODULE "COMM_STUB"

#include <algorithm>
#include <arpa/inet.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <netdb.h>
#include <pthread.h>
#include <string>
#include <sys/time.h>
#include <unordered_map>
#include <unistd.h>
#include <vector>
#include <cstdlib>

#include "acl/acl_base.h"
#include "acl/acl_rt.h"
#include "sim_log.h"
#include "hccp_common.h"
#include "sim_ip_address.h"
#include "runtime/base.h"
#include "runtime_state/db_sim_runner_common.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "operation_data/operation_data_ops.h"
#include "storage/table_access.h"
#include "hccl_proxy_common.h"
#include "level1_proxy_common.h"

using sim::runtime::g_cur_comm_key;
using sim::runtime::g_cur_device_key;

extern "C" void SimBindHcclCommMember(HcclComm comm, uint64_t communicatorId);

namespace {
constexpr uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;
constexpr uint32_t kInvalidSubCommId = 0xFFFFFFFFU;
constexpr size_t kSizeMb = 1024ULL * 1024ULL;
constexpr uint32_t kIpAddressBufferLen = 64;
constexpr uint32_t kRootInfoIdentifierMaxLength = 128;
constexpr uint32_t kDefaultRootPort = 29500;
constexpr uint32_t kMaxNotifyPerThread = 40;
constexpr uint32_t kMaxNotifyPerChannel = 64;

struct RootInfoInternal {
    char ip[kIpAddressBufferLen];
    uint32_t port;
    uint32_t nicDeploy;
    char identifier[kRootInfoIdentifierMaxLength];
    uint32_t deviceId;
};

// HcclComm 仅在当前进程有效; value 是该进程, 本 rank 对应的 Communicator.id.
// 每个 rank 进程各自维护这张表, 进程之间不共享该索引.
std::unordered_map<HcclComm, uint64_t> g_hcclCommToCommunicatorId;

uint64_t HashDeviceIds(const std::vector<uint64_t>& deviceIds)
{
    uint64_t hash = kFnvOffset;
    hash ^= deviceIds.size();
    hash *= kFnvPrime;
    for (const uint64_t deviceId : deviceIds) {
        hash ^= deviceId;
        hash *= kFnvPrime;
    }
    return hash;
}

uint64_t
InsertOperatorMember(const std::string& name, uint32_t rankSize, uint32_t rankId, uint64_t deviceId, uint64_t commHash)
{
    // 一条记录只描述一个 rank; 同一 comm_id 下的全部记录共同描述完整通信域.
    sim::runtime::Communicator record{};
    std::strncpy(record.comm_id, name.c_str(), sizeof(record.comm_id) - 1);
    record.comm_hash = commHash;
    record.rank_size = rankSize;
    record.rank_id = rankId;
    record.device_id = deviceId;
    record.id = 0;
    auto inserted = sim::runtime::Db::Add<sim::runtime::Communicator>(record);
    return inserted.ok() && inserted.value.has_value() ? inserted.value->value : 0;
}

// RTLD_NEXT 无法穿透 torch_npu 等场景中以 RTLD_LOCAL 方式加载的真实 HCCL 库链,
// 失败时按 SONAME 显式加载 libhccl.so (其依赖 libhcomm.so 导出全部北向符号) 后重试.
void* DlsymRealHccl(const char* funcName) { return sim::DlsymRealWithFallback(funcName, "libhccl.so"); }

bool QueryCommunicator(HcclComm comm, std::string& commName, uint32_t& rankSize, uint32_t& rankId)
{
    // 主域信息以真实 HCCL 已创建的通信域为准, 不再从 CommConfig 或拓扑配置推导.
    using GetCommNameFunc = HcclResult (*)(HcclComm, char*);
    using GetRankSizeFunc = HcclResult (*)(HcclComm, uint32_t*);
    using GetRankIdFunc = HcclResult (*)(HcclComm, uint32_t*);
    const auto getCommName = reinterpret_cast<GetCommNameFunc>(DlsymRealHccl("HcclGetCommName"));
    const auto getRankSize = reinterpret_cast<GetRankSizeFunc>(DlsymRealHccl("HcclGetRankSize"));
    const auto getRankId = reinterpret_cast<GetRankIdFunc>(DlsymRealHccl("HcclGetRankId"));
    if (comm == nullptr || getCommName == nullptr || getRankSize == nullptr || getRankId == nullptr) {
        return false;
    }

    char name[COMM_NAME_MAX_LENGTH] = {};
    if (getCommName(comm, name) != HCCL_SUCCESS || getRankSize(comm, &rankSize) != HCCL_SUCCESS
        || getRankId(comm, &rankId) != HCCL_SUCCESS || name[0] == '\0' || rankSize == 0 || rankId >= rankSize) {
        return false;
    }
    commName = name;
    return true;
}

bool BuildWorldMember(HcclComm comm, uint64_t& communicatorId)
{
    std::string commName;
    uint32_t rankSize = 0;
    uint32_t rankId = 0;
    if (!QueryCommunicator(comm, commName, rankSize, rankId) || g_cur_device_key == 0) {
        return false;
    }
    constexpr uint64_t kWorldCommHash = 0;
    // 每个 rank 无条件插入自己的主域表行, 不查找, 不复用其他 rank 的记录.
    communicatorId = InsertOperatorMember(commName, rankSize, rankId, g_cur_device_key, kWorldCommHash);
    if (communicatorId == 0) {
        return false;
    }
    // 表行插入后再等待所有 rank 到齐, 后续子域创建才能按 rank 查询完整的 device 映射.
    if (!sim::runtime::WaitCommunicatorReady(commName.c_str(), kWorldCommHash, rankSize)) {
        (void)sim::runtime::Db::Delete<sim::runtime::Communicator>(
            HcclSim::Storage::Eq(&sim::runtime::Communicator::id, communicatorId));
        communicatorId = 0;
        return false;
    }
    return true;
}

bool BuildSubCommMember(
    HcclComm subComm, const std::vector<uint64_t>& deviceIds, uint32_t rankNum, uint64_t& communicatorId)
{
    std::string commName;
    uint32_t rankSize = 0;
    uint32_t rankId = 0;
    if (!QueryCommunicator(subComm, commName, rankSize, rankId) || g_cur_device_key == 0) {
        return false;
    }
    if (rankNum == 0 || rankNum != rankSize || deviceIds.size() != rankNum || rankId >= deviceIds.size()
        || deviceIds[rankId] != g_cur_device_key) {
        return false;
    }
    const uint64_t commHash = HashDeviceIds(deviceIds);
    communicatorId = InsertOperatorMember(commName, rankSize, rankId, deviceIds[rankId], commHash);
    if (communicatorId == 0) {
        return false;
    }
    // 使用真实 HCCL 子域名称等待所有成员建模完成，避免算子记录时缺少 rank -> device 映射。
    if (!sim::runtime::WaitCommunicatorReady(commName.c_str(), commHash, rankSize)) {
        (void)sim::runtime::Db::Delete<sim::runtime::Communicator>(
            HcclSim::Storage::Eq(&sim::runtime::Communicator::id, communicatorId));
        communicatorId = 0;
        return false;
    }
    return true;
}

bool IsLevel1Mode()
{
    const char* level = std::getenv("HCCL_VM_LEVEL");
    return level != nullptr && std::strcmp(level, "1") == 0;
}

uint32_t GetLevel1DefaultBufferSizeMb()
{
    const char* envValue = std::getenv("HCCL_COMM_BUFFSIZE");
    if (envValue != nullptr && envValue[0] != '\0') {
        char* endptr = nullptr;
        const long value = std::strtol(envValue, &endptr, 10);
        if (endptr != envValue && (*endptr == '\0' || *endptr == '\n') && value > 0
            && value <= static_cast<long>(UINT32_MAX)) {
            return static_cast<uint32_t>(value);
        }
    }
    return HCCL_COMM_DEFAULT_BUFFSIZE;
}

bool ResolveLevel1DeviceArgument(int32_t device, uint64_t& deviceId)
{
    if (device < 0) {
        return false;
    }
    const uint64_t currentServerKey = sim::runtime::GetCurServerId();
    if (currentServerKey == 0) {
        HCCL_VM_ERROR("cannot resolve Level1 device argument: current server is unavailable");
        return false;
    }
    const uint64_t value = static_cast<uint64_t>(device);
    const auto dbDevice = sim::runtime::Db::GetOneByPred<sim::runtime::Device>(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&sim::runtime::Device::server_id, currentServerKey),
        HcclSim::Storage::Or(
            HcclSim::Storage::Eq(&sim::runtime::Device::id, value),
            HcclSim::Storage::Eq(&sim::runtime::Device::physical_id, value),
            HcclSim::Storage::Eq(&sim::runtime::Device::logic_id, value))));
    if (!dbDevice.ok()) {
        return false;
    }
    deviceId = dbDevice.value->id;
    return true;
}

bool ResolveLevel1DeviceId(uint32_t rankId, uint64_t& deviceId)
{
    const uint64_t currentServerKey = sim::runtime::GetCurServerId();
    if (currentServerKey == 0) {
        HCCL_VM_ERROR("cannot resolve Level1 device: current server is unavailable");
        return false;
    }
    if (g_cur_device_key != 0) {
        const auto currentDevice = sim::runtime::Db::GetById<sim::runtime::Device>(g_cur_device_key);
        // aclrtSetDevice 已按当前 server 和逻辑设备选择设备,并将对应的数据库主键保存到 g_cur_device_key
        if (currentDevice.ok() && currentDevice.value->server_id == currentServerKey) {
            deviceId = currentDevice.value->id;
            return true;
        }
    }
    const int rankTableDeviceId = sim::RankTable::Instance().GetDeviceId(rankId);
    if (rankTableDeviceId < 0) {
        return false;
    }
    const auto device = sim::runtime::Db::GetOneByPred<sim::runtime::Device>(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&sim::runtime::Device::server_id, currentServerKey),
        HcclSim::Storage::Eq(&sim::runtime::Device::physical_id, static_cast<uint32_t>(rankTableDeviceId))));
    if (!device.ok()) {
        return false;
    }
    deviceId = device.value->id;
    return true;
}

HcclResult
CreateLevel1Communicator(const char* clusterInfo, uint32_t rank, const HcclCommConfig* config, HcclComm* comm)
{
    if (comm == nullptr) {
        return HCCL_E_PTR;
    }
    const char* rankTablePath = clusterInfo != nullptr ? clusterInfo : std::getenv("RANK_TABLE_FILE");
    if (rankTablePath == nullptr) {
        HCCL_VM_ERROR("{}: clusterInfo is nullptr and RANK_TABLE_FILE env not set", __func__);
        return HCCL_E_INTERNAL;
    }
    if (!sim::RankTable::Instance().Load(rankTablePath)) {
        HCCL_VM_ERROR("{}: failed to load rank table from: {}", __func__, rankTablePath);
        return HCCL_E_OPEN_FILE_FAILURE;
    }
    const auto existingComms = sim::runtime::Db::GetByPred<sim::runtime::Communicator>(
        HcclSim::Storage::Eq(&sim::runtime::Communicator::rank_id, rank));
    if (existingComms.ok() && existingComms.value.has_value() && !existingComms.value->empty()) {
        HCCL_VM_ERROR(
            "{}: communicator already initialized for rank {:d}, duplicate init is not allowed", __func__, rank);
        return HCCL_E_INTERNAL;
    }
    const uint32_t rankSize = sim::RankTable::Instance().GetRankSize();
    if (rank >= rankSize) {
        return HCCL_E_PARA;
    }

    uint64_t deviceId = 0;
    if (!ResolveLevel1DeviceId(rank, deviceId)) {
        return HCCL_E_INTERNAL;
    }
    // level1 当前仅建模单通信域；未配置名称时，所有 rank 使用真实 HCCL 的默认名称。
    char commName[COMM_NAME_MAX_LENGTH] = {};
    if (config != nullptr && config->hcclCommName[0] != '\0') {
        std::strncpy(commName, config->hcclCommName, sizeof(commName) - 1);
    } else {
        std::strncpy(commName, "hccl_world_group", sizeof(commName) - 1);
    }

    sim::runtime::Communicator record{};
    record.id = 0;
    std::strncpy(record.comm_id, commName, sizeof(record.comm_id) - 1);
    record.comm_hash = 0;
    record.rank_size = rankSize;
    record.rank_id = rank;
    record.device_id = deviceId;
    record.deterministic = config != nullptr && config->hcclDeterministic != HCCL_COMM_DETERMINISTIC_CONFIG_NOT_SET ?
                               config->hcclDeterministic :
                               HCCL_COMM_DEFAULT_DETERMINISTIC;
    record.op_expansion_mode
        = config != nullptr ? static_cast<uint8_t>(config->hcclOpExpansionMode) : HCCL_COMM_DEFAULT_OP_EXPANSION_MODE;
    record.rdma_traffic_class
        = config != nullptr && config->hcclRdmaTrafficClass != HCCL_COMM_TRAFFIC_CLASS_CONFIG_NOT_SET ?
              config->hcclRdmaTrafficClass :
              HCCL_COMM_TRAFFIC_CLASS_CONFIG_NOT_SET;
    record.rdma_service_level
        = config != nullptr && config->hcclRdmaServiceLevel != HCCL_COMM_SERVICE_LEVEL_CONFIG_NOT_SET ?
              config->hcclRdmaServiceLevel :
              HCCL_COMM_SERVICE_LEVEL_CONFIG_NOT_SET;

    auto inserted = sim::runtime::Db::Add<sim::runtime::Communicator>(record);
    const uint64_t communicatorId = inserted.ok() && inserted.value.has_value() ? inserted.value->value : 0;
    if (communicatorId == 0) {
        return HCCL_E_INTERNAL;
    }

    // 当前 rank 完成成员登记后，等待同一通信域的所有 rank 都登记完成。
    // 算子执行阶段需要通过 rankId 查询对应的 deviceId，过早返回会导致
    // 先启动的 rank 在其他 rank 尚未写入 Communicator 记录时查询失败。
    if (!sim::runtime::WaitCommunicatorReady(commName, 0, rankSize)) {
        (void)sim::runtime::Db::Delete<sim::runtime::Communicator>(
            HcclSim::Storage::Eq(&sim::runtime::Communicator::id, communicatorId));
        return HCCL_E_INTERNAL;
    }

    const uint32_t bufferMb = config != nullptr && config->hcclBufferSize != HCCL_COMM_BUFFSIZE_CONFIG_NOT_SET ?
                                  config->hcclBufferSize :
                                  GetLevel1DefaultBufferSizeMb();
    void* buffer = nullptr;
    if (aclrtMalloc(&buffer, static_cast<size_t>(bufferMb) * kSizeMb, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS
        || buffer == nullptr) {
        return HCCL_E_MEMORY;
    }
    sim::runtime::HcclBuffer hcclBuffer{};
    hcclBuffer.id = 0;
    hcclBuffer.commId = communicatorId;
    hcclBuffer.addr = reinterpret_cast<uint64_t>(buffer);
    hcclBuffer.size = static_cast<uint64_t>(bufferMb) * kSizeMb;
    (void)sim::runtime::Db::Add<sim::runtime::HcclBuffer>(hcclBuffer);
    *comm = reinterpret_cast<HcclComm>(communicatorId);
    return HCCL_SUCCESS;
}

HcclResult CreateLevel1CommAll(uint32_t ndev, int32_t* devices, HcclComm* comms)
{
    if (ndev == 0) {
        HCCL_VM_ERROR("{}: ndev is 0", __func__);
        return HCCL_E_PARA;
    }
    if (devices == nullptr) {
        HCCL_VM_ERROR("{}: devices is nullptr", __func__);
        return HCCL_E_PTR;
    }
    if (comms == nullptr) {
        HCCL_VM_ERROR("{}: comms is nullptr", __func__);
        return HCCL_E_PTR;
    }

    // Preserve level1's original single-process, per-device creation path.
    const size_t bufferSizeBytes = static_cast<size_t>(GetLevel1DefaultBufferSizeMb()) * kSizeMb;
    for (uint32_t index = 0; index < ndev; ++index) {
        uint64_t deviceId = 0;
        if (!ResolveLevel1DeviceArgument(devices[index], deviceId)) {
            HCCL_VM_ERROR("{}: cannot resolve deviceId for device {:d}", __func__, devices[index]);
            return HCCL_E_INTERNAL;
        }

        sim::runtime::Communicator communicator{};
        communicator.id = 0;
        // HcclCommInitAll 创建的是同一个 world 通信域的各个 rank。
        std::strncpy(communicator.comm_id, "hccl_world_group", sizeof(communicator.comm_id) - 1);
        communicator.rank_size = ndev;
        communicator.rank_id = index;
        communicator.device_id = deviceId;
        communicator.comm_hash = 0;
        communicator.deterministic = HCCL_COMM_DEFAULT_DETERMINISTIC;
        communicator.op_expansion_mode = HCCL_COMM_DEFAULT_OP_EXPANSION_MODE;
        communicator.rdma_traffic_class = HCCL_COMM_TRAFFIC_CLASS_CONFIG_NOT_SET;
        communicator.rdma_service_level = HCCL_COMM_SERVICE_LEVEL_CONFIG_NOT_SET;
        auto inserted = sim::runtime::Db::Add<sim::runtime::Communicator>(communicator);
        const uint64_t communicatorId = inserted.ok() && inserted.value.has_value() ? inserted.value->value : 0;
        if (communicatorId == 0) {
            HCCL_VM_ERROR("{}: failed to add communicator to DB for device {:d}", __func__, devices[index]);
            return HCCL_E_INTERNAL;
        }

        void* buffer = nullptr;
        if (aclrtMalloc(&buffer, bufferSizeBytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS || buffer == nullptr) {
            HCCL_VM_ERROR(
                "{}: aclrtMalloc failed for hcclBuffer, device={:d}, size={:d}", __func__, devices[index],
                bufferSizeBytes);
            return HCCL_E_MEMORY;
        }
        sim::runtime::HcclBuffer hcclBuffer{};
        hcclBuffer.id = 0;
        hcclBuffer.commId = communicatorId;
        hcclBuffer.addr = reinterpret_cast<uint64_t>(buffer);
        hcclBuffer.size = bufferSizeBytes;
        (void)sim::runtime::Db::Add<sim::runtime::HcclBuffer>(hcclBuffer);

        comms[index] = reinterpret_cast<HcclComm>(communicatorId);
        HCCL_VM_INFO(
            "{} device {:d}: commId={:d}, rank={:d}, devId={:d}", __func__, index, communicatorId, index,
            devices[index]);
    }
    HCCL_VM_INFO("{} success, ndev={:d}", __func__, ndev);
    return HCCL_SUCCESS;
}

HcclResult DestroyLevel1Communicator(HcclComm comm)
{
    if (comm == nullptr) {
        HCCL_VM_ERROR("{}: comm is nullptr", __func__);
        return HCCL_E_PTR;
    }

    const uint64_t commId = reinterpret_cast<uint64_t>(comm);
    const auto communicator = sim::runtime::Db::GetById<sim::runtime::Communicator>(commId);
    if (!communicator.ok()) {
        HCCL_VM_WARN("{}: communicator {:d} not found, may already be destroyed", __func__, commId);
        return HCCL_SUCCESS;
    }

    if (!sim::runtime::WaitCommunicatorDestroyReady(commId)) {
        HCCL_VM_ERROR("{}: destroy barrier failed, commId={:d}", __func__, commId);
        return HCCL_E_INTERNAL;
    }

    auto buffersResult = sim::runtime::Db::GetByPred<sim::runtime::HcclBuffer>(
        HcclSim::Storage::Eq(&sim::runtime::HcclBuffer::commId, commId));
    if (buffersResult.ok() && buffersResult.value.has_value()) {
        for (const auto& buffer : *buffersResult.value) {
            (void)sim::runtime::Db::Delete<sim::runtime::HcclBuffer>(
                HcclSim::Storage::Eq(&sim::runtime::HcclBuffer::id, buffer.id));
        }
    }
    auto threadsResult = sim::runtime::Db::GetByPred<sim::runtime::HcclThread>(
        HcclSim::Storage::Eq(&sim::runtime::HcclThread::commId, commId));
    if (threadsResult.ok() && threadsResult.value.has_value()) {
        for (const auto& thread : *threadsResult.value) {
            for (uint32_t index = 0; index < thread.notifyNum && index < kMaxNotifyPerThread; ++index) {
                if (thread.notifyId[index] != 0) {
                    (void)sim::runtime::Db::Delete<sim::runtime::Notify>(
                        HcclSim::Storage::Eq(&sim::runtime::Notify::id, thread.notifyId[index]));
                }
            }
            (void)sim::runtime::Db::Delete<sim::runtime::HcclThread>(
                HcclSim::Storage::Eq(&sim::runtime::HcclThread::id, thread.id));
        }
    }
    auto channelsResult = sim::runtime::Db::GetByPred<sim::runtime::HcclChannel>(
        HcclSim::Storage::Eq(&sim::runtime::HcclChannel::commId, commId));
    if (channelsResult.ok() && channelsResult.value.has_value()) {
        for (const auto& channel : *channelsResult.value) {
            for (uint32_t index = 0; index < channel.notifyNum && index < kMaxNotifyPerChannel; ++index) {
                if (channel.notifyId[index] != 0) {
                    (void)sim::runtime::Db::Delete<sim::runtime::Notify>(
                        HcclSim::Storage::Eq(&sim::runtime::Notify::id, channel.notifyId[index]));
                }
            }
            (void)sim::runtime::Db::Delete<sim::runtime::HcclChannel>(
                HcclSim::Storage::Eq(&sim::runtime::HcclChannel::id, channel.id));
        }
    }
    EngineCtxDestroyByCommId(commId);
    auto memoriesResult = sim::runtime::Db::GetByPred<sim::runtime::HcclMem>(
        HcclSim::Storage::Eq(&sim::runtime::HcclMem::commId, commId));
    if (memoriesResult.ok() && memoriesResult.value.has_value()) {
        for (const auto& memory : *memoriesResult.value) {
            (void)sim::runtime::Db::Delete<sim::runtime::HcclMem>(
                HcclSim::Storage::Eq(&sim::runtime::HcclMem::id, memory.id));
        }
    }
    // Communicator 记录同时被 opDetails.commId 引用，必须保留到 checker 完成读取后，
    // 由 HcclVmResetCommDomain 统一清理，不能在单个 HcclComm 销毁时删除。
    HCCL_VM_INFO(
        "{} success, commId={:d}, rank={:d}, rankSize={:d}", __func__, commId, communicator.value->rank_id,
        communicator.value->rank_size);
    return HCCL_SUCCESS;
}

HcclResult GetLevel1RootInfo(HcclRootInfo* rootInfo)
{
    const uint64_t deviceId = sim::runtime::GetCurrDeviceId();
    if (deviceId == 0 || deviceId > UINT32_MAX) {
        HCCL_VM_ERROR("{}: failed to get current deviceId={}", __func__, deviceId);
        return HCCL_E_INTERNAL;
    }

    char hostIp[kIpAddressBufferLen] = "127.0.0.1";
    char hostname[256] = {};
    if (gethostname(hostname, sizeof(hostname)) == 0) {
        const hostent* hostEntry = gethostbyname(hostname);
        if (hostEntry != nullptr && hostEntry->h_addrtype == AF_INET && hostEntry->h_addr != nullptr) {
            in_addr address{};
            std::memcpy(&address, hostEntry->h_addr, sizeof(address));
            if (inet_ntop(AF_INET, &address, hostIp, sizeof(hostIp)) == nullptr) {
                std::strncpy(hostIp, "127.0.0.1", sizeof(hostIp));
            }
        }
    }
    timeval timeValue{};
    gettimeofday(&timeValue, nullptr);
    const uint64_t timestamp
        = static_cast<uint64_t>(timeValue.tv_sec) * 1000ULL + static_cast<uint64_t>(timeValue.tv_usec) / 1000ULL;
    char identifier[kRootInfoIdentifierMaxLength] = {};
    std::snprintf(
        identifier, sizeof(identifier), "%s_%u_%u_%lu", hostIp, kDefaultRootPort, static_cast<uint32_t>(deviceId),
        static_cast<unsigned long>(timestamp));

    RootInfoInternal internal{};
    std::strncpy(internal.ip, hostIp, kIpAddressBufferLen - 1);
    internal.port = kDefaultRootPort;
    internal.nicDeploy = 1;
    std::memcpy(internal.identifier, identifier, kRootInfoIdentifierMaxLength);
    internal.deviceId = static_cast<uint32_t>(deviceId);
    static_assert(
        sizeof(RootInfoInternal) <= HCCL_ROOT_INFO_BYTES, "RootInfoInternal exceeds HcclRootInfo.internal buffer size");
    std::memset(rootInfo->internal, 0, HCCL_ROOT_INFO_BYTES);
    std::memcpy(rootInfo->internal, &internal, sizeof(internal));
    HCCL_VM_INFO(
        "{} success, rankId={:d}, deviceId={:d}, ip={}, port={:d}, identifier={}", __func__, deviceId, deviceId, hostIp,
        kDefaultRootPort, identifier);
    return HCCL_SUCCESS;
}

} // namespace

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
extern HcclResult HcclGetRootInfo(HcclRootInfo* rootInfo);
extern HcclResult HcclCommInitRootInfo(uint32_t nRanks, const HcclRootInfo* rootInfo, uint32_t rank, HcclComm* comm);
extern HcclResult HcclCommInitRootInfoConfig(
    uint32_t nRanks, const HcclRootInfo* rootInfo, uint32_t rank, const HcclCommConfig* config, HcclComm* comm);
extern HcclResult HcclCommInitClusterInfo(const char* clusterInfo, uint32_t rank, HcclComm* comm);
extern HcclResult
HcclCommInitClusterInfoConfig(const char* clusterInfo, uint32_t rank, HcclCommConfig* config, HcclComm* comm);
extern HcclResult HcclCommDestroy(HcclComm comm);

void SimBindHcclCommMember(HcclComm comm, uint64_t communicatorId)
{
    // 真 HCCL 句柄创建成功后, 记录它与当前 rank 本地表行的对应关系.
    if (comm != nullptr && communicatorId != 0) {
        g_hcclCommToCommunicatorId[comm] = communicatorId;
    }
}

bool SimFindHcclCommMember(HcclComm comm, uint64_t* communicatorId)
{
    const auto it = g_hcclCommToCommunicatorId.find(comm);
    if (it == g_hcclCommToCommunicatorId.end() || communicatorId == nullptr) {
        return false;
    }
    *communicatorId = it->second;
    return true;
}

bool SimFindHcclCommHandle(uint64_t communicatorId, HcclComm* comm)
{
    if (communicatorId == 0 || comm == nullptr) {
        return false;
    }
    for (const auto& entry : g_hcclCommToCommunicatorId) {
        if (entry.second == communicatorId) {
            *comm = entry.first;
            return true;
        }
    }
    return false;
}

void SimimSetThreadName(const std::string& threadStr)
{
    // 线程名应限制在15个字符内，防止被截断
    s32 sRet = pthread_setname_np(pthread_self(), threadStr.c_str());
    if (sRet != 0) {
        HCCL_VM_WARN("err[{}] link[{}] threadNameSet failed.", sRet, threadStr.c_str());
    }
}

/**
 * @brief 销毁HCCL通信域，释放通信域相关资源。
 *
 * level0 模式下阻塞等待同通信域所有卡都调用到本接口（快慢卡销毁同步），
 * 再透传真实 HCCL 执行销毁；未建模句柄直接透传。
 * 同一通信域多次调用Destroy不会报错（幂等）。
 *
 * @param comm 输入：待销毁的通信域句柄，由HcclCommInitXxx系列接口返回。
 *
 * @return HcclResult 接口成功返回HCCL_SUCCESS，其他失败。
 * @retval HCCL_SUCCESS 销毁成功
 * @retval HCCL_E_PTR comm为空
 * @retval HCCL_E_NOT_SUPPORT 未找到真实HcclCommDestroy接口
 * @retval HCCL_E_INTERNAL 销毁同步等待超时等内部错误
 */
HcclResult HcclCommDestroy(HcclComm comm)
{
    if (IsLevel1Mode()) {
        return DestroyLevel1Communicator(comm);
    }
    using DestroyFunc = HcclResult (*)(HcclComm);
    const auto destroy = reinterpret_cast<DestroyFunc>(DlsymRealHccl("HcclCommDestroy"));
    if (destroy == nullptr) {
        return HCCL_E_NOT_SUPPORT;
    }
    // 快慢卡同步: 等同通信域所有卡到达本接口后再执行真正销毁.
    uint64_t communicatorId = 0;
    if (SimFindHcclCommMember(comm, &communicatorId)) {
        if (!sim::runtime::MarkAndSyncCommunicatorDestroy(communicatorId)) {
            HCCL_VM_ERROR("communicator destroy sync failed, commId={}", communicatorId);
            return HCCL_E_INTERNAL;
        }
    }
    const HcclResult ret = destroy(comm);
    if (ret == HCCL_SUCCESS) {
        // 句柄失效后只清除进程内索引, 数据库的建模记录保留给 Checker 后续查询.
        g_hcclCommToCommunicatorId.erase(comm);
    }
    return ret;
}

/**
 * @brief 获取root节点的rank标识信息（HcclRootInfo）。
 *
 * 此接口需要在HCCL初始化接口（HcclCommInitRootInfo或HcclCommInitRootInfoConfig）前调用，
 * 仅需在root节点调用，用于生成root节点的rank标识信息。返回的HcclRootInfo内部包含
 * host IP、监听端口、设备物理ID、时间戳和rank ID等信息，需广播至集群内所有rank。
 *
 * @param rootInfo 输出：本rank的标识信息，主要包含device ip、device id等信息。
 *                 此信息需广播至集群内所有rank用来进行HCCL初始化。
 *                 HcclRootInfo类型的定义可参见HcclRootInfo。
 *
 * @return HcclResult 接口成功返回HCCL_SUCCESS，其他失败。
 * @retval HCCL_SUCCESS 成功获取rootInfo
 * @retval HCCL_E_PTR rootInfo指针为空
 * @retval HCCL_E_INTERNAL 内部错误（无法获取rankId或rank table信息）
 */
HcclResult HcclGetRootInfo(HcclRootInfo* rootInfo)
{
    if (IsLevel1Mode()) {
        return GetLevel1RootInfo(rootInfo);
    }
    (void)rootInfo;
    return HCCL_SUCCESS;
}

/**
 * @brief 基于rank table初始化HCCL，创建HCCL通信域。
 *
 * Rank table文件是一个JSON格式的文件，配置了参与集合通信的NPU资源信息，
 * 关于rank table文件的配置可参见集群信息配置。该接口会解析rank table并
 * 在数据库中创建通信域记录，用于后续的集合通信操作。
 *
 * @param clusterInfo 输入：Rank table的内容或文件路径。如果以'{'开头，则视为JSON内容字符串；
 *                    否则视为文件路径。作为字符串最大长度为4096字节，含结束符。
 * @param rank 输入：本rank的id。需要注意，此参数取值需要与rank table中对应的"rank_id"字段取值一致。
 * @param comm 输出：将初始化后的通信域以指针的信息回传给调用者。
 *                  HcclComm类型的定义可参见HcclComm。
 */
HcclResult HcclCommInitClusterInfo(const char* clusterInfo, uint32_t rank, HcclComm* comm)
{
    if (IsLevel1Mode()) {
        return CreateLevel1Communicator(clusterInfo, rank, nullptr, comm);
    }
    using InitFunc = HcclResult (*)(const char*, uint32_t, HcclComm*);
    const auto init = reinterpret_cast<InitFunc>(DlsymRealHccl("HcclCommInitClusterInfo"));
    if (init == nullptr) {
        return HCCL_E_NOT_SUPPORT;
    }
    const HcclResult ret = init(clusterInfo, rank, comm);
    if (ret != HCCL_SUCCESS || comm == nullptr || *comm == nullptr) {
        return ret;
    }

    // 真函数完成主域创建后, 再为当前 rank 建模并关联返回的 HcclComm.
    uint64_t communicatorId = 0;
    if (BuildWorldMember(*comm, communicatorId)) {
        SimBindHcclCommMember(*comm, communicatorId);
    } else {
        HCCL_VM_ERROR("BuildWorldMember fail.");
        return HCCL_E_INTERNAL;
    }
    return ret;
}

/**
 * @brief 基于rank table初始化HCCL，创建具有特定配置的HCCL通信域。
 *
 * 与HcclCommInitClusterInfo类似，但允许用户通过HcclCommConfig指定通信域的配置参数，
 * 包括缓存区大小、确定性计算开关、通信域名称、算子展开模式等。
 *
 * @param clusterInfo 输入：Rank table的内容或文件路径。如果以'{'开头，则视为JSON内容字符串；
 *                    否则视为文件路径。作为字符串最大长度为4096字节，含结束符。
 * @param rank 输入：本rank的id，需要与rank table中对应的"rank_id"字段取值一致。
 * @param config 输入：通信域配置信息，包含缓存区大小、确定性计算开关、通信域名称、
 *                    算子展开模式、RDMA traffic class、RDMA service level等配置。
 * @param comm 输出：将初始化后的通信域以指针的信息回传给调用者。
 *
 * @return HcclResult 接口成功返回HCCL_SUCCESS，其他失败。
 * @retval HCCL_SUCCESS 初始化成功
 * @retval HCCL_E_PTR 指针参数为空
 * @retval HCCL_E_INTERNAL 内部错误或重复初始化
 * @retval HCCL_E_OPEN_FILE_FAILURE 打开rank table文件失败
 * @retval HCCL_E_PARA 参数错误（rank超出范围）
 *
 * @note 同一通信域不支持重复初始化。
 */
HcclResult HcclCommInitClusterInfoConfig(const char* clusterInfo, uint32_t rank, HcclCommConfig* config, HcclComm* comm)
{
    if (IsLevel1Mode()) {
        if (config == nullptr) {
            HCCL_VM_ERROR("{}: config is nullptr", __func__);
            return HCCL_E_PTR;
        }
        return CreateLevel1Communicator(clusterInfo, rank, config, comm);
    }
    using InitFunc = HcclResult (*)(const char*, uint32_t, HcclCommConfig*, HcclComm*);
    const auto init = reinterpret_cast<InitFunc>(DlsymRealHccl("HcclCommInitClusterInfoConfig"));
    if (init == nullptr) {
        return HCCL_E_NOT_SUPPORT;
    }
    const HcclResult ret = init(clusterInfo, rank, config, comm);
    if (ret != HCCL_SUCCESS || comm == nullptr || *comm == nullptr) {
        return ret;
    }

    // Config 入口与普通入口保持同一建模规则, 避免产生两套主域表结构.
    uint64_t communicatorId = 0;
    if (BuildWorldMember(*comm, communicatorId)) {
        SimBindHcclCommMember(*comm, communicatorId);
    } else {
        HCCL_VM_ERROR("BuildWorldMember fail.");
        return HCCL_E_INTERNAL;
    }
    return ret;
}

/**
 * @brief 基于已有的通信域创建子通信域（带配置）。
 * @note 当前平台不支持子通信域功能，level1始终返回HCCL_E_NOT_SUPPORT；level2由真实HCCL实现处理。
 */
HcclResult HcclCreateSubCommConfig(
    HcclComm* comm, uint32_t rankNum, uint32_t* rankIds, uint64_t subCommId, uint32_t subCommRankId,
    HcclCommConfig* config, HcclComm* subComm)
{
    if (IsLevel1Mode()) {
        (void)comm;
        (void)rankNum;
        (void)rankIds;
        (void)subCommId;
        (void)subCommRankId;
        (void)config;
        (void)subComm;
        return HCCL_E_NOT_SUPPORT;
    }
    using CreateSubCommFunc
        = HcclResult (*)(HcclComm*, uint32_t, uint32_t*, uint64_t, uint32_t, HcclCommConfig*, HcclComm*);
    const auto create = reinterpret_cast<CreateSubCommFunc>(DlsymRealHccl("HcclCreateSubCommConfig"));
    if (create == nullptr) {
        return HCCL_E_NOT_SUPPORT;
    }
    uint64_t communicatorId = 0;
    // 不参与子域的 rank 仍交给真函数处理, 但不为其插入子域 Communicator 表行.
    const bool participates = rankIds != nullptr && subCommId != kInvalidSubCommId;
    std::vector<uint64_t> deviceIds;
    if (participates) {
        if (comm == nullptr || *comm == nullptr || rankNum == 0) {
            return HCCL_E_PARA;
        }
        uint64_t parentCommunicatorId = 0;
        if (!SimFindHcclCommMember(*comm, &parentCommunicatorId)) {
            return HCCL_E_INTERNAL;
        }
        std::vector<sim::runtime::CommunicatorMemberInfo> parentMembers;
        if (!sim::runtime::GetCommunicatorMembers(parentCommunicatorId, parentMembers)) {
            return HCCL_E_INTERNAL;
        }
        deviceIds.resize(rankNum, 0);
        for (uint32_t index = 0; index < rankNum; ++index) {
            const auto member = std::find_if(
                parentMembers.begin(), parentMembers.end(),
                [parentRankId = rankIds[index]](const sim::runtime::CommunicatorMemberInfo& info) {
                    return info.rankId == parentRankId;
                });
            if (member == parentMembers.end()) {
                return HCCL_E_INTERNAL;
            }
            deviceIds[index] = member->deviceId;
        }
        if (subCommRankId >= rankNum || deviceIds[subCommRankId] != g_cur_device_key) {
            return HCCL_E_INTERNAL;
        }
    }
    const HcclResult ret = create(comm, rankNum, rankIds, subCommId, subCommRankId, config, subComm);
    if (ret != HCCL_SUCCESS || !participates || subComm == nullptr || *subComm == nullptr) {
        return ret;
    }

    if (!BuildSubCommMember(*subComm, deviceIds, rankNum, communicatorId)) {
        HCCL_VM_ERROR("BuildSubCommMember fail, subCommId={}, subCommRankId={}", subCommId, subCommRankId);
        (void)HcclCommDestroy(*subComm);
        *subComm = nullptr;
        return HCCL_E_INTERNAL;
    }
    // 真实子域名称的所有成员已就绪后，才让算子路径使用该子域句柄。
    SimBindHcclCommMember(*subComm, communicatorId);
    return ret;
}

/**
 * @brief 基于rootInfo初始化HCCL，创建HCCL通信域。
 *
 * Rank table文件路径从环境变量RANK_TABLE_FILE中读取。rootInfo参数在本桩函数中不使用，
 * 实际初始化逻辑与HcclCommInitClusterInfo一致，通过解析环境变量中指定的rank table文件创建通信域。
 *
 * @param nRanks 输入：通信域的rank总数，本桩函数中不使用。
 * @param rootInfo 输入：root节点的rank标识信息，本桩函数中不使用。
 * @param rank 输入：本rank的id，需要与rank table中对应的"rank_id"字段取值一致。
 * @param comm 输出：将初始化后的通信域句柄回传给调用者。
 *
 * @return HcclResult 接口成功返回HCCL_SUCCESS，其他失败。
 * @retval HCCL_SUCCESS 初始化成功
 * @retval HCCL_E_PTR comm指针为空
 * @retval HCCL_E_INTERNAL 内部错误、重复初始化或未设置RANK_TABLE_FILE环境变量
 * @retval HCCL_E_PARA 参数错误（rank超出范围）
 * @retval HCCL_E_MEMORY 内存分配失败
 */
HcclResult HcclCommInitRootInfo(uint32_t nRanks, const HcclRootInfo* rootInfo, uint32_t rank, HcclComm* comm)
{
    (void)nRanks;
    (void)rootInfo;
    // 解析ranktable文件，初始化通信域相关数据库表项
    const char* clusterInfo = std::getenv("RANK_TABLE_FILE");
    if (!clusterInfo) {
        HCCL_VM_ERROR("RANK_TABLE_FILE env not set, please check your config.");
        return HcclResult::HCCL_E_INTERNAL;
    }

    return HcclCommInitClusterInfo(clusterInfo, rank, comm);
}

/**
 * @brief 基于rootInfo初始化HCCL，创建具有特定配置的HCCL通信域。
 *
 * Rank table文件路径从环境变量RANK_TABLE_FILE中读取。rootInfo参数在本桩函数中不使用，
 * 实际初始化逻辑与HcclCommInitClusterInfoConfig一致，并使用HcclCommConfig指定通信域配置。
 *
 * @param nRanks 输入：通信域的rank总数，本桩函数中不使用。
 * @param rootInfo 输入：root节点的rank标识信息，本桩函数中不使用。
 * @param rank 输入：本rank的id，需要与rank table中对应的"rank_id"字段取值一致。
 * @param config 输入：通信域配置信息。
 * @param comm 输出：将初始化后的通信域句柄回传给调用者。
 *
 * @return HcclResult 接口成功返回HCCL_SUCCESS，其他失败。
 * @retval HCCL_SUCCESS 初始化成功
 * @retval HCCL_E_PTR config或comm指针为空
 * @retval HCCL_E_INTERNAL 内部错误、重复初始化或未设置RANK_TABLE_FILE环境变量
 * @retval HCCL_E_PARA 参数错误（rank超出范围）
 * @retval HCCL_E_MEMORY 内存分配失败
 *
 * @note 同一通信域不支持重复初始化。
 */
HcclResult HcclCommInitRootInfoConfig(
    uint32_t nRanks, const HcclRootInfo* rootInfo, uint32_t rank, const HcclCommConfig* config, HcclComm* comm)
{
    (void)nRanks;
    (void)rootInfo;
    const char* clusterInfo = getenv("RANK_TABLE_FILE");
    HcclCommConfig* cfg = const_cast<HcclCommConfig*>(config);
    return HcclCommInitClusterInfoConfig(clusterInfo, rank, cfg, comm);
}

HcclResult SimGetDeviceComm(uint32_t ndev, const uint32_t rank, const uint32_t logicDeviceId, HcclComm& comm)
{
    HCCL_VM_INFO("rank[{}] Get device comm...", rank);
    // 给当前线程添加名字
    SimimSetThreadName("Hccl_GetDevComm");

    if (aclrtSetDevice(logicDeviceId) != ACL_SUCCESS) {
        HCCL_VM_ERROR("set fail logicDeviceId[{}].", logicDeviceId);
        return HcclResult::HCCL_E_INTERNAL;
    }

    const char* clusterInfo = std::getenv("RANK_TABLE_FILE");
    if (!clusterInfo) {
        HCCL_VM_ERROR("RANK_TABLE_FILE env not set, please check your config.");
        return HcclResult::HCCL_E_INTERNAL;
    }

    auto ret = HcclCommInitClusterInfo(clusterInfo, rank, &comm);
    if (ret != HCCL_SUCCESS || comm == nullptr) {
        comm = nullptr;
        HCCL_VM_ERROR("rank[{}] Get device comm failed!", rank);
        if (aclrtResetDevice(logicDeviceId) != ACL_SUCCESS) {
            HCCL_VM_ERROR("reset fail logicDeviceId[{}].", logicDeviceId);
            return ret;
        }
        return ret;
    }
    return HCCL_SUCCESS;
}

/**
 * @brief 单机多卡场景下，通过一个进程统一创建多张卡的通信域。
 *
 * 单机通信场景中，每个device在通信域表中创建一条DB记录。devices[0]作为root rank，
 * 该接口为每张卡分别分配通信域，并存储在comms输出数组中。
 *
 * @param ndev 输入：通信域数量，即device的数量。
 * @param devices 输入：device ID数组，数组长度为ndev。
 * @param comms 输出：通信域句柄数组，数组长度为ndev，每个元素对应一张卡的通信域。
 *
 * @return HcclResult 接口成功返回HCCL_SUCCESS，其他失败。
 * @retval HCCL_SUCCESS 全部通信域初始化成功
 * @retval HCCL_E_PTR devices或comms指针为空
 * @retval HCCL_E_PARA ndev为0
 * @retval HCCL_E_INTERNAL DB写入失败
 * @retval HCCL_E_MEMORY 内存分配失败
 */
HcclResult HcclCommInitAll(uint32_t ndev, int32_t* devices, HcclComm* comms)
{
    if (IsLevel1Mode()) {
        return CreateLevel1CommAll(ndev, devices, comms);
    }
    HCCL_VM_INFO("Init all comm...");
    SimimSetThreadName("Hccl_GetCommAll");

    if (aclrtSetDevice(devices[0]) != ACL_SUCCESS) {
        HCCL_VM_ERROR("set fail devices[0][{}].", devices[0]);
        return HcclResult::HCCL_E_INTERNAL;
    }

    // 获取通信域之前, 先把所有通信域设置为空
    for (uint32_t i = 0; i < ndev; i++) {
        comms[i] = nullptr;
    }

    std::vector<std::unique_ptr<std::thread>> threads(ndev);
    for (uint32_t rankId = 0; rankId < ndev; rankId++) {
        threads[rankId].reset(
            new (std::nothrow) std::thread(&SimGetDeviceComm, ndev, rankId, devices[rankId], std::ref(comms[rankId])));
        if (!threads[rankId]) {
            HCCL_VM_ERROR("threads[{}] start failed ", rankId);
            return HcclResult::HCCL_E_INTERNAL;
        }
    }
    for (uint32_t i = 0; i < ndev; i++) {
        threads[i]->join();
    }

    // 如果任何一个通信域初始化失败，将所有已经成功创建的通信域销毁
    bool isFailed = false;
    for (uint32_t i = 0; i < ndev; ++i) {
        if (comms[i] == nullptr) {
            HCCL_VM_ERROR("rank[{}] get comm failed!", i);
            isFailed = true;
            break;
        }
    }
    if (isFailed) {
        for (uint32_t i = 0; i < ndev; ++i) {
            if (comms[i] != nullptr) {
                (void)HcclCommDestroy(comms[i]);
            }
        }
        return HCCL_E_INTERNAL;
    }

    if (aclrtResetDevice(devices[0]) != ACL_SUCCESS) {
        HCCL_VM_ERROR("reset fail devices[0][{}].", devices[0]);
        return HcclResult::HCCL_E_INTERNAL;
    }

    return HCCL_SUCCESS;
}

#ifdef __cplusplus
}
#endif // __cplusplus
