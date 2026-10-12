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
 * Description: 控制面基础资源管理打桩函数（北向劫持）
 *              - HcommThreadAlloc / HcommThreadFree
 *                线程资源管理接口，与 HcclThreadAcquire 的区别在于它们不绑定任何通信域。
 *              - HcommChannelDescInit / HcommChannelCreate / HcommChannelGetStatus /
 *                HcommChannelGetNotifyNum / HcommChannelDestroy
 *                通道资源管理接口，与 HcclChannel* 的区别同样在于它们不绑定通信域。
 * Create: 2026-07-15
 */

#define HCCL_VM_MODULE "BR_STUB"

#include <arpa/inet.h>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <type_traits>
#include <vector>

// 正式类型头内含 static inline HcommChannelDescInit，而本文件保留了同名外部占位桩。
// 仅在包含头文件期间重命名该 inline 函数，避免同一翻译单元重定义，随后立即恢复宏环境。
#define HcommChannelDescInit HcommChannelDescInitHeaderInline
#include "hcomm/hcomm_res_defs.h"
#undef HcommChannelDescInit
#include "acl/acl_rt.h"
#include "hccl_proxy_common.h"
#include "level1_proxy_common.h"
#include "sim_log.h"
#include "runtime_state/db_sim_runner_common.h"
#include "runtime_state/db_sim_runner_ops.h"

extern uint64_t g_cur_server_key;

namespace {

// 数据库字段按公开ABI的原始数组保存；版本升级导致长度变化时必须在编译期报错。
static_assert(
    sizeof(CommAddr::raws) == sizeof(sim::runtime::HcommEndpoint::addr), "HcommEndpoint.addr must match CommAddr.raws");
static_assert(
    sizeof(EndpointLoc::raws) == sizeof(sim::runtime::HcommEndpoint::loc),
    "HcommEndpoint.loc must match EndpointLoc.raws");
static_assert(
    sizeof(EndpointDesc::raws) == sizeof(sim::runtime::HcommEndpoint::extension),
    "HcommEndpoint.extension must match EndpointDesc.raws");

// 北向 Endpoint 与南向 sim::runtime::EndPoint 的解析结果，用于统一映射接口错误码。
enum class SouthEndpointResolveResult {
    SUCCESS = 0,           // 解析成功，包括已绑定、未解析和不适用三种状态
    DEVICE_NOT_FOUND,      // 北向物理设备在当前 Server 中不存在
    ENDPOINT_NOT_FOUND,    // 必须绑定南向 Endpoint，但没有地址匹配项
    ENDPOINT_AMBIGUOUS,    // 匹配到多条南向 Endpoint，无法唯一绑定
    ADDRESS_NOT_SUPPORTED, // 当前地址类型无法用于必需的南向绑定
};

// 桩内存描述符解析结果，用于区分长度、版本和一致性校验失败。
enum class HcommMemDescParseResult {
    SUCCESS = 0,
    INVALID_LENGTH,
    INVALID_MAGIC,
    INVALID_VERSION,
    INVALID_KEY,
};

// HcommMemExport/Import之间交换的固定版本描述符，不暴露存储层的BLOB布局。
struct HcommMemDescriptor {
    uint32_t magic;
    uint32_t version;
    uint32_t length;
    uint32_t reserved;
    uint64_t source_endpoint_id;
    uint64_t source_south_endpoint_id;
    uint64_t source_mem_id;
    uint64_t descriptor_key;
    uint64_t addr;
    uint64_t size;
    int32_t mem_type;
    uint32_t reserved_tail;
};

static_assert(std::is_trivially_copyable<HcommMemDescriptor>::value, "HcommMemDescriptor must be trivially copyable");
static_assert(sizeof(uintptr_t) <= sizeof(uint64_t), "database address field cannot hold uintptr_t");

constexpr uint64_t HCOMM_BASIC_THREAD_COMM_ID = 0u;
constexpr uint64_t HCOMM_BASIC_COMM_ID = 0u;
constexpr uint64_t HCOMM_BASIC_REMOTE_RANK_ID = 0u;

constexpr uint32_t HCOMM_BASIC_MAX_THREAD_NUM = 40u;
constexpr uint64_t HCOMM_BASIC_MAX_NOTIFY_TOTAL = 640u;
constexpr uint32_t HCOMM_BASIC_MAX_CHANNEL_NUM = 4096u;
constexpr uint64_t HCOMM_BASIC_CCU_DEFAULT_NOTIFY = 8u;
constexpr uint32_t HCOMM_BASIC_BASE_PORT = 30000u;

constexpr uint32_t HCOMM_MEM_DESC_MAGIC = 0x48434D4Du; // ASCII "HCMM"
constexpr uint32_t HCOMM_MEM_DESC_VERSION = 1u;
constexpr uint64_t FNV1A_OFFSET_BASIS = 14695981039346656037ull;
constexpr uint64_t FNV1A_PRIME = 1099511628211ull;

// 描述符返回缓冲按本地内存主键保存；它只管理指针生命周期，资源真值仍在存储层。
std::map<uint64_t, HcommMemDescriptor> g_hcommMemDescBuffers;
// 保护进程内描述符缓冲容器；返回指针的生命周期仍由注销或销毁操作界定。
std::mutex g_hcommMemDescBuffersMutex;

// 检查协议是否落在当前公开 CommProtocol 的有效枚举范围内。
bool IsValidProtocol(CommProtocol protocol)
{
    return protocol >= COMM_PROTOCOL_HCCS && protocol <= COMM_PROTOCOL_HCCS_ONLY;
}

// 检查地址类型是否为 IPv4、IPv6、ID 或 EID。
bool IsValidAddressType(CommAddrType addrType)
{
    return addrType >= COMM_ADDR_TYPE_IP_V4 && addrType <= COMM_ADDR_TYPE_EID;
}

// 判断协议是否依赖南向网络 Endpoint，依赖时不允许无绑定创建。
bool RequiresSouthEndpoint(CommProtocol protocol)
{
    return protocol == COMM_PROTOCOL_ROCE || protocol == COMM_PROTOCOL_UBC_CTP || protocol == COMM_PROTOCOL_UBC_TP
           || protocol == COMM_PROTOCOL_UBOE;
}

// 判断协议是否可能使用南向 Endpoint，用于区分 UNRESOLVED 和 NOT_APPLICABLE。
bool MayUseSouthEndpoint(CommProtocol protocol)
{
    return RequiresSouthEndpoint(protocol) || protocol == COMM_PROTOCOL_HCCS || protocol == COMM_PROTOCOL_HCCS_ONLY
           || protocol == COMM_PROTOCOL_SIO;
}

// 按北向地址类型比较 EID 或规范化后的 IPv4/IPv6 二进制地址。
bool IsSouthEndpointAddressMatch(const EndpointDesc& endpoint, const sim::runtime::EndPoint& southEndpoint)
{
    if (endpoint.commAddr.type == COMM_ADDR_TYPE_EID) {
        return std::memcmp(endpoint.commAddr.eid, southEndpoint.eid, COMM_ADDR_EID_LEN) == 0;
    }

    if (endpoint.commAddr.type == COMM_ADDR_TYPE_IP_V4) {
        struct in_addr southAddr {};
        return inet_pton(AF_INET, southEndpoint.ip_addr, &southAddr) == 1
               && std::memcmp(&endpoint.commAddr.addr, &southAddr, sizeof(southAddr)) == 0;
    }

    if (endpoint.commAddr.type == COMM_ADDR_TYPE_IP_V6) {
        struct in6_addr southAddr {};
        return inet_pton(AF_INET6, southEndpoint.ip_addr, &southAddr) == 1
               && std::memcmp(&endpoint.commAddr.addr6, &southAddr, sizeof(southAddr)) == 0;
    }

    return false;
}

// 在北向 Endpoint 所属设备内解析唯一南向 Endpoint，并写入关联ID和绑定状态。
SouthEndpointResolveResult ResolveSouthEndpoint(const EndpointDesc& endpoint, sim::runtime::HcommEndpoint& record)
{
    record.south_endpoint_id = 0u;

    // HOST Endpoint 没有对应的南向 sim::runtime::EndPoint，保留北向记录即可。
    if (endpoint.loc.locType == ENDPOINT_LOC_TYPE_HOST) {
        record.bind_state = sim::runtime::HCOMM_ENDPOINT_NOT_APPLICABLE;
        return SouthEndpointResolveResult::SUCCESS;
    }

    sim::runtime::Device device{};
    if (sim::runtime::GetDeviceByPhysicalId(endpoint.loc.device.devPhyId, device) != ACL_SUCCESS) {
        return SouthEndpointResolveResult::DEVICE_NOT_FOUND;
    }

    // ID 地址目前不能和南向 EndPoint 的 EID/IP 字段建立可靠映射。
    if (endpoint.commAddr.type == COMM_ADDR_TYPE_ID) {
        if (RequiresSouthEndpoint(endpoint.protocol)) {
            return SouthEndpointResolveResult::ADDRESS_NOT_SUPPORTED;
        }
        record.bind_state = MayUseSouthEndpoint(endpoint.protocol) ? sim::runtime::HCOMM_ENDPOINT_UNRESOLVED :
                                                                     sim::runtime::HCOMM_ENDPOINT_NOT_APPLICABLE;
        return SouthEndpointResolveResult::SUCCESS;
    }

    auto dbMatches = sim::runtime::Db::GetByPred<sim::runtime::EndPoint>(
        HcclSim::Storage::Eq(&sim::runtime::EndPoint::device_id, device.id));
    std::vector<sim::runtime::EndPoint> matches;
    if (dbMatches.ok()) {
        for (const auto& southEndpoint : *dbMatches) {
            if (IsSouthEndpointAddressMatch(endpoint, southEndpoint)) {
                matches.push_back(southEndpoint);
            }
        }
    }

    if (matches.size() == 1u) {
        record.south_endpoint_id = matches[0].id;
        record.bind_state = sim::runtime::HCOMM_ENDPOINT_BOUND;
        return SouthEndpointResolveResult::SUCCESS;
    }
    if (matches.size() > 1u) {
        return SouthEndpointResolveResult::ENDPOINT_AMBIGUOUS;
    }
    if (RequiresSouthEndpoint(endpoint.protocol)) {
        return SouthEndpointResolveResult::ENDPOINT_NOT_FOUND;
    }

    record.bind_state = MayUseSouthEndpoint(endpoint.protocol) ? sim::runtime::HCOMM_ENDPOINT_UNRESOLVED :
                                                                 sim::runtime::HCOMM_ENDPOINT_NOT_APPLICABLE;
    return SouthEndpointResolveResult::SUCCESS;
}

// 将void*不透明句柄还原为存储层主键；句柄只编码ID，不能解引用。
uint64_t DecodeOpaqueHandle(const void* handle) { return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(handle)); }

// 查询当前Runner，Runner未初始化属于桩内部状态错误。
HcommResult GetCurrentRunner(sim::runtime::Runner& runner)
{
    if (!sim::runtime::GetCurrRunnerTls(g_cur_server_key, runner) || runner.id == 0u) {
        return static_cast<HcommResult>(HCCL_E_INTERNAL);
    }
    return static_cast<HcommResult>(HCCL_SUCCESS);
}

// 查询当前Runner拥有的北向Endpoint，阻止跨进程或伪造句柄访问资源。
HcommResult GetOwnedEndpoint(
    EndpointHandle endpointHandle, sim::runtime::Runner& runner, uint64_t& endpointId,
    sim::runtime::HcommEndpoint& endpoint)
{
    endpointId = DecodeOpaqueHandle(endpointHandle);
    HcommResult result = GetCurrentRunner(runner);
    if (result != HCCL_SUCCESS) {
        return result;
    }

    auto endpointRecord = sim::runtime::Db::GetById<sim::runtime::HcommEndpoint>(endpointId);
    if (!endpointRecord.ok() || endpointRecord->runner_id != runner.id) {
        return static_cast<HcommResult>(HCCL_E_NOT_FOUND);
    }
    endpoint = *endpointRecord;
    return static_cast<HcommResult>(HCCL_SUCCESS);
}

// 查询属于指定Endpoint的本地注册内存，导入记录不能作为HcommMemHandle使用。
HcommResult
GetLocalMemory(HcommMemHandle memHandle, uint64_t endpointId, uint64_t runnerId, sim::runtime::HcommMemReg& memory)
{
    uint64_t memId = DecodeOpaqueHandle(memHandle);
    auto memoryRecord = sim::runtime::Db::GetById<sim::runtime::HcommMemReg>(memId);
    if (!memoryRecord.ok() || memoryRecord->endpoint_id != endpointId || memoryRecord->runner_id != runnerId
        || memoryRecord->record_kind != sim::runtime::HCOMM_MEM_LOCAL_REGISTERED) {
        return static_cast<HcommResult>(HCCL_E_NOT_FOUND);
    }
    memory = *memoryRecord;
    return static_cast<HcommResult>(HCCL_SUCCESS);
}

// 检查公开内存类型，当前只接受DEVICE和HOST。
bool IsValidMemType(CommMemType memType) { return memType == COMM_MEM_TYPE_DEVICE || memType == COMM_MEM_TYPE_HOST; }

// 对描述符关键字段计算稳定FNV-1a校验键，不依赖结构体填充字节。
uint64_t CalculateDescriptorKey(
    uint64_t sourceEndpointId, uint64_t sourceSouthEndpointId, uint64_t sourceMemId, uint64_t addr, uint64_t size,
    int32_t memType)
{
    uint64_t hash = FNV1A_OFFSET_BASIS;
    auto hashBytes = [&hash](const void* data, size_t length) {
        const auto* bytes = static_cast<const uint8_t*>(data);
        for (size_t index = 0; index < length; ++index) {
            hash ^= bytes[index];
            hash *= FNV1A_PRIME;
        }
    };
    hashBytes(&sourceEndpointId, sizeof(sourceEndpointId));
    hashBytes(&sourceSouthEndpointId, sizeof(sourceSouthEndpointId));
    hashBytes(&sourceMemId, sizeof(sourceMemId));
    hashBytes(&addr, sizeof(addr));
    hashBytes(&size, sizeof(size));
    hashBytes(&memType, sizeof(memType));
    return hash;
}

// 从数据库记录构造导出描述符，输出缓冲由g_hcommMemDescBuffers稳定持有。
HcommMemDescriptor
BuildMemDescriptor(const sim::runtime::HcommEndpoint& endpoint, const sim::runtime::HcommMemReg& memory)
{
    HcommMemDescriptor descriptor{};
    descriptor.magic = HCOMM_MEM_DESC_MAGIC;
    descriptor.version = HCOMM_MEM_DESC_VERSION;
    descriptor.length = sizeof(HcommMemDescriptor);
    descriptor.source_endpoint_id = endpoint.id;
    descriptor.source_south_endpoint_id = endpoint.south_endpoint_id;
    descriptor.source_mem_id = memory.id;
    descriptor.descriptor_key = memory.descriptor_key;
    descriptor.addr = memory.addr;
    descriptor.size = memory.size;
    descriptor.mem_type = memory.mem_type;
    return descriptor;
}

// 复制并校验调用方传入的描述符，避免未对齐访问和直接信任外部字节。
HcommMemDescParseResult ParseMemDescriptor(const void* memDesc, uint32_t descLen, HcommMemDescriptor& descriptor)
{
    if (descLen != sizeof(HcommMemDescriptor)) {
        return HcommMemDescParseResult::INVALID_LENGTH;
    }
    std::memcpy(&descriptor, memDesc, sizeof(descriptor));
    if (descriptor.magic != HCOMM_MEM_DESC_MAGIC) {
        return HcommMemDescParseResult::INVALID_MAGIC;
    }
    if (descriptor.version != HCOMM_MEM_DESC_VERSION || descriptor.length != sizeof(HcommMemDescriptor)) {
        return HcommMemDescParseResult::INVALID_VERSION;
    }
    uint64_t expectedKey = CalculateDescriptorKey(
        descriptor.source_endpoint_id, descriptor.source_south_endpoint_id, descriptor.source_mem_id, descriptor.addr,
        descriptor.size, descriptor.mem_type);
    if (descriptor.descriptor_key == 0u || descriptor.descriptor_key != expectedKey) {
        return HcommMemDescParseResult::INVALID_KEY;
    }
    return HcommMemDescParseResult::SUCCESS;
}

// 校验描述符引用的源Endpoint和本地内存仍存在且字段完全一致。
HcommResult ValidateDescriptorSource(const HcommMemDescriptor& descriptor)
{
    auto sourceEndpoint = sim::runtime::Db::GetById<sim::runtime::HcommEndpoint>(descriptor.source_endpoint_id);
    auto sourceMemory = sim::runtime::Db::GetById<sim::runtime::HcommMemReg>(descriptor.source_mem_id);
    if (!sourceEndpoint.ok() || !sourceMemory.ok()) {
        return static_cast<HcommResult>(HCCL_E_NOT_FOUND);
    }
    if (sourceEndpoint->south_endpoint_id != descriptor.source_south_endpoint_id
        || sourceMemory->record_kind != sim::runtime::HCOMM_MEM_LOCAL_REGISTERED
        || sourceMemory->endpoint_id != descriptor.source_endpoint_id
        || sourceMemory->descriptor_key != descriptor.descriptor_key || sourceMemory->addr != descriptor.addr
        || sourceMemory->size != descriptor.size || sourceMemory->mem_type != descriptor.mem_type) {
        return static_cast<HcommResult>(HCCL_E_PARA);
    }
    return static_cast<HcommResult>(HCCL_SUCCESS);
}

} // namespace

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 创建不绑定通信域的 HComm Endpoint。
 *
 * 本桩使用 sim::runtime::HcommEndpoint 保存完整的北向 EndpointDesc，并在可以唯一匹配时，
 * 通过 south_endpoint_id 只读关联南向拓扑 sim::runtime::EndPoint。北向句柄是数据库主键
 * 经 uintptr_t 编码后的不透明指针，只能用于后续桩接口查询，禁止解引用。
 *
 * 绑定规则：
 *   - HOST Endpoint 不关联南向 Endpoint，状态记为 NOT_APPLICABLE；
 *   - DEVICE Endpoint 先按 devPhyId 找到当前 Server 下的 sim::runtime::Device；
 *   - EID、IPv4、IPv6 在该设备下进行地址唯一匹配；
 *   - ROCE、UBC_CTP、UBC_TP、UBOE 必须唯一匹配，否则创建失败；
 *   - HCCS、HCCS_ONLY、SIO 无匹配时允许创建，状态记为 UNRESOLVED；
 *   - PCIE、UB_MEM 不依赖南向网络 Endpoint，无匹配时记为 NOT_APPLICABLE。
 *
 * @param endpoint 输入：待创建 Endpoint 的完整描述。
 * @param endpointHandle 输出：成功时返回 sim::runtime::HcommEndpoint 主键编码的句柄。
 * @return HcommResult 参数非法返回 HCCL_E_PARA；资源不存在返回 HCCL_E_NOT_FOUND；
 *         南向匹配不唯一或数据库写入失败返回 HCCL_E_INTERNAL。
 */
HcommResult HcommEndpointCreate(const EndpointDesc* endpoint, EndpointHandle* endpointHandle)
{
    if (endpoint == nullptr || endpointHandle == nullptr) {
        HCCL_VM_ERROR("{}: endpoint or endpointHandle is nullptr", __func__);
        return static_cast<HcommResult>(HCCL_E_PTR);
    }
    *endpointHandle = nullptr;

    if ((endpoint->loc.locType != ENDPOINT_LOC_TYPE_DEVICE && endpoint->loc.locType != ENDPOINT_LOC_TYPE_HOST)
        || !IsValidProtocol(endpoint->protocol) || !IsValidAddressType(endpoint->commAddr.type)) {
        HCCL_VM_ERROR(
            "{}: invalid parameter, locType={:d}, protocol={:d}, addrType={:d}", __func__,
            static_cast<int>(endpoint->loc.locType), static_cast<int>(endpoint->protocol),
            static_cast<int>(endpoint->commAddr.type));
        return static_cast<HcommResult>(HCCL_E_PARA);
    }

    // 2. 记录资源所属 Runner，防止后续跨进程误用句柄。
    sim::runtime::Runner runner{};
    HcommResult runnerResult = GetCurrentRunner(runner);
    if (runnerResult != HCCL_SUCCESS) {
        HCCL_VM_ERROR("{}: current runner is unavailable, serverKey={:d}", __func__, g_cur_server_key);
        return runnerResult;
    }

    // 3. 完整复制北向 EndpointDesc，避免依赖字段不完整的南向 EndPoint 还原。
    sim::runtime::HcommEndpoint record{};
    record.runner_id = runner.id;
    record.protocol = static_cast<int32_t>(endpoint->protocol);
    record.addr_type = static_cast<int32_t>(endpoint->commAddr.type);
    std::memcpy(record.addr, endpoint->commAddr.raws, sizeof(record.addr));
    record.loc_type = static_cast<int32_t>(endpoint->loc.locType);
    std::memcpy(record.loc, endpoint->loc.raws, sizeof(record.loc));
    std::memcpy(record.extension, endpoint->raws, sizeof(record.extension));

    // 4. 按位置、协议和地址尝试建立南向只读关联。
    SouthEndpointResolveResult resolveResult = ResolveSouthEndpoint(*endpoint, record);
    if (resolveResult == SouthEndpointResolveResult::DEVICE_NOT_FOUND
        || resolveResult == SouthEndpointResolveResult::ENDPOINT_NOT_FOUND) {
        HCCL_VM_ERROR(
            "{}: south resource not found, devPhyId={:d}, "
            "protocol={:d}, addrType={:d}, resolveResult={:d}",
            __func__, endpoint->loc.device.devPhyId, static_cast<int>(endpoint->protocol),
            static_cast<int>(endpoint->commAddr.type), static_cast<int>(resolveResult));
        return static_cast<HcommResult>(HCCL_E_NOT_FOUND);
    }
    if (resolveResult == SouthEndpointResolveResult::ADDRESS_NOT_SUPPORTED) {
        HCCL_VM_ERROR(
            "{}: addrType={:d} cannot resolve protocol={:d}, "
            "devPhyId={:d}, resolveResult={:d}",
            __func__, static_cast<int>(endpoint->commAddr.type), static_cast<int>(endpoint->protocol),
            endpoint->loc.device.devPhyId, static_cast<int>(resolveResult));
        return static_cast<HcommResult>(HCCL_E_NOT_SUPPORT);
    }
    if (resolveResult == SouthEndpointResolveResult::ENDPOINT_AMBIGUOUS) {
        HCCL_VM_ERROR(
            "{}: multiple south endpoints matched, devPhyId={:d}, "
            "protocol={:d}, addrType={:d}, resolveResult={:d}",
            __func__, endpoint->loc.device.devPhyId, static_cast<int>(endpoint->protocol),
            static_cast<int>(endpoint->commAddr.type), static_cast<int>(resolveResult));
        return static_cast<HcommResult>(HCCL_E_INTERNAL);
    }

    // 5. 仅在所有校验和关联均成功后写表，失败路径不会留下半成品记录。
    auto addResult = sim::runtime::Db::Add<sim::runtime::HcommEndpoint>(record);
    if (!addResult.ok()) {
        HCCL_VM_ERROR(
            "{}: failed to add HcommEndpoint record, runnerId={:d}, "
            "protocol={:d}, locType={:d}, bindState={:d}",
            __func__, record.runner_id, record.protocol, record.loc_type, static_cast<uint32_t>(record.bind_state));
        return static_cast<HcommResult>(HCCL_E_INTERNAL);
    }
    uint64_t endpointId = record.id;

    *endpointHandle = reinterpret_cast<EndpointHandle>(static_cast<uintptr_t>(endpointId));
    HCCL_VM_INFO(
        "{} success, endpointId={:d}, southEndpointId={:d}, "
        "protocol={:d}, addrType={:d}, locType={:d}, bindState={:d}",
        __func__, endpointId, record.south_endpoint_id, record.protocol, record.addr_type, record.loc_type,
        static_cast<uint32_t>(record.bind_state));
    return static_cast<HcommResult>(HCCL_SUCCESS);
}

/**
 * @brief 销毁北向HComm Endpoint及其所属内存记录。
 *
 * EndpointHandle编码的是sim::runtime::HcommEndpoint主键。本接口只删除北向Endpoint和
 * HcommMemReg子记录，不删除south_endpoint_id引用的南向拓扑sim::runtime::EndPoint。
 *
 * @param endpointHandle 输入：HcommEndpointCreate返回的Endpoint句柄。
 * @return 句柄不存在或不属于当前Runner返回HCCL_E_NOT_FOUND；删除失败返回
 *         HCCL_E_INTERNAL。
 */
HcommResult HcommEndpointDestroy(EndpointHandle endpointHandle)
{
    // 1. 校验Endpoint存在且属于当前Runner。
    sim::runtime::Runner runner{};
    sim::runtime::HcommEndpoint endpoint{};
    uint64_t endpointId = 0u;
    HcommResult result = GetOwnedEndpoint(endpointHandle, runner, endpointId, endpoint);
    if (result != HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "{}: endpoint not found or not owned, endpointId={:d}, "
            "result={:d}",
            __func__, endpointId, static_cast<int>(result));
        return result;
    }

    // 2. 查询全部子内存，统一清理本地注册和远端导入记录。
    auto memories = sim::runtime::Db::GetByPred<sim::runtime::HcommMemReg>(
        HcclSim::Storage::Eq(&sim::runtime::HcommMemReg::endpoint_id, endpointId));

    // 3. 查询并清理该 Endpoint 下创建的基础通道。
    auto channels = sim::runtime::Db::GetByPred<sim::runtime::HcclChannel>(
        HcclSim::Storage::Eq(&sim::runtime::HcclChannel::endpoint_id, endpointId));
    // 迁移前语义：查询失败按空集容错继续（不做解引用）。
    if (channels.ok()) {
        for (const auto& ch : *channels) {
            auto delResult = sim::runtime::Db::Delete<sim::runtime::HcclChannel>(
                HcclSim::Storage::Eq(&sim::runtime::HcclChannel::id, ch.id));
            if (!delResult.ok()) {
                HCCL_VM_ERROR(
                    "{}: failed to delete channel, endpointId={:d}, channelId={:d}", __func__, endpointId, ch.id);
                return static_cast<HcommResult>(HCCL_E_INTERNAL);
            }
        }
    }

    // 4. 存储层的BLOB字段没有数据库外键，必须手工执行级联删除。
    uint64_t deletedMemoryNum = 0;
    if (memories.ok()) {
        deletedMemoryNum = memories->size();
        for (const auto& memory : *memories) {
            auto delResult = sim::runtime::Db::Delete<sim::runtime::HcommMemReg>(
                HcclSim::Storage::Eq(&sim::runtime::HcommMemReg::id, memory.id));
            if (!delResult.ok()) {
                HCCL_VM_ERROR(
                    "{}: failed to delete memory, endpointId={:d}, memId={:d}, recordKind={:d}", __func__, endpointId,
                    memory.id, static_cast<uint32_t>(memory.record_kind));
                return static_cast<HcommResult>(HCCL_E_INTERNAL);
            }
            if (memory.record_kind == sim::runtime::HCOMM_MEM_LOCAL_REGISTERED) {
                std::lock_guard<std::mutex> lock(g_hcommMemDescBuffersMutex);
                g_hcommMemDescBuffers.erase(memory.id);
            }
        }
    }

    // 5. 最后删除父Endpoint；南向sim::runtime::EndPoint由拓扑管理，不能在此删除。
    auto delResult = sim::runtime::Db::Delete<sim::runtime::HcommEndpoint>(
        HcclSim::Storage::Eq(&sim::runtime::HcommEndpoint::id, endpointId));
    if (!delResult.ok()) {
        HCCL_VM_ERROR("{}: failed to delete endpoint, endpointId={:d}", __func__, endpointId);
        return static_cast<HcommResult>(HCCL_E_INTERNAL);
    }

    HCCL_VM_INFO(
        "{} success, endpointId={:d}, runnerId={:d}, deletedMemoryNum={:d}, southEndpointId={:d}", __func__, endpointId,
        runner.id, static_cast<uint32_t>(deletedMemoryNum), endpoint.south_endpoint_id);
    return static_cast<HcommResult>(HCCL_SUCCESS);
}

/**
 * @brief 在指定北向Endpoint上注册一段本地内存。
 *
 * 每次调用均创建独立的HcommMemReg本地记录，数据库主键编码为HcommMemHandle。
 * 相同地址和长度可以重复注册，并分别按句柄注销。
 */
HcommResult
HcommMemReg(EndpointHandle endpointHandle, const char* memTag, const CommMem* mem, HcommMemHandle* memHandle)
{
    if (memHandle == nullptr || mem == nullptr || memTag == nullptr || mem->addr == nullptr) {
        HCCL_VM_ERROR("{}: memHandle, mem, memTag or mem.addr is nullptr", __func__);
        return static_cast<HcommResult>(HCCL_E_PTR);
    }
    *memHandle = nullptr;
    if (mem->size == 0u || !IsValidMemType(mem->type)) {
        HCCL_VM_ERROR(
            "{}: invalid memory, addr={:p}, size={:d}, type={:d}", __func__, mem->addr, mem->size,
            static_cast<int>(mem->type));
        return static_cast<HcommResult>(HCCL_E_PARA);
    }

    sim::runtime::HcommMemReg memory{};
    size_t memTagLength = strnlen(memTag, sizeof(memory.mem_tag));
    if (memTagLength == sizeof(memory.mem_tag)) {
        HCCL_VM_ERROR("{}: memTag is too long, maxLength={:d}", __func__, sizeof(memory.mem_tag) - 1u);
        return static_cast<HcommResult>(HCCL_E_PARA);
    }

    // 2. 校验Endpoint归属，避免使用其他Runner或已经销毁的Endpoint。
    sim::runtime::Runner runner{};
    sim::runtime::HcommEndpoint endpoint{};
    uint64_t endpointId = 0u;
    HcommResult result = GetOwnedEndpoint(endpointHandle, runner, endpointId, endpoint);
    if (result != HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "{}: endpoint not found or not owned, endpointId={:d}, "
            "result={:d}",
            __func__, endpointId, static_cast<int>(result));
        return result;
    }

    // 3. 构造本地注册记录；地址仅按整数保存，桩不会访问或解引用该内存。
    memory.endpoint_id = endpointId;
    memory.runner_id = runner.id;
    memory.addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(mem->addr));
    memory.size = mem->size;
    memory.mem_type = static_cast<int32_t>(mem->type);
    memory.record_kind = sim::runtime::HCOMM_MEM_LOCAL_REGISTERED;
    std::memcpy(memory.mem_tag, memTag, memTagLength + 1u);

    // 4. 先插入取得内存主键，再生成包含主键的稳定描述符校验键。
    auto addResult = sim::runtime::Db::Add<sim::runtime::HcommMemReg>(memory);
    if (!addResult.ok()) {
        HCCL_VM_ERROR("{}: failed to add memory, endpointId={:d}", __func__, endpointId);
        return static_cast<HcommResult>(HCCL_E_INTERNAL);
    }
    uint64_t memId = memory.id;
    uint64_t descriptorKey = CalculateDescriptorKey(
        endpointId, endpoint.south_endpoint_id, memId, memory.addr, memory.size, memory.mem_type);
    auto updateResult = sim::runtime::Db::Update<sim::runtime::HcommMemReg>(
        HcclSim::Storage::Eq(&sim::runtime::HcommMemReg::id, memId),
        HcclSim::Storage::Set(&sim::runtime::HcommMemReg::descriptor_key, descriptorKey));
    if (!updateResult.ok()) {
        (void)sim::runtime::Db::Delete<sim::runtime::HcommMemReg>(
            HcclSim::Storage::Eq(&sim::runtime::HcommMemReg::id, memId));
        HCCL_VM_ERROR("{}: failed to update descriptor key, endpointId={:d}, memId={:d}", __func__, endpointId, memId);
        return static_cast<HcommResult>(HCCL_E_INTERNAL);
    }

    *memHandle = reinterpret_cast<HcommMemHandle>(static_cast<uintptr_t>(memId));
    HCCL_VM_INFO(
        "{} success, endpointId={:d}, memId={:d}, addr={:p}, "
        "size={:d}, type={:d}, tagLength={:d}",
        __func__, endpointId, memId, mem->addr, mem->size, static_cast<int>(mem->type),
        static_cast<uint32_t>(memTagLength));
    return static_cast<HcommResult>(HCCL_SUCCESS);
}

/**
 * @brief 注销指定Endpoint上的本地注册内存。
 */
HcommResult HcommMemUnreg(EndpointHandle endpointHandle, HcommMemHandle memHandle)
{
    // 1. 真实入口优先检查内存句柄空指针。
    if (memHandle == nullptr) {
        HCCL_VM_ERROR("{}: memHandle is nullptr", __func__);
        return static_cast<HcommResult>(HCCL_E_PTR);
    }

    // 2. 校验Endpoint及本地内存的归属和记录类型。
    sim::runtime::Runner runner{};
    sim::runtime::HcommEndpoint endpoint{};
    uint64_t endpointId = 0u;
    HcommResult result = GetOwnedEndpoint(endpointHandle, runner, endpointId, endpoint);
    if (result != HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "{}: endpoint not found or not owned, endpointId={:d}, "
            "result={:d}",
            __func__, endpointId, static_cast<int>(result));
        return result;
    }

    sim::runtime::HcommMemReg memory{};
    result = GetLocalMemory(memHandle, endpointId, runner.id, memory);
    if (result != HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "{}: local memory not found, endpointId={:d}, memId={:d}", __func__, endpointId,
            DecodeOpaqueHandle(memHandle));
        return result;
    }

    // 3. 删除存储层唯一资源记录，并释放该内存对应的导出返回缓冲。
    auto delResult = sim::runtime::Db::Delete<sim::runtime::HcommMemReg>(
        HcclSim::Storage::Eq(&sim::runtime::HcommMemReg::id, memory.id));
    if (!delResult.ok()) {
        HCCL_VM_ERROR("{}: failed to delete memory, endpointId={:d}, memId={:d}", __func__, endpointId, memory.id);
        return static_cast<HcommResult>(HCCL_E_INTERNAL);
    }
    {
        std::lock_guard<std::mutex> lock(g_hcommMemDescBuffersMutex);
        g_hcommMemDescBuffers.erase(memory.id);
    }

    HCCL_VM_INFO(
        "{} success, endpointId={:d}, memId={:d}, addr=0x{:x}, size={:d}", __func__, endpointId, memory.id, memory.addr,
        memory.size);
    return static_cast<HcommResult>(HCCL_SUCCESS);
}

/**
 * @brief 导出本地注册内存的固定版本描述符。
 *
 * 返回地址由函数内部稳定节点保存，在该内存注销或Endpoint销毁前有效；调用者
 * 不得释放该地址。
 */
HcommResult
HcommMemExport(EndpointHandle endpointHandle, HcommMemHandle memHandle, void** memDesc, uint32_t* memDescLen)
{
    if (memHandle == nullptr || memDesc == nullptr || memDescLen == nullptr) {
        HCCL_VM_ERROR("{}: memHandle, memDesc or memDescLen is nullptr", __func__);
        return static_cast<HcommResult>(HCCL_E_PTR);
    }
    *memDesc = nullptr;
    *memDescLen = 0u;

    // 2. 校验Endpoint和本地内存句柄属于同一个当前Runner资源。
    sim::runtime::Runner runner{};
    sim::runtime::HcommEndpoint endpoint{};
    uint64_t endpointId = 0u;
    HcommResult result = GetOwnedEndpoint(endpointHandle, runner, endpointId, endpoint);
    if (result != HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "{}: endpoint not found or not owned, endpointId={:d}, "
            "result={:d}",
            __func__, endpointId, static_cast<int>(result));
        return result;
    }

    sim::runtime::HcommMemReg memory{};
    result = GetLocalMemory(memHandle, endpointId, runner.id, memory);
    if (result != HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "{}: local memory not found, endpointId={:d}, memId={:d}", __func__, endpointId,
            DecodeOpaqueHandle(memHandle));
        return result;
    }

    // 3. 每个内存ID拥有独立稳定缓冲，导出其他内存不会覆盖本次返回内容。
    {
        std::lock_guard<std::mutex> lock(g_hcommMemDescBuffersMutex);
        auto& descriptor = g_hcommMemDescBuffers[memory.id];
        descriptor = BuildMemDescriptor(endpoint, memory);
        *memDesc = static_cast<void*>(&descriptor);
        *memDescLen = sizeof(descriptor);
    }

    HCCL_VM_INFO(
        "{} success, endpointId={:d}, memId={:d}, descriptorKey={:d}, descLen={:d}", __func__, endpointId, memory.id,
        memory.descriptor_key, *memDescLen);
    return static_cast<HcommResult>(HCCL_SUCCESS);
}

/**
 * @brief 将远端导出的内存描述符导入到当前Endpoint。
 *
 * HCCS/HCCS_ONLY重复导入保持幂等成功；其他协议重复导入返回HCCL_E_AGAIN。
 */
HcommResult HcommMemImport(EndpointHandle endpointHandle, const void* memDesc, uint32_t descLen, CommMem* outMem)
{
    if (memDesc == nullptr || outMem == nullptr) {
        HCCL_VM_ERROR("{}: memDesc or outMem is nullptr", __func__);
        return static_cast<HcommResult>(HCCL_E_PTR);
    }
    *outMem = CommMem{};

    // 2. 校验目标Endpoint归属并解析固定版本描述符。
    sim::runtime::Runner runner{};
    sim::runtime::HcommEndpoint endpoint{};
    uint64_t endpointId = 0u;
    HcommResult result = GetOwnedEndpoint(endpointHandle, runner, endpointId, endpoint);
    if (result != HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "{}: endpoint not found or not owned, endpointId={:d}, "
            "result={:d}",
            __func__, endpointId, static_cast<int>(result));
        return result;
    }

    HcommMemDescriptor descriptor{};
    HcommMemDescParseResult parseResult = ParseMemDescriptor(memDesc, descLen, descriptor);
    if (parseResult != HcommMemDescParseResult::SUCCESS) {
        HCCL_VM_ERROR(
            "{}: invalid descriptor, endpointId={:d}, descLen={:d}, "
            "parseResult={:d}",
            __func__, endpointId, descLen, static_cast<int>(parseResult));
        return static_cast<HcommResult>(HCCL_E_PARA);
    }
    if (!IsValidMemType(static_cast<CommMemType>(descriptor.mem_type))) {
        HCCL_VM_ERROR("{}: invalid memory type, endpointId={:d}, type={:d}", __func__, endpointId, descriptor.mem_type);
        return static_cast<HcommResult>(HCCL_E_PARA);
    }

    // 3. 共享存储环境下校验描述符引用的源资源，拒绝陈旧或篡改描述符。
    result = ValidateDescriptorSource(descriptor);
    if (result != HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "{}: source validation failed, endpointId={:d}, "
            "sourceEndpointId={:d}, sourceMemId={:d}, result={:d}",
            __func__, endpointId, descriptor.source_endpoint_id, descriptor.source_mem_id, static_cast<int>(result));
        return result;
    }

    // 4. 先处理重复导入，HCCS保持幂等，其他协议对齐真实实现返回AGAIN。
    auto imported = sim::runtime::Db::GetByPred<sim::runtime::HcommMemReg>(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&sim::runtime::HcommMemReg::endpoint_id, endpointId),
        HcclSim::Storage::Eq(&sim::runtime::HcommMemReg::record_kind, sim::runtime::HCOMM_MEM_REMOTE_IMPORTED),
        HcclSim::Storage::Eq(&sim::runtime::HcommMemReg::source_endpoint_id, descriptor.source_endpoint_id),
        HcclSim::Storage::Eq(&sim::runtime::HcommMemReg::source_mem_id, descriptor.source_mem_id),
        HcclSim::Storage::Eq(&sim::runtime::HcommMemReg::descriptor_key, descriptor.descriptor_key)));
    if (imported.ok() && !imported->empty()) {
        if (endpoint.protocol != COMM_PROTOCOL_HCCS && endpoint.protocol != COMM_PROTOCOL_HCCS_ONLY) {
            HCCL_VM_ERROR(
                "{}: descriptor already imported, endpointId={:d}, "
                "sourceMemId={:d}",
                __func__, endpointId, descriptor.source_mem_id);
            return static_cast<HcommResult>(HCCL_E_AGAIN);
        }
        outMem->addr = reinterpret_cast<void*>(static_cast<uintptr_t>((*imported)[0].addr));
        outMem->size = (*imported)[0].size;
        outMem->type = static_cast<CommMemType>((*imported)[0].mem_type);
        HCCL_VM_INFO(
            "{} idempotent success, endpointId={:d}, importedMemId={:d}", __func__, endpointId, (*imported)[0].id);
        return static_cast<HcommResult>(HCCL_SUCCESS);
    }

    // 5. 写入远端导入记录；导入接口无句柄输出，记录由描述符字段定位和释放。
    sim::runtime::HcommMemReg memory{};
    memory.endpoint_id = endpointId;
    memory.runner_id = runner.id;
    memory.source_endpoint_id = descriptor.source_endpoint_id;
    memory.source_mem_id = descriptor.source_mem_id;
    memory.descriptor_key = descriptor.descriptor_key;
    memory.addr = descriptor.addr;
    memory.size = descriptor.size;
    memory.mem_type = descriptor.mem_type;
    memory.record_kind = sim::runtime::HCOMM_MEM_REMOTE_IMPORTED;

    auto addResult = sim::runtime::Db::Add<sim::runtime::HcommMemReg>(memory);
    if (!addResult.ok()) {
        HCCL_VM_ERROR(
            "{}: failed to add imported memory, endpointId={:d}, "
            "sourceMemId={:d}",
            __func__, endpointId, descriptor.source_mem_id);
        return static_cast<HcommResult>(HCCL_E_INTERNAL);
    }
    uint64_t importedMemId = memory.id;

    outMem->addr = reinterpret_cast<void*>(static_cast<uintptr_t>(descriptor.addr));
    outMem->size = descriptor.size;
    outMem->type = static_cast<CommMemType>(descriptor.mem_type);
    HCCL_VM_INFO(
        "{} success, endpointId={:d}, importedMemId={:d}, "
        "sourceEndpointId={:d}, sourceMemId={:d}, addr={:p}, size={:d}",
        __func__, endpointId, importedMemId, descriptor.source_endpoint_id, descriptor.source_mem_id, outMem->addr,
        outMem->size);
    return static_cast<HcommResult>(HCCL_SUCCESS);
}

/**
 * @brief 释放指定Endpoint上由描述符定位的远端导入内存。
 */
HcommResult HcommMemUnimport(EndpointHandle endpointHandle, const void* memDesc, uint32_t descLen)
{
    // 1. 真实入口首先检查描述符指针。
    if (memDesc == nullptr) {
        HCCL_VM_ERROR("{}: memDesc is nullptr", __func__);
        return static_cast<HcommResult>(HCCL_E_PTR);
    }

    // 2. 校验目标Endpoint并解析描述符；源端已注销不影响目标端释放已有导入记录。
    sim::runtime::Runner runner{};
    sim::runtime::HcommEndpoint endpoint{};
    uint64_t endpointId = 0u;
    HcommResult result = GetOwnedEndpoint(endpointHandle, runner, endpointId, endpoint);
    if (result != HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "{}: endpoint not found or not owned, endpointId={:d}, "
            "result={:d}",
            __func__, endpointId, static_cast<int>(result));
        return result;
    }

    HcommMemDescriptor descriptor{};
    HcommMemDescParseResult parseResult = ParseMemDescriptor(memDesc, descLen, descriptor);
    if (parseResult != HcommMemDescParseResult::SUCCESS) {
        HCCL_VM_ERROR(
            "{}: invalid descriptor, endpointId={:d}, descLen={:d}, "
            "parseResult={:d}",
            __func__, endpointId, descLen, static_cast<int>(parseResult));
        return static_cast<HcommResult>(HCCL_E_PARA);
    }

    // 3. 导入接口没有返回句柄，按目标Endpoint和描述符源标识定位唯一记录。
    auto imported = sim::runtime::Db::GetByPred<sim::runtime::HcommMemReg>(HcclSim::Storage::And(
        HcclSim::Storage::Eq(&sim::runtime::HcommMemReg::endpoint_id, endpointId),
        HcclSim::Storage::Eq(&sim::runtime::HcommMemReg::record_kind, sim::runtime::HCOMM_MEM_REMOTE_IMPORTED),
        HcclSim::Storage::Eq(&sim::runtime::HcommMemReg::source_endpoint_id, descriptor.source_endpoint_id),
        HcclSim::Storage::Eq(&sim::runtime::HcommMemReg::source_mem_id, descriptor.source_mem_id),
        HcclSim::Storage::Eq(&sim::runtime::HcommMemReg::descriptor_key, descriptor.descriptor_key)));
    if (!imported.ok() || imported->empty()) {
        HCCL_VM_ERROR(
            "{}: imported memory not found, endpointId={:d}, "
            "sourceMemId={:d}",
            __func__, endpointId, descriptor.source_mem_id);
        return static_cast<HcommResult>(HCCL_E_NOT_FOUND);
    }
    if (imported->size() > 1u) {
        HCCL_VM_ERROR(
            "{}: duplicate imported records, endpointId={:d}, "
            "sourceMemId={:d}, count={:d}",
            __func__, endpointId, descriptor.source_mem_id, static_cast<uint32_t>(imported->size()));
        return static_cast<HcommResult>(HCCL_E_INTERNAL);
    }
    auto delResult = sim::runtime::Db::Delete<sim::runtime::HcommMemReg>(
        HcclSim::Storage::Eq(&sim::runtime::HcommMemReg::id, (*imported)[0].id));
    if (!delResult.ok()) {
        HCCL_VM_ERROR(
            "{}: failed to delete imported memory, endpointId={:d}, "
            "importedMemId={:d}",
            __func__, endpointId, (*imported)[0].id);
        return static_cast<HcommResult>(HCCL_E_INTERNAL);
    }

    HCCL_VM_INFO(
        "{} success, endpointId={:d}, importedMemId={:d}, "
        "sourceMemId={:d}",
        __func__, endpointId, (*imported)[0].id, descriptor.source_mem_id);
    return static_cast<HcommResult>(HCCL_SUCCESS);
}

/**
 * @brief 分配通信线程资源（基础资源管理接口，不绑定任何通信域）。
 *
 * HcommThreadAlloc 是控制面基础资源管理接口，与 HcclThreadAcquire 的区别在于：
 *   - 不依赖 HcclComm（通信域句柄）；
 *   - 分配的线程句柄可直接用于 HcommWriteOnThread / HcommReadOnThread 等
 *     数据面接口；若需要在集合通信中使用，可再通过 HcclThread* 系列接口绑定。
 *
 * 实现要点：
 *   - 复用 sim::runtime::HcclThread 表，commId 字段记为 HCOMM_BASIC_THREAD_COMM_ID(0) 表示
 *     “非通信域绑定”的独立线程；
 *   - 限制：单次分配 threadNum 不超过 40；累计 notifyNum 不超过 640
 *     （与 HcclThreadAcquire 对齐，保证设备资源不越界）。
 *   - 返回的 ThreadHandle 即 sim::runtime::HcclThread 的主键 id，后续可由
 *     HcommThreadFree 释放。
 *
 * @param engine               通信引擎类型（CommEngine）。
 * @param threadNum            本次需分配的线程数量。
 * @param notifyNumPerThread   每条线程上的通知资源数量。
 * @param threads              输出参数，返回 threadNum 个 ThreadHandle，
 *                             调用者需保证数组长度 >= threadNum。
 * @return HcommResult 成功返回 0，其它情况返回错误码（与 HcclResult 数值对齐）。
 */
HcommResult HcommThreadAlloc(CommEngine engine, uint32_t threadNum, uint32_t notifyNumPerThread, ThreadHandle* threads)
{
    // 1. 参数校验
    if (threads == nullptr) {
        HCCL_VM_ERROR("{}: threads is nullptr", __func__);
        return static_cast<HcommResult>(HCCL_E_PTR);
    }
    if (threadNum == 0 || threadNum > HCOMM_BASIC_MAX_THREAD_NUM) {
        HCCL_VM_ERROR("{}: threadNum={:d} out of range (0, {:d}]", __func__, threadNum, HCOMM_BASIC_MAX_THREAD_NUM);
        return static_cast<HcommResult>(HCCL_E_PARA);
    }

    // 2. 查询当前已存的"基础"线程（commId == 0），用于累计资源计数
    auto existing = sim::runtime::Db::GetByPred<sim::runtime::HcclThread>(
        HcclSim::Storage::Eq(&sim::runtime::HcclThread::commId, HCOMM_BASIC_THREAD_COMM_ID));

    // 3. 线程数量上限检查
    uint64_t existingCount = existing->size();
    if (existingCount + static_cast<uint64_t>(threadNum) > HCOMM_BASIC_MAX_THREAD_NUM) {
        HCCL_VM_ERROR(
            "{}: total threads exceed limit {:d}, "
            "already have {:d}, requested {:d}",
            __func__, HCOMM_BASIC_MAX_THREAD_NUM, existingCount, threadNum);
        return static_cast<HcommResult>(HCCL_E_INTERNAL);
    }

    // 4. notifyNum 累计上限检查
    uint64_t existingNotifyTotal = 0u;
    for (const auto& thr : *existing) {
        existingNotifyTotal += thr.notifyNum;
    }
    uint64_t requestNotifyTotal = static_cast<uint64_t>(threadNum) * static_cast<uint64_t>(notifyNumPerThread);
    if (existingNotifyTotal + requestNotifyTotal > HCOMM_BASIC_MAX_NOTIFY_TOTAL) {
        HCCL_VM_ERROR(
            "{}: notify count exceed limit {:d}, "
            "existing {:d} + request {:d}",
            __func__, HCOMM_BASIC_MAX_NOTIFY_TOTAL, existingNotifyTotal, requestNotifyTotal);
        return static_cast<HcommResult>(HCCL_E_INTERNAL);
    }

    // 5. 逐条创建 HcclThread 记录（commId = 0 表示基础资源，stream = 0）
    for (uint32_t i = 0; i < threadNum; ++i) {
        sim::runtime::HcclThread rec{};
        rec.commId = HCOMM_BASIC_THREAD_COMM_ID;
        rec.engine = static_cast<uint64_t>(engine);
        rec.notifyNum = static_cast<uint64_t>(notifyNumPerThread);
        aclrtStream stream = nullptr;
        if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
            HCCL_VM_ERROR("{}: aclrtCreateStream failed (i={:d})", __func__, i);
            for (uint32_t j = 0; j < i; ++j) {
                sim::runtime::Db::Delete<sim::runtime::HcclThread>(
                    HcclSim::Storage::Eq(&sim::runtime::HcclThread::id, threads[j]));
                threads[j] = 0u;
            }
            return static_cast<HcommResult>(HCCL_E_INTERNAL);
        }
        rec.streamId = reinterpret_cast<uint64_t>(stream);

        auto addResult = sim::runtime::Db::Add<sim::runtime::HcclThread>(rec);
        if (!addResult.ok()) {
            HCCL_VM_ERROR("{}: failed to add HcclThread record (i={:d})", __func__, i);

            // 回滚已写入的记录
            for (uint32_t j = 0; j < i; ++j) {
                sim::runtime::Db::Delete<sim::runtime::HcclThread>(
                    HcclSim::Storage::Eq(&sim::runtime::HcclThread::id, threads[j]));
                threads[j] = 0u;
            }
            return static_cast<HcommResult>(HCCL_E_INTERNAL);
        }
        uint64_t recId = rec.id;
        threads[i] = static_cast<ThreadHandle>(recId);
    }

    HCCL_VM_INFO(
        "{} success, engine={:d}, threadNum={:d}, "
        "notifyNumPerThread={:d}",
        __func__, static_cast<int>(engine), threadNum, notifyNumPerThread);
    return static_cast<HcommResult>(HCCL_SUCCESS);
}

/**
 * @brief 释放由 HcommThreadAlloc 分配的通信线程资源。
 *
 * 实现要点：
 *   - 逐句柄在 sim::runtime::HcclThread 表中查找并删除记录；
 *   - 若某个句柄对应的记录已不存在，视为“已成功释放”，不报错（幂等）；
 *   - threadNum == 0 视为空操作，直接返回成功；
 *   - threads 指针为 nullptr 返回 HCCL_E_PTR 错误。
 *
 * @param threads   待释放的 ThreadHandle 数组。
 * @param threadNum 数组中的句柄数量。
 * @return HcommResult 成功返回 0，其它情况返回错误码（与 HcclResult 数值对齐）。
 */
HcommResult HcommThreadFree(const ThreadHandle* threads, uint32_t threadNum)
{
    if (threads == nullptr) {
        HCCL_VM_ERROR("{}: threads is nullptr", __func__);
        return static_cast<HcommResult>(HCCL_E_PTR);
    }

    uint32_t freedCount = 0u;
    uint32_t skippedCount = 0u;

    // 【核心步骤】逐个处理待释放的线程句柄（幂等设计：已释放或无效句柄不报错）
    for (uint32_t i = 0; i < threadNum; ++i) {
        ThreadHandle handle = threads[i];

        // 句柄为 0 视为"已释放"（HcommThreadAlloc 失败回滚时会写入 0）
        if (handle == 0u) {
            ++skippedCount;
            continue;
        }

        // 查询记录是否存在（放宽检查：只要是合法记录都可释放，避免跨接口误用引发断言）
        auto opt = sim::runtime::Db::GetById<sim::runtime::HcclThread>(handle);
        if (!opt.ok()) {
            HCCL_VM_WARN("{}: handle {:d} not found (already freed?)", __func__, handle);
            ++skippedCount;
            continue;
        }

        // 释放该线程关联的 aclrtStream，避免资源泄漏
        if (opt->streamId != 0u) {
            aclrtStream stream = reinterpret_cast<aclrtStream>(opt->streamId);
            (void)aclrtDestroyStream(stream);
        }

        // 执行删除
        sim::runtime::Db::Delete<sim::runtime::HcclThread>(HcclSim::Storage::Eq(&sim::runtime::HcclThread::id, handle));
        ++freedCount;
    }

    HCCL_VM_INFO("{} done, requested={:d}, freed={:d}, skipped={:d}", __func__, threadNum, freedCount, skippedCount);
    return static_cast<HcommResult>(HCCL_SUCCESS);
}

/**
 * @brief 初始化通信通道描述列表。
 *
 * @note HcommChannelDescInit 在 hcomm 头文件中为 static inline 函数，北向劫持无法
 *       拦截其调用，因此此桩函数仅作为占位存在，始终返回 HCCL_E_NOT_SUPPORT。
 *       用户应使用 hcomm 头文件中提供的 static inline 版本完成初始化。
 *
 * @param channelDesc 输入：待初始化的 HcommChannelDesc 结构体数组指针。
 * @param descNum 输入：数组元素数量。
 * @return HcommResult 始终返回 HCCL_E_NOT_SUPPORT（与 HcclResult 数值对齐）。
 */
HcommResult HcommChannelDescInit(HcommChannelDesc* channelDesc, uint32_t descNum)
{
    (void)channelDesc;
    (void)descNum;
    HCCL_VM_ERROR(
        "{} is a static inline function (hcomm header), "
        "cannot be hijacked; use hcomm's inline version instead.",
        __func__);
    return static_cast<HcommResult>(HCCL_E_NOT_SUPPORT);
}

/**
 * @brief 基于已创建的网络端点批量创建通信通道。
 *
 * HcommChannelCreate 是控制面基础资源管理接口，为点对点通信或集合通信提供数据传输
 * 的基础设施。与 HcclChannelAcquire 的区别：本接口不依赖 HcclComm 通信域句柄，
 * 创建的通道为全局基础通道（commId=HCOMM_BASIC_COMM_ID）。
 *
 * 实现要点：
 *   - endpointHandle / channelDescs / channels 任一为 nullptr 返回 HCCL_E_PTR；
 *   - channelNum 取值范围为 (0, HCOMM_BASIC_MAX_CHANNEL_NUM(4096)]；
 *   - engine=COMM_ENGINE_CCU 时，notifyNum 强制等于 8（CCU 默认值），且 memHandleNum
 *     不允许超过 1（CCU 仅支持交换 1 份 memHandle）；
 *   - 每次创建均在 sim::runtime::HcclChannel 表中写入一条记录（commId=0 表示基础通道，
 *     remoteRankId=0，status=false），并将 DB 主键作为 ChannelHandle 返回；
 *   - DB 写入失败或 CCU memHandleNum 非法时回滚已写入的记录；
 *   - 通道初始 status=false（INIT），调用方需通过 HcommChannelGetStatus 轮询。
 *
 * @param endpointHandle  输入：已创建的本地网络端点句柄（本桩仅校验非空）。
 * @param engine          输入：通信引擎类型（CommEngine）。
 * @param channelDescs    输入：通道描述数组，必须先用 HcommChannelDescInit 初始化。
 * @param channelNum      输入：本次创建的通道数量。
 * @param channels        输出：返回 channelNum 个 ChannelHandle。
 * @return HcommResult 成功返回 0，其它情况返回错误码（与 HcclResult 数值对齐）。
 */
HcommResult HcommChannelCreate(
    EndpointHandle endpointHandle, CommEngine engine, HcommChannelDesc* channelDescs, uint32_t channelNum,
    ChannelHandle* channels)
{
    if (endpointHandle == nullptr || channelDescs == nullptr) {
        HCCL_VM_ERROR("{}: endpointHandle or channelDescs is nullptr", __func__);
        return static_cast<HcommResult>(HCCL_E_PTR);
    }
    if (channelNum == 0 || channelNum > HCOMM_BASIC_MAX_CHANNEL_NUM) {
        HCCL_VM_ERROR("{}: channelNum={:d} out of range (0, {:d}]", __func__, channelNum, HCOMM_BASIC_MAX_CHANNEL_NUM);
        return static_cast<HcommResult>(HCCL_E_PARA);
    }

    // 校验 Endpoint 归属，获取 endpointId 用于通道记录关联
    sim::runtime::Runner runner{};
    sim::runtime::HcommEndpoint endpoint{};
    uint64_t endpointId = 0u;
    HcommResult ownerResult = GetOwnedEndpoint(endpointHandle, runner, endpointId, endpoint);
    if (ownerResult != HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "{}: endpoint not found or not owned, "
            "result={:d}",
            __func__, static_cast<int>(ownerResult));
        return ownerResult;
    }

    for (uint32_t i = 0; i < channelNum; ++i) {
        const HcommChannelDesc& desc = channelDescs[i];

        // 【核心步骤 1】确定 notifyNum：CCU 引擎强制使用默认值 8
        uint64_t notifyNum = desc.notifyNum;
        if (engine == COMM_ENGINE_CCU) {
            notifyNum = HCOMM_BASIC_CCU_DEFAULT_NOTIFY;

            // CCU 引擎仅支持交换 1 份 memHandle（用于 HcclBuffer），超过则报错
            if (desc.memHandleNum > 1u) {
                HCCL_VM_ERROR(
                    "{}: CCU engine supports at most 1 memHandle, "
                    "index={:d}, memHandleNum={:d}",
                    __func__, i, desc.memHandleNum);

                // 回滚已创建的通道
                for (uint32_t j = 0; j < i; ++j) {
                    sim::runtime::Db::Delete<sim::runtime::HcclChannel>(
                        HcclSim::Storage::Eq(&sim::runtime::HcclChannel::id, channels[j]));
                    channels[j] = 0u;
                }
                return static_cast<HcommResult>(HCCL_E_PARA);
            }
        }

        // 【核心步骤 2】构造 HcclChannel 记录（不绑定通信域，commId=0 表示基础通道）
        sim::runtime::HcclChannel rec{};
        rec.commId = HCOMM_BASIC_COMM_ID; // 基础通道不绑定通信域
        rec.endpoint_id = endpointId;     // 关联创建通道的 Endpoint
        rec.engine = static_cast<uint64_t>(engine);
        rec.remoteRankId = HCOMM_BASIC_REMOTE_RANK_ID; // 基础通道不绑定远端 rank
        rec.notifyNum = notifyNum;
        rec.status = true; // 基础通道初始即可用（非 INIT 状态）

        // 【核心步骤 3】插入通道记录，获取 DB 主键作为 ChannelHandle
        auto addResult = sim::runtime::Db::Add<sim::runtime::HcclChannel>(rec);
        if (!addResult.ok()) {
            HCCL_VM_ERROR("{}: failed to add HcclChannel record (i={:d})", __func__, i);

            // 回滚已创建的通道
            for (uint32_t j = 0; j < i; ++j) {
                sim::runtime::Db::Delete<sim::runtime::HcclChannel>(
                    HcclSim::Storage::Eq(&sim::runtime::HcclChannel::id, channels[j]));
                channels[j] = 0u;
            }
            return static_cast<HcommResult>(HCCL_E_INTERNAL);
        }
        uint64_t recId = rec.id;

        // DB 主键 = ChannelHandle
        channels[i] = static_cast<ChannelHandle>(recId);
    }

    HCCL_VM_INFO("{} success, engine={:d}, channelNum={:d}", __func__, static_cast<int>(engine), channelNum);
    return static_cast<HcommResult>(HCCL_SUCCESS);
}

/**
 * @brief 批量查询通信通道的状态。
 *
 * HcommChannelGetStatus 用于在通道建立阶段做同步等待，调用方需持续轮询直到所有
 * 通道返回 READY 状态。本桩实现基于 sim::runtime::HcclChannel.status（bool）字段映射到
 * CANN 契约枚举 HcommChannelStatus（hcomm_channel.h）：
 *   - status=true  → HCOMM_CHANNEL_STATUS_READY(0)        建链完成
 *   - status=false → HCOMM_CHANNEL_STATUS_CONNECTING(1)   未就绪，调用方继续轮询
 * 不存在的句柄返回 CONNECTING(1)，调用方可据此决定是否继续轮询。
 *
 * @param channelList  输入：待查询的 ChannelHandle 数组。
 * @param listNum      输入：数组元素数量。
 * @param statusList   输出：与 channelList 一一对应的状态数组（HcommChannelStatus 取值）。
 * @return HcommResult 成功返回 0，其它情况返回错误码（与 HcclResult 数值对齐）。
 */
HcommResult HcommChannelGetStatus(const ChannelHandle* channelList, uint32_t listNum, int32_t* statusList)
{
    if (channelList == nullptr) {
        HCCL_VM_ERROR("{}: channelList is nullptr", __func__);
        return static_cast<HcommResult>(HCCL_E_PTR);
    }

    // 【核心步骤】逐个查询通道状态并映射到 HcommChannelStatus 枚举
    for (uint32_t i = 0; i < listNum; ++i) {
        ChannelHandle handle = channelList[i];

        // 句柄为 0 或记录不存在：视为未就绪（调用方应继续轮询）
        if (handle == 0u) {
            statusList[i] = HCOMM_CHANNEL_STATUS_CONNECTING;
            continue;
        }
        auto opt = sim::runtime::Db::GetById<sim::runtime::HcclChannel>(handle);
        if (!opt.ok()) {
            // 句柄被删除：视为未就绪（调用方应继续轮询）
            statusList[i] = HCOMM_CHANNEL_STATUS_CONNECTING;
            continue;
        }

        // 状态映射：sim::runtime::HcclChannel.status (bool) → HCOMM_CHANNEL_STATUS_READY/CONNECTING
        // （控制面基础通道在创建后即 status=true，此处主要供数据面接口的通道使用）
        statusList[i] = opt->status ? HCOMM_CHANNEL_STATUS_READY : HCOMM_CHANNEL_STATUS_CONNECTING;
    }

    HCCL_VM_INFO("{} done, listNum={:d}", __func__, listNum);
    return static_cast<HcommResult>(HCCL_SUCCESS);
}

/**
 * @brief 获取指定 channel 的 Notify 数量。
 *
 * 实现要点：
 *   - channelHandle 由 HcommChannelCreate 返回的 ChannelHandle（DB PK）；
 *   - 不存在的句柄返回 HCCL_E_NOT_FOUND；
 *   - notifyNum 字段在 HcommChannelCreate 时写入，CCU 引擎固定为 8。
 *
 * @param channelHandle 输入：待查询的通信通道句柄。
 * @param notifyNum     输出：通道上的 Notify 数量。
 * @return HcommResult 成功返回 0，其它情况返回错误码（与 HcclResult 数值对齐）。
 */
HcommResult HcommChannelGetNotifyNum(ChannelHandle channelHandle, uint32_t* notifyNum)
{
    // 【核心步骤 1】句柄 0 视为无效
    if (channelHandle == 0u) {
        HCCL_VM_ERROR("{}: channelHandle is 0", __func__);
        return static_cast<HcommResult>(HCCL_E_NOT_FOUND);
    }

    // 【核心步骤 2】查询通道记录
    auto opt = sim::runtime::Db::GetById<sim::runtime::HcclChannel>(channelHandle);
    if (!opt.ok()) {
        HCCL_VM_ERROR("{}: channel {:d} not found", __func__, channelHandle);
        return static_cast<HcommResult>(HCCL_E_NOT_FOUND);
    }

    // 【核心步骤 3】读取 notifyNum 字段（在 HcommChannelCreate 时写入）
    *notifyNum = static_cast<uint32_t>(opt->notifyNum);
    HCCL_VM_INFO("{} success, channel={:d}, notifyNum={:d}", __func__, channelHandle, *notifyNum);
    return static_cast<HcommResult>(HCCL_SUCCESS);
}

/**
 * @brief 批量销毁通信通道。
 *
 * HcommChannelDestroy 释放由 HcommChannelCreate 创建的通信通道及其占用的所有系统资源。
 * 该接口支持批量销毁，且操作是原子性的：任一通道句柄无效则整体失败，不会发生
 * “部分销毁”现象。
 *
 * 实现要点：
 *   - channels 不能为 nullptr；
 *   - channelNum == 0 视为空操作，直接返回成功；
 *   - 先逐个校验所有句柄是否有效（存在），遇首个无效句柄即返回 HCCL_E_NOT_FOUND；
 *   - 所有句柄有效后统一执行 sim::runtime::Db::Delete。
 *
 * @param channels    输入：待销毁的 ChannelHandle 数组。
 * @param channelNum  输入：数组元素数量。
 * @return HcommResult 成功返回 0，其它情况返回错误码（与 HcclResult 数值对齐）。
 */
HcommResult HcommChannelDestroy(const ChannelHandle* channels, uint32_t channelNum)
{
    if (channels == nullptr) {
        HCCL_VM_ERROR("{}: channels is nullptr", __func__);
        return static_cast<HcommResult>(HCCL_E_PTR);
    }
    if (channelNum == 0u) {
        HCCL_VM_INFO("{}: channelNum is 0, nothing to do", __func__);
        return static_cast<HcommResult>(HCCL_SUCCESS);
    }

    // 【核心步骤 1】预校验：所有句柄必须有效（原子性保证）
    // 只要有一个句柄无效就返回错误，不会发生"部分销毁"
    for (uint32_t i = 0; i < channelNum; ++i) {
        if (!sim::runtime::Db::GetById<sim::runtime::HcclChannel>(channels[i]).ok()) {
            HCCL_VM_ERROR("{}: channel {:d} not found (index={:d})", __func__, channels[i], i);
            return static_cast<HcommResult>(HCCL_E_NOT_FOUND);
        }
    }

    // 【核心步骤 2】执行批量删除
    for (uint32_t i = 0; i < channelNum; ++i) {
        sim::runtime::Db::Delete<sim::runtime::HcclChannel>(
            HcclSim::Storage::Eq(&sim::runtime::HcclChannel::id, channels[i]));
    }

    HCCL_VM_INFO("{} success, channelNum={:d}", __func__, channelNum);
    return static_cast<HcommResult>(HCCL_SUCCESS);
}

/**
 * @brief 查询 Endpoint 的监听端口。
 *
 * 真实实现中 Endpoint 创建时会绑定 socket 并获取实际监听端口。本桩不含真实
 * 网络栈，按 Endpoint ID 确定性地派生一个模拟端口返回。
 *
 * @param endpointHandle  输入：已创建的 Endpoint 句柄。
 * @param port            输出：监听端口号。
 * @return HcommResult 成功返回 0，其它情况返回错误码。
 */
HcommResult HcommEndpointGetListenPort(EndpointHandle endpointHandle, uint32_t* port)
{
    if (port == nullptr) {
        HCCL_VM_ERROR("{}: port is nullptr", __func__);
        return static_cast<HcommResult>(HCCL_E_PTR);
    }
    *port = 0u;

    sim::runtime::Runner runner{};
    sim::runtime::HcommEndpoint endpoint{};
    uint64_t endpointId = 0u;
    HcommResult result = GetOwnedEndpoint(endpointHandle, runner, endpointId, endpoint);
    if (result != HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "{}: endpoint not found or not owned, "
            "endpointId={:d}, result={:d}",
            __func__, endpointId, static_cast<int>(result));
        return result;
    }

    *port = HCOMM_BASIC_BASE_PORT + static_cast<uint32_t>(endpointId % 10000u);
    HCCL_VM_INFO("{} success, endpointId={:d}, port={:d}", __func__, endpointId, *port);
    return static_cast<HcommResult>(HCCL_SUCCESS);
}

/**
 * @brief 检查 Endpoint 是否支持指定特性（如 NDA）。
 *
 * 真实实现通过 RoCE RDMA 句柄查询 directFlag 判断 NDA 支持情况。本桩不含
 * 硬件 RDMA 栈，按协议和位置类型做静态判断：仅 RoCE + HOST 组合可能支持 NDA，
 * 其余一律返回 false。
 *
 * @param featureType   输入：待检查的特性类型（HcommEndpointFeatureType）。
 * @param endpointDesc  输入：Endpoint 描述符。
 * @param value         输出：是否支持该特性。
 * @return HcommResult 成功返回 0，其它情况返回错误码。
 */
HcommResult
HcommEndpointCheckFeature(HcommEndpointFeatureType featureType, const EndpointDesc* endpointDesc, bool* value)
{
    if (endpointDesc == nullptr || value == nullptr) {
        HCCL_VM_ERROR("{}: endpointDesc or value is nullptr", __func__);
        return static_cast<HcommResult>(HCCL_E_PTR);
    }
    *value = false;

    if (featureType == HCOMM_ENDPOINT_FEATURE_NDA) {
        if (endpointDesc->protocol == COMM_PROTOCOL_ROCE && endpointDesc->loc.locType == ENDPOINT_LOC_TYPE_HOST) {
            *value = true;
        }
    }

    HCCL_VM_INFO(
        "{}: featureType={:d}, protocol={:d}, "
        "locType={:d}, supported={}",
        __func__, static_cast<int>(featureType), static_cast<int>(endpointDesc->protocol),
        static_cast<int>(endpointDesc->loc.locType), *value);
    return static_cast<HcommResult>(HCCL_SUCCESS);
}

#ifdef __cplusplus
}
#endif
