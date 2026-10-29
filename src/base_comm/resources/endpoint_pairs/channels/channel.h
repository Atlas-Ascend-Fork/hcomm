/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHANNEL_H
#define CHANNEL_H

#include <memory>
#include "hccl/hccl_res.h"
#include "hccl/hccl_types.h"
#include "hcomm_res_defs.h"
#include <string>
#include "i_channel_entity.h"
#include "mem_device_pub.h"
#include <mutex>
#include <vector>
#include "hccl_mem_defs.h"
#include "enum_factory.h"

// Orion
#include "transport_status.h"
#include "ip_address.h"
#include "topo_common_types.h"
#include "virtual_topo.h"
#include "socket_config.h"

#include "endpoint_pair/channel/channel_pub.h"
#include "../../../../../src/legacy/ascend950/unified_platform/resource/socket/socket.h"
#include "../../../../../src/legacy/ascend950/unified_platform/resource/stream/stream.h"
#include "../../../../../src/legacy/ascend950/unified_platform/resource/transport/base_mem_transport.h"
#include "../../../../../src/legacy/ascend950/unified_platform/resource/mem/user_remote_mem_getter.h"
#include "../../../../../src/legacy/ascend950/unified_platform/resource/notify/local_notify_v2.h"

namespace hcomm {

class Endpoint;
/**
 * @brief 通道种类（与 HcommChannelRes.channelTypeList 中 u32 数值一致；由 CommEngine + CommProtocol 推导）。
 * @note  仅 base_comm 内部使用，不出边界。
 */
enum class HcommChannelKind : uint32_t {
    INVALID = 0U,
    AICPU_TS_URMA = 1U,
    AICPU_TS_ROCE_DEV_LOC = 2U,
    AICPU_TS_HCCS = 3U,
    CPU_ROCE = 4U,
    AIV_UB_MEM = 5U,
    AICPU_TS_UBOE = 6U,
    AICPU_TS_ROCE_HOST_LOC = 7U,
    AIV_URMA = 8U,
    AICPU_TS_UB_RTP = 9U,
    AICPU_TS_PCIE = 10U,
    AIV_ROCE = 11U,
    CPU_URMA = 12U,
    CCU_URMA = 13U,
};

constexpr uint32_t DEFAULT_LISTENING_PORT = 60001;
constexpr uint32_t HCCS_DEFAULT_LISTENING_PORT = 16666;
constexpr uint32_t ROCE_V1_DEFAULT_LISTENING_PORT = 16666;

const char* HcommChannelKindToString(HcommChannelKind kind);

enum class SocketBuildMode : uint8_t {
    NONE = 0U,              // 本通道不走 socket 建链这条路。
    BORROW = 1U,            // 只借不建。 不自建监听 socket。
    LISTEN_AND_BORROW = 2U, // 在 BORROW 的基础上，StartListen() 里还会走自建监听分支。
};

enum class NotifyBuildMode : uint8_t {
    NONE = 0U,
    BUILD_LOCAL = 1U,
};

enum class EndpointIpMode : uint8_t {
    NONE = 0U,           // 建链不依赖 IP，保持默认值
    FROM_COMM_ADDR = 1U, // ranktable/topo 静态下发的 commAddr（RoCE/UB_RTP）
    FROM_VNIC = 2U,      // 运行期按设备查 VNIC IP（HCCS）
};

enum class LocalNotifyType : uint8_t {
    NONE = 0U,
    UB = 1U,  // UbLocalNotify（携 rdmaHandle）
    IPC = 2U, // IpcLocalNotify
};

struct ChannelTraits {
    uint32_t defaultListenPort{DEFAULT_LISTENING_PORT};     // port==0 时的兜底监听端口
    EndpointIpMode ipMode{EndpointIpMode::NONE};            // 建链 IP 获取方式
    SocketBuildMode socketBuildMode{SocketBuildMode::NONE}; // BuildSocket 流程
    bool reservedSocketAddrTag{false}; // RESERVED 角色时 SocketConfig 按地址 tag 构造（不含监听端口）
    bool reservedNoRankId{false};      // RESERVED 构造 SocketConfig 时是否置 noRankId=true
    Hccl::ConnectProtoType listenPortType{Hccl::ConnectProtoType::PCIE};
    NotifyBuildMode notifyBuildMode{NotifyBuildMode::NONE}; // BuildNotify 流程开关
    LocalNotifyType notifyType{LocalNotifyType::NONE};      // BUILD_LOCAL 的本地 notify 具体类型
    bool memFromDescAllowed{true}; // exchangeAllMems==false 时是否允许从 channelDesc_.memHandles 收集内存
};

/**
 * @brief 本端：本端端点信息与本端侧的建链、传输资源。
 * @note 端点字段与 ChannelRemote 同构（ep/ip），epPtr/devicePhyId 为本端独有；
 * socket/notify/rdma 等建链与传输资源均由本端持有。
 */
struct ChannelLocal {
    // ---- 本端端点 ----
    Endpoint* epPtr{nullptr}; // 本端 Endpoint 对象（endpointHandle 下转）
    EndpointDesc ep{};        // 本端端点描述
    uint32_t devicePhyId{0};  // 本端设备物理 ID
    Hccl::IpAddress ip{};     // 本端建链地址（IP 或 EID）

    // ---- 本端建链与 socket 持有 ----
    uint32_t serverPort{0};     // 监听端口
    bool isSocketServer{false}; // 为 true 代表本端 socket 为 server
    bool isRecvFirst{false};    // 借出角色派生的收发次序（client 先收后发）
    std::unique_ptr<Hccl::SocketConfig> socketConfigHolder{nullptr}; // 建链配置持有
    const Hccl::SocketConfig* socketConfig{nullptr};                 // 建链配置视图
    Hccl::Socket* socket{nullptr};                                   // 借用的数据 socket
    std::unique_ptr<Hccl::Socket> serverSocket{nullptr};             // 自建监听 socket

    // ---- 本端传输与 notify 资源 ----
    uint32_t notifyNum{0};                                          // 本地 notify 数量（channelDesc_ 折算）
    Hccl::RdmaHandle rdmaHandle{nullptr};                           // notify/传输共用的 rdma 句柄
    Hccl::BaseMemTransport::CommonLocRes commonRes{};               // 传输层公共资源
    std::vector<std::unique_ptr<Hccl::BaseLocalNotify>> notifies{}; // 本地 notify 持有
    Hccl::BaseMemTransport::Attribution attr{};                     // 握手归因（opMode/devicePhyId/handshakeMsg）

    void SaveSocketConfig(const Hccl::SocketConfig& config);
};

/**
 * @brief 对端内存缓存：GetRemoteMems 查询结果的线程安全缓存。
 * @note mutex 同时保护调用方（子类）的远端 buffer 向量与缓存副本：
 * Clean 清空 / Serialize 快照读 / UpdateMemInfo 追加等复合临界区经 mutex 加锁，
 * Invalidate 须在 mutex 临界区内调用。
 */
struct RemoteMemCache {
    // 查询路径：填充 C 形态出参（自持锁）；rmtBufferVec 为调用方（子类 transport）持有的远端 buffer 向量
    template <typename VecT>
    HcclResult Get(VecT& rmtBufferVec, uint32_t* memNum, CommMem** remoteMem, char*** memInfos);

    // 失效并清空缓存副本（须在 mutex 临界区内调用）
    void Invalidate();

    std::mutex mutex{};
    bool cacheValid{false};
    std::vector<CommMem> remoteUserMems{};
    std::vector<std::string> memInfoCopies{};
    std::vector<char*> memInfoPointers{};
};

/**
 * @brief 对端：channel 持有的对端端点描述、建链地址与对端内存在本端的缓存。
 * @note 端点字段与 ChannelLocal 同构：local_.ep ↔ ep、local_.ip ↔ ip。
 */
struct ChannelRemote {
    EndpointDesc ep{};         // 对端 Endpoint 描述
    Hccl::IpAddress ip{};      // 对端建链地址（IP 或 EID）
    RemoteMemCache memCache{}; // 对端用户内存缓存（GetRemoteMems 出参；自带锁与复合临界区契约）
};

/**
 * @note 职责：一个EndPointPair上的建立的通信通道的C++抽象接口类声明。
 * 管理该通信通道Channel对上的同步信号Notify、通信队列（如qp、jetty等）等资源管理，负责建立连接，以及注册内存、同步信号等的交换。
 * 抽象上 channel 持有本端与对端两个端点（local_/remote_），仅通道自身的身份、配置与运行状态留在基类。
 */
class Channel {
public:
    Channel() {};
    virtual ~Channel();

    // 禁拷贝（避免切片/资源重复释放等）
    Channel(const Channel&) = delete;
    Channel& operator=(const Channel&) = delete;

    // 允许移动；Channel 不支持多线程并发访问，日志去重状态以普通成员维护即可
    Channel(Channel&&) = default;
    Channel& operator=(Channel&&) = default;

    // ------------------ 控制面接口 ------------------
    virtual HcclResult Init() final;
    virtual HcclResult GetNotifyNum(uint32_t* notifyNum) const = 0;
    virtual HcclResult GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos) = 0;
    virtual ChannelStatus GetStatus() = 0;
    virtual HcclResult UpdateMemInfo(HcommMemHandle* memHandles, uint32_t memHandleNum);
    // 取本通道当前全量对端条目的 blob（当刻打包，调用方拷走即弃）；空 = 本次不下发。
    // 仅 URMA / UBOE / UB_RTP 覆写：它们的 device 侧地址表由建链 kernel 填、消费方按地址查
    virtual std::vector<char> GetMemUpdatePack() { return std::vector<char>(); }

    virtual HcclResult Clean() = 0;
    virtual HcclResult Resume() = 0;

    HcommChannelKind GetChannelKind() const { return channelKind_; }
    CommEngine GetEngine() const { return engine_; }
    virtual const HcommChannelDesc& GetChannelDesc() const { return channelDesc_; }

    bool IsSharedQueue() const { return isSharedQueue_; }

    virtual HcclResult Serialize(std::shared_ptr<hccl::DeviceMem>& out);

    // ------------------ 数据面接口 ------------------
    virtual HcclResult NotifyRecord(const uint32_t remoteNotifyIdx);
    virtual HcclResult NotifyWait(const uint32_t localNotifyIdx, const uint32_t timeout);
    virtual HcclResult WriteWithNotify(void* dst, const void* src, const uint64_t len, uint32_t remoteNotifyIdx);
    virtual HcclResult Write(void* dst, const void* src, uint64_t len);
    virtual HcclResult Read(void* dst, const void* src, uint64_t len);
    virtual HcclResult ChannelFence();
    virtual HcclResult ChannelDrain() { return HCCL_E_NOT_SUPPORT; }

    virtual IChannelEntity* GetChannelEntity() { return nullptr; }

    // ------------------ 工厂 ------------------
    static HcclResult CreateChannel(
        EndpointHandle endpointHandle, CommEngine engine, const HcommChannelDesc& channelDesc,
        HcommChannelConfig config, std::shared_ptr<Channel>& out);

protected:
    Channel(
        EndpointHandle endpointHandle, const HcommChannelDesc& channelDesc, CommEngine engine,
        HcommChannelKind channelKind, const char* channelClassName);

    virtual HcclResult ParseInputParam();

    // role → local_.isSocketServer 的角色判定hook：基类默认按 SERVER/CLIENT 直译（RESERVED 等其余角色保持默认 false），
    // 子类角色语义不同时覆写（如 urma 的 role!=CLIENT、roce 的 IP 字典序兜底、host_roce 的 exchangeAllMems 联动）。
    virtual HcclResult DecideSocketRole();
    virtual HcclResult StartListen();
    virtual HcclResult BuildSocket();
    virtual HcclResult BuildConnection();
    virtual HcclResult BuildNotify();
    virtual HcclResult BuildBuffer();

    HcclResult CollectLocalRmaBuffers(std::vector<Hccl::LocalRmaBuffer*>& bufferVec);

    HcclResult GetEndpointIp(const EndpointDesc& ep, Hccl::IpAddress& ip);
    HcclResult GetFirstIpByPhyId(uint32_t devicePhyId, uint32_t superDevId, Hccl::IpAddress& ip) const;

    SocketBuildMode GetSocketBuildMode() const { return traits_.socketBuildMode; }
    Hccl::DevNetPortType GetListenPortType() const { return Hccl::DevNetPortType(traits_.listenPortType); }
    NotifyBuildMode GetNotifyBuildMode() const { return traits_.notifyBuildMode; }
    std::unique_ptr<Hccl::BaseLocalNotify> CreateLocalNotify(bool devUsed) const;

    // ------------------ 工具方法 ------------------
    ChannelStatus TransportStatusToChannelStatus(
        Hccl::TransportStatus ts, const EndpointDesc& localEp, const HcommChannelDesc& channelDesc);

    void PutSocketIfNeeded();

    // ================== channel 自身（身份、配置与运行状态）==================
    HcommChannelKind channelKind_{HcommChannelKind::INVALID}; // 通道种类
    CommEngine engine_{COMM_ENGINE_RESERVED};                 // 所属引擎
    const char* channelClassName_{nullptr};
    EndpointHandle endpointHandle_{nullptr}; // endpoint_pairs 框架句柄
    HcommChannelDesc channelDesc_{};         // 通道描述
    ChannelTraits traits_{};                 // kind 折算能力集
    ChannelStatus channelStatus_{ChannelStatus::INIT};
    Hccl::TransportStatus lastTransportStatus_{Hccl::TransportStatus::__COUNT__}; // 传输状态日志去重

    // ================== 本端 ==================
    ChannelLocal local_{}; // 本端端点、建链与传输资源

    // ================== 对端 ==================
    ChannelRemote remote_{}; // 对端端点、建链地址与内存缓存

    // ------------------ 建链交换数据校验 ------------------
    // 建链握手阶段交换数据的最大长度，对端声明长度为0或超过该值视为异常，防止超大内存申请导致OOM
    // 取值依据：buffer条数上限MAX_BUFFER_NUM(30000)，单条DTO(addr/size/rkey/memInfo)百字节量级，正常仅数MB，64MB预留约一个数量级余量
    static constexpr uint64_t MAX_EXCHANGE_DATA_SIZE = 64ULL * 1024 * 1024; // 64MB

private:
    bool isSharedQueue_{false}; // 共享队列（共享 jetty）开关
    void SetSharedQueue(bool isSharedQueue) { isSharedQueue_ = isSharedQueue; }
};

// ================== 通道状态方法实现 ==================
inline const char* HcommChannelKindToString(HcommChannelKind kind)
{
    switch (kind) {
        case HcommChannelKind::INVALID:
            return "INVALID";
        case HcommChannelKind::AICPU_TS_URMA:
            return "AICPU_TS_URMA";
        case HcommChannelKind::AICPU_TS_ROCE_DEV_LOC:
            return "AICPU_TS_ROCE_DEV_LOC";
        case HcommChannelKind::AICPU_TS_HCCS:
            return "AICPU_TS_HCCS";
        case HcommChannelKind::CPU_ROCE:
            return "CPU_ROCE";
        case HcommChannelKind::AIV_UB_MEM:
            return "AIV_UB_MEM";
        case HcommChannelKind::AICPU_TS_UBOE:
            return "AICPU_TS_UBOE";
        case HcommChannelKind::AICPU_TS_ROCE_HOST_LOC:
            return "AICPU_TS_ROCE_HOST_LOC";
        case HcommChannelKind::AIV_URMA:
            return "AIV_URMA";
        case HcommChannelKind::AICPU_TS_UB_RTP:
            return "AICPU_TS_UB_RTP";
        case HcommChannelKind::AICPU_TS_PCIE:
            return "AICPU_TS_PCIE";
        case HcommChannelKind::AIV_ROCE:
            return "AIV_ROCE";
        case HcommChannelKind::CPU_URMA:
            return "CPU_URMA";
        case HcommChannelKind::CCU_URMA:
            return "CCU_URMA";
        default:
            return "UNKNOWN";
    }
}

inline void ChannelLocal::SaveSocketConfig(const Hccl::SocketConfig& config)
{
    socketConfigHolder = std::make_unique<Hccl::SocketConfig>(config);
    socketConfig = socketConfigHolder.get();
}

template <typename VecT>
HcclResult RemoteMemCache::Get(VecT& rmtBufferVec, uint32_t* memNum, CommMem** remoteMem, char*** memInfos)
{
    std::lock_guard<std::mutex> lock(mutex);
    Hccl::RemoteMemCtx<typename VecT::value_type> remoteMemCtx{
        cacheValid, rmtBufferVec, remoteUserMems, memInfoCopies, memInfoPointers, remoteMem, memInfos, memNum};
    return Hccl::GetRemoteUserMems(remoteMemCtx);
}

inline void RemoteMemCache::Invalidate()
{
    cacheValid = false;
    remoteUserMems.clear();
    memInfoCopies.clear();
    memInfoPointers.clear();
}

} // namespace hcomm
#endif // CHANNEL_H
