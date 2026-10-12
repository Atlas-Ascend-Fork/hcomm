/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LEVEL1_PROXY_COMMON_H
#define LEVEL1_PROXY_COMMON_H

#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

/*
 * hccl_rank_graph.h会间接包含hcomm_res_defs.h，其中HcommChannelDescInit是
 * static inline函数；北向代理同时保留了同名导出桩。这里只为取得拓扑查询
 * 使用的HComm数据类型，临时重命名头文件内联函数，避免公共头的所有使用者
 * 与导出桩发生重复定义。宏在包含完成后立即取消，不影响其他代码。
 */
#define HcommChannelDescInit HcommChannelDescInitHeaderInline
#include "hccl/hccl_rank_graph.h"
#undef HcommChannelDescInit

#include "runtime_state/sim_models.h"

namespace sim {

// Level1 task generation uses its original single-communicator model rather
// than level2's HcclComm handle mapping.
uint32_t GetCurrRankId();

/**
 * @brief Rank table中记录的网络拓扑类型。
 */
enum class RankTableTopoType { CLOS = 0, MESH_1D, MESH_2D, A3_SERVER, A2_AX_SERVER, CUSTOM };

/**
 * @brief 拓扑文件中一条边的连接类型。
 */
enum class TopoLinkType { PEER_TO_PEER = 0, PEER_TO_NETWORK };

/**
 * @brief 拓扑文件中连接接口所在的位置。
 */
enum class TopoAddressPosition { DEVICE = 0, HOST };

/**
 * @brief topo.json中一条连接边的结构化信息。
 */
struct TopoEdgeInfo {
    uint32_t netLayer = 0;                /**< 连接所属网络层 */
    uint32_t topoInstanceId = 0;          /**< 连接所属拓扑实例 */
    RankTableTopoType topoType;           /**< 拓扑实例类型 */
    TopoLinkType linkType;                /**< 点到点或点到网络连接 */
    TopoAddressPosition position;         /**< 接口位于Device侧或Host侧 */
    uint32_t localA = 0;                  /**< A端本地ID */
    std::vector<std::string> localAPorts; /**< A端使用的端口 */
    uint32_t localB = 0;                  /**< B端本地ID，仅点到点连接使用 */
    std::vector<std::string> localBPorts; /**< B端使用的端口，仅点到点连接使用 */
    std::vector<std::string> protocols;   /**< 连接支持的协议，已去重 */
};

/**
 * @brief Rank在一个网络层中使用的一条通信地址信息。
 */
struct RankAddressInfo {
    std::string address;            /**< 地址文本，对应addr字段 */
    std::string addressType;        /**< 地址类型，对应addr_type字段 */
    std::string planeId;            /**< 网络平面标识，对应plane_id字段 */
    std::vector<std::string> ports; /**< 该地址对应的端口列表 */
};

/**
 * @brief Rank在一个网络层中的完整拓扑信息。
 *
 * rank table字段和topo.json连接边在Load阶段合并到同一个结构中，
 * 查询阶段只需要读取m_rankList。
 */
struct RankLevelInfo {
    uint32_t netLayer = 0;                        /**< 网络层编号，对应net_layer字段 */
    std::string netInstanceId;                    /**< 网络实例标识，对应net_instance_id字段 */
    std::string netType;                          /**< 网络类型，对应net_type字段 */
    std::string netAttr;                          /**< 网络属性，对应net_attr字段 */
    std::vector<RankAddressInfo> rankAddressList; /**< 对应rank_addr_list数组 */
    std::vector<TopoEdgeInfo> topoEdgeList;       /**< 该网络实例从topo.json获得的连接边 */
    /**
     * 本 rank 在该网络层所属的拓扑实例 ID 列表（来自 topo.json 的实例声明）。
     *
     * 归属判定依据是实例的 local id 集合是否包含本 rank 的 localId，而**不是**本 rank
     * 是否在该实例内有物理边：单 rank 网络实例（如独占 server 的 rank）没有边，但仍是
     * 一个合法的拓扑实例（HComm 记为 0 号实例）。
     */
    std::vector<uint32_t> topoInstanceIds;
};

/**
 * @brief Rank表中单个rank的信息结构体。
 */
struct RankInfo {
    uint32_t rankId = 0;                  /**< rank的ID，对应rank_id字段 */
    uint32_t deviceId = 0;                /**< 设备ID，对应device_id字段 */
    uint32_t localId = 0;                 /**< 本地ID，对应local_id字段 */
    std::vector<RankLevelInfo> levelList; /**< rank table和topo.json合并后的层信息 */
};

/**
 * @brief RankTable拓扑信息查询结果。
 */
enum class RankTableQueryStatus {
    SUCCESS = 0,
    LOAD_FAILED,
    RANK_NOT_FOUND,
    LAYER_NOT_FOUND,
    TOPO_INSTANCE_NOT_FOUND,
    ENDPOINT_NOT_FOUND,
    UNSUPPORTED_TYPE,
    INVALID_PARAMETER,
    INVALID_DATA
};

/**
 * @brief 一个可以对外描述的rank通信端点。
 *
 * 每条记录对应“一个通信接口上的一个协议”。同一接口支持两个协议时，
 * 会生成两条端点记录。
 */
struct RankEndpointInfo {
    std::string address;            /**< 地址文本 */
    std::string addressType;        /**< EID、IPV4、IPV6或ID */
    std::string protocol;           /**< 该端点使用的通信协议 */
    std::vector<std::string> ports; /**< 接口包含的端口 */
    TopoAddressPosition position;   /**< 接口位置 */
    uint32_t deviceId = 0;          /**< rank对应的物理设备ID */
    uint32_t localDieId = 0;        /**< 从端口编号得到的本地Die ID */
};

/**
 * @brief 一条源rank到目的rank的结构化通信连接。
 */
struct RankLinkInfo {
    RankEndpointInfo sourceEndpoint;      /**< 源端点 */
    RankEndpointInfo destinationEndpoint; /**< 目的端点 */
    std::string protocol;                 /**< 链路协议 */
    uint8_t hop = 1;                      /**< 链路跳数 */
};

/**
 * @brief RankTable单例类，用于解析和管理rank_table.json。
 *
 * Rank table文件是一个JSON格式的文件，配置了参与集合通信的NPU资源信息。
 * 该类在Load阶段完成解析和数据合并，后续查询只读取缓存。
 * 当前实现按照单线程调用场景设计，只有Load阶段使用互斥锁。
 *
 * 使用示例：
 * @code
 *   sim::RankTable& rt = sim::RankTable::Instance();
 *   if (rt.Load(rankTableFile)) {
 *       uint32_t rankSize = rt.GetRankSize();
 *       int deviceId = rt.GetDeviceId(rank);
 *   }
 * @endcode
 */
class RankTable {
public:
    /**
     * @brief 获取RankTable单例实例。
     * @return RankTable单例引用。
     */
    static RankTable& Instance();

    RankTable(const RankTable&) = delete;
    RankTable& operator=(const RankTable&) = delete;

    /**
     * @brief 加载rank table数据。
     * @param clusterInfo Rank table的JSON内容（以'{'开头）或文件路径。
     * @return 成功返回true，失败返回false。
     * @note 已加载后再次调用为no-op，返回true。多次调用是线程安全的。
     */
    bool Load(const char* clusterInfo);

    /**
     * @brief 确保rank table已经加载。
     * @return 已加载或从RANK_TABLE_FILE加载成功返回true，否则返回false。
     * @note 若尚未加载，则读取RANK_TABLE_FILE环境变量指定的文件。
     */
    bool EnsureLoaded();

    /**
     * @brief 查询rank table是否已加载。
     * @return 已加载返回true，否则返回false。
     */
    bool IsLoaded() const;

    /**
     * @brief 获取指定rank包含的网络层编号。
     * @param rankId rank的ID。
     * @param netLayers 输出参数，成功时保存排序并去重后的网络层编号。
     * @return RankTable拓扑信息查询结果，可区分加载失败、rank不存在和数据异常。
     * @note 返回的是数据副本，调用者不接触RankTable内部容器。
     */
    RankTableQueryStatus GetNetLayers(uint32_t rankId, std::vector<uint32_t>& netLayers);

    /**
     * @brief 获取指定rank在指定网络层所属网络实例中的所有rank。
     * @param rankId 当前rank的ID。
     * @param netLayer 网络层编号。
     * @param ranks 输出参数，成功时保存排序并去重后的rank编号。
     * @return RankTable拓扑信息查询结果。
     * @note 返回的是数据副本，调用者不接触RankTable内部容器。
     */
    RankTableQueryStatus GetRanksByLayer(uint32_t rankId, uint32_t netLayer, std::vector<uint32_t>& ranks);

    /**
     * @brief 获取指定rank在指定网络层所属网络实例中的rank数量。
     * @param rankId 当前rank的ID。
     * @param netLayer 网络层编号。
     * @param rankNum 输出参数，成功时保存当前网络实例中的rank数量。
     * @return RankTable拓扑信息查询结果。
     */
    RankTableQueryStatus GetRankSizeByLayer(uint32_t rankId, uint32_t netLayer, uint32_t& rankNum);

    /**
     * @brief 获取指定网络层中所有网络实例的rank数量列表。
     * @param rankId 当前rank的ID，用于确认当前通信域包含指定网络层。
     * @param netLayer 网络层编号。
     * @param instSizeList 输出参数，成功时按网络实例ID排序保存各实例的rank数量。
     * @return RankTable拓扑信息查询结果。
     * @note 返回的是数据副本，调用者不接触RankTable内部容器。
     */
    RankTableQueryStatus
    GetInstSizeListByLayer(uint32_t rankId, uint32_t netLayer, std::vector<uint32_t>& instSizeList);

    /**
     * @brief 获取指定rank在指定网络层中的拓扑类型。
     * @param rankId 当前rank的ID。
     * @param netLayer 网络层编号。
     * @param topoType 输出参数，成功时保存该网络层的拓扑类型。
     * @return RankTable拓扑信息查询结果。
     * @note TOPO_FILE_DESC表示用户自定义拓扑，对外对应CUSTOM类型。
     */
    RankTableQueryStatus GetTopoTypeByLayer(uint32_t rankId, uint32_t netLayer, RankTableTopoType& topoType);

    /**
     * @brief 获取指定rank在指定网络层中的拓扑实例ID列表。
     * @param rankId 当前rank的ID。
     * @param netLayer 网络层编号。
     * @param topoInstanceIds 输出参数，成功时保存排序并去重后的拓扑实例ID。
     * @return RankTable拓扑信息查询结果。
     * @note 该查询只支持rank table中net_type为TOPO_FILE_DESC的网络层，
     *       拓扑实例ID从配套的topo.json中读取。
     */
    RankTableQueryStatus
    GetTopoInstsByLayer(uint32_t rankId, uint32_t netLayer, std::vector<uint32_t>& topoInstanceIds);

    /**
     * @brief 获取指定网络层和拓扑实例的拓扑类型。
     * @param rankId 当前rank的ID。
     * @param netLayer 网络层编号。
     * @param topoInstanceId 拓扑实例ID。
     * @param topoType 输出参数，成功时保存拓扑实例类型。
     * @return RankTable拓扑信息查询结果。
     * @note 该查询只支持rank table中net_type为TOPO_FILE_DESC的网络层，
     *       实例ID和实例类型从配套的topo.json中读取。
     */
    RankTableQueryStatus
    GetTopoType(uint32_t rankId, uint32_t netLayer, uint32_t topoInstanceId, RankTableTopoType& topoType);

    /**
     * @brief 获取当前rank所在的指定拓扑实例包含的rank列表。
     * @param rankId 当前rank的ID。
     * @param netLayer 网络层编号。
     * @param topoInstanceId 拓扑实例ID。
     * @param ranks 输出参数，保存排序并去重后的rank编号。
     * @return RankTable拓扑信息查询结果。
     */
    RankTableQueryStatus
    GetRanksByTopoInst(uint32_t rankId, uint32_t netLayer, uint32_t topoInstanceId, std::vector<uint32_t>& ranks);

    /**
     * @brief 获取当前rank在指定拓扑实例上的端点数量。
     * @param rankId 当前rank的ID。
     * @param netLayer 网络层编号。
     * @param topoInstanceId 拓扑实例ID。
     * @param endpointNum 输出参数，保存匹配接口所支持的协议数量之和。
     * @return RankTable拓扑信息查询结果。
     * @note 拓扑实例没有匹配端点时返回SUCCESS，并将endpointNum设为0。
     */
    RankTableQueryStatus
    GetEndpointNum(uint32_t rankId, uint32_t netLayer, uint32_t topoInstanceId, uint32_t& endpointNum);

    /**
     * @brief 获取当前rank在指定拓扑实例上的端点描述数据。
     * @param rankId 当前rank的ID。
     * @param netLayer 网络层编号。
     * @param topoInstanceId 拓扑实例ID。
     * @param endpoints 输出参数，每项对应一个接口协议组合。
     * @return RankTable拓扑信息查询结果。
     */
    RankTableQueryStatus GetEndpointList(
        uint32_t rankId, uint32_t netLayer, uint32_t topoInstanceId, std::vector<RankEndpointInfo>& endpoints);

    /**
     * @brief 根据地址和协议查找指定rank的端点信息。
     * @param rankId 端点所属rank的ID。
     * @param addressType 地址类型。
     * @param address 地址文本。
     * @param protocol 通信协议名称。
     * @param endpoint 输出参数，保存匹配的端点信息。
     * @return RankTable拓扑信息查询结果。
     */
    RankTableQueryStatus FindEndpoint(
        uint32_t rankId, const std::string& addressType, const std::string& address, const std::string& protocol,
        RankEndpointInfo& endpoint);

    /**
     * @brief 查询两个rank在指定网络层中的通信连接。
     * @param rankId 当前通信域的本rank，用于检查网络层是否合法。
     * @param netLayer 网络层编号。
     * @param sourceRank 源rank编号。
     * @param destinationRank 目的rank编号。
     * @param links 输出参数，保存结构化链路列表。
     * @return RankTable拓扑信息查询结果。
     */
    RankTableQueryStatus GetLinks(
        uint32_t rankId, uint32_t netLayer, uint32_t sourceRank, uint32_t destinationRank,
        std::vector<RankLinkInfo>& links);

    /**
     * @brief 查询网络层拓扑类型，并直接转换成HComm公开类型。
     */
    RankTableQueryStatus GetCommTopoTypeByLayer(uint32_t rankId, uint32_t netLayer, CommTopo& topoType);

    /**
     * @brief 查询拓扑实例类型，并直接转换成HComm公开类型。
     */
    RankTableQueryStatus
    GetCommTopoType(uint32_t rankId, uint32_t netLayer, uint32_t topoInstanceId, CommTopo& topoType);

    /**
     * @brief 查询结构化链路，并直接转换成HComm的CommLink列表。
     */
    RankTableQueryStatus GetCommLinks(
        uint32_t rankId, uint32_t netLayer, uint32_t sourceRank, uint32_t destinationRank,
        std::vector<CommLink>& links);

    /**
     * @brief 查询端点，并直接转换成HComm的EndpointDesc列表。
     */
    RankTableQueryStatus GetCommEndpointDescList(
        uint32_t rankId, uint32_t netLayer, uint32_t topoInstanceId, std::vector<EndpointDesc>& endpointDescList);

    /**
     * @brief 根据HComm端点描述查询并返回指定端点属性。
     */
    RankTableQueryStatus GetCommEndpointInfo(
        uint32_t rankId, const EndpointDesc& endpointDesc, EndpointAttr endpointAttr, uint32_t infoLen, void* info);

    /**
     * @brief 获取rank table中的rank总数。
     * @return rank的总数。
     */
    uint32_t GetRankSize() const;

    /**
     * @brief 根据rank获取对应的device_id。
     * @param rank rank的ID。
     * @return 对应的device_id，失败返回-1。
     */
    int GetDeviceId(uint32_t rank) const;

    /**
     * @brief 根据device_id获取对应的rank_id。
     * @param deviceId 设备ID。
     * @return 对应的rank_id，未找到返回-1。
     */
    int GetRankIdByDeviceId(uint32_t deviceId) const;

    /**
     * @brief 根据rank获取完整的RankInfo信息。
     * @param rank rank的ID。
     * @param info 输出参数，成功时填充RankInfo结构体。
     * @return RankTable拓扑信息查询结果。
     */
    RankTableQueryStatus GetRankInfo(uint32_t rank, RankInfo& info) const;

    /**
     * @brief 重置内部状态，清除已加载的数据。
     * @note 主要用于单元测试场景。
     */
    void Reset();

private:
    RankTable();
    ~RankTable() = default;

    /**
     * @brief 从JSON字符串解析rank table。
     * @param jsonStr JSON格式的字符串。
     * @return 成功返回true，失败返回false。
     */
    bool ParseFromJsonString(const std::string& jsonStr);

    /**
     * @brief 在Load阶段解析配套topo.json，并将边信息合并进m_rankList。
     */
    bool LoadTopologyIntoRankList();

    static bool HexCharToValue(char character, uint8_t& value);
    static HcclResult SetCommAddress(const RankEndpointInfo& endpoint, CommAddr& commAddr);
    static CommProtocol GetCommProtocol(const std::string& protocol);
    static HcclResult SetEndpointDesc(const RankEndpointInfo& endpoint, EndpointDesc& endpointDesc);
    static std::string GetProtocolName(CommProtocol protocol);
    static HcclResult GetCommAddressText(const CommAddr& commAddr, std::string& addressType, std::string& address);
    static RankTableQueryStatus ConvertTopoType(RankTableTopoType internalType, bool allowCustom, CommTopo& topoType);

    std::mutex m_mutex;               /**< 仅保护Load阶段的初始化 */
    bool m_loaded = false;            /**< 是否已成功加载 */
    bool m_topologyLoaded = false;    /**< 配套拓扑文件是否已成功加载 */
    uint32_t m_rankSize = 0;          /**< rank的总数 */
    std::vector<RankInfo> m_rankList; /**< rank table和topo.json的唯一数据容器 */
    std::string m_rankTablePath;      /**< rank table文件路径 */
    /** topo.json 声明的拓扑实例类型：(netLayer, topoInstanceId) -> 拓扑类型。 */
    std::map<std::pair<uint32_t, uint32_t>, RankTableTopoType> m_topoInstanceTypes;
};

} // namespace sim

/* CANN 类型兜底声明：仅当对应 CANN 头文件未被包含时提供定义。
 * 各 guard 与 CANN 头文件自身的 include guard 对齐，避免与 CANN 同时存在时重定义冲突。
 *   hccl/hccl_types.h     -> HCCL_TYPES_H_
 *   hccl/hcomm_res_defs.h -> HCOMM_RES_DEFS_H
 *   hccl/hcomm_channel.h  -> HCOMM_CHANNEL_H
 *   hccl/hccl_comm.h      -> HCCL_COMM_H_
 *   hccl/hccl_res.h       -> HCCL_RES_H */

/* 来自 hccl/hccl_types.h */
#ifndef HCCL_TYPES_H_
typedef void* HcclCommSymWindow;
typedef enum {
    HCCL_COMM_STATUS_READY = 0,
    HCCL_COMM_STATUS_SUSPENDED = 1,
    HCCL_COMM_STATUS_ERROR = 2,
    HCCL_COMM_STATUS_RESERVED
} HcclCommStatus;
#endif /* HCCL_TYPES_H_ */

/* 来自 hccl/hcomm_res_defs.h */
#ifndef HCOMM_RES_DEFS_H
typedef int32_t HcommResult;

typedef enum {
    COMM_PROTOCOL_RESERVED = -1,
    COMM_PROTOCOL_HCCS = 0,
    COMM_PROTOCOL_ROCE = 1,
} CommProtocol;

typedef uint64_t ChannelHandle;

typedef enum {
    COMM_MEM_TYPE_INVALID = -1,
    COMM_MEM_TYPE_DEVICE = 0,
    COMM_MEM_TYPE_HOST = 1,
} CommMemType;
typedef struct {
    CommMemType type;
    void* addr;
    uint64_t size;
} CommMem;

typedef uint64_t ThreadHandle;

typedef enum {
    COMM_ENGINE_RESERVED = -1,
    COMM_ENGINE_CPU = 0,
    COMM_ENGINE_CPU_TS = 1,
    COMM_ENGINE_AICPU = 2,
    COMM_ENGINE_AICPU_TS = 3,
    COMM_ENGINE_AIV = 4,
    COMM_ENGINE_CCU = 5,
} CommEngine;

typedef void* EndpointHandle;
typedef void* HcommMemHandle;
typedef void* HcommSocket;

static const uint32_t COMM_ADDR_EID_LEN = 16u;
typedef enum {
    COMM_ADDR_TYPE_RESERVED = -1,
    COMM_ADDR_TYPE_IPV4 = 0,
    COMM_ADDR_TYPE_IPV6 = 1,
    COMM_ADDR_TYPE_EID = 2,
} CommAddrType;
typedef struct {
    CommAddrType type;
    union {
        uint8_t raws[36];
        uint32_t id;
        uint8_t eid[COMM_ADDR_EID_LEN];
    };
} CommAddr;

typedef enum {
    ENDPOINT_LOC_TYPE_RESERVED = -1,
    ENDPOINT_LOC_TYPE_DEVICE = 0,
    ENDPOINT_LOC_TYPE_HOST = 1,
} EndpointLocType;
typedef struct {
    EndpointLocType locType;
    union {
        uint8_t raws[60];
        struct {
            uint32_t devPhyId;
            uint32_t superDevId;
            uint32_t serverIdx;
            uint32_t superPodIdx;
        } device;
        struct {
            uint32_t id;
        } host;
    };
} EndpointLoc;

typedef struct {
    CommProtocol protocol;
    CommAddr commAddr;
    EndpointLoc loc;
    union {
        uint8_t raws[52];
    };
} EndpointDesc;

typedef struct {
    uint32_t version;
    uint32_t magicWord;
    uint32_t size;
    uint32_t reserved;
} CommAbiHeader;

typedef enum {
    HCOMM_SOCKET_ROLE_RESERVED = -1,
    HCOMM_SOCKET_ROLE_CLIENT = 0,
    HCOMM_SOCKET_ROLE_SERVER = 1,
} HcommSocketRole;

static const uint32_t HCOMM_CHANNEL_MAGIC_WORD = 0x0fcf0f0fu;
static const uint32_t HCOMM_CHANNEL_VERSION_ONE = 1u;
static const uint32_t HCOMM_CHANNEL_VERSION = 3u;

typedef struct {
    CommAbiHeader header;
    EndpointDesc remoteEndpoint;
    uint32_t notifyNum;
    bool exchangeAllMems;
    HcommMemHandle* memHandles;
    uint32_t memHandleNum;
    HcommSocket socket;
    HcommSocketRole role;
    uint16_t port;
    union {
        uint8_t raws[128];
        struct {
            uint32_t queueNum;
            uint32_t retryCnt;
            uint32_t retryInterval;
            uint8_t tc;
            uint8_t sl;
            uint32_t qpThreshold;
        } roceAttr;
        struct {
            uint32_t qos;
        } hccsAttr;
        struct {
            uint32_t sqDepth;
        } ubAttr;
    };
    uint32_t qos;
    const char* channelName;
} HcommChannelDesc;

typedef enum {
    HCOMM_ENDPOINT_FEATURE_INVALID = -1,
    HCOMM_ENDPOINT_FEATURE_NDA = 0,
} HcommEndpointFeatureType;
#endif /* HCOMM_RES_DEFS_H */

/* 来自 hccl/hcomm_channel.h（HcommChannelGetStatus 出参状态码） */
#ifndef HCOMM_CHANNEL_H
typedef enum {
    HCOMM_CHANNEL_STATUS_READY = 0,           /* 建链完成，通道就绪 */
    HCOMM_CHANNEL_STATUS_CONNECTING = 1,      /* 建链进行中，需继续轮询 */
    HCOMM_CHANNEL_STATUS_FAILED_INTERNAL = 2, /* 建链失败 */
} HcommChannelStatus;
#endif /* HCOMM_CHANNEL_H */

/* 来自 hccl/hccl_comm.h */
#ifndef HCCL_COMM_H_
typedef enum {
    HCCL_OP_EXPANSION_MODE_INVALID = -1,
    HCCL_OP_EXPANSION_MODE_AI_CPU = 0,
    HCCL_OP_EXPANSION_MODE_AIV = 1,
    HCCL_OP_EXPANSION_MODE_HOST = 2,
    HCCL_OP_EXPANSION_MODE_HOST_TS = 3,
    HCCL_OP_EXPANSION_MODE_CCU_MS = 4,
    HCCL_OP_EXPANSION_MODE_CCU_SCHED = 5,
    HCCL_OP_EXPANSION_AIV_ONLY = 6
} HcclOpExpansionMode;
typedef HcclOpExpansionMode HcclConfigTypeOpExpansionMode;
typedef enum {
    HCCL_CONFIG_TYPE_INVALID = -1,
    HCCL_CONFIG_TYPE_OP_EXPANSION_MODE = 0,
    HCCL_CONFIG_TYPE_HCCL_ALGO = 1,
    HCCL_CONFIG_TYPE_UB_MULTI_CHANNEL_NUM = 2,
    /* 新增项：与 CANN 9.3 的 hccl_comm.h 保持一致（新 libhccl 的 GetDeterministicLevel
     * 在 hcomm 版本高于 9.2.0.1 时会查 DETERMINISTIC；UDI 为通信域自定义标识）。 */
    HCCL_CONFIG_TYPE_DETERMINISTIC = 3,
    HCCL_CONFIG_TYPE_UDI = 4
} HcclConfigType;
#endif /* HCCL_COMM_H_ */

/* 来自 hccl/hccl_res.h
 * ThreadResTypeStream 在 CANN 中为 typedef aclrtStream，而 aclrtStream 本身即 void*，
 * 此处直接用 void* 使本头文件无需依赖 acl/acl_rt.h。 */
#ifndef HCCL_RES_H
typedef enum {
    THREAD_RES_TYPE_INVALID = -1,
    THREAD_RES_TYPE_STREAM = 0,
} ThreadResType;
typedef void* ThreadResTypeStream;
typedef void* HcclMemHandle;
#endif /* HCCL_RES_H */

/* 来自 hcomm/hcomm_primitives.h */
#ifndef HCOMM_PRIMITIVES_H
typedef enum {
    HCOMM_REDUCE_SUM = 0,
    HCOMM_REDUCE_PROD = 1,
    HCOMM_REDUCE_MAX = 2,
    HCOMM_REDUCE_MIN = 3,
    HCOMM_REDUCE_RESERVED = 255
} HcommReduceOp;

typedef enum {
    HCOMM_DATA_TYPE_INT8 = 0,
    HCOMM_DATA_TYPE_INT16 = 1,
    HCOMM_DATA_TYPE_INT32 = 2,
    HCOMM_DATA_TYPE_FP16 = 3,
    HCOMM_DATA_TYPE_FP32 = 4,
    HCOMM_DATA_TYPE_INT64 = 5,
    HCOMM_DATA_TYPE_UINT64 = 6,
    HCOMM_DATA_TYPE_UINT8 = 7,
    HCOMM_DATA_TYPE_UINT16 = 8,
    HCOMM_DATA_TYPE_UINT32 = 9,
    HCOMM_DATA_TYPE_FP64 = 10,
    HCOMM_DATA_TYPE_BFP16 = 11,
    HCOMM_DATA_TYPE_INT128 = 12,
    HCOMM_DATA_TYPE_HIF8 = 14,
    HCOMM_DATA_TYPE_FP8E4M3 = 15,
    HCOMM_DATA_TYPE_FP8E5M2 = 16,
    HCOMM_DATA_TYPE_FP8E8M0 = 17,
    HCOMM_DATA_TYPE_RESERVED = 255
} HcommDataType;
#endif /* HCOMM_PRIMITIVES_H */

/**
 * @brief 将 ctxTag 字符串哈希为 uint64_t，nullptr 返回 0。
 */
inline uint64_t HashCtxTag(const char* tag)
{
    if (tag == nullptr) {
        return 0;
    }
    return static_cast<uint64_t>(std::hash<std::string>{}(std::string(tag)));
}

/**
 * @brief 通信引擎上下文全局注册表（单例）。
 *
 * 以 Device 内存地址（addr）为 key、sim::runtime::HcclEngineCtx 为 value 维护全局 map，
 * 替代原有的 DB 表存储。负责 aclrtMalloc/aclrtFree 以及 map 的增删查。
 * 线程安全，所有方法内部加锁。
 */
class EngineCtxRegistry {
public:
    static EngineCtxRegistry& Instance();

    HcclResult Create(uint64_t commId, const char* ctxTag, CommEngine engine, uint64_t size, void** ctx);
    bool Get(uint64_t commId, uint64_t ctxTag, uint64_t engineVal, void** ctx, uint64_t* size);
    HcclResult Copy(
        uint64_t commId, uint64_t ctxTag, uint64_t engineVal, const void* srcCtx, uint64_t size, uint64_t dstCtxOffset);
    uint32_t Destroy(uint64_t commId, uint64_t ctxTag, uint64_t engineVal);
    void DestroyByCommId(uint64_t commId);
    void ResetAll();

private:
    std::mutex m_mutex;
    std::unordered_map<uint64_t, sim::runtime::HcclEngineCtx> m_map;
};

void EngineCtxDestroyByCommId(uint64_t commId);

#endif // LEVEL1_PROXY_COMMON_H
