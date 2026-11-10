/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topo_ascend_cluster_parser.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <map>
#include "runtime_state/db_sim_runner_common.h"
#include <sstream>
#include <string>
#include <unordered_map>
#include <filesystem>
#include <system_error>

namespace fs = std::filesystem;

#include "topo_cluster_dumper.h"
#include "sim_common_macro.h"
#include "sim_log.h"
#include "sim_common_api.h"
#include "sim_yaml_config.h"
#include "sim_ip_address.h"
#include "runtime_state/db_sim_runner_ops.h"
#include "storage/table_access.h"

namespace {
// EID(128bit)地址分配规则(位序从高到低):
//   127~92: 保留位, 固定全0
//   91~88:  super pod id[8:5] (super pod id的第8~5位, 供超节点数>32时追踪;
//            IP侧superpod段AA按新规则从1起编: 默认模式AA=sp_id+1,
//            动态模式按AA.BB联合编号V=((sp_id+1)<<W_bb)+(srv_id+1)拆解)
//   87~86:  die id
//   85~80:  端口号: 按文档"PG端口和物理端口说明"划分端口类型:
//           - PG端口: 仅当port_group对应ports列表中有多个物理端口时, 固定为0x3F
//           - 普通物理端口: 单个"die/port"端口的portGroup或未在port_group中指定的端口,
//             端口号取物理端口号(0-62); d2h等端口名不带端口序号的端口, 端口号取
//             配置文件pin_map中value=3(d2h端口)对应的portId
//   79~32:  保留位, 固定全0
//   31~0:   IPV4地址 (AA.BB.CC.DD)
// 对应16字节EID数组(高位字节在前): eid[4]低4位为super pod id[8:5],
// eid[5]高2位为die id, eid[5]低6位为端口号, eid[12]~eid[15]为IPV4地址
constexpr uint8_t EID_SUPERPOD_ID_HIGH_MASK = 0x0F; // eid[4]低4位: super pod id[8:5]
constexpr uint8_t EID_DIE_ID_SHIFT = 6;             // eid[5]高2位: die id
constexpr uint8_t EID_DIE_ID_MASK = 0x03;
constexpr uint8_t EID_PORT_NUM_MASK = 0x3F; // eid[5]低6位: 端口号
constexpr uint8_t EID_PORT_NUM_PG = 0x3F;   // PG端口固定端口号

// 辅助函数：将单个十六进制字符转为对应数值
static uint8_t hex_char_to_val(char c)
{
    if (c >= '0' && c <= '9') {
        return static_cast<int>(c) - '0';
    }
    if (c >= 'a' && c <= 'f') {
        return 10 + (static_cast<int>(c) - 'a');
    }
    if (c >= 'A' && c <= 'F') {
        return 10 + (static_cast<int>(c) - 'A');
    }
    return 0; // 非法字符默认返回0
}

// 核心函数：EID十六进制字符串转16字节uint8_t数组
// 参数：eid_str - 输入字符串(如"0x000000000040000000000000c0010001"，"0x"前缀可选)
//       eid_out - 输出数组(长度必须为16)
void eid_hex_to_uint8(const char* eid_str, uint8_t* eid_out)
{
    // 1. 清空输出数组（初始化全0）
    memset(eid_out, 0, 16);
    if (eid_str == nullptr) {
        return;
    }
    // 空EID(如被裁剪层端口)直接输出全0
    size_t hexLen = strlen(eid_str);
    if (hexLen < 2) {
        return;
    }
    // 2. 跳过可选的0x前缀，指向有效十六进制数据
    const char* hex_data = eid_str;
    size_t dataLen = hexLen;
    if (eid_str[0] == '0' && (eid_str[1] == 'x' || eid_str[1] == 'X')) {
        hex_data = eid_str + 2;
        dataLen = hexLen - 2;
    }

    // 3. 每2个字符转换为1个uint8_t字节，共转换16字节(数据不足时剩余字节保持0)
    for (int i = 0; i < 16; i++) {
        size_t highIdx = 2 * i;
        size_t lowIdx = 2 * i + 1;
        if (highIdx >= dataLen) {
            break;
        }
        // 高4位 + 低4位 组合成1个字节
        uint8_t high = hex_char_to_val(hex_data[highIdx]);
        uint8_t low = (lowIdx < dataLen) ? hex_char_to_val(hex_data[lowIdx]) : 0;
        eid_out[i] = (high << 4) | low;
    }
}

// 解析EID十六进制字符串的第byteIdx个字节(0~15, 高位字节在前)。
// 十六进制数据按右侧对齐: 不足32个十六进制字符时高位补0, 超出时只取低128bit。
static uint8_t eid_hex_byte_at(const std::string& eid, size_t byteIdx)
{
    if (eid.size() < 2 || byteIdx >= 16) {
        return 0;
    }
    size_t off = 0;
    if (eid[0] == '0' && (eid[1] == 'x' || eid[1] == 'X')) {
        off = 2;
    }
    size_t hexLen = eid.size() - off;
    if (hexLen > 32) {
        off = eid.size() - 32;
        hexLen = 32;
    }
    // 右对齐后第byteIdx字节对应的十六进制字符起始位置
    long start = static_cast<long>(hexLen) - static_cast<long>((16 - byteIdx) * 2);
    if (start < 0) {
        return 0;
    }
    uint8_t hi = hex_char_to_val(eid[off + start]);
    uint8_t lo = (start + 1 < static_cast<long>(hexLen)) ? hex_char_to_val(eid[off + start + 1]) : 0;
    return static_cast<uint8_t>((hi << 4) | lo);
}

// 按EID地址分配规则解码91~88位: super pod id[8:5]
static uint8_t eid_superpod_id_high(const std::string& eid)
{
    return eid_hex_byte_at(eid, 4) & EID_SUPERPOD_ID_HIGH_MASK;
}

// 按EID地址分配规则解码87~86位: die id
static uint8_t eid_die_id(const std::string& eid)
{
    return static_cast<uint8_t>((eid_hex_byte_at(eid, 5) >> EID_DIE_ID_SHIFT) & EID_DIE_ID_MASK);
}

// 按EID地址分配规则解码85~80位: 端口号
// (普通物理端口0-62; PG端口固定0x3F)
static uint8_t eid_port_num(const std::string& eid) { return eid_hex_byte_at(eid, 5) & EID_PORT_NUM_MASK; }

// 判断EID端口号是否为PG端口(portgroup端口)
// PG端口: 仅当port_group对应ports列表中有多个物理端口时, 端口号固定0x3F;
// 单个"die/port"端口的portGroup/未分组端口为普通物理端口, d2h等端口名不带端口
// 序号的端口按物理端口处理(端口号取pin_map中value=3对应的portId)
static bool eid_is_pg_port(const std::string& eid) { return (eid_port_num(eid) == EID_PORT_NUM_PG); }

// EID 字符串转 IPv4 字符串
// EID按地址分配规则, 31~0位即为IPV4地址(对应EID字符串末尾8个十六进制字符)
std::string eidToIP(const std::string& eid)
{
    // 空EID或长度不足时(如被裁剪层端口), 无法解析出IPv4地址
    if (eid.size() < 8) {
        return "0.0.0.0";
    }
    // 1. 取最后 8 个十六进制字符（IPV4地址所在位置）
    std::string ipv4_hex = eid.substr(eid.size() - 8, 8);

    // 2. 每 2 个字符转成一个 IP 段
    uint8_t o1 = (uint8_t)strtoul(ipv4_hex.substr(0, 2).c_str(), nullptr, 16);
    uint8_t o2 = (uint8_t)strtoul(ipv4_hex.substr(2, 2).c_str(), nullptr, 16);
    uint8_t o3 = (uint8_t)strtoul(ipv4_hex.substr(4, 2).c_str(), nullptr, 16);
    uint8_t o4 = (uint8_t)strtoul(ipv4_hex.substr(6, 2).c_str(), nullptr, 16);

    // 3. 拼接成 IP 字符串
    return std::to_string(o1) + "." + std::to_string(o2) + "." + std::to_string(o3) + "." + std::to_string(o4);
}

// 按文档规则: 当net_layer总层数 > 3时, 超出部分不输出
// (如配置文件中包含net_layer 0,1,2,3时, 只输出net_layer 0,1,2的内容)
// 统计**叶子节点**(server形态=server; POD形态=board)内所有net_layer
// (topo.json的links + rootinfo.json的ports), 总层数 > 3时返回排序后前3层
// (layer编号最小的3层)的集合, 否则返回空集合(表示不裁剪)
std::set<int> CollectAllowedNetLayers(const Board& board)
{
    std::set<int> boardNetLayers;
    for (const auto& link : board.links) {
        boardNetLayers.insert(link.netLayer);
    }
    for (const auto& dev : board.devices) {
        for (const auto& port : dev.ports) {
            boardNetLayers.insert(port.layer);
        }
    }

    if (boardNetLayers.size() <= 3) {
        return {};
    }

    std::set<int> allowedNetLayers;
    size_t kept = 0;
    for (int layer : boardNetLayers) {
        if (kept >= 3) {
            break;
        }
        allowedNetLayers.insert(layer);
        kept++;
    }
    return allowedNetLayers;
}

// net_instance_id的作用域(scope), 决定字符串格式:
//   BOARD    : sp_{i}_srv_{j}_board_{b}_idx_{k}  (POD形态板内实例, 每board唯一)
//   POD      : sp_{i}_srv_{j}_idx_{k}            (server形态layer0 / POD形态跨board的框内交换网)
//   SUPERPOD : sp_{i}_idx_{k}                    (跨server/POD但同超节点, 如layer1出框)
//   CLUSTER  : cluster                           (集群级或HOST/d2h层)
enum class NetInstanceScope { UNKNOWN, BOARD, POD, SUPERPOD, CLUSTER };

// 从rootinfo既有的net_instance_id识别其作用域模板。
// **继承模板而非按net_layer硬编码**: 生成侧(generate_server_topo.sh)已按拓扑形态与该层
// 是否存在PEER2PEER直连推导出scope(并可被net_instance_scope配置覆盖), 本侧只需识别并
// 用当前集群位置(sp/srv/board)重新渲染, 因此无需感知各拓扑的scope配置, 也不会把
// POD形态layer0的board_{b}段丢掉(丢掉会导致各board的layer0实例在ranktable中塌缩为同一个)
NetInstanceScope DetectNetInstanceScope(const std::string& netInstanceId)
{
    if (netInstanceId == "cluster") {
        return NetInstanceScope::CLUSTER;
    }
    unsigned int sp = 0;
    unsigned int srv = 0;
    unsigned int board = 0;
    unsigned int idxK = 0;
    unsigned int lay = 0;
    if (sscanf(netInstanceId.c_str(), "sp_%u_srv_%u_board_%u_idx_%u", &sp, &srv, &board, &idxK) == 4) {
        return NetInstanceScope::BOARD;
    }
    if (sscanf(netInstanceId.c_str(), "sp_%u_srv_%u_idx_%u", &sp, &srv, &idxK) == 3) {
        return NetInstanceScope::POD;
    }
    // 旧格式 sp_{sp}_srv_{srv}_lay_{layer}_idx_{k}: 等价于POD(server)作用域
    if (sscanf(netInstanceId.c_str(), "sp_%u_srv_%u_lay_%u_idx_%u", &sp, &srv, &lay, &idxK) == 4) {
        return NetInstanceScope::POD;
    }
    if (sscanf(netInstanceId.c_str(), "sp_%u_idx_%u", &sp, &idxK) == 2) {
        return NetInstanceScope::SUPERPOD;
    }
    return NetInstanceScope::UNKNOWN;
}

// 从net_instance_id解析idx_k(net_instance序号); 解析失败返回false(调用方回退枚举序号)
bool ParseNetInstanceIdx(const std::string& netInstanceId, uint32_t& idxK)
{
    unsigned int sp = 0;
    unsigned int srv = 0;
    unsigned int board = 0;
    unsigned int lay = 0;
    unsigned int parsedK = 0;
    if (sscanf(netInstanceId.c_str(), "sp_%u_srv_%u_board_%u_idx_%u", &sp, &srv, &board, &parsedK) == 4) {
        idxK = parsedK;
        return true;
    }
    if (sscanf(netInstanceId.c_str(), "sp_%u_srv_%u_idx_%u", &sp, &srv, &parsedK) == 3) {
        idxK = parsedK;
        return true;
    }
    if (sscanf(netInstanceId.c_str(), "sp_%u_idx_%u", &sp, &parsedK) == 2) {
        idxK = parsedK;
        return true;
    }
    if (sscanf(netInstanceId.c_str(), "sp_%u_srv_%u_lay_%u_idx_%u", &sp, &srv, &lay, &parsedK) == 4) {
        idxK = parsedK;
        return true;
    }
    return false;
}

// 按作用域渲染net_instance_id。scope为UNKNOWN时按net_layer回退到既有规则
// (layer0→POD, layer1→SUPERPOD, layer>=2→CLUSTER), 保证server形态产物不变
std::string RenderNetInstanceId(
    NetInstanceScope scope, int netLayer, uint32_t spIdx, uint32_t srvIdx, uint32_t boardIdx, uint32_t idxK)
{
    if (scope == NetInstanceScope::UNKNOWN) {
        scope = (netLayer >= 2) ? NetInstanceScope::CLUSTER :
                                  ((netLayer == 0) ? NetInstanceScope::POD : NetInstanceScope::SUPERPOD);
    }
    if (scope == NetInstanceScope::CLUSTER || netLayer >= 2) {
        // net_layer>=2固定为"cluster"(集群层, 如d2h层)。POD形态下若该层scope被显式声明为
        // board/pod/superpod, 仍按声明渲染, 以保留层级归属信息
        if (scope != NetInstanceScope::CLUSTER) {
            switch (scope) {
                case NetInstanceScope::BOARD:
                    return "sp_" + std::to_string(spIdx) + "_srv_" + std::to_string(srvIdx) + "_board_"
                           + std::to_string(boardIdx) + "_idx_" + std::to_string(idxK);
                case NetInstanceScope::POD:
                    return "sp_" + std::to_string(spIdx) + "_srv_" + std::to_string(srvIdx) + "_idx_"
                           + std::to_string(idxK);
                case NetInstanceScope::SUPERPOD:
                    return "sp_" + std::to_string(spIdx) + "_idx_" + std::to_string(idxK);
                default:
                    break;
            }
        }
        return "cluster";
    }
    switch (scope) {
        case NetInstanceScope::BOARD:
            return "sp_" + std::to_string(spIdx) + "_srv_" + std::to_string(srvIdx) + "_board_"
                   + std::to_string(boardIdx) + "_idx_" + std::to_string(idxK);
        case NetInstanceScope::POD:
            return "sp_" + std::to_string(spIdx) + "_srv_" + std::to_string(srvIdx) + "_idx_" + std::to_string(idxK);
        case NetInstanceScope::SUPERPOD:
            return "sp_" + std::to_string(spIdx) + "_idx_" + std::to_string(idxK);
        default:
            return "cluster";
    }
}

/*
 * @brief 解析ranktable.json的rank_id与EID映射, 写入RankEidMap表
 *
 * @param ranktablePath ranktable文件路径
 * @return HcclVmResult 状态码
 *
 * @note rank_id取文件显式值(修复ParseRanktable丢弃显式rank_id导致无法
 * 按文件内容定位rank的问题), 仅记录addr_type=="EID"的地址;
 * 每次通信域初始化整体重写(先清后插), 同rank重复EID去重,
 * 全零EID跳过(与InitPort空EID端口不入EndPoint的约定一致)。
 * estimate模式Communicator表只有本rank一行, 对端rankId以此表为准。
 */
HcclVmResult RecordRankEidMapFromRankTableFile(const std::string& ranktablePath)
{
    std::ifstream ifs(ranktablePath);
    if (!ifs.is_open()) {
        HCCL_VM_ERROR("failed to open ranktable file: {}", ranktablePath);
        return HcclVmResult::HCCL_SIM_E_OPEN_FILE_FAILURE;
    }

    json rankTable;
    try {
        rankTable = json::parse(ifs);
    } catch (const json::exception& e) {
        HCCL_VM_ERROR("failed to parse ranktable json: {}", e.what());
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    ifs.close();

    if (!rankTable.contains("rank_list") || !rankTable["rank_list"].is_array()) {
        HCCL_VM_ERROR("rank_list not found or not array in ranktable");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    // 整表重写前先清空（重置自增序列）
    (void)sim::runtime::Db::DeleteAll<sim::runtime::RankEidMap>(HcclSim::Storage::ClearMode::RESET_SEQUENCE);
    static const uint8_t zeroEid[URMA_EID_LEN] = {0};
    uint32_t rankCount = 0;
    uint32_t eidCount = 0;
    for (const auto& rank : rankTable["rank_list"]) {
        if (!rank.contains("rank_id") || !rank["rank_id"].is_number_unsigned()) {
            HCCL_VM_WARN("rank entry without valid rank_id, skip");
            continue;
        }
        const uint32_t rankId = rank["rank_id"].get<uint32_t>();
        rankCount++;

        const auto& levelList = rank.value("level_list", json::array());
        std::set<std::string> seenEids;
        for (const auto& level : levelList) {
            const auto& addrList = level.value("rank_addr_list", json::array());
            for (const auto& addrEntry : addrList) {
                if (addrEntry.value("addr_type", "") != "EID") {
                    continue;
                }
                const std::string addr = addrEntry.value("addr", "");
                if (addr.empty() || !seenEids.insert(addr).second) {
                    continue;
                }
                sim::runtime::RankEidMap record{};
                record.rank_id = rankId;
                eid_hex_to_uint8(addr.c_str(), record.eid);
                if (memcmp(record.eid, zeroEid, sizeof(record.eid)) == 0) {
                    continue;
                }
                if (!sim::runtime::Db::Add<sim::runtime::RankEidMap>(record).ok()) {
                    HCCL_VM_ERROR("failed to insert RankEidMap record, rankId={:d}", rankId);
                    return HcclVmResult::HCCL_SIM_E_INTERNAL;
                }
                eidCount++;
            }
        }
    }

    if (eidCount == 0) {
        HCCL_VM_WARN("no EID entry recorded from ranktable: {}", ranktablePath);
    }
    HCCL_VM_DEBUG("recorded {:d} eid entries for {:d} ranks from ranktable", eidCount, rankCount);
    return HcclVmResult::HCCL_SIM_SUCCESS;
}
} // namespace
/*
 * @brief 初始化集群拓扑数据
 *
 * @param clusterDir 集群拓扑文件目录
 * @return HcclVmResult 状态码
 *
 * @note 该函数会解析clusterDir目录下的所有文件，生成IR数据，初始化至DB层。
 * 初始化静态DB表数据：Server/Host/Device/Ccu/Port/EndPoint/EndPointPortMapping
 */
HcclVmResult AscendClusterTopoParser::InitClusterTopo(const std::string& clusterDir)
{
    HCCL_VM_DEBUG("Enter InitClusterTopo: {}", clusterDir);
    // 1. 解析集群拓扑文件
    ClusterTopoParser parser;
    ParseStatus status = parser.ParseClusterDir(clusterDir);

    if (status != ParseStatus::OK) {
        HCCL_VM_ERROR("Failed to parse cluster directory (status={:d})", static_cast<int>(status));
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    // 2. 生成IR数据
    network_ = parser.GetNetwork();

    HCCL_VM_DEBUG("Parse completed successfully!");
    HCCL_VM_DEBUG("Network version: {}", network_.version);
    HCCL_VM_DEBUG("SuperPod count: {}", network_.GetSuperPodCount());
    HCCL_VM_DEBUG("Total device count: {}", network_.GetTotalDeviceCount());
    HCCL_VM_DEBUG("Total link count: {}", network_.GetTotalLinkCount());

    fs::create_directories(fs::path(InstallPath::ResolveToInstallRoot("data")));
    ClusterTopoDumper::DumpToFile(network_, InstallPath::ResolveToInstallRoot("data/" + outputFile_));

    // 3. 初始化IR数据，保存至DB层
    if (InitClusterStaticTopoData() != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("Failed to init cluster static topo data");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

/*
 * @brief 初始化集群通信域表项(ranktable模式)
 *
 * @param ranktablePath ranktable文件路径
 * @return HcclVmResult 状态码
 *
 * @note 该函数会解析ranktable，生成IR数据，初始化至DB层。
 * 初始化本次算子的动态DB数据：Rank/Link等
 * 更新静态表字段：Device.logic_id
 */
HcclVmResult AscendClusterTopoParser::ParseRanktableAndInitCommDomain(const std::string& ranktablePath)
{
    HCCL_VM_DEBUG("Enter ParseRanktableAndInitCommDomain, ranktablePath: {}", ranktablePath);

    TopoMeta topoMeta;
    if (ParseRanktable(ranktablePath, topoMeta) != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("parse rank table failed");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    // ranktable模式: 刷新TopoMetaConfig表项与进程通信域配置缓存
    if (!sim::runtime::RefreshTopoMetaConfig("ranktable", &topoMeta)) {
        HCCL_VM_ERROR("refresh topo meta config failed");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    auto ret = InitCommunicationDomain(topoMeta, true);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        return ret;
    }
    // ranktable模式: 记录文件显式rank_id与EID的映射(供estimate模式解析对端rankId)
    if (RecordRankEidMapFromRankTableFile(ranktablePath) != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("record rank eid map from ranktable failed");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

/*
 * @brief 初始化集群通信域表项
 *
 * @param topoMeta 集群拓扑元数据
 * @return HcclVmResult 状态码
 *
 * @note 该函数会解析topoMeta，生成IR数据，初始化至DB层。
 * 初始化本次算子的动态DB数据：Rank/Link等
 * 更新静态表字段：Device.logic_id
 */
HcclVmResult AscendClusterTopoParser::InitCommunicationDomain(const TopoMeta& topoMeta, bool withRanktable)
{
    HCCL_VM_DEBUG("Enter InitCommunicationDomain");

    // 第一步：初始化所有叶子节点(server形态=server; POD形态=board)上的 Device
    // （logic_id = physical_id，保持不变；physical_id为叶子节点内物理id, POD形态即board内0~7）
    for (const auto& [superPodId, superPod] : topoMeta) {
        for (const auto& [serverId, server] : superPod) {
            for (const auto& [boardId, nodeMeta] : server) {
                auto serverKey = sim::runtime::GetServerKeyById(superPodId, serverId, boardId);
                for (uint32_t deviceIdx = 0; deviceIdx < nodeMeta.size(); ++deviceIdx) {
                    auto physicId = nodeMeta[deviceIdx];
                    // todo: 暂时不考虑容器场景，userId == logicDevId
                    auto userId = deviceIdx;
                    if (sim::runtime::UpdateDeviceLogicId(serverKey, physicId, deviceIdx, userId) != ACL_SUCCESS) {
                        HCCL_VM_ERROR(
                            "update device logic id {:d} failed, serverKey: {:d}, physicId: {:d}, userId: {:d}",
                            deviceIdx, serverKey, physicId, userId);
                        return HcclVmResult::HCCL_SIM_E_INTERNAL;
                    }
                }
            }
        }
    }

    // 第二步：按 rankId % deviceCount 的逻辑创建 Rank 表
    if (InitDynamicModelData(topoMeta) != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("init dynamic model data failed");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    // 通信域未初始化，则需创建ranktable.json文件
    if (status_ == HvmClusterStatus::COMM_DOMAIN_UNINIT && !withRanktable) {
        // 根据topoMeta生成ranktable.json文件
        if (CreateRankTableFile(topoMeta) != HcclVmResult::HCCL_SIM_SUCCESS) {
            HCCL_VM_ERROR("create rank table file failed");
            return HcclVmResult::HCCL_SIM_E_INTERNAL;
        }
    }

    status_ = HvmClusterStatus::COMM_DOMAIN_INIT_DONE;
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult AscendClusterTopoParser::InitDynamicModelData(const TopoMeta& topoMeta)
{
    for (const auto& [superPodId, superPod] : topoMeta) {
        for (const auto& [serverId, server] : superPod) {
            for (const auto& [boardId, nodeMeta] : server) {
                auto serverKey = sim::runtime::GetServerKeyById(superPodId, serverId, boardId);
                // used_dev_num: 该叶子节点(server/board)编入通信域的卡数。
                // POD形态下即为board内的卡数(如8), 故业务侧aclrtGetDeviceCount()返回8,
                // 应用按aclrtSetDevice(rank % 8)选卡, 与真机一致
                uint32_t deviceCount = static_cast<uint32_t>(nodeMeta.size());

                sim::runtime::Db::Update<sim::runtime::Server>(
                    HcclSim::Storage::Eq(&sim::runtime::Server::id, serverKey),
                    HcclSim::Storage::Set(&sim::runtime::Server::used_dev_num, deviceCount));

                for (uint32_t deviceIdx = 0; deviceIdx < deviceCount; ++deviceIdx) {
                    // rankId % deviceCount 是应用实际选择的 logicDevId
                    auto phyId = nodeMeta[deviceIdx];
                    // 查找该叶子节点上 physical_id == phyId 的 Device
                    auto devResult = sim::runtime::Db::GetOneByPred<sim::runtime::Device>(HcclSim::Storage::And(
                        HcclSim::Storage::Eq(&sim::runtime::Device::server_id, serverKey),
                        HcclSim::Storage::Eq(&sim::runtime::Device::physical_id, phyId)));
                    if (!devResult.ok()) {
                        HCCL_VM_ERROR(
                            "device not found: serverKey={}, phyDevId={} (superpod{:d}/server{:d}/board{:d})",
                            serverKey, phyId, superPodId, serverId, boardId);
                        return HcclVmResult::HCCL_SIM_E_INTERNAL;
                    }
                    const auto& device = *devResult;
                    auto deviceKey = device.id;

                    // 校验该 device 至少关联一个 CCU
                    auto ccuResult = sim::runtime::Db::GetByPred<sim::runtime::Ccu>(
                        HcclSim::Storage::Eq(&sim::runtime::Ccu::device_id, deviceKey));
                    if (!ccuResult.ok() || !ccuResult.value.has_value() || ccuResult.value->empty()) {
                        HCCL_VM_ERROR("get device all ccu failed");
                        return HcclVmResult::HCCL_SIM_E_INTERNAL;
                    }
                }
            }
        }
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult AscendClusterTopoParser::BuildLevelList(
    const Board& board, int srcDevPhyId, const std::set<int>& commDomainLocalIds, uint32_t spIdx, uint32_t srvIdx,
    uint32_t boardIdx, json& levelList, bool skipLayerAbove2, const std::set<int>& allowedNetLayers)
{
    const Device* device = board.GetDevice(srcDevPhyId);
    if (!device) {
        HCCL_VM_ERROR(
            "cannot find device by phyDevId {:d} in superpod{:d}/server{:d}/board{:d}", srcDevPhyId, spIdx, srvIdx,
            boardIdx);
        return HCCL_SIM_E_NOT_FOUND;
    }

    // Collect links from all layers that touch this device
    std::map<int, std::vector<const Link*>> layerLinks;
    for (const auto& link : board.links) {
        if (link.sideA.localId == srcDevPhyId || link.sideB.localId == srcDevPhyId) {
            layerLinks[link.netLayer].push_back(&link);
        }
    }

    // net_instance_id构造: **继承rootinfo中既有字符串的作用域(scope)模板**, 仅把sp/srv/board
    // 替换为当前集群位置、idx_k沿用生成时配置的net_instance序号(见RenderNetInstanceId)。
    // 不按net_layer硬编码, 因此POD形态下layer0的 board_{b} 段不会丢失(丢失会导致各board的
    // layer0实例在ranktable中塌缩为同一个, 不同board的同号卡被误判为同一实例内的对端)
    auto buildNetInstanceId = [&](const std::string& rootinfoNiid, int layer, uint32_t niid) -> std::string {
        return RenderNetInstanceId(DetectNetInstanceScope(rootinfoNiid), layer, spIdx, srvIdx, boardIdx, niid);
    };

    // 从rootinfo的net_instance_id中解析idx_k(net_instance序号), 兼容
    // "sp_{sp}_srv_{srv}_board_{b}_idx_{k}" / "sp_{sp}_srv_{srv}_idx_{k}" / "sp_{sp}_idx_{k}"
    // 及旧格式"sp_{sp}_srv_{srv}_lay_{layer}_idx_{k}"; 解析失败(或固定值"cluster")时返回false
    auto parseNetInstanceIdx = [](const std::string& netInstanceId, uint32_t& idxK) -> bool {
        return ParseNetInstanceIdx(netInstanceId, idxK);
    };

    // net_type按文档规则: net_layer=0时为TOPO_FILE_DESC, net_layer>=1时为CLOS
    auto netTypeByLayer = [](int layer) -> std::string {
        return (layer == 0) ? "TOPO_FILE_DESC" : "CLOS";
    };

    for (const auto& [netLayer, links] : layerLinks) {
        // 单server通信域(mock-comm配置的server num == 1)时, net_layer>=3的内容不生成
        if (netLayer >= 3 && skipLayerAbove2) {
            continue;
        }

        // net_layer总层数 > 3时, 超出部分(前3层之外)不输出
        if (!allowedNetLayers.empty() && allowedNetLayers.find(netLayer) == allowedNetLayers.end()) {
            continue;
        }

        // Separate P2P and P2N links at this netLayer
        std::vector<const Link*> p2pLinks, p2nLinks;
        for (const auto* link : links) {
            if (link->IsPeer2Peer()) {
                p2pLinks.push_back(link);
            } else if (link->IsPeer2Net()) {
                p2nLinks.push_back(link);
            }
        }

        // Process PEER2PEER links from topo
        // 按rootinfo中端口的net_instance_id分组(文档links新增net_instance_id字段):
        // idx_k与作用域(scope)均继承rootinfo中配置的net_instance_id, 支持同层多net_instance,
        // 未配置(net_instance_id解析失败)时按默认序号0与net_layer回退规则处理。
        // 分组键使用**渲染后的net_instance_id字符串**(而非仅idx_k), 因为POD形态下同一layer
        // 的不同实例可能作用域不同(board/pod), 仅按idx_k分组会把它们错误合并
        if (!p2pLinks.empty()) {
            std::map<std::string, json> niidToLevelEntry;
            for (const auto* link : p2pLinks) {
                const LinkPortRef* srcPort = nullptr;
                const LinkPortRef* dstPort = nullptr;
                if (link->sideA.localId == srcDevPhyId) {
                    srcPort = &link->sideA;
                    dstPort = &link->sideB;
                } else if (link->sideB.localId == srcDevPhyId) {
                    srcPort = &link->sideB;
                    dstPort = &link->sideA;
                } else {
                    continue;
                }

                // 从rootinfo端口(topo.json连线端口对应rootinfo中的EID端口)解析该P2P端口的
                // net_instance_id: idx_k(实例序号)与scope(作用域)均继承该字符串
                uint32_t idxK = 0;
                std::string rootinfoNiid;
                for (const auto& port : device->ports) {
                    if (port.layer == netLayer && port.linkType == LinkType::PEER2PEER && !port.netInstanceId.empty()
                        && std::find(srcPort->portIds.begin(), srcPort->portIds.end(), port.portId)
                               != srcPort->portIds.end()) {
                        rootinfoNiid = port.netInstanceId;
                        uint32_t parsedK = 0;
                        if (parseNetInstanceIdx(port.netInstanceId, parsedK)) {
                            idxK = parsedK;
                        }
                        break;
                    }
                }
                std::string renderedNiid = buildNetInstanceId(rootinfoNiid, netLayer, idxK);

                if (niidToLevelEntry.find(renderedNiid) == niidToLevelEntry.end()) {
                    json levelEntry;
                    levelEntry["net_addr"] = "";
                    levelEntry["net_type"] = netTypeByLayer(netLayer);
                    levelEntry["net_instance_id"] = renderedNiid;
                    levelEntry["net_layer"] = netLayer;
                    levelEntry["position"] = PositionToString(link->position);
                    json p2pProtos = json::array();
                    for (const auto& proto : link->protocols) {
                        p2pProtos.push_back(ProtocolToString(proto));
                    }
                    levelEntry["protocols"] = p2pProtos;
                    levelEntry["rank_addr_list"] = json::array();
                    niidToLevelEntry[renderedNiid] = levelEntry;
                }

                int peerId = (link->sideA.localId == srcDevPhyId) ? link->sideB.localId : link->sideA.localId;
                if (commDomainLocalIds.find(peerId) == commDomainLocalIds.end()) {
                    continue;
                }

                json addrEntry;
                // addr为rootinfo中分配的EID(新EID地址分配规则:
                // 91~88位super pod id[8:5], 87~86位die id, 85~80位端口号, 31~0位IPV4地址),
                // 去掉"0x"前缀后写入ranktable.json
                std::string eid = srcPort->eid;
                if (eid.size() > 2 && eid[0] == '0' && (eid[1] == 'x' || eid[1] == 'X')) {
                    eid = eid.substr(2);
                }
                addrEntry["addr"] = eid;
                addrEntry["addr_type"] = "EID";
                // peer2peer端口plane_id固定为"plane_0"
                addrEntry["plane_id"] = "plane_0";
                addrEntry["ports"] = srcPort->portIds;

                AddLinkInfo(srcPort, dstPort, netLayer, link->protocols);

                niidToLevelEntry[renderedNiid]["rank_addr_list"].push_back(addrEntry);
            }

            for (auto& [niidKey, levelEntry] : niidToLevelEntry) {
                levelList.push_back(levelEntry);
            }
        }

        // Process PEER2NET from rootinfo portGroups
        // Group ports by their net_instance_id from rootinfo to support multiple net_instances per layer.
        // Group ports by their net_instance_id from rootinfo to support multiple net_instances per layer.
        if (!p2nLinks.empty()) {
            std::map<std::string, std::map<std::string, std::vector<std::string>>> niidToEidPorts;
            std::map<std::string, std::string> eidToPlaneId;
            for (const auto& port : device->ports) {
                if (port.layer == netLayer && port.linkType == LinkType::PEER2NET && !port.eid.empty()) {
                    niidToEidPorts[port.netInstanceId][port.eid].push_back(port.portId);
                    if (!port.planeId.empty()) {
                        eidToPlaneId[port.eid] = port.planeId;
                    }
                }
            }

            if (!niidToEidPorts.empty()) {
                uint32_t niidIdx = 0;
                for (const auto& [rootinfoNiid, eidToPorts] : niidToEidPorts) {
                    json levelEntry;
                    levelEntry["net_addr"] = "";
                    levelEntry["net_type"] = netTypeByLayer(netLayer);
                    // idx_k与作用域(scope)继承rootinfo中配置的net_instance_id, 解析失败时按枚举顺序编号
                    uint32_t idxK = niidIdx;
                    uint32_t parsedK = 0;
                    if (parseNetInstanceIdx(rootinfoNiid, parsedK)) {
                        idxK = parsedK;
                    }
                    levelEntry["net_instance_id"] = buildNetInstanceId(rootinfoNiid, netLayer, idxK);
                    levelEntry["net_layer"] = netLayer;

                    // 从端口获取position和protocols
                    Position p2nPosition = Position::UNKNOWN;
                    std::vector<Protocol> p2nProtocols;
                    for (const auto& port : device->ports) {
                        if (port.layer == netLayer && port.linkType == LinkType::PEER2NET && !port.eid.empty()) {
                            p2nPosition = port.position;
                            p2nProtocols = port.protocols;
                            break;
                        }
                    }
                    levelEntry["position"] = PositionToString(p2nPosition);
                    json protosJson = json::array();
                    for (const auto& proto : p2nProtocols) {
                        protosJson.push_back(ProtocolToString(proto));
                    }
                    levelEntry["protocols"] = protosJson;

                    json rankAddrList = json::array();
                    // UBOE协议层对齐真实环境topo生成(product_server.c LayerAddUBOE:
                    // UrmaEid2CNA将EID转为IPv4后AddrSetIP, plane_id固定"plane_uboe"):
                    // hcomm的UBOE TP属性下发(tp_manager.cc Ipv4ToIpArray, inet_pton(AF_INET))
                    // 要求该层链路地址为点分IPv4, EID地址会导致HCCL_E_PARA建链失败
                    const bool isUboeLayer
                        = std::find(p2nProtocols.begin(), p2nProtocols.end(), Protocol::UBOE) != p2nProtocols.end();
                    for (const auto& [eid, ports] : eidToPorts) {
                        json addrEntry;
                        std::string cleanEid = eid;
                        if (cleanEid.size() > 2 && cleanEid[0] == '0' && (cleanEid[1] == 'x' || cleanEid[1] == 'X')) {
                            cleanEid = cleanEid.substr(2);
                        }
                        if (isUboeLayer) {
                            // 下发EID内嵌IPv4(低32位, 与EndPoint表ip_addr一致), 其余协议端口仍下发EID
                            addrEntry["addr"] = eidToIP(eid);
                            addrEntry["addr_type"] = "IPV4";
                            addrEntry["plane_id"] = "plane_uboe";
                        } else {
                            addrEntry["addr"] = cleanEid;
                            addrEntry["addr_type"] = "EID";
                            // peer2net端口plane_id继承rootinfo中的值(默认"plane_pg_0")
                            auto planeIt = eidToPlaneId.find(eid);
                            addrEntry["plane_id"] = (planeIt != eidToPlaneId.end()) ? planeIt->second : "plane_pg_0";
                        }
                        addrEntry["ports"] = ports;

                        LinkPortRef srcRef;
                        srcRef.localId = srcDevPhyId;
                        srcRef.eid = eid;
                        srcRef.portIds = ports;
                        // 使用端口的protocols，而非p2nLinks[0]->protocols
                        AddLinkInfo(&srcRef, nullptr, netLayer, p2nProtocols);

                        rankAddrList.push_back(addrEntry);
                    }

                    levelEntry["rank_addr_list"] = rankAddrList;
                    levelList.push_back(levelEntry);
                    niidIdx++;
                }
            }
        }
    }

    // Process ports from rootinfo that don't have corresponding links in topo.json
    // (e.g., d2h ports that may not have been included in edge_list)
    std::set<int> processedLayers;
    for (const auto& [netLayer, links] : layerLinks) {
        processedLayers.insert(netLayer);
    }
    for (const auto& port : device->ports) {
        if (port.linkType == LinkType::PEER2NET && !port.eid.empty()
            && processedLayers.find(port.layer) == processedLayers.end()) {
            // 单server通信域(mock-comm配置的server num == 1)时, net_layer>=3的内容不生成
            if (port.layer >= 3 && skipLayerAbove2) {
                continue;
            }
            // net_layer总层数 > 3时, 超出部分(前3层之外)不输出
            if (!allowedNetLayers.empty() && allowedNetLayers.find(port.layer) == allowedNetLayers.end()) {
                continue;
            }
            // 按rootinfo中的net_instance_id分组, 支持同层多net_instance
            std::map<std::string, std::map<std::string, std::vector<std::string>>> niidToEidPorts;
            std::map<std::string, std::string> eidToPlaneId;
            int netLayer = port.layer;
            for (const auto& p : device->ports) {
                if (p.layer == netLayer && p.linkType == LinkType::PEER2NET && !p.eid.empty()) {
                    niidToEidPorts[p.netInstanceId][p.eid].push_back(p.portId);
                    if (!p.planeId.empty()) {
                        eidToPlaneId[p.eid] = p.planeId;
                    }
                }
            }
            uint32_t niidIdx = 0;
            for (const auto& [rootinfoNiid, eidToPorts] : niidToEidPorts) {
                if (eidToPorts.empty()) {
                    continue;
                }
                json levelEntry;
                levelEntry["net_addr"] = "";
                levelEntry["net_type"] = netTypeByLayer(netLayer);
                // idx_k与作用域(scope)继承rootinfo中配置的net_instance_id, 解析失败时按枚举顺序编号
                uint32_t idxK = niidIdx;
                uint32_t parsedK = 0;
                if (parseNetInstanceIdx(rootinfoNiid, parsedK)) {
                    idxK = parsedK;
                }
                levelEntry["net_instance_id"] = buildNetInstanceId(rootinfoNiid, netLayer, idxK);
                levelEntry["net_layer"] = netLayer;
                levelEntry["position"] = PositionToString(port.position);
                json protosJson = json::array();
                for (const auto& proto : port.protocols) {
                    protosJson.push_back(ProtocolToString(proto));
                }
                levelEntry["protocols"] = protosJson;

                json rankAddrList = json::array();
                for (const auto& [eid, ports] : eidToPorts) {
                    json addrEntry;
                    std::string cleanEid = eid;
                    if (cleanEid.size() > 2 && cleanEid[0] == '0' && (cleanEid[1] == 'x' || cleanEid[1] == 'X')) {
                        cleanEid = cleanEid.substr(2);
                    }
                    addrEntry["addr"] = cleanEid;
                    addrEntry["addr_type"] = "EID";
                    // peer2net端口plane_id继承rootinfo中的值(默认"plane_pg_0")
                    auto planeIt = eidToPlaneId.find(eid);
                    addrEntry["plane_id"] = (planeIt != eidToPlaneId.end()) ? planeIt->second : "plane_pg_0";
                    addrEntry["ports"] = ports;
                    rankAddrList.push_back(addrEntry);
                }

                levelEntry["rank_addr_list"] = rankAddrList;
                levelList.push_back(levelEntry);
                niidIdx++;
            }
            if (!niidToEidPorts.empty()) {
                processedLayers.insert(netLayer);
                processedLayers.insert(netLayer);
            }
        }
    }

    // Merge level_list entries with the same (net_layer, net_instance_id)
    {
        json mergedLevelList = json::array();
        std::map<std::pair<int, std::string>, size_t> levelIndex;
        for (const auto& entry : levelList) {
            int entryNetLayer = entry.value("net_layer", 0);
            std::string entryNetInstanceId = entry.value("net_instance_id", "");
            auto key = std::make_pair(entryNetLayer, entryNetInstanceId);
            auto it = levelIndex.find(key);
            if (it != levelIndex.end()) {
                for (const auto& addr : entry["rank_addr_list"]) {
                    mergedLevelList[it->second]["rank_addr_list"].push_back(addr);
                }
            } else {
                levelIndex[key] = mergedLevelList.size();
                mergedLevelList.push_back(entry);
            }
        }
        levelList = mergedLevelList;
    }

    // Merge level_list entries with the same (net_layer, net_instance_id)
    {
        json mergedLevelList = json::array();
        std::map<std::pair<int, std::string>, size_t> levelIndex;
        for (const auto& entry : levelList) {
            int entryNetLayer = entry.value("net_layer", 0);
            std::string entryNetInstanceId = entry.value("net_instance_id", "");
            auto key = std::make_pair(entryNetLayer, entryNetInstanceId);
            auto it = levelIndex.find(key);
            if (it != levelIndex.end()) {
                for (const auto& addr : entry["rank_addr_list"]) {
                    mergedLevelList[it->second]["rank_addr_list"].push_back(addr);
                }
            } else {
                levelIndex[key] = mergedLevelList.size();
                mergedLevelList.push_back(entry);
            }
        }
        levelList = mergedLevelList;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

json AscendClusterTopoParser::BuildRankEntry(
    const Board& board, int srcDevPhyId, const std::set<int>& commDomainLocalIds, uint32_t spIdx, uint32_t srvIdx,
    uint32_t boardIdx, uint32_t rankId, bool skipLayerAbove2, const std::set<int>& allowedNetLayers)
{
    json rankEntry;
    // device_id: **叶子节点内物理id**(server形态=server内; POD形态=board内, 如0~7), 是物理id而非逻辑id:
    // hcomm的CheckRankGraphAddrs按 rank.deviceId == devPhyId 筛选条目, 并以该phyId查询硬件EID
    // local_id: 与topo.json的local_id同口径 —— server形态与device_id同值;
    // POD形态下topo.json为POD级单份(server目录), 其local_id为POD全局id, 故
    // local_id = board_id * 每board卡数 + board内物理id (业务侧/level1劫持层按ranktable的
    // local_id匹配POD级topo.json中的边, 二者值域必须一致)
    const int devNumPerBoard = board.GetDeviceCount();
    rankEntry["device_id"] = srcDevPhyId;
    rankEntry["local_id"] = static_cast<int>(boardIdx) * devNumPerBoard + srcDevPhyId;
    rankEntry["rank_id"] = rankId;
    rankEntry["level_list"] = json::array();

    BuildLevelList(
        board, srcDevPhyId, commDomainLocalIds, spIdx, srvIdx, boardIdx, rankEntry["level_list"], skipLayerAbove2,
        allowedNetLayers);

    return rankEntry;
}

HcclVmResult AscendClusterTopoParser::CreateRankTableFile(const TopoMeta& topoMeta)
{
    json rankTable;
    rankTable["rank_count"] = ShmGetPhyDeviceTotalCount(topoMeta);
    rankTable["rank_list"] = json::array();
    rankTable["version"] = "2.0";

    // 统计本次通信域(mock-comm配置)的**叶子节点**总数(server形态=server数; POD形态=board数):
    //   叶子节点数 == 1时, net_layer>=3的内容不生成; > 1时正常生成
    uint32_t leafNodeNum = 0;
    for (const auto& [podId, superPodMeta] : topoMeta) {
        for (const auto& [serverId, serverMeta] : superPodMeta) {
            leafNodeNum += static_cast<uint32_t>(serverMeta.size());
        }
    }
    // 跨超节点通信域(topoMeta包含不止一个超节点/pod)时, layer3不过滤:
    // 不执行"单叶子节点跳过net_layer>=3"与"net_layer总层数>3只保留前3层"两条裁剪规则
    bool crossSuperPod = (topoMeta.size() > 1);
    bool skipLayerAbove2 = (leafNodeNum == 1) && !crossSuperPod;
    if (skipLayerAbove2) {
        HCCL_VM_DEBUG(
            "single leaf node communication domain (leaf node num = {:d}), "
            "net_layer >= 3 entries are skipped in ranktable.json",
            leafNodeNum);
    }
    if (crossSuperPod) {
        HCCL_VM_DEBUG(
            "cross super pod communication domain (pod num = {:d}), "
            "net_layer >= 3 entries are not filtered in ranktable.json",
            topoMeta.size());
    }

    uint32_t rankId = 0;

    // rank_id分配顺序: superpod → server(POD) → board → device, 从0递增
    for (const auto& [podId, superPodMeta] : topoMeta) {
        const SuperPod* superPod = network_.GetSuperPod(static_cast<int>(podId));
        if (superPod == nullptr) {
            HCCL_VM_ERROR("superpod id {:d} out of range", podId);
            return HCCL_SIM_E_NOT_FOUND;
        }

        for (const auto& [serverId, serverMeta] : superPodMeta) {
            const Server* server = superPod->GetServer(static_cast<int>(serverId));
            if (server == nullptr) {
                HCCL_VM_ERROR("server id {:d} out of range in superpod {:d}", serverId, podId);
                return HCCL_SIM_E_NOT_FOUND;
            }

            for (const auto& [boardId, nodeMeta] : serverMeta) {
                const Board* board = server->GetBoard(static_cast<int>(boardId));
                if (board == nullptr) {
                    HCCL_VM_ERROR(
                        "board id {:d} not found in superpod{:d}/server{:d} (POD形态需在topo_meta中按board声明ranks)",
                        boardId, podId, serverId);
                    return HCCL_SIM_E_NOT_FOUND;
                }

                // 按文档规则: 当net_layer总层数 > 3时, ranktable.json中超出部分不输出
                // (如配置文件中包含net_layer 0,1,2,3时, 只输出net_layer 0,1,2的内容);
                // 层数按**叶子节点(board)**作用域统计; 跨超节点通信域不裁剪: layer3正常输出
                std::set<int> allowedNetLayers = crossSuperPod ? std::set<int>{} : CollectAllowedNetLayers(*board);
                if (!allowedNetLayers.empty()) {
                    HCCL_VM_DEBUG(
                        "net_layer total count > 3 in superpod{:d}/server{:d}/board{:d}, "
                        "net_layers beyond the first 3 are skipped in ranktable.json",
                        podId, serverId, boardId);
                }

                // 通信域内本board的卡(local_id集合): PEER2PEER只存在于board内, 故对端过滤按board范围
                std::set<int> commDomainLocalIds(nodeMeta.begin(), nodeMeta.end());

                for (uint32_t deviceIdx = 0; deviceIdx < nodeMeta.size(); ++deviceIdx) {
                    // logicDevId = rankId % 本叶子节点卡数, 与应用侧 aclrtSetDevice(rank % D) 公式一致
                    // (D = 该叶子节点编入通信域的卡数; POD形态下即board内卡数, 如8)
                    uint32_t logicDevId = rankId % static_cast<uint32_t>(nodeMeta.size());
                    PhyDeviceId phyDevId = nodeMeta[logicDevId];
                    rankTable["rank_list"].push_back(BuildRankEntry(
                        *board, phyDevId, commDomainLocalIds, podId, serverId, boardId, rankId, skipLayerAbove2,
                        allowedNetLayers));
                    rankId++;
                }
            }
        }
    }

    fs::path dataDir = fs::path(InstallPath::ResolveToInstallRoot("data"));
    std::error_code ec2;
    fs::create_directories(dataDir, ec2);
    std::string outputPath = (dataDir / "ranktable.json").string();
    std::ofstream ofs(outputPath);
    if (!ofs.is_open()) {
        HCCL_VM_ERROR("failed to open ranktable.json for writing");
        return HcclVmResult::HCCL_SIM_E_OPEN_FILE_FAILURE;
    }

    ofs << rankTable.dump(4) << std::endl;
    ofs.close();

    setenv("RANK_TABLE_FILE", outputPath.c_str(), 1);

    // yaml模式: ranktable.json刚生成, 同步记录rank_id与EID映射
    if (RecordRankEidMapFromRankTableFile(outputPath) != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("record rank eid map from generated ranktable failed");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    HCCL_VM_DEBUG("ranktable.json generated with {:d} ranks", rankId);
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

namespace {
// ---- T1 字段填充 helper：只负责原 Init* 的原字段赋值（不做 IO、不保存状态）----

void FillDevice(sim::runtime::Device& deviceDb, uint64_t serverKey, const Device& deviceIn)
{
    deviceDb.server_id = serverKey;
    deviceDb.physical_id = deviceIn.localId;
    deviceDb.overflow_mode = 0;
    if (!deviceIn.socVersion.empty()) {
        // todo: 从yaml文件中获取soc版本
        strncpy(deviceDb.soc_version, deviceIn.socVersion.c_str(), sizeof(deviceDb.soc_version) - 1);
        deviceDb.soc_version[sizeof(deviceDb.soc_version) - 1] = '\0';
    }
    deviceDb.status = 0;
    deviceDb.id = 0;
}

void FillCcu(sim::runtime::Ccu& ccu, uint64_t deviceKey, uint32_t dieId)
{
    ccu.device_id = deviceKey;
    ccu.die_id = dieId;
    ccu.status = 0;
    ccu.id = 0;
}

void FillPort(sim::runtime::Port& portDb, uint64_t deviceKey, const Port& portIn)
{
    portDb.device_id = deviceKey;
    if (portIn.dieId >= 0) {
        portDb.die_id = static_cast<uint32_t>(portIn.dieId);
    }
    strcpy(portDb.name, portIn.portId.c_str());
    portDb.id = 0;
}

void FillEndPoint(sim::runtime::EndPoint& endPoint, uint64_t deviceKey, const Port& portIn)
{
    endPoint.device_id = deviceKey;
    if (portIn.dieId >= 0) {
        endPoint.die_id = static_cast<uint32_t>(portIn.dieId);
    }
    if (portIn.layer == 0) {
        endPoint.func_id = 2;
    } else {
        endPoint.func_id = 3;
    }
    endPoint.type = 0;
    // port/port_group配置protocols含UBOE时(如ascend950_server_topo_normal_uboe.yaml
    // 的port_group layer2 UBOE端口), 标记EndPoint为UBOE端口
    endPoint.is_uboe
        = std::find(portIn.protocols.begin(), portIn.protocols.end(), Protocol::UBOE) != portIn.protocols.end();
    eid_hex_to_uint8(portIn.eid.c_str(), endPoint.eid);
    strcpy(endPoint.ip_addr, eidToIP(portIn.eid).c_str());
    endPoint.id = 0;
}
} // namespace

HcclVmResult AscendClusterTopoParser::InitClusterStaticTopoData()
{
    // 静态拓扑批量写入合并为单事务提交（T1 重排：逐 server、device ≤128 一窗、
    // 子记录 ≤ kInsertBatchMaxRows 一批）：自动无约束批次在引擎内一次取号与
    // 有界批预检，消除逐行 INCR/WATCH/EXISTS 往返；原整笔 Commit 不变
    // （表间执行顺序改变仅发生在这笔未提交初始化事务内）
    auto txnResult = HcclSim::Storage::BeginTransaction();
    if (!txnResult.ok()) {
        HCCL_VM_ERROR("begin static topo batch write transaction failed: {}", txnResult.diagnostic);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    auto& writeTxn = *txnResult;
    // Port 批处理：Port/EndPoint/EndPointPortMapping 三段批量 + ip→endpointId
    // 索引（成功批次才加入 ID 关联；同一 IP 保留最先插入，与原语义一致）
    const auto flushPortBatch = [this](
                                    const std::vector<std::pair<const Port*, uint64_t>>& portRefs,
                                    HcclSim::Storage::Transaction& txn) -> HcclVmResult {
        // Port 行：窗口容器原位构造最终 T，ID 回填后供 Mapping 引用
        std::vector<sim::runtime::Port> portRows;
        portRows.reserve(portRefs.size());
        for (const auto& ref : portRefs) {
            portRows.emplace_back();
            FillPort(portRows.back(), ref.second, *ref.first);
        }
        auto portBatch = HcclSim::Storage::InsertMany(portRows.data(), portRows.size(), txn);
        if (!portBatch.ok()) {
            HCCL_VM_ERROR("batch insert ports failed: {}", portBatch.diagnostic);
            return HcclVmResult::HCCL_SIM_E_INTERNAL;
        }
        // EndPoint：只为本批 Port 中非空 EID 且解码非全零者构造（字段、func_id
        // 与 IP 完全复用原规则；空EID端口(如rootinfo中被裁剪层的端口)不绑定）
        std::vector<sim::runtime::EndPoint> endpointRows;
        endpointRows.reserve(portRefs.size());
        std::vector<std::size_t> endpointPortIndex; // 第 k 个有效 endpoint 的 portRows 下标
        endpointPortIndex.reserve(portRefs.size());
        for (std::size_t i = 0; i < portRefs.size(); ++i) {
            const Port& portIn = *portRefs[i].first;
            if (portIn.eid.empty()) {
                continue;
            }
            // 按EID地址分配规则解码高位字段(die id/端口号), 便于日志追踪
            HCCL_VM_DEBUG(
                "port {} eid decoded: ip={}, die_id={:d}, port_num={:d}(PG={})", portIn.portId.c_str(),
                eidToIP(portIn.eid).c_str(), eid_die_id(portIn.eid), eid_port_num(portIn.eid),
                eid_is_pg_port(portIn.eid));
            // 全零 EID 不绑定 EndPoint（原 memcmp 规则；构造前预检，不产生废弃行）
            std::array<uint8_t, sizeof(sim::runtime::EndPoint::eid)> decodedEid{};
            eid_hex_to_uint8(portIn.eid.c_str(), decodedEid.data());
            if (std::all_of(decodedEid.begin(), decodedEid.end(), [](const uint8_t byte) {
                    return byte == 0;
                })) {
                continue;
            }
            endpointRows.emplace_back();
            FillEndPoint(endpointRows.back(), portRefs[i].second, portIn);
            endpointPortIndex.push_back(i);
        }
        if (!endpointRows.empty()) {
            auto endpointBatch = HcclSim::Storage::InsertMany(endpointRows.data(), endpointRows.size(), txn);
            if (!endpointBatch.ok()) {
                HCCL_VM_ERROR("batch insert endpoints failed: {}", endpointBatch.diagnostic);
                return HcclVmResult::HCCL_SIM_E_INTERNAL;
            }
            for (const auto& endPoint : endpointRows) {
                ipToEndpointId_.emplace(std::string(endPoint.ip_addr), endPoint.id);
            }
            // EndPointPortMapping：按本批有效 endpoint 顺序，对应同位 portId
            std::vector<sim::runtime::EndPointPortMapping> mappingRows;
            mappingRows.reserve(endpointRows.size());
            for (std::size_t k = 0; k < endpointRows.size(); ++k) {
                mappingRows.emplace_back();
                mappingRows.back().port_id = portRows[endpointPortIndex[k]].id;
                mappingRows.back().endpoint_id = endpointRows[k].id;
                mappingRows.back().id = 0;
            }
            auto mappingBatch = HcclSim::Storage::InsertMany(mappingRows.data(), mappingRows.size(), txn);
            if (!mappingBatch.ok()) {
                HCCL_VM_ERROR("batch insert endpoint port mappings failed: {}", mappingBatch.diagnostic);
                return HcclVmResult::HCCL_SIM_E_INTERNAL;
            }
        }
        return HcclVmResult::HCCL_SIM_SUCCESS;
    };

    // 跨叶子节点(board)累积的批量写入窗口: 每窗最多128个device。
    // POD形态下每个board只有8卡, 若按board独立成批, 1024卡集群会退化为128次小批插入,
    // 故跨board累积至窗口满(或全部处理完)再冲刷, 保持与改造前相同的批量粒度
    constexpr std::size_t kDeviceWindowRows = 128U;
    std::vector<sim::runtime::Device> deviceRows;
    // 待写入端口: (Port指针, 所属device在deviceRows中的下标)
    std::vector<std::pair<const Port*, std::size_t>> pendingPortRefs;

    auto flushDeviceWindow = [&]() -> HcclVmResult {
        if (deviceRows.empty()) {
            return HcclVmResult::HCCL_SIM_SUCCESS;
        }
        // Device 批：按输入 device 顺序原位构造最终 T，ID 回填
        auto deviceBatch = HcclSim::Storage::InsertMany(deviceRows.data(), deviceRows.size(), writeTxn);
        if (!deviceBatch.ok()) {
            HCCL_VM_ERROR("batch insert devices failed: {}", deviceBatch.diagnostic);
            return HcclVmResult::HCCL_SIM_E_INTERNAL;
        }
        // Ccu 批：每 device 依次 die=0、1（device_id 来自 Device 批回填）
        std::vector<sim::runtime::Ccu> ccuRows;
        ccuRows.reserve(deviceRows.size() * 2U);
        for (const auto& deviceRow : deviceRows) {
            for (uint32_t dieId = 0; dieId < 2U; ++dieId) {
                ccuRows.emplace_back();
                FillCcu(ccuRows.back(), deviceRow.id, dieId);
            }
        }
        auto ccuBatch = HcclSim::Storage::InsertMany(ccuRows.data(), ccuRows.size(), writeTxn);
        if (!ccuBatch.ok()) {
            HCCL_VM_ERROR("batch insert ccus failed: {}", ccuBatch.diagnostic);
            return HcclVmResult::HCCL_SIM_E_INTERNAL;
        }
        // Port/EndPoint/Mapping：按 ≤kInsertBatchMaxRows 分块冲刷(每块内 Port→EndPoint→Mapping)
        std::vector<std::pair<const Port*, uint64_t>> portRefs;
        portRefs.reserve(
            std::min(pendingPortRefs.size(), static_cast<std::size_t>(HcclSim::Storage::kInsertBatchMaxRows)));
        for (const auto& [port, deviceRowIdx] : pendingPortRefs) {
            portRefs.emplace_back(port, deviceRows[deviceRowIdx].id);
            if (portRefs.size() == HcclSim::Storage::kInsertBatchMaxRows) {
                if (flushPortBatch(portRefs, writeTxn) != HcclVmResult::HCCL_SIM_SUCCESS) {
                    return HcclVmResult::HCCL_SIM_E_INTERNAL;
                }
                portRefs.clear();
            }
        }
        if (!portRefs.empty() && flushPortBatch(portRefs, writeTxn) != HcclVmResult::HCCL_SIM_SUCCESS) {
            return HcclVmResult::HCCL_SIM_E_INTERNAL;
        }
        deviceRows.clear();
        pendingPortRefs.clear();
        return HcclVmResult::HCCL_SIM_SUCCESS;
    };

    for (const auto& superPod : network_.superPods) {
        for (const auto& server : superPod.servers) {
            // 每个叶子节点(server形态=server自身; POD形态=board)一行Server + 一行Host
            for (const auto& board : server.boards) {
                if (board.hostEid.empty()) {
                    HCCL_VM_ERROR(
                        "host eid is empty for superpod{:d}/server{:d}/board{:d}", superPod.superPodId, server.serverId,
                        board.boardId);
                    (void)writeTxn.Rollback();
                    return HcclVmResult::HCCL_SIM_E_INTERNAL;
                }
                auto serverKey = InitServer(superPod.superPodId, server, board, writeTxn);
                if (serverKey == 0U) {
                    // 事务已由 DML 失败路径熔断回滚（不在 FAILED 事务上重复 Rollback）
                    HCCL_VM_ERROR("init server {}/board {} topo data failed", server.serverId, board.boardId);
                    return HcclVmResult::HCCL_SIM_E_INTERNAL;
                }
                serverIdx2Key_[superPod.superPodId][server.serverId][board.boardId] = serverKey;

                for (const auto& device : board.devices) {
                    deviceRows.emplace_back();
                    FillDevice(deviceRows.back(), serverKey, device);
                    std::size_t deviceRowIdx = deviceRows.size() - 1;
                    for (const auto& port : device.ports) {
                        pendingPortRefs.emplace_back(&port, deviceRowIdx);
                    }
                    if (deviceRows.size() >= kDeviceWindowRows) {
                        if (flushDeviceWindow() != HcclVmResult::HCCL_SIM_SUCCESS) {
                            return HcclVmResult::HCCL_SIM_E_INTERNAL;
                        }
                    }
                }
            }
        }
    }
    if (flushDeviceWindow() != HcclVmResult::HCCL_SIM_SUCCESS) {
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    if (!writeTxn.Commit().ok()) {
        HCCL_VM_ERROR("commit static topo batch write transaction failed");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    HCCL_VM_DEBUG("InitClusterStaticTopoData completed successfully!");
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

uint64_t AscendClusterTopoParser::InitServer(
    uint32_t superPodId, const Server& server, const Board& board, HcclSim::Storage::Transaction& txn)
{
    HCCL_VM_DEBUG("Enter InitServer: {}/{}, eid= {}", server.serverId, board.boardId, board.hostEid.c_str());
    // 值初始化: Server/Host 中存在未显式赋值的字段(used_dev_num/version/arch)与定长char数组尾字节,
    // 不置零会把栈上残留值写入DB(实测未编入通信域的节点 used_dev_num 为随机大数), 且定长数组
    // 未终止会污染引擎侧的字节比较/落盘
    sim::runtime::Server serverDb{};
    serverDb.pod_id = superPodId;
    serverDb.server_id = server.serverId;
    // board_id: server形态恒为0; POD形态为POD内board序号(每board一行Server, 即一个独立host节点)
    serverDb.board_id = static_cast<uint32_t>(board.boardId < 0 ? 0 : board.boardId);
    // used_dev_num: 编入通信域的卡数, 由InitDynamicModelData在mock-comm时按叶子节点回填;
    // 未编入时为0(业务侧aclrtGetDeviceCount()据此判定该节点未编域)
    serverDb.used_dev_num = 0;
    strncpy(serverDb.hardware_type, board.hardwareType.c_str(), sizeof(serverDb.hardware_type) - 1);
    // port_type: 端口类型(x2/x4), IR缺省已为"x4"; 仅性能仿真消费(x2端口带宽减半)
    const char* portType = board.portType.empty() ? "x4" : board.portType.c_str();
    strncpy(serverDb.port_type, portType, sizeof(serverDb.port_type) - 1);
    serverDb.port_type[sizeof(serverDb.port_type) - 1] = '\0';
    serverDb.id = 0;
    auto serverInserted = HcclSim::Storage::Insert(serverDb, txn);
    if (!serverInserted.ok() || !serverInserted.value.has_value()) {
        return 0;
    }
    auto serverKey = serverInserted.value->value;

    sim::runtime::Host host{};
    host.server_id = serverKey;
    strcpy(host.ip_addr, eidToIP(board.hostEid).c_str());
    // 按EID地址分配规则解码高位字段(super pod id[8:5]), 便于日志追踪
    HCCL_VM_DEBUG("host eid decoded: ip={}, superpod_id[8:5]={:d}", host.ip_addr, eid_superpod_id_high(board.hostEid));
    host.id = 0;
    if (!HcclSim::Storage::Insert(host, txn).ok()) {
        return 0;
    }

    return serverKey;
}

uint64_t AscendClusterTopoParser::FindEndPointIdByIp(const std::string& ip)
{
    // 优先查内存索引(InitPort构建), O(1)避免对EndPoint表全表扫描
    auto it = ipToEndpointId_.find(ip);
    if (it != ipToEndpointId_.end()) {
        return it->second;
    }
    // 回退DB查询(兼容索引未覆盖的EndPoint), 并回填索引加速后续查询
    auto ret = sim::runtime::Db::GetOneByPred<sim::runtime::EndPoint>(
        HcclSim::Storage::Eq(&sim::runtime::EndPoint::ip_addr, ip));
    if (!ret.ok()) {
        return 0;
    }
    ipToEndpointId_.emplace(ip, ret->id);
    return ret->id;
}

HcclVmResult AscendClusterTopoParser::AddLinkInfo(
    const LinkPortRef* srcPort, const LinkPortRef* dstPort, uint32_t netLayer, const std::vector<Protocol>& protocols)
{
    (void)protocols;
    uint64_t srcEndPointId = 0;
    if (srcPort != nullptr && srcPort->localId != -1 && !srcPort->eid.empty()) {
        std::string srcIp = eidToIP(srcPort->eid);
        srcEndPointId = FindEndPointIdByIp(srcIp);
        if (srcEndPointId == 0) {
            HCCL_VM_ERROR("cannot find endPoint by ip {}", srcIp);
            return HCCL_SIM_E_NOT_FOUND;
        }
        // 更新Endpoint status
        if (!sim::runtime::Db::Update<sim::runtime::EndPoint>(
                 HcclSim::Storage::Eq(&sim::runtime::EndPoint::id, srcEndPointId),
                 HcclSim::Storage::Set(&sim::runtime::EndPoint::status, uint64_t{1}))
                 .ok()) {
            return HcclVmResult::HCCL_SIM_E_INTERNAL;
        }
    }

    uint64_t dstEndPointId = 0;
    if (dstPort != nullptr && dstPort->localId != -1 && !dstPort->eid.empty()) {
        std::string dstIp = eidToIP(dstPort->eid);
        dstEndPointId = FindEndPointIdByIp(dstIp);
        if (dstEndPointId == 0) {
            HCCL_VM_ERROR("cannot find endPoint by ip {}", dstIp);
            return HCCL_SIM_E_NOT_FOUND;
        }
        // 更新Endpoint status
        if (!sim::runtime::Db::Update<sim::runtime::EndPoint>(
                 HcclSim::Storage::Eq(&sim::runtime::EndPoint::id, dstEndPointId),
                 HcclSim::Storage::Set(&sim::runtime::EndPoint::status, uint64_t{1}))
                 .ok()) {
            return HcclVmResult::HCCL_SIM_E_INTERNAL;
        }
    }

    sim::runtime::Link link{};
    link.local_endpoint_id = srcEndPointId;
    link.remote_endpoint_id = dstEndPointId;
    link.net_layer = netLayer;
    link.type = 0;
    link.id = 0;
    if (!sim::runtime::Db::Add<sim::runtime::Link>(link).ok()) {
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

// 解析用户提供的ranktable文件，得到topoMeta信息，并初始化通信域相关数据库表项
HcclVmResult AscendClusterTopoParser::ParseRanktable(const std::string& ranktablePath, TopoMeta& topoMeta)
{
    std::ifstream ifs(ranktablePath);
    if (!ifs.is_open()) {
        HCCL_VM_ERROR("failed to open ranktable file: {}", ranktablePath);
        return HcclVmResult::HCCL_SIM_E_OPEN_FILE_FAILURE;
    }

    json rankTable;
    try {
        rankTable = json::parse(ifs);
    } catch (const json::exception& e) {
        HCCL_VM_ERROR("failed to parse ranktable json: {}", e.what());
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    ifs.close();

    if (!rankTable.contains("rank_list") || !rankTable["rank_list"].is_array()) {
        HCCL_VM_ERROR("rank_list not found or not array in ranktable");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    // (superPodId, serverId, boardId) → 该叶子节点内的设备物理id列表
    std::map<uint32_t, std::map<uint32_t, std::map<uint32_t, NodeMeta>>> spSrvBoardDevices;

    for (const auto& rank : rankTable["rank_list"]) {
        uint32_t localId = rank.value("local_id", 0);
        // 入库topoMeta的是**board内物理id**(DB的Device.physical_id口径): 取device_id字段;
        // POD形态下local_id为POD全局id(= board_id * 每board卡数 + board内物理id), 不能直接使用。
        // 兼容无device_id字段的旧ranktable(当时local_id与device_id同值)
        uint32_t devPhyId = rank.contains("device_id") ? rank.value("device_id", 0U) : localId;
        const auto& levelList = rank.value("level_list", json::array());

        uint32_t spIdx = 0;
        uint32_t srvIdx = 0;
        // boardIdx: POD形态(4层拓扑)下layer0的net_instance_id携带board_{b}段;
        // server形态(3层拓扑)无board段, 归属隐式board0
        uint32_t boardIdx = 0;
        bool found = false;

        for (const auto& level : levelList) {
            if (level.value("net_layer", -1) == 0) {
                std::string netInstanceId = level.value("net_instance_id", "");
                // POD形态 (net_layer=0, board作用域): sp_{spIdx}_srv_{srvIdx}_board_{boardIdx}_idx_{niid}
                unsigned int spTmp = 0;
                unsigned int srvTmp = 0;
                unsigned int boardTmp = 0;
                unsigned int idxTmp = 0;
                if (sscanf(netInstanceId.c_str(), "sp_%u_srv_%u_board_%u_idx_%u", &spTmp, &srvTmp, &boardTmp, &idxTmp)
                    == 4) {
                    spIdx = spTmp;
                    srvIdx = srvTmp;
                    boardIdx = boardTmp;
                    found = true;
                    break;
                }
                // server形态 / POD形态pod作用域: sp_{spIdx}_srv_{srvIdx}_idx_{niid}
                // (sscanf在"_idx_"前停止, 可同时兼容旧格式sp_{spIdx}_srv_{srvIdx}_lay_{layer}_idx_{niid})
                if (sscanf(netInstanceId.c_str(), "sp_%u_srv_%u", &spIdx, &srvIdx) == 2) {
                    boardIdx = 0;
                    found = true;
                    break;
                }
                // Backward compatibility: superPod{spIdx}_rack{srvIdx}
                if (sscanf(netInstanceId.c_str(), "superPod%u_rack%u", &spIdx, &srvIdx) == 2) {
                    boardIdx = 0;
                    found = true;
                    break;
                }
            }
        }

        if (!found) {
            for (const auto& level : levelList) {
                const auto& addrList = level.value("rank_addr_list", json::array());
                if (!addrList.empty()) {
                    std::string addr = addrList[0].value("addr", "");
                    std::string eidKey = "0x" + addr;
                    auto it = network_.eidIndex.find(eidKey);
                    if (it != network_.eidIndex.end() && !it->second.empty()) {
                        spIdx = static_cast<uint32_t>(std::get<0>(it->second[0]));
                        srvIdx = static_cast<uint32_t>(std::get<1>(it->second[0]));
                        boardIdx = static_cast<uint32_t>(std::get<2>(it->second[0]));
                        found = true;
                        break;
                    }
                }
            }
        }

        if (!found) {
            HCCL_VM_WARN("rank local_id={} cannot determine superpod/server/board, skip", localId);
            continue;
        }

        spSrvBoardDevices[spIdx][srvIdx][boardIdx].push_back(devPhyId);
    }

    topoMeta.clear();
    if (spSrvBoardDevices.empty()) {
        HCCL_VM_ERROR("no valid device found in ranktable");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    for (const auto& [spIdx, srvMap] : spSrvBoardDevices) {
        for (const auto& [srvIdx, boardMap] : srvMap) {
            for (const auto& [boardIdx, nodeMeta] : boardMap) {
                topoMeta[spIdx][srvIdx][boardIdx] = nodeMeta;
            }
        }
    }

    // EID回退定位依赖InitClusterTopo已构建的network_索引; 缺失时仅按net_instance_id定位
    // (原实现在此直接报错, 会导致"start后再mock-comm ranktable"必然失败, 现降级为告警)
    if (network_.superPods.empty()) {
        HCCL_VM_WARN("cluster IR is empty when parsing ranktable, EID-based rank location is unavailable "
                     "(only net_instance_id based location works)");
    }

    HCCL_VM_DEBUG(
        "parsed ranktable: superpodNum={}, leafNodeNum={}", topoMeta.size(), ShmGetPhyDeviceTotalCount(topoMeta));
    return HcclVmResult::HCCL_SIM_SUCCESS;
}
