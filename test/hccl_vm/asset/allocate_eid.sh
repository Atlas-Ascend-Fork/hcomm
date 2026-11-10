#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 统一分配EID/IP地址
# 功能：
#   1. 遍历集群输出目录下所有叶子节点的rootinfo json文件
#      (server形态: superpod{s}/server{v}/; POD形态: superpod{s}/server{v}/board{b}/;
#       POD形态的topo.json为POD级单份, 位于server目录 superpod{s}/server{v}/,
#       layer0板内直连校验时从server目录读取)
#   2. 根据IP地址分配规则，为每个叶子节点的portGroup分配IP地址
#   3. 将IP地址按EID分配规则构造EID字符串，修改rootinfo文件中的addr字段；并校验EID全局唯一
#   4. 重写rootinfo文件中的net_instance_id字段: **继承**生成时写入的作用域(scope)模板,
#      仅把sp/srv/board替换为当前集群位置(idx_k沿用生成时配置的net_instance序号):
#      - scope = board    : "sp_{sp}_srv_{srv}_board_{b}_idx_{k}" (POD形态板内实例, 每board唯一)
#      - scope = pod      : "sp_{sp}_srv_{srv}_idx_{k}" (server形态layer0 / POD形态跨board实例)
#      - scope = superpod : "sp_{sp}_idx_{k}" (跨server/POD时出框的net_instance_id相同, 即这一层内互通)
#      - scope = cluster  : "cluster" (集群层, 固定值)
#      不按net_layer硬编码模板, 因此无需感知各拓扑配置的net_instance_scope;
#      并校验: POD形态下layer0存在PEER2PEER板内直连时, 其net_instance_id必须每board唯一
#
# IP地址分配规则 (AA.BB.CC.DD), 按集群规模动态分配
# (设计文档: ai_skills/skills/generate-server-topo-json/references/IPV4_address_dynamic_allocation_design.md):
#   S = 超节点数, N = 单超节点server/POD数(取各超节点最大值), B = 单POD内board数(POD形态)
#   默认模式 (S<=255且N<=255): AA = superpod_index + 1, BB = server_index + 1
#   动态模式 (S>255或N>255):   W_bb = ceil(log2(N+1)), W_aa = 16 - W_bb,
#     V = ((sp+1)<<W_bb) + (srv+1), AA = V>>8, BB = V&0xFF
#   server形态(3层): CC:DD = device port计数器(共16bit, 容量65535), 从0.1开始
#     Host IP: AA.BB.0.0 (每个server固定一个host IP，仅供host使用)
#   POD形态(4层, board → pod → 超节点 → cluster): CC = board_id, DD = board内port计数器(容量255)
#     Host IP: AA.BB.{board_id}.0 (每个board一个独立host节点, POD自身不分配host IP)
#   Device port IPs: 从 AA.BB.CC.1 起依次递增(d2h端口也从此段分配)
#   打印位宽分配表(POD形态含board段); 动态模式下另落盘 ip_addr_allocation.txt
#
# IP→EID转换规则 (按EID地址分配规则，在IPV4基础上填充高位字段):
#   EID布局(位序从高到低, 共128bit):
#     127~92: 保留位, 固定全0
#     91~88:  super pod id[8:5] (super pod id的第8~5位, 低5位由IPV4的superpod段承载)
#     87~86:  die id (取rank_addr条目首个"die/port"端口的die号;
#             d2h等不带端口序号的端口取pin_map中h2d端口所属die, 保留原EID高位字段)
#     85~80:  端口号, 端口类型按文档"PG端口和物理端口说明"划分:
#             - PG端口: 仅当port_group对应ports列表中有多个物理端口时, 固定0x3F
#             - 普通物理端口: 单个"die/port"端口的portGroup或未在port_group中指定的端口,
#               端口号取物理端口号(0-62); d2h等端口名不带端口序号的端口, 端口号来自
#               配置文件pin_map中value=3对应的portId, 由generate_server_topo.sh写入
#               EID高位字段, 本脚本重分配IP时保留该高位字段
#     79~32:  保留位, 固定全0
#     31~0:   IPV4地址 (AA.BB.CC.DD; POD形态下CC段即board段)
#   输出格式: "0x" + 32位十六进制字符串
#   示例: 1.1.0.1 (sp0, die1, port0) → 0x00000000004000000000000001010001
#         1.1.0.8 (sp0, PG端口)     → 0x00000000003f00000000000001010008

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 脚本日志级别定义与C++日志一致: 0-trace 1-debug 2-info 3-warn 4-error 5-critical 6-off
# 级别来源优先级: 环境变量SCRIPT_LOG_LEVEL > config/log_config.yaml中script段console_level > 默认info(2)
error_exit() {
    if [[ "$SCRIPT_LOG_LEVEL" -le 4 ]]; then
        echo -e "${RED}[错误] $1${NC}" >&2
    fi
    exit 1
}

success_msg() {
    if [[ "$SCRIPT_LOG_LEVEL" -le 2 ]]; then
        echo -e "${GREEN}[成功] $1${NC}"
    fi
}

info_msg() {
    if [[ "$SCRIPT_LOG_LEVEL" -le 2 ]]; then
        echo -e "${BLUE}[信息] $1${NC}"
    fi
}

warn_msg() {
    if [[ "$SCRIPT_LOG_LEVEL" -le 3 ]]; then
        echo -e "${YELLOW}[警告] $1${NC}"
    fi
}

debug_msg() {
    if [[ "$SCRIPT_LOG_LEVEL" -le 1 ]]; then
        echo -e "$1"
    fi
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "${SCRIPT_DIR}")"
OUTPUT_DIR="${BASE_DIR}/config/network/cluster"

# 从config/log_config.yaml的script段读取console_level作为脚本日志级别;
# 环境变量SCRIPT_LOG_LEVEL已设置合法值时优先使用(支持父脚本/用户注入), 配置缺失时默认info(2)
load_log_level() {
    if [[ "${SCRIPT_LOG_LEVEL:-}" =~ ^[0-6]$ ]]; then
        return
    fi
    local lvl=""
    local cfg="${BASE_DIR}/config/log_config.yaml"
    if [[ -f "$cfg" ]]; then
        lvl=$(awk '/^script:[[:space:]]*(#.*)?$/ {in_script=1; next}
                   in_script && /^[^[:space:]#]/ {exit}
                   in_script && /^[[:space:]]*console_level:/ {
                       sub(/.*console_level:[[:space:]]*/, "")
                       sub(/[[:space:]]*#.*$/, "")
                       gsub(/[[:space:]]/, "")
                       print; exit}' "$cfg")
    fi
    if [[ "$lvl" =~ ^[0-6]$ ]]; then
        SCRIPT_LOG_LEVEL="$lvl"
    else
        SCRIPT_LOG_LEVEL=2
    fi
}
load_log_level
export SCRIPT_LOG_LEVEL

show_usage() {
    cat << EOF
用法: $(basename "$0") [OPTIONS]

选项:
    -d, --dir DIR     指定集群输出目录 (默认: ${OUTPUT_DIR})
    -h, --help        显示帮助信息

说明:
    遍历output_temp目录下所有rootinfo JSON文件，按照IP地址分配规则
    统一分配IP地址，按EID分配规则构造EID字符串后写入addr字段；
    并按net_instance_id配置规则，以superpod/server路径重写net_instance_id字段
    (net_layer=0为"sp_{sp}_srv_{srv}_idx_{k}", net_layer=1为"sp_{sp}_idx_{k}",
     net_layer>=2为"cluster", idx_k继承生成时配置的net_instance序号)。

    IP地址分配规则 (AA.BB.CC.DD), 按集群规模动态分配:
      S = 超节点数, N = 单超节点server数
      默认模式 (S<=255且N<=255): AA = superpod_index + 1, BB = server_index + 1
      动态模式 (S>255或N>255):   W_bb = ceil(log2(N+1)), W_aa = 16 - W_bb,
        V = ((sp+1)<<W_bb) + (srv+1), AA = V>>8, BB = V&0xFF
      CC:DD = device port计数器  (设备段，从0.1开始)
      Host IP: AA.BB.0.0         (每个server固定一个host IP，仅供host使用)
      Device port IPs: AA.BB.0.1, AA.BB.0.2, ... (依次递增，d2h端口也从此段分配)

    IP→EID转换 (按EID地址分配规则):
      127~92: 保留位, 全0
      91~88:  super pod id[8:5]
      87~86:  die id
      85~80:  端口号 (PG端口0x3F仅限多物理端口portGroup; 单物理端口portGroup/未分组端口
              取物理端口号0-62; d2h等端口名不带端口序号的端口, 端口号来自pin_map中
              value=3对应的portId, 保留原EID高位字段)
      79~32:  保留位, 全0
      31~0:   IPV4地址
      示例: 1.1.0.1 (sp0, die1, port0) → 0x00000000004000000000000001010001

示例:
    $(basename "$0")
    $(basename "$0") -d /tmp/my_output

EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -d|--dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            error_exit "未知参数: $1\n$(show_usage)"
            ;;
    esac
done

if [[ ! -d "$OUTPUT_DIR" ]]; then
    error_exit "输出目录不存在: $OUTPUT_DIR\n请先运行 generate_cluster_topo.sh 生成集群拓扑目录"
fi

debug_msg "=========================================="
debug_msg "  昇腾集群EID统一分配工具"
debug_msg "=========================================="
debug_msg "输出目录: ${OUTPUT_DIR}"
debug_msg ""

export OUTPUT_DIR

python3 << 'PYTHON_SCRIPT'
import json
import os
import re
import sys
import struct
import socket

# 脚本日志级别(与C++日志一致: 0-trace 1-debug 2-info 3-warn 4-error 5-critical 6-off):
# bash侧从config/log_config.yaml的script段读取console_level后经SCRIPT_LOG_LEVEL环境变量传入
try:
    LOG_LEVEL = max(0, min(6, int(os.environ.get('SCRIPT_LOG_LEVEL', '2'))))
except ValueError:
    LOG_LEVEL = 2

def log_error(msg):
    if LOG_LEVEL <= 4:
        print(msg)

def log_warn(msg):
    if LOG_LEVEL <= 3:
        print(msg)

def log_info(msg):
    if LOG_LEVEL <= 2:
        print(msg)

def log_debug(msg):
    if LOG_LEVEL <= 1:
        print(msg)

OUTPUT_DIR = os.environ.get('OUTPUT_DIR', '')
if not OUTPUT_DIR:
    log_error("[错误] OUTPUT_DIR 环境变量未设置")
    sys.exit(1)

URMA_EID_LEN = 16

# PG端口(portgroup端口)的固定端口号
EID_PG_PORT_NUM = 0x3F

def build_eid_string(ip, superpod_id=0, die_id=0, port_num=0):
    """
    将IP地址按EID地址分配规则构造EID字符串。

    EID布局(位序从高到低, 共128bit):
      127~92: 保留位, 固定全0
      91~88:  super pod id[8:5] (super pod id的第8~5位, 低5位由IPV4的superpod段承载)
      87~86:  die id
      85~80:  端口号 (普通物理端口0-62; PG端口固定0x3F)
      79~32:  保留位, 固定全0
      31~0:   IPV4地址 (AA.BB.CC.DD)

    端口类型按文档"PG端口和物理端口说明"划分:
      - PG端口: 仅当port_group对应ports列表中有多个物理端口时, 端口号固定0x3F
      - 普通物理端口: 单个"die/port"端口的portGroup或未在port_group中指定的端口,
        端口号取物理端口号; d2h等端口名不带端口序号的端口, 端口号来自pin_map中
        value=3对应的portId(保留原EID高位字段)

    示例:
        build_eid_string("1.1.0.1", 0, 1, 0)    -> "0x00000000004000000000000001010001"
        build_eid_string("1.1.0.8", 0, 0, 0x3F) -> "0x00000000003f00000000000001010008"
        build_eid_string("2.1.0.3", 1, 0, 2)    -> "0x00000000000200000000000002010003"
    """
    ip_bytes = socket.inet_pton(socket.AF_INET, ip)
    eid = bytearray(URMA_EID_LEN)
    eid[4] = (superpod_id >> 5) & 0xF                      # 91~88: super pod id[8:5]
    eid[5] = ((die_id & 0x3) << 6) | (port_num & 0x3F)     # 87~86: die id; 85~80: 端口号
    eid[12:16] = ip_bytes                                  # 31~0: IPV4地址
    hex_str = ''.join(f'{b:02x}' for b in eid)
    return f'0x{hex_str}'

def derive_eid_die_port(net_layer, ports):
    """
    从rootinfo的rank_addr条目(net_layer + ports列表)推导EID高位字段的die id与端口号。
    按文档"PG端口和物理端口说明"规则(与net_layer无关):
      - PG端口: 仅当port_group对应ports列表中有多个物理端口时, 端口号固定0x3F,
        die id取首个"die/port"端口的die号
      - 普通物理端口: ports列表为单个"die/port"端口(单物理端口portGroup或未分组的P2P端口)时,
        端口号取物理端口号(0-62)
      - d2h等端口名不带端口序号的端口或空端口列表: 不属于PG端口, 但无法从ports列表推导
        die id与端口号(端口号来自配置文件pin_map中value=3对应的portId, 由
        generate_server_topo.sh生成rootinfo时写入EID高位字段), 返回None,
        由调用方保留原EID中的die/port高位字段
    与generate_server_topo.sh中的推导规则保持一致
    """
    # 单个"die/port"端口: 按物理端口分配地址, 端口号取物理端口号
    if ports and len(ports) == 1 and '/' in ports[0]:
        die_str, port_str = ports[0].split('/', 1)
        return int(die_str) & 0x3, int(port_str) & 0x3F
    # 多个物理端口聚合的portGroup: PG端口, 端口号固定0x3F
    if ports and len(ports) > 1:
        die_id = 0
        if '/' in ports[0]:
            die_id = int(ports[0].split('/', 1)[0]) & 0x3
        return die_id, EID_PG_PORT_NUM
    # d2h等不带端口序号的端口/空端口列表: 无法从ports推导, 返回None
    return None

def extract_die_port_from_eid(eid_str):
    """
    从EID字符串("0x" + 32位十六进制)提取87~86位die id与85~80位端口号。
    用于d2h等端口名不带端口序号的端口: generate_server_topo.sh生成rootinfo时已按
    pin_map将die/port写入EID高位字段, 重分配IP时保留这些高位字段。
    无法解析时返回None。
    """
    if not isinstance(eid_str, str) or len(eid_str) != 34:
        return None
    s = eid_str.lower()
    if not s.startswith('0x'):
        return None
    try:
        byte5 = int(s[12:14], 16)
    except ValueError:
        return None
    return (byte5 >> 6) & 0x3, byte5 & 0x3F

def is_v4_mapped_eid(eid_str):
    """
    判断EID是否为纯v4映射形态(高12字节全0, 低32位为IPv4)。
    UBOE协议端口按文档"EID地址分配规则"注意项: hcomm业务强制使用IPV4地址,
    EID高位默认全0。generate_server_topo.sh生成rootinfo时UBOE条目已写入
    全0高位EID, 重分配IP时按此规则识别, 保持高位全0, 仅重写低32位IP地址。
    """
    if not isinstance(eid_str, str) or len(eid_str) != 34:
        return False
    s = eid_str.lower()
    if not s.startswith('0x'):
        return False
    try:
        return int(s[2:26], 16) == 0 and int(s[26:34], 16) != 0
    except ValueError:
        return False

def find_rootinfo_files(output_dir):
    rootinfo_files = []
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if 'rootinfo' in f and f.endswith('.json'):
                rootinfo_files.append(os.path.join(root, f))
    return rootinfo_files

def extract_indices(filepath):
    """从rootinfo路径解析 (superpod_id, server_id, board_id)。
    POD形态(4层拓扑)路径为 superpod{s}/server{v}/board{b}/xxx_rootinfo.json, board_id=b;
    server形态(3层拓扑)路径为 superpod{s}/server{v}/xxx_rootinfo.json, board_id=-1(表示无board层)。
    注意: board_id必须参与IP分配(CC段), 否则同一POD内各board会得到完全相同的IP/EID(全局重复)"""
    match = re.search(r'superpod(\d+)/server(\d+)/board(\d+)', filepath)
    if match:
        return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    match = re.search(r'superpod(\d+)/server(\d+)', filepath)
    if match:
        return (int(match.group(1)), int(match.group(2)), -1)
    return (-1, -1, -1)

def allocate_ip(aa, bb, counter):
    """server形态: CC.DD共16bit作为叶子节点内的endPoint计数器(容量65535)"""
    cc = (counter >> 8) & 0xFF
    dd = counter & 0xFF
    return f"{aa}.{bb}.{cc}.{dd}"

def allocate_ip_board(aa, bb, board_id, counter):
    """POD形态: CC段固定为board_id, DD段为board内endPoint计数器(容量255)。
    host IP为 AA.BB.{board_id}.0, device IP从 AA.BB.{board_id}.1 起递增"""
    if counter > 0xFF:
        log_error(f"[错误] superpod{aa - 1}/server{bb - 1}/board{board_id} 的endPoint数({counter})"
                  f"超出IP地址DD段容量(255), 请减少每board卡数或合并portGroup")
        sys.exit(1)
    return f"{aa}.{bb}.{board_id}.{counter}"

def parse_niid_k(net_instance_id):
    """从rootinfo的net_instance_id中解析idx_k(net_instance序号)。
    格式(按作用域scope):
      board    作用域: "sp_{sp}_srv_{srv}_board_{b}_idx_{k}"  (POD形态板内实例)
      pod      作用域: "sp_{sp}_srv_{srv}_idx_{k}"            (server形态layer0 / POD形态跨board实例)
      superpod 作用域: "sp_{sp}_idx_{k}"                      (layer1)
      cluster  作用域: "cluster"                              (layer>=2)
    兼容旧格式"sp_{sp}_srv_{srv}_lay_{layer}_idx_{k}"。
    解析失败(或net_layer>=2的固定值"cluster")时返回0"""
    if not isinstance(net_instance_id, str):
        return 0
    m = re.match(r'^sp_\d+_srv_\d+_board_\d+_idx_(\d+)$', net_instance_id)
    if m:
        return int(m.group(1))
    m = re.match(r'^sp_\d+_srv_\d+_idx_(\d+)$', net_instance_id)
    if m:
        return int(m.group(1))
    m = re.match(r'^sp_\d+_idx_(\d+)$', net_instance_id)
    if m:
        return int(m.group(1))
    m = re.match(r'^sp_\d+_srv_\d+_lay_\d+_idx_(\d+)$', net_instance_id)
    if m:
        return int(m.group(1))
    return 0

def detect_niid_scope(net_instance_id):
    """识别rootinfo既有net_instance_id的作用域(scope)模板。
    由generate_server_topo.sh按拓扑形态与该层链路类型推导写入(可被net_instance_scope覆盖):
      POD形态layer0有板内直连 → board; layer0为跨board交换网 → pod; layer1 → superpod; layer>=2 → cluster
      server形态layer0 → pod; layer1 → superpod; layer>=2 → cluster
    本脚本**继承**该模板(仅把sp/srv/board替换为当前集群位置), 不自行按net_layer硬编码,
    因此无需感知各拓扑的net_instance_scope配置。无法识别时返回None(回退按layer推导)"""
    if not isinstance(net_instance_id, str):
        return None
    if net_instance_id == 'cluster':
        return 'cluster'
    if re.match(r'^sp_\d+_srv_\d+_board_\d+_idx_\d+$', net_instance_id):
        return 'board'
    if re.match(r'^sp_\d+_srv_\d+_idx_\d+$', net_instance_id):
        return 'pod'
    if re.match(r'^sp_\d+_srv_\d+_lay_\d+_idx_\d+$', net_instance_id):
        return 'pod'   # 旧格式, 等价于pod(server)作用域
    if re.match(r'^sp_\d+_idx_\d+$', net_instance_id):
        return 'superpod'
    return None

def build_net_instance_id(sp_idx, srv_idx, board_idx, net_layer, niid, scope=None):
    """按作用域(scope)构造net_instance_id:
    - scope = board    : "sp_{sp}_srv_{srv}_board_{b}_idx_{niid}"  (POD形态板内实例, 每board唯一)
    - scope = pod      : "sp_{sp}_srv_{srv}_idx_{niid}"
    - scope = superpod : "sp_{sp}_idx_{niid}"
    - scope = cluster  : "cluster"
    scope为None时按net_layer回退到既有规则(layer0→pod, layer1→superpod, layer>=2→cluster),
    保证存量server形态产物逐字节不变。board_idx<0表示server形态(无board层)"""
    if scope is None:
        scope = 'cluster' if net_layer >= 2 else ('pod' if net_layer == 0 else 'superpod')
    # layer>=2 且作用域为cluster(或无法识别)时固定为"cluster";
    # 若生成侧显式声明了layer>=2的作用域(POD形态下layer2可能为POD间层), 则按声明保留层级归属信息
    if scope == 'cluster':
        return 'cluster'
    if scope == 'board':
        if board_idx < 0:
            log_error(f"[错误] net_instance_id作用域为board, 但路径中未解析到board目录 "
                      f"(superpod{sp_idx}/server{srv_idx}), 请确认POD形态产物已按board生成")
            sys.exit(1)
        return f'sp_{sp_idx}_srv_{srv_idx}_board_{board_idx}_idx_{niid}'
    if scope == 'pod':
        return f'sp_{sp_idx}_srv_{srv_idx}_idx_{niid}'
    return f'sp_{sp_idx}_idx_{niid}'

def find_leaf_topo_json(leaf_dir):
    """定位叶子节点使用的topo.json:
      - server形态: 即叶子目录自身的topo.json;
      - POD形态: topo.json为POD级单份, 位于server目录(board目录的上级);
        叶子目录自身存在topo.json时(旧产物)优先使用
    未找到返回None"""
    topo_path = os.path.join(leaf_dir, 'topo.json')
    if os.path.isfile(topo_path):
        return topo_path
    parent_topo_path = os.path.join(os.path.dirname(os.path.abspath(leaf_dir)), 'topo.json')
    if os.path.isfile(parent_topo_path):
        return parent_topo_path
    return None

def layer0_has_peer2peer(leaf_dir):
    """读取叶子节点对应的topo.json(POD形态为server目录下的POD级单份),
    判断net_layer=0是否存在PEER2PEER(板内/server内直连)边。
    用于校验: POD形态下layer0有直连时, 其net_instance_id必须每board唯一(board作用域),
    否则各board的板内直连组会在同一网络实例内被误合并"""
    topo_path = find_leaf_topo_json(leaf_dir)
    if topo_path is None:
        return False
    try:
        with open(topo_path, 'r') as f:
            topo = json.load(f)
    except (OSError, ValueError):
        return False
    for edge in topo.get('edge_list', []):
        if edge.get('net_layer') == 0 and edge.get('link_type') == 'PEER2PEER':
            return True
    return False

def layer0_p2p_crosses_board(leaf_dir, node_device_num):
    """读取POD级topo.json, 判断net_layer=0的PEER2PEER直连边是否存在跨board边
    (local_a与local_b归属不同board, 即 local_a//每board卡数 != local_b//每board卡数)。
    存在跨board直连时, layer0直连域为单一POD级实例(作用域pod, 各board共享同一niid);
    仅当直连全部为板内时, 各board才独立成网(作用域board, niid每board唯一)。
    node_device_num<=0(无法确定board边界)时返回False"""
    topo_path = find_leaf_topo_json(leaf_dir)
    if topo_path is None or not node_device_num or node_device_num <= 0:
        return False
    try:
        with open(topo_path, 'r') as f:
            topo = json.load(f)
    except (OSError, ValueError):
        return False
    for edge in topo.get('edge_list', []):
        if edge.get('net_layer') == 0 and edge.get('link_type') == 'PEER2PEER':
            la = edge.get('local_a')
            lb = edge.get('local_b')
            if la is None or lb is None:
                continue
            if la // node_device_num != lb // node_device_num:
                return True
    return False

# ============================================================
# IP分配模式判断与位宽计算 (按设计文档规则:
#   IPV4_address_dynamic_allocation_design.md 3.2/3.3节)
#   默认模式 (S<=255且N<=255): AA = sp_id + 1, BB = srv_id + 1
#   动态模式 (S>255或N>255):   W_bb = ceil(log2(N+1)), W_aa = 16 - W_bb,
#     V = ((sp_id+1)<<W_bb) + (srv_id+1), AA = V>>8, BB = V&0xFF
# ============================================================
def compute_ip_layout(s_num, n_num):
    """计算IP分配模式与AA.BB联合16bit位宽划分, 返回 (mode, w_aa, w_bb)"""
    if s_num <= 255 and n_num <= 255:
        return 'default', 8, 8
    # ceil(log2(N+1)) = N.bit_length() (保证 N <= 2^W_bb - 1)
    w_bb = max(1, n_num.bit_length())
    w_aa = 16 - w_bb
    if w_aa < 1 or s_num > (1 << w_aa) - 1:
        raise ValueError(
            f"集群规模超出AA.BB段总容量: S={s_num}, N={n_num} "
            f"(W_aa={w_aa}, 容量{(1 << w_aa) - 1 if w_aa >= 1 else 0}个超节点), "
            f"请缩减超节点数或增加每超节点server数")
    return 'dynamic', w_aa, w_bb

def compute_host_aa_bb(sp_idx, srv_idx, mode, w_bb):
    """按分配模式计算server的host IP地址段AA/BB (host IP = AA.BB.0.0)"""
    if mode == 'default':
        return sp_idx + 1, srv_idx + 1
    v = ((sp_idx + 1) << w_bb) + (srv_idx + 1)
    return (v >> 8) & 0xFF, v & 0xFF

rootinfo_files = find_rootinfo_files(OUTPUT_DIR)

if not rootinfo_files:
    log_warn("[警告] 未找到任何rootinfo JSON文件")
    sys.exit(0)

rootinfo_files.sort(key=extract_indices)

log_debug(f"[信息] 找到 {len(rootinfo_files)} 个rootinfo文件")
log_debug(f"[信息] IP→EID转换规则: 91~88位=super pod id[8:5], 87~86位=die id, "
      f"85~80位=端口号(多物理端口portGroup为PG端口0x3F, 单物理端口/未分组端口取物理端口号, "
      f"d2h等端口名不带序号的端口保留原EID高位字段), 31~0位=IPV4地址")
log_debug("")

# 从rootinfo路径提取的索引推导集群规模:
#   S = 超节点数(max sp_idx + 1), N = 单超节点server/POD数(取各超节点最大值),
#   B = 单POD内board数(POD形态; server形态为0, 表示无board层)
all_indices = [extract_indices(fp) for fp in rootinfo_files]
valid_indices = [(sp, srv, brd) for sp, srv, brd in all_indices if sp >= 0 and srv >= 0]
if not valid_indices:
    log_error("[错误] 无法从任何rootinfo路径提取 superpod<N>/server<M>[/board<B>] 索引")
    sys.exit(1)
S_NUM = max(sp for sp, _, _ in valid_indices) + 1
N_NUM = max(srv for _, srv, _ in valid_indices) + 1
_board_ids = [brd for _, _, brd in valid_indices if brd >= 0]
B_NUM = (max(_board_ids) + 1) if _board_ids else 0
# 形态一致性: 同一集群内不允许部分叶子节点有board目录、部分没有(会导致CC段语义冲突)
if _board_ids and len(_board_ids) != len(valid_indices):
    log_error(f"[错误] 集群内拓扑形态不一致: {len(_board_ids)}/{len(valid_indices)} 个rootinfo位于board目录下, "
              f"其余位于server目录下。同一集群必须统一为server形态(3层)或POD形态(4层)")
    sys.exit(1)
IS_POD_FORM = (B_NUM > 0)
if IS_POD_FORM and B_NUM > 256:
    log_error(f"[错误] 单POD内board数({B_NUM})超出IP地址CC段容量(256, board_id取值0~255)")
    sys.exit(1)

try:
    IP_MODE, W_AA, W_BB = compute_ip_layout(S_NUM, N_NUM)
except ValueError as e:
    log_error(f"[错误] {e}")
    sys.exit(1)

# 预扫描各叶子节点的endPoint数(rank_addr条目数), 用于位宽分配表展示与容量校验
E_MAX = 0
for filepath in rootinfo_files:
    sp_idx, srv_idx, board_idx = extract_indices(filepath)
    if sp_idx < 0 or srv_idx < 0:
        continue
    with open(filepath, 'r') as f:
        data = json.load(f)
    cnt = 0
    for rank in data.get('rank_list', []):
        for level in rank.get('level_list', []):
            cnt += len(level.get('rank_addr_list', []))
    E_MAX = max(E_MAX, cnt)
# POD形态: DD段仅8bit, 单board endPoint上限255(server形态CC.DD共16bit, 上限65535)
if IS_POD_FORM and E_MAX > 0xFF:
    log_error(f"[错误] 单个board的endPoint数({E_MAX})超出IP地址DD段容量(255), "
              f"请减少每board卡数或合并portGroup")
    sys.exit(1)
if not IS_POD_FORM and E_MAX > 0xFFFF:
    log_error(f"[错误] 单个server的endPoint数({E_MAX})超出IP地址CC.DD段容量(65535)")
    sys.exit(1)

def build_alloc_table():
    """生成位宽分配表(默认模式简化表/动态模式完整表, 按设计文档3.5/3.6节格式)。
    POD形态(4层拓扑)额外打印board段: CC=board_id、DD=board内endPoint计数,
    host IP = AA.BB.{board_id}.0(每board一个独立host节点)"""
    sep = "=" * 58
    lines = [sep]
    scale_desc = (f"集群规模: 超节点数 S={S_NUM}, server/POD数 N={N_NUM}, "
                  + (f"board数/POD B={B_NUM}, " if IS_POD_FORM else "")
                  + f"endPoint数/{'board' if IS_POD_FORM else 'server'} E={E_MAX}")
    if IP_MODE == 'default':
        lines.append("        IPV4 地址段位宽分配表（默认模式，兼容现行规则）")
        lines.append(sep)
        lines.append(scale_desc)
        lines.append("分配模式: 默认模式（S <= 255 且 N <= 255）")
        lines.append(f"拓扑形态: {'POD形态（4层: board → pod → 超节点 → cluster）' if IS_POD_FORM else 'server形态（3层: server → 超节点 → cluster）'}")
        lines.append("")
        lines.append("| 段名        | 字段  | 位宽 | 容量  | 取值范围       |")
        lines.append("|-------------|-------|------|-------|---------------|")
        lines.append("| superpod段  | AA    | 8    | 255   | 1 ~ S         |")
        lines.append("| server段    | BB    | 8    | 255   | 1 ~ N         |")
        if IS_POD_FORM:
            lines.append(f"| board段     | CC    | 8    | 256   | 0 ~ {B_NUM - 1}        |")
            lines.append("| device段    | DD    | 8    | 255   | 每board的endPoint |")
        else:
            lines.append("| device段    | CC.DD | 16   | 65535 | 0.1 ~ 255.255 |")
        lines.append("-" * 58)
        lines.append("公式: AA = sp_id + 1, BB = srv_id + 1"
                     + (", CC = board_id" if IS_POD_FORM else ""))
    else:
        if S_NUM > 255 and N_NUM > 255:
            reason = f"S={S_NUM} > 255 且 N={N_NUM} > 255"
        elif S_NUM > 255:
            reason = f"S={S_NUM} > 255"
        else:
            reason = f"N={N_NUM} > 255"
        lines.append("        IPV4 地址段位宽分配表（动态分配模式）")
        lines.append(sep)
        lines.append(scale_desc)
        lines.append(f"分配模式: 动态模式（触发原因: {reason}）")
        lines.append(f"拓扑形态: {'POD形态（4层: board → pod → 超节点 → cluster）' if IS_POD_FORM else 'server形态（3层: server → 超节点 → cluster）'}")
        lines.append("")
        lines.append("| 段名        | 字段           | 位宽    | 容量  | 取值范围              |")
        lines.append("|-------------|----------------|---------|-------|-----------------------|")
        lines.append(f"| superpod段  | AA.BB联合高位  | W_aa={W_AA}  | {1 << W_AA if W_AA >= 1 else 0:<5} | sp_field: 1 ~ S      |")
        lines.append(f"| server段    | AA.BB联合低位  | W_bb={W_BB}  | {1 << W_BB:<5} | srv_field: 1 ~ N     |")
        if IS_POD_FORM:
            lines.append(f"| board段     | CC             | 8       | 256   | board_id: 0 ~ {B_NUM - 1}     |")
            lines.append("| device段    | DD             | 8       | 255   | 每个board的endPoint   |")
        else:
            lines.append("| device段    | CC.DD          | 16      | 65535 | 每个server的endPoint  |")
        lines.append("-" * 58)
        lines.append(f"联合编号: V = ((sp_id + 1) << {W_BB}) + (srv_id + 1)")
        lines.append("AA = V >> 8, BB = V & 0xFF" + (", CC = board_id" if IS_POD_FORM else ""))
    aa0, bb0 = compute_host_aa_bb(0, 0, IP_MODE, W_BB)
    aaL, bbL = compute_host_aa_bb(S_NUM - 1, N_NUM - 1, IP_MODE, W_BB)
    if IS_POD_FORM:
        lines.append(f"示例: sp=0/srv=0/board=0 → {aa0}.{bb0}.0.0 (host), "
                     f"{aa0}.{bb0}.0.1 ~ {aa0}.{bb0}.0.{E_MAX} (device)")
        lines.append(f"示例: sp=0/srv=0/board={B_NUM - 1} → {aa0}.{bb0}.{B_NUM - 1}.0 (host)")
        lines.append(f"示例: sp={S_NUM - 1}/srv={N_NUM - 1}/board={B_NUM - 1} → "
                     f"{aaL}.{bbL}.{B_NUM - 1}.0 (host)")
        lines.append(f"叶子节点(host节点)总数: {S_NUM} × {N_NUM} × {B_NUM} = {S_NUM * N_NUM * B_NUM}")
    elif IP_MODE == 'default':
        lines.append(f"示例: sp=0/srv=0 → {aa0}.{bb0}.0.0 (host), "
                     f"{aa0}.{bb0}.0.1 ~ {aa0}.{bb0}.255.255 (device)")
        lines.append(f"示例: sp={S_NUM - 1}/srv={N_NUM - 1} → {aaL}.{bbL}.0.0 (host)")
    else:
        v0 = ((0 + 1) << W_BB) + 1
        vL = ((S_NUM - 1 + 1) << W_BB) + N_NUM
        lines.append(f"示例: sp=0/srv=0 → V={v0} → {aa0}.{bb0}.0.0 (host)")
        lines.append(f"示例: sp={S_NUM - 1}/srv={N_NUM - 1} → V={vL} → {aaL}.{bbL}.0.0 (host)")
    lines.append(sep)
    return "\n".join(lines)

# 分配地址前打印位宽分配表; 动态模式另落盘 ip_addr_allocation.txt
alloc_table = build_alloc_table()
log_info(alloc_table)
log_info("")
if IP_MODE == 'dynamic':
    table_path = os.path.join(OUTPUT_DIR, 'ip_addr_allocation.txt')
    try:
        with open(table_path, 'w') as f:
            f.write(alloc_table + "\n")
        log_info(f"[信息] 位宽分配表已保存: {table_path}")
        log_info("")
    except OSError as e:
        log_warn(f"[警告] 位宽分配表写入失败: {table_path} ({e})")

total_addrs = 0
# EID全局唯一性校验: 收集本次分配的所有EID, 出现重复即报错。
# (POD形态下若board段未参与IP分配, 同一POD内各board会得到完全相同的EID, 由此拦截)
allocated_eids = {}
# layer0网络实例归组校验: POD形态下若layer0存在板内直连(PEER2PEER), 各board的layer0
# net_instance_id必须互不相同(board作用域); 否则不同board的同号卡(local_id值域相同)
# 会在同一网络实例内互相冒充对端
pod_layer0_instances = {}

for filepath in rootinfo_files:
    sp_idx, srv_idx, board_idx = extract_indices(filepath)

    if sp_idx < 0 or srv_idx < 0:
        log_warn(f"  [警告] 无法从路径提取索引，跳过: {filepath}")
        continue

    aa, bb = compute_host_aa_bb(sp_idx, srv_idx, IP_MODE, W_BB)
    # CC段: POD形态为board_id(每board一个独立host节点); server形态恒为0(与既有规则一致)
    cc = board_idx if board_idx >= 0 else 0

    host_ip = f"{aa}.{bb}.{cc}.0"
    # host EID: die id=0, 端口号=0 (host非端口, 高位字段填0)
    host_eid = build_eid_string(host_ip, sp_idx, 0, 0)

    with open(filepath, 'r') as f:
        data = json.load(f)

    leaf_dir = os.path.dirname(filepath)
    leaf_has_p2p_l0 = layer0_has_peer2peer(leaf_dir)
    # 每board卡数即rootinfo的rank_count; 用于判断layer0直连是否跨board(决定scope应为board还是pod)
    node_device_num = data.get('rank_count', 0)
    leaf_l0_cross_board = leaf_has_p2p_l0 and layer0_p2p_crosses_board(leaf_dir, node_device_num)

    DEVICE_PORT_START = 1
    port_counter = DEVICE_PORT_START
    addr_count = 0
    first_device_eid = None
    last_device_eid = None

    for rank in data.get('rank_list', []):
        for level in rank.get('level_list', []):
            net_layer = level.get('net_layer', 0)
            # net_instance_id重写: **继承**rootinfo中既有字符串的作用域(scope)模板,
            # 仅把sp/srv/board替换为当前集群位置、idx_k沿用生成时配置的net_instance序号。
            # 不按net_layer硬编码模板, 因此POD形态下layer0的board段不会被丢掉,
            # 也无需感知各拓扑配置的net_instance_scope
            old_niid = level.get('net_instance_id')
            scope = detect_niid_scope(old_niid)
            new_niid = build_net_instance_id(
                sp_idx, srv_idx, board_idx, net_layer, parse_niid_k(old_niid), scope)
            level['net_instance_id'] = new_niid
            # 校验: POD形态下layer0有板内直连时, 该层net_instance_id必须每board唯一(board作用域)
            if board_idx >= 0 and net_layer == 0 and leaf_has_p2p_l0:
                if leaf_l0_cross_board:
                    # 跨board直连: layer0直连域为单一POD级实例, 作用域必须为pod(各board共享同一
                    # net_instance_id); 否则跨board直连边两端niid不一致, 会被业务侧过滤而失效
                    if scope != 'pod':
                        log_error(f"[错误] superpod{sp_idx}/server{srv_idx}/board{board_idx}: "
                                  f"net_layer=0存在PEER2PEER跨board直连, 但其net_instance_id为'{old_niid}'"
                                  f"(作用域{scope or '未知'})。跨board直连域为单一POD级实例, layer0作用域必须为pod"
                                  f"(不含board段), 否则跨board直连边两端net_instance_id不一致会被业务侧过滤。"
                                  f"请在server/pod拓扑配置中移除对layer0的net_instance_scope覆盖或改为pod")
                        sys.exit(1)
                    # 各board共享同一layer0实例(pod作用域), 不做每board唯一性校验
                else:
                    if scope != 'board':
                        log_error(f"[错误] superpod{sp_idx}/server{srv_idx}/board{board_idx}: "
                                  f"net_layer=0存在PEER2PEER板内直连, 但其net_instance_id为'{old_niid}'"
                                  f"(作用域{scope or '未知'}), 缺少board段。各board的local_id值域相同, "
                                  f"layer0网络实例必须每board唯一, 否则不同board的同号卡会被误判为同一实例内的对端。"
                                  f"请在server/pod拓扑配置中移除对layer0的net_instance_scope覆盖")
                        sys.exit(1)
                    key = (sp_idx, srv_idx, new_niid)
                    prev = pod_layer0_instances.get(key)
                    if prev is not None and prev != board_idx:
                        log_error(f"[错误] superpod{sp_idx}/server{srv_idx}内board{prev}与board{board_idx}"
                                  f"的layer0 net_instance_id重复('{new_niid}'), 违反每board唯一约束")
                        sys.exit(1)
                    pod_layer0_instances[key] = board_idx
            for addr_entry in level.get('rank_addr_list', []):
                # IP分配: POD形态 CC=board_id、DD=board内计数; server形态 CC.DD共16bit计数
                if board_idx >= 0:
                    ip_addr = allocate_ip_board(aa, bb, board_idx, port_counter)
                else:
                    ip_addr = allocate_ip(aa, bb, port_counter)
                # EID高位字段(die id/端口号)从rank_addr条目的net_layer与ports推导;
                # d2h等端口名不带端口序号的端口无法从ports推导, 保留原EID中已按
                # pin_map写入的die/port高位字段(generate_server_topo.sh生成)
                die_port = derive_eid_die_port(net_layer, addr_entry.get('ports', []))
                if die_port is None:
                    die_port = extract_die_port_from_eid(addr_entry.get('addr', ''))
                if die_port is None:
                    die_port = (0, 0)
                die_id, port_num = die_port
                # UBOE协议端口(原EID高位全0的v4映射形态): 保持高位全0(含superpod段),
                # 仅重写低32位IP地址, 见文档EID分配规则"注意"项
                if is_v4_mapped_eid(addr_entry.get('addr', '')):
                    sp_eid, die_id, port_num = 0, 0, 0
                else:
                    sp_eid = sp_idx
                eid_str = build_eid_string(ip_addr, sp_eid, die_id, port_num)
                # EID全局唯一性校验(UBOE等高位全0的v4映射形态由IP唯一性保证, 同样参与校验)
                dup = allocated_eids.get(eid_str)
                if dup is not None:
                    log_error(f"[错误] EID全局重复: {eid_str} 同时分配给 {dup} 与 "
                              f"superpod{sp_idx}/server{srv_idx}"
                              + (f"/board{board_idx}" if board_idx >= 0 else "")
                              + "。请检查IP分段规则(POD形态需CC段承载board_id)")
                    sys.exit(1)
                allocated_eids[eid_str] = (f"superpod{sp_idx}/server{srv_idx}"
                                           + (f"/board{board_idx}" if board_idx >= 0 else ""))
                addr_entry['addr'] = eid_str
                if first_device_eid is None:
                    first_device_eid = eid_str
                last_device_eid = eid_str
                port_counter += 1
                addr_count += 1

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    leaf_desc = (f"superpod{sp_idx}/server{srv_idx}"
                 + (f"/board{board_idx}" if board_idx >= 0 else ""))
    log_debug(f"  [成功] {leaf_desc}: 分配 {addr_count} 个EID")
    log_debug(f"         Host IP: {host_ip} → EID: {host_eid}")
    if first_device_eid:
        if board_idx >= 0:
            log_debug(f"         Device IP: {aa}.{bb}.{cc}.1 ~ {aa}.{bb}.{cc}.{(port_counter - 1) & 0xFF}")
        else:
            last_cc = ((port_counter - 1) >> 8) & 0xFF
            last_dd = (port_counter - 1) & 0xFF
            log_debug(f"         Device IP: {aa}.{bb}.0.1 ~ {aa}.{bb}.{last_cc}.{last_dd}")
        log_debug(f"         Device EID: {first_device_eid} ~ {last_device_eid}")

    total_addrs += addr_count

log_debug("")
log_debug(f"[信息] 共处理 {len(rootinfo_files)} 个叶子节点"
          f"({'board' if IS_POD_FORM else 'server'})rootinfo文件，分配 {total_addrs} 个EID(全局唯一)")

PYTHON_SCRIPT

if [[ "$SCRIPT_LOG_LEVEL" -le 2 ]]; then
    debug_msg ""
    debug_msg "=========================================="
    debug_msg "  分配完成"
    debug_msg "=========================================="
    debug_msg ""
fi
success_msg "EID统一分配完成！"
