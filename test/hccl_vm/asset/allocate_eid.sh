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
#   1. 遍历output_temp目录下所有rootinfo相关json文件
#   2. 根据IP地址分配规则，为每个server的portGroup分配IP地址
#   3. 将IP地址按EID分配规则构造EID字符串，修改rootinfo文件中的addr字段
#   4. 按net_instance_id配置规则，以superpod/server路径重写rootinfo文件中的net_instance_id字段:
#      - net_layer = 0:  "sp_{sp}_srv_{srv}_idx_{k}" (sp_i代表第i个super pod, srv_j代表第j个server,
#                        idx_k表示第k个net_instance, k继承生成时配置的net_instance序号, 未配置时默认为0)
#      - net_layer = 1:  "sp_{sp}_idx_{k}" (跨server时出框的net_instance_id相同, 即在这一层内互通)
#      - net_layer >= 2: "cluster" (集群层, 固定值)
#
# IP地址分配规则 (AA.BB.CC.DD), 按集群规模动态分配
# (设计文档: ai_skills/skills/generate-server-topo-json/references/IPV4_address_dynamic_allocation_design.md):
#   S = 超节点数, N = 单超节点server数(取各超节点最大值)
#   默认模式 (S<=255且N<=255): AA = superpod_index + 1, BB = server_index + 1
#   动态模式 (S>255或N>255):   W_bb = ceil(log2(N+1)), W_aa = 16 - W_bb,
#     V = ((sp+1)<<W_bb) + (srv+1), AA = V>>8, BB = V&0xFF
#   CC:DD = device port计数器  (设备段，从0.1开始)
#   Host IP: AA.BB.0.0         (每个server固定一个host IP，仅供host使用)
#   Device port IPs: AA.BB.0.1, AA.BB.0.2, ... (依次递增，d2h端口也从此段分配)
#   动态模式下打印位宽分配表并落盘 ip_addr_allocation.txt
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
#     31~0:   IPV4地址 (AA.BB.CC.DD)
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
    match = re.search(r'superpod(\d+)/server(\d+)', filepath)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return (-1, -1)

def allocate_ip(aa, bb, counter):
    cc = (counter >> 8) & 0xFF
    dd = counter & 0xFF
    return f"{aa}.{bb}.{cc}.{dd}"

def parse_niid_k(net_instance_id):
    """从rootinfo的net_instance_id中解析idx_k(net_instance序号)。
    新格式: net_layer=0为"sp_{sp}_srv_{srv}_idx_{k}", net_layer=1为"sp_{sp}_idx_{k}";
    兼容旧格式"sp_{sp}_srv_{srv}_lay_{layer}_idx_{k}"。
    解析失败(或net_layer>=2的固定值"cluster")时返回0"""
    if not isinstance(net_instance_id, str):
        return 0
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

def build_net_instance_id(sp_idx, srv_idx, net_layer, niid):
    """按文档net_instance_id配置规则构造net_instance_id:
    - net_layer = 0:  "sp_{sp}_srv_{srv}_idx_{niid}"
    - net_layer = 1:  "sp_{sp}_idx_{niid}"
    - net_layer >= 2: 固定为"cluster" (集群层)"""
    if net_layer >= 2:
        return 'cluster'
    if net_layer == 0:
        return f'sp_{sp_idx}_srv_{srv_idx}_idx_{niid}'
    return f'sp_{sp_idx}_idx_{niid}'

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
#   S = 超节点数(max sp_idx + 1), N = 单超节点server数(取各超节点最大值)
all_indices = [extract_indices(fp) for fp in rootinfo_files]
valid_indices = [(sp, srv) for sp, srv in all_indices if sp >= 0 and srv >= 0]
if not valid_indices:
    log_error("[错误] 无法从任何rootinfo路径提取 superpod<N>/server<M> 索引")
    sys.exit(1)
S_NUM = max(sp for sp, _ in valid_indices) + 1
N_NUM = max(srv for _, srv in valid_indices) + 1

try:
    IP_MODE, W_AA, W_BB = compute_ip_layout(S_NUM, N_NUM)
except ValueError as e:
    log_error(f"[错误] {e}")
    sys.exit(1)

# 预扫描各server的endPoint数(rank_addr条目数), 用于位宽分配表展示
E_MAX = 0
for filepath in rootinfo_files:
    sp_idx, srv_idx = extract_indices(filepath)
    if sp_idx < 0 or srv_idx < 0:
        continue
    with open(filepath, 'r') as f:
        data = json.load(f)
    cnt = 0
    for rank in data.get('rank_list', []):
        for level in rank.get('level_list', []):
            cnt += len(level.get('rank_addr_list', []))
    E_MAX = max(E_MAX, cnt)

def build_alloc_table():
    """生成位宽分配表(默认模式简化表/动态模式完整表, 按设计文档3.5节格式)"""
    sep = "=" * 58
    lines = [sep]
    if IP_MODE == 'default':
        lines.append("        IPV4 地址段位宽分配表（默认模式，兼容现行规则）")
        lines.append(sep)
        lines.append(f"集群规模: 超节点数 S={S_NUM}, server数/超节点 N={N_NUM}, "
                     f"endPoint数/server E={E_MAX}")
        lines.append("分配模式: 默认模式（S <= 255 且 N <= 255）")
        lines.append("")
        lines.append("| 段名        | 字段  | 位宽 | 容量  | 取值范围       |")
        lines.append("|-------------|-------|------|-------|---------------|")
        lines.append("| superpod段  | AA    | 8    | 255   | 1 ~ S         |")
        lines.append("| server段    | BB    | 8    | 255   | 1 ~ N         |")
        lines.append("| device段    | CC.DD | 16   | 65535 | 0.1 ~ 255.255 |")
        lines.append("-" * 58)
        lines.append("公式: AA = sp_id + 1, BB = srv_id + 1")
    else:
        if S_NUM > 255 and N_NUM > 255:
            reason = f"S={S_NUM} > 255 且 N={N_NUM} > 255"
        elif S_NUM > 255:
            reason = f"S={S_NUM} > 255"
        else:
            reason = f"N={N_NUM} > 255"
        lines.append("        IPV4 地址段位宽分配表（动态分配模式）")
        lines.append(sep)
        lines.append(f"集群规模: 超节点数 S={S_NUM}, server数/超节点 N={N_NUM}, "
                     f"endPoint数/server E={E_MAX}")
        lines.append(f"分配模式: 动态模式（触发原因: {reason}）")
        lines.append("")
        lines.append("| 段名        | 字段           | 位宽    | 容量  | 取值范围              |")
        lines.append("|-------------|----------------|---------|-------|-----------------------|")
        lines.append(f"| superpod段  | AA.BB联合高位  | W_aa={W_AA}  | {1 << W_AA if W_AA >= 1 else 0:<5} | sp_field: 1 ~ S      |")
        lines.append(f"| server段    | AA.BB联合低位  | W_bb={W_BB}  | {1 << W_BB:<5} | srv_field: 1 ~ N     |")
        lines.append("| device段    | CC.DD          | 16      | 65535 | 每个server的endPoint  |")
        lines.append("-" * 58)
        lines.append(f"联合编号: V = ((sp_id + 1) << {W_BB}) + (srv_id + 1)")
        lines.append("AA = V >> 8, BB = V & 0xFF")
    aa0, bb0 = compute_host_aa_bb(0, 0, IP_MODE, W_BB)
    aaL, bbL = compute_host_aa_bb(S_NUM - 1, N_NUM - 1, IP_MODE, W_BB)
    if IP_MODE == 'default':
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

for filepath in rootinfo_files:
    sp_idx, srv_idx = extract_indices(filepath)

    if sp_idx < 0 or srv_idx < 0:
        log_warn(f"  [警告] 无法从路径提取索引，跳过: {filepath}")
        continue

    aa, bb = compute_host_aa_bb(sp_idx, srv_idx, IP_MODE, W_BB)

    host_ip = f"{aa}.{bb}.0.0"
    # host EID: die id=0, 端口号=0 (host非端口, 高位字段填0)
    host_eid = build_eid_string(host_ip, sp_idx, 0, 0)

    with open(filepath, 'r') as f:
        data = json.load(f)

    DEVICE_PORT_START = 1
    port_counter = DEVICE_PORT_START
    addr_count = 0
    first_device_eid = None
    last_device_eid = None

    for rank in data.get('rank_list', []):
        for level in rank.get('level_list', []):
            net_layer = level.get('net_layer', 0)
            # 按文档net_instance_id配置规则, 以superpod/server路径重写net_instance_id:
            # idx_k继承rootinfo生成时配置的net_instance序号, sp/srv取自目录路径
            level['net_instance_id'] = build_net_instance_id(
                sp_idx, srv_idx, net_layer, parse_niid_k(level.get('net_instance_id')))
            for addr_entry in level.get('rank_addr_list', []):
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
                addr_entry['addr'] = eid_str
                if first_device_eid is None:
                    first_device_eid = eid_str
                last_device_eid = eid_str
                port_counter += 1
                addr_count += 1

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    log_debug(f"  [成功] superpod{sp_idx}/server{srv_idx}: "
          f"分配 {addr_count} 个EID")
    log_debug(f"         Host IP: {host_ip} → EID: {host_eid}")
    if first_device_eid:
        last_cc = ((port_counter - 1) >> 8) & 0xFF
        last_dd = (port_counter - 1) & 0xFF
        log_debug(f"         Device IP: {aa}.{bb}.0.1 ~ {aa}.{bb}.{last_cc}.{last_dd}")
        log_debug(f"         Device EID: {first_device_eid} ~ {last_device_eid}")

    total_addrs += addr_count

log_debug("")
log_debug(f"[信息] 共处理 {len(rootinfo_files)} 个文件，分配 {total_addrs} 个EID")

PYTHON_SCRIPT

if [[ "$SCRIPT_LOG_LEVEL" -le 2 ]]; then
    debug_msg ""
    debug_msg "=========================================="
    debug_msg "  分配完成"
    debug_msg "=========================================="
    debug_msg ""
fi
success_msg "EID统一分配完成！"
