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

# 根据device_links.yaml配置文件，生成server/pod粒度的topo.json和rootinfo.json
# 功能：
#   1. 解析yaml文件，获取device端口分配表和连接关系
#   2. 生成topo.json：描述server/pod内所有NPU设备的连接关系
#   3. 生成rootinfo.json：为每个device的portGroup分配EID和端口信息
#   4. EID分配规则：按照IP地址分配规则(AA.BB.CC.DD)分配IP，再按EID规则构造EID
#      - AA = 1   (superpod段，单server/pod场景S=1、N=1，恒为默认模式，superpod id=0时AA=1)
#      - BB = 1   (server段，单server/pod固定1)
#      - Host IP: 1.1.0.0 (固定为AA.BB.0.0，仅供host使用)
#      - Device port IPs: 1.1.0.1, 1.1.0.2, ... 依次递增 (从0.1开始，d2h端口也从此段分配)
#      - IP→EID: 在IPV4地址基础上，按EID分配规则填充高位字段
#        | 127~92 |   91~88   | 87~86 | 85~80 | 79~32 |    31~0    |
#        |  全0   | sp id[8:5]| die id| 端口号 |  全0  | IPV4地址   |
#        端口类型按文档"PG端口和物理端口说明"划分:
#          - PG端口: 仅当port_group对应ports列表中有多个物理端口时, 端口号固定为0x3F
#          - 普通物理端口: 单个"die/port"端口的portGroup或未在port_group中指定的端口,
#            端口号取物理端口号(0-62); d2h等端口名不带端口序号的端口, 端口号从pin_map中
#            value=3(h2d)对应的portId获取
#        port_group支持allocate_ip_by_physical_port字段(默认false): 为true时, 除portGroup
#        整体EID外, 还为ports列表中每个物理端口额外分配独立EID(输出到rootinfo/ranktable)
#   5. POD形态(board层)支持: 配置中 board_num > 1 时启用4层拓扑(board → pod → 超节点 → cluster)
#      - 配置文件中 device_group/device_ports_allocate_map/port_list/port_group 描述的是"一个board"
#        的卡结构: device id均为board内物理id(如0~7); device_num为POD总卡数,
#        每board卡数 = device_num / board_num
#      - links连接关系条目的board归属(仅POD形态):
#          fullmesh的connections/device_to_switch_links/matrix条目使用单个board_id字段
#            (互联/连交换机关系两端同board): 配置 → 仅对该board生效(links可整体按board逐一
#            描述全部板, 如1dpod64); 未配置 → 对所有board生效("单board描述"复制扩展, 如POD72 clos);
#          device_to_device_links条目使用src_board_id/dst_board_id字段分别声明两端device所属board
#            (须同时配置, 支持跨board直连, 如hf); 存在跨board直连(src≠dst)时layer0作用域必须为pod
#      - topo.json: **每POD一份**, 输出到server目录(输出目录根), 不再按board拆分;
#        其中peer_list/edge_list的local_id为POD全局id:
#          local_id = board_id * 每board卡数 + board内物理id
#      - rootinfo.json: **每board一份**, 输出到 board{b}/ 子目录(每board一个独立host节点),
#        device_id/local_id为board内物理id, 各board仅IP的CC段(board_id)与net_instance_id中的
#        board_{b}/idx_{k}序号不同; topo_file_path统一指向POD级topo.json
#      - IP分段: AA=superpod段, BB=server(POD)段, CC=board_id, DD=board内endPoint计数(<=255);
#        Host IP = AA.BB.{board_id}.0 (每board一个host); server形态CC恒为0, 与既有规则逐字节一致
#      - net_instance_id按作用域(scope)拼装:
#          board    → sp_i_srv_j_board_b_idx_k  (POD形态layer0板内直连时的默认值)
#          pod      → sp_i_srv_j_idx_k           (server形态layer0 / POD形态跨board交换网或跨board直连)
#          superpod → sp_i_idx_k                 (layer1默认)
#          cluster  → cluster                    (layer>=2默认)
#        可用net_instance_scope字段逐层覆盖; POD形态下layer0存在PEER2PEER时,
#        板内直连scope必须为board、跨board直连scope必须为pod;
#        配置了自定义字符串模板(见下)的层级绕过scope模板, 直接输出渲染后的字面量
#      - 可选字段: form_type("server"/"pod", 缺省按board_num推导)、device_num_per_board(校验用)
#
# 支持两种端口分配格式:
#   V1 (pin_map): {pin_map: {port_num_per_die, die_list: [{die_id, peer2peer, peer2net, h2d}], port_group}}
#   V2 (device_ports_allocate_map): [{die_id, pin_map: [0,2,2,1,...]}]
#
# 支持三种连接描述方法 (links列表中可共存):
#   fullmesh: link_mode + connections/device_groups
#   enum:     link_mode + device_to_device_links/device_to_switch_links
#   matrix:   link_mode + device_to_device_links(矩阵)/device_to_switch_links
#
# links条目的board归属(仅POD形态合法, 可写在link条目/connection子条目上, 后者优先):
#   fullmesh的connections/device_to_switch_links/matrix: 单个board_id字段(两端同board),
#     配置时仅对指定board生效; 未配置时对所有board生效(单board描述复制扩展)
#   device_to_device_links: src_board_id/dst_board_id字段分别声明两端所属board(须同时配置),
#     支持跨board直连(如hf); 未配置时回退board_id/复制扩展(两端同board)
#   device id字段: device_to_device_links使用src_device_id(_range)/dst_device_id(_range)
#     (board内物理id, 兼容旧命名src_local_id(_range)/dst_local_id(_range));
#     local_id由生成工具按 board_id*每board卡数 + board内物理id 换算得到
#
# net_instance_id / topo_instance_id 灵活配置 (双通道 + 优先级 + 冲突校验):
#   两个字段均支持 整数 或 字符串模板:
#   - 整数(既有语义): net_instance_id=k 即该层作用域字符串中的idx_k, 未配置默认0;
#     topo_instance_id未配置默认0
#   - 字符串模板: 占位符仅支持 {device_id}(board内物理id) 与 {board_id}(board序号,
#     server形态恒0), 未知占位符报错。net_instance_id模板逐device渲染为字面量
#     (如"cluster{device_id}"→cluster0/cluster1/...), 绕过scope模板, 且集群生成时
#     **全集群透传**(不按sp/srv重写, 各server同device渲染结果相同);
#     topo_instance_id模板渲染结果必须为整数(topo.json字段为int)
#   配置通道与优先级(对同一(layer, device)的同一字段):
#   - 通道A: port_group/port_list条目显式配置(port_list仅topo_instance_id);
#   - 通道B: links的device_to_switch_links条目显式配置(配置net_instance_id或
#     topo_instance_id时, 该条目或其父link条目必须显式声明net_layer, 否则报错——
#     d2s条目的die_id与net_layer无推断关系, 缺省net_layer=0会记错层);
#   - 仅A配置→用A; 仅B配置→用B; 均未配置→现状默认(scope模板/cluster/0);
#     A、B同时配置→逐device比较渲染值, 一致通过、不一致报错
#   约束:
#   - net_layer=0禁用net_instance_id字符串模板(消费侧按layer0的sp_*格式反解
#     superpod/server/board位置重建集群信息, 自定义字面量无法定位), 报错;
#   - 同一layer既在net_instance_scope中声明作用域又配置字符串模板: 报错(命名机制冲突);
#   - 字符串模板仅支持device_to_switch_links与port_group/port_list条目(PEER2NET/d2h
#     场景, 逐device归属明确); fullmesh的connections/device_to_device_links/matrix
#     条目出现字符串时报错(P2P边两端device不同, 逐device模板会导致两端实例id不一致);
#   - 同一net_layer内不同net_instance的实例不互通

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

SERVER_CONFIG_DIR="${BASE_DIR}/config/server_or_pod"
DEFAULT_OUTPUT_DIR="${BASE_DIR}/config/network/server_or_pod"

resolve_path() {
    local input="$1"
    local default_dir="$2"
    if [[ "$input" = /* ]]; then
        if [[ ! -f "$input" ]]; then
            error_exit "文件不存在: $input"
        fi
        realpath "$input"
    else
        local full_path="${default_dir}/${input}"
        if [[ ! -f "$full_path" ]]; then
            error_exit "文件不存在: $full_path (默认路径: ${default_dir})"
        fi
        realpath "$full_path"
    fi
}

show_usage() {
    cat << EOF
用法: $(basename "$0") [OPTIONS] <server_topo_yaml>

参数:
    server_topo_yaml   yaml配置文件路径或文件名 (支持V1/V2格式)
                       绝对路径: 直接使用
                       文件名/相对路径: 在 ${SERVER_CONFIG_DIR} 下查找

选项:
    -o, --output DIR         指定输出目录 (默认: ${DEFAULT_OUTPUT_DIR})
    -n, --name NAME          指定输出文件名前缀 (默认: 从yaml文件name字段获取)
    -b, --bare               输出文件不带前缀: topo.json 和 hccl_rootinfo.json
    -h, --help               显示帮助信息

说明:
    解析yaml配置文件，生成topo.json和rootinfo.json

    端口分配格式 (V1 - pin_map):
      pin_map:
        port_num_per_die: 9
        die_list: [{die_id, peer2peer, peer2net, h2d}]
        port_group: [{layer, port: [{die_id, port}]}]

    端口分配格式 (V2 - device_ports_allocate_map):
      device_ports_allocate_map: [{die_id, pin_map: [0,2,2,1,...]}]
      port_group: [{layer, ports: ["0/1", ...]}] 或 [{layer, port: [{die_id, port}]}]

    连接描述方法 (links列表中可共存):
      fullmesh: link_mode + connections/device_groups
      enum:     link_mode + device_to_device_links/device_to_switch_links
      matrix:   link_mode + device_to_device_links(矩阵)/device_to_switch_links

    拓扑形态 (board层):
      server形态(缺省, board_num未配置或=1): 3层拓扑(server → 超节点 → cluster),
        输出单份 topo.json / hccl_rootinfo.json 到输出目录, 卡物理id值域[0, device_num-1]
      POD形态(board_num > 1): 4层拓扑(board → pod → 超节点 → cluster),
        device_group/port_list/port_group中的device id为board内物理id, device_num为POD总卡数,
        每board卡数 = device_num / board_num;
        fullmesh的connections/device_to_switch_links/matrix条目可用单个board_id声明所属board
        (未配置 = 对所有board生效, 即"单board描述"复制扩展);
        device_to_device_links条目用src_board_id/dst_board_id声明两端所属board(支持跨board直连);
        topo.json为POD级单份, 输出到输出目录(local_id = board_id*每board卡数 + board内物理id),
        hccl_rootinfo.json每board一份输出到 board{b}/ 子目录(device_id/local_id为board内物理id),
        每board一个独立host节点(host IP = AA.BB.{board_id}.0)
      可选字段: form_type("server"/"pod")、device_num_per_board、net_instance_scope、
        links内board_id/device_to_device_links内src_board_id+dst_board_id

示例:
    $(basename "$0") ascend950_server_topo_normal.yaml
    $(basename "$0") /absolute/path/to/device_links.yaml
    $(basename "$0") -o /tmp/output ascend950_server_topo_normal.yaml -n ascend950_demo
    $(basename "$0") -o /tmp/output ascend950_server_topo_1dpod64.yaml -b   # POD形态: POD级topo.json + board0~board7的rootinfo

EOF
}

OUTPUT_DIR=""
OUTPUT_NAME=""
BARE_MODE=0

YAML_INPUT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -n|--name)
            OUTPUT_NAME="$2"
            shift 2
            ;;
        -b|--bare)
            BARE_MODE=1
            shift
            ;;
        *)
            YAML_INPUT="$1"
            shift
            ;;
    esac
done

if [[ -z "$YAML_INPUT" ]]; then
    error_exit "请指定yaml配置文件路径\n$(show_usage)"
fi

YAML_PATH="$(resolve_path "$YAML_INPUT" "$SERVER_CONFIG_DIR")"

if [[ -z "$OUTPUT_DIR" ]]; then
    YAML_BASENAME=$(basename "$YAML_PATH" .yaml)
    OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}/${YAML_BASENAME}"
fi

if [[ "$SCRIPT_LOG_LEVEL" -le 2 ]]; then
    echo "=========================================="
    echo "  Server/Pod拓扑描述文件生成工具"
    echo "=========================================="
    echo ""
    echo "输入文件: $YAML_PATH"
    echo "输出目录: $OUTPUT_DIR"
    echo ""
fi

mkdir -p "$OUTPUT_DIR"

export YAML_PATH
export OUTPUT_DIR
export OUTPUT_NAME
export BARE_MODE

python3 << 'PYTHON_SCRIPT'
import json
import os
import re
import sys
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

YAML_PATH = os.environ.get('YAML_PATH', '')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', '')
OUTPUT_NAME = os.environ.get('OUTPUT_NAME', '')
BARE_MODE = os.environ.get('BARE_MODE', '0') == '1'

try:
    import yaml
except ImportError:
    log_error("[错误] 需要PyYAML库，请安装: pip install pyyaml")
    sys.exit(1)

with open(YAML_PATH, 'r') as f:
    config = yaml.safe_load(f)

soc_version = config.get('soc_version', 'unknown')
device_num = config.get('device_num', 0)
name = config.get('name', 'topo')

# port_type: 可选字段, 端口类型, value只有"x2"和"x4"(不区分大小写), 不配置默认"x4",
# 配置为其他值报错。x2/x4端口只影响性能仿真时端口带宽表的选择(ascend960按port_type
# 选x2/x4端口带宽表; ascend950只有x4端口, 无x2端口带宽表, port_type只能为x4),
# 不影响拓扑和ranktable文件的生成。本脚本校验合法性; 字段值由集群生成脚本
# (generate_cluster_topo.sh)读取写入servers_info.json(hccl-vm工具专属文件), 再经
# C++集群解析器入库(Server.port_type), 不写入topo.json等hccl业务原生格式文件
port_type_raw = config.get('port_type', None)
port_type = 'x4'
if port_type_raw is not None:
    port_type_norm = str(port_type_raw).strip().lower()
    if port_type_norm not in ('x2', 'x4'):
        log_error(f"[错误] port_type字段值非法: {port_type_raw} (只支持x2/x4, 不区分大小写)，请检查配置文件")
        sys.exit(1)
    port_type = port_type_norm
# ascend950只有x4端口(无x2端口带宽表): port_type只能为x4, 配置x2报错
if port_type == 'x2' and str(soc_version).strip().lower().startswith('ascend950'):
    log_error(f"[错误] {soc_version}不支持port_type=x2 (ascend950只有x4端口, port_type只能为x4)，请检查配置文件")
    sys.exit(1)

if OUTPUT_NAME:
    out_name = OUTPUT_NAME
else:
    out_name = name

if BARE_MODE:
    topo_filename = 'topo.json'
    rootinfo_filename = 'hccl_rootinfo.json'
else:
    topo_filename = f'{out_name}_topo.json'
    rootinfo_filename = f'{out_name}_rootinfo.json'

if device_num == 0:
    log_error("[错误] device_num为0或未定义")
    sys.exit(1)

# ============================================================
# 0. 拓扑形态与board层解析 (POD形态: 4层拓扑 board → pod → 超节点 → cluster)
# ============================================================
# 设计说明(见 ai_skills/skills/generate-server-topo-json/references/
#   Ascend_server_topo_description_file_generation.md 1.3节):
#   - server形态(3层): 叶子节点为server, 卡物理id值域[0, device_num-1], 输出单份topo/rootinfo;
#   - POD形态(4层): device_group/port_list/port_group中的device id为board内物理id(每board卡数 =
#     device_num / board_num, 各board卡结构相同); fullmesh的connections/device_to_switch_links/matrix
#     条目可用单个board_id声明所属board —— 配置时仅对该board生效(links整体按board逐一描述全部板),
#     未配置时对所有board生效("单board描述"复制扩展); device_to_device_links条目用
#     src_board_id/dst_board_id声明两端所属board(支持跨board直连)。产物:
#       topo.json     —— POD级单份, 输出到输出目录根(server目录), local_id为POD全局id
#                        (= board_id * 每board卡数 + board内物理id);
#       rootinfo.json —— 每board一份, 输出到 board{b}/ 子目录(device_id/local_id为board内物理id,
#                        每board一个独立host节点), topo_file_path指向POD级topo.json。
#   - board_num缺省或为1 → server形态(产物与改造前逐字节一致, 零回归)。
form_type_raw = config.get('form_type', None)
form_type = ''
if form_type_raw is not None:
    form_type = str(form_type_raw).strip().lower()
    if form_type not in ('server', 'pod'):
        log_error(f"[错误] form_type字段值非法: {form_type_raw} (只支持server/pod, 不区分大小写)")
        sys.exit(1)

board_num = config.get('board_num', 1)
try:
    board_num = int(board_num)
except (TypeError, ValueError):
    log_error(f"[错误] board_num字段值非法: {board_num} (需为正整数)")
    sys.exit(1)
if board_num < 1:
    log_error(f"[错误] board_num必须 >= 1, 当前为{board_num}")
    sys.exit(1)

# form_type与board_num交叉校验: 避免"声明pod但漏配board_num"被静默降级为server形态
if form_type == 'pod' and board_num <= 1:
    log_error(f"[错误] form_type为pod但board_num={board_num}, POD形态必须配置board_num > 1")
    sys.exit(1)
if form_type == 'server' and board_num > 1:
    log_error(f"[错误] form_type为server但board_num={board_num} > 1, 二者矛盾, 请检查配置文件")
    sys.exit(1)
if form_type == '':
    form_type = 'pod' if board_num > 1 else 'server'
is_pod_form = (form_type == 'pod')

# 每board卡数: 缺省由 device_num / board_num 推导; 显式配置时必须与推导值一致
if device_num % board_num != 0:
    log_error(f"[错误] device_num({device_num}) 不能被 board_num({board_num}) 整除, "
              f"无法均匀划分board, 请检查配置文件")
    sys.exit(1)
node_device_num = device_num // board_num
per_board_raw = config.get('device_num_per_board', None)
if per_board_raw is not None:
    try:
        per_board_cfg = int(per_board_raw)
    except (TypeError, ValueError):
        log_error(f"[错误] device_num_per_board字段值非法: {per_board_raw} (需为正整数)")
        sys.exit(1)
    if per_board_cfg != node_device_num:
        log_error(f"[错误] device_num_per_board({per_board_cfg}) 与 device_num/board_num "
                  f"推导值({node_device_num})不一致, 请检查配置文件")
        sys.exit(1)
if node_device_num <= 0:
    log_error(f"[错误] 每board(叶子节点)卡数为0, device_num={device_num}, board_num={board_num}")
    sys.exit(1)
if is_pod_form and board_num > 255:
    log_error(f"[错误] board_num({board_num}) 超出IP地址CC段容量(255), 请调整组网规模")
    sys.exit(1)

# net_instance_scope: 逐层覆盖net_instance_id的作用域(scope)。
# scope取值与net_instance_id格式:
#   board    → sp_{i}_srv_{j}_board_{b}_idx_{k}
#   pod      → sp_{i}_srv_{j}_idx_{k}
#   superpod → sp_{i}_idx_{k}
#   cluster  → cluster
# 默认映射: server形态 {0: pod, 1: superpod, >=2: cluster} (与既有规则一致);
#           POD形态   layer0按该层PEER2PEER直连边的board归属推导:
#                     无直连→pod; 板内直连(每board独立成网)→board; 跨board直连→pod;
#                     layer1→superpod, >=2→cluster。
# 强制约束: POD形态下layer0存在PEER2PEER时 ——
#           板内直连scope必须为board(保证layer0的net_instance_id每board唯一, 否则不同board的
#           同号卡会在同一网络实例内互相冒充对端);
#           跨board直连scope必须为pod(直连域为单一POD级实例, 否则跨board边因两端niid不一致被过滤)。
VALID_SCOPES = ('board', 'pod', 'superpod', 'cluster')
net_instance_scope_cfg = {}
for entry in (config.get('net_instance_scope') or []):
    if not isinstance(entry, dict) or 'layer' not in entry or 'scope' not in entry:
        log_error(f"[错误] net_instance_scope条目非法: {entry} (需为{{layer: N, scope: xxx}})")
        sys.exit(1)
    scope_val = str(entry['scope']).strip().lower()
    if scope_val not in VALID_SCOPES:
        log_error(f"[错误] net_instance_scope的scope值非法: {entry['scope']} "
                  f"(只支持{'/'.join(VALID_SCOPES)})")
        sys.exit(1)
    if scope_val == 'board' and not is_pod_form:
        log_error(f"[错误] scope=board仅在POD形态(board_num > 1)下合法, 当前为{form_type}形态")
        sys.exit(1)
    net_instance_scope_cfg[int(entry['layer'])] = scope_val

# ============================================================
# 0b. net_instance_id/topo_instance_id 字符串模板机制 (灵活配置双通道, 见文件头说明)
# ============================================================
# 占位符仅支持 {device_id}(board内物理id) / {board_id}(board序号, server形态恒0);
# 自定义渲染结果为全集群透传的字面量(allocate_eid.sh/ranktable侧不按sp/srv重写),
# 故不支持 {sp_id}/{srv_id} 等随集群位置变化的占位符
TEMPLATE_PLACEHOLDERS = ('device_id', 'board_id')

def validate_instance_id_template(val, field_name, where, layer, is_niid):
    """校验net_instance_id/topo_instance_id字符串模板: 非空、占位符合法;
    net_instance_id模板禁用net_layer=0(消费侧依赖layer0的sp_*格式反解集群位置);
    同一layer已在net_instance_scope声明作用域时不得再用模板(命名机制冲突)"""
    if not str(val).strip():
        log_error(f"[错误] {where}: {field_name}字符串模板不能为空")
        sys.exit(1)
    for m in re.finditer(r'\{([^{}]*)\}', str(val)):
        if m.group(1) not in TEMPLATE_PLACEHOLDERS:
            log_error(f"[错误] {where}: {field_name}模板'{val}'含未知占位符'{{{m.group(1)}}}' "
                      f"(仅支持{'/'.join('{' + p + '}' for p in TEMPLATE_PLACEHOLDERS)})")
            sys.exit(1)
    if re.sub(r'\{(device_id|board_id)\}', '', str(val)).count('{') \
            or re.sub(r'\{(device_id|board_id)\}', '', str(val)).count('}'):
        log_error(f"[错误] {where}: {field_name}模板'{val}'含非法花括号")
        sys.exit(1)
    if is_niid:
        if int(layer) == 0:
            log_error(f"[错误] {where}: net_layer=0禁止配置net_instance_id字符串模板('{val}')——"
                      f"消费侧依赖layer0的net_instance_id(sp_*格式)反解superpod/server/board位置, "
                      f"自定义字面量无法定位。layer0请使用整数net_instance_id(idx_k)+作用域机制")
            sys.exit(1)
        if int(layer) in net_instance_scope_cfg:
            log_error(f"[错误] {where}: layer={layer}已在net_instance_scope中声明作用域"
                      f"({net_instance_scope_cfg[int(layer)]}), 又配置了net_instance_id字符串模板'{val}', "
                      f"两种命名机制冲突, 请移除net_instance_scope中该layer的声明或改用模板单一机制")
            sys.exit(1)

def render_instance_id_template(val, dev_id, board_id):
    """渲染字符串模板为字面量 (int原样返回)"""
    if not isinstance(val, str):
        return int(val)
    return val.replace('{device_id}', str(dev_id)).replace('{board_id}', str(board_id))

def render_tiid_value(val, dev_id, board_id, where):
    """渲染topo_instance_id值: int原样返回; 字符串模板渲染后必须为整数(topo.json字段为int)"""
    if not isinstance(val, str):
        return int(val)
    out = render_instance_id_template(val, dev_id, board_id)
    try:
        return int(out)
    except ValueError:
        log_error(f"[错误] {where}: topo_instance_id模板'{val}'对device{dev_id}渲染结果'{out}'不是整数, "
                  f"topo.json中topo_instance_id必须为整数")
        sys.exit(1)

def resolve_niid_for_device(a_raw, b_raw, layer, dev_id, board_id, where):
    """解析某(layer, device)的net_instance_id配置值, 双通道优先级与冲突校验:
      a_raw: 通道A(port_group条目显式配置, int/str/None)
      b_raw: 通道B(links的device_to_switch_links条目显式配置, int/str/None)
      仅A→用A; 仅B→用B; 均无→返回None(调用方回退默认); 均有→渲染值必须一致, 否则报错
    返回: 渲染后的值(int序号 或 自定义字面量字符串)"""
    if a_raw is not None and b_raw is not None:
        ra = render_instance_id_template(a_raw, dev_id, board_id)
        rb = render_instance_id_template(b_raw, dev_id, board_id)
        if ra != rb:
            log_error(f"[错误] {where}: layer{layer}的device{dev_id}同时配置了port_group"
                      f"(渲染为'{ra}')与links(渲染为'{rb}')两路net_instance_id且不一致, "
                      f"请检查配置文件(两路同时配置时必须一致)")
            sys.exit(1)
        return ra
    v = a_raw if a_raw is not None else b_raw
    return None if v is None else render_instance_id_template(v, dev_id, board_id)

log_info(f"[信息] 芯片类型: {soc_version}")
log_info(f"[信息] 拓扑形态: {form_type}"
         + (f" (4层: board → pod → 超节点 → cluster)" if is_pod_form else f" (3层: server → 超节点 → cluster)"))
if is_pod_form:
    log_info(f"[信息] POD总卡数: {device_num}, board数: {board_num}, 每board卡数: {node_device_num}")
else:
    log_info(f"[信息] Device数量: {device_num}")
log_info(f"[信息] 端口类型: {port_type}")
log_info(f"[信息] 输出文件名前缀: {out_name}")
log_info("")

# ============================================================
# 1. 解析端口分配表 (自动检测V1/V2格式, 支持group_id扩展)
# ============================================================
pin_map_raw = config.get('pin_map', {})
dpam_raw = config.get('device_ports_allocate_map', [])
dp_raw = config.get('device_ports', [])

# device_group解析: 构建 device_id → group_id 映射
device_group_raw = config.get('device_group', [])
device_to_group = {}  # device_id → group_id
group_to_devices = {}  # group_id → [device_list]
_default_group_id = 1

if device_group_raw:
    log_info(f"[信息] 检测到device_group配置, 共{len(device_group_raw)}个分组")
    _seen_group_ids = set()
    for dg in device_group_raw:
        gid = dg.get('group_id', _default_group_id)
        # 校验: group_id不能重复
        if gid in _seen_group_ids:
            log_error(f"[错误] device_group中存在重复的group_id={gid}，请检查配置文件")
            sys.exit(1)
        _seen_group_ids.add(gid)
        dev_list = dg.get('device_list', [])
        group_to_devices[gid] = dev_list
        for dev_id in dev_list:
            # 校验: device id必须落在叶子节点(server形态=server内, POD形态=board内)物理id值域内。
            # POD形态的配置以"单个board"为描述单位, 若误按POD全局id书写(如残留[8,9,...]),
            # 原实现会静默回退到默认分组并产出错误拓扑, 此处直接报错拦截
            if int(dev_id) < 0 or int(dev_id) >= node_device_num:
                log_error(f"[错误] device_group(group_id={gid})中的device id {dev_id} 超出值域"
                          f"[0, {node_device_num - 1}]"
                          + (f" (POD形态下配置描述的是单个board, 每board {node_device_num}卡, "
                             f"device id必须使用board内物理id, 不能使用POD全局id)" if is_pod_form else "")
                          + "，请检查配置文件")
                sys.exit(1)
            device_to_group[int(dev_id)] = gid
        log_debug(f"       group_id={gid}: devices={dev_list}")
else:
    # 没有device_group配置, 默认所有device为一组
    group_to_devices[_default_group_id] = list(range(node_device_num))
    for dev_id in range(node_device_num):
        device_to_group[dev_id] = _default_group_id

die_list = []  # 默认die_list (向后兼容)
port_num_per_die = 0
port_group_raw = None

# group_id → die_list 映射
group_die_lists = {}

if dpam_raw:
    log_info("[信息] 检测到V2端口分配格式 (device_ports_allocate_map)")
    _current_group_id = _default_group_id
    _temp_entries = []

    for d in dpam_raw:
        if isinstance(d, dict):
            # 检查是否是group_id元数据条目
            if 'group_id' in d and 'die_id' not in d and 'pin_map' not in d:
                # 保存之前的entries到对应的group
                if _temp_entries:
                    group_die_lists[_current_group_id] = _temp_entries
                _current_group_id = d['group_id']
                _temp_entries = []
                continue
            # 兼容旧的id字段
            if 'id' in d and 'die_id' not in d and 'pin_map' not in d:
                if _temp_entries:
                    group_die_lists[_current_group_id] = _temp_entries
                _current_group_id = d['id']
                _temp_entries = []
                continue
            # die条目
            die_id = d.get('die_id', 0)
            pin_map_vals = d.get('pin_map', d.get('port', []))
        elif isinstance(d, (list, tuple)):
            die_id = len(_temp_entries)
            pin_map_vals = list(d)
        else:
            continue

        p2p = [i for i, v in enumerate(pin_map_vals) if v == 1]
        p2n = [i for i, v in enumerate(pin_map_vals) if v == 2]
        h2d = [i for i, v in enumerate(pin_map_vals) if v == 3]

        _temp_entries.append({
            'die_id': die_id,
            'peer2peer': p2p,
            'peer2net': p2n,
            'h2d': h2d
        })

    # 保存最后一组
    if _temp_entries:
        group_die_lists[_current_group_id] = _temp_entries

    # 默认die_list (向后兼容)
    if _default_group_id in group_die_lists:
        die_list = group_die_lists[_default_group_id]
    elif group_die_lists:
        first_gid = min(group_die_lists.keys())
        die_list = group_die_lists[first_gid]

    # 打印信息
    for gid, entries in group_die_lists.items():
        dev_list_info = group_to_devices.get(gid, '全部(默认)')
        log_info(f"[信息] 端口分配表 group_id={gid}, 适用device: {dev_list_info}")
        for d in entries:
            log_debug(f"       die{d['die_id']}: PEER2PEER={d['peer2peer']}, PEER2NET={d['peer2net']}, H2D={d['h2d']}")

    if group_die_lists:
        first_entries = group_die_lists[min(group_die_lists.keys())]
        if first_entries:
            all_ports = set()
            for d in first_entries:
                all_ports.update(d['peer2peer'])
                all_ports.update(d['peer2net'])
                all_ports.update(d['h2d'])
            if all_ports:
                port_num_per_die = max(all_ports) + 1
elif dp_raw:
    log_info("[信息] 检测到V2端口分配格式 (device_ports)")
    for d in dp_raw:
        if isinstance(d, dict):
            die_id = d.get('die_id', 0)
            pin_map_vals = d.get('pin_map', d.get('port', []))
        elif isinstance(d, (list, tuple)):
            die_id = len(die_list)
            pin_map_vals = list(d)
        else:
            continue

        p2p = [i for i, v in enumerate(pin_map_vals) if v == 1]
        p2n = [i for i, v in enumerate(pin_map_vals) if v == 2]
        h2d = [i for i, v in enumerate(pin_map_vals) if v == 3]

        die_list.append({
            'die_id': die_id,
            'peer2peer': p2p,
            'peer2net': p2n,
            'h2d': h2d
        })

    if dp_raw:
        first = dp_raw[0]
        if isinstance(first, dict):
            port_num_per_die = len(first.get('pin_map', first.get('port', [])))
    # dp_raw格式: 将die_list保存到默认group
    if die_list:
        group_die_lists[_default_group_id] = die_list
elif pin_map_raw and isinstance(pin_map_raw, dict):
    log_info("[信息] 检测到V1端口分配格式 (pin_map)")
    port_num_per_die = pin_map_raw.get('port_num_per_die', 0)
    die_list_raw = pin_map_raw.get('die_list', [])
    port_group_raw = pin_map_raw.get('port_group', None)

    for d in die_list_raw:
        die_id = d.get('die_id', 0)
        p2p = d.get('peer2peer', [])
        p2n = d.get('peer2net', [])
        h2d = d.get('h2d', [])
        die_list.append({
            'die_id': die_id,
            'peer2peer': p2p,
            'peer2net': p2n if isinstance(p2n, list) and (not p2n or isinstance(p2n[0], int)) else [],
            'h2d': h2d
        })
    # V1格式: 将die_list保存到默认group
    if die_list:
        group_die_lists[_default_group_id] = die_list

has_h2d = any(len(d['h2d']) > 0 for d in die_list)

log_info(f"[信息] 端口分配表: port_num_per_die={port_num_per_die}, die数={len(die_list)}")
for d in die_list:
    log_debug(f"       die{d['die_id']}: PEER2PEER={d['peer2peer']}, PEER2NET={d['peer2net']}, H2D={d['h2d']}")
if len(group_die_lists) > 1:
    log_info(f"[信息] 多端口分配表已启用, 共{len(group_die_lists)}个不同的分组")

# ============================================================
# position/protocols默认值解析函数
# ============================================================
# 按文档规则:
#   - position: 默认"DEVICE"，可配置为"HOST"
#   - protocols: 若不配置，position=HOST时默认["ROCE"]，position=DEVICE时默认["UB_CTP","UB_MEM"]
def resolve_position_protocols(position, protocols):
    if position is None:
        position = 'DEVICE'
    if protocols is None:
        if position == 'HOST':
            protocols = ['ROCE']
        else:
            protocols = ['UB_CTP', 'UB_MEM']
    elif isinstance(protocols, str):
        protocols = [protocols]
    return position, protocols

# ============================================================
# plane_id构造函数
# ============================================================
def build_plane_id(configured_plane_id=0):
    # 按文档规则:
    #   - peer2peer端口: plane_id固定为"plane_0"
    #   - Peer2Net端口: plane_id为"plane_pg_" + str(configured_plane_id), 默认configured_plane_id=0 → "plane_pg_0"
    #     若用户在port_group/port_list中配置了plane_id字段, 则使用配置值; 否则默认为0
    if configured_plane_id is None:
        configured_plane_id = 0
    return f'plane_pg_{configured_plane_id}'

# ============================================================
# 2. 解析port_group (端口绑定关系, 支持group_id扩展)
# ============================================================
if port_group_raw is None:
    port_group_raw = config.get('port_group', [])

# group_id → {layer: entries} 映射
group_port_groups = {}
_current_group_id = _default_group_id
_temp_pg_entries = {}
_temp_d2h_layers = set()
# allocate_ip_by_physical_port: 组级开关(默认false), 表示除portGroup整体EID外,
# 是否还为portGroup内每个物理端口额外分配独立EID
_temp_allocate_ip_by_physical_port = False

for pg in port_group_raw:
    # 检查是否是group_id元数据条目
    if isinstance(pg, dict) and 'group_id' in pg and 'layer' not in pg:
        # 保存之前的entries到对应的group
        if _temp_pg_entries:
            group_port_groups[_current_group_id] = {'entries': _temp_pg_entries, 'd2h_layers': _temp_d2h_layers, 'allocate_ip_by_physical_port': _temp_allocate_ip_by_physical_port}
        _current_group_id = pg['group_id']
        _temp_pg_entries = {}
        _temp_d2h_layers = set()
        _temp_allocate_ip_by_physical_port = False
        continue

    # 检查是否是allocate_ip_by_physical_port元数据条目(组级开关, 作用于当前group)
    if isinstance(pg, dict) and 'allocate_ip_by_physical_port' in pg and 'layer' not in pg:
        _temp_allocate_ip_by_physical_port = bool(pg['allocate_ip_by_physical_port'])
        continue

    layer = pg.get('layer', 0)
    pg_topo_instance_id = pg.get('topo_instance_id', None)
    # net_instance_id默认None(区分"显式配置"与"未配置"): 未配置时回退links通道或默认0,
    # 显式配置(int序号/字符串模板)时参与A/B双通道优先级与冲突校验(见resolve_niid_for_device)
    pg_net_instance_id = pg.get('net_instance_id', None)
    if isinstance(pg_net_instance_id, str):
        validate_instance_id_template(pg_net_instance_id, 'net_instance_id',
                                      f'port_group(layer {layer})', layer, True)
    if isinstance(pg_topo_instance_id, str):
        validate_instance_id_template(pg_topo_instance_id, 'topo_instance_id',
                                      f'port_group(layer {layer})', layer, False)
    pg_plane_id = pg.get('plane_id', 0)
    pg_position = pg.get('position', None)
    pg_protocols = pg.get('protocols', None)
    if isinstance(pg_protocols, str):
        pg_protocols = [pg_protocols]
    # 条目级allocate_ip_by_physical_port配置(None表示未配置, 回退到组级开关)
    pg_allocate_ip_by_physical_port = pg.get('allocate_ip_by_physical_port', None)
    ports = pg.get('ports', pg.get('port', []))
    pg_entries = []

    if isinstance(ports, list) and ports and isinstance(ports[0], str):
        for port_str in ports:
            if port_str == 'd2h':
                _temp_d2h_layers.add(layer)
                continue
            parts = port_str.split('/')
            if len(parts) == 2:
                pg_entries.append({
                    'die_id': int(parts[0]),
                    'port': [int(parts[1])]
                })
    elif isinstance(ports, dict):
        pg_entries.append(ports)
    elif isinstance(ports, list):
        for p in ports:
            if isinstance(p, str):
                if p == 'd2h':
                    _temp_d2h_layers.add(layer)
                    continue
                parts = p.split('/')
                if len(parts) == 2:
                    pg_entries.append({
                        'die_id': int(parts[0]),
                        'port': [int(parts[1])]
                    })
            elif isinstance(p, dict):
                pg_entries.append(p)

    # 每个YAML port_group条目保持为独立分组(append而非extend)
    # 数据结构: {layer: [{'topo_instance_id': int, 'net_instance_id': int, 'plane_id': int, 'entries': [{die_id, port}, ...]}, ...]}
    # 每个条目对应一个portGroup，对应一个独立的EID
    if layer not in _temp_pg_entries:
        _temp_pg_entries[layer] = []
    _temp_pg_entries[layer].append({'topo_instance_id': pg_topo_instance_id, 'net_instance_id': pg_net_instance_id, 'plane_id': pg_plane_id, 'entries': pg_entries, 'position': pg_position, 'protocols': pg_protocols, 'allocate_ip_by_physical_port': pg_allocate_ip_by_physical_port})

# 保存最后一组
if _temp_pg_entries:
    group_port_groups[_current_group_id] = {'entries': _temp_pg_entries, 'd2h_layers': _temp_d2h_layers, 'allocate_ip_by_physical_port': _temp_allocate_ip_by_physical_port}

# 向后兼容: parsed_port_groups 和 d2h_layers
parsed_port_groups = {}
d2h_layers = set()
if _default_group_id in group_port_groups:
    parsed_port_groups = group_port_groups[_default_group_id]['entries']
    d2h_layers = group_port_groups[_default_group_id]['d2h_layers']
elif group_port_groups:
    first_gid = min(group_port_groups.keys())
    parsed_port_groups = group_port_groups[first_gid]['entries']
    d2h_layers = group_port_groups[first_gid]['d2h_layers']

if group_port_groups:
    log_info(f"[信息] port_group定义: 共{len(group_port_groups)}个分组")
    for gid, data in group_port_groups.items():
        dev_list_info = group_to_devices.get(gid, '全部(默认)')
        log_debug(f"       group_id={gid}, 适用device: {dev_list_info}, "
              f"allocate_ip_by_physical_port={data.get('allocate_ip_by_physical_port', False)}")
        for layer, groups in data['entries'].items():
            for gi, group in enumerate(groups):
                port_strs = []
                for entry in group['entries']:
                    pg_die_id = entry.get('die_id', 0)
                    pg_ports = entry.get('port', [])
                    if isinstance(pg_ports, int):
                        pg_ports = [pg_ports]
                    port_strs.extend([f"{pg_die_id}/{p}" for p in pg_ports])
                _pos, _protos = resolve_position_protocols(group.get('position'), group.get('protocols'))
                _tiid = group.get('topo_instance_id')
                _tiid_str = 'default(0)' if _tiid is None else _tiid
                log_debug(f"         layer {layer}, topo_instance_id={_tiid_str}, plane_id={group['plane_id']}, position={_pos}, protocols={_protos}, portGroup[{gi}]: {port_strs}")

# ============================================================
# 2b. 解析port_list (topo.json各netlayer的port列表, 支持group_id扩展)
# ============================================================
# port_list与port_group的区别:
#   - port_list: 用于topo.json中PEER2NET edge的local_a_ports
#   - port_group: 用于rootinfo.json中的EID分配
#   一般拓扑二者等价, 但复杂拓扑(如a5 1dpod64)可能不同
port_list_raw = config.get('port_list', [])

# group_id → {layer: entries} 映射
group_port_lists = {}
_current_group_id = _default_group_id
_temp_pl_entries = {}
_temp_d2h_layers_pl = set()

for pg in port_list_raw:
    # 检查是否是group_id元数据条目
    if isinstance(pg, dict) and 'group_id' in pg and 'layer' not in pg:
        # 保存之前的entries到对应的group
        if _temp_pl_entries:
            group_port_lists[_current_group_id] = {'entries': _temp_pl_entries, 'd2h_layers': _temp_d2h_layers_pl}
        _current_group_id = pg['group_id']
        _temp_pl_entries = {}
        _temp_d2h_layers_pl = set()
        continue

    layer = pg.get('layer', 0)
    pl_topo_instance_id = pg.get('topo_instance_id', None)
    if isinstance(pl_topo_instance_id, str):
        validate_instance_id_template(pl_topo_instance_id, 'topo_instance_id',
                                      f'port_list(layer {layer})', layer, False)
    pl_plane_id = pg.get('plane_id', 0)
    pl_position = pg.get('position', None)
    pl_protocols = pg.get('protocols', None)
    if isinstance(pl_protocols, str):
        pl_protocols = [pl_protocols]
    ports = pg.get('ports', pg.get('port', []))
    pg_entries = []

    if isinstance(ports, list) and ports and isinstance(ports[0], str):
        for port_str in ports:
            if port_str == 'd2h':
                _temp_d2h_layers_pl.add(layer)
                continue
            parts = port_str.split('/')
            if len(parts) == 2:
                pg_entries.append({
                    'die_id': int(parts[0]),
                    'port': [int(parts[1])]
                })
    elif isinstance(ports, dict):
        pg_entries.append(ports)
    elif isinstance(ports, list):
        for p in ports:
            if isinstance(p, str):
                if p == 'd2h':
                    _temp_d2h_layers_pl.add(layer)
                    continue
                parts = p.split('/')
                if len(parts) == 2:
                    pg_entries.append({
                        'die_id': int(parts[0]),
                        'port': [int(parts[1])]
                    })
            elif isinstance(p, dict):
                pg_entries.append(p)

    if layer not in _temp_pl_entries:
        _temp_pl_entries[layer] = []
    _temp_pl_entries[layer].append({'topo_instance_id': pl_topo_instance_id, 'plane_id': pl_plane_id, 'entries': pg_entries, 'position': pl_position, 'protocols': pl_protocols})

# 保存最后一组
if _temp_pl_entries:
    group_port_lists[_current_group_id] = {'entries': _temp_pl_entries, 'd2h_layers': _temp_d2h_layers_pl}

# 向后兼容: parsed_port_lists 和 d2h_layers_pl
parsed_port_lists = {}
d2h_layers_pl = set()
if _default_group_id in group_port_lists:
    parsed_port_lists = group_port_lists[_default_group_id]['entries']
    d2h_layers_pl = group_port_lists[_default_group_id]['d2h_layers']
elif group_port_lists:
    first_gid = min(group_port_lists.keys())
    parsed_port_lists = group_port_lists[first_gid]['entries']
    d2h_layers_pl = group_port_lists[first_gid]['d2h_layers']

if group_port_lists:
    log_info(f"[信息] port_list定义: 共{len(group_port_lists)}个分组")
    for gid, data in group_port_lists.items():
        dev_list_info = group_to_devices.get(gid, '全部(默认)')
        log_debug(f"       group_id={gid}, 适用device: {dev_list_info}")
        for layer, entries_list in data['entries'].items():
            for ei, entry_item in enumerate(entries_list):
                port_strs = []
                for entry in entry_item['entries']:
                    pg_die_id = entry.get('die_id', 0)
                    pg_ports = entry.get('port', [])
                    if isinstance(pg_ports, int):
                        pg_ports = [pg_ports]
                    port_strs.extend([f"{pg_die_id}/{p}" for p in pg_ports])
                _pos, _protos = resolve_position_protocols(entry_item.get('position'), entry_item.get('protocols'))
                _tiid = entry_item.get('topo_instance_id')
                _tiid_str = 'default(0)' if _tiid is None else _tiid
                log_debug(f"         layer {layer}, topo_instance_id={_tiid_str}, plane_id={entry_item.get('plane_id', 0)}, position={_pos}, protocols={_protos}: {port_strs}")

# ============================================================
# 2c. port_list与port_group的topo_instance_id一致性校验
# ============================================================
# 按文档规则: topo_instance_id字段与port_list字段是对应的 —— port_list条目合并的
# port_group, 其topo_instance_id必须一致(未配置默认0)。若port_list将topo_instance_id
# 不同的port_group合并为同一条目, 属于冲突配置, 生成时报错
def _entries_to_port_set(entries):
    port_set = set()
    for entry in entries:
        e_die_id = entry.get('die_id', 0)
        e_ports = entry.get('port', [])
        if isinstance(e_ports, int):
            e_ports = [e_ports]
        for p in e_ports:
            port_set.add(f"{e_die_id}/{p}")
    return port_set

_pl_pg_tiid_conflict = False
for gid, pl_data in group_port_lists.items():
    pg_data = group_port_groups.get(gid)
    if not pg_data:
        continue
    for layer, pl_entries in pl_data['entries'].items():
        # d2h层端口为'd2h'保留名, 不参与port合并校验
        if layer in pl_data['d2h_layers'] or layer in pg_data['d2h_layers']:
            continue
        pg_entries = pg_data['entries'].get(layer, [])
        for pl_idx, pl_item in enumerate(pl_entries):
            pl_ports = _entries_to_port_set(pl_item['entries'])
            if not pl_ports:
                continue
            pl_tiid = pl_item.get('topo_instance_id')
            pl_tiid_eff = pl_tiid if pl_tiid is not None else 0
            for pg_idx, pg_item in enumerate(pg_entries):
                pg_ports = _entries_to_port_set(pg_item['entries'])
                overlap = pl_ports & pg_ports
                if not overlap:
                    continue
                pg_tiid = pg_item.get('topo_instance_id')
                pg_tiid_eff = pg_tiid if pg_tiid is not None else 0
                if pl_tiid_eff != pg_tiid_eff:
                    log_error(f"[错误] port_list与port_group的topo_instance_id冲突: group_id={gid}, layer={layer}, "
                          f"port_list条目{pl_idx}(topo_instance_id={pl_tiid_eff})合并了topo_instance_id={pg_tiid_eff}的"
                          f"port_group条目{pg_idx}, 重叠端口: {sorted(overlap)}。"
                          f"topo_instance_id不同的port_group不能合并为同一个port_list条目, 请分别配置")
                    _pl_pg_tiid_conflict = True
if _pl_pg_tiid_conflict:
    sys.exit(1)
log_info("[信息] port_list与port_group的topo_instance_id一致性校验通过")

# ============================================================
# EID分配函数 (IP→EID转换, 按EID地址分配规则填充高位字段)
# ============================================================
URMA_EID_LEN = 16

# PG端口(portgroup端口)的固定端口号
EID_PG_PORT_NUM = 0x3F

def build_eid_string(ip, superpod_id=0, die_id=0, port_num=0):
    """按EID地址分配规则构造128bit EID字符串 (与allocate_eid.sh保持一致):
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
        端口号取物理端口号; d2h等端口名不带端口序号的端口, 端口号从pin_map中
        value=3(h2d)对应的portId获取(见resolve_named_port_die_port)
    输出为不带"0x"前缀的32位十六进制字符串, 与hcomm的AddrSetEID输出格式一致。
    示例:
        build_eid_string("1.1.0.1", 0, 1, 0)    -> "00000000004000000000000001010001"
        build_eid_string("1.1.0.8", 0, 0, 0x3F) -> "00000000003f00000000000001010008"
    """
    ip_bytes = socket.inet_pton(socket.AF_INET, ip)
    eid = bytearray(URMA_EID_LEN)
    eid[4] = (superpod_id >> 5) & 0xF                      # 91~88: super pod id[8:5]
    eid[5] = ((die_id & 0x3) << 6) | (port_num & 0x3F)     # 87~86: die id; 85~80: 端口号
    eid[12:16] = ip_bytes                                  # 31~0: IPV4地址
    hex_str = ''.join(f'{b:02x}' for b in eid)
    return hex_str

def derive_eid_die_port(net_layer, ports):
    """从rootinfo的rank_addr条目(net_layer + ports列表)推导EID高位字段的die id与端口号。
    按文档"PG端口和物理端口说明"规则(与net_layer无关):
      - PG端口: 仅当port_group对应ports列表中有多个物理端口时, 端口号固定0x3F,
        die id取首个"die/port"端口的die号
      - 普通物理端口: ports列表为单个"die/port"端口(单物理端口portGroup或未分组的P2P端口)时,
        端口号取物理端口号(0-62)
      - d2h等端口名不带端口序号的端口或空端口列表: 不属于PG端口, 按物理端口处理,
        die id与端口号从device端口分配表(pin_map)解析(见resolve_named_port_die_port),
        此处回退(0, 0)
    与allocate_eid.sh中的推导规则保持一致"""
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
    # d2h等不带端口序号的端口/空端口列表: 回退(0, 0), 由调用方按需用pin_map解析
    return 0, 0

def is_uboe_protocol_group(group):
    """判断portGroup是否为UBOE协议端口组(protocols含"UBOE")。
    按文档"EID地址分配规则"注意项: UBOE协议端口hcomm业务强制使用IPV4地址,
    EID高位(含superpod段/die id/端口号)默认全0, 即纯v4映射形态,
    不填充die/端口号等高位字段"""
    protocols = group.get('protocols') if isinstance(group, dict) else None
    if protocols is None:
        return False
    if isinstance(protocols, str):
        protocols = [protocols]
    return 'UBOE' in protocols

def resolve_named_port_die_port(port_name, dev_dl):
    """从device端口分配表(pin_map)解析端口名中不带端口序号的端口(如d2h)的die id与端口号。
    d2h端口: 端口号取pin_map中value=3(h2d)对应的portId, die id取该端口所属die;
    无法解析时回退(0, 0)"""
    if port_name == 'd2h':
        for d in dev_dl:
            for p in sorted(d.get('h2d', [])):
                return d.get('die_id', 0) & 0x3, p & 0x3F
    return 0, 0

def allocate_ip(aa, bb, counter):
    """server形态的IP分配: CC.DD共16bit作为endPoint计数器(容量65535)"""
    cc = (counter >> 8) & 0xFF
    dd = counter & 0xFF
    return f"{aa}.{bb}.{cc}.{dd}"

def alloc_endpoint_ip(counter):
    """按拓扑形态分配device endPoint的占位IP(集群生成时由allocate_eid.sh统一重分配):
      - server形态: AA.BB.(counter>>8).(counter&0xFF), CC.DD共16bit计数, 容量65535;
      - POD形态   : AA.BB.{board_id}.{counter}, CC段固定为board_id、DD段为board内计数,
                     故每board的endPoint上限为255(8卡机型实测约40~80个, 余量充足)。
       board_id取全局CUR_BOARD_ID(rootinfo按board逐份构建, 构建前设置为当前board序号)"""
    if is_pod_form:
        if counter > 0xFF:
            log_error(f"[错误] 单个board的endPoint数({counter})超出IP地址DD段容量(255), "
                      f"请减少每board卡数或合并portGroup")
            sys.exit(1)
        return f"{IP_AA}.{IP_BB}.{CUR_BOARD_ID}.{counter}"
    return allocate_ip(IP_AA, IP_BB, counter)

def build_net_instance_id(layer, niid, board_id=None):
    """按作用域(scope)构造net_instance_id (适用于所有拓扑类型):
    - scope = board    (POD形态layer0有板内直连): sp_i_srv_j_board_b_idx_k
      board_b代表该POD内第b个board。POD形态下各board的local_id值域相同(如均为0~7),
      只有layer0的net_instance_id每board唯一, 业务侧才不会把不同board的同号卡误判为同一实例内的对端
    - scope = pod      (server形态layer0 / POD形态跨board的框内交换网或跨board直连网): sp_i_srv_j_idx_k
      sp_i代表第i个super pod, srv_j代表第j个server/POD
    - scope = superpod (layer1): sp_i_idx_k
      跨server/POD时出框的net_instance_id相同, 即在这一层内互通
    - scope = cluster  (layer>=2): 固定为"cluster" (集群层, 如d2h层)
    其中sp/srv/board在单节点生成时均为占位值(0/board_id), 集群生成时由allocate_eid.sh
    按superpod/server/board目录路径重写; idx_k取links连接关系/port_group中net_instance_id的值,
    未配置时默认为0。scope由layer_scope解析得到(见5.1节), 可被net_instance_scope字段覆盖。
    niid为字符串时是自定义模板的渲染结果(字面量), 绕过scope模板直接返回(全集群透传)"""
    if isinstance(niid, str):
        return niid
    if board_id is None:
        board_id = CUR_BOARD_ID
    scope = layer_scope.get(layer, default_layer_scope(layer))
    # layer>=2 默认固定为"cluster"(集群层, 如d2h层); 仅当net_instance_scope显式声明了该层
    # 作用域时才按声明渲染(POD形态4层拓扑下layer2可能为POD间层, 需要保留层级归属信息)
    if layer >= 2 and layer not in net_instance_scope_cfg:
        return 'cluster'
    if scope == 'cluster':
        return 'cluster'
    if scope == 'board':
        return f'sp_0_srv_0_board_{board_id}_idx_{niid}'
    if scope == 'pod':
        return f'sp_0_srv_0_idx_{niid}'
    return f'sp_0_idx_{niid}'

# ============================================================
# 端口工具函数
# ============================================================

def parse_port_set(port_data):
    result = []
    if isinstance(port_data, (list, tuple)):
        for item in port_data:
            if isinstance(item, (list, tuple, set)):
                result.append(sorted([int(x) for x in item]))
            else:
                result.append([int(item)])
    elif isinstance(port_data, (int, str)):
        result.append([int(port_data)])
    return result

def get_die_list_for_device(device_id):
    """获取device对应的端口分配表die_list (支持group_id扩展)"""
    gid = device_to_group.get(device_id, _default_group_id)
    return group_die_lists.get(gid, die_list)

def get_peer2peer_portgroups(die_id, device_id=None):
    dl = get_die_list_for_device(device_id) if device_id is not None else die_list
    for d in dl:
        if d['die_id'] == die_id:
            return parse_port_set(d['peer2peer'])
    return []

def get_p2n_portgroup_for_layer(layer, device_id=None):
    """获取PEER2NET端口的port定义: 优先使用port_list(topo.json), 否则使用port_group
    返回的是该layer下所有portGroup的扁平列表 (用于topo.json)"""
    gid = device_to_group.get(device_id, _default_group_id) if device_id is not None else _default_group_id
    # 优先使用port_list (port_list不区分分组)
    if gid in group_port_lists and layer in group_port_lists[gid]['entries']:
        flat = []
        for item in group_port_lists[gid]['entries'][layer]:
            flat.extend(item['entries'])
        return flat
    if layer in parsed_port_lists:
        flat = []
        for item in parsed_port_lists[layer]:
            flat.extend(item['entries'])
        return flat
    # 回退到port_group: 扁平化所有分组
    groups = None
    if gid in group_port_groups and layer in group_port_groups[gid]['entries']:
        groups = group_port_groups[gid]['entries'][layer]
    elif layer in parsed_port_groups:
        groups = parsed_port_groups[layer]
    if groups:
        flat = []
        for group in groups:
            flat.extend(group['entries'])
        return flat
    return None

def get_p2n_portgroup_groups(layer, device_id=None):
    """获取EID分配用的port_group分组列表 (仅用于rootinfo.json)
    返回: [{'topo_instance_id': int, 'entries': [{die_id, port}, ...]}, ...]
    每个条目是一个独立的portGroup，对应一个独立的EID"""
    gid = device_to_group.get(device_id, _default_group_id) if device_id is not None else _default_group_id
    if gid in group_port_groups and layer in group_port_groups[gid]['entries']:
        return group_port_groups[gid]['entries'][layer]
    if layer in parsed_port_groups:
        return parsed_port_groups[layer]
    return None

def get_p2n_edges_info(layer, device_id=None, board_id=0):
    """获取topo.json PEER2NET edge信息: 按topo_instance_id分组
    返回: [{'topo_instance_id': int, 'tiid_explicit': bool, 'plane_id': int, 'port_strs': [str, ...], 'position': str, 'protocols': [str, ...]}, ...]
    优先使用port_list, 否则使用port_group, 最后回退到die_list的p2n端口
    topo_instance_id: 若port_list/port_group显式配置则使用配置值(int或字符串模板,
    模板按device渲染且结果必须为整数), 否则默认0 (按文档规则: 不配置则默认该字段为0);
    tiid_explicit标记来源条目是否显式配置了topo_instance_id(供A/B双通道冲突校验)"""
    gid = device_to_group.get(device_id, _default_group_id) if device_id is not None else _default_group_id
    _dev_for_tpl = device_id if device_id is not None else 0

    def resolve_tiid(tiid):
        if tiid is None:
            return 0, False
        if isinstance(tiid, str):
            return render_tiid_value(tiid, _dev_for_tpl, board_id,
                                     f'topo.json layer{layer} PEER2NET edge'), True
        return int(tiid), True

    # 优先使用port_list
    pl_groups = None
    if gid in group_port_lists and layer in group_port_lists[gid]['entries']:
        pl_groups = group_port_lists[gid]['entries'][layer]
    elif layer in parsed_port_lists:
        pl_groups = parsed_port_lists[layer]

    if pl_groups:
        result = []
        for item in pl_groups:
            port_strs = port_group_to_strs(item['entries'])
            if port_strs:
                _pos, _protos = resolve_position_protocols(item.get('position'), item.get('protocols'))
                _tiid, _tiid_explicit = resolve_tiid(item.get('topo_instance_id'))
                result.append({'topo_instance_id': _tiid, 'tiid_explicit': _tiid_explicit, 'plane_id': item.get('plane_id', 0), 'port_strs': port_strs,
                               'position': _pos, 'protocols': _protos})
        return result if result else None

    # 回退到port_group
    pg_groups = get_p2n_portgroup_groups(layer, device_id)
    if pg_groups:
        result = []
        for group in pg_groups:
            port_strs = port_group_to_strs(group['entries'])
            if port_strs:
                _pos, _protos = resolve_position_protocols(group.get('position'), group.get('protocols'))
                _tiid, _tiid_explicit = resolve_tiid(group.get('topo_instance_id'))
                result.append({'topo_instance_id': _tiid, 'tiid_explicit': _tiid_explicit, 'plane_id': group.get('plane_id', 0), 'port_strs': port_strs,
                               'position': _pos, 'protocols': _protos})
        return result if result else None

    # 最后回退到die_list的p2n端口
    dl = get_die_list_for_device(device_id) if device_id is not None else die_list
    port_strs = []
    for d in dl:
        port_strs.extend([f"{d['die_id']}/{p}" for p in get_peer2net_ports(d['die_id'], device_id)])
    if port_strs:
        return [{'topo_instance_id': 0, 'tiid_explicit': False, 'plane_id': 0, 'port_strs': port_strs, 'position': 'DEVICE', 'protocols': ['UB_CTP', 'UB_MEM']}]
    return None

def port_group_to_strs(pg_entries):
    port_strs = []
    for entry in pg_entries:
        pg_die_id = entry.get('die_id', 0)
        pg_ports = entry.get('port', [])
        if isinstance(pg_ports, int):
            pg_ports = [pg_ports]
        for p in pg_ports:
            port_strs.append(f"{pg_die_id}/{p}")
    return port_strs

def get_peer2net_ports(die_id, device_id=None):
    dl = get_die_list_for_device(device_id) if device_id is not None else die_list
    for d in dl:
        if d['die_id'] == die_id:
            p2n = d['peer2net']
            if isinstance(p2n, list):
                flat = []
                for item in p2n:
                    if isinstance(item, list):
                        flat.extend([int(x) for x in item])
                    else:
                        flat.append(int(item))
                return sorted(flat)
            return []
    return []

def get_h2d_ports(die_id, device_id=None):
    dl = get_die_list_for_device(device_id) if device_id is not None else die_list
    for d in dl:
        if d['die_id'] == die_id:
            h2d = d['h2d']
            if isinstance(h2d, list):
                flat = []
                for item in h2d:
                    if isinstance(item, list):
                        flat.extend([int(x) for x in item])
                    else:
                        flat.append(int(item))
                return sorted(flat)
            return []
    return []

# ============================================================
# 3. 解析links，生成edge列表 (支持fullmesh/enum/matrix)
# ============================================================
links = config.get('links', [])

def check_device_id_range(dev_ids, field_desc):
    """校验device id是否落在叶子节点物理id值域[0, node_device_num-1]内。
    server形态: 叶子节点为server, 值域[0, device_num-1];
    POD形态: 配置描述的是"单个board", 值域[0, 每board卡数-1]。
    POD形态下若误按POD全局id书写(如残留[8,9,...]), 原实现会静默回退到默认分组/漏连边,
    产出错误拓扑且不报错, 故此处统一拦截。"""
    for dev_id in dev_ids:
        if int(dev_id) < 0 or int(dev_id) >= node_device_num:
            log_error(f"[错误] links中{field_desc}的device id {dev_id} 超出值域"
                      f"[0, {node_device_num - 1}]"
                      + (f" (POD形态下配置描述的是单个board, 每board {node_device_num}卡, "
                         f"device id必须使用board内物理id, 不能使用POD全局id)" if is_pod_form else "")
                      + "，请检查配置文件")
            sys.exit(1)

def expand_range_field(data, field_name, list_field_name=None):
    if list_field_name is None:
        list_field_name = field_name
    raw = data.get(field_name, None)
    range_raw = data.get(f'{field_name}_range', None)
    result = []
    if raw is not None:
        if isinstance(raw, list):
            result = [int(x) for x in raw]
        elif isinstance(raw, (int, float)):
            result = [int(raw)]
    if range_raw is not None:
        if isinstance(range_raw, list) and len(range_raw) == 2:
            result.extend(list(range(int(range_raw[0]), int(range_raw[1]) + 1)))
        elif isinstance(range_raw, list):
            result.extend([int(x) for x in range_raw])
    check_device_id_range(result, field_name)
    return result

def expand_fullmesh(devices):
    edges = []
    for i in range(len(devices)):
        for j in range(i + 1, len(devices)):
            edges.append((devices[i], devices[j]))
    return edges

all_edges = []
switch_link_devices = []

# links连接关系条目的net_instance_id/topo_instance_id字段 (双通道之B, 见文件头说明):
# 显式配置时记录 (board_id, device_id, net_layer) → 配置值(int序号或字符串模板)映射,
# 供rootinfo/topo.json生成时与通道A(port_group/port_list)做优先级解析与冲突校验;
# 未配置的(bid, dev, layer)不出现在映射中, 由使用处回退默认
device_net_instance_id = {}
device_topo_instance_id = {}

def validate_board_id(raw, field_name, field_desc):
    """校验单个board归属字段值(board_id/src_board_id/dst_board_id), 返回合法board序号:
      - 仅POD形态(board_num > 1)可配置, server形态配置报错(无board层, 字段无意义);
      - 值域[0, board_num-1], 越界/非法值报错"""
    if not is_pod_form:
        log_error(f"[错误] links中{field_desc}条目配置了{field_name}({raw}), 但当前为server形态"
                  f"(board_num={board_num}), {field_name}字段仅POD形态(board_num > 1)下合法，请检查配置文件")
        sys.exit(1)
    try:
        bid = int(raw)
    except (TypeError, ValueError):
        log_error(f"[错误] links中{field_desc}条目的{field_name}字段值非法: {raw} "
                  f"(需为[0, {board_num - 1}]内的整数)，请检查配置文件")
        sys.exit(1)
    if bid < 0 or bid >= board_num:
        log_error(f"[错误] links中{field_desc}条目的{field_name}={bid} 超出值域[0, {board_num - 1}]，请检查配置文件")
        sys.exit(1)
    return bid

def parse_link_board_ids(entry, field_desc):
    """解析links条目的board_id字段(fullmesh connections/device_to_switch_links/matrix条目,
    两端device同board), 返回该条目生效的board列表 (仅POD形态可配置):
      - 未配置board_id: 对所有board生效 —— "单board描述"按board复制扩展(如POD72 clos);
      - 配置board_id  : 仅对该board生效 —— links整体可按board逐一描述全部板(如1dpod64)"""
    raw = entry.get('board_id', None)
    if raw is None:
        return list(range(board_num))
    return [validate_board_id(raw, 'board_id', field_desc)]

def parse_d2d_board_pairs(conn, entry_board_ids, field_desc):
    """解析device_to_device_links条目的board归属, 返回(src_board_id, dst_board_id)对列表:
      - 配置src_board_id/dst_board_id: 两端分别归属指定board, 支持跨board直连(如hf),
        两字段必须同时配置且不得与board_id混用;
      - 未配置: 回退board_id字段/条目级board_id(两端同board), 均未配置时对所有board生效
        ("单board描述"按board复制扩展, 两端同board)"""
    has_src = 'src_board_id' in conn
    has_dst = 'dst_board_id' in conn
    if has_src or has_dst:
        if not (has_src and has_dst):
            log_error(f"[错误] links中{field_desc}条目的src_board_id/dst_board_id必须同时配置"
                      f"(当前仅配置了{'src_board_id' if has_src else 'dst_board_id'})，请检查配置文件")
            sys.exit(1)
        if 'board_id' in conn:
            log_error(f"[错误] links中{field_desc}条目同时配置了board_id与src_board_id/dst_board_id, "
                      f"device_to_device_links条目请使用src_board_id/dst_board_id声明两端board归属，"
                      f"请检查配置文件")
            sys.exit(1)
        sbid = validate_board_id(conn['src_board_id'], 'src_board_id', field_desc)
        dbid = validate_board_id(conn['dst_board_id'], 'dst_board_id', field_desc)
        return [(sbid, dbid)]
    if 'board_id' in conn:
        entry_board_ids = parse_link_board_ids(conn, field_desc)
    return [(bid, bid) for bid in entry_board_ids]

def expand_endpoint_ids(conn, prefix):
    """展开device_to_device_links条目一端(prefix=src/dst)的device id列表:
    新命名{prefix}_device_id/{prefix}_device_id_range(board内物理id)优先,
    兼容旧命名{prefix}_local_id/{prefix}_local_id_range"""
    new_key = f'{prefix}_device_id'
    if new_key in conn or f'{new_key}_range' in conn:
        return expand_range_field(conn, new_key)
    return expand_range_field(conn, f'{prefix}_local_id')


def record_device_net_instance(devices, net_layer, niid, board_ids):
    # 仅记录显式配置net_instance_id的条目(未配置时默认0, 由使用处回退,
    # 不覆盖其他连接条目对该device的显式配置);
    # 值为int序号或字符串模板(模板合法性已在解析入口校验)
    if niid is None:
        return
    val = niid if isinstance(niid, str) else int(niid)
    for bid in board_ids:
        for dev in devices:
            device_net_instance_id[(bid, dev, net_layer)] = val

def record_device_topo_instance(devices, net_layer, tiid, board_ids):
    # 仅记录显式配置topo_instance_id的条目(未配置时默认0, 由使用处回退)
    if tiid is None:
        return
    val = tiid if isinstance(tiid, str) else int(tiid)
    for bid in board_ids:
        for dev in devices:
            device_topo_instance_id[(bid, dev, net_layer)] = val

def validate_d2s_instance_ids(sw_link, link_entry, layer, niid, tiid, where):
    """device_to_switch_links条目的net_instance_id/topo_instance_id配置校验:
    - 显式配置任一字段时, 该条目或其父link条目必须显式声明net_layer(d2s条目的die_id
      与net_layer无推断关系, 缺省会记到父条目net_layer(默认0)上, 层归属错误);
    - net_instance_id字符串模板: 占位符/layer0禁用/scope冲突校验(见validate_instance_id_template)"""
    if ('net_instance_id' in sw_link or 'topo_instance_id' in sw_link) \
            and 'net_layer' not in sw_link and 'net_layer' not in link_entry:
        log_error(f"[错误] {where}: device_to_switch_links条目配置net_instance_id/"
                  f"topo_instance_id时必须显式声明net_layer(条目级或父link条目级), "
                  f"否则无法确定实例归属层(die_id与net_layer无对应关系): {sw_link}")
        sys.exit(1)
    if isinstance(niid, str):
        validate_instance_id_template(niid, 'net_instance_id', where, layer, True)
    if isinstance(tiid, str):
        validate_instance_id_template(tiid, 'topo_instance_id', where, layer, False)

def validate_p2p_instance_ids(val, field_name, where):
    """fullmesh的connections/device_to_device_links/matrix条目的实例id字段校验:
    仅支持整数——P2P边两端device不同, 逐device渲染的字符串模板会导致两端实例id不一致,
    边被消费侧按net_instance_id过滤而失效; 字符串模板仅支持PEER2NET/d2h场景
    (device_to_switch_links与port_group/port_list条目)"""
    if isinstance(val, str):
        log_error(f"[错误] {where}: {field_name}不支持字符串模板('{val}')——"
                  f"字符串模板仅支持device_to_switch_links与port_group/port_list条目, "
                  f"PEER2PEER条目请使用整数")
        sys.exit(1)

for link_entry in links:
    link_mode = link_entry.get('link_mode', 'enum')
    link_type = link_entry.get('link_type', '1DMESH')
    die_id = link_entry.get('die_id', None)
    protocol = link_entry.get('protocol', ['UB_CTP', 'UB_MEM'])
    if isinstance(protocol, str):
        protocol = [protocol]
    net_layer = link_entry.get('net_layer', 0)
    # link条目级board_id(可被connection/子条目级board_id覆盖): 未配置时对所有board生效
    entry_board_ids = parse_link_board_ids(link_entry, link_mode)

    if link_mode == 'fullmesh':
        connections = link_entry.get('connections', [])
        device_groups = link_entry.get('device_groups', None)

        if connections:
            for conn in connections:
                conn_die_id = conn.get('die_id', die_id if die_id is not None else 0)
                conn_topo_instance_id = conn.get('topo_instance_id', 0)
                conn_net_instance_id = conn.get('net_instance_id')
                validate_p2p_instance_ids(conn_topo_instance_id, 'topo_instance_id', 'fullmesh connection')
                validate_p2p_instance_ids(conn_net_instance_id, 'net_instance_id', 'fullmesh connection')
                conn_board_ids = parse_link_board_ids(conn, 'fullmesh connection') \
                    if 'board_id' in conn else entry_board_ids
                devices = expand_range_field(conn, 'devices')
                # 兼容device_id字段
                if not devices:
                    devices = expand_range_field(conn, 'device_id')
                conn_edges = expand_fullmesh(devices)
                for bid in conn_board_ids:
                    # net_instance_id: 该connection内device所属net_instance序号
                    record_device_net_instance(devices, net_layer, conn_net_instance_id, [bid])
                    for (a, b) in conn_edges:
                        all_edges.append({
                            'net_layer': net_layer,
                            'link_type': 'PEER2PEER',
                            'topo_type': link_type,
                            'topo_instance_id': conn_topo_instance_id,
                            'src_board_id': bid,
                            'dst_board_id': bid,
                            'local_a': a,
                            'local_b': b,
                            'src_die_id': conn_die_id,
                            'dst_die_id': conn_die_id,
                            'protocols': protocol,
                            'position': 'DEVICE'
                        })
        elif device_groups:
            for group in device_groups:
                group_edges = expand_fullmesh(group)
                for bid in entry_board_ids:
                    for (a, b) in group_edges:
                        all_edges.append({
                            'net_layer': net_layer,
                            'link_type': 'PEER2PEER',
                            'topo_type': link_type,
                            'topo_instance_id': 0,
                            'src_board_id': bid,
                            'dst_board_id': bid,
                            'local_a': a,
                            'local_b': b,
                            'src_die_id': die_id if die_id is not None else 0,
                            'dst_die_id': die_id if die_id is not None else 0,
                            'protocols': protocol,
                            'position': 'DEVICE'
                        })

    elif link_mode == 'enum':
        d2d_links = link_entry.get('device_to_device_links', [])
        for conn in d2d_links:
            s_die = conn.get('src_die_id', die_id if die_id is not None else 0)
            d_die = conn.get('dst_die_id', die_id if die_id is not None else 0)
            conn_topo_instance_id = conn.get('topo_instance_id', 0)
            conn_net_instance_id = conn.get('net_instance_id')
            validate_p2p_instance_ids(conn_topo_instance_id, 'topo_instance_id', 'enum device_to_device_links')
            validate_p2p_instance_ids(conn_net_instance_id, 'net_instance_id', 'enum device_to_device_links')
            # board归属: src_board_id/dst_board_id声明两端所属board(支持跨board直连);
            # 未配置时回退board_id/条目级board_id/全部board(两端同board)
            conn_board_pairs = parse_d2d_board_pairs(conn, entry_board_ids,
                                                     'enum device_to_device_links')
            # device id为board内物理id: 新命名src_device_id(_range)/dst_device_id(_range),
            # 兼容旧命名src_local_id(_range)/dst_local_id(_range)
            s_devs = expand_endpoint_ids(conn, 'src')
            d_devs = expand_endpoint_ids(conn, 'dst')

            for (s_bid, d_bid) in conn_board_pairs:
                # net_instance_id: 该连接关系两端device所属net_instance序号(按各自board记录)
                record_device_net_instance(s_devs, net_layer, conn_net_instance_id, [s_bid])
                record_device_net_instance(d_devs, net_layer, conn_net_instance_id, [d_bid])
                for s_dev in s_devs:
                    for d_dev in d_devs:
                        all_edges.append({
                            'net_layer': net_layer,
                            'link_type': 'PEER2PEER',
                            'topo_type': link_type,
                            'topo_instance_id': conn_topo_instance_id,
                            'src_board_id': s_bid,
                            'dst_board_id': d_bid,
                            'local_a': s_dev,
                            'local_b': d_dev,
                            'src_die_id': s_die,
                            'dst_die_id': d_die,
                            'protocols': protocol,
                            'position': 'DEVICE'
                        })

        d2s_links = link_entry.get('device_to_switch_links', [])
        for sw_link in d2s_links:
            sw_die_id = sw_link.get('die_id', die_id if die_id is not None else 0)
            sw_net_layer = sw_link.get('net_layer', net_layer)
            sw_net_instance_id = sw_link.get('net_instance_id')
            sw_topo_instance_id = sw_link.get('topo_instance_id')
            validate_d2s_instance_ids(sw_link, link_entry, sw_net_layer,
                                      sw_net_instance_id, sw_topo_instance_id,
                                      'enum device_to_switch_links')
            sw_board_ids = parse_link_board_ids(sw_link, 'enum device_to_switch_links') \
                if 'board_id' in sw_link else entry_board_ids
            sw_devices = expand_range_field(sw_link, 'devices')
            for bid in sw_board_ids:
                # net_instance_id/topo_instance_id: 连交换机的device在该net_layer的实例配置(通道B)
                record_device_net_instance(sw_devices, sw_net_layer, sw_net_instance_id, [bid])
                record_device_topo_instance(sw_devices, sw_net_layer, sw_topo_instance_id, [bid])
                for dev in sw_devices:
                    switch_link_devices.append({
                        'device_id': dev,
                        'die_id': sw_die_id,
                        'board_id': bid
                    })

    elif link_mode == 'matrix':
        matrix_data = link_entry.get('device_to_device_links', [])
        if matrix_data:
            matrix_die_id = 0
            matrix_topo_instance_id = 0
            matrix_board_ids = entry_board_ids
            matrix_rows = []

            for item in matrix_data:
                if isinstance(item, dict):
                    matrix_die_id = item.get('die_id', 0)
                    matrix_topo_instance_id = item.get('topo_instance_id', 0)
                    validate_p2p_instance_ids(matrix_topo_instance_id, 'topo_instance_id',
                                              'matrix device_to_device_links')
                    if 'board_id' in item:
                        matrix_board_ids = parse_link_board_ids(item, 'matrix device_to_device_links')
                elif isinstance(item, list):
                    matrix_rows.append(item)

            # 校验: 矩阵行列数即叶子节点内的device数(server形态=server内, POD形态=board内),
            # 行列索引直接作为device物理id使用, 故必须与每叶子节点卡数一致
            if matrix_rows:
                if len(matrix_rows) != node_device_num:
                    log_error(f"[错误] matrix法的矩阵行数({len(matrix_rows)})与"
                              f"{'每board卡数' if is_pod_form else 'device_num'}({node_device_num})不一致，"
                              f"矩阵行列索引即device物理id，请检查配置文件")
                    sys.exit(1)
                for row_idx, row in enumerate(matrix_rows):
                    if len(row) != node_device_num:
                        log_error(f"[错误] matrix法第{row_idx}行的列数({len(row)})与"
                                  f"每叶子节点卡数({node_device_num})不一致，请检查配置文件")
                        sys.exit(1)

            for bid in matrix_board_ids:
                for i, row in enumerate(matrix_rows):
                    for j in range(i + 1, len(row)):
                        if j < len(row) and row[j] == 1:
                            all_edges.append({
                                'net_layer': net_layer,
                                'link_type': 'PEER2PEER',
                                'topo_type': link_type,
                                'topo_instance_id': matrix_topo_instance_id,
                                'src_board_id': bid,
                                'dst_board_id': bid,
                                'local_a': i,
                                'local_b': j,
                                'src_die_id': matrix_die_id,
                                'dst_die_id': matrix_die_id,
                                'protocols': protocol,
                                'position': 'DEVICE'
                            })

        d2s_links = link_entry.get('device_to_switch_links', [])
        for sw_link in d2s_links:
            sw_die_id = sw_link.get('die_id', die_id if die_id is not None else 0)
            sw_net_layer = sw_link.get('net_layer', net_layer)
            sw_net_instance_id = sw_link.get('net_instance_id')
            sw_topo_instance_id = sw_link.get('topo_instance_id')
            validate_d2s_instance_ids(sw_link, link_entry, sw_net_layer,
                                      sw_net_instance_id, sw_topo_instance_id,
                                      'matrix device_to_switch_links')
            sw_board_ids = parse_link_board_ids(sw_link, 'matrix device_to_switch_links') \
                if 'board_id' in sw_link else entry_board_ids
            sw_devices = expand_range_field(sw_link, 'devices')
            for bid in sw_board_ids:
                # net_instance_id/topo_instance_id: 连交换机的device在该net_layer的实例配置(通道B)
                record_device_net_instance(sw_devices, sw_net_layer, sw_net_instance_id, [bid])
                record_device_topo_instance(sw_devices, sw_net_layer, sw_topo_instance_id, [bid])
                for dev in sw_devices:
                    switch_link_devices.append({
                        'device_id': dev,
                        'die_id': sw_die_id,
                        'board_id': bid
                    })
    else:
        log_warn(f"  [警告] 不支持的link_mode: {link_mode}，跳过")

log_info(f"[信息] 解析连接关系: 生成 {len(all_edges)} 条PEER2PEER edge")

# ============================================================
# 4. 为每条PEER2PEER edge分配端口
# ============================================================
# 端口占用计数按POD全局device id(server形态即卡id)记录;
# 端口分配表/分组查询按board内物理id(各board卡结构相同)
device_port_usage = {}
for dev_global in range(device_num):
    device_port_usage[dev_global] = {}

def get_next_p2p_port(dev_global, die_id):
    key = (dev_global, die_id)
    if key not in device_port_usage[dev_global]:
        device_port_usage[dev_global][key] = 0
    portgroups = get_peer2peer_portgroups(die_id, dev_global % node_device_num)
    if not portgroups:
        return None
    idx = device_port_usage[dev_global][key]
    pg_idx = idx % len(portgroups)
    port_in_pg = idx // len(portgroups)
    pg = portgroups[pg_idx]
    if port_in_pg < len(pg):
        port = pg[port_in_pg]
    else:
        port = pg[-1]
    device_port_usage[dev_global][key] = idx + 1
    return f"{die_id}/{port}"

device_p2p_needed = [0] * device_num
for edge in all_edges:
    device_p2p_needed[edge['src_board_id'] * node_device_num + edge['local_a']] += 1
    device_p2p_needed[edge['dst_board_id'] * node_device_num + edge['local_b']] += 1

for dev_global in range(device_num):
    dev_dl = get_die_list_for_device(dev_global % node_device_num)
    total_p2p = sum(len(d['peer2peer']) for d in dev_dl)
    if device_p2p_needed[dev_global] > total_p2p:
        log_warn(f"[警告] device {dev_global} 需要{device_p2p_needed[dev_global]}个P2P端口，但只有{total_p2p}个可用")

for edge in all_edges:
    a_local = edge['local_a']
    b_local = edge['local_b']
    # POD全局device id: 两端分别按各自所属board换算(跨board直连时src/dst board不同)
    a = edge['src_board_id'] * node_device_num + a_local
    b = edge['dst_board_id'] * node_device_num + b_local
    src_die = edge.pop('src_die_id', None)
    dst_die = edge.pop('dst_die_id', None)
    dev_dl_a = get_die_list_for_device(a_local)
    dev_dl_b = get_die_list_for_device(b_local)

    if src_die is None:
        src_die = next((d['die_id'] for d in dev_dl_a if d['peer2peer']), dev_dl_a[0]['die_id'] if dev_dl_a else 0)
    if dst_die is None:
        dst_die = next((d['die_id'] for d in dev_dl_b if d['peer2peer']), dev_dl_b[0]['die_id'] if dev_dl_b else 0)

    port_a = get_next_p2p_port(a, src_die)
    port_b = get_next_p2p_port(b, dst_die)

    edge['local_a_ports'] = [port_a] if port_a else []
    edge['local_b_ports'] = [port_b] if port_b else []

log_info(f"[信息] 端口分配完成")

# ============================================================
# 5. 生成topo.json (每server/POD一份; POD形态local_id为POD全局id)
# ============================================================
# peer_list覆盖全部device: server形态为server内id[0, device_num-1];
# POD形态为POD全局id[0, device_num-1], 其中 local_id = board_id * 每board卡数 + board内物理id
peer_list = [{'local_id': i} for i in range(device_num)]

topo_edges = []

for edge in all_edges:
    # local_id为POD全局id, 两端分别按各自所属board换算(跨board直连时src/dst board不同)
    edge_dict = {
        'link_type': edge['link_type'],
        'local_a': edge['src_board_id'] * node_device_num + edge['local_a'],
        'local_a_ports': edge['local_a_ports'],
        'local_b': edge['dst_board_id'] * node_device_num + edge['local_b'],
        'local_b_ports': edge['local_b_ports'],
        'net_layer': edge['net_layer'],
        'position': edge['position'],
        'protocols': edge['protocols'],
        'topo_type': edge['topo_type'],
        'topo_instance_id': edge['topo_instance_id']
    }
    topo_edges.append(edge_dict)

# 收集所有已知端口 (用于ungrouped端口检测)
# port_group新结构: {layer: [{'topo_instance_id': int, 'entries': [{die_id, port}, ...]}, ...]}
all_pg_port_strs = set()
for gid, data in group_port_groups.items():
    for layer, groups in data['entries'].items():
        if layer not in data['d2h_layers']:
            for group in groups:
                for entry in group['entries']:
                    pg_die_id = entry.get('die_id', 0)
                    pg_ports = entry.get('port', [])
                    if isinstance(pg_ports, int):
                        pg_ports = [pg_ports]
                    for p in pg_ports:
                        all_pg_port_strs.add(f"{pg_die_id}/{p}")

# 将port_list中定义的端口也加入"已知端口"集合
for gid, data in group_port_lists.items():
    for layer, entries_list in data['entries'].items():
        if layer not in data['d2h_layers']:
            for entry_item in entries_list:
                for entry in entry_item['entries']:
                    pg_die_id = entry.get('die_id', 0)
                    pg_ports = entry.get('port', [])
                    if isinstance(pg_ports, int):
                        pg_ports = [pg_ports]
                    for p in pg_ports:
                        all_pg_port_strs.add(f"{pg_die_id}/{p}")

ungrouped_p2n_ports = []
for d in die_list:
    for p in get_peer2net_ports(d['die_id']):
        ps = f"{d['die_id']}/{p}"
        if ps not in all_pg_port_strs:
            ungrouped_p2n_ports.append(ps)
ungrouped_p2n_ports.sort()

# p2n_layers: 合并所有group的port_group和port_list中的layer (排除d2h)
_all_p2n_layers = set()
_all_d2h_layers = set()
for gid, data in group_port_groups.items():
    _all_p2n_layers.update(data['entries'].keys())
    _all_d2h_layers.update(data['d2h_layers'])
for gid, data in group_port_lists.items():
    _all_p2n_layers.update(data['entries'].keys())
    _all_d2h_layers.update(data['d2h_layers'])
p2n_layers = sorted([l for l in _all_p2n_layers if l not in _all_d2h_layers])
first_p2n_layer = p2n_layers[0] if p2n_layers else None

# 校验: links条目记录的net_instance_id/topo_instance_id所属net_layer必须在拓扑中真实存在
# (layer0为PEER2PEER直连层, 或由port_group/port_list定义的PEER2NET/d2h层),
# 否则该配置不会被任何产物消费(静默失效), 视为配置错误
_valid_instance_layers = set(_all_p2n_layers) | set(_all_d2h_layers) | {0}
for _inst_layer in sorted({l for _, _, l in device_net_instance_id} | {l for _, _, l in device_topo_instance_id}):
    if _inst_layer not in _valid_instance_layers:
        log_error(f"[错误] links条目配置的net_instance_id/topo_instance_id归属net_layer={_inst_layer}, "
                  f"但该layer在port_group/port_list中不存在(实际存在: {sorted(_valid_instance_layers)}), "
                  f"配置不会生效, 请检查device_to_switch_links等条目的net_layer字段")
        sys.exit(1)

# P2N/d2h edge按POD全局device id逐卡生成(每卡一份, 与P2P edge的local_id同值域);
# 端口分配表/port_group等查询仍按board内物理id(各board卡结构相同)
for dev_global in range(device_num):
    dev_id = dev_global % node_device_num
    dev_board_id = dev_global // node_device_num
    dev_dl = get_die_list_for_device(dev_id)
    dev_gid = device_to_group.get(dev_id, _default_group_id)
    dev_pl = group_port_lists.get(dev_gid, {}).get('entries', parsed_port_lists)
    for layer in p2n_layers:
        # topo.json的PEER2NET端口: 按topo_instance_id分组生成独立edge
        edges_info = get_p2n_edges_info(layer, dev_id, dev_board_id)
        if edges_info:
            # 通道B(links的device_to_switch_links)显式配置的per-device topo_instance_id:
            # 仅B→用B; A(port_group/port_list显式)与B同时配置→渲染值必须一致, 否则报错
            b_tiid_raw = device_topo_instance_id.get((dev_board_id, dev_id, layer))
            b_tiid_val = None
            if b_tiid_raw is not None:
                b_tiid_val = render_tiid_value(b_tiid_raw, dev_id, dev_board_id,
                                               f'topo.json layer{layer} device{dev_id}')
            for ei in edges_info:
                port_strs = list(ei['port_strs'])
                tiid = ei['topo_instance_id']
                if b_tiid_val is not None:
                    if ei['tiid_explicit'] and tiid != b_tiid_val:
                        log_error(f"[错误] layer{layer}的device{dev_id}同时配置了port_group/port_list"
                                  f"(topo_instance_id={tiid})与links(topo_instance_id渲染为{b_tiid_val})"
                                  f"两路topo_instance_id且不一致, 请检查配置文件(两路同时配置时必须一致)")
                        sys.exit(1)
                    tiid = b_tiid_val

                # 仅当该layer没有port_list定义时，才追加ungrouped端口
                if layer == first_p2n_layer and ungrouped_p2n_ports and layer not in dev_pl and tiid == 0:
                    combined = list(port_strs) + ungrouped_p2n_ports
                    combined.sort(key=lambda x: (int(x.split('/')[0]), int(x.split('/')[1])) if '/' in x else (999, 0))
                    port_strs = combined

                if port_strs:
                    edge_position = ei.get('position', 'DEVICE')
                    edge_protocols = ei.get('protocols', ['UB_CTP', 'UB_MEM'])
                    p2n_edge = {
                        'link_type': 'PEER2NET',
                        'local_a': dev_global,
                        'local_a_ports': port_strs,
                        'net_layer': layer,
                        'position': edge_position,
                        'protocols': edge_protocols,
                        'topo_type': 'CLOS',
                        'topo_instance_id': tiid
                    }
                    topo_edges.append(p2n_edge)

    # per-device d2h edge生成 (支持group_id扩展)
    # 合并 port_list 和 port_group 的 d2h_layers，遍历所有 d2h layer 生成独立 edge
    dev_d2h_layers_pl = group_port_lists.get(dev_gid, {}).get('d2h_layers', set())
    dev_d2h_layers_pg = group_port_groups.get(dev_gid, {}).get('d2h_layers', d2h_layers)
    dev_all_d2h_layers = set(dev_d2h_layers_pl) | set(dev_d2h_layers_pg)
    # 兼容: pin_map有h2d但无d2h配置时，自动分配一个layer
    dev_has_h2d = any(len(d['h2d']) > 0 for d in dev_dl)
    if dev_has_h2d and not dev_all_d2h_layers:
        dev_all_d2h_layers = {max(p2n_layers) + 1 if p2n_layers else 1}

    dev_pl_entries = group_port_lists.get(dev_gid, {}).get('entries', parsed_port_lists)
    dev_pg_entries = group_port_groups.get(dev_gid, {}).get('entries', parsed_port_groups)

    for d2h_net_layer in sorted(dev_all_d2h_layers):
        d2h_position, d2h_protocols = resolve_position_protocols('HOST', None)
        d2h_tiid = None
        # 优先从port_list查找topo_instance_id/position/protocols
        if d2h_net_layer in dev_pl_entries:
            for item in dev_pl_entries[d2h_net_layer]:
                d2h_position, d2h_protocols = resolve_position_protocols(item.get('position') or 'HOST', item.get('protocols'))
                d2h_tiid = item.get('topo_instance_id')
                break
        elif d2h_net_layer in dev_pg_entries:
            for group in dev_pg_entries[d2h_net_layer]:
                d2h_position, d2h_protocols = resolve_position_protocols(group.get('position') or 'HOST', group.get('protocols'))
                d2h_tiid = group.get('topo_instance_id')
                break
        # 若未显式配置topo_instance_id, 默认为0 (按文档规则: 不配置则默认该字段为0);
        # 字符串模板按device渲染(结果必须为整数)
        if d2h_tiid is None:
            d2h_tiid = 0
        elif isinstance(d2h_tiid, str):
            d2h_tiid = render_tiid_value(d2h_tiid, dev_id, dev_board_id,
                                         f'topo.json d2h layer{d2h_net_layer} device{dev_id}')
        d2h_edge = {
            'link_type': 'PEER2NET',
            'local_a': dev_global,
            'local_a_ports': ['d2h'],
            'net_layer': d2h_net_layer,
            'position': d2h_position,
            'protocols': d2h_protocols,
            'topo_type': 'CLOS',
            'topo_instance_id': d2h_tiid
        }
        topo_edges.append(d2h_edge)

hardware_type = config.get('hardware_type', None)
if not hardware_type:
    desc = config.get('description', '')
    if 'UBX' in desc.upper():
        hardware_type = 'UBX'
    elif 'UB' in desc.upper():
        hardware_type = 'UB'
    else:
        name_str = name.upper()
        if 'UBX' in name_str:
            hardware_type = 'UBX'
        elif 'UB' in name_str:
            hardware_type = 'UB'
        else:
            hardware_type = soc_version.upper() if soc_version != 'unknown' else 'UNKNOWN'

topo_json = {
    'edge_count': len(topo_edges),
    'edge_list': topo_edges,
    'hardware_type': hardware_type,
    'peer_count': device_num,
    'peer_list': peer_list,
    'version': '2.0'
}

# ============================================================
# 5.1 net_instance_id作用域(scope)解析
# ============================================================
# scope决定net_instance_id字符串格式(见build_net_instance_id):
#   board    → sp_{i}_srv_{j}_board_{b}_idx_{k}   (实例成员全部落在同一board内)
#   pod      → sp_{i}_srv_{j}_idx_{k}             (实例跨board但同属一个POD/server)
#   superpod → sp_{i}_idx_{k}                     (实例跨POD/server但同属一个超节点)
#   cluster  → cluster                            (集群级或HOST/d2h层)
# 默认映射:
#   server形态: {0: pod, 1: superpod, >=2: cluster} —— 与既有规则完全一致;
#   POD形态  : layer0按该层PEER2PEER直连边的board归属推导 ——
#              无直连(纯交换网)→pod; 直连全部板内(每board独立成网, 如1dpod64)→board;
#              存在跨board直连(src_board_id≠dst_board_id, 直连域跨board, 如hf)→pod;
#              layer1→superpod, >=2→cluster; 可用net_instance_scope字段逐层覆盖。
# 强制约束: POD形态下layer0存在PEER2PEER时 ——
#   板内直连(无跨board): scope必须为board, 各board的直连组是不同的拓扑实例, 只有layer0的
#     net_instance_id每board唯一, 业务侧按(net_layer, net_instance_id)归组时才不会把多个board的
#     直连组合并成同一个网络实例(否则GetRankSizeByLayer(0)由每board卡数膨胀为POD总卡数);
#   跨board直连: scope必须为pod, 直连域跨board为单一POD级实例, 若用board则各board的
#     net_instance_id不同, 跨board直连边会因两端niid不一致被业务侧过滤而失效。
_all_layers_in_topo = sorted({e['net_layer'] for e in topo_edges})
_p2p_layers_in_topo = {e['net_layer'] for e in topo_edges if e['link_type'] == 'PEER2PEER'}
# 存在跨board PEER2PEER直连边(src_board_id≠dst_board_id)的net_layer集合
_cross_board_p2p_layers = {e['net_layer'] for e in all_edges
                           if e['src_board_id'] != e['dst_board_id']}

def default_layer_scope(layer):
    """按形态与链路类型推导net_layer的默认作用域"""
    if layer >= 2:
        return 'cluster'
    if layer == 1:
        return 'superpod'
    # layer 0
    if not is_pod_form:
        return 'pod'
    if layer not in _p2p_layers_in_topo:
        return 'pod'          # 无直连, 纯交换网
    if layer in _cross_board_p2p_layers:
        return 'pod'          # 跨board直连, 直连域为单一POD级实例
    return 'board'            # 板内直连, 每board一个独立实例

layer_scope = {}
for layer in _all_layers_in_topo:
    layer_scope[layer] = net_instance_scope_cfg.get(layer, default_layer_scope(layer))
# net_instance_scope中声明了但topo里不存在的layer: 报错(防止配置写错layer号静默失效)
for layer in sorted(net_instance_scope_cfg.keys()):
    if layer not in _all_layers_in_topo:
        log_error(f"[错误] net_instance_scope声明的layer={layer}在拓扑中不存在 "
                  f"(实际存在的net_layer: {_all_layers_in_topo})，请检查配置文件")
        sys.exit(1)
# 强制约束校验: POD形态下layer0有PEER2PEER直连时, 按是否跨board确定唯一合法的scope
if is_pod_form and 0 in _p2p_layers_in_topo:
    if 0 in _cross_board_p2p_layers:
        if layer_scope.get(0) != 'pod':
            log_error(f"[错误] POD形态下net_layer=0存在跨board PEER2PEER直连(src_board_id≠dst_board_id), "
                      f"其直连域为单一POD级实例, net_instance_scope必须为pod (当前为{layer_scope.get(0)}), "
                      f"否则各board的net_instance_id不同, 跨board直连边会因两端niid不一致被业务侧过滤而失效。"
                      f"请移除net_instance_scope中layer=0的覆盖或改为pod")
            sys.exit(1)
    elif layer_scope.get(0) != 'board':
        log_error(f"[错误] POD形态下net_layer=0存在PEER2PEER板内直连, 其net_instance_scope必须为board "
                  f"(当前为{layer_scope.get(0)}), 否则各board的layer0网络实例会塌缩为同一个, "
                  f"多个board的板内直连组被误合并为一个大实例。请移除net_instance_scope中layer=0的覆盖")
        sys.exit(1)
# 反向告警: layer0无直连(跨board交换网)却声明为board作用域, 通常意味着漏配net_instance_scope
if is_pod_form and 0 in _all_layers_in_topo and 0 not in _p2p_layers_in_topo \
        and layer_scope.get(0) == 'board' and 0 not in net_instance_scope_cfg:
    log_warn(f"[警告] net_layer=0无PEER2PEER直连(跨board交换网), 但作用域推导为board, "
             f"该层网络实例将被切成每board一个; 若本层为整个POD的框内交换网, "
             f"请配置 net_instance_scope: [{{layer: 0, scope: \"pod\"}}]")

if is_pod_form:
    log_info(f"[信息] net_instance_id作用域: "
             + ", ".join(f"layer{k}={v}" for k, v in sorted(layer_scope.items())))

    # POD形态同构校验: 存在直连(PEER2PEER)时, 各board的直连边数必须一致且每个board
    # 都必须有直连条目(links按board逐一描述时最易漏配某个board)。
    # 计数口径: 板内边(src==dst)计1次; 跨board边(src!=dst)对两端board各计1次,
    # 保证对称拓扑(如hf每board既含板内fullmesh又对称参与跨board直连)各board计数相等。
    p2p_count_per_board = {}
    for edge in all_edges:
        sb = edge['src_board_id']
        db = edge['dst_board_id']
        p2p_count_per_board[sb] = p2p_count_per_board.get(sb, 0) + 1
        if db != sb:
            p2p_count_per_board[db] = p2p_count_per_board.get(db, 0) + 1
    if p2p_count_per_board:
        _counts = sorted(set(p2p_count_per_board.values()))
        if len(_counts) > 1:
            log_error(f"[错误] 各board的PEER2PEER边数不一致: "
                      f"{ {b: c for b, c in sorted(p2p_count_per_board.items())} }, "
                      f"同一POD内各board必须同构，请检查links条目的board_id/"
                      f"src_board_id/dst_board_id配置")
            sys.exit(1)
        _missing_boards = [b for b in range(board_num) if b not in p2p_count_per_board]
        if _missing_boards:
            log_error(f"[错误] board{_missing_boards}未配置任何PEER2PEER直连条目, "
                      f"而其余board各有{_counts[0]}条, 同一POD内各board必须同构，"
                      f"请检查links条目的board_id/src_board_id/dst_board_id配置")
            sys.exit(1)

# topo.json为POD/server级单份产物(local_id为POD全局id), 只构造一次, 输出到server目录
log_info(f"\n[信息] 构造topo.json完成: peer_list {device_num}个device"
         + (f" (POD全局local_id = board_id*{node_device_num} + board内物理id)" if is_pod_form else "")
         + f", edge_list {len(topo_edges)}条edge")

# ============================================================
# 6. 生成rootinfo.json (每叶子节点一份: server形态=单份; POD形态=每board独立构建)
# ============================================================
IP_AA = 1
IP_BB = 1

# 单server/pod生成: superpod id固定为0 (S=1、N=1恒为默认模式, superpod段AA = sp_id + 1 = 1)
SUPERPOD_ID = 0

DEVICE_PORT_START = 1

# 当前正在构造的board序号: POD形态下IP的CC段 = board_id, net_instance_id中携带board_{b};
# build_board_rootinfo()按board逐份构建时设置(server形态恒为0)
CUR_BOARD_ID = 0

# host IP: server形态为AA.BB.0.0(每server一个); POD形态为AA.BB.{board_id}.0(每board一个)
host_ip = f"{IP_AA}.{IP_BB}.{CUR_BOARD_ID}.0" if is_pod_form else f"{IP_AA}.{IP_BB}.0.0"
# host EID: die id=0, 端口号=0 (host非端口, 高位字段填0)
host_eid = build_eid_string(host_ip, SUPERPOD_ID, 0, 0)

# rootinfo EID分配: p2n_layers使用所有group的port_group中定义的layer
_all_pg_layers = set()
_all_pg_d2h_layers = set()
for gid, data in group_port_groups.items():
    _all_pg_layers.update(data['entries'].keys())
    _all_pg_d2h_layers.update(data['d2h_layers'])
p2n_layers_eid = sorted([l for l in _all_pg_layers if l not in _all_pg_d2h_layers])

# ============================================================
# net_layer总层数规则 (按文档):
#   当net_layer总层数 > 3时, rootinfo.json中超出部分不输出
#   (如配置文件中包含net_layer 0,1,2,3时, 只输出net_layer 0,1,2的内容)
# ============================================================
# 统计rootinfo.json中会输出的所有net_layer (与下方per-device逻辑保持一致):
#   - P2P端口固定输出在net_layer 0
#   - PEER2NET端口输出在port_group定义的各layer
#   - d2h端口输出在port_group定义的d2h layer (或自动分配的layer)
_all_rootinfo_layers = set()
for dev_id in range(node_device_num):
    dev_dl = get_die_list_for_device(dev_id)
    dev_gid = device_to_group.get(dev_id, _default_group_id)
    if any(len(d['peer2peer']) > 0 for d in dev_dl):
        _all_rootinfo_layers.add(0)
    _all_rootinfo_layers.update(p2n_layers_eid)
    dev_d2h_layers_collect = group_port_groups.get(dev_gid, {}).get('d2h_layers', d2h_layers)
    dev_all_d2h_layers_collect = set(dev_d2h_layers_collect)
    if any(len(d['h2d']) > 0 for d in dev_dl) and not dev_all_d2h_layers_collect:
        dev_all_d2h_layers_collect = {max(p2n_layers_eid) + 1 if p2n_layers_eid else 1}
    _all_rootinfo_layers.update(dev_all_d2h_layers_collect)

rootinfo_allowed_layers = None
if len(_all_rootinfo_layers) > 3:
    _sorted_rootinfo_layers = sorted(_all_rootinfo_layers)
    rootinfo_allowed_layers = set(_sorted_rootinfo_layers[:3])
    log_info(f"[信息] net_layer总层数({len(_sorted_rootinfo_layers)}) > 3, rootinfo.json不输出net_layer "
          f"{_sorted_rootinfo_layers[3:]}的内容 (仅输出net_layer {_sorted_rootinfo_layers[:3]})")

def build_board_rootinfo(board_id):
    """构建指定board的rootinfo.json (rank_list为board内物理id 0~node_device_num-1)。
    POD形态下IP的CC段=board_id、net_instance_id携带board_{b}与该board links配置的idx_{k}序号;
    server形态board_id恒为0, 产物与既有规则逐字节一致。
    返回: (rootinfo_json, 最后一个endPoint计数, 该board的EID总数)"""
    global CUR_BOARD_ID
    CUR_BOARD_ID = board_id
    port_counter = DEVICE_PORT_START
    total_eid_count = 0
    rank_list = []
    for dev_id in range(node_device_num):
        level_map = {}
        dev_dl = get_die_list_for_device(dev_id)
        dev_gid = device_to_group.get(dev_id, _default_group_id)

        # P2P端口: 按device对应的端口分配表生成 (支持group_id扩展)
        # net_layer总层数 > 3时, 超出部分不输出
        dev_has_p2p = any(len(d['peer2peer']) > 0 for d in dev_dl)
        if dev_has_p2p and (rootinfo_allowed_layers is None or 0 in rootinfo_allowed_layers):
            p2p_port_strs = []
            for d in dev_dl:
                for p in sorted(d['peer2peer']):
                    p2p_port_strs.append(f"{d['die_id']}/{p}")
            p2p_rank_addr_list = []
            for port_str in p2p_port_strs:
                ip_addr = alloc_endpoint_ip(port_counter)
                # P2P端口为普通物理端口: EID端口号取物理端口号
                die_id, port_num = derive_eid_die_port(0, [port_str])
                eid_str = build_eid_string(ip_addr, SUPERPOD_ID, die_id, port_num)
                port_counter += 1
                p2p_rank_addr_list.append({
                    'addr_type': 'EID',
                    'addr': eid_str,
                    'ports': [port_str],
                    'plane_id': 'plane_0'
                })
                total_eid_count += 1
            # net_instance序号k: 取links连接关系中该device(net_layer=0)配置的net_instance_id, 未配置默认0
            p2p_key = (0, build_net_instance_id(0, device_net_instance_id.get((board_id, dev_id, 0), 0), board_id))
            if p2p_key not in level_map:
                level_map[p2p_key] = []
            level_map[p2p_key].extend(p2p_rank_addr_list)

        # PEER2NET: EID分配使用port_group (非port_list)
        # 每个portGroup条目独立创建一个EID，按net_instance_id分组生成level_list
        # net_layer总层数 > 3时, 超出部分不输出
        for layer in p2n_layers_eid:
            if rootinfo_allowed_layers is not None and layer not in rootinfo_allowed_layers:
                continue
            pg_groups = get_p2n_portgroup_groups(layer, dev_id)

            if pg_groups:
                # 按net_instance_id分组
                # 双通道优先级(见resolve_niid_for_device): port_group条目显式配置(A)与
                # links中该device该层配置(B, 如device_to_switch_links)同时配置时逐device
                # 校验渲染值一致(不一致报错); 仅一方配置用该方; 均未配置回退默认0
                b_niid_raw = device_net_instance_id.get((board_id, dev_id, layer))
                niid_groups = {}
                for group in pg_groups:
                    niid = resolve_niid_for_device(group.get('net_instance_id'), b_niid_raw,
                                                   layer, dev_id, board_id, 'rootinfo')
                    if niid is None:
                        niid = 0
                    if niid not in niid_groups:
                        niid_groups[niid] = []
                    niid_groups[niid].append(group)

                multi_niid = len(niid_groups) > 1

                # 键可能为int序号或自定义字面量字符串, 统一排序(int在前, 与既有产物顺序一致)
                for niid in sorted(niid_groups.keys(),
                                   key=lambda x: (1, 0, x) if isinstance(x, str) else (0, x, '')):
                    groups = niid_groups[niid]
                    rank_addr_list = []
                    for group in groups:
                        port_strs = []
                        for pg_entry in group['entries']:
                            pg_die_id = pg_entry.get('die_id', 0)
                            pg_ports = pg_entry.get('port', [])
                            if isinstance(pg_ports, int):
                                pg_ports = [pg_ports]
                            for p in pg_ports:
                                port_strs.append(f"{pg_die_id}/{p}")

                        # 按文档"PG端口和物理端口说明": portGroup整体EID
                        # - ports列表有多个物理端口: PG端口, EID端口号固定0x3F
                        # - ports列表只有一个物理端口: 该端口不属于portGroup端口, 按物理端口分配地址
                        ip_addr = alloc_endpoint_ip(port_counter)
                        # UBOE协议端口: EID高位全0(纯v4映射形态), 见文档EID分配规则"注意"项
                        if is_uboe_protocol_group(group):
                            eid_str = build_eid_string(ip_addr, 0, 0, 0)
                        else:
                            die_id, port_num = derive_eid_die_port(layer, port_strs)
                            eid_str = build_eid_string(ip_addr, SUPERPOD_ID, die_id, port_num)
                        port_counter += 1

                        # Peer2Net端口: plane_id为"plane_pg_" + str(configured_plane_id), 默认0 → "plane_pg_0"
                        rank_addr_list.append({
                            'addr_type': 'EID',
                            'addr': eid_str,
                            'ports': port_strs,
                            'plane_id': build_plane_id(group.get('plane_id', 0))
                        })
                        total_eid_count += 1

                        # allocate_ip_by_physical_port=true时, 除portGroup整体EID外,
                        # 还为ports列表中每个物理端口额外分配独立EID(按物理端口分配地址);
                        # 单物理端口portGroup已按物理端口分配, 不再额外分配
                        allocate_by_physical = group.get('allocate_ip_by_physical_port')
                        if allocate_by_physical is None:
                            allocate_by_physical = group_port_groups.get(dev_gid, {}).get('allocate_ip_by_physical_port', False)
                        if allocate_by_physical and len(port_strs) > 1:
                            for port_str in port_strs:
                                ip_addr = alloc_endpoint_ip(port_counter)
                                # UBOE协议端口: EID高位全0(纯v4映射形态)
                                if is_uboe_protocol_group(group):
                                    eid_str = build_eid_string(ip_addr, 0, 0, 0)
                                else:
                                    die_id, port_num = derive_eid_die_port(layer, [port_str])
                                    eid_str = build_eid_string(ip_addr, SUPERPOD_ID, die_id, port_num)
                                port_counter += 1
                                rank_addr_list.append({
                                    'addr_type': 'EID',
                                    'addr': eid_str,
                                    'ports': [port_str],
                                    'plane_id': build_plane_id(group.get('plane_id', 0))
                                })
                                total_eid_count += 1

                    if rank_addr_list:
                        net_inst_id = build_net_instance_id(layer, niid, board_id)
                        p2n_key = (layer, net_inst_id)
                        if p2n_key not in level_map:
                            level_map[p2n_key] = []
                        level_map[p2n_key].extend(rank_addr_list)
            else:
                port_strs = []
                for d in dev_dl:
                    port_strs.extend([f"{d['die_id']}/{p}" for p in get_peer2net_ports(d['die_id'], dev_id)])

                ip_addr = alloc_endpoint_ip(port_counter)
                # 无port_group配置时的回退分支: 聚合多端口按PG端口处理, 单端口按物理端口处理
                die_id, port_num = derive_eid_die_port(layer, port_strs)
                eid_str = build_eid_string(ip_addr, SUPERPOD_ID, die_id, port_num)
                port_counter += 1

                # Peer2Net端口: plane_id默认为"plane_pg_0"
                rank_addr_list = [{
                    'addr_type': 'EID',
                    'addr': eid_str,
                    'ports': port_strs,
                    'plane_id': build_plane_id(0)
                }]
                total_eid_count += 1

                # 无port_group配置时的回退分支: net_instance序号k取links中该device该层配置值, 默认0
                fb_key = (layer, build_net_instance_id(layer, device_net_instance_id.get((board_id, dev_id, layer), 0), board_id))
                if fb_key not in level_map:
                    level_map[fb_key] = []
                level_map[fb_key].extend(rank_addr_list)

        # per-device d2h EID分配 (支持group_id扩展)
        # EID分配仅依据port_group(非port_list)，遍历port_group中所有d2h layer分配独立EID
        dev_d2h_layers_pg_eid = group_port_groups.get(dev_gid, {}).get('d2h_layers', d2h_layers)
        dev_all_d2h_layers_eid = set(dev_d2h_layers_pg_eid)
        # 兼容: pin_map有h2d但无d2h配置时，自动分配一个layer
        dev_has_h2d = any(len(d['h2d']) > 0 for d in dev_dl)
        if dev_has_h2d and not dev_all_d2h_layers_eid:
            dev_all_d2h_layers_eid = {max(p2n_layers_eid) + 1 if p2n_layers_eid else 1}

        # net_layer总层数 > 3时, 超出部分不输出
        for d2h_net_layer in sorted(dev_all_d2h_layers_eid):
            if rootinfo_allowed_layers is not None and d2h_net_layer not in rootinfo_allowed_layers:
                continue
            ip_addr = alloc_endpoint_ip(port_counter)
            # d2h端口(端口名不带端口序号)不属于PG端口, 按物理端口分配地址:
            # die id与端口号从device端口分配表(pin_map)中value=3(h2d)对应的portId获取
            die_id, port_num = resolve_named_port_die_port('d2h', dev_dl)
            eid_str = build_eid_string(ip_addr, SUPERPOD_ID, die_id, port_num)
            port_counter += 1
            # 从port_list/port_group条目中读取plane_id (默认0)
            d2h_plane_id = 0
            dev_pl_entries_d2h = group_port_lists.get(dev_gid, {}).get('entries', parsed_port_lists)
            dev_pg_entries_d2h = group_port_groups.get(dev_gid, {}).get('entries', parsed_port_groups)
            if d2h_net_layer in dev_pl_entries_d2h:
                for item in dev_pl_entries_d2h[d2h_net_layer]:
                    d2h_plane_id = item.get('plane_id', 0)
                    break
            elif d2h_net_layer in dev_pg_entries_d2h:
                for group in dev_pg_entries_d2h[d2h_net_layer]:
                    d2h_plane_id = group.get('plane_id', 0)
                    break
            # d2h层net_instance_id: 支持port_group条目显式配置(通道A, int序号或字符串模板,
            # 模板按device渲染); 多个条目显式配置且渲染值不同时报错; 未配置回退默认0
            d2h_niid = None
            if d2h_net_layer in dev_pg_entries_d2h:
                for group in dev_pg_entries_d2h[d2h_net_layer]:
                    g_niid_raw = group.get('net_instance_id')
                    if g_niid_raw is None:
                        continue
                    g_niid = resolve_niid_for_device(g_niid_raw, None, d2h_net_layer,
                                                     dev_id, board_id, f'rootinfo d2h layer{d2h_net_layer}')
                    if d2h_niid is None:
                        d2h_niid = g_niid
                    elif d2h_niid != g_niid:
                        log_error(f"[错误] d2h layer{d2h_net_layer}的多个port_group条目显式配置的"
                                  f"net_instance_id渲染值不一致('{d2h_niid}' vs '{g_niid}'), 请检查配置文件")
                        sys.exit(1)
            d2h_key = (d2h_net_layer, build_net_instance_id(
                d2h_net_layer, d2h_niid if d2h_niid is not None else 0, board_id))
            if d2h_key not in level_map:
                level_map[d2h_key] = []
            level_map[d2h_key].extend([{
                'addr_type': 'EID',
                'addr': eid_str,
                'ports': ['d2h'],
                'plane_id': build_plane_id(d2h_plane_id)
            }])
            total_eid_count += 1

        level_list = []
        for (lvl_net_layer, lvl_net_instance_id), lvl_rank_addr_list in level_map.items():
            # net_type按文档规则: net_layer=0时为TOPO_FILE_DESC, net_layer>=1时为CLOS
            level_list.append({
                'net_layer': lvl_net_layer,
                'net_instance_id': lvl_net_instance_id,
                'net_type': 'TOPO_FILE_DESC' if lvl_net_layer == 0 else 'CLOS',
                'net_attr': '',
                'rank_addr_list': lvl_rank_addr_list
            })

        rank_list.append({
            'device_id': dev_id,
            'local_id': dev_id,
            'level_list': level_list
        })

    rootinfo_json = {
        'rank_count': node_device_num,
        'rank_list': rank_list,
        # topo_file_path指向POD/server级topo.json, 由第7节输出时统一填写
        'topo_file_path': '',
        'version': '2.0'
    }
    return rootinfo_json, port_counter - 1, total_eid_count

# ============================================================
# 7. 输出文件: topo.json每server/POD一份(输出目录根); rootinfo.json每叶子节点一份
#    (server形态=输出目录; POD形态=board{b}/子目录)
# ============================================================
# POD形态下topo.json为POD级单份产物(local_id为POD全局id = board_id*每board卡数+board内物理id);
# rootinfo.json仍按board拆分(每board一个独立host节点), 各board独立构建,
# 仅IP的CC段(board_id)与net_instance_id中的board_{b}/idx_{k}序号不同,
# topo_file_path统一指向POD级topo.json
server_topo_path = os.path.join(OUTPUT_DIR, topo_filename)
with open(server_topo_path, 'w') as f:
    json.dump(topo_json, f, indent=4, ensure_ascii=False)

generated_nodes = []
last_counter = 0
total_eid_count = 0
for board_id in range(board_num):
    # POD形态: rootinfo输出到 board{b}/ 子目录(每board一个独立host节点); server形态: 直接输出到OUTPUT_DIR
    if is_pod_form:
        node_dir = os.path.join(OUTPUT_DIR, f'board{board_id}')
        os.makedirs(node_dir, exist_ok=True)
    else:
        node_dir = OUTPUT_DIR

    node_rootinfo_path = os.path.join(node_dir, rootinfo_filename)

    node_rootinfo, last_counter, total_eid_count = build_board_rootinfo(board_id)
    node_rootinfo['topo_file_path'] = server_topo_path
    with open(node_rootinfo_path, 'w') as f:
        json.dump(node_rootinfo, f, indent=4, ensure_ascii=False)

    generated_nodes.append((board_id, node_rootinfo_path))

# 输出汇总日志
if is_pod_form:
    log_info(f"\n[成功] POD形态生成完成: board数={board_num}, 每board卡数={node_device_num}, "
             f"POD总卡数={device_num}")
    log_info(f"[成功] 生成topo.json (POD级单份): {server_topo_path}")
    log_info(f"       peer_list: {device_num}个device "
             f"(local_id = board_id*{node_device_num} + board内物理id), edge_list: {len(topo_edges)}条edge")
    log_info(f"[成功] 生成rootinfo.json: {generated_nodes[0][1]} (等{board_num}份, 每board一份)")
    log_info(f"       rank_count: {node_device_num} (board内物理id), 单board EID总数: {total_eid_count} "
             f"(每board endPoint上限255)")
    log_info(f"       Host IP: {IP_AA}.{IP_BB}.0.0 ~ {IP_AA}.{IP_BB}.{board_num - 1}.0 (每board一个)")
    log_info(f"       Device IP: {IP_AA}.{IP_BB}.<board_id>.1 ~ {IP_AA}.{IP_BB}.<board_id>."
             f"{last_counter & 0xFF}")
    log_info(f"       layer0作用域: {layer_scope.get(0, 'N/A')} "
             f"(net_instance_id格式: "
             f"{'sp_i_srv_j_board_b_idx_k' if layer_scope.get(0) == 'board' else 'sp_i_srv_j_idx_k'})")
    for board_id, node_rootinfo_path in generated_nodes:
        log_debug(f"       board{board_id}: {node_rootinfo_path}")
else:
    log_info(f"\n[成功] 生成topo.json: {server_topo_path}")
    log_info(f"       peer_list: {device_num}个device, edge_list: {len(topo_edges)}条edge")
    log_info(f"[成功] 生成rootinfo.json: {generated_nodes[0][1]}")
    log_info(f"       rank_count: {node_device_num}, EID总数: {total_eid_count}")
    log_info(f"       Host IP: {host_ip} → EID: {host_eid}")
    last_cc = (last_counter >> 8) & 0xFF
    last_dd = last_counter & 0xFF
    log_info(f"       Device IP: {IP_AA}.{IP_BB}.0.1 ~ {IP_AA}.{IP_BB}.{last_cc}.{last_dd}")

PYTHON_SCRIPT

if [[ "$SCRIPT_LOG_LEVEL" -le 2 ]]; then
    echo ""
    echo "=========================================="
    echo "  生成完成"
    echo "=========================================="
    echo ""
fi
success_msg "topo.json 和 rootinfo.json 生成完成！"
