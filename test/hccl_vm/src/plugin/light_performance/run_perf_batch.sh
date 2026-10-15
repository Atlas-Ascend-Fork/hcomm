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

# run_perf_batch_v2.sh - 批量性能仿真 (逐组同步执行)
#
# 遍历算子 × 数据量 × 规模, 每组: 启动hccl-vm → mock-comm → mpirun → @estimate → 退出
# 跑一组打印一组结果, 再启动下一组
#
# 用法:
#   ./run_perf_batch_v2.sh

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT="${SCRIPT_DIR}/../../.."
INSTALL_DIR="${PROJECT_ROOT}/hccl_vm_install"
BIN_DIR="${INSTALL_DIR}/bin"
OUTPUT_DIR="${PROJECT_ROOT}/perf_batch_results"

SCALES="8p 16p 32p 128p 256p 512p 1024p"
OPS="all_reduce all_gather reduce_scatter alltoall"
SIZES="16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608 16777216 33554432 67108864 134217728 268435456 536870912 1073741824"
DATA_TYPE="int32"
REDUCE_OP="sum"
EXPANSION_MODES="AI_CPU CCU_SCHED CCU_MS"
MP_TIMEOUT=300

# ---- 环境初始化 ----
CANN_ENV="/home/workspace/Ascend/cann/set_env.sh"
[ -f "$CANN_ENV" ] && source "$CANN_ENV"
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:${ASCEND_HOME_PATH}/devlib:${LD_LIBRARY_PATH}"
export RANK_TABLE_FILE="${INSTALL_DIR}/data/ranktable.json"
export HCCL_DFS_CONFIG="cluster_heartbeat:off,inconsistent_check:off"
export HCCLVM_ENABLE_DUMP_DATA="1"
export HCCL_VM_INSTALL_DIR="${INSTALL_DIR}"
export ASCEND_GLOBAL_LOG_LEVEL="1"
HCCL_TEST_BIN="${ASCEND_HOME_PATH}/tools/hccl_test/bin"

# ---- 展开模式 -> mpirun -a 参数 ----
algo_arg_of() {
    case "$1" in
        CCU_SCHED)  echo "-a ccu_sched" ;;
        CCU_MS)     echo "-a ccu_ms" ;;
        AI_CPU)     echo "" ;;
        *)          echo "" ;;
    esac
}

# ---- 规模 -> mock-comm 编号 (单pod多server, 保证全互联) ----
topo_id_of() {
    case "$1" in
        8p)    echo "118" ;;
        16p)   echo "1116" ;;
        32p)   echo "1132" ;;
        64p)   echo "1164" ;;
        128p)  echo "12_64" ;;
        256p)  echo "14_64" ;;
        512p)  echo "18_64" ;;
        1024p) echo "116_64" ;;
        *)     echo "$1" ;;
    esac
}

# ---- 规模 -> 集群配置文件 ----
cluster_of() {
    case "$1" in
        8p|16p|32p|64p)    echo "ascend950_cluster_4_server_1dpod64.yaml" ;;
        128p|256p)         echo "ascend950_cluster_4_server_1dpod64.yaml" ;;
        512p)              echo "ascend950_cluster_8_server_1dpod64.yaml" ;;
        1024p)             echo "ascend950_cluster_16_server_1dpod64.yaml" ;;
        *)                 echo "ascend950_cluster_4_server_1dpod64.yaml" ;;
    esac
}

size_str() {
    local s=$1
    if [ $s -ge 1073741824 ]; then echo "$((s / 1073741824))GB"
    elif [ $s -ge 1048576 ]; then echo "$((s / 1048576))MB"
    elif [ $s -ge 1024 ]; then echo "$((s / 1024))KB"
    else echo "${s}B"; fi
}

mkdir -p "$OUTPUT_DIR"
RUN_LOG="$OUTPUT_DIR/_hccl_vm_run.log"
SUMMARY_CSV="$OUTPUT_DIR/summary.csv"

TOTAL=$(( $(echo $EXPANSION_MODES | wc -w) * $(echo $SCALES | wc -w) * $(echo $OPS | wc -w) * $(echo $SIZES | wc -w) ))

echo "========================================================"
echo "批量性能仿真 (逐组同步执行)"
echo "  规模: $SCALES"
echo "  算子: $OPS"
echo "  数据量: $(size_str $(echo $SIZES | awk '{print $1}')) -> $(size_str $(echo $SIZES | awk '{print $NF}'))"
echo "  数据类型: $DATA_TYPE, Reduce操作: $REDUCE_OP"
echo "  展开模式: $EXPANSION_MODES"
echo "  总计 $TOTAL 组测试"
echo "  开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"

echo "模式,规模,算子,数据量,makespan(us),状态" > "$SUMMARY_CSV"

current=0
for exp_mode in $EXPANSION_MODES; do
    export HCCL_OP_EXPANSION_MODE="$exp_mode"
    algo_arg=$(algo_arg_of "$exp_mode")

    for scale in $SCALES; do
        topo_id=$(topo_id_of "$scale")
        cluster_yaml=$(cluster_of "$scale")

        for op in $OPS; do
            extra=""
            case "$op" in
                all_reduce|reduce|reduce_scatter) extra="-o $REDUCE_OP" ;;
        esac

        for size in $SIZES; do
            current=$((current + 1))
            sstr=$(size_str "$size")
            tag="${exp_mode}_${scale}_${op}_${sstr}"
            printf "[%d/%d] %s ... " "$current" "$TOTAL" "$tag"

            # 生成这一组的命令 (reset → mock-comm → mpirun → @estimate → exit)
            printf 'hccl-vm reset\n' > "$RUN_LOG.cmds"
            printf 'hccl-vm mock-comm %s\n' "$topo_id" >> "$RUN_LOG.cmds"
            printf 'timeout %s mpirun --allow-run-as-root --oversubscribe -np 1 %s/%s_test -b %s -e %s -d %s %s %s -w 0 -n 1 -c 0 < /dev/null > /dev/null 2>&1\n' \
                "$MP_TIMEOUT" "$HCCL_TEST_BIN" "$op" "$size" "$size" "$DATA_TYPE" "$extra" "$algo_arg" >> "$RUN_LOG.cmds"
            printf 'hccl-vm plugin run @estimate\n' >> "$RUN_LOG.cmds"
            printf 'exit\n' >> "$RUN_LOG.cmds"

            # 启动 hccl-vm, 管道喂命令, 输出存日志
            cd "$BIN_DIR"
            ./hccl-vm start "$cluster_yaml" --estimate < "$RUN_LOG.cmds" > "$RUN_LOG" 2>&1

            # 从日志提取 makespan
            makespan=$(grep -oP 'makespan\s+\K[\d.]+' "$RUN_LOG" | head -1)

            if [ -n "$makespan" ]; then
                echo "${exp_mode},${scale},${op},${sstr},${makespan},OK" >> "$SUMMARY_CSV"
                echo "makespan=${makespan} us"
            else
                echo "${exp_mode},${scale},${op},${sstr},N/A,FAIL" >> "$SUMMARY_CSV"
                echo "FAIL"
            fi

            # 保存 trace json (重命名包含算子/数据量/规模/模式)
            TRACE_DIR="${INSTALL_DIR}/plugin/light_performance/data"
            LATEST_TRACE=$(ls -t "$TRACE_DIR"/*/trace_*.json 2>/dev/null | head -1)
            if [ -n "$LATEST_TRACE" ]; then
                TRACE_OUT="$OUTPUT_DIR/trace_${exp_mode}_${scale}_${op}_${sstr}.json"
                cp "$LATEST_TRACE" "$TRACE_OUT"
            fi
        done
    done
done
done

# ==================== 输出汇总表格 ====================
echo ""
echo "========================================================"
echo "性能汇总"
echo "========================================================"

for exp_mode in $EXPANSION_MODES; do
    for scale in $SCALES; do
        echo ""
        echo "### ${exp_mode} / $scale (mock-comm $(topo_id_of $scale), cluster $(cluster_of $scale))"
        echo ""
        header="| 算子 |"
        separator="|------|"
        for size in $SIZES; do
            sstr=$(size_str "$size")
            header="${header} ${sstr} |"
            separator="${separator}------|"
        done
        echo "$header"
        echo "$separator"
        for op in $OPS; do
            row="| ${op} |"
            for size in $SIZES; do
                sstr=$(size_str "$size")
                makespan=$(grep "^${exp_mode},${scale},${op},${sstr}," "$SUMMARY_CSV" | cut -d',' -f5)
                [ -z "$makespan" ] && makespan="-"
                row="${row} ${makespan} |"
            done
            echo "$row"
        done
        echo ""
    done
done

OK_COUNT=$(grep ",OK$" "$SUMMARY_CSV" | wc -l)
FAIL_COUNT=$(grep ",FAIL$" "$SUMMARY_CSV" | wc -l)

# ==================== 导出 Excel ====================
SUMMARY_XLSX="$OUTPUT_DIR/summary.xlsx"
python3 - "$SUMMARY_CSV" "$SUMMARY_XLSX" <<'PYEOF'
import csv, sys
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter

csv_path, xlsx_path = sys.argv[1], sys.argv[2]

with open(csv_path, encoding="utf-8") as f:
    rows = list(csv.reader(f))

wb = Workbook()
ws = wb.active
ws.title = "性能汇总"

header_font = Font(bold=True, color="FFFFFF", size=11)
header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
center = Alignment(horizontal="center", vertical="center")
thin_border = Border(
    left=Side(style="thin"), right=Side(style="thin"),
    top=Side(style="thin"), bottom=Side(style="thin")
)

for r_idx, row in enumerate(rows, 1):
    for c_idx, val in enumerate(row, 1):
        cell = ws.cell(row=r_idx, column=c_idx, value=val)
        cell.alignment = center
        cell.border = thin_border
        if r_idx == 1:
            cell.font = header_font
            cell.fill = header_fill

# 自适应列宽
for c_idx in range(1, len(rows[0]) + 1):
    max_len = max(len(str(rows[r_idx - 1][c_idx - 1])) for r_idx in range(1, len(rows) + 1))
    ws.column_dimensions[get_column_letter(c_idx)].width = max_len + 4

# 冻结首行
ws.freeze_panes = "A2"
wb.save(xlsx_path)
PYEOF

echo "========================================================"
echo "成功: $OK_COUNT, 失败: $FAIL_COUNT"
echo "CSV:  $SUMMARY_CSV"
echo "Excel: $SUMMARY_XLSX"
echo "日志: $RUN_LOG"
echo "trace: ${INSTALL_DIR}/plugin/light_performance/data/*/perf_profiling.json"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"
