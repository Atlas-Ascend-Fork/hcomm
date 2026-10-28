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
# 退出期专项用例 runner：
# 运行指定的退出期用例二进制（不清理直接退出 / 显式 graceful DeInit 后退出 / 显式 reset
# DeInit 后退出），断言退出码为 0；ASAN 构建时同时检查输出中无 use-after-free /
# AddressSanitizer 报错。
#
# 用法：bash run_exit_path_test.sh [用例可执行文件路径]
# 默认路径：本脚本同级目录下的 exit_without_cleanup（随 cmake 构建产物目录调整）。
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_BIN="${1:-${SCRIPT_DIR}/exit_without_cleanup}"
TEST_NAME="$(basename "${TEST_BIN}")"

if [[ ! -x "${TEST_BIN}" ]]; then
    echo "[run_exit_path_test] ERROR: test binary not found or not executable: ${TEST_BIN}" >&2
    exit 1
fi

# DeInit 顺序标记经 HCCL_INFO 输出：本二进制链接 hccl_llt 打桩库，其日志默认只打 ERROR，
# 须用 HCCL_UT_LOG_LEVEL=0 放开 INFO 落屏（ASCEND_* 变量仅对真实 slog 生效，保留给后续 ST/CI 接线）
export HCCL_UT_LOG_LEVEL=0
export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_SLOG_PRINT_TO_STDOUT=1

LOG_FILE="$(mktemp /tmp/${TEST_NAME}.XXXXXX.log)"
# 超时兜底（退出路径回归典型失效模式是死锁挂死）：240s 强杀，留 60s 余量给 ctest TIMEOUT(300)
timeout -k 5 240 "${TEST_BIN}" > "${LOG_FILE}" 2>&1
RET=$?

cat "${LOG_FILE}"

if [[ ${RET} -eq 124 || ${RET} -eq 137 ]]; then
    echo "[run_exit_path_test] FAILED: ${TEST_NAME} timed out (possible deadlock in exit path)" >&2
    rm -f "${LOG_FILE}"
    exit 1
fi

if [[ ${RET} -ne 0 ]]; then
    echo "[run_exit_path_test] FAILED: ${TEST_NAME} exit code ${RET} (expected 0)" >&2
    rm -f "${LOG_FILE}"
    exit 1
fi

# ASAN 构建时，退出码可能被 recover 选项掩盖，额外检查输出
if grep -qE 'AddressSanitizer|use-after-free|LeakSanitizer' "${LOG_FILE}"; then
    echo "[run_exit_path_test] FAILED: ${TEST_NAME} sanitizer error detected in output" >&2
    rm -f "${LOG_FILE}"
    exit 1
fi

# DeInit 动作顺序校验（仅显式 DeInit 用例）：step[1/9]~step[9/9] 标记须按序出现（R4 契约）
if [[ "${TEST_NAME}" == "exit_graceful_deinit" || "${TEST_NAME}" == "exit_reset_deinit" ]]; then
    mapfile -t STEPS < <(grep -oE 'step\[[0-9]+/9\]' "${LOG_FILE}")
    EXPECTED=()
    for i in 1 2 3 4 5 6 7 8 9; do
        EXPECTED+=("step[${i}/9]")
    done
    if [[ "${STEPS[*]:-}" != "${EXPECTED[*]}" ]]; then
        echo "[run_exit_path_test] FAILED: ${TEST_NAME} DeInit step order mismatch" >&2
        echo "  expected: ${EXPECTED[*]}" >&2
        echo "  actual:   ${STEPS[*]:-<none>}" >&2
        rm -f "${LOG_FILE}"
        exit 1
    fi
    echo "[run_exit_path_test] DeInit step order [1..9] verified"
fi

rm -f "${LOG_FILE}"
echo "[run_exit_path_test] PASSED: ${TEST_NAME} exit code 0, no sanitizer error"
exit 0
