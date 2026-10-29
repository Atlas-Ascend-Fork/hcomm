#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -e

CURRENT_DIR=$(dirname $(readlink -f ${BASH_SOURCE[0]}))
BUILD_DIR=${CURRENT_DIR}/build
BUILD_UT_DIR=${CURRENT_DIR}/build_ut
BUILD_ST_DIR=${CURRENT_DIR}/build_st
BUILD_OUTPUT_DIR=${CURRENT_DIR}/build_out
LOGS_PATH="${CURRENT_DIR}/logs"
USER_ID=$(id -u)
CPU_NUM=$(cat /proc/cpuinfo | grep "^processor" | wc -l)
ASAN="false"
COV="false"
CUSTOM_OPTION="-DCMAKE_INSTALL_PREFIX=${BUILD_OUTPUT_DIR}"
ENABLE_BUILD_DEVICE="OFF"
ENABLE_BUILD_AARCH="OFF"
ENABLE_EXPERIMENTAL="OFF"
CANN_3RD_LIB_PATH="${CURRENT_DIR}/third_party"
CUSTOM_SIGN_SCRIPT=""
ENABLE_SIGN="false"
PACKAGE_TYPE="run"

MOCK_FWK_HLT="0"

BUILD_CB_TEST="false"

ENABLE_UT="off"
ENABLE_ST="off"
ST_TASKS=()
ENABLE_GCOV="off"
ENABLE_NO_EXEC="off"
BUILD_TYPE="Release"

if [ "${USER_ID}" != "0" ]; then
    DEFAULT_TOOLKIT_INSTALL_DIR="${HOME}/Ascend/ascend-toolkit/latest"
    DEFAULT_INSTALL_DIR="${HOME}/Ascend/latest"
else
    DEFAULT_TOOLKIT_INSTALL_DIR="/usr/local/Ascend/ascend-toolkit/latest"
    DEFAULT_INSTALL_DIR="/usr/local/Ascend/latest"
fi

function log() {
    local current_time=`date +"%Y-%m-%d %H:%M:%S"`
    echo "[$current_time] "$1
}

function set_env()
{
    source $ASCEND_CANN_PACKAGE_PATH/set_env.sh || echo "0"
}

function clean()
{
    if [ -n "${BUILD_DIR}" ];then
        rm -rf ${BUILD_DIR}
    fi

    if [ -n "${BUILD_OUTPUT_DIR}" ];then
        rm -rf ${BUILD_OUTPUT_DIR}
    fi
}

function build_cb_test_verify(){
    cd ${CURRENT_DIR}/examples/
    bash build.sh
}

function run_ctest() {
    # 设置 --noexec 选项，则跳过执行测试用例
    if [[ "$ENABLE_NO_EXEC" = "on" ]]; then
        log "Info: Skip executing tests"
        return 0
    fi

    local suite_name="$1"   # "ut" or "st"
    local log_dir="${LOGS_PATH}/${suite_name}"
    local ctest_log="${log_dir}/run.log"

    # 创建日志目录
    mk_dir "${log_dir}"

    # CTest 执行用例（超时时间：300s）
    log "Info: Running ${suite_name} testcases with ${CPU_NUM} parallel jobs"
    ctest -j ${CPU_NUM} \
          --verbose \
          --build-nocmake \
          --timeout 300 \
          --output-on-failure \
          --stop-on-failure \
          --test-output-size-failed 10000000 \
          2>&1 | tee "${ctest_log}"

    local ctest_ret=${PIPESTATUS[0]}
    if [ "${ctest_ret}" -ne 0 ]; then
        log "Error: Testcases failed"
    fi
    return ${ctest_ret}
}

function build_hccp_test() {
    log "Info: Building hccp_test with ${CPU_NUM} parallel jobs"
    # 显式回到主构建目录：函数内使用相对路径 cmake ..，须与调用方 cwd 解耦
    cd "${BUILD_DIR}"
    ENABLE_ST="on"
    ENABLE_TEST="on"

    local rc=0
    cmake .. ${CUSTOM_OPTION} -DENABLE_TEST=${ENABLE_TEST} -DENABLE_ST=${ENABLE_ST} -DPRODUCT_SIDE=host -DUSE_CXX11_ABI=0 -DHCCP_TEST_DIR=${BUILD_DIR}/hccp_test || rc=$?
    if [ "${rc}" -ne 0 ]; then
        log "Error: build_hccp_test: cmake config failed (rc=${rc})"
        exit 1
    fi
    cmake --build . --target hccp_test.bin hccp_verdict_test.bin test_hccp_service.bin ascend_hal_host_stub ascend_hal_device_stub urma mock net_adapt ccu-user-drv ascend_rdma_lite hrn3-rdmav34 hns-rdmav17 -j ${CPU_NUM} || rc=$?
    if [ "${rc}" -ne 0 ]; then
        log "Error: build_hccp_test: cmake build failed (rc=${rc})"
        exit 1
    fi
    cmake --install . --component hccp_test || rc=$?
    if [ "${rc}" -ne 0 ]; then
        log "Error: build_hccp_test: cmake install failed (rc=${rc})"
        exit 1
    fi
    # 设置 --noexec 选项，则跳过执行测试用例
    if [[ "${ENABLE_NO_EXEC}" = "on" ]]; then
        log "Info: Skip executing hccp_test cases"
        return 0
    fi
    # 运行 HCCP ST 用例。set -e 下命令失败会立即退出 shell，`if [ $? -ne 0 ]` 的错误分支
    # 不可达，本函数统一以 `|| rc=$?` 显式捕获退出码后再判定；124=超时，1=用例失败，139 等=崩溃
    # LD_LIBRARY_PATH 整体加引号并用 ${VAR:+:} 拼接：防参数分词与空值尾随冒号（空条目使 loader 搜 CWD）
    local hccp_timeout=300   # 全量实测约 50s，5 分钟上限（6 倍余量）
    local hccp_kill_after=30 # TERM 后 KILL 宽限，仅失败路径延长（最坏 330s）
    local hccp_test_dir="${BUILD_DIR}/hccp_test"
    local hccp_test_bin="${hccp_test_dir}/host/hccp_test.bin"
    local hccp_test_libs="${hccp_test_dir}/host:${hccp_test_dir}/common${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    # 判定链自测独立运行，不受业务用例的 gtest 过滤或重复次数影响。
    # 与下方业务路径同构显式捕获退出码：if ! 形式会丢弃原始 rc，超时(124)与用例失败不可区分
    local verdict_rc=0
    timeout --kill-after="${hccp_kill_after}" "${hccp_timeout}" \
        "${hccp_test_dir}/host/hccp_verdict_test.bin" --gtest_filter='*' --gtest_repeat=1 || verdict_rc=$?
    if [ "${verdict_rc}" -ne 0 ]; then
        if [ "${verdict_rc}" -eq 124 ]; then
            log "Error: build_hccp_test: verdict self-tests timed out after ${hccp_timeout}s"
        else
            log "Error: build_hccp_test: verdict self-tests failed (rc=${verdict_rc})"
        fi
        exit 1
    fi
    local hccp_rc=0
    timeout --kill-after="${hccp_kill_after}" "${hccp_timeout}" env \
        "LD_LIBRARY_PATH=${hccp_test_libs}" "${hccp_test_bin}" || hccp_rc=$?
    if [ "${hccp_rc}" -ne 0 ]; then
        if [ "${hccp_rc}" -eq 124 ]; then
            log "Error: build_hccp_test: hccp test cases timed out after ${hccp_timeout}s"
        else
            log "Error: build_hccp_test: hccp test cases failed (rc=${hccp_rc})"
        fi
        exit 1
    fi

    log "Info: Build and tests completed successfully!"
}

function build_st() {
    log "Info: Building st with ${CPU_NUM} parallel jobs"
    mk_dir "${BUILD_ST_DIR}"
    cd "${BUILD_ST_DIR}"

    # 配置 ST 用例代码
    local st_tasks=$(printf '%s;' "${ST_TASKS[@]}" | sed 's/;$//')
    log "Info: build_st: ST_TASKS=${st_tasks}"

    cmake -S ../ -B . \
        ${CUSTOM_OPTION} \
        -DST_TASKS=${st_tasks}
    if [ $? -ne 0 ]; then
        log "Error: build_st: cmake config failed"
        exit 1
    fi

    # 编译 ST 用例代码
    cmake --build . -j ${CPU_NUM}
    if [ $? -ne 0 ]; then
        log "Error: build_st: cmake build failed"
        exit 1
    fi

    # CTest 运行用例
    run_ctest "st"
    if [ $? -ne 0 ]; then
        log "Error: build_st: ctest execution failed"
        exit 1
    fi

    log "Info: Build and tests completed successfully!"
}


function mk_dir() {
  local create_dir="$1"  # the target to make
  mkdir -pv "${create_dir}"
  log "Info: Created ${create_dir}"
}

function build_ut() {
    log "Info: Building ut with ${CPU_NUM} parallel jobs"
    mk_dir "${BUILD_UT_DIR}"
    cd "${BUILD_UT_DIR}"

    # 避免加载系统库
    unset LD_LIBRARY_PATH

    # 配置 UT 用例代码
    cmake -S ../ -B . \
        ${CUSTOM_OPTION}
    if [ $? -ne 0 ]; then
        log "Error: build_ut: cmake config failed"
        exit 1
    fi

    # 编译 UT 用例代码
    cmake --build . -j ${CPU_NUM}
    if [ $? -ne 0 ]; then
        log "Error: build_ut: cmake build failed"
        exit 1
    fi

    # CTest 运行用例
    run_ctest "ut"
    if [ $? -ne 0 ]; then
        log "Error: build_ut: ctest execution failed"
        exit 1
    fi

    log "Info: Build and tests completed successfully!"
}

# 覆盖率报告生成公共实现，由 make_ut_gov / make_st_gov 调用
# 用法: make_coverage_report <报告类型(UT/ST)> <输出目录> <lcov捕获目录>...
# 示例: make_coverage_report UT cov_ut ${BUILD_UT_DIR}/test/ut/ ${BUILD_UT_DIR}/test/legacy/ut/
#       make_coverage_report ST cov_st ${BUILD_ST_DIR}/
function make_coverage_report() {
    local report_tag=$1
    local out_dir=$2
    shift 2

    # --noexec 时用例未执行，构建树里只有 .gcno 没有 .gcda，lcov/genhtml 会以非 0 退出；
    # 叠加 set -e 会让“只编译不执行”的流水线整体失败，这里直接跳过并提示
    if [[ "X${ENABLE_NO_EXEC}" = "Xon" ]]; then
        log "Warn: --noexec is set, no test data collected, skip ${report_tag} coverage report"
        return 0
    fi

    log "Info: Generating ${report_tag} coverage statistics, please wait..."
    rm -rf ${CURRENT_DIR}/${out_dir}
    mk_dir ${CURRENT_DIR}/${out_dir}
    cd ${CURRENT_DIR}/${out_dir}
 
    # 解析 lcov 版本号
    local major_version
    if ! major_version=$(set -o pipefail; lcov --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)*' | head -1 | cut -d. -f1); then
        log "Error: Failed to parse lcov major version number, please check 'lcov --version'" >&2
        exit 1
    fi
 
    # 适配 lcov 2.x 版本，支持并行处理覆盖率报告
    # 注意：--ignore-errors需与其取值整体写入变量，lcov 1.x分支置空时整个参数随之消失；
    # 否则残留的孤立--ignore-errors会吞掉后续-o，导致"Extra parameter found"参数错位报错
    local LCOV_IGNORE_ERRORS LCOV_PARALLEL GENHTML_IGNORE_ERRORS

    if [[ "${major_version}" -ge 2 ]]; then
        log "Info: Detected lcov version 2.x, running with ${CPU_NUM} parallel jobs"
        LCOV_IGNORE_ERRORS="--ignore-errors mismatch,corrupt,empty,inconsistent,negative,unused"
        LCOV_PARALLEL="-j ${CPU_NUM}"
        GENHTML_IGNORE_ERRORS="--ignore-errors inconsistent,corrupt"
    else
        LCOV_IGNORE_ERRORS=""
        LCOV_PARALLEL=""
        GENHTML_IGNORE_ERRORS=""
    fi
 
    # 捕获覆盖率数据（lcov -c 的捕获目录由剩余参数传入，支持一个或多个）
    local capture_args=""
    local dir
    for dir in "$@"; do
        capture_args="${capture_args} -d ${dir}"
    done
    lcov -c \
         ${LCOV_PARALLEL} \
         ${capture_args} \
         ${LCOV_IGNORE_ERRORS} \
         -o coverage.info
    # 排除路径
    lcov -r coverage.info \
         */src/base_comm/resources/hccp/external_depends/* \
         ${LCOV_PARALLEL} \
         ${LCOV_IGNORE_ERRORS} \
         -o coverage.info
    # 提取目标路径
    lcov -e coverage.info \
         */src/* \
         ${LCOV_PARALLEL} \
         ${LCOV_IGNORE_ERRORS} \
         -o coverage.info
 
    # 生成覆盖率报告，html格式
    # 必须显式指定 -o：lcov 2.0-1 的 genhtml 未传 -o 时会拼接出 /cmd_line 并因无权限写失败；
    # 选项放在 tracefile 之前，避免 POSIXLY_CORRECT 等环境下选项被当作输入文件
    genhtml ${LCOV_PARALLEL} ${GENHTML_IGNORE_ERRORS} -o . coverage.info
    # 明确统计范围：lcov -e 只保留 coverage.info 中已存在的记录，未插桩的代码不会出现在报告里，
    # 因此报告百分比的分母只是编译进 ${report_tag} target 的那部分源码，而非全量 src/
    log "Warn: Coverage scope is limited to instrumented sources linked into the ${report_tag} target;"
    log "Warn: sources that were never instrumented are absent from the report (not shown as 0%),"
    log "Warn: so the reported percentage is NOT the coverage of the whole src/ directory"
    log "Info: ${report_tag} coverage statistics generated successfully"
}

function make_ut_gov() {
    # 生成 UT 覆盖率报告
    if [[ "X$ENABLE_UT" = "Xon" && "X$ENABLE_GCOV" = "Xon" ]]; then
        make_coverage_report UT cov_ut ${BUILD_UT_DIR}/test/ut/ ${BUILD_UT_DIR}/test/legacy/ut/
    fi
}
 
function make_st_gov() {
    # 生成 ST 覆盖率报告
    if [[ "X$ENABLE_ST" = "Xon" && "X$ENABLE_GCOV" = "Xon" ]]; then
        make_coverage_report ST cov_st ${BUILD_ST_DIR}/
    fi
}

function build_hcomm() {
    log "Info: Building hcomm with ${CPU_NUM} parallel jobs"

    # 设置 hcc 编译器工具链
    export TOOLCHAIN_DIR="${ASCEND_CANN_PACKAGE_PATH}/toolkit/toolchain/hcc"

    # 创建构建目录
    mk_dir "${BUILD_DIR}"
    cd "${BUILD_DIR}"

    # 配置
    cmake -S ../ -B . ${CUSTOM_OPTION}
    if [ $? -ne 0 ]; then
        log "Error: cmake config failed"
        return 1
    fi

    # 编译
    cmake --build . -j ${CPU_NUM}
    if [ $? -ne 0 ]; then
        log "Error: cmake build failed"
        return 1
    fi

    # 打包
    make package -j ${CPU_NUM}
    if [ $? -ne 0 ]; then
        log "Error: make package failed"
        return 1
    fi
}

# print usage message
function usage() {
  echo "Usage:"
  echo "  sh build.sh --pkg [-h | --help] [-j<N>]"
  echo "              [--cann_3rd_lib_path=<PATH>] [-p|--package-path <PATH>]"
  echo "              [--asan]"
  echo "              [--sign-script <PATH>] [--enable-sign]"
  echo ""
  echo "Options:"
  echo "    -h, --help     Print usage"
  echo "    --asan         Enable AddressSanitizer"
  echo "    --build-type=<TYPE>"
  echo "                   Specify build type (TYPE options: Release/Debug), Default: Release"
  echo "    --cov          Enable coverage statistics (with -u/--ut or -s/--st)"
  echo "    -j<N>          Set the number of threads used for building, default is 8"
  echo "    --pkg-type=<TYPE>"
  echo "                   Specify package type (TYPE option: run/rpm/deb/all), Default: run"
  echo "    --cann_3rd_lib_path=<PATH>"
  echo "                   Set ascend third_party package install path, default ./output/third_party"
  echo "    -p, --package-path <PATH>"
  echo "                   Set ascend package install path, default /usr/local/Ascend/cann"
  echo "    --sign-script <PATH>"
  echo "                   Set sign-script's path to <PATH>"
  echo "    --enable-sign"
  echo "                   Enable to sign"
  echo "    -u, --ut       Run all unit tests (UT)"
  echo "    -s, --st       Run all system tests (ST)"
  echo "    --hccp_test    Build and run HCCP ST test (hccp_test.bin)"
  echo "    --noexec       Run build and skip executing tests"
  echo "    --experimental Build experimental components"
  echo ""
}

while [[ $# -gt 0 ]]; do
    case "$1" in
      -h | --help)
        usage
        exit 0
        ;;
    -j*)
        if [[ "$1" == "-j" ]]; then
            CPU_NUM="$2"
            shift 2
        else
            CPU_NUM="${1#-j}"
            shift
        fi
        ;;
    --build-type=*)
        OPTARG=$1
        BUILD_TYPE="${OPTARG#*=}"
        shift
        ;;
    --ccache)
        CCACHE_PROGRAM="$2"
        shift 2
        ;;
    -p|--package-path)
        ascend_package_path="$2"
        shift 2
        ;;
    --pkg)
        # 跳过 --pkg，不做处理
        shift
        ;;
    --pkg-type=*)
 	     PACKAGE_TYPE="${1#*=}"
 	     case "${PACKAGE_TYPE}" in
 	         run|rpm|deb|all) ;;
 	         *)
 	             log "Error: --pkg-type only supports run/rpm/deb/all"
 	             exit 1
 	             ;;
 	     esac
 	     shift
 	     ;;
    --cann_3rd_lib_path=*)
        OPTARG=$1
        CANN_3RD_LIB_PATH="$(realpath ${OPTARG#*=})"
        shift
        ;;
    --noexec)
        ENABLE_NO_EXEC="on"
        shift
        ;;
    --experimental)
        ENABLE_EXPERIMENTAL="ON"
        shift
        ;;
    -u|--ut)
        ENABLE_TEST="on"
        ENABLE_UT="on"
        shift
        ;;
    -s|--st)
        ENABLE_TEST="on"
        ENABLE_ST="on"
        ST_TASKS+=("all")
        shift
        ;;
    --hccp_test)
        ENABLE_TEST="on"
        ENABLE_ST="on"
        RUN_HCCP_TEST="on"
        shift
        ;;
    --open_hccl_test)
        ENABLE_TEST="on"
        ENABLE_ST="on"
        ST_TASKS+=("open_hccl_test")
        shift
        ;;
    --executor_hccl_test)
        ENABLE_TEST="on"
        ENABLE_ST="on"
        ST_TASKS+=("executor_hccl_test")
        shift
        ;;
    --executor_reduce_hccl_test)
        ENABLE_TEST="on"
        ENABLE_ST="on"
        ST_TASKS+=("executor_reduce_hccl_test")
        shift
        ;;
    --executor_pipeline_hccl_test)
        ENABLE_TEST="on"
        ENABLE_ST="on"
        ST_TASKS+=("executor_pipeline_hccl_test")
        shift
        ;;
    --legacy_all_testcase)
        ENABLE_TEST="on"
        ENABLE_ST="on"
        ST_TASKS+=("legacy_all_testcase")
        shift
        ;;
    --legacy_aicpu_2d_testcase)
        ENABLE_TEST="on"
        ENABLE_ST="on"
        ST_TASKS+=("legacy_aicpu_2d_testcase")
        shift
        ;;
    --legacy_ccu_2d_testcase)
        ENABLE_TEST="on"
        ENABLE_ST="on"
        ST_TASKS+=("legacy_ccu_2d_testcase")
        shift
        ;;
    --legacy_ccu_1d_hf16p_testcase)
        ENABLE_TEST="on"
        ENABLE_ST="on"
        ST_TASKS+=("legacy_ccu_1d_hf16p_testcase")
        shift
        ;;
    --legacy_ccu_1d_testcase_part1)
        ENABLE_TEST="on"
        ENABLE_ST="on"
        ST_TASKS+=("legacy_ccu_1d_testcase_part1")
        shift
        ;;
    --legacy_ccu_1d_testcase_part2)
        ENABLE_TEST="on"
        ENABLE_ST="on"
        ST_TASKS+=("legacy_ccu_1d_testcase_part2")
        shift
        ;;
    --legacy_alg_ccu_reduce)
        ENABLE_TEST="on"
        ENABLE_ST="on"
        ST_TASKS+=("legacy_alg_ccu_reduce")
        shift
        ;;
    --legacy_function_ut_testcase)
        ENABLE_TEST="on"
        ENABLE_ST="on"
        ST_TASKS+=("legacy_function_ut_testcase")
        shift
        ;;
    --legacy_alg_testcase)
        ENABLE_TEST="on"
        ENABLE_ST="on"
        ST_TASKS+=("legacy_alg_testcase")
        shift
        ;;
     --aicpu)
        ENABLE_BUILD_DEVICE="ON"
        shift
        ;;
    --full)
        ENABLE_BUILD_DEVICE="ON"
        shift
        ;;
    --build_aarch)
        ENABLE_BUILD_AARCH="ON"
        shift
        ;;
    --asan)
        ASAN="true"
        shift
        ;;
    --cov)
        ENABLE_GCOV="on"
        COV="true"
        shift
        ;;
    --noclean)
        # 默认不清理构建目录
        shift
        ;;
    --cb_test_verify)
        BUILD_CB_TEST="true"
        shift
        ;;
    --enable-sign)
        ENABLE_SIGN="true"
        shift
        ;;
    --sign-script)
        CUSTOM_SIGN_SCRIPT="$(realpath $2)"
        shift 2
        ;;
    *)
        log "Error: Undefined option: $1"
        usage
        exit 1
        ;;
    esac
done

if [ "${ASAN}" == "true" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_ASAN=ON"
fi

if [ "${COV}" == "true" ];then
    # 覆盖率报告由 UT/ST 流程生成，单独使用 --cov 不会产生任何报告，直接报错避免静默无效
    if [[ "X${ENABLE_UT}" != "Xon" && "X${ENABLE_ST}" != "Xon" ]]; then
        log "Error: --cov must be used together with -u/--ut or -s/--st"
        exit 1
    fi
    if [ "${RUN_HCCP_TEST}" == "on" ]; then
        log "Error: --cov is not supported with --hccp_test"
        exit 1
    fi
    CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_GCOV=ON"
fi

if [ -n "${ascend_package_path}" ];then
    ASCEND_CANN_PACKAGE_PATH=${ascend_package_path}
elif [ -n "${ASCEND_HOME_PATH}" ];then
    ASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME_PATH}
elif [ -n "${ASCEND_OPP_PATH}" ];then
    ASCEND_CANN_PACKAGE_PATH=$(dirname ${ASCEND_OPP_PATH})
elif [ -d "${DEFAULT_TOOLKIT_INSTALL_DIR}" ];then
    ASCEND_CANN_PACKAGE_PATH=${DEFAULT_TOOLKIT_INSTALL_DIR}
elif [ -d "${DEFAULT_INSTALL_DIR}" ];then
    ASCEND_CANN_PACKAGE_PATH=${DEFAULT_INSTALL_DIR}
else
    log "Error: Please set the toolkit package installation directory through parameter -p|--package-path."
    exit 1
fi

CUSTOM_OPTION="${CUSTOM_OPTION} -DASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}"
CUSTOM_OPTION="${CUSTOM_OPTION} -DASCEND_INSTALL_PATH=${ASCEND_CANN_PACKAGE_PATH}"
CUSTOM_OPTION="${CUSTOM_OPTION} -DCANN_3RD_LIB_PATH=${CANN_3RD_LIB_PATH}"

CUSTOM_OPTION="${CUSTOM_OPTION} -DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_BUILD_DEVICE=${ENABLE_BUILD_DEVICE}"
CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_BUILD_AARCH=${ENABLE_BUILD_AARCH}"
CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_EXPERIMENTAL=${ENABLE_EXPERIMENTAL}"
CUSTOM_OPTION="${CUSTOM_OPTION} -DPACKAGE_TYPE=${PACKAGE_TYPE}"

CUSTOM_OPTION="${CUSTOM_OPTION} -DCUSTOM_SIGN_SCRIPT=${CUSTOM_SIGN_SCRIPT}"
CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_SIGN=${ENABLE_SIGN}"
CUSTOM_OPTION="${CUSTOM_OPTION} -DPRODUCT=ascend"

CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_TEST=${ENABLE_TEST}"
CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_UT=${ENABLE_UT}"
CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_ST=${ENABLE_ST}"
CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_GCOV=${ENABLE_GCOV}"
# 导出 compile_commands.json，供 clang-tidy 增量检查使用
CUSTOM_OPTION="${CUSTOM_OPTION} -DCMAKE_EXPORT_COMPILE_COMMANDS=ON"

set_env

mkdir -p "${BUILD_DIR}"
cd ${BUILD_DIR}

if [ "${ENABLE_UT}" == "on" ]; then
    # Clang-Tidy 增量检查脚本单测（纯 python，无 CANN 依赖）
    python3 "${CURRENT_DIR}/scripts/clang-tidy/clang_tidy_check_test.py"
    # 编译并执行 UT
    build_ut
    make_ut_gov
elif [ "${RUN_HCCP_TEST}" == "on" ]; then
    # 编译并执行 HCCP ST 测试 (hccp_test.bin)
    build_hccp_test
elif [ "${ENABLE_ST}" == "on" ]; then
    # 编译并执行 ST
    build_st
    make_st_gov
    # 触发 HCCP ST 测试 (hccp_test.bin)
    build_hccp_test
elif [ "${BUILD_CB_TEST}" == "true" ]; then
    # 前冒烟测试
    log "Info: Building cb_test_verify"
    # build_cb_test_verify
    if grep -q "Make Failure" ${BUILD_DIR}/build.log || grep -q "Make test Failure" ${BUILD_DIR}/build.log; then
        log "Info: Building cb_test_verify failed"
        exit 1
    else
        log "Info: Building cb_test_verify success"
    fi
else
    # 编译打包 hcomm 包
    build_hcomm
    if [ $? -ne 0 ]; then
        log "Error: build hcomm failed"
        exit 1
    fi
    # Clang-Tidy incremental check: after build, compile_commands.json ready.
    # Only blocks new-code complexity warnings. Fallback: skip (exit 0) if
    # clang-tidy / compile_commands.json / base-ref unavailable.
    if [ "${ENABLE_CLANG_TIDY_CHECK:-on}" == "on" ] && [ -f "${BUILD_DIR}/compile_commands.json" ]; then
        log "Info: Running clang-tidy incremental check (ENABLE_CLANG_TIDY_CHECK=${ENABLE_CLANG_TIDY_CHECK:-on})"
        if ! bash ${CURRENT_DIR}/scripts/clang-tidy/clang_tidy_check.sh --build-dir ${BUILD_DIR}; then
            log "Error: clang-tidy incremental check failed (new code complexity violation)"
            exit 1
        fi
    fi
fi
