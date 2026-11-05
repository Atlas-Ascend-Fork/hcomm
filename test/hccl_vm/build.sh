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

set -e

function usage() {
  echo "Usage:"
  echo "  sh build.sh [-h | --help]"
  echo "              [--pkg]"
  echo "              [--package-path <PATH>] [--hcomm-path <PATH>]"
  echo "              [--version <VERSION>]"
  echo ""
  echo "Options:"
  echo "    -h, --help     Print usage"
  echo "    --pkg <PATH>    Set ascend package install path, default /usr/local/Ascend/cann"
  echo "    --package-path <PATH>"
  echo "                   Set ascend package install path, default /usr/local/Ascend/cann"
  echo "    --hcomm-path <PATH>"
  echo "                   Set hcomm code path, default /home/teamserver/xx/hcomm"
  echo "    --level1       Build northbound subset (proxy level0/1/2 + support libs"
  echo "                   + hccl-vm + aiv_kernel) AND the device side (aarch64),"
  echo "                   consistent with --full; hcomm repo is optional: when"
  echo "                   present (--hcomm-path/HCOMM_CODE_HOME) the hcomm-dependent"
  echo "                   plugins (checker/runner) are also built. Mutually"
  echo "                   exclusive with --full/--aicpu"
    echo "    --version <VERSION>"
    echo "                   Set build version to <VERSION>"
    echo "    --cmake-arg <ARG>"
    echo "                   Pass a single CMake argument through to the configure"
    echo "                   step (repeatable, e.g. --cmake-arg -DBUILD_TESTS=ON)"
  echo ""
echo "Environment:"
echo "    (存储后端模块随 src/storage_backends/ 目录编入，无环境变量开关；"
echo "     后端模块自身的构建选项经 --cmake-arg 透传)"
  echo ""
}

BUILD_MODE="HCCLVM"
aiv_build_enable="OFF"
level1_only_enable="OFF"
package_enable=false
cmake_extra_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
      -h | --help)
        usage
        exit 0
        ;;
    --pkg)
        package_enable=true
        shift 1
        ;;
    --aicpu)
        if [[ "${BUILD_MODE}" == "LEVEL1" ]]; then
            echo "Error: --aicpu is mutually exclusive with --level1."
            exit 1
        fi
        BUILD_MODE="DEVICE"
        shift 1
        ;;
    --aiv)
        aiv_build_enable="ON"
        shift 1
        ;;
    --cmake-arg)
        cmake_extra_args+=("$2")
        shift 2
        ;;
    --full)
        if [[ "${BUILD_MODE}" == "LEVEL1" ]]; then
            echo "Error: --full is mutually exclusive with --level1."
            exit 1
        fi
        BUILD_MODE="FULL"
        aiv_build_enable="ON"
        shift 1
        ;;
    --level1)
        if [[ "${BUILD_MODE}" != "HCCLVM" ]]; then
            echo "Error: --level1 is mutually exclusive with --full/--aicpu."
            exit 1
        fi
        BUILD_MODE="LEVEL1"
        # aiv_kernel 仅依赖 HCCL 仓(不依赖 hcomm), level1 模式保留构建
        aiv_build_enable="ON"
        level1_only_enable="ON"
        shift 1
        ;;
    --package-path)
        ascend_package_path="$2"
        shift 2
        ;;
    --hcomm-path)
        hcomm_code_path="$2"
        shift 2
        ;;
    --hccl-path)
        hccl_code_path="$2"
        shift 2
        ;;
    --version)
        VERSION_INFO="$2"
        shift 2
        ;;
    *)
        echo "Error: Undefined option: $1"
        usage
        exit 1
        ;;
    esac
done

# 检查 HCOMM_CODE_HOME 是否配置(level1 裁剪模式不强制依赖 hcomm 源码仓;
# 显式指定 --hcomm-path 时仍导出, 用于按需构建 checker/runner 插件)
if [[ "${BUILD_MODE}" != "LEVEL1" ]]; then
    if [ -n "${hcomm_code_path}" ];then
        HCOMM_CODE_PATH=${hcomm_code_path}
        export HCOMM_CODE_HOME=${HCOMM_CODE_PATH}  # hcomm仓代码根路径
    elif [[ -n "${HCOMM_CODE_HOME}" ]]; then
        HCOMM_CODE_PATH=${HCOMM_CODE_HOME}
    else
        echo "Error: Please set the hcomm code path through parameter --hcomm-path or HCOMM_CODE_HOME environment variable."
        echo "Set by environment variable: export HCOMM_CODE_HOME=/path/to/hcomm"
        echo "Set by parameter: bash build.sh --hcomm-path=/path/to/hcomm"
        exit 1
    fi

    # 检查 HCOMM_CODE_HOME 路径是否存在
    if [[ ! -d "${HCOMM_CODE_PATH}" ]]; then
        echo "Error: HCOMM_CODE_PATH directory does not exist: ${HCOMM_CODE_PATH}"
        exit 1
    fi
elif [[ -n "${hcomm_code_path}" ]]; then
    export HCOMM_CODE_HOME=${hcomm_code_path}
fi

# 当 BUILD_MODE 为 FULL/DEVICE 或 指定构建aiv时检查 HCCL_CODE_HOME 是否配置
if [[ "$BUILD_MODE" == "FULL" || "$BUILD_MODE" == "DEVICE" || "$aiv_build_enable" == "ON" ]]; then
    if [ -n "${hccl_code_path}" ]; then
        HCCL_CODE_PATH=${hccl_code_path}
        export HCCL_CODE_HOME=${HCCL_CODE_PATH}  # 设置 hccl 仓代码根路径
    elif [[ -n "${HCCL_CODE_HOME}" ]]; then
        HCCL_CODE_PATH=${HCCL_CODE_HOME}
    else
        echo "Error: BUILD_MODE is $BUILD_MODE. Please set the hccl code path through parameter --hccl-path or HCCL_CODE_HOME environment variable."
        echo "Set by environment variable: export HCCL_CODE_HOME=/path/to/hccl"
        echo "Set by parameter: bash build.sh --hccl-path=/path/to/hccl"
        exit 1
    fi

    # 检查 HCCL_CODE_PATH 路径是否存在
    if [[ ! -d "${HCCL_CODE_PATH}" ]]; then
        echo "Error: HCCL_CODE_PATH directory does not exist: ${HCCL_CODE_PATH}"
        exit 1
    fi
    
    echo "HCCL_CODE_PATH is set to: ${HCCL_CODE_PATH}"
fi

# Resolve CANN toolkit path via priority chain
_non_empty_candidates=("${ascend_package_path}" "${ASCEND_HOME_PATH}")
[ -n "${ASCEND_OPP_PATH}" ] && _non_empty_candidates+=("$(dirname "${ASCEND_OPP_PATH}")")
_dir_candidates=("${DEFAULT_TOOLKIT_INSTALL_DIR}" "${DEFAULT_INSTALL_DIR}")

_resolved=false
for _p in "${_non_empty_candidates[@]}"; do
    [ -n "${_p}" ] && { ASCEND_CANN_PACKAGE_PATH="${_p}"; _resolved=true; break; }
done
if ! ${_resolved}; then
    for _p in "${_dir_candidates[@]}"; do
        [ -d "${_p}" ] && { ASCEND_CANN_PACKAGE_PATH="${_p}"; _resolved=true; break; }
    done
fi

if ${_resolved} && [ "${ASCEND_CANN_PACKAGE_PATH}" = "${ascend_package_path}" ]; then
    source "${ASCEND_CANN_PACKAGE_PATH}/set_env.sh"
fi
${_resolved} || { echo "Error: Unable to locate toolkit installation path. Use --package-path to specify it."; exit 1; }

# 检查 source 是否成功（可选）
if [[ $? -ne 0 ]]; then
    echo "Error: Failed to source set_env.sh of cann"
    exit 1
fi

# CMake 选项均取目录默认（存储后端模块随目录编入）；模块专属配置（如后端模块
# 的链接形态选项）经 --cmake-arg 逐项透传，本脚本不感知具体后端名。

# ccache 存在则前置 PATH(Debian 系 wrapper 位于 /usr/lib/ccache), 对 host、
# device 交叉编译及 ExternalProject 子工程全部透明生效
if command -v ccache >/dev/null 2>&1 && [ -d /usr/lib/ccache ]; then
    export PATH="/usr/lib/ccache:${PATH}"
fi

# 按 MemAvailable/1.5GB 估算并行度(实测最大编译单元峰值~0.9GB), 夹在 [1, nproc];
# 可用环境变量 HVM_MAKE_JOBS 直接指定, 如 HVM_MAKE_JOBS=20 bash build.sh --full
function calc_make_jobs() {
    if [[ -n "${HVM_MAKE_JOBS:-}" ]]; then
        echo "${HVM_MAKE_JOBS}"
        return
    fi
    local jobs
    jobs=$(nproc)
    local mem_kb
    mem_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
    local mem_jobs=$(( mem_kb / 1024 / 1536 ))
    if (( mem_jobs < jobs )); then
        jobs=$mem_jobs
    fi
    if (( jobs < 1 )); then
        jobs=1
    fi
    echo "$jobs"
}

build_host() {
    # 创建 build 目录（如果不存在）并进入
    mkdir -p build
    cd build

    # 执行编译命令
    echo "Running cmake and make..."
    cmake -DBUILD_DEVICE_ARM=OFF -DBUILD_AIV=$aiv_build_enable -DBUILD_LEVEL1_ONLY=$level1_only_enable ${cmake_extra_args[@]+"${cmake_extra_args[@]}"} .. && make -j"$(calc_make_jobs)" && make install

    # 检查编译是否成功
    if [[ $? -eq 0 ]]; then
        echo "Build completed successfully!"
    else
        echo "Build failed!"
        exit 1
    fi

    if [[ $package_enable == true ]]; then
        cd ..
        tar -czvf hccl_vm_install.tar.gz hccl_vm_install
        echo "Package completed successfully!"
    fi
}

build_device() {
    # 独立顶层配置目录: 若与 host 共用 build/, --full 会先后以 ON/OFF 翻转
    # BUILD_DEVICE_ARM, 每次翻转都触发全量重新配置与 host 对象重编
    mkdir -p build_device_top
    cd build_device_top
    # 与 build_host 一致透传 level1 开关: 否则 level1 模式下 device 配置
    # (BUILD_DEVICE_ARM=ON) 仍会触发顶层对 HCOMM_CODE_HOME 的强制检查
    cmake -DBUILD_DEVICE_ARM=ON -DBUILD_LEVEL1_ONLY=$level1_only_enable ${cmake_extra_args[@]+"${cmake_extra_args[@]}"} .. && make -j"$(calc_make_jobs)"
    cd ..
}

if [[ "${BUILD_MODE}" == "HCCLVM" ]]; then
    build_host
elif [[ "${BUILD_MODE}" == "DEVICE" ]]; then
    build_device
elif [[ "${BUILD_MODE}" == "FULL" || "${BUILD_MODE}" == "LEVEL1" ]]; then
    # level1 与 full 一致: 先交叉编译 device 侧(bin/device 及设备侧打桩库),
    # 再构建 host 北向组件; 缺 device 会在 aclrtSetDevice 拉起 AICPU 子进程时失败
    build_device
    build_host
fi
