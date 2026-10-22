#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -euo pipefail

CURRENT_DIR=$(dirname $(dirname $(readlink -f ${BASH_SOURCE[0]})))
WORK_DIR="${CURRENT_DIR}/third_party"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --3rd_party)
            if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
                echo "Error: --3rd_party requires a directory." >&2
                exit 1
            fi
            WORK_DIR="$2"
            shift 2
            ;;
        --3rd_party=*)
            WORK_DIR="${1#*=}"
            if [[ -z "${WORK_DIR}" ]]; then
                echo "Error: --3rd_party requires a directory." >&2
                exit 1
            fi
            shift
            ;;
        -h|--help)
            echo "Usage: bash scripts/download_3rd_party.sh [--3rd_party <directory>]"
            echo "Default: <repo_root>/third_party; all patches are saved in <directory>/patch."
            exit 0
            ;;
        *)
            echo "Error: unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

OBS_URL="https://cann-3rd.obs.cn-north-4.myhuaweicloud.com"
# hcomm_utils 预编译包与构建机架构一一对应，仅下载当前架构的包；哈希与 cmake/hcomm_utils.cmake 一致
case "$(uname -m)" in
    x86_64)
        HCOMM_UTILS_ARCH="x86_64"
        HCOMM_UTILS_SHA256="f2c6f5e8f4ba3561e39625f0773abb98b94923e31a7c20010959566aaacc29df"
        ;;
    aarch64)
        HCOMM_UTILS_ARCH="aarch64"
        HCOMM_UTILS_SHA256="f33f9bc5b6f6e2c74c8ac461a1f0e368219500ab9361cf63b73d1ac99567b3f3"
        ;;
    *)
        echo "Error: unsupported architecture $(uname -m), cann-hcomm-utils only provides x86_64/aarch64 packages." >&2
        exit 1
        ;;
esac
HCOMM_UTILS_FILE="cann-hcomm-utils_9.2.0_linux-${HCOMM_UTILS_ARCH}.tar.gz"
# Source: https://gitcode.com/cann/cmake/tree/master-061/third_party
# Tag commit: 738d391e5bc046d6823f8a04af7b30e2afa9ba96
# Format: "URL SHA256 Path relative to third_party"
DOWNLOAD_LIST=(
    "${OBS_URL}/abseil-cpp/abseil-cpp-20230802.1.tar.gz 987ce98f02eefbaf930d6e38ab16aa05737234d7afbab2d5c4ea7adbe50c28ed abseil-cpp-20230802.1.tar.gz"
    "https://gitcode.com/cann-src-third-party/abseil-cpp/releases/download/20230802.1-h0/backport-CVE-2025-0838.patch a4e4ef6fc4f4c1ed871a3472c5e650068aaadaad2adfef2c8ccb00918bb36d8c patch/backport-CVE-2025-0838.patch"
    "${OBS_URL}/boost/boost_1_87_0.tar.gz f55c340aa49763b1925ccf02b2e83f35fdcf634c9d5164a2acb87540173c741d boost_1_87_0.tar.gz"
    "https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20260729_newest/${HCOMM_UTILS_FILE} ${HCOMM_UTILS_SHA256} ${HCOMM_UTILS_FILE}"
    # Version and SHA256 match cmake/fetch_cann_cmake.cmake.
    "${OBS_URL}/cmake/cmake-master-061.tar.gz d609b42d3acbb4c5a0177b6d85cf3c120839c123978e953fd20630d5585e3f78 cmake-master-061.tar.gz"
    "${OBS_URL}/googletest/googletest-1.14.0.tar.gz 8ad598c73ad796e0d8280b082cebd82a630d73e73cd3c70057938a6501bba5d7 googletest-1.14.0.tar.gz"
    "${OBS_URL}/json/json-3.12.0.tar.gz 4b92eb0c06d10683f7447ce9406cb97cd4b453be18d7279320f7b2f025c10187 json-3.12.0.tar.gz"
    "${OBS_URL}/makeself/makeself-release-2.5.0.tar.gz 705d0376db9109a8ef1d4f3876c9997ee6bed454a23619e1dbc03d25033e46ea makeself-release-2.5.0.tar.gz"
    "${OBS_URL}/makeself/fix/makeself-2.5.0.patch 0ecac3fe200221353322626900d5257a59d98347f83653b98456e08475994486 patch/makeself-2.5.0.patch"
    "${OBS_URL}/mockcpp/mockcpp-2.7.tar.gz 73ab0a8b6d1052361c2cebd85e022c0396f928d2e077bf132790ae3be766f603 mockcpp-2.7.tar.gz"
    "https://gitcode.com/cann-src-third-party/mockcpp/releases/download/v2.7-h6/mockcpp-2.7-h6.patch 238de206a3462886d186ea6c60105ed7d095143cbeb3c083599cc1986cf593f1 patch/mockcpp-2.7-h6.patch"
    "${OBS_URL}/openssl/openssl-openssl-3.0.9.tar.gz 2eec31f2ac0e126ff68d8107891ef534159c4fcfb095365d4cd4dc57d82616ee openssl-openssl-3.0.9.tar.gz"
    "${OBS_URL}/protobuf/protobuf-25.1.tar.gz 9bd87b8280ef720d3240514f884e56a712f2218f0d693b48050c836028940a42 protobuf-25.1.tar.gz"
    "${OBS_URL}/rdma-core/rdma-core-42.7.tar.gz aa935de1fcd07c42f7237b0284b5697b1ace2a64f2bcfca3893185bc91b8c74d rdma-core-42.7.tar.gz"
    "${OBS_URL}/rdma-core/rdma-core-42.7.patch 54ca56b3b68bc465a78dd5839cd7110610745c7152a1dc3a72b265deeebb905f patch/rdma-core-42.7.patch"
)

TEMP_FILE=""
trap 'if [[ -n "${TEMP_FILE}" ]]; then rm -f -- "${TEMP_FILE}"; fi' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for TOOL in wget sha256sum; do
    if ! command -v "${TOOL}" >/dev/null 2>&1; then
        echo "Error: ${TOOL} is required." >&2
        exit 1
    fi
done

check_sha256() {
    printf '%s  %s\n' "$1" "$2" | sha256sum --check --status
}

for ITEM in "${DOWNLOAD_LIST[@]}"; do
    read -r URL SHA256 FILE_NAME <<< "${ITEM}"
    FILE_PATH="${WORK_DIR}/${FILE_NAME}"

    if [[ -f "${FILE_PATH}" ]]; then
        if check_sha256 "${SHA256}" "${FILE_PATH}"; then
            echo "Verified cache, skipping: ${FILE_NAME}"
            continue
        fi
        echo "Cached SHA256 mismatch, downloading again: ${FILE_NAME}"
        rm -f -- "${FILE_PATH}"
    fi

    mkdir -p -- "$(dirname "${FILE_PATH}")"
    TEMP_FILE=$(mktemp -- "${FILE_PATH}.part.XXXXXX")
    echo "Download from ${URL} to ${FILE_PATH}"
    # 证书校验失败（如企业网关代理重签名证书）时降级重试；文件完整性由 SHA256 校验保证
    if ! wget -q --show-progress --timeout=60 --tries=3 -O "${TEMP_FILE}" "${URL}"; then
        echo "Warning: secure download failed, retrying without certificate check: ${FILE_NAME}" >&2
        if ! wget -q --no-check-certificate --show-progress --timeout=60 --tries=3 -O "${TEMP_FILE}" "${URL}"; then
            echo "Error: download failed: ${FILE_NAME}" >&2
            exit 1
        fi
    fi
    if ! check_sha256 "${SHA256}" "${TEMP_FILE}"; then
        echo "Error: SHA256 mismatch: ${FILE_NAME}" >&2
        exit 1
    fi
    mv -f -- "${TEMP_FILE}" "${FILE_PATH}"
    TEMP_FILE=""
    echo "Downloaded and verified: ${FILE_NAME}"
done

# master-061 reads the rdma-core and abseil-cpp patches from separate Host and Device directories.
mkdir -p -- "${WORK_DIR}/patch/device"
cp -f -- "${WORK_DIR}/patch/backport-CVE-2025-0838.patch" "${WORK_DIR}/patch/device/backport-CVE-2025-0838.patch"
cp -f -- "${WORK_DIR}/patch/rdma-core-42.7.patch" "${WORK_DIR}/patch/device/rdma-core-42.7.patch"

echo "All downloads verified. Files are saved in '${WORK_DIR}'."
