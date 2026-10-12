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

# pre-commit_check.sh - clang-format + oat-check wrapper
# Usage:
#   ./pre-commit_check.sh file1.cc file2.h      check specified files
#   ./pre-commit_check.sh --all / -all          check all C/C++ files
#   ./pre-commit_check.sh --modified / -modified check git modified files

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CXX_PATTERN='\.(c|h|cc|hh|cpp|hpp|cxx|hxx|h\.in|proto|inl)$'
OAT_PY=""
OAT_REPORT_DIR=""
TAB=$'\t'

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'

error()   { echo -e "${RED}ERROR: $*${NC}" >&2; }
success() { echo -e "${GREEN}$*${NC}"; }
warn()    { echo -e "${YELLOW}$*${NC}"; }
info()    { echo -e "${CYAN}$*${NC}"; }

cleanup() {
    [[ -n "${OAT_REPORT_DIR}" && -d "${OAT_REPORT_DIR}" ]] && rm -rf "${OAT_REPORT_DIR}"
}
trap cleanup EXIT

usage() {
    echo "Usage:"
    echo "  $0 file1.cc file2.h       check specified files"
    echo "  $0 --all / -all           check all C/C++ files in repo"
    echo "  $0 --modified / -modified check git modified files (staged + unstaged)"
    exit 1
}

check_deps() {
    if ! command -v pre-commit &>/dev/null; then
        error "pre-commit not found. Install it first:"
        echo "  pip3 install pre-commit"
        echo "  cd ${SCRIPT_DIR} && pre-commit install"
        exit 1
    fi
    if [[ ! -f "${SCRIPT_DIR}/.pre-commit-config.yaml" ]]; then
        error ".pre-commit-config.yaml not found in ${SCRIPT_DIR}"
        exit 1
    fi

    OAT_PY="$(find -L ~/.cache/pre-commit -path "*/py_env-*/bin/python" 2>/dev/null | head -1)"
    if [[ -z "${OAT_PY}" ]]; then
        warn "WARNING: OAT python binary not found in pre-commit cache. Run 'pre-commit run --all-files' once to install it."
        warn "oat-check will be skipped."
    fi
}

is_git_repo() {
    git -C "${SCRIPT_DIR}" rev-parse --is-inside-work-tree &>/dev/null
}

collect_files() {
    local mode="$1"
    shift
    local files=()

    case "${mode}" in
        --all)
            if is_git_repo; then
                while IFS= read -r f; do
                    files+=("${SCRIPT_DIR}/${f}")
                done < <(git -C "${SCRIPT_DIR}" ls-files || true)
            else
                while IFS= read -r f; do
                    files+=("${f}")
                done < <(find "${SCRIPT_DIR}" -type f || true)
            fi
            ;;
        --modified)
            if ! is_git_repo; then
                error "--modified requires a git repository"
                exit 1
            fi
            while IFS= read -r f; do
                files+=("${SCRIPT_DIR}/${f}")
            done < <(git -C "${SCRIPT_DIR}" diff --cached --name-only --diff-filter=ACM || true)
            while IFS= read -r f; do
                files+=("${SCRIPT_DIR}/${f}")
            done < <(git -C "${SCRIPT_DIR}" diff --name-only --diff-filter=ACM || true)
            while IFS= read -r f; do
                files+=("${SCRIPT_DIR}/${f}")
            done < <(git -C "${SCRIPT_DIR}" ls-files --others --exclude-standard || true)
            ;;
        --files)
            for f in "$@"; do
                if [[ -f "${f}" ]]; then
                    local abs_path
                    abs_path="$(cd "$(dirname "${f}")" && pwd)/$(basename "${f}")"
                    files+=("${abs_path}")
                else
                    warn "WARNING: file not found, skipping: ${f}" >&2
                fi
            done
            ;;
    esac

    local unique_files=()
    local seen=""
    for f in "${files[@]}"; do
        if [[ ! " ${seen} " =~ " ${f} " ]]; then
            unique_files+=("${f}")
            seen="${seen} ${f}"
        fi
    done

    printf '%s\n' "${unique_files[@]}"
}

filter_cxx_files() {
    local files=("$@")
    local cxx_files=()
    for f in "${files[@]}"; do
        if echo "${f}" | grep -qE "${CXX_PATTERN}"; then
            cxx_files+=("${f}")
        fi
    done
    printf '%s\n' "${cxx_files[@]}"
}

run_clang_format() {
    local files=("$@")
    if [[ ${#files[@]} -eq 0 ]]; then
        return 0
    fi

    info "Running clang-format on ${#files[@]} file(s)..."

    declare -A md5_before
    for f in "${files[@]}"; do
        md5_before["${f}"]="$(md5sum "${f}" | awk '{print $1}')"
    done

    local pc_output
    pc_output="$(pre-commit run clang-format --files "${files[@]}" 2>&1)" || true

    CLANG_FORMAT_MODIFIED=()
    for f in "${files[@]}"; do
        local md5_after
        md5_after="$(md5sum "${f}" | awk '{print $1}')"
        if [[ "${md5_before[${f}]}" != "${md5_after}" ]]; then
            CLANG_FORMAT_MODIFIED+=("${f}")
        fi
    done

    if [[ ${#CLANG_FORMAT_MODIFIED[@]} -gt 0 ]]; then
        return 1
    fi
    return 0
}

run_oat_check() {
    local files=("$@")
    if [[ ${#files[@]} -eq 0 ]]; then
        return 0
    fi

    info "Running oat-check on ${#files[@]} file(s)..."

    if [[ -z "${OAT_PY}" ]]; then
        warn "  skipped (OAT binary not available)"
        return 0
    fi

    OAT_REPORT_DIR="$(mktemp -d /tmp/oat_check_XXXXXX)"

    local rel_files=()
    for f in "${files[@]}"; do
        local rel
        rel="$(realpath --relative-to="${SCRIPT_DIR}" "${f}" 2>/dev/null || echo "${f}")"
        rel_files+=("${rel}")
    done

    local file_list
    file_list="$(IFS=,; echo "${rel_files[*]}")"

    local oat_output
    oat_output="$("${OAT_PY}" -m oat \
        -mode s \
        -s "${SCRIPT_DIR}" \
        -r "${OAT_REPORT_DIR}" \
        -n "CheckerL2" \
        -w 1 \
        -f "${file_list}" \
        2>&1)" || true

    OAT_FAILED_FILES=()
    OAT_FAILED_REASONS=()

    local report_file
    report_file="$(find "${OAT_REPORT_DIR}" -name "PlainReport_*.txt" -type f 2>/dev/null | head -1)"

    if [[ -n "${report_file}" && -f "${report_file}" ]]; then
        while IFS= read -r line; do
            local issue_name=""
            local issue_file=""
            local issue_content=""
            if [[ "${line}" =~ Name:${TAB}([^${TAB}]+)${TAB} ]]; then
                issue_name="${BASH_REMATCH[1]}"
            fi
            if [[ "${line}" =~ Content:${TAB}([^${TAB}]+)${TAB} ]]; then
                issue_content="${BASH_REMATCH[1]}"
            fi
            if [[ "${line}" =~ File:${TAB}([^${TAB}]+) ]]; then
                issue_file="${BASH_REMATCH[1]}"
            fi
            if [[ -n "${issue_file}" && "${issue_name}" == *"Invalid"* ]]; then
                OAT_FAILED_FILES+=("${issue_file}")
                OAT_FAILED_REASONS+=("${issue_name} (${issue_content:-unknown})")
            fi
        done < <(grep -E "^Name:.*Invalid" "${report_file}" 2>/dev/null || true)
    fi

    if [[ ${#OAT_FAILED_FILES[@]} -gt 0 ]]; then
        return 1
    fi
    return 0
}

main() {
    if [[ $# -eq 0 ]]; then
        usage
    fi

    local mode=""
    local user_files=()

    case "$1" in
        --all|-all)       mode="--all" ;;
        --modified|-modified) mode="--modified" ;;
        -h|--help)  usage ;;
        *)          mode="--files"; user_files=("$@") ;;
    esac

    check_deps
    cd "${SCRIPT_DIR}"

    local files_raw
    files_raw="$(collect_files "${mode}" "${user_files[@]+"${user_files[@]}"}")"

    if [[ -z "${files_raw}" ]]; then
        echo "No files found to check."
        exit 0
    fi

    local files=()
    while IFS= read -r line; do
        [[ -n "${line}" ]] && files+=("${line}")
    done <<< "${files_raw}"

    # Filter C/C++ files for clang-format
    local cxx_files_raw
    cxx_files_raw="$(filter_cxx_files "${files[@]}")"
    local cxx_files=()
    if [[ -n "${cxx_files_raw}" ]]; then
        while IFS= read -r line; do
            [[ -n "${line}" ]] && cxx_files+=("${line}")
        done <<< "${cxx_files_raw}"
    fi

    echo ""
    info "Found ${#files[@]} file(s) to check (${#cxx_files[@]} C/C++ for clang-format, ${#files[@]} for oat-check)"
    echo ""

    local exit_code=0
    CLANG_FORMAT_MODIFIED=()
    OAT_FAILED_FILES=()
    OAT_FAILED_REASONS=()

    if [[ ${#cxx_files[@]} -gt 0 ]]; then
        if ! run_clang_format "${cxx_files[@]}"; then
            exit_code=1
        fi
    else
        info "No C/C++ files found, skipping clang-format."
    fi

    if ! run_oat_check "${files[@]}"; then
        exit_code=1
    fi

    echo ""
    echo "======================================"
    if [[ ${exit_code} -eq 0 ]]; then
        success "PASS: all ${#files[@]} file(s) passed clang-format and oat-check"
    else
        local total_issues=0
        local issues_detail=""

        if [[ ${#CLANG_FORMAT_MODIFIED[@]} -gt 0 ]]; then
            total_issues=$((total_issues + ${#CLANG_FORMAT_MODIFIED[@]}))
            for f in "${CLANG_FORMAT_MODIFIED[@]}"; do
                local rel
                rel="$(realpath --relative-to="${SCRIPT_DIR}" "${f}" 2>/dev/null || basename "${f}")"
                issues_detail+="  ${RED}✗${NC} ${rel}\n"
                issues_detail+="    - clang-format: auto-formatted (run 'git add ${rel}' and retry)\n"
            done
        fi

        if [[ ${#OAT_FAILED_FILES[@]} -gt 0 ]]; then
            total_issues=$((total_issues + ${#OAT_FAILED_FILES[@]}))
            for i in "${!OAT_FAILED_FILES[@]}"; do
                local rel
                rel="$(realpath --relative-to="${SCRIPT_DIR}" "${OAT_FAILED_FILES[$i]}" 2>/dev/null || basename "${OAT_FAILED_FILES[$i]}")"
                local reason="${OAT_FAILED_REASONS[$i]:-compliance issue}"
                issues_detail+="  ${RED}✗${NC} ${rel}\n"
                issues_detail+="    - oat-check: ${reason}\n"
            done
        fi

        error "FAILED: ${total_issues} issue(s) found"
        echo ""
        echo -e "${issues_detail}"
    fi
    echo "======================================"

    exit ${exit_code}
}

main "$@"
