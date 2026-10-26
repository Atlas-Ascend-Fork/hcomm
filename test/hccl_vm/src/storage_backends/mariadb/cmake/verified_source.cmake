# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# See the License in the root of the software repository for the full text of the License.

# 公共 verified-source 基础设施：第三方源码归档的下载、SHA-256 校验、内容
# 指纹复核与解包重建。各依赖（mariadb 等）的 .cmake 文件必须在本文件之后
# include，并携带各自的版本/来源/哈希与目标定义。

include_guard(GLOBAL)

# Build a stable, content-only manifest. The extraction stamp itself is excluded so
# that a changed stamp cannot bless a changed source tree.
function(hcclvm_compute_source_manifest source_dir stamp_name output_variable)
    if(NOT IS_DIRECTORY "${source_dir}")
        message(FATAL_ERROR "[ThirdParty] Cannot manifest missing source directory: ${source_dir}")
    endif()

    file(GLOB_RECURSE _manifest_entries
        LIST_DIRECTORIES FALSE
        RELATIVE "${source_dir}"
        "${source_dir}/*"
    )
    list(SORT _manifest_entries)
    set(_manifest_payload "")
    foreach(_relative_path IN LISTS _manifest_entries)
        if("${_relative_path}" STREQUAL "${stamp_name}")
            continue()
        endif()
        set(_absolute_path "${source_dir}/${_relative_path}")
        if(IS_SYMLINK "${_absolute_path}")
            file(READ_SYMLINK "${_absolute_path}" _link_target)
            string(APPEND _manifest_payload "L\t${_relative_path}\t${_link_target}\n")
        else()
            file(SHA256 "${_absolute_path}" _entry_sha256)
            string(APPEND _manifest_payload "F\t${_relative_path}\t${_entry_sha256}\n")
        endif()
    endforeach()
    string(SHA256 _manifest_sha256 "${_manifest_payload}")
    set(${output_variable} "${_manifest_sha256}" PARENT_SCOPE)
endfunction()

# Validate the archive on every configure. A missing/corrupt archive, a changed
# extracted file, or a changed/missing identity stamp rebuilds only source_dir from
# the newly verified archive.
function(hcclvm_prepare_verified_source
    dependency_name version provenance_url download_url expected_sha256
    expected_manifest_sha256 archive_path source_dir stamp_name)
    get_filename_component(_archive_directory "${archive_path}" DIRECTORY)
    file(MAKE_DIRECTORY "${_archive_directory}")
    set(_archive_repaired FALSE)

    if(EXISTS "${archive_path}" OR IS_SYMLINK "${archive_path}")
        if(IS_DIRECTORY "${archive_path}" OR IS_SYMLINK "${archive_path}")
            message(WARNING
                "[ThirdParty] ${dependency_name} cached archive is not a regular file; "
                "discarding only ${archive_path}")
            file(REMOVE_RECURSE "${archive_path}")
            set(_archive_repaired TRUE)
        else()
            file(SHA256 "${archive_path}" _archive_actual_sha256)
            if(NOT "${_archive_actual_sha256}" STREQUAL "${expected_sha256}")
                message(WARNING
                    "[ThirdParty] ${dependency_name} cached archive SHA-256 mismatch; "
                    "discarding only ${archive_path}")
                file(REMOVE "${archive_path}")
                set(_archive_repaired TRUE)
            endif()
        endif()
    endif()

    if(NOT EXISTS "${archive_path}")
        set(_download_path "${archive_path}.download")
        file(REMOVE_RECURSE "${_download_path}")
        message(STATUS "[ThirdParty] ${dependency_name} provenance: ${provenance_url}")
        message(STATUS "[ThirdParty] Downloading ${dependency_name} from ${download_url}")
        file(DOWNLOAD
            "${download_url}"
            "${_download_path}"
            STATUS _download_status
            TLS_VERIFY OFF
            TIMEOUT 120
        )
        list(GET _download_status 0 _download_code)
        if(NOT _download_code EQUAL 0)
            file(REMOVE_RECURSE "${_download_path}")
            message(FATAL_ERROR
                "[ThirdParty] Failed to download verified ${dependency_name} archive: ${_download_status}\n"
                "Provenance: ${provenance_url}\nTransport: ${download_url}\n"
                "Expected SHA-256: ${expected_sha256}")
        endif()
        file(RENAME "${_download_path}" "${archive_path}")
        set(_archive_repaired TRUE)
    endif()

    # This check is intentionally unconditional: an old cache is never trusted by
    # existence alone, even when the extracted source and stamp look valid.
    if(NOT EXISTS "${archive_path}" OR
        IS_DIRECTORY "${archive_path}" OR IS_SYMLINK "${archive_path}")
        message(FATAL_ERROR
            "[ThirdParty] ${dependency_name} archive repair did not produce a regular file")
    endif()
    file(SHA256 "${archive_path}" _archive_actual_sha256)
    if(NOT "${_archive_actual_sha256}" STREQUAL "${expected_sha256}")
        message(FATAL_ERROR
            "[ThirdParty] ${dependency_name} archive failed post-download SHA-256 validation")
    endif()

    set(_expected_stamp
        "dependency=${dependency_name}\nversion=${version}\narchive_sha256=${expected_sha256}\nsource_manifest_sha256=${expected_manifest_sha256}\n")
    set(_stamp_path "${source_dir}/${stamp_name}")
    set(_source_rebuild "${_archive_repaired}")

    if(IS_DIRECTORY "${source_dir}")
        set(_stamp_matches FALSE)
        if(EXISTS "${_stamp_path}" OR IS_SYMLINK "${_stamp_path}")
            if(IS_DIRECTORY "${_stamp_path}" OR IS_SYMLINK "${_stamp_path}")
                message(WARNING
                    "[ThirdParty] ${dependency_name} source stamp is not a regular file; "
                    "rebuilding only ${source_dir}")
                set(_source_rebuild TRUE)
            else()
                file(READ "${_stamp_path}" _actual_stamp)
                if("${_actual_stamp}" STREQUAL "${_expected_stamp}")
                    set(_stamp_matches TRUE)
                endif()
            endif()
        endif()
        hcclvm_compute_source_manifest("${source_dir}" "${stamp_name}" _actual_manifest_sha256)
        if(NOT _stamp_matches OR
            NOT "${_actual_manifest_sha256}" STREQUAL "${expected_manifest_sha256}")
            set(_source_rebuild TRUE)
        endif()
    else()
        set(_source_rebuild TRUE)
    endif()

    if(_source_rebuild)
        message(STATUS "[ThirdParty] Rebuilding verified ${dependency_name} source cache")
        file(REMOVE_RECURSE "${source_dir}")
        set(_extract_directory "${_archive_directory}/.${dependency_name}-${version}-extract")
        file(REMOVE_RECURSE "${_extract_directory}")
        file(MAKE_DIRECTORY "${_extract_directory}")
        file(ARCHIVE_EXTRACT INPUT "${archive_path}" DESTINATION "${_extract_directory}")
        file(GLOB _extracted_roots LIST_DIRECTORIES TRUE "${_extract_directory}/*")
        list(LENGTH _extracted_roots _root_count)
        if(NOT _root_count EQUAL 1)
            file(REMOVE_RECURSE "${_extract_directory}")
            message(FATAL_ERROR
                "[ThirdParty] ${dependency_name} archive must contain exactly one source root")
        endif()
        list(GET _extracted_roots 0 _extracted_root)
        if(NOT IS_DIRECTORY "${_extracted_root}")
            file(REMOVE_RECURSE "${_extract_directory}")
            message(FATAL_ERROR
                "[ThirdParty] ${dependency_name} archive root is not a directory")
        endif()
        file(RENAME "${_extracted_root}" "${source_dir}")
        file(REMOVE_RECURSE "${_extract_directory}")

        hcclvm_compute_source_manifest("${source_dir}" "${stamp_name}" _rebuilt_manifest_sha256)
        if(NOT "${_rebuilt_manifest_sha256}" STREQUAL "${expected_manifest_sha256}")
            file(REMOVE_RECURSE "${source_dir}")
            message(FATAL_ERROR
                "[ThirdParty] ${dependency_name} extracted source manifest mismatch.\n"
                "Expected: ${expected_manifest_sha256}\nActual: ${_rebuilt_manifest_sha256}")
        endif()
        file(WRITE "${_stamp_path}" "${_expected_stamp}")
    else()
        message(STATUS "[ThirdParty] Verified ${dependency_name} source cache: ${source_dir}")
    endif()
endfunction()

# Script-mode mutation tests include only the production cache functions above.
if(HCCLVM_SOURCE_CACHE_HELPER_ONLY)
    return()
endif()
