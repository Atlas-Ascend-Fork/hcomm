# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 公共归档解压工具：兼容 CMake < 3.18（file(ARCHIVE_EXTRACT) 自 3.18 引入）。
# 基于 cmake -E tar（CMake 内置 libarchive，自动识别 zip/tar.gz），零外部依赖。
function(hcclvm_archive_extract)
    cmake_parse_arguments(_ae "" "ARCHIVE;DESTINATION" "" ${ARGN})
    if(NOT _ae_ARCHIVE)
        message(FATAL_ERROR "[ThirdParty] hcclvm_archive_extract requires ARCHIVE")
    endif()
    if(NOT _ae_DESTINATION)
        message(FATAL_ERROR "[ThirdParty] hcclvm_archive_extract requires DESTINATION")
    endif()
    get_filename_component(_archive_abs "${_ae_ARCHIVE}" ABSOLUTE)
    if(NOT EXISTS "${_archive_abs}")
        message(FATAL_ERROR "[ThirdParty] archive not found: ${_archive_abs}")
    endif()
    file(MAKE_DIRECTORY "${_ae_DESTINATION}")

    if("${_archive_abs}" MATCHES "\\.zip$")
        set(_tar_flags xf)
    else()
        set(_tar_flags xzf)
    endif()

    execute_process(
        COMMAND ${CMAKE_COMMAND} -E chdir "${_ae_DESTINATION}"
                ${CMAKE_COMMAND} -E tar ${_tar_flags} "${_archive_abs}"
        RESULT_VARIABLE _extract_rc
    )
    if(NOT _extract_rc EQUAL 0)
        message(FATAL_ERROR
            "[ThirdParty] Failed to extract archive '${_archive_abs}' into "
            "'${_ae_DESTINATION}' (cmake -E tar exited with ${_extract_rc}).")
    endif()
endfunction()
