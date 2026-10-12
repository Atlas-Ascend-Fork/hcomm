# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# See the License in the specific file packaged with this Software for details.

# SQLite 后端模块的客户库依赖配方（Host 架构）。
# 源码缓存归属模块自身（src/storage_backends/sqlite/third_party/，git 忽略）；
# device 子构建经 HCCLVM_SQLITE3_SOURCE_DIR 交接变量复用同一份已解压源码
#（二进制输出各自独立，见 cmake/sqlite3_device.cmake）。
include_guard(GLOBAL)

set(SQLITE3_VERSION_NUM "3510300")
set(SQLITE3_FILE "sqlite-amalgamation-${SQLITE3_VERSION_NUM}.zip")
set(SQLITE3_URL "https://www.sqlite.org/2026/${SQLITE3_FILE}")
set(SQLITE3_MODULE_THIRD_PARTY_DIR "${CMAKE_CURRENT_LIST_DIR}/../third_party")
set(SQLITE3_PKG_PATH ${SQLITE3_MODULE_THIRD_PARTY_DIR}/${SQLITE3_FILE})
set(SQLITE3_TARGET_DIR ${SQLITE3_MODULE_THIRD_PARTY_DIR}/sqlite)
# 交接变量：device 子构建（ExternalProject 参数）由此取得同一份源码缓存
set(HCCLVM_SQLITE3_SOURCE_DIR "${SQLITE3_TARGET_DIR}")

message(STATUS "[StorageBackends:sqlite] SQLITE3_VERSION_NUM=${SQLITE3_VERSION_NUM}")
message(STATUS "[StorageBackends:sqlite] SQLITE3_URL=${SQLITE3_URL}")
message(STATUS "[StorageBackends:sqlite] SQLITE3_TARGET_DIR=${SQLITE3_TARGET_DIR}")

# 缓存命中：源码目录已存在，跳过下载/解压；否则下载并解压
if(NOT (EXISTS "${SQLITE3_TARGET_DIR}/sqlite3.c" AND EXISTS "${SQLITE3_TARGET_DIR}/sqlite3.h"))
    # 获取 zip：优先用本地缓存，否则在线下载
    if(NOT EXISTS "${SQLITE3_PKG_PATH}")
        message(STATUS "[StorageBackends:sqlite] Downloading SQLite3 from ${SQLITE3_URL}")
        file(DOWNLOAD
            ${SQLITE3_URL}
            ${SQLITE3_PKG_PATH}
            TLS_VERIFY OFF
            STATUS _dl_status
            TIMEOUT 600
        )
        list(GET _dl_status 0 _dl_code)
        if(NOT _dl_code EQUAL 0)
            message(FATAL_ERROR
                "[StorageBackends:sqlite] Failed to download SQLite3 archive (${_dl_status}).\n"
                "URL: ${SQLITE3_URL}\n"
                "You can manually download and place the zip at: ${SQLITE3_PKG_PATH}")
        endif()
        message(STATUS "[StorageBackends:sqlite] SQLite3 archive saved to ${SQLITE3_PKG_PATH}")
    else()
        message(STATUS "[StorageBackends:sqlite] Found local SQLite3 package: ${SQLITE3_PKG_PATH}")
    endif()

    # 解压
    set(_sqlite3_extract_dir "${CMAKE_BINARY_DIR}/_sqlite3_tmp")
    file(MAKE_DIRECTORY ${_sqlite3_extract_dir})
    file(ARCHIVE_EXTRACT
        INPUT ${SQLITE3_PKG_PATH}
        DESTINATION ${_sqlite3_extract_dir}
    )
    file(GLOB _sqlite3_extracted_entries "${_sqlite3_extract_dir}/*")
    list(GET _sqlite3_extracted_entries 0 _extracted_dir)

    # 规范化目录名：sqlite-amalgamation-3510300 -> sqlite
    if(EXISTS "${SQLITE3_TARGET_DIR}")
        file(REMOVE_RECURSE ${SQLITE3_TARGET_DIR})
    endif()
    file(RENAME ${_extracted_dir} ${SQLITE3_TARGET_DIR})
    file(REMOVE_RECURSE ${_sqlite3_extract_dir})

    # 验证产物
    if(NOT EXISTS "${SQLITE3_TARGET_DIR}/sqlite3.c" OR NOT EXISTS "${SQLITE3_TARGET_DIR}/sqlite3.h")
        message(FATAL_ERROR
            "[StorageBackends:sqlite] SQLite3 source extraction failed. "
            "Missing sqlite3.c or sqlite3.h in ${SQLITE3_TARGET_DIR}")
    endif()
    message(STATUS "[StorageBackends:sqlite] SQLite3 source extracted to ${SQLITE3_TARGET_DIR}")
else()
    message(STATUS "[StorageBackends:sqlite] SQLite3 source already available in ${SQLITE3_TARGET_DIR}")
endif()

# 源码就绪后再创建 target，避免引用尚未下载的源文件
if(NOT TARGET sqlite3)
    add_library(sqlite3 STATIC ${SQLITE3_TARGET_DIR}/sqlite3.c)
    target_include_directories(sqlite3 PUBLIC ${SQLITE3_TARGET_DIR})
    set_target_properties(sqlite3 PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_STANDARD 17
    )
    target_compile_options(sqlite3 PRIVATE
        -O2
        -DSQLITE_THREADSAFE=1
        -DSQLITE_OMIT_LOAD_EXTENSION
        -DSQLITE_DEFAULT_MEMSTATUS=0
        -DSQLITE_DEFAULT_WAL_SYNCHRONOUS=0
        -DSQLITE_LIKE_DOESNT_MATCH_BLOBS
        -DSQLITE_OMIT_DEPRECATED
    )
endif()
