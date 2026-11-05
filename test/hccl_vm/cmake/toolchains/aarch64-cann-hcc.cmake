# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

if(NOT DEFINED CANN_HCC_ROOT OR CANN_HCC_ROOT STREQUAL "")
    message(FATAL_ERROR "CANN_HCC_ROOT is not set")
endif()
set(CANN_HCC_ROOT "${CANN_HCC_ROOT}" CACHE PATH "CANN HCC toolchain root" FORCE)
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES CANN_HCC_ROOT)

set(_cann_hcc_target aarch64-target-linux-gnu)
set(_cann_hcc_bin "${CANN_HCC_ROOT}/bin")
set(_cann_hcc_sysroot "${CANN_HCC_ROOT}/sysroot")

include("${CMAKE_CURRENT_LIST_DIR}/../cann_hcc_validation.cmake")
hcclvm_validate_cann_hcc("${CANN_HCC_ROOT}" _cann_hcc_is_complete _cann_hcc_missing_path)
if(NOT _cann_hcc_is_complete)
    message(FATAL_ERROR "Incomplete CANN HCC toolchain: ${_cann_hcc_missing_path} was not found")
endif()

set(CMAKE_C_COMPILER "${_cann_hcc_bin}/${_cann_hcc_target}-gcc" CACHE FILEPATH "CANN HCC C compiler" FORCE)
set(CMAKE_CXX_COMPILER "${_cann_hcc_bin}/${_cann_hcc_target}-g++" CACHE FILEPATH "CANN HCC C++ compiler" FORCE)
set(CMAKE_LINKER "${_cann_hcc_bin}/${_cann_hcc_target}-ld" CACHE FILEPATH "CANN HCC linker" FORCE)
set(CMAKE_AR "${_cann_hcc_bin}/${_cann_hcc_target}-ar" CACHE FILEPATH "CANN HCC archiver" FORCE)
set(CMAKE_RANLIB "${_cann_hcc_bin}/${_cann_hcc_target}-ranlib" CACHE FILEPATH "CANN HCC ranlib" FORCE)
set(CMAKE_STRIP "${_cann_hcc_bin}/${_cann_hcc_target}-strip" CACHE FILEPATH "CANN HCC strip" FORCE)
set(CMAKE_LD "${_cann_hcc_bin}/${_cann_hcc_target}-ld" CACHE FILEPATH "CANN HCC ld" FORCE)
set(CMAKE_NM "${_cann_hcc_bin}/${_cann_hcc_target}-nm" CACHE FILEPATH "CANN HCC nm" FORCE)
set(CMAKE_OBJCOPY "${_cann_hcc_bin}/${_cann_hcc_target}-objcopy" CACHE FILEPATH "CANN HCC objcopy" FORCE)

# hcc 编译器以绝对路径调用, PATH 前置拦截不到, 经 launcher 接入 ccache
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "ccache launcher" FORCE)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "ccache launcher" FORCE)
endif()

set(CMAKE_SYSROOT "${_cann_hcc_sysroot}" CACHE PATH "CANN HCC sysroot" FORCE)
set(CMAKE_FIND_ROOT_PATH "${_cann_hcc_sysroot}" CACHE PATH "CANN HCC target search root" FORCE)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

set(CMAKE_SKIP_RPATH TRUE)
