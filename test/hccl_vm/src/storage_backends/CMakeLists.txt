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

# 文件职责：存储后端模块的通用发现、记录与接入规则（收集器）。
# 只枚举本目录的一级子目录（按名称排序）：每个子目录都是一个后端模块，必须
# 自带 CMakeLists.txt 并声明与目录一致的注册名；损坏/不完整模块明确 configure
# 失败，不静默跳过；一个模块都没有同样失败。本文件不列举、也不依赖任何具体
# 后端名——后端的源码、依赖、配置与测试全部由模块目录自述。
#
# 【调用序列】（由各构建形态按序调用；重复 include 本文件由函数定义幂等保证）
#   hcclvm_storage_backends_prepare()     发现 + 模块依赖准备 + device 透传参数收集
#                                         （顶层 configure 早期，Host/Device 两种形态都调用）
#   hcclvm_storage_backends_device_args(OUT <var>)
#                                         取出 device ExternalProject 的透传参数列表
#   hcclvm_storage_backends_attach(TARGET <引擎接入目标> COMPOSITION_TARGET <装配接入目标>)
#                                         engine 源码/依赖接入 TARGET，composition 源码与
#                                         生成注册清单接入 COMPOSITION_TARGET（Host 形态）
#   hcclvm_storage_backends_attach_device(TARGET <device storage 目标>)
#                                         device 子工程内把模块全部源码/依赖与生成注册
#                                         清单接入同一目标（Device 形态）
#   hcclvm_storage_backends_add_test_dir(<test 目录>)
#                                         test 树内按模块名接入引擎专属测试子目录
#                                         （test/<领域>/<模块名>/；模块不存在则跳过）
#
# 【模块 CMakeLists.txt 合同】被收集器按阶段变量 HCCLVM_STORAGE_BACKEND_PHASE
# include（PREPARE/ATTACH/DEVICE），模块自述：
#   set(HCCLVM_STORAGE_BACKEND_NAME <注册名>)          # 必须等于目录名
#   set(HCCLVM_STORAGE_BACKEND_DEFAULT_CANDIDATE ON)   # 可选：无配置默认候选（至多一个）
#   set(HCCLVM_STORAGE_BACKEND_ENTRY_HEADER <包含路径>) # 生成注册清单 include 的模块入口头
#   set(HCCLVM_STORAGE_BACKEND_ENTRY_SYMBOL <符号>)     # 入口头里的模块单例函数符号
#   PREPARE 阶段：include 依赖配方；list(APPEND HCCLVM_STORAGE_BACKEND_DEVICE_ARGS "-D<VAR>=<值>")
#   ATTACH 阶段：target_sources(${HCCLVM_STORAGE_BACKEND_ATTACH_TARGET} ...) 等
#   DEVICE 阶段：target_sources(${HCCLVM_STORAGE_BACKEND_DEVICE_TARGET} ...) 等
# 引擎专属测试不驻留模块目录：测试文件在 test/ 树内按引擎分子目录，由
# hcclvm_storage_backends_add_test_dir 按模块存在性接入。

# 收集器根目录（include 时捕获；函数内 CMAKE_CURRENT_LIST_DIR 是调用方目录，不可用）
set_property(GLOBAL PROPERTY HCCLVM_STORAGE_BACKENDS_ROOT "${CMAKE_CURRENT_LIST_DIR}")

# ---- 模块发现（排序、防重复、防损坏） ----
function(hcclvm_storage_backend_modules out_var)
    get_property(_backend_root GLOBAL PROPERTY HCCLVM_STORAGE_BACKENDS_ROOT)
    # 注意：glob pattern 必须是绝对路径——相对 pattern 按调用方 CWD 解析，
    # 结果与本目录无关。
    file(GLOB _backend_entries LIST_DIRECTORIES TRUE RELATIVE "${_backend_root}" "${_backend_root}/*")
    set(_backend_modules "")
    foreach(_entry IN LISTS _backend_entries)
        if(_entry MATCHES "^\\.")
            continue() # 版本管理目录等隐藏项不是模块
        endif()
        if(NOT IS_DIRECTORY "${_backend_root}/${_entry}")
            continue() # 一级文件（说明文档等）不是模块
        endif()
        if(NOT EXISTS "${_backend_root}/${_entry}/CMakeLists.txt")
            message(FATAL_ERROR
                "incomplete storage backend module (missing CMakeLists.txt): ${_backend_root}/${_entry}")
        endif()
        list(APPEND _backend_modules "${_entry}")
    endforeach()
    list(SORT _backend_modules)
    if(NOT _backend_modules)
        message(FATAL_ERROR
            "no storage backend module found under ${_backend_root}: "
            "at least one backend module directory is required")
    endif()
    set(${out_var} "${_backend_modules}" PARENT_SCOPE)
endfunction()

# ---- 单阶段执行：逐模块 include 并登记元数据 ----
function(_hcclvm_storage_backends_run_phase phase)
    get_property(_backends_root GLOBAL PROPERTY HCCLVM_STORAGE_BACKENDS_ROOT)
    hcclvm_storage_backend_modules(_modules)
    foreach(_module IN LISTS _modules)
        set(HCCLVM_STORAGE_BACKEND_PHASE "${phase}")
        set(HCCLVM_STORAGE_BACKEND_DIR "${_backends_root}/${_module}")
        unset(HCCLVM_STORAGE_BACKEND_NAME)
        unset(HCCLVM_STORAGE_BACKEND_DEFAULT_CANDIDATE)
        unset(HCCLVM_STORAGE_BACKEND_ENTRY_HEADER)
        unset(HCCLVM_STORAGE_BACKEND_ENTRY_SYMBOL)
        unset(HCCLVM_STORAGE_BACKEND_DEVICE_ARGS)
        include("${HCCLVM_STORAGE_BACKEND_DIR}/CMakeLists.txt")
        if(NOT DEFINED HCCLVM_STORAGE_BACKEND_NAME)
            message(FATAL_ERROR
                "storage backend module ${_module} did not declare HCCLVM_STORAGE_BACKEND_NAME")
        endif()
        if(NOT "${HCCLVM_STORAGE_BACKEND_NAME}" STREQUAL "${_module}")
            message(FATAL_ERROR
                "storage backend module name mismatch: directory ${_module} declares "
                "${HCCLVM_STORAGE_BACKEND_NAME}")
        endif()
        if(NOT DEFINED HCCLVM_STORAGE_BACKEND_ENTRY_HEADER OR
            NOT DEFINED HCCLVM_STORAGE_BACKEND_ENTRY_SYMBOL)
            message(FATAL_ERROR
                "storage backend module ${_module} did not declare its entry header/symbol")
        endif()
        # 模块元数据登记（同名幂等覆盖；名称清单去重后即为编入模块集合）
        set_property(GLOBAL APPEND PROPERTY HCCLVM_STORAGE_BACKEND_MODULE_NAMES
            "${HCCLVM_STORAGE_BACKEND_NAME}")
        set_property(GLOBAL PROPERTY
            "HCCLVM_STORAGE_BACKEND_${HCCLVM_STORAGE_BACKEND_NAME}_ENTRY_HEADER"
            "${HCCLVM_STORAGE_BACKEND_ENTRY_HEADER}")
        set_property(GLOBAL PROPERTY
            "HCCLVM_STORAGE_BACKEND_${HCCLVM_STORAGE_BACKEND_NAME}_ENTRY_SYMBOL"
            "${HCCLVM_STORAGE_BACKEND_ENTRY_SYMBOL}")
        if(HCCLVM_STORAGE_BACKEND_DEFAULT_CANDIDATE)
            set_property(GLOBAL APPEND PROPERTY HCCLVM_STORAGE_BACKEND_DEFAULT_CANDIDATES
                "${HCCLVM_STORAGE_BACKEND_NAME}")
        endif()
        # device 透传参数（PREPARE 阶段模块追加；Host 形态下收集但不用）
        if(DEFINED HCCLVM_STORAGE_BACKEND_DEVICE_ARGS)
            set_property(GLOBAL APPEND PROPERTY HCCLVM_STORAGE_BACKEND_DEVICE_ARGS_ALL
                ${HCCLVM_STORAGE_BACKEND_DEVICE_ARGS})
        endif()
    endforeach()
    set_property(GLOBAL PROPERTY HCCLVM_STORAGE_BACKEND_MODULE_NAMES
        "${_modules}") # 发现序（已排序、目录名天然唯一）为准
endfunction()

# ---- 阶段一：发现 + 依赖准备（顶层 configure 早期） ----
function(hcclvm_storage_backends_prepare)
    _hcclvm_storage_backends_run_phase(PREPARE)
    get_property(_defaults GLOBAL PROPERTY HCCLVM_STORAGE_BACKEND_DEFAULT_CANDIDATES)
    list(REMOVE_DUPLICATES _defaults)
    list(LENGTH _defaults _default_count)
    if(_default_count GREATER 1)
        message(FATAL_ERROR
            "multiple default storage backend candidates are compiled in: ${_defaults} "
            "(at most one module may declare HCCLVM_STORAGE_BACKEND_DEFAULT_CANDIDATE)")
    endif()
endfunction()

# ---- device ExternalProject 透传参数 ----
function(hcclvm_storage_backends_device_args out_var)
    get_property(_args GLOBAL PROPERTY HCCLVM_STORAGE_BACKEND_DEVICE_ARGS_ALL)
    set(${out_var} "${_args}" PARENT_SCOPE)
endfunction()

# ---- 生成注册清单（仅存在于 build 目录；显式引用保证静态库模块被实际链接） ----
function(_hcclvm_storage_backends_generate_registration composition_target)
    get_property(_names GLOBAL PROPERTY HCCLVM_STORAGE_BACKEND_MODULE_NAMES)
    if(NOT _names)
        message(FATAL_ERROR "no storage backend module registered for the generated backend list")
    endif()
    set(_generated_dir "${CMAKE_BINARY_DIR}/generated/storage_composition")
    file(MAKE_DIRECTORY "${_generated_dir}")

    set(_header_content
"// 本文件由 src/storage_backends/CMakeLists.txt 收集器生成（仅存在于 build 目录），
// 请勿手工修改或提交。声明已编入后端模块清单的访问入口；公共 composition 经
// 此生成入口发现模块，不出现具体后端名。
#ifndef HCCL_VM_SIMULATION_STORAGE_COMPOSITION_COMPILED_BACKENDS_H
#define HCCL_VM_SIMULATION_STORAGE_COMPOSITION_COMPILED_BACKENDS_H

#include <cstddef>

#include \"storage_composition/backend_module.h\"

namespace sim::backends {
/// 已编入后端模块清单（按模块目录名排序；元素指向各模块的静态单例）
const BackendModule* const* CompiledBackendModules();
/// 已编入后端模块数量
std::size_t CompiledBackendCount();
} // namespace sim::backends

#endif // HCCL_VM_SIMULATION_STORAGE_COMPOSITION_COMPILED_BACKENDS_H
")
    file(WRITE "${_generated_dir}/compiled_backends.h" "${_header_content}")

    set(_source_content
"// 本文件由 src/storage_backends/CMakeLists.txt 收集器生成（仅存在于 build 目录），
// 请勿手工修改或提交。以显式调用接入各模块入口，禁止依赖全局构造器自动注册。
#include <cstddef>

#include \"storage_composition/compiled_backends.h\"
")
    foreach(_name IN LISTS _names)
        get_property(_entry_header GLOBAL PROPERTY "HCCLVM_STORAGE_BACKEND_${_name}_ENTRY_HEADER")
        get_property(_entry_symbol GLOBAL PROPERTY "HCCLVM_STORAGE_BACKEND_${_name}_ENTRY_SYMBOL")
        string(APPEND _source_content "#include \"${_entry_header}\"\n")
    endforeach()
    string(APPEND _source_content
"\nnamespace {
const sim::backends::BackendModule* const kCompiledBackendModules[] = {
")
    foreach(_name IN LISTS _names)
        get_property(_entry_symbol GLOBAL PROPERTY "HCCLVM_STORAGE_BACKEND_${_name}_ENTRY_SYMBOL")
        string(APPEND _source_content "    &${_entry_symbol}(),\n")
    endforeach()
    string(APPEND _source_content
"};
} // namespace

namespace sim::backends {
const BackendModule* const* CompiledBackendModules()
{
    return kCompiledBackendModules;
}

std::size_t CompiledBackendCount()
{
    return sizeof(kCompiledBackendModules) / sizeof(kCompiledBackendModules[0]);
}
} // namespace sim::backends
")
    file(WRITE "${_generated_dir}/compiled_backends.cc" "${_source_content}")

    target_sources(${composition_target} PRIVATE "${_generated_dir}/compiled_backends.cc")
    target_include_directories(${composition_target} PRIVATE "${CMAKE_BINARY_DIR}/generated")
endfunction()

# ---- 阶段二：Host 接入（engine → TARGET；composition + 生成注册 → COMPOSITION_TARGET） ----
function(hcclvm_storage_backends_attach)
    set(_options)
    set(_one_value_keywords TARGET COMPOSITION_TARGET)
    set(_multi_value_keywords)
    cmake_parse_arguments(HCCLVM_STORAGE_BACKEND_ATTACH "${_options}" "${_one_value_keywords}"
        "${_multi_value_keywords}" ${ARGN})
    if(NOT HCCLVM_STORAGE_BACKEND_ATTACH_TARGET)
        message(FATAL_ERROR "hcclvm_storage_backends_attach requires TARGET")
    endif()
    if(NOT HCCLVM_STORAGE_BACKEND_ATTACH_COMPOSITION_TARGET)
        message(FATAL_ERROR "hcclvm_storage_backends_attach requires COMPOSITION_TARGET")
    endif()
    if(NOT TARGET ${HCCLVM_STORAGE_BACKEND_ATTACH_TARGET})
        message(FATAL_ERROR "storage backend attach target does not exist: ${HCCLVM_STORAGE_BACKEND_ATTACH_TARGET}")
    endif()
    if(NOT TARGET ${HCCLVM_STORAGE_BACKEND_ATTACH_COMPOSITION_TARGET})
        message(FATAL_ERROR
            "storage backend composition target does not exist: ${HCCLVM_STORAGE_BACKEND_ATTACH_COMPOSITION_TARGET}")
    endif()
    set(HCCLVM_STORAGE_BACKEND_ATTACH_TARGET "${HCCLVM_STORAGE_BACKEND_ATTACH_TARGET}")
    set(HCCLVM_STORAGE_BACKEND_COMPOSITION_TARGET "${HCCLVM_STORAGE_BACKEND_ATTACH_COMPOSITION_TARGET}")
    _hcclvm_storage_backends_run_phase(ATTACH)
    _hcclvm_storage_backends_generate_registration("${HCCLVM_STORAGE_BACKEND_COMPOSITION_TARGET}")
endfunction()

# ---- 阶段二（Device）：device 子工程内接入（全部编入同一 storage 目标） ----
function(hcclvm_storage_backends_attach_device)
    set(_options)
    set(_one_value_keywords TARGET)
    cmake_parse_arguments(HCCLVM_STORAGE_BACKEND_DEVICE "${_options}" "${_one_value_keywords}" "" ${ARGN})
    if(NOT HCCLVM_STORAGE_BACKEND_DEVICE_TARGET)
        message(FATAL_ERROR "hcclvm_storage_backends_attach_device requires TARGET")
    endif()
    if(NOT TARGET ${HCCLVM_STORAGE_BACKEND_DEVICE_TARGET})
        message(FATAL_ERROR
            "storage backend device attach target does not exist: ${HCCLVM_STORAGE_BACKEND_DEVICE_TARGET}")
    endif()
    set(HCCLVM_STORAGE_BACKEND_DEVICE_TARGET "${HCCLVM_STORAGE_BACKEND_DEVICE_TARGET}")
    _hcclvm_storage_backends_run_phase(DEVICE)
    _hcclvm_storage_backends_generate_registration("${HCCLVM_STORAGE_BACKEND_DEVICE_TARGET}")
endfunction()

# ---- 引擎专属测试子目录接入（test 树内调用；模块不存在则不生成其测试目标）----
# 测试文件保留在 test/ 树内按引擎分子目录（test/<领域>/<模块名>/）：发布裁剪
# 时后端目录与同名测试子目录一起摘除；源码树中该模块不存在时（或其测试子目录
# 缺失时）直接跳过，不生成测试目标，也不用 SKIP 假装验证成功。
function(hcclvm_storage_backends_add_test_dir test_dir)
    hcclvm_storage_backend_modules(_modules)
    foreach(_module IN LISTS _modules)
        if(IS_DIRECTORY "${test_dir}/${_module}" AND EXISTS "${test_dir}/${_module}/CMakeLists.txt")
            add_subdirectory("${test_dir}/${_module}"
                "${CMAKE_CURRENT_BINARY_DIR}/backend_tests/${_module}")
        endif()
    endforeach()
endfunction()
