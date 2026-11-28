# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 定义 device 侧 base_comm 独立动态库 libbase_comm_aicpu.so
# 承载编译进 ccl_kernel 的 base_comm 源文件，由 ccl_kernel 链接。
# 不挂 version-script，保留 RunAicpuThread*/HcommThreadCreateAicpu 等入口供 AICPU 按名解析。
add_library(base_comm_aicpu SHARED)

# 编译选项（与 ccl_kernel 保持一致）
if(BUILD_OPEN_PROJECT)
    target_compile_options(base_comm_aicpu PRIVATE
        -Werror
        -fno-common
        -fno-strict-aliasing
        $<$<CONFIG:Debug>:-Og -g>
        $<$<CONFIG:Release>:-O3>
    )
else()
    target_compile_options(base_comm_aicpu PRIVATE
        -Werror
        -fno-common
        -fno-strict-aliasing
        -O3
    )
    target_link_options(base_comm_aicpu PRIVATE
        -s
    )
endif()

# 预处理宏定义（与 ccl_kernel 保持一致）
target_compile_definitions(base_comm_aicpu PRIVATE
    HCCD
    CCL_KERNEL_AICPU
)

if(BUILD_OPEN_PROJECT)
    target_compile_definitions(base_comm_aicpu PRIVATE
        OPEN_BUILD_PROJECT
    )

    target_link_libraries(base_comm_aicpu PRIVATE
        $<BUILD_INTERFACE:intf_pub>
        $<BUILD_INTERFACE:c_sec_headers>
        $<BUILD_INTERFACE:acl_rt_headers>
        $<BUILD_INTERFACE:asc_host_headers>
        $<BUILD_INTERFACE:asc_kernel_headers>
        $<BUILD_INTERFACE:ascend_hal_headers>
        $<BUILD_INTERFACE:kernel_tiling_headers>
        $<BUILD_INTERFACE:slog_headers>
        $<BUILD_INTERFACE:atrace_headers>
        $<BUILD_INTERFACE:mmpa_headers>
        $<BUILD_INTERFACE:runtime_headers>
        $<BUILD_INTERFACE:rdma_core_headers>
        $<BUILD_INTERFACE:json>
        -Wl,--no-as-needed
        ascend_hal
        c_sec
        mmpa
        -Wl,--as-needed
    )
else()
    target_include_directories(base_comm_aicpu PRIVATE
        ${TOP_DIR}/inc
        ${TOP_DIR}/inc/driver
        ${TOP_DIR}/metadef/inc/external
        ${TOP_DIR}/metadef/pkg_inc
        ${TOP_DIR}/open_source/json/include
        ${TOP_DIR}/inc/aicpu/
        ${TOP_DIR}/atc/opcompiler/ascendc_compiler/api
        ${TOP_DIR}/atc/opcompiler/ascendc_compiler/api/impl
        ${TOP_DIR}/atc/opcompiler/ascendc_compiler/api/include
        ${TOP_DIR}/abl/atrace/inc/utrace
        ${TOP_DIR}/runtime/include/external
        ${TOP_DIR}/runtime/include/external/acl
        ${TOP_DIR}/runtime/pkg_inc
        ${TOP_DIR}/runtime/pkg_inc/runtime
        ${TOP_DIR}/runtime/pkg_inc/profiling
        ${TOP_DIR}/runtime/pkg_inc/trace
        ${TOP_DIR}/runtime/pkg_inc/base
        ${TOP_DIR}/runtime/pkg_inc/aicpu_sched
        ${TOP_DIR}/asc/asc-devkit
        ${TOP_DIR}/asc/asc-devkit/include/adv_api/hccl/internal
    )

    target_link_libraries(base_comm_aicpu PRIVATE
        $<BUILD_INTERFACE:intf_pub>
        $<BUILD_INTERFACE:c_sec_headers>
        $<BUILD_INTERFACE:mmpa_headers>
        $<BUILD_INTERFACE:msprof_headers>
        $<BUILD_INTERFACE:slog_headers>
        $<BUILD_INTERFACE:npu_runtime_headers>
        $<BUILD_INTERFACE:atrace_headers>
        $<BUILD_INTERFACE:kernel_tiling_headers>
        $<BUILD_INTERFACE:ascend_hal_headers>
        -Wl,--no-as-needed
        ascend_hal
        c_sec
        mmpa
        -Wl,--as-needed
        ofed_headers
    )
endif()


# 头文件搜索路径（按 ascend910 -> ascend950 排序，显式维护，避免隐式继承 ccl_kernel）
# 仅保留当前 device 源文件实际引用到的目录；新增 legacy 头依赖时同步补充。
# 架构约束：不得加入 coll_communicator_mgr 路径。
include(${CMAKE_CURRENT_LIST_DIR}/base_comm_aicpu_inc.cmake)
target_include_directories(base_comm_aicpu PRIVATE ${BASE_COMM_AICPU_INCLUDE_DIRS})

# 安装 libbase_comm_aicpu.so，位置与 ccl_kernel 的 device 库保持一致，
# 避免 ccl_kernel / aicpu_custom 的 DT_NEEDED 在安装后找不到依赖库。
install(TARGETS base_comm_aicpu
    LIBRARY DESTINATION ${INSTALL_DEVICE_LIBRARY_DIR} ${INSTALL_OPTIONAL}
    COMPONENT hcomm
)
# 与 ccl_kernel 保持一致：只有非开源（产品）构建才额外安装到 INSTALL_LIBRARY_DIR；
# 开源构建只安装到 INSTALL_DEVICE_LIBRARY_DIR（device 目录）。
# 否则会多出一个未被 filelist 跟踪的 aarch64-linux/lib64 副本，导致卸载残留。
if(NOT BUILD_OPEN_PROJECT)
    install(TARGETS base_comm_aicpu
        LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} ${INSTALL_OPTIONAL}
        COMPONENT hcomm
    )
endif()
