# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_library(hccl_device_proxy SHARED "")

target_sources(hccl_device_proxy PRIVATE
    aicpu_stub.cc
    ascend_hal_stub.cc
    device_sqe_parse_stub.cc
    aprofiling_stub.cc
    hcomm_api_stub.cc
    dlopen_stub.cc
    # 说明：这里不编译日志实现(sim_log*.cc)。device 进程的日志状态 owner 是
    # bin/device(经 -Wl,--export-dynamic 导出)，本库只引用、不实例化第二份，既有公共工具源与编译口径保持不变。
    ${HCCL_VM_SRC}/utils/sim_yaml_config.cc
    ${HCCL_VM_SRC}/utils/sim_common_api.cc
    ${HCCL_VM_SRC}/proxy/level2/log_stub.cc
)

# level2 打桩库只被 device 进程加载：日志状态(后端/sink/身份)由 bin/device 提供。
# 本库不编译读取 HCCLVM_LOG_PROCESS_ROLE_DEVICE 的源(sim_log_cann.cc)，
# 故此处不定义该宏；运行期角色由 bin/device 提供。
target_compile_definitions(hccl_device_proxy PRIVATE
    NO_YAML_CONFIG
    _GLIBCXX_USE_CXX11_ABI=0
)

target_link_libraries(hccl_device_proxy PRIVATE ${CMAKE_DL_LIBS})

# -O3: LD_PRELOADed into the device process under qemu TCG, instruction count matters
target_compile_options(hccl_device_proxy PRIVATE
    -O3
    -g
)

target_include_directories(hccl_device_proxy PRIVATE
    ${HCCL_VM_SRC}/utils
    # 业务兼容头（runtime_state/db_sim_runner_ops.h）经 table_access.h → table.h 引用
    # src/ 下的引擎合同头（storage/engine/i_storage_engine.h 等）：与 device 内 storage
    # 静态库的 PUBLIC include 集保持一致（链接闭包同一解析域）。
    ${HCCL_VM_SRC}
    ${HCCL_VM_INCLUDE}
    ${HCCL_VM_INCLUDE}/store

    ${THRID_PARTY_DIR}
    ${THRID_PARTY_DIR}/spdlog
    ${HCCL_VM_SRC}/device_arm
    ${HCCL_VM_SRC}/device_arm/common

    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/pkg_inc/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/include
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/include/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/pkg_inc/driver
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/pkg_inc/runtime
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/pkg_inc/toolchain
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/pkg_inc/profiling
)

install(TARGETS hccl_device_proxy
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib/aarch64
    ARCHIVE DESTINATION lib/aarch64
)

add_library(slog SHARED slog_stub.cc)
target_compile_options(slog PRIVATE
    -g
    -O3
)
install(TARGETS slog
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib/aarch64
    ARCHIVE DESTINATION lib/aarch64
)

add_library(aicpu_sharder SHARED depends_stub.cc)
target_compile_options(aicpu_sharder PRIVATE
    -g
    -O3
)
install(TARGETS aicpu_sharder
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib/aarch64
    ARCHIVE DESTINATION lib/aarch64
)
