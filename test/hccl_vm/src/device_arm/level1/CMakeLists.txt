# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_library(hccl_device_proxy_level1 SHARED "")

target_sources(hccl_device_proxy_level1 PRIVATE
    data_op_device_stub.cc
    comm_domain_device_stub.cc
    dfx_device_stub.cc
    other_device_stub.cc
    dpu_rpc_device_stub.cc
    # 说明：这里**不**编译日志实现(sim_log*.cc)。一个 device 进程只能有一份日志
    # 状态(后端选择、sink、身份 generation)，owner 是 bin/device：它用
    # -Wl,--export-dynamic 把日志运行态导出到全局符号作用域，本打桩库对
    # g_logger / hcclvm::CannLogGate / hcclvm::EmitCannLog 等的引用在加载期解析到
    # 可执行文件的那一份，不再各自实例化。
    # 仅保留既有公共工具源与其编译口径(NO_YAML_CONFIG)不变。
    ${HCCL_VM_SRC}/utils/sim_yaml_config.cc
    ${HCCL_VM_SRC}/utils/sim_common_api.cc
    ${HCCL_VM_SRC}/proxy/level2/log_stub.cc
)

# level1 打桩库只被 device 进程加载：日志状态(后端/sink/身份)由 bin/device 通过
# --export-dynamic 导出的符号提供，本库不再各自实例化一份日志后端。
# 本库不编译读取 HCCLVM_LOG_PROCESS_ROLE_DEVICE 的源(sim_log_cann.cc)，
# 故此处不定义该宏；运行期角色由 bin/device 提供。
target_compile_definitions(hccl_device_proxy_level1 PRIVATE
    NO_YAML_CONFIG
    _GLIBCXX_USE_CXX11_ABI=0
)

target_link_libraries(hccl_device_proxy_level1 PRIVATE ${CMAKE_DL_LIBS})
# -O3: LD_PRELOADed into the device process under qemu TCG, instruction count matters
target_compile_options(hccl_device_proxy_level1 PRIVATE
    -O3
    -g
)

target_include_directories(hccl_device_proxy_level1 PRIVATE
    ${HCCL_VM_SRC}/utils
    ${HCCL_VM_INCLUDE}
    ${HCCL_VM_INCLUDE}/store
    ${HCCL_VM_SRC}

    ${THRID_PARTY_DIR}
    ${THRID_PARTY_DIR}/spdlog
    ${HCCL_VM_SRC}/device_arm
    ${HCCL_VM_SRC}/device_arm/common

    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/pkg_inc/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/include
    ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/include/hccl
)

install(TARGETS hccl_device_proxy_level1
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib/aarch64
    ARCHIVE DESTINATION lib/aarch64
)
