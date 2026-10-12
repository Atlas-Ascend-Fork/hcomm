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
    ${HCCL_VM_SRC}/utils/sim_log.cc
    ${HCCL_VM_SRC}/utils/sim_yaml_config.cc
    ${HCCL_VM_SRC}/utils/sim_common_api.cc
    ${HCCL_VM_SRC}/proxy/level2/log_stub.cc
)

target_compile_definitions(hccl_device_proxy_level1 PRIVATE
    NO_YAML_CONFIG
    _GLIBCXX_USE_CXX11_ABI=0
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
