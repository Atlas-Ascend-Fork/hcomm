# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# Level0: HCCL算子入口劫持
add_library(hccl_proxy_level0 SHARED "")
target_include_directories(hccl_proxy_level0 PRIVATE ${PROXY_INC_DIRS})
target_include_directories(hccl_proxy_level0 PRIVATE ${HCOMM_VM_ROOT_PATH}/src/proxy/level1)
target_sources(hccl_proxy_level0 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/hccl_op_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/hccl_comm_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/mpi_stub.cc
	${CMAKE_CURRENT_SOURCE_DIR}/ccu_fallback_stub.cc
)
target_compile_options(hccl_proxy_level0 PRIVATE ${PROXY_COMPILE_OPTS})

target_link_options(hccl_proxy_level0 PRIVATE
        -Wl,--unresolved-symbols=ignore-in-object-files)
target_link_libraries(hccl_proxy_level0 PRIVATE
        dl
        rt
        ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/lib64/libc_sec.a)
install(TARGETS hccl_proxy_level0
    DESTINATION "lib/${CMAKE_SYSTEM_PROCESSOR}"
    PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE
)
