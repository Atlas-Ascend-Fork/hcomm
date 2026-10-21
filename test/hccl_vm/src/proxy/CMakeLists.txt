# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# proxy模块按劫持层级组织:
#   level0/ : HCCL算子入口劫持 (hccl_proxy_level0)
#   level1/ : HCCP通信域/控制面接口劫持 (hccl_proxy_level1)
#   level2/ : Runtime/CANN底层接口劫持 (hccl_proxy_level2, 含aiv_kernel)
#   common/ : 各层级共用的源码与头文件 (无独立构建目标, 编入level2)

message(STATUS "TOP_DIR: ${TOP_DIR}")
message(STATUS "ASCEND_CANN_PACKAGE_PATH: ${ASCEND_CANN_PACKAGE_PATH}")

set(PROXY_INC_DIRS
        # hccl_proxy依赖的HCCL业务代码头文件, 这里的头文件有要求: 开源用户可获取
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/profiling
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
        ${HCOMM_CODE_ROOT}/src/pub_inc
        ${HCOMM_CODE_ROOT}/src/framework/communicator/impl/independent_op/rank_graph
        ${HCOMM_CODE_ROOT}/src/framework/inc
        ${HCOMM_CODE_ROOT}/src/platform/hccp/external_depends/inc/log
        # hccl_proxy依赖的Checker L2其他模块的头文件
        ${HCOMM_VM_ROOT_PATH}/include
        ${HCOMM_VM_ROOT_PATH}/src/store
        # cann包头文件
        ${ASCEND_CANN_PACKAGE_PATH}/include/driver
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/driver
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hccl
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/trace
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime/runtime
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime/runtime/rts
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/aicpu
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/aicpu/tsd
        ${HCOMM_CODE_ROOT}/src/base_comm/resources/ccu/ccu_microcode
        ${HCOMM_CODE_ROOT}/src/base_comm/resources/ccu/pub_inc
        # hccl_vm头文件
        ${HCOMM_VM_ROOT_PATH}/include/store
        ${HCOMM_VM_ROOT_PATH}/include/topo
        ${HCOMM_VM_ROOT_PATH}/src/cmd
        ${HCOMM_VM_ROOT_PATH}/src/common
        ${HCCL_VM_SRC}
        ${HCCL_VM_SRC}/store
        # hcomm头文件
        ${HCOMM_CODE_PLATFORM}/hccp
        ${HCOMM_CODE_PLATFORM}/hccp/inc
        ${HCOMM_CODE_PLATFORM}/hccp/inc/network
        ${HCOMM_CODE_PLATFORM}/hccp/external_depends/ccu
        ${HCOMM_CODE_ROOT}/src/base_comm/inc
        # hcomm头文件(适配hcomm PR#2167)
        ${HCOMM_CODE_ROOT}/src/base_comm/resources/hccp
        ${HCOMM_CODE_ROOT}/src/base_comm/resources/hccp/inc
        ${HCOMM_CODE_ROOT}/src/base_comm/resources/hccp/inc/network
        ${HCOMM_CODE_ROOT}/src/base_comm/resources/hccp/external_depends/ccu
        ${HCOMM_CODE_ROOT}/src/legacy/ascend910/pub_inc
        # hccl_proxy模块内部头文件
        ${CMAKE_CURRENT_SOURCE_DIR}/common
)

set(PROXY_COMPILE_OPTS
        -Werror
        -fno-common
        -fno-strict-aliasing
        -pipe
        -O2
        -g
        -D_GLIBCXX_USE_CXX11_ABI=0
)

set(PROXY_LINK_LIBS
        -Wl,--whole-archive
        -Wl,--start-group
        log
        common
        runtime_state
        operation_data

        store
        -Wl,--end-group
        -Wl,--no-whole-archive
        dl
        rt
        JSON::JSON
        ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/lib64/libc_sec.a
)

add_subdirectory(level0)
add_subdirectory(level1)
add_subdirectory(level2)
