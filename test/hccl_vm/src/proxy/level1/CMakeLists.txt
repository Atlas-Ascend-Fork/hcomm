# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_library(hccl_proxy_level1 SHARED "")

target_include_directories(hccl_proxy_level1 PRIVATE
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
)

target_include_directories(hccl_proxy_level1 PRIVATE
        # hccl_proxy依赖的Checker L2其他模块的头文件
        ${HCOMM_VM_ROOT_PATH}/include
        ${HCOMM_VM_ROOT_PATH}/src/common
        ${HCOMM_VM_ROOT_PATH}/src/store
        ${HCOMM_VM_ROOT_PATH}/src/host/hccl_vm
        ${HCOMM_VM_ROOT_PATH}/src/host/hccl_check_plugin
        ${HCOMM_VM_ROOT_PATH}/src/host/hccl_check_plugin/checker/utils
)

target_include_directories(hccl_proxy_level1 PRIVATE
        # cann包头文件
        ${ASCEND_CANN_PACKAGE_PATH}/include/driver
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hccl
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/trace
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime/runtime
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/runtime/runtime/rts
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hcomm/ccu
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/asc/hccl/internal/hcomm/pkg_inc/hcomm/ccu
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/aicpu
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/aicpu/tsd
        # CCU 北向桩依赖的 CANN 包 CCU 公共头(ccu_types.h/ccu_res.h/ccu_res_defs.h/ccu_launch.h)
        ${ASCEND_CANN_PACKAGE_PATH}/include/hcomm/ccu

        # hccl_vm头文件
        ${HCOMM_VM_ROOT_PATH}/include
        ${HCOMM_VM_ROOT_PATH}/include/store
        ${HCOMM_VM_ROOT_PATH}/hccl_host/hccl_cmd
        ${HCOMM_VM_ROOT_PATH}/include/topo
        ${HCOMM_VM_ROOT_PATH}/src/cmd

        ${HCCL_VM_SRC}
        ${HCCL_VM_SRC}/store

        # hcomm头文件
        ${HCOMM_CODE_PLATFORM}/hccp
        ${HCOMM_CODE_PLATFORM}/hccp/inc
        ${HCOMM_CODE_PLATFORM}/hccp/inc/network
        ${HCOMM_CODE_PLATFORM}/hccp/external_depends/ccu

        # hcomm头文件(适配hcomm PR#2167)
        ${HCOMM_CODE_ROOT}/src/base_comm/resources/hccp
        ${HCOMM_CODE_ROOT}/src/base_comm/resources/hccp/inc
        ${HCOMM_CODE_ROOT}/src/base_comm/resources/hccp/inc/network
        ${HCOMM_CODE_ROOT}/src/base_comm/resources/hccp/external_depends/ccu
        ${HCOMM_CODE_ROOT}/src/legacy/ascend910/pub_inc

        # hccl_proxy level1模块内部头文件
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

target_sources(hccl_proxy_level1 PRIVATE
        # 通信域管理
        ${CMAKE_CURRENT_SOURCE_DIR}/comm_domain_manage_stub.cc
        # 控制面接口
        ${CMAKE_CURRENT_SOURCE_DIR}/ctrl_topo_query_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ctrl_base_resource_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ctrl_comm_resource_stub.cc
        # 数据面接口
        ${CMAKE_CURRENT_SOURCE_DIR}/data_local_op_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/data_comm_op_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/data_other_op_stub.cc
        # CCU 北向劫持（全自研路线）
        ${CMAKE_CURRENT_SOURCE_DIR}/ccu/ccu_resource_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ccu/ccu_control_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ccu/ccu_data_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ccu/ccu_kernel_sim.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ccu/ccu_level1_res.cc
        # ccu_level1_common.cc 暂不编入：与 ccu_kernel_sim.cc 存在同名符号
        # 待后续从 ccu_kernel_sim.cc 迁出公共函数后启用
        # ${CMAKE_CURRENT_SOURCE_DIR}/ccu/ccu_level1_common.cc
        # HostDPU 北向劫持（执行器 + 控制面 + 数据面）
        ${CMAKE_CURRENT_SOURCE_DIR}/dpu/dpu_level1_res.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/dpu/dpu_executor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/dpu/dpu_control_stub.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/dpu/dpu_data_stub.cc
        # 其他接口
        ${CMAKE_CURRENT_SOURCE_DIR}/level1_other_stub.cc
        # DFX 诊断接口
        ${CMAKE_CURRENT_SOURCE_DIR}/dfx_stub.cc
        # 公共
        ${CMAKE_CURRENT_SOURCE_DIR}/level1_proxy_common.cc
)

target_compile_options(hccl_proxy_level1 PRIVATE
        -Werror
        -fno-common
        -fno-strict-aliasing
        -pipe
        -O2
        -g
        -D_GLIBCXX_USE_CXX11_ABI=0
)

# level1 不直接链接 log/runtime_state/operation_data/store/cmd 等公共库, 避免与 level2 产生
# 重复全局符号(单例)导致崩溃。这些符号在运行期由 LD_PRELOAD 链中靠后的 level2
# 提供, 链接期用 --unresolved-symbols=ignore-in-object-files 放行。
target_link_options(hccl_proxy_level1 PRIVATE
        -Wl,--unresolved-symbols=ignore-in-object-files)

target_link_libraries(hccl_proxy_level1 PRIVATE
        dl
        rt
        JSON::JSON
        ${ASCEND_CANN_PACKAGE_PATH}/${CANN_ARCH_DIR}/lib64/libc_sec.a
)

install(TARGETS hccl_proxy_level1
    DESTINATION "lib/${CMAKE_SYSTEM_PROCESSOR}"
    # 文件的权限 (755)
    PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE
)
