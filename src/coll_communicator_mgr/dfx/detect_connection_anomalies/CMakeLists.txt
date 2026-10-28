# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/detect_connection_anomalies.cc
)

if(TARGET hcomm)
    target_sources(hcomm PRIVATE
        ${src_list}
    )
    target_include_directories(hcomm PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${HCOMM_DIR}/src/base_comm/common
        ${HCOMM_DIR}/src/base_comm/resources/endpoint_pairs/sockets
        ${HCOMM_DIR}/src/base_comm/resources/endpoints/server_socket
        ${HCOMM_DIR}/src/legacy/ascend950/unified_platform/resource/socket
        # detect_connection_anomalies.cc include base_comm_node_accessors.h（src/base_comm 根），
        # 其闭包依赖以下目录；主干 6868a2fb0 已从 hcomm.cmake 移除裸露的 src/base_comm 根，
        # 故在此按 TU 最小集补齐（依据：flags.make 实际 -I 集合 + 头文件闭包解析）
        ${HCOMM_DIR}/src/base_comm
        ${HCOMM_DIR}/src/base_comm/resources/ccu/ccu_device
        ${HCOMM_DIR}/src/base_comm/resources/ccu/ccu_kernel
        ${HCOMM_DIR}/src/base_comm/resources/ccu/ccu_microcode
        ${HCOMM_DIR}/src/base_comm/resources/ccu/ccu_representation
        ${HCOMM_DIR}/src/base_comm/resources/ccu/ccu_representation/context
        ${HCOMM_DIR}/src/base_comm/resources/ccu/ccu_representation/interface
        ${HCOMM_DIR}/src/base_comm/resources/ccu/ccu_representation/reps/arithmetic
        ${HCOMM_DIR}/src/base_comm/resources/ccu/ccu_representation/reps/common
        ${HCOMM_DIR}/src/base_comm/resources/ccu/ccu_representation/reps/control
        ${HCOMM_DIR}/src/base_comm/resources/ccu/ccu_representation/reps/data
        ${HCOMM_DIR}/src/base_comm/resources/ccu/ccu_representation/reps/logical
        ${HCOMM_DIR}/src/base_comm/resources/ccu/ccu_representation/reps/loop
        ${HCOMM_DIR}/src/base_comm/resources/ccu/ccu_representation/reps/shift
        ${HCOMM_DIR}/src/base_comm/resources/ccu/ccu_representation/reps/sync
        ${HCOMM_DIR}/src/base_comm/resources/ccu/ccu_representation/reps/translator
        ${HCOMM_DIR}/src/base_comm/resources/endpoints
        ${HCOMM_DIR}/src/base_comm/resources/reged_mems
        ${HCOMM_DIR}/src/legacy/ascend950/unified_platform
    )
endif()

# coll_comm 侧通过 coll_comm_mgr.h / my_rank.h 引用本模块头文件，需为隔离编译目标补充头文件路径
if(TARGET hcomm_coll_comm_obj)
    target_include_directories(hcomm_coll_comm_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
