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
    )
endif()

# coll_comm 侧通过 coll_comm_mgr.h / my_rank.h 引用本模块头文件，需为隔离编译目标补充头文件路径
if(TARGET hcomm_coll_comm_obj)
    target_include_directories(hcomm_coll_comm_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
