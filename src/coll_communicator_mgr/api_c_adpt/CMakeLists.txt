# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_subdirectory(dev)

set(src_list
    coll_comm_res_c_adpt.cc
    coll_comm_c_adpt.cc
    coll_comm_ccu_c_adpt.cc
    exchange_info_c_adpt.cc
    coll_comm_profiling.cc
    coll_comm_rank_graph_a_adpt.cc
    hccl_channel_config.cc
)

if(TARGET hcomm)
    target_sources(hcomm PRIVATE
        ${src_list}
    )
    target_include_directories(hcomm PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        # HostDpu 收编：dfx / communicator 侧经 CpuThread 反查 HOST CPU 线程（cpu_thread.h -> legacy_dpu_thread.h -> dpu_types.h）
        ${HCOMM_DIR}/src/base_comm/resources/comm_engine_res/threads/cpu_thread
        ${HCOMM_DIR}/src/base_comm/resources/comm_engine_res/threads/cpu_thread/legacy_dpu_thread
        # coll_comm_c_adpt.cc 的 HcommFenceOnThread 直接调 FlushManager（flush 迁 base_comm resources/flush）
        ${HCOMM_DIR}/src/base_comm/resources/flush
    )
endif()

if(TARGET ccl_kernel)
    target_sources(ccl_kernel PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/coll_comm_rank_graph_a_adpt.cc
    )
    target_include_directories(ccl_kernel PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()

add_subdirectory(resource)