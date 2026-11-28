# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(src_list
    aicpu_channel_process.cc
    dev_aicpu_ts_channel_mgr.cc
    dev_aicpu_ts_roce_dev_loc_channel.cc
    aicpu_ts_urma_channel_kernel.cc
    dev_aicpu_ts_hccs_channel.cc
)

set(ep_mem_table_src
    dev_ep_mem_table.cc
    aicpu_ts_ep_mem_table_kernel.cc
)

if(TARGET base_comm_aicpu)
    target_sources(base_comm_aicpu PRIVATE
        ${src_list}
        ${ep_mem_table_src}
    )
    target_include_directories(base_comm_aicpu PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
    )
endif()

if(TARGET hccl_v2)
    target_sources(hccl_v2 PRIVATE
        dev_ep_mem_table.cc
    )
    target_include_directories(hccl_v2 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()

if(TARGET ccl_kernel_plf)
    target_sources(ccl_kernel_plf PRIVATE
        dev_ep_mem_table.cc
    )
    target_include_directories(ccl_kernel_plf PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()

if(TARGET ccl_kernel_plf_a)
    target_include_directories(ccl_kernel_plf_a PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
