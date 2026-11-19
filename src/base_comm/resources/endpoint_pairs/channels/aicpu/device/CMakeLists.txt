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

# DevEpMemTable（device 侧 endpoint 本地内存活账本表）：
# dev_ep_mem_table.cc 同时供 ccl_kernel（device 表类）与 hccl_v2（ub_transport_lite_impl.cc
# 建链收敛/查表切换引用其符号）编译链接；kernel 入口 aicpu_ts_ep_mem_table_kernel.cc 仅 device 侧
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

# PLF 味（910 平台 ccl_kernel_plf/_a）同样编译 ub_transport_lite_impl.cc（查表引用本表符号）。
# 非对称接入：SHARED 版不反向依赖 base_comm，须自含表源；_a 版不编入——其唯一链接方
# aicpu_custom 以 --whole-archive 同时打入 base_comm_aicpu_a（克隆自 base_comm_aicpu，已含本源），
# 此处再编一份必致全量符号重复定义；_a 保留 include 路径供 ub_transport_lite_impl.cc 编译，
# 未解析引用由链接期 base_comm_aicpu_a 的副本提供。kernel 入口 aicpu_ts_ep_mem_table_kernel.cc 两版均不编入
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
