# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 创建基础通信层中间对象库, host编hcomm_base_obj
if(PRODUCT_SIDE STREQUAL "host")
    include(hcomm_base_obj.cmake)
endif()

# device 侧 base_comm 独立动态库 libbase_comm_aicpu.so
if(PRODUCT_SIDE STREQUAL "device")
    include(base_comm_aicpu.cmake)
endif()


add_subdirectory(common)
add_subdirectory(primitives)
add_subdirectory(resources)
add_subdirectory(config_mgr)
add_subdirectory(dfx)

# device 侧 base_comm 静态库 libbase_comm_aicpu.a，供 aicpu_custom 静态链接，
# 避免 aicpu_custom -> base_comm -> aicpu_custom 的循环符号依赖。
# 必须在所有子目录挂接完源文件后创建，确保静态库包含完整的 base_comm 设备侧源文件。
if(PRODUCT_SIDE STREQUAL "device" AND TARGET base_comm_aicpu)
    add_library(base_comm_aicpu_a STATIC)
    clone_cann_target(
        ORIGIN base_comm_aicpu
        OUTPUT base_comm_aicpu_a
        IGNORE_PROP LINK_OPTIONS     # 静态库不参与链接，忽略 -s 等链接选项
    )
    set_target_properties(base_comm_aicpu_a PROPERTIES
        OUTPUT_NAME base_comm_aicpu
        POSITION_INDEPENDENT_CODE ON
    )
endif()

set(src_list
    base_comm_mgr.cc
    handle_registry_procs.cc
    base_comm_res_per_dev.cc
    legacy950_res_mgr_proc.cc
    base_comm_node_accessors.cc
)

if(TARGET hcomm_base_obj)
    target_sources(hcomm_base_obj PRIVATE
        ${src_list}
    )
endif()