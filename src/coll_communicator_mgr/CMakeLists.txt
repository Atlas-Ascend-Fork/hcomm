# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 将coll_comm编译到legacy，因为存在legacy依赖base_comm，这里
# 将coll_comm和legacy进行隔离，降低legacy的影响来确定coll_comm依赖base_comm的边界


if(PRODUCT_SIDE STREQUAL "host")
    include(hcomm_coll_comm.cmake)
endif()

add_subdirectory(common)
add_subdirectory(resource_mgr)
add_subdirectory(api_c_adpt)
add_subdirectory(rank_graph)
add_subdirectory(rank_info_detect)
add_subdirectory(dfx)
add_subdirectory(config_mgr)
add_subdirectory(communicator)
add_subdirectory(team)

if(TARGET hcomm_coll_comm_obj)
    target_include_directories(hcomm_coll_comm_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
