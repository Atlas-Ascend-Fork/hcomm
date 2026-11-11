# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License).
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(device_src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/dfx_profiling_handler_lite.cc
)

# host 侧(hcomm_base_obj)仅引用头文件, 不编译 device 源
if(TARGET hcomm_base_obj)
    target_include_directories(hcomm_base_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/common
    )
endif()

if(TARGET base_comm_aicpu)
    target_sources(base_comm_aicpu PRIVATE
        ${device_src_list}
    )
    target_include_directories(base_comm_aicpu PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/common
        ${HCOMM_DIR}/src/base_comm/resources/comm_engine_res/threads
    )
endif()

if(TARGET hccl_v2)
    target_sources(hccl_v2 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/dfx_profiling_handler_lite.cc
    )
    target_include_directories(hccl_v2 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/common
    )
endif()
