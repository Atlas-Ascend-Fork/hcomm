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
    cpu_ts_thread.cc
    aicpu_ts_thread.cc
    aiv_thread.cc
    thread.cc
)


if(TARGET hcomm_base_obj)
    target_sources(hcomm_base_obj PRIVATE
        ${src_list}
        # HOST CPU 线程（HostDpu kernel）依赖 aclrt*/Hrt*，device 侧没有这些接口，
        # 故单独挂 hcomm_base_obj，不进上面同时喂 ccl_kernel 的 src_list
        cpu_thread/cpu_thread.cc
        cpu_thread/legacy_dpu_thread/legacy_dpu_thread.cc
        cpu_thread/legacy_dpu_thread/dpu_shmem_mgr.cc
    )
    target_include_directories(hcomm_base_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/device
        ${CMAKE_CURRENT_SOURCE_DIR}/cpu_thread
    )
endif()

add_subdirectory(device)
if(TARGET ccl_kernel)
    target_sources(ccl_kernel PRIVATE
        ${src_list}
    )
    target_include_directories(ccl_kernel PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
