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

# device 侧需要 AicpuTsThread（RunAicpuThread* 入口），以及 thread.cc 在
# CCL_KERNEL_AICPU 分支下提供的 Thread::FormatThreadInfo、HcommThreadCreateAicpu/
# Destroy/RegisterDestroyCallback 等实现（ccl_kernel 的 thread_aicpu_mgr.cc 依赖）。
# cpu_ts_thread/aiv_thread 只被 host 侧使用，不进 base_comm_aicpu。
set(device_src_list
    aicpu_ts_thread.cc
    thread.cc
)


if(TARGET hcomm_base_obj)
    target_sources(hcomm_base_obj PRIVATE
        thread_entity_resources.cc
        ${src_list}
        # HOST CPU 线程（HostDpu kernel）依赖 aclrt*/Hrt*，device 侧没有这些接口，
        # 故单独挂 hcomm_base_obj，不进上面同时喂 ccl_kernel 的 src_list
        cpu_thread/cpu_thread.cc
        # 两条通道共用的 DPU 域机制（XPU 切换/记账、dpu 流、kernel 下发、终止握手、DPUTAG）
        cpu_thread/cpu_thread_base.cc
        cpu_thread/legacy_dpu_thread/legacy_dpu_thread.cc
        cpu_thread/legacy_dpu_thread/dpu_shmem_mgr.cc
        # SQ/CQ 任务管道：与 ccl_dpu 共用同一份源码（编两遍），两侧各内部绑定
        cpu_thread/cpu_task_pipe.cc
    )
    target_include_directories(hcomm_base_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/device
        ${CMAKE_CURRENT_SOURCE_DIR}/cpu_thread
    )
endif()

add_subdirectory(device)
if(TARGET base_comm_aicpu)
    target_sources(base_comm_aicpu PRIVATE
        ${device_src_list}
    )
    target_include_directories(base_comm_aicpu PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
