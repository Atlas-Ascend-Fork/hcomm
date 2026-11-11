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
    aicpu_thread_process.cc
    aicpu_thread_kernel.cc
    aicpu_rtsq_poll_daemon.cc
    # DPU 队列名片投递 kernel（回填 device 侧 DPU 线程实体）
    cpu_device_queue_namecard_restore_kernel.cc
    # DPU notify 扩容投递 kernel（host 重分配共享内存后回填新地址/容量）
    cpu_device_notify_addr_update_kernel.cc
    # DPU 线程 device 侧实体与它的 handle→实体 登记表（L2 建、L3 销，同镜像共享）
    cpu_device_thread.cc
    cpu_device_thread_registry.cc
)

if(TARGET base_comm_aicpu)
    target_sources(base_comm_aicpu PRIVATE
        ${src_list}
    )
    target_include_directories(base_comm_aicpu PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        # dpu_sync_layout.h 是跨三侧的布局契约，与宿主侧 cpu_thread 共用同一份
        ${CMAKE_CURRENT_SOURCE_DIR}/../cpu_thread
    )
endif()
