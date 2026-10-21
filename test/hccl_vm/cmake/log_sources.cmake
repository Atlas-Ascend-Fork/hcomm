# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# host log 库与 device 可执行文件共享实现；device proxy 不重复编译日志状态。
set(HCCLVM_LOG_SOURCES
    ${HCCL_VM_SRC}/utils/sim_log.cc
    ${HCCL_VM_SRC}/utils/sim_log_cann.cc
    ${HCCL_VM_SRC}/utils/sim_yaml_config.cc
    ${HCCL_VM_SRC}/utils/sim_common_api.cc
)
