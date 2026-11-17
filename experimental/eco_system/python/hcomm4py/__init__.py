# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""hcomm4py — HCCL/HCOMM communication-domain Python bindings (hcomm repo)."""

from .types import (
    HcclResult,
    HcclReduceOp,
    HcclDataType,
    HcclSendRecvType,
    HcclRootInfo,
    HcclCommConfig,
    HcclSendRecvItem,
    HcclError,
    HCCL_ROOT_INFO_BYTES,
    HCCL_COMM_CONFIG_BYTES,
    HCCL_COMM_CONFIG_MAGIC_WORD,
    HCCL_COMM_CONFIG_SIZE_V10,
    HCCL_COMM_CONFIG_SIZE_V11,
    HCCL_COMM_CONFIG_SIZE_V12,
    HCCL_COMM_CONFIG_VERSION_DEFAULT,
    HCCL_COMM_CONFIG_LOGICAL_SIZE_DEFAULT,
)
from .comm import (
    get_root_info,
    get_comm_config_capability,
    comm_config_init,
    comm_init_root_info_config,
    comm_init_cluster_info_config,
    comm_destroy,
    init_rank_config,
    Comm,
)

__all__ = [
    "HcclResult",
    "HcclReduceOp",
    "HcclDataType",
    "HcclSendRecvType",
    "HcclRootInfo",
    "HcclCommConfig",
    "HcclSendRecvItem",
    "HcclError",
    "HCCL_ROOT_INFO_BYTES",
    "HCCL_COMM_CONFIG_BYTES",
    "HCCL_COMM_CONFIG_MAGIC_WORD",
    "HCCL_COMM_CONFIG_SIZE_V10",
    "HCCL_COMM_CONFIG_SIZE_V11",
    "HCCL_COMM_CONFIG_SIZE_V12",
    "HCCL_COMM_CONFIG_VERSION_DEFAULT",
    "HCCL_COMM_CONFIG_LOGICAL_SIZE_DEFAULT",
    "get_root_info",
    "get_comm_config_capability",
    "comm_config_init",
    "comm_init_root_info_config",
    "comm_init_cluster_info_config",
    "comm_destroy",
    "init_rank_config",
    "Comm",
]

__version__ = "0.1.0"
