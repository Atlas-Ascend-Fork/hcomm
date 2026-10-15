# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""HCCL 算法性能仿真工具包 (以 docs/light_performance/HCCLSIM 为基准收编)。"""

from .config import SimConfig
from .task_model import Task, TaskType, INVALID
from .dependency_graph import DependencyGraph
from .bandwidth_profile import BandwidthProfile, BandwidthProfileSet
from .topo_resolver import TopoResolver
from .simulator import HcclSimulator, NetworkPort
from .task_timing import (
    calc_notify_record_delay, calc_notify_wait_delay, calc_hbm_delay,
    calc_write_with_notify_delay, get_task_timing_info,
)
from .input_builder import _streams_to_dep_graph, load_config
from .report import generate_report, export_trace

__all__ = [
    "SimConfig",
    "Task", "TaskType", "INVALID",
    "DependencyGraph",
    "BandwidthProfile", "BandwidthProfileSet",
    "TopoResolver",
    "HcclSimulator", "NetworkPort",
    "calc_notify_record_delay", "calc_notify_wait_delay", "calc_hbm_delay",
    "calc_write_with_notify_delay",
    "get_task_timing_info",
    "_streams_to_dep_graph", "load_config",
    "generate_report", "export_trace",
]
