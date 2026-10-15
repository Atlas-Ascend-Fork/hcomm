# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_subdirectory(runner)
add_subdirectory(checker)

#install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/checker"
#    DESTINATION "plugin/validate"
#    # 文件的权限 (755)
#    FILE_PERMISSIONS
#        OWNER_READ OWNER_WRITE OWNER_EXECUTE
#        GROUP_READ GROUP_EXECUTE
#        WORLD_READ WORLD_EXECUTE
#    # 目录的权限 (755)
#    DIRECTORY_PERMISSIONS
#        OWNER_READ OWNER_WRITE OWNER_EXECUTE
#        GROUP_READ GROUP_EXECUTE
#        WORLD_READ WORLD_EXECUTE
#)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/insight"
    DESTINATION "plugin/visualization"
    # 文件的权限 (755)
    FILE_PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE
    # 目录的权限 (755)
    DIRECTORY_PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE
    PATTERN "frontend" EXCLUDE
    PATTERN "frontend_v3" EXCLUDE
)

# estimate 性能仿真插件 (设计文档 docs/light_performance 3.4 节):
# 插件目录即 light_performance 包目录 (plugin/light_performance), manifest.json、
# main.py (一次性执行入口, 宿主 `hccl-vm plugin run @estimate` 触发) 与缺省配置
# sim_config.json / bw_profile.json (单位 us/GBps, 设计文档 7.2 节) 均在包目录内,
# 整目录安装即可; 排除运行期产物 __pycache__ 与 data/
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/light_performance"
    DESTINATION "plugin"
    FILE_PERMISSIONS
        OWNER_READ OWNER_WRITE
        GROUP_READ
        WORLD_READ
    DIRECTORY_PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE
    PATTERN "__pycache__" EXCLUDE
    PATTERN "data" EXCLUDE
)
