# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# rank_graph 模块 ODR 整改 namespace 迁移边界说明（issue #1110 第一批）：
# coll_comm 侧类是否迁入 namespace hccl，判定规则 = 「该头是否在 legacy/ascend950 镜像目录存在同名头」：
#   - 存在同名镜像头（跨 SO 同名同 guard 同类名，严格 ODR 冲突）→ coll_comm 侧迁 namespace hccl，
#     legacy950 侧加 _legacy950 后缀 + guard 区分（情况2 则两份 .h 声明一致 + 共用 coll_comm 版 .cc）。
#   - 无同名镜像头 → 留 namespace Hccl 不迁。
# 第一批已迁：topo_info(组3) / phy_topo_builder(组4) / new_rank_info(组5) / changed_rank_info(组7) / rank_table_info(组11)。
# 后续批次（组9-20）按同规则逐组迁入 hccl，详见 doc/ODR整改方案.md 与 doc/hcomm重复头文件清单.html。

set(src_list
    ${CMAKE_CURRENT_SOURCE_DIR}/rank_graph.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/rank_graph_v2.cc
)

set(v2_rank_graph_include_dirs
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/phy_topo
    ${CMAKE_CURRENT_SOURCE_DIR}/phy_topo_builder
    ${CMAKE_CURRENT_SOURCE_DIR}/rank_graph_builder
    ${CMAKE_CURRENT_SOURCE_DIR}/rank_table_info
    ${CMAKE_CURRENT_SOURCE_DIR}/topo_info
    ${HCOMM_DIR}/src/legacy/ascend950/common/types
    ${HCOMM_DIR}/src/legacy/ascend950/common/utils
    ${HCOMM_DIR}/src/legacy/ascend950/framework/topo/topo_addr_info/include
)

if(TARGET hcomm_coll_comm_obj)
    target_sources(hcomm_coll_comm_obj PRIVATE
        ${src_list}
    )
    target_include_directories(hcomm_coll_comm_obj BEFORE PRIVATE ${v2_rank_graph_include_dirs})
endif()

if(TARGET ccl_kernel)
    target_include_directories(ccl_kernel PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    # 裸名 graph.h 存在 legacy 同名镜像（new_topo_builder/common，同 guard GROUP_H），BEFORE 前插优先命中本目录版本，勿调整 -I 顺序
    target_include_directories(ccl_kernel BEFORE PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/common)
endif()

if(TARGET hccl_v2)
    target_include_directories(hccl_v2 BEFORE PRIVATE ${v2_rank_graph_include_dirs})
endif()

# hcomm target 同 hccl_v2 前插 coll_comm rank_graph 目录：裸名 rank_table_info.h 等优先解析 coll_comm 版，
# 避免解析到 legacy950 镜像版（hcomm.cmake 挂的镜像目录）造成同 SO 跨 TU 两份定义（ODR 违规）。
# 依赖此前插生效的同名头清单（coll_comm 版 vs legacy950 镜像版，同为 namespace Hccl、同 guard 但定义不同，
# 第一批未改名的 6 对，按组待整改）：address_info.h(组15)、control_plane.h(组14)、rank_level_info.h(组17)、
# edge_info.h(组16)、peer_info.h(组9)、phy_topo.h(组18)。
# 任一 -I 顺序调整即静默回退到镜像版；后续批次按组改名/迁 ns 后，相应项移除此清单并加入 odr_coexist 测试分组。
if(TARGET hcomm)
    target_include_directories(hcomm BEFORE PRIVATE ${v2_rank_graph_include_dirs})
endif()

add_subdirectory(common)
add_subdirectory(json)
add_subdirectory(phy_topo)
add_subdirectory(phy_topo_builder)
add_subdirectory(rank_graph)
add_subdirectory(rank_graph_builder)
add_subdirectory(rank_table_info)
add_subdirectory(topo_info)
