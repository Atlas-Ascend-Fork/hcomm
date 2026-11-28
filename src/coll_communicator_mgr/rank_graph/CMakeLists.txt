# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# rank_graph 模块 ODR 整改 namespace 迁移边界说明（issue #1110）：
# coll_comm 侧类是否迁入 namespace hccl，判定规则 = 「该头是否在 legacy/ascend950 镜像目录存在同名头」：
#   - 存在同名镜像头（跨 SO 同名同 guard 同类名，严格 ODR 冲突）→ coll_comm 侧迁 namespace hccl，
#     legacy950 侧加 _legacy950 后缀 + guard 区分（两份 .h 声明一致）。
#     注：不经 bridge 的镜像头（如 AddressInfo/ControlPlane/PeerInfo）仅作布局锁定声明，无对应 .cc，成员函数不可调用；
#     经 bridge 跨 SO 传对象的类（RankTableInfo）两份声明存在成员差异（本侧引用已迁 namespace 的 hccl::NewRankInfo 等，
#     legacy950 侧引用 Hccl::NewRankInfo 且多 Deserialize 等），靠 CMake BEFORE 路径分离避免同 SO 双包含，共用 coll_comm 版 .cc。
#   - 无同名镜像头 → 留 namespace Hccl 不迁。
# 已迁 namespace hccl：topo_info.h(TopoInfo) / phy_topo_builder.h(PhyTopoBuilder) / new_rank_info.h(NewRankInfo) /
# changed_rank_info.h(ChangedRankInfo) / peer_info.h(PeerInfo) / control_plane.h(ControlPlane) /
# address_info.h(AddressInfo) / graph.h(Graph) / edge_info.h(EdgeInfo) / rank_level_info.h(RankLevelInfo) /
# phy_topo.h(PhyTopo)。
# rank_table_info.h(RankTableInfo) 经 bridge 跨 SO 传对象，保持 namespace Hccl（靠 CMake BEFORE 路径分离 +
# legacy950 侧 include 改 _legacy950.h 自洽），不迁 namespace。

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
    # graph.h 已迁 namespace hccl，legacy 侧改名为 graph_legacy950.h（guard GRAPH_LEGACY950_H），全仓唯一 graph.h 为本目录版本，BEFORE 前插保持优先命中
    target_include_directories(ccl_kernel BEFORE PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/common)
endif()

if(TARGET hccl_v2)
    target_include_directories(hccl_v2 BEFORE PRIVATE ${v2_rank_graph_include_dirs})
endif()

# hcomm target 同 hccl_v2 前插 coll_comm rank_graph 目录：裸名 rank_table_info.h 优先解析 coll_comm 版，
# 避免解析到 legacy950 镜像版（hcomm.cmake 挂的镜像目录）造成同 SO 跨 TU 两份定义（ODR 违规）。
# 当前唯一仍依赖此顺序的同名头：rank_table_info.h（两份同名同 guard NEW_RANK_TABLED_H、同 namespace Hccl、定义不同，
# 经 bridge 不迁 namespace，靠 include 路径分离）；其余原同名头（rank_level_info.h/edge_info.h/phy_topo.h）已迁 namespace hccl +
# 改名 _legacy950.h + guard 区分，同 SO 跨 TU 不再两份定义，已加入 odr_coexist 测试分组。
# 任一 -I 顺序调整即静默回退到镜像版；后续若有同名头按本模式（迁 namespace + 改名 + guard 区分）整改，相应项移除此清单并加入 odr_coexist 测试分组。
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
