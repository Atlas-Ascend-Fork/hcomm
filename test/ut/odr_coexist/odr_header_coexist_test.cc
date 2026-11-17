/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file odr_header_coexist_test.cc
 * @brief ODR 整改回归测试：验证可分离 ns 的整改对（组3/4/5/7）两侧同名头能在同一 TU 共存不重定义。
 *
 * 本测试为编译级测试（无需运行断言）。组3/4/5/7 两侧头文件能同时被一个 TU 包含且编译通过，
 * 即证明 namespace/guard 分离生效、同名类已隔离到不同 ns。
 * 组8（情况1：同 ns 同类名，文件名/guard 区分）与组11（同 guard 同 ns，靠 include 路径分离）受架构约束
 * 不能在同一 TU 同时 include 两份，仅 include 单份验证可编译，为已知边界（见各组处注释）。
 * 后续组整改 PR 沿用本文件逐个加对。
 */

// 组3 topo_info.h：coll_comm 侧（namespace hccl）+ legacy950 侧（namespace Hccl）
#include <cstddef> // offsetof
#include "topo_info.h"
#include "topo_info_legacy950.h"

// 组4 phy_topo_builder.h：coll_comm 侧（namespace hccl）+ legacy950 侧（namespace Hccl）
#include "phy_topo_builder.h"
#include "phy_topo_builder_legacy950.h"

// 组5 new_rank_info.h：coll_comm 侧（namespace hccl）+ legacy950 侧（namespace Hccl）
#include "new_rank_info.h"
#include "new_rank_info_legacy950.h"

// 组7 changed_rank_info.h：coll_comm 侧（namespace hccl）+ legacy950 侧（namespace Hccl）
#include "changed_rank_info.h"
#include "changed_rank_info_legacy950.h"

// 组8 zero_copy_address_mgr.h：legacy910 两份（情况1：同 namespace hccl 同类名，文件名/guard 区分）
// 注意：情况1 整改规则要求 ns 不变、仅文件名/guard 区分。两份同类名同 namespace，同 TU 同时 include 会重定义
// （guard 区分只能保证不同 TU 各自展开一份，不能保证同 TU 两份共存）。
// 这是情况1（同 ns 同类名）的已知边界（同 TU 不能共存）。
// 情况1 仅解决 include 歧义、不解决定义一致性：同 ns 同类名两份头若声明有差异，同 SO 内跨 TU 仍 ODR 违规。
// 本批已将两份声明逐字对齐（除 guard 外完全一致），保证同 SO 各 TU 看到相同定义；
// 彻底消除（dev 侧独立 namespace/类名）属后续组根治方向。
// 此处仅 include 一份（comm 版），验证其可独立编译。
#include "zero_copy_address_mgr_legacy910_comm.h"
// #include "zero_copy_address_mgr_legacy910_dev.h"  // 情况1：同 ns 同类名，同 TU 不能共存

// 组11 rank_table_info.h：coll_comm 侧 + legacy950 侧（同名同 guard 同 namespace Hccl，靠 include 路径分离）
// 情况2 声明对齐：两份声明归一化 namespace 后一致，但 ranks 元素类型字面不同——
// coll_comm 侧为 vector<hccl::NewRankInfo>（组5 已迁 ns hccl），legacy950 侧为 vector<Hccl::NewRankInfo>。
// 两份同名同 guard（NEW_RANK_TABLED_H）同 ns（Hccl），同 TU 同时 include 会重定义（第二次被 guard 跳过），
// 且 ranks 元素类型不同构成已知 ODR 边界（根治需将组11 迁 ns，会动 bridge，本批不处理）。
// 此处仅 include 一份（coll_comm 版），验证其可独立编译。生产靠 include 路径分离：hccl_v2 与 libhcomm
// 各自解析对应版本。
#include "rank_table_info.h"

// 引用两侧符号，确保两侧类型均可见；并断言两侧布局一致（固化"声明对齐"约束，防单边修改导致跨 SO 布局分叉）。
// 组3/4/5/7 两份声明完全一致，用 sizeof== 断言总大小一致 + offsetof 断言序列化成员偏移一致
// （sizeof== 只校验总大小，offsetof 校验成员偏移，防成员顺序交换时 sizeof 不变但偏移漂移破坏序列化）；
// 组8/组11 为已知边界，无法断言布局一致（见各处注释）。
static void UseOdrSymbols()
{
    // 组3：两侧 hccl::TopoInfo 分属 hccl / Hccl，不冲突；断言两侧大小 + 序列化成员偏移一致
    static_assert(sizeof(hccl::TopoInfo) > 0, "hccl::TopoInfo must be defined");
    static_assert(sizeof(Hccl::TopoInfo) > 0, "Hccl::TopoInfo must be defined");
    static_assert(sizeof(hccl::TopoInfo) == sizeof(Hccl::TopoInfo), "TopoInfo layout must match");
    static_assert(
        offsetof(hccl::TopoInfo, version) == offsetof(Hccl::TopoInfo, version), "TopoInfo version offset must match");
    static_assert(
        offsetof(hccl::TopoInfo, peerCount) == offsetof(Hccl::TopoInfo, peerCount),
        "TopoInfo peerCount offset must match");
    static_assert(
        offsetof(hccl::TopoInfo, edgeCount) == offsetof(Hccl::TopoInfo, edgeCount),
        "TopoInfo edgeCount offset must match");
    static_assert(
        offsetof(hccl::TopoInfo, peers) == offsetof(Hccl::TopoInfo, peers), "TopoInfo peers offset must match");
    static_assert(
        offsetof(hccl::TopoInfo, edges) == offsetof(Hccl::TopoInfo, edges), "TopoInfo edges offset must match");

    // 组4：两侧 hccl::PhyTopoBuilder 分属 hccl / Hccl；仅断言 sizeof==（PhyTopoBuilder 成员多为 private，
    // offsetof 不适用）。sizeof== 只校验成员集合尺寸，无法捕获成员顺序/类型漂移；该类不参与跨 SO 序列化，风险有限。
    static_assert(sizeof(hccl::PhyTopoBuilder) > 0, "hccl::PhyTopoBuilder must be defined");
    static_assert(sizeof(Hccl::PhyTopoBuilder) > 0, "Hccl::PhyTopoBuilder must be defined");
    static_assert(sizeof(hccl::PhyTopoBuilder) == sizeof(Hccl::PhyTopoBuilder), "PhyTopoBuilder layout must match");

    // 组5：两侧 hccl::NewRankInfo 分属 hccl / Hccl；断言两侧大小 + 序列化成员偏移一致
    static_assert(sizeof(hccl::NewRankInfo) > 0, "hccl::NewRankInfo must be defined");
    static_assert(sizeof(Hccl::NewRankInfo) > 0, "Hccl::NewRankInfo must be defined");
    static_assert(sizeof(hccl::NewRankInfo) == sizeof(Hccl::NewRankInfo), "NewRankInfo layout must match");
    static_assert(
        offsetof(hccl::NewRankInfo, rankId) == offsetof(Hccl::NewRankInfo, rankId),
        "NewRankInfo rankId offset must match");
    static_assert(
        offsetof(hccl::NewRankInfo, deviceId) == offsetof(Hccl::NewRankInfo, deviceId),
        "NewRankInfo deviceId offset must match");
    static_assert(
        offsetof(hccl::NewRankInfo, localId) == offsetof(Hccl::NewRankInfo, localId),
        "NewRankInfo localId offset must match");
    static_assert(
        offsetof(hccl::NewRankInfo, replacedLocalId) == offsetof(Hccl::NewRankInfo, replacedLocalId),
        "NewRankInfo replacedLocalId offset must match");
    static_assert(
        offsetof(hccl::NewRankInfo, devicePort) == offsetof(Hccl::NewRankInfo, devicePort),
        "NewRankInfo devicePort offset must match");
    static_assert(
        offsetof(hccl::NewRankInfo, hostPort) == offsetof(Hccl::NewRankInfo, hostPort),
        "NewRankInfo hostPort offset must match");
    static_assert(
        offsetof(hccl::NewRankInfo, rankLevelInfos) == offsetof(Hccl::NewRankInfo, rankLevelInfos),
        "NewRankInfo rankLevelInfos offset must match");
    static_assert(
        offsetof(hccl::NewRankInfo, controlPlane) == offsetof(Hccl::NewRankInfo, controlPlane),
        "NewRankInfo controlPlane offset must match");
    static_assert(
        offsetof(hccl::NewRankInfo, tlsStatus) == offsetof(Hccl::NewRankInfo, tlsStatus),
        "NewRankInfo tlsStatus offset must match");
    static_assert(
        offsetof(hccl::NewRankInfo, hostDpuTlsStatus) == offsetof(Hccl::NewRankInfo, hostDpuTlsStatus),
        "NewRankInfo hostDpuTlsStatus offset must match");

    // 组7：两侧 hccl::ChangedRankInfo 分属 hccl / Hccl；断言两侧大小 + 序列化成员偏移一致
    static_assert(sizeof(hccl::ChangedRankInfo) > 0, "hccl::ChangedRankInfo must be defined");
    static_assert(sizeof(Hccl::ChangedRankInfo) > 0, "Hccl::ChangedRankInfo must be defined");
    static_assert(sizeof(hccl::ChangedRankInfo) == sizeof(Hccl::ChangedRankInfo), "ChangedRankInfo layout must match");
    static_assert(
        offsetof(hccl::ChangedRankInfo, version) == offsetof(Hccl::ChangedRankInfo, version),
        "ChangedRankInfo version offset must match");
    static_assert(
        offsetof(hccl::ChangedRankInfo, rankCount) == offsetof(Hccl::ChangedRankInfo, rankCount),
        "ChangedRankInfo rankCount offset must match");
    static_assert(
        offsetof(hccl::ChangedRankInfo, ranks) == offsetof(Hccl::ChangedRankInfo, ranks),
        "ChangedRankInfo ranks offset must match");

    // 组8：情况1 整改（同 ns 只改文件名/guard），同 TU 不能同时 include 两份，仅验证 comm 版可编译（已知边界）。
    // 两份声明已逐字对齐（除 guard 外一致），但因同 TU 不共存、无法在同 TU 断言布局一致。
    static_assert(sizeof(hccl::ZeroCopyAddressMgr) > 0, "hccl::ZeroCopyAddressMgr must be defined");

    // 组11：情况2 声明对齐（同 ns Hccl 同 guard，靠 include 路径分离；ranks 元素类型字面不同，已知边界），
    // 仅验证 coll_comm 版可编译。单份 include 依赖 CMake include 路径顺序解析到 coll_comm 版
    // （UT_COMMON_INCLUDE_LIST 含 coll_comm rank_table_info 目录且先于 legacy950 目录），若目录顺序调整
    // 可能解析到 legacy950 版而仍通过，失去回归意义。ranks 元素类型不同（hccl::NewRankInfo vs
    // Hccl::NewRankInfo）导致两份声明不完全一致，无法断言布局一致；根治需组11 迁 ns，会动 bridge，本批不处理。
    static_assert(sizeof(Hccl::RankTableInfo) > 0, "Hccl::RankTableInfo must be defined");
}

int main()
{
    // 编译级测试：组3/4/5/7 两侧同 TU 共存编译通过即证明 ns/guard 分离生效；
    // 组8/组11 仅单份冒烟（已知边界，不证明 ODR 隔离），详见各组处注释。
    UseOdrSymbols();
    return 0;
}
