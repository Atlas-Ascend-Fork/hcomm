/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdio>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "device_spec/device_spec_types.h"
#include "device_spec_yaml_loader.h"

namespace {

using sim::devspec::CcuResType;
using sim::devspec::CcuVersion;
using sim::devspec::DeviceSpec;
using sim::devspec::LoadDeviceSpecYaml;

const std::string kDir = HCCLVM_DEVICE_SPEC_TEST_DIR;

std::string Path(const std::string& name) { return kDir + "/" + name; }

/// 把参考实现复制到临时文件并按给定规则做文本替换，用于构造非法配置。
std::string MakeBroken(const std::string& src, const std::string& from, const std::string& to)
{
    std::ifstream in(src);
    if (!in.is_open()) {
        ADD_FAILURE() << "无法打开源文件: " << src;
        return std::string();
    }
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();
    const size_t pos = content.find(from);
    EXPECT_NE(pos, std::string::npos) << "替换锚点未找到: " << from;
    if (pos != std::string::npos) {
        content.replace(pos, from.size(), to);
    }
    char tmpl[] = "/tmp/device_spec_broken_XXXXXX";
    const int fd = ::mkstemp(tmpl);
    EXPECT_GE(fd, 0);
    if (fd >= 0) {
        ::close(fd);
    }
    std::ofstream out(tmpl);
    out << content;
    out.close();
    return std::string(tmpl);
}

// ===================== 正常解析 =====================

TEST(DeviceSpecLoader, ParseAscend950Ok)
{
    DeviceSpec spec;
    std::string err;
    ASSERT_TRUE(LoadDeviceSpecYaml(Path("device_spec_yaml_demo_ascend950.yaml"), spec, err)) << err;

    EXPECT_EQ(spec.specVersion, 1U);
    EXPECT_EQ(spec.name, "ascend950");
    EXPECT_EQ(spec.ccuVersion, CcuVersion::CCU_V1);
    EXPECT_EQ(spec.dieNum, 2U);
    // 决策 (i)：加载器完全不解析 yaml 的 soc.soc_version
    EXPECT_TRUE(spec.socVersion.empty()) << "soc_version 必须不被解析（决策 (i)）";
    EXPECT_EQ(spec.regions.size(), 9U);
    EXPECT_TRUE(spec.HasGsa());
    EXPECT_TRUE(spec.FindRegion(CcuResType::GSA) != nullptr);
    EXPECT_EQ(spec.resourceSpaceSize, 0x4800000ULL);
    EXPECT_EQ(spec.BackingSize(), 0x4800000ULL);
    EXPECT_EQ(spec.countXnNum, 0U);
    EXPECT_EQ(spec.reservedSyncSlots, 4U);
    EXPECT_FALSE(spec.SupportsAddrDecode()) << "A5(V1) 用 id 编址，不做地址反查";
    // 本期不建模 WQE（D16 / O12）
    EXPECT_EQ(spec.FindRegion(CcuResType::WQE), nullptr);
    EXPECT_EQ(spec.deviceRes.streamNum, 2016U);
    EXPECT_EQ(spec.deviceRes.notifyNum, 131072U);
    EXPECT_EQ(spec.deviceRes.ubJetty.normalPoolSize, 4096U);
    EXPECT_EQ(spec.vectorCoreNum, 64U);
}

TEST(DeviceSpecLoader, ParseAscend960Ok)
{
    DeviceSpec spec;
    std::string err;
    ASSERT_TRUE(LoadDeviceSpecYaml(Path("device_spec_yaml_demo_ascend960.yaml"), spec, err)) << err;

    EXPECT_EQ(spec.specVersion, 1U);
    EXPECT_EQ(spec.name, "ascend960");
    EXPECT_EQ(spec.ccuVersion, CcuVersion::CCU_V2);
    EXPECT_EQ(spec.dieNum, 2U);
    EXPECT_TRUE(spec.socVersion.empty());
    EXPECT_EQ(spec.regions.size(), 8U);
    EXPECT_FALSE(spec.HasGsa()) << "A6 无 GSA 寄存器堆";
    EXPECT_EQ(spec.resourceSpaceSize, 0x4800000ULL);
    EXPECT_EQ(spec.countXnNum, 4U);
    EXPECT_TRUE(spec.SupportsAddrDecode());
    EXPECT_EQ(spec.FindRegion(CcuResType::WQE), nullptr);
    EXPECT_EQ(spec.deviceRes.streamNum, 2016U);
    EXPECT_EQ(spec.deviceRes.notifyNum, 131072U);
    EXPECT_EQ(spec.deviceRes.ubJetty.normalPoolSize, 4096U);
    EXPECT_EQ(spec.vectorCoreNum, 64U);
}

/// A5/A6 的 address_space 段必须逐字段相同（芯片无关，验收 A14）
TEST(DeviceSpecLoader, AddressSpaceIdenticalAcrossChips)
{
    DeviceSpec a5, a6;
    std::string err;
    ASSERT_TRUE(LoadDeviceSpecYaml(Path("device_spec_yaml_demo_ascend950.yaml"), a5, err)) << err;
    ASSERT_TRUE(LoadDeviceSpecYaml(Path("device_spec_yaml_demo_ascend960.yaml"), a6, err)) << err;

    EXPECT_EQ(a5.addr.rankBase, a6.addr.rankBase);
    EXPECT_EQ(a5.addr.rankStride, a6.addr.rankStride);
    EXPECT_EQ(a5.addr.rankWindowSize, a6.addr.rankWindowSize);
    EXPECT_EQ(a5.addr.ccuWindowOffset, a6.addr.ccuWindowOffset);
    EXPECT_EQ(a5.addr.ccuWindowSize, a6.addr.ccuWindowSize);
    EXPECT_EQ(a5.addr.ccuDieSize, a6.addr.ccuDieSize);
    EXPECT_EQ(a5.addr.ccuDieOffsets, a6.addr.ccuDieOffsets);
    ASSERT_EQ(a5.addr.partitions.size(), a6.addr.partitions.size());
    for (size_t i = 0; i < a5.addr.partitions.size(); ++i) {
        EXPECT_EQ(a5.addr.partitions[i].name, a6.addr.partitions[i].name) << "partition " << i;
        EXPECT_EQ(a5.addr.partitions[i].offset, a6.addr.partitions[i].offset) << "partition " << i;
        EXPECT_EQ(a5.addr.partitions[i].size, a6.addr.partitions[i].size) << "partition " << i;
    }
    EXPECT_EQ(a5.addr.userMemTotal, a6.addr.userMemTotal);
    EXPECT_EQ(a5.addr.userMemTotal, 0xFF80000000ULL) << "user_mem 合计须为 1022 GiB";
    EXPECT_EQ(a5.addr.userMemSegs.size(), 2U);
    EXPECT_EQ(a5.resourceSpaceSize, a6.resourceSpaceSize) << "V1/V2 共用 CCU_V1_V2_RESOURCE_SIZE";
}

/// MS 几何参数两代完全相同，仅 ccua_windows 基址不同（设计文档 §4.4.3 修正 3）
TEST(DeviceSpecLoader, MsGeometryIdenticalBaseDiffers)
{
    DeviceSpec a5, a6;
    std::string err;
    ASSERT_TRUE(LoadDeviceSpecYaml(Path("device_spec_yaml_demo_ascend950.yaml"), a5, err)) << err;
    ASSERT_TRUE(LoadDeviceSpecYaml(Path("device_spec_yaml_demo_ascend960.yaml"), a6, err)) << err;

    EXPECT_EQ(a5.ms.msSize, a6.ms.msSize);
    EXPECT_EQ(a5.ms.msSize, 4096U);
    EXPECT_EQ(a5.ms.msNum, a6.ms.msNum);
    EXPECT_EQ(a5.ms.msNum, 1536U);
    EXPECT_EQ(a5.ms.ccuaNum, a6.ms.ccuaNum);
    EXPECT_EQ(a5.ms.ccuaNum, 4U);
    EXPECT_EQ(a5.ms.interleave, a6.ms.interleave);
    EXPECT_EQ(a5.ms.interleave, 8U);
    ASSERT_EQ(a5.ms.windows.size(), 4U);
    ASSERT_EQ(a6.ms.windows.size(), 4U);
    EXPECT_EQ(a5.ms.windows[0].offset, 0x0ULL) << "A5 MS 基址为 0x0（CCUM 在 CCUA 之后）";
    EXPECT_EQ(a6.ms.windows[0].offset, 0x4000000ULL) << "A6 MS 基址为 0x4000000（CCUM 在 CCUA 之前）";
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(a5.ms.windows[i].size, a6.ms.windows[i].size);
        EXPECT_EQ(a5.ms.windows[i].size, 0x200000ULL);
    }
}

/// A5 的 INSTR/GSA/XN/CKE 偏移必须与 hcomm ccu_res_specs 推导值一致
TEST(DeviceSpecLoader, Ascend950RegionOffsetsMatchHcomm)
{
    DeviceSpec spec;
    std::string err;
    ASSERT_TRUE(LoadDeviceSpecYaml(Path("device_spec_yaml_demo_ascend950.yaml"), spec, err)) << err;
    auto offOf = [&](CcuResType t) -> uint64_t {
        const auto* r = spec.FindRegion(t);
        return (r == nullptr) ? UINT64_MAX : r->offset;
    };
    // ccu_res_specs.h:26 CCU_V1_CCUM_OFFSET = 0x800000
    EXPECT_EQ(offOf(CcuResType::INSTR), 0x800000ULL);
    // ccu_res_specs.cc GetXnBaseAddr: CCUM + INS(0x100000) = 0x900000 → GSA
    EXPECT_EQ(offOf(CcuResType::GSA), 0x900000ULL);
    // + GSA_RESERVE(0x8000) = 0x908000
    EXPECT_EQ(offOf(CcuResType::XN), 0x908000ULL);
    // + XN_V1_RESERVE(0x8000) = 0x910000
    EXPECT_EQ(offOf(CcuResType::CKE), 0x910000ULL);
    // XN 预留 A5=32K vs A6=256K（ccu_res_specs.h:55-56）
    EXPECT_EQ(spec.FindRegion(CcuResType::XN)->size, 0x8000ULL);
}

TEST(DeviceSpecLoader, Ascend960RegionOffsetsMatchHcomm)
{
    DeviceSpec spec;
    std::string err;
    ASSERT_TRUE(LoadDeviceSpecYaml(Path("device_spec_yaml_demo_ascend960.yaml"), spec, err)) << err;
    auto offOf = [&](CcuResType t) -> uint64_t {
        const auto* r = spec.FindRegion(t);
        return (r == nullptr) ? UINT64_MAX : r->offset;
    };
    // ccu_res_specs.h:27 CCU_V2_CCUM_OFFSET = 0x0
    EXPECT_EQ(offOf(CcuResType::INSTR), 0x0ULL);
    // GetXnBaseAddr V2: 0 + INS(0x100000)
    EXPECT_EQ(offOf(CcuResType::XN), 0x100000ULL);
    // GetCkeBaseAddr V2: + XN_V2_RESERVE(0x40000)
    EXPECT_EQ(offOf(CcuResType::CKE), 0x140000ULL);
    EXPECT_EQ(spec.FindRegion(CcuResType::XN)->size, 0x40000ULL);
    EXPECT_EQ(spec.FindRegion(CcuResType::XN)->count, 16384U);
}

// ===================== GetCaps 交叉验证（验收 A15）=====================

TEST(DeviceSpecCaps, Ascend950MatchesHccpStubV1)
{
    DeviceSpec spec;
    std::string err;
    ASSERT_TRUE(LoadDeviceSpecYaml(Path("device_spec_yaml_demo_ascend950.yaml"), spec, err)) << err;
    const auto caps = spec.GetCaps();
    // 期望值取自 src/proxy/level2/hccp_ccu_stub.cc:69-96 SetCcuV1ResourceBasicInfo
    EXPECT_EQ(caps.instructionNum, 0x8000U);
    EXPECT_EQ(caps.missionNum, 16U);
    EXPECT_EQ(caps.loopEngineNum, 200U);
    EXPECT_EQ(caps.gsaNum, 3072U);
    EXPECT_EQ(caps.xnNum, 3072U);
    EXPECT_EQ(caps.ckeNum, 1024U);
    EXPECT_EQ(caps.msNum, 1536U);
    EXPECT_EQ(caps.channelNum, 128U);
    EXPECT_EQ(caps.jettyNum, 128U);
    EXPECT_EQ(caps.pfeNum, 16U);
}

TEST(DeviceSpecCaps, Ascend960MatchesHccpStubV2)
{
    DeviceSpec spec;
    std::string err;
    ASSERT_TRUE(LoadDeviceSpecYaml(Path("device_spec_yaml_demo_ascend960.yaml"), spec, err)) << err;
    const auto caps = spec.GetCaps();
    // 期望值取自 src/proxy/level2/hccp_ccu_stub.cc:98-125 SetCcuV2ResourceBasicInfo
    EXPECT_EQ(caps.instructionNum, 0x8000U);
    EXPECT_EQ(caps.missionNum, 16U);
    EXPECT_EQ(caps.loopEngineNum, 512U);
    EXPECT_EQ(caps.gsaNum, 0U) << "A6 无 GSA region → 派生为 0";
    EXPECT_EQ(caps.xnNum, 16384U);
    EXPECT_EQ(caps.ckeNum, 1024U);
    EXPECT_EQ(caps.msNum, 1536U);
    EXPECT_EQ(caps.channelNum, 1024U);
    EXPECT_EQ(caps.jettyNum, 128U);
    EXPECT_EQ(caps.pfeNum, 20U);
}

// ===================== 派生量 =====================

TEST(DeviceSpecDerived, MaxModelledEndAndBackwardCompatibility)
{
    DeviceSpec a5, a6;
    std::string err;
    ASSERT_TRUE(LoadDeviceSpecYaml(Path("device_spec_yaml_demo_ascend950.yaml"), a5, err)) << err;
    ASSERT_TRUE(LoadDeviceSpecYaml(Path("device_spec_yaml_demo_ascend960.yaml"), a6, err)) << err;

    // A5: regions 末端 0x940000；MS 末端 0x800000 → max = 0x940000（9.25 MiB）
    EXPECT_EQ(a5.MaxModelledEnd(), 0x940000ULL);
    // A6: regions 末端 0x188000；MS 末端 0x4800000 → max = 0x4800000（与权威值巧合相等）
    EXPECT_EQ(a6.MaxModelledEnd(), 0x4800000ULL);
    // ★ backingSize 取权威值，不是 max —— A5 上两者相差 62.75 MiB（D10-补）
    EXPECT_EQ(a5.BackingSize(), 0x4800000ULL);
    EXPECT_GT(a5.BackingSize(), a5.MaxModelledEnd());
    EXPECT_EQ(a5.BackingSize() - a5.MaxModelledEnd(), 0x4800000ULL - 0x940000ULL);
    EXPECT_EQ(a6.BackingSize(), a6.MaxModelledEnd());
}

// ===================== 校验失败路径 =====================

TEST(DeviceSpecValidation, RejectsInvalidDeviceRes)
{
    const std::vector<std::pair<std::string, std::string>> mutations = {
        {"stream_num: 2016", "stream_num: 0"},
        {"notify_num: 131072", "notify_num: 131073"},
        {"vector_core_num: 64", "vector_core_num: 0"},
    };
    for (const auto& mutation : mutations) {
        const auto f = MakeBroken(Path("device_spec_yaml_demo_ascend960.yaml"), mutation.first, mutation.second);
        DeviceSpec spec;
        std::string err;
        EXPECT_FALSE(LoadDeviceSpecYaml(f, spec, err)) << mutation.first;
        EXPECT_FALSE(err.empty());
        ::remove(f.c_str());
    }
}

TEST(DeviceSpecValidation, RejectsMissingFile)
{
    DeviceSpec spec;
    std::string err;
    EXPECT_FALSE(LoadDeviceSpecYaml(Path("no_such_file.yaml"), spec, err));
    EXPECT_FALSE(err.empty());
}

TEST(DeviceSpecValidation, RejectsRegionOverlap)
{
    // 把 cke 的 offset 改成与 xn 重叠
    const auto f = MakeBroken(
        Path("device_spec_yaml_demo_ascend960.yaml"), R"(type: "CKE",      offset: 0x140000)",
        R"(type: "CKE",      offset: 0x120000)");
    DeviceSpec spec;
    std::string err;
    EXPECT_FALSE(LoadDeviceSpecYaml(f, spec, err));
    EXPECT_NE(err.find("校验1"), std::string::npos) << err;
    ::remove(f.c_str());
}

/// 校验 1 必须把 regions 与 ms.ccua_windows **合并**检查
TEST(DeviceSpecValidation, RejectsRegionOverlappingMsWindow)
{
    // A5 的 MS 占 [0, 0x800000)；把 INSTR 挪到 0x400000 与之重叠
    const auto f = MakeBroken(
        Path("device_spec_yaml_demo_ascend950.yaml"), R"(type: "INSTR",   offset: 0x800000)",
        R"(type: "INSTR",   offset: 0x400000)");
    DeviceSpec spec;
    std::string err;
    EXPECT_FALSE(LoadDeviceSpecYaml(f, spec, err));
    EXPECT_NE(err.find("校验1"), std::string::npos) << err;
    ::remove(f.c_str());
}

TEST(DeviceSpecValidation, RejectsEntrySizeTimesCountOverSize)
{
    // xn: entry_size 8 × count 16384 = 0x20000 > size 0x10000
    const auto f = MakeBroken(
        Path("device_spec_yaml_demo_ascend960.yaml"), R"(type: "XN",       offset: 0x100000,  size: 0x40000)",
        R"(type: "XN",       offset: 0x100000,  size: 0x10000)");
    DeviceSpec spec;
    std::string err;
    EXPECT_FALSE(LoadDeviceSpecYaml(f, spec, err));
    EXPECT_NE(err.find("校验2"), std::string::npos) << err;
    ::remove(f.c_str());
}

/**
 * 校验 3 的异构窗口反例：求和式会漏判，per-window 才能抓住。
 * 把 A6 的 ccua_windows[0] 缩小到 0x100000（256 槽），其余保持 0x200000（512 槽）：
 *   Σ 槽数 = 256+512*3 = 1792 ≥ ms_num 1536  → 求和式通过
 *   localIdx_max = 383 > 255                 → per-window 失败
 */
TEST(DeviceSpecValidation, RejectsHeterogeneousMsWindowOverflow)
{
    const auto f = MakeBroken(
        Path("device_spec_yaml_demo_ascend960.yaml"), R"(- {index: 0, offset: 0x4000000, size: 0x200000})",
        R"(- {index: 0, offset: 0x4000000, size: 0x100000})");
    DeviceSpec spec;
    std::string err;
    EXPECT_FALSE(LoadDeviceSpecYaml(f, spec, err));
    EXPECT_NE(err.find("校验3"), std::string::npos) << err;
    ::remove(f.c_str());
}

TEST(DeviceSpecValidation, RejectsCcuaNumMismatch)
{
    const auto f = MakeBroken(Path("device_spec_yaml_demo_ascend960.yaml"), "ccua_num:   4", "ccua_num:   8");
    DeviceSpec spec;
    std::string err;
    EXPECT_FALSE(LoadDeviceSpecYaml(f, spec, err));
    EXPECT_NE(err.find("校验3"), std::string::npos) << err;
    ::remove(f.c_str());
}

TEST(DeviceSpecValidation, RejectsDieOffsetsCountMismatch)
{
    const auto f = MakeBroken(
        Path("device_spec_yaml_demo_ascend960.yaml"), "die_offsets: [0x0, 0x40000000]", "die_offsets: [0x0]");
    DeviceSpec spec;
    std::string err;
    EXPECT_FALSE(LoadDeviceSpecYaml(f, spec, err));
    EXPECT_NE(err.find("校验4"), std::string::npos) << err;
    ::remove(f.c_str());
}

TEST(DeviceSpecValidation, RejectsDieWindowOverlap)
{
    const auto f = MakeBroken(
        Path("device_spec_yaml_demo_ascend960.yaml"), "die_offsets: [0x0, 0x40000000]",
        "die_offsets: [0x0, 0x20000000]");
    DeviceSpec spec;
    std::string err;
    EXPECT_FALSE(LoadDeviceSpecYaml(f, spec, err));
    EXPECT_NE(err.find("校验4"), std::string::npos) << err;
    ::remove(f.c_str());
}

/// 校验 5：partitions 必须完整铺满窗口
TEST(DeviceSpecValidation, RejectsPartitionsNotTiling)
{
    // 段1 size 少 1 → 末端不达 rank_window_size
    const auto f
        = MakeBroken(Path("device_spec_yaml_demo_ascend960.yaml"), "size: 0xF180000000}", "size: 0xF17FFFFFFF}");
    DeviceSpec spec;
    std::string err;
    EXPECT_FALSE(LoadDeviceSpecYaml(f, spec, err));
    EXPECT_NE(err.find("校验5"), std::string::npos) << err;
    ::remove(f.c_str());
}

TEST(DeviceSpecValidation, RejectsPartitionGap)
{
    // 段1 起点后移 → 与 ccu 段之间出现空洞
    const auto f = MakeBroken(
        Path("device_spec_yaml_demo_ascend960.yaml"), R"({name: "user_mem",  offset: 0xE80000000,)",
        R"({name: "user_mem",  offset: 0xE90000000,)");
    DeviceSpec spec;
    std::string err;
    EXPECT_FALSE(LoadDeviceSpecYaml(f, spec, err));
    EXPECT_NE(err.find("校验5"), std::string::npos) << err;
    ::remove(f.c_str());
}

TEST(DeviceSpecValidation, RejectsMissingCcuPartition)
{
    const auto f = MakeBroken(
        Path("device_spec_yaml_demo_ascend960.yaml"), R"({name: "ccu",       offset: 0xE00000000,  size: 0x80000000})",
        R"({name: "user_mem",  offset: 0xE00000000,  size: 0x80000000})");
    DeviceSpec spec;
    std::string err;
    EXPECT_FALSE(LoadDeviceSpecYaml(f, spec, err));
    EXPECT_NE(err.find("校验5"), std::string::npos) << err;
    ::remove(f.c_str());
}

TEST(DeviceSpecValidation, RejectsUnknownPartitionName)
{
    const auto f = MakeBroken(
        Path("device_spec_yaml_demo_ascend960.yaml"), R"({name: "ccu",       offset: 0xE00000000,  size: 0x80000000})",
        R"({name: "aiv_ub",    offset: 0xE00000000,  size: 0x80000000})");
    DeviceSpec spec;
    std::string err;
    EXPECT_FALSE(LoadDeviceSpecYaml(f, spec, err));
    EXPECT_NE(err.find("白名单"), std::string::npos) << err;
    ::remove(f.c_str());
}

TEST(DeviceSpecValidation, RejectsCcuPartitionTooSmallForDies)
{
    const auto f = MakeBroken(
        Path("device_spec_yaml_demo_ascend960.yaml"), R"(size: 0x80000000}    #       56 ~   58 GiB（2 die × 1 GiB）)",
        R"(size: 0x40000000}    #       56 ~   57 GiB（只装得下 1 die）)");
    DeviceSpec spec;
    std::string err;
    EXPECT_FALSE(LoadDeviceSpecYaml(f, spec, err));
    EXPECT_NE(err.find("校验5"), std::string::npos) << err;
    ::remove(f.c_str());
}

/// 校验 8：max(regions ∪ ms 末端) ≤ resource_space_size
TEST(DeviceSpecValidation, RejectsResourceSpaceSizeTooSmall)
{
    // 缩到 0x4000000（64 MiB）< MS 末端 0x4800000
    const auto f = MakeBroken(
        Path("device_spec_yaml_demo_ascend960.yaml"), "resource_space_size: 0x4800000",
        "resource_space_size: 0x4000000");
    DeviceSpec spec;
    std::string err;
    EXPECT_FALSE(LoadDeviceSpecYaml(f, spec, err));
    EXPECT_NE(err.find("校验8"), std::string::npos) << err;
    ::remove(f.c_str());
}

TEST(DeviceSpecValidation, RejectsResourceSpaceSizeOverDieSize)
{
    const auto f = MakeBroken(
        Path("device_spec_yaml_demo_ascend960.yaml"), "resource_space_size: 0x4800000",
        "resource_space_size: 0x80000000");
    DeviceSpec spec;
    std::string err;
    EXPECT_FALSE(LoadDeviceSpecYaml(f, spec, err));
    EXPECT_NE(err.find("校验8"), std::string::npos) << err;
    ::remove(f.c_str());
}

TEST(DeviceSpecValidation, RejectsUnknownRegionType)
{
    const auto f = MakeBroken(Path("device_spec_yaml_demo_ascend960.yaml"), R"(type: "PFE",)", R"(type: "PFEX",)");
    DeviceSpec spec;
    std::string err;
    EXPECT_FALSE(LoadDeviceSpecYaml(f, spec, err));
    EXPECT_NE(err.find("非法取值"), std::string::npos) << err;
    ::remove(f.c_str());
}

TEST(DeviceSpecValidation, RejectsMsAsRegion)
{
    const auto f = MakeBroken(
        Path("device_spec_yaml_demo_ascend960.yaml"), R"(type: "PFE",      offset: 0x148000)",
        R"(type: "MS",       offset: 0x148000)");
    DeviceSpec spec;
    std::string err;
    EXPECT_FALSE(LoadDeviceSpecYaml(f, spec, err));
    EXPECT_NE(err.find("MS 应在"), std::string::npos) << err;
    ::remove(f.c_str());
}

TEST(DeviceSpecValidation, RejectsDuplicateRegionType)
{
    const auto f = MakeBroken(
        Path("device_spec_yaml_demo_ascend960.yaml"), R"(type: "JETTY",    offset: 0x170000)",
        R"(type: "XN",       offset: 0x170000)");
    DeviceSpec spec;
    std::string err;
    EXPECT_FALSE(LoadDeviceSpecYaml(f, spec, err));
    // 可能先被校验1（重叠）或重复 type 拦下，两者都算正确 fail-fast
    EXPECT_FALSE(err.empty()) << err;
    ::remove(f.c_str());
}

TEST(DeviceSpecValidation, RejectsUnsupportedDieNum)
{
    const auto f = MakeBroken(Path("device_spec_yaml_demo_ascend960.yaml"), "die_num: 2 ", "die_num: 4 ");
    DeviceSpec spec;
    std::string err;
    EXPECT_FALSE(LoadDeviceSpecYaml(f, spec, err));
    EXPECT_NE(err.find("O4"), std::string::npos) << err;
    ::remove(f.c_str());
}

TEST(DeviceSpecValidation, RejectsUnknownCcuVersion)
{
    const auto f = MakeBroken(
        Path("device_spec_yaml_demo_ascend960.yaml"), R"(ccu_version: "CCU_V2")", R"(ccu_version: "CCU_V9")");
    DeviceSpec spec;
    std::string err;
    EXPECT_FALSE(LoadDeviceSpecYaml(f, spec, err));
    EXPECT_NE(err.find("ccu_version"), std::string::npos) << err;
    ::remove(f.c_str());
}

TEST(DeviceSpecValidation, RejectsMissingRequiredField)
{
    const auto f = MakeBroken(
        Path("device_spec_yaml_demo_ascend960.yaml"), "  rank_window_size: 0x10000000000",
        "  # rank_window_size 被删除");
    DeviceSpec spec;
    std::string err;
    EXPECT_FALSE(LoadDeviceSpecYaml(f, spec, err));
    EXPECT_NE(err.find("rank_window_size"), std::string::npos) << err;
    ::remove(f.c_str());
}

/// ★ 大数值必须用 uint64 读取：把 rank_stride 换成 uint32 装不下的值时仍应成功解析，
///   证明加载器用的是 uint64 路径（设计文档 §4.2.2 解析注意）
TEST(DeviceSpecValidation, ParsesValuesAbove32Bit)
{
    DeviceSpec spec;
    std::string err;
    ASSERT_TRUE(LoadDeviceSpecYaml(Path("device_spec_yaml_demo_ascend960.yaml"), spec, err)) << err;
    EXPECT_GT(spec.addr.rankStride, static_cast<uint64_t>(UINT32_MAX));
    EXPECT_GT(spec.addr.rankBase, static_cast<uint64_t>(UINT32_MAX));
    ASSERT_EQ(spec.addr.userMemSegs.size(), 2U);
    EXPECT_GT(spec.addr.userMemSegs[1].size, static_cast<uint64_t>(UINT32_MAX));
    EXPECT_GT(spec.addr.userMemTotal, static_cast<uint64_t>(UINT32_MAX));
}

TEST(DeviceSpecValidation, RejectsNameFileNameMismatch)
{
    // name 改成 ascend950 但文件名仍是 ascend960 → 由 Registry 拦截；
    // 这里只验证 loader 本身接受（name 与文件名的一致性检查在 Registry::LoadFromFile）
    const auto f
        = MakeBroken(Path("device_spec_yaml_demo_ascend960.yaml"), R"(name: "ascend960")", R"(name: "ascend950")");
    DeviceSpec spec;
    std::string err;
    EXPECT_TRUE(LoadDeviceSpecYaml(f, spec, err)) << err;
    EXPECT_EQ(spec.name, "ascend950");
    ::remove(f.c_str());
}

} // namespace
