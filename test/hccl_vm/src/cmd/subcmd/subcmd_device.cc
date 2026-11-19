/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "subcmd_device.h"

#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include "device_spec/ccu_address_space.h"
#include "device_spec/device_spec_registry.h"
#include "device_spec/rank_address_space.h"
#include "sim_log.h"

namespace {

/// 解析用户输入的十六进制/十进制地址。支持 0x 前缀。
bool ParseUint64(const std::string& s, uint64_t& out)
{
    if (s.empty()) {
        return false;
    }
    try {
        size_t pos = 0;
        const int base = (s.size() > 2 && s.compare(0, 2, "0x") == 0) ? 16 : 10;
        out = std::stoull(s, &pos, base);
        return pos == s.size();
    } catch (const std::exception&) {
        return false;
    }
}

std::string Hex(uint64_t v)
{
    std::ostringstream oss;
    oss << "0x" << std::hex << v;
    return oss.str();
}

std::string JsonEscape(const std::string& s)
{
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        if (c == '"' || c == '\\') {
            out.push_back('\\');
            out.push_back(c);
        } else if (c == '\n') {
            out += "\\n";
        } else {
            out.push_back(c);
        }
    }
    return out;
}

} // namespace

namespace HcclSim {

using sim::devspec::CcuAddressSpace;
using sim::devspec::CcuResType;
using sim::devspec::CcuResTypeFromString;
using sim::devspec::CcuResTypeToString;
using sim::devspec::CcuVersionToString;
using sim::devspec::DeviceSpec;
using sim::devspec::DeviceSpecRegistry;
using sim::devspec::RankAddressSpace;

void DeviceCommand::Setup(CLI::App& app)
{
    auto sub = app.add_subcommand("device", "device 规格查询与 CCU/user_mem 地址编解码");
    sub->require_subcommand(1);

    // ---- device spec ----
    auto spec = sub->add_subcommand("spec", "device 规格（config/device/*.yaml）");
    spec->require_subcommand(1);

    auto show = spec->add_subcommand("show", "打印解析后的规格与校验结果");
    show->add_option("soc", m_soc, "soc_version 或 spec 名（如 Ascend950DT_95A1 / ascend950）；省略则列出全部");
    show->add_option("--dir", m_specDir, "覆盖 device 配置目录（默认 <install_root>/config/device）");
    show->callback([this]() {
        ShowSpec();
    });

    auto exp = spec->add_subcommand("export", "导出 JSON，供第三方消费");
    exp->add_option("soc", m_soc, "soc_version 或 spec 名")->required();
    exp->add_option("-o,--output", m_outFile, "输出文件；省略则打印到 stdout");
    exp->add_option("--dir", m_specDir, "覆盖 device 配置目录");
    exp->callback([this]() {
        ExportSpec();
    });

    // ---- device addr ----
    auto addr = sub->add_subcommand("addr", "CCU 资源地址编解码");
    addr->require_subcommand(1);

    auto dec = addr->add_subcommand("decode", "设备 VA → (rank, die, region, id)");
    dec->add_option("va", m_addrStr, "设备虚拟地址，如 0x8e00140008")->required();
    dec->add_option(
        "--soc", m_soc,
        "限定 spec（soc_version 或 spec 名）。★ 省略时会遍历全部已加载 spec —— "
        "A5/A6 的 address_space 按设计完全相同，同一个 VA 在不同芯片下解码结果不同，"
        "故 VA 本身不足以判定芯片型号");
    dec->add_option("--dir", m_specDir, "覆盖 device 配置目录");
    dec->callback([this]() {
        DecodeAddr();
    });

    auto enc = addr->add_subcommand("encode", "(rank, die, region, id) → 设备 VA");
    enc->add_option("--rank", m_rank, "rank 索引（= Device.physical_id - 1）")->required();
    enc->add_option("--die", m_die, "die id")->required();
    enc->add_option("--type", m_resType, "资源类型：INSTR|GSA|XN|CKE|PFE|CHANNEL|JETTY|MISSION|LOOP|MS")->required();
    enc->add_option("--id", m_resId, "资源 id")->required();
    enc->add_option("--soc", m_soc, "soc_version 或 spec 名；省略则用唯一已加载的 spec");
    enc->add_option("--dir", m_specDir, "覆盖 device 配置目录");
    enc->callback([this]() {
        EncodeAddr();
    });
}

bool DeviceCommand::EnsureLoaded()
{
    auto& reg = DeviceSpecRegistry::GetInstance();
    if (reg.Size() == 0 && !reg.LoadAll(m_specDir)) {
        std::cerr << "[HVM] [ERROR] 加载 device 规格失败: " << reg.LastError() << std::endl;
        return false;
    }
    if (reg.Size() == 0) {
        std::cerr << "[HVM] [ERROR] 未找到任何 device 规格文件" << std::endl;
        return false;
    }
    return true;
}

/// 按 soc_version 或 spec 名解析出目标 spec；soc 为空且只有一份时返回那一份。
static const DeviceSpec* ResolveSpec(const std::string& socOrName)
{
    auto& reg = DeviceSpecRegistry::GetInstance();
    if (socOrName.empty()) {
        const auto all = reg.All();
        if (all.size() == 1) {
            return all.front();
        }
        return nullptr;
    }
    if (const DeviceSpec* s = reg.FindByName(socOrName)) {
        return s;
    }
    return reg.FindBySocVersion(socOrName);
}

static void PrintSpec(std::ostream& os, const DeviceSpec& spec)
{
    const RankAddressSpace ras(spec.addr);
    os << "=== device spec: " << spec.name << " ===\n";
    os << "  source            : " << spec.sourcePath << "\n";
    os << "  spec_version      : " << spec.specVersion << "\n";
    os << "  ccu_version       : " << CcuVersionToString(spec.ccuVersion) << "\n";
    os << "  die_num           : " << spec.dieNum << "\n";
    os << "  soc_version(yaml) : " << (spec.socVersion.empty() ? "<加载器不解析，决策 (i)>" : spec.socVersion) << "\n";
    os << "  supports_addr_decode : " << (spec.SupportsAddrDecode() ? "yes" : "no (V1 用 id 编址)") << "\n";

    os << "\n  -- address_space --\n";
    os << "  rank_base         : " << Hex(spec.addr.rankBase) << "\n";
    os << "  rank_stride       : " << Hex(spec.addr.rankStride) << "\n";
    os << "  rank_window_size  : " << Hex(spec.addr.rankWindowSize) << "\n";
    os << "  partitions:\n";
    for (const auto& p : spec.addr.partitions) {
        os << "    - " << std::left << std::setw(10) << p.name << " offset=" << Hex(p.offset) << " size=" << Hex(p.size)
           << " (" << (p.size >> 30) << " GiB)\n";
    }
    os << "  ccu_window        : offset=" << Hex(spec.addr.ccuWindowOffset) << " size=" << Hex(spec.addr.ccuWindowSize)
       << "\n";
    os << "  ccu_die_size      : " << Hex(spec.addr.ccuDieSize) << "\n";
    os << "  ccu_die_offsets   : [";
    for (size_t i = 0; i < spec.addr.ccuDieOffsets.size(); ++i) {
        os << (i ? ", " : "") << Hex(spec.addr.ccuDieOffsets[i]);
    }
    os << "]\n";
    os << "  user_mem 段数     : " << spec.addr.userMemSegs.size() << "\n";
    for (size_t i = 0; i < spec.addr.userMemSegs.size(); ++i) {
        os << "    seg[" << i << "] offset=" << Hex(ras.UserMemSegOffset(i)) << " size=" << Hex(ras.UserMemSegSize(i))
           << " (" << (ras.UserMemSegSize(i) >> 30) << " GiB)\n";
    }
    os << "  user_mem 合计     : " << Hex(spec.addr.userMemTotal) << " (" << (spec.addr.userMemTotal >> 30)
       << " GiB)\n";
    os << "  rank 槽位上界     : 无（集群 rank 数由拓扑决定，仅受 uint64 VA 溢出防护约束）\n";
    os << "  47-bit VA 上界    : " << ras.Va47BitRankLimit() << " (诊断用，超出仅告警不阻断，见 R4)\n";

    os << "\n  -- ccu --\n";
    os << "  resource_space_size (= backingSize): " << Hex(spec.resourceSpaceSize) << " ("
       << (spec.resourceSpaceSize >> 20) << " MiB)\n";
    os << "  max_modelled_end                   : " << Hex(spec.MaxModelledEnd()) << " ("
       << (spec.MaxModelledEnd() >> 20) << " MiB)\n";
    os << "  未建模留白                         : " << Hex(spec.resourceSpaceSize - spec.MaxModelledEnd()) << " ("
       << ((spec.resourceSpaceSize - spec.MaxModelledEnd()) / 1024.0 / 1024.0) << " MiB)\n";
    os << "  reserved_sync_slots: " << spec.reservedSyncSlots << "\n";
    os << "  count_xn_num       : " << spec.countXnNum << "\n";

    os << "  regions (" << spec.regions.size() << "):\n";
    os << "    " << std::left << std::setw(16) << "name" << std::setw(9) << "type" << std::setw(13) << "offset"
       << std::setw(13) << "size" << std::setw(12) << "entry_size" << "count\n";
    for (const auto& r : spec.regions) {
        os << "    " << std::left << std::setw(16) << r.name << std::setw(9) << CcuResTypeToString(r.type)
           << std::setw(13) << Hex(r.offset) << std::setw(13) << Hex(r.size) << std::setw(12) << r.entrySize << r.count
           << "\n";
    }

    os << "  ms: ms_size=" << spec.ms.msSize << " ms_num=" << spec.ms.msNum << " ccua_num=" << spec.ms.ccuaNum
       << " interleave=" << spec.ms.interleave << "\n";
    for (const auto& w : spec.ms.windows) {
        os << "    ccua[" << w.index << "] offset=" << Hex(w.offset) << " size=" << Hex(w.size) << " ("
           << (w.size / spec.ms.msSize) << " 槽)\n";
    }

    const auto caps = spec.GetCaps();
    os << "\n  -- GetCaps() 推导值（上报 hcomm 的口径）--\n";
    os << "    instruction_num=" << caps.instructionNum << " mission_num=" << caps.missionNum
       << " loop_engine_num=" << caps.loopEngineNum << "\n";
    os << "    gsa_num=" << caps.gsaNum << " xn_num=" << caps.xnNum << " cke_num=" << caps.ckeNum
       << " ms_num=" << caps.msNum << "\n";
    os << "    channel_num=" << caps.channelNum << " jetty_num=" << caps.jettyNum << " pfe_num=" << caps.pfeNum << "\n";

    os << "\n  -- 典型地址 --\n";
    for (uint32_t r = 0; r < 2; ++r) {
        os << "    rank" << r << " base=" << Hex(ras.RankBase(r));
        for (uint32_t d = 0; d < spec.dieNum; ++d) {
            os << "  die" << d << "_ccu=" << Hex(ras.CcuDieBase(r, d));
        }
        os << "\n";
    }
}

void DeviceCommand::ShowSpec()
{
    if (!EnsureLoaded()) {
        return;
    }
    auto& reg = DeviceSpecRegistry::GetInstance();
    if (m_soc.empty()) {
        for (const auto* s : reg.All()) {
            PrintSpec(std::cout, *s);
            std::cout << "\n";
        }
        // 顺带跑校验 9（soc_version 覆盖性）。
        // --dir 模式下从 device 目录的父目录推导 config 根，否则用 install root。
        std::string configRoot;
        if (!m_specDir.empty()) {
            std::string d = m_specDir;
            while (d.size() > 1 && d.back() == '/') {
                d.pop_back();
            }
            const size_t slash = d.find_last_of('/');
            configRoot = (slash == std::string::npos) ? std::string() : d.substr(0, slash);
        }
        std::vector<std::string> violations;
        if (reg.CheckSocVersionCoverage(configRoot, violations)) {
            std::cout << "校验9（soc_version 覆盖性）: 通过\n";
        } else {
            std::cout << "校验9（soc_version 覆盖性）: 失败\n";
            for (const auto& v : violations) {
                std::cout << "  - " << v << "\n";
            }
        }
        return;
    }
    const DeviceSpec* spec = ResolveSpec(m_soc);
    if (spec == nullptr) {
        std::cerr << "[HVM] [ERROR] 未找到匹配 \"" << m_soc << "\" 的 device spec: " << reg.LastError() << std::endl;
        return;
    }
    PrintSpec(std::cout, *spec);
}

void DeviceCommand::ExportSpec()
{
    if (!EnsureLoaded()) {
        return;
    }
    const DeviceSpec* spec = ResolveSpec(m_soc);
    if (spec == nullptr) {
        std::cerr << "[HVM] [ERROR] 未找到匹配 \"" << m_soc << "\" 的 device spec" << std::endl;
        return;
    }
    const auto caps = spec->GetCaps();
    std::ostringstream j;
    j << "{\n";
    j << "  \"name\": \"" << JsonEscape(spec->name) << "\",\n";
    j << "  \"spec_version\": " << spec->specVersion << ",\n";
    j << "  \"ccu_version\": \"" << CcuVersionToString(spec->ccuVersion) << "\",\n";
    j << "  \"die_num\": " << spec->dieNum << ",\n";
    j << "  \"supports_addr_decode\": " << (spec->SupportsAddrDecode() ? "true" : "false") << ",\n";
    j << "  \"address_space\": {\n";
    j << "    \"rank_base\": \"" << Hex(spec->addr.rankBase) << "\",\n";
    j << "    \"rank_stride\": \"" << Hex(spec->addr.rankStride) << "\",\n";
    j << "    \"rank_window_size\": \"" << Hex(spec->addr.rankWindowSize) << "\",\n";
    j << "    \"ccu_window_offset\": \"" << Hex(spec->addr.ccuWindowOffset) << "\",\n";
    j << "    \"ccu_window_size\": \"" << Hex(spec->addr.ccuWindowSize) << "\",\n";
    j << "    \"ccu_die_size\": \"" << Hex(spec->addr.ccuDieSize) << "\",\n";
    j << "    \"user_mem_total\": \"" << Hex(spec->addr.userMemTotal) << "\",\n";
    j << "    \"user_mem_segments\": [\n";
    for (size_t i = 0; i < spec->addr.userMemSegs.size(); ++i) {
        const auto& s = spec->addr.userMemSegs[i];
        j << "      {\"offset\": \"" << Hex(s.offset) << "\", \"size\": \"" << Hex(s.size) << "\"}"
          << (i + 1 < spec->addr.userMemSegs.size() ? "," : "") << "\n";
    }
    j << "    ],\n";
    j << "    \"ccu_die_offsets\": [";
    for (size_t i = 0; i < spec->addr.ccuDieOffsets.size(); ++i) {
        j << (i ? ", " : "") << "\"" << Hex(spec->addr.ccuDieOffsets[i]) << "\"";
    }
    j << "]\n  },\n";
    j << "  \"ccu\": {\n";
    j << "    \"resource_space_size\": \"" << Hex(spec->resourceSpaceSize) << "\",\n";
    j << "    \"reserved_sync_slots\": " << spec->reservedSyncSlots << ",\n";
    j << "    \"count_xn_num\": " << spec->countXnNum << ",\n";
    j << "    \"regions\": [\n";
    for (size_t i = 0; i < spec->regions.size(); ++i) {
        const auto& r = spec->regions[i];
        j << "      {\"name\": \"" << JsonEscape(r.name) << "\", \"type\": \"" << CcuResTypeToString(r.type)
          << "\", \"offset\": \"" << Hex(r.offset) << "\", \"size\": \"" << Hex(r.size)
          << "\", \"entry_size\": " << r.entrySize << ", \"count\": " << r.count << "}"
          << (i + 1 < spec->regions.size() ? "," : "") << "\n";
    }
    j << "    ],\n";
    j << "    \"ms\": {\n";
    j << "      \"ms_size\": " << spec->ms.msSize << ", \"ms_num\": " << spec->ms.msNum
      << ", \"ccua_num\": " << spec->ms.ccuaNum << ", \"interleave\": " << spec->ms.interleave << ",\n";
    j << "      \"ccua_windows\": [\n";
    for (size_t i = 0; i < spec->ms.windows.size(); ++i) {
        const auto& w = spec->ms.windows[i];
        j << "        {\"index\": " << w.index << ", \"offset\": \"" << Hex(w.offset) << "\", \"size\": \""
          << Hex(w.size) << "\"}" << (i + 1 < spec->ms.windows.size() ? "," : "") << "\n";
    }
    j << "      ]\n    }\n  },\n";
    j << "  \"caps\": {\n";
    j << "    \"instruction_num\": " << caps.instructionNum << ", \"mission_num\": " << caps.missionNum
      << ", \"loop_engine_num\": " << caps.loopEngineNum << ",\n";
    j << "    \"gsa_num\": " << caps.gsaNum << ", \"xn_num\": " << caps.xnNum << ", \"cke_num\": " << caps.ckeNum
      << ", \"ms_num\": " << caps.msNum << ",\n";
    j << "    \"channel_num\": " << caps.channelNum << ", \"jetty_num\": " << caps.jettyNum
      << ", \"pfe_num\": " << caps.pfeNum << "\n  }\n";
    j << "}\n";

    if (m_outFile.empty()) {
        std::cout << j.str();
        return;
    }
    std::ofstream out(m_outFile);
    if (!out.is_open()) {
        std::cerr << "[HVM] [ERROR] 无法写入 " << m_outFile << std::endl;
        return;
    }
    out << j.str();
    out.close();
    std::cout << "已导出 " << spec->name << " 到 " << m_outFile << std::endl;
}

void DeviceCommand::DecodeAddr()
{
    if (!EnsureLoaded()) {
        return;
    }
    uint64_t va = 0;
    if (!ParseUint64(m_addrStr, va)) {
        std::cerr << "[HVM] [ERROR] 无法解析地址 \"" << m_addrStr << "\"" << std::endl;
        return;
    }
    auto& reg = DeviceSpecRegistry::GetInstance();

    // ★ A5/A6 的 address_space 按设计完全相同（芯片无关），故同一个 VA 在多份 spec 下
    //   都"落在 rank 窗口内"，但 region 布局不同 → 解码结果不同。VA 本身不足以判定芯片。
    //   因此：指定 --soc 时只解一份；未指定时遍历全部并逐一给出结果 + 歧义提示。
    std::vector<const DeviceSpec*> candidates;
    if (!m_soc.empty()) {
        const DeviceSpec* spec = ResolveSpec(m_soc);
        if (spec == nullptr) {
            std::cerr << "[HVM] [ERROR] 未找到匹配 \"" << m_soc << "\" 的 device spec: " << reg.LastError()
                      << std::endl;
            return;
        }
        candidates.push_back(spec);
    } else {
        candidates = reg.All();
    }

    size_t matched = 0;
    for (const auto* spec : candidates) {
        const RankAddressSpace ras(spec->addr);
        if (spec->addr.rankStride == 0 || va < spec->addr.rankBase) {
            continue;
        }
        const uint64_t rel = va - spec->addr.rankBase;
        const uint64_t rankIdx64 = rel / spec->addr.rankStride;
        if (rankIdx64 > UINT32_MAX) {
            continue;
        }
        const uint32_t rankIdx = static_cast<uint32_t>(rankIdx64);
        const uint64_t rankBase = ras.RankBase(rankIdx);
        if (va >= rankBase + spec->addr.rankWindowSize) {
            continue;
        }
        if (matched > 0) {
            std::cout << "\n";
        }
        ++matched;
        std::cout << "=== spec: " << spec->name << " (" << CcuVersionToString(spec->ccuVersion) << ") ===\n";
        std::cout << "  rank      : " << rankIdx << "\n";
        std::cout << "  rankBase  : " << Hex(rankBase) << "\n";
        std::cout << "  窗口内偏移: " << Hex(va - rankBase) << "\n";

        uint32_t hitRank = 0, hitDie = 0;
        if (ras.IsInCcuWindow(va, hitRank, hitDie)) {
            const uint64_t dieBase = ras.CcuDieBase(hitRank, hitDie);
            const uint64_t offInDie = va - dieBase;
            std::cout << "  区域      : CCU die" << hitDie << "\n";
            std::cout << "  dieBase   : " << Hex(dieBase) << "\n";
            std::cout << "  die 内偏移: " << Hex(offInDie) << "\n";
            const CcuAddressSpace cas(*spec);
            CcuResType type = CcuResType::UNKNOWN;
            uint64_t id = 0;
            if (cas.AddrToResId(dieBase, va, type, id)) {
                std::cout << "  资源类型  : " << CcuResTypeToString(type) << "\n";
                std::cout << "  资源 id   : " << id << "\n";
            } else {
                // 区分三种情况，避免笼统报"未建模"
                const CcuResType t = cas.TypeByOffset(offInDie);
                if (t == CcuResType::MS) {
                    std::cout << "  资源类型  : MS 窗口内，但地址未按 ms_size(" << spec->ms.msSize
                              << ") 对齐，或换算出的 msId 超出 ms_num(" << spec->ms.msNum << ")\n";
                } else if (t != CcuResType::UNKNOWN) {
                    std::cout << "  资源类型  : " << CcuResTypeToString(t)
                              << " 区域内，但未按 entry_size 对齐或 id 超出 count\n";
                } else if (cas.IsInResourceSpace(dieBase, va)) {
                    std::cout << "  资源类型  : <未建模区域>（不在任何 region/MS 窗口内，但仍在 "
                                 "resource_space_size "
                              << Hex(spec->resourceSpaceSize) << " 的 backing 范围内）\n";
                } else {
                    std::cout << "  资源类型  : <超出资源空间>（偏移 " << Hex(offInDie) << " ≥ resource_space_size "
                              << Hex(spec->resourceSpaceSize) << "）\n";
                }
            }
            std::cout << "  在资源空间内: " << (cas.IsInResourceSpace(dieBase, va) ? "yes" : "no") << "\n";
        } else if (ras.IsUserMemVa(va, rankIdx)) {
            std::cout << "  区域      : user_mem（rtMalloc 可分配）\n";
            for (size_t i = 0; i < ras.UserMemSegCount(); ++i) {
                const uint64_t segVa = rankBase + ras.UserMemSegOffset(i);
                if (va >= segVa && va < segVa + ras.UserMemSegSize(i)) {
                    std::cout << "  所在段    : seg[" << i << "]，段内偏移 " << Hex(va - segVa) << "\n";
                    break;
                }
            }
        } else {
            std::cout << "  区域      : <未识别>\n";
        }
    }

    if (matched == 0) {
        std::cerr << "[HVM] [ERROR] 地址 " << Hex(va) << " 不落在任何已加载 spec 的 rank 窗口内" << std::endl;
        return;
    }
    if (matched > 1) {
        std::cout << "\n⚠ 该 VA 命中 " << matched << " 份 spec（A5/A6 共用同一 address_space，"
                  << "仅 CCU 区域内布局不同）。请用 --soc 限定芯片型号以获得唯一结果。\n";
    }
}

void DeviceCommand::EncodeAddr()
{
    if (!EnsureLoaded()) {
        return;
    }
    const DeviceSpec* spec = ResolveSpec(m_soc);
    if (spec == nullptr) {
        auto& reg = DeviceSpecRegistry::GetInstance();
        std::cerr << "[HVM] [ERROR] 无法确定 device spec（--soc 未给且已加载 " << reg.Size()
                  << " 份，请用 --soc 指定）: " << reg.LastError() << std::endl;
        return;
    }
    CcuResType type = CcuResType::UNKNOWN;
    if (!CcuResTypeFromString(m_resType, type)) {
        std::cerr << "[HVM] [ERROR] 非法资源类型 \"" << m_resType
                  << "\"（允许 INSTR|CCUM|GSA|XN|CKE|PFE|CHANNEL|JETTY|MISSION|LOOP|WQE|MS）" << std::endl;
        return;
    }
    const RankAddressSpace ras(spec->addr);
    const uint64_t dieBase = ras.CcuDieBase(m_rank, m_die);
    if (dieBase == 0) {
        std::cerr << "[HVM] [ERROR] 非法 rank/die: rank=" << m_rank << " die=" << m_die << "（die_num=" << spec->dieNum
                  << "）" << std::endl;
        return;
    }
    const CcuAddressSpace cas(*spec);
    uint64_t addr = 0;
    if (!cas.ResIdToAddr(dieBase, type, m_resId, addr)) {
        std::cerr << "[HVM] [ERROR] 编址失败: type=" << CcuResTypeToString(type) << " id=" << m_resId
                  << "（该 spec 可能无此区域，或 id 超出 count）" << std::endl;
        return;
    }
    std::cout << "spec     : " << spec->name << "\n";
    std::cout << "rank     : " << m_rank << "\n";
    std::cout << "die      : " << m_die << "\n";
    std::cout << "dieBase  : " << Hex(dieBase) << "\n";
    std::cout << "资源类型 : " << CcuResTypeToString(type) << "\n";
    std::cout << "资源 id  : " << m_resId << "\n";
    std::cout << "设备 VA  : " << Hex(addr) << "\n";
    std::cout << "die 内偏移: " << Hex(addr - dieBase) << "\n";
    // 自校验：逆运算必须回到同一 id
    CcuResType backType = CcuResType::UNKNOWN;
    uint64_t backId = 0;
    if (cas.AddrToResId(dieBase, addr, backType, backId)) {
        std::cout << "逆运算   : " << CcuResTypeToString(backType) << " id=" << backId << " "
                  << ((backType == type && backId == m_resId) ? "✓ 一致" : "✗ 不一致") << "\n";
    }
}

static inline CommandAutoRegister<DeviceCommand> g_device_cmd_reg{};

} // namespace HcclSim
