/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "device_spec_yaml_loader.h"

#include <algorithm>
#include <cctype>
#include <cstdarg>
#include <cstdio>
#include <set>
#include <sstream>

#if !defined(NO_YAML_CONFIG) && defined(HAVE_YAML_CPP)
#include "yaml-cpp/yaml.h"
#endif

#include "sim_log.h"
#include "sim_capacity_limits.h"

namespace sim {
namespace devspec {
    namespace {

        constexpr const char* PART_USER_MEM = "user_mem";
        constexpr const char* PART_CCU = "ccu";

        std::string Fmt(const char* fmt, ...)
        {
            char buf[1024] = {0};
            va_list args;
            va_start(args, fmt);
            (void)vsnprintf(buf, sizeof(buf) - 1, fmt, args);
            va_end(args);
            return std::string(buf);
        }

        std::string ToLower(const std::string& s)
        {
            std::string r = s;
            std::transform(r.begin(), r.end(), r.begin(), [](unsigned char c) {
                return static_cast<char>(tolower(c));
            });
            return r;
        }

#if !defined(NO_YAML_CONFIG) && defined(HAVE_YAML_CPP)

        /**
         * 读取一个必须存在的 uint64 标量。
         *
         * ★ 必须用 uint64_t：rank_stride(0x10000000000)、user_mem 段1 size(0xF180000000)、
         *   rank_base(0x8000000000) 均 ≥ 2^32，用 32 位类型取会因 num_get 置 failbit
         *   而抛 TypedBadConversion（设计文档 §4.2.2 解析注意）。
         * ★ 0x 前缀无需特判：yaml-cpp 的 convert<T>::decode 内部 unsetf(ios::dec) 走 base-0，
         *   PyYAML(YAML 1.1) 亦原生支持，两侧一致。
         */
        bool ReadU64(
            const YAML::Node& node, const std::string& key, const std::string& path, uint64_t& out, std::string& err)
        {
            if (!node[key]) {
                err = path + "." + key + ": 字段缺失";
                return false;
            }
            try {
                out = node[key].as<uint64_t>();
            } catch (const std::exception& e) {
                err = Fmt(
                    "%s.%s: 无法解析为 uint64（标量=\"%s\"）: %s", path.c_str(), key.c_str(),
                    node[key].Scalar().c_str(), e.what());
                return false;
            }
            return true;
        }

        /// 读取一个必须存在的 uint32 标量。
        bool ReadU32(
            const YAML::Node& node, const std::string& key, const std::string& path, uint32_t& out, std::string& err)
        {
            uint64_t v = 0;
            if (!ReadU64(node, key, path, v, err)) {
                return false;
            }
            if (v > UINT32_MAX) {
                err = Fmt(
                    "%s.%s: 值 %llu 超出 uint32 范围", path.c_str(), key.c_str(), static_cast<unsigned long long>(v));
                return false;
            }
            out = static_cast<uint32_t>(v);
            return true;
        }

        /// 读取一个必须存在的字符串标量。
        bool ReadStr(
            const YAML::Node& node, const std::string& key, const std::string& path, std::string& out, std::string& err)
        {
            if (!node[key]) {
                err = path + "." + key + ": 字段缺失";
                return false;
            }
            try {
                out = node[key].as<std::string>();
            } catch (const std::exception& e) {
                err = Fmt("%s.%s: 无法解析为字符串: %s", path.c_str(), key.c_str(), e.what());
                return false;
            }
            if (out.empty()) {
                err = path + "." + key + ": 值为空";
                return false;
            }
            return true;
        }

        /// 读取一个可选的 uint32 标量；缺失时保留 def。
        bool ReadU32Opt(
            const YAML::Node& node, const std::string& key, const std::string& path, uint32_t def, uint32_t& out,
            std::string& err)
        {
            if (!node[key]) {
                out = def;
                return true;
            }
            return ReadU32(node, key, path, out, err);
        }

        bool
        ReadBool(const YAML::Node& node, const std::string& key, const std::string& path, bool& out, std::string& err)
        {
            if (!node[key]) {
                err = path + "." + key + ": 字段缺失";
                return false;
            }
            try {
                out = node[key].as<bool>();
            } catch (const std::exception& e) {
                err = Fmt("%s.%s: 无法解析为 bool: %s", path.c_str(), key.c_str(), e.what());
                return false;
            }
            return true;
        }

        bool ParseCompute(const YAML::Node& root, DeviceSpec& spec, std::string& err)
        {
            const auto compute = root["compute"];
            if (!compute || !compute.IsMap()) {
                err = "compute: 段缺失或不是映射";
                return false;
            }
            return ReadU32(compute, "vector_core_num", "compute", spec.vectorCoreNum, err);
        }

        bool ParseDeviceRes(const YAML::Node& root, DeviceSpec& spec, std::string& err)
        {
            const auto res = root["device_res"];
            if (!res || !res.IsMap()) {
                err = "device_res: 段缺失或不是映射";
                return false;
            }
            DeviceResSpec& out = spec.deviceRes;
            if (!ReadU32(res, "stream_num", "device_res", out.streamNum, err)
                || !ReadU32(res, "notify_num", "device_res", out.notifyNum, err)) {
                return false;
            }
            const auto jetty = res["ub_jetty"];
            if (!jetty || !jetty.IsMap()) {
                err = "device_res.ub_jetty: 段缺失或不是映射";
                return false;
            }
            return ReadU32(jetty, "normal_pool_size", "device_res.ub_jetty", out.ubJetty.normalPoolSize, err);
        }

        bool ValidateDeviceRes(const DeviceSpec& spec, std::string& err)
        {
            const DeviceResSpec& res = spec.deviceRes;
            if (res.streamNum == 0 || res.notifyNum == 0 || res.ubJetty.normalPoolSize == 0
                || spec.vectorCoreNum == 0) {
                err = "device_res.stream_num/notify_num/ub_jetty.normal_pool_size/"
                      "compute.vector_core_num 不得为 0";
                return false;
            }
            if (res.notifyNum > HcclSim::HCCL_VM_DEVICE_NOTIFY_ID_CAPACITY) {
                err = "device_res.notify_num: 超过 17bit Notify 编址容量 131072";
                return false;
            }
            return true;
        }

        bool ParseSoc(const YAML::Node& root, DeviceSpec& spec, std::string& err)
        {
            const auto soc = root["soc"];
            if (!soc || !soc.IsMap()) {
                err = "soc: 段缺失或不是映射";
                return false;
            }
            // ★ 决策 (i)：soc.soc_version 字段【完全不解析】——不参与匹配、不参与校验、不落库。
            //   匹配只认 spec.name（大小写不敏感最长前缀）。见设计文档 §4.7.4。
            //   此处刻意不读该 key，spec.socVersion 保持空串。
            std::string ver;
            if (!ReadStr(soc, "ccu_version", "soc", ver, err)) {
                return false;
            }
            if (!CcuVersionFromString(ver, spec.ccuVersion)) {
                err = Fmt("soc.ccu_version: 非法取值 \"%s\"（允许 CCU_V1 | CCU_V2 | CCU_V3）", ver.c_str());
                return false;
            }
            if (!ReadU32(soc, "die_num", "soc", spec.dieNum, err)) {
                return false;
            }
            if (spec.dieNum == 0 || spec.dieNum > 2) {
                // O4：本期仅支持 1/2；Ccu 表当前每 device 固定 2 行（topo_ascend_cluster_parser.cc:1125-1133）
                err = Fmt("soc.die_num: 值 %u 本期仅支持 1 或 2（见开放问题 O4）", spec.dieNum);
                return false;
            }
            return true;
        }

        bool ParseAddrSpace(const YAML::Node& root, DeviceSpec& spec, std::string& err)
        {
            const auto as = root["address_space"];
            if (!as || !as.IsMap()) {
                err = "address_space: 段缺失或不是映射";
                return false;
            }
            AddrSpace& a = spec.addr;
            if (!ReadU64(as, "rank_base", "address_space", a.rankBase, err)
                || !ReadU64(as, "rank_stride", "address_space", a.rankStride, err)
                || !ReadU64(as, "rank_window_size", "address_space", a.rankWindowSize, err)) {
                return false;
            }

            const auto parts = as["partitions"];
            if (!parts || !parts.IsSequence() || parts.size() == 0) {
                err = "address_space.partitions: 缺失或不是非空序列";
                return false;
            }
            for (size_t i = 0; i < parts.size(); ++i) {
                const auto p = parts[i];
                const std::string path = Fmt("address_space.partitions[%zu]", i);
                Partition part;
                if (!ReadStr(p, "name", path, part.name, err) || !ReadU64(p, "offset", path, part.offset, err)
                    || !ReadU64(p, "size", path, part.size, err)) {
                    return false;
                }
                // 校验 5：name 白名单（本期只允许 user_mem / ccu）
                if (part.name != PART_USER_MEM && part.name != PART_CCU) {
                    err = Fmt(
                        "%s.name: 非法取值 \"%s\"（本期白名单：user_mem | ccu）", path.c_str(), part.name.c_str());
                    return false;
                }
                if (part.size == 0) {
                    err = path + ".size: 不得为 0";
                    return false;
                }
                a.partitions.push_back(part);
            }

            const auto ccu = as["ccu"];
            if (!ccu || !ccu.IsMap()) {
                err = "address_space.ccu: 段缺失或不是映射";
                return false;
            }
            const auto offs = ccu["die_offsets"];
            if (!offs || !offs.IsSequence()) {
                err = "address_space.ccu.die_offsets: 缺失或不是序列";
                return false;
            }
            for (size_t i = 0; i < offs.size(); ++i) {
                try {
                    a.ccuDieOffsets.push_back(offs[i].as<uint64_t>());
                } catch (const std::exception& e) {
                    err = Fmt("address_space.ccu.die_offsets[%zu]: 无法解析为 uint64: %s", i, e.what());
                    return false;
                }
            }
            if (!ReadU64(ccu, "die_size", "address_space.ccu", a.ccuDieSize, err)) {
                return false;
            }
            return true;
        }

        bool ParseRegions(const YAML::Node& ccu, DeviceSpec& spec, std::string& err)
        {
            const auto regions = ccu["regions"];
            if (!regions || !regions.IsSequence() || regions.size() == 0) {
                err = "ccu.regions: 缺失或不是非空序列";
                return false;
            }
            for (size_t i = 0; i < regions.size(); ++i) {
                const auto r = regions[i];
                const std::string path = Fmt("ccu.regions[%zu]", i);
                RegionSpec rs;
                if (!ReadStr(r, "name", path, rs.name, err) || !ReadU64(r, "offset", path, rs.offset, err)
                    || !ReadU64(r, "size", path, rs.size, err) || !ReadU32(r, "entry_size", path, rs.entrySize, err)
                    || !ReadU32(r, "count", path, rs.count, err)) {
                    return false;
                }
                std::string typeStr;
                if (!ReadStr(r, "type", path, typeStr, err)) {
                    return false;
                }
                if (!CcuResTypeFromString(typeStr, rs.type)) {
                    err = Fmt(
                        "%s.type: 非法取值 \"%s\"（闭集见 device_spec_types.h 的 CcuResType）", path.c_str(),
                        typeStr.c_str());
                    return false;
                }
                if (rs.size == 0) {
                    err = path + ".size: 不得为 0";
                    return false;
                }
                // MS 不走 regions 表（其地址由 ms.ccua_windows 交织计算）
                if (rs.type == CcuResType::MS) {
                    err = path + ".type: MS 应在 ccu.ms.ccua_windows 中描述，不得作为 region";
                    return false;
                }
                spec.regions.push_back(rs);
            }
            // 同一 type 不得重复出现（否则 GetCaps/FindRegion 语义歧义）
            std::set<CcuResType> seen;
            for (const auto& rs : spec.regions) {
                if (!seen.insert(rs.type).second) {
                    err = Fmt("ccu.regions: type \"%s\" 重复出现", CcuResTypeToString(rs.type));
                    return false;
                }
            }
            return true;
        }

        bool ParseMs(const YAML::Node& ccu, DeviceSpec& spec, std::string& err)
        {
            const auto ms = ccu["ms"];
            if (!ms || !ms.IsMap()) {
                err = "ccu.ms: 段缺失或不是映射";
                return false;
            }
            MsSpec& m = spec.ms;
            if (!ReadU32(ms, "ms_size", "ccu.ms", m.msSize, err) || !ReadU32(ms, "ms_num", "ccu.ms", m.msNum, err)
                || !ReadU32(ms, "ccua_num", "ccu.ms", m.ccuaNum, err)
                || !ReadU32(ms, "interleave", "ccu.ms", m.interleave, err)) {
                return false;
            }
            if (m.msSize == 0 || m.msNum == 0 || m.ccuaNum == 0 || m.interleave == 0) {
                err = "ccu.ms: ms_size/ms_num/ccua_num/interleave 均不得为 0";
                return false;
            }
            const auto wins = ms["ccua_windows"];
            if (!wins || !wins.IsSequence()) {
                err = "ccu.ms.ccua_windows: 缺失或不是序列";
                return false;
            }
            for (size_t i = 0; i < wins.size(); ++i) {
                const auto w = wins[i];
                const std::string path = Fmt("ccu.ms.ccua_windows[%zu]", i);
                MsWindow mw;
                if (!ReadU32(w, "index", path, mw.index, err) || !ReadU64(w, "offset", path, mw.offset, err)
                    || !ReadU64(w, "size", path, mw.size, err)) {
                    return false;
                }
                if (mw.size == 0) {
                    err = path + ".size: 不得为 0";
                    return false;
                }
                m.windows.push_back(mw);
            }
            return true;
        }

        // ===================== 校验（设计文档 §4.3.1）=====================

        /// 校验 1：regions ∪ ms.ccua_windows 区间互不重叠，且全部 ⊂ [0, die_size)
        bool ValidateNoOverlap(const DeviceSpec& spec, std::string& err)
        {
            struct Iv {
                uint64_t lo, hi;
                std::string who;
            };
            std::vector<Iv> ivs;
            for (const auto& r : spec.regions) {
                ivs.push_back({r.offset, r.offset + r.size, r.name});
            }
            for (const auto& w : spec.ms.windows) {
                ivs.push_back({w.offset, w.offset + w.size, Fmt("ms.ccua_windows[%u]", w.index)});
            }
            std::sort(ivs.begin(), ivs.end(), [](const Iv& a, const Iv& b) {
                return a.lo < b.lo;
            });
            const uint64_t dieSize = spec.addr.ccuDieSize;
            for (size_t i = 0; i < ivs.size(); ++i) {
                if (ivs[i].hi > dieSize) {
                    err = Fmt(
                        "校验1 失败: %s [%#llx, %#llx) 超出 [0, die_size=%#llx)", ivs[i].who.c_str(),
                        (unsigned long long)ivs[i].lo, (unsigned long long)ivs[i].hi, (unsigned long long)dieSize);
                    return false;
                }
                if (i + 1 < ivs.size() && ivs[i].hi > ivs[i + 1].lo) {
                    err = Fmt(
                        "校验1 失败: %s [%#llx, %#llx) 与 %s [%#llx, %#llx) 重叠", ivs[i].who.c_str(),
                        (unsigned long long)ivs[i].lo, (unsigned long long)ivs[i].hi, ivs[i + 1].who.c_str(),
                        (unsigned long long)ivs[i + 1].lo, (unsigned long long)ivs[i + 1].hi);
                    return false;
                }
            }
            return true;
        }

        /// 校验 2：entry_size × count ≤ size（entry_size == 0 的不透明块跳过）
        bool ValidateEntryFits(const DeviceSpec& spec, std::string& err)
        {
            for (const auto& r : spec.regions) {
                if (r.entrySize == 0) {
                    continue;
                }
                const uint64_t need = static_cast<uint64_t>(r.entrySize) * r.count;
                if (need > r.size) {
                    err = Fmt(
                        "校验2 失败: region \"%s\" entry_size(%u) × count(%u) = %#llx > size %#llx", r.name.c_str(),
                        r.entrySize, r.count, (unsigned long long)need, (unsigned long long)r.size);
                    return false;
                }
            }
            return true;
        }

        /**
         * 校验 3：MS per-window 覆盖 + 结构一致性。
         *
         * ★ 不用求和式 ms_num ≤ Σ窗口槽数 —— 异构窗口下求和式会漏判：
         *   窗口容量 [256,512,512,512] 时 Σ=1792 ≥ 1536 通过，但 localIdx_max=383 > 255
         *   使窗口 0 溢出（设计文档 §4.3.1 校验 3）。
         */
        bool ValidateMs(const DeviceSpec& spec, std::string& err)
        {
            const MsSpec& m = spec.ms;
            if (m.windows.size() != m.ccuaNum) {
                err = Fmt("校验3 失败: ccua_num(%u) != len(ccua_windows)(%zu)", m.ccuaNum, m.windows.size());
                return false;
            }
            for (size_t i = 0; i < m.windows.size(); ++i) {
                if (m.windows[i].index != i) {
                    err = Fmt(
                        "校验3 失败: ccua_windows[%zu].index = %u，期望 %zu（须有序无重复）", i, m.windows[i].index, i);
                    return false;
                }
                if (m.windows[i].size % m.msSize != 0) {
                    err = Fmt(
                        "校验3 失败: ccua_windows[%zu].size %#llx 不是 ms_size %u 的整数倍", i,
                        (unsigned long long)m.windows[i].size, m.msSize);
                    return false;
                }
            }
            const uint64_t stride = static_cast<uint64_t>(m.ccuaNum) * m.interleave;
            const uint64_t localIdxMax = ((m.msNum - 1) / stride) * m.interleave + (m.interleave - 1);
            for (const auto& w : m.windows) {
                const uint64_t slots = w.size / m.msSize;
                if (localIdxMax >= slots) {
                    err = Fmt(
                        "校验3 失败: ccua_windows[%u] 槽数 %llu 不足，需 localIdx_max %llu "
                        "(ms_num=%u, ccua_num=%u, interleave=%u)",
                        w.index, (unsigned long long)slots, (unsigned long long)localIdxMax, m.msNum, m.ccuaNum,
                        m.interleave);
                    return false;
                }
            }
            return true;
        }

        /// 校验 4：die_offsets 数量 == die_num；die 窗口互不重叠
        bool ValidateDieOffsets(const DeviceSpec& spec, std::string& err)
        {
            const AddrSpace& a = spec.addr;
            if (a.ccuDieOffsets.size() != spec.dieNum) {
                err = Fmt(
                    "校验4 失败: len(ccu.die_offsets)=%zu != soc.die_num=%u", a.ccuDieOffsets.size(), spec.dieNum);
                return false;
            }
            std::vector<uint64_t> offs = a.ccuDieOffsets;
            std::sort(offs.begin(), offs.end());
            for (size_t i = 0; i + 1 < offs.size(); ++i) {
                if (offs[i] + a.ccuDieSize > offs[i + 1]) {
                    err = Fmt(
                        "校验4 失败: die 窗口重叠 —— die_offsets[%zu]=%#llx + die_size %#llx > "
                        "die_offsets[%zu]=%#llx",
                        i, (unsigned long long)offs[i], (unsigned long long)a.ccuDieSize, i + 1,
                        (unsigned long long)offs[i + 1]);
                    return false;
                }
            }
            return true;
        }

        /**
         * 校验 5：partitions 铺满性 + user_mem 派生 + ccu 段容量。
         *
         * 派生量写回 spec.addr（ccuWindowOffset/ccuWindowSize/userMemSegs/userMemTotal），
         * 供 RankAddressSpace 与 AllocVirMem 使用。
         */
        bool ValidatePartitions(DeviceSpec& spec, std::string& err)
        {
            AddrSpace& a = spec.addr;
            if (a.rankStride == 0 || a.rankWindowSize == 0) {
                err = "校验5 失败: rank_stride / rank_window_size 不得为 0";
                return false;
            }
            if (a.rankWindowSize > a.rankStride) {
                err = Fmt(
                    "校验5 失败: rank_window_size %#llx > rank_stride %#llx（相邻 rank 窗口会重叠）",
                    (unsigned long long)a.rankWindowSize, (unsigned long long)a.rankStride);
                return false;
            }

            std::vector<Partition> sorted = a.partitions;
            std::sort(sorted.begin(), sorted.end(), [](const Partition& x, const Partition& y) {
                return x.offset < y.offset;
            });

            // 完整铺满 [0, rank_window_size)：首段 offset==0、相邻首尾相接、末段抵达窗口末端
            if (sorted.front().offset != 0) {
                err = Fmt(
                    "校验5 失败: partitions 首段 offset = %#llx，须为 0", (unsigned long long)sorted.front().offset);
                return false;
            }
            for (size_t i = 0; i + 1 < sorted.size(); ++i) {
                const uint64_t end = sorted[i].offset + sorted[i].size;
                if (end != sorted[i + 1].offset) {
                    err = Fmt(
                        "校验5 失败: partitions 未铺满 —— \"%s\" 末端 %#llx 与 \"%s\" 起点 %#llx "
                        "不衔接（存在空洞或重叠）",
                        sorted[i].name.c_str(), (unsigned long long)end, sorted[i + 1].name.c_str(),
                        (unsigned long long)sorted[i + 1].offset);
                    return false;
                }
            }
            const uint64_t lastEnd = sorted.back().offset + sorted.back().size;
            if (lastEnd != a.rankWindowSize) {
                err = Fmt(
                    "校验5 失败: partitions 末端 %#llx != rank_window_size %#llx", (unsigned long long)lastEnd,
                    (unsigned long long)a.rankWindowSize);
                return false;
            }

            // 恰好一个 ccu 段；≥1 个 user_mem 段
            uint64_t ccuOff = 0, ccuSize = 0;
            size_t ccuCnt = 0;
            std::vector<Partition> userSegs;
            for (const auto& p : sorted) {
                if (p.name == PART_CCU) {
                    ++ccuCnt;
                    ccuOff = p.offset;
                    ccuSize = p.size;
                } else {
                    userSegs.push_back(p);
                }
            }
            if (ccuCnt != 1) {
                err = Fmt("校验5 失败: partitions 中 name==\"ccu\" 的段数为 %zu，须恰好 1", ccuCnt);
                return false;
            }
            if (userSegs.empty()) {
                err = "校验5 失败: partitions 中没有 user_mem 段，rtMalloc 将无地址可分配";
                return false;
            }

            // ccu 段须容纳全部 die 窗口
            uint64_t maxDieOff = 0;
            for (uint64_t o : a.ccuDieOffsets) {
                maxDieOff = std::max(maxDieOff, o);
            }
            if (maxDieOff + a.ccuDieSize > ccuSize) {
                err = Fmt(
                    "校验5 失败: ccu 段 size %#llx 装不下 die 窗口 —— max(die_offsets) %#llx + "
                    "die_size %#llx = %#llx",
                    (unsigned long long)ccuSize, (unsigned long long)maxDieOff, (unsigned long long)a.ccuDieSize,
                    (unsigned long long)(maxDieOff + a.ccuDieSize));
                return false;
            }

            uint64_t total = 0;
            for (const auto& s : userSegs) {
                total += s.size;
            }
            if (total == 0) {
                err = "校验5 失败: userMemTotal == 0";
                return false;
            }

            a.ccuWindowOffset = ccuOff;
            a.ccuWindowSize = ccuSize;
            a.userMemSegs = userSegs; // 已按 offset 升序
            a.userMemTotal = total;
            return true;
        }

        /// 校验 8：max(regions ∪ ms 末端) ≤ resource_space_size ≤ die_size
        bool ValidateResourceSpaceSize(const DeviceSpec& spec, std::string& err)
        {
            if (spec.resourceSpaceSize == 0) {
                err = "校验8 失败: ccu.resource_space_size 缺失或为 0";
                return false;
            }
            const uint64_t modelled = spec.MaxModelledEnd();
            if (modelled > spec.resourceSpaceSize) {
                err = Fmt(
                    "校验8 失败: max(regions ∪ ms 末端) = %#llx > resource_space_size %#llx "
                    "—— 已建模区域超出 backing 范围，hcomm/runner 访问会落到 mmap 之外",
                    (unsigned long long)modelled, (unsigned long long)spec.resourceSpaceSize);
                return false;
            }
            if (spec.resourceSpaceSize > spec.addr.ccuDieSize) {
                err = Fmt(
                    "校验8 失败: resource_space_size %#llx > ccu.die_size %#llx",
                    (unsigned long long)spec.resourceSpaceSize, (unsigned long long)spec.addr.ccuDieSize);
                return false;
            }
            return true;
        }

#endif // HAVE_YAML_CPP

    } // namespace

    bool LoadDeviceSpecYaml(const std::string& absPath, DeviceSpec& out, std::string& err)
    {
        err.clear();
#if !defined(NO_YAML_CONFIG) && defined(HAVE_YAML_CPP)
        YAML::Node root;
        try {
            root = YAML::LoadFile(absPath);
        } catch (const std::exception& e) {
            err = Fmt("%s: YAML 解析失败: %s", absPath.c_str(), e.what());
            return false;
        }
        if (!root || !root.IsMap()) {
            err = absPath + ": 顶层不是映射";
            return false;
        }

        DeviceSpec spec;
        spec.sourcePath = absPath;

        std::string typeStr;
        if (!ReadStr(root, "type", "<root>", typeStr, err) || typeStr != "device_spec") {
            err = err.empty() ? Fmt("<root>.type: 期望 \"device_spec\"，实际 \"%s\"", typeStr.c_str()) : err;
            return false;
        }
        if (!ReadU32(root, "spec_version", "<root>", spec.specVersion, err)
            || !ReadStr(root, "name", "<root>", spec.name, err)) {
            return false;
        }
        // description 可选
        if (root["description"]) {
            try {
                spec.description = root["description"].as<std::string>();
            } catch (const std::exception&) {
                spec.description.clear();
            }
        }
        if (ToLower(spec.name).empty()) {
            err = "<root>.name: 不得为空";
            return false;
        }

        if (!ParseSoc(root, spec, err) || !ParseAddrSpace(root, spec, err)) {
            return false;
        }

        const auto ccu = root["ccu"];
        if (!ccu || !ccu.IsMap()) {
            err = "ccu: 段缺失或不是映射";
            return false;
        }
        if (!ReadU64(ccu, "resource_space_size", "ccu", spec.resourceSpaceSize, err) || !ParseRegions(ccu, spec, err)
            || !ParseMs(ccu, spec, err)) {
            return false;
        }
        if (!ReadU32(ccu, "reserved_sync_slots", "ccu", spec.reservedSyncSlots, err)) {
            return false;
        }
        // count_xn_num 可选（A5 为 0，A6 为 4）；缺失按 0 处理
        if (!ReadU32Opt(ccu, "count_xn_num", "ccu", 0, spec.countXnNum, err)) {
            return false;
        }
        if (!ParseCompute(root, spec, err) || !ParseDeviceRes(root, spec, err)) {
            return false;
        }
        // bandwidth 仍为占位段，本期不解析、不消费。

        // ---- 校验（fail-fast）----
        // 先做 partitions 校验，因为它派生出 ccuWindowOffset / userMemSegs
        if (!ValidatePartitions(spec, err) || !ValidateDieOffsets(spec, err) || !ValidateNoOverlap(spec, err)
            || !ValidateEntryFits(spec, err) || !ValidateMs(spec, err) || !ValidateResourceSpaceSize(spec, err)
            || !ValidateDeviceRes(spec, err)) {
            err = absPath + ": " + err;
            return false;
        }

        out = std::move(spec);
        HCCL_VM_INFO(
            "device spec loaded: name={} ccu_version={} die_num={} regions={} "
            "resource_space_size={:#x} user_mem_total={:#x} ({} segs) src={}",
            out.name, CcuVersionToString(out.ccuVersion), out.dieNum, out.regions.size(), out.resourceSpaceSize,
            out.addr.userMemTotal, out.addr.userMemSegs.size(), out.sourcePath);
        return true;
#else
        (void)absPath;
        (void)out;
        err = "device_spec: 编译时未启用 yaml-cpp（HAVE_YAML_CPP 未定义）";
        return false;
#endif
    }

} // namespace devspec
} // namespace sim
