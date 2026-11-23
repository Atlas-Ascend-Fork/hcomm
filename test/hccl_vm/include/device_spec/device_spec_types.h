/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIM_DEVICE_SPEC_TYPES_H
#define SIM_DEVICE_SPEC_TYPES_H

#include <cstdint>
#include <string>
#include <vector>

namespace sim {
namespace devspec {

    /**
     * @brief CCU 资源区域类型。
     *
     * 取代 runner 的 CcuComponerntType（ccu_resource_manager.h:41-50）与 checker 的同名复制品
     * （ccu_all_rank_param_recorder_v3.h:37-46）：去掉 _A6 后缀（该枚举本是版本无关的）、
     * 修正 "Componernt" 拼写、补齐 INSTR/CCUM/GSA/WQE/MS（原枚举只有 A6 的 7 个可反查区域）。
     */
    enum class CcuResType : uint16_t {
        UNKNOWN = 0,
        INSTR = 1, ///< 指令空间（hcomm 术语 CCUM = CCU Microcode）
        CCUM = 2,  ///< 预留：hcomm 把 CCUM 与 INS 视为同一段，此处保留枚举位以便 V3 拆分
        GSA = 3,   ///< A5 独有；A6 无 GSA 寄存器堆（但 V2 微码仍有 gsaOffset 位域作地址步长）
        XN = 4,
        CKE = 5,
        PFE = 6,
        CHANNEL = 7,
        JETTY = 8,
        MISSION = 9,
        LOOP = 10,
        WQE = 11, ///< SQ buffer 区；本期两份 device yaml 均不建模（设计文档 O12/D16）
        MS = 12,  ///< memory slice；地址由 ms.ccua_windows 交织计算，不走 regions 表
    };

    /// CCU 微码/资源规格版本。V3 为预留（hcomm ccu_res_specs.h:25,28-31 已定义）。
    enum class CcuVersion : uint8_t {
        CCU_INVALID = 0,
        CCU_V1 = 1, ///< A5 / Ascend950
        CCU_V2 = 2, ///< A6 / Ascend960
        CCU_V3 = 3, ///< 预留：CCUA 4→8、资源空间 128MB、私有/共享分块
    };

    const char* CcuResTypeToString(CcuResType type);
    bool CcuResTypeFromString(const std::string& str, CcuResType& out);
    const char* CcuVersionToString(CcuVersion ver);
    bool CcuVersionFromString(const std::string& str, CcuVersion& out);

    /// 单个 CCU 资源区域（yaml ccu.regions[] 的一项）。
    struct RegionSpec {
        std::string name;
        CcuResType type{CcuResType::UNKNOWN};
        uint64_t offset{0};    ///< 相对 ccuDieBase
        uint64_t size{0};      ///< 区域地址空间大小
        uint32_t entrySize{0}; ///< 单个资源的字节步进；0 表示不按 id 编址（不透明块）
        uint32_t count{0};     ///< 硬件容量（= 上报 hcomm 的 caps 口径）
    };

    /// MS 的单个 CCUA bank 窗口（yaml ccu.ms.ccua_windows[] 的一项）。
    struct MsWindow {
        uint32_t index{0};
        uint64_t offset{0}; ///< 相对 ccuDieBase
        uint64_t size{0};
    };

    /// MS（memory slice）布局。A5/A6 几何完全相同，仅 ccua_windows[].offset 不同。
    struct MsSpec {
        uint32_t msSize{0};     ///< 单个 MS 容量（4096）
        uint32_t msNum{0};      ///< MS 总数（1536）
        uint32_t ccuaNum{0};    ///< CCUA bank 数（V1/V2 = 4，V3 = 8）
        uint32_t interleave{0}; ///< 每 group 交织到各 CCUA 的 MS 数（8）
        std::vector<MsWindow> windows;
    };

    /// rank 窗口内的一个分区段。user_mem 会出现多段（被 ccu 窗口切开）。
    struct Partition {
        std::string name;   ///< "user_mem" | "ccu"
        uint64_t offset{0}; ///< 相对 rankBase
        uint64_t size{0};
    };

    /// 全局地址空间划分（芯片无关，A5/A6 逐字段相同）。
    struct AddrSpace {
        uint64_t rankBase{0};
        uint64_t rankStride{0};
        uint64_t rankWindowSize{0};
        std::vector<Partition> partitions;   ///< yaml 原样保留（甲 schema）
        std::vector<uint64_t> ccuDieOffsets; ///< 各 die 相对 CCU 窗口起点的偏移
        uint64_t ccuDieSize{0};

        // ---- 以下为加载器派生量，不在 yaml 中出现 ----
        uint64_t ccuWindowOffset{0};        ///< partitions[name=="ccu"].offset
        uint64_t ccuWindowSize{0};          ///< partitions[name=="ccu"].size
        std::vector<Partition> userMemSegs; ///< partitions 中 name=="user_mem"，按 offset 升序
        uint64_t userMemTotal{0};           ///< Σ userMemSegs.size
    };

    /// 上报给 hcomm 的能力寄存器口径。全部由 regions/ms 推导，yaml 中不设 caps 段。
    struct CcuCaps {
        uint32_t instructionNum{0};
        uint32_t missionNum{0};
        uint32_t loopEngineNum{0};
        uint32_t gsaNum{0};
        uint32_t xnNum{0};
        uint32_t ckeNum{0};
        uint32_t msNum{0};
        uint32_t channelNum{0};
        uint32_t jettyNum{0};
        uint32_t pfeNum{0};
    };

    struct UbJettySpec {
        uint32_t normalPoolSize{0};
    };

    struct DeviceResSpec {
        uint32_t streamNum{0};
        uint32_t notifyNum{0};
        UbJettySpec ubJetty;
    };

    /// 一份 device 规格（对应一个 asset/cluster_model/config/device/<name>.yaml）。
    struct DeviceSpec {
        uint32_t specVersion{0};
        std::string name; ///< = 文件名 = 阶段 0 的匹配键
        std::string description;
        /// 决策 (i)：阶段 0 加载器【完全忽略】yaml 的 soc.soc_version，本成员恒为空串。
        /// 匹配只认 name（大小写不敏感最长前缀）。见设计文档 §4.7.4。
        std::string socVersion;
        CcuVersion ccuVersion{CcuVersion::CCU_INVALID};
        uint32_t dieNum{0};
        DeviceResSpec deviceRes;
        uint32_t vectorCoreNum{0};

        AddrSpace addr;
        std::vector<RegionSpec> regions;
        MsSpec ms;
        uint32_t reservedSyncSlots{0};
        uint32_t countXnNum{0};
        /// hcomm 权威值 CCU_V1_V2_RESOURCE_SIZE（72 MiB）/ CCU_V3_RESOURCE_SIZE（128 MiB）。
        /// = backingSize。★ 不是 max(regions ∪ ms 末端)——见设计文档 §4.5.2 / D10-补。
        uint64_t resourceSpaceSize{0};

        /// 源文件绝对路径（诊断用）。
        std::string sourcePath;

        CcuCaps GetCaps() const;
        uint64_t BackingSize() const { return resourceSpaceSize; }
        const RegionSpec* FindRegion(CcuResType type) const;
        bool HasGsa() const { return FindRegion(CcuResType::GSA) != nullptr; }
        /// A5(V1) 指令用 id 编址，不做地址反查；A6(V2)/V3 才需要 AddrToResId。
        bool SupportsAddrDecode() const { return ccuVersion != CcuVersion::CCU_V1; }
        /// max(regions ∪ ms 窗口 的 offset+size)，用于校验 8。
        uint64_t MaxModelledEnd() const;
    };

} // namespace devspec
} // namespace sim

#endif // SIM_DEVICE_SPEC_TYPES_H
