/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_ALL_RANK_PARAM_RECORD_V3_H
#define HCCLV2_CCU_ALL_RANK_PARAM_RECORD_V3_H

#include <map>
#include <set>
#include <utility>
#include <vector>
#include <hccl_types.h>
#include "base.h"
#include "log.h"
#include "../task_def_v3.h"
#include "dtype_common.h"

namespace HcclSim {
namespace TaskGraphGeneratorV3 {

    struct CcuPostNodeMetaV3 {
        DeviceId recordDeviceId{INVALID_DEVICE_ID};
        DeviceId waitDeviceId{INVALID_DEVICE_ID};
        uint32_t dieId{INVALID_DIE_ID};
        uint16_t ckeId{INVALID_CCU_CKE};
        uint16_t remainingCkeMask{0};
        bool isLocal{false};
    };

    // 记录CCU中偏移地址的类型
    enum class CcuComponerntType : uint16_t {
        UNKNOWN = 0,
        XN_A6 = 1,
        CKE_A6 = 2,
        PFE_A6 = 3,
        CHANNEL_A6 = 4,
        JETTY_A6 = 5,
        MISSION_A6 = 6,
        LOOP_A6 = 7,
    };

    class AllRankParamRecorder {
    public:
        static AllRankParamRecorder* Global();
        void InitParam();
        void Reset();
        HcclResult CheckAllPostMatch();
        void RegisterPostNode(TaskNode* node, const CcuPostNodeMetaV3& meta);
        uint32_t GetPostRemainingCkeMask(const TaskNode* node) const;
        void SetPostRemainingCkeMask(TaskNode* node, uint32_t remainingCkeMask);
        const CcuPostNodeMetaV3* GetPostNodeMeta(const TaskNode* node) const;

        HcclResult SetXn(DeviceId deviceId, uint32_t dieId, uint16_t xnId, uint64_t xnValue);
        HcclResult SetGSA(DeviceId deviceId, uint32_t dieId, uint16_t gsaId, uint64_t gsaValue);
        HcclResult SetCKE(DeviceId deviceId, uint32_t dieId, uint16_t ckeId, uint16_t ckeValue);
        HcclResult SetHBM(DeviceId deviceId, uint32_t dieId, uint64_t hbmAddr, const std::vector<uint64_t>& data);

        HcclResult GetXn(DeviceId deviceId, uint32_t dieId, uint16_t xnId, uint64_t& xnValue);
        HcclResult GetGSA(DeviceId deviceId, uint32_t dieId, uint16_t gsaId, uint64_t& gsaValue);
        HcclResult GetCKE(DeviceId deviceId, uint32_t dieId, uint16_t ckeId, uint16_t& ckeValue);
        HcclResult GetHBM(DeviceId deviceId, uint32_t dieId, uint64_t hbmAddr, std::vector<uint64_t>& data);
        // 通过MS的地址找到MS的Id
        HcclResult GetMSIdByAddr(uint32_t dieId, uint64_t msAddr, uint16_t& msId);
        // 通过XnId所在的地址值来找到XnId以及寄存器的类型
        HcclResult GetXnAndTypeIdByAddr(uint32_t dieId, uint64_t xnAddr, CcuComponerntType& type, uint16_t& xnId);
        // 通过XnId所在的地址值来找到XnId
        HcclResult GetXnIdByAddr(uint32_t dieId, CcuComponerntType type, uint64_t xnAddr, uint16_t& xnId);
        // 通过XnId所在的地址值来找到XnId
        HcclResult GetAddrByXnId(uint32_t dieId, CcuComponerntType type, uint16_t xnId, uint64_t& xnAddr);
        std::map<uint16_t, uint64_t> GetXnSnapshot(DeviceId deviceId, uint32_t dieId) const;
        std::map<uint16_t, uint64_t> GetGSASnapshot(DeviceId deviceId, uint32_t dieId) const;
        std::map<uint16_t, uint16_t> GetCKESnapshot(DeviceId deviceId, uint32_t dieId) const;

        // 记录 LoopGroup 指令处 xpId/xmId 寄存器的运行时 value（供 convert-ccu-microcode 规则三使用）
        void
        RecordLoopGroupRegSnapshot(DeviceId deviceId, uint32_t dieId, uint32_t pc, uint64_t xpValue, uint64_t xmValue);
        // 返回某 device/die 下 curXn 中已使用的最大 xn 寄存器 Id（供规则一使用），为空时返回 UINT16_MAX
        uint16_t GetMaxUsedXnId(DeviceId deviceId, uint32_t dieId) const;

        DevType GetDevType() const { return devType_; }

        // deviceId -> dieId -> 寄存器Id -> 寄存器value
        std::map<uint32_t, std::map<uint32_t, std::map<uint16_t, uint64_t>>> curXn;
        std::map<uint32_t, std::map<uint32_t, std::map<uint16_t, uint64_t>>> curGSA; // A6没有GSA，A5使用
        std::map<uint32_t, std::map<uint32_t, std::map<uint16_t, uint16_t>>> curCKE;

        std::map<uint32_t, std::map<uint32_t, std::map<uint64_t, std::vector<uint64_t>>>>
            curHBM; // 模拟HBM，记录每个device的每个die的每个HBM的使用情况

        std::map<uint32_t, std::map<uint32_t, std::map<uint16_t, std::set<TaskNode*>>>> seenPost;
        std::map<const TaskNode*, CcuPostNodeMetaV3> postNodeMeta;

        // LoopGroup 寄存器快照表：deviceId -> dieId -> LoopGroup 指令 PC -> (xpValue, xmValue)
        std::map<uint32_t, std::map<uint32_t, std::map<uint32_t, std::pair<uint64_t, uint64_t>>>> loopGroupRegSnapshot;

    public:
        DevType devType_{DevType::DEV_TYPE_COUNT}; // 初始化无效值
        std::vector<uint64_t> ccu_resource_base_addr_{};
    };

} // namespace TaskGraphGeneratorV3
} // namespace HcclSim

#endif
