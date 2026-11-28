/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CCU_PROFILING_H
#define HCOMM_CCU_PROFILING_H

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "ccu_rep_context_v1.h"
#include "ccu_types.h"
#include "hccl_types.h"
#include "hcomm_primitives.h"

namespace hcomm {

struct GroupInfo {
    uint16_t loopParamId;
    uint16_t parallelParamId;
    uint16_t residualId;
};

struct GroupOpConfig {
    uint32_t msInterleave;
    uint32_t loopCount;
    uint64_t memSlice;
};

// kernel 执行期 profiling 数据的采集与整理。只读写基类 CcuRepContext，不依赖 CcuKernel；
// 由 CcuKernel 按值持有，对外入口仍在 CcuKernel 上并转发到这里。
class CcuProfiling {
public:
    explicit CcuProfiling(CcuRep::CcuRepContext& repCtx) : repCtx_(repCtx) {}
    CcuProfiling(const CcuProfiling&) = delete;
    CcuProfiling& operator=(const CcuProfiling&) = delete;

    HcclResult SetupProfilingInfo(const char* kernelFuncName);
    // 选定 die 之后刷新已采集的 profiling 信息中的 dieId。
    HcclResult UpdateProfilingDieId(uint32_t dieId);

    HcclResult AddProfilingInfo(
        const ChannelHandle* channels, uint32_t channelNum, HcclDataType dataType, HcclDataType outputDataType,
        HcclReduceOp opType, const std::string& opName);
    HcclResult AddCcuProfiling(
        GroupInfo groupInfo, const std::vector<ChannelHandle>& channelHandle, HcclDataType dataType,
        HcclDataType outputDataType, HcclReduceOp opType, const std::string& opName);
    HcclResult AddCcuProfiling(
        const ChannelHandle* channels, uint32_t channelNum, HcclDataType dataType, HcclDataType outputDataType,
        HcclReduceOp opType, const std::string& opName);

    // 双路输出：ccuStaticInfo 为 SQE/WaitCKE 静态部分（shared_ptr 引用计数共享，0 拷贝），
    // ccuDetailInfo 为 LoopGroup 动态部分（深拷贝，跨 launch 安全）；空时为 nullptr。
    // instrId 由 CcuKernelMgr 在注册后分配，构造时还不存在，因此每次调用时传入。
    HcclResult GetCcuProfilingInfo(
        const uint64_t* taskArgs, uint32_t argSize, std::shared_ptr<std::vector<CcuProfilingInfo>>& ccuStaticInfo,
        std::shared_ptr<std::vector<CcuProfilingInfo>>& ccuDetailInfo, uint32_t instrId);

    // 按 isSaveProfilingData 开关决定是否收集 profiling（为 false 时直接返回不收集）；
    // 收集失败仅 WARNING 不中断 launch（profiling 非关键路径）。
    struct CcuDfxResult {
        std::shared_ptr<std::vector<CcuProfilingInfo>> ccuStaticInfo;
        std::shared_ptr<std::vector<CcuProfilingInfo>> ccuDetailInfo;
    };
    HcclResult ConstructProfilingInfo(
        const uint64_t* taskArgs, uint32_t argNum, CcuDfxResult& result, bool isSaveProfilingData, uint32_t instrId);

    const std::vector<CcuProfilingInfo>& GetAllCcuProfilingInfo() { return *cache_.workVec; }
    const std::string& GetName() const { return name_; }
    const std::vector<GroupInfo>& GetGroupOpSizeInfo() const { return groupOpSizeInfo_; }
    const GroupOpConfig& GetMoConfig() const { return moConfig_; }

private:
    // 预解析的 argIndex 缓存条目：把 LoopGroup 三个 varId 经链式 map 解析后的最终 argIndex
    // （taskArgs 下标）固化为 uint32_t，同一 kernel 第二次起 launch 直接按下标取值，不再查 map。
    struct GroupArgIndex {
        static constexpr uint32_t INVALID = UINT32_MAX;
        uint32_t loopParamArgIdx{INVALID};
        uint32_t parallelParamArgIdx{INVALID};
        uint32_t residualArgIdx{INVALID};
    };

    // 同一 kernel 第二次起 launch 复用的缓存：记录 SQE/WaitCKE 构建结果、varId 映射表、
    // argIndex 预解析结果、SQE/WaitCKE 缓存，第二次起 launch 不再重复构建。
    struct CacheEntry {
        std::shared_ptr<std::vector<CcuProfilingInfo>> workVec = std::make_shared<std::vector<CcuProfilingInfo>>();
        size_t sqeWaitCkeSize{0};
        bool sqeWaitCkeBuilt{false};
        std::unordered_map<uint16_t, uint32_t> varId2ArgIndexMap;
        std::unordered_map<uint16_t, uint16_t> varId2VarIdMap;
        bool varIdMapsCached{false};
        std::vector<GroupArgIndex> groupArgIndexCache;
        bool groupArgIndexCached{false};
        std::shared_ptr<std::vector<CcuProfilingInfo>> sqeWaitCkeCache;
        std::unordered_map<std::string, uint64_t> nameItemIdCache;
    };

    // GetCcuProfilingInfo 的子步骤：处理 sqe & waitcke 类型的 profiling 信息，结果直接 push 到 workVec 中。
    HcclResult CollectSqeAndWaitCkeProfilingInfo(uint32_t instrId);
    // GetCcuProfilingInfo 的子步骤：根据 LoopGroup 的 profiling 缓存构建
    // varId -> argIndex 与 varId -> varId 两个查找表，供 LoopGroup 段查询入参使用。
    HcclResult BuildLoopGroupVarIdMaps(
        std::unordered_map<uint16_t, uint32_t>& varId2ArgIndexMap,
        std::unordered_map<uint16_t, uint16_t>& varId2VarIdMap);
    // GetCcuProfilingInfo 的子步骤：处理 LoopGroup 的 profiling 信息，结果 push 到 workVec 中。
    HcclResult CollectLoopGroupProfilingInfo(
        const uint64_t* taskArgs, uint32_t argSize, const std::unordered_map<uint16_t, uint32_t>& varId2ArgIndexMap,
        const std::unordered_map<uint16_t, uint16_t>& varId2VarIdMap);
    // CollectLoopGroupProfilingInfo 的单次循环体：处理下标 i 对应的 loopParam 与 parallelParam 分支。
    HcclResult CollectSingleLoopGroupProfiling(
        uint32_t i, size_t repSize, const uint64_t* taskArgs, uint32_t argSize,
        const std::unordered_map<uint16_t, uint32_t>& varId2ArgIndexMap,
        const std::unordered_map<uint16_t, uint16_t>& varId2VarIdMap);
    // 链式 map 解析 varId→argIndex：先查 varId2ArgIndexMap，未命中则沿 varId2VarIdMap 回溯
    // 源 varId（支持多层 assign 链 A=B=C），再查 argIndexMap。
    static HcclResult ResolveArgIndex(
        const std::unordered_map<uint16_t, uint16_t>& varId2VarIdMap,
        const std::unordered_map<uint16_t, uint32_t>& varId2ArgIndexMap, uint16_t varId, uint32_t& argIndex);
    HcclResult ResolveArgIndex(
        const std::unordered_map<uint16_t, uint16_t>& varId2VarIdMap,
        const std::unordered_map<uint16_t, uint32_t>& varId2ArgIndexMap, const uint64_t* taskArgs, uint32_t argSize,
        uint16_t varId, uint64_t& argValue);

    CcuRep::CcuRepContext& repCtx_;
    std::string name_{};
    GroupOpConfig moConfig_{0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF};
    std::vector<GroupInfo> groupOpSizeInfo_;
    CacheEntry cache_;
};

} // namespace hcomm

#endif // HCOMM_CCU_PROFILING_H
