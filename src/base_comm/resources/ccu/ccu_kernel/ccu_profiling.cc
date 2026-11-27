/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_profiling.h"

#include <cstring>

#include "ccu_log.h"
#include "ccu_rep_v1.h"
#include "ccu_rep_type_v1.h"
#include "ccu_rep_loadarg_v1.h"
#include "ccu_rep_assign_v1.h"
#include "ccu_rep_loopgroup_bundle_v1.h"
#include "dfx/prof_utils_pub.h"
#include "hcomm_c_adpt.h"
#include "securec.h"
#include "../../endpoint_pairs/channels/ccu/ccu_urma_channel.h"

namespace hcomm {

namespace {

    void DumpCcuProfilingInfo(const std::vector<CcuProfilingInfo>& ccuProfilingInfo)
    {
        if (UNLIKELY(!HcclCheckLogLevel(HCCL_LOG_INFO))) {
            return;
        }
        auto dumpLinkInfo = [](const CcuProfilingInfo& info) -> void {
            for (int i = 0; i < CCU_MAX_CHANNEL_NUM; i++) {
                if (info.channelId[i] == INVALID_VALUE_CHANNELID) {
                    continue;
                }
                HCCL_INFO("channelId(%u), remoteRankId(%u).", info.channelId[i], info.remoteRankId[i]);
            }
        };

        for (const auto& profInfo : ccuProfilingInfo) {
            if (profInfo.type == static_cast<uint8_t>(CcuProfilinType::CCU_TASK_PROFILING)) {
                HCCL_INFO(
                    "Dump CCU Profiling Info:SQE Profiling Info: ctxSignature(%s), "
                    "dieId(%d), missionId(%d), instrId(%d).",
                    profInfo.name.c_str(), static_cast<int>(profInfo.dieId), static_cast<int>(profInfo.missionId),
                    static_cast<int>(profInfo.instrId));
            } else if (profInfo.type == static_cast<uint8_t>(CcuProfilinType::CCU_WAITCKE_PROFILING)) {
                HCCL_INFO(
                    "Microcode WaitCKE Profiling Info: name(%s), "
                    "dieId(%d), missionId(%d), instrId(%d), ckeId(%u), mask(%u).",
                    profInfo.name.c_str(), static_cast<int>(profInfo.dieId), static_cast<int>(profInfo.missionId),
                    static_cast<int>(profInfo.instrId), profInfo.ckeId, profInfo.mask);
                dumpLinkInfo(profInfo);
            } else if (profInfo.type == static_cast<uint8_t>(CcuProfilinType::CCU_LOOPGROUP_PROFILING)) {
                HCCL_INFO(
                    "Microcode LoopGroup Profiling Info: name(%s), "
                    "dieId(%d), missionId(%d), instrId(%d), reduceOpType(%d), inputDataType(%d), "
                    "outputDataType(%d), dataSize(%lu).",
                    profInfo.name.c_str(), static_cast<int>(profInfo.dieId), static_cast<int>(profInfo.missionId),
                    static_cast<int>(profInfo.instrId), static_cast<int>(profInfo.reduceOpType),
                    static_cast<int>(profInfo.inputDataType), static_cast<int>(profInfo.outputDataType),
                    profInfo.dataSize);
                dumpLinkInfo(profInfo);
            }
        }
    }

    constexpr uint64_t SetBits(uint16_t end) { return ((uint64_t(1) << (end + 1)) - uint64_t(1)); }

    uint16_t ParseRepeatNumFromParallelParam(uint64_t parallelParam)
    {
        // 与编码侧 ccu_assist.cc GetParallelParam 保持一致：repeatBitNum=7 对应 SetBits(7)=0xFF（8 位掩码）
        constexpr uint16_t repeatBitNum = 7;
        constexpr uint16_t repeatNumShiftBit = 55;
        return (parallelParam >> repeatNumShiftBit) & SetBits(repeatBitNum);
    }

    void ConstructProfilingInfoLog(
        const std::shared_ptr<std::vector<CcuProfilingInfo>>& ccuStaticInfo,
        const std::shared_ptr<std::vector<CcuProfilingInfo>>& ccuDetailInfo)
    {
        if (!HcclCheckLogLevel(HCCL_LOG_INFO)) {
            return;
        }
        auto logFn = [](const std::shared_ptr<std::vector<CcuProfilingInfo>>& info) {
            if (info == nullptr) {
                return;
            }
            for (const CcuProfilingInfo& profInfo : *info) {
                for (int idx = 0; idx < CCU_MAX_CHANNEL_NUM; idx++) {
                    if (profInfo.channelId[idx] == INVALID_VALUE_CHANNELID) {
                        break;
                    }
                    HCCL_INFO(
                        "[%s]idx[%d]: channelId[%u], channelHandle[0x%llx]", __func__, idx, profInfo.channelId[idx],
                        static_cast<unsigned long long>(profInfo.channelHandle[idx]));
                }
            }
        };
        logFn(ccuStaticInfo);
        logFn(ccuDetailInfo);
    }

} // anonymous namespace

HcclResult CcuKernelProfiling::SetupProfilingInfo(const char* kernelFuncName)
{
    if (kernelFuncName == nullptr || strlen(kernelFuncName) == 0) {
        name_ = std::string("CCU_KERNEL"); // 默认名称
        repCtx_.AddSqeProfiling(name_);
        return HcclResult::HCCL_SUCCESS;
    }

    constexpr size_t MAX_KERNEL_FUNC_NAME_LEN = 128;
    const auto nameLen = strlen(kernelFuncName);
    if (nameLen > MAX_KERNEL_FUNC_NAME_LEN) {
        name_ = std::string(kernelFuncName, MAX_KERNEL_FUNC_NAME_LEN);
        HCCL_WARNING("[CcuKernel][%s] kernelFuncName is too long, reset to %s.", __func__, name_.c_str());
    } else {
        name_ = std::string(kernelFuncName);
    }

    // 生成SQE粒度profiling信息，此时未选择die，默认die 0
    repCtx_.AddSqeProfiling(name_);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelProfiling::UpdateProfilingDieId(uint32_t dieId)
{
    if (dieId == 0) {
        return HcclResult::HCCL_SUCCESS;
    }
    auto& profilingInfos = repCtx_.profilingInfo;
    if (UNLIKELY(profilingInfos.empty())) {
        HCCL_INFO("[%s] passed, profiling infos are empty, ccu kernel func[%s].", __func__, name_.c_str());
        return HcclResult::HCCL_SUCCESS;
    }
    for (auto& info : profilingInfos) {
        info.dieId = dieId;
    }
    HCCL_INFO("[%s] reset profiling info dieId to [%u], ccu kernel func[%s].", __func__, dieId, name_.c_str());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelProfiling::ResolveArgIndex(
    const std::unordered_map<uint16_t, uint16_t>& varId2VarIdMap,
    const std::unordered_map<uint16_t, uint32_t>& varId2ArgIndexMap, uint16_t varId, uint32_t& argIndex)
{
    auto item = varId2ArgIndexMap.find(varId);
    if (item == varId2ArgIndexMap.end()) {
        uint16_t oriVarId = varId;
        auto iter = varId2VarIdMap.find(varId);
        while (iter != varId2VarIdMap.end()) {
            oriVarId = iter->second;
            iter = varId2VarIdMap.find(oriVarId);
        }
        if (oriVarId != varId) {
            item = varId2ArgIndexMap.find(oriVarId);
            if (item == varId2ArgIndexMap.end()) {
                HCCL_ERROR("[%s] fail, Invalid goSize variable id(%u), oriVarId(%u)", __func__, varId, oriVarId);
                return HCCL_E_PARA;
            }
        } else {
            HCCL_ERROR("[%s]fail, Invalid goSize variable id(%u)", __func__, varId);
            return HCCL_E_PARA;
        }
    }
    argIndex = item->second;
    return HCCL_SUCCESS;
}

HcclResult CcuKernelProfiling::ResolveArgIndex(
    const std::unordered_map<uint16_t, uint16_t>& varId2VarIdMap,
    const std::unordered_map<uint16_t, uint32_t>& varId2ArgIndexMap, const uint64_t* taskArgs, uint32_t argSize,
    uint16_t varId, uint64_t& argValue)
{
    uint32_t idx = 0;
    HcclResult ret = ResolveArgIndex(varId2VarIdMap, varId2ArgIndexMap, varId, idx);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] ResolveArgIndex failed, ret[%d], varId(%u)", __func__, ret, varId);
        return ret;
    }
    if (idx >= argSize) {
        HCCL_ERROR("[%s] Invalid goSize variable index(%u), argSize(%u).", __func__, idx, argSize);
        return HCCL_E_PARA;
    }
    argValue = taskArgs[idx];
    return HCCL_SUCCESS;
}

HcclResult CcuKernelProfiling::CollectSqeAndWaitCkeProfilingInfo(uint32_t instrId)
{
    auto& vec = *cache_.workVec;
    auto& ccuProfilingCache = repCtx_.GetProfilingInfo();
    const auto& waitCkeReps = repCtx_.GetWaiteCkeProfilingReps();
    uint32_t count{0};
    HCCL_INFO("[%s] Process sqe&waitcke profiling info start.", __func__);
    for (auto& profInfo : ccuProfilingCache) {
        profInfo.missionId = repCtx_.GetMissionId();
        if (profInfo.type == static_cast<uint8_t>(hcomm::CcuProfilinType::CCU_TASK_PROFILING)) {
            profInfo.instrId = instrId;
            vec.push_back(profInfo);
            continue;
        }
        if (count >= waitCkeReps.size()) {
            HCCL_ERROR(
                "count[%u] out of range[0, %zu], cache size(%zu).", count, waitCkeReps.size(),
                ccuProfilingCache.size());
            return HCCL_E_INTERNAL;
        }
        auto waitCkeRep = waitCkeReps[count];
        if (waitCkeRep.get() == nullptr) {
            HCCL_ERROR("[%s] localWaitRep is nullptr.", __func__);
            return HCCL_E_PTR;
        }
        profInfo.instrId = waitCkeRep->StartInstrId();
        if (profInfo.ckeId == INVALID_CKE_ID) {
            profInfo.ckeId = waitCkeRep->GetId();
            HCCL_INFO("[%s] waitcke[%u]", __func__, profInfo.ckeId);
        }
        vec.push_back(profInfo);
        count++;
    }
    return HCCL_SUCCESS;
}

HcclResult CcuKernelProfiling::BuildLoopGroupVarIdMaps(
    std::unordered_map<uint16_t, uint32_t>& varId2ArgIndexMap, std::unordered_map<uint16_t, uint16_t>& varId2VarIdMap)
{
    auto& lgProfInfo = repCtx_.GetLGProfilingInfo();
    HCCL_INFO("[%s] create varId2ArgIndexMap start. size=%lu", __func__, lgProfInfo.loadRep2ArgIdxMap.size());
    for (auto& iter : lgProfInfo.loadRep2ArgIdxMap) {
        if (iter.first.get() == nullptr) {
            HCCL_ERROR("[%s] loadRep is nullptr.", __func__);
            return HCCL_E_PTR;
        }
        auto* loadRep = static_cast<CcuRep::CcuRepLoadArg*>(iter.first.get());
        if (loadRep->Type() != CcuRep::CcuRepType::LOAD_ARG) {
            HCCL_ERROR("[%s] unexpected rep type for loadRep", __func__);
            return HCCL_E_INTERNAL;
        }
        varId2ArgIndexMap[loadRep->GetVarId()] = iter.second;
    }

    HCCL_INFO("[%s] create varId2VarIdMap start. size=%lu", __func__, lgProfInfo.assignProfilingReps.size());
    for (auto& iter : lgProfInfo.assignProfilingReps) {
        if (iter.get() == nullptr) {
            HCCL_ERROR("[%s] assignRep is nullptr.", __func__);
            return HCCL_E_PTR;
        }
        auto* assignRep = static_cast<CcuRep::CcuRepAssign*>(iter.get());
        if (assignRep->Type() != CcuRep::CcuRepType::ASSIGN) {
            HCCL_ERROR("[%s] unexpected rep type for assignRep", __func__);
            return HCCL_E_INTERNAL;
        }
        varId2VarIdMap[assignRep->GetVarB().Id()] = assignRep->GetVarA().Id();
    }
    return HCCL_SUCCESS;
}

HcclResult CcuKernelProfiling::CollectSingleLoopGroupProfiling(
    uint32_t i, size_t repSize, const uint64_t* taskArgs, uint32_t argSize,
    const std::unordered_map<uint16_t, uint32_t>& varId2ArgIndexMap,
    const std::unordered_map<uint16_t, uint16_t>& varId2VarIdMap)
{
    if (argSize == 0) {
        return HCCL_SUCCESS;
    }
    auto& lgProfInfo = repCtx_.GetLGProfilingInfo();
    uint64_t loopParam{0};
    uint64_t parallelParam{0};

    // 同一 kernel 第二次起 launch 缓存命中：直接从 groupArgIndexCache 取 argIndex，不再查 map
    if (cache_.groupArgIndexCached && i < cache_.groupArgIndexCache.size()) {
        const auto& argIdx = cache_.groupArgIndexCache[i];
        if (argIdx.loopParamArgIdx != GroupArgIndex::INVALID && argIdx.loopParamArgIdx < argSize
            && argIdx.parallelParamArgIdx != GroupArgIndex::INVALID && argIdx.parallelParamArgIdx < argSize) {
            loopParam = taskArgs[argIdx.loopParamArgIdx];
            parallelParam = taskArgs[argIdx.parallelParamArgIdx];
        } else {
            if (varId2ArgIndexMap.empty()) {
                return HCCL_SUCCESS;
            }
            CHK_RET(ResolveArgIndex(
                varId2VarIdMap, varId2ArgIndexMap, taskArgs, argSize, groupOpSizeInfo_[i].loopParamId, loopParam));
            CHK_RET(ResolveArgIndex(
                varId2VarIdMap, varId2ArgIndexMap, taskArgs, argSize, groupOpSizeInfo_[i].parallelParamId,
                parallelParam));
        }
    } else {
        if (varId2ArgIndexMap.empty()) {
            return HCCL_SUCCESS;
        }
        CHK_RET(ResolveArgIndex(
            varId2VarIdMap, varId2ArgIndexMap, taskArgs, argSize, groupOpSizeInfo_[i].loopParamId, loopParam));
        CHK_RET(ResolveArgIndex(
            varId2VarIdMap, varId2ArgIndexMap, taskArgs, argSize, groupOpSizeInfo_[i].parallelParamId, parallelParam));
    }
    HCCL_INFO(
        "Collect loopgroup profiling info: repSize[%zu], index[%u],"
        "loopParam[%lu], parallelParam[%lu].",
        lgProfInfo.lgProfilingReps.size(), i, loopParam, parallelParam);

    if (loopParam != 0) {
        if (lgProfInfo.lgProfilingReps[i].get() == nullptr) {
            HCCL_ERROR("[CollectSingleLoopGroupProfiling] lgProfilingRep[%u] is nullptr, repSize(%zu).", i, repSize);
            return HCCL_E_PTR;
        }
        auto& profInfo = lgProfInfo.ccuProfilingInfos[i];
        if (moConfig_.loopCount != 0xFFFFFFFF && moConfig_.memSlice != 0xFFFFFFFFFFFFFFFF) {
            profInfo.dataSize = loopParam * moConfig_.loopCount * moConfig_.memSlice;
        } else {
            profInfo.dataSize = 0;
        }
        profInfo.instrId
            = static_cast<CcuRep::CcuRepLoopGroupBundle*>(lgProfInfo.lgProfilingReps[i].get())->StartInstrId();
        cache_.workVec->push_back(profInfo);
    }

    if (parallelParam != 0) {
        HCCL_INFO("[CollectSingleLoopGroupProfiling] collect lg, residual start i=%u", i);
        uint64_t residual{0};
        if (cache_.groupArgIndexCached && i < cache_.groupArgIndexCache.size()
            && cache_.groupArgIndexCache[i].residualArgIdx != GroupArgIndex::INVALID
            && cache_.groupArgIndexCache[i].residualArgIdx < argSize) {
            residual = taskArgs[cache_.groupArgIndexCache[i].residualArgIdx];
        } else {
            CHK_RET(ResolveArgIndex(
                varId2VarIdMap, varId2ArgIndexMap, taskArgs, argSize, groupOpSizeInfo_[i].residualId, residual));
        }
        uint64_t repeatNum = ParseRepeatNumFromParallelParam(parallelParam);
        auto& profInfo = lgProfInfo.ccuProfilingInfos[i];
        if (moConfig_.memSlice != 0xFFFFFFFFFFFFFFFF) {
            profInfo.dataSize = repeatNum * moConfig_.memSlice + residual;
        } else {
            profInfo.dataSize = 0;
        }
        if (i + 1 >= repSize) {
            HCCL_WARNING(
                "[CollectLoopGroupProfilingInfo] parallelParam != 0 but no paired rep,"
                " index(%u), repSize(%zu), skip parallelParam profiling.",
                i, repSize);
            return HCCL_SUCCESS;
        }
        if (lgProfInfo.lgProfilingReps[i + 1].get() == nullptr) {
            HCCL_ERROR(
                "[CollectSingleLoopGroupProfiling] lgProfilingRep[%u] is nullptr, repSize(%zu).", i + 1, repSize);
            return HCCL_E_PTR;
        }
        profInfo.instrId
            = static_cast<CcuRep::CcuRepLoopGroupBundle*>(lgProfInfo.lgProfilingReps[i + 1].get())->StartInstrId();
        cache_.workVec->push_back(profInfo);
    }
    return HCCL_SUCCESS;
}

HcclResult CcuKernelProfiling::CollectLoopGroupProfilingInfo(
    const uint64_t* taskArgs, uint32_t argSize, const std::unordered_map<uint16_t, uint32_t>& varId2ArgIndexMap,
    const std::unordered_map<uint16_t, uint16_t>& varId2VarIdMap)
{
    auto& lgProfInfo = repCtx_.GetLGProfilingInfo();
    HCCL_INFO(
        "[%s] process loop group profiling start: "
        "lgsize(%lu), goSize(%lu)",
        __func__, lgProfInfo.lgProfilingReps.size(), groupOpSizeInfo_.size());
    size_t repSize = lgProfInfo.lgProfilingReps.size();
    size_t goSizeNum = groupOpSizeInfo_.size();
    size_t profSize = lgProfInfo.ccuProfilingInfos.size();
    size_t safeSize = (repSize < goSizeNum) ? repSize : goSizeNum;
    safeSize = (profSize < safeSize) ? profSize : safeSize;
    if (safeSize != repSize) {
        HCCL_WARNING(
            "[CollectLoopGroupProfilingInfo] size mismatch: lgReps[%zu], goSize[%zu], ccuProf[%zu],"
            " use safeSize[%zu].",
            repSize, goSizeNum, profSize, safeSize);
    }
    // 步长 2：一次通信任务拆成两段下发——主循环段(loopParam 完整循环次数)和不足一轮的
    // 剩余段(parallelParam)，各建一个 LoopGroupBundle；故相邻两个 rep 属于同一任务，
    // rep[i] 取主循环段、rep[i+1] 取剩余段(见 CollectSingleLoopGroupProfiling)。
    for (uint32_t i = 0; i < safeSize; i += 2) {
        CHK_RET(CollectSingleLoopGroupProfiling(i, repSize, taskArgs, argSize, varId2ArgIndexMap, varId2VarIdMap));
    }
    return HCCL_SUCCESS;
}

/*
 * variable/maskSignal等资源变量Id，一定要在获取ccu profiling时才获取；
 * 原因：在创建context Rep时，其资源Id属于虚拟资源；翻译时，才会绑定固定的物理资源。
 *
 * 同一 kernel 第二次起 launch 的缓存复用机制：
 * 第一次 launch 构建 SQE/WaitCKE + varId映射 + argIndex预解析 + SQE/WaitCKE 缓存；
 * 第二次起只 resize 截掉 loopGroup 尾部、跳过映射构建、直接共享 SQE/WaitCKE 缓存不再重建。
 */
HcclResult CcuKernelProfiling::GetCcuProfilingInfo(
    const uint64_t* taskArgs, uint32_t argSize, std::shared_ptr<std::vector<CcuProfilingInfo>>& ccuStaticInfo,
    std::shared_ptr<std::vector<CcuProfilingInfo>>& ccuDetailInfo, uint32_t instrId)
{
    HCCL_INFO("[GetCcuProfilingInfo] Enter.");
    auto& vec = *cache_.workVec;

    // 第一次 launch 构建 SQE/WaitCKE；第二次起跳过
    if (!cache_.sqeWaitCkeBuilt) {
        vec.clear();
        auto& ccuProfilingCache = repCtx_.GetProfilingInfo();
        auto& lgProfInfo = repCtx_.GetLGProfilingInfo();
        size_t estimatedSize = ccuProfilingCache.size() + lgProfInfo.ccuProfilingInfos.size() * 2;
        if (vec.capacity() < estimatedSize) {
            vec.reserve(estimatedSize);
        }
        CHK_RET(CollectSqeAndWaitCkeProfilingInfo(instrId));
        cache_.sqeWaitCkeSize = vec.size();
        cache_.sqeWaitCkeBuilt = true;
    } else {
        // 第二次起：截断 loopGroup 尾部，保留 sqe/waitCke
        // instrId 跨 launch 恒定（SetInstrId 仅在 kernel 注册时调用一次），首次构建已写入，无需回填
        vec.resize(cache_.sqeWaitCkeSize);
    }

    // 第一次构建 varId 映射 + 预解析 argIndex；第二次起跳过
    if (!cache_.varIdMapsCached) {
        CHK_RET(BuildLoopGroupVarIdMaps(cache_.varId2ArgIndexMap, cache_.varId2VarIdMap));
        cache_.groupArgIndexCache.clear();
        cache_.groupArgIndexCache.reserve(groupOpSizeInfo_.size());
        for (const auto& info : groupOpSizeInfo_) {
            GroupArgIndex argIdx{};
            (void)ResolveArgIndex(
                cache_.varId2VarIdMap, cache_.varId2ArgIndexMap, info.loopParamId, argIdx.loopParamArgIdx);
            (void)ResolveArgIndex(
                cache_.varId2VarIdMap, cache_.varId2ArgIndexMap, info.parallelParamId, argIdx.parallelParamArgIdx);
            (void)ResolveArgIndex(
                cache_.varId2VarIdMap, cache_.varId2ArgIndexMap, info.residualId, argIdx.residualArgIdx);
            cache_.groupArgIndexCache.push_back(argIdx);
        }
        cache_.varIdMapsCached = true;
        cache_.groupArgIndexCached = true;
    }

    CHK_RET(CollectLoopGroupProfilingInfo(taskArgs, argSize, cache_.varId2ArgIndexMap, cache_.varId2VarIdMap));

    // itemId 预填充：name 跨 launch 不变，GetProfHashId 返回值跨 launch 不变。
    // 首次构建时填充（itemId == 0 表示未填充），后续 launch 直接复用。
    // SQE/WaitCKE 部分：随 sqeWaitCkeCache shared_ptr 共享，后续 launch 不再重算。
    // LoopGroup 部分：每次深拷贝复制 itemId，后续 launch 不再重算。
    // name→itemId 映射在 per-comm cache 内去重，避免同 name 重复调 ProfStr2Id。
    // 仅缓存非 0 值：stub 返回 0 时不缓存，避免 dlsym 延迟成功后 cache 阻止动态修复。
    for (auto& info : vec) {
        if (info.itemId == 0 && !info.name.empty()) {
            auto it = cache_.nameItemIdCache.find(info.name);
            if (it != cache_.nameItemIdCache.end()) {
                info.itemId = it->second;
            } else {
                info.itemId = hcomm::ProfStr2Id(info.name.c_str(), static_cast<uint32_t>(info.name.length()));
                if (info.itemId != 0) {
                    cache_.nameItemIdCache[info.name] = info.itemId;
                }
            }
        }
    }

    DumpCcuProfilingInfo(vec);

    // SQE/WaitCKE 缓存：第一次 launch 拷一份存起来，后续 launch 直接共享不再重建
    // 冻结前校验静态段 itemId 全非 0：存在 0 值（dlsym 延迟未成功）时推迟冻结，
    // 保持 nullptr，下次 launch 重新检查；避免冻结副本阻止动态修复
    if (cache_.sqeWaitCkeCache == nullptr && cache_.sqeWaitCkeBuilt && cache_.sqeWaitCkeSize > 0) {
        bool hasZeroItemId = false;
        for (size_t i = 0; i < cache_.sqeWaitCkeSize; i++) {
            if (vec[i].itemId == 0) {
                hasZeroItemId = true;
                break;
            }
        }
        if (!hasZeroItemId) {
            cache_.sqeWaitCkeCache
                = std::make_shared<std::vector<CcuProfilingInfo>>(vec.begin(), vec.begin() + cache_.sqeWaitCkeSize);
        }
    }
    ccuStaticInfo = cache_.sqeWaitCkeCache;
    // 动态部分深拷贝尾部；无 LoopGroup 时置 nullptr，下游判空跳过
    if (vec.size() > cache_.sqeWaitCkeSize) {
        ccuDetailInfo = std::make_shared<std::vector<CcuProfilingInfo>>(vec.begin() + cache_.sqeWaitCkeSize, vec.end());
    } else {
        ccuDetailInfo = nullptr;
    }
    return HCCL_SUCCESS;
}

HcclResult CcuKernelProfiling::ConstructProfilingInfo(
    const uint64_t* taskArgs, uint32_t argNum, CcuDfxResult& result, bool isSaveProfilingData, uint32_t instrId)
{
    if (!isSaveProfilingData) {
        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult ret = GetCcuProfilingInfo(taskArgs, argNum, result.ccuStaticInfo, result.ccuDetailInfo, instrId);
    if (ret != HCCL_SUCCESS) {
        HCCL_WARNING("[%s] GetCcuProfilingInfo failed, ret[%d], skip profiling but continue launch", __func__, ret);
        return HcclResult::HCCL_SUCCESS;
    }
    if ((result.ccuStaticInfo == nullptr || result.ccuStaticInfo->empty())
        && (result.ccuDetailInfo == nullptr || result.ccuDetailInfo->empty())) {
        return HcclResult::HCCL_SUCCESS;
    }
    ConstructProfilingInfoLog(result.ccuStaticInfo, result.ccuDetailInfo);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuKernelProfiling::AddProfilingInfo(
    const ChannelHandle* channels, uint32_t channelNum, HcclDataType dataType, HcclDataType outputDataType,
    HcclReduceOp opType, const std::string& opName)
{
    CHK_PTR_NULL(channels);
    if (channelNum > CCU_MAX_CHANNEL_NUM) {
        HCCL_ERROR("[%s] channelNum[%u] exceeds CCU_MAX_CHANNEL_NUM[%u]", __func__, channelNum, CCU_MAX_CHANNEL_NUM);
        return HCCL_E_PARA;
    }
    repCtx_.ccuProfilingInfoCache.type = static_cast<uint8_t>(CcuProfilinType::CCU_LOOPGROUP_PROFILING);
    repCtx_.ccuProfilingInfoCache.name = opName;
    repCtx_.ccuProfilingInfoCache.reduceOpType = opType;
    repCtx_.ccuProfilingInfoCache.inputDataType = dataType;
    repCtx_.ccuProfilingInfoCache.outputDataType = outputDataType;
    repCtx_.ccuProfilingInfoCache.missionId = repCtx_.GetMissionId();

    CHK_SAFETY_FUNC_RET(memset_s(
        repCtx_.ccuProfilingInfoCache.channelId, sizeof(repCtx_.ccuProfilingInfoCache.channelId),
        INVALID_VALUE_CHANNELID, sizeof(repCtx_.ccuProfilingInfoCache.channelId)));
    for (uint32_t i = 0; i < channelNum; i++) {
        void* channelPtr{nullptr};
        CHK_RET(static_cast<HcclResult>(HcommChannelGet(channels[i], &channelPtr)));
        auto* channelImpl = dynamic_cast<CcuUrmaChannel*>(static_cast<Channel*>(channelPtr));
        CHK_PTR_NULL(channelImpl);
        repCtx_.ccuProfilingInfoCache.channelId[i] = channelImpl->GetChannelId();
        repCtx_.ccuProfilingInfoCache.channelHandle[i] = channels[i];
        HCCL_INFO(
            "[%s]type[%d], name[%s], opType[%d], dataType[%d], outputDataType[%d], missionId[%u], "
            "channelHandle[0x%llx], channelId[%u]",
            __func__, repCtx_.ccuProfilingInfoCache.type, repCtx_.ccuProfilingInfoCache.name.c_str(), opType, dataType,
            outputDataType, repCtx_.ccuProfilingInfoCache.missionId,
            static_cast<unsigned long long>(repCtx_.ccuProfilingInfoCache.channelHandle[i]),
            repCtx_.ccuProfilingInfoCache.channelId[i]);
    }
    repCtx_.lgProfilingInfo.ccuProfilingInfos.push_back(repCtx_.ccuProfilingInfoCache);
    if (repCtx_.allLgProfilingReps.empty()) {
        HCCL_WARNING("[%s] allLgProfilingReps is empty, skip LoopGroup rep association", __func__);
    } else {
        repCtx_.lgProfilingInfo.lgProfilingReps.push_back(repCtx_.allLgProfilingReps.back());
    }
    return HCCL_SUCCESS;
}

HcclResult CcuKernelProfiling::AddCcuProfiling(
    GroupInfo groupInfo, const std::vector<ChannelHandle>& channelHandle, HcclDataType dataType,
    HcclDataType outputDataType, HcclReduceOp opType, const std::string& opName)
{
    CHK_RET(AddCcuProfiling(channelHandle.data(), channelHandle.size(), dataType, outputDataType, opType, opName));
    groupOpSizeInfo_.push_back(groupInfo);
    return HCCL_SUCCESS;
}

HcclResult CcuKernelProfiling::AddCcuProfiling(
    const ChannelHandle* channels, uint32_t channelNum, HcclDataType dataType, HcclDataType outputDataType,
    HcclReduceOp opType, const std::string& opName)
{
    CHK_PTR_NULL(channels);
    CHK_RET(AddProfilingInfo(channels, channelNum, dataType, outputDataType, opType, opName));
    return HCCL_SUCCESS;
}

} // namespace hcomm
