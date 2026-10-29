/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_rep_translator_v1.h"
#include "ccu_device.h"

#include <algorithm>

#include "exception_util.h"
#include "log.h"
#include "ccu_log.h"
#include "ccu_api_exception.h"
#include "ccu_rep_loopcall_v1.h"
#include "ccu_rep_funccall_v1.h"
#include "ccu_rep_type_v1.h"
#include "ccu_rep_loop_v1.h"
#include "ccu_rep_loadarg_v1.h"
#include "ccu_assist_v1.h"

#include "ccu_dev_mgr_imp.h"
#include "dtype_common.h"

#include "ccu_ins_generator_base.h"
#include "ccu_ins_generator_v1.h"
#include "ccu_ins_generator_v2.h"
#include "../../../ccu_device/ccu_res_specs.h"

#include "config_plf_log_v2.h"

#include "microcode_optimizer.h"

namespace hcomm {
namespace CcuRep {

    using Hccl::PLF_TASK;

    template <typename T>
    bool CheckType(const std::shared_ptr<CcuRepBlock>& refer)
    {
        if (refer == nullptr) {
            HCCL_ERROR("[%s] input refer is nullptr", __func__);
            return false;
        }
        HCCL_INFO("[CheckType] refer->Type() = %d", refer->Type());
        return false;
    }

    template <>
    bool CheckType<CcuRepFuncBlock>(const std::shared_ptr<CcuRepBlock>& refer)
    {
        if (refer == nullptr) {
            HCCL_ERROR("[%s] input refer is nullptr", __func__);
            return false;
        }
        return refer->Type() == CcuRepType::FUNC_BLOCK ? true : false;
    }

    template <>
    bool CheckType<CcuRepLoopBlock>(const std::shared_ptr<CcuRepBlock>& refer)
    {
        if (refer == nullptr) {
            HCCL_ERROR("[%s] input refer is nullptr", __func__);
            return false;
        }
        return refer->Type() == CcuRepType::LOOP_BLOCK ? true : false;
    }

    template <typename T1, typename T2>
    CcuResult CcuRepTranslator::BuildReference(const std::shared_ptr<CcuRepBase>& rep)
    {
        auto caller = std::static_pointer_cast<T1>(rep);
        auto label = caller->GetLabel();
        // 特例：针对函数地址调用，不需要依靠函数名索引
        if (label == "") {
            return CCU_SUCCESS;
        }
        std::shared_ptr<CcuRepBlock> refer;
        CCU_TRY_CATCH_RETURN(refer = refManager->GetRefBlock(label););
        if (CheckType<T2>(refer)) {
            caller->Reference(std::static_pointer_cast<T2>(refer));
        } else {
            HCCL_ERROR("[%s] Invalid Reference: %s", __func__, label.c_str());
            return CCU_E_INTERNAL;
        }
        return CCU_SUCCESS;
    }

    CcuRepTranslator::CcuRepTranslator(
        int32_t userDevId, uint8_t dieId, std::shared_ptr<CcuRepReferenceManager> refManager,
        std::array<uint16_t, CCU_MAX_IODIE_NUM>& reserverChannalId, std::pair<uint64_t, uint64_t>& ccuTokenInfo,
        uint64_t hbmTokenInfo)
        : refManager(refManager)
    {
        transDep.userDevId = userDevId;
        transDep.dieId = dieId;
        s32 result = memcpy_s(
            transDep.reserveChannalId, sizeof(transDep.reserveChannalId), reserverChannalId.data(),
            sizeof(reserverChannalId));
        if (result != 0) {
            Hccl::THROW<Hccl::InternalException>(Hccl::StringFormat(
                "[NsRecovery] CcuRepTranslator::CcuRepTranslator: memcpy_s failed, ret = %d", result));
        }

        HcclResult ret = CcuDevMgrImp::GetCcuVersion(transDep.userDevId, ccuVersion);
        if (ret != HcclResult::HCCL_SUCCESS || ccuVersion == CcuVersion::CCU_INVALID) {
            Hccl::THROW<Hccl::CcuApiException>("[CcuRepTranslator] Constructor: Invalid CCU Type!");
        }

        // 获取本die的xn起始地址
        ret = CcuDevMgrImp::GetXnBaseAddr(userDevId, dieId, transDep.xnBaseAddr[dieId]);
        if (ret != HcclResult::HCCL_SUCCESS) {
            Hccl::THROW<Hccl::CcuApiException>(
                "Failed to get xn base address. userDevId = %d, dieId = %u, ret = %d", userDevId, dieId, ret);
        }

        // 若另一die使能，在A6场景下获得其xn起始地址
        uint8_t anotherDieId = dieId == 0 ? 1 : 0;
        bool anotherDieEnable = false;
        if (static_cast<HcclResult>(CcuGetDieEnableInfo(userDevId, anotherDieId, anotherDieEnable))
            != HcclResult::HCCL_SUCCESS) {
            Hccl::THROW<Hccl::CcuApiException>(
                "Failed to get CcuGetDieEnableInfo. userDevId = %d, dieId = %u", userDevId, dieId);
        }
        if (ccuVersion == CcuVersion::CCU_V2 && anotherDieEnable == true) {
            ret = CcuDevMgrImp::GetXnBaseAddr(userDevId, anotherDieId, transDep.xnBaseAddr[anotherDieId]);
            if (ret != HcclResult::HCCL_SUCCESS) {
                Hccl::THROW<Hccl::CcuApiException>(
                    "Failed to get xn base address. userDevId = %d, dieId = %u, "
                    "ret = %d",
                    userDevId, anotherDieId, ret);
            }
        }

        transDep.ccuResSpaceTokenInfo = CcuRep::GetToken(ccuTokenInfo.first, ccuTokenInfo.second, 1);
        // 获取hbm token信息
        transDep.memTokenInfo = hbmTokenInfo;
    }

    CcuRepTranslator::CcuRepTranslator(std::shared_ptr<CcuRepReferenceManager> refManager, const TransDep& transDep)
        : refManager(refManager),
          transDep(transDep)
    {
        HcclResult ret = CcuDevMgrImp::GetCcuVersion(transDep.userDevId, ccuVersion);
        if (ret != HcclResult::HCCL_SUCCESS || ccuVersion == CcuVersion::CCU_INVALID) {
            Hccl::THROW<Hccl::CcuApiException>("[CcuRepTranslator] Constructor: Invalid CCU Type!");
        }
    }

    uint32_t CcuRepTranslator::GetInstrNum(const int32_t userDevId)
    {
        CcuVersion tempCcuVersion = CcuVersion::CCU_INVALID;
        HcclResult ret = CcuDevMgrImp::GetCcuVersion(userDevId, tempCcuVersion);
        if (ret != HcclResult::HCCL_SUCCESS || tempCcuVersion == CcuVersion::CCU_INVALID) {
            Hccl::THROW<Hccl::CcuApiException>("[CcuRepTranslator] GetInstrNum: Invalid CCU Type!");
        }

        return tempCcuVersion == CcuVersion::CCU_V1 ?
                   4 // 4:翻译器翻译过程中额外需要的指令空间大小(插入3条通用操作指令+1条终止指令)
                   :
                   13; // 13:翻译器翻译过程中额外需要的指令空间大小(插入3条通用操作指令+1条终止指令+9条repJump)
    }

    CcuResult CcuRepTranslator::GetResReq(const int32_t userDevId, uint8_t dieId, CcuResReq& resReq)
    {
        // 申请离散 xn 资源
        // 需要申请若干xn、gsa、cke设置为固定值用于通用操作
        CcuVersion tempCcuVersion = CcuVersion::CCU_INVALID;
        HcclResult ret = CcuDevMgrImp::GetCcuVersion(userDevId, tempCcuVersion);
        if (ret != HcclResult::HCCL_SUCCESS || tempCcuVersion == CcuVersion::CCU_INVALID) {
            HCCL_ERROR("[CcuRepTranslator] GetResReq: Invalid CCU Type!");
            return CCU_E_INTERNAL;
        }

        int varNum = XN_NUM;
        int gsaNum = tempCcuVersion == CcuVersion::CCU_V1 ? GSA_NUM : 0;
        resReq.xnReq[dieId] = varNum;
        resReq.gsaReq[dieId] = gsaNum;
        resReq.ckeReq[dieId] = CKE_NUM;
        return CCU_SUCCESS;
    }

    void CcuRepTranslator::GetRes(CcuRepResource& res)
    {
        int varNum = XN_NUM;
        int gsaNum = ccuVersion == CcuVersion::CCU_V1 ? GSA_NUM : 0;
        for (int i = 0; i < varNum; i++) {
            res.variable[transDep.dieId].push_back(var[i]);
        }
        for (int i = 0; i < gsaNum; i++) {
            res.address[transDep.dieId].push_back(addr[i]);
        }
        for (int i = 0; i < CKE_NUM; i++) {
            res.completedEvent[transDep.dieId].push_back(signal[i]);
        }
    }

    CcuResult CcuRepTranslator::PreProcess(std::shared_ptr<CcuRepBase> rep)
    {
        auto repType = rep->Type();
        if (repType == CcuRepType::FUNC_BLOCK) {
            auto funcBlock = std::static_pointer_cast<CcuRepFuncBlock>(rep);
            CCU_CHK_RET(refManager->SetRefBlock(funcBlock->GetLabel(), funcBlock));
            funcBlock->SetFuncManager(refManager.get());
        } else if (repType == CcuRepType::LOOP_BLOCK) {
            auto loopBlock = std::static_pointer_cast<CcuRepLoopBlock>(rep);
            CCU_CHK_RET(refManager->SetRefBlock(loopBlock->GetLabel(), loopBlock));
        } else if (repType == CcuRepType::FUNC_CALL) {
            CCU_CHK_RET((BuildReference<CcuRepFuncCall, CcuRepFuncBlock>(rep)));
            auto funcCall = std::static_pointer_cast<CcuRepFuncCall>(rep);
            funcCall->SetFuncManager(refManager.get());
        } else if (repType == CcuRepType::ADDRESSABLE_FUNC_BLOCK) {
            // Addr 链路的调用点按 pc 跳转、不按 label 解析，故无需 SetRefBlock；funcManager 仅用于翻译 func body
            auto funcBlock = std::static_pointer_cast<CcuRepAddressableFuncBlock>(rep);
            funcBlock->SetFuncManager(refManager.get());
        } else if (repType == CcuRepType::LOOP_CALL) {
            CCU_CHK_RET((BuildReference<CcuRepLoopCall, CcuRepLoopBlock>(rep)));
        } else if (repType == CcuRepType::LOOP) {
            CCU_CHK_RET((BuildReference<CcuRepLoop, CcuRepLoopBlock>(rep)));
        }
        return CCU_SUCCESS;
    }

    CcuResult CcuRepTranslator::Translate(
        CcuKernel* ccuKernel, const std::vector<std::shared_ptr<CcuRepBase>>& repVec, CcuInstr*& instr,
        uint16_t& instrId, std::function<bool(std::shared_ptr<CcuRepBase>)> filter)
    {
        constexpr uint32_t maxTryCount = 10; // 最大尝试次数10
        uint32_t tryCount = 0;
        uint32_t restCount = 0;

        auto funcInVar = refManager.get()->GetFuncIn();
        int funcArgIndex = 0;

        do {
            restCount = 0;
            for (uint32_t index = 0; index < repVec.size(); index++) {
                if (!filter(repVec[index])) {
                    continue;
                }

                if (repVec[index]->Translated()) {
                    continue;
                }

                if (repVec[index]->Type() == CcuRepType::LOAD_ARG && transDep.isFuncBlock) {
                    transDep.loadXnId = funcInVar[funcArgIndex++].Id();
                }

                CCU_CHK_RET(PreProcess(repVec[index]));
                bool flag = false;
                CCU_TRY_CATCH_RETURN(flag = repVec[index]->Translate(ccuKernel, instr, instrId, transDep););
                if (!flag) {
                    restCount++;
                }
            }
            tryCount++;
            HCCL_INFO("tryCount = %u, remaining representation = %u", tryCount, restCount);
        } while (restCount > 0 && tryCount < maxTryCount);

        if (tryCount == maxTryCount && restCount > 0) {
            HCCL_ERROR(
                "After translation, remaining representation: tryCount = %u, restCount = %u ", tryCount, restCount);
            for (uint32_t index = 0; index < repVec.size(); index++) {
                if (!repVec[index]->Translated()) {
                    HCCL_ERROR("index[%u], %s", index, repVec[index]->Describe().c_str());
                }
            }
            HCCL_ERROR("Translation Failed");
            return CCU_E_INTERNAL;
        }
        return CCU_SUCCESS;
    }

    namespace {

        // 收集 LOAD_ARG rep 并按全局 argId 升序排序, 确保 LoadSqeArgs 指令在 mission 切分时
        // 落入与其 slot id 匹配的 mission (避免用户乱序 LoadArg 导致取参错位).
        std::vector<std::shared_ptr<CcuRepBase>>
        CollectSortedLoadArgReps(const std::vector<std::shared_ptr<CcuRepBase>>& repVec)
        {
            std::vector<std::shared_ptr<CcuRepBase>> sortedLoadArgReps;
            sortedLoadArgReps.reserve(repVec.size());
            for (const auto& rep : repVec) {
                if (rep->Type() == CcuRepType::LOAD_ARG) {
                    sortedLoadArgReps.push_back(rep);
                }
            }
            std::stable_sort(
                sortedLoadArgReps.begin(), sortedLoadArgReps.end(),
                [](const std::shared_ptr<CcuRepBase>& a, const std::shared_ptr<CcuRepBase>& b) {
                    return std::static_pointer_cast<CcuRepLoadArg>(a)->GetFullArgId()
                           < std::static_pointer_cast<CcuRepLoadArg>(b)->GetFullArgId();
                });
            return sortedLoadArgReps;
        }

        // 可寻址 func 的入口下标发布到微码之外, FixReferences 修不到, 在此按 originIndex 反转回填.
        void RemapAddressableFuncPcInBlock(
            const std::vector<std::shared_ptr<CcuRepBase>>& reps, const std::vector<int32_t>& origToOut,
            uint16_t startInstrId)
        {
            for (const auto& rep : reps) {
                if (rep == nullptr) {
                    continue;
                }
                if (rep->Type() == CcuRepType::ADDRESSABLE_FUNC_BLOCK) {
                    auto funcBlock = std::static_pointer_cast<CcuRepAddressableFuncBlock>(rep);
                    funcBlock->RemapStartInstrId(origToOut, startInstrId);
                }
                auto block = std::dynamic_pointer_cast<CcuRepBlock>(rep);
                if (block != nullptr) {
                    RemapAddressableFuncPcInBlock(block->GetReps(), origToOut, startInstrId);
                }
            }
        }

        [[maybe_unused]] void RemapAddressableFuncPc(
            const std::vector<std::shared_ptr<CcuRepBase>>& repVec, const std::vector<int32_t>& originIndex,
            size_t inputInstrCount, uint16_t startInstrId)
        {
            std::vector<int32_t> origToOut(inputInstrCount, -1);
            for (size_t outIdx = 0; outIdx < originIndex.size(); ++outIdx) {
                const int32_t inIdx = originIndex[outIdx];
                if (inIdx >= 0 && static_cast<size_t>(inIdx) < origToOut.size()) {
                    origToOut[inIdx] = static_cast<int32_t>(outIdx);
                }
            }
            // 递归展开：当前 FuncBlockBeginNamed 只允许 top level 声明命名 func，顶层遍历已完备；
            // 但若将来放开声明位置，嵌套的 addressable FuncBlock 漏回填会静默发布过期 pc，
            // 与 PrepareConstValueResources 的递归展开保持一致，避免该隐患复发。
            RemapAddressableFuncPcInBlock(repVec, origToOut, startInstrId);
        }

        // V2 后端优化 (CkeOnly 档: 只按 CKE 写后读补 NOP, 不重排 / 不重命名寄存器).
        // 定义 CCU_MICROCODE_OPT_DISABLE 时本函数不被引用, 故标注避免告警.
        [[maybe_unused]] CcuInstrInfo RunV2BackendOptimizer(
            const CcuInstrInfo& instrInfo, const CcuKernel* ccuKernel, const TransDep& transDep,
            const std::vector<std::shared_ptr<CcuRepBase>>& repVec)
        {
            HCCL_INFO("[CcuMicrocodeOpt] running backend optimizer (V2)");
            // 把 kernel 登记的 pinned 组 (array = 连续 XN 寄存器) 转成优化器侧轻量结构, 供 XN 写后读
            // (LoadX/StoreX/ClearX) 把散落 xn 操作数归约到所属 array 组. kernel 为空时退化为不归约.
            std::vector<CcuOpt::PinnedGroup> pinnedGroups;
            if (ccuKernel != nullptr) {
                const auto& groups = ccuKernel->GetPinnedRegGroups();
                pinnedGroups.reserve(groups.size());
                for (const auto& g : groups) {
                    pinnedGroups.push_back(CcuOpt::PinnedGroup{g.baseVar.Id(), g.count});
                }
            }
            std::vector<int32_t> originIndex;
            CcuInstrInfo optimized = CcuOpt::MicrocodeOptimizer::Run(
                instrInfo, transDep.reserveXnId, transDep.reserveCkeId, pinnedGroups, &originIndex);
            HCCL_INFO(
                "[CcuMicrocodeOpt] before instrCount=%u, after instrCount=%u", instrInfo.instrCount,
                optimized.instrCount);
            RemapAddressableFuncPc(repVec, originIndex, instrInfo.instrVec.size(), instrInfo.startInstrId);
            return optimized;
        }

    } // namespace

    CcuInstrInfo CcuRepTranslator::Translate(
        CcuKernel* ccuKernel, const std::vector<std::shared_ptr<CcuRepBase>>& repVec, uint16_t startInstrId,
        bool isFuncBlock)
    {
        constexpr uint32_t defaultInstrCapacity = 32 * 1024; // 默认最大容量32 * 1024条
        CcuInstrInfo instrInfo;
        instrInfo.instrVec.resize(defaultInstrCapacity);
        CcuInstr* instr = instrInfo.instrVec.data();
        uint16_t curInstrId = startInstrId;

        BindResource(isFuncBlock);

        // 翻译LoopBlock
        CCU_CHK_RET_THROW(
            Hccl::CcuApiException, "Translate LoopBlock failed",
            Translate(ccuKernel, repVec, instr, curInstrId, [](std::shared_ptr<CcuRepBase> rep) -> bool {
                return rep->Type() == CcuRepType::LOOP_BLOCK;
            }));

        // 翻译funcBlock（含新 callfunc 链路的可寻址 func 定义块，与 FUNC_BLOCK 同在 mission 体前落位）
        CCU_CHK_RET_THROW(
            Hccl::CcuApiException, "Translate FuncBlock failed",
            Translate(ccuKernel, repVec, instr, curInstrId, [](std::shared_ptr<CcuRepBase> rep) -> bool {
                return rep->Type() == CcuRepType::FUNC_BLOCK || rep->Type() == CcuRepType::ADDRESSABLE_FUNC_BLOCK;
            }));

        uint16_t missionStartInstrId = curInstrId;

        // 翻译Load (按全局 argId 升序).
        std::vector<std::shared_ptr<CcuRepBase>> sortedLoadArgReps = CollectSortedLoadArgReps(repVec);
        CCU_CHK_RET_THROW(
            Hccl::CcuApiException, "Translate LoadArg failed",
            Translate(ccuKernel, sortedLoadArgReps, instr, curInstrId, [](std::shared_ptr<CcuRepBase> rep) -> bool {
                return rep->Type() == CcuRepType::LOAD_ARG;
            }));

        // 插入通用操作
        CCU_CHK_RET_THROW(Hccl::CcuApiException, "CommonProcess failed", CommonProcess(ccuKernel, instr, curInstrId));

        // 翻译主体
        CCU_CHK_RET_THROW(
            Hccl::CcuApiException, "Translate main body failed",
            Translate(
                ccuKernel, repVec, instr, curInstrId, []([[maybe_unused]] std::shared_ptr<CcuRepBase> rep) -> bool {
                    return true;
                }));

        CCU_CHK_RET_THROW(Hccl::CcuApiException, "FinishMainBlock failed", FinishMainBlock(instr, curInstrId));

        instrInfo.startInstrId = startInstrId;
        instrInfo.instrCount = curInstrId - startInstrId;
        instrInfo.missionStartInstrId = missionStartInstrId;
        instrInfo.missionInstrCount = curInstrId - missionStartInstrId;
        instrInfo.instrVec.resize(instrInfo.instrCount);

        DumpRep(repVec, instrInfo);
        DumpInstruction(instrInfo, "CcuOriginInstrInfo");

        if (ccuVersion == CcuVersion::CCU_V2) {
#if !defined(CCU_MICROCODE_OPT_DISABLE)
            // 按 CKE 写后读补 NOP; 定义 CCU_MICROCODE_OPT_DISABLE 可关闭.
            CcuInstrInfo optimizedInstrInfo = RunV2BackendOptimizer(instrInfo, ccuKernel, transDep, repVec);
            HCCL_INFO("[CcuMicrocodeOpt] dump instruction after backend optimizer:");
            DumpInstruction(optimizedInstrInfo);
            return optimizedInstrInfo;
#else
            HCCL_INFO("[CcuMicrocodeOpt] backend optimizer disabled; return raw instrInfo");
#endif
        }

        return instrInfo;
    }

    CcuResult CcuRepTranslator::CommonProcess(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId)
    {
        if (ccuVersion == CcuVersion::CCU_INVALID) {
            HCCL_ERROR("[%s] Unexpected ccuVersion[Invalid]", __func__);
            return CCU_E_INTERNAL;
        }
        if (ccuVersion == CcuVersion::CCU_V1) {
            LoadImdToXnInstr(instr++, var[0].Id(), 0);
            LoadImdToGSAInstr(instr++, addr[0].Id(), 0);
            SetCKEInstr(instr++, signal[0].Id(), 0xffff, 0, 0, 1);
        } else {
            CcuV2::AssignI(instr++, var[0].Id(), 0);
            CcuV2::AssignI(instr++, var[XN_NUM - 1].Id(), 0);
            SetCKEInstr(instr++, signal[0].Id(), 0xffff, 0, 0, 1);
        }

        // 遍历需要赋值的常量，A5场景下暂为空表
        std::unordered_map<uint64_t, CcuRep::Variable>& constValue2VarMap = ccuKernel->GetConstValue2VarMap();
        u32 constValueNum = constValue2VarMap.size();
        for (auto elem : constValue2VarMap) {
            uint64_t constValue = elem.first;
            CcuRep::Variable curVariable = elem.second;
            if (ccuVersion == CcuVersion::CCU_V1) {
                LoadImdToXnInstr(instr++, curVariable.Id(), constValue);
            } else {
                CcuV2::AssignI(instr++, curVariable.Id(), constValue);
            }
        }

        u32 instrNum = 3 + constValueNum;
        if (instrId > UINT16_MAX - instrNum) {
            HCCL_ERROR("[%s] integer overflow occurs", __func__);
            return CCU_E_INTERNAL;
        }
        instrId += instrNum; // 插入3条指令
        return CCU_SUCCESS;
    }

    CcuResult CcuRepTranslator::FinishMainBlock(CcuInstr*& instr, uint16_t& instrId)
    {
        if (ccuVersion == CcuVersion::CCU_V1) {
            if (transDep.isFuncBlock) {
                const Variable* funcRet = nullptr;
                CCU_TRY_CATCH_RETURN(funcRet = &(refManager.get()->GetFuncRet(FUNC_NEST_MAX)););
                JumpInstr(instr++, funcRet->Id(), transDep.reserveXnId, 1);
            } else {
                LoadImdToXnInstr(instr++, var[0].Id(), 0);
            }
            instrId++;
        } else {
            if (transDep.isFuncBlock) {
                const Variable* funcRet = nullptr;
                CCU_TRY_CATCH_RETURN(funcRet = &(refManager.get()->GetFuncRet(FUNC_NEST_MAX)););
                CcuV2::RelJmp(
                    instr, funcRet->Id(), instrId + V2_RELJMP_INSTR_NUM, transDep.commXn[0], transDep.commXn[1]);
                instr += V2_RELJMP_INSTR_NUM;
                CcuV2::Jump(instr++, funcRet->Id(), transDep.reserveXnId, transDep.reserveXnId, 0);
            } else {
                for (int i = 0; i < V2_FINISH_BLOCK_INSTR_NUM; i++) {
                    HCCL_INFO("FinishMainBlock Add Nop: index[%d], instrId[%d]", i, instrId);
                    CcuV2::Nop(instr++);
                }
            }
            instrId += V2_FINISH_BLOCK_INSTR_NUM;
        }

        if (instrId > UINT16_MAX - 1) {
            HCCL_ERROR("[%s] integer overflow occurs", __func__);
            return CCU_E_INTERNAL;
        }
        return CCU_SUCCESS;
    }

    void CcuRepTranslator::DumpInstruction(const CcuInstrInfo& instrInfo, const std::string& tag) const
    {
        HCCL_INFO(
            "%s: startInstrId = %u, instrCount = %u, missionStartInstrId = %u, missionInstrCount = %u", tag.c_str(),
            instrInfo.startInstrId, instrInfo.instrCount, instrInfo.missionStartInstrId, instrInfo.missionInstrCount);
        for (uint16_t index = 0; index < instrInfo.instrVec.size(); index++) {
            if (ccuVersion == CcuVersion::CCU_V1) {
                HCCL_INFO(
                    "%d: %s", instrInfo.startInstrId + index, ParseInstr(instrInfo.instrVec.data() + index).c_str());
            } else {
                HCCL_INFO(
                    "%d: %s", instrInfo.startInstrId + index,
                    CcuV2::ParseInstrV2(instrInfo.instrVec.data() + index).c_str());
            }
        }
    }

    void CcuRepTranslator::DumpRep(
        const std::vector<std::shared_ptr<CcuRepBase>>& repVec, const CcuInstrInfo& instrInfo) const
    {
        PLF_CONFIG_INFO(PLF_TASK, "Translated Ccu Rep:");
        for (uint32_t index = 0; index < repVec.size(); index++) {
            uint16_t startInstrId = repVec[index]->StartInstrId();
            uint32_t sum = static_cast<uint32_t>(startInstrId) + repVec[index]->InstrCount();
            if (sum > UINT16_MAX) {
                HCCL_ERROR(
                    "instrId overflow: startInstrId[%u] + InstrCount[%u] = %u exceeds UINT16_MAX", startInstrId,
                    repVec[index]->InstrCount(), sum);
                continue;
            }
            uint16_t endInstrId = static_cast<uint16_t>(sum);
            PLF_CONFIG_INFO(
                PLF_TASK, "rep[%u]: %s Instr[%u--%u]", index, repVec[index]->Describe().c_str(), startInstrId,
                endInstrId);
            for (uint16_t instrId = startInstrId; instrId < endInstrId; instrId++) {
                if (instrId < instrInfo.startInstrId) {
                    HCCL_ERROR("instrId[%u] less than startInstrId[%u]", instrId, instrInfo.startInstrId);
                    continue;
                }
                if (ccuVersion == CcuVersion::CCU_V1) {
                    PLF_CONFIG_INFO(
                        PLF_TASK, "microcode[%u]: %s", instrId,
                        ParseInstr(instrInfo.instrVec.data() + (instrId - instrInfo.startInstrId)).c_str());
                } else {
                    PLF_CONFIG_INFO(
                        PLF_TASK, "microcode[%u]: %s", instrId,
                        CcuV2::ParseInstrV2(instrInfo.instrVec.data() + (instrId - instrInfo.startInstrId)).c_str());
                }
            }
        }
    }

    void CcuRepTranslator::BindResource(bool isFuncBlock)
    {
        // 在V2场景下不会为addr数组申请资源，故reserveGsaId和commGsa中实际填的都是默认值0
        transDep.reserveXnId = var[0].Id();
        transDep.reserveGsaId = addr[0].Id();
        transDep.reserveCkeId = signal[0].Id();
        // commXn 长度与 XN_NUM 联动：var[0] 是 reserveXn，其余 XN_NUM-1 个依次填入 commXn，
        // 故 commXn 的最后一个即 var[XN_NUM-1]（CcuInsGeneratorV2 的 COMM_XN_LAST_INDEX 依赖该等式）。
        static_assert(
            XN_NUM - 1 == static_cast<int>(sizeof(TransDep::commXn) / sizeof(TransDep::commXn[0])),
            "TransDep::commXn length must stay in sync with XN_NUM - 1");
        for (int i = 0; i < XN_NUM - 1; i++) {
            transDep.commXn[i] = var[i + 1].Id();
        }
        for (int i = 0; i < GSA_NUM - 1; i++) {
            transDep.commGsa[i] = addr[i + 1].Id();
        }
        transDep.commSignal = signal[1].Id();
        transDep.isFuncBlock = isFuncBlock;
        HCCL_INFO(
            "TransDep info: userDevId = %d, dieId = %u, reserveXnId = %u, reserveGsaId = %u, reserveCkeId = %u, "
            "innerDieChannelId = %u, interDieChannelId = %u",
            transDep.userDevId, transDep.dieId, transDep.reserveXnId, transDep.reserveGsaId, transDep.reserveCkeId,
            transDep.reserveChannalId[0], transDep.reserveChannalId[1]);
    }
}; // namespace CcuRep
}; // namespace hcomm
