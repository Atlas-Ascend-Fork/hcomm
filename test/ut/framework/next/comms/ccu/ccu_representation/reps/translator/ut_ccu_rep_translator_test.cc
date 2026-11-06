/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_ins_generator_v1.h"
#include "ccu_ins_generator_v2.h"
#include "ccu_rep_reference_manager_v1.h"
#include "ccu_rep_type_v1.h"
#include "ccu_api_exception.h"
#include "ccu_rep_load_v1.h"
#include "ccu_rep_load_var_v1.h"
#include "ccu_rep_store_v1.h"
#include "ccu_rep_store_var_v1.h"
#include "src/base_comm/resources/ccu/ccu_kernel/ccu_kernel.h"
#include "src/base_comm/resources/ccu/ccu_microcode/ccu_microcode_v1.h"
#include "src/base_comm/resources/ccu/ccu_representation/reps/sync/ccu_rep_record_shared_notify.h"
#include <gtest/gtest.h>
#include <string>

namespace hcomm {
namespace CcuRep {

    extern uint32_t GetRelativeInstrId(uint32_t currentInstrId, uint32_t targetInstrId);

    namespace {

        class GetRelativeInstrIdTest : public ::testing::Test {};

        TEST_F(GetRelativeInstrIdTest, ForwardOffset)
        {
            EXPECT_EQ(GetRelativeInstrId(10, 20), 10u);
            EXPECT_EQ(GetRelativeInstrId(0, 1), 1u);
            EXPECT_EQ(GetRelativeInstrId(5, 100), 95u);
        }

        TEST_F(GetRelativeInstrIdTest, EqualIds) { EXPECT_EQ(GetRelativeInstrId(50, 50), 0x10000u); }

        TEST_F(GetRelativeInstrIdTest, BackwardOffset)
        {
            EXPECT_EQ(GetRelativeInstrId(20, 10), 0x10000u - 10u);
            EXPECT_EQ(GetRelativeInstrId(100, 5), 0x10000u - 95u);
        }

        TEST_F(GetRelativeInstrIdTest, WrapAround)
        {
            EXPECT_EQ(GetRelativeInstrId(0xFFFF, 0), 1u);
            EXPECT_EQ(GetRelativeInstrId(0xFFFF, 1), 2u);
            EXPECT_EQ(GetRelativeInstrId(1, 0xFFFF), 0x10000u - 2u);
        }

        class CcuInsGeneratorV2GetInstrCountTest : public ::testing::Test {
        protected:
            CcuInsGeneratorV2 insGen;
        };

        TEST_F(CcuInsGeneratorV2GetInstrCountTest, Nop) { EXPECT_EQ(insGen.GetInstrCount(CcuRepType::NOP), 1u); }

        TEST_F(CcuInsGeneratorV2GetInstrCountTest, Add) { EXPECT_GT(insGen.GetInstrCount(CcuRepType::ADD), 0u); }

        TEST_F(CcuInsGeneratorV2GetInstrCountTest, Assign) { EXPECT_GT(insGen.GetInstrCount(CcuRepType::ASSIGN), 0u); }

        TEST_F(CcuInsGeneratorV2GetInstrCountTest, Jump) { EXPECT_GT(insGen.GetInstrCount(CcuRepType::JUMP), 0u); }

        TEST_F(CcuInsGeneratorV2GetInstrCountTest, Load) { EXPECT_GT(insGen.GetInstrCount(CcuRepType::LOAD), 0u); }

        TEST_F(CcuInsGeneratorV2GetInstrCountTest, Shr) { EXPECT_EQ(insGen.GetInstrCount(CcuRepType::SHR), 1u); }

        TEST_F(CcuInsGeneratorV2GetInstrCountTest, Shl) { EXPECT_EQ(insGen.GetInstrCount(CcuRepType::SHL), 1u); }

        // 契约固化: 表值须等于翻译侧两分支实际发射条数(跨Die: AssignI+StoreXToMem+SetCKE; 本Die: SetCKE+2*Nop)
        TEST_F(CcuInsGeneratorV2GetInstrCountTest, RecordSharedNotify)
        {
            EXPECT_EQ(insGen.GetInstrCount(CcuRepType::RECORD_SHARED_NOTIFY), 3u);
        }

        class CcuInsGeneratorV1GetInstrCountTest : public ::testing::Test {
        protected:
            CcuInsGeneratorV1 insGen;
        };

        TEST_F(CcuInsGeneratorV1GetInstrCountTest, Nop) { EXPECT_EQ(insGen.GetInstrCount(CcuRepType::NOP), 1u); }

        TEST_F(CcuInsGeneratorV1GetInstrCountTest, Add) { EXPECT_GT(insGen.GetInstrCount(CcuRepType::ADD), 0u); }

        TEST_F(CcuInsGeneratorV1GetInstrCountTest, Mul_ThrowsException_WhenNotInTable)
        {
            EXPECT_THROW(insGen.GetInstrCount(CcuRepType::MUL), Hccl::CcuApiException);
        }

        TEST_F(CcuInsGeneratorV1GetInstrCountTest, Shr_ThrowsException_WhenNotInTable)
        {
            EXPECT_THROW(insGen.GetInstrCount(CcuRepType::SHR), Hccl::CcuApiException);
        }

        class CcuRepReferenceManagerTest : public ::testing::Test {
        protected:
            std::unique_ptr<CcuRepReferenceManager> mgr;

            void SetUp() override { mgr = std::make_unique<CcuRepReferenceManager>(0); }
        };

        TEST_F(CcuRepReferenceManagerTest, Constructor_Normal) { EXPECT_NE(mgr, nullptr); }

        TEST_F(CcuRepReferenceManagerTest, GetResReq_Normal)
        {
            CcuResReq req = CcuRepReferenceManager::GetResReq(0);
            EXPECT_GT(req.xnReq[0], 0u);
        }

        TEST_F(CcuRepReferenceManagerTest, GetResReq_DifferentDieId)
        {
            CcuResReq req = CcuRepReferenceManager::GetResReq(1);
            EXPECT_GT(req.xnReq[1], 0u);
        }

        TEST_F(CcuRepReferenceManagerTest, GetFuncIn_Normal)
        {
            const auto& funcIn = mgr->GetFuncIn();
            EXPECT_EQ(funcIn.size(), static_cast<size_t>(FUNC_ARG_MAX));
        }

        TEST_F(CcuRepReferenceManagerTest, GetFuncOut_Normal)
        {
            const auto& funcOut = mgr->GetFuncOut();
            EXPECT_EQ(funcOut.size(), static_cast<size_t>(FUNC_ARG_MAX));
        }

        TEST_F(CcuRepReferenceManagerTest, GetFuncCall_Normal)
        {
            const Variable& funcCall = mgr->GetFuncCall();
            (void)funcCall;
        }

        TEST_F(CcuRepReferenceManagerTest, GetFuncRet_Layer0)
        {
            const Variable& ret = mgr->GetFuncRet(0);
            (void)ret;
        }

        TEST_F(CcuRepReferenceManagerTest, GetFuncRet_LayerMax)
        {
            const Variable& ret = mgr->GetFuncRet(FUNC_NEST_MAX);
            (void)ret;
        }

        TEST_F(CcuRepReferenceManagerTest, GetFuncRet_ExceedMax_ThrowsException)
        {
            EXPECT_THROW(mgr->GetFuncRet(FUNC_NEST_MAX + 1), Hccl::CcuApiException);
        }

        TEST_F(CcuRepReferenceManagerTest, GetRefBlock_NotExist_ThrowsException)
        {
            EXPECT_THROW(mgr->GetRefBlock("nonexistent"), Hccl::CcuApiException);
        }

        TEST_F(CcuRepReferenceManagerTest, SetRefBlock_ThenGetRefBlock_Success)
        {
            CcuInsGeneratorV2 insGen;
            auto block = std::make_shared<CcuRepBlock>(&insGen, "testLabel");
            EXPECT_EQ(mgr->SetRefBlock("testLabel", block), CcuResult::CCU_SUCCESS);
            auto result = mgr->GetRefBlock("testLabel");
            EXPECT_NE(result, nullptr);
        }

        TEST_F(CcuRepReferenceManagerTest, SetRefBlock_Duplicate_Expect_E_INTERNAL)
        {
            CcuInsGeneratorV2 insGen;
            auto block1 = std::make_shared<CcuRepBlock>(&insGen, "label1");
            auto block2 = std::make_shared<CcuRepBlock>(&insGen, "label1");
            EXPECT_EQ(mgr->SetRefBlock("label1", block1), CcuResult::CCU_SUCCESS);
            EXPECT_EQ(mgr->SetRefBlock("label1", block2), CcuResult::CCU_E_INTERNAL);
        }

        TEST_F(CcuRepReferenceManagerTest, SetRefBlock_DifferentLabels_Success)
        {
            CcuInsGeneratorV2 insGen;
            auto block1 = std::make_shared<CcuRepBlock>(&insGen, "label1");
            auto block2 = std::make_shared<CcuRepBlock>(&insGen, "label2");
            EXPECT_EQ(mgr->SetRefBlock("label1", block1), CcuResult::CCU_SUCCESS);
            EXPECT_EQ(mgr->SetRefBlock("label2", block2), CcuResult::CCU_SUCCESS);
            EXPECT_NE(mgr->GetRefBlock("label1"), nullptr);
            EXPECT_NE(mgr->GetRefBlock("label2"), nullptr);
        }

        TEST_F(CcuRepReferenceManagerTest, GetFuncAddr_NotExist_ThrowsException)
        {
            EXPECT_THROW(mgr->GetFuncAddr("nonexistent"), Hccl::CcuApiException);
        }

        TEST_F(CcuRepReferenceManagerTest, GetFuncAddr_WrongType_ThrowsException)
        {
            CcuInsGeneratorV2 insGen;
            auto block = std::make_shared<CcuRepBlock>(&insGen, "notFunc");
            EXPECT_EQ(mgr->SetRefBlock("notFunc", block), CcuResult::CCU_SUCCESS);
            EXPECT_THROW(mgr->GetFuncAddr("notFunc"), Hccl::CcuApiException);
        }

        TEST_F(CcuRepReferenceManagerTest, ClearRepReference_AfterClear_GetThrows)
        {
            CcuInsGeneratorV2 insGen;
            auto block = std::make_shared<CcuRepBlock>(&insGen, "label");
            EXPECT_EQ(mgr->SetRefBlock("label", block), CcuResult::CCU_SUCCESS);
            mgr->ClearRepReference();
            EXPECT_THROW(mgr->GetRefBlock("label"), Hccl::CcuApiException);
        }

        TEST_F(CcuRepReferenceManagerTest, ClearRepReference_ThenSetAgain_Success)
        {
            CcuInsGeneratorV2 insGen;
            auto block1 = std::make_shared<CcuRepBlock>(&insGen, "label");
            EXPECT_EQ(mgr->SetRefBlock("label", block1), CcuResult::CCU_SUCCESS);
            mgr->ClearRepReference();
            auto block2 = std::make_shared<CcuRepBlock>(&insGen, "label");
            EXPECT_EQ(mgr->SetRefBlock("label", block2), CcuResult::CCU_SUCCESS);
            EXPECT_NE(mgr->GetRefBlock("label"), nullptr);
        }

        TEST_F(CcuRepReferenceManagerTest, Dump_DoesNotCrash)
        {
            CcuInsGeneratorV2 insGen;
            auto block = std::make_shared<CcuRepBlock>(&insGen, "dumpLabel");
            EXPECT_EQ(mgr->SetRefBlock("dumpLabel", block), CcuResult::CCU_SUCCESS);
            mgr->Dump();
            SUCCEED();
        }

        TEST_F(CcuRepReferenceManagerTest, GetRes_Normal)
        {
            CcuRepResource res;
            mgr->GetRes(res);
            EXPECT_FALSE(res.variable[0].empty());
        }

        // A6 敏感常量(token)脱敏链路集成用例: PrepareConstValue 登记并挑出 token 标记 ->
        // IsSecConstValue 查询 -> GenAddressableFuncPrelude 装载打 secFlag -> ParseInstrV2 dump 脱敏
        class SecConstValueDumpTest : public ::testing::Test {
        protected:
            void SetUp() override
            {
                // dieId 默认 UINT32_MAX, 常量登记的 CreateVariable 按其索引资源账本, 须设合法值
                kernel.SetDieId(0);
            }
            CcuInsGeneratorV2 insGen;
            CcuKernel kernel;
        };

        TEST_F(SecConstValueDumpTest, SecConstValueDumpAsTokenInfo)
        {
            const uint64_t addr = 0x1000;
            const uint64_t tokenValue = 0xABCD1234;
            TransDep dep{};
            dep.memTokenInfo = tokenValue;

            // 登记路径: PrepareLoadConstValue 登记 {addr, memToken, size} 并把 memToken 挑进敏感集合
            CcuRepLoad rep(&insGen, addr, kernel.CreateVariable(), 2);
            EXPECT_EQ(insGen.PrepareConstValue(&rep, dep, &kernel), HcclResult::HCCL_SUCCESS);
            EXPECT_EQ(kernel.GetConstValue2VarMap().size(), 3u);
            EXPECT_TRUE(kernel.IsSecConstValue(tokenValue));
            EXPECT_FALSE(kernel.IsSecConstValue(addr));

            // 装载打标路径: 3 条前缀(AssignI/AssignI/SetCKE)后逐条装载常量表
            CcuInstr instrs[8]{};
            CcuInstr* instr = instrs;
            uint16_t curInstrId = 0;
            insGen.GenAddressableFuncPrelude(&kernel, instr, curInstrId, dep);
            EXPECT_EQ(curInstrId, 6u);

            uint32_t secFlagCnt = 0;
            uint32_t plainCnt = 0;
            std::string secDump;
            std::string addrDump;
            for (uint32_t i = 3; i < curInstrId; i++) {
                EXPECT_EQ(instrs[i].header.type, 0x0); // LOAD_TYPE
                EXPECT_EQ(instrs[i].header.code, 0x2); // LOADIMDTOX_CODE
                if (instrs[i].v2.loadImdToX.immediate == tokenValue) {
                    EXPECT_EQ(instrs[i].v2.loadImdToX.secFlag, CCU_LOAD_TO_XN_SEC_INFO);
                    secFlagCnt++;
                    secDump = CcuV2::ParseInstrV2(&instrs[i]);
                } else {
                    EXPECT_EQ(instrs[i].v2.loadImdToX.secFlag, 0u);
                    plainCnt++;
                    if (instrs[i].v2.loadImdToX.immediate == addr) {
                        addrDump = CcuV2::ParseInstrV2(&instrs[i]);
                    }
                }
            }
            EXPECT_EQ(secFlagCnt, 1u); // 仅 token 一条打标
            EXPECT_EQ(plainCnt, 2u);   // addr + size 不打标

            // dump 断言: 敏感常量显示 tokenInfo 且明文不出现, 非敏感常量仍打印明文
            EXPECT_NE(secDump.find("tokenInfo"), std::string::npos);
            EXPECT_EQ(secDump.find("2882343476"), std::string::npos); // 明文 token(0xABCD1234) 不得出现
            EXPECT_NE(addrDump.find("4096"), std::string::npos);      // 0x1000 明文
            EXPECT_EQ(addrDump.find("tokenInfo"), std::string::npos);
        }

        // Prepare*ConstValue 按 rep 类型登记 token: 可低成本构造的 rep 类型逐个断言,
        // PrepareConstValue 后 token 应进入敏感集合(IsSecConstValue 为 true);
        // StoreVar 的 HSCB 分支不含 token, 敏感集合应为空
        TEST_F(SecConstValueDumpTest, PrepareConstValueMarksTokenForRepTypes)
        {
            const uint64_t tokenValue = 0xABCD1234;
            TransDep dep{};
            dep.memTokenInfo = tokenValue;

            // LOAD: 登记 {addr, token, size}, 敏感 {token}
            {
                CcuKernel kernel;
                kernel.SetDieId(0);
                CcuRepLoad rep(&insGen, 0x1000, kernel.CreateVariable(), 1);
                EXPECT_EQ(insGen.PrepareConstValue(&rep, dep, &kernel), HcclResult::HCCL_SUCCESS);
                EXPECT_TRUE(kernel.IsSecConstValue(tokenValue));
                EXPECT_EQ(kernel.GetConstValue2VarMap().size(), 3u);
            }
            // LOAD_VAR: 登记 {token, size}
            {
                CcuKernel kernel;
                kernel.SetDieId(0);
                CcuRepLoadVar rep(&insGen, kernel.CreateVariable(), kernel.CreateVariable(), 1);
                EXPECT_EQ(insGen.PrepareConstValue(&rep, dep, &kernel), HcclResult::HCCL_SUCCESS);
                EXPECT_TRUE(kernel.IsSecConstValue(tokenValue));
            }
            // STORE: 登记 {addr, token, size}
            {
                CcuKernel kernel;
                kernel.SetDieId(0);
                CcuRepStore rep(&insGen, kernel.CreateVariable(), 0x2000, 1);
                EXPECT_EQ(insGen.PrepareConstValue(&rep, dep, &kernel), HcclResult::HCCL_SUCCESS);
                EXPECT_TRUE(kernel.IsSecConstValue(tokenValue));
            }
            // STORE_VAR(非HSCB): 登记 {size, token}
            {
                CcuKernel kernel;
                kernel.SetDieId(0);
                CcuRepStoreVar rep(&insGen, kernel.CreateVariable(), kernel.CreateVariable(), 1, false);
                EXPECT_EQ(insGen.PrepareConstValue(&rep, dep, &kernel), HcclResult::HCCL_SUCCESS);
                EXPECT_TRUE(kernel.IsSecConstValue(tokenValue));
            }
            // STORE_VAR(HSCB): 登记 {size}, 不含 token
            {
                CcuKernel kernel;
                kernel.SetDieId(0);
                CcuRepStoreVar rep(&insGen, kernel.CreateVariable(), kernel.CreateVariable(), 1, true);
                EXPECT_EQ(insGen.PrepareConstValue(&rep, dep, &kernel), HcclResult::HCCL_SUCCESS);
                EXPECT_FALSE(kernel.IsSecConstValue(tokenValue));
            }
            // RECORD_SHARED_NOTIFY: 登记 {token, maskLen, mask}
            {
                CcuKernel kernel;
                kernel.SetDieId(0);
                LocalNotify notify;
                CcuRepRecordSharedNotify rep(&insGen, notify, 0xF);
                EXPECT_EQ(insGen.PrepareConstValue(&rep, dep, &kernel), HcclResult::HCCL_SUCCESS);
                EXPECT_TRUE(kernel.IsSecConstValue(tokenValue));
            }
        }

    } // namespace
} // namespace CcuRep
} // namespace hcomm
