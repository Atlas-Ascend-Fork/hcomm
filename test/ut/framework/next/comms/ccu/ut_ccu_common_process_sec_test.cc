/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <memory>
#include <string>

#include "ccu_res_specs.h"
#include "src/base_comm/resources/ccu/ccu_kernel/ccu_kernel.h"
#include "src/base_comm/resources/ccu/ccu_representation/reps/translator/ccu_rep_reference_manager_v1.h"
#include "src/base_comm/resources/ccu/ccu_representation/reps/translator/ccu_rep_translator_v1.h"
#include "src/base_comm/resources/ccu/ccu_microcode/ccu_microcode_v1.h"

namespace hcomm {
namespace CcuRep {
    namespace {

        // 覆盖 mission 序言 CommonProcess 的 secFlag 决策(ccu_rep_translator.cc):
        // 经 public 的 Translate 走真实装载路径, 断言敏感常量(token)的装载指令
        // secFlag==CCU_LOAD_TO_XN_SEC_INFO 且 dump 显示 tokenInfo, 非敏感常量 secFlag==0
        // 且 dump 仍打印明文
        class CommonProcessSecFlagTest : public ::testing::Test {
        protected:
            void SetUp() override
            {
                // 直接初始化 CcuResSpecifications 单例为 V2(参照 ut_ccu_res_specs 的 SetupSpecs 模式),
                // CcuRepTranslator 构造经 CcuDevMgrImp::GetCcuVersion 真实路径读到版本, 无需打桩
                auto& specs = CcuResSpecifications::GetInstance(devLogicId_);
                specs.initFlag_ = true;
                specs.ccuVersion_ = CcuVersion::CCU_V2;
                specs.serveMode_ = ServeMode::NORMAL;
                specs.dieEnableFlags_[0] = true;
                specs.dieEnableFlags_[1] = true;

                kernel_.SetDieId(0);
                // 统一登记接口: 普通常量与 token 一次登记, token 同步进入敏感集合
                kernel_.Add2ConstValue2VarMap({tokenValue_, normalValue_}, {tokenValue_});
                EXPECT_TRUE(kernel_.IsSecConstValue(tokenValue_));
                EXPECT_FALSE(kernel_.IsSecConstValue(normalValue_));
            }

            void TearDown() override
            {
                // 复位单例状态, 避免 V2 配置残留影响后续使用该槽位的用例
                auto& specs = CcuResSpecifications::GetInstance(devLogicId_);
                specs.initFlag_ = false;
                specs.ccuVersion_ = CcuVersion::CCU_INVALID;
                specs.serveMode_ = ServeMode::ARMX86;
                specs.dieEnableFlags_.fill(false);
            }

            static constexpr int32_t devLogicId_ = 63; // 独立设备号, 避免单例影响其他用例
            static constexpr uint64_t tokenValue_ = 0xABCD1234;
            static constexpr uint64_t normalValue_ = 0x1000;
            CcuKernel kernel_;
        };

        TEST_F(CommonProcessSecFlagTest, SecConstValueFlagInMissionPrelude)
        {
            auto refManager = std::make_shared<CcuRepReferenceManager>(0);
            TransDep dep{};
            dep.userDevId = devLogicId_;
            CcuRepTranslator translator(refManager, dep);

            // 空 repVec 走完整 mission 序言: BindResource -> CommonProcess(常量表装载) -> FinishMainBlock
            CcuInstrInfo instrInfo = translator.Translate(&kernel_, {}, 0, false);

            uint32_t secFlagCnt = 0;
            uint32_t plainCnt = 0;
            std::string secDump;
            std::string plainDump;
            for (uint32_t i = 0; i < instrInfo.instrCount; i++) {
                const CcuInstr& instr = instrInfo.instrVec[i];
                if (instr.header.type != 0x0 || instr.header.code != 0x2) { // LOAD_TYPE / LOADIMDTOX_CODE
                    continue;
                }
                const uint64_t imm = instr.v2.loadImdToX.immediate;
                if (imm == tokenValue_) {
                    EXPECT_EQ(instr.v2.loadImdToX.secFlag, CCU_LOAD_TO_XN_SEC_INFO);
                    secFlagCnt++;
                    secDump = CcuV2::ParseInstrV2(&instr);
                } else if (imm == normalValue_) {
                    EXPECT_EQ(instr.v2.loadImdToX.secFlag, 0u);
                    plainCnt++;
                    plainDump = CcuV2::ParseInstrV2(&instr);
                }
            }
            EXPECT_EQ(secFlagCnt, 1u);                                 // 仅 token 一条打标
            EXPECT_EQ(plainCnt, 1u);                                   // 普通常量不打标
            EXPECT_NE(secDump.find("tokenInfo"), std::string::npos);   // token 脱敏
            EXPECT_EQ(secDump.find("2882343476"), std::string::npos);  // 明文 token(0xABCD1234) 不得出现
            EXPECT_NE(plainDump.find("4096"), std::string::npos);      // 0x1000 明文
            EXPECT_EQ(plainDump.find("tokenInfo"), std::string::npos); // 非敏感不被误伤
        }

    } // namespace
} // namespace CcuRep
} // namespace hcomm
