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
#include <climits>

#include "ccu_ins_generator_v1.h"
#include "ccu_rep_base_v1.h"
#include "ccu_rep_block_v1.h"
#include "ccu_rep_nop_v1.h"
#include "ccu_rep_arg_v1.h"
#include "ccu_rep_loopblock_v1.h"
#include "ccu_rep_loopgroup_bundle_v1.h"
#include "ccu_datatype_v1.h"
#include "ccu_microcode_v1.h"
#include "ccu_assist_v1.h"
#include "ccu_api_exception.h"
#include "null_ptr_exception.h"
#include "internal_exception.h"
#include "ccu_types.h"

namespace hcomm {
namespace CcuRep {
    namespace {

        class CcuRepLoopBlockTest : public ::testing::Test {
        protected:
            CcuInsGeneratorV1 insGen{};
            void SetUp() override {}
        };

        TEST_F(CcuRepLoopBlockTest, Constructor)
        {
            CcuRepLoopBlock loopBlock(&insGen, "test_block");

            EXPECT_EQ(loopBlock.GetLabel(), "test_block");
            EXPECT_EQ(loopBlock.Type(), CcuRepType::LOOP_BLOCK);
        }

        TEST_F(CcuRepLoopBlockTest, Describe)
        {
            CcuRepLoopBlock loopBlock(&insGen, "test_block");

            std::string desc = loopBlock.Describe();
            EXPECT_NE(desc.find("test_block"), std::string::npos);
        }

        TEST_F(CcuRepLoopBlockTest, DefineArg_Variable)
        {
            CcuRepLoopBlock loopBlock(&insGen, "test_block");
            Variable var(nullptr);

            loopBlock.DefineArg(var);

            CcuRepArg& arg = loopBlock.GetArg(0);
            EXPECT_EQ(arg.type, CcuArgType::VARIABLE);
        }

        TEST_F(CcuRepLoopBlockTest, DefineArg_Memory)
        {
            CcuRepLoopBlock loopBlock(&insGen, "test_block");
            Address addr(nullptr);
            Variable token(nullptr);
            Memory mem(addr, token);

            loopBlock.DefineArg(mem);

            CcuRepArg& arg = loopBlock.GetArg(0);
            EXPECT_EQ(arg.type, CcuArgType::MEMORY);
        }

        TEST_F(CcuRepLoopBlockTest, DefineArg_LocalAddr)
        {
            CcuRepLoopBlock loopBlock(&insGen, "test_block");
            Address addr(nullptr);
            Variable token(nullptr);
            LocalAddr localAddr(addr, token);

            loopBlock.DefineArg(localAddr);

            CcuRepArg& arg = loopBlock.GetArg(0);
            EXPECT_EQ(arg.type, CcuArgType::LOCAL_ADDR);
        }

        TEST_F(CcuRepLoopBlockTest, DefineArg_RemoteAddr)
        {
            CcuRepLoopBlock loopBlock(&insGen, "test_block");
            Address addr(nullptr);
            Variable token(nullptr);
            RemoteAddr remoteAddr(addr, token);

            loopBlock.DefineArg(remoteAddr);

            CcuRepArg& arg = loopBlock.GetArg(0);
            EXPECT_EQ(arg.type, CcuArgType::REMOTE_ADDR);
        }

        TEST_F(CcuRepLoopBlockTest, DefineArg_VariableList)
        {
            CcuRepLoopBlock loopBlock(&insGen, "test_block");
            std::vector<Variable> varList = {Variable(nullptr), Variable(nullptr)};

            loopBlock.DefineArg(varList);

            CcuRepArg& arg = loopBlock.GetArg(0);
            EXPECT_EQ(arg.type, CcuArgType::VARIABLE_LIST);
            EXPECT_EQ(arg.varList.size(), 2U);
        }

        TEST_F(CcuRepLoopBlockTest, DefineArg_MemoryList)
        {
            CcuRepLoopBlock loopBlock(&insGen, "test_block");
            Address addr(nullptr);
            Variable token(nullptr);
            std::vector<Memory> memList = {Memory(addr, token), Memory(addr, token)};

            loopBlock.DefineArg(memList);

            CcuRepArg& arg = loopBlock.GetArg(0);
            EXPECT_EQ(arg.type, CcuArgType::MEMORY_LIST);
            EXPECT_EQ(arg.memList.size(), 2U);
        }

        TEST_F(CcuRepLoopBlockTest, DefineArg_LocalAddrList)
        {
            CcuRepLoopBlock loopBlock(&insGen, "test_block");
            Address addr(nullptr);
            Variable token(nullptr);
            std::vector<LocalAddr> addrList = {LocalAddr(addr, token), LocalAddr(addr, token)};

            loopBlock.DefineArg(addrList);

            CcuRepArg& arg = loopBlock.GetArg(0);
            EXPECT_EQ(arg.type, CcuArgType::LOCAL_ADDR_LIST);
            EXPECT_EQ(arg.localAddrList.size(), 2U);
        }

        TEST_F(CcuRepLoopBlockTest, DefineArg_RemoteAddrList)
        {
            CcuRepLoopBlock loopBlock(&insGen, "test_block");
            Address addr(nullptr);
            Variable token(nullptr);
            std::vector<RemoteAddr> addrList = {RemoteAddr(addr, token), RemoteAddr(addr, token)};

            loopBlock.DefineArg(addrList);

            CcuRepArg& arg = loopBlock.GetArg(0);
            EXPECT_EQ(arg.type, CcuArgType::REMOTE_ADDR_LIST);
            EXPECT_EQ(arg.remoteAddrList.size(), 2U);
        }

        TEST_F(CcuRepLoopBlockTest, GetArg_OutOfRange)
        {
            CcuRepLoopBlock loopBlock(&insGen, "test_block");

            EXPECT_THROW(loopBlock.GetArg(0), Hccl::CcuApiException);
        }

        class CcuRepLoopGroupBundleTest : public ::testing::Test {
        protected:
            CcuInsGeneratorV1 insGen{};
            void SetUp() override {}

            static CcuRepLoopGroupBundle::LoopEntry MakeTranslatedLoopEntry()
            {
                CcuInsGeneratorV1 insGen;
                CcuRepLoopGroupBundle::LoopEntry entry;
                entry.config = CcuLoopCfg{};
                entry.executor = Executor(nullptr);
                entry.repLoopBlock = std::make_shared<CcuRepLoopBlock>(&insGen, "loop_block");
                entry.repLoopBlock->Append(std::make_shared<CcuRepNop>(&insGen));
                entry.loopParamVar = Variable(nullptr);
                entry.layout = CcuRepLoopGroupBundle::Layout::Config;

                CcuInstr blockInstr[1]{};
                CcuInstr* blockPtr = blockInstr;
                uint16_t blockInstrId = 0;
                TransDep blockDep{0, 0, 1, 1, 0, {0, 0}, {0}, 0, 0, {0, 0, 0}, {0, 0}, 0, false};
                entry.repLoopBlock->Translate(nullptr, blockPtr, blockInstrId, blockDep);
                return entry;
            }
        };

        TEST_F(CcuRepLoopGroupBundleTest, Constructor)
        {
            CcuLoopGroupCfg grpCfg{};
            Variable parallelParam(nullptr);
            Variable offsetParam(nullptr);
            CcuRepLoopGroupBundle bundle(&insGen, grpCfg, parallelParam, offsetParam);

            EXPECT_EQ(bundle.Type(), CcuRepType::LOOPGROUP);
        }

        TEST_F(CcuRepLoopGroupBundleTest, GetOffsetParam)
        {
            CcuLoopGroupCfg grpCfg{};
            Variable parallelParam(nullptr);
            Variable offsetParam(nullptr);
            CcuRepLoopGroupBundle bundle(&insGen, grpCfg, parallelParam, offsetParam);

            EXPECT_EQ(bundle.GetOffsetParam().Id(), offsetParam.Id());
        }

        TEST_F(CcuRepLoopGroupBundleTest, Translate)
        {
            CcuLoopGroupCfg grpCfg{};
            Variable parallelParam(nullptr);
            Variable offsetParam(nullptr);
            CcuRepLoopGroupBundle bundle(&insGen, grpCfg, parallelParam, offsetParam);
            bundle.AddLoop(MakeTranslatedLoopEntry());

            constexpr uint16_t kExpectedInstrCount = 8;
            CcuInstr instr[kExpectedInstrCount] = {};
            CcuInstr* instrPtr = instr;
            uint16_t instrId = 0;
            TransDep dep{0, 0, 1, 1, 0, {0, 0}, {0}, 0, 0, {0, 0, 0}, {0, 0}, 0, false};

            bool result = bundle.Translate(nullptr, instrPtr, instrId, dep);

            EXPECT_TRUE(result);
            EXPECT_EQ(instrId, kExpectedInstrCount);
        }

        TEST_F(CcuRepLoopGroupBundleTest, GetStartLoopInstrId)
        {
            CcuLoopGroupCfg grpCfg{};
            Variable parallelParam(nullptr);
            Variable offsetParam(nullptr);
            CcuRepLoopGroupBundle bundle(&insGen, grpCfg, parallelParam, offsetParam);
            bundle.AddLoop(MakeTranslatedLoopEntry());

            constexpr uint16_t kBundleInstrCount = 8;
            CcuInstr instr[kBundleInstrCount] = {};
            CcuInstr* instrPtr = instr;
            uint16_t instrId = 10;
            TransDep dep{0, 0, 1, 1, 0, {0, 0}, {0}, 0, 0, {0, 0, 0}, {0, 0}, 0, false};

            bundle.Translate(nullptr, instrPtr, instrId, dep);

            EXPECT_EQ(bundle.GetStartLoopInstrId(), 16U);
        }

        TEST_F(CcuRepLoopGroupBundleTest, Describe)
        {
            CcuLoopGroupCfg grpCfg{};
            Variable parallelParam(nullptr);
            Variable offsetParam(nullptr);
            CcuRepLoopGroupBundle bundle(&insGen, grpCfg, parallelParam, offsetParam);

            std::string desc = bundle.Describe();
            EXPECT_NE(desc.find("LoopGroupBundle"), std::string::npos);
        }

    } // namespace
} // namespace CcuRep
} // namespace hcomm
